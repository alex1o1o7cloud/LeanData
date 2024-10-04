import Mathlib

namespace solve_for_y_l183_183619

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l183_183619


namespace lcm_18_20_25_l183_183559

-- Lean 4 statement to prove the smallest positive integer divisible by 18, 20, and 25 is 900
theorem lcm_18_20_25 : Nat.lcm (Nat.lcm 18 20) 25 = 900 :=
by
  sorry

end lcm_18_20_25_l183_183559


namespace sum_of_consecutive_numbers_with_lcm_168_l183_183916

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l183_183916


namespace range_of_a_l183_183573

variable (a : ℝ)

def p := ∀ x : ℝ, x^2 + a ≥ 0
def q := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (h : p a ∧ q a) : 0 ≤ a :=
by
  sorry

end range_of_a_l183_183573


namespace intersection_S_T_eq_T_l183_183091

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183091


namespace intersection_S_T_eq_T_l183_183095

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183095


namespace A_beats_B_by_40_meters_l183_183149

-- Definitions based on conditions
def distance_A := 1000 -- Distance in meters
def time_A := 240      -- Time in seconds
def time_diff := 10      -- Time difference in seconds

-- Intermediate calculations
def velocity_A : ℚ := distance_A / time_A
def time_B := time_A + time_diff
def velocity_B : ℚ := distance_A / time_B

-- Distance B covers in 240 seconds
def distance_B_in_240 : ℚ := velocity_B * time_A

-- Proof goal
theorem A_beats_B_by_40_meters : (distance_A - distance_B_in_240 = 40) :=
by
  -- Insert actual steps to prove here
  sorry

end A_beats_B_by_40_meters_l183_183149


namespace lemons_needed_l183_183655

theorem lemons_needed (lemons_per_48_gallons : ℚ) (limeade_factor : ℚ) (total_gallons : ℚ) (split_gallons : ℚ) :
  lemons_per_48_gallons = 36 / 48 →
  limeade_factor = 2 →
  total_gallons = 18 →
  split_gallons = total_gallons / 2 →
  (split_gallons * (36 / 48) + split_gallons * (2 * (36 / 48))) = 20.25 :=
by
  intros h1 h2 h3 h4
  sorry

end lemons_needed_l183_183655


namespace john_weight_end_l183_183726

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_l183_183726


namespace solve_for_y_l183_183496

theorem solve_for_y (y : ℝ) : 5 * y - 100 = 125 ↔ y = 45 := by
  sorry

end solve_for_y_l183_183496


namespace no_descending_multiple_of_111_l183_183874

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end no_descending_multiple_of_111_l183_183874


namespace anna_current_age_l183_183253

theorem anna_current_age (A : ℕ) (Clara_now : ℕ) (years_ago : ℕ) (Clara_age_ago : ℕ) 
    (H1 : Clara_now = 80) 
    (H2 : years_ago = 41) 
    (H3 : Clara_age_ago = Clara_now - years_ago) 
    (H4 : Clara_age_ago = 3 * (A - years_ago)) : 
    A = 54 :=
by
  sorry

end anna_current_age_l183_183253


namespace man_swim_downstream_distance_l183_183659

-- Define the given conditions
def t_d : ℝ := 6
def t_u : ℝ := 6
def d_u : ℝ := 18
def V_m : ℝ := 4.5

-- The distance the man swam downstream
def distance_downstream : ℝ := 36

-- Prove that given the conditions, the man swam 36 km downstream
theorem man_swim_downstream_distance (V_c : ℝ) :
  (d_u / (V_m - V_c) = t_u) →
  (distance_downstream / (V_m + V_c) = t_d) →
  distance_downstream = 36 :=
by
  sorry

end man_swim_downstream_distance_l183_183659


namespace actual_positions_correct_l183_183718

-- Define the five athletes
inductive Athlete
| A | B | C | D | E
deriving DecidableEq, Repr

open Athlete

-- Define the two predictions as lists
def first_prediction : List Athlete := [A, B, C, D, E]
def second_prediction : List Athlete := [C, E, A, B, D]

-- Define the actual positions
def actual_positions : List Athlete := [C, B, A, D, E]

-- Prove that the first prediction correctly predicted exactly three athletes
def first_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD first_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

-- Prove that the second prediction correctly predicted exactly two athletes
def second_prediction_correct : Nat := List.sum (List.map (λ i => if List.getD second_prediction i Athlete.A == List.getD actual_positions i Athlete.A then 1 else 0) [0, 1, 2, 3, 4])

theorem actual_positions_correct :
  first_prediction_correct = 3 ∧ second_prediction_correct = 2 ∧
  actual_positions = [C, B, A, D, E] :=
by
  -- Placeholder for actual proof
  sorry

end actual_positions_correct_l183_183718


namespace smaller_angle_at_3_20_l183_183942

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l183_183942


namespace dogs_not_eat_either_l183_183301

-- Let's define the conditions
variables (total_dogs : ℕ) (dogs_like_carrots : ℕ) (dogs_like_chicken : ℕ) (dogs_like_both : ℕ)

-- Given conditions
def conditions : Prop :=
  total_dogs = 85 ∧
  dogs_like_carrots = 12 ∧
  dogs_like_chicken = 62 ∧
  dogs_like_both = 8

-- Problem to solve
theorem dogs_not_eat_either (h : conditions total_dogs dogs_like_carrots dogs_like_chicken dogs_like_both) :
  (total_dogs - (dogs_like_carrots - dogs_like_both + dogs_like_chicken - dogs_like_both + dogs_like_both)) = 19 :=
by {
  sorry 
}

end dogs_not_eat_either_l183_183301


namespace equalChargesAtFour_agencyADecisionWhenTen_l183_183978

-- Define the conditions as constants
def fullPrice : ℕ := 240
def agencyADiscount : ℕ := 50
def agencyBDiscount : ℕ := 60

-- Define the total charge function for both agencies
def totalChargeAgencyA (students: ℕ) : ℕ :=
  fullPrice * students * agencyADiscount / 100 + fullPrice

def totalChargeAgencyB (students: ℕ) : ℕ :=
  fullPrice * (students + 1) * agencyBDiscount / 100

-- Define the equivalence when the number of students is 4
theorem equalChargesAtFour : totalChargeAgencyA 4 = totalChargeAgencyB 4 := by sorry

-- Define the decision when there are 10 students
theorem agencyADecisionWhenTen : totalChargeAgencyA 10 < totalChargeAgencyB 10 := by sorry

end equalChargesAtFour_agencyADecisionWhenTen_l183_183978


namespace instantaneous_velocity_at_3_l183_183192

noncomputable def displacement (t : ℝ) : ℝ := 
  - (1 / 3) * t^3 + 2 * t^2 - 5

theorem instantaneous_velocity_at_3 : 
  (deriv displacement 3 = 3) :=
by
  sorry

end instantaneous_velocity_at_3_l183_183192


namespace max_ab_is_5_l183_183568

noncomputable def max_ab : ℝ :=
  sorry

theorem max_ab_is_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h : a / 4 + b / 5 = 1) : max_ab = 5 :=
  sorry

end max_ab_is_5_l183_183568


namespace cube_difference_l183_183425

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l183_183425


namespace find_other_number_l183_183919

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l183_183919


namespace circles_exceeding_n_squared_l183_183454

noncomputable def num_circles (n : ℕ) : ℕ :=
  if n >= 8 then 
    5 * n + 4 * (n - 1)
  else 
    n * n

theorem circles_exceeding_n_squared (n : ℕ) (hn : n ≥ 8) : num_circles n > n^2 := 
by {
  sorry
}

end circles_exceeding_n_squared_l183_183454


namespace anusha_receives_84_l183_183375

-- Define the conditions as given in the problem
def anusha_amount (A : ℕ) (B : ℕ) (E : ℕ) : Prop :=
  12 * A = 8 * B ∧ 12 * A = 6 * E ∧ A + B + E = 378

-- Lean statement to prove the amount Anusha gets is 84
theorem anusha_receives_84 (A B E : ℕ) (h : anusha_amount A B E) : A = 84 :=
sorry

end anusha_receives_84_l183_183375


namespace james_sodas_per_day_l183_183161

theorem james_sodas_per_day : 
  (∃ packs : ℕ, ∃ sodas_per_pack : ℕ, ∃ additional_sodas : ℕ, ∃ days : ℕ,
    packs = 5 ∧ sodas_per_pack = 12 ∧ additional_sodas = 10 ∧ days = 7 ∧
    ((packs * sodas_per_pack + additional_sodas) / days) = 10) :=
by
  use 5, 12, 10, 7
  split; simp; sorry

end james_sodas_per_day_l183_183161


namespace num_valid_values_n_l183_183680

theorem num_valid_values_n :
  ∃ n : ℕ, (∃ a b c : ℕ,
    8 * a + 88 * b + 888 * c = 8880 ∧
    n = a + 2 * b + 3  * c) ∧
  (∃! k : ℕ, k = 119) :=
by sorry

end num_valid_values_n_l183_183680


namespace study_tour_buses_l183_183657

variable (x : ℕ) (num_people : ℕ)

def seats_A := 45
def seats_B := 60
def extra_people := 30
def fewer_B := 6

theorem study_tour_buses (h : seats_A * x + extra_people = seats_B * (x - fewer_B)) : 
  x = 26 ∧ (seats_A * 26 + extra_people = 1200) := 
  sorry

end study_tour_buses_l183_183657


namespace starWars_earnings_correct_l183_183484

-- Define the given conditions
def lionKing_cost : ℕ := 10
def lionKing_earnings : ℕ := 200
def starWars_cost : ℕ := 25
def lionKing_profit : ℕ := lionKing_earnings - lionKing_cost
def starWars_profit : ℕ := lionKing_profit * 2
def starWars_earnings : ℕ := starWars_profit + starWars_cost

-- The theorem which states that the Star Wars earnings are indeed 405 million
theorem starWars_earnings_correct : starWars_earnings = 405 := by
  -- proof goes here
  sorry

end starWars_earnings_correct_l183_183484


namespace servings_per_pie_l183_183164

theorem servings_per_pie (serving_apples : ℝ) (guests : ℕ) (pies : ℕ) (apples_per_guest : ℝ)
  (H_servings: serving_apples = 1.5) 
  (H_guests: guests = 12)
  (H_pies: pies = 3)
  (H_apples_per_guest: apples_per_guest = 3) :
  (guests * apples_per_guest) / (serving_apples * pies) = 8 :=
by
  rw [H_servings, H_guests, H_pies, H_apples_per_guest]
  sorry

end servings_per_pie_l183_183164


namespace larger_integer_is_30_l183_183764

-- Define the problem statement using the given conditions
theorem larger_integer_is_30 (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h1 : a / b = 5 / 2) (h2 : a * b = 360) :
  max a b = 30 :=
sorry

end larger_integer_is_30_l183_183764


namespace intersection_S_T_l183_183045

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183045


namespace cubic_difference_l183_183427

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l183_183427


namespace max_value_of_sum_l183_183899

open Real

theorem max_value_of_sum (x y z : ℝ)
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
    (h2 : (1 / x) + (1 / y) + (1 / z) + x + y + z = 0)
    (h3 : (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1) ∧ (z ≤ -1 ∨ z ≥ 1)) :
    x + y + z ≤ 0 := 
sorry

end max_value_of_sum_l183_183899


namespace sqrt2_times_sqrt5_eq_sqrt10_l183_183641

theorem sqrt2_times_sqrt5_eq_sqrt10 : (Real.sqrt 2) * (Real.sqrt 5) = Real.sqrt 10 := 
by
  sorry

end sqrt2_times_sqrt5_eq_sqrt10_l183_183641


namespace sum_sum_sum_sum_eq_one_l183_183316

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Mathematical problem statement
theorem sum_sum_sum_sum_eq_one :
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits (2017^2017)))) = 1 := 
sorry

end sum_sum_sum_sum_eq_one_l183_183316


namespace simplify_expression_l183_183333

theorem simplify_expression (y : ℝ) (hy : y ≠ 0) : 
  (2 / y^2 - y⁻¹) = (2 - y) / y^2 :=
by sorry

end simplify_expression_l183_183333


namespace cat_catches_total_birds_l183_183241

theorem cat_catches_total_birds :
  let morning_birds := 15
  let morning_success_rate := 0.60
  let afternoon_birds := 25
  let afternoon_success_rate := 0.80
  let night_birds := 20
  let night_success_rate := 0.90
  
  let morning_caught := morning_birds * morning_success_rate
  let afternoon_initial_caught := 2 * morning_caught
  let afternoon_caught := min (afternoon_birds * afternoon_success_rate) afternoon_initial_caught
  let night_caught := night_birds * night_success_rate

  let total_caught := morning_caught + afternoon_caught + night_caught
  total_caught = 47 := 
by
  sorry

end cat_catches_total_birds_l183_183241


namespace percentage_customers_return_books_l183_183403

theorem percentage_customers_return_books 
  (total_customers : ℕ) (price_per_book : ℕ) (sales_after_returns : ℕ) 
  (h1 : total_customers = 1000) 
  (h2 : price_per_book = 15) 
  (h3 : sales_after_returns = 9450) : 
  ((total_customers - (sales_after_returns / price_per_book)) / total_customers) * 100 = 37 := 
by
  sorry

end percentage_customers_return_books_l183_183403


namespace intersection_S_T_l183_183014

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183014


namespace sin_add_pi_over_4_eq_l183_183698

variable (α : Real)
variables (hα1 : 0 < α ∧ α < Real.pi) (hα2 : Real.tan (α - Real.pi / 4) = 1 / 3)

theorem sin_add_pi_over_4_eq : Real.sin (Real.pi / 4 + α) = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end sin_add_pi_over_4_eq_l183_183698


namespace intersection_S_T_l183_183074

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183074


namespace age_of_15th_student_is_15_l183_183761

-- Define the total number of students
def total_students : Nat := 15

-- Define the average age of all 15 students together
def avg_age_all_students : Nat := 15

-- Define the average age of the first group of 7 students
def avg_age_first_group : Nat := 14

-- Define the average age of the second group of 7 students
def avg_age_second_group : Nat := 16

-- Define the total age based on the average age and number of students
def total_age_all_students : Nat := total_students * avg_age_all_students
def total_age_first_group : Nat := 7 * avg_age_first_group
def total_age_second_group : Nat := 7 * avg_age_second_group

-- Define the age of the 15th student
def age_of_15th_student : Nat := total_age_all_students - (total_age_first_group + total_age_second_group)

-- Theorem: prove that the age of the 15th student is 15 years
theorem age_of_15th_student_is_15 : age_of_15th_student = 15 := by
  -- The proof will go here
  sorry

end age_of_15th_student_is_15_l183_183761


namespace minute_hand_rotation_l183_183670

theorem minute_hand_rotation :
  (10 / 60) * (2 * Real.pi) = (- Real.pi / 3) :=
by
  sorry

end minute_hand_rotation_l183_183670


namespace distinct_valid_sets_count_l183_183861

-- Define non-negative powers of 2 and 3
def is_non_neg_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a ∨ n = 3^b

-- Define the condition for sum of elements in set S to be 2014
def valid_sets (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, is_non_neg_power x) ∧ (S.sum id = 2014)

theorem distinct_valid_sets_count : ∃ (number_of_distinct_sets : ℕ), number_of_distinct_sets = 64 :=
  sorry

end distinct_valid_sets_count_l183_183861


namespace factor_exp_l183_183814

theorem factor_exp (k : ℕ) : 3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 :=
by
  intro h
  sorry

end factor_exp_l183_183814


namespace simplify_fraction_l183_183908

theorem simplify_fraction (b y : ℝ) (h : b^2 ≠ y^2) :
  (sqrt(b^2 + y^2) + (y^2 - b^2) / sqrt(b^2 + y^2)) / (b^2 - y^2) = (b^2 + y^2) / (b^2 - y^2) :=
by
  sorry

end simplify_fraction_l183_183908


namespace intersection_S_T_eq_T_l183_183118

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183118


namespace gcd_q_r_min_value_l183_183707

theorem gcd_q_r_min_value (p q r : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0)
  (h1 : Nat.gcd p q = 210) (h2 : Nat.gcd p r = 770) : Nat.gcd q r = 10 :=
sorry

end gcd_q_r_min_value_l183_183707


namespace composite_shape_sum_l183_183980

def triangular_prism_faces := 5
def triangular_prism_edges := 9
def triangular_prism_vertices := 6

def pentagonal_prism_additional_faces := 7
def pentagonal_prism_additional_edges := 10
def pentagonal_prism_additional_vertices := 5

def pyramid_additional_faces := 5
def pyramid_additional_edges := 5
def pyramid_additional_vertices := 1

def resulting_shape_faces := triangular_prism_faces - 1 + pentagonal_prism_additional_faces + pyramid_additional_faces
def resulting_shape_edges := triangular_prism_edges + pentagonal_prism_additional_edges + pyramid_additional_edges
def resulting_shape_vertices := triangular_prism_vertices + pentagonal_prism_additional_vertices + pyramid_additional_vertices

def sum_faces_edges_vertices := resulting_shape_faces + resulting_shape_edges + resulting_shape_vertices

theorem composite_shape_sum : sum_faces_edges_vertices = 51 :=
by
  unfold sum_faces_edges_vertices resulting_shape_faces resulting_shape_edges resulting_shape_vertices
  unfold triangular_prism_faces triangular_prism_edges triangular_prism_vertices
  unfold pentagonal_prism_additional_faces pentagonal_prism_additional_edges pentagonal_prism_additional_vertices
  unfold pyramid_additional_faces pyramid_additional_edges pyramid_additional_vertices
  simp
  sorry

end composite_shape_sum_l183_183980


namespace problem1_l183_183521

theorem problem1 (f : ℝ → ℝ) (x : ℝ) : 
  (f (x + 1/x) = x^2 + 1/x^2) -> f x = x^2 - 2 := 
sorry

end problem1_l183_183521


namespace intersection_at_one_point_l183_183696

theorem intersection_at_one_point (m : ℝ) :
  (∃ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 ∧
            ∀ x' : ℝ, (m - 4) * x'^2 - 2 * m * x' - m - 6 = 0 → x' = x) ↔
  m = -4 ∨ m = 3 ∨ m = 4 := 
by
  sorry

end intersection_at_one_point_l183_183696


namespace intersection_S_T_l183_183043

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183043


namespace intersection_of_S_and_T_l183_183103

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183103


namespace find_other_number_l183_183922

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l183_183922


namespace largest_cube_surface_area_l183_183964

theorem largest_cube_surface_area (width length height: ℕ) (h_w: width = 12) (h_l: length = 16) (h_h: height = 14) :
  (6 * (min width (min length height))^2) = 864 := by
  sorry

end largest_cube_surface_area_l183_183964


namespace no_perfect_square_in_range_l183_183134

def f (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 2

theorem no_perfect_square_in_range : ∀ (n : ℕ), 5 ≤ n → n ≤ 15 → ¬ ∃ (m : ℕ), f n = m^2 := by
  intros n h1 h2
  sorry

end no_perfect_square_in_range_l183_183134


namespace intersection_S_T_eq_T_l183_183123

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183123


namespace valid_votes_other_candidate_l183_183237

theorem valid_votes_other_candidate (total_votes : ℕ) (invalid_percentage : ℕ) (candidate1_percentage : ℕ) (valid_votes_other_candidate : ℕ) : 
  total_votes = 7500 → 
  invalid_percentage = 20 → 
  candidate1_percentage = 55 → 
  valid_votes_other_candidate = 2700 :=
by
  sorry

end valid_votes_other_candidate_l183_183237


namespace labels_closer_than_distance_l183_183407

noncomputable def exists_points_with_labels_closer_than_distance (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ |f P - f Q| < dist P Q

-- Statement of the problem
theorem labels_closer_than_distance :
  ∀ (f : ℝ × ℝ → ℝ), exists_points_with_labels_closer_than_distance f :=
sorry

end labels_closer_than_distance_l183_183407


namespace min_total_balls_l183_183202

theorem min_total_balls (R G B : Nat) (hG : G = 12) (hRG : R + G < 24) : 23 ≤ R + G + B :=
by {
  sorry
}

end min_total_balls_l183_183202


namespace find_ages_l183_183660

theorem find_ages (M F S : ℕ) 
  (h1 : M = 2 * F / 5)
  (h2 : M + 10 = (F + 10) / 2)
  (h3 : S + 10 = 3 * (F + 10) / 4) :
  M = 20 ∧ F = 50 ∧ S = 35 := 
by
  sorry

end find_ages_l183_183660


namespace shortest_side_of_similar_triangle_l183_183389

theorem shortest_side_of_similar_triangle (a b : ℕ) (c : ℝ) 
  (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2)
  (scale_factor : ℝ) (shortest_side_first : ℝ) (hypo_second : ℝ)
  (h4 : scale_factor = 100 / 25) 
  (h5 : hypo_second = 100) 
  (h6 : b = 7) 
  : (shortest_side_first * scale_factor = 28) :=
by
  sorry

end shortest_side_of_similar_triangle_l183_183389


namespace no_descending_digits_multiple_of_111_l183_183875

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end no_descending_digits_multiple_of_111_l183_183875


namespace find_common_ratio_l183_183168

variable {a : ℕ → ℝ} {q : ℝ}

-- Define that a is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : 0 < q)
  (h3 : a 1 * a 3 = 1)
  (h4 : sum_first_n_terms a 3 = 7) :
  q = 1 / 2 :=
sorry

end find_common_ratio_l183_183168


namespace intersection_S_T_l183_183019

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183019


namespace find_x_l183_183443

theorem find_x (y x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 := 
sorry

end find_x_l183_183443


namespace max_items_with_discount_l183_183665

theorem max_items_with_discount (total_money items original_price discount : ℕ) 
  (h_orig: original_price = 30)
  (h_discount: discount = 24) 
  (h_limit: items > 5 → (total_money <= 270)) : items ≤ 10 :=
by
  sorry

end max_items_with_discount_l183_183665


namespace solve_for_y_l183_183616

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l183_183616


namespace clock_angle_at_3_20_l183_183952

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l183_183952


namespace total_count_pens_pencils_markers_l183_183928

-- Define the conditions
def ratio_pens_pencils (pens pencils : ℕ) : Prop :=
  6 * pens = 5 * pencils

def nine_more_pencils (pens pencils : ℕ) : Prop :=
  pencils = pens + 9

def ratio_markers_pencils (markers pencils : ℕ) : Prop :=
  3 * markers = 4 * pencils

-- Theorem statement to be proved 
theorem total_count_pens_pencils_markers 
  (pens pencils markers : ℕ) 
  (h1 : ratio_pens_pencils pens pencils)
  (h2 : nine_more_pencils pens pencils)
  (h3 : ratio_markers_pencils markers pencils) : 
  pens + pencils + markers = 171 :=
sorry

end total_count_pens_pencils_markers_l183_183928


namespace solve_r_l183_183688

-- Definitions related to the problem
def satisfies_equation (r : ℝ) : Prop := ⌊r⌋ + 2 * r = 16

-- Theorem statement
theorem solve_r : ∃ (r : ℝ), satisfies_equation r ∧ r = 5.5 :=
by
  sorry

end solve_r_l183_183688


namespace distance_between_vertices_of_hyperbola_l183_183690

theorem distance_between_vertices_of_hyperbola :
  (∃ a b : ℝ, a^2 = 144 ∧ b^2 = 64 ∧ 
    ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → true) → 
  (2 * real.sqrt 144 = 24) :=
by
  sorry

end distance_between_vertices_of_hyperbola_l183_183690


namespace floor_expression_equality_l183_183817

theorem floor_expression_equality :
  ⌊((2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023))⌋ = 8 := 
sorry

end floor_expression_equality_l183_183817


namespace Joe_speed_first_part_l183_183880

theorem Joe_speed_first_part
  (dist1 dist2 : ℕ)
  (speed2 avg_speed total_distance total_time : ℕ)
  (h1 : dist1 = 180)
  (h2 : dist2 = 120)
  (h3 : speed2 = 40)
  (h4 : avg_speed = 50)
  (h5 : total_distance = dist1 + dist2)
  (h6 : total_distance = 300)
  (h7 : total_time = total_distance / avg_speed)
  (h8 : total_time = 6) :
  ∃ v : ℕ, (dist1 / v + dist2 / speed2 = total_time) ∧ v = 60 :=
by
  sorry

end Joe_speed_first_part_l183_183880


namespace find_y_l183_183837

open Real

def vecV (y : ℝ) : ℝ × ℝ := (1, y)
def vecW : ℝ × ℝ := (6, 4)

noncomputable def dotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

noncomputable def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dotProduct v w) / (dotProduct w w)
  (scalar * w.1, scalar * w.2)

theorem find_y (y : ℝ) (h : projection (vecV y) vecW = (3, 2)) : y = 5 := by
  sorry

end find_y_l183_183837


namespace area_of_square_with_diagonal_l183_183865

theorem area_of_square_with_diagonal (c : ℝ) : 
  (∃ (s : ℝ), 2 * s^2 = c^4) → (∃ (A : ℝ), A = (c^4 / 2)) :=
  by
    sorry

end area_of_square_with_diagonal_l183_183865


namespace game_ends_and_last_numbers_depend_on_start_l183_183741
-- Given that there are three positive integers a, b, c initially.
variables (a b c : ℕ)
-- Assume that a, b, and c are greater than zero.
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Define the gcd of the three numbers.
def g := gcd (gcd a b) c

-- Define the game step condition.
def step_condition (a b c : ℕ): Prop := a > gcd b c

-- Define the termination condition.
def termination_condition (a b c : ℕ): Prop := ¬ step_condition a b c

-- The main theorem
theorem game_ends_and_last_numbers_depend_on_start (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ n, ∃ b' c', termination_condition n b' c' ∧
  n = g ∧ b' = g ∧ c' = g :=
sorry

end game_ends_and_last_numbers_depend_on_start_l183_183741


namespace maximize_expression_l183_183886

theorem maximize_expression
  (a b c : ℝ)
  (h1 : a ≥ 0)
  (h2 : b ≥ 0)
  (h3 : c ≥ 0)
  (h_sum : a + b + c = 3) :
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 729 / 432 := 
sorry

end maximize_expression_l183_183886


namespace largest_n_binomial_l183_183207

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l183_183207


namespace like_terms_set_l183_183645

theorem like_terms_set (a b : ℕ) (x y : ℝ) : 
  (¬ (a = b)) ∧
  ((-2 * x^3 * y^3 = y^3 * x^3)) ∧ 
  (¬ (1 * x * y = 2 * x * y^3)) ∧ 
  (¬ (-6 = x)) :=
by
  sorry

end like_terms_set_l183_183645


namespace conference_center_capacity_l183_183242

theorem conference_center_capacity (n_rooms : ℕ) (fraction_full : ℚ) (current_people : ℕ) (full_capacity : ℕ) (people_per_room : ℕ) 
  (h1 : n_rooms = 6) (h2 : fraction_full = 2/3) (h3 : current_people = 320) (h4 : current_people = fraction_full * full_capacity) 
  (h5 : people_per_room = full_capacity / n_rooms) : people_per_room = 80 :=
by
  -- The proof will go here.
  sorry

end conference_center_capacity_l183_183242


namespace investment_interest_min_l183_183673

theorem investment_interest_min (x y : ℝ) (hx : x + y = 25000) (hmax : x ≤ 11000) : 
  0.07 * x + 0.12 * y ≥ 2450 :=
by
  sorry

end investment_interest_min_l183_183673


namespace seashells_increase_l183_183408

def initial_seashells : ℕ := 50
def final_seashells : ℕ := 130
def week_increment (x : ℕ) : ℕ := 4 * x + initial_seashells

theorem seashells_increase (x : ℕ) (h: final_seashells = week_increment x) : x = 8 :=
by {
  sorry
}

end seashells_increase_l183_183408


namespace min_value_m_n_l183_183423

theorem min_value_m_n (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_geom_mean : a * b = 4)
    (m n : ℝ) (h_m : m = b + 1 / a) (h_n : n = a + 1 / b) : m + n ≥ 5 :=
by
  sorry

end min_value_m_n_l183_183423


namespace ax_by_powers_l183_183138

theorem ax_by_powers (a b x y : ℝ) (h1 : a * x + b * y = 5) 
                      (h2: a * x^2 + b * y^2 = 11)
                      (h3: a * x^3 + b * y^3 = 25)
                      (h4: a * x^4 + b * y^4 = 59) : 
                      a * x^5 + b * y^5 = 145 := 
by 
  -- Include the proof steps here if needed 
  sorry

end ax_by_powers_l183_183138


namespace diagonals_of_seven_sided_polygon_l183_183531

-- Define the number of sides of the polygon
def n : ℕ := 7

-- Calculate the number of diagonals in a polygon with n sides
def numberOfDiagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- The statement to prove
theorem diagonals_of_seven_sided_polygon : numberOfDiagonals n = 14 := by
  -- Here we will write the proof steps, but they're not needed now.
  sorry

end diagonals_of_seven_sided_polygon_l183_183531


namespace solution_set_of_inequality_l183_183494

theorem solution_set_of_inequality (x : ℝ) : x^2 < -2 * x + 15 ↔ -5 < x ∧ x < 3 := 
sorry

end solution_set_of_inequality_l183_183494


namespace probability_of_continuous_stripe_pattern_l183_183823

def tetrahedron_stripes := 
  let faces := 4
  let configurations_per_face := 2
  2 ^ faces

def continuous_stripe_probability := 
  let total_configurations := tetrahedron_stripes
  1 / total_configurations * 4 -- Since final favorable outcomes calculation is already given and inferred to be 1/4.
  -- or any other logic that follows here based on problem description but this matches problem's derivation

theorem probability_of_continuous_stripe_pattern : continuous_stripe_probability = 1 / 4 := by
  sorry

end probability_of_continuous_stripe_pattern_l183_183823


namespace real_values_of_x_l183_183829

theorem real_values_of_x :
  {x : ℝ | (∃ y, y = (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ∧ y ≥ -1)} =
  {x | -1 ≤ x ∧ x < -1/3 ∨ -1/3 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 1 < x} := 
sorry

end real_values_of_x_l183_183829


namespace clock_angle_320_l183_183945

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l183_183945


namespace solve_m_n_l183_183748

theorem solve_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 :=
sorry

end solve_m_n_l183_183748


namespace arithmetic_sequence_sum_l183_183990

theorem arithmetic_sequence_sum :
  let a1 := 1
  let d := 2
  let n := 10
  let an := 19
  let sum := 100
  let general_term := fun (n : ℕ) => a1 + (n - 1) * d
  (general_term n = an) → (n = 10) → (sum = (n * (a1 + an)) / 2) →
  sum = 100 :=
by
  sorry

end arithmetic_sequence_sum_l183_183990


namespace ellen_needs_thirteen_golf_carts_l183_183825

theorem ellen_needs_thirteen_golf_carts :
  ∀ (patrons_from_cars patrons_from_bus patrons_per_cart : ℕ), 
  patrons_from_cars = 12 → 
  patrons_from_bus = 27 → 
  patrons_per_cart = 3 →
  (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := 
by 
  intros patrons_from_cars patrons_from_bus patrons_per_cart h1 h2 h3 
  have h: patrons_from_cars + patrons_from_bus = 39 := by 
    rw [h1, h2] 
    norm_num
  rw[h, h3]
  norm_num
  sorry

end ellen_needs_thirteen_golf_carts_l183_183825


namespace intersection_S_T_l183_183022

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183022


namespace rainfall_ratio_l183_183160

theorem rainfall_ratio (S M T : ℝ) (h1 : M = S + 3) (h2 : S = 4) (h3 : S + M + T = 25) : T / M = 2 :=
by
  sorry

end rainfall_ratio_l183_183160


namespace pot_holds_three_liters_l183_183909

theorem pot_holds_three_liters (drips_per_minute : ℕ) (ml_per_drop : ℕ) (minutes : ℕ) (full_pot_volume : ℕ) :
  drips_per_minute = 3 → ml_per_drop = 20 → minutes = 50 → full_pot_volume = (drips_per_minute * ml_per_drop * minutes) / 1000 →
  full_pot_volume = 3 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end pot_holds_three_liters_l183_183909


namespace smaller_angle_at_3_20_l183_183951

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l183_183951


namespace minimum_Q_l183_183392

def is_special (m : ℕ) : Prop :=
  let d1 := m / 10 
  let d2 := m % 10
  d1 ≠ d2 ∧ d1 ≠ 0 ∧ d2 ≠ 0

def F (m : ℕ) : ℤ :=
  let d1 := m / 10
  let d2 := m % 10
  (d1 * 100 + d2 * 10 + d1) - (d2 * 100 + d1 * 10 + d2) / 99

def Q (s t : ℕ) : ℚ :=
  (t - s) / s

variables (a b x y : ℕ)
variables (h1 : 1 ≤ b ∧ b < a ∧ a ≤ 7)
variables (h2 : 1 ≤ x ∧ x ≤ 8)
variables (h3 : 1 ≤ y ∧ y ≤ 8)
variables (hs_is_special : is_special (10 * a + b))
variables (ht_is_special : is_special (10 * x + y))
variables (s := 10 * a + b)
variables (t := 10 * x + y)
variables (h4 : (F s % 5) = 1)
variables (h5 : F t - F s + 18 * x = 36)

theorem minimum_Q : Q s t = -42 / 73 := sorry

end minimum_Q_l183_183392


namespace solve_for_y_l183_183622

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l183_183622


namespace inequality_proof_l183_183732

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l183_183732


namespace bell_ratio_l183_183176

theorem bell_ratio :
  ∃ (B3 B2 : ℕ), 
  B2 = 2 * 50 ∧ 
  50 + B2 + B3 = 550 ∧ 
  (B3 / B2 = 4) := 
sorry

end bell_ratio_l183_183176


namespace isosceles_trapezoid_AB_length_l183_183456

theorem isosceles_trapezoid_AB_length (BC AD : ℝ) (r : ℝ) (a : ℝ) (h_isosceles : BC = a) (h_ratio : AD = 3 * a) (h_area : 4 * a * r = Real.sqrt 3 / 2) (h_radius : r = a * Real.sqrt 3 / 2) :
  2 * a = 1 :=
by
 sorry

end isosceles_trapezoid_AB_length_l183_183456


namespace interval_satisfies_inequality_l183_183830

theorem interval_satisfies_inequality :
  { x : ℝ | x ∈ [-1, -1/3) ∪ (-1/3, 0) ∪ (0, 1) ∪ (1, ∞) } =
  { x : ℝ | x^2 + 2*x^3 - 3*x^4 ≠ 0 ∧ x + 2*x^2 - 3*x^3 ≠ 0 ∧ (x >= -1 ∧ (x < 1 ∨ x > -1/3)) ∧ 
            x^2 + 2*x^3 - 3*x^4 / (x + 2*x^2 - 3*x^3) ≥ -1 } := sorry

end interval_satisfies_inequality_l183_183830


namespace part_a_part_b_l183_183351

noncomputable def tsunami_area_center_face (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  180000 * Real.pi + 270000 * Real.sqrt 3

noncomputable def tsunami_area_mid_edge (l : ℝ) (v : ℝ) (t : ℝ) : ℝ :=
  720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7

theorem part_a (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_center_face l v t = 180000 * Real.pi + 270000 * Real.sqrt 3 :=
by
  sorry

theorem part_b (l v t : ℝ) (hl : l = 900) (hv : v = 300) (ht : t = 2) :
  tsunami_area_mid_edge l v t = 720000 * Real.arccos (3 / 4) + 135000 * Real.sqrt 7 :=
by
  sorry

end part_a_part_b_l183_183351


namespace more_green_peaches_than_red_l183_183630

theorem more_green_peaches_than_red : 
  let red_peaches := 7
  let green_peaches := 8
  green_peaches - red_peaches = 1 := 
by
  let red_peaches := 7
  let green_peaches := 8
  show green_peaches - red_peaches = 1 
  sorry

end more_green_peaches_than_red_l183_183630


namespace expression_value_l183_183437

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add_prop (a b : ℝ) : f (a + b) = f a * f b
axiom f_one_val : f 1 = 2

theorem expression_value : 
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 +
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 
  = 16 := 
sorry

end expression_value_l183_183437


namespace both_subjects_sum_l183_183406

-- Define the total number of students
def N : ℕ := 1500

-- Define the bounds for students studying Biology (B) and Chemistry (C)
def B_min : ℕ := 900
def B_max : ℕ := 1050

def C_min : ℕ := 600
def C_max : ℕ := 750

-- Let x and y be the smallest and largest number of students studying both subjects
def x : ℕ := B_max + C_max - N
def y : ℕ := B_min + C_min - N

-- Prove that y + x = 300
theorem both_subjects_sum : y + x = 300 := by
  sorry

end both_subjects_sum_l183_183406


namespace triangle_ratio_l183_183594

/-- Problem statement -/
theorem triangle_ratio
  {A B C D E : Type}
  {angle_A : ℝ} {angle_B : ℝ} {angle_ADE : ℝ}
  {area_ADE_area_ABC_ratio : ℝ}
  (hA : angle_A = 60)
  (hB : angle_B = 45)
  (hADE : angle_ADE = 75)
  (h_area_ratio : area_ADE_area_ABC_ratio = 1 / 3) :
  AD / AB = 1 / 3 :=
sorry

end triangle_ratio_l183_183594


namespace cube_difference_l183_183426

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l183_183426


namespace sum_of_first_five_multiples_of_15_l183_183511

theorem sum_of_first_five_multiples_of_15 : (15 + 30 + 45 + 60 + 75) = 225 :=
by sorry

end sum_of_first_five_multiples_of_15_l183_183511


namespace cab_speed_fraction_l183_183523

theorem cab_speed_fraction :
  ∀ (S R : ℝ),
    (75 * S = 90 * R) →
    (R / S = 5 / 6) :=
by
  intros S R h
  sorry

end cab_speed_fraction_l183_183523


namespace intersection_S_T_eq_T_l183_183117

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183117


namespace intersection_S_T_eq_T_l183_183063

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183063


namespace slope_of_line_l183_183391

theorem slope_of_line :
  ∃ (m : ℝ), (∃ b : ℝ, ∀ x y : ℝ, y = m * x + b) ∧
             (b = 2 ∧ ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ = 0 ∧ x₂ = 269 ∧ y₁ = 2 ∧ y₂ = 540 ∧ 
             m = (y₂ - y₁) / (x₂ - x₁)) ∧
             m = 2 :=
by {
  sorry
}

end slope_of_line_l183_183391


namespace turtle_reaches_watering_hole_l183_183646

theorem turtle_reaches_watering_hole:
  ∀ (x y : ℝ)
   (dist_first_cub : ℝ := 6 * y)
   (dist_turtle : ℝ := 32 * x)
   (time_first_encounter := (dist_first_cub - dist_turtle) / (y - x))
   (time_second_encounter := dist_turtle / (x + 1.5 * y))
   (time_between_encounters = 2.4),
   time_second_encounter - time_first_encounter = time_between_encounters →
   (time_turtle_reaches := time_second_encounter + 32 - 3.2) -- since total_time - the time between second 
   →
   time_turtle_reaches = 28.8 :=
begin
  intros x y,
  sorry
end

end turtle_reaches_watering_hole_l183_183646


namespace polynomial_roots_cubed_l183_183600

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 3
noncomputable def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 3

theorem polynomial_roots_cubed {r : ℝ} (h : f r = 0) :
  g (r^3) = 0 := by
  sorry

end polynomial_roots_cubed_l183_183600


namespace sheila_initial_savings_l183_183332

noncomputable def initial_savings (monthly_savings : ℕ) (years : ℕ) (family_addition : ℕ) (total_amount : ℕ) : ℕ :=
  total_amount - (monthly_savings * 12 * years + family_addition)

def sheila_initial_savings_proof : Prop :=
  initial_savings 276 4 7000 23248 = 3000

theorem sheila_initial_savings : sheila_initial_savings_proof :=
  by
    -- Proof goes here
    sorry

end sheila_initial_savings_l183_183332


namespace oranges_per_pack_correct_l183_183338

-- Definitions for the conditions.
def num_trees : Nat := 10
def oranges_per_tree_per_day : Nat := 12
def price_per_pack : Nat := 2
def total_earnings : Nat := 840
def weeks : Nat := 3
def days_per_week : Nat := 7

-- Theorem statement:
theorem oranges_per_pack_correct :
  let oranges_per_day := num_trees * oranges_per_tree_per_day
  let total_days := weeks * days_per_week
  let total_oranges := oranges_per_day * total_days
  let num_packs := total_earnings / price_per_pack
  total_oranges / num_packs = 6 :=
by
  sorry

end oranges_per_pack_correct_l183_183338


namespace circle_equation_correct_l183_183355

theorem circle_equation_correct (x y : ℝ) :
  let h : ℝ := -2
  let k : ℝ := 2
  let r : ℝ := 5
  ((x - h)^2 + (y - k)^2 = r^2) ↔ ((x + 2)^2 + (y - 2)^2 = 25) :=
by
  sorry

end circle_equation_correct_l183_183355


namespace sum_sum_sum_sum_eq_one_l183_183315

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Mathematical problem statement
theorem sum_sum_sum_sum_eq_one :
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits (2017^2017)))) = 1 := 
sorry

end sum_sum_sum_sum_eq_one_l183_183315


namespace find_other_number_l183_183920

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l183_183920


namespace man_speed_against_current_eq_l183_183386

-- Definitions
def downstream_speed : ℝ := 22 -- Man's speed with the current in km/hr
def current_speed : ℝ := 5 -- Speed of the current in km/hr

-- Man's speed in still water
def man_speed_in_still_water : ℝ := downstream_speed - current_speed

-- Man's speed against the current
def speed_against_current : ℝ := man_speed_in_still_water - current_speed

-- Theorem: The man's speed against the current is 12 km/hr.
theorem man_speed_against_current_eq : speed_against_current = 12 := by
  sorry

end man_speed_against_current_eq_l183_183386


namespace smaller_angle_clock_3_20_l183_183947

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l183_183947


namespace complex_power_of_sum_l183_183700

theorem complex_power_of_sum (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end complex_power_of_sum_l183_183700


namespace cylinder_line_intersection_l183_183832

noncomputable def intersection_points (R x₀ y₀ a b : ℝ) : Set ℝ :=
  {t | let delta := (x₀ * a + y₀ * b) ^ 2 - (a ^ 2 + b ^ 2) * (x₀ ^ 2 + y₀ ^ 2 - R ^ 2) in
       let numerator₁ := - (x₀ * a + y₀ * b) + Real.sqrt delta in
       let numerator₂ := - (x₀ * a + y₀ * b) - Real.sqrt delta in
       let denominator := a ^ 2 + b ^ 2 in
       t = numerator₁ / denominator ∨ t = numerator₂ / denominator }

theorem cylinder_line_intersection
  (R x₀ y₀ z₀ a b c : ℝ) :
  ∀ t ∈ intersection_points R x₀ y₀ a b,
  (let Lx := x₀ + a * t in
   let Ly := y₀ + b * t in
   let Lz := z₀ + c * t in
   Lx ^ 2 + Ly ^ 2 = R ^ 2) :=
by intros t ht
   obtain ⟨delta, numerator₁, numerator₂, denominator, ht₁, ht₂⟩ := ht
   sorry

end cylinder_line_intersection_l183_183832


namespace trapezoid_midsegment_inscribed_circle_l183_183195

theorem trapezoid_midsegment_inscribed_circle (P : ℝ) (hP : P = 40) 
    (inscribed : Π (a b c d : ℝ), a + b = c + d) : 
    (∃ (c d : ℝ), (c + d) / 2 = 10) :=
by
  sorry

end trapezoid_midsegment_inscribed_circle_l183_183195


namespace sufficient_but_not_necessary_condition_l183_183697

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 0 → |x| > 0) ∧ (¬ (|x| > 0 → x > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l183_183697


namespace find_first_term_of_sequence_l183_183169

theorem find_first_term_of_sequence
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : ∀ n, a (n+1) = a n + d)
  (h2 : a 0 + a 1 + a 2 = 12)
  (h3 : a 0 * a 1 * a 2 = 48)
  (h4 : ∀ n m, n < m → a n ≤ a m) :
  a 0 = 2 :=
sorry

end find_first_term_of_sequence_l183_183169


namespace total_volume_of_five_boxes_l183_183369

-- Define the edge length of each cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_cube (s : ℕ) : ℕ := s ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 5

-- Define the total volume
def total_volume (s : ℕ) (n : ℕ) : ℕ := n * (volume_of_cube s)

-- The theorem to prove
theorem total_volume_of_five_boxes :
  total_volume edge_length number_of_cubes = 625 := 
by
  -- Proof is skipped
  sorry

end total_volume_of_five_boxes_l183_183369


namespace intersection_S_T_eq_T_l183_183066

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183066


namespace two_candidates_solve_all_problems_l183_183379

-- Definitions for the conditions and problem context
def candidates : Nat := 200
def problems : Nat := 6 
def solved_by (p : Nat) : Nat := 120 -- at least 120 participants solve each problem.

-- The main theorem representing the proof problem
theorem two_candidates_solve_all_problems :
  (∃ c1 c2 : Fin candidates, ∀ p : Fin problems, (solved_by p ≥ 120)) :=
by
  sorry

end two_candidates_solve_all_problems_l183_183379


namespace intersection_eq_T_l183_183028

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183028


namespace monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l183_183855

def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_when_non_positive (a : ℝ) (h_a : a ≤ 0) : ∀ x : ℝ, (a * Real.exp x - 1) ≤ 0 := 
by 
  intro x
  sorry

theorem monotonicity_when_positive (a : ℝ) (h_a : a > 0) : 
  ∀ x : ℝ, 
    (x < Real.log (1 / a) → (a * Real.exp x - 1) < 0) ∧ 
    (x > Real.log (1 / a) → (a * Real.exp x - 1) > 0) := 
by 
  intro x
  sorry

theorem inequality_when_positive (a : ℝ) (h_a : a > 0) : ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := 
by 
  intro x
  sorry

end monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l183_183855


namespace minimum_steps_to_catch_thief_l183_183808

-- Definitions of positions A, B, C, D, etc., along the board
-- Assuming the positions and movement rules are predefined somewhere in the environment.
-- For a simple abstract model, we assume the following:
-- The positions are nodes in a graph, and each move is one step along the edges of this graph.

def Position : Type := String -- This can be refined to reflect the actual chessboard structure.
def neighbor (p1 p2 : Position) : Prop := sorry -- Predicate defining that p1 and p2 are neighbors.

-- Positions are predefined for simplicity.
def A : Position := "A"
def B : Position := "B"
def C : Position := "C"
def D : Position := "D"
def F : Position := "F"

-- Condition: policeman and thief take turns moving, starting with the policeman.
-- Initial positions of the policeman and the thief.
def policemanStart : Position := A
def thiefStart : Position := B

-- Statement: Prove that the policeman can catch the thief in a minimum of 4 moves.
theorem minimum_steps_to_catch_thief (policeman thief : Position) (turns : ℕ) :
  policeman = policemanStart →
  thief = thiefStart →
  (∀ t < turns, (neighbor policeman thief)) →
  (turns = 4) :=
sorry

end minimum_steps_to_catch_thief_l183_183808


namespace percentage_of_trout_is_correct_l183_183398

-- Define the conditions
def video_game_cost := 60
def last_weekend_earnings := 35
def earnings_per_trout := 5
def earnings_per_bluegill := 4
def total_fish_caught := 5
def additional_savings_needed := 2

-- Define the total amount needed to buy the game
def total_required_savings := video_game_cost - additional_savings_needed

-- Define the amount earned this Sunday
def earnings_this_sunday := total_required_savings - last_weekend_earnings

-- Define the number of trout and blue-gill caught thisSunday
def num_trout := 3
def num_bluegill := 2    -- Derived from the conditions

-- Theorem: given the conditions, prove that the percentage of trout is 60%
theorem percentage_of_trout_is_correct :
  (num_trout + num_bluegill = total_fish_caught) ∧
  (earnings_per_trout * num_trout + earnings_per_bluegill * num_bluegill = earnings_this_sunday) →
  100 * num_trout / total_fish_caught = 60 := 
by
  sorry

end percentage_of_trout_is_correct_l183_183398


namespace remaining_distance_is_one_l183_183459

def total_distance_to_grandma : ℕ := 78
def initial_distance_traveled : ℕ := 35
def bakery_detour : ℕ := 7
def pie_distance : ℕ := 18
def gift_detour : ℕ := 3
def next_travel_distance : ℕ := 12
def scenic_detour : ℕ := 2

def total_distance_traveled : ℕ :=
  initial_distance_traveled + bakery_detour + pie_distance + gift_detour + next_travel_distance + scenic_detour

theorem remaining_distance_is_one :
  total_distance_to_grandma - total_distance_traveled = 1 := by
  sorry

end remaining_distance_is_one_l183_183459


namespace seth_spent_more_on_ice_cream_l183_183610

-- Definitions based on the conditions
def cartons_ice_cream := 20
def cartons_yogurt := 2
def cost_per_carton_ice_cream := 6
def cost_per_carton_yogurt := 1

-- Theorem statement
theorem seth_spent_more_on_ice_cream :
  (cartons_ice_cream * cost_per_carton_ice_cream) - (cartons_yogurt * cost_per_carton_yogurt) = 118 :=
by
  sorry

end seth_spent_more_on_ice_cream_l183_183610


namespace pirate_rick_dig_time_l183_183327
noncomputable def time_to_dig_up_treasure (initial_sand : ℕ) (final_sand : ℕ) (dig_time_per_foot : ℕ) : ℕ :=
  final_sand * dig_time_per_foot

theorem pirate_rick_dig_time :
  let initial_sand := 8 in
  let initial_time := 4 in
  let storm_fraction := 1 / 2 in
  let tsunami_sand := 2 in
  let dig_rate := initial_time / initial_sand in
  let storm_sand := initial_sand * storm_fraction in
  let total_sand := storm_sand + tsunami_sand in
  time_to_dig_up_treasure initial_sand total_sand dig_rate = 3 :=
by
  sorry

end pirate_rick_dig_time_l183_183327


namespace intersection_S_T_eq_T_l183_183057

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183057


namespace smaller_angle_clock_3_20_l183_183946

theorem smaller_angle_clock_3_20 : 
  let angle := 160 in
  angle = 160 := by
sorry

end smaller_angle_clock_3_20_l183_183946


namespace count_valid_B_is_6_l183_183131

def S : Finset ℕ := {0, 1, 2, 3, 4, 5}

def is_isolated_element (A : Finset ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ (x - 1 ∉ A) ∧ (x + 1 ∉ A)

def no_isolated_elements (A : Finset ℕ) : Prop :=
  ∀ x ∈ A, ¬ is_isolated_element A x

def valid_B (B : Finset ℕ) : Prop :=
  B ⊆ S ∧ B.card = 4 ∧ no_isolated_elements B

noncomputable def count_valid_B : ℕ :=
  (S.powerset.filter valid_B).card

theorem count_valid_B_is_6 : count_valid_B = 6 := by
  sorry

end count_valid_B_is_6_l183_183131


namespace expected_score_particular_player_l183_183543

-- Define types of dice
inductive DiceType : Type
| A | B | C

-- Define the faces of each dice type
def DiceFaces : DiceType → List ℕ
| DiceType.A => [2, 2, 4, 4, 9, 9]
| DiceType.B => [1, 1, 6, 6, 8, 8]
| DiceType.C => [3, 3, 5, 5, 7, 7]

-- Define a function to calculate the score of a player given their roll and opponents' rolls
def player_score (p_roll : ℕ) (opp_rolls : List ℕ) : ℕ :=
  opp_rolls.foldl (λ acc roll => if roll < p_roll then acc + 1 else acc) 0

-- Define a function to calculate the expected score of a player
noncomputable def expected_score (dice_choice : DiceType) : ℚ :=
  let rolls := DiceFaces dice_choice
  let total_possibilities := (rolls.length : ℚ) ^ 3
  let score_sum := rolls.foldl (λ acc p_roll =>
    acc + rolls.foldl (λ acc1 opp1_roll =>
        acc1 + rolls.foldl (λ acc2 opp2_roll =>
            acc2 + player_score p_roll [opp1_roll, opp2_roll]
          ) 0
      ) 0
    ) 0
  score_sum / total_possibilities

-- The main theorem statement
theorem expected_score_particular_player : (expected_score DiceType.A + expected_score DiceType.B + expected_score DiceType.C) / 3 = 
(8 : ℚ) / 9 := sorry

end expected_score_particular_player_l183_183543


namespace right_triangle_perimeter_area_ratio_l183_183994

theorem right_triangle_perimeter_area_ratio 
  (a b : ℝ) (h : a > 0 ∧ b > 0) 
  (hyp : ∀ c, c = Real.sqrt (a^2 + b^2))
  : (a + b + Real.sqrt (a^2 + b^2)) / (0.5 * a * b) = 5 → (∃! x y : ℝ, x + y + Real.sqrt (x^2 + y^2) / (0.5 * x * y) = 5) :=
by
  sorry   -- Proof is omitted as per instructions.

end right_triangle_perimeter_area_ratio_l183_183994


namespace solve_for_y_l183_183617

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l183_183617


namespace arithmetic_sequence_a15_l183_183564

theorem arithmetic_sequence_a15 (a_n S_n : ℕ → ℝ) (a_9 : a_n 9 = 4) (S_15 : S_n 15 = 30) :
  let a_1 := (-12 : ℝ)
  let d := (2 : ℝ)
  a_n 15 = 16 :=
by
  sorry

end arithmetic_sequence_a15_l183_183564


namespace johns_earnings_without_bonus_l183_183596
-- Import the Mathlib library to access all necessary functions and definitions

-- Define the conditions of the problem
def hours_without_bonus : ℕ := 8
def bonus_amount : ℕ := 20
def extra_hours_for_bonus : ℕ := 2
def hours_with_bonus : ℕ := hours_without_bonus + extra_hours_for_bonus
def hourly_wage_with_bonus : ℕ := 10

-- Define the total earnings with the performance bonus
def total_earnings_with_bonus : ℕ := hours_with_bonus * hourly_wage_with_bonus

-- Statement to prove the earnings without the bonus
theorem johns_earnings_without_bonus :
  total_earnings_with_bonus - bonus_amount = 80 :=
by
  -- Placeholder for the proof
  sorry

end johns_earnings_without_bonus_l183_183596


namespace intersection_S_T_l183_183070

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183070


namespace P_and_Q_together_l183_183178

theorem P_and_Q_together (W : ℝ) (H : W > 0) :
  (1 / (1 / 4 + 1 / (1 / 3 * (1 / 4)))) = 3 :=
by
  sorry

end P_and_Q_together_l183_183178


namespace parabola_ord_l183_183324

theorem parabola_ord {M : ℝ × ℝ} (h1 : M.1 = (M.2 * M.2) / 8) (h2 : dist M (2, 0) = 4) : M.2 = 4 ∨ M.2 = -4 := 
sorry

end parabola_ord_l183_183324


namespace contrapositive_example_l183_183906

theorem contrapositive_example (a b : ℝ) :
  (a > b → a - 1 > b - 2) ↔ (a - 1 ≤ b - 2 → a ≤ b) := 
by
  sorry

end contrapositive_example_l183_183906


namespace magic_square_expression_l183_183397

theorem magic_square_expression : 
  let a := 8
  let b := 6
  let c := 14
  let d := 10
  let e := 11
  let f := 5
  let g := 3
  a - b - c + d + e + f - g = 11 :=
by
  sorry

end magic_square_expression_l183_183397


namespace valid_selling_price_l183_183452

-- Define the initial conditions
def cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def sales_increase_per_dollar_decrease : ℝ := 4
def max_profit : ℝ := 13600
def min_selling_price : ℝ := 150

-- Define x as the price reduction per item
variable (x : ℝ)

-- Define the function relationship of the daily sales volume y with respect to x
def sales_volume (x : ℝ) := 100 + 4 * x

-- Define the selling price based on the price reduction
def selling_price (x : ℝ) := 200 - x

-- Calculate the profit based on the selling price and sales volume
def profit (x : ℝ) := (selling_price x - cost_price) * (sales_volume x)

-- Lean theorem statement to prove the given conditions lead to the valid selling price
theorem valid_selling_price (x : ℝ) 
  (h1 : profit x = 13600)
  (h2 : selling_price x ≥ 150) : 
  selling_price x = 185 :=
sorry

end valid_selling_price_l183_183452


namespace correct_calculation_l183_183232

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end correct_calculation_l183_183232


namespace fred_total_cards_l183_183414

theorem fred_total_cards 
  (initial_cards : ℕ := 26) 
  (cards_given_to_mary : ℕ := 18) 
  (unopened_box_cards : ℕ := 40) : 
  initial_cards - cards_given_to_mary + unopened_box_cards = 48 := 
by 
  sorry

end fred_total_cards_l183_183414


namespace find_f_value_find_g_monotonicity_and_extremes_l183_183436

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 3) - cos (2 * x - π / 3)
noncomputable def g (x : ℝ) : ℝ := -2 * cos ((1/2) * x - π / 3)

theorem find_f_value :
  f (π / 24) = - (sqrt 6 + sqrt 2) / 2 := sorry

theorem find_g_monotonicity_and_extremes :
  (∀ k : ℤ, ∃ min_x max_x : ℝ, 
    g (4 * k * π + 2 * π / 3) ≤ g x ∧ g x ≤ g (4 * k * π + 14 * π / 3) ∧ 
    min_x = 4 * k * π + 8 * π / 3 ∧ max_x = 4 * k * π + 2 * π / 3) ∧
  g (-π / 3) = 0 ∧ g (2π / 3) = -2 := sorry

end find_f_value_find_g_monotonicity_and_extremes_l183_183436


namespace consecutive_integers_sum_l183_183280

theorem consecutive_integers_sum (a b : ℤ) (h1 : a < b) (h2 : b = a + 1) (h3 : a < Real.sqrt 17) (h4 : Real.sqrt 17 < b) : a + b = 9 :=
sorry

end consecutive_integers_sum_l183_183280


namespace shopkeeper_loss_percent_l183_183528

theorem shopkeeper_loss_percent 
  (C : ℝ) (P : ℝ) (L : ℝ) 
  (hC : C = 100) 
  (hP : P = 10) 
  (hL : L = 50) : 
  ((C - (((C * (1 - L / 100)) * (1 + P / 100))) / C) * 100) = 45 :=
by
  sorry

end shopkeeper_loss_percent_l183_183528


namespace range_of_a_l183_183128

noncomputable def f (x a : ℝ) : ℝ := x - (a+1) * Real.log x - a / x

noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 + Real.exp x - x * Real.exp x

theorem range_of_a :
  ∀ a : ℝ, (a < 1) →
  (∃ x1 ∈ Icc Real.exp (Real.exp 2), ∀ x2 ∈ Icc (-2) 0, f x1 a < g x2) →
  a ∈ Ioo ((Real.exp 2 - 2 * Real.exp) / (Real.exp + 1)) 1 :=
sorry

end range_of_a_l183_183128


namespace perpendicular_lines_condition_l183_183193

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * (m + 1) * x + (m - 3) * y + 7 - 5 * m = 0) ↔ (∀ x y : ℝ, (m - 3) * x + 2 * y - 5 = 0) →
  (m = 3 ∨ m = -2) :=
sorry

end perpendicular_lines_condition_l183_183193


namespace factory_selection_and_probability_l183_183363

/-- Total number of factories in districts A, B, and C --/
def factories_A := 18
def factories_B := 27
def factories_C := 18

/-- Total number of factories and sample size --/
def total_factories := factories_A + factories_B + factories_C
def sample_size := 7

/-- Number of factories selected from districts A, B, and C --/
def selected_from_A := factories_A * sample_size / total_factories
def selected_from_B := factories_B * sample_size / total_factories
def selected_from_C := factories_C * sample_size / total_factories

/-- Number of ways to choose 2 factories out of the 7 --/
noncomputable def comb_7_2 := Nat.choose 7 2

/-- Number of favorable outcomes where at least one factory comes from district A --/
noncomputable def favorable_outcomes := 11

/-- Probability that at least one of the 2 factories comes from district A --/
noncomputable def probability := favorable_outcomes / comb_7_2

theorem factory_selection_and_probability :
  selected_from_A = 2 ∧ selected_from_B = 3 ∧ selected_from_C = 2 ∧ probability = 11 / 21 := by
  sorry

end factory_selection_and_probability_l183_183363


namespace intersection_S_T_eq_T_l183_183008

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183008


namespace num_multiples_3003_in_form_10j_sub_10i_l183_183136

theorem num_multiples_3003_in_form_10j_sub_10i :
  ∃ n : ℕ, n = 192 ∧ ∀ i j : ℕ, 0 ≤ i → i < j → j ≤ 50 →
  (3003 ∣ (10^j - 10^i) ↔ ∃ k : ℕ, j - i = 6 * k) :=
by {
  use 192,
  sorry
}

end num_multiples_3003_in_form_10j_sub_10i_l183_183136


namespace coefficient_of_x9_in_expansion_l183_183367

-- Definitions as given in the problem
def binomial_expansion_coeff (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * a^(n - k) * b^k

-- Mathematically equivalent statement in Lean 4
theorem coefficient_of_x9_in_expansion : binomial_expansion_coeff 10 9 (-2) 1 = -20 :=
by
  sorry

end coefficient_of_x9_in_expansion_l183_183367


namespace christine_final_throw_difference_l183_183679

def christine_first_throw : ℕ := 20
def janice_first_throw : ℕ := christine_first_throw - 4
def christine_second_throw : ℕ := christine_first_throw + 10
def janice_second_throw : ℕ := janice_first_throw * 2
def janice_final_throw : ℕ := christine_first_throw + 17
def highest_throw : ℕ := 37

theorem christine_final_throw_difference :
  ∃ x : ℕ, christine_second_throw + x = highest_throw ∧ x = 7 := by 
sorry

end christine_final_throw_difference_l183_183679


namespace div_40_of_prime_ge7_l183_183706

theorem div_40_of_prime_ge7 (p : ℕ) (hp_prime : Prime p) (hp_ge7 : p ≥ 7) : 40 ∣ (p^2 - 1) :=
sorry

end div_40_of_prime_ge7_l183_183706


namespace prove_positions_l183_183716

def athlete : Type := { A : ℕ, B : ℕ, C : ℕ, D : ℕ, E : ℕ }

def first_prediction (x : athlete) : athlete :=
  { A := 1, B := 2, C := 3, D := 4, E := 5 }

def second_prediction (x : athlete) : athlete :=
  { A := 3, B := 4, C := 1, D := 5, E := 2 }

def actual_positions : athlete :=
  { A := 3, B := 2, C := 1, D := 4, E := 5 }

theorem prove_positions :
  (λ x, first_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 3 ∧
  (λ x, second_prediction x = { A := x.A, B := x.B, C := x.C, D := x.D, E := x.E }) actual_positions = 2 ∧
  actual_positions.A = 3 ∧ actual_positions.B = 2 ∧ actual_positions.C = 1 ∧ actual_positions.D = 4 ∧ actual_positions.E = 5 := 
  sorry

end prove_positions_l183_183716


namespace smallest_integer_proof_l183_183509

def smallest_integer_condition (n : ℤ) : Prop := n^2 - 15 * n + 56 ≤ 0

theorem smallest_integer_proof :
  ∃ n : ℤ, smallest_integer_condition n ∧ ∀ m : ℤ, smallest_integer_condition m → n ≤ m :=
sorry

end smallest_integer_proof_l183_183509


namespace tan_double_angle_l183_183416

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin (Real.pi / 2 + theta) + Real.sin (Real.pi + theta) = 0) :
  Real.tan (2 * theta) = -4 / 3 :=
by
  sorry

end tan_double_angle_l183_183416


namespace intersection_S_T_l183_183076

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183076


namespace unattainable_y_l183_183274

theorem unattainable_y (x : ℝ) (h : x ≠ -(5 / 4)) :
    (∀ y : ℝ, y = (2 - 3 * x) / (4 * x + 5) → y ≠ -3 / 4) :=
by
  -- Placeholder for the proof
  sorry

end unattainable_y_l183_183274


namespace property_holds_for_1_and_4_l183_183490

theorem property_holds_for_1_and_4 (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end property_holds_for_1_and_4_l183_183490


namespace temperature_at_4km_l183_183925

theorem temperature_at_4km (ground_temp : ℤ) (drop_rate : ℤ) (altitude : ℕ) (ΔT : ℤ) : 
  ground_temp = 15 ∧ drop_rate = -5 ∧ ΔT = altitude * drop_rate ∧ altitude = 4 → 
  ground_temp + ΔT = -5 :=
by
  sorry

end temperature_at_4km_l183_183925


namespace trigonometric_inequality_l183_183473

theorem trigonometric_inequality (x : ℝ) : 0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧ 
                                            5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 :=
by
  sorry

end trigonometric_inequality_l183_183473


namespace sum_of_consecutive_numbers_with_lcm_168_l183_183917

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l183_183917


namespace set_union_complement_l183_183284

-- Definitions based on provided problem statement
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}
def CRQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- The theorem to prove
theorem set_union_complement : P ∪ CRQ = {x | -2 < x ∧ x ≤ 3} :=
by
  -- Skip the proof
  sorry

end set_union_complement_l183_183284


namespace largest_integer_comb_l183_183219

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l183_183219


namespace first_box_weight_l183_183313

theorem first_box_weight (X : ℕ) 
  (h1 : 11 + 5 + X = 18) : X = 2 := 
by
  sorry

end first_box_weight_l183_183313


namespace cards_probability_l183_183139

-- Definitions based on conditions
def total_cards := 52
def suits := 4
def cards_per_suit := 13

-- Introducing probabilities for the conditions mentioned
def prob_first := 1
def prob_second := 39 / 52
def prob_third := 26 / 52
def prob_fourth := 13 / 52
def prob_fifth := 26 / 52

-- The problem statement
theorem cards_probability :
  (prob_first * prob_second * prob_third * prob_fourth * prob_fifth) = (3 / 64) :=
by
  sorry

end cards_probability_l183_183139


namespace intersection_S_T_eq_T_l183_183119

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183119


namespace equivalent_resistance_A_B_l183_183413

-- Parameters and conditions
def resistor_value : ℝ := 5 -- in MΩ
def num_resistors : ℕ := 4
def has_bridging_wire : Prop := true
def negligible_wire_resistance : Prop := true

-- Problem: Prove the equivalent resistance (R_eff) between points A and B is 5 MΩ.
theorem equivalent_resistance_A_B : 
  ∀ (R : ℝ) (n : ℕ) (bridge : Prop) (negligible_wire : Prop),
    R = 5 → n = 4 → bridge → negligible_wire → R = 5 :=
by sorry

end equivalent_resistance_A_B_l183_183413


namespace circumcenter_locus_l183_183816

noncomputable def tangent_circles (ω Ω : Circle) (X Y T : Point) (P S : Point) (hx : Center ω = X) (hy : Center Ω = Y)
  (ht : IsTangentInternally ω Ω T) (on_omega : OnCircle P Ω) (on_ω : OnCircle S ω)
  (tan : TangentToLine PS ω S) : Set Point :=
{O | IsCircumcenter O P S T}

theorem circumcenter_locus (ω Ω : Circle) (X Y T : Point) (P S : Point) (hx : Center ω = X) (hy : Center Ω = Y)
  (ht : IsTangentInternally ω Ω T) (on_omega : OnCircle P Ω) (on_ω : OnCircle S ω)
  (tan : TangentToLine PS ω S) :
  ∀ O ∈ tangent_circles ω Ω X Y T P S hx hy ht on_omega on_ω tan,
  (Distance O Y = sqrt (Distance X Y * Distance X T)) ∧ O ∉ LineIntersectCircle XY (CircleCenterRadius Y (sqrt (Distance X Y * Distance X T))) :=
sorry

end circumcenter_locus_l183_183816


namespace geometric_sequence_a5_l183_183421

variable {a : Nat → ℝ} {q : ℝ}

-- Conditions
def is_geometric_sequence (a : Nat → ℝ) (q : ℝ) :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = q * a n

def condition_eq (a : Nat → ℝ) :=
  a 5 + a 4 = 3 * (a 3 + a 2)

-- Proof statement
theorem geometric_sequence_a5 (hq : q ≠ -1)
  (hg : is_geometric_sequence a q)
  (hc : condition_eq a) : a 5 = 9 :=
  sorry

end geometric_sequence_a5_l183_183421


namespace smallest_number_diminished_by_10_l183_183518

theorem smallest_number_diminished_by_10 (x : ℕ) (h : ∀ n, x - 10 = 24 * n) : x = 34 := 
  sorry

end smallest_number_diminished_by_10_l183_183518


namespace average_annual_percentage_decrease_l183_183675

theorem average_annual_percentage_decrease (P2018 P2020 : ℝ) (x : ℝ) 
  (h_initial : P2018 = 20000)
  (h_final : P2020 = 16200) :
  P2018 * (1 - x)^2 = P2020 :=
by
  sorry

end average_annual_percentage_decrease_l183_183675


namespace fractional_part_of_water_after_replacements_l183_183969

theorem fractional_part_of_water_after_replacements :
  let total_quarts := 25
  let removed_quarts := 5
  (1 - removed_quarts / (total_quarts : ℚ))^3 = 64 / 125 :=
by
  sorry

end fractional_part_of_water_after_replacements_l183_183969


namespace roxy_bought_flowering_plants_l183_183182

-- Definitions based on conditions
def initial_flowering_plants : ℕ := 7
def initial_fruiting_plants : ℕ := 2 * initial_flowering_plants
def plants_after_saturday (F : ℕ) : ℕ := initial_flowering_plants + F + initial_fruiting_plants + 2
def plants_after_sunday (F : ℕ) : ℕ := (initial_flowering_plants + F - 1) + (initial_fruiting_plants + 2 - 4)
def final_plants_in_garden : ℕ := 21

-- The proof statement
theorem roxy_bought_flowering_plants (F : ℕ) :
  plants_after_sunday F = final_plants_in_garden → F = 3 := 
sorry

end roxy_bought_flowering_plants_l183_183182


namespace intersection_S_T_l183_183081

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183081


namespace probability_of_specific_card_sequence_is_25_3978_l183_183260

def probability_top_red_second_black_third_red_heart :=
  let total_cards := 104
  let total_red := 52
  let total_black := 52
  let total_hearts := 26
  let probability_first_red := (total_red : ℚ) / total_cards
  let remaining_after_first_red := total_cards - 1
  let remaining_red_after_first := total_red - 1
  let probability_second_black := (total_black : ℚ) / remaining_after_first_red
  let remaining_after_second_black := remaining_after_first_red - 1
  let remaining_hearts_after_first := total_hearts - 1
  let probability_third_red_heart := (remaining_hearts_after_first : ℚ) / remaining_after_second_black
  probability_first_red * probability_second_black * probability_third_red_heart

theorem probability_of_specific_card_sequence_is_25_3978 :
  probability_top_red_second_black_third_red_heart = (25 : ℚ) / 3978 := 
by
  sorry

end probability_of_specific_card_sequence_is_25_3978_l183_183260


namespace incorrect_desc_is_C_l183_183394
noncomputable def incorrect_geometric_solid_desc : Prop :=
  ¬ (∀ (plane_parallel: Prop), 
      plane_parallel ∧ 
      (∀ (frustum: Prop), frustum ↔ 
        (∃ (base section_cut cone : Prop), 
          cone ∧ 
          (section_cut = plane_parallel) ∧ 
          (frustum = (base ∧ section_cut)))))

theorem incorrect_desc_is_C (plane_parallel frustum base section_cut cone : Prop) :
  incorrect_geometric_solid_desc := 
by
  sorry

end incorrect_desc_is_C_l183_183394


namespace units_digit_of_m_squared_plus_3_to_the_m_l183_183172

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_3_to_the_m 
  (hm : m = 2021^3 + 3^2021) : 
  ((m^2 + 3^m) % 10) = 7 := 
by 
  -- Here you would input the proof steps, however, we skip it now with sorry.
  sorry

end units_digit_of_m_squared_plus_3_to_the_m_l183_183172


namespace Greg_more_than_Sharon_l183_183133

-- Define the harvest amounts
def Greg_harvest : ℝ := 0.4
def Sharon_harvest : ℝ := 0.1

-- Show that Greg harvested 0.3 more acres than Sharon
theorem Greg_more_than_Sharon : Greg_harvest - Sharon_harvest = 0.3 := by
  sorry

end Greg_more_than_Sharon_l183_183133


namespace calculate_highest_score_l183_183188

noncomputable def highest_score (avg_60 : ℕ) (delta_HL : ℕ) (avg_58 : ℕ) : ℕ :=
  let total_60 := 60 * avg_60
  let total_58 := 58 * avg_58
  let sum_HL := total_60 - total_58
  let L := (sum_HL - delta_HL) / 2
  let H := L + delta_HL
  H

theorem calculate_highest_score :
  highest_score 55 200 52 = 242 :=
by
  sorry

end calculate_highest_score_l183_183188


namespace can_determine_number_of_spies_l183_183671

def determine_spies (V : Fin 15 → ℕ) (S : Fin 15 → ℕ) : Prop :=
  V 0 = S 0 + S 1 ∧ 
  ∀ i : Fin 13, V (Fin.succ (Fin.succ i)) = S i + S (Fin.succ i) + S (Fin.succ (Fin.succ i)) ∧
  V 14 = S 13 + S 14

theorem can_determine_number_of_spies :
  ∃ S : Fin 15 → ℕ, ∀ V : Fin 15 → ℕ, determine_spies V S :=
sorry

end can_determine_number_of_spies_l183_183671


namespace monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l183_183854

def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_when_non_positive (a : ℝ) (h_a : a ≤ 0) : ∀ x : ℝ, (a * Real.exp x - 1) ≤ 0 := 
by 
  intro x
  sorry

theorem monotonicity_when_positive (a : ℝ) (h_a : a > 0) : 
  ∀ x : ℝ, 
    (x < Real.log (1 / a) → (a * Real.exp x - 1) < 0) ∧ 
    (x > Real.log (1 / a) → (a * Real.exp x - 1) > 0) := 
by 
  intro x
  sorry

theorem inequality_when_positive (a : ℝ) (h_a : a > 0) : ∀ x : ℝ, f a x > 2 * Real.log a + 3/2 := 
by 
  intro x
  sorry

end monotonicity_when_non_positive_monotonicity_when_positive_inequality_when_positive_l183_183854


namespace intersection_S_T_l183_183051

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183051


namespace problem1_problem2_l183_183279

-- Define the propositions
def S (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

def p (m : ℝ) : Prop := 0 < m ∧ m < 2

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ 1 ≤ m := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hpq : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l183_183279


namespace Margo_James_pairs_probability_l183_183383

def total_students : ℕ := 32
def Margo_pairs_prob : ℚ := 1 / 31
def James_pairs_prob : ℚ := 1 / 30
def total_prob : ℚ := Margo_pairs_prob * James_pairs_prob

theorem Margo_James_pairs_probability :
  total_prob = 1 / 930 := 
by
  -- sorry allows us to skip the proof steps, only statement needed
  sorry

end Margo_James_pairs_probability_l183_183383


namespace min_value_problem_l183_183170

noncomputable def min_value (a b c d e f : ℝ) := (2 / a) + (3 / b) + (9 / c) + (16 / d) + (25 / e) + (36 / f)

theorem min_value_problem 
  (a b c d e f : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) 
  (h_sum : a + b + c + d + e + f = 10) : 
  min_value a b c d e f >= (329 + 38 * Real.sqrt 6) / 10 := 
sorry

end min_value_problem_l183_183170


namespace mrs_jackson_boxes_l183_183470

theorem mrs_jackson_boxes (decorations_per_box used_decorations given_decorations : ℤ) 
(h1 : decorations_per_box = 15)
(h2 : used_decorations = 35)
(h3 : given_decorations = 25) :
  (used_decorations + given_decorations) / decorations_per_box = 4 := 
by sorry

end mrs_jackson_boxes_l183_183470


namespace maximum_value_expression_l183_183889

theorem maximum_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_sum : a + b + c = 3) : 
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 1 :=
sorry

end maximum_value_expression_l183_183889


namespace first_stack_height_l183_183606

theorem first_stack_height (x : ℕ) (h1 : x + (x + 2) + (x - 3) + (x + 2) = 21) : x = 5 :=
by
  sorry

end first_stack_height_l183_183606


namespace athlete_positions_l183_183715

theorem athlete_positions {A B C D E : Type} 
  (h1 : A = 1 ∧ B = 2 ∧ C = 3 ∧ D = 4 ∧ E = 5)
  (h2 : C = 1 ∧ E = 2 ∧ A = 3 ∧ B = 4 ∧ D = 5)
  (h3 : (A = 1 ∧ B = 2 ∧ C = 3 ∧ ¬(D = 4 ∨ E = 5)) ∨ 
       (A = 1 ∧ ¬(B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)) ∨ 
       (B = 2 ∧ C = 3 ∧ ¬(A = 1 ∨ D = 4 ∨ E = 5)) ∨ 
       (¬(A = 1 ∨ B = 2 ∨ C = 3 ∨ D = 4 ∨ E = 5)))
  (h4 : (C = 1 ∧ ¬(A = 3 ∨ B = 4 ∨ D = 5)) ∨ 
       (A = 3 ∧ ¬(C = 1 ∨ B = 4 ∨ D = 5)) ∨ 
       (B = 4 ∧ ¬(C = 1 ∨ A = 3 ∨ D = 5)))
  : (C = 1 ∧ B = 2 ∧ A = 3 ∧ D = 4 ∧ E = 5) :=
by sorry

end athlete_positions_l183_183715


namespace intersection_of_S_and_T_l183_183097

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183097


namespace lcm_of_numbers_l183_183555

/-- Define the numbers involved -/
def a := 456
def b := 783
def c := 935
def d := 1024
def e := 1297

/-- Prove the LCM of these numbers is 2308474368000 -/
theorem lcm_of_numbers :
  Int.lcm (Int.lcm (Int.lcm (Int.lcm a b) c) d) e = 2308474368000 :=
by
  sorry

end lcm_of_numbers_l183_183555


namespace largest_integer_binom_l183_183211

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l183_183211


namespace no_descending_digits_multiple_of_111_l183_183876

theorem no_descending_digits_multiple_of_111 (n : ℕ) (h_desc : (∀ i j, i < j → (n % 10 ^ (i + 1)) / 10 ^ i ≥ (n % 10 ^ (j + 1)) / 10 ^ j)) :
  ¬(111 ∣ n) :=
sorry

end no_descending_digits_multiple_of_111_l183_183876


namespace intersection_eq_T_l183_183038

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183038


namespace problem_statement_l183_183173

-- Given the conditions and the goal
theorem problem_statement (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxyz_sum : x + y + z = 1) :
  (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 :=
by
  sorry

end problem_statement_l183_183173


namespace cakes_served_for_lunch_l183_183800

theorem cakes_served_for_lunch (total_cakes: ℕ) (dinner_cakes: ℕ) (lunch_cakes: ℕ) 
  (h1: total_cakes = 15) 
  (h2: dinner_cakes = 9) 
  (h3: total_cakes = lunch_cakes + dinner_cakes) : 
  lunch_cakes = 6 := 
by 
  sorry

end cakes_served_for_lunch_l183_183800


namespace find_number_l183_183377

theorem find_number (n : ℝ) : (2629.76 / n = 528.0642570281125) → n = 4.979 :=
by
  intro h
  sorry

end find_number_l183_183377


namespace pirate_rick_digging_time_l183_183326

theorem pirate_rick_digging_time :
  ∀ (initial_depth rate: ℕ) (storm_factor tsunami_added: ℕ),
  initial_depth = 8 →
  rate = 2 →
  storm_factor = 2 →
  tsunami_added = 2 →
  (initial_depth / storm_factor + tsunami_added) / rate = 3 := 
by
  intros
  sorry

end pirate_rick_digging_time_l183_183326


namespace probability_at_most_one_hit_l183_183380

noncomputable def P {Ω : Type*} [MeasureSpace Ω] (P : MeasureTheory.ProbabilityMeasure Ω) (A B : Set Ω) : ℝ := 
  P.measure A * P.measure B + (P.measure (Aᶜ) * P.measure B) + (P.measure A * P.measure (Bᶜ)) + (P.measure (Aᶜ) * P.measure (Bᶜ))

theorem probability_at_most_one_hit (P : MeasureTheory.ProbabilityMeasure) (A B : Set Ω) 
  (hA : P.measure A = 0.6) 
  (hB : P.measure B = 0.7) 
  (h_indep : MeasureTheory.Independence P A B) :
  P.measure A * P.measure B = 0.42 → 
  P.measure (Aᶜ ∩ Aᶜ) = 0.58 :=
  sorry

end probability_at_most_one_hit_l183_183380


namespace karl_total_miles_l183_183166

def car_mileage_per_gallon : ℕ := 30
def full_tank_gallons : ℕ := 14
def initial_drive_miles : ℕ := 300
def gas_bought_gallons : ℕ := 10
def final_tank_fraction : ℚ := 1 / 3

theorem karl_total_miles (initial_fuel : ℕ) :
  initial_fuel = full_tank_gallons →
  (initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons) = initial_fuel - (initial_fuel * final_tank_fraction) / car_mileage_per_gallon + (580 - initial_drive_miles) / car_mileage_per_gallon →
  initial_drive_miles + (initial_fuel - initial_drive_miles / car_mileage_per_gallon + gas_bought_gallons - initial_fuel * final_tank_fraction / car_mileage_per_gallon) * car_mileage_per_gallon = 580 := 
sorry

end karl_total_miles_l183_183166


namespace fraction_B_A_C_l183_183791

theorem fraction_B_A_C (A B C : ℕ) (x : ℚ) 
  (h1 : A = (1 / 3) * (B + C)) 
  (h2 : A = B + 10) 
  (h3 : A + B + C = 360) : 
  x = 2 / 7 ∧ B = x * (A + C) :=
by
  sorry -- The proof steps can be filled in

end fraction_B_A_C_l183_183791


namespace sum_of_cubes_l183_183586

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 :=
by
  sorry

end sum_of_cubes_l183_183586


namespace smallest_integer_y_l183_183510

theorem smallest_integer_y : ∃ (y : ℤ), (7 + 3 * y < 25) ∧ (∀ z : ℤ, (7 + 3 * z < 25) → y ≤ z) ∧ y = 5 :=
by
  sorry

end smallest_integer_y_l183_183510


namespace horses_lcm_l183_183499

theorem horses_lcm :
  let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
  let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
  let time_T := lcm_six
  lcm_six = 420 ∧ (Nat.digits 10 time_T).sum = 6 := by
    let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
    let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
    let time_T := lcm_six
    have h1 : lcm_six = 420 := sorry
    have h2 : (Nat.digits 10 time_T).sum = 6 := sorry
    exact ⟨h1, h2⟩

end horses_lcm_l183_183499


namespace intersection_S_T_eq_T_l183_183065

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183065


namespace intersection_S_T_l183_183021

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183021


namespace intersection_of_S_and_T_l183_183109

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183109


namespace find_oysters_first_day_l183_183203

variable (O : ℕ)  -- Number of oysters on the rocks on the first day

def count_crabs_first_day := 72  -- Number of crabs on the beach on the first day

def oysters_second_day := O / 2  -- Number of oysters on the rocks on the second day

def crabs_second_day := (2 / 3) * count_crabs_first_day  -- Number of crabs on the beach on the second day

def total_count := 195  -- Total number of oysters and crabs counted over the two days

theorem find_oysters_first_day (h:  O + oysters_second_day O + count_crabs_first_day + crabs_second_day = total_count) : 
  O = 50 := by
  sorry

end find_oysters_first_day_l183_183203


namespace intersection_S_T_eq_T_l183_183087

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183087


namespace closest_integer_to_a2013_l183_183765

noncomputable def sequence (n : ℕ) : ℝ :=
  nat.rec_on n 100 (λ k ak, ak + 1 / ak)

theorem closest_integer_to_a2013 : ∃ (z : ℤ), z = 118 ∧ abs (sequence 2013 - z) = min (abs (sequence 2013 - (z - 1))) (abs (sequence 2013 - (z + 1))) :=
begin
  sorry
end

end closest_integer_to_a2013_l183_183765


namespace solve_for_y_l183_183615

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l183_183615


namespace madeline_flower_count_l183_183175

theorem madeline_flower_count 
    (r w : ℕ) 
    (b_percent : ℝ) 
    (total : ℕ) 
    (h_r : r = 4)
    (h_w : w = 2)
    (h_b_percent : b_percent = 0.40)
    (h_total : r + w + (b_percent * total) = total) : 
    total = 10 :=
by 
    sorry

end madeline_flower_count_l183_183175


namespace find_x_l183_183289

theorem find_x (x : ℕ) (hv1 : x % 6 = 0) (hv2 : x^2 > 144) (hv3 : x < 30) : x = 18 ∨ x = 24 :=
  sorry

end find_x_l183_183289


namespace greatest_integer_jo_thinking_of_l183_183721

theorem greatest_integer_jo_thinking_of :
  ∃ n : ℕ, n < 150 ∧ (∃ k : ℕ, n = 9 * k - 1) ∧ (∃ m : ℕ, n = 5 * m - 2) ∧ n = 143 :=
by
  sorry

end greatest_integer_jo_thinking_of_l183_183721


namespace intersection_S_T_eq_T_l183_183003

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183003


namespace exists_set_X_gcd_condition_l183_183897

theorem exists_set_X_gcd_condition :
  ∃ (X : Finset ℕ), X.card = 2022 ∧
  (∀ (a b c : ℕ) (n : ℕ) (ha : a ∈ X) (hb : b ∈ X) (hc : c ∈ X) (hn_pos : 0 < n)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c),
  Nat.gcd (a^n + b^n) c = 1) :=
sorry

end exists_set_X_gcd_condition_l183_183897


namespace range_of_a_l183_183446

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x - a

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a > 2 - 2 * Real.log 2 :=
by
  sorry

end range_of_a_l183_183446


namespace intersection_S_T_eq_T_l183_183124

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183124


namespace intersection_S_T_l183_183050

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183050


namespace no_descending_multiple_of_111_l183_183872

-- Hypotheses
def digits_descending (n : ℕ) : Prop := 
  ∀ i j, i < j → (n.digits.get i) > (n.digits.get j)

def is_multiple_of_111 (n : ℕ) : Prop := 
  n % 111 = 0

-- Conclusion
theorem no_descending_multiple_of_111 :
  ∀ n : ℕ, digits_descending n ∧ is_multiple_of_111 n → false :=
by sorry

end no_descending_multiple_of_111_l183_183872


namespace mary_rental_hours_l183_183322

-- Definitions of the given conditions
def fixed_fee : ℝ := 17
def hourly_rate : ℝ := 7
def total_paid : ℝ := 80

-- Goal: Prove that the number of hours Mary paid for is 9
theorem mary_rental_hours : (total_paid - fixed_fee) / hourly_rate = 9 := 
by
  sorry

end mary_rental_hours_l183_183322


namespace not_divisible_l183_183519

theorem not_divisible (x y : ℕ) (hx : x % 61 ≠ 0) (hy : y % 61 ≠ 0) (h : (7 * x + 34 * y) % 61 = 0) : (5 * x + 16 * y) % 61 ≠ 0 := 
sorry

end not_divisible_l183_183519


namespace rationalize_denominator_l183_183475

theorem rationalize_denominator (A B C D E : ℤ) 
  (hB_lt_D : B < D) (h_fraction : (5 : ℝ) / (4*real.sqrt 7 + 3*real.sqrt 13) = (A*real.sqrt B + C*real.sqrt D) / E) 
  (h_simplest_form : true) -- assume we have the simplest terms, this would need further detail in a full proof
  : A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_l183_183475


namespace total_dog_food_per_day_l183_183410

-- Definitions based on conditions
def dog1_eats_per_day : ℝ := 0.125
def dog2_eats_per_day : ℝ := 0.125
def number_of_dogs : ℕ := 2

-- Mathematically equivalent proof problem statement
theorem total_dog_food_per_day : dog1_eats_per_day + dog2_eats_per_day = 0.25 := 
by
  sorry

end total_dog_food_per_day_l183_183410


namespace ellen_needs_thirteen_golf_carts_l183_183824

theorem ellen_needs_thirteen_golf_carts :
  ∀ (patrons_from_cars patrons_from_bus patrons_per_cart : ℕ), 
  patrons_from_cars = 12 → 
  patrons_from_bus = 27 → 
  patrons_per_cart = 3 →
  (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := 
by 
  intros patrons_from_cars patrons_from_bus patrons_per_cart h1 h2 h3 
  have h: patrons_from_cars + patrons_from_bus = 39 := by 
    rw [h1, h2] 
    norm_num
  rw[h, h3]
  norm_num
  sorry

end ellen_needs_thirteen_golf_carts_l183_183824


namespace intersection_of_S_and_T_l183_183099

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183099


namespace intersection_eq_T_l183_183029

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183029


namespace reach_any_composite_from_4_l183_183159

def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

def can_reach (A : ℕ) : Prop :=
  ∀ n : ℕ, is_composite n → ∃ seq : ℕ → ℕ, seq 0 = A ∧ seq (n + 1) - seq n ∣ seq n ∧ seq (n + 1) ≠ seq n ∧ seq (n + 1) ≠ 1 ∧ seq (n + 1) = n

theorem reach_any_composite_from_4 : can_reach 4 :=
  sorry

end reach_any_composite_from_4_l183_183159


namespace find_phi_l183_183141

open Real

theorem find_phi (φ : ℝ) (hφ : |φ| < π / 2)
  (h_symm : ∀ x, sin (2 * x + φ) = sin (2 * ((2 * π / 3 - x) / 2) + φ)) :
  φ = -π / 6 :=
by
  sorry

end find_phi_l183_183141


namespace peg_board_unique_arrangement_l183_183544

-- Let's assume a type of colors for clarity
inductive Color
| yellow
| red
| green
| blue
| orange

-- The number of pegs for each color
def yellow_pegs := 6
def red_pegs := 5
def green_pegs := 4
def blue_pegs := 3
def orange_pegs := 2

noncomputable def unique_peg_arrangement : Prop :=
  ∃ arrangement : Fin yellow_pegs → Fin yellow_pegs → Option Color,
    (∀ i, (∃ c, ∀ j, arrangement i j = some c) ∧ (∀ j₁ j₂, j₁ ≠ j₂ → arrangement i j₁ ≠ arrangement i j₂)) ∧
    (∀ j, (∃ c, ∀ i, arrangement i j = some c) ∧ (∀ i₁ i₂, i₁ ≠ i₂ → arrangement i j ≠ arrangement i j₂)) ∧
    (∃! σ, ∀ (i : Fin yellow_pegs), arrangement i i = some σ)

theorem peg_board_unique_arrangement : unique_peg_arrangement := 
by sorry

end peg_board_unique_arrangement_l183_183544


namespace first_floor_cost_l183_183460

-- Definitions and assumptions
variables (F : ℝ)
variables (earnings_first_floor earnings_second_floor earnings_third_floor : ℝ)
variables (total_monthly_earnings : ℝ)

-- Conditions from the problem
def costs := F
def second_floor_costs := F + 20
def third_floor_costs := 2 * F
def first_floor_rooms := 3 * costs
def second_floor_rooms := 3 * second_floor_costs
def third_floor_rooms := 3 * third_floor_costs

-- Total monthly earnings
def total_earnings := first_floor_rooms + second_floor_rooms + third_floor_rooms

-- Equality condition
axiom total_earnings_is_correct : total_earnings = 165

-- Theorem to be proved
theorem first_floor_cost :
  (F = 8.75) :=
by
  have earnings_first_floor_eq := first_floor_rooms
  have earnings_second_floor_eq := second_floor_rooms
  have earnings_third_floor_eq := third_floor_rooms
  have total_earning_eq := total_earnings_is_correct
  sorry

end first_floor_cost_l183_183460


namespace simplify_expression_l183_183613

theorem simplify_expression (y : ℝ) : 
  y - 3 * (2 + y) + 4 * (2 - y^2) - 5 * (2 + 3 * y) = -4 * y^2 - 17 * y - 8 :=
by
  sorry

end simplify_expression_l183_183613


namespace train_speed_l183_183936

/-- 
Train A leaves the station traveling at a certain speed v. 
Two hours later, Train B leaves the same station traveling in the same direction at 36 miles per hour. 
Train A was overtaken by Train B 360 miles from the station.
We need to prove that the speed of Train A was 30 miles per hour.
-/
theorem train_speed (v : ℕ) (t : ℕ) (h1 : 36 * (t - 2) = 360) (h2 : v * t = 360) : v = 30 :=
by 
  sorry

end train_speed_l183_183936


namespace remainder_when_divided_by_3x_minus_6_l183_183640

def polynomial (x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 9 * x^4 + 3 * x^3 - 7

def evaluate_at (f : ℝ → ℝ) (a : ℝ) : ℝ := f a

theorem remainder_when_divided_by_3x_minus_6 :
  evaluate_at polynomial 2 = 897 :=
by
  -- Compute this value manually or use automated tools
  sorry

end remainder_when_divided_by_3x_minus_6_l183_183640


namespace tiles_difference_eighth_sixth_l183_183798

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := n

-- Define the number of tiles given the side length
def number_of_tiles (n : ℕ) : ℕ := n * n

-- State the theorem about the difference in tiles between the 8th and 6th squares
theorem tiles_difference_eighth_sixth :
  number_of_tiles (side_length 8) - number_of_tiles (side_length 6) = 28 :=
by
  -- skipping the proof
  sorry

end tiles_difference_eighth_sixth_l183_183798


namespace AF_passes_through_incenter_l183_183145

variables {A B C D E F L M N P Q : Point} [EuclideanGeometry] [Triangle A B C]
          (mid_BC_L : Midpoint L B C) (mid_CA_M : Midpoint M C A) (mid_AB_N : Midpoint N A B)
          (D_on_BC : OnLine D B C) (E_on_AB : OnLine E A B)
          (AD_perimeter_bisects : Bisects AD (perimeter (Triangle A B C)))
          (CE_perimeter_bisects : Bisects CE (perimeter (Triangle A B C)))
          (P_symmetric : Symmetric P D L) (Q_symmetric : Symmetric Q E N)
          (intersect_PQ_LM_F : IntersectsAt PQ LM F)
          (AB_greater_AC : AB > AC)

theorem AF_passes_through_incenter : (PassesThrough A F (Incenter (Triangle A B C))) :=
by 
  sorry

end AF_passes_through_incenter_l183_183145


namespace correctFinishingOrder_l183_183717

-- Define the types of athletes
inductive Athlete
| A | B | C | D | E

-- Define the type for predictions
def Prediction := Athlete → Nat

-- Define the first prediction
def firstPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 1
    | Athlete.B => 2
    | Athlete.C => 3
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the second prediction
def secondPrediction : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 4
    | Athlete.C => 1
    | Athlete.D => 5
    | Athlete.E => 2

-- Define the actual finishing positions
noncomputable def actualFinishingPositions : Prediction :=
  fun x =>
    match x with
    | Athlete.A => 3
    | Athlete.B => 2
    | Athlete.C => 1
    | Athlete.D => 4
    | Athlete.E => 5

-- Define the correctness conditions for the predictions
def correctPositions (actual : Prediction) (prediction : Prediction) (k : Nat) : Prop :=
  (Set.card {x : Athlete | actual x = prediction x} = k)

-- The proof statement
theorem correctFinishingOrder :
  correctPositions actualFinishingPositions firstPrediction 3 ∧
  correctPositions actualFinishingPositions secondPrediction 2 :=
sorry

end correctFinishingOrder_l183_183717


namespace line_points_satisfy_equation_l183_183282

theorem line_points_satisfy_equation (x_2 y_3 : ℝ) 
  (h_slope : ∃ k : ℝ, k = 2) 
  (h_P1 : ∃ P1 : ℝ × ℝ, P1 = (3, 5)) 
  (h_P2 : ∃ P2 : ℝ × ℝ, P2 = (x_2, 7)) 
  (h_P3 : ∃ P3 : ℝ × ℝ, P3 = (-1, y_3)) 
  (h_line : ∀ (x y : ℝ), y - 5 = 2 * (x - 3) ↔ 2 * x - y - 1 = 0) :
  x_2 = 4 ∧ y_3 = -3 :=
sorry

end line_points_satisfy_equation_l183_183282


namespace rectangle_area_l183_183799

theorem rectangle_area :
  ∃ (a b : ℕ), a ≠ b ∧ Even a ∧ (a * b = 3 * (2 * a + 2 * b)) ∧ (a * b = 162) :=
by
  sorry

end rectangle_area_l183_183799


namespace intersection_eq_T_l183_183031

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183031


namespace plot_area_is_nine_hectares_l183_183388

-- Definition of the dimensions of the plot
def length := 450
def width := 200

-- Definition of conversion factor from square meters to hectares
def sqMetersPerHectare := 10000

-- Calculated area in hectares
def area_hectares := (length * width) / sqMetersPerHectare

-- Theorem statement: prove that the area in hectares is 9
theorem plot_area_is_nine_hectares : area_hectares = 9 := 
by
  sorry

end plot_area_is_nine_hectares_l183_183388


namespace contrapositive_of_zero_implication_l183_183196

theorem contrapositive_of_zero_implication (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) → (a * b ≠ 0 → (a ≠ 0 ∧ b ≠ 0)) :=
by
  intro h
  sorry

end contrapositive_of_zero_implication_l183_183196


namespace area_of_sector_l183_183905

theorem area_of_sector (r l : ℝ) (h1 : l + 2 * r = 12) (h2 : l / r = 2) : (1 / 2) * l * r = 9 :=
by
  sorry

end area_of_sector_l183_183905


namespace correct_aggregate_insurance_amount_correct_deductible_correct_insurance_rules_l183_183877

-- Definitions of the conditions
def insurance_amount_desc : Prop := 
  "страховая сумма, которая будет уменьшаться после каждой осуществлённой выплаты."

def insurer_exemption_desc : Prop := 
  "которая представляет собой освобождение страховщика от оплаты ущерба определённого размера."

def insurance_contract_doc_desc : Prop := 
  "В качестве приложения к договору страхования сотрудник страховой компании выдал Петру Ивановичу документы, которые содержат разработанные и утверждённые страховой компанией основные положения договора страхования, которые являются обязательными для обеих сторон."

-- The missing words we need to prove as the correct insertions
def aggregate_insurance_amount : String := "агрегатная страховая сумма"
def deductible : String := "франшиза"
def insurance_rules : String := "правила страхования"

-- The statements to be proved
theorem correct_aggregate_insurance_amount (h : insurance_amount_desc) : 
  aggregate_insurance_amount = "агрегатная страховая сумма" := 
sorry

theorem correct_deductible (h : insurer_exemption_desc) : 
  deductible = "франшиза" := 
sorry

theorem correct_insurance_rules (h : insurance_contract_doc_desc) : 
  insurance_rules = "правила страхования" := 
sorry

end correct_aggregate_insurance_amount_correct_deductible_correct_insurance_rules_l183_183877


namespace smallest_palindrome_div_3_5_l183_183227

theorem smallest_palindrome_div_3_5 : ∃ n : ℕ, n = 50205 ∧ 
  (∃ a b c : ℕ, n = 5 * 10^4 + a * 10^3 + b * 10^2 + a * 10 + 5) ∧ 
  n % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 10000 ∧ 
  n < 100000 :=
by
  sorry

end smallest_palindrome_div_3_5_l183_183227


namespace machine_A_sprockets_per_hour_l183_183603

theorem machine_A_sprockets_per_hour :
  ∀ (A T : ℝ),
    (T > 0 ∧
    (∀ P Q, P = 1.1 * A ∧ Q = 330 / P ∧ Q = 330 / A + 10) →
      A = 3) := 
by
  intro A T
  intro h
  sorry

end machine_A_sprockets_per_hour_l183_183603


namespace trigonometric_identity_l183_183417

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π / 2 - θ)) / (Real.sin θ ^ 2 + Real.cos (2 * θ) + Real.cos θ ^ 2) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l183_183417


namespace miguel_run_time_before_ariana_catches_up_l183_183893

theorem miguel_run_time_before_ariana_catches_up
  (head_start : ℕ := 20)
  (ariana_speed : ℕ := 6)
  (miguel_speed : ℕ := 4)
  (head_start_distance : ℕ := miguel_speed * head_start)
  (t_catchup : ℕ := (head_start_distance) / (ariana_speed - miguel_speed))
  (total_time : ℕ := t_catchup + head_start) :
  total_time = 60 := sorry

end miguel_run_time_before_ariana_catches_up_l183_183893


namespace find_other_number_l183_183923

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l183_183923


namespace Albert_cabbage_count_l183_183984

-- Define the conditions
def rows := 12
def heads_per_row := 15

-- State the theorem
theorem Albert_cabbage_count : rows * heads_per_row = 180 := 
by sorry

end Albert_cabbage_count_l183_183984


namespace exception_to_roots_l183_183353

theorem exception_to_roots (x : ℝ) :
    ¬ (∃ x₀, (x₀ ∈ ({x | x = x} ∩ {x | x = x - 2}))) :=
by sorry

end exception_to_roots_l183_183353


namespace roots_product_l183_183125

theorem roots_product (x1 x2 : ℝ) (h : ∀ x : ℝ, x^2 - 4 * x + 1 = 0 → x = x1 ∨ x = x2) : x1 * x2 = 1 :=
sorry

end roots_product_l183_183125


namespace total_cost_correct_l183_183396

-- Defining the conditions
def charges_per_week : ℕ := 3
def weeks_per_year : ℕ := 52
def cost_per_charge : ℝ := 0.78

-- Defining the total cost proof statement
theorem total_cost_correct : (charges_per_week * weeks_per_year : ℝ) * cost_per_charge = 121.68 :=
by
  sorry

end total_cost_correct_l183_183396


namespace total_sections_l183_183768

theorem total_sections (boys girls : ℕ) (h_boys : boys = 408) (h_girls : girls = 240) :
  let gcd_boys_girls := Nat.gcd boys girls
  let sections_boys := boys / gcd_boys_girls
  let sections_girls := girls / gcd_boys_girls
  sections_boys + sections_girls = 27 :=
by
  sorry

end total_sections_l183_183768


namespace max_full_marks_probability_l183_183482

-- Define the total number of mock exams
def total_mock_exams : ℕ := 20
-- Define the number of full marks scored in mock exams
def full_marks_in_mocks : ℕ := 8

-- Define the probability of event A (scoring full marks in the first test)
def P_A : ℚ := full_marks_in_mocks / total_mock_exams

-- Define the probability of not scoring full marks in the first test
def P_neg_A : ℚ := 1 - P_A

-- Define the probability of event B (scoring full marks in the second test)
def P_B : ℚ := 1 / 2

-- Define the maximum probability of scoring full marks in either the first or the second test
def max_probability : ℚ := P_A + P_neg_A * P_B

-- The main theorem conjecture
theorem max_full_marks_probability :
  max_probability = 7 / 10 :=
by
  -- Inserting placeholder to skip the proof for now
  sorry

end max_full_marks_probability_l183_183482


namespace square_of_product_of_third_sides_l183_183331

-- Given data for triangles P1 and P2
variables {a b c d : ℝ}

-- Areas of triangles P1 and P2
def area_P1_pos (a b : ℝ) : Prop := a * b / 2 = 3
def area_P2_pos (a d : ℝ) : Prop := a * d / 2 = 6

-- Condition that b = d / 2
def side_ratio (b d : ℝ) : Prop := b = d / 2

-- Pythagorean theorem applied to both triangles
def pythagorean_P1 (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def pythagorean_P2 (a d c : ℝ) : Prop := a^2 + d^2 = c^2

-- The goal is to prove (cd)^2 = 120
theorem square_of_product_of_third_sides (a b c d : ℝ)
  (h_area_P1: area_P1_pos a b) 
  (h_area_P2: area_P2_pos a d) 
  (h_side_ratio: side_ratio b d) 
  (h_pythagorean_P1: pythagorean_P1 a b c) 
  (h_pythagorean_P2: pythagorean_P2 a d c) :
  (c * d)^2 = 120 := 
sorry

end square_of_product_of_third_sides_l183_183331


namespace intersection_S_T_l183_183042

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183042


namespace least_multiple_of_25_gt_450_correct_l183_183638

def least_multiple_of_25_gt_450 : ℕ :=
  475

theorem least_multiple_of_25_gt_450_correct (n : ℕ) (h1 : 25 ∣ n) (h2 : n > 450) : n ≥ least_multiple_of_25_gt_450 :=
by
  sorry

end least_multiple_of_25_gt_450_correct_l183_183638


namespace transformed_area_l183_183901

noncomputable def area_transformation (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : (1 / 2 * ((x2 - x1) * ((3 * f x3) - (3 * f x1))) - 1 / 2 * ((x3 - x2) * ((3 * f x1) - (3 * f x2)))) = 27) : Prop :=
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5

theorem transformed_area
  (f : ℝ → ℝ) (x1 x2 x3 : ℝ)
  (h : 1 / 2 * ((x2 - x1) * (f x3 - f x1) - (x3 - x2) * (f x1 - f x2)) = 27) :
  1 / 2 * ((0.5 * x2 - 0.5 * x1) * (3 * f (2 * x3) - 3 * f (2 * x1)) - 1 / 2 * (0.5 * x3 - 0.5 * x2) * (3 * f (2 * x1) - 3 * f (2 * x2))) = 40.5 := sorry

end transformed_area_l183_183901


namespace contractor_engaged_days_l183_183243

theorem contractor_engaged_days (x y : ℕ) (earnings_per_day : ℕ) (fine_per_day : ℝ) 
    (total_earnings : ℝ) (absent_days : ℕ) 
    (h1 : earnings_per_day = 25) 
    (h2 : fine_per_day = 7.50) 
    (h3 : total_earnings = 555) 
    (h4 : absent_days = 6) 
    (h5 : total_earnings = (earnings_per_day * x : ℝ) - fine_per_day * y) 
    (h6 : y = absent_days) : 
    x = 24 := 
by
  sorry

end contractor_engaged_days_l183_183243


namespace average_speed_car_l183_183199

theorem average_speed_car (speed_first_hour ground_speed_headwind speed_second_hour : ℝ) (time_first_hour time_second_hour : ℝ) (h1 : speed_first_hour = 90) (h2 : ground_speed_headwind = 10) (h3 : speed_second_hour = 55) (h4 : time_first_hour = 1) (h5 : time_second_hour = 1) : 
(speed_first_hour + ground_speed_headwind) * time_first_hour + speed_second_hour * time_second_hour / (time_first_hour + time_second_hour) = 77.5 :=
sorry

end average_speed_car_l183_183199


namespace solution_set_eq_two_l183_183702

theorem solution_set_eq_two (m : ℝ) (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) :
  m = -1 :=
sorry

end solution_set_eq_two_l183_183702


namespace false_statement_l183_183395

theorem false_statement :
  ¬ (∀ x : ℝ, x^2 + 1 > 3 * x) = (∃ x : ℝ, x^2 + 1 ≤ 3 * x) := sorry

end false_statement_l183_183395


namespace min_distance_between_graphs_l183_183205

noncomputable def minimum_distance (a : ℝ) (h : 1 < a) : ℝ :=
  if h1 : a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a)

theorem min_distance_between_graphs (a : ℝ) (h1 : 1 < a) :
  minimum_distance a h1 = 
  if a ≤ Real.exp (1 / Real.exp 1) then 0
  else Real.sqrt 2 * (1 + Real.log (Real.log a)) / (Real.log a) :=
by
  intros
  sorry

end min_distance_between_graphs_l183_183205


namespace miracle_tree_fruit_count_l183_183653

theorem miracle_tree_fruit_count :
  ∃ (apples oranges pears : ℕ), 
  apples + oranges + pears = 30 ∧
  apples = 6 ∧ oranges = 9 ∧ pears = 15 := by
  sorry

end miracle_tree_fruit_count_l183_183653


namespace solution_to_quadratic_solution_to_cubic_l183_183836

-- Problem 1: x^2 = 4
theorem solution_to_quadratic (x : ℝ) : x^2 = 4 -> x = 2 ∨ x = -2 := by
  sorry

-- Problem 2: 64x^3 + 27 = 0
theorem solution_to_cubic (x : ℝ) : 64 * x^3 + 27 = 0 -> x = -3 / 4 := by
  sorry

end solution_to_quadratic_solution_to_cubic_l183_183836


namespace number_of_ordered_triplets_l183_183699

theorem number_of_ordered_triplets :
  ∃ count : ℕ, (∀ (a b c : ℕ), lcm a b = 1000 ∧ lcm b c = 2000 ∧ lcm c a = 2000 →
  count = 70) :=
sorry

end number_of_ordered_triplets_l183_183699


namespace phase_shift_of_sine_function_l183_183558

theorem phase_shift_of_sine_function :
  ∀ x : ℝ, y = 3 * Real.sin (3 * x + π / 4) → (∃ φ : ℝ, φ = -π / 12) :=
by sorry

end phase_shift_of_sine_function_l183_183558


namespace least_possible_coins_l183_183778

theorem least_possible_coins : 
  ∃ b : ℕ, b % 7 = 3 ∧ b % 4 = 2 ∧ ∀ n : ℕ, (n % 7 = 3 ∧ n % 4 = 2) → b ≤ n :=
sorry

end least_possible_coins_l183_183778


namespace core_temperature_calculation_l183_183939

-- Define the core temperature of the Sun, given in degrees Celsius
def T_Sun : ℝ := 19200000

-- Define the multiple factor
def factor : ℝ := 312.5

-- The expected result in scientific notation
def expected_temperature : ℝ := 6.0 * (10 ^ 9)

-- Prove that the calculated temperature is equal to the expected temperature
theorem core_temperature_calculation : (factor * T_Sun) = expected_temperature := by
  sorry

end core_temperature_calculation_l183_183939


namespace couple_tickets_sold_l183_183934

theorem couple_tickets_sold (S C : ℕ) :
  20 * S + 35 * C = 2280 ∧ S + 2 * C = 128 -> C = 56 :=
by
  intro h
  sorry

end couple_tickets_sold_l183_183934


namespace factor_expression_l183_183269

theorem factor_expression (x : ℝ) :
  84 * x ^ 5 - 210 * x ^ 9 = -42 * x ^ 5 * (5 * x ^ 4 - 2) :=
by
  sorry

end factor_expression_l183_183269


namespace final_weight_is_200_l183_183723

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end final_weight_is_200_l183_183723


namespace max_eggs_l183_183979

theorem max_eggs (x : ℕ) 
  (h1 : x < 200) 
  (h2 : x % 3 = 2) 
  (h3 : x % 4 = 3) 
  (h4 : x % 5 = 4) : 
  x = 179 := 
by
  sorry

end max_eggs_l183_183979


namespace probability_reaches_or_exceeds_6_units_at_some_point_l183_183720

def fair_coin_toss_10 : Fin 1024 → Fin 11 → ℤ := sorry

/-- The probability that Jerry reaches or exceeds 6 units in the positive direction 
at some point during the 10 tosses is 193 / 512. -/
theorem probability_reaches_or_exceeds_6_units_at_some_point :
  (∑ (i : Fin 1024), if ∃ (j : Fin 11), fair_coin_toss_10 i j ≥ 6 then 1 else 0) / 1024 = 193 / 512 :=
  sorry

end probability_reaches_or_exceeds_6_units_at_some_point_l183_183720


namespace number_difference_l183_183767

theorem number_difference (x y : ℕ) (h₁ : x + y = 41402) (h₂ : ∃ k : ℕ, x = 100 * k) (h₃ : y = x / 100) : x - y = 40590 :=
sorry

end number_difference_l183_183767


namespace largest_n_binomial_l183_183218

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l183_183218


namespace haley_picked_carrots_l183_183286

variable (H : ℕ)
variable (mom_carrots : ℕ := 38)
variable (good_carrots : ℕ := 64)
variable (bad_carrots : ℕ := 13)
variable (total_carrots : ℕ := good_carrots + bad_carrots)

theorem haley_picked_carrots : H + mom_carrots = total_carrots → H = 39 := by
  sorry

end haley_picked_carrots_l183_183286


namespace intersection_of_S_and_T_l183_183104

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183104


namespace five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l183_183656

-- Define what "5 PM" and "10 PM" mean in hours
def five_pm: ℕ := 17
def ten_pm: ℕ := 22

-- Define function for converting from PM to 24-hour time
def pm_to_hours (n: ℕ): ℕ := n + 12

-- Define the times in minutes for comparison
def time_16_40: ℕ := 16 * 60 + 40
def time_17_20: ℕ := 17 * 60 + 20

-- Define the differences in minutes
def minutes_passed (start end_: ℕ): ℕ := end_ - start

-- Prove the equivalences
theorem five_pm_is_seventeen_hours: pm_to_hours 5 = five_pm := by 
  unfold pm_to_hours
  unfold five_pm
  rfl

theorem ten_pm_is_twenty_two_hours: pm_to_hours 10 = ten_pm := by 
  unfold pm_to_hours
  unfold ten_pm
  rfl

theorem time_difference_is_forty_minutes: minutes_passed time_16_40 time_17_20 = 40 := by 
  unfold time_16_40
  unfold time_17_20
  unfold minutes_passed
  rfl

#check five_pm_is_seventeen_hours
#check ten_pm_is_twenty_two_hours
#check time_difference_is_forty_minutes

end five_pm_is_seventeen_hours_ten_pm_is_twenty_two_hours_time_difference_is_forty_minutes_l183_183656


namespace intersection_S_T_l183_183079

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183079


namespace necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l183_183137

theorem necessary_ab_given_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 4) : 
  a + b ≥ 4 :=
sorry

theorem not_sufficient_ab_given_a_b : 
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b ≥ 4 ∧ a * b < 4 :=
sorry

end necessary_ab_given_a_b_not_sufficient_ab_given_a_b_l183_183137


namespace tangent_condition_l183_183783

theorem tangent_condition (a b : ℝ) :
  (4 * a^2 + b^2 = 1) ↔ 
  ∀ x y : ℝ, (y = 2 * x + 1) → ((x^2 / a^2) + (y^2 / b^2) = 1) → (∃! y, y = 2 * x + 1 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

end tangent_condition_l183_183783


namespace expression_evaluation_l183_183512

theorem expression_evaluation :
  1 - (2 - (3 - 4 - (5 - 6))) = -1 :=
sorry

end expression_evaluation_l183_183512


namespace largest_integer_binom_l183_183221

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l183_183221


namespace uncle_ben_eggs_l183_183635

def number_of_eggs (C R N E : ℕ) :=
  let hens := C - R
  let laying_hens := hens - N
  laying_hens * E

-- Given:
-- C = 440 (Total number of chickens)
-- R = 39 (Number of roosters)
-- N = 15 (Non-laying hens)
-- E = 3 (Eggs per laying hen)
theorem uncle_ben_eggs : number_of_eggs 440 39 15 3 = 1158 := by
  unfold number_of_eggs
  calc
    403 - 15 * 3 : sorry

end uncle_ben_eggs_l183_183635


namespace log_tan_ratio_l183_183565

noncomputable def sin_add (α β : ℝ) : ℝ := Real.sin (α + β)
noncomputable def sin_sub (α β : ℝ) : ℝ := Real.sin (α - β)
noncomputable def tan_ratio (α β : ℝ) : ℝ := Real.tan α / Real.tan β

theorem log_tan_ratio (α β : ℝ)
  (h1 : sin_add α β = 1 / 2)
  (h2 : sin_sub α β = 1 / 3) :
  Real.logb 5 (tan_ratio α β) = 1 := by
sorry

end log_tan_ratio_l183_183565


namespace symmetric_point_in_third_quadrant_l183_183593

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the function to find the symmetric point about the y-axis
def symmetric_about_y (P : Point) : Point :=
  Point.mk (-P.x) P.y

-- Define the original point P
def P : Point := { x := 3, y := -2 }

-- Define the symmetric point P' about the y-axis
def P' : Point := symmetric_about_y P

-- Define a condition to determine if a point is in the third quadrant
def is_in_third_quadrant (P : Point) : Prop :=
  P.x < 0 ∧ P.y < 0

-- The theorem stating that the symmetric point of P about the y-axis is in the third quadrant
theorem symmetric_point_in_third_quadrant : is_in_third_quadrant P' :=
  by
  sorry

end symmetric_point_in_third_quadrant_l183_183593


namespace gcd_2197_2208_is_1_l183_183368

def gcd_2197_2208 : ℕ := Nat.gcd 2197 2208

theorem gcd_2197_2208_is_1 : gcd_2197_2208 = 1 :=
by
  sorry

end gcd_2197_2208_is_1_l183_183368


namespace total_students_l183_183589

-- Define the conditions
variables (S : ℕ) -- total number of students
variable (h1 : (3/5 : ℚ) * S + (1/5 : ℚ) * S + 10 = S)

-- State the theorem
theorem total_students (HS : S = 50) : 3 / 5 * S + 1 / 5 * S + 10 = S := by
  -- Here we declare the proof is to be filled in later.
  sorry

end total_students_l183_183589


namespace reduced_price_tickets_first_week_l183_183550

theorem reduced_price_tickets_first_week (total_tickets sold_at_full_price : ℕ) 
  (condition1 : total_tickets = 25200) 
  (condition2 : sold_at_full_price = 16500)
  (condition3 : ∃ R, total_tickets = R + 5 * R) : 
  ∃ R : ℕ, R = 3300 := 
by sorry

end reduced_price_tickets_first_week_l183_183550


namespace hyperbola_is_given_equation_l183_183129

noncomputable def hyperbola_equation : Prop :=
  ∃ a b : ℝ, 
    (a > 0 ∧ b > 0) ∧ 
    (4^2 = a^2 + b^2) ∧ 
    (a = b) ∧ 
    (∀ x y : ℝ, (x^2 / 8 - y^2 / 8 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1))

theorem hyperbola_is_given_equation : hyperbola_equation :=
sorry

end hyperbola_is_given_equation_l183_183129


namespace intersection_S_T_eq_T_l183_183060

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183060


namespace min_value_fraction_l183_183792

variable (a b : ℝ)
variable (h1 : 2 * a - 2 * b + 2 = 0) -- This corresponds to a + b = 1 based on the given center (-1, 2)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_fraction (h1 : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (4 / a) + (1 / b) ≥ 9 :=
  sorry

end min_value_fraction_l183_183792


namespace ineq_medians_triangle_l183_183278

theorem ineq_medians_triangle (a b c s_a s_b s_c : ℝ)
  (h_mediana : s_a = 1 / 2 * Real.sqrt (2 * b^2 + 2 * c^2 - a^2))
  (h_medianb : s_b = 1 / 2 * Real.sqrt (2 * a^2 + 2 * c^2 - b^2))
  (h_medianc : s_c = 1 / 2 * Real.sqrt (2 * a^2 + 2 * b^2 - c^2))
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c > s_a + s_b + s_c ∧ s_a + s_b + s_c > (3 / 4) * (a + b + c) := 
sorry

end ineq_medians_triangle_l183_183278


namespace intersection_S_T_eq_T_l183_183064

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183064


namespace solve_quadratic_equation_l183_183354

theorem solve_quadratic_equation : ∀ x : ℝ, x * (x - 14) = 0 ↔ x = 0 ∨ x = 14 :=
by
  sorry

end solve_quadratic_equation_l183_183354


namespace largest_n_binom_identity_l183_183225

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l183_183225


namespace intersection_S_T_eq_T_l183_183090

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183090


namespace correct_calculation_l183_183231

theorem correct_calculation : 
  ¬(2 * Real.sqrt 3 + 3 * Real.sqrt 2 = 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  ¬(5 * Real.sqrt 3 * 5 * Real.sqrt 2 = 5 * Real.sqrt 6) ∧
  ¬(Real.sqrt (4 + 1 / 2) = 2 * Real.sqrt (1 / 2)) :=
by {
  -- Using the conditions to prove the correct option B
  sorry
}

end correct_calculation_l183_183231


namespace intersection_of_S_and_T_l183_183108

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183108


namespace isosceles_triangle_relationship_l183_183252

theorem isosceles_triangle_relationship (x y : ℝ) (h1 : 2 * x + y = 30) (h2 : 7.5 < x) (h3 : x < 15) : 
  y = 30 - 2 * x :=
  by sorry

end isosceles_triangle_relationship_l183_183252


namespace cube_edge_length_l183_183485

-- Define the edge length 'a'
variable (a : ℝ)

-- Given conditions: 6a^2 = 24
theorem cube_edge_length (h : 6 * a^2 = 24) : a = 2 :=
by {
  -- The actual proof would go here, but we use sorry to skip it as per instructions.
  sorry
}

end cube_edge_length_l183_183485


namespace no_adjacent_standing_probability_l183_183902

noncomputable def probability_no_adjacent_standing : ℚ := 
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := 123
  favorable_outcomes / total_outcomes

theorem no_adjacent_standing_probability :
  probability_no_adjacent_standing = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l183_183902


namespace points_below_line_l183_183562

theorem points_below_line (d q x1 x2 y1 y2 : ℝ) 
  (h1 : 2 = 1 + 3 * d)
  (h2 : x1 = 1 + d)
  (h3 : x2 = x1 + d)
  (h4 : 2 = q ^ 3)
  (h5 : y1 = q)
  (h6 : y2 = q ^ 2) :
  x1 > y1 ∧ x2 > y2 :=
by {
  sorry
}

end points_below_line_l183_183562


namespace lollipops_left_for_becky_l183_183676
-- Import the Mathlib library

-- Define the conditions as given in the problem
def lemon_lollipops : ℕ := 75
def peppermint_lollipops : ℕ := 210
def watermelon_lollipops : ℕ := 6
def marshmallow_lollipops : ℕ := 504
def friends : ℕ := 13

-- Total number of lollipops
def total_lollipops : ℕ := lemon_lollipops + peppermint_lollipops + watermelon_lollipops + marshmallow_lollipops

-- Statement to prove that the remainder after distributing the total lollipops among friends is 2
theorem lollipops_left_for_becky : total_lollipops % friends = 2 := by
  -- Proof goes here
  sorry

end lollipops_left_for_becky_l183_183676


namespace trigonometric_identity_l183_183560

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.sin (2 * x + Real.pi / 5) = Real.sqrt 3 / 3) : 
  Real.sin (4 * Real.pi / 5 - 2 * x) + Real.sin (3 * Real.pi / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 :=
by
  sorry

end trigonometric_identity_l183_183560


namespace intersection_S_T_l183_183052

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183052


namespace orange_juice_fraction_l183_183937

theorem orange_juice_fraction :
  let capacity1 := 500
  let capacity2 := 600
  let fraction1 := (1/4 : ℚ)
  let fraction2 := (1/3 : ℚ)
  let juice1 := capacity1 * fraction1
  let juice2 := capacity2 * fraction2
  let total_juice := juice1 + juice2
  let total_volume := capacity1 + capacity2
  (total_juice / total_volume = (13/44 : ℚ)) := sorry

end orange_juice_fraction_l183_183937


namespace kelsey_more_than_ekon_l183_183771

theorem kelsey_more_than_ekon :
  ∃ (K E U : ℕ), (K = 160) ∧ (E = U - 17) ∧ (K + E + U = 411) ∧ (K - E = 43) :=
by
  sorry

end kelsey_more_than_ekon_l183_183771


namespace tangent_series_identity_l183_183751

noncomputable def series_tangent (x : ℝ) : ℝ := ∑' n, (1 / (2 ^ n)) * Real.tan (x / (2 ^ n))

theorem tangent_series_identity (x : ℝ) : 
  (1 / x) - (1 / Real.tan x) = series_tangent x := 
sorry

end tangent_series_identity_l183_183751


namespace find_n_l183_183308

open Nat

theorem find_n (n : ℕ) (h : n ≥ 6) (h_eq : binomial n 5 * 3^5 = binomial n 6 * 3^6) : n = 7 := 
sorry

end find_n_l183_183308


namespace intersection_S_T_l183_183016

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183016


namespace intersection_S_T_eq_T_l183_183114

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183114


namespace intersection_S_T_l183_183041

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183041


namespace sum_of_x_y_l183_183693

theorem sum_of_x_y (x y : ℝ) (h1 : 3 * x + 2 * y = 10) (h2 : 2 * x + 3 * y = 5) : x + y = 3 := 
by
  sorry

end sum_of_x_y_l183_183693


namespace intersection_S_T_l183_183078

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183078


namespace range_of_B_l183_183434

theorem range_of_B (A : ℝ × ℝ) (hA : A = (1, 2)) (h : 2 * A.1 - B * A.2 + 3 ≥ 0) : B ≤ 2.5 :=
by sorry

end range_of_B_l183_183434


namespace intersection_S_T_l183_183049

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183049


namespace closest_integer_to_a2013_l183_183766

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 100 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) + (1 / a (n + 1))

theorem closest_integer_to_a2013 (a : ℕ → ℝ) (h : seq a) : abs (a 2013 - 118) < 0.5 :=
sorry

end closest_integer_to_a2013_l183_183766


namespace problem_l183_183844

theorem problem (a b : ℤ) (h : (2 * a + b) ^ 2 + |b - 2| = 0) : (-a - b) ^ 2014 = 1 := 
by
  sorry

end problem_l183_183844


namespace solve_for_y_l183_183621

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l183_183621


namespace maximum_M_for_right_triangle_l183_183281

theorem maximum_M_for_right_triangle (a b c : ℝ) (h1 : a ≤ b) (h2 : b < c) (h3 : a^2 + b^2 = c^2) :
  (1 / a + 1 / b + 1 / c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) :=
sorry

end maximum_M_for_right_triangle_l183_183281


namespace total_truck_loads_needed_l183_183533

noncomputable def truck_loads_of_material : ℝ :=
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5 -- log is the natural logarithm in Lean
  sand + dirt + cement + gravel

theorem total_truck_loads_needed : truck_loads_of_material = 1.8401374808985008 := by
  sorry

end total_truck_loads_needed_l183_183533


namespace distance_between_vertices_l183_183689

/-
Problem statement:
Prove that the distance between the vertices of the hyperbola
\(\frac{x^2}{144} - \frac{y^2}{64} = 1\) is 24.
-/

/-- 
We define the given hyperbola equation:
\frac{x^2}{144} - \frac{y^2}{64} = 1
-/
def hyperbola (x y : ℝ) : Prop := x^2 / 144 - y^2 / 64 = 1

/--
We establish that the distance between the vertices of the hyperbola is 24.
-/
theorem distance_between_vertices : 
  (∀ x y : ℝ, hyperbola x y → dist (12, 0) (-12, 0) = 24) :=
by
  sorry

end distance_between_vertices_l183_183689


namespace solution_set_of_inequality_eq_l183_183628

noncomputable def inequality_solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

theorem solution_set_of_inequality_eq :
  {x : ℝ | (2 * x) / (x - 1) < 1} = inequality_solution_set := by
  sorry

end solution_set_of_inequality_eq_l183_183628


namespace intersection_eq_T_l183_183040

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183040


namespace intersection_S_T_l183_183044

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183044


namespace shirt_cost_l183_183516

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 66) : S = 12 :=
by
  sorry

end shirt_cost_l183_183516


namespace digit_sum_solution_l183_183317

def S (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_solution : S (S (S (S (2017 ^ 2017)))) = 1 := 
by
  sorry

end digit_sum_solution_l183_183317


namespace intersection_S_T_eq_T_l183_183092

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183092


namespace actual_positions_correct_l183_183712

def athletes := {A, B, C, D, E}
def p1 : athletes → ℕ
| A := 1
| B := 2
| C := 3
| D := 4
| E := 5

def p2 : athletes → ℕ
| C := 1
| E := 2
| A := 3
| B := 4
| D := 5

def actual_positions : athletes → ℕ
| A := 3
| B := 2
| C := 1
| D := 4
| E := 5

theorem actual_positions_correct :
  (∑ a in athletes, ite (actual_positions a = p1 a) 1 0 = 3) ∧ 
  (∑ a in athletes, ite (actual_positions a = p2 a) 1 0 = 2) :=
by
  sorry

end actual_positions_correct_l183_183712


namespace percentage_increase_l183_183795

theorem percentage_increase (original_value : ℕ) (percentage_increase : ℚ) :  
  original_value = 1200 → 
  percentage_increase = 0.40 →
  original_value * (1 + percentage_increase) = 1680 :=
by
  intros h1 h2
  sorry

end percentage_increase_l183_183795


namespace largest_number_sum13_product36_l183_183819

-- helper definitions for sum and product of digits
def sum_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.sum
def mul_digits (n : ℕ) : ℕ := Nat.digits 10 n |> List.foldr (· * ·) 1

theorem largest_number_sum13_product36 : 
  ∃ n : ℕ, sum_digits n = 13 ∧ mul_digits n = 36 ∧ ∀ m : ℕ, sum_digits m = 13 ∧ mul_digits m = 36 → m ≤ n :=
sorry

end largest_number_sum13_product36_l183_183819


namespace unique_integer_solution_l183_183687

theorem unique_integer_solution (m n : ℤ) :
  (m + n)^4 = m^2 * n^2 + m^2 + n^2 + 6 * m * n ↔ m = 0 ∧ n = 0 :=
by
  sorry

end unique_integer_solution_l183_183687


namespace complex_number_value_l183_183567

theorem complex_number_value (i : ℂ) (h : i^2 = -1) : i^13 * (1 + i) = -1 + i :=
by
  sorry

end complex_number_value_l183_183567


namespace saved_money_is_30_l183_183991

def week_payout : ℕ := 5 * 3
def total_payout (weeks: ℕ) : ℕ := weeks * week_payout
def shoes_cost : ℕ := 120
def remaining_weeks : ℕ := 6
def remaining_earnings : ℕ := total_payout remaining_weeks
def saved_money : ℕ := shoes_cost - remaining_earnings

theorem saved_money_is_30 : saved_money = 30 := by
  -- Proof steps go here
  sorry

end saved_money_is_30_l183_183991


namespace replace_90_percent_in_3_days_cannot_replace_all_banknotes_l183_183311

-- Define constants and conditions
def total_old_banknotes : ℕ := 3628800
def daily_cost : ℕ := 90000
def major_repair_cost : ℕ := 700000
def max_daily_print_after_repair : ℕ := 1000000
def budget_limit : ℕ := 1000000

-- Define the day's print capability function (before repair)
def daily_print (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  if num_days = 1 then banknotes_remaining / 2
  else (banknotes_remaining / (num_days + 1))

-- Define the budget calculation before repair
def print_costs (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  daily_cost * num_days

-- Lean theorem to be stated proving that 90% of the banknotes can be replaced within 3 days
theorem replace_90_percent_in_3_days :
  ∃ (days : ℕ) (banknotes_replaced : ℕ), days = 3 ∧ banknotes_replaced = 3265920 ∧ print_costs days total_old_banknotes ≤ budget_limit :=
sorry

-- Lean theorem to be stated proving that not all banknotes can be replaced within the given budget
theorem cannot_replace_all_banknotes :
  ∀ banknotes_replaced cost : ℕ,
  banknotes_replaced < total_old_banknotes ∧ cost ≤ budget_limit →
  banknotes_replaced + (total_old_banknotes / (4 + 1)) < total_old_banknotes :=
sorry

end replace_90_percent_in_3_days_cannot_replace_all_banknotes_l183_183311


namespace jake_buys_packages_l183_183878

theorem jake_buys_packages:
  ∀ (pkg_weight cost_per_pound total_paid : ℕ),
    pkg_weight = 2 →
    cost_per_pound = 4 →
    total_paid = 24 →
    (total_paid / (pkg_weight * cost_per_pound)) = 3 :=
by
  intros pkg_weight cost_per_pound total_paid hw_cp ht
  sorry

end jake_buys_packages_l183_183878


namespace number_of_valid_M_l183_183894

def base_4_representation (M : ℕ) :=
  let c_3 := (M / 256) % 4
  let c_2 := (M / 64) % 4
  let c_1 := (M / 16) % 4
  let c_0 := M % 4
  (256 * c_3) + (64 * c_2) + (16 * c_1) + (4 * c_0)

def base_7_representation (M : ℕ) :=
  let d_3 := (M / 343) % 7
  let d_2 := (M / 49) % 7
  let d_1 := (M / 7) % 7
  let d_0 := M % 7
  (343 * d_3) + (49 * d_2) + (7 * d_1) + d_0

def valid_M (M T : ℕ) :=
  1000 ≤ M ∧ M < 10000 ∧ 
  T = base_4_representation M + base_7_representation M ∧ 
  (T % 100) = ((3 * M) % 100)

theorem number_of_valid_M : 
  ∃ n : ℕ, n = 81 ∧ ∀ M T, valid_M M T → n = (81 : ℕ) :=
sorry

end number_of_valid_M_l183_183894


namespace sum_of_consecutive_numbers_LCM_168_l183_183913

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l183_183913


namespace intersection_S_T_eq_T_l183_183001

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183001


namespace gcd_16_12_eq_4_l183_183204

theorem gcd_16_12_eq_4 : Nat.gcd 16 12 = 4 := by
  -- Skipping proof using sorry
  sorry

end gcd_16_12_eq_4_l183_183204


namespace sum_of_possible_values_l183_183288

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 24) : 
  ∃ x1 x2 : ℝ, (x1 + 3) * (x1 - 4) = 24 ∧ (x2 + 3) * (x2 - 4) = 24 ∧ x1 + x2 = 1 := 
by
  sorry

end sum_of_possible_values_l183_183288


namespace movies_watched_total_l183_183362

theorem movies_watched_total :
  ∀ (Timothy2009 Theresa2009 Timothy2010 Theresa2010 total : ℕ),
    Timothy2009 = 24 →
    Timothy2010 = Timothy2009 + 7 →
    Theresa2010 = 2 * Timothy2010 →
    Theresa2009 = Timothy2009 / 2 →
    total = Timothy2009 + Timothy2010 + Theresa2009 + Theresa2010 →
    total = 129 :=
by
  intros Timothy2009 Theresa2009 Timothy2010 Theresa2010 total
  sorry

end movies_watched_total_l183_183362


namespace a_eq_zero_l183_183574

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, ax + 2 ≠ 0) : a = 0 := by
  sorry

end a_eq_zero_l183_183574


namespace positive_integer_solutions_x_plus_2y_eq_5_l183_183835

theorem positive_integer_solutions_x_plus_2y_eq_5 :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (x + 2 * y = 5) ∧ ((x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 1)) :=
by
  sorry

end positive_integer_solutions_x_plus_2y_eq_5_l183_183835


namespace speed_of_first_part_l183_183245

theorem speed_of_first_part (v : ℝ) (h1 : v > 0)
  (h_total_distance : 50 = 25 + 25)
  (h_average_speed : 44 = 50 / ((25 / v) + (25 / 33))) :
  v = 66 :=
by sorry

end speed_of_first_part_l183_183245


namespace largest_n_for_binom_equality_l183_183216

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l183_183216


namespace remainder_pow_2023_l183_183775

theorem remainder_pow_2023 (a b : ℕ) (h : b = 2023) : (3 ^ b) % 11 = 5 :=
by
  sorry

end remainder_pow_2023_l183_183775


namespace cubes_not_touching_foil_l183_183632

-- Define the variables for length, width, height, and total cubes
variables (l w h : ℕ)

-- Conditions extracted from the problem
def width_is_twice_length : Prop := w = 2 * l
def width_is_twice_height : Prop := w = 2 * h
def foil_covered_prism_width : Prop := w + 2 = 10

-- The proof statement
theorem cubes_not_touching_foil (l w h : ℕ) 
  (h1 : width_is_twice_length l w) 
  (h2 : width_is_twice_height w h) 
  (h3 : foil_covered_prism_width w) : 
  l * w * h = 128 := 
by sorry

end cubes_not_touching_foil_l183_183632


namespace min_chocolates_for_most_l183_183998

theorem min_chocolates_for_most (a b c d : ℕ) (h : a < b ∧ b < c ∧ c < d)
  (h_sum : a + b + c + d = 50) : d ≥ 14 := sorry

end min_chocolates_for_most_l183_183998


namespace intersection_S_T_eq_T_l183_183122

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183122


namespace clock_angle_320_l183_183944

theorem clock_angle_320 :
  let initial_angle_3_00 := 90
  let minute_hand_movement_per_minute := 6
  let hour_hand_movement_per_minute := 0.5
  let angle_change_per_minute := minute_hand_movement_per_minute - hour_hand_movement_per_minute
  let total_minutes := 20
  let angle_change := angle_change_per_minute * total_minutes
  let final_angle := initial_angle_3_00 + angle_change
  let smaller_angle := if final_angle > 180 then 360 - final_angle else final_angle
  in smaller_angle = 160 :=
by
  sorry

end clock_angle_320_l183_183944


namespace gcd_of_cubic_sum_and_linear_is_one_l183_183412

theorem gcd_of_cubic_sum_and_linear_is_one (n : ℕ) (h : n > 27) : Nat.gcd (n^3 + 8) (n + 3) = 1 :=
sorry

end gcd_of_cubic_sum_and_linear_is_one_l183_183412


namespace Inez_initial_money_l183_183157

theorem Inez_initial_money (X : ℝ) (h : X - (X / 2 + 50) = 25) : X = 150 :=
by
  sorry

end Inez_initial_money_l183_183157


namespace profit_calculation_l183_183451

-- Define the initial conditions
def initial_cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def price_decrease_effect : ℝ := 4
def daily_profit_target : ℝ := 13600
def minimum_selling_price : ℝ := 150

-- Define the function relationship of daily sales volume with respect to x
def sales_volume (x : ℝ) : ℝ := initial_sales_volume + price_decrease_effect * x

-- Define the selling price
def selling_price (x : ℝ) : ℝ := initial_selling_price - x

-- Define the profit function
def profit (x : ℝ) : ℝ := (selling_price x - initial_cost_price) * sales_volume x

theorem profit_calculation (x : ℝ) (hx : selling_price x ≥ minimum_selling_price) :
  profit x = daily_profit_target ↔ selling_price x = 185 := by
  sorry

end profit_calculation_l183_183451


namespace cars_overtake_distance_l183_183503

def speed_red_car : ℝ := 30
def speed_black_car : ℝ := 50
def time_to_overtake : ℝ := 1
def distance_between_cars : ℝ := 20

theorem cars_overtake_distance :
  (speed_black_car - speed_red_car) * time_to_overtake = distance_between_cars :=
by sorry

end cars_overtake_distance_l183_183503


namespace infinite_seq_contains_all_nat_l183_183251

-- Define the sequence a
def sequence_a (a : ℕ → ℕ) : Prop :=
  (∀ i, 0 ≤ a i ∧ a i ≤ i) ∧
  (∀ k, (∑ i in Finset.range (k + 1), Nat.choose k (a i)) = 2^k)

-- Lean statement of the problem
theorem infinite_seq_contains_all_nat (a : ℕ → ℕ) (H : sequence_a a) :
  ∀ (N : ℕ), ∃ (i : ℕ), a i = N :=
by
  sorry

end infinite_seq_contains_all_nat_l183_183251


namespace intersection_of_S_and_T_l183_183098

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183098


namespace find_common_difference_l183_183307

variable (a₁ d : ℝ)

theorem find_common_difference
  (h1 : a₁ + (a₁ + 6 * d) = 22)
  (h2 : (a₁ + 3 * d) + (a₁ + 9 * d) = 40) :
  d = 3 := by
  sorry

end find_common_difference_l183_183307


namespace largest_common_number_in_range_l183_183806

theorem largest_common_number_in_range (n1 d1 n2 d2 : ℕ) (h1 : n1 = 2) (h2 : d1 = 4) (h3 : n2 = 5) (h4 : d2 = 6) :
  ∃ k : ℕ, k ≤ 200 ∧ (∀ n3 : ℕ, n3 = n1 + d1 * k) ∧ (∀ n4 : ℕ, n4 = n2 + d2 * k) ∧ n3 = 190 ∧ n4 = 190 := 
by {
  sorry
}

end largest_common_number_in_range_l183_183806


namespace gasoline_needed_l183_183476

variable (distance_trip : ℕ) (fuel_per_trip_distance : ℕ) (trip_distance : ℕ) (fuel_needed : ℕ)

theorem gasoline_needed (h1 : distance_trip = 140)
                       (h2 : fuel_per_trip_distance = 10)
                       (h3 : trip_distance = 70)
                       (h4 : fuel_needed = 20) :
  (fuel_per_trip_distance * (distance_trip / trip_distance)) = fuel_needed :=
by sorry

end gasoline_needed_l183_183476


namespace Northton_time_capsule_depth_l183_183183

theorem Northton_time_capsule_depth:
  ∀ (d_southton d_northton : ℝ),
  d_southton = 15 →
  d_northton = (4 * d_southton) - 12 →
  d_northton = 48 :=
by
  intros d_southton d_northton h_southton h_northton
  rw [h_southton] at h_northton
  rw [← h_northton]
  sorry

end Northton_time_capsule_depth_l183_183183


namespace intersection_S_T_eq_T_l183_183111

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183111


namespace intersection_S_T_l183_183073

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183073


namespace smallest_positive_period_intervals_monotonic_increase_max_min_values_l183_183847

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem smallest_positive_period (x : ℝ) : (f (x + π)) = f x :=
sorry

theorem intervals_monotonic_increase (k : ℤ) (x : ℝ) : (k * π - π/3) ≤ x ∧ x ≤ (k * π + π/6) → ∃ a b : ℝ, a < b ∧ ∀ x : ℝ, (a ≤ x ∧ x ≤ b) →
  (f x < f (x + 1)) :=
sorry

theorem max_min_values (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ π/4) : (∃ y : ℝ, y = max (f 0) (f (π/6)) ∧ y = 1) ∧ (∃ z : ℝ, z = min (f 0) (f (π/6)) ∧ z = 0) :=
sorry

end smallest_positive_period_intervals_monotonic_increase_max_min_values_l183_183847


namespace div_pow_sub_one_l183_183649

theorem div_pow_sub_one (n : ℕ) (h : n > 1) : (n - 1) ^ 2 ∣ n ^ (n - 1) - 1 :=
sorry

end div_pow_sub_one_l183_183649


namespace intersection_S_T_eq_T_l183_183062

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183062


namespace cube_sphere_surface_area_l183_183840

open Real

noncomputable def cube_edge_length := 1
noncomputable def cube_space_diagonal := sqrt 3
noncomputable def sphere_radius := cube_space_diagonal / 2
noncomputable def sphere_surface_area := 4 * π * (sphere_radius ^ 2)

theorem cube_sphere_surface_area :
  sphere_surface_area = 3 * π :=
by
  sorry

end cube_sphere_surface_area_l183_183840


namespace find_operation_l183_183127

theorem find_operation (a b : Int) (h : a + b = 0) : (7 + (-7) = 0) := 
by
  sorry

end find_operation_l183_183127


namespace roots_sum_product_l183_183442

theorem roots_sum_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h : ∀ x : ℝ, x^2 - p*x - 2*q = 0) :
  (p + q = p) ∧ (p * q = -2*q) :=
by
  sorry

end roots_sum_product_l183_183442


namespace no_three_distinct_zeros_l183_183703

noncomputable def f (a x : ℝ) : ℝ := Real.exp (2 * x) + a * Real.exp x - (a + 2) * x

theorem no_three_distinct_zeros (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0) → False :=
by
  sorry

end no_three_distinct_zeros_l183_183703


namespace total_spent_on_toys_l183_183536

-- Definition of the costs
def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

-- The theorem to prove the total amount spent on toys
theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 :=
by sorry

end total_spent_on_toys_l183_183536


namespace palm_meadows_total_beds_l183_183811

theorem palm_meadows_total_beds :
  ∃ t : Nat, t = 31 → 
    (∀ r1 r2 r3 : Nat, r1 = 13 → r2 = 8 → r3 = r1 - r2 → 
      t = (r2 * 2 + r3 * 3)) :=
by
  sorry

end palm_meadows_total_beds_l183_183811


namespace regular_polygon_with_side_PD_exists_l183_183929

theorem regular_polygon_with_side_PD_exists
    (circle : Type)
    (inscribed : circle → Prop)
    (A B C D E F G H P Q : circle)
    (regular_octagon : inscribed A ∧ inscribed B ∧ inscribed C ∧ inscribed D ∧ inscribed E ∧ inscribed F ∧ inscribed G ∧ inscribed H ∧ 
    (∃ R : ℝ, (∀ i j : fin 8, i ≠ j → dist (octagon_points i) (octagon_points j) = R)))
    (equilateral_triangle : inscribed A ∧ inscribed P ∧ inscribed Q ∧ 
    ∃ S : ℝ, (∀ i j : fin 3, dist (triangle_points i) (triangle_points j) = S) ∧ 
    ∀ x : circle, x = P ∨ x = Q ∨ x = A)
    (between_P_C_D : inscribed P ∧ inscribed C ∧ inscribed D ∧ 
    (∃ T : ℝ, dist P C = T ∧ dist P D = T))
    : ∃ n : ℕ, n = 24 := 
sorry

end regular_polygon_with_side_PD_exists_l183_183929


namespace malcom_cards_left_l183_183538

theorem malcom_cards_left (brandon_cards : ℕ) (h1 : brandon_cards = 20) (h2 : ∀ malcom_cards : ℕ, malcom_cards = brandon_cards + 8) (h3 : ∀ mark_cards : ℕ, mark_cards = (malcom_cards / 2)) : 
  malcom_cards - mark_cards = 14 := 
by 
  let malcom_cards := 28
  let mark_cards := (malcom_cards / 2)
  have h4 : mark_cards = 14, from rfl
  show malcom_cards - mark_cards = 14, from sorry

end malcom_cards_left_l183_183538


namespace no_descending_multiple_of_111_l183_183873

theorem no_descending_multiple_of_111 :
  ¬ ∃ (n : ℕ), (∀ (i j : ℕ), (i < j ∧ (n / 10^i % 10) < (n / 10^j % 10)) ∨ (i = j)) ∧ 111 ∣ n :=
by
  sorry

end no_descending_multiple_of_111_l183_183873


namespace cricket_team_age_difference_l183_183340

theorem cricket_team_age_difference :
  ∀ (captain_age : ℕ) (keeper_age : ℕ) (team_size : ℕ) (team_average_age : ℕ) (remaining_size : ℕ),
  captain_age = 28 →
  keeper_age = captain_age + 3 →
  team_size = 11 →
  team_average_age = 25 →
  remaining_size = team_size - 2 →
  (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 24 →
  team_average_age - (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 1 :=
by
  intros captain_age keeper_age team_size team_average_age remaining_size h1 h2 h3 h4 h5 h6
  sorry

end cricket_team_age_difference_l183_183340


namespace find_height_l183_183200

-- Definitions from the problem conditions
def Area : ℕ := 442
def width : ℕ := 7
def length : ℕ := 8

-- The statement to prove
theorem find_height (h : ℕ) (H : 2 * length * width + 2 * length * h + 2 * width * h = Area) : h = 11 := 
by
  sorry

end find_height_l183_183200


namespace range_omega_l183_183849

noncomputable def f (ω x : ℝ) := Real.cos (ω * x + Real.pi / 6)

theorem range_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi → -1 ≤ f ω x ∧ f ω x ≤ Real.sqrt 3 / 2) →
  ω ∈ Set.Icc (5 / 6) (5 / 3) :=
  sorry

end range_omega_l183_183849


namespace intersection_S_T_eq_T_l183_183084

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183084


namespace order_options_count_l183_183708

/-- Define the number of options for each category -/
def drinks : ℕ := 3
def salads : ℕ := 2
def pizzas : ℕ := 5

/-- The theorem statement that we aim to prove -/
theorem order_options_count : drinks * salads * pizzas = 30 :=
by
  sorry -- Proof is skipped as instructed

end order_options_count_l183_183708


namespace three_pow_2023_mod_eleven_l183_183777

theorem three_pow_2023_mod_eleven :
  (3 ^ 2023) % 11 = 5 :=
sorry

end three_pow_2023_mod_eleven_l183_183777


namespace fraction_numerator_l183_183189

theorem fraction_numerator (x : ℚ) : 
  (∃ y : ℚ, y = 4 * x + 4 ∧ x / y = 3 / 7) → x = -12 / 5 :=
by
  sorry

end fraction_numerator_l183_183189


namespace ratio_x_y_l183_183870

-- Definitions based on conditions
variables (a b c x y : ℝ) 

-- Conditions
def right_triangle (a b c : ℝ) := (a^2 + b^2 = c^2)
def a_b_ratio (a b : ℝ) := (a / b = 2 / 5)
def segments_ratio (a b c x y : ℝ) := (x = a^2 / c) ∧ (y = b^2 / c)
def perpendicular_division (x y a b : ℝ) := ((a^2 / x) = c) ∧ ((b^2 / y) = c)

-- The proof statement we need
theorem ratio_x_y : 
  ∀ (a b c x y : ℝ),
    right_triangle a b c → 
    a_b_ratio a b → 
    segments_ratio a b c x y → 
    (x / y = 4 / 25) :=
by sorry

end ratio_x_y_l183_183870


namespace factor_polynomial_l183_183542

theorem factor_polynomial (x : ℝ) : 
  (20 * x^3 + 100 * x - 10) - (-3 * x^3 + 5 * x - 15) = 5 * (23 * x^3 + 19 * x + 1) := 
by 
  -- Proof can be filled in here
  sorry

end factor_polynomial_l183_183542


namespace integer_solutions_of_xyz_equation_l183_183553

/--
  Find all integer solutions of the equation \( x + y + z = xyz \).
  The integer solutions are expected to be:
  \[
  (1, 2, 3), (2, 1, 3), (3, 1, 2), (3, 2, 1), (1, 3, 2), (2, 3, 1), (-a, 0, a) \text{ for } (a : ℤ).
  \]
-/
theorem integer_solutions_of_xyz_equation (x y z : ℤ) :
    x + y + z = x * y * z ↔ 
    (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
    (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) ∨ 
    (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 3 ∧ z = 1) ∨ 
    ∃ a : ℤ, (x = -a ∧ y = 0 ∧ z = a) := by
  sorry


end integer_solutions_of_xyz_equation_l183_183553


namespace smaller_angle_at_3_20_correct_l183_183957

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l183_183957


namespace solve_for_y_l183_183620

theorem solve_for_y (y : ℚ) : 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 :=
by
  intro h
  sorry

end solve_for_y_l183_183620


namespace part_a_part_b_l183_183420

variable {α β γ δ AB CD : ℝ}
variable {A B C D : Point}
variable {A_obtuse B_obtuse : Prop}
variable {α_gt_δ β_gt_γ : Prop}

-- Definition of a convex quadrilateral
def convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Conditions for part (a)
axiom angle_A_obtuse : A_obtuse
axiom angle_B_obtuse : B_obtuse

-- Conditions for part (b)
axiom angle_α_gt_δ : α_gt_δ
axiom angle_β_gt_γ : β_gt_γ

-- Part (a) statement: Given angles A and B are obtuse, AB ≤ CD
theorem part_a {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_A_obtuse : A_obtuse) (h_B_obtuse : B_obtuse) : AB ≤ CD :=
sorry

-- Part (b) statement: Given angle A > angle D and angle B > angle C, AB < CD
theorem part_b {A B C D : Point} (h_convex : convex_quadrilateral A B C D) 
    (h_angle_α_gt_δ : α_gt_δ) (h_angle_β_gt_γ : β_gt_γ) : AB < CD :=
sorry

end part_a_part_b_l183_183420


namespace intersection_of_S_and_T_l183_183110

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183110


namespace parallel_lines_condition_l183_183654

theorem parallel_lines_condition {a : ℝ} :
  (∀ x y : ℝ, a * x + 2 * y + 3 * a = 0) ∧ (∀ x y : ℝ, 3 * x + (a - 1) * y = a - 7) ↔ a = 3 :=
by
  sorry

end parallel_lines_condition_l183_183654


namespace orange_pear_difference_l183_183728

theorem orange_pear_difference :
  let O1 := 37
  let O2 := 10
  let O3 := 2 * O2
  let P1 := 30
  let P2 := 3 * P1
  let P3 := P2 + 4
  (O1 + O2 + O3 - (P1 + P2 + P3)) = -147 := 
by
  sorry

end orange_pear_difference_l183_183728


namespace problem1_problem2_l183_183240

-- Problem 1
theorem problem1 : (-2) ^ 2 + (Real.sqrt 2 - 1) ^ 0 - 1 = 4 := by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) (A : ℝ) (B : ℝ) (h1 : A = a - 1) (h2 : B = -a + 3) (h3 : A > B) : a > 2 := by
  sorry

end problem1_problem2_l183_183240


namespace no_descending_multiple_of_111_l183_183871

-- Hypotheses
def digits_descending (n : ℕ) : Prop := 
  ∀ i j, i < j → (n.digits.get i) > (n.digits.get j)

def is_multiple_of_111 (n : ℕ) : Prop := 
  n % 111 = 0

-- Conclusion
theorem no_descending_multiple_of_111 :
  ∀ n : ℕ, digits_descending n ∧ is_multiple_of_111 n → false :=
by sorry

end no_descending_multiple_of_111_l183_183871


namespace symmetric_line_eq_l183_183841

theorem symmetric_line_eq (x y : ℝ) :
  (∀ x y, 2 * x - y + 1 = 0 → y = -x) → (∀ x y, x - 2 * y + 1 = 0) :=
by sorry

end symmetric_line_eq_l183_183841


namespace sector_area_ratio_l183_183742

theorem sector_area_ratio (angle_AOE angle_FOB : ℝ) (h1 : angle_AOE = 40) (h2 : angle_FOB = 60) : 
  (180 - angle_AOE - angle_FOB) / 360 = 2 / 9 :=
by
  sorry

end sector_area_ratio_l183_183742


namespace maximize_expression_l183_183887

theorem maximize_expression
  (a b c : ℝ)
  (h1 : a ≥ 0)
  (h2 : b ≥ 0)
  (h3 : c ≥ 0)
  (h_sum : a + b + c = 3) :
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 729 / 432 := 
sorry

end maximize_expression_l183_183887


namespace intersection_S_T_eq_T_l183_183094

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183094


namespace molecular_weight_3_moles_ascorbic_acid_l183_183941

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_formula_ascorbic_acid : List (ℝ × ℕ) :=
  [(atomic_weight_C, 6), (atomic_weight_H, 8), (atomic_weight_O, 6)]

def molecular_weight (formula : List (ℝ × ℕ)) : ℝ :=
  formula.foldl (λ acc (aw, count) => acc + aw * count) 0.0

def weight_of_moles (mw : ℝ) (moles : ℕ) : ℝ :=
  mw * moles

theorem molecular_weight_3_moles_ascorbic_acid :
  weight_of_moles (molecular_weight molecular_formula_ascorbic_acid) 3 = 528.372 :=
by
  sorry

end molecular_weight_3_moles_ascorbic_acid_l183_183941


namespace number_of_internal_cubes_l183_183631

theorem number_of_internal_cubes :
  ∀ l w h : ℕ,
    w = 2 * l →
    w = 2 * h →
    w = 6 →
    l * w * h = 54 :=
by
  intros l w h h_w2l h_w2h h_w
  rw [h_w2l, h_w2h] at h_w
  have l_value := (h_w.symm ▸ rfl : l = 3)
  have h_value := (h_w.symm ▸ rfl : h = 3)
  rw [l_value, h_w, h_value]
  norm_num

end number_of_internal_cubes_l183_183631


namespace salt_solution_percentage_l183_183522

theorem salt_solution_percentage
  (x : ℝ)
  (y : ℝ)
  (h1 : 600 + y = 1000)
  (h2 : 600 * x + y * 0.12 = 1000 * 0.084) :
  x = 0.06 :=
by
  -- The proof goes here.
  sorry

end salt_solution_percentage_l183_183522


namespace area_of_region_AGF_l183_183898

theorem area_of_region_AGF 
  (ABCD_area : ℝ)
  (hABCD_area : ABCD_area = 160)
  (E F G : ℝ)
  (hE_midpoint : E = (A + B) / 2)
  (hF_midpoint : F = (C + D) / 2)
  (EF_divides : EF_area = ABCD_area / 2)
  (hEF_midpoint : G = (E + F) / 2)
  (AG_divides_upper : AG_area = EF_area / 2) :
  AGF_area = 40 := 
sorry

end area_of_region_AGF_l183_183898


namespace henry_games_l183_183158

theorem henry_games {N H : ℕ} (hN : N = 7) (hH : H = 4 * N) 
    (h_final: H - 6 = 4 * (N + 6)) : H = 58 :=
by
  -- Proof would be inserted here, but skipped using sorry
  sorry

end henry_games_l183_183158


namespace june_walked_miles_l183_183328

theorem june_walked_miles
  (step_counter_reset : ℕ)
  (resets_per_year : ℕ)
  (final_steps : ℕ)
  (steps_per_mile : ℕ)
  (h1 : step_counter_reset = 100000)
  (h2 : resets_per_year = 52)
  (h3 : final_steps = 30000)
  (h4 : steps_per_mile = 2000) :
  (resets_per_year * step_counter_reset + final_steps) / steps_per_mile = 2615 := 
by 
  sorry

end june_walked_miles_l183_183328


namespace intersection_S_T_eq_T_l183_183058

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183058


namespace diff_not_equal_l183_183884

variable (A B : Set ℕ)

def diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem diff_not_equal (A B : Set ℕ) :
  A ≠ ∅ ∧ B ≠ ∅ → (diff A B ≠ diff B A) :=
by
  sorry

end diff_not_equal_l183_183884


namespace contradiction_assumption_l183_183309

theorem contradiction_assumption (a b : ℝ) (h : |a - 1| * |b - 1| = 0) : ¬ (a ≠ 1 ∧ b ≠ 1) :=
  sorry

end contradiction_assumption_l183_183309


namespace valid_selling_price_l183_183453

-- Define the initial conditions
def cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def sales_increase_per_dollar_decrease : ℝ := 4
def max_profit : ℝ := 13600
def min_selling_price : ℝ := 150

-- Define x as the price reduction per item
variable (x : ℝ)

-- Define the function relationship of the daily sales volume y with respect to x
def sales_volume (x : ℝ) := 100 + 4 * x

-- Define the selling price based on the price reduction
def selling_price (x : ℝ) := 200 - x

-- Calculate the profit based on the selling price and sales volume
def profit (x : ℝ) := (selling_price x - cost_price) * (sales_volume x)

-- Lean theorem statement to prove the given conditions lead to the valid selling price
theorem valid_selling_price (x : ℝ) 
  (h1 : profit x = 13600)
  (h2 : selling_price x ≥ 150) : 
  selling_price x = 185 :=
sorry

end valid_selling_price_l183_183453


namespace solve_for_x_l183_183614

theorem solve_for_x (x : ℂ) (h : 5 - 2 * I * x = 4 - 5 * I * x) : x = I / 3 :=
by
  sorry

end solve_for_x_l183_183614


namespace bothStoresSaleSameDate_l183_183382

-- Define the conditions
def isBookstoreSaleDay (d : ℕ) : Prop := d % 4 = 0
def isShoeStoreSaleDay (d : ℕ) : Prop := ∃ k : ℕ, d = 5 + 7 * k
def isJulyDay (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

-- Define the problem statement
theorem bothStoresSaleSameDate : 
  (∃ d1 d2 : ℕ, isJulyDay d1 ∧ isBookstoreSaleDay d1 ∧ isShoeStoreSaleDay d1 ∧
                 isJulyDay d2 ∧ isBookstoreSaleDay d2 ∧ isShoeStoreSaleDay d2 ∧ d1 ≠ d2) :=
sorry

end bothStoresSaleSameDate_l183_183382


namespace mrs_sheridan_fish_distribution_l183_183740

theorem mrs_sheridan_fish_distribution :
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium
  fish_in_large_aquarium = 225 :=
by {
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium

  have : fish_in_large_aquarium = 225 := by sorry
  exact this
}

end mrs_sheridan_fish_distribution_l183_183740


namespace scientific_notation_of_400000_l183_183471

theorem scientific_notation_of_400000 :
  (400000: ℝ) = 4 * 10^5 :=
by 
  sorry

end scientific_notation_of_400000_l183_183471


namespace probability_ratio_l183_183405

open_locale big_operators

-- Parameters representing conditions
def n_balls : ℕ := 20
def n_bins : ℕ := 5
def p : ℚ := 
  ((fintype.card {s : finset (fin n_bins) // s.card = 1 ∧ ∀ (x : ℕ), x ∈ s → x = 3})
  * (fintype.card {s : finset (fin n_bins) // s.card = 1 ∧ ∀ (x : ℕ), x ∈ s → x = 5})
  * (fintype.card {s : finset (fin n_bins) // s.card = 3 ∧ ∀ (x : ℕ), x ∈ s → x = 4}))
  / fintype.card {f : fin n_bins → fin n_balls}
def q : ℚ := 
  fintype.card {f : fin n_bins → fin n_balls // ∀ i, f i = 4}
  / fintype.card {f : fin n_bins → fin n_balls}

theorem probability_ratio :
  (p / q) = 16 :=
by sorry

end probability_ratio_l183_183405


namespace intersection_S_T_l183_183015

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183015


namespace total_rainfall_in_2011_l183_183295

-- Define the given conditions
def avg_monthly_rainfall_2010 : ℝ := 36.8
def increase_2011 : ℝ := 3.5

-- Define the resulting average monthly rainfall in 2011
def avg_monthly_rainfall_2011 : ℝ := avg_monthly_rainfall_2010 + increase_2011

-- Calculate the total annual rainfall
def total_rainfall_2011 : ℝ := avg_monthly_rainfall_2011 * 12

-- State the proof problem
theorem total_rainfall_in_2011 :
  total_rainfall_2011 = 483.6 := by
  sorry

end total_rainfall_in_2011_l183_183295


namespace Malcom_cards_after_giving_away_half_l183_183540

def Brandon_cards : ℕ := 20
def Malcom_initial_cards : ℕ := Brandon_cards + 8
def Malcom_remaining_cards : ℕ := Malcom_initial_cards - (Malcom_initial_cards / 2)

theorem Malcom_cards_after_giving_away_half :
  Malcom_remaining_cards = 14 :=
by
  sorry

end Malcom_cards_after_giving_away_half_l183_183540


namespace lines_are_parallel_l183_183927

-- Definitions of the conditions
variable (θ a p : Real)
def line1 := θ = a
def line2 := p * Real.sin (θ - a) = 1

-- The proof problem: Prove the two lines are parallel
theorem lines_are_parallel (h1 : line1 θ a) (h2 : line2 θ a p) : False :=
by
  sorry

end lines_are_parallel_l183_183927


namespace mod_equiv_1_l183_183623

theorem mod_equiv_1 : (179 * 933 / 7) % 50 = 1 := by
  sorry

end mod_equiv_1_l183_183623


namespace correct_prediction_l183_183714

-- Declare the permutation type
def Permutation := (ℕ × ℕ × ℕ × ℕ × ℕ)

-- Declare the actual finish positions
def actual_positions : Permutation := (3, 2, 1, 4, 5)

-- Declare conditions
theorem correct_prediction 
  (P1 : Permutation) (P2 : Permutation) 
  (hP1 : P1 = (1, 2, 3, 4, 5))
  (hP2 : P2 = (1, 2, 3, 4, 5))
  (hP1_correct : ∃! i ∈ (1..5), (λ i, nth (P1.to_list) (i - 1) == nth (actual_positions.to_list))  = 3)
  (hP2_correct : ∃! i ∈ (1..5), (λ i, nth (P2.to_list) (i - 1) == nth (actual_positions.to_list))  = 2)
  : actual_positions = (3, 2, 1, 4, 5) 
:= sorry

end correct_prediction_l183_183714


namespace correct_option_l183_183235

theorem correct_option :
  (2 * Real.sqrt 5) + (3 * Real.sqrt 5) = 5 * Real.sqrt 5 :=
by sorry

end correct_option_l183_183235


namespace intersection_eq_T_l183_183030

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183030


namespace probability_two_females_l183_183591

theorem probability_two_females (total_contestants females males choose_two : ℕ) 
  (h_total : total_contestants = 7) (h_females : females = 4) (h_males : males = 3) 
  (h_choose_two : choose_two = 2) :
  (females.choose choose_two : ℚ) / (total_contestants.choose choose_two : ℚ) = 2 / 7 :=
by
  rw [h_total, h_females, h_males, h_choose_two]
  norm_num
  sorry

end probability_two_females_l183_183591


namespace percentage_increase_l183_183447

-- Define the initial and final prices as constants
def P_inicial : ℝ := 5.00
def P_final : ℝ := 5.55

-- Define the percentage increase proof
theorem percentage_increase : ((P_final - P_inicial) / P_inicial) * 100 = 11 := 
by
  sorry

end percentage_increase_l183_183447


namespace intersection_S_T_l183_183024

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183024


namespace oranges_total_and_avg_cost_l183_183986

-- Conditions
def rate_4_for_15 := 15 / 4
def rate_7_for_25 := 25 / 7
def purchase_groups (n : ℕ) : ℕ := 7 * n
def total_cost (n : ℕ) : ℕ := 25 * n

-- Problem statement: Prove if buying 28 oranges, the total cost is 100 cents and the average cost per orange is 25/7 cents.
theorem oranges_total_and_avg_cost (n : ℕ) (h : purchase_groups n = 28) : 
  total_cost n = 100 ∧ (total_cost n) / 28 = (25 / 7 : ℚ) :=
by
  sorry

end oranges_total_and_avg_cost_l183_183986


namespace max_gcd_dn_l183_183257

def a (n : ℕ) := 101 + n^2

def d (n : ℕ) := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_dn : ∃ n : ℕ, ∀ m : ℕ, d m ≤ 3 := sorry

end max_gcd_dn_l183_183257


namespace total_boys_fraction_of_girls_l183_183491

theorem total_boys_fraction_of_girls
  (n : ℕ)
  (b1 g1 b2 g2 : ℕ)
  (h_equal_students : b1 + g1 = b2 + g2)
  (h_ratio_class1 : b1 / g1 = 2 / 3)
  (h_ratio_class2: b2 / g2 = 4 / 5) :
  ((b1 + b2) / (g1 + g2) = 19 / 26) :=
by sorry

end total_boys_fraction_of_girls_l183_183491


namespace expand_binomial_trinomial_l183_183267

theorem expand_binomial_trinomial (x y z : ℝ) :
  (x + 12) * (3 * y + 4 * z + 15) = 3 * x * y + 4 * x * z + 15 * x + 36 * y + 48 * z + 180 :=
by sorry

end expand_binomial_trinomial_l183_183267


namespace max_x_minus_2y_l183_183692

open Real

theorem max_x_minus_2y (x y : ℝ) (h : (x^2) / 16 + (y^2) / 9 = 1) : 
  ∃ t : ℝ, t = 2 * sqrt 13 ∧ x - 2 * y = t := 
sorry

end max_x_minus_2y_l183_183692


namespace necessarily_positive_l183_183330

theorem necessarily_positive (x y z : ℝ) (hx : -1 < x ∧ x < 1) 
                      (hy : -1 < y ∧ y < 0) 
                      (hz : 1 < z ∧ z < 2) : 
    y + z > 0 := 
by
  sorry

end necessarily_positive_l183_183330


namespace ratio_accepted_rejected_l183_183448

-- Definitions for the conditions given
def eggs_per_day : ℕ := 400
def ratio_accepted_to_rejected : ℕ × ℕ := (96, 4)
def additional_accepted_eggs : ℕ := 12

/-- The ratio of accepted eggs to rejected eggs on that particular day is 99:1. -/
theorem ratio_accepted_rejected (a r : ℕ) (h1 : ratio_accepted_to_rejected = (a, r)) 
  (h2 : (a + r) * (eggs_per_day / (a + r)) = eggs_per_day) 
  (h3 : additional_accepted_eggs = 12) :
  (a + additional_accepted_eggs) / r = 99 :=
  sorry

end ratio_accepted_rejected_l183_183448


namespace num_of_chords_l183_183684

theorem num_of_chords (n : ℕ) (h : n = 8) : (n.choose 2) = 28 :=
by
  -- Proof of this theorem will be here
  sorry

end num_of_chords_l183_183684


namespace find_m_for_eccentric_ellipse_l183_183626

theorem find_m_for_eccentric_ellipse (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/5 + (y^2)/m = 1) ∧
  (∀ e : ℝ, e = (Real.sqrt 10)/5) → 
  (m = 25/3 ∨ m = 3) := sorry

end find_m_for_eccentric_ellipse_l183_183626


namespace distance_between_parallel_lines_l183_183190

/-- Given two parallel lines y=2x and y=2x+5, the distance between them is √5. -/
theorem distance_between_parallel_lines :
  let A := -2
  let B := 1
  let C1 := 0
  let C2 := -5
  let distance := (|C2 - C1|: ℝ) / Real.sqrt (A ^ 2 + B ^ 2)
  distance = Real.sqrt 5 := by
  -- Assuming calculations as done in the original solution
  sorry

end distance_between_parallel_lines_l183_183190


namespace age_difference_l183_183337

theorem age_difference (x y : ℕ) (h1 : 3 * x + 4 * x = 42) (h2 : 18 - y = (24 - y) / 2) : 
  y = 12 :=
  sorry

end age_difference_l183_183337


namespace find_f1_l183_183466

theorem find_f1 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + (-x) ^ 2 = -(f x + x ^ 2))
  (h2 : ∀ x, f (-x) + 2 ^ (-x) = f x + 2 ^ x) :
  f 1 = -7 / 4 := by
sorry

end find_f1_l183_183466


namespace radius_of_2007_l183_183627

-- Define the conditions
def given_condition (n : ℕ) (r : ℕ → ℝ) : Prop :=
  r 1 = 1 ∧ (∀ i, 1 ≤ i ∧ i < n → r (i + 1) = 3 * r i)

-- State the theorem we want to prove
theorem radius_of_2007 (r : ℕ → ℝ) : given_condition 2007 r → r 2007 = 3^2006 :=
by
  sorry -- Proof placeholder

end radius_of_2007_l183_183627


namespace x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l183_183579

variable {x : ℝ}

theorem x_cubed_lt_one_of_x_lt_one (hx : x < 1) : x^3 < 1 :=
sorry

theorem abs_x_lt_one_of_x_lt_one (hx : x < 1) : |x| < 1 :=
sorry

end x_cubed_lt_one_of_x_lt_one_abs_x_lt_one_of_x_lt_one_l183_183579


namespace symmetric_angles_l183_183293

theorem symmetric_angles (α β : ℝ) (k : ℤ) (h : α + β = 2 * k * Real.pi) : α = 2 * k * Real.pi - β :=
by
  sorry

end symmetric_angles_l183_183293


namespace negation_of_exists_leq_zero_l183_183194

theorem negation_of_exists_leq_zero (x : ℝ) : ¬(∃ x ≥ 1, 2^x ≤ 0) ↔ ∀ x ≥ 1, 2^x > 0 :=
by
  sorry

end negation_of_exists_leq_zero_l183_183194


namespace current_ratio_of_employees_l183_183711

-- Definitions for the number of current male employees and the ratio if 3 more men are hired
variables (M : ℕ) (F : ℕ)
variables (hM : M = 189)
variables (ratio_hired : (M + 3) / F = 8 / 9)

-- Conclusion we want to prove
theorem current_ratio_of_employees (M F : ℕ) (hM : M = 189) (ratio_hired : (M + 3) / F = 8 / 9) : 
  M / F = 7 / 8 :=
sorry

end current_ratio_of_employees_l183_183711


namespace strategy_probabilities_l183_183867

noncomputable def P1 : ℚ := 1 / 3
noncomputable def P2 : ℚ := 1 / 2
noncomputable def P3 : ℚ := 2 / 3

theorem strategy_probabilities :
  (P1 < P2) ∧
  (P1 < P3) ∧
  (2 * P1 = P3) := by
  sorry

end strategy_probabilities_l183_183867


namespace house_number_is_fourteen_l183_183790

theorem house_number_is_fourteen (a b c n : ℕ) (h1 : a * b * c = 40) (h2 : a + b + c = n) (h3 : 
  ∃ (a b c : ℕ), a * b * c = 40 ∧ (a = 1 ∧ b = 5 ∧ c = 8) ∨ (a = 2 ∧ b = 2 ∧ c = 10) ∧ n = 14) :
  n = 14 :=
sorry

end house_number_is_fourteen_l183_183790


namespace age_ratio_in_two_years_l183_183658

theorem age_ratio_in_two_years :
  ∀ (B M : ℕ), B = 10 → M = B + 12 → (M + 2) / (B + 2) = 2 := by
  intros B M hB hM
  sorry

end age_ratio_in_two_years_l183_183658


namespace prime_squared_mod_six_l183_183374

theorem prime_squared_mod_six (p : ℕ) (hp1 : p > 5) (hp2 : Nat.Prime p) : (p ^ 2) % 6 = 1 :=
sorry

end prime_squared_mod_six_l183_183374


namespace intersection_S_T_eq_T_l183_183006

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183006


namespace Malcom_cards_after_giving_away_half_l183_183539

def Brandon_cards : ℕ := 20
def Malcom_initial_cards : ℕ := Brandon_cards + 8
def Malcom_remaining_cards : ℕ := Malcom_initial_cards - (Malcom_initial_cards / 2)

theorem Malcom_cards_after_giving_away_half :
  Malcom_remaining_cards = 14 :=
by
  sorry

end Malcom_cards_after_giving_away_half_l183_183539


namespace intersection_S_T_l183_183026

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183026


namespace f_inequality_l183_183857

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l183_183857


namespace chair_capacity_l183_183590

theorem chair_capacity
  (total_chairs : ℕ)
  (total_board_members : ℕ)
  (not_occupied_fraction : ℚ)
  (occupied_people_per_chair : ℕ)
  (attending_board_members : ℕ)
  (total_chairs_eq : total_chairs = 40)
  (not_occupied_fraction_eq : not_occupied_fraction = 2/5)
  (occupied_people_per_chair_eq : occupied_people_per_chair = 2)
  (attending_board_members_eq : attending_board_members = 48)
  : total_board_members = 48 := 
by
  sorry

end chair_capacity_l183_183590


namespace cos4_x_minus_sin4_x_l183_183862

theorem cos4_x_minus_sin4_x (x : ℝ) (h : x = π / 12) : (Real.cos x) ^ 4 - (Real.sin x) ^ 4 = (Real.sqrt 3) / 2 := by
  sorry

end cos4_x_minus_sin4_x_l183_183862


namespace intersection_S_T_l183_183069

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183069


namespace ab_multiple_of_7_2010_l183_183745

theorem ab_multiple_of_7_2010 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : 7 ^ 2009 ∣ a^2 + b^2) : 7 ^ 2010 ∣ a * b :=
by
  sorry

end ab_multiple_of_7_2010_l183_183745


namespace binomial_expectation_l183_183501

noncomputable def E (X : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ :=
∑ k in Finset.range (n+1), k * (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem binomial_expectation (n : ℕ) (p : ℝ) (h : p = 0.3) (h2 : n = 8) :
  E (λ k, k * (n.choose k : ℝ) * p^k * (1 - p)^(n - k)) n p = 2.4 := by
  rw [h, h2]
  sorry

end binomial_expectation_l183_183501


namespace volume_tetrahedron_BE_GH_eq_l183_183155

noncomputable def volume_tetrahedron (a b c : ℝ) : ℝ :=
  (volume_of_tetrahedron ABD BE .perpendicular AC)
  * sqrt(3) * b^3/(12 * (a + b) * (b + c))

theorem volume_tetrahedron_BE_GH_eq (a b c : ℝ) :
  (A D = a) → (B E = b) → (C F = c) → (A B = A C = B C = 1) →
  volume_tetrahedron a b c = sqrt(3) * b^3 / (12 * (a + b) * (b + c)) :=
begin
  sorry
end

end volume_tetrahedron_BE_GH_eq_l183_183155


namespace combined_volume_of_all_cubes_l183_183602

/-- Lily has 4 cubes each with side length 3, Mark has 3 cubes each with side length 4,
    and Zoe has 2 cubes each with side length 5. Prove that the combined volume of all
    the cubes is 550. -/
theorem combined_volume_of_all_cubes 
  (lily_cubes : ℕ := 4) (lily_side_length : ℕ := 3)
  (mark_cubes : ℕ := 3) (mark_side_length : ℕ := 4)
  (zoe_cubes : ℕ := 2) (zoe_side_length : ℕ := 5) :
  (lily_cubes * lily_side_length ^ 3) + 
  (mark_cubes * mark_side_length ^ 3) + 
  (zoe_cubes * zoe_side_length ^ 3) = 550 :=
by
  have lily_volume : ℕ := lily_cubes * lily_side_length ^ 3
  have mark_volume : ℕ := mark_cubes * mark_side_length ^ 3
  have zoe_volume : ℕ := zoe_cubes * zoe_side_length ^ 3
  have total_volume : ℕ := lily_volume + mark_volume + zoe_volume
  sorry

end combined_volume_of_all_cubes_l183_183602


namespace point_in_fourth_quadrant_l183_183306

def point_x := 3
def point_y := -4

def first_quadrant (x y : Int) := x > 0 ∧ y > 0
def second_quadrant (x y : Int) := x < 0 ∧ y > 0
def third_quadrant (x y : Int) := x < 0 ∧ y < 0
def fourth_quadrant (x y : Int) := x > 0 ∧ y < 0

theorem point_in_fourth_quadrant : fourth_quadrant point_x point_y :=
by
  simp [point_x, point_y, fourth_quadrant]
  sorry

end point_in_fourth_quadrant_l183_183306


namespace palm_meadows_total_beds_l183_183812

theorem palm_meadows_total_beds :
  ∃ t : Nat, t = 31 → 
    (∀ r1 r2 r3 : Nat, r1 = 13 → r2 = 8 → r3 = r1 - r2 → 
      t = (r2 * 2 + r3 * 3)) :=
by
  sorry

end palm_meadows_total_beds_l183_183812


namespace smallest_positive_integer_g_l183_183648

theorem smallest_positive_integer_g (g : ℕ) (h_pos : g > 0) (h_square : ∃ k : ℕ, 3150 * g = k^2) : g = 14 := 
  sorry

end smallest_positive_integer_g_l183_183648


namespace g_g_g_g_2_eq_16_l183_183546

def g (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem g_g_g_g_2_eq_16 : g (g (g (g 2))) = 16 := by
  sorry

end g_g_g_g_2_eq_16_l183_183546


namespace calculate_49_squared_l183_183935

theorem calculate_49_squared : 
  ∀ (a b : ℕ), a = 50 → b = 2 → (a - b)^2 = a^2 - 2 * a * b + b^2 → (49^2 = 50^2 - 196) :=
by
  intro a b h1 h2 h3
  sorry

end calculate_49_squared_l183_183935


namespace brian_cards_after_waine_takes_l183_183813

-- Define the conditions
def brian_initial_cards : ℕ := 76
def wayne_takes_away : ℕ := 59

-- Define the expected result
def brian_remaining_cards : ℕ := 17

-- The statement of the proof problem
theorem brian_cards_after_waine_takes : brian_initial_cards - wayne_takes_away = brian_remaining_cards := 
by 
-- the proof would be provided here 
sorry

end brian_cards_after_waine_takes_l183_183813


namespace find_other_number_l183_183921

theorem find_other_number (a b : ℕ) (h1 : Nat.gcd a b = 24) (h2 : Nat.lcm a b = 5040) (h3 : a = 240) : b = 504 :=
by {
  sorry
}

end find_other_number_l183_183921


namespace neg_proposition_equiv_l183_183704

theorem neg_proposition_equiv (p : Prop) : (¬ (∃ n : ℕ, 2^n > 1000)) = (∀ n : ℕ, 2^n ≤ 1000) :=
by
  sorry

end neg_proposition_equiv_l183_183704


namespace coefficient_of_x9_in_expansion_l183_183366

-- Definitions as given in the problem
def binomial_expansion_coeff (n k : ℕ) (a b : ℤ) : ℤ :=
  (Nat.choose n k) * a^(n - k) * b^k

-- Mathematically equivalent statement in Lean 4
theorem coefficient_of_x9_in_expansion : binomial_expansion_coeff 10 9 (-2) 1 = -20 :=
by
  sorry

end coefficient_of_x9_in_expansion_l183_183366


namespace sum_of_terms_l183_183440

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sum_of_terms (h : ∀ n, S n = n^2) : a 5 + a 6 + a 7 = 33 :=
by
  sorry

end sum_of_terms_l183_183440


namespace intersection_eq_T_l183_183037

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183037


namespace simplify_fraction_l183_183685

theorem simplify_fraction (num denom : ℚ) (h_num: num = (3/7 + 5/8)) (h_denom: denom = (5/12 + 2/3)) :
  (num / denom) = (177/182) := 
  sorry

end simplify_fraction_l183_183685


namespace intersection_S_T_l183_183082

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183082


namespace stacy_faster_than_heather_l183_183754

-- Definitions for the conditions
def distance : ℝ := 40
def heather_rate : ℝ := 5
def heather_distance : ℝ := 17.090909090909093
def heather_delay : ℝ := 0.4
def stacy_distance : ℝ := distance - heather_distance
def stacy_rate (S : ℝ) (T : ℝ) : Prop := S * T = stacy_distance
def heather_time (T : ℝ) : ℝ := T - heather_delay
def heather_walk_eq (T : ℝ) : Prop := heather_rate * heather_time T = heather_distance

-- The proof problem statement
theorem stacy_faster_than_heather :
  ∃ (S T : ℝ), stacy_rate S T ∧ heather_walk_eq T ∧ (S - heather_rate = 1) :=
by
  sorry

end stacy_faster_than_heather_l183_183754


namespace hyperbola_range_l183_183342

theorem hyperbola_range (m : ℝ) : (∃ x y : ℝ, (x^2 / (2 + m) + y^2 / (m + 1) = 1)) → (-2 < m ∧ m < -1) :=
by
  sorry

end hyperbola_range_l183_183342


namespace cannot_form_shape_B_l183_183513

-- Define the given pieces
def pieces : List (List (Nat × Nat)) :=
  [ [(1, 1)],
    [(1, 2)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 3)],
    [(1, 3)] ]

-- Define shape B requirement
def shapeB : List (Nat × Nat) := [(1, 6)]

theorem cannot_form_shape_B :
  ¬ (∃ (combinations : List (List (Nat × Nat))), combinations ⊆ pieces ∧ 
     (List.foldr (λ x acc => acc + x) 0 (combinations.map (List.foldr (λ y acc => acc + (y.1 * y.2)) 0)) = 6)) :=
sorry

end cannot_form_shape_B_l183_183513


namespace Q_div_P_l183_183763

theorem Q_div_P (P Q : ℚ) (h : ∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
  P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x * (x + 3) * (x - 5))) :
  Q / P = 1 / 3 :=
by
  sorry

end Q_div_P_l183_183763


namespace expand_expression_l183_183268

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 :=
by sorry

end expand_expression_l183_183268


namespace problem_solution_l183_183276

theorem problem_solution
  (k : ℝ)
  (y : ℝ → ℝ)
  (quadratic_fn : ∀ x, y x = (k + 2) * x^(k^2 + k - 4))
  (increase_for_neg_x : ∀ x : ℝ, x < 0 → y (x + 1) > y x) :
  k = -3 ∧ (∀ m n : ℝ, -2 ≤ m ∧ m ≤ 1 → y m = n → -4 ≤ n ∧ n ≤ 0) := 
sorry

end problem_solution_l183_183276


namespace final_price_difference_l183_183992

noncomputable def OP : ℝ := 78.2 / 0.85
noncomputable def IP : ℝ := 78.2 + 0.25 * 78.2
noncomputable def DP : ℝ := 97.75 - 0.10 * 97.75
noncomputable def FP : ℝ := 87.975 + 0.0725 * 87.975

theorem final_price_difference : OP - FP = -2.3531875 := 
by sorry

end final_price_difference_l183_183992


namespace area_inside_rectangle_outside_circles_is_4_l183_183180

-- Specify the problem in Lean 4
theorem area_inside_rectangle_outside_circles_is_4 :
  let CD := 3
  let DA := 5
  let radius_A := 1
  let radius_B := 2
  let radius_C := 3
  let area_rectangle := CD * DA
  let area_circles := (radius_A^2 + radius_B^2 + radius_C^2) * Real.pi / 4
  abs (area_rectangle - area_circles - 4) < 1 :=
by
  repeat { sorry }

end area_inside_rectangle_outside_circles_is_4_l183_183180


namespace correct_average_l183_183904

theorem correct_average 
  (n : ℕ) (initial_average : ℚ) (wrong_number : ℚ) (correct_number : ℚ) (wrong_average : ℚ)
  (h_n : n = 10) 
  (h_initial : initial_average = 14) 
  (h_wrong_number : wrong_number = 26) 
  (h_correct_number : correct_number = 36) 
  (h_wrong_average : wrong_average = 14) : 
  (initial_average * n - wrong_number + correct_number) / n = 15 := 
by
  sorry

end correct_average_l183_183904


namespace asymptotes_of_hyperbola_l183_183343

-- Definitions for the hyperbola and the asymptotes
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1
def asymptote_equation (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = - (Real.sqrt 2 / 2) * x

-- The theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) (h : hyperbola_equation x y) :
  asymptote_equation x y :=
sorry

end asymptotes_of_hyperbola_l183_183343


namespace kaleb_final_score_l183_183151

variable (score_first_half : ℝ) (bonus_special_q : ℝ) (bonus_streak : ℝ) (score_second_half : ℝ) (penalty_speed_round : ℝ) (penalty_lightning_round : ℝ)

-- Given conditions from the problem statement
def kaleb_initial_scores (score_first_half score_second_half : ℝ) := 
  score_first_half = 43 ∧ score_second_half = 23

def kaleb_bonuses (score_first_half bonus_special_q bonus_streak : ℝ) :=
  bonus_special_q = 0.20 * score_first_half ∧ bonus_streak = 0.05 * score_first_half

def kaleb_penalties (score_second_half penalty_speed_round penalty_lightning_round : ℝ) := 
  penalty_speed_round = 0.10 * score_second_half ∧ penalty_lightning_round = 0.08 * score_second_half

-- The final score adjusted with all bonuses and penalties
def kaleb_adjusted_score (score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round : ℝ) : ℝ := 
  score_first_half + bonus_special_q + bonus_streak + score_second_half - penalty_speed_round - penalty_lightning_round

theorem kaleb_final_score :
  kaleb_initial_scores score_first_half score_second_half ∧
  kaleb_bonuses score_first_half bonus_special_q bonus_streak ∧
  kaleb_penalties score_second_half penalty_speed_round penalty_lightning_round →
  kaleb_adjusted_score score_first_half score_second_half bonus_special_q bonus_streak penalty_speed_round penalty_lightning_round = 72.61 :=
by
  intros
  sorry

end kaleb_final_score_l183_183151


namespace six_digit_count_div_by_217_six_digit_count_div_by_218_l183_183575

-- Definitions for the problem
def six_digit_format (n : ℕ) : Prop :=
  ∃ a b : ℕ, (0 ≤ a ∧ a < 10) ∧ (0 ≤ b ∧ b < 10) ∧ n = 100001 * a + 10010 * b + 100 * a + 10 * b + a

def divisible_by (n : ℕ) (divisor : ℕ) : Prop :=
  n % divisor = 0

-- Problem Part a: How many six-digit numbers of the form are divisible by 217
theorem six_digit_count_div_by_217 :
  ∃ count : ℕ, count = 3 ∧ ∀ n : ℕ, six_digit_format n → divisible_by n 217  → (n = 313131 ∨ n = 626262 ∨ n = 939393) :=
sorry

-- Problem Part b: How many six-digit numbers of the form are divisible by 218
theorem six_digit_count_div_by_218 :
  ∀ n : ℕ, six_digit_format n → divisible_by n 218 → false :=
sorry

end six_digit_count_div_by_217_six_digit_count_div_by_218_l183_183575


namespace linear_transform_determined_by_points_l183_183612

theorem linear_transform_determined_by_points
  (z1 z2 w1 w2 : ℂ)
  (h1 : z1 ≠ z2)
  (h2 : w1 ≠ w2)
  : ∃ (a b : ℂ), ∀ (z : ℂ), a = (w2 - w1) / (z2 - z1) ∧ b = (w1 * z2 - w2 * z1) / (z2 - z1) ∧ (a * z1 + b = w1) ∧ (a * z2 + b = w2) := 
sorry

end linear_transform_determined_by_points_l183_183612


namespace total_combinations_l183_183802

def varieties_of_wrapping_paper : Nat := 10
def colors_of_ribbon : Nat := 4
def types_of_gift_cards : Nat := 5
def kinds_of_decorative_stickers : Nat := 2

theorem total_combinations : varieties_of_wrapping_paper * colors_of_ribbon * types_of_gift_cards * kinds_of_decorative_stickers = 400 := by
  sorry

end total_combinations_l183_183802


namespace find_m_l183_183433

-- Define the conditions with variables a, b, and m.
variable (a b m : ℝ)
variable (ha : 2^a = m)
variable (hb : 5^b = m)
variable (hc : 1/a + 1/b = 2)

-- Define the statement to be proven.
theorem find_m : m = Real.sqrt 10 :=
by
  sorry


end find_m_l183_183433


namespace fly_reaches_x_coordinate_l183_183246

theorem fly_reaches_x_coordinate (n : ℕ) : 
  (∀ (x : ℕ), x < n → random_walk_2d x 0) → 
  ∀ (x y : ℕ), ∃ (t : ℕ), (random_walk t (0, 0) = (2011, y)) :=
by
  sorry

end fly_reaches_x_coordinate_l183_183246


namespace proof_problem_l183_183818

def problem_statement := 
  let m : ℕ := 2022 in 
  ⌊ (2023^3 / (2021 * 2022) - (2021^3 / (2022 * 2023)) ) ⌋ = 8

theorem proof_problem : problem_statement :=
by sorry

end proof_problem_l183_183818


namespace arc_length_solution_l183_183148

variable (r : ℝ) (α : ℝ)

theorem arc_length_solution (h1 : r = 8) (h2 : α = 5 * Real.pi / 3) : 
    r * α = 40 * Real.pi / 3 := 
by 
    sorry

end arc_length_solution_l183_183148


namespace intersection_S_T_l183_183046

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183046


namespace number_of_sheep_l183_183352

-- Define the conditions as given in the problem
variables (S H : ℕ)
axiom ratio_condition : S * 7 = H * 3
axiom food_condition : H * 230 = 12880

-- The theorem to prove
theorem number_of_sheep : S = 24 :=
by sorry

end number_of_sheep_l183_183352


namespace enthalpy_change_l183_183782

def DeltaH_prods : Float := -286.0 - 297.0
def DeltaH_reacts : Float := -20.17
def HessLaw (DeltaH_prods DeltaH_reacts : Float) : Float := DeltaH_prods - DeltaH_reacts

theorem enthalpy_change : HessLaw DeltaH_prods DeltaH_reacts = -1125.66 := by
  -- Lean needs a proof, which is not needed per instructions
  sorry

end enthalpy_change_l183_183782


namespace avg_books_per_student_l183_183299

theorem avg_books_per_student 
  (total_students : ℕ)
  (students_zero_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (max_books_per_student : ℕ) 
  (remaining_students_min_books : ℕ)
  (total_books : ℕ)
  (avg_books : ℚ)
  (h1 : total_students = 32)
  (h2 : students_zero_books = 2)
  (h3 : students_one_book = 12)
  (h4 : students_two_books = 10)
  (h5 : max_books_per_student = 11)
  (h6 : remaining_students_min_books = 8)
  (h7 : total_books = 0 * students_zero_books + 1 * students_one_book + 2 * students_two_books + 3 * remaining_students_min_books)
  (h8 : avg_books = total_books / total_students) :
  avg_books = 1.75 :=
by {
  -- Additional constraints and intermediate steps can be added here if necessary
  sorry
}

end avg_books_per_student_l183_183299


namespace yunjeong_locker_problem_l183_183489

theorem yunjeong_locker_problem
  (l r f b : ℕ)
  (h_l : l = 7)
  (h_r : r = 13)
  (h_f : f = 8)
  (h_b : b = 14)
  (same_rows : ∀ pos1 pos2 : ℕ, pos1 = pos2) :
  (l - 1) + (r - 1) + (f - 1) + (b - 1) = 399 := sorry

end yunjeong_locker_problem_l183_183489


namespace correct_option_is_C_l183_183643

-- Definitions for given conditions
def optionA (x y : ℝ) : Prop := 3 * x + 3 * y = 6 * x * y
def optionB (x y : ℝ) : Prop := 4 * x * y^2 - 5 * x * y^2 = -1
def optionC (x : ℝ) : Prop := -2 * (x - 3) = -2 * x + 6
def optionD (a : ℝ) : Prop := 2 * a + a = 3 * a^2

-- The proof statement to show that Option C is the correct calculation
theorem correct_option_is_C (x y a : ℝ) : 
  ¬ optionA x y ∧ ¬ optionB x y ∧ optionC x ∧ ¬ optionD a :=
by
  -- Proof not required, using sorry to compile successfully
  sorry

end correct_option_is_C_l183_183643


namespace sequence_general_term_l183_183552

theorem sequence_general_term (n : ℕ) : 
  (∀ (a : ℕ → ℚ), (a 1 = 1) ∧ (a 2 = 2 / 3) ∧ (a 3 = 3 / 7) ∧ (a 4 = 4 / 15) ∧ (a 5 = 5 / 31) → a n = n / (2^n - 1)) :=
by
  sorry

end sequence_general_term_l183_183552


namespace correct_calculation_l183_183370

theorem correct_calculation (n : ℕ) (h : n - 59 = 43) : n - 46 = 56 :=
by
  sorry

end correct_calculation_l183_183370


namespace problem_solution_l183_183319

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x < -6 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 ))
  (h2 : a < b)
  : a + 2 * b + 3 * c = 74 := 
sorry

end problem_solution_l183_183319


namespace rationalize_denominator_sum_l183_183474

theorem rationalize_denominator_sum :
  let A := -4
  let B := 7
  let C := 3
  let D := 13
  let E := 1
  A + B + C + D + E = 20 := by
    sorry

end rationalize_denominator_sum_l183_183474


namespace tickets_sold_at_reduced_price_first_week_l183_183549

variable (T : ℕ) (F : ℕ) (R : ℕ)
variable hT : T = 25200
variable hF : F = 16500
variable hR : T - F = R
variable hFullPriceMultiple : F = 5 * R

theorem tickets_sold_at_reduced_price_first_week : 
  R = 8700 :=
by
  rw [hT, hF, hR, hFullPriceMultiple] at *
  sorry

end tickets_sold_at_reduced_price_first_week_l183_183549


namespace similar_triangle_perimeter_l183_183807

noncomputable def is_similar_triangles (a b c a' b' c' : ℝ) := 
  ∃ (k : ℝ), k > 0 ∧ (a = k * a') ∧ (b = k * b') ∧ (c = k * c')

noncomputable def is_isosceles (a b c : ℝ) := (a = b) ∨ (a = c) ∨ (b = c)

theorem similar_triangle_perimeter :
  ∀ (a b c a' b' c' : ℝ),
    is_isosceles a b c → 
    is_similar_triangles a b c a' b' c' →
    c' = 42 →
    (a = 12) → 
    (b = 12) → 
    (c = 14) →
    (b' = 36) →
    (a' = 36) →
    a' + b' + c' = 114 :=
by
  intros
  sorry

end similar_triangle_perimeter_l183_183807


namespace find_x2_plus_y2_l183_183581

theorem find_x2_plus_y2
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := 
by
  sorry

end find_x2_plus_y2_l183_183581


namespace circle_radius_l183_183993

noncomputable def circle_problem (rD rE : ℝ) (m n : ℝ) :=
  rD = 2 * rE ∧
  rD = (Real.sqrt m) - n ∧
  m ≥ 0 ∧ n ≥ 0

theorem circle_radius (rE rD : ℝ) (m n : ℝ) (h : circle_problem rD rE m n) :
  m + n = 5.76 :=
by
  sorry

end circle_radius_l183_183993


namespace sector_angle_measure_l183_183584

theorem sector_angle_measure (r α : ℝ) 
  (h1 : 2 * r + α * r = 6)
  (h2 : (1 / 2) * α * r^2 = 2) :
  α = 1 ∨ α = 4 := 
sorry

end sector_angle_measure_l183_183584


namespace symmetric_circle_l183_183457

-- Define given circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 8 * y + 12 = 0

-- Define the line of symmetry
def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 5 = 0

-- Define the symmetric circle equation we need to prove
def symm_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

-- Lean 4 theorem statement
theorem symmetric_circle (x y : ℝ) :
  (∃ a b : ℝ, circle_equation 2 4 ∧ line_equation a b ∧ (a, b) = (0, 0)) →
  symm_circle_equation x y :=
by sorry

end symmetric_circle_l183_183457


namespace max_slope_without_lattice_points_l183_183530

theorem max_slope_without_lattice_points :
  ∃ b : ℚ,
    (∀ m : ℚ, (1 : ℚ) / 3 < m → m < b → 
      ∀ x : ℤ, 1 ≤ x → x ≤ 150 → ∀ y : ℤ, y ≠ m * x + 3) ∧
    b = 50 / 151 := 
begin
  -- proof here
  sorry
end

end max_slope_without_lattice_points_l183_183530


namespace quadratic_inequality_solution_set_l183_183572

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, 1 < x ∧ x < 3 → x^2 < ax + b) : b^a = 81 :=
sorry

end quadratic_inequality_solution_set_l183_183572


namespace smaller_angle_at_3_20_l183_183943

noncomputable def clock_angle_3_20 : ℝ := {
  let minute_hand_angle := 20 * 6 in -- minute hand movement from 12 o'clock
  let hour_hand_angle := 90 + (20 * 0.5) in -- hour hand movement from 3 o'clock position
  let angle_between_hands := abs (minute_hand_angle - hour_hand_angle) in
  if angle_between_hands <= 180 then
    angle_between_hands
  else
    360 - angle_between_hands
}

theorem smaller_angle_at_3_20 : clock_angle_3_20 = 20.0 := by
  -- The condition and intermediary steps are asserted in the definition itself
  sorry

end smaller_angle_at_3_20_l183_183943


namespace john_weight_end_l183_183725

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end_l183_183725


namespace intersection_S_T_eq_T_l183_183004

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183004


namespace sequence_a7_l183_183347

theorem sequence_a7 (a b : ℕ) (h1 : a1 = a) (h2 : a2 = b) {a3 a4 a5 a6 a7 : ℕ}
  (h3 : a_3 = a + b)
  (h4 : a_4 = a + 2 * b)
  (h5 : a_5 = 2 * a + 3 * b)
  (h6 : a_6 = 3 * a + 5 * b)
  (h_a6 : a_6 = 50) :
  a_7 = 5 * a + 8 * b :=
by
  sorry

end sequence_a7_l183_183347


namespace line_l_passes_fixed_point_line_l_perpendicular_value_a_l183_183891

variable (a : ℝ)

def line_l (a : ℝ) : ℝ × ℝ → Prop :=
  λ p => (a + 1) * p.1 + p.2 + 2 - a = 0

def perpendicular_line : ℝ × ℝ → Prop :=
  λ p => 2 * p.1 - 3 * p.2 + 4 = 0

theorem line_l_passes_fixed_point :
  line_l a (1, -3) :=
by
  sorry

theorem line_l_perpendicular_value_a (a : ℝ) :
  (∀ p : ℝ × ℝ, perpendicular_line p → line_l a p) → 
  a = 1 / 2 :=
by
  sorry

end line_l_passes_fixed_point_line_l_perpendicular_value_a_l183_183891


namespace largest_perfect_square_factor_9240_l183_183637

theorem largest_perfect_square_factor_9240 :
  ∃ n : ℕ, n * n = 36 ∧ ∃ m : ℕ, m ∣ 9240 ∧ m = n * n :=
by
  -- We will construct the proof here using the prime factorization
  sorry

end largest_perfect_square_factor_9240_l183_183637


namespace total_points_scored_l183_183298

theorem total_points_scored (n m T : ℕ) 
  (h1 : T = 2 * n + 5 * m) 
  (h2 : n = m + 3 ∨ m = n + 3)
  : T = 20 :=
sorry

end total_points_scored_l183_183298


namespace maximum_value_expression_l183_183888

theorem maximum_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_sum : a + b + c = 3) : 
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 1 :=
sorry

end maximum_value_expression_l183_183888


namespace find_a_of_tangent_area_l183_183291

theorem find_a_of_tangent_area (a : ℝ) (h : a > 0) (h_area : (a^3 / 4) = 2) : a = 2 :=
by
  -- Proof is omitted as it's not required.
  sorry

end find_a_of_tangent_area_l183_183291


namespace intersection_S_T_eq_T_l183_183086

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183086


namespace fraction_received_A_correct_l183_183296

def fraction_of_students_received_A := 0.7
def fraction_of_students_received_B := 0.2
def fraction_of_students_received_A_or_B := 0.9

theorem fraction_received_A_correct :
  fraction_of_students_received_A_or_B - fraction_of_students_received_B = fraction_of_students_received_A :=
by
  sorry

end fraction_received_A_correct_l183_183296


namespace number_subtracted_from_10000_l183_183444

theorem number_subtracted_from_10000 (x : ℕ) (h : 10000 - x = 9001) : x = 999 := by
  sorry

end number_subtracted_from_10000_l183_183444


namespace figure_can_be_rearranged_to_square_l183_183399

def can_form_square (n : ℕ) : Prop :=
  let s := Nat.sqrt n
  s * s = n

theorem figure_can_be_rearranged_to_square (n : ℕ) :
  (∃ a b c : ℕ, a + b + c = n) → (can_form_square n) → (n % 1 = 0) :=
by
  intros _ _
  sorry

end figure_can_be_rearranged_to_square_l183_183399


namespace largest_integer_binom_l183_183222

theorem largest_integer_binom
  (n : ℕ)
  (h₁ : binom 10 4 + binom 10 5 = binom 11 n) :
  n = 6 :=
sorry

end largest_integer_binom_l183_183222


namespace tagged_fish_in_second_catch_l183_183449

theorem tagged_fish_in_second_catch :
  ∀ (T : ℕ),
    (40 > 0) →
    (800 > 0) →
    (T / 40 = 40 / 800) →
    T = 2 := 
by
  intros T h1 h2 h3
  sorry

end tagged_fish_in_second_catch_l183_183449


namespace coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l183_183364

theorem coefficient_of_x9_in_expansion_of_x_minus_2_pow_10 :
  ∃ c : ℤ, (x - 2)^10 = ∑ k in finset.range (11), (nat.choose 10 k) * x^k * (-2)^(10 - k) ∧ c = -20 := 
begin 
  use -20,
  { sorry },
end

end coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l183_183364


namespace reciprocal_of_neg_one_fifth_l183_183197

theorem reciprocal_of_neg_one_fifth : (-(1 / 5) : ℚ)⁻¹ = -5 :=
by
  sorry

end reciprocal_of_neg_one_fifth_l183_183197


namespace output_correct_l183_183780

-- Definitions derived from the conditions
def initial_a : Nat := 3
def initial_b : Nat := 4

-- Proof that the final output of PRINT a, b is (4, 4)
theorem output_correct : 
  let a := initial_a;
  let b := initial_b;
  let a := b;
  let b := a;
  (a, b) = (4, 4) :=
by
  sorry

end output_correct_l183_183780


namespace surface_area_increase_l183_183527

theorem surface_area_increase (r h : ℝ) (cs : Bool) : -- cs is a condition switch, True for circular cut, False for rectangular cut
  0 < r ∧ 0 < h →
  let inc_area := if cs then 2 * π * r^2 else 2 * h * r 
  inc_area > 0 :=
by 
  sorry

end surface_area_increase_l183_183527


namespace yoonseok_handshakes_l183_183785

-- Conditions
def totalFriends : ℕ := 12
def yoonseok := "Yoonseok"
def adjacentFriends (i : ℕ) : Prop := i = 1 ∨ i = (totalFriends - 1)

-- Problem Statement
theorem yoonseok_handshakes : 
  ∀ (totalFriends : ℕ) (adjacentFriends : ℕ → Prop), 
    totalFriends = 12 → 
    (∀ i, adjacentFriends i ↔ i = 1 ∨ i = (totalFriends - 1)) → 
    (totalFriends - 1 - 2 = 9) := by
  intros totalFriends adjacentFriends hTotal hAdjacent
  have hSub : totalFriends - 1 - 2 = 9 := by sorry
  exact hSub

end yoonseok_handshakes_l183_183785


namespace lisa_interest_correct_l183_183756

noncomputable def lisa_interest : ℝ :=
  let P := 2000
  let r := 0.035
  let n := 10
  let A := P * (1 + r) ^ n
  A - P

theorem lisa_interest_correct :
  lisa_interest = 821 := by
  sorry

end lisa_interest_correct_l183_183756


namespace geometric_sequence_second_term_l183_183384

theorem geometric_sequence_second_term
  (first_term : ℕ) (fourth_term : ℕ) (r : ℕ)
  (h1 : first_term = 6)
  (h2 : first_term * r^3 = fourth_term)
  (h3 : fourth_term = 768) :
  first_term * r = 24 := by
  sorry

end geometric_sequence_second_term_l183_183384


namespace filling_time_with_ab_l183_183551

theorem filling_time_with_ab (a b c l : ℝ) (h1 : a + b + c - l = 5 / 6) (h2 : a + c - l = 1 / 2) (h3 : b + c - l = 1 / 3) : 
  1 / (a + b) = 1.2 :=
by
  sorry

end filling_time_with_ab_l183_183551


namespace monotonicity_case1_monotonicity_case2_lower_bound_l183_183850

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 (a : ℝ) (h : a ≤ 0) : 
  ∀ x y : ℝ, x < y → f a x > f a y := 
by
  sorry

theorem monotonicity_case2 (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, x < Real.log (1 / a) → f a x > f a (Real.log (1 / a)) ∧ 
           x > Real.log (1 / a) → f a x < f a (Real.log (1 / a)) := 
by
  sorry

theorem lower_bound (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ,  f a x > 2 * Real.log a + 3 / 2 := 
by
  sorry

end monotonicity_case1_monotonicity_case2_lower_bound_l183_183850


namespace car_total_travel_time_l183_183323

def T_NZ : ℕ := 60

def T_NR : ℕ := 8 / 10 * T_NZ -- 80% of T_NZ

def T_ZV : ℕ := 3 / 4 * T_NR -- 75% of T_NR

theorem car_total_travel_time :
  T_NZ + T_NR + T_ZV = 144 := by
  sorry

end car_total_travel_time_l183_183323


namespace carter_cheesecakes_l183_183254

theorem carter_cheesecakes (C : ℕ) (nm : ℕ) (nr : ℕ) (increase : ℕ) (this_week_cakes : ℕ) (usual_cakes : ℕ) :
  nm = 5 → nr = 8 → increase = 38 → 
  this_week_cakes = 3 * C + 3 * nm + 3 * nr → 
  usual_cakes = C + nm + nr → 
  this_week_cakes = usual_cakes + increase → 
  C = 6 :=
by
  intros hnm hnr hinc htw husual hcakes
  sorry

end carter_cheesecakes_l183_183254


namespace expression_values_l183_183577

variable {a b c : ℚ}

theorem expression_values (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = 2) ∨ 
  (a / abs a + b / abs b + c / abs c - (a * b * c) / abs (a * b * c) = -2) := 
sorry

end expression_values_l183_183577


namespace prime_factorization_675_l183_183995

theorem prime_factorization_675 :
  ∃ (n h : ℕ), n > 1 ∧ n = 3 ∧ h = 225 ∧ 675 = (3^3) * (5^2) :=
by
  sorry

end prime_factorization_675_l183_183995


namespace solve_x_eq_40_l183_183478

theorem solve_x_eq_40 : ∀ (x : ℝ), x + 2 * x = 400 - (3 * x + 4 * x) → x = 40 :=
by
  intro x
  intro h
  sorry

end solve_x_eq_40_l183_183478


namespace sum_of_consecutive_numbers_with_lcm_168_l183_183910

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l183_183910


namespace arithmetic_seq_solution_l183_183150

theorem arithmetic_seq_solution (a : ℕ → ℝ) (n : ℕ) (S : ℕ → ℝ) 
  (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith : ∀ n ≥ 2, a (n+1) - a n ^ 2 + a (n-1) = 0) 
  (h_sum : ∀ k, S k = (k * (a 1 + a k)) / 2) :
  S (2 * n - 1) - 4 * n = -2 := 
sorry

end arithmetic_seq_solution_l183_183150


namespace find_actual_positions_l183_183713

namespace RunningCompetition

variables (A B C D E : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

/-- The first prediction: $A$ finishes first, $B$ is second, $C$ is third, $D$ is fourth, $E$ is fifth. --/
def first_prediction := [A, B, C, D, E]

/-- The second prediction: $C$ finishes first, $E$ is second, $A$ is third, $B$ is fourth, $D$ is fifth. --/
def second_prediction := [C, E, A, B, D]

/-- The first prediction correctly predicted the positions of exactly three athletes. --/
def first_prediction_correct_count : ℕ := 3

/-- The second prediction correctly predicted the positions of exactly two athletes. --/
def second_prediction_correct_count : ℕ := 2

/-- Proves that the actual finishing order is $C, B, A, D, E$. --/
theorem find_actual_positions : 
  ∀ (order : list (Type)), 
  (order = [C, B, A, D, E]) →
  list.count order first_prediction_correct_count = 3 ∧ 
  list.count order second_prediction_correct_count = 2 :=
by
  sorry

end RunningCompetition

end find_actual_positions_l183_183713


namespace percentage_problem_l183_183524

noncomputable def percentage_of_value (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
  (y / x) * 100

theorem percentage_problem :
  percentage_of_value 2348 (528.0642570281125 * 4.98) = 112 := 
by
  sorry

end percentage_problem_l183_183524


namespace a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l183_183275

noncomputable def a_0 (n : ℕ) : ℕ := 2^n
noncomputable def S_n (n : ℕ) : ℕ := 3^n - 2^n
noncomputable def T_n (n : ℕ) : ℕ := (n - 2) * 2^n + 2 * n^2

theorem a_0_eq_2_pow_n (n : ℕ) (h : n > 0) : a_0 n = 2^n := sorry

theorem S_n_eq_3_pow_n_minus_2_pow_n (n : ℕ) (h : n > 0) : S_n n = 3^n - 2^n := sorry

theorem S_n_magnitude_comparison : 
  ∀ (n : ℕ), 
    (n = 1 → S_n n > T_n n) ∧
    (n = 2 ∨ n = 3 → S_n n < T_n n) ∧
    (n ≥ 4 → S_n n > T_n n) := sorry

end a_0_eq_2_pow_n_S_n_eq_3_pow_n_minus_2_pow_n_S_n_magnitude_comparison_l183_183275


namespace new_trailers_added_l183_183502

theorem new_trailers_added :
  let initial_trailers := 25
  let initial_average_age := 15
  let years_passed := 3
  let current_average_age := 12
  let total_initial_age := initial_trailers * (initial_average_age + years_passed)
  ∀ n : Nat, 
    ((25 * 18) + (n * 3) = (25 + n) * 12) →
    n = 17 := 
by
  intros
  sorry

end new_trailers_added_l183_183502


namespace starWarsEarned405_l183_183483

-- Definitions and Hypotheses
variables (cost_LionKing : ℕ) (earnings_LionKing : ℕ) (cost_StarWars : ℕ) (earnings_StarWars : ℕ)
variables (profit_LionKing half_profit_StarWars profit_StarWars : ℕ)

-- Conditions
def lionKingCost : cost_LionKing = 10 := rfl
def lionKingEarnings : earnings_LionKing = 200 := rfl
def starWarsCost : cost_StarWars = 25 := rfl
def lionKingProfit : profit_LionKing = earnings_LionKing - cost_LionKing := by rw [lionKingEarnings, lionKingCost]
def halfProfitCondition : profit_LionKing = profit_StarWars / 2 := by rw lionKingProfit
def starWarsEarnings : earnings_StarWars = cost_StarWars + profit_StarWars := by rw [starWarsCost]

-- Main theorem
theorem starWarsEarned405 : earnings_StarWars = 405 :=
by {
  rw [starWarsEarnings, lionKingProfit, halfProfitCondition, lionKingEarnings, starWarsCost],
  simp, 
  refine sorry,  -- Proof steps skipped
}

end starWarsEarned405_l183_183483


namespace a_pow_11_b_pow_11_l183_183321

theorem a_pow_11_b_pow_11 (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 :=
by
  sorry

end a_pow_11_b_pow_11_l183_183321


namespace silver_excess_in_third_chest_l183_183535

theorem silver_excess_in_third_chest :
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ),
    x1 + x2 + x3 = 40 →
    y1 + y2 + y3 = 40 →
    x1 = y1 + 7 →
    y2 = x2 - 15 →
    y3 = x3 + 22 :=
by
  intros x1 y1 x2 y2 x3 y3 h1 h2 h3 h4
  sorry

end silver_excess_in_third_chest_l183_183535


namespace reciprocal_difference_decreases_l183_183310

theorem reciprocal_difference_decreases (n : ℕ) (hn : n > 0) : 
  (1 / (n : ℝ) - 1 / (n + 1 : ℝ)) < (1 / (n * n : ℝ)) :=
by 
  sorry

end reciprocal_difference_decreases_l183_183310


namespace find_xy_l183_183601

theorem find_xy (x y : ℝ) (h1 : (x / 6) * 12 = 11) (h2 : 4 * (x - y) + 5 = 11) : 
  x = 5.5 ∧ y = 4 :=
sorry

end find_xy_l183_183601


namespace gigi_initial_batches_l183_183838

-- Define the conditions
def flour_per_batch := 2 
def initial_flour := 20 
def remaining_flour := 14 
def future_batches := 7

-- Prove the number of batches initially baked is 3
theorem gigi_initial_batches :
  (initial_flour - remaining_flour) / flour_per_batch = 3 :=
by
  sorry

end gigi_initial_batches_l183_183838


namespace boxes_in_carton_l183_183973

theorem boxes_in_carton (cost_per_pack : ℕ) (packs_per_box : ℕ) (cost_dozen_cartons : ℕ) 
  (h1 : cost_per_pack = 1) (h2 : packs_per_box = 10) (h3 : cost_dozen_cartons = 1440) :
  (cost_dozen_cartons / 12) / (cost_per_pack * packs_per_box) = 12 :=
by
  sorry

end boxes_in_carton_l183_183973


namespace intersection_S_T_eq_T_l183_183056

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183056


namespace quadratic_distinct_real_roots_l183_183583

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + m - 1 = 0) ∧ (x2^2 - 4*x2 + m - 1 = 0)) → m < 5 := sorry

end quadratic_distinct_real_roots_l183_183583


namespace find_greatest_and_second_greatest_problem_solution_l183_183415

theorem find_greatest_and_second_greatest
  (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : (a > b) ∧ (b > c) ∧ (c > d) :=
by 
  sorry

def greatest_and_second_greatest_eq (x1 x2 : ℝ) : Prop :=
  x1 = 4 ^ (1 / 4) ∧ x2 = 5 ^ (1 / 5)

theorem problem_solution (a b c d : ℝ)
  (ha : a = 4 ^ (1 / 4))
  (hb : b = 5 ^ (1 / 5))
  (hc : c = 16 ^ (1 / 16))
  (hd : d = 25 ^ (1 / 25))
  : greatest_and_second_greatest_eq a b :=
by 
  sorry

end find_greatest_and_second_greatest_problem_solution_l183_183415


namespace solve_equation_l183_183336

theorem solve_equation (x y : ℤ) (h : 3 * (y - 2) = 5 * (x - 1)) :
  (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 7) :=
sorry

end solve_equation_l183_183336


namespace find_sum_xyz_l183_183859

-- Define the problem
def system_of_equations (x y z : ℝ) : Prop :=
  x^2 + x * y + y^2 = 27 ∧
  y^2 + y * z + z^2 = 9 ∧
  z^2 + z * x + x^2 = 36

-- The main theorem to be proved
theorem find_sum_xyz (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 18 :=
sorry

end find_sum_xyz_l183_183859


namespace largest_n_binom_equality_l183_183213

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l183_183213


namespace evaluate_fraction_l183_183598

theorem evaluate_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - b * (1 / a) ≠ 0) :
  (a^2 - 1 / b^2) / (b^2 - 1 / a^2) = a^2 / b^2 :=
by
  sorry

end evaluate_fraction_l183_183598


namespace eval_expression_l183_183784

-- We define the expression that needs to be evaluated
def expression := (0.76)^3 - (0.1)^3 / (0.76)^2 + 0.076 + (0.1)^2

-- The statement to prove
theorem eval_expression : expression = 0.5232443982683983 :=
by
  sorry

end eval_expression_l183_183784


namespace value_of_h_otimes_h_otimes_h_l183_183996

variable (h x y : ℝ)

-- Define the new operation
def otimes (x y : ℝ) := x^3 - x * y + y^2

-- Prove that h ⊗ (h ⊗ h) = h^6 - h^4 + h^3
theorem value_of_h_otimes_h_otimes_h :
  otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end value_of_h_otimes_h_otimes_h_l183_183996


namespace algebraic_expression_value_l183_183695

variable (a b : ℝ)

theorem algebraic_expression_value
  (h : a^2 + 2 * b^2 - 1 = 0) :
  (a - b)^2 + b * (2 * a + b) = 1 :=
by
  sorry

end algebraic_expression_value_l183_183695


namespace boys_and_girls_are_equal_l183_183895

theorem boys_and_girls_are_equal (B G : ℕ) (h1 : B + G = 30)
    (h2 : ∀ b₁ b₂, b₁ ≠ b₂ → (0 ≤ b₁) ∧ (b₁ ≤ G - 1) → (0 ≤ b₂) ∧ (b₂ ≤ G - 1) → b₁ ≠ b₂)
    (h3 : ∀ g₁ g₂, g₁ ≠ g₂ → (0 ≤ g₁) ∧ (g₁ ≤ B - 1) → (0 ≤ g₂) ∧ (g₂ ≤ B - 1) → g₁ ≠ g₂) : 
    B = 15 ∧ G = 15 := by
  sorry

end boys_and_girls_are_equal_l183_183895


namespace solve_inequality_when_a_is_one_range_of_values_for_a_l183_183571

open Real

-- Part (1) Statement
theorem solve_inequality_when_a_is_one (a x : ℝ) (h : a = 1) : 
  |x - a| + |x + 2| ≤ 5 → -3 ≤ x ∧ x ≤ 2 := 
by sorry

-- Part (2) Statement
theorem range_of_values_for_a (a : ℝ) : 
  (∃ x_0 : ℝ, |x_0 - a| + |x_0 + 2| ≤ |2 * a + 1|) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by sorry

end solve_inequality_when_a_is_one_range_of_values_for_a_l183_183571


namespace uncle_ben_eggs_l183_183634

noncomputable def total_eggs (total_chickens : ℕ) (roosters : ℕ) (non_egg_laying_hens : ℕ) (eggs_per_hen : ℕ) : ℕ :=
  let total_hens := total_chickens - roosters
  let egg_laying_hens := total_hens - non_egg_laying_hens
  egg_laying_hens * eggs_per_hen

theorem uncle_ben_eggs :
  total_eggs 440 39 15 3 = 1158 :=
by
  unfold total_eggs
  -- Correct steps to prove the theorem can be skipped with sorry
  sorry

end uncle_ben_eggs_l183_183634


namespace find_other_number_l183_183924

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l183_183924


namespace intersection_eq_T_l183_183032

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183032


namespace find_youngest_age_l183_183661

noncomputable def youngest_child_age 
  (meal_cost_mother : ℝ) 
  (meal_cost_per_year : ℝ) 
  (total_bill : ℝ) 
  (triplets_count : ℕ) := 
  {y : ℝ // 
    (∃ t : ℝ, 
      meal_cost_mother + meal_cost_per_year * (triplets_count * t + y) = total_bill ∧ y = 2 ∨ y = 5)}

theorem find_youngest_age : 
  youngest_child_age 3.75 0.50 12.25 3 := 
sorry

end find_youngest_age_l183_183661


namespace total_fish_in_lake_l183_183972

-- Given conditions:
def initiallyTaggedFish : ℕ := 100
def capturedFish : ℕ := 100
def taggedFishInAugust : ℕ := 5
def taggedFishMortalityRate : ℝ := 0.3
def newcomerFishRate : ℝ := 0.2

-- Proof to show that the total number of fish at the beginning of April is 1120
theorem total_fish_in_lake (initiallyTaggedFish capturedFish taggedFishInAugust : ℕ) 
  (taggedFishMortalityRate newcomerFishRate : ℝ) : 
  (taggedFishInAugust : ℝ) / (capturedFish * (1 - newcomerFishRate)) = 
  ((initiallyTaggedFish * (1 - taggedFishMortalityRate)) : ℝ) / (1120 : ℝ) :=
by 
  sorry

end total_fish_in_lake_l183_183972


namespace train_length_is_correct_l183_183803

noncomputable def length_of_train (speed_train_kmh : ℝ) (speed_man_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := relative_speed_kmh * (5/18)
  let length := relative_speed_ms * time_s
  length

theorem train_length_is_correct (h1 : 84 = 84) (h2 : 6 = 6) (h3 : 4.399648028157747 = 4.399648028157747) :
  length_of_train 84 6 4.399648028157747 = 110.991201 := by
  dsimp [length_of_train]
  norm_num
  sorry

end train_length_is_correct_l183_183803


namespace calc_result_l183_183989

theorem calc_result :
  12 / 4 - 3 - 16 + 4 * 6 = 8 := by
  sorry

end calc_result_l183_183989


namespace intersection_S_T_l183_183023

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183023


namespace part1_proof_part2_proof_l183_183701

open Real

-- Definitions for the conditions
variables (x y z : ℝ)
variable (h₁ : 0 < x)
variable (h₂ : 0 < y)
variable (h₃ : 0 < z)

-- Part 1
theorem part1_proof : (1 / x + 1 / y ≥ 4 / (x + y)) :=
by sorry

-- Part 2
theorem part2_proof : (1 / x + 1 / y + 1 / z ≥ 2 / (x + y) + 2 / (y + z) + 2 / (z + x)) :=
by sorry

end part1_proof_part2_proof_l183_183701


namespace correct_calculation_l183_183229

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end correct_calculation_l183_183229


namespace beautifulEquations_1_find_n_l183_183142

def isBeautifulEquations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ x y : ℝ, eq1 x ∧ eq2 y ∧ x + y = 1

def eq1a (x : ℝ) : Prop := 4 * x - (x + 5) = 1
def eq2a (y : ℝ) : Prop := -2 * y - y = 3

theorem beautifulEquations_1 : isBeautifulEquations eq1a eq2a :=
sorry

def eq1b (x : ℝ) (n : ℝ) : Prop := 2 * x - n + 3 = 0
def eq2b (x : ℝ) (n : ℝ) : Prop := x + 5 * n - 1 = 0

theorem find_n (n : ℝ) : (∀ x1 x2 : ℝ, eq1b x1 n ∧ eq2b x2 n ∧ x1 + x2 = 1) → n = -1 / 3 :=
sorry

end beautifulEquations_1_find_n_l183_183142


namespace describe_T_l183_183464

def T : Set (ℝ × ℝ) := 
  { p | ∃ x y : ℝ, p = (x, y) ∧ (
      (5 = x + 3 ∧ y - 6 ≤ 5) ∨
      (5 = y - 6 ∧ x + 3 ≤ 5) ∨
      (x + 3 = y - 6 ∧ x + 3 ≤ 5 ∧ y - 6 ≤ 5)
  )}

theorem describe_T : T = { p | ∃ x y : ℝ, p = (2, y) ∧ y ≤ 11 ∨
                                      p = (x, 11) ∧ x ≤ 2 ∨
                                      p = (x, x + 9) ∧ x ≤ 2 ∧ x + 9 ≤ 11 } :=
by
  sorry

end describe_T_l183_183464


namespace part_a_gray_black_area_difference_l183_183376

theorem part_a_gray_black_area_difference :
    ∀ (a b : ℕ), 
        a = 4 → 
        b = 3 →
        a^2 - b^2 = 7 :=
by
  intros a b h_a h_b
  sorry

end part_a_gray_black_area_difference_l183_183376


namespace original_number_people_l183_183249

theorem original_number_people (n : ℕ) (h1 : n / 3 * 2 / 2 = 18) : n = 54 :=
sorry

end original_number_people_l183_183249


namespace remainder_a25_div_26_l183_183465

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Placeholder function for concatenating numbers from 1 to n
  sorry

theorem remainder_a25_div_26 :
  let a_25 := concatenate_numbers 25
  a_25 % 26 = 13 :=
by sorry

end remainder_a25_div_26_l183_183465


namespace correct_calculation_l183_183233

theorem correct_calculation : sqrt 8 / sqrt 2 = 2 :=
by
-- sorry

end correct_calculation_l183_183233


namespace digits_interchanged_l183_183290

theorem digits_interchanged (a b k : ℤ) (h : 10 * a + b = k * (a + b) + 2) :
  10 * b + a = (k + 9) * (a + b) + 2 :=
by
  sorry

end digits_interchanged_l183_183290


namespace solution_exists_l183_183822

variable (x y : ℝ)

noncomputable def condition (x y : ℝ) : Prop :=
  (3 + 5 * x = -4 + 6 * y) ∧ (2 + (-6) * x = 6 + 8 * y)

theorem solution_exists : ∃ (x y : ℝ), condition x y ∧ x = -20 / 19 ∧ y = 11 / 38 := 
  by
  sorry

end solution_exists_l183_183822


namespace lcm_of_18_50_120_l183_183691

theorem lcm_of_18_50_120 : Nat.lcm (Nat.lcm 18 50) 120 = 1800 := by
  sorry

end lcm_of_18_50_120_l183_183691


namespace intersection_of_S_and_T_l183_183106

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183106


namespace neither_sufficient_nor_necessary_l183_183890

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem neither_sufficient_nor_necessary (a : ℝ) :
  (a ∈ M → a ∈ N) = false ∧ (a ∈ N → a ∈ M) = false := by
  sorry

end neither_sufficient_nor_necessary_l183_183890


namespace find_n_in_geometric_series_l183_183248

theorem find_n_in_geometric_series (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = 126 →
  S n = a 1 * (2^n - 1) / (2 - 1) →
  n = 6 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end find_n_in_geometric_series_l183_183248


namespace intersection_S_T_eq_T_l183_183007

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183007


namespace sheila_hourly_wage_l183_183611

def weekly_working_hours : Nat :=
  (8 * 3) + (6 * 2)

def weekly_earnings : Nat :=
  468

def hourly_wage : Nat :=
  weekly_earnings / weekly_working_hours

theorem sheila_hourly_wage : hourly_wage = 13 :=
by
  sorry

end sheila_hourly_wage_l183_183611


namespace find_x_l183_183285

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem find_x
  (h : dot_product vector_a (vector_b x) = 0) :
  x = 2 :=
by
  sorry

end find_x_l183_183285


namespace point_in_fourth_quadrant_l183_183305

def point : ℝ × ℝ := (3, -4)

def isFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def isSecondQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def isThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def isFourthQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : isFourthQuadrant point :=
by
  sorry

end point_in_fourth_quadrant_l183_183305


namespace potato_gun_distance_l183_183987

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

end potato_gun_distance_l183_183987


namespace point_on_circle_l183_183561

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def circle_radius := 5

def A : Point := {x := 2, y := -3}
def M : Point := {x := 5, y := -7}

theorem point_on_circle :
  distance A.x A.y M.x M.y = circle_radius :=
by
  sorry

end point_on_circle_l183_183561


namespace find_r_and_s_l183_183463

theorem find_r_and_s (r s : ℝ) :
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + 10 * x = m * (x - 10) + 5) ↔ r < m ∧ m < s) →
  r + s = 60 :=
sorry

end find_r_and_s_l183_183463


namespace integer_satisfies_mod_l183_183940

theorem integer_satisfies_mod (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 23) (h3 : 38635 % 23 = n % 23) :
  n = 18 := 
sorry

end integer_satisfies_mod_l183_183940


namespace intersection_S_T_eq_T_l183_183059

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183059


namespace largest_n_satisfying_conditions_l183_183834

open Nat

-- Define Euler's totient function φ
noncomputable def totient (n : ℕ) : ℕ := 
  if n = 0 then 0 else finset.card { m ∈ finset.range n | gcd m n = 1 }

-- Define the main theorem statement
theorem largest_n_satisfying_conditions : 
  ∃ (n : ℕ), (∑ m in finset.range (n + 1), ((n / m) - ((n - 1) / m))) = 1992 ∧ totient n ∣ n ∧ 
  (∀ m : ℕ, ( (∑ k in finset.range (m + 1), (m / k) - ((m - 1) / k) = 1992) 
            → (totient m ∣ m) 
            → m ≤ n)) ∧ 
  (n = 2^(1991)) :=
begin
  sorry
end

end largest_n_satisfying_conditions_l183_183834


namespace parallel_lines_implies_value_of_a_l183_183143

theorem parallel_lines_implies_value_of_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y = 0 ∧ x + (a-1)*y + (a^2-1) = 0 → 
  (- a / 2) = - (1 / (a-1))) → a = 2 :=
sorry

end parallel_lines_implies_value_of_a_l183_183143


namespace intersection_S_T_eq_T_l183_183120

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183120


namespace intersection_S_T_eq_T_l183_183012

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183012


namespace triangle_is_isosceles_l183_183144

theorem triangle_is_isosceles
    (A B C : ℝ)
    (h_angle_sum : A + B + C = 180)
    (h_sinB : Real.sin B = 2 * Real.cos C * Real.sin A)
    : (A = C) := 
by
    sorry

end triangle_is_isosceles_l183_183144


namespace cost_of_greenhouses_possible_renovation_plans_l183_183250

noncomputable def cost_renovation (x y : ℕ) : Prop :=
  (2 * x = y + 6) ∧ (x + 2 * y = 48)

theorem cost_of_greenhouses : ∃ x y, cost_renovation x y ∧ x = 12 ∧ y = 18 :=
by {
  sorry
}

noncomputable def renovation_plan (m : ℕ) : Prop :=
  (5 * m + 3 * (8 - m) ≤ 35) ∧ (12 * m + 18 * (8 - m) ≤ 128)

theorem possible_renovation_plans : ∃ m, renovation_plan m ∧ (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  sorry
}

end cost_of_greenhouses_possible_renovation_plans_l183_183250


namespace remaining_people_l183_183757

def initial_football_players : ℕ := 13
def initial_cheerleaders : ℕ := 16
def quitting_football_players : ℕ := 10
def quitting_cheerleaders : ℕ := 4

theorem remaining_people :
  (initial_football_players - quitting_football_players) 
  + (initial_cheerleaders - quitting_cheerleaders) = 15 := by
    -- Proof steps would go here, if required
    sorry

end remaining_people_l183_183757


namespace find_x_parallel_vectors_l183_183694

theorem find_x_parallel_vectors
   (x : ℝ)
   (ha : (x, 2) = (x, 2))
   (hb : (-2, 4) = (-2, 4))
   (hparallel : ∀ (k : ℝ), (x, 2) = (k * -2, k * 4)) :
   x = -1 :=
by
  sorry

end find_x_parallel_vectors_l183_183694


namespace tea_blend_gain_percent_l183_183797

theorem tea_blend_gain_percent :
  let cost_18 := 18
  let cost_20 := 20
  let ratio_5_to_3 := (5, 3)
  let selling_price := 21
  let total_cost := (ratio_5_to_3.1 * cost_18) + (ratio_5_to_3.2 * cost_20)
  let total_weight := ratio_5_to_3.1 + ratio_5_to_3.2
  let cost_price_per_kg := total_cost / total_weight
  let gain_percent := ((selling_price - cost_price_per_kg) / cost_price_per_kg) * 100
  gain_percent = 12 :=
by
  sorry

end tea_blend_gain_percent_l183_183797


namespace perfect_even_multiples_of_3_under_3000_l183_183705

theorem perfect_even_multiples_of_3_under_3000 :
  ∃ n : ℕ, n = 9 ∧ ∀ (k : ℕ), (36 * k^2 < 3000) → (36 * k^2) % 2 = 0 ∧ (36 * k^2) % 3 = 0 ∧ ∃ m : ℕ, m^2 = 36 * k^2 :=
by
  sorry

end perfect_even_multiples_of_3_under_3000_l183_183705


namespace intersection_S_T_eq_T_l183_183112

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183112


namespace intersection_S_T_eq_T_l183_183067

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183067


namespace sum_of_consecutive_numbers_LCM_168_l183_183915

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l183_183915


namespace intersection_S_T_eq_T_l183_183083

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183083


namespace intersection_of_S_and_T_l183_183101

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183101


namespace digit_sum_solution_l183_183318

def S (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_solution : S (S (S (S (2017 ^ 2017)))) = 1 := 
by
  sorry

end digit_sum_solution_l183_183318


namespace carol_total_points_l183_183965

-- Define the conditions for Carol's game points.
def first_round_points := 17
def second_round_points := 6
def last_round_points := -16

-- Prove that the total points at the end of the game are 7.
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end carol_total_points_l183_183965


namespace find_s_for_g_neg1_zero_l183_183727

def g (x s : ℝ) : ℝ := 3 * x^4 + x^3 - 2 * x^2 - 4 * x + s

theorem find_s_for_g_neg1_zero (s : ℝ) : g (-1) s = 0 ↔ s = -4 := by
  sorry

end find_s_for_g_neg1_zero_l183_183727


namespace total_movies_attended_l183_183359

-- Defining the conditions for Timothy's movie attendance
def Timothy_2009 := 24
def Timothy_2010 := Timothy_2009 + 7

-- Defining the conditions for Theresa's movie attendance
def Theresa_2009 := Timothy_2009 / 2
def Theresa_2010 := Timothy_2010 * 2

-- Prove that the total number of movies Timothy and Theresa went to in both years is 129
theorem total_movies_attended :
  (Timothy_2009 + Timothy_2010 + Theresa_2009 + Theresa_2010) = 129 :=
by
  -- proof goes here
  sorry

end total_movies_attended_l183_183359


namespace largest_n_binomial_l183_183208

theorem largest_n_binomial :
  (∃n : ℕ, binom 10 4 + binom 10 5 = binom 11 n) → (n ≤ 6) :=
by
  sorry

end largest_n_binomial_l183_183208


namespace amount_paid_l183_183926

theorem amount_paid (cost_price : ℝ) (percent_more : ℝ) (h1 : cost_price = 6525) (h2 : percent_more = 0.24) : 
  cost_price + percent_more * cost_price = 8091 :=
by 
  -- Proof here
  sorry

end amount_paid_l183_183926


namespace largest_increase_between_2006_and_2007_l183_183674

-- Define the number of students taking the AMC in each year
def students_2002 := 50
def students_2003 := 55
def students_2004 := 63
def students_2005 := 70
def students_2006 := 75
def students_2007_AMC10 := 90
def students_2007_AMC12 := 15

-- Define the total number of students participating in any AMC contest each year
def total_students_2002 := students_2002
def total_students_2003 := students_2003
def total_students_2004 := students_2004
def total_students_2005 := students_2005
def total_students_2006 := students_2006
def total_students_2007 := students_2007_AMC10 + students_2007_AMC12

-- Function to calculate percentage increase
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old : ℕ) : ℚ) / old * 100

-- Calculate percentage increases between the years
def inc_2002_2003 := percentage_increase total_students_2002 total_students_2003
def inc_2003_2004 := percentage_increase total_students_2003 total_students_2004
def inc_2004_2005 := percentage_increase total_students_2004 total_students_2005
def inc_2005_2006 := percentage_increase total_students_2005 total_students_2006
def inc_2006_2007 := percentage_increase total_students_2006 total_students_2007

-- Prove that the largest percentage increase is between 2006 and 2007
theorem largest_increase_between_2006_and_2007 :
  inc_2006_2007 > inc_2005_2006 ∧
  inc_2006_2007 > inc_2004_2005 ∧
  inc_2006_2007 > inc_2003_2004 ∧
  inc_2006_2007 > inc_2002_2003 := 
by {
  sorry
}

end largest_increase_between_2006_and_2007_l183_183674


namespace intersection_S_T_l183_183075

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183075


namespace largest_n_binom_l183_183210

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l183_183210


namespace circle_eq_of_points_value_of_m_l183_183845

-- Define the points on the circle
def P : ℝ × ℝ := (0, -4)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (3, -1)

-- Statement 1: The equation of the circle passing through P, Q, and R
theorem circle_eq_of_points (C : ℝ × ℝ → Prop) :
  (C P ∧ C Q ∧ C R) ↔ ∀ x y : ℝ, C (x, y) ↔ (x - 1)^2 + (y + 2)^2 = 5 := sorry

-- Define the line intersecting the circle and the chord length condition |AB| = 4
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 = 0

-- Statement 2: The value of m such that the chord length |AB| is 4
theorem value_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) → m = 4 / 3 := sorry

end circle_eq_of_points_value_of_m_l183_183845


namespace sum_of_six_numbers_l183_183666

theorem sum_of_six_numbers:
  ∃ (A B C D E F : ℕ), 
    A > B ∧ B > C ∧ C > D ∧ D > E ∧ E > F ∧
    E > F ∧ C > F ∧ D > F ∧ A + B + C + D + E + F = 141 := 
sorry

end sum_of_six_numbers_l183_183666


namespace probability_correct_guesses_l183_183963

theorem probability_correct_guesses:
  let p_wrong := (5/6 : ℚ)
  let p_miss_all := p_wrong ^ 5
  let p_at_least_one_correct := 1 - p_miss_all
  p_at_least_one_correct = 4651/7776 := by
  sorry

end probability_correct_guesses_l183_183963


namespace sum_of_consecutive_numbers_with_lcm_168_l183_183912

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l183_183912


namespace intersection_S_T_eq_T_l183_183121

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183121


namespace triangle_side_ratio_l183_183746

theorem triangle_side_ratio
  (α β γ : Real)
  (a b c p q r : Real)
  (h1 : (Real.tan α) / (Real.tan β) = p / q)
  (h2 : (Real.tan β) / (Real.tan γ) = q / r)
  (h3 : (Real.tan γ) / (Real.tan α) = r / p) :
  a^2 / b^2 / c^2 = (1/q + 1/r) / (1/r + 1/p) / (1/p + 1/q) := 
sorry

end triangle_side_ratio_l183_183746


namespace find_some_number_l183_183709

-- Conditions on operations
axiom plus_means_mult (a b : ℕ) : (a + b) = (a * b)
axiom minus_means_plus (a b : ℕ) : (a - b) = (a + b)
axiom mult_means_div (a b : ℕ) : (a * b) = (a / b)
axiom div_means_minus (a b : ℕ) : (a / b) = (a - b)

-- Problem statement
theorem find_some_number (some_number : ℕ) :
  (6 - 9 + some_number * 3 / 25 = 5 ↔
   6 + 9 * some_number / 3 - 25 = 5) ∧
  some_number = 8 := by
  sorry

end find_some_number_l183_183709


namespace movies_watched_total_l183_183361

theorem movies_watched_total :
  ∀ (Timothy2009 Theresa2009 Timothy2010 Theresa2010 total : ℕ),
    Timothy2009 = 24 →
    Timothy2010 = Timothy2009 + 7 →
    Theresa2010 = 2 * Timothy2010 →
    Theresa2009 = Timothy2009 / 2 →
    total = Timothy2009 + Timothy2010 + Theresa2009 + Theresa2010 →
    total = 129 :=
by
  intros Timothy2009 Theresa2009 Timothy2010 Theresa2010 total
  sorry

end movies_watched_total_l183_183361


namespace only_rotationally_symmetric_curve_is_circle_l183_183179

-- Definitions related to the problem
variable {O : ℝ × ℝ} -- Point O
variable {K : set (ℝ × ℝ)} -- Curve K

-- Conditions
axiom maps_to_itself_under_120_deg_rotation (p : ℝ × ℝ) : (p ∈ K) → 
    (complex.exp (2 * π * complex.I / 3) * (p.1 + p.2 * complex.I) = fst p + snd p * complex.I)

-- Theorem: Curve must be a circle
theorem only_rotationally_symmetric_curve_is_circle (h : ∃ x ∈ K, true) :
 ∃ r : ℝ, ∀ p ∈ K, (complex.abs (complex.mk p.1 p.2 - complex.mk O.1 O.2) = r) := sorry

end only_rotationally_symmetric_curve_is_circle_l183_183179


namespace intersection_S_T_eq_T_l183_183011

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183011


namespace ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l183_183997

theorem ones_digit_largest_power_of_2_divides_32_factorial : 
  (2^31 % 10) = 8 := 
by
  sorry

theorem ones_digit_largest_power_of_3_divides_32_factorial : 
  (3^14 % 10) = 9 := 
by
  sorry

end ones_digit_largest_power_of_2_divides_32_factorial_ones_digit_largest_power_of_3_divides_32_factorial_l183_183997


namespace f_plus_2012_odd_l183_183820

def f : ℝ → ℝ → ℝ := sorry

lemma f_property (α β : ℝ) : f α β = 2012 := sorry

theorem f_plus_2012_odd : ∀ x : ℝ, f (-x) + 2012 = -(f x + 2012) :=
by
  sorry

end f_plus_2012_odd_l183_183820


namespace trigonometric_identity_l183_183372

theorem trigonometric_identity (α β : ℝ) : 
  ((Real.tan α + Real.tan β) / Real.tan (α + β)) 
  + ((Real.tan α - Real.tan β) / Real.tan (α - β)) 
  + 2 * (Real.tan α) ^ 2 
 = 2 / (Real.cos α) ^ 2 :=
  sorry

end trigonometric_identity_l183_183372


namespace count_valid_subsets_l183_183557

open Finset
open Fintype

def is_valid_subset (S : Finset ℕ) : Prop :=
  S.nonempty ∧ ∀ T ⊆ S, T.sum id ≠ 10

theorem count_valid_subsets : ∃ (n : ℕ), n = 34 ∧ (univ.filter is_valid_subset).card = n :=
by
  sorry

end count_valid_subsets_l183_183557


namespace walt_part_time_job_l183_183938

theorem walt_part_time_job (x : ℝ) 
  (h1 : 0.09 * x + 0.08 * 4000 = 770) : 
  x + 4000 = 9000 := by
  sorry

end walt_part_time_job_l183_183938


namespace gcd_of_28430_and_39674_l183_183781

theorem gcd_of_28430_and_39674 : Nat.gcd 28430 39674 = 2 := 
by 
  sorry

end gcd_of_28430_and_39674_l183_183781


namespace malcom_cards_left_l183_183537

theorem malcom_cards_left (brandon_cards : ℕ) (h1 : brandon_cards = 20) (h2 : ∀ malcom_cards : ℕ, malcom_cards = brandon_cards + 8) (h3 : ∀ mark_cards : ℕ, mark_cards = (malcom_cards / 2)) : 
  malcom_cards - mark_cards = 14 := 
by 
  let malcom_cards := 28
  let mark_cards := (malcom_cards / 2)
  have h4 : mark_cards = 14, from rfl
  show malcom_cards - mark_cards = 14, from sorry

end malcom_cards_left_l183_183537


namespace base_prime_rep_360_l183_183262

-- Define the value 360 as n
def n : ℕ := 360

-- Function to compute the base prime representation.
noncomputable def base_prime_representation (n : ℕ) : ℕ :=
  -- Normally you'd implement the actual function to convert n to its base prime representation here
  sorry

-- The theorem statement claiming that the base prime representation of 360 is 213
theorem base_prime_rep_360 : base_prime_representation n = 213 := 
  sorry

end base_prime_rep_360_l183_183262


namespace opening_night_customers_l183_183794

theorem opening_night_customers
  (matinee_tickets : ℝ := 5)
  (evening_tickets : ℝ := 7)
  (opening_night_tickets : ℝ := 10)
  (popcorn_cost : ℝ := 10)
  (num_matinee_customers : ℝ := 32)
  (num_evening_customers : ℝ := 40)
  (total_revenue : ℝ := 1670) :
  ∃ x : ℝ, 
    (matinee_tickets * num_matinee_customers + 
    evening_tickets * num_evening_customers + 
    opening_night_tickets * x + 
    popcorn_cost * (num_matinee_customers + num_evening_customers + x) / 2 = total_revenue) 
    ∧ x = 58 := 
by
  use 58
  sorry

end opening_night_customers_l183_183794


namespace largest_n_binom_l183_183209

theorem largest_n_binom (n : ℕ) (h : 0 ≤ n ∧ n ≤ 11) : 
  (binomial 10 4 + binomial 10 5 = binomial 11 n) → n = 6 :=
by
  sorry

end largest_n_binom_l183_183209


namespace correct_exponentiation_operation_l183_183644

theorem correct_exponentiation_operation (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_exponentiation_operation_l183_183644


namespace missing_fraction_correct_l183_183930

theorem missing_fraction_correct : 
  (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-2 / 15) + (3 / 5) = 0.13333333333333333 :=
by sorry

end missing_fraction_correct_l183_183930


namespace clock_angle_at_3_20_is_160_l183_183955

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l183_183955


namespace largest_n_binomial_l183_183217

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l183_183217


namespace problem_statement_l183_183441

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h₀ : ∀ x, f x = 4 * x + 3) (h₁ : a > 0) (h₂ : b > 0) :
  (∀ x, |f x + 5| < a ↔ |x + 3| < b) ↔ b ≤ a / 4 :=
sorry

end problem_statement_l183_183441


namespace soccer_team_games_l183_183605

theorem soccer_team_games (pizzas : ℕ) (slices_per_pizza : ℕ) (average_goals_per_game : ℕ) (total_games : ℕ) 
  (h1 : pizzas = 6) 
  (h2 : slices_per_pizza = 12) 
  (h3 : average_goals_per_game = 9) 
  (h4 : total_games = (pizzas * slices_per_pizza) / average_goals_per_game) :
  total_games = 8 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end soccer_team_games_l183_183605


namespace bus_total_distance_l183_183788

theorem bus_total_distance
  (distance40 : ℝ)
  (distance60 : ℝ)
  (speed40 : ℝ)
  (speed60 : ℝ)
  (total_time : ℝ)
  (distance40_eq : distance40 = 100)
  (speed40_eq : speed40 = 40)
  (speed60_eq : speed60 = 60)
  (total_time_eq : total_time = 5)
  (time40 : ℝ)
  (time40_eq : time40 = distance40 / speed40)
  (time_equation : time40 + distance60 / speed60 = total_time) :
  distance40 + distance60 = 250 := sorry

end bus_total_distance_l183_183788


namespace probability_at_most_one_hit_l183_183381

theorem probability_at_most_one_hit (p_A : ℝ) (p_B : ℝ) (h_independent : independent A B) 
  (h_p_A : p_A = 0.6) (h_p_B : p_B = 0.7) :
  P (at_most_one_hit A B) = 0.58 := 
sorry

end probability_at_most_one_hit_l183_183381


namespace heptagon_triangulation_count_l183_183303

/-- The number of ways to divide a regular heptagon (7-sided polygon) 
    into 5 triangles using non-intersecting diagonals is 4. -/
theorem heptagon_triangulation_count : ∃ (n : ℕ), n = 4 ∧ ∀ (p : ℕ), (p = 7 ∧ (∀ (k : ℕ), k = 5 → (n = 4))) :=
by {
  -- The proof is non-trivial and omitted here
  sorry
}

end heptagon_triangulation_count_l183_183303


namespace correct_multiplication_result_l183_183786

theorem correct_multiplication_result :
  ∃ x : ℕ, (x * 9 = 153) ∧ (x * 6 = 102) :=
by
  sorry

end correct_multiplication_result_l183_183786


namespace sum_mod_9_l183_183678

theorem sum_mod_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  sorry

end sum_mod_9_l183_183678


namespace largest_n_for_binom_equality_l183_183215

theorem largest_n_for_binom_equality : ∃ (n : ℕ), (∑ i : {i // i = 4 ∨ i = 5}, Nat.choose 10 i) = Nat.choose 11 n ∧ n = 6 :=
by
  sorry

end largest_n_for_binom_equality_l183_183215


namespace silver_medals_count_l183_183968

def total_medals := 67
def gold_medals := 19
def bronze_medals := 16
def silver_medals := total_medals - gold_medals - bronze_medals

theorem silver_medals_count : silver_medals = 32 := by
  -- Proof goes here
  sorry

end silver_medals_count_l183_183968


namespace age_of_15th_student_l183_183760

theorem age_of_15th_student (avg15 : ℕ) (avg7_first : ℕ) (avg7_second : ℕ) : 
  (avg15 = 15) → 
  (avg7_first = 14) → 
  (avg7_second = 16) →
  (let T := 15 * avg15 in
   let sum_first := 7 * avg7_first in
   let sum_second := 7 * avg7_second in
   T - (sum_first + sum_second) = 15) :=
by
  intros h1 h2 h3
  sorry

end age_of_15th_student_l183_183760


namespace largest_n_binom_equality_l183_183214

theorem largest_n_binom_equality :
  ∃ n : ℕ, (binomial 10 4 + binomial 10 5 = binomial 11 n) ∧ ∀ m : ℕ, (binomial 11 m = binomial 11 5) → m ≤ 6 :=
by
  sorry

end largest_n_binom_equality_l183_183214


namespace original_price_of_goods_l183_183881

theorem original_price_of_goods
  (rebate_percent : ℝ := 0.06)
  (tax_percent : ℝ := 0.10)
  (total_paid : ℝ := 6876.1) :
  ∃ P : ℝ, (P - P * rebate_percent) * (1 + tax_percent) = total_paid ∧ P = 6650 :=
sorry

end original_price_of_goods_l183_183881


namespace JessicaPathsAvoidRiskySite_l183_183879

-- Definitions for the conditions.
def West (x y : ℕ) : Prop := (x > 0)
def East (x y : ℕ) : Prop := (x < 4)
def North (x y : ℕ) : Prop := (y < 3)
def AtOrigin (x y : ℕ) : Prop := (x = 0 ∧ y = 0)
def AtAnna (x y : ℕ) : Prop := (x = 4 ∧ y = 3)
def RiskySite (x y : ℕ) : Prop := (x = 2 ∧ y = 1)

-- Function to calculate binomial coefficient, binom(n, k)
def binom : ℕ → ℕ → ℕ
  | n, 0 => 1
  | 0, k + 1 => 0
  | n + 1, k + 1 => binom n k + binom n (k + 1)

-- Number of total valid paths avoiding the risky site.
theorem JessicaPathsAvoidRiskySite :
  let totalPaths := binom 7 4
  let pathsThroughRisky := binom 3 2 * binom 4 2
  (totalPaths - pathsThroughRisky) = 17 :=
by
  sorry

end JessicaPathsAvoidRiskySite_l183_183879


namespace solution_set_of_quadratic_inequality_l183_183495

theorem solution_set_of_quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := 
by
  sorry

end solution_set_of_quadratic_inequality_l183_183495


namespace count_satisfying_n_l183_183135

def isPerfectSquare (m : ℤ) : Prop :=
  ∃ k : ℤ, m = k * k

def countNsatisfying (low high : ℤ) (e : ℤ → ℤ) : ℤ :=
  (Finset.range (Int.natAbs (high - low + 1))).count (λ i, isPerfectSquare (e (low + i)))

theorem count_satisfying_n : countNsatisfying 5 15 (λ n, 2 * n^2 + 3 * n + 2) = 1 :=
by
  sorry

end count_satisfying_n_l183_183135


namespace correct_calculation_l183_183230

theorem correct_calculation :
  (2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5) ∧
  (Real.sqrt 8 / Real.sqrt 2 = 2) ∧
  (5 * Real.sqrt 3 * 5 * Real.sqrt 2 ≠ 5 * Real.sqrt 6) ∧
  (Real.sqrt (4 + 1/2) ≠ 2 * Real.sqrt (1/2)) :=
by
  sorry

end correct_calculation_l183_183230


namespace quadratic_roots_condition_l183_183839

theorem quadratic_roots_condition (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 0) :
  ¬ ((∃ x y : ℝ, ax^2 + 2*x + 1 = 0 ∧ ax^2 + 2*y + 1 = 0 ∧ x*y < 0) ↔
     (a > 0 ∧ a ≠ 0)) :=
by
  sorry

end quadratic_roots_condition_l183_183839


namespace min_isosceles_triangle_area_l183_183759

theorem min_isosceles_triangle_area 
  (x y n : ℕ)
  (h1 : 2 * x * y = 7 * n^2)
  (h2 : ∃ m k, m = n / 2 ∧ k = 2 * m) 
  (h3 : n % 3 = 0) : 
  x = 4 * n / 3 ∧ y = n / 3 ∧ 
  ∃ A, A = 21 / 4 := 
sorry

end min_isosceles_triangle_area_l183_183759


namespace mean_equality_l183_183349

theorem mean_equality (y : ℝ) (h : (6 + 9 + 18) / 3 = (12 + y) / 2) : y = 10 :=
by sorry

end mean_equality_l183_183349


namespace feeding_times_per_day_l183_183750

theorem feeding_times_per_day (p f d : ℕ) (h₁ : p = 7) (h₂ : f = 105) (h₃ : d = 5) : 
  (f / d) / p = 3 := by
  sorry

end feeding_times_per_day_l183_183750


namespace cube_difference_l183_183424

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 :=
sorry

end cube_difference_l183_183424


namespace find_investment_amount_l183_183348

noncomputable def brokerage_fee (market_value : ℚ) : ℚ := (1 / 4 / 100) * market_value

noncomputable def actual_cost (market_value : ℚ) : ℚ := market_value + brokerage_fee market_value

noncomputable def income_per_100_face_value (interest_rate : ℚ) : ℚ := (interest_rate / 100) * 100

noncomputable def investment_amount (income : ℚ) (actual_cost_per_100 : ℚ) (income_per_100 : ℚ) : ℚ :=
  (income * actual_cost_per_100) / income_per_100

theorem find_investment_amount :
  investment_amount 756 (actual_cost 124.75) (income_per_100_face_value 10.5) = 9483.65625 :=
sorry

end find_investment_amount_l183_183348


namespace total_loads_washed_l183_183981

theorem total_loads_washed (a b : ℕ) (h1 : a = 8) (h2 : b = 6) : a + b = 14 :=
by
  sorry

end total_loads_washed_l183_183981


namespace avg_weight_l183_183339

theorem avg_weight (A B C : ℝ)
  (h1 : (A + B) / 2 = 40)
  (h2 : (B + C) / 2 = 43)
  (h3 : B = 31) :
  (A + B + C) / 3 = 45 :=
by sorry

end avg_weight_l183_183339


namespace new_prism_volume_l183_183801

theorem new_prism_volume (L W H : ℝ) 
  (h_volume : L * W * H = 54)
  (L_new : ℝ := 2 * L)
  (W_new : ℝ := 3 * W)
  (H_new : ℝ := 1.5 * H) :
  L_new * W_new * H_new = 486 := 
by
  sorry

end new_prism_volume_l183_183801


namespace half_height_of_triangular_prism_l183_183187

theorem half_height_of_triangular_prism (volume base_area height : ℝ) 
  (h_volume : volume = 576)
  (h_base_area : base_area = 3)
  (h_prism : volume = base_area * height) :
  height / 2 = 96 :=
by
  have h : height = volume / base_area := by sorry
  rw [h_volume, h_base_area] at h
  have h_height : height = 192 := by sorry
  rw [h_height]
  norm_num

end half_height_of_triangular_prism_l183_183187


namespace intersection_of_S_and_T_l183_183107

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183107


namespace ball_radius_l183_183514

theorem ball_radius 
  (r_cylinder : ℝ) (h_rise : ℝ) (v_approx : ℝ)
  (r_cylinder_value : r_cylinder = 12)
  (h_rise_value : h_rise = 6.75)
  (v_approx_value : v_approx = 3053.628) :
  ∃ (r_ball : ℝ), (4 / 3) * Real.pi * r_ball^3 = v_approx ∧ r_ball = 9 := 
by 
  use 9
  sorry

end ball_radius_l183_183514


namespace confetti_left_correct_l183_183828

-- Define the number of pieces of red and green confetti collected by Eunji
def red_confetti : ℕ := 1
def green_confetti : ℕ := 9

-- Define the total number of pieces of confetti collected by Eunji
def total_confetti : ℕ := red_confetti + green_confetti

-- Define the number of pieces of confetti given to Yuna
def given_to_Yuna : ℕ := 4

-- Define the number of pieces of confetti left with Eunji
def confetti_left : ℕ :=  red_confetti + green_confetti - given_to_Yuna

-- Goal to prove
theorem confetti_left_correct : confetti_left = 6 := by
  -- Here the steps proving the equality would go, but we add sorry to skip the proof
  sorry

end confetti_left_correct_l183_183828


namespace sum_consecutive_triangular_sum_triangular_2020_l183_183985

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- The theorem to be proved
theorem sum_consecutive_triangular (n : ℕ) : triangular n + triangular (n + 1) = (n + 1)^2 :=
by 
  sorry

-- Applying the theorem for the specific case of n = 2020
theorem sum_triangular_2020 : triangular 2020 + triangular 2021 = 2021^2 :=
by 
  exact sum_consecutive_triangular 2020

end sum_consecutive_triangular_sum_triangular_2020_l183_183985


namespace probability_is_one_sixth_l183_183147

noncomputable def probability_on_line : ℚ :=
  let A := {0, 1, 2, 3, 4, 5}
  let total_points := (A.product A).card
  let favorable_points := (A.filter (λ (a : ℕ), a ∈ A ∧ a ∈ A)).card
  favorable_points / total_points

theorem probability_is_one_sixth : probability_on_line = 1 / 6 := by
  sorry

end probability_is_one_sixth_l183_183147


namespace common_remainder_zero_l183_183198

theorem common_remainder_zero (n r : ℕ) (h1: n > 1) 
(h2 : n % 25 = r) (h3 : n % 7 = r) (h4 : n = 175) : r = 0 :=
by
  sorry

end common_remainder_zero_l183_183198


namespace solution1_solution2_l183_183563

noncomputable def problem1 (a : ℝ) : Prop :=
  (∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0) ∨
  (∃ x : ℝ, x^2 + (a-1)*x + 4 < 0)

theorem solution1 (a : ℝ) : problem1 a ↔ a < -3 ∨ a ≥ -1 := 
  sorry

noncomputable def problem2 (a : ℝ) (x : ℝ) : Prop :=
  (-2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0)

noncomputable def condition2 (a x : ℝ) : Prop :=
  (2*a < x ∧ x < a+1)

theorem solution2 (a : ℝ) : (∀ x, condition2 a x → problem2 a x) → a ≥ -1/2 :=
  sorry

end solution1_solution2_l183_183563


namespace problem_I_problem_II_l183_183848

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := |x - 2| + |2 * x + a|

-- Problem (I): Inequality solution when a = 1
theorem problem_I (x : ℝ) : f x 1 ≥ 5 ↔ x ∈ (Set.Iic (-4 / 3) ∪ Set.Ici 2) :=
sorry

-- Problem (II): Range of a given the conditions
theorem problem_II (x₀ : ℝ) (a : ℝ) (h : f x₀ a + |x₀ - 2| < 3) : -7 < a ∧ a < -1 :=
sorry

end problem_I_problem_II_l183_183848


namespace number_of_discounted_tickets_l183_183469

def total_tickets : ℕ := 10
def full_price_ticket_cost : ℝ := 2.0
def discounted_ticket_cost : ℝ := 1.6
def total_spent : ℝ := 18.40

theorem number_of_discounted_tickets (F D : ℕ) : 
    F + D = total_tickets → 
    full_price_ticket_cost * ↑F + discounted_ticket_cost * ↑D = total_spent → 
    D = 4 :=
by
  intros h1 h2
  sorry

end number_of_discounted_tickets_l183_183469


namespace inequality_proof_l183_183733

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l183_183733


namespace sum_a2_to_a5_eq_zero_l183_183580

theorem sum_a2_to_a5_eq_zero 
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : ∀ x : ℝ, x * (1 - 2 * x)^4 = a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_2 + a_3 + a_4 + a_5 = 0 :=
sorry

end sum_a2_to_a5_eq_zero_l183_183580


namespace average_daily_visitors_l183_183793

theorem average_daily_visitors
    (avg_sun : ℕ)
    (avg_other : ℕ)
    (days : ℕ)
    (starts_sun : Bool)
    (H1 : avg_sun = 630)
    (H2 : avg_other = 240)
    (H3 : days = 30)
    (H4 : starts_sun = true) :
    (5 * avg_sun + 25 * avg_other) / days = 305 :=
by
  sorry

end average_daily_visitors_l183_183793


namespace lines_through_origin_l183_183462

theorem lines_through_origin (n : ℕ) (h : 0 < n) :
    ∃ S : Finset (ℤ × ℤ), 
    (∀ xy : ℤ × ℤ, xy ∈ S ↔ (0 ≤ xy.1 ∧ xy.1 ≤ n ∧ 0 ≤ xy.2 ∧ xy.2 ≤ n ∧ Int.gcd xy.1 xy.2 = 1)) ∧
    S.card ≥ n^2 / 4 := 
sorry

end lines_through_origin_l183_183462


namespace max_rectangle_area_l183_183977

theorem max_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) (h1 : l + w = 20) (hlw : l = 10 ∨ w = 10) : 
(l = 10 ∧ w = 10 ∧ l * w = 100) :=
by sorry

end max_rectangle_area_l183_183977


namespace min_dist_AB_l183_183592

-- Definitions of the conditions
structure Point3D where
  x : Float
  y : Float
  z : Float

def O := Point3D.mk 0 0 0
def B := Point3D.mk (Float.sqrt 3) (Float.sqrt 2) 2

def dist (P Q : Point3D) : Float :=
  Float.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- Given points
variables (A : Point3D)
axiom AO_eq_1 : dist A O = 1

-- Minimum value of |AB|
theorem min_dist_AB : dist A B ≥ 2 := 
sorry

end min_dist_AB_l183_183592


namespace intersection_eq_T_l183_183036

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183036


namespace simplify_expression_l183_183181

variables {K : Type*} [Field K]

theorem simplify_expression (a b c : K) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) : 
    (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
by
  sorry

end simplify_expression_l183_183181


namespace tangent_line_through_origin_l183_183907

theorem tangent_line_through_origin (f : ℝ → ℝ) (x : ℝ) (H1 : ∀ x < 0, f x = Real.log (-x))
  (H2 : ∀ x < 0, DifferentiableAt ℝ f x) (H3 : ∀ (x₀ : ℝ), x₀ < 0 → x₀ = -Real.exp 1 → deriv f x₀ = -1 / Real.exp 1)
  : ∀ x, -Real.exp 1 = x → ∀ y, y = -1 / Real.exp 1 * x → y = 0 → y = -1 / Real.exp 1 * x :=
by
  sorry

end tangent_line_through_origin_l183_183907


namespace sum_of_consecutive_numbers_LCM_168_l183_183914

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end sum_of_consecutive_numbers_LCM_168_l183_183914


namespace cubes_difference_l183_183430

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l183_183430


namespace smaller_angle_at_3_20_l183_183950

theorem smaller_angle_at_3_20 : 
  let degrees_per_hour : ℝ := 360 / 12
  let degrees_per_minute : ℝ := 360 / 60
  let hour_angle := degrees_per_hour / 60 * 20
  let minute_angle := degrees_per_minute * 20
  let initial_angle := degrees_per_hour * 3
  let total_angle := abs (initial_angle - (hour_angle + minute_angle))
  let smaller_angle := if total_angle > 180 then 360 - total_angle else total_angle
in 
  smaller_angle = 160.0 := 
by
  sorry

end smaller_angle_at_3_20_l183_183950


namespace betty_total_blue_and_green_beads_l183_183677

theorem betty_total_blue_and_green_beads (r b g : ℕ) (h1 : 5 * b = 3 * r) (h2 : 5 * g = 2 * r) (h3 : r = 50) : b + g = 50 :=
by
  sorry

end betty_total_blue_and_green_beads_l183_183677


namespace cows_gift_by_friend_l183_183385

-- Define the base conditions
def initial_cows : Nat := 39
def cows_died : Nat := 25
def cows_sold : Nat := 6
def cows_increase : Nat := 24
def cows_bought : Nat := 43
def final_cows : Nat := 83

-- Define the computation to get the number of cows after each event
def cows_after_died : Nat := initial_cows - cows_died
def cows_after_sold : Nat := cows_after_died - cows_sold
def cows_after_increase : Nat := cows_after_sold + cows_increase
def cows_after_bought : Nat := cows_after_increase + cows_bought

-- Define the proof problem
theorem cows_gift_by_friend : (final_cows - cows_after_bought) = 8 := by
  sorry

end cows_gift_by_friend_l183_183385


namespace inequality_abc_l183_183736

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l183_183736


namespace inequality_proof_l183_183729

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l183_183729


namespace album_count_l183_183739

def albums_total (A B K M C : ℕ) : Prop :=
  A = 30 ∧ B = A - 15 ∧ K = 6 * B ∧ M = 5 * K ∧ C = 3 * M ∧ (A + B + K + M + C) = 1935

theorem album_count (A B K M C : ℕ) : albums_total A B K M C :=
by
  sorry

end album_count_l183_183739


namespace inscribed_sphere_radius_l183_183885

variable (a b r : ℝ)

theorem inscribed_sphere_radius (ha : 0 < a) (hb : 0 < b) (hr : 0 < r)
 (h : ∃ A B C D : ℝˣ, true) : r < (a * b) / (2 * (a + b)) := 
sorry

end inscribed_sphere_radius_l183_183885


namespace greatest_integer_b_for_no_real_roots_l183_183831

theorem greatest_integer_b_for_no_real_roots (b : ℤ) :
  (∀ x : ℝ, x^2 + (b:ℝ)*x + 10 ≠ 0) ↔ b ≤ 6 :=
sorry

end greatest_integer_b_for_no_real_roots_l183_183831


namespace number_of_people_l183_183201

-- Define the total number of candy bars
def total_candy_bars : ℝ := 5.0

-- Define the amount of candy each person gets
def candy_per_person : ℝ := 1.66666666699999

-- Define a theorem to state that dividing the total candy bars by candy per person gives 3 people
theorem number_of_people : total_candy_bars / candy_per_person = 3 :=
  by
  -- Proof omitted
  sorry

end number_of_people_l183_183201


namespace find_s_for_g_eq_0_l183_183258

def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

theorem find_s_for_g_eq_0 : ∃ (s : ℝ), g 3 s = 0 → s = -867 :=
by
  sorry

end find_s_for_g_eq_0_l183_183258


namespace no_rational_multiples_pi_tan_sum_two_l183_183686

theorem no_rational_multiples_pi_tan_sum_two (x y : ℚ) (hx : 0 < x * π ∧ x * π < y * π ∧ y * π < π / 2) (hxy : Real.tan (x * π) + Real.tan (y * π) = 2) : False :=
sorry

end no_rational_multiples_pi_tan_sum_two_l183_183686


namespace cost_of_fencing_each_side_l183_183445

theorem cost_of_fencing_each_side (total_cost : ℕ) (num_sides : ℕ) (h1 : total_cost = 288) (h2 : num_sides = 4) : (total_cost / num_sides) = 72 := by
  sorry

end cost_of_fencing_each_side_l183_183445


namespace find_missing_number_l183_183292

theorem find_missing_number :
  ∀ (x y : ℝ),
    (12 + x + 42 + 78 + 104) / 5 = 62 →
    (128 + y + 511 + 1023 + x) / 5 = 398.2 →
    y = 255 :=
by
  intros x y h1 h2
  sorry

end find_missing_number_l183_183292


namespace roots_of_polynomial_l183_183682

theorem roots_of_polynomial :
  (x^2 - 5 * x + 6) * (x - 1) * (x + 3) = 0 ↔ (x = -3 ∨ x = 1 ∨ x = 2 ∨ x = 3) :=
by {
  sorry
}

end roots_of_polynomial_l183_183682


namespace smallest_four_digit_divisible_by_55_l183_183508

theorem smallest_four_digit_divisible_by_55 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 55 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 55 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_55_l183_183508


namespace susie_total_earnings_l183_183625

def pizza_prices (type : String) (is_whole : Bool) : ℝ :=
  match type, is_whole with
  | "Margherita", false => 3
  | "Margherita", true  => 15
  | "Pepperoni", false  => 4
  | "Pepperoni", true   => 18
  | "Veggie Supreme", false => 5
  | "Veggie Supreme", true  => 22
  | "Meat Lovers", false => 6
  | "Meat Lovers", true  => 25
  | "Hawaiian", false   => 4.5
  | "Hawaiian", true    => 20
  | _, _                => 0

def topping_price (is_weekend : Bool) : ℝ :=
  if is_weekend then 1 else 2

def happy_hour_price : ℝ := 3

noncomputable def susie_earnings : ℝ :=
  let margherita_slices := 12 * happy_hour_price + 12 * pizza_prices "Margherita" false
  let pepperoni_slices := 8 * happy_hour_price + 8 * pizza_prices "Pepperoni" false + 6 * topping_price true
  let veggie_supreme_pizzas := 4 * pizza_prices "Veggie Supreme" true + 8 * topping_price true
  let margherita_whole_discounted := 3 * pizza_prices "Margherita" true - (3 * pizza_prices "Margherita" true) * 0.1
  let meat_lovers_slices := 10 * happy_hour_price + 10 * pizza_prices "Meat Lovers" false
  let hawaiian_slices := 12 * pizza_prices "Hawaiian" false + 4 * topping_price true
  let pepperoni_whole := pizza_prices "Pepperoni" true + 3 * topping_price true
  margherita_slices + pepperoni_slices + veggie_supreme_pizzas + margherita_whole_discounted + meat_lovers_slices + hawaiian_slices + pepperoni_whole

theorem susie_total_earnings : susie_earnings = 439.5 := by
  sorry

end susie_total_earnings_l183_183625


namespace bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l183_183357

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Bose-Einstein distribution, satisfying the given conditions. 
-/
theorem bose_einstein_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 72 := 
  by
  sorry

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Fermi-Dirac distribution, satisfying the given conditions. 
-/
theorem fermi_dirac_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 246 := 
  by
  sorry

end bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l183_183357


namespace intersection_A_B_l183_183132

def setA : Set ℝ := {x | 0 < x}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}
def intersectionAB : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = intersectionAB := by
  sorry

end intersection_A_B_l183_183132


namespace simplification_evaluation_l183_183752

theorem simplification_evaluation (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  ( (2 * x - 6) / (x - 2) ) / ( (5 / (x - 2)) - (x + 2) ) = Real.sqrt 2 - 2 :=
sorry

end simplification_evaluation_l183_183752


namespace point_in_fourth_quadrant_l183_183304

-- Define a structure for a Cartesian point
structure Point where
  x : ℝ
  y : ℝ

-- Define different quadrants
inductive Quadrant
| first
| second
| third
| fourth

-- Function to determine the quadrant of a given point
def quadrant (p : Point) : Quadrant :=
  if p.x > 0 ∧ p.y > 0 then Quadrant.first
  else if p.x < 0 ∧ p.y > 0 then Quadrant.second
  else if p.x < 0 ∧ p.y < 0 then Quadrant.third
  else Quadrant.fourth

-- The main theorem stating the point (3, -4) lies in the fourth quadrant
theorem point_in_fourth_quadrant : quadrant { x := 3, y := -4 } = Quadrant.fourth :=
  sorry

end point_in_fourth_quadrant_l183_183304


namespace fixed_point_l183_183488

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x - 1)

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : f a (1/2) = 1 :=
by
  sorry

end fixed_point_l183_183488


namespace circle_area_l183_183154

open Real

noncomputable def radius_square (x : ℝ) (DE : ℝ) (EF : ℝ) : ℝ :=
  let DE_square := DE^2
  let r_square_1 := x^2 + DE_square
  let product_DE_EF := DE * EF
  let r_square_2 := product_DE_EF + x^2
  r_square_2

theorem circle_area (x : ℝ) (h1 : OE = x) (h2 : DE = 8) (h3 : EF = 4) :
  π * radius_square x 8 4 = 96 * π :=
by
  sorry

end circle_area_l183_183154


namespace intersection_S_T_eq_T_l183_183068

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183068


namespace game_A_vs_game_B_l183_183971

-- Define the problem in Lean 4
theorem game_A_vs_game_B (p_head : ℝ) (p_tail : ℝ) (independent : Prop)
  (prob_A : ℝ) (prob_B : ℝ) (delta : ℝ) :
  p_head = 3/4 → p_tail = 1/4 → independent →
  prob_A = (binom 4 3) * (p_head ^ 3) * (p_tail ^ 1) + (p_head ^ 4) →
  prob_B = ((p_head ^ 2 + p_tail ^ 2) ^ 2) →
  delta = prob_A - prob_B →
  delta = 89/256 :=
by
  intros hph hpt hind hpa hpb hdelta
  rw [hph, hpt, hpa, hpb, hdelta]
  sorry

end game_A_vs_game_B_l183_183971


namespace no_integer_solution_l183_183350

theorem no_integer_solution (m n : ℤ) : m^2 - 11 * m * n - 8 * n^2 ≠ 88 :=
sorry

end no_integer_solution_l183_183350


namespace product_of_last_two_digits_l183_183582

theorem product_of_last_two_digits (A B : ℕ) (hn1 : 10 * A + B ≡ 0 [MOD 5]) (hn2 : A + B = 16) : A * B = 30 :=
sorry

end product_of_last_two_digits_l183_183582


namespace northton_time_capsule_depth_l183_183184

def southton_depth : ℕ := 15

def northton_depth : ℕ := 4 * southton_depth + 12

theorem northton_time_capsule_depth : northton_depth = 72 := by
  sorry

end northton_time_capsule_depth_l183_183184


namespace intersection_three_points_l183_183642

def circle_eq (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = a^2
def parabola_eq (a : ℝ) (x y : ℝ) : Prop := y = x^2 - 3 * a

theorem intersection_three_points (a : ℝ) :
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    circle_eq a x1 y1 ∧ parabola_eq a x1 y1 ∧
    circle_eq a x2 y2 ∧ parabola_eq a x2 y2 ∧
    circle_eq a x3 y3 ∧ parabola_eq a x3 y3 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    (x1 ≠ x3 ∨ y1 ≠ y3) ∧
    (x2 ≠ x3 ∨ y2 ≠ y3)) ↔ 
  a = 1/3 := by
  sorry

end intersection_three_points_l183_183642


namespace intersection_S_T_l183_183071

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183071


namespace completing_square_correctness_l183_183335

theorem completing_square_correctness :
  (2 * x^2 - 4 * x - 7 = 0) ->
  ((x - 1)^2 = 9 / 2) :=
sorry

end completing_square_correctness_l183_183335


namespace opposite_reciprocal_of_neg_five_l183_183492

theorem opposite_reciprocal_of_neg_five : 
  ∀ x : ℝ, x = -5 → - (1 / x) = 1 / 5 :=
by
  sorry

end opposite_reciprocal_of_neg_five_l183_183492


namespace total_movies_attended_l183_183360

-- Defining the conditions for Timothy's movie attendance
def Timothy_2009 := 24
def Timothy_2010 := Timothy_2009 + 7

-- Defining the conditions for Theresa's movie attendance
def Theresa_2009 := Timothy_2009 / 2
def Theresa_2010 := Timothy_2010 * 2

-- Prove that the total number of movies Timothy and Theresa went to in both years is 129
theorem total_movies_attended :
  (Timothy_2009 + Timothy_2010 + Theresa_2009 + Theresa_2010) = 129 :=
by
  -- proof goes here
  sorry

end total_movies_attended_l183_183360


namespace div_by_66_l183_183472

theorem div_by_66 :
  (43 ^ 23 + 23 ^ 43) % 66 = 0 := 
sorry

end div_by_66_l183_183472


namespace intersection_S_T_l183_183080

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183080


namespace total_number_of_fish_l183_183497

theorem total_number_of_fish (fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1: fishbowls = 261) (h2: fish_per_bowl = 23) : 
  fishbowls * fish_per_bowl = 6003 := 
by
  sorry

end total_number_of_fish_l183_183497


namespace cuboid_length_l183_183833

theorem cuboid_length (A b h : ℝ) (A_eq : A = 2400) (b_eq : b = 10) (h_eq : h = 16) :
    ∃ l : ℝ, 2 * (l * b + b * h + h * l) = A ∧ l = 40 := by
  sorry

end cuboid_length_l183_183833


namespace gcd_442872_312750_l183_183636

theorem gcd_442872_312750 : Nat.gcd 442872 312750 = 18 :=
by
  sorry

end gcd_442872_312750_l183_183636


namespace copper_content_range_l183_183358

theorem copper_content_range (x2 : ℝ) (y : ℝ) (h1 : 0 ≤ x2) (h2 : x2 ≤ 4 / 9) (hy : y = 0.4 + 0.075 * x2) : 
  40 ≤ 100 * y ∧ 100 * y ≤ 130 / 3 :=
by { sorry }

end copper_content_range_l183_183358


namespace inequality_abc_l183_183735

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l183_183735


namespace math_equivalence_proof_l183_183520

noncomputable def problem_conditions : Prop :=
∀ (A B C D E : ℝ) (circle : ℝ),
  AB = 5 ∧ BC = 5 ∧ CD = 5 ∧ DE = 5 ∧ AE = 2

theorem math_equivalence_proof :
  problem_conditions → 
  (1 - real.cos angle_B) * (1 - real.cos angle_ACE) = (1/25) :=
by
  intro h
  sorry

end math_equivalence_proof_l183_183520


namespace discount_percentage_l183_183961

theorem discount_percentage (original_price sale_price : ℝ) (h_original : original_price = 150) (h_sale : sale_price = 135) :
  ((original_price - sale_price) / original_price) * 100 = 10 :=
by
  -- Original price is 150
  rw h_original
  -- Sale price is 135
  rw h_sale
  -- Calculate the discount
  norm_num
  -- Prove the final percentage
  norm_num
  trivial
  sorry

end discount_percentage_l183_183961


namespace distance_and_ratio_correct_l183_183554

noncomputable def distance_and_ratio (a : ℝ) : ℝ × ℝ :=
  let dist : ℝ := a / Real.sqrt 3
  let ratio : ℝ := 1 / 2
  ⟨dist, ratio⟩

theorem distance_and_ratio_correct (a : ℝ) :
  distance_and_ratio a = (a / Real.sqrt 3, 1 / 2) := by
  -- Proof omitted
  sorry

end distance_and_ratio_correct_l183_183554


namespace intersection_S_T_eq_T_l183_183005

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183005


namespace johnny_worked_hours_l183_183882

theorem johnny_worked_hours (total_earned hourly_wage hours_worked : ℝ) 
(h1 : total_earned = 16.5) (h2 : hourly_wage = 8.25) (h3 : total_earned / hourly_wage = hours_worked) : 
hours_worked = 2 := 
sorry

end johnny_worked_hours_l183_183882


namespace number_of_club_members_l183_183526

theorem number_of_club_members
  (num_committee : ℕ)
  (pair_of_committees_has_unique_member : ∀ (c1 c2 : Fin num_committee), c1 ≠ c2 → ∃! m : ℕ, c1 ≠ c2 ∧ c2 ≠ c1 ∧ m = m)
  (members_belong_to_two_committees : ∀ m : ℕ, ∃ (c1 c2 : Fin num_committee), c1 ≠ c2 ∧ m = m)
  : num_committee = 5 → ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end number_of_club_members_l183_183526


namespace joan_already_put_in_cups_l183_183595

def recipe_cups : ℕ := 7
def cups_needed : ℕ := 4

theorem joan_already_put_in_cups : (recipe_cups - cups_needed = 3) :=
by
  sorry

end joan_already_put_in_cups_l183_183595


namespace intersection_S_T_eq_T_l183_183096

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183096


namespace non_congruent_right_triangles_count_l183_183287

def isRightTriangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

def areaEqualsFourTimesPerimeter (a b c : ℕ) : Prop :=
  a * b = 8 * (a + b + c)

theorem non_congruent_right_triangles_count :
  {n : ℕ // ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ isRightTriangle a b c ∧ areaEqualsFourTimesPerimeter a b c ∧ n = 3} := sorry

end non_congruent_right_triangles_count_l183_183287


namespace intersection_S_T_l183_183047

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183047


namespace diana_can_paint_statues_l183_183263

theorem diana_can_paint_statues :
  ∀ (paint_remaining paint_per_statue : ℚ), paint_remaining = 7 / 8 → paint_per_statue = 1 / 8 → 
  paint_remaining / paint_per_statue = 7 :=
by
  intros paint_remaining paint_per_statue h_remaining h_per_statue
  rw [h_remaining, h_per_statue]
  norm_num
  sorry

end diana_can_paint_statues_l183_183263


namespace part1_monotonicity_part2_inequality_l183_183853

variable (a : ℝ) (x : ℝ)
variable (h : a > 0)

def f := a * (Real.exp x + a) - x

theorem part1_monotonicity :
  if a ≤ 0 then ∀ x : ℝ, (f a x) < (f a (x + 1)) else
  if a > 0 then ∀ x : ℝ, (x < Real.log (1 / a) → (f a x) > (f a (x + 1))) ∧ 
                       (x > Real.log (1 / a) → (f a x) < (f a (x - 1))) else 
  (False) := sorry

theorem part2_inequality :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3 / 2 := sorry

end part1_monotonicity_part2_inequality_l183_183853


namespace remainder_when_divided_by_8_l183_183238

theorem remainder_when_divided_by_8 (k : ℤ) : ((63 * k + 25) % 8) = 1 := 
by sorry

end remainder_when_divided_by_8_l183_183238


namespace pizza_non_crust_percentage_l183_183662

theorem pizza_non_crust_percentage (total_weight crust_weight : ℕ) (h₁ : total_weight = 200) (h₂ : crust_weight = 50) :
  (total_weight - crust_weight) * 100 / total_weight = 75 :=
by
  sorry

end pizza_non_crust_percentage_l183_183662


namespace angles_of_terminal_side_on_line_y_equals_x_l183_183493

noncomputable def set_of_angles_on_y_equals_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 180 + 45

theorem angles_of_terminal_side_on_line_y_equals_x (α : ℝ) :
  (∃ k : ℤ, α = k * 360 + 45) ∨ (∃ k : ℤ, α = k * 360 + 225) ↔ set_of_angles_on_y_equals_x α :=
by
  sorry

end angles_of_terminal_side_on_line_y_equals_x_l183_183493


namespace investor_pieces_impossible_to_be_2002_l183_183244

theorem investor_pieces_impossible_to_be_2002 : 
  ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := 
by
  sorry

end investor_pieces_impossible_to_be_2002_l183_183244


namespace abs_neg_one_eq_one_l183_183758

theorem abs_neg_one_eq_one : abs (-1 : ℚ) = 1 := 
by
  sorry

end abs_neg_one_eq_one_l183_183758


namespace intersection_S_T_eq_T_l183_183113

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183113


namespace max_value_of_y_l183_183294

open Classical

noncomputable def satisfies_equation (x y : ℝ) : Prop := y * x * (x + y) = x - y

theorem max_value_of_y : 
  ∀ (y : ℝ), (∃ (x : ℝ), x > 0 ∧ satisfies_equation x y) → y ≤ 1 / 3 := 
sorry

end max_value_of_y_l183_183294


namespace expression_equals_5776_l183_183239

-- Define constants used in the problem
def a : ℕ := 476
def b : ℕ := 424
def c : ℕ := 4

-- Define the expression using the constants
def expression : ℕ := (a + b) ^ 2 - c * a * b

-- The target proof statement
theorem expression_equals_5776 : expression = 5776 := by
  sorry

end expression_equals_5776_l183_183239


namespace time_to_pass_bridge_l183_183236

noncomputable def train_length : Real := 357
noncomputable def speed_km_per_hour : Real := 42
noncomputable def bridge_length : Real := 137

noncomputable def speed_m_per_s : Real := speed_km_per_hour * (1000 / 3600)

noncomputable def total_distance : Real := train_length + bridge_length

noncomputable def time_to_pass : Real := total_distance / speed_m_per_s

theorem time_to_pass_bridge : abs (time_to_pass - 42.33) < 0.01 :=
sorry

end time_to_pass_bridge_l183_183236


namespace subset_condition_l183_183402

theorem subset_condition (a : ℝ) :
  (∀ x : ℝ, |2 * x - 1| < 1 → x^2 - 2 * a * x + a^2 - 1 > 0) →
  (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end subset_condition_l183_183402


namespace max_squares_at_a1_bksq_l183_183770

noncomputable def maximizePerfectSquares (a b : ℕ) : Prop := 
a ≠ b ∧ 
(∃ k : ℕ, k ≠ 1 ∧ b = k^2) ∧ 
a = 1

theorem max_squares_at_a1_bksq (a b : ℕ) : maximizePerfectSquares a b := 
by 
  sorry

end max_squares_at_a1_bksq_l183_183770


namespace inequality_abc_l183_183737

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end inequality_abc_l183_183737


namespace Maddie_bought_two_white_packs_l183_183468

theorem Maddie_bought_two_white_packs 
  (W : ℕ)
  (total_cost : ℕ)
  (cost_per_shirt : ℕ)
  (white_pack_size : ℕ)
  (blue_pack_size : ℕ)
  (blue_packs : ℕ)
  (cost_per_white_pack : ℕ)
  (cost_per_blue_pack : ℕ) :
  total_cost = 66 ∧ cost_per_shirt = 3 ∧ white_pack_size = 5 ∧ blue_pack_size = 3 ∧ blue_packs = 4 ∧ cost_per_white_pack = white_pack_size * cost_per_shirt ∧ cost_per_blue_pack = blue_pack_size * cost_per_shirt ∧ 3 * (white_pack_size * W + blue_pack_size * blue_packs) = total_cost → W = 2 :=
by
  sorry

end Maddie_bought_two_white_packs_l183_183468


namespace intersection_of_S_and_T_l183_183100

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183100


namespace race_dead_heat_l183_183647

theorem race_dead_heat 
  (L Vb : ℝ) 
  (speed_a : ℝ := (16/15) * Vb)
  (speed_c : ℝ := (20/15) * Vb) 
  (time_a : ℝ := L / speed_a)
  (time_b : ℝ := L / Vb)
  (time_c : ℝ := L / speed_c) :
  (1 / (16 / 15) = 3 / 4) → 
  (1 - 3 / 4) = 1 / 4 :=
by 
  sorry

end race_dead_heat_l183_183647


namespace mark_more_hours_than_kate_l183_183743

-- Definitions for the problem
variable (K : ℕ)  -- K is the number of hours charged by Kate
variable (P : ℕ)  -- P is the number of hours charged by Pat
variable (M : ℕ)  -- M is the number of hours charged by Mark

-- Conditions
def total_hours := K + P + M = 216
def pat_kate_relation := P = 2 * K
def pat_mark_relation := P = (1 / 3) * M

-- The statement to be proved
theorem mark_more_hours_than_kate (K P M : ℕ) (h1 : total_hours K P M)
  (h2 : pat_kate_relation K P) (h3 : pat_mark_relation P M) :
  (M - K = 120) :=
by
  sorry

end mark_more_hours_than_kate_l183_183743


namespace medium_stores_count_l183_183868

-- Define the total number of stores
def total_stores : ℕ := 300

-- Define the number of medium stores
def medium_stores : ℕ := 75

-- Define the sample size
def sample_size : ℕ := 20

-- Define the expected number of medium stores in the sample
def expected_medium_stores : ℕ := 5

-- The theorem statement claiming that the number of medium stores in the sample is 5
theorem medium_stores_count : 
  (sample_size * medium_stores) / total_stores = expected_medium_stores :=
by
  -- Proof omitted
  sorry

end medium_stores_count_l183_183868


namespace units_digit_of_expression_l183_183548

theorem units_digit_of_expression :
  (3 * 19 * 1981 - 3^4) % 10 = 6 :=
sorry

end units_digit_of_expression_l183_183548


namespace subtraction_to_nearest_thousandth_l183_183185

theorem subtraction_to_nearest_thousandth : 
  (456.789 : ℝ) - (234.567 : ℝ) = 222.222 :=
by
  sorry

end subtraction_to_nearest_thousandth_l183_183185


namespace sequence_eq_l183_183277

-- Define the sequence and the conditions
def is_sequence (a : ℕ → ℕ) :=
  (∀ i, a i > 0) ∧ (∀ i j, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j)

-- The theorem we want to prove: for all i, a_i = i
theorem sequence_eq (a : ℕ → ℕ) (h : is_sequence a) : ∀ i, a i = i :=
by
  sorry

end sequence_eq_l183_183277


namespace a_100_correct_l183_183283

variable (a_n : ℕ → ℕ) (S₉ : ℕ) (a₁₀ : ℕ)

def is_arth_seq (a_n : ℕ → ℕ) := ∃ a d, ∀ n, a_n n = a + n * d

noncomputable def a_100 (a₅ d : ℕ) : ℕ := a₅ + 95 * d

theorem a_100_correct
  (h1 : ∃ S₉, 9 * a_n 4 = S₉)
  (h2 : a_n 9 = 8)
  (h3 : is_arth_seq a_n) :
  a_100 (a_n 4) 1 = 98 :=
by
  sorry

end a_100_correct_l183_183283


namespace cubes_difference_l183_183431

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l183_183431


namespace intersection_S_T_l183_183025

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183025


namespace exists_b_c_with_integral_roots_l183_183264

theorem exists_b_c_with_integral_roots :
  ∃ (b c : ℝ), (∃ (p q : ℤ), (x^2 + b * x + c = 0) ∧ (x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
               ((x - p) * (x - q) = x^2 - (p + q) * x + p*q)) ∧
              (∃ (r s : ℤ), (x^2 + (b+1) * x + (c+1) = 0) ∧ 
              ((x - r) * (x - s) = x^2 - (r + s) * x + r*s)) :=
by
  sorry

end exists_b_c_with_integral_roots_l183_183264


namespace rows_before_change_l183_183247

-- Definitions and conditions
variables {r c : ℕ}

-- The total number of tiles before and after the change
def total_tiles_before (r c : ℕ) := r * c = 30
def total_tiles_after (r c : ℕ) := (r + 4) * (c - 2) = 30

-- Prove that the number of rows before the change is 3
theorem rows_before_change (h1 : total_tiles_before r c) (h2 : total_tiles_after r c) : r = 3 := 
sorry

end rows_before_change_l183_183247


namespace three_pow_2023_mod_eleven_l183_183776

theorem three_pow_2023_mod_eleven :
  (3 ^ 2023) % 11 = 5 :=
sorry

end three_pow_2023_mod_eleven_l183_183776


namespace arithmetic_sequence_a7_l183_183302

theorem arithmetic_sequence_a7 (S_13 : ℕ → ℕ → ℕ) (n : ℕ) (a7 : ℕ) (h1: S_13 13 52 = 52) (h2: S_13 13 a7 = 13 * a7):
  a7 = 4 :=
by
  sorry

end arithmetic_sequence_a7_l183_183302


namespace cost_of_apple_l183_183334

variable (A O : ℝ)

theorem cost_of_apple :
  (6 * A + 3 * O = 1.77) ∧ (2 * A + 5 * O = 1.27) → A = 0.21 :=
by
  intro h
  -- Proof goes here
  sorry

end cost_of_apple_l183_183334


namespace graph_passes_through_fixed_point_l183_183346

theorem graph_passes_through_fixed_point (a : ℝ) : (0, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a ^ x + 1) } :=
sorry

end graph_passes_through_fixed_point_l183_183346


namespace intersection_S_T_eq_T_l183_183061

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183061


namespace intersection_S_T_l183_183054

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183054


namespace number_of_votes_for_winner_l183_183651

-- Define the conditions
def total_votes : ℝ := 1000
def winner_percentage : ℝ := 0.55
def margin_of_victory : ℝ := 100

-- The statement to prove
theorem number_of_votes_for_winner :
  0.55 * total_votes = 550 :=
by
  -- We are supposed to provide the proof but it's skipped here
  sorry

end number_of_votes_for_winner_l183_183651


namespace cos_double_angle_l183_183843

theorem cos_double_angle (α : ℝ) (h : Real.sin α = 3 / 5) : Real.cos (2 * α) = 7 / 25 :=
sorry

end cos_double_angle_l183_183843


namespace tom_total_amount_l183_183461

-- Definitions of the initial conditions
def initial_amount : ℕ := 74
def amount_earned : ℕ := 86

-- Main statement to prove
theorem tom_total_amount : initial_amount + amount_earned = 160 := 
by
  -- sorry added to skip the proof
  sorry

end tom_total_amount_l183_183461


namespace least_number_subtracted_l183_183228

theorem least_number_subtracted (n m : ℕ) (h₁ : m = 2590) (h₂ : n = 2590 - 16) :
  (n % 9 = 6) ∧ (n % 11 = 6) ∧ (n % 13 = 6) :=
by
  sorry

end least_number_subtracted_l183_183228


namespace trig_problems_l183_183588

variable {A B C : ℝ}
variable {a b c : ℝ}

-- The main theorem statement to prove the magnitude of angle B and find b under given conditions.
theorem trig_problems
  (h₁ : (2 * a - c) * Real.cos B = b * Real.cos C)
  (h₂ : a = Real.sqrt 3)
  (h₃ : c = Real.sqrt 3) :
  Real.cos B = 1 / 2 ∧ b = Real.sqrt 3 := by
sorry

end trig_problems_l183_183588


namespace minimum_value_nine_l183_183599

noncomputable def min_value (a b c k : ℝ) : ℝ :=
  (a + b + k) / c + (a + c + k) / b + (b + c + k) / a

theorem minimum_value_nine (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  min_value a b c k ≥ 9 :=
sorry

end minimum_value_nine_l183_183599


namespace find_number_l183_183587

noncomputable def x : ℝ :=
  36 / (Real.sqrt 0.85 * (2 / 3 * Real.pi))

theorem find_number : x ≈ 18.656854 :=
by
  have h1 : Real.sqrt 0.85 ≈ 0.921954 := sorry
  have h2 : 2 / 3 * Real.pi ≈ 2.094395 := sorry
  have h3 : Real.sqrt 0.85 * (2 / 3 * Real.pi) ≈ 1.930297 := sorry
  show x ≈ 18.656854 from
    calc
      x = 36 / (Real.sqrt 0.85 * (2 / 3 * Real.pi)) : rfl
      ... ≈ 36 / 1.930297 : by { rw [h3] }
      ... ≈ 18.656854    : sorry

end find_number_l183_183587


namespace solve_inequalities_l183_183860

theorem solve_inequalities (a b : ℝ) :
  (∀ x : ℝ, -1 < x ∧ x < 3 → x - a < 1 ∧ x - 2 * b > 3) ↔ (a = 2 ∧ b = -2) := 
  by 
    sorry

end solve_inequalities_l183_183860


namespace smaller_angle_at_3_20_correct_l183_183956

noncomputable def smaller_angle_at_3_20 (angle_3_00 : ℝ)
  (minute_hand_rate : ℝ)
  (hour_hand_rate : ℝ) : ℝ :=
  let angle_change := (3.20 - 3.00) * (minute_hand_rate - hour_hand_rate)
  let total_angle := angle_3_00 + angle_change
  let smaller_angle := if total_angle <= 180 then total_angle else 360 - total_angle
  smaller_angle

theorem smaller_angle_at_3_20_correct :
  smaller_angle_at_3_20 90 6 0.5 = 160.0 :=
by
  sorry

end smaller_angle_at_3_20_correct_l183_183956


namespace intersection_S_T_eq_T_l183_183088

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183088


namespace intersection_S_T_l183_183013

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183013


namespace total_beds_in_hotel_l183_183809

theorem total_beds_in_hotel (total_rooms : ℕ) (rooms_two_beds rooms_three_beds : ℕ) (beds_two beds_three : ℕ) 
  (h1 : total_rooms = 13) 
  (h2 : rooms_two_beds = 8) 
  (h3 : rooms_three_beds = total_rooms - rooms_two_beds) 
  (h4 : beds_two = 2) 
  (h5 : beds_three = 3) : 
  rooms_two_beds * beds_two + rooms_three_beds * beds_three = 31 :=
by
  sorry

end total_beds_in_hotel_l183_183809


namespace clock_angle_at_3_20_l183_183953

theorem clock_angle_at_3_20 
  (hour_hand_3oclock : ℝ := 90)
  (minute_hand_3oclock : ℝ := 0)
  (minute_hand_per_min : ℝ := 6)
  (hour_hand_per_min : ℝ := 0.5)
  (minutes_passed : ℝ := 20) :
  let minute_hand_position := minute_hand_3oclock + minute_hand_per_min * minutes_passed in
  let hour_hand_position := hour_hand_3oclock + hour_hand_per_min * minutes_passed in
  let angle_between := minute_hand_position - hour_hand_position in
  angle_between = 20.0 :=
by
  sorry

end clock_angle_at_3_20_l183_183953


namespace sum_of_consecutive_numbers_with_lcm_168_l183_183911

theorem sum_of_consecutive_numbers_with_lcm_168 (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : Nat.lcm a (Nat.lcm b c) = 168) : a + b + c = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l183_183911


namespace inequality_proof_l183_183731

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l183_183731


namespace intersection_S_T_eq_T_l183_183010

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183010


namespace clock_angle_at_3_20_l183_183948

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l183_183948


namespace final_weight_is_200_l183_183724

def initial_weight : ℕ := 220
def percentage_lost : ℕ := 10
def weight_gained : ℕ := 2

theorem final_weight_is_200 :
  initial_weight - (initial_weight * percentage_lost / 100) + weight_gained = 200 := by
  sorry

end final_weight_is_200_l183_183724


namespace counted_integer_twice_l183_183719

theorem counted_integer_twice (x n : ℕ) (hn : n = 100) 
  (h_sum : (n * (n + 1)) / 2 + x = 5053) : x = 3 := by
  sorry

end counted_integer_twice_l183_183719


namespace AB_not_together_correct_l183_183498

-- Definitions based on conditions
def total_people : ℕ := 5

-- The result from the complementary counting principle
def total_arrangements : ℕ := 120
def AB_together_arrangements : ℕ := 48

-- The arrangement count of A and B not next to each other
def AB_not_together_arrangements : ℕ := total_arrangements - AB_together_arrangements

theorem AB_not_together_correct : 
  AB_not_together_arrangements = 72 :=
sorry

end AB_not_together_correct_l183_183498


namespace distance_from_tangency_to_tangent_theorem_l183_183271

noncomputable def distance_from_tangency_to_tangent (R r : ℝ) : ℝ :=
  2 * R * r / (R + r)

theorem distance_from_tangency_to_tangent_theorem (R r : ℝ) :
  ∃ d : ℝ, d = distance_from_tangency_to_tangent R r :=
by
  use 2 * R * r / (R + r)
  sorry

end distance_from_tangency_to_tangent_theorem_l183_183271


namespace largest_n_for_binomial_equality_l183_183223

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l183_183223


namespace number_of_observations_l183_183500

theorem number_of_observations (n : ℕ) (h1 : 200 - 6 = 194) (h2 : 200 * n - n * 6 = n * 194) :
  n > 0 :=
by
  sorry

end number_of_observations_l183_183500


namespace count_oddly_powerful_integers_l183_183261

def is_oddly_powerful (m : ℕ) : Prop :=
  ∃ (c d : ℕ), d > 1 ∧ d % 2 = 1 ∧ c^d = m

theorem count_oddly_powerful_integers :
  ∃ (S : Finset ℕ), 
  (∀ m, m ∈ S ↔ (m < 1500 ∧ is_oddly_powerful m)) ∧ S.card = 13 :=
by
  sorry

end count_oddly_powerful_integers_l183_183261


namespace domain_of_f_l183_183191

theorem domain_of_f :
  {x : ℝ | x > -1 ∧ x ≠ 0 ∧ x ≤ 2} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l183_183191


namespace james_daily_soda_consumption_l183_183162

theorem james_daily_soda_consumption
  (N_p : ℕ) -- number of packs
  (S_p : ℕ) -- sodas per pack
  (S_i : ℕ) -- initial sodas
  (D : ℕ)  -- days in a week
  (h1 : N_p = 5)
  (h2 : S_p = 12)
  (h3 : S_i = 10)
  (h4 : D = 7) : 
  (N_p * S_p + S_i) / D = 10 := 
by 
  sorry

end james_daily_soda_consumption_l183_183162


namespace total_beds_in_hotel_l183_183810

theorem total_beds_in_hotel (total_rooms : ℕ) (rooms_two_beds rooms_three_beds : ℕ) (beds_two beds_three : ℕ) 
  (h1 : total_rooms = 13) 
  (h2 : rooms_two_beds = 8) 
  (h3 : rooms_three_beds = total_rooms - rooms_two_beds) 
  (h4 : beds_two = 2) 
  (h5 : beds_three = 3) : 
  rooms_two_beds * beds_two + rooms_three_beds * beds_three = 31 :=
by
  sorry

end total_beds_in_hotel_l183_183810


namespace intersection_S_T_l183_183048

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183048


namespace minimum_omega_l183_183320

noncomputable def f (omega phi x : ℝ) : ℝ := Real.sin (omega * x + phi)

theorem minimum_omega {omega : ℝ} (h_pos : omega > 0) (h_even : ∀ x : ℝ, f omega (Real.pi / 2) x = f omega (Real.pi / 2) (-x)) 
  (h_zero_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f omega (Real.pi / 2) x = 0) :
  omega ≥ 1 / 2 :=
sorry

end minimum_omega_l183_183320


namespace slope_angle_line_l183_183569
open Real

theorem slope_angle_line (x y : ℝ) :
  x + sqrt 3 * y - 1 = 0 → ∃ θ : ℝ, θ = 150 ∧
  ∃ (m : ℝ), m = -sqrt 3 / 3 ∧ θ = arctan m :=
by
  sorry

end slope_angle_line_l183_183569


namespace theon_speed_l183_183629

theorem theon_speed (VTheon VYara D : ℕ) (h1 : VYara = 30) (h2 : D = 90) (h3 : D / VTheon = D / VYara + 3) : VTheon = 15 := by
  sorry

end theon_speed_l183_183629


namespace chessboard_tiling_impossible_l183_183297

theorem chessboard_tiling_impossible :
  ¬ ∃ (cover : (Fin 5 × Fin 7 → Prop)), 
    (cover (0, 3) = false) ∧
    (∀ i j, (cover (i, j) → cover (i + 1, j) ∨ cover (i, j + 1)) ∧
             ∀ x y z w, cover (x, y) → cover (z, w) → (x ≠ z ∨ y ≠ w)) :=
sorry

end chessboard_tiling_impossible_l183_183297


namespace sandy_age_l183_183892

variable (S M N : ℕ)

theorem sandy_age (h1 : M = S + 20)
                  (h2 : (S : ℚ) / M = 7 / 9)
                  (h3 : S + M + N = 120)
                  (h4 : N - M = (S - M) / 2) :
                  S = 70 := 
sorry

end sandy_age_l183_183892


namespace Heath_current_age_l183_183710

variable (H J : ℕ) -- Declare variables for Heath's and Jude's ages
variable (h1 : J = 2) -- Jude's current age is 2
variable (h2 : H + 5 = 3 * (J + 5)) -- In 5 years, Heath will be 3 times as old as Jude

theorem Heath_current_age : H = 16 :=
by
  -- Proof to be filled in later
  sorry

end Heath_current_age_l183_183710


namespace intersection_S_T_eq_T_l183_183002

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183002


namespace starting_player_wins_by_taking_2_white_first_l183_183152

-- Define initial setup
def initial_blue_balls : ℕ := 15
def initial_white_balls : ℕ := 12

-- Define conditions of the game
def can_take_blue_balls (n : ℕ) : Prop := n % 3 = 0
def can_take_white_balls (n : ℕ) : Prop := n % 2 = 0
def player_win_condition (blue white : ℕ) : Prop := 
  (blue = 0 ∧ white = 0)

-- Define the game strategy to establish and maintain the ratio 3/2
def maintain_ratio (blue white : ℕ) : Prop := blue * 2 = white * 3

-- Prove that the starting player should take 2 white balls first to ensure winning
theorem starting_player_wins_by_taking_2_white_first :
  (can_take_white_balls 2) →
  maintain_ratio initial_blue_balls (initial_white_balls - 2) →
  ∀ (blue white : ℕ), player_win_condition blue white :=
by
  intros h_take_white h_maintain_ratio blue white
  sorry

end starting_player_wins_by_taking_2_white_first_l183_183152


namespace students_on_bus_l183_183933

theorem students_on_bus (initial_students : ℝ) (students_got_on : ℝ) (total_students : ℝ) 
  (h1 : initial_students = 10.0) (h2 : students_got_on = 3.0) : 
  total_students = 13.0 :=
by 
  sorry

end students_on_bus_l183_183933


namespace clock_angle_at_3_20_is_160_l183_183954

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end clock_angle_at_3_20_is_160_l183_183954


namespace intersection_S_T_eq_T_l183_183009

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183009


namespace sum_of_roots_l183_183256

theorem sum_of_roots (x : ℝ) :
  (3 * x - 2) * (x - 3) + (3 * x - 2) * (2 * x - 8) = 0 ->
  x = 2 / 3 ∨ x = 11 / 3 ->
  (2 / 3) + (11 / 3) = 13 / 3 :=
by
  sorry

end sum_of_roots_l183_183256


namespace weightlifter_total_weight_l183_183669

theorem weightlifter_total_weight (weight_one_hand : ℕ) (num_hands : ℕ) (condition: weight_one_hand = 8 ∧ num_hands = 2) :
  2 * weight_one_hand = 16 :=
by
  sorry

end weightlifter_total_weight_l183_183669


namespace complex_num_z_imaginary_square_l183_183126

theorem complex_num_z_imaginary_square (z : ℂ) (h1 : z.im ≠ 0) (h2 : z.re = 0) (h3 : ((z + 1) ^ 2).re = 0) :
  z = Complex.I ∨ z = -Complex.I :=
by
  sorry

end complex_num_z_imaginary_square_l183_183126


namespace tangent_line_to_ellipse_l183_183896

variable (a b x y x₀ y₀ : ℝ)

-- Definitions
def is_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def point_on_ellipse (x₀ y₀ a b : ℝ) : Prop :=
  x₀^2 / a^2 + y₀^2 / b^2 = 1

-- Theorem
theorem tangent_line_to_ellipse
  (h₁ : point_on_ellipse x₀ y₀ a b) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_to_ellipse_l183_183896


namespace total_students_is_48_l183_183999

-- Definitions according to the given conditions
def boys'_row := 24
def girls'_row := 24

-- Theorem based on the question and the correct answer
theorem total_students_is_48 :
  boys'_row + girls'_row = 48 :=
by
  sorry

end total_students_is_48_l183_183999


namespace cone_base_radius_l183_183504

open Real

theorem cone_base_radius (r_sector : ℝ) (θ_sector : ℝ) : 
    r_sector = 6 ∧ θ_sector = 120 → (∃ r : ℝ, 2 * π * r = θ_sector * π * r_sector / 180 ∧ r = 2) :=
by
  sorry

end cone_base_radius_l183_183504


namespace common_difference_arithmetic_sequence_l183_183455

noncomputable def first_term : ℕ := 5
noncomputable def last_term : ℕ := 50
noncomputable def sum_terms : ℕ := 275

theorem common_difference_arithmetic_sequence :
  ∃ d n, (last_term = first_term + (n - 1) * d) ∧ (sum_terms = n * (first_term + last_term) / 2) ∧ d = 5 :=
  sorry

end common_difference_arithmetic_sequence_l183_183455


namespace sam_gave_fraction_l183_183738

/-- Given that Mary bought 1500 stickers and shared them between Susan, Andrew, 
and Sam in the ratio 1:1:3. After Sam gave some stickers to Andrew, Andrew now 
has 900 stickers. Prove that the fraction of Sam's stickers given to Andrew is 2/3. -/
theorem sam_gave_fraction (total_stickers : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
    (initial_A : ℕ) (initial_B : ℕ) (initial_C : ℕ) (final_B : ℕ) (given_stickers : ℕ) :
    total_stickers = 1500 → ratio_A = 1 → ratio_B = 1 → ratio_C = 3 →
    initial_A = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_B = total_stickers / (ratio_A + ratio_B + ratio_C) →
    initial_C = 3 * (total_stickers / (ratio_A + ratio_B + ratio_C)) →
    final_B = 900 →
    initial_B + given_stickers = final_B →
    given_stickers / initial_C = 2 / 3 :=
by
  intros
  sorry

end sam_gave_fraction_l183_183738


namespace total_cost_of_topsoil_l183_183608

def cost_per_cubic_foot : ℝ := 8
def cubic_yards_to_cubic_feet : ℝ := 27
def volume_in_yards : ℝ := 7

theorem total_cost_of_topsoil :
  (cubic_yards_to_cubic_feet * volume_in_yards) * cost_per_cubic_foot = 1512 :=
by
  sorry

end total_cost_of_topsoil_l183_183608


namespace intersection_S_T_l183_183077

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183077


namespace intersection_S_T_l183_183017

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183017


namespace intersection_S_T_eq_T_l183_183116

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183116


namespace equality_equiv_l183_183329

-- Problem statement
theorem equality_equiv (a b c : ℝ) :
  (a + b + c ≠ 0 → ( (a * (b - c)) / (b + c) + (b * (c - a)) / (c + a) + (c * (a - b)) / (a + b) = 0 ↔
  (a^2 * (b - c)) / (b + c) + (b^2 * (c - a)) / (c + a) + (c^2 * (a - b)) / (a + b) = 0)) ∧
  (a + b + c = 0 → ∀ w x y z: ℝ, w * x + y * z = 0) :=
by
  sorry

end equality_equiv_l183_183329


namespace no_prime_for_equation_l183_183458

theorem no_prime_for_equation (x k : ℕ) (p : ℕ) (h_prime : p.Prime) (h_eq : x^5 + 2 * x + 3 = p^k) : False := 
sorry

end no_prime_for_equation_l183_183458


namespace intersection_S_T_eq_T_l183_183089

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183089


namespace rowing_distance_l183_183387

noncomputable def effective_speed_with_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed + current_speed

noncomputable def effective_speed_against_current (rowing_speed current_speed : ℕ) : ℕ :=
  rowing_speed - current_speed

noncomputable def distance (speed time : ℕ) : ℕ :=
  speed * time

theorem rowing_distance (rowing_speed current_speed total_time : ℕ) 
  (hrowing_speed : rowing_speed = 10)
  (hcurrent_speed : current_speed = 2)
  (htotal_time : total_time = 30) : 
  (distance 8 18) = 144 := 
by
  sorry

end rowing_distance_l183_183387


namespace exists_set_B_l183_183744

noncomputable def construct_set_B (A : Finset ℕ) : Finset ℕ :=
  let m := (A.max' (by exact' A.nonempty)).succ in
  let x := λ i : ℕ, Nat.fact m * List.prod (List.range i) - 1 in
  Finset.union A (Finset.range m ∪ Finset.range.succ (λ i : ℕ, x i))

theorem exists_set_B (A : Finset ℕ) (hA : ∀ x ∈ A, 0 < x) :
  ∃ B : Finset ℕ, A ⊆ B ∧ (Finset.prod B id) = (Finset.sum B (λ x, x^2)) :=
by
  have m : ℕ := (A.max' (by exact' A.nonempty)).succ
  let x := λ i : ℕ, Nat.fact m * List.prod (List.range i) - 1
  have B := construct_set_B A
  use B
  split
  · -- Proving A ⊆ B
    sorry
  · -- Proving Finset.prod B id = Finset.sum B (λ x, x^2)
    sorry

end exists_set_B_l183_183744


namespace num_integers_divisors_l183_183821

-- Defining the sequence \( \{a_n\}_{n \geq 1} \)
def seq (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else (n^(seq (n - 1)))

-- Prove the main statement
theorem num_integers_divisors : 
  (finset.filter (λ k, (k + 1) ∣ (seq k) - 1) (finset.Icc 2 2020)).card = 1009 :=
by
  sorry

end num_integers_divisors_l183_183821


namespace intersection_S_T_eq_T_l183_183093

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183093


namespace correct_population_growth_pattern_statement_l183_183779

-- Definitions based on the conditions provided
def overall_population_growth_modern (world_population : ℕ) : Prop :=
  -- The overall pattern of population growth worldwide is already in the modern stage
  sorry

def transformation_synchronized (world_population : ℕ) : Prop :=
  -- The transformation of population growth patterns in countries or regions around the world is synchronized
  sorry

def developed_countries_transformed (world_population : ℕ) : Prop :=
  -- Developed countries have basically completed the transformation of population growth patterns
  sorry

def transformation_determined_by_population_size (world_population : ℕ) : Prop :=
  -- The process of transformation in population growth patterns is determined by the population size of each area
  sorry

-- The statement to be proven
theorem correct_population_growth_pattern_statement (world_population : ℕ) :
  developed_countries_transformed world_population := sorry

end correct_population_growth_pattern_statement_l183_183779


namespace animals_on_stump_l183_183967

def possible_n_values (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 15

theorem animals_on_stump (n : ℕ) (h1 : n ≥ 3) (h2 : n ≤ 20)
  (h3 : 11 ≥ (n + 1) / 3) (h4 : 9 ≥ n - (n + 1) / 3) : possible_n_values n :=
by {
  sorry
}

end animals_on_stump_l183_183967


namespace intersection_S_T_eq_T_l183_183085

noncomputable def S : Set ℤ := { s | ∃ (n : ℤ), s = 2 * n + 1 }
noncomputable def T : Set ℤ := { t | ∃ (n : ℤ), t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := sorry

end intersection_S_T_eq_T_l183_183085


namespace total_pencils_owned_l183_183401

def SetA_pencils := 10
def SetB_pencils := 20
def SetC_pencils := 30

def friends_SetA_Buys := 3
def friends_SetB_Buys := 2
def friends_SetC_Buys := 2

def Chloe_SetA_Buys := 1
def Chloe_SetB_Buys := 1
def Chloe_SetC_Buys := 1

def total_friends_pencils := friends_SetA_Buys * SetA_pencils + friends_SetB_Buys * SetB_pencils + friends_SetC_Buys * SetC_pencils
def total_Chloe_pencils := Chloe_SetA_Buys * SetA_pencils + Chloe_SetB_Buys * SetB_pencils + Chloe_SetC_Buys * SetC_pencils
def total_pencils := total_friends_pencils + total_Chloe_pencils

theorem total_pencils_owned : total_pencils = 190 :=
by
  sorry

end total_pencils_owned_l183_183401


namespace second_movie_time_difference_l183_183597

def first_movie_length := 90 -- 1 hour and 30 minutes in minutes
def popcorn_time := 10 -- Time spent making popcorn in minutes
def fries_time := 2 * popcorn_time -- Time spent making fries in minutes
def total_time := 4 * 60 -- Total time for cooking and watching movies in minutes

theorem second_movie_time_difference :
  (total_time - (popcorn_time + fries_time + first_movie_length)) - first_movie_length = 30 :=
by
  sorry

end second_movie_time_difference_l183_183597


namespace rectangle_area_perimeter_l183_183664

-- Defining the problem conditions
def positive_int (n : Int) : Prop := n > 0

-- The main statement of the problem
theorem rectangle_area_perimeter (a b : Int) (h1 : positive_int a) (h2 : positive_int b) : 
  ¬ (a + 2) * (b + 2) - 4 = 146 :=
by
  sorry

end rectangle_area_perimeter_l183_183664


namespace six_digit_ababab_divisible_by_101_l183_183167

theorem six_digit_ababab_divisible_by_101 (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) :
  ∃ k : ℕ, 101 * k = 101010 * a + 10101 * b :=
sorry

end six_digit_ababab_divisible_by_101_l183_183167


namespace Christine_distance_went_l183_183255

-- Definitions from conditions
def Speed : ℝ := 20 -- miles per hour
def Time : ℝ := 4  -- hours

-- Statement of the problem
def Distance_went : ℝ := Speed * Time

-- The theorem we need to prove
theorem Christine_distance_went : Distance_went = 80 :=
by
  sorry

end Christine_distance_went_l183_183255


namespace intersection_eq_T_l183_183027

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183027


namespace divisors_of_30240_l183_183439

theorem divisors_of_30240 : 
  ∃ s : Finset ℕ, (s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (∀ d ∈ s, (30240 % d = 0)) ∧ (s.card = 9) :=
by
  sorry

end divisors_of_30240_l183_183439


namespace evaluate_tensor_expression_l183_183863

-- Define the tensor operation
def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- The theorem we want to prove
theorem evaluate_tensor_expression : tensor (tensor 5 3) 2 = 293 / 15 := by
  sorry

end evaluate_tensor_expression_l183_183863


namespace determine_c_div_d_l183_183545

theorem determine_c_div_d (x y c d : ℝ) (h1 : 4 * x + 8 * y = c) (h2 : 5 * x - 10 * y = d) (h3 : d ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) : c / d = -4 / 5 :=
by
sorry

end determine_c_div_d_l183_183545


namespace intersection_S_T_l183_183053

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l183_183053


namespace simplify_expression_l183_183477

theorem simplify_expression (w : ℝ) : (5 - 2 * w) - (4 + 5 * w) = 1 - 7 * w := by 
  sorry

end simplify_expression_l183_183477


namespace max_band_members_l183_183769

theorem max_band_members 
  (m : ℤ)
  (h1 : 30 * m % 31 = 7)
  (h2 : 30 * m < 1500) : 
  30 * m = 720 :=
sorry

end max_band_members_l183_183769


namespace max_product_ge_993_squared_l183_183171

theorem max_product_ge_993_squared (a : Fin 1985 → Fin 1985) (hperm : ∀ n : Fin 1985, ∃ k : Fin 1985, a k = n ∧ ∃ m : Fin 1985, a m = n) :
  ∃ k : Fin 1985, a k * k ≥ 993^2 :=
sorry

end max_product_ge_993_squared_l183_183171


namespace range_of_m_l183_183585

theorem range_of_m (m : ℝ) (x : ℝ) : (∀ x, (1 - m) * x = 2 - 3 * x → x > 0) ↔ m < 4 :=
by
  sorry

end range_of_m_l183_183585


namespace bicycle_car_speed_l183_183409

theorem bicycle_car_speed (x : Real) (h1 : x > 0) :
  10 / x - 10 / (2 * x) = 1 / 3 :=
by
  sorry

end bicycle_car_speed_l183_183409


namespace intersection_of_S_and_T_l183_183102

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183102


namespace paperclip_day_l183_183163

theorem paperclip_day:
  ∃ k : ℕ, 5 * 3 ^ k > 500 ∧ ∀ m : ℕ, m < k → 5 * 3 ^ m ≤ 500 ∧ k % 7 = 5 :=
sorry

end paperclip_day_l183_183163


namespace total_metal_wasted_l183_183532

noncomputable def wasted_metal (a b : ℝ) (h : b ≤ 2 * a) : ℝ := 
  2 * a * b - (b ^ 2 / 2)

theorem total_metal_wasted (a b : ℝ) (h : b ≤ 2 * a) : 
  wasted_metal a b h = 2 * a * b - b ^ 2 / 2 :=
sorry

end total_metal_wasted_l183_183532


namespace parabola_focus_and_directrix_l183_183130

theorem parabola_focus_and_directrix :
  (∀ x y : ℝ, x^2 = 4 * y → ∃ a b : ℝ, (a, b) = (0, 1) ∧ y = -1) :=
by
  -- Here, we would provide definitions and logical steps if we were completing the proof.
  -- For now, we will leave it unfinished.
  sorry

end parabola_focus_and_directrix_l183_183130


namespace intersection_eq_T_l183_183035

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183035


namespace intersection_S_T_l183_183020

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183020


namespace set_intersection_l183_183858

def M := {x : ℝ | x^2 > 4}
def N := {x : ℝ | 1 < x ∧ x ≤ 3}
def complement_M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def intersection := N ∩ complement_M

theorem set_intersection : intersection = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

end set_intersection_l183_183858


namespace gain_percent_is_150_l183_183373

theorem gain_percent_is_150 (CP SP : ℝ) (hCP : CP = 10) (hSP : SP = 25) : (SP - CP) / CP * 100 = 150 := by
  sorry

end gain_percent_is_150_l183_183373


namespace inequality_proof_l183_183467

theorem inequality_proof
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (6841 * x - 1) / 9973 + (9973 * y - 1) / 6841 = z) :
  x / 9973 + y / 6841 > 1 :=
sorry

end inequality_proof_l183_183467


namespace discount_percentage_l183_183962

theorem discount_percentage (original_price sale_price : ℝ) (h1 : original_price = 150) (h2 : sale_price = 135) : 
  (original_price - sale_price) / original_price * 100 = 10 :=
by 
  sorry

end discount_percentage_l183_183962


namespace inverse_proportion_passing_through_l183_183435

theorem inverse_proportion_passing_through (k : ℝ) :
  (∀ x y : ℝ, (y = k / x) → (x = 3 → y = 2)) → k = 6 := 
by
  sorry

end inverse_proportion_passing_through_l183_183435


namespace largest_integer_binom_l183_183212

theorem largest_integer_binom (n : ℕ) (h : nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) : n = 6 :=
sorry

end largest_integer_binom_l183_183212


namespace sum_a_b_eq_five_l183_183438

theorem sum_a_b_eq_five (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) : a + b = 5 :=
sorry

end sum_a_b_eq_five_l183_183438


namespace find_z_coordinate_of_point_on_line_passing_through_l183_183975

theorem find_z_coordinate_of_point_on_line_passing_through
  (p1 p2 : ℝ × ℝ × ℝ)
  (x_value : ℝ)
  (z_value : ℝ)
  (h1 : p1 = (1, 3, 2))
  (h2 : p2 = (4, 2, -1))
  (h3 : x_value = 3)
  (param : ℝ)
  (h4 : x_value = (1 + 3 * param))
  (h5 : z_value = (2 - 3 * param)) :
  z_value = 0 := by
  sorry

end find_z_coordinate_of_point_on_line_passing_through_l183_183975


namespace arithmetic_geometric_mean_identity_l183_183609

theorem arithmetic_geometric_mean_identity (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 96) : x^2 + y^2 = 1408 :=
by
  sorry

end arithmetic_geometric_mean_identity_l183_183609


namespace largest_n_for_divisibility_l183_183371

theorem largest_n_for_divisibility :
  ∃ (n : ℕ), n = 5 ∧ 3^n ∣ (4^27000 - 82) ∧ ¬ 3^(n + 1) ∣ (4^27000 - 82) :=
by
  sorry

end largest_n_for_divisibility_l183_183371


namespace sequence_formula_l183_183259

theorem sequence_formula (u : ℕ → ℤ) (h0 : u 0 = 1) (h1 : u 1 = 4)
  (h_rec : ∀ n : ℕ, u (n + 2) = 5 * u (n + 1) - 6 * u n) :
  ∀ n : ℕ, u n = 2 * 3^n - 2^n :=
by 
  sorry

end sequence_formula_l183_183259


namespace total_circles_l183_183390

theorem total_circles (n : ℕ) (h1 : ∀ k : ℕ, k = n + 14 → n^2 = (k * (k + 1) / 2)) : 
  n = 35 → n^2 = 1225 :=
by
  sorry

end total_circles_l183_183390


namespace henry_income_percent_increase_l183_183515

theorem henry_income_percent_increase :
  let original_income : ℝ := 120
  let new_income : ℝ := 180
  let increase := new_income - original_income
  let percent_increase := (increase / original_income) * 100
  percent_increase = 50 :=
by
  sorry

end henry_income_percent_increase_l183_183515


namespace car_a_speed_l183_183400

theorem car_a_speed (d_gap : ℕ) (v_B : ℕ) (t : ℕ) (d_ahead : ℕ) (v_A : ℕ) 
  (h1 : d_gap = 24) (h2 : v_B = 50) (h3 : t = 4) (h4 : d_ahead = 8)
  (h5 : v_A = (d_gap + v_B * t + d_ahead) / t) : v_A = 58 :=
by {
  exact (sorry : v_A = 58)
}

end car_a_speed_l183_183400


namespace no_nat_solution_l183_183683

theorem no_nat_solution (x y z : ℕ) : ¬ (x^3 + 2 * y^3 = 4 * z^3) :=
sorry

end no_nat_solution_l183_183683


namespace roque_bike_time_l183_183312

-- Definitions of conditions
def roque_walk_time_per_trip : ℕ := 2
def roque_walk_trips_per_week : ℕ := 3
def roque_bike_trips_per_week : ℕ := 2
def total_commuting_time_per_week : ℕ := 16

-- Statement of the problem to prove
theorem roque_bike_time (B : ℕ) :
  (roque_walk_time_per_trip * 2 * roque_walk_trips_per_week + roque_bike_trips_per_week * 2 * B = total_commuting_time_per_week) → 
  B = 1 :=
by
  sorry

end roque_bike_time_l183_183312


namespace min_m_plus_n_l183_183624

open Nat

theorem min_m_plus_n (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 45 * m = n^3) (h_mult_of_five : 5 ∣ n) :
  m + n = 90 :=
sorry

end min_m_plus_n_l183_183624


namespace correct_calculation_l183_183234

theorem correct_calculation : sqrt 8 / sqrt 2 = 2 :=
by
-- sorry

end correct_calculation_l183_183234


namespace monotonicity_case1_monotonicity_case2_lower_bound_l183_183851

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem monotonicity_case1 (a : ℝ) (h : a ≤ 0) : 
  ∀ x y : ℝ, x < y → f a x > f a y := 
by
  sorry

theorem monotonicity_case2 (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, x < Real.log (1 / a) → f a x > f a (Real.log (1 / a)) ∧ 
           x > Real.log (1 / a) → f a x < f a (Real.log (1 / a)) := 
by
  sorry

theorem lower_bound (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ,  f a x > 2 * Real.log a + 3 / 2 := 
by
  sorry

end monotonicity_case1_monotonicity_case2_lower_bound_l183_183851


namespace average_age_increase_39_l183_183959

variable (n : ℕ) (A : ℝ)
noncomputable def average_age_increase (r : ℝ) : Prop :=
  (r = 7) →
  (n + 1) * (A + r) = n * A + 39 →
  (n + 1) * (A - 1) = n * A + 15 →
  r = 7

theorem average_age_increase_39 : ∀ (n : ℕ) (A : ℝ), average_age_increase n A 7 :=
by
  intros n A
  unfold average_age_increase
  intros hr h1 h2
  exact hr

end average_age_increase_39_l183_183959


namespace f_inequality_l183_183856

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end f_inequality_l183_183856


namespace building_height_is_74_l183_183507

theorem building_height_is_74
  (building_shadow : ℚ)
  (flagpole_height : ℚ)
  (flagpole_shadow : ℚ)
  (ratio_valid : building_shadow / flagpole_shadow = 21 / 8)
  (flagpole_height_value : flagpole_height = 28)
  (building_shadow_value : building_shadow = 84)
  (flagpole_shadow_value : flagpole_shadow = 32) :
  ∃ (h : ℚ), h = 74 := by
  sorry

end building_height_is_74_l183_183507


namespace ellen_golf_cart_trips_l183_183827

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_l183_183827


namespace intersection_S_T_l183_183072

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T := by sorry

end intersection_S_T_l183_183072


namespace functional_equation_true_l183_183480

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : f x > 0
axiom f_property (a b : ℝ) : f a * f b = f (a + b)

theorem functional_equation_true :
  (f 0 = 1) ∧ 
  (∀ a, f (-a) = 1 / f a) ∧ 
  (∀ a, f a = (f (4 * a)) ^ (1 / 4)) ∧ 
  (∀ a, f (a^2) = (f a)^2) :=
by {
  sorry
}

end functional_equation_true_l183_183480


namespace log_expression_l183_183815

section log_problem

variable (log : ℝ → ℝ)
variable (m n : ℝ)

-- Assume the properties of logarithms:
-- 1. log(m^n) = n * log(m)
axiom log_pow (m : ℝ) (n : ℝ) : log (m ^ n) = n * log m
-- 2. log(m * n) = log(m) + log(n)
axiom log_mul (m n : ℝ) : log (m * n) = log m + log n
-- 3. log(1) = 0
axiom log_one : log 1 = 0

theorem log_expression : log 5 * log 2 + log (2 ^ 2) - log 2 = 0 := by
  sorry

end log_problem

end log_expression_l183_183815


namespace tan_alpha_beta_l183_183419

theorem tan_alpha_beta (α β : ℝ) (h : 2 * Real.sin β = Real.sin (2 * α + β)) :
  Real.tan (α + β) = 3 * Real.tan α := 
sorry

end tan_alpha_beta_l183_183419


namespace ellen_golf_cart_trips_l183_183826

def patrons_from_cars : ℕ := 12
def patrons_from_bus : ℕ := 27
def patrons_per_cart : ℕ := 3

theorem ellen_golf_cart_trips : (patrons_from_cars + patrons_from_bus) / patrons_per_cart = 13 := by
  sorry

end ellen_golf_cart_trips_l183_183826


namespace projectile_reaches_75_feet_l183_183976

def projectile_height (t : ℝ) : ℝ := -16 * t^2 + 80 * t

theorem projectile_reaches_75_feet :
  ∃ t : ℝ, projectile_height t = 75 ∧ t = 1.25 :=
by
  -- Skipping the proof as instructed
  sorry

end projectile_reaches_75_feet_l183_183976


namespace sunday_dogs_count_l183_183796

-- Define initial conditions
def initial_dogs : ℕ := 2
def monday_dogs : ℕ := 3
def total_dogs : ℕ := 10
def sunday_dogs (S : ℕ) : Prop :=
  initial_dogs + S + monday_dogs = total_dogs

-- State the theorem
theorem sunday_dogs_count : ∃ S : ℕ, sunday_dogs S ∧ S = 5 := by
  sorry

end sunday_dogs_count_l183_183796


namespace p_sufficient_not_necessary_for_q_l183_183418

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l183_183418


namespace intersection_S_T_l183_183018

def int_set_odd : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def int_set_four_n_plus_one : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : int_set_odd ∩ int_set_four_n_plus_one = int_set_four_n_plus_one :=
by
  sorry

end intersection_S_T_l183_183018


namespace find_difference_l183_183140

variable (k1 k2 t1 t2 : ℝ)

theorem find_difference (h1 : t1 = 5 / 9 * (k1 - 32))
                        (h2 : t2 = 5 / 9 * (k2 - 32))
                        (h3 : t1 = 105)
                        (h4 : t2 = 80) :
  k1 - k2 = 45 :=
by
  sorry

end find_difference_l183_183140


namespace total_white_balls_l183_183932

theorem total_white_balls : ∃ W R B : ℕ,
  W + R = 300 ∧ B = 100 ∧
  ∃ (bw1 bw2 rw3 rw W3 : ℕ),
  bw1 = 27 ∧
  rw3 + rw = 42 ∧
  W3 = rw ∧
  B = bw1 + W3 + rw3 + bw2 ∧
  W = bw1 + 2 * bw2 + 3 * W3 ∧
  R = 3 * rw3 + rw ∧
  W = 158 :=
by
  sorry

end total_white_balls_l183_183932


namespace fish_size_difference_l183_183344

variables {S J W : ℝ}

theorem fish_size_difference (h1 : S = J + 21.52) (h2 : J = W - 12.64) : S - W = 8.88 :=
sorry

end fish_size_difference_l183_183344


namespace sum_of_first_2009_terms_l183_183153

variable (a : ℕ → ℝ) (d : ℝ)

-- conditions: arithmetic sequence and specific sum condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_condition (a : ℕ → ℝ) : Prop :=
  a 1004 + a 1005 + a 1006 = 3

-- sum of the first 2009 terms
noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n.succ * (a 0 + a n.succ) / 2)

-- proof problem
theorem sum_of_first_2009_terms (h1 : is_arithmetic_sequence a d) (h2 : sum_condition a) :
  sum_first_n_terms a 2008 = 2009 :=
sorry

end sum_of_first_2009_terms_l183_183153


namespace ratio_of_square_areas_l183_183479

theorem ratio_of_square_areas (y : ℝ) (hy : y > 0) : 
  (y^2 / (3 * y)^2) = 1 / 9 :=
sorry

end ratio_of_square_areas_l183_183479


namespace largest_number_l183_183633

theorem largest_number (a b c : ℕ) (h1: a ≤ b) (h2: b ≤ c) 
  (h3: (a + b + c) = 90) (h4: b = 32) (h5: b = a + 4) : c = 30 :=
sorry

end largest_number_l183_183633


namespace music_marks_l183_183481

variable (M : ℕ) -- Variable to represent marks in music

/-- Conditions -/
def science_marks : ℕ := 70
def social_studies_marks : ℕ := 85
def total_marks : ℕ := 275
def physics_marks : ℕ := M / 2

theorem music_marks :
  science_marks + M + social_studies_marks + physics_marks M = total_marks → M = 80 :=
by
  sorry

end music_marks_l183_183481


namespace largest_3_digit_sum_l183_183576

-- Defining the condition that ensures X, Y, Z are different digits ranging from 0 to 9
def valid_digits (X Y Z : ℕ) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- Problem statement: Proving the largest possible 3-digit sum is 994
theorem largest_3_digit_sum : ∃ (X Y Z : ℕ), valid_digits X Y Z ∧ 111 * X + 11 * Y + Z = 994 :=
by
  sorry

end largest_3_digit_sum_l183_183576


namespace brenda_age_l183_183982

-- Define ages of Addison, Brenda, Carlos, and Janet
variables (A B C J : ℕ)

-- Formalize the conditions from the problem
def condition1 := A = 4 * B
def condition2 := C = 2 * B
def condition3 := A = J

-- State the theorem we aim to prove
theorem brenda_age (A B C J : ℕ) (h1 : condition1 A B)
                                (h2 : condition2 C B)
                                (h3 : condition3 A J) :
  B = J / 4 :=
sorry

end brenda_age_l183_183982


namespace work_duration_B_l183_183789

theorem work_duration_B (x : ℕ) (h : x = 10) : 
  (x * (1 / 15 : ℚ)) + (2 * (1 / 6 : ℚ)) = 1 := 
by 
  rw [h]
  sorry

end work_duration_B_l183_183789


namespace gym_cost_l183_183722

theorem gym_cost (x : ℕ) (hx : x > 0) (h1 : 50 + 12 * x + 48 * x = 650) : x = 10 :=
by
  sorry

end gym_cost_l183_183722


namespace min_sum_fraction_sqrt_l183_183556

open Real

theorem min_sum_fraction_sqrt (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ min, min = sqrt 2 ∧ ∀ z, (z = (x / sqrt (1 - x) + y / sqrt (1 - y))) → z ≥ sqrt 2 :=
sorry

end min_sum_fraction_sqrt_l183_183556


namespace multiplication_in_S_l183_183314

-- Define the set S as given in the conditions
variable (S : Set ℝ)

-- Condition 1: 1 ∈ S
def condition1 : Prop := 1 ∈ S

-- Condition 2: ∀ a b ∈ S, a - b ∈ S
def condition2 : Prop := ∀ a b : ℝ, a ∈ S → b ∈ S → (a - b) ∈ S

-- Condition 3: ∀ a ∈ S, a ≠ 0 → 1 / a ∈ S
def condition3 : Prop := ∀ a : ℝ, a ∈ S → a ≠ 0 → (1 / a) ∈ S

-- Theorem to prove: ∀ a b ∈ S, ab ∈ S
theorem multiplication_in_S (h1 : condition1 S) (h2 : condition2 S) (h3 : condition3 S) :
  ∀ a b : ℝ, a ∈ S → b ∈ S → (a * b) ∈ S := 
  sorry

end multiplication_in_S_l183_183314


namespace train_speed_l183_183667

theorem train_speed
  (length_of_train : ℕ)
  (time_to_cross_bridge : ℕ)
  (length_of_bridge : ℕ)
  (speed_conversion_factor : ℕ)
  (H1 : length_of_train = 120)
  (H2 : time_to_cross_bridge = 30)
  (H3 : length_of_bridge = 255)
  (H4 : speed_conversion_factor = 36) : 
  (length_of_train + length_of_bridge) / (time_to_cross_bridge / speed_conversion_factor) = 45 :=
by
  sorry

end train_speed_l183_183667


namespace largest_n_for_binomial_equality_l183_183224

theorem largest_n_for_binomial_equality :
  ∃ n : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 n ) ∧
  ∀ m : ℕ, ( (nat.choose 10 4) + (nat.choose 10 5) = nat.choose 11 m ) → m ≤ 6 :=
begin
  sorry
end

end largest_n_for_binomial_equality_l183_183224


namespace profit_correct_A_B_l183_183525

noncomputable def profit_per_tire_A (batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A : ℕ) : ℚ :=
  let cost_first_5000 := batch_cost_A1 + (cost_per_tire_A1 * 5000)
  let revenue_first_5000 := sell_price_tire_A1 * 5000
  let profit_first_5000 := revenue_first_5000 - cost_first_5000
  let cost_remaining := batch_cost_A2 + (cost_per_tire_A2 * (produced_A - 5000))
  let revenue_remaining := sell_price_tire_A2 * (produced_A - 5000)
  let profit_remaining := revenue_remaining - cost_remaining
  let total_profit := profit_first_5000 + profit_remaining
  total_profit / produced_A

noncomputable def profit_per_tire_B (batch_cost_B cost_per_tire_B sell_price_tire_B produced_B : ℕ) : ℚ :=
  let cost := batch_cost_B + (cost_per_tire_B * produced_B)
  let revenue := sell_price_tire_B * produced_B
  let profit := revenue - cost
  profit / produced_B

theorem profit_correct_A_B
  (batch_cost_A1 : ℕ := 22500) 
  (batch_cost_A2 : ℕ := 20000) 
  (cost_per_tire_A1 : ℕ := 8) 
  (cost_per_tire_A2 : ℕ := 6) 
  (sell_price_tire_A1 : ℕ := 20) 
  (sell_price_tire_A2 : ℕ := 18) 
  (produced_A : ℕ := 15000)
  (batch_cost_B : ℕ := 24000) 
  (cost_per_tire_B : ℕ := 7) 
  (sell_price_tire_B : ℕ := 19) 
  (produced_B : ℕ := 10000) :
  profit_per_tire_A batch_cost_A1 batch_cost_A2 cost_per_tire_A1 cost_per_tire_A2 sell_price_tire_A1 sell_price_tire_A2 produced_A = 9.17 ∧
  profit_per_tire_B batch_cost_B cost_per_tire_B sell_price_tire_B produced_B = 9.60 :=
by
  sorry

end profit_correct_A_B_l183_183525


namespace inequality_proof_l183_183734

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l183_183734


namespace boat_speed_of_stream_l183_183517

theorem boat_speed_of_stream :
  ∀ (x : ℝ), 
    (∀ s_b : ℝ, s_b = 18) → 
    (∀ d1 d2 : ℝ, d1 = 48 → d2 = 32 → d1 / (18 + x) = d2 / (18 - x)) → 
    x = 3.6 :=
by 
  intros x h_speed h_distance
  sorry

end boat_speed_of_stream_l183_183517


namespace evaluate_expression_l183_183753

theorem evaluate_expression (a b : ℤ) (h_a : a = 1) (h_b : b = -2) : 
  2 * (a^2 - 3 * a * b + 1) - (2 * a^2 - b^2) + 5 * a * b = 8 :=
by
  sorry

end evaluate_expression_l183_183753


namespace feeding_times_per_day_l183_183749

-- Definitions for the given conditions
def number_of_puppies : ℕ := 7
def total_portions : ℕ := 105
def number_of_days : ℕ := 5

-- Theorem to prove the answer to the question
theorem feeding_times_per_day : 
  let portions_per_day := total_portions / number_of_days in
  let times_per_puppy := portions_per_day / number_of_puppies in
  times_per_puppy = 3 :=
by
  -- We should provide the proof here, but we will use 'sorry' to skip it
  sorry

end feeding_times_per_day_l183_183749


namespace bus_ride_difference_l183_183325

theorem bus_ride_difference :
  ∀ (Oscar_bus Charlie_bus : ℝ),
  Oscar_bus = 0.75 → Charlie_bus = 0.25 → Oscar_bus - Charlie_bus = 0.50 :=
by
  intros Oscar_bus Charlie_bus hOscar hCharlie
  rw [hOscar, hCharlie]
  norm_num

end bus_ride_difference_l183_183325


namespace correct_statement_A_l183_183672

-- Definitions for conditions
def general_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

def actinomycetes_dilution_range : Set ℕ := {10^3, 10^4, 10^5}

def fungi_dilution_range : Set ℕ := {10^2, 10^3, 10^4}

def first_experiment_dilution_range : Set ℕ := {10^3, 10^4, 10^5, 10^6, 10^7}

-- Statement to prove
theorem correct_statement_A : 
  (general_dilution_range = {10^3, 10^4, 10^5, 10^6, 10^7}) :=
sorry

end correct_statement_A_l183_183672


namespace cube_surface_area_with_holes_l183_183393

theorem cube_surface_area_with_holes 
    (edge_length : ℝ) 
    (hole_side_length : ℝ) 
    (num_faces : ℕ) 
    (parallel_edges : Prop)
    (holes_centered : Prop)
    (h_edge : edge_length = 5)
    (h_hole : hole_side_length = 2)
    (h_faces : num_faces = 6)
    (h_inside_area : parallel_edges ∧ holes_centered)
    : (150 - 24 + 96 = 222) :=
by
    sorry

end cube_surface_area_with_holes_l183_183393


namespace bijection_if_injective_or_surjective_l183_183378

variables {X Y : Type} [Fintype X] [Fintype Y] (f : X → Y)

theorem bijection_if_injective_or_surjective (hX : Fintype.card X = Fintype.card Y)
  (hf : Function.Injective f ∨ Function.Surjective f) : Function.Bijective f :=
by
  sorry

end bijection_if_injective_or_surjective_l183_183378


namespace train_speed_in_kmh_l183_183668

/-- Definition of length of the train in meters. -/
def train_length : ℕ := 200

/-- Definition of time taken to cross the electric pole in seconds. -/
def time_to_cross : ℕ := 20

/-- The speed of the train in km/h is 36 given the length of the train and time to cross. -/
theorem train_speed_in_kmh (length : ℕ) (time : ℕ) (h_len : length = train_length) (h_time: time = time_to_cross) : 
  (length / time : ℚ) * 3.6 = 36 := 
by
  sorry

end train_speed_in_kmh_l183_183668


namespace part1_monotonicity_part2_inequality_l183_183852

variable (a : ℝ) (x : ℝ)
variable (h : a > 0)

def f := a * (Real.exp x + a) - x

theorem part1_monotonicity :
  if a ≤ 0 then ∀ x : ℝ, (f a x) < (f a (x + 1)) else
  if a > 0 then ∀ x : ℝ, (x < Real.log (1 / a) → (f a x) > (f a (x + 1))) ∧ 
                       (x > Real.log (1 / a) → (f a x) < (f a (x - 1))) else 
  (False) := sorry

theorem part2_inequality :
  ∀ x : ℝ, f a x > 2 * Real.log a + 3 / 2 := sorry

end part1_monotonicity_part2_inequality_l183_183852


namespace parallel_lines_condition_l183_183966

theorem parallel_lines_condition (a : ℝ) :
  ( ∀ x y : ℝ, (a * x + 2 * y + 2 = 0 → ∃ C₁ : ℝ, x - 2 * y = C₁) 
  ∧ (x + (a - 1) * y + 1 = 0 → ∃ C₂ : ℝ, x - 2 * y = C₂) )
  ↔ a = -1 :=
sorry

end parallel_lines_condition_l183_183966


namespace arithmetic_sum_S11_l183_183422

noncomputable def Sn_sum (a1 an n : ℕ) : ℕ := n * (a1 + an) / 2

theorem arithmetic_sum_S11 (a1 a9 a8 a5 a11 : ℕ) (h1 : Sn_sum a1 a9 9 = 54)
    (h2 : Sn_sum a1 a8 8 - Sn_sum a1 a5 5 = 30) : Sn_sum a1 a11 11 = 88 := by
  sorry

end arithmetic_sum_S11_l183_183422


namespace evaluate_complex_fraction_l183_183266

def complex_fraction : Prop :=
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  expr = 76 / 29

theorem evaluate_complex_fraction : complex_fraction :=
by
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  show expr = 76 / 29
  sorry

end evaluate_complex_fraction_l183_183266


namespace find_min_value_omega_l183_183570

noncomputable def min_value_ω (ω : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 2 * Real.sin (ω * x)) → ω > 0 →
  (∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 4 → f x ≥ -2) →
  ω = 3 / 2

-- The statement to be proved:
theorem find_min_value_omega : ∃ ω : ℝ, min_value_ω ω :=
by
  use 3 / 2
  sorry

end find_min_value_omega_l183_183570


namespace arcsin_cos_eq_l183_183265

theorem arcsin_cos_eq :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  have h1 : Real.cos (2 * Real.pi / 3) = -1 / 2 := sorry
  have h2 : Real.arcsin (-1 / 2) = -Real.pi / 6 := sorry
  rw [h1, h2]

end arcsin_cos_eq_l183_183265


namespace new_persons_joined_l183_183529

theorem new_persons_joined :
  ∀ (A : ℝ) (N : ℕ) (avg_new : ℝ) (avg_combined : ℝ), 
  N = 15 → avg_new = 15 → avg_combined = 15.5 → 1 = (N * avg_combined + N * avg_new - 232.5) / (avg_combined - avg_new) := by
  intros A N avg_new avg_combined
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end new_persons_joined_l183_183529


namespace original_price_of_car_l183_183505

theorem original_price_of_car (spent price_percent original_price : ℝ) (h1 : spent = 15000) (h2 : price_percent = 0.40) (h3 : spent = price_percent * original_price) : original_price = 37500 :=
by
  sorry

end original_price_of_car_l183_183505


namespace solve_for_xy_l183_183900

theorem solve_for_xy (x y : ℝ) (h1 : 3 * x ^ 2 - 9 * y ^ 2 = 0) (h2 : x + y = 5) :
    (x = (15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 * Real.sqrt 3 - 5) / 2) ∨
    (x = (-15 - 5 * Real.sqrt 3) / 2 ∧ y = (5 + 5 * Real.sqrt 3) / 2) :=
by
  sorry

end solve_for_xy_l183_183900


namespace num_possible_sums_l183_183174

theorem num_possible_sums (s : Finset ℕ) (hs : s.card = 80) (hsub: s ⊆ Finset.range 121) : 
  ∃ (n : ℕ), (n = 3201) ∧ ∀ U, U = s.sum id → ∃ (U_min U_max : ℕ), U_min = 3240 ∧ U_max = 6440 ∧ (U_min ≤ U ∧ U ≤ U_max) :=
sorry

end num_possible_sums_l183_183174


namespace numerology_eq_l183_183772

theorem numerology_eq : 2222 - 222 + 22 - 2 = 2020 :=
by
  sorry

end numerology_eq_l183_183772


namespace remainder_pow_2023_l183_183774

theorem remainder_pow_2023 (a b : ℕ) (h : b = 2023) : (3 ^ b) % 11 = 5 :=
by
  sorry

end remainder_pow_2023_l183_183774


namespace clock_angle_at_3_20_l183_183949

def degrees_per_hour := 360 / 12
def degrees_per_minute := 360 / 60
def hour_at_three := 3 * degrees_per_hour
def minute_at_twenty := 20 * degrees_per_minute
def hour_hand_move_per_min := degrees_per_hour / 60

theorem clock_angle_at_3_20 
  (h : hour_at_three = 90)
  (m : minute_at_twenty = 120)
  (h_move : hour_hand_move_per_min = 0.5) :
  (abs (minute_at_twenty - (hour_at_three + 20 * hour_hand_move_per_min))) = 20 :=
by
  sorry

end clock_angle_at_3_20_l183_183949


namespace candied_apple_price_l183_183541

theorem candied_apple_price
  (x : ℝ) -- price of each candied apple in dollars
  (h1 : 15 * x + 12 * 1.5 = 48) -- total earnings equation
  : x = 2 := 
sorry

end candied_apple_price_l183_183541


namespace consecutive_odd_numbers_sum_power_fourth_l183_183931

theorem consecutive_odd_numbers_sum_power_fourth :
  ∃ x1 x2 x3 : ℕ, 
  x1 % 2 = 1 ∧ x2 % 2 = 1 ∧ x3 % 2 = 1 ∧ 
  x1 + 2 = x2 ∧ x2 + 2 = x3 ∧ 
  (∃ n : ℕ, n < 10 ∧ (x1 + x2 + x3 = n^4)) :=
sorry

end consecutive_odd_numbers_sum_power_fourth_l183_183931


namespace minimum_value_of_expression_l183_183411

noncomputable def f (x : ℝ) : ℝ := 16^x - 2^x + x^2 + 1

theorem minimum_value_of_expression : ∃ (x : ℝ), f x = 1 ∧ ∀ y : ℝ, f y ≥ 1 := 
sorry

end minimum_value_of_expression_l183_183411


namespace probability_log_interval_l183_183272

open Set Real

noncomputable def probability_in_interval (a b c d : ℝ) (I J : Set ℝ) := 
  (b - a) / (d - c)

theorem probability_log_interval : 
  probability_in_interval 2 4 0 6 (Icc 0 6) (Ioo 2 4) = 1 / 3 := 
sorry

end probability_log_interval_l183_183272


namespace intersection_S_T_eq_T_l183_183115

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end intersection_S_T_eq_T_l183_183115


namespace area_PM_N_l183_183345

noncomputable def parabola : (ℝ × ℝ) → Prop :=
λ p, p.2 ^ 2 = 4 * p.1

noncomputable def line (t : ℝ) : (ℝ × ℝ) → Prop :=
λ p, p.1 = t * p.2 + 7

def focus : ℝ × ℝ := (1, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2

noncomputable def intersects_parabola_line (t : ℝ) (M N : (ℝ × ℝ)) : Prop :=
(parabola M ∧ line t M) ∧ (parabola N ∧ line t N)

noncomputable def tangents_intersect_at (M N P : (ℝ × ℝ)) : Prop :=
let tangent_at_M := {p | ∃ k, p.2 = k * (p.1 - M.1) + M.2} in
let tangent_at_N := {p | ∃ k, p.2 = k * (p.1 - N.1) + N.2} in
tangent_at_M P ∧ tangent_at_N P

noncomputable def vectors_dot_product_zero (M N : (ℝ × ℝ)) : Prop :=
let MF := (1 - M.1, 0 - M.2) in
let NF := (1 - N.1, 0 - N.2) in
dot_product MF NF = 0

def area_of_triangle (P M N : (ℝ × ℝ)) : ℝ :=
(1 / 2) * abs (P.1 * (M.2 - N.2) + M.1 * (N.2 - P.2) + N.1 * (P.2 - M.2))

theorem area_PM_N
  (M N P : (ℝ × ℝ))
  (t : ℝ)
  (h1 : intersects_parabola_line t M N)
  (h2 : vectors_dot_product_zero M N)
  (h3 : tangents_intersect_at M N P) :
  area_of_triangle P M N = 108 := by
  sorry

end area_PM_N_l183_183345


namespace circle_minor_arc_probability_l183_183663

noncomputable def probability_minor_arc_length_lt_one 
  (circle_circumference : ℝ) 
  (A B : ℝ) 
  (h_circumference : circle_circumference = 3) 
  (h_A : 0 ≤ A ∧ A < circle_circumference) 
  (h_B : 0 ≤ B ∧ B < circle_circumference) 
  : ℝ := 
  if abs (B - A) < 1 then sorry else sorry

theorem circle_minor_arc_probability 
  (circle_circumference : ℝ) 
  (h_circumference : circle_circumference = 3) 
  : (probability (λ (B : ℝ), B ∈ set.Ico 0 circle_circumference ∧ 
                  abs (B - 0) < 1 ∨ abs (circle_circumference - abs (B - 0)) < 1)) = 2 / 3 :=
sorry

end circle_minor_arc_probability_l183_183663


namespace reciprocal_sum_l183_183974

theorem reciprocal_sum (x1 x2 x3 k : ℝ) (h : ∀ x, x^2 + k * x - k * x3 = 0 ∧ x ≠ 0 → x = x1 ∨ x = x2) :
  (1 / x1 + 1 / x2 = 1 / x3) := by
  sorry

end reciprocal_sum_l183_183974


namespace ratio_of_probabilities_l183_183404

noncomputable def balls_toss (balls bins : ℕ) : Nat := by
  sorry

def prob_A : ℚ := by
  sorry
  
def prob_B : ℚ := by
  sorry

theorem ratio_of_probabilities (balls : ℕ) (bins : ℕ) 
  (h_balls : balls = 20) (h_bins : bins = 5) (p q : ℚ) 
  (h_p : p = prob_A) (h_q : q = prob_B) :
  (p / q) = 4 := by
  sorry

end ratio_of_probabilities_l183_183404


namespace min_expr_value_min_expr_value_iff_l183_183842

theorem min_expr_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 :=
by {
  sorry
}

theorem min_expr_value_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2) = 4 / 9) ↔ (x = 2.5 ∧ y = 2.5) :=
by {
  sorry
}

end min_expr_value_min_expr_value_iff_l183_183842


namespace soda_cost_proof_l183_183804

theorem soda_cost_proof (b s : ℤ) (h1 : 4 * b + 3 * s = 440) (h2 : 3 * b + 2 * s = 310) : s = 80 :=
by
  sorry

end soda_cost_proof_l183_183804


namespace find_f2a_eq_zero_l183_183566

variable {α : Type} [LinearOrderedField α]

-- Definitions for the function f and its inverse
variable (f : α → α)
variable (finv : α → α)

-- Given conditions
variable (a : α)
variable (h_nonzero : a ≠ 0)
variable (h_inverse1 : ∀ x : α, finv (x + a) = f (x + a)⁻¹)
variable (h_inverse2 : ∀ x : α, f (x) = finv⁻¹ x)
variable (h_fa : f a = a)

-- Statement to be proved in Lean
theorem find_f2a_eq_zero : f (2 * a) = 0 :=
sorry

end find_f2a_eq_zero_l183_183566


namespace prime_solution_l183_183270

theorem prime_solution (p : ℕ) (x y : ℕ) (hp : Prime p) (hx : 0 < x) (hy : 0 < y) :
  p^x = y^3 + 1 → p = 2 ∨ p = 3 :=
by
  sorry

end prime_solution_l183_183270


namespace percent_non_union_women_l183_183300

-- Definitions used in the conditions:
def total_employees := 100
def percent_men := 50 / 100
def percent_union := 60 / 100
def percent_union_men := 70 / 100

-- Calculate intermediate values
def num_men := total_employees * percent_men
def num_union := total_employees * percent_union
def num_union_men := num_union * percent_union_men
def num_non_union := total_employees - num_union
def num_non_union_men := num_men - num_union_men
def num_non_union_women := num_non_union - num_non_union_men

-- Statement of the problem in Lean
theorem percent_non_union_women : (num_non_union_women / num_non_union) * 100 = 80 := 
by {
  sorry
}

end percent_non_union_women_l183_183300


namespace quadratic_factoring_even_a_l183_183762

theorem quadratic_factoring_even_a (a : ℤ) :
  (∃ (m p n q : ℤ), 21 * x^2 + a * x + 21 = (m * x + n) * (p * x + q) ∧ m * p = 21 ∧ n * q = 21 ∧ (∃ (k : ℤ), a = 2 * k)) :=
sorry

end quadratic_factoring_even_a_l183_183762


namespace coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l183_183365

theorem coefficient_of_x9_in_expansion_of_x_minus_2_pow_10 :
  ∃ c : ℤ, (x - 2)^10 = ∑ k in finset.range (11), (nat.choose 10 k) * x^k * (-2)^(10 - k) ∧ c = -20 := 
begin 
  use -20,
  { sorry },
end

end coefficient_of_x9_in_expansion_of_x_minus_2_pow_10_l183_183365


namespace intersection_eq_T_l183_183033

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183033


namespace largest_n_binom_identity_l183_183226

open Nat

theorem largest_n_binom_identity :
  (∃ n, binom 11 n = binom 10 4 + binom 10 5) ∧
  (∀ m, binom 11 m = binom 10 4 + binom 10 5 → m ≤ 6) :=
by
  sorry

end largest_n_binom_identity_l183_183226


namespace max_jogs_l183_183988

theorem max_jogs (jags jigs jogs jugs : ℕ) : 2 * jags + 3 * jigs + 8 * jogs + 5 * jugs = 72 → jags ≥ 1 → jigs ≥ 1 → jugs ≥ 1 → jogs ≤ 7 :=
by
  sorry

end max_jogs_l183_183988


namespace neg_sub_eq_sub_l183_183960

theorem neg_sub_eq_sub (a b : ℝ) : - (a - b) = b - a := 
by
  sorry

end neg_sub_eq_sub_l183_183960


namespace cubic_difference_l183_183428

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l183_183428


namespace merill_has_30_marbles_l183_183604

variable (M E : ℕ)

-- Conditions
def merill_twice_as_many_as_elliot : Prop := M = 2 * E
def together_five_fewer_than_selma : Prop := M + E = 45

theorem merill_has_30_marbles (h1 : merill_twice_as_many_as_elliot M E) (h2 : together_five_fewer_than_selma M E) : M = 30 := 
by
  sorry

end merill_has_30_marbles_l183_183604


namespace original_price_of_car_l183_183506

-- Define the original price of the car based on the condition of the problem
def original_price (spent : ℝ) (percentage : ℝ) : ℝ := spent / percentage

-- Given conditions
def venny_spent : ℝ := 15000
def percentage_of_original : ℝ := 0.40

-- Statement to be proved
theorem original_price_of_car : original_price venny_spent percentage_of_original = 37500 := by
  sorry

end original_price_of_car_l183_183506


namespace intersection_S_T_eq_T_l183_183055

-- Definitions based on the given conditions
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- The theorem statement to be proved
theorem intersection_S_T_eq_T : S ∩ T = T := 
sorry

end intersection_S_T_eq_T_l183_183055


namespace joe_average_test_score_l183_183165

theorem joe_average_test_score 
  (A B C : ℕ) 
  (Hsum : A + B + C = 135) 
  : (A + B + C + 25) / 4 = 40 :=
by
  sorry

end joe_average_test_score_l183_183165


namespace president_vice_president_ways_l183_183177

theorem president_vice_president_ways :
  let boys := 14
  let girls := 10
  let total_boys_ways := boys * (boys - 1)
  let total_girls_ways := girls * (girls - 1)
  total_boys_ways + total_girls_ways = 272 := 
by
  sorry

end president_vice_president_ways_l183_183177


namespace cubic_difference_l183_183429

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l183_183429


namespace remainder_of_sum_of_powers_div_2_l183_183639

theorem remainder_of_sum_of_powers_div_2 : 
  (1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7 + 8^8 + 9^9) % 2 = 1 :=
by 
  sorry

end remainder_of_sum_of_powers_div_2_l183_183639


namespace profit_calculation_l183_183450

-- Define the initial conditions
def initial_cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def price_decrease_effect : ℝ := 4
def daily_profit_target : ℝ := 13600
def minimum_selling_price : ℝ := 150

-- Define the function relationship of daily sales volume with respect to x
def sales_volume (x : ℝ) : ℝ := initial_sales_volume + price_decrease_effect * x

-- Define the selling price
def selling_price (x : ℝ) : ℝ := initial_selling_price - x

-- Define the profit function
def profit (x : ℝ) : ℝ := (selling_price x - initial_cost_price) * sales_volume x

theorem profit_calculation (x : ℝ) (hx : selling_price x ≥ minimum_selling_price) :
  profit x = daily_profit_target ↔ selling_price x = 185 := by
  sorry

end profit_calculation_l183_183450


namespace parallel_slope_l183_183958

theorem parallel_slope (x y : ℝ) (h : 3 * x + 6 * y = -21) : 
    ∃ m : ℝ, m = -1 / 2 :=
by
  sorry

end parallel_slope_l183_183958


namespace expression_evaluate_l183_183773

theorem expression_evaluate :
  50 * (50 - 5) - (50 * 50 - 5) = -245 :=
by
  sorry

end expression_evaluate_l183_183773


namespace angle_DAE_l183_183156

variable (A B C D O E : Point)
variable [circle : Circle O A B C]
variable (hACB : ∠ A C B = 50)
variable (hCBA : ∠ C B A = 70)
variable (hD_perpendicular : ∃ H, ⟂ H D)
variable (hE_diameter : (∀ x, x ∈ diameter O ↔ x = A ∨ x = E))

theorem angle_DAE (A B C D O E : Point) 
  (hACB : ∠ A C B = 50) 
  (hCBA : ∠ C B A = 70) 
  (hD_perpendicular : D = foot_of_perpendicular A B C) 
  (hE_diameter : E = other_end_of_diameter_through A circle): 
  ∠ D A E = 20 := 
sorry

end angle_DAE_l183_183156


namespace paige_science_problems_l183_183607

variable (S : ℤ)

theorem paige_science_problems (h1 : 43 + S - 44 = 11) : S = 12 :=
by
  sorry

end paige_science_problems_l183_183607


namespace distance_interval_l183_183805

theorem distance_interval (d : ℝ) :
  (d < 8) ∧ (d > 7) ∧ (d > 5) ∧ (d ≠ 3) ↔ (7 < d ∧ d < 8) :=
by
  sorry

end distance_interval_l183_183805


namespace smallest_value_of_sum_l183_183864

theorem smallest_value_of_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 3 * a = 4 * b ∧ 4 * b = 7 * c) : a + b + c = 61 :=
sorry

end smallest_value_of_sum_l183_183864


namespace number_of_people_l183_183755

theorem number_of_people (total_bowls : ℕ) (bowls_per_person : ℚ) : total_bowls = 55 ∧ bowls_per_person = 1 + 1/2 + 1/3 → total_bowls / bowls_per_person = 30 :=
by
  sorry

end number_of_people_l183_183755


namespace intersection_eq_T_l183_183039

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183039


namespace intersection_eq_T_l183_183034

open Set

-- Define S and T based on the conditions
def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove S ∩ T = T
theorem intersection_eq_T : S ∩ T = T :=
by sorry

end intersection_eq_T_l183_183034


namespace largest_integer_comb_l183_183220

theorem largest_integer_comb (n: ℕ) (h1 : n = 6) : 
  (nat.choose 10 4 + nat.choose 10 5 = nat.choose 11 n) :=
by
  rw [h1]
  exact sorry

end largest_integer_comb_l183_183220


namespace ratio_a_b_is_zero_l183_183487

-- Setting up the conditions
variables (a y b : ℝ)
variable (d : ℝ)
-- Condition for arithmetic sequence
axiom h1 : a + d = y
axiom h2 : y + d = b
axiom h3 : b + d = 3 * y

-- The Lean statement to prove
theorem ratio_a_b_is_zero (h1 : a + d = y) (h2 : y + d = b) (h3 : b + d = 3 * y) : a / b = 0 :=
sorry

end ratio_a_b_is_zero_l183_183487


namespace transport_cost_6725_l183_183747

variable (P : ℝ) (T : ℝ)

theorem transport_cost_6725
  (h1 : 0.80 * P = 17500)
  (h2 : 1.10 * P = 24475)
  (h3 : 17500 + T + 250 = 24475) :
  T = 6725 := 
sorry

end transport_cost_6725_l183_183747


namespace crayon_difference_l183_183883

theorem crayon_difference:
  let karen := 639
  let cindy := 504
  let peter := 752
  let rachel := 315
  max karen (max cindy (max peter rachel)) - min karen (min cindy (min peter rachel)) = 437 :=
by
  sorry

end crayon_difference_l183_183883


namespace sum_of_consecutive_numbers_with_lcm_168_l183_183918

theorem sum_of_consecutive_numbers_with_lcm_168 (n : ℕ) (h_lcm : Nat.lcm (Nat.lcm n (n + 1)) (n + 2) = 168) : n + (n + 1) + (n + 2) = 21 :=
sorry

end sum_of_consecutive_numbers_with_lcm_168_l183_183918


namespace agatha_bike_budget_l183_183983

def total_initial : ℕ := 60
def cost_frame : ℕ := 15
def cost_front_wheel : ℕ := 25
def total_spent : ℕ := cost_frame + cost_front_wheel
def total_left : ℕ := total_initial - total_spent

theorem agatha_bike_budget : total_left = 20 := by
  sorry

end agatha_bike_budget_l183_183983


namespace ninth_term_arithmetic_sequence_l183_183486

theorem ninth_term_arithmetic_sequence 
  (a1 a17 d a9 : ℚ) 
  (h1 : a1 = 2 / 3) 
  (h17 : a17 = 3 / 2) 
  (h_formula : a17 = a1 + 16 * d) 
  (h9_formula : a9 = a1 + 8 * d) :
  a9 = 13 / 12 := by
  sorry

end ninth_term_arithmetic_sequence_l183_183486


namespace part1_l183_183652

theorem part1 (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 ≠ b^2) :
  (a^2 + a * b + b^2) / (a + b) - (a^2 - a * b + b^2) / (a - b) + (2 * b^2 - b^2 + a^2) / (a^2 - b^2) = 1 := 
sorry

end part1_l183_183652


namespace cubes_difference_l183_183432

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l183_183432


namespace intersection_of_S_and_T_l183_183105

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end intersection_of_S_and_T_l183_183105


namespace intersection_S_T_eq_T_l183_183000

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l183_183000


namespace slope_of_line_AB_on_ellipse_l183_183273

theorem slope_of_line_AB_on_ellipse 
  (A B : ℝ × ℝ) 
  (hA : A.1^2 / 9 + A.2^2 / 3 = 1)
  (hB : B.1^2 / 9 + B.2^2 / 3 = 1)
  (M : ℝ × ℝ) 
  (hM : M = (Real.sqrt 3, Real.sqrt 2))
  (hS : ∃ k : ℝ, A.2 - M.2 = k * (A.1 - M.1) ∧ B.2 - M.2 = (-1/k) * (B.1 - M.1)) : 
  (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 6 / 6 :=
sorry

end slope_of_line_AB_on_ellipse_l183_183273


namespace number_of_morse_code_symbols_l183_183146

-- Define the number of sequences for different lengths
def sequences_of_length (n : Nat) : Nat :=
  2 ^ n

theorem number_of_morse_code_symbols : 
  (sequences_of_length 1) + (sequences_of_length 2) + (sequences_of_length 3) + (sequences_of_length 4) + (sequences_of_length 5) = 62 := by
  sorry

end number_of_morse_code_symbols_l183_183146


namespace february_first_is_friday_l183_183869

-- Definition of conditions
def february_has_n_mondays (n : ℕ) : Prop := n = 3
def february_has_n_fridays (n : ℕ) : Prop := n = 5

-- The statement to prove
theorem february_first_is_friday (n_mondays n_fridays : ℕ) (h_mondays : february_has_n_mondays n_mondays) (h_fridays : february_has_n_fridays n_fridays) : 
  (1 : ℕ) % 7 = 5 :=
by
  sorry

end february_first_is_friday_l183_183869


namespace highest_place_value_quotient_and_remainder_l183_183206

-- Conditions
def dividend := 438
def divisor := 4

-- Theorem stating that the highest place value of the quotient is the hundreds place, and the remainder is 2
theorem highest_place_value_quotient_and_remainder : 
  (dividend = divisor * (dividend / divisor) + (dividend % divisor)) ∧ 
  ((dividend / divisor) >= 100) ∧ 
  ((dividend % divisor) = 2) :=
by
  sorry

end highest_place_value_quotient_and_remainder_l183_183206


namespace num_boys_and_girls_l183_183534

def num_ways_to_select (x : ℕ) := (x * (x - 1) / 2) * (8 - x) * 6

theorem num_boys_and_girls (x : ℕ) (h1 : num_ways_to_select x = 180) :
    x = 5 ∨ x = 6 :=
by
  sorry

end num_boys_and_girls_l183_183534


namespace ratio_shorter_to_longer_l183_183787

-- Constants for the problem
def total_length : ℝ := 49
def shorter_piece_length : ℝ := 14

-- Definition of longer piece length based on the given conditions
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- The theorem to be proved
theorem ratio_shorter_to_longer : 
  shorter_piece_length / longer_piece_length = 2 / 5 :=
by
  -- This is where the proof would go
  sorry

end ratio_shorter_to_longer_l183_183787


namespace max_discarded_grapes_l183_183970

theorem max_discarded_grapes (n : ℕ) : ∃ r, r < 8 ∧ n % 8 = r ∧ r = 7 :=
by
  sorry

end max_discarded_grapes_l183_183970


namespace hulk_jump_geometric_sequence_l183_183903

theorem hulk_jump_geometric_sequence (n : ℕ) (a_n : ℕ) : 
  (a_n = 3 * 2^(n - 1)) → (a_n > 3000) → n = 11 :=
by
  sorry

end hulk_jump_geometric_sequence_l183_183903


namespace zero_in_interval_l183_183186

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 3

theorem zero_in_interval : 
    (∀ x y : ℝ, 0 < x → x < y → f x < f y) → 
    (f 1 = -2) →
    (f 2 = Real.log 2 + 5) →
    (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by 
    sorry

end zero_in_interval_l183_183186


namespace solve_for_y_l183_183618

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 5 * y)) : y = 250 / 7 := 
by
  sorry

end solve_for_y_l183_183618


namespace inequality_proof_l183_183730

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h : a * b * c = 1) : 
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (c + a)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l183_183730


namespace position_of_2019_in_splits_l183_183341

def sum_of_consecutive_odds (n : ℕ) : ℕ :=
  n^2 - (n - 1)

theorem position_of_2019_in_splits : ∃ n : ℕ, sum_of_consecutive_odds n = 2019 ∧ n = 45 :=
by
  sorry

end position_of_2019_in_splits_l183_183341


namespace avg_weight_difference_l183_183650

-- Define the weights of the boxes following the given conditions.
def box1_weight : ℕ := 200
def box3_weight : ℕ := box1_weight + (25 * box1_weight / 100)
def box2_weight : ℕ := box3_weight + (20 * box3_weight / 100)
def box4_weight : ℕ := 350
def box5_weight : ℕ := box4_weight * 100 / 70

-- Define the average weight of the four heaviest boxes.
def avg_heaviest : ℕ := (box2_weight + box3_weight + box4_weight + box5_weight) / 4

-- Define the average weight of the four lightest boxes.
def avg_lightest : ℕ := (box1_weight + box2_weight + box3_weight + box4_weight) / 4

-- Define the difference between the average weights of the heaviest and lightest boxes.
def avg_difference : ℕ := avg_heaviest - avg_lightest

-- State the theorem with the expected result.
theorem avg_weight_difference : avg_difference = 75 :=
by
  -- Proof is not provided.
  sorry

end avg_weight_difference_l183_183650


namespace original_design_ratio_built_bridge_ratio_l183_183866

-- Definitions
variables (v1 v2 r1 r2 : ℝ)

-- Conditions as per the problem
def original_height_relation : Prop := v1 = 3 * v2
def built_radius_relation : Prop := r2 = 2 * r1

-- Prove the required ratios
theorem original_design_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v1 / r1 = 3 / 4) := sorry

theorem built_bridge_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v2 / r2 = 1 / 8) := sorry

end original_design_ratio_built_bridge_ratio_l183_183866


namespace cubic_roots_sum_of_cubes_l183_183578

theorem cubic_roots_sum_of_cubes (r s t a b c : ℚ) 
  (h1 : r + s + t = a) 
  (h2 : r * s + r * t + s * t = b)
  (h3 : r * s * t = c) 
  (h_poly : ∀ x : ℚ, x^3 - a*x^2 + b*x - c = 0 ↔ (x = r ∨ x = s ∨ x = t)) :
  r^3 + s^3 + t^3 = a^3 - 3 * a * b + 3 * c :=
sorry

end cubic_roots_sum_of_cubes_l183_183578


namespace minimum_value_l183_183846

open Real

noncomputable def curve (x : ℝ) : ℝ := x^3 - 2 * x^2 + 2
noncomputable def tangent (x : ℝ) : ℝ := 4 * x - 6
def line (m n l : ℝ) (x y : ℝ) : Prop := m * x + n * y = l

theorem minimum_value (m n l : ℝ) (A : ℝ × ℝ)
  (A_on_curve : A.2 = curve A.1)
  (tangent_eq : ∀ x, ∃ k, tangent x = curve x + k * (x - A.1))
  (A_on_line : line m n l A.1 A.2)
  (m_pos : 0 < m) (n_pos : 0 < n) :
  ∃ k : ℝ, (m * A.1 + n * A.2 = l) ∧ ((1 / m) + (2 / n) = k) ∧ k = 6 + 4 * sqrt 2 :=
sorry

end minimum_value_l183_183846


namespace expected_number_of_students_who_get_back_their_cards_l183_183356

-- Condition: There are 2012 students in a secondary school, each student writes a new year card,
-- the cards are mixed up and randomly distributed, and each student gets one and only one card.
noncomputable def students : List ℕ := List.range 2012

-- Define the indicator random variable that indicates if the i-th student gets their own card.
def indicator_variable (i : ℕ) : ℕ → ℕ :=
  λ card : ℕ => if i == card then 1 else 0

-- Define the random variable representing the number of students who get back their own cards.
def X (distribution : List ℕ) : ℕ :=
  List.foldr (λ i acc => indicator_variable i (List.nthLe distribution i (by simp))) 0 students

-- Define the expected value
def expected_value (students : List ℕ) : ℕ :=
  1

-- The final statement
theorem expected_number_of_students_who_get_back_their_cards :
  ∀ (distribution : List ℕ), List.length distribution = 2012 → X distribution = expected_value students :=
begin
  sorry
end

end expected_number_of_students_who_get_back_their_cards_l183_183356


namespace minimum_value_of_f_l183_183681

variable (a k : ℝ)
variable (k_gt_1 : k > 1)
variable (a_gt_0 : a > 0)

noncomputable def f (x : ℝ) : ℝ := k * Real.sqrt (a^2 + x^2) - x

theorem minimum_value_of_f : ∃ x_0, ∀ x, f a k x ≥ f a k x_0 ∧ f a k x_0 = a * Real.sqrt (k^2 - 1) :=
by
  sorry

end minimum_value_of_f_l183_183681


namespace distinct_points_count_l183_183547

theorem distinct_points_count :
  ∃ (P : Finset (ℝ × ℝ)), 
    (∀ p ∈ P, p.1^2 + p.2^2 = 1 ∧ p.1^2 + 9 * p.2^2 = 9) ∧ P.card = 2 :=
by
  sorry

end distinct_points_count_l183_183547
