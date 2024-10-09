import Mathlib

namespace not_buy_either_l2364_236415

-- Definitions
variables (n T C B : ℕ)
variables (h_n : n = 15)
variables (h_T : T = 9)
variables (h_C : C = 7)
variables (h_B : B = 3)

-- Theorem statement
theorem not_buy_either (n T C B : ℕ) (h_n : n = 15) (h_T : T = 9) (h_C : C = 7) (h_B : B = 3) :
  n - (T - B) - (C - B) - B = 2 :=
sorry

end not_buy_either_l2364_236415


namespace jason_initial_quarters_l2364_236435

theorem jason_initial_quarters (q_d q_n q_i : ℕ) (h1 : q_d = 25) (h2 : q_n = 74) :
  q_i = q_n - q_d → q_i = 49 :=
by
  sorry

end jason_initial_quarters_l2364_236435


namespace remainder_37_remainder_73_l2364_236402

theorem remainder_37 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 37 = 20 := sorry

theorem remainder_73 (N : ℕ) (k : ℕ) (h : N = 1554 * k + 131) : N % 73 = 58 := sorry

end remainder_37_remainder_73_l2364_236402


namespace car_speed_is_120_l2364_236414

theorem car_speed_is_120 (v t : ℝ) (h1 : v > 0) (h2 : t > 0) (h3 : v * t = 75)
  (h4 : 1.5 * v * (t - (12.5 / 60)) = 75) : v = 120 := by
  sorry

end car_speed_is_120_l2364_236414


namespace student_count_l2364_236483

open Nat

theorem student_count :
  ∃ n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by {
  -- placeholder for the proof
  sorry
}

end student_count_l2364_236483


namespace seconds_in_8_point_5_minutes_l2364_236412

def minutesToSeconds (minutes : ℝ) : ℝ := minutes * 60

theorem seconds_in_8_point_5_minutes : minutesToSeconds 8.5 = 510 := 
by
  sorry

end seconds_in_8_point_5_minutes_l2364_236412


namespace smallest_n_for_n_cubed_ends_in_888_l2364_236493

/-- Proof Problem: Prove that 192 is the smallest positive integer \( n \) such that the last three digits of \( n^3 \) are 888. -/
theorem smallest_n_for_n_cubed_ends_in_888 : ∃ n : ℕ, n > 0 ∧ (n^3 % 1000 = 888) ∧ ∀ m : ℕ, 0 < m ∧ (m^3 % 1000 = 888) → n ≤ m :=
by
  sorry

end smallest_n_for_n_cubed_ends_in_888_l2364_236493


namespace solve_system1_solve_system2_l2364_236476

theorem solve_system1 (x y : ℝ) (h1 : 2 * x + 3 * y = 9) (h2 : x = 2 * y + 1) : x = 3 ∧ y = 1 := 
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x - y = 6) (h2 : 3 * x + 2 * y = 2) : x = 2 ∧ y = -2 := 
by sorry

end solve_system1_solve_system2_l2364_236476


namespace find_d_l2364_236426

theorem find_d
  (a b c d : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (h_d_pos : d > 0)
  (h_max : a * 1 + d = 5)
  (h_min : a * (-1) + d = -3) :
  d = 1 := 
sorry

end find_d_l2364_236426


namespace sum_of_x_and_y_l2364_236405

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 8*x - 10*y + 5) : x + y = -1 := by
  sorry

end sum_of_x_and_y_l2364_236405


namespace square_of_binomial_l2364_236460

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (3 * x + b) ^ 2 = 9 * x^2 - 18 * x + a) ↔ a = 9 :=
by
  sorry

end square_of_binomial_l2364_236460


namespace fourth_watercraft_is_submarine_l2364_236482

-- Define the conditions as Lean definitions
def same_direction_speed (w1 w2 w3 w4 : Type) : Prop :=
  -- All watercraft are moving in the same direction at the same speed
  true

def separation (w1 w2 w3 w4 : Type) (d : ℝ) : Prop :=
  -- Each pair of watercraft is separated by distance d
  true

def cargo_ship (w : Type) : Prop := true
def fishing_boat (w : Type) : Prop := true
def passenger_vessel (w : Type) : Prop := true

-- Define that the fourth watercraft is unique
def unique_watercraft (w : Type) : Prop := true

-- Proof statement that the fourth watercraft is a submarine
theorem fourth_watercraft_is_submarine 
  (w1 w2 w3 w4 : Type)
  (h1 : same_direction_speed w1 w2 w3 w4)
  (h2 : separation w1 w2 w3 w4 100)
  (h3 : cargo_ship w1)
  (h4 : fishing_boat w2)
  (h5 : passenger_vessel w3) :
  unique_watercraft w4 := 
sorry

end fourth_watercraft_is_submarine_l2364_236482


namespace sum_of_coefficients_l2364_236451

-- Defining the given conditions
def vertex : ℝ × ℝ := (5, -4)
def point : ℝ × ℝ := (3, -2)

-- Defining the problem to prove the sum of the coefficients
theorem sum_of_coefficients (a b c : ℝ)
  (h_eq : ∀ y, 5 = a * ((-4) + y)^2 + c)
  (h_pt : 3 = a * ((-4) + (-2))^2 + b * (-2) + c) :
  a + b + c = -15 / 2 :=
sorry

end sum_of_coefficients_l2364_236451


namespace highlighter_count_l2364_236438

-- Define the quantities of highlighters.
def pinkHighlighters := 3
def yellowHighlighters := 7
def blueHighlighters := 5

-- Define the total number of highlighters.
def totalHighlighters := pinkHighlighters + yellowHighlighters + blueHighlighters

-- The theorem states that the total number of highlighters is 15.
theorem highlighter_count : totalHighlighters = 15 := by
  -- Proof skipped for now.
  sorry

end highlighter_count_l2364_236438


namespace largest_divisor_of_product_of_five_consecutive_integers_l2364_236437

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∀ (n : ℤ), ∃ (d : ℤ), d = 60 ∧ d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l2364_236437


namespace no_intersection_points_l2364_236446

theorem no_intersection_points :
  ¬ ∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|2 * x + 1| :=
by
  sorry

end no_intersection_points_l2364_236446


namespace xiaoming_comprehensive_score_l2364_236427

theorem xiaoming_comprehensive_score :
  ∀ (a b c d : ℝ),
  a = 92 → b = 90 → c = 88 → d = 95 →
  (0.4 * a + 0.3 * b + 0.2 * c + 0.1 * d) = 90.9 :=
by
  intros a b c d ha hb hc hd
  simp [ha, hb, hc, hd]
  norm_num
  done

end xiaoming_comprehensive_score_l2364_236427


namespace rectangular_field_area_l2364_236431

noncomputable def length : ℝ := 1.2
noncomputable def width : ℝ := (3/4) * length

theorem rectangular_field_area : (length * width = 1.08) :=
by 
  -- The proof steps would go here
  sorry

end rectangular_field_area_l2364_236431


namespace dave_deleted_17_apps_l2364_236470

-- Define the initial and final state of Dave's apps
def initial_apps : Nat := 10
def added_apps : Nat := 11
def apps_left : Nat := 4

-- The total number of apps before deletion
def total_apps : Nat := initial_apps + added_apps

-- The expected number of deleted apps
def deleted_apps : Nat := total_apps - apps_left

-- The proof statement
theorem dave_deleted_17_apps : deleted_apps = 17 := by
  -- detailed steps are not required
  sorry

end dave_deleted_17_apps_l2364_236470


namespace function_passes_through_point_l2364_236475

-- Lean 4 Statement
theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧ (a^(x-1) + 4) = y :=
by
  use 1
  use 5
  sorry

end function_passes_through_point_l2364_236475


namespace ants_movement_impossible_l2364_236439

theorem ants_movement_impossible (initial_positions final_positions : Fin 3 → ℝ × ℝ) :
  initial_positions 0 = (0,0) ∧ initial_positions 1 = (0,1) ∧ initial_positions 2 = (1,0) →
  final_positions 0 = (-1,0) ∧ final_positions 1 = (0,1) ∧ final_positions 2 = (1,0) →
  (∀ t : ℕ, ∃ m : Fin 3, 
    ∀ i : Fin 3, (i ≠ m → ∃ k l : ℝ, 
      (initial_positions i).2 - l * (initial_positions i).1 = 0 ∧ 
      ∀ (p : ℕ → ℝ × ℝ), p 0 = initial_positions i ∧ p t = final_positions i → 
      (p 0).1 + k * (p 0).2 = 0)) →
  false :=
by 
  sorry

end ants_movement_impossible_l2364_236439


namespace min_buses_needed_l2364_236484

theorem min_buses_needed (n : ℕ) : 325 / 45 ≤ n ∧ n < 325 / 45 + 1 ↔ n = 8 :=
by
  sorry

end min_buses_needed_l2364_236484


namespace geometric_seq_a5_l2364_236465

theorem geometric_seq_a5 : ∃ (a₁ q : ℝ), 0 < q ∧ a₁ + 2 * a₁ * q = 4 ∧ (a₁ * q^3)^2 = 4 * (a₁ * q^2) * (a₁ * q^6) ∧ (a₅ = a₁ * q^4) := 
  by
    sorry

end geometric_seq_a5_l2364_236465


namespace choose_president_and_secretary_l2364_236489

theorem choose_president_and_secretary (total_members boys girls : ℕ) (h_total : total_members = 30) (h_boys : boys = 18) (h_girls : girls = 12) : 
  (boys * girls = 216) :=
by
  sorry

end choose_president_and_secretary_l2364_236489


namespace opposite_of_2023_is_neg_2023_l2364_236455

theorem opposite_of_2023_is_neg_2023 : -2023 = -(2023) :=
by
  sorry

end opposite_of_2023_is_neg_2023_l2364_236455


namespace find_second_cert_interest_rate_l2364_236425

theorem find_second_cert_interest_rate
  (initial_investment : ℝ := 12000)
  (first_term_months : ℕ := 8)
  (first_interest_rate : ℝ := 8 / 100)
  (second_term_months : ℕ := 10)
  (final_amount : ℝ := 13058.40)
  : ∃ s : ℝ, (s = 3.984) := sorry

end find_second_cert_interest_rate_l2364_236425


namespace salary_increase_l2364_236480

theorem salary_increase (x : ℕ) (hB_C_sum : 2*x + 3*x = 6000) : 
  ((3 * x - 1 * x) / (1 * x) ) * 100 = 200 :=
by
  -- Placeholder for the proof
  sorry

end salary_increase_l2364_236480


namespace baseball_card_decrease_l2364_236400

theorem baseball_card_decrease (V : ℝ) (hV : V > 0) (x : ℝ) :
  (1 - x / 100) * (1 - 0.30) = 1 - 0.44 -> x = 20 :=
by {
  -- proof omitted 
  sorry
}

end baseball_card_decrease_l2364_236400


namespace solve_for_x_l2364_236450

theorem solve_for_x (x y z : ℝ) (h1 : x * y = 8 - 3 * x - 2 * y) 
                                  (h2 : y * z = 8 - 2 * y - 3 * z) 
                                  (h3 : x * z = 35 - 5 * x - 3 * z) : 
  x = 8 :=
sorry

end solve_for_x_l2364_236450


namespace lines_intersect_at_3_6_l2364_236422

theorem lines_intersect_at_3_6 (c d : ℝ) 
  (h1 : 3 = 2 * 6 + c) 
  (h2 : 6 = 2 * 3 + d) : 
  c + d = -9 := by 
  sorry

end lines_intersect_at_3_6_l2364_236422


namespace approximate_probability_hit_shot_l2364_236466

-- Define the data from the table
def shots : List ℕ := [10, 50, 100, 150, 200, 500, 1000, 2000]
def hits : List ℕ := [9, 40, 70, 108, 143, 361, 721, 1440]
def hit_rates : List ℚ := [0.9, 0.8, 0.7, 0.72, 0.715, 0.722, 0.721, 0.72]

-- State the theorem that the stabilized hit rate is approximately 0.72
theorem approximate_probability_hit_shot : 
  ∃ (p : ℚ), p = 0.72 ∧ 
  ∀ (n : ℕ), n ∈ [150, 200, 500, 1000, 2000] → 
     ∃ (r : ℚ), r = 0.72 ∧ 
     r = (hits.get ⟨shots.indexOf n, sorry⟩ : ℚ) / n := sorry

end approximate_probability_hit_shot_l2364_236466


namespace candy_left_l2364_236442

theorem candy_left (total_candy : ℕ) (eaten_per_person : ℕ) (number_of_people : ℕ)
  (h_total_candy : total_candy = 68)
  (h_eaten_per_person : eaten_per_person = 4)
  (h_number_of_people : number_of_people = 2) :
  total_candy - (eaten_per_person * number_of_people) = 60 :=
by
  sorry

end candy_left_l2364_236442


namespace second_number_is_11_l2364_236487

-- Define the conditions
variables (x : ℕ) (h1 : 5 * x = 55)

-- The theorem we want to prove
theorem second_number_is_11 : x = 11 :=
sorry

end second_number_is_11_l2364_236487


namespace Ana_age_eight_l2364_236485

theorem Ana_age_eight (A B n : ℕ) (h1 : A - 1 = 7 * (B - 1)) (h2 : A = 4 * B) (h3 : A - B = n) : A = 8 :=
by
  sorry

end Ana_age_eight_l2364_236485


namespace gas_station_total_boxes_l2364_236462

theorem gas_station_total_boxes
  (chocolate_boxes : ℕ)
  (sugar_boxes : ℕ)
  (gum_boxes : ℕ)
  (licorice_boxes : ℕ)
  (sour_boxes : ℕ)
  (h_chocolate : chocolate_boxes = 3)
  (h_sugar : sugar_boxes = 5)
  (h_gum : gum_boxes = 2)
  (h_licorice : licorice_boxes = 4)
  (h_sour : sour_boxes = 7) :
  chocolate_boxes + sugar_boxes + gum_boxes + licorice_boxes + sour_boxes = 21 := by
  sorry

end gas_station_total_boxes_l2364_236462


namespace solve_inequality_l2364_236443

noncomputable def inequality_statement (x : ℝ) : Prop :=
  2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 20

theorem solve_inequality (x : ℝ) :
  (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (inequality_statement x ↔ (x < -3 ∨ (-2 < x ∧ x < 2) ∨ (3 < x ∧ x < 4) ∨ (5 < x ∧ x < 7) ∨ 8 < x)) :=
by sorry

end solve_inequality_l2364_236443


namespace problem1_problem2_l2364_236403

theorem problem1 (x : ℝ) : (4 * x ^ 2 + 12 * x - 7 ≤ 0) ∧ (a = 0) ∧ (x < -3 ∨ x > 3) → (-7/2 ≤ x ∧ x < -3) := by
  sorry

theorem problem2 (a : ℝ) : (∀ x : ℝ, 4 * x ^ 2 + 12 * x - 7 ≤ 0 → a - 3 ≤ x ∧ x ≤ a + 3) → (-5/2 ≤ a ∧ a ≤ -1/2) := by
  sorry

end problem1_problem2_l2364_236403


namespace sine_sum_zero_l2364_236486

open Real 

theorem sine_sum_zero (α β γ : ℝ) :
  (sin α / (sin (α - β) * sin (α - γ))
  + sin β / (sin (β - α) * sin (β - γ))
  + sin γ / (sin (γ - α) * sin (γ - β)) = 0) :=
sorry

end sine_sum_zero_l2364_236486


namespace remainder_sum_div_11_l2364_236407

theorem remainder_sum_div_11 :
  ((100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007 + 100008 + 100009 + 100010) % 11) = 10 :=
by
  sorry

end remainder_sum_div_11_l2364_236407


namespace minimum_candies_to_identify_coins_l2364_236420

-- Set up the problem: define the relevant elements.
inductive Coin : Type
| C1 : Coin
| C2 : Coin
| C3 : Coin
| C4 : Coin
| C5 : Coin

def values : List ℕ := [1, 2, 5, 10, 20]

-- Statement of the problem in Lean 4, no means to identify which is which except through purchases and change from vending machine.
theorem minimum_candies_to_identify_coins : ∃ n : ℕ, n = 4 :=
by
  -- Skipping the proof
  sorry

end minimum_candies_to_identify_coins_l2364_236420


namespace original_profit_percentage_l2364_236497

theorem original_profit_percentage (C S : ℝ) 
  (h1 : S - 1.12 * C = 0.5333333333333333 * S) : 
  ((S - C) / C) * 100 = 140 :=
sorry

end original_profit_percentage_l2364_236497


namespace total_shaded_area_is_2pi_l2364_236458

theorem total_shaded_area_is_2pi (sm_radius large_radius : ℝ) 
  (h_sm_radius : sm_radius = 1) 
  (h_large_radius : large_radius = 2) 
  (sm_circle_area large_circle_area total_shaded_area : ℝ) 
  (h_sm_circle_area : sm_circle_area = π * sm_radius^2) 
  (h_large_circle_area : large_circle_area = π * large_radius^2) 
  (h_total_shaded_area : total_shaded_area = large_circle_area - 2 * sm_circle_area) :
  total_shaded_area = 2 * π :=
by
  -- Proof goes here
  sorry

end total_shaded_area_is_2pi_l2364_236458


namespace find_total_students_l2364_236444

variables (x X : ℕ)
variables (x_percent_students : ℕ) (total_students : ℕ)
variables (boys_fraction : ℝ)

-- Provided Conditions
axiom a1 : x_percent_students = 120
axiom a2 : boys_fraction = 0.30
axiom a3 : total_students = X

-- The theorem we need to prove
theorem find_total_students (a1 : 120 = x_percent_students) 
                            (a2 : boys_fraction = 0.30) 
                            (a3 : total_students = X) : 
  120 = (x / 100) * (boys_fraction * total_students) :=
sorry

end find_total_students_l2364_236444


namespace circle_radius_formula_correct_l2364_236421

noncomputable def touch_circles_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  let numerator := c * Real.sqrt ((s - a) * (s - b) * (s - c))
  let denominator := c * Real.sqrt s + 2 * Real.sqrt ((s - a) * (s - b) * (s - c))
  numerator / denominator

theorem circle_radius_formula_correct (a b c : ℝ) : 
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  ∀ (r : ℝ), (r = touch_circles_radius a b c) :=
sorry

end circle_radius_formula_correct_l2364_236421


namespace A_ge_B_l2364_236406

def A (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b^2 + 2 * b^2 + 3 * b
def B (a b : ℝ) : ℝ := a^3 - a^2 * b^2 + b^2 + 3 * b

theorem A_ge_B (a b : ℝ) : A a b ≥ B a b := by
  sorry

end A_ge_B_l2364_236406


namespace hands_coincide_again_l2364_236492

-- Define the angular speeds of minute and hour hands
def speed_minute_hand : ℝ := 6
def speed_hour_hand : ℝ := 0.5

-- Define the initial condition: coincidence at midnight
def initial_time : ℝ := 0

-- Define the function that calculates the angle of the minute hand at time t
def angle_minute_hand (t : ℝ) : ℝ := speed_minute_hand * t

-- Define the function that calculates the angle of the hour hand at time t
def angle_hour_hand (t : ℝ) : ℝ := speed_hour_hand * t

-- Define the time at which the hands coincide again after midnight
noncomputable def coincidence_time : ℝ := 720 / 11

-- The proof problem statement: The hands coincide again at coincidence_time minutes
theorem hands_coincide_again : 
  angle_minute_hand coincidence_time = angle_hour_hand coincidence_time + 360 :=
sorry

end hands_coincide_again_l2364_236492


namespace largest_two_numbers_l2364_236491

def a : Real := 2^(1/2)
def b : Real := 3^(1/3)
def c : Real := 8^(1/8)
def d : Real := 9^(1/9)

theorem largest_two_numbers : 
  (max (max (max a b) c) d = b) ∧ 
  (max (max a c) d = a) := 
sorry

end largest_two_numbers_l2364_236491


namespace problem_set_equiv_l2364_236452

def positive_nats (x : ℕ) : Prop := x > 0

def problem_set : Set ℕ := {x | positive_nats x ∧ x - 3 < 2}

theorem problem_set_equiv : problem_set = {1, 2, 3, 4} :=
by 
  sorry

end problem_set_equiv_l2364_236452


namespace geometric_sequence_general_term_l2364_236454

variable (a : ℕ → ℝ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n
  
theorem geometric_sequence_general_term 
  (h_geo : is_geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  a n = 3 * 2^(n-3) :=
by sorry

end geometric_sequence_general_term_l2364_236454


namespace age_of_b_l2364_236411

variable (a b c d : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : d = b / 2)
variable (h4 : a + b + c + d = 44)

theorem age_of_b : b = 14 :=
by 
  sorry

end age_of_b_l2364_236411


namespace charles_nickels_l2364_236423

theorem charles_nickels :
  ∀ (num_pennies num_cents penny_value nickel_value n : ℕ),
  num_pennies = 6 →
  num_cents = 21 →
  penny_value = 1 →
  nickel_value = 5 →
  (num_cents - num_pennies * penny_value) / nickel_value = n →
  n = 3 :=
by
  intros num_pennies num_cents penny_value nickel_value n hnum_pennies hnum_cents hpenny_value hnickel_value hn
  sorry

end charles_nickels_l2364_236423


namespace irrational_pi_l2364_236434

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem irrational_pi : is_irrational π := by
  sorry

end irrational_pi_l2364_236434


namespace imaginary_part_z_l2364_236490

theorem imaginary_part_z : 
  ∀ (z : ℂ), z = (5 - I) / (1 - I) → z.im = 2 := 
by
  sorry

end imaginary_part_z_l2364_236490


namespace fish_count_together_l2364_236463

namespace FishProblem

def JerkTunaFish : ℕ := 144
def TallTunaFish : ℕ := 2 * JerkTunaFish
def SwellTunaFish : ℕ := TallTunaFish + (TallTunaFish / 2)
def totalFish : ℕ := JerkTunaFish + TallTunaFish + SwellTunaFish

theorem fish_count_together : totalFish = 864 := by
  sorry

end FishProblem

end fish_count_together_l2364_236463


namespace sequence_either_increases_or_decreases_l2364_236494

theorem sequence_either_increases_or_decreases {x : ℕ → ℝ} (x1_pos : 0 < x 1) (x1_ne_one : x 1 ≠ 1) 
    (recurrence : ∀ n : ℕ, x (n + 1) = x n * (x n ^ 2 + 3) / (3 * x n ^ 2 + 1)) :
    (∀ n : ℕ, x n < x (n + 1)) ∨ (∀ n : ℕ, x n > x (n + 1)) :=
sorry

end sequence_either_increases_or_decreases_l2364_236494


namespace sarah_mean_score_l2364_236488

noncomputable def john_mean_score : ℝ := 86
noncomputable def john_num_tests : ℝ := 4
noncomputable def test_scores : List ℝ := [78, 80, 85, 87, 90, 95, 100]
noncomputable def total_sum : ℝ := test_scores.sum
noncomputable def sarah_num_tests : ℝ := 3

theorem sarah_mean_score :
  let john_total_score := john_mean_score * john_num_tests
  let sarah_total_score := total_sum - john_total_score
  let sarah_mean_score := sarah_total_score / sarah_num_tests
  sarah_mean_score = 90.3 :=
by
  sorry

end sarah_mean_score_l2364_236488


namespace sum_first_seven_terms_geometric_sequence_l2364_236453

noncomputable def sum_geometric_sequence (a r : ℚ) (n : ℕ) : ℚ := 
  a * (1 - r^n) / (1 - r)

theorem sum_first_seven_terms_geometric_sequence : 
  sum_geometric_sequence (1/4) (1/4) 7 = 16383 / 49152 := 
by
  sorry

end sum_first_seven_terms_geometric_sequence_l2364_236453


namespace average_visitors_on_sundays_l2364_236477

theorem average_visitors_on_sundays 
  (avg_other_days : ℕ) (avg_per_day : ℕ) (days_in_month : ℕ) (sundays : ℕ) (S : ℕ)
  (h_avg_other_days : avg_other_days = 240)
  (h_avg_per_day : avg_per_day = 310)
  (h_days_in_month : days_in_month = 30)
  (h_sundays : sundays = 5) :
  (sundays * S + (days_in_month - sundays) * avg_other_days = avg_per_day * days_in_month) → 
  S = 660 :=
by
  intros h
  rw [h_avg_other_days, h_avg_per_day, h_days_in_month, h_sundays] at h
  sorry

end average_visitors_on_sundays_l2364_236477


namespace P_at_10_l2364_236416

-- Define the main properties of the polynomial
variable (P : ℤ → ℤ)
axiom quadratic (a b c : ℤ) : (∀ n : ℤ, P n = a * n^2 + b * n + c) 

-- Conditions for the polynomial
axiom int_coefficients : ∃ (a b c : ℤ), ∀ n : ℤ, P n = a * n^2 + b * n + c
axiom relatively_prime (n : ℤ) (hn : 0 < n) : Int.gcd (P n) n = 1 ∧ Int.gcd (P (P n)) n = 1
axiom P_at_3 : P 3 = 89

-- The main theorem to prove
theorem P_at_10 : P 10 = 859 := by sorry

end P_at_10_l2364_236416


namespace nth_equation_l2364_236469

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := by
  sorry

end nth_equation_l2364_236469


namespace find_AX_l2364_236498

theorem find_AX
  (AB AC BC : ℚ)
  (H : AB = 80)
  (H1 : AC = 50)
  (H2 : BC = 30)
  (angle_bisector_theorem_1 : ∀ (AX XC y : ℚ), AX = 8 * y ∧ XC = 3 * y ∧ 11 * y = AC → y = 50 / 11)
  (angle_bisector_theorem_2 : ∀ (BD DC z : ℚ), BD = 8 * z ∧ DC = 5 * z ∧ 13 * z = BC → z = 30 / 13) :
  AX = 400 / 11 := 
sorry

end find_AX_l2364_236498


namespace ones_digit_of_prime_in_arithmetic_sequence_is_one_l2364_236456

theorem ones_digit_of_prime_in_arithmetic_sequence_is_one 
  (p q r s : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (hs : Prime s) 
  (h₁ : p > 10) 
  (h₂ : q = p + 10) 
  (h₃ : r = q + 10) 
  (h₄ : s = r + 10) 
  (h₅ : s > r) 
  (h₆ : r > q) 
  (h₇ : q > p) : 
  p % 10 = 1 :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_is_one_l2364_236456


namespace number_of_episodes_last_season_more_than_others_l2364_236447

-- Definitions based on conditions
def episodes_per_other_season : ℕ := 22
def initial_seasons : ℕ := 9
def duration_per_episode : ℚ := 0.5
def total_hours_after_last_season : ℚ := 112

-- Derived definitions based on conditions (not solution steps)
def total_hours_first_9_seasons := initial_seasons * episodes_per_other_season * duration_per_episode
def additional_hours_last_season := total_hours_after_last_season - total_hours_first_9_seasons
def episodes_last_season := additional_hours_last_season / duration_per_episode

-- Proof problem statement
theorem number_of_episodes_last_season_more_than_others : 
  episodes_last_season = episodes_per_other_season + 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_episodes_last_season_more_than_others_l2364_236447


namespace donut_selection_l2364_236410

-- Lean statement for the proof problem
theorem donut_selection (n k : ℕ) (h1 : n = 5) (h2 : k = 4) : (n + k - 1).choose (k - 1) = 56 :=
by
  rw [h1, h2]
  sorry

end donut_selection_l2364_236410


namespace problem_solution_l2364_236461

theorem problem_solution (x : ℝ) (h : ∃ (A B : Set ℝ), A = {0, 1, 2, 4, 5} ∧ B = {x-2, x, x+2} ∧ A ∩ B = {0, 2}) : x = 0 :=
sorry

end problem_solution_l2364_236461


namespace find_m_values_l2364_236430

theorem find_m_values {m : ℝ} :
  (∀ x : ℝ, mx^2 + (m+2) * x + (1 / 2) * m + 1 = 0 → x = 0) 
  ↔ (m = 0 ∨ m = 2 ∨ m = -2) :=
by sorry

end find_m_values_l2364_236430


namespace total_votes_l2364_236408

theorem total_votes (V : ℝ) (h1 : 0.70 * V = V - 240) (h2 : 0.30 * V = 240) : V = 800 :=
by
  sorry

end total_votes_l2364_236408


namespace probability_xi_l2364_236418

noncomputable def xi_distribution (k : ℕ) : ℚ :=
  if h : k > 0 then 1 / (2 : ℚ)^k else 0

theorem probability_xi (h : ∀ k : ℕ, k > 0 → xi_distribution k = 1 / (2 : ℚ)^k) :
  (xi_distribution 3 + xi_distribution 4) = 3 / 16 :=
by
  sorry

end probability_xi_l2364_236418


namespace original_decimal_l2364_236473

variable (x : ℝ)

theorem original_decimal (h : x - x / 100 = 1.485) : x = 1.5 :=
sorry

end original_decimal_l2364_236473


namespace water_fee_relationship_xiao_qiangs_water_usage_l2364_236417

variable (x y : ℝ)
variable (H1 : x > 10)
variable (H2 : y = 3 * x - 8)

theorem water_fee_relationship : y = 3 * x - 8 := 
  by 
    exact H2

theorem xiao_qiangs_water_usage : y = 67 → x = 25 :=
  by
    intro H
    have H_eq : 67 = 3 * x - 8 := by 
      rw [←H2, H]
    linarith

end water_fee_relationship_xiao_qiangs_water_usage_l2364_236417


namespace bug_total_distance_l2364_236449

theorem bug_total_distance : 
  let start := 3
  let first_point := 9
  let second_point := -4
  let distance_1 := abs (first_point - start)
  let distance_2 := abs (second_point - first_point)
  distance_1 + distance_2 = 19 := 
by
  sorry

end bug_total_distance_l2364_236449


namespace smallest_number_of_students_l2364_236440

-- Define the structure of the problem
def unique_row_configurations (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∣ n → k < 10) → ∃ divs : Finset ℕ, divs.card = 9 ∧ ∀ d ∈ divs, d ∣ n ∧ (∀ d' ∈ divs, d ≠ d') 

-- The main statement to be proven in Lean 4
theorem smallest_number_of_students : ∃ n : ℕ, unique_row_configurations n ∧ n = 36 :=
by
  sorry

end smallest_number_of_students_l2364_236440


namespace problem_solution_l2364_236448

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)
variables (h3 : 3 * log 101 ((1030301 - a - b) / (3 * a * b)) = 3 - 2 * log 101 (a * b))

theorem problem_solution : 101 - (a)^(1/3) - (b)^(1/3) = 0 :=
by
  sorry

end problem_solution_l2364_236448


namespace sum_of_reciprocals_l2364_236457

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) : (1/x) + (1/y) = 3/8 :=
by
  sorry

end sum_of_reciprocals_l2364_236457


namespace inequality_of_function_l2364_236496

theorem inequality_of_function (x : ℝ) : 
  (1 / 2 : ℝ) ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ (3 / 2 : ℝ) :=
sorry

end inequality_of_function_l2364_236496


namespace right_triangle_side_lengths_l2364_236495

theorem right_triangle_side_lengths (x : ℝ) :
  (2 * x + 2)^2 + (x + 2)^2 = (x + 4)^2 ∨ (2 * x + 2)^2 + (x + 4)^2 = (x + 2)^2 ↔ (x = 1 ∨ x = 4) :=
by sorry

end right_triangle_side_lengths_l2364_236495


namespace madeline_water_intake_l2364_236432

def water_bottle_capacity : ℕ := 12
def number_of_refills : ℕ := 7
def additional_water_needed : ℕ := 16
def total_water_needed : ℕ := 100

theorem madeline_water_intake : water_bottle_capacity * number_of_refills + additional_water_needed = total_water_needed :=
by
  sorry

end madeline_water_intake_l2364_236432


namespace markers_per_box_l2364_236424

theorem markers_per_box (original_markers new_boxes total_markers : ℕ) 
    (h1 : original_markers = 32) (h2 : new_boxes = 6) (h3 : total_markers = 86) : 
    total_markers - original_markers = new_boxes * 9 :=
by sorry

end markers_per_box_l2364_236424


namespace red_cars_count_l2364_236468

variable (R B : ℕ)
variable (h1 : R * 8 = 3 * B)
variable (h2 : B = 90)

theorem red_cars_count : R = 33 :=
by
  -- here we would provide the proof
  sorry

end red_cars_count_l2364_236468


namespace terminating_decimal_count_l2364_236428

theorem terminating_decimal_count : ∃ n, n = 23 ∧ (∀ k, 1 ≤ k ∧ k ≤ 499 → (∃ m, k = 21 * m)) :=
by
  sorry

end terminating_decimal_count_l2364_236428


namespace fraction_ordering_l2364_236472

theorem fraction_ordering :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  (b < c) ∧ (c < a) :=
by
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 14 - 1 / 56
  let c := (6 : ℚ) / 17
  sorry

end fraction_ordering_l2364_236472


namespace routes_from_M_to_N_l2364_236401

structure Paths where
  -- Specify the paths between nodes
  C_to_N : ℕ
  D_to_N : ℕ
  A_to_C : ℕ
  A_to_D : ℕ
  B_to_N : ℕ
  B_to_A : ℕ
  B_to_C : ℕ
  M_to_B : ℕ
  M_to_A : ℕ

theorem routes_from_M_to_N (p : Paths) : 
  p.C_to_N = 1 → 
  p.D_to_N = 1 →
  p.A_to_C = 1 →
  p.A_to_D = 1 →
  p.B_to_N = 1 →
  p.B_to_A = 1 →
  p.B_to_C = 1 →
  p.M_to_B = 1 →
  p.M_to_A = 1 →
  (p.M_to_B * (p.B_to_N + (p.B_to_A * (p.A_to_C + p.A_to_D)) + p.B_to_C)) + 
  (p.M_to_A * (p.A_to_C + p.A_to_D)) = 6 
:= by
  sorry

end routes_from_M_to_N_l2364_236401


namespace technician_round_trip_l2364_236464

-- Definitions based on conditions
def trip_to_center_completion : ℝ := 0.5 -- Driving to the center is 50% of the trip
def trip_from_center_completion (percent_completed: ℝ) : ℝ := 0.5 * percent_completed -- Completion percentage of the return trip
def total_trip_completion : ℝ := trip_to_center_completion + trip_from_center_completion 0.3 -- Total percentage completed

-- Theorem statement
theorem technician_round_trip : total_trip_completion = 0.65 :=
by
  sorry

end technician_round_trip_l2364_236464


namespace total_items_l2364_236471

theorem total_items (slices_of_bread bottles_of_milk cookies : ℕ) (h1 : slices_of_bread = 58)
  (h2 : bottles_of_milk = slices_of_bread - 18) (h3 : cookies = slices_of_bread + 27) :
  slices_of_bread + bottles_of_milk + cookies = 183 :=
by
  sorry

end total_items_l2364_236471


namespace p_minus_q_l2364_236413

-- Define the given equation as a predicate.
def eqn (x : ℝ) : Prop := (3*x - 9) / (x*x + 3*x - 18) = x + 3

-- Define the values p and q as distinct solutions.
def p_and_q (p q : ℝ) : Prop := eqn p ∧ eqn q ∧ p ≠ q ∧ p > q

theorem p_minus_q {p q : ℝ} (h : p_and_q p q) : p - q = 2 := sorry

end p_minus_q_l2364_236413


namespace rhombus_side_length_l2364_236404

/-
  Define the length of the rhombus diagonal and the area of the rhombus.
-/
def diagonal1 : ℝ := 20
def area : ℝ := 480

/-
  The theorem states that given these conditions, the length of each side of the rhombus is 26 m.
-/
theorem rhombus_side_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = diagonal1) (h2 : A = area):
  2 * 26 * 26 * 2 = A * 2 * 2 + (d1 / 2) * (d1 / 2) :=
sorry

end rhombus_side_length_l2364_236404


namespace petya_numbers_l2364_236467

-- Define the arithmetic sequence property
def arithmetic_seq (a d : ℕ) : ℕ → ℕ
| 0     => a
| (n+1) => a + (n + 1) * d

-- Given conditions
theorem petya_numbers (a d : ℕ) : 
  (arithmetic_seq a d 0 = 6) ∧
  (arithmetic_seq a d 1 = 15) ∧
  (arithmetic_seq a d 2 = 24) ∧
  (arithmetic_seq a d 3 = 33) ∧
  (arithmetic_seq a d 4 = 42) :=
sorry

end petya_numbers_l2364_236467


namespace divisible_by_91_l2364_236436

def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 202020
  | _ => -- Define the sequence here, ensuring it constructs the number properly with inserted '2's
    sorry -- this might be a more complex function to define

theorem divisible_by_91 (n : ℕ) : 91 ∣ a n :=
  sorry

end divisible_by_91_l2364_236436


namespace shorter_piece_length_l2364_236499

/-- A 69-inch board is cut into 2 pieces. One piece is 2 times the length of the other.
    Prove that the length of the shorter piece is 23 inches. -/
theorem shorter_piece_length (x : ℝ) :
  let shorter := x
  let longer := 2 * x
  (shorter + longer = 69) → shorter = 23 :=
by
  intro h
  sorry

end shorter_piece_length_l2364_236499


namespace paint_cost_per_liter_l2364_236478

def cost_brush : ℕ := 20
def cost_canvas : ℕ := 3 * cost_brush
def min_liters : ℕ := 5
def total_earning : ℕ := 200
def total_profit : ℕ := 80
def total_cost : ℕ := total_earning - total_profit

theorem paint_cost_per_liter :
  (total_cost = cost_brush + cost_canvas + (5 * 8)) :=
by
  sorry

end paint_cost_per_liter_l2364_236478


namespace remaining_players_average_points_l2364_236433

-- Define the conditions
def total_points : ℕ := 270
def total_players : ℕ := 9
def players_averaged_50 : ℕ := 5
def average_points_50 : ℕ := 50

-- Define the query
theorem remaining_players_average_points :
  (total_points - players_averaged_50 * average_points_50) / (total_players - players_averaged_50) = 5 :=
by
  sorry

end remaining_players_average_points_l2364_236433


namespace laura_mowing_time_correct_l2364_236441

noncomputable def laura_mowing_time : ℝ := 
  let combined_time := 1.71428571429
  let sammy_time := 3
  let combined_rate := 1 / combined_time
  let sammy_rate := 1 / sammy_time
  let laura_rate := combined_rate - sammy_rate
  1 / laura_rate

theorem laura_mowing_time_correct : laura_mowing_time = 4.2 := 
  by
    sorry

end laura_mowing_time_correct_l2364_236441


namespace emily_collected_8484_eggs_l2364_236429

def number_of_baskets : ℕ := 303
def eggs_per_basket : ℕ := 28
def total_eggs : ℕ := number_of_baskets * eggs_per_basket

theorem emily_collected_8484_eggs : total_eggs = 8484 :=
by
  sorry

end emily_collected_8484_eggs_l2364_236429


namespace johann_mail_l2364_236479

def pieces_of_mail_total : ℕ := 180
def pieces_of_mail_friends : ℕ := 41
def friends : ℕ := 2
def pieces_of_mail_johann : ℕ := pieces_of_mail_total - (pieces_of_mail_friends * friends)

theorem johann_mail : pieces_of_mail_johann = 98 := by
  sorry

end johann_mail_l2364_236479


namespace max_value_f_on_interval_l2364_236445

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, (∀ y ∈ Set.Icc (-1 : ℝ) 1, f y ≤ f x) ∧ f x = Real.exp 1 - 1 :=
sorry

end max_value_f_on_interval_l2364_236445


namespace find_n_l2364_236419

def P_X_eq_2 (n : ℕ) : Prop :=
  (3 * n) / ((n + 3) * (n + 2)) = (7 : ℚ) / 30

theorem find_n (n : ℕ) (h : P_X_eq_2 n) : n = 7 :=
by sorry

end find_n_l2364_236419


namespace range_of_a_l2364_236481

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + Real.exp (x + 2) - 2 * Real.exp 4
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * x - 3 * a * Real.exp x
def A : Set ℝ := { x | f x = 0 }
def B (a : ℝ) : Set ℝ := { x | g x a = 0 }

theorem range_of_a (a : ℝ) :
  (∃ x₁ ∈ A, ∃ x₂ ∈ B a, |x₁ - x₂| < 1) →
  a ∈ Set.Ici (1 / (3 * Real.exp 1)) ∩ Set.Iic (4 / (3 * Real.exp 4)) :=
sorry

end range_of_a_l2364_236481


namespace distance_l1_l2_l2364_236474

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

theorem distance_l1_l2 :
  distance_between_parallel_lines 3 4 (-3) 2 = 1 :=
by
  -- Add the conditions needed to assert the theorem
  let l1 := (3, 4, -3) -- definition of line l1
  let l2 := (3, 4, 2)  -- definition of line l2
  -- Calculate the distance using the given formula
  let d := distance_between_parallel_lines 3 4 (-3) 2
  -- Assert the result
  show d = 1
  sorry

end distance_l1_l2_l2364_236474


namespace minimum_value_of_f_l2364_236409

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 5)

theorem minimum_value_of_f : ∃ (x : ℝ), x > 5 ∧ f x = 20 :=
by
  use 10
  sorry

end minimum_value_of_f_l2364_236409


namespace trip_time_80_minutes_l2364_236459

noncomputable def v : ℝ := 1 / 2
noncomputable def speed_highway := 4 * v -- 4 times speed on the highway
noncomputable def time_mountain : ℝ := 20 / v -- Distance on mountain road divided by speed on mountain road
noncomputable def time_highway : ℝ := 80 / speed_highway -- Distance on highway divided by speed on highway
noncomputable def total_time := time_mountain + time_highway

theorem trip_time_80_minutes : total_time = 80 :=
by sorry

end trip_time_80_minutes_l2364_236459
