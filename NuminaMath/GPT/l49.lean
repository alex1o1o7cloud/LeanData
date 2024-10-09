import Mathlib

namespace shortest_wire_length_l49_4969

theorem shortest_wire_length
  (d1 d2 : ℝ) (h_d1 : d1 = 10) (h_d2 : d2 = 30) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let straight_sections := 2 * (r2 - r1)
  let curved_sections := 2 * Real.pi * r1 + 2 * Real.pi * r2
  let total_wire_length := straight_sections + curved_sections
  total_wire_length = 20 + 40 * Real.pi :=
by
  sorry

end shortest_wire_length_l49_4969


namespace acute_angled_triangle_range_l49_4973

theorem acute_angled_triangle_range (x : ℝ) (h : (x^2 + 6)^2 < (x^2 + 4)^2 + (4 * x)^2) : x > (Real.sqrt 15) / 3 := sorry

end acute_angled_triangle_range_l49_4973


namespace incorrect_regression_intercept_l49_4996

theorem incorrect_regression_intercept (points : List (ℕ × ℝ)) (h_points : points = [(1, 0.5), (2, 0.8), (3, 1.0), (4, 1.2), (5, 1.5)]) :
  ¬ (∃ (a : ℝ), a = 0.26 ∧ ∀ x : ℕ, x ∈ ([1, 2, 3, 4, 5] : List ℕ) → (∃ y : ℝ, y = 0.24 * x + a)) := sorry

end incorrect_regression_intercept_l49_4996


namespace number_of_triangles_l49_4935

-- Defining the problem conditions
def ten_points : Finset (ℕ) := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The main theorem to prove
theorem number_of_triangles : (ten_points.card.choose 3) = 120 :=
by
  sorry

end number_of_triangles_l49_4935


namespace solution_set_m5_range_m_sufficient_condition_l49_4907

theorem solution_set_m5 (x : ℝ) : 
  (|x + 1| + |x - 2| > 5) ↔ (x < -2 ∨ x > 3) := 
sorry

theorem range_m_sufficient_condition (x m : ℝ) (h : ∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) : 
  m ≤ 1 := 
sorry

end solution_set_m5_range_m_sufficient_condition_l49_4907


namespace integer_solutions_count_l49_4932

theorem integer_solutions_count : 
  ∃ (s : Finset ℤ), 
    (∀ x ∈ s, 2 * x + 1 > -3 ∧ -x + 3 ≥ 0) ∧ 
    s.card = 5 := 
by 
  sorry

end integer_solutions_count_l49_4932


namespace time_for_one_mile_l49_4941

theorem time_for_one_mile (d v : ℝ) (mile_in_feet : ℝ) (num_circles : ℕ) 
  (circle_circumference : ℝ) (distance_in_miles : ℝ) (time : ℝ) :
  d = 50 ∧ v = 10 ∧ mile_in_feet = 5280 ∧ num_circles = 106 ∧ 
  circle_circumference = 50 * Real.pi ∧ 
  distance_in_miles = (106 * 50 * Real.pi) / 5280 ∧ 
  time = distance_in_miles / v →
  time = Real.pi / 10 :=
by {
  sorry
}

end time_for_one_mile_l49_4941


namespace cos_75_eq_sqrt6_sub_sqrt2_div_4_l49_4968

theorem cos_75_eq_sqrt6_sub_sqrt2_div_4 :
  Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := sorry

end cos_75_eq_sqrt6_sub_sqrt2_div_4_l49_4968


namespace alpha_arctan_l49_4988

open Real

theorem alpha_arctan {α : ℝ} (h1 : α ∈ Set.Ioo 0 (π/4)) (h2 : tan (α + (π/4)) = 2 * cos (2 * α)) : 
  α = arctan (2 - sqrt 3) := by
  sorry

end alpha_arctan_l49_4988


namespace gcd_16016_20020_l49_4912

theorem gcd_16016_20020 : Int.gcd 16016 20020 = 4004 :=
by
  sorry

end gcd_16016_20020_l49_4912


namespace negation_of_universal_l49_4944

theorem negation_of_universal:
  ¬(∀ x : ℕ, x^2 > 1) ↔ ∃ x : ℕ, x^2 ≤ 1 :=
by sorry

end negation_of_universal_l49_4944


namespace sqrt_31_between_5_and_6_l49_4904

theorem sqrt_31_between_5_and_6
  (h1 : Real.sqrt 25 = 5)
  (h2 : Real.sqrt 36 = 6)
  (h3 : 25 < 31)
  (h4 : 31 < 36) :
  5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 :=
sorry

end sqrt_31_between_5_and_6_l49_4904


namespace zachary_needs_more_money_l49_4956

def cost_in_usd_football (euro_to_usd : ℝ) (football_cost_eur : ℝ) : ℝ :=
  football_cost_eur * euro_to_usd

def cost_in_usd_shorts (gbp_to_usd : ℝ) (shorts_cost_gbp : ℝ) (pairs : ℕ) : ℝ :=
  shorts_cost_gbp * pairs * gbp_to_usd

def cost_in_usd_shoes (shoes_cost_usd : ℝ) : ℝ :=
  shoes_cost_usd

def cost_in_usd_socks (jpy_to_usd : ℝ) (socks_cost_jpy : ℝ) (pairs : ℕ) : ℝ :=
  socks_cost_jpy * pairs * jpy_to_usd

def cost_in_usd_water_bottle (krw_to_usd : ℝ) (water_bottle_cost_krw : ℝ) : ℝ :=
  water_bottle_cost_krw * krw_to_usd

def total_cost_before_discount (cost_football_usd cost_shorts_usd cost_shoes_usd
                                cost_socks_usd cost_water_bottle_usd : ℝ) : ℝ :=
  cost_football_usd + cost_shorts_usd + cost_shoes_usd + cost_socks_usd + cost_water_bottle_usd

def discounted_total_cost (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost * (1 - discount)

def additional_money_needed (discounted_total_cost current_money : ℝ) : ℝ :=
  discounted_total_cost - current_money

theorem zachary_needs_more_money (euro_to_usd : ℝ) (gbp_to_usd : ℝ) (jpy_to_usd : ℝ) (krw_to_usd : ℝ)
  (football_cost_eur : ℝ) (shorts_cost_gbp : ℝ) (pairs_shorts : ℕ) (shoes_cost_usd : ℝ)
  (socks_cost_jpy : ℝ) (pairs_socks : ℕ) (water_bottle_cost_krw : ℝ) (current_money_usd : ℝ)
  (discount : ℝ) : additional_money_needed 
      (discounted_total_cost
          (total_cost_before_discount
            (cost_in_usd_football euro_to_usd football_cost_eur)
            (cost_in_usd_shorts gbp_to_usd shorts_cost_gbp pairs_shorts)
            (cost_in_usd_shoes shoes_cost_usd)
            (cost_in_usd_socks jpy_to_usd socks_cost_jpy pairs_socks)
            (cost_in_usd_water_bottle krw_to_usd water_bottle_cost_krw)) 
          discount) 
      current_money_usd = 7.127214 := 
sorry

end zachary_needs_more_money_l49_4956


namespace no_real_solutions_iff_k_gt_4_l49_4937

theorem no_real_solutions_iff_k_gt_4 (k : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x + k ≠ 0) ↔ k > 4 :=
sorry

end no_real_solutions_iff_k_gt_4_l49_4937


namespace find_four_digit_number_l49_4972

theorem find_four_digit_number : ∃ N : ℕ, 999 < N ∧ N < 10000 ∧ (∃ a : ℕ, a^2 = N) ∧ 
  (∃ b : ℕ, b^3 = N % 1000) ∧ (∃ c : ℕ, c^4 = N % 100) ∧ N = 9216 := 
by
  sorry

end find_four_digit_number_l49_4972


namespace sweet_cookies_more_than_salty_l49_4936

-- Definitions for the given conditions
def sweet_cookies_ate : Nat := 32
def salty_cookies_ate : Nat := 23

-- The statement to prove
theorem sweet_cookies_more_than_salty :
  sweet_cookies_ate - salty_cookies_ate = 9 := by
  sorry

end sweet_cookies_more_than_salty_l49_4936


namespace problem_statement_l49_4961

-- Define the repeating decimal and the required gcd condition
def repeating_decimal_value := (356 : ℚ) / 999
def gcd_condition (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the main theorem stating the required sum
theorem problem_statement (a b : ℕ) 
                          (h_a : a = 356) 
                          (h_b : b = 999) 
                          (h_gcd : gcd_condition a b) : 
    a + b = 1355 := by
  sorry

end problem_statement_l49_4961


namespace fraction_of_surface_area_is_red_l49_4924

structure Cube :=
  (edge_length : ℕ)
  (small_cubes : ℕ)
  (num_red_cubes : ℕ)
  (num_blue_cubes : ℕ)
  (blue_cube_edge_length : ℕ)
  (red_outer_layer : ℕ)

def surface_area (c : Cube) : ℕ := 6 * (c.edge_length * c.edge_length)

theorem fraction_of_surface_area_is_red (c : Cube) 
  (h_edge_length : c.edge_length = 4)
  (h_small_cubes : c.small_cubes = 64)
  (h_num_red_cubes : c.num_red_cubes = 40)
  (h_num_blue_cubes : c.num_blue_cubes = 24)
  (h_blue_cube_edge_length : c.blue_cube_edge_length = 2)
  (h_red_outer_layer : c.red_outer_layer = 1)
  : (surface_area c) / (surface_area c) = 1 := 
by
  sorry

end fraction_of_surface_area_is_red_l49_4924


namespace average_sleep_per_day_l49_4930

-- Define a structure for time duration
structure TimeDuration where
  hours : ℕ
  minutes : ℕ

-- Define instances for each day
def mondayNight : TimeDuration := ⟨8, 15⟩
def mondayNap : TimeDuration := ⟨0, 30⟩
def tuesdayNight : TimeDuration := ⟨7, 45⟩
def tuesdayNap : TimeDuration := ⟨0, 45⟩
def wednesdayNight : TimeDuration := ⟨8, 10⟩
def wednesdayNap : TimeDuration := ⟨0, 50⟩
def thursdayNight : TimeDuration := ⟨10, 25⟩
def thursdayNap : TimeDuration := ⟨0, 20⟩
def fridayNight : TimeDuration := ⟨7, 50⟩
def fridayNap : TimeDuration := ⟨0, 40⟩

-- Function to convert TimeDuration to total minutes
def totalMinutes (td : TimeDuration) : ℕ :=
  td.hours * 60 + td.minutes

-- Define the total sleep time for each day
def mondayTotal := totalMinutes mondayNight + totalMinutes mondayNap
def tuesdayTotal := totalMinutes tuesdayNight + totalMinutes tuesdayNap
def wednesdayTotal := totalMinutes wednesdayNight + totalMinutes wednesdayNap
def thursdayTotal := totalMinutes thursdayNight + totalMinutes thursdayNap
def fridayTotal := totalMinutes fridayNight + totalMinutes fridayNap

-- Sum of all sleep times
def totalSleep := mondayTotal + tuesdayTotal + wednesdayTotal + thursdayTotal + fridayTotal
-- Average sleep in minutes per day
def averageSleep := totalSleep / 5
-- Convert average sleep in total minutes back to hours and minutes
def averageHours := averageSleep / 60
def averageMinutes := averageSleep % 60

theorem average_sleep_per_day :
  averageHours = 9 ∧ averageMinutes = 6 := by
  sorry

end average_sleep_per_day_l49_4930


namespace exists_N_minimal_l49_4920

-- Assuming m and n are positive and coprime
variables (m n : ℕ)
variables (h_pos_m : 0 < m) (h_pos_n : 0 < n)
variables (h_coprime : Nat.gcd m n = 1)

-- Statement of the mathematical problem
theorem exists_N_minimal :
  ∃ N : ℕ, (∀ k : ℕ, k ≥ N → ∃ a b : ℕ, k = a * m + b * n) ∧
           (N = m * n - m - n + 1) := 
  sorry

end exists_N_minimal_l49_4920


namespace Tanika_total_boxes_sold_l49_4949

theorem Tanika_total_boxes_sold:
  let friday_boxes := 60
  let saturday_boxes := friday_boxes + 0.5 * friday_boxes
  let sunday_boxes := saturday_boxes - 0.3 * saturday_boxes
  friday_boxes + saturday_boxes + sunday_boxes = 213 :=
by
  sorry

end Tanika_total_boxes_sold_l49_4949


namespace total_molecular_weight_is_1317_12_l49_4981

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

def molecular_weight_Al2S3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_S)
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + (1 * atomic_weight_O)
def molecular_weight_CO2 : ℝ := (1 * atomic_weight_C) + (2 * atomic_weight_O)

def total_weight_7_Al2S3 : ℝ := 7 * molecular_weight_Al2S3
def total_weight_5_H2O : ℝ := 5 * molecular_weight_H2O
def total_weight_4_CO2 : ℝ := 4 * molecular_weight_CO2

def total_molecular_weight : ℝ := total_weight_7_Al2S3 + total_weight_5_H2O + total_weight_4_CO2

theorem total_molecular_weight_is_1317_12 : total_molecular_weight = 1317.12 := by
  sorry

end total_molecular_weight_is_1317_12_l49_4981


namespace emily_pen_selections_is_3150_l49_4945

open Function

noncomputable def emily_pen_selections : ℕ :=
  (Nat.choose 10 4) * (Nat.choose 6 2)

theorem emily_pen_selections_is_3150 : emily_pen_selections = 3150 :=
by
  sorry

end emily_pen_selections_is_3150_l49_4945


namespace gasoline_needed_l49_4942

variable (distance_trip : ℕ) (fuel_per_trip_distance : ℕ) (trip_distance : ℕ) (fuel_needed : ℕ)

theorem gasoline_needed (h1 : distance_trip = 140)
                       (h2 : fuel_per_trip_distance = 10)
                       (h3 : trip_distance = 70)
                       (h4 : fuel_needed = 20) :
  (fuel_per_trip_distance * (distance_trip / trip_distance)) = fuel_needed :=
by sorry

end gasoline_needed_l49_4942


namespace average_goal_l49_4979

-- Define the list of initial rolls
def initial_rolls : List ℕ := [1, 3, 2, 4, 3, 5, 3, 4, 4, 2]

-- Define the next roll
def next_roll : ℕ := 2

-- Define the goal for the average
def goal_average : ℕ := 3

-- The theorem to prove that Ronald's goal for the average of all his rolls is 3
theorem average_goal : (List.sum (initial_rolls ++ [next_roll]) / (List.length (initial_rolls ++ [next_roll]))) = goal_average :=
by
  -- The proof will be provided later
  sorry

end average_goal_l49_4979


namespace four_digit_number_count_l49_4962

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_4962


namespace vector_dot_product_l49_4939

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the operation to calculate (a + 2b)
def two_b : ℝ × ℝ := (2 * b.1, 2 * b.2)
def a_plus_2b : ℝ × ℝ := (a.1 + two_b.1, a.2 + two_b.2)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- State the theorem
theorem vector_dot_product : dot_product a_plus_2b b = 14 := by
  sorry

end vector_dot_product_l49_4939


namespace total_pieces_of_bread_correct_l49_4913

-- Define the constants for the number of bread pieces needed per type of sandwich
def pieces_per_regular_sandwich : ℕ := 2
def pieces_per_double_meat_sandwich : ℕ := 3

-- Define the quantities of each type of sandwich
def regular_sandwiches : ℕ := 14
def double_meat_sandwiches : ℕ := 12

-- Define the total pieces of bread calculation
def total_pieces_of_bread : ℕ := pieces_per_regular_sandwich * regular_sandwiches + pieces_per_double_meat_sandwich * double_meat_sandwiches

-- State the theorem
theorem total_pieces_of_bread_correct : total_pieces_of_bread = 64 :=
by
  -- Proof goes here (using sorry for now)
  sorry

end total_pieces_of_bread_correct_l49_4913


namespace average_homework_time_decrease_l49_4999

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end average_homework_time_decrease_l49_4999


namespace buckets_needed_l49_4921

variable {C : ℝ} (hC : C > 0)

theorem buckets_needed (h : 42 * C = 42 * C) : 
  (42 * C) / ((2 / 5) * C) = 105 :=
by
  sorry

end buckets_needed_l49_4921


namespace f_neg1_l49_4957

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom symmetry_about_x2 : ∀ x : ℝ, f (2 + x) = f (2 - x)
axiom f3_value : f 3 = 3

theorem f_neg1 : f (-1) = 3 := by
  sorry

end f_neg1_l49_4957


namespace number_of_terms_in_expanded_polynomial_l49_4908

theorem number_of_terms_in_expanded_polynomial : 
  ∀ (a : Fin 4 → Type) (b : Fin 2 → Type) (c : Fin 3 → Type), 
  (4 * 2 * 3 = 24) := 
by
  intros a b c
  sorry

end number_of_terms_in_expanded_polynomial_l49_4908


namespace probability_at_least_one_tree_survives_l49_4952

noncomputable def prob_at_least_one_survives (survival_rate_A survival_rate_B : ℚ) (n_A n_B : ℕ) : ℚ :=
  1 - ((1 - survival_rate_A)^(n_A) * (1 - survival_rate_B)^(n_B))

theorem probability_at_least_one_tree_survives :
  prob_at_least_one_survives (5/6) (4/5) 2 2 = 899 / 900 :=
by
  sorry

end probability_at_least_one_tree_survives_l49_4952


namespace simplify_expression_l49_4989

theorem simplify_expression : 
  (1 / (1 / (1 / 3) ^ 1 + 1 / (1 / 3) ^ 2 + 1 / (1 / 3) ^ 3 + 1 / (1 / 3) ^ 4)) = 1 / 120 :=
by
  sorry

end simplify_expression_l49_4989


namespace conic_section_eccentricity_l49_4903

theorem conic_section_eccentricity (m : ℝ) (h : 2 * 8 = m^2) :
    (∃ e : ℝ, ((e = (Real.sqrt 2) / 2) ∨ (e = Real.sqrt 3))) :=
by
  sorry

end conic_section_eccentricity_l49_4903


namespace proof_valid_x_values_l49_4984

noncomputable def valid_x_values (x : ℝ) : Prop :=
  (x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3) ≤ 1

theorem proof_valid_x_values :
  {x : ℝ | valid_x_values x} = {x : ℝ | (x < -1) ∨ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1)} :=
by {
  sorry
}

end proof_valid_x_values_l49_4984


namespace common_noninteger_root_eq_coeffs_l49_4976

theorem common_noninteger_root_eq_coeffs (p1 p2 q1 q2 : ℤ) (α : ℝ) :
  (α^2 + (p1: ℝ) * α + (q1: ℝ) = 0) ∧ (α^2 + (p2: ℝ) * α + (q2: ℝ) = 0) ∧ ¬(∃ (k : ℤ), α = k) → p1 = p2 ∧ q1 = q2 :=
by {
  sorry
}

end common_noninteger_root_eq_coeffs_l49_4976


namespace math_proof_l49_4958

noncomputable def f (ω x : ℝ) := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem math_proof (h1 : ∀ x, f ω x = f ω (x + π)) (h2 : 0 < ω) :
  (ω = 2) ∧ (f 2 (-5 * Real.pi / 6) = 0) ∧ ¬∀ x : ℝ, x ∈ Set.Ioo (Real.pi / 3) (11 * Real.pi / 12) → 
  (∃ x₁ x₂ : ℝ, f 2 x₁ < f 2 x₂) ∧ (∀ x : ℝ, f 2 (x - Real.pi / 3) ≠ Real.cos (2 * x - Real.pi / 6)) := 
by
  sorry

end math_proof_l49_4958


namespace measure_α_l49_4975

noncomputable def measure_α_proof (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : ℝ :=
  let α := 120
  α

theorem measure_α (AB BC : ℝ) (h1: AB = 1) (h2 : BC = 2) : measure_α_proof AB BC h1 h2 = 120 :=
  sorry

end measure_α_l49_4975


namespace probability_one_die_shows_4_given_sum_7_l49_4925

def outcomes_with_sum_7 : List (ℕ × ℕ) := [(1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1)]

def outcome_has_4 (outcome : ℕ × ℕ) : Bool :=
  outcome.fst = 4 ∨ outcome.snd = 4

def favorable_outcomes : List (ℕ × ℕ) :=
  outcomes_with_sum_7.filter outcome_has_4

theorem probability_one_die_shows_4_given_sum_7 :
  (favorable_outcomes.length : ℚ) / (outcomes_with_sum_7.length : ℚ) = 1 / 3 := sorry

end probability_one_die_shows_4_given_sum_7_l49_4925


namespace alice_more_than_half_sum_l49_4998

-- Conditions
def row_of_fifty_coins (denominations : List ℤ) : Prop :=
  denominations.length = 50 ∧ (List.sum denominations) % 2 = 1

def alice_starts (denominations : List ℤ) : Prop := True
def bob_follows (denominations : List ℤ) : Prop := True
def alternating_selection (denominations : List ℤ) : Prop := True

-- Question/Proof Goal
theorem alice_more_than_half_sum (denominations : List ℤ) 
  (h1 : row_of_fifty_coins denominations)
  (h2 : alice_starts denominations)
  (h3 : bob_follows denominations)
  (h4 : alternating_selection denominations) :
  ∃ s_A : ℤ, s_A > (List.sum denominations) / 2 ∧ s_A ≤ List.sum denominations :=
sorry

end alice_more_than_half_sum_l49_4998


namespace sandy_change_correct_l49_4909

def football_cost : ℚ := 914 / 100
def baseball_cost : ℚ := 681 / 100
def payment : ℚ := 20

def total_cost : ℚ := football_cost + baseball_cost
def change_received : ℚ := payment - total_cost

theorem sandy_change_correct :
  change_received = 405 / 100 :=
by
  -- The proof should go here
  sorry

end sandy_change_correct_l49_4909


namespace arithmetic_sequence_common_difference_l49_4954

theorem arithmetic_sequence_common_difference
  (a : ℤ)
  (a_n : ℤ)
  (S_n : ℤ)
  (n : ℤ)
  (d : ℚ)
  (h1 : a = 3)
  (h2 : a_n = 34)
  (h3 : S_n = 222)
  (h4 : S_n = n * (a + a_n) / 2)
  (h5 : a_n = a + (n - 1) * d) :
  d = 31 / 11 :=
by
  sorry

end arithmetic_sequence_common_difference_l49_4954


namespace max_b_value_l49_4987

theorem max_b_value
  (a b c : ℕ)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = 240) : b = 10 :=
  sorry

end max_b_value_l49_4987


namespace rate_of_interest_is_20_l49_4948

-- Definitions of the given conditions
def principal := 400
def simple_interest := 160
def time := 2

-- Definition of the rate of interest based on the given formula
def rate_of_interest (P SI T : ℕ) : ℕ := (SI * 100) / (P * T)

-- Theorem stating that the rate of interest is 20% given the conditions
theorem rate_of_interest_is_20 :
  rate_of_interest principal simple_interest time = 20 := by
  sorry

end rate_of_interest_is_20_l49_4948


namespace pascals_triangle_ratio_456_l49_4955

theorem pascals_triangle_ratio_456 (n : ℕ) :
  (∃ r : ℕ,
    (n.choose r * 5 = (n.choose (r + 1)) * 4) ∧
    ((n.choose (r + 1)) * 6 = (n.choose (r + 2)) * 5)) →
  n = 98 :=
sorry

end pascals_triangle_ratio_456_l49_4955


namespace intersection_A_B_l49_4934

def setA : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, x^2)}
def setB : Set (ℝ × ℝ) := {p | ∃ (x: ℝ), p = (x, Real.sqrt x)}

theorem intersection_A_B :
  (setA ∩ setB) = {(0, 0), (1, 1)} := by
  sorry

end intersection_A_B_l49_4934


namespace kitten_length_after_4_months_l49_4983

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end kitten_length_after_4_months_l49_4983


namespace same_sum_sufficient_days_l49_4960

variable {S Wb Wc : ℝ}
variable (h1 : S = 12 * Wb)
variable (h2 : S = 24 * Wc)

theorem same_sum_sufficient_days : ∃ D : ℝ, D = 8 ∧ S = D * (Wb + Wc) :=
by
  use 8
  sorry

end same_sum_sufficient_days_l49_4960


namespace range_of_a_l49_4963

   noncomputable section

   variable {f : ℝ → ℝ}

   /-- The requried theorem based on the given conditions and the correct answer -/
   theorem range_of_a (even_f : ∀ x, f (-x) = f x)
                      (increasing_f : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y)
                      (h : f a ≤ f 2) : a ≤ -2 ∨ a ≥ 2 :=
   sorry
   
end range_of_a_l49_4963


namespace triangle_height_relationship_l49_4953

theorem triangle_height_relationship
  (b : ℝ) (h1 h2 h3 : ℝ)
  (area1 area2 area3 : ℝ)
  (h_equal_angle : area1 / area2 = 16 / 25)
  (h_diff_angle : area1 / area3 = 4 / 9) :
  4 * h2 = 5 * h1 ∧ 6 * h2 = 5 * h3 := by
    sorry

end triangle_height_relationship_l49_4953


namespace solution_to_system_of_equations_l49_4916

def augmented_matrix_system_solution (x y : ℝ) : Prop :=
  (x + 3 * y = 5) ∧ (2 * x + 4 * y = 6)

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), augmented_matrix_system_solution x y ∧ x = -1 ∧ y = 2 :=
by {
  sorry
}

end solution_to_system_of_equations_l49_4916


namespace count_students_neither_math_physics_chemistry_l49_4905

def total_students := 150

def students_math := 90
def students_physics := 70
def students_chemistry := 40

def students_math_and_physics := 20
def students_math_and_chemistry := 15
def students_physics_and_chemistry := 10
def students_all_three := 5

theorem count_students_neither_math_physics_chemistry :
  (total_students - 
   (students_math + students_physics + students_chemistry - 
    students_math_and_physics - students_math_and_chemistry - 
    students_physics_and_chemistry + students_all_three)) = 5 := by
  sorry

end count_students_neither_math_physics_chemistry_l49_4905


namespace petunia_fertilizer_problem_l49_4906

theorem petunia_fertilizer_problem
  (P : ℕ)
  (h1 : 4 * P * 8 + 3 * 6 * 3 + 2 * 2 = 314) :
  P = 8 :=
by
  sorry

end petunia_fertilizer_problem_l49_4906


namespace value_of_M_l49_4929

theorem value_of_M
  (M : ℝ)
  (h : 25 / 100 * M = 55 / 100 * 4500) :
  M = 9900 :=
sorry

end value_of_M_l49_4929


namespace faith_overtime_hours_per_day_l49_4978

noncomputable def normal_pay_per_hour : ℝ := 13.50
noncomputable def normal_daily_hours : ℕ := 8
noncomputable def normal_weekly_days : ℕ := 5
noncomputable def total_weekly_earnings : ℝ := 675
noncomputable def overtime_rate_multiplier : ℝ := 1.5

noncomputable def normal_weekly_hours := normal_daily_hours * normal_weekly_days
noncomputable def normal_weekly_earnings := normal_pay_per_hour * normal_weekly_hours
noncomputable def overtime_earnings := total_weekly_earnings - normal_weekly_earnings
noncomputable def overtime_pay_per_hour := normal_pay_per_hour * overtime_rate_multiplier
noncomputable def total_overtime_hours := overtime_earnings / overtime_pay_per_hour
noncomputable def overtime_hours_per_day := total_overtime_hours / normal_weekly_days

theorem faith_overtime_hours_per_day :
  overtime_hours_per_day = 1.33 := 
by 
  sorry

end faith_overtime_hours_per_day_l49_4978


namespace no_nat_nums_satisfy_gcd_lcm_condition_l49_4964

theorem no_nat_nums_satisfy_gcd_lcm_condition :
  ¬ ∃ (x y : ℕ), Nat.gcd x y + Nat.lcm x y = x + y + 2021 := 
sorry

end no_nat_nums_satisfy_gcd_lcm_condition_l49_4964


namespace problem_l49_4946

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}
def complement_U (S : Set ℕ) : Set ℕ := U \ S

theorem problem : ((complement_U M) ∩ (complement_U N)) = {5} :=
by
  sorry

end problem_l49_4946


namespace sum_of_fifth_powers_divisibility_l49_4967

theorem sum_of_fifth_powers_divisibility (a b c d e : ℤ) :
  (a^5 + b^5 + c^5 + d^5 + e^5) % 25 = 0 → (a % 5 = 0) ∨ (b % 5 = 0) ∨ (c % 5 = 0) ∨ (d % 5 = 0) ∨ (e % 5 = 0) :=
by
  sorry

end sum_of_fifth_powers_divisibility_l49_4967


namespace expected_value_of_biased_die_l49_4971

-- Definitions for probabilities
def prob1 : ℚ := 1 / 15
def prob2 : ℚ := 1 / 15
def prob3 : ℚ := 1 / 15
def prob4 : ℚ := 1 / 15
def prob5 : ℚ := 1 / 5
def prob6 : ℚ := 3 / 5

-- Definition for expected value
def expected_value : ℚ := (prob1 * 1) + (prob2 * 2) + (prob3 * 3) + (prob4 * 4) + (prob5 * 5) + (prob6 * 6)

theorem expected_value_of_biased_die : expected_value = 16 / 3 :=
by sorry

end expected_value_of_biased_die_l49_4971


namespace find_c_l49_4992

theorem find_c (c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 6) → c = 16 :=
by
  intros h
  sorry

end find_c_l49_4992


namespace fixed_point_of_exponential_function_l49_4914

-- The function definition and conditions are given as hypotheses
theorem fixed_point_of_exponential_function
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ P : ℝ × ℝ, (∀ x : ℝ, (x = 1) → P = (x, a^(x-1) - 2)) → P = (1, -1) :=
by
  sorry

end fixed_point_of_exponential_function_l49_4914


namespace find_nabla_l49_4915

theorem find_nabla : ∀ (nabla : ℤ), 5 * (-4) = nabla + 2 → nabla = -22 :=
by
  intros nabla h
  sorry

end find_nabla_l49_4915


namespace smaller_of_x_and_y_l49_4922

theorem smaller_of_x_and_y 
  (x y a b c d : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b + 1) 
  (h3 : x + y = c) 
  (h4 : x - y = d) 
  (h5 : x / y = a / (b + 1)) :
  min x y = (ac/(a + b + 1)) := 
by
  sorry

end smaller_of_x_and_y_l49_4922


namespace imaginary_part_of_l49_4947

theorem imaginary_part_of (i : ℂ) (h : i.im = 1) : (1 + i) ^ 5 = -14 - 4 * i := by
  sorry

end imaginary_part_of_l49_4947


namespace rectangle_length_width_difference_l49_4901

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : x + y = 40)
  (h2 : x^2 + y^2 = 800) :
  x - y = 0 :=
sorry

end rectangle_length_width_difference_l49_4901


namespace polynomial_no_positive_real_roots_l49_4997

theorem polynomial_no_positive_real_roots : 
  ¬ ∃ x : ℝ, x > 0 ∧ x^3 + 6 * x^2 + 11 * x + 6 = 0 :=
sorry

end polynomial_no_positive_real_roots_l49_4997


namespace shift_sine_graph_l49_4910

theorem shift_sine_graph (x : ℝ) : 
  (∃ θ : ℝ, θ = (5 * Real.pi) / 4 ∧ 
  y = Real.sin (x - Real.pi / 4) → y = Real.sin (x + θ) 
  ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := sorry

end shift_sine_graph_l49_4910


namespace sum_of_squares_and_cubes_l49_4923

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  ∃ x1 x2 : ℤ, a^2 - 2*b = x1^2 + x2^2 ∧ 3*a*b - a^3 = x1^3 + x2^3 :=
by
  sorry

end sum_of_squares_and_cubes_l49_4923


namespace percentage_decrease_is_20_l49_4900

-- Define the original and new prices in Rs.
def original_price : ℕ := 775
def new_price : ℕ := 620

-- Define the decrease in price
def decrease_in_price : ℕ := original_price - new_price

-- Define the formula to calculate the percentage decrease
def percentage_decrease (orig_price new_price : ℕ) : ℕ :=
  (decrease_in_price * 100) / orig_price

-- Prove that the percentage decrease is 20%
theorem percentage_decrease_is_20 :
  percentage_decrease original_price new_price = 20 :=
by
  sorry

end percentage_decrease_is_20_l49_4900


namespace quadratic_real_solutions_l49_4918

theorem quadratic_real_solutions (m : ℝ) :
  (∃ x : ℝ, (m - 3) * x^2 + 4 * x + 1 = 0) ↔ (m ≤ 7 ∧ m ≠ 3) :=
by
  sorry

end quadratic_real_solutions_l49_4918


namespace value_of_expression_l49_4931

theorem value_of_expression (m : ℝ) 
  (h : m^2 - 2 * m - 1 = 0) : 3 * m^2 - 6 * m + 2020 = 2023 := 
by 
  /- Proof is omitted -/
  sorry

end value_of_expression_l49_4931


namespace percentage_of_invalid_votes_l49_4982

-- Candidate A got 60% of the total valid votes.
-- The total number of votes is 560000.
-- The number of valid votes polled in favor of candidate A is 285600.
variable (total_votes valid_votes_A : ℝ)
variable (percent_A : ℝ := 0.60)
variable (valid_votes total_invalid_votes percent_invalid_votes : ℝ)

axiom h1 : total_votes = 560000
axiom h2 : valid_votes_A = 285600
axiom h3 : valid_votes_A = percent_A * valid_votes
axiom h4 : total_invalid_votes = total_votes - valid_votes
axiom h5 : percent_invalid_votes = (total_invalid_votes / total_votes) * 100

theorem percentage_of_invalid_votes : percent_invalid_votes = 15 := by
  sorry

end percentage_of_invalid_votes_l49_4982


namespace acute_triangle_side_range_l49_4911

theorem acute_triangle_side_range {x : ℝ} (h : ∀ a b c : ℝ, a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) :
  2 < 4 ∧ 4 < x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
  sorry

end acute_triangle_side_range_l49_4911


namespace inradius_circumradius_inequality_l49_4917

variable {R r a b c : ℝ}

def inradius (ABC : Triangle) := r
def circumradius (ABC : Triangle) := R
def side_a (ABC : Triangle) := a
def side_b (ABC : Triangle) := b
def side_c (ABC : Triangle) := c

theorem inradius_circumradius_inequality (ABC : Triangle) :
  R / (2 * r) ≥ (64 * a^2 * b^2 * c^2 / ((4 * a^2 - (b - c)^2) * (4 * b^2 - (c - a)^2) * (4 * c^2 - (a - b)^2)))^2 :=
sorry

end inradius_circumradius_inequality_l49_4917


namespace max_daily_sales_revenue_l49_4902

noncomputable def P (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 24 then t + 2 else if 25 ≤ t ∧ t ≤ 30 then 100 - t else 0

noncomputable def Q (t : ℕ) : ℕ :=
  if 1 ≤ t ∧ t ≤ 30 then 40 - t else 0

noncomputable def y (t : ℕ) : ℕ :=
  P t * Q t

theorem max_daily_sales_revenue :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ y t = 115 :=
sorry

end max_daily_sales_revenue_l49_4902


namespace regression_decrease_by_three_l49_4970

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 3 * x

-- Prove that when the explanatory variable increases by 1 unit, the predicted variable decreases by 3 units
theorem regression_decrease_by_three : ∀ x : ℝ, regression_equation (x + 1) = regression_equation x - 3 :=
by
  intro x
  unfold regression_equation
  sorry

end regression_decrease_by_three_l49_4970


namespace identity_of_polynomials_l49_4933

theorem identity_of_polynomials (a b : ℝ) : 
  (2 * x + a)^3 = 
  5 * x^3 + (3 * x + b) * (x^2 - x - 1) - 10 * x^2 + 10 * x 
  → a = -1 ∧ b = 1 := 
by 
  sorry

end identity_of_polynomials_l49_4933


namespace factorization_identity_l49_4938

-- We are asked to prove the mathematical equality under given conditions.
theorem factorization_identity (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = (x^2 + 6*x + 8)^2 :=
by
  sorry

end factorization_identity_l49_4938


namespace union_of_A_and_B_intersection_of_complement_A_and_B_l49_4994

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2 * x - 1 ∧ 2 * x - 1 < 19}

-- Define the universal set here, which encompass all real numbers
def universal_set : Set ℝ := {x | true}

-- Define the complement of A with respect to the real numbers
def C_R (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- Prove that A ∪ B is {x | 2 < x < 10}
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

-- Prove that (C_R A) ∩ B is {x | 2 < x < 3 ∨ 7 < x < 10}
theorem intersection_of_complement_A_and_B : (C_R A) ∪ B = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by
  sorry

end union_of_A_and_B_intersection_of_complement_A_and_B_l49_4994


namespace marguerites_fraction_l49_4943

variable (x r b s : ℕ)

theorem marguerites_fraction
  (h1 : r = 5 * (x - r))
  (h2 : b = (x - b) / 5)
  (h3 : r + b + s = x) : s = 0 := by sorry

end marguerites_fraction_l49_4943


namespace perp_lines_of_parallel_planes_l49_4993

variables {Line Plane : Type} 
variables (m n : Line) (α β : Plane)
variable (is_parallel : Line → Plane → Prop)
variable (is_perpendicular : Line → Plane → Prop)
variable (planes_parallel : Plane → Plane → Prop)
variable (lines_perpendicular : Line → Line → Prop)

-- Given Conditions
variables (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β)

-- Prove that
theorem perp_lines_of_parallel_planes (h1 : planes_parallel α β) (h2 : is_perpendicular m α) (h3 : is_parallel n β) : lines_perpendicular m n := 
sorry

end perp_lines_of_parallel_planes_l49_4993


namespace sam_total_pennies_l49_4951

def a : ℕ := 98
def b : ℕ := 93

theorem sam_total_pennies : a + b = 191 :=
by
  sorry

end sam_total_pennies_l49_4951


namespace det_scaled_matrix_l49_4990

variable {R : Type*} [CommRing R]

def det2x2 (a b c d : R) : R := a * d - b * c

theorem det_scaled_matrix 
  (x y z w : R) 
  (h : det2x2 x y z w = 3) : 
  det2x2 (3 * x) (3 * y) (6 * z) (6 * w) = 54 := by
  sorry

end det_scaled_matrix_l49_4990


namespace compute_star_difference_l49_4940

def star (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_star_difference : (star 6 3) - (star 3 6) = 45 := by
  sorry

end compute_star_difference_l49_4940


namespace find_integer_m_l49_4927

theorem find_integer_m 
  (m : ℤ)
  (h1 : 30 ≤ m ∧ m ≤ 80)
  (h2 : ∃ k : ℤ, m = 6 * k)
  (h3 : m % 8 = 2)
  (h4 : m % 5 = 2) : 
  m = 42 := 
sorry

end find_integer_m_l49_4927


namespace angle_is_30_degrees_l49_4950

theorem angle_is_30_degrees (A : ℝ) (h_acute : A > 0 ∧ A < π / 2) (h_sin : Real.sin A = 1/2) : A = π / 6 := 
by 
  sorry

end angle_is_30_degrees_l49_4950


namespace find_number_l49_4974

theorem find_number (x : ℝ) : 
  (72 = 0.70 * x + 30) -> x = 60 :=
by
  sorry

end find_number_l49_4974


namespace second_child_birth_year_l49_4965

theorem second_child_birth_year (first_child_birth : ℕ)
  (second_child_birth : ℕ)
  (third_child_birth : ℕ)
  (fourth_child_birth : ℕ)
  (first_child_years_ago : first_child_birth = 15)
  (third_child_on_second_child_fourth_birthday : third_child_birth = second_child_birth + 4)
  (fourth_child_two_years_after_third : fourth_child_birth = third_child_birth + 2)
  (fourth_child_age : fourth_child_birth = 8) :
  second_child_birth = first_child_birth - 14 := 
by
  sorry

end second_child_birth_year_l49_4965


namespace total_legs_l49_4926

-- Define the conditions
def chickens : Nat := 7
def sheep : Nat := 5
def legs_chicken : Nat := 2
def legs_sheep : Nat := 4

-- State the problem as a theorem
theorem total_legs :
  chickens * legs_chicken + sheep * legs_sheep = 34 :=
by
  sorry -- Proof not provided

end total_legs_l49_4926


namespace fertilizer_needed_l49_4959

def p_flats := 4
def p_per_flat := 8
def p_ounces := 8

def r_flats := 3
def r_per_flat := 6
def r_ounces := 3

def s_flats := 5
def s_per_flat := 10
def s_ounces := 6

def o_flats := 2
def o_per_flat := 4
def o_ounces := 4

def vf_quantity := 2
def vf_ounces := 2

def total_fertilizer : ℕ := 
  p_flats * p_per_flat * p_ounces +
  r_flats * r_per_flat * r_ounces +
  s_flats * s_per_flat * s_ounces +
  o_flats * o_per_flat * o_ounces +
  vf_quantity * vf_ounces

theorem fertilizer_needed : total_fertilizer = 646 := by
  -- proof goes here
  sorry

end fertilizer_needed_l49_4959


namespace items_counted_l49_4980

def convert_counter (n : Nat) : Nat := sorry

theorem items_counted
  (counter_reading : Nat) 
  (condition_1 : ∀ d, d ∈ [5, 6, 7] → ¬(d ∈ [0, 1, 2, 3, 4, 8, 9]))
  (condition_2 : ∀ d1 d2, d1 = 4 → d2 = 8 → ¬(d2 = 5 ∨ d2 = 6 ∨ d2 = 7)) :
  convert_counter 388 = 151 :=
sorry

end items_counted_l49_4980


namespace parabola_vertex_l49_4919

theorem parabola_vertex:
  ∃ x y: ℝ, y^2 + 8 * y + 2 * x + 1 = 0 ∧ (x, y) = (7.5, -4) := sorry

end parabola_vertex_l49_4919


namespace determinant_eq_sum_of_products_l49_4928

theorem determinant_eq_sum_of_products (x y z : ℝ) :
  Matrix.det (Matrix.of ![![1, x + z, y], ![1, x + y + z, y + z], ![1, x + z, x + y + z]]) = x * y + y * z + z * x :=
by
  sorry

end determinant_eq_sum_of_products_l49_4928


namespace dance_contradiction_l49_4991

variable {Boy Girl : Type}
variable {danced_with : Boy → Girl → Prop}

theorem dance_contradiction
    (H1 : ¬ ∃ g : Boy, ∀ f : Girl, danced_with g f)
    (H2 : ∀ f : Girl, ∃ g : Boy, danced_with g f) :
    ∃ (g g' : Boy) (f f' : Girl),
        danced_with g f ∧ ¬ danced_with g f' ∧
        danced_with g' f' ∧ ¬ danced_with g' f :=
by
  -- Proof will be inserted here
  sorry

end dance_contradiction_l49_4991


namespace marks_in_math_l49_4985

theorem marks_in_math (e p c b : ℕ) (avg : ℚ) (n : ℕ) (total_marks_other_subjects : ℚ) :
  e = 45 →
  p = 52 →
  c = 47 →
  b = 55 →
  avg = 46.8 →
  n = 5 →
  total_marks_other_subjects = (e + p + c + b : ℕ) →
  (avg * n) - total_marks_other_subjects = 35 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_in_math_l49_4985


namespace min_fraction_expression_l49_4995

theorem min_fraction_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 1 / b = 1) : 
  ∃ a b, ∃ (h : 1 / a + 1 / b = 1), a > 1 ∧ b > 1 ∧ (1 / (a - 1) + 4 / (b - 1)) = 4 := 
by 
  sorry

end min_fraction_expression_l49_4995


namespace fraction_of_number_l49_4966

theorem fraction_of_number (F : ℚ) (h : 0.5 * F * 120 = 36) : F = 3 / 5 :=
by
  sorry

end fraction_of_number_l49_4966


namespace mean_proportional_l49_4986

theorem mean_proportional (a b c : ℝ) (ha : a = 1) (hb : b = 2) (h : c ^ 2 = a * b) : c = Real.sqrt 2 :=
by
  sorry

end mean_proportional_l49_4986


namespace largest_divisor_same_remainder_l49_4977

theorem largest_divisor_same_remainder (n : ℕ) (h : 17 % n = 30 % n) : n = 13 :=
sorry

end largest_divisor_same_remainder_l49_4977
