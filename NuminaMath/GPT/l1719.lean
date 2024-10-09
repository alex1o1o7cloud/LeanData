import Mathlib

namespace Harry_Terry_difference_l1719_171954

theorem Harry_Terry_difference : 
(12 - (4 * 3)) - (12 - 4 * 3) = -24 := 
by
  sorry

end Harry_Terry_difference_l1719_171954


namespace nathan_paintable_area_l1719_171928

def total_paintable_area (rooms : ℕ) (length width height : ℕ) (non_paintable_area : ℕ) : ℕ :=
  let wall_area := 2 * (length * height + width * height)
  rooms * (wall_area - non_paintable_area)

theorem nathan_paintable_area :
  total_paintable_area 4 15 12 9 75 = 1644 :=
by sorry

end nathan_paintable_area_l1719_171928


namespace odd_square_minus_one_div_by_eight_l1719_171997

theorem odd_square_minus_one_div_by_eight (n : ℤ) : ∃ k : ℤ, (2 * n + 1) ^ 2 - 1 = 8 * k :=
by
  sorry

end odd_square_minus_one_div_by_eight_l1719_171997


namespace product_sum_diff_l1719_171906

variable (a b : ℝ) -- Real numbers

theorem product_sum_diff (a b : ℝ) : (a + b) * (a - b) = (a + b) * (a - b) :=
by
  sorry

end product_sum_diff_l1719_171906


namespace translated_function_symmetry_center_l1719_171990

theorem translated_function_symmetry_center :
  let f := fun x : ℝ => Real.sin (6 * x + π / 4)
  let g := fun x : ℝ => f (x / 3)
  let h := fun x : ℝ => g (x - π / 8)
  h π / 2 = 0 :=
by
  sorry

end translated_function_symmetry_center_l1719_171990


namespace student_weight_l1719_171952

theorem student_weight (S W : ℕ) (h1 : S - 5 = 2 * W) (h2 : S + W = 104) : S = 71 :=
by {
  sorry
}

end student_weight_l1719_171952


namespace diagonals_from_vertex_l1719_171964

theorem diagonals_from_vertex (n : ℕ) (h : (n-2) * 180 + 360 = 1800) : (n - 3) = 7 :=
sorry

end diagonals_from_vertex_l1719_171964


namespace product_polynomial_coeffs_l1719_171930

theorem product_polynomial_coeffs
  (g h : ℚ)
  (h1 : 7 * d^2 - 3 * d + g * (3 * d^2 + h * d - 5) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) :
  g + h = -28/9 := 
  sorry

end product_polynomial_coeffs_l1719_171930


namespace find_f_neg_a_l1719_171947

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 4) : f (-a) = 0 :=
by
  sorry

end find_f_neg_a_l1719_171947


namespace max_m_value_l1719_171933

noncomputable def f (x m : ℝ) : ℝ := x * Real.log x + x^2 - m * x + Real.exp (2 - x)

theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) → m ≤ 3 :=
sorry

end max_m_value_l1719_171933


namespace area_of_PQRSUV_proof_l1719_171987

noncomputable def PQRSW_area (PQ QR RS SW : ℝ) : ℝ :=
  (1 / 2) * PQ * QR + (1 / 2) * (RS + SW) * 5

noncomputable def WUV_area (WU UV : ℝ) : ℝ :=
  WU * UV

theorem area_of_PQRSUV_proof 
  (PQ QR RS SW WU UV : ℝ)
  (hPQ : PQ = 8) (hQR : QR = 5) (hRS : RS = 7) (hSW : SW = 10)
  (hWU : WU = 6) (hUV : UV = 7) :
  PQRSW_area PQ QR RS SW + WUV_area WU UV = 147 :=
by
  simp only [PQRSW_area, WUV_area, hPQ, hQR, hRS, hSW, hWU, hUV]
  norm_num
  sorry

end area_of_PQRSUV_proof_l1719_171987


namespace compute_sqrt_eq_419_l1719_171950

theorem compute_sqrt_eq_419 : Real.sqrt ((22 * 21 * 20 * 19) + 1) = 419 :=
by
  sorry

end compute_sqrt_eq_419_l1719_171950


namespace triangle_side_AC_l1719_171992

theorem triangle_side_AC 
  (AB BC : ℝ)
  (angle_C : ℝ)
  (h1 : AB = Real.sqrt 13)
  (h2 : BC = 3)
  (h3 : angle_C = Real.pi / 3) :
  ∃ AC : ℝ, AC = 4 :=
by 
  sorry

end triangle_side_AC_l1719_171992


namespace Chris_age_l1719_171991

variable (a b c : ℕ)

theorem Chris_age : a + b + c = 36 ∧ b = 2*c + 9 ∧ b = a → c = 4 :=
by
  sorry

end Chris_age_l1719_171991


namespace potatoes_fraction_l1719_171900

theorem potatoes_fraction (w : ℝ) (x : ℝ) (h_weight : w = 36) (h_fraction : w / x = 36) : x = 1 :=
by
  sorry

end potatoes_fraction_l1719_171900


namespace pyramid_volume_theorem_l1719_171905

noncomputable def volume_of_regular_square_pyramid : ℝ := 
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * side_edge_length * Real.sqrt 3
  (1 / 3) * base_area * height

theorem pyramid_volume_theorem :
  let side_edge_length := 2 * Real.sqrt 3
  let angle := Real.pi / 3 -- 60 degrees in radians
  let height := side_edge_length * Real.sin angle
  let base_area := 2 * (1 / 2) * (side_edge_length * Real.sqrt 3)
  (1 / 3) * base_area * height = 6 := 
by
  sorry

end pyramid_volume_theorem_l1719_171905


namespace speed_in_still_water_l1719_171931

-- Define the velocities (speeds)
def speed_downstream (V_w V_s : ℝ) : ℝ := V_w + V_s
def speed_upstream (V_w V_s : ℝ) : ℝ := V_w - V_s

-- Define the given conditions
def downstream_condition (V_w V_s : ℝ) : Prop := speed_downstream V_w V_s = 9
def upstream_condition (V_w V_s : ℝ) : Prop := speed_upstream V_w V_s = 1

-- The main theorem to prove
theorem speed_in_still_water (V_s V_w : ℝ) (h1 : downstream_condition V_w V_s) (h2 : upstream_condition V_w V_s) : V_w = 5 :=
  sorry

end speed_in_still_water_l1719_171931


namespace chess_group_players_l1719_171940

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end chess_group_players_l1719_171940


namespace find_y_l1719_171998

theorem find_y (y : ℝ) (h : 2 * y / 3 = 30) : y = 45 :=
by
  sorry

end find_y_l1719_171998


namespace containers_per_truck_l1719_171910

theorem containers_per_truck (trucks1 boxes1 trucks2 boxes2 boxes_to_containers total_trucks : ℕ)
  (h1 : trucks1 = 7) 
  (h2 : boxes1 = 20) 
  (h3 : trucks2 = 5) 
  (h4 : boxes2 = 12) 
  (h5 : boxes_to_containers = 8) 
  (h6 : total_trucks = 10) :
  (((trucks1 * boxes1) + (trucks2 * boxes2)) * boxes_to_containers) / total_trucks = 160 := 
sorry

end containers_per_truck_l1719_171910


namespace last_four_digits_of_5_pow_2016_l1719_171963

theorem last_four_digits_of_5_pow_2016 :
  (5^2016) % 10000 = 625 :=
by
  -- Establish periodicity of last four digits in powers of 5
  sorry

end last_four_digits_of_5_pow_2016_l1719_171963


namespace nicolai_peaches_6_pounds_l1719_171969

noncomputable def amount_peaches (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ) : ℕ :=
  let total_ounces := total_pounds * 16
  let total_consumed := oz_oranges + oz_apples
  let remaining_ounces := total_ounces - total_consumed
  remaining_ounces / 16

theorem nicolai_peaches_6_pounds (total_pounds : ℕ) (oz_oranges : ℕ) (oz_apples : ℕ)
  (h_total_pounds : total_pounds = 8) (h_oz_oranges : oz_oranges = 8) (h_oz_apples : oz_apples = 24) :
  amount_peaches total_pounds oz_oranges oz_apples = 6 :=
by
  rw [h_total_pounds, h_oz_oranges, h_oz_apples]
  unfold amount_peaches
  sorry

end nicolai_peaches_6_pounds_l1719_171969


namespace min_value_frac_l1719_171927

open Real

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 1) :
  (1 / a + 2 / b) = 9 + 4 * sqrt 2 :=
sorry

end min_value_frac_l1719_171927


namespace beach_weather_condition_l1719_171911

theorem beach_weather_condition
  (T : ℝ) -- Temperature in degrees Fahrenheit
  (sunny : Prop) -- Whether it is sunny
  (crowded : Prop) -- Whether the beach is crowded
  (H1 : ∀ (T : ℝ) (sunny : Prop), (T ≥ 80) ∧ sunny → crowded) -- Condition 1
  (H2 : ¬ crowded) -- Condition 2
  : T < 80 ∨ ¬ sunny := sorry

end beach_weather_condition_l1719_171911


namespace polygon_area_l1719_171914

theorem polygon_area (n : ℕ) (s : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : n = 24) 
  (h2 : n * s = perimeter) 
  (h3 : perimeter = 48) 
  (h4 : s = 2) 
  (h5 : area = n * s^2 / 2) : 
  area = 96 :=
by
  sorry

end polygon_area_l1719_171914


namespace total_weekly_reading_time_l1719_171966

def morning_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def morning_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

def evening_reading_weekdays (daily_minutes : ℕ) (days : ℕ) : ℕ :=
  daily_minutes * days

def evening_reading_weekends (daily_minutes : ℕ) : ℕ :=
  2 * daily_minutes * 2

theorem total_weekly_reading_time :
  let morning_minutes := 30
  let evening_minutes := 60
  let weekdays := 5
  let weekend_days := 2
  morning_reading_weekdays morning_minutes weekdays +
  morning_reading_weekends morning_minutes +
  evening_reading_weekdays evening_minutes weekdays +
  evening_reading_weekends evening_minutes = 810 :=
by
  sorry

end total_weekly_reading_time_l1719_171966


namespace scooterValue_after_4_years_with_maintenance_l1719_171967

noncomputable def scooterDepreciation (initial_value : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((3 : ℝ) / 4) ^ years

theorem scooterValue_after_4_years_with_maintenance (M : ℝ) :
  scooterDepreciation 40000 4 - 4 * M = 12656.25 - 4 * M :=
by
  sorry

end scooterValue_after_4_years_with_maintenance_l1719_171967


namespace contrapositive_l1719_171956

theorem contrapositive (p q : Prop) (h : p → q) : ¬q → ¬p :=
by
  sorry

end contrapositive_l1719_171956


namespace no_real_solutions_l1719_171994

theorem no_real_solutions (x : ℝ) : ¬ (3 * x^2 + 5 = |4 * x + 2| - 3) :=
by
  sorry

end no_real_solutions_l1719_171994


namespace area_inequalities_l1719_171920

noncomputable def f1 (x : ℝ) : ℝ := 1 - (1 / 2) * x
noncomputable def f2 (x : ℝ) : ℝ := 1 / (x + 1)
noncomputable def f3 (x : ℝ) : ℝ := 1 - (1 / 2) * x^2

noncomputable def S1 : ℝ := 1 - (1 / 4)
noncomputable def S2 : ℝ := Real.log 2
noncomputable def S3 : ℝ := (5 / 6)

theorem area_inequalities : S2 < S1 ∧ S1 < S3 := by
  sorry

end area_inequalities_l1719_171920


namespace max_value_k_l1719_171926

noncomputable def max_k (S : Finset ℕ) (A : ℕ → Finset ℕ) (k : ℕ) :=
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2)

theorem max_value_k : ∀ (S : Finset ℕ) (A : ℕ → Finset ℕ), 
  S = Finset.range 14 \{0} → 
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2) →
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) →
  ∃ k, max_k S A k ∧ k = 4 :=
sorry

end max_value_k_l1719_171926


namespace sum_of_money_l1719_171953

theorem sum_of_money (x : ℝ)
  (hC : 0.50 * x = 64)
  (hB : ∀ x, B_shares = 0.75 * x)
  (hD : ∀ x, D_shares = 0.25 * x) :
  let total_sum := x + 0.75 * x + 0.50 * x + 0.25 * x
  total_sum = 320 :=
by
  sorry

end sum_of_money_l1719_171953


namespace solution_system_of_inequalities_l1719_171904

theorem solution_system_of_inequalities (x : ℝ) : 
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * (x^2) - x - 1 > 0 ↔ (-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6) :=
by {
  sorry
}

end solution_system_of_inequalities_l1719_171904


namespace kid_ticket_price_l1719_171917

theorem kid_ticket_price (adult_price kid_tickets tickets total_profit : ℕ) 
  (h_adult_price : adult_price = 6) 
  (h_kid_tickets : kid_tickets = 75) 
  (h_tickets : tickets = 175) 
  (h_total_profit : total_profit = 750) : 
  (total_profit - (tickets - kid_tickets) * adult_price) / kid_tickets = 2 :=
by
  sorry

end kid_ticket_price_l1719_171917


namespace bridge_length_l1719_171988

noncomputable def length_of_bridge (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_of_train_ms := (speed_of_train_kmh * 1000) / 3600
  let total_distance := speed_of_train_ms * time_seconds
  total_distance - length_of_train

theorem bridge_length (length_of_train : ℕ) (speed_of_train_kmh : ℕ) (time_seconds : ℕ) (h1 : length_of_train = 170) (h2 : speed_of_train_kmh = 45) (h3 : time_seconds = 30) :
  length_of_bridge length_of_train speed_of_train_kmh time_seconds = 205 :=
by 
  rw [h1, h2, h3]
  unfold length_of_bridge
  simp
  sorry

end bridge_length_l1719_171988


namespace remaining_sand_fraction_l1719_171921

theorem remaining_sand_fraction (total_weight : ℕ) (used_weight : ℕ) (h1 : total_weight = 50) (h2 : used_weight = 30) : 
  (total_weight - used_weight) / total_weight = 2 / 5 :=
by 
  sorry

end remaining_sand_fraction_l1719_171921


namespace focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l1719_171968

noncomputable def focus_of_parabola (a b c : ℝ) : ℝ × ℝ :=
  let p := b^2 / (4 * a) - c / (4 * a)
  (p, 1 / (4 * a))

theorem focus_parabola_y_eq_neg4x2_plus_4x_minus_1 :
  focus_of_parabola (-4) 4 (-1) = (1 / 2, -1 / 8) :=
sorry

end focus_parabola_y_eq_neg4x2_plus_4x_minus_1_l1719_171968


namespace completing_the_square_transformation_l1719_171979

theorem completing_the_square_transformation (x : ℝ) : 
  (x^2 - 4 * x + 1 = 0) → ((x - 2)^2 = 3) :=
by
  intro h
  -- Transformation steps will be shown in the proof
  sorry

end completing_the_square_transformation_l1719_171979


namespace range_of_m_l1719_171961

def f (m x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
def g (m x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
by
  sorry

end range_of_m_l1719_171961


namespace soccer_team_students_l1719_171999

theorem soccer_team_students :
  ∀ (n p b m : ℕ),
    n = 25 →
    p = 10 →
    b = 6 →
    n - (p - b) = m →
    m = 21 :=
by
  intros n p b m h_n h_p h_b h_trivial
  sorry

end soccer_team_students_l1719_171999


namespace no_such_fractions_l1719_171959

open Nat

theorem no_such_fractions : ¬ ∃ (x y : ℕ), (x.gcd y = 1) ∧ (x > 0) ∧ (y > 0) ∧ ((x + 1) * 5 * y = ((y + 1) * 6 * x)) :=
by
  sorry

end no_such_fractions_l1719_171959


namespace acute_triangle_angles_l1719_171984

theorem acute_triangle_angles (x y z : ℕ) (angle1 angle2 angle3 : ℕ) 
  (h1 : angle1 = 7 * x) 
  (h2 : angle2 = 9 * y) 
  (h3 : angle3 = 11 * z) 
  (h4 : angle1 + angle2 + angle3 = 180)
  (hx : 1 ≤ x ∧ x ≤ 12)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (hz : 1 ≤ z ∧ z ≤ 8)
  (ha1 : angle1 < 90)
  (ha2 : angle2 < 90)
  (ha3 : angle3 < 90)
  : angle1 = 42 ∧ angle2 = 72 ∧ angle3 = 66 
  ∨ angle1 = 49 ∧ angle2 = 54 ∧ angle3 = 77 
  ∨ angle1 = 56 ∧ angle2 = 36 ∧ angle3 = 88 
  ∨ angle1 = 84 ∧ angle2 = 63 ∧ angle3 = 33 :=
sorry

end acute_triangle_angles_l1719_171984


namespace persimmons_count_l1719_171985

variables {P T : ℕ}

-- Conditions from the problem
axiom total_eq : P + T = 129
axiom diff_eq : P = T - 43

-- Theorem to prove that there are 43 persimmons
theorem persimmons_count : P = 43 :=
by
  -- Putting the proof placeholder
  sorry

end persimmons_count_l1719_171985


namespace average_speed_of_trip_l1719_171946

theorem average_speed_of_trip 
  (total_distance : ℝ)
  (first_leg_distance : ℝ)
  (first_leg_speed : ℝ)
  (second_leg_distance : ℝ)
  (second_leg_speed : ℝ)
  (h_dist : total_distance = 50)
  (h_first_leg : first_leg_distance = 25)
  (h_second_leg : second_leg_distance = 25)
  (h_first_speed : first_leg_speed = 60)
  (h_second_speed : second_leg_speed = 30) :
  (total_distance / 
   ((first_leg_distance / first_leg_speed) + (second_leg_distance / second_leg_speed)) = 40) :=
by
  sorry

end average_speed_of_trip_l1719_171946


namespace smallest_number_l1719_171975

theorem smallest_number (A B C : ℕ) 
  (h1 : A / 3 = B / 5) 
  (h2 : B / 5 = C / 7) 
  (h3 : C = 56) 
  (h4 : C - A = 32) : 
  A = 24 := 
sorry

end smallest_number_l1719_171975


namespace converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l1719_171942

-- Definitions
variables {α : Type} [LinearOrderedField α] {a b : α}
variables {A B C : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]

-- Proof Problem for Question 1
theorem converse_angle_bigger_side (A B C : Type) [LinearOrder A] [LinearOrder B] [LinearOrder C]
  (angle_C angle_B : A) (side_AB side_AC : B) (h : angle_C > angle_B) : side_AB > side_AC :=
sorry

-- Proof Problem for Question 2
theorem negation_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

-- Proof Problem for Question 3
theorem contrapositive_ab_zero (a b : α) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end converse_angle_bigger_side_negation_ab_zero_contrapositive_ab_zero_l1719_171942


namespace intersection_M_N_l1719_171912

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def N : Set ℝ := { y | 0 < y }

theorem intersection_M_N : (M ∩ N) = { z | 0 < z ∧ z ≤ 2 } :=
by
  -- proof to be completed
  sorry

end intersection_M_N_l1719_171912


namespace greatest_value_inequality_l1719_171913

theorem greatest_value_inequality (x : ℝ) :
  x^2 - 6 * x + 8 ≤ 0 → x ≤ 4 := 
sorry

end greatest_value_inequality_l1719_171913


namespace seashells_broken_l1719_171960

theorem seashells_broken (total_seashells : ℕ) (unbroken_seashells : ℕ) (broken_seashells : ℕ) : 
  total_seashells = 6 → unbroken_seashells = 2 → broken_seashells = total_seashells - unbroken_seashells → broken_seashells = 4 :=
by
  intros ht hu hb
  rw [ht, hu] at hb
  exact hb

end seashells_broken_l1719_171960


namespace conical_surface_radius_l1719_171922

theorem conical_surface_radius (r : ℝ) :
  (2 * Real.pi * r = 5 * Real.pi) → r = 2.5 :=
by
  sorry

end conical_surface_radius_l1719_171922


namespace sum_ratio_l1719_171989

noncomputable def S (n : ℕ) : ℝ := sorry -- placeholder definition

def arithmetic_geometric_sum : Prop :=
  S 3 = 2 ∧ S 6 = 18

theorem sum_ratio :
  arithmetic_geometric_sum → S 10 / S 5 = 33 :=
by
  intros h 
  sorry 

end sum_ratio_l1719_171989


namespace smallest_is_B_l1719_171965

def A : ℕ := 32 + 7
def B : ℕ := (3 * 10) + 3
def C : ℕ := 50 - 9

theorem smallest_is_B : min A (min B C) = B := 
by 
  have hA : A = 39 := by rfl
  have hB : B = 33 := by rfl
  have hC : C = 41 := by rfl
  rw [hA, hB, hC]
  exact sorry

end smallest_is_B_l1719_171965


namespace correct_statements_l1719_171951

-- Definitions for statements A, B, C, and D
def statementA (x : ℝ) : Prop := |x| > 1 → x > 1
def statementB (A B C : ℝ) : Prop := (C > 90) ↔ (A + B + C = 180 ∧ (A > 90 ∨ B > 90 ∨ C > 90))
def statementC (a b : ℝ) : Prop := (a * b ≠ 0) ↔ (a ≠ 0 ∧ b ≠ 0)
def statementD (a b : ℝ) : Prop := a > b → 1 / a < 1 / b

-- Proof problem stating which statements are correct
theorem correct_statements :
  (∀ x : ℝ, statementA x = false) ∧ 
  (∀ (A B C : ℝ), statementB A B C = false) ∧ 
  (∀ (a b : ℝ), statementC a b) ∧ 
  (∀ (a b : ℝ), statementD a b = false) :=
by
  sorry

end correct_statements_l1719_171951


namespace bank1_more_advantageous_l1719_171932

-- Define the quarterly interest rate for Bank 1
def bank1_quarterly_rate : ℝ := 0.8

-- Define the annual interest rate for Bank 2
def bank2_annual_rate : ℝ := 9.0

-- Define the annual compounded interest rate for Bank 1
def bank1_annual_yield : ℝ :=
  (1 + bank1_quarterly_rate) ^ 4

-- Define the annual rate directly for Bank 2
def bank2_annual_yield : ℝ :=
  1 + bank2_annual_rate

-- The theorem stating that Bank 1 is more advantageous than Bank 2
theorem bank1_more_advantageous : bank1_annual_yield > bank2_annual_yield :=
  sorry

end bank1_more_advantageous_l1719_171932


namespace max_area_of_triangle_on_parabola_l1719_171945

noncomputable def area_of_triangle_ABC (p : ℝ) : ℝ :=
  (1 / 2) * abs (3 * p^2 - 14 * p + 15)

theorem max_area_of_triangle_on_parabola :
  ∃ p : ℝ, 1 ≤ p ∧ p ≤ 3 ∧ area_of_triangle_ABC p = 2 := sorry

end max_area_of_triangle_on_parabola_l1719_171945


namespace min_gb_for_plan_y_to_be_cheaper_l1719_171962

theorem min_gb_for_plan_y_to_be_cheaper (g : ℕ) : 20 * g > 3000 + 10 * g → g ≥ 301 := by
  sorry

end min_gb_for_plan_y_to_be_cheaper_l1719_171962


namespace num_ways_to_designated_face_l1719_171936

-- Define the structure of the dodecahedron
inductive Face
| Top
| Bottom
| TopRing (n : ℕ)   -- n ranges from 1 to 5
| BottomRing (n : ℕ)  -- n ranges from 1 to 5
deriving Repr, DecidableEq

-- Define adjacency relations on Faces (simplified)
def adjacent : Face → Face → Prop
| Face.Top, Face.TopRing n          => true
| Face.TopRing n, Face.TopRing m    => (m = (n % 5) + 1) ∨ (m = ((n + 3) % 5) + 1)
| Face.TopRing n, Face.BottomRing m => true
| Face.BottomRing n, Face.BottomRing m => true
| _, _ => false

-- Predicate for specific face on the bottom ring
def designated_bottom_face (f : Face) : Prop :=
  match f with
  | Face.BottomRing 1 => true
  | _ => false

-- Define the number of ways to move from top to the designated bottom face
noncomputable def num_ways : ℕ :=
  5 + 10

-- Lean statement that represents our equivalent proof problem
theorem num_ways_to_designated_face :
  num_ways = 15 := by
  sorry

end num_ways_to_designated_face_l1719_171936


namespace find_z_l1719_171971

noncomputable def solve_for_z (i : ℂ) (z : ℂ) :=
  (2 - i) * z = i ^ 2021

theorem find_z (i z : ℂ) (h1 : solve_for_z i z) : 
  z = -1/5 + 2/5 * i := 
by 
  sorry

end find_z_l1719_171971


namespace correct_model_is_pakistan_traditional_l1719_171907

-- Given definitions
def hasPrimitiveModel (country : String) : Prop := country = "Nigeria"
def hasTraditionalModel (country : String) : Prop := country = "India" ∨ country = "Pakistan" ∨ country = "Nigeria"
def hasModernModel (country : String) : Prop := country = "China"

-- The proposition to prove
theorem correct_model_is_pakistan_traditional :
  (hasPrimitiveModel "Nigeria")
  ∧ (hasModernModel "China")
  ∧ (hasTraditionalModel "India")
  ∧ (hasTraditionalModel "Pakistan") →
  (hasTraditionalModel "Pakistan") := by
  intros h
  exact (h.right.right.right)

end correct_model_is_pakistan_traditional_l1719_171907


namespace bacteria_population_at_2_15_l1719_171955

noncomputable def bacteria_at_time (initial_pop : ℕ) (start_time end_time : ℕ) (interval : ℕ) : ℕ :=
  initial_pop * 2 ^ ((end_time - start_time) / interval)

theorem bacteria_population_at_2_15 :
  let initial_pop := 50
  let start_time := 0  -- 2:00 p.m.
  let end_time := 15   -- 2:15 p.m.
  let interval := 4
  bacteria_at_time initial_pop start_time end_time interval = 400 := sorry

end bacteria_population_at_2_15_l1719_171955


namespace vertex_of_parabola_l1719_171902

theorem vertex_of_parabola (x : ℝ) : 
  ∀ x y : ℝ, (y = x^2 - 6 * x + 1) → (∃ h k : ℝ, y = (x - h)^2 + k ∧ h = 3 ∧ k = -8) :=
by
  -- This is to state that given the parabola equation x^2 - 6x + 1, its vertex coordinates are (3, -8).
  sorry

end vertex_of_parabola_l1719_171902


namespace max_xy_l1719_171915

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions given in the problem
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom eq1 : x + 1/y = 3
axiom eq2 : y + 2/x = 3

theorem max_xy : ∃ (xy : ℝ), 
  xy = x * y ∧ xy = 3 + Real.sqrt 7 := sorry

end max_xy_l1719_171915


namespace rectangle_area_inscribed_circle_l1719_171957

theorem rectangle_area_inscribed_circle 
  (radius : ℝ) (width len : ℝ) 
  (h_radius : radius = 5) 
  (h_width : width = 2 * radius) 
  (h_len_ratio : len = 3 * width) 
  : width * len = 300 := 
by
  sorry

end rectangle_area_inscribed_circle_l1719_171957


namespace Correct_Statement_l1719_171939

theorem Correct_Statement : 
  (∀ x : ℝ, 7 * x = 4 * x - 3 → 7 * x - 4 * x = -3) ∧
  (∀ x : ℝ, (2 * x - 1) / 3 = 1 + (x - 3) / 2 → 2 * (2 * x - 1) = 6 + 3 * (x - 3)) ∧
  (∀ x : ℝ, 2 * (2 * x - 1) - 3 * (x - 3) = 1 → 4 * x - 2 - 3 * x + 9 = 1) ∧
  (∀ x : ℝ, 2 * (x + 1) = x + 7 → x = 5) :=
by
  sorry

end Correct_Statement_l1719_171939


namespace shortest_path_Dasha_Vasya_l1719_171996

-- Definitions for the given distances
def dist_Asya_Galia : ℕ := 12
def dist_Galia_Borya : ℕ := 10
def dist_Asya_Borya : ℕ := 8
def dist_Dasha_Galia : ℕ := 15
def dist_Vasya_Galia : ℕ := 17

-- Definition for shortest distance by roads from Dasha to Vasya
def shortest_dist_Dasha_Vasya : ℕ := 18

-- Proof statement of the goal that shortest distance from Dasha to Vasya is 18 km
theorem shortest_path_Dasha_Vasya : 
  dist_Dasha_Galia + dist_Vasya_Galia - dist_Asya_Galia - dist_Galia_Borya = shortest_dist_Dasha_Vasya := by
  sorry

end shortest_path_Dasha_Vasya_l1719_171996


namespace nina_total_spending_l1719_171983

-- Defining the quantities and prices of each category of items
def num_toys : Nat := 3
def price_per_toy : Nat := 10

def num_basketball_cards : Nat := 2
def price_per_card : Nat := 5

def num_shirts : Nat := 5
def price_per_shirt : Nat := 6

-- Calculating the total cost for each category
def cost_toys : Nat := num_toys * price_per_toy
def cost_cards : Nat := num_basketball_cards * price_per_card
def cost_shirts : Nat := num_shirts * price_per_shirt

-- Calculating the total amount spent
def total_cost : Nat := cost_toys + cost_cards + cost_shirts

-- The final theorem statement to verify the answer
theorem nina_total_spending : total_cost = 70 :=
by
  sorry

end nina_total_spending_l1719_171983


namespace find_sum_of_squares_l1719_171995

theorem find_sum_of_squares (x y : ℝ) (h1: x * y = 16) (h2: x^2 + y^2 = 34) : (x + y) ^ 2 = 66 :=
by sorry

end find_sum_of_squares_l1719_171995


namespace minimum_shirts_for_saving_money_l1719_171978

-- Define the costs for Acme and Gamma
def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

-- Prove that the minimum number of shirts x for which a customer saves money by using Acme is 13
theorem minimum_shirts_for_saving_money : ∃ (x : ℕ), 60 + 10 * x < 15 * x ∧ x = 13 := by
  sorry

end minimum_shirts_for_saving_money_l1719_171978


namespace total_cost_proof_l1719_171903

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end total_cost_proof_l1719_171903


namespace josh_total_money_left_l1719_171981

-- Definitions of the conditions
def profit_per_bracelet : ℝ := 1.5 - 1
def total_bracelets : ℕ := 12
def cost_of_cookies : ℝ := 3

-- The proof problem: 
theorem josh_total_money_left : total_bracelets * profit_per_bracelet - cost_of_cookies = 3 :=
by
  sorry

end josh_total_money_left_l1719_171981


namespace solution_to_equation_l1719_171935

theorem solution_to_equation (x : ℝ) (h : (5 - x / 2)^(1/3) = 2) : x = -6 :=
sorry

end solution_to_equation_l1719_171935


namespace angle_of_inclination_l1719_171924

theorem angle_of_inclination (θ : ℝ) : 
  (∀ x y : ℝ, x - y + 3 = 0 → ∃ θ : ℝ, Real.tan θ = 1 ∧ θ = Real.pi / 4) := by
  sorry

end angle_of_inclination_l1719_171924


namespace length_of_second_train_is_319_95_l1719_171976

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kph : ℝ) (speed_second_train_kph : ℝ) (time_to_cross_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kph * 1000 / 3600
  let speed_second_train_mps := speed_second_train_kph * 1000 / 3600
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross_seconds
  let length_second_train := total_distance_covered - length_first_train
  length_second_train

theorem length_of_second_train_is_319_95 :
  length_of_second_train 180 120 80 9 = 319.95 :=
sorry

end length_of_second_train_is_319_95_l1719_171976


namespace original_number_l1719_171977

theorem original_number (n : ℕ) (h : (n + 1) % 30 = 0) : n = 29 :=
by
  sorry

end original_number_l1719_171977


namespace tram_speed_l1719_171949

theorem tram_speed
  (L v : ℝ)
  (h1 : L = 2 * v)
  (h2 : 96 + L = 10 * v) :
  v = 12 := 
by sorry

end tram_speed_l1719_171949


namespace sum_first_ten_terms_arithmetic_sequence_l1719_171948

theorem sum_first_ten_terms_arithmetic_sequence (a₁ d : ℤ) (h₁ : a₁ = -3) (h₂ : d = 4) : 
  let a₁₀ := a₁ + (9 * d)
  let S := ((a₁ + a₁₀) / 2) * 10
  S = 150 :=
by
  subst h₁
  subst h₂
  let a₁₀ := -3 + (9 * 4)
  let S := ((-3 + a₁₀) / 2) * 10
  sorry

end sum_first_ten_terms_arithmetic_sequence_l1719_171948


namespace no_injective_function_l1719_171943

theorem no_injective_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m * n) = f m + f n) : ¬ Function.Injective f := 
sorry

end no_injective_function_l1719_171943


namespace prob_sin_ge_half_l1719_171980

theorem prob_sin_ge_half : 
  let a := -Real.pi / 6
  let b := Real.pi / 2
  let p := (Real.pi / 2 - Real.pi / 6) / (Real.pi / 2 + Real.pi / 6)
  a ≤ b ∧ a = -Real.pi / 6 ∧ b = Real.pi / 2 → p = 1 / 2 :=
by
  sorry

end prob_sin_ge_half_l1719_171980


namespace smaller_number_of_product_l1719_171929

theorem smaller_number_of_product :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5610 ∧ a = 34 :=
by
  -- Proof would go here
  sorry

end smaller_number_of_product_l1719_171929


namespace find_constants_C_D_l1719_171925

theorem find_constants_C_D
  (C : ℚ) (D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → (5 * x - 3) / (x^2 - 5 * x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32 / 9 ∧ D = 13 / 9 :=
by
  sorry

end find_constants_C_D_l1719_171925


namespace sequence_sum_l1719_171970

theorem sequence_sum:
  ∀ (y : ℕ → ℕ), 
  (y 1 = 100) → 
  (∀ k ≥ 2, y k = y (k - 1) ^ 2 + 2 * y (k - 1) + 1) →
  ( ∑' n, 1 / (y n + 1) = 1 / 101 ) :=
by
  sorry

end sequence_sum_l1719_171970


namespace geometric_seq_arith_seq_problem_l1719_171934

theorem geometric_seq_arith_seq_problem (a : ℕ → ℝ) (q : ℝ)
  (h : ∀ n, a (n + 1) = q * a n)
  (h_q_pos : q > 0)
  (h_arith : 2 * (1/2 : ℝ) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 2014 - a 2015) / (a 2016 - a 2017) = 1 / 9 := 
sorry

end geometric_seq_arith_seq_problem_l1719_171934


namespace total_fuel_usage_is_250_l1719_171941

-- Define John's fuel consumption per km
def fuel_consumption_per_km : ℕ := 5

-- Define the distance of the first trip
def distance_trip1 : ℕ := 30

-- Define the distance of the second trip
def distance_trip2 : ℕ := 20

-- Define the fuel usage calculation
def fuel_usage_trip1 := distance_trip1 * fuel_consumption_per_km
def fuel_usage_trip2 := distance_trip2 * fuel_consumption_per_km
def total_fuel_usage := fuel_usage_trip1 + fuel_usage_trip2

-- Prove that the total fuel usage is 250 liters
theorem total_fuel_usage_is_250 : total_fuel_usage = 250 := by
  sorry

end total_fuel_usage_is_250_l1719_171941


namespace age_of_replaced_man_l1719_171919

-- Definitions based on conditions
def avg_age_men (A : ℝ) := A
def age_man1 := 10
def avg_age_women := 23
def total_age_women := 2 * avg_age_women
def new_avg_age_men (A : ℝ) := A + 2

-- Proposition stating that given conditions yield the age of the other replaced man
theorem age_of_replaced_man (A M : ℝ) :
  8 * avg_age_men A - age_man1 - M + total_age_women = 8 * new_avg_age_men A + 16 →
  M = 20 :=
by
  sorry

end age_of_replaced_man_l1719_171919


namespace quotient_of_f_div_g_l1719_171958

-- Define the polynomial f(x) = x^5 + 5
def f (x : ℝ) : ℝ := x ^ 5 + 5

-- Define the divisor polynomial g(x) = x - 1
def g (x : ℝ) : ℝ := x - 1

-- Define the expected quotient polynomial q(x) = x^4 + x^3 + x^2 + x + 1
def q (x : ℝ) : ℝ := x ^ 4 + x ^ 3 + x ^ 2 + x + 1

-- State and prove the main theorem
theorem quotient_of_f_div_g (x : ℝ) :
  ∃ r : ℝ, f x = g x * (q x) + r :=
by
  sorry

end quotient_of_f_div_g_l1719_171958


namespace min_abs_ab_perpendicular_lines_l1719_171937

theorem min_abs_ab_perpendicular_lines (a b : ℝ) (h : a * b = a ^ 2 + 1) : |a * b| = 1 :=
by sorry

end min_abs_ab_perpendicular_lines_l1719_171937


namespace find_m_if_root_zero_l1719_171982

theorem find_m_if_root_zero (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + x + (m^2 - 1) = 0) → m = -1 :=
by
  intro h
  -- Term after this point not necessary, hence a placeholder
  sorry

end find_m_if_root_zero_l1719_171982


namespace integer_solution_x_l1719_171986

theorem integer_solution_x (x : ℤ) (h₁ : x + 8 > 10) (h₂ : -3 * x < -9) : x ≥ 4 ↔ x > 3 := by
  sorry

end integer_solution_x_l1719_171986


namespace triangular_array_nth_row_4th_number_l1719_171974

theorem triangular_array_nth_row_4th_number (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, k = 4 ∧ (2: ℕ)^(n * (n - 1) / 2 + 3) = 2^((n^2 - n + 6) / 2) :=
by
  sorry

end triangular_array_nth_row_4th_number_l1719_171974


namespace butcher_net_loss_l1719_171972

noncomputable def dishonest_butcher (advertised_price actual_price : ℝ) (quantity_sold : ℕ) (fine : ℝ) : ℝ :=
  let dishonest_gain_per_kg := actual_price - advertised_price
  let total_dishonest_gain := dishonest_gain_per_kg * quantity_sold
  fine - total_dishonest_gain

theorem butcher_net_loss 
  (advertised_price : ℝ) 
  (actual_price : ℝ) 
  (quantity_sold : ℕ) 
  (fine : ℝ)
  (h_advertised_price : advertised_price = 3.79)
  (h_actual_price : actual_price = 4.00)
  (h_quantity_sold : quantity_sold = 1800)
  (h_fine : fine = 500) :
  dishonest_butcher advertised_price actual_price quantity_sold fine = 122 := 
by
  simp [dishonest_butcher, h_advertised_price, h_actual_price, h_quantity_sold, h_fine]
  sorry

end butcher_net_loss_l1719_171972


namespace varies_fix_l1719_171918

variable {x y z : ℝ}

theorem varies_fix {k j : ℝ} 
  (h1 : x = k * y^4)
  (h2 : y = j * z^(1/3)) : x = (k * j^4) * z^(4/3) := by
  sorry

end varies_fix_l1719_171918


namespace triangle_perimeter_l1719_171973

theorem triangle_perimeter (r A : ℝ) (h_r : r = 2.5) (h_A : A = 50) : 
  ∃ p : ℝ, p = 40 :=
by
  sorry

end triangle_perimeter_l1719_171973


namespace value_of_B_l1719_171993

theorem value_of_B (x y : ℕ) (h1 : x > y) (h2 : y > 1) (h3 : x * y = x + y + 22) :
  (x / y) = 12 :=
sorry

end value_of_B_l1719_171993


namespace solution_set_of_x_x_plus_2_lt_3_l1719_171944

theorem solution_set_of_x_x_plus_2_lt_3 :
  {x : ℝ | x*(x + 2) < 3} = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_x_x_plus_2_lt_3_l1719_171944


namespace darts_game_score_l1719_171909

variable (S1 S2 S3 : ℕ)
variable (n : ℕ)

theorem darts_game_score :
  n = 8 →
  S2 = 2 * S1 →
  S3 = (3 * S1) →
  S2 = 48 :=
by
  intros h1 h2 h3
  sorry

end darts_game_score_l1719_171909


namespace awards_distribution_count_l1719_171938

-- Define the problem conditions
def num_awards : Nat := 5
def num_students : Nat := 3

-- Verify each student gets at least one award
def each_student_gets_at_least_one (distributions : List (List Nat)) : Prop :=
  ∀ (dist : List Nat), dist ∈ distributions → (∀ (d : Nat), d > 0)

-- Define the main theorem to be proved
theorem awards_distribution_count :
  ∃ (distributions : List (List Nat)), each_student_gets_at_least_one distributions ∧ distributions.length = 150 :=
sorry

end awards_distribution_count_l1719_171938


namespace slope_range_of_tangent_line_l1719_171908

theorem slope_range_of_tangent_line (x : ℝ) (h : x ≠ 0) : (1 - 1/(x^2)) < 1 :=
by
  calc 
    1 - 1/(x^2) < 1 := sorry

end slope_range_of_tangent_line_l1719_171908


namespace polar_to_cartesian_l1719_171901

theorem polar_to_cartesian (ρ θ x y : ℝ) (h1 : ρ = 2 * Real.sin θ)
  (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end polar_to_cartesian_l1719_171901


namespace a_value_for_even_function_l1719_171923

def f (x a : ℝ) := (x + 1) * (x + a)

theorem a_value_for_even_function (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = -1 :=
by
  sorry

end a_value_for_even_function_l1719_171923


namespace evaluate_expression_l1719_171916

theorem evaluate_expression (a : ℝ) (h : a = -3) : 
  (3 * a⁻¹ + (a⁻¹ / 3)) / a = 10 / 27 :=
by 
  sorry

end evaluate_expression_l1719_171916
