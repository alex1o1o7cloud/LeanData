import Mathlib

namespace fractional_shaded_area_l232_23297

noncomputable def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)

theorem fractional_shaded_area :
  let a := (7 : ℚ) / 16
  let r := (1 : ℚ) / 16
  geometric_series_sum a r = 7 / 15 :=
by
  sorry

end fractional_shaded_area_l232_23297


namespace smallest_value_of_a_l232_23298

theorem smallest_value_of_a :
  ∃ (a b : ℤ) (r1 r2 r3 : ℕ), 
  r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
  r1 * r2 * r3 = 2310 ∧ r1 + r2 + r3 = a ∧ 
  (∀ (r1' r2' r3' : ℕ), (r1' > 0 ∧ r2' > 0 ∧ r3' > 0 ∧ r1' * r2' * r3' = 2310) → r1' + r2' + r3' ≥ a) ∧ 
  a = 88 :=
by sorry

end smallest_value_of_a_l232_23298


namespace cistern_filling_time_l232_23225

/-- Define the rates at which the cistern is filled and emptied -/
def fill_rate := (1 : ℚ) / 3
def empty_rate := (1 : ℚ) / 8

/-- Define the net rate of filling when both taps are open -/
def net_rate := fill_rate - empty_rate

/-- Define the volume of the cistern -/
def cistern_volume := (1 : ℚ)

/-- Compute the time to fill the cistern given the net rate -/
def fill_time := cistern_volume / net_rate

theorem cistern_filling_time :
  fill_time = 4.8 := by
sorry

end cistern_filling_time_l232_23225


namespace expected_hit_targets_correct_expected_hit_targets_at_least_half_l232_23260

noncomputable def expected_hit_targets (n : ℕ) : ℝ :=
  n * (1 - (1 - (1 : ℝ) / n)^n)

theorem expected_hit_targets_correct (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n = n * (1 - (1 - (1 : ℝ) / n)^n) :=
by
  unfold expected_hit_targets
  sorry

theorem expected_hit_targets_at_least_half (n : ℕ) (h_pos : n > 0) :
  expected_hit_targets n >= n / 2 :=
by
  unfold expected_hit_targets
  sorry

end expected_hit_targets_correct_expected_hit_targets_at_least_half_l232_23260


namespace houston_firewood_l232_23243

theorem houston_firewood (k e h : ℕ) (k_collected : k = 10) (e_collected : e = 13) (total_collected : k + e + h = 35) : h = 12 :=
by
  sorry

end houston_firewood_l232_23243


namespace unique_two_digit_number_l232_23274

theorem unique_two_digit_number (x y : ℕ) (h1 : 10 ≤ 10 * x + y ∧ 10 * x + y < 100) (h2 : 3 * y = 2 * x) (h3 : y + 3 = x) : 10 * x + y = 63 :=
by
  sorry

end unique_two_digit_number_l232_23274


namespace katie_total_expenditure_l232_23285

-- Define the conditions
def flower_cost : ℕ := 6
def roses_bought : ℕ := 5
def daisies_bought : ℕ := 5

-- Define the total flowers bought
def total_flowers_bought : ℕ := roses_bought + daisies_bought

-- Calculate the total cost
def total_cost (flower_cost : ℕ) (total_flowers_bought : ℕ) : ℕ :=
  total_flowers_bought * flower_cost

-- Prove that Katie spent 60 dollars
theorem katie_total_expenditure : total_cost flower_cost total_flowers_bought = 60 := sorry

end katie_total_expenditure_l232_23285


namespace a3_mul_a7_eq_36_l232_23264

-- Definition of a geometric sequence term
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ r : ℤ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given conditions
def a (n : ℕ) : ℤ := sorry  -- Placeholder for the geometric sequence

axiom a5_eq_6 : a 5 = 6  -- Given that a_5 = 6

axiom geo_seq : geometric_sequence a  -- The sequence is geometric

-- Problem statement: Prove that a_3 * a_7 = 36
theorem a3_mul_a7_eq_36 : a 3 * a 7 = 36 :=
  sorry

end a3_mul_a7_eq_36_l232_23264


namespace minimum_throws_for_repeated_sum_l232_23205

theorem minimum_throws_for_repeated_sum :
  let min_sum := 4 * 1
  let max_sum := 4 * 6
  let num_distinct_sums := max_sum - min_sum + 1
  let min_throws := num_distinct_sums + 1
  min_throws = 22 :=
by
  sorry

end minimum_throws_for_repeated_sum_l232_23205


namespace samples_from_workshop_l232_23244

theorem samples_from_workshop (T S P : ℕ) (hT : T = 2048) (hS : S = 128) (hP : P = 256) : 
  (s : ℕ) → (s : ℕ) = (256 * 128 / 2048) → s = 16 :=
by
  intros s hs
  rw [Nat.div_eq (256 * 128) 2048] at hs
  sorry

end samples_from_workshop_l232_23244


namespace hospital_staff_total_l232_23275

def initial_doctors := 11
def initial_nurses := 18
def initial_medical_assistants := 9
def initial_interns := 6

def doctors_quit := 5
def nurses_quit := 2
def medical_assistants_quit := 3
def nurses_transferred := 2
def interns_transferred := 4
def doctors_vacation := 4
def nurses_vacation := 3

def new_doctors := 3
def new_nurses := 5

def remaining_doctors := initial_doctors - doctors_quit - doctors_vacation
def remaining_nurses := initial_nurses - nurses_quit - nurses_transferred - nurses_vacation
def remaining_medical_assistants := initial_medical_assistants - medical_assistants_quit
def remaining_interns := initial_interns - interns_transferred

def final_doctors := remaining_doctors + new_doctors
def final_nurses := remaining_nurses + new_nurses
def final_medical_assistants := remaining_medical_assistants
def final_interns := remaining_interns

def total_staff := final_doctors + final_nurses + final_medical_assistants + final_interns

theorem hospital_staff_total : total_staff = 29 := by
  unfold total_staff
  unfold final_doctors
  unfold final_nurses
  unfold final_medical_assistants
  unfold final_interns
  unfold remaining_doctors
  unfold remaining_nurses
  unfold remaining_medical_assistants
  unfold remaining_interns
  unfold initial_doctors initial_nurses initial_medical_assistants initial_interns
  unfold doctors_quit nurses_quit medical_assistants_quit nurses_transferred interns_transferred
  unfold doctors_vacation nurses_vacation
  unfold new_doctors new_nurses
  sorry

end hospital_staff_total_l232_23275


namespace Haleigh_needs_leggings_l232_23266

/-- Haleigh's pet animals -/
def dogs : Nat := 4
def cats : Nat := 3
def legs_per_dog : Nat := 4
def legs_per_cat : Nat := 4
def leggings_per_pair : Nat := 2

/-- The proof statement -/
theorem Haleigh_needs_leggings : (dogs * legs_per_dog + cats * legs_per_cat) / leggings_per_pair = 14 := by
  sorry

end Haleigh_needs_leggings_l232_23266


namespace problem_inequality_l232_23284

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : x^2 + y^2 + z^2 + x*y + y*z + z*x ≤ 1) : 
  (1/x - 1) * (1/y - 1) * (1/z - 1) ≥ 9 * Real.sqrt 6 - 19 :=
sorry

end problem_inequality_l232_23284


namespace time_after_2021_hours_l232_23259

-- Definition of starting time and day
def start_time : Nat := 20 * 60 + 21  -- converting 20:21 to minutes
def hours_per_day : Nat := 24
def minutes_per_hour : Nat := 60
def days_per_week : Nat := 7

-- Define the main statement
theorem time_after_2021_hours :
  let total_minutes := 2021 * minutes_per_hour
  let total_days := total_minutes / (hours_per_day * minutes_per_hour)
  let remaining_minutes := total_minutes % (hours_per_day * minutes_per_hour)
  let final_minutes := start_time + remaining_minutes
  let final_day := (total_days + 1) % days_per_week -- start on Monday (0), hence +1 for Tuesday
  final_minutes / minutes_per_hour = 1 ∧ final_minutes % minutes_per_hour = 21 ∧ final_day = 2 :=
by
  sorry

end time_after_2021_hours_l232_23259


namespace jar_water_fraction_l232_23218

theorem jar_water_fraction
  (S L : ℝ)
  (h1 : S = (1 / 5) * S)
  (h2 : S = x * L)
  (h3 : (1 / 5) * S + x * L = (2 / 5) * L) :
  x = (1 / 10) :=
by
  sorry

end jar_water_fraction_l232_23218


namespace decrease_by_150_percent_l232_23249

theorem decrease_by_150_percent (x : ℝ) (h : x = 80) : x - 1.5 * x = -40 :=
by
  sorry

end decrease_by_150_percent_l232_23249


namespace acute_angle_at_7_20_is_100_degrees_l232_23235

theorem acute_angle_at_7_20_is_100_degrees :
  let minute_hand_angle := 4 * 30 -- angle of the minute hand (in degrees)
  let hour_hand_progress := 20 / 60 -- progress of hour hand between 7 and 8
  let hour_hand_angle := 7 * 30 + hour_hand_progress * 30 -- angle of the hour hand (in degrees)

  ∃ angle_acute : ℝ, 
  angle_acute = abs (minute_hand_angle - hour_hand_angle) ∧
  angle_acute = 100 :=
by
  sorry

end acute_angle_at_7_20_is_100_degrees_l232_23235


namespace votes_for_Crow_l232_23263

theorem votes_for_Crow 
  (J : ℕ)
  (P V K : ℕ)
  (ε1 ε2 ε3 ε4 : ℤ)
  (h₁ : P + V = 15 + ε1)
  (h₂ : V + K = 18 + ε2)
  (h₃ : K + P = 20 + ε3)
  (h₄ : P + V + K = 59 + ε4)
  (bound₁ : |ε1| ≤ 13)
  (bound₂ : |ε2| ≤ 13)
  (bound₃ : |ε3| ≤ 13)
  (bound₄ : |ε4| ≤ 13)
  : V = 13 :=
sorry

end votes_for_Crow_l232_23263


namespace sum_of_g_is_zero_l232_23295

def g (x : ℝ) : ℝ := x^3 * (1 - x)^3

theorem sum_of_g_is_zero :
  (Finset.range 2022).sum (λ k => (-1)^(k + 1) * g ((k + 1 : ℝ) / 2023)) = 0 :=
by
  sorry

end sum_of_g_is_zero_l232_23295


namespace old_supervisor_salary_correct_l232_23261

def old_supervisor_salary (W S_old : ℝ) : Prop :=
  let avg_old := (W + S_old) / 9
  let avg_new := (W + 510) / 9
  avg_old = 430 ∧ avg_new = 390 → S_old = 870

theorem old_supervisor_salary_correct (W : ℝ) :
  old_supervisor_salary W 870 :=
by
  unfold old_supervisor_salary
  intro h
  sorry

end old_supervisor_salary_correct_l232_23261


namespace option_b_correct_l232_23215

theorem option_b_correct (a b m : ℝ) (h1 : a > b) (h2 : b > 0) (h3: a ≠ 1) (h4: b ≠ 1) (h5 : 0 < m) (h6 : m < 1) :
  m^a < m^b :=
sorry

end option_b_correct_l232_23215


namespace algebra_expression_value_l232_23210

theorem algebra_expression_value (x y : ℝ) (h1 : x * y = 3) (h2 : x - y = -2) : x^2 * y - x * y^2 = -6 := 
by
  sorry

end algebra_expression_value_l232_23210


namespace find_smallest_n_l232_23231

open Matrix Complex

noncomputable def rotation_matrix := ![
  ![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
  ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]
]

def I_2 := (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem find_smallest_n (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = rotation_matrix) : 
  ∃ (n : ℕ), 0 < n ∧ A ^ n = I_2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → A ^ m ≠ I_2 :=
by {
  sorry
}

end find_smallest_n_l232_23231


namespace exists_n_for_pn_consecutive_zeros_l232_23241

theorem exists_n_for_pn_consecutive_zeros (p : ℕ) (hp : Nat.Prime p) (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, (∃ k : ℕ, (p^n) / 10^(k+m) % 10^m = 0) := sorry

end exists_n_for_pn_consecutive_zeros_l232_23241


namespace find_integer_m_l232_23228

theorem find_integer_m (m : ℤ) :
  (∃! x : ℤ, |2 * x - m| ≤ 1 ∧ x = 2) → m = 4 :=
by
  intro h
  sorry

end find_integer_m_l232_23228


namespace value_of_A_l232_23217

theorem value_of_A (h p a c k e : ℤ) 
  (H : h = 8)
  (PACK : p + a + c + k = 50)
  (PECK : p + e + c + k = 54)
  (CAKE : c + a + k + e = 40) : 
  a = 25 :=
by 
  sorry

end value_of_A_l232_23217


namespace option_C_is_different_l232_23279

def cause_and_effect_relationship (description: String) : Prop :=
  description = "A: Great teachers produce outstanding students" ∨
  description = "B: When the water level rises, the boat goes up" ∨
  description = "D: The higher you climb, the farther you see"

def not_cause_and_effect_relationship (description: String) : Prop :=
  description = "C: The brighter the moon, the fewer the stars"

theorem option_C_is_different :
  ∀ (description: String),
  (not_cause_and_effect_relationship description) →
  ¬ cause_and_effect_relationship description :=
by intros description h1 h2; sorry

end option_C_is_different_l232_23279


namespace intersection_value_l232_23256

theorem intersection_value (x y : ℝ) (h₁ : y = 10 / (x^2 + 5)) (h₂ : x + 2 * y = 5) : 
  x = 1 :=
sorry

end intersection_value_l232_23256


namespace number_as_A_times_10_pow_N_integer_l232_23253

theorem number_as_A_times_10_pow_N_integer (A : ℝ) (N : ℝ) (hA1 : 1 ≤ A) (hA2 : A < 10) (hN : A * 10^N > 10) : ∃ (n : ℤ), N = n := 
sorry

end number_as_A_times_10_pow_N_integer_l232_23253


namespace min_distance_point_curve_to_line_l232_23245

noncomputable def curve (x : ℝ) : ℝ := x^2 - Real.log x

def line (x y : ℝ) : Prop := x - y - 2 = 0

theorem min_distance_point_curve_to_line :
  ∀ (P : ℝ × ℝ), 
  curve P.1 = P.2 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 :=
by
  sorry

end min_distance_point_curve_to_line_l232_23245


namespace right_triangle_area_l232_23278

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l232_23278


namespace max_value_of_function_for_x_lt_0_l232_23219

noncomputable def f (x : ℝ) : ℝ :=
  x + 4 / x

theorem max_value_of_function_for_x_lt_0 :
  ∀ x : ℝ, x < 0 → f x ≤ -4 ∧ (∃ y : ℝ, f y = -4 ∧ y < 0) := sorry

end max_value_of_function_for_x_lt_0_l232_23219


namespace find_k_l232_23230

theorem find_k (t k : ℤ) (h1 : t = 35) (h2 : t = 5 * (k - 32) / 9) : k = 95 :=
sorry

end find_k_l232_23230


namespace max_xy_on_line_AB_l232_23280

noncomputable def pointA : ℝ × ℝ := (3, 0)
noncomputable def pointB : ℝ × ℝ := (0, 4)

-- Define the line passing through points A and B
def on_line_AB (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P.1 = 3 - 3 * t ∧ P.2 = 4 * t

theorem max_xy_on_line_AB : ∃ (P : ℝ × ℝ), on_line_AB P ∧ P.1 * P.2 = 3 := 
sorry

end max_xy_on_line_AB_l232_23280


namespace average_price_per_share_l232_23257

-- Define the conditions
def Microtron_price_per_share := 36
def Dynaco_price_per_share := 44
def total_shares := 300
def Dynaco_shares_sold := 150

-- Define the theorem to be proved
theorem average_price_per_share : 
  (Dynaco_shares_sold * Dynaco_price_per_share + (total_shares - Dynaco_shares_sold) * Microtron_price_per_share) / total_shares = 40 :=
by
  -- Skip the actual proof here
  sorry

end average_price_per_share_l232_23257


namespace minimize_wire_length_l232_23288

theorem minimize_wire_length :
  ∃ (x : ℝ), (x > 0) ∧ (2 * (x + 4 / x) = 8) :=
by
  sorry

end minimize_wire_length_l232_23288


namespace option_C_cannot_form_right_triangle_l232_23234

def is_right_triangle_sides (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem option_C_cannot_form_right_triangle :
  ¬ (is_right_triangle_sides 1.5 2 3) :=
by
  -- This is intentionally left incomplete as per instructions
  sorry

end option_C_cannot_form_right_triangle_l232_23234


namespace triangle_area_solutions_l232_23289

theorem triangle_area_solutions (ABC BDE : ℝ) (k : ℝ) (h₁ : BDE = k^2) : 
  S >= 4 * k^2 ∧ (if S = 4 * k^2 then solutions = 1 else solutions = 2) :=
by
  sorry

end triangle_area_solutions_l232_23289


namespace ABC_books_sold_eq_4_l232_23200

/-- "TOP" book cost in dollars --/
def TOP_price : ℕ := 8

/-- "ABC" book cost in dollars --/
def ABC_price : ℕ := 23

/-- Number of "TOP" books sold --/
def TOP_books_sold : ℕ := 13

/-- Difference in earnings in dollars --/
def earnings_difference : ℕ := 12

/-- Prove the number of "ABC" books sold --/
theorem ABC_books_sold_eq_4 (x : ℕ) (h : TOP_books_sold * TOP_price - x * ABC_price = earnings_difference) : x = 4 :=
by
  sorry

end ABC_books_sold_eq_4_l232_23200


namespace production_profit_range_l232_23277

theorem production_profit_range (x : ℝ) (t : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 10) (h3 : 0 ≤ t) :
  (200 * (5 * x + 1 - 3 / x) ≥ 3000) → (3 ≤ x ∧ x ≤ 10) :=
sorry

end production_profit_range_l232_23277


namespace proof_problem_l232_23286

def h (x : ℝ) : ℝ := 2 * x + 4
def k (x : ℝ) : ℝ := 4 * x + 6

theorem proof_problem : h (k 3) - k (h 3) = -6 :=
by
  sorry

end proof_problem_l232_23286


namespace diane_faster_than_rhonda_l232_23201

theorem diane_faster_than_rhonda :
  ∀ (rhonda_time sally_time diane_time total_time : ℕ), 
  rhonda_time = 24 →
  sally_time = rhonda_time + 2 →
  total_time = 71 →
  total_time = rhonda_time + sally_time + diane_time →
  (rhonda_time - diane_time) = 3 :=
by
  intros rhonda_time sally_time diane_time total_time
  intros h_rhonda h_sally h_total h_sum
  sorry

end diane_faster_than_rhonda_l232_23201


namespace tom_reads_700_pages_in_7_days_l232_23291

theorem tom_reads_700_pages_in_7_days
  (total_hours : ℕ)
  (total_days : ℕ)
  (pages_per_hour : ℕ)
  (reads_same_amount_every_day : Prop)
  (h1 : total_hours = 10)
  (h2 : total_days = 5)
  (h3 : pages_per_hour = 50)
  (h4 : reads_same_amount_every_day) :
  (total_hours / total_days) * (pages_per_hour * 7) = 700 :=
by
  -- Begin and skip proof with sorry
  sorry

end tom_reads_700_pages_in_7_days_l232_23291


namespace sum_of_angles_l232_23212

theorem sum_of_angles (A B C D E F : ℝ)
  (h1 : A + B + C = 180) 
  (h2 : D + E + F = 180) : 
  A + B + C + D + E + F = 360 := 
by 
  sorry

end sum_of_angles_l232_23212


namespace quadratic_inequality_solution_l232_23214

theorem quadratic_inequality_solution 
  (a b c : ℝ)
  (h1 : a < 0)
  (h2 : 1 + 2 = b / a)
  (h3 : 1 * 2 = c / a) :
  ∀ x : ℝ, cx^2 + bx + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1 / 2 :=
by
  sorry

end quadratic_inequality_solution_l232_23214


namespace greatest_y_least_y_greatest_integer_y_l232_23271

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l232_23271


namespace min_max_SX_SY_l232_23208

theorem min_max_SX_SY (n : ℕ) (hn : 2 ≤ n) (a : Finset ℕ) 
  (ha_sum : Finset.sum a id = 2 * n - 1) :
  ∃ (min_val max_val : ℕ), 
    (min_val = 2 * n - 2) ∧ 
    (max_val = n * (n - 1)) :=
sorry

end min_max_SX_SY_l232_23208


namespace first_term_of_infinite_geometric_series_l232_23207

theorem first_term_of_infinite_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : a = 12 := 
sorry

end first_term_of_infinite_geometric_series_l232_23207


namespace smallest_integer_geq_l232_23262

theorem smallest_integer_geq : ∃ (n : ℤ), (n^2 - 9*n + 18 ≥ 0) ∧ ∀ (m : ℤ), (m^2 - 9*m + 18 ≥ 0) → n ≤ m :=
by
  sorry

end smallest_integer_geq_l232_23262


namespace bridge_length_proof_l232_23209

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_of_train_km_per_hr : ℝ) (time_to_cross_bridge : ℝ) : ℝ :=
  let speed_of_train_m_per_s := speed_of_train_km_per_hr * (1000 / 3600)
  let total_distance := speed_of_train_m_per_s * time_to_cross_bridge
  total_distance - length_of_train

theorem bridge_length_proof : length_of_bridge 100 75 11.279097672186225 = 135 := by
  simp [length_of_bridge]
  sorry

end bridge_length_proof_l232_23209


namespace inequality_and_equality_condition_l232_23299

theorem inequality_and_equality_condition (x : ℝ)
  (h : x ∈ (Set.Iio 0 ∪ Set.Ioi 0)) :
  max 0 (Real.log (|x|)) ≥ 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)
  ∧ (max 0 (Real.log (|x|)) = 
      ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
      (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
      (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
      x = (Real.sqrt 5 - 1) / 2 ∨ 
      x = -(Real.sqrt 5 - 1) / 2 ∨ 
      x = (Real.sqrt 5 + 1) / 2 ∨ 
      x = -(Real.sqrt 5 + 1) / 2) :=
by
  sorry

end inequality_and_equality_condition_l232_23299


namespace total_cost_calculation_l232_23272

-- Definitions
def coffee_price : ℕ := 4
def cake_price : ℕ := 7
def ice_cream_price : ℕ := 3

def mell_coffee_qty : ℕ := 2
def mell_cake_qty : ℕ := 1
def friends_coffee_qty : ℕ := 2
def friends_cake_qty : ℕ := 1
def friends_ice_cream_qty : ℕ := 1

def total_coffee_qty : ℕ := 3 * mell_coffee_qty
def total_cake_qty : ℕ := 3 * mell_cake_qty
def total_ice_cream_qty : ℕ := 2 * friends_ice_cream_qty

def total_cost : ℕ := total_coffee_qty * coffee_price + total_cake_qty * cake_price + total_ice_cream_qty * ice_cream_price

-- Theorem Statement
theorem total_cost_calculation : total_cost = 51 := by
  sorry

end total_cost_calculation_l232_23272


namespace domain_of_v_l232_23248

noncomputable def v (x : ℝ) : ℝ := 1 / (x^(1/3))

theorem domain_of_v : {x : ℝ | ∃ y, y = v x} = {x : ℝ | x ≠ 0} := by
  sorry

end domain_of_v_l232_23248


namespace scientific_notation_50000000000_l232_23265

theorem scientific_notation_50000000000 :
  ∃ (a : ℝ) (n : ℤ), 50000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ (a = 5.0 ∨ a = 5) ∧ n = 10 :=
by
  sorry

end scientific_notation_50000000000_l232_23265


namespace infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l232_23269

theorem infinite_nat_sum_of_squares_and_cubes_not_sixth_powers :
  ∃ (N : ℕ) (k : ℕ), N > 0 ∧
  (N = 250 * 3^(6 * k)) ∧
  (∃ (x y : ℕ), N = x^2 + y^2) ∧
  (∃ (a b : ℕ), N = a^3 + b^3) ∧
  (∀ (u v : ℕ), N ≠ u^6 + v^6) :=
by
  sorry

end infinite_nat_sum_of_squares_and_cubes_not_sixth_powers_l232_23269


namespace mileage_per_gallon_l232_23250

-- Define the conditions
def miles_driven : ℝ := 100
def gallons_used : ℝ := 5

-- Define the question as a theorem to be proven
theorem mileage_per_gallon : (miles_driven / gallons_used) = 20 := by
  sorry

end mileage_per_gallon_l232_23250


namespace evaluate_expression_l232_23236

theorem evaluate_expression :
  1 + (3 / (4 + (5 / (6 + (7 / 8))))) = 85 / 52 := 
by
  sorry

end evaluate_expression_l232_23236


namespace michael_saves_more_l232_23290

-- Definitions for the conditions
def price_per_pair : ℝ := 50
def discount_a (price : ℝ) : ℝ := price + 0.6 * price
def discount_b (price : ℝ) : ℝ := 2 * price - 15

-- Statement to prove
theorem michael_saves_more (price : ℝ) (h : price = price_per_pair) : discount_b price - discount_a price = 5 :=
by
  sorry

end michael_saves_more_l232_23290


namespace Gwen_money_left_l232_23242

theorem Gwen_money_left (received spent : ℕ) (h_received : received = 14) (h_spent : spent = 8) : 
  received - spent = 6 := 
by 
  sorry

end Gwen_money_left_l232_23242


namespace minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l232_23221

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem minimum_value_when_a_is_1 : ∀ x : ℝ, ∃ m : ℝ, 
  (∀ y : ℝ, f y 1 ≥ f x 1) ∧ (f x 1 = m) :=
sorry

theorem range_of_a_given_fx_geq_0 : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 0 ≤ f x a) ↔ 1 ≤ a :=
sorry

end minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l232_23221


namespace six_box_four_div_three_eight_box_two_div_four_l232_23227

def fills_middle_zero (d : Nat) : Prop :=
  d < 3

def fills_last_zero (d : Nat) : Prop :=
  (80 + d) % 4 = 0

theorem six_box_four_div_three {d : Nat} : fills_middle_zero d → ((600 + d * 10 + 4) / 3) % 100 / 10 = 0 :=
  sorry

theorem eight_box_two_div_four {d : Nat} : fills_last_zero d → ((800 + d * 10 + 2) / 4) % 10 = 0 :=
  sorry

end six_box_four_div_three_eight_box_two_div_four_l232_23227


namespace sum_of_products_is_50_l232_23273

theorem sum_of_products_is_50
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a + b + c = 16) :
  a * b + b * c + a * c = 50 :=
by
  sorry

end sum_of_products_is_50_l232_23273


namespace spinner_probability_l232_23281

theorem spinner_probability :
  let p_A := (1 / 4)
  let p_B := (1 / 3)
  let p_C := (5 / 12)
  let p_D := 1 - (p_A + p_B + p_C)
  p_D = 0 :=
by
  sorry

end spinner_probability_l232_23281


namespace partition_no_infinite_arith_prog_l232_23229

theorem partition_no_infinite_arith_prog :
  ∃ (A B : Set ℕ), 
  (∀ n ∈ A, n ∈ B → False) ∧ 
  (∀ (a b : ℕ) (d : ℕ), (a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ (a - b) % d = 0) → False) ∧
  (∀ (a b : ℕ) (d : ℕ), (a ∈ B ∧ b ∈ B ∧ a ≠ b ∧ (a - b) % d = 0) → False) :=
sorry

end partition_no_infinite_arith_prog_l232_23229


namespace george_initial_amount_l232_23276

-- Definitions as per conditions
def cost_of_shirt : ℕ := 24
def cost_of_socks : ℕ := 11
def amount_left : ℕ := 65

-- Goal: Prove that the initial amount of money George had is 100
theorem george_initial_amount : (cost_of_shirt + cost_of_socks + amount_left) = 100 := 
by sorry

end george_initial_amount_l232_23276


namespace tree_planting_activity_l232_23270

theorem tree_planting_activity (x y : ℕ) 
  (h1 : y = 2 * x + 15)
  (h2 : x = y / 3 + 6) : 
  y = 81 ∧ x = 33 := 
by sorry

end tree_planting_activity_l232_23270


namespace valid_punching_settings_l232_23237

theorem valid_punching_settings :
  let total_patterns := 2^9
  let symmetric_patterns := 2^6
  total_patterns - symmetric_patterns = 448 :=
by
  sorry

end valid_punching_settings_l232_23237


namespace area_of_given_triangle_l232_23268

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_given_triangle :
  area_of_triangle (-2) 3 7 (-3) 4 6 = 31.5 :=
by
  sorry

end area_of_given_triangle_l232_23268


namespace find_coordinates_l232_23213

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def satisfiesCondition (A B P : Point) : Prop :=
  2 * (P.x - A.x) = (B.x - P.x) ∧ 2 * (P.y - A.y) = (B.y - P.y)

theorem find_coordinates (P : Point) (h : satisfiesCondition A B P) : 
  P = ⟨6, -9⟩ :=
  sorry

end find_coordinates_l232_23213


namespace positive_difference_l232_23247

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l232_23247


namespace smallest_number_in_sample_l232_23251

theorem smallest_number_in_sample :
  ∀ (N : ℕ) (k : ℕ) (n : ℕ), 
  0 < k → 
  N = 80 → 
  k = 5 →
  n = 42 →
  ∃ (a : ℕ), (0 ≤ a ∧ a < k) ∧
  42 = (N / k) * (42 / (N / k)) + a ∧
  ∀ (m : ℕ), (0 ≤ m ∧ m < k) → 
    (∀ (j : ℕ), (j = (N / k) * m + 10)) → 
    m = 0 → a = 10 := 
by
  sorry

end smallest_number_in_sample_l232_23251


namespace violet_prob_l232_23204

noncomputable def total_candies := 8 + 5 + 9 + 10 + 6

noncomputable def prob_green_first := (8 : ℚ) / total_candies
noncomputable def prob_yellow_second := (10 : ℚ) / (total_candies - 1)
noncomputable def prob_pink_third := (6 : ℚ) / (total_candies - 2)

noncomputable def combined_prob := prob_green_first * prob_yellow_second * prob_pink_third

theorem violet_prob :
  combined_prob = (20 : ℚ) / 2109 := by
    sorry

end violet_prob_l232_23204


namespace melissa_earnings_from_sales_l232_23258

noncomputable def commission_earned (coupe_price suv_price commission_rate : ℕ) : ℕ :=
  (coupe_price + suv_price) * commission_rate / 100

theorem melissa_earnings_from_sales : 
  commission_earned 30000 60000 2 = 1800 :=
by
  sorry

end melissa_earnings_from_sales_l232_23258


namespace set_D_cannot_form_triangle_l232_23206

-- Definition for triangle inequality theorem
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given lengths
def length_1 := 1
def length_2 := 2
def length_3 := 3

-- The proof problem statement
theorem set_D_cannot_form_triangle : ¬ triangle_inequality length_1 length_2 length_3 :=
  by sorry

end set_D_cannot_form_triangle_l232_23206


namespace wendy_candy_in_each_box_l232_23223

variable (x : ℕ)

def brother_candy : ℕ := 6
def total_candy : ℕ := 12
def wendy_boxes : ℕ := 2 * x

theorem wendy_candy_in_each_box :
  2 * x + brother_candy = total_candy → x = 3 :=
by
  intro h
  sorry

end wendy_candy_in_each_box_l232_23223


namespace radius_of_circle_l232_23222

theorem radius_of_circle (r : ℝ) (h : π * r^2 = 81 * π) : r = 9 :=
by
  sorry

end radius_of_circle_l232_23222


namespace count_squares_ending_in_4_l232_23220

theorem count_squares_ending_in_4 (n : ℕ) : 
  (∀ k : ℕ, (n^2 < 5000) → (n^2 % 10 = 4) → (k ≤ 70)) → 
  (∃ m : ℕ, m = 14) :=
by 
  sorry

end count_squares_ending_in_4_l232_23220


namespace sufficient_but_not_necessary_condition_l232_23296

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 2) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l232_23296


namespace fraction_meaningful_l232_23224

theorem fraction_meaningful (x : ℝ) : 2 * x - 1 ≠ 0 ↔ x ≠ 1 / 2 :=
by
  sorry

end fraction_meaningful_l232_23224


namespace number_of_possible_values_for_c_l232_23282

theorem number_of_possible_values_for_c : 
  (∃ c_values : Finset ℕ, (∀ c ∈ c_values, c ≥ 2 ∧ c^2 ≤ 256 ∧ 256 < c^3) 
  ∧ c_values.card = 10) :=
sorry

end number_of_possible_values_for_c_l232_23282


namespace jeans_cost_l232_23226

theorem jeans_cost (initial_money pizza_cost soda_cost quarter_value after_quarters : ℝ) (quarters_count: ℕ) :
  initial_money = 40 ->
  pizza_cost = 2.75 ->
  soda_cost = 1.50 ->
  quarter_value = 0.25 ->
  quarters_count = 97 ->
  after_quarters = quarters_count * quarter_value ->
  initial_money - (pizza_cost + soda_cost) - after_quarters = 11.50 :=
by
  intros h_initial h_pizza h_soda h_quarter_val h_quarters h_after_quarters
  sorry

end jeans_cost_l232_23226


namespace juice_spilled_l232_23294

def initial_amount := 1.0
def Youngin_drank := 0.1
def Narin_drank := Youngin_drank + 0.2
def remaining_amount := 0.3

theorem juice_spilled :
  initial_amount - (Youngin_drank + Narin_drank) - remaining_amount = 0.3 :=
by
  sorry

end juice_spilled_l232_23294


namespace set_union_intersection_l232_23246

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {2, 3, 4}

theorem set_union_intersection :
  (A ∩ B) ∪ C = {1, 2, 3, 4} := 
by
  sorry

end set_union_intersection_l232_23246


namespace plant_cost_and_max_green_lily_students_l232_23240

-- Given conditions
def two_green_lily_three_spider_plants_cost (x y : ℕ) : Prop :=
  2 * x + 3 * y = 36

def one_green_lily_two_spider_plants_cost (x y : ℕ) : Prop :=
  x + 2 * y = 21

def total_students := 48

def cost_constraint (x y m : ℕ) : Prop :=
  9 * m + 6 * (48 - m) ≤ 378

-- Prove that x = 9, y = 6 and m ≤ 30
theorem plant_cost_and_max_green_lily_students :
  ∃ x y m : ℕ, two_green_lily_three_spider_plants_cost x y ∧ 
               one_green_lily_two_spider_plants_cost x y ∧ 
               cost_constraint x y m ∧ 
               x = 9 ∧ y = 6 ∧ m ≤ 30 :=
by
  sorry

end plant_cost_and_max_green_lily_students_l232_23240


namespace graph_passes_through_point_l232_23267

theorem graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
    ∀ x y : ℝ, (y = a^(x-2) + 2) → (x = 2) → (y = 3) :=
by
    intros x y hxy hx
    rw [hx] at hxy
    simp at hxy
    sorry

end graph_passes_through_point_l232_23267


namespace pairs_count_l232_23293

noncomputable def count_pairs (n : ℕ) : ℕ :=
  3^n

theorem pairs_count (A : Finset ℕ) (h : A.card = n) :
  ∃ f : Finset ℕ × Finset ℕ → Finset ℕ, ∀ B C, (B ≠ ∅ ∧ B ⊆ C ∧ C ⊆ A) → (f (B, C)).card = count_pairs n :=
sorry

end pairs_count_l232_23293


namespace jordan_rectangle_width_l232_23202

theorem jordan_rectangle_width (length_carol width_carol length_jordan width_jordan : ℝ)
  (h1: length_carol = 15) (h2: width_carol = 20) (h3: length_jordan = 6)
  (area_equal: length_carol * width_carol = length_jordan * width_jordan) :
  width_jordan = 50 :=
by
  sorry

end jordan_rectangle_width_l232_23202


namespace system_of_equations_solution_l232_23232

theorem system_of_equations_solution (x y : ℝ) (h1 : 4 * x + 3 * y = 11) (h2 : 4 * x - 3 * y = 5) :
  x = 2 ∧ y = 1 :=
by {
  sorry
}

end system_of_equations_solution_l232_23232


namespace symmetric_origin_coordinates_l232_23254

-- Given the coordinates (m, n) of point P
variables (m n : ℝ)
-- Define point P
def P := (m, n)

-- Define point P' which is symmetric to P with respect to the origin O
def P'_symmetric_origin : ℝ × ℝ := (-m, -n)

-- Prove that the coordinates of P' are (-m, -n)
theorem symmetric_origin_coordinates :
  P'_symmetric_origin m n = (-m, -n) :=
by
  -- Proof content goes here but we're skipping it with sorry
  sorry

end symmetric_origin_coordinates_l232_23254


namespace oscar_bus_ride_length_l232_23292

/-- Oscar's bus ride to school is some distance, and Charlie's bus ride is 0.25 mile.
Oscar's bus ride is 0.5 mile longer than Charlie's. Prove that Oscar's bus ride is 0.75 mile. -/
theorem oscar_bus_ride_length (charlie_ride : ℝ) (h1 : charlie_ride = 0.25) 
  (oscar_ride : ℝ) (h2 : oscar_ride = charlie_ride + 0.5) : oscar_ride = 0.75 :=
by sorry

end oscar_bus_ride_length_l232_23292


namespace cos_double_angle_l232_23238

theorem cos_double_angle (α β : ℝ) (h1 : Real.sin (α - β) = 1 / 3) (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_double_angle_l232_23238


namespace example_problem_l232_23283

-- Define vectors a and b with the given conditions
def a (k : ℝ) : ℝ × ℝ := (2, k)
def b : ℝ × ℝ := (6, 4)

-- Define the condition that vectors are perpendicular
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Calculate the sum of two vectors
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Check if a vector is collinear
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- The main theorem with the given conditions
theorem example_problem (k : ℝ) (hk : perpendicular (a k) b) :
  collinear (vector_add (a k) b) (-16, -2) :=
by
  sorry

end example_problem_l232_23283


namespace arithmetic_sequence_proof_l232_23211

open Nat

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  a 1 = 2 ∧ (a 2) ^ 2 = (a 1) * (a 5)

def general_formula (a : ℕ → ℤ) (d : ℤ) : Prop :=
  (d = 0 ∧ ∀ n, a n = 2) ∨ (d = 4 ∧ ∀ n, a n = 4 * n - 2)

def sum_seq (a : ℕ → ℤ) (S_n : ℕ → ℤ) (d : ℤ) : Prop :=
  ((∀ n, a n = 2) ∧ (∀ n, S_n n = 2 * n)) ∨ ((∀ n, a n = 4 * n - 2) ∧ (∀ n, S_n n = 4 * n^2 - 2 * n))

theorem arithmetic_sequence_proof :
  ∃ a : ℕ → ℤ, ∃ d : ℤ, arithmetic_seq a d ∧ general_formula a d ∧ ∃ S_n : ℕ → ℤ, sum_seq a S_n d := by
  sorry

end arithmetic_sequence_proof_l232_23211


namespace minor_premise_l232_23287

variables (A B C : Prop)

theorem minor_premise (hA : A) (hB : B) (hC : C) : B := 
by
  exact hB

end minor_premise_l232_23287


namespace expression_equivalence_l232_23252

theorem expression_equivalence :
  (4 + 3) * (4^2 + 3^2) * (4^4 + 3^4) * (4^8 + 3^8) * (4^16 + 3^16) * (4^32 + 3^32) * (4^64 + 3^64) = 3^128 - 4^128 :=
by
  sorry

end expression_equivalence_l232_23252


namespace evaluate_expression_l232_23216

def S (n : ℕ) : ℤ :=
  if n % 2 = 1 then (n + 1) / 2
  else -n / 2

theorem evaluate_expression : S 19 * S 31 + S 48 = 136 :=
by sorry

end evaluate_expression_l232_23216


namespace find_value_l232_23255

theorem find_value (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x - 3 = 12 :=
by
  sorry

end find_value_l232_23255


namespace sequence_general_formula_l232_23239

-- Define conditions: The sum of the first n terms of the sequence is Sn = an - 3
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
axiom condition (n : ℕ) : S n = a n - 3

-- Define the main theorem to prove
theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : a n = 2 * 3 ^ n :=
sorry

end sequence_general_formula_l232_23239


namespace linear_function_result_l232_23233

variable {R : Type*} [LinearOrderedField R]

noncomputable def linear_function (g : R → R) : Prop :=
  ∃ (a b : R), ∀ x, g x = a * x + b

theorem linear_function_result (g : R → R) (h_lin : linear_function g) (h : g 5 - g 1 = 16) : g 13 - g 1 = 48 :=
  by
  sorry

end linear_function_result_l232_23233


namespace find_expression_for_a_n_l232_23203

noncomputable def seq (n : ℕ) : ℕ := sorry
def sumFirstN (n : ℕ) : ℕ := sorry

theorem find_expression_for_a_n (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h_pos : ∀ n, 0 < a n)
  (h_arith_seq : ∀ n, S n + 1 = 2 * a n) :
  ∀ n, a n = 2^(n-1) :=
sorry

end find_expression_for_a_n_l232_23203
