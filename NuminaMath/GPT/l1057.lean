import Mathlib

namespace range_of_a_l1057_105775

variables (a : ℝ)

def prop_p : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 16 > 0
def prop_q : Prop := (2 * a - 2)^2 - 8 * (3 * a - 7) ≥ 0
def combined : Prop := prop_p a ∧ prop_q a

theorem range_of_a (a : ℝ) : combined a ↔ -4 < a ∧ a ≤ 3 :=
by
  sorry

end range_of_a_l1057_105775


namespace factory_car_production_l1057_105712

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l1057_105712


namespace bert_bought_300_stamps_l1057_105711

theorem bert_bought_300_stamps (x : ℝ) 
(H1 : x / 2 + x = 450) : x = 300 :=
by
  sorry

end bert_bought_300_stamps_l1057_105711


namespace stamps_per_page_l1057_105736

def a : ℕ := 924
def b : ℕ := 1386
def c : ℕ := 1848

theorem stamps_per_page : gcd (gcd a b) c = 462 :=
sorry

end stamps_per_page_l1057_105736


namespace H2CO3_formation_l1057_105752

-- Define the given conditions
def one_to_one_reaction (a b : ℕ) := a = b

-- Define the reaction
theorem H2CO3_formation (m_CO2 m_H2O : ℕ) 
  (h : one_to_one_reaction m_CO2 m_H2O) : 
  m_CO2 = 2 → m_H2O = 2 → m_CO2 = 2 ∧ m_H2O = 2 := 
by 
  intros h1 h2
  exact ⟨h1, h2⟩

end H2CO3_formation_l1057_105752


namespace jill_water_jars_l1057_105723

theorem jill_water_jars (x : ℕ) (h : x * (1 / 4 + 1 / 2 + 1) = 28) : 3 * x = 48 :=
by
  sorry

end jill_water_jars_l1057_105723


namespace intersection_of_A_and_B_l1057_105757

noncomputable def A : Set ℝ := { x | -1 < x - 3 ∧ x - 3 ≤ 2 }
noncomputable def B : Set ℝ := { x | 3 ≤ x ∧ x < 6 }

theorem intersection_of_A_and_B : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end intersection_of_A_and_B_l1057_105757


namespace sum_non_solutions_is_neg21_l1057_105798

noncomputable def sum_of_non_solutions (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2) : ℝ :=
  -21

theorem sum_non_solutions_is_neg21 (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) = 2) : 
  ∃! (x1 x2 : ℝ), ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2 → x = x1 ∨ x = x2 ∧ x1 + x2 = -21 :=
sorry

end sum_non_solutions_is_neg21_l1057_105798


namespace any_nat_representation_as_fraction_l1057_105794

theorem any_nat_representation_as_fraction (n : ℕ) : 
    ∃ x y : ℕ, y ≠ 0 ∧ (x^3 : ℚ) / (y^4 : ℚ) = n := by
  sorry

end any_nat_representation_as_fraction_l1057_105794


namespace candy_problem_l1057_105747

theorem candy_problem (N S a : ℕ) (h1 : a = S - a - 7) (h2 : a > 1) : S = 21 := 
sorry

end candy_problem_l1057_105747


namespace model_tower_height_l1057_105739

theorem model_tower_height (h_real : ℝ) (vol_real : ℝ) (vol_model : ℝ) 
  (h_real_eq : h_real = 60) (vol_real_eq : vol_real = 150000) (vol_model_eq : vol_model = 0.15) :
  (h_real * (vol_model / vol_real)^(1/3) = 0.6) :=
by
  sorry

end model_tower_height_l1057_105739


namespace dihedral_angle_sum_bounds_l1057_105749

variable (α β γ : ℝ)

/-- The sum of the internal dihedral angles of a trihedral angle is greater than 180 degrees and less than 540 degrees. -/
theorem dihedral_angle_sum_bounds (hα: α < 180) (hβ: β < 180) (hγ: γ < 180) : 180 < α + β + γ ∧ α + β + γ < 540 :=
by
  sorry

end dihedral_angle_sum_bounds_l1057_105749


namespace intersection_A_B_union_A_B_subset_C_A_l1057_105760

def set_A : Set ℝ := { x | x^2 - x - 2 > 0 }
def set_B : Set ℝ := { x | 3 - abs x ≥ 0 }
def set_C (p : ℝ) : Set ℝ := { x | 4 * x + p < 0 }

theorem intersection_A_B : set_A ∩ set_B = { x | (-3 ≤ x ∧ x < -1) ∨ (2 < x ∧ x ≤ 3) } :=
sorry

theorem union_A_B : set_A ∪ set_B = Set.univ :=
sorry

theorem subset_C_A (p : ℝ) : set_C p ⊆ set_A → p ≥ 4 :=
sorry

end intersection_A_B_union_A_B_subset_C_A_l1057_105760


namespace problem_l1057_105734

theorem problem (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) + Real.cos α^2 = 16 / 5 :=
sorry

end problem_l1057_105734


namespace base_n_representation_of_b_l1057_105735

theorem base_n_representation_of_b (n a b : ℕ) (hn : n > 8) 
  (h_n_solution : ∃ m, m ≠ n ∧ n * m = b ∧ n + m = a) 
  (h_a_base_n : 1 * n + 8 = a) :
  (b = 8 * n) :=
by
  sorry

end base_n_representation_of_b_l1057_105735


namespace global_school_math_students_l1057_105713

theorem global_school_math_students (n : ℕ) (h1 : n < 600) (h2 : n % 28 = 27) (h3 : n % 26 = 20) : n = 615 :=
by
  -- skip the proof
  sorry

end global_school_math_students_l1057_105713


namespace total_guests_at_least_one_reunion_l1057_105780

-- Definitions used in conditions
def attendeesOates := 42
def attendeesYellow := 65
def attendeesBoth := 7

-- Definition of the total number of guests attending at least one of the reunions
def totalGuests := attendeesOates + attendeesYellow - attendeesBoth

-- Theorem stating that the total number of guests is equal to 100
theorem total_guests_at_least_one_reunion : totalGuests = 100 :=
by
  -- skipping the proof with sorry
  sorry

end total_guests_at_least_one_reunion_l1057_105780


namespace sum_mod_9_l1057_105786

theorem sum_mod_9 : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 :=
by sorry

end sum_mod_9_l1057_105786


namespace fraction_of_men_collected_dues_l1057_105790

theorem fraction_of_men_collected_dues
  (M W : ℕ)
  (x : ℚ)
  (h1 : 45 * x * M + 5 * W = 17760)
  (h2 : M + W = 3552)
  (h3 : 1 / 12 * W = W / 12) :
  x = 1 / 9 :=
by
  -- Proof steps would go here
  sorry

end fraction_of_men_collected_dues_l1057_105790


namespace expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l1057_105763

section
variables (a b : ℚ)

theorem expansion_of_a_plus_b_pow_4 :
  (a + b) ^ 4 = a ^ 4 + 4 * a ^ 3 * b + 6 * a ^ 2 * b ^ 2 + 4 * a * b ^ 3 + b ^ 4 :=
sorry

theorem expansion_of_a_plus_b_pow_5 :
  (a + b) ^ 5 = a ^ 5 + 5 * a ^ 4 * b + 10 * a ^ 3 * b ^ 2 + 10 * a ^ 2 * b ^ 3 + 5 * a * b ^ 4 + b ^ 5 :=
sorry

theorem computation_of_formula :
  2^4 + 4*2^3*(-1/3) + 6*2^2*(-1/3)^2 + 4*2*(-1/3)^3 + (-1/3)^4 = 625 / 81 :=
sorry
end

end expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l1057_105763


namespace determine_angle_A_l1057_105703

-- Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
def sin_rule_condition (a b c A B C : ℝ) : Prop :=
  (a + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C

-- The proof statement
theorem determine_angle_A (a b c A B C : ℝ) (h : sin_rule_condition a b c A B C) : A = π / 3 :=
  sorry

end determine_angle_A_l1057_105703


namespace percentage_increase_l1057_105748

theorem percentage_increase (C S : ℝ) (h1 : S = 4.2 * C) 
  (h2 : ∃ X : ℝ, (S - (C + (X / 100) * C) = (2 / 3) * S)) : 
  ∃ X : ℝ, (C + (X / 100) * C - C)/(C) = 40 / 100 := 
by
  sorry

end percentage_increase_l1057_105748


namespace find_distance_d_l1057_105716

theorem find_distance_d (d : ℝ) (XR : ℝ) (YP : ℝ) (XZ : ℝ) (YZ : ℝ) (XY : ℝ) (h1 : XR = 3) (h2 : YP = 12) (h3 : XZ = 3 + d) (h4 : YZ = 12 + d) (h5 : XY = 15) (h6 : (XZ)^2 + (XY)^2 = (YZ)^2) : d = 5 :=
sorry

end find_distance_d_l1057_105716


namespace smallest_possible_sum_l1057_105772

theorem smallest_possible_sum :
  ∃ (B : ℕ) (c : ℕ), B + c = 34 ∧ 
    (B ≥ 0 ∧ B < 5) ∧ 
    (c > 7) ∧ 
    (31 * B = 4 * c + 4) := 
by
  sorry

end smallest_possible_sum_l1057_105772


namespace reimbursement_proof_l1057_105717

-- Define the rates
def rate_industrial_weekday : ℝ := 0.36
def rate_commercial_weekday : ℝ := 0.42
def rate_weekend : ℝ := 0.45

-- Define the distances for each day
def distance_monday : ℝ := 18
def distance_tuesday : ℝ := 26
def distance_wednesday : ℝ := 20
def distance_thursday : ℝ := 20
def distance_friday : ℝ := 16
def distance_saturday : ℝ := 12

-- Calculate the reimbursement for each day
def reimbursement_monday : ℝ := distance_monday * rate_industrial_weekday
def reimbursement_tuesday : ℝ := distance_tuesday * rate_commercial_weekday
def reimbursement_wednesday : ℝ := distance_wednesday * rate_industrial_weekday
def reimbursement_thursday : ℝ := distance_thursday * rate_commercial_weekday
def reimbursement_friday : ℝ := distance_friday * rate_industrial_weekday
def reimbursement_saturday : ℝ := distance_saturday * rate_weekend

-- Calculate the total reimbursement
def total_reimbursement : ℝ :=
  reimbursement_monday + reimbursement_tuesday + reimbursement_wednesday +
  reimbursement_thursday + reimbursement_friday + reimbursement_saturday

-- State the theorem to be proven
theorem reimbursement_proof : total_reimbursement = 44.16 := by
  sorry

end reimbursement_proof_l1057_105717


namespace power_function_increasing_l1057_105753

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^m > 0) → m = 2 := 
by 
  sorry

end power_function_increasing_l1057_105753


namespace find_x_squared_plus_inverse_squared_l1057_105783

theorem find_x_squared_plus_inverse_squared (x : ℝ) 
(h : x^4 + (1 / x^4) = 2398) : 
  x^2 + (1 / x^2) = 20 * Real.sqrt 6 :=
sorry

end find_x_squared_plus_inverse_squared_l1057_105783


namespace product_of_two_numbers_l1057_105743

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := 
by {
  sorry
}

end product_of_two_numbers_l1057_105743


namespace probability_purple_or_orange_face_l1057_105707

theorem probability_purple_or_orange_face 
  (total_faces : ℕ) (green_faces : ℕ) (purple_faces : ℕ) (orange_faces : ℕ) 
  (h_total : total_faces = 10) 
  (h_green : green_faces = 5) 
  (h_purple : purple_faces = 3) 
  (h_orange : orange_faces = 2) :
  (purple_faces + orange_faces) / total_faces = 1 / 2 :=
by 
  sorry

end probability_purple_or_orange_face_l1057_105707


namespace bacterium_descendants_in_range_l1057_105726

theorem bacterium_descendants_in_range (total_bacteria : ℕ) (initial : ℕ) 
  (h_total : total_bacteria = 1000) (h_initial : initial = total_bacteria) 
  (descendants : ℕ → ℕ)
  (h_step : ∀ k, descendants (k+1) ≤ descendants k / 2) :
  ∃ k, 334 ≤ descendants k ∧ descendants k ≤ 667 :=
by
  sorry

end bacterium_descendants_in_range_l1057_105726


namespace definite_integral_eval_l1057_105793

theorem definite_integral_eval :
  ∫ x in (1:ℝ)..(3:ℝ), (2 * x - 1 / x ^ 2) = 22 / 3 :=
by
  sorry

end definite_integral_eval_l1057_105793


namespace r_squared_is_one_l1057_105781

theorem r_squared_is_one (h : ∀ (x : ℝ), ∃ (y : ℝ), ∃ (m : ℝ) (b : ℝ), m ≠ 0 ∧ y = m * x + b) : R_squared = 1 :=
sorry

end r_squared_is_one_l1057_105781


namespace time_per_mask_after_first_hour_l1057_105730

-- Define the conditions as given in the problem
def rate_in_first_hour := 1 / 4 -- Manolo makes one face-mask every four minutes
def total_face_masks := 45 -- Manolo makes 45 face-masks in four hours
def first_hour_duration := 60 -- The duration of the first hour in minutes
def total_duration := 4 * 60 -- The total duration in minutes (4 hours)

-- Define the number of face-masks made in the first hour
def face_masks_first_hour := first_hour_duration / 4 -- 60 minutes / 4 minutes per face-mask = 15 face-masks

-- Calculate the number of face-masks made in the remaining time
def face_masks_remaining_hours := total_face_masks - face_masks_first_hour -- 45 - 15 = 30 face-masks

-- Define the duration of the remaining hours
def remaining_duration := total_duration - first_hour_duration -- 180 minutes (3 hours)

-- The target is to prove that the rate after the first hour is 6 minutes per face-mask
theorem time_per_mask_after_first_hour : remaining_duration / face_masks_remaining_hours = 6 := by
  sorry

end time_per_mask_after_first_hour_l1057_105730


namespace B_pow_2021_eq_B_l1057_105778

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1 / 2, 0, -Real.sqrt 3 / 2],
  ![0, -1, 0],
  ![Real.sqrt 3 / 2, 0, 1 / 2]
]

theorem B_pow_2021_eq_B : B ^ 2021 = B := 
by sorry

end B_pow_2021_eq_B_l1057_105778


namespace mans_rate_in_still_water_l1057_105782

theorem mans_rate_in_still_water : 
  ∀ (V_m V_s : ℝ), 
  V_m + V_s = 16 → 
  V_m - V_s = 4 → 
  V_m = 10 :=
by
  intros V_m V_s h1 h2
  sorry

end mans_rate_in_still_water_l1057_105782


namespace solution_l1057_105797

noncomputable def problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : Prop :=
  x + y ≥ 9

theorem solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1/x + 4/y = 1) : problem x y h1 h2 h3 :=
  sorry

end solution_l1057_105797


namespace remaining_files_l1057_105742

def initial_music_files : ℕ := 16
def initial_video_files : ℕ := 48
def deleted_files : ℕ := 30

theorem remaining_files :
  initial_music_files + initial_video_files - deleted_files = 34 := 
by
  sorry

end remaining_files_l1057_105742


namespace calculate_minus_one_minus_two_l1057_105774

theorem calculate_minus_one_minus_two : -1 - 2 = -3 := by
  sorry

end calculate_minus_one_minus_two_l1057_105774


namespace new_elephants_entry_rate_l1057_105733

-- Definitions
def initial_elephants := 30000
def exodus_rate := 2880
def exodus_duration := 4
def final_elephants := 28980
def new_elephants_duration := 7

-- Prove that the rate of new elephants entering the park is 1500 elephants per hour
theorem new_elephants_entry_rate :
  let elephants_left_after_exodus := initial_elephants - exodus_rate * exodus_duration
  let new_elephants := final_elephants - elephants_left_after_exodus
  let new_entry_rate := new_elephants / new_elephants_duration
  new_entry_rate = 1500 :=
by
  sorry

end new_elephants_entry_rate_l1057_105733


namespace cost_of_dowels_l1057_105702

variable (V S : ℝ)

theorem cost_of_dowels 
  (hV : V = 7)
  (h_eq : 0.85 * (V + S) = V + 0.5 * S) :
  S = 3 :=
by
  sorry

end cost_of_dowels_l1057_105702


namespace pairs_sum_gcd_l1057_105738

theorem pairs_sum_gcd (a b : ℕ) (h_sum : a + b = 288) (h_gcd : Int.gcd a b = 36) :
  (a = 36 ∧ b = 252) ∨ (a = 252 ∧ b = 36) ∨ (a = 108 ∧ b = 180) ∨ (a = 180 ∧ b = 108) :=
by {
   sorry
}

end pairs_sum_gcd_l1057_105738


namespace time_to_get_to_lawrence_house_l1057_105727

def distance : ℝ := 12
def speed : ℝ := 2

theorem time_to_get_to_lawrence_house : (distance / speed) = 6 :=
by
  sorry

end time_to_get_to_lawrence_house_l1057_105727


namespace roots_of_Q_are_fifth_powers_of_roots_of_P_l1057_105796

def P (x : ℝ) : ℝ := x^3 - 3 * x + 1

noncomputable def Q (y : ℝ) : ℝ := y^3 + 15 * y^2 - 198 * y + 1

theorem roots_of_Q_are_fifth_powers_of_roots_of_P : 
  ∀ α β γ : ℝ, (P α = 0) ∧ (P β = 0) ∧ (P γ = 0) →
  (Q (α^5) = 0) ∧ (Q (β^5) = 0) ∧ (Q (γ^5) = 0) := 
by 
  intros α β γ h
  sorry

end roots_of_Q_are_fifth_powers_of_roots_of_P_l1057_105796


namespace inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l1057_105725

-- Definitions for the conditions
def inside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 < r^2 ∧ (M.1 ≠ 0 ∨ M.2 ≠ 0)

def on_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 = r^2

def outside_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1^2 + M.2^2 > r^2

def line_l_intersects_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 < r^2 ∨ M.1 * M.1 + M.2 * M.2 = r^2

def line_l_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 = r^2

def line_l_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) : Prop :=
  M.1 * M.1 + M.2 * M.2 > r^2

-- Propositions
theorem inside_circle_implies_line_intersects_circle (M : ℝ × ℝ) (r : ℝ) : 
  inside_circle M r → line_l_intersects_circle M r := 
sorry

theorem on_circle_implies_line_tangent_to_circle (M : ℝ × ℝ) (r : ℝ) :
  on_circle M r → line_l_tangent_to_circle M r :=
sorry

theorem outside_circle_implies_line_does_not_intersect_circle (M : ℝ × ℝ) (r : ℝ) :
  outside_circle M r → line_l_does_not_intersect_circle M r :=
sorry

end inside_circle_implies_line_intersects_circle_on_circle_implies_line_tangent_to_circle_outside_circle_implies_line_does_not_intersect_circle_l1057_105725


namespace quarters_for_soda_l1057_105792

def quarters_for_chips := 4
def total_dollars := 4

theorem quarters_for_soda :
  (total_dollars * 4) - quarters_for_chips = 12 :=
by
  sorry

end quarters_for_soda_l1057_105792


namespace number_of_cookies_l1057_105769

def candy : ℕ := 63
def brownies : ℕ := 21
def people : ℕ := 7
def dessert_per_person : ℕ := 18

theorem number_of_cookies : 
  (people * dessert_per_person) - (candy + brownies) = 42 := 
by
  sorry

end number_of_cookies_l1057_105769


namespace monday_dressing_time_l1057_105771

theorem monday_dressing_time 
  (Tuesday_time Wednesday_time Thursday_time Friday_time Old_average_time : ℕ)
  (H_tuesday : Tuesday_time = 4)
  (H_wednesday : Wednesday_time = 3)
  (H_thursday : Thursday_time = 4)
  (H_friday : Friday_time = 2)
  (H_average : Old_average_time = 3) :
  ∃ Monday_time : ℕ, Monday_time = 2 :=
by
  let Total_time_5_days := Old_average_time * 5
  let Total_time := 4 + 3 + 4 + 2
  let Monday_time := Total_time_5_days - Total_time
  exact ⟨Monday_time, sorry⟩

end monday_dressing_time_l1057_105771


namespace students_qualifying_percentage_l1057_105788

theorem students_qualifying_percentage (N B G : ℕ) (boy_percent : ℝ) (girl_percent : ℝ) :
  N = 400 →
  G = 100 →
  B = N - G →
  boy_percent = 0.60 →
  girl_percent = 0.80 →
  (boy_percent * B + girl_percent * G) / N * 100 = 65 :=
by
  intros hN hG hB hBoy hGirl
  simp [hN, hG, hB, hBoy, hGirl]
  sorry

end students_qualifying_percentage_l1057_105788


namespace age_difference_l1057_105765

variable (A B C : ℕ)

theorem age_difference : A + B = B + C + 11 → A - C = 11 := by
  sorry

end age_difference_l1057_105765


namespace part_one_part_two_l1057_105787

namespace ProofProblem

def setA (a : ℝ) := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def setB := {x : ℝ | 0 < x ∧ x < 1}

theorem part_one (a : ℝ) (h : a = 1/2) : 
  setA a ∩ setB = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

theorem part_two (a : ℝ) (h_subset : setB ⊆ setA a) : 
  0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end ProofProblem

end part_one_part_two_l1057_105787


namespace side_length_sum_area_l1057_105799

theorem side_length_sum_area (a b c d : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 12) :
  d = 13 :=
by
  -- Proof is not required
  sorry

end side_length_sum_area_l1057_105799


namespace dog_food_bags_count_l1057_105766

-- Define the constants based on the problem statement
def CatFoodBags := 327
def DogFoodMore := 273

-- Define the total number of dog food bags based on the given conditions
def DogFoodBags : ℤ := CatFoodBags + DogFoodMore

-- State the theorem we want to prove
theorem dog_food_bags_count : DogFoodBags = 600 := by
  sorry

end dog_food_bags_count_l1057_105766


namespace quarters_initially_l1057_105737

theorem quarters_initially (quarters_borrowed : ℕ) (quarters_now : ℕ) (initial_quarters : ℕ) 
   (h1 : quarters_borrowed = 3) (h2 : quarters_now = 5) :
   initial_quarters = quarters_now + quarters_borrowed :=
by
  -- Proof goes here
  sorry

end quarters_initially_l1057_105737


namespace m_greater_than_p_l1057_105756

theorem m_greater_than_p (p m n : ℕ) (prime_p : Prime p) (pos_m : 0 < m) (pos_n : 0 < n)
    (eq : p^2 + m^2 = n^2) : m > p := 
by 
  sorry

end m_greater_than_p_l1057_105756


namespace megan_markers_final_count_l1057_105700

theorem megan_markers_final_count :
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  final_markers = 582 :=
by
  let initial_markers := 217
  let robert_gave := 109
  let sarah_took := 35
  let teacher_multiplier := 3
  let final_markers := (initial_markers + robert_gave - sarah_took) * (1 + teacher_multiplier) / 2
  have h : final_markers = 582 := sorry
  exact h

end megan_markers_final_count_l1057_105700


namespace find_third_number_l1057_105720

theorem find_third_number (x : ℝ) 
  (h : (20 + 40 + x) / 3 = (10 + 50 + 45) / 3 + 5) : x = 60 :=
sorry

end find_third_number_l1057_105720


namespace emily_catch_catfish_l1057_105762

-- Definitions based on given conditions
def num_trout : ℕ := 4
def num_bluegills : ℕ := 5
def weight_trout : ℕ := 2
def weight_catfish : ℚ := 1.5
def weight_bluegill : ℚ := 2.5
def total_fish_weight : ℚ := 25

-- Lean statement to prove the number of catfish
theorem emily_catch_catfish : ∃ (num_catfish : ℕ), 
  num_catfish * weight_catfish = total_fish_weight - (num_trout * weight_trout + num_bluegills * weight_bluegill) ∧
  num_catfish = 3 := by
  sorry

end emily_catch_catfish_l1057_105762


namespace gcd_of_720_120_168_is_24_l1057_105710

theorem gcd_of_720_120_168_is_24 : Int.gcd (Int.gcd 720 120) 168 = 24 := 
by sorry

end gcd_of_720_120_168_is_24_l1057_105710


namespace greatest_integer_third_side_l1057_105773

/-- 
 Given a triangle with sides a and b, where a = 5 and b = 10, 
 prove that the greatest integer value for the third side c, 
 satisfying the Triangle Inequality, is 14.
-/
theorem greatest_integer_third_side (x : ℝ) (h₁ : 5 < x) (h₂ : x < 15) : x ≤ 14 :=
sorry

end greatest_integer_third_side_l1057_105773


namespace time_period_principal_1000_amount_1120_interest_5_l1057_105759

-- Definitions based on the conditions
def principal : ℝ := 1000
def amount : ℝ := 1120
def interest_rate : ℝ := 0.05

-- Lean 4 statement asserting the time period
theorem time_period_principal_1000_amount_1120_interest_5
  (P : ℝ) (A : ℝ) (r : ℝ) (T : ℝ) 
  (hP : P = principal)
  (hA : A = amount)
  (hr : r = interest_rate) :
  (A - P) * 100 / (P * r * 100) = 2.4 :=
by 
  -- The proof is filled in by 'sorry'
  sorry

end time_period_principal_1000_amount_1120_interest_5_l1057_105759


namespace prime_square_remainder_l1057_105741

theorem prime_square_remainder (p : ℕ) (hp : Nat.Prime p) (h5 : p > 5) : 
  ∃! r : ℕ, r < 180 ∧ (p^2 ≡ r [MOD 180]) := 
by
  sorry

end prime_square_remainder_l1057_105741


namespace problem_l1057_105729

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x + a * Real.cos x

theorem problem (a : ℝ) (h₀ : a < 0) (h₁ : ∀ x : ℝ, f x a ≤ 2) : f (π / 6) a = -1 :=
by {
  sorry
}

end problem_l1057_105729


namespace proposition_3_correct_l1057_105791

open Real

def is_obtuse (A B C : ℝ) : Prop :=
  A + B + C = π ∧ (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)

theorem proposition_3_correct (A B C : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : 0 < C) (h₃ : A + B + C = π)
  (h : sin A ^ 2 + sin B ^ 2 + cos C ^ 2 < 1) : is_obtuse A B C :=
by
  sorry

end proposition_3_correct_l1057_105791


namespace area_of_kite_l1057_105718

theorem area_of_kite (A B C D : ℝ × ℝ) (hA : A = (2, 3)) (hB : B = (6, 7)) (hC : C = (10, 3)) (hD : D = (6, 0)) : 
  let base := (C.1 - A.1)
  let height := (B.2 - D.2)
  let area := 2 * (1 / 2 * base * height)
  area = 56 := 
by
  sorry

end area_of_kite_l1057_105718


namespace converse_proposition_inverse_proposition_contrapositive_proposition_l1057_105795

theorem converse_proposition (x y : ℝ) : (xy = 0 → x^2 + y^2 = 0) = false :=
sorry

theorem inverse_proposition (x y : ℝ) : (x^2 + y^2 ≠ 0 → xy ≠ 0) = false :=
sorry

theorem contrapositive_proposition (x y : ℝ) : (xy ≠ 0 → x^2 + y^2 ≠ 0) = true :=
sorry

end converse_proposition_inverse_proposition_contrapositive_proposition_l1057_105795


namespace simple_interest_two_years_l1057_105770
-- Import the necessary Lean library for mathematical concepts

-- Define the problem conditions and the proof statement
theorem simple_interest_two_years (P r t : ℝ) (CI SI : ℝ)
  (hP : P = 17000) (ht : t = 2) (hCI : CI = 11730) : SI = 5100 :=
by
  -- Principal (P), Rate (r), and Time (t) definitions
  let P := 17000
  let t := 2

  -- Given Compound Interest (CI)
  let CI := 11730

  -- Correct value for Simple Interest (SI) that we need to prove
  let SI := 5100

  -- Formalize the assumptions
  have h1 : P = 17000 := rfl
  have h2 : t = 2 := rfl
  have h3 : CI = 11730 := rfl

  -- Crucial parts of the problem are used here
  sorry  -- This is a placeholder for the actual proof steps

end simple_interest_two_years_l1057_105770


namespace simplify_expression_l1057_105721

theorem simplify_expression : 8^5 + 8^5 + 8^5 + 8^5 = 8^(17/3) :=
by
  -- Proof will be completed here
  sorry

end simplify_expression_l1057_105721


namespace stephanie_oranges_l1057_105758

theorem stephanie_oranges (times_at_store : ℕ) (oranges_per_time : ℕ) (total_oranges : ℕ) 
  (h1 : times_at_store = 8) (h2 : oranges_per_time = 2) :
  total_oranges = 16 :=
by
  sorry

end stephanie_oranges_l1057_105758


namespace cart_max_speed_l1057_105715

noncomputable def maximum_speed (a R : ℝ) : ℝ :=
  (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4)

theorem cart_max_speed (a R v : ℝ) (h : v = maximum_speed a R) : 
  v = (16 * a^2 * R^2 * Real.pi^2 / (1 + 16 * Real.pi^2)) ^ (1/4) :=
by
  -- Proof is omitted
  sorry

end cart_max_speed_l1057_105715


namespace reduced_price_is_60_l1057_105779

variable (P R: ℝ) -- Declare the variables P and R as real numbers.

-- Define the conditions as hypotheses.
axiom h1 : R = 0.7 * P
axiom h2 : 1800 / R = 1800 / P + 9

-- The theorem stating the problem to prove.
theorem reduced_price_is_60 (P R : ℝ) (h1 : R = 0.7 * P) (h2 : 1800 / R = 1800 / P + 9) : R = 60 :=
by sorry

end reduced_price_is_60_l1057_105779


namespace exterior_angle_regular_octagon_l1057_105732

theorem exterior_angle_regular_octagon : 
  ∀ {θ : ℝ}, 
  (8 - 2) * 180 / 8 = θ →
  180 - θ = 45 := 
by 
  intro θ hθ
  sorry

end exterior_angle_regular_octagon_l1057_105732


namespace eggs_in_each_basket_l1057_105755

theorem eggs_in_each_basket :
  ∃ (n : ℕ), (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧
    (∀ m : ℕ, (m ∣ 30) ∧ (m ∣ 45) ∧ (m ≥ 5) → m ≤ n) ∧ n = 15 :=
by
  -- Condition 1: n divides 30
  -- Condition 2: n divides 45
  -- Condition 3: n is greater than or equal to 5
  -- Condition 4: n is the largest such divisor
  -- Therefore, n = 15
  sorry

end eggs_in_each_basket_l1057_105755


namespace evaluate_expression_l1057_105785

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 240 / 961 := 
by 
  sorry

end evaluate_expression_l1057_105785


namespace unique_sums_count_l1057_105722

open Set

-- Defining the sets of chips in bags C and D
def BagC : Set ℕ := {1, 3, 7, 9}
def BagD : Set ℕ := {4, 6, 8}

-- The proof problem: show there are 7 unique sums
theorem unique_sums_count : (BagC ×ˢ BagD).image (λ p => p.1 + p.2) = {5, 7, 9, 11, 13, 15, 17} :=
by
  -- Proof omitted; complete proof would go here
  sorry

end unique_sums_count_l1057_105722


namespace range_satisfying_f_inequality_l1057_105764

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (1 + |x|) - (1 / (1 + x^2))

theorem range_satisfying_f_inequality : 
  ∀ x : ℝ, (1 / 3) < x ∧ x < 1 → f x > f (2 * x - 1) :=
by
  intro x hx
  sorry

end range_satisfying_f_inequality_l1057_105764


namespace American_carmakers_produce_l1057_105750

theorem American_carmakers_produce :
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  total = 5650000 :=
by
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  show total = 5650000
  sorry

end American_carmakers_produce_l1057_105750


namespace cos_alpha_given_tan_alpha_and_quadrant_l1057_105777

theorem cos_alpha_given_tan_alpha_and_quadrant 
  (α : ℝ) 
  (h1 : Real.tan α = -1/3)
  (h2 : π/2 < α ∧ α < π) : 
  Real.cos α = -3*Real.sqrt 10 / 10 :=
by
  sorry

end cos_alpha_given_tan_alpha_and_quadrant_l1057_105777


namespace least_number_to_make_divisible_by_9_l1057_105789

theorem least_number_to_make_divisible_by_9 (n : ℕ) :
  ∃ m : ℕ, (228712 + m) % 9 = 0 ∧ n = 5 :=
by
  sorry

end least_number_to_make_divisible_by_9_l1057_105789


namespace mean_days_jogged_l1057_105719

open Real

theorem mean_days_jogged 
  (p1 : ℕ := 5) (d1 : ℕ := 1)
  (p2 : ℕ := 4) (d2 : ℕ := 3)
  (p3 : ℕ := 10) (d3 : ℕ := 5)
  (p4 : ℕ := 7) (d4 : ℕ := 10)
  (p5 : ℕ := 3) (d5 : ℕ := 15)
  (p6 : ℕ := 1) (d6 : ℕ := 20) : 
  ( (p1 * d1 + p2 * d2 + p3 * d3 + p4 * d4 + p5 * d5 + p6 * d6) / (p1 + p2 + p3 + p4 + p5 + p6) : ℝ) = 6.73 :=
by
  sorry

end mean_days_jogged_l1057_105719


namespace grandmother_age_l1057_105731

theorem grandmother_age (minyoung_age_current : ℕ)
                         (minyoung_age_future : ℕ)
                         (grandmother_age_future : ℕ)
                         (h1 : minyoung_age_future = minyoung_age_current + 3)
                         (h2 : grandmother_age_future = 65)
                         (h3 : minyoung_age_future = 10) : grandmother_age_future - (minyoung_age_future -minyoung_age_current) = 62 := by
  sorry

end grandmother_age_l1057_105731


namespace nth_equation_l1057_105704

theorem nth_equation (n : ℕ) (h : 0 < n) : 9 * (n - 1) + n = 10 * n - 9 := 
  sorry

end nth_equation_l1057_105704


namespace prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l1057_105768

-- Definitions for the conditions.

-- cube configuration
def num_dice : ℕ := 27
def num_visible_dice : ℕ := 26
def num_faces_per_die : ℕ := 6
def num_visible_faces : ℕ := 54

-- Given probabilities
def prob_six (face : ℕ) : ℚ := 1/6
def prob_not_six (face : ℕ) : ℚ := 5/6
def prob_not_one (face : ℕ) : ℚ := 5/6

-- Expected values given conditions
def expected_num_sixes : ℚ := 9
def expected_sum_faces : ℚ := 189
def expected_diff_digits : ℚ := 6 - (5^6) / (2 * 3^17)

-- Probabilities given conditions
def prob_25_sixes_on_surface : ℚ := (26 * 5) / (6^26)
def prob_at_least_one_one : ℚ := 1 - (5^6) / (2^2 * 3^18)

-- Lean statements for proof

theorem prob_of_25_sixes_on_surface :
  prob_25_sixes_on_surface = 31 / (2^13 * 3^18) := by
  sorry

theorem prob_of_at_least_one_one_on_surface :
  prob_at_least_one_one = 0.99998992 := by
  sorry

theorem expected_number_of_sixes_on_surface :
  expected_num_sixes = 9 := by
  sorry

theorem expected_sum_of_numbers_on_surface :
  expected_sum_faces = 189 := by
  sorry

theorem expected_value_of_diff_digits_on_surface :
  expected_diff_digits = 6 - (5^6) / (2 * 3^17) := by
  sorry

end prob_of_25_sixes_on_surface_prob_of_at_least_one_one_on_surface_expected_number_of_sixes_on_surface_expected_sum_of_numbers_on_surface_expected_value_of_diff_digits_on_surface_l1057_105768


namespace rain_on_tuesday_l1057_105706

theorem rain_on_tuesday 
  (rain_monday : ℝ)
  (rain_less : ℝ) 
  (h1 : rain_monday = 0.9) 
  (h2 : rain_less = 0.7) : 
  (rain_monday - rain_less) = 0.2 :=
by
  sorry

end rain_on_tuesday_l1057_105706


namespace complex_root_product_l1057_105761

theorem complex_root_product (w : ℂ) (hw1 : w^3 = 1) (hw2 : w^2 + w + 1 = 0) :
(1 - w + w^2) * (1 + w - w^2) = 4 :=
sorry

end complex_root_product_l1057_105761


namespace seashells_count_l1057_105784

theorem seashells_count : 18 + 47 = 65 := by
  sorry

end seashells_count_l1057_105784


namespace company_fund_amount_l1057_105754

theorem company_fund_amount (n : ℕ) (h : 70 * n + 160 = 80 * n - 8) : 
  80 * n - 8 = 1352 :=
sorry

end company_fund_amount_l1057_105754


namespace total_toothpicks_correct_l1057_105701

-- Define the number of vertical lines and toothpicks in them
def num_vertical_lines : ℕ := 41
def num_toothpicks_per_vertical_line : ℕ := 20
def vertical_toothpicks : ℕ := num_vertical_lines * num_toothpicks_per_vertical_line

-- Define the number of horizontal lines and toothpicks in them
def num_horizontal_lines : ℕ := 21
def num_toothpicks_per_horizontal_line : ℕ := 40
def horizontal_toothpicks : ℕ := num_horizontal_lines * num_toothpicks_per_horizontal_line

-- Define the dimensions of the triangle
def triangle_base : ℕ := 20
def triangle_height : ℕ := 20
def triangle_hypotenuse : ℕ := 29 -- approximated

-- Total toothpicks in the triangle
def triangle_toothpicks : ℕ := triangle_height + triangle_hypotenuse

-- Total toothpicks used in the structure
def total_toothpicks : ℕ := vertical_toothpicks + horizontal_toothpicks + triangle_toothpicks

-- Theorem to prove the total number of toothpicks used is 1709
theorem total_toothpicks_correct : total_toothpicks = 1709 := by
  sorry

end total_toothpicks_correct_l1057_105701


namespace ratio_m_n_l1057_105776

theorem ratio_m_n (m n : ℕ) (h : (n : ℚ) / m = 3 / 7) : (m + n : ℚ) / m = 10 / 7 := by 
  sorry

end ratio_m_n_l1057_105776


namespace sphere_radius_l1057_105746

theorem sphere_radius (R : ℝ) (h : 4 * Real.pi * R^2 = 4 * Real.pi) : R = 1 :=
sorry

end sphere_radius_l1057_105746


namespace inequality_proof_l1057_105740

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l1057_105740


namespace lcm_16_35_l1057_105767

theorem lcm_16_35 : Nat.lcm 16 35 = 560 := by
  sorry

end lcm_16_35_l1057_105767


namespace haley_marbles_l1057_105728

theorem haley_marbles (boys : ℕ) (marbles_per_boy : ℕ) (h_boys : boys = 13) (h_marbles_per_boy : marbles_per_boy = 2) :
  boys * marbles_per_boy = 26 := 
by 
  sorry

end haley_marbles_l1057_105728


namespace common_difference_l1057_105751

theorem common_difference (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ)
    (h₁ : a 5 + a 6 = -10)
    (h₂ : S 14 = -14)
    (h₃ : ∀ n, S n = n * (a 1 + a n) / 2)
    (h₄ : ∀ n, a (n + 1) = a n + d) :
  d = 2 :=
sorry

end common_difference_l1057_105751


namespace range_of_a_l1057_105744

-- Defining the propositions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

-- Main theorem statement
theorem range_of_a (a : ℝ) : (¬(∃ x, p x) → ¬(∃ x, q x a)) → a < -3 :=
by
  sorry

end range_of_a_l1057_105744


namespace polynomial_product_l1057_105745

noncomputable def sum_of_coefficients (g h : ℤ) : ℤ := g + h

theorem polynomial_product (g h : ℤ) :
  (9 * d^3 - 5 * d^2 + g) * (4 * d^2 + h * d - 9) = 36 * d^5 - 11 * d^4 - 49 * d^3 + 45 * d^2 - 9 * d →
  sum_of_coefficients g h = 18 :=
by
  intro
  sorry

end polynomial_product_l1057_105745


namespace mass_of_man_is_correct_l1057_105708

-- Definitions for conditions
def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def sinking_depth : ℝ := 0.012
def density_of_water : ℝ := 1000

-- Volume of water displaced
def volume_displaced := length_of_boat * breadth_of_boat * sinking_depth

-- Mass of the man
def mass_of_man := density_of_water * volume_displaced

-- Prove that the mass of the man is 72 kg
theorem mass_of_man_is_correct : mass_of_man = 72 := by
  sorry

end mass_of_man_is_correct_l1057_105708


namespace intersection_l1057_105714

namespace Proof

def A := {x : ℝ | 0 ≤ x ∧ x ≤ 6}
def B := {x : ℝ | 3 * x^2 + x - 8 ≤ 0}

theorem intersection (x : ℝ) : x ∈ A ∩ B ↔ 0 ≤ x ∧ x ≤ (4:ℝ)/3 := 
by 
  sorry  -- proof placeholder

end Proof

end intersection_l1057_105714


namespace mary_jenny_red_marble_ratio_l1057_105709

def mary_red_marble := 30  -- Given that Mary collects the same as Jenny.
def jenny_red_marble := 30 -- Given
def jenny_blue_marble := 25 -- Given
def anie_red_marble := mary_red_marble + 20 -- Anie's red marbles count
def anie_blue_marble := 2 * jenny_blue_marble -- Anie's blue marbles count
def mary_blue_marble := anie_blue_marble / 2 -- Mary's blue marbles count

theorem mary_jenny_red_marble_ratio : 
  mary_red_marble / jenny_red_marble = 1 :=
by
  sorry

end mary_jenny_red_marble_ratio_l1057_105709


namespace wall_width_8_l1057_105724

theorem wall_width_8 (w h l : ℝ) (V : ℝ) 
  (h_eq : h = 6 * w) 
  (l_eq : l = 7 * h) 
  (vol_eq : w * h * l = 129024) : 
  w = 8 := 
by 
  sorry

end wall_width_8_l1057_105724


namespace arithmetic_sequence_next_term_perfect_square_sequence_next_term_l1057_105705

theorem arithmetic_sequence_next_term (a : ℕ → ℕ) (n : ℕ) (h₀ : a 0 = 0) (h₁ : ∀ n, a (n + 1) = a n + 3) :
  a 5 = 15 :=
by sorry

theorem perfect_square_sequence_next_term (b : ℕ → ℕ) (k : ℕ) (h₀ : ∀ k, b k = (k + 1) * (k + 1)) :
  b 5 = 36 :=
by sorry

end arithmetic_sequence_next_term_perfect_square_sequence_next_term_l1057_105705
