import Mathlib

namespace NUMINAMATH_GPT_log_base_half_cuts_all_horizontal_lines_l2386_238603

theorem log_base_half_cuts_all_horizontal_lines (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_eq : y = Real.logb 0.5 x) : ∃ x, ∀ k, k = Real.logb 0.5 x ↔ x > 0 := 
sorry

end NUMINAMATH_GPT_log_base_half_cuts_all_horizontal_lines_l2386_238603


namespace NUMINAMATH_GPT_tops_count_l2386_238619

def price_eq (C T : ℝ) : Prop := 3 * C + 6 * T = 1500 ∧ C + 12 * T = 1500

def tops_to_buy (C T : ℝ) (num_tops : ℝ) : Prop := 500 = 100 * num_tops

theorem tops_count (C T num_tops : ℝ) (h1 : price_eq C T) (h2 : tops_to_buy C T num_tops) : num_tops = 5 :=
by
  sorry

end NUMINAMATH_GPT_tops_count_l2386_238619


namespace NUMINAMATH_GPT_digit_difference_one_l2386_238678

variable (d C D : ℕ)

-- Assumptions
variables (h1 : d > 8)
variables (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3)

theorem digit_difference_one (h1 : d > 8) (h2 : d * d * d + C * d + D + d * d * d + C * d + C = 2 * d * d + 5 * d + 3) :
  C - D = 1 :=
by
  sorry

end NUMINAMATH_GPT_digit_difference_one_l2386_238678


namespace NUMINAMATH_GPT_range_of_m_l2386_238620

def f (m x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def f_derivative_nonnegative_on_interval (m : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0

theorem range_of_m (m : ℝ) : f_derivative_nonnegative_on_interval m ↔ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2386_238620


namespace NUMINAMATH_GPT_rainfall_difference_correct_l2386_238630

def rainfall_difference (monday_rain : ℝ) (tuesday_rain : ℝ) : ℝ :=
  monday_rain - tuesday_rain

theorem rainfall_difference_correct : rainfall_difference 0.9 0.2 = 0.7 :=
by
  simp [rainfall_difference]
  sorry

end NUMINAMATH_GPT_rainfall_difference_correct_l2386_238630


namespace NUMINAMATH_GPT_hexagon_classroom_students_l2386_238615

-- Define the number of sleeping students
def num_sleeping_students (students_detected : Nat → Nat) :=
  students_detected 2 + students_detected 3 + students_detected 6

-- Define the condition that the sum of snore-o-meter readings is 7
def snore_o_meter_sum (students_detected : Nat → Nat) :=
  2 * students_detected 2 + 3 * students_detected 3 + 6 * students_detected 6 = 7

-- Proof that the number of sleeping students is 3 given the conditions
theorem hexagon_classroom_students : 
  ∀ (students_detected : Nat → Nat), snore_o_meter_sum students_detected → num_sleeping_students students_detected = 3 :=
by
  intro students_detected h
  sorry

end NUMINAMATH_GPT_hexagon_classroom_students_l2386_238615


namespace NUMINAMATH_GPT_ten_digit_number_contains_repeated_digit_l2386_238645

open Nat

theorem ten_digit_number_contains_repeated_digit
  (n : ℕ)
  (h1 : 10^9 ≤ n^2 + 1)
  (h2 : n^2 + 1 < 10^10) :
  ∃ d1 d2 : ℕ, d1 ≠ d2 ∧ (d1 ∈ (digits 10 (n^2 + 1))) ∧ (d2 ∈ (digits 10 (n^2 + 1))) :=
sorry

end NUMINAMATH_GPT_ten_digit_number_contains_repeated_digit_l2386_238645


namespace NUMINAMATH_GPT_count_solutions_eq_4_l2386_238652

theorem count_solutions_eq_4 :
  ∀ x : ℝ, (x^2 - 5)^2 = 16 → x = 3 ∨ x = -3 ∨ x = 1 ∨ x = -1  := sorry

end NUMINAMATH_GPT_count_solutions_eq_4_l2386_238652


namespace NUMINAMATH_GPT_min_value_fraction_l2386_238689

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y, y = x - 4 ∧ (x + 11) / Real.sqrt (x - 4) = 2 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l2386_238689


namespace NUMINAMATH_GPT_impossibility_triplet_2002x2002_grid_l2386_238638

theorem impossibility_triplet_2002x2002_grid: 
  ∀ (M : Matrix ℕ (Fin 2002) (Fin 2002)),
    (∀ i j : Fin 2002, ∃ (r1 r2 r3 : Fin 2002), 
      (M i r1 > 0 ∧ M i r2 > 0 ∧ M i r3 > 0) ∨ 
      (M r1 j > 0 ∧ M r2 j > 0 ∧ M r3 j > 0)) →
    ¬ (∀ i j : Fin 2002, ∃ (a b c : ℕ), 
      M i j = a ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
      (∃ (r1 r2 r3 : Fin 2002), 
        (M i r1 = a ∨ M i r1 = b ∨ M i r1 = c) ∧ 
        (M i r2 = a ∨ M i r2 = b ∨ M i r2 = c) ∧ 
        (M i r3 = a ∨ M i r3 = b ∨ M i r3 = c)) ∨
      (∃ (c1 c2 c3 : Fin 2002), 
        (M c1 j = a ∨ M c1 j = b ∨ M c1 j = c) ∧ 
        (M c2 j = a ∨ M c2 j = b ∨ M c2 j = c) ∧ 
        (M c3 j = a ∨ M c3 j = b ∨ M c3 j = c)))
:= sorry

end NUMINAMATH_GPT_impossibility_triplet_2002x2002_grid_l2386_238638


namespace NUMINAMATH_GPT_max_pots_l2386_238671

theorem max_pots (x y z : ℕ) (h₁ : 3 * x + 4 * y + 9 * z = 100) (h₂ : 1 ≤ x) (h₃ : 1 ≤ y) (h₄ : 1 ≤ z) : 
  z ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_pots_l2386_238671


namespace NUMINAMATH_GPT_trader_sold_23_bags_l2386_238648

theorem trader_sold_23_bags
    (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) (x : ℕ)
    (h_initial : initial_stock = 55)
    (h_restocked : restocked = 132)
    (h_final : final_stock = 164)
    (h_equation : initial_stock - x + restocked = final_stock) :
    x = 23 :=
by
    -- Here will be the proof of the theorem
    sorry

end NUMINAMATH_GPT_trader_sold_23_bags_l2386_238648


namespace NUMINAMATH_GPT_triangle_y_values_l2386_238614

theorem triangle_y_values (y : ℕ) :
  (8 + 11 > y^2) ∧ (y^2 + 8 > 11) ∧ (y^2 + 11 > 8) ↔ y = 2 ∨ y = 3 ∨ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_y_values_l2386_238614


namespace NUMINAMATH_GPT_diagonal_length_of_rectangular_prism_l2386_238657

-- Define the dimensions of the rectangular prism
variables (a b c : ℕ) (a_pos : a = 12) (b_pos : b = 15) (c_pos : c = 8)

-- Define the theorem statement
theorem diagonal_length_of_rectangular_prism : 
  ∃ d : ℝ, d = Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) ∧ d = Real.sqrt 433 := 
by
  -- Note that the proof is intentionally omitted
  sorry

end NUMINAMATH_GPT_diagonal_length_of_rectangular_prism_l2386_238657


namespace NUMINAMATH_GPT_average_weight_of_a_b_c_l2386_238674

theorem average_weight_of_a_b_c (A B C : ℕ) 
  (h1 : (A + B) / 2 = 25) 
  (h2 : (B + C) / 2 = 28) 
  (hB : B = 16) : 
  (A + B + C) / 3 = 30 := 
by 
  sorry

end NUMINAMATH_GPT_average_weight_of_a_b_c_l2386_238674


namespace NUMINAMATH_GPT_one_third_percent_of_150_l2386_238665

theorem one_third_percent_of_150 : (1/3) * (150 / 100) = 0.5 := by
  sorry

end NUMINAMATH_GPT_one_third_percent_of_150_l2386_238665


namespace NUMINAMATH_GPT_unique_solution_of_function_eq_l2386_238695

theorem unique_solution_of_function_eq (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y) : f = id := 
sorry

end NUMINAMATH_GPT_unique_solution_of_function_eq_l2386_238695


namespace NUMINAMATH_GPT_total_fish_caught_l2386_238655

-- Definitions based on conditions
def brenden_morning_fish := 8
def brenden_fish_thrown_back := 3
def brenden_afternoon_fish := 5
def dad_fish := 13

-- Theorem representing the main question and its answer
theorem total_fish_caught : 
  (brenden_morning_fish + brenden_afternoon_fish - brenden_fish_thrown_back) + dad_fish = 23 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_total_fish_caught_l2386_238655


namespace NUMINAMATH_GPT_kids_bike_wheels_l2386_238616

theorem kids_bike_wheels
  (x : ℕ) 
  (h1 : 7 * 2 + 11 * x = 58) :
  x = 4 :=
sorry

end NUMINAMATH_GPT_kids_bike_wheels_l2386_238616


namespace NUMINAMATH_GPT_hexagon_angles_l2386_238622

theorem hexagon_angles (a e : ℝ) (h1 : a = e - 60) (h2 : 4 * a + 2 * e = 720) :
  e = 160 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_angles_l2386_238622


namespace NUMINAMATH_GPT_analytical_expression_smallest_positive_period_min_value_max_value_l2386_238632

noncomputable def P (x : ℝ) : ℝ × ℝ :=
  (Real.cos (2 * x) + 1, 1)

noncomputable def Q (x : ℝ) : ℝ × ℝ :=
  (1, Real.sqrt 3 * Real.sin (2 * x) + 1)

noncomputable def f (x : ℝ) : ℝ :=
  (P x).1 * (Q x).1 + (P x).2 * (Q x).2

theorem analytical_expression (x : ℝ) : 
  f x = 2 * Real.sin (2 * x + Real.pi / 6) + 2 :=
sorry

theorem smallest_positive_period : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
sorry

theorem min_value : 
  ∃ x : ℝ, f x = 0 :=
sorry

theorem max_value : 
  ∃ y : ℝ, f y = 4 :=
sorry

end NUMINAMATH_GPT_analytical_expression_smallest_positive_period_min_value_max_value_l2386_238632


namespace NUMINAMATH_GPT_kyle_age_l2386_238673

theorem kyle_age :
  ∃ (kyle shelley julian frederick tyson casey : ℕ),
    shelley = kyle - 3 ∧ 
    shelley = julian + 4 ∧
    julian = frederick - 20 ∧
    frederick = 2 * tyson ∧
    tyson = 2 * casey ∧
    casey = 15 ∧ 
    kyle = 47 :=
by
  sorry

end NUMINAMATH_GPT_kyle_age_l2386_238673


namespace NUMINAMATH_GPT_sphere_volume_ratio_l2386_238659

theorem sphere_volume_ratio (r1 r2 : ℝ) (S1 S2 V1 V2 : ℝ) 
(h1 : S1 = 4 * Real.pi * r1^2)
(h2 : S2 = 4 * Real.pi * r2^2)
(h3 : V1 = (4 / 3) * Real.pi * r1^3)
(h4 : V2 = (4 / 3) * Real.pi * r2^3)
(h_surface_ratio : S1 / S2 = 2 / 3) :
V1 / V2 = (2 * Real.sqrt 6) / 9 :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_ratio_l2386_238659


namespace NUMINAMATH_GPT_man_age_twice_son_age_in_two_years_l2386_238685

theorem man_age_twice_son_age_in_two_years :
  ∀ (S M X : ℕ), S = 30 → M = S + 32 → (M + X = 2 * (S + X)) → X = 2 :=
by
  intros S M X hS hM h
  sorry

end NUMINAMATH_GPT_man_age_twice_son_age_in_two_years_l2386_238685


namespace NUMINAMATH_GPT_takeoff_run_length_l2386_238618

theorem takeoff_run_length
  (t : ℕ) (h_t : t = 15)
  (v_kmh : ℕ) (h_v : v_kmh = 100)
  (uniform_acc : Prop) :
  ∃ S : ℕ, S = 208 := by
  sorry

end NUMINAMATH_GPT_takeoff_run_length_l2386_238618


namespace NUMINAMATH_GPT_students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l2386_238676

noncomputable def numStudentsKnowingSecret (n : ℕ) : ℕ :=
  (3^(n + 1) - 1) / 2

theorem students_on_seventh_day :
  (numStudentsKnowingSecret 7) = 3280 :=
by
  sorry

theorem day_of_week (n : ℕ) : String :=
  if n % 7 = 0 then "Monday" else
  if n % 7 = 1 then "Tuesday" else
  if n % 7 = 2 then "Wednesday" else
  if n % 7 = 3 then "Thursday" else
  if n % 7 = 4 then "Friday" else
  if n % 7 = 5 then "Saturday" else
  "Sunday"

theorem day_when_3280_students_know_secret :
  day_of_week 7 = "Sunday" :=
by
  sorry

end NUMINAMATH_GPT_students_on_seventh_day_day_of_week_day_when_3280_students_know_secret_l2386_238676


namespace NUMINAMATH_GPT_age_difference_l2386_238600

variable (E Y : ℕ)

theorem age_difference (hY : Y = 35) (hE : E - 15 = 2 * (Y - 15)) : E - Y = 20 := by
  -- Assertions and related steps could be handled subsequently.
  sorry

end NUMINAMATH_GPT_age_difference_l2386_238600


namespace NUMINAMATH_GPT_roof_area_l2386_238651

theorem roof_area (w l : ℕ) (h1 : l = 4 * w) (h2 : l - w = 42) : l * w = 784 :=
by
  sorry

end NUMINAMATH_GPT_roof_area_l2386_238651


namespace NUMINAMATH_GPT_inequality_holds_l2386_238682

theorem inequality_holds (a : ℝ) : 3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l2386_238682


namespace NUMINAMATH_GPT_tomatoes_initially_l2386_238602

-- Conditions
def tomatoes_picked_yesterday : ℕ := 56
def tomatoes_picked_today : ℕ := 41
def tomatoes_left_after_yesterday : ℕ := 104

-- The statement to prove
theorem tomatoes_initially : tomatoes_left_after_yesterday + tomatoes_picked_yesterday + tomatoes_picked_today = 201 :=
  by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_tomatoes_initially_l2386_238602


namespace NUMINAMATH_GPT_largest_possible_length_d_l2386_238667

theorem largest_possible_length_d (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : a ≤ b)
  (h3 : b ≤ c)
  (h4 : c ≤ d) 
  (h5 : d < a + b + c) : 
  d < 1 :=
sorry

end NUMINAMATH_GPT_largest_possible_length_d_l2386_238667


namespace NUMINAMATH_GPT_minimum_value_x_squared_plus_y_squared_l2386_238623

-- We define our main proposition in Lean
theorem minimum_value_x_squared_plus_y_squared (x y : ℝ) 
  (h : (x + 5)^2 + (y - 12)^2 = 196) : x^2 + y^2 ≥ 169 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_squared_plus_y_squared_l2386_238623


namespace NUMINAMATH_GPT_first_group_men_l2386_238611

theorem first_group_men (x : ℕ) (days1 days2 : ℝ) (men2 : ℕ) (h1 : days1 = 25) (h2 : days2 = 17.5) (h3 : men2 = 20) (h4 : x * days1 = men2 * days2) : x = 14 := 
by
  sorry

end NUMINAMATH_GPT_first_group_men_l2386_238611


namespace NUMINAMATH_GPT_intersection_S_T_eq_T_l2386_238647

-- Define the sets S and T
def S : Set ℤ := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- State the theorem
theorem intersection_S_T_eq_T : S ∩ T = T :=
sorry

end NUMINAMATH_GPT_intersection_S_T_eq_T_l2386_238647


namespace NUMINAMATH_GPT_email_count_first_day_l2386_238612

theorem email_count_first_day (E : ℕ) 
  (h1 : ∃ E, E + E / 2 + E / 4 + E / 8 = 30) : E = 16 :=
by
  sorry

end NUMINAMATH_GPT_email_count_first_day_l2386_238612


namespace NUMINAMATH_GPT_interest_rate_proof_l2386_238658

noncomputable def compound_interest_rate (P A : ℝ) (t n : ℕ) : ℝ :=
  (((A / P)^(1 / (n * t))) - 1) * n

theorem interest_rate_proof :
  ∀ P A : ℝ, ∀ t n : ℕ, P = 1093.75 → A = 1183 → t = 2 → n = 1 →
  compound_interest_rate P A t n = 0.0399 :=
by
  intros P A t n hP hA ht hn
  rw [hP, hA, ht, hn]
  unfold compound_interest_rate
  sorry

end NUMINAMATH_GPT_interest_rate_proof_l2386_238658


namespace NUMINAMATH_GPT_y_intercept_of_line_l2386_238642

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 6 * y = 24) : y = 4 := by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l2386_238642


namespace NUMINAMATH_GPT_pascal_sum_difference_l2386_238663

open BigOperators

noncomputable def a_i (i : ℕ) := Nat.choose 3005 i
noncomputable def b_i (i : ℕ) := Nat.choose 3006 i
noncomputable def c_i (i : ℕ) := Nat.choose 3007 i

theorem pascal_sum_difference :
  (∑ i in Finset.range 3007, (b_i i) / (c_i i)) - (∑ i in Finset.range 3006, (a_i i) / (b_i i)) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_pascal_sum_difference_l2386_238663


namespace NUMINAMATH_GPT_ratio_ab_bd_l2386_238699

-- Definitions based on the given conditions
def ab : ℝ := 4
def bc : ℝ := 8
def cd : ℝ := 5
def bd : ℝ := bc + cd

-- Theorem statement
theorem ratio_ab_bd :
  ((ab / bd) = (4 / 13)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_ab_bd_l2386_238699


namespace NUMINAMATH_GPT_david_more_pushups_than_zachary_l2386_238684

theorem david_more_pushups_than_zachary :
  ∀ (zachary_pushups zachary_crunches david_crunches : ℕ),
    zachary_pushups = 34 →
    zachary_crunches = 62 →
    david_crunches = 45 →
    david_crunches + 17 = zachary_crunches →
    david_crunches + 17 - zachary_pushups = 17 :=
by
  intros zachary_pushups zachary_crunches david_crunches
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_david_more_pushups_than_zachary_l2386_238684


namespace NUMINAMATH_GPT_contradiction_method_conditions_l2386_238654

theorem contradiction_method_conditions :
  (using_judgments_contrary_to_conclusion ∧ using_conditions_of_original_proposition ∧ using_axioms_theorems_definitions) =
  (needed_conditions_method_of_contradiction) :=
sorry

end NUMINAMATH_GPT_contradiction_method_conditions_l2386_238654


namespace NUMINAMATH_GPT_smallest_n_conditions_l2386_238686

theorem smallest_n_conditions (n : ℕ) : 
  (∃ k m : ℕ, 4 * n = k^2 ∧ 5 * n = m^5 ∧ ∀ n' : ℕ, (∃ k' m' : ℕ, 4 * n' = k'^2 ∧ 5 * n' = m'^5) → n ≤ n') → 
  n = 625 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_n_conditions_l2386_238686


namespace NUMINAMATH_GPT_part_a_contradiction_l2386_238683

theorem part_a_contradiction :
  ¬ (225 / 25 + 75 = 100 - 16 → 25 * (9 / (1 + 3)) = 84) :=
by
  sorry

end NUMINAMATH_GPT_part_a_contradiction_l2386_238683


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l2386_238610

variable (a b : ℝ)

def in_interval (x : ℝ) := 0 < x ∧ x < 1

theorem relationship_between_a_and_b 
  (ha : in_interval a)
  (hb : in_interval b)
  (h : (1 - a) * b > 1 / 4) : a < b :=
sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l2386_238610


namespace NUMINAMATH_GPT_weight_of_replaced_person_l2386_238639

theorem weight_of_replaced_person 
  (avg_increase : ℝ)
  (num_persons : ℕ)
  (new_person_weight : ℝ)
  (weight_increase : ℝ)
  (new_person_might_be_90_kg : new_person_weight = 90)
  (average_increase_by_3_5_kg : avg_increase = 3.5)
  (group_of_8_persons : num_persons = 8)
  (total_weight_increase_formula : weight_increase = num_persons * avg_increase)
  (weight_of_replaced_person : ℝ)
  (weight_difference_formula : weight_of_replaced_person = new_person_weight - weight_increase) :
  weight_of_replaced_person = 62 :=
sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l2386_238639


namespace NUMINAMATH_GPT_max_value_of_quadratic_on_interval_l2386_238641

theorem max_value_of_quadratic_on_interval : 
  ∃ (x : ℝ), -2 ≤ x ∧ x ≤ 2 ∧ (∀ y, (∃ x, -2 ≤ x ∧ x ≤ 2 ∧ y = (x + 1)^2 - 4) → y ≤ 5) :=
sorry

end NUMINAMATH_GPT_max_value_of_quadratic_on_interval_l2386_238641


namespace NUMINAMATH_GPT_find_d_l2386_238626

open Real

-- Define the given conditions
variable (a b c d e : ℝ)

axiom cond1 : 3 * (a^2 + b^2 + c^2) + 4 = 2 * d + sqrt (a + b + c - d + e)
axiom cond2 : e = 1

-- Define the theorem stating that d = 7/4 under the given conditions
theorem find_d : d = 7/4 := by
  sorry

end NUMINAMATH_GPT_find_d_l2386_238626


namespace NUMINAMATH_GPT_find_x_l2386_238680

theorem find_x (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z)
(h₄ : x^2 / y = 3) (h₅ : y^2 / z = 4) (h₆ : z^2 / x = 5) : 
  x = (6480 : ℝ)^(1/7 : ℝ) :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l2386_238680


namespace NUMINAMATH_GPT_cost_of_old_car_l2386_238640

theorem cost_of_old_car (C_old C_new : ℝ): 
  C_new = 2 * C_old → 
  1800 + 2000 = C_new → 
  C_old = 1900 :=
by
  intros H1 H2
  sorry

end NUMINAMATH_GPT_cost_of_old_car_l2386_238640


namespace NUMINAMATH_GPT_minimum_value_expression_l2386_238698

theorem minimum_value_expression 
  (a b c d : ℝ)
  (h1 : (2 * a^2 - Real.log a) / b = 1)
  (h2 : (3 * c - 2) / d = 1) :
  ∃ min_val : ℝ, min_val = (a - c)^2 + (b - d)^2 ∧ min_val = 1 / 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_value_expression_l2386_238698


namespace NUMINAMATH_GPT_question1_question2_l2386_238694

theorem question1 :
  (1:ℝ) * (Real.sqrt 12 + Real.sqrt 20) + (Real.sqrt 3 - Real.sqrt 5) = 3 * Real.sqrt 3 + Real.sqrt 5 := 
by sorry

theorem question2 :
  (4 * Real.sqrt 2 - 3 * Real.sqrt 6) / (2 * Real.sqrt 2) - (Real.sqrt 8 + Real.pi)^0 = 1 - 3 * Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_GPT_question1_question2_l2386_238694


namespace NUMINAMATH_GPT_area_of_ring_between_outermost_and_middle_circle_l2386_238601

noncomputable def pi : ℝ := Real.pi

theorem area_of_ring_between_outermost_and_middle_circle :
  let r_outermost := 12
  let r_middle := 8
  let A_outermost := pi * r_outermost^2
  let A_middle := pi * r_middle^2
  A_outermost - A_middle = 80 * pi :=
by 
  sorry

end NUMINAMATH_GPT_area_of_ring_between_outermost_and_middle_circle_l2386_238601


namespace NUMINAMATH_GPT_students_wanted_fruit_l2386_238646

theorem students_wanted_fruit (red_apples green_apples extra_fruit : ℕ)
  (h_red : red_apples = 42)
  (h_green : green_apples = 7)
  (h_extra : extra_fruit = 40) :
  red_apples + green_apples + extra_fruit - (red_apples + green_apples) = 40 :=
by
  sorry

end NUMINAMATH_GPT_students_wanted_fruit_l2386_238646


namespace NUMINAMATH_GPT_line_ellipse_intersection_l2386_238621

-- Define the problem conditions and the proof problem statement.
theorem line_ellipse_intersection (k m : ℝ) : 
  (∀ x y, y - k * x - 1 = 0 → ((x^2 / 5) + (y^2 / m) = 1)) →
  (m ≥ 1) ∧ (m ≠ 5) ∧ (m < 5 ∨ m > 5) :=
sorry

end NUMINAMATH_GPT_line_ellipse_intersection_l2386_238621


namespace NUMINAMATH_GPT_smallest_multiple_of_seven_gt_neg50_l2386_238670

theorem smallest_multiple_of_seven_gt_neg50 : ∃ (n : ℤ), n % 7 = 0 ∧ n > -50 ∧ ∀ (m : ℤ), m % 7 = 0 → m > -50 → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_multiple_of_seven_gt_neg50_l2386_238670


namespace NUMINAMATH_GPT_minimum_perimeter_l2386_238696

def fractional_part (x : ℚ) : ℚ := x - x.floor

-- Define l, m, n being sides of the triangle with l > m > n
variables (l m n : ℤ)

-- Defining conditions as Lean predicates
def triangle_sides (l m n : ℤ) : Prop := l > m ∧ m > n

def fractional_part_condition (l m n : ℤ) : Prop :=
  fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4) ∧
  fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)

-- Prove the minimum perimeter is 3003 given above conditions
theorem minimum_perimeter (l m n : ℤ) :
  triangle_sides l m n →
  fractional_part_condition l m n →
  l + m + n = 3003 :=
by
  intros h_sides h_fractional
  sorry

end NUMINAMATH_GPT_minimum_perimeter_l2386_238696


namespace NUMINAMATH_GPT_total_tickets_sold_l2386_238664

theorem total_tickets_sold (n : ℕ) 
  (h1 : n * n = 1681) : 
  2 * n = 82 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l2386_238664


namespace NUMINAMATH_GPT_mark_speed_l2386_238656

theorem mark_speed
  (chris_speed : ℕ)
  (distance_to_school : ℕ)
  (mark_total_distance : ℕ)
  (mark_time_longer : ℕ)
  (chris_speed_eq : chris_speed = 3)
  (distance_to_school_eq : distance_to_school = 9)
  (mark_total_distance_eq : mark_total_distance = 15)
  (mark_time_longer_eq : mark_time_longer = 2) :
  mark_total_distance / (distance_to_school / chris_speed + mark_time_longer) = 3 := 
by
  sorry 

end NUMINAMATH_GPT_mark_speed_l2386_238656


namespace NUMINAMATH_GPT_periodic_decimal_to_fraction_l2386_238625

theorem periodic_decimal_to_fraction : (0.7 + 0.32 : ℝ) == (1013 / 990 : ℝ) := by
  sorry

end NUMINAMATH_GPT_periodic_decimal_to_fraction_l2386_238625


namespace NUMINAMATH_GPT_simplify_fraction_l2386_238653

theorem simplify_fraction :
  ((1 / 4) + (1 / 6)) / ((3 / 8) - (1 / 3)) = 10 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2386_238653


namespace NUMINAMATH_GPT_M_subset_N_l2386_238644

def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2) * 180 + 45}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4) * 180 + 45}

theorem M_subset_N : M ⊆ N :=
sorry

end NUMINAMATH_GPT_M_subset_N_l2386_238644


namespace NUMINAMATH_GPT_jenni_age_l2386_238606

theorem jenni_age 
    (B J : ℤ)
    (h1 : B + J = 70)
    (h2 : B - J = 32) : 
    J = 19 :=
by
  sorry

end NUMINAMATH_GPT_jenni_age_l2386_238606


namespace NUMINAMATH_GPT_range_of_a_l2386_238691

theorem range_of_a (h : ¬ ∃ x : ℝ, x < 2023 ∧ x > a) : a ≥ 2023 := 
sorry

end NUMINAMATH_GPT_range_of_a_l2386_238691


namespace NUMINAMATH_GPT_F_2_f_3_equals_341_l2386_238668

def f (a : ℕ) : ℕ := a^2 - 2
def F (a b : ℕ) : ℕ := b^3 - a

theorem F_2_f_3_equals_341 : F 2 (f 3) = 341 := by
  sorry

end NUMINAMATH_GPT_F_2_f_3_equals_341_l2386_238668


namespace NUMINAMATH_GPT_num_teachers_l2386_238643

variable (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ)

theorem num_teachers (h1 : num_students = 20) (h2 : ticket_cost = 5) (h3 : total_cost = 115) :
  (total_cost / ticket_cost - num_students = 3) :=
by
  sorry

end NUMINAMATH_GPT_num_teachers_l2386_238643


namespace NUMINAMATH_GPT_total_cookies_and_brownies_l2386_238624

-- Define the conditions
def bagsOfCookies : ℕ := 272
def cookiesPerBag : ℕ := 45
def bagsOfBrownies : ℕ := 158
def browniesPerBag : ℕ := 32

-- Define the total cookies, total brownies, and total items
def totalCookies := bagsOfCookies * cookiesPerBag
def totalBrownies := bagsOfBrownies * browniesPerBag
def totalItems := totalCookies + totalBrownies

-- State the theorem to prove
theorem total_cookies_and_brownies : totalItems = 17296 := by
  sorry

end NUMINAMATH_GPT_total_cookies_and_brownies_l2386_238624


namespace NUMINAMATH_GPT_probability_at_least_one_l2386_238697

theorem probability_at_least_one (
    pA pB pC : ℝ
) (hA : pA = 0.9) (hB : pB = 0.8) (hC : pC = 0.7) (independent : true) : 
    (1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.994 := 
by
  rw [hA, hB, hC]
  sorry

end NUMINAMATH_GPT_probability_at_least_one_l2386_238697


namespace NUMINAMATH_GPT_measure_45_minutes_l2386_238627

-- Definitions of the conditions
structure Conditions where
  lighter : Prop
  strings : ℕ
  burn_time : ℕ → ℕ
  non_uniform_burn : Prop

-- We can now state the problem in Lean
theorem measure_45_minutes (c : Conditions) (h1 : c.lighter) (h2 : c.strings = 2)
  (h3 : ∀ s, s < 2 → c.burn_time s = 60) (h4 : c.non_uniform_burn) :
  ∃ t, t = 45 := 
sorry

end NUMINAMATH_GPT_measure_45_minutes_l2386_238627


namespace NUMINAMATH_GPT_plan_b_more_cost_effective_l2386_238679

noncomputable def fare (x : ℝ) : ℝ :=
if x < 3 then 5
else if x <= 10 then 1.2 * x + 1.4
else 1.8 * x - 4.6

theorem plan_b_more_cost_effective :
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  plan_a > plan_b :=
by
  let plan_a := 2 * fare 15
  let plan_b := 3 * fare 10
  sorry

end NUMINAMATH_GPT_plan_b_more_cost_effective_l2386_238679


namespace NUMINAMATH_GPT_neg_p_sufficient_for_neg_q_l2386_238628

def p (x : ℝ) : Prop := |2 * x - 3| > 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem neg_p_sufficient_for_neg_q :
  (∀ x, ¬ p x → ¬ q x) ∧ ¬ (∀ x, ¬ q x → ¬ p x) :=
by
  -- Placeholder to indicate skipping the proof
  sorry

end NUMINAMATH_GPT_neg_p_sufficient_for_neg_q_l2386_238628


namespace NUMINAMATH_GPT_fraction_work_completed_by_third_group_l2386_238690

def working_speeds (name : String) : ℚ :=
  match name with
  | "A"  => 1
  | "B"  => 2
  | "C"  => 1.5
  | "D"  => 2.5
  | "E"  => 3
  | "F"  => 2
  | "W1" => 1
  | "W2" => 1.5
  | "W3" => 1
  | "W4" => 1
  | "W5" => 0.5
  | "W6" => 1
  | "W7" => 1.5
  | "W8" => 1
  | _    => 0

def work_done_per_hour (workers : List String) : ℚ :=
  workers.map working_speeds |>.sum

def first_group : List String := ["A", "B", "C", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
def second_group : List String := ["A", "B", "C", "D", "E", "F", "W1", "W2"]
def third_group : List String := ["A", "B", "C", "D", "E", "W1", "W2"]

theorem fraction_work_completed_by_third_group :
  (work_done_per_hour third_group) / (work_done_per_hour second_group) = 25 / 29 :=
by
  sorry

end NUMINAMATH_GPT_fraction_work_completed_by_third_group_l2386_238690


namespace NUMINAMATH_GPT_nearest_whole_number_l2386_238677

theorem nearest_whole_number (x : ℝ) (h : x = 7263.4987234) : Int.floor (x + 0.5) = 7263 := by
  sorry

end NUMINAMATH_GPT_nearest_whole_number_l2386_238677


namespace NUMINAMATH_GPT_tan_phi_eq_sqrt3_l2386_238693

theorem tan_phi_eq_sqrt3
  (φ : ℝ)
  (h1 : Real.cos (Real.pi / 2 - φ) = Real.sqrt 3 / 2)
  (h2 : abs φ < Real.pi / 2) :
  Real.tan φ = Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_phi_eq_sqrt3_l2386_238693


namespace NUMINAMATH_GPT_linear_equation_in_one_variable_proof_l2386_238660

noncomputable def is_linear_equation_in_one_variable (eq : String) : Prop :=
  eq = "3x = 2x" ∨ eq = "ax + b = 0"

theorem linear_equation_in_one_variable_proof :
  is_linear_equation_in_one_variable "3x = 2x" ∧ ¬is_linear_equation_in_one_variable "3x - (4 + 3x) = 2"
  ∧ ¬is_linear_equation_in_one_variable "x + y = 1" ∧ ¬is_linear_equation_in_one_variable "x^2 + 1 = 5" :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_in_one_variable_proof_l2386_238660


namespace NUMINAMATH_GPT_find_ab_l2386_238637

theorem find_ab (a b : ℝ) (h1 : a - b = 26) (h2 : a + b = 15) :
  a = 41 / 2 ∧ b = 11 / 2 :=
sorry

end NUMINAMATH_GPT_find_ab_l2386_238637


namespace NUMINAMATH_GPT_quadratic_real_root_exists_l2386_238681

theorem quadratic_real_root_exists :
  ¬ (∃ x : ℝ, x^2 + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 + x + 1 = 0) ∧
  ¬ (∃ x : ℝ, x^2 - x + 1 = 0) ∧
  (∃ x : ℝ, x^2 - x - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_root_exists_l2386_238681


namespace NUMINAMATH_GPT_international_news_duration_l2386_238633

theorem international_news_duration
  (total_duration : ℕ := 30)
  (national_news : ℕ := 12)
  (sports : ℕ := 5)
  (weather_forecasts : ℕ := 2)
  (advertising : ℕ := 6) :
  total_duration - national_news - sports - weather_forecasts - advertising = 5 :=
by
  sorry

end NUMINAMATH_GPT_international_news_duration_l2386_238633


namespace NUMINAMATH_GPT_probability_one_marble_each_color_l2386_238666

theorem probability_one_marble_each_color :
  let total_marbles := 9
  let total_ways := Nat.choose total_marbles 3
  let favorable_ways := 3 * 3 * 3
  let probability := favorable_ways / total_ways
  probability = 9 / 28 :=
by
  sorry

end NUMINAMATH_GPT_probability_one_marble_each_color_l2386_238666


namespace NUMINAMATH_GPT_validCardSelections_l2386_238688

def numberOfValidSelections : ℕ :=
  let totalCards := 12
  let redCards := 4
  let otherColors := 8 -- 4 yellow + 4 blue
  let totalSelections := Nat.choose totalCards 3
  let nonRedSelections := Nat.choose otherColors 3
  let oneRedSelections := Nat.choose redCards 1 * Nat.choose otherColors 2
  let sameColorSelections := 3 * Nat.choose 4 3 -- 3 colors, 4 cards each, selecting 3
  (nonRedSelections + oneRedSelections)

theorem validCardSelections : numberOfValidSelections = 160 := by
  sorry

end NUMINAMATH_GPT_validCardSelections_l2386_238688


namespace NUMINAMATH_GPT_find_a5_l2386_238631

variable {α : Type*} [Field α]

def geometric_seq (a : α) (q : α) (n : ℕ) : α := a * q ^ (n - 1)

theorem find_a5 (a q : α) 
  (h1 : geometric_seq a q 2 = 4)
  (h2 : geometric_seq a q 6 * geometric_seq a q 7 = 16 * geometric_seq a q 9) :
  geometric_seq a q 5 = 32 ∨ geometric_seq a q 5 = -32 :=
by
  -- Proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_find_a5_l2386_238631


namespace NUMINAMATH_GPT_find_value_of_B_l2386_238692

theorem find_value_of_B (B : ℚ) (h : 4 * B + 4 = 33) : B = 29 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_B_l2386_238692


namespace NUMINAMATH_GPT_product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l2386_238675

theorem product_of_three_divisors_of_5_pow_4_eq_5_pow_4 (a b c : ℕ) (h1 : a * b * c = 5^4) (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c) : a + b + c = 131 :=
sorry

end NUMINAMATH_GPT_product_of_three_divisors_of_5_pow_4_eq_5_pow_4_l2386_238675


namespace NUMINAMATH_GPT_find_roots_l2386_238608

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_roots 
  (h_symm : ∀ x : ℝ, f (2 + x) = f (2 - x))
  (h_three_roots : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0)
  (h_zero_root : f 0 = 0) :
  ∃ a b : ℝ, a = 2 ∧ b = 4 ∧ f a = 0 ∧ f b = 0 :=
sorry

end NUMINAMATH_GPT_find_roots_l2386_238608


namespace NUMINAMATH_GPT_reduced_price_tickets_first_week_l2386_238672

theorem reduced_price_tickets_first_week (total_tickets sold_at_full_price : ℕ) 
  (condition1 : total_tickets = 25200) 
  (condition2 : sold_at_full_price = 16500)
  (condition3 : ∃ R, total_tickets = R + 5 * R) : 
  ∃ R : ℕ, R = 3300 := 
by sorry

end NUMINAMATH_GPT_reduced_price_tickets_first_week_l2386_238672


namespace NUMINAMATH_GPT_benny_cards_left_l2386_238635

theorem benny_cards_left (n : ℕ) : ℕ :=
  (n + 4) / 2

end NUMINAMATH_GPT_benny_cards_left_l2386_238635


namespace NUMINAMATH_GPT_average_length_tapes_l2386_238687

def lengths (l1 l2 l3 l4 l5 : ℝ) : Prop :=
  l1 = 35 ∧ l2 = 29 ∧ l3 = 35.5 ∧ l4 = 36 ∧ l5 = 30.5

theorem average_length_tapes
  (l1 l2 l3 l4 l5 : ℝ)
  (h : lengths l1 l2 l3 l4 l5) :
  (l1 + l2 + l3 + l4 + l5) / 5 = 33.2 := 
by
  sorry

end NUMINAMATH_GPT_average_length_tapes_l2386_238687


namespace NUMINAMATH_GPT_fabric_delivered_on_monday_amount_l2386_238669

noncomputable def cost_per_yard : ℝ := 2
noncomputable def earnings : ℝ := 140

def fabric_delivered_on_monday (x : ℝ) : Prop :=
  let tuesday := 2 * x
  let wednesday := (1 / 4) * tuesday
  let total_yards := x + tuesday + wednesday
  let total_earnings := total_yards * cost_per_yard
  total_earnings = earnings

theorem fabric_delivered_on_monday_amount : ∃ x : ℝ, fabric_delivered_on_monday x ∧ x = 20 :=
by sorry

end NUMINAMATH_GPT_fabric_delivered_on_monday_amount_l2386_238669


namespace NUMINAMATH_GPT_power_of_7_mod_8_l2386_238617

theorem power_of_7_mod_8 : 7^123 % 8 = 7 :=
by sorry

end NUMINAMATH_GPT_power_of_7_mod_8_l2386_238617


namespace NUMINAMATH_GPT_symmetric_scanning_codes_count_l2386_238649

structure Grid (n : ℕ) :=
  (cells : Fin n × Fin n → Bool)

def is_symmetric_90 (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - j, i)

def is_symmetric_reflection_mid_side (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (7 - i, j) ∧ g.cells (i, j) = g.cells (i, 7 - j)

def is_symmetric_reflection_diagonal (g : Grid 8) : Prop :=
  ∀ i j, g.cells (i, j) = g.cells (j, i)

def has_at_least_one_black_and_one_white (g : Grid 8) : Prop :=
  ∃ i j, g.cells (i, j) ∧ ∃ i j, ¬g.cells (i, j)

noncomputable def count_symmetric_scanning_codes : ℕ :=
  (sorry : ℕ)

theorem symmetric_scanning_codes_count : count_symmetric_scanning_codes = 62 :=
  sorry

end NUMINAMATH_GPT_symmetric_scanning_codes_count_l2386_238649


namespace NUMINAMATH_GPT_highest_probability_highspeed_rail_l2386_238607

def total_balls : ℕ := 10
def beidou_balls : ℕ := 3
def tianyan_balls : ℕ := 2
def highspeed_rail_balls : ℕ := 5

theorem highest_probability_highspeed_rail :
  (highspeed_rail_balls : ℚ) / total_balls > (beidou_balls : ℚ) / total_balls ∧
  (highspeed_rail_balls : ℚ) / total_balls > (tianyan_balls : ℚ) / total_balls :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_highest_probability_highspeed_rail_l2386_238607


namespace NUMINAMATH_GPT_multiply_and_simplify_fractions_l2386_238650

theorem multiply_and_simplify_fractions :
  (2 / 3) * (4 / 7) * (9 / 11) = 24 / 77 := 
by
  sorry

end NUMINAMATH_GPT_multiply_and_simplify_fractions_l2386_238650


namespace NUMINAMATH_GPT_find_m_value_l2386_238604

def quadratic_inequality_solution_set (a b c : ℝ) (m : ℝ) := {x : ℝ | 0 < x ∧ x < 2}

theorem find_m_value (a b c : ℝ) (m : ℝ) 
  (h1 : a = -1/2) 
  (h2 : b = 2) 
  (h3 : c = m) 
  (h4 : quadratic_inequality_solution_set a b c m = {x : ℝ | 0 < x ∧ x < 2}) : 
  m = 1 := 
sorry

end NUMINAMATH_GPT_find_m_value_l2386_238604


namespace NUMINAMATH_GPT_problem1_problem2_l2386_238636

-- (Problem 1)
def A : Set ℝ := {x | x^2 + 2 * x < 0}
def B : Set ℝ := {x | x ≥ -1}
def complement_A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 0}
def intersection_complement_A_B : Set ℝ := {x | x ≥ 0}

theorem problem1 : (complement_A ∩ B) = intersection_complement_A_B :=
by
  sorry

-- (Problem 2)
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a + 1}

theorem problem2 {a : ℝ} : (C a ⊆ A) ↔ (a ≤ -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2386_238636


namespace NUMINAMATH_GPT_part1_part2_l2386_238605

theorem part1 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n) : n ∣ m := 
sorry

theorem part2 (m n : ℕ) (h1 : m > n) (h2 : Nat.gcd m n + Nat.lcm m n = m + n)
(h3 : m - n = 10) : (m, n) = (11, 1) ∨ (m, n) = (12, 2) ∨ (m, n) = (15, 5) ∨ (m, n) = (20, 10) := 
sorry

end NUMINAMATH_GPT_part1_part2_l2386_238605


namespace NUMINAMATH_GPT_time_for_A_and_C_to_complete_work_l2386_238609

variable (A_rate B_rate C_rate : ℝ)

theorem time_for_A_and_C_to_complete_work
  (hA : A_rate = 1 / 4)
  (hBC : 1 / 3 = B_rate + C_rate)
  (hB : B_rate = 1 / 12) :
  1 / (A_rate + C_rate) = 2 :=
by
  -- Here would be the proof logic
  sorry

end NUMINAMATH_GPT_time_for_A_and_C_to_complete_work_l2386_238609


namespace NUMINAMATH_GPT_obtuse_angled_triangles_in_polygon_l2386_238634

/-- The number of obtuse-angled triangles formed by the vertices of a regular polygon with 2n+1 sides -/
theorem obtuse_angled_triangles_in_polygon (n : ℕ) : 
  (2 * n + 1) * (n * (n - 1)) / 2 = (2 * n + 1) * (n * (n - 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_obtuse_angled_triangles_in_polygon_l2386_238634


namespace NUMINAMATH_GPT_simplify_expression_correct_l2386_238662

variable {R : Type} [CommRing R]

def simplify_expression (x : R) : R :=
  2 * x^2 * (4 * x^3 - 3 * x + 1) - 7 * (x^3 - 3 * x^2 + 2 * x - 8)

theorem simplify_expression_correct (x : R) : 
  simplify_expression x = 8 * x^5 + 0 * x^4 - 13 * x^3 + 23 * x^2 - 14 * x + 56 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l2386_238662


namespace NUMINAMATH_GPT_prism_closed_polygonal_chain_impossible_l2386_238613

theorem prism_closed_polygonal_chain_impossible
  (lateral_edges : ℕ)
  (base_edges : ℕ)
  (total_edges : ℕ)
  (h_lateral : lateral_edges = 171)
  (h_base : base_edges = 171)
  (h_total : total_edges = 513)
  (h_total_sum : total_edges = 2 * base_edges + lateral_edges) :
  ¬ (∃ f : Fin 513 → (ℝ × ℝ × ℝ), (f 513 = f 0) ∧
    ∀ i, ( f (i + 1) - f i = (1, 0, 0) ∨ f (i + 1) - f i = (0, 1, 0) ∨ f (i + 1) - f i = (0, 0, 1) ∨ f (i + 1) - f i = (0, 0, -1) )) :=
by
  sorry

end NUMINAMATH_GPT_prism_closed_polygonal_chain_impossible_l2386_238613


namespace NUMINAMATH_GPT_solve_for_a_l2386_238661

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem solve_for_a (a : ℤ) : star a 3 = 18 → a = 15 := by
  intro h₁
  sorry

end NUMINAMATH_GPT_solve_for_a_l2386_238661


namespace NUMINAMATH_GPT_selection_ways_l2386_238629

/-- There are a total of 70 ways to select 3 people from 4 teachers and 5 students,
with the condition that there must be at least one teacher and one student among the selected. -/
theorem selection_ways (teachers students : ℕ) (T : 4 = teachers) (S : 5 = students) :
  ∃ (ways : ℕ), ways = 70 := by
  sorry

end NUMINAMATH_GPT_selection_ways_l2386_238629
