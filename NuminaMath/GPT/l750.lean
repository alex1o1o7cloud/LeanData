import Mathlib

namespace subtraction_contradiction_l750_75061

theorem subtraction_contradiction (k t : ℕ) (hk_non_zero : k ≠ 0) (ht_non_zero : t ≠ 0) : 
  ¬ ((8 * 100 + k * 10 + 8) - (k * 100 + 8 * 10 + 8) = 1 * 100 + 6 * 10 + t * 1) :=
by
  sorry

end subtraction_contradiction_l750_75061


namespace parallel_lines_slope_l750_75047

theorem parallel_lines_slope {a : ℝ} 
    (h1 : ∀ x y : ℝ, 4 * y + 3 * x - 5 = 0 → y = -3 / 4 * x + 5 / 4)
    (h2 : ∀ x y : ℝ, 6 * y + a * x + 4 = 0 → y = -a / 6 * x - 2 / 3)
    (h_parallel : ∀ x₁ y₁ x₂ y₂ : ℝ, (4 * y₁ + 3 * x₁ - 5 = 0 ∧ 6 * y₂ + a * x₂ + 4 = 0) → -3 / 4 = -a / 6) : 
  a = 4.5 := sorry

end parallel_lines_slope_l750_75047


namespace hillary_stops_short_of_summit_l750_75058

noncomputable def distance_to_summit_from_base_camp : ℝ := 4700
noncomputable def hillary_climb_rate : ℝ := 800
noncomputable def eddy_climb_rate : ℝ := 500
noncomputable def hillary_descent_rate : ℝ := 1000
noncomputable def time_of_departure : ℝ := 6
noncomputable def time_of_passing : ℝ := 12

theorem hillary_stops_short_of_summit :
  ∃ x : ℝ, 
    (time_of_passing - time_of_departure) * hillary_climb_rate = distance_to_summit_from_base_camp - x →
    (time_of_passing - time_of_departure) * eddy_climb_rate = x →
    x = 2900 :=
by
  sorry

end hillary_stops_short_of_summit_l750_75058


namespace total_bottles_capped_in_10_minutes_l750_75080

-- Define the capacities per minute for the three machines
def machine_a_capacity : ℕ := 12
def machine_b_capacity : ℕ := machine_a_capacity - 2
def machine_c_capacity : ℕ := machine_b_capacity + 5

-- Define the total capping capacity for 10 minutes
def total_capacity_in_10_minutes (a b c : ℕ) : ℕ := a * 10 + b * 10 + c * 10

-- The theorem we aim to prove
theorem total_bottles_capped_in_10_minutes :
  total_capacity_in_10_minutes machine_a_capacity machine_b_capacity machine_c_capacity = 370 :=
by
  -- Directly use the capacities defined above
  sorry

end total_bottles_capped_in_10_minutes_l750_75080


namespace complex_expression_evaluation_l750_75057

theorem complex_expression_evaluation (i : ℂ) (h1 : i^(4 : ℤ) = 1) (h2 : i^(1 : ℤ) = i)
   (h3 : i^(2 : ℤ) = -1) (h4 : i^(3 : ℤ) = -i) (h5 : i^(0 : ℤ) = 1) : 
   i^(245 : ℤ) + i^(246 : ℤ) + i^(247 : ℤ) + i^(248 : ℤ) + i^(249 : ℤ) = i :=
by
  sorry

end complex_expression_evaluation_l750_75057


namespace functional_equation_solution_l750_75098

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  intro h
  sorry

end functional_equation_solution_l750_75098


namespace total_cups_used_l750_75048

theorem total_cups_used (butter flour sugar : ℕ) (h1 : 2 * sugar = 3 * butter) (h2 : 5 * sugar = 3 * flour) (h3 : sugar = 12) : butter + flour + sugar = 40 :=
by
  sorry

end total_cups_used_l750_75048


namespace calculate_regular_rate_l750_75090

def regular_hours_per_week : ℕ := 6 * 10
def total_weeks : ℕ := 4
def total_regular_hours : ℕ := regular_hours_per_week * total_weeks
def total_worked_hours : ℕ := 245
def overtime_hours : ℕ := total_worked_hours - total_regular_hours
def overtime_rate : ℚ := 4.20
def total_earning : ℚ := 525
def total_overtime_pay : ℚ := overtime_hours * overtime_rate
def total_regular_pay : ℚ := total_earning - total_overtime_pay
def regular_rate : ℚ := total_regular_pay / total_regular_hours

theorem calculate_regular_rate : regular_rate = 2.10 :=
by
  -- The proof would go here
  sorry

end calculate_regular_rate_l750_75090


namespace profit_percent_l750_75004

variable (P C : ℝ)
variable (h₁ : (2/3) * P = 0.84 * C)

theorem profit_percent (P C : ℝ) (h₁ : (2/3) * P = 0.84 * C) : 
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end profit_percent_l750_75004


namespace square_number_n_value_l750_75068

theorem square_number_n_value
  (n : ℕ)
  (h : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2) :
  n = 10 :=
sorry

end square_number_n_value_l750_75068


namespace cost_of_paving_is_correct_l750_75051

def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 4
def rate_per_square_meter : ℝ := 950
def area_of_room : ℝ := length_of_room * width_of_room
def cost_of_paving : ℝ := area_of_room * rate_per_square_meter

theorem cost_of_paving_is_correct : cost_of_paving = 20900 := 
by
  sorry

end cost_of_paving_is_correct_l750_75051


namespace logarithmic_relationship_l750_75065

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_relationship
  (h1 : 0 < Real.cos 1)
  (h2 : Real.cos 1 < Real.sin 1)
  (h3 : Real.sin 1 < 1)
  (h4 : 1 < Real.tan 1) :
  log_base (Real.sin 1) (Real.tan 1) < log_base (Real.cos 1) (Real.tan 1) ∧
  log_base (Real.cos 1) (Real.tan 1) < log_base (Real.cos 1) (Real.sin 1) ∧
  log_base (Real.cos 1) (Real.sin 1) < log_base (Real.sin 1) (Real.cos 1) :=
sorry

end logarithmic_relationship_l750_75065


namespace point_relationship_l750_75072

variables {m x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) (m : ℝ) : ℝ :=
  (x + m - 3)*(x - m) + 3

theorem point_relationship 
  (hx1_lt_x2 : x1 < x2)
  (hA : y1 = quadratic_function x1 m)
  (hB : y2 = quadratic_function x2 m)
  (h_sum_lt : x1 + x2 < 3) :
  y1 > y2 :=
sorry

end point_relationship_l750_75072


namespace sum_of_reciprocals_l750_75007

theorem sum_of_reciprocals (x y : ℝ) (h₁ : x + y = 15) (h₂ : x * y = 56) : (1/x) + (1/y) = 15/56 := 
by 
  sorry

end sum_of_reciprocals_l750_75007


namespace find_b10_l750_75079

def sequence_b (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = b (n + 1) + b n

theorem find_b10 (b : ℕ → ℕ) (h0 : ∀ n, b n > 0) (h1 : b 9 = 544) (h2 : sequence_b b) : b 10 = 883 :=
by
  -- We could provide steps of the proof here, but we use 'sorry' to omit the proof content
  sorry

end find_b10_l750_75079


namespace quadratic_root_c_l750_75003

theorem quadratic_root_c (c : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + c = (x + (3/2))^2 - 7/4) → c = 1/2 :=
by
  sorry

end quadratic_root_c_l750_75003


namespace floor_neg_seven_over_four_l750_75011

theorem floor_neg_seven_over_four : Int.floor (- 7 / 4 : ℝ) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l750_75011


namespace necessary_but_not_sufficient_condition_l750_75050

def condition_neq_1_or_neq_2 (a b : ℤ) : Prop :=
  a ≠ 1 ∨ b ≠ 2

def statement_sum_neq_3 (a b : ℤ) : Prop :=
  a + b ≠ 3

theorem necessary_but_not_sufficient_condition :
  ∀ (a b : ℤ), condition_neq_1_or_neq_2 a b → ¬ (statement_sum_neq_3 a b) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l750_75050


namespace round_to_nearest_tenth_l750_75070

theorem round_to_nearest_tenth (x : Float) (h : x = 42.63518) : Float.round (x * 10) / 10 = 42.6 := by
  sorry

end round_to_nearest_tenth_l750_75070


namespace determine_n_l750_75010

theorem determine_n (n : ℕ) (h1 : n > 2020) (h2 : ∃ m : ℤ, (n - 2020) = m^2 * (2120 - n)) : 
  n = 2070 ∨ n = 2100 ∨ n = 2110 := 
sorry

end determine_n_l750_75010


namespace optimal_cylinder_dimensions_l750_75036

variable (R : ℝ)

noncomputable def optimal_cylinder_height : ℝ := (2 * R) / Real.sqrt 3
noncomputable def optimal_cylinder_radius : ℝ := R * Real.sqrt (2 / 3)

theorem optimal_cylinder_dimensions :
  ∃ (h r : ℝ), 
    (h = optimal_cylinder_height R ∧ r = optimal_cylinder_radius R) ∧
    ∀ (h' r' : ℝ), (4 * R^2 = 4 * r'^2 + h'^2) → 
      (h' = optimal_cylinder_height R ∧ r' = optimal_cylinder_radius R) → 
      (π * r' ^ 2 * h' ≤ π * r ^ 2 * h) :=
by
  -- Proof omitted
  sorry

end optimal_cylinder_dimensions_l750_75036


namespace intersection_of_A_and_B_l750_75069

-- Definitions based on conditions
def set_A : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def set_B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Statement of the proof problem
theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -2 ≤ x ∧ x ≤ -1} :=
  sorry

end intersection_of_A_and_B_l750_75069


namespace quadrilateral_area_offset_l750_75046

theorem quadrilateral_area_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h_d : d = 26)
  (h_y : y = 6)
  (h_A : A = 195) :
  A = 1/2 * (x + y) * d → x = 9 :=
by
  sorry

end quadrilateral_area_offset_l750_75046


namespace probability_at_least_one_pen_l750_75023

noncomputable def PAs  := 3/5
noncomputable def PBs  := 2/3
noncomputable def PABs := PAs * PBs

theorem probability_at_least_one_pen : PAs + PBs - PABs = 13 / 15 := by
  sorry

end probability_at_least_one_pen_l750_75023


namespace four_p_minus_three_is_square_l750_75009

theorem four_p_minus_three_is_square
  (n : ℕ) (p : ℕ)
  (hn_pos : n > 1)
  (hp_prime : Prime p)
  (h1 : n ∣ (p - 1))
  (h2 : p ∣ (n^3 - 1)) : ∃ k : ℕ, 4 * p - 3 = k^2 := sorry

end four_p_minus_three_is_square_l750_75009


namespace solve_for_x_l750_75055

theorem solve_for_x (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := by
  sorry

end solve_for_x_l750_75055


namespace compute_xy_l750_75081

theorem compute_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 198) : xy = 5 :=
by
  sorry

end compute_xy_l750_75081


namespace intersecting_graphs_l750_75076

theorem intersecting_graphs (a b c d : ℝ) 
  (h1 : -2 * |1 - a| + b = 4) 
  (h2 : 2 * |1 - c| + d = 4)
  (h3 : -2 * |7 - a| + b = 0) 
  (h4 : 2 * |7 - c| + d = 0) : a + c = 10 := 
sorry

end intersecting_graphs_l750_75076


namespace quadratic_has_real_roots_l750_75002

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 - 4 * x - 2 * k + 8 = 0) ->
  k ≥ 2 :=
by
  sorry

end quadratic_has_real_roots_l750_75002


namespace school_students_l750_75041

theorem school_students (x y : ℕ) (h1 : x + y = 432) (h2 : x - 16 = (y + 16) + 24) : x = 244 ∧ y = 188 := by
  sorry

end school_students_l750_75041


namespace product_sequence_l750_75012

theorem product_sequence : 
  let seq := [1/3, 9/1, 1/27, 81/1, 1/243, 729/1, 1/2187, 6561/1, 1/19683, 59049/1]
  ((seq[0] * seq[1]) * (seq[2] * seq[3]) * (seq[4] * seq[5]) * (seq[6] * seq[7]) * (seq[8] * seq[9])) = 243 :=
by
  sorry

end product_sequence_l750_75012


namespace number_of_times_difference_fits_is_20_l750_75031

-- Definitions for Ralph's pictures
def ralph_wild_animals := 75
def ralph_landscapes := 36
def ralph_family_events := 45
def ralph_cars := 20
def ralph_total_pictures := ralph_wild_animals + ralph_landscapes + ralph_family_events + ralph_cars

-- Definitions for Derrick's pictures
def derrick_wild_animals := 95
def derrick_landscapes := 42
def derrick_family_events := 55
def derrick_cars := 25
def derrick_airplanes := 10
def derrick_total_pictures := derrick_wild_animals + derrick_landscapes + derrick_family_events + derrick_cars + derrick_airplanes

-- Combined total number of pictures
def combined_total_pictures := ralph_total_pictures + derrick_total_pictures

-- Difference in wild animals pictures
def difference_wild_animals := derrick_wild_animals - ralph_wild_animals

-- Number of times the difference fits into the combined total (rounded down)
def times_difference_fits := combined_total_pictures / difference_wild_animals

-- Statement of the problem
theorem number_of_times_difference_fits_is_20 : times_difference_fits = 20 := by
  -- The proof will be written here
  sorry

end number_of_times_difference_fits_is_20_l750_75031


namespace measles_cases_1995_l750_75021

-- Definitions based on the conditions
def initial_cases_1970 : ℕ := 300000
def final_cases_2000 : ℕ := 200
def cases_1990 : ℕ := 1000
def decrease_rate : ℕ := 14950 -- Annual linear decrease from 1970-1990
def a : ℤ := -8 -- Coefficient for the quadratic phase

-- Function modeling the number of cases in the quadratic phase (1990-2000)
def measles_cases (x : ℕ) : ℤ := a * (x - 1990)^2 + cases_1990

-- The statement we want to prove
theorem measles_cases_1995 : measles_cases 1995 = 800 := by
  sorry

end measles_cases_1995_l750_75021


namespace find_number_eq_fifty_l750_75037

theorem find_number_eq_fifty (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 := by 
  sorry

end find_number_eq_fifty_l750_75037


namespace angle_K_is_72_l750_75038

variables {J K L M : ℝ}

/-- Given that $JKLM$ is a trapezoid with parallel sides $\overline{JK}$ and $\overline{LM}$,
and given $\angle J = 3\angle M$, $\angle L = 2\angle K$, $\angle J + \angle K = 180^\circ$,
and $\angle L + \angle M = 180^\circ$, prove that $\angle K = 72^\circ$. -/
theorem angle_K_is_72 {J K L M : ℝ}
  (h1 : J = 3 * M)
  (h2 : L = 2 * K)
  (h3 : J + K = 180)
  (h4 : L + M = 180) :
  K = 72 :=
by
  sorry

end angle_K_is_72_l750_75038


namespace imaginary_part_of_complex_l750_75067

open Complex -- Opens the complex numbers namespace

theorem imaginary_part_of_complex:
  ∀ (a b c d : ℂ), (a = (2 + I) / (1 - I) - (2 - I) / (1 + I)) → (a.im = 3) :=
by
  sorry

end imaginary_part_of_complex_l750_75067


namespace no_real_solution_l750_75075

theorem no_real_solution :
  ∀ x : ℝ, ((x - 4 * x + 15)^2 + 3)^2 + 1 ≠ -|x|^2 :=
by
  intro x
  sorry

end no_real_solution_l750_75075


namespace distinct_real_roots_eq_one_l750_75022

theorem distinct_real_roots_eq_one : 
  (∃ x : ℝ, |x| - 4/x = (3 * |x|) / x) ∧ 
  ¬∃ x1 x2 : ℝ, 
    x1 ≠ x2 ∧ 
    (|x1| - 4/x1 = (3 * |x1|) / x1) ∧ 
    (|x2| - 4/x2 = (3 * |x2|) / x2) :=
sorry

end distinct_real_roots_eq_one_l750_75022


namespace man_rate_in_still_water_l750_75084

-- The conditions
def speed_with_stream : ℝ := 20
def speed_against_stream : ℝ := 4

-- The problem rephrased as a Lean statement
theorem man_rate_in_still_water : 
  (speed_with_stream + speed_against_stream) / 2 = 12 := 
by
  sorry

end man_rate_in_still_water_l750_75084


namespace bears_in_shipment_l750_75091

theorem bears_in_shipment
  (initial_bears : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ)
  (total_bears_after_shipment : ℕ) 
  (initial_bears_eq : initial_bears = 5)
  (shelves_eq : shelves = 2)
  (bears_per_shelf_eq : bears_per_shelf = 6)
  (total_bears_calculation : total_bears_after_shipment = shelves * bears_per_shelf)
  : total_bears_after_shipment - initial_bears = 7 :=
by
  sorry

end bears_in_shipment_l750_75091


namespace cost_of_fencing_is_289_l750_75056

def side_lengths : List ℕ := [10, 20, 15, 18, 12, 22]

def cost_per_meter : List ℚ := [3, 2, 4, 3.5, 2.5, 3]

def cost_of_side (length : ℕ) (rate : ℚ) : ℚ :=
  (length : ℚ) * rate

def total_cost : ℚ :=
  List.zipWith cost_of_side side_lengths cost_per_meter |>.sum

theorem cost_of_fencing_is_289 : total_cost = 289 := by
  sorry

end cost_of_fencing_is_289_l750_75056


namespace original_money_l750_75045

theorem original_money (M : ℕ) (h1 : 3 * M / 8 ≤ M)
  (h2 : 1 * (M - 3 * M / 8) / 5 ≤ M - 3 * M / 8)
  (h3 : M - 3 * M / 8 - (1 * (M - 3 * M / 8) / 5) = 36) : M = 72 :=
sorry

end original_money_l750_75045


namespace super_rare_snake_cost_multiple_l750_75025

noncomputable def price_of_regular_snake : ℕ := 250
noncomputable def total_money_obtained : ℕ := 2250
noncomputable def number_of_snakes : ℕ := 3
noncomputable def eggs_per_snake : ℕ := 2

theorem super_rare_snake_cost_multiple :
  (total_money_obtained - (number_of_snakes * eggs_per_snake - 1) * price_of_regular_snake) / price_of_regular_snake = 4 :=
by
  sorry

end super_rare_snake_cost_multiple_l750_75025


namespace eq_square_sum_five_l750_75083

theorem eq_square_sum_five (a b : ℝ) (i : ℂ) (h : i * i = -1) (h_eq : (a - 2 * i) * i^2013 = b - i) : a^2 + b^2 = 5 :=
by
  -- Proof will be filled in later
  sorry

end eq_square_sum_five_l750_75083


namespace solve_equation_l750_75071

theorem solve_equation (x : ℝ) (h : 3 + 1 / (2 - x) = 2 * (1 / (2 - x))) : x = 5 / 3 := 
  sorry

end solve_equation_l750_75071


namespace express_in_scientific_notation_l750_75095

def scientific_notation (n : ℤ) (x : ℝ) :=
  ∃ (a : ℝ) (b : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^b

theorem express_in_scientific_notation : scientific_notation (-8206000) (-8.206 * 10^6) :=
by
  sorry

end express_in_scientific_notation_l750_75095


namespace solve_frac_difference_of_squares_l750_75086

theorem solve_frac_difference_of_squares :
  (108^2 - 99^2) / 9 = 207 := by
  sorry

end solve_frac_difference_of_squares_l750_75086


namespace polygon_sides_l750_75016

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 4 * 360) : 
  n = 6 :=
by sorry

end polygon_sides_l750_75016


namespace proof_problem_l750_75028

theorem proof_problem (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 + 2 * |y| = 2 * x * y) :
  (x > 0 → x + y > 3) ∧ (x < 0 → x + y < -3) :=
by
  sorry

end proof_problem_l750_75028


namespace sum_of_zeros_gt_two_l750_75077

noncomputable def f (a x : ℝ) := 2 * a * Real.log x + x ^ 2 - 2 * (a + 1) * x

theorem sum_of_zeros_gt_two (a x1 x2 : ℝ) (h_a : -0.5 < a ∧ a < 0)
  (h_fx_zeros : f a x1 = 0 ∧ f a x2 = 0) (h_x_order : x1 < x2) : x1 + x2 > 2 := 
sorry

end sum_of_zeros_gt_two_l750_75077


namespace don_eats_80_pizzas_l750_75032

variable (D Daria : ℝ)

-- Condition 1: Daria consumes 2.5 times the amount of pizza that Don does.
def condition1 : Prop := Daria = 2.5 * D

-- Condition 2: Together, they eat 280 pizzas.
def condition2 : Prop := D + Daria = 280

-- Conclusion: The number of pizzas Don eats is 80.
theorem don_eats_80_pizzas (h1 : condition1 D Daria) (h2 : condition2 D Daria) : D = 80 :=
by
  sorry

end don_eats_80_pizzas_l750_75032


namespace gingerbreads_per_tray_l750_75063

-- Given conditions
def total_baked_gb (x : ℕ) : Prop := 4 * 25 + 3 * x = 160

-- The problem statement
theorem gingerbreads_per_tray (x : ℕ) (h : total_baked_gb x) : x = 20 := 
by sorry

end gingerbreads_per_tray_l750_75063


namespace count_4_letter_words_with_A_l750_75074

-- Define the alphabet set and the properties
def Alphabet : Finset (Char) := {'A', 'B', 'C', 'D', 'E'}
def total_words := (Alphabet.card ^ 4 : ℕ)
def total_words_without_A := (Alphabet.erase 'A').card ^ 4
def total_words_with_A := total_words - total_words_without_A

-- The key theorem to prove
theorem count_4_letter_words_with_A : total_words_with_A = 369 := sorry

end count_4_letter_words_with_A_l750_75074


namespace perimeter_of_triangle_ABC_l750_75015

-- Define the focal points and their radius
def radius : ℝ := 2

-- Define the distances between centers of the tangent circles
def center_distance : ℝ := 2 * radius

-- Define the lengths of the sides of the triangle ABC based on the problem constraints
def AB : ℝ := 2 * radius + 2 * center_distance
def BC : ℝ := 2 * radius + center_distance
def CA : ℝ := 2 * radius + center_distance

-- Define the perimeter calculation
def perimeter : ℝ := AB + BC + CA

-- Theorem stating the actual perimeter of the triangle ABC
theorem perimeter_of_triangle_ABC : perimeter = 28 := by
  sorry

end perimeter_of_triangle_ABC_l750_75015


namespace cost_of_bananas_l750_75042

theorem cost_of_bananas (A B : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = 7) : B = 3 :=
by
  sorry

end cost_of_bananas_l750_75042


namespace machine_B_fewer_bottles_l750_75088

-- Definitions and the main theorem statement
def MachineA_caps_per_minute : ℕ := 12
def MachineC_additional_capacity : ℕ := 5
def total_bottles_in_10_minutes : ℕ := 370

theorem machine_B_fewer_bottles (B : ℕ) 
  (h1 : MachineA_caps_per_minute * 10 + 10 * B + 10 * (B + MachineC_additional_capacity) = total_bottles_in_10_minutes) :
  MachineA_caps_per_minute - B = 2 :=
by
  sorry

end machine_B_fewer_bottles_l750_75088


namespace cos_theta_value_sin_theta_plus_pi_over_3_value_l750_75089

variable (θ : ℝ)
variable (H1 : 0 < θ ∧ θ < π / 2)
variable (H2 : Real.sin θ = 4 / 5)

theorem cos_theta_value : Real.cos θ = 3 / 5 := sorry

theorem sin_theta_plus_pi_over_3_value : 
    Real.sin (θ + π / 3) = (4 + 3 * Real.sqrt 3) / 10 := sorry

end cos_theta_value_sin_theta_plus_pi_over_3_value_l750_75089


namespace jars_proof_l750_75053

def total_plums : ℕ := 240
def exchange_ratio : ℕ := 7
def mangoes_per_jar : ℕ := 5

def ripe_plums (total_plums : ℕ) := total_plums / 4
def unripe_plums (total_plums : ℕ) := 3 * total_plums / 4
def unripe_plums_kept : ℕ := 46

def plums_for_trade (total_plums unripe_plums_kept : ℕ) : ℕ :=
  ripe_plums total_plums + (unripe_plums total_plums - unripe_plums_kept)

def mangoes_received (plums_for_trade exchange_ratio : ℕ) : ℕ :=
  plums_for_trade / exchange_ratio

def jars_of_mangoes (mangoes_received mangoes_per_jar : ℕ) : ℕ :=
  mangoes_received / mangoes_per_jar

theorem jars_proof : jars_of_mangoes (mangoes_received (plums_for_trade total_plums unripe_plums_kept) exchange_ratio) mangoes_per_jar = 5 :=
by
  sorry

end jars_proof_l750_75053


namespace partnership_total_profit_l750_75073

theorem partnership_total_profit
  (total_capital : ℝ)
  (A_share : ℝ := 1/3)
  (B_share : ℝ := 1/4)
  (C_share : ℝ := 1/5)
  (D_share : ℝ := 1 - (A_share + B_share + C_share))
  (A_profit : ℝ := 805)
  (A_capital : ℝ := total_capital * A_share)
  (total_capital_positive : 0 < total_capital)
  (shares_add_up : A_share + B_share + C_share + D_share = 1) :
  (A_profit / (total_capital * A_share)) * total_capital = 2415 :=
by
  -- Proof will go here.
  sorry

end partnership_total_profit_l750_75073


namespace surveyed_parents_women_l750_75034

theorem surveyed_parents_women (W : ℝ) :
  (5/6 : ℝ) * W + (3/4 : ℝ) * (1 - W) = 0.8 → W = 0.6 :=
by
  intro h
  have hw : W * (1/6) + (1 - W) * (1/4) = 0.2 := sorry
  have : W = 0.6 := sorry
  exact this

end surveyed_parents_women_l750_75034


namespace tanner_remaining_money_l750_75014
-- Import the entire Mathlib library

-- Define the conditions using constants
def s_Sep : ℕ := 17
def s_Oct : ℕ := 48
def s_Nov : ℕ := 25
def v_game : ℕ := 49

-- Define the total amount left and prove it equals 41
theorem tanner_remaining_money :
  (s_Sep + s_Oct + s_Nov - v_game) = 41 :=
by { sorry }

end tanner_remaining_money_l750_75014


namespace meaningful_fraction_l750_75052

theorem meaningful_fraction (x : ℝ) : (x - 1 ≠ 0) ↔ (x ≠ 1) :=
by sorry

end meaningful_fraction_l750_75052


namespace area_of_curve_l750_75024

noncomputable def polar_curve (φ : Real) : Real :=
  (1 / 2) + Real.sin φ

noncomputable def area_enclosed_by_polar_curve : Real :=
  2 * ((1 / 2) * ∫ (φ : Real) in (-Real.pi / 2)..(Real.pi / 2), (polar_curve φ) ^ 2)

theorem area_of_curve : area_enclosed_by_polar_curve = (3 * Real.pi) / 4 :=
by
  sorry

end area_of_curve_l750_75024


namespace max_knights_seated_next_to_two_knights_l750_75049

theorem max_knights_seated_next_to_two_knights 
  (total_knights total_samurais total_people knights_with_samurai_on_right : ℕ)
  (h_total_knights : total_knights = 40)
  (h_total_samurais : total_samurais = 10)
  (h_total_people : total_people = total_knights + total_samurais)
  (h_knights_with_samurai_on_right : knights_with_samurai_on_right = 7) :
  ∃ k, k = 32 ∧ ∀ n, (n ≤ total_knights) → (knights_with_samurai_on_right = 7) → (n = 32) :=
by
  sorry

end max_knights_seated_next_to_two_knights_l750_75049


namespace difference_of_squares_division_l750_75039

theorem difference_of_squares_division :
  let a := 121
  let b := 112
  (a^2 - b^2) / 3 = 699 :=
by
  sorry

end difference_of_squares_division_l750_75039


namespace relatively_prime_divisibility_l750_75099

theorem relatively_prime_divisibility (x y : ℕ) (h1 : Nat.gcd x y = 1) (h2 : y^2 * (y - x)^2 ∣ x^2 * (x + y)) :
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 1) :=
sorry

end relatively_prime_divisibility_l750_75099


namespace smallest_value_of_x_l750_75085

theorem smallest_value_of_x : ∃ x, (2 * x^2 + 30 * x - 84 = x * (x + 15)) ∧ (∀ y, (2 * y^2 + 30 * y - 84 = y * (y + 15)) → x ≤ y) ∧ x = -28 := by
  sorry

end smallest_value_of_x_l750_75085


namespace sin_cos_product_l750_75006

theorem sin_cos_product (ϕ : ℝ) (h : Real.tan (ϕ + Real.pi / 4) = 5) : 
  1 / (Real.sin ϕ * Real.cos ϕ) = 13 / 6 :=
by
  sorry

end sin_cos_product_l750_75006


namespace max_value_a7_a14_l750_75092

noncomputable def arithmetic_sequence_max_product (a_1 d : ℝ) : ℝ :=
  let a_7 := a_1 + 6 * d
  let a_14 := a_1 + 13 * d
  a_7 * a_14

theorem max_value_a7_a14 {a_1 d : ℝ} 
  (h : 10 = 2 * a_1 + 19 * d)
  (sum_first_20 : 100 = (10) * (a_1 + a_1 + 19 * d)) :
  arithmetic_sequence_max_product a_1 d = 25 :=
by
  sorry

end max_value_a7_a14_l750_75092


namespace divide_area_into_squares_l750_75059

theorem divide_area_into_squares :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x / y = 4 / 3 ∧ (x^2 + y^2 = 100) ∧ x = 8 ∧ y = 6) := 
by {
  sorry
}

end divide_area_into_squares_l750_75059


namespace set_B_can_form_right_angled_triangle_l750_75035

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l750_75035


namespace a_range_l750_75062

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.log x - (1 / 2) * x^2 + 3 * x

def is_monotonic_on_interval (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a (a + 1), 4 / x - x + 3 > 0

theorem a_range (a : ℝ) :
  is_monotonic_on_interval a → (0 < a ∧ a ≤ 3) :=
by 
  sorry

end a_range_l750_75062


namespace find_c_value_l750_75030

theorem find_c_value (a b : ℝ) (h1 : 12 = (6 / 100) * a) (h2 : 6 = (12 / 100) * b) : b / a = 0.25 :=
by
  sorry

end find_c_value_l750_75030


namespace faye_total_books_l750_75064

def initial_books : ℕ := 34
def books_given_away : ℕ := 3
def books_bought : ℕ := 48

theorem faye_total_books : initial_books - books_given_away + books_bought = 79 :=
by
  sorry

end faye_total_books_l750_75064


namespace arithmetic_sequence_num_terms_l750_75018

theorem arithmetic_sequence_num_terms (a d l : ℕ) (h1 : a = 15) (h2 : d = 4) (h3 : l = 159) :
  ∃ n : ℕ, l = a + (n-1) * d ∧ n = 37 :=
by {
  sorry
}

end arithmetic_sequence_num_terms_l750_75018


namespace find_w_squared_l750_75001

theorem find_w_squared (w : ℝ) (h : (2 * w + 19) ^ 2 = (4 * w + 9) * (3 * w + 13)) :
  w ^ 2 = ((6 + Real.sqrt 524) / 4) ^ 2 :=
sorry

end find_w_squared_l750_75001


namespace sum_of_squares_l750_75097

theorem sum_of_squares (a b : ℕ) (h_side_lengths : 20^2 = a^2 + b^2) : a + b = 28 :=
sorry

end sum_of_squares_l750_75097


namespace arcsin_of_neg_one_l750_75044

theorem arcsin_of_neg_one : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_of_neg_one_l750_75044


namespace domain_of_fraction_is_all_real_l750_75017

theorem domain_of_fraction_is_all_real (k : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 3 * x + 4 * k ≠ 0) ↔ k < -9 / 112 :=
by sorry

end domain_of_fraction_is_all_real_l750_75017


namespace inverse_of_original_l750_75054

-- Definitions based on conditions
def original_proposition : Prop := ∀ (x y : ℝ), x = y → |x| = |y|

def inverse_proposition : Prop := ∀ (x y : ℝ), |x| = |y| → x = y

-- Lean 4 statement
theorem inverse_of_original : original_proposition → inverse_proposition :=
sorry

end inverse_of_original_l750_75054


namespace max_value_abs_expression_l750_75040

noncomputable def circle_eq (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

theorem max_value_abs_expression (x y : ℝ) (h : circle_eq x y) : 
  ∃ t : ℝ, |3 * x + 4 * y - 3| = t ∧ t ≤ 8 :=
sorry

end max_value_abs_expression_l750_75040


namespace Faye_age_correct_l750_75094

def ages (C D E F G : ℕ) : Prop :=
  D = E - 2 ∧
  C = E + 3 ∧
  F = C - 1 ∧
  D = 16 ∧
  G = D - 5

theorem Faye_age_correct (C D E F G : ℕ) (h : ages C D E F G) : F = 20 :=
by {
  sorry
}

end Faye_age_correct_l750_75094


namespace earnings_ratio_l750_75008

-- Definitions for conditions
def jerusha_earnings : ℕ := 68
def total_earnings : ℕ := 85
def lottie_earnings : ℕ := total_earnings - jerusha_earnings

-- Prove that the ratio of Jerusha's earnings to Lottie's earnings is 4:1
theorem earnings_ratio : 
  ∃ (k : ℕ), jerusha_earnings = k * lottie_earnings ∧ (jerusha_earnings + lottie_earnings = total_earnings) ∧ (jerusha_earnings = 68) ∧ (total_earnings = 85) →
  68 / (total_earnings - 68) = 4 := 
by
  sorry

end earnings_ratio_l750_75008


namespace longest_side_of_triangle_l750_75066

theorem longest_side_of_triangle (x : ℝ) (h1 : 8 + (2 * x + 5) + (3 * x + 2) = 40) : 
  max (max 8 (2 * x + 5)) (3 * x + 2) = 17 := 
by 
  -- proof goes here
  sorry

end longest_side_of_triangle_l750_75066


namespace total_cost_for_tickets_l750_75029

-- Definitions given in conditions
def num_students : ℕ := 20
def num_teachers : ℕ := 3
def ticket_cost : ℕ := 5

-- Proof Statement 
theorem total_cost_for_tickets : num_students + num_teachers * ticket_cost = 115 := by
  sorry

end total_cost_for_tickets_l750_75029


namespace sum_of_distinct_products_l750_75082

theorem sum_of_distinct_products (G H : ℕ) (hG : G < 10) (hH : H < 10) :
  (3 * H + 8) % 8 = 0 ∧ ((6 + 2 + 8 + G + 4 + 0 + 9 + 3 + H + 8) % 9 = 0) →
  (G * H = 6 ∨ G * H = 48) →
  6 + 48 = 54 :=
by
  intros _ _
  sorry

end sum_of_distinct_products_l750_75082


namespace integer_ratio_condition_l750_75013

variable (x y : ℝ)

theorem integer_ratio_condition 
  (h : 3 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 6)
  (h_int : ∃ t : ℤ, x = t * y) :
  ∃ t : ℤ, t = -2 :=
by
  sorry

end integer_ratio_condition_l750_75013


namespace negation_of_forall_prop_l750_75043

theorem negation_of_forall_prop :
  ¬ (∀ x : ℝ, x^2 + x > 0) ↔ ∃ x : ℝ, x^2 + x ≤ 0 :=
by
  sorry

end negation_of_forall_prop_l750_75043


namespace cubic_polynomial_solution_l750_75020

noncomputable def q (x : ℝ) : ℝ := - (4 / 3) * x^3 + 6 * x^2 - (50 / 3) * x - (14 / 3)

theorem cubic_polynomial_solution :
  q 1 = -8 ∧ q 2 = -12 ∧ q 3 = -20 ∧ q 4 = -40 := by
  have h₁ : q 1 = -8 := by sorry
  have h₂ : q 2 = -12 := by sorry
  have h₃ : q 3 = -20 := by sorry
  have h₄ : q 4 = -40 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end cubic_polynomial_solution_l750_75020


namespace find_ab_l750_75026

noncomputable def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

theorem find_ab (a b : ℝ) :
  (f 1 a b = 10) ∧ ((3 * 1^2 - 2 * a * 1 - b = 0)) → (a, b) = (-4, 11) ∨ (a, b) = (3, -3) :=
by
  sorry

end find_ab_l750_75026


namespace sqrt7_minus_3_lt_sqrt5_minus_2_l750_75093

theorem sqrt7_minus_3_lt_sqrt5_minus_2:
  (2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3) ∧ (2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) -> 
  Real.sqrt 7 - 3 < Real.sqrt 5 - 2 := by
  sorry

end sqrt7_minus_3_lt_sqrt5_minus_2_l750_75093


namespace total_books_in_classroom_l750_75027

-- Define the given conditions using Lean definitions
def num_children : ℕ := 15
def books_per_child : ℕ := 12
def additional_books : ℕ := 22

-- Define the hypothesis and the corresponding proof statement
theorem total_books_in_classroom : num_children * books_per_child + additional_books = 202 := 
by sorry

end total_books_in_classroom_l750_75027


namespace inequality_C_l750_75060

theorem inequality_C (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
by
  sorry

end inequality_C_l750_75060


namespace escalator_walk_rate_l750_75005

theorem escalator_walk_rate (v : ℝ) : (v + 15) * 10 = 200 → v = 5 := by
  sorry

end escalator_walk_rate_l750_75005


namespace perimeter_difference_l750_75087

-- Define the dimensions of the two figures
def width1 : ℕ := 6
def height1 : ℕ := 3
def width2 : ℕ := 6
def height2 : ℕ := 2

-- Define the perimeters of the two figures
def perimeter1 : ℕ := 2 * (width1 + height1)
def perimeter2 : ℕ := 2 * (width2 + height2)

-- Prove the positive difference in perimeters is 2 units
theorem perimeter_difference : (perimeter1 - perimeter2) = 2 := by
  sorry

end perimeter_difference_l750_75087


namespace eliot_account_balance_l750_75000

variable (A E F : ℝ)

theorem eliot_account_balance
  (h1 : A > E)
  (h2 : F > A)
  (h3 : A - E = (1 : ℝ) / 12 * (A + E))
  (h4 : F - A = (1 : ℝ) / 8 * (F + A))
  (h5 : 1.1 * A = 1.2 * E + 21)
  (h6 : 1.05 * F = 1.1 * A + 40) :
  E = 210 := 
sorry

end eliot_account_balance_l750_75000


namespace polycarp_error_l750_75033

def three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem polycarp_error (a b n : ℕ) (ha : three_digit a) (hb : three_digit b)
  (h : 10000 * a + b = n * a * b) : n = 73 :=
by
  sorry

end polycarp_error_l750_75033


namespace quadratic_two_distinct_real_roots_l750_75078

theorem quadratic_two_distinct_real_roots (k : ℝ) (h1 : k ≠ 0) : 
  (∀ Δ > 0, Δ = (-2)^2 - 4 * k * (-1)) ↔ (k > -1) :=
by
  -- Since Δ = 4 + 4k, we need to show that (4 + 4k > 0) ↔ (k > -1)
  sorry

end quadratic_two_distinct_real_roots_l750_75078


namespace comparison_b_a_c_l750_75096

noncomputable def a : ℝ := Real.sqrt 1.2
noncomputable def b : ℝ := Real.exp 0.1
noncomputable def c : ℝ := 1 + Real.log 1.1

theorem comparison_b_a_c : b > a ∧ a > c :=
by
  unfold a b c
  sorry

end comparison_b_a_c_l750_75096


namespace fly_total_distance_l750_75019

noncomputable def total_distance_traveled (r : ℝ) (d3 : ℝ) : ℝ :=
  let d1 := 2 * r
  let d2 := Real.sqrt (d1^2 - d3^2)
  d1 + d2 + d3

theorem fly_total_distance (r : ℝ) (h_r : r = 60) (d3 : ℝ) (h_d3 : d3 = 90) :
  total_distance_traveled r d3 = 289.37 :=
by
  rw [h_r, h_d3]
  simp [total_distance_traveled]
  sorry

end fly_total_distance_l750_75019
