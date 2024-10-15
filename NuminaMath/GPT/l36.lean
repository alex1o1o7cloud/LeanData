import Mathlib

namespace NUMINAMATH_GPT_aluminum_weight_l36_3624

variable {weight_iron : ℝ}
variable {weight_aluminum : ℝ}
variable {difference : ℝ}

def weight_aluminum_is_correct (weight_iron weight_aluminum difference : ℝ) : Prop := 
  weight_iron = weight_aluminum + difference

theorem aluminum_weight 
  (H1 : weight_iron = 11.17)
  (H2 : difference = 10.33)
  (H3 : weight_aluminum_is_correct weight_iron weight_aluminum difference) : 
  weight_aluminum = 0.84 :=
sorry

end NUMINAMATH_GPT_aluminum_weight_l36_3624


namespace NUMINAMATH_GPT_chinese_character_symmetry_l36_3651

-- Definitions of the characters and their symmetry properties
def is_symmetric (ch : String) : Prop :=
  ch = "喜"

-- Hypotheses (conditions)
def option_A := "喜"
def option_B := "欢"
def option_C := "数"
def option_D := "学"

-- Lean statement to prove the symmetry
theorem chinese_character_symmetry :
  is_symmetric option_A ∧ 
  ¬ is_symmetric option_B ∧ 
  ¬ is_symmetric option_C ∧ 
  ¬ is_symmetric option_D :=
by
  sorry

end NUMINAMATH_GPT_chinese_character_symmetry_l36_3651


namespace NUMINAMATH_GPT_managers_in_sample_l36_3620

-- Definitions based on the conditions
def total_employees : ℕ := 160
def number_salespeople : ℕ := 104
def number_managers : ℕ := 32
def number_logistics : ℕ := 24
def sample_size : ℕ := 20

-- Theorem statement
theorem managers_in_sample : (number_managers * sample_size) / total_employees = 4 := by
  -- Proof omitted, as per the instructions
  sorry

end NUMINAMATH_GPT_managers_in_sample_l36_3620


namespace NUMINAMATH_GPT_graph_of_equation_l36_3615

theorem graph_of_equation (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) := 
by sorry

end NUMINAMATH_GPT_graph_of_equation_l36_3615


namespace NUMINAMATH_GPT_average_percent_decrease_is_35_percent_l36_3603

-- Given conditions
def last_week_small_price_per_pack := 7 / 3
def this_week_small_price_per_pack := 5 / 4
def last_week_large_price_per_pack := 8 / 2
def this_week_large_price_per_pack := 9 / 3

-- Calculate percent decrease for small packs
def small_pack_percent_decrease := ((last_week_small_price_per_pack - this_week_small_price_per_pack) / last_week_small_price_per_pack) * 100

-- Calculate percent decrease for large packs
def large_pack_percent_decrease := ((last_week_large_price_per_pack - this_week_large_price_per_pack) / last_week_large_price_per_pack) * 100

-- Calculate average percent decrease
def average_percent_decrease := (small_pack_percent_decrease + large_pack_percent_decrease) / 2

theorem average_percent_decrease_is_35_percent : average_percent_decrease = 35 := by
  sorry

end NUMINAMATH_GPT_average_percent_decrease_is_35_percent_l36_3603


namespace NUMINAMATH_GPT_fraction_of_number_l36_3600

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_fraction_of_number_l36_3600


namespace NUMINAMATH_GPT_Henry_trays_per_trip_l36_3692

theorem Henry_trays_per_trip (trays1 trays2 trips : ℕ) (h1 : trays1 = 29) (h2 : trays2 = 52) (h3 : trips = 9) :
  (trays1 + trays2) / trips = 9 :=
by
  sorry

end NUMINAMATH_GPT_Henry_trays_per_trip_l36_3692


namespace NUMINAMATH_GPT_required_weekly_hours_approx_27_l36_3645

noncomputable def planned_hours_per_week : ℝ := 25
noncomputable def planned_weeks : ℝ := 15
noncomputable def total_amount : ℝ := 4500
noncomputable def sick_weeks : ℝ := 3
noncomputable def increased_wage_weeks : ℝ := 5
noncomputable def wage_increase_factor : ℝ := 1.5 -- 50%

-- Normal hourly wage
noncomputable def normal_hourly_wage : ℝ := total_amount / (planned_hours_per_week * planned_weeks)

-- Increased hourly wage
noncomputable def increased_hourly_wage : ℝ := normal_hourly_wage * wage_increase_factor

-- Earnings in the last 5 weeks at increased wage
noncomputable def earnings_in_last_5_weeks : ℝ := increased_hourly_wage * planned_hours_per_week * increased_wage_weeks

-- Amount needed before the wage increase
noncomputable def amount_needed_before_wage_increase : ℝ := total_amount - earnings_in_last_5_weeks

-- We have 7 weeks before the wage increase
noncomputable def weeks_before_increase : ℝ := planned_weeks - sick_weeks - increased_wage_weeks

-- New required weekly hours before wage increase
noncomputable def required_weekly_hours : ℝ := amount_needed_before_wage_increase / (normal_hourly_wage * weeks_before_increase)

theorem required_weekly_hours_approx_27 :
  abs (required_weekly_hours - 27) < 1 :=
sorry

end NUMINAMATH_GPT_required_weekly_hours_approx_27_l36_3645


namespace NUMINAMATH_GPT_least_lcm_of_x_and_z_l36_3623

theorem least_lcm_of_x_and_z (x y z : ℕ) (h₁ : Nat.lcm x y = 20) (h₂ : Nat.lcm y z = 28) : 
  ∃ l, l = Nat.lcm x z ∧ l = 35 := 
sorry

end NUMINAMATH_GPT_least_lcm_of_x_and_z_l36_3623


namespace NUMINAMATH_GPT_mod_congruent_integers_l36_3684

theorem mod_congruent_integers (n : ℕ) : (∃ k : ℕ, n = 13 * k + 7 ∧ 7 + 13 * k < 2000) ↔ 
  (7 + 13 * 153 < 2000 ∧ ∀ m : ℕ, (0 ≤ m ∧ m ≤ 153) → n = 13 * m + 7) := by
  sorry

end NUMINAMATH_GPT_mod_congruent_integers_l36_3684


namespace NUMINAMATH_GPT_pitchers_of_lemonade_l36_3677

theorem pitchers_of_lemonade (glasses_per_pitcher : ℕ) (total_glasses_served : ℕ)
  (h1 : glasses_per_pitcher = 5) (h2 : total_glasses_served = 30) :
  total_glasses_served / glasses_per_pitcher = 6 := by
  sorry

end NUMINAMATH_GPT_pitchers_of_lemonade_l36_3677


namespace NUMINAMATH_GPT_total_weight_proof_l36_3634

-- Definitions of the conditions in the problem.
def bags_on_first_trip : ℕ := 10
def common_ratio : ℕ := 2
def number_of_trips : ℕ := 20
def weight_per_bag_kg : ℕ := 50

-- Function to compute the total number of bags transported.
noncomputable def total_number_of_bags : ℕ :=
  bags_on_first_trip * (1 - common_ratio^number_of_trips) / (1 - common_ratio)

-- Function to compute the total weight of onions harvested.
noncomputable def total_weight_of_onions : ℕ :=
  total_number_of_bags * weight_per_bag_kg

-- Theorem stating that the total weight of onions harvested is 524,287,500 kgs.
theorem total_weight_proof : total_weight_of_onions = 524287500 := by
  sorry

end NUMINAMATH_GPT_total_weight_proof_l36_3634


namespace NUMINAMATH_GPT_black_haired_girls_count_l36_3610

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end NUMINAMATH_GPT_black_haired_girls_count_l36_3610


namespace NUMINAMATH_GPT_least_rice_l36_3670

variable (o r : ℝ)

-- Conditions
def condition_1 : Prop := o ≥ 8 + r / 2
def condition_2 : Prop := o ≤ 3 * r

-- The main theorem we want to prove
theorem least_rice (h1 : condition_1 o r) (h2 : condition_2 o r) : r ≥ 4 :=
sorry

end NUMINAMATH_GPT_least_rice_l36_3670


namespace NUMINAMATH_GPT_hulk_jump_exceeds_2000_l36_3674

theorem hulk_jump_exceeds_2000 {n : ℕ} (h : n ≥ 1) :
  2^(n - 1) > 2000 → n = 12 :=
by
  sorry

end NUMINAMATH_GPT_hulk_jump_exceeds_2000_l36_3674


namespace NUMINAMATH_GPT_problem_statement_l36_3678

noncomputable def nonreal_omega_root (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω^2 + ω + 1 = 0

theorem problem_statement (ω : ℂ) (h : nonreal_omega_root ω) :
  (1 - 2 * ω + ω^2)^6 + (1 + 2 * ω - ω^2)^6 = 1458 :=
sorry

end NUMINAMATH_GPT_problem_statement_l36_3678


namespace NUMINAMATH_GPT_calculate_f_ff_f60_l36_3665

def f (N : ℝ) : ℝ := 0.3 * N + 2

theorem calculate_f_ff_f60 : f (f (f 60)) = 4.4 := by
  sorry

end NUMINAMATH_GPT_calculate_f_ff_f60_l36_3665


namespace NUMINAMATH_GPT_sum_common_seq_first_n_l36_3631

def seq1 (n : ℕ) := 2 * n - 1
def seq2 (n : ℕ) := 3 * n - 2

def common_seq (n : ℕ) := 6 * n - 5

def sum_first_n_terms (a : ℕ) (d : ℕ) (n : ℕ) := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_common_seq_first_n (n : ℕ) : 
  sum_first_n_terms 1 6 n = 3 * n^2 - 2 * n := 
by sorry

end NUMINAMATH_GPT_sum_common_seq_first_n_l36_3631


namespace NUMINAMATH_GPT_infection_in_fourth_round_l36_3641

-- Define the initial conditions and the function for the geometric sequence
def initial_infected : ℕ := 1
def infection_ratio : ℕ := 20

noncomputable def infected_computers (rounds : ℕ) : ℕ :=
  initial_infected * infection_ratio^(rounds - 1)

-- The theorem to prove
theorem infection_in_fourth_round : infected_computers 4 = 8000 :=
by
  -- proof will be added later
  sorry

end NUMINAMATH_GPT_infection_in_fourth_round_l36_3641


namespace NUMINAMATH_GPT_IMO1991Q1_l36_3614

theorem IMO1991Q1 (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
    (h4 : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_GPT_IMO1991Q1_l36_3614


namespace NUMINAMATH_GPT_gardener_tree_arrangement_l36_3668

theorem gardener_tree_arrangement :
  let maple_trees := 4
  let oak_trees := 5
  let birch_trees := 6
  let total_trees := maple_trees + oak_trees + birch_trees
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial maple_trees * Nat.factorial oak_trees * Nat.factorial birch_trees)
  let valid_slots := 9  -- as per slots identified in the solution
  let valid_arrangements := 1 * Nat.choose valid_slots oak_trees
  let probability := valid_arrangements / total_arrangements
  probability = 1 / 75075 →
  (1 + 75075) = 75076 := by {
    sorry
  }

end NUMINAMATH_GPT_gardener_tree_arrangement_l36_3668


namespace NUMINAMATH_GPT_b1f_hex_to_dec_l36_3633

/-- 
  Convert the given hexadecimal digit to its corresponding decimal value.
  -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | _ => 0

/-- 
  Convert a hexadecimal string to a decimal number.
  -/
def hex_string_to_dec (s : String) : Nat :=
  s.foldl (λ acc c => acc * 16 + hex_to_dec c) 0

theorem b1f_hex_to_dec : hex_string_to_dec "B1F" = 2847 :=
by
  sorry

end NUMINAMATH_GPT_b1f_hex_to_dec_l36_3633


namespace NUMINAMATH_GPT_parabola_hyperbola_focus_l36_3673

theorem parabola_hyperbola_focus (p : ℝ) (hp : 0 < p) :
  (∃ k : ℝ, y^2 = 2 * k * x ∧ k > 0) ∧ (x^2 - y^2 / 3 = 1) → (p = 4) :=
by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_focus_l36_3673


namespace NUMINAMATH_GPT_grandmaster_plays_21_games_l36_3657

theorem grandmaster_plays_21_games (a : ℕ → ℕ) (n : ℕ) :
  (∀ i, 1 ≤ a (i + 1) - a i) ∧ (∀ i, a (i + 7) - a i ≤ 10) →
  ∃ (i j : ℕ), i < j ∧ (a j - a i = 21) :=
sorry

end NUMINAMATH_GPT_grandmaster_plays_21_games_l36_3657


namespace NUMINAMATH_GPT_find_a_no_solution_l36_3609

noncomputable def no_solution_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (8 * |x - 4 * a| + |x - a^2| + 7 * x - 2 * a = 0)

theorem find_a_no_solution :
  ∀ a : ℝ, no_solution_eq a ↔ (a < -22 ∨ a > 0) :=
by
  intro a
  sorry

end NUMINAMATH_GPT_find_a_no_solution_l36_3609


namespace NUMINAMATH_GPT_range_of_a_l36_3619

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l36_3619


namespace NUMINAMATH_GPT_total_percentage_change_l36_3628

theorem total_percentage_change (X : ℝ) (fall_increase : X' = 1.08 * X) (spring_decrease : X'' = 0.8748 * X) :
  ((X'' - X) / X) * 100 = -12.52 := 
by
  sorry

end NUMINAMATH_GPT_total_percentage_change_l36_3628


namespace NUMINAMATH_GPT_license_plate_palindrome_probability_l36_3627

theorem license_plate_palindrome_probability :
  let p := 507
  let q := 2028
  p + q = 2535 :=
by
  sorry

end NUMINAMATH_GPT_license_plate_palindrome_probability_l36_3627


namespace NUMINAMATH_GPT_other_root_zero_l36_3690

theorem other_root_zero (b : ℝ) (x : ℝ) (hx_root : x^2 + b * x = 0) (h_x_eq_minus_two : x = -2) : 
  (0 : ℝ) = 0 :=
by
  sorry

end NUMINAMATH_GPT_other_root_zero_l36_3690


namespace NUMINAMATH_GPT_geometric_sequence_a3_a5_l36_3698

variable {a : ℕ → ℝ}

theorem geometric_sequence_a3_a5 (h₀ : a 1 > 0) 
                                (h₁ : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 16) : 
                                a 3 + a 5 = 4 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a3_a5_l36_3698


namespace NUMINAMATH_GPT_eval_expression_l36_3617

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l36_3617


namespace NUMINAMATH_GPT_total_sales_correct_l36_3605

-- Define the conditions
def total_tickets : ℕ := 65
def senior_ticket_price : ℕ := 10
def regular_ticket_price : ℕ := 15
def regular_tickets_sold : ℕ := 41

-- Calculate the senior citizen tickets sold
def senior_tickets_sold : ℕ := total_tickets - regular_tickets_sold

-- Calculate the revenue from senior citizen tickets
def revenue_senior : ℕ := senior_ticket_price * senior_tickets_sold

-- Calculate the revenue from regular tickets
def revenue_regular : ℕ := regular_ticket_price * regular_tickets_sold

-- Define the total sales amount
def total_sales_amount : ℕ := revenue_senior + revenue_regular

-- The statement we need to prove
theorem total_sales_correct : total_sales_amount = 855 := by
  sorry

end NUMINAMATH_GPT_total_sales_correct_l36_3605


namespace NUMINAMATH_GPT_sqrt_expr_eq_l36_3680

theorem sqrt_expr_eq : (Real.sqrt 2 + Real.sqrt 3)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 3) = 6 + 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_GPT_sqrt_expr_eq_l36_3680


namespace NUMINAMATH_GPT_find_x_l36_3639

theorem find_x (x y z p q r: ℝ) 
  (h1 : (x * y) / (x + y) = p)
  (h2 : (x * z) / (x + z) = q)
  (h3 : (y * z) / (y + z) = r)
  (hp_nonzero : p ≠ 0)
  (hq_nonzero : q ≠ 0)
  (hr_nonzero : r ≠ 0)
  (hxy : x ≠ -y)
  (hxz : x ≠ -z)
  (hyz : y ≠ -z)
  (hpq : p = 3 * q)
  (hpr : p = 2 * r) : x = 3 * p / 2 := 
sorry

end NUMINAMATH_GPT_find_x_l36_3639


namespace NUMINAMATH_GPT_solve_abs_eq_l36_3616

theorem solve_abs_eq (x : ℝ) : |x - 4| = 3 - x ↔ x = 7 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l36_3616


namespace NUMINAMATH_GPT_march_1_falls_on_friday_l36_3612

-- Definitions of conditions
def march_days : ℕ := 31
def mondays_in_march : ℕ := 4
def thursdays_in_march : ℕ := 4

-- Lean 4 statement to prove March 1 falls on a Friday
theorem march_1_falls_on_friday 
  (h1 : march_days = 31)
  (h2 : mondays_in_march = 4)
  (h3 : thursdays_in_march = 4)
  : ∃ d : ℕ, d = 5 :=
by sorry

end NUMINAMATH_GPT_march_1_falls_on_friday_l36_3612


namespace NUMINAMATH_GPT_solve_special_sine_system_l36_3687

noncomputable def special_sine_conditions1 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  let z := -(Real.pi / 2) + 2 * Real.pi * k
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = (-1)^n * Real.pi / 6 + Real.pi * n ∧
  z = -Real.pi / 2 + 2 * Real.pi * k

noncomputable def special_sine_conditions2 (m n k : ℤ) : Prop :=
  let x := (Real.pi / 2) + 2 * Real.pi * m
  let y := -Real.pi / 2 + 2 * Real.pi * k
  let z := (-1 : ℤ)^n * (Real.pi / 6) + Real.pi * n
  x = Real.pi / 2 + 2 * Real.pi * m ∧
  y = -Real.pi / 2 + 2 * Real.pi * k ∧
  z = (-1)^n * Real.pi / 6 + Real.pi * n

theorem solve_special_sine_system (m n k : ℤ) :
  special_sine_conditions1 m n k ∨ special_sine_conditions2 m n k :=
sorry

end NUMINAMATH_GPT_solve_special_sine_system_l36_3687


namespace NUMINAMATH_GPT_geometric_series_sum_l36_3625

theorem geometric_series_sum :
  2016 * (1 / (1 + (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32))) = 1024 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l36_3625


namespace NUMINAMATH_GPT_reb_min_biking_speed_l36_3655

theorem reb_min_biking_speed (driving_time_minutes driving_speed driving_distance biking_distance_minutes biking_reduction_percentage biking_distance_hours : ℕ) 
  (driving_time_eqn: driving_time_minutes = 45) 
  (driving_speed_eqn: driving_speed = 40) 
  (driving_distance_eqn: driving_distance = driving_speed * driving_time_minutes / 60)
  (biking_reduction_percentage_eqn: biking_reduction_percentage = 20)
  (biking_distance_eqn: biking_distance = driving_distance * (100 - biking_reduction_percentage) / 100)
  (biking_distance_hours_eqn: biking_distance_minutes = 120)
  (biking_hours_eqn: biking_distance_hours = biking_distance_minutes / 60)
  : (biking_distance / biking_distance_hours) ≥ 12 := 
by
  sorry

end NUMINAMATH_GPT_reb_min_biking_speed_l36_3655


namespace NUMINAMATH_GPT_vertex_of_parabola_l36_3648

theorem vertex_of_parabola : 
  (exists (a b: ℝ), ∀ x: ℝ, (a * (x - 1)^2 + b = (x - 1)^2 - 2)) → (1, -2) = (1, -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l36_3648


namespace NUMINAMATH_GPT_circle_area_l36_3618

/-!

# Problem: Prove that the area of the circle defined by the equation \( x^2 + y^2 - 2x + 4y + 1 = 0 \) is \( 4\pi \).
-/

theorem circle_area : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 = 0) →
  ∃ (A : ℝ), A = 4 * Real.pi := 
by
  sorry

end NUMINAMATH_GPT_circle_area_l36_3618


namespace NUMINAMATH_GPT_yellow_tint_percentage_l36_3642

theorem yellow_tint_percentage (V₀ : ℝ) (P₀Y : ℝ) (V_additional : ℝ) 
  (hV₀ : V₀ = 40) (hP₀Y : P₀Y = 0.35) (hV_additional : V_additional = 8) : 
  (100 * ((V₀ * P₀Y + V_additional) / (V₀ + V_additional)) = 45.83) :=
by
  sorry

end NUMINAMATH_GPT_yellow_tint_percentage_l36_3642


namespace NUMINAMATH_GPT_quadratic_less_than_zero_for_x_in_0_1_l36_3669

theorem quadratic_less_than_zero_for_x_in_0_1 (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, 0 < x ∧ x < 1 → (a * x^2 + b * x + c) < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_less_than_zero_for_x_in_0_1_l36_3669


namespace NUMINAMATH_GPT_minimum_value_of_a_l36_3649

theorem minimum_value_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y ≥ 9) : ∃ a > 0, a ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l36_3649


namespace NUMINAMATH_GPT_min_students_in_class_l36_3643

theorem min_students_in_class (b g : ℕ) (hb : 3 * b = 4 * g) : b + g = 7 :=
sorry

end NUMINAMATH_GPT_min_students_in_class_l36_3643


namespace NUMINAMATH_GPT_log_product_eq_one_l36_3681

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_product_eq_one :
  log_base 5 2 * log_base 4 25 = 1 := 
by
  sorry

end NUMINAMATH_GPT_log_product_eq_one_l36_3681


namespace NUMINAMATH_GPT_a11_a12_a13_eq_105_l36_3679

variable (a : ℕ → ℝ) -- Define the arithmetic sequence
variable (d : ℝ) -- Define the common difference

-- Assume the conditions given in step a)
axiom arith_seq (n : ℕ) : a n = a 0 + n * d
axiom sum_3_eq_15 : a 0 + a 1 + a 2 = 15
axiom prod_3_eq_80 : a 0 * a 1 * a 2 = 80
axiom pos_diff : d > 0

theorem a11_a12_a13_eq_105 : a 10 + a 11 + a 12 = 105 :=
sorry

end NUMINAMATH_GPT_a11_a12_a13_eq_105_l36_3679


namespace NUMINAMATH_GPT_probability_red_buttons_l36_3664

/-- 
Initial condition: Jar A contains 6 red buttons and 10 blue buttons.
Carla removes the same number of red buttons as blue buttons from Jar A and places them in Jar B.
Jar A's state after action: Jar A retains 3/4 of its original number of buttons.
Question: What is the probability that both selected buttons are red? Express your answer as a common fraction.
-/
theorem probability_red_buttons :
  let initial_red_a := 6
  let initial_blue_a := 10
  let total_buttons_a := initial_red_a + initial_blue_a
  
  -- Jar A after removing buttons
  let retained_fraction := 3 / 4
  let remaining_buttons_a := retained_fraction * total_buttons_a
  let removed_buttons := total_buttons_a - remaining_buttons_a
  let removed_red_buttons := removed_buttons / 2
  let removed_blue_buttons := removed_buttons / 2
  
  -- Remaining red and blue buttons in Jar A
  let remaining_red_a := initial_red_a - removed_red_buttons
  let remaining_blue_a := initial_blue_a - removed_blue_buttons

  -- Total remaining buttons in Jar A
  let total_remaining_a := remaining_red_a + remaining_blue_a

  -- Jar B contains the removed buttons
  let total_buttons_b := removed_buttons
  
  -- Probability calculations
  let probability_red_a := remaining_red_a / total_remaining_a
  let probability_red_b := removed_red_buttons / total_buttons_b

  -- Combined probability of selecting red button from both jars
  probability_red_a * probability_red_b = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_buttons_l36_3664


namespace NUMINAMATH_GPT_trader_profit_l36_3629

theorem trader_profit
  (CP : ℝ)
  (MP : ℝ)
  (SP : ℝ)
  (h1 : MP = CP * 1.12)
  (discount_percent : ℝ)
  (h2 : discount_percent = 0.09821428571428571)
  (discount : ℝ)
  (h3 : discount = MP * discount_percent)
  (actual_SP : ℝ)
  (h4 : actual_SP = MP - discount)
  (h5 : CP = 100) :
  (actual_SP / CP = 1.01) :=
by
  sorry

end NUMINAMATH_GPT_trader_profit_l36_3629


namespace NUMINAMATH_GPT_find_f7_l36_3656

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x^7 + b * x^3 + c * x - 5

theorem find_f7 (a b c : ℝ) (h : f (-7) a b c = 7) : f 7 a b c = -17 :=
by
  sorry

end NUMINAMATH_GPT_find_f7_l36_3656


namespace NUMINAMATH_GPT_value_of_m_l36_3635

theorem value_of_m (m : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 - m * x + m - 1) 
  (h_eq : f 0 = f 2) : m = 2 :=
sorry

end NUMINAMATH_GPT_value_of_m_l36_3635


namespace NUMINAMATH_GPT_star_problem_l36_3666

def star_problem_proof (p q r s u : ℤ) (S : ℤ): Prop :=
  (S = 64) →
  ({n : ℤ | n = 19 ∨ n = 21 ∨ n = 23 ∨ n = 25 ∨ n = 27} = {p, q, r, s, u}) →
  (p + q + r + s + u = 115) →
  (9 + p + q + 7 = S) →
  (3 + p + u + 15 = S) →
  (3 + q + r + 11 = S) →
  (9 + u + s + 11 = S) →
  (15 + s + r + 7 = S) →
  (q = 27)

theorem star_problem : ∃ p q r s u S, star_problem_proof p q r s u S := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_star_problem_l36_3666


namespace NUMINAMATH_GPT_y_coordinate_of_intersection_l36_3662

def line_eq (x t : ℝ) : ℝ := -2 * x + t

def parabola_eq (x : ℝ) : ℝ := (x - 1) ^ 2 + 1

def intersection_condition (x y t : ℝ) : Prop :=
  y = line_eq x t ∧ y = parabola_eq x ∧ x ≥ 0 ∧ y ≥ 0

theorem y_coordinate_of_intersection (x y : ℝ) (t : ℝ) (h_t : t = 11)
  (h_intersection : intersection_condition x y t) :
  y = 5 := by
  sorry

end NUMINAMATH_GPT_y_coordinate_of_intersection_l36_3662


namespace NUMINAMATH_GPT_new_cylinder_volume_l36_3626

theorem new_cylinder_volume (r h : ℝ) (π_ne_zero : 0 < π) (original_volume : π * r^2 * h = 10) : 
  π * (3 * r)^2 * (2 * h) = 180 :=
by
  sorry

end NUMINAMATH_GPT_new_cylinder_volume_l36_3626


namespace NUMINAMATH_GPT_total_students_in_line_l36_3613

-- Define the conditions
def students_in_front : Nat := 15
def students_behind : Nat := 12

-- Define the statement to prove: total number of students in line is 28
theorem total_students_in_line : students_in_front + 1 + students_behind = 28 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_students_in_line_l36_3613


namespace NUMINAMATH_GPT_speed_with_stream_l36_3661

noncomputable def man_speed_still_water : ℝ := 5
noncomputable def speed_against_stream : ℝ := 4

theorem speed_with_stream :
  ∃ V_s, man_speed_still_water + V_s = 6 :=
by
  use man_speed_still_water - speed_against_stream
  sorry

end NUMINAMATH_GPT_speed_with_stream_l36_3661


namespace NUMINAMATH_GPT_find_a_l36_3647

noncomputable def set_A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem find_a (a : ℝ) (h : 3 ∈ set_A a) : a = -3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l36_3647


namespace NUMINAMATH_GPT_polar_to_rectangular_l36_3689

theorem polar_to_rectangular (r θ : ℝ) (h₁ : r = 5) (h₂ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (5 / 2, -5 * Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_GPT_polar_to_rectangular_l36_3689


namespace NUMINAMATH_GPT_fraction_of_B_l36_3699

theorem fraction_of_B (A B C : ℝ) 
  (h1 : A = (1/3) * (B + C)) 
  (h2 : A = B + 20) 
  (h3 : A + B + C = 720) : 
  B / (A + C) = 2 / 7 :=
  by 
  sorry

end NUMINAMATH_GPT_fraction_of_B_l36_3699


namespace NUMINAMATH_GPT_tournament_total_games_l36_3650

theorem tournament_total_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 4) : 
  (n * (n - 1) / 2) * k = 1740 := by
  -- Given conditions
  have h1 : n = 30 := h_n
  have h2 : k = 4 := h_k

  -- Calculation using provided values
  sorry

end NUMINAMATH_GPT_tournament_total_games_l36_3650


namespace NUMINAMATH_GPT_min_of_quadratic_l36_3659

theorem min_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, x^2 + 7 * x + 3 ≤ y^2 + 7 * y + 3) ∧ x = -7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_of_quadratic_l36_3659


namespace NUMINAMATH_GPT_min_value_l36_3653

theorem min_value (x : ℝ) (h : x > 2) : ∃ y, y = 22 ∧ 
  ∀ z, (z > 2) → (y ≤ (z^2 + 8) / (Real.sqrt (z - 2))) := 
sorry

end NUMINAMATH_GPT_min_value_l36_3653


namespace NUMINAMATH_GPT_exists_tetrahedra_volume_and_face_area_conditions_l36_3695

noncomputable def volume (T : Tetrahedron) : ℝ := sorry
noncomputable def face_area (T : Tetrahedron) : List ℝ := sorry

-- The existence of two tetrahedra such that the volume of T1 > T2 
-- and the area of each face of T1 does not exceed any face of T2.
theorem exists_tetrahedra_volume_and_face_area_conditions :
  ∃ (T1 T2 : Tetrahedron), 
    (volume T1 > volume T2) ∧ 
    (∀ (a1 : ℝ), a1 ∈ face_area T1 → 
      ∃ (a2 : ℝ), a2 ∈ face_area T2 ∧ a2 ≥ a1) :=
sorry

end NUMINAMATH_GPT_exists_tetrahedra_volume_and_face_area_conditions_l36_3695


namespace NUMINAMATH_GPT_boundary_line_f_g_l36_3632

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def g (x : ℝ) : ℝ := 0.5 * (x - 1 / x)

theorem boundary_line_f_g :
  ∀ (x : ℝ), 1 ≤ x → (x - 1) ≤ f x ∧ (g x) ≤ (x - 1) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_boundary_line_f_g_l36_3632


namespace NUMINAMATH_GPT_sum_of_diagonals_l36_3640

def FG : ℝ := 4
def HI : ℝ := 4
def GH : ℝ := 11
def IJ : ℝ := 11
def FJ : ℝ := 15

theorem sum_of_diagonals (x y z : ℝ) (h1 : z^2 = 4 * x + 121) (h2 : z^2 = 11 * y + 16)
  (h3 : x * y = 44 + 15 * z) (h4 : x * z = 4 * z + 225) (h5 : y * z = 11 * z + 60) :
  3 * z + x + y = 90 :=
sorry

end NUMINAMATH_GPT_sum_of_diagonals_l36_3640


namespace NUMINAMATH_GPT_solve_ab_eq_l36_3685

theorem solve_ab_eq:
  ∃ a b : ℝ, (1 + (2 : ℂ) * (Complex.I)) * (a : ℂ) + (b : ℂ) = (2 : ℂ) * (Complex.I) ∧ a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_GPT_solve_ab_eq_l36_3685


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l36_3646

theorem boat_speed_in_still_water (V_b : ℝ) (D : ℝ) (V_s : ℝ) 
  (h1 : V_s = 3) 
  (h2 : D = (V_b + V_s) * 1) 
  (h3 : D = (V_b - V_s) * 1.5) : 
  V_b = 15 := 
by 
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l36_3646


namespace NUMINAMATH_GPT_expression_divisible_by_264_l36_3693

theorem expression_divisible_by_264 (n : ℕ) (h : n > 1) : ∃ k : ℤ, 7^(2*n) - 4^(2*n) - 297 = 264 * k :=
by 
  sorry

end NUMINAMATH_GPT_expression_divisible_by_264_l36_3693


namespace NUMINAMATH_GPT_function_increasing_probability_l36_3658

noncomputable def is_increasing_on_interval (a b : ℤ) : Prop :=
∀ x : ℝ, x > 1 → 2 * a * x - 2 * b > 0

noncomputable def valid_pairs : List (ℤ × ℤ) :=
[(0, -1), (1, -1), (1, 1), (2, -1), (2, 1)]

noncomputable def total_pairs : ℕ :=
3 * 4

noncomputable def probability_of_increasing_function : ℚ :=
(valid_pairs.length : ℚ) / total_pairs

theorem function_increasing_probability :
  probability_of_increasing_function = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_function_increasing_probability_l36_3658


namespace NUMINAMATH_GPT_unknown_number_eq_0_5_l36_3697

theorem unknown_number_eq_0_5 : 
  ∃ x : ℝ, x + ((2 / 3) * (3 / 8) + 4) - (8 / 16) = 4.25 ∧ x = 0.5 :=
by
  use 0.5
  sorry

end NUMINAMATH_GPT_unknown_number_eq_0_5_l36_3697


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l36_3691

-- Define the problem conditions and goal
theorem common_ratio_of_geometric_sequence 
  (a1 : ℝ)  -- nonzero first term
  (h₁ : a1 ≠ 0) -- first term is nonzero
  (r : ℝ)  -- common ratio
  (h₂ : r > 0) -- ratio is positive
  (h₃ : ∀ n m : ℕ, n ≠ m → a1 * r^n ≠ a1 * r^m) -- distinct terms in sequence
  (h₄ : a1 * r * r * r = (a1 * r) * (a1 * r^3) ∧ a1 * r ≠ (a1 * r^4)) -- arithmetic sequence condition
  : r = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l36_3691


namespace NUMINAMATH_GPT_find_function_l36_3652

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_function (f : ℝ → ℝ)
  (cond : ∀ x y z : ℝ, x + y + z = 0 → f (x^3) + (f y)^3 + (f z)^3 = 3 * x * y * z) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_GPT_find_function_l36_3652


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l36_3660

variable (a b c : ℝ)
variable (pos_a : a > 0)
variable (pos_b : b > 0)
variable (pos_c : c > 0)

theorem cyclic_sum_inequality :
  ( (a^3 + b^3) / (a^2 + a * b + b^2) + 
    (b^3 + c^3) / (b^2 + b * c + c^2) + 
    (c^3 + a^3) / (c^2 + c * a + a^2) ) ≥ 
  (2 / 3) * (a + b + c) := 
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l36_3660


namespace NUMINAMATH_GPT_greatest_n_l36_3675

def S := { xy : ℕ × ℕ | ∃ x y : ℕ, xy = (x * y, x + y) }

def in_S (a : ℕ) : Prop := ∃ x y : ℕ, a = x * y * (x + y)

def pow_mod (a b m : ℕ) : ℕ := (a ^ b) % m

def satisfies_condition (a : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → in_S (a + pow_mod 2 k 9)

theorem greatest_n (a : ℕ) (n : ℕ) : 
  satisfies_condition a n → n ≤ 3 :=
sorry

end NUMINAMATH_GPT_greatest_n_l36_3675


namespace NUMINAMATH_GPT_reciprocal_of_neg_2023_l36_3654

theorem reciprocal_of_neg_2023 : (-2023) * (-1 / 2023) = 1 := 
by sorry

end NUMINAMATH_GPT_reciprocal_of_neg_2023_l36_3654


namespace NUMINAMATH_GPT_cats_remained_on_island_l36_3637

theorem cats_remained_on_island : 
  ∀ (n m1 : ℕ), 
  n = 1800 → 
  m1 = 600 → 
  (n - m1) / 2 = 600 → 
  (n - m1) - ((n - m1) / 2) = 600 :=
by sorry

end NUMINAMATH_GPT_cats_remained_on_island_l36_3637


namespace NUMINAMATH_GPT_find_x_l36_3694

theorem find_x (x : ℝ) : 17 + x + 2 * x + 13 = 60 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l36_3694


namespace NUMINAMATH_GPT_inequality_proof_l36_3622

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  b + (1 / a) > a + (1 / b) := 
by sorry

end NUMINAMATH_GPT_inequality_proof_l36_3622


namespace NUMINAMATH_GPT_share_of_a_l36_3696

theorem share_of_a 
  (A B C : ℝ)
  (h1 : A = (2/3) * (B + C))
  (h2 : B = (2/3) * (A + C))
  (h3 : A + B + C = 200) :
  A = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_share_of_a_l36_3696


namespace NUMINAMATH_GPT_minimum_resistors_required_l36_3683

-- Define the grid configuration and the connectivity condition
def isReliableGrid (m : ℕ) (n : ℕ) (failures : Finset (ℕ × ℕ)) : Prop :=
m * n > 9 ∧ (∀ (a b : ℕ), a ≠ b → (a, b) ∉ failures)

-- Minimum number of resistors ensuring connectivity with up to 9 failures
theorem minimum_resistors_required :
  ∃ (m n : ℕ), 5 * 5 = 25 ∧ isReliableGrid 5 5 ∅ :=
by
  let m : ℕ := 5
  let n : ℕ := 5
  have h₁ : m * n = 25 := by rfl
  have h₂ : isReliableGrid 5 5 ∅ := by
    unfold isReliableGrid
    exact ⟨by norm_num, sorry⟩ -- formal proof omitted for brevity
  exact ⟨m, n, h₁, h₂⟩

end NUMINAMATH_GPT_minimum_resistors_required_l36_3683


namespace NUMINAMATH_GPT_rectangle_ratio_l36_3638

open Real

-- Definition of the terms
variables {x y : ℝ}

-- Conditions as per the problem statement
def diagonalSavingsRect (x y : ℝ) := x + y - sqrt (x^2 + y^2) = (2 / 3) * y

-- The ratio of the shorter side to the longer side of the rectangle
theorem rectangle_ratio
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (h : diagonalSavingsRect x y) : x / y = 8 / 9 :=
by
sorry

end NUMINAMATH_GPT_rectangle_ratio_l36_3638


namespace NUMINAMATH_GPT_find_c_l36_3611

-- Definitions of r and s
def r (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

-- Given and proved statement
theorem find_c (c : ℝ) : r (s 2 c) = 11 → c = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_c_l36_3611


namespace NUMINAMATH_GPT_gcd_12m_18n_with_gcd_mn_18_l36_3604

theorem gcd_12m_18n_with_gcd_mn_18 (m n : ℕ) (hm : Nat.gcd m n = 18) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  Nat.gcd (12 * m) (18 * n) = 108 :=
by sorry

end NUMINAMATH_GPT_gcd_12m_18n_with_gcd_mn_18_l36_3604


namespace NUMINAMATH_GPT_empty_can_weight_l36_3602

theorem empty_can_weight (W w : ℝ) :
  (W + 2 * w = 0.6) →
  (W + 5 * w = 0.975) →
  W = 0.35 :=
by sorry

end NUMINAMATH_GPT_empty_can_weight_l36_3602


namespace NUMINAMATH_GPT_tangent_30_degrees_l36_3606

theorem tangent_30_degrees (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) (hA : ∃ α : ℝ, α = 30 ∧ (y / x) = Real.tan (π / 6)) :
  y / x = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_30_degrees_l36_3606


namespace NUMINAMATH_GPT_goldbach_134_l36_3688

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0
noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem goldbach_134 (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum : p + q = 134) (h_diff : p ≠ q) : 
  ∃ (d : ℕ), d = 134 - (2 * p) ∧ d ≤ 128 := 
sorry

end NUMINAMATH_GPT_goldbach_134_l36_3688


namespace NUMINAMATH_GPT_greatest_product_two_integers_sum_2004_l36_3663

theorem greatest_product_two_integers_sum_2004 : 
  (∃ x y : ℤ, x + y = 2004 ∧ x * y = 1004004) :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_two_integers_sum_2004_l36_3663


namespace NUMINAMATH_GPT_percentage_increase_twice_l36_3682

theorem percentage_increase_twice {P : ℝ} (x : ℝ) :
  (P * (1 + x)^2) = (P * (1 + 0.6900000000000001)) →
  x = 0.30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_twice_l36_3682


namespace NUMINAMATH_GPT_best_possible_overall_standing_l36_3607

noncomputable def N : ℕ := 100 -- number of participants
noncomputable def M : ℕ := 14  -- number of stages

-- Define a competitor finishing 93rd in each stage
def finishes_93rd_each_stage (finishes : ℕ → ℕ) : Prop :=
  ∀ i, i < M → finishes i = 93

-- Define the best possible overall standing
theorem best_possible_overall_standing
  (finishes : ℕ → ℕ) -- function representing stage finishes for the competitor
  (h : finishes_93rd_each_stage finishes) :
  ∃ k, k = 2 := 
sorry

end NUMINAMATH_GPT_best_possible_overall_standing_l36_3607


namespace NUMINAMATH_GPT_find_number_l36_3667

theorem find_number (x : ℝ) (h: x - (3 / 5) * x = 58) : x = 145 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l36_3667


namespace NUMINAMATH_GPT_birds_joined_l36_3671

variable (initialBirds : ℕ) (totalBirds : ℕ)

theorem birds_joined (h1 : initialBirds = 2) (h2 : totalBirds = 6) : (totalBirds - initialBirds) = 4 :=
by
  sorry

end NUMINAMATH_GPT_birds_joined_l36_3671


namespace NUMINAMATH_GPT_alpha_beta_square_eq_eight_l36_3676

open Real

theorem alpha_beta_square_eq_eight :
  ∃ α β : ℝ, 
  (∀ x : ℝ, x^2 - 2 * x - 1 = 0 ↔ x = α ∨ x = β) → 
  (α ≠ β) → 
  (α - β)^2 = 8 :=
sorry

end NUMINAMATH_GPT_alpha_beta_square_eq_eight_l36_3676


namespace NUMINAMATH_GPT_max_possible_x_l36_3621

theorem max_possible_x (x y z : ℝ) 
  (h1 : 3 * x + 2 * y + z = 10)
  (h2 : x * y + x * z + y * z = 6) :
  x ≤ 2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_GPT_max_possible_x_l36_3621


namespace NUMINAMATH_GPT_number_of_valid_subsets_l36_3644

theorem number_of_valid_subsets (n : ℕ) :
  let total      := 16^n
  let invalid1   := 3 * 12^n
  let invalid2   := 2 * 10^n
  let invalidAll := 8^n
  let valid      := total - invalid1 + invalid2 + 9^n - invalidAll
  valid = 16^n - 3 * 12^n + 2 * 10^n + 9^n - 8^n :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_number_of_valid_subsets_l36_3644


namespace NUMINAMATH_GPT_hexagon_perimeter_l36_3672

theorem hexagon_perimeter (s : ℕ) (P : ℕ) (h1 : s = 8) (h2 : 6 > 0) 
                          (h3 : P = 6 * s) : P = 48 := by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l36_3672


namespace NUMINAMATH_GPT_dot_product_neg_vec_n_l36_3601

-- Vector definitions
def vec_m : ℝ × ℝ := (2, -1)
def vec_n : ℝ × ℝ := (3, 2)
def neg_vec_n : ℝ × ℝ := (-vec_n.1, -vec_n.2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Proof statement
theorem dot_product_neg_vec_n :
  dot_product vec_m neg_vec_n = -4 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_dot_product_neg_vec_n_l36_3601


namespace NUMINAMATH_GPT_horner_evaluation_l36_3630

def f (x : ℝ) := x^5 + 3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 11

theorem horner_evaluation : f 4 = 1559 := by
  sorry

end NUMINAMATH_GPT_horner_evaluation_l36_3630


namespace NUMINAMATH_GPT_time_to_ascend_non_working_escalator_l36_3608

-- Definitions from the conditions
def length_of_escalator := 1
def time_standing := 1
def time_running := 24 / 60
def escalator_speed := 1 / 60
def gavrila_speed := 1 / 40

-- The proof problem statement 
theorem time_to_ascend_non_working_escalator 
  (length_of_escalator : ℝ)
  (time_standing : ℝ)
  (time_running : ℝ)
  (escalator_speed : ℝ)
  (gavrila_speed : ℝ) :
  time_standing = 1 →
  time_running = 24 / 60 →
  escalator_speed = 1 / 60 →
  gavrila_speed = 1 / 40 →
  length_of_escalator = 1 →
  1 / gavrila_speed = 40 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end NUMINAMATH_GPT_time_to_ascend_non_working_escalator_l36_3608


namespace NUMINAMATH_GPT_larger_number_is_84_l36_3686

theorem larger_number_is_84 (x y : ℕ) (HCF LCM : ℕ)
  (h_hcf : HCF = 84)
  (h_lcm : LCM = 21)
  (h_ratio : x * 4 = y)
  (h_product : x * y = HCF * LCM) :
  y = 84 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_84_l36_3686


namespace NUMINAMATH_GPT_problem_eight_sided_polygon_interiors_l36_3636

-- Define the condition of the problem
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- One interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℚ := sum_of_interior_angles n / n

-- The main theorem stating the problem
theorem problem_eight_sided_polygon_interiors (n : ℕ) (h1: diagonals_from_vertex n = 5) : 
  one_interior_angle n = 135 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_problem_eight_sided_polygon_interiors_l36_3636
