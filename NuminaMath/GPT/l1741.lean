import Mathlib

namespace stories_in_building_l1741_174127

-- Definitions of the conditions
def apartments_per_floor := 4
def people_per_apartment := 2
def total_people := 200

-- Definition of people per floor
def people_per_floor := apartments_per_floor * people_per_apartment

-- The theorem stating the desired conclusion
theorem stories_in_building :
  total_people / people_per_floor = 25 :=
by
  -- Insert the proof here
  sorry

end stories_in_building_l1741_174127


namespace sqrt_expr_equals_sum_l1741_174194

theorem sqrt_expr_equals_sum :
  ∃ x y z : ℤ,
    (x + y * Int.sqrt z = Real.sqrt (77 + 28 * Real.sqrt 3)) ∧
    (x^2 + y^2 * z = 77) ∧
    (2 * x * y = 28) ∧
    (x + y + z = 16) :=
by
  sorry

end sqrt_expr_equals_sum_l1741_174194


namespace jonah_walked_8_miles_l1741_174141

def speed : ℝ := 4
def time : ℝ := 2
def distance (s t : ℝ) : ℝ := s * t

theorem jonah_walked_8_miles : distance speed time = 8 := sorry

end jonah_walked_8_miles_l1741_174141


namespace initial_velocity_calculation_l1741_174136

-- Define conditions
def acceleration_due_to_gravity := 10 -- m/s^2
def time_to_highest_point := 2 -- s
def velocity_at_highest_point := 0 -- m/s
def initial_observed_acceleration := 15 -- m/s^2

-- Theorem to prove the initial velocity
theorem initial_velocity_calculation
  (a_gravity : ℝ := acceleration_due_to_gravity)
  (t_highest : ℝ := time_to_highest_point)
  (v_highest : ℝ := velocity_at_highest_point)
  (a_initial : ℝ := initial_observed_acceleration) :
  ∃ (v_initial : ℝ), v_initial = 30 := 
sorry

end initial_velocity_calculation_l1741_174136


namespace tan_of_tan_squared_2025_deg_l1741_174145

noncomputable def tan_squared (x : ℝ) : ℝ := (Real.tan x) ^ 2

theorem tan_of_tan_squared_2025_deg : 
  Real.tan (tan_squared (2025 * Real.pi / 180)) = Real.tan (Real.pi / 180) :=
by
  sorry

end tan_of_tan_squared_2025_deg_l1741_174145


namespace find_y_l1741_174138

theorem find_y
  (XYZ_is_straight_line : XYZ_is_straight_line)
  (angle_XYZ : ℝ)
  (angle_YWZ : ℝ)
  (y : ℝ)
  (exterior_angle_theorem : angle_XYZ = y + angle_YWZ)
  (h1 : angle_XYZ = 150)
  (h2 : angle_YWZ = 58) :
  y = 92 :=
by
  sorry

end find_y_l1741_174138


namespace problem_l1741_174169

variable {a b c : ℝ} -- Introducing variables a, b, c as real numbers

-- Conditions:
-- a, b, c are distinct positive real numbers
def distinct_pos (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a 

theorem problem (h : distinct_pos a b c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
sorry 

end problem_l1741_174169


namespace masha_guessed_number_l1741_174164

theorem masha_guessed_number (a b : ℕ) (h1 : a + b = 2002 ∨ a * b = 2002)
  (h2 : ∀ x y, x + y = 2002 → x ≠ 1001 → y ≠ 1001)
  (h3 : ∀ x y, x * y = 2002 → x ≠ 1001 → y ≠ 1001) :
  b = 1001 :=
by {
  sorry
}

end masha_guessed_number_l1741_174164


namespace daily_evaporation_rate_l1741_174126

/-- A statement that verifies the daily water evaporation rate -/
theorem daily_evaporation_rate
  (initial_water : ℝ)
  (evaporation_percentage : ℝ)
  (evaporation_period : ℕ) :
  initial_water = 15 →
  evaporation_percentage = 0.05 →
  evaporation_period = 15 →
  (evaporation_percentage * initial_water / evaporation_period) = 0.05 :=
by
  intros h_water h_percentage h_period
  sorry

end daily_evaporation_rate_l1741_174126


namespace number_of_players_taking_mathematics_l1741_174102

-- Define the conditions
def total_players := 15
def players_physics := 10
def players_both := 4

-- Define the conclusion to be proven
theorem number_of_players_taking_mathematics : (total_players - players_physics + players_both) = 9 :=
by
  -- Placeholder for proof
  sorry

end number_of_players_taking_mathematics_l1741_174102


namespace cafeteria_extra_fruits_l1741_174105

def extra_fruits (ordered wanted : Nat) : Nat :=
  ordered - wanted

theorem cafeteria_extra_fruits :
  let red_apples_ordered := 6
  let red_apples_wanted := 5
  let green_apples_ordered := 15
  let green_apples_wanted := 8
  let oranges_ordered := 10
  let oranges_wanted := 6
  let bananas_ordered := 8
  let bananas_wanted := 7
  extra_fruits red_apples_ordered red_apples_wanted = 1 ∧
  extra_fruits green_apples_ordered green_apples_wanted = 7 ∧
  extra_fruits oranges_ordered oranges_wanted = 4 ∧
  extra_fruits bananas_ordered bananas_wanted = 1 := 
by
  sorry

end cafeteria_extra_fruits_l1741_174105


namespace g_ln_1_div_2017_l1741_174186

open Real

-- Define the functions fulfilling the given conditions
variables (f g : ℝ → ℝ) (a : ℝ)

-- Define assumptions as required by the conditions
axiom f_property : ∀ m n : ℝ, f (m + n) = f m + f n - 1
axiom g_def : ∀ x : ℝ, g x = f x + a^x / (a^x + 1)
axiom a_property : a > 0 ∧ a ≠ 1
axiom g_ln_2017 : g (log 2017) = 2018

-- The theorem to prove
theorem g_ln_1_div_2017 : g (log (1 / 2017)) = -2015 := by
  sorry

end g_ln_1_div_2017_l1741_174186


namespace part1_part2_l1741_174128

variable (x k : ℝ)

-- Part (1)
theorem part1 (h1 : x = 3) : ∀ k : ℝ, (1 + k) * 3 ≤ k^2 + k + 4 := sorry

-- Part (2)
theorem part2 (h2 : ∀ k : ℝ, -4 ≤ k → (1 + k) * x ≤ k^2 + k + 4) : -5 ≤ x ∧ x ≤ 3 := sorry

end part1_part2_l1741_174128


namespace area_of_ABCD_l1741_174159

theorem area_of_ABCD (area_AMOP area_CNOQ : ℝ) 
  (h1: area_AMOP = 8) (h2: area_CNOQ = 24.5) : 
  ∃ (area_ABCD : ℝ), area_ABCD = 60.5 :=
by
  sorry

end area_of_ABCD_l1741_174159


namespace regression_equation_pos_corr_l1741_174181

noncomputable def linear_regression (x y : ℝ) : ℝ := 0.4 * x + 2.5

theorem regression_equation_pos_corr (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (mean_x : ℝ := 2.5) (mean_y : ℝ := 3.5)
    (pos_corr : x * y > 0)
    (cond1 : mean_x = 2.5)
    (cond2 : mean_y = 3.5) :
    linear_regression mean_x mean_y = mean_y :=
by
  sorry

end regression_equation_pos_corr_l1741_174181


namespace minimum_value_ab_l1741_174103

theorem minimum_value_ab (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h : a * b - 2 * a - b = 0) :
  8 ≤ a * b :=
by sorry

end minimum_value_ab_l1741_174103


namespace percent_fair_hair_l1741_174149

theorem percent_fair_hair 
  (total_employees : ℕ) 
  (percent_women_fair_hair : ℝ) 
  (percent_fair_hair_women : ℝ)
  (total_women_fair_hair : ℕ)
  (total_fair_hair : ℕ)
  (h1 : percent_women_fair_hair = 30 / 100)
  (h2 : percent_fair_hair_women = 40 / 100)
  (h3 : total_women_fair_hair = percent_women_fair_hair * total_employees)
  (h4 : percent_fair_hair_women * total_fair_hair = total_women_fair_hair)
  : total_fair_hair = 75 / 100 * total_employees := 
by
  sorry

end percent_fair_hair_l1741_174149


namespace bill_face_value_l1741_174154

theorem bill_face_value
  (TD : ℝ) (T : ℝ) (r : ℝ) (FV : ℝ)
  (h1 : TD = 210)
  (h2 : T = 0.75)
  (h3 : r = 0.16) :
  FV = 1960 :=
by 
  sorry

end bill_face_value_l1741_174154


namespace exercise_felt_weight_l1741_174166

variable (n w : ℕ)
variable (p : ℝ)

def total_weight (n : ℕ) (w : ℕ) : ℕ := n * w

def felt_weight (total_weight : ℕ) (p : ℝ) : ℝ := total_weight * (1 + p)

theorem exercise_felt_weight (h1 : n = 10) (h2 : w = 30) (h3 : p = 0.20) : 
  felt_weight (total_weight n w) p = 360 :=
by 
  sorry

end exercise_felt_weight_l1741_174166


namespace sum_is_zero_l1741_174120

theorem sum_is_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) = 0 :=
by
  sorry

end sum_is_zero_l1741_174120


namespace max_profit_at_90_l1741_174146

-- Definitions for conditions
def fixed_cost : ℝ := 5
def price_per_unit : ℝ := 100

noncomputable def variable_cost (x : ℕ) : ℝ :=
  if h : x < 80 then
    0.5 * x^2 + 40 * x
  else
    101 * x + 8100 / x - 2180

-- Definition of the profit function
noncomputable def profit (x : ℕ) : ℝ :=
  if h : x < 80 then
    -0.5 * x^2 + 60 * x - fixed_cost
  else
    1680 - x - 8100 / x

-- Maximum profit occurs at x = 90
theorem max_profit_at_90 : ∀ x : ℕ, profit 90 ≥ profit x := 
by {
  sorry
}

end max_profit_at_90_l1741_174146


namespace percentage_increase_l1741_174193

variables (A B C D E : ℝ)
variables (A_inc B_inc C_inc D_inc E_inc : ℝ)

-- Conditions
def conditions (A_inc B_inc C_inc D_inc E_inc : ℝ) :=
  A_inc = 0.1 * A ∧
  B_inc = (1/15) * B ∧
  C_inc = 0.05 * C ∧
  D_inc = 0.04 * D ∧
  E_inc = (1/30) * E ∧
  B = 1.5 * A ∧
  C = 2 * A ∧
  D = 2.5 * A ∧
  E = 3 * A

-- Theorem to prove
theorem percentage_increase (A B C D E : ℝ) (A_inc B_inc C_inc D_inc E_inc : ℝ) :
  conditions A B C D E A_inc B_inc C_inc D_inc E_inc →
  (A_inc + B_inc + C_inc + D_inc + E_inc) / (A + B + C + D + E) = 0.05 :=
by
  sorry

end percentage_increase_l1741_174193


namespace man_l1741_174104

theorem man's_speed_kmph (length_train : ℝ) (time_seconds : ℝ) (speed_train_kmph : ℝ) : ℝ :=
  let speed_train_mps := speed_train_kmph * (5/18)
  let rel_speed_mps := length_train / time_seconds
  let man_speed_mps := rel_speed_mps - speed_train_mps
  man_speed_mps * (18/5)

example : man's_speed_kmph 120 6 65.99424046076315 = 6.00735873483709 := by
  sorry

end man_l1741_174104


namespace maximum_value_of_a_l1741_174188

theorem maximum_value_of_a
  (a b c d : ℝ)
  (h1 : b + c + d = 3 - a)
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) :
  a ≤ 2 := by
  sorry

end maximum_value_of_a_l1741_174188


namespace checkerboards_that_cannot_be_covered_l1741_174172

-- Define the dimensions of the checkerboards
def checkerboard_4x6 := (4, 6)
def checkerboard_3x7 := (3, 7)
def checkerboard_5x5 := (5, 5)
def checkerboard_7x4 := (7, 4)
def checkerboard_5x6 := (5, 6)

-- Define a function to calculate the number of squares
def num_squares (dims : Nat × Nat) : Nat := dims.1 * dims.2

-- Define a function to check if a board can be exactly covered by dominoes
def can_be_covered_by_dominoes (dims : Nat × Nat) : Bool := (num_squares dims) % 2 == 0

-- Statement to be proven
theorem checkerboards_that_cannot_be_covered :
  ¬ can_be_covered_by_dominoes checkerboard_3x7 ∧ ¬ can_be_covered_by_dominoes checkerboard_5x5 :=
by
  sorry

end checkerboards_that_cannot_be_covered_l1741_174172


namespace total_area_of_storage_units_l1741_174134

theorem total_area_of_storage_units (total_units remaining_units : ℕ) 
    (size_8_by_4 length width unit_area_200 : ℕ)
    (h1 : total_units = 42)
    (h2 : remaining_units = 22)
    (h3 : length = 8)
    (h4 : width = 4)
    (h5 : unit_area_200 = 200) 
    (h6 : ∀ i : ℕ, i < 20 → unit_area_8_by_4 = length * width) 
    (h7 : ∀ j : ℕ, j < 22 → unit_area_200 = 200) :
    total_area_of_all_units = 5040 :=
by
  let unit_area_8_by_4 := length * width
  let total_area_20_units := 20 * unit_area_8_by_4
  let total_area_22_units := 22 * unit_area_200
  let total_area_of_all_units := total_area_20_units + total_area_22_units
  sorry

end total_area_of_storage_units_l1741_174134


namespace no_integer_solution_for_Px_eq_x_l1741_174182

theorem no_integer_solution_for_Px_eq_x (P : ℤ → ℤ) (hP_int_coeff : ∀ n : ℤ, ∃ k : ℤ, P n = k * n + k) 
  (hP3 : P 3 = 4) (hP4 : P 4 = 3) :
  ¬ ∃ x : ℤ, P x = x := 
by 
  sorry

end no_integer_solution_for_Px_eq_x_l1741_174182


namespace barbed_wire_cost_l1741_174122

noncomputable def total_cost_barbed_wire (area : ℕ) (cost_per_meter : ℝ) (gate_width : ℕ) : ℝ :=
  let s := Real.sqrt area
  let perimeter := 4 * s - 2 * gate_width
  perimeter * cost_per_meter

theorem barbed_wire_cost :
  total_cost_barbed_wire 3136 3.5 1 = 777 := by
  sorry

end barbed_wire_cost_l1741_174122


namespace calculate_treatment_received_l1741_174113

variable (drip_rate : ℕ) (duration_hours : ℕ) (drops_convert : ℕ) (ml_convert : ℕ)

theorem calculate_treatment_received (h1 : drip_rate = 20) (h2 : duration_hours = 2) 
    (h3 : drops_convert = 100) (h4 : ml_convert = 5) : 
    (drip_rate * (duration_hours * 60) * ml_convert) / drops_convert = 120 := 
by
  sorry

end calculate_treatment_received_l1741_174113


namespace ratio_of_numbers_l1741_174187

theorem ratio_of_numbers (A B : ℕ) (h_lcm : Nat.lcm A B = 48) (h_hcf : Nat.gcd A B = 4) : A / 4 = 3 ∧ B / 4 = 4 :=
sorry

end ratio_of_numbers_l1741_174187


namespace symmetry_graph_l1741_174196

theorem symmetry_graph (θ:ℝ) (hθ: θ > 0):
  (∀ k: ℤ, 2 * (3 * Real.pi / 4) + (Real.pi / 3) - 2 * θ = k * Real.pi + Real.pi / 2) 
  → θ = Real.pi / 6 :=
by 
  sorry

end symmetry_graph_l1741_174196


namespace inequality_D_holds_l1741_174114

theorem inequality_D_holds (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
sorry

end inequality_D_holds_l1741_174114


namespace edge_length_of_cube_l1741_174175

theorem edge_length_of_cube {V_cube V_cuboid : ℝ} (base_area : ℝ) (height : ℝ)
  (h1 : base_area = 10) (h2 : height = 73) (h3 : V_cube = V_cuboid - 1)
  (h4 : V_cuboid = base_area * height) :
  ∃ (a : ℝ), a^3 = V_cube ∧ a = 9 :=
by
  /- The proof is omitted -/
  sorry

end edge_length_of_cube_l1741_174175


namespace second_set_length_is_correct_l1741_174129

variables (first_set_length second_set_length : ℝ)

theorem second_set_length_is_correct 
  (h1 : first_set_length = 4)
  (h2 : second_set_length = 5 * first_set_length) : 
  second_set_length = 20 := 
by 
  sorry

end second_set_length_is_correct_l1741_174129


namespace num_geography_books_l1741_174118

theorem num_geography_books
  (total_books : ℕ)
  (history_books : ℕ)
  (math_books : ℕ)
  (h1 : total_books = 100)
  (h2 : history_books = 32)
  (h3 : math_books = 43) :
  total_books - history_books - math_books = 25 :=
by
  sorry

end num_geography_books_l1741_174118


namespace expense_recording_l1741_174170

-- Define the recording of income and expenses
def record_income (amount : Int) : Int := amount
def record_expense (amount : Int) : Int := -amount

-- Given conditions
def income_example := record_income 500
def expense_example := record_expense 400

-- Prove that an expense of 400 yuan is recorded as -400 yuan
theorem expense_recording : record_expense 400 = -400 :=
  by sorry

end expense_recording_l1741_174170


namespace total_num_problems_eq_30_l1741_174153

-- Define the conditions
def test_points : ℕ := 100
def points_per_3_point_problem : ℕ := 3
def points_per_4_point_problem : ℕ := 4
def num_4_point_problems : ℕ := 10

-- Define the number of 3-point problems
def num_3_point_problems : ℕ :=
  (test_points - num_4_point_problems * points_per_4_point_problem) / points_per_3_point_problem

-- Prove the total number of problems is 30
theorem total_num_problems_eq_30 :
  num_3_point_problems + num_4_point_problems = 30 := 
sorry

end total_num_problems_eq_30_l1741_174153


namespace exists_consecutive_divisible_by_cube_l1741_174177

theorem exists_consecutive_divisible_by_cube (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ j : ℕ, j < k → ∃ m : ℕ, 1 < m ∧ (n + j) % (m^3) = 0 := 
sorry

end exists_consecutive_divisible_by_cube_l1741_174177


namespace smallest_n_for_roots_of_unity_l1741_174198

theorem smallest_n_for_roots_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ (n : ℕ), n = 18 ∧ z^n = 1 :=
by {
  sorry
}

end smallest_n_for_roots_of_unity_l1741_174198


namespace find_B_l1741_174142

theorem find_B (A B C : ℝ) (h : ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -1 → 
    2 / ((x-7)*(x+1)^2) = A / (x-7) + B / (x+1) + C / (x+1)^2) : 
  B = 1 / 16 :=
sorry

end find_B_l1741_174142


namespace chord_bisection_l1741_174124

theorem chord_bisection {r : ℝ} (PQ RS : Set (ℝ × ℝ)) (O T P Q R S M : ℝ × ℝ)
  (radius_OP : dist O P = 6) (radius_OQ : dist O Q = 6)
  (radius_OR : dist O R = 6) (radius_OS : dist O S = 6) (radius_OT : dist O T = 6)
  (radius_OM : dist O M = 2 * Real.sqrt 13) 
  (PT_eq_8 : dist P T = 8) (TQ_eq_8 : dist T Q = 8)
  (sin_theta_eq_4_5 : Real.sin (Real.arcsin (8 / 10)) = 4 / 5) :
  4 * 5 = 20 :=
by
  sorry

end chord_bisection_l1741_174124


namespace find_xy_l1741_174161

theorem find_xy (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 :=
by
  sorry

end find_xy_l1741_174161


namespace people_later_than_yoongi_l1741_174192

variable (total_students : ℕ) (people_before_yoongi : ℕ)

theorem people_later_than_yoongi
    (h1 : total_students = 20)
    (h2 : people_before_yoongi = 11) :
    total_students - (people_before_yoongi + 1) = 8 := 
sorry

end people_later_than_yoongi_l1741_174192


namespace quadratic_vertex_transform_l1741_174148

theorem quadratic_vertex_transform {p q r m k : ℝ} (h : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - h)^2 + k) →
  h = -3 :=
by
  intros h1 h2
  -- The actual proof goes here
  sorry

end quadratic_vertex_transform_l1741_174148


namespace not_divisible_by_5_l1741_174152

theorem not_divisible_by_5 (n : ℤ) : ¬ (n^2 - 8) % 5 = 0 :=
by sorry

end not_divisible_by_5_l1741_174152


namespace emily_workers_needed_l1741_174155

noncomputable def least_workers_needed
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ) :
  ℕ :=
  (remaining_work / remaining_days) / (work_done / initial_days / total_workers) * total_workers

theorem emily_workers_needed 
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 40)
  (h2 : initial_days = 10)
  (h3 : total_workers = 12)
  (h4 : work_done = 40)
  (h5 : remaining_work = 60)
  (h6 : remaining_days = 30) :
  least_workers_needed total_days initial_days total_workers work_done remaining_work remaining_days = 6 := 
sorry

end emily_workers_needed_l1741_174155


namespace police_arrangements_l1741_174110

theorem police_arrangements (officers : Fin 5) (A B : Fin 5) (intersections : Fin 3) :
  A ≠ B →
  (∃ arrangement : Fin 5 → Fin 3, (∀ i j : Fin 3, i ≠ j → ∃ off : Fin 5, arrangement off = i ∧ arrangement off = j) ∧
    arrangement A = arrangement B) →
  ∃ arrangements_count : Nat, arrangements_count = 36 :=
by
  sorry

end police_arrangements_l1741_174110


namespace find_x_l1741_174117

def star (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_x (x : ℝ) (h : star 4 x = 52) : x = 8 :=
by
  sorry

end find_x_l1741_174117


namespace degree_of_divisor_l1741_174140

theorem degree_of_divisor (f q r d : Polynomial ℝ)
  (h_f : f.degree = 15)
  (h_q : q.degree = 9)
  (h_r : r = Polynomial.C 5 * X^4 + Polynomial.C 3 * X^3 - Polynomial.C 2 * X^2 + Polynomial.C 9 * X - Polynomial.C 7)
  (h_div : f = d * q + r) :
  d.degree = 6 :=
by sorry

end degree_of_divisor_l1741_174140


namespace total_travel_time_l1741_174150

-- Define the given conditions
def speed_jogging : ℝ := 5
def speed_bus : ℝ := 30
def distance_to_school : ℝ := 6.857142857142858

-- State the theorem to prove
theorem total_travel_time :
  (distance_to_school / speed_jogging) + (distance_to_school / speed_bus) = 1.6 :=
by
  sorry

end total_travel_time_l1741_174150


namespace exclude_chairs_l1741_174183

-- Definitions
def total_chairs : ℕ := 10000
def perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Statement
theorem exclude_chairs (n : ℕ) (h₁ : n = total_chairs) :
  perfect_square n → (n - total_chairs) = 0 := 
sorry

end exclude_chairs_l1741_174183


namespace arithmetic_sequence_value_of_n_l1741_174107

theorem arithmetic_sequence_value_of_n :
  ∀ (a n d : ℕ), a = 1 → d = 3 → (a + (n - 1) * d = 2005) → n = 669 :=
by
  intros a n d h_a1 h_d ha_n
  sorry

end arithmetic_sequence_value_of_n_l1741_174107


namespace sculptures_not_on_display_approx_400_l1741_174143

theorem sculptures_not_on_display_approx_400 (A : ℕ) (hA : A = 900) :
  (2 / 3 * A - 2 / 9 * A) = 400 := by
  sorry

end sculptures_not_on_display_approx_400_l1741_174143


namespace quadratic_inequality_l1741_174179

theorem quadratic_inequality (m y1 y2 y3 : ℝ)
  (h1 : m < -2)
  (h2 : y1 = (m-1)^2 - 2*(m-1))
  (h3 : y2 = m^2 - 2*m)
  (h4 : y3 = (m+1)^2 - 2*(m+1)) :
  y3 < y2 ∧ y2 < y1 :=
by
  sorry

end quadratic_inequality_l1741_174179


namespace F5_div_641_Fermat_rel_prime_l1741_174167

def Fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

theorem F5_div_641 : Fermat_number 5 % 641 = 0 := 
  sorry

theorem Fermat_rel_prime (k n : ℕ) (hk: k ≠ n) : Nat.gcd (Fermat_number k) (Fermat_number n) = 1 :=
  sorry

end F5_div_641_Fermat_rel_prime_l1741_174167


namespace function_is_increasing_on_interval_l1741_174123

noncomputable def f (m x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3

theorem function_is_increasing_on_interval {m : ℝ} :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3 ≥ (1/3) * (x - dx)^3 - (1/2) * m * (x - dx)^2 + 4 * (x - dx) - 3)
  ↔ m ≤ 4 :=
sorry

end function_is_increasing_on_interval_l1741_174123


namespace log_base_change_l1741_174180

theorem log_base_change (a b : ℝ) (h₁ : Real.log 5 / Real.log 3 = a) (h₂ : Real.log 7 / Real.log 3 = b) :
    Real.log 35 / Real.log 15 = (a + b) / (1 + a) :=
by
  sorry

end log_base_change_l1741_174180


namespace system_solution_5_3_l1741_174171

variables (x y : ℤ)

theorem system_solution_5_3 :
  (x = 5) ∧ (y = 3) → (2 * x - 3 * y = 1) :=
by intros; sorry

end system_solution_5_3_l1741_174171


namespace g_g_x_has_exactly_4_distinct_real_roots_l1741_174151

noncomputable def g (d x : ℝ) : ℝ := x^2 + 8*x + d

theorem g_g_x_has_exactly_4_distinct_real_roots (d : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ g d (g d x1) = 0 ∧ g d (g d x2) = 0 ∧ g d (g d x3) = 0 ∧ g d (g d x4) = 0) ↔ d < 4 := by {
  sorry
}

end g_g_x_has_exactly_4_distinct_real_roots_l1741_174151


namespace percentage_change_difference_l1741_174197

theorem percentage_change_difference (total_students : ℕ) (initial_enjoy : ℕ) (initial_not_enjoy : ℕ) (final_enjoy : ℕ) (final_not_enjoy : ℕ) :
  total_students = 100 →
  initial_enjoy = 40 →
  initial_not_enjoy = 60 →
  final_enjoy = 80 →
  final_not_enjoy = 20 →
  (40 ≤ y ∧ y ≤ 80) ∧ (40 - 40 = 0) ∧ (80 - 40 = 40) ∧ (80 - 40 = 40) :=
by
  sorry

end percentage_change_difference_l1741_174197


namespace eating_time_correct_l1741_174158

-- Define the rates at which each individual eats cereal
def rate_fat : ℚ := 1 / 20
def rate_thin : ℚ := 1 / 30
def rate_medium : ℚ := 1 / 15

-- Define the combined rate of eating cereal together
def combined_rate : ℚ := rate_fat + rate_thin + rate_medium

-- Define the total pounds of cereal
def total_cereal : ℚ := 5

-- Define the time taken by everyone to eat the cereal
def time_taken : ℚ := total_cereal / combined_rate

-- Proof statement
theorem eating_time_correct :
  time_taken = 100 / 3 :=
by sorry

end eating_time_correct_l1741_174158


namespace units_digit_of_expression_l1741_174137

theorem units_digit_of_expression :
  (4 ^ 101 * 5 ^ 204 * 9 ^ 303 * 11 ^ 404) % 10 = 0 := 
sorry

end units_digit_of_expression_l1741_174137


namespace number_of_digits_in_sum_l1741_174111

theorem number_of_digits_in_sum (C D : ℕ) (hC : C ≠ 0 ∧ C < 10) (hD : D % 2 = 0 ∧ D < 10) : 
  (Nat.digits 10 (8765 + (C * 100 + 43) + (D * 10 + 2))).length = 4 := 
by
  sorry

end number_of_digits_in_sum_l1741_174111


namespace infinite_squares_and_circles_difference_l1741_174191

theorem infinite_squares_and_circles_difference 
  (side_length : ℝ)
  (h₁ : side_length = 1)
  (square_area_sum : ℝ)
  (circle_area_sum : ℝ)
  (h_square_area : square_area_sum = (∑' n : ℕ, (side_length / 2^n)^2))
  (h_circle_area : circle_area_sum = (∑' n : ℕ, π * (side_length / 2^(n+1))^2 ))
  : square_area_sum - circle_area_sum = 2 - (π / 2) :=
by 
  sorry 

end infinite_squares_and_circles_difference_l1741_174191


namespace f_increasing_l1741_174101

noncomputable def f (x : ℝ) : ℝ := 2 * x - Real.sin x

theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end f_increasing_l1741_174101


namespace purely_imaginary_sol_l1741_174133

theorem purely_imaginary_sol {m : ℝ} (h : (m^2 - 3 * m) = 0) (h2 : (m^2 - 5 * m + 6) ≠ 0) : m = 0 :=
sorry

end purely_imaginary_sol_l1741_174133


namespace units_digit_6_pow_6_l1741_174184

theorem units_digit_6_pow_6 : (6 ^ 6) % 10 = 6 := 
by {
  sorry
}

end units_digit_6_pow_6_l1741_174184


namespace find_x_eq_nine_fourths_l1741_174162

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end find_x_eq_nine_fourths_l1741_174162


namespace fewest_trips_l1741_174178

theorem fewest_trips (total_objects : ℕ) (capacity : ℕ) (h_objects : total_objects = 17) (h_capacity : capacity = 3) : 
  (total_objects + capacity - 1) / capacity = 6 :=
by
  sorry

end fewest_trips_l1741_174178


namespace solve_for_x_l1741_174174

theorem solve_for_x (x : ℝ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  sorry

end solve_for_x_l1741_174174


namespace additional_grassy_ground_l1741_174125

theorem additional_grassy_ground (r1 r2 : ℝ) (π : ℝ) :
  r1 = 12 → r2 = 18 → π = Real.pi →
  (π * r2^2 - π * r1^2) = 180 * π := by
sorry

end additional_grassy_ground_l1741_174125


namespace roll_four_fair_dice_l1741_174116
noncomputable def roll_four_fair_dice_prob : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 6
  let prob_all_same : ℚ := favorable_outcomes / total_outcomes
  let prob_not_all_same : ℚ := 1 - prob_all_same
  prob_not_all_same

theorem roll_four_fair_dice :
  roll_four_fair_dice_prob = 215 / 216 :=
by
  sorry

end roll_four_fair_dice_l1741_174116


namespace football_throwing_distance_l1741_174168

theorem football_throwing_distance 
  (T : ℝ)
  (yards_per_throw_at_T : ℝ)
  (yards_per_throw_at_80 : ℝ)
  (throws_on_Saturday : ℕ)
  (throws_on_Sunday : ℕ)
  (saturday_distance sunday_distance : ℝ)
  (total_distance : ℝ) :
  yards_per_throw_at_T = 20 →
  yards_per_throw_at_80 = 40 →
  throws_on_Saturday = 20 →
  throws_on_Sunday = 30 →
  saturday_distance = throws_on_Saturday * yards_per_throw_at_T →
  sunday_distance = throws_on_Sunday * yards_per_throw_at_80 →
  total_distance = saturday_distance + sunday_distance →
  total_distance = 1600 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end football_throwing_distance_l1741_174168


namespace exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l1741_174109

theorem exists_two_numbers_with_gcd_quotient_ge_p_plus_one (p : ℕ) (hp : Nat.Prime p)
  (l : List ℕ) (hl_len : l.length = p + 1) (hl_distinct : l.Nodup) :
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ l ∧ b ∈ l ∧ a > b ∧ a / (Nat.gcd a b) ≥ p + 1 := sorry

end exists_two_numbers_with_gcd_quotient_ge_p_plus_one_l1741_174109


namespace distance_home_to_school_l1741_174189

-- Define the variables and conditions
variables (D T : ℝ)
def boy_travel_5km_hr_late := 5 * (T + 5 / 60) = D
def boy_travel_10km_hr_early := 10 * (T - 10 / 60) = D

-- State the theorem to prove
theorem distance_home_to_school 
    (H1 : boy_travel_5km_hr_late D T) 
    (H2 : boy_travel_10km_hr_early D T) : 
  D = 2.5 :=
by
  sorry

end distance_home_to_school_l1741_174189


namespace expand_product_l1741_174106

theorem expand_product (x a : ℝ) : 2 * (x + (a + 2)) * (x + (a - 3)) = 2 * x^2 + (4 * a - 2) * x + 2 * a^2 - 2 * a - 12 :=
by
  sorry

end expand_product_l1741_174106


namespace find_a_l1741_174195

variables (x y : ℝ) (a : ℝ)

-- Condition 1: Original profit equation
def original_profit := y - x = x * (a / 100)

-- Condition 2: New profit equation with 5% cost decrease
def new_profit := y - 0.95 * x = 0.95 * x * ((a + 15) / 100)

theorem find_a (h1 : original_profit x y a) (h2 : new_profit x y a) : a = 185 :=
sorry

end find_a_l1741_174195


namespace simplify_and_evaluate_l1741_174121

theorem simplify_and_evaluate (a : ℕ) (h : a = 2022) :
  (a - 1) / a / (a - 1 / a) = 1 / 2023 :=
by
  sorry

end simplify_and_evaluate_l1741_174121


namespace lines_parallel_l1741_174160

-- Define line l1 and line l2
def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

-- Prove that l1 is parallel to l2
theorem lines_parallel : ∀ x : ℝ, (l1 x - l2 x) = -4 := by
  sorry

end lines_parallel_l1741_174160


namespace rate_percent_simple_interest_l1741_174108

theorem rate_percent_simple_interest (SI P T R : ℝ) (h₁ : SI = 500) (h₂ : P = 2000) (h₃ : T = 2)
  (h₄ : SI = (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for the proof
  sorry

end rate_percent_simple_interest_l1741_174108


namespace engineers_to_designers_ratio_l1741_174176

-- Define the given conditions for the problem
variables (e d : ℕ) -- e is the number of engineers, d is the number of designers
variables (h1 : (48 * e + 60 * d) / (e + d) = 52)

-- Theorem statement: The ratio of the number of engineers to the number of designers is 2:1
theorem engineers_to_designers_ratio (h1 : (48 * e + 60 * d) / (e + d) = 52) : e = 2 * d :=
by {
  sorry  
}

end engineers_to_designers_ratio_l1741_174176


namespace simplify_complex_expression_l1741_174199

noncomputable def i : ℂ := Complex.I

theorem simplify_complex_expression : 
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i := by 
  sorry

end simplify_complex_expression_l1741_174199


namespace painted_prisms_l1741_174157

theorem painted_prisms (n : ℕ) (h : n > 2) :
  2 * ((n - 2) * (n - 1) + (n - 2) * n + (n - 1) * n) = (n - 2) * (n - 1) * n ↔ n = 7 :=
by sorry

end painted_prisms_l1741_174157


namespace c_10_value_l1741_174144

def c : ℕ → ℤ
| 0 => 3
| 1 => 9
| (n + 1) => c n * c (n - 1)

theorem c_10_value : c 10 = 3^89 :=
by
  sorry

end c_10_value_l1741_174144


namespace arithmetic_sequence_problem_l1741_174131

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)
  (h1 : a 6 + a 9 = 16)
  (h2 : a 4 = 1)
  (h_arith : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) :
  a 11 = 15 :=
by
  sorry

end arithmetic_sequence_problem_l1741_174131


namespace water_overflow_amount_l1741_174190

-- Declare the conditions given in the problem
def tap_production_per_hour : ℕ := 200
def tap_run_duration_in_hours : ℕ := 24
def tank_capacity_in_ml : ℕ := 4000

-- Define the total water produced by the tap
def total_water_produced : ℕ := tap_production_per_hour * tap_run_duration_in_hours

-- Define the amount of water that overflows
def water_overflowed : ℕ := total_water_produced - tank_capacity_in_ml

-- State the theorem to prove the amount of overflowing water
theorem water_overflow_amount : water_overflowed = 800 :=
by
  -- Placeholder for the proof
  sorry

end water_overflow_amount_l1741_174190


namespace product_of_solutions_t_squared_eq_49_l1741_174185

theorem product_of_solutions_t_squared_eq_49 (t : ℝ) (h1 : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end product_of_solutions_t_squared_eq_49_l1741_174185


namespace possible_last_digits_count_l1741_174115

theorem possible_last_digits_count : 
  ∃ s : Finset Nat, s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ ∀ n ∈ s, ∃ m, (m % 10 = n) ∧ (m % 3 = 0) := 
sorry

end possible_last_digits_count_l1741_174115


namespace jerry_age_l1741_174165

theorem jerry_age (M J : ℤ) (h1 : M = 16) (h2 : M = 2 * J - 8) : J = 12 :=
by
  sorry

end jerry_age_l1741_174165


namespace two_numbers_ratio_l1741_174156

theorem two_numbers_ratio (A B : ℕ) (h_lcm : Nat.lcm A B = 30) (h_sum : A + B = 25) :
  ∃ x y : ℕ, x = 2 ∧ y = 3 ∧ A / B = x / y := 
sorry

end two_numbers_ratio_l1741_174156


namespace parabola_standard_eq_l1741_174130

theorem parabola_standard_eq (p : ℝ) (x y : ℝ) :
  (∃ x y, 3 * x - 4 * y - 12 = 0) →
  ( (p = 6 ∧ x^2 = -12 * y ∧ y = -3) ∨ (p = 8 ∧ y^2 = 16 * x ∧ x = 4)) :=
sorry

end parabola_standard_eq_l1741_174130


namespace zachary_seventh_day_cans_l1741_174119

-- Define the number of cans found by Zachary every day.
def cans_found_on (day : ℕ) : ℕ :=
  if day = 1 then 4
  else if day = 2 then 9
  else if day = 3 then 14
  else 5 * (day - 1) - 1

-- The theorem to prove the number of cans found on the seventh day.
theorem zachary_seventh_day_cans : cans_found_on 7 = 34 :=
by 
  sorry

end zachary_seventh_day_cans_l1741_174119


namespace modular_inverse_of_31_mod_35_is_1_l1741_174163

theorem modular_inverse_of_31_mod_35_is_1 :
  ∃ a : ℕ, 0 ≤ a ∧ a < 35 ∧ 31 * a % 35 = 1 := sorry

end modular_inverse_of_31_mod_35_is_1_l1741_174163


namespace min_value_is_144_l1741_174132

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2

theorem min_value_is_144 (x y z : ℝ) (hxyz : x * y * z = 48) : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ xyz = 48 ∧ min_value_expression x y z = 144 :=
by 
  sorry

end min_value_is_144_l1741_174132


namespace part1_part2_l1741_174147

-- Part (1): Prove k = 3 given x = -1 is a solution
theorem part1 (k : ℝ) (h : k * (-1)^2 + 4 * (-1) + 1 = 0) : k = 3 := 
sorry

-- Part (2): Prove k ≤ 4 and k ≠ 0 for the quadratic equation to have two real roots
theorem part2 (k : ℝ) (h : 16 - 4 * k ≥ 0) : k ≤ 4 ∧ k ≠ 0 :=
sorry

end part1_part2_l1741_174147


namespace total_pages_written_l1741_174100

-- Define the conditions
def timeMon : ℕ := 60  -- Minutes on Monday
def rateMon : ℕ := 30  -- Minutes per page on Monday

def timeTue : ℕ := 45  -- Minutes on Tuesday
def rateTue : ℕ := 15  -- Minutes per page on Tuesday

def pagesWed : ℕ := 5  -- Pages written on Wednesday

-- Function to compute pages written based on time and rate
def pages_written (time rate : ℕ) : ℕ := time / rate

-- Define the theorem to be proved
theorem total_pages_written :
  pages_written timeMon rateMon + pages_written timeTue rateTue + pagesWed = 10 :=
sorry

end total_pages_written_l1741_174100


namespace math_team_count_l1741_174112

open Nat

theorem math_team_count :
  let girls := 7
  let boys := 12
  let total_team := 16
  let count_ways (n k : ℕ) := choose n k
  (count_ways girls 3) * (count_ways boys 5) * (count_ways (girls - 3 + boys - 5) 8) = 456660 :=
by
  sorry

end math_team_count_l1741_174112


namespace converse_not_true_prop_B_l1741_174173

noncomputable def line_in_plane (b : Type) (α : Type) : Prop := sorry
noncomputable def perp_line_plane (b : Type) (β : Type) : Prop := sorry
noncomputable def perp_planes (α : Type) (β : Type) : Prop := sorry
noncomputable def parallel_planes (α : Type) (β : Type) : Prop := sorry

variables (a b c : Type) (α β : Type)

theorem converse_not_true_prop_B :
  (line_in_plane b α) → (perp_planes α β) → ¬ (perp_line_plane b β) :=
sorry

end converse_not_true_prop_B_l1741_174173


namespace find_k_l1741_174139

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (u v : V)

theorem find_k (h : ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ u + t • (v - u) = k • u + (5 / 8) • v) :
  k = 3 / 8 := sorry

end find_k_l1741_174139


namespace sarah_score_is_122_l1741_174135

-- Define the problem parameters and state the theorem
theorem sarah_score_is_122 (s g : ℝ)
  (h1 : s = g + 40)
  (h2 : (s + g) / 2 = 102) :
  s = 122 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end sarah_score_is_122_l1741_174135
