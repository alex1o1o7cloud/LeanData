import Mathlib

namespace initial_population_l88_88728

variable (P : ℕ)

theorem initial_population
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℚ := 1.2) :
  (P = 3000) :=
by
  sorry

end initial_population_l88_88728


namespace axis_of_symmetry_l88_88560

theorem axis_of_symmetry (x : ℝ) : 
  ∀ y, y = x^2 - 2 * x - 3 → (∃ k : ℝ, k = 1 ∧ ∀ x₀ : ℝ, y = (x₀ - k)^2 + C) := 
sorry

end axis_of_symmetry_l88_88560


namespace samuel_faster_l88_88827

theorem samuel_faster (S T_h : ℝ) (hT_h : T_h = 1.3) (hS : S = 30) :
  (T_h * 60) - S = 48 :=
by
  sorry

end samuel_faster_l88_88827


namespace min_square_side_length_l88_88303

theorem min_square_side_length 
  (table_length : ℕ) (table_breadth : ℕ) (cube_side : ℕ) (num_tables : ℕ)
  (cond1 : table_length = 12)
  (cond2 : table_breadth = 16)
  (cond3 : cube_side = 4)
  (cond4 : num_tables = 4) :
  (2 * table_length + 2 * table_breadth) = 56 := 
by
  sorry

end min_square_side_length_l88_88303


namespace solve_fun_problem_l88_88345

variable (f : ℝ → ℝ)

-- Definitions of the conditions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_monotonic_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- The main theorem
theorem solve_fun_problem (h_even : is_even f) (h_monotonic : is_monotonic_on_pos f) :
  {x : ℝ | f (x + 1) = f (2 * x)} = {1, -1 / 3} := 
sorry

end solve_fun_problem_l88_88345


namespace two_dice_sum_greater_than_four_l88_88241
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l88_88241


namespace minimum_bailing_rate_is_seven_l88_88688

noncomputable def minimum_bailing_rate (shore_distance : ℝ) (paddling_speed : ℝ) 
                                       (water_intake_rate : ℝ) (max_capacity : ℝ) : ℝ := 
  let time_to_shore := shore_distance / paddling_speed
  let intake_total := water_intake_rate * time_to_shore
  let required_rate := (intake_total - max_capacity) / time_to_shore
  required_rate

theorem minimum_bailing_rate_is_seven 
  (shore_distance : ℝ) (paddling_speed : ℝ) (water_intake_rate : ℝ) (max_capacity : ℝ) :
  shore_distance = 2 →
  paddling_speed = 3 →
  water_intake_rate = 8 →
  max_capacity = 40 →
  minimum_bailing_rate shore_distance paddling_speed water_intake_rate max_capacity = 7 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end minimum_bailing_rate_is_seven_l88_88688


namespace max_ab_l88_88125

noncomputable def f (a x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - (1/2) * x^2

theorem max_ab (a b : ℝ) (h₁ : 0 < a)
  (h₂ : ∀ x, f a x ≥ - (1/2) * x^2 + a * x + b) : 
  ab ≤ ((Real.exp 1) / 2) :=
sorry

end max_ab_l88_88125


namespace derivative_at_neg_one_l88_88410

def f (x : ℝ) : ℝ := List.prod (List.map (λ k => (x^3 + k)) (List.range' 1 100))

theorem derivative_at_neg_one : deriv f (-1) = 3 * Nat.factorial 99 := by
  sorry

end derivative_at_neg_one_l88_88410


namespace sin_two_alpha_l88_88771

theorem sin_two_alpha (alpha : ℝ) (h : Real.cos (π / 4 - alpha) = 4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_two_alpha_l88_88771


namespace pascal_triangle_row_15_element_4_l88_88996

theorem pascal_triangle_row_15_element_4 : Nat.choose 15 3 = 455 := 
by 
  sorry

end pascal_triangle_row_15_element_4_l88_88996


namespace probability_of_problem_being_solved_l88_88956

-- Define the probabilities of solving the problem.
def prob_A_solves : ℚ := 1 / 5
def prob_B_solves : ℚ := 1 / 3

-- Define the proof statement
theorem probability_of_problem_being_solved :
  (1 - ((1 - prob_A_solves) * (1 - prob_B_solves))) = 7 / 15 :=
by
  sorry

end probability_of_problem_being_solved_l88_88956


namespace packages_eq_nine_l88_88816

-- Definitions of the given conditions
def x : ℕ := 50
def y : ℕ := 5
def z : ℕ := 5

-- Statement: Prove that the number of packages Amy could make equals 9
theorem packages_eq_nine : (x - y) / z = 9 :=
by
  sorry

end packages_eq_nine_l88_88816


namespace find_x_in_sequence_l88_88864

theorem find_x_in_sequence
  (x d1 d2 : ℤ)
  (h1 : d1 = x - 1370)
  (h2 : d2 = 1070 - x)
  (h3 : -180 - 1070 = -1250)
  (h4 : -6430 - (-180) = -6250)
  (h5 : d2 - d1 = 5000) :
  x = 3720 :=
by
-- Proof omitted
sorry

end find_x_in_sequence_l88_88864


namespace possible_third_side_of_triangle_l88_88519

theorem possible_third_side_of_triangle (a b : ℝ) (ha : a = 3) (hb : b = 6) (x : ℝ) :
  3 < x ∧ x < 9 → x = 6 :=
by
  intros h
  have h1 : 3 < x := h.left
  have h2 : x < 9 := h.right
  have h3 : a + b > x := by linarith
  have h4 : b - a < x := by linarith
  sorry

end possible_third_side_of_triangle_l88_88519


namespace steve_take_home_pay_l88_88399

def annual_salary : ℕ := 40000
def tax_percentage : ℝ := 0.20
def healthcare_percentage : ℝ := 0.10
def union_dues : ℕ := 800

theorem steve_take_home_pay :
  annual_salary - (annual_salary * tax_percentage).to_nat - (annual_salary * healthcare_percentage).to_nat - union_dues = 27200 :=
by
  sorry

end steve_take_home_pay_l88_88399


namespace arithmetic_sequence_a7_value_l88_88364

variable (a : ℕ → ℝ) (a1 a13 a7 : ℝ)

theorem arithmetic_sequence_a7_value
  (h1 : a 1 = a1)
  (h13 : a 13 = a13)
  (h_sum : a1 + a13 = 12)
  (h_arith : 2 * a7 = a1 + a13) :
  a7 = 6 :=
by
  sorry

end arithmetic_sequence_a7_value_l88_88364


namespace probability_of_x_gt_5y_l88_88973

theorem probability_of_x_gt_5y :
  let rectangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y ≤ 2500}
  let area_of_rectangle := 3000 * 2500
  let triangle := {(x, y) | 0 ≤ x ∧ x ≤ 3000 ∧ 0 ≤ y ∧ y < x / 5}
  let area_of_triangle := (3000 * 600) / 2
  ∃ prob : ℚ, (area_of_triangle / area_of_rectangle = prob) ∧ prob = 3 / 25 := by
  sorry

end probability_of_x_gt_5y_l88_88973


namespace cylinder_volume_increase_l88_88946

theorem cylinder_volume_increase (r h : ℝ) (V : ℝ) (hV : V = Real.pi * r^2 * h) :
    let new_height := 3 * h
    let new_radius := 2.5 * r
    let new_volume := Real.pi * (new_radius ^ 2) * new_height
    new_volume = 18.75 * V :=
by
  sorry

end cylinder_volume_increase_l88_88946


namespace ways_to_score_at_least_7_points_l88_88049

-- Definitions based on the given conditions
def red_balls : Nat := 4
def white_balls : Nat := 6
def points_red : Nat := 2
def points_white : Nat := 1

-- Function to count the number of combinations for choosing k elements from n elements
def choose (n : Nat) (k : Nat) : Nat :=
  if h : k ≤ n then
    Nat.descFactorial n k / Nat.factorial k
  else
    0

-- The main theorem to prove the number of ways to get at least 7 points by choosing 5 balls out
theorem ways_to_score_at_least_7_points : 
  (choose red_balls 4 * choose white_balls 1) +
  (choose red_balls 3 * choose white_balls 2) +
  (choose red_balls 2 * choose white_balls 3) = 186 := 
sorry

end ways_to_score_at_least_7_points_l88_88049


namespace average_sales_six_months_l88_88587

theorem average_sales_six_months :
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  average_sales = 7000 :=
by
  let sales1 := 4000
  let sales2 := 6524
  let sales3 := 5689
  let sales4 := 7230
  let sales5 := 6000
  let sales6 := 12557
  let total_sales_first_five := sales1 + sales2 + sales3 + sales4 + sales5
  let total_sales_six := total_sales_first_five + sales6
  let average_sales := total_sales_six / 6
  have h : total_sales_first_five = 29443 := by sorry
  have h1 : total_sales_six = 42000 := by sorry
  have h2 : average_sales = 7000 := by sorry
  exact h2

end average_sales_six_months_l88_88587


namespace bus_problem_l88_88295

theorem bus_problem
  (initial_children : ℕ := 18)
  (final_total_children : ℕ := 25) :
  final_total_children - initial_children = 7 :=
by
  sorry

end bus_problem_l88_88295


namespace compare_fractions_l88_88318

theorem compare_fractions : (-8 / 21: ℝ) > (-3 / 7: ℝ) :=
sorry

end compare_fractions_l88_88318


namespace sum_of_interior_angles_of_pentagon_l88_88710

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end sum_of_interior_angles_of_pentagon_l88_88710


namespace problem_statement_l88_88644

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x : ℝ, f x = x^2 + x + 1) 
  (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
sorry

end problem_statement_l88_88644


namespace average_of_last_three_numbers_l88_88197

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l88_88197


namespace remaining_homes_proof_l88_88370

-- Define the total number of homes
def total_homes : ℕ := 200

-- Distributed homes after the first hour
def homes_distributed_first_hour : ℕ := (2 * total_homes) / 5

-- Remaining homes after the first hour
def remaining_homes_first_hour : ℕ := total_homes - homes_distributed_first_hour

-- Distributed homes in the next 2 hours
def homes_distributed_next_two_hours : ℕ := (60 * remaining_homes_first_hour) / 100

-- Remaining homes after the next 2 hours
def homes_remaining : ℕ := remaining_homes_first_hour - homes_distributed_next_two_hours

theorem remaining_homes_proof : homes_remaining = 48 := by
  sorry

end remaining_homes_proof_l88_88370


namespace congruence_problem_l88_88943

theorem congruence_problem {x : ℤ} (h : 4 * x + 5 ≡ 3 [ZMOD 20]) : 3 * x + 8 ≡ 2 [ZMOD 10] :=
sorry

end congruence_problem_l88_88943


namespace gcd_1021_2729_l88_88765

theorem gcd_1021_2729 : Int.gcd 1021 2729 = 1 :=
by
  sorry

end gcd_1021_2729_l88_88765


namespace kim_branch_marking_l88_88534

theorem kim_branch_marking (L : ℝ) (rem_frac : ℝ) (third_piece : ℝ) (F : ℝ) :
  L = 3 ∧ rem_frac = 0.6 ∧ third_piece = 1 ∧ L * rem_frac = 1.8 → F = 1 / 15 :=
by sorry

end kim_branch_marking_l88_88534


namespace cindy_total_time_to_travel_one_mile_l88_88315

-- Definitions for the conditions
def run_speed : ℝ := 3 -- Cindy's running speed in miles per hour.
def walk_speed : ℝ := 1 -- Cindy's walking speed in miles per hour.
def run_distance : ℝ := 0.5 -- Distance run by Cindy in miles.
def walk_distance : ℝ := 0.5 -- Distance walked by Cindy in miles.

-- Theorem statement
theorem cindy_total_time_to_travel_one_mile : 
  ((run_distance / run_speed) + (walk_distance / walk_speed)) * 60 = 40 := 
by
  sorry

end cindy_total_time_to_travel_one_mile_l88_88315


namespace graph_intersects_x_axis_once_l88_88517

noncomputable def f (m x : ℝ) : ℝ := (m - 1) * x^2 - 6 * x + (3 / 2) * m

theorem graph_intersects_x_axis_once (m : ℝ) :
  (∃ x : ℝ, f m x = 0 ∧ ∀ y : ℝ, f m y = 0 → y = x) ↔ (m = 1 ∨ m = 3 ∨ m = -2) :=
by
  sorry

end graph_intersects_x_axis_once_l88_88517


namespace product_possible_values_l88_88311

theorem product_possible_values (N L M M_5: ℤ) :
  M = L + N → 
  M_5 = M - 8 → 
  ∃ L_5, L_5 = L + 5 ∧ |M_5 - L_5| = 6 →
  N = 19 ∨ N = 7 → 19 * 7 = 133 :=
by {
  sorry
}

end product_possible_values_l88_88311


namespace average_of_last_three_numbers_l88_88193

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l88_88193


namespace dangerous_animals_remaining_in_swamp_l88_88660

-- Define the initial counts of each dangerous animals
def crocodiles_initial := 42
def alligators_initial := 35
def vipers_initial := 10
def water_moccasins_initial := 28
def cottonmouth_snakes_initial := 15
def piranha_fish_initial := 120

-- Define the counts of migrating animals
def crocodiles_migrating := 9
def alligators_migrating := 7
def vipers_migrating := 3

-- Define the total initial dangerous animals
def total_initial : Nat :=
  crocodiles_initial + alligators_initial + vipers_initial + water_moccasins_initial + cottonmouth_snakes_initial + piranha_fish_initial

-- Define the total migrating dangerous animals
def total_migrating : Nat :=
  crocodiles_migrating + alligators_migrating + vipers_migrating

-- Define the total remaining dangerous animals
def total_remaining : Nat :=
  total_initial - total_migrating

theorem dangerous_animals_remaining_in_swamp :
  total_remaining = 231 :=
by
  -- simply using the calculation we know
  sorry

end dangerous_animals_remaining_in_swamp_l88_88660


namespace cannot_all_be_zero_l88_88972

theorem cannot_all_be_zero :
  ¬ ∃ (f : ℕ → ℕ), (∀ i, f i ∈ { x : ℕ | 1 ≤ x ∧ x ≤ 1989 }) ∧
                   (∀ i j, f (i + j) = f i - f j) ∧
                   (∃ n, ∀ i, f (i + n) = 0) :=
by
  sorry

end cannot_all_be_zero_l88_88972


namespace fourth_quadrant_negative_half_x_axis_upper_half_plane_l88_88621

theorem fourth_quadrant (m : ℝ) : ((-7 < m ∧ m < 3) ↔ ((m^2 - 8 * m + 15 > 0) ∧ (m^2 + 3 * m - 28 < 0))) :=
sorry

theorem negative_half_x_axis (m : ℝ) : (m = 4 ↔ ((m^2 - 8 * m + 15 < 0) ∧ (m^2 + 3 * m - 28 = 0))) :=
sorry

theorem upper_half_plane (m : ℝ) : ((m ≥ 4 ∨ m ≤ -7) ↔ (m^2 + 3 * m - 28 ≥ 0)) :=
sorry

end fourth_quadrant_negative_half_x_axis_upper_half_plane_l88_88621


namespace hillside_camp_boys_percentage_l88_88659

theorem hillside_camp_boys_percentage (B G : ℕ) 
  (h1 : B + G = 60) 
  (h2 : G = 6) : (B: ℕ) / 60 * 100 = 90 :=
by
  sorry

end hillside_camp_boys_percentage_l88_88659


namespace derivative_y_l88_88326

noncomputable def y (x : ℝ) : ℝ :=
  (3 * x + 1)^4 * Real.arcsin (1 / (3 * x + 1))
  + (3 * x^2 + 2 * x + 1) * Real.sqrt (9 * x^2 + 6 * x)

theorem derivative_y (x : ℝ) (h : 3 * x + 1 > 0) :
  deriv y x = 12 * (3 * x + 1)^3 * Real.arcsin (1 / (3 * x + 1))
  + (3 * x + 1) * (18 * x^2) / Real.sqrt (9 * x^2 + 6 * x) := by
  sorry

end derivative_y_l88_88326


namespace parabola_with_given_focus_l88_88565

-- Defining the given condition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1

-- Defining the focus coordinates
def focus_coords : ℝ × ℝ := (-3, 0)

-- Proving that the standard equation of the parabola with the left focus of the hyperbola as its focus is y^2 = -12x
theorem parabola_with_given_focus :
  ∃ p : ℝ, (∃ focus : ℝ × ℝ, focus = focus_coords) → 
  ∀ y x : ℝ, y^2 = 4 * p * x → y^2 = -12 * x :=
by
  -- placeholder for proof
  sorry

end parabola_with_given_focus_l88_88565


namespace cut_scene_length_l88_88446

theorem cut_scene_length
  (original_length final_length : ℕ)
  (h_original : original_length = 60)
  (h_final : final_length = 54) :
  original_length - final_length = 6 :=
by 
  sorry

end cut_scene_length_l88_88446


namespace digit_theta_l88_88718

noncomputable def theta : ℕ := 7

theorem digit_theta (Θ : ℕ) (h1 : 378 / Θ = 40 + Θ + Θ) : Θ = theta :=
by {
  sorry
}

end digit_theta_l88_88718


namespace sound_frequency_and_speed_glass_proof_l88_88734

def length_rod : ℝ := 1.10 -- Length of the glass rod, l in meters
def nodal_distance_air : ℝ := 0.12 -- Distance between nodal points in air, l' in meters
def speed_sound_air : ℝ := 340 -- Speed of sound in air, V in meters per second

-- Frequency of the sound produced
def frequency_sound_produced : ℝ := 1416.67

-- Speed of longitudinal waves in the glass
def speed_longitudinal_glass : ℝ := 3116.67

theorem sound_frequency_and_speed_glass_proof :
  (2 * nodal_distance_air = 0.24) ∧
  (frequency_sound_produced * (2 * length_rod) = speed_longitudinal_glass) :=
by
  -- Here we will include real equivalent math proof in the future
  sorry

end sound_frequency_and_speed_glass_proof_l88_88734


namespace parabola_focus_l88_88618

-- Define the parabola
def parabolaEquation (x y : ℝ) : Prop := y^2 = -6 * x

-- Define the focus
def focus (x y : ℝ) : Prop := x = -3 / 2 ∧ y = 0

-- The proof problem: showing the focus of the given parabola
theorem parabola_focus : ∃ x y : ℝ, parabolaEquation x y ∧ focus x y :=
by
    sorry

end parabola_focus_l88_88618


namespace cheesecake_factory_working_days_l88_88028

-- Define the savings rates
def robby_saves := 2 / 5
def jaylen_saves := 3 / 5
def miranda_saves := 1 / 2

-- Define their hourly rate and daily working hours
def hourly_rate := 10 -- dollars per hour
def work_hours_per_day := 10 -- hours per day

-- Define their combined savings after four weeks and the combined savings target
def four_weeks := 4 * 7
def combined_savings_target := 3000 -- dollars

-- Question: Prove that the number of days they work per week is 7
theorem cheesecake_factory_working_days (d : ℕ) (h : d * 400 = combined_savings_target / 4) : d = 7 := sorry

end cheesecake_factory_working_days_l88_88028


namespace total_cookies_correct_l88_88455

-- Define the conditions
def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

-- Define the number of cookies collected by each person
def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℕ := (grayson_boxes * cookies_per_box).to_nat
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box

-- Define the total number of cookies collected
def total_cookies : ℕ := abigail_cookies + grayson_cookies + olivia_cookies

-- Prove that the total number of cookies collected is 276
theorem total_cookies_correct : total_cookies = 276 := sorry

end total_cookies_correct_l88_88455


namespace number_is_93_75_l88_88439

theorem number_is_93_75 (x : ℝ) (h : 0.16 * (0.40 * x) = 6) : x = 93.75 :=
by
  -- The proof is omitted.
  sorry

end number_is_93_75_l88_88439


namespace inequality_cube_of_greater_l88_88812

variable {a b : ℝ}

theorem inequality_cube_of_greater (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a > b) : a^3 > b^3 :=
sorry

end inequality_cube_of_greater_l88_88812


namespace probability_correct_l88_88234

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l88_88234


namespace find_a_and_b_l88_88496

-- Given conditions
def curve (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_and_b (a b : ℝ) :
  (∀ x, ∀ y, tangent_line x y → y = b ∧ x = 0) ∧
  (∀ x, ∀ y, y = curve x a b) →
  a = 1 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l88_88496


namespace total_stones_is_odd_l88_88744

variable (d : ℕ) (total_distance : ℕ)

theorem total_stones_is_odd (h1 : d = 10) (h2 : total_distance = 4800) :
  ∃ (N : ℕ), N % 2 = 1 ∧ total_distance = ((N - 1) * 2 * d) :=
by
  -- Let's denote the number of stones as N
  -- Given dx = 10 and total distance as 4800, we want to show that N is odd and 
  -- satisfies the equation: total_distance = ((N - 1) * 2 * d)
  sorry

end total_stones_is_odd_l88_88744


namespace quadratic_no_real_roots_l88_88924

theorem quadratic_no_real_roots 
  (p q a b c : ℝ) 
  (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c)
  (h6 : p ≠ q)
  (h7 : a^2 = p * q)
  (h8 : b + c = p + q)
  (h9 : b = (2 * p + q) / 3)
  (h10 : c = (p + 2 * q) / 3) :
  (∀ x : ℝ, ¬ (b * x^2 - 2 * a * x + c = 0)) := 
by
  sorry

end quadratic_no_real_roots_l88_88924


namespace decorations_cost_correct_l88_88385

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l88_88385


namespace walt_age_l88_88995

-- Conditions
variables (T W : ℕ)
axiom h1 : T = 3 * W
axiom h2 : T + 12 = 2 * (W + 12)

-- Goal: Prove W = 12
theorem walt_age : W = 12 :=
sorry

end walt_age_l88_88995


namespace two_dice_sum_greater_than_four_l88_88239
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l88_88239


namespace power_function_is_odd_l88_88521

theorem power_function_is_odd (m : ℝ) (x : ℝ) (h : (m^2 - m - 1) * (-x)^m = -(m^2 - m - 1) * x^m) : m = -1 :=
sorry

end power_function_is_odd_l88_88521


namespace average_physics_chemistry_l88_88879

theorem average_physics_chemistry (P C M : ℕ) 
  (h1 : (P + C + M) / 3 = 80)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 80) :
  (P + C) / 2 = 70 := 
sorry

end average_physics_chemistry_l88_88879


namespace find_all_f_l88_88484

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_all_f :
  (∀ x : ℝ, f x ≥ 0) ∧
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x + 2 * y^2) →
  ∃ a c : ℝ, (∀ x : ℝ, f x = x^2 + a * x + c) ∧ (a^2 - 4 * c ≤ 0) := sorry

end find_all_f_l88_88484


namespace chord_on_ellipse_midpoint_l88_88036

theorem chord_on_ellipse_midpoint :
  ∀ (A B : ℝ × ℝ)
    (hx1 : (A.1^2) / 2 + A.2^2 = 1)
    (hx2 : (B.1^2) / 2 + B.2^2 = 1)
    (mid : (A.1 + B.1) / 2 = 1/2 ∧ (A.2 + B.2) / 2 = 1/2),
  ∃ (k : ℝ), ∀ (x y : ℝ), y - 1/2 = k * (x - 1/2) ↔ 2 * x + 4 * y = 3 := 
sorry

end chord_on_ellipse_midpoint_l88_88036


namespace D_72_is_22_l88_88163

def D (n : ℕ) : ℕ :=
   -- function definition for D that satisfies the problem's conditions
   sorry

theorem D_72_is_22 : D 72 = 22 :=
by sorry

end D_72_is_22_l88_88163


namespace sphere_radius_eq_three_l88_88650

theorem sphere_radius_eq_three (R : ℝ) :
  4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3 → R = 3 :=
by
  intro h
  have h_eq : R^2 = (1 / 3) * R^3 := by
    have h_canceled : R^2 = (1 / 3) * R^3 := by
      -- simplify the given condition to the core relation
      sorry
  have h_nonzero : R ≠ 0 := by
    -- argument ensuring R is nonzero
    sorry
  have h_final : R = 3 := by
    -- deduce the radius
    sorry
  exact h_final

end sphere_radius_eq_three_l88_88650


namespace person_B_D_coins_l88_88806

theorem person_B_D_coins
  (a d : ℤ)
  (h1 : a - 3 * d = 58)
  (h2 : a - 2 * d = 58)
  (h3 : a + d = 60)
  (h4 : a + 2 * d = 60)
  (h5 : a + 3 * d = 60) :
  (a - 2 * d = 28) ∧ (a = 24) :=
by
  sorry

end person_B_D_coins_l88_88806


namespace goods_train_speed_l88_88060

theorem goods_train_speed (Vm : ℝ) (T : ℝ) (L : ℝ) (Vg : ℝ) :
  Vm = 50 → T = 9 → L = 280 →
  Vg = ((L / T) - (Vm * 1000 / 3600)) * 3600 / 1000 →
  Vg = 62 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end goods_train_speed_l88_88060


namespace number_of_zeros_of_f_l88_88211

def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 2*x - 3 else -2 + log x

theorem number_of_zeros_of_f :
  {x : ℝ | f x = 0}.finite.to_finset.card = 2 := sorry

end number_of_zeros_of_f_l88_88211


namespace park_is_square_l88_88476

-- Defining the concept of a square field
def square_field : ℕ := 4

-- Given condition: The sum of the right angles from the park and the square field
axiom angles_sum (park_angles : ℕ) : park_angles + square_field = 8

-- The theorem to be proven
theorem park_is_square (park_angles : ℕ) (h : park_angles + square_field = 8) : park_angles = 4 :=
by sorry

end park_is_square_l88_88476


namespace solve_for_y_l88_88556

theorem solve_for_y (y : ℕ) : 9^y = 3^12 → y = 6 :=
by
  sorry

end solve_for_y_l88_88556


namespace no_two_digit_numbers_form_perfect_cube_sum_l88_88507

theorem no_two_digit_numbers_form_perfect_cube_sum :
  ∀ (N : ℕ), (10 ≤ N ∧ N < 100) →
  ∀ (t u : ℕ), (N = 10 * t + u) →
  let reversed_N := 10 * u + t in
  let sum := N + reversed_N in
  (∃ k : ℕ, sum = k^3) → false :=
by
  sorry

end no_two_digit_numbers_form_perfect_cube_sum_l88_88507


namespace roots_of_quadratic_l88_88170

theorem roots_of_quadratic (m n : ℝ) (h₁ : m + n = -2) (h₂ : m * n = -2022) (h₃ : ∀ x, x^2 + 2 * x - 2022 = 0 → x = m ∨ x = n) :
  m^2 + 3 * m + n = 2020 :=
sorry

end roots_of_quadratic_l88_88170


namespace Vasechkin_result_l88_88684

-- Define the operations
def P (x : ℕ) : ℕ := (x / 2 * 7) - 1001
def V (x : ℕ) : ℕ := (x / 8) ^ 2 - 1001

-- Define the proposition
theorem Vasechkin_result (x : ℕ) (h_prime : P x = 7) : V x = 295 := 
by {
  -- Proof is omitted
  sorry
}

end Vasechkin_result_l88_88684


namespace finish_remaining_work_l88_88862

theorem finish_remaining_work (x y : ℕ) (hx : x = 30) (hy : y = 15) (hy_work_days : y_work_days = 10) :
  x = 10 :=
by
  sorry

end finish_remaining_work_l88_88862


namespace sum_of_interior_angles_of_pentagon_l88_88709

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  sorry

end sum_of_interior_angles_of_pentagon_l88_88709


namespace root_equation_value_l88_88359

theorem root_equation_value (m : ℝ) (h : m^2 - 2 * m - 3 = 0) : 2026 - m^2 + 2 * m = 2023 :=
sorry

end root_equation_value_l88_88359


namespace roots_sum_equality_l88_88786

theorem roots_sum_equality {a b c : ℝ} {x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ} :
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 1 = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 2 = 0 → x = y₁ ∨ x = y₂ ∨ x = y₃ ∨ x = y₄) →
  x₁ + x₂ = x₃ + x_₄ →
  y₁ + y₂ = y₃ + y₄ :=
sorry

end roots_sum_equality_l88_88786


namespace cindy_total_travel_time_l88_88314

def speed_run := 3 -- Cindy's running speed in miles per hour
def speed_walk := 1 -- Cindy's walking speed in miles per hour
def distance_run := 0.5 -- Distance Cindy runs in miles
def distance_walk := 0.5 -- Distance Cindy walks in miles
def time_run := distance_run / speed_run * 60 -- Time to run half a mile in minutes
def time_walk := distance_walk / speed_walk * 60 -- Time to walk half a mile in minutes

theorem cindy_total_travel_time : time_run + time_walk = 40 := by
  -- skipping proof
  sorry

end cindy_total_travel_time_l88_88314


namespace tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l88_88622

-- Definitions
variables {α β : ℝ}

-- Condition: 0 < α < π / 2
def valid_alpha (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Condition: sin α = 4 / 5
def sin_alpha (α : ℝ) : Prop := Real.sin α = 4 / 5

-- Condition: 0 < β < π / 2
def valid_beta (β : ℝ) : Prop := 0 < β ∧ β < Real.pi / 2

-- Condition: cos (α + β) = -1 / 2
def cos_alpha_add_beta (α β : ℝ) : Prop := Real.cos (α + β) = - 1 / 2

/-- Proofs begin -/
-- Proof for tan α = 4 / 3 given 0 < α < π / 2 and sin α = 4 / 5
theorem tan_alpha_eq (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : Real.tan α = 4 / 3 := 
  sorry

-- Proof for cos (2α + π / 4) = -31√2 / 50 given 0 < α < π / 2 and sin α = 4 / 5
theorem cos_two_alpha_plus_quarter_pi (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : 
  Real.cos (2 * α + Real.pi / 4) = -31 * Real.sqrt 2 / 50 := 
  sorry

-- Proof for sin β = 4 + 3√3 / 10 given 0 < α < π / 2, sin α = 4 / 5, 0 < β < π / 2 and cos (α + β) = -1 / 2
theorem sin_beta_eq (α β : ℝ) (h_validα : valid_alpha α) (h_sinα : sin_alpha α) 
  (h_validβ : valid_beta β) (h_cosαβ : cos_alpha_add_beta α β) : Real.sin β = 4 + 3 * Real.sqrt 3 / 10 := 
  sorry

end tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l88_88622


namespace age_ratio_l88_88998

-- Conditions
def DeepakPresentAge := 27
def RahulAgeAfterSixYears := 42
def YearsToReach42 := 6

-- The theorem to prove the ratio of their ages
theorem age_ratio (R D : ℕ) (hR : R + YearsToReach42 = RahulAgeAfterSixYears) (hD : D = DeepakPresentAge) : R / D = 4 / 3 := by
  sorry

end age_ratio_l88_88998


namespace sum_of_interior_angles_of_pentagon_l88_88706

theorem sum_of_interior_angles_of_pentagon :
  let n := 5 in (n - 2) * 180 = 540 := 
by 
  let n := 5
  show (n - 2) * 180 = 540
  sorry

end sum_of_interior_angles_of_pentagon_l88_88706


namespace johns_outfit_cost_l88_88369

theorem johns_outfit_cost (pants_cost shirt_cost outfit_cost : ℝ)
    (h_pants : pants_cost = 50)
    (h_shirt : shirt_cost = pants_cost + 0.6 * pants_cost)
    (h_outfit : outfit_cost = pants_cost + shirt_cost) :
    outfit_cost = 130 :=
by
  sorry

end johns_outfit_cost_l88_88369


namespace work_completion_days_l88_88725

theorem work_completion_days (A B : ℕ) (hA : A = 20) (hB : B = 20) : A + B / (A + B) / 2 = 10 :=
by 
  rw [hA, hB]
  -- Proof omitted
  sorry

end work_completion_days_l88_88725


namespace simplify_fraction_l88_88285

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l88_88285


namespace bethany_saw_16_portraits_l88_88888

variable (P S : ℕ)

def bethany_conditions : Prop :=
  S = 4 * P ∧ P + S = 80

theorem bethany_saw_16_portraits (P S : ℕ) (h : bethany_conditions P S) : P = 16 := by
  sorry

end bethany_saw_16_portraits_l88_88888


namespace initial_stickers_correct_l88_88173

-- Definitions based on the conditions
def initial_stickers (X : ℕ) : ℕ := X
def after_buying (X : ℕ) : ℕ := X + 26
def after_birthday (X : ℕ) : ℕ := after_buying X + 20
def after_giving (X : ℕ) : ℕ := after_birthday X - 6
def after_decorating (X : ℕ) : ℕ := after_giving X - 58

-- Theorem stating the problem and the expected answer
theorem initial_stickers_correct (X : ℕ) (h : after_decorating X = 2) : initial_stickers X = 26 :=
by {
  sorry
}

end initial_stickers_correct_l88_88173


namespace remainder_of_k_l88_88433

theorem remainder_of_k {k : ℕ} (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 8 = 7) (h4 : k % 11 = 3) (h5 : k < 168) :
  k % 13 = 8 := 
sorry

end remainder_of_k_l88_88433


namespace decorations_cost_l88_88383

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l88_88383


namespace area_of_enclosed_region_is_zero_l88_88748

theorem area_of_enclosed_region_is_zero :
  (∃ (x y : ℝ), x^2 + y^2 = |x| - |y|) → (0 = 0) :=
sorry

end area_of_enclosed_region_is_zero_l88_88748


namespace total_cases_after_three_days_l88_88546

def initial_cases : ℕ := 2000
def increase_rate : ℝ := 0.20
def recovery_rate : ℝ := 0.02

def day_cases (n : ℕ) : ℝ :=
  match n with
  | 0 => initial_cases
  | n + 1 => 
      let prev_cases := day_cases n
      let new_cases := increase_rate * prev_cases
      let recovered := recovery_rate * prev_cases
      prev_cases + new_cases - recovered

theorem total_cases_after_three_days : day_cases 3 = 3286 := by sorry

end total_cases_after_three_days_l88_88546


namespace sin_double_pi_minus_theta_eq_l88_88118

variable {θ : ℝ}
variable {k : ℤ}
variable (h1 : 3 * (Real.cos θ) ^ 2 = Real.tan θ + 3)
variable (h2 : θ ≠ k * Real.pi)

theorem sin_double_pi_minus_theta_eq :
  Real.sin (2 * (Real.pi - θ)) = 2 / 3 :=
sorry

end sin_double_pi_minus_theta_eq_l88_88118


namespace probability_sum_greater_than_four_l88_88223

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88223


namespace action_figure_price_l88_88904

theorem action_figure_price (x : ℝ) (h1 : 2 + 4 * x = 30) : x = 7 :=
by
  -- The proof is provided here
  sorry

end action_figure_price_l88_88904


namespace simplify_expr1_simplify_expr2_l88_88187

variables (x y a b : ℝ)

-- Problem 1
theorem simplify_expr1 : 3 * (4 * x - 2 * y) - 3 * (-y + 8 * x) = -12 * x - 3 * y := 
by sorry

-- Problem 2
theorem simplify_expr2 : 3 * a^2 - 2 * (2 * a^2 - (2 * a * b - a^2) + 4 * a * b) = -3 * a^2 - 4 * a * b := 
by sorry

end simplify_expr1_simplify_expr2_l88_88187


namespace value_of_power_l88_88940

theorem value_of_power (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2014 = 1 :=
by
  sorry

end value_of_power_l88_88940


namespace pow_two_div_factorial_iff_exists_l88_88974

theorem pow_two_div_factorial_iff_exists (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k-1)) ↔ 2^(n-1) ∣ n! := 
by {
  sorry
}

end pow_two_div_factorial_iff_exists_l88_88974


namespace average_last_three_l88_88200

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l88_88200


namespace constant_fraction_condition_l88_88055

theorem constant_fraction_condition 
    (a1 b1 c1 a2 b2 c2 : ℝ) : 
    (∀ x : ℝ, (a1 * x^2 + b1 * x + c1) / (a2 * x^2 + b2 * x + c2) = k) ↔ 
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) :=
by
  sorry

end constant_fraction_condition_l88_88055


namespace cost_of_one_pencil_l88_88800

theorem cost_of_one_pencil (students : ℕ) (more_than_half : ℕ) (pencil_cost : ℕ) (pencils_each : ℕ)
  (total_cost : ℕ) (students_condition : students = 36) 
  (more_than_half_condition : more_than_half > 18) 
  (pencil_count_condition : pencils_each > 1) 
  (cost_condition : pencil_cost > pencils_each) 
  (total_cost_condition : students * pencil_cost * pencils_each = 1881) : 
  pencil_cost = 17 :=
sorry

end cost_of_one_pencil_l88_88800


namespace probability_sum_greater_than_four_l88_88231

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l88_88231


namespace func_above_x_axis_l88_88749

theorem func_above_x_axis (a : ℝ) :
  (∀ x : ℝ, (x^4 + 4*x^3 + a*x^2 - 4*x + 1) > 0) ↔ a > 2 :=
sorry

end func_above_x_axis_l88_88749


namespace anthony_more_shoes_than_jim_l88_88976

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem anthony_more_shoes_than_jim : (anthony_shoes - jim_shoes) = 2 :=
by
  sorry

end anthony_more_shoes_than_jim_l88_88976


namespace percentage_concentration_acid_l88_88218

-- Definitions based on the given conditions
def volume_acid : ℝ := 1.6
def total_volume : ℝ := 8.0

-- Lean statement to prove the percentage concentration is 20%
theorem percentage_concentration_acid : (volume_acid / total_volume) * 100 = 20 := by
  sorry

end percentage_concentration_acid_l88_88218


namespace arithmetic_seq_sum_l88_88005

-- Definition of an arithmetic sequence using a common difference d
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Statement of the problem
theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (hs : arithmetic_sequence a d)
  (hmean : (a 3 + a 8) / 2 = 10) : 
  a 1 + a 10 = 20 :=
sorry

end arithmetic_seq_sum_l88_88005


namespace probability_sum_greater_than_four_l88_88225

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88225


namespace bruce_and_anne_clean_together_l88_88463

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l88_88463


namespace max_non_overlapping_squares_l88_88824

theorem max_non_overlapping_squares (m n : ℕ) : 
  ∃ max_squares : ℕ, max_squares = m :=
by
  sorry

end max_non_overlapping_squares_l88_88824


namespace samuel_faster_than_sarah_l88_88826

-- Definitions based on the conditions
def time_samuel : ℝ := 30
def time_sarah : ℝ := 1.3 * 60

-- The theorem to prove that Samuel finished his homework 48 minutes faster than Sarah
theorem samuel_faster_than_sarah : (time_sarah - time_samuel) = 48 := by
  sorry

end samuel_faster_than_sarah_l88_88826


namespace decorations_cost_correct_l88_88380

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l88_88380


namespace largest_common_divisor_462_330_l88_88854

theorem largest_common_divisor_462_330 :
  ∃ d : ℕ, (d ∣ 462) ∧ (d ∣ 330) ∧
  (∀ k : ℕ, (k ∣ 462) → (k ∣ 330) → k ≤ d) ∧ d = 66 :=
by
  have prime_factors_462 : prime_factors 462 = [2, 3, 7, 11] :=
    sorry
  have prime_factors_330 : prime_factors 330 = [2, 3, 5, 11] :=
    sorry
  have common_factors := [2, 3, 11]
  have largest_common_divisor := 2 * 3 * 11
  use 66
  split
  sorry -- d ∣ 462 and d ∣ 330 proof
  split
  sorry -- d ∣ 330 proof
  split
  sorry -- d is the largest common factor proof
  refl -- d = 66

end largest_common_divisor_462_330_l88_88854


namespace find_a_of_odd_function_l88_88497

noncomputable def f (a : ℝ) (x : ℝ) := 1 + a / (2^x + 1)

theorem find_a_of_odd_function (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = -2 :=
by
  sorry

end find_a_of_odd_function_l88_88497


namespace hockey_players_count_l88_88146

theorem hockey_players_count (cricket_players : ℕ) (football_players : ℕ) (softball_players : ℕ) (total_players : ℕ) 
(h_cricket : cricket_players = 16) 
(h_football : football_players = 18) 
(h_softball : softball_players = 13) 
(h_total : total_players = 59) : 
  total_players - (cricket_players + football_players + softball_players) = 12 := 
by sorry

end hockey_players_count_l88_88146


namespace find_g_l88_88039

def nabla (g h : ℤ) : ℤ := g ^ 2 - h ^ 2

theorem find_g (g : ℤ) (h : ℤ)
  (H1 : 0 < g)
  (H2 : nabla g 6 = 45) :
  g = 9 :=
by
  sorry

end find_g_l88_88039


namespace intersects_l88_88782

-- Define the given conditions
def radius : ℝ := 5
def distance_to_line : ℝ := 3 * Real.sqrt 2

-- Define the relationship to prove
def line_intersects_circle : Prop :=
  radius > distance_to_line

-- Proof Statement
theorem intersects (r d : ℝ) (h_r : r = radius) (h_d : d = distance_to_line) : r > d :=
by {
  rw [h_r, h_d],
  exact Real.lt_of_lt_of_le (by norm_num) (by norm_num),
}

end intersects_l88_88782


namespace smallest_other_number_l88_88412

theorem smallest_other_number (x : ℕ)  (h_pos : 0 < x) (n : ℕ)
  (h_gcd : Nat.gcd 60 n = x + 3)
  (h_lcm : Nat.lcm 60 n = x * (x + 3)) :
  n = 45 :=
sorry

end smallest_other_number_l88_88412


namespace kolacky_bounds_l88_88269

theorem kolacky_bounds (x y : ℕ) (h : 9 * x + 4 * y = 219) :
  294 ≤ 12 * x + 6 * y ∧ 12 * x + 6 * y ≤ 324 :=
sorry

end kolacky_bounds_l88_88269


namespace number_of_nickels_is_three_l88_88286

def coin_problem : Prop :=
  ∃ p n d q : ℕ,
    p + n + d + q = 12 ∧
    p + 5 * n + 10 * d + 25 * q = 128 ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧
    q = 2 * d ∧
    n = 3

theorem number_of_nickels_is_three : coin_problem := 
by 
  sorry

end number_of_nickels_is_three_l88_88286


namespace number_sequence_53rd_l88_88959

theorem number_sequence_53rd (n : ℕ) (h₁ : n = 53) : n = 53 :=
by {
  sorry
}

end number_sequence_53rd_l88_88959


namespace decorations_cost_correct_l88_88378

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l88_88378


namespace solution_set_inequality_l88_88631

-- Definitions of the conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2^x - 3 else - (2^(-x) - 3)

-- Statement to prove
theorem solution_set_inequality :
  is_odd_function f ∧ (∀ x > 0, f x = 2^x - 3)
  → {x : ℝ | f x ≤ -5} = {x : ℝ | x ≤ -3} := by
  sorry

end solution_set_inequality_l88_88631


namespace total_votes_l88_88363

theorem total_votes (A B C V : ℝ)
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V)
  : V = 60000 := 
sorry

end total_votes_l88_88363


namespace energy_fraction_l88_88547

-- Conditions
variables (E : ℝ → ℝ)
variable (x : ℝ)
variable (h : ∀ x, E (x + 1) = 31.6 * E x)

-- The statement to be proven
theorem energy_fraction (x : ℝ) (h : ∀ x, E (x + 1) = 31.6 * E x) : 
  E (x - 1) / E x = 1 / 31.6 :=
by
  sorry

end energy_fraction_l88_88547


namespace sniper_B_has_greater_chance_of_winning_l88_88526

-- Define the probabilities for sniper A
def p_A_1 := 0.4
def p_A_2 := 0.1
def p_A_3 := 0.5

-- Define the probabilities for sniper B
def p_B_1 := 0.1
def p_B_2 := 0.6
def p_B_3 := 0.3

-- Define the expected scores for sniper A and B
def E_A := 1 * p_A_1 + 2 * p_A_2 + 3 * p_A_3
def E_B := 1 * p_B_1 + 2 * p_B_2 + 3 * p_B_3

-- The statement we want to prove
theorem sniper_B_has_greater_chance_of_winning : E_B > E_A := by
  simp [E_A, E_B, p_A_1, p_A_2, p_A_3, p_B_1, p_B_2, p_B_3]
  sorry

end sniper_B_has_greater_chance_of_winning_l88_88526


namespace probability_event_comparison_l88_88031

theorem probability_event_comparison (m n : ℕ) :
  let P_A := (2 * m * n) / (m + n)^2
  let P_B := (m^2 + n^2) / (m + n)^2
  P_A ≤ P_B ∧ (P_A = P_B ↔ m = n) :=
by
  sorry

end probability_event_comparison_l88_88031


namespace coeff_x_expansion_l88_88807

noncomputable def poly : ℕ → ℕ := 
  λ n, ((x^2 + 3*x + 2)^n).coeff 1

theorem coeff_x_expansion :
  poly 5 = 240 :=
by sorry

end coeff_x_expansion_l88_88807


namespace tip_customers_count_l88_88885

-- Definitions and given conditions
def initial_customers : ℕ := 29
def added_customers : ℕ := 20
def no_tip_customers : ℕ := 34

-- Total customers computation
def total_customers : ℕ := initial_customers + added_customers

-- Lean 4 statement for proof problem
theorem tip_customers_count : (total_customers - no_tip_customers) = 15 := by
  sorry

end tip_customers_count_l88_88885


namespace value_of_m_l88_88939

theorem value_of_m (m : ℤ) : (∃ (f : ℤ → ℤ), ∀ x : ℤ, x^2 + m * x + 16 = (f x)^2) ↔ (m = 8 ∨ m = -8) := 
by
  sorry

end value_of_m_l88_88939


namespace chromium_first_alloy_percentage_l88_88803

-- Defining the conditions
def percentage_chromium_first_alloy : ℝ := 10 
def percentage_chromium_second_alloy : ℝ := 6
def mass_first_alloy : ℝ := 15
def mass_second_alloy : ℝ := 35
def percentage_chromium_new_alloy : ℝ := 7.2

-- Proving the percentage of chromium in the first alloy is 10%
theorem chromium_first_alloy_percentage : percentage_chromium_first_alloy = 10 :=
by
  sorry

end chromium_first_alloy_percentage_l88_88803


namespace second_number_l88_88580

theorem second_number (A B : ℝ) (h1 : A = 200) (h2 : 0.30 * A = 0.60 * B + 30) : B = 50 :=
by
  -- proof goes here
  sorry

end second_number_l88_88580


namespace find_x_l88_88942

-- Define the condition as a Lean equation
def equation (x : ℤ) : Prop :=
  45 - (28 - (37 - (x - 19))) = 58

-- The proof statement: if the equation holds, then x = 15
theorem find_x (x : ℤ) (h : equation x) : x = 15 := by
  sorry

end find_x_l88_88942


namespace num_solutions_in_S_l88_88171

open Rat

theorem num_solutions_in_S :
  let S := {x : ℚ | 0 < x ∧ x < 5 / 8}
  let f (qp : ℚ) := (qp.num.add 1) / qp.denom in
  (∃ p q : ℕ, p ≠ 0 ∧ coprime p q ∧ q / p ∈ S ∧ f (q / p) = 2 / 3) → 
  {qp : ℚ | ∃ p q : ℕ, p ≠ 0 ∧ coprime p q ∧ q / p ∈ S ∧ f (q / p) = 2 / 3}.size = 5 :=
by
  sorry

end num_solutions_in_S_l88_88171


namespace problem1_solutions_problem2_solutions_l88_88555

-- Problem 1: Solve x² - 7x + 6 = 0

theorem problem1_solutions (x : ℝ) : 
  x^2 - 7 * x + 6 = 0 ↔ (x = 1 ∨ x = 6) := by
  sorry

-- Problem 2: Solve (2x + 3)² = (x - 3)² 

theorem problem2_solutions (x : ℝ) : 
  (2 * x + 3)^2 = (x - 3)^2 ↔ (x = 0 ∨ x = -6) := by
  sorry

end problem1_solutions_problem2_solutions_l88_88555


namespace sin_cos_sixth_power_sum_l88_88963

theorem sin_cos_sixth_power_sum (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 0.8125 :=
by
  sorry

end sin_cos_sixth_power_sum_l88_88963


namespace no_other_integer_solutions_l88_88263

theorem no_other_integer_solutions :
  (∀ (x : ℤ), (x + 1) ^ 3 + (x + 2) ^ 3 + (x + 3) ^ 3 = (x + 4) ^ 3 → x = 2) := 
by sorry

end no_other_integer_solutions_l88_88263


namespace laps_needed_to_reach_total_distance_l88_88818

-- Define the known conditions
def total_distance : ℕ := 2400
def lap_length : ℕ := 150
def laps_run_each : ℕ := 6
def total_laps_run : ℕ := 2 * laps_run_each

-- Define the proof goal
theorem laps_needed_to_reach_total_distance :
  (total_distance - total_laps_run * lap_length) / lap_length = 4 :=
by
  sorry

end laps_needed_to_reach_total_distance_l88_88818


namespace problem_l88_88931

-- Define proposition p: for all x in ℝ, x^2 + 1 ≥ 1
def p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 1

-- Define proposition q: for angles A and B in a triangle, A > B ↔ sin A > sin B
def q : Prop := ∀ {A B : ℝ}, A > B ↔ Real.sin A > Real.sin B

-- The problem definition: prove that p ∨ q is true
theorem problem (hp : p) (hq : q) : p ∨ q := sorry

end problem_l88_88931


namespace probability_sum_greater_than_four_is_5_over_6_l88_88248

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l88_88248


namespace inequality_transform_l88_88357

theorem inequality_transform (x y : ℝ) (h : y > x) : 2 * y > 2 * x := 
  sorry

end inequality_transform_l88_88357


namespace solve_inequality_l88_88831

theorem solve_inequality : { x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4 } = { x | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) } :=
by
  sorry

end solve_inequality_l88_88831


namespace total_floor_area_is_correct_l88_88219

-- Define the combined area of the three rugs
def combined_area_of_rugs : ℕ := 212

-- Define the area covered by exactly two layers of rug
def area_covered_by_two_layers : ℕ := 24

-- Define the area covered by exactly three layers of rug
def area_covered_by_three_layers : ℕ := 24

-- Define the total floor area covered by the rugs
def total_floor_area_covered : ℕ :=
  combined_area_of_rugs - area_covered_by_two_layers - 2 * area_covered_by_three_layers

-- The theorem stating the total floor area covered
theorem total_floor_area_is_correct : total_floor_area_covered = 140 := by
  sorry

end total_floor_area_is_correct_l88_88219


namespace probability_sum_greater_than_four_l88_88224

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88224


namespace abc_is_cube_l88_88673

theorem abc_is_cube (a b c : ℤ) (h : (a:ℚ) / (b:ℚ) + (b:ℚ) / (c:ℚ) + (c:ℚ) / (a:ℚ) = 3) : ∃ x : ℤ, abc = x^3 :=
by
  sorry

end abc_is_cube_l88_88673


namespace problem1_problem2_l88_88894

-- Problem 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : 
  (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2
theorem problem2 (m : ℝ) : 
  (m^2 - 4) / (4 + 4 * m + m^2) / ((m - 2) / (2 * m - 2)) * ((m + 2) / (m - 1)) = 2 := 
sorry

end problem1_problem2_l88_88894


namespace find_a_l88_88912

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l88_88912


namespace total_birds_and_storks_l88_88442

theorem total_birds_and_storks (initial_birds initial_storks additional_storks : ℕ) 
  (h1 : initial_birds = 3) 
  (h2 : initial_storks = 4) 
  (h3 : additional_storks = 6) 
  : initial_birds + initial_storks + additional_storks = 13 := 
  by sorry

end total_birds_and_storks_l88_88442


namespace number_of_ostriches_l88_88089

theorem number_of_ostriches
    (x y : ℕ)
    (h1 : x + y = 150)
    (h2 : 2 * x + 6 * y = 624) :
    x = 69 :=
by
  -- Proof omitted
  sorry

end number_of_ostriches_l88_88089


namespace probability_sum_greater_than_four_is_5_over_6_l88_88251

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l88_88251


namespace parallel_lines_iff_a_eq_1_l88_88967

theorem parallel_lines_iff_a_eq_1 (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y - 1 = 0 ↔ x + 2*y + 4 = 0) ↔ (a = 1) := 
sorry

end parallel_lines_iff_a_eq_1_l88_88967


namespace six_points_within_circle_l88_88689

/-- If six points are placed inside or on a circle with radius 1, then 
there always exist at least two points such that the distance between 
them is at most 1. -/
theorem six_points_within_circle : ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (points i).1^2 + (points i).2^2 ≤ 1) → 
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 1 :=
by
  -- Condition: Circle of radius 1
  intro points h_points
  sorry

end six_points_within_circle_l88_88689


namespace volume_of_rectangular_prism_l88_88834

-- Define the dimensions a, b, c as non-negative real numbers
variables (a b c : ℝ)

-- Given conditions
def condition_1 := a * b = 30
def condition_2 := a * c = 50
def condition_3 := b * c = 75

-- The theorem statement
theorem volume_of_rectangular_prism :
  (a * b * c) = 335 :=
by
  -- Assume the given conditions
  assume h1 : condition_1 a b,
  assume h2 : condition_2 a c,
  assume h3 : condition_3 b c,
  -- Proof skipped
  sorry

end volume_of_rectangular_prism_l88_88834


namespace each_child_gets_twelve_cupcakes_l88_88577

def total_cupcakes := 96
def children := 8
def cupcakes_per_child : ℕ := total_cupcakes / children

theorem each_child_gets_twelve_cupcakes :
  cupcakes_per_child = 12 :=
by
  sorry

end each_child_gets_twelve_cupcakes_l88_88577


namespace fraction_area_above_line_l88_88737

-- Define the problem conditions
def point1 : ℝ × ℝ := (4, 1)
def point2 : ℝ × ℝ := (9, 5)
def vertex1 : ℝ × ℝ := (4, 0)
def vertex2 : ℝ × ℝ := (9, 0)
def vertex3 : ℝ × ℝ := (9, 5)
def vertex4 : ℝ × ℝ := (4, 5)

-- Define the theorem statement
theorem fraction_area_above_line :
  let area_square := 25
  let area_below_line := 2.5
  let area_above_line := area_square - area_below_line
  area_above_line / area_square = 9 / 10 :=
by
  sorry -- Proof omitted

end fraction_area_above_line_l88_88737


namespace probability_sum_greater_than_four_l88_88243

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88243


namespace cost_of_3600_pens_l88_88739

theorem cost_of_3600_pens
  (pack_size : ℕ)
  (pack_cost : ℝ)
  (n_pens : ℕ)
  (pen_cost : ℝ)
  (total_cost : ℝ)
  (h1: pack_size = 150)
  (h2: pack_cost = 45)
  (h3: n_pens = 3600)
  (h4: pen_cost = pack_cost / pack_size)
  (h5: total_cost = n_pens * pen_cost) :
  total_cost = 1080 :=
sorry

end cost_of_3600_pens_l88_88739


namespace abs_inequality_no_solution_l88_88141

theorem abs_inequality_no_solution (a : ℝ) : (∀ x : ℝ, |x - 5| + |x + 3| ≥ a) ↔ a ≤ 8 :=
by sorry

end abs_inequality_no_solution_l88_88141


namespace solve_linear_equation_l88_88204

theorem solve_linear_equation :
  ∀ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 :=
by
  sorry

end solve_linear_equation_l88_88204


namespace problem1_problem2_problem3_problem4_l88_88892

theorem problem1 : (70.8 - 1.25 - 1.75 = 67.8) := sorry

theorem problem2 : ((8 + 0.8) * 1.25 = 11) := sorry

theorem problem3 : (125 * 0.48 = 600) := sorry

theorem problem4 : (6.7 * (9.3 * (6.2 + 1.7)) = 554.559) := sorry

end problem1_problem2_problem3_problem4_l88_88892


namespace sum_of_solutions_eq_l88_88481

theorem sum_of_solutions_eq (x : ℝ) : (5 * x - 7) * (4 * x + 11) = 0 ->
  -((27 : ℝ) / (20 : ℝ)) =
  - ((5 * - 7) * (4 * x + 11)) / ((5 * x - 7) * 4) :=
by
  intro h
  sorry

end sum_of_solutions_eq_l88_88481


namespace quadratic_complete_square_l88_88652

/-- Given quadratic expression, complete the square to find the equivalent form
    and calculate the sum of the coefficients a, h, k. -/
theorem quadratic_complete_square (a h k : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + 2 = a * (x - h)^2 + k) → a + h + k = -2 :=
by
  intro h₁
  sorry

end quadratic_complete_square_l88_88652


namespace simplify_336_to_fraction_l88_88274

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l88_88274


namespace geometric_series_sum_l88_88617

theorem geometric_series_sum :
  let a := 1 / 4
  let r := - (1 / 4)
  ∃ S : ℚ, S = (a * (1 - r^6)) / (1 - r) ∧ S = 4095 / 81920 :=
by
  let a := 1 / 4
  let r := - (1 / 4)
  exists (a * (1 - r^6)) / (1 - r)
  sorry

end geometric_series_sum_l88_88617


namespace geometric_sequence_problem_l88_88954

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given condition for the geometric sequence
variables {a : ℕ → ℝ} (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27)

-- Theorem to be proven
theorem geometric_sequence_problem (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27) : a 1 * a 9 = 9 :=
sorry

end geometric_sequence_problem_l88_88954


namespace range_of_expressions_l88_88509

theorem range_of_expressions (x y : ℝ) (h1 : 30 < x ∧ x < 42) (h2 : 16 < y ∧ y < 24) :
  46 < x + y ∧ x + y < 66 ∧ -18 < x - 2 * y ∧ x - 2 * y < 10 ∧ (5 / 4) < (x / y) ∧ (x / y) < (21 / 8) :=
sorry

end range_of_expressions_l88_88509


namespace PTAFinalAmount_l88_88190

theorem PTAFinalAmount (initial_amount : ℝ) (spent_on_supplies_fraction : ℝ) (spent_on_food_fraction : ℝ) : 
  initial_amount = 400 → 
  spent_on_supplies_fraction = 1 / 4 → 
  spent_on_food_fraction = 1 / 2 → 
  (initial_amount - (initial_amount * spent_on_supplies_fraction)) / 2 = 150 := 
by
  intros h_initial h_supplies h_food
  rw [h_initial, h_supplies, h_food]
  norm_num
  sorry

end PTAFinalAmount_l88_88190


namespace fifth_inequality_proof_l88_88176

theorem fifth_inequality_proof : 
  1 + (1 / (2:ℝ)^2) + (1 / (3:ℝ)^2) + (1 / (4:ℝ)^2) + (1 / (5:ℝ)^2) + (1 / (6:ℝ)^2) < (11 / 6) :=
by {
  sorry
}

end fifth_inequality_proof_l88_88176


namespace sara_remaining_red_balloons_l88_88395

-- Given conditions
def initial_red_balloons := 31
def red_balloons_given := 24

-- Statement to prove
theorem sara_remaining_red_balloons : (initial_red_balloons - red_balloons_given = 7) :=
by
  -- Proof can be skipped
  sorry

end sara_remaining_red_balloons_l88_88395


namespace jerry_average_increase_l88_88669

-- Definitions of conditions
def first_three_tests_average (avg : ℕ) : Prop := avg = 85
def fourth_test_score (score : ℕ) : Prop := score = 97
def desired_average_increase (increase : ℕ) : Prop := increase = 3

-- The theorem to prove
theorem jerry_average_increase
  (first_avg first_avg_value : ℕ)
  (fourth_score fourth_score_value : ℕ)
  (increase_points : ℕ)
  (h1 : first_three_tests_average first_avg)
  (h2 : fourth_test_score fourth_score)
  (h3 : desired_average_increase increase_points) :
  fourth_score = 97 → (first_avg + fourth_score) / 4 = 88 → increase_points = 3 :=
by
  intros _ _
  sorry

end jerry_average_increase_l88_88669


namespace betty_paid_total_l88_88603

def cost_slippers (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_lipsticks (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_hair_colors (count : ℕ) (price : ℝ) : ℝ := count * price

def total_cost := 
  cost_slippers 6 2.5 +
  cost_lipsticks 4 1.25 +
  cost_hair_colors 8 3

theorem betty_paid_total :
  total_cost = 44 := 
  sorry

end betty_paid_total_l88_88603


namespace problem_statement_l88_88431

theorem problem_statement (x y z : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : z = 3) :
  x^2 + y^2 + z^2 + 2*x*z = 26 :=
by
  rw [h1, h2, h3]
  norm_num

end problem_statement_l88_88431


namespace grid_area_l88_88006

-- Definitions based on problem conditions
def num_lines : ℕ := 36
def perimeter : ℕ := 72
def side_length : ℕ := perimeter / num_lines

-- Problem statement
theorem grid_area (h : num_lines = 36) (p : perimeter = 72)
  (s : side_length = 2) :
  let n_squares := (8 - 1) * (4 - 1)
  let area_square := side_length ^ 2
  let total_area := n_squares * area_square
  total_area = 84 :=
by {
  -- Skipping proof
  sorry
}

end grid_area_l88_88006


namespace can_form_triangle_8_6_4_l88_88599

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle_8_6_4 : can_form_triangle 8 6 4 :=
by
  unfold can_form_triangle
  simp
  exact ⟨by linarith, by linarith, by linarith⟩

end can_form_triangle_8_6_4_l88_88599


namespace Anthony_vs_Jim_l88_88983

variable (Scott_pairs : ℕ)
variable (Anthony_pairs : ℕ)
variable (Jim_pairs : ℕ)

axiom Scott_value : Scott_pairs = 7
axiom Anthony_value : Anthony_pairs = 3 * Scott_pairs
axiom Jim_value : Jim_pairs = Anthony_pairs - 2

theorem Anthony_vs_Jim (Scott_pairs Anthony_pairs Jim_pairs : ℕ) 
  (Scott_value : Scott_pairs = 7) 
  (Anthony_value : Anthony_pairs = 3 * Scott_pairs) 
  (Jim_value : Jim_pairs = Anthony_pairs - 2) :
  Anthony_pairs - Jim_pairs = 2 := 
sorry

end Anthony_vs_Jim_l88_88983


namespace shanghai_mock_exam_problem_l88_88863

noncomputable def a_n : ℕ → ℝ := sorry -- Defines the arithmetic sequence 

theorem shanghai_mock_exam_problem 
  (a_is_arithmetic : ∃ d a₀, ∀ n, a_n n = a₀ + n * d)
  (h₁ : a_n 1 + a_n 3 + a_n 5 = 9)
  (h₂ : a_n 2 + a_n 4 + a_n 6 = 15) :
  a_n 3 + a_n 4 = 8 := 
  sorry

end shanghai_mock_exam_problem_l88_88863


namespace cubic_third_root_l88_88294

theorem cubic_third_root (f : ℝ → ℝ) (line_eq : ℝ → ℝ) (x1 x2 x3 : ℝ) :
  f = λ x, x^3 + x + 2014 →
  line_eq = λ x, 877 * x - 7506 →
  f 20 = 10034 →
  f 14 = 4772 →
  x1 = 20 →
  x2 = 14 →
  x3 = -34 →
  (x1 - 20) * (x1 - 14) * (x1 - (-34)) = 0 := sorry

end cubic_third_root_l88_88294


namespace min_b_l88_88634

-- Definitions
def S (n : ℕ) : ℤ := 2^n - 1
def a (n : ℕ) : ℤ :=
  if n = 1 then 1 else 2^(n-1)
def b (n : ℕ) : ℤ := (a n)^2 - 7 * (a n) + 6

-- Theorem
theorem min_b : ∃ n : ℕ, (b n = -6) :=
sorry

end min_b_l88_88634


namespace part1_l88_88550

theorem part1 (m : ℝ) (a b : ℝ) (h : m > 0) : 
  ( (a + m * b) / (1 + m) )^2 ≤ (a^2 + m * b^2) / (1 + m) :=
sorry

end part1_l88_88550


namespace sequence_inequality_l88_88349

theorem sequence_inequality (a : ℕ → ℤ) (h₀ : a 1 > a 0) 
  (h₁ : ∀ n : ℕ, n ≥ 1 → a (n+1) = 3 * a n - 2 * a (n-1)) : 
  a 100 > 2^99 := 
sorry

end sequence_inequality_l88_88349


namespace probability_sum_greater_than_four_l88_88260

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88260


namespace proof_problem_l88_88502

open Real

-- Let p and q be the given propositions
def p := ∃ x_0 ∈ Iio 0, 2^x_0 < 3^x_0
def q := ∀ x ∈ Ioo 0 (π / 2), sin x < x

-- We need to prove the compound proposition
theorem proof_problem : (¬ p) ∧ q :=
by sorry

end proof_problem_l88_88502


namespace intersection_of_squares_perimeter_l88_88207

noncomputable def perimeter_of_rectangle (side1 side2 : ℝ) : ℝ :=
2 * (side1 + side2)

theorem intersection_of_squares_perimeter
  (side_length : ℝ)
  (diagonal : ℝ)
  (distance_between_centers : ℝ)
  (h1 : 4 * side_length = 8) 
  (h2 : (side1^2 + side2^2) = diagonal^2)
  (h3 : (2 - side1)^2 + (2 - side2)^2 = distance_between_centers^2) : 
10 * (perimeter_of_rectangle side1 side2) = 25 :=
sorry

end intersection_of_squares_perimeter_l88_88207


namespace Bruce_Anne_combined_cleaning_time_l88_88466

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l88_88466


namespace jackson_points_l88_88655

theorem jackson_points (team_total_points : ℕ)
                       (num_other_players : ℕ)
                       (average_points_other_players : ℕ)
                       (points_other_players: ℕ)
                       (points_jackson: ℕ)
                       (h_team_total_points : team_total_points = 65)
                       (h_num_other_players : num_other_players = 5)
                       (h_average_points_other_players : average_points_other_players = 6)
                       (h_points_other_players : points_other_players = num_other_players * average_points_other_players)
                       (h_points_total: points_jackson + points_other_players = team_total_points) :
  points_jackson = 35 :=
by
  -- proof will be done here
  sorry

end jackson_points_l88_88655


namespace samuel_faster_than_sarah_l88_88825

-- Definitions based on the conditions
def time_samuel : ℝ := 30
def time_sarah : ℝ := 1.3 * 60

-- The theorem to prove that Samuel finished his homework 48 minutes faster than Sarah
theorem samuel_faster_than_sarah : (time_sarah - time_samuel) = 48 := by
  sorry

end samuel_faster_than_sarah_l88_88825


namespace relationship_M_N_l88_88677

-- Define the sets M and N based on the conditions
def M : Set ℕ := {x | ∃ n : ℕ, x = 3^n}
def N : Set ℕ := {x | ∃ n : ℕ, x = 3 * n}

-- The statement to be proved
theorem relationship_M_N : ¬ (M ⊆ N) ∧ ¬ (N ⊆ M) :=
by
  sorry

end relationship_M_N_l88_88677


namespace probability_sum_greater_than_four_l88_88259

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88259


namespace range_of_a_l88_88719

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → x^2 - 2*x + a ≥ 0) → a ≥ 1 :=
by
  sorry

end range_of_a_l88_88719


namespace find_eccentricity_find_equation_l88_88777

open Real

-- Conditions for the first question
def is_ellipse (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def are_focus (a b : ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  F1 = ( - sqrt (a^2 - b^2), 0) ∧ F2 = (sqrt (a^2 - b^2), 0)

def arithmetic_sequence (a b : ℝ) (A B : ℝ × ℝ) (F1 : ℝ × ℝ) : Prop :=
  let dist_AF1 := abs (A.1 - F1.1)
  let dist_BF1 := abs (B.1 - F1.1)
  let dist_AB := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (dist_AF1 + dist_AB + dist_BF1 = 4 * a) ∧
  (dist_AF1 + dist_BF1 = 2 * dist_AB)

-- Proof statement for the eccentricity
theorem find_eccentricity (a b : ℝ) (F1 F2 A B : ℝ × ℝ)
  (h1 : a > b) (h2 : b > 0) (h3 : is_ellipse a b)
  (h4 : are_focus a b F1 F2)
  (h5 : arithmetic_sequence a b A B F1) :
  ∃ e : ℝ, e = sqrt 2 / 2 :=
sorry

-- Conditions for the second question
def geometric_property (a b : ℝ) (A B P : ℝ × ℝ) : Prop :=
  ∀ x y : ℝ, P = (0, -1) → 
             (x^2 / a^2) + (y^2 / b^2) = 1 → 
             abs ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 
             abs ((P.1 - B.1)^2 + (P.2 - B.2)^2)

-- Proof statement for the equation of the ellipse
theorem find_equation (a b : ℝ) (A B P : ℝ × ℝ)
  (h1 : a = 3 * sqrt 2) (h2 : b = 3) (h3 : P = (0, -1))
  (h4 : is_ellipse a b) (h5 : geometric_property a b A B P) :
  ∃ E : Prop, E = ((x : ℝ) * 2 / 18 + (y : ℝ) * 2 / 9 = 1) :=
sorry

end find_eccentricity_find_equation_l88_88777


namespace problem_l88_88478

def binom (n k : ℕ) : ℕ := n.choose k

def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem problem : binom 10 3 * perm 8 2 = 6720 := by
  sorry

end problem_l88_88478


namespace sequence_contains_at_most_one_square_l88_88397

theorem sequence_contains_at_most_one_square 
  (a : ℕ → ℕ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∀ m n, (m ≠ n) → ¬ (∃ k, a m = k^2 ∧ a n = k^2) :=
sorry

end sequence_contains_at_most_one_square_l88_88397


namespace triangle_angle_range_l88_88044

theorem triangle_angle_range (α β γ : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α = 2 * γ)
  (h3 : α ≥ β)
  (h4 : β ≥ γ) :
  45 ≤ β ∧ β ≤ 72 := 
sorry

end triangle_angle_range_l88_88044


namespace art_gallery_total_l88_88457

theorem art_gallery_total (A : ℕ) (h₁ : (1 / 3) * A = D)
                         (h₂ : (1 / 6) * D = sculptures_on_display)
                         (h₃ : (1 / 3) * N = paintings_not_on_display)
                         (h₄ : (2 / 3) * N = 1200)
                         (D = (1 / 3) * A)
                         (N = A - D)
                         (N = (2 / 3) * A) :
                         A = 2700 :=
by
  sorry

end art_gallery_total_l88_88457


namespace intersection_point_l88_88698

theorem intersection_point (x y : ℝ) (h1 : y = x + 1) (h2 : y = -x + 1) : (x = 0) ∧ (y = 1) := 
by
  sorry

end intersection_point_l88_88698


namespace least_possible_integer_discussed_l88_88735
open Nat

theorem least_possible_integer_discussed (N : ℕ) (H : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → k ≠ 8 ∧ k ≠ 9 → k ∣ N) : N = 2329089562800 :=
sorry

end least_possible_integer_discussed_l88_88735


namespace product_of_fractions_l88_88572

theorem product_of_fractions :
  (1 / 2) * (3 / 5) * (5 / 6) = 1 / 4 := 
by
  sorry

end product_of_fractions_l88_88572


namespace subway_distance_per_minute_l88_88724

theorem subway_distance_per_minute :
  let total_distance := 120 -- kilometers
  let total_time := 110 -- minutes (1 hour and 50 minutes)
  let bus_time := 70 -- minutes (1 hour and 10 minutes)
  let bus_distance := (14 * 40.8) / 6 -- kilometers
  let subway_distance := total_distance - bus_distance -- kilometers
  let subway_time := total_time - bus_time -- minutes
  let distance_per_minute := subway_distance / subway_time
  distance_per_minute = 0.62 := 
by
  sorry

end subway_distance_per_minute_l88_88724


namespace gcd_390_455_546_l88_88701

theorem gcd_390_455_546 :
  Nat.gcd (Nat.gcd 390 455) 546 = 13 := 
sorry

end gcd_390_455_546_l88_88701


namespace simplify_cube_root_21952000_l88_88988

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem simplify_cube_root_21952000 : 
  cube_root 21952000 = 280 := 
by {
  sorry
}

end simplify_cube_root_21952000_l88_88988


namespace range_of_a_l88_88119

variable (f : ℝ → ℝ)
variable (a : ℝ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def holds_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, (1/2) ≤ x ∧ x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (h1 : is_even f)
                   (h2 : is_increasing_on_nonneg f)
                   (h3 : holds_on_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l88_88119


namespace MinValue_x3y2z_l88_88013

theorem MinValue_x3y2z (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : 1/x + 1/y + 1/z = 6) : x^3 * y^2 * z ≥ 1 / 108 :=
by
  sorry

end MinValue_x3y2z_l88_88013


namespace area_triangle_formed_by_line_l88_88033

theorem area_triangle_formed_by_line (b : ℝ) (h : (1 / 2) * |b * (-b / 2)| > 1) : b < -2 ∨ b > 2 :=
by 
  sorry

end area_triangle_formed_by_line_l88_88033


namespace fraction_equation_solution_l88_88645

theorem fraction_equation_solution (x y : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 5) (hy1 : y ≠ 0) (hy2 : y ≠ 7)
  (h : (3 / x) + (2 / y) = 1 / 3) : 
  x = (9 * y) / (y - 6) :=
sorry

end fraction_equation_solution_l88_88645


namespace fraction_representation_of_3_36_l88_88277

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l88_88277


namespace product_of_constants_l88_88107

theorem product_of_constants (x t a b : ℤ) (h1 : x^2 + t * x - 12 = (x + a) * (x + b)) :
  ∃ ts : Finset ℤ, ∏ t in ts, t = 1936 :=
by
  sorry

end product_of_constants_l88_88107


namespace weight_of_second_piece_l88_88070

-- Given conditions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

def weight (density : ℚ) (area : ℕ) : ℚ := density * area

-- Given dimensions and weight of the first piece
def length1 : ℕ := 4
def width1 : ℕ := 3
def area1 : ℕ := area length1 width1
def weight1 : ℚ := 18

-- Given dimensions of the second piece
def length2 : ℕ := 6
def width2 : ℕ := 4
def area2 : ℕ := area length2 width2

-- Uniform density implies a proportional relationship between area and weight
def density1 : ℚ := weight1 / area1

-- The main theorem to prove
theorem weight_of_second_piece :
  weight density1 area2 = 36 :=
by
  -- use sorry to skip the proof
  sorry

end weight_of_second_piece_l88_88070


namespace geometric_sequence_theorem_l88_88809

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n+1) = a n * r

def holds_condition (a : ℕ → ℝ) : Prop := 
  a 1 * a 10 = -2

theorem geometric_sequence_theorem (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_cond : holds_condition a) : a 4 * a 7 = -2 :=
by
  sorry

end geometric_sequence_theorem_l88_88809


namespace maria_traveled_portion_of_distance_l88_88901

theorem maria_traveled_portion_of_distance (total_distance first_stop remaining_distance_to_destination : ℝ) 
  (h1 : total_distance = 560) 
  (h2 : first_stop = total_distance / 2) 
  (h3 : remaining_distance_to_destination = 210) : 
  ((first_stop - (first_stop - (remaining_distance_to_destination + (first_stop - total_distance / 2)))) / (total_distance - first_stop)) = 1 / 4 :=
by
  sorry

end maria_traveled_portion_of_distance_l88_88901


namespace last_two_digits_of_7_pow_5_pow_6_l88_88615

theorem last_two_digits_of_7_pow_5_pow_6 : (7 ^ (5 ^ 6)) % 100 = 7 := 
  sorry

end last_two_digits_of_7_pow_5_pow_6_l88_88615


namespace total_credit_hours_l88_88017

def max_courses := 40
def max_courses_per_semester := 5
def max_courses_per_semester_credit := 3
def max_additional_courses_last_semester := 2
def max_additional_course_credit := 4
def sid_courses_multiplier := 4
def sid_additional_courses_multiplier := 2

theorem total_credit_hours (total_max_courses : Nat) 
                           (avg_max_courses_per_semester : Nat) 
                           (max_course_credit : Nat) 
                           (extra_max_courses_last_sem : Nat) 
                           (extra_max_course_credit : Nat) 
                           (sid_courses_mult : Nat) 
                           (sid_extra_courses_mult : Nat) 
                           (max_total_courses : total_max_courses = max_courses)
                           (max_avg_courses_per_semester : avg_max_courses_per_semester = max_courses_per_semester)
                           (max_course_credit_def : max_course_credit = max_courses_per_semester_credit)
                           (extra_max_courses_last_sem_def : extra_max_courses_last_sem = max_additional_courses_last_semester)
                           (extra_max_courses_credit_def : extra_max_course_credit = max_additional_course_credit)
                           (sid_courses_mult_def : sid_courses_mult = sid_courses_multiplier)
                           (sid_extra_courses_mult_def : sid_extra_courses_mult = sid_additional_courses_multiplier) : 
  total_max_courses * max_course_credit + extra_max_courses_last_sem * extra_max_course_credit + 
  (sid_courses_mult * total_max_courses - sid_extra_courses_mult * extra_max_courses_last_sem) * max_course_credit + sid_extra_courses_mult * extra_max_courses_last_sem * extra_max_course_credit = 606 := 
  by 
    sorry

end total_credit_hours_l88_88017


namespace exists_function_passing_through_point_l88_88723

-- Define the function that satisfies f(2) = 0
theorem exists_function_passing_through_point : ∃ f : ℝ → ℝ, f 2 = 0 := 
sorry

end exists_function_passing_through_point_l88_88723


namespace students_in_photo_l88_88414

theorem students_in_photo (m n : ℕ) (h1 : n = m + 5) (h2 : n = m + 5 ∧ m = 3) : 
  m * n = 24 :=
by
  -- h1: n = m + 5    (new row is 4 students fewer)
  -- h2: m = 3        (all rows have the same number of students after rearrangement)
  -- Prove m * n = 24
  sorry

end students_in_photo_l88_88414


namespace ab_value_l88_88182

theorem ab_value (a b : ℝ) (h1 : 3^a = 81^(b + 2)) (h2 : 125^b = 5^(a - 3)) : a * b = 60 := by
  sorry

end ab_value_l88_88182


namespace total_items_given_out_l88_88019

-- Miss Davis gave 15 popsicle sticks and 20 straws to each group.
def popsicle_sticks_per_group := 15
def straws_per_group := 20
def items_per_group := popsicle_sticks_per_group + straws_per_group

-- There are 10 groups in total.
def number_of_groups := 10

-- Prove the total number of items given out equals 350.
theorem total_items_given_out : items_per_group * number_of_groups = 350 :=
by
  sorry

end total_items_given_out_l88_88019


namespace geometric_sequence_sum_l88_88779

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_a1 : a 1 = 1)
  (h_sum : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end geometric_sequence_sum_l88_88779


namespace sales_of_stationery_accessories_l88_88406

def percentage_of_sales_notebooks : ℝ := 25
def percentage_of_sales_markers : ℝ := 40
def total_sales_percentage : ℝ := 100

theorem sales_of_stationery_accessories : 
  percentage_of_sales_notebooks + percentage_of_sales_markers = 65 → 
  total_sales_percentage - (percentage_of_sales_notebooks + percentage_of_sales_markers) = 35 :=
by
  sorry

end sales_of_stationery_accessories_l88_88406


namespace linear_function_expression_l88_88486

theorem linear_function_expression (k b : ℝ) (h : ∀ x : ℝ, (1 ≤ x ∧ x ≤ 4 → 3 ≤ k * x + b ∧ k * x + b ≤ 6)) :
  (k = 1 ∧ b = 2) ∨ (k = -1 ∧ b = 7) :=
by
  sorry

end linear_function_expression_l88_88486


namespace gain_percentage_for_40_clocks_is_10_l88_88300

-- Condition: Cost price per clock
def cost_price := 79.99999999999773

-- Condition: Selling price of 50 clocks at a gain of 20%
def selling_price_50 := 50 * cost_price * 1.20

-- Uniform profit condition
def uniform_profit_total := 90 * cost_price * 1.15

-- Given total revenue difference Rs. 40
def total_revenue := uniform_profit_total + 40

-- Question: Prove that selling price of 40 clocks leads to 10% gain
theorem gain_percentage_for_40_clocks_is_10 :
    40 * cost_price * 1.10 = total_revenue - selling_price_50 :=
by
  sorry

end gain_percentage_for_40_clocks_is_10_l88_88300


namespace min_value_of_f_in_interval_l88_88702

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) ∧ f x = -Real.sqrt 2 / 2 :=
by
  sorry

end min_value_of_f_in_interval_l88_88702


namespace evaluate_complex_power_expression_l88_88616

theorem evaluate_complex_power_expression : (i : ℂ)^23 + ((i : ℂ)^105 * (i : ℂ)^17) = -i - 1 := by
  sorry

end evaluate_complex_power_expression_l88_88616


namespace train_length_250_meters_l88_88882

open Real

noncomputable def speed_in_ms (speed_km_hr: ℝ): ℝ :=
  speed_km_hr * (1000 / 3600)

noncomputable def length_of_train (speed: ℝ) (time: ℝ): ℝ :=
  speed * time

theorem train_length_250_meters (speed_km_hr: ℝ) (time_seconds: ℝ) :
  speed_km_hr = 40 → time_seconds = 22.5 → length_of_train (speed_in_ms speed_km_hr) time_seconds = 250 :=
by
  intros
  sorry

end train_length_250_meters_l88_88882


namespace ball_bounce_height_l88_88867

theorem ball_bounce_height (b : ℕ) (h₀: ℝ) (r: ℝ) (h_final: ℝ) :
  h₀ = 200 ∧ r = 3 / 4 ∧ h_final = 25 →
  200 * (3 / 4) ^ b < 25 ↔ b ≥ 25 := by
  sorry

end ball_bounce_height_l88_88867


namespace remainder_div_l88_88549

theorem remainder_div (N : ℕ) (n : ℕ) : 
  (N % 2^n) = (N % 10^n % 2^n) ∧ (N % 5^n) = (N % 10^n % 5^n) := by
  sorry

end remainder_div_l88_88549


namespace polynomials_with_three_different_roots_count_l88_88613

theorem polynomials_with_three_different_roots_count :
  (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6: ℕ), 
    a_0 = 0 ∧ 
    (a_6 = 0 ∨ a_6 = 1) ∧
    (a_5 = 0 ∨ a_5 = 1) ∧
    (a_4 = 0 ∨ a_4 = 1) ∧
    (a_3 = 0 ∨ a_3 = 1) ∧
    (a_2 = 0 ∨ a_2 = 1) ∧
    (a_1 = 0 ∨ a_1 = 1) ∧
    (1 + a_6 + a_5 + a_4 + a_3 + a_2 + a_1) % 2 = 0 ∧
    (1 - a_6 + a_5 - a_4 + a_3 - a_2 + a_1) % 2 = 0) -> 
  ∃ (n : ℕ), n = 8 :=
sorry

end polynomials_with_three_different_roots_count_l88_88613


namespace max_value_of_determinant_l88_88619

noncomputable def determinant_of_matrix (θ : ℝ) : ℝ :=
  Matrix.det ![
    ![1, 1, 1],
    ![1, 1 + Real.sin (2 * θ), 1],
    ![1, 1, 1 + Real.cos (2 * θ)]
  ]

theorem max_value_of_determinant : 
  ∃ θ : ℝ, (∀ θ : ℝ, determinant_of_matrix θ ≤ (1 / 2)) ∧ determinant_of_matrix (θ_at_maximum) = (1 / 2) :=
sorry

end max_value_of_determinant_l88_88619


namespace diophantine_solution_l88_88554

theorem diophantine_solution :
  ∃ (x y k : ℤ), 1990 * x - 173 * y = 11 ∧ x = -22 + 173 * k ∧ y = 253 - 1990 * k :=
by {
  sorry
}

end diophantine_solution_l88_88554


namespace only_D_is_odd_l88_88574

-- Define the functions
def fA (x : ℝ) := if x ≥ 0 then real.sqrt x else 0
def fB (x : ℝ) := abs (real.sin x)
def fC (x : ℝ) := real.cos x
def fD (x : ℝ) := real.exp x - real.exp (-x)

-- Definitions needed for the statement
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x
def is_neither_odd_nor_even (f : ℝ → ℝ) := ¬(is_odd_function f) ∧ ¬(is_even_function f)

-- The statement
theorem only_D_is_odd :
  is_neither_odd_nor_even fA ∧
  is_even_function fB ∧
  is_even_function fC ∧
  is_odd_function fD :=
by sorry

end only_D_is_odd_l88_88574


namespace mul_65_35_l88_88895

theorem mul_65_35 : (65 * 35) = 2275 := by
  -- define a and b
  let a := 50
  let b := 15
  -- use the equivalence (a + b) and (a - b)
  have h1 : 65 = a + b := by rfl
  have h2 : 35 = a - b := by rfl
  -- use the difference of squares formula
  have h_diff_squares : (a + b) * (a - b) = a^2 - b^2 := by sorry
  -- calculate each square
  have ha_sq : a^2 = 2500 := by sorry
  have hb_sq : b^2 = 225 := by sorry
  -- combine the results
  have h_result : a^2 - b^2 = 2500 - 225 := by sorry
  -- finish the proof
  have final_result : (65 * 35) = 2275 := by sorry
  exact final_result

end mul_65_35_l88_88895


namespace inverse_proportion_inequality_l88_88117

variable (x1 x2 k : ℝ)
variable (y1 y2 : ℝ)

theorem inverse_proportion_inequality (h1 : x1 < 0) (h2 : 0 < x2) (hk : k < 0)
  (hy1 : y1 = k / x1) (hy2 : y2 = k / x2) : y2 < 0 ∧ 0 < y1 := 
by sorry

end inverse_proportion_inequality_l88_88117


namespace geom_seq_solution_l88_88711

theorem geom_seq_solution (a b x y : ℝ) 
  (h1 : x * (1 + y + y^2) = a) 
  (h2 : x^2 * (1 + y^2 + y^4) = b) :
  x = 1 / (4 * a) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨ 
  x = 1 / (4 * a) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∧
  y = 1 / (2 * (a^2 - b)) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨
  y = 1 / (2 * (a^2 - b)) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) := 
  sorry

end geom_seq_solution_l88_88711


namespace range_of_a_l88_88516

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * a * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x, 1 < x ∧ x < 4 → deriv (f a) x < 0) ∧
  (∀ x, 6 < x → deriv (f a) x > 0) →
  5 ≤ a ∧ a ≤ 7 :=
sorry

end range_of_a_l88_88516


namespace chocolate_bars_per_box_l88_88736

theorem chocolate_bars_per_box (total_chocolate_bars num_small_boxes : ℕ) (h1 : total_chocolate_bars = 300) (h2 : num_small_boxes = 15) : 
  total_chocolate_bars / num_small_boxes = 20 :=
by 
  sorry

end chocolate_bars_per_box_l88_88736


namespace probability_sum_greater_than_four_l88_88257

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88257


namespace probability_sum_greater_than_four_is_5_over_6_l88_88249

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l88_88249


namespace length_of_arc_correct_l88_88407

open Real

noncomputable def length_of_arc (r θ : ℝ) := θ * r

theorem length_of_arc_correct (A r θ : ℝ) (hA : A = (θ / (2 * π)) * (π * r^2)) (hr : r = 5) (hA_val : A = 13.75) :
  length_of_arc r θ = 5.5 :=
by
  -- Proof steps are omitted
  sorry

end length_of_arc_correct_l88_88407


namespace find_c_l88_88937

theorem find_c (c d : ℝ) (h : ∀ x : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) : c = 16 :=
sorry

end find_c_l88_88937


namespace books_loaned_out_l88_88858

theorem books_loaned_out (initial_books loaned_books returned_percentage end_books missing_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : end_books = 66)
  (h3 : returned_percentage = 70)
  (h4 : initial_books - end_books = missing_books)
  (h5 : missing_books = (loaned_books * (100 - returned_percentage)) / 100):
  loaned_books = 30 :=
by
  sorry

end books_loaned_out_l88_88858


namespace additional_laps_needed_l88_88821

-- Definitions of problem conditions
def total_required_distance : ℕ := 2400
def lap_length : ℕ := 150
def madison_laps : ℕ := 6
def gigi_laps : ℕ := 6

-- Target statement to prove the number of additional laps needed
theorem additional_laps_needed : (total_required_distance - (madison_laps + gigi_laps) * lap_length) / lap_length = 4 := by
  sorry

end additional_laps_needed_l88_88821


namespace min_value_reciprocal_l88_88111

theorem min_value_reciprocal (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_eq : 2 * a + b = 4) : 
  (∀ (x : ℝ), (∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 4 -> x ≥ 1 / (2 * a * b)) -> x ≥ 1 / 2) := 
by
  sorry

end min_value_reciprocal_l88_88111


namespace mushroom_problem_l88_88489

variables (x1 x2 x3 x4 : ℕ)

theorem mushroom_problem
  (h1 : x1 + x2 = 6)
  (h2 : x1 + x3 = 7)
  (h3 : x2 + x3 = 9)
  (h4 : x2 + x4 = 11)
  (h5 : x3 + x4 = 12)
  (h6 : x1 + x4 = 9) :
  x1 = 2 ∧ x2 = 4 ∧ x3 = 5 ∧ x4 = 7 := 
  by
    sorry

end mushroom_problem_l88_88489


namespace geometric_sequence_sum_l88_88540

theorem geometric_sequence_sum (a : ℕ → ℤ)
  (h1 : a 0 = 1)
  (h_q : ∀ n, a (n + 1) = a n * -2) :
  a 0 + |a 1| + a 2 + |a 3| = 15 := by
  sorry

end geometric_sequence_sum_l88_88540


namespace simplify_336_to_fraction_l88_88275

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l88_88275


namespace sheila_will_attend_picnic_l88_88212

noncomputable def prob_sheila_attends_picnic (P_Rain P_Attend_if_Rain P_Attend_if_Sunny P_Special : ℝ) : ℝ :=
  let P_Sunny := 1 - P_Rain
  let P_Rain_and_Attend := P_Rain * P_Attend_if_Rain
  let P_Sunny_and_Attend := P_Sunny * P_Attend_if_Sunny
  let P_Attends := P_Rain_and_Attend + P_Sunny_and_Attend + P_Special - P_Rain_and_Attend * P_Special - P_Sunny_and_Attend * P_Special
  P_Attends

theorem sheila_will_attend_picnic :
  prob_sheila_attends_picnic 0.3 0.25 0.7 0.15 = 0.63025 :=
by
  sorry

end sheila_will_attend_picnic_l88_88212


namespace cube_sufficient_but_not_necessary_l88_88114

theorem cube_sufficient_but_not_necessary (x : ℝ) : (x^3 > 27 → |x| > 3) ∧ (¬(|x| > 3 → x^3 > 27)) :=
by
  sorry

end cube_sufficient_but_not_necessary_l88_88114


namespace arithmetic_geometric_sequence_problem_l88_88215

variable {a_n : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 0 + n * d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = (n * (a_n 0 + a_n (n-1))) / 2

def forms_geometric_sequence (a1 a3 a4 : ℝ) :=
  a3^2 = a1 * a4

-- The main proof statement
theorem arithmetic_geometric_sequence_problem
        (h_arith : is_arithmetic_sequence a_n)
        (h_sum : sum_of_first_n_terms a_n S)
        (h_geom : forms_geometric_sequence (a_n 0) (a_n 2) (a_n 3)) :
        (S 3 - S 2) / (S 5 - S 3) = 2 ∨ (S 3 - S 2) / (S 5 - S 3) = 1 / 2 :=
  sorry

end arithmetic_geometric_sequence_problem_l88_88215


namespace range_of_a_l88_88336

def f (x : ℝ) : ℝ := 3 * x * |x|

theorem range_of_a : {a : ℝ | f (1 - a) + f (2 * a) < 0 } = {a : ℝ | a < -1} :=
by
  sorry

end range_of_a_l88_88336


namespace Sasha_earnings_proof_l88_88551

def Monday_hours : ℕ := 90  -- 1.5 hours * 60 minutes/hour
def Tuesday_minutes : ℕ := 75  -- 1 hour * 60 minutes/hour + 15 minutes
def Wednesday_minutes : ℕ := 115  -- 11:10 AM - 9:15 AM
def Thursday_minutes : ℕ := 45

def total_minutes_worked : ℕ := Monday_hours + Tuesday_minutes + Wednesday_minutes + Thursday_minutes

def hourly_rate : ℚ := 4.50
def total_hours : ℚ := total_minutes_worked / 60

def weekly_earnings : ℚ := total_hours * hourly_rate

theorem Sasha_earnings_proof : weekly_earnings = 24 := by
  sorry

end Sasha_earnings_proof_l88_88551


namespace new_student_weight_l88_88559

theorem new_student_weight :
  ∀ (W : ℝ) (total_weight_19 : ℝ) (total_weight_20 : ℝ),
    total_weight_19 = 19 * 15 →
    total_weight_20 = 20 * 14.8 →
    total_weight_19 + W = total_weight_20 →
    W = 11 :=
by
  intros W total_weight_19 total_weight_20 h1 h2 h3
  -- Skipping the proof as instructed
  sorry

end new_student_weight_l88_88559


namespace score_in_first_round_l88_88291

theorem score_in_first_round (cards : List ℕ) (scores : List ℕ) 
  (total_rounds : ℕ) (last_round_score : ℕ) (total_score : ℕ) : 
  cards = [2, 4, 7, 13] ∧ scores = [16, 17, 21, 24] ∧ total_rounds = 3 ∧ last_round_score = 2 ∧ total_score = 16 →
  ∃ first_round_score, first_round_score = 7 := by
  sorry

end score_in_first_round_l88_88291


namespace robert_finite_moves_l88_88986

noncomputable def onlyFiniteMoves (numbers : List ℕ) : Prop :=
  ∀ (a b : ℕ), a > b → ∃ (moves : ℕ), moves < numbers.length

theorem robert_finite_moves (numbers : List ℕ) :
  onlyFiniteMoves numbers := sorry

end robert_finite_moves_l88_88986


namespace scooterValue_after_4_years_with_maintenance_l88_88567

noncomputable def scooterDepreciation (initial_value : ℝ) (years : ℕ) : ℝ :=
  initial_value * ((3 : ℝ) / 4) ^ years

theorem scooterValue_after_4_years_with_maintenance (M : ℝ) :
  scooterDepreciation 40000 4 - 4 * M = 12656.25 - 4 * M :=
by
  sorry

end scooterValue_after_4_years_with_maintenance_l88_88567


namespace steve_take_home_pay_l88_88401

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end steve_take_home_pay_l88_88401


namespace sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l88_88413

theorem sum_of_digits_of_greatest_prime_divisor_of_16385_is_13 : 
  ∃ p : ℕ, (p ∣ 16385 ∧ Nat.Prime p ∧ (∀ q : ℕ, q ∣ 16385 → Nat.Prime q → q ≤ p)) ∧ (Nat.digits 10 p).sum = 13 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_16385_is_13_l88_88413


namespace isabel_camera_pics_l88_88663

-- Conditions
def phone_pics := 2
def albums := 3
def pics_per_album := 2

-- Define the total pictures and camera pictures
def total_pics := albums * pics_per_album
def camera_pics := total_pics - phone_pics

theorem isabel_camera_pics : camera_pics = 4 :=
by
  -- The goal is translated from the correct answer in step b)
  sorry

end isabel_camera_pics_l88_88663


namespace average_salary_excluding_manager_l88_88527

theorem average_salary_excluding_manager (A : ℝ) 
  (num_employees : ℝ := 20)
  (manager_salary : ℝ := 3300)
  (salary_increase : ℝ := 100)
  (total_salary_with_manager : ℝ := 21 * (A + salary_increase)) :
  20 * A + manager_salary = total_salary_with_manager → A = 1200 := 
by
  intro h
  sorry

end average_salary_excluding_manager_l88_88527


namespace cone_lateral_area_l88_88035

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  (1 / 2) * (2 * Real.pi * r) * l = 15 * Real.pi :=
by
  rw [h_r, h_l]
  sorry

end cone_lateral_area_l88_88035


namespace find_a_for_square_binomial_l88_88914

theorem find_a_for_square_binomial (a : ℚ) : (∃ (r s : ℚ), a = r^2 ∧ 20 = 2 * r * s ∧ 9 = s^2) → a = 100 / 9 :=
by
  intro h
  cases' h with r hr
  cases' hr with s hs
  cases' hs with ha1 hs1
  cases' hs1 with ha2 ha3
  have s_val : s = 3 ∨ s = -3 := by
    have s2_eq := eq_of_sq_eq_sq ha3
    subst s; split; linarith; linarith
  cases s_val with s_eq3 s_eq_neg3
  -- case s = 3
  { rw [s_eq3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }
  -- case s = -3
  { rw [s_eq_neg3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }

end find_a_for_square_binomial_l88_88914


namespace min_value_of_function_l88_88342

theorem min_value_of_function (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, y = x + 1 / (x - 4) ∧ (∀ z : ℝ, z = x + 1 / (x - 4) → z ≥ 6) :=
sorry

end min_value_of_function_l88_88342


namespace tan_alpha_plus_pi_div_four_l88_88643

theorem tan_alpha_plus_pi_div_four (α : ℝ) (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3) : 
  Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_alpha_plus_pi_div_four_l88_88643


namespace one_minus_repeating_decimal_three_equals_two_thirds_l88_88095

-- Define the repeating decimal as a fraction
def repeating_decimal_three : ℚ := 1 / 3

-- Prove the desired equality
theorem one_minus_repeating_decimal_three_equals_two_thirds :
  1 - repeating_decimal_three = 2 / 3 :=
by
  sorry

end one_minus_repeating_decimal_three_equals_two_thirds_l88_88095


namespace smallest_positive_angle_l88_88267

def coterminal_angle (θ : ℤ) : ℤ := θ % 360

theorem smallest_positive_angle (θ : ℤ) (hθ : θ % 360 ≠ 0) : 
  0 < coterminal_angle θ ∧ coterminal_angle θ = 158 :=
by
  sorry

end smallest_positive_angle_l88_88267


namespace average_last_three_l88_88199

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l88_88199


namespace f_not_factorable_l88_88539

noncomputable def f (n : ℕ) (x : ℕ) : ℕ := x^n + 5 * x^(n - 1) + 3

theorem f_not_factorable (n : ℕ) (hn : n > 1) :
  ¬ ∃ g h : ℕ → ℕ, (∀ a b : ℕ, a ≠ 0 ∧ b ≠ 0 → g a * h b = f n a * f n b) ∧ 
    (∀ a b : ℕ, (g a = 0 ∧ h b = 0) → (a = 0 ∧ b = 0)) ∧ 
    (∃ pg qh : ℕ, pg ≥ 1 ∧ qh ≥ 1 ∧ g 1 = 1 ∧ h 1 = 1 ∧ (pg + qh = n)) := 
sorry

end f_not_factorable_l88_88539


namespace gcf_lcm_15_l88_88966

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_15 : 
  GCF (LCM 9 15) (LCM 10 21) = 15 :=
by 
  sorry

end gcf_lcm_15_l88_88966


namespace minimum_rows_l88_88447

theorem minimum_rows (n : ℕ) (C : ℕ → ℕ) (hC_bounds : ∀ i, 1 ≤ C i ∧ C i ≤ 39) 
  (hC_sum : (Finset.range n).sum C = 1990) :
  ∃ k, k = 12 ∧ ∀ (R : ℕ) (hR : R = 199), 
    ∀ (seating : ℕ → ℕ) (h_seating : ∀ i, seating i ≤ R) 
    (h_seating_capacity : (Finset.range k).sum seating = 1990),
    True := sorry

end minimum_rows_l88_88447


namespace number_53_in_sequence_l88_88960

theorem number_53_in_sequence (n : ℕ) (hn : n = 53) :
  let seq := (λ (k : ℕ), k + 1) 0 in
  (seq 52 = 53) :=
by
  sorry

end number_53_in_sequence_l88_88960


namespace salary_for_may_l88_88837

theorem salary_for_may (J F M A May : ℝ) 
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8200)
  (h3 : J = 5700) : 
  May = 6500 :=
by 
  have eq1 : J + F + M + A = 32000 := by
    linarith
  have eq2 : F + M + A + May = 32800 := by
    linarith
  have eq3 : May - J = 800 := by
    linarith [eq1, eq2]
  have eq4 : May = 6500 := by
    linarith [eq3, h3]
  exact eq4

end salary_for_may_l88_88837


namespace factorize_expr_l88_88761

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l88_88761


namespace min_value_hyperbola_l88_88001

open Real 

theorem min_value_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (3 * x^2 - 2 * y ≥ 143 / 12) ∧ 
                                          (∃ (y' : ℝ), y = y' ∧  3 * (2 + 2*y'^2)^2 - 2 * y' = 143 / 12) := 
by
  sorry

end min_value_hyperbola_l88_88001


namespace sum_of_ages_l88_88159

theorem sum_of_ages (age1 age2 age3 : ℕ) (h : age1 * age2 * age3 = 128) : age1 + age2 + age3 = 18 :=
sorry

end sum_of_ages_l88_88159


namespace lcm_of_two_numbers_l88_88290

theorem lcm_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 9) (h2 : a * b = 1800) : Nat.lcm a b = 200 :=
by
  sorry

end lcm_of_two_numbers_l88_88290


namespace discount_percentage_l88_88066

theorem discount_percentage (original_price new_price : ℕ) (h₁ : original_price = 120) (h₂ : new_price = 96) : 
  ((original_price - new_price) * 100 / original_price) = 20 := 
by
  -- sorry is used here to skip the proof
  sorry

end discount_percentage_l88_88066


namespace yvettes_final_bill_l88_88667

theorem yvettes_final_bill :
  let alicia : ℝ := 7.5
  let brant : ℝ := 10
  let josh : ℝ := 8.5
  let yvette : ℝ := 9
  let tip_percentage : ℝ := 0.2
  ∃ final_bill : ℝ, final_bill = (alicia + brant + josh + yvette) * (1 + tip_percentage) ∧ final_bill = 42 :=
by
  sorry

end yvettes_final_bill_l88_88667


namespace factorization_correct_l88_88755

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l88_88755


namespace find_a_l88_88913

theorem find_a (a r s : ℚ) (h1 : a = r^2) (h2 : 20 = 2 * r * s) (h3 : 9 = s^2) : a = 100 / 9 := by
  sorry

end find_a_l88_88913


namespace decorations_cost_correct_l88_88379

def cost_of_decorations (num_tables : ℕ) (cost_tablecloth per_tablecloth : ℕ) (num_place_settings per_table : ℕ) (cost_place_setting per_setting : ℕ) (num_roses per_centerpiece : ℕ) (cost_rose per_rose : ℕ) (num_lilies per_centerpiece : ℕ) (cost_lily per_lily : ℕ) : ℕ :=
  let cost_roses := cost_rose * num_roses
  let cost_lilies := cost_lily * num_lilies
  let cost_settings := cost_place_setting * num_place_settings
  let cost_per_table := cost_roses + cost_lilies + cost_settings + cost_tablecloth
  num_tables * cost_per_table

theorem decorations_cost_correct :
  cost_of_decorations 20 25 4 10 10 5 15 4 = 3500 :=
by
  sorry

end decorations_cost_correct_l88_88379


namespace fifth_term_sequence_l88_88638

theorem fifth_term_sequence 
  (a : ℕ → ℤ)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 5 = -6 := 
by
  sorry

end fifth_term_sequence_l88_88638


namespace area_of_new_shape_l88_88304

noncomputable def unit_equilateral_triangle_area : ℝ :=
  (1 : ℝ)^2 * Real.sqrt 3 / 4

noncomputable def area_removed_each_step (k : ℕ) : ℝ :=
  3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def total_removed_area : ℝ :=
  ∑' k, 3 * (4 ^ (k - 1)) * (Real.sqrt 3 / (4 * (9 ^ k)))

noncomputable def final_area := unit_equilateral_triangle_area - total_removed_area

theorem area_of_new_shape :
  final_area = Real.sqrt 3 / 10 := sorry

end area_of_new_shape_l88_88304


namespace three_digit_difference_divisible_by_9_l88_88037

theorem three_digit_difference_divisible_by_9 :
  ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    (100 * a + 10 * b + c - (a + b + c)) % 9 = 0 :=
by
  intros a b c h
  sorry

end three_digit_difference_divisible_by_9_l88_88037


namespace problem_1_problem_2_l88_88126

def f (x : ℝ) : ℝ := |(1 - 2 * x)| - |(1 + x)|

theorem problem_1 :
  {x | f x ≥ 4} = {x | x ≤ -2 ∨ x ≥ 6} :=
sorry

theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, a^2 + 2 * a + |(1 + x)| > f x) → (a < -3 ∨ a > 1) :=
sorry

end problem_1_problem_2_l88_88126


namespace A_P_not_76_l88_88301

theorem A_P_not_76 :
    ∀ (w : ℕ), w > 0 → (2 * w^2 + 6 * w) ≠ 76 :=
by
  intro w hw
  sorry

end A_P_not_76_l88_88301


namespace fraction_reach_impossible_l88_88348

theorem fraction_reach_impossible :
  ¬ ∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end fraction_reach_impossible_l88_88348


namespace initial_number_of_girls_l88_88034

theorem initial_number_of_girls (n A : ℕ) (new_girl_weight : ℕ := 80) (original_girl_weight : ℕ := 40)
  (avg_increase : ℕ := 2)
  (condition : n * (A + avg_increase) - n * A = 40) :
  n = 20 :=
by
  sorry

end initial_number_of_girls_l88_88034


namespace solution_set_of_inequality_l88_88043

def fraction_inequality_solution : Set ℝ := {x : ℝ | -4 < x ∧ x < -1}

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 1 ↔ -4 < x ∧ x < -1 := by
sorry

end solution_set_of_inequality_l88_88043


namespace sum_of_two_numbers_l88_88420

theorem sum_of_two_numbers (x : ℤ) (sum certain value : ℤ) (h₁ : 25 - x = 5) : 25 + x = 45 := by
  sorry

end sum_of_two_numbers_l88_88420


namespace exists_equilateral_triangle_same_color_l88_88091

-- Define a type for colors
inductive Color
| red : Color
| blue : Color

-- Define our statement
-- Given each point in the plane is colored either red or blue,
-- there exists an equilateral triangle with vertices of the same color.
theorem exists_equilateral_triangle_same_color (coloring : ℝ × ℝ → Color) : 
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ 
    dist p₁ p₂ = dist p₂ p₃ ∧ dist p₂ p₃ = dist p₃ p₁ ∧ 
    (coloring p₁ = coloring p₂ ∧ coloring p₂ = coloring p₃) :=
by
  sorry

end exists_equilateral_triangle_same_color_l88_88091


namespace sine_curve_transformation_l88_88124

theorem sine_curve_transformation (x y x' y' : ℝ) 
  (h1 : x' = (1 / 2) * x) 
  (h2 : y' = 3 * y) :
  (y = Real.sin x) ↔ (y' = 3 * Real.sin (2 * x')) := by 
  sorry

end sine_curve_transformation_l88_88124


namespace same_quadratic_function_b_l88_88791

theorem same_quadratic_function_b (a c b : ℝ) :
    (∀ x : ℝ, a * (x - 2)^2 + c = (2 * x - 5) * (x - b)) → b = 3 / 2 :=
by
  sorry

end same_quadratic_function_b_l88_88791


namespace gcd_fact_8_fact_6_sq_l88_88097

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_fact_8_fact_6_sq : gcd (factorial 8) ((factorial 6)^2) = 11520 := by
  sorry

end gcd_fact_8_fact_6_sq_l88_88097


namespace intersection_M_N_l88_88787

open Set

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | x^2 - 2*x - 3 < 0}
def intersection_sets := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = intersection_sets :=
  sorry

end intersection_M_N_l88_88787


namespace probability_sum_greater_than_four_is_5_over_6_l88_88247

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l88_88247


namespace exist_n_consecutive_not_perfect_power_l88_88373

theorem exist_n_consecutive_not_perfect_power (n : ℕ) (h : n > 0) : 
  ∃ m : ℕ, ∀ k : ℕ, k < n → ¬ (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (m + k) = a ^ b) :=
sorry

end exist_n_consecutive_not_perfect_power_l88_88373


namespace inequality_proof_l88_88178

theorem inequality_proof (a : ℝ) : (3 * a - 6) * (2 * a^2 - a^3) ≤ 0 := 
by 
  sorry

end inequality_proof_l88_88178


namespace number_of_ways_to_score_l88_88048

-- Define the conditions
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def score_red : ℕ := 2
def score_white : ℕ := 1
def total_balls : ℕ := 5
def min_score : ℕ := 7

-- Prove the equivalent proof problem
theorem number_of_ways_to_score :
  ∃ ways : ℕ, 
    (ways = ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
             (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
             (Nat.choose red_balls 2) * (Nat.choose white_balls 3))) ∧
    ways = 186 :=
by
  let ways := ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
               (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
               (Nat.choose red_balls 2) * (Nat.choose white_balls 3))
  use ways
  constructor
  . rfl
  . sorry

end number_of_ways_to_score_l88_88048


namespace sum_of_first_six_terms_l88_88656

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

def is_arithmetic_sequence (a_n : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a_n (n + 1) = a_n n + d 

theorem sum_of_first_six_terms (a_3 a_4 : ℕ) (h : a_3 + a_4 = 30) :
  ∃ a_n d, is_arithmetic_sequence a_n d ∧ 
  a_n 3 = a_3 ∧ a_n 4 = a_4 ∧ 
  (3 * (a_n 1 + (a_n 1 + 5 * d))) = 90 := 
sorry

end sum_of_first_six_terms_l88_88656


namespace checkerboard_sums_l88_88693

-- Define the dimensions and the arrangement of the checkerboard
def n : ℕ := 10
def board (i j : ℕ) : ℕ := i * n + j + 1

-- Define corner positions
def top_left_corner : ℕ := board 0 0
def top_right_corner : ℕ := board 0 (n - 1)
def bottom_left_corner : ℕ := board (n - 1) 0
def bottom_right_corner : ℕ := board (n - 1) (n - 1)

-- Sum of the corners
def corner_sum : ℕ := top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner

-- Define the positions of the main diagonals
def main_diagonal (i : ℕ) : ℕ := board i i
def anti_diagonal (i : ℕ) : ℕ := board i (n - 1 - i)

-- Sum of the main diagonals
def diagonal_sum : ℕ := (Finset.range n).sum main_diagonal + (Finset.range n).sum anti_diagonal - (main_diagonal 0 + main_diagonal (n - 1))

-- Statement to prove
theorem checkerboard_sums : corner_sum = 202 ∧ diagonal_sum = 101 :=
by
-- Proof is not required as per the instructions
sorry

end checkerboard_sums_l88_88693


namespace number_of_pages_correct_number_of_ones_correct_l88_88670

noncomputable def number_of_pages (total_digits : ℕ) : ℕ :=
  let single_digit_odd_pages := 5
  let double_digit_odd_pages := 45
  let triple_digit_odd_pages := (total_digits - (single_digit_odd_pages + 2 * double_digit_odd_pages)) / 3
  single_digit_odd_pages + double_digit_odd_pages + triple_digit_odd_pages

theorem number_of_pages_correct : number_of_pages 125 = 60 :=
by sorry

noncomputable def number_of_ones (total_digits : ℕ) : ℕ :=
  let ones_in_units_place := 12
  let ones_in_tens_place := 18
  let ones_in_hundreds_place := 10
  ones_in_units_place + ones_in_tens_place + ones_in_hundreds_place

theorem number_of_ones_correct : number_of_ones 125 = 40 :=
by sorry

end number_of_pages_correct_number_of_ones_correct_l88_88670


namespace sum_of_arithmetic_sequence_l88_88625

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (a_6 : a 6 = 2) : 
  (11 * (a 1 + (a 1 + 10 * ((a 6 - a 1) / 5))) / 2) = 22 :=
by
  sorry

end sum_of_arithmetic_sequence_l88_88625


namespace triangle_sequence_relation_l88_88932

theorem triangle_sequence_relation (b d c k : ℤ) (h₁ : b % d = 0) (h₂ : c % k = 0) (h₃ : b^2 + (b + 2*d)^2 = (c + 6*k)^2) :
  c = 0 :=
sorry

end triangle_sequence_relation_l88_88932


namespace max_value_proof_l88_88629

noncomputable def max_value_b_minus_a (a b : ℝ) : ℝ :=
  b - a

theorem max_value_proof (a b : ℝ) (h1 : a < 0) (h2 : ∀ x, (x^2 + 2017 * a) * (x + 2016 * b) ≥ 0) : max_value_b_minus_a a b ≤ 2017 :=
sorry

end max_value_proof_l88_88629


namespace bruce_anne_cleaning_house_l88_88468

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l88_88468


namespace infinite_hexagons_exist_l88_88798

theorem infinite_hexagons_exist :
  ∃ (a1 a2 a3 a4 a5 a6 : ℤ), 
  (a1 + a2 + a3 + a4 + a5 + a6 = 20) ∧
  (a1 ≤ a2) ∧ (a1 + a2 ≤ a3) ∧ (a2 + a3 ≤ a4) ∧
  (a3 + a4 ≤ a5) ∧ (a4 + a5 ≤ a6) ∧ (a1 + a2 + a3 + a4 + a5 > a6) :=
sorry

end infinite_hexagons_exist_l88_88798


namespace children_multiple_of_four_l88_88308

theorem children_multiple_of_four (C : ℕ) 
  (h_event : ∃ (A : ℕ) (T : ℕ), A = 12 ∧ T = 4 ∧ 12 % T = 0 ∧ C % T = 0) : ∃ k : ℕ, C = 4 * k :=
by
  obtain ⟨A, T, hA, hT, hA_div, hC_div⟩ := h_event
  rw [hA, hT] at *
  sorry

end children_multiple_of_four_l88_88308


namespace inequality_preserves_neg_half_l88_88356

variable (a b : ℝ)

theorem inequality_preserves_neg_half (h : a ≤ b) : -a / 2 ≥ -b / 2 := by
  sorry

end inequality_preserves_neg_half_l88_88356


namespace sum_of_ages_l88_88160

theorem sum_of_ages (age1 age2 age3 : ℕ) (h : age1 * age2 * age3 = 128) : age1 + age2 + age3 = 18 :=
sorry

end sum_of_ages_l88_88160


namespace C_completes_work_in_4_days_l88_88584

theorem C_completes_work_in_4_days
  (A_days : ℕ)
  (B_efficiency : ℕ → ℕ)
  (C_efficiency : ℕ → ℕ)
  (hA : A_days = 12)
  (hB : ∀ {x}, B_efficiency x = x * 3 / 2)
  (hC : ∀ {x}, C_efficiency x = x * 2) :
  (1 / (1 / (C_efficiency (B_efficiency A_days)))) = 4 := by
  sorry

end C_completes_work_in_4_days_l88_88584


namespace negation_correct_l88_88209

def original_statement (x : ℝ) : Prop := x > 0 → x^2 + 3 * x - 2 > 0

def negated_statement (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 2 ≤ 0

theorem negation_correct : (¬ ∀ x, original_statement x) ↔ ∃ x, negated_statement x := by
  sorry

end negation_correct_l88_88209


namespace sequence_inequality_l88_88729

theorem sequence_inequality (a : ℕ → ℝ) 
  (h₀ : a 0 = 5) 
  (h₁ : ∀ n, a (n + 1) * a n - a n ^ 2 = 1) : 
  35 < a 600 ∧ a 600 < 35.1 :=
sorry

end sequence_inequality_l88_88729


namespace total_seashells_l88_88018

-- Definitions of the initial number of seashells and the number found
def initial_seashells : Nat := 19
def found_seashells : Nat := 6

-- Theorem stating the total number of seashells in the collection
theorem total_seashells : initial_seashells + found_seashells = 25 := by
  sorry

end total_seashells_l88_88018


namespace lemonade_water_requirement_l88_88817

variables (W S L H : ℕ)

-- Definitions based on the conditions
def water_equation (W S : ℕ) := W = 5 * S
def sugar_equation (S L : ℕ) := S = 3 * L
def honey_equation (H L : ℕ) := H = L
def lemon_juice_amount (L : ℕ) := L = 2

-- Theorem statement for the proof problem
theorem lemonade_water_requirement :
  ∀ (W S L H : ℕ), 
  (water_equation W S) →
  (sugar_equation S L) →
  (honey_equation H L) →
  (lemon_juice_amount L) →
  W = 30 :=
by
  intros W S L H hW hS hH hL
  sorry

end lemonade_water_requirement_l88_88817


namespace range_of_a_squared_minus_2b_l88_88337

variable (a b : ℝ)

def quadratic_has_two_real_roots_in_01 (a b : ℝ) : Prop :=
  b ≥ 0 ∧ 1 + a + b ≥ 0 ∧ -2 ≤ a ∧ a ≤ 0 ∧ a^2 - 4 * b ≥ 0

theorem range_of_a_squared_minus_2b (a b : ℝ)
  (h : quadratic_has_two_real_roots_in_01 a b) : 0 ≤ a^2 - 2 * b ∧ a^2 - 2 * b ≤ 2 :=
sorry

end range_of_a_squared_minus_2b_l88_88337


namespace exists_equal_mod_p_l88_88082

theorem exists_equal_mod_p (p : ℕ) [hp_prime : Fact p.Prime] 
  (m : Fin p → ℕ) 
  (h_consecutive : ∀ i j : Fin p, (i : ℕ) < j → m i + 1 = m j) 
  (sigma : Equiv (Fin p) (Fin p)) :
  ∃ (k l : Fin p), k ≠ l ∧ (m k * m (sigma k) - m l * m (sigma l)) % p = 0 :=
by
  sorry

end exists_equal_mod_p_l88_88082


namespace factorize_expression_l88_88758

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l88_88758


namespace proof_problem_l88_88046

-- Define the problem:
def problem := ∀ (a : Fin 100 → ℝ), 
  (∀ i j, i ≠ j → a i ≠ a j) →  -- All numbers are distinct
  ∃ i : Fin 100, a i + a (⟨i.val + 3, sorry⟩) > a (⟨i.val + 1, sorry⟩) + a (⟨i.val + 2, sorry⟩)
-- Summarize: there exists four consecutive points on the circle such that 
-- the sum of the numbers at the ends is greater than the sum of the numbers in the middle.

theorem proof_problem : problem := sorry

end proof_problem_l88_88046


namespace seating_arrangement_l88_88903

theorem seating_arrangement (n x : ℕ) (h1 : 7 * x + 6 * (n - x) = 53) : x = 5 :=
sorry

end seating_arrangement_l88_88903


namespace smallest_whole_number_gt_total_sum_l88_88329

-- Declarations of the fractions involved
def term1 : ℚ := 3 + 1/3
def term2 : ℚ := 4 + 1/6
def term3 : ℚ := 5 + 1/12
def term4 : ℚ := 6 + 1/8

-- Definition of the entire sum
def total_sum : ℚ := term1 + term2 + term3 + term4

-- Statement of the theorem
theorem smallest_whole_number_gt_total_sum : 
  ∀ n : ℕ, (n > total_sum) → (∀ m : ℕ, (m >= 0) → (m > total_sum) → (n ≤ m)) → n = 19 := by
  sorry -- the proof is omitted

end smallest_whole_number_gt_total_sum_l88_88329


namespace steve_take_home_pay_l88_88402

-- Defining the conditions
def annual_salary : ℕ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℕ := 800

-- Defining the taxes function
def taxes (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the healthcare function
def healthcare (salary : ℕ) (rate : ℝ) : ℝ :=
  salary * rate

-- Defining the total deductions function
def total_deductions (salary : ℕ) (tax_rate : ℝ) (healthcare_rate : ℝ) (dues : ℕ) : ℝ :=
  (taxes salary tax_rate) + (healthcare salary healthcare_rate) + dues

-- Defining the take-home pay function
def take_home_pay (salary : ℕ) (deductions : ℝ) : ℝ :=
  salary - deductions

-- Using definitions to assert the take-home pay
theorem steve_take_home_pay : take_home_pay annual_salary (total_deductions annual_salary tax_rate healthcare_rate union_dues) = 27200 :=
by
  -- proof has been skipped
  sorry

end steve_take_home_pay_l88_88402


namespace find_x_l88_88331

theorem find_x (x : ℝ) : 
  (∀ (y : ℝ), 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := sorry

end find_x_l88_88331


namespace preimage_of_43_is_21_l88_88128

def f (x y : ℝ) : ℝ × ℝ := (x + 2 * y, 2 * x - y)

theorem preimage_of_43_is_21 : f 2 1 = (4, 3) :=
by {
  -- Proof omitted
  sorry
}

end preimage_of_43_is_21_l88_88128


namespace range_of_x_l88_88353

def interval1 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def interval2 : Set ℝ := {x | x < 1 ∨ x > 4}
def false_statement (x : ℝ) : Prop := x ∈ interval1 ∨ x ∈ interval2

theorem range_of_x (x : ℝ) (h : ¬ false_statement x) : x ∈ Set.Ico 1 2 :=
by
  sorry

end range_of_x_l88_88353


namespace new_train_distance_l88_88872

theorem new_train_distance (old_train_distance : ℕ) (additional_factor : ℕ) (h₀ : old_train_distance = 300) (h₁ : additional_factor = 50) :
  let new_train_distance := old_train_distance + (additional_factor * old_train_distance / 100)
  new_train_distance = 450 :=
by
  sorry

end new_train_distance_l88_88872


namespace minimum_value_of_f_l88_88485

noncomputable def f (x : ℝ) : ℝ := x + 1 / x + 1 / (x + 1 / x) + 1 / (x^2 + 1 / x^2)

theorem minimum_value_of_f :
  (∀ x > 0, f x ≥ 3) ∧ (f 1 = 3) :=
by
  sorry

end minimum_value_of_f_l88_88485


namespace decorations_cost_correct_l88_88386

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l88_88386


namespace tiffany_uploaded_7_pics_from_her_phone_l88_88221

theorem tiffany_uploaded_7_pics_from_her_phone
  (camera_pics : ℕ)
  (albums : ℕ)
  (pics_per_album : ℕ)
  (total_pics : ℕ)
  (h_camera_pics : camera_pics = 13)
  (h_albums : albums = 5)
  (h_pics_per_album : pics_per_album = 4)
  (h_total_pics : total_pics = albums * pics_per_album) :
  total_pics - camera_pics = 7 := by
  sorry

end tiffany_uploaded_7_pics_from_her_phone_l88_88221


namespace hyperbola_chord_line_eq_l88_88113

theorem hyperbola_chord_line_eq (m n s t : ℝ) (h_mn_pos : m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0)
  (h_mn_sum : m + n = 2)
  (h_m_n_s_t : m / s + n / t = 9)
  (h_s_t_min : s + t = 4 / 9)
  (h_midpoint : (2 : ℝ) = (m + n)) :
  ∃ (c : ℝ), (∀ (x1 y1 x2 y2 : ℝ), 
    (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧ 
    (x1 ^ 2 / 4 - y1 ^ 2 / 2 = 1 ∧ x2 ^ 2 / 4 - y2 ^ 2 / 2 = 1) → 
    y2 - y1 = c * (x2 - x1)) ∧ (c = 1 / 2) →
  ∀ (x y : ℝ), x - 2 * y + 1 = 0 :=
by sorry

end hyperbola_chord_line_eq_l88_88113


namespace two_dice_sum_greater_than_four_l88_88238
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l88_88238


namespace total_fish_count_l88_88712

theorem total_fish_count (num_fishbowls : ℕ) (fish_per_bowl : ℕ)
  (h1 : num_fishbowls = 261) (h2 : fish_per_bowl = 23) : 
  num_fishbowls * fish_per_bowl = 6003 := 
  by 
    sorry

end total_fish_count_l88_88712


namespace find_t_l88_88417

theorem find_t :
  ∃ (B : ℝ × ℝ) (t : ℝ), 
  B.1^2 + B.2^2 = 100 ∧ 
  B.1 - 2 * B.2 + 10 = 0 ∧ 
  B.1 > 0 ∧ B.2 > 0 ∧ 
  t = 20 ∧ 
  (∃ m : ℝ, 
    m = -2 ∧ 
    B.2 = m * B.1 + (8 + 2 * B.1 - m * B.1)) := 
by
  sorry

end find_t_l88_88417


namespace mary_shirt_fraction_l88_88822

theorem mary_shirt_fraction (f : ℝ) : 
  26 * (1 - f) + 36 - 36 / 3 = 37 → f = 1 / 2 :=
by
  sorry

end mary_shirt_fraction_l88_88822


namespace probability_sum_greater_than_four_is_5_over_6_l88_88250

-- Define the sample space for two dice.
def sample_space : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6) 

-- Define the event where the sum is greater than four.
def event_sum_greater_than_four : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p, p.1 + p.2 > 4)

-- Compute the probability of the event.
def probability_sum_greater_than_four : ℚ :=
  (event_sum_greater_than_four.card : ℚ) / (sample_space.card : ℚ)

-- Theorem to be proven
theorem probability_sum_greater_than_four_is_5_over_6 :
  probability_sum_greater_than_four = 5 / 6 :=
by
  -- Proof would go here
  sorry

end probability_sum_greater_than_four_is_5_over_6_l88_88250


namespace probability_correct_l88_88236

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l88_88236


namespace katie_roll_probability_l88_88010

def prob_less_than_five (d : ℕ) : ℚ :=
if d < 5 then 1 else 0

def prob_even (d : ℕ) : ℚ :=
if d % 2 = 0 then 1 else 0

theorem katie_roll_probability :
  (prob_less_than_five 1 + prob_less_than_five 2 + prob_less_than_five 3 + prob_less_than_five 4 +
  prob_less_than_five 5 + prob_less_than_five 6) / 6 *
  (prob_even 1 + prob_even 2 + prob_even 3 + prob_even 4 +
  prob_even 5 + prob_even 6) / 6 = 1 / 3 :=
sorry

end katie_roll_probability_l88_88010


namespace factorize_expression_l88_88757

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l88_88757


namespace parallel_lines_l88_88505

-- Definitions of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

-- Definition of parallel lines: slopes are equal and the lines are not identical
def slopes_equal (m : ℝ) : Prop := -(3 + m) / 4 = -2 / (5 + m)
def not_identical_lines (m : ℝ) : Prop := l1 m ≠ l2 m

-- Theorem stating the given conditions
theorem parallel_lines (m : ℝ) (x y : ℝ) : slopes_equal m → not_identical_lines m → m = -7 := by
  sorry

end parallel_lines_l88_88505


namespace sum_of_interior_angles_of_pentagon_l88_88705

theorem sum_of_interior_angles_of_pentagon :
  let n := 5 in (n - 2) * 180 = 540 := 
by 
  let n := 5
  show (n - 2) * 180 = 540
  sorry

end sum_of_interior_angles_of_pentagon_l88_88705


namespace boots_ratio_l88_88823

noncomputable def problem_statement : Prop :=
  let total_money : ℝ := 50
  let cost_toilet_paper : ℝ := 12
  let cost_groceries : ℝ := 2 * cost_toilet_paper
  let remaining_after_groceries : ℝ := total_money - cost_toilet_paper - cost_groceries
  let extra_money_per_person : ℝ := 35
  let total_extra_money : ℝ := 2 * extra_money_per_person
  let total_cost_boots : ℝ := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots : ℝ := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3

theorem boots_ratio (total_money : ℝ) (cost_toilet_paper : ℝ) (extra_money_per_person : ℝ) : 
  let cost_groceries := 2 * cost_toilet_paper
  let remaining_after_groceries := total_money - cost_toilet_paper - cost_groceries
  let total_extra_money := 2 * extra_money_per_person
  let total_cost_boots := remaining_after_groceries + total_extra_money
  let cost_per_pair_boots := total_cost_boots / 2
  let ratio := cost_per_pair_boots / remaining_after_groceries
  ratio = 3 :=
by
  sorry

end boots_ratio_l88_88823


namespace gcd_bc_minimum_l88_88164

theorem gcd_bc_minimum
  (a b c : ℕ)
  (h1 : Nat.gcd a b = 360)
  (h2 : Nat.gcd a c = 1170)
  (h3 : ∃ k1 : ℕ, b = 5 * k1)
  (h4 : ∃ k2 : ℕ, c = 13 * k2) : Nat.gcd b c = 90 :=
by
  sorry

end gcd_bc_minimum_l88_88164


namespace additional_laps_needed_l88_88820

-- Definitions of problem conditions
def total_required_distance : ℕ := 2400
def lap_length : ℕ := 150
def madison_laps : ℕ := 6
def gigi_laps : ℕ := 6

-- Target statement to prove the number of additional laps needed
theorem additional_laps_needed : (total_required_distance - (madison_laps + gigi_laps) * lap_length) / lap_length = 4 := by
  sorry

end additional_laps_needed_l88_88820


namespace original_price_of_petrol_l88_88287

variable (P : ℝ)

theorem original_price_of_petrol (h : 0.9 * (200 / P - 200 / (0.9 * P)) = 5) : 
  (P = 20 / 4.5) :=
sorry

end original_price_of_petrol_l88_88287


namespace oil_bill_for_January_l88_88576

variable {F J : ℕ}

theorem oil_bill_for_January (h1 : 2 * F = 3 * J) (h2 : 3 * (F + 20) = 5 * J) : J = 120 := by
  sorry

end oil_bill_for_January_l88_88576


namespace probability_sum_greater_than_four_l88_88255

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l88_88255


namespace great_wall_scientific_notation_l88_88217

theorem great_wall_scientific_notation :
  6700000 = 6.7 * 10^6 :=
sorry

end great_wall_scientific_notation_l88_88217


namespace conic_section_is_ellipse_l88_88056

open Real

def is_conic_section_ellipse (x y : ℝ) (k : ℝ) (p1 p2 : ℝ × ℝ) : Prop :=
  sqrt ((x - p1.1) ^ 2 + (y - p1.2) ^ 2) + sqrt ((x - p2.1) ^ 2 + (y - p2.2) ^ 2) = k

theorem conic_section_is_ellipse :
  is_conic_section_ellipse 2 (-2) 12 (2, -2) (-3, 5) :=
by
  sorry

end conic_section_is_ellipse_l88_88056


namespace max_value_l88_88122

noncomputable def satisfies_equation (x y : ℝ) : Prop :=
  x + 4 * y - x * y = 0

theorem max_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : satisfies_equation x y) :
  ∃ m, m = (4 / (x + y)) ∧ m ≤ (4 / 9) :=
by
  sorry

end max_value_l88_88122


namespace bruce_and_anne_clean_house_l88_88471

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l88_88471


namespace region_area_l88_88852

theorem region_area (x y : ℝ) : 
  (x^2 + y^2 + 14 * x + 18 * y = 0) → 
  (π * 130) = 130 * π :=
by 
  sorry

end region_area_l88_88852


namespace perfect_square_A_plus_2B_plus_4_l88_88014

theorem perfect_square_A_plus_2B_plus_4 (n : ℕ) (hn : 0 < n) :
  let A := (4 / 9 : ℚ) * (10 ^ (2 * n) - 1)
  let B := (8 / 9 : ℚ) * (10 ^ n - 1)
  ∃ k : ℚ, A + 2 * B + 4 = k^2 := 
by {
  sorry
}

end perfect_square_A_plus_2B_plus_4_l88_88014


namespace half_angle_second_quadrant_l88_88642

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (k * π + π / 4 < α / 2 ∧ α / 2 < k * π + π / 2) :=
sorry

end half_angle_second_quadrant_l88_88642


namespace remaining_homes_l88_88371

theorem remaining_homes (total_homes : ℕ) (first_hour_fraction : ℚ) (second_hour_fraction : ℚ) : 
  total_homes = 200 →
  first_hour_fraction = 2/5 →
  second_hour_fraction = 60/100 →
  let
    first_distributed := first_hour_fraction * total_homes,
    remaining_after_first := total_homes - first_distributed,
    second_distributed := second_hour_fraction * remaining_after_first,
    remaining_after_second := remaining_after_first - second_distributed
  in
  remaining_after_second = 48 := 
by
  intros h_total h_first_fraction h_second_fraction,
  let first_distributed := first_hour_fraction * total_homes,
  let remaining_after_first := total_homes - first_distributed,
  let second_distributed := second_hour_fraction * remaining_after_first,
  let remaining_after_second := remaining_after_first - second_distributed,
  sorry -- proof goes here

end remaining_homes_l88_88371


namespace probability_sum_greater_than_four_l88_88244

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88244


namespace value_of_a_l88_88515

noncomputable def coefficient_of_x2_term (a : ℝ) : ℝ :=
  a^4 * Nat.choose 8 4

theorem value_of_a (a : ℝ) (h : coefficient_of_x2_term a = 70) : a = 1 ∨ a = -1 := by
  sorry

end value_of_a_l88_88515


namespace coefficient_x3_expansion_sum_l88_88696

theorem coefficient_x3_expansion_sum :
  (∑ k in (finset.range 15).map (λ i, i + 1), (nat.choose (k + 3) 3)) = 1820 :=
  sorry

end coefficient_x3_expansion_sum_l88_88696


namespace total_thread_needed_l88_88186

def keychain_length : Nat := 12
def friends_in_classes : Nat := 10
def multiplier_for_club_friends : Nat := 2
def thread_per_class_friend : Nat := 16
def thread_per_club_friend : Nat := 20

theorem total_thread_needed :
  10 * thread_per_class_friend + (10 * multiplier_for_club_friends) * thread_per_club_friend = 560 := by
  sorry

end total_thread_needed_l88_88186


namespace average_decrease_l88_88191

theorem average_decrease (avg_6 : ℝ) (obs_7 : ℝ) (new_avg : ℝ) (decrease : ℝ) :
  avg_6 = 11 → obs_7 = 4 → (6 * avg_6 + obs_7) / 7 = new_avg → avg_6 - new_avg = decrease → decrease = 1 :=
  by
    intros h1 h2 h3 h4
    rw [h1, h2] at *
    sorry

end average_decrease_l88_88191


namespace yvette_final_bill_l88_88665

def cost_alicia : ℝ := 7.50
def cost_brant : ℝ := 10.00
def cost_josh : ℝ := 8.50
def cost_yvette : ℝ := 9.00
def tip_rate : ℝ := 0.20

def total_cost := cost_alicia + cost_brant + cost_josh + cost_yvette
def tip := tip_rate * total_cost
def final_bill := total_cost + tip

theorem yvette_final_bill :
  final_bill = 42.00 :=
  sorry

end yvette_final_bill_l88_88665


namespace smallest_number_of_students_l88_88299

-- Define the structure of the problem
def unique_row_configurations (n : ℕ) : Prop :=
  (∀ k : ℕ, k ∣ n → k < 10) → ∃ divs : Finset ℕ, divs.card = 9 ∧ ∀ d ∈ divs, d ∣ n ∧ (∀ d' ∈ divs, d ≠ d') 

-- The main statement to be proven in Lean 4
theorem smallest_number_of_students : ∃ n : ℕ, unique_row_configurations n ∧ n = 36 :=
by
  sorry

end smallest_number_of_students_l88_88299


namespace survey_total_parents_l88_88880

theorem survey_total_parents (P : ℝ)
  (h1 : 0.15 * P + 0.60 * P + 0.20 * 0.25 * P + 0.05 * P = P)
  (h2 : 0.05 * P = 6) : 
  P = 120 :=
sorry

end survey_total_parents_l88_88880


namespace pascal_triangle_ratio_l88_88321

theorem pascal_triangle_ratio (n r : ℕ) :
  (r + 1 = (4 * (n - r)) / 5) ∧ (r + 2 = (5 * (n - r - 1)) / 6) → n = 53 :=
by sorry

end pascal_triangle_ratio_l88_88321


namespace route_comparison_l88_88020

noncomputable def t_X : ℝ := (8 / 40) * 60 -- time in minutes for Route X
noncomputable def t_Y1 : ℝ := (5.5 / 50) * 60 -- time in minutes for the normal speed segment of Route Y
noncomputable def t_Y2 : ℝ := (1 / 25) * 60 -- time in minutes for the construction zone segment of Route Y
noncomputable def t_Y3 : ℝ := (0.5 / 20) * 60 -- time in minutes for the park zone segment of Route Y
noncomputable def t_Y : ℝ := t_Y1 + t_Y2 + t_Y3 -- total time in minutes for Route Y

theorem route_comparison : t_X - t_Y = 1.5 :=
by {
  -- Proof is skipped using sorry
  sorry
}

end route_comparison_l88_88020


namespace find_positive_number_l88_88069

-- The definition to state the given condition
def condition1 (n : ℝ) : Prop := n > 0 ∧ n^2 + n = 245

-- The theorem stating the problem and its solution
theorem find_positive_number (n : ℝ) (h : condition1 n) : n = 14 :=
by sorry

end find_positive_number_l88_88069


namespace triangle_inequality_l88_88372

variable (a b c : ℝ)
variable (h1 : a * b + b * c + c * a = 18)
variable (h2 : 1 < a)
variable (h3 : 1 < b)
variable (h4 : 1 < c)

theorem triangle_inequality :
  (1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3) > (1 / (a + b + c - 3)) :=
by
  sorry

end triangle_inequality_l88_88372


namespace average_of_new_sequence_l88_88985

variable (c : ℕ)  -- c is a positive integer
variable (d : ℕ)  -- d is the average of the sequence starting from c 

def average_of_sequence (seq : List ℕ) : ℕ :=
  if h : seq.length ≠ 0 then seq.sum / seq.length else 0

theorem average_of_new_sequence (h : d = average_of_sequence [c, c+1, c+2, c+3, c+4, c+5, c+6]) :
  average_of_sequence [d, d+1, d+2, d+3, d+4, d+5, d+6] = c + 6 := 
sorry

end average_of_new_sequence_l88_88985


namespace income_difference_l88_88900

theorem income_difference
  (D W : ℝ)
  (hD : 0.08 * D = 800)
  (hW : 0.08 * W = 840) :
  (W + 840) - (D + 800) = 540 := 
  sorry

end income_difference_l88_88900


namespace gcd_factorial_8_6_squared_l88_88101

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l88_88101


namespace number_of_segments_l88_88691

theorem number_of_segments (tangent_chords : ℕ) (angle_ABC : ℝ) (h : angle_ABC = 80) :
  tangent_chords = 18 :=
sorry

end number_of_segments_l88_88691


namespace probability_sum_greater_than_four_l88_88258

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88258


namespace measure_diagonal_without_pythagorean_theorem_l88_88390

variables (a b c : ℝ)

-- Definition of the function to measure the diagonal distance
def diagonal_method (a b c : ℝ) : ℝ :=
  -- by calculating the hypotenuse scaled by sqrt(3), we ignore using the Pythagorean theorem directly
  sorry

-- Calculate distance by arranging bricks
theorem measure_diagonal_without_pythagorean_theorem (distance_extreme_corners : ℝ) :
  distance_extreme_corners = (diagonal_method a b c) :=
  sorry

end measure_diagonal_without_pythagorean_theorem_l88_88390


namespace dining_bill_split_l88_88289

theorem dining_bill_split (original_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (total_bill_with_tip : ℝ) (amount_per_person : ℝ)
  (h1 : original_bill = 139.00)
  (h2 : num_people = 3)
  (h3 : tip_percent = 0.10)
  (h4 : total_bill_with_tip = original_bill + (tip_percent * original_bill))
  (h5 : amount_per_person = total_bill_with_tip / num_people) :
  amount_per_person = 50.97 :=
by 
  sorry

end dining_bill_split_l88_88289


namespace final_price_of_book_l88_88743

theorem final_price_of_book (original_price : ℝ) (d1_percentage : ℝ) (d2_percentage : ℝ) 
  (first_discount : ℝ) (second_discount : ℝ) (new_price1 : ℝ) (final_price : ℝ) :
  original_price = 15 ∧ d1_percentage = 0.20 ∧ d2_percentage = 0.25 ∧
  first_discount = d1_percentage * original_price ∧ new_price1 = original_price - first_discount ∧
  second_discount = d2_percentage * new_price1 ∧ 
  final_price = new_price1 - second_discount → final_price = 9 := 
by 
  sorry

end final_price_of_book_l88_88743


namespace scientific_notation_of_87000000_l88_88365

theorem scientific_notation_of_87000000 :
  87000000 = 8.7 * 10^7 := 
sorry

end scientific_notation_of_87000000_l88_88365


namespace expr1_val_expr2_val_l88_88078

noncomputable def expr1 : ℝ :=
  (1 / Real.sin (10 * Real.pi / 180)) - (Real.sqrt 3 / Real.cos (10 * Real.pi / 180))

theorem expr1_val : expr1 = 4 :=
  sorry

noncomputable def expr2 : ℝ :=
  (Real.sin (50 * Real.pi / 180) * (1 + Real.sqrt 3 * Real.tan (10 * Real.pi / 180)) - Real.cos (20 * Real.pi / 180)) /
  (Real.cos (80 * Real.pi / 180) * Real.sqrt (1 - Real.cos (20 * Real.pi / 180)))

theorem expr2_val : expr2 = Real.sqrt 2 :=
  sorry

end expr1_val_expr2_val_l88_88078


namespace verify_original_prices_l88_88421

noncomputable def original_price_of_sweater : ℝ := 43.11
noncomputable def original_price_of_shirt : ℝ := 35.68
noncomputable def original_price_of_pants : ℝ := 71.36

def price_of_shirt (sweater_price : ℝ) : ℝ := sweater_price - 7.43
def price_of_pants (shirt_price : ℝ) : ℝ := 2 * shirt_price
def discounted_sweater_price (sweater_price : ℝ) : ℝ := 0.85 * sweater_price
def total_cost (shirt_price pants_price discounted_sweater_price : ℝ) : ℝ := shirt_price + pants_price + discounted_sweater_price

theorem verify_original_prices 
  (total_cost_value : ℝ)
  (price_of_shirt_value : ℝ)
  (price_of_pants_value : ℝ)
  (discounted_sweater_price_value : ℝ) :
  total_cost_value = 143.67 ∧ 
  price_of_shirt_value = original_price_of_shirt ∧ 
  price_of_pants_value = original_price_of_pants ∧
  discounted_sweater_price_value = discounted_sweater_price original_price_of_sweater →
  total_cost (price_of_shirt original_price_of_sweater) 
             (price_of_pants (price_of_shirt original_price_of_sweater)) 
             (discounted_sweater_price original_price_of_sweater) = 143.67 :=
by
  intros
  sorry

end verify_original_prices_l88_88421


namespace compare_neg_fractions_l88_88317

theorem compare_neg_fractions : 
  (- (8:ℚ) / 21) > - (3 / 7) :=
by sorry

end compare_neg_fractions_l88_88317


namespace division_then_multiplication_l88_88266

theorem division_then_multiplication : (180 / 6) * 3 = 90 := 
by
  have step1 : 180 / 6 = 30 := sorry
  have step2 : 30 * 3 = 90 := sorry
  sorry

end division_then_multiplication_l88_88266


namespace fraction_representation_of_3_36_l88_88278

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l88_88278


namespace alcohol_water_ratio_l88_88851

theorem alcohol_water_ratio (V : ℝ) (hV_pos : V > 0) :
  let jar1_alcohol := (2 / 3) * V
  let jar1_water := (1 / 3) * V
  let jar2_alcohol := (3 / 2) * V
  let jar2_water := (1 / 2) * V
  let total_alcohol := jar1_alcohol + jar2_alcohol
  let total_water := jar1_water + jar2_water
  (total_alcohol / total_water) = (13 / 5) :=
by
  -- Placeholder for the proof
  sorry

end alcohol_water_ratio_l88_88851


namespace solve_for_x_l88_88553

theorem solve_for_x (x : ℝ) (h : x ≠ 2) : (7 * x) / (x - 2) - 5 / (x - 2) = 3 / (x - 2) → x = 8 / 7 :=
by
  sorry

end solve_for_x_l88_88553


namespace value_of_m_l88_88651

theorem value_of_m (m : ℝ) (h : m ≠ 0)
  (h_roots : ∀ x, m * x^2 + 8 * m * x + 60 = 0 ↔ x = -5 ∨ x = -3) :
  m = 4 :=
sorry

end value_of_m_l88_88651


namespace balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l88_88488

noncomputable def ways_with_ball_in_box_one : Nat := 369
noncomputable def ways_with_two_empty_boxes : Nat := 360
noncomputable def ways_with_three_empty_boxes : Nat := 140
noncomputable def ways_ball_A_not_less_than_B : Nat := 375

theorem balls_in_boxes_with_one_in_one 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_1 : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_1 = 1 → 
  ∃ ways, ways = ways_with_ball_in_box_one := 
sorry

theorem balls_in_boxes_with_two_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 2 → 
  ∃ ways, ways = ways_with_two_empty_boxes := 
sorry

theorem balls_in_boxes_with_three_empty 
  (n_balls : Nat) (n_boxes : Nat) 
  (empty_boxes : Nat) :
  n_balls = 4 → n_boxes = 5 → empty_boxes = 3 → 
  ∃ ways, ways = ways_with_three_empty_boxes := 
sorry

theorem balls_in_boxes_A_not_less_B 
  (n_balls : Nat) (n_boxes : Nat) 
  (ball_A : Nat) (ball_B : Nat) :
  n_balls = 4 → n_boxes = 5 → ball_A ≠ ball_B →
  ∃ ways, ways = ways_ball_A_not_less_than_B := 
sorry

end balls_in_boxes_with_one_in_one_balls_in_boxes_with_two_empty_balls_in_boxes_with_three_empty_balls_in_boxes_A_not_less_B_l88_88488


namespace probability_sum_greater_than_four_l88_88252

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l88_88252


namespace loss_per_metre_is_5_l88_88878

-- Definitions
def selling_price (total_meters : ℕ) : ℕ := 18000
def cost_price_per_metre : ℕ := 65
def total_meters : ℕ := 300

-- Loss per meter calculation
def loss_per_metre (selling_price : ℕ) (cost_price_per_metre : ℕ) (total_meters : ℕ) : ℕ :=
  ((cost_price_per_metre * total_meters) - selling_price) / total_meters

-- Theorem statement
theorem loss_per_metre_is_5 : loss_per_metre (selling_price total_meters) cost_price_per_metre total_meters = 5 :=
by
  sorry

end loss_per_metre_is_5_l88_88878


namespace total_items_purchased_l88_88799

/-- Proof that Ike and Mike buy a total of 9 items given the constraints. -/
theorem total_items_purchased
  (total_money : ℝ)
  (sandwich_cost : ℝ)
  (drink_cost : ℝ)
  (combo_factor : ℕ)
  (money_spent_on_sandwiches : ℝ)
  (number_of_sandwiches : ℕ)
  (number_of_drinks : ℕ)
  (num_free_sandwiches : ℕ) :
  total_money = 40 →
  sandwich_cost = 5 →
  drink_cost = 1.5 →
  combo_factor = 5 →
  number_of_sandwiches = 9 →
  number_of_drinks = 0 →
  money_spent_on_sandwiches = number_of_sandwiches * sandwich_cost →
  total_money = money_spent_on_sandwiches →
  num_free_sandwiches = number_of_sandwiches / combo_factor →
  number_of_sandwiches = number_of_sandwiches + num_free_sandwiches →
  number_of_sandwiches + number_of_drinks = 9 :=
by
  intros
  sorry

end total_items_purchased_l88_88799


namespace necessary_but_not_sufficient_condition_l88_88697

theorem necessary_but_not_sufficient_condition (a : ℝ)
    (h : -2 ≤ a ∧ a ≤ 2)
    (hq : ∃ x y : ℂ, x ≠ y ∧ (x ^ 2 + (a : ℂ) * x + 1 = 0) ∧ (y ^ 2 + (a : ℂ) * y + 1 = 0)) :
    ∃ z : ℂ, z ^ 2 + (a : ℂ) * z + 1 = 0 ∧ (¬ ∀ b, -2 < b ∧ b < 2 → b = a) :=
sorry

end necessary_but_not_sufficient_condition_l88_88697


namespace percentage_reduction_is_correct_l88_88508

def percentage_reduction_alcohol_concentration (V_original V_added : ℚ) (C_original : ℚ) : ℚ :=
  let V_total := V_original + V_added
  let Amount_alcohol := V_original * C_original
  let C_new := Amount_alcohol / V_total
  ((C_original - C_new) / C_original) * 100

theorem percentage_reduction_is_correct :
  percentage_reduction_alcohol_concentration 12 28 0.20 = 70 := by
  sorry

end percentage_reduction_is_correct_l88_88508


namespace part1_part2_l88_88115

-- Part (1)
theorem part1 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hx : x > 0) (hy : y < 0) : x + y = -4 :=
sorry

-- Part (2)
theorem part2 (x y : ℝ) (h1 : abs x = 3) (h2 : abs y = 7) (hxy : x < y) : x - y = -10 ∨ x - y = -4 :=
sorry

end part1_part2_l88_88115


namespace inverse_matrix_correct_l88_88917

def A : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![1, 2, 3],
    ![0, -1, 2],
    ![3, 0, 7]
  ]

def A_inv_correct : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![-1/2, -1, 1/2],
    ![3/7, -1/7, -1/7],
    ![3/14, 3/7, -1/14]
  ]

theorem inverse_matrix_correct : A⁻¹ = A_inv_correct := by
  sorry

end inverse_matrix_correct_l88_88917


namespace set_intersection_l88_88503

theorem set_intersection :
  let A := {x : ℝ | 0 < x}
  let B := {x : ℝ | -1 ≤ x ∧ x < 3}
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := 
by
  sorry

end set_intersection_l88_88503


namespace find_a_l88_88910

-- Define the condition that the quadratic can be expressed as the square of a binomial
variables (a r s : ℝ)

-- State the condition
def is_square_of_binomial (p q : ℝ) := (r * p + q) * (r * p + q)

-- The theorem to prove
theorem find_a (h : is_square_of_binomial x s = ax^2 + 20 * x + 9) : a = 100 / 9 := 
sorry

end find_a_l88_88910


namespace possible_third_side_of_triangle_l88_88520

theorem possible_third_side_of_triangle (a b : ℝ) (ha : a = 3) (hb : b = 6) (x : ℝ) :
  3 < x ∧ x < 9 → x = 6 :=
by
  intros h
  have h1 : 3 < x := h.left
  have h2 : x < 9 := h.right
  have h3 : a + b > x := by linarith
  have h4 : b - a < x := by linarith
  sorry

end possible_third_side_of_triangle_l88_88520


namespace age_relation_l88_88073

/--
Given that a woman is 42 years old and her daughter is 8 years old,
prove that in 9 years, the mother will be three times as old as her daughter.
-/
theorem age_relation (x : ℕ) (mother_age daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) 
  (h3 : 42 + x = 3 * (8 + x)) : 
  x = 9 :=
by
  sorry

end age_relation_l88_88073


namespace sign_of_a_l88_88355

theorem sign_of_a (a b c d : ℝ) (h : b * (3 * d + 2) ≠ 0) (ineq : a / b < -c / (3 * d + 2)) : 
  (a = 0 ∨ a > 0 ∨ a < 0) :=
sorry

end sign_of_a_l88_88355


namespace fraction_evaluation_l88_88763

theorem fraction_evaluation :
  (1 / 2 + 1 / 3) / (3 / 7 - 1 / 5) = 175 / 48 :=
by
  sorry

end fraction_evaluation_l88_88763


namespace scholarship_awards_l88_88970

theorem scholarship_awards (x : ℕ) (h : 10000 * x + 2000 * (28 - x) = 80000) : x = 3 ∧ (28 - x) = 25 :=
by {
  sorry
}

end scholarship_awards_l88_88970


namespace combined_ratio_l88_88051

theorem combined_ratio (cayley_students fermat_students : ℕ) 
                       (cayley_ratio_boys cayley_ratio_girls fermat_ratio_boys fermat_ratio_girls : ℕ) 
                       (h_cayley : cayley_students = 400) 
                       (h_cayley_ratio : (cayley_ratio_boys, cayley_ratio_girls) = (3, 2)) 
                       (h_fermat : fermat_students = 600) 
                       (h_fermat_ratio : (fermat_ratio_boys, fermat_ratio_girls) = (2, 3)) :
  (480 : ℚ) / 520 = 12 / 13 := 
by 
  sorry

end combined_ratio_l88_88051


namespace hypotenuse_of_right_triangle_l88_88430

theorem hypotenuse_of_right_triangle (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ c^2 = a^2 + b^2 :=
by
  sorry

end hypotenuse_of_right_triangle_l88_88430


namespace simplify_fraction_l88_88283

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l88_88283


namespace solution_set_f_ge_1_l88_88172

noncomputable def f (x : ℝ) (a : ℝ) :=
  if x >= 0 then |x - 2| + a else -(|-x - 2| + a)

theorem solution_set_f_ge_1 {a : ℝ} (ha : a = -2) :
  {x : ℝ | f x a ≥ 1} = {x : ℝ | x ≤ -1 ∨ x ≥ 5} :=
by sorry

end solution_set_f_ge_1_l88_88172


namespace person_birth_year_and_age_l88_88573

theorem person_birth_year_and_age (x y: ℕ) (h1: x ≤ 9) (h2: y ≤ 9) (hy: y = (88 - 10 * x) / (x + 1)):
  1988 - (1900 + 10 * x + y) = x * y → 1900 + 10 * x + y = 1964 ∧ 1988 - (1900 + 10 * x + y) = 24 :=
by
  sorry

end person_birth_year_and_age_l88_88573


namespace mixed_oil_rate_l88_88000

theorem mixed_oil_rate :
  let v₁ := 10
  let p₁ := 50
  let v₂ := 5
  let p₂ := 68
  let v₃ := 8
  let p₃ := 42
  let v₄ := 7
  let p₄ := 62
  let v₅ := 12
  let p₅ := 55
  let v₆ := 6
  let p₆ := 75
  let total_cost := v₁ * p₁ + v₂ * p₂ + v₃ * p₃ + v₄ * p₄ + v₅ * p₅ + v₆ * p₆
  let total_volume := v₁ + v₂ + v₃ + v₄ + v₅ + v₆
  let rate := total_cost / total_volume
  rate = 56.67 :=
by
  sorry

end mixed_oil_rate_l88_88000


namespace total_games_played_l88_88184

-- Define the conditions as Lean 4 definitions
def games_won : Nat := 12
def games_lost : Nat := 4

-- Prove the total number of games played is 16
theorem total_games_played : games_won + games_lost = 16 := 
by
  -- Place a proof placeholder
  sorry

end total_games_played_l88_88184


namespace inscribed_circle_radius_l88_88149

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l88_88149


namespace simplify_expr_l88_88552

theorem simplify_expr (x : ℝ) : 1 - (1 - (1 - (1 + (1 - (1 - x))))) = 2 - x :=
by
  sorry

end simplify_expr_l88_88552


namespace probability_correct_l88_88233

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l88_88233


namespace Total_marbles_equal_231_l88_88612

def Connie_marbles : Nat := 39
def Juan_marbles : Nat := Connie_marbles + 25
def Maria_marbles : Nat := 2 * Juan_marbles
def Total_marbles : Nat := Connie_marbles + Juan_marbles + Maria_marbles

theorem Total_marbles_equal_231 : Total_marbles = 231 := sorry

end Total_marbles_equal_231_l88_88612


namespace treble_of_doubled_and_increased_l88_88068

theorem treble_of_doubled_and_increased (initial_number : ℕ) (result : ℕ) : 
  initial_number = 15 → (initial_number * 2 + 5) * 3 = result → result = 105 := 
by 
  intros h1 h2
  rw [h1] at h2
  linarith

end treble_of_doubled_and_increased_l88_88068


namespace find_m_value_l88_88067

theorem find_m_value (m : Real) (h : (3 * m + 8) * (m - 3) = 72) : m = (1 + Real.sqrt 1153) / 6 :=
by
  sorry

end find_m_value_l88_88067


namespace fractional_part_sum_leq_l88_88646

noncomputable def fractional_part (z : ℝ) : ℝ :=
  z - (⌊z⌋ : ℝ)

theorem fractional_part_sum_leq (x y : ℝ) :
  fractional_part (x + y) ≤ fractional_part x + fractional_part y :=
by
  sorry

end fractional_part_sum_leq_l88_88646


namespace necessary_but_not_sufficient_l88_88923

-- Define conditions P and Q
def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

-- Statement to prove
theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x ∧ ¬ (Q x → P x) :=
by {
  sorry
}

end necessary_but_not_sufficient_l88_88923


namespace find_a_l88_88911

-- Define the condition that the quadratic can be expressed as the square of a binomial
variables (a r s : ℝ)

-- State the condition
def is_square_of_binomial (p q : ℝ) := (r * p + q) * (r * p + q)

-- The theorem to prove
theorem find_a (h : is_square_of_binomial x s = ax^2 + 20 * x + 9) : a = 100 / 9 := 
sorry

end find_a_l88_88911


namespace rectangular_field_area_l88_88177

-- Given a rectangle with one side 4 meters and diagonal 5 meters, prove that its area is 12 square meters.
theorem rectangular_field_area
  (w l d : ℝ)
  (h_w : w = 4)
  (h_d : d = 5)
  (h_pythagoras : w^2 + l^2 = d^2) :
  w * l = 12 := 
by
  sorry

end rectangular_field_area_l88_88177


namespace point_a_coordinates_l88_88805

open Set

theorem point_a_coordinates (A B : ℝ × ℝ) :
  B = (2, 4) →
  (A.1 = B.1 + 3 ∨ A.1 = B.1 - 3) ∧ A.2 = B.2 →
  dist A B = 3 →
  A = (5, 4) ∨ A = (-1, 4) :=
by
  intros hB hA hDist
  sorry

end point_a_coordinates_l88_88805


namespace inequality_proof_l88_88389

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
by 
  sorry

end inequality_proof_l88_88389


namespace decorations_cost_l88_88381

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l88_88381


namespace decorations_cost_correct_l88_88384

def cost_of_roses_per_centerpiece := 5 * 10
def cost_of_lilies_per_centerpiece := 4 * 15
def cost_of_place_settings_per_table := 4 * 10
def cost_of_tablecloth_per_table := 25
def cost_per_table := cost_of_roses_per_centerpiece + cost_of_lilies_per_centerpiece + cost_of_place_settings_per_table + cost_of_tablecloth_per_table
def number_of_tables := 20
def total_cost_of_decorations := cost_per_table * number_of_tables

theorem decorations_cost_correct :
  total_cost_of_decorations = 3500 := by
  sorry

end decorations_cost_correct_l88_88384


namespace initial_population_l88_88860

theorem initial_population (P : ℝ)
  (h1 : P * 1.25 * 0.75 = 18750) : P = 20000 :=
sorry

end initial_population_l88_88860


namespace new_trailer_homes_count_l88_88845

theorem new_trailer_homes_count :
  let old_trailers : ℕ := 30
  let old_avg_age : ℕ := 15
  let years_since : ℕ := 3
  let new_avg_age : ℕ := 10
  let total_age := (old_trailers * (old_avg_age + years_since)) + (3 * new_trailers)
  let total_trailers := old_trailers + new_trailers
  let total_avg_age := total_age / total_trailers
  total_avg_age = new_avg_age → new_trailers = 34 :=
by
  sorry

end new_trailer_homes_count_l88_88845


namespace inscribed_circle_radius_l88_88151

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) :
    ∃ x : ℝ, (∀ P Px OP O1P : ℝ, Px = sqrt((R - x) ^ 2 - x ^ 2) ∧ O1P = sqrt((r + x) ^ 2 - x ^ 2)
                 ∧ Px + r = O1P) ∧ x = 8 :=
begin
  sorry
end

end inscribed_circle_radius_l88_88151


namespace decorations_cost_l88_88382

def tablecloth_cost : ℕ := 20 * 25
def place_setting_cost : ℕ := 20 * 4 * 10
def rose_cost : ℕ := 20 * 10 * 5
def lily_cost : ℕ := 20 * 15 * 4

theorem decorations_cost :
  tablecloth_cost + place_setting_cost + rose_cost + lily_cost = 3500 :=
by sorry

end decorations_cost_l88_88382


namespace exists_integers_for_prime_l88_88548

theorem exists_integers_for_prime (p : ℕ) (hp : Nat.Prime p) : 
  ∃ x y z w : ℤ, x^2 + y^2 + z^2 = w * p ∧ 0 < w ∧ w < p :=
by 
  sorry

end exists_integers_for_prime_l88_88548


namespace problem_statement_negation_statement_l88_88637

variable {a b : ℝ}

theorem problem_statement (h : a * b ≤ 0) : a ≤ 0 ∨ b ≤ 0 :=
sorry

theorem negation_statement (h : a * b > 0) : a > 0 ∧ b > 0 :=
sorry

end problem_statement_negation_statement_l88_88637


namespace part_a_part_b_part_c_l88_88671

-- Define the conditions
inductive Color
| blue
| red
| green
| yellow

-- Each square can be painted in one of the colors: blue, red, or green.
def square_colors : List Color := [Color.blue, Color.red, Color.green]

-- Each triangle can be painted in one of the colors: blue, red, or yellow.
def triangle_colors : List Color := [Color.blue, Color.red, Color.yellow]

-- Condition that polygons with a common side cannot share the same color
def different_color (c1 c2 : Color) : Prop := c1 ≠ c2

-- Part (a)
theorem part_a : ∃ n : Nat, n = 7 := sorry

-- Part (b)
theorem part_b : ∃ n : Nat, n = 43 := sorry

-- Part (c)
theorem part_c : ∃ n : Nat, n = 667 := sorry

end part_a_part_b_part_c_l88_88671


namespace rabbit_catch_up_time_l88_88848

theorem rabbit_catch_up_time :
  let rabbit_speed := 25 -- miles per hour
  let cat_speed := 20 -- miles per hour
  let head_start := 15 / 60 -- hours, which is 0.25 hours
  let initial_distance := cat_speed * head_start
  let relative_speed := rabbit_speed - cat_speed
  initial_distance / relative_speed = 1 := by
  sorry

end rabbit_catch_up_time_l88_88848


namespace product_of_constants_t_l88_88106

theorem product_of_constants_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (ts : Finset ℤ), (ts = {11, 4, 1, -1, -4, -11}) ∧ ts.prod (λ x, x) = -1936 :=
by sorry

end product_of_constants_t_l88_88106


namespace number_of_customers_trimmed_l88_88795

-- Definitions based on the conditions
def total_sounds : ℕ := 60
def sounds_per_person : ℕ := 20

-- Statement to prove
theorem number_of_customers_trimmed :
  ∃ n : ℕ, n * sounds_per_person = total_sounds ∧ n = 3 :=
sorry

end number_of_customers_trimmed_l88_88795


namespace sum_of_squares_of_roots_l88_88015

theorem sum_of_squares_of_roots : 
  (∃ (a b c d : ℝ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^4 - 15 * x^2 + 56 = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d)) ∧
    (a^2 + b^2 + c^2 + d^2 = 30)) :=
sorry

end sum_of_squares_of_roots_l88_88015


namespace sum_of_interior_angles_of_remaining_polygon_l88_88075

theorem sum_of_interior_angles_of_remaining_polygon (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 5) :
  (n - 2) * 180 ≠ 270 :=
by 
  sorry

end sum_of_interior_angles_of_remaining_polygon_l88_88075


namespace population_growth_l88_88563

theorem population_growth (P_present P_future : ℝ) (r : ℝ) (n : ℕ)
  (h1 : P_present = 7800)
  (h2 : P_future = 10860.72)
  (h3 : n = 2) :
  P_future = P_present * (1 + r / 100)^n → r = 18.03 :=
by sorry

end population_growth_l88_88563


namespace smallest_angle_opposite_smallest_side_l88_88802

theorem smallest_angle_opposite_smallest_side 
  (a b c : ℝ) 
  (h_triangle : triangle_inequality_proof)
  (h_condition : 3 * a = b + c) :
  smallest_angle_proof :=
sorry

end smallest_angle_opposite_smallest_side_l88_88802


namespace solve_quadratic_eq_l88_88690

theorem solve_quadratic_eq (x : ℝ) : x^2 - x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end solve_quadratic_eq_l88_88690


namespace f_neg1_plus_f_2_l88_88133

def f (x : Int) : Int :=
  if x = -3 then -1
  else if x = -2 then -5
  else if x = -1 then -2
  else if x = 0 then 0
  else if x = 1 then 2
  else if x = 2 then 1
  else if x = 3 then 4
  else 0  -- This handles x values not explicitly in the table, although technically unnecessary.

theorem f_neg1_plus_f_2 : f (-1) + f (2) = -1 := by
  sorry

end f_neg1_plus_f_2_l88_88133


namespace binary_to_decimal_101101_l88_88898

theorem binary_to_decimal_101101 : 
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 :=
by
  let bit0 := 0
  let bit1 := 1
  let binary_num := [bit1, bit0, bit1, bit1, bit0, bit1]
  have h : (bit1 * 2^0 + bit0 * 2^1 + bit1 * 2^2 + bit1 * 2^3 + bit0 * 2^4 + bit1 * 2^5) = 45 := sorry
  exact h

end binary_to_decimal_101101_l88_88898


namespace anthony_has_more_pairs_l88_88980

theorem anthony_has_more_pairs (scott_pairs : ℕ) (anthony_pairs : ℕ) (jim_pairs : ℕ) :
  (scott_pairs = 7) →
  (anthony_pairs = 3 * scott_pairs) →
  (jim_pairs = anthony_pairs - 2) →
  (anthony_pairs - jim_pairs = 2) :=
by
  intro h_scott h_anthony h_jim
  sorry

end anthony_has_more_pairs_l88_88980


namespace geometric_sequence_term_l88_88843

theorem geometric_sequence_term
  (r a : ℝ)
  (h1 : 180 * r = a)
  (h2 : a * r = 81 / 32)
  (h3 : a > 0) :
  a = 135 / 19 :=
by sorry

end geometric_sequence_term_l88_88843


namespace volume_increase_factor_l88_88945

-- Defining the initial volume of the cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- Defining the modified height and radius
def new_height (h : ℝ) : ℝ := 3 * h
def new_radius (r : ℝ) : ℝ := 2.5 * r

-- Calculating the new volume with the modified dimensions
def new_volume (r h : ℝ) : ℝ := volume (new_radius r) (new_height h)

-- Proof statement to verify the volume factor
theorem volume_increase_factor (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  new_volume r h = 18.75 * volume r h :=
by
  sorry

end volume_increase_factor_l88_88945


namespace apples_final_count_l88_88394

theorem apples_final_count :
  let initial_apples := 200
  let shared_apples := 5
  let remaining_after_share := initial_apples - shared_apples
  let sister_takes := remaining_after_share / 2
  let half_rounded_down := 97 -- explicitly rounding down since 195 cannot be split exactly
  let remaining_after_sister := remaining_after_share - half_rounded_down
  let received_gift := 7
  let final_count := remaining_after_sister + received_gift
  final_count = 105 :=
by
  sorry

end apples_final_count_l88_88394


namespace divides_5n_4n_iff_n_is_multiple_of_3_l88_88920

theorem divides_5n_4n_iff_n_is_multiple_of_3 (n : ℕ) (h : n > 0) : 
  61 ∣ (5^n - 4^n) ↔ ∃ k : ℕ, n = 3 * k :=
by
  sorry

end divides_5n_4n_iff_n_is_multiple_of_3_l88_88920


namespace variance_transformation_l88_88408

theorem variance_transformation (a1 a2 a3 : ℝ) 
  (h1 : (a1 + a2 + a3) / 3 = 4) 
  (h2 : ((a1 - 4)^2 + (a2 - 4)^2 + (a3 - 4)^2) / 3 = 3) : 
  ((3 * a1 - 2 - (3 * 4 - 2))^2 + (3 * a2 - 2 - (3 * 4 - 2))^2 + (3 * a3 - 2 - (3 * 4 - 2))^2) / 3 = 27 := 
sorry

end variance_transformation_l88_88408


namespace iPhones_sold_l88_88887

theorem iPhones_sold (x : ℕ) (h1 : (1000 * x + 18000 + 16000) / (x + 100) = 670) : x = 100 :=
by
  sorry

end iPhones_sold_l88_88887


namespace johns_pool_depth_l88_88961

theorem johns_pool_depth : 
  ∀ (j s : ℕ), (j = 2 * s + 5) → (s = 5) → (j = 15) := 
by 
  intros j s h1 h2
  rw [h2] at h1
  exact h1

end johns_pool_depth_l88_88961


namespace bruce_anne_cleaning_house_l88_88469

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l88_88469


namespace solve_inequality_l88_88292

variable (x : ℝ)

theorem solve_inequality : 3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2) → x ≥ 4 / 5 :=
by
  sorry

end solve_inequality_l88_88292


namespace least_n_for_multiple_of_8_l88_88288

def is_positive_integer (n : ℕ) : Prop := n > 0

def is_multiple_of_8 (k : ℕ) : Prop := ∃ m : ℕ, k = 8 * m

theorem least_n_for_multiple_of_8 :
  ∀ n : ℕ, (is_positive_integer n → is_multiple_of_8 (Nat.factorial n)) → n ≥ 6 :=
by
  sorry

end least_n_for_multiple_of_8_l88_88288


namespace common_ratio_of_geometric_sequence_l88_88530

variable (a : ℕ → ℝ) -- The geometric sequence {a_n}
variable (q : ℝ)     -- The common ratio

-- Conditions
axiom h1 : a 2 = 18
axiom h2 : a 4 = 8

theorem common_ratio_of_geometric_sequence :
  (∀ n : ℕ, a (n + 1) = a n * q) ∧ q^2 = 4/9 → q = 2/3 ∨ q = -2/3 := by
  sorry

end common_ratio_of_geometric_sequence_l88_88530


namespace division_addition_l88_88581

theorem division_addition (n : ℕ) (h : 32 - 16 = n * 4) : n / 4 + 16 = 17 :=
by 
  sorry

end division_addition_l88_88581


namespace betty_total_cost_l88_88605

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end betty_total_cost_l88_88605


namespace arithmetic_seq_proof_l88_88633

open Nat

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (n : ℕ) : ℤ :=
n * (a 1) + n * (n - 1) / 2 * d

theorem arithmetic_seq_proof (a : ℕ → ℤ) (d : ℤ)
  (h1 : arithmetic_seq a d)
  (h2 : a 2 = 0)
  (h3 : sum_of_arithmetic_seq a d 3 + sum_of_arithmetic_seq a d 4 = 6) :
  a 5 + a 6 = 21 :=
sorry

end arithmetic_seq_proof_l88_88633


namespace AM_GM_inequality_example_l88_88767

open Real

theorem AM_GM_inequality_example 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ((a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2)) ≥ 9 * (a^2 * b^2 * c^2) :=
sorry

end AM_GM_inequality_example_l88_88767


namespace factorization_example_l88_88600

open Function

theorem factorization_example (a b : ℤ) :
  (a - 1) * (b - 1) = ab - a - b + 1 :=
by
  sorry

end factorization_example_l88_88600


namespace directrix_parabola_l88_88839

theorem directrix_parabola (x y : ℝ) :
  (x^2 = (1/4 : ℝ) * y) → (y = -1/16) :=
sorry

end directrix_parabola_l88_88839


namespace simplify_336_to_fraction_l88_88273

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l88_88273


namespace example_calculation_l88_88456

theorem example_calculation (a : ℝ) : (a^2)^3 = a^6 :=
by 
  sorry

end example_calculation_l88_88456


namespace tax_on_other_items_l88_88023

theorem tax_on_other_items (total_amount clothing_amount food_amount other_items_amount tax_on_clothing tax_on_food total_tax : ℝ) (tax_percent_other : ℝ) 
(h1 : clothing_amount = 0.5 * total_amount)
(h2 : food_amount = 0.2 * total_amount)
(h3 : other_items_amount = 0.3 * total_amount)
(h4 : tax_on_clothing = 0.04 * clothing_amount)
(h5 : tax_on_food = 0) 
(h6 : total_tax = 0.044 * total_amount)
: 
(tax_percent_other = 8) := 
by
  -- Definitions from the problem
  -- Define the total tax paid as the sum of taxes on clothing, food, and other items
  let tax_other_items : ℝ := tax_percent_other / 100 * other_items_amount
  
  -- Total tax equation
  have h7 : tax_on_clothing + tax_on_food + tax_other_items = total_tax
  sorry

  -- Substitution values into the given conditions and solving
  have h8 : tax_on_clothing + tax_percent_other / 100 * other_items_amount = total_tax
  sorry
  
  have h9 : 0.04 * 0.5 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h10 : 0.02 * total_amount + tax_percent_other / 100 * 0.3 * total_amount = 0.044 * total_amount
  sorry

  have h11 : tax_percent_other / 100 * 0.3 * total_amount = 0.024 * total_amount
  sorry

  have h12 : tax_percent_other / 100 * 0.3 = 0.024
  sorry

  have h13 : tax_percent_other / 100 = 0.08
  sorry

  have h14 : tax_percent_other = 8
  sorry

  exact h14

end tax_on_other_items_l88_88023


namespace smallest_a_plus_b_l88_88776

theorem smallest_a_plus_b : ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ 2^3 * 3^7 * 7^2 = a^b ∧ a + b = 380 :=
sorry

end smallest_a_plus_b_l88_88776


namespace modified_cube_cubies_l88_88444

structure RubiksCube :=
  (original_cubies : ℕ := 27)
  (removed_corners : ℕ := 8)
  (total_layers : ℕ := 3)
  (edges_per_layer : ℕ := 4)
  (faces_center_cubies : ℕ := 6)
  (center_cubie : ℕ := 1)

noncomputable def cubies_with_n_faces (n : ℕ) : ℕ :=
  if n = 4 then 12
  else if n = 1 then 6
  else if n = 0 then 1
  else 0

theorem modified_cube_cubies :
  (cubies_with_n_faces 4 = 12) ∧ (cubies_with_n_faces 1 = 6) ∧ (cubies_with_n_faces 0 = 1) := by
  sorry

end modified_cube_cubies_l88_88444


namespace amanda_days_needed_to_meet_goal_l88_88886

def total_tickets : ℕ := 80
def first_day_friends : ℕ := 5
def first_day_per_friend : ℕ := 4
def first_day_tickets : ℕ := first_day_friends * first_day_per_friend
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_days_needed_to_meet_goal : 
  first_day_tickets + second_day_tickets + third_day_tickets = total_tickets → 
  3 = 3 :=
by
  intro h
  sorry

end amanda_days_needed_to_meet_goal_l88_88886


namespace find_a_l88_88350

open Set

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | -1 ≤ x ∧ x ≤ a}

theorem find_a (a : ℝ) (h : (A ∪ B a) ⊆ (A ∩ B a)) : a = 1 :=
sorry

end find_a_l88_88350


namespace unique_solution_iff_a_values_l88_88899

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 5 * a

theorem unique_solution_iff_a_values (a : ℝ) :
  (∃! x : ℝ, |f x a| ≤ 3) ↔ (a = 3 / 4 ∨ a = -3 / 4) :=
by
  sorry

end unique_solution_iff_a_values_l88_88899


namespace total_pairs_is_11_l88_88438

-- Definitions for the conditions
def soft_lens_price : ℕ := 150
def hard_lens_price : ℕ := 85
def total_sales_last_week : ℕ := 1455

-- Variables
variables (H S : ℕ)

-- Condition that she sold 5 more pairs of soft lenses than hard lenses
def sold_more_soft : Prop := S = H + 5

-- Equation for total sales
def total_sales_eq : Prop := (hard_lens_price * H) + (soft_lens_price * S) = total_sales_last_week

-- Total number of pairs of contact lenses sold
def total_pairs_sold : ℕ := H + S

-- The theorem to prove
theorem total_pairs_is_11 (H S : ℕ) (h1 : sold_more_soft H S) (h2 : total_sales_eq H S) : total_pairs_sold H S = 11 :=
sorry

end total_pairs_is_11_l88_88438


namespace parallelogram_height_l88_88916

theorem parallelogram_height (A B H : ℝ) 
    (h₁ : A = 96) 
    (h₂ : B = 12) 
    (h₃ : A = B * H) :
  H = 8 := 
by {
  sorry
}

end parallelogram_height_l88_88916


namespace smallest_s_plus_d_l88_88135

theorem smallest_s_plus_d (s d : ℕ) (h_pos_s : s > 0) (h_pos_d : d > 0)
  (h_eq : 1 / s + 1 / (2 * s) + 1 / (3 * s) = 1 / (d^2 - 2 * d)) :
  s + d = 50 :=
sorry

end smallest_s_plus_d_l88_88135


namespace total_amount_spent_l88_88109

def speakers : ℝ := 118.54
def new_tires : ℝ := 106.33
def window_tints : ℝ := 85.27
def seat_covers : ℝ := 79.99
def scheduled_maintenance : ℝ := 199.75
def steering_wheel_cover : ℝ := 15.63
def air_fresheners_set : ℝ := 12.96
def car_wash : ℝ := 25.0

theorem total_amount_spent :
  speakers + new_tires + window_tints + seat_covers + scheduled_maintenance + steering_wheel_cover + air_fresheners_set + car_wash = 643.47 :=
by
  sorry

end total_amount_spent_l88_88109


namespace distance_between_trees_l88_88143

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (yard_length_eq : yard_length = 325) (num_trees_eq : num_trees = 26) :
  (yard_length / (num_trees - 1)) = 13 := by
  sorry

end distance_between_trees_l88_88143


namespace a_2_pow_100_value_l88_88167

theorem a_2_pow_100_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (2 * n) = 3 * n * a n) :
  a (2^100) = 2^4852 * 3^4950 :=
by
  sorry

end a_2_pow_100_value_l88_88167


namespace arithmetic_expression_evaluation_l88_88062

theorem arithmetic_expression_evaluation :
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := 
by
  -- Skipping the proof.
  sorry

end arithmetic_expression_evaluation_l88_88062


namespace dollar_op_5_neg2_l88_88085

def dollar_op (x y : Int) : Int := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_5_neg2 :
  dollar_op 5 (-2) = -45 := by
  sorry

end dollar_op_5_neg2_l88_88085


namespace team_b_wins_probability_l88_88213

theorem team_b_wins_probability :
  let p_A := 2 / 3 -- probability of Team A winning a set
  let p_B := 1 / 3 -- probability of Team B winning a set
  let match_probability := 
    p_B + p_A * p_B + (p_A)^2 * p_B -- compute the probability of Team B winning the match
  in match_probability = 19 / 27 :=
by
  -- conditions
  let p_A := (2 : ℚ) / 3
  let p_B := (1 : ℚ) / 3
  -- set calculation
  let match_probability := p_B + p_A * p_B + (p_A)^2 * p_B
  -- claim
  exact Eq.refl _
  -- the actual proof goes here
  sorry

end team_b_wins_probability_l88_88213


namespace james_choices_count_l88_88668

-- Define the conditions as Lean definitions
def isAscending (a b c d e : ℕ) : Prop := a < b ∧ b < c ∧ c < d ∧ d < e

def inRange (a b c d e : ℕ) : Prop := a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8 ∧ d ≤ 8 ∧ e ≤ 8

def meanEqualsMedian (a b c d e : ℕ) : Prop :=
  (a + b + c + d + e) / 5 = c

-- Define the problem statement
theorem james_choices_count :
  ∃ (s : Finset (ℕ × ℕ × ℕ × ℕ × ℕ)), 
    (∀ (a b c d e : ℕ), (a, b, c, d, e) ∈ s ↔ isAscending a b c d e ∧ inRange a b c d e ∧ meanEqualsMedian a b c d e) ∧
    s.card = 10 :=
sorry

end james_choices_count_l88_88668


namespace sum_of_arithmetic_progression_l88_88307

theorem sum_of_arithmetic_progression :
  let a := 30
  let d := -3
  let n := 20
  let S_n := n / 2 * (2 * a + (n - 1) * d)
  S_n = 30 :=
by
  sorry

end sum_of_arithmetic_progression_l88_88307


namespace find_whole_number_M_l88_88596

-- Define the conditions
def condition (M : ℕ) : Prop :=
  21 < M ∧ M < 23

-- Define the main theorem to be proven
theorem find_whole_number_M (M : ℕ) (h : condition M) : M = 22 := by
  sorry

end find_whole_number_M_l88_88596


namespace base3_last_two_digits_l88_88265

open Nat

theorem base3_last_two_digits (a b c : ℕ) (h1 : a = 2005) (h2 : b = 2003) (h3 : c = 2004) :
  (2005 ^ (2003 ^ 2004 + 3) % 81) = 11 :=
by
  sorry

end base3_last_two_digits_l88_88265


namespace inequality_holds_for_unit_interval_l88_88687

theorem inequality_holds_for_unit_interval (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
    5 * (x ^ 2 + y ^ 2) ^ 2 ≤ 4 + (x + y) ^ 4 :=
by
    sorry

end inequality_holds_for_unit_interval_l88_88687


namespace gcd_fact8_fact6_squared_l88_88104

-- Definition of 8! and (6!)²
def fact8 : ℕ := 8!
def fact6_squared : ℕ := (6!)^2

-- The theorem statement to be proved
theorem gcd_fact8_fact6_squared : Nat.gcd fact8 fact6_squared = 11520 := 
by
    sorry

end gcd_fact8_fact6_squared_l88_88104


namespace marbles_remainder_l88_88770

theorem marbles_remainder 
  (g r p : ℕ) 
  (hg : g % 8 = 5) 
  (hr : r % 7 = 2) 
  (hp : p % 7 = 4) : 
  (r + p + g) % 7 = 4 := 
sorry

end marbles_remainder_l88_88770


namespace find_d_l88_88992

theorem find_d 
  (d : ℝ)
  (d_gt_zero : d > 0)
  (line_eq : ∀ x : ℝ, (2 * x - 6 = 0) → x = 3)
  (y_intercept : ∀ y : ℝ, (2 * 0 - 6 = y) → y = -6)
  (area_condition : (1/2 * 3 * 6 = 9) → (1/2 * (d - 3) * (2 * d - 6) = 36)) :
  d = 9 :=
sorry

end find_d_l88_88992


namespace probability_sum_greater_than_four_l88_88229

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l88_88229


namespace total_blocks_per_day_l88_88181

def blocks_to_park : ℕ := 4
def blocks_to_hs : ℕ := 7
def blocks_to_home : ℕ := 11
def walks_per_day : ℕ := 3

theorem total_blocks_per_day :
  (blocks_to_park + blocks_to_hs + blocks_to_home) * walks_per_day = 66 :=
by
  sorry

end total_blocks_per_day_l88_88181


namespace steve_take_home_pay_l88_88400

def annual_salary : ℕ := 40000
def tax_percentage : ℝ := 0.20
def healthcare_percentage : ℝ := 0.10
def union_dues : ℕ := 800

theorem steve_take_home_pay :
  annual_salary - (annual_salary * tax_percentage).to_nat - (annual_salary * healthcare_percentage).to_nat - union_dues = 27200 :=
by
  sorry

end steve_take_home_pay_l88_88400


namespace necessary_but_not_sufficient_condition_l88_88137

theorem necessary_but_not_sufficient_condition (x : ℝ) : (|x - 1| < 1 → x^2 - 5 * x < 0) ∧ (¬(x^2 - 5 * x < 0 → |x - 1| < 1)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l88_88137


namespace largest_common_factor_462_330_l88_88853

-- Define the factors of 462
def factors_462 : Set ℕ := {1, 2, 3, 6, 7, 14, 21, 33, 42, 66, 77, 154, 231, 462}

-- Define the factors of 330
def factors_330 : Set ℕ := {1, 2, 3, 5, 6, 10, 11, 15, 30, 33, 55, 66, 110, 165, 330}

-- Define the statement of the theorem
theorem largest_common_factor_462_330 : 
  (∀ d : ℕ, d ∈ (factors_462 ∩ factors_330) → d ≤ 66) ∧
  66 ∈ (factors_462 ∩ factors_330) :=
sorry

end largest_common_factor_462_330_l88_88853


namespace complement_union_l88_88343

open Set

universe u

variable {U : Type u} [Fintype U] [DecidableEq U]
variable {A B : Set U}

def complement (s : Set U) : Set U := {x | x ∉ s}

theorem complement_union {U : Set ℕ} (A B : Set ℕ) 
  (h1 : complement A ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : complement A ∩ complement B = {2}) :
  complement (A ∪ B) = {2} :=
by sorry

end complement_union_l88_88343


namespace fraction_value_l88_88641

theorem fraction_value (x : ℝ) (h : 1 - 5 / x + 6 / x^3 = 0) : 3 / x = 3 / 2 :=
by
  sorry

end fraction_value_l88_88641


namespace power_addition_rule_l88_88510

variable {a : ℝ}
variable {m n : ℕ}

theorem power_addition_rule (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end power_addition_rule_l88_88510


namespace oranges_packed_in_a_week_l88_88870

open Nat

def oranges_per_box : Nat := 15
def boxes_per_day : Nat := 2150
def days_per_week : Nat := 7

theorem oranges_packed_in_a_week : oranges_per_box * boxes_per_day * days_per_week = 225750 :=
  sorry

end oranges_packed_in_a_week_l88_88870


namespace find_f_half_l88_88120

noncomputable def g (x : ℝ) : ℝ := 1 - 2 * x
noncomputable def f (y : ℝ) : ℝ := if y ≠ 0 then (1 - y^2) / y^2 else 0

theorem find_f_half :
  f (g (1 / 4)) = 15 :=
by
  have g_eq : g (1 / 4) = 1 / 2 := sorry
  rw [g_eq]
  have f_eq : f (1 / 2) = 15 := sorry
  exact f_eq

end find_f_half_l88_88120


namespace quadratic_equal_roots_l88_88950

theorem quadratic_equal_roots (a : ℝ) : (∀ x : ℝ, x * (x + 1) + a * x = 0) → a = -1 :=
by sorry

end quadratic_equal_roots_l88_88950


namespace quadratic_equal_roots_l88_88947

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ (x * (x + 1) + a * x = 0) ∧ ((1 + a)^2 = 0)) →
  a = -1 :=
by
  sorry

end quadratic_equal_roots_l88_88947


namespace rect_RS_over_HJ_zero_l88_88392

theorem rect_RS_over_HJ_zero :
  ∃ (A B C D H I J R S: ℝ × ℝ),
    (A = (0, 6)) ∧
    (B = (8, 6)) ∧
    (C = (8, 0)) ∧
    (D = (0, 0)) ∧
    (H = (5, 6)) ∧
    (I = (8, 4)) ∧
    (J = (3, 0)) ∧
    (R = (15 / 13, -12 / 13)) ∧
    (S = (15 / 13, -12 / 13)) ∧
    (RS = dist R S) ∧
    (HJ = dist H J) ∧
    (HJ ≠ 0) ∧
    (RS / HJ = 0) :=
sorry

end rect_RS_over_HJ_zero_l88_88392


namespace inequality1_inequality2_l88_88830

theorem inequality1 (x : ℝ) : x ≠ 2 → (x + 1)/(x - 2) ≥ 3 → 2 < x ∧ x ≤ 7/2 :=
sorry

theorem inequality2 (x a : ℝ) : 
  (x^2 - a * x - 2 * a^2 ≤ 0) → 
  (a = 0 → x = 0) ∧ 
  (a > 0 → -a ≤ x ∧ x ≤ 2 * a) ∧ 
  (a < 0 → 2 * a ≤ x ∧ x ≤ -a) :=
sorry

end inequality1_inequality2_l88_88830


namespace number_arrangement_impossible_l88_88745

theorem number_arrangement_impossible :
  ¬ ∃ (a b : Fin (3972)) (S T : Finset (Fin (3972))),
    S.card = 1986 ∧ T.card = 1986 ∧
    (∀ k : Nat, 1 ≤ k ∧ k ≤ 1986 →
      ∃ (ak bk : Fin (3972)), ak ∈ S ∧ bk ∈ T ∧ ak < bk ∧ bk.val - ak.val = k + 1) :=
sorry

end number_arrangement_impossible_l88_88745


namespace value_of_a_pow_sum_l88_88790

variable {a : ℝ}
variable {m n : ℕ}

theorem value_of_a_pow_sum (h1 : a^m = 5) (h2 : a^n = 3) : a^(m + n) = 15 := by
  sorry

end value_of_a_pow_sum_l88_88790


namespace opposite_of_neg_3_is_3_l88_88703

theorem opposite_of_neg_3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg_3_is_3_l88_88703


namespace john_max_correct_answers_l88_88591

theorem john_max_correct_answers 
  (c w b : ℕ) -- define c, w, b as natural numbers
  (h1 : c + w + b = 30) -- condition 1: total questions
  (h2 : 4 * c - 3 * w = 36) -- condition 2: scoring equation
  : c ≤ 12 := -- statement to prove
sorry

end john_max_correct_answers_l88_88591


namespace best_sampling_method_l88_88731

/-- 
  Given a high school that wants to understand the psychological 
  pressure of students from three different grades, prove that 
  stratified sampling is the best method to use, assuming students
  from different grades may experience different levels of psychological
  pressure.
-/
theorem best_sampling_method
  (students_from_three_grades : Type)
  (survey_psychological_pressure : students_from_three_grades → ℝ)
  (potential_differences_by_grade : students_from_three_grades → ℝ → Prop):
  ∃ sampling_method, sampling_method = "stratified_sampling" :=
sorry

end best_sampling_method_l88_88731


namespace eq_fractions_l88_88575

theorem eq_fractions : 
  (1 + 1 / (1 + 1 / (1 + 1 / 2))) = 8 / 5 := 
  sorry

end eq_fractions_l88_88575


namespace find_value_l88_88926

theorem find_value (
  a b c d e f : ℝ) 
  (h1 : a * b * c = 65) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 1000) 
  (h4 : (a * f) / (c * d) = 0.25) :
  d * e * f = 250 := 
sorry

end find_value_l88_88926


namespace smallest_n_for_simplest_form_l88_88919

-- Definitions and conditions
def simplest_form_fractions (n : ℕ) :=
  ∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + 2) = 1

-- Problem statement
theorem smallest_n_for_simplest_form :
  ∃ n : ℕ, simplest_form_fractions (n) ∧ ∀ m : ℕ, m < n → ¬ simplest_form_fractions (m) := 
by 
  sorry

end smallest_n_for_simplest_form_l88_88919


namespace part_I_part_II_l88_88929

noncomputable def f (x : ℝ) (a : ℝ) := x - (2 * a - 1) / x - 2 * a * Real.log x

theorem part_I (a : ℝ) (h : a = 3 / 2) : 
  (∀ x, 0 < x ∧ x < 1 → f x a < 0) ∧ (∀ x, 1 < x ∧ x < 2 → f x a > 0) ∧ (∀ x, 2 < x → f x a < 0) := sorry

theorem part_II (a : ℝ) : (∀ x, 1 ≤ x → f x a ≥ 0) → a ≤ 1 := sorry

end part_I_part_II_l88_88929


namespace no_simultaneous_squares_l88_88477

theorem no_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + 2 * y = a^2 ∧ y^2 + 2 * x = b^2) :=
by
  sorry

end no_simultaneous_squares_l88_88477


namespace min_value_of_a_plus_b_minus_c_l88_88627

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ)
  (h : ∀ x y : ℝ, 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) :
  a = 3 ∧ b = 4 ∧ -5 ≤ c ∧ c ≤ 5 ∧ a + b - c = 2 :=
by {
  sorry
}

end min_value_of_a_plus_b_minus_c_l88_88627


namespace probability_sum_greater_than_four_l88_88228

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l88_88228


namespace proof_C_I_M_cap_N_l88_88375

open Set

variable {𝕜 : Type _} [LinearOrderedField 𝕜]

def I : Set 𝕜 := Set.univ
def M : Set 𝕜 := {x : 𝕜 | -2 ≤ x ∧ x ≤ 2}
def N : Set 𝕜 := {x : 𝕜 | x < 1}
def C_I (A : Set 𝕜) : Set 𝕜 := I \ A

theorem proof_C_I_M_cap_N :
  C_I M ∩ N = {x : 𝕜 | x < -2} := by
  sorry

end proof_C_I_M_cap_N_l88_88375


namespace students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l88_88154

structure BusRoute where
  first_stop : Nat
  second_stop_on : Nat
  second_stop_off : Nat
  third_stop_on : Nat
  third_stop_off : Nat
  fourth_stop_on : Nat
  fourth_stop_off : Nat

def mondays_and_wednesdays := BusRoute.mk 39 29 12 35 18 27 15
def tuesdays_and_thursdays := BusRoute.mk 39 33 10 5 0 8 4
def fridays := BusRoute.mk 39 25 10 40 20 10 5

def students_after_last_stop (route : BusRoute) : Nat :=
  let stop1 := route.first_stop
  let stop2 := stop1 + route.second_stop_on - route.second_stop_off
  let stop3 := stop2 + route.third_stop_on - route.third_stop_off
  stop3 + route.fourth_stop_on - route.fourth_stop_off

theorem students_after_last_stop_on_mondays_and_wednesdays :
  students_after_last_stop mondays_and_wednesdays = 85 := by
  sorry

theorem students_after_last_stop_on_tuesdays_and_thursdays :
  students_after_last_stop tuesdays_and_thursdays = 71 := by
  sorry

theorem students_after_last_stop_on_fridays :
  students_after_last_stop fridays = 79 := by
  sorry

end students_after_last_stop_on_mondays_and_wednesdays_students_after_last_stop_on_tuesdays_and_thursdays_students_after_last_stop_on_fridays_l88_88154


namespace number_of_apples_l88_88608

theorem number_of_apples (A : ℝ) (h : 0.75 * A * 0.5 + 0.25 * A * 0.1 = 40) : A = 100 :=
by
  sorry

end number_of_apples_l88_88608


namespace pollutant_decay_l88_88449

noncomputable def p (t : ℝ) (p0 : ℝ) := p0 * 2^(-t / 30)

theorem pollutant_decay : 
  ∃ p0 : ℝ, p0 = 300 ∧ p 60 p0 = 75 * Real.log 2 := 
by
  sorry

end pollutant_decay_l88_88449


namespace determinant_example_l88_88699

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)
noncomputable def sin_deg (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)

-- Define the determinant of a 2x2 matrix in terms of its entries
def determinant_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Proposed theorem statement in Lean 4
theorem determinant_example : 
  determinant_2x2 (cos_deg 45) (sin_deg 75) (sin_deg 135) (cos_deg 105) = - (Real.sqrt 3 / 2) := 
by sorry

end determinant_example_l88_88699


namespace min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l88_88678

section
variables {a x : ℝ}

/-- Define the function f(x) = ax^3 - 2x^2 + x + c where c = 1 -/
def f (a x : ℝ) : ℝ := a * x^3 - 2 * x^2 + x + 1

/-- Proposition 1: Minimum value of f when a = 1 and f passes through (0,1) is 1 -/
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f 1 x ≥ 1) := 
by {
  -- Sorry for the full proof
  sorry
}

/-- Proposition 2: If f has no extremum points, then a ≥ 4/3 -/
theorem no_extrema_implies_a_ge_four_thirds (h : ∀ x : ℝ, 3 * a * x^2 - 4 * x + 1 ≠ 0) : 
  a ≥ (4 / 3) :=
by {
  -- Sorry for the full proof
  sorry
}

end

end min_value_f_when_a_eq_1_no_extrema_implies_a_ge_four_thirds_l88_88678


namespace extrema_of_f_l88_88205

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3 * x + 1

theorem extrema_of_f :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) :=
by
  sorry

end extrema_of_f_l88_88205


namespace find_a_for_square_binomial_l88_88915

theorem find_a_for_square_binomial (a : ℚ) : (∃ (r s : ℚ), a = r^2 ∧ 20 = 2 * r * s ∧ 9 = s^2) → a = 100 / 9 :=
by
  intro h
  cases' h with r hr
  cases' hr with s hs
  cases' hs with ha1 hs1
  cases' hs1 with ha2 ha3
  have s_val : s = 3 ∨ s = -3 := by
    have s2_eq := eq_of_sq_eq_sq ha3
    subst s; split; linarith; linarith
  cases s_val with s_eq3 s_eq_neg3
  -- case s = 3
  { rw [s_eq3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }
  -- case s = -3
  { rw [s_eq_neg3, mul_assoc] at ha2
    simp at ha2
    subst r; subst s
    norm_num
    simp [ha2, ha1, show (10/3:ℚ) ^ 2 = 100/9 from by norm_num] }

end find_a_for_square_binomial_l88_88915


namespace prove_range_of_a_l88_88498

noncomputable def f (x a : ℝ) := x^2 + (a + 1) * x + Real.log (abs (a + 2))

def is_increasing (f : ℝ → ℝ) (interval : Set ℝ) :=
 ∀ ⦃x y⦄, x ∈ interval → y ∈ interval → x ≤ y → f x ≤ f y

def g (x a : ℝ) := (a + 1) * x
def is_decreasing (g : ℝ → ℝ) :=
 ∀ ⦃x y⦄, x ≤ y → g y ≤ g x

def proposition_p (a : ℝ) : Prop :=
  is_increasing (f a) (Set.Ici ((a + 1)^2))

def proposition_q (a : ℝ) : Prop :=
  is_decreasing (g a)

theorem prove_range_of_a (a : ℝ) (h : ¬ (proposition_p a ↔ proposition_q a)) :
  a > -3 / 2 :=
sorry

end prove_range_of_a_l88_88498


namespace area_of_curvilinear_trapezoid_steps_l88_88327

theorem area_of_curvilinear_trapezoid_steps (steps : List String) :
  (steps = ["division", "approximation", "summation", "taking the limit"]) :=
sorry

end area_of_curvilinear_trapezoid_steps_l88_88327


namespace probability_sum_greater_than_four_l88_88256

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l88_88256


namespace speed_in_km_per_hr_l88_88063

noncomputable def side : ℝ := 40
noncomputable def time : ℝ := 64

-- Theorem statement
theorem speed_in_km_per_hr (side : ℝ) (time : ℝ) (h₁ : side = 40) (h₂ : time = 64) : 
  (4 * side * 3600) / (time * 1000) = 9 := by
  rw [h₁, h₂]
  sorry

end speed_in_km_per_hr_l88_88063


namespace units_digit_m_squared_plus_2_pow_m_l88_88012

-- Define the value of m
def m : ℕ := 2023^2 + 2^2023

-- Define the property we need to prove
theorem units_digit_m_squared_plus_2_pow_m :
  ((m^2 + 2^m) % 10) = 7 :=
by
  sorry

end units_digit_m_squared_plus_2_pow_m_l88_88012


namespace valid_assignment_l88_88571

/-- A function to check if an expression is a valid assignment expression -/
def is_assignment (lhs : String) (rhs : String) : Prop :=
  lhs = "x" ∧ (rhs = "3" ∨ rhs = "x + 1")

theorem valid_assignment :
  (is_assignment "x" "x + 1") ∧
  ¬(is_assignment "3" "x") ∧
  ¬(is_assignment "x" "3") ∧
  ¬(is_assignment "x" "x2 + 1") :=
by
  sorry

end valid_assignment_l88_88571


namespace distinct_arrangement_count_l88_88025

open Finset

noncomputable def distinct_letter_arrangements (grid_size : ℕ) : ℕ :=
  (choose (grid_size * grid_size) 2 * (choose ((grid_size - 1) * (grid_size - 1)) 2))

theorem distinct_arrangement_count :
  distinct_letter_arrangements 4 = 120 := 
by
  sorry

end distinct_arrangement_count_l88_88025


namespace parallel_lines_iff_m_eq_neg2_l88_88838

theorem parallel_lines_iff_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0 → m * x + 2 * y - m + 2 = 0 ↔ m = -2) :=
sorry

end parallel_lines_iff_m_eq_neg2_l88_88838


namespace solution_set_of_inequality_l88_88038

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h₁ : ∀ x > 0, deriv f x + 2 * f x > 0) :
  {x : ℝ | x + 2018 > 0 ∧ x + 2018 < 5} = {x : ℝ | -2018 < x ∧ x < -2013} := 
by
  sorry

end solution_set_of_inequality_l88_88038


namespace vitamin_supplement_problem_l88_88377

theorem vitamin_supplement_problem :
  let packA := 7
  let packD := 17
  (∀ n : ℕ, n ≠ 0 → (packA * n = packD * n)) → n = 119 :=
by
  sorry

end vitamin_supplement_problem_l88_88377


namespace bruce_and_anne_clean_house_l88_88472

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l88_88472


namespace smallest_number_of_pencils_l88_88185

theorem smallest_number_of_pencils 
  (p : ℕ) 
  (h1 : p % 6 = 5)
  (h2 : p % 7 = 3)
  (h3 : p % 8 = 7) :
  p = 35 := 
sorry

end smallest_number_of_pencils_l88_88185


namespace pens_cost_l88_88742

theorem pens_cost (pens_pack_cost : ℝ) (pens_pack_quantity : ℕ) (total_pens : ℕ) (unit_price : ℝ) (total_cost : ℝ)
  (h1 : pens_pack_cost = 45) (h2 : pens_pack_quantity = 150) (h3 : total_pens = 3600) (h4 : unit_price = pens_pack_cost / pens_pack_quantity)
  (h5 : total_cost = total_pens * unit_price) : total_cost = 1080 := by
  sorry

end pens_cost_l88_88742


namespace triangle_area_from_squares_l88_88347

noncomputable def area_of_triangle (S1 S2 : ℝ) : ℝ :=
  let side1 := Real.sqrt S1
  let side2 := Real.sqrt S2
  0.5 * side1 * side2

theorem triangle_area_from_squares
  (A1 A2 : ℝ)
  (h1 : A1 = 196)
  (h2 : A2 = 100) :
  area_of_triangle A1 A2 = 70 :=
by
  rw [h1, h2]
  unfold area_of_triangle
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
  norm_num
  sorry

end triangle_area_from_squares_l88_88347


namespace find_prime_solution_l88_88325

theorem find_prime_solution :
  ∀ p x y : ℕ, Prime p → x > 0 → y > 0 →
    (p ^ x = y ^ 3 + 1) ↔ 
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := 
by
  sorry

end find_prime_solution_l88_88325


namespace find_f2_l88_88928

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + a^(-x)

theorem find_f2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) : f a 2 = 7 := 
by 
  sorry

end find_f2_l88_88928


namespace taimour_paint_time_l88_88061

theorem taimour_paint_time (T : ℝ) :
  (1 / T + 2 / T) * 7 = 1 → T = 21 :=
by
  intro h
  sorry

end taimour_paint_time_l88_88061


namespace percentage_is_12_l88_88941

variable (x : ℝ) (p : ℝ)

-- Given the conditions
def condition_1 : Prop := 0.25 * x = (p / 100) * 1500 - 15
def condition_2 : Prop := x = 660

-- We need to prove that the percentage p is 12
theorem percentage_is_12 (h1 : condition_1 x p) (h2 : condition_2 x) : p = 12 := by
  sorry

end percentage_is_12_l88_88941


namespace team_A_wins_set_team_A_wins_entire_match_possible_x_values_and_probabilities_l88_88153

noncomputable theory

open ProbabilityTheory

variables {A B : Type}

-- Definition of the probability space for the volleyball match scenario
-- Assuming all events are equally likely.
variables (sample_space : ℕ → ProbState A B) (prob : ℕ → Prob)

-- Part (1): Probability of Team A winning
def prob_team_A_wins_set : ℕ → ℝ := 1 / 2 -- The probability that Team A wins a set.
def prob_team_A_wins : ℝ := 3 / 4 -- The probability that Team A wins the entire match.

-- Part (2): Possible values of x and corresponding probabilities
def prob_team_A_scores_serving : ℝ := 2 / 5 -- Probability of Team A scoring when they are serving
def prob_team_A_scores_receiving : ℝ := 3 / 5 -- Probability of Team A scoring when they are receiving
def prob_x_less_than_eq_4 : ℝ := 172 / 625 -- The combined probability of Team A winning in 2 or 4 rallies.

theorem team_A_wins_set (prob_A_wins_step : ℝ) : 
  prob_team_A_wins_set ∘ sample_space = prob ∘ prob_team_A_wins_set :=
begin
  sorry
end

theorem team_A_wins_entire_match : 
  prob_team_A_wins (sample_space 1) = 3 / 4 :=
begin
  sorry
end

theorem possible_x_values_and_probabilities :
  possible_x_values ∘ sample_space = 
    ∃ (x : ℕ), (x ≤ 4 ∧ (P(x = 2) = 4 / 25) ∧ (P(x = 4) = 72 / 625)) :=
begin
  sorry
end

end team_A_wins_set_team_A_wins_entire_match_possible_x_values_and_probabilities_l88_88153


namespace football_defeat_points_l88_88801

theorem football_defeat_points (V D F : ℕ) (x : ℕ) :
    3 * V + D + x * F = 8 →
    27 + 6 * x = 32 →
    x = 0 :=
by
    intros h1 h2
    sorry

end football_defeat_points_l88_88801


namespace one_minus_repeating_three_l88_88092

theorem one_minus_repeating_three : ∀ b : ℚ, b = 1 / 3 → 1 - b = 2 / 3 :=
by
  intro b hb
  rw [hb]
  norm_num

end one_minus_repeating_three_l88_88092


namespace find_c_of_binomial_square_l88_88936

theorem find_c_of_binomial_square (c : ℝ) (h : ∃ d : ℝ, (9*x^2 - 24*x + c = (3*x + d)^2)) : c = 16 := sorry

end find_c_of_binomial_square_l88_88936


namespace ordering_of_exponentiations_l88_88166

def a : ℕ := 3 ^ 34
def b : ℕ := 2 ^ 51
def c : ℕ := 4 ^ 25

theorem ordering_of_exponentiations : c < b ∧ b < a := by
  sorry

end ordering_of_exponentiations_l88_88166


namespace coin_exchange_l88_88592

theorem coin_exchange :
  ∃ (t1 t2 t5 t10 : ℕ), 
    t2 = (3 / 5) * t1 ∧ 
    t5 = (3 / 5) * t2 ∧ 
    t10 = (3 / 5) * t5 - 7 ∧ 
    (50 ≤ (1 * t1 + 2 * t2 + 5 * t5 + 10 * t10) / 100 ∧ (1 * t1 + 2 * t2 + 5 * t5 + 10 * t10) / 100 ≤ 100) ∧ 
    t1 = 1375 ∧ 
    t2 = 825 ∧ 
    t5 = 495 ∧ 
    t10 = 290 :=
by 
  existsi [1375, 825, 495, 290]
  split; try {norm_num}; intros; linarith

end coin_exchange_l88_88592


namespace find_ratio_eq_eighty_six_l88_88052

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 45}

-- Define the sum of the first n natural numbers function
def sum_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define our specific scenario setup
def selected_numbers (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x * y = sum_n_nat 45 - (x + y)

-- Prove the resulting ratio condition
theorem find_ratio_eq_eighty_six (x y : ℕ) (h : selected_numbers x y) : 
  x < y → y / x = 86 :=
by
  sorry

end find_ratio_eq_eighty_six_l88_88052


namespace replace_square_l88_88792

theorem replace_square (x : ℝ) (h : 10.0003 * x = 10000.3) : x = 1000 :=
sorry

end replace_square_l88_88792


namespace ratio_of_a_to_b_l88_88778

-- Given conditions
variables {a b x : ℝ}
-- a and b are positive real numbers distinct from 1
variables (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1)
-- Given equation involving logarithms
variables (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2)

-- Prove that the ratio of a to b is a^(sqrt(7/5))
theorem ratio_of_a_to_b (h1 : a > 0) (h2 : a ≠ 1) (h3 : b > 0) (h4 : b ≠ 1) (h5 : 5 * (Real.log x / Real.log a) ^ 2 + 7 * (Real.log x / Real.log b) ^ 2 = 10 * (Real.log x) ^ 2) :
  b = a ^ Real.sqrt (7 / 5) :=
sorry

end ratio_of_a_to_b_l88_88778


namespace betty_total_cost_l88_88606

theorem betty_total_cost :
    (6 * 2.5) + (4 * 1.25) + (8 * 3) = 44 :=
by
    sorry

end betty_total_cost_l88_88606


namespace simplify_fraction_l88_88284

theorem simplify_fraction (h1 : 3.36 = 3 + 0.36) 
                          (h2 : 0.36 = (36 : ℚ) / 100) 
                          (h3 : (36 : ℚ) / 100 = 9 / 25) 
                          : 3.36 = 84 / 25 := 
by 
  rw [h1, h2, h3]
  norm_num
  rw [←Rat.add_div, show 3 = 75 / 25 by norm_num]
  norm_num
  
  sorry  -- This line can be safely removed when the proof is complete.

end simplify_fraction_l88_88284


namespace problem_b_problem_d_l88_88139

variable (x y t : ℝ)

def condition_curve (t : ℝ) : Prop :=
  ∃ C : ℝ × ℝ → Prop, ∀ x y : ℝ, C (x, y) ↔ (x^2 / (5 - t) + y^2 / (t - 1) = 1)

theorem problem_b (h1 : t < 1) : condition_curve t → ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → ¬(5 - t) < 0 ∧ (t - 1) < 0 := 
sorry

theorem problem_d (h1 : 3 < t) (h2 : t < 5) (h3 : condition_curve t) : ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → 0 < (t - 1) ∧ (t - 1) > (5 - t) := 
sorry

end problem_b_problem_d_l88_88139


namespace double_root_condition_l88_88161

theorem double_root_condition (a : ℝ) : 
  (∃! x : ℝ, (x+2)^2 * (x+7)^2 + a = 0) ↔ a = -625 / 16 :=
sorry

end double_root_condition_l88_88161


namespace angle_BDC_eq_88_l88_88654

-- Define the problem scenario
variable (A B C : ℝ)
variable (α : ℝ)
variable (B1 B2 B3 C1 C2 C3 : ℝ)

-- Conditions provided
axiom angle_A_eq_42 : α = 42
axiom trisectors_ABC : B = B1 + B2 + B3 ∧ C = C1 + C2 + C3
axiom trisectors_eq : B1 = B2 ∧ B2 = B3 ∧ C1 = C2 ∧ C2 = C3
axiom angle_sum_ABC : α + B + C = 180

-- Proving the measure of ∠BDC
theorem angle_BDC_eq_88 :
  α + (B/3) + (C/3) = 88 :=
by
  sorry

end angle_BDC_eq_88_l88_88654


namespace method_1_more_cost_effective_l88_88593

open BigOperators

def racket_price : ℕ := 20
def shuttlecock_price : ℕ := 5
def rackets_bought : ℕ := 4
def shuttlecocks_bought : ℕ := 30
def discount_rate : ℚ := 0.92

def total_price (rackets shuttlecocks : ℕ) := racket_price * rackets + shuttlecock_price * shuttlecocks

def method_1_cost (rackets shuttlecocks : ℕ) := 
  total_price rackets shuttlecocks - shuttlecock_price * rackets

def method_2_cost (total : ℚ) :=
  total * discount_rate

theorem method_1_more_cost_effective :
  method_1_cost rackets_bought shuttlecocks_bought
  <
  method_2_cost (total_price rackets_bought shuttlecocks_bought) :=
by
  sorry

end method_1_more_cost_effective_l88_88593


namespace hired_is_B_l88_88733

-- Define the individuals
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

open Person

-- Define the statements made by each person
def statement (p : Person) (hired : Person) : Prop :=
  match p with
  | A => hired = C
  | B => hired ≠ B
  | C => hired = D
  | D => hired ≠ D

-- The main theorem is to prove B is hired given the conditions
theorem hired_is_B :
  (∃! p : Person, ∃ t : Person → Prop,
    (∀ h : Person, t h ↔ h = p) ∧
    (∃ q : Person, statement q q ∧ ∀ r : Person, r ≠ q → ¬statement r q) ∧
    t B) :=
by
  sorry

end hired_is_B_l88_88733


namespace determine_a_l88_88624

theorem determine_a (a : ℕ)
  (h1 : 2 / (2 + 3 + a) = 1 / 3) : a = 1 :=
by
  sorry

end determine_a_l88_88624


namespace intersection_complement_range_m_l88_88132

open Set

variable (A : Set ℝ) (B : ℝ → Set ℝ) (m : ℝ)

def setA : Set ℝ := Icc (-1 : ℝ) (3 : ℝ)
def setB (m : ℝ) : Set ℝ := Icc m (m + 6)

theorem intersection_complement (m : ℝ) (h : m = 2) : 
  (setA ∩ (setB 2)ᶜ) = Ico (-1 : ℝ) (2 : ℝ) :=
by
  sorry

theorem range_m (m : ℝ) : 
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 :=
by
  sorry

end intersection_complement_range_m_l88_88132


namespace α_eq_β_plus_two_l88_88116

-- Definitions based on the given conditions:
-- α(n): number of ways n can be expressed as a sum of the integers 1 and 2, considering different orders as distinct ways.
-- β(n): number of ways n can be expressed as a sum of integers greater than 1, considering different orders as distinct ways.

def α (n : ℕ) : ℕ := sorry
def β (n : ℕ) : ℕ := sorry

-- The proof statement that needs to be proved.
theorem α_eq_β_plus_two (n : ℕ) (h : 0 < n) : α n = β (n + 2) := 
  sorry

end α_eq_β_plus_two_l88_88116


namespace anthony_more_shoes_than_jim_l88_88977

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem anthony_more_shoes_than_jim : (anthony_shoes - jim_shoes) = 2 :=
by
  sorry

end anthony_more_shoes_than_jim_l88_88977


namespace initial_percentage_of_managers_l88_88844

theorem initial_percentage_of_managers (P : ℕ) (h : 0 ≤ P ∧ P ≤ 100)
  (total_employees initial_managers : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : initial_managers = P * total_employees / 100) 
  (remaining_employees remaining_managers : ℕ)
  (h3 : remaining_employees = total_employees - 250)
  (h4 : remaining_managers = initial_managers - 250)
  (h5 : remaining_managers * 100 = 98 * remaining_employees) :
  P = 99 := 
by
  sorry

end initial_percentage_of_managers_l88_88844


namespace remainder_when_divided_by_9_l88_88057

variable (k : ℕ)

theorem remainder_when_divided_by_9 :
  (∃ k, k % 5 = 2 ∧ k % 6 = 3 ∧ k % 8 = 7 ∧ k < 100) →
  k % 9 = 6 :=
sorry

end remainder_when_divided_by_9_l88_88057


namespace original_price_l88_88871

theorem original_price (P : ℝ) (h1 : ∃ P : ℝ, (120 : ℝ) = P + 0.2 * P) : P = 100 :=
by
  obtain ⟨P, h⟩ := h1
  sorry

end original_price_l88_88871


namespace simplify_336_to_fraction_l88_88271

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l88_88271


namespace factorize_expr_l88_88762

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l88_88762


namespace probability_sum_greater_than_four_l88_88254

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l88_88254


namespace min_fraction_sum_l88_88775

theorem min_fraction_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (∃ (z : ℝ), z = (1 / (x + 1)) + (4 / (y + 2)) ∧ z = 9 / 4) :=
by 
  sorry

end min_fraction_sum_l88_88775


namespace gcd_is_18_l88_88518

-- Define gcdX that represents the greatest common divisor of X and Y.
noncomputable def gcdX (X Y : ℕ) : ℕ := Nat.gcd X Y

-- Given conditions
def cond_lcm (X Y : ℕ) : Prop := Nat.lcm X Y = 180
def cond_ratio (X Y : ℕ) : Prop := ∃ k : ℕ, X = 2 * k ∧ Y = 5 * k

-- Theorem to prove that the gcd of X and Y is 18
theorem gcd_is_18 {X Y : ℕ} (h1 : cond_lcm X Y) (h2 : cond_ratio X Y) : gcdX X Y = 18 :=
by
  sorry

end gcd_is_18_l88_88518


namespace incorrect_statement_C_l88_88393

theorem incorrect_statement_C (x : ℝ) (h : x > -2) : (6 / x) > -3 :=
sorry

end incorrect_statement_C_l88_88393


namespace dividend_value_l88_88330

def dividend (divisor quotient remainder : ℝ) := (divisor * quotient) + remainder

theorem dividend_value :
  dividend 35.8 21.65 11.3 = 786.47 :=
by
  sorry

end dividend_value_l88_88330


namespace intersection_S_T_l88_88339

open Set

def S : Set ℝ := { x | x ≥ 1 }
def T : Set ℝ := { -2, -1, 0, 1, 2 }

theorem intersection_S_T : S ∩ T = { 1, 2 } := by
  sorry

end intersection_S_T_l88_88339


namespace francie_remaining_money_l88_88490

noncomputable def total_savings_before_investment : ℝ :=
  (5 * 8) + (6 * 6) + 20

noncomputable def investment_return : ℝ :=
  0.05 * 10

noncomputable def total_savings_after_investment : ℝ :=
  total_savings_before_investment + investment_return

noncomputable def spent_on_clothes : ℝ :=
  total_savings_after_investment / 2

noncomputable def remaining_after_clothes : ℝ :=
  total_savings_after_investment - spent_on_clothes

noncomputable def amount_remaining : ℝ :=
  remaining_after_clothes - 35

theorem francie_remaining_money : amount_remaining = 13.25 := 
  sorry

end francie_remaining_money_l88_88490


namespace find_x_solution_l88_88333

theorem find_x_solution (x : ℚ) : (∀ y : ℚ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by
  sorry

end find_x_solution_l88_88333


namespace parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l88_88896

noncomputable def parabola (p x : ℝ) : ℝ := (p-1) * x^2 + 2 * p * x + 4

-- 1. Prove that if \( p = 2 \), the parabola \( g_p \) is tangent to the \( x \)-axis.
theorem parabola_tangent_xaxis_at_p2 : ∀ x, parabola 2 x = (x + 2)^2 := 
by 
  intro x
  sorry

-- 2. Prove that if \( p = 0 \), the vertex of the parabola \( g_p \) lies on the \( y \)-axis.
theorem parabola_vertex_yaxis_at_p0 : ∃ x, parabola 0 x = 4 := 
by 
  sorry

-- 3. Prove the parabolas for \( p = 2 \) and \( p = 0 \) are symmetric with respect to \( M(-1, 2) \).
theorem parabolas_symmetric_m_point : ∀ x, 
  (parabola 2 x = (x + 2)^2) → 
  (parabola 0 x = -x^2 + 4) → 
  (-1, 2) = (-1, 2) := 
by 
  sorry

-- 4. Prove that the points \( (0, 4) \) and \( (-2, 0) \) lie on the curve for all \( p \).
theorem parabola_familiy_point_through : ∀ p, 
  parabola p 0 = 4 ∧ 
  parabola p (-2) = 0 :=
by 
  sorry

end parabola_tangent_xaxis_at_p2_parabola_vertex_yaxis_at_p0_parabolas_symmetric_m_point_parabola_familiy_point_through_l88_88896


namespace math_problem_solution_l88_88340

open Real

noncomputable def math_problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a + b + c + d = 4) : Prop :=
  (b / sqrt (a + 2 * c) + c / sqrt (b + 2 * d) + d / sqrt (c + 2 * a) + a / sqrt (d + 2 * b)) ≥ (4 * sqrt 3) / 3

theorem math_problem_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) :
  math_problem a b c d ha hb hc hd h := by sorry

end math_problem_solution_l88_88340


namespace sum_of_interior_angles_pentagon_l88_88708

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end sum_of_interior_angles_pentagon_l88_88708


namespace probability_correct_l88_88232

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l88_88232


namespace compare_fractions_l88_88319

theorem compare_fractions : (-8 / 21: ℝ) > (-3 / 7: ℝ) :=
sorry

end compare_fractions_l88_88319


namespace find_power_l88_88869

noncomputable def x : Real := 14.500000000000002
noncomputable def target : Real := 126.15

theorem find_power (n : Real) (h : (3/5) * x^n = target) : n = 2 :=
sorry

end find_power_l88_88869


namespace bruce_anne_clean_in_4_hours_l88_88461

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l88_88461


namespace a_minus_b_eq_zero_l88_88796

-- Definitions from the conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- The point (0, b)
def point_b (b : ℝ) : (ℝ × ℝ) := (0, b)

-- Slope condition at point (0, b)
def slope_of_f_at_0 (a : ℝ) : ℝ := a
def slope_of_tangent_line : ℝ := 1

-- Prove a - b = 0 given the conditions
theorem a_minus_b_eq_zero (a b : ℝ) 
    (h1 : f 0 a b = b)
    (h2 : tangent_line 0 b) 
    (h3 : slope_of_f_at_0 a = slope_of_tangent_line) : a - b = 0 :=
by
  sorry

end a_minus_b_eq_zero_l88_88796


namespace triangle_area_546_l88_88883

theorem triangle_area_546 :
  ∀ (a b c : ℕ), a = 13 ∧ b = 84 ∧ c = 85 ∧ a^2 + b^2 = c^2 →
  (1 / 2 : ℝ) * (a * b) = 546 :=
by
  intro a b c
  intro h
  sorry

end triangle_area_546_l88_88883


namespace rectangle_ratio_l88_88487

theorem rectangle_ratio (s y x : ℝ) 
  (inner_square_area outer_square_area : ℝ) 
  (h1 : inner_square_area = s^2)
  (h2 : outer_square_area = 9 * inner_square_area)
  (h3 : outer_square_area = (3 * s)^2)
  (h4 : s + 2 * y = 3 * s)
  (h5 : x + y = 3 * s)
  : x / y = 2 := 
by
  -- Proof steps will go here
  sorry

end rectangle_ratio_l88_88487


namespace sum_of_ages_is_18_l88_88157

-- Define the conditions
def product_of_ages (kiana twin : ℕ) := kiana * twin^2 = 128

-- Define the proof problem statement
theorem sum_of_ages_is_18 : ∃ (kiana twin : ℕ), product_of_ages kiana twin ∧ twin > kiana ∧ kiana + twin + twin = 18 :=
by
  sorry

end sum_of_ages_is_18_l88_88157


namespace ellipse_problem_l88_88815

-- Let the left and right foci of the ellipse be F1 and F2 respectively
def ellipse_foci (a b c : ℝ) (hab : a > b) (hbc : b > 0) (h_eq : a = 2 * c) : Prop :=
  let F1 := (-c, 0)
  let F2 := (c, 0)
  ∃ P : ℝ × ℝ, P = (a, b) ∧ (a > b) ∧ (b > 0) ∧ (∥P - F2∥ = ∥F1 - F2∥) 

-- Given conditions and calculate eccentricity 'e'
def eccentricity (a c : ℝ) (h_eq : a = 2 * c) : ℝ := c / a

-- The main problem statement to prove
theorem ellipse_problem (a b c : ℝ)
  (hab : a > b) (hbc : b > 0) (h_eq : a = 2 * c) 
  (h_eccentricity : eccentricity a c h_eq = 1/2) :
  ellipse_foci a b c hab hbc h_eq → ∃ k: ℝ, ellipse.eq (a:=k*sqrt 3) (b:=3*k) (c:=4*k^2) :=
sorry

end ellipse_problem_l88_88815


namespace bruce_and_anne_clean_house_l88_88473

theorem bruce_and_anne_clean_house 
  (A : ℚ) (B : ℚ) (H1 : A = 1/12) 
  (H2 : 3 * (B + 2 * A) = 1) :
  1 / (B + A) = 4 := 
sorry

end bruce_and_anne_clean_house_l88_88473


namespace sum_of_integers_l88_88201

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 144) : x + y = 24 :=
sorry

end sum_of_integers_l88_88201


namespace maximal_n_is_k_minus_1_l88_88717

section
variable (k : ℕ) (n : ℕ)
variable (cards : Finset ℕ)
variable (red : List ℕ) (blue : List (List ℕ))

-- Conditions
axiom h_k_pos : k > 1
axiom h_card_count : cards = Finset.range (2 * n + 1)
axiom h_initial_red : red = (List.range' 1 (2 * n)).reverse
axiom h_initial_blue : blue.length = k

-- Question translated to a goal
theorem maximal_n_is_k_minus_1 (h : ∀ (n' : ℕ), n' ≤ (k - 1)) : n = k - 1 :=
sorry
end

end maximal_n_is_k_minus_1_l88_88717


namespace base6_to_decimal_l88_88648

theorem base6_to_decimal (m : ℕ) (h : 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 4934) : m = 4 :=
by
  sorry

end base6_to_decimal_l88_88648


namespace length_of_arc_l88_88832

def radius : ℝ := 5
def area_of_sector : ℝ := 10
def expected_length_of_arc : ℝ := 4

theorem length_of_arc (r : ℝ) (A : ℝ) (l : ℝ) (h₁ : r = radius) (h₂ : A = area_of_sector) : l = expected_length_of_arc := by
  sorry

end length_of_arc_l88_88832


namespace number_div_by_3_l88_88443

theorem number_div_by_3 (x : ℕ) (h : 54 = x - 39) : x / 3 = 31 :=
by
  sorry

end number_div_by_3_l88_88443


namespace sum_of_squares_l88_88523

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hwx : w * x = a^2) 
  (hwy : w * y = b^2) 
  (hwz : w * z = c^2) 
  (hw : w ≠ 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := 
by
  sorry

end sum_of_squares_l88_88523


namespace problem1_correct_problem2_correct_l88_88475

noncomputable def problem1 : ℚ :=
  (1/2 - 5/9 + 7/12) * (-36)

theorem problem1_correct : problem1 = -19 := 
by 
  sorry

noncomputable def mixed_number (a : ℤ) (b : ℚ) : ℚ := a + b

noncomputable def problem2 : ℚ :=
  (mixed_number (-199) (24/25)) * 5

theorem problem2_correct : problem2 = -999 - 4/5 :=
by
  sorry

end problem1_correct_problem2_correct_l88_88475


namespace range_of_a_l88_88626

theorem range_of_a (a m : ℝ) (hp : 3 * a < m ∧ m < 4 * a) 
  (hq : 1 < m ∧ m < 3 / 2) :
  1 / 3 ≤ a ∧ a ≤ 3 / 8 :=
by
  sorry

end range_of_a_l88_88626


namespace inequality_solution_range_l88_88140

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x ^ 2 - 8 * x - 4 - a > 0) → a < -4 :=
by
  sorry

end inequality_solution_range_l88_88140


namespace brody_calculator_battery_life_l88_88891

theorem brody_calculator_battery_life (h : ∃ t : ℕ, (3 / 4) * t + 2 + 13 = t) : ∃ t : ℕ, t = 60 :=
by
  -- Define the quarters used by Brody and the remaining battery life after the exam.
  obtain ⟨t, ht⟩ := h
  -- Simplify the equation (3/4) * t + 2 + 13 = t to get t = 60
  sorry

end brody_calculator_battery_life_l88_88891


namespace find_a_for_binomial_square_l88_88908

theorem find_a_for_binomial_square :
  ∃ a : ℚ, (∀ x : ℚ, (∃ r : ℚ, 6 * r = 20 ∧ (r^2 * x^2 + 6 * r * x + 9) = ax^2 + 20x + 9)) ∧ a = 100 / 9 :=
by
  sorry

end find_a_for_binomial_square_l88_88908


namespace negate_proposition_l88_88501

open Classical

variable (x : ℝ)

theorem negate_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0 :=
by
  sorry

end negate_proposition_l88_88501


namespace mandy_difference_of_cinnamon_and_nutmeg_l88_88969

theorem mandy_difference_of_cinnamon_and_nutmeg :
  let cinnamon := 0.6666666666666666
  let nutmeg := 0.5
  let difference := cinnamon - nutmeg
  difference = 0.1666666666666666 :=
by
  sorry

end mandy_difference_of_cinnamon_and_nutmeg_l88_88969


namespace find_n_from_binomial_expansion_l88_88768

theorem find_n_from_binomial_expansion (x a : ℝ) (n : ℕ)
  (h4 : (Nat.choose n 3) * x^(n - 3) * a^3 = 210)
  (h5 : (Nat.choose n 4) * x^(n - 4) * a^4 = 420)
  (h6 : (Nat.choose n 5) * x^(n - 5) * a^5 = 630) :
  n = 19 :=
sorry

end find_n_from_binomial_expansion_l88_88768


namespace tan_4x_eq_cos_x_has_9_solutions_l88_88134

theorem tan_4x_eq_cos_x_has_9_solutions :
  ∃ (s : Finset ℝ), s.card = 9 ∧ ∀ x ∈ s, (0 ≤ x ∧ x ≤ 2 * Real.pi) ∧ (Real.tan (4 * x) = Real.cos x) :=
sorry

end tan_4x_eq_cos_x_has_9_solutions_l88_88134


namespace Anthony_vs_Jim_l88_88984

variable (Scott_pairs : ℕ)
variable (Anthony_pairs : ℕ)
variable (Jim_pairs : ℕ)

axiom Scott_value : Scott_pairs = 7
axiom Anthony_value : Anthony_pairs = 3 * Scott_pairs
axiom Jim_value : Jim_pairs = Anthony_pairs - 2

theorem Anthony_vs_Jim (Scott_pairs Anthony_pairs Jim_pairs : ℕ) 
  (Scott_value : Scott_pairs = 7) 
  (Anthony_value : Anthony_pairs = 3 * Scott_pairs) 
  (Jim_value : Jim_pairs = Anthony_pairs - 2) :
  Anthony_pairs - Jim_pairs = 2 := 
sorry

end Anthony_vs_Jim_l88_88984


namespace net_salary_change_l88_88214

variable (S : ℝ)

theorem net_salary_change (h1 : S > 0) : 
  (1.3 * S - 0.3 * (1.3 * S)) - S = -0.09 * S := by
  sorry

end net_salary_change_l88_88214


namespace vector_equation_solution_l88_88506

open Real

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b x : V) (h : 3 • a + 4 • (b - x) = 0) : 
  x = (3 / 4) • a + b := 
sorry

end vector_equation_solution_l88_88506


namespace marc_watch_days_l88_88016

theorem marc_watch_days (bought_episodes : ℕ) (watch_fraction : ℚ) (episodes_per_day : ℚ) (total_days : ℕ) : 
  bought_episodes = 50 → 
  watch_fraction = 1 / 10 → 
  episodes_per_day = (50 : ℚ) * watch_fraction → 
  total_days = (bought_episodes : ℚ) / episodes_per_day →
  total_days = 10 := 
sorry

end marc_watch_days_l88_88016


namespace sum_of_remaining_digit_is_correct_l88_88268

-- Define the local value calculation function for a particular digit with its place value
def local_value (digit place_value : ℕ) : ℕ := digit * place_value

-- Define the number in question
def number : ℕ := 2345

-- Define the local values for each digit in their respective place values
def local_value_2 : ℕ := local_value 2 1000
def local_value_3 : ℕ := local_value 3 100
def local_value_4 : ℕ := local_value 4 10
def local_value_5 : ℕ := local_value 5 1

-- Define the given sum of the local values
def given_sum : ℕ := 2345

-- Define the sum of the local values of the digits 2, 3, and 5
def sum_of_other_digits : ℕ := local_value_2 + local_value_3 + local_value_5

-- Define the target sum which is the sum of the local value of the remaining digit
def target_sum : ℕ := given_sum - sum_of_other_digits

-- Prove that the sum of the local value of the remaining digit is equal to 40
theorem sum_of_remaining_digit_is_correct : target_sum = 40 := 
by
  -- The proof will be provided here
  sorry

end sum_of_remaining_digit_is_correct_l88_88268


namespace remaining_pictures_l88_88058

-- Definitions based on the conditions
def pictures_in_first_book : ℕ := 44
def pictures_in_second_book : ℕ := 35
def pictures_in_third_book : ℕ := 52
def pictures_in_fourth_book : ℕ := 48
def colored_pictures : ℕ := 37

-- Statement of the theorem based on the question and correct answer
theorem remaining_pictures :
  pictures_in_first_book + pictures_in_second_book + pictures_in_third_book + pictures_in_fourth_book - colored_pictures = 142 := by
  sorry

end remaining_pictures_l88_88058


namespace find_x_l88_88514
-- Lean 4 equivalent problem setup

-- Assuming a and b are the tens and units digits respectively.
def number (a b : ℕ) := 10 * a + b
def interchangedNumber (a b : ℕ) := 10 * b + a
def digitsDifference (a b : ℕ) := a - b

-- Given conditions
variable (a b k : ℕ)

def condition1 := number a b = k * digitsDifference a b
def condition2 (x : ℕ) := interchangedNumber a b = x * digitsDifference a b

-- Theorem to prove
theorem find_x (h1 : condition1 a b k) : ∃ x, condition2 a b x ∧ x = k - 9 := 
by sorry

end find_x_l88_88514


namespace sum_of_angles_is_360_l88_88808

-- Let's define the specific angles within our geometric figure
variables (A B C D F G : ℝ)

-- Define a condition stating that these angles form a quadrilateral inside a geometric figure, such that their sum is valid
def angles_form_quadrilateral (A B C D F G : ℝ) : Prop :=
  (A + B + C + D + F + G = 360)

-- Finally, we declare the theorem we want to prove
theorem sum_of_angles_is_360 (A B C D F G : ℝ) (h : angles_form_quadrilateral A B C D F G) : A + B + C + D + F + G = 360 :=
  h


end sum_of_angles_is_360_l88_88808


namespace average_last_three_l88_88198

/-- The average of the last three numbers is 65, given that the average of six numbers is 60
  and the average of the first three numbers is 55. -/
theorem average_last_three (a b c d e f : ℝ) (h1 : (a + b + c + d + e + f) / 6 = 60) (h2 : (a + b + c) / 3 = 55) :
  (d + e + f) / 3 = 65 :=
by
  sorry

end average_last_three_l88_88198


namespace sphere_volume_in_cone_l88_88876

theorem sphere_volume_in_cone :
  let d := 24
  let theta := 90
  let r := 24 * (Real.sqrt 2 - 1)
  let V := (4 / 3) * Real.pi * r^3
  ∃ (R : ℝ), r = R ∧ V = (4 / 3) * Real.pi * R^3 := by
  sorry

end sphere_volume_in_cone_l88_88876


namespace dot_product_OA_OB_l88_88004

theorem dot_product_OA_OB :
  let A := (Real.cos 110, Real.sin 110)
  let B := (Real.cos 50, Real.sin 50)
  (A.1 * B.1 + A.2 * B.2) = 1 / 2 :=
by
  sorry

end dot_product_OA_OB_l88_88004


namespace jacoby_lottery_winning_l88_88531

theorem jacoby_lottery_winning :
  let total_needed := 5000
  let job_earning := 20 * 10
  let cookies_earning := 4 * 24
  let total_earnings_before_lottery := job_earning + cookies_earning
  let after_lottery := total_earnings_before_lottery - 10
  let gift_from_sisters := 500 * 2
  let total_earnings_and_gifts := after_lottery + gift_from_sisters
  let total_so_far := total_needed - 3214
  total_so_far - total_earnings_and_gifts = 500 :=
by
  sorry

end jacoby_lottery_winning_l88_88531


namespace probability_sum_greater_than_four_l88_88227

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l88_88227


namespace three_point_three_six_as_fraction_l88_88281

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l88_88281


namespace non_overlapping_squares_area_l88_88053

noncomputable def non_overlapping_area (a : ℝ) : ℝ :=
  2 * (1 - real.sqrt 3 / 3) * a^2

theorem non_overlapping_squares_area (a : ℝ) :
  let shaded_area := (real.sqrt 3 / 3) * a^2 in
  2 * (a^2 - shaded_area) = non_overlapping_area a :=
by
  -- This theorem asserts that the area of the non-overlapping parts of the two squares is
  -- equal to the computed formula for non_overlapping_area given a.
  sorry

end non_overlapping_squares_area_l88_88053


namespace probability_sum_greater_than_four_l88_88253

def sum_greater_than_four_probability :=
  (5 / 6: ℚ)

theorem probability_sum_greater_than_four :
  let outcomes := { (a, b) | a in (Finset.range 1 7) ∧ b in (Finset.range 1 7) }
  let favorable_outcomes := outcomes.filter (λ pair, (pair.1 + pair.2) > 4)
  let probability := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  probability = sum_greater_than_four_probability :=
by
  sorry

end probability_sum_greater_than_four_l88_88253


namespace sin_cos_identity_proof_l88_88766

noncomputable def solution : ℝ := Real.sin (Real.pi / 6) * Real.cos (Real.pi / 12) + Real.cos (Real.pi / 6) * Real.sin (Real.pi / 12)

theorem sin_cos_identity_proof : solution = Real.sqrt 2 / 2 := by
  sorry

end sin_cos_identity_proof_l88_88766


namespace prime_sum_probability_l88_88483

-- Definition of the problem conditions
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Definition of the probability calculation
def num_valid_pairs : ℕ := 3
def total_pairs : ℕ := 45
def probability_prime_sum_gt_10 : ℚ := num_valid_pairs / total_pairs

-- Problem statement in Lean
theorem prime_sum_probability : probability_prime_sum_gt_10 = 1 / 15 :=
by
  sorry

end prime_sum_probability_l88_88483


namespace Bruno_wants_2_5_dozens_l88_88474

theorem Bruno_wants_2_5_dozens (total_pens : ℕ) (dozen_pens : ℕ) (h_total_pens : total_pens = 30) (h_dozen_pens : dozen_pens = 12) : (total_pens / dozen_pens : ℚ) = 2.5 :=
by 
  sorry

end Bruno_wants_2_5_dozens_l88_88474


namespace steve_take_home_pay_l88_88404

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end steve_take_home_pay_l88_88404


namespace xyz_expression_l88_88086

theorem xyz_expression (x y z : ℝ) 
  (h1 : x^2 - y * z = 2)
  (h2 : y^2 - z * x = 2)
  (h3 : z^2 - x * y = 2) :
  x * y + y * z + z * x = -2 :=
sorry

end xyz_expression_l88_88086


namespace sum_first_19_terms_l88_88999

variable {α : Type} [LinearOrderedField α]

def arithmetic_sequence (a d : α) (n : ℕ) : α := a + n * d

def sum_of_arithmetic_sequence (a d : α) (n : ℕ) : α := (n : α) / 2 * (2 * a + (n - 1) * d)

theorem sum_first_19_terms (a d : α) 
  (h1 : ∀ n, arithmetic_sequence a d (2 + n) + arithmetic_sequence a d (16 + n) = 10)
  (S19 : α) :
  sum_of_arithmetic_sequence a d 19 = 95 := by
  sorry

end sum_first_19_terms_l88_88999


namespace coordinates_of_B_l88_88955

theorem coordinates_of_B (m : ℝ) (h : m + 2 = 0) : 
  (m + 5, m - 1) = (3, -3) :=
by
  -- proof goes here
  sorry

end coordinates_of_B_l88_88955


namespace birds_reduction_on_third_day_l88_88855

theorem birds_reduction_on_third_day
  {a b c : ℕ} 
  (h1 : a = 300)
  (h2 : b = 2 * a)
  (h3 : c = 1300)
  : (b - (c - (a + b))) = 200 :=
by sorry

end birds_reduction_on_third_day_l88_88855


namespace sequence_general_term_l88_88994

theorem sequence_general_term (a : ℕ → ℤ) : 
  (∀ n, a n = (-1)^(n + 1) * (3 * n - 2)) ↔ 
  (a 1 = 1 ∧ a 2 = -4 ∧ a 3 = 7 ∧ a 4 = -10 ∧ a 5 = 13) :=
by
  sorry

end sequence_general_term_l88_88994


namespace scientific_notation_periodicals_l88_88306

theorem scientific_notation_periodicals :
  (56000000 : ℝ) = 5.6 * 10^7 := by
sorry

end scientific_notation_periodicals_l88_88306


namespace number_of_throwers_l88_88545

theorem number_of_throwers (T N : ℕ) :
  (T + N = 61) ∧ ((2 * N) / 3 = 53 - T) → T = 37 :=
by 
  sorry

end number_of_throwers_l88_88545


namespace radius_increase_50_percent_l88_88951

theorem radius_increase_50_percent 
  (r : ℝ)
  (h1 : 1.5 * r = r + r * 0.5) : 
  (3 * Real.pi * r = 2 * Real.pi * r + (2 * Real.pi * r * 0.5)) ∧
  (2.25 * Real.pi * r^2 = Real.pi * r^2 + (Real.pi * r^2 * 1.25)) := 
sorry

end radius_increase_50_percent_l88_88951


namespace game_points_product_l88_88598

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def calculate_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_points_product :
  calculate_points allie_rolls * calculate_points betty_rolls = 702 :=
by
  sorry

end game_points_product_l88_88598


namespace polygon_interior_angle_l88_88793

theorem polygon_interior_angle (n : ℕ) (h1 : ∀ (i : ℕ), i < n → (n - 2) * 180 / n = 140): n = 9 := 
sorry

end polygon_interior_angle_l88_88793


namespace gcd_factorial_8_6_squared_l88_88102

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l88_88102


namespace polynomial_function_correct_l88_88991

theorem polynomial_function_correct :
  ∀ (f : ℝ → ℝ),
  (∀ (x : ℝ), f (x^2 + 1) = x^4 + 5 * x^2 + 3) →
  ∀ (x : ℝ), f (x^2 - 1) = x^4 + x^2 - 3 :=
by
  sorry

end polynomial_function_correct_l88_88991


namespace man_speed_l88_88451

theorem man_speed (time_in_minutes : ℕ) (distance_in_km : ℕ) 
  (h_time : time_in_minutes = 30) 
  (h_distance : distance_in_km = 5) : 
  (distance_in_km : ℝ) / (time_in_minutes / 60 : ℝ) = 10 :=
by 
  sorry

end man_speed_l88_88451


namespace percent_y_of_x_l88_88727

-- Definitions and assumptions based on the problem conditions
variables (x y : ℝ)
-- Given: 20% of (x - y) = 14% of (x + y)
axiom h : 0.20 * (x - y) = 0.14 * (x + y)

-- Prove that y is 0.1765 (or 17.65%) of x
theorem percent_y_of_x (x y : ℝ) (h : 0.20 * (x - y) = 0.14 * (x + y)) : 
  y = 0.1765 * x :=
sorry

end percent_y_of_x_l88_88727


namespace James_average_speed_l88_88532

theorem James_average_speed (TotalDistance : ℝ) (BreakTime : ℝ) (TotalTripTime : ℝ) (h1 : TotalDistance = 42) (h2 : BreakTime = 1) (h3 : TotalTripTime = 9) :
  (TotalDistance / (TotalTripTime - BreakTime)) = 5.25 :=
by
  sorry

end James_average_speed_l88_88532


namespace rearrange_squares_into_one_square_l88_88640

theorem rearrange_squares_into_one_square 
  (a b : ℕ) (h_a : a = 3) (h_b : b = 1) 
  (parts : Finset (ℕ × ℕ)) 
  (h_parts1 : parts.card ≤ 3)
  (h_parts2 : ∀ p ∈ parts, p.1 * p.2 = a * a ∨ p.1 * p.2 = b * b)
  : ∃ c : ℕ, (c * c = (a * a) + (b * b)) :=
by
  sorry

end rearrange_squares_into_one_square_l88_88640


namespace probability_correct_l88_88873

namespace ProbabilitySongs

/-- Define the total number of ways to choose 2 out of 4 songs -/ 
def total_ways : ℕ := Nat.choose 4 2

/-- Define the number of ways to choose 2 songs such that neither A nor B is chosen (only C and D can be chosen) -/
def ways_without_AB : ℕ := Nat.choose 2 2

/-- The probability of playing at least one of A and B is calculated via the complementary rule -/
def probability_at_least_one_AB_played : ℚ := 1 - (ways_without_AB / total_ways)

theorem probability_correct : probability_at_least_one_AB_played = 5 / 6 := sorry
end ProbabilitySongs

end probability_correct_l88_88873


namespace shorter_side_length_l88_88513

theorem shorter_side_length 
  (L W : ℝ) 
  (h1 : L * W = 117) 
  (h2 : 2 * L + 2 * W = 44) :
  L = 9 ∨ W = 9 :=
by
  sorry

end shorter_side_length_l88_88513


namespace solution_I_solution_II_l88_88866

noncomputable def problem_I : Prop :=
  let white_ball_prob := (2 : ℝ) / 5
  let black_ball_prob := (3 : ℝ) / 5
  let diff_color_prob := white_ball_prob * black_ball_prob + black_ball_prob * white_ball_prob
  diff_color_prob = 12 / 25

noncomputable def problem_II : Prop :=
  let P_xi_0 := (3 : ℝ) / 5 * (2 / 4)
  let P_xi_1 := (3 / 5) * (2 / 4) + (2 / 5) * (3 / 4)
  let P_xi_2 := (2 / 5) * (1 / 4)
  let E_xi := 0 * P_xi_0 + 1 * P_xi_1 + 2 * P_xi_2
  let Var_xi := (0 - E_xi) ^ 2 * P_xi_0 + (1 - E_xi) ^ 2 * P_xi_1 + (2 - E_xi) ^ 2 * P_xi_2
  E_xi = 4 / 5 ∧ Var_xi = 9 / 25

theorem solution_I : problem_I := 
by sorry

theorem solution_II : problem_II := 
by sorry

end solution_I_solution_II_l88_88866


namespace expression_value_l88_88077

theorem expression_value :
  3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3 :=
by
  sorry

end expression_value_l88_88077


namespace side_length_of_square_IJKL_l88_88769

theorem side_length_of_square_IJKL 
  (x y : ℝ) (hypotenuse : ℝ) 
  (h1 : x - y = 3) 
  (h2 : x + y = 9) 
  (h3 : hypotenuse = Real.sqrt (x^2 + y^2)) : 
  hypotenuse = 3 * Real.sqrt 5 :=
by
  sorry

end side_length_of_square_IJKL_l88_88769


namespace find_m_for_line_passing_through_circle_center_l88_88108

theorem find_m_for_line_passing_through_circle_center :
  ∀ (m : ℝ), (∀ (x y : ℝ), 2 * x + y + m = 0 ↔ (x - 1)^2 + (y + 2)^2 = 5) → m = 0 :=
by
  intro m
  intro h
  -- Here we construct that the center (1, -2) must lie on the line 2x + y + m = 0
  -- using the given condition of the circle center.
  have center := h 1 (-2)
  -- solving for the equation at the point (1, -2) must yield m = 0
  sorry

end find_m_for_line_passing_through_circle_center_l88_88108


namespace students_drawn_from_class_A_l88_88361

-- Given conditions
def classA_students : Nat := 40
def classB_students : Nat := 50
def total_sample : Nat := 18

-- Predicate that checks if the number of students drawn from Class A is correct
theorem students_drawn_from_class_A (students_from_A : Nat) : students_from_A = 9 :=
by
  sorry

end students_drawn_from_class_A_l88_88361


namespace factorization_problem1_factorization_problem2_l88_88907

-- Define the first problem: Factorization of 3x^2 - 27
theorem factorization_problem1 (x : ℝ) : 3 * x^2 - 27 = 3 * (x + 3) * (x - 3) :=
by
  sorry 

-- Define the second problem: Factorization of (a + 1)(a - 5) + 9
theorem factorization_problem2 (a : ℝ) : (a + 1) * (a - 5) + 9 = (a - 2) ^ 2 :=
by
  sorry

end factorization_problem1_factorization_problem2_l88_88907


namespace probability_sum_greater_than_four_l88_88245

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88245


namespace farm_problem_l88_88568

theorem farm_problem (D C : ℕ) (h1 : D + C = 15) (h2 : 2 * D + 4 * C = 42) : C = 6 :=
sorry

end farm_problem_l88_88568


namespace find_smallest_d_l88_88620

theorem find_smallest_d (d : ℕ) : (5 + 6 + 2 + 4 + 8 + d) % 9 = 0 → d = 2 :=
by
  sorry

end find_smallest_d_l88_88620


namespace customer_payment_l88_88040

noncomputable def cost_price : ℝ := 4090.9090909090905
noncomputable def markup : ℝ := 0.32
noncomputable def selling_price : ℝ := cost_price * (1 + markup)

theorem customer_payment :
  selling_price = 5400 :=
by
  unfold selling_price
  unfold cost_price
  unfold markup
  sorry

end customer_payment_l88_88040


namespace correct_solution_l88_88262

variable (x y : ℤ) (a b : ℤ) (h1 : 2 * x + a * y = 6) (h2 : b * x - 7 * y = 16)

theorem correct_solution : 
  (∃ x y : ℤ, 2 * x - 3 * y = 6 ∧ 5 * x - 7 * y = 16 ∧ x = 6 ∧ y = 2) :=
by
  use 6, 2
  constructor
  · exact sorry -- 2 * 6 - 3 * 2 = 6
  constructor
  · exact sorry -- 5 * 6 - 7 * 2 = 16
  constructor
  · exact rfl
  · exact rfl

end correct_solution_l88_88262


namespace y_intercept_of_linear_function_l88_88840

theorem y_intercept_of_linear_function 
  (k : ℝ)
  (h : (∃ k: ℝ, ∀ x y: ℝ, y = k * (x - 1) ∧ (x, y) = (-1, -2))) : 
  ∃ y : ℝ, (0, y) = (0, -1) :=
by {
  -- Skipping the proof as per the instruction
  sorry
}

end y_intercept_of_linear_function_l88_88840


namespace calculate_lives_lost_l88_88856

-- Define the initial number of lives
def initial_lives : ℕ := 98

-- Define the remaining number of lives
def remaining_lives : ℕ := 73

-- Define the number of lives lost
def lives_lost : ℕ := initial_lives - remaining_lives

-- Prove that Kaleb lost 25 lives
theorem calculate_lives_lost : lives_lost = 25 := 
by {
  -- The proof would go here, but we'll skip it
  sorry
}

end calculate_lives_lost_l88_88856


namespace two_dice_sum_greater_than_four_l88_88240
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l88_88240


namespace gcd_factorial_l88_88100

theorem gcd_factorial (a b : ℕ) : 
    ∃ (g : ℕ), nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2) = g ∧ g = 5760 := 
by 
  let g := nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2)
  existsi (5760 : ℕ)
  split
  · sorry
  · rfl

end gcd_factorial_l88_88100


namespace angle_bisector_b_c_sum_l88_88884

theorem angle_bisector_b_c_sum (A B C : ℝ × ℝ)
  (hA : A = (4, -3))
  (hB : B = (-6, 21))
  (hC : C = (10, 7)) :
  ∃ b c : ℝ, (3 * x + b * y + c = 0) ∧ (b + c = correct_answer) :=
by
  sorry

end angle_bisector_b_c_sum_l88_88884


namespace michael_bought_crates_on_thursday_l88_88680

theorem michael_bought_crates_on_thursday :
  ∀ (eggs_per_crate crates_tuesday crates_given current_eggs bought_on_thursday : ℕ),
    crates_tuesday = 6 →
    crates_given = 2 →
    eggs_per_crate = 30 →
    current_eggs = 270 →
    bought_on_thursday = (current_eggs - (crates_tuesday * eggs_per_crate - crates_given * eggs_per_crate)) / eggs_per_crate →
    bought_on_thursday = 5 :=
by
  intros _ _ _ _ _
  sorry

end michael_bought_crates_on_thursday_l88_88680


namespace cylinder_volume_l88_88428

noncomputable def volume_cylinder (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) : ℝ :=
  let ratio_r := r_cylinder / r_cone
  let ratio_h := h_cylinder / h_cone
  (3 : ℝ) * ratio_r^2 * ratio_h * V_cone

theorem cylinder_volume (V_cone : ℝ) (r_cylinder r_cone h_cylinder h_cone : ℝ) :
    r_cylinder / r_cone = 2 / 3 →
    h_cylinder / h_cone = 4 / 3 →
    V_cone = 5.4 →
    volume_cylinder V_cone r_cylinder r_cone h_cylinder h_cone = 3.2 :=
by
  intros h1 h2 h3
  rw [volume_cylinder, h1, h2, h3]
  sorry

end cylinder_volume_l88_88428


namespace simplify_and_evaluate_division_l88_88398

theorem simplify_and_evaluate_division (m : ℕ) (h : m = 10) : 
  (1 - (m / (m + 2))) / ((m^2 - 4 * m + 4) / (m^2 - 4)) = 1 / 4 :=
by sorry

end simplify_and_evaluate_division_l88_88398


namespace inscribed_circle_radius_l88_88150

-- Define the given conditions
def radius_large : ℝ := 18
def radius_small : ℝ := 9
def radius_inscribed : ℝ := 8

-- Define tangency conditions and relationships based on the problem statement
def large_semicircle (R : ℝ) := { x : ℝ // 0 <= x ∧ x <= R }
def small_semicircle (r : ℝ) := { x : ℝ // 0 <= x ∧ x <= r }

-- Prove the radius of the circle inscribed between the two semicircles
theorem inscribed_circle_radius :
  large_semicircle radius_large ∧ small_semicircle radius_small →
  ∃ (x : ℝ), x = radius_inscribed := 
by
  intro h;  -- Assume the hypothesis h
  exists radius_inscribed;  -- Show the existence of the radius of the inscribed circle
  have hp1 : sqrt (324 - 36 * radius_inscribed) + radius_small = sqrt (81 + 18 * radius_inscribed) := sorry,
  have hp2 : sqrt (324 - 36 * radius_inscribed) = sqrt (81 + 18 * radius_inscribed) - 9 := sorry,
  have h_sqr : (324 - 36 * radius_inscribed) = (sqrt (81 + 18 * radius_inscribed) - 9)^2 := sorry,
  sorry  -- Proof skipped for simplicity of problem setup

end inscribed_circle_radius_l88_88150


namespace num_consecutive_sets_summing_to_90_l88_88562

-- Define the arithmetic sequence sum properties
theorem num_consecutive_sets_summing_to_90 : 
  ∃ n : ℕ, n ≥ 2 ∧
    ∃ (a : ℕ), 2 * a + n - 1 = 180 / n ∧
      (∃ k : ℕ, 
         k ≥ 2 ∧
         ∃ b : ℕ, 2 * b + k - 1 = 180 / k) ∧
      (∃ m : ℕ, 
         m ≥ 2 ∧ 
         ∃ c : ℕ, 2 * c + m - 1 = 180 / m) ∧
      (n = 3 ∨ n = 5 ∨ n = 9) :=
sorry

end num_consecutive_sets_summing_to_90_l88_88562


namespace complex_number_solution_l88_88785

theorem complex_number_solution {i z : ℂ} (h : (2 : ℂ) / (1 + i) = z + i) : z = 1 + 2 * i :=
sorry

end complex_number_solution_l88_88785


namespace probability_sum_greater_than_four_l88_88246

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88246


namespace quadratic_equal_roots_l88_88948

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ (x * (x + 1) + a * x = 0) ∧ ((1 + a)^2 = 0)) →
  a = -1 :=
by
  sorry

end quadratic_equal_roots_l88_88948


namespace work_day_meeting_percent_l88_88542

open Nat

theorem work_day_meeting_percent :
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 35 := 
by
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  sorry

end work_day_meeting_percent_l88_88542


namespace total_pints_l88_88902

variables (Annie Kathryn Ben Sam : ℕ)

-- Conditions
def condition1 := Annie = 16
def condition2 (Annie : ℕ) := Kathryn = 2 * Annie + 2
def condition3 (Kathryn : ℕ) := Ben = Kathryn / 2 - 3
def condition4 (Ben Kathryn : ℕ) := Sam = 2 * (Ben + Kathryn) / 3

-- Statement to prove
theorem total_pints (Annie Kathryn Ben Sam : ℕ) 
  (h1 : condition1 Annie) 
  (h2 : condition2 Annie Kathryn) 
  (h3 : condition3 Kathryn Ben) 
  (h4 : condition4 Ben Kathryn Sam) : 
  Annie + Kathryn + Ben + Sam = 96 :=
sorry

end total_pints_l88_88902


namespace bruce_and_anne_clean_together_l88_88462

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l88_88462


namespace calculate_expression_value_l88_88607

theorem calculate_expression_value (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 8) :
  (7 * x + 5 * y) / (70 * x * y) = 57 / 400 := by
  sorry

end calculate_expression_value_l88_88607


namespace max_months_with_5_sundays_l88_88590

theorem max_months_with_5_sundays (months : ℕ) (days_in_year : ℕ) (extra_sundays : ℕ) :
  months = 12 ∧ (days_in_year = 365 ∨ days_in_year = 366) ∧ extra_sundays = days_in_year % 7
  → ∃ max_months_with_5_sundays, max_months_with_5_sundays = 5 := 
by
  sorry

end max_months_with_5_sundays_l88_88590


namespace anthony_more_shoes_than_jim_l88_88978

def scott_shoes : ℕ := 7
def anthony_shoes : ℕ := 3 * scott_shoes
def jim_shoes : ℕ := anthony_shoes - 2

theorem anthony_more_shoes_than_jim : (anthony_shoes - jim_shoes) = 2 :=
by
  sorry

end anthony_more_shoes_than_jim_l88_88978


namespace cubic_intersection_unique_point_l88_88783

-- Define the cubic functions f and g
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d
def g (a b c d x : ℝ) : ℝ := -a * x^3 + b * x^2 - c * x + d

-- Translate conditions into Lean conditions
variables (a b c d : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Lean statement to prove the intersection point
theorem cubic_intersection_unique_point :
  ∀ x y : ℝ, (f a b c d x = y) ↔ (g a b c d x = y) → (x = 0 ∧ y = d) :=
by
  -- Mathematical steps would go here (omitted with sorry)
  sorry

end cubic_intersection_unique_point_l88_88783


namespace find_x_l88_88332

theorem find_x (x : ℝ) : 
  (∀ (y : ℝ), 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := sorry

end find_x_l88_88332


namespace unique_ordered_triple_l88_88597

theorem unique_ordered_triple (a b c : ℕ) (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_eq : a^3 + b^3 + c^3 + 648 = (a + b + c)^3) :
  (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) ∨ (a, b, c) = (3, 3, 3) :=
sorry

end unique_ordered_triple_l88_88597


namespace yvettes_final_bill_l88_88666

theorem yvettes_final_bill :
  let alicia : ℝ := 7.5
  let brant : ℝ := 10
  let josh : ℝ := 8.5
  let yvette : ℝ := 9
  let tip_percentage : ℝ := 0.2
  ∃ final_bill : ℝ, final_bill = (alicia + brant + josh + yvette) * (1 + tip_percentage) ∧ final_bill = 42 :=
by
  sorry

end yvettes_final_bill_l88_88666


namespace solution_set_x_plus_3_f_x_plus_4_l88_88780

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Given conditions
axiom even_f_x_plus_1 : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom deriv_negative_f : ∀ x : ℝ, x > 1 → f' x < 0
axiom f_at_4_equals_zero : f 4 = 0

-- To prove
theorem solution_set_x_plus_3_f_x_plus_4 :
  {x : ℝ | (x + 3) * f (x + 4) < 0} = {x : ℝ | -6 < x ∧ x < -3} ∪ {x : ℝ | x > 0} := sorry

end solution_set_x_plus_3_f_x_plus_4_l88_88780


namespace am_gm_inequality_l88_88676

theorem am_gm_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a^3 + b^3 + a + b ≥ 4 * a * b :=
by
  sorry

end am_gm_inequality_l88_88676


namespace inequality_proof_l88_88374

theorem inequality_proof
  (a b c d : ℝ)
  (hpos: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (hcond: (a + b) * (b + c) * (c + d) * (d + a) = 1) :
  (2 * a + b + c) * (2 * b + c + d) * (2 * c + d + a) * (2 * d + a + b) * (a * b * c * d) ^ 2 ≤ 1 / 16 := 
by
  sorry

end inequality_proof_l88_88374


namespace juliet_age_l88_88810

theorem juliet_age
    (M J R : ℕ)
    (h1 : J = M + 3)
    (h2 : J = R - 2)
    (h3 : M + R = 19) : J = 10 := by
  sorry

end juliet_age_l88_88810


namespace cashier_correction_l88_88064

theorem cashier_correction (y : ℕ) :
  let quarter_value := 25
  let nickel_value := 5
  let penny_value := 1
  let dime_value := 10
  let quarters_as_nickels_value := y * (quarter_value - nickel_value)
  let pennies_as_dimes_value := y * (dime_value - penny_value)
  let total_correction := quarters_as_nickels_value - pennies_as_dimes_value
  total_correction = 11 * y := by
  sorry

end cashier_correction_l88_88064


namespace no_nat_x_y_square_l88_88313

theorem no_nat_x_y_square (x y : ℕ) : ¬(∃ a b : ℕ, x^2 + y = a^2 ∧ y^2 + x = b^2) := 
by 
  sorry

end no_nat_x_y_square_l88_88313


namespace sphere_radius_equals_three_l88_88649

noncomputable def radius_of_sphere : ℝ := 3

theorem sphere_radius_equals_three {R : ℝ} (h1 : 4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) : 
  R = radius_of_sphere :=
by
  sorry

end sphere_radius_equals_three_l88_88649


namespace three_point_three_six_as_fraction_l88_88282

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l88_88282


namespace arithmetic_progression_rth_term_l88_88918

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 2 * n + 3 * n^2

theorem arithmetic_progression_rth_term : (S r) - (S (r - 1)) = 6 * r - 1 :=
by
  sorry

end arithmetic_progression_rth_term_l88_88918


namespace average_of_last_three_numbers_l88_88192

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l88_88192


namespace probability_of_one_of_each_color_l88_88110

-- Definitions based on the conditions
def total_marbles : ℕ := 12
def marbles_of_each_color : ℕ := 3
def number_of_selected_marbles : ℕ := 4

-- Calculation based on problem requirements
def total_ways_to_choose_marbles : ℕ := Nat.choose total_marbles number_of_selected_marbles
def favorable_ways_to_choose : ℕ := marbles_of_each_color ^ number_of_selected_marbles

-- The main theorem to prove the probability
theorem probability_of_one_of_each_color :
  (favorable_ways_to_choose : ℚ) / total_ways_to_choose = 9 / 55 := by
  sorry

end probability_of_one_of_each_color_l88_88110


namespace hyperbola_parameters_sum_l88_88953

theorem hyperbola_parameters_sum :
  ∃ (h k a b : ℝ), 
    (h = 2 ∧ k = 0 ∧ a = 3 ∧ b = 3 * Real.sqrt 3) ∧
    h + k + a + b = 3 * Real.sqrt 3 + 5 := by
  sorry

end hyperbola_parameters_sum_l88_88953


namespace find_f_a_plus_1_l88_88635

def f (x : ℝ) : ℝ := x^2 + 1

theorem find_f_a_plus_1 (a : ℝ) : f (a + 1) = a^2 + 2 * a + 2 := by
  sorry

end find_f_a_plus_1_l88_88635


namespace probability_sum_greater_than_four_l88_88261

theorem probability_sum_greater_than_four : 
  let num_dice := 2
  let sides_per_die := 6
  let favorable_outcomes := { (a, b) | a > 0 ∧ a ≤ sides_per_die ∧ b > 0 ∧ b ≤ sides_per_die ∧ a + b > 4 }
  let total_outcomes := sides_per_die * sides_per_die
  let probability := (favorable_outcomes.card : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88261


namespace propositions_correctness_l88_88537

variable {a b c d : ℝ}

theorem propositions_correctness (h0 : a > b) (h1 : c > d) (h2 : c > 0) :
  (a > b ∧ c > d → a + c > b + d) ∧ 
  (a > b ∧ c > d → ¬(a - c > b - d)) ∧ 
  (a > b ∧ c > d → ¬(a * c > b * d)) ∧ 
  (a > b ∧ c > 0 → a * c > b * c) :=
by
  sorry

end propositions_correctness_l88_88537


namespace eggs_in_fridge_l88_88561

theorem eggs_in_fridge (total_eggs : ℕ) (eggs_per_cake : ℕ) (num_cakes : ℕ) (eggs_used : ℕ) (eggs_in_fridge : ℕ)
  (h1 : total_eggs = 60)
  (h2 : eggs_per_cake = 5)
  (h3 : num_cakes = 10)
  (h4 : eggs_used = eggs_per_cake * num_cakes)
  (h5 : eggs_in_fridge = total_eggs - eggs_used) :
  eggs_in_fridge = 10 :=
by
  sorry

end eggs_in_fridge_l88_88561


namespace total_odd_green_red_marbles_l88_88029

def Sara_green : ℕ := 3
def Sara_red : ℕ := 5
def Tom_green : ℕ := 4
def Tom_red : ℕ := 7
def Lisa_green : ℕ := 5
def Lisa_red : ℕ := 3

theorem total_odd_green_red_marbles : 
  (if Sara_green % 2 = 1 then Sara_green else 0) +
  (if Sara_red % 2 = 1 then Sara_red else 0) +
  (if Tom_green % 2 = 1 then Tom_green else 0) +
  (if Tom_red % 2 = 1 then Tom_red else 0) +
  (if Lisa_green % 2 = 1 then Lisa_green else 0) +
  (if Lisa_red % 2 = 1 then Lisa_red else 0) = 23 := by
  sorry

end total_odd_green_red_marbles_l88_88029


namespace bug_crawl_distance_l88_88583

theorem bug_crawl_distance : 
  let start : ℤ := 3
  let first_stop : ℤ := -4
  let second_stop : ℤ := 7
  let final_stop : ℤ := -1
  |first_stop - start| + |second_stop - first_stop| + |final_stop - second_stop| = 26 := 
by
  sorry

end bug_crawl_distance_l88_88583


namespace fraction_of_work_left_l88_88445

theorem fraction_of_work_left 
  (A_days : ℕ) (B_days : ℕ) (work_days : ℕ) 
  (A_rate : ℚ := 1 / A_days) (B_rate : ℚ := 1 / B_days) (combined_rate : ℚ := 1 / A_days + 1 / B_days) 
  (work_completed : ℚ := combined_rate * work_days) (fraction_left : ℚ := 1 - work_completed)
  (hA : A_days = 15) (hB : B_days = 20) (hW : work_days = 4) 
  : fraction_left = 8 / 15 :=
sorry

end fraction_of_work_left_l88_88445


namespace sum_of_eggs_is_3712_l88_88993

-- Definitions based on the conditions
def eggs_yesterday : ℕ := 1925
def eggs_fewer_today : ℕ := 138
def eggs_today : ℕ := eggs_yesterday - eggs_fewer_today

-- Theorem stating the equivalence of the sum of eggs
theorem sum_of_eggs_is_3712 : eggs_yesterday + eggs_today = 3712 :=
by
  sorry

end sum_of_eggs_is_3712_l88_88993


namespace general_term_formula_l88_88418

theorem general_term_formula (n : ℕ) :
  ∀ (S : ℕ → ℝ), (∀ k : ℕ, S k = 1 - 2^k) → 
  (∀ a : ℕ → ℝ, a 1 = (S 1) ∧ (∀ m : ℕ, m > 1 → a m = S m - S (m - 1)) → 
  a n = -2 ^ (n - 1)) :=
by
  intro S hS a ha
  sorry

end general_term_formula_l88_88418


namespace length_of_X_l88_88147

theorem length_of_X
  {X : ℝ}
  (h1 : 2 + 2 + X = 4 + X)
  (h2 : 3 + 4 + 1 = 8)
  (h3 : ∃ y : ℝ, y * (4 + X) = 29) : 
  X = 4 := sorry

end length_of_X_l88_88147


namespace order_of_a_b_c_l88_88773

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := (1 / 2) * (Real.log 5 / Real.log 2)

theorem order_of_a_b_c : a > c ∧ c > b :=
by
  -- proof here
  sorry

end order_of_a_b_c_l88_88773


namespace product_of_t_l88_88105

theorem product_of_t (a b : ℤ) (t : ℤ) (h1 : a * b = -12) (h2 : t = a + b) :
  ∃ (t_values : Finset ℤ), 
  (∀ x ∈ t_values, ∃ a b : ℤ, a * b = -12 ∧ x = a + b) ∧ 
  (t_values.product = -1936) :=
by
  sorry

end product_of_t_l88_88105


namespace train_lengths_combined_l88_88716

noncomputable def speed_to_mps (kmph : ℤ) : ℚ := (kmph : ℚ) * 5 / 18

def length_of_train (speed : ℚ) (time : ℚ) : ℚ := speed * time

theorem train_lengths_combined :
  let speed1_kmph := 100
  let speed2_kmph := 120
  let time1_sec := 9
  let time2_sec := 8
  let speed1_mps := speed_to_mps speed1_kmph
  let speed2_mps := speed_to_mps speed2_kmph
  let length1 := length_of_train speed1_mps time1_sec
  let length2 := length_of_train speed2_mps time2_sec
  length1 + length2 = 516.66 :=
by
  sorry

end train_lengths_combined_l88_88716


namespace john_days_off_l88_88533

def streams_per_week (earnings_per_week : ℕ) (rate_per_hour : ℕ) : ℕ := earnings_per_week / rate_per_hour

def streaming_sessions (hours_per_week : ℕ) (hours_per_session : ℕ) : ℕ := hours_per_week / hours_per_session

def days_off_per_week (total_days : ℕ) (streaming_days : ℕ) : ℕ := total_days - streaming_days

theorem john_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (total_days : ℕ) :
  hours_per_session = 4 → 
  hourly_rate = 10 → 
  weekly_earnings = 160 → 
  total_days = 7 → 
  days_off_per_week total_days (streaming_sessions (streams_per_week weekly_earnings hourly_rate) hours_per_session) = 3 := 
by
  intros
  sorry

end john_days_off_l88_88533


namespace find_sum_of_xyz_l88_88692

theorem find_sum_of_xyz (x y z : ℕ) (h1 : 0 < x ∧ 0 < y ∧ 0 < z)
  (h2 : (x + y + z)^3 - x^3 - y^3 - z^3 = 300) : x + y + z = 7 :=
by
  sorry

end find_sum_of_xyz_l88_88692


namespace sum_first_4_terms_l88_88123

-- Define the sequence and its properties
def a (n : ℕ) : ℝ := sorry   -- The actual definition will be derived based on n, a_1, and q
def S (n : ℕ) : ℝ := sorry   -- The sum of the first n terms, also will be derived

-- Define the initial sequence properties based on the given conditions
axiom h1 : 0 < a 1  -- The sequence is positive
axiom h2 : a 4 * a 6 = 1 / 4
axiom h3 : a 7 = 1 / 8

-- The goal is to prove the sum of the first 4 terms equals 15
theorem sum_first_4_terms : S 4 = 15 := by
  sorry

end sum_first_4_terms_l88_88123


namespace sum_of_abcd_l88_88811

variable (a b c d : ℚ)

def condition (x : ℚ) : Prop :=
  x = a + 3 ∧
  x = b + 7 ∧
  x = c + 5 ∧
  x = d + 9 ∧
  x = a + b + c + d + 13

theorem sum_of_abcd (x : ℚ) (h : condition a b c d x) : a + b + c + d = -28 / 3 := 
by sorry

end sum_of_abcd_l88_88811


namespace noah_uses_36_cups_of_water_l88_88681

theorem noah_uses_36_cups_of_water
  (O : ℕ) (hO : O = 4)
  (S : ℕ) (hS : S = 3 * O)
  (W : ℕ) (hW : W = 3 * S) :
  W = 36 := 
  by sorry

end noah_uses_36_cups_of_water_l88_88681


namespace probability_max_roll_correct_l88_88746
open Classical

noncomputable def probability_max_roll_fourth : ℚ :=
  let six_sided_max := 1 / 6
  let eight_sided_max := 3 / 4
  let ten_sided_max := 4 / 5

  let prob_A_given_B1 := (1 / 6) ^ 3
  let prob_A_given_B2 := (3 / 4) ^ 3
  let prob_A_given_B3 := (4 / 5) ^ 3

  let prob_B1 := 1 / 3
  let prob_B2 := 1 / 3
  let prob_B3 := 1 / 3

  let prob_A := prob_A_given_B1 * prob_B1 + prob_A_given_B2 * prob_B2 + prob_A_given_B3 * prob_B3

  -- Calculate probabilities with Bayes' Theorem
  let P_B1_A := (prob_A_given_B1 * prob_B1) / prob_A
  let P_B2_A := (prob_A_given_B2 * prob_B2) / prob_A
  let P_B3_A := (prob_A_given_B3 * prob_B3) / prob_A

  -- Probability of the fourth roll showing the maximum face value
  P_B1_A * six_sided_max + P_B2_A * eight_sided_max + P_B3_A * ten_sided_max

theorem probability_max_roll_correct : 
  ∃ (p q : ℕ), probability_max_roll_fourth = p / q ∧ Nat.gcd p q = 1 ∧ p + q = 4386 :=
by sorry

end probability_max_roll_correct_l88_88746


namespace probability_of_A_l88_88202

variable (Ω : Type) [ProbabilitySpace Ω]
variables (A B : Event Ω)

theorem probability_of_A :
  independent A B →
  0 < P(A) →
  P(A) = 2 * P(B) →
  P(A ∪ B) = 8 * P(A ∩ B) →
  P(A) = 1 / 3 := by
sory

end probability_of_A_l88_88202


namespace sufficient_not_necessary_a_eq_one_l88_88722

noncomputable def f (a x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + a^2) - x)

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x + f (-x) = 0

theorem sufficient_not_necessary_a_eq_one 
  (a : ℝ) 
  (h₁ : a = 1) 
  : is_odd_function (f a) := sorry

end sufficient_not_necessary_a_eq_one_l88_88722


namespace sally_credit_card_balance_l88_88183

theorem sally_credit_card_balance (G P : ℝ) (X : ℝ)  
  (h1 : P = 2 * G)  
  (h2 : XP = X * P)  
  (h3 : G / 3 + XP = (5 / 12) * P) : 
  X = 1 / 4 :=
by
  sorry

end sally_credit_card_balance_l88_88183


namespace M_is_listed_correctly_l88_88131

noncomputable def M : Set ℕ := { m | ∃ n : ℕ+, 3 / (5 - m : ℝ) = n }

theorem M_is_listed_correctly : M = { 2, 4 } :=
by
  sorry

end M_is_listed_correctly_l88_88131


namespace num_friends_bought_robots_l88_88458

def robot_cost : Real := 8.75
def tax_charged : Real := 7.22
def change_left : Real := 11.53
def initial_amount : Real := 80.0
def friends_bought_robots : Nat := 7

theorem num_friends_bought_robots :
  (initial_amount - (change_left + tax_charged)) / robot_cost = friends_bought_robots := sorry

end num_friends_bought_robots_l88_88458


namespace probability_of_two_eights_l88_88323

-- Define a function that calculates the factorial of a number
noncomputable def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Definition of binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  fact n / (fact k * fact (n - k))

-- Probability of exactly two dice showing 8 out of eight 8-sided dice
noncomputable def prob_exactly_two_eights : ℚ :=
  binom 8 2 * ((1 / 8 : ℚ) ^ 2) * ((7 / 8 : ℚ) ^ 6)

-- Main theorem statement
theorem probability_of_two_eights :
  prob_exactly_two_eights = 0.196 := by
  sorry

end probability_of_two_eights_l88_88323


namespace solution_set_of_inequality_l88_88842

theorem solution_set_of_inequality :
  {x : ℝ // (2 < x ∨ x < 2) ∧ x ≠ 3} =
  {x : ℝ // x < 2 ∨ 3 < x } :=
sorry

end solution_set_of_inequality_l88_88842


namespace vasechkin_result_l88_88685

theorem vasechkin_result (x : ℕ) (h : (x / 2 * 7) - 1001 = 7) : (x / 8) ^ 2 - 1001 = 295 :=
by
  sorry

end vasechkin_result_l88_88685


namespace harmonic_power_identity_l88_88026

open Real

theorem harmonic_power_identity (a b c : ℝ) (n : ℕ) (hn : n % 2 = 1) 
(h : (1 / a + 1 / b + 1 / c) = 1 / (a + b + c)) :
  (1 / (a ^ n) + 1 / (b ^ n) + 1 / (c ^ n) = 1 / (a ^ n + b ^ n + c ^ n)) :=
sorry

end harmonic_power_identity_l88_88026


namespace age_proof_l88_88589

theorem age_proof (M S Y : ℕ) (h1 : M = 36) (h2 : S = 12) (h3 : M = 3 * S) : 
  (M + Y = 2 * (S + Y)) ↔ (Y = 12) :=
by 
  sorry

end age_proof_l88_88589


namespace james_weekly_expenses_l88_88155

theorem james_weekly_expenses :
  let rent := 1200 in
  let utilities := rent * 0.20 in
  let hours_per_week_per_employee := 16 * 5 in
  let total_hours := 2 * hours_per_week_per_employee in
  let employee_wages := total_hours * 12.50 in
  let total_expenses := rent + utilities + employee_wages in
  total_expenses = 3440 :=
by
  let rent := 1200
  let utilities := rent * 0.20
  let hours_per_week_per_employee := 16 * 5
  let total_hours := 2 * hours_per_week_per_employee
  let employee_wages := total_hours * 12.50
  let total_expenses := rent + utilities + employee_wages
  sorry

end james_weekly_expenses_l88_88155


namespace two_dice_sum_greater_than_four_l88_88237
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l88_88237


namespace fraction_with_buddy_l88_88003

variable (s_6 n_9 : ℕ)

def sixth_graders_paired : ℚ := s_6 / 3
def ninth_graders_paired : ℚ := n_9 / 4

-- Given condition: 1/4 of ninth graders are paired with 1/3 of sixth graders
axiom pairing_condition : ninth_graders_paired = sixth_graders_paired

-- Prove that the fraction of the total number of sixth and ninth graders who have a buddy is 1/7
theorem fraction_with_buddy (h : pairing_condition s_6 n_9) :
  (sixth_graders_paired s_6 / (n_9 + s_6 : ℚ)) = 1 / 7 :=
  sorry

end fraction_with_buddy_l88_88003


namespace no_polygon_with_1974_diagonals_l88_88088

theorem no_polygon_with_1974_diagonals :
  ¬ ∃ N : ℕ, N * (N - 3) / 2 = 1974 :=
sorry

end no_polygon_with_1974_diagonals_l88_88088


namespace proof_problem_l88_88264

def a : ℕ := 5^2
def b : ℕ := a^4

theorem proof_problem : b = 390625 := 
by 
  sorry

end proof_problem_l88_88264


namespace average_remaining_two_numbers_l88_88835

theorem average_remaining_two_numbers 
  (h1 : (40.5 : ℝ) = 10 * 4.05)
  (h2 : (11.1 : ℝ) = 3 * 3.7)
  (h3 : (11.85 : ℝ) = 3 * 3.95)
  (h4 : (8.6 : ℝ) = 2 * 4.3)
  : (4.475 : ℝ) = (40.5 - (11.1 + 11.85 + 8.6)) / 2 := 
sorry

end average_remaining_two_numbers_l88_88835


namespace cube_root_simplification_l88_88987

theorem cube_root_simplification :
    ∀ (x y z : ℕ), x = 21952 → y = 1000 → z = 28^3 → (x * y) = 21952000 → real.pow (x * y) (1/3) = 280 :=
by
  intros x y z H1 H2 H3 H4
  sorry

end cube_root_simplification_l88_88987


namespace cookies_collected_total_is_276_l88_88454

noncomputable def number_of_cookies_in_one_box : ℕ := 48

def abigail_boxes : ℕ := 2
def grayson_boxes : ℕ := 3 / 4
def olivia_boxes : ℕ := 3

def total_cookies_collected : ℕ :=
  abigail_boxes * number_of_cookies_in_one_box + 
  (grayson_boxes * number_of_cookies_in_one_box) + 
  olivia_boxes * number_of_cookies_in_one_box

theorem cookies_collected_total_is_276 : total_cookies_collected = 276 := sorry

end cookies_collected_total_is_276_l88_88454


namespace probability_sum_greater_than_four_l88_88230

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let outcomes_sum_less_or_equal_4 := 6 in
  let prob_sum_less_or_equal_4 := outcomes_sum_less_or_equal_4 / total_outcomes in
  prob_sum_less_or_equal_4 = (1 : ℝ) / 6 → 
  (1 - prob_sum_less_or_equal_4) = (5 : ℝ) / 6 := 
by 
  intros total_outcomes outcomes_sum_less_or_equal_4 prob_sum_less_or_equal_4 h1,
  sorry

end probability_sum_greater_than_four_l88_88230


namespace solve_eq_solve_ineq_l88_88578

-- Proof Problem 1 statement
theorem solve_eq (x : ℝ) : (2 / (x + 3) - (x - 3) / (2 * x + 6) = 1) → (x = 1 / 3) :=
by sorry

-- Proof Problem 2 statement
theorem solve_ineq (x : ℝ) : (2 * x - 1 > 3 * (x - 1)) ∧ ((5 - x) / 2 < x + 4) → (-1 < x ∧ x < 2) :=
by sorry

end solve_eq_solve_ineq_l88_88578


namespace inequality_inequality_always_holds_l88_88136

theorem inequality_inequality_always_holds (x y : ℝ) (h : x > y) : |x| > y :=
sorry

end inequality_inequality_always_holds_l88_88136


namespace secretary_longest_time_l88_88695

def ratio_times (x : ℕ) : Prop := 
  let t1 := 2 * x
  let t2 := 3 * x
  let t3 := 5 * x
  (t1 + t2 + t3 = 110) ∧ (t3 = 55)

theorem secretary_longest_time :
  ∃ x : ℕ, ratio_times x :=
sorry

end secretary_longest_time_l88_88695


namespace percentage_decrease_of_b_l88_88415

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b)
  (h1 : a / b = 4 / 5)
  (h2 : x = a + 0.25 * a)
  (h3 : m = b * (1 - p / 100))
  (h4 : m / x = 0.4) :
  p = 60 :=
by
  sorry

end percentage_decrease_of_b_l88_88415


namespace num_sides_of_length4_eq_4_l88_88614

-- Definitions of the variables and conditions
def total_sides : ℕ := 6
def total_perimeter : ℕ := 30
def side_length1 : ℕ := 7
def side_length2 : ℕ := 4

-- The conditions imposed by the problem
def is_hexagon (x y : ℕ) : Prop := x + y = total_sides
def perimeter_condition (x y : ℕ) : Prop := side_length1 * x + side_length2 * y = total_perimeter

-- The proof problem: Prove that the number of sides of length 4 is 4
theorem num_sides_of_length4_eq_4 (x y : ℕ) 
    (h1 : is_hexagon x y) 
    (h2 : perimeter_condition x y) : y = 4 :=
sorry

end num_sides_of_length4_eq_4_l88_88614


namespace circle_tangent_lines_l88_88065

theorem circle_tangent_lines (h k : ℝ) (r : ℝ) (h_gt_10 : h > 10) (k_gt_10 : k > 10)
  (tangent_y_eq_10 : k - 10 = r)
  (tangent_y_eq_x : r = (|h - k| / Real.sqrt 2)) :
  (h, k) = (10 + (1 + Real.sqrt 2) * r, 10 + r) :=
by
  sorry

end circle_tangent_lines_l88_88065


namespace rectangle_width_l88_88002

-- The Lean statement only with given conditions and the final proof goal
theorem rectangle_width (w l : ℕ) (P : ℕ) (h1 : l = w - 3) (h2 : P = 2 * w + 2 * l) (h3 : P = 54) :
  w = 15 :=
by
  sorry

end rectangle_width_l88_88002


namespace birds_not_herons_are_geese_l88_88011

-- Define the given conditions
def percentage_geese : ℝ := 0.35
def percentage_swans : ℝ := 0.20
def percentage_herons : ℝ := 0.15
def percentage_ducks : ℝ := 0.30

-- Definition without herons
def percentage_non_herons : ℝ := 1 - percentage_herons

-- Theorem to prove
theorem birds_not_herons_are_geese :
  (percentage_geese / percentage_non_herons) * 100 = 41 :=
by
  sorry

end birds_not_herons_are_geese_l88_88011


namespace train_capacity_l88_88388

theorem train_capacity (T : ℝ) (h : 2 * (T / 6) = 40) : T = 120 :=
sorry

end train_capacity_l88_88388


namespace laps_needed_to_reach_total_distance_l88_88819

-- Define the known conditions
def total_distance : ℕ := 2400
def lap_length : ℕ := 150
def laps_run_each : ℕ := 6
def total_laps_run : ℕ := 2 * laps_run_each

-- Define the proof goal
theorem laps_needed_to_reach_total_distance :
  (total_distance - total_laps_run * lap_length) / lap_length = 4 :=
by
  sorry

end laps_needed_to_reach_total_distance_l88_88819


namespace marks_difference_l88_88422

variable (P C M : ℕ)

-- Conditions
def total_marks_more_than_physics := P + C + M > P
def average_chemistry_mathematics := (C + M) / 2 = 65

-- Proof Statement
theorem marks_difference (h1 : total_marks_more_than_physics P C M) (h2 : average_chemistry_mathematics C M) : 
  P + C + M = P + 130 := by
  sorry

end marks_difference_l88_88422


namespace empty_atm_l88_88674

theorem empty_atm (a : ℕ → ℕ) (b : ℕ → ℕ) (h1 : a 9 < b 9)
    (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ 8 → a k ≠ b k) 
    (n : ℕ) (h₀ : n = 1) : 
    ∃ (sequence : ℕ → ℕ), (∀ i, sequence i ≤ n) → (∀ k, ∃ i, k > i → sequence k = 0) :=
sorry

end empty_atm_l88_88674


namespace cheaper_price_difference_is_75_cents_l88_88079

noncomputable def list_price := 42.50
noncomputable def store_a_discount := 12.00
noncomputable def store_b_discount_percent := 0.30

noncomputable def store_a_price := list_price - store_a_discount
noncomputable def store_b_price := (1 - store_b_discount_percent) * list_price
noncomputable def price_difference_in_dollars := store_a_price - store_b_price
noncomputable def price_difference_in_cents := price_difference_in_dollars * 100

theorem cheaper_price_difference_is_75_cents :
  price_difference_in_cents = 75 := by
  sorry

end cheaper_price_difference_is_75_cents_l88_88079


namespace employed_males_percent_l88_88957

variable (population : ℝ) (percent_employed : ℝ) (percent_employed_females : ℝ)

theorem employed_males_percent :
  percent_employed = 120 →
  percent_employed_females = 33.33333333333333 →
  2 / 3 * percent_employed = 80 :=
by
  intros h1 h2
  sorry

end employed_males_percent_l88_88957


namespace Eval_trig_exp_l88_88906

theorem Eval_trig_exp :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end Eval_trig_exp_l88_88906


namespace john_fan_usage_per_day_l88_88009

theorem john_fan_usage_per_day
  (power : ℕ := 75) -- fan's power in watts
  (energy_per_month_kwh : ℕ := 18) -- energy consumption per month in kWh
  (days_in_month : ℕ := 30) -- number of days in a month
  : (energy_per_month_kwh * 1000) / power / days_in_month = 8 := 
by
  sorry

end john_fan_usage_per_day_l88_88009


namespace chang_total_apples_l88_88611

def sweet_apple_price : ℝ := 0.5
def sour_apple_price : ℝ := 0.1
def sweet_apple_percentage : ℝ := 0.75
def sour_apple_percentage : ℝ := 1 - sweet_apple_percentage
def total_earnings : ℝ := 40

theorem chang_total_apples : 
  (total_earnings / (sweet_apple_percentage * sweet_apple_price + sour_apple_percentage * sour_apple_price)) = 100 :=
by
  sorry

end chang_total_apples_l88_88611


namespace negation_of_proposition_l88_88930

theorem negation_of_proposition (p : Prop) : 
  (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0) ↔ ¬(∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end negation_of_proposition_l88_88930


namespace three_card_deal_probability_l88_88569

theorem three_card_deal_probability :
  (4 / 52) * (4 / 51) * (4 / 50) = 16 / 33150 := 
by 
  sorry

end three_card_deal_probability_l88_88569


namespace paint_replacement_l88_88188

theorem paint_replacement :
  ∀ (original_paint new_paint : ℝ), 
  original_paint = 100 →
  new_paint = 0.10 * (original_paint - 0.5 * original_paint) + 0.20 * (0.5 * original_paint) →
  new_paint / original_paint = 0.15 :=
by
  intros original_paint new_paint h_orig h_new
  sorry

end paint_replacement_l88_88188


namespace triangle_right_angle_l88_88661

theorem triangle_right_angle (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B - C) : B = 90 :=
by sorry

end triangle_right_angle_l88_88661


namespace total_spectators_l88_88713

-- Definitions of conditions
def num_men : Nat := 7000
def num_children : Nat := 2500
def num_women := num_children / 5

-- Theorem stating the total number of spectators
theorem total_spectators : (num_men + num_children + num_women) = 10000 := by
  sorry

end total_spectators_l88_88713


namespace max_receptivity_compare_receptivity_l88_88846

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if 10 < x ∧ x <= 16 then 59
  else if 16 < x ∧ x <= 30 then -3 * x + 107
  else 0 -- To cover the case when x is outside the given ranges

-- Problem 1
theorem max_receptivity :
  f 10 = 59 ∧ ∀ x, 10 < x ∧ x ≤ 16 → f x = 59 :=
by
  sorry

-- Problem 2
theorem compare_receptivity :
  f 5 > f 20 :=
by
  sorry

end max_receptivity_compare_receptivity_l88_88846


namespace inequality_generalization_l88_88630

theorem inequality_generalization (x : ℝ) (n : ℕ) (hn : 0 < n) (hx : 0 < x) :
  x + n^n / x^n ≥ n + 1 :=
sorry

end inequality_generalization_l88_88630


namespace alex_minus_sam_eq_negative_2_50_l88_88142

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.15
def packaging_fee : ℝ := 2.50

def alex_total (original_price tax_rate discount_rate : ℝ) : ℝ :=
  let price_with_tax := original_price * (1 + tax_rate)
  let final_price := price_with_tax * (1 - discount_rate)
  final_price

def sam_total (original_price tax_rate discount_rate packaging_fee : ℝ) : ℝ :=
  let price_with_discount := original_price * (1 - discount_rate)
  let price_with_tax := price_with_discount * (1 + tax_rate)
  let final_price := price_with_tax + packaging_fee
  final_price

theorem alex_minus_sam_eq_negative_2_50 :
  alex_total original_price tax_rate discount_rate - sam_total original_price tax_rate discount_rate packaging_fee = -2.50 := by
  sorry

end alex_minus_sam_eq_negative_2_50_l88_88142


namespace numberOfTermsArithmeticSequence_l88_88788

theorem numberOfTermsArithmeticSequence (a1 d l : ℕ) (h1 : a1 = 3) (h2 : d = 4) (h3 : l = 2012) :
  ∃ n : ℕ, 3 + (n - 1) * 4 ≤ 2012 ∧ (n : ℕ) = 502 :=
by {
  sorry
}

end numberOfTermsArithmeticSequence_l88_88788


namespace additional_cost_per_kg_l88_88309

theorem additional_cost_per_kg (l a : ℝ) 
  (h1 : 30 * l + 3 * a = 333) 
  (h2 : 30 * l + 6 * a = 366) 
  (h3 : 15 * l = 150) 
  : a = 11 := 
by
  sorry

end additional_cost_per_kg_l88_88309


namespace probability_sum_greater_than_four_l88_88222

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88222


namespace term_37_l88_88405

section GeometricSequence

variable {a b : ℕ → ℝ}
variable (q p : ℝ)

-- Definition of geometric sequences
def is_geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n ≥ 1, a (n + 1) = r * a n

-- Given conditions
axiom a1_25 : a 1 = 25
axiom b1_4 : b 1 = 4
axiom a2b2_100 : a 2 * b 2 = 100

-- Assume a and b are geometric sequences
axiom a_geom_seq : is_geometric_seq a q
axiom b_geom_seq : is_geometric_seq b p

-- Main theorem to prove
theorem term_37 (n : ℕ) (hn : n = 37) : (a n * b n) = 100 :=
sorry

end GeometricSequence

end term_37_l88_88405


namespace units_digit_of_product_of_seven_consecutive_integers_is_zero_l88_88479

/-- Define seven consecutive positive integers and show the units digit of their product is 0 -/
theorem units_digit_of_product_of_seven_consecutive_integers_is_zero (n : ℕ) :
  ∃ (k : ℕ), k = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 ∧ k = 0 :=
by {
  -- We state that the units digit k of the product of seven consecutive integers
  -- starting from n is 0
  sorry
}

end units_digit_of_product_of_seven_consecutive_integers_is_zero_l88_88479


namespace range_of_m_iff_l88_88639

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∀ (x y : ℝ), (0 < x) → (0 < y) → ((2 / x) + (1 / y) = 1) → (x + 2 * y > m^2 + 2 * m)

theorem range_of_m_iff : (range_of_m m) ↔ (-4 < m ∧ m < 2) :=
  sorry

end range_of_m_iff_l88_88639


namespace sin_120_eq_sqrt3_div_2_l88_88080

theorem sin_120_eq_sqrt3_div_2
  (h1 : 120 = 180 - 60)
  (h2 : ∀ θ, Real.sin (180 - θ) = Real.sin θ)
  (h3 : Real.sin 60 = Real.sqrt 3 / 2) :
  Real.sin 120 = Real.sqrt 3 / 2 :=
sorry

end sin_120_eq_sqrt3_div_2_l88_88080


namespace manager_to_employee_ratio_l88_88952

/-- In a certain company, the number of female managers is 300.
    The total number of female employees is 750.
    Prove that the ratio of female managers to all employees
    in the company is 2/5. -/
theorem manager_to_employee_ratio 
  (num_female_managers : ℕ) (total_female_employees : ℕ)
  (h1 : num_female_managers = 300)
  (h2 : total_female_employees = 750) :
  num_female_managers / total_female_employees = 2 / 5 :=
sorry

end manager_to_employee_ratio_l88_88952


namespace average_of_last_three_numbers_l88_88194

theorem average_of_last_three_numbers (nums : List ℝ) (h_len : nums.length = 6) 
    (h_avg6 : nums.sum / 6 = 60) (h_avg3 : (nums.take 3).sum / 3 = 55) : 
    ((nums.drop 3).sum) / 3 = 65 := 
sorry

end average_of_last_three_numbers_l88_88194


namespace abigail_writing_time_l88_88305

def total_additional_time (words_needed : ℕ) (words_per_half_hour : ℕ) (words_already_written : ℕ) (proofreading_time : ℕ) : ℕ :=
  let remaining_words := words_needed - words_already_written
  let half_hour_blocks := (remaining_words + words_per_half_hour - 1) / words_per_half_hour -- ceil(remaining_words / words_per_half_hour)
  let writing_time := half_hour_blocks * 30
  writing_time + proofreading_time

theorem abigail_writing_time :
  total_additional_time 1500 250 200 45 = 225 :=
by {
  -- Adding the proof in Lean:
  -- fail to show you the detailed steps, hence added sorry
  sorry
}

end abigail_writing_time_l88_88305


namespace compare_neg_fractions_l88_88316

theorem compare_neg_fractions : 
  (- (8:ℚ) / 21) > - (3 / 7) :=
by sorry

end compare_neg_fractions_l88_88316


namespace xyz_value_l88_88628

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (xy + xz + yz) = 40) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) 
  : x * y * z = 10 :=
sorry

end xyz_value_l88_88628


namespace PTA_money_left_l88_88189

theorem PTA_money_left (initial_savings : ℝ) (spent_on_supplies : ℝ) (spent_on_food : ℝ) :
  initial_savings = 400 →
  spent_on_supplies = initial_savings / 4 →
  spent_on_food = (initial_savings - spent_on_supplies) / 2 →
  (initial_savings - spent_on_supplies - spent_on_food) = 150 :=
by
  intro initial_savings_eq
  intro spent_on_supplies_eq
  intro spent_on_food_eq
  sorry

end PTA_money_left_l88_88189


namespace arithmetic_mean_of_fractions_l88_88087

def mean (a b : ℚ) : ℚ := (a + b) / 2

theorem arithmetic_mean_of_fractions (a b c : ℚ) (h₁ : a = 8/11)
                                      (h₂ : b = 5/6) (h₃ : c = 19/22) :
  mean a c = b :=
by
  sorry

end arithmetic_mean_of_fractions_l88_88087


namespace roots_sum_and_product_l88_88042

theorem roots_sum_and_product (p q : ℝ) (h_sum : p / 3 = 9) (h_prod : q / 3 = 24) : p + q = 99 :=
by
  -- We are given h_sum: p / 3 = 9
  -- We are given h_prod: q / 3 = 24
  -- We need to prove p + q = 99
  sorry

end roots_sum_and_product_l88_88042


namespace simplify_336_to_fraction_l88_88272

theorem simplify_336_to_fraction : (336 / 100) = (84 / 25) :=
by sorry

end simplify_336_to_fraction_l88_88272


namespace range_of_n_l88_88636

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.exp 1) * Real.exp x + (1 / 2) * x^2 - x

theorem range_of_n :
  (∃ m : ℝ, f m ≤ 2 * n^2 - n) ↔ (n ≤ -1/2 ∨ 1 ≤ n) :=
sorry

end range_of_n_l88_88636


namespace student_departments_l88_88220

variable {Student : Type}
variable (Anna Vika Masha : Student)

-- Let Department be an enumeration type representing the three departments
inductive Department
| Literature : Department
| History : Department
| Biology : Department

open Department

variables (isLit : Student → Prop) (isHist : Student → Prop) (isBio : Student → Prop)

-- Conditions
axiom cond1 : isLit Anna → ¬isHist Masha
axiom cond2 : ¬isHist Vika → isLit Anna
axiom cond3 : ¬isLit Masha → isBio Vika

-- Target conclusion
theorem student_departments :
  isHist Vika ∧ isLit Masha ∧ isBio Anna :=
sorry

end student_departments_l88_88220


namespace betty_paid_total_l88_88604

def cost_slippers (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_lipsticks (count : ℕ) (price : ℝ) : ℝ := count * price
def cost_hair_colors (count : ℕ) (price : ℝ) : ℝ := count * price

def total_cost := 
  cost_slippers 6 2.5 +
  cost_lipsticks 4 1.25 +
  cost_hair_colors 8 3

theorem betty_paid_total :
  total_cost = 44 := 
  sorry

end betty_paid_total_l88_88604


namespace gcd_factorial_l88_88099

theorem gcd_factorial (a b : ℕ) : 
    ∃ (g : ℕ), nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2) = g ∧ g = 5760 := 
by 
  let g := nat.gcd (nat.factorial 8) ((nat.factorial 6) ^ 2)
  existsi (5760 : ℕ)
  split
  · sorry
  · rfl

end gcd_factorial_l88_88099


namespace three_cards_probability_l88_88714

noncomputable def probability_first_king_second_queen_third_heart : ℚ :=
  (4 / 52) * (4 / 51) * (12 / 50)

theorem three_cards_probability :
  probability_first_king_second_queen_third_heart = 8 / 5525 := by
  sorry

end three_cards_probability_l88_88714


namespace average_of_last_three_numbers_l88_88195

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l88_88195


namespace bruce_anne_clean_in_4_hours_l88_88460

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l88_88460


namespace fewer_blue_than_green_l88_88881

-- Definitions for given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def total_buttons : ℕ := 275
def blue_buttons : ℕ := total_buttons - (green_buttons + yellow_buttons)

-- Theorem statement to be proved
theorem fewer_blue_than_green : green_buttons - blue_buttons = 5 :=
by
  -- Proof is omitted as per the instructions
  sorry

end fewer_blue_than_green_l88_88881


namespace find_alpha_angle_l88_88657

theorem find_alpha_angle :
  ∃ α : ℝ, (7 * α + 8 * α + 45) = 180 ∧ α = 9 :=
by 
  sorry

end find_alpha_angle_l88_88657


namespace line_intersects_circle_l88_88781

theorem line_intersects_circle (r d : ℝ) (hr : r = 5) (hd : d = 3 * Real.sqrt 2) : d < r :=
by
  rw [hr, hd]
  exact sorry

end line_intersects_circle_l88_88781


namespace book_set_cost_l88_88672

theorem book_set_cost (charge_per_sqft : ℝ) (lawn_length lawn_width : ℝ) (num_lawns : ℝ) (additional_area : ℝ) (total_cost : ℝ) :
  charge_per_sqft = 0.10 ∧ lawn_length = 20 ∧ lawn_width = 15 ∧ num_lawns = 3 ∧ additional_area = 600 ∧ total_cost = 150 →
  (num_lawns * (lawn_length * lawn_width) * charge_per_sqft + additional_area * charge_per_sqft = total_cost) :=
by
  sorry

end book_set_cost_l88_88672


namespace algebraic_expression_value_l88_88528

theorem algebraic_expression_value (m n : ℝ) 
  (h1 : m * n = 3) 
  (h2 : n = m + 1) : 
  (m - n) ^ 2 * ((1 / n) - (1 / m)) = -1 / 3 :=
by sorry

end algebraic_expression_value_l88_88528


namespace prob_union_of_mutually_exclusive_l88_88944

-- Let's denote P as a probability function
variable {Ω : Type} (P : Set Ω → ℝ)

-- Define the mutually exclusive condition
def mutually_exclusive (A B : Set Ω) : Prop :=
  (A ∩ B) = ∅

-- State the theorem that we want to prove
theorem prob_union_of_mutually_exclusive (A B : Set Ω) 
  (h : mutually_exclusive A B) : P (A ∪ B) = P A + P B :=
sorry

end prob_union_of_mutually_exclusive_l88_88944


namespace proof_problem_l88_88990

theorem proof_problem (x y : ℚ) : 
  (x ^ 2 - 9 * y ^ 2 = 0) ∧ 
  (x + y = 1) ↔ 
  ((x = 3/4 ∧ y = 1/4) ∨ (x = 3/2 ∧ y = -1/2)) :=
by
  sorry

end proof_problem_l88_88990


namespace parabola_properties_l88_88121

theorem parabola_properties (p : ℝ) (h : p > 0) (F : ℝ × ℝ) (l : ℝ → ℝ) (A B : ℝ × ℝ) (M : ℝ × ℝ)
  (hp : p = 4) 
  (hF : F = (p / 2, 0)) 
  (hA : A.2^2 = 2 * p * A.1) 
  (hB : B.2^2 = 2 * p * B.1) 
  (hM : M = ((A.1 + B.1) / 2, 2)) 
  (hl : ∀ x, l x = 2 * x - 4) 
  : (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) → 
    (p = 4) ∧ (l 0 = -4) ∧ (A ≠ B) ∧ (|A.1 - B.1| + |A.2 - B.2| = 10) :=
by 
  sorry

end parabola_properties_l88_88121


namespace tan_pi_div_4_sub_theta_l88_88623

theorem tan_pi_div_4_sub_theta (theta : ℝ) (h : Real.tan theta = 1 / 2) : 
  Real.tan (π / 4 - theta) = 1 / 3 := 
sorry

end tan_pi_div_4_sub_theta_l88_88623


namespace charles_ate_no_bananas_l88_88434

theorem charles_ate_no_bananas (W C B : ℝ) (h1 : W = 48) (h2 : C = 35) (h3 : W + C = 83) : B = 0 :=
by
  -- Proof goes here
  sorry

end charles_ate_no_bananas_l88_88434


namespace tail_length_l88_88368

theorem tail_length {length body tail : ℝ} (h1 : length = 30) (h2 : tail = body / 2) (h3 : length = body) : tail = 15 := by
  sorry

end tail_length_l88_88368


namespace coat_price_reduction_l88_88704

theorem coat_price_reduction (original_price : ℝ) (reduction_percent : ℝ)
  (price_is_500 : original_price = 500)
  (reduction_is_30 : reduction_percent = 0.30) :
  original_price * reduction_percent = 150 :=
by
  sorry

end coat_price_reduction_l88_88704


namespace problem_statement_l88_88504

-- Definitions based on the conditions
def P : Prop := ∀ x : ℝ, (0 < x ∧ x < 1) ↔ (x / (x - 1) < 0)
def Q : Prop := ∀ (A B : ℝ), (A > B) → (A > 90 ∨ B < 90)

-- The proof problem statement
theorem problem_statement : P ∧ ¬Q := 
by
  sorry

end problem_statement_l88_88504


namespace betty_paid_44_l88_88601

def slippers := 6
def slippers_cost := 2.5
def lipstick := 4
def lipstick_cost := 1.25
def hair_color := 8
def hair_color_cost := 3

noncomputable def total_cost := (slippers * slippers_cost) + (lipstick * lipstick_cost) + (hair_color * hair_color_cost)

theorem betty_paid_44 : total_cost = 44 :=
by
  sorry

end betty_paid_44_l88_88601


namespace not_prime_expression_l88_88975

theorem not_prime_expression (x y : ℕ) : ¬ Prime (x^8 - x^7 * y + x^6 * y^2 - x^5 * y^3 + x^4 * y^4 
  - x^3 * y^5 + x^2 * y^6 - x * y^7 + y^8) :=
sorry

end not_prime_expression_l88_88975


namespace apple_eating_contest_l88_88324

theorem apple_eating_contest (a z : ℕ) (h_most : a = 8) (h_fewest : z = 1) : a - z = 7 :=
by
  sorry

end apple_eating_contest_l88_88324


namespace maximum_integer_value_of_a_l88_88127

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x - a * Real.log x

theorem maximum_integer_value_of_a (a : ℝ) (h : ∀ x ≥ 1, f x a > 0) : a ≤ 2 :=
sorry

end maximum_integer_value_of_a_l88_88127


namespace different_colors_probability_l88_88522

-- Definitions of the chips in the bag
def purple_chips := 7
def green_chips := 6
def orange_chips := 5
def total_chips := purple_chips + green_chips + orange_chips

-- Calculating probabilities for drawing chips of different colors and ensuring the final probability of different colors is correct
def probability_different_colors : ℚ :=
  let P := purple_chips
  let G := green_chips
  let O := orange_chips
  let T := total_chips
  (P / T) * ((G + O) / T) + (G / T) * ((P + O) / T) + (O / T) * ((P + G) / T)

theorem different_colors_probability : probability_different_colors = (107 / 162) := by
  sorry

end different_colors_probability_l88_88522


namespace shift_upwards_l88_88411

theorem shift_upwards (a : ℝ) :
  (∀ x : ℝ, y = -2 * x + a) -> (a = 1) :=
by
  sorry

end shift_upwards_l88_88411


namespace ways_to_score_at_least_7_points_l88_88050

-- Definitions based on the given conditions
def red_balls : Nat := 4
def white_balls : Nat := 6
def points_red : Nat := 2
def points_white : Nat := 1

-- Function to count the number of combinations for choosing k elements from n elements
def choose (n : Nat) (k : Nat) : Nat :=
  if h : k ≤ n then
    Nat.descFactorial n k / Nat.factorial k
  else
    0

-- The main theorem to prove the number of ways to get at least 7 points by choosing 5 balls out
theorem ways_to_score_at_least_7_points : 
  (choose red_balls 4 * choose white_balls 1) +
  (choose red_balls 3 * choose white_balls 2) +
  (choose red_balls 2 * choose white_balls 3) = 186 := 
sorry

end ways_to_score_at_least_7_points_l88_88050


namespace original_number_solution_l88_88416

theorem original_number_solution (x : ℝ) (h : x^2 + 45 = 100) : x = Real.sqrt 55 ∨ x = -Real.sqrt 55 :=
by
  sorry

end original_number_solution_l88_88416


namespace area_of_triangle_l88_88346

theorem area_of_triangle (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 144) : 
  1/2 * a * b = 30 :=
by sorry

end area_of_triangle_l88_88346


namespace anthony_has_more_pairs_l88_88979

theorem anthony_has_more_pairs (scott_pairs : ℕ) (anthony_pairs : ℕ) (jim_pairs : ℕ) :
  (scott_pairs = 7) →
  (anthony_pairs = 3 * scott_pairs) →
  (jim_pairs = anthony_pairs - 2) →
  (anthony_pairs - jim_pairs = 2) :=
by
  intro h_scott h_anthony h_jim
  sorry

end anthony_has_more_pairs_l88_88979


namespace rockets_win_series_in_exactly_7_games_l88_88694

theorem rockets_win_series_in_exactly_7_games :
  let p_warriors_win := (3 / 4 : ℚ),
      p_rockets_win := (1 / 4 : ℚ),
      comb_6_3 := (Nat.choose 6 3 : ℚ),
      prob_3_3_tie := comb_6_3 * (p_rockets_win ^ 3) * (p_warriors_win ^ 3),
      prob_rockets_win_7th := p_rockets_win
  in (prob_3_3_tie * prob_rockets_win_7th) = (135 / 4096 : ℚ) := by
  sorry

end rockets_win_series_in_exactly_7_games_l88_88694


namespace find_p_l88_88797

theorem find_p (p: ℝ) (x1 x2: ℝ) (h1: p > 0) (h2: x1^2 + p * x1 + 1 = 0) (h3: x2^2 + p * x2 + 1 = 0) (h4: |x1^2 - x2^2| = p) : p = 5 :=
sorry

end find_p_l88_88797


namespace least_5_digit_number_divisible_by_15_25_40_75_125_140_l88_88328

theorem least_5_digit_number_divisible_by_15_25_40_75_125_140 : 
  ∃ n : ℕ, (10000 ≤ n) ∧ (n < 100000) ∧ 
  (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧ (125 ∣ n) ∧ (140 ∣ n) ∧ (n = 21000) :=
by
  sorry

end least_5_digit_number_divisible_by_15_25_40_75_125_140_l88_88328


namespace arithmetic_sequence_tenth_term_l88_88679

noncomputable def sum_of_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

def nth_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_tenth_term
  (a1 d : ℝ)
  (h1 : a1 + (a1 + d) + (a1 + 2 * d) = (a1 + 3 * d) + (a1 + 4 * d))
  (h2 : sum_of_arithmetic_sequence a1 d 5 = 60) :
  nth_term a1 d 10 = 26 :=
sorry

end arithmetic_sequence_tenth_term_l88_88679


namespace original_population_before_changes_l88_88074

open Nat

def halved_population (p: ℕ) (years: ℕ) : ℕ := p / (2^years)

theorem original_population_before_changes (P_init P_final : ℕ)
    (new_people : ℕ) (people_moved_out : ℕ) :
    new_people = 100 →
    people_moved_out = 400 →
    ∀ years, (years = 4 → halved_population P_final years = 60) →
    ∃ P_before_change, P_before_change = 780 ∧
    P_init = P_before_change + new_people - people_moved_out ∧
    halved_population P_init years = P_final := 
by
  intros
  sorry

end original_population_before_changes_l88_88074


namespace num_assignments_l88_88427

/-- 
Mr. Wang originally planned to grade at a rate of 6 assignments per hour.
After grading for 2 hours, he increased his rate to 8 assignments per hour,
finishing 3 hours earlier than initially planned. 
Prove that the total number of assignments is 84. 
-/
theorem num_assignments (x : ℕ) (h : ℕ) (H1 : 6 * h = x) (H2 : 8 * (h - 5) = x - 12) : x = 84 :=
by
  sorry

end num_assignments_l88_88427


namespace task_completion_time_l88_88794

theorem task_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ t : ℝ, t = (a * b) / (a + b) := 
sorry

end task_completion_time_l88_88794


namespace functional_relationship_max_daily_profit_price_reduction_1200_profit_l88_88071

noncomputable def y : ℝ → ℝ := λ x => -2 * x^2 + 60 * x + 800

theorem functional_relationship :
  ∀ x : ℝ, y x = (40 - x) * (20 + 2 * x) := 
by
  intro x
  sorry

theorem max_daily_profit :
  y 15 = 1250 :=
by
  sorry

theorem price_reduction_1200_profit :
  ∀ x : ℝ, y x = 1200 → x = 10 ∨ x = 20 :=
by
  intro x
  sorry

end functional_relationship_max_daily_profit_price_reduction_1200_profit_l88_88071


namespace Melies_money_left_l88_88021

variable (meat_weight : ℕ)
variable (meat_cost_per_kg : ℕ)
variable (initial_money : ℕ)

def money_left_after_purchase (meat_weight : ℕ) (meat_cost_per_kg : ℕ) (initial_money : ℕ) : ℕ :=
  initial_money - (meat_weight * meat_cost_per_kg)

theorem Melies_money_left : 
  money_left_after_purchase 2 82 180 = 16 :=
by
  sorry

end Melies_money_left_l88_88021


namespace distinct_integers_are_squares_l88_88579

theorem distinct_integers_are_squares
  (n : ℕ) 
  (h_n : n = 2000) 
  (x : Fin n → ℕ) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → x i ≠ x j)
  (h_product_square : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ∃ (m : ℕ), x i * x j * x k = m^2) :
  ∀ i : Fin n, ∃ (m : ℕ), x i = m^2 := 
sorry

end distinct_integers_are_squares_l88_88579


namespace polygon_interior_plus_exterior_l88_88566

theorem polygon_interior_plus_exterior (n : ℕ) 
  (h : (n - 2) * 180 + 60 = 1500) : n = 10 :=
sorry

end polygon_interior_plus_exterior_l88_88566


namespace paths_via_checkpoint_l88_88083

/-- Define the grid configuration -/
structure Point :=
  (x : ℕ) (y : ℕ)

/-- Calculate the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  n.choose k

/-- Define points A, B, C -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨5, 4⟩
def C : Point := ⟨3, 2⟩

/-- Calculate number of paths from A to C -/
def paths_A_to_C : ℕ :=
  binomial (3 + 2) 2

/-- Calculate number of paths from C to B -/
def paths_C_to_B : ℕ :=
  binomial (2 + 2) 2

/-- Calculate total number of paths from A to B via C -/
def total_paths_A_to_B_via_C : ℕ :=
  (paths_A_to_C * paths_C_to_B)

theorem paths_via_checkpoint :
  total_paths_A_to_B_via_C = 60 :=
by
  -- The proof is skipped as per the instruction
  sorry

end paths_via_checkpoint_l88_88083


namespace solution_set_for_x_l88_88419

theorem solution_set_for_x (x : ℝ) (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 :=
sorry

end solution_set_for_x_l88_88419


namespace simplify_336_to_fraction_l88_88276

theorem simplify_336_to_fraction :
  let gcd_36_100 := Nat.gcd 36 100
  3.36 = (84 : ℚ) / 25 := 
by
  let g := Nat.gcd 36 100
  have h1 : 3.36 = 3 + 0.36 := by norm_num
  have h2 : 0.36 = 36 / 100 := by norm_num
  have h3 : g = 4 := by norm_num [Nat.gcd, Nat.gcd_def, Nat.gcd_rec]
  have h4 : (36 : ℚ) / 100 = 9 / 25 := by norm_num; field_simp [h3];
  have h5 : (3 : ℚ) + (9 / 25) = 84 / 25 := by norm_num; field_simp;
  rw [h1, h2, h4, h5]

end simplify_336_to_fraction_l88_88276


namespace chang_total_apples_l88_88610

def sweet_apple_price : ℝ := 0.5
def sour_apple_price : ℝ := 0.1
def sweet_apple_percentage : ℝ := 0.75
def sour_apple_percentage : ℝ := 1 - sweet_apple_percentage
def total_earnings : ℝ := 40

theorem chang_total_apples : 
  (total_earnings / (sweet_apple_percentage * sweet_apple_price + sour_apple_percentage * sour_apple_price)) = 100 :=
by
  sorry

end chang_total_apples_l88_88610


namespace triangle_properties_l88_88662

theorem triangle_properties :
  (∀ (α β γ : ℝ), α + β + γ = 180 → 
    (α = β ∨ α = γ ∨ β = γ ∨ 
     (α = 60 ∧ β = 60 ∧ γ = 60) ∨
     ¬(α = 90 ∧ β = 90))) :=
by
  -- Placeholder for the actual proof, ensuring the theorem can build
  intros α β γ h₁
  sorry

end triangle_properties_l88_88662


namespace solve_for_y_l88_88432

-- Define the given condition as a Lean definition
def equation (y : ℝ) : Prop :=
  (2 / y) + ((3 / y) / (6 / y)) = 1.2

-- Theorem statement proving the solution given the condition
theorem solve_for_y (y : ℝ) (h : equation y) : y = 20 / 7 := by
  sorry

-- Example usage to instantiate and make use of the definition
example : equation (20 / 7) := by
  unfold equation
  sorry

end solve_for_y_l88_88432


namespace vasechkin_result_l88_88686

theorem vasechkin_result (x : ℕ) (h : (x / 2 * 7) - 1001 = 7) : (x / 8) ^ 2 - 1001 = 295 :=
by
  sorry

end vasechkin_result_l88_88686


namespace find_r_s_l88_88962

def is_orthogonal (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2.1 * v₂.2.1 + v₁.2.2 * v₂.2.2 = 0

def have_equal_magnitudes (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1^2 + v₁.2.1^2 + v₁.2.2^2 = v₂.1^2 + v₂.2.1^2 + v₂.2.2^2

theorem find_r_s (r s : ℝ) :
  is_orthogonal (4, r, -2) (-1, 2, s) ∧
  have_equal_magnitudes (4, r, -2) (-1, 2, s) →
  r = -11 / 4 ∧ s = -19 / 4 :=
by
  intro h
  sorry

end find_r_s_l88_88962


namespace sum_of_two_numbers_l88_88997

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : 1/x = 3 * (1/y)) : 
  x + y = 16 * Real.sqrt 3 / 3 :=
by
  sorry

end sum_of_two_numbers_l88_88997


namespace excess_calories_l88_88367

-- Conditions
def calories_from_cheezits (bags: ℕ) (ounces_per_bag: ℕ) (calories_per_ounce: ℕ) : ℕ :=
  bags * ounces_per_bag * calories_per_ounce

def calories_from_chocolate_bars (bars: ℕ) (calories_per_bar: ℕ) : ℕ :=
  bars * calories_per_bar

def calories_from_popcorn (calories: ℕ) : ℕ :=
  calories

def calories_burned_running (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_swimming (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

def calories_burned_cycling (minutes: ℕ) (calories_per_minute: ℕ) : ℕ :=
  minutes * calories_per_minute

-- Hypothesis
def total_calories_consumed : ℕ :=
  calories_from_cheezits 3 2 150 + calories_from_chocolate_bars 2 250 + calories_from_popcorn 500

def total_calories_burned : ℕ :=
  calories_burned_running 40 12 + calories_burned_swimming 30 15 + calories_burned_cycling 20 10

-- Theorem
theorem excess_calories : total_calories_consumed - total_calories_burned = 770 := by
  sorry

end excess_calories_l88_88367


namespace three_point_three_six_as_fraction_l88_88280

theorem three_point_three_six_as_fraction : 3.36 = (84 : ℚ) / 25 := 
by
  sorry

end three_point_three_six_as_fraction_l88_88280


namespace price_of_refrigerator_l88_88857

variable (R W : ℝ)

theorem price_of_refrigerator 
  (h1 : W = R - 1490) 
  (h2 : R + W = 7060) 
  : R = 4275 :=
sorry

end price_of_refrigerator_l88_88857


namespace expenditure_of_negative_amount_l88_88647

theorem expenditure_of_negative_amount (x : ℝ) (h : x < 0) : 
  ∃ y : ℝ, y > 0 ∧ x = -y :=
by
  sorry

end expenditure_of_negative_amount_l88_88647


namespace length_of_faster_train_l88_88865

theorem length_of_faster_train (speed_faster_train : ℝ) (speed_slower_train : ℝ) (elapsed_time : ℝ) (relative_speed : ℝ) (length_train : ℝ)
  (h1 : speed_faster_train = 50) 
  (h2 : speed_slower_train = 32) 
  (h3 : elapsed_time = 15) 
  (h4 : relative_speed = (speed_faster_train - speed_slower_train) * (1000 / 3600)) 
  (h5 : length_train = relative_speed * elapsed_time) :
  length_train = 75 :=
sorry

end length_of_faster_train_l88_88865


namespace negation_even_l88_88270

open Nat

theorem negation_even (x : ℕ) (h : 0 < x) :
  (∀ x : ℕ, 0 < x → Even x) ↔ ¬ (∃ x : ℕ, 0 < x ∧ Odd x) :=
by
  sorry

end negation_even_l88_88270


namespace appropriate_investigation_method_l88_88558

theorem appropriate_investigation_method
  (volume_of_investigation_large : Prop)
  (no_need_for_comprehensive_investigation : Prop) :
  (∃ (method : String), method = "sampling investigation") :=
by
  sorry

end appropriate_investigation_method_l88_88558


namespace sufficient_but_not_necessary_condition_l88_88491

def p (x : ℝ) := x^2 + x - 2 > 0
def q (x a : ℝ) := x > a

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x, q x a → p x) ∧ (∃ x, ¬q x a ∧ p x) → a ∈ Set.Ici 1 :=
by
  sorry

end sufficient_but_not_necessary_condition_l88_88491


namespace sin_alpha_minus_pi_over_6_l88_88772

open Real

theorem sin_alpha_minus_pi_over_6 (α : ℝ) (h : sin (α + π / 6) + 2 * sin (α / 2) ^ 2 = 1 - sqrt 2 / 2) : 
  sin (α - π / 6) = -sqrt 2 / 2 :=
sorry

end sin_alpha_minus_pi_over_6_l88_88772


namespace election_result_l88_88429

def votes_A : ℕ := 12
def votes_B : ℕ := 3
def votes_C : ℕ := 15

def is_class_president (candidate_votes : ℕ) : Prop :=
  candidate_votes = max (max votes_A votes_B) votes_C

theorem election_result : is_class_president votes_C :=
by
  unfold is_class_president
  rw [votes_A, votes_B, votes_C]
  sorry

end election_result_l88_88429


namespace complement_of_M_in_U_l88_88541

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U :
  U \ M = {3, 5, 6} := by
  sorry

end complement_of_M_in_U_l88_88541


namespace weight_of_b_l88_88409

-- Definitions based on conditions
variables (A B C : ℝ)

def avg_abc := (A + B + C) / 3 = 45
def avg_ab := (A + B) / 2 = 40
def avg_bc := (B + C) / 2 = 44

-- The theorem to prove
theorem weight_of_b (h1 : avg_abc A B C) (h2 : avg_ab A B) (h3 : avg_bc B C) :
  B = 33 :=
sorry

end weight_of_b_l88_88409


namespace num_invalid_d_l88_88841

noncomputable def square_and_triangle_problem (d : ℕ) : Prop :=
  ∃ a b : ℕ, 3 * a - 4 * b = 1989 ∧ a - b = d ∧ b > 0

theorem num_invalid_d : ∀ (d : ℕ), (d ≤ 663) → ¬ square_and_triangle_problem d :=
by {
  sorry
}

end num_invalid_d_l88_88841


namespace circle_radius_equivalence_l88_88850

theorem circle_radius_equivalence (OP_radius : ℝ) (QR : ℝ) (a : ℝ) (P : ℝ × ℝ) (S : ℝ × ℝ)
  (h1 : P = (12, 5))
  (h2 : S = (a, 0))
  (h3 : QR = 5)
  (h4 : OP_radius = 13) :
  a = 8 := 
sorry

end circle_radius_equivalence_l88_88850


namespace range_of_a_l88_88130

noncomputable def A : Set ℝ := {x : ℝ | ((x^2) - x - 2) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x ∈ A, (x^2 - a*x - a - 2) ≤ 0) → a ≥ (2/3) :=
by
  intro h
  sorry

end range_of_a_l88_88130


namespace ways_to_start_writing_l88_88041

def ratio_of_pens_to_notebooks (pens notebooks : ℕ) : Prop := 
    pens * 4 = notebooks * 5

theorem ways_to_start_writing 
    (pens notebooks : ℕ) 
    (h_ratio : ratio_of_pens_to_notebooks pens notebooks) 
    (h_pens : pens = 50)
    (h_notebooks : notebooks = 40) : 
    ∃ ways : ℕ, ways = 40 :=
by
  sorry

end ways_to_start_writing_l88_88041


namespace handshake_count_l88_88076

def total_handshakes (men women : ℕ) := 
  (men * (men - 1)) / 2 + men * (women - 1)

theorem handshake_count :
  let men := 13
  let women := 13
  total_handshakes men women = 234 :=
by
  sorry

end handshake_count_l88_88076


namespace find_c_of_binomial_square_l88_88935

theorem find_c_of_binomial_square (c : ℝ) (h : ∃ d : ℝ, (9*x^2 - 24*x + c = (3*x + d)^2)) : c = 16 := sorry

end find_c_of_binomial_square_l88_88935


namespace part_a_solution_part_b_solution_l88_88989

-- Part (a)
theorem part_a_solution (x y : ℝ) :
  x^2 + y^2 - 4*x + 6*y + 13 = 0 ↔ (x = 2 ∧ y = -3) :=
sorry

-- Part (b)
theorem part_b_solution (x y : ℝ) :
  xy - 1 = x - y ↔ ((x = 1 ∨ y = 1) ∨ (x ≠ 1 ∧ y ≠ 1)) :=
sorry

end part_a_solution_part_b_solution_l88_88989


namespace rays_dog_daily_walk_l88_88180

theorem rays_dog_daily_walk :
  ∀ (walks_to_park walks_to_school walks_home trips_per_day : ℕ),
    walks_to_park = 4 →
    walks_to_school = 7 →
    walks_home = 11 →
    trips_per_day = 3 →
    trips_per_day * (walks_to_park + walks_to_school + walks_home) = 66 :=
by
  intros walks_to_park walks_to_school walks_home trips_per_day
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end rays_dog_daily_walk_l88_88180


namespace max_value_expression_l88_88165

theorem max_value_expression (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 :=
sorry

end max_value_expression_l88_88165


namespace algebraic_expression_zero_l88_88112

theorem algebraic_expression_zero (a b : ℝ) (h : a^2 + 2 * a * b + b^2 = 0) : 
  a * (a + 4 * b) - (a + 2 * b) * (a - 2 * b) = 0 :=
by
  sorry

end algebraic_expression_zero_l88_88112


namespace part1_69_part1_97_not_part2_difference_numbers_in_range_l88_88152

def is_difference_number (n : ℕ) : Prop :=
  (n % 7 = 6) ∧ (n % 5 = 4)

theorem part1_69 : is_difference_number 69 :=
sorry

theorem part1_97_not : ¬is_difference_number 97 :=
sorry

theorem part2_difference_numbers_in_range :
  {n : ℕ | is_difference_number n ∧ 500 < n ∧ n < 600} = {524, 559, 594} :=
sorry

end part1_69_part1_97_not_part2_difference_numbers_in_range_l88_88152


namespace find_n_l88_88653

-- Definitions of the conditions
variables (x n : ℝ)
variable (h1 : (x / 4) * n + 10 - 12 = 48)
variable (h2 : x = 40)

-- Theorem statement
theorem find_n (x n : ℝ) (h1 : (x / 4) * n + 10 - 12 = 48) (h2 : x = 40) : n = 5 :=
by
  sorry

end find_n_l88_88653


namespace graph_equiv_l88_88206

theorem graph_equiv {x y : ℝ} :
  (x^3 - 2 * x^2 * y + x * y^2 - 2 * y^3 = 0) ↔ (x = 2 * y) :=
sorry

end graph_equiv_l88_88206


namespace find_c_l88_88208

theorem find_c (a b c : ℝ) (h1 : a * 2 = 3 * b / 2) (h2 : a * 2 + 9 = c) (h3 : 4 - 3 * b = -c) : 
  c = 12 :=
by
  sorry

end find_c_l88_88208


namespace Susan_ate_six_candies_l88_88751

def candy_consumption_weekly : Prop :=
  ∀ (candies_bought_Tue candies_bought_Wed candies_bought_Thu candies_bought_Fri : ℕ)
    (candies_left : ℕ) (total_spending : ℕ),
    candies_bought_Tue = 3 →
    candies_bought_Wed = 0 →
    candies_bought_Thu = 5 →
    candies_bought_Fri = 2 →
    candies_left = 4 →
    total_spending = 9 →
    candies_bought_Tue + candies_bought_Wed + candies_bought_Thu + candies_bought_Fri - candies_left = 6

theorem Susan_ate_six_candies : candy_consumption_weekly :=
by {
  -- The proof will be filled in later
  sorry
}

end Susan_ate_six_candies_l88_88751


namespace max_volume_of_cuboid_l88_88586

theorem max_volume_of_cuboid (x y z : ℝ) (h1 : 4 * (x + y + z) = 60) : 
  x * y * z ≤ 125 :=
by
  sorry

end max_volume_of_cuboid_l88_88586


namespace pentagon_stack_valid_sizes_l88_88492

def valid_stack_size (n : ℕ) : Prop :=
  ¬ (n = 1) ∧ ¬ (n = 3)

theorem pentagon_stack_valid_sizes (n : ℕ) :
  valid_stack_size n :=
sorry

end pentagon_stack_valid_sizes_l88_88492


namespace number_of_four_digit_numbers_l88_88054

theorem number_of_four_digit_numbers (digits: Finset ℕ) (h: digits = {1, 1, 2, 0}) :
  ∃ count : ℕ, (count = 9) ∧ 
  (∀ n ∈ digits, n ≠ 0 → n * 1000 + n ≠ 0) := 
sorry

end number_of_four_digit_numbers_l88_88054


namespace holiday_not_on_22nd_l88_88874

def isThirdWednesday (d : ℕ) : Prop :=
  d = 15 ∨ d = 16 ∨ d = 17 ∨ d = 18 ∨ d = 19 ∨ d = 20 ∨ d = 21

theorem holiday_not_on_22nd :
  ¬ isThirdWednesday 22 :=
by
  intro h
  cases h
  repeat { contradiction }

end holiday_not_on_22nd_l88_88874


namespace count_with_consecutive_ones_l88_88933

noncomputable def countValidIntegers : ℕ := 512
noncomputable def invalidCount : ℕ := 89

theorem count_with_consecutive_ones :
  countValidIntegers - invalidCount = 423 :=
by
  sorry

end count_with_consecutive_ones_l88_88933


namespace quadratic_equal_roots_l88_88949

theorem quadratic_equal_roots (a : ℝ) : (∀ x : ℝ, x * (x + 1) + a * x = 0) → a = -1 :=
by sorry

end quadratic_equal_roots_l88_88949


namespace domain_f_log_l88_88784

noncomputable def domain_f (u : Real) : u ∈ Set.Icc (1 : Real) 2 := sorry

theorem domain_f_log (x : Real) : (x ∈ Set.Icc (4 : Real) 16) :=
by
  have h : ∀ x, (1 : Real) ≤ 2^x ∧ 2^x ≤ 2
  { intro x
    sorry }
  have h_log : ∀ x, 2 ≤ x ∧ x ≤ 4 
  { intro x
    sorry }
  have h_domain : ∀ x, 4 ≤ x ∧ x ≤ 16
  { intro x
    sorry }
  exact sorry

end domain_f_log_l88_88784


namespace gcd_fact_8_fact_6_sq_l88_88098

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_fact_8_fact_6_sq : gcd (factorial 8) ((factorial 6)^2) = 11520 := by
  sorry

end gcd_fact_8_fact_6_sq_l88_88098


namespace range_of_k_l88_88774

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x < 3 → x - k < 2 * k) → 1 ≤ k :=
by
  sorry

end range_of_k_l88_88774


namespace total_cookies_collected_l88_88453

theorem total_cookies_collected 
  (abigail_boxes : ℕ) (grayson_boxes : ℕ) (olivia_boxes : ℕ) (cookies_per_box : ℕ)
  (h1 : abigail_boxes = 2) (h2 : grayson_boxes = 3) (h3 : olivia_boxes = 3) (h4 : cookies_per_box = 48) :
  (abigail_boxes * cookies_per_box) + ((grayson_boxes * (cookies_per_box / 4))) + (olivia_boxes * cookies_per_box) = 276 := 
by 
  sorry

end total_cookies_collected_l88_88453


namespace factorization_correct_l88_88754

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l88_88754


namespace sum_of_ages_is_18_l88_88158

-- Define the conditions
def product_of_ages (kiana twin : ℕ) := kiana * twin^2 = 128

-- Define the proof problem statement
theorem sum_of_ages_is_18 : ∃ (kiana twin : ℕ), product_of_ages kiana twin ∧ twin > kiana ∧ kiana + twin + twin = 18 :=
by
  sorry

end sum_of_ages_is_18_l88_88158


namespace find_min_value_l88_88965

noncomputable def min_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem find_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (cond : 2/x + 3/y + 5/z = 10) : min_value x y z = 390625 / 1296 :=
sorry

end find_min_value_l88_88965


namespace parabola_focus_line_ratio_l88_88494

noncomputable def ratio_AF_BF : ℝ := (Real.sqrt 5 + 3) / 2

theorem parabola_focus_line_ratio :
  ∀ (F A B : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (A.2 = 2 * A.1 - 2 ∧ A.2^2 = 4 * A.1 ) ∧ 
    (B.2 = 2 * B.1 - 2 ∧ B.2^2 = 4 * B.1) ∧ 
    A.2 > 0 -> 
  |(A.1 - F.1) / (B.1 - F.1)| = ratio_AF_BF :=
by
  sorry

end parabola_focus_line_ratio_l88_88494


namespace quadratic_solution_transformation_l88_88632

theorem quadratic_solution_transformation
  (m h k : ℝ)
  (h_nonzero : m ≠ 0)
  (x1 x2 : ℝ)
  (h_sol1 : m * (x1 - h)^2 - k = 0)
  (h_sol2 : m * (x2 - h)^2 - k = 0)
  (h_x1 : x1 = 2)
  (h_x2 : x2 = 5) :
  (∃ x1' x2', x1' = 1 ∧ x2' = 4 ∧ m * (x1' - h + 1)^2 = k ∧ m * (x2' - h + 1)^2 = k) :=
by 
  -- Proof here
  sorry

end quadratic_solution_transformation_l88_88632


namespace solve_inequality_l88_88480

theorem solve_inequality :
  {x : ℝ | 8*x^3 - 6*x^2 + 5*x - 5 < 0} = {x : ℝ | x < 1/2} :=
sorry

end solve_inequality_l88_88480


namespace number_of_paths_in_MATHEMATICIAN_diagram_l88_88905

theorem number_of_paths_in_MATHEMATICIAN_diagram : ∃ n : ℕ, n = 8191 :=
by
  -- Define necessary structure
  -- Number of rows and binary choices
  let rows : ℕ := 12
  let choices_per_position : ℕ := 2
  -- Total paths calculation
  let total_paths := choices_per_position ^ rows
  -- Including symmetry and subtracting duplicate
  let final_paths := 2 * total_paths - 1
  use final_paths
  have : final_paths = 8191 :=
    by norm_num
  exact this

end number_of_paths_in_MATHEMATICIAN_diagram_l88_88905


namespace ap_minus_aq_eq_8_l88_88495

theorem ap_minus_aq_eq_8 (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) (p q : ℕ) 
  (h1 : ∀ n, S_n n = n^2 - 5 * n) 
  (h2 : ∀ n ≥ 2, a_n n = S_n n - S_n (n - 1)) 
  (h3 : p - q = 4) :
  a_n p - a_n q = 8 := sorry

end ap_minus_aq_eq_8_l88_88495


namespace smallest_r_l88_88964

theorem smallest_r {p q r : ℕ} (h1 : p < q) (h2 : q < r) (h3 : 2 * q = p + r) (h4 : r * r = p * q) : r = 5 :=
sorry

end smallest_r_l88_88964


namespace books_per_shelf_l88_88423

theorem books_per_shelf (total_distance : ℕ) (total_shelves : ℕ) (one_way_distance : ℕ) 
  (h1 : total_distance = 3200) (h2 : total_shelves = 4) (h3 : one_way_distance = total_distance / 2) 
  (h4 : one_way_distance = 1600) :
  ∀ books_per_shelf : ℕ, books_per_shelf = one_way_distance / total_shelves := 
by
  sorry

end books_per_shelf_l88_88423


namespace number_of_students_l88_88145

theorem number_of_students (S G : ℕ) (h1 : G = 2 * S / 3) (h2 : 8 = 2 * G / 5) : S = 30 :=
by
  sorry

end number_of_students_l88_88145


namespace value_of_S_l88_88138

theorem value_of_S (x R S : ℝ) (h1 : x + 1/x = R) (h2 : R = 6) : x^3 + 1/x^3 = 198 :=
by
  sorry

end value_of_S_l88_88138


namespace tangent_line_at_point_l88_88968

noncomputable def tangent_line_equation (f : ℝ → ℝ) (x : ℝ) : Prop :=
  x - f 0 + 2 = 0

theorem tangent_line_at_point (f : ℝ → ℝ)
  (h_mono : ∀ x y : ℝ, x ≤ y → f x ≤ f y)
  (h_eq : ∀ x : ℝ, f (f x - Real.exp x) = Real.exp 1 + 1) :
  tangent_line_equation f 0 :=
by
  sorry

end tangent_line_at_point_l88_88968


namespace inequality_proof_l88_88813

variables {a b c : ℝ}

theorem inequality_proof (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l88_88813


namespace lowest_possible_sale_price_percentage_l88_88726

def list_price : ℝ := 80
def initial_discount : ℝ := 0.5
def additional_discount : ℝ := 0.2

theorem lowest_possible_sale_price_percentage 
  (list_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) :
  ( (list_price - (list_price * initial_discount)) - (list_price * additional_discount) ) / list_price * 100 = 30 :=
by
  sorry

end lowest_possible_sale_price_percentage_l88_88726


namespace average_of_last_three_numbers_l88_88196

theorem average_of_last_three_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 60) 
  (h2 : (a + b + c) / 3 = 55) : 
  (d + e + f) / 3 = 65 :=
sorry

end average_of_last_three_numbers_l88_88196


namespace triangle_PQ_length_l88_88849

theorem triangle_PQ_length (RP PQ : ℝ) (n : ℕ) (h_rp : RP = 2.4) (h_n : n = 25) : RP = 2.4 → PQ = 3 := by
  sorry

end triangle_PQ_length_l88_88849


namespace barefoot_kids_l88_88424

theorem barefoot_kids (total_kids kids_socks kids_shoes kids_both : ℕ) 
  (h1 : total_kids = 22) 
  (h2 : kids_socks = 12) 
  (h3 : kids_shoes = 8) 
  (h4 : kids_both = 6) : 
  (total_kids - (kids_socks - kids_both + kids_shoes - kids_both + kids_both) = 8) :=
by
  -- following sorry to skip proof.
  sorry

end barefoot_kids_l88_88424


namespace hall_paving_l88_88450

theorem hall_paving :
  ∀ (hall_length hall_breadth stone_length stone_breadth : ℕ),
    hall_length = 72 →
    hall_breadth = 30 →
    stone_length = 8 →
    stone_breadth = 10 →
    let Area_hall := hall_length * hall_breadth
    let Length_stone := stone_length / 10
    let Breadth_stone := stone_breadth / 10
    let Area_stone := Length_stone * Breadth_stone 
    (Area_hall / Area_stone) = 2700 :=
by
  intros hall_length hall_breadth stone_length stone_breadth
  intro h1 h2 h3 h4
  let Area_hall := hall_length * hall_breadth
  let Length_stone := stone_length / 10
  let Breadth_stone := stone_breadth / 10
  let Area_stone := Length_stone * Breadth_stone 
  have h5 : Area_hall / Area_stone = 2700 := sorry
  exact h5

end hall_paving_l88_88450


namespace one_minus_repeating_three_l88_88093

theorem one_minus_repeating_three : ∀ b : ℚ, b = 1 / 3 → 1 - b = 2 / 3 :=
by
  intro b hb
  rw [hb]
  norm_num

end one_minus_repeating_three_l88_88093


namespace gcd_fact8_fact6_squared_l88_88103

-- Definition of 8! and (6!)²
def fact8 : ℕ := 8!
def fact6_squared : ℕ := (6!)^2

-- The theorem statement to be proved
theorem gcd_fact8_fact6_squared : Nat.gcd fact8 fact6_squared = 11520 := 
by
    sorry

end gcd_fact8_fact6_squared_l88_88103


namespace value_of_expression_l88_88922

open Real

theorem value_of_expression {a : ℝ} (h : a^2 + 4 * a - 5 = 0) : 3 * a^2 + 12 * a = 15 :=
by sorry

end value_of_expression_l88_88922


namespace pens_cost_l88_88741

theorem pens_cost (pens_pack_cost : ℝ) (pens_pack_quantity : ℕ) (total_pens : ℕ) (unit_price : ℝ) (total_cost : ℝ)
  (h1 : pens_pack_cost = 45) (h2 : pens_pack_quantity = 150) (h3 : total_pens = 3600) (h4 : unit_price = pens_pack_cost / pens_pack_quantity)
  (h5 : total_cost = total_pens * unit_price) : total_cost = 1080 := by
  sorry

end pens_cost_l88_88741


namespace select_team_with_girls_l88_88543

theorem select_team_with_girls 
  (boys girls : ℕ) 
  (team_size min_girls : ℕ) 
  (boys = 7) 
  (girls = 10) 
  (team_size = 5) 
  (min_girls = 2) : 
  (∑ g in finset.range(min_girls, team_size + 1), 
     nat.choose girls g * nat.choose boys (team_size - g)) = 5817 := 
by
  sorry

end select_team_with_girls_l88_88543


namespace setC_not_basis_l88_88536

-- Definitions based on the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e₁ e₂ : V)
variables (v₁ v₂ : V)

-- Assuming e₁ and e₂ are non-collinear
axiom non_collinear : ¬Collinear ℝ {e₁, e₂}

-- The vectors in the set C
def setC_v1 : V := 3 • e₁ - 2 • e₂
def setC_v2 : V := 4 • e₂ - 6 • e₁

-- The proof problem statement
theorem setC_not_basis : Collinear ℝ {setC_v1 e₁ e₂, setC_v2 e₁ e₂} :=
sorry

end setC_not_basis_l88_88536


namespace retail_price_before_discount_l88_88875

variable (R : ℝ) -- Let R be the retail price of each machine before the discount

theorem retail_price_before_discount :
    let wholesale_price := 126
    let machines := 10
    let bulk_discount_rate := 0.05
    let profit_margin := 0.20
    let sales_tax_rate := 0.07
    let discount_rate := 0.10

    -- Calculate wholesale total price
    let wholesale_total := machines * wholesale_price

    -- Calculate bulk purchase discount
    let bulk_discount := bulk_discount_rate * wholesale_total

    -- Calculate total amount paid
    let amount_paid := wholesale_total - bulk_discount

    -- Calculate profit per machine
    let profit_per_machine := profit_margin * wholesale_price
    
    -- Calculate total profit
    let total_profit := machines * profit_per_machine

    -- Calculate sales tax on profit
    let tax_on_profit := sales_tax_rate * total_profit

    -- Calculate total amount after paying tax
    let total_amount_after_tax := (amount_paid + total_profit) - tax_on_profit

    -- Express total selling price after discount
    let total_selling_after_discount := machines * (0.90 * R)

    -- Total selling price after discount is equal to total amount after tax
    (9 * R = total_amount_after_tax) →
    R = 159.04 :=
by
  sorry

end retail_price_before_discount_l88_88875


namespace platform_length_l88_88297

theorem platform_length (train_length : ℕ) (tree_cross_time : ℕ) (platform_cross_time : ℕ) (platform_length : ℕ)
  (h_train_length : train_length = 1200)
  (h_tree_cross_time : tree_cross_time = 120)
  (h_platform_cross_time : platform_cross_time = 160)
  (h_speed_calculation : (train_length / tree_cross_time = 10))
  : (train_length + platform_length) / 10 = platform_cross_time → platform_length = 400 :=
sorry

end platform_length_l88_88297


namespace randy_piggy_bank_balance_l88_88179

def initial_amount : ℕ := 200
def store_trip_cost : ℕ := 2
def trips_per_month : ℕ := 4
def extra_cost_trip : ℕ := 1
def extra_trip_interval : ℕ := 3
def months_in_year : ℕ := 12
def weekly_income : ℕ := 15
def internet_bill_per_month : ℕ := 20
def birthday_gift : ℕ := 100
def weeks_in_year : ℕ := 52

-- To be proved
theorem randy_piggy_bank_balance : 
  initial_amount 
  + (weekly_income * weeks_in_year) 
  + birthday_gift 
  - ((store_trip_cost * trips_per_month * months_in_year)
  + (months_in_year / extra_trip_interval) * extra_cost_trip
  + (internet_bill_per_month * months_in_year))
  = 740 :=
by
  sorry

end randy_piggy_bank_balance_l88_88179


namespace adjacent_product_negative_l88_88366

-- Define the sequence
def a (n : ℕ) : ℤ := 2*n - 17

-- Define the claim about the product of adjacent terms being negative
theorem adjacent_product_negative : a 8 * a 9 < 0 :=
by sorry

end adjacent_product_negative_l88_88366


namespace cylindrical_to_cartesian_l88_88344

theorem cylindrical_to_cartesian :
  ∀ (r θ z : ℝ), r = 2 → θ = π / 3 → z = 2 → 
  (r * Real.cos θ, r * Real.sin θ, z) = (1, Real.sqrt 3, 2) :=
by
  intros r θ z hr hθ hz
  sorry

end cylindrical_to_cartesian_l88_88344


namespace number_of_ways_to_score_l88_88047

-- Define the conditions
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def score_red : ℕ := 2
def score_white : ℕ := 1
def total_balls : ℕ := 5
def min_score : ℕ := 7

-- Prove the equivalent proof problem
theorem number_of_ways_to_score :
  ∃ ways : ℕ, 
    (ways = ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
             (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
             (Nat.choose red_balls 2) * (Nat.choose white_balls 3))) ∧
    ways = 186 :=
by
  let ways := ((Nat.choose red_balls 4) * (Nat.choose white_balls 1) + 
               (Nat.choose red_balls 3) * (Nat.choose white_balls 2) + 
               (Nat.choose red_balls 2) * (Nat.choose white_balls 3))
  use ways
  constructor
  . rfl
  . sorry

end number_of_ways_to_score_l88_88047


namespace mixture_price_l88_88435

-- Define constants
noncomputable def V1 (X : ℝ) : ℝ := 3.50 * X
noncomputable def V2 : ℝ := 4.30 * 6.25
noncomputable def W2 : ℝ := 6.25
noncomputable def W1 (X : ℝ) : ℝ := X

-- Define the total mixture weight condition
theorem mixture_price (X : ℝ) (P : ℝ) (h1 : W1 X + W2 = 10) (h2 : 10 * P = V1 X + V2) :
  P = 4 := by
  sorry

end mixture_price_l88_88435


namespace new_sequence_after_removal_is_geometric_l88_88921

theorem new_sequence_after_removal_is_geometric (a : ℕ → ℝ) (a₁ q : ℝ) (k : ℕ)
  (h_geo : ∀ n, a n = a₁ * q ^ n) :
  ∀ n, (a (n + k)) = a₁ * q ^ (n + k) :=
by
  sorry

end new_sequence_after_removal_is_geometric_l88_88921


namespace pushups_fri_is_39_l88_88175

/-- Defining the number of pushups done by Miriam -/
def pushups_mon := 5
def pushups_tue := 7
def pushups_wed := pushups_tue * 2
def pushups_total_mon_to_wed := pushups_mon + pushups_tue + pushups_wed
def pushups_thu := pushups_total_mon_to_wed / 2
def pushups_total_mon_to_thu := pushups_mon + pushups_tue + pushups_wed + pushups_thu
def pushups_fri := pushups_total_mon_to_thu

/-- Prove the number of pushups Miriam does on Friday equals 39 -/
theorem pushups_fri_is_39 : pushups_fri = 39 := by 
  sorry

end pushups_fri_is_39_l88_88175


namespace tom_gave_jessica_some_seashells_l88_88847

theorem tom_gave_jessica_some_seashells
  (original_seashells : ℕ := 5)
  (current_seashells : ℕ := 3) :
  original_seashells - current_seashells = 2 :=
by
  sorry

end tom_gave_jessica_some_seashells_l88_88847


namespace probability_of_chosen_primes_l88_88715

def is_prime (n : ℕ) : Prop := sorry -- Assume we have a function to check primality

def total_ways : ℕ := Nat.choose 30 2
def primes_up_to_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
def primes_not_divisible_by_5 : List ℕ := [2, 3, 7, 11, 13, 17, 19, 23, 29]

def chosen_primes (s : Finset ℕ) : Prop :=
  s.card = 2 ∧
  (∀ n ∈ s, n ∈ primes_not_divisible_by_5)  ∧
  (∀ n ∈ s, n ≠ 5) -- (5 is already excluded in the prime list, but for completeness)

def favorable_ways : ℕ := Nat.choose 9 2  -- 9 primes not divisible by 5

def probability := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_chosen_primes:
  probability = (12 / 145 : ℚ) :=
by
  sorry

end probability_of_chosen_primes_l88_88715


namespace money_left_correct_l88_88022

variables (cost_per_kg initial_money kg_bought total_cost money_left : ℕ)

def condition1 : cost_per_kg = 82 := sorry
def condition2 : kg_bought = 2 := sorry
def condition3 : initial_money = 180 := sorry
def condition4 : total_cost = cost_per_kg * kg_bought := sorry
def condition5 : money_left = initial_money - total_cost := sorry

theorem money_left_correct : money_left = 16 := by
  have h1 : cost_per_kg = 82, from condition1
  have h2 : kg_bought = 2, from condition2
  have h3 : initial_money = 180, from condition3
  have h4 : total_cost = cost_per_kg * kg_bought, from condition4
  have h5 : money_left = initial_money - total_cost, from condition5
  rw [h1, h2, h3, h4, h5]
  sorry

end money_left_correct_l88_88022


namespace ninety_nine_fives_not_perfect_square_l88_88360

-- Define the property of the given number "n"
def ninety_nine_fives_one_different_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), 0 ≤ d < 10 ∧
  ((d ≠ 5) ∧ (99.fives.push(d)) = n)

-- Theorem
theorem ninety_nine_fives_not_perfect_square (n : ℕ) :
  ninety_nine_fives_one_different_digit(n) → ¬ (∃ k : ℕ, k^2 = n) :=
by
  sorry

end ninety_nine_fives_not_perfect_square_l88_88360


namespace fraction_value_l88_88814

theorem fraction_value (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = -4 / 3 :=
sorry

end fraction_value_l88_88814


namespace probability_at_least_one_spade_or_ace_l88_88585

open ProbabilityTheory

-- Definitions of the problem conditions
def card_deck : Finset ℕ := Finset.range 54 -- 54 cards represented as numbers from 0 to 53
def spades_or_aces : Finset ℕ := (Finset.range 13) ∪ (Finset.range 13).erase 0 ∪ (Finset.range 13).erase 1 ∪ (Finset.range 13).erase 2

theorem probability_at_least_one_spade_or_ace : 
  (1 - ((54 - 16) / 54) ^ 2) = (368 / 729) := 
by
  sorry
  -- Proof not required

end probability_at_least_one_spade_or_ace_l88_88585


namespace smallest_triangle_perimeter_l88_88452

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def smallest_possible_prime_perimeter : ℕ :=
  31

theorem smallest_triangle_perimeter :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
                  a > 5 ∧ b > 5 ∧ c > 5 ∧
                  is_prime a ∧ is_prime b ∧ is_prime c ∧
                  triangle_inequality a b c ∧
                  is_prime (a + b + c) ∧
                  a + b + c = smallest_possible_prime_perimeter :=
sorry

end smallest_triangle_perimeter_l88_88452


namespace sum_of_interior_angles_pentagon_l88_88707

theorem sum_of_interior_angles_pentagon : (5 - 2) * 180 = 540 := by
  sorry

end sum_of_interior_angles_pentagon_l88_88707


namespace integer_solutions_exist_l88_88391

theorem integer_solutions_exist (R₀ : ℝ) : 
  ∃ (x₁ x₂ x₃ : ℤ), (x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃) ∧ (R₀ < x₁) ∧ (R₀ < x₂) ∧ (R₀ < x₃) := 
sorry

end integer_solutions_exist_l88_88391


namespace first_term_and_common_difference_l88_88493

theorem first_term_and_common_difference (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) :
  a 1 = 1 ∧ (a 2 - a 1) = 4 :=
by
  sorry

end first_term_and_common_difference_l88_88493


namespace inscribed_circle_radius_l88_88148

theorem inscribed_circle_radius (R r : ℝ) (hR : R = 18) (hr : r = 9) : ∃ x : ℝ, x = 8 := 
by
  use 8
  sorry


end inscribed_circle_radius_l88_88148


namespace mixture_weight_l88_88059

theorem mixture_weight (a b : ℝ) (h1 : a = 26.1) (h2 : a / (a + b) = 9 / 20) : a + b = 58 :=
sorry

end mixture_weight_l88_88059


namespace max_PM_PN_l88_88500

noncomputable def C1 := { p : ℝ × ℝ | (p.1 / 2)^2 + (p.2 / (Real.sqrt 3))^2 = 1 }
noncomputable def C2 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }

theorem max_PM_PN (M N : ℝ × ℝ) (P : ℝ × ℝ) 
  (hM : M ∈ C1) (hN : N ∈ C1) (hP : P ∈ C2) :
  ∃ M N, is_vertex_of_C1 M ∧ is_vertex_of_C1 N ∧ 
         ∀ P, P ∈ C2 → PM_PN_criterion M N P (|P - M| + |P - N|) ≤ 2 * Real.sqrt 7 :=
begin
  sorry
end

end max_PM_PN_l88_88500


namespace smallest_x_l88_88720

theorem smallest_x (x : ℕ) : 
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 9 = 8) ↔ x = 314 := 
by
  sorry

end smallest_x_l88_88720


namespace find_c_l88_88938

theorem find_c (c d : ℝ) (h : ∀ x : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) : c = 16 :=
sorry

end find_c_l88_88938


namespace sum_of_fractions_l88_88893

theorem sum_of_fractions :
  (3 / 20 : ℝ) +  (7 / 200) + (8 / 2000) + (3 / 20000) = 0.1892 :=
by 
  sorry

end sum_of_fractions_l88_88893


namespace sum_of_absolute_values_l88_88529

variables {a : ℕ → ℤ} {S₁₀ S₁₈ : ℤ} {T₁₈ : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

def sum_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem sum_of_absolute_values 
  (h1 : a 0 > 0) 
  (h2 : a 9 * a 10 < 0) 
  (h3 : sum_n_terms a 9 = 36) 
  (h4 : sum_n_terms a 17 = 12) :
  (sum_n_terms a 9) - (sum_n_terms a 17 - sum_n_terms a 9) = 60 :=
sorry

end sum_of_absolute_values_l88_88529


namespace Bruce_Anne_combined_cleaning_time_l88_88467

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l88_88467


namespace perpendicular_condition_l88_88351

theorem perpendicular_condition (a : ℝ) :
  let l1 (x y : ℝ) := x + a * y - 2
  let l2 (x y : ℝ) := x - a * y - 1
  (∀ x y, (l1 x y = 0 ↔ l2 x y ≠ 0) ↔ 1 - a * a = 0) →
  (a = -1) ∨ (a = 1) :=
by
  intro
  sorry

end perpendicular_condition_l88_88351


namespace roots_of_quadratic_l88_88169

theorem roots_of_quadratic (m n : ℝ) (h₁ : m + n = -2) (h₂ : m * n = -2022) (h₃ : ∀ x, x^2 + 2 * x - 2022 = 0 → x = m ∨ x = n) :
  m^2 + 3 * m + n = 2020 :=
sorry

end roots_of_quadratic_l88_88169


namespace evaluate_e_T_l88_88535

open Real

noncomputable def T : ℝ :=
  ∫ x in 0..(ln 2), (2 * exp (3 * x) + exp (2 * x) - 1) / (exp (3 * x) + exp (2 * x) - exp x + 1)

theorem evaluate_e_T : exp T = 11 / 4 :=
by
  sorry

end evaluate_e_T_l88_88535


namespace one_minus_repeating_decimal_three_equals_two_thirds_l88_88094

-- Define the repeating decimal as a fraction
def repeating_decimal_three : ℚ := 1 / 3

-- Prove the desired equality
theorem one_minus_repeating_decimal_three_equals_two_thirds :
  1 - repeating_decimal_three = 2 / 3 :=
by
  sorry

end one_minus_repeating_decimal_three_equals_two_thirds_l88_88094


namespace max_volume_prism_l88_88525

theorem max_volume_prism (a b h : ℝ) (h_congruent_lateral : a = b) (sum_areas_eq_48 : a * h + b * h + a * b = 48) : 
  ∃ V : ℝ, V = 64 :=
by
  sorry

end max_volume_prism_l88_88525


namespace worker_bees_in_hive_l88_88144

variable (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ)

def finalWorkerBees (initialWorkerBees leavingWorkerBees returningWorkerBees : ℕ) : ℕ :=
  initialWorkerBees - leavingWorkerBees + returningWorkerBees

theorem worker_bees_in_hive
  (initialWorkerBees : ℕ := 400)
  (leavingWorkerBees : ℕ := 28)
  (returningWorkerBees : ℕ := 15) :
  finalWorkerBees initialWorkerBees leavingWorkerBees returningWorkerBees = 387 := by
  sorry

end worker_bees_in_hive_l88_88144


namespace sum_of_three_consecutive_divisible_by_three_l88_88027

theorem sum_of_three_consecutive_divisible_by_three (n : ℕ) : ∃ k : ℕ, (n + (n + 1) + (n + 2)) = 3 * k := by
  sorry

end sum_of_three_consecutive_divisible_by_three_l88_88027


namespace central_angle_of_sector_l88_88302

theorem central_angle_of_sector (r : ℝ) (θ : ℝ) (h_perimeter: 2 * r + θ * r = π * r / 2) : θ = π - 2 :=
sorry

end central_angle_of_sector_l88_88302


namespace cost_of_3600_pens_l88_88740

theorem cost_of_3600_pens
  (pack_size : ℕ)
  (pack_cost : ℝ)
  (n_pens : ℕ)
  (pen_cost : ℝ)
  (total_cost : ℝ)
  (h1: pack_size = 150)
  (h2: pack_cost = 45)
  (h3: n_pens = 3600)
  (h4: pen_cost = pack_cost / pack_size)
  (h5: total_cost = n_pens * pen_cost) :
  total_cost = 1080 :=
sorry

end cost_of_3600_pens_l88_88740


namespace prob_a_prob_b_l88_88129

def A (a : ℝ) := {x : ℝ | 0 < x + a ∧ x + a ≤ 5}
def B := {x : ℝ | -1/2 ≤ x ∧ x < 6}

theorem prob_a (a : ℝ) : (A a ⊆ B) → (-1 < a ∧ a ≤ 1/2) :=
sorry

theorem prob_b (a : ℝ) : (∃ x, A a ∩ B = {x}) → a = 11/2 :=
sorry

end prob_a_prob_b_l88_88129


namespace find_square_digit_l88_88557

-- Define the known sum of the digits 4, 7, 6, and 9
def sum_known_digits := 4 + 7 + 6 + 9

-- Define the condition that the number 47,69square must be divisible by 6
def is_multiple_of_6 (d : ℕ) : Prop :=
  d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∧ (sum_known_digits + d) % 3 = 0

-- Theorem statement that verifies both the conditions and finds possible values of square
theorem find_square_digit (d : ℕ) (h : is_multiple_of_6 d) : d = 4 ∨ d = 8 :=
by sorry

end find_square_digit_l88_88557


namespace no_solution_iff_k_nonnegative_l88_88499

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1 / 2) ^ x

theorem no_solution_iff_k_nonnegative (k : ℝ) :
  (¬ ∃ x : ℝ, f k (f k x) = 3 / 2) ↔ k ≥ 0 :=
  sorry

end no_solution_iff_k_nonnegative_l88_88499


namespace memorial_visits_l88_88396

theorem memorial_visits (x : ℕ) (total_visits : ℕ) (difference : ℕ) 
  (h1 : total_visits = 589) 
  (h2 : difference = 56) 
  (h3 : 2 * x + difference = total_visits - x) : 
  2 * x + 56 = 589 - x :=
by
  -- proof steps would go here
  sorry

end memorial_visits_l88_88396


namespace factorize_expression_l88_88759

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end factorize_expression_l88_88759


namespace probability_sum_greater_than_four_l88_88226

theorem probability_sum_greater_than_four :
  let all_outcomes := (Fin 6) × (Fin 6)
  let favorable_outcomes := {p : Fin 6 × Fin 6 | (p.1.val + 1) + (p.2.val + 1) > 4}
  (favorable_outcomes.card : ℚ) / ((Fin 6 × Fin 6).card : ℚ) = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88226


namespace daily_rental_cost_l88_88868

theorem daily_rental_cost
  (daily_rent : ℝ)
  (cost_per_mile : ℝ)
  (max_budget : ℝ)
  (miles : ℝ)
  (H1 : cost_per_mile = 0.18)
  (H2 : max_budget = 75)
  (H3 : miles = 250)
  (H4 : daily_rent + (cost_per_mile * miles) = max_budget) : daily_rent = 30 :=
by sorry

end daily_rental_cost_l88_88868


namespace water_needed_l88_88595

-- Definitions as per conditions
def heavy_wash : ℕ := 20
def regular_wash : ℕ := 10
def light_wash : ℕ := 2
def extra_light_wash (bleach : ℕ) : ℕ := bleach * light_wash

def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_loads : ℕ := 2

-- Function to calculate total water usage
def total_water_used : ℕ :=
  (num_heavy_washes * heavy_wash) +
  (num_regular_washes * regular_wash) +
  (num_light_washes * light_wash) + 
  (extra_light_wash num_bleached_loads)

-- Theorem to be proved
theorem water_needed : total_water_used = 76 := by
  sorry

end water_needed_l88_88595


namespace billy_distance_l88_88890

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem billy_distance :
  distance 0 0 (7 + 4 * Real.sqrt 2) (4 * (Real.sqrt 2 + 1)) = Real.sqrt (129 + 88 * Real.sqrt 2) :=
by
  -- proof goes here
  sorry

end billy_distance_l88_88890


namespace average_student_age_before_leaving_l88_88732

theorem average_student_age_before_leaving
  (A : ℕ)
  (student_count : ℕ := 30)
  (leaving_student_age : ℕ := 11)
  (teacher_age : ℕ := 41)
  (new_avg_age : ℕ := 11)
  (new_total_students : ℕ := 30)
  (initial_total_age : ℕ := 30 * A)
  (remaining_students : ℕ := 29)
  (total_age_after_leaving : ℕ := initial_total_age - leaving_student_age)
  (total_age_including_teacher : ℕ := total_age_after_leaving + teacher_age) :
  total_age_including_teacher / new_total_students = new_avg_age → A = 10 := 
  by
    intros h
    sorry

end average_student_age_before_leaving_l88_88732


namespace Bruce_Anne_combined_cleaning_time_l88_88465

-- Define the conditions
def Anne_clean_time : ℕ := 12
def Anne_speed_doubled_time : ℕ := 3
def Bruce_clean_time : ℕ := 6
def Combined_time_with_doubled_speed : ℚ := 1 / 3
def Combined_time_current_speed : ℚ := 1 / 4

-- Prove the problem statement
theorem Bruce_Anne_combined_cleaning_time : 
  (Anne_clean_time = 12) ∧ 
  ((1 / Bruce_clean_time + 1 / 6) = Combined_time_with_doubled_speed) →
  (1 / Combined_time_current_speed) = 4 := 
by
  intro h1
  sorry

end Bruce_Anne_combined_cleaning_time_l88_88465


namespace sum_of_common_ratios_l88_88168

theorem sum_of_common_ratios (k p r a2 a3 b2 b3 : ℝ)
  (h1 : a3 = k * p^2) (h2 : a2 = k * p) 
  (h3 : b3 = k * r^2) (h4 : b2 = k * r)
  (h5 : p ≠ r)
  (h6 : 3 * a3 - 4 * b3 = 5 * (3 * a2 - 4 * b2)) :
  p + r = 5 :=
by {
  sorry
}

end sum_of_common_ratios_l88_88168


namespace strands_of_duct_tape_used_l88_88352

-- Define the conditions
def hannah_cut_rate : ℕ := 8  -- Hannah's cutting rate
def son_cut_rate : ℕ := 3     -- Son's cutting rate
def minutes : ℕ := 2          -- Time taken to free the younger son

-- Define the total cutting rate
def total_cut_rate : ℕ := hannah_cut_rate + son_cut_rate

-- Define the total number of strands
def total_strands : ℕ := total_cut_rate * minutes

-- State the theorem to prove
theorem strands_of_duct_tape_used : total_strands = 22 :=
by
  sorry

end strands_of_duct_tape_used_l88_88352


namespace hospital_staff_total_l88_88362

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

end hospital_staff_total_l88_88362


namespace evaluate_polynomial_l88_88752

-- Define the polynomial function
def polynomial (x : ℝ) : ℝ := x^3 + 3 * x^2 - 9 * x - 5

-- Define the condition: x is the positive root of the quadratic equation
def is_positive_root_of_quadratic (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 9 = 0

-- The main theorem stating the polynomial evaluates to 22 given the condition
theorem evaluate_polynomial {x : ℝ} (h : is_positive_root_of_quadratic x) : polynomial x = 22 := 
by 
  sorry

end evaluate_polynomial_l88_88752


namespace snowfall_difference_l88_88889

def baldMountainSnowfallMeters : ℝ := 1.5
def billyMountainSnowfallMeters : ℝ := 3.5
def mountPilotSnowfallCentimeters : ℝ := 126
def cmPerMeter : ℝ := 100

theorem snowfall_difference :
  billyMountainSnowfallMeters * cmPerMeter + mountPilotSnowfallCentimeters - baldMountainSnowfallMeters * cmPerMeter = 326 :=
by
  sorry

end snowfall_difference_l88_88889


namespace circumscribed_radius_of_triangle_ABC_l88_88747

variable (A B C R : ℝ) (a b c : ℝ)

noncomputable def triangle_ABC (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ B = 2 * A ∧ C = 3 * A

noncomputable def side_length (A a : ℝ) : Prop :=
  a = 6

noncomputable def circumscribed_radius (A B C a R : ℝ) : Prop :=
  2 * R = a / (Real.sin (Real.pi * A / 180))

theorem circumscribed_radius_of_triangle_ABC:
  triangle_ABC A B C →
  side_length A a →
  circumscribed_radius A B C a R →
  R = 6 :=
by
  intros
  sorry

end circumscribed_radius_of_triangle_ABC_l88_88747


namespace final_result_l88_88072

/-- A student chose a number, multiplied it by 5, then subtracted 138 
from the result. The number he chose was 48. What was the final result 
after subtracting 138? -/
theorem final_result (x : ℕ) (h1 : x = 48) : (x * 5) - 138 = 102 := by
  sorry

end final_result_l88_88072


namespace fixed_point_of_function_l88_88544

theorem fixed_point_of_function (a : ℝ) : 
  (a - 1) * 2^1 - 2 * a = -2 := by
  sorry

end fixed_point_of_function_l88_88544


namespace vectors_projection_l88_88721

noncomputable def p := (⟨-44 / 53, 154 / 53⟩ : ℝ × ℝ)

theorem vectors_projection :
  let u := (⟨-4, 2⟩ : ℝ × ℝ)
  let v := (⟨3, 4⟩ : ℝ × ℝ)
  let w := (⟨7, 2⟩ : ℝ × ℝ)
  (⟨(7 * (24 / 53)) - 4, (2 * (24 / 53)) + 2⟩ : ℝ × ℝ) = p :=
by {
  -- proof skipped
  sorry
}

end vectors_projection_l88_88721


namespace find_x_solution_l88_88334

theorem find_x_solution (x : ℚ) : (∀ y : ℚ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by
  sorry

end find_x_solution_l88_88334


namespace first_discount_calculation_l88_88877

-- Define the given conditions and final statement
theorem first_discount_calculation (P : ℝ) (D : ℝ) :
  (1.35 * (1 - D / 100) * 0.85 = 1.03275) → (D = 10.022) :=
by
  -- Proof is not provided, to be done.
  sorry

end first_discount_calculation_l88_88877


namespace bumper_car_rides_correct_l88_88312

def tickets_per_ride : ℕ := 7
def total_tickets : ℕ := 63
def ferris_wheel_rides : ℕ := 5

def tickets_for_bumper_cars : ℕ :=
  total_tickets - ferris_wheel_rides * tickets_per_ride

def bumper_car_rides : ℕ :=
  tickets_for_bumper_cars / tickets_per_ride

theorem bumper_car_rides_correct : bumper_car_rides = 4 :=
by
  sorry

end bumper_car_rides_correct_l88_88312


namespace find_a_for_binomial_square_l88_88909

theorem find_a_for_binomial_square :
  ∃ a : ℚ, (∀ x : ℚ, (∃ r : ℚ, 6 * r = 20 ∧ (r^2 * x^2 + 6 * r * x + 9) = ax^2 + 20x + 9)) ∧ a = 100 / 9 :=
by
  sorry

end find_a_for_binomial_square_l88_88909


namespace one_divides_the_other_l88_88162

theorem one_divides_the_other (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  ∃ m n : ℕ, (x = m * y) ∨ (y = n * x) :=
by 
  -- Proof goes here
  sorry

end one_divides_the_other_l88_88162


namespace sum_of_first_10_terms_l88_88216

-- Mathematical definition of the sequence term
def sequence (n : ℕ) : ℚ := 1 / ((3 * n - 2) * (3 * n + 1))

-- Proves that the sum of the first 10 terms of the sequence equals 10/31
theorem sum_of_first_10_terms : ∑ i in Finset.range 10, sequence i = 10 / 31 :=
by
  sorry

end sum_of_first_10_terms_l88_88216


namespace third_cyclist_speed_l88_88440

theorem third_cyclist_speed (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : 
  ∃ V : ℝ, V = (a + 3 * b + Real.sqrt (a^2 - 10 * a * b + 9 * b^2)) / 4 :=
by
  sorry

end third_cyclist_speed_l88_88440


namespace factorization_correct_l88_88756

theorem factorization_correct (x : ℝ) : 2 * x ^ 2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l88_88756


namespace problem_statement_l88_88675

open BigOperators

def max_value_and_permutations (s : List ℕ) : ℕ × ℕ :=
  let perms := s.permutations
  let values := perms.map (λ l, (l.zip_with (*) l.rotate ++ [l.head' * l.last']).sum)
  let M := values.maximum
  let N := values.filter (λ v, v = M).length
  (M, N)

theorem problem_statement :
  let M_N := max_value_and_permutations [1, 2, 3, 4, 5, 6]
  M_N.fst + M_N.snd = 88 := by
  sorry

end problem_statement_l88_88675


namespace solution_set_l88_88700

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := deriv f x

-- Definitions for conditions
axiom domain_f : ∀ x : ℝ, f x ∈ ℝ
axiom f_at_0 : f 0 = 2
axiom f_ineq : ∀ x : ℝ, f x + f' x > 1

-- Theorem statement
theorem solution_set (x: ℝ) : (e^x * f x > e^x + 1) ↔ (0 < x) :=
by 
  sorry

end solution_set_l88_88700


namespace midpoint_trajectory_of_circle_l88_88738

theorem midpoint_trajectory_of_circle 
  (M P : ℝ × ℝ)
  (B : ℝ × ℝ)
  (hx : B = (3, 0))
  (hp : ∃(a b : ℝ), (P = (2 * a - 3, 2 * b)) ∧ (a^2 + b^2 = 1))
  (hm : M = ((P.1 + B.1) / 2, (P.2 + B.2) / 2)) :
  M.1^2 + M.2^2 - 3 * M.1 + 2 = 0 :=
by {
  -- Proof goes here
  sorry
}

end midpoint_trajectory_of_circle_l88_88738


namespace age_relation_l88_88588

theorem age_relation (S M D Y : ℝ)
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2))
  (h3 : D = S - 4)
  (h4 : M + Y = 3 * (D + Y))
  : Y = -10.5 :=
by
  sorry

end age_relation_l88_88588


namespace second_rice_price_l88_88008

theorem second_rice_price (P : ℝ) 
  (price_first : ℝ := 3.10) 
  (price_mixture : ℝ := 3.25) 
  (ratio_first_to_second : ℝ := 3 / 7) :
  (3 * price_first + 7 * P) / 10 = price_mixture → 
  P = 3.3142857142857145 :=
by
  sorry

end second_rice_price_l88_88008


namespace units_digit_quotient_l88_88322

theorem units_digit_quotient (n : ℕ) (h1 : n % 2 = 1): 
  (4^n + 6^n) / 10 % 10 = 1 :=
by 
  -- Given the cyclical behavior of 4^n % 10 and 6^n % 10
  -- 4^n % 10 cycles between 4 and 6, 6^n % 10 is always 6
  -- Since n is odd, 4^n % 10 = 4 and 6^n % 10 = 6
  -- Adding them gives us 4 + 6 = 10, and thus a quotient of 1
  sorry

end units_digit_quotient_l88_88322


namespace power_addition_rule_l88_88511

variable {a : ℝ}
variable {m n : ℕ}

theorem power_addition_rule (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + n) = 6 := by
  sorry

end power_addition_rule_l88_88511


namespace days_in_first_quarter_2010_l88_88203

theorem days_in_first_quarter_2010 : 
  let not_leap_year := ¬ (2010 % 4 = 0)
  let days_in_february := 28
  let days_in_january_and_march := 31
  not_leap_year → days_in_february = 28 → days_in_january_and_march = 31 → (31 + 28 + 31 = 90)
:= 
sorry

end days_in_first_quarter_2010_l88_88203


namespace total_number_of_elements_l88_88836

theorem total_number_of_elements (a b c : ℕ) : 
  (a = 2 ∧ b = 2 ∧ c = 2) ∧ 
  (3.95 = ((4.4 * 2 + 3.85 * 2 + 3.6000000000000014 * 2) / 6)) ->
  a + b + c = 6 := 
by
  sorry

end total_number_of_elements_l88_88836


namespace find_n_l88_88032

-- Definitions based on conditions
variable (n : ℕ)  -- number of persons
variable (A : Fin n → Finset (Fin n))  -- acquaintance relation, specified as a set of neighbors for each person
-- Condition 1: Each person is acquainted with exactly 8 others
def acquaintances := ∀ i : Fin n, (A i).card = 8
-- Condition 2: Any two acquainted persons have exactly 4 common acquaintances
def common_acquaintances_adj := ∀ i j : Fin n, i ≠ j → j ∈ (A i) → (A i ∩ A j).card = 4
-- Condition 3: Any two non-acquainted persons have exactly 2 common acquaintances
def common_acquaintances_non_adj := ∀ i j : Fin n, i ≠ j → j ∉ (A i) → (A i ∩ A j).card = 2

-- Statement to prove
theorem find_n (h1 : acquaintances n A) (h2 : common_acquaintances_adj n A) (h3 : common_acquaintances_non_adj n A) :
  n = 21 := 
sorry

end find_n_l88_88032


namespace bruce_anne_cleaning_house_l88_88470

theorem bruce_anne_cleaning_house (A B : ℝ) (h1 : A = 1 / 12) (h2 : 2 * A + B = 1 / 3) : 
  1 / (A + B) = 4 :=
by
  -- Define Anne's doubled rate and Bruce's rate from the given conditions
  have h_doubled_rate : 2 * A = 1 / 6, from calc
    2 * A = 2 * (1 / 12) : by rw [h1]
    ... = 1 / 6 : by norm_num,
  -- Substitute Anne's doubled rate into the combined rate equation
  have h_B : B = 1 / 3 - 1 / 6, from calc
    B = 1 / 3 - 2 * A : by rw [←sub_eq_add_neg, vol]
    ... = 1 / 3 - 1 / 6 : by rw [h_doubled_rate],
  -- Calculate the total rate A + B
  have h_total_rate : A + B = 1 / 12 + 1 / 6, from calc
    A + B = A + (1 / 3 - 1 / 6) : by rw [h_B]
    ... = 1 / 12 + 1 / 6 : by rw [h1]
    ... = 1 / 4 : by norm_num,
  -- Verify the time T it takes for Bruce and Anne to clean the house is 4 hours
  show 1 / (A + B) = 4, 
  by rw [h_total_rate]; norm_num

-- Proof is skipped as indicated
sorry

end bruce_anne_cleaning_house_l88_88470


namespace basketball_player_ft_rate_l88_88298

theorem basketball_player_ft_rate :
  ∃ P : ℝ, 1 - P^2 = 16 / 25 ∧ P = 3 / 5 := sorry

end basketball_player_ft_rate_l88_88298


namespace SMUG_TWC_minimum_bouts_l88_88570

noncomputable def minimum_bouts (n : ℕ) : ℕ :=
  let total_edges := (n * (n - 1)) / 2
  let turan_edges := (n^2) / 4
  total_edges - turan_edges

theorem SMUG_TWC_minimum_bouts :
  minimum_bouts 2008 = 999000 :=
by
  let total_edges := (2008 * 2007) / 2
  let turan_edges := (2008^2) / 4
  exact total_edges - turan_edges

end SMUG_TWC_minimum_bouts_l88_88570


namespace negate_proposition_l88_88564

theorem negate_proposition : (∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ ¬ (∃ x : ℝ, x^3 - x^2 + 1 > 1) :=
by
  sorry

end negate_proposition_l88_88564


namespace fraction_representation_of_3_36_l88_88279

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l88_88279


namespace bruce_anne_clean_in_4_hours_l88_88459

variable (B : ℝ) -- time it takes for Bruce to clean the house alone
variable (anne_rate := 1 / 12) -- Anne's rate of cleaning the house
variable (double_anne_rate := 1 / 6) -- Anne's rate if her speed is doubled
variable (combined_rate_when_doubled := 1 / 3) -- Combined rate if Anne's speed is doubled

-- Condition: Combined rate of Bruce and doubled Anne is 1/3 house per hour
axiom condition1 : (1 / B + double_anne_rate = combined_rate_when_doubled)

-- Prove that it takes Bruce and Anne together 4 hours to clean the house at their current rates
theorem bruce_anne_clean_in_4_hours (B : ℝ) (h1 : anne_rate = 1/12) (h2 : (1 / B + double_anne_rate = combined_rate_when_doubled)) :
  (1 / (1 / B + anne_rate) = 4) :=
by
  sorry

end bruce_anne_clean_in_4_hours_l88_88459


namespace polynomial_roots_l88_88096

theorem polynomial_roots :
  (∀ x : ℤ, (x^3 - 4*x^2 - 11*x + 24 = 0) ↔ (x = 4 ∨ x = 3 ∨ x = -1)) :=
sorry

end polynomial_roots_l88_88096


namespace solve_problems_l88_88524

theorem solve_problems (x y : ℕ) (hx : x + y = 14) (hy : 7 * x - 12 * y = 60) : x = 12 :=
sorry

end solve_problems_l88_88524


namespace calculate_triangle_area_l88_88764

-- Define the side lengths of the triangle.
def side1 : ℕ := 13
def side2 : ℕ := 13
def side3 : ℕ := 24

-- Define the area calculation.
noncomputable def triangle_area : ℕ := 60

-- Statement of the theorem we wish to prove.
theorem calculate_triangle_area :
  ∃ (a b c : ℕ) (area : ℕ), a = side1 ∧ b = side2 ∧ c = side3 ∧ area = triangle_area :=
sorry

end calculate_triangle_area_l88_88764


namespace vat_percentage_is_15_l88_88789

def original_price : ℝ := 1700
def final_price : ℝ := 1955
def tax_amount := final_price - original_price

theorem vat_percentage_is_15 :
  (tax_amount / original_price) * 100 = 15 := 
sorry

end vat_percentage_is_15_l88_88789


namespace steve_take_home_pay_l88_88403

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end steve_take_home_pay_l88_88403


namespace n_is_900_l88_88090

theorem n_is_900 
  (m n : ℕ) 
  (h1 : ∃ x y : ℤ, m = x^2 ∧ n = y^2) 
  (h2 : Prime (m - n)) : n = 900 := 
sorry

end n_is_900_l88_88090


namespace locus_of_points_line_or_point_l88_88971

theorem locus_of_points_line_or_point {n : ℕ} (A B : ℕ → ℝ) (k : ℝ) (h : ∀ i, 1 ≤ i ∧ i < n → (A (i + 1) - A i) / (B (i + 1) - B i) = k) :
  ∃ l : ℝ, ∀ i, 1 ≤ i ∧ i ≤ n → (A i + l*B i) = A 1 + l*B 1 :=
by
  sorry

end locus_of_points_line_or_point_l88_88971


namespace probability_of_divisibility_l88_88358

noncomputable def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def is_prime_digit_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, is_prime_digit d

noncomputable def is_divisible_by_3_and_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0

theorem probability_of_divisibility (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999 ∨ 10 ≤ n ∧ n ≤ 99) →
  is_prime_digit_number n →
  ¬ is_divisible_by_3_and_4 n :=
by
  intros h1 h2
  sorry

end probability_of_divisibility_l88_88358


namespace negation_statement_l88_88210

variable {α : Type} (teacher generous : α → Prop)

theorem negation_statement :
  ¬ ∀ x, teacher x → generous x ↔ ∃ x, teacher x ∧ ¬ generous x := by
sorry

end negation_statement_l88_88210


namespace max_distance_circle_to_line_l88_88007

open Real

-- Definitions of polar equations and transformations to Cartesian coordinates
def circle_eq (ρ θ : ℝ) : Prop := (ρ = 8 * sin θ)
def line_eq (θ : ℝ) : Prop := (θ = π / 3)

-- Cartesian coordinate transformations
def circle_cartesian (x y : ℝ) : Prop := (x^2 + (y - 4)^2 = 16)
def line_cartesian (x y : ℝ) : Prop := (y = sqrt 3 * x)

-- Maximum distance problem statement
theorem max_distance_circle_to_line : 
  ∀ (x y : ℝ), circle_cartesian x y → 
  (∀ x y, line_cartesian x y → 
  ∃ d : ℝ, d = 6) :=
by
  sorry

end max_distance_circle_to_line_l88_88007


namespace rooks_same_distance_l88_88425

theorem rooks_same_distance (rooks : Fin 8 → (ℕ × ℕ)) 
    (h_non_attacking : ∀ i j, i ≠ j → Prod.fst (rooks i) ≠ Prod.fst (rooks j) ∧ Prod.snd (rooks i) ≠ Prod.snd (rooks j)) 
    : ∃ i j k l, i ≠ j ∧ k ≠ l ∧ (Prod.fst (rooks i) - Prod.fst (rooks k))^2 + (Prod.snd (rooks i) - Prod.snd (rooks k))^2 = (Prod.fst (rooks j) - Prod.fst (rooks l))^2 + (Prod.snd (rooks j) - Prod.snd (rooks l))^2 :=
by 
  -- Proof goes here
  sorry

end rooks_same_distance_l88_88425


namespace prism_volume_l88_88833

theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 5 :=
by
  sorry

end prism_volume_l88_88833


namespace accuracy_l88_88441

-- Given number and accuracy statement
def given_number : ℝ := 3.145 * 10^8
def expanded_form : ℕ := 314500000

-- Proof statement: the number is accurate to the hundred thousand's place
theorem accuracy (h : given_number = expanded_form) : 
  ∃ n : ℕ, expanded_form = n * 10^5 ∧ (n % 10) ≠ 0 := 
by
  sorry

end accuracy_l88_88441


namespace factorize_expr_l88_88760

noncomputable def example_expr (x : ℝ) : ℝ := 2 * x^2 - 4 * x

theorem factorize_expr (x : ℝ) : example_expr x = 2 * x * (x - 2) := by
  sorry

end factorize_expr_l88_88760


namespace first_plot_germination_rate_l88_88335

-- Define the known quantities and conditions
def plot1_seeds : ℕ := 300
def plot2_seeds : ℕ := 200
def plot2_germination_rate : ℚ := 35 / 100
def total_germination_percentage : ℚ := 26 / 100

-- Define a statement to prove the percentage of seeds that germinated in the first plot
theorem first_plot_germination_rate : 
  ∃ (x : ℚ), (x / 100) * plot1_seeds + (plot2_germination_rate * plot2_seeds) = total_germination_percentage * (plot1_seeds + plot2_seeds) ∧ x = 20 :=
by
  sorry

end first_plot_germination_rate_l88_88335


namespace value_of_f_neg_2009_l88_88338

def f (a b x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem value_of_f_neg_2009 (a b : ℝ) (h : f a b 2009 = 10) :
  f a b (-2009) = -14 :=
by 
  sorry

end value_of_f_neg_2009_l88_88338


namespace yvette_final_bill_l88_88664

def cost_alicia : ℝ := 7.50
def cost_brant : ℝ := 10.00
def cost_josh : ℝ := 8.50
def cost_yvette : ℝ := 9.00
def tip_rate : ℝ := 0.20

def total_cost := cost_alicia + cost_brant + cost_josh + cost_yvette
def tip := tip_rate * total_cost
def final_bill := total_cost + tip

theorem yvette_final_bill :
  final_bill = 42.00 :=
  sorry

end yvette_final_bill_l88_88664


namespace bridge_crossing_possible_l88_88426

/-- 
  There are four people A, B, C, and D. 
  The time it takes for each of them to cross the bridge is 2, 4, 6, and 8 minutes respectively.
  No more than two people can be on the bridge at the same time.
  Prove that it is possible for all four people to cross the bridge in 10 minutes.
--/
theorem bridge_crossing_possible : 
  ∃ (cross : ℕ → ℕ), 
  cross 1 = 2 ∧ cross 2 = 4 ∧ cross 3 = 6 ∧ cross 4 = 8 ∧
  (∀ (t : ℕ), t ≤ 2 → cross 1 + cross 2 + cross 3 + cross 4 = 10) :=
by
  sorry

end bridge_crossing_possible_l88_88426


namespace yoongi_age_l88_88436

theorem yoongi_age (H Yoongi : ℕ) : H = Yoongi + 2 ∧ H + Yoongi = 18 → Yoongi = 8 :=
by
  sorry

end yoongi_age_l88_88436


namespace city_population_l88_88296

theorem city_population (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 :=
by
  sorry

end city_population_l88_88296


namespace probability_sum_greater_than_four_l88_88242

theorem probability_sum_greater_than_four : 
  let total_outcomes := 36 in
  let favorable_outcomes := total_outcomes - 6 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 5 / 6 :=
by
  sorry

end probability_sum_greater_than_four_l88_88242


namespace original_expenditure_mess_l88_88861

theorem original_expenditure_mess : 
  ∀ (x : ℝ), 
  35 * x + 42 = 42 * (x - 1) + 35 * x → 
  35 * 12 = 420 :=
by
  intro x
  intro h
  sorry

end original_expenditure_mess_l88_88861


namespace total_games_l88_88682

-- The conditions
def working_games : ℕ := 6
def bad_games : ℕ := 5

-- The theorem to prove
theorem total_games : working_games + bad_games = 11 :=
by
  sorry

end total_games_l88_88682


namespace missing_bricks_is_26_l88_88934

-- Define the number of bricks per row and the number of rows
def bricks_per_row : Nat := 10
def number_of_rows : Nat := 6

-- Calculate the total number of bricks for a fully completed wall
def total_bricks_full_wall : Nat := bricks_per_row * number_of_rows

-- Assume the number of bricks currently present
def bricks_currently_present : Nat := total_bricks_full_wall - 26

-- Define a function that calculates the number of missing bricks
def number_of_missing_bricks (total_bricks : Nat) (bricks_present : Nat) : Nat :=
  total_bricks - bricks_present

-- Prove that the number of missing bricks is 26
theorem missing_bricks_is_26 : 
  number_of_missing_bricks total_bricks_full_wall bricks_currently_present = 26 :=
by
  sorry

end missing_bricks_is_26_l88_88934


namespace samuel_faster_l88_88828

theorem samuel_faster (S T_h : ℝ) (hT_h : T_h = 1.3) (hS : S = 30) :
  (T_h * 60) - S = 48 :=
by
  sorry

end samuel_faster_l88_88828


namespace solve_for_x_l88_88482

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 3 * x) :
  (4 * y^2 + y + 6 = 3 * (9 * x^2 + y + 3)) ↔ (x = 1 ∨ x = -1/3) :=
by
  sorry

end solve_for_x_l88_88482


namespace probability_correct_l88_88235

def outcome (s₁ s₂ : ℕ) : Prop := s₁ ≥ 1 ∧ s₁ ≤ 6 ∧ s₂ ≥ 1 ∧ s₂ ≤ 6

def sum_outcome_greater_than_four (s₁ s₂ : ℕ) : Prop := outcome s₁ s₂ ∧ s₁ + s₂ > 4

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 30 -- As derived from 36 - 6

def probability_sum_greater_than_four : ℚ := favorable_outcomes / total_outcomes

theorem probability_correct : probability_sum_greater_than_four = 5 / 6 := 
by 
  sorry

end probability_correct_l88_88235


namespace area_of_black_region_l88_88958

def side_length_square : ℝ := 10
def length_rectangle : ℝ := 5
def width_rectangle : ℝ := 2

theorem area_of_black_region :
  (side_length_square * side_length_square) - (length_rectangle * width_rectangle) = 90 := by
sorry

end area_of_black_region_l88_88958


namespace fish_farm_estimated_mass_l88_88448

noncomputable def total_fish_mass_in_pond 
  (initial_fry: ℕ) 
  (survival_rate: ℝ) 
  (haul1_count: ℕ) (haul1_avg_weight: ℝ) 
  (haul2_count: ℕ) (haul2_avg_weight: ℝ) 
  (haul3_count: ℕ) (haul3_avg_weight: ℝ) : ℝ :=
  let surviving_fish := initial_fry * survival_rate
  let total_mass_haul1 := haul1_count * haul1_avg_weight
  let total_mass_haul2 := haul2_count * haul2_avg_weight
  let total_mass_haul3 := haul3_count * haul3_avg_weight
  let average_weight_per_fish := (total_mass_haul1 + total_mass_haul2 + total_mass_haul3) / (haul1_count + haul2_count + haul3_count)
  average_weight_per_fish * surviving_fish

theorem fish_farm_estimated_mass :
  total_fish_mass_in_pond 
    80000           -- initial fry
    0.95            -- survival rate
    40 2.5          -- first haul: 40 fish, 2.5 kg each
    25 2.2          -- second haul: 25 fish, 2.2 kg each
    35 2.8          -- third haul: 35 fish, 2.8 kg each
    = 192280 := by
  sorry

end fish_farm_estimated_mass_l88_88448


namespace jeans_price_increase_l88_88437

theorem jeans_price_increase (M R C : ℝ) (hM : M = 100) 
  (hR : R = M * 1.4)
  (hC : C = R * 1.1) : 
  (C - M) / M * 100 = 54 :=
by
  sorry

end jeans_price_increase_l88_88437


namespace connie_total_markers_l88_88081

theorem connie_total_markers :
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  red_markers + blue_markers + green_markers + purple_markers = 15225 :=
by
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  -- Proof would go here, but we use sorry to skip it for now
  sorry

end connie_total_markers_l88_88081


namespace phraseCompletion_l88_88730

-- Define the condition for the problem
def isCorrectPhrase (phrase : String) : Prop :=
  phrase = "crying"

-- State the theorem to be proven
theorem phraseCompletion : ∃ phrase, isCorrectPhrase phrase :=
by
  use "crying"
  sorry

end phraseCompletion_l88_88730


namespace newspapers_on_sunday_l88_88024

theorem newspapers_on_sunday (papers_weekend : ℕ) (diff_papers : ℕ) 
  (h1 : papers_weekend = 110) 
  (h2 : diff_papers = 20) 
  (h3 : ∃ (S Su : ℕ), Su = S + diff_papers ∧ S + Su = papers_weekend) :
  ∃ Su, Su = 65 :=
by
  sorry

end newspapers_on_sunday_l88_88024


namespace fruit_problem_l88_88859

variables (A O x : ℕ) -- Natural number variables for apples, oranges, and oranges put back

theorem fruit_problem :
  (A + O = 10) ∧
  (40 * A + 60 * O = 480) ∧
  (240 + 60 * (O - x) = 45 * (10 - x)) →
  A = 6 ∧ O = 4 ∧ x = 2 :=
  sorry

end fruit_problem_l88_88859


namespace Anthony_vs_Jim_l88_88982

variable (Scott_pairs : ℕ)
variable (Anthony_pairs : ℕ)
variable (Jim_pairs : ℕ)

axiom Scott_value : Scott_pairs = 7
axiom Anthony_value : Anthony_pairs = 3 * Scott_pairs
axiom Jim_value : Jim_pairs = Anthony_pairs - 2

theorem Anthony_vs_Jim (Scott_pairs Anthony_pairs Jim_pairs : ℕ) 
  (Scott_value : Scott_pairs = 7) 
  (Anthony_value : Anthony_pairs = 3 * Scott_pairs) 
  (Jim_value : Jim_pairs = Anthony_pairs - 2) :
  Anthony_pairs - Jim_pairs = 2 := 
sorry

end Anthony_vs_Jim_l88_88982


namespace find_k_l88_88084

def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 6
def g (k x : ℝ) : ℝ := 2 * x^2 - k * x + 2

theorem find_k (k : ℝ) : 
  f 5 - g k 5 = 15 -> k = -15.8 :=
by
  intro h
  sorry

end find_k_l88_88084


namespace nth_equation_l88_88387

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - 1 = 4 * n * (n + 1) := 
by
  sorry

end nth_equation_l88_88387


namespace binary_to_decimal_101101_l88_88897

theorem binary_to_decimal_101101 :
  let b := [1, 0, 1, 1, 0, 1] in
  let decimal := b[0] * 2^5 + b[1] * 2^4 + b[2] * 2^3 + b[3] * 2^2 + b[4] * 2^1 + b[5] * 2^0 in
  decimal = 45 :=
by
  let b := [1, 0, 1, 1, 0, 1];
  let decimal := b[0] * 2^5 + b[1] * 2^4 + b[2] * 2^3 + b[3] * 2^2 + b[4] * 2^1 + b[5] * 2^0;
  show decimal = 45;
  sorry

end binary_to_decimal_101101_l88_88897


namespace polynomial_simplification_l88_88829

theorem polynomial_simplification (x : ℝ) :
    (3 * x - 2) * (5 * x^12 - 3 * x^11 + 4 * x^9 - 2 * x^8)
    = 15 * x^13 - 19 * x^12 + 6 * x^11 + 12 * x^10 - 14 * x^9 - 4 * x^8 := by
  sorry

end polynomial_simplification_l88_88829


namespace coordinates_of_point_A_l88_88804

theorem coordinates_of_point_A (A B : ℝ × ℝ) (hAB : B.1 = 2 ∧ B.2 = 4) (hParallel : A.2 = B.2) (hDist : abs (A.1 - B.1) = 3) :
  A = (5, 4) ∨ A = (-1, 4) :=
by
  cases hAB with hx hy
  rw [hx, hy] at *
  cases hParallel
  rw [hParallel] at hDist
  cases abs_eq (A.1 - 2) (A.1 - 2) with ha ha
  case h1 =>
    rw [add_comm, add_right_eq_self, sub_eq_iff_eq_add] at ha
    left
    exact ⟨hx.symm ▸ ha, hParallel⟩
  case h2 =>
    rw [add_neg_cancel_right, add_eq_zero_iff_eq_neg_eq] at ha
    right
    exact ⟨hx.symm ▸ ha, hParallel⟩

end coordinates_of_point_A_l88_88804


namespace unique_solution_k_l88_88750

theorem unique_solution_k (k : ℝ) : 
  (∀ x : ℝ, (3 * x + 5) * (x - 3) = -15 + k * x → (3 * x ^ 2 - (4 + k) * x = 0)) → k = -4 :=
begin
  sorry
end

end unique_solution_k_l88_88750


namespace train_length_eq_l88_88594

theorem train_length_eq (L : ℝ) (time_tree time_platform length_platform : ℝ)
  (h_tree : time_tree = 60) (h_platform : time_platform = 105) (h_length_platform : length_platform = 450)
  (h_speed_eq : L / time_tree = (L + length_platform) / time_platform) :
  L = 600 :=
by
  sorry

end train_length_eq_l88_88594


namespace angle_between_vectors_acute_l88_88341

def isAcuteAngle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 > 0

def notCollinear (a b : ℝ × ℝ) : Prop :=
  ¬ ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem angle_between_vectors_acute (m : ℝ) :
  let a := (-1, 1)
  let b := (2 * m, m + 3)
  isAcuteAngle a b ∧ notCollinear a b ↔ m < 3 ∧ m ≠ -1 :=
by
  sorry

end angle_between_vectors_acute_l88_88341


namespace bruce_and_anne_clean_together_l88_88464

noncomputable def clean_together (A B : ℕ) : ℕ := (A*B) / (A + B)

theorem bruce_and_anne_clean_together :
  ∀ (A B T : ℕ), A = 12 → ((2 / A) * T = 1) → 
  B = 6 → 
  T = 3 →
  clean_together A B = 4 :=
by
  intros A B T h1 h2 h3 h4
  dsimp [clean_together]
  rw [h1, h3, nat.zero_div, nat.zero_div, add_comm]  
  sorry

end bruce_and_anne_clean_together_l88_88464


namespace find_number_of_3cm_books_l88_88582

-- Define the conditions
def total_books : ℕ := 46
def total_thickness : ℕ := 200
def thickness_3cm : ℕ := 3
def thickness_5cm : ℕ := 5

-- Let x be the number of 3 cm thick books, y be the number of 5 cm thick books
variable (x y : ℕ)

-- Define the system of equations based on the given conditions
axiom total_books_eq : x + y = total_books
axiom total_thickness_eq : thickness_3cm * x + thickness_5cm * y = total_thickness

-- The theorem to prove: x = 15
theorem find_number_of_3cm_books : x = 15 :=
by
  sorry

end find_number_of_3cm_books_l88_88582


namespace number_of_apples_l88_88609

theorem number_of_apples (A : ℝ) (h : 0.75 * A * 0.5 + 0.25 * A * 0.1 = 40) : A = 100 :=
by
  sorry

end number_of_apples_l88_88609


namespace betty_paid_44_l88_88602

def slippers := 6
def slippers_cost := 2.5
def lipstick := 4
def lipstick_cost := 1.25
def hair_color := 8
def hair_color_cost := 3

noncomputable def total_cost := (slippers * slippers_cost) + (lipstick * lipstick_cost) + (hair_color * hair_color_cost)

theorem betty_paid_44 : total_cost = 44 :=
by
  sorry

end betty_paid_44_l88_88602


namespace students_scoring_130_or_higher_l88_88512

noncomputable def math_exam_students_scoring_130_or_higher : ℕ :=
  let μ := 120
  let σ := 10  -- Since variance is 100, standard deviation σ = sqrt(100) = 10
  let total_students := 40
  -- The given condition to use
  let p_value := 0.1587
  -- Multiply probability by the total number of students
  let number_of_students := total_students * p_value 
  -- Round to the nearest whole number
  in round number_of_students

theorem students_scoring_130_or_higher (μ σ : ℝ) (total_students : ℕ) : 
  μ = 120 ∧ σ = 10 ∧ total_students = 40 → math_exam_students_scoring_130_or_higher = 6 :=
by
  intro h
  rw [math_exam_students_scoring_130_or_higher]
  have μ : ℝ := 120
  have σ : ℝ := 10
  have total_students : ℕ := 40
  have p_value : ℝ := 0.1587
  have number_of_students := total_students * p_value
  finish_round number_of_students = 6
  sorry

end students_scoring_130_or_higher_l88_88512


namespace fraction_zero_if_abs_x_eq_one_l88_88045

theorem fraction_zero_if_abs_x_eq_one (x : ℝ) : 
  (|x| - 1) = 0 → (x^2 - 2 * x + 1 ≠ 0) → x = -1 := 
by 
  sorry

end fraction_zero_if_abs_x_eq_one_l88_88045


namespace linear_function_passing_points_l88_88927

theorem linear_function_passing_points :
  ∃ k b : ℝ, (∀ x : ℝ, y = k * x + b) ∧ (k * 0 + b = 3) ∧ (k * (-4) + b = 0)
  →
  (∃ a : ℝ, y = -((3:ℝ) / (4:ℝ)) * x + 3 ∧ (∀ x y : ℝ, y = -((3:ℝ) / (4:ℝ)) * a + 3 → y = 6 → a = -4)) :=
by sorry

end linear_function_passing_points_l88_88927


namespace cake_pieces_kept_l88_88376

theorem cake_pieces_kept (total_pieces : ℕ) (two_fifths_eaten : ℕ) (extra_pieces_eaten : ℕ)
  (h1 : total_pieces = 35)
  (h2 : two_fifths_eaten = 2 * total_pieces / 5)
  (h3 : extra_pieces_eaten = 3)
  (correct_answer : ℕ)
  (h4 : correct_answer = total_pieces - (two_fifths_eaten + extra_pieces_eaten)) :
  correct_answer = 18 := by
  sorry

end cake_pieces_kept_l88_88376


namespace pushups_fri_is_39_l88_88174

/-- Defining the number of pushups done by Miriam -/
def pushups_mon := 5
def pushups_tue := 7
def pushups_wed := pushups_tue * 2
def pushups_total_mon_to_wed := pushups_mon + pushups_tue + pushups_wed
def pushups_thu := pushups_total_mon_to_wed / 2
def pushups_total_mon_to_thu := pushups_mon + pushups_tue + pushups_wed + pushups_thu
def pushups_fri := pushups_total_mon_to_thu

/-- Prove the number of pushups Miriam does on Friday equals 39 -/
theorem pushups_fri_is_39 : pushups_fri = 39 := by 
  sorry

end pushups_fri_is_39_l88_88174


namespace Vasechkin_result_l88_88683

-- Define the operations
def P (x : ℕ) : ℕ := (x / 2 * 7) - 1001
def V (x : ℕ) : ℕ := (x / 8) ^ 2 - 1001

-- Define the proposition
theorem Vasechkin_result (x : ℕ) (h_prime : P x = 7) : V x = 295 := 
by {
  -- Proof is omitted
  sorry
}

end Vasechkin_result_l88_88683


namespace alice_wins_chomp_l88_88753

def symmetrical_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ turn : ℕ × ℕ, 
    strategy turn = 
      if turn = (1,1) then (1,1)
      else if turn.fst = 2 ∧ turn.snd = 2 then (2,2)
      else if turn.fst = 1 then (turn.snd, 1)
      else (1, turn.fst)) 

theorem alice_wins_chomp (n : ℕ) (h : 1 ≤ n) : 
  symmetrical_strategy n := 
sorry

end alice_wins_chomp_l88_88753


namespace cartesian_equation_of_circle_sum_distances_PA_PB_l88_88658

noncomputable def polar_to_cartesian_circle (rho : ℝ → ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, rho (real.sqrt (x^2 + y^2)) = 2 * real.sqrt 5 * real.sin y

def parametric_equation_line (t : ℝ) : ℝ × ℝ :=
  (3 - (real.sqrt 2 / 2) * t, real.sqrt 5 - (real.sqrt 2 / 2) * t)

theorem cartesian_equation_of_circle : ∀ x y : ℝ, 
  (polar_to_cartesian_circle (λ θ, 2 * real.sqrt 5 * real.sin θ) (x, y)) → 
  x^2 + y^2 - 2 * real.sqrt 5 * y = 0 :=
by sorry

theorem sum_distances_PA_PB : 
  ∃ A B : ℝ × ℝ, ((parametric_equation_line t = A) ∧ (parametric_equation_line t = B)) → 
  (∃ P : ℝ × ℝ, P = (3, real.sqrt 5) → 
  ∑ (distances : ℝ), abs (distances) = 3 * real.sqrt 2) :=
by sorry

end cartesian_equation_of_circle_sum_distances_PA_PB_l88_88658


namespace find_5b_l88_88354

-- Define variables and conditions
variables (a b : ℝ)
axiom h1 : 6 * a + 3 * b = 0
axiom h2 : a = b - 3

-- State the theorem to prove
theorem find_5b : 5 * b = 10 :=
sorry

end find_5b_l88_88354


namespace solve_for_x_l88_88030

theorem solve_for_x (x : ℝ) (h : (1 / 2) * (1 / 7) * x = 14) : x = 196 :=
by
  sorry

end solve_for_x_l88_88030


namespace necessary_but_not_sufficient_condition_l88_88293
-- Import the required Mathlib library in Lean 4

-- State the equivalent proof problem
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (|a| ≤ 1 → a ≤ 1) ∧ ¬ (a ≤ 1 → |a| ≤ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l88_88293


namespace tan_405_eq_1_l88_88320

theorem tan_405_eq_1 : Real.tan (405 * Real.pi / 180) = 1 := 
by 
  sorry

end tan_405_eq_1_l88_88320


namespace min_value_of_expression_l88_88538

theorem min_value_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 6) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 :=
by
  sorry

end min_value_of_expression_l88_88538


namespace minimum_s_value_l88_88925

theorem minimum_s_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  ∃ (s : ℝ), s = 8 * Real.sqrt 6 ∧ ∀ (x' y' z' : ℝ), (0 < x' ∧ 0 < y' ∧ 0 < z' ∧ 3 * x'^2 + 2 * y'^2 + z'^2 = 1) → 
      s ≤ (1 + z') / (x' * y' * z') :=
sorry

end minimum_s_value_l88_88925


namespace james_weekly_expenses_l88_88156

noncomputable def utility_cost (rent: ℝ):  ℝ := 0.2 * rent
noncomputable def weekly_hours_open (hours_per_day: ℕ) (days_per_week: ℕ): ℕ := hours_per_day * days_per_week
noncomputable def employee_weekly_wages (wage_per_hour: ℝ) (weekly_hours: ℕ): ℝ := wage_per_hour * weekly_hours
noncomputable def total_employee_wages (employees: ℕ) (weekly_wages: ℝ): ℝ := employees * weekly_wages
noncomputable def total_weekly_expenses (rent: ℝ) (utilities: ℝ) (employee_wages: ℝ): ℝ := rent + utilities + employee_wages

theorem james_weekly_expenses : 
  let rent := 1200
  let utility_percentage := 0.2
  let hours_per_day := 16
  let days_per_week := 5
  let employees := 2
  let wage_per_hour := 12.5
  let weekly_hours := weekly_hours_open hours_per_day days_per_week
  let utilities := utility_cost rent
  let employee_wages_per_week := employee_weekly_wages wage_per_hour weekly_hours
  let total_employee_wages_per_week := total_employee_wages employees employee_wages_per_week
  total_weekly_expenses rent utilities total_employee_wages_per_week = 3440 := 
by
  sorry

end james_weekly_expenses_l88_88156


namespace anthony_has_more_pairs_l88_88981

theorem anthony_has_more_pairs (scott_pairs : ℕ) (anthony_pairs : ℕ) (jim_pairs : ℕ) :
  (scott_pairs = 7) →
  (anthony_pairs = 3 * scott_pairs) →
  (jim_pairs = anthony_pairs - 2) →
  (anthony_pairs - jim_pairs = 2) :=
by
  intro h_scott h_anthony h_jim
  sorry

end anthony_has_more_pairs_l88_88981


namespace waiter_earnings_l88_88310

def num_customers : ℕ := 9
def num_no_tip : ℕ := 5
def tip_per_customer : ℕ := 8
def num_tipping_customers := num_customers - num_no_tip

theorem waiter_earnings : num_tipping_customers * tip_per_customer = 32 := by
  sorry

end waiter_earnings_l88_88310
