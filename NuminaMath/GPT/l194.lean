import Mathlib

namespace smaller_number_l194_194798

theorem smaller_number (L S : ℕ) (h₁ : L - S = 2395) (h₂ : L = 6 * S + 15) : S = 476 :=
by
sorry

end smaller_number_l194_194798


namespace employed_population_is_60_percent_l194_194128

def percent_employed (P : ℝ) (E : ℝ) : Prop :=
  ∃ (P_0 : ℝ) (E_male : ℝ) (E_female : ℝ),
    P_0 = P * 0.45 ∧    -- 45 percent of the population are employed males
    E_female = (E * 0.25) * P ∧   -- 25 percent of the employed people are females
    (0.75 * E = 0.45) ∧    -- 75 percent of the employed people are males which equals to 45% of the total population
    E = 0.6            -- 60% of the population are employed

theorem employed_population_is_60_percent (P : ℝ) (E : ℝ):
  percent_employed P E :=
by
  sorry

end employed_population_is_60_percent_l194_194128


namespace increase_in_tire_radius_l194_194666

theorem increase_in_tire_radius
  (r : ℝ)
  (d1 d2 : ℝ)
  (conv_factor : ℝ)
  (original_radius : r = 16)
  (odometer_reading_outbound : d1 = 500)
  (odometer_reading_return : d2 = 485)
  (conversion_factor : conv_factor = 63360) :
  ∃ Δr : ℝ, Δr = 0.33 :=
by
  sorry

end increase_in_tire_radius_l194_194666


namespace difference_of_integers_l194_194167

theorem difference_of_integers : ∃ (x y : ℕ), x + y = 20 ∧ x * y = 96 ∧ (x - y = 4 ∨ y - x = 4) :=
by
  sorry

end difference_of_integers_l194_194167


namespace minimum_participants_l194_194808

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l194_194808


namespace min_value_expression_l194_194161

variable {a b : ℝ}

theorem min_value_expression
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : a + b = 4) : 
  (∃ C, (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) ≥ C) ∧ 
         (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) = C)) ∧ 
         C = 3 :=
  by sorry

end min_value_expression_l194_194161


namespace undefined_sum_slope_y_intercept_of_vertical_line_l194_194202

theorem undefined_sum_slope_y_intercept_of_vertical_line :
  ∀ (C D : ℝ × ℝ), C.1 = 8 → D.1 = 8 → C.2 ≠ D.2 →
  ∃ (m b : ℝ), false :=
by
  intros
  sorry

end undefined_sum_slope_y_intercept_of_vertical_line_l194_194202


namespace seeds_in_small_gardens_l194_194692

theorem seeds_in_small_gardens 
  (total_seeds : ℕ)
  (planted_seeds : ℕ)
  (small_gardens : ℕ)
  (remaining_seeds := total_seeds - planted_seeds) 
  (seeds_per_garden := remaining_seeds / small_gardens) :
  total_seeds = 101 → planted_seeds = 47 → small_gardens = 9 → seeds_per_garden = 6 := by
  sorry

end seeds_in_small_gardens_l194_194692


namespace B_starts_6_hours_after_A_l194_194015

theorem B_starts_6_hours_after_A 
    (A_walk_speed : ℝ) (B_cycle_speed : ℝ) (catch_up_distance : ℝ)
    (hA : A_walk_speed = 10) (hB : B_cycle_speed = 20) (hD : catch_up_distance = 120) :
    ∃ t : ℝ, t = 6 :=
by
  sorry

end B_starts_6_hours_after_A_l194_194015


namespace jack_should_leave_300_in_till_l194_194457

-- Defining the amounts of each type of bill
def num_100_bills := 2
def num_50_bills := 1
def num_20_bills := 5
def num_10_bills := 3
def num_5_bills := 7
def num_1_bills := 27

-- The amount he needs to hand in
def amount_to_hand_in := 142

-- Calculating the total amount in notes
def total_in_notes := 
  (num_100_bills * 100) + 
  (num_50_bills * 50) + 
  (num_20_bills * 20) + 
  (num_10_bills * 10) + 
  (num_5_bills * 5) + 
  (num_1_bills * 1)

-- Calculating the amount to leave in the till
def amount_to_leave := total_in_notes - amount_to_hand_in

-- Proof statement
theorem jack_should_leave_300_in_till :
  amount_to_leave = 300 :=
by sorry

end jack_should_leave_300_in_till_l194_194457


namespace min_value_frac_sum_l194_194462

open Real

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) := 
sorry

end min_value_frac_sum_l194_194462


namespace trigonometric_identity_l194_194511

open Real

theorem trigonometric_identity
  (x : ℝ)
  (h1 : sin x * cos x = 1 / 8)
  (h2 : π / 4 < x)
  (h3 : x < π / 2) :
  cos x - sin x = - (sqrt 3 / 2) :=
sorry

end trigonometric_identity_l194_194511


namespace diminish_to_divisible_l194_194602

-- Definitions based on conditions
def LCM (a b : ℕ) : ℕ := Nat.lcm a b
def numbers : List ℕ := [12, 16, 18, 21, 28]
def lcm_numbers : ℕ := List.foldr LCM 1 numbers
def n : ℕ := 1011
def x : ℕ := 3

-- The proof problem statement
theorem diminish_to_divisible :
  ∃ x : ℕ, n - x = lcm_numbers := sorry

end diminish_to_divisible_l194_194602


namespace sqrt_450_eq_15_sqrt_2_l194_194587

theorem sqrt_450_eq_15_sqrt_2 : Real.sqrt 450 = 15 * Real.sqrt 2 := 
by
  sorry

end sqrt_450_eq_15_sqrt_2_l194_194587


namespace interval_for_systematic_sampling_l194_194352

-- Define the total population size
def total_population : ℕ := 1203

-- Define the sample size
def sample_size : ℕ := 40

-- Define the interval for systematic sampling
def interval (n m : ℕ) : ℕ := (n - (n % m)) / m

-- The proof statement that the interval \( k \) for segmenting is 30
theorem interval_for_systematic_sampling : interval total_population sample_size = 30 :=
by
  show interval 1203 40 = 30
  sorry

end interval_for_systematic_sampling_l194_194352


namespace average_growth_rate_bing_dwen_dwen_l194_194119

noncomputable def sales_growth_rate (v0 v2 : ℕ) (x : ℝ) : Prop :=
  (1 + x) ^ 2 = (v2 : ℝ) / (v0 : ℝ)

theorem average_growth_rate_bing_dwen_dwen :
  ∀ (v0 v2 : ℕ) (x : ℝ),
    v0 = 10000 →
    v2 = 12100 →
    sales_growth_rate v0 v2 x →
    x = 0.1 :=
by
  intros v0 v2 x h₀ h₂ h_growth
  sorry

end average_growth_rate_bing_dwen_dwen_l194_194119


namespace smallest_m_exists_l194_194728

theorem smallest_m_exists : ∃ (m : ℕ), (∀ n : ℕ, (n > 0) → ((10000 * n % 53 = 0) → (m ≤ n))) ∧ (10000 * m % 53 = 0) :=
by
  sorry

end smallest_m_exists_l194_194728


namespace p_sufficient_not_necessary_for_q_l194_194486

noncomputable def p (x : ℝ) : Prop := |x - 3| < 1
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (p x → q x) ∧ (¬ (q x → p x)) := by
  sorry

end p_sufficient_not_necessary_for_q_l194_194486


namespace smallest_n_for_simplest_form_l194_194759

-- Definitions and conditions
def simplest_form_fractions (n : ℕ) :=
  ∀ k : ℕ, 7 ≤ k ∧ k ≤ 31 → Nat.gcd k (n + 2) = 1

-- Problem statement
theorem smallest_n_for_simplest_form :
  ∃ n : ℕ, simplest_form_fractions (n) ∧ ∀ m : ℕ, m < n → ¬ simplest_form_fractions (m) := 
by 
  sorry

end smallest_n_for_simplest_form_l194_194759


namespace peanuts_in_box_l194_194594

variable (original_peanuts : Nat)
variable (additional_peanuts : Nat)

theorem peanuts_in_box (h1 : original_peanuts = 4) (h2 : additional_peanuts = 4) :
  original_peanuts + additional_peanuts = 8 := 
by
  sorry

end peanuts_in_box_l194_194594


namespace line_parallel_l194_194238

theorem line_parallel (a : ℝ) : (∀ x y : ℝ, ax + y = 0) ↔ (x + ay + 1 = 0) → a = 1 ∨ a = -1 := 
sorry

end line_parallel_l194_194238


namespace remainder_when_two_pow_thirty_three_div_nine_l194_194165

-- Define the base and the exponent
def base : ℕ := 2
def exp : ℕ := 33
def modulus : ℕ := 9

-- The main statement to prove
theorem remainder_when_two_pow_thirty_three_div_nine :
  (base ^ exp) % modulus = 8 :=
by
  sorry

end remainder_when_two_pow_thirty_three_div_nine_l194_194165


namespace gcd_of_a_and_b_is_one_l194_194847

theorem gcd_of_a_and_b_is_one {a b : ℕ} (h1 : a > b) (h2 : Nat.gcd (a + b) (a - b) = 1) : Nat.gcd a b = 1 :=
by
  sorry

end gcd_of_a_and_b_is_one_l194_194847


namespace find_m_l194_194966

theorem find_m (m : ℝ) (h : |m - 4| = |2 * m + 7|) : m = -11 ∨ m = -1 :=
sorry

end find_m_l194_194966


namespace pipe_fill_time_without_leak_l194_194766

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : T > 0) 
  (h2 : 1/T - 1/8 = 1/8) :
  T = 4 := 
sorry

end pipe_fill_time_without_leak_l194_194766


namespace number_of_green_balls_l194_194011

theorem number_of_green_balls (b g : ℕ) (h1 : b = 9) (h2 : (b : ℚ) / (b + g) = 3 / 10) : g = 21 :=
sorry

end number_of_green_balls_l194_194011


namespace complement_B_in_A_l194_194593

noncomputable def A : Set ℝ := {x | x < 2}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x < 2}

theorem complement_B_in_A : {x | x ∈ A ∧ x ∉ B} = {x | x ≤ 1} :=
by
  sorry

end complement_B_in_A_l194_194593


namespace motorcyclist_travel_distances_l194_194110

-- Define the total distance traveled in three days
def total_distance : ℕ := 980

-- Define the total distance traveled in the first two days
def first_two_days_distance : ℕ := 725

-- Define the extra distance traveled on the second day compared to the third day
def second_day_extra : ℕ := 123

-- Define the distances traveled on the first, second, and third days respectively
def day_1_distance : ℕ := 347
def day_2_distance : ℕ := 378
def day_3_distance : ℕ := 255

-- Formalize the theorem statement
theorem motorcyclist_travel_distances :
  total_distance = day_1_distance + day_2_distance + day_3_distance ∧
  first_two_days_distance = day_1_distance + day_2_distance ∧
  day_2_distance = day_3_distance + second_day_extra :=
by 
  sorry

end motorcyclist_travel_distances_l194_194110


namespace product_xyz_eq_one_l194_194763

theorem product_xyz_eq_one (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) : x * y * z = 1 := 
sorry

end product_xyz_eq_one_l194_194763


namespace positive_number_property_l194_194480

theorem positive_number_property (x : ℝ) (h : (100 - x) / 100 * x = 16) :
  x = 40 ∨ x = 60 :=
sorry

end positive_number_property_l194_194480


namespace total_cans_l194_194730

theorem total_cans (c o : ℕ) (h1 : c = 8) (h2 : o = 2 * c) : c + o = 24 := by
  sorry

end total_cans_l194_194730


namespace circle_equation_l194_194098

theorem circle_equation (C : ℝ → ℝ → Prop)
  (h₁ : C 1 0)
  (h₂ : C 0 (Real.sqrt 3))
  (h₃ : C (-3) 0) :
  ∃ D E F : ℝ, (∀ x y, C x y ↔ x^2 + y^2 + D * x + E * y + F = 0) ∧ D = 2 ∧ E = 0 ∧ F = -3 := 
by
  sorry

end circle_equation_l194_194098


namespace problem1_problem2_problem3_l194_194513

def point : Type := (ℝ × ℝ)
def vec (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def A : point := (-2, 4)
noncomputable def B : point := (3, -1)
noncomputable def C : point := (-3, -4)

noncomputable def a : point := vec A B
noncomputable def b : point := vec B C
noncomputable def c : point := vec C A

-- Problem 1
theorem problem1 : (3 * a.1 + b.1 - 3 * c.1, 3 * a.2 + b.2 - 3 * c.2) = (6, -42) :=
sorry

-- Problem 2
theorem problem2 : ∃ m n : ℝ, a = (m * b.1 + n * c.1, m * b.2 + n * c.2) ∧ m = -1 ∧ n = -1 :=
sorry

-- Helper function for point addition
def add_point (p1 p2 : point) : point := (p1.1 + p2.1, p1.2 + p2.2)
def scale_point (k : ℝ) (p : point) : point := (k * p.1, k * p.2)

-- problem 3
noncomputable def M : point := add_point (scale_point 3 c) C
noncomputable def N : point := add_point (scale_point (-2) b) C

theorem problem3 : M = (0, 20) ∧ N = (9, 2) ∧ vec M N = (9, -18) :=
sorry

end problem1_problem2_problem3_l194_194513


namespace soap_box_height_l194_194243

theorem soap_box_height
  (carton_length carton_width carton_height : ℕ)
  (soap_length soap_width h : ℕ)
  (max_soap_boxes : ℕ)
  (h_carton_dim : carton_length = 30)
  (h_carton_width : carton_width = 42)
  (h_carton_height : carton_height = 60)
  (h_soap_length : soap_length = 7)
  (h_soap_width : soap_width = 6)
  (h_max_soap_boxes : max_soap_boxes = 360) :
  h = 1 :=
by
  sorry

end soap_box_height_l194_194243


namespace determine_m_l194_194975

theorem determine_m (a b : ℝ) (m : ℝ) :
  (2 * (a ^ 2 - 2 * a * b - b ^ 2) - (a ^ 2 + m * a * b + 2 * b ^ 2)) = a ^ 2 - (4 + m) * a * b - 4 * b ^ 2 →
  ¬(∃ (c : ℝ), (a ^ 2 - (4 + m) * a * b - 4 * b ^ 2) = a ^ 2 + c * (a * b) + k) →
  m = -4 :=
sorry

end determine_m_l194_194975


namespace num_different_configurations_of_lights_l194_194761

-- Definition of initial conditions
def num_rows : Nat := 6
def num_columns : Nat := 6
def possible_switch_states (n : Nat) : Nat := 2^n

-- Problem statement to be verified
theorem num_different_configurations_of_lights :
  let num_configurations := (possible_switch_states num_rows - 1) * (possible_switch_states num_columns - 1) + 1
  num_configurations = 3970 :=
by
  sorry

end num_different_configurations_of_lights_l194_194761


namespace second_player_cannot_prevent_first_l194_194569

noncomputable def player_choice (set_x2_coeff_to_zero : Prop) (first_player_sets : Prop) (second_player_cannot_prevent : Prop) : Prop :=
  ∀ (b : ℝ) (c : ℝ), (set_x2_coeff_to_zero ∧ first_player_sets ∧ second_player_cannot_prevent) → 
  (∀ x : ℝ, x^3 + b * x + c = 0 → ∃! x : ℝ, x^3 + b * x + c = 0)

theorem second_player_cannot_prevent_first (b c : ℝ) :
  player_choice (set_x2_coeff_to_zero := true)
                (first_player_sets := true)
                (second_player_cannot_prevent := true) :=
sorry

end second_player_cannot_prevent_first_l194_194569


namespace exists_infinite_solutions_l194_194988

noncomputable def infinite_solutions_exist (m : ℕ) : Prop := 
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧  (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem exists_infinite_solutions : infinite_solutions_exist 12 :=
  sorry

end exists_infinite_solutions_l194_194988


namespace total_trail_length_l194_194767

-- Definitions based on conditions
variables (a b c d e : ℕ)

-- Conditions
def condition1 : Prop := a + b + c = 36
def condition2 : Prop := b + c + d = 48
def condition3 : Prop := c + d + e = 45
def condition4 : Prop := a + d = 31

-- Theorem statement
theorem total_trail_length (h1 : condition1 a b c) (h2 : condition2 b c d) (h3 : condition3 c d e) (h4 : condition4 a d) : 
  a + b + c + d + e = 81 :=
by 
  sorry

end total_trail_length_l194_194767


namespace percentage_paid_to_x_l194_194177

theorem percentage_paid_to_x (X Y : ℕ) (h₁ : Y = 350) (h₂ : X + Y = 770) :
  (X / Y) * 100 = 120 :=
by
  sorry

end percentage_paid_to_x_l194_194177


namespace not_and_implication_l194_194039

variable (p q : Prop)

theorem not_and_implication : ¬ (p ∧ q) → (¬ p ∨ ¬ q) :=
by
  sorry

end not_and_implication_l194_194039


namespace inequality_always_true_l194_194582

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ a > 1/4 :=
sorry

end inequality_always_true_l194_194582


namespace pickles_per_cucumber_l194_194253

theorem pickles_per_cucumber (jars cucumbers vinegar_initial vinegar_left pickles_per_jar vinegar_per_jar total_pickles_per_cucumber : ℕ) 
    (h1 : jars = 4) 
    (h2 : cucumbers = 10) 
    (h3 : vinegar_initial = 100) 
    (h4 : vinegar_left = 60) 
    (h5 : pickles_per_jar = 12) 
    (h6 : vinegar_per_jar = 10) 
    (h7 : total_pickles_per_cucumber = 4): 
    total_pickles_per_cucumber = (vinegar_initial - vinegar_left) / vinegar_per_jar * pickles_per_jar / cucumbers := 
by 
  sorry

end pickles_per_cucumber_l194_194253


namespace second_pipe_fills_in_15_minutes_l194_194701

theorem second_pipe_fills_in_15_minutes :
  ∀ (x : ℝ),
  (∀ (x : ℝ), (1 / 2 + (7.5 / x)) = 1 → x = 15) :=
by
  intros
  sorry

end second_pipe_fills_in_15_minutes_l194_194701


namespace find_sixth_number_l194_194958

theorem find_sixth_number 
  (A : ℕ → ℝ)
  (h1 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 11 = 60))
  (h2 : ((A 1 + A 2 + A 3 + A 4 + A 5 + A 6) / 6 = 58))
  (h3 : ((A 6 + A 7 + A 8 + A 9 + A 10 + A 11) / 6 = 65)) 
  : A 6 = 78 :=
by
  sorry

end find_sixth_number_l194_194958


namespace smallest_solution_of_equation_l194_194630

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (x^4 - 26 * x^2 + 169 = 0) ∧ x = -Real.sqrt 13 :=
by
  sorry

end smallest_solution_of_equation_l194_194630


namespace smallest_angle_in_icosagon_l194_194191

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end smallest_angle_in_icosagon_l194_194191


namespace distinct_strings_after_operations_l194_194650

def valid_strings (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else valid_strings (n-1) + valid_strings (n-2)

theorem distinct_strings_after_operations :
  valid_strings 10 = 144 := by
  sorry

end distinct_strings_after_operations_l194_194650


namespace problem_1_problem_2_problem_3_l194_194003

theorem problem_1 (x y : ℝ) : x^2 + y^2 + x * y + x + y ≥ -1 / 3 := 
by sorry

theorem problem_2 (x y z : ℝ) : x^2 + y^2 + z^2 + x * y + y * z + z * x + x + y + z ≥ -3 / 8 := 
by sorry

theorem problem_3 (x y z r : ℝ) : x^2 + y^2 + z^2 + r^2 + x * y + x * z + x * r + y * z + y * r + z * r + x + y + z + r ≥ -2 / 5 := 
by sorry

end problem_1_problem_2_problem_3_l194_194003


namespace other_point_on_circle_l194_194266

noncomputable def circle_center_radius (p : ℝ × ℝ) (r : ℝ) : Prop :=
  dist p (0, 0) = r

theorem other_point_on_circle (r : ℝ) (h : r = 16) (point_on_circle : circle_center_radius (16, 0) r) :
  circle_center_radius (-16, 0) r :=
by
  sorry

end other_point_on_circle_l194_194266


namespace related_sequence_exists_l194_194369

theorem related_sequence_exists :
  ∃ b : Fin 5 → ℕ, b = ![11, 10, 9, 8, 7] :=
by
  let a : Fin 5 → ℕ := ![1, 5, 9, 13, 17]
  let b : Fin 5 → ℕ := ![
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 0) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 1) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 2) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 3) / 4,
    (a 0 + a 1 + a 2 + a 3 + a 4 - a 4) / 4
  ]
  existsi b
  sorry

end related_sequence_exists_l194_194369


namespace cindy_marbles_l194_194749

-- Define the initial constants and their values
def initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def number_of_friends : ℕ := 4

-- Define the problem statement in Lean 4
theorem cindy_marbles :
  4 * (initial_marbles - (marbles_per_friend * number_of_friends)) = 720 := by
  sorry

end cindy_marbles_l194_194749


namespace eval_fraction_l194_194225

theorem eval_fraction : (144 : ℕ) = 12 * 12 → (12 ^ 10 / (144 ^ 4) : ℝ) = 144 := by
  intro h
  have h1 : (144 : ℕ) = 12 ^ 2 := by
    exact h
  sorry

end eval_fraction_l194_194225


namespace tabs_in_all_browsers_l194_194443

-- Definitions based on conditions
def windows_per_browser := 3
def tabs_per_window := 10
def number_of_browsers := 2

-- Total tabs calculation
def total_tabs := number_of_browsers * (windows_per_browser * tabs_per_window)

-- Proving the total number of tabs is 60
theorem tabs_in_all_browsers : total_tabs = 60 := by
  sorry

end tabs_in_all_browsers_l194_194443


namespace remainder_ab_mod_n_l194_194927

theorem remainder_ab_mod_n (n : ℕ) (a c : ℤ) (h1 : a * c ≡ 1 [ZMOD n]) (h2 : b = a * c) :
    (a * b % n) = (a % n) :=
  by
  sorry

end remainder_ab_mod_n_l194_194927


namespace peter_has_4_finches_l194_194372

variable (parakeet_eats_per_day : ℕ) (parrot_eats_per_day : ℕ) (finch_eats_per_day : ℕ)
variable (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ)
variable (total_birdseed : ℕ)

theorem peter_has_4_finches
    (h1 : parakeet_eats_per_day = 2)
    (h2 : parrot_eats_per_day = 14)
    (h3 : finch_eats_per_day = 1)
    (h4 : num_parakeets = 3)
    (h5 : num_parrots = 2)
    (h6 : total_birdseed = 266)
    (h7 : total_birdseed = (num_parakeets * parakeet_eats_per_day + num_parrots * parrot_eats_per_day) * 7 + num_finches * finch_eats_per_day * 7) :
    num_finches = 4 :=
by
  sorry

end peter_has_4_finches_l194_194372


namespace segment_length_tangent_circles_l194_194952

theorem segment_length_tangent_circles
  (r1 r2 : ℝ)
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (h3 : 7 - 4 * Real.sqrt 3 ≤ r1 / r2)
  (h4 : r1 / r2 ≤ 7 + 4 * Real.sqrt 3)
  :
  ∃ d : ℝ, d^2 = (1 / 12) * (14 * r1 * r2 - r1^2 - r2^2) :=
sorry

end segment_length_tangent_circles_l194_194952


namespace salt_cups_l194_194600

theorem salt_cups (S : ℕ) (h1 : 8 = S + 1) : S = 7 := by
  -- Problem conditions
  -- 1. The recipe calls for 8 cups of sugar.
  -- 2. Mary needs to add 1 more cup of sugar than cups of salt.
  -- This corresponds to h1.

  -- Prove S = 7
  sorry

end salt_cups_l194_194600


namespace student_l194_194985

theorem student's_incorrect_answer (D I : ℕ) (h1 : D / 36 = 58) (h2 : D / 87 = I) : I = 24 :=
sorry

end student_l194_194985


namespace nested_fraction_simplifies_l194_194534

theorem nested_fraction_simplifies : 
  (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = 8 / 21 := 
by 
  sorry

end nested_fraction_simplifies_l194_194534


namespace negation_of_universal_proposition_l194_194567

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x < 0 :=
by sorry

end negation_of_universal_proposition_l194_194567


namespace employees_working_abroad_l194_194286

theorem employees_working_abroad
  (total_employees : ℕ)
  (fraction_abroad : ℝ)
  (h_total : total_employees = 450)
  (h_fraction : fraction_abroad = 0.06) :
  total_employees * fraction_abroad = 27 := 
by
  sorry

end employees_working_abroad_l194_194286


namespace petya_run_time_l194_194375

-- Definitions
def time_petya_4_to_1 : ℕ := 12

-- Conditions
axiom time_mom_condition : ∃ (time_mom : ℕ), time_petya_4_to_1 = time_mom - 2
axiom time_mom_5_to_1_condition : ∃ (time_petya_5_to_1 : ℕ), ∀ time_mom : ℕ, time_mom = time_petya_5_to_1 - 2

-- Proof statement
theorem petya_run_time :
  ∃ (time_petya_4_to_1 : ℕ), time_petya_4_to_1 = 12 :=
sorry

end petya_run_time_l194_194375


namespace probability_of_ace_then_spade_l194_194439

theorem probability_of_ace_then_spade :
  let P := (1 / 52) * (12 / 51) + (3 / 52) * (13 / 51)
  P = (3 / 127) :=
by
  sorry

end probability_of_ace_then_spade_l194_194439


namespace correct_statement_is_A_l194_194242

theorem correct_statement_is_A : 
  (∀ x : ℝ, 0 ≤ x → abs x = x) ∧
  ¬ (∀ x : ℝ, x ≤ 0 → -x = x) ∧
  ¬ (∀ x : ℝ, (x ≠ 0 ∧ x⁻¹ = x) → (x = 1 ∨ x = -1 ∨ x = 0)) ∧
  ¬ (∀ x y : ℝ, x < 0 ∧ y < 0 → abs x < abs y → x < y) :=
by
  sorry

end correct_statement_is_A_l194_194242


namespace minimum_value_of_expr_l194_194068

noncomputable def expr (x : ℝ) : ℝ := x + (1 / (x - 5))

theorem minimum_value_of_expr : ∀ (x : ℝ), x > 5 → expr x ≥ 7 ∧ (expr x = 7 ↔ x = 6) := 
by 
  sorry

end minimum_value_of_expr_l194_194068


namespace triangle_side_ratio_l194_194083

variable (A B C : ℝ)  -- angles in radians
variable (a b c : ℝ)  -- sides of triangle

theorem triangle_side_ratio
  (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) :
  b / a = Real.sqrt 2 :=
by sorry

end triangle_side_ratio_l194_194083


namespace prob_at_least_one_multiple_of_4_60_l194_194037

def num_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def total_numbers_in_range (n : ℕ) : ℕ :=
  n

def num_not_multiples_of_4 (n : ℕ) : ℕ :=
  total_numbers_in_range n - num_multiples_of_4 n

def prob_no_multiple_of_4 (n : ℕ) : ℚ :=
  let p := num_not_multiples_of_4 n / total_numbers_in_range n
  p * p

def prob_at_least_one_multiple_of_4 (n : ℕ) : ℚ :=
  1 - prob_no_multiple_of_4 n

theorem prob_at_least_one_multiple_of_4_60 :
  prob_at_least_one_multiple_of_4 60 = 7 / 16 :=
by
  -- Proof is skipped.
  sorry

end prob_at_least_one_multiple_of_4_60_l194_194037


namespace beds_with_fewer_beds_l194_194295

theorem beds_with_fewer_beds:
  ∀ (total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x : ℕ),
    total_rooms = 13 →
    rooms_with_fewer_beds = 8 →
    rooms_with_three_beds = total_rooms - rooms_with_fewer_beds →
    total_beds = 31 →
    8 * x + 3 * (total_rooms - rooms_with_fewer_beds) = total_beds →
    x = 2 :=
by
  intros total_rooms rooms_with_fewer_beds rooms_with_three_beds total_beds x
  intros ht_rooms hrwb hrwtb htb h_eq
  sorry

end beds_with_fewer_beds_l194_194295


namespace product_of_sums_of_conjugates_l194_194109

theorem product_of_sums_of_conjugates :
  let a := 8 - Real.sqrt 500
  let b := 8 + Real.sqrt 500
  let c := 12 - Real.sqrt 72
  let d := 12 + Real.sqrt 72
  (a + b) * (c + d) = 384 :=
by
  sorry

end product_of_sums_of_conjugates_l194_194109


namespace king_gvidon_descendants_l194_194915

def number_of_sons : ℕ := 5
def number_of_descendants_with_sons : ℕ := 100
def number_of_sons_each : ℕ := 3
def number_of_grandsons : ℕ := number_of_descendants_with_sons * number_of_sons_each

def total_descendants : ℕ := number_of_sons + number_of_grandsons

theorem king_gvidon_descendants : total_descendants = 305 :=
by
  sorry

end king_gvidon_descendants_l194_194915


namespace son_age_l194_194228

-- Defining the variables
variables (S F : ℕ)

-- The conditions
def condition1 : Prop := F = S + 25
def condition2 : Prop := F + 2 = 2 * (S + 2)

-- The statement to be proved
theorem son_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 23 :=
sorry

end son_age_l194_194228


namespace machine_b_finishes_in_12_hours_l194_194953

noncomputable def machine_b_time : ℝ :=
  let rA := 1 / 4  -- rate of Machine A
  let rC := 1 / 6  -- rate of Machine C
  let rTotalTogether := 1 / 2  -- rate of all machines working together
  let rB := (rTotalTogether - rA - rC)  -- isolate the rate of Machine B
  1 / rB  -- time for Machine B to finish the job

theorem machine_b_finishes_in_12_hours : machine_b_time = 12 :=
by
  sorry

end machine_b_finishes_in_12_hours_l194_194953


namespace partition_count_l194_194164

theorem partition_count (A B : Finset ℕ) :
  (∀ n, n ∈ A ∨ n ∈ B) ∧ 
  (∀ n, n ∈ A → 1 ≤ n ∧ n ≤ 9) ∧ 
  (∀ n, n ∈ B → 1 ≤ n ∧ n ≤ 9) ∧ 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
  (8 * A.sum id = B.sum id) ∧ 
  (A.sum id + B.sum id = 45) → 
  ∃! (num_ways : ℕ), num_ways = 3 :=
sorry

end partition_count_l194_194164


namespace geo_seq_a6_eight_l194_194925

-- Definitions based on given conditions
variable (a : ℕ → ℝ) -- the sequence
variable (q : ℝ) -- common ratio
-- Conditions for a_1 * a_3 = 4 and a_4 = 4
def geometric_sequence := ∃ (q : ℝ), ∀ n : ℕ, a (n + 1) = a n * q
def condition1 := a 1 * a 3 = 4
def condition2 := a 4 = 4

-- Proof problem: Prove a_6 = 8 given the conditions above
theorem geo_seq_a6_eight (h1 : condition1 a) (h2 : condition2 a) (hs : geometric_sequence a) : 
  a 6 = 8 :=
sorry

end geo_seq_a6_eight_l194_194925


namespace translation_invariant_line_l194_194994

theorem translation_invariant_line (k : ℝ) :
  (∀ x : ℝ, k * (x - 2) + 5 = k * x + 2) → k = 3 / 2 :=
by
  sorry

end translation_invariant_line_l194_194994


namespace line_ellipse_tangent_l194_194338

theorem line_ellipse_tangent (m : ℝ) : 
  (∀ x y : ℝ, (y = m * x + 2) → (x^2 + (y^2 / 4) = 1)) → m^2 = 0 :=
sorry

end line_ellipse_tangent_l194_194338


namespace find_a_if_odd_function_l194_194652

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + Real.sqrt (a + x^2))

theorem find_a_if_odd_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = - f x a) → a = 1 :=
by
  sorry

end find_a_if_odd_function_l194_194652


namespace value_of_m_l194_194892

theorem value_of_m (x m : ℝ) (h : 2 * x + m - 6 = 0) (hx : x = 1) : m = 4 :=
by
  sorry

end value_of_m_l194_194892


namespace polygon_triangle_even_l194_194590

theorem polygon_triangle_even (n m : ℕ) (h : (3 * m - n) % 2 = 0) : (m + n) % 2 = 0 :=
sorry

noncomputable def number_of_distinct_interior_sides (n m : ℕ) : ℕ :=
(3 * m - n) / 2

noncomputable def number_of_distinct_interior_vertices (n m : ℕ) : ℕ :=
(m - n + 2) / 2

end polygon_triangle_even_l194_194590


namespace hyperbola_midpoint_l194_194974

theorem hyperbola_midpoint :
  ∃ A B : ℝ × ℝ, 
    (A.1^2 - A.2^2 / 9 = 1) ∧ (B.1^2 - B.2^2 / 9 = 1) ∧ 
    ((A.1 + B.1) / 2 = -1) ∧ ((A.2 + B.2) / 2 = -4) :=
sorry

end hyperbola_midpoint_l194_194974


namespace intersection_and_complement_l194_194309

open Set

def A := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B := {x : ℝ | x + 3 ≥ 0}

theorem intersection_and_complement : 
  (A ∩ B = {x | -3 ≤ x ∧ x ≤ -2}) ∧ (compl (A ∩ B) = {x | x < -3 ∨ x > -2}) :=
by
  sorry

end intersection_and_complement_l194_194309


namespace ashok_average_marks_l194_194536

theorem ashok_average_marks (avg_6 : ℝ) (marks_6 : ℝ) (total_sub : ℕ) (sub_6 : ℕ)
  (h1 : avg_6 = 75) (h2 : marks_6 = 80) (h3 : total_sub = 6) (h4 : sub_6 = 5) :
  (avg_6 * total_sub - marks_6) / sub_6 = 74 :=
by
  sorry

end ashok_average_marks_l194_194536


namespace parallelogram_area_increase_l194_194640

theorem parallelogram_area_increase (b h : ℕ) :
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  (A2 - A1) * 100 / A1 = 300 :=
by
  let A1 := b * h
  let b' := 2 * b
  let h' := 2 * h
  let A2 := b' * h'
  sorry

end parallelogram_area_increase_l194_194640


namespace time_after_seconds_l194_194436

def initial_time : Nat := 8 * 60 * 60 -- 8:00:00 a.m. in seconds
def seconds_passed : Nat := 8035
def target_time : Nat := (10 * 60 * 60 + 13 * 60 + 35) -- 10:13:35 in seconds

theorem time_after_seconds : initial_time + seconds_passed = target_time := by
  -- proof skipped
  sorry

end time_after_seconds_l194_194436


namespace oil_leak_l194_194235

theorem oil_leak (a b c : ℕ) (h₁ : a = 6522) (h₂ : b = 11687) (h₃ : c = b - a) : c = 5165 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end oil_leak_l194_194235


namespace range_of_a_l194_194063

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 3 then a - x else a * log x / log 2

theorem range_of_a (a : ℝ) (h : f a 2 < f a 4) : a > -2 := by
  sorry

end range_of_a_l194_194063


namespace cars_sold_proof_l194_194576

noncomputable def total_cars_sold : Nat := 300
noncomputable def perc_audi : ℝ := 0.10
noncomputable def perc_toyota : ℝ := 0.15
noncomputable def perc_acura : ℝ := 0.20
noncomputable def perc_honda : ℝ := 0.18

theorem cars_sold_proof : total_cars_sold * (1 - (perc_audi + perc_toyota + perc_acura + perc_honda)) = 111 := by
  sorry

end cars_sold_proof_l194_194576


namespace scaling_transformation_l194_194770

theorem scaling_transformation:
  ∀ (x y x' y': ℝ), 
  (x^2 + y^2 = 1) ∧ (x' = 5 * x) ∧ (y' = 3 * y) → 
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by intros x y x' y'
   sorry

end scaling_transformation_l194_194770


namespace remainder_div_197_l194_194005

theorem remainder_div_197 (x q : ℕ) (h_pos : 0 < x) (h_div : 100 = q * x + 3) : 197 % x = 3 :=
sorry

end remainder_div_197_l194_194005


namespace petrol_price_increase_l194_194579

variable (P C : ℝ)

/- The original price of petrol is P per unit, and the user consumes C units of petrol.
   The new consumption after a 28.57142857142857% reduction is (5/7) * C units.
   The expenditure remains constant, i.e., P * C = P' * (5/7) * C.
-/
theorem petrol_price_increase (h : P * C = (P * (7/5)) * (5/7) * C) :
  (P * (7/5) - P) / P * 100 = 40 :=
by
  sorry

end petrol_price_increase_l194_194579


namespace jeremy_tylenol_duration_l194_194407

theorem jeremy_tylenol_duration (num_pills : ℕ) (pill_mg : ℕ) (dose_mg : ℕ) (hours_per_dose : ℕ) (hours_per_day : ℕ) 
  (total_tylenol_mg : ℕ := num_pills * pill_mg)
  (num_doses : ℕ := total_tylenol_mg / dose_mg)
  (total_hours : ℕ := num_doses * hours_per_dose) :
  num_pills = 112 → pill_mg = 500 → dose_mg = 1000 → hours_per_dose = 6 → hours_per_day = 24 → 
  total_hours / hours_per_day = 14 := 
by 
  intros; 
  sorry

end jeremy_tylenol_duration_l194_194407


namespace burrito_count_l194_194807

def burrito_orders (wraps beef_fillings chicken_fillings : ℕ) :=
  if wraps = 5 ∧ beef_fillings >= 4 ∧ chicken_fillings >= 3 then 25 else 0

theorem burrito_count : burrito_orders 5 4 3 = 25 := by
  sorry

end burrito_count_l194_194807


namespace domain_of_sqrt_sum_l194_194385

theorem domain_of_sqrt_sum (x : ℝ) (h1 : 3 + x ≥ 0) (h2 : 1 - x ≥ 0) : -3 ≤ x ∧ x ≤ 1 := by
  sorry

end domain_of_sqrt_sum_l194_194385


namespace proof_problem_l194_194075

open Set

variable (U : Set ℕ)
variable (P : Set ℕ)
variable (Q : Set ℕ)

noncomputable def problem_statement : Set ℕ :=
  compl (P ∪ Q) ∩ U

theorem proof_problem :
  U = {1, 2, 3, 4} →
  P = {1, 2} →
  Q = {2, 3} →
  compl (P ∪ Q) ∩ U = {4} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end proof_problem_l194_194075


namespace expected_value_T_l194_194959

def boys_girls_expected_value (M N : ℕ) : ℚ :=
  2 * ((M / (M + N : ℚ)) * (N / (M + N - 1 : ℚ)))

theorem expected_value_T (M N : ℕ) (hM : M = 10) (hN : N = 10) :
  boys_girls_expected_value M N = 20 / 19 :=
by 
  rw [hM, hN]
  sorry

end expected_value_T_l194_194959


namespace addition_terms_correct_l194_194523

def first_seq (n : ℕ) : ℕ := 2 * n + 1
def second_seq (n : ℕ) : ℕ := 5 * n - 1

theorem addition_terms_correct :
  first_seq 10 = 21 ∧ second_seq 10 = 49 ∧
  first_seq 80 = 161 ∧ second_seq 80 = 399 :=
by
  sorry

end addition_terms_correct_l194_194523


namespace largest_of_three_l194_194378

theorem largest_of_three (a b c : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : ab + ac + bc = -8) 
  (h3 : abc = -20) : 
  max a (max b c) = (1 + Real.sqrt 41) / 2 := 
by 
  sorry

end largest_of_three_l194_194378


namespace stratified_sampling_elderly_l194_194762

theorem stratified_sampling_elderly (total_elderly middle_aged young total_sample total_population elderly_to_sample : ℕ) 
  (h1: total_elderly = 30) 
  (h2: middle_aged = 90) 
  (h3: young = 60) 
  (h4: total_sample = 36) 
  (h5: total_population = total_elderly + middle_aged + young) 
  (h6: 1 / 5 * total_elderly = elderly_to_sample)
  : elderly_to_sample = 6 := 
  by 
    sorry

end stratified_sampling_elderly_l194_194762


namespace cylindrical_to_rectangular_l194_194034

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 10) (hθ : θ = 3 * Real.pi / 4) (hz : z = 2) :
    ∃ (x y z' : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (z' = z) ∧ (x = -5 * Real.sqrt 2) ∧ (y = 5 * Real.sqrt 2) ∧ (z' = 2) :=
by
  sorry

end cylindrical_to_rectangular_l194_194034


namespace weight_of_dog_l194_194990

theorem weight_of_dog (k r d : ℕ) (h1 : k + r + d = 30) (h2 : k + r = 2 * d) (h3 : k + d = r) : d = 10 :=
by
  sorry

end weight_of_dog_l194_194990


namespace distinct_distances_l194_194016

theorem distinct_distances (points : Finset (ℝ × ℝ)) (h : points.card = 2016) :
  ∃ s : Finset ℝ, s.card ≥ 45 ∧ ∀ p ∈ points, ∃ q ∈ points, p ≠ q ∧ 
    (s = (points.image (λ r => dist p r)).filter (λ x => x ≠ 0)) :=
by
  sorry

end distinct_distances_l194_194016


namespace det_is_18_l194_194305

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1],
    ![2, 5]]

theorem det_is_18 : det A = 18 := by
  sorry

end det_is_18_l194_194305


namespace sqrt_36_eq_6_cube_root_neg_a_125_l194_194714

theorem sqrt_36_eq_6 : ∀ (x : ℝ), 0 ≤ x ∧ x^2 = 36 → x = 6 :=
by sorry

theorem cube_root_neg_a_125 : ∀ (a y : ℝ), y^3 = - a / 125 → y = - (a^(1/3)) / 5 :=
by sorry

end sqrt_36_eq_6_cube_root_neg_a_125_l194_194714


namespace exists_small_area_triangle_l194_194820

structure LatticePoint where
  x : Int
  y : Int

def isValidPoint (p : LatticePoint) : Prop := 
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

def noThreeCollinear (points : List LatticePoint) : Prop := 
  ∀ (p1 p2 p3 : LatticePoint), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  ((p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y))

def triangleArea (p1 p2 p3 : LatticePoint) : ℝ :=
  0.5 * |(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) : ℝ)|

theorem exists_small_area_triangle
  (points : List LatticePoint)
  (h1 : ∀ p ∈ points, isValidPoint p)
  (h2 : noThreeCollinear points) :
  ∃ (p1 p2 p3 : LatticePoint), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l194_194820


namespace find_number_l194_194688

theorem find_number (x : ℕ) (h : x / 3 = 3) : x = 9 :=
sorry

end find_number_l194_194688


namespace inequality_always_true_l194_194371

-- Definitions from the conditions
variables {a b c : ℝ}
variable (h1 : a < b)
variable (h2 : b < c)
variable (h3 : a + b + c = 0)

-- The statement to prove
theorem inequality_always_true : c * a < c * b :=
by
  -- Proof steps go here.
  sorry

end inequality_always_true_l194_194371


namespace polyhedron_euler_formula_l194_194090

variable (A F S : ℕ)
variable (closed_polyhedron : Prop)

theorem polyhedron_euler_formula (h : closed_polyhedron) : A + 2 = F + S := sorry

end polyhedron_euler_formula_l194_194090


namespace sum_of_squares_l194_194024

theorem sum_of_squares (b j s : ℕ) (h : b + j + s = 34) : b^2 + j^2 + s^2 = 406 :=
sorry

end sum_of_squares_l194_194024


namespace matches_played_by_team_B_from_city_A_l194_194910

-- Define the problem setup, conditions, and the conclusion we need to prove
structure Tournament :=
  (cities : ℕ)
  (teams_per_city : ℕ)

-- Assuming each team except Team A of city A has played a unique number of matches,
-- find the number of matches played by Team B of city A.
theorem matches_played_by_team_B_from_city_A (t : Tournament)
  (unique_match_counts_except_A : ∀ (i j : ℕ), i ≠ j → (i < t.cities → (t.teams_per_city * i ≠ t.teams_per_city * j)) ∧ (i < t.cities - 1 → (t.teams_per_city * i ≠ t.teams_per_city * (t.cities - 1)))) :
  (t.cities = 16) → (t.teams_per_city = 2) → ∃ n, n = 15 :=
by
  sorry

end matches_played_by_team_B_from_city_A_l194_194910


namespace smallest_non_factor_product_of_factors_of_72_l194_194859

theorem smallest_non_factor_product_of_factors_of_72 : 
  ∃ x y : ℕ, x ≠ y ∧ x * y ∣ 72 ∧ ¬ (x * y ∣ 72) ∧ x * y = 32 := 
by
  sorry

end smallest_non_factor_product_of_factors_of_72_l194_194859


namespace estimate_total_children_l194_194893

variables (k m n T : ℕ)

/-- There are k children initially given red ribbons. 
    Then m children are randomly selected, 
    and n of them have red ribbons. -/

theorem estimate_total_children (h : n * T = k * m) : T = k * m / n :=
by sorry

end estimate_total_children_l194_194893


namespace number_of_employees_is_five_l194_194921

theorem number_of_employees_is_five
  (rudy_speed : ℕ)
  (joyce_speed : ℕ)
  (gladys_speed : ℕ)
  (lisa_speed : ℕ)
  (mike_speed : ℕ)
  (average_speed : ℕ)
  (h1 : rudy_speed = 64)
  (h2 : joyce_speed = 76)
  (h3 : gladys_speed = 91)
  (h4 : lisa_speed = 80)
  (h5 : mike_speed = 89)
  (h6 : average_speed = 80) :
  (rudy_speed + joyce_speed + gladys_speed + lisa_speed + mike_speed) / average_speed = 5 :=
by
  sorry

end number_of_employees_is_five_l194_194921


namespace existence_of_indices_l194_194708

theorem existence_of_indices 
  (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) (h4 : 0 < a4) (h5 : 0 < a5) : 
  ∃ (i j k l : Fin 5), 
    (i ≠ j) ∧ (i ≠ k) ∧ (i ≠ l) ∧ (j ≠ k) ∧ (j ≠ l) ∧ (k ≠ l) ∧ 
    |(a1 / a2) - (a3 / a4)| < 1/2 :=
by 
  sorry

end existence_of_indices_l194_194708


namespace total_initial_yield_l194_194529

variable (x y z : ℝ)

theorem total_initial_yield (h1 : 0.4 * x + 0.2 * y = 5) 
                           (h2 : 0.4 * y + 0.2 * z = 10) 
                           (h3 : 0.4 * z + 0.2 * x = 9) 
                           : x + y + z = 40 := 
sorry

end total_initial_yield_l194_194529


namespace sector_radius_l194_194662

theorem sector_radius (A L : ℝ) (hA : A = 240 * Real.pi) (hL : L = 20 * Real.pi) : 
  ∃ r : ℝ, r = 24 :=
by
  sorry

end sector_radius_l194_194662


namespace negation_is_false_l194_194074

-- Define the proposition and its negation
def proposition (x y : ℝ) : Prop := (x > 2 ∧ y > 3) → (x + y > 5)
def negation_proposition (x y : ℝ) : Prop := ¬ proposition x y

-- The proposition and its negation
theorem negation_is_false : ∀ (x y : ℝ), negation_proposition x y = false :=
by sorry

end negation_is_false_l194_194074


namespace guitar_center_discount_is_correct_l194_194644

-- Define the suggested retail price
def retail_price : ℕ := 1000

-- Define the shipping fee of Guitar Center
def shipping_fee : ℕ := 100

-- Define the discount percentage offered by Sweetwater
def sweetwater_discount_rate : ℕ := 10

-- Define the amount saved by buying from the cheaper store
def savings : ℕ := 50

-- Define the discount offered by Guitar Center
def guitar_center_discount : ℕ :=
  retail_price - ((retail_price * (100 - sweetwater_discount_rate) / 100) + savings - shipping_fee)

-- Theorem: Prove that the discount offered by Guitar Center is $150
theorem guitar_center_discount_is_correct : guitar_center_discount = 150 :=
  by
    -- The proof will be filled in based on the given conditions
    sorry

end guitar_center_discount_is_correct_l194_194644


namespace dani_pants_after_5_years_l194_194162

theorem dani_pants_after_5_years :
  ∀ (pairs_per_year : ℕ) (pants_per_pair : ℕ) (initial_pants : ℕ) (years : ℕ),
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants = 50 →
  years = 5 →
  initial_pants + years * (pairs_per_year * pants_per_pair) = 90 :=
by sorry

end dani_pants_after_5_years_l194_194162


namespace quadratic_root_exists_in_range_l194_194453

theorem quadratic_root_exists_in_range :
  ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ x^2 + 3 * x - 5 = 0 := 
by
  sorry

end quadratic_root_exists_in_range_l194_194453


namespace cameron_answers_l194_194693

theorem cameron_answers (q_per_tourist : ℕ := 2) 
  (group_1 : ℕ := 6) 
  (group_2 : ℕ := 11) 
  (group_3 : ℕ := 8) 
  (group_3_inquisitive : ℕ := 1) 
  (group_4 : ℕ := 7) :
  (q_per_tourist * group_1) +
  (q_per_tourist * group_2) +
  (q_per_tourist * (group_3 - group_3_inquisitive)) +
  (q_per_tourist * 3 * group_3_inquisitive) +
  (q_per_tourist * group_4) = 68 :=
by
  sorry

end cameron_answers_l194_194693


namespace mailman_junk_mail_l194_194945

variable (junk_mail_per_house : ℕ) (houses_per_block : ℕ)

theorem mailman_junk_mail (h1 : junk_mail_per_house = 2) (h2 : houses_per_block = 7) :
  junk_mail_per_house * houses_per_block = 14 :=
by
  sorry

end mailman_junk_mail_l194_194945


namespace solve_inequality_system_l194_194186

theorem solve_inequality_system (x : ℝ) :
  (x / 3 + 2 > 0) ∧ (2 * x + 5 ≥ 3) ↔ (x ≥ -1) :=
by
  sorry

end solve_inequality_system_l194_194186


namespace history_homework_time_l194_194326

def total_time := 180
def math_homework := 45
def english_homework := 30
def science_homework := 50
def special_project := 30

theorem history_homework_time : total_time - (math_homework + english_homework + science_homework + special_project) = 25 := by
  sorry

end history_homework_time_l194_194326


namespace decompose_96_l194_194473

theorem decompose_96 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 96) (h4 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) :=
sorry

end decompose_96_l194_194473


namespace smallest_four_digit_integer_l194_194313

theorem smallest_four_digit_integer (n : ℕ) :
  (75 * n ≡ 225 [MOD 450]) ∧ (1000 ≤ n ∧ n < 10000) → n = 1005 :=
sorry

end smallest_four_digit_integer_l194_194313


namespace delaney_missed_bus_time_l194_194773

def busDepartureTime : Nat := 480 -- 8:00 a.m. = 8 * 60 minutes
def travelTime : Nat := 30 -- 30 minutes
def departureFromHomeTime : Nat := 470 -- 7:50 a.m. = 7 * 60 + 50 minutes

theorem delaney_missed_bus_time :
  (departureFromHomeTime + travelTime - busDepartureTime) = 20 :=
by
  -- proof would go here
  sorry

end delaney_missed_bus_time_l194_194773


namespace factorization_identity_l194_194755

theorem factorization_identity (x : ℝ) : 
  3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 :=
by
  sorry

end factorization_identity_l194_194755


namespace correct_ordering_of_f_values_l194_194394

variable {f : ℝ → ℝ}

theorem correct_ordering_of_f_values
  (hf_even : ∀ x : ℝ, f (-x) = f x)
  (hf_decreasing : ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ > f x₂) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end correct_ordering_of_f_values_l194_194394


namespace minimum_sum_of_natural_numbers_with_lcm_2012_l194_194678

/-- 
Prove that the minimum sum of seven natural numbers whose least common multiple is 2012 is 512.
-/

theorem minimum_sum_of_natural_numbers_with_lcm_2012 : 
  ∃ (a b c d e f g : ℕ), Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm a b) c) d) e) f) g = 2012 ∧ (a + b + c + d + e + f + g) = 512 :=
sorry

end minimum_sum_of_natural_numbers_with_lcm_2012_l194_194678


namespace initial_nickels_l194_194622

variable (q0 n0 : Nat)
variable (d_nickels : Nat := 3) -- His dad gave him 3 nickels
variable (final_nickels : Nat := 12) -- Tim has now 12 nickels

theorem initial_nickels (q0 : Nat) (n0 : Nat) (d_nickels : Nat) (final_nickels : Nat) :
  final_nickels = n0 + d_nickels → n0 = 9 :=
by
  sorry

end initial_nickels_l194_194622


namespace edwards_final_money_l194_194821

def small_lawn_rate : ℕ := 8
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def first_garden_rate : ℕ := 10
def second_garden_rate : ℕ := 12
def additional_garden_rate : ℕ := 15

def num_small_lawns : ℕ := 3
def num_medium_lawns : ℕ := 1
def num_large_lawns : ℕ := 1
def num_gardens_cleaned : ℕ := 5

def fuel_expense : ℕ := 10
def equipment_rental_expense : ℕ := 15
def initial_savings : ℕ := 7

theorem edwards_final_money : 
  (num_small_lawns * small_lawn_rate + 
   num_medium_lawns * medium_lawn_rate + 
   num_large_lawns * large_lawn_rate + 
   (first_garden_rate + second_garden_rate + (num_gardens_cleaned - 2) * additional_garden_rate) + 
   initial_savings - 
   (fuel_expense + equipment_rental_expense)) = 100 := 
  by 
  -- The proof goes here
  sorry

end edwards_final_money_l194_194821


namespace equal_cylinder_volumes_l194_194036

theorem equal_cylinder_volumes (x : ℝ) (hx : x > 0) :
  π * (5 + x) ^ 2 * 4 = π * 25 * (4 + x) → x = 35 / 4 :=
by
  sorry

end equal_cylinder_volumes_l194_194036


namespace remainder_t100_mod_7_l194_194880

theorem remainder_t100_mod_7 :
  ∀ T : ℕ → ℕ, (T 1 = 3) →
  (∀ n : ℕ, n > 1 → T n = 3 ^ (T (n - 1))) →
  (T 100 % 7 = 6) :=
by
  intro T h1 h2
  -- sorry to skip the actual proof
  sorry

end remainder_t100_mod_7_l194_194880


namespace cupcakes_left_at_home_correct_l194_194067

-- Definitions of the conditions
def total_cupcakes_baked : ℕ := 53
def boxes_given_away : ℕ := 17
def cupcakes_per_box : ℕ := 3

-- Calculate the total number of cupcakes given away
def total_cupcakes_given_away := boxes_given_away * cupcakes_per_box

-- Calculate the number of cupcakes left at home
def cupcakes_left_at_home := total_cupcakes_baked - total_cupcakes_given_away

-- Prove that the number of cupcakes left at home is 2
theorem cupcakes_left_at_home_correct : cupcakes_left_at_home = 2 := by
  sorry

end cupcakes_left_at_home_correct_l194_194067


namespace total_sharks_l194_194886

-- Define the number of sharks at each beach.
def N : ℕ := 22
def D : ℕ := 4 * N
def H : ℕ := D / 2

-- Proof that the total number of sharks on the three beaches is 154.
theorem total_sharks : N + D + H = 154 := by
  sorry

end total_sharks_l194_194886


namespace minimum_employees_needed_l194_194469

-- Conditions
def water_monitors : ℕ := 95
def air_monitors : ℕ := 80
def soil_monitors : ℕ := 45
def water_and_air : ℕ := 30
def air_and_soil : ℕ := 20
def water_and_soil : ℕ := 15
def all_three : ℕ := 10

-- Theorems/Goals
theorem minimum_employees_needed 
  (water : ℕ := water_monitors)
  (air : ℕ := air_monitors)
  (soil : ℕ := soil_monitors)
  (water_air : ℕ := water_and_air)
  (air_soil : ℕ := air_and_soil)
  (water_soil : ℕ := water_and_soil)
  (all_3 : ℕ := all_three) :
  water + air + soil - water_air - air_soil - water_soil + all_3 = 165 :=
by
  sorry

end minimum_employees_needed_l194_194469


namespace complete_the_square_l194_194314

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l194_194314


namespace smallest_square_length_proof_l194_194481

-- Define square side length required properties
noncomputable def smallest_square_side_length (rect_w rect_h min_side : ℝ) : ℝ :=
  if h : min_side^2 % (rect_w * rect_h) = 0 then min_side 
  else if h : (min_side + 1)^2 % (rect_w * rect_h) = 0 then min_side + 1
  else if h : (min_side + 2)^2 % (rect_w * rect_h) = 0 then min_side + 2
  else if h : (min_side + 3)^2 % (rect_w * rect_h) = 0 then min_side + 3
  else if h : (min_side + 4)^2 % (rect_w * rect_h) = 0 then min_side + 4
  else if h : (min_side + 5)^2 % (rect_w * rect_h) = 0 then min_side + 5
  else if h : (min_side + 6)^2 % (rect_w * rect_h) = 0 then min_side + 6
  else if h : (min_side + 7)^2 % (rect_w * rect_h) = 0 then min_side + 7
  else if h : (min_side + 8)^2 % (rect_w * rect_h) = 0 then min_side + 8
  else if h : (min_side + 9)^2 % (rect_w * rect_h) = 0 then min_side + 9
  else min_side + 2 -- ensuring it can't be less than min_side

-- State the theorem
theorem smallest_square_length_proof : smallest_square_side_length 2 3 10 = 12 :=
by 
  unfold smallest_square_side_length
  norm_num
  sorry

end smallest_square_length_proof_l194_194481


namespace sum_a_b_l194_194002

theorem sum_a_b (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 2) (h_bound : a^b < 500)
  (h_max : ∀ a' b', a' > 0 → b' > 2 → a'^b' < 500 → a'^b' ≤ a^b) :
  a + b = 8 :=
by sorry

end sum_a_b_l194_194002


namespace chipmunk_families_went_away_l194_194155

theorem chipmunk_families_went_away :
  ∀ (total_families left_families went_away_families : ℕ),
  total_families = 86 →
  left_families = 21 →
  went_away_families = total_families - left_families →
  went_away_families = 65 :=
by
  intros total_families left_families went_away_families ht hl hw
  rw [ht, hl] at hw
  exact hw

end chipmunk_families_went_away_l194_194155


namespace range_of_a_l194_194114

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 → Real.exp (a * x) ≥ 2 * Real.log x + x^2 - a * x) ↔ 0 ≤ a :=
sorry

end range_of_a_l194_194114


namespace x_sq_plus_inv_sq_l194_194797

theorem x_sq_plus_inv_sq (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
  sorry

end x_sq_plus_inv_sq_l194_194797


namespace solution_set_for_composed_function_l194_194085

theorem solution_set_for_composed_function :
  ∀ x : ℝ, (∀ y : ℝ, y = 2 * x - 1 → (2 * y - 1) ≥ 1) ↔ x ≥ 1 := by
  sorry

end solution_set_for_composed_function_l194_194085


namespace question1_solution_question2_solution_l194_194876

-- Define the function f for any value of a
def f (a : ℝ) (x : ℝ) : ℝ :=
  abs (x + 1) - abs (a * x - 1)

-- Definition specifically for question (1) setting a = 1
def f1 (x : ℝ) : ℝ :=
  f 1 x

-- Definition of the set for the inequality in (1)
def solution_set_1 : Set ℝ :=
  { x | f1 x > 1 }

-- Theorem for question (1)
theorem question1_solution :
  solution_set_1 = { x : ℝ | x > 1/2 } :=
sorry

-- Condition for question (2)
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  f a x > x

-- Define the interval for x in question (2)
def interval_0_1 (x : ℝ) : Prop :=
  0 < x ∧ x < 1

-- Theorem for question (2)
theorem question2_solution {a : ℝ} :
  (∀ x ∈ {x | interval_0_1 x}, inequality_condition a x) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end question1_solution_question2_solution_l194_194876


namespace eq_d_is_quadratic_l194_194949

def is_quadratic (eq : ℕ → ℤ) : Prop :=
  ∃ a b c, a ≠ 0 ∧ eq 2 = a ∧ eq 1 = b ∧ eq 0 = c

def eq_cond_1 (n : ℕ) : ℤ :=
  match n with
  | 2 => 1  -- x^2 coefficient
  | 1 => 0  -- x coefficient
  | 0 => -1 -- constant term
  | _ => 0

theorem eq_d_is_quadratic : is_quadratic eq_cond_1 :=
  sorry

end eq_d_is_quadratic_l194_194949


namespace amoeba_growth_after_5_days_l194_194076

theorem amoeba_growth_after_5_days : (3 : ℕ)^5 = 243 := by
  sorry

end amoeba_growth_after_5_days_l194_194076


namespace trivia_team_total_members_l194_194234

theorem trivia_team_total_members (x : ℕ) (h1 : 4 ≤ x) (h2 : (x - 4) * 8 = 64) : x = 12 :=
sorry

end trivia_team_total_members_l194_194234


namespace cos_double_angle_zero_l194_194669

variable (θ : ℝ)

-- Conditions
def tan_eq_one : Prop := Real.tan θ = 1

-- Objective
theorem cos_double_angle_zero (h : tan_eq_one θ) : Real.cos (2 * θ) = 0 :=
sorry

end cos_double_angle_zero_l194_194669


namespace distinct_possible_lunches_l194_194500

namespace SchoolCafeteria

def main_courses : List String := ["Hamburger", "Veggie Burger", "Chicken Sandwich", "Pasta"]
def beverages_when_meat_free : List String := ["Water", "Soda"]
def beverages_when_meat : List String := ["Water"]
def snacks : List String := ["Apple Pie", "Fruit Cup"]

-- Count the total number of distinct possible lunches
def count_distinct_lunches : Nat :=
  let count_options (main_course : String) : Nat :=
    if main_course = "Hamburger" ∨ main_course = "Chicken Sandwich" then
      beverages_when_meat.length * snacks.length
    else
      beverages_when_meat_free.length * snacks.length
  (main_courses.map count_options).sum

theorem distinct_possible_lunches : count_distinct_lunches = 12 := by
  sorry

end SchoolCafeteria

end distinct_possible_lunches_l194_194500


namespace minute_hand_rotation_l194_194537

theorem minute_hand_rotation (h : ℕ) (radians_per_rotation : ℝ) : h = 5 → radians_per_rotation = 2 * Real.pi → - (h * radians_per_rotation) = -10 * Real.pi :=
by
  intros h_eq rp_eq
  rw [h_eq, rp_eq]
  sorry

end minute_hand_rotation_l194_194537


namespace bekah_days_left_l194_194053

theorem bekah_days_left 
  (total_pages : ℕ)
  (pages_read : ℕ)
  (pages_per_day : ℕ)
  (remaining_pages : ℕ := total_pages - pages_read)
  (days_left : ℕ := remaining_pages / pages_per_day) :
  total_pages = 408 →
  pages_read = 113 →
  pages_per_day = 59 →
  days_left = 5 :=
by {
  sorry
}

end bekah_days_left_l194_194053


namespace salary_percentage_difference_l194_194634

theorem salary_percentage_difference (A B : ℝ) (h : A = 0.8 * B) :
  (B - A) / A * 100 = 25 :=
sorry

end salary_percentage_difference_l194_194634


namespace equilateral_triangle_area_decrease_l194_194643

theorem equilateral_triangle_area_decrease (A : ℝ) (A' : ℝ) (s s' : ℝ) 
  (h1 : A = 121 * Real.sqrt 3) 
  (h2 : A = (s^2 * Real.sqrt 3) / 4) 
  (h3 : s' = s - 8) 
  (h4 : A' = (s'^2 * Real.sqrt 3) / 4) :
  A - A' = 72 * Real.sqrt 3 := 
by sorry

end equilateral_triangle_area_decrease_l194_194643


namespace totalTaxIsCorrect_l194_194801

-- Define the different income sources
def dividends : ℝ := 50000
def couponIncomeOFZ : ℝ := 40000
def couponIncomeCorporate : ℝ := 30000
def capitalGain : ℝ := (100 * 200) - (100 * 150)

-- Define the tax rates
def taxRateDividends : ℝ := 0.13
def taxRateCorporateBond : ℝ := 0.13
def taxRateCapitalGain : ℝ := 0.13

-- Calculate the tax for each type of income
def taxOnDividends : ℝ := dividends * taxRateDividends
def taxOnCorporateCoupon : ℝ := couponIncomeCorporate * taxRateCorporateBond
def taxOnCapitalGain : ℝ := capitalGain * taxRateCapitalGain

-- Sum of all tax amounts
def totalTax : ℝ := taxOnDividends + taxOnCorporateCoupon + taxOnCapitalGain

-- Prove that total tax equals the calculated figure
theorem totalTaxIsCorrect : totalTax = 11050 := by
  sorry

end totalTaxIsCorrect_l194_194801


namespace green_toads_per_acre_l194_194166

theorem green_toads_per_acre (brown_toads spotted_brown_toads green_toads : ℕ) 
  (h1 : ∀ g, 25 * g = brown_toads) 
  (h2 : spotted_brown_toads = brown_toads / 4) 
  (h3 : spotted_brown_toads = 50) : 
  green_toads = 8 :=
by
  sorry

end green_toads_per_acre_l194_194166


namespace magic_square_x_value_l194_194525

theorem magic_square_x_value 
  (a b c d e f g h : ℤ) 
  (h1 : x + b + c = d + e + c)
  (h2 : x + f + e = a + b + d)
  (h3 : x + e + c = a + g + 19)
  (h4 : b + f + e = a + g + 96) 
  (h5 : 19 = b)
  (h6 : 96 = c)
  (h7 : 1 = f)
  (h8 : a + d + x = b + c + f) : 
    x = 200 :=
by
  sorry

end magic_square_x_value_l194_194525


namespace f_at_1_is_neg7007_l194_194674

variable (a b c : ℝ)

def g (x : ℝ) := x^3 + a * x^2 + x + 10
def f (x : ℝ) := x^4 + x^3 + b * x^2 + 100 * x + c

theorem f_at_1_is_neg7007
  (a b c : ℝ)
  (h1 : ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ g a (r1) = 0 ∧ g a (r2) = 0 ∧ g a (r3) = 0)
  (h2 : ∀ x, f x = 0 → g x = 0) :
  f 1 = -7007 := 
sorry

end f_at_1_is_neg7007_l194_194674


namespace candy_eaten_l194_194524

/--
Given:
- Faye initially had 47 pieces of candy
- Faye ate x pieces the first night
- Faye's sister gave her 40 more pieces
- Now Faye has 62 pieces of candy

We need to prove:
- Faye ate 25 pieces of candy the first night.
-/
theorem candy_eaten (x : ℕ) (h1 : 47 - x + 40 = 62) : x = 25 :=
by
  sorry

end candy_eaten_l194_194524


namespace probability_of_three_correct_deliveries_l194_194856

-- Define a combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the problem with conditions and derive the required probability
theorem probability_of_three_correct_deliveries :
  (combination 5 3) / (factorial 5) = 1 / 12 := by
  sorry

end probability_of_three_correct_deliveries_l194_194856


namespace find_x_l194_194446

def op (a b : ℕ) : ℕ := a * b - b + b ^ 2

theorem find_x (x : ℕ) : (∃ x : ℕ, op x 8 = 80) :=
  sorry

end find_x_l194_194446


namespace ab_abs_value_l194_194908

theorem ab_abs_value {a b : ℤ} (ha : a ≠ 0) (hb : b ≠ 0)
  (hroots : ∃ r s : ℤ, (x - r)^2 * (x - s) = x^3 + a * x^2 + b * x + 9 * a) :
  |a * b| = 1344 := 
sorry

end ab_abs_value_l194_194908


namespace student_question_choice_l194_194236

/-- A student needs to choose 8 questions from part A and 5 questions from part B. Both parts contain 10 questions each.
   This Lean statement proves that the student can choose the questions in 11340 different ways. -/
theorem student_question_choice : (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by
  sorry

end student_question_choice_l194_194236


namespace find_smaller_number_l194_194772

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  -- Proof steps will be filled in here
  sorry

end find_smaller_number_l194_194772


namespace smallest_positive_integer_n_l194_194108

theorem smallest_positive_integer_n (n : ℕ) (h : 5 * n ≡ 1463 [MOD 26]) : n = 23 :=
sorry

end smallest_positive_integer_n_l194_194108


namespace heart_ratio_correct_l194_194609

def heart (n m : ℕ) : ℕ := n^3 + m^2

theorem heart_ratio_correct : (heart 3 5 : ℚ) / (heart 5 3) = 26 / 67 :=
by
  sorry

end heart_ratio_correct_l194_194609


namespace percentage_of_students_absent_l194_194979

theorem percentage_of_students_absent (total_students : ℕ) (students_present : ℕ) 
(h_total : total_students = 50) (h_present : students_present = 43)
(absent_students := total_students - students_present) :
((absent_students : ℝ) / total_students) * 100 = 14 :=
by sorry

end percentage_of_students_absent_l194_194979


namespace cubes_with_red_face_l194_194731

theorem cubes_with_red_face :
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  redFaceCubes = 488 :=
by
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  sorry

end cubes_with_red_face_l194_194731


namespace factorize_difference_of_squares_l194_194917

-- We are proving that the factorization of m^2 - 9 is equal to (m+3)(m-3)
theorem factorize_difference_of_squares (m : ℝ) : m ^ 2 - 9 = (m + 3) * (m - 3) := 
by 
  sorry

end factorize_difference_of_squares_l194_194917


namespace constant_S13_l194_194413

-- Defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Defining the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (List.range n).map a |> List.sum

-- Defining the given conditions as hypotheses
variable {a : ℕ → ℤ} {d : ℤ}
variable (h_arith : arithmetic_sequence a d)
variable (constant_sum : a 2 + a 4 + a 15 = k)

-- Goal to prove: S_13 is a constant
theorem constant_S13 (k : ℤ) :
  sum_first_n_terms a 13 = k :=
  sorry

end constant_S13_l194_194413


namespace trade_ratio_blue_per_red_l194_194422

-- Define the problem conditions
def initial_total_marbles : ℕ := 10
def blue_percentage : ℕ := 40
def kept_red_marbles : ℕ := 1
def final_total_marbles : ℕ := 15

-- Find the number of blue marbles initially
def initial_blue_marbles : ℕ := (blue_percentage * initial_total_marbles) / 100

-- Calculate the number of red marbles initially
def initial_red_marbles : ℕ := initial_total_marbles - initial_blue_marbles

-- Calculate the number of red marbles traded
def traded_red_marbles : ℕ := initial_red_marbles - kept_red_marbles

-- Calculate the number of marbles received from the trade
def traded_marbles : ℕ := final_total_marbles - (initial_blue_marbles + kept_red_marbles)

-- The number of blue marbles received per each red marble traded
def blue_per_red : ℕ := traded_marbles / traded_red_marbles

-- Theorem stating that Pete's friend trades 2 blue marbles for each red marble
theorem trade_ratio_blue_per_red : blue_per_red = 2 := by
  -- Proof steps would go here
  sorry

end trade_ratio_blue_per_red_l194_194422


namespace sum_black_cells_even_l194_194610

-- Define a rectangular board with cells colored in a chess manner.

structure ChessBoard (m n : ℕ) :=
  (cells : Fin m → Fin n → Int)
  (row_sums_even : ∀ i : Fin m, (Finset.univ.sum (λ j => cells i j)) % 2 = 0)
  (column_sums_even : ∀ j : Fin n, (Finset.univ.sum (λ i => cells i j)) % 2 = 0)

def is_black_cell (i j : ℕ) : Bool :=
  (i + j) % 2 = 0

theorem sum_black_cells_even {m n : ℕ} (B : ChessBoard m n) :
    (Finset.univ.sum (λ (i : Fin m) =>
         Finset.univ.sum (λ (j : Fin n) =>
            if (is_black_cell i.val j.val) then B.cells i j else 0))) % 2 = 0 :=
by
  sorry

end sum_black_cells_even_l194_194610


namespace sequence_12th_term_l194_194464

theorem sequence_12th_term (C : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = C / a n) (h4 : C = 12) : a 12 = 4 :=
sorry

end sequence_12th_term_l194_194464


namespace log_eq_solution_l194_194386

open Real

theorem log_eq_solution (x : ℝ) (h : x > 0) : log x + log (x + 1) = 2 ↔ x = (-1 + sqrt 401) / 2 :=
by
  sorry

end log_eq_solution_l194_194386


namespace replaced_person_is_65_l194_194038

-- Define the conditions of the problem context
variable (W : ℝ)
variable (avg_increase : ℝ := 3.5)
variable (num_persons : ℕ := 8)
variable (new_person_weight : ℝ := 93)

-- Express the given condition in the problem: 
-- The total increase in weight is given by the number of persons multiplied by the average increase in weight
def total_increase : ℝ := num_persons * avg_increase

-- Express the relationship between the new person's weight and the person who was replaced
def replaced_person_weight (W : ℝ) : ℝ := new_person_weight - total_increase

-- Stating the theorem to be proved
theorem replaced_person_is_65 : replaced_person_weight W = 65 := by
  sorry

end replaced_person_is_65_l194_194038


namespace calculate_expression_l194_194137

theorem calculate_expression (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 26) :
  6 * (p^4 + q^4 + r^4 + s^4) - (p^3 + q^3 + r^3 + s^3) =
    6 * ((p-1)^4 + (q-1)^4 + (r-1)^4 + (s-1)^4) - ((p-1)^3 + (q-1)^3 + (r-1)^3 + (s-1)^3) :=
by {
  sorry
}

end calculate_expression_l194_194137


namespace range_of_k_for_empty_solution_set_l194_194976

theorem range_of_k_for_empty_solution_set :
  ∀ (k : ℝ), (∀ (x : ℝ), k * x^2 - 2 * |x - 1| + 3 * k < 0 → False) ↔ k ≥ 1 :=
by sorry

end range_of_k_for_empty_solution_set_l194_194976


namespace constant_term_expansion_l194_194458

theorem constant_term_expansion (x : ℝ) (n : ℕ) (h : (x + 2 + 1/x)^n = 20) : n = 3 :=
by
sorry

end constant_term_expansion_l194_194458


namespace partition_nats_100_subsets_l194_194918

theorem partition_nats_100_subsets :
  ∃ (S : ℕ → ℕ), (∀ n, 1 ≤ S n ∧ S n ≤ 100) ∧
    (∀ a b c : ℕ, a + 99 * b = c → S a = S c ∨ S a = S b ∨ S b = S c) :=
by
  sorry

end partition_nats_100_subsets_l194_194918


namespace Laran_large_posters_daily_l194_194791

/-
Problem statement:
Laran has started a poster business. She is selling 5 posters per day at school. Some posters per day are her large posters that sell for $10. The large posters cost her $5 to make. The remaining posters are small posters that sell for $6. They cost $3 to produce. Laran makes a profit of $95 per 5-day school week. How many large posters does Laran sell per day?
-/

/-
Mathematically equivalent proof problem:
Prove that the number of large posters Laran sells per day is 2, given the following conditions:
1) L + S = 5
2) 5L + 3S = 19
-/

variables (L S : ℕ)

-- Given conditions
def condition1 := L + S = 5
def condition2 := 5 * L + 3 * S = 19

-- Prove the desired statement
theorem Laran_large_posters_daily 
    (h1 : condition1 L S) 
    (h2 : condition2 L S) : 
    L = 2 := 
sorry

end Laran_large_posters_daily_l194_194791


namespace johns_piano_total_cost_l194_194936

theorem johns_piano_total_cost : 
  let piano_cost := 500
  let original_lessons_cost := 20 * 40
  let discount := (25 / 100) * original_lessons_cost
  let discounted_lessons_cost := original_lessons_cost - discount
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  total_cost = 1275 := 
by
  let piano_cost := 500
  let original_lessons_cost := 800
  let discount := 200
  let discounted_lessons_cost := 600
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  -- Proof skipped
  sorry

end johns_piano_total_cost_l194_194936


namespace width_at_bottom_of_stream_l194_194964

theorem width_at_bottom_of_stream 
    (top_width : ℝ) (area : ℝ) (height : ℝ) (bottom_width : ℝ) :
    top_width = 10 → area = 640 → height = 80 → 
    2 * area = height * (top_width + bottom_width) → 
    bottom_width = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Finding bottom width
  have h5 : 2 * 640 = 80 * (10 + bottom_width) := h4
  norm_num at h5
  linarith [h5]

#check width_at_bottom_of_stream

end width_at_bottom_of_stream_l194_194964


namespace log_eq_one_l194_194796

theorem log_eq_one (log : ℝ → ℝ) (h1 : ∀ a b, log (a ^ b) = b * log a) (h2 : ∀ a b, log (a * b) = log a + log b) :
  (log 5) ^ 2 + log 2 * log 50 = 1 :=
sorry

end log_eq_one_l194_194796


namespace golden_section_PB_l194_194265

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem golden_section_PB {A B P : ℝ} (h1 : P = (1 - 1/(golden_ratio)) * A + (1/(golden_ratio)) * B)
  (h2 : AB = 2)
  (h3 : A ≠ B) : PB = 3 - Real.sqrt 5 :=
by
  sorry

end golden_section_PB_l194_194265


namespace problem_statement_l194_194334

variable (a b c : ℝ)
variable (x : ℝ)

theorem problem_statement (h : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |a * x^2 - b * x + c| < 1) :
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, |(a + b) * x^2 + c| < 1 :=
by
  intros x hx
  let f := fun x => a * x^2 - b * x + c
  let g := fun x => (a + b) * x^2 + c
  have h1 : ∀ x ∈ Set.Icc (-1 : ℝ) 1, |f x| < 1 := h
  sorry

end problem_statement_l194_194334


namespace bill_bathroom_visits_per_day_l194_194922

theorem bill_bathroom_visits_per_day
  (squares_per_use : ℕ)
  (rolls : ℕ)
  (squares_per_roll : ℕ)
  (days_supply : ℕ)
  (total_uses : squares_per_use = 5)
  (total_rolls : rolls = 1000)
  (squares_from_each_roll : squares_per_roll = 300)
  (total_days : days_supply = 20000) :
  ( (rolls * squares_per_roll) / days_supply / squares_per_use ) = 3 :=
by
  sorry

end bill_bathroom_visits_per_day_l194_194922


namespace rank_of_A_l194_194817

def A : Matrix (Fin 3) (Fin 5) ℝ :=
  ![![1, 2, 3, 5, 8],
    ![0, 1, 4, 6, 9],
    ![0, 0, 1, 7, 10]]

theorem rank_of_A : A.rank = 3 :=
by sorry

end rank_of_A_l194_194817


namespace jelly_bean_probabilities_l194_194106

theorem jelly_bean_probabilities :
  let p_red := 0.15
  let p_orange := 0.35
  let p_yellow := 0.2
  let p_green := 0.3
  p_red + p_orange + p_yellow + p_green = 1 :=
by
  sorry

end jelly_bean_probabilities_l194_194106


namespace rope_length_l194_194819

theorem rope_length (x S : ℝ) (H1 : x + 7 * S = 140)
(H2 : x - S = 20) : x = 35 := by
sorry

end rope_length_l194_194819


namespace quadratic_no_real_roots_l194_194818

theorem quadratic_no_real_roots (a b c d : ℝ)  :
  a^2 - 4 * b < 0 → c^2 - 4 * d < 0 → ( (a + c) / 2 )^2 - 4 * ( (b + d) / 2 ) < 0 :=
by
  sorry

end quadratic_no_real_roots_l194_194818


namespace a_5_eq_31_l194_194552

def seq (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (∀ n, a (n + 1) = 2 * a n + 1)

theorem a_5_eq_31 (a : ℕ → ℕ) (h : seq a) : a 5 = 31 :=
by
  sorry
 
end a_5_eq_31_l194_194552


namespace fewest_number_of_students_l194_194613

def satisfiesCongruences (n : ℕ) : Prop :=
  n % 6 = 3 ∧
  n % 7 = 4 ∧
  n % 8 = 5 ∧
  n % 9 = 2

theorem fewest_number_of_students : ∃ n : ℕ, satisfiesCongruences n ∧ n = 765 :=
by
  have h_ex : ∃ n : ℕ, satisfiesCongruences n := sorry
  obtain ⟨n, hn⟩ := h_ex
  use 765
  have h_correct : satisfiesCongruences 765 := sorry
  exact ⟨h_correct, rfl⟩

end fewest_number_of_students_l194_194613


namespace smallest_number_of_eggs_proof_l194_194526

noncomputable def smallest_number_of_eggs (c : ℕ) : ℕ := 15 * c - 3

theorem smallest_number_of_eggs_proof :
  ∃ c : ℕ, c ≥ 11 ∧ smallest_number_of_eggs c = 162 ∧ smallest_number_of_eggs c > 150 :=
by
  sorry

end smallest_number_of_eggs_proof_l194_194526


namespace sqrt_14400_eq_120_l194_194771

theorem sqrt_14400_eq_120 : Real.sqrt 14400 = 120 :=
by
  sorry

end sqrt_14400_eq_120_l194_194771


namespace negation_of_proposition_l194_194163

theorem negation_of_proposition (x : ℝ) :
  ¬ (x > 1 → x ^ 2 > x) ↔ (x ≤ 1 → x ^ 2 ≤ x) :=
by 
  sorry

end negation_of_proposition_l194_194163


namespace bus_seats_capacity_l194_194170

-- Define the conditions
variable (x : ℕ) -- number of people each seat can hold
def left_side_seats := 15
def right_side_seats := left_side_seats - 3
def back_seat_capacity := 7
def total_capacity := left_side_seats * x + right_side_seats * x + back_seat_capacity

-- State the theorem
theorem bus_seats_capacity :
  total_capacity x = 88 → x = 3 := by
  sorry

end bus_seats_capacity_l194_194170


namespace expressions_equal_constant_generalized_identity_l194_194907

noncomputable def expr1 := (Real.sin (13 * Real.pi / 180))^2 + (Real.cos (17 * Real.pi / 180))^2 - Real.sin (13 * Real.pi / 180) * Real.cos (17 * Real.pi / 180)
noncomputable def expr2 := (Real.sin (15 * Real.pi / 180))^2 + (Real.cos (15 * Real.pi / 180))^2 - Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
noncomputable def expr3 := (Real.sin (-18 * Real.pi / 180))^2 + (Real.cos (48 * Real.pi / 180))^2 - Real.sin (-18 * Real.pi / 180) * Real.cos (48 * Real.pi / 180)
noncomputable def expr4 := (Real.sin (-25 * Real.pi / 180))^2 + (Real.cos (55 * Real.pi / 180))^2 - Real.sin (-25 * Real.pi / 180) * Real.cos (55 * Real.pi / 180)

theorem expressions_equal_constant :
  expr1 = 3/4 ∧ expr2 = 3/4 ∧ expr3 = 3/4 ∧ expr4 = 3/4 :=
sorry

theorem generalized_identity (α : ℝ) :
  (Real.sin α)^2 + (Real.cos (30 * Real.pi / 180 - α))^2 - Real.sin α * Real.cos (30 * Real.pi / 180 - α) = 3 / 4 :=
sorry

end expressions_equal_constant_generalized_identity_l194_194907


namespace largest_in_set_average_11_l194_194608

theorem largest_in_set_average_11 :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), (a_1 < a_2) ∧ (a_2 < a_3) ∧ (a_3 < a_4) ∧ (a_4 < a_5) ∧
  (1 ≤ a_1 ∧ 1 ≤ a_2 ∧ 1 ≤ a_3 ∧ 1 ≤ a_4 ∧ 1 ≤ a_5) ∧
  (a_1 + a_2 + a_3 + a_4 + a_5 = 55) ∧
  (a_5 = 45) := 
sorry

end largest_in_set_average_11_l194_194608


namespace auditorium_rows_l194_194095

theorem auditorium_rows (x : ℕ) (hx : (320 / x + 4) * (x + 1) = 420) : x = 20 :=
by
  sorry

end auditorium_rows_l194_194095


namespace geese_count_l194_194642

-- Define the number of ducks in the marsh
def number_of_ducks : ℕ := 37

-- Define the total number of birds in the marsh
def total_number_of_birds : ℕ := 95

-- Define the number of geese in the marsh
def number_of_geese : ℕ := total_number_of_birds - number_of_ducks

-- Theorem stating the number of geese in the marsh is 58
theorem geese_count : number_of_geese = 58 := by
  sorry

end geese_count_l194_194642


namespace length_of_unfenced_side_l194_194182

theorem length_of_unfenced_side :
  ∃ L W : ℝ, L * W = 320 ∧ 2 * W + L = 56 ∧ L = 40 :=
by
  sorry

end length_of_unfenced_side_l194_194182


namespace speed_of_boat_in_still_water_l194_194782

theorem speed_of_boat_in_still_water :
  ∀ (v : ℚ), (33 = (v + 3) * (44 / 60)) → v = 42 := 
by
  sorry

end speed_of_boat_in_still_water_l194_194782


namespace sally_initial_orange_balloons_l194_194929

variable (initial_orange_balloons : ℕ)  -- The initial number of orange balloons Sally had
variable (lost_orange_balloons : ℕ := 2)  -- The number of orange balloons Sally lost
variable (current_orange_balloons : ℕ := 7)  -- The number of orange balloons Sally currently has

theorem sally_initial_orange_balloons : 
  current_orange_balloons + lost_orange_balloons = initial_orange_balloons := 
by
  sorry

end sally_initial_orange_balloons_l194_194929


namespace preston_receives_total_amount_l194_194472

theorem preston_receives_total_amount :
  let price_per_sandwich := 5
  let delivery_fee := 20
  let num_sandwiches := 18
  let tip_percent := 0.10
  let sandwich_cost := num_sandwiches * price_per_sandwich
  let initial_total := sandwich_cost + delivery_fee
  let tip := initial_total * tip_percent
  let final_total := initial_total + tip
  final_total = 121 := 
by
  sorry

end preston_receives_total_amount_l194_194472


namespace jaguars_total_games_l194_194176

-- Defining constants for initial conditions
def initial_win_rate : ℚ := 0.55
def additional_wins : ℕ := 8
def additional_losses : ℕ := 2
def final_win_rate : ℚ := 0.6

-- Defining the main problem statement
theorem jaguars_total_games : 
  ∃ y x : ℕ, (x = initial_win_rate * y) ∧ (x + additional_wins = final_win_rate * (y + (additional_wins + additional_losses))) ∧ (y + (additional_wins + additional_losses) = 50) :=
sorry

end jaguars_total_games_l194_194176


namespace distance_Xiaolan_to_Xiaohong_reverse_l194_194935

def Xiaohong_to_Xiaolan := 30
def Xiaolu_to_Xiaohong := 26
def Xiaolan_to_Xiaolu := 28

def total_perimeter : ℕ := Xiaohong_to_Xiaolan + Xiaolan_to_Xiaolu + Xiaolu_to_Xiaohong

theorem distance_Xiaolan_to_Xiaohong_reverse : total_perimeter - Xiaohong_to_Xiaolan = 54 :=
by
  rw [total_perimeter]
  norm_num
  sorry

end distance_Xiaolan_to_Xiaohong_reverse_l194_194935


namespace ways_to_place_7_balls_in_3_boxes_l194_194244

theorem ways_to_place_7_balls_in_3_boxes : ∃ n : ℕ, n = 8 ∧ (∀ x y z : ℕ, x + y + z = 7 → x ≥ y → y ≥ z → z ≥ 0) := 
by
  sorry

end ways_to_place_7_balls_in_3_boxes_l194_194244


namespace cost_comparison_cost_effectiveness_47_l194_194648

section
variable (x : ℕ)

-- Conditions
def price_teapot : ℕ := 25
def price_teacup : ℕ := 5
def quantity_teapots : ℕ := 4
def discount_scheme_2 : ℝ := 0.94

-- Total cost for Scheme 1
def cost_scheme_1 (x : ℕ) : ℕ :=
  (quantity_teapots * price_teapot) + (price_teacup * (x - quantity_teapots))

-- Total cost for Scheme 2
def cost_scheme_2 (x : ℕ) : ℝ :=
  (quantity_teapots * price_teapot + price_teacup * x : ℝ) * discount_scheme_2

-- The proof problem
theorem cost_comparison (x : ℕ) (h : x ≥ 4) :
  cost_scheme_1 x = 5 * x + 80 ∧ cost_scheme_2 x = 4.7 * x + 94 :=
sorry

-- When x = 47
theorem cost_effectiveness_47 : cost_scheme_2 47 < cost_scheme_1 47 :=
sorry

end

end cost_comparison_cost_effectiveness_47_l194_194648


namespace Fran_speed_l194_194027

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end Fran_speed_l194_194027


namespace two_pow_2023_mod_17_l194_194942

theorem two_pow_2023_mod_17 : (2 ^ 2023) % 17 = 4 := 
by
  sorry

end two_pow_2023_mod_17_l194_194942


namespace number_of_female_democrats_l194_194903

-- Definitions and conditions
variables (F M D_F D_M D_T : ℕ)
axiom participant_total : F + M = 780
axiom female_democrats : D_F = 1 / 2 * F
axiom male_democrats : D_M = 1 / 4 * M
axiom total_democrats : D_T = 1 / 3 * (F + M)

-- Target statement to be proven
theorem number_of_female_democrats : D_T = 260 → D_F = 130 :=
by
  intro h
  sorry

end number_of_female_democrats_l194_194903


namespace moles_of_ammonium_nitrate_formed_l194_194285

def ammonia := ℝ
def nitric_acid := ℝ
def ammonium_nitrate := ℝ

-- Define the stoichiometric coefficients from the balanced equation.
def stoichiometric_ratio_ammonia : ℝ := 1
def stoichiometric_ratio_nitric_acid : ℝ := 1
def stoichiometric_ratio_ammonium_nitrate : ℝ := 1

-- Define the initial moles of reactants.
def initial_moles_ammonia (moles : ℝ) : Prop := moles = 3
def initial_moles_nitric_acid (moles : ℝ) : Prop := moles = 3

-- The reaction goes to completion as all reactants are used:
theorem moles_of_ammonium_nitrate_formed :
  ∀ (moles_ammonia moles_nitric_acid : ℝ),
    initial_moles_ammonia moles_ammonia →
    initial_moles_nitric_acid moles_nitric_acid →
    (moles_ammonia / stoichiometric_ratio_ammonia) = 
    (moles_nitric_acid / stoichiometric_ratio_nitric_acid) →
    (moles_ammonia / stoichiometric_ratio_ammonia) * stoichiometric_ratio_ammonium_nitrate = 3 :=
by
  intros moles_ammonia moles_nitric_acid h_ammonia h_nitric_acid h_ratio
  rw [h_ammonia, h_nitric_acid] at *
  simp only [stoichiometric_ratio_ammonia, stoichiometric_ratio_nitric_acid, stoichiometric_ratio_ammonium_nitrate] at *
  sorry

end moles_of_ammonium_nitrate_formed_l194_194285


namespace problem_l194_194972

theorem problem (A B : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : B > A) (n c : ℝ) 
  (h₃ : B = A * (1 + n / 100)) (h₄ : A = B * (1 - c / 100)) :
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) :=
by
  sorry

end problem_l194_194972


namespace boat_speed_in_still_water_l194_194327

-- Boat's speed in still water in km/hr
variable (B S : ℝ)

-- Conditions given for the boat's speed along and against the stream
axiom cond1 : B + S = 11
axiom cond2 : B - S = 5

-- Prove that the speed of the boat in still water is 8 km/hr
theorem boat_speed_in_still_water : B = 8 :=
by
  sorry

end boat_speed_in_still_water_l194_194327


namespace teddy_bears_per_shelf_l194_194260

theorem teddy_bears_per_shelf :
  (98 / 14 = 7) := 
by
  sorry

end teddy_bears_per_shelf_l194_194260


namespace initially_calculated_average_weight_l194_194399

theorem initially_calculated_average_weight 
  (A : ℚ)
  (h1 : ∀ sum_weight_corr : ℚ, sum_weight_corr = 20 * 58.65)
  (h2 : ∀ misread_weight_corr : ℚ, misread_weight_corr = 56)
  (h3 : ∀ correct_weight_corr : ℚ, correct_weight_corr = 61)
  (h4 : (20 * A + (correct_weight_corr - misread_weight_corr)) = 20 * 58.65) :
  A = 58.4 := 
sorry

end initially_calculated_average_weight_l194_194399


namespace trapezoid_area_division_l194_194066

/-- Given a trapezoid where one base is 150 units longer than the other base and the segment joining the midpoints of the legs divides the trapezoid into two regions whose areas are in the ratio 3:4, prove that the greatest integer less than or equal to (x^2 / 150) is 300, where x is the length of the segment that joins the midpoints of the legs and divides the trapezoid into two equal areas. -/
theorem trapezoid_area_division (b h x : ℝ) (h_b : b = 112.5) (h_x : x = 150) :
  ⌊x^2 / 150⌋ = 300 :=
by
  sorry

end trapezoid_area_division_l194_194066


namespace sin_of_300_degrees_l194_194351

theorem sin_of_300_degrees : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3) / 2 := by
  sorry

end sin_of_300_degrees_l194_194351


namespace max_integer_a_for_real_roots_l194_194810

theorem max_integer_a_for_real_roots (a : ℤ) :
  (((a - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1) → a ≤ 0 ∧ (∀ b : ℤ, ((b - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1 → b ≤ 0) :=
sorry

end max_integer_a_for_real_roots_l194_194810


namespace solve_equation_2021_l194_194913

theorem solve_equation_2021 (x : ℝ) (hx : 0 ≤ x) : 
  2021 * x = 2022 * (x ^ (2021 : ℕ)) ^ (1 / (2021 : ℕ)) - 1 → x = 1 := 
by
  sorry

end solve_equation_2021_l194_194913


namespace parallel_lines_slope_l194_194425

theorem parallel_lines_slope {m : ℝ} : 
  (∃ m, (∀ x y : ℝ, 3 * x + 4 * y - 3 = 0 → 6 * x + m * y + 14 = 0)) ↔ m = 8 :=
by
  sorry

end parallel_lines_slope_l194_194425


namespace find_angle_C_find_sin_A_plus_sin_B_l194_194515

open Real

namespace TriangleProblem

variables (a b c : ℝ) (A B C : ℝ)

def sides_opposite_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  c^2 = a^2 + b^2 + a * b

def given_c (c : ℝ) : Prop :=
  c = 4 * sqrt 7

def perimeter (a b c : ℝ) : Prop :=
  a + b + c = 12 + 4 * sqrt 7

theorem find_angle_C (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C) : 
  C = 2 * pi / 3 :=
sorry

theorem find_sin_A_plus_sin_B (a b c A B C : ℝ)
  (h1 : sides_opposite_angles a b c A B C)
  (h2 : given_c c)
  (h3 : perimeter a b c) : 
  sin A + sin B = 3 * sqrt 21 / 28 :=
sorry

end TriangleProblem

end find_angle_C_find_sin_A_plus_sin_B_l194_194515


namespace modulo_multiplication_l194_194403

theorem modulo_multiplication (m : ℕ) (h : 0 ≤ m ∧ m < 50) :
  152 * 936 % 50 = 22 :=
by
  sorry

end modulo_multiplication_l194_194403


namespace two_f_of_x_l194_194213

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + x)

theorem two_f_of_x (x : ℝ) (h : x > 0) : 2 * f x = 18 / (9 + x) :=
  sorry

end two_f_of_x_l194_194213


namespace solve_inequalities_l194_194231

theorem solve_inequalities (x : ℝ) :
  ( (-x + 3)/2 < x ∧ 2*(x + 6) ≥ 5*x ) ↔ (1 < x ∧ x ≤ 4) :=
by
  sorry

end solve_inequalities_l194_194231


namespace simplify_expression_l194_194467

theorem simplify_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = -2 * Real.cos θ :=
sorry

end simplify_expression_l194_194467


namespace cricket_target_runs_l194_194255

theorem cricket_target_runs 
  (run_rate1 : ℝ) (run_rate2 : ℝ) (overs : ℕ)
  (h1 : run_rate1 = 5.4) (h2 : run_rate2 = 10.6) (h3 : overs = 25) :
  (run_rate1 * overs + run_rate2 * overs = 400) :=
by sorry

end cricket_target_runs_l194_194255


namespace horner_evaluation_l194_194722

-- Define the polynomial function
def f (x : ℝ) : ℝ := 4 * x^4 + 3 * x^3 - 6 * x^2 + x - 1

-- The theorem that we need to prove
theorem horner_evaluation : f (-1) = -5 :=
  by
  -- This is the statement without the proof steps
  sorry

end horner_evaluation_l194_194722


namespace circle_radius_5_l194_194680

theorem circle_radius_5 (k : ℝ) : 
  (∃ x y : ℝ, x^2 + 14 * x + y^2 + 8 * y - k = 0) ↔ k = -40 :=
by
  sorry

end circle_radius_5_l194_194680


namespace line_equation_l194_194566

theorem line_equation {k b : ℝ} 
  (h1 : (∀ x : ℝ, k * x + b = -4 * x + 2023 → k = -4))
  (h2 : b = -5) :
  ∀ x : ℝ, k * x + b = -4 * x - 5 := by
sorry

end line_equation_l194_194566


namespace adam_tickets_left_l194_194071

-- Define the initial number of tickets, cost per ticket, and total spending on the ferris wheel
def initial_tickets : ℕ := 13
def cost_per_ticket : ℕ := 9
def total_spent : ℕ := 81

-- Define the number of tickets Adam has after riding the ferris wheel
def tickets_left (initial_tickets cost_per_ticket total_spent : ℕ) : ℕ :=
  initial_tickets - (total_spent / cost_per_ticket)

-- Proposition to prove that Adam has 4 tickets left
theorem adam_tickets_left : tickets_left initial_tickets cost_per_ticket total_spent = 4 :=
by
  sorry

end adam_tickets_left_l194_194071


namespace endpoint_of_vector_a_l194_194805

theorem endpoint_of_vector_a (x y : ℝ) (h : (x - 3) / -3 = (y + 1) / 4) : 
    x = 13 / 5 ∧ y = 2 / 5 :=
by sorry

end endpoint_of_vector_a_l194_194805


namespace subset_condition_l194_194297

noncomputable def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : (B m ⊆ A) ↔ m ≤ 3 :=
sorry

end subset_condition_l194_194297


namespace min_unplowed_cells_l194_194258

theorem min_unplowed_cells (n k : ℕ) (hn : n > 0) (hk : k > 0) (hnk : n > k) :
  ∃ M : ℕ, M = (n - k)^2 := by
  sorry

end min_unplowed_cells_l194_194258


namespace chocolate_game_winner_l194_194946

-- Definitions of conditions for the problem
def chocolate_bar (m n : ℕ) := m * n

-- Theorem statement with conditions and conclusion
theorem chocolate_game_winner (m n : ℕ) (h1 : chocolate_bar m n = 48) : 
  ( ∃ first_player_wins : true, true) :=
by sorry

end chocolate_game_winner_l194_194946


namespace simplify_expression_l194_194868

theorem simplify_expression :
  (4.625 - 13/18 * 9/26) / (9/4) + 2.5 / 1.25 / 6.75 / 1 + 53/68 / ((1/2 - 0.375) / 0.125 + (5/6 - 7/12) / (0.358 - 1.4796 / 13.7)) = 17/27 :=
by sorry

end simplify_expression_l194_194868


namespace sum_smallest_largest_l194_194558

theorem sum_smallest_largest (n a : ℕ) (h_even_n : n % 2 = 0) (y x : ℕ)
  (h_y : y = a + n - 1)
  (h_x : x = (a + 3 * (n / 3 - 1)) * (n / 3)) : 
  2 * y = a + (a + 2 * (n - 1)) :=
by
  sorry

end sum_smallest_largest_l194_194558


namespace find_k_l194_194902

theorem find_k (k a : ℤ)
  (h₁ : 49 + k = a^2)
  (h₂ : 361 + k = (a + 2)^2)
  (h₃ : 784 + k = (a + 4)^2) :
  k = 6035 :=
by sorry

end find_k_l194_194902


namespace prime_divisibility_l194_194568

theorem prime_divisibility
  (a b : ℕ) (p q : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hm1 : ¬ p ∣ q - 1)
  (hm2 : q ∣ a ^ p - b ^ p) : q ∣ a - b :=
sorry

end prime_divisibility_l194_194568


namespace find_m_values_l194_194077

-- Defining the sets and conditions
def A : Set ℝ := { x | x ^ 2 - 9 * x - 10 = 0 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- Stating the proof problem
theorem find_m_values : {m | A ∪ B m = A} = {0, 1, -1 / 10} :=
by
  sorry

end find_m_values_l194_194077


namespace restaurant_sales_l194_194154

theorem restaurant_sales :
  let meals_sold_8 := 10
  let price_per_meal_8 := 8
  let meals_sold_10 := 5
  let price_per_meal_10 := 10
  let meals_sold_4 := 20
  let price_per_meal_4 := 4
  let total_sales := meals_sold_8 * price_per_meal_8 + meals_sold_10 * price_per_meal_10 + meals_sold_4 * price_per_meal_4
  total_sales = 210 :=
by
  sorry

end restaurant_sales_l194_194154


namespace breadth_of_hall_l194_194465

theorem breadth_of_hall (length_hall : ℝ) (stone_length_dm : ℝ) (stone_breadth_dm : ℝ)
    (num_stones : ℕ) (area_stone_m2 : ℝ) (total_area_m2 : ℝ) (breadth_hall : ℝ):
    length_hall = 36 → 
    stone_length_dm = 8 → 
    stone_breadth_dm = 5 → 
    num_stones = 1350 → 
    area_stone_m2 = (stone_length_dm * stone_breadth_dm) / 100 → 
    total_area_m2 = num_stones * area_stone_m2 → 
    breadth_hall = total_area_m2 / length_hall → 
    breadth_hall = 15 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4] at *
  simp [h5, h6, h7]
  sorry

end breadth_of_hall_l194_194465


namespace simplify_expression_l194_194401

theorem simplify_expression (y : ℝ) : 2 - (2 - (2 - (2 - (2 - y)))) = 4 - y :=
by
  sorry

end simplify_expression_l194_194401


namespace grace_earnings_in_september_l194_194663

theorem grace_earnings_in_september
  (hours_mowing : ℕ) (hours_pulling_weeds : ℕ) (hours_putting_mulch : ℕ)
  (rate_mowing : ℕ) (rate_pulling_weeds : ℕ) (rate_putting_mulch : ℕ)
  (total_hours_mowing : hours_mowing = 63) (total_hours_pulling_weeds : hours_pulling_weeds = 9) (total_hours_putting_mulch : hours_putting_mulch = 10)
  (rate_for_mowing : rate_mowing = 6) (rate_for_pulling_weeds : rate_pulling_weeds = 11) (rate_for_putting_mulch : rate_putting_mulch = 9) :
  hours_mowing * rate_mowing + hours_pulling_weeds * rate_pulling_weeds + hours_putting_mulch * rate_putting_mulch = 567 :=
by
  intros
  sorry

end grace_earnings_in_september_l194_194663


namespace binary_multiplication_l194_194977

/-- 
Calculate the product of two binary numbers and validate the result.
Given:
  a = 1101 in base 2,
  b = 111 in base 2,
Prove:
  a * b = 1011110 in base 2. 
-/
theorem binary_multiplication : 
  let a := 0b1101
  let b := 0b111
  a * b = 0b1011110 :=
by
  sorry

end binary_multiplication_l194_194977


namespace solution_set_of_inequality_l194_194853

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l194_194853


namespace ratio_of_areas_l194_194383

structure Triangle :=
  (AB BC AC AD AE : ℝ)
  (AB_pos : 0 < AB)
  (BC_pos : 0 < BC)
  (AC_pos : 0 < AC)
  (AD_pos : 0 < AD)
  (AE_pos : 0 < AE)

theorem ratio_of_areas (t : Triangle)
  (hAB : t.AB = 30)
  (hBC : t.BC = 45)
  (hAC : t.AC = 54)
  (hAD : t.AD = 24)
  (hAE : t.AE = 18) :
  (t.AD / t.AB) * (t.AE / t.AC) / (1 - (t.AD / t.AB) * (t.AE / t.AC)) = 4 / 11 :=
by
  sorry

end ratio_of_areas_l194_194383


namespace certain_number_l194_194079

theorem certain_number (x : ℝ) (h : 0.65 * 40 = (4/5) * x + 6) : x = 25 :=
sorry

end certain_number_l194_194079


namespace number_of_managers_in_sample_l194_194410

def totalStaff : ℕ := 160
def salespeople : ℕ := 104
def managers : ℕ := 32
def logisticsPersonnel : ℕ := 24
def sampleSize : ℕ := 20

theorem number_of_managers_in_sample : 
  (managers * (sampleSize / totalStaff) = 4) := by
  sorry

end number_of_managers_in_sample_l194_194410


namespace work_together_zero_days_l194_194333

theorem work_together_zero_days (a b : ℝ) (ha : a = 1/18) (hb : b = 1/9) (x : ℝ) (hx : 1 - x * a = 2/3) : x = 6 →
  (a - a) * (b - b) = 0 := by
  sorry

end work_together_zero_days_l194_194333


namespace remainder_of_division_l194_194328

theorem remainder_of_division (dividend divisor quotient remainder : ℕ)
  (h1 : dividend = 55053)
  (h2 : divisor = 456)
  (h3 : quotient = 120)
  (h4 : remainder = dividend - divisor * quotient) : 
  remainder = 333 := by
  sorry

end remainder_of_division_l194_194328


namespace total_ear_muffs_bought_l194_194282

-- Define the number of ear muffs bought before December
def ear_muffs_before_dec : ℕ := 1346

-- Define the number of ear muffs bought during December
def ear_muffs_during_dec : ℕ := 6444

-- The total number of ear muffs bought by customers
theorem total_ear_muffs_bought : ear_muffs_before_dec + ear_muffs_during_dec = 7790 :=
by
  sorry

end total_ear_muffs_bought_l194_194282


namespace simplified_expression_l194_194420

theorem simplified_expression :
  ( (81 / 16) ^ (3 / 4) - (-1) ^ 0 ) = 19 / 8 := 
by 
  -- It is a placeholder for the actual proof.
  sorry

end simplified_expression_l194_194420


namespace union_A_B_inter_A_compl_B_range_of_a_l194_194355

-- Define the sets A, B, and C
def A := {x : ℝ | -1 ≤ x ∧ x < 3}
def B := {x : ℝ | 2 * x - 4 ≥ x - 2}
def C (a : ℝ) := {x : ℝ | x ≥ a - 1}

-- Prove A ∪ B = {x | -1 ≤ x}
theorem union_A_B : A ∪ B = {x : ℝ | -1 ≤ x} :=
by sorry

-- Prove A ∩ (complement B) = {x | -1 ≤ x < 2}
theorem inter_A_compl_B : A ∩ (compl B) = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by sorry

-- Prove the range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 :=
by sorry

end union_A_B_inter_A_compl_B_range_of_a_l194_194355


namespace computer_price_increase_l194_194982

theorem computer_price_increase (c : ℕ) (h : 2 * c = 540) : c + (c * 30 / 100) = 351 :=
by
  sorry

end computer_price_increase_l194_194982


namespace michael_savings_l194_194274

theorem michael_savings :
  let price := 45
  let tax_rate := 0.08
  let promo_A_dis := 0.40
  let promo_B_dis := 15
  let before_tax_A := price + price * (1 - promo_A_dis)
  let before_tax_B := price + (price - promo_B_dis)
  let after_tax_A := before_tax_A * (1 + tax_rate)
  let after_tax_B := before_tax_B * (1 + tax_rate)
  after_tax_B - after_tax_A = 3.24 :=
by
  sorry

end michael_savings_l194_194274


namespace value_of_x_squared_plus_reciprocal_squared_l194_194673

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (h : 45 = x^4 + 1 / x^4) : 
  x^2 + 1 / x^2 = Real.sqrt 47 :=
by
  sorry

end value_of_x_squared_plus_reciprocal_squared_l194_194673


namespace min_value_of_frac_l194_194874

theorem min_value_of_frac (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : 2 * m + n = 1) (hm : m > 0) (hn : n > 0) :
  (1 / m) + (2 / n) = 8 :=
sorry

end min_value_of_frac_l194_194874


namespace model_to_reality_length_l194_194363

-- Defining conditions
def scale_factor := 50 -- one centimeter represents 50 meters
def model_length := 7.5 -- line segment in the model is 7.5 centimeters

-- Statement of the problem
theorem model_to_reality_length (scale_factor model_length : ℝ) 
  (scale_condition : scale_factor = 50) (length_condition : model_length = 7.5) :
  model_length * scale_factor = 375 := 
by
  rw [length_condition, scale_condition]
  norm_num

end model_to_reality_length_l194_194363


namespace ratio_of_fractions_proof_l194_194261

noncomputable def ratio_of_fractions (x y : ℝ) : Prop :=
  (5 * x = 6 * y) → (x ≠ 0 ∧ y ≠ 0) → ((1/3) * x / ((1/5) * y) = 2)

theorem ratio_of_fractions_proof (x y : ℝ) (hx: 5 * x = 6 * y) (hnz: x ≠ 0 ∧ y ≠ 0) : ((1/3) * x / ((1/5) * y) = 2) :=
  by 
  sorry

end ratio_of_fractions_proof_l194_194261


namespace cars_to_sell_l194_194331

theorem cars_to_sell (clients : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) (total_clients : ℕ) (h1 : selections_per_client = 2) 
  (h2 : selections_per_car = 3) (h3 : total_clients = 24) : (total_clients * selections_per_client / selections_per_car = 16) :=
by
  sorry

end cars_to_sell_l194_194331


namespace solve_system_a_l194_194745

theorem solve_system_a (x y : ℝ) (h1 : x^2 - 3 * x * y - 4 * y^2 = 0) (h2 : x^3 + y^3 = 65) : 
    x = 4 ∧ y = 1 :=
sorry

end solve_system_a_l194_194745


namespace piecewise_linear_function_y_at_x_10_l194_194956

theorem piecewise_linear_function_y_at_x_10
  (k1 k2 : ℝ)
  (y : ℝ → ℝ)
  (hx1 : ∀ x < 0, y x = k1 * x)
  (hx2 : ∀ x ≥ 0, y x = k2 * x)
  (h_y_pos : y 2 = 4)
  (h_y_neg : y (-5) = -20) :
  y 10 = 20 :=
by
  sorry

end piecewise_linear_function_y_at_x_10_l194_194956


namespace cubic_intersection_2_points_l194_194758

theorem cubic_intersection_2_points (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^3 - 3*x₁ + c = 0) ∧ (x₂^3 - 3*x₂ + c = 0)) 
  → (c = -2 ∨ c = 2) :=
sorry

end cubic_intersection_2_points_l194_194758


namespace seating_5_out_of_6_around_circle_l194_194546

def number_of_ways_to_seat_5_out_of_6_in_circle : Nat :=
  Nat.factorial 4

theorem seating_5_out_of_6_around_circle : number_of_ways_to_seat_5_out_of_6_in_circle = 24 :=
by {
  -- proof would be here
  sorry
}

end seating_5_out_of_6_around_circle_l194_194546


namespace monks_mantou_l194_194823

theorem monks_mantou (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + y / 3 = 100) :
  (3 * x + (100 - x) / 3 = 100) ∧ (x + y = 100 ∧ 3 * x + y / 3 = 100) :=
by
  sorry

end monks_mantou_l194_194823


namespace fraction_calculation_l194_194044

theorem fraction_calculation : (36 - 12) / (12 - 4) = 3 :=
by
  sorry

end fraction_calculation_l194_194044


namespace no_absolute_winner_l194_194256

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l194_194256


namespace positive_value_of_X_l194_194138

-- Definition for the problem's conditions
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Statement of the proof problem
theorem positive_value_of_X (X : ℝ) (h : hash X 7 = 170) : X = 11 :=
by
  sorry

end positive_value_of_X_l194_194138


namespace average_weight_increase_l194_194444

theorem average_weight_increase (A : ℝ) (X : ℝ) (h : (8 * A - 65 + 93) / 8 = A + X) :
  X = 3.5 :=
sorry

end average_weight_increase_l194_194444


namespace initial_percentage_is_30_l194_194501

def percentage_alcohol (P : ℝ) : Prop :=
  let initial_alcohol := (P / 100) * 50
  let mixed_solution_volume := 50 + 30
  let final_percentage_alcohol := 18.75
  let final_alcohol := (final_percentage_alcohol / 100) * mixed_solution_volume
  initial_alcohol = final_alcohol

theorem initial_percentage_is_30 :
  percentage_alcohol 30 :=
by
  unfold percentage_alcohol
  sorry

end initial_percentage_is_30_l194_194501


namespace initial_men_count_l194_194312

variable (M : ℕ)

theorem initial_men_count
  (work_completion_time : ℕ)
  (men_leaving : ℕ)
  (remaining_work_time : ℕ)
  (completion_days : ℕ) :
  work_completion_time = 40 →
  men_leaving = 20 →
  remaining_work_time = 40 →
  completion_days = 10 →
  M = 80 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_men_count_l194_194312


namespace selling_price_of_book_l194_194726

   theorem selling_price_of_book
     (cost_price : ℝ)
     (profit_rate : ℝ)
     (profit := (profit_rate / 100) * cost_price)
     (selling_price := cost_price + profit)
     (hp : cost_price = 50)
     (hr : profit_rate = 60) :
     selling_price = 80 := sorry
   
end selling_price_of_book_l194_194726


namespace total_bill_is_correct_l194_194009

-- Define conditions as constant values
def cost_per_scoop : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

-- Define the total bill calculation
def total_bill := (pierre_scoops * cost_per_scoop) + (mom_scoops * cost_per_scoop)

-- State the theorem that the total bill equals 14
theorem total_bill_is_correct : total_bill = 14 := by
  sorry

end total_bill_is_correct_l194_194009


namespace jason_initial_cards_l194_194934

-- Definitions based on conditions
def cards_given_away : ℕ := 4
def cards_left : ℕ := 5

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 9 :=
by sorry

end jason_initial_cards_l194_194934


namespace new_average_l194_194391

theorem new_average (avg : ℕ) (n : ℕ) (k : ℕ) (new_avg : ℕ) 
  (h1 : avg = 23) (h2 : n = 10) (h3 : k = 4) : 
  new_avg = (n * avg + n * k) / n → new_avg = 27 :=
by
  intro H
  sorry

end new_average_l194_194391


namespace right_triangle_candidate_l194_194760

theorem right_triangle_candidate :
  (∃ a b c : ℕ, (a, b, c) = (1, 2, 3) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (2, 3, 4) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (4, 5, 6) ∧ a^2 + b^2 = c^2) ↔
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) :=
by
  sorry

end right_triangle_candidate_l194_194760


namespace multiples_of_15_between_17_and_158_l194_194357

theorem multiples_of_15_between_17_and_158 : 
  let first := 30
  let last := 150
  let step := 15
  Nat.succ ((last - first) / step) = 9 := 
by
  sorry

end multiples_of_15_between_17_and_158_l194_194357


namespace smallest_possible_value_of_N_l194_194056

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end smallest_possible_value_of_N_l194_194056


namespace complex_addition_result_l194_194598

theorem complex_addition_result (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : a + b * i = (1 - i) * (2 + i)) : a + b = 2 :=
sorry

end complex_addition_result_l194_194598


namespace two_person_subcommittees_from_eight_l194_194043

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end two_person_subcommittees_from_eight_l194_194043


namespace emergency_vehicle_reachable_area_l194_194117

theorem emergency_vehicle_reachable_area :
  let speed_roads := 60 -- velocity on roads in miles per hour
    let speed_sand := 10 -- velocity on sand in miles per hour
    let time_limit := 5 / 60 -- time limit in hours
    let max_distance_on_roads := speed_roads * time_limit -- max distance on roads
    let radius_sand_circle := (10 / 12) -- radius on the sand
    -- calculate area covered
  (5 * 5 + 4 * (1 / 4 * Real.pi * (radius_sand_circle)^2)) = (25 + (25 * Real.pi) / 36) :=
by
  sorry

end emergency_vehicle_reachable_area_l194_194117


namespace round_trip_ticket_percentage_l194_194445

theorem round_trip_ticket_percentage (P R : ℝ) 
  (h1 : 0.20 * P = 0.50 * R) : R = 0.40 * P :=
by
  sorry

end round_trip_ticket_percentage_l194_194445


namespace total_money_raised_l194_194656

def tickets_sold : ℕ := 25
def price_per_ticket : ℝ := 2.0
def donation_count : ℕ := 2
def donation_amount : ℝ := 15.0
def additional_donation : ℝ := 20.0

theorem total_money_raised :
  (tickets_sold * price_per_ticket) + (donation_count * donation_amount) + additional_donation = 100 :=
by
  sorry

end total_money_raised_l194_194656


namespace sides_of_polygons_l194_194916

theorem sides_of_polygons (p : ℕ) (γ : ℝ) (n1 n2 : ℕ) (h1 : p = 5) (h2 : γ = 12 / 7) 
    (h3 : n2 = n1 + p) 
    (h4 : 360 / n1 - 360 / n2 = γ) : 
    n1 = 30 ∧ n2 = 35 := 
  sorry

end sides_of_polygons_l194_194916


namespace initial_bananas_proof_l194_194725

noncomputable def initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) : ℕ :=
  (extra_bananas * (total_children - absent_children)) / (total_children - extra_bananas)

theorem initial_bananas_proof
  (total_children : ℕ)
  (absent_children : ℕ)
  (extra_bananas : ℕ)
  (h_total : total_children = 640)
  (h_absent : absent_children = 320)
  (h_extra : extra_bananas = 2) : initial_bananas_per_child total_children absent_children extra_bananas = 2 :=
by
  sorry

end initial_bananas_proof_l194_194725


namespace ellipse_focus_xaxis_l194_194845

theorem ellipse_focus_xaxis (k : ℝ) (h : 1 - k > 2 + k ∧ 2 + k > 0) : -2 < k ∧ k < -1/2 :=
by sorry

end ellipse_focus_xaxis_l194_194845


namespace find_b_n_find_T_n_l194_194302

-- Conditions
def S (n : ℕ) : ℕ := 3 * n^2 + 8 * n
def a (n : ℕ) : ℕ := S n - S (n - 1) -- provided n > 1
def b : ℕ → ℕ := sorry -- This is what we need to prove
def c (n : ℕ) : ℕ := (a n + 1)^(n + 1) / (b n + 2)^n  -- Definition of c_n
def T (n : ℕ) : ℕ := sorry -- The sum of the first n terms of c_n

-- Proof requirements
def proof_b_n := ∀ n : ℕ, b n = 3 * n + 1
def proof_T_n := ∀ n : ℕ, T n = 3 * n * 2^(n+2)

theorem find_b_n : proof_b_n := 
by sorry

theorem find_T_n : proof_T_n := 
by sorry

end find_b_n_find_T_n_l194_194302


namespace choose_stick_l194_194267

-- Define the lengths of the sticks Xiaoming has
def xm_stick1 : ℝ := 4
def xm_stick2 : ℝ := 7

-- Define the lengths of the sticks Xiaohong has
def stick2 : ℝ := 2
def stick3 : ℝ := 3
def stick8 : ℝ := 8
def stick12 : ℝ := 12

-- Define the condition for a valid stick choice from Xiaohong's sticks
def valid_stick (x : ℝ) : Prop := 3 < x ∧ x < 11

-- State the problem as a theorem to be proved
theorem choose_stick : valid_stick stick8 := by
  sorry

end choose_stick_l194_194267


namespace part1_part2_part3_l194_194203

def folklore {a b m n : ℤ} (h1 : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : Prop :=
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n

theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

theorem part2 : 13 + 4 * Real.sqrt 3 = (1 + 2 * Real.sqrt 3) ^ 2 :=
by sorry

theorem part3 (a m n : ℤ) (h1 : 4 = 2 * m * n) (h2 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = 7 ∨ a = 13 :=
by sorry

end part1_part2_part3_l194_194203


namespace probability_interval_contains_p_l194_194746

theorem probability_interval_contains_p (P_A P_B p : ℝ) 
  (hA : P_A = 5 / 6) 
  (hB : P_B = 3 / 4) 
  (hp : p = P_A + P_B - 1) : 
  (5 / 12 ≤ p ∧ p ≤ 3 / 4) :=
by
  -- The proof is skipped by sorry as per the instructions.
  sorry

end probability_interval_contains_p_l194_194746


namespace survey_respondents_l194_194983

theorem survey_respondents (X Y : ℕ) (hX : X = 150) (ratio : X = 5 * Y) : X + Y = 180 :=
by
  sorry

end survey_respondents_l194_194983


namespace gumball_sharing_l194_194869

theorem gumball_sharing (init_j : ℕ) (init_jq : ℕ) (mult_j : ℕ) (mult_jq : ℕ) :
  init_j = 40 → init_jq = 60 → mult_j = 5 → mult_jq = 3 →
  (init_j + mult_j * init_j + init_jq + mult_jq * init_jq) / 2 = 240 :=
by
  intros h1 h2 h3 h4
  sorry

end gumball_sharing_l194_194869


namespace cylinder_base_radius_l194_194277

theorem cylinder_base_radius (l w : ℝ) (h_l : l = 6) (h_w : w = 4) (h_circ : l = 2 * Real.pi * r ∨ w = 2 * Real.pi * r) : 
    r = 3 / Real.pi ∨ r = 2 / Real.pi := by
  sorry

end cylinder_base_radius_l194_194277


namespace reggie_games_lost_l194_194431

-- Given conditions:
def initial_marbles : ℕ := 100
def marbles_per_game : ℕ := 10
def games_played : ℕ := 9
def marbles_after_games : ℕ := 90

-- The statement to prove:
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / marbles_per_game = 1 := 
sorry

end reggie_games_lost_l194_194431


namespace prime_pairs_square_l194_194146

noncomputable def is_square (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem prime_pairs_square (a b : ℤ) (ha : is_prime a) (hb : is_prime b) :
  is_square (3 * a^2 * b + 16 * a * b^2) ↔ (a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3) :=
by
  sorry

end prime_pairs_square_l194_194146


namespace pipe_c_empty_time_l194_194014

theorem pipe_c_empty_time :
  (1 / 45 + 1 / 60 - x = 1 / 40) → (1 / x = 72) :=
by
  sorry

end pipe_c_empty_time_l194_194014


namespace fraction_of_cracked_pots_is_2_over_5_l194_194509

-- Definitions for the problem conditions
def total_pots : ℕ := 80
def price_per_pot : ℕ := 40
def total_revenue : ℕ := 1920

-- Statement to prove the fraction of cracked pots
theorem fraction_of_cracked_pots_is_2_over_5 
  (C : ℕ) 
  (h1 : (total_pots - C) * price_per_pot = total_revenue) : 
  C / total_pots = 2 / 5 :=
by
  sorry

end fraction_of_cracked_pots_is_2_over_5_l194_194509


namespace abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l194_194986

theorem abs_x_minus_1_le_1_is_equivalent_to_x_le_2 (x : ℝ) :
  (|x - 1| ≤ 1) ↔ (x ≤ 2) := sorry

end abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l194_194986


namespace curve_to_polar_l194_194206

noncomputable def polar_eq_of_curve (x y : ℝ) (ρ θ : ℝ) : Prop :=
  (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) ∧ (x ^ 2 + y ^ 2 - 2 * x = 0) → (ρ = 2 * Real.cos θ)

theorem curve_to_polar (x y ρ θ : ℝ) :
  polar_eq_of_curve x y ρ θ :=
sorry

end curve_to_polar_l194_194206


namespace simplify_expression_l194_194694

variable (m : ℕ) (h1 : m ≠ 2) (h2 : m ≠ 3)

theorem simplify_expression : 
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) :=
by sorry

end simplify_expression_l194_194694


namespace kolya_time_segment_DE_l194_194301

-- Definitions representing the conditions
def time_petya_route : ℝ := 12  -- Petya takes 12 minutes
def time_kolya_route : ℝ := 12  -- Kolya also takes 12 minutes
def kolya_speed_factor : ℝ := 1.2

-- Proof problem: Prove that Kolya spends 1 minute traveling the segment D-E
theorem kolya_time_segment_DE 
    (v : ℝ)  -- Assume v is Petya's speed
    (time_petya_A_B_C : ℝ := time_petya_route)  
    (time_kolya_A_D_E_F_C : ℝ := time_kolya_route)
    (kolya_fast_factor : ℝ := kolya_speed_factor)
    : (time_petya_A_B_C / kolya_fast_factor - time_petya_A_B_C) / (2 / kolya_fast_factor) = 1 := 
by 
    sorry

end kolya_time_segment_DE_l194_194301


namespace exists_subset_no_double_l194_194980

theorem exists_subset_no_double (s : Finset ℕ) (h₁ : s = Finset.range 3000) :
  ∃ t : Finset ℕ, t.card = 2000 ∧ (∀ x ∈ t, ∀ y ∈ t, x ≠ 2 * y ∧ y ≠ 2 * x) :=
by
  sorry

end exists_subset_no_double_l194_194980


namespace ant_to_vertices_probability_l194_194082

noncomputable def event_A_probability : ℝ :=
  1 - (Real.sqrt 3 * Real.pi / 24)

theorem ant_to_vertices_probability :
  let side_length := 4
  let event_A := "the distance from the ant to all three vertices is more than 1"
  event_A_probability = 1 - Real.sqrt 3 * Real.pi / 24
:=
sorry

end ant_to_vertices_probability_l194_194082


namespace sign_of_x_and_y_l194_194738

theorem sign_of_x_and_y (x y : ℝ) (h1 : x * y > 1) (h2 : x + y ≥ 0) : x > 0 ∧ y > 0 :=
sorry

end sign_of_x_and_y_l194_194738


namespace price_increase_problem_l194_194814

variable (P P' x : ℝ)

theorem price_increase_problem
  (h1 : P' = P * (1 + x / 100))
  (h2 : P = P' * (1 - 23.076923076923077 / 100)) :
  x = 30 :=
by
  sorry

end price_increase_problem_l194_194814


namespace solution_set_of_fraction_inequality_l194_194419

theorem solution_set_of_fraction_inequality
  (a b : ℝ) (h₀ : ∀ x : ℝ, x > 1 → ax - b > 0) :
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 2} :=
by
  sorry

end solution_set_of_fraction_inequality_l194_194419


namespace augustus_makes_3_milkshakes_l194_194775

def augMilkshakePerHour (A : ℕ) (Luna : ℕ) (hours : ℕ) (totalMilkshakes : ℕ) : Prop :=
  (A + Luna) * hours = totalMilkshakes

theorem augustus_makes_3_milkshakes :
  augMilkshakePerHour 3 7 8 80 :=
by
  -- We assume the proof here
  sorry

end augustus_makes_3_milkshakes_l194_194775


namespace ratio_third_to_second_is_one_l194_194659

variable (x y : ℕ)

-- The second throw skips 2 more times than the first throw
def second_throw := x + 2
-- The third throw skips y times
def third_throw := y
-- The fourth throw skips 3 fewer times than the third throw
def fourth_throw := y - 3
-- The fifth throw skips 1 more time than the fourth throw
def fifth_throw := (y - 3) + 1

-- The fifth throw skipped 8 times
axiom fifth_throw_condition : fifth_throw y = 8
-- The total number of skips between all throws is 33
axiom total_skips_condition : x + second_throw x + y + fourth_throw y + fifth_throw y = 33

-- Prove the ratio of skips in third throw to the second throw is 1:1
theorem ratio_third_to_second_is_one : (third_throw y) / (second_throw x) = 1 := sorry

end ratio_third_to_second_is_one_l194_194659


namespace correct_average_is_18_l194_194641

theorem correct_average_is_18 (incorrect_avg : ℕ) (incorrect_num : ℕ) (true_num : ℕ) (n : ℕ) 
  (h1 : incorrect_avg = 16) (h2 : incorrect_num = 25) (h3 : true_num = 45) (h4 : n = 10) : 
  (incorrect_avg * n + (true_num - incorrect_num)) / n = 18 :=
by
  sorry

end correct_average_is_18_l194_194641


namespace total_workers_construction_l194_194628

def number_of_monkeys : Nat := 239
def number_of_termites : Nat := 622
def total_workers (m : Nat) (t : Nat) : Nat := m + t

theorem total_workers_construction : total_workers number_of_monkeys number_of_termites = 861 := by
  sorry

end total_workers_construction_l194_194628


namespace tangent_value_prism_QABC_l194_194437

-- Assuming R is the radius of the sphere and considering the given conditions
variables {R x : ℝ} (P Q A B C M H : Type)

-- Given condition: Angle between lateral face and base of prism P-ABC is 45 degrees
def angle_PABC : ℝ := 45
-- Required to prove: tan(angle between lateral face and base of prism Q-ABC) = 4
def tangent_QABC : ℝ := 4

theorem tangent_value_prism_QABC
  (h1 : angle_PABC = 45)
  (h2 : 5 * x - 2 * R = 0) -- Derived condition from the solution
  (h3 : x = 2 * R / 5) -- x, the distance calculation
: tangent_QABC = 4 := by
  sorry

end tangent_value_prism_QABC_l194_194437


namespace find_hourly_rate_l194_194573

-- Definitions of conditions in a)
def hourly_rate : ℝ := sorry  -- This is what we will find.
def hours_worked : ℝ := 3
def tip_percentage : ℝ := 0.2
def total_paid : ℝ := 54

-- Functions based on the conditions
def cost_without_tip (rate : ℝ) : ℝ := hours_worked * rate
def tip_amount (rate : ℝ) : ℝ := tip_percentage * (cost_without_tip rate)
def total_cost (rate : ℝ) : ℝ := (cost_without_tip rate) + (tip_amount rate)

-- The goal is to prove that the rate is 15
theorem find_hourly_rate : total_cost 15 = total_paid :=
by
  sorry

end find_hourly_rate_l194_194573


namespace n_greater_than_7_l194_194416

theorem n_greater_than_7 (m n : ℕ) (hmn : m > n) (h : ∃k:ℕ, 22220038^m - 22220038^n = 10^8 * k) : n > 7 :=
sorry

end n_greater_than_7_l194_194416


namespace coin_difference_l194_194257

/-- 
  Given that Paul has 5-cent, 20-cent, and 15-cent coins, 
  prove that the difference between the maximum and minimum number of coins
  needed to make exactly 50 cents is 6.
-/
theorem coin_difference :
  ∃ (coins : Nat → Nat),
    (coins 5 + coins 20 + coins 15) = 6 ∧
    (5 * coins 5 + 20 * coins 20 + 15 * coins 15 = 50) :=
sorry

end coin_difference_l194_194257


namespace count_multiples_of_7_not_14_lt_400_l194_194134

theorem count_multiples_of_7_not_14_lt_400 : 
  ∃ (n : ℕ), n = 29 ∧ ∀ (m : ℕ), (m < 400 ∧ m % 7 = 0 ∧ m % 14 ≠ 0) ↔ (∃ k : ℕ, 1 ≤ k ∧ k ≤ 29 ∧ m = 7 * (2 * k - 1)) :=
by
  sorry

end count_multiples_of_7_not_14_lt_400_l194_194134


namespace new_students_count_l194_194099

theorem new_students_count (x : ℕ) (avg_age_group new_avg_age avg_new_students : ℕ)
  (h1 : avg_age_group = 14) (h2 : new_avg_age = 15) (h3 : avg_new_students = 17)
  (initial_students : ℕ) (initial_avg_age : ℕ)
  (h4 : initial_students = 10) (h5 : initial_avg_age = initial_students * avg_age_group)
  (h6 : new_avg_age * (initial_students + x) = initial_avg_age + (x * avg_new_students)) :
  x = 5 := 
by
  sorry

end new_students_count_l194_194099


namespace even_factors_count_l194_194181

theorem even_factors_count (n : ℕ) (h : n = 2^3 * 3 * 7^2 * 5) : 
  ∃ k, k = 36 ∧ 
       (∀ a b c d : ℕ, 1 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 0 ≤ d ∧ d ≤ 1 →
       ∃ m, m = 2^a * 3^b * 7^c * 5^d ∧ 2 ∣ m ∧ m ∣ n) := sorry

end even_factors_count_l194_194181


namespace smallest_number_bob_l194_194765

-- Define the conditions given in the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_factors (x : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ x }

-- The problem statement
theorem smallest_number_bob (b : ℕ) (h1 : prime_factors 30 = prime_factors b) : b = 30 :=
by
  sorry

end smallest_number_bob_l194_194765


namespace dave_earnings_l194_194049

def total_games : Nat := 10
def non_working_games : Nat := 2
def price_per_game : Nat := 4
def working_games : Nat := total_games - non_working_games
def money_earned : Nat := working_games * price_per_game

theorem dave_earnings : money_earned = 32 := by
  sorry

end dave_earnings_l194_194049


namespace solution_set_for_log_inequality_l194_194505

noncomputable def log_base_0_1 (x: ℝ) : ℝ := Real.log x / Real.log 0.1

theorem solution_set_for_log_inequality :
  ∀ x : ℝ, (0 < x) → 
  log_base_0_1 (2^x - 1) < 0 ↔ x > 1 :=
by
  sorry

end solution_set_for_log_inequality_l194_194505


namespace carson_giant_slide_rides_l194_194978

theorem carson_giant_slide_rides :
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  -- Convert hours to minutes
  let total_minutes := total_hours * 60
  -- Calculate total wait time for roller coaster
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  -- Calculate total wait time for tilt-a-whirl
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  -- Calculate total wait time for roller coaster and tilt-a-whirl
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  -- Calculate remaining time
  let remaining_time := total_minutes - total_wait
  -- Calculate how many times Carson can ride the giant slide
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  giant_slide_rides = 4 := by
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  let total_minutes := total_hours * 60
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  let remaining_time := total_minutes - total_wait
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  show giant_slide_rides = 4
  sorry

end carson_giant_slide_rides_l194_194978


namespace range_of_ab_l194_194051

noncomputable def f (x : ℝ) : ℝ := abs (2 - x^2)

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 0 < a * b ∧ a * b < 2 :=
by
  sorry

end range_of_ab_l194_194051


namespace years_since_marriage_l194_194058

theorem years_since_marriage (x : ℕ) (ave_age_husband_wife_at_marriage : ℕ)
  (total_family_age_now : ℕ) (child_age : ℕ) (family_members : ℕ) :
  ave_age_husband_wife_at_marriage = 23 →
  total_family_age_now = 19 →
  child_age = 1 →
  family_members = 3 →
  (46 + 2 * x) + child_age = 57 →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end years_since_marriage_l194_194058


namespace find_f_2011_l194_194488

noncomputable def f : ℝ → ℝ := sorry

axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom specific_interval (x : ℝ) (h2 : 2 < x) (h4 : x < 4) : f x = x + 3

theorem find_f_2011 : f 2011 = 6 :=
by {
  -- Leave this part to be filled with the actual proof,
  -- satisfying the initial conditions and concluding f(2011) = 6
  sorry
}

end find_f_2011_l194_194488


namespace f_five_l194_194685

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = - f x
axiom f_one : f 1 = 1 / 2
axiom functional_equation : ∀ x : ℝ, f (x + 2) = f x + f 2

theorem f_five : f 5 = 5 / 2 :=
by sorry

end f_five_l194_194685


namespace original_length_in_meters_l194_194483

-- Conditions
def erased_length : ℝ := 10 -- 10 cm
def remaining_length : ℝ := 90 -- 90 cm

-- Question: What is the original length of the line in meters?
theorem original_length_in_meters : (remaining_length + erased_length) / 100 = 1 := 
by 
  -- The proof is omitted
  sorry

end original_length_in_meters_l194_194483


namespace rectangle_area_eq_l194_194233

theorem rectangle_area_eq (d : ℝ) (w : ℝ) (h1 : w = d / (2 * (5 : ℝ) ^ (1/2))) (h2 : 3 * w = (3 * d) / (2 * (5 : ℝ) ^ (1/2))) : 
  (3 * w^2) = (3 / 10) * d^2 := 
by sorry

end rectangle_area_eq_l194_194233


namespace train_speed_l194_194784

theorem train_speed (length : ℕ) (time : ℝ)
  (h_length : length = 160)
  (h_time : time = 18) :
  (length / time * 3.6 : ℝ) = 32 :=
by
  sorry

end train_speed_l194_194784


namespace checkerboard_probability_not_on_perimeter_l194_194512

def total_squares : ℕ := 81

def perimeter_squares : ℕ := 32

def non_perimeter_squares : ℕ := total_squares - perimeter_squares

noncomputable def probability_not_on_perimeter : ℚ := non_perimeter_squares / total_squares

theorem checkerboard_probability_not_on_perimeter :
  probability_not_on_perimeter = 49 / 81 :=
by
  sorry

end checkerboard_probability_not_on_perimeter_l194_194512


namespace find_k_l194_194586

noncomputable section

open Polynomial

-- Define the conditions
variables (h k : Polynomial ℚ)
variables (C : k.eval (-1) = 15) (H : h.comp k = h * k) (nonzero_h : h ≠ 0)

-- The goal is to prove k(x) = x^2 + 21x - 35
theorem find_k : k = X^2 + 21 * X - 35 :=
  by sorry

end find_k_l194_194586


namespace find_f_neg2_l194_194865

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem find_f_neg2 : f (-2) = 15 :=
by
  sorry

end find_f_neg2_l194_194865


namespace opposite_of_neg_2023_l194_194841

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l194_194841


namespace probability_of_same_team_is_one_third_l194_194008

noncomputable def probability_same_team : ℚ :=
  let teams := 3
  let total_combinations := teams * teams
  let successful_outcomes := teams
  successful_outcomes / total_combinations

theorem probability_of_same_team_is_one_third :
  probability_same_team = 1 / 3 := by
  sorry

end probability_of_same_team_is_one_third_l194_194008


namespace no_triangles_with_geometric_progression_angles_l194_194499

theorem no_triangles_with_geometric_progression_angles :
  ¬ ∃ (a r : ℕ), a ≥ 10 ∧ (a + a * r + a * r^2 = 180) ∧ (a ≠ a * r) ∧ (a ≠ a * r^2) ∧ (a * r ≠ a * r^2) :=
sorry

end no_triangles_with_geometric_progression_angles_l194_194499


namespace expression_evaluation_l194_194833

theorem expression_evaluation : (50 + 12) ^ 2 - (12 ^ 2 + 50 ^ 2) = 1200 := 
by
  sorry

end expression_evaluation_l194_194833


namespace find_amount_l194_194279

theorem find_amount (amount : ℝ) (h : 0.25 * amount = 75) : amount = 300 :=
sorry

end find_amount_l194_194279


namespace find_y_l194_194828

theorem find_y (x y : ℝ) (h1 : x = 4 * y) (h2 : (1 / 2) * x = 1) : y = 1 / 2 :=
by
  sorry

end find_y_l194_194828


namespace work_completion_time_l194_194025

noncomputable def work_rate_A : ℚ := 1 / 12
noncomputable def work_rate_B : ℚ := 1 / 14

theorem work_completion_time : 
  (work_rate_A + work_rate_B)⁻¹ = 84 / 13 := by
  sorry

end work_completion_time_l194_194025


namespace solution_inequality_l194_194278

-- Conditions
variables {a b x : ℝ}
theorem solution_inequality (h1 : a < 0) (h2 : b = a) :
  {x : ℝ | (ax + b) ≤ 0} = {x : ℝ | x ≥ -1} →
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end solution_inequality_l194_194278


namespace divisor_is_three_l194_194617

theorem divisor_is_three (n d q p : ℕ) (h1 : n = d * q + 3) (h2 : n^2 = d * p + 3) : d = 3 := 
sorry

end divisor_is_three_l194_194617


namespace problem_statement_l194_194565

theorem problem_statement (x y : ℝ) (h1 : |x| + x - y = 16) (h2 : x - |y| + y = -8) : x + y = -8 := sorry

end problem_statement_l194_194565


namespace replace_90_percent_in_3_days_cannot_replace_all_banknotes_l194_194367

-- Define constants and conditions
def total_old_banknotes : ℕ := 3628800
def daily_cost : ℕ := 90000
def major_repair_cost : ℕ := 700000
def max_daily_print_after_repair : ℕ := 1000000
def budget_limit : ℕ := 1000000

-- Define the day's print capability function (before repair)
def daily_print (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  if num_days = 1 then banknotes_remaining / 2
  else (banknotes_remaining / (num_days + 1))

-- Define the budget calculation before repair
def print_costs (num_days : ℕ) (banknotes_remaining : ℕ) : ℕ :=
  daily_cost * num_days

-- Lean theorem to be stated proving that 90% of the banknotes can be replaced within 3 days
theorem replace_90_percent_in_3_days :
  ∃ (days : ℕ) (banknotes_replaced : ℕ), days = 3 ∧ banknotes_replaced = 3265920 ∧ print_costs days total_old_banknotes ≤ budget_limit :=
sorry

-- Lean theorem to be stated proving that not all banknotes can be replaced within the given budget
theorem cannot_replace_all_banknotes :
  ∀ banknotes_replaced cost : ℕ,
  banknotes_replaced < total_old_banknotes ∧ cost ≤ budget_limit →
  banknotes_replaced + (total_old_banknotes / (4 + 1)) < total_old_banknotes :=
sorry

end replace_90_percent_in_3_days_cannot_replace_all_banknotes_l194_194367


namespace pipe_B_fill_time_l194_194303

theorem pipe_B_fill_time (T : ℕ) (h1 : 50 > 0) (h2 : 30 > 0)
  (h3 : (1/50 + 1/T = 1/30)) : T = 75 := 
sorry

end pipe_B_fill_time_l194_194303


namespace sandwiches_difference_l194_194284

-- Conditions definitions
def sandwiches_at_lunch_monday : ℤ := 3
def sandwiches_at_dinner_monday : ℤ := 2 * sandwiches_at_lunch_monday
def total_sandwiches_monday : ℤ := sandwiches_at_lunch_monday + sandwiches_at_dinner_monday
def sandwiches_on_tuesday : ℤ := 1

-- Proof goal
theorem sandwiches_difference :
  total_sandwiches_monday - sandwiches_on_tuesday = 8 :=
  by
  sorry

end sandwiches_difference_l194_194284


namespace difference_between_sevens_l194_194029

-- Define the numeral
def numeral : ℕ := 54179759

-- Define a function to find the place value of a digit at a specific position in a number
def place_value (n : ℕ) (pos : ℕ) : ℕ :=
  let digit := (n / 10^pos) % 10
  digit * 10^pos

-- Define specific place values for the two sevens
def first_seven_place : ℕ := place_value numeral 4  -- Ten-thousands place
def second_seven_place : ℕ := place_value numeral 1 -- Tens place

-- Define their values
def first_seven_value : ℕ := 7 * 10^4  -- 70,000
def second_seven_value : ℕ := 7 * 10^1  -- 70

-- Prove the difference between these place values
theorem difference_between_sevens : first_seven_value - second_seven_value = 69930 := by
  sorry

end difference_between_sevens_l194_194029


namespace darrel_will_receive_l194_194424

noncomputable def darrel_coins_value : ℝ := 
  let quarters := 127 
  let dimes := 183 
  let nickels := 47 
  let pennies := 237 
  let half_dollars := 64 
  let euros := 32 
  let pounds := 55 
  let quarter_fee_rate := 0.12 
  let dime_fee_rate := 0.07 
  let nickel_fee_rate := 0.15 
  let penny_fee_rate := 0.10 
  let half_dollar_fee_rate := 0.05 
  let euro_exchange_rate := 1.18 
  let euro_fee_rate := 0.03 
  let pound_exchange_rate := 1.39 
  let pound_fee_rate := 0.04 
  let quarters_value := 127 * 0.25 
  let quarters_fee := quarters_value * 0.12 
  let quarters_after_fee := quarters_value - quarters_fee 
  let dimes_value := 183 * 0.10 
  let dimes_fee := dimes_value * 0.07 
  let dimes_after_fee := dimes_value - dimes_fee 
  let nickels_value := 47 * 0.05 
  let nickels_fee := nickels_value * 0.15 
  let nickels_after_fee := nickels_value - nickels_fee 
  let pennies_value := 237 * 0.01 
  let pennies_fee := pennies_value * 0.10 
  let pennies_after_fee := pennies_value - pennies_fee 
  let half_dollars_value := 64 * 0.50 
  let half_dollars_fee := half_dollars_value * 0.05 
  let half_dollars_after_fee := half_dollars_value - half_dollars_fee 
  let euros_value := 32 * 1.18 
  let euros_fee := euros_value * 0.03 
  let euros_after_fee := euros_value - euros_fee 
  let pounds_value := 55 * 1.39 
  let pounds_fee := pounds_value * 0.04 
  let pounds_after_fee := pounds_value - pounds_fee 
  quarters_after_fee + dimes_after_fee + nickels_after_fee + pennies_after_fee + half_dollars_after_fee + euros_after_fee + pounds_after_fee

theorem darrel_will_receive : darrel_coins_value = 189.51 := by
  unfold darrel_coins_value
  sorry

end darrel_will_receive_l194_194424


namespace intersection_count_l194_194221

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem intersection_count (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < Real.pi / 2) 
  (h_max : ∀ x, f x ω φ ≤ f (Real.pi / 6) ω φ)
  (h_period : ∀ x, f x ω φ = f (x + 2 * Real.pi / ω) ω φ) :
  ∃! x : ℝ, f x ω φ = -x + 2 * Real.pi / 3 :=
sorry

end intersection_count_l194_194221


namespace length_of_train_l194_194007

theorem length_of_train (v : ℝ) (t : ℝ) (L : ℝ) 
  (h₁ : v = 36) 
  (h₂ : t = 1) 
  (h_eq_lengths : true) -- assuming the equality of lengths tacitly without naming
  : L = 300 := 
by 
  -- proof steps would go here
  sorry

end length_of_train_l194_194007


namespace fraction_savings_on_makeup_l194_194691

theorem fraction_savings_on_makeup (savings : ℝ) (sweater_cost : ℝ) (makeup_cost : ℝ) (h_savings : savings = 80) (h_sweater : sweater_cost = 20) (h_makeup : makeup_cost = savings - sweater_cost) : makeup_cost / savings = 3 / 4 := by
  sorry

end fraction_savings_on_makeup_l194_194691


namespace probability_AEMC9_is_1_over_84000_l194_194709

-- Define possible symbols for each category.
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def nonVowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

-- Define the total number of possible license plates.
def totalLicensePlates : Nat := 
  (vowels.length) * (vowels.length - 1) * 
  (nonVowels.length) * (nonVowels.length - 1) * 
  (digits.length)

-- Define the number of favorable outcomes.
def favorableOutcomes : Nat := 1

-- Define the probability calculation.
noncomputable def probabilityAEMC9 : ℚ := favorableOutcomes / totalLicensePlates

-- The theorem to prove.
theorem probability_AEMC9_is_1_over_84000 :
  probabilityAEMC9 = 1 / 84000 := by
  sorry

end probability_AEMC9_is_1_over_84000_l194_194709


namespace angle_P_of_extended_sides_l194_194627

noncomputable def regular_pentagon_angle_sum : ℕ := 540

noncomputable def internal_angle_regular_pentagon (n : ℕ) (h : 5 = n) : ℕ :=
  regular_pentagon_angle_sum / n

def interior_angle_pentagon : ℕ := 108

theorem angle_P_of_extended_sides (ABCDE : Prop) (h1 : interior_angle_pentagon = 108)
  (P : Prop) (h3 : 72 + 72 = 144) : 180 - 144 = 36 := by 
  sorry

end angle_P_of_extended_sides_l194_194627


namespace incorrect_line_pass_through_Q_l194_194438

theorem incorrect_line_pass_through_Q (a b : ℝ) : 
  (∀ (k : ℝ), ∃ (Q : ℝ × ℝ), Q = (0, b) ∧ y = k * x + b) →
  (¬ ∃ k : ℝ, ∀ y x, y = k * x + b ∧ x = 0)
:= 
sorry

end incorrect_line_pass_through_Q_l194_194438


namespace number_of_red_balls_l194_194774

-- Definitions and conditions
def ratio_white_red (w : ℕ) (r : ℕ) : Prop := (w : ℤ) * 3 = 5 * (r : ℤ)
def white_balls : ℕ := 15

-- The theorem to prove
theorem number_of_red_balls (r : ℕ) (h : ratio_white_red white_balls r) : r = 9 :=
by
  sorry

end number_of_red_balls_l194_194774


namespace distance_swim_downstream_correct_l194_194291

def speed_man_still_water : ℝ := 7
def time_taken : ℝ := 5
def distance_upstream : ℝ := 25

lemma distance_swim_downstream (V_m : ℝ) (t : ℝ) (d_up : ℝ) : 
  t * ((V_m + (V_m - d_up / t)) / 2) = 45 :=
by
  have h_speed_upstream : (V_m - (d_up / t)) = d_up / t := by sorry
  have h_speed_stream : (d_up / t) = (V_m - (d_up / t)) := by sorry
  have h_distance_downstream : t * ((V_m + (V_m - (d_up / t)) / 2)) = t * (V_m + (V_m - (V_m - d_up / t))) := by sorry
  sorry

noncomputable def distance_swim_downstream_value : ℝ :=
  9 * 5

theorem distance_swim_downstream_correct :
  distance_swim_downstream_value = 45 :=
by
  sorry

end distance_swim_downstream_correct_l194_194291


namespace omega_range_l194_194851

theorem omega_range (ω : ℝ) (a b : ℝ) (hω_pos : ω > 0) (h_range : π ≤ a ∧ a < b ∧ b ≤ 2 * π)
  (h_sin : Real.sin (ω * a) + Real.sin (ω * b) = 2) :
  ω ∈ Set.Icc (9 / 4 : ℝ) (5 / 2) ∪ Set.Ici (13 / 4) :=
by
  sorry

end omega_range_l194_194851


namespace minimum_pie_pieces_l194_194843

theorem minimum_pie_pieces (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n, (∀ k, k = p ∨ k = q → (n ≠ 0 → n % k = 0)) ∧ n = p + q - 1 :=
by {
  sorry
}

end minimum_pie_pieces_l194_194843


namespace questions_two_and_four_equiv_questions_three_and_seven_equiv_l194_194615

-- Definitions representing conditions about students in classes A and B:
def ClassA (student : Student) : Prop := sorry
def ClassB (student : Student) : Prop := sorry
def taller (x y : Student) : Prop := sorry
def shorter (x y : Student) : Prop := sorry
def tallest (students : Set Student) : Student := sorry
def shortest (students : Set Student) : Student := sorry
def averageHeight (students : Set Student) : ℝ := sorry
def totalHeight (students : Set Student) : ℝ := sorry
def medianHeight (students : Set Student) : ℝ := sorry

-- Equivalence of question 2 and question 4:
theorem questions_two_and_four_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, taller a b) ↔ 
  (∀ b ∈ students_B, ∃ a ∈ students_A, taller a b) :=
sorry

-- Equivalence of question 3 and question 7:
theorem questions_three_and_seven_equiv (students_A students_B : Set Student) :
  (∀ a ∈ students_A, ∃ b ∈ students_B, shorter b a) ↔ 
  (shorter (shortest students_B) (shortest students_A)) :=
sorry

end questions_two_and_four_equiv_questions_three_and_seven_equiv_l194_194615


namespace attendance_proof_l194_194194

noncomputable def next_year_attendance (this_year: ℕ) := 2 * this_year
noncomputable def last_year_attendance (next_year: ℕ) := next_year - 200
noncomputable def total_attendance (last_year this_year next_year: ℕ) := last_year + this_year + next_year

theorem attendance_proof (this_year: ℕ) (h1: this_year = 600):
    total_attendance (last_year_attendance (next_year_attendance this_year)) this_year (next_year_attendance this_year) = 2800 :=
by
  sorry

end attendance_proof_l194_194194


namespace trig_expression_eq_zero_l194_194299

theorem trig_expression_eq_zero (α : ℝ) (h1 : Real.sin α = -2 / Real.sqrt 5) (h2 : Real.cos α = 1 / Real.sqrt 5) :
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 0 := by
  sorry

end trig_expression_eq_zero_l194_194299


namespace coffee_mix_price_l194_194606

theorem coffee_mix_price 
  (P : ℝ)
  (pound_2nd : ℝ := 2.45)
  (total_pounds : ℝ := 18)
  (final_price_per_pound : ℝ := 2.30)
  (pounds_each_kind : ℝ := 9) :
  9 * P + 9 * pound_2nd = total_pounds * final_price_per_pound →
  P = 2.15 :=
by
  intros h
  sorry

end coffee_mix_price_l194_194606


namespace total_candies_third_set_l194_194581

-- Variables for the quantities of candies
variables (L1 L2 L3 S1 S2 S3 M1 M2 M3 : ℕ)

-- Conditions as stated in the problem
axiom equal_totals : L1 + L2 + L3 = S1 + S2 + S3 ∧ S1 + S2 + S3 = M1 + M2 + M3
axiom first_set_chocolates_gummy : S1 = M1
axiom first_set_hard_candies : L1 = S1 + 7
axiom second_set_hard_chocolates : L2 = S2
axiom second_set_gummy_candies : M2 = L2 - 15
axiom third_set_no_hard_candies : L3 = 0

-- Theorem to be proven: the number of candies in the third set
theorem total_candies_third_set : 
  L3 + S3 + M3 = 29 :=
by
  sorry

end total_candies_third_set_l194_194581


namespace geom_seq_sum_relation_l194_194010

variable {a : ℕ → ℝ}
variable {q : ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_sum_relation (h_geom : is_geometric_sequence a q)
  (h_pos : ∀ n, a n > 0) (h_q_ne_one : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 :=
by
  sorry

end geom_seq_sum_relation_l194_194010


namespace minimum_people_in_troupe_l194_194832

-- Let n be the number of people in the troupe.
variable (n : ℕ)

-- Conditions: n must be divisible by 8, 10, and 12.
def is_divisible_by (m k : ℕ) := m % k = 0
def divides_all (n : ℕ) := is_divisible_by n 8 ∧ is_divisible_by n 10 ∧ is_divisible_by n 12

-- The minimum number of people in the troupe that can form groups of 8, 10, or 12 with none left over.
theorem minimum_people_in_troupe (n : ℕ) : divides_all n → n = 120 :=
by
  sorry

end minimum_people_in_troupe_l194_194832


namespace find_unknown_towel_rate_l194_194506

theorem find_unknown_towel_rate 
    (cost_known1 : ℕ := 300)
    (cost_known2 : ℕ := 750)
    (total_towels : ℕ := 10)
    (average_price : ℕ := 150)
    (total_cost : ℕ := total_towels * average_price) :
  let total_cost_known := cost_known1 + cost_known2
  let cost_unknown := 2 * x
  300 + 750 + 2 * x = total_cost → x = 225 :=
by
  sorry

end find_unknown_towel_rate_l194_194506


namespace units_digit_of_product_l194_194522

theorem units_digit_of_product : 
  (3 ^ 401 * 7 ^ 402 * 23 ^ 403) % 10 = 9 := 
by
  sorry

end units_digit_of_product_l194_194522


namespace ordered_pairs_count_l194_194324

theorem ordered_pairs_count : 
    ∃ (s : Finset (ℝ × ℝ)), 
        (∀ (x y : ℝ), (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1 ↔ (x, y) ∈ s)) ∧ 
        s.card = 3 :=
    by
    sorry

end ordered_pairs_count_l194_194324


namespace geom_seq_q_eq_l194_194962

theorem geom_seq_q_eq (a1 : ℕ := 2) (S3 : ℕ := 26) 
  (h1 : a1 = 2) 
  (h2 : S3 = 26) : 
  ∃ q : ℝ, (q = 3 ∨ q = -4) := by
  sorry

end geom_seq_q_eq_l194_194962


namespace Zachary_did_47_pushups_l194_194292

-- Define the conditions and the question
def Zachary_pushups (David_pushups difference : ℕ) : ℕ :=
  David_pushups - difference

theorem Zachary_did_47_pushups :
  Zachary_pushups 62 15 = 47 :=
by
  -- Provide the proof here (we'll use sorry for now)
  sorry

end Zachary_did_47_pushups_l194_194292


namespace revenue_fraction_large_cups_l194_194639

theorem revenue_fraction_large_cups (total_cups : ℕ) (price_small : ℚ) (price_large : ℚ)
  (h1 : price_large = (7 / 6) * price_small) 
  (h2 : (1 / 5 : ℚ) * total_cups = total_cups - (4 / 5 : ℚ) * total_cups) :
  ((4 / 5 : ℚ) * (7 / 6 * price_small) * total_cups) / 
  (((1 / 5 : ℚ) * price_small + (4 / 5 : ℚ) * (7 / 6 * price_small)) * total_cups) = (14 / 17 : ℚ) :=
by
  intros
  have h_total_small := (1 / 5 : ℚ) * total_cups
  have h_total_large := (4 / 5 : ℚ) * total_cups
  have revenue_small := h_total_small * price_small
  have revenue_large := h_total_large * price_large
  have total_revenue := revenue_small + revenue_large
  have revenue_large_frac := revenue_large / total_revenue
  have target_frac := (14 / 17 : ℚ)
  have target := revenue_large_frac = target_frac
  sorry

end revenue_fraction_large_cups_l194_194639


namespace increasing_power_function_l194_194220

theorem increasing_power_function (m : ℝ) (h_power : m^2 - 1 = 1)
    (h_increasing : ∀ x : ℝ, x > 0 → (m^2 - 1) * m * x^(m-1) > 0) : m = Real.sqrt 2 :=
by
  sorry

end increasing_power_function_l194_194220


namespace terry_lunch_combos_l194_194947

def num_lettuce : ℕ := 2
def num_tomatoes : ℕ := 3
def num_olives : ℕ := 4
def num_soups : ℕ := 2

theorem terry_lunch_combos : num_lettuce * num_tomatoes * num_olives * num_soups = 48 :=
by
  sorry

end terry_lunch_combos_l194_194947


namespace student_chose_124_l194_194878

theorem student_chose_124 (x : ℤ) (h : 2 * x - 138 = 110) : x = 124 := 
by {
  sorry
}

end student_chose_124_l194_194878


namespace find_divisor_l194_194300

-- Definitions
def dividend := 199
def quotient := 11
def remainder := 1

-- Statement of the theorem
theorem find_divisor : ∃ x : ℕ, dividend = (x * quotient) + remainder ∧ x = 18 := by
  sorry

end find_divisor_l194_194300


namespace circles_intersect_at_two_points_l194_194872

noncomputable def point_intersection_count (A B : ℝ × ℝ) (rA rB d : ℝ) : ℕ :=
  let distance := (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2
  if rA + rB >= d ∧ d >= |rA - rB| then 2 else if d = rA + rB ∨ d = |rA - rB| then 1 else 0

theorem circles_intersect_at_two_points :
  point_intersection_count (0, 0) (8, 0) 3 6 8 = 2 :=
by 
  -- Proof for the statement will go here
  sorry

end circles_intersect_at_two_points_l194_194872


namespace possible_value_is_121_l194_194268

theorem possible_value_is_121
  (x a y z b : ℕ) 
  (hx : x = 1 / 6 * a) 
  (hz : z = 1 / 6 * b) 
  (hy : y = (a + b) % 5) 
  (h_single_digit : ∀ n, n ∈ [x, a, y, z, b] → n < 10 ∧ 0 < n) : 
  100 * x + 10 * y + z = 121 :=
by
  sorry

end possible_value_is_121_l194_194268


namespace paperback_copies_sold_l194_194434

theorem paperback_copies_sold
  (H : ℕ) (P : ℕ)
  (h1 : H = 36000)
  (h2 : P = 9 * H)
  (h3 : H + P = 440000) :
  P = 360000 := by
  sorry

end paperback_copies_sold_l194_194434


namespace simplify_expression_l194_194799

variable (t : ℝ)

theorem simplify_expression (ht : t > 0) (ht_ne : t ≠ 1 / 2) :
  (1 - Real.sqrt (2 * t)) / ( (1 - Real.sqrt (4 * t ^ (3 / 4))) / (1 - Real.sqrt (2 * t ^ (1 / 4))) - Real.sqrt (2 * t)) *
  (Real.sqrt (1 / (1 / 2) + Real.sqrt (4 * t ^ 2)) / (1 + Real.sqrt (1 / (2 * t))) - Real.sqrt (2 * t))⁻¹ = 1 :=
by
  sorry

end simplify_expression_l194_194799


namespace determine_values_of_a_and_b_l194_194559

def ab_product_eq_one (a b : ℝ) : Prop := a * b = 1

def given_equation (a b : ℝ) : Prop :=
  (a + b + 2) / 4 = (1 / (a + 1)) + (1 / (b + 1))

theorem determine_values_of_a_and_b (a b : ℝ) (h1 : ab_product_eq_one a b) (h2 : given_equation a b) :
  a = 1 ∧ b = 1 :=
by
  sorry

end determine_values_of_a_and_b_l194_194559


namespace perfect_squares_less_than_20000_representable_l194_194115

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the difference of two consecutive perfect squares
def consecutive_difference (b : ℕ) : ℕ :=
  (b + 1) ^ 2 - b ^ 2

-- Define the condition under which the perfect square is less than 20000
def less_than_20000 (n : ℕ) : Prop :=
  n < 20000

-- Define the main problem statement using the above definitions
theorem perfect_squares_less_than_20000_representable :
  ∃ count : ℕ, (∀ n : ℕ, (is_perfect_square n) ∧ (less_than_20000 n) →
  ∃ b : ℕ, n = consecutive_difference b) ∧ count = 69 :=
sorry

end perfect_squares_less_than_20000_representable_l194_194115


namespace polygon_to_triangle_l194_194214

theorem polygon_to_triangle {n : ℕ} (h : n > 4) :
  ∃ (a b c : ℕ), (a + b > c ∧ a + c > b ∧ b + c > a) :=
sorry

end polygon_to_triangle_l194_194214


namespace mary_cut_10_roses_l194_194941

-- Define the initial and final number of roses
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- Define the number of roses cut as the difference between final and initial
def roses_cut : ℕ :=
  final_roses - initial_roses

-- Theorem stating the number of roses cut by Mary
theorem mary_cut_10_roses : roses_cut = 10 := by
  sorry

end mary_cut_10_roses_l194_194941


namespace right_triangle_area_l194_194335

theorem right_triangle_area (a : ℝ) (r : ℝ) (area : ℝ) :
  a = 3 → r = 3 / 8 → area = 21 / 16 :=
by 
  sorry

end right_triangle_area_l194_194335


namespace sum_of_lengths_of_edges_geometric_progression_l194_194196

theorem sum_of_lengths_of_edges_geometric_progression :
  ∃ (a r : ℝ), (a / r) * a * (a * r) = 8 ∧ 2 * (a / r * a + a * a * r + a * r * a / r) = 32 ∧ 
  4 * ((a / r) + a + (a * r)) = 32 :=
by
  sorry

end sum_of_lengths_of_edges_geometric_progression_l194_194196


namespace rahul_spends_10_percent_on_clothes_l194_194377

theorem rahul_spends_10_percent_on_clothes 
    (salary : ℝ) (house_rent_percent : ℝ) (education_percent : ℝ) (remaining_after_expense : ℝ) (expenses : ℝ) (clothes_percent : ℝ) 
    (h_salary : salary = 2125) 
    (h_house_rent_percent : house_rent_percent = 0.20)
    (h_education_percent : education_percent = 0.10)
    (h_remaining_after_expense : remaining_after_expense = 1377)
    (h_expenses : expenses = salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)
    (h_clothes_expense : remaining_after_expense = salary - (salary * house_rent_percent + (salary - salary * house_rent_percent) * education_percent + (salary - salary * house_rent_percent - (salary - salary * house_rent_percent) * education_percent) * clothes_percent)) :
    clothes_percent = 0.10 := 
by 
  sorry

end rahul_spends_10_percent_on_clothes_l194_194377


namespace crayons_left_l194_194969

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
  (initial_crayons_eq : initial_crayons = 48) 
  (kiley_fraction_eq : kiley_fraction = 1/4) 
  (joe_fraction_eq : joe_fraction = 1/2): 
  (initial_crayons - initial_crayons * kiley_fraction - (initial_crayons - initial_crayons * kiley_fraction) * joe_fraction) = 18 := 
by 
  sorry

end crayons_left_l194_194969


namespace percentage_women_red_and_men_dark_l194_194426

-- Define the conditions as variables
variables (w_fair_hair w_dark_hair w_red_hair m_fair_hair m_dark_hair m_red_hair : ℝ)

-- Define the percentage of women with red hair and men with dark hair
def women_red_men_dark (w_red_hair m_dark_hair : ℝ) : ℝ := w_red_hair + m_dark_hair

-- Define the main theorem to be proven
theorem percentage_women_red_and_men_dark 
  (hw_fair_hair : w_fair_hair = 30)
  (hw_dark_hair : w_dark_hair = 28)
  (hw_red_hair : w_red_hair = 12)
  (hm_fair_hair : m_fair_hair = 20)
  (hm_dark_hair : m_dark_hair = 35)
  (hm_red_hair : m_red_hair = 5) :
  women_red_men_dark w_red_hair m_dark_hair = 47 := 
sorry

end percentage_women_red_and_men_dark_l194_194426


namespace donovan_correct_answers_l194_194599

variable (C : ℝ)
variable (incorrectAnswers : ℝ := 13)
variable (percentageCorrect : ℝ := 0.7292)

theorem donovan_correct_answers :
  (C / (C + incorrectAnswers)) = percentageCorrect → C = 35 := by
  sorry

end donovan_correct_answers_l194_194599


namespace cones_to_cylinder_volume_ratio_l194_194151

theorem cones_to_cylinder_volume_ratio :
  let π := Real.pi
  let r_cylinder := 4
  let h_cylinder := 18
  let r_cone := 4
  let h_cone1 := 6
  let h_cone2 := 9
  let V_cylinder := π * r_cylinder^2 * h_cylinder
  let V_cone1 := (1 / 3) * π * r_cone^2 * h_cone1
  let V_cone2 := (1 / 3) * π * r_cone^2 * h_cone2
  let V_totalCones := V_cone1 + V_cone2
  V_totalCones / V_cylinder = 5 / 18 :=
by
  sorry

end cones_to_cylinder_volume_ratio_l194_194151


namespace meters_to_centimeters_l194_194342

theorem meters_to_centimeters : (3.5 : ℝ) * 100 = 350 :=
by
  sorry

end meters_to_centimeters_l194_194342


namespace find_value_l194_194210

theorem find_value (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^2 + b^2 + a * b = 7 :=
by
  sorry

end find_value_l194_194210


namespace c_share_l194_194905

theorem c_share (A B C : ℕ) (h1 : A = B / 2) (h2 : B = C / 2) (h3 : A + B + C = 392) : C = 224 :=
by
  sorry

end c_share_l194_194905


namespace part1_part2_l194_194354

noncomputable def f (x a : ℝ) : ℝ := |x + a|
noncomputable def g (x : ℝ) : ℝ := |x + 3| - x

theorem part1 (x : ℝ) : f x 1 < g x → x < 2 :=
sorry

theorem part2 (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x a < g x) → -2 < a ∧ a < 2 :=
sorry

end part1_part2_l194_194354


namespace remaining_bollards_to_be_installed_l194_194963

-- Definitions based on the problem's conditions
def total_bollards_per_side := 4000
def sides := 2
def total_bollards := total_bollards_per_side * sides
def installed_fraction := 3 / 4
def installed_bollards := installed_fraction * total_bollards
def remaining_bollards := total_bollards - installed_bollards

-- The statement to be proved
theorem remaining_bollards_to_be_installed : remaining_bollards = 2000 :=
  by sorry

end remaining_bollards_to_be_installed_l194_194963


namespace smallest_possible_integer_l194_194968

theorem smallest_possible_integer (a b : ℤ)
  (a_lt_10 : a < 10)
  (b_lt_10 : b < 10)
  (a_lt_b : a < b)
  (sum_eq_45 : a + b + 32 = 45)
  : a = 4 :=
by
  sorry

end smallest_possible_integer_l194_194968


namespace negative_integer_solutions_l194_194889

theorem negative_integer_solutions (x : ℤ) : 3 * x + 1 ≥ -5 ↔ x = -2 ∨ x = -1 := 
by
  sorry

end negative_integer_solutions_l194_194889


namespace determine_q_l194_194319

theorem determine_q (q : ℝ → ℝ) :
  (∀ x, q x = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ k : ℝ, k < 3) ∧ -- indicating degree considerations for asymptotes
  (q 2 = 18) →
  q = (fun x => (-18 / 5) * x ^ 2 + 162 / 5) :=
by
  sorry

end determine_q_l194_194319


namespace no_rational_roots_of_odd_coeffs_l194_194276

theorem no_rational_roots_of_odd_coeffs (a b c : ℤ) (h_a_odd : a % 2 = 1) (h_b_odd : b % 2 = 1) (h_c_odd : c % 2 = 1)
  (h_rational_root : ∃ (p q : ℤ), q ≠ 0 ∧ (a * (p / q : ℚ)^2 + b * (p / q : ℚ) + c = 0)) : false :=
sorry

end no_rational_roots_of_odd_coeffs_l194_194276


namespace P_sufficient_but_not_necessary_for_Q_l194_194539

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |x - 2| ≤ 3
def Q (x : ℝ) : Prop := x ≥ -1 ∨ x ≤ 5

-- Define the statement to prove
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬ P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l194_194539


namespace arithmetic_seq_a8_l194_194517

theorem arithmetic_seq_a8
  (a : ℕ → ℤ)
  (h1 : a 5 = 10)
  (h2 : a 1 + a 2 + a 3 = 3) :
  a 8 = 19 := sorry

end arithmetic_seq_a8_l194_194517


namespace hyperbola_asymptotes_l194_194421

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 16 - y^2 / 9 = 1) → (y = 3/4 * x ∨ y = -3/4 * x) :=
by
  sorry

end hyperbola_asymptotes_l194_194421


namespace find_m_l194_194993

-- Mathematical conditions definitions
def line1 (x y : ℝ) (m : ℝ) : Prop := 3 * x + m * y - 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0

-- Given the lines are parallel
def lines_parallel (l1 l2 : ℝ → ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∀ x y : ℝ, l1 x y m → l2 x y m → (3 / (m + 2)) = (m / (-(m - 2)))

-- The proof problem statement
theorem find_m (m : ℝ) : 
  lines_parallel (line1) (line2) m → (m = -6 ∨ m = 1) :=
by
  sorry

end find_m_l194_194993


namespace kate_needs_more_money_l194_194336

theorem kate_needs_more_money
  (pen_price : ℝ)
  (notebook_price : ℝ)
  (artset_price : ℝ)
  (kate_pen_money_fraction : ℝ)
  (notebook_discount : ℝ)
  (artset_discount : ℝ)
  (kate_artset_money : ℝ) :
  pen_price = 30 →
  notebook_price = 20 →
  artset_price = 50 →
  kate_pen_money_fraction = 1/3 →
  notebook_discount = 0.15 →
  artset_discount = 0.4 →
  kate_artset_money = 10 →
  (pen_price - kate_pen_money_fraction * pen_price) +
  (notebook_price * (1 - notebook_discount)) +
  (artset_price * (1 - artset_discount) - kate_artset_money) = 57 :=
by
  sorry

end kate_needs_more_money_l194_194336


namespace length_of_bridge_l194_194578

/-- A train that is 357 meters long is running at a speed of 42 km/hour. 
    It takes 42.34285714285714 seconds to pass a bridge. 
    Prove that the length of the bridge is 136.7142857142857 meters. -/
theorem length_of_bridge : 
  let train_length := 357 -- meters
  let speed_kmh := 42 -- km/hour
  let passing_time := 42.34285714285714 -- seconds
  let speed_mps := 42 * (1000 / 3600) -- meters/second
  let total_distance := speed_mps * passing_time -- meters
  let bridge_length := total_distance - train_length -- meters
  bridge_length = 136.7142857142857 :=
by
  sorry

end length_of_bridge_l194_194578


namespace participants_count_l194_194390

theorem participants_count (x y : ℕ) 
    (h1 : y = x + 41)
    (h2 : y = 3 * x - 35) : 
    x = 38 ∧ y = 79 :=
by
  sorry

end participants_count_l194_194390


namespace games_played_in_tournament_l194_194031

def number_of_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem games_played_in_tournament : number_of_games 18 = 153 :=
  by
    sorry

end games_played_in_tournament_l194_194031


namespace flora_needs_more_daily_l194_194596

-- Definitions based on conditions
def totalMilk : ℕ := 105   -- Total milk requirement in gallons
def weeks : ℕ := 3         -- Total weeks
def daysInWeek : ℕ := 7    -- Days per week
def floraPlan : ℕ := 3     -- Flora's planned gallons per day

-- Proof statement
theorem flora_needs_more_daily : (totalMilk / (weeks * daysInWeek)) - floraPlan = 2 := 
by
  sorry

end flora_needs_more_daily_l194_194596


namespace proof_problem_l194_194830

variable (α β : ℝ) (a b : ℝ × ℝ) (m : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 4)
variable (hβ : β = Real.pi)
variable (ha_def : a = (Real.tan (α + β / 4) - 1, 0))
variable (hb : b = (Real.cos α, 2))
variable (ha_dot : a.1 * b.1 + a.2 * b.2 = m)

-- Proof statement
theorem proof_problem :
  (0 < α ∧ α < Real.pi / 4) ∧
  β = Real.pi ∧
  a = (Real.tan (α + β / 4) - 1, 0) ∧
  b = (Real.cos α, 2) ∧
  (a.1 * b.1 + a.2 * b.2 = m) →
  (2 * Real.cos α * Real.cos α + Real.sin (2 * (α + β))) / (Real.cos α - Real.sin β) = 2 * (m + 2) := by
  sorry

end proof_problem_l194_194830


namespace repeating_decimal_to_fraction_l194_194954

theorem repeating_decimal_to_fraction : (∃ (x : ℚ), x = 0.4 + 4 / 9) :=
sorry

end repeating_decimal_to_fraction_l194_194954


namespace number_of_members_l194_194040

theorem number_of_members (n : ℕ) (h1 : n * n = 5929) : n = 77 :=
sorry

end number_of_members_l194_194040


namespace sum_of_coordinates_of_B_l194_194380

-- Definitions
def Point := (ℝ × ℝ)
def isMidpoint (M A B : Point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Given conditions
def M : Point := (4, 8)
def A : Point := (10, 4)

-- Statement to prove
theorem sum_of_coordinates_of_B (B : Point) (h : isMidpoint M A B) :
  B.1 + B.2 = 10 :=
by
  sorry

end sum_of_coordinates_of_B_l194_194380


namespace exists_three_distinct_integers_in_A_l194_194078

noncomputable def A (m n : ℤ) : Set ℤ := { x^2 + m * x + n | x : ℤ }

theorem exists_three_distinct_integers_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a = b * c :=
by
  sorry

end exists_three_distinct_integers_in_A_l194_194078


namespace arithmetic_sequence_sum_false_statement_l194_194933

theorem arithmetic_sequence_sum_false_statement (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n.succ - a_n n = a_n 1 - a_n 0)
  (h_S : ∀ n, S n = (n + 1) * a_n 0 + (n * (n + 1) * (a_n 1 - a_n 0)) / 2)
  (h1 : S 6 < S 7) (h2 : S 7 = S 8) (h3 : S 8 > S 9) : ¬ (S 10 > S 6) :=
by
  sorry

end arithmetic_sequence_sum_false_statement_l194_194933


namespace geom_seq_S6_l194_194503

theorem geom_seq_S6 :
  ∃ (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ),
  (q = 2) →
  (S 3 = 7) →
  (∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) →
  S 6 = 63 :=
sorry

end geom_seq_S6_l194_194503


namespace not_all_on_C_implies_exists_not_on_C_l194_194957

def F (x y : ℝ) : Prop := sorry  -- Define F according to specifics
def on_curve_C (x y : ℝ) : Prop := sorry -- Define what it means to be on curve C according to specifics

theorem not_all_on_C_implies_exists_not_on_C (h : ¬ ∀ x y : ℝ, F x y → on_curve_C x y) :
  ∃ x y : ℝ, F x y ∧ ¬ on_curve_C x y := sorry

end not_all_on_C_implies_exists_not_on_C_l194_194957


namespace kenya_peanuts_correct_l194_194104

def jose_peanuts : ℕ := 85
def kenya_extra_peanuts : ℕ := 48
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_correct : kenya_peanuts = 133 := 
by 
  sorry

end kenya_peanuts_correct_l194_194104


namespace A_n_eq_B_n_l194_194895

open Real

noncomputable def A_n (n : ℕ) : ℝ :=
  1408 * (1 - (1 / (2 : ℝ) ^ n))

noncomputable def B_n (n : ℕ) : ℝ :=
  (3968 / 3) * (1 - (1 / (-2 : ℝ) ^ n))

theorem A_n_eq_B_n : A_n 5 = B_n 5 := sorry

end A_n_eq_B_n_l194_194895


namespace printer_diff_l194_194931

theorem printer_diff (A B : ℚ) (hA : A * 60 = 35) (hAB : (A + B) * 24 = 35) : B - A = 7 / 24 := by
  sorry

end printer_diff_l194_194931


namespace cost_of_flute_l194_194227

def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7
def flute_cost : ℝ := 142.46

theorem cost_of_flute :
  total_spent - (music_stand_cost + song_book_cost) = flute_cost :=
by
  sorry

end cost_of_flute_l194_194227


namespace polygon_sides_l194_194135

theorem polygon_sides {n : ℕ} (h : (n - 2) * 180 = 1080) : n = 8 :=
by
  sorry

end polygon_sides_l194_194135


namespace pythagorean_triangle_product_divisible_by_60_l194_194703

theorem pythagorean_triangle_product_divisible_by_60 : 
  ∀ (a b c : ℕ),
  (∃ m n : ℕ,
  m > n ∧ (m % 2 = 0 ∨ n % 2 = 0) ∧ m.gcd n = 1 ∧
  a = m^2 - n^2 ∧ b = 2 * m * n ∧ c = m^2 + n^2 ∧ a^2 + b^2 = c^2) →
  60 ∣ (a * b * c) :=
sorry

end pythagorean_triangle_product_divisible_by_60_l194_194703


namespace simplify_fraction_l194_194023

theorem simplify_fraction (a b m : ℝ) (h1 : (a / b) ^ m = (a^m) / (b^m)) (h2 : (-1 : ℝ) ^ (0 : ℝ) = 1) :
  ( (81 / 16) ^ (3 / 4) ) - 1 = 19 / 8 :=
by
  sorry

end simplify_fraction_l194_194023


namespace compare_M_N_l194_194597

variables (a : ℝ)

-- Definitions based on given conditions
def M : ℝ := 2 * a * (a - 2) + 3
def N : ℝ := (a - 1) * (a - 3)

theorem compare_M_N : M a ≥ N a := 
by {
  sorry
}

end compare_M_N_l194_194597


namespace rectangle_area_excluding_hole_l194_194026

theorem rectangle_area_excluding_hole (x : ℝ) (h : x > 5 / 3) :
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  A_large - A_hole = -x^2 + 17 * x + 38 :=
by
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  sorry

end rectangle_area_excluding_hole_l194_194026


namespace triangle_side_length_l194_194057

variable (A C : ℝ) (a c b : ℝ)

theorem triangle_side_length (h1 : c = 48) (h2 : a = 27) (h3 : C = 3 * A) : b = 35 := by
  sorry

end triangle_side_length_l194_194057


namespace basketball_player_ft_rate_l194_194245

theorem basketball_player_ft_rate :
  ∃ P : ℝ, 1 - P^2 = 16 / 25 ∧ P = 3 / 5 := sorry

end basketball_player_ft_rate_l194_194245


namespace impossible_return_l194_194489

def Point := (ℝ × ℝ)

-- Conditions
def is_valid_point (p: Point) : Prop :=
  let (a, b) := p
  ∃ a_int b_int : ℤ, (a = a_int + b_int * Real.sqrt 2 ∧ b = a_int + b_int * Real.sqrt 2)

def valid_movement (p q: Point) : Prop :=
  let (x1, y1) := p
  let (x2, y2) := q
  abs x2 > abs x1 ∧ abs y2 > abs y1 

-- Theorem statement
theorem impossible_return (start: Point) (h: start = (1, Real.sqrt 2)) 
  (valid_start: is_valid_point start) :
  ∀ (p: Point), (is_valid_point p ∧ valid_movement start p) → p ≠ start :=
sorry

end impossible_return_l194_194489


namespace lines_perpendicular_l194_194508

theorem lines_perpendicular 
  (a b : ℝ) (θ : ℝ)
  (L1 : ∀ x y : ℝ, x * Real.cos θ + y * Real.sin θ + a = 0)
  (L2 : ∀ x y : ℝ, x * Real.sin θ - y * Real.cos θ + b = 0)
  : ∀ m1 m2 : ℝ, m1 = -(Real.cos θ) / (Real.sin θ) → m2 = (Real.sin θ) / (Real.cos θ) → m1 * m2 = -1 :=
by 
  intros m1 m2 h1 h2
  sorry

end lines_perpendicular_l194_194508


namespace sum_of_coeffs_eq_92_l194_194555

noncomputable def sum_of_integer_coeffs_in_factorization (x y : ℝ) : ℝ :=
  let f := 27 * (x ^ 6) - 512 * (y ^ 6)
  3 - 8 + 9 + 24 + 64  -- Sum of integer coefficients

theorem sum_of_coeffs_eq_92 (x y : ℝ) : sum_of_integer_coeffs_in_factorization x y = 92 :=
by
  -- proof steps go here
  sorry

end sum_of_coeffs_eq_92_l194_194555


namespace excircle_diameter_l194_194017

noncomputable def diameter_of_excircle (a b c S : ℝ) (s : ℝ) : ℝ :=
  2 * S / (s - a)

theorem excircle_diameter (a b c S h_A : ℝ) (s : ℝ) (h_v : 2 * ((a + b + c) / 2) = a + b + c) :
    diameter_of_excircle a b c S s = 2 * S / (s - a) :=
by
  sorry

end excircle_diameter_l194_194017


namespace total_wheels_l194_194881

theorem total_wheels (bicycles tricycles : ℕ) (wheels_per_bicycle wheels_per_tricycle : ℕ) 
  (h1 : bicycles = 50) (h2 : tricycles = 20) (h3 : wheels_per_bicycle = 2) (h4 : wheels_per_tricycle = 3) : 
  (bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160) :=
by
  sorry

end total_wheels_l194_194881


namespace bowling_average_before_last_match_l194_194645

theorem bowling_average_before_last_match
  (wickets_before_last : ℕ)
  (wickets_last_match : ℕ)
  (runs_last_match : ℕ)
  (decrease_in_average : ℝ)
  (average_before_last : ℝ) :

  wickets_before_last = 115 →
  wickets_last_match = 6 →
  runs_last_match = 26 →
  decrease_in_average = 0.4 →
  (average_before_last - decrease_in_average) = 
  ((wickets_before_last * average_before_last + runs_last_match) / 
  (wickets_before_last + wickets_last_match)) →
  average_before_last = 12.4 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end bowling_average_before_last_match_l194_194645


namespace truncated_pyramid_volume_ratio_l194_194189

/-
Statement: Given a truncated triangular pyramid with a plane drawn through a side of the upper base parallel to the opposite lateral edge,
and the corresponding sides of the bases in the ratio 1:2, prove that the volume of the truncated pyramid is divided in the ratio 3:4.
-/

theorem truncated_pyramid_volume_ratio (S1 S2 h : ℝ) 
  (h_ratio : S1 = 4 * S2) :
  (h * S2) / ((7 * h * S2) / 3 - h * S2) = 3 / 4 :=
by
  sorry

end truncated_pyramid_volume_ratio_l194_194189


namespace solve_inequality_l194_194973

theorem solve_inequality (x : ℝ) : 3 * (x + 1) > 9 → x > 2 :=
by sorry

end solve_inequality_l194_194973


namespace prime_sum_of_primes_unique_l194_194190

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_sum_of_primes_unique (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum_prime : is_prime (p^q + q^p)) :
  p = 2 ∧ q = 3 :=
sorry

end prime_sum_of_primes_unique_l194_194190


namespace binom_arithmetic_sequence_l194_194651

noncomputable def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_arithmetic_sequence {n : ℕ} (h : 2 * binom n 5 = binom n 4 + binom n 6) (n_eq : n = 14) : binom n 12 = 91 := by
  sorry

end binom_arithmetic_sequence_l194_194651


namespace credibility_of_relationship_l194_194200

theorem credibility_of_relationship
  (sample_size : ℕ)
  (chi_squared_value : ℝ)
  (table : ℕ → ℝ × ℝ)
  (h_sample : sample_size = 5000)
  (h_chi_squared : chi_squared_value = 6.109)
  (h_table : table 5 = (5.024, 0.025) ∧ table 6 = (6.635, 0.010)) :
  credible_percent = 97.5 :=
by
  sorry

end credibility_of_relationship_l194_194200


namespace lowest_score_within_two_std_devs_l194_194535

variable (mean : ℝ) (std_dev : ℝ) (jack_score : ℝ)

def within_two_std_devs (mean : ℝ) (std_dev : ℝ) (score : ℝ) : Prop :=
  score >= mean - 2 * std_dev

theorem lowest_score_within_two_std_devs :
  mean = 60 → std_dev = 10 → within_two_std_devs mean std_dev jack_score → (40 ≤ jack_score) :=
by
  intros h1 h2 h3
  change mean = 60 at h1
  change std_dev = 10 at h2
  sorry

end lowest_score_within_two_std_devs_l194_194535


namespace product_of_numbers_l194_194411

theorem product_of_numbers (x y : ℕ) (h1 : x + y = 15) (h2 : x - y = 11) : x * y = 26 :=
by
  sorry

end product_of_numbers_l194_194411


namespace speed_of_first_train_l194_194684

theorem speed_of_first_train
  (length_train1 length_train2 : ℕ)
  (speed_train2 : ℕ)
  (time_seconds : ℝ)
  (distance_km : ℝ := (length_train1 + length_train2) / 1000)
  (time_hours : ℝ := time_seconds / 3600)
  (relative_speed : ℝ := distance_km / time_hours) :
  length_train1 = 111 →
  length_train2 = 165 →
  speed_train2 = 120 →
  time_seconds = 4.516002356175142 →
  relative_speed = 220 →
  speed_train2 + 100 = relative_speed :=
by
  intros
  sorry

end speed_of_first_train_l194_194684


namespace locus_of_C_l194_194293

variable (a : ℝ) (h : a > 0)

theorem locus_of_C : 
  ∃ (x y : ℝ), 
  (1 - a) * x^2 - 2 * a * x + (1 + a) * y^2 = 0 :=
sorry

end locus_of_C_l194_194293


namespace difference_of_two_distinct_members_sum_of_two_distinct_members_l194_194702

theorem difference_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ N, N = 19 ∧ (∀ n, 1 ≤ n ∧ n ≤ N → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ n = a - b)) :=
by
  sorry

theorem sum_of_two_distinct_members (S : Set ℕ) (h : S = {n | n ∈ Finset.range 20 ∧ 1 ≤ n ∧ n ≤ 20}) :
  (∃ M, M = 37 ∧ (∀ m, 3 ≤ m ∧ m ≤ 39 → ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ m = a + b)) :=
by
  sorry

end difference_of_two_distinct_members_sum_of_two_distinct_members_l194_194702


namespace verify_min_n_for_coprime_subset_l194_194101

def is_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∀ (a b : ℕ) (ha : a ∈ s) (hb : b ∈ s), a ≠ b → Nat.gcd a b = 1

def contains_4_pairwise_coprime (s : Finset ℕ) : Prop :=
  ∃ t : Finset ℕ, t ⊆ s ∧ t.card = 4 ∧ is_pairwise_coprime t

def min_n_for_coprime_subset : ℕ :=
  111

theorem verify_min_n_for_coprime_subset (S : Finset ℕ) (hS : S = Finset.range 151) :
  ∀ (n : ℕ), (∀ s : Finset ℕ, s ⊆ S ∧ s.card = n → contains_4_pairwise_coprime s) ↔ (n ≥ min_n_for_coprime_subset) :=
sorry

end verify_min_n_for_coprime_subset_l194_194101


namespace cos_A_eq_a_eq_l194_194471

-- Defining the problem conditions:
variables {A B C a b c : ℝ}
variable (sin_eq : Real.sin (B + C) = 3 * Real.sin (A / 2) ^ 2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 6)
variable (sum_eq : b + c = 8)
variable (bc_prod_eq : b * c = 13)
variable (cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)

-- Proving the statements:
theorem cos_A_eq : Real.cos A = 5 / 13 :=
sorry

theorem a_eq : a = 3 * Real.sqrt 2 :=
sorry

end cos_A_eq_a_eq_l194_194471


namespace paperboy_delivery_sequences_l194_194677

noncomputable def D : ℕ → ℕ
| 0       => 1  -- D_0 is a dummy value to facilitate indexing
| 1       => 2
| 2       => 4
| 3       => 7
| (n + 4) => D (n + 3) + D (n + 2) + D (n + 1)

theorem paperboy_delivery_sequences : D 11 = 927 := by
  sorry

end paperboy_delivery_sequences_l194_194677


namespace field_ratio_l194_194998

theorem field_ratio (w : ℝ) (h : ℝ) (pond_len : ℝ) (field_len : ℝ) 
  (h1 : pond_len = 8) 
  (h2 : field_len = 112) 
  (h3 : w > 0) 
  (h4 : field_len = w * h) 
  (h5 : pond_len * pond_len = (1 / 98) * (w * h * h)) : 
  field_len / h = 2 := 
by 
  sorry

end field_ratio_l194_194998


namespace mark_age_in_5_years_l194_194507

-- Definitions based on the conditions
def Amy_age := 15
def age_difference := 7

-- Statement specifying the age Mark will be in 5 years
theorem mark_age_in_5_years : (Amy_age + age_difference + 5) = 27 := 
by
  sorry

end mark_age_in_5_years_l194_194507


namespace length_of_each_brick_l194_194589

theorem length_of_each_brick (wall_length wall_height wall_thickness : ℝ) (brick_length brick_width brick_height : ℝ) (num_bricks_used : ℝ) 
  (h1 : wall_length = 8) 
  (h2 : wall_height = 6) 
  (h3 : wall_thickness = 0.02) 
  (h4 : brick_length = 0.11) 
  (h5 : brick_width = 0.05) 
  (h6 : brick_height = 0.06) 
  (h7 : num_bricks_used = 2909.090909090909) : 
  brick_length = 0.11 :=
by
  -- variables and assumptions
  have vol_wall : ℝ := wall_length * wall_height * wall_thickness
  have vol_brick : ℝ := brick_length * brick_width * brick_height
  have calc_bricks : ℝ := vol_wall / vol_brick
  -- skipping proof
  sorry

end length_of_each_brick_l194_194589


namespace ellipse_foci_coordinates_l194_194948

theorem ellipse_foci_coordinates :
  (∀ (x y : ℝ), (x^2 / 16 + y^2 / 25 = 1) → (∃ (c : ℝ), c = 3 ∧ (x = 0 ∧ (y = c ∨ y = -c)))) :=
by
  sorry

end ellipse_foci_coordinates_l194_194948


namespace problem_l194_194288

theorem problem (a b : ℝ) :
  (∀ x : ℝ, 3 * x - 1 ≤ a ∧ 2 * x ≥ 6 - b → -1 ≤ x ∧ x ≤ 2) →
  a + b = 13 := by
  sorry

end problem_l194_194288


namespace smallest_percent_both_coffee_tea_l194_194928

noncomputable def smallest_percent_coffee_tea (P_C P_T P_not_C_or_T : ℝ) : ℝ :=
  let P_C_or_T := 1 - P_not_C_or_T
  let P_C_and_T := P_C + P_T - P_C_or_T
  P_C_and_T

theorem smallest_percent_both_coffee_tea :
  smallest_percent_coffee_tea 0.9 0.85 0.15 = 0.9 :=
by
  sorry

end smallest_percent_both_coffee_tea_l194_194928


namespace ticket_price_l194_194827

variable (x : ℝ)

def tickets_condition1 := 3 * x
def tickets_condition2 := 5 * x
def total_spent := 3 * x + 5 * x

theorem ticket_price : total_spent x = 32 → x = 4 :=
by
  -- Proof steps will be provided here.
  sorry

end ticket_price_l194_194827


namespace john_trip_time_l194_194442

theorem john_trip_time (x : ℝ) (h : x + 2 * x + 2 * x = 10) : x = 2 :=
by
  sorry

end john_trip_time_l194_194442


namespace number_of_primes_l194_194325

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem number_of_primes (p : ℕ)
  (H_prime : is_prime p)
  (H_square : is_perfect_square (1 + p + p^2 + p^3 + p^4)) :
  p = 3 :=
sorry

end number_of_primes_l194_194325


namespace trains_crossing_time_l194_194460

-- Definitions based on given conditions
noncomputable def length_A : ℝ := 2500
noncomputable def time_A : ℝ := 50
noncomputable def length_B : ℝ := 3500
noncomputable def speed_factor : ℝ := 1.2

-- Speed computations
noncomputable def speed_A : ℝ := length_A / time_A
noncomputable def speed_B : ℝ := speed_A * speed_factor

-- Relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := speed_A + speed_B

-- Total distance covered when crossing each other
noncomputable def total_distance : ℝ := length_A + length_B

-- Time taken to cross each other
noncomputable def time_to_cross : ℝ := total_distance / relative_speed

-- Proof statement: Time taken is approximately 54.55 seconds
theorem trains_crossing_time :
  |time_to_cross - 54.55| < 0.01 := by
  sorry

end trains_crossing_time_l194_194460


namespace zero_in_A_l194_194696

-- Define the set A
def A : Set ℝ := { x | x * (x - 2) = 0 }

-- State the theorem
theorem zero_in_A : 0 ∈ A :=
by {
  -- Skipping the actual proof with "sorry"
  sorry
}

end zero_in_A_l194_194696


namespace compare_travel_times_l194_194906

variable (v : ℝ) (t1 t2 : ℝ)

def travel_time_first := t1 = 100 / v
def travel_time_second := t2 = 200 / v

theorem compare_travel_times (h1 : travel_time_first v t1) (h2 : travel_time_second v t2) : 
  t2 = 2 * t1 :=
by
  sorry

end compare_travel_times_l194_194906


namespace differences_impossible_l194_194019

def sum_of_digits (n : ℕ) : ℕ :=
  -- A simple definition for the sum of digits function
  n.digits 10 |>.sum

theorem differences_impossible (a : Fin 100 → ℕ) :
    ¬∃ (perm : Fin 100 → Fin 100), 
      (∀ i, a i - sum_of_digits (a (perm (i : ℕ) % 100)) = i + 1) :=
by
  sorry

end differences_impossible_l194_194019


namespace angle_AC_B₁C₁_is_60_l194_194084

-- Redefine the conditions of the problem using Lean definitions
-- We define a regular triangular prism, equilateral triangle condition,
-- and parallel lines relation.

structure TriangularPrism :=
  (A B C A₁ B₁ C₁ : Type)
  (is_regular : Prop) -- Property stating it is a regular triangular prism
  (base_is_equilateral : Prop) -- Property stating the base is an equilateral triangle
  (B₁C₁_parallel_to_BC : Prop) -- Property stating B₁C₁ is parallel to BC

-- Assume a regular triangular prism with the given properties
variable (prism : TriangularPrism)
axiom isRegularPrism : prism.is_regular
axiom baseEquilateral : prism.base_is_equilateral
axiom parallelLines : prism.B₁C₁_parallel_to_BC

-- Define the angle calculation statement in Lean 4
theorem angle_AC_B₁C₁_is_60 :
  ∃ (angle : ℝ), angle = 60 :=
by
  -- Proof is omitted using sorry
  exact ⟨60, sorry⟩

end angle_AC_B₁C₁_is_60_l194_194084


namespace extended_fishing_rod_length_l194_194120

def original_length : ℝ := 48
def increase_factor : ℝ := 1.33
def extended_length (orig_len : ℝ) (factor : ℝ) : ℝ := orig_len * factor

theorem extended_fishing_rod_length : extended_length original_length increase_factor = 63.84 :=
  by
    -- proof goes here
    sorry

end extended_fishing_rod_length_l194_194120


namespace compute_expression_l194_194353
-- Import the standard math library to avoid import errors.

-- Define the theorem statement based on the given conditions and the correct answer.
theorem compute_expression :
  (75 * 2424 + 25 * 2424) / 2 = 121200 :=
by
  sorry

end compute_expression_l194_194353


namespace dice_probability_l194_194332

theorem dice_probability (D1 D2 D3 : ℕ) (hD1 : 0 ≤ D1) (hD1' : D1 < 10) (hD2 : 0 ≤ D2) (hD2' : D2 < 10) (hD3 : 0 ≤ D3) (hD3' : D3 < 10) :
  ∃ p : ℚ, p = 1 / 10 :=
by
  let outcomes := 10 * 10 * 10
  let favorable := 100
  let expected_probability : ℚ := favorable / outcomes
  use expected_probability
  sorry

end dice_probability_l194_194332


namespace solution_set_inequality_l194_194441

theorem solution_set_inequality (a : ℕ) (h : ∀ x : ℝ, (a-2) * x > (a-2) → x < 1) : a = 0 ∨ a = 1 :=
by
  sorry

end solution_set_inequality_l194_194441


namespace cosine_value_l194_194142

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (3, 4)

noncomputable def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

noncomputable def magnitude (x : ℝ × ℝ) : ℝ :=
  (x.1 ^ 2 + x.2 ^ 2).sqrt

noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

theorem cosine_value :
  cos_angle a b = 2 * (5:ℝ).sqrt / 25 :=
by
  sorry

end cosine_value_l194_194142


namespace Barbier_theorem_for_delta_curves_l194_194965

def delta_curve (h : ℝ) : Type := sorry 
def can_rotate_freely_in_3gon (K : delta_curve h) : Prop := sorry
def length_of_curve (K : delta_curve h) : ℝ := sorry

theorem Barbier_theorem_for_delta_curves
  (K : delta_curve h)
  (h : ℝ)
  (H : can_rotate_freely_in_3gon K)
  : length_of_curve K = (2 * Real.pi * h) / 3 := 
sorry

end Barbier_theorem_for_delta_curves_l194_194965


namespace expand_polynomial_l194_194502

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 := 
by
  sorry

end expand_polynomial_l194_194502


namespace gain_percent_l194_194870

theorem gain_percent (C S : ℝ) (h : 50 * C = 30 * S) : ((S - C) / C) * 100 = 200 / 3 :=
by 
  sorry

end gain_percent_l194_194870


namespace solve_for_x_l194_194620

noncomputable def avg (a b : ℝ) := (a + b) / 2

noncomputable def B (t : List ℝ) : List ℝ :=
  match t with
  | [a, b, c, d, e] => [avg a b, avg b c, avg c d, avg d e]
  | _ => []

noncomputable def B_iter (m : ℕ) (t : List ℝ) : List ℝ :=
  match m with
  | 0 => t
  | k + 1 => B (B_iter k t)

theorem solve_for_x (x : ℝ) (h1 : 0 < x) (h2 : B_iter 4 [1, x, x^2, x^3, x^4] = [1/4]) :
  x = Real.sqrt 2 - 1 :=
sorry

end solve_for_x_l194_194620


namespace product_calc_l194_194527

theorem product_calc : (16 * 0.5 * 4 * 0.125 = 4) :=
by
  sorry

end product_calc_l194_194527


namespace proof_y_pow_x_equal_1_by_9_l194_194492

theorem proof_y_pow_x_equal_1_by_9 
  (x y : ℝ)
  (h : (x - 2)^2 + abs (y + 1/3) = 0) :
  y^x = 1/9 := by
  sorry

end proof_y_pow_x_equal_1_by_9_l194_194492


namespace rachel_makes_money_l194_194059

theorem rachel_makes_money (cost_per_bar total_bars remaining_bars : ℕ) (h_cost : cost_per_bar = 2) (h_total : total_bars = 13) (h_remaining : remaining_bars = 4) :
  cost_per_bar * (total_bars - remaining_bars) = 18 :=
by 
  sorry

end rachel_makes_money_l194_194059


namespace greatest_common_factor_of_two_digit_palindromes_is_11_l194_194006

-- Define a two-digit palindrome
def is_two_digit_palindrome (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n / 10 = n % 10)

-- Define the GCD of the set of all such numbers
def GCF_two_digit_palindromes : ℕ :=
  gcd (11 * 1) (gcd (11 * 2) (gcd (11 * 3) (gcd (11 * 4)
  (gcd (11 * 5) (gcd (11 * 6) (gcd (11 * 7) (gcd (11 * 8) (11 * 9))))))))

-- The statement to prove
theorem greatest_common_factor_of_two_digit_palindromes_is_11 :
  GCF_two_digit_palindromes = 11 :=
by
  sorry

end greatest_common_factor_of_two_digit_palindromes_is_11_l194_194006


namespace total_weight_tommy_ordered_l194_194549

theorem total_weight_tommy_ordered :
  let apples := 3
  let oranges := 1
  let grapes := 3
  let strawberries := 3
  apples + oranges + grapes + strawberries = 10 := by
  sorry

end total_weight_tommy_ordered_l194_194549


namespace five_circles_intersect_l194_194124

-- Assume we have five circles
variables (circle1 circle2 circle3 circle4 circle5 : Set Point)

-- Assume every four of them intersect at a single point
axiom four_intersect (c1 c2 c3 c4 : Set Point) : ∃ p : Point, p ∈ c1 ∧ p ∈ c2 ∧ p ∈ c3 ∧ p ∈ c4

-- The goal is to prove that there exists a point through which all five circles pass.
theorem five_circles_intersect :
  (∃ p : Point, p ∈ circle1 ∧ p ∈ circle2 ∧ p ∈ circle3 ∧ p ∈ circle4 ∧ p ∈ circle5) :=
sorry

end five_circles_intersect_l194_194124


namespace value_of_a_l194_194605

theorem value_of_a (a : ℝ) (k : ℝ) (hA : -5 = k * 3) (hB : a = k * (-6)) : a = 10 :=
by
  sorry

end value_of_a_l194_194605


namespace find_c_plus_d_l194_194510

theorem find_c_plus_d (a b c d : ℝ) (h1 : a + b = 12) (h2 : b + c = 9) (h3 : a + d = 6) : 
  c + d = 3 := 
sorry

end find_c_plus_d_l194_194510


namespace right_triangle_perimeter_l194_194042

def right_triangle_circumscribed_perimeter (r c : ℝ) (a b : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter : 
  ∀ (a b : ℝ),
  (4 : ℝ) * (a + b + (26 : ℝ)) = a * b ∧ a^2 + b^2 = (26 : ℝ)^2 →
  right_triangle_circumscribed_perimeter 4 26 a b = 60 := sorry

end right_triangle_perimeter_l194_194042


namespace solve_system_l194_194846

theorem solve_system (x₁ x₂ x₃ : ℝ) (h₁ : 2 * x₁^2 / (1 + x₁^2) = x₂) (h₂ : 2 * x₂^2 / (1 + x₂^2) = x₃) (h₃ : 2 * x₃^2 / (1 + x₃^2) = x₁) :
  (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1) :=
sorry

end solve_system_l194_194846


namespace problem_statement_l194_194330

-- Define the binary operation "*"
def custom_mul (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the problem with the conditions
theorem problem_statement : custom_mul 5 (-3) = 1 := by
  sorry

end problem_statement_l194_194330


namespace center_distance_correct_l194_194521

noncomputable def ball_diameter : ℝ := 6
noncomputable def ball_radius : ℝ := ball_diameter / 2
noncomputable def R₁ : ℝ := 150
noncomputable def R₂ : ℝ := 50
noncomputable def R₃ : ℝ := 90
noncomputable def R₄ : ℝ := 120
noncomputable def elevation : ℝ := 4

noncomputable def adjusted_R₁ : ℝ := R₁ - ball_radius
noncomputable def adjusted_R₂ : ℝ := R₂ + ball_radius + elevation
noncomputable def adjusted_R₃ : ℝ := R₃ - ball_radius
noncomputable def adjusted_R₄ : ℝ := R₄ - ball_radius

noncomputable def distance_R₁ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₁
noncomputable def distance_R₂ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₂
noncomputable def distance_R₃ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₃
noncomputable def distance_R₄ : ℝ := 1/2 * 2 * Real.pi * adjusted_R₄

noncomputable def total_distance : ℝ := distance_R₁ + distance_R₂ + distance_R₃ + distance_R₄

theorem center_distance_correct : total_distance = 408 * Real.pi := 
  by
  sorry

end center_distance_correct_l194_194521


namespace kittens_more_than_twice_puppies_l194_194719

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens
def num_kittens : ℕ := 78

-- Define the problem statement
theorem kittens_more_than_twice_puppies :
  num_kittens = 2 * num_puppies + 14 :=
by sorry

end kittens_more_than_twice_puppies_l194_194719


namespace blue_balls_taken_out_l194_194780

theorem blue_balls_taken_out (x : ℕ) :
  ∀ (total_balls : ℕ) (initial_blue_balls : ℕ)
    (remaining_probability : ℚ),
    total_balls = 25 ∧ initial_blue_balls = 9 ∧ remaining_probability = 1/5 →
    (9 - x : ℚ) / (25 - x : ℚ) = 1/5 →
    x = 5 :=
by
  intros total_balls initial_blue_balls remaining_probability
  rintro ⟨h_total_balls, h_initial_blue_balls, h_remaining_probability⟩ h_eq
  -- Proof goes here
  sorry

end blue_balls_taken_out_l194_194780


namespace no_consecutive_primes_sum_65_l194_194149

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p q : ℕ) : Prop := 
  is_prime p ∧ is_prime q ∧ (q = p + 2 ∨ q = p - 2)

theorem no_consecutive_primes_sum_65 : 
  ¬ ∃ p q : ℕ, consecutive_primes p q ∧ p + q = 65 :=
by 
  sorry

end no_consecutive_primes_sum_65_l194_194149


namespace no_whole_numbers_satisfy_eqn_l194_194710

theorem no_whole_numbers_satisfy_eqn :
  ¬ ∃ (x y z : ℤ), (x - y) ^ 3 + (y - z) ^ 3 + (z - x) ^ 3 = 2021 :=
by
  sorry

end no_whole_numbers_satisfy_eqn_l194_194710


namespace initial_average_quiz_score_l194_194891

theorem initial_average_quiz_score 
  (n : ℕ) (A : ℝ) (dropped_avg : ℝ) (drop_score : ℝ)
  (students_before : n = 16)
  (students_after : n - 1 = 15)
  (dropped_avg_eq : dropped_avg = 64.0)
  (drop_score_eq : drop_score = 8) 
  (total_sum_before_eq : n * A = 16 * A)
  (total_sum_after_eq : (n - 1) * dropped_avg = 15 * 64):
  A = 60.5 := 
by
  sorry

end initial_average_quiz_score_l194_194891


namespace value_of_a_l194_194247

theorem value_of_a (m : ℝ) (f : ℝ → ℝ) (h : f = fun x => (1/3)^x + m - 1/3) 
  (h_m : ∀ x, f x ≥ 0 ↔ m ≥ -2/3) : m ≥ -2/3 :=
by
  sorry

end value_of_a_l194_194247


namespace find_s_l194_194304

variable {t s : Real}

theorem find_s (h1 : t = 8 * s^2) (h2 : t = 4) : s = Real.sqrt 2 / 2 :=
by
  sorry

end find_s_l194_194304


namespace max_subjects_per_teacher_l194_194054

theorem max_subjects_per_teacher (math_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h_math : math_teachers = 4)
  (h_physics : physics_teachers = 3)
  (h_chemistry : chemistry_teachers = 3)
  (h_min_teachers : min_teachers = 5) :
  (math_teachers + physics_teachers + chemistry_teachers) / min_teachers = 2 :=
by
  sorry

end max_subjects_per_teacher_l194_194054


namespace randy_mango_trees_l194_194944

theorem randy_mango_trees (M C : ℕ) 
  (h1 : C = M / 2 - 5) 
  (h2 : M + C = 85) : 
  M = 60 := 
sorry

end randy_mango_trees_l194_194944


namespace arithmetic_sequence_properties_l194_194995

theorem arithmetic_sequence_properties
  (a_n : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * ((a_n 0 + a_n (n-1)) / 2))
  (h2 : S 6 < S 7)
  (h3 : S 7 > S 8) :
  (a_n 8 - a_n 7 < 0) ∧ (S 9 < S 6) ∧ (∀ m, S m ≤ S 7) :=
by
  sorry

end arithmetic_sequence_properties_l194_194995


namespace distance_rowed_downstream_l194_194839

-- Define the conditions
def speed_in_still_water (b s: ℝ) := b - s = 60 / 4
def speed_of_stream (s: ℝ) := s = 3
def time_downstream (t: ℝ) := t = 4

-- Define the function that computes the downstream speed
def downstream_speed (b s t: ℝ) := (b + s) * t

-- The theorem we want to prove
theorem distance_rowed_downstream (b s t : ℝ) 
    (h1 : speed_in_still_water b s)
    (h2 : speed_of_stream s)
    (h3 : time_downstream t) : 
    downstream_speed b s t = 84 := by
    sorry

end distance_rowed_downstream_l194_194839


namespace symmetry_proof_l194_194028

-- Define the initial point P and its reflection P' about the x-axis
def P : ℝ × ℝ := (-1, 2)
def P' : ℝ × ℝ := (-1, -2)

-- Define the property of symmetry about the x-axis
def symmetric_about_x_axis (P P' : ℝ × ℝ) : Prop :=
  P'.fst = P.fst ∧ P'.snd = -P.snd

-- The theorem to prove that point P' is symmetric to point P about the x-axis
theorem symmetry_proof : symmetric_about_x_axis P P' :=
  sorry

end symmetry_proof_l194_194028


namespace product_of_squares_l194_194126

theorem product_of_squares (a_1 a_2 a_3 b_1 b_2 b_3 : ℕ) (N : ℕ) (h1 : (a_1 * b_1)^2 = N) (h2 : (a_2 * b_2)^2 = N) (h3 : (a_3 * b_3)^2 = N) 
: (a_1^2 * b_1^2) = 36 ∨  (a_2^2 * b_2^2) = 36 ∨ (a_3^2 * b_3^2) = 36:= 
sorry

end product_of_squares_l194_194126


namespace tan_theta_solution_l194_194451

theorem tan_theta_solution (θ : ℝ) (h : 2 * Real.sin θ = 1 + Real.cos θ) :
  Real.tan θ = 0 ∨ Real.tan θ = 4 / 3 :=
sorry

end tan_theta_solution_l194_194451


namespace find_k_l194_194867

noncomputable def vector_a : ℝ × ℝ := (3, 1)
noncomputable def vector_b : ℝ × ℝ := (1, 0)
noncomputable def vector_c (k : ℝ) : ℝ × ℝ := (vector_a.1 + k * vector_b.1, vector_a.2 + k * vector_b.2)

theorem find_k (k : ℝ) (h : vector_a.1 * (vector_a.1 + k * vector_b.1) + vector_a.2 * (vector_a.2 + k * vector_b.2) = 0) : k = -10 / 3 :=
by sorry

end find_k_l194_194867


namespace elmer_saving_percent_l194_194199

theorem elmer_saving_percent (x c : ℝ) (hx : x > 0) (hc : c > 0) :
  let old_car_fuel_efficiency := x
  let new_car_fuel_efficiency := 1.6 * x
  let gasoline_cost := c
  let diesel_cost := 1.25 * c
  let trip_distance := 300
  let old_car_fuel_needed := trip_distance / old_car_fuel_efficiency
  let new_car_fuel_needed := trip_distance / new_car_fuel_efficiency
  let old_car_cost := old_car_fuel_needed * gasoline_cost
  let new_car_cost := new_car_fuel_needed * diesel_cost
  let cost_saving := old_car_cost - new_car_cost
  let percent_saving := (cost_saving / old_car_cost) * 100
  percent_saving = 21.875 :=
by
  sorry

end elmer_saving_percent_l194_194199


namespace constant_term_is_19_l194_194715

theorem constant_term_is_19 (x y C : ℝ) 
  (h1 : 7 * x + y = C) 
  (h2 : x + 3 * y = 1) 
  (h3 : 2 * x + y = 5) : 
  C = 19 :=
sorry

end constant_term_is_19_l194_194715


namespace mother_daughter_age_l194_194743

theorem mother_daughter_age (x : ℕ) :
  let mother_age := 42
  let daughter_age := 8
  (mother_age + x = 3 * (daughter_age + x)) → x = 9 :=
by
  let mother_age := 42
  let daughter_age := 8
  intro h
  sorry

end mother_daughter_age_l194_194743


namespace average_age_nine_students_l194_194045

theorem average_age_nine_students (total_age_15_students : ℕ)
                                (total_age_5_students : ℕ)
                                (age_15th_student : ℕ)
                                (h1 : total_age_15_students = 225)
                                (h2 : total_age_5_students = 65)
                                (h3 : age_15th_student = 16) :
                                (total_age_15_students - total_age_5_students - age_15th_student) / 9 = 16 := by
  sorry

end average_age_nine_students_l194_194045


namespace colin_speed_l194_194415

variable (B T Br C D : ℝ)

-- Given conditions
axiom cond1 : C = 6 * Br
axiom cond2 : Br = (1/3) * T^2
axiom cond3 : T = 2 * B
axiom cond4 : D = (1/4) * C
axiom cond5 : B = 1

-- Prove Colin's speed C is 8 mph
theorem colin_speed :
  C = 8 :=
by
  sorry

end colin_speed_l194_194415


namespace avg_percentage_decrease_l194_194858

theorem avg_percentage_decrease (x : ℝ) 
  (h : 16 * (1 - x)^2 = 9) : x = 0.25 :=
sorry

end avg_percentage_decrease_l194_194858


namespace train_speed_correct_l194_194204

noncomputable def train_speed : ℝ :=
  let distance := 120 -- meters
  let time := 5.999520038396929 -- seconds
  let speed_m_s := distance / time -- meters per second
  speed_m_s * 3.6 -- converting to km/hr

theorem train_speed_correct : train_speed = 72.004800384 := by
  simp [train_speed]
  sorry

end train_speed_correct_l194_194204


namespace sum_of_roots_proof_l194_194992

noncomputable def sum_of_roots (x1 x2 x3 : ℝ) : ℝ :=
  let eq1 := (11 - x1)^3 + (13 - x1)^3 = (24 - 2 * x1)^3
  let eq2 := (11 - x2)^3 + (13 - x2)^3 = (24 - 2 * x2)^3
  let eq3 := (11 - x3)^3 + (13 - x3)^3 = (24 - 2 * x3)^3
  x1 + x2 + x3

theorem sum_of_roots_proof : sum_of_roots 11 12 13 = 36 :=
  sorry

end sum_of_roots_proof_l194_194992


namespace smallest_side_length_1008_l194_194376

def smallest_side_length_original_square :=
  let n := Nat.lcm 7 8
  let n := Nat.lcm n 9
  let lcm := Nat.lcm n 10
  2 * lcm

theorem smallest_side_length_1008 :
  smallest_side_length_original_square = 1008 := by
  sorry

end smallest_side_length_1008_l194_194376


namespace fraction_simplification_l194_194408

theorem fraction_simplification : 
  (320 / 18) * (9 / 144) * (4 / 5) = 1 / 2 :=
by sorry

end fraction_simplification_l194_194408


namespace average_test_score_of_remainder_l194_194287

variable (score1 score2 score3 totalAverage : ℝ)
variable (percentage1 percentage2 percentage3 : ℝ)

def equation (score1 score2 score3 totalAverage : ℝ) (percentage1 percentage2 percentage3: ℝ) : Prop :=
  (percentage1 * score1) + (percentage2 * score2) + (percentage3 * score3) = totalAverage

theorem average_test_score_of_remainder
  (h1 : percentage1 = 0.15)
  (h2 : score1 = 100)
  (h3 : percentage2 = 0.5)
  (h4 : score2 = 78)
  (h5 : percentage3 = 0.35)
  (total : totalAverage = 76.05) :
  (score3 = 63) :=
sorry

end average_test_score_of_remainder_l194_194287


namespace find_age_l194_194035

theorem find_age (x : ℕ) (h : 5 * (x + 5) - 5 * (x - 5) = x) : x = 50 :=
by
  sorry

end find_age_l194_194035


namespace vector_triangle_c_solution_l194_194655

theorem vector_triangle_c_solution :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (-2, 4)
  let c : ℝ × ℝ := (4, -6)
  (4 • a + (3 • b - 2 • a) + c = (0, 0)) →
  c = (4, -6) :=
by
  intro h
  sorry

end vector_triangle_c_solution_l194_194655


namespace sandy_gave_puppies_l194_194712

theorem sandy_gave_puppies 
  (original_puppies : ℕ) 
  (puppies_with_spots : ℕ) 
  (puppies_left : ℕ) 
  (h1 : original_puppies = 8) 
  (h2 : puppies_with_spots = 3) 
  (h3 : puppies_left = 4) : 
  original_puppies - puppies_left = 4 := 
by {
  -- This is a placeholder for the proof.
  sorry
}

end sandy_gave_puppies_l194_194712


namespace line_through_ellipse_and_midpoint_l194_194572

theorem line_through_ellipse_and_midpoint :
  ∃ l : ℝ → ℝ → Prop,
    (∀ (x y : ℝ), l x y ↔ (x + y) = 0) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      (x₁ + x₂ = 2 ∧ y₁ + y₂ = 1) ∧
      (x₁^2 / 2 + y₁^2 = 1 ∧ x₂^2 / 2 + y₂^2 = 1) ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧
      ∀ (mx my : ℝ), (mx, my) = (1, 0.5) → (mx = (x₁ + x₂) / 2 ∧ my = (y₁ + y₂) / 2))
  := sorry

end line_through_ellipse_and_midpoint_l194_194572


namespace dishonest_shopkeeper_weight_l194_194574

noncomputable def weight_used (gain_percent : ℝ) (correct_weight : ℝ) : ℝ :=
  correct_weight / (1 + gain_percent / 100)

theorem dishonest_shopkeeper_weight :
  weight_used 5.263157894736836 1000 = 950 := 
by
  sorry

end dishonest_shopkeeper_weight_l194_194574


namespace value_of_expression_l194_194984

variable (a b : ℝ)

def system_of_equations : Prop :=
  (2 * a - b = 12) ∧ (a + 2 * b = 8)

theorem value_of_expression (h : system_of_equations a b) : 3 * a + b = 20 :=
  sorry

end value_of_expression_l194_194984


namespace find_m_l194_194804

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3 * m - 1}
def B (m : ℝ) : Set ℝ := {2 * m, -3}
def C : Set ℝ := {-3}

theorem find_m (m : ℝ) : A m ∩ B m = C → m = 1 :=
by 
  intros h
  sorry

end find_m_l194_194804


namespace find_nickels_l194_194800

noncomputable def num_quarters1 := 25
noncomputable def num_dimes := 15
noncomputable def num_quarters2 := 15
noncomputable def value_quarter := 25
noncomputable def value_dime := 10
noncomputable def value_nickel := 5

theorem find_nickels (n : ℕ) :
  value_quarter * num_quarters1 + value_dime * num_dimes = value_quarter * num_quarters2 + value_nickel * n → 
  n = 80 :=
by
  sorry

end find_nickels_l194_194800


namespace consecutive_integer_sets_sum_27_l194_194456

theorem consecutive_integer_sets_sum_27 :
  ∃! s : Set (List ℕ), ∀ l ∈ s, 
  (∃ n a, n ≥ 3 ∧ l = List.range n ++ [a] ∧ (List.sum l) = 27)
:=
sorry

end consecutive_integer_sets_sum_27_l194_194456


namespace office_person_count_l194_194724

theorem office_person_count
    (N : ℕ)
    (avg_age_all : ℕ)
    (num_5 : ℕ)
    (avg_age_5 : ℕ)
    (num_9 : ℕ)
    (avg_age_9 : ℕ)
    (age_15th : ℕ)
    (h1 : avg_age_all = 15)
    (h2 : num_5 = 5)
    (h3 : avg_age_5 = 14)
    (h4 : num_9 = 9)
    (h5 : avg_age_9 = 16)
    (h6 : age_15th = 86)
    (h7 : 15 * N = (num_5 * avg_age_5) + (num_9 * avg_age_9) + age_15th) :
    N = 20 :=
by
    -- Proof will be provided here
    sorry

end office_person_count_l194_194724


namespace coefficient_sum_eq_512_l194_194123

theorem coefficient_sum_eq_512 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ) :
  (1 - x) ^ 9 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + 
                a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| + |a_9| = 512 :=
sorry

end coefficient_sum_eq_512_l194_194123


namespace min_value_fraction_l194_194739

theorem min_value_fraction (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2 * m + n = 1) : (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_fraction_l194_194739


namespace find_factor_l194_194178

theorem find_factor (n f : ℤ) (h₁ : n = 124) (h₂ : n * f - 138 = 110) : f = 2 := by
  sorry

end find_factor_l194_194178


namespace radar_coverage_proof_l194_194004

theorem radar_coverage_proof (n : ℕ) (r : ℝ) (w : ℝ) (d : ℝ) (A : ℝ) : 
  n = 9 ∧ r = 37 ∧ w = 24 ∧ d = 35 / Real.sin (Real.pi / 9) ∧
  A = 1680 * Real.pi / Real.tan (Real.pi / 9) → 
  ∃ OB S_ring, OB = d ∧ S_ring = A 
:= by sorry

end radar_coverage_proof_l194_194004


namespace cartesian_to_polar_coords_l194_194753

theorem cartesian_to_polar_coords :
  ∃ ρ θ : ℝ, 
  (ρ = 2) ∧ (θ = 2 * Real.pi / 3) ∧ 
  (-1, Real.sqrt 3) = (ρ * Real.cos θ, ρ * Real.sin θ) :=
sorry

end cartesian_to_polar_coords_l194_194753


namespace base12_remainder_l194_194156

def base12_to_base10 (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

theorem base12_remainder (a b c d : ℕ) 
  (h1531 : base12_to_base10 a b c d = 1 * 12^3 + 5 * 12^2 + 3 * 12^1 + 1 * 12^0):
  (base12_to_base10 a b c d) % 8 = 5 :=
by
  unfold base12_to_base10 at h1531
  sorry

end base12_remainder_l194_194156


namespace orthocentric_tetrahedron_equivalence_l194_194769

def isOrthocentricTetrahedron 
  (sums_of_squares_of_opposite_edges_equal : Prop) 
  (products_of_cosines_of_opposite_dihedral_angles_equal : Prop)
  (angles_between_opposite_edges_equal : Prop) : Prop :=
  sums_of_squares_of_opposite_edges_equal ∨
  products_of_cosines_of_opposite_dihedral_angles_equal ∨
  angles_between_opposite_edges_equal

theorem orthocentric_tetrahedron_equivalence
  (sums_of_squares_of_opposite_edges_equal 
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal : Prop) :
  isOrthocentricTetrahedron
    sums_of_squares_of_opposite_edges_equal
    products_of_cosines_of_opposite_dihedral_angles_equal
    angles_between_opposite_edges_equal :=
sorry

end orthocentric_tetrahedron_equivalence_l194_194769


namespace cost_of_45_daffodils_equals_75_l194_194226

-- Conditions
def cost_of_15_daffodils : ℝ := 25
def number_of_daffodils_in_bouquet_15 : ℕ := 15
def number_of_daffodils_in_bouquet_45 : ℕ := 45
def directly_proportional (n m : ℕ) (c_n c_m : ℝ) : Prop := c_n / n = c_m / m

-- Statement to prove
theorem cost_of_45_daffodils_equals_75 :
  ∀ (c : ℝ), directly_proportional number_of_daffodils_in_bouquet_45 number_of_daffodils_in_bouquet_15 c cost_of_15_daffodils → c = 75 :=
by
  intro c hypothesis
  -- Proof would go here.
  sorry

end cost_of_45_daffodils_equals_75_l194_194226


namespace a_2n_perfect_square_l194_194344

-- Define the sequence a_n following the described recurrence relation.
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n-1) + a (n-3) + a (n-4)

-- Define the main theorem to prove
theorem a_2n_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k * k := by
  sorry

end a_2n_perfect_square_l194_194344


namespace problem1_problem2_l194_194000

def count_good_subsets (n : ℕ) : ℕ := 
if n % 2 = 1 then 2^(n - 1) 
else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2)

def sum_f_good_subsets (n : ℕ) : ℕ :=
if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2)

theorem problem1 (n : ℕ)  :
  (count_good_subsets n = (if n % 2 = 1 then 2^(n - 1) else 2^(n - 1) - (1 / 2) * Nat.choose n (n / 2))) :=
sorry

theorem problem2 (n : ℕ) :
  (sum_f_good_subsets n = (if n % 2 = 1 then n * (n + 1) * 2^(n - 3) + (n + 1) / 4 * Nat.choose n ((n - 1) / 2)
  else n * (n + 1) * 2^(n - 3) - (n / 2) * ((n / 2) + 1) * Nat.choose (n / 2) (n / 2))) := 
sorry

end problem1_problem2_l194_194000


namespace parallelogram_angle_B_eq_130_l194_194240

theorem parallelogram_angle_B_eq_130 (A C B D : ℝ) (parallelogram_ABCD : true) 
(angles_sum_A_C : A + C = 100) (A_eq_C : A = C): B = 130 := by
  sorry

end parallelogram_angle_B_eq_130_l194_194240


namespace number_of_sandwiches_l194_194374

-- Definitions based on conditions
def breads : Nat := 5
def meats : Nat := 7
def cheeses : Nat := 6
def total_sandwiches : Nat := breads * meats * cheeses
def turkey_mozzarella_exclusions : Nat := breads
def rye_beef_exclusions : Nat := cheeses

-- The proof problem statement
theorem number_of_sandwiches (total_sandwiches := 210) 
  (turkey_mozzarella_exclusions := 5) 
  (rye_beef_exclusions := 6) : 
  total_sandwiches - turkey_mozzarella_exclusions - rye_beef_exclusions = 199 := 
by sorry

end number_of_sandwiches_l194_194374


namespace distinct_real_roots_m_range_root_zero_other_root_l194_194792

open Real

-- Definitions of the quadratic equation and the conditions
def quadratic_eq (m x : ℝ) := x^2 + 2 * (m - 1) * x + m^2 - 1

-- Problem (1)
theorem distinct_real_roots_m_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0) → m < 1 :=
by
  sorry

-- Problem (2)
theorem root_zero_other_root (m x : ℝ) :
  (quadratic_eq m 0 = 0 ∧ quadratic_eq m x = 0) → (m = 1 ∧ x = 0) ∨ (m = -1 ∧ x = 4) :=
by
  sorry

end distinct_real_roots_m_range_root_zero_other_root_l194_194792


namespace lucas_fence_painting_l194_194540

-- Define the conditions
def total_time := 60
def time_painting := 12
def rate_per_minute := 1 / total_time

-- State the theorem
theorem lucas_fence_painting :
  let work_done := rate_per_minute * time_painting
  work_done = 1 / 5 :=
by
  -- Proof omitted
  sorry

end lucas_fence_painting_l194_194540


namespace natasha_dimes_l194_194631

theorem natasha_dimes (n : ℕ) (h1 : 10 < n) (h2 : n < 100) (h3 : n % 3 = 1) (h4 : n % 4 = 1) (h5 : n % 5 = 1) : n = 61 :=
sorry

end natasha_dimes_l194_194631


namespace exist_positive_integers_x_y_z_l194_194308

theorem exist_positive_integers_x_y_z (n : ℕ) (hn : n > 0) : 
  ∃ (x y z : ℕ), 
    x = 2^(n^2) * 3^(n+1) ∧
    y = 2^(n^2 - n) * 3^n ∧
    z = 2^(n^2 - 2*n + 2) * 3^(n-1) ∧
    x^(n-1) + y^n = z^(n+1) :=
by {
  -- placeholder for the proof
  sorry
}

end exist_positive_integers_x_y_z_l194_194308


namespace coin_count_l194_194430

-- Define the conditions and the proof goal
theorem coin_count (total_value : ℕ) (coin_value_20 : ℕ) (coin_value_25 : ℕ) 
    (num_20_paise_coins : ℕ) (total_value_paise : total_value = 7100)
    (value_20_paise : coin_value_20 = 20) (value_25_paise : coin_value_25 = 25)
    (num_20_paise : num_20_paise_coins = 300) : 
    (300 + 44 = 344) :=
by
  -- The proof would go here, currently omitted with sorry
  sorry

end coin_count_l194_194430


namespace slope_of_line_l194_194621

-- Definition of the line equation
def lineEquation (x y : ℝ) : Prop := 4 * x - 7 * y = 14

-- The statement that we need to prove
theorem slope_of_line : ∀ x y, lineEquation x y → ∃ m, m = 4 / 7 :=
by {
  sorry
}

end slope_of_line_l194_194621


namespace solve_for_lambda_l194_194660

def vector_dot_product : (ℤ × ℤ) → (ℤ × ℤ) → ℤ
| (x1, y1), (x2, y2) => x1 * x2 + y1 * y2

theorem solve_for_lambda
  (a : ℤ × ℤ) (b : ℤ × ℤ) (lambda : ℤ)
  (h1 : a = (3, -2))
  (h2 : b = (1, 2))
  (h3 : vector_dot_product (a.1 + lambda * b.1, a.2 + lambda * b.2) a = 0) :
  lambda = 13 :=
sorry

end solve_for_lambda_l194_194660


namespace valid_m_values_l194_194840

theorem valid_m_values :
  ∃ (m : ℕ), (m ∣ 720) ∧ (m ≠ 1) ∧ (m ≠ 720) ∧ ((720 / m) > 1) ∧ ((30 - 2) = 28) := 
sorry

end valid_m_values_l194_194840


namespace inequality_proof_l194_194197

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (2 * a^2) / (1 + a + a * b)^2 + (2 * b^2) / (1 + b + b * c)^2 + (2 * c^2) / (1 + c + c * a)^2 +
  9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a)) ≥ 1 :=
by {
  sorry -- The proof goes here
}

end inequality_proof_l194_194197


namespace distance_to_bus_stand_l194_194398

variable (D : ℝ)

theorem distance_to_bus_stand :
  (D / 4 - D / 5 = 1 / 4) → D = 5 :=
sorry

end distance_to_bus_stand_l194_194398


namespace triangle_side_ratio_sqrt2_l194_194237

variables (A B C A1 B1 C1 X Y : Point)
variable (triangle : IsAcuteAngledTriangle A B C)
variable (altitudes : AreAltitudes A B C A1 B1 C1)
variable (midpoints : X = Midpoint A C1 ∧ Y = Midpoint A1 C)
variable (equality : Distance X Y = Distance B B1)

theorem triangle_side_ratio_sqrt2 :
  ∃ (AC AB : ℝ), (AC / AB = Real.sqrt 2) := sorry

end triangle_side_ratio_sqrt2_l194_194237


namespace nth_monomial_is_correct_l194_194531

-- conditions
def coefficient (n : ℕ) : ℕ := 2 * n - 1
def exponent (n : ℕ) : ℕ := n
def monomial (n : ℕ) : ℕ × ℕ := (coefficient n, exponent n)

-- theorem to prove the nth monomial
theorem nth_monomial_is_correct (n : ℕ) : monomial n = (2 * n - 1, n) := 
by 
    sorry

end nth_monomial_is_correct_l194_194531


namespace smallest_portion_is_five_thirds_l194_194932

theorem smallest_portion_is_five_thirds
    (a1 a2 a3 a4 a5 : ℚ)
    (h1 : a2 = a1 + 1)
    (h2 : a3 = a1 + 2)
    (h3 : a4 = a1 + 3)
    (h4 : a5 = a1 + 4)
    (h_sum : a1 + a2 + a3 + a4 + a5 = 100)
    (h_cond : (1 / 7) * (a3 + a4 + a5) = a1 + a2) :
    a1 = 5 / 3 :=
by
  sorry

end smallest_portion_is_five_thirds_l194_194932


namespace find_x_l194_194987

theorem find_x (x : ℝ) (hx_pos : 0 < x) (h: (x / 100) * x = 4) : x = 20 := by
  sorry

end find_x_l194_194987


namespace sixty_percent_is_240_l194_194871

variable (x : ℝ)

-- Conditions
def forty_percent_eq_160 : Prop := 0.40 * x = 160

-- Proof problem
theorem sixty_percent_is_240 (h : forty_percent_eq_160 x) : 0.60 * x = 240 :=
sorry

end sixty_percent_is_240_l194_194871


namespace find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l194_194560

noncomputable def board : Type := (Fin 5) × (Fin 5)

def is_counterfeit (c1 : board) (c2 : board) : Prop :=
  (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 + 1 = c2.2)) ∨
  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 + 1 = c2.1))

theorem find_13_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 13 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem find_15_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ∃ C : Finset board, C.card = 15 ∧ ∀ c ∈ C, coins c = coins (0,0) :=
sorry

theorem cannot_find_17_real_coins (coins : board → ℝ) (c1 c2 : board) :
  (coins c1 < coins (0,0) ∧ coins c2 < coins (0,0)) ∧ is_counterfeit c1 c2 →
  ¬ (∃ C : Finset board, C.card = 17 ∧ ∀ c ∈ C, coins c = coins (0,0)) :=
sorry

end find_13_real_coins_find_15_real_coins_cannot_find_17_real_coins_l194_194560


namespace meal_cost_l194_194113

theorem meal_cost:
  ∀ (s c p k : ℝ), 
  (2 * s + 5 * c + 2 * p + 3 * k = 6.30) →
  (3 * s + 8 * c + 2 * p + 4 * k = 8.40) →
  (s + c + p + k = 3.15) :=
by
  intros s c p k h1 h2
  sorry

end meal_cost_l194_194113


namespace binary_to_decimal_l194_194341

theorem binary_to_decimal :
  1 * 2^8 + 0 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 379 :=
by
  sorry

end binary_to_decimal_l194_194341


namespace prime_divisor_of_form_l194_194147

theorem prime_divisor_of_form (a p : ℕ) (hp1 : a > 0) (hp2 : Prime p) (hp3 : p ∣ (a^3 - 3 * a + 1)) (hp4 : p ≠ 3) :
  ∃ k : ℤ, p = 9 * k + 1 ∨ p = 9 * k - 1 :=
by
  sorry

end prime_divisor_of_form_l194_194147


namespace max_value_of_f_l194_194370

noncomputable def f (x : ℝ) : ℝ := 8 * Real.sin x + 15 * Real.cos x

theorem max_value_of_f : ∃ x : ℝ, f x = 17 :=
sorry

end max_value_of_f_l194_194370


namespace stuart_initial_marbles_is_56_l194_194494

-- Define the initial conditions
def betty_initial_marbles : ℕ := 60
def percentage_given_to_stuart : ℚ := 40 / 100
def stuart_marbles_after_receiving : ℕ := 80

-- Define the calculation of how many marbles Betty gave to Stuart
def marbles_given_to_stuart := (percentage_given_to_stuart * betty_initial_marbles)

-- Define the target: Stuart's initial number of marbles
def stuart_initial_marbles := stuart_marbles_after_receiving - marbles_given_to_stuart

-- Main theorem stating the problem
theorem stuart_initial_marbles_is_56 : stuart_initial_marbles = 56 :=
by 
  sorry

end stuart_initial_marbles_is_56_l194_194494


namespace charging_time_is_correct_l194_194217

-- Lean definitions for the given conditions
def smartphone_charge_time : ℕ := 26
def tablet_charge_time : ℕ := 53
def phone_half_charge_time : ℕ := smartphone_charge_time / 2

-- Definition for the total charging time based on conditions
def total_charging_time : ℕ :=
  tablet_charge_time + phone_half_charge_time

-- Proof problem statement
theorem charging_time_is_correct : total_charging_time = 66 := by
  sorry

end charging_time_is_correct_l194_194217


namespace probability_of_sum_5_when_two_dice_rolled_l194_194789

theorem probability_of_sum_5_when_two_dice_rolled :
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_possible_outcomes : ℝ) = (1 / 9 : ℝ) :=
by
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  have h : (favorable_outcomes : ℝ) / (total_possible_outcomes : ℝ) = (1 / 9 : ℝ) := sorry
  exact h

end probability_of_sum_5_when_two_dice_rolled_l194_194789


namespace Katie_homework_problems_l194_194088

theorem Katie_homework_problems :
  let finished_problems := 5
  let remaining_problems := 4
  let total_problems := finished_problems + remaining_problems
  total_problems = 9 :=
by
  sorry

end Katie_homework_problems_l194_194088


namespace tangent_line_circle_l194_194307

theorem tangent_line_circle (a : ℝ) : (∀ x y : ℝ, a * x + y + 1 = 0) → (∀ x y : ℝ, x^2 + y^2 - 4 * x = 0) → a = 3 / 4 :=
by
  sorry

end tangent_line_circle_l194_194307


namespace xyz_inequality_l194_194198

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end xyz_inequality_l194_194198


namespace each_monkey_gets_bananas_l194_194187

-- Define the conditions
def total_monkeys : ℕ := 12
def total_piles : ℕ := 10
def first_piles : ℕ := 6
def first_pile_hands : ℕ := 9
def first_hand_bananas : ℕ := 14
def remaining_piles : ℕ := total_piles - first_piles
def remaining_pile_hands : ℕ := 12
def remaining_hand_bananas : ℕ := 9

-- Define the number of bananas in each type of pile
def bananas_in_first_piles : ℕ := first_piles * first_pile_hands * first_hand_bananas
def bananas_in_remaining_piles : ℕ := remaining_piles * remaining_pile_hands * remaining_hand_bananas
def total_bananas : ℕ := bananas_in_first_piles + bananas_in_remaining_piles

-- Define the main theorem to be proved
theorem each_monkey_gets_bananas : total_bananas / total_monkeys = 99 := by
  sorry

end each_monkey_gets_bananas_l194_194187


namespace two_digit_primes_with_digit_sum_10_count_l194_194364

def digits_sum_to_ten (n : ℕ) : Prop :=
  let d1 := n / 10
  let d2 := n % 10
  d1 + d2 = 10

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_with_digit_sum_10_count : 
  ∃ count : ℕ, count = 4 ∧ ∀ n, (two_digit_number n ∧ digits_sum_to_ten n ∧ Prime n) → count = 4 := 
by
  sorry

end two_digit_primes_with_digit_sum_10_count_l194_194364


namespace Johnson_Martinez_tied_at_end_of_september_l194_194402

open Nat

-- Define the monthly home runs for Johnson and Martinez
def Johnson_runs : List Nat := [3, 8, 15, 12, 5, 7, 14]
def Martinez_runs : List Nat := [0, 3, 9, 20, 7, 12, 13]

-- Define the cumulated home runs for Johnson and Martinez over the months
def total_runs (runs : List Nat) : List Nat :=
  runs.scanl (· + ·) 0

-- State the theorem to prove that they are tied in total runs at the end of September
theorem Johnson_Martinez_tied_at_end_of_september :
  (total_runs Johnson_runs).getLast (by decide) =
  (total_runs Martinez_runs).getLast (by decide) := by
  sorry

end Johnson_Martinez_tied_at_end_of_september_l194_194402


namespace least_small_barrels_l194_194103

theorem least_small_barrels (total_oil : ℕ) (large_barrel : ℕ) (small_barrel : ℕ) (L S : ℕ)
  (h1 : total_oil = 745) (h2 : large_barrel = 11) (h3 : small_barrel = 7)
  (h4 : 11 * L + 7 * S = 745) (h5 : total_oil - 11 * L = 7 * S) : S = 1 :=
by
  sorry

end least_small_barrels_l194_194103


namespace num_distinct_terms_expansion_a_b_c_10_l194_194861

-- Define the expansion of (a+b+c)^10
def num_distinct_terms_expansion (n : ℕ) : ℕ :=
  Nat.choose (n + 3 - 1) (3 - 1)

-- Theorem statement
theorem num_distinct_terms_expansion_a_b_c_10 : num_distinct_terms_expansion 10 = 66 :=
by
  sorry

end num_distinct_terms_expansion_a_b_c_10_l194_194861


namespace fixed_real_root_l194_194911

theorem fixed_real_root (k x : ℝ) (h : x^2 + (k + 3) * x + (k + 2) = 0) : x = -1 :=
sorry

end fixed_real_root_l194_194911


namespace geometric_sequence_value_of_m_l194_194519

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_value_of_m (r : ℝ) (hr : r ≠ 1) 
    (h1 : is_geometric_sequence a r)
    (h2 : a 5 * a 6 + a 4 * a 7 = 18) 
    (h3 : a 1 * a m = 9) :
  m = 10 :=
by
  sorry

end geometric_sequence_value_of_m_l194_194519


namespace point_distance_units_l194_194272

theorem point_distance_units (d : ℝ) (h : |d| = 4) : d = 4 ∨ d = -4 := 
sorry

end point_distance_units_l194_194272


namespace parallelogram_not_symmetrical_l194_194310

-- Define the shapes
inductive Shape
| circle
| rectangle
| isosceles_trapezoid
| parallelogram

-- Define what it means for a shape to be symmetrical
def is_symmetrical (s: Shape) : Prop :=
  match s with
  | Shape.circle => True
  | Shape.rectangle => True
  | Shape.isosceles_trapezoid => True
  | Shape.parallelogram => False -- The condition we're interested in proving

-- The main theorem stating the problem
theorem parallelogram_not_symmetrical : is_symmetrical Shape.parallelogram = False :=
by
  sorry

end parallelogram_not_symmetrical_l194_194310


namespace sqrt_64_eq_8_l194_194894

theorem sqrt_64_eq_8 : Real.sqrt 64 = 8 := 
by
  sorry

end sqrt_64_eq_8_l194_194894


namespace xy_relationship_l194_194809

theorem xy_relationship : 
  (∀ x y, (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 9) ∨ (x = 4 ∧ y = 16) ∨ (x = 5 ∧ y = 25) 
  → y = x * x) :=
by {
  sorry
}

end xy_relationship_l194_194809


namespace perfect_cube_divisor_count_l194_194490

noncomputable def num_perfect_cube_divisors : Nat :=
  let a_choices := Nat.succ (38 / 3)
  let b_choices := Nat.succ (17 / 3)
  let c_choices := Nat.succ (7 / 3)
  let d_choices := Nat.succ (4 / 3)
  a_choices * b_choices * c_choices * d_choices

theorem perfect_cube_divisor_count :
  num_perfect_cube_divisors = 468 :=
by
  sorry

end perfect_cube_divisor_count_l194_194490


namespace unique_solution_for_divisibility_l194_194815

theorem unique_solution_for_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) ∣ (a^3 + 1) ∧ (a^2 + b^2) ∣ (b^3 + 1) → (a = 1 ∧ b = 1) :=
by
  intro h
  sorry

end unique_solution_for_divisibility_l194_194815


namespace cost_of_coffee_B_per_kg_l194_194323

-- Define the cost of coffee A per kilogram
def costA : ℝ := 10

-- Define the amount of coffee A used in the mixture
def amountA : ℝ := 240

-- Define the amount of coffee B used in the mixture
def amountB : ℝ := 240

-- Define the total amount of the mixture
def totalAmount : ℝ := 480

-- Define the selling price of the mixture per kilogram
def sellingPrice : ℝ := 11

-- Define the cost of coffee B per kilogram as a variable B
variable (B : ℝ)

-- Define the total cost of the mixture
def totalCost : ℝ := totalAmount * sellingPrice

-- Define the cost of coffee A used
def costOfA : ℝ := amountA * costA

-- Define the cost of coffee B used as total cost minus the cost of A
def costOfB : ℝ := totalCost - costOfA

-- Calculate the cost of coffee B per kilogram
theorem cost_of_coffee_B_per_kg : B = 12 :=
by
  have h1 : costOfA = 2400 := by sorry
  have h2 : totalCost = 5280 := by sorry
  have h3 : costOfB = 2880 := by sorry
  have h4 : B = costOfB / amountB := by sorry
  have h5 : B = 2880 / 240 := by sorry
  have h6 : B = 12 := by sorry
  exact h6

end cost_of_coffee_B_per_kg_l194_194323


namespace vectors_perpendicular_l194_194619

open Real

def vector := ℝ × ℝ

def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def perpendicular (v w : vector) : Prop :=
  dot_product v w = 0

def vector_sub (v w : vector) : vector :=
  (v.1 - w.1, v.2 - w.2)

theorem vectors_perpendicular :
  let a : vector := (2, 0)
  let b : vector := (1, 1)
  perpendicular (vector_sub a b) b :=
by
  sorry

end vectors_perpendicular_l194_194619


namespace bonnets_per_orphanage_l194_194862

/--
Mrs. Young makes bonnets for kids in the orphanage.
On Monday, she made 10 bonnets.
On Tuesday and Wednesday combined she made twice more than on Monday.
On Thursday she made 5 more than on Monday.
On Friday she made 5 less than on Thursday.
She divided up the bonnets evenly and sent them to 5 orphanages.
Prove that the number of bonnets Mrs. Young sent to each orphanage is 11.
-/
theorem bonnets_per_orphanage :
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let total_bonnets := monday + tuesday_wednesday + thursday + friday
  let orphanages := 5
  total_bonnets / orphanages = 11 :=
by
  sorry

end bonnets_per_orphanage_l194_194862


namespace symmetric_circle_l194_194368

theorem symmetric_circle :
  ∀ (C D : Type) (hD : ∀ x y : ℝ, (x + 2)^2 + (y - 6)^2 = 1) (hline : ∀ x y : ℝ, x - y + 5 = 0), 
  (∀ x y : ℝ, (x - 1)^2 + (y - 3)^2 = 1) := 
by sorry

end symmetric_circle_l194_194368


namespace interval_of_monotonic_decrease_range_of_k_l194_194432
open Real

noncomputable def f (x : ℝ) : ℝ := 
  let m := (sqrt 3 * sin (x / 4), 1)
  let n := (cos (x / 4), cos (x / 2))
  m.1 * n.1 + m.2 * n.2 -- vector dot product

-- Prove the interval of monotonic decrease for f(x)
theorem interval_of_monotonic_decrease (k : ℤ) : 
  4 * k * π + 2 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 8 * π / 3 → f x = sin (x / 2 + π / 6) + 1 / 2 :=
sorry

-- Prove the range of k such that the zero condition is satisfied for g(x) - k
theorem range_of_k (k : ℝ) :
  0 ≤ k ∧ k ≤ 3 / 2 → ∃ x ∈ [0, 7 * π / 3], (sin (x / 2 - π / 6) + 1 / 2) - k = 0 :=
sorry

end interval_of_monotonic_decrease_range_of_k_l194_194432


namespace mrs_jackson_decorations_l194_194699

theorem mrs_jackson_decorations (boxes decorations_in_each_box decorations_used : Nat) 
  (h1 : boxes = 4) 
  (h2 : decorations_in_each_box = 15) 
  (h3 : decorations_used = 35) :
  boxes * decorations_in_each_box - decorations_used = 25 := 
  by
  sorry

end mrs_jackson_decorations_l194_194699


namespace solve_quadratics_l194_194556

theorem solve_quadratics (x : ℝ) :
  (x^2 - 7 * x - 18 = 0 → x = 9 ∨ x = -2) ∧
  (4 * x^2 + 1 = 4 * x → x = 1/2) :=
by
  sorry

end solve_quadratics_l194_194556


namespace edward_skee_ball_tickets_l194_194346

theorem edward_skee_ball_tickets (w_tickets : Nat) (candy_cost : Nat) (num_candies : Nat) (total_tickets : Nat) (skee_ball_tickets : Nat) :
  w_tickets = 3 ∧ candy_cost = 4 ∧ num_candies = 2 ∧ total_tickets = num_candies * candy_cost ∧ total_tickets - w_tickets = skee_ball_tickets → 
  skee_ball_tickets = 5 :=
by
  sorry

end edward_skee_ball_tickets_l194_194346


namespace solve_for_x_l194_194802

variable (a b c x y z : ℝ)
variable (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem solve_for_x (h1 : (x * y) / (x + y) = a)
                   (h2 : (x * z) / (x + z) = b)
                   (h3 : (y * z) / (y + z) = c) :
                   x = (2 * a * b * c) / (a * c + b * c - a * b) :=
by 
  sorry

end solve_for_x_l194_194802


namespace cube_faces_one_third_blue_l194_194687

theorem cube_faces_one_third_blue (n : ℕ) (h1 : ∃ n, n > 0 ∧ (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 := by
  sorry

end cube_faces_one_third_blue_l194_194687


namespace xy_inequality_l194_194169

theorem xy_inequality (x y : ℝ) (h: x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end xy_inequality_l194_194169


namespace express_h_l194_194055

variable (a b S h : ℝ)
variable (h_formula : S = 1/2 * (a + b) * h)
variable (h_nonzero : a + b ≠ 0)

theorem express_h : h = 2 * S / (a + b) := 
by 
  sorry

end express_h_l194_194055


namespace equal_student_distribution_l194_194454

theorem equal_student_distribution
  (students_bus1_initial : ℕ)
  (students_bus2_initial : ℕ)
  (students_to_move : ℕ)
  (students_bus1_final : ℕ)
  (students_bus2_final : ℕ)
  (total_students : ℕ) :
  students_bus1_initial = 57 →
  students_bus2_initial = 31 →
  total_students = students_bus1_initial + students_bus2_initial →
  students_to_move = 13 →
  students_bus1_final = students_bus1_initial - students_to_move →
  students_bus2_final = students_bus2_initial + students_to_move →
  students_bus1_final = 44 ∧ students_bus2_final = 44 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end equal_student_distribution_l194_194454


namespace seed_selection_valid_l194_194207

def seeds : List Nat := [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07]

def extractValidSeeds (lst : List Nat) (startIndex : Nat) (maxValue : Nat) (count : Nat) : List Nat :=
  lst.drop startIndex
  |>.filter (fun n => n < maxValue)
  |>.take count

theorem seed_selection_valid :
  extractValidSeeds seeds 10 850 4 = [169, 555, 671, 105] :=
by
  sorry

end seed_selection_valid_l194_194207


namespace equation_no_solution_for_k_7_l194_194518

theorem equation_no_solution_for_k_7 :
  ∀ x : ℝ, (x ≠ 3 ∧ x ≠ 5) → ¬ (x ^ 2 - 1) / (x - 3) = (x ^ 2 - 7) / (x - 5) :=
by
  intro x h
  have h1 : x ≠ 3 := h.1
  have h2 : x ≠ 5 := h.2
  sorry

end equation_no_solution_for_k_7_l194_194518


namespace no_all_blue_possible_l194_194720

-- Define initial counts of chameleons
def initial_red : ℕ := 25
def initial_green : ℕ := 12
def initial_blue : ℕ := 8

-- Define the invariant condition
def invariant (r g : ℕ) : Prop := (r - g) % 3 = 1

-- Define the main theorem statement
theorem no_all_blue_possible : ¬∃ r g, r = 0 ∧ g = 0 ∧ invariant r g :=
by {
  sorry
}

end no_all_blue_possible_l194_194720


namespace union_sets_l194_194577

def S : Set ℕ := {0, 1}
def T : Set ℕ := {0, 3}

theorem union_sets : S ∪ T = {0, 1, 3} :=
by
  sorry

end union_sets_l194_194577


namespace range_of_a_l194_194897

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, (a * x^2 - 3 * x - 4 = 0) ∧ (a * y^2 - 3 * y - 4 = 0) → x = y) ↔ (a ≤ -9 / 16 ∨ a = 0) := 
by
  sorry

end range_of_a_l194_194897


namespace elmer_saves_21_875_percent_l194_194707

noncomputable def old_car_efficiency (x : ℝ) := x
noncomputable def new_car_efficiency (x : ℝ) := 1.6 * x

noncomputable def gasoline_cost (c : ℝ) := c
noncomputable def diesel_cost (c : ℝ) := 1.25 * c

noncomputable def trip_distance := 1000

noncomputable def old_car_fuel_consumption (x : ℝ) := trip_distance / x
noncomputable def new_car_fuel_consumption (x : ℝ) := trip_distance / (new_car_efficiency x)

noncomputable def old_car_trip_cost (x c : ℝ) := (trip_distance / x) * c
noncomputable def new_car_trip_cost (x c : ℝ) := (trip_distance / (new_car_efficiency x)) * (diesel_cost c)

noncomputable def savings (x c : ℝ) := old_car_trip_cost x c - new_car_trip_cost x c
noncomputable def percentage_savings (x c : ℝ) := (savings x c) / (old_car_trip_cost x c) * 100

theorem elmer_saves_21_875_percent (x c : ℝ) : percentage_savings x c = 21.875 := 
sorry

end elmer_saves_21_875_percent_l194_194707


namespace simplify_expression_l194_194885

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
    ((x^2 + 1) / (x - 1)) - (2 * x / (x - 1)) = x - 1 := 
by
    sorry

end simplify_expression_l194_194885


namespace mass_of_sodium_acetate_formed_l194_194683

-- Define the reaction conditions and stoichiometry
def initial_moles_acetic_acid : ℝ := 3
def initial_moles_sodium_hydroxide : ℝ := 4
def initial_reaction_moles_acetic_acid_with_sodium_carbonate : ℝ := 2
def initial_reaction_moles_sodium_carbonate : ℝ := 1
def product_moles_sodium_acetate_from_step1 : ℝ := 2
def remaining_moles_acetic_acid : ℝ := initial_moles_acetic_acid - initial_reaction_moles_acetic_acid_with_sodium_carbonate
def product_moles_sodium_acetate_from_step2 : ℝ := remaining_moles_acetic_acid
def total_moles_sodium_acetate : ℝ := product_moles_sodium_acetate_from_step1 + product_moles_sodium_acetate_from_step2
def molar_mass_sodium_acetate : ℝ := 82.04

-- Translate to the equivalent proof problem
theorem mass_of_sodium_acetate_formed :
  total_moles_sodium_acetate * molar_mass_sodium_acetate = 246.12 :=
by
  -- The detailed proof steps would go here
  sorry

end mass_of_sodium_acetate_formed_l194_194683


namespace chinese_chess_draw_probability_l194_194723

theorem chinese_chess_draw_probability (pMingNotLosing : ℚ) (pDongLosing : ℚ) : 
    pMingNotLosing = 3/4 → 
    pDongLosing = 1/2 → 
    (pMingNotLosing - (1 - pDongLosing)) = 1/4 :=
by
  intros
  sorry

end chinese_chess_draw_probability_l194_194723


namespace polynomial_proof_l194_194926

theorem polynomial_proof (x : ℝ) : 
  (2 * x^2 + 5 * x + 4) = (2 * x^2 + 5 * x - 2) + (10 * x + 6) :=
by sorry

end polynomial_proof_l194_194926


namespace hyperbola_represents_l194_194618

theorem hyperbola_represents (k : ℝ) : 
  (k - 2) * (5 - k) < 0 ↔ (k < 2 ∨ k > 5) :=
by
  sorry

end hyperbola_represents_l194_194618


namespace divisor_of_a_l194_194899

theorem divisor_of_a (a b c d : ℕ) (h1 : Nat.gcd a b = 18) (h2 : Nat.gcd b c = 45) 
  (h3 : Nat.gcd c d = 75) (h4 : 80 < Nat.gcd d a ∧ Nat.gcd d a < 120) : 
  7 ∣ a :=
by
  sorry

end divisor_of_a_l194_194899


namespace units_digit_sum_l194_194547

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum
  (h1 : units_digit 13 = 3)
  (h2 : units_digit 41 = 1)
  (h3 : units_digit 27 = 7)
  (h4 : units_digit 34 = 4) :
  units_digit ((13 * 41) + (27 * 34)) = 1 :=
by
  sorry

end units_digit_sum_l194_194547


namespace molecular_weight_constant_l194_194585

-- Define the molecular weight of bleach
def molecular_weight_bleach (num_moles : Nat) : Nat := 222

-- Theorem stating the molecular weight of any amount of bleach is 222 g/mol
theorem molecular_weight_constant (n : Nat) : molecular_weight_bleach n = 222 :=
by
  sorry

end molecular_weight_constant_l194_194585


namespace y_intercept_l194_194241

theorem y_intercept (x y : ℝ) (h : 2 * x - 3 * y = 6) : x = 0 → y = -2 :=
by
  intro h₁
  sorry

end y_intercept_l194_194241


namespace probability_snow_at_least_once_l194_194838

-- Define the probabilities given in the conditions
def p_day_1_3 : ℚ := 1 / 3
def p_day_4_7 : ℚ := 1 / 4
def p_day_8_10 : ℚ := 1 / 2

-- Define the complementary no-snow probabilities
def p_no_snow_day_1_3 : ℚ := 2 / 3
def p_no_snow_day_4_7 : ℚ := 3 / 4
def p_no_snow_day_8_10 : ℚ := 1 / 2

-- Compute the total probability of no snow for all ten days
def p_no_snow_all_days : ℚ :=
  (p_no_snow_day_1_3 ^ 3) * (p_no_snow_day_4_7 ^ 4) * (p_no_snow_day_8_10 ^ 3)

-- Define the proof problem: Calculate probability of at least one snow day
theorem probability_snow_at_least_once : (1 - p_no_snow_all_days) = 2277 / 2304 := by
  sorry

end probability_snow_at_least_once_l194_194838


namespace lily_has_26_dollars_left_for_coffee_l194_194816

-- Define the initial amount of money Lily has
def initialMoney : ℕ := 60

-- Define the costs of items
def celeryCost : ℕ := 5
def cerealCost : ℕ := 12 / 2
def breadCost : ℕ := 8
def milkCost : ℕ := 10 * 9 / 10
def potatoCostEach : ℕ := 1
def numberOfPotatoes : ℕ := 6
def totalPotatoCost : ℕ := potatoCostEach * numberOfPotatoes

-- Define the total amount spent on the items
def totalSpent : ℕ := celeryCost + cerealCost + breadCost + milkCost + totalPotatoCost

-- Define the amount left for coffee
def amountLeftForCoffee : ℕ := initialMoney - totalSpent

-- The theorem to prove
theorem lily_has_26_dollars_left_for_coffee :
  amountLeftForCoffee = 26 := by
  sorry

end lily_has_26_dollars_left_for_coffee_l194_194816


namespace part1_part2_l194_194144

/-- Definition of set A as roots of the equation x^2 - 3x + 2 = 0 --/
def set_A : Set ℝ := {x | x ^ 2 - 3 * x + 2 = 0}

/-- Definition of set B as roots of the equation x^2 + (a - 1)x + a^2 - 5 = 0 --/
def set_B (a : ℝ) : Set ℝ := {x | x ^ 2 + (a - 1) * x + a ^ 2 - 5 = 0}

/-- Proof for intersection condition --/
theorem part1 (a : ℝ) : (set_A ∩ set_B a = {2}) → (a = -3 ∨ a = 1) := by
  sorry

/-- Proof for union condition --/
theorem part2 (a : ℝ) : (set_A ∪ set_B a = set_A) → (a ≤ -3 ∨ a > 7 / 3) := by
  sorry

end part1_part2_l194_194144


namespace travel_speed_is_four_l194_194195
-- Import the required library

-- Define the conditions
def jacksSpeed (x : ℝ) : ℝ := x^2 - 13 * x - 26
def jillsDistance (x : ℝ) : ℝ := x^2 - 5 * x - 66
def jillsTime (x : ℝ) : ℝ := x + 8

-- Prove the equivalent statement
theorem travel_speed_is_four (x : ℝ) (h : x = 15) :
  jillsDistance x / jillsTime x = 4 ∧ jacksSpeed x = 4 := 
by sorry

end travel_speed_is_four_l194_194195


namespace determine_constants_l194_194706

theorem determine_constants (P Q R : ℝ) :
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
    (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ↔
    (P = 7 ∧ Q = -9 ∧ R = 5) :=
by
  sorry

end determine_constants_l194_194706


namespace find_y_l194_194601

theorem find_y : ∀ (x y : ℤ), x > 0 ∧ y > 0 ∧ x % y = 9 ∧ (x:ℝ) / (y:ℝ) = 96.15 → y = 60 :=
by
  intros x y h
  sorry

end find_y_l194_194601


namespace cell_division_after_three_hours_l194_194756

theorem cell_division_after_three_hours : (2 ^ 6) = 64 := by
  sorry

end cell_division_after_three_hours_l194_194756


namespace scientific_notation_of_42000_l194_194737

theorem scientific_notation_of_42000 : 42000 = 4.2 * 10^4 := 
by 
  sorry

end scientific_notation_of_42000_l194_194737


namespace find_minimum_a_l194_194193

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x

theorem find_minimum_a (a : ℝ) :
  (∀ x, 1 ≤ x → 0 ≤ 3 * x^2 + a) ↔ a ≥ -3 :=
by
  sorry

end find_minimum_a_l194_194193


namespace solve_for_y_l194_194061

noncomputable def log5 (x : ℝ) : ℝ := (Real.log x) / (Real.log 5)

theorem solve_for_y (y : ℝ) (h₀ : log5 ((2 * y + 10) / (3 * y - 6)) + log5 ((3 * y - 6) / (y - 4)) = 3) : 
  y = 170 / 41 :=
sorry

end solve_for_y_l194_194061


namespace sum_of_repeating_decimals_l194_194094

noncomputable def x : ℚ := 1 / 9
noncomputable def y : ℚ := 2 / 99
noncomputable def z : ℚ := 3 / 999

theorem sum_of_repeating_decimals :
  x + y + z = 134 / 999 := by
  sorry

end sum_of_repeating_decimals_l194_194094


namespace ratio_red_to_green_apple_l194_194854

def total_apples : ℕ := 496
def green_apples : ℕ := 124
def red_apples : ℕ := total_apples - green_apples

theorem ratio_red_to_green_apple :
  red_apples / green_apples = 93 / 31 :=
by
  sorry

end ratio_red_to_green_apple_l194_194854


namespace complement_of_A_l194_194001

theorem complement_of_A (U : Set ℕ) (A : Set ℕ) (C_UA : Set ℕ) :
  U = {2, 3, 4} →
  A = {x | (x - 1) * (x - 4) < 0 ∧ x ∈ Set.univ} →
  C_UA = {x ∈ U | x ∉ A} →
  C_UA = {4} :=
by
  intros hU hA hCUA
  -- proof omitted, sorry placeholder
  sorry

end complement_of_A_l194_194001


namespace tangerines_in_basket_l194_194553

/-- Let n be the initial number of tangerines in the basket. -/
theorem tangerines_in_basket
  (n : ℕ)
  (c1 : ∃ m : ℕ, m = 10) -- Minyoung ate 10 tangerines from the basket initially
  (c2 : ∃ k : ℕ, k = 6)  -- An hour later, Minyoung ate 6 more tangerines
  (c3 : n = 10 + 6)      -- The basket was empty after these were eaten
  : n = 16 := sorry

end tangerines_in_basket_l194_194553


namespace min_a2_b2_l194_194173

noncomputable def minimum_a2_b2 (a b : ℝ) : Prop :=
  (∃ a b : ℝ, (|(-2*a - 2*b + 4)|) / (Real.sqrt (a^2 + (2*b)^2)) = 2) → (a^2 + b^2 = 2)

theorem min_a2_b2 : minimum_a2_b2 a b :=
by
  sorry

end min_a2_b2_l194_194173


namespace operation_B_is_not_algorithm_l194_194475

-- Define what constitutes an algorithm.
def is_algorithm (desc : String) : Prop :=
  desc = "clear and finite steps to solve a certain type of problem"

-- Define given operations.
def operation_A : String := "Calculating the area of a circle given its radius"
def operation_B : String := "Calculating the possibility of reaching 24 by randomly drawing 4 playing cards"
def operation_C : String := "Finding the equation of a line given two points in the coordinate plane"
def operation_D : String := "Operations of addition, subtraction, multiplication, and division"

-- Define expected property of an algorithm.
def is_algorithm_A : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_B : Prop := is_algorithm "cannot describe precise steps"
def is_algorithm_C : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"
def is_algorithm_D : Prop := is_algorithm "procedural, explicit, and finite steps lead to the result"

theorem operation_B_is_not_algorithm :
  ¬ (is_algorithm operation_B) :=
by
   -- Change this line to the theorem proof.
   sorry

end operation_B_is_not_algorithm_l194_194475


namespace inequality_solution_l194_194366

theorem inequality_solution (x : ℝ) : |2 * x - 7| < 3 → 2 < x ∧ x < 5 :=
by
  sorry

end inequality_solution_l194_194366


namespace vehicle_combinations_count_l194_194215

theorem vehicle_combinations_count :
  ∃ (x y : ℕ), (4 * x + y = 79) ∧ (∃ (n : ℕ), n = 19) :=
sorry

end vehicle_combinations_count_l194_194215


namespace Chad_savings_l194_194647

theorem Chad_savings :
  let earnings_mowing := 600
  let earnings_birthday := 250
  let earnings_video_games := 150
  let earnings_odd_jobs := 150
  let total_earnings := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := 0.40
  let savings := savings_rate * total_earnings
  savings = 460 :=
by
  -- Definitions
  let earnings_mowing : ℤ := 600
  let earnings_birthday : ℤ := 250
  let earnings_video_games : ℤ := 150
  let earnings_odd_jobs : ℤ := 150
  let total_earnings : ℤ := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := (40:ℚ) / 100
  let savings : ℚ := savings_rate * total_earnings
  -- Proof (to be completed by sorry)
  exact sorry

end Chad_savings_l194_194647


namespace peanut_butter_revenue_l194_194563

theorem peanut_butter_revenue :
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  plantation_length * plantation_width * peanuts_per_sqft * butter_from_peanuts_ratio / 1000 * butter_price_per_kg = 31250 := 
by
  let plantation_length := 500
  let plantation_width := 500
  let peanuts_per_sqft := 50
  let butter_from_peanuts_ratio := 5 / 20
  let butter_price_per_kg := 10
  sorry

end peanut_butter_revenue_l194_194563


namespace find_k_no_solution_l194_194532

-- Conditions
def vector1 : ℝ × ℝ := (1, 3)
def direction1 : ℝ × ℝ := (5, -8)
def vector2 : ℝ × ℝ := (0, -1)
def direction2 (k : ℝ) : ℝ × ℝ := (-2, k)

-- Statement
theorem find_k_no_solution (k : ℝ) : 
  (∀ t s : ℝ, vector1 + t • direction1 ≠ vector2 + s • direction2 k) ↔ k = 16 / 5 :=
sorry

end find_k_no_solution_l194_194532


namespace sin_double_angle_l194_194970

theorem sin_double_angle (A : ℝ) (h1 : π / 2 < A) (h2 : A < π) (h3 : Real.sin A = 4 / 5) : Real.sin (2 * A) = -24 / 25 := 
by 
  sorry

end sin_double_angle_l194_194970


namespace radius_of_circle_l194_194632

noncomputable def circle_radius (x y : ℝ) : ℝ := 
  let lhs := x^2 - 8 * x + y^2 - 4 * y + 16
  if lhs = 0 then 2 else 0

theorem radius_of_circle : circle_radius 0 0 = 2 :=
sorry

end radius_of_circle_l194_194632


namespace prob_of_three_digit_divisible_by_3_l194_194477

/-- Define the exponents and the given condition --/
def a : ℕ := 5
def b : ℕ := 2
def c : ℕ := 3
def d : ℕ := 1

def condition : Prop := (2^a) * (3^b) * (5^c) * (7^d) = 252000

/-- The probability that a randomly chosen three-digit number formed by any 3 of a, b, c, d 
    is divisible by 3 and less than 250 is 1/4 --/
theorem prob_of_three_digit_divisible_by_3 :
  condition →
  ((sorry : ℝ) = 1/4) := sorry

end prob_of_three_digit_divisible_by_3_l194_194477


namespace road_completion_l194_194073

/- 
  The company "Roga and Kopyta" undertook a project to build a road 100 km long. 
  The construction plan is: 
  - In the first month, 1 km of the road will be built.
  - Subsequently, if by the beginning of some month A km is already completed, then during that month an additional 1 / A^10 km of road will be constructed.
  Prove that the road will be completed within 100^11 months.
-/

theorem road_completion (L : ℕ → ℝ) (h1 : L 1 = 1)
  (h2 : ∀ n ≥ 1, L (n + 1) = L n + 1 / (L n) ^ 10) :
  ∃ m ≤ 100 ^ 11, L m ≥ 100 := 
  sorry

end road_completion_l194_194073


namespace least_number_of_trees_l194_194882

theorem least_number_of_trees (n : ℕ) :
  (∃ k₄ k₅ k₆, n = 4 * k₄ ∧ n = 5 * k₅ ∧ n = 6 * k₆) ↔ n = 60 :=
by 
  sorry

end least_number_of_trees_l194_194882


namespace actual_distance_traveled_l194_194943

theorem actual_distance_traveled (D : ℝ) (T : ℝ) (h1 : D = 15 * T) (h2 : D + 35 = 25 * T) : D = 52.5 := 
by
  sorry

end actual_distance_traveled_l194_194943


namespace amount_spent_on_machinery_l194_194470

-- Define the given conditions
def raw_materials_spent : ℤ := 80000
def total_amount : ℤ := 137500
def cash_spent : ℤ := (20 * total_amount) / 100

-- The goal is to prove the amount spent on machinery
theorem amount_spent_on_machinery : 
  ∃ M : ℤ, raw_materials_spent + M + cash_spent = total_amount ∧ M = 30000 := by
  sorry

end amount_spent_on_machinery_l194_194470


namespace ellipse_hyperbola_foci_l194_194960

theorem ellipse_hyperbola_foci (c d : ℝ) 
  (h_ellipse : d^2 - c^2 = 25) 
  (h_hyperbola : c^2 + d^2 = 64) : |c * d| = Real.sqrt 868.5 := by
  sorry

end ellipse_hyperbola_foci_l194_194960


namespace household_waste_per_day_l194_194704

theorem household_waste_per_day (total_waste_4_weeks : ℝ) (h : total_waste_4_weeks = 30.8) : 
  (total_waste_4_weeks / 4 / 7) = 1.1 :=
by
  sorry

end household_waste_per_day_l194_194704


namespace trajectory_of_M_is_ellipse_l194_194636

def circle_eq (x y : ℝ) : Prop := ((x + 3)^2 + y^2 = 100)

def point_B (x y : ℝ) : Prop := (x = 3 ∧ y = 0)

def point_on_circle (P : ℝ × ℝ) : Prop :=
  ∃ x y, P = (x, y) ∧ circle_eq x y

def perpendicular_bisector_intersects_CQ_at_M (B P M : ℝ × ℝ) : Prop :=
  (B.fst = 3 ∧ B.snd = 0) ∧
  point_on_circle P ∧
  ∃ r : ℝ, (P.fst + B.fst) / 2 = M.fst ∧ r = (M.snd - P.snd) / (M.fst - P.fst) ∧ 
  r = -(P.fst - B.fst) / (P.snd - B.snd)

theorem trajectory_of_M_is_ellipse (M : ℝ × ℝ) 
  (hC : ∀ x y, circle_eq x y)
  (hB : point_B 3 0)
  (hP : ∃ P : ℝ × ℝ, point_on_circle P)
  (hM : ∃ B P : ℝ × ℝ, perpendicular_bisector_intersects_CQ_at_M B P M) 
: (M.fst^2 / 25 + M.snd^2 / 16 = 1) := 
sorry

end trajectory_of_M_is_ellipse_l194_194636


namespace area_of_sector_l194_194822

theorem area_of_sector (r : ℝ) (n : ℝ) (h_r : r = 3) (h_n : n = 120) : 
  (n / 360) * π * r^2 = 3 * π :=
by
  rw [h_r, h_n] -- Plugin in the given values first
  norm_num     -- Normalize numerical expressions
  sorry        -- Placeholder for further simplification if needed. 

end area_of_sector_l194_194822


namespace sum_x1_x2_l194_194388

open ProbabilityTheory

variable {Ω : Type*} {X : Ω → ℝ}
variable (p1 p2 : ℝ) (x1 x2 : ℝ)
variable (h1 : 2/3 * x1 + 1/3 * x2 = 4/9)
variable (h2 : 2/3 * (x1 - 4/9)^2 + 1/3 * (x2 - 4/9)^2 = 2)
variable (h3 : x1 < x2)

theorem sum_x1_x2 : x1 + x2 = 17/9 :=
by
  sorry

end sum_x1_x2_l194_194388


namespace problem1_problem2_problem3_l194_194764

theorem problem1 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 := sorry
theorem problem2 (p q : ℝ) : (-p * q)^3 = -p^3 * q^3 := sorry
theorem problem3 (a : ℝ) : a^3 * a^4 * a + (a^2)^4 - (-2 * a^4)^2 = -2 * a^8 := sorry

end problem1_problem2_problem3_l194_194764


namespace equal_pair_c_l194_194273

theorem equal_pair_c : (-4)^3 = -(4^3) := 
by {
  sorry
}

end equal_pair_c_l194_194273


namespace lemons_for_lemonade_l194_194713

theorem lemons_for_lemonade (lemons_gallons_ratio : 30 / 25 = x / 10) : x = 12 :=
by
  sorry

end lemons_for_lemonade_l194_194713


namespace GE_eq_GH_l194_194896

variables (A B C D E F G H : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
          [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H]
          
variables (AC : Line A C) (AB : Line A B) (BE : Line B E) (DE : Line D E)
          (BG : Line B G) (AF : Line A F) (DE' : Line D E') (angleC : Angle C = 90)

variables (circB : Circle B BC) (tangentDE : Tangent DE circB E) (perpAB : Perpendicular AC AB)
          (intersectionF : Intersect (PerpendicularLine C AB) BE F)
          (intersectionG : Intersect AF DE G) (intersectionH : Intersect (ParallelLine A BG) DE H)

theorem GE_eq_GH : GE = GH := sorry

end GE_eq_GH_l194_194896


namespace kelly_snacks_l194_194316

theorem kelly_snacks (peanuts raisins : ℝ) (h_peanuts : peanuts = 0.1) (h_raisins : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end kelly_snacks_l194_194316


namespace total_number_of_feet_l194_194635

theorem total_number_of_feet 
  (H C F : ℕ)
  (h1 : H + C = 44)
  (h2 : H = 24)
  (h3 : F = 2 * H + 4 * C) : 
  F = 128 :=
by
  sorry

end total_number_of_feet_l194_194635


namespace interest_rate_first_part_l194_194185

theorem interest_rate_first_part (A A1 A2 I : ℝ) (r : ℝ) :
  A = 3200 →
  A1 = 800 →
  A2 = A - A1 →
  I = 144 →
  (A1 * r / 100 + A2 * 5 / 100 = I) →
  r = 3 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end interest_rate_first_part_l194_194185


namespace goats_difference_l194_194435

-- Definitions of Adam's, Andrew's, and Ahmed's goats
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 2 * adam_goats + 5
def ahmed_goats : ℕ := 13

-- The theorem to prove the difference in goats
theorem goats_difference : andrew_goats - ahmed_goats = 6 :=
by
  sorry

end goats_difference_l194_194435


namespace sum_eq_zero_l194_194340

variable {a b c : ℝ}

theorem sum_eq_zero (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
    (h4 : a ≠ b ∨ b ≠ c ∨ c ≠ a)
    (h5 : (a^2) / (2 * (a^2) + b * c) + (b^2) / (2 * (b^2) + c * a) + (c^2) / (2 * (c^2) + a * b) = 1) :
  a + b + c = 0 :=
sorry

end sum_eq_zero_l194_194340


namespace field_ratio_l194_194744

theorem field_ratio
  (l w : ℕ)
  (pond_length : ℕ)
  (pond_area_ratio : ℚ)
  (field_length : ℕ)
  (field_area : ℕ)
  (hl : l = 24)
  (hp : pond_length = 6)
  (hr : pond_area_ratio = 1 / 8)
  (hm : l % w = 0)
  (ha : field_area = 36 * 8)
  (hf : l * w = field_area) :
  l / w = 2 :=
by
  sorry

end field_ratio_l194_194744


namespace ratio_of_flowers_given_l194_194955

-- Definitions based on conditions
def Collin_flowers : ℕ := 25
def Ingrid_flowers_initial : ℕ := 33
def petals_per_flower : ℕ := 4
def Collin_petals_total : ℕ := 144

-- The ratio of the number of flowers Ingrid gave to Collin to the number of flowers Ingrid had initially
theorem ratio_of_flowers_given :
  let Ingrid_flowers_given := (Collin_petals_total - (Collin_flowers * petals_per_flower)) / petals_per_flower
  let ratio := Ingrid_flowers_given / Ingrid_flowers_initial
  ratio = 1 / 3 :=
by
  sorry

end ratio_of_flowers_given_l194_194955


namespace train_pass_platform_time_l194_194689

theorem train_pass_platform_time (l v t : ℝ) (h1 : v = l / t) (h2 : l > 0) (h3 : t > 0) :
  ∃ T : ℝ, T = 3.5 * t := by
  sorry

end train_pass_platform_time_l194_194689


namespace sqrt_sum_inequality_l194_194930

variable (a b c d : ℝ)

theorem sqrt_sum_inequality
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : a + d = b + c) :
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c :=
by
  sorry

end sqrt_sum_inequality_l194_194930


namespace gcd_459_357_l194_194382

theorem gcd_459_357 :
  Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l194_194382


namespace min_value_of_box_l194_194127

theorem min_value_of_box 
  (a b : ℤ) 
  (h_distinct : a ≠ b) 
  (h_eq : (a * x + b) * (b * x + a) = 34 * x^2 + Box * x + 34) 
  (h_prod : a * b = 34) :
  ∃ (Box : ℤ), Box = 293 :=
by
  sorry

end min_value_of_box_l194_194127


namespace negation_of_proposition_l194_194623

theorem negation_of_proposition (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by
  sorry

end negation_of_proposition_l194_194623


namespace initially_working_machines_l194_194629

theorem initially_working_machines (N R x : ℝ) 
  (h1 : N * R = x / 3) 
  (h2 : 45 * R = x / 2) : 
  N = 30 := by
  sorry

end initially_working_machines_l194_194629


namespace discount_price_l194_194392

theorem discount_price (a : ℝ) (original_price : ℝ) (sold_price : ℝ) :
  original_price = 200 ∧ sold_price = 148 → (original_price * (1 - a/100) * (1 - a/100) = sold_price) :=
by
  sorry

end discount_price_l194_194392


namespace inequality_holds_l194_194842

variable (b : ℝ)

theorem inequality_holds (b : ℝ) : (3 * b - 1) * (4 * b + 1) > (2 * b + 1) * (5 * b - 3) :=
by
  sorry

end inequality_holds_l194_194842


namespace categorize_numbers_l194_194428

noncomputable def positive_numbers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x > 0}

noncomputable def non_neg_integers (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x ≥ 0 ∧ ∃ n : ℤ, x = n}

noncomputable def negative_fractions (nums : Set ℝ) : Set ℝ :=
  {x | x ∈ nums ∧ x < 0 ∧ ∃ n d : ℤ, d ≠ 0 ∧ (x = n / d)}

def given_set : Set ℝ := {6, -3, 2.4, -3/4, 0, -3.14, 2, -7/2, 2/3}

theorem categorize_numbers :
  positive_numbers given_set = {6, 2.4, 2, 2/3} ∧
  non_neg_integers given_set = {6, 0, 2} ∧
  negative_fractions given_set = {-3/4, -3.14, -7/2} :=
by
  sorry

end categorize_numbers_l194_194428


namespace initial_violet_marbles_eq_l194_194345

variable {initial_violet_marbles : Nat}
variable (red_marbles : Nat := 14)
variable (total_marbles : Nat := 78)

theorem initial_violet_marbles_eq :
  initial_violet_marbles = total_marbles - red_marbles := by
  sorry

end initial_violet_marbles_eq_l194_194345


namespace equation_of_perpendicular_line_intersection_l194_194271

theorem equation_of_perpendicular_line_intersection  :
  ∃ (x y : ℝ), 4 * x + 2 * y + 5 = 0 ∧ 3 * x - 2 * y + 9 = 0 ∧ 
               (∃ (m : ℝ), m = 2 ∧ 4 * x - 2 * y + 11 = 0) := 
sorry

end equation_of_perpendicular_line_intersection_l194_194271


namespace liars_count_l194_194624

inductive Person
| Knight
| Liar
| Eccentric

open Person

def isLiarCondition (p : Person) (right : Person) : Prop :=
  match p with
  | Knight => right = Liar
  | Liar => right ≠ Liar
  | Eccentric => True

theorem liars_count (people : Fin 100 → Person) (h : ∀ i, isLiarCondition (people i) (people ((i + 1) % 100))) :
  (∃ n : ℕ, n = 0 ∨ n = 50) :=
sorry

end liars_count_l194_194624


namespace cost_per_meter_l194_194021

-- Defining the parameters and their relationships
def length : ℝ := 58
def breadth : ℝ := length - 16
def total_cost : ℝ := 5300
def perimeter : ℝ := 2 * (length + breadth)

-- Proving the cost per meter of fencing
theorem cost_per_meter : total_cost / perimeter = 26.50 := 
by
  sorry

end cost_per_meter_l194_194021


namespace bijection_condition_l194_194080

variable {n m : ℕ}
variable (f : Fin n → Fin n)

theorem bijection_condition (h_even : m % 2 = 0)
(h_prime : Nat.Prime (n + 1))
(h_bij : Function.Bijective f) :
  ∀ x y : Fin n, (n : ℕ) ∣ (m * x - y : ℕ) → (n + 1) ∣ (f x).val ^ m - (f y).val := sorry

end bijection_condition_l194_194080


namespace rick_ironed_27_pieces_l194_194145

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l194_194145


namespace arithmetic_sequence_common_difference_l194_194264

theorem arithmetic_sequence_common_difference :
  let a := 5
  let a_n := 50
  let S_n := 330
  exists (d n : ℤ), (a + (n - 1) * d = a_n) ∧ (n * (a + a_n) / 2 = S_n) ∧ (d = 45 / 11) :=
by
  let a := 5
  let a_n := 50
  let S_n := 330
  use 45 / 11, 12
  sorry

end arithmetic_sequence_common_difference_l194_194264


namespace manufacturing_percentage_l194_194317

theorem manufacturing_percentage (deg_total : ℝ) (deg_manufacturing : ℝ) (h1 : deg_total = 360) (h2 : deg_manufacturing = 126) : 
  (deg_manufacturing / deg_total * 100) = 35 := by
  sorry

end manufacturing_percentage_l194_194317


namespace sin_cos_sum_l194_194794

/--
Given point P with coordinates (-3, 4) lies on the terminal side of angle α, prove that
sin α + cos α = 1/5.
-/
theorem sin_cos_sum (α : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4)) :
  Real.sin α + Real.cos α = 1/5 := sorry

end sin_cos_sum_l194_194794


namespace weightlifting_winner_l194_194790

theorem weightlifting_winner
  (A B C : ℝ)
  (h1 : A + B = 220)
  (h2 : A + C = 240)
  (h3 : B + C = 250) :
  max A (max B C) = 135 := 
sorry

end weightlifting_winner_l194_194790


namespace value_of_a_set_of_x_l194_194834

open Real

noncomputable def f (x a : ℝ) : ℝ := sin (x + π / 6) + sin (x - π / 6) + cos x + a

theorem value_of_a : ∀ a, (∀ x, f x a ≤ 1) → a = -1 :=
sorry

theorem set_of_x (a : ℝ) (k : ℤ) : a = -1 →
  {x : ℝ | f x a = 0} = {x | ∃ k : ℤ, x = 2 * k * π ∨ x = 2 * k * π + 2 * π / 3} :=
sorry

end value_of_a_set_of_x_l194_194834


namespace extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l194_194229

-- Define the function f(x) = 2*x^3 + 3*(a-2)*x^2 - 12*a*x
def f (x : ℝ) (a : ℝ) := 2*x^3 + 3*(a-2)*x^2 - 12*a*x

-- Define the function f(x) when a = 0
def f_a_zero (x : ℝ) := f x 0

-- Define the intervals and extreme values problem
theorem extreme_values_of_f_a_zero_on_interval :
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 4, f_a_zero x ≤ max ∧ f_a_zero x ≥ min) ∧ max = 32 ∧ min = -40 :=
sorry

-- Define the function for the derivative of f(x)
def f_derivative (x : ℝ) (a : ℝ) := 6*x^2 + 6*(a-2)*x - 12*a

-- Prove the monotonicity based on the value of a
theorem monotonicity_of_f (a : ℝ) :
  (a > -2 → (∀ x, x < -a → f_derivative x a > 0) ∧ (∀ x, -a < x ∧ x < 2 → f_derivative x a < 0) ∧ (∀ x, x > 2 → f_derivative x a > 0)) ∧
  (a = -2 → ∀ x, f_derivative x a ≥ 0) ∧
  (a < -2 → (∀ x, x < 2 → f_derivative x a > 0) ∧ (∀ x, 2 < x ∧ x < -a → f_derivative x a < 0) ∧ (∀ x, x > -a → f_derivative x a > 0)) :=
sorry

end extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l194_194229


namespace cos_double_angle_value_l194_194105

theorem cos_double_angle_value (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 :=
by
  sorry

end cos_double_angle_value_l194_194105


namespace remainders_are_distinct_l194_194668

theorem remainders_are_distinct (a : ℕ → ℕ) (H1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i ≠ a (i % 100 + 1))
  (H2 : ∃ r1 r2 : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ 100 → a i % a (i % 100 + 1) = r1 ∨ a i % a (i % 100 + 1) = r2) :
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 100 → (a (i % 100 + 1) % a i) ≠ (a (j % 100 + 1) % a j) :=
by
  sorry

end remainders_are_distinct_l194_194668


namespace segments_form_quadrilateral_l194_194698

theorem segments_form_quadrilateral (a d : ℝ) (h_pos : a > 0 ∧ d > 0) (h_sum : 4 * a + 6 * d = 3) : 
  (∃ s1 s2 s3 s4 : ℝ, s1 + s2 + s3 > s4 ∧ s1 + s2 + s4 > s3 ∧ s1 + s3 + s4 > s2 ∧ s2 + s3 + s4 > s1) :=
sorry

end segments_form_quadrilateral_l194_194698


namespace smallest_a_b_sum_l194_194887

theorem smallest_a_b_sum :
  ∃ (a b : ℕ), 3^6 * 5^3 * 7^2 = a^b ∧ a + b = 317 := 
sorry

end smallest_a_b_sum_l194_194887


namespace machine_p_takes_longer_l194_194757

variable (MachineP MachineQ MachineA : Type)
variable (s_prockets_per_hr : MachineA → ℝ)
variable (time_produce_s_prockets : MachineP → ℝ → ℝ)

noncomputable def machine_a_production : ℝ := 3
noncomputable def machine_q_production : ℝ := machine_a_production + 0.10 * machine_a_production

noncomputable def machine_q_time : ℝ := 330 / machine_q_production
noncomputable def additional_time : ℝ := sorry -- Since L is undefined

axiom machine_p_time : ℝ
axiom machine_p_time_eq_machine_q_time_plus_additional : machine_p_time = machine_q_time + additional_time

theorem machine_p_takes_longer : machine_p_time > machine_q_time := by
  rw [machine_p_time_eq_machine_q_time_plus_additional]
  exact lt_add_of_pos_right machine_q_time sorry  -- Need the exact L to conclude


end machine_p_takes_longer_l194_194757


namespace find_t_l194_194570

theorem find_t (s t : ℝ) (h1 : 15 * s + 7 * t = 236) (h2 : t = 2 * s + 1) : t = 16.793 :=
by
  sorry

end find_t_l194_194570


namespace total_stairs_climbed_l194_194604

theorem total_stairs_climbed (samir_stairs veronica_stairs ravi_stairs total_stairs_climbed : ℕ) 
  (h_samir : samir_stairs = 318)
  (h_veronica : veronica_stairs = (318 / 2) + 18)
  (h_ravi : ravi_stairs = (3 * veronica_stairs) / 2) :
  samir_stairs + veronica_stairs + ravi_stairs = total_stairs_climbed ->
  total_stairs_climbed = 761 :=
by
  sorry

end total_stairs_climbed_l194_194604


namespace negation_of_existential_statement_l194_194750

theorem negation_of_existential_statement : 
  (¬∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end negation_of_existential_statement_l194_194750


namespace problem1_problem2_problem3_l194_194013

variables (x y a b c : ℚ)

-- Definition of the operation *
def op_star (x y : ℚ) : ℚ := x * y + 1

-- Prove that 2 * 3 = 7 using the operation *
theorem problem1 : op_star 2 3 = 7 :=
by
  sorry

-- Prove that (1 * 4) * (-1/2) = -3/2 using the operation *
theorem problem2 : op_star (op_star 1 4) (-1/2) = -3/2 :=
by
  sorry

-- Prove the relationship a * (b + c) + 1 = a * b + a * c using the operation *
theorem problem3 : op_star a (b + c) + 1 = op_star a b + op_star a c :=
by
  sorry

end problem1_problem2_problem3_l194_194013


namespace toothpicks_at_20th_stage_l194_194172

def toothpicks_in_stage (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_at_20th_stage : toothpicks_in_stage 20 = 61 :=
by 
  sorry

end toothpicks_at_20th_stage_l194_194172


namespace perfect_square_trinomial_l194_194047

theorem perfect_square_trinomial (m : ℝ) :
  (∃ (a : ℝ), (x^2 + mx + 1) = (x + a)^2) ↔ (m = 2 ∨ m = -2) := sorry

end perfect_square_trinomial_l194_194047


namespace grandma_Olga_grandchildren_l194_194551

def daughters : Nat := 3
def sons : Nat := 3
def sons_per_daughter : Nat := 6
def daughters_per_son : Nat := 5

theorem grandma_Olga_grandchildren : 
  (daughters * sons_per_daughter) + (sons * daughters_per_son) = 33 := by
  sorry

end grandma_Olga_grandchildren_l194_194551


namespace no_triples_of_consecutive_numbers_l194_194111

theorem no_triples_of_consecutive_numbers (n : ℤ) (a : ℕ) (h : 1 ≤ a ∧ a ≤ 9) :
  ¬(3 * n^2 + 2 = 1111 * a) :=
by sorry

end no_triples_of_consecutive_numbers_l194_194111


namespace guilt_proof_l194_194065

theorem guilt_proof (X Y : Prop) (h1 : X ∨ Y) (h2 : ¬X) : Y :=
by
  sorry

end guilt_proof_l194_194065


namespace set_operation_correct_l194_194254

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

-- Define the operation A * B
def set_operation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- State the theorem to be proved
theorem set_operation_correct : set_operation A B = {1, 3} :=
sorry

end set_operation_correct_l194_194254


namespace number_of_even_red_faces_cubes_l194_194914

def painted_cubes_even_faces : Prop :=
  let block_length := 4
  let block_width := 4
  let block_height := 1
  let edge_cubes_count := 8  -- The count of edge cubes excluding corners
  edge_cubes_count = 8

theorem number_of_even_red_faces_cubes : painted_cubes_even_faces := by
  sorry

end number_of_even_red_faces_cubes_l194_194914


namespace problem_statement_l194_194545

def A := {x : ℝ | x * (x - 1) < 0}
def B := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem problem_statement : A ⊆ {y : ℝ | y ≥ 0} :=
sorry

end problem_statement_l194_194545


namespace index_card_area_reduction_l194_194289

theorem index_card_area_reduction :
  ∀ (length width : ℕ),
  (length = 5 ∧ width = 7) →
  ((length - 2) * width = 21) →
  (length * (width - 2) = 25) :=
by
  intros length width h1 h2
  rcases h1 with ⟨h_length, h_width⟩
  sorry

end index_card_area_reduction_l194_194289


namespace faye_science_problems_l194_194321

variable (total_problems math_problems science_problems : Nat)
variable (finished_at_school left_for_homework : Nat)

theorem faye_science_problems :
  finished_at_school = 40 ∧ left_for_homework = 15 ∧ math_problems = 46 →
  total_problems = finished_at_school + left_for_homework →
  science_problems = total_problems - math_problems →
  science_problems = 9 :=
by
  sorry

end faye_science_problems_l194_194321


namespace monotonic_increasing_intervals_l194_194682

noncomputable def f (x : ℝ) : ℝ := (Real.cos (x - Real.pi / 6))^2

theorem monotonic_increasing_intervals (k : ℤ) : 
  ∃ t : Set ℝ, t = Set.Ioo (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi) ∧ 
    ∀ x y, x ∈ t → y ∈ t → x ≤ y → f x ≤ f y :=
sorry

end monotonic_increasing_intervals_l194_194682


namespace total_flowers_l194_194583

def number_of_pots : ℕ := 141
def flowers_per_pot : ℕ := 71

theorem total_flowers : number_of_pots * flowers_per_pot = 10011 :=
by
  -- formal proof goes here
  sorry

end total_flowers_l194_194583


namespace negation_proposition_l194_194735

theorem negation_proposition :
  ¬(∃ x_0 : ℝ, x_0^2 + x_0 - 2 < 0) ↔ ∀ x_0 : ℝ, x_0^2 + x_0 - 2 ≥ 0 :=
by
  sorry

end negation_proposition_l194_194735


namespace acute_triangle_tangent_sum_geq_3_sqrt_3_l194_194179

theorem acute_triangle_tangent_sum_geq_3_sqrt_3 {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (h_sum : α + β + γ = π)
  (acute_α : α < π / 2) (acute_β : β < π / 2) (acute_γ : γ < π / 2) :
  Real.tan α + Real.tan β + Real.tan γ >= 3 * Real.sqrt 3 :=
sorry

end acute_triangle_tangent_sum_geq_3_sqrt_3_l194_194179


namespace bananas_in_each_box_l194_194455

theorem bananas_in_each_box 
    (bananas : ℕ) (boxes : ℕ) 
    (h_bananas : bananas = 40) 
    (h_boxes : boxes = 10) : 
    bananas / boxes = 4 := by
  sorry

end bananas_in_each_box_l194_194455


namespace same_percentage_loss_as_profit_l194_194543

theorem same_percentage_loss_as_profit (CP SP L : ℝ) (h_prof : SP = 1720)
  (h_loss : L = CP - (14.67 / 100) * CP)
  (h_25_prof : 1.25 * CP = 1875) :
  L = 1280 := 
  sorry

end same_percentage_loss_as_profit_l194_194543


namespace num_of_valid_three_digit_numbers_l194_194052

def valid_three_digit_numbers : ℕ :=
  let valid_numbers : List (ℕ × ℕ × ℕ) :=
    [(2, 3, 4), (4, 6, 8)]
  valid_numbers.length

theorem num_of_valid_three_digit_numbers :
  valid_three_digit_numbers = 2 :=
by
  sorry

end num_of_valid_three_digit_numbers_l194_194052


namespace exponential_decreasing_l194_194131

theorem exponential_decreasing (a : ℝ) : (∀ x y : ℝ, x < y → (2 * a - 1)^y < (2 * a - 1)^x) ↔ (1 / 2 < a ∧ a < 1) := 
by
    sorry

end exponential_decreasing_l194_194131


namespace seven_distinct_numbers_with_reversed_digits_l194_194592

theorem seven_distinct_numbers_with_reversed_digits (x y : ℕ) :
  (∃ a b c d e f g : ℕ, 
  (10 * a + b + 18 = 10 * b + a) ∧ (10 * c + d + 18 = 10 * d + c) ∧ 
  (10 * e + f + 18 = 10 * f + e) ∧ (10 * g + y + 18 = 10 * y + g) ∧ 
  a ≠ c ∧ a ≠ e ∧ a ≠ g ∧ 
  c ≠ e ∧ c ≠ g ∧ 
  e ≠ g ∧ 
  (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧
  (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧
  (1 ≤ e ∧ e <= 9) ∧ (1 ≤ f ∧ f <= 9) ∧
  (1 ≤ g ∧ g <= 9) ∧ (1 ≤ y ∧ y <= 9)) :=
sorry

end seven_distinct_numbers_with_reversed_digits_l194_194592


namespace intersection_of_A_and_B_l194_194591

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := { y | ∃ x ∈ A, y = x + 1 }

theorem intersection_of_A_and_B :
  A ∩ B = {2, 3, 4} :=
sorry

end intersection_of_A_and_B_l194_194591


namespace polynomial_problem_l194_194607

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem polynomial_problem (f_nonzero : ∀ x, f x ≠ 0) 
  (h1 : ∀ x, f (g x) = f x * g x)
  (h2 : g 3 = 10)
  (h3 : ∃ a b, g x = a * x + b) :
  g x = 2 * x + 4 :=
sorry

end polynomial_problem_l194_194607


namespace max_value_of_exp_sum_l194_194831

theorem max_value_of_exp_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_pos : 0 < a * b) :
    ∃ θ : ℝ, a * Real.exp θ + b * Real.exp (-θ) = 2 * Real.sqrt (a * b) :=
by
  sorry

end max_value_of_exp_sum_l194_194831


namespace local_tax_deduction_in_cents_l194_194676

def aliciaHourlyWageInDollars : ℝ := 25
def taxDeductionRate : ℝ := 0.02
def aliciaHourlyWageInCents := aliciaHourlyWageInDollars * 100

theorem local_tax_deduction_in_cents :
  taxDeductionRate * aliciaHourlyWageInCents = 50 :=
by 
  -- Proof goes here
  sorry

end local_tax_deduction_in_cents_l194_194676


namespace committee_membership_l194_194393

theorem committee_membership (n : ℕ) (h1 : 2 * n = 6) (h2 : (n - 1 : ℚ) / 5 = 0.4) : n = 3 := 
sorry

end committee_membership_l194_194393


namespace star_in_S_star_associative_l194_194873

def S (x : ℕ) : Prop :=
  x > 1 ∧ x % 2 = 1

def f (x : ℕ) : ℕ :=
  Nat.log2 x

def star (a b : ℕ) : ℕ :=
  a + 2 ^ (f a) * (b - 3)

theorem star_in_S (a b : ℕ) (h_a : S a) (h_b : S b) : S (star a b) :=
  sorry

theorem star_associative (a b c : ℕ) (h_a : S a) (h_b : S b) (h_c : S c) :
  star (star a b) c = star a (star b c) :=
  sorry

end star_in_S_star_associative_l194_194873


namespace problem1_problem2_problem3_l194_194740

-- First problem: Prove x = 4.2 given x + 2x = 12.6
theorem problem1 (x : ℝ) (h1 : x + 2 * x = 12.6) : x = 4.2 :=
  sorry

-- Second problem: Prove x = 2/5 given 1/4 * x + 1/2 = 3/5
theorem problem2 (x : ℚ) (h2 : (1 / 4) * x + 1 / 2 = 3 / 5) : x = 2 / 5 :=
  sorry

-- Third problem: Prove x = 20 given x + 130% * x = 46 (where 130% is 130/100)
theorem problem3 (x : ℝ) (h3 : x + (130 / 100) * x = 46) : x = 20 :=
  sorry

end problem1_problem2_problem3_l194_194740


namespace problem_solution_l194_194350

open Real

noncomputable def length_and_slope_MP 
    (length_MN : ℝ) 
    (slope_MN : ℝ) 
    (length_NP : ℝ) 
    (slope_NP : ℝ) 
    : (ℝ × ℝ) := sorry

theorem problem_solution :
  length_and_slope_MP 6 14 7 8 = (5.55, 25.9) :=
  sorry

end problem_solution_l194_194350


namespace product_increased_l194_194212

theorem product_increased (a b c : ℕ) (h1 : a = 1) (h2: b = 1) (h3: c = 676) :
  ((a - 3) * (b - 3) * (c - 3) = a * b * c + 2016) :=
by
  simp [h1, h2, h3]
  sorry

end product_increased_l194_194212


namespace boxes_needed_l194_194230

theorem boxes_needed (balls : ℕ) (balls_per_box : ℕ) (h1 : balls = 10) (h2 : balls_per_box = 5) : balls / balls_per_box = 2 := by
  sorry

end boxes_needed_l194_194230


namespace rupert_jumps_more_l194_194603

theorem rupert_jumps_more (Ronald_jumps Rupert_jumps total_jumps : ℕ)
  (h1 : Ronald_jumps = 157)
  (h2 : total_jumps = 243)
  (h3 : Rupert_jumps + Ronald_jumps = total_jumps) :
  Rupert_jumps - Ronald_jumps = 86 :=
by
  sorry

end rupert_jumps_more_l194_194603


namespace ants_no_collision_probability_l194_194951

-- Definitions
def cube_vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

def adjacent (v : ℕ) : Finset ℕ :=
  match v with
  | 0 => {1, 3, 4}
  | 1 => {0, 2, 5}
  | 2 => {1, 3, 6}
  | 3 => {0, 2, 7}
  | 4 => {0, 5, 7}
  | 5 => {1, 4, 6}
  | 6 => {2, 5, 7}
  | 7 => {3, 4, 6}
  | _ => ∅

-- Hypothesis: Each ant moves independently to one of the three adjacent vertices.

-- Result to prove
def X : ℕ := sorry  -- The number of valid ways ants can move without collisions

theorem ants_no_collision_probability : 
  ∃ X, (X / (3 : ℕ)^8 = X / 6561) :=
  by
    sorry

end ants_no_collision_probability_l194_194951


namespace evaluate_101_times_101_l194_194311

theorem evaluate_101_times_101 : (101 * 101 = 10201) :=
by {
  sorry
}

end evaluate_101_times_101_l194_194311


namespace area_of_triangle_DEF_l194_194263

-- Define point D
def pointD : ℝ × ℝ := (2, 5)

-- Reflect D over the y-axis to get E
def reflectY (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, P.2)
def pointE : ℝ × ℝ := reflectY pointD

-- Reflect E over the line y = -x to get F
def reflectYX (P : ℝ × ℝ) : ℝ × ℝ := (-P.2, -P.1)
def pointF : ℝ × ℝ := reflectYX pointE

-- Define function to calculate the area of the triangle given three points
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

-- Define the Lean 4 statement
theorem area_of_triangle_DEF : triangle_area pointD pointE pointF = 6 := by
  sorry

end area_of_triangle_DEF_l194_194263


namespace carbon_atoms_in_compound_l194_194232

theorem carbon_atoms_in_compound 
    (molecular_weight : ℕ := 65)
    (carbon_weight : ℕ := 12)
    (hydrogen_weight : ℕ := 1)
    (oxygen_weight : ℕ := 16)
    (hydrogen_atoms : ℕ := 1)
    (oxygen_atoms : ℕ := 1) :
    ∃ (carbon_atoms : ℕ), molecular_weight = (carbon_atoms * carbon_weight) + (hydrogen_atoms * hydrogen_weight) + (oxygen_atoms * oxygen_weight) ∧ carbon_atoms = 4 :=
by
  sorry

end carbon_atoms_in_compound_l194_194232


namespace ceilings_left_correct_l194_194100

def total_ceilings : ℕ := 28
def ceilings_painted_this_week : ℕ := 12
def ceilings_painted_next_week : ℕ := ceilings_painted_this_week / 4
def ceilings_left_to_paint : ℕ := total_ceilings - (ceilings_painted_this_week + ceilings_painted_next_week)

theorem ceilings_left_correct : ceilings_left_to_paint = 13 := by
  sorry

end ceilings_left_correct_l194_194100


namespace equation_of_line_projection_l194_194658

theorem equation_of_line_projection (x y : ℝ) (m : ℝ) (x1 x2 : ℝ) (d : ℝ)
  (h1 : (5, 3) ∈ {(x, y) | y = 3 + m * (x - 5)})
  (h2 : x1 = (16 + 20 * m - 12) / (4 * m + 3))
  (h3 : x2 = (1 + 20 * m - 12) / (4 * m + 3))
  (h4 : abs (x1 - x2) = 1) :
  (y = 3 * x - 12 ∨ y = -4.5 * x + 25.5) :=
sorry

end equation_of_line_projection_l194_194658


namespace mixture_milk_quantity_l194_194742

variable (M W : ℕ)

theorem mixture_milk_quantity
  (h1 : M = 2 * W)
  (h2 : 6 * (W + 10) = 5 * M) :
  M = 30 := by
  sorry

end mixture_milk_quantity_l194_194742


namespace avg_rate_of_change_l194_194020

def f (x : ℝ) := 2 * x + 1

theorem avg_rate_of_change : (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end avg_rate_of_change_l194_194020


namespace probability_of_four_of_a_kind_is_correct_l194_194450

noncomputable def probability_four_of_a_kind: ℚ :=
  let total_ways := Nat.choose 52 5
  let successful_ways := 13 * 1 * 12 * 4
  (successful_ways: ℚ) / (total_ways: ℚ)

theorem probability_of_four_of_a_kind_is_correct :
  probability_four_of_a_kind = 13 / 54145 := 
by
  -- sorry is used because we are only writing the statement, no proof required
  sorry

end probability_of_four_of_a_kind_is_correct_l194_194450


namespace no_real_solutions_for_eqn_l194_194672

theorem no_real_solutions_for_eqn :
  ¬ ∃ x : ℝ, (x + 4) ^ 2 = 3 * (x - 2) := 
by 
  sorry

end no_real_solutions_for_eqn_l194_194672


namespace inequality_non_empty_solution_l194_194783

theorem inequality_non_empty_solution (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) → a ≤ 1 := sorry

end inequality_non_empty_solution_l194_194783


namespace minimum_value_of_f_l194_194412

noncomputable def f (x : ℝ) : ℝ :=
  (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y ≥ (5 / 2), f y = 1 := by
  sorry

end minimum_value_of_f_l194_194412


namespace only_one_student_remains_l194_194793

theorem only_one_student_remains (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2002) :
  (∃! k, k = n ∧ n % 1331 = 0) ↔ n = 1331 :=
by
  sorry

end only_one_student_remains_l194_194793


namespace arithmetic_sequence_problem_l194_194174

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h1 : a 2 + a 3 = 4)
  (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 11 :=
sorry

end arithmetic_sequence_problem_l194_194174


namespace Isabel_paper_used_l194_194139

theorem Isabel_paper_used
  (initial_pieces : ℕ)
  (remaining_pieces : ℕ)
  (initial_condition : initial_pieces = 900)
  (remaining_condition : remaining_pieces = 744) :
  initial_pieces - remaining_pieces = 156 :=
by 
  -- Admitting the proof for now
  sorry

end Isabel_paper_used_l194_194139


namespace rectangular_box_proof_l194_194484

noncomputable def rectangular_box_surface_area
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) : ℝ :=
2 * (a * b + b * c + c * a)

theorem rectangular_box_proof
  (a b c : ℝ)
  (h1 : 4 * (a + b + c) = 140)
  (h2 : a^2 + b^2 + c^2 = 21^2) :
  rectangular_box_surface_area a b c h1 h2 = 784 :=
by
  sorry

end rectangular_box_proof_l194_194484


namespace find_line_equation_l194_194614

theorem find_line_equation (k : ℝ) (x y : ℝ) :
  (∀ k, (∃ x y, y = k * x + 1 ∧ x^2 + y^2 - 2 * x - 3 = 0) ↔ x - y + 1 = 0) :=
by
  sorry

end find_line_equation_l194_194614


namespace polygon_diagonalization_l194_194849

theorem polygon_diagonalization (n : ℕ) (h : n ≥ 3) : 
  ∃ (triangles : ℕ), triangles = n - 2 ∧ 
  (∀ (polygons : ℕ), 3 ≤ polygons → polygons < n → ∃ k, k = polygons - 2) := 
by {
  -- base case
  sorry
}

end polygon_diagonalization_l194_194849


namespace problem_statement_l194_194223

variables {c c' d d' : ℝ}

theorem problem_statement (hc : c ≠ 0) (hc' : c' ≠ 0)
  (h : (-d) / (2 * c) = 2 * ((-d') / (3 * c'))) :
  (d / (2 * c)) = 2 * (d' / (3 * c')) :=
by
  sorry

end problem_statement_l194_194223


namespace max_two_integers_abs_leq_50_l194_194991

theorem max_two_integers_abs_leq_50
  (a b c : ℤ) (h_a : a > 100) :
  ∀ {x1 x2 x3 : ℤ}, (abs (a * x1^2 + b * x1 + c) ≤ 50) →
                    (abs (a * x2^2 + b * x2 + c) ≤ 50) →
                    (abs (a * x3^2 + b * x3 + c) ≤ 50) →
                    false :=
sorry

end max_two_integers_abs_leq_50_l194_194991


namespace circumcircle_eq_l194_194239

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B : (ℝ × ℝ) := (4, 0)
noncomputable def C : (ℝ × ℝ) := (0, 6)

theorem circumcircle_eq :
  ∃ h k r, h = 2 ∧ k = 3 ∧ r = 13 ∧ (∀ x y, ((x - h)^2 + (y - k)^2 = r) ↔ (x - 2)^2 + (y - 3)^2 = 13) := sorry

end circumcircle_eq_l194_194239


namespace factor_expression_l194_194143

variable (x y : ℝ)

theorem factor_expression :
  4 * x ^ 2 - 4 * x - y ^ 2 + 4 * y - 3 = (2 * x + y - 3) * (2 * x - y + 1) := by
  sorry

end factor_expression_l194_194143


namespace points_opposite_sides_line_l194_194250

theorem points_opposite_sides_line (m : ℝ) :
  (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ↔ -7 < m ∧ m < 24 :=
by
  sorry

end points_opposite_sides_line_l194_194250


namespace expression_simplification_l194_194427

theorem expression_simplification : (2^2020 + 2^2018) / (2^2020 - 2^2018) = 5 / 3 := 
by 
    sorry

end expression_simplification_l194_194427


namespace multiplication_result_l194_194653

theorem multiplication_result :
  3^2 * 5^2 * 7 * 11^2 = 190575 :=
by sorry

end multiplication_result_l194_194653


namespace necessarily_positive_l194_194504

theorem necessarily_positive (x y w : ℝ) (h1 : 0 < x ∧ x < 0.5) (h2 : -0.5 < y ∧ y < 0) (h3 : 0.5 < w ∧ w < 1) : 
  0 < w - y :=
sorry

end necessarily_positive_l194_194504


namespace smallest_inverse_defined_l194_194033

theorem smallest_inverse_defined (n : ℤ) : n = 5 :=
by sorry

end smallest_inverse_defined_l194_194033


namespace miles_total_instruments_l194_194961

theorem miles_total_instruments :
  let fingers := 10
  let hands := 2
  let heads := 1
  let trumpets := fingers - 3
  let guitars := hands + 2
  let trombones := heads + 2
  let french_horns := guitars - 1
  (trumpets + guitars + trombones + french_horns) = 17 :=
by
  sorry

end miles_total_instruments_l194_194961


namespace recurring_decimal_to_fraction_l194_194349

theorem recurring_decimal_to_fraction :
  ∃ (frac : ℚ), frac = 1045 / 1998 ∧ 0.5 + (23 / 999) = frac :=
by
  sorry

end recurring_decimal_to_fraction_l194_194349


namespace average_rainfall_correct_l194_194981

-- Define the monthly rainfall
def january_rainfall := 150
def february_rainfall := 200
def july_rainfall := 366
def other_months_rainfall := 100

-- Calculate total yearly rainfall
def total_yearly_rainfall := 
  january_rainfall + 
  february_rainfall + 
  july_rainfall + 
  (9 * other_months_rainfall)

-- Calculate total hours in a year
def days_per_month := 30
def total_days_in_year := 12 * days_per_month
def hours_per_day := 24
def total_hours_in_year := total_days_in_year * hours_per_day

-- Calculate average rainfall per hour
def average_rainfall_per_hour := 
  total_yearly_rainfall / total_hours_in_year

theorem average_rainfall_correct :
  average_rainfall_per_hour = (101 / 540) := sorry

end average_rainfall_correct_l194_194981


namespace proof_problem_l194_194667

-- Define the propositions p and q.
def p (a : ℝ) : Prop := a < -1/2 

def q (a b : ℝ) : Prop := a > b → (1 / (a + 1)) < (1 / (b + 1))

-- Define the final proof problem: proving that "p or q" is true.
theorem proof_problem (a b : ℝ) : (p a) ∨ (q a b) := by
  sorry

end proof_problem_l194_194667


namespace last_digit_base5_89_l194_194160

theorem last_digit_base5_89 (n : ℕ) (h : n = 89) : (n % 5) = 4 :=
by 
  sorry

end last_digit_base5_89_l194_194160


namespace scientific_notation_of_0_0000025_l194_194478

theorem scientific_notation_of_0_0000025 :
  0.0000025 = 2.5 * 10^(-6) :=
by
  sorry

end scientific_notation_of_0_0000025_l194_194478


namespace range_of_a_l194_194192

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_of_a_l194_194192


namespace decrease_in_sales_percentage_l194_194989

theorem decrease_in_sales_percentage (P Q : Real) :
  let P' := 1.40 * P
  let R := P * Q
  let R' := 1.12 * R
  ∃ (D : Real), Q' = Q * (1 - D / 100) ∧ R' = P' * Q' → D = 20 :=
by
  sorry

end decrease_in_sales_percentage_l194_194989


namespace polar_coordinates_of_2_neg2_l194_194562

noncomputable def rect_to_polar_coord (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let theta := if y < 0 
                then 2 * Real.pi - Real.arctan (x / (-y)) 
                else Real.arctan (y / x)
  (r, theta)

theorem polar_coordinates_of_2_neg2 :
  rect_to_polar_coord 2 (-2) = (2 * Real.sqrt 2, 7 * Real.pi / 4) :=
by 
  sorry

end polar_coordinates_of_2_neg2_l194_194562


namespace cow_manure_growth_percentage_l194_194096

variable (control_height bone_meal_growth_percentage cow_manure_height : ℝ)
variable (bone_meal_height : ℝ := bone_meal_growth_percentage * control_height)
variable (percentage_growth : ℝ := (cow_manure_height / bone_meal_height) * 100)

theorem cow_manure_growth_percentage 
  (h₁ : control_height = 36)
  (h₂ : bone_meal_growth_percentage = 1.25)
  (h₃ : cow_manure_height = 90) :
  percentage_growth = 200 :=
by {
  sorry
}

end cow_manure_growth_percentage_l194_194096


namespace dangerous_animals_remaining_in_swamp_l194_194298

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

end dangerous_animals_remaining_in_swamp_l194_194298


namespace alice_meets_bob_at_25_km_l194_194168

-- Define variables for times, speeds, and distances
variables (t : ℕ) (d : ℕ)

-- Conditions
def distance_between_homes := 41
def alice_speed := 5
def bob_speed := 4
def alice_start_time := 1

-- Relating the distances covered by Alice and Bob when they meet
def alice_walk_distance := alice_speed * (t + alice_start_time)
def bob_walk_distance := bob_speed * t
def total_walk_distance := alice_walk_distance + bob_walk_distance

-- Alexander walks 25 kilometers before meeting Bob
theorem alice_meets_bob_at_25_km :
  total_walk_distance = distance_between_homes → alice_walk_distance = 25 :=
by
  sorry

end alice_meets_bob_at_25_km_l194_194168


namespace husband_weekly_saving_l194_194544

variable (H : ℕ)

-- conditions
def weekly_wife : ℕ := 225
def months : ℕ := 6
def weeks_per_month : ℕ := 4
def weeks := months * weeks_per_month
def amount_per_child : ℕ := 1680
def num_children : ℕ := 4

-- total savings calculation
def total_saving : ℕ := weeks * H + weeks * weekly_wife

-- half of total savings divided among children
def half_savings_div_by_children : ℕ := num_children * amount_per_child

-- proof statement
theorem husband_weekly_saving : H = 335 :=
by
  let total_children_saving := half_savings_div_by_children
  have half_saving : ℕ := total_children_saving 
  have total_saving_eq : total_saving = 2 * total_children_saving := sorry
  have total_saving_eq_simplified : weeks * H + weeks * weekly_wife = 13440 := sorry
  have H_eq : H = 335 := sorry
  exact H_eq

end husband_weekly_saving_l194_194544


namespace m_plus_n_sum_l194_194379

theorem m_plus_n_sum :
  let m := 271
  let n := 273
  m + n = 544 :=
by {
  -- sorry included to skip the proof steps
  sorry
}

end m_plus_n_sum_l194_194379


namespace players_taking_physics_l194_194541

-- Definitions based on the conditions
def total_players : ℕ := 30
def players_taking_math : ℕ := 15
def players_taking_both : ℕ := 6

-- The main theorem to prove
theorem players_taking_physics : total_players - players_taking_math + players_taking_both = 21 := by
  sorry

end players_taking_physics_l194_194541


namespace quarters_to_dimes_difference_l194_194495

variable (p : ℝ)

theorem quarters_to_dimes_difference :
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  difference_dimes = 12.5 * p - 15 :=
by
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  sorry

end quarters_to_dimes_difference_l194_194495


namespace smith_boxes_l194_194171

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end smith_boxes_l194_194171


namespace find_m_n_l194_194112

theorem find_m_n : ∃ (m n : ℕ), m^m + (m * n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end find_m_n_l194_194112


namespace n_mult_n_plus_1_eq_square_l194_194201

theorem n_mult_n_plus_1_eq_square (n : ℤ) : (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := 
by sorry

end n_mult_n_plus_1_eq_square_l194_194201


namespace cake_angle_between_adjacent_pieces_l194_194069

theorem cake_angle_between_adjacent_pieces 
  (total_angle : ℝ := 360)
  (total_pieces : ℕ := 10)
  (eaten_pieces : ℕ := 1)
  (angle_per_piece := total_angle / total_pieces)
  (remaining_pieces := total_pieces - eaten_pieces)
  (new_angle_per_piece := total_angle / remaining_pieces) :
  (new_angle_per_piece - angle_per_piece = 4) := 
by
  sorry

end cake_angle_between_adjacent_pieces_l194_194069


namespace loss_of_450_is_negative_450_l194_194912

-- Define the concept of profit and loss based on given conditions.
def profit (x : Int) := x
def loss (x : Int) := -x

-- The mathematical statement:
theorem loss_of_450_is_negative_450 :
  (profit 1000 = 1000) → (loss 450 = -450) :=
by
  intro h
  sorry

end loss_of_450_is_negative_450_l194_194912


namespace determine_dimensions_l194_194132

theorem determine_dimensions (a b : ℕ) (h : a < b) 
    (h1 : ∃ (m n : ℕ), 49 * 51 = (m * a) * (n * b))
    (h2 : ∃ (p q : ℕ), 99 * 101 = (p * a) * (q * b)) : 
    a = 1 ∧ b = 3 :=
  by 
  sorry

end determine_dimensions_l194_194132


namespace quad_factor_value_l194_194997

theorem quad_factor_value (c d : ℕ) (h1 : c + d = 14) (h2 : c * d = 40) (h3 : c > d) : 4 * d - c = 6 :=
sorry

end quad_factor_value_l194_194997


namespace Tim_cookie_packages_l194_194898

theorem Tim_cookie_packages 
    (cookies_in_package : ℕ)
    (packets_in_package : ℕ)
    (min_packet_count : ℕ)
    (h1 : cookies_in_package = 5)
    (h2 : packets_in_package = 7)
    (h3 : min_packet_count = 30) :
  ∃ (cookie_packages : ℕ) (packet_packages : ℕ),
    cookie_packages = 7 ∧ packet_packages = 5 ∧
    cookie_packages * cookies_in_package = packet_packages * packets_in_package ∧
    packet_packages * packets_in_package ≥ min_packet_count :=
by
  sorry

end Tim_cookie_packages_l194_194898


namespace find_a_and_other_root_l194_194249

theorem find_a_and_other_root (a : ℝ) (h : (2 : ℝ) ^ 2 - 3 * (2 : ℝ) + a = 0) :
  a = 2 ∧ ∃ x : ℝ, x ^ 2 - 3 * x + a = 0 ∧ x ≠ 2 ∧ x = 1 := 
by
  sorry

end find_a_and_other_root_l194_194249


namespace complete_square_correct_l194_194679

theorem complete_square_correct (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by 
  intro h
  sorry

end complete_square_correct_l194_194679


namespace find_divisor_l194_194318

theorem find_divisor (Q R D V : ℤ) (hQ : Q = 65) (hR : R = 5) (hV : V = 1565) (hEquation : V = D * Q + R) : D = 24 :=
by
  sorry

end find_divisor_l194_194318


namespace place_value_ratio_56439_2071_l194_194211

theorem place_value_ratio_56439_2071 :
  let n := 56439.2071
  let digit_6_place_value := 1000
  let digit_2_place_value := 0.1
  digit_6_place_value / digit_2_place_value = 10000 :=
by
  sorry

end place_value_ratio_56439_2071_l194_194211


namespace extremum_point_iff_nonnegative_condition_l194_194779

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + x) - (a * x) / (x + 1)

theorem extremum_point_iff (a : ℝ) (h : 0 < a) :
  (∃ (x : ℝ), x = 1 ∧ ∀ (f' : ℝ), f' = (1 + x - a) / (x + 1)^2 ∧ f' = 0) ↔ a = 2 :=
by
  sorry

theorem nonnegative_condition (a : ℝ) (h0 : 0 < a) :
  (∀ (x : ℝ), x ∈ Set.Ici 0 → f a x ≥ 0) ↔ 0 < a ∧ a ≤ 1 :=
by
  sorry

end extremum_point_iff_nonnegative_condition_l194_194779


namespace find_kgs_of_apples_l194_194625

def cost_of_apples_per_kg : ℝ := 2
def num_packs_of_sugar : ℝ := 3
def cost_of_sugar_per_pack : ℝ := cost_of_apples_per_kg - 1
def weight_walnuts_kg : ℝ := 0.5
def cost_of_walnuts_per_kg : ℝ := 6
def cost_of_walnuts : ℝ := cost_of_walnuts_per_kg * weight_walnuts_kg
def total_cost : ℝ := 16

theorem find_kgs_of_apples (A : ℝ) :
  2 * A + (num_packs_of_sugar * cost_of_sugar_per_pack) + cost_of_walnuts = total_cost →
  A = 5 :=
by
  sorry

end find_kgs_of_apples_l194_194625


namespace total_books_for_girls_l194_194365

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l194_194365


namespace cookies_taken_in_four_days_l194_194087

def initial_cookies : ℕ := 70
def cookies_left : ℕ := 28
def days_in_week : ℕ := 7
def days_taken : ℕ := 4
def daily_cookies_taken (total_cookies_taken : ℕ) : ℕ := total_cookies_taken / days_in_week
def total_cookies_taken : ℕ := initial_cookies - cookies_left

theorem cookies_taken_in_four_days :
  daily_cookies_taken total_cookies_taken * days_taken = 24 := by
  sorry

end cookies_taken_in_four_days_l194_194087


namespace molecular_weight_correct_l194_194329

def potassium_weight : ℝ := 39.10
def chromium_weight : ℝ := 51.996
def oxygen_weight : ℝ := 16.00

def num_potassium_atoms : ℕ := 2
def num_chromium_atoms : ℕ := 2
def num_oxygen_atoms : ℕ := 7

def molecular_weight_of_compound : ℝ :=
  (num_potassium_atoms * potassium_weight) +
  (num_chromium_atoms * chromium_weight) +
  (num_oxygen_atoms * oxygen_weight)

theorem molecular_weight_correct :
  molecular_weight_of_compound = 294.192 :=
by
  sorry

end molecular_weight_correct_l194_194329


namespace first_number_percentage_of_second_l194_194362

theorem first_number_percentage_of_second (X : ℝ) (h1 : First = 0.06 * X) (h2 : Second = 0.18 * X) : 
  (First / Second) * 100 = 33.33 := 
by 
  sorry

end first_number_percentage_of_second_l194_194362


namespace total_jumps_correct_l194_194939

-- Define Ronald's jumps
def Ronald_jumps : ℕ := 157

-- Define the difference in jumps between Rupert and Ronald
def difference : ℕ := 86

-- Define Rupert's jumps
def Rupert_jumps : ℕ := Ronald_jumps + difference

-- Define the total number of jumps
def total_jumps : ℕ := Ronald_jumps + Rupert_jumps

-- State the main theorem we want to prove
theorem total_jumps_correct : total_jumps = 400 := 
by sorry

end total_jumps_correct_l194_194939


namespace n_eq_14_l194_194283

variable {a : ℕ → ℕ}  -- the arithmetic sequence
variable {S : ℕ → ℕ}  -- the sum function of the first n terms
variable {d : ℕ}      -- the common difference of the arithmetic sequence

-- Given Conditions
axiom Sn_eq_4 : S 4 = 40
axiom Sn_eq_210 : ∃ (n : ℕ), S n = 210
axiom Sn_minus_4_eq_130 : ∃ (n : ℕ), S (n - 4) = 130

-- Main theorem to prove
theorem n_eq_14 : ∃ (n : ℕ),  S n = 210 ∧ S (n - 4) = 130 ∧ n = 14 :=
by
  sorry

end n_eq_14_l194_194283


namespace expression_value_l194_194909

theorem expression_value
  (x y a b : ℤ)
  (h1 : x = 1)
  (h2 : y = 2)
  (h3 : a + 2 * b = 3) :
  2 * a + 4 * b - 5 = 1 := 
by sorry

end expression_value_l194_194909


namespace log_expression_equals_eight_l194_194121

theorem log_expression_equals_eight :
  (Real.log 4 / Real.log 10) + 
  2 * (Real.log 5 / Real.log 10) + 
  3 * (Real.log 2 / Real.log 10) + 
  6 * (Real.log 5 / Real.log 10) + 
  (Real.log 8 / Real.log 10) = 8 := 
by 
  sorry

end log_expression_equals_eight_l194_194121


namespace find_white_balls_l194_194459

-- Define a structure to hold the probabilities and total balls
structure BallProperties where
  totalBalls : Nat
  probRed : Real
  probBlack : Real

-- Given data as conditions
def givenData : BallProperties := 
  { totalBalls := 50, probRed := 0.15, probBlack := 0.45 }

-- The statement to prove the number of white balls
theorem find_white_balls (data : BallProperties) : 
  data.totalBalls = 50 →
  data.probRed = 0.15 →
  data.probBlack = 0.45 →
  ∃ whiteBalls : Nat, whiteBalls = 20 :=
by
  sorry

end find_white_balls_l194_194459


namespace Tom_age_problem_l194_194542

theorem Tom_age_problem 
  (T : ℝ) 
  (h1 : T = T1 + T2 + T3 + T4) 
  (h2 : T - 3 = 3 * (T - 3 - 3 - 3 - 3)) : 
  T / 3 = 5.5 :=
by 
  -- sorry here to skip the proof
  sorry

end Tom_age_problem_l194_194542


namespace perpendicular_bisector_eqn_l194_194514

-- Definitions based on given conditions
def C₁ (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem perpendicular_bisector_eqn {ρ θ : ℝ} :
  (∃ A B : ℝ × ℝ,
    A ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₁ ρ θ} ∧
    B ∈ {p : ℝ × ℝ | ∃ ρ θ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ C₂ ρ θ}) →
  ρ * Real.sin θ + ρ * Real.cos θ = 1 :=
sorry

end perpendicular_bisector_eqn_l194_194514


namespace problem_statement_l194_194516

theorem problem_statement : (1021 ^ 1022) % 1023 = 4 := 
by
  sorry

end problem_statement_l194_194516


namespace parrot_seeds_consumed_l194_194183

theorem parrot_seeds_consumed (H1 : ∃ T : ℝ, 0.40 * T = 8) : 
  (∃ T : ℝ, 0.40 * T = 8 ∧ 2 * T = 40) :=
sorry

end parrot_seeds_consumed_l194_194183


namespace prime_cubic_solution_l194_194786

theorem prime_cubic_solution :
  ∃ p1 p2 : ℕ, (Nat.Prime p1 ∧ Nat.Prime p2) ∧ p1 ≠ p2 ∧
  (p1^3 + p1^2 - 18*p1 + 26 = 0) ∧ (p2^3 + p2^2 - 18*p2 + 26 = 0) :=
by
  sorry

end prime_cubic_solution_l194_194786


namespace pastries_and_juices_count_l194_194030

theorem pastries_and_juices_count 
  (budget : ℕ) 
  (cost_per_pastry : ℕ) 
  (cost_per_juice : ℕ) 
  (total_money : budget = 50)
  (pastry_cost : cost_per_pastry = 7) 
  (juice_cost : cost_per_juice = 2) : 
  ∃ (p j : ℕ), 7 * p + 2 * j ≤ 50 ∧ p + j = 7 :=
by
  sorry

end pastries_and_juices_count_l194_194030


namespace correct_statement_B_l194_194417

-- Definitions as per the conditions
noncomputable def total_students : ℕ := 6700
noncomputable def selected_students : ℕ := 300

-- Definitions as per the question
def is_population (n : ℕ) : Prop := n = 6700
def is_sample (m n : ℕ) : Prop := m = 300 ∧ n = 6700
def is_individual (m n : ℕ) : Prop := m < n
def is_census (m n : ℕ) : Prop := m = n

-- The statement that needs to be proved
theorem correct_statement_B : 
  is_sample selected_students total_students :=
by
  -- Proof steps would go here
  sorry

end correct_statement_B_l194_194417


namespace find_c_l194_194633

theorem find_c (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : c = 0 :=
by
  sorry

end find_c_l194_194633


namespace max_area_of_triangle_l194_194322

theorem max_area_of_triangle 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a = 2)
  (h2 : 4 * (Real.cos (A / 2))^2 -  Real.cos (2 * (B + C)) = 7 / 2)
  (h3 : A + B + C = Real.pi) :
  (Real.sqrt 3 / 2 * b * c) ≤ Real.sqrt 3 :=
sorry

end max_area_of_triangle_l194_194322


namespace circles_tangent_l194_194060

theorem circles_tangent (m : ℝ) :
  (∀ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9 → 
                (x + 1)^2 + (y - m)^2 = 4 →
                ∃ m, m = -1 ∨ m = -2) := 
sorry

end circles_tangent_l194_194060


namespace percentage_boys_from_school_A_is_20_l194_194474

-- Definitions and conditions based on the problem
def total_boys : ℕ := 200
def non_science_boys_from_A : ℕ := 28
def science_ratio : ℝ := 0.30
def non_science_ratio : ℝ := 1 - science_ratio

-- To prove: The percentage of the total boys that are from school A is 20%
theorem percentage_boys_from_school_A_is_20 :
  ∃ (x : ℝ), x = 20 ∧ 
  (non_science_ratio * (x / 100 * total_boys) = non_science_boys_from_A) := 
sorry

end percentage_boys_from_school_A_is_20_l194_194474


namespace sum_of_four_digit_numbers_l194_194358

theorem sum_of_four_digit_numbers :
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324 :=
by
  let digits := [2, 4, 5, 3]
  let factorial := Nat.factorial (List.length digits)
  let each_appearance := factorial / (List.length digits)
  show (each_appearance * (2 + 4 + 5 + 3) * (1000 + 100 + 10 + 1)) = 93324
  sorry

end sum_of_four_digit_numbers_l194_194358


namespace garden_breadth_l194_194826

theorem garden_breadth (perimeter length breadth : ℕ) 
    (h₁ : perimeter = 680)
    (h₂ : length = 258)
    (h₃ : perimeter = 2 * (length + breadth)) : 
    breadth = 82 := 
sorry

end garden_breadth_l194_194826


namespace inequality_holds_for_all_y_l194_194050

theorem inequality_holds_for_all_y (x : ℝ) :
  (∀ y : ℝ, y^2 - (5^x - 1) * (y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end inequality_holds_for_all_y_l194_194050


namespace find_a_l194_194466

theorem find_a (a : ℝ) (h : Nat.choose 5 2 * (-a)^3 = 10) : a = -1 :=
by
  sorry

end find_a_l194_194466


namespace measure_of_C_angle_maximum_area_triangle_l194_194837

-- Proof Problem 1: Measure of angle C
theorem measure_of_C_angle (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < C ∧ C < Real.pi)
  (m n : ℝ × ℝ)
  (h2 : m = (Real.sin A, Real.sin B))
  (h3 : n = (Real.cos B, Real.cos A))
  (h4 : m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C)) :
  C = 2 * Real.pi / 3 :=
sorry

-- Proof Problem 2: Maximum area of triangle ABC
theorem maximum_area_triangle (A B C : ℝ) (a b c S : ℝ)
  (h1 : c = 2 * Real.sqrt 3)
  (h2 : Real.cos C = -1 / 2)
  (h3 : S = 1 / 2 * a * b * Real.sin (2 * Real.pi / 3)): 
  S ≤ Real.sqrt 3 :=
sorry

end measure_of_C_angle_maximum_area_triangle_l194_194837


namespace map_scale_l194_194275

theorem map_scale (map_distance : ℝ) (time : ℝ) (speed : ℝ) (actual_distance : ℝ) (scale : ℝ) 
  (h1 : map_distance = 5) 
  (h2 : time = 1.5) 
  (h3 : speed = 60) 
  (h4 : actual_distance = speed * time) 
  (h5 : scale = map_distance / actual_distance) : 
  scale = 1 / 18 :=
by 
  sorry

end map_scale_l194_194275


namespace gnome_count_l194_194888

theorem gnome_count (g_R: ℕ) (g_W: ℕ) (h1: g_R = 4 * g_W) (h2: g_W = 20) : g_R - (40 * g_R / 100) = 48 := by
  sorry

end gnome_count_l194_194888


namespace general_term_formula_l194_194575

-- Define the given sequence as a function
def seq (n : ℕ) : ℤ :=
  match n with
  | 0 => 3
  | n + 1 => if (n % 2 = 0) then 4 * (n + 1) - 1 else -(4 * (n + 1) - 1)

-- Define the proposed general term formula
def a_n (n : ℕ) : ℤ :=
  (-1)^(n+1) * (4 * n - 1)

-- State the theorem that general term of the sequence equals the proposed formula
theorem general_term_formula : ∀ n : ℕ, seq n = a_n n := 
by
  sorry

end general_term_formula_l194_194575


namespace probability_of_symmetry_line_l194_194270

-- Define the conditions of the problem.
def is_on_symmetry_line (P Q : (ℤ × ℤ)) :=
  (Q.fst = P.fst) ∨ (Q.snd = P.snd) ∨ (Q.fst - P.fst = Q.snd - P.snd) ∨ (Q.fst - P.fst = P.snd - Q.snd)

-- Define the main statement of the theorem to be proved.
theorem probability_of_symmetry_line :
  let grid_size := 11
  let total_points := grid_size * grid_size
  let center : (ℤ × ℤ) := (grid_size / 2, grid_size / 2)
  let other_points := total_points - 1
  let symmetric_points := 40
  /- Here we need to calculate the probability, which is the ratio of symmetric points to other points,
     and this should equal 1/3 -/
  (symmetric_points : ℚ) / other_points = 1 / 3 :=
by sorry

end probability_of_symmetry_line_l194_194270


namespace fraction_meaningful_condition_l194_194281

theorem fraction_meaningful_condition (x : ℝ) : (4 / (x + 2) ≠ 0) ↔ (x ≠ -2) := 
by 
  sorry

end fraction_meaningful_condition_l194_194281


namespace pete_bus_ride_blocks_l194_194046

theorem pete_bus_ride_blocks : 
  ∀ (total_walk_blocks bus_blocks total_blocks : ℕ), 
  total_walk_blocks = 10 → 
  total_blocks = 50 → 
  total_walk_blocks + 2 * bus_blocks = total_blocks → 
  bus_blocks = 20 :=
by
  intros total_walk_blocks bus_blocks total_blocks h1 h2 h3
  sorry

end pete_bus_ride_blocks_l194_194046


namespace problem_product_of_areas_eq_3600x6_l194_194041

theorem problem_product_of_areas_eq_3600x6 
  (x : ℝ) 
  (bottom_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ)
  (bottom_area_eq : bottom_area = 12 * x ^ 2)
  (side_area_eq : side_area = 15 * x ^ 2)
  (front_area_eq : front_area = 20 * x ^ 2)
  (dimensions_proportional : ∃ a b c : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x 
                            ∧ bottom_area = a * b ∧ side_area = a * c ∧ front_area = b * c)
  : bottom_area * side_area * front_area = 3600 * x ^ 6 :=
by 
  -- Proof omitted
  sorry

end problem_product_of_areas_eq_3600x6_l194_194041


namespace largest_n_for_two_digit_quotient_l194_194251

-- Lean statement for the given problem.
theorem largest_n_for_two_digit_quotient (n : ℕ) (h₀ : 0 ≤ n) (h₃ : n ≤ 9) :
  (10 ≤ (n * 100 + 5) / 5 ∧ (n * 100 + 5) / 5 < 100) ↔ n = 4 :=
by sorry

end largest_n_for_two_digit_quotient_l194_194251


namespace find_N_l194_194860

theorem find_N
  (N : ℕ)
  (h : (4 / 10 : ℝ) * (16 / (16 + N : ℝ)) + (6 / 10 : ℝ) * (N / (16 + N : ℝ)) = 0.58) :
  N = 144 :=
sorry

end find_N_l194_194860


namespace x_plus_p_eq_2p_plus_2_l194_194296

-- Define the conditions and the statement to be proved
theorem x_plus_p_eq_2p_plus_2 (x p : ℝ) (h1 : x > 2) (h2 : |x - 2| = p) : x + p = 2 * p + 2 :=
by
  -- Proof goes here
  sorry

end x_plus_p_eq_2p_plus_2_l194_194296


namespace max_slope_avoiding_lattice_points_l194_194423

theorem max_slope_avoiding_lattice_points :
  ∃ a : ℝ, (1 < a ∧ ∀ m : ℝ, (1 < m ∧ m < a) → (∀ x : ℤ, (10 < x ∧ x ≤ 200) → ∃ k : ℝ, y = m * x + 5 ∧ (m * x + 5 ≠ k))) ∧ a = 101 / 100 :=
sorry

end max_slope_avoiding_lattice_points_l194_194423


namespace problem_equivalence_l194_194290

theorem problem_equivalence (n : ℕ) (H₁ : 2 * 2006 = 1) (H₂ : ∀ n : ℕ, (2 * n + 2) * 2006 = 3 * (2 * n * 2006)) :
  2008 * 2006 = 3 ^ 1003 :=
by
  sorry

end problem_equivalence_l194_194290


namespace mike_picked_12_pears_l194_194584

theorem mike_picked_12_pears (k_picked k_gave_away k_m_together k_left m_left : ℕ) 
  (hkp : k_picked = 47) 
  (hkg : k_gave_away = 46) 
  (hkt : k_m_together = 13)
  (hkl : k_left = k_picked - k_gave_away) 
  (hlt : k_m_left = k_left + m_left) : 
  m_left = 12 := by
  sorry

end mike_picked_12_pears_l194_194584


namespace max_tiles_to_spell_CMWMC_l194_194528

theorem max_tiles_to_spell_CMWMC {Cs Ms Ws : ℕ} (hC : Cs = 8) (hM : Ms = 8) (hW : Ws = 8) : 
  ∃ (max_draws : ℕ), max_draws = 18 :=
by
  -- Assuming we have 8 C's, 8 M's, and 8 W's in the bag
  sorry

end max_tiles_to_spell_CMWMC_l194_194528


namespace milk_for_6_cookies_l194_194468

/-- Given conditions for baking cookies -/
def quarts_to_pints : ℕ := 2 -- 2 pints in a quart
def milk_for_24_cookies : ℕ := 5 -- 5 quarts of milk for 24 cookies

/-- Theorem to prove the number of pints needed to bake 6 cookies -/
theorem milk_for_6_cookies : 
  (milk_for_24_cookies * quarts_to_pints * 6 / 24 : ℝ) = 2.5 := 
by 
  sorry -- Proof is omitted

end milk_for_6_cookies_l194_194468


namespace cyclists_meet_at_starting_point_l194_194062

/--
Given a circular track of length 1200 meters, and three cyclists with speeds of 36 kmph, 54 kmph, and 72 kmph,
prove that all three cyclists will meet at the starting point for the first time after 4 minutes.
-/
theorem cyclists_meet_at_starting_point :
  let track_length := 1200
  let speed_a_kmph := 36
  let speed_b_kmph := 54
  let speed_c_kmph := 72
  
  let speed_a_m_per_min := speed_a_kmph * 1000 / 60
  let speed_b_m_per_min := speed_b_kmph * 1000 / 60
  let speed_c_m_per_min := speed_c_kmph * 1000 / 60
  
  let time_a := track_length / speed_a_m_per_min
  let time_b := track_length / speed_b_m_per_min
  let time_c := track_length / speed_c_m_per_min
  
  let lcm := (2 : ℚ)

  (time_a = 2) ∧ (time_b = 4 / 3) ∧ (time_c = 1) → 
  ∀ t, t = lcm * 3 → t = 12 / 3 → t = 4 :=
by
  sorry

end cyclists_meet_at_starting_point_l194_194062


namespace woody_saving_weeks_l194_194429

variable (cost_needed current_savings weekly_allowance : ℕ)

theorem woody_saving_weeks (h₁ : cost_needed = 282)
                           (h₂ : current_savings = 42)
                           (h₃ : weekly_allowance = 24) :
  (cost_needed - current_savings) / weekly_allowance = 10 := by
  sorry

end woody_saving_weeks_l194_194429


namespace find_n_mod_11_l194_194825

theorem find_n_mod_11 : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 50000 [MOD 11] ∧ n = 5 :=
sorry

end find_n_mod_11_l194_194825


namespace ramu_profit_percent_l194_194448

noncomputable def carCost : ℝ := 42000
noncomputable def repairCost : ℝ := 13000
noncomputable def sellingPrice : ℝ := 60900
noncomputable def totalCost : ℝ := carCost + repairCost
noncomputable def profit : ℝ := sellingPrice - totalCost
noncomputable def profitPercent : ℝ := (profit / totalCost) * 100

theorem ramu_profit_percent : profitPercent = 10.73 := 
by
  sorry

end ramu_profit_percent_l194_194448


namespace length_of_one_pencil_l194_194400

theorem length_of_one_pencil (l : ℕ) (h1 : 2 * l = 24) : l = 12 :=
by {
  sorry
}

end length_of_one_pencil_l194_194400


namespace path_area_and_cost_l194_194857

theorem path_area_and_cost:
  let length_grass_field := 75
  let width_grass_field := 55
  let path_width := 3.5
  let cost_per_sq_meter := 2
  let length_with_path := length_grass_field + 2 * path_width
  let width_with_path := width_grass_field + 2 * path_width
  let area_with_path := length_with_path * width_with_path
  let area_grass_field := length_grass_field * width_grass_field
  let area_path := area_with_path - area_grass_field
  let cost_of_construction := area_path * cost_per_sq_meter
  area_path = 959 ∧ cost_of_construction = 1918 :=
by
  sorry

end path_area_and_cost_l194_194857


namespace total_fish_l194_194479

theorem total_fish (fish_lilly fish_rosy : ℕ) (hl : fish_lilly = 10) (hr : fish_rosy = 14) :
  fish_lilly + fish_rosy = 24 := 
by 
  sorry

end total_fish_l194_194479


namespace triangle_largest_angle_l194_194580

theorem triangle_largest_angle (A B C : ℚ) (sinA sinB sinC : ℚ) 
(h_ratio : sinA / sinB = 3 / 5)
(h_ratio2 : sinB / sinC = 5 / 7)
(h_sum : A + B + C = 180) : C = 120 := 
sorry

end triangle_largest_angle_l194_194580


namespace old_conveyor_time_l194_194396

theorem old_conveyor_time (x : ℝ) : 
  (1 / x) + (1 / 15) = 1 / 8.75 → 
  x = 21 := 
by 
  intro h 
  sorry

end old_conveyor_time_l194_194396


namespace negation_of_p_l194_194813

variable (x : ℝ)

-- Define the original proposition p
def p := ∀ x, x^2 < 1 → x < 1

-- Define the negation of p
def neg_p := ∃ x₀, x₀^2 ≥ 1 ∧ x₀ < 1

-- State the theorem that negates p
theorem negation_of_p : ¬ p ↔ neg_p :=
by
  sorry

end negation_of_p_l194_194813


namespace gross_pay_calculation_l194_194032

theorem gross_pay_calculation
    (NetPay : ℕ) (Taxes : ℕ) (GrossPay : ℕ) 
    (h1 : NetPay = 315) 
    (h2 : Taxes = 135) 
    (h3 : GrossPay = NetPay + Taxes) : 
    GrossPay = 450 :=
by
    -- We need to prove this part
    sorry

end gross_pay_calculation_l194_194032


namespace interval_after_speed_limit_l194_194520

noncomputable def car_speed_before : ℝ := 80 -- speed before the sign in km/h
noncomputable def car_speed_after : ℝ := 60 -- speed after the sign in km/h
noncomputable def initial_interval : ℕ := 10 -- interval between the cars in meters

-- Convert speeds from km/h to m/s
noncomputable def v : ℝ := car_speed_before * 1000 / 3600
noncomputable def u : ℝ := car_speed_after * 1000 / 3600

-- Given the initial interval and speed before the sign, calculate the time it takes for the second car to reach the sign
noncomputable def delta_t : ℝ := initial_interval / v

-- Given u and delta_t, calculate the new interval after slowing down
noncomputable def new_interval : ℝ := u * delta_t

-- Theorem statement in Lean
theorem interval_after_speed_limit : new_interval = 7.5 :=
sorry

end interval_after_speed_limit_l194_194520


namespace triangle_side_lengths_exist_l194_194548

theorem triangle_side_lengths_exist 
  (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  ∃ (x y z : ℝ), 
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ 
  (a = y + z) ∧ (b = x + z) ∧ (c = x + y) :=
by
  let x := (a - b + c) / 2
  let y := (a + b - c) / 2
  let z := (-a + b + c) / 2
  have hx : x > 0 := sorry
  have hy : y > 0 := sorry
  have hz : z > 0 := sorry
  have ha : a = y + z := sorry
  have hb : b = x + z := sorry
  have hc : c = x + y := sorry
  exact ⟨x, y, z, hx, hy, hz, ha, hb, hc⟩

end triangle_side_lengths_exist_l194_194548


namespace no_value_of_a_l194_194844

theorem no_value_of_a (a : ℝ) (x y : ℝ) : ¬∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1^2 + y^2 + 2 * x1 = abs (x1 - a) - 1) ∧ (x2^2 + y^2 + 2 * x2 = abs (x2 - a) - 1) := 
by
  sorry

end no_value_of_a_l194_194844


namespace Oates_reunion_l194_194209

-- Declare the conditions as variables
variables (total_guests both_reunions yellow_reunion : ℕ)
variables (H1 : total_guests = 100)
variables (H2 : both_reunions = 7)
variables (H3 : yellow_reunion = 65)

-- The proof problem statement
theorem Oates_reunion (O : ℕ) (H4 : total_guests = O + yellow_reunion - both_reunions) : O = 42 :=
sorry

end Oates_reunion_l194_194209


namespace determine_angle_A_l194_194259

noncomputable section

open Real

-- Definition of an acute triangle and its sides
variables {A B : ℝ} {a b : ℝ}

-- Additional conditions that are given before providing the theorem
variables (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
          (h5 : 2 * a * sin B = sqrt 3 * b)

-- Theorem statement
theorem determine_angle_A (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2)
  (h5 : 2 * a * sin B = sqrt 3 * b) : A = π / 3 :=
sorry

end determine_angle_A_l194_194259


namespace systematic_sampling_first_group_l194_194072

theorem systematic_sampling_first_group
  (a : ℕ → ℕ)
  (d : ℕ)
  (n : ℕ)
  (a₁ : ℕ)
  (a₁₆ : ℕ)
  (h₁ : d = 8)
  (h₂ : a 16 = a₁₆)
  (h₃ : a₁₆ = 125)
  (h₄ : a n = a₁ + (n - 1) * d) :
  a 1 = 5 :=
by
  sorry

end systematic_sampling_first_group_l194_194072


namespace park_will_have_9_oak_trees_l194_194901

def current_oak_trees : Nat := 5
def additional_oak_trees : Nat := 4
def total_oak_trees : Nat := current_oak_trees + additional_oak_trees

theorem park_will_have_9_oak_trees : total_oak_trees = 9 :=
by
  sorry

end park_will_have_9_oak_trees_l194_194901


namespace acute_triangle_orthocenter_l194_194496

variables (A B C H : Point) (a b c h_a h_b h_c : Real)

def acute_triangle (α β γ : Point) : Prop := 
-- Definition that ensures triangle αβγ is acute
sorry

def orthocenter (α β γ ω : Point) : Prop := 
-- Definition that ω is the orthocenter of triangle αβγ 
sorry

def sides_of_triangle (α β γ : Point) : (Real × Real × Real) := 
-- Function that returns the side lengths of triangle αβγ as (a, b, c)
sorry

def altitudes_of_triangle (α β γ θ : Point) : (Real × Real × Real) := 
-- Function that returns the altitudes of triangle αβγ with orthocenter θ as (h_a, h_b, h_c)
sorry

theorem acute_triangle_orthocenter 
  (A B C H : Point)
  (a b c h_a h_b h_c : Real)
  (ht : acute_triangle A B C)
  (orth : orthocenter A B C H)
  (sides : sides_of_triangle A B C = (a, b, c))
  (alts : altitudes_of_triangle A B C H = (h_a, h_b, h_c)) :
  AH * h_a + BH * h_b + CH * h_c = (a^2 + b^2 + c^2) / 2 :=
by sorry


end acute_triangle_orthocenter_l194_194496


namespace monotonic_solution_l194_194157

-- Definition of a monotonic function
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- The main theorem
theorem monotonic_solution (f : ℝ → ℝ) 
  (mon : monotonic f) 
  (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = -x) :=
sorry

end monotonic_solution_l194_194157


namespace barrel_tank_ratio_l194_194611

theorem barrel_tank_ratio
  (B T : ℝ)
  (h1 : (3 / 4) * B = (5 / 8) * T) :
  B / T = 5 / 6 :=
sorry

end barrel_tank_ratio_l194_194611


namespace fill_up_minivans_l194_194649

theorem fill_up_minivans (service_cost : ℝ) (fuel_cost_per_liter : ℝ) (total_cost : ℝ)
  (mini_van_liters : ℝ) (truck_percent_bigger : ℝ) (num_trucks : ℕ) (num_minivans : ℕ) :
  service_cost = 2.3 ∧ fuel_cost_per_liter = 0.7 ∧ total_cost = 396 ∧
  mini_van_liters = 65 ∧ truck_percent_bigger = 1.2 ∧ num_trucks = 2 →
  num_minivans = 4 :=
by
  sorry

end fill_up_minivans_l194_194649


namespace geometric_sequence_general_term_l194_194681

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2:ℝ)^(n-1)

theorem geometric_sequence_general_term : 
  ∀ (n : ℕ), 
  (∀ (n : ℕ), 0 < a_n n) ∧ a_n 1 = 1 ∧ (a_n 1 + a_n 2 + a_n 3 = 7) → 
  a_n n = 2^(n-1) :=
by
  sorry

end geometric_sequence_general_term_l194_194681


namespace percentage_of_500_l194_194122

theorem percentage_of_500 : (110 * 500) / 100 = 550 :=
by
  sorry

end percentage_of_500_l194_194122


namespace bells_toll_together_l194_194129

noncomputable def LCM (a b : Nat) : Nat := (a * b) / (Nat.gcd a b)

theorem bells_toll_together :
  let intervals := [2, 4, 6, 8, 10, 12]
  let lcm := intervals.foldl LCM 1
  lcm = 120 →
  let duration := 30 * 60 -- 1800 seconds
  let tolls := duration / lcm
  tolls + 1 = 16 :=
by
  sorry

end bells_toll_together_l194_194129


namespace base_six_to_ten_2154_l194_194406

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  2 * 6^3 + 1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_six_to_ten_2154 :
  convert_base_six_to_ten 2154 = 502 :=
by
  sorry

end base_six_to_ten_2154_l194_194406


namespace determine_remainder_l194_194140

-- Define the sequence and its sum
def geom_series_sum_mod (a r n m : ℕ) : ℕ := 
  ((r^(n+1) - 1) / (r - 1)) % m

-- Define the specific geometric series and modulo
theorem determine_remainder :
  geom_series_sum_mod 1 11 1800 500 = 1 :=
by
  -- Using geom_series_sum_mod to define the series
  let S := geom_series_sum_mod 1 11 1800 500
  -- Remainder when the series is divided by 500
  show S = 1
  sorry

end determine_remainder_l194_194140


namespace alpha_beta_square_inequality_l194_194686

theorem alpha_beta_square_inequality
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 :=
by
  sorry

end alpha_beta_square_inequality_l194_194686


namespace g_at_5_l194_194855

-- Define the function g(x) that satisfies the given condition
def g (x : ℝ) : ℝ := sorry

-- Axiom stating that the function g satisfies the given equation for all x ∈ ℝ
axiom g_condition : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2

-- The theorem to prove
theorem g_at_5 : g 5 = -66 / 7 :=
by
  -- Proof will be added here.
  sorry

end g_at_5_l194_194855


namespace quadratic_eq_real_roots_roots_diff_l194_194116

theorem quadratic_eq_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ 
  (x^2 + (m-2)*x - m = 0) ∧
  (y^2 + (m-2)*y - m = 0) := sorry

theorem roots_diff (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0)
  (h_roots : (m^2 + (m-2)*m - m = 0) ∧ (n^2 + (m-2)*n - m = 0)) :
  m - n = 5/2 := sorry

end quadratic_eq_real_roots_roots_diff_l194_194116


namespace pentagon_quadrilateral_sum_of_angles_l194_194812

   theorem pentagon_quadrilateral_sum_of_angles
     (exterior_angle_pentagon : ℕ := 72)
     (interior_angle_pentagon : ℕ := 108)
     (sum_interior_angles_quadrilateral : ℕ := 360)
     (reflex_angle : ℕ := 252) :
     (sum_interior_angles_quadrilateral - reflex_angle = interior_angle_pentagon) :=
   by
     sorry
   
end pentagon_quadrilateral_sum_of_angles_l194_194812


namespace PQ_sum_l194_194452

theorem PQ_sum (P Q : ℕ) (h1 : 5 / 7 = P / 63) (h2 : 5 / 7 = 70 / Q) : P + Q = 143 :=
by
  sorry

end PQ_sum_l194_194452


namespace max_value_sqrt_sum_l194_194884

theorem max_value_sqrt_sum {x y z : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  ∃ (M : ℝ), M = (Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x))) ∧ M = Real.sqrt 2 + 1 :=
by sorry

end max_value_sqrt_sum_l194_194884


namespace remainder_is_3_l194_194835

-- Define the polynomial p(x)
def p (x : ℝ) := x^3 - 3 * x + 5

-- Define the divisor d(x)
def d (x : ℝ) := x - 1

-- The theorem: remainder when p(x) is divided by d(x)
theorem remainder_is_3 : p 1 = 3 := by 
  sorry

end remainder_is_3_l194_194835


namespace number_of_three_digit_numbers_is_48_l194_194626

-- Define the problem: the cards and their constraints
def card1 := (1, 2)
def card2 := (3, 4)
def card3 := (5, 6)

-- The condition given is that 6 cannot be used as 9

-- Define the function to compute the number of different three-digit numbers
def number_of_three_digit_numbers : Nat := 6 * 4 * 2

/- Prove that the number of different three-digit numbers that can be formed is 48 -/
theorem number_of_three_digit_numbers_is_48 : number_of_three_digit_numbers = 48 :=
by
  -- We skip the proof here
  sorry

end number_of_three_digit_numbers_is_48_l194_194626


namespace vector_decomposition_unique_l194_194776

variable {m : ℝ}
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (m - 1, m + 3)

theorem vector_decomposition_unique (m : ℝ) : (m + 3 ≠ 2 * (m - 1)) ↔ (m ≠ 5) := 
sorry

end vector_decomposition_unique_l194_194776


namespace barbara_wins_gameA_l194_194732

noncomputable def gameA_winning_strategy : Prop :=
∃ (has_winning_strategy : (ℤ → ℝ) → Prop),
  has_winning_strategy (fun n => n : ℤ → ℝ)

theorem barbara_wins_gameA :
  gameA_winning_strategy := sorry

end barbara_wins_gameA_l194_194732


namespace least_amount_of_money_l194_194224

variable (money : String → ℝ)
variable (Bo Coe Flo Jo Moe Zoe : String)

theorem least_amount_of_money :
  (money Bo ≠ money Coe) ∧ (money Bo ≠ money Flo) ∧ (money Bo ≠ money Jo) ∧ (money Bo ≠ money Moe) ∧ (money Bo ≠ money Zoe) ∧ 
  (money Coe ≠ money Flo) ∧ (money Coe ≠ money Jo) ∧ (money Coe ≠ money Moe) ∧ (money Coe ≠ money Zoe) ∧ 
  (money Flo ≠ money Jo) ∧ (money Flo ≠ money Moe) ∧ (money Flo ≠ money Zoe) ∧ 
  (money Jo ≠ money Moe) ∧ (money Jo ≠ money Zoe) ∧ 
  (money Moe ≠ money Zoe) ∧ 
  (money Flo > money Jo) ∧ (money Flo > money Bo) ∧
  (money Bo > money Moe) ∧ (money Coe > money Moe) ∧ 
  (money Jo > money Moe) ∧ (money Jo < money Bo) ∧ 
  (money Zoe > money Jo) ∧ (money Zoe < money Coe) →
  money Moe < money Bo ∧ money Moe < money Coe ∧ money Moe < money Flo ∧ money Moe < money Jo ∧ money Moe < money Zoe := 
sorry

end least_amount_of_money_l194_194224


namespace betty_oranges_l194_194093

-- Define the givens and result as Lean definitions and theorems
theorem betty_oranges (kg_apples : ℕ) (cost_apples_per_kg cost_oranges_per_kg total_cost_oranges num_oranges : ℕ) 
    (h1 : kg_apples = 3)
    (h2 : cost_apples_per_kg = 2)
    (h3 : cost_apples_per_kg * 2 = cost_oranges_per_kg)
    (h4 : 12 = total_cost_oranges)
    (h5 : total_cost_oranges / cost_oranges_per_kg = num_oranges) :
    num_oranges = 3 :=
sorry

end betty_oranges_l194_194093


namespace distance_difference_l194_194384

theorem distance_difference (t : ℕ) (speed_alice speed_bob : ℕ) :
  speed_alice = 15 → speed_bob = 10 → t = 6 → (speed_alice * t) - (speed_bob * t) = 30 :=
by
  intros h1 h2 h3
  sorry

end distance_difference_l194_194384


namespace equation_represents_3x_minus_7_equals_2x_plus_5_l194_194778

theorem equation_represents_3x_minus_7_equals_2x_plus_5 (x : ℝ) :
  (3 * x - 7 = 2 * x + 5) :=
sorry

end equation_represents_3x_minus_7_equals_2x_plus_5_l194_194778


namespace mark_remaining_money_l194_194397

theorem mark_remaining_money 
  (initial_money : ℕ) (num_books : ℕ) (cost_per_book : ℕ) (total_cost : ℕ) (remaining_money : ℕ) 
  (H1 : initial_money = 85)
  (H2 : num_books = 10)
  (H3 : cost_per_book = 5)
  (H4 : total_cost = num_books * cost_per_book)
  (H5 : remaining_money = initial_money - total_cost) : 
  remaining_money = 35 := 
by
  sorry

end mark_remaining_money_l194_194397


namespace pet_store_earnings_l194_194248

theorem pet_store_earnings :
  let kitten_price := 6
  let puppy_price := 5
  let kittens_sold := 2
  let puppies_sold := 1 
  let total_earnings := kittens_sold * kitten_price + puppies_sold * puppy_price
  total_earnings = 17 :=
by
  sorry

end pet_store_earnings_l194_194248


namespace proof_problem_l194_194588

-- Definitions needed for conditions
def a := -7 / 4
def b := -2 / 3
def m : ℚ := 1  -- m can be any rational number
def n : ℚ := -m  -- since m and n are opposite numbers

-- Lean statement to prove the given problem
theorem proof_problem : 4 * a / b + 3 * (m + n) = 21 / 2 := by
  -- Definitions ensuring a, b, m, n meet the conditions
  have habs : |a| = 7 / 4 := by sorry
  have brecip : 1 / b = -3 / 2 := by sorry
  have moppos : m + n = 0 := by sorry
  sorry

end proof_problem_l194_194588


namespace social_event_handshakes_l194_194781

def handshake_count (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) : ℕ :=
  let introductions_handshakes := group_b * (group_b - 1) / 2
  let direct_handshakes := group_b * (group_a - 1)
  introductions_handshakes + direct_handshakes

theorem social_event_handshakes :
  handshake_count 40 25 15 = 465 := by
  sorry

end social_event_handshakes_l194_194781


namespace marble_problem_l194_194498

theorem marble_problem {r b : ℕ} 
  (h1 : 9 * r - b = 27) 
  (h2 : 3 * r - b = 3) : r + b = 13 := 
by
  sorry

end marble_problem_l194_194498


namespace product_of_coordinates_of_intersection_l194_194785

-- Conditions: Defining the equations of the two circles
def circle1_eq (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 25 = 0
def circle2_eq (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 37 = 0

-- Translated problem to prove the question equals the correct answer
theorem product_of_coordinates_of_intersection :
  ∃ (x y : ℝ), circle1_eq x y ∧ circle2_eq x y ∧ x * y = 10 :=
sorry

end product_of_coordinates_of_intersection_l194_194785


namespace complex_number_powers_l194_194133

theorem complex_number_powers (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 :=
sorry

end complex_number_powers_l194_194133


namespace find_f_l194_194359

theorem find_f (q f : ℕ) (h_digit_q : q ≤ 9) (h_digit_f : f ≤ 9)
  (h_distinct : q ≠ f) 
  (h_div_by_36 : (457 * 1000 + q * 100 + 89 * 10 + f) % 36 = 0)
  (h_sum_3 : q + f = 3) :
  f = 2 :=
sorry

end find_f_l194_194359


namespace dad_steps_l194_194405

theorem dad_steps (dad_steps_ratio: ℕ) (masha_steps_ratio: ℕ) (masha_steps: ℕ)
  (masha_and_yasha_steps: ℕ) (total_steps: ℕ)
  (h1: dad_steps_ratio * 3 = masha_steps_ratio * 5)
  (h2: masha_steps * 3 = masha_and_yasha_steps * 5)
  (h3: masha_and_yasha_steps = total_steps)
  (h4: total_steps = 400) :
  dad_steps_ratio * 30 = 90 :=
by
  sorry

end dad_steps_l194_194405


namespace arrangement_count_l194_194184

/-- April has five different basil plants and five different tomato plants. --/
def basil_plants : ℕ := 5
def tomato_plants : ℕ := 5

/-- All tomato plants must be placed next to each other. --/
def tomatoes_next_to_each_other := true

/-- The row must start with a basil plant. --/
def starts_with_basil := true

/-- The number of ways to arrange the plants in a row under the given conditions is 11520. --/
theorem arrangement_count :
  basil_plants = 5 ∧ tomato_plants = 5 ∧ tomatoes_next_to_each_other ∧ starts_with_basil → 
  ∃ arrangements : ℕ, arrangements = 11520 :=
by 
  sorry

end arrangement_count_l194_194184


namespace simplify_frac_l194_194852

theorem simplify_frac :
  (1 / (1 / (Real.sqrt 3 + 2) + 2 / (Real.sqrt 5 - 2))) = 
  (Real.sqrt 3 - 2 * Real.sqrt 5 - 2) :=
by
  sorry

end simplify_frac_l194_194852


namespace cos_double_angle_sum_l194_194787

theorem cos_double_angle_sum (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (h : Real.sin (α + π/6) = 3/5) : 
  Real.cos (2*α + π/12) = 31 / 50 * Real.sqrt 2 := sorry

end cos_double_angle_sum_l194_194787


namespace solve_inequality_l194_194361

theorem solve_inequality :
  { x : ℝ | (x - 5) / (x - 3)^2 < 0 } = { x : ℝ | x < 3 } ∪ { x : ℝ | 3 < x ∧ x < 5 } :=
by
  sorry

end solve_inequality_l194_194361


namespace combined_instruments_correct_l194_194923

-- Definitions of initial conditions
def Charlie_flutes : Nat := 1
def Charlie_horns : Nat := 2
def Charlie_harps : Nat := 1
def Carli_flutes : Nat := 2 * Charlie_flutes
def Carli_horns : Nat := Charlie_horns / 2
def Carli_harps : Nat := 0

-- Calculation of total instruments
def Charlie_total_instruments : Nat := Charlie_flutes + Charlie_horns + Charlie_harps
def Carli_total_instruments : Nat := Carli_flutes + Carli_horns + Carli_harps
def combined_total_instruments : Nat := Charlie_total_instruments + Carli_total_instruments

-- Theorem statement
theorem combined_instruments_correct : combined_total_instruments = 7 := 
by
  sorry

end combined_instruments_correct_l194_194923


namespace theo_cookie_price_l194_194175

theorem theo_cookie_price :
  (∃ (dough_amount total_earnings per_cookie_earnings_carla per_cookie_earnings_theo : ℕ) 
     (cookies_carla cookies_theo : ℝ), 
  dough_amount = 120 ∧ 
  cookies_carla = 20 ∧ 
  per_cookie_earnings_carla = 50 ∧ 
  cookies_theo = 15 ∧ 
  total_earnings = cookies_carla * per_cookie_earnings_carla ∧ 
  per_cookie_earnings_theo = total_earnings / cookies_theo ∧ 
  per_cookie_earnings_theo = 67) :=
sorry

end theo_cookie_price_l194_194175


namespace relationship_between_a_and_b_l194_194440

-- Define the objects and their relationships
noncomputable def α_parallel_β : Prop := sorry
noncomputable def a_parallel_α : Prop := sorry
noncomputable def b_perpendicular_β : Prop := sorry

-- Define the relationship we want to prove
noncomputable def a_perpendicular_b : Prop := sorry

-- The statement we want to prove
theorem relationship_between_a_and_b (h1 : α_parallel_β) (h2 : a_parallel_α) (h3 : b_perpendicular_β) : a_perpendicular_b :=
sorry

end relationship_between_a_and_b_l194_194440


namespace range_of_function_l194_194315

theorem range_of_function : ∀ (y : ℝ), (0 < y ∧ y ≤ 1 / 2) ↔ ∃ (x : ℝ), y = 1 / (x^2 + 2) := 
by
  sorry

end range_of_function_l194_194315


namespace f_monotone_f_inequality_solution_l194_194538

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → ∃ y, f y = x
axiom f_at_2: f 2 = 1
axiom f_mul : ∀ x y, f (x * y) = f x + f y
axiom f_positive : ∀ x, x > 1 → f x > 0

theorem f_monotone (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) : x₁ < x₂ → f x₁ < f x₂ :=
sorry

theorem f_inequality_solution (x : ℝ) (hx : x > 2 ∧ x ≤ 4) : f x + f (x - 2) ≤ 3 :=
sorry

end f_monotone_f_inequality_solution_l194_194538


namespace mike_typing_time_l194_194616

-- Definitions based on the given conditions
def original_speed : ℕ := 65
def speed_reduction : ℕ := 20
def document_words : ℕ := 810
def reduced_speed : ℕ := original_speed - speed_reduction

-- The statement to prove
theorem mike_typing_time : (document_words / reduced_speed) = 18 :=
  by
    sorry

end mike_typing_time_l194_194616


namespace find_length_of_room_l194_194795

theorem find_length_of_room (width area_existing area_needed : ℕ) (h_width : width = 15) (h_area_existing : area_existing = 16) (h_area_needed : area_needed = 149) :
  (area_existing + area_needed) / width = 11 :=
by
  sorry

end find_length_of_room_l194_194795


namespace additional_miles_proof_l194_194777

-- Define the distances
def distance_to_bakery : ℕ := 9
def distance_bakery_to_grandma : ℕ := 24
def distance_grandma_to_apartment : ℕ := 27

-- Define the total distances
def total_distance_with_bakery : ℕ := distance_to_bakery + distance_bakery_to_grandma + distance_grandma_to_apartment
def total_distance_without_bakery : ℕ := 2 * distance_grandma_to_apartment

-- Define the additional miles
def additional_miles_with_bakery : ℕ := total_distance_with_bakery - total_distance_without_bakery

-- Theorem statement
theorem additional_miles_proof : additional_miles_with_bakery = 6 :=
by {
  -- Here should be the proof, but we insert sorry to indicate it's skipped
  sorry
}

end additional_miles_proof_l194_194777


namespace solution_value_a_l194_194048

theorem solution_value_a (x a : ℝ) (h₁ : x = 2) (h₂ : 2 * x + a = 3) : a = -1 :=
by
  -- Proof goes here
  sorry

end solution_value_a_l194_194048


namespace distance_between_points_l194_194717

theorem distance_between_points : abs (3 - (-2)) = 5 := 
by
  sorry

end distance_between_points_l194_194717


namespace probability_exactly_one_solves_problem_l194_194557

-- Define the context in which A and B solve the problem with given probabilities.
variables (p1 p2 : ℝ)

-- Define the constraint that the probabilities are between 0 and 1
axiom prob_A_nonneg : 0 ≤ p1
axiom prob_A_le_one : p1 ≤ 1
axiom prob_B_nonneg : 0 ≤ p2
axiom prob_B_le_one : p2 ≤ 1

-- Define the context that A and B solve the problem independently.
axiom A_and_B_independent : true

-- The theorem statement to prove the desired probability of exactly one solving the problem.
theorem probability_exactly_one_solves_problem : (p1 * (1 - p2) + p2 * (1 - p1)) =  p1 * (1 - p2) + p2 * (1 - p1) :=
by
  sorry

end probability_exactly_one_solves_problem_l194_194557


namespace mabel_marble_ratio_l194_194148

variable (A K M : ℕ)

-- Conditions
def condition1 : Prop := A + 12 = 2 * K
def condition2 : Prop := M = 85
def condition3 : Prop := M = A + 63

-- The main statement to prove
theorem mabel_marble_ratio (h1 : condition1 A K) (h2 : condition2 M) (h3 : condition3 A M) : M / K = 5 :=
by
  sorry

end mabel_marble_ratio_l194_194148


namespace lowest_possible_sale_price_is_30_percent_l194_194476

noncomputable def list_price : ℝ := 80
noncomputable def discount_factor : ℝ := 0.5
noncomputable def additional_discount : ℝ := 0.2
noncomputable def lowest_price := (list_price - discount_factor * list_price) - additional_discount * list_price
noncomputable def percent_of_list_price := (lowest_price / list_price) * 100

theorem lowest_possible_sale_price_is_30_percent :
  percent_of_list_price = 30 := by
  sorry

end lowest_possible_sale_price_is_30_percent_l194_194476


namespace range_of_a_if_f_has_three_zeros_l194_194097

def f (a x : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_if_f_has_three_zeros (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, f a x1 = 0 ∧ f a x2 = 0 ∧ f a x3 = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) ↔ a < -3 := 
by
  sorry

end range_of_a_if_f_has_three_zeros_l194_194097


namespace radius_increase_l194_194153

/-- Proving that the radius increases by 7/π inches when the circumference increases from 50 inches to 64 inches -/
theorem radius_increase (C₁ C₂ : ℝ) (h₁ : C₁ = 50) (h₂ : C₂ = 64) :
  (C₂ / (2 * Real.pi) - C₁ / (2 * Real.pi)) = 7 / Real.pi :=
by
  sorry

end radius_increase_l194_194153


namespace Sara_team_wins_l194_194657

theorem Sara_team_wins (total_games losses wins : ℕ) (h1 : total_games = 12) (h2 : losses = 4) (h3 : wins = total_games - losses) :
  wins = 8 :=
by
  sorry

end Sara_team_wins_l194_194657


namespace focal_distance_of_ellipse_l194_194343

theorem focal_distance_of_ellipse :
  ∀ (x y : ℝ), (x^2 / 16) + (y^2 / 9) = 1 → (2 * Real.sqrt 7) = 2 * Real.sqrt 7 :=
by
  intros x y hxy
  sorry

end focal_distance_of_ellipse_l194_194343


namespace average_age_l194_194294

theorem average_age (Jared Molly Hakimi : ℕ) (h1 : Jared = Hakimi + 10) (h2 : Molly = 30) (h3 : Hakimi = 40) :
  (Jared + Molly + Hakimi) / 3 = 40 :=
by
  sorry

end average_age_l194_194294


namespace actual_cost_of_article_l194_194904

noncomputable def article_actual_cost (x : ℝ) : Prop :=
  (0.58 * x = 1050) → x = 1810.34

theorem actual_cost_of_article : ∃ x : ℝ, article_actual_cost x :=
by
  use 1810.34
  sorry

end actual_cost_of_article_l194_194904


namespace rearrange_possible_l194_194136

theorem rearrange_possible (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (f : ℕ → ℕ), (∀ i < n, f i + 1 < n → (f (i + 1) - f i = 3 ∨ f (i + 1) - f i = 5)) :=
  sorry

end rearrange_possible_l194_194136


namespace find_point_on_x_axis_l194_194449

theorem find_point_on_x_axis (a : ℝ) (h : abs (3 * a + 6) = 30) : (a = -12) ∨ (a = 8) :=
sorry

end find_point_on_x_axis_l194_194449


namespace max_participants_won_at_least_three_matches_l194_194463

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end max_participants_won_at_least_three_matches_l194_194463


namespace fraction_absent_l194_194222

theorem fraction_absent (p : ℕ) (x : ℚ) (h : (W / p) * 1.2 = W / (p * (1 - x))) : x = 1 / 6 :=
by
  sorry

end fraction_absent_l194_194222


namespace problems_left_to_grade_l194_194864

-- Defining all the conditions
def worksheets_total : ℕ := 14
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 2

-- Stating the proof problem
theorem problems_left_to_grade : 
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 14 := 
by
  sorry

end problems_left_to_grade_l194_194864


namespace sum_of_tens_and_units_digit_of_7_pow_2023_l194_194262

theorem sum_of_tens_and_units_digit_of_7_pow_2023 :
  let n := 7 ^ 2023
  (n % 100).div 10 + (n % 10) = 16 :=
by
  sorry

end sum_of_tens_and_units_digit_of_7_pow_2023_l194_194262


namespace total_votes_cast_l194_194414

-- Problem statement and conditions
variable (V : ℝ) (candidateVotes : ℝ) (rivalVotes : ℝ)
variable (h1 : candidateVotes = 0.35 * V)
variable (h2 : rivalVotes = candidateVotes + 1350)

-- Target to prove
theorem total_votes_cast : V = 4500 := by
  -- pseudo code proof would be filled here in real Lean environment
  sorry

end total_votes_cast_l194_194414


namespace solution_set_of_quadratic_inequality_l194_194999

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} :=
sorry

end solution_set_of_quadratic_inequality_l194_194999


namespace sum_of_abs_values_l194_194733

-- Define the problem conditions
variable (a b c d m : ℤ)
variable (h1 : a + b + c + d = 1)
variable (h2 : a * b + a * c + a * d + b * c + b * d + c * d = 0)
variable (h3 : a * b * c + a * b * d + a * c * d + b * c * d = -4023)
variable (h4 : a * b * c * d = m)

-- Prove the required sum of absolute values
theorem sum_of_abs_values : |a| + |b| + |c| + |d| = 621 :=
by
  sorry

end sum_of_abs_values_l194_194733


namespace remainder_1234567_div_145_l194_194491

theorem remainder_1234567_div_145 : 1234567 % 145 = 67 := by
  sorry

end remainder_1234567_div_145_l194_194491


namespace Katie_cupcakes_l194_194348

theorem Katie_cupcakes (initial_cupcakes sold_cupcakes final_cupcakes : ℕ) (h1 : initial_cupcakes = 26) (h2 : sold_cupcakes = 20) (h3 : final_cupcakes = 26) :
  (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20 :=
by
  sorry

end Katie_cupcakes_l194_194348


namespace hands_in_class_not_including_peters_l194_194829

def total_students : ℕ := 11
def hands_per_student : ℕ := 2
def peter_hands : ℕ := 2

theorem hands_in_class_not_including_peters :  (total_students * hands_per_student) - peter_hands = 20 :=
by
  sorry

end hands_in_class_not_including_peters_l194_194829


namespace max_value_of_p_l194_194883

theorem max_value_of_p
  (p q r s : ℕ)
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90)
  (h5 : 0 < s)
  (h6 : 0 < r)
  (h7 : 0 < q)
  (h8 : 0 < p):
  p ≤ 5324 :=
by
  sorry

end max_value_of_p_l194_194883


namespace boat_speed_in_still_water_l194_194064

variable (b r d v t : ℝ)

theorem boat_speed_in_still_water (hr : r = 3) 
                                 (hd : d = 3.6) 
                                 (ht : t = 1/5) 
                                 (hv : v = b + r) 
                                 (dist_eq : d = v * t) : 
  b = 15 := 
by
  sorry

end boat_speed_in_still_water_l194_194064


namespace initial_total_fish_l194_194218

def total_days (weeks : ℕ) : ℕ := weeks * 7
def fish_added (rate : ℕ) (days : ℕ) : ℕ := rate * days
def initial_fish (final_count : ℕ) (added : ℕ) : ℕ := final_count - added

theorem initial_total_fish {final_goldfish final_koi rate_goldfish rate_koi days init_goldfish init_koi : ℕ}
    (h_final_goldfish : final_goldfish = 200)
    (h_final_koi : final_koi = 227)
    (h_rate_goldfish : rate_goldfish = 5)
    (h_rate_koi : rate_koi = 2)
    (h_days : days = total_days 3)
    (h_init_goldfish : init_goldfish = initial_fish final_goldfish (fish_added rate_goldfish days))
    (h_init_koi : init_koi = initial_fish final_koi (fish_added rate_koi days)) :
    init_goldfish + init_koi = 280 :=
by
    sorry -- skipping the proof

end initial_total_fish_l194_194218


namespace five_b_value_l194_194670

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := 
by
  sorry

end five_b_value_l194_194670


namespace ratio_of_boxes_sold_l194_194690

-- Definitions for conditions
variables (T W Tu : ℕ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  W = 2 * T ∧
  Tu = 2 * W ∧
  T = 1200

-- The statement to prove the ratio Tu / W = 2
theorem ratio_of_boxes_sold (T W Tu : ℕ) (h : conditions T W Tu) :
  Tu / W = 2 :=
by
  sorry

end ratio_of_boxes_sold_l194_194690


namespace intersection_M_N_l194_194130

-- Definitions of the sets M and N based on the conditions
def M (x : ℝ) : Prop := ∃ (y : ℝ), y = Real.log (x^2 - 3*x - 4)
def N (y : ℝ) : Prop := ∃ (x : ℝ), y = 2^(x - 1)

-- The proof statement
theorem intersection_M_N : { x : ℝ | M x } ∩ { x : ℝ | ∃ y : ℝ, N y ∧ y = Real.log (x^2 - 3*x - 4) } = { x : ℝ | x > 4 } :=
by
  sorry

end intersection_M_N_l194_194130


namespace magnitude_of_a_plus_b_l194_194705

open Real

noncomputable def magnitude (x y : ℝ) : ℝ :=
  sqrt (x^2 + y^2)

theorem magnitude_of_a_plus_b (m : ℝ) (a b : ℝ × ℝ)
  (h₁ : a = (m+2, 1))
  (h₂ : b = (1, -2*m))
  (h₃ : (a.1 * b.1 + a.2 * b.2 = 0)) :
  magnitude (a.1 + b.1) (a.2 + b.2) = sqrt 34 :=
by
  sorry

end magnitude_of_a_plus_b_l194_194705


namespace production_average_l194_194089

-- Define the conditions and question
theorem production_average (n : ℕ) (P : ℕ) (P_new : ℕ) (h1 : P = n * 70) (h2 : P_new = P + 90) (h3 : P_new = (n + 1) * 75) : n = 3 := 
by sorry

end production_average_l194_194089


namespace max_value_of_function_l194_194150

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ m, ∀ y, y = 4 * x * (3 - 2 * x) → m = 9 / 2 :=
sorry

end max_value_of_function_l194_194150


namespace express_q_as_polynomial_l194_194070

def q (x : ℝ) : ℝ := x^3 + 4

theorem express_q_as_polynomial (x : ℝ) : 
  q x + (2 * x^6 + x^5 + 4 * x^4 + 6 * x^2) = (5 * x^4 + 10 * x^3 - x^2 + 8 * x + 15) → 
  q x = -2 * x^6 - x^5 + x^4 + 10 * x^3 - 7 * x^2 + 8 * x + 15 := by
  sorry

end express_q_as_polynomial_l194_194070


namespace different_genre_pairs_count_l194_194269

theorem different_genre_pairs_count 
  (mystery_books : Finset ℕ)
  (fantasy_books : Finset ℕ)
  (biographies : Finset ℕ)
  (h1 : mystery_books.card = 4)
  (h2 : fantasy_books.card = 4)
  (h3 : biographies.card = 4) :
  (mystery_books.product (fantasy_books ∪ biographies)).card +
  (fantasy_books.product (mystery_books ∪ biographies)).card +
  (biographies.product (mystery_books ∪ fantasy_books)).card = 48 := 
sorry

end different_genre_pairs_count_l194_194269


namespace find_multiple_l194_194788

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_ineq : (m * n - 15) > 2 * n) : m = 6 := 
by {
  sorry
}

end find_multiple_l194_194788


namespace message_spread_in_24_hours_l194_194533

theorem message_spread_in_24_hours : ∃ T : ℕ, (T = (2^25 - 1)) :=
by 
  let T := 2^24 - 1
  use T
  sorry

end message_spread_in_24_hours_l194_194533


namespace rebecca_tent_stakes_l194_194736

-- Given conditions
variable (x : ℕ) -- number of tent stakes

axiom h1 : x + 3 * x + (x + 2) = 22 -- Total number of items equals 22

-- Proof objective
theorem rebecca_tent_stakes : x = 4 :=
by 
  -- Place for the proof. Using sorry to indicate it.
  sorry

end rebecca_tent_stakes_l194_194736


namespace differences_l194_194497

def seq (n : ℕ) : ℕ := n^2 + 1

def first_diff (n : ℕ) : ℕ := (seq (n + 1)) - (seq n)

def second_diff (n : ℕ) : ℕ := (first_diff (n + 1)) - (first_diff n)

def third_diff (n : ℕ) : ℕ := (second_diff (n + 1)) - (second_diff n)

theorem differences (n : ℕ) : first_diff n = 2 * n + 1 ∧ 
                             second_diff n = 2 ∧ 
                             third_diff n = 0 := by 
  sorry

end differences_l194_194497


namespace d_in_N_l194_194550

def M := {x : ℤ | ∃ n : ℤ, x = 3 * n}
def N := {x : ℤ | ∃ n : ℤ, x = 3 * n + 1}
def P := {x : ℤ | ∃ n : ℤ, x = 3 * n - 1}

theorem d_in_N (a b c d : ℤ) (ha : a ∈ M) (hb : b ∈ N) (hc : c ∈ P) (hd : d = a - b + c) : d ∈ N :=
by sorry

end d_in_N_l194_194550


namespace trig_identity_theorem_l194_194716

noncomputable def trig_identity_proof : Prop :=
  (1 + Real.cos (Real.pi / 9)) * 
  (1 + Real.cos (2 * Real.pi / 9)) * 
  (1 + Real.cos (4 * Real.pi / 9)) * 
  (1 + Real.cos (5 * Real.pi / 9)) = 
  (1 / 2) * (Real.sin (Real.pi / 9))^4

#check trig_identity_proof

theorem trig_identity_theorem : trig_identity_proof := by
  sorry

end trig_identity_theorem_l194_194716


namespace largest_unsatisfiable_group_l194_194940

theorem largest_unsatisfiable_group :
  ∃ n : ℕ, (∀ a b c : ℕ, n ≠ 6 * a + 9 * b + 20 * c) ∧ (∀ m : ℕ, m > n → ∃ a b c : ℕ, m = 6 * a + 9 * b + 20 * c) ∧ n = 43 :=
by
  sorry

end largest_unsatisfiable_group_l194_194940


namespace first_two_digits_of_52x_l194_194654

-- Define the digit values that would make 52x divisible by 6.
def digit_values (x : Nat) : Prop :=
  x = 2 ∨ x = 5 ∨ x = 8

-- The main theorem to prove the first two digits are 52 given the conditions.
theorem first_two_digits_of_52x (x : Nat) (h : digit_values x) : (52 * 10 + x) / 10 = 52 :=
by sorry

end first_two_digits_of_52x_l194_194654


namespace rainfall_ratio_l194_194695

theorem rainfall_ratio (R_1 R_2 : ℕ) (h1 : R_1 + R_2 = 25) (h2 : R_2 = 15) : R_2 / R_1 = 3 / 2 :=
by
  sorry

end rainfall_ratio_l194_194695


namespace not_divisor_of_44_l194_194638

theorem not_divisor_of_44 (m j : ℤ) (H1 : m = j * (j + 1) * (j + 2) * (j + 3))
  (H2 : 11 ∣ m) : ¬ (∀ j : ℤ, 44 ∣ j * (j + 1) * (j + 2) * (j + 3)) :=
by
  sorry

end not_divisor_of_44_l194_194638


namespace colleen_pencils_l194_194938

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (extra_cost : ℕ) (colleen_paid : ℕ)
  (H1 : joy_pencils = 30)
  (H2 : pencil_cost = 4)
  (H3 : extra_cost = 80)
  (H4 : colleen_paid = (joy_pencils * pencil_cost) + extra_cost) :
  colleen_paid / pencil_cost = 50 := 
by 
  -- Hints, if necessary
sorry

end colleen_pencils_l194_194938


namespace proof_of_inequality_l194_194219

theorem proof_of_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 :=
sorry

end proof_of_inequality_l194_194219


namespace min_value_of_expression_l194_194280

open Classical

theorem min_value_of_expression (x : ℝ) (hx : x > 0) : 
  ∃ y, x + 16 / (x + 1) = y ∧ ∀ z, (z > 0 → z + 16 / (z + 1) ≥ y) := 
by
  use 7
  sorry

end min_value_of_expression_l194_194280


namespace largest_c_in_range_l194_194996

theorem largest_c_in_range (c : ℝ) (h : ∃ x : ℝ,  2 * x ^ 2 - 4 * x + c = 5) : c ≤ 7 :=
by sorry

end largest_c_in_range_l194_194996


namespace problem_solution_l194_194158

theorem problem_solution :
  3 ^ (0 ^ (2 ^ 2)) + ((3 ^ 1) ^ 0) ^ 2 = 2 :=
by
  sorry

end problem_solution_l194_194158


namespace yellow_balls_in_bag_l194_194752

theorem yellow_balls_in_bag (r y : ℕ) (P : ℚ) 
  (h1 : r = 10) 
  (h2 : P = 2 / 7) 
  (h3 : P = r / (r + y)) : 
  y = 25 := 
sorry

end yellow_balls_in_bag_l194_194752


namespace milkman_pure_milk_l194_194159

theorem milkman_pure_milk (x : ℝ) 
  (h_cost : 3.60 * x = 3 * (x + 5)) : x = 25 :=
  sorry

end milkman_pure_milk_l194_194159


namespace second_digging_breadth_l194_194482

theorem second_digging_breadth :
  ∀ (A B depth1 length1 breadth1 depth2 length2 : ℕ),
  (A / B) = 1 → -- Assuming equal number of days and people
  depth1 = 100 → length1 = 25 → breadth1 = 30 → 
  depth2 = 75 → length2 = 20 → 
  (A = depth1 * length1 * breadth1) → 
  (B = depth2 * length2 * x) →
  x = 50 :=
by sorry

end second_digging_breadth_l194_194482


namespace minimum_value_of_m_plus_n_l194_194564

noncomputable def m (a b : ℝ) : ℝ := b + (1 / a)
noncomputable def n (a b : ℝ) : ℝ := a + (1 / b)

theorem minimum_value_of_m_plus_n (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 1) :
  m a b + n a b = 4 :=
sorry

end minimum_value_of_m_plus_n_l194_194564


namespace convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l194_194727

noncomputable def pi_deg : ℝ := 180 -- Define pi in degrees
notation "°" => pi_deg -- Define a notation for degrees

theorem convert_radian_to_degree_part1 : (π / 12) * (180 / π) = 15 := 
by
  sorry

theorem convert_radian_to_degree_part2 : (13 * π / 6) * (180 / π) = 390 := 
by
  sorry

theorem convert_radian_to_degree_part3 : -(5 / 12) * π * (180 / π) = -75 := 
by
  sorry

theorem convert_degree_to_radian_part1 : 36 * (π / 180) = (π / 5) := 
by
  sorry

theorem convert_degree_to_radian_part2 : -105 * (π / 180) = -(7 * π / 12) := 
by
  sorry

end convert_radian_to_degree_part1_convert_radian_to_degree_part2_convert_radian_to_degree_part3_convert_degree_to_radian_part1_convert_degree_to_radian_part2_l194_194727


namespace keith_attended_games_l194_194320

-- Definitions based on the given conditions
def total_games : ℕ := 8
def missed_games : ℕ := 4

-- The proof goal: Keith's attendance
def attended_games : ℕ := total_games - missed_games

-- Main statement to prove the total games Keith attended
theorem keith_attended_games : attended_games = 4 := by
  -- Sorry is a placeholder for the proof
  sorry

end keith_attended_games_l194_194320


namespace quadratic_discriminant_l194_194919

theorem quadratic_discriminant {a b c : ℝ} (h : (a + b + c) * c ≤ 0) : b^2 ≥ 4 * a * c :=
sorry

end quadratic_discriminant_l194_194919


namespace trains_cross_time_l194_194850

theorem trains_cross_time 
  (len_train1 len_train2 : ℕ) 
  (speed_train1_kmph speed_train2_kmph : ℕ) 
  (len_train1_eq : len_train1 = 200) 
  (len_train2_eq : len_train2 = 300) 
  (speed_train1_eq : speed_train1_kmph = 70) 
  (speed_train2_eq : speed_train2_kmph = 50) 
  : (500 / (120 * 1000 / 3600)) = 15 := 
by sorry

end trains_cross_time_l194_194850


namespace coefficient_of_y_squared_l194_194306

/-- Given the equation ay^2 - 8y + 55 = 59 and y = 2, prove that the coefficient a is 5. -/
theorem coefficient_of_y_squared (a y : ℝ) (h_y : y = 2) (h_eq : a * y^2 - 8 * y + 55 = 59) : a = 5 := by
  sorry

end coefficient_of_y_squared_l194_194306


namespace supply_lasts_for_8_months_l194_194404

-- Define the conditions
def pills_per_supply : ℕ := 120
def days_per_pill : ℕ := 2
def days_per_month : ℕ := 30

-- Define the function to calculate the duration in days
def supply_duration_in_days (pills : ℕ) (days_per_pill : ℕ) : ℕ :=
  pills * days_per_pill

-- Define the function to convert days to months
def days_to_months (days : ℕ) (days_per_month : ℕ) : ℕ :=
  days / days_per_month

-- Main statement to prove
theorem supply_lasts_for_8_months :
  days_to_months (supply_duration_in_days pills_per_supply days_per_pill) days_per_month = 8 :=
by
  sorry

end supply_lasts_for_8_months_l194_194404


namespace find_three_digit_number_l194_194900

theorem find_three_digit_number : 
  ∀ (c d e : ℕ), 0 ≤ c ∧ c < 10 ∧ 0 ≤ d ∧ d < 10 ∧ 0 ≤ e ∧ e < 10 ∧ 
  (10 * c + d) / 99 + (100 * c + 10 * d + e) / 999 = 44 / 99 → 
  100 * c + 10 * d + e = 400 :=
by {
  sorry
}

end find_three_digit_number_l194_194900


namespace marcus_percentage_of_team_points_l194_194754

theorem marcus_percentage_of_team_points
  (three_point_goals : ℕ)
  (two_point_goals : ℕ)
  (team_points : ℕ)
  (h1 : three_point_goals = 5)
  (h2 : two_point_goals = 10)
  (h3 : team_points = 70) :
  ((three_point_goals * 3 + two_point_goals * 2) / team_points : ℚ) * 100 = 50 := by
sorry

end marcus_percentage_of_team_points_l194_194754


namespace meeting_point_l194_194356

/-- Along a straight alley with 400 streetlights placed at equal intervals, numbered consecutively from 1 to 400,
    Alla and Boris set out towards each other from opposite ends of the alley with different constant speeds.
    Alla starts at streetlight number 1 and Boris starts at streetlight number 400. When Alla is at the 55th streetlight,
    Boris is at the 321st streetlight. The goal is to prove that they will meet at the 163rd streetlight.
-/
theorem meeting_point (n : ℕ) (h1 : n = 400) (h2 : ∀ i j k l : ℕ, i = 55 → j = 321 → k = 1 → l = 400) : 
  ∃ m, m = 163 := 
by
  sorry

end meeting_point_l194_194356


namespace six_times_more_coats_l194_194205

/-- The number of lab coats is 6 times the number of uniforms. --/
def coats_per_uniforms (c u : ℕ) : Prop := c = 6 * u

/-- There are 12 uniforms. --/
def uniforms : ℕ := 12

/-- Each lab tech gets 14 coats and uniforms in total. --/
def total_per_tech : ℕ := 14

/-- Show that the number of lab coats is 6 times the number of uniforms. --/
theorem six_times_more_coats (c u : ℕ) (h1 : coats_per_uniforms c u) (h2 : u = 12) :
  c / u = 6 :=
by
  sorry

end six_times_more_coats_l194_194205


namespace C_alone_work_days_l194_194107

theorem C_alone_work_days (A_work_days B_work_days combined_work_days : ℝ) 
  (A_work_rate B_work_rate C_work_rate combined_work_rate : ℝ)
  (hA : A_work_days = 6)
  (hB : B_work_days = 5)
  (hCombined : combined_work_days = 2)
  (hA_work_rate : A_work_rate = 1 / A_work_days)
  (hB_work_rate : B_work_rate = 1 / B_work_days)
  (hCombined_work_rate : combined_work_rate = 1 / combined_work_days)
  (work_rate_eq : A_work_rate + B_work_rate + C_work_rate = combined_work_rate):
  (1 / C_work_rate) = 7.5 :=
by
  sorry

end C_alone_work_days_l194_194107


namespace range_of_a_l194_194387

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a - 1)*x - 1 < 0
def r (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1 ∨ a > 3

theorem range_of_a (a : ℝ) (h : (p a ∨ q a) ∧ ¬(p a ∧ q a)) : r a := 
by sorry

end range_of_a_l194_194387


namespace infinite_representable_and_nonrepresentable_terms_l194_194967

def a (n : ℕ) : ℕ :=
  2^n + 2^(n / 2)

def is_representable (k : ℕ) : Prop :=   
  -- A nonnegative integer is defined to be representable if it can
  -- be expressed as a sum of distinct terms from the sequence a(n).
  sorry  -- Definition will depend on the specific notion of representability

theorem infinite_representable_and_nonrepresentable_terms :
  (∃ᶠ n in at_top, is_representable (a n)) ∧ (∃ᶠ n in at_top, ¬is_representable (a n)) :=
sorry  -- This is the main theorem claiming infinitely many representable and non-representable terms.

end infinite_representable_and_nonrepresentable_terms_l194_194967


namespace stratified_sampling_BA3_count_l194_194637

-- Defining the problem parameters
def num_Om_BA1 : ℕ := 60
def num_Om_BA2 : ℕ := 20
def num_Om_BA3 : ℕ := 40
def total_sample_size : ℕ := 30

-- Proving using stratified sampling
theorem stratified_sampling_BA3_count : 
  (total_sample_size * num_Om_BA3 / (num_Om_BA1 + num_Om_BA2 + num_Om_BA3)) = 10 :=
by
  -- Since Lean doesn't handle reals and integers simplistically,
  -- we need to translate the division and multiplication properly.
  sorry

end stratified_sampling_BA3_count_l194_194637


namespace mikails_age_l194_194337

-- Define the conditions
def dollars_per_year_old : ℕ := 5
def total_dollars_given : ℕ := 45

-- Main theorem statement
theorem mikails_age (age : ℕ) : (age * dollars_per_year_old = total_dollars_given) → age = 9 :=
by
  sorry

end mikails_age_l194_194337


namespace problem_part1_problem_part2_l194_194806

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log a
noncomputable def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := 2 * Real.log (2 * x + t) / Real.log a

theorem problem_part1 (a t : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  f a 1 - g a t 1 = 0 → t = -2 + Real.sqrt 2 :=
sorry

theorem problem_part2 (a t : ℝ) (ha_bound : 0 < a ∧ a < 1) :
  (∀ x, 0 ≤ x ∧ x ≤ 15 → f a x ≥ g a t x) → t ≥ 1 :=
sorry

end problem_part1_problem_part2_l194_194806


namespace intersection_with_y_axis_l194_194937

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end intersection_with_y_axis_l194_194937


namespace distinct_remainders_l194_194409

theorem distinct_remainders
  (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n)
  (h_div : n ∣ a^n - 1) :
  ∀ i j : ℕ, i ∈ (Finset.range n).image (· + 1) →
            j ∈ (Finset.range n).image (· + 1) →
            (a^i + i) % n = (a^j + j) % n →
            i = j :=
by
  intros i j hi hj h
  sorry

end distinct_remainders_l194_194409


namespace intersecting_lines_l194_194246

theorem intersecting_lines {c d : ℝ} 
  (h₁ : 12 = 2 * 4 + c) 
  (h₂ : 12 = -4 + d) : 
  c + d = 20 := 
sorry

end intersecting_lines_l194_194246


namespace find_number_l194_194381

theorem find_number : ∃ x : ℝ, (6 * ((x / 8 + 8) - 30) = 12) ∧ x = 192 :=
by sorry

end find_number_l194_194381


namespace elena_pen_cost_l194_194022

theorem elena_pen_cost (cost_X : ℝ) (cost_Y : ℝ) (total_pens : ℕ) (brand_X_pens : ℕ) 
    (purchased_X_cost : cost_X = 4.0) (purchased_Y_cost : cost_Y = 2.8)
    (total_pens_condition : total_pens = 12) (brand_X_pens_condition : brand_X_pens = 8) :
    cost_X * brand_X_pens + cost_Y * (total_pens - brand_X_pens) = 43.20 :=
    sorry

end elena_pen_cost_l194_194022


namespace no_integer_roots_l194_194360

theorem no_integer_roots (x : ℤ) : ¬ (x^3 - 5 * x^2 - 11 * x + 35 = 0) := 
sorry

end no_integer_roots_l194_194360


namespace full_price_tickets_count_l194_194718

def num_tickets_reduced := 5400
def total_tickets := 25200
def num_tickets_full := 5 * num_tickets_reduced

theorem full_price_tickets_count :
  num_tickets_reduced + num_tickets_full = total_tickets → num_tickets_full = 27000 :=
by
  sorry

end full_price_tickets_count_l194_194718


namespace part1_part2_l194_194118

def f (x : ℝ) : ℝ := abs (x + 3) + abs (x - 2)

theorem part1 : {x : ℝ | f x > 7} = {x : ℝ | x < -4} ∪ {x : ℝ | x > 3} :=
by
  sorry

theorem part2 (m : ℝ) (h : m > 1) : ∃ x : ℝ, f x = (4 / (m - 1)) + m :=
by
  sorry

end part1_part2_l194_194118


namespace range_of_a_l194_194395

theorem range_of_a (a : ℝ) (H : ∀ x : ℝ, x ≤ 1 → 4 - a * 2^x > 0) : a < 2 :=
sorry

end range_of_a_l194_194395


namespace oldest_child_age_l194_194665

open Nat

def avg_age (a b c d : ℕ) := (a + b + c + d) / 4

theorem oldest_child_age 
  (h_avg : avg_age 5 8 11 x = 9) : x = 12 :=
by
  sorry

end oldest_child_age_l194_194665


namespace tan_45_deg_l194_194373

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l194_194373


namespace sum_of_bases_l194_194086

theorem sum_of_bases (R_1 R_2 : ℕ) 
  (hF1 : (4 * R_1 + 8) / (R_1 ^ 2 - 1) = (3 * R_2 + 6) / (R_2 ^ 2 - 1))
  (hF2 : (8 * R_1 + 4) / (R_1 ^ 2 - 1) = (6 * R_2 + 3) / (R_2 ^ 2 - 1)) : 
  R_1 + R_2 = 21 :=
sorry

end sum_of_bases_l194_194086


namespace bus_speed_express_mode_l194_194748

theorem bus_speed_express_mode (L : ℝ) (t_red : ℝ) (speed_increase : ℝ) (x : ℝ) (normal_speed : ℝ) :
  L = 16 ∧ t_red = 1 / 15 ∧ speed_increase = 8 ∧ normal_speed = x - 8 ∧ 
  (16 / normal_speed - 16 / x = 1 / 15) → x = 48 :=
by
  sorry

end bus_speed_express_mode_l194_194748


namespace largest_even_integer_of_product_2880_l194_194252

theorem largest_even_integer_of_product_2880 :
  ∃ n : ℤ, (n-2) * n * (n+2) = 2880 ∧ n + 2 = 22 := 
by {
  sorry
}

end largest_even_integer_of_product_2880_l194_194252


namespace age_difference_ratio_l194_194595

theorem age_difference_ratio (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R + 2 = 2 * (J + 2))
  (h3 : (R + 2) * (K + 2) = 192) :
  (R - J) / (R - K) = 2 := by
  sorry

end age_difference_ratio_l194_194595


namespace total_weight_of_7_moles_CaO_l194_194890

/-- Definitions necessary for the problem --/
def atomic_weight_Ca : ℝ := 40.08 -- atomic weight of calcium in g/mol
def atomic_weight_O : ℝ := 16.00 -- atomic weight of oxygen in g/mol
def molecular_weight_CaO : ℝ := atomic_weight_Ca + atomic_weight_O -- molecular weight of CaO in g/mol
def number_of_moles_CaO : ℝ := 7 -- number of moles of CaO

/-- The main theorem statement --/
theorem total_weight_of_7_moles_CaO :
  molecular_weight_CaO * number_of_moles_CaO = 392.56 :=
by
  sorry

end total_weight_of_7_moles_CaO_l194_194890


namespace number_properties_l194_194571

theorem number_properties (a b x : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 375) 
  (h3 : a - b = x) : 
  (a = 25 ∧ b = 15 ∧ x = 10) ∨ (a = 15 ∧ b = 25 ∧ x = 10) :=
by
  sorry

end number_properties_l194_194571


namespace intersection_A_B_l194_194612

def setA : Set ℝ := { x | |x| < 2 }
def setB : Set ℝ := { x | x^2 - 4 * x + 3 < 0 }
def setC : Set ℝ := { x | 1 < x ∧ x < 2 }

theorem intersection_A_B : setA ∩ setB = setC := by
  sorry

end intersection_A_B_l194_194612


namespace min_rows_required_to_seat_students_l194_194875

-- Definitions based on the conditions
def seats_per_row : ℕ := 168
def total_students : ℕ := 2016
def max_students_per_school : ℕ := 40

def min_number_of_rows : ℕ :=
  -- Given that the minimum number of rows required to seat all students following the conditions is 15
  15

-- Lean statement expressing the proof problem
theorem min_rows_required_to_seat_students :
  ∃ rows : ℕ, rows = min_number_of_rows ∧
  (∀ school_sizes : List ℕ, (∀ size ∈ school_sizes, size ≤ max_students_per_school)
    → (List.sum school_sizes = total_students)
    → ∀ school_arrangement : List (List ℕ), 
        (∀ row_sizes ∈ school_arrangement, List.sum row_sizes ≤ seats_per_row) 
        → List.length school_arrangement ≤ rows) :=
sorry

end min_rows_required_to_seat_students_l194_194875


namespace non_adjacent_arrangement_l194_194675

-- Define the number of people
def numPeople : ℕ := 8

-- Define the number of specific people who must not be adjacent
def numSpecialPeople : ℕ := 3

-- Define the number of general people who are not part of the specific group
def numGeneralPeople : ℕ := numPeople - numSpecialPeople

-- Permutations calculation for general people
def permuteGeneralPeople : ℕ := Nat.factorial numGeneralPeople

-- Number of gaps available after arranging general people
def numGaps : ℕ := numGeneralPeople + 1

-- Permutations calculation for special people placed in the gaps
def permuteSpecialPeople : ℕ := Nat.descFactorial numGaps numSpecialPeople

-- Total permutations
def totalPermutations : ℕ := permuteSpecialPeople * permuteGeneralPeople

theorem non_adjacent_arrangement :
  totalPermutations = Nat.descFactorial 6 3 * Nat.factorial 5 := by
  sorry

end non_adjacent_arrangement_l194_194675


namespace total_filled_water_balloons_l194_194092

theorem total_filled_water_balloons :
  let max_rate := 2
  let max_time := 30
  let zach_rate := 3
  let zach_time := 40
  let popped_balloons := 10
  let max_balloons := max_rate * max_time
  let zach_balloons := zach_rate * zach_time
  let total_balloons := max_balloons + zach_balloons - popped_balloons
  total_balloons = 170 :=
by
  sorry

end total_filled_water_balloons_l194_194092


namespace find_y_of_set_with_mean_l194_194493

theorem find_y_of_set_with_mean (y : ℝ) (h : ((8 + 15 + 20 + 6 + y) / 5 = 12)) : y = 11 := 
by 
    sorry

end find_y_of_set_with_mean_l194_194493


namespace initial_percentage_of_water_l194_194485

theorem initial_percentage_of_water (C V final_volume : ℝ) (P : ℝ) 
  (hC : C = 80)
  (hV : V = 36)
  (h_final_volume : final_volume = (3/4) * C)
  (h_initial_equation: (P / 100) * C + V = final_volume) : 
  P = 30 :=
by
  sorry

end initial_percentage_of_water_l194_194485


namespace options_implication_l194_194447

theorem options_implication (a b : ℝ) :
  ((b > 0 ∧ a < 0) ∨ (a < 0 ∧ b < 0 ∧ a > b) ∨ (a > 0 ∧ b > 0 ∧ a > b)) → (1 / a < 1 / b) :=
by sorry

end options_implication_l194_194447


namespace find_x_plus_y_l194_194700

theorem find_x_plus_y (x y : ℝ) (h1 : |x| - x + y = 13) (h2 : x - |y| + y = 7) : x + y = 20 := 
by
  sorry

end find_x_plus_y_l194_194700


namespace pumpkin_pie_filling_l194_194091

theorem pumpkin_pie_filling (price_per_pumpkin : ℕ) (total_earnings : ℕ) (total_pumpkins : ℕ) (pumpkins_per_can : ℕ) :
  price_per_pumpkin = 3 →
  total_earnings = 96 →
  total_pumpkins = 83 →
  pumpkins_per_can = 3 →
  (total_pumpkins - total_earnings / price_per_pumpkin) / pumpkins_per_can = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end pumpkin_pie_filling_l194_194091


namespace smallest_positive_multiple_of_6_and_5_l194_194646

theorem smallest_positive_multiple_of_6_and_5 : ∃ (n : ℕ), (n > 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
  sorry

end smallest_positive_multiple_of_6_and_5_l194_194646


namespace expected_number_of_games_l194_194141
noncomputable def probability_of_A_winning (g : ℕ) : ℚ := 2 / 3
noncomputable def probability_of_B_winning (g : ℕ) : ℚ := 1 / 3
noncomputable def expected_games: ℚ := 266 / 81

theorem expected_number_of_games 
  (match_ends : ∀ g : ℕ, (∃ p1 p2 : ℕ, (p1 = g ∧ p2 = 0) ∨ (p1 = 0 ∧ p2 = g))) 
  (independent_outcomes : ∀ g1 g2 : ℕ, g1 ≠ g2 → probability_of_A_winning g1 * probability_of_A_winning g2 = (2 / 3) * (2 / 3) ∧ probability_of_B_winning g1 * probability_of_B_winning g2 = (1 / 3) * (1 / 3)) :
  (expected_games = 266 / 81) := 
sorry

end expected_number_of_games_l194_194141


namespace polygon_sides_given_interior_angle_l194_194671

theorem polygon_sides_given_interior_angle
  (h : ∀ (n : ℕ), (n > 2) → ((n - 2) * 180 = n * 140)): n = 9 := by
  sorry

end polygon_sides_given_interior_angle_l194_194671


namespace train_speed_l194_194741

theorem train_speed 
  (length_train : ℕ) 
  (time_crossing : ℕ) 
  (speed_kmph : ℕ)
  (h_length : length_train = 120)
  (h_time : time_crossing = 9)
  (h_speed : speed_kmph = 48) : 
  length_train / time_crossing * 3600 / 1000 = speed_kmph := 
by 
  sorry

end train_speed_l194_194741


namespace fraction_to_decimal_l194_194554

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 := by
  sorry

end fraction_to_decimal_l194_194554


namespace bankers_gain_correct_l194_194180

def PW : ℝ := 600
def R : ℝ := 0.10
def n : ℕ := 2

def A : ℝ := PW * (1 + R)^n
def BG : ℝ := A - PW

theorem bankers_gain_correct : BG = 126 :=
by
  sorry

end bankers_gain_correct_l194_194180


namespace ethan_pages_left_l194_194747

-- Definitions based on the conditions
def total_pages := 360
def pages_read_morning := 40
def pages_read_night := 10
def pages_read_saturday := pages_read_morning + pages_read_night
def pages_read_sunday := 2 * pages_read_saturday
def total_pages_read := pages_read_saturday + pages_read_sunday

-- Lean 4 statement for the proof problem
theorem ethan_pages_left : total_pages - total_pages_read = 210 := by
  sorry

end ethan_pages_left_l194_194747


namespace algebraic_expression_value_l194_194389

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a - 2 * b + 2 = 0) :
  2024 + 2 * a - b = 2023 :=
by
  sorry

end algebraic_expression_value_l194_194389


namespace proof_problem_l194_194487

def p : Prop := ∃ k : ℕ, 0 = 2 * k
def q : Prop := ∃ k : ℕ, 3 = 2 * k

theorem proof_problem : p ∨ q :=
by
  sorry

end proof_problem_l194_194487


namespace sum_of_yellow_and_blue_is_red_l194_194433

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) : ∃ k : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * k + 1 :=
by sorry

end sum_of_yellow_and_blue_is_red_l194_194433


namespace consumption_increase_l194_194877

theorem consumption_increase (T C : ℝ) (P : ℝ) (h : 0.82 * (1 + P / 100) = 0.943) :
  P = 15.06 := by
  sorry

end consumption_increase_l194_194877


namespace compare_polynomials_l194_194188

noncomputable def f (x : ℝ) : ℝ := 2*x^2 + 5*x + 3
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x + 2

theorem compare_polynomials (x : ℝ) : f x > g x :=
by sorry

end compare_polynomials_l194_194188


namespace smallest_k_l194_194824

theorem smallest_k :
  ∃ k : ℤ, k > 1 ∧ k % 13 = 1 ∧ k % 8 = 1 ∧ k % 4 = 1 ∧ k = 105 :=
by
  sorry

end smallest_k_l194_194824


namespace fraction_of_painted_surface_area_l194_194347

def total_surface_area_of_smaller_prisms : ℕ := 
  let num_smaller_prisms := 27
  let num_square_faces := num_smaller_prisms * 3
  let num_triangular_faces := num_smaller_prisms * 2
  num_square_faces + num_triangular_faces

def painted_surface_area_of_larger_prism : ℕ :=
  let painted_square_faces := 3 * 9
  let painted_triangular_faces := 2 * 9
  painted_square_faces + painted_triangular_faces

theorem fraction_of_painted_surface_area : 
  (painted_surface_area_of_larger_prism : ℚ) / (total_surface_area_of_smaller_prisms : ℚ) = 1 / 3 :=
by sorry

end fraction_of_painted_surface_area_l194_194347


namespace bobby_weekly_salary_l194_194018

variable (S : ℝ)
variables (federal_tax : ℝ) (state_tax : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) (city_fee : ℝ) (net_paycheck : ℝ)

def bobby_salary_equation := 
  S - (federal_tax * S) - (state_tax * S) - health_insurance - life_insurance - city_fee = net_paycheck

theorem bobby_weekly_salary 
  (S : ℝ) 
  (federal_tax : ℝ := 1/3) 
  (state_tax : ℝ := 0.08) 
  (health_insurance : ℝ := 50) 
  (life_insurance : ℝ := 20) 
  (city_fee : ℝ := 10) 
  (net_paycheck : ℝ := 184) 
  (valid_solution : bobby_salary_equation S (1/3) 0.08 50 20 10 184) : 
  S = 450.03 := 
  sorry

end bobby_weekly_salary_l194_194018


namespace golden_ticket_problem_l194_194879

open Real

/-- The golden ratio -/
noncomputable def φ := (1 + sqrt 5) / 2

/-- Assume the proportions and the resulting area -/
theorem golden_ticket_problem
  (a b : ℝ)
  (h : 0 + b * φ = 
        φ - (5 + sqrt 5) / (8 * φ)) :
  b / a = -4 / 3 :=
  sorry

end golden_ticket_problem_l194_194879


namespace not_right_triangle_D_l194_194751

theorem not_right_triangle_D : 
  ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 ∧
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) ∧
  (7^2 + 24^2 = 25^2) ∧
  (5^2 + 12^2 = 13^2) := 
by 
  have hA : 1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2 := by norm_num
  have hB : 7^2 + 24^2 = 25^2 := by norm_num
  have hC : 5^2 + 12^2 = 13^2 := by norm_num
  have hD : ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 := by norm_num
  exact ⟨hD, hA, hB, hC⟩

#print axioms not_right_triangle_D

end not_right_triangle_D_l194_194751


namespace rectangle_new_area_l194_194339

theorem rectangle_new_area
  (L W : ℝ) (h1 : L * W = 600) :
  let L' := 0.8 * L
  let W' := 1.3 * W
  (L' * W' = 624) :=
by
  -- Let L' = 0.8 * L
  -- Let W' = 1.3 * W
  -- Proof goes here
  sorry

end rectangle_new_area_l194_194339


namespace lives_per_player_l194_194664

theorem lives_per_player (num_players total_lives : ℕ) (h1 : num_players = 8) (h2 : total_lives = 64) :
  total_lives / num_players = 8 := by
  sorry

end lives_per_player_l194_194664


namespace taylor_scores_l194_194811

theorem taylor_scores 
    (yellow_scores : ℕ := 78)
    (white_ratio : ℕ := 7)
    (black_ratio : ℕ := 6)
    (total_ratio : ℕ := white_ratio + black_ratio)
    (difference_ratio : ℕ := white_ratio - black_ratio) :
    (2/3 : ℚ) * (yellow_scores * difference_ratio / total_ratio) = 4 := by
  sorry

end taylor_scores_l194_194811


namespace even_quadratic_increasing_l194_194734

theorem even_quadratic_increasing (m : ℝ) (h : ∀ x : ℝ, (m-1)*x^2 + 2*m*x + 1 = (m-1)*(-x)^2 + 2*m*(-x) + 1) :
  ∀ x1 x2 : ℝ, x1 < x2 ∧ x2 ≤ 0 → ((m-1)*x1^2 + 2*m*x1 + 1) < ((m-1)*x2^2 + 2*m*x2 + 1) :=
sorry

end even_quadratic_increasing_l194_194734


namespace amount_of_c_l194_194461

theorem amount_of_c (A B C : ℕ) (h1 : A + B + C = 350) (h2 : A + C = 200) (h3 : B + C = 350) : C = 200 :=
sorry

end amount_of_c_l194_194461


namespace solve_eq_log_base_l194_194729

theorem solve_eq_log_base (x : ℝ) : (9 : ℝ)^(x+8) = (10 : ℝ)^x → x = Real.logb (10 / 9) ((9 : ℝ)^8) := by
  intro h
  sorry

end solve_eq_log_base_l194_194729


namespace range_is_fixed_points_l194_194216

variable (f : ℕ → ℕ)

axiom functional_eq : ∀ m n, f (m + f n) = f (f m) + f n

theorem range_is_fixed_points :
  {n : ℕ | ∃ m : ℕ, f m = n} = {n : ℕ | f n = n} :=
sorry

end range_is_fixed_points_l194_194216


namespace total_travel_time_l194_194836

/-
Define the conditions:
1. Distance_1 is 150 miles,
2. Speed_1 is 50 mph,
3. Stop_time is 0.5 hours,
4. Distance_2 is 200 miles,
5. Speed_2 is 75 mph.

and prove that the total time equals 6.17 hours.
-/

theorem total_travel_time :
  let distance1 := 150
  let speed1 := 50
  let stop_time := 0.5
  let distance2 := 200
  let speed2 := 75
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  time1 + stop_time + time2 = 6.17 :=
by {
  -- sorry to skip the proof part
  sorry
}

end total_travel_time_l194_194836


namespace correct_average_mark_l194_194768

theorem correct_average_mark (
  num_students : ℕ := 50)
  (incorrect_avg : ℚ := 85.4)
  (wrong_mark_A : ℚ := 73.6) (correct_mark_A : ℚ := 63.5)
  (wrong_mark_B : ℚ := 92.4) (correct_mark_B : ℚ := 96.7)
  (wrong_mark_C : ℚ := 55.3) (correct_mark_C : ℚ := 51.8) :
  (incorrect_avg*num_students + 
   (correct_mark_A - wrong_mark_A) + 
   (correct_mark_B - wrong_mark_B) + 
   (correct_mark_C - wrong_mark_C)) / 
   num_students = 85.214 :=
sorry

end correct_average_mark_l194_194768


namespace remainder_of_98_pow_50_mod_50_l194_194863

theorem remainder_of_98_pow_50_mod_50 : (98 ^ 50) % 50 = 0 := by
  sorry

end remainder_of_98_pow_50_mod_50_l194_194863


namespace exists_horizontal_chord_l194_194561

theorem exists_horizontal_chord (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_eq : f 0 = f 1) : ∃ n : ℕ, n ≥ 1 ∧ ∃ x : ℝ, 0 ≤ x ∧ x + 1/n ≤ 1 ∧ f x = f (x + 1/n) :=
by
  sorry

end exists_horizontal_chord_l194_194561


namespace eat_five_pounds_in_46_875_min_l194_194102

theorem eat_five_pounds_in_46_875_min
  (fat_rate : ℝ) (thin_rate : ℝ) (combined_rate : ℝ) (total_fruit : ℝ)
  (hf1 : fat_rate = 1 / 15)
  (hf2 : thin_rate = 1 / 25)
  (h_comb : combined_rate = fat_rate + thin_rate)
  (h_fruit : total_fruit = 5) :
  total_fruit / combined_rate = 46.875 :=
by
  sorry

end eat_five_pounds_in_46_875_min_l194_194102


namespace cos_alpha_second_quadrant_l194_194950

variable (α : Real)
variable (h₁ : α ∈ Set.Ioo (π / 2) π)
variable (h₂ : Real.sin α = 5 / 13)

theorem cos_alpha_second_quadrant : Real.cos α = -12 / 13 := by
  sorry

end cos_alpha_second_quadrant_l194_194950


namespace speed_of_stream_l194_194081

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 11) (h2 : upstream_speed = 8) : 
    (downstream_speed - upstream_speed) / 2 = 1.5 :=
by
  rw [h1, h2]
  simp
  norm_num

end speed_of_stream_l194_194081


namespace solution_set_of_inequality_l194_194208

theorem solution_set_of_inequality :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -0.5 ∨ x > 2} := 
sorry

end solution_set_of_inequality_l194_194208


namespace union_A_B_l194_194125

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 0}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 3}

theorem union_A_B :
  A ∪ B = {x : ℝ | -2 ≤ x ∧ x ≤ 3} :=
sorry

end union_A_B_l194_194125


namespace manufacturing_section_degrees_l194_194803

def circle_total_degrees : ℕ := 360
def percentage_to_degree (percentage : ℕ) : ℕ := (circle_total_degrees / 100) * percentage
def manufacturing_percentage : ℕ := 60

theorem manufacturing_section_degrees : percentage_to_degree manufacturing_percentage = 216 :=
by
  -- Proof goes here
  sorry

end manufacturing_section_degrees_l194_194803


namespace part1_part2_l194_194920

variable (a b : ℝ)

theorem part1 (h : |a - 3| + |b + 6| = 0) : a + b - 2 = -5 := sorry

theorem part2 (h : |a - 3| + |b + 6| = 0) : a - b - 2 = 7 := sorry

end part1_part2_l194_194920


namespace black_area_fraction_after_four_changes_l194_194418

/-- 
Problem: Prove that after four changes, the fractional part of the original black area 
remaining black in an equilateral triangle is 81/256, given that each change splits the 
triangle into 4 smaller congruent equilateral triangles, and one of those turns white.
-/

theorem black_area_fraction_after_four_changes :
  (3 / 4) ^ 4 = 81 / 256 := sorry

end black_area_fraction_after_four_changes_l194_194418


namespace tan_eq_2sqrt3_over_3_l194_194721

theorem tan_eq_2sqrt3_over_3 (θ : ℝ) (h : 2 * Real.cos (θ - Real.pi / 3) = 3 * Real.cos θ) : 
  Real.tan θ = 2 * Real.sqrt 3 / 3 :=
by 
  sorry -- Proof is omitted as per the instructions

end tan_eq_2sqrt3_over_3_l194_194721


namespace rectangular_to_cylindrical_l194_194530

theorem rectangular_to_cylindrical (x y z : ℝ) (r θ : ℝ) (h1 : x = -3) (h2 : y = 4) (h3 : z = 5) (h4 : r = 5) (h5 : θ = Real.pi - Real.arctan (4 / 3)) :
  (r, θ, z) = (5, Real.pi - Real.arctan (4 / 3), 5) :=
by
  sorry

end rectangular_to_cylindrical_l194_194530


namespace graph_of_y_eq_neg2x_passes_quadrant_II_IV_l194_194661

-- Definitions
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x

def is_in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

def is_in_quadrant_IV (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- The main statement
theorem graph_of_y_eq_neg2x_passes_quadrant_II_IV :
  ∀ (x : ℝ), (is_in_quadrant_II x (linear_function (-2) x) ∨ 
               is_in_quadrant_IV x (linear_function (-2) x)) :=
by
  sorry

end graph_of_y_eq_neg2x_passes_quadrant_II_IV_l194_194661


namespace even_function_a_value_l194_194697

theorem even_function_a_value (a : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = (x + 1) * (x - a))
  (h_even : ∀ x, f x = f (-x)) : a = -1 :=
by
  sorry

end even_function_a_value_l194_194697


namespace expected_number_of_sixes_l194_194711

-- Define the probability of not rolling a 6 on one die
def prob_not_six : ℚ := 5 / 6

-- Define the probability of rolling zero 6's on three dice
def prob_zero_six : ℚ := prob_not_six ^ 3

-- Define the probability of rolling exactly one 6 among the three dice
def prob_one_six (n : ℕ) : ℚ := n * (1 / 6) * (prob_not_six ^ (n - 1))

-- Calculate the probabilities of each specific outcomes
def prob_exactly_zero_six : ℚ := prob_zero_six
def prob_exactly_one_six : ℚ := prob_one_six 3 * (prob_not_six ^ 2)
def prob_exactly_two_six : ℚ := prob_one_six 3 * (1 / 6) * prob_not_six
def prob_exactly_three_six : ℚ := (1 / 6) ^ 3

-- Define the expected value calculation
noncomputable def expected_value : ℚ :=
  0 * prob_exactly_zero_six
  + 1 * prob_exactly_one_six
  + 2 * prob_exactly_two_six
  + 3 * prob_exactly_three_six

-- Prove that the expected value equals to 1/2
theorem expected_number_of_sixes : expected_value = 1 / 2 :=
  by
    sorry

end expected_number_of_sixes_l194_194711


namespace most_prolific_mathematician_is_euler_l194_194971

noncomputable def prolific_mathematician (collected_works_volume_count: ℕ) (publishing_organization: String) : String :=
  if collected_works_volume_count > 75 ∧ publishing_organization = "Swiss Society of Natural Sciences" then
    "Leonhard Euler"
  else
    "Unknown"

theorem most_prolific_mathematician_is_euler :
  prolific_mathematician 76 "Swiss Society of Natural Sciences" = "Leonhard Euler" :=
by
  sorry

end most_prolific_mathematician_is_euler_l194_194971


namespace find_lines_and_intersections_l194_194866

-- Define the intersection point conditions
def intersection_point (m n : ℝ) : Prop :=
  (2 * m - n + 7 = 0) ∧ (m + n - 1 = 0)

-- Define the perpendicular line to l1 passing through (-2, 3)
def perpendicular_line_through_A (x y : ℝ) : Prop :=
  x + 2 * y - 4 = 0

-- Define the parallel line to l passing through (-2, 3)
def parallel_line_through_A (x y : ℝ) : Prop :=
  2 * x - 3 * y + 13 = 0

-- main theorem
theorem find_lines_and_intersections :
  ∃ m n : ℝ, intersection_point m n ∧ m = -2 ∧ n = 3 ∧
  ∃ l3 : ℝ → ℝ → Prop, l3 = perpendicular_line_through_A ∧
  ∃ l4 : ℝ → ℝ → Prop, l4 = parallel_line_through_A :=
sorry

end find_lines_and_intersections_l194_194866


namespace prime_divisors_of_390_l194_194924

theorem prime_divisors_of_390 : 
  (2 * 195 = 390) → 
  (3 * 65 = 195) → 
  (5 * 13 = 65) → 
  ∃ (S : Finset ℕ), 
    (∀ p ∈ S, Nat.Prime p) ∧ 
    (S.card = 4) ∧ 
    (∀ d ∈ S, d ∣ 390) := 
by
  sorry

end prime_divisors_of_390_l194_194924


namespace minimum_value_of_expression_l194_194152

theorem minimum_value_of_expression (x : ℝ) (hx : x ≠ 0) : 
  (x^2 + 1 / x^2) ≥ 2 ∧ (x^2 + 1 / x^2 = 2 ↔ x = 1 ∨ x = -1) := 
by
  sorry

end minimum_value_of_expression_l194_194152


namespace find_sale_month_4_l194_194848

-- Define the given sales data
def sale_month_1: ℕ := 5124
def sale_month_2: ℕ := 5366
def sale_month_3: ℕ := 5808
def sale_month_5: ℕ := 6124
def sale_month_6: ℕ := 4579
def average_sale_per_month: ℕ := 5400

-- Define the goal: Sale in the fourth month
def sale_month_4: ℕ := 5399

-- Prove that the total sales conforms to the given average sale
theorem find_sale_month_4 :
  sale_month_1 + sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 + sale_month_6 = 6 * average_sale_per_month :=
by
  sorry

end find_sale_month_4_l194_194848


namespace sum_of_reciprocals_of_factors_of_12_l194_194012

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l194_194012
