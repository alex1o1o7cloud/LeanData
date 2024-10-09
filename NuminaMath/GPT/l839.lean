import Mathlib

namespace difference_q_r_share_l839_83945

theorem difference_q_r_share (p q r : ℕ) (x : ℕ) (h_ratio : p = 3 * x) (h_ratio_q : q = 7 * x) (h_ratio_r : r = 12 * x) (h_diff_pq : q - p = 4400) : q - r = 5500 :=
by
  sorry

end difference_q_r_share_l839_83945


namespace min_product_of_three_l839_83975

theorem min_product_of_three :
  ∀ (list : List Int), 
    list = [-9, -7, -1, 2, 4, 6, 8] →
    ∃ (a b c : Int), a ∈ list ∧ b ∈ list ∧ c ∈ list ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ (x y z : Int), x ∈ list → y ∈ list → z ∈ list → x ≠ y → y ≠ z → x ≠ z → x * y * z ≥ a * b * c) ∧
    a * b * c = -432 :=
by
  sorry

end min_product_of_three_l839_83975


namespace perpendicular_lines_k_value_l839_83934

theorem perpendicular_lines_k_value (k : ℝ) : 
  (∃ (m₁ m₂ : ℝ), (m₁ = k/3) ∧ (m₂ = 3) ∧ (m₁ * m₂ = -1)) → k = -1 :=
by
  sorry

end perpendicular_lines_k_value_l839_83934


namespace find_positive_x_l839_83985

theorem find_positive_x (x y z : ℝ) (h1 : x * y = 10 - 3 * x - 2 * y)
  (h2 : y * z = 8 - 3 * y - 2 * z) (h3 : x * z = 40 - 5 * x - 3 * z) :
  x = 3 :=
by sorry

end find_positive_x_l839_83985


namespace total_points_combined_l839_83908

-- Definitions of the conditions
def Jack_points : ℕ := 8972
def Alex_Bella_points : ℕ := 21955

-- The problem statement to be proven
theorem total_points_combined : Jack_points + Alex_Bella_points = 30927 :=
by sorry

end total_points_combined_l839_83908


namespace find_a_l839_83922

theorem find_a 
  (x y a : ℝ) 
  (hx : x = 1) 
  (hy : y = -3) 
  (h : a * x - y = 1) : 
  a = -2 := 
  sorry

end find_a_l839_83922


namespace value_of_T_l839_83932

-- Define the main variables and conditions
variables {M T : ℝ}

-- State the conditions given in the problem
def condition1 (M T : ℝ) := 2 * M + T = 7000
def condition2 (M T : ℝ) := M + 2 * T = 9800

-- State the theorem to be proved
theorem value_of_T : 
  ∀ (M T : ℝ), condition1 M T ∧ condition2 M T → T = 4200 :=
by 
  -- Proof would go here; for now, we use "sorry" to skip it
  sorry

end value_of_T_l839_83932


namespace jack_total_plates_after_smashing_and_buying_l839_83957

def initial_flower_plates : ℕ := 6
def initial_checked_plates : ℕ := 9
def initial_striped_plates : ℕ := 3
def smashed_flower_plates : ℕ := 2
def smashed_striped_plates : ℕ := 1
def new_polka_dotted_plates : ℕ := initial_checked_plates * initial_checked_plates

theorem jack_total_plates_after_smashing_and_buying : 
  initial_flower_plates - smashed_flower_plates
  + initial_checked_plates
  + initial_striped_plates - smashed_striped_plates
  + new_polka_dotted_plates = 96 := 
by {
  -- calculation proof here
  sorry
}

end jack_total_plates_after_smashing_and_buying_l839_83957


namespace parabola_focus_coordinates_l839_83917

noncomputable def parabola_focus (a b : ℝ) := (0, (1 / (4 * a)) + 2)

theorem parabola_focus_coordinates (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x : ℝ, abs (a * x^2 + b * x + 2) ≥ 2) :
  parabola_focus a b = (0, 2 + (1 / (4 * a))) := sorry

end parabola_focus_coordinates_l839_83917


namespace tangent_parallel_to_line_at_point_l839_83913

theorem tangent_parallel_to_line_at_point (P0 : ℝ × ℝ) 
  (curve : ℝ → ℝ) (line_slope : ℝ) : 
  curve = (fun x => x^3 + x - 2) ∧ line_slope = 4 ∧
  (∃ x0, P0 = (x0, curve x0) ∧ 3*x0^2 + 1 = line_slope) → 
  P0 = (1, 0) :=
by 
  sorry

end tangent_parallel_to_line_at_point_l839_83913


namespace simplify_fraction_l839_83970

theorem simplify_fraction : (150 / 4350 : ℚ) = 1 / 29 :=
  sorry

end simplify_fraction_l839_83970


namespace value_subtracted_3_times_number_eq_1_l839_83916

variable (n : ℝ) (v : ℝ)

theorem value_subtracted_3_times_number_eq_1 (h1 : n = 1.0) (h2 : 3 * n - v = 2 * n) : v = 1 :=
by
  sorry

end value_subtracted_3_times_number_eq_1_l839_83916


namespace range_of_a_minus_b_l839_83986

theorem range_of_a_minus_b (a b : ℝ) (h₁ : -1 < a) (h₂ : a < 1) (h₃ : 1 < b) (h₄ : b < 3) : 
  -4 < a - b ∧ a - b < 0 := by
  sorry

end range_of_a_minus_b_l839_83986


namespace verify_triangle_inequality_l839_83960

-- Conditions of the problem
variables (L : ℕ → ℕ)
-- The rods lengths are arranged in increasing order
axiom rods_in_order : ∀ i : ℕ, L i ≤ L (i + 1)

-- Define the critical check
def critical_check : Prop :=
  L 98 + L 99 > L 100

-- Prove that verifying the critical_check is sufficient
theorem verify_triangle_inequality (h : critical_check L) :
  ∀ i j k : ℕ, 1 ≤ i → i < j → j < k → k ≤ 100 → L i + L j > L k :=
by
  sorry

end verify_triangle_inequality_l839_83960


namespace range_of_f_l839_83907

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 2 ^ x

theorem range_of_f :
  {x : ℝ | f x + f (x - 0.5) > 1} = {x : ℝ | x > -0.25} :=
by
  sorry

end range_of_f_l839_83907


namespace hydropump_output_l839_83974

theorem hydropump_output :
  ∀ (rate : ℕ) (time_hours : ℚ), 
    rate = 600 → 
    time_hours = 1.5 → 
    rate * time_hours = 900 :=
by
  intros rate time_hours rate_cond time_cond 
  sorry

end hydropump_output_l839_83974


namespace triangle_angle_solution_exists_l839_83984

noncomputable def possible_angles (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (A = 120 ∨ B = 120 ∨ C = 120) ∧
  (
    ((A = 40 ∧ B = 20) ∨ (A = 20 ∧ B = 40)) ∨
    ((A = 45 ∧ B = 15) ∨ (A = 15 ∧ B = 45))
  )
  
theorem triangle_angle_solution_exists :
  ∃ A B C : ℝ, possible_angles A B C :=
sorry

end triangle_angle_solution_exists_l839_83984


namespace conditional_probability_l839_83969

-- Definitions of the events and probabilities given in the conditions
def event_A (red : ℕ) : Prop := red % 3 = 0
def event_B (red blue : ℕ) : Prop := red + blue > 8

-- The actual values of probabilities calculated in the solution
def P_A : ℚ := 1/3
def P_B : ℚ := 1/3
def P_AB : ℚ := 5/36

-- Definition of conditional probability
def P_B_given_A : ℚ := P_AB / P_A

-- The claim we want to prove
theorem conditional_probability :
  P_B_given_A = 5 / 12 :=
sorry

end conditional_probability_l839_83969


namespace total_distance_traveled_is_7_75_l839_83924

open Real

def walking_time_minutes : ℝ := 30
def walking_rate : ℝ := 3.5

def running_time_minutes : ℝ := 45
def running_rate : ℝ := 8

theorem total_distance_traveled_is_7_75 :
  let walking_hours := walking_time_minutes / 60
  let distance_walked := walking_rate * walking_hours
  let running_hours := running_time_minutes / 60
  let distance_run := running_rate * running_hours
  let total_distance := distance_walked + distance_run
  total_distance = 7.75 :=
by
  sorry

end total_distance_traveled_is_7_75_l839_83924


namespace set_listing_method_l839_83961

theorem set_listing_method :
  {x : ℤ | -3 < 2 * x - 1 ∧ 2 * x - 1 < 5} = {0, 1, 2} :=
by
  sorry

end set_listing_method_l839_83961


namespace shaded_area_correct_l839_83900

noncomputable def shaded_area (s r_small : ℝ) : ℝ :=
  let hex_area := (3 * Real.sqrt 3 / 2) * s^2
  let semi_area := 6 * (1/2 * Real.pi * (s/2)^2)
  let small_circle_area := 6 * (Real.pi * (r_small)^2)
  hex_area - (semi_area + small_circle_area)

theorem shaded_area_correct : shaded_area 4 0.5 = 24 * Real.sqrt 3 - (27 * Real.pi / 2) := by
  sorry

end shaded_area_correct_l839_83900


namespace complex_fraction_sum_zero_l839_83901

section complex_proof
open Complex

theorem complex_fraction_sum_zero (z1 z2 : ℂ) (hz1 : z1 = 1 + I) (hz2 : z2 = 1 - I) :
  (z1 / z2) + (z2 / z1) = 0 := by
  sorry
end complex_proof

end complex_fraction_sum_zero_l839_83901


namespace no_integer_solutions_for_trapezoid_bases_l839_83993

theorem no_integer_solutions_for_trapezoid_bases :
  ∃ (A h : ℤ) (b1_b2 : ℤ → Prop),
    A = 2800 ∧ h = 80 ∧
    (∀ m n : ℤ, b1_b2 (12 * m) ∧ b1_b2 (12 * n) → (12 * m + 12 * n = 70) → false) :=
by
  sorry

end no_integer_solutions_for_trapezoid_bases_l839_83993


namespace books_added_is_10_l839_83995

-- Define initial number of books on the shelf
def initial_books : ℕ := 38

-- Define the final number of books on the shelf
def final_books : ℕ := 48

-- Define the number of books that Marta added
def books_added : ℕ := final_books - initial_books

-- Theorem stating that Marta added 10 books
theorem books_added_is_10 : books_added = 10 :=
by
  sorry

end books_added_is_10_l839_83995


namespace trajectory_circle_equation_l839_83918

theorem trajectory_circle_equation :
  (∀ (x y : ℝ), dist (x, y) (0, 0) = 4 ↔ x^2 + y^2 = 16) :=  
sorry

end trajectory_circle_equation_l839_83918


namespace jon_toaster_total_cost_l839_83976

def total_cost_toaster (MSRP : ℝ) (std_ins_pct : ℝ) (premium_upgrade_cost : ℝ) (state_tax_pct : ℝ) (environmental_fee : ℝ) : ℝ :=
  let std_ins_cost := std_ins_pct * MSRP
  let premium_ins_cost := std_ins_cost + premium_upgrade_cost
  let subtotal_before_tax := MSRP + premium_ins_cost
  let state_tax := state_tax_pct * subtotal_before_tax
  let total_before_env_fee := subtotal_before_tax + state_tax
  total_before_env_fee + environmental_fee

theorem jon_toaster_total_cost :
  total_cost_toaster 30 0.2 7 0.5 5 = 69.5 :=
by
  sorry

end jon_toaster_total_cost_l839_83976


namespace focus_of_parabola_l839_83929

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0, and for any real number x, it holds that |f(x)| ≥ 2,
    prove that the coordinates of the focus of the parabolic curve are (0, 1 / (4 * a) + 2). -/
theorem focus_of_parabola (a b : ℝ) (h_a : a ≠ 0)
  (h_f : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
  (0, (1 / (4 * a) + 2)) = (0, (1 / (4 * a) + 2)) :=
by
  sorry

end focus_of_parabola_l839_83929


namespace x_gt_one_iff_x_cube_gt_one_l839_83954

theorem x_gt_one_iff_x_cube_gt_one (x : ℝ) : x > 1 ↔ x^3 > 1 :=
by sorry

end x_gt_one_iff_x_cube_gt_one_l839_83954


namespace mean_equality_l839_83965

theorem mean_equality (y : ℝ) : 
  (6 + 9 + 18) / 3 = (12 + y) / 2 → y = 10 :=
by
  intros h
  sorry

end mean_equality_l839_83965


namespace river_current_speed_l839_83909

theorem river_current_speed :
  ∀ (D v A_speed B_speed time_interval : ℝ),
    D = 200 →
    A_speed = 36 →
    B_speed = 64 →
    time_interval = 4 →
    3 * D = (A_speed + v) * 2 * (1 + time_interval / ((A_speed + v) + (B_speed - v))) * 200 :=
sorry

end river_current_speed_l839_83909


namespace negation_of_existential_l839_83982

theorem negation_of_existential :
  ¬ (∃ x : ℝ, x^2 - 2 * x - 3 < 0) ↔ ∀ x : ℝ, x^2 - 2 * x - 3 ≥ 0 :=
by sorry

end negation_of_existential_l839_83982


namespace initial_pills_count_l839_83943

theorem initial_pills_count 
  (pills_taken_first_2_days : ℕ)
  (pills_taken_next_3_days : ℕ)
  (pills_taken_sixth_day : ℕ)
  (pills_left : ℕ)
  (h1 : pills_taken_first_2_days = 2 * 3 * 2)
  (h2 : pills_taken_next_3_days = 1 * 3 * 3)
  (h3 : pills_taken_sixth_day = 2)
  (h4 : pills_left = 27) :
  ∃ initial_pills : ℕ, initial_pills = pills_taken_first_2_days + pills_taken_next_3_days + pills_taken_sixth_day + pills_left :=
by
  sorry

end initial_pills_count_l839_83943


namespace sum_formula_l839_83964

open Nat

/-- The sequence a_n defined as (-1)^n * (2 * n - 1) -/
def a_n (n : ℕ) : ℤ :=
  (-1) ^ n * (2 * n - 1)

/-- The partial sum S_n of the first n terms of the sequence a_n -/
def S_n : ℕ → ℤ
| 0     => 0
| (n+1) => S_n n + a_n (n + 1)

/-- The main theorem: For all n in natural numbers, S_n = (-1)^n * n -/
theorem sum_formula (n : ℕ) : S_n n = (-1) ^ n * n := by
  sorry

end sum_formula_l839_83964


namespace distinct_seatings_l839_83923

theorem distinct_seatings : 
  ∃ n : ℕ, (n = 288000) ∧ 
  (∀ (men wives : Fin 6 → ℕ),
  ∃ (f : (Fin 12) → ℕ), 
  (∀ i, f (i + 1) % 12 ≠ f i) ∧
  (∀ i, f i % 2 = 0) ∧
  (∀ j, f (2 * j) = men j ∧ f (2 * j + 1) = wives j)) :=
by
  sorry

end distinct_seatings_l839_83923


namespace factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l839_83903

-- Problem 1: Factorize x^2 - y^2 + 2x - 2y
theorem factorize_expression (x y : ℝ) : x^2 - y^2 + 2 * x - 2 * y = (x - y) * (x + y + 2) := 
by sorry

-- Problem 2: Determine the shape of a triangle given a^2 + c^2 - 2b(a - b + c) = 0
theorem equilateral_triangle_of_sides (a b c : ℝ) (h : a^2 + c^2 - 2 * b * (a - b + c) = 0) : a = b ∧ b = c :=
by sorry

-- Problem 3: Prove that 2p = m + n given (1/4)(m - n)^2 = (p - n)(m - p)
theorem two_p_eq_m_plus_n (m n p : ℝ) (h : (1/4) * (m - n)^2 = (p - n) * (m - p)) : 2 * p = m + n := 
by sorry

end factorize_expression_equilateral_triangle_of_sides_two_p_eq_m_plus_n_l839_83903


namespace project_estimated_hours_l839_83914

theorem project_estimated_hours (extra_hours_per_day : ℕ) (normal_work_hours : ℕ) (days_to_finish : ℕ)
  (total_hours_estimation : ℕ)
  (h1 : extra_hours_per_day = 5)
  (h2 : normal_work_hours = 10)
  (h3 : days_to_finish = 100)
  (h4 : total_hours_estimation = days_to_finish * (normal_work_hours + extra_hours_per_day))
  : total_hours_estimation = 1500 :=
  by
  -- Proof to be provided 
  sorry

end project_estimated_hours_l839_83914


namespace rope_segments_l839_83997

theorem rope_segments (total_length : ℝ) (n : ℕ) (h1 : total_length = 3) (h2 : n = 7) :
  (∃ segment_fraction : ℝ, segment_fraction = 1 / n ∧
   ∃ segment_length : ℝ, segment_length = total_length / n) :=
sorry

end rope_segments_l839_83997


namespace julia_money_remaining_l839_83940

theorem julia_money_remaining 
  (initial_amount : ℝ)
  (tablet_percentage : ℝ)
  (phone_percentage : ℝ)
  (game_percentage : ℝ)
  (case_percentage : ℝ) 
  (final_money : ℝ) :
  initial_amount = 120 → 
  tablet_percentage = 0.45 → 
  phone_percentage = 1/3 → 
  game_percentage = 0.25 → 
  case_percentage = 0.10 → 
  final_money = initial_amount * (1 - tablet_percentage) * (1 - phone_percentage) * (1 - game_percentage) * (1 - case_percentage) →
  final_money = 29.70 :=
by
  intros
  sorry

end julia_money_remaining_l839_83940


namespace find_six_digit_numbers_l839_83938

variable (m n : ℕ)

-- Definition that the original number becomes six-digit when multiplied by 4
def is_six_digit (x : ℕ) : Prop := x ≥ 100000 ∧ x < 1000000

-- Conditions
def original_number := 100 * m + n
def new_number := 10000 * n + m
def satisfies_conditions (m n : ℕ) : Prop :=
  is_six_digit (100 * m + n) ∧
  is_six_digit (10000 * n + m) ∧
  4 * (100 * m + n) = 10000 * n + m

-- Theorem statement
theorem find_six_digit_numbers (h₁ : satisfies_conditions 1428 57)
                               (h₂ : satisfies_conditions 1904 76)
                               (h₃ : satisfies_conditions 2380 95) :
  ∃ m n, satisfies_conditions m n :=
  sorry -- Proof omitted

end find_six_digit_numbers_l839_83938


namespace angle_in_third_quadrant_l839_83972

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) (h : π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π) :
  ∃ m : ℤ, -π - 2 * m * π < π / 2 - α ∧ (π / 2 - α) < -π / 2 - 2 * m * π :=
by
  -- Lean users note: The proof isn't required here, just setting up the statement as instructed.
  sorry

end angle_in_third_quadrant_l839_83972


namespace tim_pencils_l839_83935

-- Problem statement: If x = 2 and z = 5, then y = z - x where y is the number of pencils Tim placed.
def pencils_problem (x y z : Nat) : Prop :=
  x = 2 ∧ z = 5 → y = z - x

theorem tim_pencils : pencils_problem 2 3 5 :=
by
  sorry

end tim_pencils_l839_83935


namespace pints_in_two_liters_l839_83947

theorem pints_in_two_liters (p : ℝ) (h : p = 1.575 / 0.75) : 2 * p = 4.2 := 
sorry

end pints_in_two_liters_l839_83947


namespace albania_inequality_l839_83990

variable (a b c r R s : ℝ)
variable (h1 : a + b > c)
variable (h2 : b + c > a)
variable (h3 : c + a > b)
variable (h4 : r > 0)
variable (h5 : R > 0)
variable (h6 : s = (a + b + c) / 2)

theorem albania_inequality :
    1 / (a + b) + 1 / (a + c) + 1 / (b + c) ≤ r / (16 * R * s) + s / (16 * R * r) + 11 / (8 * s) :=
sorry

end albania_inequality_l839_83990


namespace find_d_l839_83981

theorem find_d (c d : ℝ) (h1 : c / d = 5) (h2 : c = 18 - 7 * d) : d = 3 / 2 := by
  sorry

end find_d_l839_83981


namespace find_n_in_arithmetic_sequence_l839_83968

noncomputable def arithmetic_sequence_n : ℕ :=
  sorry

theorem find_n_in_arithmetic_sequence (a : ℕ → ℕ) (d n : ℕ) :
  (a 3) + (a 4) = 10 → (a (n-3) + a (n-2)) = 30 → n * (a 1 + a n) / 2 = 100 → n = 10 :=
  sorry

end find_n_in_arithmetic_sequence_l839_83968


namespace lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l839_83980

def lassis_per_three_mangoes := 15
def smoothies_per_mango := 1
def bananas_per_smoothie := 2

-- proving the number of lassis Caroline can make with eighteen mangoes
theorem lassis_with_eighteen_mangoes :
  (18 / 3) * lassis_per_three_mangoes = 90 :=
by 
  sorry

-- proving the number of smoothies Caroline can make with eighteen mangoes and thirty-six bananas
theorem smoothies_with_eighteen_mangoes_and_thirtysix_bananas :
  min (18 / smoothies_per_mango) (36 / bananas_per_smoothie) = 18 :=
by 
  sorry

end lassis_with_eighteen_mangoes_smoothies_with_eighteen_mangoes_and_thirtysix_bananas_l839_83980


namespace smallest_value_3a_plus_1_l839_83919

theorem smallest_value_3a_plus_1 (a : ℚ) (h : 8 * a^2 + 6 * a + 5 = 2) : 3 * a + 1 = -5 / 4 :=
sorry

end smallest_value_3a_plus_1_l839_83919


namespace angle_equiv_terminal_side_l839_83942

theorem angle_equiv_terminal_side (θ : ℤ) : 
  let θ_deg := (750 : ℕ)
  let reduced_angle := θ_deg % 360
  0 ≤ reduced_angle ∧ reduced_angle < 360 ∧ reduced_angle = 30:=
by
  sorry

end angle_equiv_terminal_side_l839_83942


namespace find_third_integer_l839_83951

noncomputable def third_odd_integer (x : ℤ) :=
  x + 4

theorem find_third_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : third_odd_integer x = 15 :=
by
  sorry

end find_third_integer_l839_83951


namespace probability_all_same_color_l839_83936

theorem probability_all_same_color :
  let total_marbles := 20
  let red_marbles := 5
  let white_marbles := 7
  let blue_marbles := 8
  let total_ways_to_draw_3 := (total_marbles * (total_marbles - 1) * (total_marbles - 2)) / 6
  let ways_to_draw_3_red := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) / 6
  let ways_to_draw_3_white := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) / 6
  let ways_to_draw_3_blue := (blue_marbles * (blue_marbles - 1) * (blue_marbles - 2)) / 6
  let probability := (ways_to_draw_3_red + ways_to_draw_3_white + ways_to_draw_3_blue) / total_ways_to_draw_3
  probability = 101/1140 :=
by
  sorry

end probability_all_same_color_l839_83936


namespace milk_production_days_l839_83953

variable (x : ℕ)
def cows := 2 * x
def cans := 2 * x + 2
def days := 2 * x + 1
def total_cows := 2 * x + 4
def required_cans := 2 * x + 10

theorem milk_production_days :
  (total_cows * required_cans) = ((2 * x) * (2 * x + 1) * required_cans) / ((2 * x + 2) * (2 * x + 4)) :=
sorry

end milk_production_days_l839_83953


namespace henry_finishes_on_thursday_l839_83966

theorem henry_finishes_on_thursday :
  let total_days := 210
  let start_day := 4  -- Assume Thursday is 4th day of the week in 0-indexed (0=Sunday, 1=Monday, ..., 6=Saturday)
  (start_day + total_days) % 7 = start_day :=
by
  sorry

end henry_finishes_on_thursday_l839_83966


namespace darcy_folded_shorts_l839_83959

-- Define the conditions
def total_shirts : Nat := 20
def total_shorts : Nat := 8
def folded_shirts : Nat := 12
def remaining_pieces : Nat := 11

-- Expected result to prove
def folded_shorts : Nat := 5

-- The statement to prove
theorem darcy_folded_shorts : total_shorts - (remaining_pieces - (total_shirts - folded_shirts)) = folded_shorts :=
by
  sorry

end darcy_folded_shorts_l839_83959


namespace volume_frustum_as_fraction_of_original_l839_83933

theorem volume_frustum_as_fraction_of_original :
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  (volume_frustum / volume_original) = (87 / 96) :=
by
  let original_base_edge := 40
  let original_altitude := 20
  let smaller_altitude := original_altitude / 3
  let smaller_base_edge := original_base_edge / 3
  let volume_original := (1 / 3) * (original_base_edge * original_base_edge) * original_altitude
  let volume_smaller := (1 / 3) * (smaller_base_edge * smaller_base_edge) * smaller_altitude
  let volume_frustum := volume_original - volume_smaller
  have h : volume_frustum / volume_original = 87 / 96 := sorry
  exact h

end volume_frustum_as_fraction_of_original_l839_83933


namespace product_sum_125_l839_83927

theorem product_sum_125 :
  ∀ (m n : ℕ), m ≥ n ∧
              (∀ (k : ℕ), 0 < k → |Real.log m - Real.log k| < Real.log n → k ≠ 0)
              → (m * n = 125) :=
by sorry

end product_sum_125_l839_83927


namespace initial_order_cogs_l839_83999

theorem initial_order_cogs (x : ℕ) (h : (x + 60 : ℚ) / (x / 36 + 1) = 45) : x = 60 := 
sorry

end initial_order_cogs_l839_83999


namespace max_distance_traveled_l839_83971

theorem max_distance_traveled (fare: ℝ) (x: ℝ) :
  fare = 17.2 → 
  x > 3 →
  1.4 * (x - 3) + 6 ≤ fare → 
  x ≤ 11 := by
  sorry

end max_distance_traveled_l839_83971


namespace scramble_time_is_correct_l839_83958

-- Define the conditions
def sausages : ℕ := 3
def fry_time_per_sausage : ℕ := 5
def eggs : ℕ := 6
def total_time : ℕ := 39

-- Define the time to scramble each egg
def scramble_time_per_egg : ℕ :=
  let frying_time := sausages * fry_time_per_sausage
  let scrambling_time := total_time - frying_time
  scrambling_time / eggs

-- The theorem stating the main question and desired answer
theorem scramble_time_is_correct : scramble_time_per_egg = 4 := by
  sorry

end scramble_time_is_correct_l839_83958


namespace domain_ln_l839_83962

theorem domain_ln (x : ℝ) (h : x - 1 > 0) : x > 1 := 
sorry

end domain_ln_l839_83962


namespace combined_weight_of_three_l839_83987

theorem combined_weight_of_three (Mary Jamison John : ℝ) 
  (h₁ : Mary = 160) 
  (h₂ : Jamison = Mary + 20) 
  (h₃ : John = Mary + (1/4) * Mary) :
  Mary + Jamison + John = 540 := by
  sorry

end combined_weight_of_three_l839_83987


namespace maximum_root_l839_83930

noncomputable def max_root (α β γ : ℝ) : ℝ := 
  if α ≥ β ∧ α ≥ γ then α 
  else if β ≥ α ∧ β ≥ γ then β 
  else γ

theorem maximum_root :
  ∃ α β γ : ℝ, α + β + γ = 14 ∧ α^2 + β^2 + γ^2 = 84 ∧ α^3 + β^3 + γ^3 = 584 ∧ max_root α β γ = 8 :=
by
  sorry

end maximum_root_l839_83930


namespace find_S16_l839_83963

theorem find_S16 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 12 = -8)
  (h2 : S 9 = -9)
  (h_sum : ∀ n, S n = (n * (a 1 + a n) / 2)) :
  S 16 = -72 := 
by
  sorry

end find_S16_l839_83963


namespace arithmetic_sequence_solution_l839_83939

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence : Prop :=
  ∀ n, a (n+1) = a n + d

-- The given condition a_3 + a_5 = 12 - a_7
def condition : Prop :=
  a 3 + a 5 = 12 - a 7

-- The proof statement
theorem arithmetic_sequence_solution 
  (h_arith : is_arithmetic_sequence a d) 
  (h_cond : condition a): a 1 + a 9 = 8 :=
sorry

end arithmetic_sequence_solution_l839_83939


namespace find_f_l839_83902

theorem find_f (d e f : ℝ) (h_g : 16 = g) 
  (h_mean_of_zeros : -d / 12 = 3 + d + e + f + 16) 
  (h_product_of_zeros_two_at_a_time : -d / 12 = e / 3) : 
  f = -39 :=
by
  sorry

end find_f_l839_83902


namespace log_neg_inequality_l839_83978

theorem log_neg_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  Real.log (-a) > Real.log (-b) := 
sorry

end log_neg_inequality_l839_83978


namespace coconut_grove_problem_l839_83906

theorem coconut_grove_problem
  (x : ℤ)
  (T40 : ℤ := x + 2)
  (T120 : ℤ := x)
  (T180 : ℤ := x - 2)
  (N_total : ℤ := 40 * (x + 2) + 120 * x + 180 * (x - 2))
  (T_total : ℤ := (x + 2) + x + (x - 2))
  (average_yield : ℤ := 100) :
  (N_total / T_total) = average_yield → x = 7 :=
by
  sorry

end coconut_grove_problem_l839_83906


namespace sin_cos_special_l839_83973

def special_operation (a b : ℝ) : ℝ := a^2 - a * b - b^2

theorem sin_cos_special (x : ℝ) : 
  special_operation (Real.sin (x / 12)) (Real.cos (x / 12)) = -(1 + 2 * Real.sqrt 3) / 4 :=
  sorry

end sin_cos_special_l839_83973


namespace zongzi_cost_per_bag_first_batch_l839_83904

theorem zongzi_cost_per_bag_first_batch (x : ℝ)
  (h1 : 7500 / (x - 4) = 3 * (3000 / x))
  (h2 : 3000 > 0)
  (h3 : 7500 > 0)
  (h4 : x > 4) :
  x = 24 :=
by sorry

end zongzi_cost_per_bag_first_batch_l839_83904


namespace speed_of_stream_l839_83949

def upstream_speed (v : ℝ) := 72 - v
def downstream_speed (v : ℝ) := 72 + v

theorem speed_of_stream (v : ℝ) (h : 1 / upstream_speed v = 2 * (1 / downstream_speed v)) : v = 24 :=
by 
  sorry

end speed_of_stream_l839_83949


namespace least_possible_value_of_quadratic_l839_83989

theorem least_possible_value_of_quadratic (p q : ℝ) (hq : ∀ x : ℝ, x^2 + p * x + q ≥ 0) : q = (p^2) / 4 :=
sorry

end least_possible_value_of_quadratic_l839_83989


namespace xy_value_l839_83911

theorem xy_value :
  ∃ a b c x y : ℝ,
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧
    3 * a + 2 * b + c = 5 ∧
    2 * a + b - 3 * c = 1 ∧
    (∀ m, m = 3 * a + b - 7 * c → (m = x ∨ m = y)) ∧
    x = -5 / 7 ∧
    y = -1 / 11 ∧
    x * y = 5 / 77 :=
sorry

end xy_value_l839_83911


namespace find_W_from_conditions_l839_83992

theorem find_W_from_conditions :
  ∀ (x y : ℝ), (y = 1 / x ∧ y = |x| + 1) → (x + y = Real.sqrt 5) :=
by
  sorry

end find_W_from_conditions_l839_83992


namespace prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l839_83912

noncomputable def prob_A_wins_game := 2 / 3
noncomputable def prob_B_wins_game := 1 / 3

/-- The probability that the match ends after two games with player A's victory is 4/9. -/
theorem prob_match_ends_two_games_A_wins :
  prob_A_wins_game * prob_A_wins_game = 4 / 9 := by
  sorry

/-- The probability that the match ends exactly after four games is 20/81. -/
theorem prob_match_ends_four_games :
  2 * prob_A_wins_game * prob_B_wins_game * (prob_A_wins_game^2 + prob_B_wins_game^2) = 20 / 81 := by
  sorry

/-- The probability that player A wins the match overall is 74/81. -/
theorem prob_A_wins_overall :
  (prob_A_wins_game^2 + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game^2
  + 2 * prob_A_wins_game * prob_B_wins_game * prob_A_wins_game * prob_B_wins_game) / (prob_A_wins_game + prob_B_wins_game) = 74 / 81 := by
  sorry

end prob_match_ends_two_games_A_wins_prob_match_ends_four_games_prob_A_wins_overall_l839_83912


namespace students_neither_football_nor_cricket_l839_83950

def total_students : ℕ := 450
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def both_players : ℕ := 100

theorem students_neither_football_nor_cricket : 
  total_students - (football_players + cricket_players - both_players) = 50 := by
  sorry

end students_neither_football_nor_cricket_l839_83950


namespace even_square_is_even_l839_83979

theorem even_square_is_even (a : ℤ) (h : Even (a^2)) : Even a :=
sorry

end even_square_is_even_l839_83979


namespace top_cell_pos_cases_l839_83967

-- Define the rule for the cell sign propagation
def cell_sign (a b : ℤ) : ℤ := 
  if a = b then 1 else -1

-- The pyramid height
def pyramid_height : ℕ := 5

-- Define the final condition for the top cell in the pyramid to be "+"
def top_cell_sign (a b c d e : ℤ) : ℤ :=
  a * b * c * d * e

-- Define the proof statement
theorem top_cell_pos_cases :
  (∃ a b c d e : ℤ,
    (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    top_cell_sign a b c d e = 1) ∧
  (∃ n, n = 11) :=
by
  sorry

end top_cell_pos_cases_l839_83967


namespace circle_reflection_l839_83905

theorem circle_reflection (x y : ℝ) (hx : x = 8) (hy : y = -3)
    (new_x new_y : ℝ) (hne_x : new_x = 3) (hne_y : new_y = -8) :
    (new_x, new_y) = (-y, -x) := by
  sorry

end circle_reflection_l839_83905


namespace deepak_present_age_l839_83998

def present_age_rahul (x : ℕ) : ℕ := 4 * x
def present_age_deepak (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age : ∀ (x : ℕ), 
  (present_age_rahul x + 22 = 26) →
  present_age_deepak x = 3 := 
by
  intros x h
  sorry

end deepak_present_age_l839_83998


namespace jackson_has_1900_more_than_brandon_l839_83925

-- Conditions
def initial_investment : ℝ := 500
def jackson_multiplier : ℝ := 4
def brandon_multiplier : ℝ := 0.20

-- Final values
def jackson_final_value := jackson_multiplier * initial_investment
def brandon_final_value := brandon_multiplier * initial_investment

-- Statement to prove the difference
theorem jackson_has_1900_more_than_brandon : jackson_final_value - brandon_final_value = 1900 := 
    by sorry

end jackson_has_1900_more_than_brandon_l839_83925


namespace range_for_a_l839_83994

theorem range_for_a (a : ℝ) : 
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by
  sorry

end range_for_a_l839_83994


namespace passing_marks_l839_83937

theorem passing_marks (T P : ℝ) (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) : P = 120 := 
by
  sorry

end passing_marks_l839_83937


namespace subtraction_makes_divisible_l839_83931

theorem subtraction_makes_divisible :
  ∃ n : Nat, 9671 - n % 2 = 0 ∧ n = 1 :=
by
  sorry

end subtraction_makes_divisible_l839_83931


namespace bikers_meet_again_in_36_minutes_l839_83926

theorem bikers_meet_again_in_36_minutes :
    Nat.lcm 12 18 = 36 :=
sorry

end bikers_meet_again_in_36_minutes_l839_83926


namespace quadratic_inequality_solution_l839_83920

theorem quadratic_inequality_solution
  (a b : ℝ)
  (h1 : ∀ x : ℝ, x^2 + a * x + b > 0 ↔ (x < -2 ∨ -1/2 < x)) :
  ∀ x : ℝ, b * x^2 + a * x + 1 < 0 ↔ -2 < x ∧ x < -1/2 :=
by
  sorry

end quadratic_inequality_solution_l839_83920


namespace scientific_notation_of_0_0000012_l839_83921

theorem scientific_notation_of_0_0000012 :
  0.0000012 = 1.2 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000012_l839_83921


namespace passengers_from_other_continents_l839_83991

theorem passengers_from_other_continents :
  (∀ (n NA EU AF AS : ℕ),
     NA = n / 4 →
     EU = n / 8 →
     AF = n / 12 →
     AS = n / 6 →
     96 = n →
     n - (NA + EU + AF + AS) = 36) :=
by
  sorry

end passengers_from_other_continents_l839_83991


namespace remaining_stock_is_120_l839_83948

-- Definitions derived from conditions
def green_beans_weight : ℕ := 60
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 10
def rice_lost_weight : ℕ := rice_weight / 3
def sugar_lost_weight : ℕ := sugar_weight / 5
def remaining_rice : ℕ := rice_weight - rice_lost_weight
def remaining_sugar : ℕ := sugar_weight - sugar_lost_weight
def remaining_stock_weight : ℕ := remaining_rice + remaining_sugar + green_beans_weight

-- Theorem
theorem remaining_stock_is_120 : remaining_stock_weight = 120 := by
  sorry

end remaining_stock_is_120_l839_83948


namespace percentage_increase_l839_83996

theorem percentage_increase (original new : ℝ) (h₁ : original = 50) (h₂ : new = 80) :
  ((new - original) / original) * 100 = 60 :=
by
  sorry

end percentage_increase_l839_83996


namespace valid_range_of_x_l839_83956

theorem valid_range_of_x (x : ℝ) (h1 : 2 - x ≥ 0) (h2 : x + 1 ≠ 0) : x ≤ 2 ∧ x ≠ -1 :=
sorry

end valid_range_of_x_l839_83956


namespace tracy_sold_paintings_l839_83952

-- Definitions of conditions
def total_customers := 20
def first_group_customers := 4
def paintings_per_first_group_customer := 2
def second_group_customers := 12
def paintings_per_second_group_customer := 1
def third_group_customers := 4
def paintings_per_third_group_customer := 4

-- Statement of the problem
theorem tracy_sold_paintings :
  (first_group_customers * paintings_per_first_group_customer) +
  (second_group_customers * paintings_per_second_group_customer) +
  (third_group_customers * paintings_per_third_group_customer) = 36 :=
by
  sorry

end tracy_sold_paintings_l839_83952


namespace joe_initial_cars_l839_83983

theorem joe_initial_cars (x : ℕ) (h : x + 12 = 62) : x = 50 :=
by {
  sorry
}

end joe_initial_cars_l839_83983


namespace uncle_dave_ice_cream_sandwiches_l839_83944

theorem uncle_dave_ice_cream_sandwiches (n : ℕ) (s : ℕ) (total : ℕ) 
  (h1 : n = 11) (h2 : s = 13) (h3 : total = n * s) : total = 143 := by
  sorry

end uncle_dave_ice_cream_sandwiches_l839_83944


namespace find_length_of_AL_l839_83928

noncomputable def length_of_AL 
  (A B C L : ℝ) 
  (AB AC AL : ℝ)
  (BC : ℝ)
  (AB_ratio_AC : AB / AC = 5 / 2)
  (BAC_bisector : ∃k, L = k * BC)
  (vector_magnitude : (2 * AB + 5 * AC) = 2016) : Prop :=
  AL = 288

theorem find_length_of_AL 
  (A B C L : ℝ)
  (AB AC AL : ℝ)
  (BC : ℝ)
  (h1 : AB / AC = 5 / 2)
  (h2 : ∃k, L = k * BC)
  (h3 : (2 * AB + 5 * AC) = 2016) : length_of_AL A B C L AB AC AL BC h1 h2 h3 := sorry

end find_length_of_AL_l839_83928


namespace robbery_participants_l839_83946

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l839_83946


namespace max_intersections_quadrilateral_l839_83988

-- Define intersection properties
def max_intersections_side : ℕ := 2
def sides_of_quadrilateral : ℕ := 4

theorem max_intersections_quadrilateral : 
  (max_intersections_side * sides_of_quadrilateral) = 8 :=
by 
  -- The proof goes here
  sorry

end max_intersections_quadrilateral_l839_83988


namespace machine_worked_minutes_l839_83977

theorem machine_worked_minutes
  (shirts_today : ℕ)
  (rate : ℕ)
  (h1 : shirts_today = 8)
  (h2 : rate = 2) :
  (shirts_today / rate) = 4 :=
by
  sorry

end machine_worked_minutes_l839_83977


namespace monotonic_decreasing_interval_of_f_l839_83941

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval_of_f :
  { x : ℝ | x > Real.exp 1 } = {y : ℝ | ∀ ε > 0, (x : ℝ) → (0 < x → (f (x + ε) < f x) ∧ (f x < f (x + ε)))}
:=
sorry

end monotonic_decreasing_interval_of_f_l839_83941


namespace amusement_park_ticket_price_l839_83915

theorem amusement_park_ticket_price
  (num_people_weekday : ℕ)
  (num_people_saturday : ℕ)
  (num_people_sunday : ℕ)
  (total_people_week : ℕ)
  (total_revenue_week : ℕ)
  (people_per_day_weekday : num_people_weekday = 100)
  (people_saturday : num_people_saturday = 200)
  (people_sunday : num_people_sunday = 300)
  (total_people : total_people_week = 1000)
  (total_revenue : total_revenue_week = 3000)
  (total_people_calc : 5 * num_people_weekday + num_people_saturday + num_people_sunday = total_people_week)
  (revenue_eq : total_people_week * 3 = total_revenue_week) :
  3 = 3 :=
by
  sorry

end amusement_park_ticket_price_l839_83915


namespace more_action_figures_than_books_l839_83910

-- Definitions of initial conditions
def books : ℕ := 3
def initial_action_figures : ℕ := 4
def added_action_figures : ℕ := 2

-- Definition of final number of action figures
def final_action_figures : ℕ := initial_action_figures + added_action_figures

-- Proposition to be proved
theorem more_action_figures_than_books : final_action_figures - books = 3 := by
  -- We leave the proof empty
  sorry

end more_action_figures_than_books_l839_83910


namespace sum_of_y_coordinates_of_other_vertices_l839_83955

theorem sum_of_y_coordinates_of_other_vertices
  (A B : ℝ × ℝ)
  (C D : ℝ × ℝ)
  (hA : A = (2, 15))
  (hB : B = (8, -2))
  (h_mid : midpoint ℝ A B = midpoint ℝ C D) :
  C.snd + D.snd = 13 := 
sorry

end sum_of_y_coordinates_of_other_vertices_l839_83955
