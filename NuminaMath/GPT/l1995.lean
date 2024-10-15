import Mathlib

namespace NUMINAMATH_GPT_second_player_can_ensure_symmetry_l1995_199507

def is_symmetric (seq : List ℕ) : Prop :=
  seq.reverse = seq

def swap_digits (seq : List ℕ) (i j : ℕ) : List ℕ :=
  if h : i < seq.length ∧ j < seq.length then
    seq.mapIdx (λ k x => if k = i then seq.get ⟨j, h.2⟩ 
                        else if k = j then seq.get ⟨i, h.1⟩ 
                        else x)
  else seq

theorem second_player_can_ensure_symmetry (seq : List ℕ) (h : seq.length = 1999) :
  (∃ swappable_seq : List ℕ, is_symmetric swappable_seq) :=
by
  sorry

end NUMINAMATH_GPT_second_player_can_ensure_symmetry_l1995_199507


namespace NUMINAMATH_GPT_quadratic_equation_in_one_variable_l1995_199540

-- Definitions for each condition
def equation_A (x : ℝ) : Prop := x^2 = -1
def equation_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def equation_C (x : ℝ) : Prop := 2 * (x + 1)^2 = (Real.sqrt 2 * x - 1)^2
def equation_D (x : ℝ) : Prop := x + 1 / x = 1

-- Main theorem statement
theorem quadratic_equation_in_one_variable (x : ℝ) :
  equation_A x ∧ ¬(∃ a b c, equation_B a b c x ∧ a ≠ 0) ∧ ¬equation_C x ∧ ¬equation_D x :=
  sorry

end NUMINAMATH_GPT_quadratic_equation_in_one_variable_l1995_199540


namespace NUMINAMATH_GPT_total_volume_of_water_l1995_199568

-- Define the conditions
def volume_of_hemisphere : ℕ := 4
def number_of_hemispheres : ℕ := 2734

-- Define the total volume
def total_volume : ℕ := volume_of_hemisphere * number_of_hemispheres

-- State the theorem
theorem total_volume_of_water : total_volume = 10936 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_total_volume_of_water_l1995_199568


namespace NUMINAMATH_GPT_problem_statement_eq_l1995_199531

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem problem_statement_eq :
  dollar ((x + y) ^ 2) ((y + x) ^ 2) = 0 := by
  sorry

end NUMINAMATH_GPT_problem_statement_eq_l1995_199531


namespace NUMINAMATH_GPT_number_of_men_l1995_199566

variable (W D X : ℝ)

theorem number_of_men (M_eq_2W : M = 2 * W)
  (wages_40_women : 21600 = 40 * W * D)
  (men_wages : 14400 = X * M * 20) :
  X = (2 / 3) * D :=
  by
  sorry

end NUMINAMATH_GPT_number_of_men_l1995_199566


namespace NUMINAMATH_GPT_max_sum_xy_l1995_199567

theorem max_sum_xy (x y : ℤ) (h1 : x^2 + y^2 = 64) (h2 : x ≥ 0) (h3 : y ≥ 0) : x + y ≤ 8 :=
by sorry

end NUMINAMATH_GPT_max_sum_xy_l1995_199567


namespace NUMINAMATH_GPT_students_not_enrolled_in_biology_class_l1995_199544

theorem students_not_enrolled_in_biology_class (total_students : ℕ) (percent_biology : ℕ) 
  (h1 : total_students = 880) (h2 : percent_biology = 35) : 
  total_students - (percent_biology * total_students / 100) = 572 := by
  sorry

end NUMINAMATH_GPT_students_not_enrolled_in_biology_class_l1995_199544


namespace NUMINAMATH_GPT_statement_correctness_l1995_199508

def correct_statements := [4, 8]
def incorrect_statements := [1, 2, 3, 5, 6, 7]

theorem statement_correctness :
  correct_statements = [4, 8] ∧ incorrect_statements = [1, 2, 3, 5, 6, 7] :=
  by sorry

end NUMINAMATH_GPT_statement_correctness_l1995_199508


namespace NUMINAMATH_GPT_not_sum_of_squares_of_form_4m_plus_3_l1995_199517

theorem not_sum_of_squares_of_form_4m_plus_3 (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ a b : ℤ, n = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_not_sum_of_squares_of_form_4m_plus_3_l1995_199517


namespace NUMINAMATH_GPT_orange_ring_weight_l1995_199576

theorem orange_ring_weight :
  ∀ (p w t o : ℝ), 
  p = 0.33 → w = 0.42 → t = 0.83 → t - (p + w) = o → 
  o = 0.08 :=
by
  intro p w t o hp hw ht h
  rw [hp, hw, ht] at h
  -- Additional steps would go here, but
  sorry -- Skipping the proof as instructed

end NUMINAMATH_GPT_orange_ring_weight_l1995_199576


namespace NUMINAMATH_GPT_divisibility_by_3_divisibility_by_4_l1995_199543

-- Proof that 5n^2 + 10n + 8 is divisible by 3 if and only if n ≡ 2 (mod 3)
theorem divisibility_by_3 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 3 = 0 ↔ n % 3 = 2 := 
    sorry

-- Proof that 5n^2 + 10n + 8 is divisible by 4 if and only if n ≡ 0 (mod 2)
theorem divisibility_by_4 (n : ℤ) : (5 * n^2 + 10 * n + 8) % 4 = 0 ↔ n % 2 = 0 :=
    sorry

end NUMINAMATH_GPT_divisibility_by_3_divisibility_by_4_l1995_199543


namespace NUMINAMATH_GPT_percentage_of_hexagon_area_is_closest_to_17_l1995_199515

noncomputable def tiling_area_hexagon_percentage : Real :=
  let total_area := 2 * 3
  let square_area := 1 * 1 
  let squares_count := 5 -- Adjusted count from 8 to fit total area properly
  let square_total_area := squares_count * square_area
  let hexagon_area := total_area - square_total_area
  let percentage := (hexagon_area / total_area) * 100
  percentage

theorem percentage_of_hexagon_area_is_closest_to_17 :
  abs (tiling_area_hexagon_percentage - 17) < 1 :=
sorry

end NUMINAMATH_GPT_percentage_of_hexagon_area_is_closest_to_17_l1995_199515


namespace NUMINAMATH_GPT_plants_same_height_after_54_years_l1995_199542

noncomputable def h1 (t : ℝ) : ℝ := 44 + (3 / 2) * t
noncomputable def h2 (t : ℝ) : ℝ := 80 + (5 / 6) * t

theorem plants_same_height_after_54_years :
  ∃ t : ℝ, h1 t = h2 t :=
by
  use 54
  sorry

end NUMINAMATH_GPT_plants_same_height_after_54_years_l1995_199542


namespace NUMINAMATH_GPT_simplify_evaluate_expr_l1995_199502

noncomputable def expr (x : ℝ) : ℝ := 
  ( ( (x^2 - 3) / (x + 2) - x + 2 ) / ( (x^2 - 4) / (x^2 + 4*x + 4) ) )

theorem simplify_evaluate_expr : 
  expr (Real.sqrt 2 + 1) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_GPT_simplify_evaluate_expr_l1995_199502


namespace NUMINAMATH_GPT_apples_left_over_l1995_199593

-- Defining the number of apples collected by Liam, Mia, and Noah
def liam_apples := 53
def mia_apples := 68
def noah_apples := 22

-- The total number of apples collected
def total_apples := liam_apples + mia_apples + noah_apples

-- Proving that the remainder when the total number of apples is divided by 10 is 3
theorem apples_left_over : total_apples % 10 = 3 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_apples_left_over_l1995_199593


namespace NUMINAMATH_GPT_wine_problem_solution_l1995_199501

theorem wine_problem_solution (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 200) (h2 : (200 - x) * (180 - x) / 200 = 144) : x = 20 := 
by
  sorry

end NUMINAMATH_GPT_wine_problem_solution_l1995_199501


namespace NUMINAMATH_GPT_count_perfect_squares_l1995_199536

theorem count_perfect_squares (N : Nat) :
  ∃ k : Nat, k = 1666 ∧ ∀ m, (∃ n, m = n * n ∧ m < 10^8 ∧ 36 ∣ m) ↔ (m = 36 * k ^ 2 ∧ k < 10^4) :=
sorry

end NUMINAMATH_GPT_count_perfect_squares_l1995_199536


namespace NUMINAMATH_GPT_geometric_sequence_a7_l1995_199580

theorem geometric_sequence_a7 (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = 3)
  (h2 : a 2 + a 3 = 6)
  (h_geometric : ∀ n, a (n + 1) = q * a n) :
  a 7 = 64 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a7_l1995_199580


namespace NUMINAMATH_GPT_reciprocal_of_neg_five_l1995_199506

theorem reciprocal_of_neg_five : (1 / (-5 : ℝ)) = -1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_five_l1995_199506


namespace NUMINAMATH_GPT_force_of_water_pressure_on_plate_l1995_199503

noncomputable def force_on_plate_under_water (γ : ℝ) (g : ℝ) (a b : ℝ) : ℝ :=
  γ * g * (b^2 - a^2) / 2

theorem force_of_water_pressure_on_plate :
  let γ : ℝ := 1000 -- kg/m^3
  let g : ℝ := 9.81  -- m/s^2
  let a : ℝ := 0.5   -- top depth
  let b : ℝ := 2.5   -- bottom depth
  force_on_plate_under_water γ g a b = 29430 := sorry

end NUMINAMATH_GPT_force_of_water_pressure_on_plate_l1995_199503


namespace NUMINAMATH_GPT_smallest_root_abs_eq_six_l1995_199560

theorem smallest_root_abs_eq_six : 
  (∃ x : ℝ, (abs (x - 1)) / (x^2) = 6 ∧ ∀ y : ℝ, (abs (y - 1)) / (y^2) = 6 → y ≥ x) → x = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_smallest_root_abs_eq_six_l1995_199560


namespace NUMINAMATH_GPT_total_copies_l1995_199551

-- Conditions: Defining the rates of two copy machines and the time duration
def rate1 : ℕ := 35 -- rate in copies per minute for the first machine
def rate2 : ℕ := 65 -- rate in copies per minute for the second machine
def time : ℕ := 30 -- time in minutes

-- The theorem stating that the total number of copies made by both machines in 30 minutes is 3000
theorem total_copies : rate1 * time + rate2 * time = 3000 := by
  sorry

end NUMINAMATH_GPT_total_copies_l1995_199551


namespace NUMINAMATH_GPT_brandon_skittles_final_l1995_199549
-- Conditions
def brandon_initial_skittles := 96
def brandon_lost_skittles := 9

-- Theorem stating the question and answer
theorem brandon_skittles_final : brandon_initial_skittles - brandon_lost_skittles = 87 := 
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_brandon_skittles_final_l1995_199549


namespace NUMINAMATH_GPT_total_handshakes_l1995_199535

theorem total_handshakes (total_people : ℕ) (first_meeting_people : ℕ) (second_meeting_new_people : ℕ) (common_people : ℕ)
  (total_people_is : total_people = 12)
  (first_meeting_people_is : first_meeting_people = 7)
  (second_meeting_new_people_is : second_meeting_new_people = 5)
  (common_people_is : common_people = 2)
  (first_meeting_handshakes : ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2)
  (second_meeting_handshakes: ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2 - (common_people * (common_people - 1)) / 2):
  first_meeting_handshakes + second_meeting_handshakes = 41 := 
sorry

end NUMINAMATH_GPT_total_handshakes_l1995_199535


namespace NUMINAMATH_GPT_count_integers_between_3250_and_3500_with_increasing_digits_l1995_199528

theorem count_integers_between_3250_and_3500_with_increasing_digits :
  ∃ n : ℕ, n = 20 ∧
    (∀ x : ℕ, 3250 ≤ x ∧ x ≤ 3500 →
      ∀ (d1 d2 d3 d4 : ℕ),
        d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧
        (x = d1 * 1000 + d2 * 100 + d3 * 10 + d4) →
        (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4)) :=
  sorry

end NUMINAMATH_GPT_count_integers_between_3250_and_3500_with_increasing_digits_l1995_199528


namespace NUMINAMATH_GPT_smallest_value_of_m_plus_n_l1995_199572

theorem smallest_value_of_m_plus_n :
  ∃ m n : ℕ, 1 < m ∧ 
  (∃ l : ℝ, l = (m^2 - 1 : ℝ) / (m * n) ∧ l = 1 / 2021) ∧
  m + n = 85987 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_m_plus_n_l1995_199572


namespace NUMINAMATH_GPT_scientific_notation_correct_l1995_199538

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let a := x * 10^9
  (a, -9)

theorem scientific_notation_correct :
  scientific_notation 0.000000007 = (7, -9) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1995_199538


namespace NUMINAMATH_GPT_min_value_a_plus_b_l1995_199587

theorem min_value_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (2 / a) + (2 / b) = 1) :
  a + b >= 8 :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_b_l1995_199587


namespace NUMINAMATH_GPT_hare_wins_by_10_meters_l1995_199509

def speed_tortoise := 3 -- meters per minute
def speed_hare_sprint := 12 -- meters per minute
def speed_hare_walk := 1 -- meters per minute
def time_total := 50 -- minutes
def time_hare_sprint := 10 -- minutes
def time_hare_walk := time_total - time_hare_sprint -- minutes

def distance_tortoise := speed_tortoise * time_total -- meters
def distance_hare := (speed_hare_sprint * time_hare_sprint) + (speed_hare_walk * time_hare_walk) -- meters

theorem hare_wins_by_10_meters : (distance_hare - distance_tortoise) = 10 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_hare_wins_by_10_meters_l1995_199509


namespace NUMINAMATH_GPT_compute_x_l1995_199562

theorem compute_x :
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = (∑' n : ℕ, 1 / (9^n)) →
  (∑' n : ℕ, (1 / (3^n)) * (1 / (3^n) * (-1)^n)) = 1 / (1 - (1 / 9)) →
  9 = 9 :=
by
  sorry

end NUMINAMATH_GPT_compute_x_l1995_199562


namespace NUMINAMATH_GPT_faye_initial_coloring_books_l1995_199588

theorem faye_initial_coloring_books (gave_away1 gave_away2 remaining : ℝ) 
    (h1 : gave_away1 = 34.0) (h2 : gave_away2 = 3.0) (h3 : remaining = 11.0) :
    gave_away1 + gave_away2 + remaining = 48.0 := 
by
  sorry

end NUMINAMATH_GPT_faye_initial_coloring_books_l1995_199588


namespace NUMINAMATH_GPT_derivative_of_f_l1995_199530

noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

theorem derivative_of_f (x : ℝ) (hx : x ≠ 0) : deriv f x = ((x * Real.exp x - Real.exp x) / (x * x)) :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_f_l1995_199530


namespace NUMINAMATH_GPT_janet_spending_difference_l1995_199575

-- Defining hourly rates and weekly hours for each type of lessons
def clarinet_hourly_rate := 40
def clarinet_weekly_hours := 3
def piano_hourly_rate := 28
def piano_weekly_hours := 5
def violin_hourly_rate := 35
def violin_weekly_hours := 2
def singing_hourly_rate := 45
def singing_weekly_hours := 1

-- Calculating weekly costs
def clarinet_weekly_cost := clarinet_hourly_rate * clarinet_weekly_hours
def piano_weekly_cost := piano_hourly_rate * piano_weekly_hours
def violin_weekly_cost := violin_hourly_rate * violin_weekly_hours
def singing_weekly_cost := singing_hourly_rate * singing_weekly_hours
def combined_weekly_cost := piano_weekly_cost + violin_weekly_cost + singing_weekly_cost

-- Calculating annual costs with 52 weeks in a year
def weeks_per_year := 52
def clarinet_annual_cost := clarinet_weekly_cost * weeks_per_year
def combined_annual_cost := combined_weekly_cost * weeks_per_year

-- Proving the final statement
theorem janet_spending_difference :
  combined_annual_cost - clarinet_annual_cost = 7020 := by sorry

end NUMINAMATH_GPT_janet_spending_difference_l1995_199575


namespace NUMINAMATH_GPT_freshmen_count_l1995_199541

theorem freshmen_count (n : ℕ) (h1 : n < 600) (h2 : n % 17 = 16) (h3 : n % 19 = 18) : n = 322 := 
by 
  sorry

end NUMINAMATH_GPT_freshmen_count_l1995_199541


namespace NUMINAMATH_GPT_smallest_constant_l1995_199518

theorem smallest_constant (D : ℝ) :
  (∀ (x y : ℝ), x^2 + 2*y^2 + 5 ≥ D*(2*x + 3*y) + 4) → D ≤ Real.sqrt (8 / 17) :=
by
  intros
  sorry

end NUMINAMATH_GPT_smallest_constant_l1995_199518


namespace NUMINAMATH_GPT_binary_division_remainder_l1995_199569

theorem binary_division_remainder (n : ℕ) (h_n : n = 0b110110011011) : n % 8 = 3 :=
by {
  -- This sorry statement skips the actual proof
  sorry
}

end NUMINAMATH_GPT_binary_division_remainder_l1995_199569


namespace NUMINAMATH_GPT_new_person_weight_l1995_199537

noncomputable def weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

theorem new_person_weight 
  (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) 
  (weight_eqn : weight_increase n avg_increase = new_weight - old_weight) : 
  new_weight = 87.5 :=
by
  have n := 9
  have avg_increase := 2.5
  have old_weight := 65
  have weight_increase := 9 * 2.5
  have weight_eqn := weight_increase = 87.5 - 65
  sorry

end NUMINAMATH_GPT_new_person_weight_l1995_199537


namespace NUMINAMATH_GPT_common_chord_length_l1995_199511

theorem common_chord_length (x y : ℝ) : 
    (x^2 + y^2 = 4) → 
    (x^2 + y^2 - 4*x + 4*y - 12 = 0) → 
    ∃ l : ℝ, l = 2 * Real.sqrt 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_common_chord_length_l1995_199511


namespace NUMINAMATH_GPT_find_b_l1995_199586

-- Definitions
variable (k : ℤ) (b : ℤ)
def x := 3 * k
def y := 4 * k
def z := 7 * k

-- Conditions
axiom ratio : x / y = 3 / 4 ∧ y / z = 4 / 7
axiom equation : y = 15 * b - 5

-- Theorem statement
theorem find_b : ∃ b : ℤ, 4 * k = 15 * b - 5 ∧ b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1995_199586


namespace NUMINAMATH_GPT_angle_F_measure_l1995_199559

theorem angle_F_measure (α β γ : ℝ) (hD : α = 84) (hAngleSum : α + β + γ = 180) (hBeta : β = 4 * γ + 18) :
  γ = 15.6 := by
  sorry

end NUMINAMATH_GPT_angle_F_measure_l1995_199559


namespace NUMINAMATH_GPT_second_person_time_l1995_199525

theorem second_person_time (x : ℝ) (h1 : ∀ t : ℝ, t = 3) 
(h2 : (1/3 + 1/x) = 5/12) : x = 12 := 
by sorry

end NUMINAMATH_GPT_second_person_time_l1995_199525


namespace NUMINAMATH_GPT_additional_cost_per_kg_l1995_199523

theorem additional_cost_per_kg (l a : ℝ) 
  (h1 : 30 * l + 3 * a = 333) 
  (h2 : 30 * l + 6 * a = 366) 
  (h3 : 15 * l = 150) 
  : a = 11 := 
by
  sorry

end NUMINAMATH_GPT_additional_cost_per_kg_l1995_199523


namespace NUMINAMATH_GPT_solve_exponential_equation_l1995_199570

theorem solve_exponential_equation :
  ∃ x, (2:ℝ)^(2*x) - 8 * (2:ℝ)^x + 12 = 0 ↔ x = 1 ∨ x = 1 + Real.log 3 / Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_exponential_equation_l1995_199570


namespace NUMINAMATH_GPT_base_six_digits_unique_l1995_199510

theorem base_six_digits_unique (b : ℕ) (h : (b-1)^2*(b-2) = 100) : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_base_six_digits_unique_l1995_199510


namespace NUMINAMATH_GPT_sale_price_of_sarees_l1995_199524

theorem sale_price_of_sarees 
  (P : ℝ) 
  (d1 d2 d3 d4 tax_rate : ℝ) 
  (P_initial : P = 510) 
  (d1_val : d1 = 0.12) 
  (d2_val : d2 = 0.15) 
  (d3_val : d3 = 0.20) 
  (d4_val : d4 = 0.10) 
  (tax_val : tax_rate = 0.10) :
  let discount_step (price discount : ℝ) := price * (1 - discount)
  let tax_step (price tax_rate : ℝ) := price * (1 + tax_rate)
  let P1 := discount_step P d1
  let P2 := discount_step P1 d2
  let P3 := discount_step P2 d3
  let P4 := discount_step P3 d4
  let final_price := tax_step P4 tax_rate
  abs (final_price - 302.13) < 0.01 := 
sorry

end NUMINAMATH_GPT_sale_price_of_sarees_l1995_199524


namespace NUMINAMATH_GPT_f_f_of_2_l1995_199594

def f (x : ℤ) : ℤ := 4 * x ^ 3 - 3 * x + 1

theorem f_f_of_2 : f (f 2) = 78652 := 
by
  sorry

end NUMINAMATH_GPT_f_f_of_2_l1995_199594


namespace NUMINAMATH_GPT_kids_played_on_monday_l1995_199533

theorem kids_played_on_monday (m t a : Nat) (h1 : t = 7) (h2 : a = 19) (h3 : a = m + t) : m = 12 := 
by 
  sorry

end NUMINAMATH_GPT_kids_played_on_monday_l1995_199533


namespace NUMINAMATH_GPT_find_fraction_l1995_199548

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_fraction_l1995_199548


namespace NUMINAMATH_GPT_number_of_routes_from_A_to_B_l1995_199504

-- Define the grid dimensions
def grid_rows : ℕ := 3
def grid_columns : ℕ := 2

-- Define the total number of steps needed to travel from A to B
def total_steps : ℕ := grid_rows + grid_columns

-- Define the number of right moves (R) and down moves (D)
def right_moves : ℕ := grid_rows
def down_moves : ℕ := grid_columns

-- Calculate the number of different routes using combination formula
def number_of_routes : ℕ := Nat.choose total_steps right_moves

-- The main statement to be proven
theorem number_of_routes_from_A_to_B : number_of_routes = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_routes_from_A_to_B_l1995_199504


namespace NUMINAMATH_GPT_min_sum_a_b_l1995_199522

theorem min_sum_a_b {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : 1/a + 9/b = 1) : a + b ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_sum_a_b_l1995_199522


namespace NUMINAMATH_GPT_class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l1995_199595

theorem class_contribution_Miss_Evans :
  let total_contribution : ℝ := 90
  let class_funds_Evans : ℝ := 14
  let num_students_Evans : ℕ := 19
  let individual_contribution_Evans : ℝ := (total_contribution - class_funds_Evans) / num_students_Evans
  individual_contribution_Evans = 4 := 
sorry

theorem class_contribution_Mr_Smith :
  let total_contribution : ℝ := 90
  let class_funds_Smith : ℝ := 20
  let num_students_Smith : ℕ := 15
  let individual_contribution_Smith : ℝ := (total_contribution - class_funds_Smith) / num_students_Smith
  individual_contribution_Smith = 4.67 := 
sorry

theorem class_contribution_Mrs_Johnson :
  let total_contribution : ℝ := 90
  let class_funds_Johnson : ℝ := 30
  let num_students_Johnson : ℕ := 25
  let individual_contribution_Johnson : ℝ := (total_contribution - class_funds_Johnson) / num_students_Johnson
  individual_contribution_Johnson = 2.40 := 
sorry

end NUMINAMATH_GPT_class_contribution_Miss_Evans_class_contribution_Mr_Smith_class_contribution_Mrs_Johnson_l1995_199595


namespace NUMINAMATH_GPT_roots_sum_equality_l1995_199571

theorem roots_sum_equality {a b c : ℝ} {x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ} :
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 1 = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
  (∀ x, x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x - 2 = 0 → x = y₁ ∨ x = y₂ ∨ x = y₃ ∨ x = y₄) →
  x₁ + x₂ = x₃ + x_₄ →
  y₁ + y₂ = y₃ + y₄ :=
sorry

end NUMINAMATH_GPT_roots_sum_equality_l1995_199571


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l1995_199500

theorem quadratic_distinct_real_roots (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + c = 0 ∧ y^2 - 2*y + c = 0) ↔ c < 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l1995_199500


namespace NUMINAMATH_GPT_percentage_difference_l1995_199513

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) : ((x - y) / x) * 100 = 66.67 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1995_199513


namespace NUMINAMATH_GPT_Susie_possible_values_l1995_199579

theorem Susie_possible_values (n : ℕ) (h1 : n > 43) (h2 : 2023 % n = 43) : 
  (∃ count : ℕ, count = 19 ∧ ∀ n, n > 43 ∧ 2023 % n = 43 → 1980 ∣ (2023 - 43)) :=
sorry

end NUMINAMATH_GPT_Susie_possible_values_l1995_199579


namespace NUMINAMATH_GPT_interest_rate_l1995_199583

theorem interest_rate (P CI SI: ℝ) (r: ℝ) : P = 5100 → CI = P * (1 + r)^2 - P → SI = P * r * 2 → (CI - SI = 51) → r = 0.1 :=
by
  intros
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_interest_rate_l1995_199583


namespace NUMINAMATH_GPT_find_a_and_b_l1995_199598

theorem find_a_and_b (a b : ℚ) (h : ∀ (n : ℕ), 1 / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) : 
  a = 1/2 ∧ b = -1/2 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_and_b_l1995_199598


namespace NUMINAMATH_GPT_arctan_sum_pi_div_two_l1995_199584

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_arctan_sum_pi_div_two_l1995_199584


namespace NUMINAMATH_GPT_distinct_divisor_sum_l1995_199585

theorem distinct_divisor_sum (n : ℕ) (x : ℕ) (h : x < n.factorial) :
  ∃ (k : ℕ) (d : Fin k → ℕ), (k ≤ n) ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n.factorial) ∧ (x = Finset.sum Finset.univ d) :=
sorry

end NUMINAMATH_GPT_distinct_divisor_sum_l1995_199585


namespace NUMINAMATH_GPT_no_such_functions_exist_l1995_199565

open Function

theorem no_such_functions_exist : ¬ (∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3) := 
sorry

end NUMINAMATH_GPT_no_such_functions_exist_l1995_199565


namespace NUMINAMATH_GPT_update_year_l1995_199545

def a (n : ℕ) : ℕ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5 / 4) ^ (n - 7)

noncomputable def S (n : ℕ) : ℕ :=
  if n ≤ 7 then n^2 + 3 * n else 80 * ((5 / 4) ^ (n - 7)) - 10

noncomputable def avg_maintenance_cost (n : ℕ) : ℚ :=
  (S n : ℚ) / n

theorem update_year (n : ℕ) (h : avg_maintenance_cost n > 12) : n = 9 :=
  by
  sorry

end NUMINAMATH_GPT_update_year_l1995_199545


namespace NUMINAMATH_GPT_min_a_b_l1995_199591

theorem min_a_b (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 45 * a + b = 2021) : a + b = 85 :=
sorry

end NUMINAMATH_GPT_min_a_b_l1995_199591


namespace NUMINAMATH_GPT_total_nails_polished_l1995_199558

-- Defining the number of girls
def num_girls : ℕ := 5

-- Defining the number of fingers and toes per person
def num_fingers_per_person : ℕ := 10
def num_toes_per_person : ℕ := 10

-- Defining the total number of nails per person
def nails_per_person : ℕ := num_fingers_per_person + num_toes_per_person

-- The theorem stating that the total number of nails polished for 5 girls is 100 nails
theorem total_nails_polished : num_girls * nails_per_person = 100 := by
  sorry

end NUMINAMATH_GPT_total_nails_polished_l1995_199558


namespace NUMINAMATH_GPT_laundry_loads_l1995_199526

-- Conditions
def wash_time_per_load : ℕ := 45 -- in minutes
def dry_time_per_load : ℕ := 60 -- in minutes
def total_time : ℕ := 14 -- in hours

theorem laundry_loads (L : ℕ) 
  (h1 : total_time = 14)
  (h2 : total_time * 60 = L * (wash_time_per_load + dry_time_per_load)) :
  L = 8 :=
by
  sorry

end NUMINAMATH_GPT_laundry_loads_l1995_199526


namespace NUMINAMATH_GPT_sum_of_squares_l1995_199582

theorem sum_of_squares (a b c d : ℝ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : d = a + 3) :
  a^2 + b^2 = c^2 + d^2 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1995_199582


namespace NUMINAMATH_GPT_hudson_daily_burger_spending_l1995_199512

-- Definitions based on conditions
def total_spent := 465
def days_in_december := 31

-- Definition of the question
def amount_spent_per_day := total_spent / days_in_december

-- The theorem to prove
theorem hudson_daily_burger_spending : amount_spent_per_day = 15 := by
  sorry

end NUMINAMATH_GPT_hudson_daily_burger_spending_l1995_199512


namespace NUMINAMATH_GPT_sequence_equality_l1995_199578

theorem sequence_equality (a : ℕ → ℤ) (h : ∀ n, a (n + 2) ^ 2 + a (n + 1) * a n ≤ a (n + 2) * (a (n + 1) + a n)) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 2) = a n :=
by sorry

end NUMINAMATH_GPT_sequence_equality_l1995_199578


namespace NUMINAMATH_GPT_number_of_lightsabers_in_order_l1995_199564

-- Let's define the given conditions
def metal_arcs_per_lightsaber : ℕ := 2
def cost_per_metal_arc : ℕ := 400
def apparatus_production_rate : ℕ := 20 -- lightsabers per hour
def combined_app_expense_rate : ℕ := 300 -- units per hour
def total_order_cost : ℕ := 65200
def lightsaber_cost : ℕ := metal_arcs_per_lightsaber * cost_per_metal_arc + (combined_app_expense_rate / apparatus_production_rate)

-- Define the main theorem to prove
theorem number_of_lightsabers_in_order : 
  (total_order_cost / lightsaber_cost) = 80 :=
by
  sorry

end NUMINAMATH_GPT_number_of_lightsabers_in_order_l1995_199564


namespace NUMINAMATH_GPT_tissue_magnification_l1995_199573

theorem tissue_magnification (d_image d_actual : ℝ) (h_image : d_image = 0.3) (h_actual : d_actual = 0.0003) :
  (d_image / d_actual) = 1000 :=
by
  sorry

end NUMINAMATH_GPT_tissue_magnification_l1995_199573


namespace NUMINAMATH_GPT_find_number_l1995_199599

theorem find_number (x : ℤ) (h : 4 * x - 7 = 13) : x = 5 := 
sorry

end NUMINAMATH_GPT_find_number_l1995_199599


namespace NUMINAMATH_GPT_geometric_sequence_example_l1995_199546

theorem geometric_sequence_example
  (a : ℕ → ℝ)
  (h1 : ∀ n, 0 < a n)
  (h2 : ∃ r, ∀ n, a (n + 1) = r * a n)
  (h3 : Real.log (a 2) / Real.log 2 + Real.log (a 8) / Real.log 2 = 1) :
  a 3 * a 7 = 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_example_l1995_199546


namespace NUMINAMATH_GPT_container_capacity_in_liters_l1995_199597

-- Defining the conditions
def portions : Nat := 10
def portion_size_ml : Nat := 200

-- Statement to prove
theorem container_capacity_in_liters : (portions * portion_size_ml / 1000 = 2) :=
by 
  sorry

end NUMINAMATH_GPT_container_capacity_in_liters_l1995_199597


namespace NUMINAMATH_GPT_range_of_a_in_triangle_l1995_199527

open Real

noncomputable def law_of_sines_triangle (A B C : ℝ) (a b c : ℝ) :=
  sin A / a = sin B / b ∧ sin B / b = sin C / c

theorem range_of_a_in_triangle (b : ℝ) (B : ℝ) (a : ℝ) (h1 : b = 2) (h2 : B = pi / 4) (h3 : true) :
  2 < a ∧ a < 2 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_in_triangle_l1995_199527


namespace NUMINAMATH_GPT_cyclic_cosine_inequality_l1995_199557

theorem cyclic_cosine_inequality
  (α β γ : ℝ)
  (hα : 0 ≤ α ∧ α ≤ π / 2)
  (hβ : 0 ≤ β ∧ β ≤ π / 2)
  (hγ : 0 ≤ γ ∧ γ ≤ π / 2)
  (cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  2 ≤ (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4 ∧
    (1 + Real.cos α ^ 2) ^ 2 * (Real.sin α) ^ 4
       + (1 + Real.cos β ^ 2) ^ 2 * (Real.sin β) ^ 4
       + (1 + Real.cos γ ^ 2) ^ 2 * (Real.sin γ) ^ 4
      ≤ (1 + Real.cos α ^ 2) * (1 + Real.cos β ^ 2) * (1 + Real.cos γ ^ 2) :=
by 
  sorry

end NUMINAMATH_GPT_cyclic_cosine_inequality_l1995_199557


namespace NUMINAMATH_GPT_squares_expression_l1995_199555

theorem squares_expression (a : ℕ) : 
  a^2 + 5*a + 7 = (a+3) * (a+2)^2 + (a+2) * 1^2 := 
by
  sorry

end NUMINAMATH_GPT_squares_expression_l1995_199555


namespace NUMINAMATH_GPT_K1K2_eq_one_over_four_l1995_199561

theorem K1K2_eq_one_over_four
  (K1 : ℝ) (hK1 : K1 ≠ 0)
  (K2 : ℝ)
  (x1 y1 x2 y2 : ℝ)
  (hx1y1 : x1^2 - 4 * y1^2 = 4)
  (hx2y2 : x2^2 - 4 * y2^2 = 4)
  (hx0 : x0 = (x1 + x2) / 2)
  (hy0 : y0 = (y1 + y2) / 2)
  (K1_eq : K1 = (y1 - y2) / (x1 - x2))
  (K2_eq : K2 = y0 / x0) :
  K1 * K2 = 1 / 4 :=
sorry

end NUMINAMATH_GPT_K1K2_eq_one_over_four_l1995_199561


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1995_199592

variable {a : ℕ → ℝ} -- Sequence terms
variable {S : ℕ → ℝ} -- Sum of the first n terms

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n+1) = a n * q
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n, S n = a 0 * (1 - (a n)) / (1 - a 1)
def is_arithmetic_sequence (x y z : ℝ) := 2 * y = x + z
def term_1_equals_1 (a : ℕ → ℝ) := a 0 = 1

-- Question: Prove that given the conditions, S_5 = 31
theorem geometric_sequence_sum (q : ℝ) (h_geom : is_geometric_sequence a q) 
  (h_sum : sum_of_first_n_terms a S) (h_arith : is_arithmetic_sequence (4 * a 0) (2 * a 1) (a 2)) 
  (h_a1 : term_1_equals_1 a) : S 5 = 31 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1995_199592


namespace NUMINAMATH_GPT_mass_of_circle_is_one_l1995_199529

variable (x y z : ℝ)

theorem mass_of_circle_is_one (h1 : 3 * y = 2 * x)
                              (h2 : 2 * y = x + 1)
                              (h3 : 5 * z = x + y)
                              (h4 : true) : z = 1 :=
sorry

end NUMINAMATH_GPT_mass_of_circle_is_one_l1995_199529


namespace NUMINAMATH_GPT_clock_hand_speed_ratio_l1995_199550

theorem clock_hand_speed_ratio :
  (360 / 720 : ℝ) / (360 / 60 : ℝ) = (2 / 24 : ℝ) := by
    sorry

end NUMINAMATH_GPT_clock_hand_speed_ratio_l1995_199550


namespace NUMINAMATH_GPT_max_ab_value_l1995_199596

theorem max_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 4) : ab ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_ab_value_l1995_199596


namespace NUMINAMATH_GPT_distance_between_stations_l1995_199554

/-- Two trains start at the same time from two stations and proceed towards each other.
    Train 1 travels at 20 km/hr.
    Train 2 travels at 25 km/hr.
    When they meet, Train 2 has traveled 55 km more than Train 1.
    Prove that the distance between the two stations is 495 km. -/
theorem distance_between_stations : ∃ x t : ℕ, 20 * t = x ∧ 25 * t = x + 55 ∧ 2 * x + 55 = 495 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_between_stations_l1995_199554


namespace NUMINAMATH_GPT_spot_area_l1995_199532

/-- Proving the area of the accessible region outside the doghouse -/
theorem spot_area
  (pentagon_side : ℝ)
  (rope_length : ℝ)
  (accessible_area : ℝ) 
  (h1 : pentagon_side = 1) 
  (h2 : rope_length = 3)
  (h3 : accessible_area = (37 * π) / 5) :
  accessible_area = (π * (rope_length^2) * (288 / 360)) + 2 * (π * (pentagon_side^2) * (36 / 360)) := 
  sorry

end NUMINAMATH_GPT_spot_area_l1995_199532


namespace NUMINAMATH_GPT_Harriet_age_now_l1995_199552

variable (P H: ℕ)

theorem Harriet_age_now (P : ℕ) (H : ℕ) (h1 : P + 4 = 2 * (H + 4)) (h2 : P = 60 / 2) : H = 13 := by
  sorry

end NUMINAMATH_GPT_Harriet_age_now_l1995_199552


namespace NUMINAMATH_GPT_Clarence_total_oranges_l1995_199590

def Clarence_oranges_initial := 5
def oranges_from_Joyce := 3

theorem Clarence_total_oranges : Clarence_oranges_initial + oranges_from_Joyce = 8 := by
  sorry

end NUMINAMATH_GPT_Clarence_total_oranges_l1995_199590


namespace NUMINAMATH_GPT_range_j_l1995_199577

def h (x : ℝ) : ℝ := 4 * x - 3
def j (x : ℝ) : ℝ := h (h (h x))

theorem range_j : ∀ x, 0 ≤ x ∧ x ≤ 3 → -63 ≤ j x ∧ j x ≤ 129 :=
by
  intro x
  intro hx
  sorry

end NUMINAMATH_GPT_range_j_l1995_199577


namespace NUMINAMATH_GPT_walking_times_relationship_l1995_199519

theorem walking_times_relationship (x : ℝ) (h : x > 0) :
  (15 / x) - (15 / (x + 1)) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_walking_times_relationship_l1995_199519


namespace NUMINAMATH_GPT_smallest_constant_inequality_l1995_199521

open Real

theorem smallest_constant_inequality (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
    sqrt (x / (y + z + w)) + sqrt (y / (x + z + w)) + sqrt (z / (x + y + w)) + sqrt (w / (x + y + z)) ≤ 2 := by
  sorry

end NUMINAMATH_GPT_smallest_constant_inequality_l1995_199521


namespace NUMINAMATH_GPT_correct_equation_l1995_199556

variable (x : ℤ)
variable (cost_of_chickens : ℤ)

-- Condition 1: If each person contributes 9 coins, there will be an excess of 11 coins.
def condition1 : Prop := 9 * x - cost_of_chickens = 11

-- Condition 2: If each person contributes 6 coins, there will be a shortage of 16 coins.
def condition2 : Prop := 6 * x - cost_of_chickens = -16

-- The goal is to prove the correct equation given the conditions.
theorem correct_equation (h1 : condition1 (x) (cost_of_chickens)) (h2 : condition2 (x) (cost_of_chickens)) :
  9 * x - 11 = 6 * x + 16 :=
sorry

end NUMINAMATH_GPT_correct_equation_l1995_199556


namespace NUMINAMATH_GPT_volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l1995_199505

theorem volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron (R r : ℝ) (h : r = R / 3) : 
  (4/3 * π * r^3) / (4/3 * π * R^3) = 1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_inscribed_circumscribed_sphere_regular_tetrahedron_l1995_199505


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l1995_199581

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (Real.cos (2 * α), Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (1, 2 * Real.sin α - 1)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi)
    (h3 : dot_product (vec_a α) (vec_b α) = 0) :
    Real.tan (α + Real.pi / 4) = -1 := sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l1995_199581


namespace NUMINAMATH_GPT_solution_set_inequality_l1995_199547

theorem solution_set_inequality (x : ℝ) : 4 * x < 3 * x + 2 → x < 2 :=
by
  intro h
  -- Add actual proof here, but for now; we use sorry
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1995_199547


namespace NUMINAMATH_GPT_percent_increase_correct_l1995_199574

noncomputable def last_year_ticket_price : ℝ := 85
noncomputable def last_year_tax_rate : ℝ := 0.10
noncomputable def this_year_ticket_price : ℝ := 102
noncomputable def this_year_tax_rate : ℝ := 0.12
noncomputable def student_discount_rate : ℝ := 0.15

noncomputable def last_year_total_cost : ℝ := last_year_ticket_price * (1 + last_year_tax_rate)
noncomputable def discounted_ticket_price_this_year : ℝ := this_year_ticket_price * (1 - student_discount_rate)
noncomputable def total_cost_this_year : ℝ := discounted_ticket_price_this_year * (1 + this_year_tax_rate)

noncomputable def percent_increase : ℝ := ((total_cost_this_year - last_year_total_cost) / last_year_total_cost) * 100

theorem percent_increase_correct :
  abs (percent_increase - 3.854) < 0.001 := sorry

end NUMINAMATH_GPT_percent_increase_correct_l1995_199574


namespace NUMINAMATH_GPT_solve_nat_eqn_l1995_199539

theorem solve_nat_eqn (n k l m : ℕ) (hl : l > 1) 
  (h_eq : (1 + n^k)^l = 1 + n^m) : (n, k, l, m) = (2, 1, 2, 3) := 
sorry

end NUMINAMATH_GPT_solve_nat_eqn_l1995_199539


namespace NUMINAMATH_GPT_area_tripled_radius_increase_l1995_199534

theorem area_tripled_radius_increase (m r : ℝ) (h : (r + m)^2 = 3 * r^2) :
  r = m * (1 + Real.sqrt 3) / 2 :=
sorry

end NUMINAMATH_GPT_area_tripled_radius_increase_l1995_199534


namespace NUMINAMATH_GPT_machining_defect_probability_l1995_199589

theorem machining_defect_probability :
  let defect_rate_process1 := 0.03
  let defect_rate_process2 := 0.05
  let non_defective_rate_process1 := 1 - defect_rate_process1
  let non_defective_rate_process2 := 1 - defect_rate_process2
  let non_defective_rate := non_defective_rate_process1 * non_defective_rate_process2
  let defective_rate := 1 - non_defective_rate
  defective_rate = 0.0785 :=
by
  sorry

end NUMINAMATH_GPT_machining_defect_probability_l1995_199589


namespace NUMINAMATH_GPT_tetrahedron_cube_volume_ratio_l1995_199563

theorem tetrahedron_cube_volume_ratio (a : ℝ) :
  let V_tetrahedron := (a * Real.sqrt 2)^3 * Real.sqrt 2 / 12
  let V_cube := a^3
  (V_tetrahedron / V_cube) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_cube_volume_ratio_l1995_199563


namespace NUMINAMATH_GPT_spider_moves_away_from_bee_l1995_199553

noncomputable def bee : ℝ × ℝ := (14, 5)
noncomputable def spider_line (x : ℝ) : ℝ := -3 * x + 25
noncomputable def perpendicular_line (x : ℝ) : ℝ := (1 / 3) * x + 14 / 3

theorem spider_moves_away_from_bee : ∃ (c d : ℝ), 
  (d = spider_line c) ∧ (d = perpendicular_line c) ∧ c + d = 13.37 := 
sorry

end NUMINAMATH_GPT_spider_moves_away_from_bee_l1995_199553


namespace NUMINAMATH_GPT_points_four_units_away_l1995_199520

theorem points_four_units_away (x : ℚ) (h : |x| = 4) : x = -4 ∨ x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_points_four_units_away_l1995_199520


namespace NUMINAMATH_GPT_solve_for_m_l1995_199514

theorem solve_for_m : 
  ∀ m : ℝ, (3 * (-2) + 5 = -2 - m) → m = -1 :=
by
  intros m h
  sorry

end NUMINAMATH_GPT_solve_for_m_l1995_199514


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1995_199516

-- Define the conditions
def is_arithmetic_sequence (first_term last_term : ℕ) (terms : ℕ) : Prop :=
  ∃ (a l : ℕ) (n : ℕ), a = first_term ∧ l = last_term ∧ n = terms ∧ n > 1

-- State the theorem
theorem sum_of_arithmetic_sequence (a l n : ℕ) (h_arith: is_arithmetic_sequence 5 41 10):
  n = 10 ∧ a = 5 ∧ l = 41 → (n * (a + l) / 2) = 230 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1995_199516
