import Mathlib

namespace NUMINAMATH_GPT_youngsville_population_l76_7670

def initial_population : ℕ := 684
def increase_rate : ℝ := 0.25
def decrease_rate : ℝ := 0.40

theorem youngsville_population : 
  let increased_population := initial_population + ⌊increase_rate * ↑initial_population⌋
  let decreased_population := increased_population - ⌊decrease_rate * increased_population⌋
  decreased_population = 513 :=
by
  sorry

end NUMINAMATH_GPT_youngsville_population_l76_7670


namespace NUMINAMATH_GPT_rectangle_perimeter_change_l76_7679

theorem rectangle_perimeter_change :
  ∀ (a b : ℝ), 
  (2 * (a + b) = 2 * (1.3 * a + 0.8 * b)) →
  ((2 * (0.8 * a + 1.95 * b) - 2 * (a + b)) / (2 * (a + b)) = 0.1) :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_change_l76_7679


namespace NUMINAMATH_GPT_find_k_l76_7657

noncomputable def arithmetic_sum (n : ℕ) (a1 d : ℚ) : ℚ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem find_k 
  (a1 d : ℚ) (k : ℕ)
  (h1 : arithmetic_sum (k - 2) a1 d = -4)
  (h2 : arithmetic_sum k a1 d = 0)
  (h3 : arithmetic_sum (k + 2) a1 d = 8) :
  k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l76_7657


namespace NUMINAMATH_GPT_no_n_gt_1_divisibility_l76_7692

theorem no_n_gt_1_divisibility (n : ℕ) (h : n > 1) : ¬ (3 ^ (n - 1) + 5 ^ (n - 1)) ∣ (3 ^ n + 5 ^ n) :=
by
  sorry

end NUMINAMATH_GPT_no_n_gt_1_divisibility_l76_7692


namespace NUMINAMATH_GPT_bus_driver_hours_worked_last_week_l76_7601

-- Definitions for given conditions
def regular_rate : ℝ := 12
def passenger_rate : ℝ := 0.50
def overtime_rate_1 : ℝ := 1.5 * regular_rate
def overtime_rate_2 : ℝ := 2 * regular_rate
def total_compensation : ℝ := 1280
def total_passengers : ℝ := 350
def earnings_from_passengers : ℝ := total_passengers * passenger_rate
def earnings_from_hourly_rate : ℝ := total_compensation - earnings_from_passengers
def regular_hours : ℝ := 40
def first_tier_overtime_hours : ℝ := 5

-- Theorem to prove the number of hours worked is 67
theorem bus_driver_hours_worked_last_week :
  ∃ (total_hours : ℝ),
    total_hours = 67 ∧
    earnings_from_passengers = total_passengers * passenger_rate ∧
    earnings_from_hourly_rate = total_compensation - earnings_from_passengers ∧
    (∃ (overtime_hours : ℝ),
      (overtime_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2) ∧
      total_hours = regular_hours + first_tier_overtime_hours + (earnings_from_hourly_rate - (regular_hours * regular_rate) - (first_tier_overtime_hours * overtime_rate_1)) / overtime_rate_2 )
  :=
sorry

end NUMINAMATH_GPT_bus_driver_hours_worked_last_week_l76_7601


namespace NUMINAMATH_GPT_rhombus_min_rotation_l76_7638

theorem rhombus_min_rotation (α : ℝ) (h1 : α = 60) : ∃ θ, θ = 180 := 
by 
  -- The proof here will show that the minimum rotation angle is 180°
  sorry

end NUMINAMATH_GPT_rhombus_min_rotation_l76_7638


namespace NUMINAMATH_GPT_range_of_a_l76_7697

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), 3 * a * x + (a^2 - 3 * a + 2) * y - 9 < 0 → (3 * a * x + (a^2 - 3 * a + 2) * y - 9 = 0 → y > 0)) ↔ (1 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l76_7697


namespace NUMINAMATH_GPT_smallest_three_digit_times_largest_single_digit_l76_7688

theorem smallest_three_digit_times_largest_single_digit :
  let x := 100
  let y := 9
  ∃ z : ℕ, z = x * y ∧ 100 ≤ z ∧ z < 1000 :=
by
  let x := 100
  let y := 9
  use x * y
  sorry

end NUMINAMATH_GPT_smallest_three_digit_times_largest_single_digit_l76_7688


namespace NUMINAMATH_GPT_symmetric_circle_equation_l76_7696

theorem symmetric_circle_equation :
  ∀ (a b : ℝ), 
    (∀ (x y : ℝ), (x-2)^2 + (y+1)^2 = 4 → y = x + 1) → 
    (∃ x y : ℝ, (x + 2)^2 + (y - 3)^2 = 4) :=
  by
    sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l76_7696


namespace NUMINAMATH_GPT_total_pencils_in_drawer_l76_7649

-- Definitions based on conditions from the problem
def initial_pencils : ℕ := 138
def pencils_by_Nancy : ℕ := 256
def pencils_by_Steven : ℕ := 97

-- The theorem proving the total number of pencils in the drawer
theorem total_pencils_in_drawer : initial_pencils + pencils_by_Nancy + pencils_by_Steven = 491 :=
by
  -- This statement is equivalent to the mathematical problem given
  sorry

end NUMINAMATH_GPT_total_pencils_in_drawer_l76_7649


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l76_7612

-- We define the polynomial f(x)
def f (x : ℝ) := x^4 - 6 * x^3 + 11 * x^2 + 20 * x - 8

-- We need to show that the remainder when f(x) is divided by (x - 2) is 44
theorem remainder_when_divided_by_x_minus_2 : f 2 = 44 :=
by {
  -- this is where the proof would go
  sorry
}

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l76_7612


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l76_7605

theorem tenth_term_arithmetic_sequence :
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  aₙ 10 = 3 :=
by
  let a₁ := 3 / 4
  let d := 1 / 4
  let aₙ (n : ℕ) := a₁ + (n - 1) * d
  show aₙ 10 = 3
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l76_7605


namespace NUMINAMATH_GPT_value_of_ab_l76_7684

theorem value_of_ab (a b : ℝ) (x : ℝ) 
  (h : ∀ x, a * (-x) + b * (-x)^2 = -(a * x + b * x^2)) : a * b = 0 :=
sorry

end NUMINAMATH_GPT_value_of_ab_l76_7684


namespace NUMINAMATH_GPT_members_playing_both_sports_l76_7633

theorem members_playing_both_sports 
    (N : ℕ) (B : ℕ) (T : ℕ) (D : ℕ)
    (hN : N = 30) (hB : B = 18) (hT : T = 19) (hD : D = 2) :
    N - D = 28 ∧ B + T = 37 ∧ B + T - (N - D) = 9 :=
by
  sorry

end NUMINAMATH_GPT_members_playing_both_sports_l76_7633


namespace NUMINAMATH_GPT_find_number_l76_7671

-- Define the given condition
def number_div_property (num : ℝ) : Prop :=
  num / 0.3 = 7.3500000000000005

-- State the theorem to prove
theorem find_number (num : ℝ) (h : number_div_property num) : num = 2.205 :=
by sorry

end NUMINAMATH_GPT_find_number_l76_7671


namespace NUMINAMATH_GPT_jack_pages_l76_7682

theorem jack_pages (pages_per_booklet : ℕ) (num_booklets : ℕ) (h1 : pages_per_booklet = 9) (h2 : num_booklets = 49) : num_booklets * pages_per_booklet = 441 :=
by {
  sorry
}

end NUMINAMATH_GPT_jack_pages_l76_7682


namespace NUMINAMATH_GPT_evaluate_polynomial_l76_7689

-- Define the polynomial function
def polynomial (x : ℝ) : ℝ := x^3 + 3 * x^2 - 9 * x - 5

-- Define the condition: x is the positive root of the quadratic equation
def is_positive_root_of_quadratic (x : ℝ) : Prop := x > 0 ∧ x^2 + 3 * x - 9 = 0

-- The main theorem stating the polynomial evaluates to 22 given the condition
theorem evaluate_polynomial {x : ℝ} (h : is_positive_root_of_quadratic x) : polynomial x = 22 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_l76_7689


namespace NUMINAMATH_GPT_operation_three_six_l76_7683

theorem operation_three_six : (3 * 3 * 6) / (3 + 6) = 6 :=
by
  calc (3 * 3 * 6) / (3 + 6) = 6 := sorry

end NUMINAMATH_GPT_operation_three_six_l76_7683


namespace NUMINAMATH_GPT_parabola_passes_through_points_and_has_solution_4_l76_7664

theorem parabola_passes_through_points_and_has_solution_4 
  (a h k m: ℝ) :
  (∀ x, y = a * (x - h) ^ 2 + k → 
    (y = 0 → (x = -1 → x = 5))) → 
  (∃ m, ∀ x, (a * (x - h + m) ^ 2 + k = 0) → x = 4) → 
  m = -5 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_parabola_passes_through_points_and_has_solution_4_l76_7664


namespace NUMINAMATH_GPT_min_weighings_to_identify_fake_l76_7646

def piles := 1000000
def coins_per_pile := 1996
def weight_real_coin := 10
def weight_fake_coin := 9
def expected_total_weight : Nat :=
  (piles * (piles + 1) / 2) * weight_real_coin

theorem min_weighings_to_identify_fake :
  (∃ k : ℕ, k < piles ∧ 
  ∀ (W : ℕ), W = expected_total_weight - k → k = expected_total_weight - W) →
  true := 
by
  sorry

end NUMINAMATH_GPT_min_weighings_to_identify_fake_l76_7646


namespace NUMINAMATH_GPT_number_of_badminton_players_l76_7626

-- Definitions based on the given conditions
variable (Total_members : ℕ := 30)
variable (Tennis_players : ℕ := 19)
variable (No_sport_players : ℕ := 3)
variable (Both_sport_players : ℕ := 9)

-- The goal is to prove the number of badminton players is 17
theorem number_of_badminton_players :
  ∀ (B : ℕ), Total_members = B + Tennis_players - Both_sport_players + No_sport_players → B = 17 :=
by
  intro B
  intro h
  sorry

end NUMINAMATH_GPT_number_of_badminton_players_l76_7626


namespace NUMINAMATH_GPT_BC_length_l76_7611

theorem BC_length (AD BC MN : ℝ) (h1 : AD = 2) (h2 : MN = 6) (h3 : MN = 0.5 * (AD + BC)) : BC = 10 :=
by
  sorry

end NUMINAMATH_GPT_BC_length_l76_7611


namespace NUMINAMATH_GPT_intersection_M_N_l76_7613

theorem intersection_M_N :
  let M := {x | x^2 < 36}
  let N := {2, 4, 6, 8}
  M ∩ N = {2, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l76_7613


namespace NUMINAMATH_GPT_power_function_characterization_l76_7654

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_characterization (f : ℝ → ℝ) (h : f 2 = Real.sqrt 2) : 
  ∀ x : ℝ, f x = x ^ (1 / 2) :=
sorry

end NUMINAMATH_GPT_power_function_characterization_l76_7654


namespace NUMINAMATH_GPT_biased_die_sum_is_odd_l76_7600

def biased_die_probabilities : Prop :=
  let p_odd := 1 / 3
  let p_even := 2 / 3
  let scenarios := [
    (1/3) * (2/3)^2,
    (1/3)^3
  ]
  let sum := scenarios.sum
  sum = 13 / 27

theorem biased_die_sum_is_odd :
  biased_die_probabilities := by
    sorry

end NUMINAMATH_GPT_biased_die_sum_is_odd_l76_7600


namespace NUMINAMATH_GPT_determine_b_from_quadratic_l76_7604

theorem determine_b_from_quadratic (b n : ℝ) (h1 : b > 0) 
  (h2 : ∀ x, x^2 + b*x + 36 = (x + n)^2 + 20) : b = 8 := 
by 
  sorry

end NUMINAMATH_GPT_determine_b_from_quadratic_l76_7604


namespace NUMINAMATH_GPT_find_general_equation_of_line_l76_7632

variables {x y k b : ℝ}

-- Conditions: slope of the line is -2 and sum of its intercepts is 12.
def slope_of_line (l : ℝ → ℝ → Prop) : Prop := ∃ b, ∀ x y, l x y ↔ y = -2 * x + b
def sum_of_intercepts (l : ℝ → ℝ → Prop) : Prop := ∃ b, b + (b / 2) = 12

-- Question: What is the general equation of the line?
noncomputable def general_equation (l : ℝ → ℝ → Prop) : Prop :=
  slope_of_line l ∧ sum_of_intercepts l → ∀ x y, l x y ↔ 2 * x + y - 8 = 0

-- The theorem we need to prove
theorem find_general_equation_of_line (l : ℝ → ℝ → Prop) : general_equation l :=
sorry

end NUMINAMATH_GPT_find_general_equation_of_line_l76_7632


namespace NUMINAMATH_GPT_prism_volume_l76_7661

noncomputable def volume_of_prism (x y z : ℝ) : ℝ :=
  x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 40) (h2 : x * z = 50) (h3 : y * z = 100) :
  volume_of_prism x y z = 100 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_prism_volume_l76_7661


namespace NUMINAMATH_GPT_Ryan_learning_days_l76_7631

theorem Ryan_learning_days
  (hours_english_per_day : ℕ)
  (hours_chinese_per_day : ℕ)
  (total_hours : ℕ)
  (h1 : hours_english_per_day = 6)
  (h2 : hours_chinese_per_day = 7)
  (h3 : total_hours = 65) :
  total_hours / (hours_english_per_day + hours_chinese_per_day) = 5 := by
  sorry

end NUMINAMATH_GPT_Ryan_learning_days_l76_7631


namespace NUMINAMATH_GPT_cheryl_distance_walked_l76_7615

theorem cheryl_distance_walked (speed : ℕ) (time : ℕ) (distance_away : ℕ) (distance_home : ℕ) 
  (h1 : speed = 2) 
  (h2 : time = 3) 
  (h3 : distance_away = speed * time) 
  (h4 : distance_home = distance_away) : 
  distance_away + distance_home = 12 := 
by
  sorry

end NUMINAMATH_GPT_cheryl_distance_walked_l76_7615


namespace NUMINAMATH_GPT_number_of_arrangements_l76_7630

-- Definitions of the problem's conditions
def student_set : Finset ℕ := {1, 2, 3, 4, 5}

def specific_students : Finset ℕ := {1, 2}

def remaining_students : Finset ℕ := student_set \ specific_students

-- Formalize the problem statement
theorem number_of_arrangements : 
  ∀ (students : Finset ℕ) 
    (specific : Finset ℕ) 
    (remaining : Finset ℕ),
    students = student_set →
    specific = specific_students →
    remaining = remaining_students →
    (specific.card = 2 ∧ students.card = 5 ∧ remaining.card = 3) →
    (∃ (n : ℕ), n = 12) :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l76_7630


namespace NUMINAMATH_GPT_induction_divisibility_l76_7673

theorem induction_divisibility (k x y : ℕ) (h : k > 0) :
  (x^(2*k-1) + y^(2*k-1)) ∣ (x + y) → 
  (x^(2*k+1) + y^(2*k+1)) ∣ (x + y) :=
sorry

end NUMINAMATH_GPT_induction_divisibility_l76_7673


namespace NUMINAMATH_GPT_base7_number_l76_7656

theorem base7_number (A B C : ℕ) (h1 : 1 ≤ A ∧ A ≤ 6) (h2 : 1 ≤ B ∧ B ≤ 6) (h3 : 1 ≤ C ∧ C ≤ 6)
  (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_condition1 : B + C = 7)
  (h_condition2 : A + 1 = C)
  (h_condition3 : A + B = C) :
  A = 5 ∧ B = 1 ∧ C = 6 :=
sorry

end NUMINAMATH_GPT_base7_number_l76_7656


namespace NUMINAMATH_GPT_find_multiple_of_pages_l76_7619

-- Definitions based on conditions
def beatrix_pages : ℕ := 704
def cristobal_extra_pages : ℕ := 1423
def cristobal_pages (x : ℕ) : ℕ := x * beatrix_pages + 15

-- Proposition to prove the multiple x equals 2
theorem find_multiple_of_pages (x : ℕ) (h : cristobal_pages x = beatrix_pages + cristobal_extra_pages) : x = 2 :=
  sorry

end NUMINAMATH_GPT_find_multiple_of_pages_l76_7619


namespace NUMINAMATH_GPT_speed_of_first_train_l76_7658

-- Define the conditions
def distance_pq := 110 -- km
def speed_q := 25 -- km/h
def meet_time := 10 -- hours from midnight
def start_p := 7 -- hours from midnight
def start_q := 8 -- hours from midnight

-- Define the total travel time for each train
def travel_time_p := meet_time - start_p -- hours
def travel_time_q := meet_time - start_q -- hours

-- Define the distance covered by each train
def distance_covered_p (V_p : ℕ) : ℕ := V_p * travel_time_p
def distance_covered_q := speed_q * travel_time_q

-- Theorem to prove the speed of the first train
theorem speed_of_first_train (V_p : ℕ) : distance_covered_p V_p + distance_covered_q = distance_pq → V_p = 20 :=
sorry

end NUMINAMATH_GPT_speed_of_first_train_l76_7658


namespace NUMINAMATH_GPT_triangle_inequality_proof_l76_7625

theorem triangle_inequality_proof 
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end NUMINAMATH_GPT_triangle_inequality_proof_l76_7625


namespace NUMINAMATH_GPT_Marissa_has_21_more_marbles_than_Jonny_l76_7674

noncomputable def Mara_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Markus_marbles (bags : ℕ) (marbles : ℕ) : ℕ :=
bags * marbles

noncomputable def Jonny_marbles (total_marbles : ℕ) (bags : ℕ) : ℕ :=
total_marbles

noncomputable def Marissa_marbles (bags1 : ℕ) (marbles1 : ℕ) (bags2 : ℕ) (marbles2 : ℕ) : ℕ :=
(bags1 * marbles1) + (bags2 * marbles2)

noncomputable def Jonny : ℕ := Jonny_marbles 18 3

noncomputable def Marissa : ℕ := Marissa_marbles 3 5 3 8

theorem Marissa_has_21_more_marbles_than_Jonny : (Marissa - Jonny) = 21 :=
by
  sorry

end NUMINAMATH_GPT_Marissa_has_21_more_marbles_than_Jonny_l76_7674


namespace NUMINAMATH_GPT_unique_paintings_count_l76_7616

-- Given the conditions of the problem:
-- - N = 6 disks
-- - 3 disks are blue
-- - 2 disks are red
-- - 1 disk is green
-- - Two paintings that can be obtained from one another by a rotation or a reflection are considered the same

-- Define a theorem to calculate the number of unique paintings.
theorem unique_paintings_count : 
    ∃ n : ℕ, n = 13 :=
sorry

end NUMINAMATH_GPT_unique_paintings_count_l76_7616


namespace NUMINAMATH_GPT_total_play_time_in_hours_l76_7636

def football_time : ℕ := 60
def basketball_time : ℕ := 60

theorem total_play_time_in_hours : (football_time + basketball_time) / 60 = 2 := by
  sorry

end NUMINAMATH_GPT_total_play_time_in_hours_l76_7636


namespace NUMINAMATH_GPT_quadratic_sum_terms_l76_7680

theorem quadratic_sum_terms (a b c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + 16 * x - 72 = a * (x + b)^2 + c) → a + b + c = -46 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_sum_terms_l76_7680


namespace NUMINAMATH_GPT_percentage_of_red_shirts_l76_7695

variable (total_students : ℕ) (blue_percent green_percent : ℕ) (other_students : ℕ)
  (H_total : total_students = 800)
  (H_blue : blue_percent = 45)
  (H_green : green_percent = 15)
  (H_other : other_students = 136)
  (H_blue_students : 0.45 * 800 = 360)
  (H_green_students : 0.15 * 800 = 120)
  (H_sum : 360 + 120 + 136 = 616)
  
theorem percentage_of_red_shirts :
  ((total_students - (360 + 120 + other_students)) / total_students) * 100 = 23 := 
by {
  sorry
}

end NUMINAMATH_GPT_percentage_of_red_shirts_l76_7695


namespace NUMINAMATH_GPT_simplify_trig_expression_trig_identity_l76_7608

-- Defining the necessary functions
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

-- First problem
theorem simplify_trig_expression (α : ℝ) :
  (sin (2 * Real.pi - α) * sin (Real.pi + α) * cos (-Real.pi - α)) / (sin (3 * Real.pi - α) * cos (Real.pi - α)) = sin α :=
sorry

-- Second problem
theorem trig_identity (x : ℝ) (hx : cos x ≠ 0) (hx' : 1 - sin x ≠ 0) :
  (cos x / (1 - sin x)) = ((1 + sin x) / cos x) :=
sorry

end NUMINAMATH_GPT_simplify_trig_expression_trig_identity_l76_7608


namespace NUMINAMATH_GPT_find_value_l76_7643

-- Define the variables and given conditions
variables (x y z : ℚ)
variables (h1 : 2 * x - y = 4)
variables (h2 : 3 * x + z = 7)
variables (h3 : y = 2 * z)

-- Define the goal to prove
theorem find_value : 6 * x - 3 * y + 3 * z = 51 / 4 := by 
  sorry

end NUMINAMATH_GPT_find_value_l76_7643


namespace NUMINAMATH_GPT_not_perfect_square_l76_7603

theorem not_perfect_square (n : ℕ) (h₁ : 100 + 200 = 300) (h₂ : ¬(300 % 9 = 0)) : ¬(∃ m : ℕ, n = m * m) :=
by
  intros
  sorry

end NUMINAMATH_GPT_not_perfect_square_l76_7603


namespace NUMINAMATH_GPT_digit_difference_is_one_l76_7623

theorem digit_difference_is_one {p q : ℕ} (h : 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ p ≠ q)
  (digits_distinct : ∀ n ∈ [p, q], ∀ m ∈ [p, q], n ≠ m)
  (interchange_effect : 10 * p + q - (10 * q + p) = 9) : p - q = 1 :=
sorry

end NUMINAMATH_GPT_digit_difference_is_one_l76_7623


namespace NUMINAMATH_GPT_complementary_event_l76_7698

def car_a_selling_well : Prop := sorry
def car_b_selling_poorly : Prop := sorry

def event_A : Prop := car_a_selling_well ∧ car_b_selling_poorly
def event_complement (A : Prop) : Prop := ¬A

theorem complementary_event :
  event_complement event_A = (¬car_a_selling_well ∨ ¬car_b_selling_poorly) :=
by
  sorry

end NUMINAMATH_GPT_complementary_event_l76_7698


namespace NUMINAMATH_GPT_find_a_b_solve_inequality_l76_7694

-- Definitions for the given conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 6 > 4
def sol_set1 (x : ℝ) (b : ℝ) : Prop := x < 1 ∨ x > b
def root_eq (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 2 = 0

-- The final Lean statements for the proofs
theorem find_a_b (a b : ℝ) : (∀ x, (inequality1 a x) ↔ (sol_set1 x b)) → a = 1 ∧ b = 2 :=
sorry

theorem solve_inequality (c : ℝ) : 
  (∀ x, (root_eq 1 x) ↔ (x = 1 ∨ x = 2)) → 
  (c > 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (2 < x ∧ x < c)) ∧
  (c < 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (c < x ∧ x < 2)) ∧
  (c = 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ false) :=
sorry

end NUMINAMATH_GPT_find_a_b_solve_inequality_l76_7694


namespace NUMINAMATH_GPT_cost_of_pastrami_l76_7620

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_pastrami_l76_7620


namespace NUMINAMATH_GPT_closest_to_fraction_l76_7614

theorem closest_to_fraction (n d : ℝ) (h_n : n = 510) (h_d : d = 0.125) :
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 5000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 6000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 7000 ∧
  abs ((510 : ℝ) / (0.125 : ℝ)) - 4000 < abs ((510 : ℝ) / (0.125 : ℝ)) - 8000 :=
by
  sorry

end NUMINAMATH_GPT_closest_to_fraction_l76_7614


namespace NUMINAMATH_GPT_abs_ineq_range_k_l76_7624

theorem abs_ineq_range_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_range_k_l76_7624


namespace NUMINAMATH_GPT_smallest_x_for_perfect_cube_l76_7642

theorem smallest_x_for_perfect_cube (x : ℕ) (M : ℤ) (hx : x > 0) (hM : ∃ M, 1680 * x = M^3) : x = 44100 :=
sorry

end NUMINAMATH_GPT_smallest_x_for_perfect_cube_l76_7642


namespace NUMINAMATH_GPT_problem_l76_7660

def a (x : ℕ) : ℕ := 2005 * x + 2006
def b (x : ℕ) : ℕ := 2005 * x + 2007
def c (x : ℕ) : ℕ := 2005 * x + 2008

theorem problem (x : ℕ) : (a x)^2 + (b x)^2 + (c x)^2 - (a x) * (b x) - (a x) * (c x) - (b x) * (c x) = 3 :=
by sorry

end NUMINAMATH_GPT_problem_l76_7660


namespace NUMINAMATH_GPT_sum_of_distances_condition_l76_7641

theorem sum_of_distances_condition (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| < a) → a > 4 :=
sorry

end NUMINAMATH_GPT_sum_of_distances_condition_l76_7641


namespace NUMINAMATH_GPT_track_length_l76_7621

theorem track_length (x : ℝ) (tom_dist1 jerry_dist1 : ℝ) (tom_dist2 jerry_dist2 : ℝ) (deg_gap : ℝ) :
  deg_gap = 120 ∧ 
  tom_dist1 = 120 ∧ 
  (tom_dist1 + jerry_dist1 = x * deg_gap / 360) ∧ 
  (jerry_dist1 + jerry_dist2 = x * deg_gap / 360 + 180) →
  x = 630 :=
by
  sorry

end NUMINAMATH_GPT_track_length_l76_7621


namespace NUMINAMATH_GPT_find_k_l76_7672

-- Define the conditions
variables (x y k : ℕ)
axiom part_sum : x + y = 36
axiom first_part : x = 19
axiom value_eq : 8 * x + k * y = 203

-- Prove that k is 3
theorem find_k : k = 3 :=
by
  -- Insert your proof here
  sorry

end NUMINAMATH_GPT_find_k_l76_7672


namespace NUMINAMATH_GPT_cube_difference_l76_7655

theorem cube_difference {a b : ℝ} (h1 : a - b = 5) (h2 : a^2 + b^2 = 35) : a^3 - b^3 = 200 :=
sorry

end NUMINAMATH_GPT_cube_difference_l76_7655


namespace NUMINAMATH_GPT_e_count_estimation_l76_7685

-- Define the various parameters used in the conditions
def num_problems : Nat := 76
def avg_words_per_problem : Nat := 40
def avg_letters_per_word : Nat := 5
def frequency_of_e : Float := 0.1
def actual_e_count : Nat := 1661

-- The goal is to prove that the actual number of "e"s is 1661
theorem e_count_estimation : actual_e_count = 1661 := by
  -- Sorry, no proof is required.
  sorry

end NUMINAMATH_GPT_e_count_estimation_l76_7685


namespace NUMINAMATH_GPT_chord_division_ratio_l76_7627

theorem chord_division_ratio (R AB PO DP PC x AP PB : ℝ)
  (hR : R = 11)
  (hAB : AB = 18)
  (hPO : PO = 7)
  (hDP : DP = R - PO)
  (hPC : PC = R + PO)
  (hPower : AP * PB = DP * PC)
  (hChord : AP + PB = AB) :
  AP = 12 ∧ PB = 6 ∨ AP = 6 ∧ PB = 12 :=
by
  -- Structure of the theorem is provided.
  -- Proof steps are skipped and marked with sorry.
  sorry

end NUMINAMATH_GPT_chord_division_ratio_l76_7627


namespace NUMINAMATH_GPT_number_of_real_solutions_eq_2_l76_7681

theorem number_of_real_solutions_eq_2 :
  ∃! (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -5 / 3 :=
sorry

end NUMINAMATH_GPT_number_of_real_solutions_eq_2_l76_7681


namespace NUMINAMATH_GPT_find_functional_l76_7653

noncomputable def functional_equation_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (f x + y) = 2 * x + f (f y - x)

theorem find_functional (f : ℝ → ℝ) :
  functional_equation_solution f → ∃ c : ℝ, ∀ x, f x = x + c := 
by
  sorry

end NUMINAMATH_GPT_find_functional_l76_7653


namespace NUMINAMATH_GPT_exists_neg_monomial_l76_7699

theorem exists_neg_monomial (a : ℤ) (x y : ℤ) (m n : ℕ) (hq : a < 0) (hd : m + n = 5) :
  ∃ a m n, a < 0 ∧ m + n = 5 ∧ a * x^m * y^n = -x^2 * y^3 :=
by
  sorry

end NUMINAMATH_GPT_exists_neg_monomial_l76_7699


namespace NUMINAMATH_GPT_pencils_per_row_cannot_be_determined_l76_7686

theorem pencils_per_row_cannot_be_determined
  (rows : ℕ)
  (total_crayons : ℕ)
  (crayons_per_row : ℕ)
  (h_total_crayons: total_crayons = 210)
  (h_rows: rows = 7)
  (h_crayons_per_row: crayons_per_row = 30) :
  ∀ (pencils_per_row : ℕ), false :=
by
  sorry

end NUMINAMATH_GPT_pencils_per_row_cannot_be_determined_l76_7686


namespace NUMINAMATH_GPT_compute_54_mul_46_l76_7628

theorem compute_54_mul_46 : (54 * 46 = 2484) :=
by sorry

end NUMINAMATH_GPT_compute_54_mul_46_l76_7628


namespace NUMINAMATH_GPT_Tim_gave_kittens_to_Jessica_l76_7650

def Tim_original_kittens : ℕ := 6
def kittens_given_to_Jessica := 3
def kittens_given_by_Sara : ℕ := 9 
def Tim_final_kittens : ℕ := 12

theorem Tim_gave_kittens_to_Jessica :
  (Tim_original_kittens + kittens_given_by_Sara - kittens_given_to_Jessica = Tim_final_kittens) :=
by sorry

end NUMINAMATH_GPT_Tim_gave_kittens_to_Jessica_l76_7650


namespace NUMINAMATH_GPT_actual_cost_of_article_l76_7639

theorem actual_cost_of_article {x : ℝ} (h : 0.76 * x = 760) : x = 1000 :=
by
  sorry

end NUMINAMATH_GPT_actual_cost_of_article_l76_7639


namespace NUMINAMATH_GPT_simplify_and_evaluate_l76_7607

variable (a : ℝ)
noncomputable def given_expression : ℝ :=
    (3 * a / (a^2 - 4)) * (1 - 2 / a) - (4 / (a + 2))

theorem simplify_and_evaluate (h : a = Real.sqrt 2 - 1) : 
  given_expression a = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l76_7607


namespace NUMINAMATH_GPT_milk_leftover_after_milkshakes_l76_7635

theorem milk_leftover_after_milkshakes
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_milk : ℕ)
  (total_ice_cream : ℕ)
  (milkshakes_made : ℕ)
  (milk_used : ℕ)
  (milk_left : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  milkshakes_made = total_ice_cream / ice_cream_per_milkshake →
  milk_used = milkshakes_made * milk_per_milkshake →
  milk_left = total_milk - milk_used →
  milk_left = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_milk_leftover_after_milkshakes_l76_7635


namespace NUMINAMATH_GPT_train_speed_l76_7690

theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_conversion_factor : ℝ) (expected_speed : ℝ) (h_time_conversion : time_conversion_factor = 1 / 60) (h_time : time_minutes / 60 = 0.5) (h_distance : distance = 51) (h_expected_speed : expected_speed = 102) : distance / (time_minutes / 60) = expected_speed :=
by 
  sorry

end NUMINAMATH_GPT_train_speed_l76_7690


namespace NUMINAMATH_GPT_jett_profit_l76_7634

def initial_cost : ℕ := 600
def vaccination_cost : ℕ := 500
def daily_food_cost : ℕ := 20
def number_of_days : ℕ := 40
def selling_price : ℕ := 2500

def total_expenses : ℕ := initial_cost + vaccination_cost + daily_food_cost * number_of_days
def profit : ℕ := selling_price - total_expenses

theorem jett_profit : profit = 600 :=
by
  -- Completed proof steps
  sorry

end NUMINAMATH_GPT_jett_profit_l76_7634


namespace NUMINAMATH_GPT_amount_saved_percentage_l76_7667

variable (S : ℝ) 

-- Condition: Last year, Sandy saved 7% of her annual salary
def amount_saved_last_year (S : ℝ) : ℝ := 0.07 * S

-- Condition: This year, she made 15% more money than last year
def salary_this_year (S : ℝ) : ℝ := 1.15 * S

-- Condition: This year, she saved 10% of her salary
def amount_saved_this_year (S : ℝ) : ℝ := 0.10 * salary_this_year S

-- The statement to prove
theorem amount_saved_percentage (S : ℝ) : 
  amount_saved_this_year S = 1.642857 * amount_saved_last_year S :=
by 
  sorry

end NUMINAMATH_GPT_amount_saved_percentage_l76_7667


namespace NUMINAMATH_GPT_divisor_of_70th_number_l76_7602

-- Define the conditions
def s (d : ℕ) (n : ℕ) : ℕ := n * d + 5

-- Theorem stating the given problem
theorem divisor_of_70th_number (d : ℕ) (h : s d 70 = 557) : d = 8 :=
by
  -- The proof is to be filled in later. 
  -- Now, just create the structure.
  sorry

end NUMINAMATH_GPT_divisor_of_70th_number_l76_7602


namespace NUMINAMATH_GPT_Elle_in_seat_2_given_conditions_l76_7622

theorem Elle_in_seat_2_given_conditions
    (seats : Fin 4 → Type) -- Representation of the seating arrangement.
    (Garry Elle Fiona Hank : Type)
    (seat_of : Type → Fin 4)
    (h1 : seat_of Garry = 0) -- Garry is in seat #1 (index 0)
    (h2 : ¬ (seat_of Elle = seat_of Hank + 1 ∨ seat_of Elle = seat_of Hank - 1)) -- Elle is not next to Hank
    (h3 : ¬ (seat_of Fiona > seat_of Garry ∧ seat_of Fiona < seat_of Hank) ∧ ¬ (seat_of Fiona < seat_of Garry ∧ seat_of Fiona > seat_of Hank)) -- Fiona is not between Garry and Hank
    : seat_of Elle = 1 :=  -- Conclusion: Elle is in seat #2 (index 1)
    sorry

end NUMINAMATH_GPT_Elle_in_seat_2_given_conditions_l76_7622


namespace NUMINAMATH_GPT_LeRoy_should_pay_30_l76_7691

/-- Define the empirical amounts paid by LeRoy and Bernardo, and the total discount. -/
def LeRoy_paid : ℕ := 240
def Bernardo_paid : ℕ := 360
def total_discount : ℕ := 60

/-- Define total expenses pre-discount. -/
def total_expenses : ℕ := LeRoy_paid + Bernardo_paid

/-- Define total expenses post-discount. -/
def adjusted_expenses : ℕ := total_expenses - total_discount

/-- Define each person's adjusted share. -/
def each_adjusted_share : ℕ := adjusted_expenses / 2

/-- Define the amount LeRoy should pay Bernardo. -/
def leroy_to_pay : ℕ := each_adjusted_share - LeRoy_paid

/-- Prove that LeRoy should pay Bernardo $30 to equalize their expenses post-discount. -/
theorem LeRoy_should_pay_30 : leroy_to_pay = 30 :=
by 
  -- Proof goes here...
  sorry

end NUMINAMATH_GPT_LeRoy_should_pay_30_l76_7691


namespace NUMINAMATH_GPT_negation_of_sine_bound_l76_7645

theorem negation_of_sine_bound (p : ∀ x : ℝ, Real.sin x ≤ 1) : ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x₀ : ℝ, Real.sin x₀ > 1 := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_sine_bound_l76_7645


namespace NUMINAMATH_GPT_common_ratio_of_sequence_l76_7610

theorem common_ratio_of_sequence 
  (a1 a2 a3 a4 : ℤ)
  (h1 : a1 = 25)
  (h2 : a2 = -50)
  (h3 : a3 = 100)
  (h4 : a4 = -200)
  (is_geometric : ∀ (i : ℕ), a1 * (-2) ^ i = if i = 0 then a1 else if i = 1 then a2 else if i = 2 then a3 else a4) : 
  (-50 / 25 = -2) ∧ (100 / -50 = -2) ∧ (-200 / 100 = -2) :=
by 
  sorry

end NUMINAMATH_GPT_common_ratio_of_sequence_l76_7610


namespace NUMINAMATH_GPT_find_c_share_l76_7617

noncomputable def shares (a b c d : ℝ) : Prop :=
  (5 * a = 4 * c) ∧ (7 * b = 4 * c) ∧ (2 * d = 4 * c) ∧ (a + b + c + d = 1200)

theorem find_c_share (A B C D : ℝ) (h : shares A B C D) : C = 275 :=
  by
  sorry

end NUMINAMATH_GPT_find_c_share_l76_7617


namespace NUMINAMATH_GPT_problem1_problem2_l76_7651

-- Definitions for the number of combinations
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Problem 1
theorem problem1 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 4) + (C r 3 * C w 1) + (C r 2 * C w 2) = 115 := 
sorry

-- Problem 2
theorem problem2 (n r w: ℕ) (hc1: r = 4) (hc2: w = 6) :
  (C r 2 * C w 3) + (C r 3 * C w 2) + (C r 4 * C w 1) = 186 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l76_7651


namespace NUMINAMATH_GPT_smallest_rel_prime_210_l76_7677

theorem smallest_rel_prime_210 (x : ℕ) (hx : x > 1) (hrel_prime : Nat.gcd x 210 = 1) : x = 11 :=
sorry

end NUMINAMATH_GPT_smallest_rel_prime_210_l76_7677


namespace NUMINAMATH_GPT_net_profit_expression_and_break_even_point_l76_7606

-- Definitions based on the conditions in a)
def investment : ℝ := 600000
def initial_expense : ℝ := 80000
def expense_increase : ℝ := 20000
def annual_income : ℝ := 260000

-- Define the net profit function as given in the solution
def net_profit (n : ℕ) : ℝ :=
  - (n : ℝ)^2 + 19 * n - 60

-- Statement about the function and where the dealer starts making profit
theorem net_profit_expression_and_break_even_point :
  net_profit n = - (n : ℝ)^2 + 19 * n - 60 ∧ ∃ n ≥ 5, net_profit n > 0 :=
sorry

end NUMINAMATH_GPT_net_profit_expression_and_break_even_point_l76_7606


namespace NUMINAMATH_GPT_coins_amount_correct_l76_7676

-- Definitions based on the conditions
def cost_of_flour : ℕ := 5
def cost_of_cake_stand : ℕ := 28
def amount_given_in_bills : ℕ := 20 + 20
def change_received : ℕ := 10

-- Total cost of items
def total_cost : ℕ := cost_of_flour + cost_of_cake_stand

-- Total money given
def total_money_given : ℕ := total_cost + change_received

-- Amount given in loose coins
def loose_coins_given : ℕ := total_money_given - amount_given_in_bills

-- Proposition statement
theorem coins_amount_correct : loose_coins_given = 3 := by
  sorry

end NUMINAMATH_GPT_coins_amount_correct_l76_7676


namespace NUMINAMATH_GPT_right_angled_triangle_not_axisymmetric_l76_7693

-- Define a type for geometric figures
inductive Figure
| Angle : Figure
| EquilateralTriangle : Figure
| LineSegment : Figure
| RightAngledTriangle : Figure

open Figure

-- Define a function to determine if a figure is axisymmetric
def is_axisymmetric: Figure -> Prop
| Angle => true
| EquilateralTriangle => true
| LineSegment => true
| RightAngledTriangle => false

-- Statement of the problem
theorem right_angled_triangle_not_axisymmetric : 
  is_axisymmetric RightAngledTriangle = false :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_not_axisymmetric_l76_7693


namespace NUMINAMATH_GPT_correct_number_of_true_propositions_l76_7687

noncomputable def true_proposition_count : ℕ := 1

theorem correct_number_of_true_propositions (a b c : ℝ) :
    (∀ a b : ℝ, (a > b) ↔ (a^2 > b^2) = false) →
    (∀ a b : ℝ, (a > b) ↔ (a^3 > b^3) = true) →
    (∀ a b : ℝ, (a > b) → (|a| > |b|) = false) →
    (∀ a b c : ℝ, (a > b) → (a*c^2 ≤ b*c^2) = false) →
    (true_proposition_count = 1) :=
by
  sorry

end NUMINAMATH_GPT_correct_number_of_true_propositions_l76_7687


namespace NUMINAMATH_GPT_geometric_sequence_problem_l76_7665

variable (a : ℕ → ℝ)
variable (q : ℝ)

-- Geometric sequence definition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Given conditions
def condition_1 : Prop := a 5 * a 8 = 6
def condition_2 : Prop := a 3 + a 10 = 5

-- Concluded value of q^7
def q_seven (q : ℝ) (a : ℕ → ℝ) : Prop := 
  q^7 = a 20 / a 13

theorem geometric_sequence_problem
  (h1 : is_geometric_sequence a q)
  (h2 : condition_1 a)
  (h3 : condition_2 a) :
  q_seven q a = (q = 3/2) ∨ (q = 2/3) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l76_7665


namespace NUMINAMATH_GPT_find_m_given_sampling_conditions_l76_7678

-- Definitions for population and sampling conditions
def population_divided_into_groups : Prop :=
  ∀ n : ℕ, n < 100 → ∃ k : ℕ, k < 10 ∧ n / 10 = k

def systematic_sampling_condition (m k : ℕ) : Prop :=
  k < 10 ∧ m < 10 ∧ (m + k - 1) % 10 < 10 ∧ (m + k - 11) % 10 < 10

-- Given conditions
def given_conditions (m k : ℕ) (n : ℕ) : Prop :=
  k = 6 ∧ n = 52 ∧ systematic_sampling_condition m k

-- The statement to prove
theorem find_m_given_sampling_conditions :
  ∃ m : ℕ, given_conditions m 6 52 ∧ m = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_m_given_sampling_conditions_l76_7678


namespace NUMINAMATH_GPT_negation_exists_l76_7659

theorem negation_exists {x : ℝ} (h : ∀ x, x > 0 → x^2 - x ≤ 0) : ∃ x, x > 0 ∧ x^2 - x > 0 :=
sorry

end NUMINAMATH_GPT_negation_exists_l76_7659


namespace NUMINAMATH_GPT_maximum_value_a_plus_b_cubed_plus_c_fourth_l76_7609

theorem maximum_value_a_plus_b_cubed_plus_c_fourth (a b c : ℝ)
    (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
    (h_sum : a + b + c = 2) : a + b^3 + c^4 ≤ 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_a_plus_b_cubed_plus_c_fourth_l76_7609


namespace NUMINAMATH_GPT_nesting_rectangles_exists_l76_7675

theorem nesting_rectangles_exists :
  ∀ (rectangles : List (ℕ × ℕ)), rectangles.length = 101
    ∧ (∀ r ∈ rectangles, r.fst ≤ 100 ∧ r.snd ≤ 100) 
    → ∃ (A B C : ℕ × ℕ), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles 
    ∧ (A.fst < B.fst ∧ A.snd < B.snd) 
    ∧ (B.fst < C.fst ∧ B.snd < C.snd) := 
by sorry

end NUMINAMATH_GPT_nesting_rectangles_exists_l76_7675


namespace NUMINAMATH_GPT_shortest_chord_intercepted_by_line_l76_7666

theorem shortest_chord_intercepted_by_line (k : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 3 = 0 → y = k*x + 1 → (x - y + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_shortest_chord_intercepted_by_line_l76_7666


namespace NUMINAMATH_GPT_simplify_sqrt_expression_l76_7618

theorem simplify_sqrt_expression :
  (Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)) = 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_expression_l76_7618


namespace NUMINAMATH_GPT_evaluate_expression_at_three_l76_7652

-- Define the evaluation of the expression (x^x)^(x^x) at x=3
theorem evaluate_expression_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_three_l76_7652


namespace NUMINAMATH_GPT_line_slope_l76_7669

theorem line_slope (t : ℝ) : 
  (∃ (t : ℝ), x = 1 + 2 * t ∧ y = 2 - 3 * t) → 
  (∃ (m : ℝ), m = -3 / 2) :=
sorry

end NUMINAMATH_GPT_line_slope_l76_7669


namespace NUMINAMATH_GPT_abscissa_of_point_P_l76_7663

open Real

noncomputable def hyperbola_abscissa (x y : ℝ) : Prop :=
  (x^2 - y^2 = 4) ∧
  (x > 0) ∧
  ((x + 2 * sqrt 2) * (x - 2 * sqrt 2) = -y^2)

theorem abscissa_of_point_P :
  ∃ (x y : ℝ), hyperbola_abscissa x y ∧ x = sqrt 6 := by
  sorry

end NUMINAMATH_GPT_abscissa_of_point_P_l76_7663


namespace NUMINAMATH_GPT_width_of_vessel_is_5_l76_7647

open Real

noncomputable def width_of_vessel : ℝ :=
  let edge := 5
  let rise := 2.5
  let base_length := 10
  let volume_cube := edge ^ 3
  let volume_displaced := volume_cube
  let width := volume_displaced / (base_length * rise)
  width

theorem width_of_vessel_is_5 :
  width_of_vessel = 5 := by
    sorry

end NUMINAMATH_GPT_width_of_vessel_is_5_l76_7647


namespace NUMINAMATH_GPT_b_sequence_periodic_l76_7662

theorem b_sequence_periodic (b : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h_b1 : b 1 = 2 + Real.sqrt 3)
  (h_b2021 : b 2021 = 11 + Real.sqrt 3) :
  b 2048 = b 2 :=
sorry

end NUMINAMATH_GPT_b_sequence_periodic_l76_7662


namespace NUMINAMATH_GPT_max_number_of_kids_on_school_bus_l76_7637

-- Definitions based on the conditions from the problem
def totalRowsLowerDeck : ℕ := 15
def totalRowsUpperDeck : ℕ := 10
def capacityLowerDeckRow : ℕ := 5
def capacityUpperDeckRow : ℕ := 3
def reservedSeatsLowerDeck : ℕ := 10
def staffMembers : ℕ := 4

-- The total capacity of the lower and upper decks
def totalCapacityLowerDeck := totalRowsLowerDeck * capacityLowerDeckRow
def totalCapacityUpperDeck := totalRowsUpperDeck * capacityUpperDeckRow
def totalCapacity := totalCapacityLowerDeck + totalCapacityUpperDeck

-- The maximum number of different kids that can ride the bus
def maxKids := totalCapacity - reservedSeatsLowerDeck - staffMembers

theorem max_number_of_kids_on_school_bus : maxKids = 91 := 
by 
  -- Step-by-step proof not required for this task
  sorry

end NUMINAMATH_GPT_max_number_of_kids_on_school_bus_l76_7637


namespace NUMINAMATH_GPT_find_non_negative_integer_solutions_l76_7640

theorem find_non_negative_integer_solutions :
  ∃ (x y z w : ℕ), 2 ^ x * 3 ^ y - 5 ^ z * 7 ^ w = 1 ∧
  ((x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
   (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
   (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1)) := by
  sorry

end NUMINAMATH_GPT_find_non_negative_integer_solutions_l76_7640


namespace NUMINAMATH_GPT_andrew_age_proof_l76_7648

def andrew_age_problem : Prop :=
  ∃ (a g : ℚ), g = 15 * a ∧ g - a = 60 ∧ a = 30 / 7

theorem andrew_age_proof : andrew_age_problem :=
by
  sorry

end NUMINAMATH_GPT_andrew_age_proof_l76_7648


namespace NUMINAMATH_GPT_money_after_purchase_l76_7668

def initial_money : ℕ := 4
def cost_of_candy_bar : ℕ := 1
def money_left : ℕ := 3

theorem money_after_purchase :
  initial_money - cost_of_candy_bar = money_left := by
  sorry

end NUMINAMATH_GPT_money_after_purchase_l76_7668


namespace NUMINAMATH_GPT_sum_of_coeffs_eq_225_l76_7629

/-- The sum of the coefficients of all terms in the expansion
of (C_x + C_x^2 + C_x^3 + C_x^4)^2 is equal to 225. -/
theorem sum_of_coeffs_eq_225 (C_x : ℝ) : 
  (C_x + C_x^2 + C_x^3 + C_x^4)^2 = 225 :=
sorry

end NUMINAMATH_GPT_sum_of_coeffs_eq_225_l76_7629


namespace NUMINAMATH_GPT_max_friday_more_than_wednesday_l76_7644

-- Definitions and conditions
def played_hours_wednesday : ℕ := 2
def played_hours_thursday : ℕ := 2
def played_average_hours : ℕ := 3
def played_days : ℕ := 3

-- Total hours over three days
def total_hours : ℕ := played_average_hours * played_days

-- Hours played on Friday
def played_hours_wednesday_thursday : ℕ := played_hours_wednesday + played_hours_thursday

def played_hours_friday : ℕ := total_hours - played_hours_wednesday_thursday

-- Proof problem statement
theorem max_friday_more_than_wednesday : 
  played_hours_friday - played_hours_wednesday = 3 := 
sorry

end NUMINAMATH_GPT_max_friday_more_than_wednesday_l76_7644
