import Mathlib

namespace NUMINAMATH_GPT_square_root_and_quadratic_solution_l2167_216770

theorem square_root_and_quadratic_solution
  (a b : ℤ)
  (h1 : 2 * a + b = 0)
  (h2 : 3 * b + 12 = 0) :
  (2 * a - 3 * b = 16) ∧ (a * x^2 + 4 * b - 2 = 0 → x^2 = 9) :=
by {
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_square_root_and_quadratic_solution_l2167_216770


namespace NUMINAMATH_GPT_evaluate_expression_l2167_216726

theorem evaluate_expression (x : ℕ) (h : x = 3) : (x^x)^(x^x) = 27^27 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2167_216726


namespace NUMINAMATH_GPT_value_of_f_g_l2167_216750

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g (h₁ : f (g 3) = 35) (h₂ : g (f 3) = 11) : f (g 3) - g (f 3) = 24 :=
by
  calc
    f (g 3) - g (f 3) = 35 - 11 := by rw [h₁, h₂]
                      _         = 24 := by norm_num

end NUMINAMATH_GPT_value_of_f_g_l2167_216750


namespace NUMINAMATH_GPT_determine_mass_l2167_216772

noncomputable def mass_of_water 
  (P : ℝ) (t1 t2 : ℝ) (deltaT : ℝ) (cw : ℝ) : ℝ :=
  P * t1 / ((cw * deltaT) + ((cw * deltaT) / t2) * t1)

theorem determine_mass (P : ℝ) (t1 : ℝ) (deltaT : ℝ) (t2 : ℝ) (cw : ℝ) :
  P = 1000 → t1 = 120 → deltaT = 2 → t2 = 60 → cw = 4200 →
  mass_of_water P t1 deltaT t2 cw = 4.76 :=
by
  intros hP ht1 hdeltaT ht2 hcw
  sorry

end NUMINAMATH_GPT_determine_mass_l2167_216772


namespace NUMINAMATH_GPT_range_of_t_l2167_216781

noncomputable def condition (t : ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 5 / 2 ∧ (t * x^2 + 2 * x - 2 > 0)

theorem range_of_t (t : ℝ) : ¬¬ condition t → t > - 1 / 2 :=
by
  intros h
  -- The actual proof should be here
  sorry

end NUMINAMATH_GPT_range_of_t_l2167_216781


namespace NUMINAMATH_GPT_transmission_time_calc_l2167_216719

theorem transmission_time_calc
  (blocks : ℕ) (chunks_per_block : ℕ) (transmission_rate : ℕ) (time_in_minutes : ℕ)
  (h_blocks : blocks = 80)
  (h_chunks_per_block : chunks_per_block = 640)
  (h_transmission_rate : transmission_rate = 160) 
  (h_time_in_minutes : time_in_minutes = 5) : 
  (blocks * chunks_per_block / transmission_rate) / 60 = time_in_minutes := 
by
  sorry

end NUMINAMATH_GPT_transmission_time_calc_l2167_216719


namespace NUMINAMATH_GPT_log_expression_value_l2167_216729

theorem log_expression_value : (Real.log 8 / Real.log 10) + 3 * (Real.log 5 / Real.log 10) = 3 :=
by
  -- Assuming necessary properties and steps are already known and prove the theorem accordingly:
  sorry

end NUMINAMATH_GPT_log_expression_value_l2167_216729


namespace NUMINAMATH_GPT_triangle_median_inequality_l2167_216784

variable (a b c m_a m_b m_c D : ℝ)

-- Assuming the conditions are required to make the proof valid
axiom median_formula_m_a : 4 * m_a^2 + a^2 = 2 * b^2 + 2 * c^2
axiom median_formula_m_b : 4 * m_b^2 + b^2 = 2 * c^2 + 2 * a^2
axiom median_formula_m_c : 4 * m_c^2 + c^2 = 2 * a^2 + 2 * b^2

theorem triangle_median_inequality : 
  a^2 + b^2 <= m_c * 6 * D ∧ b^2 + c^2 <= m_a * 6 * D ∧ c^2 + a^2 <= m_b * 6 * D → 
  (a^2 + b^2) / m_c + (b^2 + c^2) / m_a + (c^2 + a^2) / m_b <= 6 * D := 
by
  sorry

end NUMINAMATH_GPT_triangle_median_inequality_l2167_216784


namespace NUMINAMATH_GPT_find_a_l2167_216794

theorem find_a (a b c : ℚ)
  (h1 : c / b = 4)
  (h2 : b / a = 2)
  (h3 : c = 20 - 7 * b) : a = 10 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2167_216794


namespace NUMINAMATH_GPT_ratio_of_eggs_l2167_216797

/-- Megan initially had 24 eggs (12 from the store and 12 from her neighbor). She used 6 eggs in total (2 for an omelet and 4 for baking). She set aside 9 eggs for three meals (3 eggs per meal). Finally, Megan divided the remaining 9 eggs by giving 9 to her aunt and keeping 9 for herself. The ratio of the eggs she gave to her aunt to the eggs she kept is 1:1. -/
theorem ratio_of_eggs
  (eggs_bought : ℕ)
  (eggs_from_neighbor : ℕ)
  (eggs_omelet : ℕ)
  (eggs_baking : ℕ)
  (meals : ℕ)
  (eggs_per_meal : ℕ)
  (aunt_got : ℕ)
  (kept_for_meals : ℕ)
  (initial_eggs := eggs_bought + eggs_from_neighbor)
  (used_eggs := eggs_omelet + eggs_baking)
  (remaining_eggs := initial_eggs - used_eggs)
  (assigned_eggs := meals * eggs_per_meal)
  (final_eggs := remaining_eggs - assigned_eggs)
  (ratio : ℚ := aunt_got / kept_for_meals) :
  eggs_bought = 12 ∧
  eggs_from_neighbor = 12 ∧
  eggs_omelet = 2 ∧
  eggs_baking = 4 ∧
  meals = 3 ∧
  eggs_per_meal = 3 ∧
  aunt_got = 9 ∧
  kept_for_meals = assigned_eggs →
  ratio = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_of_eggs_l2167_216797


namespace NUMINAMATH_GPT_no_possible_blue_socks_l2167_216774

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end NUMINAMATH_GPT_no_possible_blue_socks_l2167_216774


namespace NUMINAMATH_GPT_hyperbola_foci_y_axis_condition_l2167_216705

theorem hyperbola_foci_y_axis_condition (m n : ℝ) (h : m * n < 0) : 
  (mx^2 + ny^2 = 1) →
  (m < 0 ∧ n > 0) :=
sorry

end NUMINAMATH_GPT_hyperbola_foci_y_axis_condition_l2167_216705


namespace NUMINAMATH_GPT_greatest_possible_integer_l2167_216720

theorem greatest_possible_integer 
  (n k l : ℕ) 
  (h1 : n < 150) 
  (h2 : n = 9 * k - 2) 
  (h3 : n = 6 * l - 4) : 
  n = 146 := 
sorry

end NUMINAMATH_GPT_greatest_possible_integer_l2167_216720


namespace NUMINAMATH_GPT_lincoln_high_fraction_of_girls_l2167_216767

noncomputable def fraction_of_girls_in_science_fair (total_girls total_boys : ℕ) (frac_girls_participated frac_boys_participated : ℚ) : ℚ :=
  let participating_girls := frac_girls_participated * total_girls
  let participating_boys := frac_boys_participated * total_boys
  participating_girls / (participating_girls + participating_boys)

theorem lincoln_high_fraction_of_girls 
  (total_girls : ℕ) (total_boys : ℕ)
  (frac_girls_participated : ℚ) (frac_boys_participated : ℚ)
  (h1 : total_girls = 150) (h2 : total_boys = 100)
  (h3 : frac_girls_participated = 4/5) (h4 : frac_boys_participated = 3/4) :
  fraction_of_girls_in_science_fair total_girls total_boys frac_girls_participated frac_boys_participated = 8/13 := 
by
  sorry

end NUMINAMATH_GPT_lincoln_high_fraction_of_girls_l2167_216767


namespace NUMINAMATH_GPT_clock_angle_at_8_20_is_130_degrees_l2167_216777

/--
A clock has 12 hours, and each hour represents 30 degrees.
The minute hand moves 6 degrees per minute.
The hour hand moves 0.5 degrees per minute from its current hour position.
Prove that the smaller angle between the hour and minute hands at 8:20 p.m. is 130 degrees.
-/
theorem clock_angle_at_8_20_is_130_degrees
    (hours_per_clock : ℝ := 12)
    (degrees_per_hour : ℝ := 360 / hours_per_clock)
    (minutes_per_hour : ℝ := 60)
    (degrees_per_minute : ℝ := 360 / minutes_per_hour)
    (hour_slider_per_minute : ℝ := degrees_per_hour / minutes_per_hour)
    (minute_hand_at_20 : ℝ := 20 * degrees_per_minute)
    (hour_hand_at_8: ℝ := 8 * degrees_per_hour)
    (hour_hand_move_in_20_minutes : ℝ := 20 * hour_slider_per_minute)
    (hour_hand_at_8_20 : ℝ := hour_hand_at_8 + hour_hand_move_in_20_minutes) :
  |hour_hand_at_8_20 - minute_hand_at_20| = 130 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_at_8_20_is_130_degrees_l2167_216777


namespace NUMINAMATH_GPT_smallest_positive_period_max_value_in_interval_l2167_216762

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi :=
sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 5 / 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_max_value_in_interval_l2167_216762


namespace NUMINAMATH_GPT_analytic_expression_of_f_l2167_216745

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi / 2)

noncomputable def g (α : ℝ) := Real.cos (α - Real.pi / 3)

theorem analytic_expression_of_f :
  (∀ x, f x = Real.cos x) ∧
  (∀ α, α ∈ Set.Icc 0 Real.pi → g α = 1/2 → (α = 0 ∨ α = 2 * Real.pi / 3)) :=
by
  sorry

end NUMINAMATH_GPT_analytic_expression_of_f_l2167_216745


namespace NUMINAMATH_GPT_max_value_x_plus_2y_max_of_x_plus_2y_l2167_216737

def on_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 / 4 = 1

theorem max_value_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  x + 2 * y ≤ Real.sqrt 22 :=
sorry

theorem max_of_x_plus_2y (x y : ℝ) (h : on_ellipse x y) :
  ∃θ ∈ Set.Icc 0 (2 * Real.pi), (x = Real.sqrt 6 * Real.cos θ) ∧ (y = 2 * Real.sin θ) :=
sorry

end NUMINAMATH_GPT_max_value_x_plus_2y_max_of_x_plus_2y_l2167_216737


namespace NUMINAMATH_GPT_correct_judgement_l2167_216723

noncomputable def f (x : ℝ) : ℝ :=
if -2 ≤ x ∧ x ≤ 2 then (1 / 2) * Real.sqrt (4 - x^2)
else - (1 / 2) * Real.sqrt (x^2 - 4)

noncomputable def F (x : ℝ) : ℝ := f x + x

theorem correct_judgement : (∀ y : ℝ, ∃ x : ℝ, (f x = y) ↔ (y ∈ Set.Iic 1)) ∧ (∃! x : ℝ, F x = 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_judgement_l2167_216723


namespace NUMINAMATH_GPT_bowling_average_l2167_216722

theorem bowling_average (gretchen_score mitzi_score beth_score : ℤ) (h1 : gretchen_score = 120) (h2 : mitzi_score = 113) (h3 : beth_score = 85) :
  (gretchen_score + mitzi_score + beth_score) / 3 = 106 :=
by
  sorry

end NUMINAMATH_GPT_bowling_average_l2167_216722


namespace NUMINAMATH_GPT_volume_of_dug_out_earth_l2167_216769

theorem volume_of_dug_out_earth
  (diameter depth : ℝ)
  (h_diameter : diameter = 2) 
  (h_depth : depth = 14) 
  : abs ((π * (1 / 2 * diameter / 2) ^ 2 * depth) - 44) < 0.1 :=
by
  -- Provide a placeholder for the proof
  sorry

end NUMINAMATH_GPT_volume_of_dug_out_earth_l2167_216769


namespace NUMINAMATH_GPT_determine_xyz_l2167_216776

theorem determine_xyz (x y z : ℂ) (h1 : x * y + 3 * y = -9) (h2 : y * z + 3 * z = -9) (h3 : z * x + 3 * x = -9) : 
  x * y * z = 27 := 
by
  sorry

end NUMINAMATH_GPT_determine_xyz_l2167_216776


namespace NUMINAMATH_GPT_exist_a_b_not_triangle_l2167_216716

theorem exist_a_b_not_triangle (h₁ : ∀ a b : ℕ, (a > 1000) → (b > 1000) →
  ∃ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  ∃ (a b : ℕ), (a > 1000 ∧ b > 1000) ∧ 
  ∀ c : ℕ, (∃ (k : ℕ), c = k * k) →
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
sorry

end NUMINAMATH_GPT_exist_a_b_not_triangle_l2167_216716


namespace NUMINAMATH_GPT_abs_case_inequality_solution_l2167_216768

theorem abs_case_inequality_solution (x : ℝ) :
  (|x + 1| + |x - 4| ≥ 7) ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 5) :=
by
  sorry

end NUMINAMATH_GPT_abs_case_inequality_solution_l2167_216768


namespace NUMINAMATH_GPT_tan_diff_angle_neg7_l2167_216778

-- Define the main constants based on the conditions given
variables (α : ℝ)
axiom sin_alpha : Real.sin α = -3/5
axiom alpha_in_fourth_quadrant : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2

-- Define the statement that needs to be proven based on the question and the correct answer
theorem tan_diff_angle_neg7 : 
  Real.tan (α - Real.pi / 4) = -7 :=
sorry

end NUMINAMATH_GPT_tan_diff_angle_neg7_l2167_216778


namespace NUMINAMATH_GPT_problem1_problem2_l2167_216721

-- Problem 1
theorem problem1 : (-1 : ℤ) ^ 2024 + (1 / 3 : ℝ) ^ (-2 : ℤ) - (3.14 - Real.pi) ^ 0 = 9 := 
sorry

-- Problem 2
theorem problem2 (x : ℤ) (y : ℤ) (hx : x = 2) (hy : y = 3) : 
  x * (x + 2 * y) - (x + 1) ^ 2 + 2 * x = 11 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2167_216721


namespace NUMINAMATH_GPT_base5_2004_to_decimal_is_254_l2167_216766

def base5_to_decimal (n : Nat) : Nat :=
  match n with
  | 2004 => 2 * 5^3 + 0 * 5^2 + 0 * 5^1 + 4 * 5^0
  | _ => 0

theorem base5_2004_to_decimal_is_254 :
  base5_to_decimal 2004 = 254 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_base5_2004_to_decimal_is_254_l2167_216766


namespace NUMINAMATH_GPT_value_of_a_c_l2167_216732

theorem value_of_a_c {a b c d : ℝ} :
  (∀ x y : ℝ, y = -|x - a| + b → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) ∧
  (∀ x y : ℝ, y = |x - c| - d → (x = 1 ∧ y = 4) ∨ (x = 7 ∧ y = 2)) →
  a + c = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_c_l2167_216732


namespace NUMINAMATH_GPT_walking_time_l2167_216746

theorem walking_time 
  (speed_km_hr : ℝ := 10) 
  (distance_km : ℝ := 6) 
  : (distance_km / (speed_km_hr / 60)) = 36 :=
by
  sorry

end NUMINAMATH_GPT_walking_time_l2167_216746


namespace NUMINAMATH_GPT_problem_proof_l2167_216786

noncomputable def arithmetic_sequences (a b : ℕ → ℤ) (S T : ℕ → ℤ) :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2 ∧
         T n = (n * (2 * b 0 + (n - 1) * (b 1 - b 0))) / 2

theorem problem_proof 
  (a b : ℕ → ℤ) 
  (S T : ℕ → ℤ)
  (h_seq : arithmetic_sequences a b S T)
  (h_relation : ∀ n, S n / T n = (7 * n : ℤ) / (n + 3)) :
  (a 5) / (b 5) = 21 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_problem_proof_l2167_216786


namespace NUMINAMATH_GPT_percentage_of_water_in_fresh_grapes_l2167_216730

theorem percentage_of_water_in_fresh_grapes
  (P : ℝ)  -- Let P be the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 5)  -- weight of fresh grapes in kg
  (dried_grapes_weight : ℝ := 0.625)  -- weight of dried grapes in kg
  (dried_water_percentage : ℝ := 20)  -- percentage of water in dried grapes
  (h1 : (100 - P) / 100 * fresh_grapes_weight = (100 - dried_water_percentage) / 100 * dried_grapes_weight) :
  P = 90 := 
sorry

end NUMINAMATH_GPT_percentage_of_water_in_fresh_grapes_l2167_216730


namespace NUMINAMATH_GPT_remainder_of_sum_l2167_216783

theorem remainder_of_sum (a b c : ℕ) (h₁ : a * b * c % 7 = 1) (h₂ : 2 * c % 7 = 5) (h₃ : 3 * b % 7 = (4 + b) % 7) :
  (a + b + c) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l2167_216783


namespace NUMINAMATH_GPT_daily_wage_male_worker_l2167_216755

variables
  (num_male : ℕ) (num_female : ℕ) (num_child : ℕ)
  (wage_female : ℝ) (wage_child : ℝ) (avg_wage : ℝ)
  (total_workers : ℕ := num_male + num_female + num_child)
  (total_wage_all : ℝ := avg_wage * total_workers)
  (total_wage_female : ℝ := num_female * wage_female)
  (total_wage_child : ℝ := num_child * wage_child)
  (total_wage_male : ℝ := total_wage_all - (total_wage_female + total_wage_child))
  (wage_per_male : ℝ := total_wage_male / num_male)

theorem daily_wage_male_worker :
  num_male = 20 →
  num_female = 15 →
  num_child = 5 →
  wage_female = 20 →
  wage_child = 8 →
  avg_wage = 21 →
  wage_per_male = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_daily_wage_male_worker_l2167_216755


namespace NUMINAMATH_GPT_length_of_second_train_l2167_216713

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train : ℝ)
  (speed_second_train : ℝ)
  (cross_time : ℝ)
  (opposite_directions : Bool) :
  speed_first_train = 120 / 3.6 →
  speed_second_train = 80 / 3.6 →
  cross_time = 9 →
  length_first_train = 260 →
  opposite_directions = true →
  ∃ (length_second_train : ℝ), length_second_train = 240 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_train_l2167_216713


namespace NUMINAMATH_GPT_impossible_to_place_numbers_l2167_216752

noncomputable def divisible (a b : ℕ) : Prop := ∃ k : ℕ, a * k = b

def connected (G : Finset (ℕ × ℕ)) (u v : ℕ) : Prop := (u, v) ∈ G ∨ (v, u) ∈ G

def valid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, connected G i j → divisible (f i) (f j) ∨ divisible (f j) (f i)

def invalid_assignment (G : Finset (ℕ × ℕ)) (f : ℕ → ℕ) : Prop :=
  ∀ ⦃i j⦄, ¬ connected G i j → ¬ divisible (f i) (f j) ∧ ¬ divisible (f j) (f i)

theorem impossible_to_place_numbers (G : Finset (ℕ × ℕ)) :
  (∃ f : ℕ → ℕ, valid_assignment G f ∧ invalid_assignment G f) → False :=
by
  sorry

end NUMINAMATH_GPT_impossible_to_place_numbers_l2167_216752


namespace NUMINAMATH_GPT_first_term_arithmetic_sequence_median_1010_last_2015_l2167_216742

theorem first_term_arithmetic_sequence_median_1010_last_2015 (a₁ : ℕ) :
  let median := 1010
  let last_term := 2015
  (a₁ + last_term = 2 * median) → a₁ = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_first_term_arithmetic_sequence_median_1010_last_2015_l2167_216742


namespace NUMINAMATH_GPT_combination_property_problem_solution_l2167_216796

open Nat

def combination (n k : ℕ) : ℕ :=
  if h : k ≤ n then (factorial n) / (factorial k * factorial (n - k)) else 0

theorem combination_property (n k : ℕ) (h₀ : 1 ≤ k) (h₁ : k ≤ n) :
  combination n k + combination n (k - 1) = combination (n + 1) k := sorry

theorem problem_solution :
  (combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + combination 7 2 + 
   combination 8 2 + combination 9 2 + combination 10 2 + combination 11 2 + combination 12 2 + 
   combination 13 2 + combination 14 2 + combination 15 2 + combination 16 2 + combination 17 2 + 
   combination 18 2 + combination 19 2) = 1139 := sorry

end NUMINAMATH_GPT_combination_property_problem_solution_l2167_216796


namespace NUMINAMATH_GPT_y_intercept_of_line_l2167_216761

theorem y_intercept_of_line : 
  ∀ (x y : ℝ), 3 * x - 5 * y = 7 → y = -7 / 5 :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l2167_216761


namespace NUMINAMATH_GPT_part1_part2_l2167_216751

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2| + |x - a|

-- Prove part 1: For all x in ℝ, log(f(x, -8)) ≥ 1
theorem part1 : ∀ x : ℝ, Real.log (f x (-8)) ≥ 1 :=
by 
  sorry

-- Prove part 2: For all x in ℝ, if f(x,a) ≥ a, then a ≤ 1
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ a) → a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2167_216751


namespace NUMINAMATH_GPT_total_spent_is_correct_l2167_216706

def meal_prices : List ℕ := [12, 15, 10, 18, 20]
def ice_cream_prices : List ℕ := [2, 3, 3, 4, 4]
def tip_percentage : ℝ := 0.15
def tax_percentage : ℝ := 0.08

def total_meal_cost (prices : List ℕ) : ℝ :=
  prices.sum

def total_ice_cream_cost (prices : List ℕ) : ℝ :=
  prices.sum

def calculate_tip (total_meal_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  total_meal_cost * tip_percentage

def calculate_tax (total_meal_cost : ℝ) (tax_percentage : ℝ) : ℝ :=
  total_meal_cost * tax_percentage

def total_amount_spent (meal_prices : List ℕ) (ice_cream_prices : List ℕ) (tip_percentage : ℝ) (tax_percentage : ℝ) : ℝ :=
  let total_meal := total_meal_cost meal_prices
  let total_ice_cream := total_ice_cream_cost ice_cream_prices
  let tip := calculate_tip total_meal tip_percentage
  let tax := calculate_tax total_meal tax_percentage
  total_meal + total_ice_cream + tip + tax

theorem total_spent_is_correct :
  total_amount_spent meal_prices ice_cream_prices tip_percentage tax_percentage = 108.25 := 
by
  sorry

end NUMINAMATH_GPT_total_spent_is_correct_l2167_216706


namespace NUMINAMATH_GPT_convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l2167_216733

noncomputable def cost_per_pure_milk_box (x : ℕ) : ℝ := 2000 / x
noncomputable def cost_per_yogurt_box (x : ℕ) : ℝ := 4800 / (1.5 * x)

theorem convenience_store_pure_milk_quantity
  (x : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30) :
  x = 40 :=
by
  sorry

noncomputable def pure_milk_price := 80
noncomputable def yogurt_price (cost_per_yogurt_box : ℝ) : ℝ := cost_per_yogurt_box * 1.25

theorem convenience_store_yogurt_discount
  (x y : ℕ)
  (hx : cost_per_yogurt_box x - cost_per_pure_milk_box x = 30)
  (total_profit : ℕ)
  (profit_condition :
    pure_milk_price * x +
    yogurt_price (cost_per_yogurt_box x) * (1.5 * x - y) +
    yogurt_price (cost_per_yogurt_box x) * 0.9 * y - 2000 - 4800 = total_profit)
  (pure_milk_quantity : x = 40)
  (profit_value : total_profit = 2150) :
  y = 25 :=
by
  sorry

end NUMINAMATH_GPT_convenience_store_pure_milk_quantity_convenience_store_yogurt_discount_l2167_216733


namespace NUMINAMATH_GPT_new_average_age_l2167_216724

/--
The average age of 7 people in a room is 28 years.
A 22-year-old person leaves the room, and a 30-year-old person enters the room.
Prove that the new average age of the people in the room is \( 29 \frac{1}{7} \).
-/
theorem new_average_age (avg_age : ℕ) (num_people : ℕ) (leaving_age : ℕ) (entering_age : ℕ)
  (H1 : avg_age = 28)
  (H2 : num_people = 7)
  (H3 : leaving_age = 22)
  (H4 : entering_age = 30) :
  (avg_age * num_people - leaving_age + entering_age) / num_people = 29 + 1 / 7 := 
by
  sorry

end NUMINAMATH_GPT_new_average_age_l2167_216724


namespace NUMINAMATH_GPT_width_of_foil_covered_prism_l2167_216709

theorem width_of_foil_covered_prism (L W H : ℝ) 
  (h1 : W = 2 * L)
  (h2 : W = 2 * H)
  (h3 : L * W * H = 128)
  (h4 : L = H) :
  W + 2 = 8 :=
sorry

end NUMINAMATH_GPT_width_of_foil_covered_prism_l2167_216709


namespace NUMINAMATH_GPT_dance_team_recruits_l2167_216702

theorem dance_team_recruits :
  ∃ (x : ℕ), x + 2 * x + (2 * x + 10) = 100 ∧ (2 * x + 10) = 46 :=
by
  sorry

end NUMINAMATH_GPT_dance_team_recruits_l2167_216702


namespace NUMINAMATH_GPT_negation_of_no_honors_students_attend_school_l2167_216791

-- Definitions (conditions and question)
def honors_student (x : Type) : Prop := sorry -- The condition defining an honors student
def attends_school (x : Type) : Prop := sorry -- The condition defining a student attending the school

-- The theorem statement
theorem negation_of_no_honors_students_attend_school :
  (¬ ∃ x : Type, honors_student x ∧ attends_school x) ↔ (∃ x : Type, honors_student x ∧ attends_school x) :=
sorry

end NUMINAMATH_GPT_negation_of_no_honors_students_attend_school_l2167_216791


namespace NUMINAMATH_GPT_dorothy_annual_earnings_correct_l2167_216773

-- Define the conditions
def dorothyEarnings (X : ℝ) : Prop :=
  X - 0.18 * X = 49200

-- Define the amount Dorothy earns a year
def dorothyAnnualEarnings : ℝ := 60000

-- State the theorem
theorem dorothy_annual_earnings_correct : dorothyEarnings dorothyAnnualEarnings :=
by
-- The proof will be inserted here
sorry

end NUMINAMATH_GPT_dorothy_annual_earnings_correct_l2167_216773


namespace NUMINAMATH_GPT_racing_cars_lcm_l2167_216799

theorem racing_cars_lcm :
  let a := 28
  let b := 24
  let c := 32
  Nat.lcm a (Nat.lcm b c) = 672 :=
by
  sorry

end NUMINAMATH_GPT_racing_cars_lcm_l2167_216799


namespace NUMINAMATH_GPT_Sanji_received_86_coins_l2167_216790

noncomputable def total_coins := 280

def Jack_coins (x : ℕ) := x
def Jimmy_coins (x : ℕ) := x + 11
def Tom_coins (x : ℕ) := x - 15
def Sanji_coins (x : ℕ) := x + 20

theorem Sanji_received_86_coins (x : ℕ) (hx : Jack_coins x + Jimmy_coins x + Tom_coins x + Sanji_coins x = total_coins) : Sanji_coins x = 86 :=
sorry

end NUMINAMATH_GPT_Sanji_received_86_coins_l2167_216790


namespace NUMINAMATH_GPT_xy_in_N_l2167_216795

def M : Set ℤ := {x | ∃ n : ℤ, x = 3 * n + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n - 1}

theorem xy_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  -- hint: use any knowledge and axioms from Mathlib to aid your proof
  sorry

end NUMINAMATH_GPT_xy_in_N_l2167_216795


namespace NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l2167_216779

variable (m : ℝ)

/-- The problem statement and proof condition -/
theorem condition_necessary_but_not_sufficient :
  (∀ x : ℝ, |x - 2| + |x + 2| > m) → (∀ x : ℝ, x^2 + m * x + 4 > 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l2167_216779


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l2167_216740

theorem sum_of_arithmetic_sequence (a d1 d2 : ℕ) 
  (h1 : d1 = d2 + 2) 
  (h2 : d1 + d2 = 24) 
  (a_pos : 0 < a) : 
  (a + (a + d1) + (a + d1) + (a + d1 + d2) = 54) := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l2167_216740


namespace NUMINAMATH_GPT_billy_points_l2167_216710

theorem billy_points (B : ℤ) (h : B - 9 = 2) : B = 11 := 
by 
  sorry

end NUMINAMATH_GPT_billy_points_l2167_216710


namespace NUMINAMATH_GPT_maximal_intersection_area_of_rectangles_l2167_216749

theorem maximal_intersection_area_of_rectangles :
  ∀ (a b : ℕ), a * b = 2015 ∧ a < b →
  ∀ (c d : ℕ), c * d = 2016 ∧ c > d →
  ∃ (max_area : ℕ), max_area = 1302 ∧ ∀ intersection_area, intersection_area ≤ 1302 := 
by
  sorry

end NUMINAMATH_GPT_maximal_intersection_area_of_rectangles_l2167_216749


namespace NUMINAMATH_GPT_ratio_of_logs_l2167_216792

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem ratio_of_logs (a b: ℝ) (h1 : log_base 8 a = log_base 18 b) 
    (h2 : log_base 18 b = log_base 32 (a + b)) 
    (hpos : 0 < a ∧ 0 < b) :
    b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) :=
by 
    sorry

end NUMINAMATH_GPT_ratio_of_logs_l2167_216792


namespace NUMINAMATH_GPT_probability_of_orange_face_l2167_216764

theorem probability_of_orange_face :
  ∃ (G O P : ℕ) (total_faces : ℕ), total_faces = 10 ∧ G = 5 ∧ O = 3 ∧ P = 2 ∧
  (O / total_faces : ℚ) = 3 / 10 := by 
  sorry

end NUMINAMATH_GPT_probability_of_orange_face_l2167_216764


namespace NUMINAMATH_GPT_transport_cost_B_condition_l2167_216756

-- Define the parameters for coal from Mine A
def calories_per_gram_A := 4
def price_per_ton_A := 20
def transport_cost_A := 8

-- Define the parameters for coal from Mine B
def calories_per_gram_B := 6
def price_per_ton_B := 24

-- Define the total cost for transporting one ton from Mine A to city N
def total_cost_A := price_per_ton_A + transport_cost_A

-- Define the question as a Lean theorem
theorem transport_cost_B_condition : 
  ∀ (transport_cost_B : ℝ), 
  (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) → 
  transport_cost_B = 18 :=
by
  intros transport_cost_B h
  have h_eq : (total_cost_A : ℝ) / (calories_per_gram_A : ℝ) = (price_per_ton_B + transport_cost_B) / (calories_per_gram_B : ℝ) := h
  sorry

end NUMINAMATH_GPT_transport_cost_B_condition_l2167_216756


namespace NUMINAMATH_GPT_fewest_keystrokes_to_256_l2167_216725

def fewest_keystrokes (start target : Nat) : Nat :=
if start = 1 && target = 256 then 8 else sorry

theorem fewest_keystrokes_to_256 : fewest_keystrokes 1 256 = 8 :=
by
  sorry

end NUMINAMATH_GPT_fewest_keystrokes_to_256_l2167_216725


namespace NUMINAMATH_GPT_toaster_popularity_l2167_216793

theorem toaster_popularity
  (c₁ c₂ : ℤ) (p₁ p₂ k : ℤ)
  (h₀ : p₁ * c₁ = k)
  (h₁ : p₁ = 12)
  (h₂ : c₁ = 500)
  (h₃ : c₂ = 750)
  (h₄ : k = p₁ * c₁) :
  p₂ * c₂ = k → p₂ = 8 :=
by
  sorry

end NUMINAMATH_GPT_toaster_popularity_l2167_216793


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2167_216711

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ) (k : ℕ) :
  a 1 = 1 →
  (∀ n, a (n + 1) = a n + 2) →
  (∀ n, S n = n * n) →
  S (k + 2) - S k = 24 →
  k = 5 :=
by
  intros a1 ha hS hSk
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2167_216711


namespace NUMINAMATH_GPT_dosage_range_l2167_216734

theorem dosage_range (d : ℝ) (h : 60 ≤ d ∧ d ≤ 120) : 15 ≤ (d / 4) ∧ (d / 4) ≤ 30 :=
by
  sorry

end NUMINAMATH_GPT_dosage_range_l2167_216734


namespace NUMINAMATH_GPT_vector_computation_l2167_216757

def v1 : ℤ × ℤ := (3, -5)
def v2 : ℤ × ℤ := (2, -10)
def s1 : ℤ := 4
def s2 : ℤ := 3

theorem vector_computation : s1 • v1 - s2 • v2 = (6, 10) :=
  sorry

end NUMINAMATH_GPT_vector_computation_l2167_216757


namespace NUMINAMATH_GPT_value_of_f_at_3_l2167_216758

def f (a c x : ℝ) : ℝ := a * x^3 + c * x + 5

theorem value_of_f_at_3 (a c : ℝ) (h : f a c (-3) = -3) : f a c 3 = 13 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_3_l2167_216758


namespace NUMINAMATH_GPT_urn_contains_four_each_color_after_six_steps_l2167_216760

noncomputable def probability_urn_four_each_color : ℚ := 2 / 7

def urn_problem (urn_initial : ℕ) (draws : ℕ) (final_urn : ℕ) (extra_balls : ℕ) : Prop :=
urn_initial = 2 ∧ draws = 6 ∧ final_urn = 8 ∧ extra_balls > 0

theorem urn_contains_four_each_color_after_six_steps :
  urn_problem 2 6 8 2 → probability_urn_four_each_color = 2 / 7 :=
by
  intro h
  cases h
  sorry

end NUMINAMATH_GPT_urn_contains_four_each_color_after_six_steps_l2167_216760


namespace NUMINAMATH_GPT_ratio_of_trout_l2167_216785

-- Definition of the conditions
def trout_caught_by_Sara : Nat := 5
def trout_caught_by_Melanie : Nat := 10

-- Theorem stating the main claim to be proved
theorem ratio_of_trout : trout_caught_by_Melanie / trout_caught_by_Sara = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_trout_l2167_216785


namespace NUMINAMATH_GPT_bridge_length_is_correct_l2167_216736

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := speed_mps * crossing_time_seconds
  total_distance - train_length

theorem bridge_length_is_correct :
  length_of_bridge 200 (60) 45 = 550.15 :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_is_correct_l2167_216736


namespace NUMINAMATH_GPT_sequence_property_l2167_216743

theorem sequence_property (a : ℕ → ℝ)
    (h_rec : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
    (h_a1 : a 1 = 1 + Real.sqrt 7)
    (h_1776 : a 1776 = 13 + Real.sqrt 7) :
    a 2009 = -1 + 2 * Real.sqrt 7 := 
    sorry

end NUMINAMATH_GPT_sequence_property_l2167_216743


namespace NUMINAMATH_GPT_terry_total_miles_l2167_216739

def total_gasoline_used := 9 + 17
def average_gas_mileage := 30

theorem terry_total_miles (M : ℕ) : 
  total_gasoline_used * average_gas_mileage = M → M = 780 :=
by
  intro h
  rw [←h]
  sorry

end NUMINAMATH_GPT_terry_total_miles_l2167_216739


namespace NUMINAMATH_GPT_third_quadrant_point_m_l2167_216780

theorem third_quadrant_point_m (m : ℤ) (h1 : 2 - m < 0) (h2 : m - 4 < 0) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_third_quadrant_point_m_l2167_216780


namespace NUMINAMATH_GPT_rohan_monthly_salary_expenses_l2167_216707

theorem rohan_monthly_salary_expenses 
    (food_expense_pct : ℝ)
    (house_rent_expense_pct : ℝ)
    (entertainment_expense_pct : ℝ)
    (conveyance_expense_pct : ℝ)
    (utilities_expense_pct : ℝ)
    (misc_expense_pct : ℝ)
    (monthly_saved_amount : ℝ)
    (entertainment_expense_increase_after_6_months : ℝ)
    (conveyance_expense_decrease_after_6_months : ℝ)
    (monthly_salary : ℝ)
    (savings_pct : ℝ)
    (new_savings_pct : ℝ) : 
    (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct = 90) → 
    (100 - (food_expense_pct + house_rent_expense_pct + entertainment_expense_pct + conveyance_expense_pct + utilities_expense_pct + misc_expense_pct) = savings_pct) → 
    (monthly_saved_amount = monthly_salary * savings_pct / 100) → 
    (entertainment_expense_pct + entertainment_expense_increase_after_6_months = 20) → 
    (conveyance_expense_pct - conveyance_expense_decrease_after_6_months = 7) → 
    (new_savings_pct = 100 - (30 + 25 + (entertainment_expense_pct + entertainment_expense_increase_after_6_months) + (conveyance_expense_pct - conveyance_expense_decrease_after_6_months) + 5 + 5)) → 
    monthly_salary = 15000 ∧ new_savings_pct = 8 := 
sorry

end NUMINAMATH_GPT_rohan_monthly_salary_expenses_l2167_216707


namespace NUMINAMATH_GPT_marcus_has_210_cards_l2167_216731

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the increment of baseball cards Marcus has over Carter
def increment : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + increment

-- Prove that Marcus has 210 baseball cards
theorem marcus_has_210_cards : marcus_cards = 210 :=
by simp [marcus_cards, carter_cards, increment]

end NUMINAMATH_GPT_marcus_has_210_cards_l2167_216731


namespace NUMINAMATH_GPT_largest_number_in_systematic_sample_l2167_216753

theorem largest_number_in_systematic_sample (n_products : ℕ) (start : ℕ) (interval : ℕ) (sample_size : ℕ) (largest_number : ℕ)
  (h1 : n_products = 500)
  (h2 : start = 7)
  (h3 : interval = 25)
  (h4 : sample_size = n_products / interval)
  (h5 : sample_size = 20)
  (h6 : largest_number = start + interval * (sample_size - 1))
  (h7 : largest_number = 482) :
  largest_number = 482 := 
  sorry

end NUMINAMATH_GPT_largest_number_in_systematic_sample_l2167_216753


namespace NUMINAMATH_GPT_A_inter_B_empty_l2167_216788

def Z_plus := { n : ℤ // 0 < n }

def A : Set ℤ := { x | ∃ n : Z_plus, x = 2 * (n.1) - 1 }
def B : Set ℤ := { y | ∃ x ∈ A, y = 3 * x - 1 }

theorem A_inter_B_empty : A ∩ B = ∅ :=
by {
  sorry
}

end NUMINAMATH_GPT_A_inter_B_empty_l2167_216788


namespace NUMINAMATH_GPT_equivalent_statements_l2167_216782

variable (P Q R : Prop)

theorem equivalent_statements :
  ((¬ P ∧ ¬ Q) → R) ↔ (P ∨ Q ∨ R) :=
sorry

end NUMINAMATH_GPT_equivalent_statements_l2167_216782


namespace NUMINAMATH_GPT_meal_total_cost_l2167_216798

theorem meal_total_cost (x : ℝ) (h_initial: x/5 - 15 = x/8) : x = 200 :=
by sorry

end NUMINAMATH_GPT_meal_total_cost_l2167_216798


namespace NUMINAMATH_GPT_exists_triplet_with_gcd_conditions_l2167_216704

-- Given the conditions as definitions in Lean.
variables (S : Set ℕ)
variable [Infinite S] -- S is an infinite set of positive integers.
variables {a b c d x y z : ℕ}
variable (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S)
variable (hdistinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) 
variable (hgcd_neq : gcd a b ≠ gcd c d)

-- The formal proof statement.
theorem exists_triplet_with_gcd_conditions :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ gcd x y = gcd y z ∧ gcd y z ≠ gcd z x :=
sorry

end NUMINAMATH_GPT_exists_triplet_with_gcd_conditions_l2167_216704


namespace NUMINAMATH_GPT_random_events_l2167_216765

-- Define what it means for an event to be random
def is_random_event (e : Prop) : Prop := ∃ (h : Prop), e ∨ ¬e

-- Define the events based on the problem statements
def event1 := ∃ (good_cups : ℕ), good_cups = 3
def event2 := ∃ (half_hit_targets : ℕ), half_hit_targets = 50
def event3 := ∃ (correct_digit : ℕ), correct_digit = 1
def event4 := true -- Opposite charges attract each other, which is always true
def event5 := ∃ (first_prize : ℕ), first_prize = 1

-- State the problem as a theorem
theorem random_events :
  is_random_event event1 ∧ is_random_event event2 ∧ is_random_event event3 ∧ is_random_event event5 :=
by
  sorry

end NUMINAMATH_GPT_random_events_l2167_216765


namespace NUMINAMATH_GPT_cashier_correction_l2167_216763

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

end NUMINAMATH_GPT_cashier_correction_l2167_216763


namespace NUMINAMATH_GPT_discount_rate_l2167_216708

theorem discount_rate (cost_shoes cost_socks cost_bag paid_price total_cost discount_amount amount_subject_to_discount discount_rate: ℝ)
  (h1 : cost_shoes = 74)
  (h2 : cost_socks = 2 * 2)
  (h3 : cost_bag = 42)
  (h4 : paid_price = 118)
  (h5 : total_cost = cost_shoes + cost_socks + cost_bag)
  (h6 : discount_amount = total_cost - paid_price)
  (h7 : amount_subject_to_discount = total_cost - 100)
  (h8 : discount_rate = (discount_amount / amount_subject_to_discount) * 100) :
  discount_rate = 10 := sorry

end NUMINAMATH_GPT_discount_rate_l2167_216708


namespace NUMINAMATH_GPT_sum_of_variables_l2167_216771

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_of_variables (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧ log 3 (log 4 (log 2 y)) = 0 ∧ log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_variables_l2167_216771


namespace NUMINAMATH_GPT_jackie_break_duration_l2167_216787

noncomputable def push_ups_no_breaks : ℕ := 30

noncomputable def push_ups_with_breaks : ℕ := 22

noncomputable def total_breaks : ℕ := 2

theorem jackie_break_duration :
  (5 * 6 - push_ups_with_breaks) * (10 / 5) / total_breaks = 8 := by
-- Given that
-- 1) Jackie does 5 push-ups in 10 seconds
-- 2) Jackie takes 2 breaks in one minute and performs 22 push-ups
-- We need to prove the duration of each break
sorry

end NUMINAMATH_GPT_jackie_break_duration_l2167_216787


namespace NUMINAMATH_GPT_octagon_area_l2167_216712

noncomputable def area_of_octagon_concentric_squares : ℚ :=
  let m := 1
  let n := 8
  (m + n)

theorem octagon_area (O : ℝ × ℝ) (side_small side_large : ℚ) (AB : ℚ) 
  (h1 : side_small = 2) (h2 : side_large = 3) (h3 : AB = 1/4) : 
  area_of_octagon_concentric_squares = 9 := 
  by
  have h_area : 1/8 = 1/8 := rfl
  sorry

end NUMINAMATH_GPT_octagon_area_l2167_216712


namespace NUMINAMATH_GPT_decimal_to_base_five_correct_l2167_216727

theorem decimal_to_base_five_correct : 
  ∃ (d0 d1 d2 d3 : ℕ), 256 = d3 * 5^3 + d2 * 5^2 + d1 * 5^1 + d0 * 5^0 ∧ 
                          d3 = 2 ∧ d2 = 0 ∧ d1 = 1 ∧ d0 = 1 :=
by sorry

end NUMINAMATH_GPT_decimal_to_base_five_correct_l2167_216727


namespace NUMINAMATH_GPT_domain_of_f_l2167_216748

noncomputable def f (x : ℝ) := 1 / ((x - 3) + (x - 6))

theorem domain_of_f :
  (∀ x : ℝ, x ≠ 9/2 → ∃ y : ℝ, f x = y) ∧ (∀ x : ℝ, x = 9/2 → ¬ (∃ y : ℝ, f x = y)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2167_216748


namespace NUMINAMATH_GPT_monthly_salary_l2167_216744

variables (S : ℕ) (h1 : S * 20 / 100 * 96 / 100 = 4 * 250)

theorem monthly_salary : S = 6250 :=
by sorry

end NUMINAMATH_GPT_monthly_salary_l2167_216744


namespace NUMINAMATH_GPT_antonieta_tickets_needed_l2167_216728

-- Definitions based on conditions:
def ferris_wheel_tickets : ℕ := 6
def roller_coaster_tickets : ℕ := 5
def log_ride_tickets : ℕ := 7
def antonieta_initial_tickets : ℕ := 2

-- Theorem to prove the required number of tickets Antonieta should buy
theorem antonieta_tickets_needed : ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - antonieta_initial_tickets = 16 :=
by
  sorry

end NUMINAMATH_GPT_antonieta_tickets_needed_l2167_216728


namespace NUMINAMATH_GPT_inequality_solution_set_minimum_value_expression_l2167_216703

-- Definition of the function f
def f (x : ℝ) : ℝ := |x + 2| + 2 * |x - 1|

-- Inequality solution set for f(x) ≤ 4
theorem inequality_solution_set :
  { x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3 } = { x : ℝ | f x ≤ 4 } := 
sorry

-- Minimum value of the given expression given conditions on a and b
theorem minimum_value_expression (a b : ℝ) (h1 : a > 1) (h2 : b > 0)
  (h3 : a + 2 * b = 3) :
  (1 / (a - 1)) + (2 / b) = 9 / 2 := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_minimum_value_expression_l2167_216703


namespace NUMINAMATH_GPT_greatest_distance_P_D_l2167_216741

noncomputable def greatest_distance_from_D (P : ℝ × ℝ) (A B C : ℝ × ℝ) (D : ℝ × ℝ) : ℝ :=
  let u := (P.1 - A.1)^2 + (P.2 - A.2)^2
  let v := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let w := (P.1 - C.1)^2 + (P.2 - C.2)^2
  if u + v = w + 1 then ((P.1 - D.1)^2 + (P.2 - D.2)^2).sqrt else 0

theorem greatest_distance_P_D (P : ℝ × ℝ) (u v w : ℝ)
  (h1 : u^2 + v^2 = w^2 + 1) :
  greatest_distance_from_D P (0,0) (2,0) (2,2) (0,2) = 5 :=
sorry

end NUMINAMATH_GPT_greatest_distance_P_D_l2167_216741


namespace NUMINAMATH_GPT_pipe_Q_fill_time_l2167_216735

theorem pipe_Q_fill_time (x : ℝ) (h1 : 6 > 0)
    (h2 : 24 > 0)
    (h3 : 3.4285714285714284 > 0)
    (h4 : (1 / 6) + (1 / x) + (1 / 24) = 1 / 3.4285714285714284) :
    x = 8 := by
  sorry

end NUMINAMATH_GPT_pipe_Q_fill_time_l2167_216735


namespace NUMINAMATH_GPT_toads_per_acre_l2167_216715

theorem toads_per_acre (b g : ℕ) (h₁ : b = 25 * g)
  (h₂ : b / 4 = 50) : g = 8 :=
by
  -- Condition h₁: For every green toad, there are 25 brown toads.
  -- Condition h₂: One-quarter of the brown toads are spotted, and there are 50 spotted brown toads per acre.
  sorry

end NUMINAMATH_GPT_toads_per_acre_l2167_216715


namespace NUMINAMATH_GPT_ways_to_sum_2022_l2167_216759

theorem ways_to_sum_2022 : 
  ∃ n : ℕ, (∀ a b : ℕ, (2022 = 2 * a + 3 * b) ∧ n = (b - a) / 4 ∧ n = 338) := 
sorry

end NUMINAMATH_GPT_ways_to_sum_2022_l2167_216759


namespace NUMINAMATH_GPT_angle_AOD_128_57_l2167_216754

-- Define angles as real numbers
variables {α β : ℝ}

-- Define the conditions
def perp (v1 v2 : ℝ) := v1 = 90 - v2

theorem angle_AOD_128_57 
  (h1 : perp α 90)
  (h2 : perp β 90)
  (h3 : α = 2.5 * β) :
  α = 128.57 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_angle_AOD_128_57_l2167_216754


namespace NUMINAMATH_GPT_range_of_a_l2167_216714

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≤ 0) → a ≤ -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2167_216714


namespace NUMINAMATH_GPT_axis_of_symmetry_y_range_l2167_216701

/-- 
The equation of the curve is given by |x| + y^2 - 3y = 0.
We aim to prove two properties:
1. The axis of symmetry of this curve is x = 0.
2. The range of possible values for y is [0, 3].
-/
noncomputable def curve (x y : ℝ) : ℝ := |x| + y^2 - 3*y

theorem axis_of_symmetry : ∀ x y : ℝ, curve x y = 0 → x = 0 :=
sorry

theorem y_range : ∀ y : ℝ, ∃ x : ℝ, curve x y = 0 → (0 ≤ y ∧ y ≤ 3) :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_y_range_l2167_216701


namespace NUMINAMATH_GPT_lenny_remaining_amount_l2167_216717

theorem lenny_remaining_amount :
  let initial_amount := 270
  let console_price := 149
  let console_discount := 0.15 * console_price
  let final_console_price := console_price - console_discount
  let groceries_price := 60
  let groceries_discount := 0.10 * groceries_price
  let final_groceries_price := groceries_price - groceries_discount
  let lunch_cost := 30
  let magazine_cost := 3.99
  let total_expenses := final_console_price + final_groceries_price + lunch_cost + magazine_cost
  initial_amount - total_expenses = 55.36 :=
by
  sorry

end NUMINAMATH_GPT_lenny_remaining_amount_l2167_216717


namespace NUMINAMATH_GPT_probability_gpa_at_least_3_is_2_over_9_l2167_216775

def gpa_points (grade : ℕ) : ℕ :=
  match grade with
  | 4 => 4 -- A
  | 3 => 3 -- B
  | 2 => 2 -- C
  | 1 => 1 -- D
  | _ => 0 -- otherwise

def probability_of_GPA_at_least_3 : ℚ :=
  let points_physics := gpa_points 4
  let points_chemistry := gpa_points 4
  let points_biology := gpa_points 3
  let total_known_points := points_physics + points_chemistry + points_biology
  let required_points := 18 - total_known_points -- 18 points needed in total for a GPA of at least 3.0
  -- Probabilities in Mathematics:
  let prob_math_A := 1 / 9
  let prob_math_B := 4 / 9
  let prob_math_C :=  4 / 9
  -- Probabilities in Sociology:
  let prob_soc_A := 1 / 3
  let prob_soc_B := 1 / 3
  let prob_soc_C := 1 / 3
  -- Calculate the total probability of achieving at least 7 points from Mathematics and Sociology
  let prob_case_1 := prob_math_A * prob_soc_A -- Both A in Mathematics and Sociology
  let prob_case_2 := prob_math_A * prob_soc_B -- A in Mathematics and B in Sociology
  let prob_case_3 := prob_math_B * prob_soc_A -- B in Mathematics and A in Sociology
  prob_case_1 + prob_case_2 + prob_case_3 -- Total Probability

theorem probability_gpa_at_least_3_is_2_over_9 : probability_of_GPA_at_least_3 = 2 / 9 :=
by sorry

end NUMINAMATH_GPT_probability_gpa_at_least_3_is_2_over_9_l2167_216775


namespace NUMINAMATH_GPT_floor_sum_eq_126_l2167_216700

-- Define the problem conditions
variable (a b c d : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
variable (h5 : a^2 + b^2 = 2008) (h6 : c^2 + d^2 = 2008)
variable (h7 : a * c = 1000) (h8 : b * d = 1000)

-- Prove the solution
theorem floor_sum_eq_126 : ⌊a + b + c + d⌋ = 126 :=
by
  sorry

end NUMINAMATH_GPT_floor_sum_eq_126_l2167_216700


namespace NUMINAMATH_GPT_complete_square_ratio_l2167_216747

theorem complete_square_ratio (k : ℝ) :
  ∃ c p q : ℝ, 
    8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q ∧ 
    q / p = -142 / 3 :=
sorry

end NUMINAMATH_GPT_complete_square_ratio_l2167_216747


namespace NUMINAMATH_GPT_kiwis_to_add_for_25_percent_oranges_l2167_216789

theorem kiwis_to_add_for_25_percent_oranges :
  let oranges := 24
  let kiwis := 30
  let apples := 15
  let bananas := 20
  let total_fruits := oranges + kiwis + apples + bananas
  let target_total_fruits := (oranges : ℝ) / 0.25
  let fruits_to_add := target_total_fruits - (total_fruits : ℝ)
  fruits_to_add = 7 := by
  sorry

end NUMINAMATH_GPT_kiwis_to_add_for_25_percent_oranges_l2167_216789


namespace NUMINAMATH_GPT_area_of_right_triangle_with_hypotenuse_and_angle_l2167_216718

theorem area_of_right_triangle_with_hypotenuse_and_angle 
  (hypotenuse : ℝ) (angle : ℝ) (h_hypotenuse : hypotenuse = 9 * Real.sqrt 3) (h_angle : angle = 30) : 
  ∃ (area : ℝ), area = 364.5 := 
by
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_with_hypotenuse_and_angle_l2167_216718


namespace NUMINAMATH_GPT_angle_B_of_isosceles_triangle_l2167_216738

theorem angle_B_of_isosceles_triangle (A B C : ℝ) (h_iso : (A = B ∨ A = C) ∨ (B = C ∨ B = A) ∨ (C = A ∨ C = B)) (h_angle_A : A = 70) :
  B = 70 ∨ B = 55 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_of_isosceles_triangle_l2167_216738
