import Mathlib

namespace NUMINAMATH_GPT_sum_of_cubes_l2045_204500

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l2045_204500


namespace NUMINAMATH_GPT_good_students_options_l2045_204515

variables (E B : ℕ)

-- Define the condition that the class has 25 students
def total_students : Prop := E + B = 25

-- Define the condition given by the first group of students
def first_group_condition : Prop := B > 12

-- Define the condition given by the second group of students
def second_group_condition : Prop := B = 3 * (E - 1)

-- Define the problem statement
theorem good_students_options (E B : ℕ) :
  total_students E B → first_group_condition B → second_group_condition E B → (E = 5 ∨ E = 7) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_good_students_options_l2045_204515


namespace NUMINAMATH_GPT_bianca_drawing_time_at_home_l2045_204548

-- Define the conditions
def drawing_time_at_school : ℕ := 22
def total_drawing_time : ℕ := 41

-- Define the calculation for drawing time at home
def drawing_time_at_home : ℕ := total_drawing_time - drawing_time_at_school

-- The proof goal
theorem bianca_drawing_time_at_home : drawing_time_at_home = 19 := by
  sorry

end NUMINAMATH_GPT_bianca_drawing_time_at_home_l2045_204548


namespace NUMINAMATH_GPT_xy_condition_l2045_204563

variable (x y : ℝ) -- This depends on the problem context specifying real numbers.

theorem xy_condition (h : x ≠ 0 ∧ y ≠ 0) : (x + y = 0 ↔ y / x + x / y = -2) :=
  sorry

end NUMINAMATH_GPT_xy_condition_l2045_204563


namespace NUMINAMATH_GPT_fred_final_baseball_cards_l2045_204506

-- Conditions
def initial_cards : ℕ := 25
def sold_to_melanie : ℕ := 7
def traded_with_kevin : ℕ := 3
def bought_from_alex : ℕ := 5

-- Proof statement (Lean theorem)
theorem fred_final_baseball_cards : initial_cards - sold_to_melanie - traded_with_kevin + bought_from_alex = 20 := by
  sorry

end NUMINAMATH_GPT_fred_final_baseball_cards_l2045_204506


namespace NUMINAMATH_GPT_sum_of_c_l2045_204526

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℕ :=
  2^(n - 1)

-- Define the sequence c_n
def c (n : ℕ) : ℕ :=
  a n * b n

-- Define the sum S_n of the first n terms of c_n
def S (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => c (i + 1))

-- The main Lean statement
theorem sum_of_c (n : ℕ) : S n = 3 + (n - 1) * 2^(n + 1) :=
  sorry

end NUMINAMATH_GPT_sum_of_c_l2045_204526


namespace NUMINAMATH_GPT_total_number_of_squares_up_to_50th_ring_l2045_204571

def number_of_squares_up_to_50th_ring : Nat :=
  let central_square := 1
  let sum_rings := (50 * (50 + 1)) * 4  -- Using the formula for arithmetic series sum where a = 8 and d = 8 and n = 50
  central_square + sum_rings

theorem total_number_of_squares_up_to_50th_ring : number_of_squares_up_to_50th_ring = 10201 :=
  by  -- This statement means we believe the theorem is true and will be proven.
    sorry                                                      -- Proof omitted, will need to fill this in later

end NUMINAMATH_GPT_total_number_of_squares_up_to_50th_ring_l2045_204571


namespace NUMINAMATH_GPT_midpoint_trajectory_of_chord_l2045_204503

theorem midpoint_trajectory_of_chord {x y : ℝ} :
  (∃ (A B : ℝ × ℝ), 
    (A.1^2 / 3 + A.2^2 = 1) ∧ 
    (B.1^2 / 3 + B.2^2 = 1) ∧ 
    ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (x, y) ∧ 
    ∃ t : ℝ, ((-1, 0) = ((1 - t) * A.1 + t * B.1, (1 - t) * A.2 + t * B.2))) -> 
  x^2 + x + 3 * y^2 = 0 :=
by sorry

end NUMINAMATH_GPT_midpoint_trajectory_of_chord_l2045_204503


namespace NUMINAMATH_GPT_trench_digging_l2045_204534

theorem trench_digging 
  (t : ℝ) (T : ℝ) (work_units : ℝ)
  (h1 : 4 * t = 10)
  (h2 : T = 5 * t) :
  work_units = 80 :=
by
  sorry

end NUMINAMATH_GPT_trench_digging_l2045_204534


namespace NUMINAMATH_GPT_inequality_holds_l2045_204521

variable (a b c : ℝ)

theorem inequality_holds : 
  (a * b + b * c + c * a - 1)^2 ≤ (a^2 + 1) * (b^2 + 1) * (c^2 + 1) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_holds_l2045_204521


namespace NUMINAMATH_GPT_Jason_seashells_l2045_204542

theorem Jason_seashells (initial_seashells given_to_Tim remaining_seashells : ℕ) :
  initial_seashells = 49 → given_to_Tim = 13 → remaining_seashells = initial_seashells - given_to_Tim →
  remaining_seashells = 36 :=
by intros; sorry

end NUMINAMATH_GPT_Jason_seashells_l2045_204542


namespace NUMINAMATH_GPT_mary_rental_hours_l2045_204559

def ocean_bike_fixed_fee := 17
def ocean_bike_hourly_rate := 7
def total_paid := 80

def calculate_hours (fixed_fee : Nat) (hourly_rate : Nat) (total_amount : Nat) : Nat :=
  (total_amount - fixed_fee) / hourly_rate

theorem mary_rental_hours :
  calculate_hours ocean_bike_fixed_fee ocean_bike_hourly_rate total_paid = 9 :=
by
  sorry

end NUMINAMATH_GPT_mary_rental_hours_l2045_204559


namespace NUMINAMATH_GPT_exists_q_no_zero_in_decimal_l2045_204527

theorem exists_q_no_zero_in_decimal : ∃ q : ℕ, ∀ (d : ℕ), q * 2 ^ 1967 ≠ 10 * d := 
sorry

end NUMINAMATH_GPT_exists_q_no_zero_in_decimal_l2045_204527


namespace NUMINAMATH_GPT_segment_length_l2045_204574

theorem segment_length (x y : ℝ) (A B : ℝ × ℝ) 
  (h1 : A.2^2 = 4 * A.1) 
  (h2 : B.2^2 = 4 * B.1) 
  (h3 : A.2 = 2 * A.1 - 2)
  (h4 : B.2 = 2 * B.1 - 2)
  (h5 : A ≠ B) :
  dist A B = 5 :=
sorry

end NUMINAMATH_GPT_segment_length_l2045_204574


namespace NUMINAMATH_GPT_circle_radius_l2045_204529

theorem circle_radius (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 10) : r = 20 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l2045_204529


namespace NUMINAMATH_GPT_valid_parameterizations_l2045_204549

noncomputable def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

def is_valid_parameterization (p d : ℝ × ℝ) (m b : ℝ) : Prop :=
  lies_on_line p m b ∧ is_scalar_multiple d (2, 1)

theorem valid_parameterizations :
  (is_valid_parameterization (7, 18) (-1, -2) 2 4) ∧
  (is_valid_parameterization (1, 6) (5, 10) 2 4) ∧
  (is_valid_parameterization (2, 8) (20, 40) 2 4) ∧
  ¬ (is_valid_parameterization (-4, -4) (1, -1) 2 4) ∧
  ¬ (is_valid_parameterization (-3, -2) (0.5, 1) 2 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_valid_parameterizations_l2045_204549


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2045_204505

variable (a : ℕ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a^2 / (1 - 2 / a) = 7 / 5 :=
by
  -- Assign the condition
  let a := 5
  sorry -- skip the proof

end NUMINAMATH_GPT_simplify_and_evaluate_l2045_204505


namespace NUMINAMATH_GPT_sum_of_two_integers_l2045_204587

theorem sum_of_two_integers (a b : ℕ) (h1 : a * b + a + b = 113) (h2 : Nat.gcd a b = 1) (h3 : a < 25) (h4 : b < 25) : a + b = 23 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_integers_l2045_204587


namespace NUMINAMATH_GPT_time_for_C_to_complete_work_l2045_204519

variable (A B C : ℕ) (R : ℚ)

def work_completion_in_days (days : ℕ) (portion : ℚ) :=
  portion = 1 / days

theorem time_for_C_to_complete_work :
  work_completion_in_days A 8 →
  work_completion_in_days B 12 →
  work_completion_in_days (A + B + C) 4 →
  C = 24 :=
by
  sorry

end NUMINAMATH_GPT_time_for_C_to_complete_work_l2045_204519


namespace NUMINAMATH_GPT_flavoring_ratio_comparison_l2045_204566

theorem flavoring_ratio_comparison (f_st cs_st w_st : ℕ) (f_sp cs_sp w_sp : ℕ) :
  f_st = 1 → cs_st = 12 → w_st = 30 →
  w_sp = 75 → cs_sp = 5 →
  f_sp / w_sp = f_st / (2 * w_st) →
  (f_st / cs_st) * 3 = f_sp / cs_sp :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_flavoring_ratio_comparison_l2045_204566


namespace NUMINAMATH_GPT_sum_of_three_squares_l2045_204537

theorem sum_of_three_squares (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_three_squares_l2045_204537


namespace NUMINAMATH_GPT_correct_inequality_l2045_204588

theorem correct_inequality :
  (1 / 2)^(2 / 3) < (1 / 2)^(1 / 3) ∧ (1 / 2)^(1 / 3) < 1 :=
by sorry

end NUMINAMATH_GPT_correct_inequality_l2045_204588


namespace NUMINAMATH_GPT_vertical_axis_residuals_of_residual_plot_l2045_204550

theorem vertical_axis_residuals_of_residual_plot :
  ∀ (vertical_axis : Type), 
  (vertical_axis = Residuals ∨ 
   vertical_axis = SampleNumber ∨ 
   vertical_axis = EstimatedValue) →
  (vertical_axis = Residuals) :=
by
  sorry

end NUMINAMATH_GPT_vertical_axis_residuals_of_residual_plot_l2045_204550


namespace NUMINAMATH_GPT_increasing_exponential_function_range_l2045_204562

theorem increasing_exponential_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ (x : ℝ), f x = a ^ x) 
    (h2 : a > 0)
    (h3 : a ≠ 1)
    (h4 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) : a > 1 := 
sorry

end NUMINAMATH_GPT_increasing_exponential_function_range_l2045_204562


namespace NUMINAMATH_GPT_range_of_m_l2045_204564

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| > 4) ↔ m > 3 ∨ m < -5 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2045_204564


namespace NUMINAMATH_GPT_earnings_of_r_l2045_204543

theorem earnings_of_r (P Q R : ℕ) (h1 : 9 * (P + Q + R) = 1710) (h2 : 5 * (P + R) = 600) (h3 : 7 * (Q + R) = 910) : 
  R = 60 :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_earnings_of_r_l2045_204543


namespace NUMINAMATH_GPT_orchids_initially_l2045_204539

-- Definitions and Conditions
def initial_orchids (current_orchids: ℕ) (cut_orchids: ℕ) : ℕ :=
  current_orchids + cut_orchids

-- Proof statement
theorem orchids_initially (current_orchids: ℕ) (cut_orchids: ℕ) : initial_orchids current_orchids cut_orchids = 3 :=
by 
  have h1 : current_orchids = 7 := sorry
  have h2 : cut_orchids = 4 := sorry
  have h3 : initial_orchids current_orchids cut_orchids = 7 + 4 := sorry
  have h4 : initial_orchids current_orchids cut_orchids = 3 := sorry
  sorry

end NUMINAMATH_GPT_orchids_initially_l2045_204539


namespace NUMINAMATH_GPT_trainers_hours_split_equally_l2045_204536

noncomputable def dolphins := 12
noncomputable def hours_per_dolphin := 5
noncomputable def trainers := 4

theorem trainers_hours_split_equally :
  (dolphins * hours_per_dolphin) / trainers = 15 :=
by
  sorry

end NUMINAMATH_GPT_trainers_hours_split_equally_l2045_204536


namespace NUMINAMATH_GPT_lisa_speed_correct_l2045_204533

def eugene_speed := 5

def carlos_speed := (3 / 4) * eugene_speed

def lisa_speed := (4 / 3) * carlos_speed

theorem lisa_speed_correct : lisa_speed = 5 := by
  sorry

end NUMINAMATH_GPT_lisa_speed_correct_l2045_204533


namespace NUMINAMATH_GPT_inequality_positive_reals_l2045_204576

theorem inequality_positive_reals (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_GPT_inequality_positive_reals_l2045_204576


namespace NUMINAMATH_GPT_focus_of_hyperbola_l2045_204575

theorem focus_of_hyperbola (m : ℝ) :
  let focus_parabola := (0, 4)
  let focus_hyperbola_upper := (0, 4)
  ∃ focus_parabola, ∃ focus_hyperbola_upper, 
    (focus_parabola = (0, 4)) ∧ (focus_hyperbola_upper = (0, 4)) ∧ 
    (3 + m = 16) → m = 13 :=
by
  sorry

end NUMINAMATH_GPT_focus_of_hyperbola_l2045_204575


namespace NUMINAMATH_GPT_factor_of_polynomial_l2045_204586

theorem factor_of_polynomial :
  (x^4 + 4 * x^2 + 16) % (x^2 + 4) = 0 :=
sorry

end NUMINAMATH_GPT_factor_of_polynomial_l2045_204586


namespace NUMINAMATH_GPT_positive_integer_sum_representation_l2045_204551

theorem positive_integer_sum_representation :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → ∃ (a : Fin 2004 → ℕ), 
    (∀ i j : Fin 2004, i < j → a i < a j) ∧ 
    (∀ i : Fin 2003, a i ∣ a (i + 1)) ∧
    (n = (Finset.univ.sum a)) := 
sorry

end NUMINAMATH_GPT_positive_integer_sum_representation_l2045_204551


namespace NUMINAMATH_GPT_triangle_possible_side_lengths_l2045_204570

theorem triangle_possible_side_lengths (x : ℕ) (hx : x > 0) (h1 : x^2 + 9 > 12) (h2 : x^2 + 12 > 9) (h3 : 9 + 12 > x^2) : x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_possible_side_lengths_l2045_204570


namespace NUMINAMATH_GPT_sin_15_mul_sin_75_l2045_204541

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_sin_15_mul_sin_75_l2045_204541


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_45_l2045_204598

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_45_l2045_204598


namespace NUMINAMATH_GPT_eggs_collected_week_l2045_204514

def num_chickens : ℕ := 6
def num_ducks : ℕ := 4
def num_geese : ℕ := 2
def eggs_per_chicken : ℕ := 3
def eggs_per_duck : ℕ := 2
def eggs_per_goose : ℕ := 1

def eggs_per_day (num_birds eggs_per_bird : ℕ) : ℕ := num_birds * eggs_per_bird

def eggs_collected_monday_to_saturday : ℕ :=
  6 * (eggs_per_day num_chickens eggs_per_chicken +
       eggs_per_day num_ducks eggs_per_duck +
       eggs_per_day num_geese eggs_per_goose)

def eggs_collected_sunday : ℕ :=
  eggs_per_day num_chickens (eggs_per_chicken - 1) +
  eggs_per_day num_ducks (eggs_per_duck - 1) +
  eggs_per_day num_geese (eggs_per_goose - 1)

def total_eggs_collected : ℕ :=
  eggs_collected_monday_to_saturday + eggs_collected_sunday

theorem eggs_collected_week : total_eggs_collected = 184 :=
by sorry

end NUMINAMATH_GPT_eggs_collected_week_l2045_204514


namespace NUMINAMATH_GPT_first_reduction_is_12_percent_l2045_204535

theorem first_reduction_is_12_percent (P : ℝ) (x : ℝ) (h1 : (1 - x / 100) * 0.9 * P = 0.792 * P) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_first_reduction_is_12_percent_l2045_204535


namespace NUMINAMATH_GPT_time_interval_for_birth_and_death_rates_l2045_204501

theorem time_interval_for_birth_and_death_rates
  (birth_rate : ℝ)
  (death_rate : ℝ)
  (population_net_increase_per_day : ℝ)
  (number_of_minutes_per_day : ℝ)
  (net_increase_per_interval : ℝ)
  (time_intervals_per_day : ℝ)
  (time_interval_in_minutes : ℝ):

  birth_rate = 10 →
  death_rate = 2 →
  population_net_increase_per_day = 345600 →
  number_of_minutes_per_day = 1440 →
  net_increase_per_interval = birth_rate - death_rate →
  time_intervals_per_day = population_net_increase_per_day / net_increase_per_interval →
  time_interval_in_minutes = number_of_minutes_per_day / time_intervals_per_day →
  time_interval_in_minutes = 48 :=
by
  intros
  sorry

end NUMINAMATH_GPT_time_interval_for_birth_and_death_rates_l2045_204501


namespace NUMINAMATH_GPT_Aunt_Lucy_gift_correct_l2045_204538

def Jade_initial : ℕ := 38
def Julia_initial : ℕ := Jade_initial / 2
def Jack_initial : ℕ := 12
def John_initial : ℕ := 15
def Jane_initial : ℕ := 20

def Aunt_Mary_gift : ℕ := 65
def Aunt_Susan_gift : ℕ := 70

def total_initial : ℕ :=
  Jade_initial + Julia_initial + Jack_initial + John_initial + Jane_initial

def total_after_gifts : ℕ := 225
def total_gifts : ℕ := total_after_gifts - total_initial
def Aunt_Lucy_gift : ℕ := total_gifts - (Aunt_Mary_gift + Aunt_Susan_gift)

theorem Aunt_Lucy_gift_correct :
  Aunt_Lucy_gift = total_after_gifts - total_initial - (Aunt_Mary_gift + Aunt_Susan_gift) := by
  sorry

end NUMINAMATH_GPT_Aunt_Lucy_gift_correct_l2045_204538


namespace NUMINAMATH_GPT_triangle_problem_l2045_204544

open Real

theorem triangle_problem (a b S : ℝ) (A B : ℝ) (hA_cos : cos A = (sqrt 6) / 3) (hA_val : a = 3) (hB_val : B = A + π / 2):
  b = 3 * sqrt 2 ∧
  S = (3 * sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_problem_l2045_204544


namespace NUMINAMATH_GPT_markup_calculation_l2045_204509

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.25
def net_profit : ℝ := 12

def overhead := purchase_price * overhead_percentage
def total_cost := purchase_price + overhead
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_calculation : markup = 24 := by
  sorry

end NUMINAMATH_GPT_markup_calculation_l2045_204509


namespace NUMINAMATH_GPT_polynomial_evaluation_l2045_204513

noncomputable def Q (x : ℝ) : ℝ :=
  x^4 + x^3 + 2 * x

theorem polynomial_evaluation :
  Q (3) = 114 := by
  -- We assume the conditions implicitly in this equivalence.
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l2045_204513


namespace NUMINAMATH_GPT_smallest_same_terminal_1000_l2045_204502

def has_same_terminal_side (theta phi : ℝ) : Prop :=
  ∃ n : ℤ, theta = phi + n * 360

theorem smallest_same_terminal_1000 : ∀ θ : ℝ,
  θ ≥ 0 → θ < 360 → has_same_terminal_side θ 1000 → θ = 280 :=
by
  sorry

end NUMINAMATH_GPT_smallest_same_terminal_1000_l2045_204502


namespace NUMINAMATH_GPT_find_l_find_C3_l2045_204594

-- Circle definitions
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0

-- Given line passes through common points of C1 and C2
theorem find_l (x y : ℝ) (h1 : C1 x y) (h2 : C2 x y) : x = 1 := by
  sorry

-- Circle C3 passes through intersection points of C1 and C2, and its center lies on y = x
def C3 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def on_line_y_eq_x (x y : ℝ) : Prop := y = x

theorem find_C3 (x y : ℝ) (hx : C3 x y) (hy : on_line_y_eq_x x y) : (x - 1)^2 + (y - 1)^2 = 1 := by
  sorry

end NUMINAMATH_GPT_find_l_find_C3_l2045_204594


namespace NUMINAMATH_GPT_geometric_series_sum_l2045_204593

theorem geometric_series_sum : 
  let a := 1 
  let r := 2 
  let n := 11 
  let S_n := (a * (1 - r^n)) / (1 - r)
  S_n = 2047 := by
  -- The proof steps would normally go here.
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2045_204593


namespace NUMINAMATH_GPT_perfect_squares_digit_4_5_6_l2045_204546

theorem perfect_squares_digit_4_5_6 (n : ℕ) (hn : n^2 < 2000) : 
  (∃ k : ℕ, k = 18) :=
  sorry

end NUMINAMATH_GPT_perfect_squares_digit_4_5_6_l2045_204546


namespace NUMINAMATH_GPT_monster_ratio_l2045_204599

theorem monster_ratio (r : ℝ) :
  (121 + 121 * r + 121 * r^2 = 847) → r = 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_monster_ratio_l2045_204599


namespace NUMINAMATH_GPT_find_sale_month_4_l2045_204569

-- Definitions based on the given conditions
def avg_sale_per_month : ℕ := 6500
def num_months : ℕ := 6
def sale_month_1 : ℕ := 6435
def sale_month_2 : ℕ := 6927
def sale_month_3 : ℕ := 6855
def sale_month_5 : ℕ := 6562
def sale_month_6 : ℕ := 4991

theorem find_sale_month_4 : 
  (avg_sale_per_month * num_months) - (sale_month_1 + sale_month_2 + sale_month_3 + sale_month_5 + sale_month_6) = 7230 :=
by
  -- The proof will be provided below
  sorry

end NUMINAMATH_GPT_find_sale_month_4_l2045_204569


namespace NUMINAMATH_GPT_correct_statements_l2045_204554

-- Define the statements
def statement_1 := true
def statement_2 := false
def statement_3 := true
def statement_4 := true

-- Define a function to count the number of true statements
def num_correct_statements (s1 s2 s3 s4 : Bool) : Nat :=
  [s1, s2, s3, s4].countP id

-- Define the theorem to prove that the number of correct statements is 3
theorem correct_statements :
  num_correct_statements statement_1 statement_2 statement_3 statement_4 = 3 :=
by
  -- You can use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_correct_statements_l2045_204554


namespace NUMINAMATH_GPT_geometric_sequence_properties_l2045_204591

theorem geometric_sequence_properties (a : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_geom : ∀ (m k : ℕ), a (m + k) = a m * q ^ k) 
  (h_sum : a 1 + a n = 66) 
  (h_prod : a 3 * a (n - 2) = 128) 
  (h_s_n : (a 1 * (1 - q ^ n)) / (1 - q) = 126) : 
  n = 6 ∧ (q = 2 ∨ q = 1/2) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l2045_204591


namespace NUMINAMATH_GPT_distinct_bead_arrangements_on_bracelet_l2045_204524

open Nat

-- Definition of factorial
def fact : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * fact n

-- Theorem stating the number of distinct arrangements of 7 beads on a bracelet
theorem distinct_bead_arrangements_on_bracelet : 
  fact 7 / 14 = 360 := 
by 
  sorry

end NUMINAMATH_GPT_distinct_bead_arrangements_on_bracelet_l2045_204524


namespace NUMINAMATH_GPT_convex_functions_exist_l2045_204568

noncomputable def exponential_function (x : ℝ) : ℝ :=
  4 - 5 * (1 / 2) ^ x

noncomputable def inverse_tangent_function (x : ℝ) : ℝ :=
  (10 / Real.pi) * Real.arctan x - 1

theorem convex_functions_exist :
  ∃ (f1 f2 : ℝ → ℝ),
    (∀ x, 0 < x → f1 x = exponential_function x) ∧
    (∀ x, 0 < x → f2 x = inverse_tangent_function x) ∧
    (∀ x, 0 < x → f1 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x, 0 < x → f2 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f1 x1 + f1 x2 < 2 * f1 ((x1 + x2) / 2)) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f2 x1 + f2 x2 < 2 * f2 ((x1 + x2) / 2)) :=
sorry

end NUMINAMATH_GPT_convex_functions_exist_l2045_204568


namespace NUMINAMATH_GPT_circumference_of_smaller_circle_l2045_204590

theorem circumference_of_smaller_circle (r R : ℝ)
  (h1 : 4 * R^2 = 784) 
  (h2 : R = (7/3) * r) :
  2 * Real.pi * r = 12 * Real.pi := 
by {
  sorry
}

end NUMINAMATH_GPT_circumference_of_smaller_circle_l2045_204590


namespace NUMINAMATH_GPT_geom_sequence_third_term_l2045_204552

theorem geom_sequence_third_term (a : ℕ → ℝ) (r : ℝ) (h : ∀ n, a n = a 1 * r ^ (n - 1)) (h_cond : a 1 * a 5 = a 3) : a 3 = 1 :=
sorry

end NUMINAMATH_GPT_geom_sequence_third_term_l2045_204552


namespace NUMINAMATH_GPT_black_ball_on_second_draw_given_white_ball_on_first_draw_l2045_204582

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 5
def total_balls : ℕ := num_white_balls + num_black_balls

def P_A : ℚ := num_white_balls / total_balls
def P_AB : ℚ := (num_white_balls * num_black_balls) / (total_balls * (total_balls - 1))
def P_B_given_A : ℚ := P_AB / P_A

theorem black_ball_on_second_draw_given_white_ball_on_first_draw : P_B_given_A = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_black_ball_on_second_draw_given_white_ball_on_first_draw_l2045_204582


namespace NUMINAMATH_GPT_length_of_box_l2045_204557

theorem length_of_box (v : ℝ) (w : ℝ) (h : ℝ) (l : ℝ) (conversion_factor : ℝ) (v_gallons : ℝ)
  (h_inch : ℝ) (conversion_inches_feet : ℝ) :
  v_gallons / conversion_factor = v → 
  h_inch / conversion_inches_feet = h →
  v = l * w * h →
  w = 25 →
  v_gallons = 4687.5 →
  conversion_factor = 7.5 →
  h_inch = 6 →
  conversion_inches_feet = 12 →
  l = 50 :=
by
  sorry

end NUMINAMATH_GPT_length_of_box_l2045_204557


namespace NUMINAMATH_GPT_f_inequality_l2045_204561

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem f_inequality (x : ℝ) : f (3^x) ≥ f (2^x) := 
by 
  sorry

end NUMINAMATH_GPT_f_inequality_l2045_204561


namespace NUMINAMATH_GPT_compute_expression_l2045_204589

theorem compute_expression :
  ( ((15 ^ 15) / (15 ^ 10)) ^ 3 * 5 ^ 6 ) / (25 ^ 2) = 3 ^ 15 * 5 ^ 17 :=
by
  -- We'll use sorry here as proof is not required
  sorry

end NUMINAMATH_GPT_compute_expression_l2045_204589


namespace NUMINAMATH_GPT_polynomial_is_quadratic_l2045_204565

theorem polynomial_is_quadratic (m : ℤ) (h : (m - 2 ≠ 0) ∧ (|m| = 2)) : m = -2 :=
by sorry

end NUMINAMATH_GPT_polynomial_is_quadratic_l2045_204565


namespace NUMINAMATH_GPT_sum_of_angles_l2045_204580

theorem sum_of_angles (p q r s t u v w x y : ℝ)
  (H1 : p + r + t + v + x = 360)
  (H2 : q + s + u + w + y = 360) :
  p + q + r + s + t + u + v + w + x + y = 720 := 
by sorry

end NUMINAMATH_GPT_sum_of_angles_l2045_204580


namespace NUMINAMATH_GPT_simplifies_to_minus_18_point_5_l2045_204573

theorem simplifies_to_minus_18_point_5 (x y : ℝ) (h_x : x = 1/2) (h_y : y = -2) :
  ((2 * x + y)^2 - (2 * x - y) * (x + y) - 2 * (x - 2 * y) * (x + 2 * y)) / y = -18.5 :=
by
  -- Let's replace x and y with their values
  -- Expand and simplify the expression
  -- Divide the expression by y
  -- Prove the final result is equal to -18.5
  sorry

end NUMINAMATH_GPT_simplifies_to_minus_18_point_5_l2045_204573


namespace NUMINAMATH_GPT_find_r_l2045_204583

theorem find_r (r : ℝ) (h_curve : r = -2 * r^2 + 5 * r - 2) : r = 1 :=
sorry

end NUMINAMATH_GPT_find_r_l2045_204583


namespace NUMINAMATH_GPT_Ram_has_amount_l2045_204516

theorem Ram_has_amount (R G K : ℕ)
    (h1 : R = 7 * G / 17)
    (h2 : G = 7 * K / 17)
    (h3 : K = 3757) : R = 637 := by
  sorry

end NUMINAMATH_GPT_Ram_has_amount_l2045_204516


namespace NUMINAMATH_GPT_find_n_l2045_204597

noncomputable def cube_probability_solid_color (num_cubes edge_length num_corner num_edge num_face_center num_center : ℕ)
  (corner_prob edge_prob face_center_prob center_prob : ℚ) : ℚ :=
  have total_corner_prob := corner_prob ^ num_corner
  have total_edge_prob := edge_prob ^ num_edge
  have total_face_center_prob := face_center_prob ^ num_face_center
  have total_center_prob := center_prob ^ num_center
  2 * (total_corner_prob * total_edge_prob * total_face_center_prob * total_center_prob)

theorem find_n : ∃ n : ℕ, cube_probability_solid_color 27 3 8 12 6 1
  (1/8) (1/4) (1/2) 1 = (1 / (2 : ℚ) ^ n) ∧ n = 53 := by
  use 53
  simp only [cube_probability_solid_color]
  sorry

end NUMINAMATH_GPT_find_n_l2045_204597


namespace NUMINAMATH_GPT_sum_of_angles_in_figure_l2045_204510

theorem sum_of_angles_in_figure : 
  let triangles := 3
  let angles_in_triangle := 180
  let square_angles := 4 * 90
  (triangles * angles_in_triangle + square_angles) = 900 := by
  sorry

end NUMINAMATH_GPT_sum_of_angles_in_figure_l2045_204510


namespace NUMINAMATH_GPT_profit_ratio_l2045_204584

def praveen_initial_capital : ℝ := 3500
def hari_initial_capital : ℝ := 9000.000000000002
def total_months : ℕ := 12
def months_hari_invested : ℕ := total_months - 5

def effective_capital (initial_capital : ℝ) (months : ℕ) : ℝ :=
  initial_capital * months

theorem profit_ratio :
  effective_capital praveen_initial_capital total_months / effective_capital hari_initial_capital months_hari_invested 
  = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_profit_ratio_l2045_204584


namespace NUMINAMATH_GPT_derivative_at_one_l2045_204517

-- Definition of the function
def f (x : ℝ) : ℝ := x^2

-- Condition
def x₀ : ℝ := 1

-- Problem statement
theorem derivative_at_one : (deriv f x₀) = 2 :=
sorry

end NUMINAMATH_GPT_derivative_at_one_l2045_204517


namespace NUMINAMATH_GPT_solution_pairs_l2045_204581

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end NUMINAMATH_GPT_solution_pairs_l2045_204581


namespace NUMINAMATH_GPT_find_largest_n_l2045_204592

theorem find_largest_n : ∃ n x y z : ℕ, n > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 
  ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6
  ∧ (∀ m x' y' z' : ℕ, m > n → x' > 0 → y' > 0 → z' > 0 
  → m^2 ≠ x'^2 + y'^2 + z'^2 + 2*x'*y' + 2*y'*z' + 2*z'*x' + 3*x' + 3*y' + 3*z' - 6) :=
sorry

end NUMINAMATH_GPT_find_largest_n_l2045_204592


namespace NUMINAMATH_GPT_kiera_fruit_cups_l2045_204572

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def total_cost : ℕ := 17

theorem kiera_fruit_cups : ∃ kiera_fruit_cups : ℕ, muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cups = total_cost - (muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups) :=
by
  let francis_cost := muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  let remaining_cost := total_cost - francis_cost
  let kiera_fruit_cups := remaining_cost / fruit_cup_cost
  exact ⟨kiera_fruit_cups, by sorry⟩

end NUMINAMATH_GPT_kiera_fruit_cups_l2045_204572


namespace NUMINAMATH_GPT_tangent_line_slope_l2045_204578

theorem tangent_line_slope (x₀ y₀ k : ℝ)
    (h_tangent_point : y₀ = x₀ + Real.exp (-x₀))
    (h_tangent_line : y₀ = k * x₀) :
    k = 1 - Real.exp 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_slope_l2045_204578


namespace NUMINAMATH_GPT_depth_of_well_l2045_204531

theorem depth_of_well 
  (t1 t2 : ℝ) 
  (d : ℝ) 
  (h1: t1 + t2 = 8) 
  (h2: d = 32 * t1^2) 
  (h3: t2 = d / 1100) 
  : d = 1348 := 
  sorry

end NUMINAMATH_GPT_depth_of_well_l2045_204531


namespace NUMINAMATH_GPT_ratio_c_d_l2045_204558

theorem ratio_c_d (a b c d : ℝ) (h_eq : ∀ x, a * x^3 + b * x^2 + c * x + d = 0) 
    (h_roots : ∀ r, r = 2 ∨ r = 4 ∨ r = 5 ↔ (a * r^3 + b * r^2 + c * r + d = 0)) :
    c / d = 19 / 20 :=
by
  sorry

end NUMINAMATH_GPT_ratio_c_d_l2045_204558


namespace NUMINAMATH_GPT_soda_price_before_increase_l2045_204522

theorem soda_price_before_increase
  (candy_box_after : ℝ)
  (soda_after : ℝ)
  (candy_box_increase : ℝ)
  (soda_increase : ℝ)
  (new_price_soda : soda_after = 9)
  (new_price_candy_box : candy_box_after = 10)
  (percent_candy_box_increase : candy_box_increase = 0.25)
  (percent_soda_increase : soda_increase = 0.50) :
  ∃ P : ℝ, 1.5 * P = 9 ∧ P = 6 := 
by
  sorry

end NUMINAMATH_GPT_soda_price_before_increase_l2045_204522


namespace NUMINAMATH_GPT_arithmetic_sequence_a1a6_eq_l2045_204555

noncomputable def a_1 : ℤ := 2
noncomputable def d : ℤ := 1
noncomputable def a_n (n : ℕ) : ℤ := a_1 + (n - 1) * d

theorem arithmetic_sequence_a1a6_eq :
  (a_1 * a_n 6) = 14 := by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1a6_eq_l2045_204555


namespace NUMINAMATH_GPT_prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l2045_204518

noncomputable def cartesian_eq_C1 (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 1)^2 = 4

noncomputable def cartesian_eq_C2 (x y : ℝ) : Prop :=
  (4 * x - y - 1 = 0)

noncomputable def min_distance_C1_C2 : ℝ :=
  (10 * Real.sqrt 17 / 17) - 2

theorem prove_cartesian_eq_C1 (x y t : ℝ) (h : x = -2 + 2 * Real.cos t ∧ y = 1 + 2 * Real.sin t) :
  cartesian_eq_C1 x y :=
sorry

theorem prove_cartesian_eq_C2 (ρ θ : ℝ) (h : 4 * ρ * Real.cos θ - ρ * Real.sin θ - 1 = 0) :
  cartesian_eq_C2 (ρ * Real.cos θ) (ρ * Real.sin θ) :=
sorry

theorem prove_min_distance_C1_C2 (h1 : ∀ x y, cartesian_eq_C1 x y) (h2 : ∀ x y, cartesian_eq_C2 x y) :
  ∀ P Q : ℝ × ℝ, (cartesian_eq_C1 P.1 P.2) → (cartesian_eq_C2 Q.1 Q.2) →
  (min_distance_C1_C2 = (Real.sqrt (4^2 + (-1)^2) / Real.sqrt 17) - 2) :=
sorry

end NUMINAMATH_GPT_prove_cartesian_eq_C1_prove_cartesian_eq_C2_prove_min_distance_C1_C2_l2045_204518


namespace NUMINAMATH_GPT_sqrt_factorial_div_l2045_204508

theorem sqrt_factorial_div:
  Real.sqrt (↑(Nat.factorial 9) / 90) = 4 * Real.sqrt 42 := 
by
  -- Steps of the proof
  sorry

end NUMINAMATH_GPT_sqrt_factorial_div_l2045_204508


namespace NUMINAMATH_GPT_range_of_a_l2045_204540

noncomputable def f (x a : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x
noncomputable def f' (x a : ℝ) : ℝ := 1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ 0) ↔ -1 / 3 ≤ a ∧ a ≤ 1 / 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2045_204540


namespace NUMINAMATH_GPT_union_complement_B_A_equals_a_values_l2045_204553

namespace ProofProblem

-- Define the universal set R as real numbers
def R := Set ℝ

-- Define set A and set B as per the conditions
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Complement of B in R
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}

-- Union of complement of B with A
def union_complement_B_A : Set ℝ := complement_B ∪ A

-- The first statement to be proven
theorem union_complement_B_A_equals : 
  union_complement_B_A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by
  sorry

-- Define set C as per the conditions
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- The second statement to be proven
theorem a_values (a : ℝ) (h : C a ⊆ B) : 
  2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_union_complement_B_A_equals_a_values_l2045_204553


namespace NUMINAMATH_GPT_side_length_of_largest_square_l2045_204595

theorem side_length_of_largest_square (S : ℝ) 
  (h1 : 2 * (S / 2)^2 + 2 * (S / 4)^2 = 810) : S = 36 :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_side_length_of_largest_square_l2045_204595


namespace NUMINAMATH_GPT_tiling_problem_l2045_204547

theorem tiling_problem (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ n = 4 * k) 
  ↔ (∃ (L_tile T_tile : ℕ), n * n = 3 * L_tile + 4 * T_tile) :=
by
  sorry

end NUMINAMATH_GPT_tiling_problem_l2045_204547


namespace NUMINAMATH_GPT_find_prime_p_l2045_204525

def f (x : ℕ) : ℕ :=
  (x^4 + 2 * x^3 + 4 * x^2 + 2 * x + 1)^5

theorem find_prime_p : ∃! p, Nat.Prime p ∧ f p = 418195493 := by
  sorry

end NUMINAMATH_GPT_find_prime_p_l2045_204525


namespace NUMINAMATH_GPT_only_1996_is_leap_l2045_204523

def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0))

def is_leap_year_1996 := is_leap_year 1996
def is_leap_year_1998 := is_leap_year 1998
def is_leap_year_2010 := is_leap_year 2010
def is_leap_year_2100 := is_leap_year 2100

theorem only_1996_is_leap : 
  is_leap_year_1996 ∧ ¬is_leap_year_1998 ∧ ¬is_leap_year_2010 ∧ ¬is_leap_year_2100 :=
by 
  -- proof will be added here later
  sorry

end NUMINAMATH_GPT_only_1996_is_leap_l2045_204523


namespace NUMINAMATH_GPT_curve_crosses_itself_at_point_l2045_204577

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  (2 * t₁^2 + 1 = 2 * t₂^2 + 1) ∧ 
  (2 * t₁^3 - 6 * t₁^2 + 8 = 2 * t₂^3 - 6 * t₂^2 + 8) ∧ 
  2 * t₁^2 + 1 = 1 ∧ 2 * t₁^3 - 6 * t₁^2 + 8 = 8 :=
by
  sorry

end NUMINAMATH_GPT_curve_crosses_itself_at_point_l2045_204577


namespace NUMINAMATH_GPT_carbonate_weight_l2045_204567

namespace MolecularWeight

def molecular_weight_Al2_CO3_3 : ℝ := 234
def molecular_weight_Al : ℝ := 26.98
def num_Al_atoms : ℕ := 2

theorem carbonate_weight :
  molecular_weight_Al2_CO3_3 - (num_Al_atoms * molecular_weight_Al) = 180.04 :=
sorry

end MolecularWeight

end NUMINAMATH_GPT_carbonate_weight_l2045_204567


namespace NUMINAMATH_GPT_trig_identity_l2045_204512

theorem trig_identity : 4 * Real.sin (15 * Real.pi / 180) * Real.sin (105 * Real.pi / 180) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l2045_204512


namespace NUMINAMATH_GPT_tricycle_wheels_l2045_204579

theorem tricycle_wheels (T : ℕ) 
  (h1 : 3 * 2 = 6) 
  (h2 : 7 * 1 = 7) 
  (h3 : 6 + 7 + 4 * T = 25) : T = 3 :=
sorry

end NUMINAMATH_GPT_tricycle_wheels_l2045_204579


namespace NUMINAMATH_GPT_solve_fra_eq_l2045_204556

theorem solve_fra_eq : ∀ x : ℝ, (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 → x = 3 :=
by 
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_solve_fra_eq_l2045_204556


namespace NUMINAMATH_GPT_num_factors_34848_l2045_204585

/-- Define the number 34848 and its prime factorization -/
def n : ℕ := 34848
def p_factors : List (ℕ × ℕ) := [(2, 5), (3, 2), (11, 2)]

/-- Helper function to calculate the number of divisors from prime factors -/
def num_divisors (factors : List (ℕ × ℕ)) : ℕ := 
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.2 + 1)) 1

/-- Formal statement of the problem -/
theorem num_factors_34848 : num_divisors p_factors = 54 :=
by
  -- Proof that 34848 has the prime factorization 3^2 * 2^5 * 11^2 
  -- and that the number of factors is 54 would go here.
  sorry

end NUMINAMATH_GPT_num_factors_34848_l2045_204585


namespace NUMINAMATH_GPT_min_speed_A_l2045_204520

theorem min_speed_A (V_B V_C V_A : ℕ) (d_AB d_AC wind extra_speed : ℕ) :
  V_B = 50 →
  V_C = 70 →
  d_AB = 40 →
  d_AC = 280 →
  wind = 5 →
  V_A > ((d_AB * (V_A + wind + extra_speed)) / (d_AC - d_AB) - wind) :=
sorry

end NUMINAMATH_GPT_min_speed_A_l2045_204520


namespace NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l2045_204504

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ m n : ℤ, m^3 + 8 * m^2 + 17 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1 :=
sorry

end NUMINAMATH_GPT_no_integer_pairs_satisfy_equation_l2045_204504


namespace NUMINAMATH_GPT_mrs_generous_jelly_beans_l2045_204532

-- Define necessary terms and state the problem
def total_children (x : ℤ) : ℤ := x + (x + 3)

theorem mrs_generous_jelly_beans :
  ∃ x : ℤ, x^2 + (x + 3)^2 = 490 ∧ total_children x = 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_mrs_generous_jelly_beans_l2045_204532


namespace NUMINAMATH_GPT_find_a1_l2045_204528

noncomputable def a (n : ℕ) : ℤ := sorry -- the definition of sequence a_n is not computable without initial terms
noncomputable def S (n : ℕ) : ℤ := sorry -- similarly, the definition of S_n without initial terms isn't given

axiom recurrence_relation (n : ℕ) (h : n ≥ 3): 
  a (n) = a (n - 1) - a (n - 2)

axiom S9 : S 9 = 6
axiom S10 : S 10 = 5

theorem find_a1 : a 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l2045_204528


namespace NUMINAMATH_GPT_proposition_relationship_l2045_204545
-- Import library

-- Statement of the problem
theorem proposition_relationship (p q : Prop) (hpq : p ∨ q) (hnp : ¬p) : ¬p ∧ q :=
  by
  sorry

end NUMINAMATH_GPT_proposition_relationship_l2045_204545


namespace NUMINAMATH_GPT_largest_angle_in_pentagon_l2045_204560

-- Define the angles and sum condition
variables (x : ℝ) {P Q R S T : ℝ}

-- Conditions
def angle_P : P = 90 := sorry
def angle_Q : Q = 70 := sorry
def angle_R : R = x := sorry
def angle_S : S = x := sorry
def angle_T : T = 2*x + 20 := sorry
def sum_of_angles : P + Q + R + S + T = 540 := sorry

-- Prove the largest angle
theorem largest_angle_in_pentagon (hP : P = 90) (hQ : Q = 70)
    (hR : R = x) (hS : S = x) (hT : T = 2*x + 20) 
    (h_sum : P + Q + R + S + T = 540) : T = 200 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_pentagon_l2045_204560


namespace NUMINAMATH_GPT_find_x_l2045_204507

def vec := (ℝ × ℝ)

def a : vec := (1, 1)
def b (x : ℝ) : vec := (3, x)

def add_vec (v1 v2 : vec) : vec := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : vec) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x (x : ℝ) (h : dot_product a (add_vec a (b x)) = 0) : x = -5 :=
by
  -- Proof steps (irrelevant for now)
  sorry

end NUMINAMATH_GPT_find_x_l2045_204507


namespace NUMINAMATH_GPT_gum_distribution_l2045_204530

theorem gum_distribution : 
  ∀ (John Cole Aubrey: ℕ), 
    John = 54 → 
    Cole = 45 → 
    Aubrey = 0 → 
    ((John + Cole + Aubrey) / 3) = 33 := 
by
  intros John Cole Aubrey hJohn hCole hAubrey
  sorry

end NUMINAMATH_GPT_gum_distribution_l2045_204530


namespace NUMINAMATH_GPT_region_area_l2045_204596

theorem region_area (x y : ℝ) : (x^2 + y^2 + 6*x - 4*y - 11 = 0) → (∃ (A : ℝ), A = 24 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_region_area_l2045_204596


namespace NUMINAMATH_GPT_sum_central_square_l2045_204511

noncomputable def table_sum : ℕ := 10200
noncomputable def a : ℕ := 1200
noncomputable def central_sum : ℕ := 720

theorem sum_central_square :
  ∃ (a : ℕ), table_sum = a * (1 + (1 / 3) + (1 / 9) + (1 / 27)) * (1 + (1 / 4) + (1 / 16) + (1 / 64)) ∧ 
              central_sum = (a / 3) + (a / 12) + (a / 9) + (a / 36) :=
by
  sorry

end NUMINAMATH_GPT_sum_central_square_l2045_204511
