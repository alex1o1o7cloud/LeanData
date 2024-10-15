import Mathlib

namespace NUMINAMATH_GPT_flour_qualification_l2338_233844

def acceptable_weight_range := {w : ℝ | 24.75 ≤ w ∧ w ≤ 25.25}

theorem flour_qualification :
  (24.80 ∈ acceptable_weight_range) ∧ 
  (24.70 ∉ acceptable_weight_range) ∧ 
  (25.30 ∉ acceptable_weight_range) ∧ 
  (25.51 ∉ acceptable_weight_range) :=
by 
  -- The proof would go here, but we are adding sorry to skip it.
  sorry

end NUMINAMATH_GPT_flour_qualification_l2338_233844


namespace NUMINAMATH_GPT_motion_of_Q_is_clockwise_with_2ω_l2338_233802

variables {ω t : ℝ} {P Q : ℝ × ℝ}

def moving_counterclockwise (P : ℝ × ℝ) (ω t : ℝ) : Prop :=
  P = (Real.cos (ω * t), Real.sin (ω * t))

def motion_of_Q (x y : ℝ): ℝ × ℝ :=
  (-2 * x * y, y^2 - x^2)

def is_on_unit_circle (Q : ℝ × ℝ) : Prop :=
  Q.fst ^ 2 + Q.snd ^ 2 = 1

theorem motion_of_Q_is_clockwise_with_2ω 
  (P : ℝ × ℝ) (ω t : ℝ) (x y : ℝ) :
  moving_counterclockwise P ω t →
  P = (x, y) →
  is_on_unit_circle P →
  is_on_unit_circle (motion_of_Q x y) ∧
  Q = (x, y) →
  Q.fst = Real.cos (2 * ω * t + 3 * Real.pi / 2) ∧ 
  Q.snd = Real.sin (2 * ω * t + 3 * Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_motion_of_Q_is_clockwise_with_2ω_l2338_233802


namespace NUMINAMATH_GPT_minimum_inequality_l2338_233841

theorem minimum_inequality 
  (x_1 x_2 x_3 x_4 : ℝ) 
  (h1 : x_1 > 0) 
  (h2 : x_2 > 0) 
  (h3 : x_3 > 0) 
  (h4 : x_4 > 0) 
  (h_sum : x_1^2 + x_2^2 + x_3^2 + x_4^2 = 4) :
  (x_1 / (1 - x_1^2) + x_2 / (1 - x_2^2) + x_3 / (1 - x_3^2) + x_4 / (1 - x_4^2)) ≥ 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_inequality_l2338_233841


namespace NUMINAMATH_GPT_contradiction_to_at_least_one_not_greater_than_60_l2338_233899

-- Define a condition for the interior angles of a triangle being > 60
def all_angles_greater_than_60 (α β γ : ℝ) : Prop :=
  α > 60 ∧ β > 60 ∧ γ > 60

-- Define the negation of the proposition "At least one of the interior angles is not greater than 60"
def at_least_one_not_greater_than_60 (α β γ : ℝ) : Prop :=
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60

-- The mathematically equivalent proof problem
theorem contradiction_to_at_least_one_not_greater_than_60 (α β γ : ℝ) :
  ¬ at_least_one_not_greater_than_60 α β γ ↔ all_angles_greater_than_60 α β γ := by
  sorry

end NUMINAMATH_GPT_contradiction_to_at_least_one_not_greater_than_60_l2338_233899


namespace NUMINAMATH_GPT_expected_value_is_correct_l2338_233839

noncomputable def expected_value_max_two_rolls : ℝ :=
  let p_max_1 := (1/6) * (1/6)
  let p_max_2 := (2/6) * (2/6) - (1/6) * (1/6)
  let p_max_3 := (3/6) * (3/6) - (2/6) * (2/6)
  let p_max_4 := (4/6) * (4/6) - (3/6) * (3/6)
  let p_max_5 := (5/6) * (5/6) - (4/6) * (4/6)
  let p_max_6 := 1 - (5/6) * (5/6)
  1 * p_max_1 + 2 * p_max_2 + 3 * p_max_3 + 4 * p_max_4 + 5 * p_max_5 + 6 * p_max_6

theorem expected_value_is_correct :
  expected_value_max_two_rolls = 4.5 :=
sorry

end NUMINAMATH_GPT_expected_value_is_correct_l2338_233839


namespace NUMINAMATH_GPT_dealer_cannot_prevent_l2338_233817

theorem dealer_cannot_prevent (m n : ℕ) (h : m < 3 * n ∧ n < 3 * m) :
  ∃ (a b : ℕ), (a = 3 * b ∨ b = 3 * a) ∨ (a = 0 ∧ b = 0):=
sorry

end NUMINAMATH_GPT_dealer_cannot_prevent_l2338_233817


namespace NUMINAMATH_GPT_rectangle_side_deficit_l2338_233832

theorem rectangle_side_deficit (L W : ℝ) (p : ℝ)
  (h1 : 1.05 * L * (1 - p) * W - L * W = 0.8 / 100 * L * W)
  (h2 : 0 < L) (h3 : 0 < W) : p = 0.04 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_side_deficit_l2338_233832


namespace NUMINAMATH_GPT_greatest_candies_to_office_l2338_233808

-- Problem statement: Prove that the greatest possible number of candies given to the office is 7 when distributing candies among 8 students.

theorem greatest_candies_to_office (n : ℕ) : 
  ∃ k : ℕ, k = n % 8 ∧ k ≤ 7 ∧ k = 7 :=
by
  sorry

end NUMINAMATH_GPT_greatest_candies_to_office_l2338_233808


namespace NUMINAMATH_GPT_option_a_option_b_option_c_option_d_l2338_233880

open Real

theorem option_a (x : ℝ) (h1 : 0 < x) (h2 : x < π) : x > sin x :=
sorry

theorem option_b (x : ℝ) (h : 0 < x) : ¬ (1 - (1 / x) > log x) :=
sorry

theorem option_c (x : ℝ) : (x + 1) * exp x >= -1 / (exp 2) :=
sorry

theorem option_d : ¬ (∀ x : ℝ, x^2 > - (1 / x)) :=
sorry

end NUMINAMATH_GPT_option_a_option_b_option_c_option_d_l2338_233880


namespace NUMINAMATH_GPT_taxi_ride_cost_l2338_233872

theorem taxi_ride_cost (base_fare : ℝ) (rate_per_mile : ℝ) (additional_charge : ℝ) (distance : ℕ) (cost : ℝ) :
  base_fare = 2 ∧ rate_per_mile = 0.30 ∧ additional_charge = 5 ∧ distance = 12 ∧ 
  cost = base_fare + (rate_per_mile * distance) + additional_charge → cost = 10.60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_taxi_ride_cost_l2338_233872


namespace NUMINAMATH_GPT_final_amoeba_is_blue_l2338_233865

-- We define the initial counts of each type of amoeba
def initial_red : ℕ := 26
def initial_blue : ℕ := 31
def initial_yellow : ℕ := 16

-- We define the final count of amoebas
def final_amoebas : ℕ := 1

-- The type of the final amoeba (we're proving it's 'blue')
inductive AmoebaColor
| Red
| Blue
| Yellow

-- Given initial counts, we aim to prove the final amoeba is blue
theorem final_amoeba_is_blue :
  initial_red = 26 ∧ initial_blue = 31 ∧ initial_yellow = 16 ∧ final_amoebas = 1 → 
  ∃ c : AmoebaColor, c = AmoebaColor.Blue :=
by sorry

end NUMINAMATH_GPT_final_amoeba_is_blue_l2338_233865


namespace NUMINAMATH_GPT_largest_multiple_of_9_less_than_110_l2338_233893

theorem largest_multiple_of_9_less_than_110 : ∃ x, (x < 110 ∧ x % 9 = 0 ∧ ∀ y, (y < 110 ∧ y % 9 = 0) → y ≤ x) ∧ x = 108 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_9_less_than_110_l2338_233893


namespace NUMINAMATH_GPT_balanced_scale_l2338_233815

def children's_book_weight : ℝ := 1.1

def weight1 : ℝ := 0.5
def weight2 : ℝ := 0.3
def weight3 : ℝ := 0.3

theorem balanced_scale :
  (weight1 + weight2 + weight3) = children's_book_weight :=
by
  sorry

end NUMINAMATH_GPT_balanced_scale_l2338_233815


namespace NUMINAMATH_GPT_smallest_constant_c_l2338_233891

def satisfies_conditions (f : ℝ → ℝ) :=
  ∀ ⦃x : ℝ⦄, (0 ≤ x ∧ x ≤ 1) → (f x ≥ 0 ∧ (x = 1 → f 1 = 1) ∧
  (∀ y, 0 ≤ y → y ≤ 1 → x + y ≤ 1 → f x + f y ≤ f (x + y)))

theorem smallest_constant_c :
  ∀ {f : ℝ → ℝ},
  satisfies_conditions f →
  ∃ c : ℝ, (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ c * x) ∧
  (∀ c', c' < 2 → ∃ x, 0 ≤ x → x ≤ 1 ∧ f x > c' * x) :=
by sorry

end NUMINAMATH_GPT_smallest_constant_c_l2338_233891


namespace NUMINAMATH_GPT_total_waiting_days_l2338_233894

-- Definitions based on the conditions
def wait_for_first_appointment : ℕ := 4
def wait_for_second_appointment : ℕ := 20
def wait_for_effectiveness : ℕ := 2 * 7  -- 2 weeks converted to days

-- The main theorem statement
theorem total_waiting_days : wait_for_first_appointment + wait_for_second_appointment + wait_for_effectiveness = 38 :=
by
  sorry

end NUMINAMATH_GPT_total_waiting_days_l2338_233894


namespace NUMINAMATH_GPT_sum_of_possible_values_l2338_233818

theorem sum_of_possible_values {x : ℝ} :
  (3 * (x - 3)^2 = (x - 2) * (x + 5)) →
  (∃ (x1 x2 : ℝ), x1 + x2 = 10.5) :=
by sorry

end NUMINAMATH_GPT_sum_of_possible_values_l2338_233818


namespace NUMINAMATH_GPT_ratio_equality_l2338_233879

theorem ratio_equality (x y u v p q : ℝ) (h : (x / y) * (u / v) * (p / q) = 1) :
  (x / y) * (u / v) * (p / q) = 1 := 
by sorry

end NUMINAMATH_GPT_ratio_equality_l2338_233879


namespace NUMINAMATH_GPT_solve_quadratic_eq_l2338_233889

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 → x = 2 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l2338_233889


namespace NUMINAMATH_GPT_english_students_23_l2338_233886

def survey_students_total : Nat := 35
def students_in_all_three : Nat := 2
def solely_english_three_times_than_french (x y : Nat) : Prop := y = 3 * x
def english_but_not_french_or_spanish (x y : Nat) : Prop := y + students_in_all_three = 35 ∧ y - students_in_all_three = 23

theorem english_students_23 :
  ∃ (x y : Nat), solely_english_three_times_than_french x y ∧ english_but_not_french_or_spanish x y :=
by
  sorry

end NUMINAMATH_GPT_english_students_23_l2338_233886


namespace NUMINAMATH_GPT_trains_clear_time_l2338_233823

-- Definitions based on conditions
def length_train1 : ℕ := 160
def length_train2 : ℕ := 280
def speed_train1_kmph : ℕ := 42
def speed_train2_kmph : ℕ := 30

-- Conversion factor from km/h to m/s
def kmph_to_mps (s : ℕ) : ℕ := s * 1000 / 3600

-- Computation of relative speed in m/s
def relative_speed_mps : ℕ := kmph_to_mps (speed_train1_kmph + speed_train2_kmph)

-- Total distance to be covered for the trains to clear each other
def total_distance : ℕ := length_train1 + length_train2

-- Time taken for the trains to clear each other
def time_to_clear_each_other : ℕ := total_distance / relative_speed_mps

-- Theorem stating that time taken is 22 seconds
theorem trains_clear_time : time_to_clear_each_other = 22 := by
  sorry

end NUMINAMATH_GPT_trains_clear_time_l2338_233823


namespace NUMINAMATH_GPT_positive_integer_divisibility_l2338_233859

theorem positive_integer_divisibility :
  ∀ n : ℕ, 0 < n → (5^(n-1) + 3^(n-1) ∣ 5^n + 3^n) → n = 1 :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_divisibility_l2338_233859


namespace NUMINAMATH_GPT_correct_operation_l2338_233890

theorem correct_operation : 
  (a^2 + a^2 = 2 * a^2) = false ∧ 
  ((-3 * a * b^2)^2 = -6 * a^2 * b^4) = false ∧ 
  (a^6 / (-a)^2 = a^4) = true ∧ 
  ((a - b)^2 = a^2 - b^2) = false :=
sorry

end NUMINAMATH_GPT_correct_operation_l2338_233890


namespace NUMINAMATH_GPT_angle_equiv_470_110_l2338_233876

theorem angle_equiv_470_110 : ∃ (k : ℤ), 470 = k * 360 + 110 :=
by
  use 1
  exact rfl

end NUMINAMATH_GPT_angle_equiv_470_110_l2338_233876


namespace NUMINAMATH_GPT_curve_is_circle_l2338_233877

theorem curve_is_circle (r θ : ℝ) (h : r = 3 * Real.sin θ) : 
  ∃ c : ℝ × ℝ, c = (0, 3 / 2) ∧ ∀ p : ℝ × ℝ, ∃ R : ℝ, R = 3 / 2 ∧ 
  (p.1 - c.1)^2 + (p.2 - c.2)^2 = R^2 :=
sorry

end NUMINAMATH_GPT_curve_is_circle_l2338_233877


namespace NUMINAMATH_GPT_find_y_from_condition_l2338_233896

variable (y : ℝ) (h : (3 * y) / 7 = 15)

theorem find_y_from_condition : y = 35 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_from_condition_l2338_233896


namespace NUMINAMATH_GPT_solution_1_solution_2_l2338_233821

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - (a + 1) * x + Real.log x

def critical_point_condition (a x : ℝ) : Prop :=
  (x = 1 / 4) → deriv (f a) x = 0

def pseudo_symmetry_point_condition (a : ℝ) (x0 : ℝ) : Prop :=
  let f' := fun x => 2 * x^2 - 5 * x + Real.log x
  let g := fun x => (4 * x0^2 - 5 * x0 + 1) / x0 * (x - x0) + 2 * x0^2 - 5 * x0 + Real.log x0
  ∀ x : ℝ, 
    (0 < x ∧ x < x0) → (f' x - g x < 0) ∧ 
    (x > x0) → (f' x - g x > 0)

theorem solution_1 (a : ℝ) (h1 : a > 0) (h2 : critical_point_condition a (1/4)) :
  a = 4 := 
sorry

theorem solution_2 (x0 : ℝ) (h1 : x0 = 1/2) :
  pseudo_symmetry_point_condition 4 x0 :=
sorry


end NUMINAMATH_GPT_solution_1_solution_2_l2338_233821


namespace NUMINAMATH_GPT_silvia_shorter_route_l2338_233878

theorem silvia_shorter_route :
  let jerry_distance := 3 + 4
  let silvia_distance := Real.sqrt (3^2 + 4^2)
  let percentage_reduction := ((jerry_distance - silvia_distance) / jerry_distance) * 100
  (28.5 ≤ percentage_reduction ∧ percentage_reduction < 30.5) →
  percentage_reduction = 30 := by
  intro h
  sorry

end NUMINAMATH_GPT_silvia_shorter_route_l2338_233878


namespace NUMINAMATH_GPT_problem_correctness_l2338_233888

theorem problem_correctness
  (correlation_A : ℝ)
  (correlation_B : ℝ)
  (chi_squared : ℝ)
  (P_chi_squared_5_024 : ℝ)
  (P_chi_squared_6_635 : ℝ)
  (P_X_leq_2 : ℝ)
  (P_X_lt_0 : ℝ) :
  correlation_A = 0.66 →
  correlation_B = -0.85 →
  chi_squared = 6.352 →
  P_chi_squared_5_024 = 0.025 →
  P_chi_squared_6_635 = 0.01 →
  P_X_leq_2 = 0.68 →
  P_X_lt_0 = 0.32 →
  (abs correlation_B > abs correlation_A) ∧
  (1 - P_chi_squared_5_024 < 0.99) ∧
  (P_X_lt_0 = 1 - P_X_leq_2) ∧
  (false) := sorry

end NUMINAMATH_GPT_problem_correctness_l2338_233888


namespace NUMINAMATH_GPT_certain_number_is_four_l2338_233828

theorem certain_number_is_four (k : ℕ) (h₁ : k = 16) : 64 / k = 4 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_four_l2338_233828


namespace NUMINAMATH_GPT_job_completion_time_l2338_233834

theorem job_completion_time (initial_men : ℕ) (initial_days : ℕ) (extra_men : ℕ) (interval_days : ℕ) (total_days : ℕ) : 
  initial_men = 20 → 
  initial_days = 15 → 
  extra_men = 10 → 
  interval_days = 5 → 
  total_days = 12 → 
  ∀ n, (20 * 5 + (20 + 10) * 5 + (20 + 10 + 10) * n.succ = 300 → n + 10 + n.succ = 12) :=
by
  intro h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_job_completion_time_l2338_233834


namespace NUMINAMATH_GPT_diamond_op_example_l2338_233868

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y

theorem diamond_op_example : diamond_op 2 7 = 41 :=
by {
    -- proof goes here
    sorry
}

end NUMINAMATH_GPT_diamond_op_example_l2338_233868


namespace NUMINAMATH_GPT_jacob_fifth_test_score_l2338_233838

theorem jacob_fifth_test_score (s1 s2 s3 s4 s5 : ℕ) :
  s1 = 85 ∧ s2 = 79 ∧ s3 = 92 ∧ s4 = 84 ∧ ((s1 + s2 + s3 + s4 + s5) / 5 = 85) →
  s5 = 85 :=
sorry

end NUMINAMATH_GPT_jacob_fifth_test_score_l2338_233838


namespace NUMINAMATH_GPT_sum_S17_l2338_233875

-- Definitions of the required arithmetic sequence elements.
variable (a1 d : ℤ)

-- Definition of the arithmetic sequence
def aₙ (n : ℤ) : ℤ := a1 + (n - 1) * d
def Sₙ (n : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Theorem for the problem statement
theorem sum_S17 : (aₙ a1 d 7 + aₙ a1 d 5) = (3 + aₙ a1 d 5) → (a1 + 8 * d = 3) → Sₙ a1 d 17 = 51 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sum_S17_l2338_233875


namespace NUMINAMATH_GPT_minimum_value_of_f_l2338_233840

def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 6 * x + 1

theorem minimum_value_of_f :
  exists (x : ℝ), x = 1 + 1 / Real.sqrt 3 ∧ ∀ (y : ℝ), f (1 + 1 / Real.sqrt 3) ≤ f y := sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2338_233840


namespace NUMINAMATH_GPT_side_length_of_square_l2338_233807

-- Mathematical definitions and conditions
def square_area (side : ℕ) : ℕ := side * side

theorem side_length_of_square {s : ℕ} (h : square_area s = 289) : s = 17 :=
sorry

end NUMINAMATH_GPT_side_length_of_square_l2338_233807


namespace NUMINAMATH_GPT_distance_from_point_to_line_l2338_233871

open Real

noncomputable def point_to_line_distance (a b c x0 y0 : ℝ) : ℝ :=
  abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2)

theorem distance_from_point_to_line (a b c x0 y0 : ℝ) :
  point_to_line_distance a b c x0 y0 = abs (a * x0 + b * y0 + c) / sqrt (a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_point_to_line_l2338_233871


namespace NUMINAMATH_GPT_integer_solutions_of_equation_l2338_233861

theorem integer_solutions_of_equation :
  ∀ (x y : ℤ), (x^4 + y^4 = 3 * x^3 * y) → (x = 0 ∧ y = 0) := by
  intros x y h
  sorry

end NUMINAMATH_GPT_integer_solutions_of_equation_l2338_233861


namespace NUMINAMATH_GPT_wechat_balance_l2338_233830

def transaction1 : ℤ := 48
def transaction2 : ℤ := -30
def transaction3 : ℤ := -50

theorem wechat_balance :
  transaction1 + transaction2 + transaction3 = -32 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_wechat_balance_l2338_233830


namespace NUMINAMATH_GPT_count_multiples_of_30_between_two_multiples_l2338_233824

theorem count_multiples_of_30_between_two_multiples : 
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  count = 871 :=
by
  let lower := 900
  let upper := 27000
  let multiple := 30
  let count := (upper / multiple) - (lower / multiple) + 1
  sorry

end NUMINAMATH_GPT_count_multiples_of_30_between_two_multiples_l2338_233824


namespace NUMINAMATH_GPT_coloring_impossible_l2338_233857

theorem coloring_impossible :
  ¬ ∃ (color : ℕ → Prop), (∀ n m : ℕ, (m = n + 5 → color n ≠ color m) ∧ (m = 2 * n → color n ≠ color m)) :=
sorry

end NUMINAMATH_GPT_coloring_impossible_l2338_233857


namespace NUMINAMATH_GPT_three_digit_reverse_sum_to_1777_l2338_233856

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_reverse_sum_to_1777_l2338_233856


namespace NUMINAMATH_GPT_initial_calculated_average_l2338_233829

theorem initial_calculated_average (S : ℕ) (initial_average correct_average : ℕ) (num_wrongly_read correctly_read wrong_value correct_value : ℕ)
    (h1 : num_wrongly_read = 36) 
    (h2 : correctly_read = 26) 
    (h3 : correct_value = 6)
    (h4 : S = 10 * correct_value) :
    initial_average = (S - (num_wrongly_read - correctly_read)) / 10 → initial_average = 5 :=
sorry

end NUMINAMATH_GPT_initial_calculated_average_l2338_233829


namespace NUMINAMATH_GPT_find_x_for_g_equal_20_l2338_233816

theorem find_x_for_g_equal_20 (g f : ℝ → ℝ) (h₁ : ∀ x, g x = 4 * (f⁻¹ x))
    (h₂ : ∀ x, f x = 30 / (x + 5)) :
    ∃ x, g x = 20 ∧ x = 3 := by
  sorry

end NUMINAMATH_GPT_find_x_for_g_equal_20_l2338_233816


namespace NUMINAMATH_GPT_solution_for_system_l2338_233853
open Real

noncomputable def solve_system (a b x y : ℝ) : Prop :=
  (a * x + b * y = 7 ∧ b * x + a * y = 8)

noncomputable def solve_linear (a b m n : ℝ) : Prop :=
  (a * (m + n) + b * (m - n) = 7 ∧ b * (m + n) + a * (m - n) = 8)

theorem solution_for_system (a b : ℝ) : solve_system a b 2 3 → solve_linear a b (5/2) (-1/2) :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_for_system_l2338_233853


namespace NUMINAMATH_GPT_find_b9_l2338_233850

theorem find_b9 {b : ℕ → ℕ} 
  (h1 : ∀ n, b (n + 2) = b (n + 1) + b n)
  (h2 : b 8 = 100) :
  b 9 = 194 :=
sorry

end NUMINAMATH_GPT_find_b9_l2338_233850


namespace NUMINAMATH_GPT_longest_side_of_region_l2338_233812

theorem longest_side_of_region :
  (∃ (x y : ℝ), x + y ≤ 5 ∧ 3 * x + y ≥ 3 ∧ x ≥ 1 ∧ y ≥ 1) →
  (∃ (l : ℝ), l = Real.sqrt 130 / 3 ∧ 
    (l = Real.sqrt ((1 - 1)^2 + (4 - 1)^2) ∨ 
     l = Real.sqrt (((1 + 4 / 3) - 1)^2 + (1 - 1)^2) ∨ 
     l = Real.sqrt ((1 - (1 + 4 / 3))^2 + (1 - 1)^2))) :=
by
  sorry

end NUMINAMATH_GPT_longest_side_of_region_l2338_233812


namespace NUMINAMATH_GPT_isosceles_triangle_sides_l2338_233810

theorem isosceles_triangle_sides (a b c : ℕ) (h₁ : a + b + c = 10) (h₂ : (a = b ∨ b = c ∨ a = c)) 
  (h₃ : a + b > c) (h₄ : a + c > b) (h₅ : b + c > a) : 
  (a = 3 ∧ b = 3 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 4) := 
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_sides_l2338_233810


namespace NUMINAMATH_GPT_tangent_line_equations_l2338_233870

theorem tangent_line_equations (k b : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x, l x = k * x + b) ∧
    (∃ x₁, x₁^2 = k * x₁ + b) ∧ -- Tangency condition with C1: y = x²
    (∃ x₂, -(x₂ - 2)^2 = k * x₂ + b)) -- Tangency condition with C2: y = -(x-2)²
  → ((k = 0 ∧ b = 0) ∨ (k = 4 ∧ b = -4)) := sorry

end NUMINAMATH_GPT_tangent_line_equations_l2338_233870


namespace NUMINAMATH_GPT_olympiad_scores_l2338_233897

theorem olympiad_scores (a : Fin 20 → ℕ) 
  (h_distinct : ∀ i j : Fin 20, i < j → a i < a j)
  (h_condition : ∀ i j k : Fin 20, i ≠ j ∧ i ≠ k ∧ j ≠ k → a i < a j + a k) : 
  ∀ i : Fin 20, a i > 18 :=
by
  sorry

end NUMINAMATH_GPT_olympiad_scores_l2338_233897


namespace NUMINAMATH_GPT_time_between_peanuts_l2338_233874

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := 4
def flight_time_hours : ℕ := 2

theorem time_between_peanuts (peanuts_per_bag number_of_bags flight_time_hours : ℕ) (h1 : peanuts_per_bag = 30) (h2 : number_of_bags = 4) (h3 : flight_time_hours = 2) :
  (flight_time_hours * 60) / (peanuts_per_bag * number_of_bags) = 1 := by
  sorry

end NUMINAMATH_GPT_time_between_peanuts_l2338_233874


namespace NUMINAMATH_GPT_lindsey_squat_weight_l2338_233837

theorem lindsey_squat_weight :
  let bandA := 7
  let bandB := 5
  let bandC := 3
  let leg_weight := 10
  let dumbbell := 15
  let total_weight := (2 * bandA) + (2 * bandB) + (2 * bandC) + (2 * leg_weight) + dumbbell
  total_weight = 65 :=
by
  sorry

end NUMINAMATH_GPT_lindsey_squat_weight_l2338_233837


namespace NUMINAMATH_GPT_find_x_l2338_233822

theorem find_x (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l2338_233822


namespace NUMINAMATH_GPT_frank_money_left_l2338_233887

theorem frank_money_left (initial_money : ℝ) (spent_groceries : ℝ) (spent_magazine : ℝ) :
  initial_money = 600 →
  spent_groceries = (1/5) * initial_money →
  spent_magazine = (1/4) * (initial_money - spent_groceries) →
  initial_money - spent_groceries - spent_magazine = 360 := 
by
  intro h1 h2 h3
  rw [h1] at *
  rw [h2] at *
  rw [h3] at *
  sorry

end NUMINAMATH_GPT_frank_money_left_l2338_233887


namespace NUMINAMATH_GPT_total_income_l2338_233814

theorem total_income (I : ℝ) (h1 : 0.10 * I * 2 + 0.20 * I + 0.06 * (I - 0.40 * I) = 0.46 * I) (h2 : 0.54 * I = 500) : I = 500 / 0.54 :=
by
  sorry

end NUMINAMATH_GPT_total_income_l2338_233814


namespace NUMINAMATH_GPT_value_of_B_l2338_233869

theorem value_of_B (B : ℝ) : 3 * B ^ 2 + 3 * B + 2 = 29 ↔ (B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2) :=
by sorry

end NUMINAMATH_GPT_value_of_B_l2338_233869


namespace NUMINAMATH_GPT_charlie_book_pages_l2338_233833

theorem charlie_book_pages :
  (2 * 40) + (4 * 45) + 20 = 280 :=
by 
  sorry

end NUMINAMATH_GPT_charlie_book_pages_l2338_233833


namespace NUMINAMATH_GPT_tangent_slope_at_point_552_32_l2338_233866

noncomputable def slope_of_tangent_at_point (cx cy px py : ℚ) : ℚ :=
if py - cy = 0 then 
  0 
else 
  (px - cx) / (py - cy)

theorem tangent_slope_at_point_552_32 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 :=
by
  -- Conditions from problem
  have h1 : slope_of_tangent_at_point 3 2 5 5 = -2 / 3 := 
    sorry
  
  exact h1

end NUMINAMATH_GPT_tangent_slope_at_point_552_32_l2338_233866


namespace NUMINAMATH_GPT_triangle_side_value_l2338_233847

theorem triangle_side_value
  (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 1)
  (h2 : b = 4)
  (h3 : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h4 : a^2 + b^2 - 2 * a * b * Real.cos C = c^2) :
  c = Real.sqrt 13 :=
sorry

end NUMINAMATH_GPT_triangle_side_value_l2338_233847


namespace NUMINAMATH_GPT_cobbler_mends_3_pairs_per_hour_l2338_233842

def cobbler_hours_per_day_mon_thu := 8
def cobbler_hours_friday := 11 - 8
def cobbler_total_hours_week := 4 * cobbler_hours_per_day_mon_thu + cobbler_hours_friday
def cobbler_pairs_per_week := 105
def cobbler_pairs_per_hour := cobbler_pairs_per_week / cobbler_total_hours_week

theorem cobbler_mends_3_pairs_per_hour : cobbler_pairs_per_hour = 3 := 
by 
  -- Add the steps if necessary but in this scenario, we are skipping proof details
  sorry

end NUMINAMATH_GPT_cobbler_mends_3_pairs_per_hour_l2338_233842


namespace NUMINAMATH_GPT_factor_problem_l2338_233854

theorem factor_problem 
  (a b : ℕ) (h1 : a > b)
  (h2 : (∀ x, x^2 - 16 * x + 64 = (x - a) * (x - b))) 
  : 3 * b - a = 16 := by
  sorry

end NUMINAMATH_GPT_factor_problem_l2338_233854


namespace NUMINAMATH_GPT_total_cost_is_13_l2338_233800

-- Definition of pencil cost
def pencil_cost : ℕ := 2

-- Definition of pen cost based on pencil cost
def pen_cost : ℕ := pencil_cost + 9

-- The total cost of both items
def total_cost := pencil_cost + pen_cost

theorem total_cost_is_13 : total_cost = 13 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_13_l2338_233800


namespace NUMINAMATH_GPT_system_solution_l2338_233863

theorem system_solution (x y: ℝ) 
  (h1: x + y = 2) 
  (h2: 3 * x + y = 4) : 
  x = 1 ∧ y = 1 :=
sorry

end NUMINAMATH_GPT_system_solution_l2338_233863


namespace NUMINAMATH_GPT_emily_beads_l2338_233881

-- Definitions of the conditions as per step a)
def beads_per_necklace : ℕ := 8
def necklaces : ℕ := 2

-- Theorem statement to prove the equivalent math problem
theorem emily_beads : beads_per_necklace * necklaces = 16 :=
by
  sorry

end NUMINAMATH_GPT_emily_beads_l2338_233881


namespace NUMINAMATH_GPT_average_marks_l2338_233803

theorem average_marks :
  let class1_students := 26
  let class1_avg_marks := 40
  let class2_students := 50
  let class2_avg_marks := 60
  let total_students := class1_students + class2_students
  let total_marks := (class1_students * class1_avg_marks) + (class2_students * class2_avg_marks)
  (total_marks / total_students : ℝ) = 53.16 := by
sorry

end NUMINAMATH_GPT_average_marks_l2338_233803


namespace NUMINAMATH_GPT_neg_neg_one_eq_one_l2338_233825

theorem neg_neg_one_eq_one : -(-1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_neg_neg_one_eq_one_l2338_233825


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l2338_233819

theorem sum_of_squares_of_roots
  (x1 x2 : ℝ) (h : 5 * x1^2 + 6 * x1 - 15 = 0) (h' : 5 * x2^2 + 6 * x2 - 15 = 0) :
  x1^2 + x2^2 = 186 / 25 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l2338_233819


namespace NUMINAMATH_GPT_sqrt_of_product_of_powers_l2338_233827

theorem sqrt_of_product_of_powers :
  (Real.sqrt (4^2 * 5^6) = 500) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_product_of_powers_l2338_233827


namespace NUMINAMATH_GPT_simplify_tan_product_l2338_233846

noncomputable def tan_deg (d : ℝ) : ℝ := Real.tan (d * Real.pi / 180)

theorem simplify_tan_product :
  (1 + tan_deg 10) * (1 + tan_deg 35) = 2 := 
by
  -- Given conditions
  have h1 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  have h2 : tan_deg 10 + tan_deg 35 = 1 - tan_deg 10 * tan_deg 35 :=
    by sorry -- Use tan addition formula here
  -- Proof of the theorem follows from here
  sorry

end NUMINAMATH_GPT_simplify_tan_product_l2338_233846


namespace NUMINAMATH_GPT_wire_length_between_poles_l2338_233855

theorem wire_length_between_poles :
  let d := 18  -- distance between the bottoms of the poles
  let h1 := 6 + 3  -- effective height of the shorter pole
  let h2 := 20  -- height of the taller pole
  let vertical_distance := h2 - h1 -- vertical distance between the tops of the poles
  let hypotenuse := Real.sqrt (d^2 + vertical_distance^2)
  hypotenuse = Real.sqrt 445 :=
by
  sorry

end NUMINAMATH_GPT_wire_length_between_poles_l2338_233855


namespace NUMINAMATH_GPT_ellipse_range_of_k_l2338_233882

theorem ellipse_range_of_k (k : ℝ) :
  (4 - k > 0) → (k - 1 > 0) → (4 - k ≠ k - 1) → (1 < k ∧ k < 4 ∧ k ≠ 5 / 2) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_ellipse_range_of_k_l2338_233882


namespace NUMINAMATH_GPT_unique_peg_placement_l2338_233883

theorem unique_peg_placement :
  ∃! f : Fin 6 → Fin 6 → Option (Fin 6), ∀ i j k, 
    (∃ c, f i k = some c) →
    (∃ c, f j k = some c) →
    i = j ∧ match f i j with
    | some c => f j k ≠ some c
    | none => True :=
  sorry

end NUMINAMATH_GPT_unique_peg_placement_l2338_233883


namespace NUMINAMATH_GPT_trip_first_part_distance_l2338_233885

theorem trip_first_part_distance (x : ℝ) :
  let total_distance : ℝ := 60
  let speed_first : ℝ := 48
  let speed_remaining : ℝ := 24
  let avg_speed : ℝ := 32
  (x / speed_first + (total_distance - x) / speed_remaining = total_distance / avg_speed) ↔ (x = 30) :=
by sorry

end NUMINAMATH_GPT_trip_first_part_distance_l2338_233885


namespace NUMINAMATH_GPT_compute_b_l2338_233864

-- Defining the polynomial and the root conditions
def poly (x a b : ℝ) := x^3 + a * x^2 + b * x + 21

theorem compute_b (a b : ℚ) (h1 : poly (3 + Real.sqrt 5) a b = 0) (h2 : poly (3 - Real.sqrt 5) a b = 0) : 
  b = -27.5 := 
sorry

end NUMINAMATH_GPT_compute_b_l2338_233864


namespace NUMINAMATH_GPT_number_of_children_l2338_233831

theorem number_of_children :
  ∃ a : ℕ, (a % 8 = 5) ∧ (a % 10 = 7) ∧ (100 ≤ a) ∧ (a ≤ 150) ∧ (a = 125) :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l2338_233831


namespace NUMINAMATH_GPT_two_digit_number_satisfies_conditions_l2338_233884

theorem two_digit_number_satisfies_conditions :
  ∃ N : ℕ, (N > 0) ∧ (N < 100) ∧ (N % 2 = 1) ∧ (N % 13 = 0) ∧ (∃ a b : ℕ, N = 10 * a + b ∧ (a * b) = (k : ℕ) * k) ∧ (N = 91) :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_satisfies_conditions_l2338_233884


namespace NUMINAMATH_GPT_range_of_a_l2338_233867

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h_roots : x1 < 1 ∧ 1 < x2) (h_eq : ∀ x, x^2 + a * x - 2 = (x - x1) * (x - x2)) : a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2338_233867


namespace NUMINAMATH_GPT_new_percentage_water_is_correct_l2338_233845

def initial_volume : ℕ := 120
def initial_percentage_water : ℚ := 20 / 100
def added_water : ℕ := 8

def initial_volume_water : ℚ := initial_percentage_water * initial_volume
def initial_volume_wine : ℚ := initial_volume - initial_volume_water
def new_volume_water : ℚ := initial_volume_water + added_water
def new_total_volume : ℚ := initial_volume + added_water

def calculate_new_percentage_water : ℚ :=
  (new_volume_water / new_total_volume) * 100

theorem new_percentage_water_is_correct :
  calculate_new_percentage_water = 25 := 
by
  sorry

end NUMINAMATH_GPT_new_percentage_water_is_correct_l2338_233845


namespace NUMINAMATH_GPT_range_of_expression_l2338_233898

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  - π / 6 < 2 * α - β / 2 ∧ 2 * α - β / 2 < π :=
sorry

end NUMINAMATH_GPT_range_of_expression_l2338_233898


namespace NUMINAMATH_GPT_find_original_expression_l2338_233809

theorem find_original_expression (a b c X : ℤ) :
  X + (a * b - 2 * b * c + 3 * a * c) = 2 * b * c - 3 * a * c + 2 * a * b →
  X = 4 * b * c - 6 * a * c + a * b :=
by
  sorry

end NUMINAMATH_GPT_find_original_expression_l2338_233809


namespace NUMINAMATH_GPT_turtle_speed_l2338_233801

theorem turtle_speed
  (hare_speed : ℝ)
  (race_distance : ℝ)
  (head_start : ℝ) :
  hare_speed = 10 → race_distance = 20 → head_start = 18 → 
  (race_distance / (head_start + race_distance / hare_speed) = 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_turtle_speed_l2338_233801


namespace NUMINAMATH_GPT_ab_range_l2338_233806

theorem ab_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a * b = a + b) : 1 / 4 ≤ a * b :=
sorry

end NUMINAMATH_GPT_ab_range_l2338_233806


namespace NUMINAMATH_GPT_rickshaw_distance_l2338_233811

theorem rickshaw_distance (km1_charge : ℝ) (rate_per_km : ℝ) (total_km : ℝ) (total_charge : ℝ) :
  km1_charge = 13.50 → rate_per_km = 2.50 → total_km = 13 → total_charge = 103.5 → (total_charge - km1_charge) / rate_per_km = 36 :=
by
  intro h1 h2 h3 h4
  -- We would fill in proof steps here, but skipping as required.
  sorry

end NUMINAMATH_GPT_rickshaw_distance_l2338_233811


namespace NUMINAMATH_GPT_common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l2338_233858

-- a) Prove the statements about the number of weeks and extra days
theorem common_year_has_52_weeks_1_day: 
  ∀ (days_in_common_year : ℕ), 
  days_in_common_year = 365 → 
  (days_in_common_year / 7 = 52 ∧ days_in_common_year % 7 = 1)
:= by
  sorry

theorem leap_year_has_52_weeks_2_days: 
  ∀ (days_in_leap_year : ℕ), 
  days_in_leap_year = 366 → 
  (days_in_leap_year / 7 = 52 ∧ days_in_leap_year % 7 = 2)
:= by
  sorry

-- b) If a common year starts on a Tuesday, prove the following year starts on a Wednesday
theorem next_year_starts_on_wednesday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (365 % 7 = 1) → 
  ((start_day + 365 % 7) % 7 = 3)
:= by
  sorry

-- c) If a leap year starts on a Tuesday, prove the following year starts on a Thursday
theorem next_year_starts_on_thursday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (366 % 7 = 2) →
  ((start_day + 366 % 7) % 7 = 4)
:= by
  sorry

end NUMINAMATH_GPT_common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l2338_233858


namespace NUMINAMATH_GPT_find_a5_l2338_233849

variable {a_n : ℕ → ℤ} -- Type of the arithmetic sequence
variable (d : ℤ)       -- Common difference of the sequence

-- Assuming the sequence is defined as an arithmetic progression
axiom arithmetic_seq (a d : ℤ) : ∀ n : ℕ, a_n n = a + n * d

theorem find_a5
  (h : a_n 3 + a_n 4 + a_n 5 + a_n 6 + a_n 7 = 45):
  a_n 5 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_find_a5_l2338_233849


namespace NUMINAMATH_GPT_problem1_problem2_l2338_233860

-- Definitions from the conditions
def A (x : ℝ) : Prop := -1 < x ∧ x < 3

def B (x m : ℝ) : Prop := x^2 - 2 * m * x + m^2 - 1 < 0

-- Intersection problem
theorem problem1 (h₁ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₂ : ∀ x, B x 3 ↔ (2 < x ∧ x < 4)) :
  ∀ x, (A x ∧ B x 3) ↔ (2 < x ∧ x < 3) := by
  sorry

-- Union problem
theorem problem2 (h₃ : ∀ x, A x ↔ (-1 < x ∧ x < 3))
  (h₄ : ∀ x m, B x m ↔ ((x - m)^2 < 1)) :
  ∀ m, (0 ≤ m ∧ m ≤ 2) ↔ (∀ x, A x ∨ B x m → A x) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2338_233860


namespace NUMINAMATH_GPT_smallest_m_4_and_n_229_l2338_233892

def satisfies_condition (m n : ℕ) : Prop :=
  19 * m + 8 * n = 1908

def is_smallest_m (m n : ℕ) : Prop :=
  ∀ m' n', satisfies_condition m' n' → m' > 0 → n' > 0 → m ≤ m'

theorem smallest_m_4_and_n_229 : ∃ (m n : ℕ), satisfies_condition m n ∧ is_smallest_m m n ∧ m = 4 ∧ n = 229 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_4_and_n_229_l2338_233892


namespace NUMINAMATH_GPT_borrowed_quarters_l2338_233820

def original_quarters : ℕ := 8
def remaining_quarters : ℕ := 5

theorem borrowed_quarters : original_quarters - remaining_quarters = 3 :=
by
  sorry

end NUMINAMATH_GPT_borrowed_quarters_l2338_233820


namespace NUMINAMATH_GPT_integer_solution_a_l2338_233826

theorem integer_solution_a (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_a_l2338_233826


namespace NUMINAMATH_GPT_cost_of_each_hotdog_l2338_233835

theorem cost_of_each_hotdog (number_of_hotdogs : ℕ) (total_cost : ℕ) (cost_per_hotdog : ℕ) 
    (h1 : number_of_hotdogs = 6) (h2 : total_cost = 300) : cost_per_hotdog = 50 :=
by
  have h3 : cost_per_hotdog = total_cost / number_of_hotdogs :=
    sorry -- here we would normally write the division step
  sorry -- here we would show that h3 implies cost_per_hotdog = 50, given h1 and h2

end NUMINAMATH_GPT_cost_of_each_hotdog_l2338_233835


namespace NUMINAMATH_GPT_total_selling_price_is_correct_l2338_233852

def original_price : ℝ := 120
def discount_rate : ℝ := 0.30
def tax_rate : ℝ := 0.15

def discount : ℝ := discount_rate * original_price
def sale_price : ℝ := original_price - discount
def tax : ℝ := tax_rate * sale_price
def total_selling_price : ℝ := sale_price + tax

theorem total_selling_price_is_correct : total_selling_price = 96.6 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_is_correct_l2338_233852


namespace NUMINAMATH_GPT_coeff_x_squared_l2338_233836

theorem coeff_x_squared (n : ℕ) (t h : ℕ)
  (h_t : t = 4^n) 
  (h_h : h = 2^n) 
  (h_sum : t + h = 272)
  (C : ℕ → ℕ → ℕ) -- binomial coefficient notation, we'll skip the direct proof of properties for simplicity
  : (C 4 4) * (3^0) = 1 := 
by 
  /-
  Proof steps (informal, not needed in Lean statement):
  Since the sum of coefficients is t, we have t = 4^n.
  For the sum of binomial coefficients, we have h = 2^n.
  Given t + h = 272, solve for n:
    4^n + 2^n = 272 
    implies 2^n = 16, so n = 4.
  Substitute into the general term (\(T_{r+1}\):
    T_{r+1} = C_4^r * 3^(4-r) * x^((8+r)/6)
  For x^2 term, set (8+r)/6 = 2, yielding r = 4.
  The coefficient is C_4^4 * 3^0 = 1.
  -/
  sorry

end NUMINAMATH_GPT_coeff_x_squared_l2338_233836


namespace NUMINAMATH_GPT_distance_between_homes_l2338_233804

-- Define the parameters
def maxwell_speed : ℝ := 4  -- km/h
def brad_speed : ℝ := 6     -- km/h
def maxwell_time_to_meet : ℝ := 2  -- hours
def brad_start_delay : ℝ := 1  -- hours

-- Definitions related to the timings
def brad_time_to_meet : ℝ := maxwell_time_to_meet - brad_start_delay  -- hours

-- Define the distances covered by each
def maxwell_distance : ℝ := maxwell_speed * maxwell_time_to_meet  -- km
def brad_distance : ℝ := brad_speed * brad_time_to_meet  -- km

-- Define the total distance between their homes
def total_distance : ℝ := maxwell_distance + brad_distance  -- km

-- Statement to prove
theorem distance_between_homes : total_distance = 14 :=
by
  -- The proof is omitted; add 'sorry' to indicate this.
  sorry

end NUMINAMATH_GPT_distance_between_homes_l2338_233804


namespace NUMINAMATH_GPT_fourth_root_of_25000000_eq_70_7_l2338_233873

theorem fourth_root_of_25000000_eq_70_7 :
  Real.sqrt (Real.sqrt 25000000) = 70.7 :=
sorry

end NUMINAMATH_GPT_fourth_root_of_25000000_eq_70_7_l2338_233873


namespace NUMINAMATH_GPT_friends_received_pebbles_l2338_233813

-- Define the conditions as expressions
def total_weight_kg : ℕ := 36
def weight_per_pebble_g : ℕ := 250
def pebbles_per_friend : ℕ := 4

-- Convert the total weight from kilograms to grams
def total_weight_g : ℕ := total_weight_kg * 1000

-- Calculate the total number of pebbles
def total_pebbles : ℕ := total_weight_g / weight_per_pebble_g

-- Calculate the total number of friends who received pebbles
def number_of_friends : ℕ := total_pebbles / pebbles_per_friend

-- The theorem to prove the number of friends
theorem friends_received_pebbles : number_of_friends = 36 := by
  sorry

end NUMINAMATH_GPT_friends_received_pebbles_l2338_233813


namespace NUMINAMATH_GPT_max_marks_l2338_233851

theorem max_marks (M : ℝ) (h1 : 80 + 10 = 90) (h2 : 0.30 * M = 90) : M = 300 :=
by
  sorry

end NUMINAMATH_GPT_max_marks_l2338_233851


namespace NUMINAMATH_GPT_trey_total_hours_l2338_233862

def num_clean_house := 7
def num_shower := 1
def num_make_dinner := 4
def minutes_per_item := 10
def total_items := num_clean_house + num_shower + num_make_dinner
def total_minutes := total_items * minutes_per_item
def minutes_in_hour := 60

theorem trey_total_hours : total_minutes / minutes_in_hour = 2 := by
  sorry

end NUMINAMATH_GPT_trey_total_hours_l2338_233862


namespace NUMINAMATH_GPT_least_odd_prime_factor_2027_l2338_233848

-- Definitions for the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def order_divides (a n p : ℕ) : Prop := a ^ n % p = 1

-- Define lean function to denote the problem.
theorem least_odd_prime_factor_2027 :
  ∀ p : ℕ, 
  is_prime p → 
  order_divides 2027 12 p ∧ ¬ order_divides 2027 6 p → 
  p ≡ 1 [MOD 12] → 
  2027^6 + 1 % p = 0 → 
  p = 37 :=
by
  -- skipping proof steps
  sorry

end NUMINAMATH_GPT_least_odd_prime_factor_2027_l2338_233848


namespace NUMINAMATH_GPT_arc_length_sector_l2338_233805

theorem arc_length_sector (r : ℝ) (α : ℝ) (h1 : r = 2) (h2 : α = π / 3) : 
  α * r = 2 * π / 3 := 
by 
  sorry

end NUMINAMATH_GPT_arc_length_sector_l2338_233805


namespace NUMINAMATH_GPT_total_cost_for_round_trip_l2338_233843

def time_to_cross_one_way : ℕ := 4 -- time in hours to cross the lake one way
def cost_per_hour : ℕ := 10 -- cost in dollars per hour

def total_time := time_to_cross_one_way * 2 -- total time in hours for a round trip
def total_cost := total_time * cost_per_hour -- total cost in dollars for the assistant

theorem total_cost_for_round_trip : total_cost = 80 := by
  repeat {sorry} -- Leaving the proof for now

end NUMINAMATH_GPT_total_cost_for_round_trip_l2338_233843


namespace NUMINAMATH_GPT_zeros_of_f_l2338_233895

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem zeros_of_f : (f (-1) = 0) ∧ (f 1 = 0) ∧ (f 2 = 0) :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_zeros_of_f_l2338_233895
