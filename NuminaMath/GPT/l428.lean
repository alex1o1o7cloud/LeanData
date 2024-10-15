import Mathlib

namespace NUMINAMATH_GPT_binom_20_19_eq_20_l428_42860

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_20_19_eq_20 : binom 20 19 = 20 := by
  sorry

end NUMINAMATH_GPT_binom_20_19_eq_20_l428_42860


namespace NUMINAMATH_GPT_Jenine_pencil_count_l428_42868

theorem Jenine_pencil_count
  (sharpenings_per_pencil : ℕ)
  (hours_per_sharpening : ℝ)
  (total_hours_needed : ℝ)
  (cost_per_pencil : ℝ)
  (budget : ℝ)
  (already_has_pencils : ℕ) :
  sharpenings_per_pencil = 5 →
  hours_per_sharpening = 1.5 →
  total_hours_needed = 105 →
  cost_per_pencil = 2 →
  budget = 8 →
  already_has_pencils = 10 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_Jenine_pencil_count_l428_42868


namespace NUMINAMATH_GPT_speed_with_stream_l428_42844

-- Define the given conditions
def V_m : ℝ := 7 -- Man's speed in still water (7 km/h)
def V_as : ℝ := 10 -- Man's speed against the stream (10 km/h)

-- Define the stream's speed as the difference
def V_s : ℝ := V_m - V_as

-- Define man's speed with the stream
def V_ws : ℝ := V_m + V_s

-- (Correct Answer): Prove the man's speed with the stream is 10 km/h
theorem speed_with_stream :
  V_ws = 10 := by
  -- Sorry for no proof required in this task
  sorry

end NUMINAMATH_GPT_speed_with_stream_l428_42844


namespace NUMINAMATH_GPT_solve_for_x_l428_42894

theorem solve_for_x : (∃ x : ℝ, (x / 18) * (x / 72) = 1) → ∃ x : ℝ, x = 36 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l428_42894


namespace NUMINAMATH_GPT_correlation_comparison_l428_42805

-- Definitions of the datasets
def data_XY : List (ℝ × ℝ) := [(10,1), (11.3,2), (11.8,3), (12.5,4), (13,5)]
def data_UV : List (ℝ × ℝ) := [(10,5), (11.3,4), (11.8,3), (12.5,2), (13,1)]

-- Definitions of the linear correlation coefficients
noncomputable def r1 : ℝ := sorry -- Calculation of correlation coefficient between X and Y
noncomputable def r2 : ℝ := sorry -- Calculation of correlation coefficient between U and V

-- The proof statement
theorem correlation_comparison :
  r2 < 0 ∧ 0 < r1 :=
sorry

end NUMINAMATH_GPT_correlation_comparison_l428_42805


namespace NUMINAMATH_GPT_count_prime_boring_lt_10000_l428_42873

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_boring (n : ℕ) : Prop := 
  let digits := n.digits 10
  match digits with
  | [] => false
  | (d::ds) => ds.all (fun x => x = d)

theorem count_prime_boring_lt_10000 : 
  ∃! n, is_prime n ∧ is_boring n ∧ n < 10000 := 
by 
  sorry

end NUMINAMATH_GPT_count_prime_boring_lt_10000_l428_42873


namespace NUMINAMATH_GPT_length_of_shorter_leg_l428_42831

variable (h x : ℝ)

theorem length_of_shorter_leg 
  (h_med : h / 2 = 5 * Real.sqrt 3) 
  (hypotenuse_relation : h = 2 * x) 
  (median_relation : h / 2 = h / 2) :
  x = 5 := by sorry

end NUMINAMATH_GPT_length_of_shorter_leg_l428_42831


namespace NUMINAMATH_GPT_percentage_increase_of_gross_sales_l428_42825

theorem percentage_increase_of_gross_sales 
  (P R : ℝ) 
  (orig_gross new_price new_qty new_gross : ℝ)
  (h1 : new_price = 0.8 * P)
  (h2 : new_qty = 1.8 * R)
  (h3 : orig_gross = P * R)
  (h4 : new_gross = new_price * new_qty) :
  ((new_gross - orig_gross) / orig_gross) * 100 = 44 :=
by sorry

end NUMINAMATH_GPT_percentage_increase_of_gross_sales_l428_42825


namespace NUMINAMATH_GPT_distinct_sequences_count_l428_42892

def letters := ["E", "Q", "U", "A", "L", "S"]

noncomputable def count_sequences : Nat :=
  let remaining_letters := ["E", "Q", "U", "A"] -- 'L' and 'S' are already considered
  3 * (4 * 3) -- as analyzed: (LS__) + (L_S_) + (L__S)

theorem distinct_sequences_count : count_sequences = 36 := 
  by
    unfold count_sequences
    sorry

end NUMINAMATH_GPT_distinct_sequences_count_l428_42892


namespace NUMINAMATH_GPT_ryan_fraction_l428_42896

-- Define the total amount of money
def total_money : ℕ := 48

-- Define that Ryan owns a fraction R of the total money
variable {R : ℚ}

-- Define the debts
def ryan_owes_leo : ℕ := 10
def leo_owes_ryan : ℕ := 7

-- Define the final amount Leo has after settling the debts
def leo_final_amount : ℕ := 19

-- Define the condition that Leo and Ryan together have $48
def leo_plus_ryan (leo_amount ryan_amount : ℚ) : Prop := 
  leo_amount + ryan_amount = total_money

-- Define Ryan's amount as a fraction R of the total money
def ryan_amount (R : ℚ) : ℚ := R * total_money

-- Define Leo's amount before debts were settled
def leo_amount_before_debts : ℚ := (leo_final_amount : ℚ) + leo_owes_ryan

-- Define the equation after settling debts
def leo_final_eq (leo_amount_before_debts : ℚ) : Prop :=
  (leo_amount_before_debts - ryan_owes_leo = leo_final_amount)

-- The Lean theorem that needs to be proved
theorem ryan_fraction :
  ∃ (R : ℚ), leo_plus_ryan (leo_amount_before_debts - ryan_owes_leo) (ryan_amount R)
  ∧ leo_final_eq leo_amount_before_debts
  ∧ R = 11 / 24 :=
sorry

end NUMINAMATH_GPT_ryan_fraction_l428_42896


namespace NUMINAMATH_GPT_tensor_identity_l428_42832

namespace tensor_problem

def otimes (x y : ℝ) : ℝ := x^2 + y

theorem tensor_identity (a : ℝ) : otimes a (otimes a a) = 2 * a^2 + a :=
by sorry

end tensor_problem

end NUMINAMATH_GPT_tensor_identity_l428_42832


namespace NUMINAMATH_GPT_trig_identity_l428_42814

theorem trig_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) : 
  Real.sin ((5 * π) / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l428_42814


namespace NUMINAMATH_GPT_train_ride_cost_difference_l428_42865

-- Definitions based on the conditions
def bus_ride_cost : ℝ := 1.40
def total_cost : ℝ := 9.65

-- Lemma to prove the mathematical question
theorem train_ride_cost_difference :
  ∃ T : ℝ, T + bus_ride_cost = total_cost ∧ (T - bus_ride_cost) = 6.85 :=
by
  sorry

end NUMINAMATH_GPT_train_ride_cost_difference_l428_42865


namespace NUMINAMATH_GPT_exists_n_lt_p_minus_1_not_div_p2_l428_42804

theorem exists_n_lt_p_minus_1_not_div_p2 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_3 : p > 3) :
  ∃ (n : ℕ), n < p - 1 ∧ ¬(p^2 ∣ (n^((p - 1)) - 1)) ∧ ¬(p^2 ∣ ((n + 1)^((p - 1)) - 1)) := 
sorry

end NUMINAMATH_GPT_exists_n_lt_p_minus_1_not_div_p2_l428_42804


namespace NUMINAMATH_GPT_parallel_lines_minimum_distance_l428_42801

theorem parallel_lines_minimum_distance :
  ∀ (m n : ℝ) (k : ℝ), 
  k = 2 ∧ ∀ (L1 L2 : ℝ → ℝ), -- we define L1 and L2 as functions
  (L1 = λ y => 2 * y + 3) ∧ (L2 = λ y => k * y - 1) ∧ 
  ((L1 n = m) ∧ (L2 (n + k) = m + 2)) → 
  dist (m, n) (m + 2, n + 2) = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_parallel_lines_minimum_distance_l428_42801


namespace NUMINAMATH_GPT_ratio_of_volumes_total_surface_area_smaller_cube_l428_42853

-- Definitions using the conditions in (a)
def edge_length_smaller_cube := 4 -- in inches
def edge_length_larger_cube := 24 -- in inches (2 feet converted to inches)

-- Propositions based on the correct answers in (b)
theorem ratio_of_volumes : 
  (edge_length_smaller_cube ^ 3) / (edge_length_larger_cube ^ 3) = 1 / 216 := by
  sorry

theorem total_surface_area_smaller_cube : 
  6 * (edge_length_smaller_cube ^ 2) = 96 := by
  sorry

end NUMINAMATH_GPT_ratio_of_volumes_total_surface_area_smaller_cube_l428_42853


namespace NUMINAMATH_GPT_exists_invertible_int_matrix_l428_42820

theorem exists_invertible_int_matrix (m : ℕ) (k : Fin m → ℤ) : 
  ∃ A : Matrix (Fin m) (Fin m) ℤ,
    (∀ j, IsUnit (A + k j • (1 : Matrix (Fin m) (Fin m) ℤ))) :=
sorry

end NUMINAMATH_GPT_exists_invertible_int_matrix_l428_42820


namespace NUMINAMATH_GPT_ellipse_product_l428_42833

/-- Given conditions:
1. OG = 8
2. The diameter of the inscribed circle of triangle ODG is 4
3. O is the center of an ellipse with major axis AB and minor axis CD
4. Point G is one focus of the ellipse
--/
theorem ellipse_product :
  ∀ (O G D : Point) (a b : ℝ),
    OG = 8 → 
    (a^2 - b^2 = 64) →
    (a - b = 4) →
    (AB = 2*a) →
    (CD = 2*b) →
    (AB * CD = 240) :=
by
  intros O G D a b hOG h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_ellipse_product_l428_42833


namespace NUMINAMATH_GPT_total_cost_of_gas_l428_42840

theorem total_cost_of_gas :
  ∃ x : ℚ, (4 * (x / 4) - 4 * (x / 7) = 40) ∧ x = 280 / 3 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_gas_l428_42840


namespace NUMINAMATH_GPT_simplify_expression_l428_42816

-- Define the variables and conditions
variables {a b x y : ℝ}
variable (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
variable (h2 : x ≠ -(a * y) / b)
variable (h3 : x ≠ (b * y) / a)

-- The Theorem to prove
theorem simplify_expression
  (a b x y : ℝ)
  (h1 : a * b * (x^2 - y^2) + x * y * (a^2 - b^2) ≠ 0)
  (h2 : x ≠ -(a * y) / b)
  (h3 : x ≠ (b * y) / a) :
  (a * b * (x^2 + y^2) + x * y * (a^2 + b^2)) *
  ((a * x + b * y)^2 - 4 * a * b * x * y) /
  (a * b * (x^2 - y^2) + x * y * (a^2 - b^2)) = 
  a^2 * x^2 - b^2 * y^2 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l428_42816


namespace NUMINAMATH_GPT_latus_rectum_of_parabola_l428_42886

theorem latus_rectum_of_parabola :
  (∃ p : ℝ, ∀ x y : ℝ, y = - (1 / 6) * x^2 → y = p ∧ p = 3 / 2) :=
sorry

end NUMINAMATH_GPT_latus_rectum_of_parabola_l428_42886


namespace NUMINAMATH_GPT_evaluate_expression_l428_42806

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = 2) : 
  (x^3 * y^4 * z)^2 = 1 / 104976 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l428_42806


namespace NUMINAMATH_GPT_problem1_problem2_l428_42887

theorem problem1 : 24 - (-16) + (-25) - 15 = 0 :=
by
  sorry

theorem problem2 : (-81) + 2 * (1 / 4) * (4 / 9) / (-16) = -81 - (1 / 16) :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l428_42887


namespace NUMINAMATH_GPT_marina_total_cost_l428_42897

theorem marina_total_cost (E P R X : ℕ) 
    (h1 : 15 + E + P = 47)
    (h2 : 15 + R + X = 58) :
    15 + E + P + R + X = 90 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_marina_total_cost_l428_42897


namespace NUMINAMATH_GPT_jimmy_yellow_marbles_correct_l428_42859

def lorin_black_marbles : ℕ := 4
def alex_black_marbles : ℕ := 2 * lorin_black_marbles
def alex_total_marbles : ℕ := 19
def alex_yellow_marbles : ℕ := alex_total_marbles - alex_black_marbles
def jimmy_yellow_marbles : ℕ := 2 * alex_yellow_marbles

theorem jimmy_yellow_marbles_correct : jimmy_yellow_marbles = 22 := by
  sorry

end NUMINAMATH_GPT_jimmy_yellow_marbles_correct_l428_42859


namespace NUMINAMATH_GPT_correct_system_of_equations_l428_42883

theorem correct_system_of_equations (x y : ℝ) :
  (y - x = 4.5) ∧ (x - y / 2 = 1) ↔
  ((y - x = 4.5) ∧ (x - y / 2 = 1)) :=
by sorry

end NUMINAMATH_GPT_correct_system_of_equations_l428_42883


namespace NUMINAMATH_GPT_quadratic_expression_sum_l428_42874

theorem quadratic_expression_sum :
  ∃ a h k : ℝ, (∀ x, 4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
sorry

end NUMINAMATH_GPT_quadratic_expression_sum_l428_42874


namespace NUMINAMATH_GPT_triangles_side_product_relation_l428_42871

-- Define the two triangles with their respective angles and side lengths
variables (A B C A1 B1 C1 : Type) 
          (angle_A angle_A1 angle_B angle_B1 : ℝ) 
          (a b c a1 b1 c1 : ℝ)

-- Given conditions
def angles_sum_to_180 (angle_A angle_A1 : ℝ) : Prop :=
  angle_A + angle_A1 = 180

def angles_equal (angle_B angle_B1 : ℝ) : Prop :=
  angle_B = angle_B1

-- The main theorem to be proven
theorem triangles_side_product_relation 
  (h1 : angles_sum_to_180 angle_A angle_A1)
  (h2 : angles_equal angle_B angle_B1) :
  a * a1 = b * b1 + c * c1 :=
sorry

end NUMINAMATH_GPT_triangles_side_product_relation_l428_42871


namespace NUMINAMATH_GPT_hex_to_decimal_B4E_l428_42852

def hex_B := 11
def hex_4 := 4
def hex_E := 14
def base := 16
def hex_value := hex_B * base^2 + hex_4 * base^1 + hex_E * base^0

theorem hex_to_decimal_B4E : hex_value = 2894 :=
by
  -- here we would write the proof steps, this is skipped with "sorry"
  sorry

end NUMINAMATH_GPT_hex_to_decimal_B4E_l428_42852


namespace NUMINAMATH_GPT_find_n_l428_42866

noncomputable def angles_periodic_mod_eq (n : ℤ) : Prop :=
  -100 < n ∧ n < 100 ∧ Real.tan (n * Real.pi / 180) = Real.tan (216 * Real.pi / 180)

theorem find_n (n : ℤ) (h : angles_periodic_mod_eq n) : n = 36 :=
  sorry

end NUMINAMATH_GPT_find_n_l428_42866


namespace NUMINAMATH_GPT_function_value_at_minus_two_l428_42812

theorem function_value_at_minus_two {f : ℝ → ℝ} (h : ∀ x : ℝ, x ≠ 0 → f (1/x) + (1/x) * f (-x) = 2 * x) : f (-2) = 7 / 2 :=
sorry

end NUMINAMATH_GPT_function_value_at_minus_two_l428_42812


namespace NUMINAMATH_GPT_grasshoppers_after_transformations_l428_42889

-- Define initial conditions and transformation rules
def initial_crickets : ℕ := 30
def initial_grasshoppers : ℕ := 30

-- Define the transformations
def red_haired_transforms (g : ℕ) (c : ℕ) : ℕ × ℕ :=
  (g - 4, c + 1)

def green_haired_transforms (c : ℕ) (g : ℕ) : ℕ × ℕ :=
  (c - 5, g + 2)

-- Define the total number of transformations and the resulting condition
def total_transformations : ℕ := 18
def final_crickets : ℕ := 0

-- The proof goal
theorem grasshoppers_after_transformations : 
  initial_grasshoppers = 30 → 
  initial_crickets = 30 → 
  (∀ t, t = total_transformations → 
          ∀ g c, 
          (g, c) = (0, 6) → 
          (∃ m n, (m + n = t ∧ final_crickets = c))) →
  final_grasshoppers = 6 :=
by
  sorry

end NUMINAMATH_GPT_grasshoppers_after_transformations_l428_42889


namespace NUMINAMATH_GPT_Kim_morning_routine_time_l428_42807

def total_employees : ℕ := 9
def senior_employees : ℕ := 3
def overtime_employees : ℕ := 4
def regular_employees : ℕ := total_employees - senior_employees
def non_overtime_employees : ℕ := total_employees - overtime_employees

def coffee_time : ℕ := 5
def status_update_time (regular senior : ℕ) : ℕ := (regular * 2) + (senior * 3)
def payroll_update_time (overtime non_overtime : ℕ) : ℕ := (overtime * 3) + (non_overtime * 1)
def email_time : ℕ := 10
def task_allocation_time : ℕ := 7

def total_morning_routine_time : ℕ :=
  coffee_time +
  status_update_time regular_employees senior_employees +
  payroll_update_time overtime_employees non_overtime_employees +
  email_time +
  task_allocation_time

theorem Kim_morning_routine_time : total_morning_routine_time = 60 := by
  sorry

end NUMINAMATH_GPT_Kim_morning_routine_time_l428_42807


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l428_42855

variable {α : Type*} [AddGroup α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
∀ n, a (n + 1) = a n + (a 2 - a 1)

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_a2 : a 2 = 2)
  (h_a3 : a 3 = -4) :
  a 3 - a 2 = -6 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l428_42855


namespace NUMINAMATH_GPT_minimum_sum_l428_42817

theorem minimum_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) + ((a^2 * b) / (18 * b * c)) ≥ 4 / 9 :=
sorry

end NUMINAMATH_GPT_minimum_sum_l428_42817


namespace NUMINAMATH_GPT_reduction_when_fifth_runner_twice_as_fast_l428_42823

theorem reduction_when_fifth_runner_twice_as_fast (T T1 T2 T3 T4 T5 : ℝ)
  (h1 : T = T1 + T2 + T3 + T4 + T5)
  (h_T1 : (T1 / 2 + T2 + T3 + T4 + T5) = 0.95 * T)
  (h_T2 : (T1 + T2 / 2 + T3 + T4 + T5) = 0.90 * T)
  (h_T3 : (T1 + T2 + T3 / 2 + T4 + T5) = 0.88 * T)
  (h_T4 : (T1 + T2 + T3 + T4 / 2 + T5) = 0.85 * T)
  : (T1 + T2 + T3 + T4 + T5 / 2) = 0.92 * T := 
sorry

end NUMINAMATH_GPT_reduction_when_fifth_runner_twice_as_fast_l428_42823


namespace NUMINAMATH_GPT_fred_final_cards_l428_42829

def initial_cards : ℕ := 40
def keith_bought : ℕ := 22
def linda_bought : ℕ := 15

theorem fred_final_cards : initial_cards - keith_bought - linda_bought = 3 :=
by sorry

end NUMINAMATH_GPT_fred_final_cards_l428_42829


namespace NUMINAMATH_GPT_sin_value_l428_42881

theorem sin_value (theta : ℝ) (h : Real.cos (3 * Real.pi / 14 - theta) = 1 / 3) : 
  Real.sin (2 * Real.pi / 7 + theta) = 1 / 3 :=
by
  -- Sorry replaces the actual proof which is not required for this task
  sorry

end NUMINAMATH_GPT_sin_value_l428_42881


namespace NUMINAMATH_GPT_product_as_difference_of_squares_l428_42876

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ( (a + b) / 2 )^2 - ( (a - b) / 2 )^2 :=
by
  sorry

end NUMINAMATH_GPT_product_as_difference_of_squares_l428_42876


namespace NUMINAMATH_GPT_find_c_k_l428_42843

noncomputable def a_n (n d : ℕ) := 1 + (n - 1) * d
noncomputable def b_n (n r : ℕ) := r ^ (n - 1)
noncomputable def c_n (n d r : ℕ) := a_n n d + b_n n r

theorem find_c_k (d r k : ℕ) (hd1 : c_n (k - 1) d r = 200) (hd2 : c_n (k + 1) d r = 2000) :
  c_n k d r = 423 :=
sorry

end NUMINAMATH_GPT_find_c_k_l428_42843


namespace NUMINAMATH_GPT_find_multiple_of_sons_age_l428_42810

theorem find_multiple_of_sons_age (F S k : ℕ) 
  (h1 : F = 33)
  (h2 : F = k * S + 3)
  (h3 : F + 3 = 2 * (S + 3) + 10) : 
  k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_multiple_of_sons_age_l428_42810


namespace NUMINAMATH_GPT_total_time_per_week_l428_42856

noncomputable def meditating_time_per_day : ℝ := 1
noncomputable def reading_time_per_day : ℝ := 2 * meditating_time_per_day
noncomputable def exercising_time_per_day : ℝ := 0.5 * meditating_time_per_day
noncomputable def practicing_time_per_day : ℝ := (1/3) * reading_time_per_day

noncomputable def total_time_per_day : ℝ :=
  meditating_time_per_day + reading_time_per_day + exercising_time_per_day + practicing_time_per_day

theorem total_time_per_week :
  total_time_per_day * 7 = 29.17 := by
  sorry

end NUMINAMATH_GPT_total_time_per_week_l428_42856


namespace NUMINAMATH_GPT_inradius_length_l428_42813

noncomputable def inradius (BC AB AC IC : ℝ) (r : ℝ) : Prop :=
  ∀ (r : ℝ), ((BC = 40) ∧ (AB = AC) ∧ (IC = 24)) →
    r = 4 * Real.sqrt 11

theorem inradius_length (BC AB AC IC : ℝ) (r : ℝ) :
  (BC = 40) ∧ (AB = AC) ∧ (IC = 24) →
  r = 4 * Real.sqrt 11 := 
by
  sorry

end NUMINAMATH_GPT_inradius_length_l428_42813


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l428_42858

theorem eccentricity_of_ellipse :
  (∃ θ : Real, (x = 3 * Real.cos θ) ∧ (y = 4 * Real.sin θ))
  → (∃ e : Real, e = Real.sqrt 7 / 4) := 
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l428_42858


namespace NUMINAMATH_GPT_range_of_k_l428_42882

noncomputable def e := Real.exp 1

theorem range_of_k (k : ℝ) (h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ e ^ (x1 - 1) = |k * x1| ∧ e ^ (x2 - 1) = |k * x2| ∧ e ^ (x3 - 1) = |k * x3|) : k^2 > 1 := sorry

end NUMINAMATH_GPT_range_of_k_l428_42882


namespace NUMINAMATH_GPT_negation_of_p_l428_42890

-- Define the proposition p: ∀ x ∈ ℝ, sin x ≤ 1
def proposition_p : Prop := ∀ x : ℝ, Real.sin x ≤ 1

-- The statement to prove the negation of proposition p
theorem negation_of_p : ¬proposition_p ↔ ∃ x : ℝ, Real.sin x > 1 := by
  sorry

end NUMINAMATH_GPT_negation_of_p_l428_42890


namespace NUMINAMATH_GPT_trapezoid_area_no_solutions_l428_42824

noncomputable def no_solutions_to_trapezoid_problem : Prop :=
  ∀ (b1 b2 : ℕ), 
    (∃ (m n : ℕ), b1 = 10 * m ∧ b2 = 10 * n) →
    (b1 + b2 = 72) → false

theorem trapezoid_area_no_solutions : no_solutions_to_trapezoid_problem :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_no_solutions_l428_42824


namespace NUMINAMATH_GPT_original_number_solution_l428_42818

theorem original_number_solution (x : ℝ) (h1 : 0 < x) (h2 : 1000 * x = 3 * (1 / x)) : x = Real.sqrt 30 / 100 :=
by
  sorry

end NUMINAMATH_GPT_original_number_solution_l428_42818


namespace NUMINAMATH_GPT_find_x_l428_42899

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 5 = 17 ↔ x = 80 / 3 := by
  sorry

end NUMINAMATH_GPT_find_x_l428_42899


namespace NUMINAMATH_GPT_distribution_ways_l428_42888

def number_of_ways_to_distribute_problems : ℕ :=
  let friends := 10
  let problems := 7
  let max_receivers := 3
  let ways_to_choose_friends := Nat.choose friends max_receivers
  let ways_to_distribute_problems := max_receivers ^ problems
  ways_to_choose_friends * ways_to_distribute_problems

theorem distribution_ways :
  number_of_ways_to_distribute_problems = 262440 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_distribution_ways_l428_42888


namespace NUMINAMATH_GPT_max_value_expr_bound_l428_42849

noncomputable def max_value_expr (x : ℝ) : ℝ := 
  x^6 / (x^10 + x^8 - 6 * x^6 + 27 * x^4 + 64)

theorem max_value_expr_bound : 
  ∃ x : ℝ, max_value_expr x ≤ 1 / 8.38 := sorry

end NUMINAMATH_GPT_max_value_expr_bound_l428_42849


namespace NUMINAMATH_GPT_optimal_chalk_length_l428_42872

theorem optimal_chalk_length (l : ℝ) (h₁: 10 ≤ l) (h₂: l ≤ 15) (h₃: l = 12) : l = 12 :=
by
  sorry

end NUMINAMATH_GPT_optimal_chalk_length_l428_42872


namespace NUMINAMATH_GPT_bake_sale_cookies_l428_42857

theorem bake_sale_cookies (raisin_cookies : ℕ) (oatmeal_cookies : ℕ) 
  (h1 : raisin_cookies = 42) 
  (h2 : raisin_cookies / oatmeal_cookies = 6) :
  raisin_cookies + oatmeal_cookies = 49 :=
sorry

end NUMINAMATH_GPT_bake_sale_cookies_l428_42857


namespace NUMINAMATH_GPT_extremum_of_cubic_function_l428_42870

noncomputable def cubic_function (x : ℝ) : ℝ := 2 - x^2 - x^3

theorem extremum_of_cubic_function : 
  ∃ x_max x_min : ℝ, 
    cubic_function x_max = x_max_value ∧ 
    cubic_function x_min = x_min_value ∧ 
    ∀ x : ℝ, cubic_function x ≤ cubic_function x_max ∧ cubic_function x_min ≤ cubic_function x :=
sorry

end NUMINAMATH_GPT_extremum_of_cubic_function_l428_42870


namespace NUMINAMATH_GPT_workshop_total_number_of_workers_l428_42880

theorem workshop_total_number_of_workers
  (average_salary_all : ℝ)
  (average_salary_technicians : ℝ)
  (average_salary_non_technicians : ℝ)
  (num_technicians : ℕ)
  (total_salary_all : ℝ -> ℝ)
  (total_salary_technicians : ℕ -> ℝ)
  (total_salary_non_technicians : ℕ -> ℝ -> ℝ)
  (h1 : average_salary_all = 9000)
  (h2 : average_salary_technicians = 12000)
  (h3 : average_salary_non_technicians = 6000)
  (h4 : num_technicians = 7)
  (h5 : ∀ W, total_salary_all W = average_salary_all * W )
  (h6 : ∀ n, total_salary_technicians n = n * average_salary_technicians )
  (h7 : ∀ n W, total_salary_non_technicians n W = (W - n) * average_salary_non_technicians)
  (h8 : ∀ W, total_salary_all W = total_salary_technicians num_technicians + total_salary_non_technicians num_technicians W) :
  ∃ W, W = 14 :=
by
  sorry

end NUMINAMATH_GPT_workshop_total_number_of_workers_l428_42880


namespace NUMINAMATH_GPT_custom_operation_difference_correct_l428_42839

def custom_operation (x y : ℕ) : ℕ := x * y + 2 * x

theorem custom_operation_difference_correct :
  custom_operation 5 3 - custom_operation 3 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_custom_operation_difference_correct_l428_42839


namespace NUMINAMATH_GPT_probability_sum_10_l428_42884

def is_valid_roll (d1 d2 d3 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6 ∧ 1 ≤ d3 ∧ d3 ≤ 6

def sum_is_10 (d1 d2 d3 : ℕ) : Prop :=
  d1 + d2 + d3 = 10

def valid_rolls_count : ℕ :=
  216 -- 6^3 distinct rolls of three 6-sided dice

def successful_rolls_count : ℕ :=
  24 -- number of valid rolls that sum to 10

theorem probability_sum_10 :
  (successful_rolls_count : ℚ) / valid_rolls_count = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_probability_sum_10_l428_42884


namespace NUMINAMATH_GPT_ratio_of_heights_l428_42851

-- Define the height of the first rocket.
def H1 : ℝ := 500

-- Define the combined height of the two rockets.
def combined_height : ℝ := 1500

-- Define the height of the second rocket.
def H2 : ℝ := combined_height - H1

-- The statement to be proven.
theorem ratio_of_heights : H2 / H1 = 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_of_heights_l428_42851


namespace NUMINAMATH_GPT_solution_exists_l428_42854

-- Defining the variables x and y
variables (x y : ℝ)

-- Defining the conditions
def condition_1 : Prop :=
  3 * x ≥ 2 * y + 16

def condition_2 : Prop :=
  x^4 + 2 * (x^2) * (y^2) + y^4 + 25 - 26 * (x^2) - 26 * (y^2) = 72 * x * y

-- Stating the theorem that (6, 1) satisfies the conditions
theorem solution_exists : condition_1 6 1 ∧ condition_2 6 1 :=
by
  -- Convert conditions into expressions
  have h1 : condition_1 6 1 := by sorry
  have h2 : condition_2 6 1 := by sorry
  -- Conjunction of both conditions is satisfied
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_solution_exists_l428_42854


namespace NUMINAMATH_GPT_A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l428_42879

def A : Set ℝ := { x | x^2 + x - 2 < 0 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

theorem A_union_B_when_m_neg_half : A ∪ B (-1/2) = { x | -2 < x ∧ x < 3/2 } :=
by
  sorry

theorem B_subset_A_implies_m_geq_zero (m : ℝ) : B m ⊆ A → 0 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_A_union_B_when_m_neg_half_B_subset_A_implies_m_geq_zero_l428_42879


namespace NUMINAMATH_GPT_vertex_at_fixed_point_l428_42867

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 1

theorem vertex_at_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_vertex_at_fixed_point_l428_42867


namespace NUMINAMATH_GPT_right_angled_triangle_only_B_l428_42846

def forms_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem right_angled_triangle_only_B :
  forms_right_angled_triangle 1 (Real.sqrt 3) 2 ∧
  ¬forms_right_angled_triangle 1 2 2 ∧
  ¬forms_right_angled_triangle 4 5 6 ∧
  ¬forms_right_angled_triangle 1 1 (Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_only_B_l428_42846


namespace NUMINAMATH_GPT_friends_prove_l428_42847

theorem friends_prove (a b c d : ℕ) (h1 : 3^a * 7^b = 3^c * 7^d) (h2 : 3^a * 7^b = 21) :
  (a - 1) * (d - 1) = (b - 1) * (c - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_friends_prove_l428_42847


namespace NUMINAMATH_GPT_fraction_simplified_form_l428_42837

variables (a b c : ℝ)

noncomputable def fraction : ℝ := (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b)

theorem fraction_simplified_form (h : a^2 - c^2 + b^2 + 2 * a * b ≠ 0) :
  fraction a b c = (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b) :=
by sorry

end NUMINAMATH_GPT_fraction_simplified_form_l428_42837


namespace NUMINAMATH_GPT_rod_total_length_l428_42828

theorem rod_total_length (n : ℕ) (piece_length : ℝ) (total_length : ℝ) 
  (h1 : n = 50) 
  (h2 : piece_length = 0.85) 
  (h3 : total_length = n * piece_length) : 
  total_length = 42.5 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_rod_total_length_l428_42828


namespace NUMINAMATH_GPT_largest_sum_valid_set_l428_42802

-- Define the conditions for the set S
def valid_set (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, 0 < x ∧ x ≤ 15) ∧
  ∀ (A B : Finset ℕ), A ⊆ S → B ⊆ S → A ≠ B → A ∩ B = ∅ → A.sum id ≠ B.sum id

-- The theorem stating the largest sum of such a set
theorem largest_sum_valid_set : ∃ (S : Finset ℕ), valid_set S ∧ S.sum id = 61 :=
sorry

end NUMINAMATH_GPT_largest_sum_valid_set_l428_42802


namespace NUMINAMATH_GPT_simplify_expression_l428_42861

theorem simplify_expression (x y : ℝ) (P Q : ℝ) (hP : P = 2 * x + 3 * y) (hQ : Q = 3 * x + 2 * y) :
  ((P + Q) / (P - Q)) - ((P - Q) / (P + Q)) = (24 * x ^ 2 + 52 * x * y + 24 * y ^ 2) / (5 * x * y - 5 * y ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l428_42861


namespace NUMINAMATH_GPT_find_integer_b_l428_42877

theorem find_integer_b (z : ℝ) : ∃ b : ℝ, (z^2 - 6*z + 17 = (z - 3)^2 + b) ∧ b = 8 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_find_integer_b_l428_42877


namespace NUMINAMATH_GPT_evaluate_g_l428_42850

def g (a b c d : ℤ) : ℚ := (d * (c + 2 * a)) / (c + b)

theorem evaluate_g : g 4 (-1) (-8) 2 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_g_l428_42850


namespace NUMINAMATH_GPT_total_spending_march_to_july_l428_42842

-- Define the conditions
def beginning_of_march_spending : ℝ := 1.2
def end_of_july_spending : ℝ := 4.8

-- State the theorem to prove
theorem total_spending_march_to_july : 
  end_of_july_spending - beginning_of_march_spending = 3.6 :=
sorry

end NUMINAMATH_GPT_total_spending_march_to_july_l428_42842


namespace NUMINAMATH_GPT_train_length_is_approx_l428_42895

noncomputable def train_length : ℝ :=
  let speed_kmh : ℝ := 54
  let conversion_factor : ℝ := 1000 / 3600
  let speed_ms : ℝ := speed_kmh * conversion_factor
  let time_seconds : ℝ := 11.999040076793857
  speed_ms * time_seconds

theorem train_length_is_approx : abs (train_length - 179.99) < 0.001 := 
by
  sorry

end NUMINAMATH_GPT_train_length_is_approx_l428_42895


namespace NUMINAMATH_GPT_find_A_and_B_l428_42878

theorem find_A_and_B (A : ℕ) (B : ℕ) (x y : ℕ) 
  (h1 : 1000 ≤ A ∧ A ≤ 9999) 
  (h2 : B = 10^5 * x + 10 * A + y) 
  (h3 : B = 21 * A)
  (h4 : x < 10) 
  (h5 : y < 10) : 
  A = 9091 ∧ B = 190911 :=
sorry

end NUMINAMATH_GPT_find_A_and_B_l428_42878


namespace NUMINAMATH_GPT_player_matches_l428_42836

theorem player_matches (n : ℕ) :
  (34 * n + 78 = 38 * (n + 1)) → n = 10 :=
by
  intro h
  have h1 : 34 * n + 78 = 38 * n + 38 := by sorry
  have h2 : 78 = 4 * n + 38 := by sorry
  have h3 : 40 = 4 * n := by sorry
  have h4 : n = 10 := by sorry
  exact h4

end NUMINAMATH_GPT_player_matches_l428_42836


namespace NUMINAMATH_GPT_fill_time_with_leak_is_correct_l428_42835

-- Define the conditions
def time_to_fill_without_leak := 8
def time_to_empty_with_leak := 24

-- Define the rates
def fill_rate := 1 / time_to_fill_without_leak
def leak_rate := 1 / time_to_empty_with_leak
def effective_fill_rate := fill_rate - leak_rate

-- Prove the time to fill with leak
def time_to_fill_with_leak := 1 / effective_fill_rate

-- The theorem to prove that the time is 12 hours
theorem fill_time_with_leak_is_correct :
  time_to_fill_with_leak = 12 := by
  simp [time_to_fill_without_leak, time_to_empty_with_leak, fill_rate, leak_rate, effective_fill_rate, time_to_fill_with_leak]
  sorry

end NUMINAMATH_GPT_fill_time_with_leak_is_correct_l428_42835


namespace NUMINAMATH_GPT_speed_of_third_part_l428_42891

theorem speed_of_third_part (d : ℝ) (v : ℝ)
  (h1 : 3 * d = 3.000000000000001)
  (h2 : d / 3 + d / 4 + d / v = 47/60) :
  v = 5 := by
  sorry

end NUMINAMATH_GPT_speed_of_third_part_l428_42891


namespace NUMINAMATH_GPT_student_correct_ans_l428_42863

theorem student_correct_ans (c w : ℕ) (h1 : c + w = 80) (h2 : 4 * c - w = 120) : c = 40 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_ans_l428_42863


namespace NUMINAMATH_GPT_fruit_seller_stock_l428_42800

-- Define the given conditions
def remaining_oranges : ℝ := 675
def remaining_percentage : ℝ := 0.25

-- Define the problem function
def original_stock (O : ℝ) : Prop :=
  remaining_percentage * O = remaining_oranges

-- Prove the original stock of oranges was 2700 kg
theorem fruit_seller_stock : original_stock 2700 :=
by
  sorry

end NUMINAMATH_GPT_fruit_seller_stock_l428_42800


namespace NUMINAMATH_GPT_coach_recommendation_l428_42885

def shots_A : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def shots_B : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) (mean : ℚ) : ℚ :=
  (l.map (λ x => (x - mean) ^ 2)).sum / l.length

noncomputable def recommendation (shots_A shots_B : List ℕ) : String :=
  let avg_A := average shots_A
  let avg_B := average shots_B
  let var_A := variance shots_A avg_A
  let var_B := variance shots_B avg_B
  if avg_A = avg_B ∧ var_A > var_B then "player B" else "player A"

theorem coach_recommendation : recommendation shots_A shots_B = "player B" :=
  by
  sorry

end NUMINAMATH_GPT_coach_recommendation_l428_42885


namespace NUMINAMATH_GPT_Yuna_place_l428_42838

theorem Yuna_place (Eunji_place : ℕ) (distance : ℕ) (Yuna_place : ℕ) 
  (h1 : Eunji_place = 100) 
  (h2 : distance = 11) 
  (h3 : Yuna_place = Eunji_place + distance) : 
  Yuna_place = 111 := 
sorry

end NUMINAMATH_GPT_Yuna_place_l428_42838


namespace NUMINAMATH_GPT_factorize_x4_plus_81_l428_42862

noncomputable def factorize_poly (x : ℝ) : (ℝ × ℝ) :=
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  (p, q)

theorem factorize_x4_plus_81 : ∀ x : ℝ, (x^4 + 81) = (factorize_poly x).fst * (factorize_poly x).snd := by
  intro x
  let p := (x^2 + 3*x + 4.5)
  let q := (x^2 - 3*x + 4.5)
  have h : x^4 + 81 = p * q
  { sorry }
  exact h

end NUMINAMATH_GPT_factorize_x4_plus_81_l428_42862


namespace NUMINAMATH_GPT_jane_spent_more_on_ice_cream_l428_42845

-- Definitions based on the conditions
def ice_cream_cone_cost : ℕ := 5
def pudding_cup_cost : ℕ := 2
def ice_cream_cones_bought : ℕ := 15
def pudding_cups_bought : ℕ := 5

-- The mathematically equivalent proof statement
theorem jane_spent_more_on_ice_cream : 
  (ice_cream_cones_bought * ice_cream_cone_cost - pudding_cups_bought * pudding_cup_cost) = 65 := 
by
  sorry

end NUMINAMATH_GPT_jane_spent_more_on_ice_cream_l428_42845


namespace NUMINAMATH_GPT_closest_point_l428_42819

noncomputable def point_on_line_closest_to (x y : ℝ) : ℝ × ℝ :=
( -11 / 5, 7 / 5 )

theorem closest_point (x y : ℝ) (h_line : y = 2 * x + 3) (h_point : (x, y) = (3, -4)) :
  point_on_line_closest_to x y = ( -11 / 5, 7 / 5 ) :=
sorry

end NUMINAMATH_GPT_closest_point_l428_42819


namespace NUMINAMATH_GPT_max_min_sum_l428_42808

variable {α : Type*} [LinearOrderedField α]

def is_odd_function (g : α → α) : Prop :=
∀ x, g (-x) = - g x

def has_max_min (f : α → α) (M N : α) : Prop :=
  (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ (∀ x, N ≤ f x) ∧ (∃ x₁, f x₁ = N)

theorem max_min_sum (g f : α → α) (M N : α)
  (h_odd : is_odd_function g)
  (h_def : ∀ x, f x = g (x - 2) + 1)
  (h_max_min : has_max_min f M N) :
  M + N = 2 :=
sorry

end NUMINAMATH_GPT_max_min_sum_l428_42808


namespace NUMINAMATH_GPT_find_A_l428_42821

theorem find_A (A B : ℕ) (h1: 3 + 6 * (100 + 10 * A + B) = 691) (h2 : 100 ≤ 6 * (100 + 10 * A + B) ∧ 6 * (100 + 10 * A + B) < 1000) : 
A = 8 :=
sorry

end NUMINAMATH_GPT_find_A_l428_42821


namespace NUMINAMATH_GPT_coffee_decaf_percentage_l428_42815

variable (initial_stock : ℝ) (initial_decaf_percent : ℝ)
variable (new_stock : ℝ) (new_decaf_percent : ℝ)

noncomputable def decaf_coffee_percentage : ℝ :=
  let initial_decaf : ℝ := initial_stock * (initial_decaf_percent / 100)
  let new_decaf : ℝ := new_stock * (new_decaf_percent / 100)
  let total_decaf : ℝ := initial_decaf + new_decaf
  let total_stock : ℝ := initial_stock + new_stock
  (total_decaf / total_stock) * 100

theorem coffee_decaf_percentage :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  new_stock = 100 →
  new_decaf_percent = 50 →
  decaf_coffee_percentage initial_stock initial_decaf_percent new_stock new_decaf_percent = 26 :=
by
  intros
  sorry

end NUMINAMATH_GPT_coffee_decaf_percentage_l428_42815


namespace NUMINAMATH_GPT_call_center_agents_ratio_l428_42898

noncomputable def fraction_of_agents (calls_A calls_B total_agents total_calls : ℕ) : ℚ :=
  let calls_A_per_agent := calls_A / total_agents
  let calls_B_per_agent := calls_B / total_agents
  let ratio_calls_A_B := (3: ℚ) / 5
  let fraction_calls_B := (8: ℚ) / 11
  let fraction_calls_A := (3: ℚ) / 11
  let ratio_of_agents := (5: ℚ) / 11
  if (calls_A_per_agent * fraction_calls_A = ratio_calls_A_B * calls_B_per_agent) then ratio_of_agents else 0

theorem call_center_agents_ratio (calls_A calls_B total_agents total_calls agents_A agents_B : ℕ) :
  (calls_A : ℚ) / (calls_B : ℚ) = (3 / 5) →
  (calls_B : ℚ) = (8 / 11) * total_calls →
  (agents_A : ℚ) = (5 / 11) * (agents_B : ℚ) :=
sorry

end NUMINAMATH_GPT_call_center_agents_ratio_l428_42898


namespace NUMINAMATH_GPT_min_value_l428_42864

theorem min_value (a : ℝ) (h : a > 0) : a + 4 / a ≥ 4 :=
by sorry

end NUMINAMATH_GPT_min_value_l428_42864


namespace NUMINAMATH_GPT_length_of_first_train_l428_42834

theorem length_of_first_train
  (speed_first : ℕ)
  (speed_second : ℕ)
  (length_second : ℕ)
  (distance_between : ℕ)
  (time_to_cross : ℕ)
  (h1 : speed_first = 10)
  (h2 : speed_second = 15)
  (h3 : length_second = 150)
  (h4 : distance_between = 50)
  (h5 : time_to_cross = 60) :
  ∃ L : ℕ, L = 100 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_train_l428_42834


namespace NUMINAMATH_GPT_binomial_expansion_judgments_l428_42822

theorem binomial_expansion_judgments :
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r) ∧
  (∃ n : ℕ, n > 0 ∧ ∃ r : ℕ, n = 4 * r + 3) :=
by
  sorry

end NUMINAMATH_GPT_binomial_expansion_judgments_l428_42822


namespace NUMINAMATH_GPT_sasha_questions_per_hour_l428_42893

-- Define the total questions and the time she worked, and the remaining questions
def total_questions : ℕ := 60
def time_worked : ℕ := 2
def remaining_questions : ℕ := 30

-- Define the number of questions she completed
def questions_completed := total_questions - remaining_questions

-- Define the rate at which she completes questions per hour
def questions_per_hour := questions_completed / time_worked

-- The theorem to prove
theorem sasha_questions_per_hour : questions_per_hour = 15 := 
by
  -- Here we would prove the theorem, but we're using sorry to skip the proof for now
  sorry

end NUMINAMATH_GPT_sasha_questions_per_hour_l428_42893


namespace NUMINAMATH_GPT_players_in_physics_class_l428_42826

theorem players_in_physics_class (total players_math players_both : ℕ)
    (h1 : total = 15)
    (h2 : players_math = 9)
    (h3 : players_both = 4) :
    (players_math - players_both) + (total - (players_math - players_both + players_both)) + players_both = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_players_in_physics_class_l428_42826


namespace NUMINAMATH_GPT_infinite_gcd_one_l428_42803

theorem infinite_gcd_one : ∃ᶠ n in at_top, Int.gcd n ⌊Real.sqrt 2 * n⌋ = 1 := sorry

end NUMINAMATH_GPT_infinite_gcd_one_l428_42803


namespace NUMINAMATH_GPT_eval_expression_l428_42848

theorem eval_expression : (-3)^4 + (-3)^3 + (-3)^2 + 3^2 + 3^3 + 3^4 = 180 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l428_42848


namespace NUMINAMATH_GPT_multiply_powers_zero_exponent_distribute_term_divide_powers_l428_42809

-- 1. Prove a^{2} \cdot a^{3} = a^{5}
theorem multiply_powers (a : ℝ) : a^2 * a^3 = a^5 := 
sorry

-- 2. Prove (3.142 - π)^{0} = 1
theorem zero_exponent : (3.142 - Real.pi)^0 = 1 := 
sorry

-- 3. Prove 2a(a^{2} - 1) = 2a^{3} - 2a
theorem distribute_term (a : ℝ) : 2 * a * (a^2 - 1) = 2 * a^3 - 2 * a := 
sorry

-- 4. Prove (-m^{3})^{2} \div m^{4} = m^{2}
theorem divide_powers (m : ℝ) : ((-m^3)^2) / (m^4) = m^2 := 
sorry

end NUMINAMATH_GPT_multiply_powers_zero_exponent_distribute_term_divide_powers_l428_42809


namespace NUMINAMATH_GPT_total_toys_l428_42811

theorem total_toys (n : ℕ) (h1 : 3 * (n / 4) = 18) : n = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_toys_l428_42811


namespace NUMINAMATH_GPT_max_star_player_salary_l428_42875

-- Define the constants given in the problem
def num_players : Nat := 12
def min_salary : Nat := 20000
def total_salary_cap : Nat := 1000000

-- Define the statement we want to prove
theorem max_star_player_salary :
  (∃ star_player_salary : Nat, 
    star_player_salary ≤ total_salary_cap - (num_players - 1) * min_salary ∧
    star_player_salary = 780000) :=
sorry

end NUMINAMATH_GPT_max_star_player_salary_l428_42875


namespace NUMINAMATH_GPT_area_of_segment_solution_max_sector_angle_solution_l428_42869
open Real

noncomputable def area_of_segment (α R : ℝ) : ℝ :=
  let l := (R * α)
  let sector := 0.5 * R * l
  let triangle := 0.5 * R^2 * sin α
  sector - triangle

theorem area_of_segment_solution : area_of_segment (π / 3) 10 = 50 * ((π / 3) - (sqrt 3 / 2)) :=
by sorry

noncomputable def max_sector_angle (c : ℝ) (hc : c > 0) : ℝ :=
  2

theorem max_sector_angle_solution (c : ℝ) (hc : c > 0) : max_sector_angle c hc = 2 :=
by sorry

end NUMINAMATH_GPT_area_of_segment_solution_max_sector_angle_solution_l428_42869


namespace NUMINAMATH_GPT_primes_satisfying_condition_l428_42827

theorem primes_satisfying_condition :
    {p : ℕ | p.Prime ∧ ∀ q : ℕ, q.Prime ∧ q < p → ¬ ∃ n : ℕ, n^2 ∣ (p - (p / q) * q)} =
    {2, 3, 5, 7, 13} :=
by sorry

end NUMINAMATH_GPT_primes_satisfying_condition_l428_42827


namespace NUMINAMATH_GPT_min_ties_to_ensure_pairs_l428_42830

variable (red blue green yellow : Nat)
variable (total_ties : Nat)
variable (pairs_needed : Nat)

-- Define the conditions
def conditions : Prop :=
  red = 120 ∧
  blue = 90 ∧
  green = 70 ∧
  yellow = 50 ∧
  total_ties = 27 ∧
  pairs_needed = 12

-- Define the statement to be proven
theorem min_ties_to_ensure_pairs : conditions red blue green yellow total_ties pairs_needed → total_ties = 27 :=
sorry

end NUMINAMATH_GPT_min_ties_to_ensure_pairs_l428_42830


namespace NUMINAMATH_GPT_single_fraction_l428_42841

theorem single_fraction (c : ℕ) : (6 + 5 * c) / 5 + 3 = (21 + 5 * c) / 5 :=
by sorry

end NUMINAMATH_GPT_single_fraction_l428_42841
