import Mathlib

namespace NUMINAMATH_GPT_incorrect_statement_S9_lt_S10_l2368_236883

variable {a : ℕ → ℝ} -- Sequence
variable {S : ℕ → ℝ} -- Sum of the first n terms
variable {d : ℝ}     -- Common difference

-- Arithmetic sequence definition
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0 + n * (n-1) * d / 2)

-- Given conditions
variable 
  (arith_seq : arithmetic_sequence a d)
  (sum_terms : sum_of_first_n_terms a S)
  (H1 : S 9 < S 8)
  (H2 : S 8 = S 7)

-- Prove the statement
theorem incorrect_statement_S9_lt_S10 : 
  ¬ (S 9 < S 10) := 
sorry

end NUMINAMATH_GPT_incorrect_statement_S9_lt_S10_l2368_236883


namespace NUMINAMATH_GPT_total_time_iggy_runs_correct_l2368_236813

noncomputable def total_time_iggy_runs : ℝ :=
  let monday_time := 3 * (10 + 1 + 0.5);
  let tuesday_time := 5 * (9 + 1 + 1);
  let wednesday_time := 7 * (12 - 2 + 2);
  let thursday_time := 10 * (8 + 2 + 4);
  let friday_time := 4 * (10 + 0.25);
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem total_time_iggy_runs_correct : total_time_iggy_runs = 354.5 := by
  sorry

end NUMINAMATH_GPT_total_time_iggy_runs_correct_l2368_236813


namespace NUMINAMATH_GPT_ratio_of_side_lengths_l2368_236827

theorem ratio_of_side_lengths (t p : ℕ) (h1 : 3 * t = 30) (h2 : 5 * p = 30) : t / p = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_side_lengths_l2368_236827


namespace NUMINAMATH_GPT_product_469160_9999_l2368_236879

theorem product_469160_9999 :
  469160 * 9999 = 4690696840 :=
by
  sorry

end NUMINAMATH_GPT_product_469160_9999_l2368_236879


namespace NUMINAMATH_GPT_problem_I_problem_II_problem_III_l2368_236841

noncomputable def f (a x : ℝ) := a * x * Real.exp x
noncomputable def f' (a x : ℝ) := a * (1 + x) * Real.exp x

theorem problem_I (a : ℝ) (h : a ≠ 0) :
  (if a > 0 then ∀ x, (f' a x > 0 ↔ x > -1) ∧ (f' a x < 0 ↔ x < -1)
  else ∀ x, (f' a x > 0 ↔ x < -1) ∧ (f' a x < 0 ↔ x > -1)) :=
sorry

theorem problem_II (h : ∃ a : ℝ, a = 1) :
  ∃ (x : ℝ) (y : ℝ), x = -1 ∧ f 1 (-1) = -1 / Real.exp 1 ∧ ¬ ∃ y, ∀ x, y = f 1 x ∧ (f' 1 x) < 0 :=
sorry

theorem problem_III (h : ∃ m : ℝ, f 1 m = e * m * Real.exp m ∧ f' 1 m = e * (1 + m) * Real.exp m) :
  ∃ a : ℝ, a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_problem_III_l2368_236841


namespace NUMINAMATH_GPT_pie_cost_correct_l2368_236842

-- Define the initial and final amounts of money Mary had.
def initial_amount : ℕ := 58
def final_amount : ℕ := 52

-- Define the cost of the pie as the difference between initial and final amounts.
def pie_cost : ℕ := initial_amount - final_amount

-- State the theorem that given the initial and final amounts, the cost of the pie is 6.
theorem pie_cost_correct : pie_cost = 6 := by 
  sorry

end NUMINAMATH_GPT_pie_cost_correct_l2368_236842


namespace NUMINAMATH_GPT_bob_correct_answers_l2368_236801

-- Define the variables, c for correct answers, w for incorrect answers, total problems 15, score 54
variables (c w : ℕ)

-- Define the conditions
axiom total_problems : c + w = 15
axiom total_score : 6 * c - 3 * w = 54

-- Prove that the number of correct answers is 11
theorem bob_correct_answers : c = 11 :=
by
  -- Here, you would provide the proof, but for the sake of the statement, we'll use sorry.
  sorry

end NUMINAMATH_GPT_bob_correct_answers_l2368_236801


namespace NUMINAMATH_GPT_tangent_value_of_k_k_range_l2368_236826

noncomputable def f (x : Real) : Real := Real.exp (2 * x)
def g (k x : Real) : Real := k * x + 1

theorem tangent_value_of_k (k : Real) :
  (∃ t : Real, f t = g k t ∧ deriv f t = deriv (g k) t) → k = 2 :=
by
  sorry

theorem k_range (k : Real) (h : k > 0) :
  (∃ m : Real, m > 0 ∧ ∀ x : Real, 0 < x → x < m → |f x - g k x| > 2 * x) → 4 < k :=
by
  sorry

end NUMINAMATH_GPT_tangent_value_of_k_k_range_l2368_236826


namespace NUMINAMATH_GPT_savings_correct_l2368_236874

noncomputable def savings (income expenditure : ℕ) : ℕ :=
income - expenditure

theorem savings_correct (I E : ℕ) (h_ratio :  I / E = 10 / 4) (h_income : I = 19000) :
  savings I E = 11400 :=
sorry

end NUMINAMATH_GPT_savings_correct_l2368_236874


namespace NUMINAMATH_GPT_trigonometric_inequality_l2368_236805

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  0 < (1 / (Real.sin x)^2) - (1 / x^2) ∧ (1 / (Real.sin x)^2) - (1 / x^2) < 1 := 
sorry

end NUMINAMATH_GPT_trigonometric_inequality_l2368_236805


namespace NUMINAMATH_GPT_problem1_coefficient_of_x_problem2_maximum_coefficient_term_l2368_236854

-- Problem 1: Coefficient of x term
theorem problem1_coefficient_of_x (n : ℕ) 
  (A : ℕ := (3 + 1)^n) 
  (B : ℕ := 2^n) 
  (h1 : A + B = 272) 
  : true :=  -- Replacing true with actual condition
by sorry

-- Problem 2: Term with maximum coefficient
theorem problem2_maximum_coefficient_term (n : ℕ)
  (h : 1 + n + (n * (n - 1)) / 2 = 79) 
  : true :=  -- Replacing true with actual condition
by sorry

end NUMINAMATH_GPT_problem1_coefficient_of_x_problem2_maximum_coefficient_term_l2368_236854


namespace NUMINAMATH_GPT_solution_in_quadrant_I_l2368_236823

theorem solution_in_quadrant_I (k : ℝ) :
  ∃ x y : ℝ, (2 * x - y = 5 ∧ k * x + 2 * y = 4 ∧ x > 0 ∧ y > 0) ↔ -4 < k ∧ k < 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solution_in_quadrant_I_l2368_236823


namespace NUMINAMATH_GPT_percentage_calculation_l2368_236875

theorem percentage_calculation (amount : ℝ) (percentage : ℝ) (res : ℝ) :
  amount = 400 → percentage = 0.25 → res = amount * percentage → res = 100 := by
  intro h_amount h_percentage h_res
  rw [h_amount, h_percentage] at h_res
  norm_num at h_res
  exact h_res

end NUMINAMATH_GPT_percentage_calculation_l2368_236875


namespace NUMINAMATH_GPT_red_sea_glass_pieces_l2368_236820

theorem red_sea_glass_pieces (R : ℕ) 
    (h_bl : ∃ g : ℕ, g = 12) 
    (h_rose_red : ∃ r_b : ℕ, r_b = 9)
    (h_rose_blue : ∃ b : ℕ, b = 11) 
    (h_dorothy_red : 2 * (R + 9) + 3 * 11 = 57) : R = 3 :=
  by
    sorry

end NUMINAMATH_GPT_red_sea_glass_pieces_l2368_236820


namespace NUMINAMATH_GPT_turner_total_tickets_l2368_236888

-- Definition of conditions
def days := 3
def rollercoaster_rides_per_day := 3
def catapult_rides_per_day := 2
def ferris_wheel_rides_per_day := 1

def rollercoaster_ticket_cost := 4
def catapult_ticket_cost := 4
def ferris_wheel_ticket_cost := 1

-- Proof statement
theorem turner_total_tickets : 
  days * (rollercoaster_rides_per_day * rollercoaster_ticket_cost 
  + catapult_rides_per_day * catapult_ticket_cost 
  + ferris_wheel_rides_per_day * ferris_wheel_ticket_cost) 
  = 63 := 
by
  sorry

end NUMINAMATH_GPT_turner_total_tickets_l2368_236888


namespace NUMINAMATH_GPT_domain_is_correct_l2368_236861

def domain_of_function (x : ℝ) : Prop :=
  (3 - x ≥ 0) ∧ (x + 1 ≠ 0) ∧ (x + 2 > 0)

theorem domain_is_correct :
  { x : ℝ | domain_of_function x } = { x : ℝ | -2 < x ∧ x ≤ 3 ∧ x ≠ -1 } :=
by
  sorry

end NUMINAMATH_GPT_domain_is_correct_l2368_236861


namespace NUMINAMATH_GPT_max_product_of_two_integers_with_sum_2004_l2368_236870

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end NUMINAMATH_GPT_max_product_of_two_integers_with_sum_2004_l2368_236870


namespace NUMINAMATH_GPT_find_product_of_offsets_l2368_236834

theorem find_product_of_offsets
  (a b c : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a * b + a + b = 99)
  (h3 : b * c + b + c = 99)
  (h4 : c * a + c + a = 99) :
  (a + 1) * (b + 1) * (c + 1) = 1000 := by
  sorry

end NUMINAMATH_GPT_find_product_of_offsets_l2368_236834


namespace NUMINAMATH_GPT_equation_1_solutions_equation_2_solutions_l2368_236847

-- Equation 1: Proving solutions for (x+8)(x+1) = -12
theorem equation_1_solutions (x : ℝ) :
  (x + 8) * (x + 1) = -12 ↔ x = -4 ∨ x = -5 :=
sorry

-- Equation 2: Proving solutions for (2x-3)^2 = 5(2x-3)
theorem equation_2_solutions (x : ℝ) :
  (2 * x - 3) ^ 2 = 5 * (2 * x - 3) ↔ x = 3 / 2 ∨ x = 4 :=
sorry

end NUMINAMATH_GPT_equation_1_solutions_equation_2_solutions_l2368_236847


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_cubes_l2368_236873

theorem arithmetic_sequence_sum_cubes (x : ℤ) (k : ℕ) (h : ∀ i, 0 <= i ∧ i <= k → (x + 2 * i : ℤ)^3 =
  -1331) (hk : k > 3) : k = 6 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_cubes_l2368_236873


namespace NUMINAMATH_GPT_base_satisfying_eq_l2368_236864

theorem base_satisfying_eq : ∃ a : ℕ, (11 < a) ∧ (293 * a^2 + 9 * a + 3 + (4 * a^2 + 6 * a + 8) = 7 * a^2 + 3 * a + 11) ∧ (a = 12) :=
by
  sorry

end NUMINAMATH_GPT_base_satisfying_eq_l2368_236864


namespace NUMINAMATH_GPT_find_x_l2368_236833

theorem find_x (x : ℝ) : 0.5 * x + (0.3 * 0.2) = 0.26 ↔ x = 0.4 := by
  sorry

end NUMINAMATH_GPT_find_x_l2368_236833


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2368_236882

variable (a b x : ℝ)
variable (h1 : ∀ x, ax + b > 0 ↔ 1 < x)

theorem solution_set_of_inequality : ∀ x, (ax + b) * (x - 2) < 0 ↔ (1 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2368_236882


namespace NUMINAMATH_GPT_resulting_solid_vertices_l2368_236817

theorem resulting_solid_vertices (s1 s2 : ℕ) (orig_vertices removed_cubes : ℕ) :
  s1 = 5 → s2 = 2 → orig_vertices = 8 → removed_cubes = 8 → 
  (orig_vertices - removed_cubes + removed_cubes * (4 * 3 - 3)) = 40 := by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_resulting_solid_vertices_l2368_236817


namespace NUMINAMATH_GPT_molecular_weight_of_Aluminium_hydroxide_l2368_236890

-- Given conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Definition of molecular weight of Aluminium hydroxide
def molecular_weight_Al_OH_3 : ℝ := 
  atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

-- Proof statement
theorem molecular_weight_of_Aluminium_hydroxide : molecular_weight_Al_OH_3 = 78.01 :=
  by sorry

end NUMINAMATH_GPT_molecular_weight_of_Aluminium_hydroxide_l2368_236890


namespace NUMINAMATH_GPT_flower_problem_l2368_236840

theorem flower_problem
  (O : ℕ) 
  (total : ℕ := 105)
  (pink_purple : ℕ := 30)
  (red := 2 * O)
  (yellow := 2 * O - 5)
  (pink := pink_purple / 2)
  (purple := pink)
  (H1 : pink + purple = pink_purple)
  (H2 : pink_purple = 30)
  (H3 : pink = purple)
  (H4 : O + red + yellow + pink + purple = total)
  (H5 : total = 105):
  O = 16 := 
by 
  sorry

end NUMINAMATH_GPT_flower_problem_l2368_236840


namespace NUMINAMATH_GPT_lcm_of_denominators_l2368_236837

theorem lcm_of_denominators (x : ℕ) [NeZero x] : Nat.lcm (Nat.lcm x (2 * x)) (3 * x^2) = 6 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_denominators_l2368_236837


namespace NUMINAMATH_GPT_complement_A_eq_interval_l2368_236868

-- Define the universal set U as the set of all real numbers.
def U : Set ℝ := Set.univ

-- Define the set A using the condition x^2 - 2x - 3 > 0.
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U.
def A_complement : Set ℝ := { x | -1 <= x ∧ x <= 3 }

theorem complement_A_eq_interval : A_complement = { x | -1 <= x ∧ x <= 3 } :=
by
  sorry

end NUMINAMATH_GPT_complement_A_eq_interval_l2368_236868


namespace NUMINAMATH_GPT_initial_water_in_hole_l2368_236893

theorem initial_water_in_hole (total_needed additional_needed initial : ℕ) (h1 : total_needed = 823) (h2 : additional_needed = 147) :
  initial = total_needed - additional_needed :=
by
  sorry

end NUMINAMATH_GPT_initial_water_in_hole_l2368_236893


namespace NUMINAMATH_GPT_sin_cos_cos_sin_unique_pair_exists_uniq_l2368_236880

noncomputable def theta (x : ℝ) : ℝ := Real.sin (Real.cos x) - x

theorem sin_cos_cos_sin_unique_pair_exists_uniq (h : 0 < c ∧ c < (1/2) * Real.pi ∧ 0 < d ∧ d < (1/2) * Real.pi) :
  (∃! (c d : ℝ), Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d ∧ c < d) :=
sorry

end NUMINAMATH_GPT_sin_cos_cos_sin_unique_pair_exists_uniq_l2368_236880


namespace NUMINAMATH_GPT_first_machine_rate_l2368_236804

theorem first_machine_rate (x : ℝ) (h : (x + 55) * 30 = 2400) : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_first_machine_rate_l2368_236804


namespace NUMINAMATH_GPT_probability_of_darkness_l2368_236822

theorem probability_of_darkness (rev_per_min : ℕ) (stay_in_dark_time : ℕ) (revolution_time : ℕ) (stay_fraction : ℕ → ℚ) :
  rev_per_min = 2 →
  stay_in_dark_time = 10 →
  revolution_time = 60 / rev_per_min →
  stay_fraction stay_in_dark_time / revolution_time = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_darkness_l2368_236822


namespace NUMINAMATH_GPT_last_student_calls_out_l2368_236818

-- Define the transformation rules as a function
def next_student (n : ℕ) : ℕ :=
  if n < 10 then n + 8 else (n % 10) + 7

-- Define the sequence generation function
noncomputable def student_number : ℕ → ℕ
| 0       => 1  -- the 1st student starts with number 1
| (n + 1) => next_student (student_number n)

-- The main theorem to prove
theorem last_student_calls_out (n : ℕ) : student_number 2013 = 12 :=
sorry

end NUMINAMATH_GPT_last_student_calls_out_l2368_236818


namespace NUMINAMATH_GPT_gcd_12_20_l2368_236877

theorem gcd_12_20 : Nat.gcd 12 20 = 4 := by
  sorry

end NUMINAMATH_GPT_gcd_12_20_l2368_236877


namespace NUMINAMATH_GPT_new_rectangle_area_l2368_236862

theorem new_rectangle_area :
  let a := 3
  let b := 4
  let diagonal := Real.sqrt (a^2 + b^2)
  let sum_of_sides := a + b
  let area := diagonal * sum_of_sides
  area = 35 :=
by
  sorry

end NUMINAMATH_GPT_new_rectangle_area_l2368_236862


namespace NUMINAMATH_GPT_complementary_angle_l2368_236844

theorem complementary_angle (α : ℝ) (h : α = 35 + 30 / 60) : 90 - α = 54 + 30 / 60 :=
by
  sorry

end NUMINAMATH_GPT_complementary_angle_l2368_236844


namespace NUMINAMATH_GPT_smallest_next_divisor_l2368_236819

noncomputable def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

noncomputable def has_divisor_323 (n : ℕ) : Prop := 323 ∣ n

theorem smallest_next_divisor (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : has_divisor_323 n) :
  ∃ m : ℕ, m > 323 ∧ m ∣ n ∧ (∀ k : ℕ, k > 323 ∧ k < m → ¬ k ∣ n) ∧ m = 340 :=
sorry

end NUMINAMATH_GPT_smallest_next_divisor_l2368_236819


namespace NUMINAMATH_GPT_quadratic_function_range_l2368_236855

theorem quadratic_function_range (f : ℝ → ℝ) (a : ℝ)
  (h_quad : ∃ p q r : ℝ, ∀ x, f x = p * x^2 + q * x + r)
  (h_sym : ∀ x, f (2 + x) = f (2 - x))
  (h_cond : f a ≤ f 0 ∧ f 0 < f 1) :
  a ≤ 0 ∨ a ≥ 4 :=
sorry

end NUMINAMATH_GPT_quadratic_function_range_l2368_236855


namespace NUMINAMATH_GPT_evaluate_expression_l2368_236803

theorem evaluate_expression : 
  (3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2 = 1372 * 10^1003 := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2368_236803


namespace NUMINAMATH_GPT_students_with_both_pets_l2368_236865

theorem students_with_both_pets
  (D C : Finset ℕ)
  (h_union : (D ∪ C).card = 48)
  (h_D : D.card = 30)
  (h_C : C.card = 34) :
  (D ∩ C).card = 16 :=
by sorry

end NUMINAMATH_GPT_students_with_both_pets_l2368_236865


namespace NUMINAMATH_GPT_max_value_of_z_l2368_236850

theorem max_value_of_z (k : ℝ) (x y : ℝ)
  (h1 : x + 2 * y - 1 ≥ 0)
  (h2 : x - y ≥ 0)
  (h3 : 0 ≤ x)
  (h4 : x ≤ k)
  (h5 : ∀ x y, x + 2 * y - 1 ≥ 0 ∧ x - y ≥ 0 ∧ 0 ≤ x ∧ x ≤ k → x + k * y ≥ -2) :
  ∃ (x y : ℝ), x + k * y = 20 := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_z_l2368_236850


namespace NUMINAMATH_GPT_rabbit_catch_up_time_l2368_236800

theorem rabbit_catch_up_time :
  let rabbit_speed := 25 -- miles per hour
  let cat_speed := 20 -- miles per hour
  let head_start := 15 / 60 -- hours, which is 0.25 hours
  let initial_distance := cat_speed * head_start
  let relative_speed := rabbit_speed - cat_speed
  initial_distance / relative_speed = 1 := by
  sorry

end NUMINAMATH_GPT_rabbit_catch_up_time_l2368_236800


namespace NUMINAMATH_GPT_eva_fruit_diet_l2368_236821

noncomputable def dietary_requirements : Prop :=
  ∃ (days_in_week : ℕ) (days_in_month : ℕ) (apples : ℕ) (bananas : ℕ) (pears : ℕ) (oranges : ℕ),
    days_in_week = 7 ∧
    days_in_month = 30 ∧
    apples = 2 * days_in_week ∧
    bananas = days_in_week / 2 ∧
    pears = 4 ∧
    oranges = days_in_month / 3 ∧
    apples = 14 ∧
    bananas = 4 ∧
    pears = 4 ∧
    oranges = 10

theorem eva_fruit_diet : dietary_requirements :=
sorry

end NUMINAMATH_GPT_eva_fruit_diet_l2368_236821


namespace NUMINAMATH_GPT_graph_passes_quadrants_l2368_236811

theorem graph_passes_quadrants {x y : ℝ} (h : y = -x - 2) :
  -- Statement that the graph passes through the second, third, and fourth quadrants.
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x < 0 ∧ (∃ (y : ℝ), y < 0 ∧ y = -x - 2)) ∧
  (∃ (x : ℝ), x > 0 ∧ (∃ (y : ℝ), y > 0 ∧ y = -x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_quadrants_l2368_236811


namespace NUMINAMATH_GPT_find_k_l2368_236857

variable (m n k : ℚ)

def line_eq (x y : ℚ) : Prop := x - (5/2 : ℚ) * y + 1 = 0

theorem find_k (h1 : line_eq m n) (h2 : line_eq (m + 1/2) (n + 1/k)) : k = 3/5 := by
  sorry

end NUMINAMATH_GPT_find_k_l2368_236857


namespace NUMINAMATH_GPT_smallest_b_undefined_inverse_l2368_236896

theorem smallest_b_undefined_inverse (b : ℕ) (h1 : Nat.gcd b 84 > 1) (h2 : Nat.gcd b 90 > 1) : b = 6 :=
sorry

end NUMINAMATH_GPT_smallest_b_undefined_inverse_l2368_236896


namespace NUMINAMATH_GPT_sin_double_angle_shifted_l2368_236891

theorem sin_double_angle_shifted (θ : ℝ) (h : Real.cos (θ + Real.pi) = - 1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = - 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_shifted_l2368_236891


namespace NUMINAMATH_GPT_radius_of_inscribed_semicircle_in_isosceles_triangle_l2368_236885

theorem radius_of_inscribed_semicircle_in_isosceles_triangle
    (BC : ℝ) (h : ℝ) (r : ℝ)
    (H_eq : BC = 24)
    (H_height : h = 18)
    (H_area : 0.5 * BC * h = 0.5 * 24 * 18) :
    r = 18 / π := by
    sorry

end NUMINAMATH_GPT_radius_of_inscribed_semicircle_in_isosceles_triangle_l2368_236885


namespace NUMINAMATH_GPT_arccos_neg_one_l2368_236849

theorem arccos_neg_one : Real.arccos (-1) = Real.pi := by
  sorry

end NUMINAMATH_GPT_arccos_neg_one_l2368_236849


namespace NUMINAMATH_GPT_minimum_value_am_bn_l2368_236856

theorem minimum_value_am_bn (a b m n : ℝ) (hp_a : a > 0)
    (hp_b : b > 0) (hp_m : m > 0) (hp_n : n > 0) (ha_b : a + b = 1)
    (hm_n : m * n = 2) :
    (am + bn) * (bm + an) ≥ 3/2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_am_bn_l2368_236856


namespace NUMINAMATH_GPT_traders_gain_percentage_l2368_236878

theorem traders_gain_percentage (C : ℝ) (h : 0 < C) : 
  let cost_of_100_pens := 100 * C
  let gain := 40 * C
  let selling_price := cost_of_100_pens + gain
  let gain_percentage := (gain / cost_of_100_pens) * 100
  gain_percentage = 40 := by
  sorry

end NUMINAMATH_GPT_traders_gain_percentage_l2368_236878


namespace NUMINAMATH_GPT_alpha_value_l2368_236839

theorem alpha_value (α : ℝ) (h : (α * (α - 1) * (-1 : ℝ)^(α - 2)) = 4) : α = -4 :=
by
  sorry

end NUMINAMATH_GPT_alpha_value_l2368_236839


namespace NUMINAMATH_GPT_restocked_bags_correct_l2368_236881

def initial_stock := 55
def sold_bags := 23
def final_stock := 164

theorem restocked_bags_correct :
  (final_stock - (initial_stock - sold_bags)) = 132 :=
by
  -- The proof would go here, but we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_restocked_bags_correct_l2368_236881


namespace NUMINAMATH_GPT_gallons_per_cubic_foot_l2368_236866

theorem gallons_per_cubic_foot (mix_per_pound : ℝ) (capacity_cubic_feet : ℕ) (weight_per_gallon : ℝ)
    (price_per_tbs : ℝ) (total_cost : ℝ) (total_gallons : ℝ) :
  mix_per_pound = 1.5 →
  capacity_cubic_feet = 6 →
  weight_per_gallon = 8 →
  price_per_tbs = 0.5 →
  total_cost = 270 →
  total_gallons = total_cost / (price_per_tbs * mix_per_pound * weight_per_gallon) →
  total_gallons / capacity_cubic_feet = 7.5 :=
by
  intro h1 h2 h3 h4 h5 h6
  rw [h2, h6]
  sorry

end NUMINAMATH_GPT_gallons_per_cubic_foot_l2368_236866


namespace NUMINAMATH_GPT_inspectors_in_group_B_l2368_236859

theorem inspectors_in_group_B
  (a b : ℕ)  -- a: number of original finished products, b: daily production
  (A_inspectors := 8)  -- Number of inspectors in group A
  (total_days := 5) -- Group B inspects in 5 days
  (inspects_same_speed : (2 * a + 2 * 2 * b) * total_days/A_inspectors = (2 * a + 2 * 5 * b) * (total_days/3))
  : ∃ (B_inspectors : ℕ), B_inspectors = 12 := 
by
  sorry

end NUMINAMATH_GPT_inspectors_in_group_B_l2368_236859


namespace NUMINAMATH_GPT_order_A_C_B_l2368_236825

noncomputable def A (a b : ℝ) : ℝ := Real.log ((a + b) / 2)
noncomputable def B (a b : ℝ) : ℝ := Real.sqrt (Real.log a * Real.log b)
noncomputable def C (a b : ℝ) : ℝ := (Real.log a + Real.log b) / 2

theorem order_A_C_B (a b : ℝ) (h1 : 1 < b) (h2 : b < a) :
  A a b > C a b ∧ C a b > B a b :=
by 
  sorry

end NUMINAMATH_GPT_order_A_C_B_l2368_236825


namespace NUMINAMATH_GPT_no_natural_number_n_exists_l2368_236851

theorem no_natural_number_n_exists (n : ℕ) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 + y^2 = 2 * n * (n + 1) * (n + 2) * (n + 3) + 12 := 
sorry

end NUMINAMATH_GPT_no_natural_number_n_exists_l2368_236851


namespace NUMINAMATH_GPT_calculate_v2_using_horner_method_l2368_236843

def f (x : ℕ) : ℕ := x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1

def horner_step (x b a : ℕ) := a * x + b

def horner_eval (coeffs : List ℕ) (x : ℕ) : ℕ :=
coeffs.foldr (horner_step x) 0

theorem calculate_v2_using_horner_method :
  horner_eval [1, 5, 10, 10, 5, 1] 2 = 24 :=
by
  -- This is the theorem statement, the proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_calculate_v2_using_horner_method_l2368_236843


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2368_236836

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  1 / (1 + a) + 4 / (2 + b)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + 3 * b = 7) : 
  min_expression_value a b ≥ (13 + 4 * Real.sqrt 3) / 14 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2368_236836


namespace NUMINAMATH_GPT_solve_inequality_l2368_236898

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2368_236898


namespace NUMINAMATH_GPT_sum_series_eq_one_quarter_l2368_236815

noncomputable def series_term (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))

theorem sum_series_eq_one_quarter : 
  (∑' n, series_term (n + 1)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_one_quarter_l2368_236815


namespace NUMINAMATH_GPT_sphere_surface_area_quadruple_l2368_236806

theorem sphere_surface_area_quadruple (r : ℝ) :
  (4 * π * (2 * r)^2) = 4 * (4 * π * r^2) :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_quadruple_l2368_236806


namespace NUMINAMATH_GPT_kids_with_red_hair_l2368_236892

theorem kids_with_red_hair (total_kids : ℕ) (ratio_red ratio_blonde ratio_black : ℕ) 
  (h_ratio : ratio_red + ratio_blonde + ratio_black = 16) (h_total : total_kids = 48) :
  (total_kids / (ratio_red + ratio_blonde + ratio_black)) * ratio_red = 9 :=
by
  sorry

end NUMINAMATH_GPT_kids_with_red_hair_l2368_236892


namespace NUMINAMATH_GPT_captain_age_l2368_236863

-- Definitions
def num_team_members : ℕ := 11
def total_team_age : ℕ := 11 * 24
def total_age_remainder := 9 * (24 - 1)
def combined_age_of_captain_and_keeper := total_team_age - total_age_remainder

-- The actual proof statement
theorem captain_age (C : ℕ) (W : ℕ) 
  (hW : W = C + 5)
  (h_total_team : total_team_age = 264)
  (h_total_remainders : total_age_remainder = 207)
  (h_combined_age : combined_age_of_captain_and_keeper = 57) :
  C = 26 :=
by sorry

end NUMINAMATH_GPT_captain_age_l2368_236863


namespace NUMINAMATH_GPT_sum_of_products_nonpos_l2368_236895

theorem sum_of_products_nonpos (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 :=
sorry

end NUMINAMATH_GPT_sum_of_products_nonpos_l2368_236895


namespace NUMINAMATH_GPT_correct_options_l2368_236860

-- Definitions of conditions in Lean 
def is_isosceles (T : Triangle) : Prop := sorry -- Define isosceles triangle
def is_right_angle (T : Triangle) : Prop := sorry -- Define right-angled triangle
def similar (T₁ T₂ : Triangle) : Prop := sorry -- Define similarity of triangles
def equal_vertex_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal vertex angle
def equal_base_angle (T₁ T₂ : Triangle) : Prop := sorry -- Define equal base angle

-- Theorem statement to verify correct options (2) and (4)
theorem correct_options {T₁ T₂ : Triangle} :
  (is_right_angle T₁ ∧ is_right_angle T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) ∧ 
  (equal_vertex_angle T₁ T₂ ∧ is_isosceles T₁ ∧ is_isosceles T₂ → similar T₁ T₂) :=
sorry -- proof not required

end NUMINAMATH_GPT_correct_options_l2368_236860


namespace NUMINAMATH_GPT_correct_option_l2368_236828

theorem correct_option (x y a b : ℝ) :
  ((x + 2 * y) ^ 2 ≠ x ^ 2 + 4 * y ^ 2) ∧
  ((-2 * (a ^ 3)) ^ 2 = 4 * (a ^ 6)) ∧
  (-6 * (a ^ 2) * (b ^ 5) + a * b ^ 2 ≠ -6 * a * (b ^ 3)) ∧
  (2 * (a ^ 2) * 3 * (a ^ 3) ≠ 6 * (a ^ 6)) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l2368_236828


namespace NUMINAMATH_GPT_find_x_l2368_236824

noncomputable def solution_x (m n y : ℝ) (m_gt_3n : m > 3 * n) : ℝ :=
  (n * m) / (m + n)

theorem find_x (m n y : ℝ) (m_gt_3n : m > 3 * n) :
  let initial_acid := m * (m / 100)
  let final_volume := m + (solution_x m n y m_gt_3n) + y
  let final_acid := (m - n) / 100 * final_volume
  initial_acid = final_acid → 
  solution_x m n y m_gt_3n = (n * m) / (m + n) :=
by sorry

end NUMINAMATH_GPT_find_x_l2368_236824


namespace NUMINAMATH_GPT_compute_expression_l2368_236887

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l2368_236887


namespace NUMINAMATH_GPT_number_of_ways_to_distribute_balls_l2368_236812

theorem number_of_ways_to_distribute_balls : 
  ∃ n : ℕ, n = 81 ∧ n = 3^4 := 
by sorry

end NUMINAMATH_GPT_number_of_ways_to_distribute_balls_l2368_236812


namespace NUMINAMATH_GPT_two_b_leq_a_plus_c_l2368_236838

variable (t a b c : ℝ)

theorem two_b_leq_a_plus_c (ht : t > 1)
  (h : 2 / Real.log t / Real.log b = 1 / Real.log t / Real.log a + 1 / Real.log t / Real.log c) :
  2 * b ≤ a + c := by sorry

end NUMINAMATH_GPT_two_b_leq_a_plus_c_l2368_236838


namespace NUMINAMATH_GPT_maximize_S_n_l2368_236807

def a1 : ℚ := 5
def d : ℚ := -5 / 7

def S_n (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem maximize_S_n :
  (∃ n : ℕ, (S_n n ≥ S_n (n - 1)) ∧ (S_n n ≥ S_n (n + 1))) →
  (n = 7 ∨ n = 8) :=
sorry

end NUMINAMATH_GPT_maximize_S_n_l2368_236807


namespace NUMINAMATH_GPT_find_PF2_l2368_236867

-- Statement of the problem

def hyperbola_1 (x y: ℝ) := (x^2 / 16) - (y^2 / 20) = 1

theorem find_PF2 (x y PF1 PF2: ℝ) (a : ℝ)
    (h_hyperbola : hyperbola_1 x y)
    (h_a : a = 4) 
    (h_dist_PF1 : PF1 = 9) :
    abs (PF1 - PF2) = 2 * a → PF2 = 17 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_find_PF2_l2368_236867


namespace NUMINAMATH_GPT_power_modulus_l2368_236871

theorem power_modulus (n : ℕ) : (2 : ℕ) ^ 345 % 5 = 2 :=
by sorry

end NUMINAMATH_GPT_power_modulus_l2368_236871


namespace NUMINAMATH_GPT_compare_fx_l2368_236808

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ :=
  x^2 - b * x + c

theorem compare_fx (b c : ℝ) (x : ℝ) (h1 : ∀ x : ℝ, f (1 - x) b c = f (1 + x) b c) (h2 : f 0 b c = 3) :
  f (2^x) b c ≤ f (3^x) b c :=
by
  sorry

end NUMINAMATH_GPT_compare_fx_l2368_236808


namespace NUMINAMATH_GPT_trigonometric_identity_l2368_236846

variable (A B C a b c : ℝ)
variable (h_triangle : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum_angles : A + B + C = π)
variable (h_condition : (c / b) + (b / c) = (5 * Real.cos A) / 2)

theorem trigonometric_identity 
  (h_triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sides_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_angles_eq : A + B + C = π) 
  (h_given : (c / b) + (b / c) = (5 * Real.cos A) / 2) : 
  (Real.tan A / Real.tan B) + (Real.tan A / Real.tan C) = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2368_236846


namespace NUMINAMATH_GPT_negation_of_proposition_l2368_236852

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 = 1 → x = 1 ∨ x = -1) ↔ ∀ x : ℝ, x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1 := by
sorry

end NUMINAMATH_GPT_negation_of_proposition_l2368_236852


namespace NUMINAMATH_GPT_quadratic_solution_l2368_236832

theorem quadratic_solution :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := sorry

end NUMINAMATH_GPT_quadratic_solution_l2368_236832


namespace NUMINAMATH_GPT_largest_possible_number_of_red_socks_l2368_236886

noncomputable def max_red_socks (t : ℕ) (r : ℕ) : Prop :=
  t ≤ 1991 ∧
  ((r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = 1 / 2) ∧
  ∀ r', r' ≤ 990 → (t ≤ 1991 ∧
    ((r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = 1 / 2) → r ≤ r')

theorem largest_possible_number_of_red_socks :
  ∃ t r, max_red_socks t r ∧ r = 990 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_number_of_red_socks_l2368_236886


namespace NUMINAMATH_GPT_ratio_of_ages_l2368_236858

theorem ratio_of_ages (F C : ℕ) (h1 : F = C) (h2 : F = 75) :
  (C + 5 * 15) / (F + 15) = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ages_l2368_236858


namespace NUMINAMATH_GPT_transportation_cost_l2368_236830

theorem transportation_cost 
  (cost_per_kg : ℝ) 
  (weight_communication : ℝ) 
  (weight_sensor : ℝ) 
  (extra_sensor_cost_percentage : ℝ) 
  (cost_communication : ℝ)
  (basic_cost_sensor : ℝ)
  (extra_cost_sensor : ℝ)
  (total_cost : ℝ) : 
  cost_per_kg = 25000 → 
  weight_communication = 0.5 → 
  weight_sensor = 0.3 → 
  extra_sensor_cost_percentage = 0.10 →
  cost_communication = weight_communication * cost_per_kg →
  basic_cost_sensor = weight_sensor * cost_per_kg →
  extra_cost_sensor = extra_sensor_cost_percentage * basic_cost_sensor →
  total_cost = cost_communication + basic_cost_sensor + extra_cost_sensor →
  total_cost = 20750 :=
by sorry

end NUMINAMATH_GPT_transportation_cost_l2368_236830


namespace NUMINAMATH_GPT_cricket_team_members_l2368_236869

theorem cricket_team_members (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (team_avg_age : ℕ) 
  (remaining_avg_age : ℕ) 
  (h1 : captain_age = 26)
  (h2 : wicket_keeper_age = 29)
  (h3 : team_avg_age = 23)
  (h4 : remaining_avg_age = 22) 
  (h5 : team_avg_age * n = remaining_avg_age * (n - 2) + captain_age + wicket_keeper_age) : 
  n = 11 := 
sorry

end NUMINAMATH_GPT_cricket_team_members_l2368_236869


namespace NUMINAMATH_GPT_disinfectant_usage_l2368_236884

theorem disinfectant_usage (x : ℝ) (hx1 : 0 < x) (hx2 : 120 / x / 2 = 120 / (x + 4)) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_disinfectant_usage_l2368_236884


namespace NUMINAMATH_GPT_distribution_of_collection_items_l2368_236831

-- Declaring the collections
structure Collection where
  stickers : Nat
  baseball_cards : Nat
  keychains : Nat
  stamps : Nat

-- Defining the individual collections based on the conditions
def Karl : Collection := { stickers := 25, baseball_cards := 15, keychains := 5, stamps := 10 }
def Ryan : Collection := { stickers := Karl.stickers + 20, baseball_cards := Karl.baseball_cards - 10, keychains := Karl.keychains + 2, stamps := Karl.stamps }
def Ben_scenario1 : Collection := { stickers := Ryan.stickers - 10, baseball_cards := (Ryan.baseball_cards / 2), keychains := Karl.keychains * 2, stamps := Karl.stamps + 5 }

-- Total number of items in the collection
def total_items_scenario1 :=
  Karl.stickers + Karl.baseball_cards + Karl.keychains + Karl.stamps +
  Ryan.stickers + Ryan.baseball_cards + Ryan.keychains + Ryan.stamps +
  Ben_scenario1.stickers + Ben_scenario1.baseball_cards + Ben_scenario1.keychains + Ben_scenario1.stamps

-- The proof statement
theorem distribution_of_collection_items :
  total_items_scenario1 = 184 ∧ total_items_scenario1 % 4 = 0 → (184 / 4 = 46) := 
by
  sorry

end NUMINAMATH_GPT_distribution_of_collection_items_l2368_236831


namespace NUMINAMATH_GPT_pick_three_cards_in_order_l2368_236894

theorem pick_three_cards_in_order (deck_size : ℕ) (first_card_ways : ℕ) (second_card_ways : ℕ) (third_card_ways : ℕ) 
  (total_combinations : ℕ) (h1 : deck_size = 52) (h2 : first_card_ways = 52) 
  (h3 : second_card_ways = 51) (h4 : third_card_ways = 50) (h5 : total_combinations = first_card_ways * second_card_ways * third_card_ways) : 
  total_combinations = 132600 := 
by 
  sorry

end NUMINAMATH_GPT_pick_three_cards_in_order_l2368_236894


namespace NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l2368_236810

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = - Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l2368_236810


namespace NUMINAMATH_GPT_min_sum_of_dimensions_l2368_236897

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 2310) :
  a + b + c = 42 :=
sorry

end NUMINAMATH_GPT_min_sum_of_dimensions_l2368_236897


namespace NUMINAMATH_GPT_even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l2368_236814

open Int
open Nat

theorem even_n_square_mod_8 (n : ℤ) (h : n % 2 = 0) : (n^2 % 8 = 0) ∨ (n^2 % 8 = 4) := sorry

theorem odd_n_square_mod_8 (n : ℤ) (h : n % 2 = 1) : n^2 % 8 = 1 := sorry

theorem odd_n_fourth_mod_8 (n : ℤ) (h : n % 2 = 1) : n^4 % 8 = 1 := sorry

end NUMINAMATH_GPT_even_n_square_mod_8_odd_n_square_mod_8_odd_n_fourth_mod_8_l2368_236814


namespace NUMINAMATH_GPT_maximum_value_of_f_l2368_236835

theorem maximum_value_of_f (x : ℝ) (h : x^4 + 36 ≤ 13 * x^2) : 
  ∃ (m : ℝ), m = 18 ∧ ∀ (x : ℝ), (x^4 + 36 ≤ 13 * x^2) → (x^3 - 3 * x ≤ m) :=
sorry

end NUMINAMATH_GPT_maximum_value_of_f_l2368_236835


namespace NUMINAMATH_GPT_tommy_number_of_nickels_l2368_236899

theorem tommy_number_of_nickels
  (d p n q : ℕ)
  (h1 : d = p + 10)
  (h2 : n = 2 * d)
  (h3 : q = 4)
  (h4 : p = 10 * q) : n = 100 :=
sorry

end NUMINAMATH_GPT_tommy_number_of_nickels_l2368_236899


namespace NUMINAMATH_GPT_integer_division_condition_l2368_236872

theorem integer_division_condition (n : ℕ) (h1 : n > 1): (∃ k : ℕ, 2^n + 1 = k * n^2) → n = 3 :=
by sorry

end NUMINAMATH_GPT_integer_division_condition_l2368_236872


namespace NUMINAMATH_GPT_vector_add_sub_l2368_236809

open Matrix

section VectorProof

/-- Define the vectors a, b, and c. -/
def a : Matrix (Fin 2) (Fin 1) ℤ := ![![3], ![-6]]
def b : Matrix (Fin 2) (Fin 1) ℤ := ![![-1], ![5]]
def c : Matrix (Fin 2) (Fin 1) ℤ := ![![5], ![-20]]

/-- State the proof problem. -/
theorem vector_add_sub :
  2 • a + 4 • b - c = ![![-3], ![28]] :=
by
  sorry

end VectorProof

end NUMINAMATH_GPT_vector_add_sub_l2368_236809


namespace NUMINAMATH_GPT_divisor_is_11_l2368_236853

noncomputable def least_subtracted_divisor : Nat := 11

def problem_condition (D : Nat) (x : Nat) : Prop :=
  2000 - x = 1989 ∧ (2000 - x) % D = 0

theorem divisor_is_11 (D : Nat) (x : Nat) (h : problem_condition D x) : D = least_subtracted_divisor :=
by
  sorry

end NUMINAMATH_GPT_divisor_is_11_l2368_236853


namespace NUMINAMATH_GPT_odd_prime_does_not_divide_odd_nat_number_increment_l2368_236845

theorem odd_prime_does_not_divide_odd_nat_number_increment (p n : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) (hn_odd : n % 2 = 1) :
  ¬ (p * n + 1 ∣ p ^ p - 1) :=
by
  sorry

end NUMINAMATH_GPT_odd_prime_does_not_divide_odd_nat_number_increment_l2368_236845


namespace NUMINAMATH_GPT_find_m_b_l2368_236889

noncomputable def line_equation (x y : ℝ) :=
  (⟨-1, 4⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -5⟩ : ℝ × ℝ) = 0

theorem find_m_b : ∃ m b : ℝ, (∀ (x y : ℝ), line_equation x y → y = m * x + b) ∧ m = 1 / 4 ∧ b = -23 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_b_l2368_236889


namespace NUMINAMATH_GPT_total_pages_in_book_l2368_236876

theorem total_pages_in_book (x : ℕ) : 
  (x - (x / 6 + 8) - ((5 * x / 6 - 8) / 5 + 10) - ((4 * x / 6 - 18) / 4 + 12) = 72) → 
  x = 195 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l2368_236876


namespace NUMINAMATH_GPT_number_of_plain_lemonade_sold_l2368_236829

theorem number_of_plain_lemonade_sold
  (price_per_plain_lemonade : ℝ)
  (earnings_strawberry_lemonade : ℝ)
  (earnings_more_plain_than_strawberry : ℝ)
  (P : ℝ)
  (H1 : price_per_plain_lemonade = 0.75)
  (H2 : earnings_strawberry_lemonade = 16)
  (H3 : earnings_more_plain_than_strawberry = 11)
  (H4 : price_per_plain_lemonade * P = earnings_strawberry_lemonade + earnings_more_plain_than_strawberry) :
  P = 36 :=
by
  sorry

end NUMINAMATH_GPT_number_of_plain_lemonade_sold_l2368_236829


namespace NUMINAMATH_GPT_Sherry_catches_train_within_5_minutes_l2368_236802

-- Defining the probabilities given in the conditions
def P_A : ℝ := 0.75  -- Probability of train arriving
def P_N : ℝ := 0.75  -- Probability of Sherry not noticing the train

-- Event that no train arrives combined with event that train arrives but not noticed
def P_not_catch_in_a_minute : ℝ := 1 - P_A + P_A * P_N

-- Generalizing to 5 minutes
def P_not_catch_in_5_minutes : ℝ := P_not_catch_in_a_minute ^ 5

-- Probability Sherry catches the train within 5 minutes
def P_C : ℝ := 1 - P_not_catch_in_5_minutes

theorem Sherry_catches_train_within_5_minutes : P_C = 1 - (13 / 16) ^ 5 := by
  sorry

end NUMINAMATH_GPT_Sherry_catches_train_within_5_minutes_l2368_236802


namespace NUMINAMATH_GPT_product_of_fractions_l2368_236816

theorem product_of_fractions : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l2368_236816


namespace NUMINAMATH_GPT_choose_questions_l2368_236848

theorem choose_questions (q : ℕ) (last : ℕ) (total : ℕ) (chosen : ℕ) 
  (condition : q ≥ 3) 
  (n : last = 5) 
  (m : total = 10) 
  (k : chosen = 6) : 
  ∃ (ways : ℕ), ways = 155 := 
by
  sorry

end NUMINAMATH_GPT_choose_questions_l2368_236848
