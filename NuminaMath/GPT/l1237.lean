import Mathlib

namespace NUMINAMATH_GPT_spherical_to_cartesian_l1237_123758

theorem spherical_to_cartesian 
  (ρ θ φ : ℝ)
  (hρ : ρ = 3) 
  (hθ : θ = 7 * Real.pi / 12) 
  (hφ : φ = Real.pi / 4) :
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) = 
  (3 * Real.sqrt 2 / 2 * Real.cos (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2 * Real.sin (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_cartesian_l1237_123758


namespace NUMINAMATH_GPT_average_class_weight_l1237_123799

theorem average_class_weight
  (n_boys n_girls n_total : ℕ)
  (avg_weight_boys avg_weight_girls total_students : ℕ)
  (h1 : n_boys = 15)
  (h2 : n_girls = 10)
  (h3 : n_total = 25)
  (h4 : avg_weight_boys = 48)
  (h5 : avg_weight_girls = 405 / 10) 
  (h6 : total_students = 25) :
  (48 * 15 + 40.5 * 10) / 25 = 45 := 
sorry

end NUMINAMATH_GPT_average_class_weight_l1237_123799


namespace NUMINAMATH_GPT_problem1_l1237_123782

theorem problem1 :
  (-1 : ℤ)^2024 - (-1 : ℤ)^2023 = 2 := by
  sorry

end NUMINAMATH_GPT_problem1_l1237_123782


namespace NUMINAMATH_GPT_airport_distance_l1237_123764

theorem airport_distance (d t : ℝ) (h1 : d = 45 * (t + 0.75))
                         (h2 : d - 45 = 65 * (t - 1.25)) :
  d = 241.875 :=
by
  sorry

end NUMINAMATH_GPT_airport_distance_l1237_123764


namespace NUMINAMATH_GPT_mul_mod_eq_l1237_123714

theorem mul_mod_eq :
  (66 * 77 * 88) % 25 = 16 :=
by 
  sorry

end NUMINAMATH_GPT_mul_mod_eq_l1237_123714


namespace NUMINAMATH_GPT_find_m_f_monotonicity_l1237_123720

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / x - x ^ m

theorem find_m : ∃ (m : ℝ), f 4 m = -7 / 2 := sorry

noncomputable def g (x : ℝ) : ℝ := 2 / x - x

theorem f_monotonicity : ∀ x1 x2 : ℝ, (0 < x2 ∧ x2 < x1) → f x1 1 < f x2 1 := sorry

end NUMINAMATH_GPT_find_m_f_monotonicity_l1237_123720


namespace NUMINAMATH_GPT_intervals_of_monotonicity_l1237_123716

noncomputable def y (x : ℝ) : ℝ := 2 ^ (x^2 - 2*x + 4)

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x > 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ < y x₂)) ∧
  (∀ x : ℝ, x < 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ > y x₂)) :=
by
  sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_l1237_123716


namespace NUMINAMATH_GPT_B_2_2_eq_16_l1237_123723

def B : ℕ → ℕ → ℕ
| 0, n       => n + 2
| (m+1), 0   => B m 2
| (m+1), (n+1) => B m (B (m+1) n)

theorem B_2_2_eq_16 : B 2 2 = 16 := by
  sorry

end NUMINAMATH_GPT_B_2_2_eq_16_l1237_123723


namespace NUMINAMATH_GPT_josiah_total_expenditure_l1237_123730

noncomputable def cookies_per_day := 2
noncomputable def cost_per_cookie := 16
noncomputable def days_in_march := 31

theorem josiah_total_expenditure :
  (cookies_per_day * days_in_march * cost_per_cookie) = 992 :=
by sorry

end NUMINAMATH_GPT_josiah_total_expenditure_l1237_123730


namespace NUMINAMATH_GPT_problem_l1237_123743

-- Define the polynomial g(x) with given coefficients
def g (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x^2 + x + 8

-- Define the polynomial f(x) with given coefficients
def f (x : ℝ) (a b c : ℝ) : ℝ :=
  x^4 + x^3 + b * x^2 + 50 * x + c

-- Define the conditions
def conditions (a b c r : ℝ) : Prop :=
  ∃ roots : Finset ℝ, (∀ x ∈ roots, g x a = 0) ∧ (∀ x ∈ roots, f x a b c = 0) ∧ (roots.card = 3) ∧
  (8 - r = 50) ∧ (a - r = 1) ∧ (1 - a * r = b) ∧ (-8 * r = c)

-- Define the theorem to be proved
theorem problem (a b c r : ℝ) (h : conditions a b c r) : f 1 a b c = -1333 :=
by sorry

end NUMINAMATH_GPT_problem_l1237_123743


namespace NUMINAMATH_GPT_algebraic_expression_value_l1237_123753

theorem algebraic_expression_value (b a c : ℝ) (h₁ : b < a) (h₂ : a < 0) (h₃ : 0 < c) :
  |b| - |b - a| + |c - a| - |a + b| = b + c - a :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1237_123753


namespace NUMINAMATH_GPT_proof_max_difference_l1237_123728

/-- Digits as displayed on the engineering calculator -/
structure Digits :=
  (a b c d e f g h i : ℕ)

-- Possible digits based on broken displays
axiom a_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom b_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom c_values : {x // x = 3 ∨ x = 4 ∨ x = 8 ∨ x = 9}
axiom d_values : {x // x = 2 ∨ x = 3 ∨ x = 7}
axiom e_values : {x // x = 3 ∨ x = 5 ∨ x = 9}
axiom f_values : {x // x = 1 ∨ x = 4 ∨ x = 7}
axiom g_values : {x // x = 4 ∨ x = 5 ∨ x = 9}
axiom h_values : {x // x = 2}
axiom i_values : {x // x = 4 ∨ x = 5 ∨ x = 9}

-- Minuend and subtrahend values
def minuend := 923
def subtrahend := 394

-- Maximum possible value of the difference
def max_difference := 529

theorem proof_max_difference : 
  ∃ (digits : Digits),
    digits.a = 9 ∧ digits.b = 2 ∧ digits.c = 3 ∧
    digits.d = 3 ∧ digits.e = 9 ∧ digits.f = 4 ∧
    digits.g = 5 ∧ digits.h = 2 ∧ digits.i = 9 ∧
    minuend - subtrahend = max_difference :=
by
  sorry

end NUMINAMATH_GPT_proof_max_difference_l1237_123728


namespace NUMINAMATH_GPT_gcd_111_148_l1237_123791

theorem gcd_111_148 : Nat.gcd 111 148 = 37 :=
by
  sorry

end NUMINAMATH_GPT_gcd_111_148_l1237_123791


namespace NUMINAMATH_GPT_banana_distinct_arrangements_l1237_123726

theorem banana_distinct_arrangements :
  let n := 6
  let f_B := 1
  let f_N := 2
  let f_A := 3
  (n.factorial) / (f_B.factorial * f_N.factorial * f_A.factorial) = 60 := by
sorry

end NUMINAMATH_GPT_banana_distinct_arrangements_l1237_123726


namespace NUMINAMATH_GPT_max_sum_arithmetic_sequence_l1237_123736

theorem max_sum_arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ) (h1 : (a + 2) ^ 2 = (a + 8) * (a - 2))
  (h2 : ∀ k, S k = (k * (2 * a + (k - 1) * d)) / 2)
  (h3 : 10 = a) (h4 : -2 = d) :
  S 10 = 90 :=
sorry

end NUMINAMATH_GPT_max_sum_arithmetic_sequence_l1237_123736


namespace NUMINAMATH_GPT_max_loaves_given_l1237_123772

variables {a1 d : ℕ}

-- Mathematical statement: The conditions given in the problem
def arith_sequence_correct (a1 d : ℕ) : Prop :=
  (5 * a1 + 10 * d = 60) ∧ (2 * a1 + 7 * d = 3 * a1 + 3 * d)

-- Lean theorem statement
theorem max_loaves_given (a1 d : ℕ) (h : arith_sequence_correct a1 d) : a1 + 4 * d = 16 :=
sorry

end NUMINAMATH_GPT_max_loaves_given_l1237_123772


namespace NUMINAMATH_GPT_minimum_value_of_a_l1237_123752

theorem minimum_value_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 / 2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l1237_123752


namespace NUMINAMATH_GPT_total_steps_five_days_l1237_123719

def steps_monday : ℕ := 150 + 170
def steps_tuesday : ℕ := 140 + 170
def steps_wednesday : ℕ := 160 + 210 + 25
def steps_thursday : ℕ := 150 + 140 + 30 + 15
def steps_friday : ℕ := 180 + 200 + 20

theorem total_steps_five_days :
  steps_monday + steps_tuesday + steps_wednesday + steps_thursday + steps_friday = 1760 :=
by
  have h1 : steps_monday = 320 := rfl
  have h2 : steps_tuesday = 310 := rfl
  have h3 : steps_wednesday = 395 := rfl
  have h4 : steps_thursday = 335 := rfl
  have h5 : steps_friday = 400 := rfl
  show 320 + 310 + 395 + 335 + 400 = 1760
  sorry

end NUMINAMATH_GPT_total_steps_five_days_l1237_123719


namespace NUMINAMATH_GPT_xy_sum_equal_two_or_minus_two_l1237_123744

/-- 
Given the conditions |x| = 3, |y| = 5, and xy < 0, prove that x + y = 2 or x + y = -2. 
-/
theorem xy_sum_equal_two_or_minus_two (x y : ℝ) (hx : |x| = 3) (hy : |y| = 5) (hxy : x * y < 0) : x + y = 2 ∨ x + y = -2 := 
  sorry

end NUMINAMATH_GPT_xy_sum_equal_two_or_minus_two_l1237_123744


namespace NUMINAMATH_GPT_percentage_in_first_subject_l1237_123788

theorem percentage_in_first_subject (P : ℝ) (H1 : 80 = 80) (H2 : 75 = 75) (H3 : (P + 80 + 75) / 3 = 75) : P = 70 :=
by
  sorry

end NUMINAMATH_GPT_percentage_in_first_subject_l1237_123788


namespace NUMINAMATH_GPT_combined_money_half_l1237_123798

theorem combined_money_half
  (J S : ℚ)
  (h1 : J = S)
  (h2 : J - (3/7 * J + 2/5 * J + 1/4 * J) = 24)
  (h3 : S - (1/2 * S + 1/3 * S) = 36) :
  1.5 * J = 458.18 := 
by
  sorry

end NUMINAMATH_GPT_combined_money_half_l1237_123798


namespace NUMINAMATH_GPT_solution_to_problem_l1237_123781

theorem solution_to_problem (a x y n m : ℕ) (h1 : a * (x^n - x^m) = (a * x^m - 4) * y^2)
  (h2 : m % 2 = n % 2) (h3 : (a * x) % 2 = 1) : 
  x = 1 :=
sorry

end NUMINAMATH_GPT_solution_to_problem_l1237_123781


namespace NUMINAMATH_GPT_balloons_problem_l1237_123760

variable (b_J b_S b_J_f b_g : ℕ)

theorem balloons_problem
  (h1 : b_J = 9)
  (h2 : b_S = 5)
  (h3 : b_J_f = 12)
  (h4 : b_g = (b_J + b_S) - b_J_f)
  : b_g = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_balloons_problem_l1237_123760


namespace NUMINAMATH_GPT_eval_expr_eq_zero_l1237_123732

def ceiling_floor_sum (x : ℚ) : ℤ :=
  Int.ceil (x) + Int.floor (-x)

theorem eval_expr_eq_zero : ceiling_floor_sum (7/3) = 0 := by
  sorry

end NUMINAMATH_GPT_eval_expr_eq_zero_l1237_123732


namespace NUMINAMATH_GPT_perimeter_of_rectangle_l1237_123707

theorem perimeter_of_rectangle (b l : ℝ) (h1 : l = 3 * b) (h2 : b * l = 75) : 2 * l + 2 * b = 40 := 
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_l1237_123707


namespace NUMINAMATH_GPT_weight_of_daughter_l1237_123733

variable (M D G S : ℝ)

theorem weight_of_daughter :
  M + D + G + S = 200 →
  D + G = 60 →
  G = M / 5 →
  S = 2 * D →
  D = 800 / 15 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_weight_of_daughter_l1237_123733


namespace NUMINAMATH_GPT_stratified_sampling_correct_l1237_123739

-- Define the total number of students and the ratio of students in grades 10, 11, and 12
def total_students : ℕ := 4000
def ratio_grade10 : ℕ := 32
def ratio_grade11 : ℕ := 33
def ratio_grade12 : ℕ := 35

-- The total sample size
def sample_size : ℕ := 200

-- Define the expected numbers of students drawn from each grade in the sample
def sample_grade10 : ℕ := 64
def sample_grade11 : ℕ := 66
def sample_grade12 : ℕ := 70

-- The theorem to be proved
theorem stratified_sampling_correct :
  (sample_grade10 + sample_grade11 + sample_grade12 = sample_size) ∧
  (sample_grade10 = (ratio_grade10 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade11 = (ratio_grade11 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade12 = (ratio_grade12 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l1237_123739


namespace NUMINAMATH_GPT_vector_MN_l1237_123790

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

theorem vector_MN :
  vector_sub N M = (-2, -4) :=
by
  sorry

end NUMINAMATH_GPT_vector_MN_l1237_123790


namespace NUMINAMATH_GPT_population_factor_proof_l1237_123712

-- Define the conditions given in the problem
variables (N x y z : ℕ)

theorem population_factor_proof :
  (N = x^2) ∧ (N + 100 = y^2 + 1) ∧ (N + 200 = z^2) → (7 ∣ N) :=
by sorry

end NUMINAMATH_GPT_population_factor_proof_l1237_123712


namespace NUMINAMATH_GPT_cubical_box_edge_length_l1237_123766

noncomputable def edge_length_of_box_in_meters : ℝ :=
  let number_of_cubes := 999.9999999999998
  let edge_length_cube_cm := 10
  let volume_cube_cm := edge_length_cube_cm^3
  let total_volume_box_cm := volume_cube_cm * number_of_cubes
  let total_volume_box_meters := total_volume_box_cm / (100^3)
  (total_volume_box_meters)^(1/3)

theorem cubical_box_edge_length :
  edge_length_of_box_in_meters = 1 := 
sorry

end NUMINAMATH_GPT_cubical_box_edge_length_l1237_123766


namespace NUMINAMATH_GPT_find_starting_number_of_range_l1237_123734

theorem find_starting_number_of_range : 
  ∃ (n : ℤ), 
    (∀ k, (0 ≤ k ∧ k < 7) → (n + k * 3 ≤ 31 ∧ n + k * 3 % 3 = 0)) ∧ 
    n + 6 * 3 = 30 - 6 * 3 :=
by
  sorry

end NUMINAMATH_GPT_find_starting_number_of_range_l1237_123734


namespace NUMINAMATH_GPT_num_sets_B_l1237_123706

open Set

def A : Set ℕ := {1, 3}

theorem num_sets_B :
  ∃ (B : ℕ → Set ℕ), (∀ b, B b ∪ A = {1, 3, 5}) ∧ (∃ s t u v, B s = {5} ∧
                                                   B t = {1, 5} ∧
                                                   B u = {3, 5} ∧
                                                   B v = {1, 3, 5} ∧ 
                                                   s ≠ t ∧ s ≠ u ∧ s ≠ v ∧
                                                   t ≠ u ∧ t ≠ v ∧
                                                   u ≠ v) :=
sorry

end NUMINAMATH_GPT_num_sets_B_l1237_123706


namespace NUMINAMATH_GPT_athena_total_spent_l1237_123735

def cost_of_sandwiches (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) : ℝ :=
  num_sandwiches * cost_per_sandwich

def cost_of_drinks (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  num_drinks * cost_per_drink

def total_cost (num_sandwiches : ℕ) (cost_per_sandwich : ℝ) (num_drinks : ℕ) (cost_per_drink : ℝ) : ℝ :=
  cost_of_sandwiches num_sandwiches cost_per_sandwich + cost_of_drinks num_drinks cost_per_drink

theorem athena_total_spent :
  total_cost 3 3 2 2.5 = 14 :=
by 
  sorry

end NUMINAMATH_GPT_athena_total_spent_l1237_123735


namespace NUMINAMATH_GPT_evaluation_expression_l1237_123759

theorem evaluation_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 :=
by
  rw [h1, h2]
  -- Here we would perform the arithmetic steps to show the equality
  sorry

end NUMINAMATH_GPT_evaluation_expression_l1237_123759


namespace NUMINAMATH_GPT_probability_jqka_is_correct_l1237_123701

noncomputable def probability_sequence_is_jqka : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49)

theorem probability_jqka_is_correct :
  probability_sequence_is_jqka = (16 / 4048375) :=
by
  sorry

end NUMINAMATH_GPT_probability_jqka_is_correct_l1237_123701


namespace NUMINAMATH_GPT_relationship_between_u_and_v_l1237_123784

variables {r u v p : ℝ}
variables (AB G : ℝ)

theorem relationship_between_u_and_v (hAB : AB = 2 * r) (hAG_GF : u = (p^2 / (2 * r)) - p) :
    v^2 = u^3 / (2 * r - u) :=
sorry

end NUMINAMATH_GPT_relationship_between_u_and_v_l1237_123784


namespace NUMINAMATH_GPT_average_income_of_all_customers_l1237_123774

theorem average_income_of_all_customers
  (n m : ℕ) 
  (a b : ℝ) 
  (customers_responded : n = 50) 
  (wealthiest_count : m = 10) 
  (other_customers_count : n - m = 40) 
  (wealthiest_avg_income : a = 55000) 
  (other_avg_income : b = 42500) : 
  (m * a + (n - m) * b) / n = 45000 := 
by
  -- transforming given conditions into useful expressions
  have h1 : m = 10 := by assumption
  have h2 : n = 50 := by assumption
  have h3 : n - m = 40 := by assumption
  have h4 : a = 55000 := by assumption
  have h5 : b = 42500 := by assumption
  sorry

end NUMINAMATH_GPT_average_income_of_all_customers_l1237_123774


namespace NUMINAMATH_GPT_max_value_inequality_l1237_123742

theorem max_value_inequality (a x₁ x₂ : ℝ) (h_a : a < 0)
  (h_sol : ∀ x, x^2 - 4 * a * x + 3 * a^2 < 0 ↔ x₁ < x ∧ x < x₂) :
    x₁ + x₂ + a / (x₁ * x₂) ≤ - 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_GPT_max_value_inequality_l1237_123742


namespace NUMINAMATH_GPT_goldbach_conjecture_2024_l1237_123776

-- Definitions for the problem
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement for the proof problem
theorem goldbach_conjecture_2024 :
  is_even 2024 ∧ 2024 > 2 → ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 2024 = p1 + p2 :=
by
  sorry

end NUMINAMATH_GPT_goldbach_conjecture_2024_l1237_123776


namespace NUMINAMATH_GPT_total_savings_l1237_123783

-- Definition to specify the denomination of each bill
def bill_value : ℕ := 100

-- Condition: Number of $100 bills Michelle has
def num_bills : ℕ := 8

-- The theorem to prove the total savings amount
theorem total_savings : num_bills * bill_value = 800 :=
by
  sorry

end NUMINAMATH_GPT_total_savings_l1237_123783


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1237_123738

theorem eccentricity_of_ellipse (a c : ℝ) (h1 : 2 * c = a) : (c / a) = (1 / 2) :=
by
  -- This is where we would write the proof, but we're using sorry to skip the proof steps.
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1237_123738


namespace NUMINAMATH_GPT_find_symmetric_point_l1237_123721

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def plane (x y z : ℝ) : ℝ := 
  4 * x + 6 * y + 4 * z - 25

def symmetric_point (M M_prime : Point3D) (plane_eq : ℝ → ℝ → ℝ → ℝ) : Prop :=
  let t : ℝ := (1 / 4)
  let M0 : Point3D := { x := (1 + 4 * t), y := (6 * t), z := (1 + 4 * t) }
  let midpoint_x := (M.x + M_prime.x) / 2
  let midpoint_y := (M.y + M_prime.y) / 2
  let midpoint_z := (M.z + M_prime.z) / 2
  M0.x = midpoint_x ∧ M0.y = midpoint_y ∧ M0.z = midpoint_z ∧
  plane_eq M0.x M0.y M0.z = 0

def M : Point3D := { x := 1, y := 0, z := 1 }

def M_prime : Point3D := { x := 3, y := 3, z := 3 }

theorem find_symmetric_point : symmetric_point M M_prime plane := by
  -- the proof is omitted here
  sorry

end NUMINAMATH_GPT_find_symmetric_point_l1237_123721


namespace NUMINAMATH_GPT_line_circle_chord_shortest_l1237_123761

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_circle_chord_shortest (m : ℝ) :
  (∀ x y : ℝ, circle_C x y → line_l x y m → m = -3 / 4) :=
sorry

end NUMINAMATH_GPT_line_circle_chord_shortest_l1237_123761


namespace NUMINAMATH_GPT_pos_int_solutions_l1237_123717

theorem pos_int_solutions (x : ℤ) : (3 * x - 4 < 2 * x) → (0 < x) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro h1 h2
  have h3 : x - 4 < 0 := by sorry  -- Step derived from inequality simplification
  have h4 : x < 4 := by sorry     -- Adding 4 to both sides
  sorry                           -- Combine conditions to get the specific solutions

end NUMINAMATH_GPT_pos_int_solutions_l1237_123717


namespace NUMINAMATH_GPT_quadratic_has_two_roots_l1237_123746

theorem quadratic_has_two_roots (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : 5 * a + b + 2 * c = 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 := 
  sorry

end NUMINAMATH_GPT_quadratic_has_two_roots_l1237_123746


namespace NUMINAMATH_GPT_radius_of_circle_l1237_123727

-- Define the problem condition
def diameter_of_circle : ℕ := 14

-- State the problem as a theorem
theorem radius_of_circle (d : ℕ) (hd : d = diameter_of_circle) : d / 2 = 7 := by 
  sorry

end NUMINAMATH_GPT_radius_of_circle_l1237_123727


namespace NUMINAMATH_GPT_perfect_square_value_of_b_l1237_123748

theorem perfect_square_value_of_b :
  (∃ b : ℝ, (11.98 * 11.98 + 11.98 * 0.04 + b * b) = (11.98 + b)^2) →
  (∃ b : ℝ, b = 0.02) :=
sorry

end NUMINAMATH_GPT_perfect_square_value_of_b_l1237_123748


namespace NUMINAMATH_GPT_area_of_triangle_DEF_eq_480_l1237_123792

theorem area_of_triangle_DEF_eq_480 (DE EF DF : ℝ) (h1 : DE = 20) (h2 : EF = 48) (h3 : DF = 52) :
  let s := (DE + EF + DF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF))
  area = 480 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_DEF_eq_480_l1237_123792


namespace NUMINAMATH_GPT_min_distinct_sums_max_distinct_sums_l1237_123794

theorem min_distinct_sums (n : ℕ) (h : 0 < n) : ∃ a b, (a + (n - 1) * b) = (n * (n + 1)) / 2 := sorry

theorem max_distinct_sums (n : ℕ) (h : 0 < n) : 
  ∃ m, m = 2^n - 1 := sorry

end NUMINAMATH_GPT_min_distinct_sums_max_distinct_sums_l1237_123794


namespace NUMINAMATH_GPT_triangle_property_l1237_123711

theorem triangle_property
  (A B C : ℝ)
  (a b c : ℝ)
  (R : ℝ)
  (hR : R = Real.sqrt 3)
  (h1 : a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0)
  (h2 : b + c = Real.sqrt 11)
  (htri : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R):
  a = 3 ∧ (1 / 2 * b * c * Real.sin A = Real.sqrt 3 / 2) := 
sorry

end NUMINAMATH_GPT_triangle_property_l1237_123711


namespace NUMINAMATH_GPT_max_people_transition_l1237_123710

theorem max_people_transition (a : ℕ) (b : ℕ) (c : ℕ) 
  (hA : a = 850 * 6 / 100) (hB : b = 1500 * 42 / 1000) (hC : c = 4536 / 72) :
  max a (max b c) = 63 := 
sorry

end NUMINAMATH_GPT_max_people_transition_l1237_123710


namespace NUMINAMATH_GPT_correct_fraction_order_l1237_123745

noncomputable def fraction_ordering : Prop := 
  (16 / 12 < 18 / 13) ∧ (18 / 13 < 21 / 14) ∧ (21 / 14 < 20 / 15)

theorem correct_fraction_order : fraction_ordering := 
by {
  repeat { sorry }
}

end NUMINAMATH_GPT_correct_fraction_order_l1237_123745


namespace NUMINAMATH_GPT_additional_distance_l1237_123725

theorem additional_distance (distance_speed_10 : ℝ) (speed1 speed2 time1 time2 distance actual_distance additional_distance : ℝ)
  (h1 : actual_distance = distance_speed_10)
  (h2 : time1 = distance_speed_10 / speed1)
  (h3 : time1 = 5)
  (h4 : speed1 = 10)
  (h5 : time2 = actual_distance / speed2)
  (h6 : speed2 = 14)
  (h7 : distance = speed2 * time1)
  (h8 : distance = 70)
  : additional_distance = distance - actual_distance
  := by
  sorry

end NUMINAMATH_GPT_additional_distance_l1237_123725


namespace NUMINAMATH_GPT_problem_l1237_123715

open Set

/-- Declaration of the universal set and other sets as per the problem -/
def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}

/-- Problem: Prove that M ∪ (U \ N) = {0, 2, 4, 6, 8} -/
theorem problem : M ∪ (U \ N) = {0, 2, 4, 6, 8} := 
by
  sorry

end NUMINAMATH_GPT_problem_l1237_123715


namespace NUMINAMATH_GPT_find_common_difference_l1237_123705

variable {a_n : ℕ → ℕ}
variable {d : ℕ}

-- Conditions
def first_term (a_n : ℕ → ℕ) := a_n 1 = 1
def common_difference (d : ℕ) := d ≠ 0
def arithmetic_def (a_n : ℕ → ℕ) (d : ℕ) := ∀ n, a_n (n+1) = a_n n + d
def geom_mean_condition (a_n : ℕ → ℕ) := a_n 2 ^ 2 = a_n 1 * a_n 4

-- Proof statement
theorem find_common_difference
  (fa : first_term a_n)
  (cd : common_difference d)
  (ad : arithmetic_def a_n d)
  (gmc : geom_mean_condition a_n) :
  d = 1 := by
  sorry

end NUMINAMATH_GPT_find_common_difference_l1237_123705


namespace NUMINAMATH_GPT_closest_fraction_to_team_aus_medals_l1237_123797

theorem closest_fraction_to_team_aus_medals 
  (won_medals : ℕ) (total_medals : ℕ) 
  (choices : List ℚ)
  (fraction_won : ℚ)
  (c1 : won_medals = 28)
  (c2 : total_medals = 150)
  (c3 : choices = [1/4, 1/5, 1/6, 1/7, 1/8])
  (c4 : fraction_won = 28 / 150) :
  abs (fraction_won - 1/5) < abs (fraction_won - 1/4) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/6) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/7) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/8) := 
sorry

end NUMINAMATH_GPT_closest_fraction_to_team_aus_medals_l1237_123797


namespace NUMINAMATH_GPT_church_path_count_is_321_l1237_123718

/-- A person starts at the bottom-left corner of an m x n grid and can only move north, east, or 
    northeast. Prove that the number of distinct paths to the top-right corner is 321 
    for a specific grid size (abstracted parameters included). -/
def distinct_paths_to_church (m n : ℕ) : ℕ :=
  let rec P : ℕ → ℕ → ℕ
    | 0, 0 => 1
    | i + 1, 0 => 1
    | 0, j + 1 => 1
    | i + 1, j + 1 => P i (j + 1) + P (i + 1) j + P i j
  P m n

theorem church_path_count_is_321 : distinct_paths_to_church m n = 321 :=
sorry

end NUMINAMATH_GPT_church_path_count_is_321_l1237_123718


namespace NUMINAMATH_GPT_distinct_meals_count_l1237_123737

def entries : ℕ := 3
def drinks : ℕ := 3
def desserts : ℕ := 3

theorem distinct_meals_count : entries * drinks * desserts = 27 :=
by
  -- sorry for skipping the proof
  sorry

end NUMINAMATH_GPT_distinct_meals_count_l1237_123737


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1237_123778

-- Define what it means for a line to be perpendicular to a plane
def line_perpendicular_to_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- Define what it means for a line to be perpendicular to countless lines in a plane
def line_perpendicular_to_countless_lines_in_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- The formal statement
theorem sufficient_but_not_necessary (l : Type) (alpha : Type) :
  (line_perpendicular_to_plane l alpha) → (line_perpendicular_to_countless_lines_in_plane l alpha) ∧ 
  ¬ ((line_perpendicular_to_countless_lines_in_plane l alpha) → (line_perpendicular_to_plane l alpha)) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1237_123778


namespace NUMINAMATH_GPT_units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l1237_123787

/-- Find the units digit of the largest power of 2 that divides into (2^5)! -/
theorem units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial : ∃ d : ℕ, d = 8 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l1237_123787


namespace NUMINAMATH_GPT_range_of_a_plus_b_l1237_123722

variable (a b : ℝ)
variable (pos_a : 0 < a)
variable (pos_b : 0 < b)
variable (h : a + b + 1/a + 1/b = 5)

theorem range_of_a_plus_b : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l1237_123722


namespace NUMINAMATH_GPT_ratio_square_pentagon_l1237_123754

theorem ratio_square_pentagon (P_sq P_pent : ℕ) 
  (h_sq : P_sq = 60) (h_pent : P_pent = 60) :
  (P_sq / 4) / (P_pent / 5) = 5 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_square_pentagon_l1237_123754


namespace NUMINAMATH_GPT_m_range_l1237_123708

open Real

-- Define the points
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (2, -1)

-- Define the line equation
def line_eq (x y m : ℝ) : Prop := x - 2*y + m = 0

-- Theorem: m must belong to the interval [-4, 5]
theorem m_range (m : ℝ) : (line_eq A.1 A.2 m) → (line_eq B.1 B.2 m) → -4 ≤ m ∧ m ≤ 5 := 
sorry

end NUMINAMATH_GPT_m_range_l1237_123708


namespace NUMINAMATH_GPT_domain_of_function_l1237_123713

theorem domain_of_function :
  { x : ℝ | -2 ≤ x ∧ x < 4 } = { x : ℝ | (x + 2 ≥ 0) ∧ (4 - x > 0) } :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1237_123713


namespace NUMINAMATH_GPT_total_tickets_l1237_123773

theorem total_tickets (n_friends : ℕ) (tickets_per_friend : ℕ) (h1 : n_friends = 6) (h2 : tickets_per_friend = 39) : n_friends * tickets_per_friend = 234 :=
by
  -- Place for proof, to be constructed
  sorry

end NUMINAMATH_GPT_total_tickets_l1237_123773


namespace NUMINAMATH_GPT_math_marks_is_95_l1237_123796

-- Define the conditions as Lean assumptions
variables (english_marks math_marks physics_marks chemistry_marks biology_marks : ℝ)
variable (average_marks : ℝ)
variable (num_subjects : ℝ)

-- State the conditions
axiom h1 : english_marks = 96
axiom h2 : physics_marks = 82
axiom h3 : chemistry_marks = 97
axiom h4 : biology_marks = 95
axiom h5 : average_marks = 93
axiom h6 : num_subjects = 5

-- Formalize the problem: Prove that math_marks = 95
theorem math_marks_is_95 : math_marks = 95 :=
by
  sorry

end NUMINAMATH_GPT_math_marks_is_95_l1237_123796


namespace NUMINAMATH_GPT_arnold_total_protein_l1237_123795

-- Conditions
def protein_in_collagen_powder (scoops: ℕ) : ℕ := 9 * scoops
def protein_in_protein_powder (scoops: ℕ) : ℕ := 21 * scoops
def protein_in_steak : ℕ := 56
def protein_in_greek_yogurt : ℕ := 15
def protein_in_almonds (cups: ℕ) : ℕ := 6 * (cups * 4) / 4
def half_cup_almonds_protein : ℕ := 12

-- Statement
theorem arnold_total_protein : 
  protein_in_collagen_powder 1 + protein_in_protein_powder 2 + protein_in_steak + protein_in_greek_yogurt + half_cup_almonds_protein = 134 :=
  by
    sorry

end NUMINAMATH_GPT_arnold_total_protein_l1237_123795


namespace NUMINAMATH_GPT_filter_replacement_month_l1237_123762

theorem filter_replacement_month (n : ℕ) (h : n = 25) : (7 * (n - 1)) % 12 = 0 → "January" = "January" :=
by
  intros
  sorry

end NUMINAMATH_GPT_filter_replacement_month_l1237_123762


namespace NUMINAMATH_GPT_M_plus_2N_equals_330_l1237_123785

theorem M_plus_2N_equals_330 (M N : ℕ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end NUMINAMATH_GPT_M_plus_2N_equals_330_l1237_123785


namespace NUMINAMATH_GPT_right_triangle_exists_and_r_inscribed_circle_l1237_123749

theorem right_triangle_exists_and_r_inscribed_circle (d : ℝ) (hd : d > 0) :
  ∃ (a b c : ℝ), 
    a < b ∧ 
    a^2 + b^2 = c^2 ∧
    b = a + d ∧ 
    c = b + d ∧ 
    (a + b - c) / 2 = d :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_exists_and_r_inscribed_circle_l1237_123749


namespace NUMINAMATH_GPT_find_w_value_l1237_123780

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_w_value
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : sqrt x / sqrt y - sqrt y / sqrt x = 7 / 12)
  (h2 : x - y = 7) :
  x + y = 25 := 
by
  sorry

end NUMINAMATH_GPT_find_w_value_l1237_123780


namespace NUMINAMATH_GPT_quadratic_bounds_l1237_123789

variable (a b c: ℝ)

-- Conditions
def quadratic_function (x: ℝ) : ℝ := a * x^2 + b * x + c

def within_range_neg_1_to_1 (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7

-- Main statement
theorem quadratic_bounds
  (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7 := sorry

end NUMINAMATH_GPT_quadratic_bounds_l1237_123789


namespace NUMINAMATH_GPT_minimum_handshakes_l1237_123777

def binom (n k : ℕ) : ℕ := n.choose k

theorem minimum_handshakes (n_A n_B k_A k_B : ℕ) (h1 : binom (n_A + n_B) 2 + n_A + n_B = 465)
  (h2 : n_A < n_B) (h3 : k_A = n_A) (h4 : k_B = n_B) : k_A = 15 :=
by sorry

end NUMINAMATH_GPT_minimum_handshakes_l1237_123777


namespace NUMINAMATH_GPT_min_jumps_required_to_visit_all_points_and_return_l1237_123763

theorem min_jumps_required_to_visit_all_points_and_return :
  ∀ (n : ℕ), n = 2016 →
  ∀ jumps : ℕ → ℕ, (∀ i, jumps i = 2 ∨ jumps i = 3) →
  (∀ i, (jumps (i + 1) + jumps (i + 2)) % n = 0) →
  ∃ (min_jumps : ℕ), min_jumps = 2017 :=
by
  sorry

end NUMINAMATH_GPT_min_jumps_required_to_visit_all_points_and_return_l1237_123763


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l1237_123793

variables (a b c : ℝ)

theorem volume_of_rectangular_prism 
  (h1 : a * b = 12) 
  (h2 : b * c = 18) 
  (h3 : c * a = 9) 
  (h4 : (1 / a) * (1 / b) * (1 / c) = (1 / 216)) :
  a * b * c = 216 :=
sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l1237_123793


namespace NUMINAMATH_GPT_evaluate_expression_l1237_123751

theorem evaluate_expression : (5 + 2) + (8 + 6) + (4 + 7) + (3 + 2) = 37 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l1237_123751


namespace NUMINAMATH_GPT_average_probable_weight_l1237_123724

-- Definitions based on the conditions
def ArunOpinion (w : ℝ) : Prop := 65 < w ∧ w < 72
def BrotherOpinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def MotherOpinion (w : ℝ) : Prop := w ≤ 68

-- The actual statement we want to prove
theorem average_probable_weight : 
  (∀ (w : ℝ), ArunOpinion w → BrotherOpinion w → MotherOpinion w → 65 < w ∧ w ≤ 68) →
  (65 + 68) / 2 = 66.5 :=
by 
  intros h1
  sorry

end NUMINAMATH_GPT_average_probable_weight_l1237_123724


namespace NUMINAMATH_GPT_donation_student_amount_l1237_123750

theorem donation_student_amount (a : ℕ) : 
  let total_amount := 3150
  let teachers_count := 5
  let donation_teachers := teachers_count * a 
  let donation_students := total_amount - donation_teachers
  donation_students = 3150 - 5 * a :=
by
  sorry

end NUMINAMATH_GPT_donation_student_amount_l1237_123750


namespace NUMINAMATH_GPT_velocity_equal_distance_l1237_123765

theorem velocity_equal_distance (v t : ℝ) (h : v * t = t) (ht : t ≠ 0) : v = 1 :=
by sorry

end NUMINAMATH_GPT_velocity_equal_distance_l1237_123765


namespace NUMINAMATH_GPT_hyperbola_focus_exists_l1237_123767

-- Define the basic premises of the problem
def is_hyperbola (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 4 = 0

-- Define a condition for the focusing property of the hyperbola.
def is_focus (x y : ℝ) : Prop :=
  (x = -2) ∧ (y = 4 + (10 * Real.sqrt 3 / 3))

-- The theorem to be proved
theorem hyperbola_focus_exists : ∃ x y : ℝ, is_hyperbola x y ∧ is_focus x y :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_hyperbola_focus_exists_l1237_123767


namespace NUMINAMATH_GPT_misha_second_attempt_points_l1237_123770

/--
Misha made a homemade dartboard at his summer cottage. The round board is 
divided into several sectors by circles, and you can throw darts at it. 
Points are awarded based on the sector hit.

Misha threw 8 darts three times. In his second attempt, he scored twice 
as many points as in his first attempt, and in his third attempt, he scored 
1.5 times more points than in his second attempt. How many points did he 
score in his second attempt?
-/
theorem misha_second_attempt_points:
  ∀ (x : ℕ), 
  (x ≥ 24) →
  (2 * x ≥ 48) →
  (3 * x = 72) →
  (2 * x = 48) :=
by
  intros x h1 h2 h3
  sorry

end NUMINAMATH_GPT_misha_second_attempt_points_l1237_123770


namespace NUMINAMATH_GPT_total_eggs_l1237_123755

theorem total_eggs (students : ℕ) (eggs_per_student : ℕ) (h1 : students = 7) (h2 : eggs_per_student = 8) :
  students * eggs_per_student = 56 :=
by
  sorry

end NUMINAMATH_GPT_total_eggs_l1237_123755


namespace NUMINAMATH_GPT_new_socks_bought_l1237_123731

theorem new_socks_bought :
  ∀ (original_socks throw_away new_socks total_socks : ℕ),
    original_socks = 28 →
    throw_away = 4 →
    total_socks = 60 →
    total_socks = original_socks - throw_away + new_socks →
    new_socks = 36 :=
by
  intros original_socks throw_away new_socks total_socks h_original h_throw h_total h_eq
  sorry

end NUMINAMATH_GPT_new_socks_bought_l1237_123731


namespace NUMINAMATH_GPT_probability_blue_or_green_face_l1237_123709

def cube_faces: ℕ := 6
def blue_faces: ℕ := 3
def red_faces: ℕ := 2
def green_faces: ℕ := 1

theorem probability_blue_or_green_face (h1: blue_faces + red_faces + green_faces = cube_faces):
  (3 + 1) / 6 = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_blue_or_green_face_l1237_123709


namespace NUMINAMATH_GPT_solve_log_equation_l1237_123786

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solve_log_equation (x : ℝ) (hx : 2 * log_base 5 x - 3 * log_base 5 4 = 1) :
  x = 4 * Real.sqrt 5 ∨ x = -4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_solve_log_equation_l1237_123786


namespace NUMINAMATH_GPT_no_base_for_final_digit_one_l1237_123769

theorem no_base_for_final_digit_one (b : ℕ) (h : 3 ≤ b ∧ b ≤ 10) : ¬ (842 % b = 1) :=
by
  cases h with 
  | intro hb1 hb2 => sorry

end NUMINAMATH_GPT_no_base_for_final_digit_one_l1237_123769


namespace NUMINAMATH_GPT_second_derivative_at_x₀_l1237_123771

noncomputable def f (x : ℝ) : ℝ := sorry
variables (x₀ a b : ℝ)

-- Condition: f(x₀ + Δx) - f(x₀) = a * Δx + b * (Δx)^2
axiom condition : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * (Δx)^2

theorem second_derivative_at_x₀ : deriv (deriv f) x₀ = 2 * b :=
sorry

end NUMINAMATH_GPT_second_derivative_at_x₀_l1237_123771


namespace NUMINAMATH_GPT_total_dogs_l1237_123702

def number_of_boxes : ℕ := 15
def dogs_per_box : ℕ := 8

theorem total_dogs : number_of_boxes * dogs_per_box = 120 := by
  sorry

end NUMINAMATH_GPT_total_dogs_l1237_123702


namespace NUMINAMATH_GPT_sequence_less_than_inverse_l1237_123747

-- Define the sequence and conditions given in the problem
variables {a : ℕ → ℝ}
axiom positive_sequence (n : ℕ) : 0 < a n
axiom sequence_inequality (n : ℕ) : a n ^ 2 ≤ a n - a (n + 1)

theorem sequence_less_than_inverse (n : ℕ) : a n < 1 / n := 
sorry

end NUMINAMATH_GPT_sequence_less_than_inverse_l1237_123747


namespace NUMINAMATH_GPT_square_area_divided_into_rectangles_l1237_123700

theorem square_area_divided_into_rectangles (l w : ℝ) 
  (h1 : 2 * (l + w) = 120)
  (h2 : l = 5 * w) :
  (5 * w * w)^2 = 2500 := 
by {
  -- Sorry placeholder for proof
  sorry
}

end NUMINAMATH_GPT_square_area_divided_into_rectangles_l1237_123700


namespace NUMINAMATH_GPT_sum_of_remainders_l1237_123756

theorem sum_of_remainders {a b c d e : ℤ} (h1 : a % 13 = 3) (h2 : b % 13 = 5) (h3 : c % 13 = 7) (h4 : d % 13 = 9) (h5 : e % 13 = 11) : 
  ((a + b + c + d + e) % 13) = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_l1237_123756


namespace NUMINAMATH_GPT_fraction_solution_l1237_123740

theorem fraction_solution (a : ℤ) (h : 0 < a ∧ (a : ℚ) / (a + 36) = 775 / 1000) : a = 124 := 
by
  sorry

end NUMINAMATH_GPT_fraction_solution_l1237_123740


namespace NUMINAMATH_GPT_football_games_per_month_l1237_123703

theorem football_games_per_month :
  let total_games := 5491
  let months := 17.0
  total_games / months = 323 := 
by
  let total_games := 5491
  let months := 17.0
  -- This is where the actual computation would happen if we were to provide a proof
  sorry

end NUMINAMATH_GPT_football_games_per_month_l1237_123703


namespace NUMINAMATH_GPT_largest_consecutive_odd_number_is_27_l1237_123704

theorem largest_consecutive_odd_number_is_27 (a b c : ℤ) 
  (h1: a + b + c = 75)
  (h2: c - a = 6)
  (h3: b = a + 2)
  (h4: c = a + 4) :
  c = 27 := 
sorry

end NUMINAMATH_GPT_largest_consecutive_odd_number_is_27_l1237_123704


namespace NUMINAMATH_GPT_polygonal_line_exists_l1237_123729

theorem polygonal_line_exists (A : Type) (n q : ℕ) (lengths : Fin q → ℝ)
  (yellow_segments : Fin q → (A × A))
  (h_lengths : ∀ i j : Fin q, i < j → lengths i < lengths j)
  (h_yellow_segments_unique : ∀ i j : Fin q, i ≠ j → yellow_segments i ≠ yellow_segments j) :
  ∃ (m : ℕ), m ≥ 2 * q / n :=
sorry

end NUMINAMATH_GPT_polygonal_line_exists_l1237_123729


namespace NUMINAMATH_GPT_parents_gave_money_l1237_123779

def money_before_birthday : ℕ := 159
def money_from_grandmother : ℕ := 25
def money_from_aunt_uncle : ℕ := 20
def total_money_after_birthday : ℕ := 279

theorem parents_gave_money :
  total_money_after_birthday = money_before_birthday + money_from_grandmother + money_from_aunt_uncle + 75 :=
by
  sorry

end NUMINAMATH_GPT_parents_gave_money_l1237_123779


namespace NUMINAMATH_GPT_num_children_with_identical_cards_l1237_123768

theorem num_children_with_identical_cards (children_mama children_nyanya children_manya total_children mixed_cards : ℕ) 
  (h_mama: children_mama = 20) 
  (h_nyanya: children_nyanya = 30) 
  (h_manya: children_manya = 40) 
  (h_total: total_children = children_mama + children_nyanya) 
  (h_mixed: mixed_cards = children_manya) 
  : total_children - children_manya = 10 :=
by
  -- Sorry to indicate the proof is skipped
  sorry

end NUMINAMATH_GPT_num_children_with_identical_cards_l1237_123768


namespace NUMINAMATH_GPT_total_amount_paid_l1237_123741

-- Define the parameters
def cost_per_night_per_person : ℕ := 40
def number_of_people : ℕ := 3
def number_of_nights : ℕ := 3

-- Define the total cost calculation
def total_cost := cost_per_night_per_person * number_of_people * number_of_nights

-- The statement of the proof problem
theorem total_amount_paid :
  total_cost = 360 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_amount_paid_l1237_123741


namespace NUMINAMATH_GPT_police_emergency_number_has_prime_divisor_gt_7_l1237_123757

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end NUMINAMATH_GPT_police_emergency_number_has_prime_divisor_gt_7_l1237_123757


namespace NUMINAMATH_GPT_solution_set_of_x_squared_lt_one_l1237_123775

theorem solution_set_of_x_squared_lt_one : {x : ℝ | x^2 < 1} = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_x_squared_lt_one_l1237_123775
