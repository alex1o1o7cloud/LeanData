import Mathlib

namespace NUMINAMATH_GPT_Suresh_completes_job_in_15_hours_l1897_189758

theorem Suresh_completes_job_in_15_hours :
  ∃ S : ℝ,
    (∀ (T_A Ashutosh_time Suresh_time : ℝ), Ashutosh_time = 15 ∧ Suresh_time = 9 
    → T_A = Ashutosh_time → 6 / T_A + Suresh_time / S = 1) ∧ S = 15 :=
by
  sorry

end NUMINAMATH_GPT_Suresh_completes_job_in_15_hours_l1897_189758


namespace NUMINAMATH_GPT_remainder_mod_5_is_0_l1897_189780

theorem remainder_mod_5_is_0 :
  (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_mod_5_is_0_l1897_189780


namespace NUMINAMATH_GPT_sum_of_ages_of_henrys_brothers_l1897_189723

theorem sum_of_ages_of_henrys_brothers (a b c : ℕ) : 
  a = 2 * b → 
  b = c ^ 2 →
  a ≠ b ∧ a ≠ c ∧ b ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  a + b + c = 14 :=
by
  intro h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_sum_of_ages_of_henrys_brothers_l1897_189723


namespace NUMINAMATH_GPT_sqrt_recursive_value_l1897_189710

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end NUMINAMATH_GPT_sqrt_recursive_value_l1897_189710


namespace NUMINAMATH_GPT_number_of_elements_in_M_l1897_189772

def positive_nats : Set ℕ := {n | n > 0}
def M : Set ℕ := {m | ∃ n ∈ positive_nats, m = 2 * n - 1 ∧ m < 60}

theorem number_of_elements_in_M : ∃ s : Finset ℕ, (∀ x, x ∈ s ↔ x ∈ M) ∧ s.card = 30 := 
by
  sorry

end NUMINAMATH_GPT_number_of_elements_in_M_l1897_189772


namespace NUMINAMATH_GPT_translate_point_correct_l1897_189753

-- Define initial point
def initial_point : ℝ × ℝ := (0, 1)

-- Define translation downward
def translate_down (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 - units)

-- Define translation to the left
def translate_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Define the expected resulting point
def expected_point : ℝ × ℝ := (-4, -1)

-- Lean statement to prove the equivalence
theorem translate_point_correct :
  (translate_left (translate_down initial_point 2) 4) = expected_point :=
by 
  -- Here, we would prove it step by step if required
  sorry

end NUMINAMATH_GPT_translate_point_correct_l1897_189753


namespace NUMINAMATH_GPT_abs_diff_expr_l1897_189745

theorem abs_diff_expr :
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  |a| - |b| = 4 :=
by
  let a := -3 * (7 - 15)
  let b := (5 - 7)^2 + (-4)^2
  sorry

end NUMINAMATH_GPT_abs_diff_expr_l1897_189745


namespace NUMINAMATH_GPT_rectangle_area_eq_six_l1897_189726

-- Define the areas of the small squares
def smallSquareArea : ℝ := 1

-- Define the number of small squares
def numberOfSmallSquares : ℤ := 2

-- Define the area of the larger square
def largeSquareArea : ℝ := (2 ^ 2)

-- Define the area of rectangle ABCD
def areaRectangleABCD : ℝ :=
  (numberOfSmallSquares * smallSquareArea) + largeSquareArea

-- The theorem we want to prove
theorem rectangle_area_eq_six :
  areaRectangleABCD = 6 := by sorry

end NUMINAMATH_GPT_rectangle_area_eq_six_l1897_189726


namespace NUMINAMATH_GPT_num_solutions_non_negative_reals_l1897_189729

-- Define the system of equations as a function to express the cyclic nature
def system_of_equations (n : ℕ) (x : ℕ → ℝ) (k : ℕ) : Prop :=
  x (k + 1 % n) + (x (if k = 0 then n else k) ^ 2) = 4 * x (if k = 0 then n else k)

-- Define the main theorem stating the number of solutions
theorem num_solutions_non_negative_reals {n : ℕ} (hn : 0 < n) : 
  ∃ (s : Finset (ℕ → ℝ)), (∀ x ∈ s, ∀ k, 0 ≤ (x k) ∧ system_of_equations n x k) ∧ s.card = 2^n :=
sorry

end NUMINAMATH_GPT_num_solutions_non_negative_reals_l1897_189729


namespace NUMINAMATH_GPT_transformation_correct_l1897_189720

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

-- Define the transformation functions
noncomputable def shift_right_by_pi_over_10 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - Real.pi / 10)
noncomputable def stretch_x_by_factor_of_2 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x / 2)

-- Define the transformed function
noncomputable def transformed_function : ℝ → ℝ :=
  stretch_x_by_factor_of_2 (shift_right_by_pi_over_10 original_function)

-- Define the expected resulting function
noncomputable def expected_function (x : ℝ) : ℝ := Real.sin (x / 2 - Real.pi / 10)

-- State the theorem
theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = expected_function x :=
by
  sorry

end NUMINAMATH_GPT_transformation_correct_l1897_189720


namespace NUMINAMATH_GPT_determine_x_l1897_189771

theorem determine_x (x : ℚ) (h : ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) : x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_determine_x_l1897_189771


namespace NUMINAMATH_GPT_vertical_angles_are_congruent_l1897_189711

def supplementary_angles (a b : ℝ) : Prop := a + b = 180
def corresponding_angles (l1 l2 t : ℝ) : Prop := l1 = l2
def exterior_angle_greater (ext int1 int2 : ℝ) : Prop := ext = int1 + int2
def vertical_angles_congruent (a b : ℝ) : Prop := a = b

theorem vertical_angles_are_congruent (a b : ℝ) (h : vertical_angles_congruent a b) : a = b := by
  sorry

end NUMINAMATH_GPT_vertical_angles_are_congruent_l1897_189711


namespace NUMINAMATH_GPT_thirty_one_star_thirty_two_l1897_189775

def complex_op (x y : ℝ) : ℝ :=
sorry

axiom op_zero (x : ℝ) : complex_op x 0 = 1

axiom op_associative (x y z : ℝ) : complex_op (complex_op x y) z = z * (x * y) + z

theorem thirty_one_star_thirty_two : complex_op 31 32 = 993 :=
by
  sorry

end NUMINAMATH_GPT_thirty_one_star_thirty_two_l1897_189775


namespace NUMINAMATH_GPT_exists_prime_q_and_positive_n_l1897_189795

theorem exists_prime_q_and_positive_n (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ q n : ℕ, Nat.Prime q ∧ q < p ∧ 0 < n ∧ p ∣ (n^2 - q) :=
by
  sorry

end NUMINAMATH_GPT_exists_prime_q_and_positive_n_l1897_189795


namespace NUMINAMATH_GPT_range_of_a_l1897_189768

noncomputable def f (x : ℝ) := Real.log (x + 1)
def A (x : ℝ) := (f (1 - 2 * x) > f x)
def B (a x : ℝ) := (a - 1 < x) ∧ (x < 2 * a^2)

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, A x ∧ B a x) ↔ (a < -1 / 2) ∨ (1 < a ∧ a < 4 / 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1897_189768


namespace NUMINAMATH_GPT_base_7_sum_of_product_l1897_189776

-- Definitions of the numbers in base-10 for base-7 numbers
def base_7_to_base_10 (d1 d0 : ℕ) : ℕ := d1 * 7 + d0

def sum_digits_base_7 (n : ℕ) : ℕ := 
  let d2 := n / 343
  let r2 := n % 343
  let d1 := r2 / 49
  let r1 := r2 % 49
  let d0 := r1 / 7 + r1 % 7
  d2 + d1 + d0

def convert_10_to_7 (n : ℕ) : ℕ := 
  let d1 := n / 7
  let r1 := n % 7
  d1 * 10 + r1

theorem base_7_sum_of_product : 
  let n36  := base_7_to_base_10 3 6
  let n52  := base_7_to_base_10 5 2
  let nadd := base_7_to_base_10 2 0
  let prod := n36 * n52
  let suma := prod + nadd
  convert_10_to_7 (sum_digits_base_7 suma) = 23 :=
by
  sorry

end NUMINAMATH_GPT_base_7_sum_of_product_l1897_189776


namespace NUMINAMATH_GPT_no_solution_ineq_l1897_189788

theorem no_solution_ineq (m : ℝ) :
  (¬ ∃ (x : ℝ), x - 1 > 1 ∧ x < m) → m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_ineq_l1897_189788


namespace NUMINAMATH_GPT_correct_equations_l1897_189763

-- Defining the problem statement
theorem correct_equations (m n : ℕ) :
  (∀ (m n : ℕ), 40 * m + 10 = 43 * m + 1 ∧ 
   (n - 10) / 40 = (n - 1) / 43) :=
by
  sorry

end NUMINAMATH_GPT_correct_equations_l1897_189763


namespace NUMINAMATH_GPT_domain_of_f_l1897_189728

def domain_f (x : ℝ) : Prop :=
  2 * x - 3 ≥ 0 ∧ x ≠ 3

def domain_set : Set ℝ :=
  { x | (3 / 2) ≤ x ∧ x < 3 ∨ 3 < x }

theorem domain_of_f :
  { x : ℝ | domain_f x } = domain_set := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1897_189728


namespace NUMINAMATH_GPT_jian_wins_cases_l1897_189791

inductive Move
| rock : Move
| paper : Move
| scissors : Move

def wins (jian shin : Move) : Prop :=
  (jian = Move.rock ∧ shin = Move.scissors) ∨
  (jian = Move.paper ∧ shin = Move.rock) ∨
  (jian = Move.scissors ∧ shin = Move.paper)

theorem jian_wins_cases : ∃ n : Nat, n = 3 ∧ (∀ jian shin, wins jian shin → n = 3) :=
by
  sorry

end NUMINAMATH_GPT_jian_wins_cases_l1897_189791


namespace NUMINAMATH_GPT_range_of_values_for_sqrt_l1897_189741

theorem range_of_values_for_sqrt (x : ℝ) : (x + 3 ≥ 0) ↔ (x ≥ -3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_values_for_sqrt_l1897_189741


namespace NUMINAMATH_GPT_no_such_rectangle_l1897_189719

theorem no_such_rectangle (a b x y : ℝ) (ha : a < b)
  (hx : x < a / 2) (hy : y < a / 2)
  (h_perimeter : 2 * (x + y) = a + b)
  (h_area : x * y = (a * b) / 2) :
  false :=
sorry

end NUMINAMATH_GPT_no_such_rectangle_l1897_189719


namespace NUMINAMATH_GPT_number_of_new_players_l1897_189709

-- Definitions based on conditions
def total_groups : Nat := 2
def players_per_group : Nat := 5
def returning_players : Nat := 6

-- Convert conditions to definition
def total_players : Nat := total_groups * players_per_group

-- Define what we want to prove
def new_players : Nat := total_players - returning_players

-- The proof problem statement
theorem number_of_new_players :
  new_players = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_new_players_l1897_189709


namespace NUMINAMATH_GPT_arun_weight_l1897_189705

theorem arun_weight (W B : ℝ) (h1 : 65 < W ∧ W < 72) (h2 : B < W ∧ W < 70) (h3 : W ≤ 68) (h4 : (B + 68) / 2 = 67) : B = 66 :=
sorry

end NUMINAMATH_GPT_arun_weight_l1897_189705


namespace NUMINAMATH_GPT_speed_first_half_proof_l1897_189734

noncomputable def speed_first_half
  (total_time: ℕ) 
  (distance: ℕ) 
  (second_half_speed: ℕ) 
  (first_half_time: ℕ) :
  ℕ :=
  distance / first_half_time

theorem speed_first_half_proof
  (total_time: ℕ)
  (distance: ℕ)
  (second_half_speed: ℕ)
  (half_distance: ℕ)
  (second_half_time: ℕ)
  (first_half_time: ℕ) :
  total_time = 12 →
  distance = 560 →
  second_half_speed = 40 →
  half_distance = distance / 2 →
  second_half_time = half_distance / second_half_speed →
  first_half_time = total_time - second_half_time →
  speed_first_half total_time half_distance second_half_speed first_half_time = 56 :=
by
  sorry

end NUMINAMATH_GPT_speed_first_half_proof_l1897_189734


namespace NUMINAMATH_GPT_solution_to_problem_l1897_189730

theorem solution_to_problem (x y : ℕ) : 
  (x.gcd y + x.lcm y = x + y) ↔ 
  ∃ (d k : ℕ), (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d) :=
by sorry

end NUMINAMATH_GPT_solution_to_problem_l1897_189730


namespace NUMINAMATH_GPT_sin_2pi_minus_theta_l1897_189762

theorem sin_2pi_minus_theta (theta : ℝ) (k : ℤ) 
  (h1 : 3 * Real.cos theta ^ 2 = Real.tan theta + 3)
  (h2 : theta ≠ k * Real.pi) :
  Real.sin (2 * (Real.pi - theta)) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_sin_2pi_minus_theta_l1897_189762


namespace NUMINAMATH_GPT_problem1_problem2_l1897_189787

-- Problem 1: Prove (-a^3)^2 * (-a^2)^3 / a = -a^11 given a is a real number.
theorem problem1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 :=
  sorry

-- Problem 2: Prove (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 given m, n are real numbers.
theorem problem2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1897_189787


namespace NUMINAMATH_GPT_quadratic_roots_range_l1897_189704

theorem quadratic_roots_range (a : ℝ) :
  (a-1) * x^2 - 2*x + 1 = 0 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a-1) * x1^2 - 2*x1 + 1 = 0 ∧ (a-1) * x2^2 - 2*x2 + 1 = 0) → (a < 2 ∧ a ≠ 1) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_range_l1897_189704


namespace NUMINAMATH_GPT_distance_between_locations_A_and_B_l1897_189755

theorem distance_between_locations_A_and_B 
  (speed_A speed_B speed_C : ℝ)
  (distance_CD : ℝ)
  (distance_initial_A : ℝ)
  (distance_A_to_B : ℝ)
  (h1 : speed_A = 3 * speed_C)
  (h2 : speed_A = 1.5 * speed_B)
  (h3 : distance_CD = 12)
  (h4 : distance_initial_A = 50)
  (h5 : distance_A_to_B = 130)
  : distance_A_to_B = 130 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_locations_A_and_B_l1897_189755


namespace NUMINAMATH_GPT_kyoko_payment_l1897_189713

noncomputable def total_cost (balls skipropes frisbees : ℕ) (ball_cost rope_cost frisbee_cost : ℝ) : ℝ :=
  (balls * ball_cost) + (skipropes * rope_cost) + (frisbees * frisbee_cost)

noncomputable def final_amount (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (discount_rate * total_cost)

theorem kyoko_payment :
  let balls := 3
  let skipropes := 2
  let frisbees := 4
  let ball_cost := 1.54
  let rope_cost := 3.78
  let frisbee_cost := 2.63
  let discount_rate := 0.07
  final_amount (total_cost balls skipropes frisbees ball_cost rope_cost frisbee_cost) discount_rate = 21.11 :=
by
  sorry

end NUMINAMATH_GPT_kyoko_payment_l1897_189713


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l1897_189781

noncomputable def x : ℚ := 75 / 99  -- 0.\overline{75}
noncomputable def y : ℚ := 223 / 99  -- 2.\overline{25}

theorem repeating_decimal_fraction : (x / y) = 2475 / 7329 :=
by
  -- Further proof details can be added here
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l1897_189781


namespace NUMINAMATH_GPT_problem_part_a_problem_part_b_l1897_189746

def is_two_squared (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a^2 + b^2 ∧ a ≠ 0 ∧ b ≠ 0

def is_three_squared (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = a^2 + b^2 + c^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def is_four_squared (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a^2 + b^2 + c^2 + d^2 ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def satisfies_prime_conditions (e : ℕ) : Prop :=
  Nat.Prime (e - 2) ∧ Nat.Prime e ∧ Nat.Prime (e + 4)

def satisfies_square_sum_conditions (a b c d e : ℕ) : Prop :=
  a^2 + b^2 + c^2 + d^2 + e^2 = 2020 ∧ a < b ∧ b < c ∧ c < d ∧ d < e

theorem problem_part_a : is_two_squared 2020 ∧ is_three_squared 2020 ∧ is_four_squared 2020 := sorry

theorem problem_part_b : ∃ a b c d e : ℕ, satisfies_prime_conditions e ∧ satisfies_square_sum_conditions a b c d e :=
  sorry

end NUMINAMATH_GPT_problem_part_a_problem_part_b_l1897_189746


namespace NUMINAMATH_GPT_range_of_a_l1897_189789

noncomputable def A : Set ℝ := {x | x ≥ abs (x^2 - 2 * x)}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1897_189789


namespace NUMINAMATH_GPT_initial_avg_mark_l1897_189761

variable (A : ℝ) -- The initial average mark

-- Conditions
def num_students : ℕ := 33
def avg_excluded_students : ℝ := 40
def num_excluded_students : ℕ := 3
def avg_remaining_students : ℝ := 95

-- Equation derived from the problem conditions
def initial_avg :=
  A * num_students - avg_excluded_students * num_excluded_students = avg_remaining_students * (num_students - num_excluded_students)

theorem initial_avg_mark :
  initial_avg A →
  A = 90 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_avg_mark_l1897_189761


namespace NUMINAMATH_GPT_sqrt_of_second_number_l1897_189759

-- Given condition: the arithmetic square root of a natural number n is x
variable (x : ℕ)
def first_number := x ^ 2
def second_number := first_number + 1

-- The theorem statement we want to prove
theorem sqrt_of_second_number (x : ℕ) : Real.sqrt (x^2 + 1) = Real.sqrt (first_number x + 1) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_second_number_l1897_189759


namespace NUMINAMATH_GPT_solution_for_factorial_equation_l1897_189725

theorem solution_for_factorial_equation:
  { (n, k) : ℕ × ℕ | 0 < n ∧ 0 < k ∧ n! + n = n^k } = {(2,2), (3,2), (5,3)} :=
by
  sorry

end NUMINAMATH_GPT_solution_for_factorial_equation_l1897_189725


namespace NUMINAMATH_GPT_abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l1897_189773

theorem abs_eq_ax_plus_1_one_negative_root_no_positive_roots (a : ℝ) :
  (∃ x : ℝ, |x| = a * x + 1 ∧ x < 0) ∧ (∀ x : ℝ, |x| = a * x + 1 → x ≤ 0) → a > -1 :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_ax_plus_1_one_negative_root_no_positive_roots_l1897_189773


namespace NUMINAMATH_GPT_correct_statements_l1897_189765

theorem correct_statements (a b c : ℝ) (h : ∀ x, ax^2 + bx + c > 0 ↔ -2 < x ∧ x < 3) :
  ( ∃ (x : ℝ), c*x^2 + b*x + a < 0 ↔ -1/2 < x ∧ x < 1/3 ) ∧
  ( ∃ (b : ℝ), ∀ b, 12/(3*b + 4) + b = 8/3 ) ∧
  ( ∀ m, ¬ (m < -1 ∨ m > 2) ) ∧
  ( c = 2 → ∀ n1 n2, (3*a*n1^2 + 6*b*n1 = -3 ∧ 3*a*n2^2 + 6*b*n2 = 1) → n2 - n1 ∈ [2, 4] ) :=
sorry

end NUMINAMATH_GPT_correct_statements_l1897_189765


namespace NUMINAMATH_GPT_energy_soda_packs_l1897_189766

-- Definitions and conditions
variables (total_bottles : ℕ) (regular_soda : ℕ) (diet_soda : ℕ) (pack_size : ℕ)
variables (complete_packs : ℕ) (remaining_regular : ℕ) (remaining_diet : ℕ) (remaining_energy : ℕ)

-- Conditions given in the problem
axiom h_total_bottles : total_bottles = 200
axiom h_regular_soda : regular_soda = 55
axiom h_diet_soda : diet_soda = 40
axiom h_pack_size : pack_size = 3

-- Proving the correct answer
theorem energy_soda_packs :
  complete_packs = (total_bottles - (regular_soda + diet_soda)) / pack_size ∧
  remaining_regular = regular_soda ∧
  remaining_diet = diet_soda ∧
  remaining_energy = (total_bottles - (regular_soda + diet_soda)) % pack_size :=
by
  sorry

end NUMINAMATH_GPT_energy_soda_packs_l1897_189766


namespace NUMINAMATH_GPT_largest_six_digit_number_l1897_189731

/-- The largest six-digit number \( A \) that is divisible by 19, 
  the number obtained by removing its last digit is divisible by 17, 
  and the number obtained by removing the last two digits in \( A \) is divisible by 13 
  is \( 998412 \). -/
theorem largest_six_digit_number (A : ℕ) (h1 : A % 19 = 0) 
  (h2 : (A / 10) % 17 = 0) 
  (h3 : (A / 100) % 13 = 0) : 
  A = 998412 :=
sorry

end NUMINAMATH_GPT_largest_six_digit_number_l1897_189731


namespace NUMINAMATH_GPT_sum_of_x_and_y_l1897_189793

theorem sum_of_x_and_y (x y : ℤ) (h1 : 3 + x = 5) (h2 : -3 + y = 5) : x + y = 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l1897_189793


namespace NUMINAMATH_GPT_total_family_members_l1897_189736

variable (members_father_side : Nat) (percent_incr : Nat)
variable (members_mother_side := members_father_side + (members_father_side * percent_incr / 100))
variable (total_members := members_father_side + members_mother_side)

theorem total_family_members 
  (h1 : members_father_side = 10) 
  (h2 : percent_incr = 30) :
  total_members = 23 :=
by
  sorry

end NUMINAMATH_GPT_total_family_members_l1897_189736


namespace NUMINAMATH_GPT_used_mystery_books_l1897_189794

theorem used_mystery_books (total_books used_adventure_books new_crime_books : ℝ)
  (h1 : total_books = 45)
  (h2 : used_adventure_books = 13.0)
  (h3 : new_crime_books = 15.0) :
  total_books - (used_adventure_books + new_crime_books) = 17.0 := by
  sorry

end NUMINAMATH_GPT_used_mystery_books_l1897_189794


namespace NUMINAMATH_GPT_contrapositive_example_l1897_189748

theorem contrapositive_example (x : ℝ) :
  (¬ (x = 3 ∧ x = 4)) → (x^2 - 7 * x + 12 ≠ 0) →
  (x^2 - 7 * x + 12 = 0) → (x = 3 ∨ x = 4) :=
by
  intros h h1 h2
  sorry  -- proof is not required

end NUMINAMATH_GPT_contrapositive_example_l1897_189748


namespace NUMINAMATH_GPT_inscribed_sphere_radius_of_tetrahedron_l1897_189797

variables (V S1 S2 S3 S4 R : ℝ)

theorem inscribed_sphere_radius_of_tetrahedron
  (hV_pos : 0 < V)
  (hS_pos : 0 < S1) (hS2_pos : 0 < S2) (hS3_pos : 0 < S3) (hS4_pos : 0 < S4) :
  R = 3 * V / (S1 + S2 + S3 + S4) :=
sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_of_tetrahedron_l1897_189797


namespace NUMINAMATH_GPT_find_value_of_expression_l1897_189747

noncomputable def p : ℝ := 3
noncomputable def q : ℝ := 7
noncomputable def r : ℝ := 5

def inequality_holds (f : ℝ → ℝ) : Prop :=
  ∀ x, (f x ≥ 0 ↔ (x ∈ Set.Icc 3 7 ∨ x > 5))

def given_condition : Prop := p < q

theorem find_value_of_expression (f : ℝ → ℝ)
  (h : inequality_holds f)
  (hc : given_condition) :
  p + 2*q + 3*r = 32 := 
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1897_189747


namespace NUMINAMATH_GPT_sum_of_coefficients_l1897_189715

theorem sum_of_coefficients (A B C : ℤ) 
  (h_factorization : ∀ x, x^3 + A * x^2 + B * x + C = (x + 2) * (x - 2) * (x - 1)) :
  A + B + C = -1 :=
by sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1897_189715


namespace NUMINAMATH_GPT_eval_expr_l1897_189760

theorem eval_expr : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 262400 := by
  sorry

end NUMINAMATH_GPT_eval_expr_l1897_189760


namespace NUMINAMATH_GPT_angel_vowels_written_l1897_189782

theorem angel_vowels_written (num_vowels : ℕ) (times_written : ℕ) (h1 : num_vowels = 5) (h2 : times_written = 4) : num_vowels * times_written = 20 := by
  sorry

end NUMINAMATH_GPT_angel_vowels_written_l1897_189782


namespace NUMINAMATH_GPT_range_of_a_l1897_189790

theorem range_of_a {A : Set ℝ} (h1: ∀ x ∈ A, 2 * x + a > 0) (h2: 1 ∉ A) (h3: 2 ∈ A) : -4 < a ∧ a ≤ -2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1897_189790


namespace NUMINAMATH_GPT_Barry_reach_l1897_189727

noncomputable def Larry_full_height : ℝ := 5
noncomputable def Larry_shoulder_height : ℝ := Larry_full_height - 0.2 * Larry_full_height
noncomputable def combined_reach : ℝ := 9

theorem Barry_reach :
  combined_reach - Larry_shoulder_height = 5 := 
by
  -- Correct answer verification comparing combined reach minus Larry's shoulder height equals 5
  sorry

end NUMINAMATH_GPT_Barry_reach_l1897_189727


namespace NUMINAMATH_GPT_ab_c_work_days_l1897_189784

noncomputable def W_ab : ℝ := 1 / 15
noncomputable def W_c : ℝ := 1 / 30
noncomputable def W_abc : ℝ := W_ab + W_c

theorem ab_c_work_days :
  (1 / W_abc) = 10 :=
by
  sorry

end NUMINAMATH_GPT_ab_c_work_days_l1897_189784


namespace NUMINAMATH_GPT_find_principal_amount_l1897_189774

theorem find_principal_amount
  (P R T SI : ℝ) 
  (rate_condition : R = 12)
  (time_condition : T = 20)
  (interest_condition : SI = 2100) :
  SI = (P * R * T) / 100 → P = 875 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1897_189774


namespace NUMINAMATH_GPT_original_weight_l1897_189756

variable (W : ℝ) -- Let W be the original weight of the side of beef

-- Conditions
def condition1 : ℝ := 0.80 * W -- Weight after first stage
def condition2 : ℝ := 0.70 * condition1 W -- Weight after second stage
def condition3 : ℝ := 0.75 * condition2 W -- Weight after third stage

-- Final weight is given as 570 pounds
theorem original_weight (h : condition3 W = 570) : W = 1357.14 :=
by 
  sorry

end NUMINAMATH_GPT_original_weight_l1897_189756


namespace NUMINAMATH_GPT_arrange_scores_l1897_189751

variable {K Q M S : ℝ}

theorem arrange_scores (h1 : Q > K) (h2 : M > S) (h3 : S < max Q (max M K)) : S < M ∧ M < Q := by
  sorry

end NUMINAMATH_GPT_arrange_scores_l1897_189751


namespace NUMINAMATH_GPT_cube_root_of_27_l1897_189779

theorem cube_root_of_27 : ∃ x : ℝ, x ^ 3 = 27 ↔ ∃ y : ℝ, y = 3 := by
  sorry

end NUMINAMATH_GPT_cube_root_of_27_l1897_189779


namespace NUMINAMATH_GPT_average_of_three_numbers_l1897_189714

theorem average_of_three_numbers (a b c : ℝ)
  (h1 : a + (b + c) / 2 = 65)
  (h2 : b + (a + c) / 2 = 69)
  (h3 : c + (a + b) / 2 = 76) :
  (a + b + c) / 3 = 35 := 
sorry

end NUMINAMATH_GPT_average_of_three_numbers_l1897_189714


namespace NUMINAMATH_GPT_max_value_expression_l1897_189712

noncomputable def expression (x : ℝ) : ℝ := 5^x - 25^x

theorem max_value_expression : 
  (∀ x : ℝ, expression x ≤ 1/4) ∧ (∃ x : ℝ, expression x = 1/4) := 
by 
  sorry

end NUMINAMATH_GPT_max_value_expression_l1897_189712


namespace NUMINAMATH_GPT_average_people_per_row_l1897_189717

theorem average_people_per_row (boys girls rows : ℕ) (h_boys : boys = 24) (h_girls : girls = 24) (h_rows : rows = 6) : 
  (boys + girls) / rows = 8 :=
by
  sorry

end NUMINAMATH_GPT_average_people_per_row_l1897_189717


namespace NUMINAMATH_GPT_woman_worked_days_l1897_189701

-- Define variables and conditions
variables (W I : ℕ)

-- Conditions
def total_days : Prop := W + I = 25
def net_earnings : Prop := 20 * W - 5 * I = 450

-- Main theorem statement
theorem woman_worked_days (h1 : total_days W I) (h2 : net_earnings W I) : W = 23 :=
sorry

end NUMINAMATH_GPT_woman_worked_days_l1897_189701


namespace NUMINAMATH_GPT_total_wheels_in_both_garages_l1897_189708

/-- Each cycle type has a different number of wheels. --/
def wheels_per_cycle (cycle_type: String) : ℕ :=
  if cycle_type = "bicycle" then 2
  else if cycle_type = "tricycle" then 3
  else if cycle_type = "unicycle" then 1
  else if cycle_type = "quadracycle" then 4
  else 0

/-- Define the counts of each type of cycle in each garage. --/
def garage1_counts := [("bicycle", 5), ("tricycle", 6), ("unicycle", 9), ("quadracycle", 3)]
def garage2_counts := [("bicycle", 2), ("tricycle", 1), ("unicycle", 3), ("quadracycle", 4)]

/-- Total steps for the calculation --/
def wheels_in_garage (garage_counts: List (String × ℕ)) (missing_wheels_unicycles: ℕ) : ℕ :=
  List.foldl (λ acc (cycle_count: String × ℕ) => 
              acc + (if cycle_count.1 = "unicycle" then (cycle_count.2 * wheels_per_cycle cycle_count.1 - missing_wheels_unicycles) 
                     else (cycle_count.2 * wheels_per_cycle cycle_count.1))) 0 garage_counts

/-- The total number of wheels in both garages. --/
def total_wheels : ℕ := wheels_in_garage garage1_counts 0 + wheels_in_garage garage2_counts 3

/-- Prove that the total number of wheels in both garages is 72. --/
theorem total_wheels_in_both_garages : total_wheels = 72 :=
  by sorry

end NUMINAMATH_GPT_total_wheels_in_both_garages_l1897_189708


namespace NUMINAMATH_GPT_sculptures_not_on_display_eq_1200_l1897_189742

-- Define the number of pieces of art in the gallery
def total_pieces_art := 2700

-- Define the number of pieces on display (1/3 of total pieces)
def pieces_on_display := total_pieces_art / 3

-- Define the number of pieces not on display
def pieces_not_on_display := total_pieces_art - pieces_on_display

-- Define the number of sculptures on display (1/6 of pieces on display)
def sculptures_on_display := pieces_on_display / 6

-- Define the number of paintings not on display (1/3 of pieces not on display)
def paintings_not_on_display := pieces_not_on_display / 3

-- Prove the number of sculptures not on display
theorem sculptures_not_on_display_eq_1200 :
  total_pieces_art = 2700 →
  pieces_on_display = total_pieces_art / 3 →
  pieces_not_on_display = total_pieces_art - pieces_on_display →
  sculptures_on_display = pieces_on_display / 6 →
  paintings_not_on_display = pieces_not_on_display / 3 →
  pieces_not_on_display - paintings_not_on_display = 1200 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_sculptures_not_on_display_eq_1200_l1897_189742


namespace NUMINAMATH_GPT_least_common_multiple_increments_l1897_189785

theorem least_common_multiple_increments :
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  Nat.lcm (Nat.lcm (Nat.lcm a' b') c') d' = 8645 :=
by
  let a := 4; let b := 6; let c := 12; let d := 18
  let a' := a + 1; let b' := b + 1; let c' := c + 1; let d' := d + 1
  sorry

end NUMINAMATH_GPT_least_common_multiple_increments_l1897_189785


namespace NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1897_189767

/--
\(a > 1\) is a sufficient but not necessary condition for \(\frac{1}{a} < 1\).
-/
theorem sufficient_condition (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by
  sorry

theorem not_necessary_condition (a : ℝ) (h : 1 / a < 1) : a > 1 ∨ a < 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_not_necessary_condition_l1897_189767


namespace NUMINAMATH_GPT_find_g_neg_6_l1897_189702

def f (x : ℚ) : ℚ := 4 * x - 9
def g (y : ℚ) : ℚ := 3 * (y * y) + 4 * y - 2

theorem find_g_neg_6 : g (-6) = 43 / 16 := by
  sorry

end NUMINAMATH_GPT_find_g_neg_6_l1897_189702


namespace NUMINAMATH_GPT_apples_shared_equally_l1897_189716

-- Definitions of the given conditions
def num_apples : ℕ := 9
def num_friends : ℕ := 3

-- Statement of the problem
theorem apples_shared_equally : num_apples / num_friends = 3 := by
  sorry

end NUMINAMATH_GPT_apples_shared_equally_l1897_189716


namespace NUMINAMATH_GPT_distance_between_points_on_line_l1897_189703

theorem distance_between_points_on_line (a b c d m k : ℝ) 
  (hab : b = m * a + k) (hcd : d = m * c + k) :
  dist (a, b) (c, d) = |a - c| * Real.sqrt (1 + m^2) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_on_line_l1897_189703


namespace NUMINAMATH_GPT_count_divisors_2022_2022_l1897_189796

noncomputable def num_divisors_2022_2022 : ℕ :=
  let fac2022 := 2022
  let factor_triplets := [(2, 3, 337), (3, 337, 2), (2, 337, 3), (337, 2, 3), (337, 3, 2), (3, 2, 337)]
  factor_triplets.length

theorem count_divisors_2022_2022 :
  num_divisors_2022_2022 = 6 :=
  by {
    sorry
  }

end NUMINAMATH_GPT_count_divisors_2022_2022_l1897_189796


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1897_189752

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2 * a * b) ↔ (a/b + b/a ≥ 2) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1897_189752


namespace NUMINAMATH_GPT_bedrooms_count_l1897_189764

/-- Number of bedrooms calculation based on given conditions -/
theorem bedrooms_count (B : ℕ) (h1 : ∀ b, b = 20 * B)
  (h2 : ∀ lr, lr = 20 * B)
  (h3 : ∀ bath, bath = 2 * 20 * B)
  (h4 : ∀ out, out = 2 * (20 * B + 20 * B + 40 * B))
  (h5 : ∀ siblings, siblings = 3)
  (h6 : ∀ work_time, work_time = 4 * 60) : B = 3 :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_bedrooms_count_l1897_189764


namespace NUMINAMATH_GPT_find_weights_l1897_189757

theorem find_weights (x y z : ℕ) (h1 : x + y + z = 11) (h2 : 3 * x + 7 * y + 14 * z = 108) :
  x = 1 ∧ y = 5 ∧ z = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_weights_l1897_189757


namespace NUMINAMATH_GPT_cylinder_surface_area_l1897_189786

theorem cylinder_surface_area (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 2 * r) : 
  2 * Real.pi * r^2 + 2 * Real.pi * r * l = 24 * Real.pi :=
by
  subst h1
  subst h2
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l1897_189786


namespace NUMINAMATH_GPT_find_f1_l1897_189744

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 2) = x * f (x)

theorem find_f1 (f : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : functional_equation f) : 
  f 1 = 0 :=
sorry

end NUMINAMATH_GPT_find_f1_l1897_189744


namespace NUMINAMATH_GPT_possible_to_form_square_l1897_189706

noncomputable def shape : Type := sorry
noncomputable def is_square (s : shape) : Prop := sorry
noncomputable def divide_into_parts (s : shape) (n : ℕ) : Prop := sorry
noncomputable def all_triangles (s : shape) : Prop := sorry

theorem possible_to_form_square (s : shape) :
  (∃ (parts : ℕ), parts ≤ 4 ∧ divide_into_parts s parts ∧ is_square s) ∧
  (∃ (parts : ℕ), parts ≤ 5 ∧ divide_into_parts s parts ∧ all_triangles s ∧ is_square s) :=
sorry

end NUMINAMATH_GPT_possible_to_form_square_l1897_189706


namespace NUMINAMATH_GPT_find_x_l1897_189777

theorem find_x : 2^4 + 3 = 5^2 - 6 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1897_189777


namespace NUMINAMATH_GPT_remaining_problems_to_grade_l1897_189798

-- Define the conditions
def problems_per_worksheet : ℕ := 3
def total_worksheets : ℕ := 15
def graded_worksheets : ℕ := 7

-- The remaining worksheets to grade
def remaining_worksheets : ℕ := total_worksheets - graded_worksheets

-- Theorems stating the amount of problems left to grade
theorem remaining_problems_to_grade : problems_per_worksheet * remaining_worksheets = 24 :=
by
  sorry

end NUMINAMATH_GPT_remaining_problems_to_grade_l1897_189798


namespace NUMINAMATH_GPT_playground_length_l1897_189743

theorem playground_length
  (L_g : ℝ) -- length of the garden
  (L_p : ℝ) -- length of the playground
  (width_garden : ℝ := 24) -- width of the garden
  (width_playground : ℝ := 12) -- width of the playground
  (perimeter_garden : ℝ := 64) -- perimeter of the garden
  (area_garden : ℝ := L_g * 24) -- area of the garden
  (area_playground : ℝ := L_p * 12) -- area of the playground
  (areas_equal : area_garden = area_playground) -- equal areas
  (perimeter_condition : 2 * (L_g + 24) = 64) -- perimeter condition
  : L_p = 16 := 
by
  sorry

end NUMINAMATH_GPT_playground_length_l1897_189743


namespace NUMINAMATH_GPT_sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l1897_189738

-- Definitions for vertices of pyramids
variables (A B C D E : ℝ)

-- Assuming E is inside pyramid ABCD
variable (inside : E ∈ convex_hull ℝ {A, B, C, D})

-- Assertion 1
theorem sum_of_edges_not_always_smaller
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : D ≠ E):
  ¬ (abs A - E + abs B - E + abs C - E < abs A - D + abs B - D + abs C - D) :=
sorry

-- Assertion 2
theorem at_least_one_edge_shorter
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A)
  (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D)
  (h7 : D ≠ E):
  abs A - E < abs A - D ∨ abs B - E < abs B - D ∨ abs C - E < abs C - D :=
sorry

end NUMINAMATH_GPT_sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l1897_189738


namespace NUMINAMATH_GPT_angle_420_mod_360_eq_60_l1897_189735

def angle_mod_equiv (a b : ℕ) : Prop := a % 360 = b

theorem angle_420_mod_360_eq_60 : angle_mod_equiv 420 60 := 
by
  sorry

end NUMINAMATH_GPT_angle_420_mod_360_eq_60_l1897_189735


namespace NUMINAMATH_GPT_initial_average_age_l1897_189700

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 9) (h2 : (n * A + 35) / (n + 1) = 17) :
  A = 15 :=
by
  sorry

end NUMINAMATH_GPT_initial_average_age_l1897_189700


namespace NUMINAMATH_GPT_square_area_l1897_189739

theorem square_area (side_length : ℕ) (h : side_length = 16) : side_length * side_length = 256 := by
  sorry

end NUMINAMATH_GPT_square_area_l1897_189739


namespace NUMINAMATH_GPT_num_real_solutions_system_l1897_189737

theorem num_real_solutions_system :
  ∃! (num_solutions : ℕ), 
  num_solutions = 5 ∧
  ∃ x y z w : ℝ, 
    (x = z + w + x * z) ∧ 
    (y = w + x + y * w) ∧ 
    (z = x + y + z * x) ∧ 
    (w = y + z + w * z) :=
sorry

end NUMINAMATH_GPT_num_real_solutions_system_l1897_189737


namespace NUMINAMATH_GPT_measure_of_alpha_l1897_189799

theorem measure_of_alpha
  (A B D α : ℝ)
  (hA : A = 50)
  (hB : B = 150)
  (hD : D = 140)
  (quadrilateral_sum : A + B + D + α = 360) : α = 20 :=
by
  rw [hA, hB, hD] at quadrilateral_sum
  sorry

end NUMINAMATH_GPT_measure_of_alpha_l1897_189799


namespace NUMINAMATH_GPT_find_m_l1897_189740

theorem find_m (m : ℝ) 
  (A : ℝ × ℝ := (-2, m))
  (B : ℝ × ℝ := (m, 4))
  (h_slope : ((B.snd - A.snd) / (B.fst - A.fst)) = -2) : 
  m = -8 :=
by 
  sorry

end NUMINAMATH_GPT_find_m_l1897_189740


namespace NUMINAMATH_GPT_days_of_supply_l1897_189792

-- Define the conditions as Lean definitions
def visits_per_day : ℕ := 3
def squares_per_visit : ℕ := 5
def total_rolls : ℕ := 1000
def squares_per_roll : ℕ := 300

-- Define the daily usage calculation
def daily_usage : ℕ := squares_per_visit * visits_per_day

-- Define the total squares calculation
def total_squares : ℕ := total_rolls * squares_per_roll

-- Define the proof statement for the number of days Bill's supply will last
theorem days_of_supply : (total_squares / daily_usage) = 20000 :=
by
  -- Placeholder for the actual proof, which is not required per instructions
  sorry

end NUMINAMATH_GPT_days_of_supply_l1897_189792


namespace NUMINAMATH_GPT_maximum_side_length_range_l1897_189718

variable (P : ℝ)
variable (a b c : ℝ)
variable (h1 : a + b + c = P)
variable (h2 : a ≤ b)
variable (h3 : b ≤ c)
variable (h4 : a + b > c)

theorem maximum_side_length_range : 
  (P / 3) ≤ c ∧ c < (P / 2) :=
by
  sorry

end NUMINAMATH_GPT_maximum_side_length_range_l1897_189718


namespace NUMINAMATH_GPT_problem_statement_l1897_189733

-- Define the universal set U, and sets A and B
def U : Set ℕ := { n | 1 ≤ n ∧ n ≤ 10 }
def A : Set ℕ := {1, 2, 3, 5, 8}
def B : Set ℕ := {1, 3, 5, 7, 9}

-- Define the complement of set A with respect to U
def complement_U_A : Set ℕ := { n | n ∈ U ∧ n ∉ A }

-- Define the intersection of complement_U_A and B
def intersection_complement_U_A_B : Set ℕ := { n | n ∈ complement_U_A ∧ n ∈ B }

-- Prove the given statement
theorem problem_statement : intersection_complement_U_A_B = {7, 9} := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1897_189733


namespace NUMINAMATH_GPT_trisha_total_distance_l1897_189721

theorem trisha_total_distance :
  let distance1 := 0.11
  let distance2 := 0.11
  let distance3 := 0.67
  distance1 + distance2 + distance3 = 0.89 :=
by
  sorry

end NUMINAMATH_GPT_trisha_total_distance_l1897_189721


namespace NUMINAMATH_GPT_sqrt_inequality_l1897_189778

theorem sqrt_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : 
  x^2 + y^2 + 1 ≤ Real.sqrt ((x^3 + y + 1) * (y^3 + x + 1)) :=
sorry

end NUMINAMATH_GPT_sqrt_inequality_l1897_189778


namespace NUMINAMATH_GPT_a_in_s_l1897_189707

-- Defining the sets and the condition
def S : Set ℕ := {1, 2}
def T (a : ℕ) : Set ℕ := {a}

-- The Lean theorem statement
theorem a_in_s (a : ℕ) (h : S ∪ T a = S) : a = 1 ∨ a = 2 := 
by 
  sorry

end NUMINAMATH_GPT_a_in_s_l1897_189707


namespace NUMINAMATH_GPT_conic_section_is_parabola_l1897_189732

-- Define the equation |y-3| = sqrt((x+4)^2 + y^2)
def equation (x y : ℝ) : Prop := |y - 3| = Real.sqrt ((x + 4) ^ 2 + y ^ 2)

-- The main theorem stating the conic section type is a parabola
theorem conic_section_is_parabola : ∀ x y : ℝ, equation x y → false := sorry

end NUMINAMATH_GPT_conic_section_is_parabola_l1897_189732


namespace NUMINAMATH_GPT_total_area_of_union_of_six_triangles_l1897_189722

theorem total_area_of_union_of_six_triangles :
  let s := 2 * Real.sqrt 2
  let area_one_triangle := (Real.sqrt 3 / 4) * s^2
  let total_area_without_overlaps := 6 * area_one_triangle
  let side_overlap := Real.sqrt 2
  let area_one_overlap := (Real.sqrt 3 / 4) * side_overlap ^ 2
  let total_overlap_area := 5 * area_one_overlap
  let net_area := total_area_without_overlaps - total_overlap_area
  net_area = 9.5 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_total_area_of_union_of_six_triangles_l1897_189722


namespace NUMINAMATH_GPT_drum_oil_ratio_l1897_189750

theorem drum_oil_ratio (C_X C_Y : ℝ) (h1 : (1 / 2) * C_X + (1 / 5) * C_Y = 0.45 * C_Y) : 
  C_Y / C_X = 2 :=
by
  -- Cannot provide the proof
  sorry

end NUMINAMATH_GPT_drum_oil_ratio_l1897_189750


namespace NUMINAMATH_GPT_five_x_minus_two_l1897_189769

theorem five_x_minus_two (x : ℚ) (h : 4 * x - 8 = 13 * x + 3) : 5 * (x - 2) = -145 / 9 := by
  sorry

end NUMINAMATH_GPT_five_x_minus_two_l1897_189769


namespace NUMINAMATH_GPT_train_speed_l1897_189724

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : length = 55) 
    (h2 : time = 5.5) 
    (h3 : speed = (length / time) * (3600 / 1000)) : 
    speed = 36 :=
sorry

end NUMINAMATH_GPT_train_speed_l1897_189724


namespace NUMINAMATH_GPT_value_of_f_2019_l1897_189783

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ)

-- Assumptions
axiom f_zero : f 0 = 2
axiom f_period : ∀ x : ℝ, f (x + 3) = -f x

-- The property to be proved
theorem value_of_f_2019 : f 2019 = -2 := sorry

end NUMINAMATH_GPT_value_of_f_2019_l1897_189783


namespace NUMINAMATH_GPT_evaluate_expression_l1897_189749

theorem evaluate_expression :
  71 * Real.sqrt (3 + 2 * Real.sqrt 2) - Real.sqrt (3 - 2 * Real.sqrt 2) = 72 + 70 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1897_189749


namespace NUMINAMATH_GPT_range_of_a_l1897_189754

-- Function definition for op
def op (x y : ℝ) : ℝ := x * (2 - y)

-- Predicate that checks the inequality for all t
def inequality_holds_for_all_t (a : ℝ) : Prop :=
  ∀ t : ℝ, (op (t - a) (t + a)) < 1

-- Prove that the range of a is (0, 2)
theorem range_of_a : 
  ∀ a : ℝ, inequality_holds_for_all_t a ↔ 0 < a ∧ a < 2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1897_189754


namespace NUMINAMATH_GPT_at_least_one_basketball_selected_l1897_189770

theorem at_least_one_basketball_selected (balls : Finset ℕ) (basketballs : Finset ℕ) (volleyballs : Finset ℕ) :
  basketballs.card = 6 → volleyballs.card = 2 → balls ⊆ (basketballs ∪ volleyballs) →
  balls.card = 3 → ∃ b ∈ balls, b ∈ basketballs :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_at_least_one_basketball_selected_l1897_189770
