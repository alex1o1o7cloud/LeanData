import Mathlib

namespace NUMINAMATH_GPT_simplify_expression_l1127_112790

theorem simplify_expression :
  (Real.sqrt (8^(1/3)) + Real.sqrt (17/4))^2 = (33 + 8 * Real.sqrt 17) / 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1127_112790


namespace NUMINAMATH_GPT_wash_cycle_time_l1127_112792

-- Definitions for the conditions
def num_loads : Nat := 8
def dry_cycle_time_minutes : Nat := 60
def total_time_hours : Nat := 14
def total_time_minutes : Nat := total_time_hours * 60

-- The actual statement we need to prove
theorem wash_cycle_time (x : Nat) (h : num_loads * x + num_loads * dry_cycle_time_minutes = total_time_minutes) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_wash_cycle_time_l1127_112792


namespace NUMINAMATH_GPT_total_local_percentage_approx_52_74_l1127_112768

-- We provide the conditions as definitions
def total_arts_students : ℕ := 400
def local_arts_percentage : ℝ := 0.50
def total_science_students : ℕ := 100
def local_science_percentage : ℝ := 0.25
def total_commerce_students : ℕ := 120
def local_commerce_percentage : ℝ := 0.85

-- Calculate the expected total percentage of local students
noncomputable def calculated_total_local_percentage : ℝ :=
  let local_arts_students := local_arts_percentage * total_arts_students
  let local_science_students := local_science_percentage * total_science_students
  let local_commerce_students := local_commerce_percentage * total_commerce_students
  let total_local_students := local_arts_students + local_science_students + local_commerce_students
  let total_students := total_arts_students + total_science_students + total_commerce_students
  (total_local_students / total_students) * 100

-- State what we need to prove
theorem total_local_percentage_approx_52_74 :
  abs (calculated_total_local_percentage - 52.74) < 1 :=
sorry

end NUMINAMATH_GPT_total_local_percentage_approx_52_74_l1127_112768


namespace NUMINAMATH_GPT_units_digit_of_m_squared_plus_two_to_m_is_3_l1127_112719

def m := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m_is_3 : (m^2 + 2^m) % 10 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_m_squared_plus_two_to_m_is_3_l1127_112719


namespace NUMINAMATH_GPT_cube_sum_gt_zero_l1127_112746

variable {x y z : ℝ}

theorem cube_sum_gt_zero (h1 : x < y) (h2 : y < z) : 
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 :=
sorry

end NUMINAMATH_GPT_cube_sum_gt_zero_l1127_112746


namespace NUMINAMATH_GPT_statement_D_incorrect_l1127_112786

theorem statement_D_incorrect (a b c : ℝ) : a^2 > b^2 ∧ a * b > 0 → ¬(1 / a < 1 / b) :=
by sorry

end NUMINAMATH_GPT_statement_D_incorrect_l1127_112786


namespace NUMINAMATH_GPT_train_speed_l1127_112703

theorem train_speed (length_train time_cross : ℝ)
  (h1 : length_train = 180)
  (h2 : time_cross = 9) : 
  (length_train / time_cross) * 3.6 = 72 :=
by
  -- This is just a placeholder proof. Replace with the actual proof.
  sorry

end NUMINAMATH_GPT_train_speed_l1127_112703


namespace NUMINAMATH_GPT_largest_primes_product_l1127_112730

theorem largest_primes_product : 7 * 97 * 997 = 679679 := by
  sorry

end NUMINAMATH_GPT_largest_primes_product_l1127_112730


namespace NUMINAMATH_GPT_multiple_of_12_l1127_112741

theorem multiple_of_12 (x : ℤ) : 
  (7 * x - 3) % 12 = 0 ↔ (x % 12 = 9 ∨ x % 12 = 1029 % 12) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_12_l1127_112741


namespace NUMINAMATH_GPT_minimal_S_n_l1127_112720

theorem minimal_S_n (a_n : ℕ → ℤ) 
  (h : ∀ n, a_n n = 3 * (n : ℤ) - 23) :
  ∃ n, (∀ m < n, (∀ k ≥ n, a_n k ≤ 0)) → n = 7 :=
by
  sorry

end NUMINAMATH_GPT_minimal_S_n_l1127_112720


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1127_112777

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (h_iso : A = B ∨ A = C ∨ B = C) (h_sum : A + B + C = 180) (h_one_angle : A = 50 ∨ B = 50 ∨ C = 50) :
  A = 80 ∨ B = 80 ∨ C = 80 ∨ A = 50 ∨ B = 50 ∨ C = 50 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l1127_112777


namespace NUMINAMATH_GPT_cube_add_constant_135002_l1127_112763

theorem cube_add_constant_135002 (n : ℤ) : 
  (∃ m : ℤ, m = n + 1 ∧ m^3 - n^3 = 135002) →
  (n = 149 ∨ n = -151) :=
by
  -- This is where the proof should go
  sorry

end NUMINAMATH_GPT_cube_add_constant_135002_l1127_112763


namespace NUMINAMATH_GPT_find_fx_l1127_112795

theorem find_fx (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = 19 * x ^ 2 + 55 * x - 44) :
  ∀ x : ℝ, f x = 19 * x ^ 2 + 93 * x + 30 :=
by
  sorry

end NUMINAMATH_GPT_find_fx_l1127_112795


namespace NUMINAMATH_GPT_sheets_of_paper_in_each_box_l1127_112723

theorem sheets_of_paper_in_each_box (S E : ℕ) 
  (h1 : S - E = 30)
  (h2 : 2 * E = S)
  (h3 : 3 * E = S - 10) :
  S = 40 :=
by
  sorry

end NUMINAMATH_GPT_sheets_of_paper_in_each_box_l1127_112723


namespace NUMINAMATH_GPT_angle_C_max_sum_of_sides_l1127_112739

theorem angle_C (a b c : ℝ) (S : ℝ) (h1 : S = (Real.sqrt 3 / 4) * (a^2 + b^2 - c^2)) :
  ∃ C : ℝ, C = Real.pi / 3 :=
by
  sorry

theorem max_sum_of_sides (a b : ℝ) (c : ℝ) (hC : c = Real.sqrt 3) :
  (a + b) ≤ 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_max_sum_of_sides_l1127_112739


namespace NUMINAMATH_GPT_age_solution_l1127_112760

theorem age_solution :
  ∃ me you : ℕ, me + you = 63 ∧ 
  ∃ x : ℕ, me = 2 * x ∧ you = x ∧ me = 36 ∧ you = 27 :=
by
  sorry

end NUMINAMATH_GPT_age_solution_l1127_112760


namespace NUMINAMATH_GPT_initial_mean_of_observations_l1127_112714

theorem initial_mean_of_observations (M : ℚ) (h : 50 * M + 11 = 50 * 36.5) : M = 36.28 := 
by
  sorry

end NUMINAMATH_GPT_initial_mean_of_observations_l1127_112714


namespace NUMINAMATH_GPT_mean_median_mode_l1127_112749

theorem mean_median_mode (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : m + 7 < n) 
  (h4 : (m + (m + 3) + (m + 7) + n + (n + 5) + (2 * n - 1)) / 6 = n)
  (h5 : ((m + 7) + n) / 2 = n)
  (h6 : (m+3 < m+7 ∧ m+7 = n ∧ n < n+5 ∧ n+5 < 2*n - 1 )) :
  m+n = 2*n := by
  sorry

end NUMINAMATH_GPT_mean_median_mode_l1127_112749


namespace NUMINAMATH_GPT_solve_for_x_l1127_112736

theorem solve_for_x (x : ℝ) (h1 : x ≠ -3) (h2 : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 5)) : x = -9 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1127_112736


namespace NUMINAMATH_GPT_secretary_work_hours_l1127_112776

theorem secretary_work_hours
  (x : ℕ)
  (h_ratio : 2 * x + 3 * x + 5 * x = 110) :
  5 * x = 55 := 
by
  sorry

end NUMINAMATH_GPT_secretary_work_hours_l1127_112776


namespace NUMINAMATH_GPT_yuri_total_puppies_l1127_112788

-- Conditions
def first_week_puppies := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies := first_week_puppies + 10

-- Total puppies
def total_puppies : ℕ := first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies

-- Theorem to prove
theorem yuri_total_puppies : total_puppies = 74 :=
by sorry

end NUMINAMATH_GPT_yuri_total_puppies_l1127_112788


namespace NUMINAMATH_GPT_trajectory_passes_quadrants_l1127_112789

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 4

-- Define the condition for a point to belong to the first quadrant
def in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

-- Define the condition for a point to belong to the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- State the theorem that the trajectory of point P passes through the first and second quadrants
theorem trajectory_passes_quadrants :
  (∃ x y : ℝ, circle_equation x y ∧ in_first_quadrant x y) ∧
  (∃ x y : ℝ, circle_equation x y ∧ in_second_quadrant x y) :=
sorry

end NUMINAMATH_GPT_trajectory_passes_quadrants_l1127_112789


namespace NUMINAMATH_GPT_lcm_of_two_numbers_l1127_112783
-- Importing the math library

-- Define constants and variables
variables (A B LCM HCF : ℕ)

-- Given conditions
def product_condition : Prop := A * B = 17820
def hcf_condition : Prop := HCF = 12
def lcm_condition : Prop := LCM = Nat.lcm A B

-- Theorem to prove
theorem lcm_of_two_numbers : product_condition A B ∧ hcf_condition HCF →
                              lcm_condition A B LCM →
                              LCM = 1485 := 
by
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_l1127_112783


namespace NUMINAMATH_GPT_number_of_distinct_cubes_l1127_112721

theorem number_of_distinct_cubes (w b : ℕ) (total_cubes : ℕ) (dim : ℕ) :
  w + b = total_cubes ∧ total_cubes = 8 ∧ dim = 2 ∧ w = 6 ∧ b = 2 →
  (number_of_distinct_orbits : ℕ) = 1 :=
by
  -- Conditions
  intros h
  -- Translation of conditions into a useful form
  let num_cubes := 8
  let distinct_configurations := 1
  -- Burnside's Lemma applied to find the distinct configurations
  sorry

end NUMINAMATH_GPT_number_of_distinct_cubes_l1127_112721


namespace NUMINAMATH_GPT_sum_consecutive_even_integers_l1127_112765

theorem sum_consecutive_even_integers (m : ℤ) :
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) = 5 * m + 20 := by
  sorry

end NUMINAMATH_GPT_sum_consecutive_even_integers_l1127_112765


namespace NUMINAMATH_GPT_angle_measure_l1127_112773

theorem angle_measure (x : ℝ) (h : 180 - x = (90 - x) - 4) : x = 60 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l1127_112773


namespace NUMINAMATH_GPT_intersect_at_four_points_l1127_112718

theorem intersect_at_four_points (a : ℝ) : 
  (∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = a^2) ∧ (p.2 = p.1^2 - a - 1) ∧ 
                 ∃ q : ℝ × ℝ, (q.1 ≠ p.1 ∧ q.2 ≠ p.2) ∧ (q.1^2 + q.2^2 = a^2) ∧ (q.2 = q.1^2 - a - 1) ∧ 
                 ∃ r : ℝ × ℝ, (r.1 ≠ p.1 ∧ r.1 ≠ q.1 ∧ r.2 ≠ p.2 ∧ r.2 ≠ q.2) ∧ (r.1^2 + r.2^2 = a^2) ∧ (r.2 = r.1^2 - a - 1) ∧
                 ∃ s : ℝ × ℝ, (s.1 ≠ p.1 ∧ s.1 ≠ q.1 ∧ s.1 ≠ r.1 ∧ s.2 ≠ p.2 ∧ s.2 ≠ q.2 ∧ s.2 ≠ r.2) ∧ (s.1^2 + s.2^2 = a^2) ∧ (s.2 = s.1^2 - a - 1))
  ↔ a > -1/2 := 
by 
  sorry

end NUMINAMATH_GPT_intersect_at_four_points_l1127_112718


namespace NUMINAMATH_GPT_probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l1127_112798

theorem probability_one_piece_is_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) : 
  (if (piece_lengths.1 = 2 ∧ piece_lengths.2 ≠ 2) ∨ (piece_lengths.1 ≠ 2 ∧ piece_lengths.2 = 2) then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 2 / 5 :=
sorry

theorem probability_both_pieces_longer_than_2m (stick_length : ℕ) (piece_lengths : ℕ × ℕ) (h1 : stick_length = 6) (h2 : piece_lengths.1 + piece_lengths.2 = stick_length) (h3 : piece_lengths.1 > 0 ∧ piece_lengths.2 > 0) :
  (if piece_lengths.1 > 2 ∧ piece_lengths.2 > 2 then 1 else 0) / 
  (if piece_lengths.1 > 0 ∧ piece_lengths.2 > 0 then 1 else 0) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_one_piece_is_2m_probability_both_pieces_longer_than_2m_l1127_112798


namespace NUMINAMATH_GPT_greatest_integer_l1127_112734

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 2) (h3 : ∃ l : ℕ, n = 8 * l - 4) : n = 124 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_l1127_112734


namespace NUMINAMATH_GPT_remainder_13_plus_x_l1127_112705

theorem remainder_13_plus_x (x : ℕ) (h1 : 7 * x % 31 = 1) : (13 + x) % 31 = 22 := 
by
  sorry

end NUMINAMATH_GPT_remainder_13_plus_x_l1127_112705


namespace NUMINAMATH_GPT_age_in_1900_l1127_112774

theorem age_in_1900 
  (x y : ℕ)
  (H1 : y = 29 * x)
  (H2 : 1901 ≤ y + x ∧ y + x ≤ 1930) :
  1900 - y = 44 := 
sorry

end NUMINAMATH_GPT_age_in_1900_l1127_112774


namespace NUMINAMATH_GPT_coffee_shop_spending_l1127_112742

variable (R S : ℝ)

theorem coffee_shop_spending (h1 : S = 0.60 * R) (h2 : R = S + 12.50) : R + S = 50 :=
by
  sorry

end NUMINAMATH_GPT_coffee_shop_spending_l1127_112742


namespace NUMINAMATH_GPT_largest_four_digit_negative_congruent_to_1_pmod_17_l1127_112778

theorem largest_four_digit_negative_congruent_to_1_pmod_17 :
  ∃ n : ℤ, 17 * n + 1 < -1000 ∧ 17 * n + 1 ≥ -9999 ∧ 17 * n + 1 ≡ 1 [ZMOD 17] := 
sorry

end NUMINAMATH_GPT_largest_four_digit_negative_congruent_to_1_pmod_17_l1127_112778


namespace NUMINAMATH_GPT_derivative_at_2_l1127_112775

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_at_2 : deriv f 2 = Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_GPT_derivative_at_2_l1127_112775


namespace NUMINAMATH_GPT_balls_into_boxes_l1127_112728

theorem balls_into_boxes :
  let n := 7 -- number of balls
  let k := 3 -- number of boxes
  let ways := Nat.choose (n + k - 1) (k - 1)
  ways = 36 :=
by
  sorry

end NUMINAMATH_GPT_balls_into_boxes_l1127_112728


namespace NUMINAMATH_GPT_b_share_1500_l1127_112724

theorem b_share_1500 (total_amount : ℕ) (parts_A parts_B parts_C : ℕ)
  (h_total_amount : total_amount = 4500)
  (h_ratio : (parts_A, parts_B, parts_C) = (2, 3, 4)) :
  parts_B * (total_amount / (parts_A + parts_B + parts_C)) = 1500 :=
by
  sorry

end NUMINAMATH_GPT_b_share_1500_l1127_112724


namespace NUMINAMATH_GPT_compute_b1c1_b2c2_b3c3_l1127_112758

theorem compute_b1c1_b2c2_b3c3 
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b1 * x + c1) * (x^2 + b2 * x + c2) * (x^2 + b3 * x + c3)) :
  b1 * c1 + b2 * c2 + b3 * c3 = -1 :=
by
  sorry

end NUMINAMATH_GPT_compute_b1c1_b2c2_b3c3_l1127_112758


namespace NUMINAMATH_GPT_constant_term_in_expansion_is_neg_42_l1127_112772

-- Define the general term formula for (x - 1/x)^8
def binomial_term (r : ℕ) : ℤ :=
  (Nat.choose 8 r) * (-1 : ℤ) ^ r

-- Define the constant term in the product expansion
def constant_term : ℤ := 
  binomial_term 4 - 2 * binomial_term 5 

-- Problem statement: Prove the constant term is -42
theorem constant_term_in_expansion_is_neg_42 :
  constant_term = -42 := 
sorry

end NUMINAMATH_GPT_constant_term_in_expansion_is_neg_42_l1127_112772


namespace NUMINAMATH_GPT_identify_1000g_weight_l1127_112799

-- Define the masses of the weights
def masses : List ℕ := [1000, 1001, 1002, 1004, 1007]

-- The statement that needs to be proven
theorem identify_1000g_weight (masses : List ℕ) (h : masses = [1000, 1001, 1002, 1004, 1007]) :
  ∃ w, w ∈ masses ∧ w = 1000 ∧ by sorry :=
sorry

end NUMINAMATH_GPT_identify_1000g_weight_l1127_112799


namespace NUMINAMATH_GPT_interest_earned_is_91_dollars_l1127_112729

-- Define the initial conditions
def P : ℝ := 2000
def r : ℝ := 0.015
def n : ℕ := 3

-- Define the compounded amount function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Prove the interest earned after 3 years is 91 dollars
theorem interest_earned_is_91_dollars : 
  (compound_interest P r n) - P = 91 :=
by
  sorry

end NUMINAMATH_GPT_interest_earned_is_91_dollars_l1127_112729


namespace NUMINAMATH_GPT_min_value_b_over_a_l1127_112755

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
  Real.log x + (Real.exp 1 - a) * x - b

theorem min_value_b_over_a 
  (a b : ℝ)
  (h_cond : ∀ x > 0, f x a b ≤ 0)
  (h_b : b = -1 - Real.log (a - Real.exp 1)) 
  (h_a_gt_e : a > Real.exp 1) :
  ∃ (x : ℝ), x = 2 * Real.exp 1 ∧ (b / a) = - (1 / Real.exp 1) := 
sorry

end NUMINAMATH_GPT_min_value_b_over_a_l1127_112755


namespace NUMINAMATH_GPT_stratified_sampling_grade10_students_l1127_112702

-- Definitions based on the given problem
def total_students := 900
def grade10_students := 300
def sample_size := 45

-- Calculation of the number of Grade 10 students in the sample
theorem stratified_sampling_grade10_students : (grade10_students * sample_size) / total_students = 15 := by
  sorry

end NUMINAMATH_GPT_stratified_sampling_grade10_students_l1127_112702


namespace NUMINAMATH_GPT_sum_constants_l1127_112710

theorem sum_constants (a b x : ℝ) 
  (h1 : (x - a) / (x + b) = (x^2 - 50 * x + 621) / (x^2 + 75 * x - 3400))
  (h2 : x^2 - 50 * x + 621 = (x - 27) * (x - 23))
  (h3 : x^2 + 75 * x - 3400 = (x - 40) * (x + 85)) :
  a + b = 112 :=
sorry

end NUMINAMATH_GPT_sum_constants_l1127_112710


namespace NUMINAMATH_GPT_airplane_cost_correct_l1127_112796

-- Define the conditions
def initial_amount : ℝ := 5.00
def change_received : ℝ := 0.72

-- Define the cost calculation
def airplane_cost (initial : ℝ) (change : ℝ) : ℝ := initial - change

-- Prove that the airplane cost is $4.28 given the conditions
theorem airplane_cost_correct : airplane_cost initial_amount change_received = 4.28 :=
by
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_airplane_cost_correct_l1127_112796


namespace NUMINAMATH_GPT_constant_a_value_l1127_112751

theorem constant_a_value (S : ℕ → ℝ)
  (a : ℝ)
  (h : ∀ n : ℕ, S n = 3 ^ (n + 1) + a) :
  a = -3 :=
sorry

end NUMINAMATH_GPT_constant_a_value_l1127_112751


namespace NUMINAMATH_GPT_next_perfect_square_l1127_112712

theorem next_perfect_square (x : ℤ) (h : ∃ k : ℤ, x = k^2) : ∃ z : ℤ, z = x + 2 * Int.sqrt x + 1 :=
by
  sorry

end NUMINAMATH_GPT_next_perfect_square_l1127_112712


namespace NUMINAMATH_GPT_greatest_expression_l1127_112733

theorem greatest_expression 
  (x1 x2 y1 y2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : x1 < x2) 
  (hx12 : x1 + x2 = 1) 
  (hy1 : 0 < y1) 
  (hy2 : y1 < y2) 
  (hy12 : y1 + y2 = 1) : 
  x1 * y1 + x2 * y2 > max (x1 * x2 + y1 * y2) (max (x1 * y2 + x2 * y1) (1/2)) := 
sorry

end NUMINAMATH_GPT_greatest_expression_l1127_112733


namespace NUMINAMATH_GPT_gray_part_area_l1127_112706

theorem gray_part_area (area_rect1 area_rect2 area_black area_white gray_part_area : ℕ)
  (h_rect1 : area_rect1 = 80)
  (h_rect2 : area_rect2 = 108)
  (h_black : area_black = 37)
  (h_white : area_white = area_rect1 - area_black)
  (h_white_correct : area_white = 43)
  : gray_part_area = area_rect2 - area_white :=
by
  sorry

end NUMINAMATH_GPT_gray_part_area_l1127_112706


namespace NUMINAMATH_GPT_probability_of_snow_at_least_once_l1127_112708

/-- Probability of snow during the first week of January -/
theorem probability_of_snow_at_least_once :
  let p_no_snow_first_four_days := (3/4 : ℚ)
  let p_no_snow_next_three_days := (2/3 : ℚ)
  let p_no_snow_first_week := p_no_snow_first_four_days^4 * p_no_snow_next_three_days^3
  let p_snow_at_least_once := 1 - p_no_snow_first_week
  p_snow_at_least_once = 68359 / 100000 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_snow_at_least_once_l1127_112708


namespace NUMINAMATH_GPT_curve_product_l1127_112722

theorem curve_product (a b : ℝ) (h1 : 8 * a + 2 * b = 2) (h2 : 12 * a + b = 9) : a * b = -3 := by
  sorry

end NUMINAMATH_GPT_curve_product_l1127_112722


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1127_112780

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 13) 
  (h2 : (5 * (a 1 + a 5)) / 2 = 35) 
  (h_arithmetic_sequence : ∀ n, a (n+1) = a n + d) : 
  d = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1127_112780


namespace NUMINAMATH_GPT_arithmetic_sequence_values_l1127_112725

noncomputable def common_difference (a₁ a₂ : ℕ) : ℕ := (a₂ - a₁) / 2

theorem arithmetic_sequence_values (x y z d: ℕ) 
    (h₁: d = common_difference 7 11) 
    (h₂: x = 7 + d) 
    (h₃: y = 11 + d) 
    (h₄: z = y + d): 
    x = 9 ∧ y = 13 ∧ z = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_values_l1127_112725


namespace NUMINAMATH_GPT_total_hits_and_misses_l1127_112757

theorem total_hits_and_misses (h : ℕ) (m : ℕ) (hc : m = 3 * h) (hm : m = 50) : h + m = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_hits_and_misses_l1127_112757


namespace NUMINAMATH_GPT_range_of_a_l1127_112793

theorem range_of_a {a : ℝ} : (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≤ -x2^2 + 4*a*x2)
  ∨ (∀ x1 x2 : ℝ, 2 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 4 → -x1^2 + 4*a*x1 ≥ -x2^2 + 4*a*x2) ↔ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1127_112793


namespace NUMINAMATH_GPT_randy_trip_length_l1127_112794

-- Define the conditions
noncomputable def fraction_gravel := (1/4 : ℚ)
noncomputable def miles_pavement := (30 : ℚ)
noncomputable def fraction_dirt := (1/6 : ℚ)

-- The proof statement
theorem randy_trip_length :
  ∃ x : ℚ, (fraction_gravel + fraction_dirt + (miles_pavement / x) = 1) ∧ x = 360 / 7 := 
by
  sorry

end NUMINAMATH_GPT_randy_trip_length_l1127_112794


namespace NUMINAMATH_GPT_nathan_weeks_l1127_112785

-- Define the conditions as per the problem
def hours_per_day_nathan : ℕ := 3
def days_per_week : ℕ := 7
def hours_per_week_nathan : ℕ := hours_per_day_nathan * days_per_week
def hours_per_day_tobias : ℕ := 5
def hours_one_week_tobias : ℕ := hours_per_day_tobias * days_per_week
def total_hours : ℕ := 77

-- The number of weeks Nathan played
def weeks_nathan (w : ℕ) : Prop :=
  hours_per_week_nathan * w + hours_one_week_tobias = total_hours

-- Prove the number of weeks Nathan played is 2
theorem nathan_weeks : ∃ w : ℕ, weeks_nathan w ∧ w = 2 :=
by
  use 2
  sorry

end NUMINAMATH_GPT_nathan_weeks_l1127_112785


namespace NUMINAMATH_GPT_proof_statement_d_is_proposition_l1127_112781

-- Define the conditions
def statement_a := "Do two points determine a line?"
def statement_b := "Take a point M on line AB"
def statement_c := "In the same plane, two lines do not intersect"
def statement_d := "The sum of two acute angles is greater than a right angle"

-- Define the property of being a proposition
def is_proposition (s : String) : Prop :=
  s ≠ "Do two points determine a line?" ∧
  s ≠ "Take a point M on line AB" ∧
  s ≠ "In the same plane, two lines do not intersect"

-- The equivalence proof that statement_d is the only proposition
theorem proof_statement_d_is_proposition :
  is_proposition statement_d ∧
  ¬is_proposition statement_a ∧
  ¬is_proposition statement_b ∧
  ¬is_proposition statement_c := by
  sorry

end NUMINAMATH_GPT_proof_statement_d_is_proposition_l1127_112781


namespace NUMINAMATH_GPT_angle_D_is_20_degrees_l1127_112709

theorem angle_D_is_20_degrees (A B C D : ℝ) 
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 40)
  (h4 : B + C = 160) : D = 20 :=
by
  sorry

end NUMINAMATH_GPT_angle_D_is_20_degrees_l1127_112709


namespace NUMINAMATH_GPT_pawpaws_basket_l1127_112759

variable (total_fruits mangoes pears lemons kiwis : ℕ)
variable (pawpaws : ℕ)

theorem pawpaws_basket
  (h1 : total_fruits = 58)
  (h2 : mangoes = 18)
  (h3 : pears = 10)
  (h4 : lemons = 9)
  (h5 : kiwis = 9)
  (h6 : total_fruits = mangoes + pears + lemons + kiwis + pawpaws) :
  pawpaws = 12 := by
  sorry

end NUMINAMATH_GPT_pawpaws_basket_l1127_112759


namespace NUMINAMATH_GPT_product_of_differences_l1127_112715

-- Define the context where x and y are real numbers
variables (x y : ℝ)

-- State the theorem to be proved
theorem product_of_differences (x y : ℝ) : 
  (-x + y) * (-x - y) = x^2 - y^2 :=
sorry

end NUMINAMATH_GPT_product_of_differences_l1127_112715


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l1127_112727

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 + r2 = 17) (h2 : r1 * r2 = 8) :
  1 / r1 + 1 / r2 = 17 / 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l1127_112727


namespace NUMINAMATH_GPT_mary_take_home_pay_l1127_112707

def hourly_wage : ℝ := 8
def regular_hours : ℝ := 20
def first_overtime_hours : ℝ := 10
def second_overtime_hours : ℝ := 10
def third_overtime_hours : ℝ := 10
def remaining_overtime_hours : ℝ := 20
def social_security_tax_rate : ℝ := 0.08
def medicare_tax_rate : ℝ := 0.02
def insurance_premium : ℝ := 50

def regular_earnings := regular_hours * hourly_wage
def first_overtime_earnings := first_overtime_hours * (hourly_wage * 1.25)
def second_overtime_earnings := second_overtime_hours * (hourly_wage * 1.5)
def third_overtime_earnings := third_overtime_hours * (hourly_wage * 1.75)
def remaining_overtime_earnings := remaining_overtime_hours * (hourly_wage * 2)

def total_earnings := 
    regular_earnings + 
    first_overtime_earnings + 
    second_overtime_earnings + 
    third_overtime_earnings + 
    remaining_overtime_earnings

def social_security_tax := total_earnings * social_security_tax_rate
def medicare_tax := total_earnings * medicare_tax_rate
def total_taxes := social_security_tax + medicare_tax

def earnings_after_taxes := total_earnings - total_taxes
def earnings_take_home := earnings_after_taxes - insurance_premium

theorem mary_take_home_pay : earnings_take_home = 706 := by
  sorry

end NUMINAMATH_GPT_mary_take_home_pay_l1127_112707


namespace NUMINAMATH_GPT_imo_1989_q6_l1127_112797

-- Define the odd integer m greater than 2
def isOdd (m : ℕ) := ∃ k : ℤ, m = 2 * k + 1

-- Define the condition for divisibility
def smallest_n (m : ℕ) (k : ℕ) (p : ℕ) : ℕ :=
  if k ≤ 1989 then 2 ^ (1989 - k) else 1

theorem imo_1989_q6 
  (m : ℕ) (h_m_gt2 : m > 2) (h_m_odd : isOdd m) (k : ℕ) (p : ℕ) (h_m_form : m = 2^k * p - 1) (h_p_odd : isOdd p) (h_k_gt1 : k > 1) :
  ∃ n : ℕ, (2^1989 ∣ m^n - 1) ∧ n = smallest_n m k p :=
by
  sorry

end NUMINAMATH_GPT_imo_1989_q6_l1127_112797


namespace NUMINAMATH_GPT_speed_of_train_approx_29_0088_kmh_l1127_112743

noncomputable def speed_of_train_in_kmh := 
  let length_train : ℝ := 288
  let length_bridge : ℝ := 101
  let time_seconds : ℝ := 48.29
  let total_distance : ℝ := length_train + length_bridge
  let speed_m_per_s : ℝ := total_distance / time_seconds
  speed_m_per_s * 3.6

theorem speed_of_train_approx_29_0088_kmh :
  abs (speed_of_train_in_kmh - 29.0088) < 0.001 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_train_approx_29_0088_kmh_l1127_112743


namespace NUMINAMATH_GPT_trick_or_treat_hours_l1127_112769

variable (num_children : ℕ)
variable (houses_per_hour : ℕ)
variable (treats_per_house_per_kid : ℕ)
variable (total_treats : ℕ)

theorem trick_or_treat_hours (h : num_children = 3)
  (h1 : houses_per_hour = 5)
  (h2 : treats_per_house_per_kid = 3)
  (h3 : total_treats = 180) :
  total_treats / (num_children * houses_per_hour * treats_per_house_per_kid) = 4 :=
by
  sorry

end NUMINAMATH_GPT_trick_or_treat_hours_l1127_112769


namespace NUMINAMATH_GPT_percentage_paid_l1127_112770

theorem percentage_paid (X Y : ℝ) (h_sum : X + Y = 572) (h_Y : Y = 260) : (X / Y) * 100 = 120 :=
by
  -- We'll prove this result by using the conditions and solving for X.
  sorry

end NUMINAMATH_GPT_percentage_paid_l1127_112770


namespace NUMINAMATH_GPT_find_ϕ_l1127_112779

noncomputable def f (ω ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)

theorem find_ϕ (ω ϕ : ℝ) (h1 : 0 < ω) (h2 : abs ϕ < Real.pi / 2) (h3 : ∀ x : ℝ, f ω ϕ (x + Real.pi / 6) = g ω x) 
  (h4 : 2 * Real.pi / ω = Real.pi) : ϕ = Real.pi / 3 :=
by sorry

end NUMINAMATH_GPT_find_ϕ_l1127_112779


namespace NUMINAMATH_GPT_rectangular_solid_volume_l1127_112754

theorem rectangular_solid_volume
  (a b c : ℝ)
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : b = 2 * a) :
  a * b * c = 12 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_volume_l1127_112754


namespace NUMINAMATH_GPT_natalie_needs_12_bushes_for_60_zucchinis_l1127_112732

-- Definitions based on problem conditions
def bushes_to_containers (bushes : ℕ) : ℕ := bushes * 10
def containers_to_zucchinis (containers : ℕ) : ℕ := (containers * 3) / 6

-- Theorem statement
theorem natalie_needs_12_bushes_for_60_zucchinis : 
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) = 60 ∧ bushes = 12 := by
  sorry

end NUMINAMATH_GPT_natalie_needs_12_bushes_for_60_zucchinis_l1127_112732


namespace NUMINAMATH_GPT_perpendicular_lines_parallel_lines_l1127_112767

-- Define the lines l1 and l2 in terms of a
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + a ^ 2 - 1 = 0

-- Define the perpendicular condition
def perp (a : ℝ) : Prop :=
  a * 1 + 2 * (a - 1) = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop :=
  a / 1 = 2 / (a - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perp a → a = 2 / 3 := by
  intro h
  sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 := by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_lines_parallel_lines_l1127_112767


namespace NUMINAMATH_GPT_graduation_problem_l1127_112740

def valid_xs : List ℕ :=
  [10, 12, 15, 18, 20, 24, 30]

noncomputable def sum_valid_xs (l : List ℕ) : ℕ :=
  l.foldr (λ x sum => x + sum) 0

theorem graduation_problem :
  sum_valid_xs valid_xs = 129 :=
by
  sorry

end NUMINAMATH_GPT_graduation_problem_l1127_112740


namespace NUMINAMATH_GPT_sword_length_difference_l1127_112787

def christopher_sword := 15.0
def jameson_sword := 2 * christopher_sword + 3
def june_sword := jameson_sword + 5
def average_length := (christopher_sword + jameson_sword + june_sword) / 3
def laura_sword := average_length - 0.1 * average_length
def difference := june_sword - laura_sword

theorem sword_length_difference :
  difference = 12.197 := 
sorry

end NUMINAMATH_GPT_sword_length_difference_l1127_112787


namespace NUMINAMATH_GPT_ambulance_ride_cost_correct_l1127_112771

noncomputable def total_bill : ℝ := 12000
noncomputable def medication_percentage : ℝ := 0.40
noncomputable def imaging_tests_percentage : ℝ := 0.15
noncomputable def surgical_procedure_percentage : ℝ := 0.20
noncomputable def overnight_stays_percentage : ℝ := 0.25
noncomputable def food_cost : ℝ := 300
noncomputable def consultation_fee : ℝ := 80

noncomputable def ambulance_ride_cost := total_bill - (food_cost + consultation_fee)

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 11620 :=
by
  sorry

end NUMINAMATH_GPT_ambulance_ride_cost_correct_l1127_112771


namespace NUMINAMATH_GPT_product_increased_five_times_l1127_112704

variables (A B : ℝ)

theorem product_increased_five_times (h : A * B = 1.6) : (5 * A) * (5 * B) = 40 :=
by
  sorry

end NUMINAMATH_GPT_product_increased_five_times_l1127_112704


namespace NUMINAMATH_GPT_room_length_perimeter_ratio_l1127_112713

theorem room_length_perimeter_ratio :
  ∀ (L W : ℕ), L = 19 → W = 11 → (L : ℚ) / (2 * (L + W)) = 19 / 60 := by
  intros L W hL hW
  sorry

end NUMINAMATH_GPT_room_length_perimeter_ratio_l1127_112713


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_simple_sequence_general_term_l1127_112744

-- Question 1
theorem geometric_sequence_common_ratio (a_3 : ℝ) (S_3 : ℝ) (q : ℝ) (h1 : a_3 = 3 / 2) (h2 : S_3 = 9 / 2) :
    q = -1 / 2 ∨ q = 1 :=
sorry

-- Question 2
theorem simple_sequence_general_term (S : ℕ → ℝ) (a : ℕ → ℝ) (h : ∀ n, S n = n^2) :
    ∀ n, a n = S n - S (n - 1) → ∀ n, a n = 2 * n - 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_simple_sequence_general_term_l1127_112744


namespace NUMINAMATH_GPT_find_z_l1127_112761

variable (x y z : ℝ)

theorem find_z (h1 : 12 * 40 = 480)
    (h2 : 15 * 50 = 750)
    (h3 : x + y + z = 270)
    (h4 : x + y = 100) :
    z = 170 := by
  sorry

end NUMINAMATH_GPT_find_z_l1127_112761


namespace NUMINAMATH_GPT_find_x_l1127_112700

noncomputable def f (x : ℝ) : ℝ := 5 * x^3 - 3

theorem find_x (x : ℝ) : (f⁻¹ (-2) = x) → x = -43 := by
  sorry

end NUMINAMATH_GPT_find_x_l1127_112700


namespace NUMINAMATH_GPT_speed_ratio_is_2_l1127_112745

def distance_to_work : ℝ := 20
def total_hours_on_road : ℝ := 6
def speed_back_home : ℝ := 10

theorem speed_ratio_is_2 :
  (∃ v : ℝ, (20 / v) + (20 / 10) = 6) → (10 = 2 * v) :=
by sorry

end NUMINAMATH_GPT_speed_ratio_is_2_l1127_112745


namespace NUMINAMATH_GPT_matt_books_second_year_l1127_112764

-- Definitions based on the conditions
variables (M : ℕ) -- number of books Matt read last year
variables (P : ℕ) -- number of books Pete read last year

-- Pete read twice as many books as Matt last year
def pete_read_last_year (M : ℕ) : ℕ := 2 * M

-- This year, Pete doubles the number of books he read last year
def pete_read_this_year (M : ℕ) : ℕ := 2 * (2 * M)

-- Matt reads 50% more books this year than he did last year
def matt_read_this_year (M : ℕ) : ℕ := M + M / 2

-- Pete read 300 books across both years
def total_books_pete_read_last_and_this_year (M : ℕ) : ℕ :=
  pete_read_last_year M + pete_read_this_year M

-- Prove that Matt read 75 books in his second year
theorem matt_books_second_year (M : ℕ) (h : total_books_pete_read_last_and_this_year M = 300) :
  matt_read_this_year M = 75 :=
by sorry

end NUMINAMATH_GPT_matt_books_second_year_l1127_112764


namespace NUMINAMATH_GPT_range_of_a_l1127_112784

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : 4 ≤ a := 
sorry

end NUMINAMATH_GPT_range_of_a_l1127_112784


namespace NUMINAMATH_GPT_objective_function_range_l1127_112701

theorem objective_function_range (x y : ℝ) 
  (h1 : x + 2 * y > 2) 
  (h2 : 2 * x + y ≤ 4) 
  (h3 : 4 * x - y ≥ 1) : 
  ∃ z_min z_max : ℝ, (∀ z : ℝ, z = 3 * x + y → z_min ≤ z ∧ z ≤ z_max) ∧ z_min = 1 ∧ z_max = 6 := 
sorry

end NUMINAMATH_GPT_objective_function_range_l1127_112701


namespace NUMINAMATH_GPT_root_in_interval_sum_eq_three_l1127_112726

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval_sum_eq_three {a b : ℤ} (h1 : b - a = 1) (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) :
  a + b = 3 :=
by
  sorry

end NUMINAMATH_GPT_root_in_interval_sum_eq_three_l1127_112726


namespace NUMINAMATH_GPT_intersection_complement_correct_l1127_112766

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A based on the condition given
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 3}

-- Define set B based on the condition given
def B : Set ℝ := {x | x > 3}

-- Define the complement of set B in the universal set U
def compl_B : Set ℝ := {x | x ≤ 3}

-- Define the expected result of A ∩ compl_B
def expected_result : Set ℝ := {x | x ≤ -3} ∪ {3}

-- State the theorem to be proven
theorem intersection_complement_correct :
  (A ∩ compl_B) = expected_result :=
sorry

end NUMINAMATH_GPT_intersection_complement_correct_l1127_112766


namespace NUMINAMATH_GPT_simplify_complex_expression_l1127_112735

theorem simplify_complex_expression (i : ℂ) (h : i^2 = -1) : 
  7 * (4 - 2 * i) - 2 * i * (3 - 4 * i) = 20 - 20 * i := 
by
  sorry

end NUMINAMATH_GPT_simplify_complex_expression_l1127_112735


namespace NUMINAMATH_GPT_no_rational_root_l1127_112711

theorem no_rational_root (x : ℚ) : 3 * x^4 - 2 * x^3 - 8 * x^2 + x + 1 ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_no_rational_root_l1127_112711


namespace NUMINAMATH_GPT_ball_box_problem_l1127_112747

theorem ball_box_problem : 
  let num_balls := 5
  let num_boxes := 4
  (num_boxes ^ num_balls) = 1024 := by
  sorry

end NUMINAMATH_GPT_ball_box_problem_l1127_112747


namespace NUMINAMATH_GPT_peach_tree_average_production_l1127_112716

-- Definitions derived from the conditions
def num_apple_trees : ℕ := 30
def kg_per_apple_tree : ℕ := 150
def num_peach_trees : ℕ := 45
def total_mass_fruit : ℕ := 7425

-- Main Statement to be proven
theorem peach_tree_average_production : 
  (total_mass_fruit - (num_apple_trees * kg_per_apple_tree)) = (num_peach_trees * 65) :=
by
  sorry

end NUMINAMATH_GPT_peach_tree_average_production_l1127_112716


namespace NUMINAMATH_GPT_problem_statement_negation_statement_l1127_112738

variable {a b : ℝ}

theorem problem_statement (h : a * b ≤ 0) : a ≤ 0 ∨ b ≤ 0 :=
sorry

theorem negation_statement (h : a * b > 0) : a > 0 ∧ b > 0 :=
sorry

end NUMINAMATH_GPT_problem_statement_negation_statement_l1127_112738


namespace NUMINAMATH_GPT_find_y_l1127_112782

theorem find_y (x y : ℤ) (h1 : x = 4) (h2 : 3 * x + 2 * y = 30) : y = 9 :=
by
  subst h1
  have h : 3 * 4 + 2 * y = 30 := by rw [h2]
  linarith

end NUMINAMATH_GPT_find_y_l1127_112782


namespace NUMINAMATH_GPT_total_candy_eaten_by_bobby_l1127_112717

-- Definitions based on the problem conditions
def candy_eaten_by_bobby_round1 : ℕ := 28
def candy_eaten_by_bobby_round2 : ℕ := 42
def chocolate_eaten_by_bobby : ℕ := 63

-- Define the statement to prove
theorem total_candy_eaten_by_bobby : 
  candy_eaten_by_bobby_round1 + candy_eaten_by_bobby_round2 + chocolate_eaten_by_bobby = 133 :=
  by
  -- Skipping the proof itself
  sorry

end NUMINAMATH_GPT_total_candy_eaten_by_bobby_l1127_112717


namespace NUMINAMATH_GPT_ruby_height_l1127_112753

variable (Ruby Pablo Charlene Janet : ℕ)

theorem ruby_height :
  (Ruby = Pablo - 2) →
  (Pablo = Charlene + 70) →
  (Janet = 62) →
  (Charlene = 2 * Janet) →
  Ruby = 192 := 
by
  sorry

end NUMINAMATH_GPT_ruby_height_l1127_112753


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1127_112756

variable {a x : ℝ} (h_neg : a < 0)

theorem solution_set_of_quadratic_inequality :
  (a * x^2 - (a + 2) * x + 2) ≥ 0 ↔ (x ∈ Set.Icc (2 / a) 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1127_112756


namespace NUMINAMATH_GPT_hoseok_position_reversed_l1127_112748

def nine_people (P : ℕ → Prop) : Prop :=
  P 1 ∧ P 2 ∧ P 3 ∧ P 4 ∧ P 5 ∧ P 6 ∧ P 7 ∧ P 8 ∧ P 9

variable (h : ℕ → Prop)

def hoseok_front_foremost : Prop :=
  nine_people h ∧ h 1 -- Hoseok is at the forefront and is the shortest

theorem hoseok_position_reversed :
  hoseok_front_foremost h → h 9 :=
by 
  sorry

end NUMINAMATH_GPT_hoseok_position_reversed_l1127_112748


namespace NUMINAMATH_GPT_percentage_increase_is_correct_l1127_112752

-- Define the original and new weekly earnings
def original_earnings : ℕ := 60
def new_earnings : ℕ := 90

-- Define the percentage increase calculation
def percentage_increase (original new : ℕ) : Rat := ((new - original) / original: Rat) * 100

-- State the theorem that the percentage increase is 50%
theorem percentage_increase_is_correct : percentage_increase original_earnings new_earnings = 50 := 
sorry

end NUMINAMATH_GPT_percentage_increase_is_correct_l1127_112752


namespace NUMINAMATH_GPT_find_x_y_l1127_112750

theorem find_x_y (x y : ℤ) (h1 : 3 * x - 482 = 2 * y) (h2 : 7 * x + 517 = 5 * y) :
  x = 3444 ∧ y = 4925 :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l1127_112750


namespace NUMINAMATH_GPT_parabola_pass_through_fixed_point_l1127_112791

theorem parabola_pass_through_fixed_point
  (p : ℝ) (hp : p > 0)
  (xM yM : ℝ) (hM : (xM, yM) = (1, -2))
  (hMp : yM^2 = 2 * p * xM)
  (xA yA xC yC xB yB xD yD : ℝ)
  (hxA : xA = xC ∨ xA ≠ xC)
  (hxB : xB = xD ∨ xB ≠ xD)
  (x2 y0 : ℝ) (h : (x2, y0) = (2, 0))
  (m1 m2 : ℝ) (hm1m2 : m1 * m2 = -1)
  (l1_intersect_A : xA = m1 * yA + 2)
  (l1_intersect_C : xC = m1 * yC + 2)
  (l2_intersect_B : xB = m2 * yB + 2)
  (l2_intersect_D : xD = m2 * yD + 2)
  (hMidM : (2 * xA + 2 * xC = 4 * xM ∧ 2 * yA + 2 * yC = 4 * yM))
  (hMidN : (2 * xB + 2 * xD = 4 * xM ∧ 2 * yB + 2 * yD = 4 * yM)) :
  (yM^2 = 4 * xM) ∧ 
  (∃ k : ℝ, ∀ x : ℝ, y = k * x ↔ y = xM / (m1 + m2) ∧ y = m1) :=
sorry

end NUMINAMATH_GPT_parabola_pass_through_fixed_point_l1127_112791


namespace NUMINAMATH_GPT_greatest_five_digit_common_multiple_l1127_112731

theorem greatest_five_digit_common_multiple (n : ℕ) :
  (n % 18 = 0) ∧ (10000 ≤ n) ∧ (n ≤ 99999) → n = 99990 :=
by
  sorry

end NUMINAMATH_GPT_greatest_five_digit_common_multiple_l1127_112731


namespace NUMINAMATH_GPT_snack_eaters_remaining_l1127_112762

theorem snack_eaters_remaining 
  (initial_population : ℕ)
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (first_half_leave : ℕ)
  (new_outsiders_2 : ℕ)
  (second_leave : ℕ)
  (final_half_leave : ℕ) 
  (h_initial_population : initial_population = 200)
  (h_initial_snackers : initial_snackers = 100)
  (h_new_outsiders_1 : new_outsiders_1 = 20)
  (h_first_half_leave : first_half_leave = (initial_snackers + new_outsiders_1) / 2)
  (h_new_outsiders_2 : new_outsiders_2 = 10)
  (h_second_leave : second_leave = 30)
  (h_final_half_leave : final_half_leave = (first_half_leave + new_outsiders_2 - second_leave) / 2) : 
  final_half_leave = 20 := 
sorry

end NUMINAMATH_GPT_snack_eaters_remaining_l1127_112762


namespace NUMINAMATH_GPT_only_positive_integer_cube_less_than_triple_l1127_112737

theorem only_positive_integer_cube_less_than_triple (n : ℕ) (h : 0 < n ∧ n^3 < 3 * n) : n = 1 :=
sorry

end NUMINAMATH_GPT_only_positive_integer_cube_less_than_triple_l1127_112737
