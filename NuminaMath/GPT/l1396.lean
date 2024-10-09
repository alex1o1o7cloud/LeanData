import Mathlib

namespace portion_left_l1396_139630

theorem portion_left (john_portion emma_portion final_portion : ℝ) (H1 : john_portion = 0.6) (H2 : emma_portion = 0.5 * (1 - john_portion)) :
  final_portion = 1 - john_portion - emma_portion :=
by
  sorry

end portion_left_l1396_139630


namespace jason_arms_tattoos_l1396_139637

variable (x : ℕ)

def jason_tattoos (x : ℕ) : ℕ := 2 * x + 3 * 2

def adam_tattoos (x : ℕ) : ℕ := 3 + 2 * (jason_tattoos x)

theorem jason_arms_tattoos : adam_tattoos x = 23 → x = 2 := by
  intro h
  sorry

end jason_arms_tattoos_l1396_139637


namespace largest_positive_integer_n_l1396_139695

theorem largest_positive_integer_n :
  ∃ n : ℕ, (∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≥ 2 / n) ∧ (∀ m : ℕ, m > n → ∃ x : ℝ, (Real.sin x) ^ m + (Real.cos x) ^ m < 2 / m) :=
sorry

end largest_positive_integer_n_l1396_139695


namespace lcm_100_40_is_200_l1396_139694

theorem lcm_100_40_is_200 : Nat.lcm 100 40 = 200 := by
  sorry

end lcm_100_40_is_200_l1396_139694


namespace exist_amusing_numbers_l1396_139658

/-- Definitions for an amusing number -/
def is_amusing (x : ℕ) : Prop :=
  (x >= 1000) ∧ (x <= 9999) ∧
  ∃ y : ℕ, y ≠ x ∧
  ((∀ d ∈ [x / 1000 % 10, x / 100 % 10, x / 10 % 10, x % 10],
    (d ≠ 0 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]) ∧
    (d ≠ 9 → (y % 1000) ∈ [(d-1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10, (d+1)*1000 + x / 100 % 10*100 + x / 10 % 10*10 + x % 10]))) ∧
  (y % x = 0)

/-- Prove the existence of four amusing four-digit numbers -/
theorem exist_amusing_numbers :
  ∃ x1 x2 x3 x4 : ℕ, is_amusing x1 ∧ is_amusing x2 ∧ is_amusing x3 ∧ is_amusing x4 ∧ 
                   x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 :=
by sorry

end exist_amusing_numbers_l1396_139658


namespace driving_speed_ratio_l1396_139690

theorem driving_speed_ratio
  (x : ℝ) (y : ℝ)
  (h1 : y = 2 * x) :
  y / x = 2 := by
  sorry

end driving_speed_ratio_l1396_139690


namespace proposition_false_l1396_139601

theorem proposition_false (a b : ℝ) (h : a + b > 0) : ¬ (a > 0 ∧ b > 0) := 
by {
  sorry -- this is a placeholder for the proof
}

end proposition_false_l1396_139601


namespace circle_area_l1396_139676

open Real

theorem circle_area (x y : ℝ) :
  (∃ r, (x + 2)^2 + (y - 3 / 2)^2 = r^2) →
  r = 7 / 2 →
  ∃ A, A = (π * (r)^2) ∧ A = (49/4) * π :=
by
  sorry

end circle_area_l1396_139676


namespace polynomial_has_real_root_l1396_139696

open Real

theorem polynomial_has_real_root (a : ℝ) : 
  ∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 - x + a = 0 :=
sorry

end polynomial_has_real_root_l1396_139696


namespace parallel_line_equation_perpendicular_line_equation_l1396_139697

theorem parallel_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, 4 * x - y - 7 = 0) :=
sorry

theorem perpendicular_line_equation {x y : ℝ} (P : ∃ x y, 2 * x + y - 5 = 0 ∧ x - 2 * y = 0) :
  (∃ (l : ℝ), ∀ x y, x + 4 * y - 6 = 0) :=
sorry

end parallel_line_equation_perpendicular_line_equation_l1396_139697


namespace negation_of_exists_l1396_139626

theorem negation_of_exists {x : ℝ} :
  (¬ ∃ x : ℝ, x^2 - 2 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2 > 0 :=
by
  sorry

end negation_of_exists_l1396_139626


namespace num_arrangements_l1396_139649

-- Define the problem conditions
def athletes : Finset ℕ := {0, 1, 2, 3, 4, 5}
def A : ℕ := 0
def B : ℕ := 1

-- Define the constraint that athlete A cannot run the first leg and athlete B cannot run the fourth leg
def valid_arrangements (sequence : Fin 4 → ℕ) : Prop :=
  sequence 0 ≠ A ∧ sequence 3 ≠ B

-- Main theorem statement: There are 252 valid arrangements
theorem num_arrangements : (Fin 4 → ℕ) → ℕ :=
  sorry

end num_arrangements_l1396_139649


namespace longest_boat_length_l1396_139672

-- Definitions of the conditions
def total_savings : ℤ := 20000
def cost_per_foot : ℤ := 1500
def license_registration : ℤ := 500
def docking_fees := 3 * license_registration

-- Calculate the reserved amount for license, registration, and docking fees
def reserved_amount := license_registration + docking_fees

-- Calculate the amount left for the boat
def amount_left := total_savings - reserved_amount

-- Calculate the maximum length of the boat Mitch can afford
def max_boat_length := amount_left / cost_per_foot

-- Theorem to prove the longest boat Mitch can buy
theorem longest_boat_length : max_boat_length = 12 :=
by
  sorry

end longest_boat_length_l1396_139672


namespace range_of_a_l1396_139688

theorem range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 + 2*x + a > 0) ↔ a ≤ 1 :=
sorry

end range_of_a_l1396_139688


namespace quintic_polynomial_p_l1396_139646

theorem quintic_polynomial_p (p q : ℝ) (h : (∀ x : ℝ, x^p + 4*x^3 - q*x^2 - 2*x + 5 = (x^5 + 4*x^3 - q*x^2 - 2*x + 5))) : -p = -5 :=
by {
  sorry
}

end quintic_polynomial_p_l1396_139646


namespace janet_has_five_dimes_l1396_139670

theorem janet_has_five_dimes (n d q : ℕ) 
    (h1 : n + d + q = 10) 
    (h2 : d + q = 7) 
    (h3 : n + d = 8) : 
    d = 5 :=
by
  -- Proof omitted
  sorry

end janet_has_five_dimes_l1396_139670


namespace num_ways_to_select_3_colors_from_9_l1396_139628

def num_ways_select_colors (n k : ℕ) : ℕ := Nat.choose n k

theorem num_ways_to_select_3_colors_from_9 : num_ways_select_colors 9 3 = 84 := by
  sorry

end num_ways_to_select_3_colors_from_9_l1396_139628


namespace equation_linear_implies_k_equals_neg2_l1396_139678

theorem equation_linear_implies_k_equals_neg2 (k : ℤ) (x : ℝ) :
  (k - 2) * x^(abs k - 1) = k + 1 → abs k - 1 = 1 ∧ k - 2 ≠ 0 → k = -2 :=
by
  sorry

end equation_linear_implies_k_equals_neg2_l1396_139678


namespace polynomial_expression_l1396_139608

theorem polynomial_expression :
  (2 * x^2 + 3 * x + 7) * (x + 1) - (x + 1) * (x^2 + 4 * x - 63) + (3 * x - 14) * (x + 1) * (x + 5) = 4 * x^3 + 4 * x^2 :=
by
  sorry

end polynomial_expression_l1396_139608


namespace joanna_estimate_is_larger_l1396_139643

theorem joanna_estimate_is_larger 
  (u v ε₁ ε₂ : ℝ) 
  (huv : u > v) 
  (hv0 : v > 0) 
  (hε₁ : ε₁ > 0) 
  (hε₂ : ε₂ > 0) : 
  (u + ε₁) - (v - ε₂) > u - v := 
sorry

end joanna_estimate_is_larger_l1396_139643


namespace salary_C_more_than_A_ratio_salary_E_to_A_and_B_l1396_139657

variable (x : ℝ)
variables (salary_A salary_B salary_C salary_D salary_E combined_salary_BCD : ℝ)

-- Conditions
def conditions : Prop :=
  salary_B = 2 * salary_A ∧
  salary_C = 3 * salary_A ∧
  salary_D = 4 * salary_A ∧
  salary_E = 5 * salary_A ∧
  combined_salary_BCD = 15000 ∧
  combined_salary_BCD = salary_B + salary_C + salary_D

-- Statements to prove
theorem salary_C_more_than_A
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  (salary_C - salary_A) / salary_A * 100 = 200 := by
  sorry

theorem ratio_salary_E_to_A_and_B
  (cond : conditions salary_A salary_B salary_C salary_D salary_E combined_salary_BCD) :
  salary_E / (salary_A + salary_B) = 5 / 3 := by
  sorry

end salary_C_more_than_A_ratio_salary_E_to_A_and_B_l1396_139657


namespace points_on_line_l1396_139644

theorem points_on_line (x : ℕ) (h₁ : 9 * x - 8 = 82) : x = 10 :=
by
  sorry

end points_on_line_l1396_139644


namespace find_amount_with_r_l1396_139618

variable (p q r s : ℝ) (total : ℝ := 9000)

-- Condition 1: Total amount is 9000 Rs
def total_amount_condition := p + q + r + s = total

-- Condition 2: r has three-quarters of the combined amount of p, q, and s
def r_amount_condition := r = (3/4) * (p + q + s)

-- The goal is to prove that r = 10800
theorem find_amount_with_r (h1 : total_amount_condition p q r s) (h2 : r_amount_condition p q r s) :
  r = 10800 :=
sorry

end find_amount_with_r_l1396_139618


namespace find_gain_percent_l1396_139651

theorem find_gain_percent (CP SP : ℝ) (h1 : CP = 20) (h2 : SP = 25) : 100 * ((SP - CP) / CP) = 25 := by
  sorry

end find_gain_percent_l1396_139651


namespace find_a_l1396_139669

-- Definitions of universal set U, set P, and complement of P in U
def U (a : ℤ) : Set ℤ := {2, 4, 3 - a^2}
def P (a : ℤ) : Set ℤ := {2, a^2 - a + 2}
def complement_U_P (a : ℤ) : Set ℤ := {-1}

-- The Lean statement asserting the conditions and the proof goal
theorem find_a (a : ℤ) (h_union : U a = P a ∪ complement_U_P a) : a = -1 :=
sorry

end find_a_l1396_139669


namespace david_twice_as_old_in_Y_years_l1396_139604

variable (R D Y : ℕ)

-- Conditions
def rosy_current_age := R = 8
def david_is_older := D = R + 12
def twice_as_old_in_Y_years := D + Y = 2 * (R + Y)

-- Proof statement
theorem david_twice_as_old_in_Y_years
  (h1 : rosy_current_age R)
  (h2 : david_is_older R D)
  (h3 : twice_as_old_in_Y_years R D Y) :
  Y = 4 := sorry

end david_twice_as_old_in_Y_years_l1396_139604


namespace angle_C_measure_l1396_139632

-- We define angles and the specific conditions given in the problem.
def measure_angle_A : ℝ := 80
def external_angle_C : ℝ := 100

theorem angle_C_measure :
  ∃ (C : ℝ) (A B : ℝ), (A + B = measure_angle_A) ∧
                       (C + external_angle_C = 180) ∧
                       (external_angle_C = measure_angle_A) →
                       C = 100 :=
by {
  -- skipping proof
  sorry
}

end angle_C_measure_l1396_139632


namespace coachClass_seats_count_l1396_139685

-- Defining the conditions as given in a)
variables (F : ℕ) -- Number of first-class seats
variables (totalSeats : ℕ := 567) -- Total number of seats is given as 567
variables (businessClassSeats : ℕ := 3 * F) -- Business class seats defined in terms of F
variables (coachClassSeats : ℕ := 7 * F + 5) -- Coach class seats defined in terms of F
variables (firstClassSeats : ℕ := F) -- The variable itself

-- The statement to prove
theorem coachClass_seats_count : 
  F + businessClassSeats + coachClassSeats = totalSeats →
  coachClassSeats = 362 :=
by
  sorry -- The proof would go here

end coachClass_seats_count_l1396_139685


namespace carrie_harvests_9000_l1396_139675

noncomputable def garden_area (length width : ℕ) := length * width
noncomputable def total_plants (plants_per_sqft sqft : ℕ) := plants_per_sqft * sqft
noncomputable def total_cucumbers (yield_plants plants : ℕ) := yield_plants * plants

theorem carrie_harvests_9000 :
  garden_area 10 12 = 120 →
  total_plants 5 120 = 600 →
  total_cucumbers 15 600 = 9000 :=
by sorry

end carrie_harvests_9000_l1396_139675


namespace ninth_graders_only_science_not_history_l1396_139623

-- Conditions
def total_students : ℕ := 120
def students_science : ℕ := 85
def students_history : ℕ := 75

-- Statement: Determine the number of students enrolled only in the science class
theorem ninth_graders_only_science_not_history : 
  (students_science - (students_science + students_history - total_students)) = 45 := by
  sorry

end ninth_graders_only_science_not_history_l1396_139623


namespace proof_A_intersection_C_U_B_l1396_139624

open Set

-- Given sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Prove that the intersection of A and C_U_B is {2, 3}
theorem proof_A_intersection_C_U_B :
  A ∩ C_U_B = {2, 3} := by
  sorry

end proof_A_intersection_C_U_B_l1396_139624


namespace trapezium_other_side_length_l1396_139661

theorem trapezium_other_side_length 
  (side1 : ℝ) (distance : ℝ) (area : ℝ) (side2 : ℝ)
  (h_side1 : side1 = 18)
  (h_distance : distance = 13)
  (h_area : area = 247)
  (h_area_formula : area = 0.5 * (side1 + side2) * distance) :
  side2 = 20 :=
by
  rw [h_side1, h_distance, h_area] at h_area_formula
  sorry

end trapezium_other_side_length_l1396_139661


namespace pipeline_equation_correct_l1396_139605

variables (m x n : ℝ) -- Length of the pipeline, kilometers per day, efficiency increase percentage
variable (h : 0 < n) -- Efficiency increase percentage is positive

theorem pipeline_equation_correct :
  (m / x) - (m / ((1 + (n / 100)) * x)) = 8 :=
sorry -- Proof omitted

end pipeline_equation_correct_l1396_139605


namespace quadratic_root_value_l1396_139635

theorem quadratic_root_value (a : ℝ) (h : a^2 + 2 * a - 3 = 0) : 2 * a^2 + 4 * a = 6 :=
by
  sorry

end quadratic_root_value_l1396_139635


namespace rowing_time_to_and_fro_l1396_139640

noncomputable def rowing_time (distance rowing_speed current_speed : ℤ) : ℤ :=
  let speed_to_place := rowing_speed - current_speed
  let speed_back_place := rowing_speed + current_speed
  let time_to_place := distance / speed_to_place
  let time_back_place := distance / speed_back_place
  time_to_place + time_back_place

theorem rowing_time_to_and_fro (distance rowing_speed current_speed : ℤ) :
  distance = 72 → rowing_speed = 10 → current_speed = 2 → rowing_time distance rowing_speed current_speed = 15 := by
  intros h_dist h_row_speed h_curr_speed
  rw [h_dist, h_row_speed, h_curr_speed]
  sorry

end rowing_time_to_and_fro_l1396_139640


namespace solve_linear_combination_l1396_139682

theorem solve_linear_combination (x y z : ℤ) 
    (h1 : x + 2 * y - z = 8) 
    (h2 : 2 * x - y + z = 18) : 
    8 * x + y + z = 70 := 
by 
    sorry

end solve_linear_combination_l1396_139682


namespace no_positive_integer_solutions_l1396_139603

def f (x : ℤ) : ℤ := x^2 + x

theorem no_positive_integer_solutions 
    (a b : ℤ) (ha : 0 < a) (hb : 0 < b) : 4 * f a ≠ f b := by
  sorry

end no_positive_integer_solutions_l1396_139603


namespace circumcircle_radius_l1396_139627

theorem circumcircle_radius (b A S : ℝ) (h_b : b = 2) 
  (h_A : A = 120 * Real.pi / 180) (h_S : S = Real.sqrt 3) : 
  ∃ R, R = 2 := 
by
  sorry

end circumcircle_radius_l1396_139627


namespace range_of_a_l1396_139673

theorem range_of_a {a : ℝ} :
  (∀ x : ℝ, (a-2)*x^2 + 2*(a-2)*x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l1396_139673


namespace probability_of_adjacent_rs_is_two_fifth_l1396_139616

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def countArrangementsWithAdjacentRs : ℕ :=
factorial 4

noncomputable def countTotalArrangements : ℕ :=
factorial 5 / factorial 2

noncomputable def probabilityOfAdjacentRs : ℚ :=
(countArrangementsWithAdjacentRs : ℚ) / (countTotalArrangements : ℚ)

theorem probability_of_adjacent_rs_is_two_fifth :
  probabilityOfAdjacentRs = 2 / 5 := by
  sorry

end probability_of_adjacent_rs_is_two_fifth_l1396_139616


namespace total_earnings_l1396_139634

theorem total_earnings (x y : ℝ) (h : 20 * x * y = 18 * x * y + 150) : 
  18 * x * y + 20 * x * y + 20 * x * y = 4350 :=
by sorry

end total_earnings_l1396_139634


namespace find_common_ratio_l1396_139648

-- Define the geometric sequence with the given conditions
variable (a_n : ℕ → ℝ)
variable (q : ℝ)

axiom a2_eq : a_n 2 = 1
axiom a4_eq : a_n 4 = 4
axiom q_pos : q > 0

-- Define the nature of the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The specific problem statement to prove
theorem find_common_ratio (h: is_geometric_sequence a_n q) : q = 2 :=
by
  sorry

end find_common_ratio_l1396_139648


namespace range_of_a_l1396_139674

theorem range_of_a (a : ℝ) : (3 + 5 > 1 - 2 * a) ∧ (3 + (1 - 2 * a) > 5) ∧ (5 + (1 - 2 * a) > 3) → -7 / 2 < a ∧ a < -1 / 2 :=
by
  sorry

end range_of_a_l1396_139674


namespace probability_of_joining_between_1890_and_1969_l1396_139613

theorem probability_of_joining_between_1890_and_1969 :
  let total_provinces_and_territories := 13
  let joined_1890_to_1929 := 3
  let joined_1930_to_1969 := 1
  let total_joined_between_1890_and_1969 := joined_1890_to_1929 + joined_1930_to_1969
  total_joined_between_1890_and_1969 / total_provinces_and_territories = 4 / 13 :=
by
  sorry

end probability_of_joining_between_1890_and_1969_l1396_139613


namespace minimum_apples_l1396_139625

theorem minimum_apples (x : ℕ) : 
  (x ≡ 10 [MOD 3]) ∧ (x ≡ 11 [MOD 4]) ∧ (x ≡ 12 [MOD 5]) → x = 67 :=
sorry

end minimum_apples_l1396_139625


namespace sum_of_coefficients_l1396_139617

def f (x : ℝ) : ℝ := (1 + 2 * x)^4

theorem sum_of_coefficients : f 1 = 81 :=
by
  -- New goal is immediately achieved since the given is precisely ensured.
  sorry

end sum_of_coefficients_l1396_139617


namespace decreasing_function_in_interval_l1396_139667

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x + Real.pi / 4)

theorem decreasing_function_in_interval (ω : ℝ) (h_omega_pos : ω > 0) (h_period : Real.pi / 3 < 2 * Real.pi / (2 * ω) ∧ 2 * Real.pi / (2 * ω) < Real.pi / 2)
    (h_symmetry : 2 * ω * 3 * Real.pi / 4 + Real.pi / 4 = (4:ℤ) * Real.pi) :
    ∀ x : ℝ, Real.pi / 6 < x ∧ x < Real.pi / 4 → f ω x < f ω (x + Real.pi / 100) :=
by
    intro x h_interval
    have ω_value : ω = 5 / 2 := sorry
    exact sorry

end decreasing_function_in_interval_l1396_139667


namespace spider_total_distance_l1396_139691

-- Define points where spider starts and moves
def start_position : ℤ := 3
def first_move : ℤ := -4
def second_move : ℤ := 8
def final_move : ℤ := 2

-- Define the total distance the spider crawls
def total_distance : ℤ :=
  |first_move - start_position| +
  |second_move - first_move| +
  |final_move - second_move|

-- Theorem statement
theorem spider_total_distance : total_distance = 25 :=
sorry

end spider_total_distance_l1396_139691


namespace fraction_sum_ratio_l1396_139664

theorem fraction_sum_ratio
    (a b c : ℝ) (m n : ℝ)
    (h1 : a = (b + c) / m)
    (h2 : b = (c + a) / n) :
    (m * n ≠ 1 → (a + b) / c = (m + n + 2) / (m * n - 1)) ∧ 
    (m = -1 ∧ n = -1 → (a + b) / c = -1) :=
by
    sorry

end fraction_sum_ratio_l1396_139664


namespace smallest_n_divisor_lcm_gcd_l1396_139641

theorem smallest_n_divisor_lcm_gcd :
  ∀ n : ℕ, n > 0 ∧ (∀ a b : ℕ, 60 = a ∧ n = b → (Nat.lcm a b / Nat.gcd a b = 50)) → n = 750 :=
by
  sorry

end smallest_n_divisor_lcm_gcd_l1396_139641


namespace equidistant_trajectory_l1396_139647

theorem equidistant_trajectory (x y : ℝ) (h : abs x = abs y) : y^2 = x^2 :=
by
  sorry

end equidistant_trajectory_l1396_139647


namespace max_area_triangle_l1396_139656

open Real

theorem max_area_triangle (a b : ℝ) (C : ℝ) (h₁ : a + b = 4) (h₂ : C = π / 6) : 
  (1 : ℝ) ≥ (1 / 2 * a * b * sin (π / 6)) := 
by 
  sorry

end max_area_triangle_l1396_139656


namespace icing_cubes_count_l1396_139689

theorem icing_cubes_count :
  let n := 5
  let total_cubes := n * n * n
  let side_faces := 4
  let cubes_per_edge_per_face := (n - 2) * (n - 1)
  let shared_edges := 4
  let icing_cubes := (side_faces * cubes_per_edge_per_face) / 2
  icing_cubes = 32 := sorry

end icing_cubes_count_l1396_139689


namespace budget_allocation_l1396_139607

theorem budget_allocation 
  (total_degrees : ℝ := 360)
  (total_budget : ℝ := 100)
  (degrees_basic_astrophysics : ℝ := 43.2)
  (percent_microphotonics : ℝ := 12)
  (percent_home_electronics : ℝ := 24)
  (percent_food_additives : ℝ := 15)
  (percent_industrial_lubricants : ℝ := 8) :
  ∃ percent_genetically_modified_microorganisms : ℝ,
  percent_genetically_modified_microorganisms = 29 :=
sorry

end budget_allocation_l1396_139607


namespace constant_term_is_24_l1396_139665

noncomputable def constant_term_of_binomial_expansion 
  (a : ℝ) (hx : π * a^2 = 4 * π) : ℝ :=
  if ha : a = 2 then 24 else 0

theorem constant_term_is_24
  (a : ℝ) (hx : π * a^2 = 4 * π) :
  constant_term_of_binomial_expansion a hx = 24 :=
by
  sorry

end constant_term_is_24_l1396_139665


namespace initial_puppies_count_l1396_139642

-- Define the initial conditions
def initial_birds : Nat := 12
def initial_cats : Nat := 5
def initial_spiders : Nat := 15
def initial_total_animals : Nat := 25
def half_birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_lost : Nat := 7

-- Define the remaining animals
def remaining_birds : Nat := initial_birds - half_birds_sold
def remaining_cats : Nat := initial_cats
def remaining_spiders : Nat := initial_spiders - spiders_lost

-- Define the total number of remaining animals excluding puppies
def remaining_non_puppy_animals : Nat := remaining_birds + remaining_cats + remaining_spiders

-- Define the remaining puppies
def remaining_puppies : Nat := initial_total_animals - remaining_non_puppy_animals
def initial_puppies : Nat := remaining_puppies + puppies_adopted

-- State the theorem
theorem initial_puppies_count :
  ∀ puppies : Nat, initial_puppies = 9 :=
by
  sorry

end initial_puppies_count_l1396_139642


namespace linear_function_unique_l1396_139636

noncomputable def f (x : ℝ) : ℝ := sorry

theorem linear_function_unique
  (h1 : ∀ x : ℝ, f (f x) = 4 * x + 6)
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :
  ∀ x : ℝ, f x = 2 * x + 2 :=
sorry

end linear_function_unique_l1396_139636


namespace ball_center_distance_traveled_l1396_139621

theorem ball_center_distance_traveled (d : ℝ) (r1 r2 r3 r4 : ℝ) (R1 R2 R3 R4 : ℝ) :
  d = 6 → 
  R1 = 120 → 
  R2 = 50 → 
  R3 = 90 → 
  R4 = 70 → 
  r1 = R1 - 3 → 
  r2 = R2 + 3 → 
  r3 = R3 - 3 → 
  r4 = R4 + 3 → 
  (1/2) * 2 * π * r1 + (1/2) * 2 * π * r2 + (1/2) * 2 * π * r3 + (1/2) * 2 * π * r4 = 330 * π :=
by
  sorry

end ball_center_distance_traveled_l1396_139621


namespace john_drinks_2_cups_per_day_l1396_139652

noncomputable def fluid_ounces_in_gallon : ℕ := 128

noncomputable def half_gallon_in_fluid_ounces : ℕ := 64

noncomputable def standard_cup_size : ℕ := 8

noncomputable def cups_in_half_gallon : ℕ :=
  half_gallon_in_fluid_ounces / standard_cup_size

noncomputable def days_to_consume_half_gallon : ℕ := 4

noncomputable def cups_per_day : ℕ :=
  cups_in_half_gallon / days_to_consume_half_gallon

theorem john_drinks_2_cups_per_day :
  cups_per_day = 2 :=
by
  -- The proof is left as an exercise, but the statement should be correct.
  sorry

end john_drinks_2_cups_per_day_l1396_139652


namespace clock_rings_in_january_l1396_139631

theorem clock_rings_in_january :
  ∀ (days_in_january hours_per_day ring_interval : ℕ)
  (first_ring_time : ℕ) (january_first_hour : ℕ), 
  days_in_january = 31 →
  hours_per_day = 24 →
  ring_interval = 7 →
  january_first_hour = 2 →
  first_ring_time = 30 →
  (days_in_january * hours_per_day) / ring_interval + 1 = 107 := by
  intros days_in_january hours_per_day ring_interval first_ring_time january_first_hour
  sorry

end clock_rings_in_january_l1396_139631


namespace white_tiles_in_square_l1396_139620

theorem white_tiles_in_square (n S : ℕ) (hn : n * n = S) (black_tiles : ℕ) (hblack_tiles : black_tiles = 81) (diagonal_black_tiles : n = 9) :
  S - black_tiles = 72 :=
by
  sorry

end white_tiles_in_square_l1396_139620


namespace base12_remainder_div_7_l1396_139629

-- Define the base-12 number 2543 in decimal form
def n : ℕ := 2 * 12^3 + 5 * 12^2 + 4 * 12^1 + 3 * 12^0

-- Theorem statement: the remainder when n is divided by 7 is 6
theorem base12_remainder_div_7 : n % 7 = 6 := by
  sorry

end base12_remainder_div_7_l1396_139629


namespace sqrt_inequality_l1396_139610

theorem sqrt_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt (a ^ 2 / b) + Real.sqrt (b ^ 2 / a) ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end sqrt_inequality_l1396_139610


namespace proposition_contradiction_l1396_139659

-- Define the proposition P for natural numbers.
def P (n : ℕ+) : Prop := sorry

theorem proposition_contradiction (h1 : ∀ k : ℕ+, P k → P (k + 1)) (h2 : ¬ P 5) : ¬ P 4 :=
by
  sorry

end proposition_contradiction_l1396_139659


namespace toy_problem_l1396_139614

theorem toy_problem :
  ∃ (n m : ℕ), 
    1500 ≤ n ∧ n ≤ 2000 ∧ 
    n % 15 = 5 ∧ n % 20 = 5 ∧ n % 30 = 5 ∧ 
    (n + m) % 12 = 0 ∧ (n + m) % 18 = 0 ∧ 
    n + m ≤ 2100 ∧ m = 31 := 
sorry

end toy_problem_l1396_139614


namespace length_of_PQ_l1396_139666

theorem length_of_PQ
  (k : ℝ) -- height of the trapezoid
  (PQ RU : ℝ) -- sides of trapezoid PQRU
  (A1 : ℝ := (PQ * k) / 2) -- area of triangle PQR
  (A2 : ℝ := (RU * k) / 2) -- area of triangle PUR
  (ratio_A1_A2 : A1 / A2 = 5 / 2) -- given ratio of areas
  (sum_PQ_RU : PQ + RU = 180) -- given sum of PQ and RU
  : PQ = 900 / 7 :=
by
  sorry

end length_of_PQ_l1396_139666


namespace neg_p_necessary_not_sufficient_for_neg_p_or_q_l1396_139655

variables (p q : Prop)

theorem neg_p_necessary_not_sufficient_for_neg_p_or_q :
  (¬ p → ¬ (p ∨ q)) ∧ (¬ (p ∨ q) → ¬ p) :=
by {
  sorry
}

end neg_p_necessary_not_sufficient_for_neg_p_or_q_l1396_139655


namespace possible_values_of_k_l1396_139650

noncomputable def has_roots (p q r s t k : ℂ) : Prop :=
  (p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧ t ≠ 0) ∧ 
  (p * k^4 + q * k^3 + r * k^2 + s * k + t = 0) ∧
  (q * k^4 + r * k^3 + s * k^2 + t * k + p = 0)

theorem possible_values_of_k (p q r s t k : ℂ) (hk : has_roots p q r s t k) : 
  k^5 = 1 :=
  sorry

end possible_values_of_k_l1396_139650


namespace range_of_a_for_quadratic_inequality_l1396_139683

theorem range_of_a_for_quadratic_inequality (a : ℝ) :
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1 > 0) →
  (a ≤ -2 ∨ a ≥ 6) :=
by
  sorry

end range_of_a_for_quadratic_inequality_l1396_139683


namespace sum_mod_17_eq_0_l1396_139684

theorem sum_mod_17_eq_0 :
  (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 0 :=
by
  sorry

end sum_mod_17_eq_0_l1396_139684


namespace tank_capacity_l1396_139686

theorem tank_capacity (C : ℕ) 
  (h : 0.9 * (C : ℝ) - 0.4 * (C : ℝ) = 63) : C = 126 := 
by
  sorry

end tank_capacity_l1396_139686


namespace probability_red_on_other_side_l1396_139633

def num_black_black_cards := 4
def num_black_red_cards := 2
def num_red_red_cards := 2

def num_red_sides_total := 
  num_black_black_cards * 0 +
  num_black_red_cards * 1 +
  num_red_red_cards * 2

def num_red_sides_with_red_on_other_side := 
  num_red_red_cards * 2

theorem probability_red_on_other_side :
  (num_red_sides_with_red_on_other_side : ℚ) / num_red_sides_total = 2 / 3 := by
  sorry

end probability_red_on_other_side_l1396_139633


namespace range_x1_x2_l1396_139693

theorem range_x1_x2
  (x1 x2 x3 : ℝ)
  (hx3_le_x2 : x3 ≤ x2)
  (hx2_le_x1 : x2 ≤ x1)
  (hx_sum : x1 + x2 + x3 = 1)
  (hfx_sum : (x1^2) + (x2^2) + (x3^2) = 1) :
  (2 / 3 : ℝ) ≤ x1 + x2 ∧ x1 + x2 ≤ (4 / 3 : ℝ) :=
sorry

end range_x1_x2_l1396_139693


namespace min_scalar_product_l1396_139611

open Real

variable {a b : ℝ → ℝ}

-- Definitions used as conditions in the problem
def condition (a b : ℝ → ℝ) : Prop :=
  |2 * a - b| ≤ 3

-- The goal to prove based on the conditions and the correct answer
theorem min_scalar_product (h : condition a b) : 
  (a x) * (b x) ≥ -9 / 8 :=
sorry

end min_scalar_product_l1396_139611


namespace tank_empty_time_correct_l1396_139654

noncomputable def tank_time_to_empty (leak_empty_time : ℕ) (inlet_rate : ℕ) (tank_capacity : ℕ) : ℕ :=
(tank_capacity / (tank_capacity / leak_empty_time - inlet_rate * 60))

theorem tank_empty_time_correct :
  tank_time_to_empty 6 3 4320 = 8 := by
  sorry

end tank_empty_time_correct_l1396_139654


namespace pentagon_side_length_l1396_139662

-- Define the side length of the equilateral triangle
def side_length_triangle : ℚ := 20 / 9

-- Define the perimeter of the equilateral triangle
def perimeter_triangle : ℚ := 3 * side_length_triangle

-- Define the side length of the regular pentagon
def side_length_pentagon : ℚ := 4 / 3

-- Prove that the side length of the regular pentagon has the same perimeter as the equilateral triangle
theorem pentagon_side_length (s : ℚ) (h1 : s = side_length_pentagon) :
  5 * s = perimeter_triangle :=
by
  -- Provide the solution
  sorry

end pentagon_side_length_l1396_139662


namespace find_usual_time_l1396_139663

noncomputable def journey_time (S T : ℝ) : Prop :=
  (6 / 5) = (T + (1 / 5)) / T

theorem find_usual_time (S T : ℝ) (h1 : ∀ S T, S / (5 / 6 * S) = (T + (12 / 60)) / T) : T = 1 :=
by
  -- Let the conditions defined by the user be:
  -- h1 : condition (e.g., the cab speed and time relationship)
  -- Given that the cab is \(\frac{5}{6}\) times its speed and is late by 12 minutes
  let h1 := journey_time S T
  sorry

end find_usual_time_l1396_139663


namespace domain_of_f_l1396_139671

noncomputable def f (x : ℝ) := Real.log (1 - x)

theorem domain_of_f : ∀ x, f x = Real.log (1 - x) → (1 - x > 0) →  x < 1 :=
by
  intro x h₁ h₂
  exact lt_of_sub_pos h₂

end domain_of_f_l1396_139671


namespace solve_system_of_equations_l1396_139639

theorem solve_system_of_equations (a b c x y z : ℝ):
  (x - a * y + a^2 * z = a^3) →
  (x - b * y + b^2 * z = b^3) →
  (x - c * y + c^2 * z = c^3) →
  x = a * b * c ∧ y = a * b + a * c + b * c ∧ z = a + b + c :=
by
  intros h1 h2 h3
  -- proof goes here
  sorry

end solve_system_of_equations_l1396_139639


namespace work_problem_l1396_139698

theorem work_problem (x : ℕ) (b_work : ℕ) (a_b_together_work : ℕ) (h1: b_work = 24) (h2: a_b_together_work = 8) :
  (1 / x) + (1 / b_work) = (1 / a_b_together_work) → x = 12 :=
by 
  intros h_eq
  have h_b : b_work = 24 := h1
  have h_ab : a_b_together_work = 8 := h2
  -- Full proof is omitted
  sorry

end work_problem_l1396_139698


namespace mary_initial_stickers_l1396_139645

theorem mary_initial_stickers (stickers_remaining : ℕ) 
  (front_page_stickers : ℕ) (other_page_stickers : ℕ) 
  (num_other_pages : ℕ) 
  (h1 : front_page_stickers = 3)
  (h2 : other_page_stickers = 7 * num_other_pages)
  (h3 : num_other_pages = 6)
  (h4 : stickers_remaining = 44) :
  ∃ initial_stickers : ℕ, initial_stickers = front_page_stickers + other_page_stickers + stickers_remaining ∧ initial_stickers = 89 :=
by
  sorry

end mary_initial_stickers_l1396_139645


namespace numbers_are_odd_l1396_139606

theorem numbers_are_odd (n : ℕ) (sum : ℕ) (h1 : n = 49) (h2 : sum = 2401) : 
      (∀ i < n, ∃ j, sum = j * 2 * i + 1) :=
by
  sorry

end numbers_are_odd_l1396_139606


namespace find_k_l1396_139681

-- Definitions of conditions
variables (x y k : ℤ)

-- System of equations as given in the problem
def system_eq1 := x + 2 * y = 7 + k
def system_eq2 := 5 * x - y = k

-- Condition that solutions x and y are additive inverses
def y_is_add_inv := y = -x

-- The statement we need to prove
theorem find_k (hx : system_eq1 x y k) (hy : system_eq2 x y k) (hz : y_is_add_inv x y) : k = -6 :=
by
  sorry -- proof will go here

end find_k_l1396_139681


namespace problem1_solution_l1396_139612

theorem problem1_solution (x y : ℝ) 
  (h1 : 3 * x + 4 * y = 16)
  (h2 : 5 * x - 6 * y = 33) : 
  x = 6 ∧ y = -1 / 2 := 
  by
  sorry

end problem1_solution_l1396_139612


namespace sum_and_product_formulas_l1396_139602

/-- 
Given an arithmetic sequence {a_n} with the sum of the first n terms S_n = 2n^2, 
and in the sequence {b_n}, b_1 = 1 and b_{n+1} = 3b_n (n ∈ ℕ*),
prove that:
(Ⅰ) The general formula for sequences {a_n} is a_n = 4n - 2,
(Ⅱ) The general formula for sequences {b_n} is b_n = 3^{n-1},
(Ⅲ) Let c_n = a_n * b_n, prove that the sum of the first n terms of the sequence {c_n}, denoted as T_n, is T_n = (2n - 2) * 3^n + 2.
-/
theorem sum_and_product_formulas (S_n : ℕ → ℕ) (b : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T_n : ℕ → ℕ) :
  (∀ n, S_n n = 2 * n^2) →
  (b 1 = 1) →
  (∀ n, b (n + 1) = 3 * (b n)) →
  (∀ n, a n = S_n n - S_n (n - 1)) →
  ∀ n, (T_n n = (2*n - 2) * 3^n + 2) := sorry

end sum_and_product_formulas_l1396_139602


namespace ellipse_solution_l1396_139677

theorem ellipse_solution :
  (∃ (a b : ℝ), a = 4 * Real.sqrt 2 + Real.sqrt 17 ∧ b = Real.sqrt (32 + 16 * Real.sqrt 34) ∧ (∀ (x y : ℝ), (3 * 0 ≤ y ∧ y ≤ 8) → (3 * 0 ≤ x ∧ x ≤ 5) → (Real.sqrt ((x+3)^2 + y^2) + Real.sqrt ((x-3)^2 + y^2) = 2 * a) → 
   (Real.sqrt ((x-0)^2 + (y-8)^2) = b))) :=
sorry

end ellipse_solution_l1396_139677


namespace total_amount_l1396_139680

def g_weight : ℝ := 2.5
def g_price : ℝ := 2.79
def r_weight : ℝ := 1.8
def r_price : ℝ := 3.25
def c_weight : ℝ := 1.2
def c_price : ℝ := 4.90
def o_weight : ℝ := 0.9
def o_price : ℝ := 5.75

theorem total_amount :
  g_weight * g_price + r_weight * r_price + c_weight * c_price + o_weight * o_price = 23.88 := by
  sorry

end total_amount_l1396_139680


namespace carlos_marbles_l1396_139653

theorem carlos_marbles :
  ∃ N : ℕ, 
    (N % 9 = 2) ∧ 
    (N % 10 = 2) ∧ 
    (N % 11 = 2) ∧ 
    (N > 1) ∧ 
    N = 992 :=
by {
  -- We need this for the example; you would remove it in a real proof.
  sorry
}

end carlos_marbles_l1396_139653


namespace angle_of_squares_attached_l1396_139600

-- Definition of the problem scenario:
-- Three squares attached as described, needing to prove x = 39 degrees.

open Real

theorem angle_of_squares_attached (x : ℝ) (h : 
  let angle1 := 30
  let angle2 := 126
  let angle3 := 75
  angle1 + angle2 + angle3 + x = 3 * 90) :
  x = 39 :=
by 
  -- This proof is omitted
  sorry

end angle_of_squares_attached_l1396_139600


namespace new_avg_weight_of_boxes_l1396_139668

theorem new_avg_weight_of_boxes :
  ∀ (x y : ℕ), x + y = 30 → (10 * x + 20 * y) / 30 = 18 → (10 * x + 20 * (y - 18)) / 12 = 15 :=
by
  intro x y h1 h2
  sorry

end new_avg_weight_of_boxes_l1396_139668


namespace initial_water_amount_l1396_139619

theorem initial_water_amount 
  (W : ℝ) 
  (evap_rate : ℝ) 
  (days : ℕ) 
  (percentage_evaporated : ℝ) 
  (evap_rate_eq : evap_rate = 0.012) 
  (days_eq : days = 50) 
  (percentage_evaporated_eq : percentage_evaporated = 0.06) 
  (total_evaporated_eq : evap_rate * days = 0.6) 
  (percentage_condition : percentage_evaporated * W = evap_rate * days) 
  : W = 10 := 
  by sorry

end initial_water_amount_l1396_139619


namespace decomposition_of_cube_l1396_139679

theorem decomposition_of_cube (m : ℕ) (h : m^2 - m + 1 = 73) : m = 9 :=
sorry

end decomposition_of_cube_l1396_139679


namespace sum_of_permutations_is_divisible_by_37_l1396_139687

theorem sum_of_permutations_is_divisible_by_37
  (A B C : ℕ)
  (h : 37 ∣ (100 * A + 10 * B + C)) :
  37 ∣ (100 * B + 10 * C + A + 100 * C + 10 * A + B) :=
by
  sorry

end sum_of_permutations_is_divisible_by_37_l1396_139687


namespace smallest_possible_other_integer_l1396_139638

theorem smallest_possible_other_integer (x m n : ℕ) (h1 : x > 0) (h2 : m = 70) 
  (h3 : gcd m n = x + 7) (h4 : lcm m n = x * (x + 7)) : n = 20 :=
sorry

end smallest_possible_other_integer_l1396_139638


namespace cloud9_total_money_l1396_139609

-- Definitions
def I : ℕ := 12000
def G : ℕ := 16000
def R : ℕ := 1600

-- Total money taken after cancellations
def T : ℕ := I + G - R

-- Theorem stating that T is 26400
theorem cloud9_total_money : T = 26400 := by
  unfold T I G R
  sorry

end cloud9_total_money_l1396_139609


namespace workers_together_complete_work_in_14_days_l1396_139622

noncomputable def efficiency (Wq : ℝ) := 1.4 * Wq

def work_done_in_one_day_p (Wp : ℝ) := Wp = 1 / 24

noncomputable def work_done_in_one_day_q (Wq : ℝ) := Wq = (1 / 24) / 1.4

noncomputable def combined_work_per_day (Wp Wq : ℝ) := Wp + Wq

noncomputable def days_to_complete_work (W : ℝ) := 1 / W

theorem workers_together_complete_work_in_14_days (Wp Wq : ℝ) 
  (h1 : Wp = efficiency Wq)
  (h2 : work_done_in_one_day_p Wp)
  (h3 : work_done_in_one_day_q Wq) :
  days_to_complete_work (combined_work_per_day Wp Wq) = 14 := 
sorry

end workers_together_complete_work_in_14_days_l1396_139622


namespace system_inconsistent_l1396_139699

theorem system_inconsistent :
  ¬(∃ (x1 x2 x3 x4 : ℝ), 
    (5 * x1 + 12 * x2 + 19 * x3 + 25 * x4 = 25) ∧
    (10 * x1 + 22 * x2 + 16 * x3 + 39 * x4 = 25) ∧
    (5 * x1 + 12 * x2 + 9 * x3 + 25 * x4 = 30) ∧
    (20 * x1 + 46 * x2 + 34 * x3 + 89 * x4 = 70)) := 
by
  sorry

end system_inconsistent_l1396_139699


namespace P_Q_sum_equals_44_l1396_139660

theorem P_Q_sum_equals_44 (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3))) :
  P + Q = 44 :=
sorry

end P_Q_sum_equals_44_l1396_139660


namespace number_of_players_sold_eq_2_l1396_139615

def initial_balance : ℕ := 100
def selling_price_per_player : ℕ := 10
def buying_cost_per_player : ℕ := 15
def number_of_players_bought : ℕ := 4
def final_balance : ℕ := 60

theorem number_of_players_sold_eq_2 :
  ∃ x : ℕ, (initial_balance + selling_price_per_player * x - buying_cost_per_player * number_of_players_bought = final_balance) ∧ (x = 2) :=
by
  sorry

end number_of_players_sold_eq_2_l1396_139615


namespace intersection_right_complement_l1396_139692

open Set

def A := {x : ℝ | x - 1 ≥ 0}
def B := {x : ℝ | 3 / x ≤ 1}

theorem intersection_right_complement :
  A ∩ (compl B) = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_right_complement_l1396_139692
