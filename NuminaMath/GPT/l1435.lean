import Mathlib

namespace smallest_positive_period_f_intervals_monotonically_increasing_f_l1435_143509

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) * (Real.sin x + Real.cos x)

-- 1. Proving the smallest positive period is π
theorem smallest_positive_period_f : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi := 
sorry

-- 2. Proving the intervals where the function is monotonically increasing
theorem intervals_monotonically_increasing_f : 
  ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (k * Real.pi - (3 * Real.pi / 8)) (k * Real.pi + (Real.pi / 8)) → 
    0 < deriv f x :=
sorry

end smallest_positive_period_f_intervals_monotonically_increasing_f_l1435_143509


namespace largest_unpayable_soldo_l1435_143505

theorem largest_unpayable_soldo : ∃ N : ℕ, N ≤ 50 ∧ (∀ a b : ℕ, a * 5 + b * 6 ≠ N) ∧ (∀ M : ℕ, (M ≤ 50 ∧ ∀ a b : ℕ, a * 5 + b * 6 ≠ M) → M ≤ 19) :=
by
  sorry

end largest_unpayable_soldo_l1435_143505


namespace least_lcm_possible_l1435_143582

theorem least_lcm_possible (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) : Nat.lcm a c = 12 :=
sorry

end least_lcm_possible_l1435_143582


namespace pieces_given_l1435_143537

def pieces_initially := 38
def pieces_now := 54

theorem pieces_given : pieces_now - pieces_initially = 16 := by
  sorry

end pieces_given_l1435_143537


namespace least_area_of_prime_dim_l1435_143590

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_area_of_prime_dim (l w : ℕ) (h_perimeter : 2 * (l + w) = 120)
    (h_integer_dims : l > 0 ∧ w > 0) (h_prime_dim : is_prime l ∨ is_prime w) :
    l * w = 116 :=
sorry

end least_area_of_prime_dim_l1435_143590


namespace part_a_part_b_part_c_l1435_143586

def Q (x : ℝ) (n : ℕ) : ℝ := (x + 1) ^ n - x ^ n - 1
def P (x : ℝ) : ℝ := x ^ 2 + x + 1

theorem part_a (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) ↔ (∀ x : ℝ, P x ∣ Q x n) := sorry

theorem part_b (n : ℕ) : 
  (∃ k : ℤ, n = 6 * k + 1) ↔ (∀ x : ℝ, (P x)^2 ∣ Q x n) := sorry

theorem part_c (n : ℕ) : 
  n = 1 ↔ (∀ x : ℝ, (P x)^3 ∣ Q x n) := sorry

end part_a_part_b_part_c_l1435_143586


namespace Sunzi_problem_correctness_l1435_143504

theorem Sunzi_problem_correctness (x y : ℕ) :
  3 * (x - 2) = 2 * x + 9 ∧ (y / 3) + 2 = (y - 9) / 2 :=
by
  sorry

end Sunzi_problem_correctness_l1435_143504


namespace part1_part2_l1435_143531

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x ^ 2 + 1

theorem part1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (0 < a ∧ a < 1) :=
sorry

theorem part2 (a : ℝ) :
  (∃ α β m : ℝ, 1 ≤ α ∧ α ≤ 4 ∧ 1 ≤ β ∧ β ≤ 4 ∧ β - α = 1 ∧ f α a = m ∧ f β a = m) ↔ 
  (Real.log 4 / 3 * (2 / 7) ≤ a ∧ a ≤ Real.log 2 * (2 / 3)) :=
sorry

end part1_part2_l1435_143531


namespace theater_ticket_difference_l1435_143503

theorem theater_ticket_difference
  (O B : ℕ)
  (h1 : O + B = 355)
  (h2 : 12 * O + 8 * B = 3320) :
  B - O = 115 :=
sorry

end theater_ticket_difference_l1435_143503


namespace probability_of_smallest_section_l1435_143525

-- Define the probabilities for the largest and next largest sections
def P_largest : ℚ := 1 / 2
def P_next_largest : ℚ := 1 / 3

-- Define the total probability constraint
def total_probability (P_smallest : ℚ) : Prop :=
  P_largest + P_next_largest + P_smallest = 1

-- State the theorem to be proved
theorem probability_of_smallest_section : 
  ∃ P_smallest : ℚ, total_probability P_smallest ∧ P_smallest = 1 / 6 := 
by
  sorry

end probability_of_smallest_section_l1435_143525


namespace sqrt_inequality_sum_of_squares_geq_sum_of_products_l1435_143530

theorem sqrt_inequality : (Real.sqrt 6) + (Real.sqrt 10) > (2 * Real.sqrt 3) + 2 := by
  sorry

theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) : 
    a^2 + b^2 + c^2 ≥ a * b + b * c + a * c := by
  sorry

end sqrt_inequality_sum_of_squares_geq_sum_of_products_l1435_143530


namespace banana_count_l1435_143535

-- Variables representing the number of bananas, oranges, and apples
variables (B O A : ℕ)

-- Conditions translated from the problem statement
def conditions : Prop :=
  (O = 2 * B) ∧
  (A = 2 * O) ∧
  (B + O + A = 35)

-- Theorem to prove the number of bananas is 5 given the conditions
theorem banana_count (B O A : ℕ) (h : conditions B O A) : B = 5 :=
sorry

end banana_count_l1435_143535


namespace dilation_0_minus_2i_to_neg3_minus_14i_l1435_143560

open Complex

def dilation_centered (z_center z zk : ℂ) (factor : ℝ) : ℂ :=
  z_center + factor * (zk - z_center)

theorem dilation_0_minus_2i_to_neg3_minus_14i :
  dilation_centered (1 + 2 * I) (0 - 2 * I) (1 + 2 * I) 4 = -3 - 14 * I :=
by
  sorry

end dilation_0_minus_2i_to_neg3_minus_14i_l1435_143560


namespace distance_from_edge_to_bottom_l1435_143555

theorem distance_from_edge_to_bottom (d x : ℕ) 
  (h1 : 63 + d + 20 = 10 + d + x) : x = 73 := by
  -- This is where the proof would go
  sorry

end distance_from_edge_to_bottom_l1435_143555


namespace three_gt_sqrt_seven_l1435_143513

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := sorry

end three_gt_sqrt_seven_l1435_143513


namespace january_salary_l1435_143546

variable (J F M A My : ℕ)

axiom average_salary_1 : (J + F + M + A) / 4 = 8000
axiom average_salary_2 : (F + M + A + My) / 4 = 8400
axiom may_salary : My = 6500

theorem january_salary : J = 4900 :=
by
  /- To be filled with the proof steps applying the given conditions -/
  sorry

end january_salary_l1435_143546


namespace percentage_increase_l1435_143519

variable (E : ℝ) (P : ℝ)
variable (h1 : 1.36 * E = 495)
variable (h2 : (1 + P) * E = 454.96)

theorem percentage_increase :
  P = 0.25 :=
by
  sorry

end percentage_increase_l1435_143519


namespace pyramid_volume_QEFGH_l1435_143579

noncomputable def volume_of_pyramid (EF FG QE : ℝ) : ℝ :=
  (1 / 3) * EF * FG * QE

theorem pyramid_volume_QEFGH :
  let EF := 10
  let FG := 5
  let QE := 9
  volume_of_pyramid EF FG QE = 150 := by
  sorry

end pyramid_volume_QEFGH_l1435_143579


namespace digits_of_2_120_l1435_143588

theorem digits_of_2_120 (h : ∀ n : ℕ, (10 : ℝ)^(n - 1) ≤ (2 : ℝ)^200 ∧ (2 : ℝ)^200 < (10 : ℝ)^n → n = 61) :
  ∀ m : ℕ, (10 : ℝ)^(m - 1) ≤ (2 : ℝ)^120 ∧ (2 : ℝ)^120 < (10 : ℝ)^m → m = 37 :=
by
  sorry

end digits_of_2_120_l1435_143588


namespace total_cost_of_tickets_l1435_143528

theorem total_cost_of_tickets (num_family_members num_adult_tickets num_children_tickets : ℕ)
    (cost_adult_ticket cost_children_ticket total_cost : ℝ) 
    (h1 : num_family_members = 7) 
    (h2 : cost_adult_ticket = 21) 
    (h3 : cost_children_ticket = 14) 
    (h4 : num_adult_tickets = 4) 
    (h5 : num_children_tickets = num_family_members - num_adult_tickets) 
    (h6 : total_cost = num_adult_tickets * cost_adult_ticket + num_children_tickets * cost_children_ticket) :
    total_cost = 126 :=
by
  sorry

end total_cost_of_tickets_l1435_143528


namespace sum_of_xyz_l1435_143516

theorem sum_of_xyz (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 18) (hxz : x * z = 3) (hyz : y * z = 6) : x + y + z = 10 := 
sorry

end sum_of_xyz_l1435_143516


namespace sin_gt_cos_range_l1435_143578

theorem sin_gt_cos_range (x : ℝ) : 
  0 < x ∧ x < 2 * Real.pi → (Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4)) := by
  sorry

end sin_gt_cos_range_l1435_143578


namespace min_value_of_fraction_l1435_143510

theorem min_value_of_fraction (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (4 / (a + 2) + 1 / (b + 1)) = 9 / 4 :=
sorry

end min_value_of_fraction_l1435_143510


namespace incorrect_scientific_statement_is_D_l1435_143591

-- Define the number of colonies screened by Student A and other students
def studentA_colonies := 150
def other_students_colonies := 50

-- Define the descriptions
def descriptionA := "The reason Student A had such results could be due to different soil samples or problems in the experimental operation."
def descriptionB := "Student A's prepared culture medium could be cultured without adding soil as a blank control, to demonstrate whether the culture medium is contaminated."
def descriptionC := "If other students use the same soil as Student A for the experiment and get consistent results with Student A, it can be proven that Student A's operation was without error."
def descriptionD := "Both experimental approaches described in options B and C follow the principle of control in the experiment."

-- The incorrect scientific statement identified
def incorrect_statement := descriptionD

-- The main theorem statement
theorem incorrect_scientific_statement_is_D : incorrect_statement = descriptionD := by
  sorry

end incorrect_scientific_statement_is_D_l1435_143591


namespace min_value_of_expression_l1435_143587

theorem min_value_of_expression {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1/a + 2/b = 3) :
  (a + 1) * (b + 2) = 50/9 :=
sorry

end min_value_of_expression_l1435_143587


namespace factorize_expression_l1435_143524

theorem factorize_expression (a b : ℝ) :
  4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) :=
by
  sorry

end factorize_expression_l1435_143524


namespace class_grades_l1435_143581

theorem class_grades (boys girls n : ℕ) (h1 : girls = boys + 3) (h2 : ∀ (fours fives : ℕ), fours = fives + 6) (h3 : ∀ (threes : ℕ), threes = 2 * (fives + 6)) : ∃ k, k = 2 ∨ k = 1 :=
by
  sorry

end class_grades_l1435_143581


namespace no_integer_x_square_l1435_143518

theorem no_integer_x_square (x : ℤ) : 
  ∀ n : ℤ, x^5 + 5 * x^4 + 10 * x^3 + 10 * x^2 + 5 * x + 1 ≠ n^2 :=
by sorry

end no_integer_x_square_l1435_143518


namespace mul_18396_9999_l1435_143569

theorem mul_18396_9999 :
  18396 * 9999 = 183941604 :=
by
  sorry

end mul_18396_9999_l1435_143569


namespace find_f_of_3_l1435_143523

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (y : ℝ) (h : y > 0) : f ((4 * y + 1) / (y + 1)) = 1 / y

theorem find_f_of_3 : f 3 = 0.5 :=
by
  have y := 2.0
  sorry

end find_f_of_3_l1435_143523


namespace amount_paid_is_51_l1435_143502

def original_price : ℕ := 204
def discount_fraction : ℚ := 0.75
def paid_fraction : ℚ := 1 - discount_fraction

theorem amount_paid_is_51 : paid_fraction * original_price = 51 := by
  sorry

end amount_paid_is_51_l1435_143502


namespace minimum_value_fraction_1_x_plus_1_y_l1435_143554

theorem minimum_value_fraction_1_x_plus_1_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 4) :
  1 / x + 1 / y = 1 :=
sorry

end minimum_value_fraction_1_x_plus_1_y_l1435_143554


namespace hyperbola_asymptotes_iff_l1435_143520

def hyperbola_asymptotes_orthogonal (a b c d e f : ℝ) : Prop :=
  a + c = 0

theorem hyperbola_asymptotes_iff (a b c d e f : ℝ) :
  (∃ x y : ℝ, a * x^2 + 2 * b * x * y + c * y^2 + d * x + e * y + f = 0) →
  hyperbola_asymptotes_orthogonal a b c d e f ↔ a + c = 0 :=
by sorry

end hyperbola_asymptotes_iff_l1435_143520


namespace trapezoidal_field_base_count_l1435_143506

theorem trapezoidal_field_base_count
  (A : ℕ) (h : ℕ) (b1 b2 : ℕ)
  (hdiv8 : ∃ m n : ℕ, b1 = 8 * m ∧ b2 = 8 * n)
  (area_eq : A = (h * (b1 + b2)) / 2)
  (A_val : A = 1400)
  (h_val : h = 50) :
  (∃ pair1 pair2 pair3, (pair1 + pair2 + pair3 = (b1 + b2))) :=
by
  sorry

end trapezoidal_field_base_count_l1435_143506


namespace range_of_a_l1435_143596

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (2 * a + 1) * x + a^2 + a < 0 → 0 < 2 * x - 1 ∧ 2 * x - 1 ≤ 10) →
  (∃ l u : ℝ, (l = 1/2) ∧ (u = 9/2) ∧ (l ≤ a ∧ a ≤ u)) :=
by
  sorry

end range_of_a_l1435_143596


namespace find_abc_l1435_143501

theorem find_abc (a b c : ℚ) 
  (h1 : a + b + c = 24)
  (h2 : a + 2 * b = 2 * c)
  (h3 : a = b / 2) : 
  a = 16 / 3 ∧ b = 32 / 3 ∧ c = 8 := 
by 
  sorry

end find_abc_l1435_143501


namespace total_steps_traveled_l1435_143566

def steps_per_mile : ℕ := 2000
def walk_to_subway : ℕ := 2000
def subway_ride_miles : ℕ := 7
def walk_to_rockefeller : ℕ := 3000
def cab_ride_miles : ℕ := 3

theorem total_steps_traveled :
  walk_to_subway +
  (subway_ride_miles * steps_per_mile) +
  walk_to_rockefeller +
  (cab_ride_miles * steps_per_mile)
  = 24000 := 
by 
  sorry

end total_steps_traveled_l1435_143566


namespace solve_quadratic_l1435_143549

def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem solve_quadratic : (quadratic_eq (-2) 1 3 (-1)) ∧ (quadratic_eq (-2) 1 3 (3/2)) :=
by
  sorry

end solve_quadratic_l1435_143549


namespace petya_winning_probability_l1435_143583

noncomputable def petya_wins_probability : ℚ :=
  (1 / 4) ^ 4

-- The main theorem statement
theorem petya_winning_probability :
  petya_wins_probability = 1 / 256 :=
by sorry

end petya_winning_probability_l1435_143583


namespace sum_of_decimals_l1435_143521

theorem sum_of_decimals : (5.76 + 4.29 = 10.05) :=
by
  sorry

end sum_of_decimals_l1435_143521


namespace symmetric_points_ab_value_l1435_143500

theorem symmetric_points_ab_value
  (a b : ℤ)
  (h₁ : a + 2 = -4)
  (h₂ : 2 = b) :
  a * b = -12 :=
by
  sorry

end symmetric_points_ab_value_l1435_143500


namespace distinct_real_solutions_exist_l1435_143512

theorem distinct_real_solutions_exist (a : ℝ) (h : a > 3 / 4) : 
  ∃ (x y : ℝ), x ≠ y ∧ x = a - y^2 ∧ y = a - x^2 := 
sorry

end distinct_real_solutions_exist_l1435_143512


namespace area_of_shaded_rectangle_l1435_143541

theorem area_of_shaded_rectangle (w₁ h₁ w₂ h₂: ℝ) 
  (hw₁: w₁ * h₁ = 6)
  (hw₂: w₂ * h₁ = 15)
  (hw₃: w₂ * h₂ = 25) :
  w₁ * h₂ = 10 :=
by
  sorry

end area_of_shaded_rectangle_l1435_143541


namespace joey_speed_on_way_back_eq_six_l1435_143597

theorem joey_speed_on_way_back_eq_six :
  ∃ (v : ℝ), 
    (∀ (d t : ℝ), 
      d = 2 ∧ t = 1 →  -- Joey runs a 2-mile distance in 1 hour
      (∀ (d_total t_avg : ℝ),
        d_total = 4 ∧ t_avg = 3 →  -- Round trip distance is 4 miles with average speed 3 mph
        (3 = 4 / (1 + 2 / v) → -- Given average speed equation
         v = 6))) := sorry

end joey_speed_on_way_back_eq_six_l1435_143597


namespace domain_sqrt_sin_cos_l1435_143508

open Real

theorem domain_sqrt_sin_cos (k : ℤ) :
  {x : ℝ | ∃ k : ℤ, (2 * k * π + π / 4 ≤ x) ∧ (x ≤ 2 * k * π + 5 * π / 4)} = 
  {x : ℝ | sin x - cos x ≥ 0} :=
sorry

end domain_sqrt_sin_cos_l1435_143508


namespace processing_rates_and_total_cost_l1435_143576

variables (products total_days total_days_A total_days_B daily_capacity_A daily_capacity_B total_cost_A total_cost_B : ℝ)

noncomputable def A_processing_rate : ℝ := daily_capacity_A
noncomputable def B_processing_rate : ℝ := daily_capacity_B

theorem processing_rates_and_total_cost
  (h1 : products = 1000)
  (h2 : total_days_A = total_days_B + 10)
  (h3 : daily_capacity_B = 1.25 * daily_capacity_A)
  (h4 : total_cost_A = 100 * total_days_A)
  (h5 : total_cost_B = 125 * total_days_B) :
  (daily_capacity_A = 20) ∧ (daily_capacity_B = 25) ∧ (total_cost_A + total_cost_B = 5000) :=
by
  sorry

end processing_rates_and_total_cost_l1435_143576


namespace counterexample_to_prime_condition_l1435_143580

theorem counterexample_to_prime_condition :
  ¬(Prime 54) ∧ ¬(Prime 52) ∧ ¬(Prime 51) := by
  -- Proof not required
  sorry

end counterexample_to_prime_condition_l1435_143580


namespace largest_perimeter_polygons_meeting_at_A_l1435_143575

theorem largest_perimeter_polygons_meeting_at_A
  (n : ℕ) 
  (r : ℝ)
  (h1 : n ≥ 3)
  (h2 : 2 * 180 * (n - 2) / n + 60 = 360) :
  2 * n * 2 = 24 := 
by
  sorry

end largest_perimeter_polygons_meeting_at_A_l1435_143575


namespace probability_two_dice_sum_gt_8_l1435_143514

def num_ways_to_get_sum_at_most_8 := 
  1 + 2 + 3 + 4 + 5 + 6 + 5

def total_outcomes := 36

def probability_sum_greater_than_8 : ℚ := 1 - (num_ways_to_get_sum_at_most_8 / total_outcomes)

theorem probability_two_dice_sum_gt_8 :
  probability_sum_greater_than_8 = 5 / 18 :=
by
  sorry

end probability_two_dice_sum_gt_8_l1435_143514


namespace calc1_calc2_calc3_calc4_l1435_143522

-- Problem 1
theorem calc1 : (-2: ℝ) ^ 2 - (7 - Real.pi) ^ 0 - (1 / 3) ^ (-1: ℝ) = 0 := by
  sorry

-- Problem 2
variable (m : ℝ)
theorem calc2 : 2 * m ^ 3 * 3 * m - (2 * m ^ 2) ^ 2 + m ^ 6 / m ^ 2 = 3 * m ^ 4 := by
  sorry

-- Problem 3
variable (a : ℝ)
theorem calc3 : (a + 1) ^ 2 + (a + 1) * (a - 2) = 2 * a ^ 2 + a - 1 := by
  sorry

-- Problem 4
variables (x y : ℝ)
theorem calc4 : (x + y - 1) * (x - y - 1) = x ^ 2 - 2 * x + 1 - y ^ 2 := by
  sorry

end calc1_calc2_calc3_calc4_l1435_143522


namespace printer_cost_comparison_l1435_143556

-- Definitions based on the given conditions
def in_store_price : ℝ := 150.00
def discount_rate : ℝ := 0.10
def installment_payment : ℝ := 28.00
def number_of_installments : ℕ := 5
def shipping_handling_charge : ℝ := 12.50

-- Discounted in-store price calculation
def discounted_in_store_price : ℝ := in_store_price * (1 - discount_rate)

-- Total cost from the television advertiser
def tv_advertiser_total_cost : ℝ := (number_of_installments * installment_payment) + shipping_handling_charge

-- Proof statement
theorem printer_cost_comparison :
  discounted_in_store_price - tv_advertiser_total_cost = -17.50 :=
by
  sorry

end printer_cost_comparison_l1435_143556


namespace no_zonk_probability_l1435_143594

theorem no_zonk_probability (Z C G : ℕ) (total_boxes : ℕ := 3) (tables : ℕ := 3)
  (no_zonk_prob : ℚ := 2 / 3) : (no_zonk_prob ^ tables) = 8 / 27 :=
by
  -- Here we would prove the theorem, but for the purpose of this task, we skip the proof.
  sorry

end no_zonk_probability_l1435_143594


namespace range_f1_l1435_143526
open Function

theorem range_f1 (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Ici (-1) → y ∈ Set.Ici (-1) → x ≤ y → (x^2 + 2*a*x + 3) ≤ (y^2 + 2*a*y + 3)) →
  6 ≤ (1^2 + 2*a*1 + 3) :=
by
  intro h
  sorry

end range_f1_l1435_143526


namespace value_of_a_plus_b_l1435_143511

def f (x : ℝ) (a b : ℝ) := x^3 + (a - 1) * x^2 + a * x + b

theorem value_of_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) → a + b = 1 :=
by
  sorry

end value_of_a_plus_b_l1435_143511


namespace possible_values_of_N_l1435_143557

theorem possible_values_of_N (N : ℕ) (h1 : N ≥ 8 + 1)
  (h2 : ∀ (i : ℕ), (i < N → (i ≥ 0 ∧ i < 1/3 * (N-1)) → false) ) 
  (h3 : ∀ (i : ℕ), (i < N → (i ≥ 1/3 * (N-1) ∨ i < 1/3 * (N-1)) → true)) :
  23 ≤ N ∧ N ≤ 25 :=
by
  sorry

end possible_values_of_N_l1435_143557


namespace pages_wed_calculation_l1435_143595

def pages_mon : ℕ := 23
def pages_tue : ℕ := 38
def pages_thu : ℕ := 12
def pages_fri : ℕ := 2 * pages_thu
def total_pages : ℕ := 158

theorem pages_wed_calculation (pages_wed : ℕ) : 
  pages_mon + pages_tue + pages_wed + pages_thu + pages_fri = total_pages → pages_wed = 61 :=
by
  intros h
  sorry

end pages_wed_calculation_l1435_143595


namespace function_equality_l1435_143565

theorem function_equality (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x : ℝ, f x = (1/2) * x^2 - x + (3/2) :=
by
  sorry

end function_equality_l1435_143565


namespace smaller_angle_at_3_15_l1435_143558

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end smaller_angle_at_3_15_l1435_143558


namespace abs_sum_bound_l1435_143515

theorem abs_sum_bound (x : ℝ) (a : ℝ) (h : |x - 4| + |x - 3| < a) (ha : 0 < a) : 1 < a :=
by
  sorry

end abs_sum_bound_l1435_143515


namespace simplify_expression_l1435_143567

theorem simplify_expression (x : ℤ) : (3 * x) ^ 3 + (2 * x) * (x ^ 4) = 27 * x ^ 3 + 2 * x ^ 5 :=
by sorry

end simplify_expression_l1435_143567


namespace solve_by_completing_square_l1435_143553

theorem solve_by_completing_square (x: ℝ) (h: x^2 + 4 * x - 3 = 0) : (x + 2)^2 = 7 := 
by 
  sorry

end solve_by_completing_square_l1435_143553


namespace chandler_saves_for_laptop_l1435_143568

theorem chandler_saves_for_laptop :
  ∃ x : ℕ, 140 + 20 * x = 800 ↔ x = 33 :=
by
  use 33
  sorry

end chandler_saves_for_laptop_l1435_143568


namespace age_sum_is_27_l1435_143532

noncomputable def a : ℕ := 12
noncomputable def b : ℕ := 10
noncomputable def c : ℕ := 5

theorem age_sum_is_27
  (h1: a = b + 2)
  (h2: b = 2 * c)
  (h3: b = 10) :
  a + b + c = 27 :=
  sorry

end age_sum_is_27_l1435_143532


namespace womenInBusinessClass_l1435_143584

-- Given conditions
def totalPassengers : ℕ := 300
def percentageWomen : ℚ := 70 / 100
def percentageWomenBusinessClass : ℚ := 15 / 100

def numberOfWomen (totalPassengers : ℕ) (percentageWomen : ℚ) : ℚ := 
  totalPassengers * percentageWomen

def numberOfWomenBusinessClass (numberOfWomen : ℚ) (percentageWomenBusinessClass : ℚ) : ℚ := 
  numberOfWomen * percentageWomenBusinessClass

-- Theorem to prove
theorem womenInBusinessClass (totalPassengers : ℕ) (percentageWomen : ℚ) (percentageWomenBusinessClass : ℚ) :
  numberOfWomenBusinessClass (numberOfWomen totalPassengers percentageWomen) percentageWomenBusinessClass = 32 := 
by 
  -- The proof steps would go here
  sorry

end womenInBusinessClass_l1435_143584


namespace roots_polynomial_value_l1435_143559

theorem roots_polynomial_value (r s t : ℝ) (h₁ : r + s + t = 15) (h₂ : r * s + s * t + t * r = 25) (h₃ : r * s * t = 10) :
  (1 + r) * (1 + s) * (1 + t) = 51 :=
by
  sorry

end roots_polynomial_value_l1435_143559


namespace geom_seq_sum_half_l1435_143563

theorem geom_seq_sum_half (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∃ L, L = ∑' n, a n ∧ L = 1 / 2) (h_abs : |q| < 1) :
  a 0 ∈ (Set.Ioo 0 (1 / 2)) ∪ (Set.Ioo (1 / 2) 1) :=
sorry

end geom_seq_sum_half_l1435_143563


namespace multiple_of_3_l1435_143572

theorem multiple_of_3 (a b : ℤ) (h1 : ∃ m : ℤ, a = 3 * m) (h2 : ∃ n : ℤ, b = 9 * n) : ∃ k : ℤ, a + b = 3 * k :=
by
  sorry

end multiple_of_3_l1435_143572


namespace length_MN_of_circle_l1435_143599

def point := ℝ × ℝ

def circle_passing_through (A B C: point) :=
  ∃ (D E F : ℝ), ∀ (p : point), p = A ∨ p = B ∨ p = C →
    (p.1^2 + p.2^2 + D * p.1 + E * p.2 + F = 0)

theorem length_MN_of_circle (A B C : point) (H : circle_passing_through A B C) :
  A = (1, 3) → B = (4, 2) → C = (1, -7) →
  ∃ M N : ℝ, (A.1 * 0 + N^2 + D * 0 + E * N + F = 0) ∧ (A.1 * 0 + M^2 + D * 0 + E * M + F = 0) ∧
  abs (M - N) = 4 * Real.sqrt 6 := 
sorry

end length_MN_of_circle_l1435_143599


namespace product_of_two_numbers_l1435_143538

theorem product_of_two_numbers 
  (x y : ℝ) 
  (h₁ : x - y = 8) 
  (h₂ : x^2 + y^2 = 160) 
  : x * y = 48 := 
sorry

end product_of_two_numbers_l1435_143538


namespace quadratic_inequality_m_range_l1435_143573

theorem quadratic_inequality_m_range (m : ℝ) : (∀ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ (m ≠ 0) :=
by
  sorry

end quadratic_inequality_m_range_l1435_143573


namespace bouquets_needed_to_earn_1000_l1435_143529

theorem bouquets_needed_to_earn_1000 :
  ∀ (cost_per_bouquet sell_price_bouquet: ℕ) (roses_per_bouquet_bought roses_per_bouquet_sold target_profit: ℕ),
    cost_per_bouquet = 20 →
    sell_price_bouquet = 20 →
    roses_per_bouquet_bought = 7 →
    roses_per_bouquet_sold = 5 →
    target_profit = 1000 →
    (target_profit / (sell_price_bouquet * roses_per_bouquet_sold / roses_per_bouquet_bought - cost_per_bouquet) * roses_per_bouquet_bought = 125) :=
by
  intros cost_per_bouquet sell_price_bouquet roses_per_bouquet_bought roses_per_bouquet_sold target_profit 
    h_cost_per_bouquet h_sell_price_bouquet h_roses_per_bouquet_bought h_roses_per_bouquet_sold h_target_profit
  sorry

end bouquets_needed_to_earn_1000_l1435_143529


namespace min_value_m_plus_n_l1435_143548

theorem min_value_m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : 45 * m = n^3) : m + n = 90 :=
sorry

end min_value_m_plus_n_l1435_143548


namespace divisors_congruent_mod8_l1435_143564

theorem divisors_congruent_mod8 (n : ℕ) (hn : n % 2 = 1) :
  ∀ d, d ∣ (2^n - 1) → d % 8 = 1 ∨ d % 8 = 7 :=
by
  sorry

end divisors_congruent_mod8_l1435_143564


namespace remove_parentheses_correct_l1435_143543

variable {a b c : ℝ}

theorem remove_parentheses_correct :
  -(a - b) = -a + b :=
by sorry

end remove_parentheses_correct_l1435_143543


namespace part1_l1435_143562

variable (a b c : ℝ) (A B : ℝ)
variable (triangle_abc : Triangle ABC)
variable (cos : ℝ → ℝ)

axiom law_of_cosines : ∀ {a b c A : ℝ}, a^2 = b^2 + c^2 - 2 * b * c * cos A

theorem part1 (h1 : b^2 + 3 * a * c * (a^2 + c^2 - b^2) / (2 * a * c) = 2 * c^2) (h2 : a = c) : A = π / 4 := 
sorry

end part1_l1435_143562


namespace probability_event_A_probability_event_B_probability_event_C_l1435_143545

-- Define the total number of basic events for three dice
def total_basic_events : ℕ := 6 * 6 * 6

-- Define events and their associated basic events
def event_A_basic_events : ℕ := 2 * 3 * 3
def event_B_basic_events : ℕ := 2 * 3 * 6
def event_C_basic_events : ℕ := 6 * 6 * 3

-- Define probabilities for each event
def P_A : ℚ := event_A_basic_events / total_basic_events
def P_B : ℚ := event_B_basic_events / total_basic_events
def P_C : ℚ := event_C_basic_events / total_basic_events

-- Statement to be proven
theorem probability_event_A : P_A = 1 / 12 := by
  sorry

theorem probability_event_B : P_B = 1 / 6 := by
  sorry

theorem probability_event_C : P_C = 1 / 2 := by
  sorry

end probability_event_A_probability_event_B_probability_event_C_l1435_143545


namespace sandwiches_prepared_l1435_143507

variable (S : ℕ)
variable (H1 : S > 0)
variable (H2 : ∃ r : ℕ, r = S / 4)
variable (H3 : ∃ b : ℕ, b = (3 * S / 4) / 6)
variable (H4 : ∃ c : ℕ, c = 2 * b)
variable (H5 : ∃ x : ℕ, 5 * x = 5)
variable (H6 : 3 * S / 8 - 5 = 4)

theorem sandwiches_prepared : S = 24 :=
by
  sorry

end sandwiches_prepared_l1435_143507


namespace extreme_value_proof_l1435_143542

noncomputable def extreme_value (x y : ℝ) := 4 * x + 3 * y 

theorem extreme_value_proof 
  (x y : ℝ)
  (hx : 0 < x)
  (hy : 0 < y)
  (h : x + y = 5 * x * y) : 
  extreme_value x y = 3 :=
sorry

end extreme_value_proof_l1435_143542


namespace minimum_value_f_l1435_143550

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / 2 + 2 / (Real.sin x)

theorem minimum_value_f (x : ℝ) (h : 0 < x ∧ x ≤ Real.pi / 2) :
  ∃ y, (∀ z, 0 < z ∧ z ≤ Real.pi / 2 → f z ≥ y) ∧ y = 5 / 2 :=
sorry

end minimum_value_f_l1435_143550


namespace y_completion_time_l1435_143593

noncomputable def work_done (days : ℕ) (rate : ℚ) : ℚ := days * rate

theorem y_completion_time (X_days Y_remaining_days : ℕ) (X_rate Y_days : ℚ) :
  X_days = 40 →
  work_done 8 (1 / X_days) = 1 / 5 →
  work_done Y_remaining_days (4 / 5 / Y_remaining_days) = 4 / 5 →
  Y_days = 35 :=
by
  intros hX hX_work_done hY_work_done
  -- With the stated conditions, we should be able to conclude that Y_days is 35.
  sorry

end y_completion_time_l1435_143593


namespace min_value_x_y_l1435_143533

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1 / x + 4 / y + 8) : 
  x + y ≥ 9 :=
sorry

end min_value_x_y_l1435_143533


namespace factor_of_7_l1435_143574

theorem factor_of_7 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : 7 ∣ (a + 2 * b)) : 7 ∣ (100 * a + 11 * b) :=
by sorry

end factor_of_7_l1435_143574


namespace julien_contribution_l1435_143592

def exchange_rate : ℝ := 1.5
def cost_of_pie : ℝ := 12
def lucas_cad : ℝ := 10

theorem julien_contribution : (cost_of_pie - lucas_cad / exchange_rate) = 16 / 3 := by
  sorry

end julien_contribution_l1435_143592


namespace range_of_a_l1435_143561

theorem range_of_a (a : ℝ) (h : ∅ ⊂ {x : ℝ | x^2 ≤ a}) : 0 ≤ a :=
by
  sorry

end range_of_a_l1435_143561


namespace trajectory_of_midpoint_l1435_143570

theorem trajectory_of_midpoint (x y x₀ y₀ : ℝ) :
  (y₀ = 2 * x₀ ^ 2 + 1) ∧ (x = (x₀ + 0) / 2) ∧ (y = (y₀ + 1) / 2) →
  y = 4 * x ^ 2 + 1 :=
by sorry

end trajectory_of_midpoint_l1435_143570


namespace car_interval_length_l1435_143547

theorem car_interval_length (S1 T : ℝ) (interval_length : ℝ) 
  (h1 : S1 = 39) 
  (h2 : (fun (n : ℕ) => S1 - 3 * n) 4 = 27)
  (h3 : 3.6 = 27 * T) 
  (h4 : interval_length = T * 60) :
  interval_length = 8 :=
by
  sorry

end car_interval_length_l1435_143547


namespace linear_system_solution_l1435_143536

/-- Given a system of three linear equations:
      x + y + z = 1
      a x + b y + c z = h
      a² x + b² y + c² z = h²
    Prove that the solution x, y, z is given by:
    x = (h - b)(h - c) / (a - b)(a - c)
    y = (h - a)(h - c) / (b - a)(b - c)
    z = (h - a)(h - b) / (c - a)(c - b) -/
theorem linear_system_solution (a b c h : ℝ) (x y z : ℝ) :
  x + y + z = 1 →
  a * x + b * y + c * z = h →
  a^2 * x + b^2 * y + c^2 * z = h^2 →
  x = (h - b) * (h - c) / ((a - b) * (a - c)) ∧
  y = (h - a) * (h - c) / ((b - a) * (b - c)) ∧
  z = (h - a) * (h - b) / ((c - a) * (c - b)) :=
by
  intros
  sorry

end linear_system_solution_l1435_143536


namespace sequence_divisible_by_11_l1435_143544

theorem sequence_divisible_by_11 {a : ℕ → ℕ} (h1 : a 1 = 1) (h2 : a 2 = 3)
    (h_rec : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
    (a 4 % 11 = 0) ∧ (a 8 % 11 = 0) ∧ (a 10 % 11 = 0) ∧ (∀ n, n ≥ 11 → a n % 11 = 0) :=
by
  sorry

end sequence_divisible_by_11_l1435_143544


namespace mandatory_state_tax_rate_l1435_143589

theorem mandatory_state_tax_rate 
  (MSRP : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) (tax_rate : ℝ) 
  (insurance_cost : ℝ := insurance_rate * MSRP)
  (cost_before_tax : ℝ := MSRP + insurance_cost)
  (tax_amount : ℝ := total_paid - cost_before_tax) :
  MSRP = 30 → total_paid = 54 → insurance_rate = 0.2 → 
  tax_amount / cost_before_tax * 100 = tax_rate →
  tax_rate = 50 :=
by
  intros MSRP_val paid_val ins_rate_val comp_tax_rate
  sorry

end mandatory_state_tax_rate_l1435_143589


namespace solve_inequality_l1435_143517

theorem solve_inequality (x : ℝ) : (1 / (x + 2) + 4 / (x + 8) ≤ 3 / 4) ↔ ((-8 < x ∧ x ≤ -4) ∨ (-4 ≤ x ∧ x ≤ 4 / 3)) ∧ x ≠ -2 ∧ x ≠ -8 :=
by
  sorry

end solve_inequality_l1435_143517


namespace min_pencils_to_ensure_18_l1435_143540

theorem min_pencils_to_ensure_18 :
  ∀ (total red green yellow blue brown black : ℕ),
  total = 120 → red = 35 → green = 23 → yellow = 14 → blue = 26 → brown = 11 → black = 11 →
  ∃ (n : ℕ), n = 88 ∧
  (∀ (picked_pencils : ℕ → ℕ), (
    (picked_pencils 0 + picked_pencils 1 + picked_pencils 2 + picked_pencils 3 + picked_pencils 4 + picked_pencils 5 = n) →
    (picked_pencils 0 ≤ red) → (picked_pencils 1 ≤ green) → (picked_pencils 2 ≤ yellow) →
    (picked_pencils 3 ≤ blue) → (picked_pencils 4 ≤ brown) → (picked_pencils 5 ≤ black) →
    (picked_pencils 0 ≥ 18 ∨ picked_pencils 1 ≥ 18 ∨ picked_pencils 2 ≥ 18 ∨ picked_pencils 3 ≥ 18 ∨ picked_pencils 4 ≥ 18 ∨ picked_pencils 5 ≥ 18)
  )) := 
sorry

end min_pencils_to_ensure_18_l1435_143540


namespace angle_B_l1435_143534

-- Define the conditions
variables {A B C : ℝ} (a b c : ℝ)
variable (h : a^2 + c^2 = b^2 + ac)

-- State the theorem
theorem angle_B (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  B = π / 3 :=
sorry

end angle_B_l1435_143534


namespace meetings_percent_l1435_143585

/-- Define the lengths of the meetings and total workday in minutes -/
def first_meeting : ℕ := 40
def second_meeting : ℕ := 80
def second_meeting_overlap : ℕ := 10
def third_meeting : ℕ := 30
def workday_minutes : ℕ := 8 * 60

/-- Define the effective duration of the second meeting -/
def effective_second_meeting : ℕ := second_meeting - second_meeting_overlap

/-- Define the total time spent in meetings -/
def total_meeting_time : ℕ := first_meeting + effective_second_meeting + third_meeting

/-- Define the percentage of the workday spent in meetings -/
noncomputable def percent_meeting_time : ℚ := (total_meeting_time * 100 : ℕ) / workday_minutes

/-- Theorem: Given Laura's workday and meeting durations, prove that the percent of her workday spent in meetings is approximately 29.17%. -/
theorem meetings_percent {epsilon : ℚ} (h : epsilon = 0.01) : abs (percent_meeting_time - 29.17) < epsilon :=
sorry

end meetings_percent_l1435_143585


namespace second_number_value_l1435_143598

theorem second_number_value (x y : ℝ) (h1 : (1/5) * x = (5/8) * y) 
                                      (h2 : x + 35 = 4 * y) : y = 40 := 
by 
  sorry

end second_number_value_l1435_143598


namespace sum_of_numerator_and_denominator_of_repeating_decimal_l1435_143552

theorem sum_of_numerator_and_denominator_of_repeating_decimal :
  let x := 0.45
  let a := 9 -- GCD of 45 and 99
  let numerator := 5
  let denominator := 11
  numerator + denominator = 16 :=
by { 
  sorry 
}

end sum_of_numerator_and_denominator_of_repeating_decimal_l1435_143552


namespace genevieve_drinks_pints_l1435_143551

theorem genevieve_drinks_pints (total_gallons : ℝ) (thermoses : ℕ) 
  (gallons_to_pints : ℝ) (genevieve_thermoses : ℕ) 
  (h1 : total_gallons = 4.5) (h2 : thermoses = 18) 
  (h3 : gallons_to_pints = 8) (h4 : genevieve_thermoses = 3) : 
  (total_gallons * gallons_to_pints / thermoses) * genevieve_thermoses = 6 := 
by
  admit

end genevieve_drinks_pints_l1435_143551


namespace students_like_burgers_l1435_143577

theorem students_like_burgers (total_students : ℕ) (french_fries_likers : ℕ) (both_likers : ℕ) (neither_likers : ℕ) 
    (h1 : total_students = 25) (h2 : french_fries_likers = 15) (h3 : both_likers = 6) (h4 : neither_likers = 6) : 
    (total_students - neither_likers) - (french_fries_likers - both_likers) = 10 :=
by
  -- The proof will go here.
  sorry

end students_like_burgers_l1435_143577


namespace train_waiting_probability_l1435_143527

-- Conditions
def trains_per_hour : ℕ := 1
def total_minutes : ℕ := 60
def wait_time : ℕ := 10

-- Proposition
theorem train_waiting_probability : 
  (wait_time : ℝ) / (total_minutes / trains_per_hour) = 1 / 6 :=
by
  -- Here we assume the proof proceeds correctly
  sorry

end train_waiting_probability_l1435_143527


namespace factorial_trailing_digits_l1435_143539

theorem factorial_trailing_digits (n : ℕ) :
  ¬ ∃ k : ℕ, (n! / 10^k) % 10000 = 1976 ∧ k > 0 := 
sorry

end factorial_trailing_digits_l1435_143539


namespace auction_site_TVs_correct_l1435_143571

-- Define the number of TVs Beatrice looked at in person
def in_person_TVs : Nat := 8

-- Define the number of TVs Beatrice looked at online
def online_TVs : Nat := 3 * in_person_TVs

-- Define the total number of TVs Beatrice looked at
def total_TVs : Nat := 42

-- Define the number of TVs Beatrice looked at on the auction site
def auction_site_TVs : Nat := total_TVs - (in_person_TVs + online_TVs)

-- Prove that the number of TVs Beatrice looked at on the auction site is 10
theorem auction_site_TVs_correct : auction_site_TVs = 10 :=
by
  sorry

end auction_site_TVs_correct_l1435_143571
