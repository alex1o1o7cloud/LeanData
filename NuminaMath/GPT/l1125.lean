import Mathlib

namespace total_cost_calc_l1125_112544

variable (a b : ℝ)

def total_cost (a b : ℝ) := 2 * a + 3 * b

theorem total_cost_calc (a b : ℝ) : total_cost a b = 2 * a + 3 * b := by
  sorry

end total_cost_calc_l1125_112544


namespace time_for_A_and_D_together_l1125_112568

theorem time_for_A_and_D_together (A_rate D_rate combined_rate : ℝ)
  (hA : A_rate = 1 / 10) (hD : D_rate = 1 / 10) 
  (h_combined : combined_rate = A_rate + D_rate) :
  1 / combined_rate = 5 :=
by
  sorry

end time_for_A_and_D_together_l1125_112568


namespace solve_system_of_equations_l1125_112560

theorem solve_system_of_equations :
  ∃ x y : ℝ, 4 * x - 6 * y = -3 ∧ 9 * x + 3 * y = 6.3 ∧ x = 0.436 ∧ y = 0.792 :=
by
  sorry

end solve_system_of_equations_l1125_112560


namespace ocean_depth_350_l1125_112583

noncomputable def depth_of_ocean (total_height : ℝ) (volume_ratio_above_water : ℝ) : ℝ :=
  let volume_ratio_below_water := 1 - volume_ratio_above_water
  let height_below_water := (volume_ratio_below_water^(1 / 3)) * total_height
  total_height - height_below_water

theorem ocean_depth_350 :
  depth_of_ocean 10000 (1 / 10) = 350 :=
by
  sorry

end ocean_depth_350_l1125_112583


namespace problem_statement_l1125_112599

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem problem_statement (a b c : ℝ) (h : f a b c (-5) = 3) : f a b c 5 + f a b c (-5) = 4 :=
by
  sorry

end problem_statement_l1125_112599


namespace distributeCandies_l1125_112527

-- Define the conditions as separate definitions.

-- Number of candies
def candies : ℕ := 10

-- Number of boxes
def boxes : ℕ := 5

-- Condition that each box gets at least one candy
def atLeastOne (candyDist : Fin boxes → ℕ) : Prop :=
  ∀ b, candyDist b > 0

-- Function to count the number of ways to distribute candies
noncomputable def countWaysToDistribute (candies : ℕ) (boxes : ℕ) : ℕ :=
  -- Function to compute the number of ways
  -- (assuming a correct implementation is provided)
  sorry -- Placeholder for the actual counting implementation

-- Theorem to prove the number of distributions
theorem distributeCandies : countWaysToDistribute candies boxes = 7 := 
by {
  -- Proof omitted
  sorry
}

end distributeCandies_l1125_112527


namespace nancy_pics_uploaded_l1125_112562

theorem nancy_pics_uploaded (a b n : ℕ) (h₁ : a = 11) (h₂ : b = 8) (h₃ : n = 5) : a + b * n = 51 := 
by 
  sorry

end nancy_pics_uploaded_l1125_112562


namespace find_y1_l1125_112513

theorem find_y1 
  (y1 y2 y3 : ℝ) 
  (h₀ : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h₁ : (1 - y1)^2 + 2 * (y1 - y2)^2 + 2 * (y2 - y3)^2 + y3^2 = 1 / 2) :
  y1 = (2 * Real.sqrt 2 - 1) / (2 * Real.sqrt 2) :=
by
  sorry

end find_y1_l1125_112513


namespace initial_sum_simple_interest_l1125_112505

theorem initial_sum_simple_interest :
  ∃ P : ℝ, (P * (3/100) + P * (5/100) + P * (4/100) + P * (6/100) = 100) ∧ (P = 5000 / 9) :=
by
  sorry

end initial_sum_simple_interest_l1125_112505


namespace movie_theater_people_l1125_112511

def totalSeats : ℕ := 750
def emptySeats : ℕ := 218
def peopleWatching := totalSeats - emptySeats

theorem movie_theater_people :
  peopleWatching = 532 := by
  sorry

end movie_theater_people_l1125_112511


namespace positive_number_is_25_over_9_l1125_112563

variable (a : ℚ) (x : ℚ)

theorem positive_number_is_25_over_9 
  (h1 : 2 * a - 1 = -a + 3)
  (h2 : ∃ r : ℚ, r^2 = x ∧ (r = 2 * a - 1 ∨ r = -a + 3)) : 
  x = 25 / 9 := 
by
  sorry

end positive_number_is_25_over_9_l1125_112563


namespace moles_of_HCl_used_l1125_112566

theorem moles_of_HCl_used (moles_amyl_alcohol : ℕ) (moles_product : ℕ) : 
  moles_amyl_alcohol = 2 ∧ moles_product = 2 → moles_amyl_alcohol = 2 :=
by
  sorry

end moles_of_HCl_used_l1125_112566


namespace range_of_m_plus_n_l1125_112551

theorem range_of_m_plus_n (f : ℝ → ℝ) (n m : ℝ)
  (h_f_def : ∀ x, f x = x^2 + n * x + m)
  (h_non_empty : ∃ x, f x = 0 ∧ f (f x) = 0)
  (h_condition : ∀ x, f x = 0 ↔ f (f x) = 0) :
  0 < m + n ∧ m + n < 4 :=
by {
  -- Proof needed here; currently skipped
  sorry
}

end range_of_m_plus_n_l1125_112551


namespace total_cost_correct_l1125_112525

noncomputable def total_cost : ℝ :=
  let first_path_area := 5 * 100
  let first_path_cost := first_path_area * 2
  let second_path_area := 4 * 80
  let second_path_cost := second_path_area * 1.5
  let diagonal_length := Real.sqrt ((100:ℝ)^2 + (80:ℝ)^2)
  let third_path_area := 6 * diagonal_length
  let third_path_cost := third_path_area * 3
  let circular_path_area := Real.pi * (10:ℝ)^2
  let circular_path_cost := circular_path_area * 4
  first_path_cost + second_path_cost + third_path_cost + circular_path_cost

theorem total_cost_correct : total_cost = 5040.64 := by
  sorry

end total_cost_correct_l1125_112525


namespace proof_BH_length_equals_lhs_rhs_l1125_112509

noncomputable def calculate_BH_length : ℝ :=
  let AB := 3
  let BC := 4
  let CA := 5
  let AG := 4  -- Since AB < AG
  let AH := 6  -- AG < AH
  let GI := 3
  let HI := 8
  let GH := Real.sqrt (GI ^ 2 + HI ^ 2)
  let p := 3
  let q := 2
  let r := 73
  let s := 1
  3 + 2 * Real.sqrt 73

theorem proof_BH_length_equals_lhs_rhs :
  let BH := 3 + 2 * Real.sqrt 73
  calculate_BH_length = BH := by
    sorry

end proof_BH_length_equals_lhs_rhs_l1125_112509


namespace y1_gt_y2_l1125_112534

theorem y1_gt_y2 (k : ℝ) (y1 y2 : ℝ) 
  (h1 : y1 = (-1)^2 - 4*(-1) + k) 
  (h2 : y2 = 3^2 - 4*3 + k) : 
  y1 > y2 := 
by
  sorry

end y1_gt_y2_l1125_112534


namespace neg_a_pow4_div_neg_a_eq_neg_a_pow3_l1125_112521

variable (a : ℝ)

theorem neg_a_pow4_div_neg_a_eq_neg_a_pow3 : (-a)^4 / (-a) = -a^3 := sorry

end neg_a_pow4_div_neg_a_eq_neg_a_pow3_l1125_112521


namespace parabola_directrix_symmetry_l1125_112570

theorem parabola_directrix_symmetry:
  (∃ (d : ℝ), (∀ x : ℝ, x = d ↔ 
  (∃ y : ℝ, y^2 = (1 / 2) * x) ∧
  (∀ y : ℝ, x = (1 / 8)) → x = - (1 / 8))) :=
sorry

end parabola_directrix_symmetry_l1125_112570


namespace negation_exists_eq_forall_l1125_112555

theorem negation_exists_eq_forall (h : ¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) : ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := 
by
  sorry

end negation_exists_eq_forall_l1125_112555


namespace base_number_unique_l1125_112584

theorem base_number_unique (y : ℕ) : (3 : ℝ) ^ 16 = (9 : ℝ) ^ y → y = 8 → (9 : ℝ) = 3 ^ (16 / y) :=
by
  sorry

end base_number_unique_l1125_112584


namespace quadratic_no_real_roots_l1125_112537

theorem quadratic_no_real_roots :
  ¬ (∃ x : ℝ, x^2 - 2 * x + 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0) ∧ (x2^2 - 3 * x2 = 0) ∧
  ∃ y : ℝ, y^2 - 2 * y + 1 = 0 :=
by
  sorry

end quadratic_no_real_roots_l1125_112537


namespace triangle_angle_C_and_equilateral_l1125_112501

variables (a b c A B C : ℝ)
variables (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
variables (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1)

theorem triangle_angle_C_and_equilateral (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
                                         (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
                                         (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1) :
  C = π / 3 ∧ A = π / 3 ∧ B = π / 3 :=
sorry

end triangle_angle_C_and_equilateral_l1125_112501


namespace max_goods_purchased_l1125_112506

theorem max_goods_purchased (initial_spend : ℕ) (reward_rate : ℕ → ℕ → ℕ) (continuous_reward : Prop) :
  initial_spend = 7020 →
  (∀ x y, reward_rate x y = (x / y) * 20) →
  continuous_reward →
  initial_spend + reward_rate initial_spend 100 + reward_rate (reward_rate initial_spend 100) 100 + 
  reward_rate (reward_rate (reward_rate initial_spend 100) 100) 100 = 8760 :=
by
  intros h1 h2 h3
  sorry

end max_goods_purchased_l1125_112506


namespace students_per_group_l1125_112579

theorem students_per_group (total_students not_picked_groups groups : ℕ) (h₁ : total_students = 65) (h₂ : not_picked_groups = 17) (h₃ : groups = 8) :
  (total_students - not_picked_groups) / groups = 6 := by
  sorry

end students_per_group_l1125_112579


namespace framed_painting_ratio_l1125_112598

/-- A rectangular painting measuring 20" by 30" is to be framed, with the longer dimension vertical.
The width of the frame at the top and bottom is three times the width of the frame on the sides.
Given that the total area of the frame equals the area of the painting, the ratio of the smaller to the 
larger dimension of the framed painting is 4:7. -/
theorem framed_painting_ratio : 
  ∀ (w h : ℝ) (side_frame_width : ℝ), 
    w = 20 ∧ h = 30 ∧ 3 * side_frame_width * (2 * (w + 2 * side_frame_width) + 2 * (h + 6 * side_frame_width) - w * h) = w * h 
    → side_frame_width = 2 
    → (w + 2 * side_frame_width) / (h + 6 * side_frame_width) = 4 / 7 :=
sorry

end framed_painting_ratio_l1125_112598


namespace base_nine_to_base_ten_conversion_l1125_112517

theorem base_nine_to_base_ten_conversion : 
  (2 * 9^3 + 8 * 9^2 + 4 * 9^1 + 7 * 9^0 = 2149) := 
by 
  sorry

end base_nine_to_base_ten_conversion_l1125_112517


namespace total_roses_l1125_112574

theorem total_roses (a : ℕ) (x y k : ℕ) (h1 : 300 ≤ a) (h2 : a ≤ 400)
  (h3 : a = 21 * x + 13) (h4 : a = 15 * y - 8) (h5 : a + 8 = 105 * k) :
  a = 307 :=
sorry

end total_roses_l1125_112574


namespace quadratic_two_real_roots_quadratic_no_real_roots_l1125_112571

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k ≤ 9 / 8 :=
by
  sorry

theorem quadratic_no_real_roots (k : ℝ) :
  ¬ (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k > 9 / 8 :=
by
  sorry

end quadratic_two_real_roots_quadratic_no_real_roots_l1125_112571


namespace b_range_given_conditions_l1125_112593

theorem b_range_given_conditions 
    (b c : ℝ)
    (roots_in_interval : ∀ x, x^2 + b * x + c = 0 → -1 ≤ x ∧ x ≤ 1)
    (ineq : 0 ≤ 3 * b + c ∧ 3 * b + c ≤ 3) :
    0 ≤ b ∧ b ≤ 2 :=
sorry

end b_range_given_conditions_l1125_112593


namespace james_passenger_count_l1125_112529

theorem james_passenger_count :
  ∀ (total_vehicles trucks buses taxis motorbikes cars trucks_population buses_population taxis_population motorbikes_population cars_population : ℕ),
  total_vehicles = 52 →
  trucks = 12 →
  buses = 2 →
  taxis = 2 * buses →
  motorbikes = total_vehicles - (trucks + buses + taxis + cars) →
  cars = 30 →
  trucks_population = 2 →
  buses_population = 15 →
  taxis_population = 2 →
  motorbikes_population = 1 →
  cars_population = 3 →
  (trucks * trucks_population + buses * buses_population + taxis * taxis_population +
   motorbikes * motorbikes_population + cars * cars_population) = 156 := 
by
  -- Placeholder for the proof
  sorry

end james_passenger_count_l1125_112529


namespace club_membership_l1125_112576

def total_people_in_club (T B TB N : ℕ) : ℕ :=
  T + B - TB + N

theorem club_membership : total_people_in_club 138 255 94 11 = 310 := by
  sorry

end club_membership_l1125_112576


namespace horizontal_length_circumference_l1125_112564

noncomputable def ratio := 16 / 9
noncomputable def diagonal := 32
noncomputable def computed_length := 32 * 16 / (Real.sqrt 337)
noncomputable def computed_perimeter := 2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337))

theorem horizontal_length 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  32 * 16 / (Real.sqrt 337) = 512 / (Real.sqrt 337) :=
by sorry

theorem circumference 
  (ratio : ℝ := 16 / 9) (diagonal : ℝ := 32) : 
  2 * (32 * 16 / (Real.sqrt 337) + 32 * 9 / (Real.sqrt 337)) = 1600 / (Real.sqrt 337) :=
by sorry

end horizontal_length_circumference_l1125_112564


namespace problem1_proof_problem2_proof_l1125_112532

-- Problem 1 proof statement
theorem problem1_proof : (-1)^10 * 2 + (-2)^3 / 4 = 0 := 
by
  sorry

-- Problem 2 proof statement
theorem problem2_proof : -24 * (5 / 6 - 4 / 3 + 3 / 8) = 3 :=
by
  sorry

end problem1_proof_problem2_proof_l1125_112532


namespace final_acid_concentration_l1125_112523

def volume1 : ℝ := 2
def concentration1 : ℝ := 0.40
def volume2 : ℝ := 3
def concentration2 : ℝ := 0.60

theorem final_acid_concentration :
  ((concentration1 * volume1 + concentration2 * volume2) / (volume1 + volume2)) = 0.52 :=
by
  sorry

end final_acid_concentration_l1125_112523


namespace imaginary_unit_problem_l1125_112594

theorem imaginary_unit_problem (i : ℂ) (h : i^2 = -1) :
  ( (1 + i) / i )^2014 = 2^(1007 : ℤ) * i :=
by sorry

end imaginary_unit_problem_l1125_112594


namespace jessica_final_balance_l1125_112526

variable (B : ℝ) (withdrawal : ℝ) (deposit : ℝ)

-- Conditions
def condition1 : Prop := withdrawal = (2 / 5) * B
def condition2 : Prop := deposit = (1 / 5) * (B - withdrawal)

-- Proof goal statement
theorem jessica_final_balance (h1 : condition1 B withdrawal)
                             (h2 : condition2 B withdrawal deposit) :
    (B - withdrawal + deposit) = 360 :=
by
  sorry

end jessica_final_balance_l1125_112526


namespace no_perfect_squares_l1125_112567

theorem no_perfect_squares (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0)
  (h5 : x * y - z * t = x + y) (h6 : x + y = z + t) : ¬(∃ a b : ℕ, a^2 = x * y ∧ b^2 = z * t) := 
by
  sorry

end no_perfect_squares_l1125_112567


namespace not_divides_two_pow_n_sub_one_l1125_112557

theorem not_divides_two_pow_n_sub_one (n : ℕ) (h1 : n > 1) : ¬ n ∣ (2^n - 1) :=
sorry

end not_divides_two_pow_n_sub_one_l1125_112557


namespace inequality_abc_l1125_112515

variable (a b c : ℝ)

theorem inequality_abc (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  a / (a^3 - a^2 + 3) + b / (b^3 - b^2 + 3) + c / (c^3 - c^2 + 3) ≤ 1 := 
sorry

end inequality_abc_l1125_112515


namespace parabola_expression_l1125_112533

theorem parabola_expression :
  ∃ a b : ℝ, (∀ x : ℝ, a * x^2 + b * x - 5 = 0 → (x = -1 ∨ x = 5)) ∧ (a * (-1)^2 + b * (-1) - 5 = 0) ∧ (a * 5^2 + b * 5 - 5 = 0) ∧ (a * 1 - 4 = 1) :=
sorry

end parabola_expression_l1125_112533


namespace James_age_after_x_years_l1125_112585

variable (x : ℕ)
variable (Justin Jessica James : ℕ)

-- Define the conditions
theorem James_age_after_x_years 
  (H1 : Justin = 26) 
  (H2 : Jessica = Justin + 6) 
  (H3 : James = Jessica + 7)
  (H4 : James + 5 = 44) : 
  James + x = 39 + x := 
by 
  -- proof steps go here 
  sorry

end James_age_after_x_years_l1125_112585


namespace find_d_l1125_112543

-- Conditions
variables (c d : ℝ)
axiom ratio_cond : c / d = 4
axiom eq_cond : c = 20 - 6 * d

theorem find_d : d = 2 :=
by
  sorry

end find_d_l1125_112543


namespace inequality_sqrt_ab_l1125_112550

theorem inequality_sqrt_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  2 / (1 / a + 1 / b) ≤ Real.sqrt (a * b) :=
sorry

end inequality_sqrt_ab_l1125_112550


namespace john_weekly_calories_l1125_112503

-- Define the calorie calculation for each meal type
def breakfast_calories : ℝ := 500
def morning_snack_calories : ℝ := 150
def lunch_calories : ℝ := breakfast_calories + 0.25 * breakfast_calories
def afternoon_snack_calories : ℝ := lunch_calories - 0.30 * lunch_calories
def dinner_calories : ℝ := 2 * lunch_calories

-- Total calories for Friday
def friday_calories : ℝ := breakfast_calories + morning_snack_calories + lunch_calories + afternoon_snack_calories + dinner_calories

-- Additional treats on Saturday and Sunday
def dessert_calories : ℝ := 350
def energy_drink_calories : ℝ := 220

-- Total calories for each day
def saturday_calories : ℝ := friday_calories + dessert_calories
def sunday_calories : ℝ := friday_calories + 2 * energy_drink_calories
def weekday_calories : ℝ := friday_calories

-- Proof statement
theorem john_weekly_calories : 
  friday_calories = 2962.5 ∧ 
  saturday_calories = 3312.5 ∧ 
  sunday_calories = 3402.5 ∧ 
  weekday_calories = 2962.5 :=
by 
  -- proof expressions would go here
  sorry

end john_weekly_calories_l1125_112503


namespace mandy_more_cinnamon_l1125_112546

def cinnamon : ℝ := 0.67
def nutmeg : ℝ := 0.5

theorem mandy_more_cinnamon : cinnamon - nutmeg = 0.17 :=
by
  sorry

end mandy_more_cinnamon_l1125_112546


namespace alex_integer_list_count_l1125_112519

theorem alex_integer_list_count : 
  let n := 12 
  let least_multiple := 2^6 * 3^3
  let count := least_multiple / n
  count = 144 :=
by
  sorry

end alex_integer_list_count_l1125_112519


namespace remainder_of_x_500_div_x2_plus_1_x2_minus_1_l1125_112512

theorem remainder_of_x_500_div_x2_plus_1_x2_minus_1 :
  (x^500) % ((x^2 + 1) * (x^2 - 1)) = 1 :=
sorry

end remainder_of_x_500_div_x2_plus_1_x2_minus_1_l1125_112512


namespace neg_real_root_condition_l1125_112502

theorem neg_real_root_condition (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ (0 < a ∧ a ≤ 1) ∨ (a < 0) :=
by
  sorry

end neg_real_root_condition_l1125_112502


namespace isabella_hair_ratio_l1125_112500

-- Conditions in the problem
variable (hair_before : ℕ) (hair_after : ℕ)
variable (hb : hair_before = 18)
variable (ha : hair_after = 36)

-- Definitions based on conditions
def hair_ratio (after : ℕ) (before : ℕ) : ℚ := (after : ℚ) / (before : ℚ)

theorem isabella_hair_ratio : 
  hair_ratio hair_after hair_before = 2 :=
by
  -- plug in the known values
  rw [hb, ha]
  -- show the equation
  norm_num
  sorry

end isabella_hair_ratio_l1125_112500


namespace find_a2_b2_c2_l1125_112538

-- Define the roots, sum of the roots, sum of the product of the roots taken two at a time, and product of the roots
variables {a b c : ℝ}
variable (h_roots : a = b ∧ b = c)
variable (h_sum : a + b + c = 12)
variable (h_sum_products : a * b + b * c + a * c = 47)
variable (h_product : a * b * c = 30)

-- State the theorem
theorem find_a2_b2_c2 : (a^2 + b^2 + c^2) = 50 :=
by {
  sorry
}

end find_a2_b2_c2_l1125_112538


namespace jerry_total_mean_l1125_112530

def receivedFromAunt : ℕ := 9
def receivedFromUncle : ℕ := 9
def receivedFromBestFriends : List ℕ := [22, 23, 22, 22]
def receivedFromSister : ℕ := 7

def totalAmountReceived : ℕ :=
  receivedFromAunt + receivedFromUncle +
  receivedFromBestFriends.sum + receivedFromSister

def totalNumberOfGifts : ℕ :=
  1 + 1 + receivedFromBestFriends.length + 1

def meanAmountReceived : ℚ :=
  totalAmountReceived / totalNumberOfGifts

theorem jerry_total_mean :
  meanAmountReceived = 16.29 := by
sorry

end jerry_total_mean_l1125_112530


namespace clock_angle_5_30_l1125_112591

theorem clock_angle_5_30 (h_degree : ℕ → ℝ) (m_degree : ℕ → ℝ) (hours_pos : ℕ → ℝ) :
  (h_degree 12 = 360) →
  (m_degree 60 = 360) →
  (hours_pos 5 + h_degree 1 - (m_degree 30 / 2) = 165) →
  (m_degree 30 = 180) →
  ∃ θ : ℝ, θ = abs (m_degree 30 - (hours_pos 5 + h_degree 1 - (m_degree 30 / 2))) ∧ θ = 15 :=
by
  sorry

end clock_angle_5_30_l1125_112591


namespace arithmetic_expression_l1125_112596

theorem arithmetic_expression : 8 / 4 + 5 * 2 ^ 2 - (3 + 7) = 12 := by
  sorry

end arithmetic_expression_l1125_112596


namespace joey_pills_sum_one_week_l1125_112587

def joey_pills (n : ℕ) : ℕ :=
  1 + 2 * n

theorem joey_pills_sum_one_week : 
  (joey_pills 0) + (joey_pills 1) + (joey_pills 2) + (joey_pills 3) + (joey_pills 4) + (joey_pills 5) + (joey_pills 6) = 49 :=
by
  sorry

end joey_pills_sum_one_week_l1125_112587


namespace value_of_a_l1125_112575

theorem value_of_a (a : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ (∃ (y : ℝ), y = 2 ∧ 9 = a ^ y)) : a = 3 := 
  by sorry

end value_of_a_l1125_112575


namespace part_a_part_b_l1125_112553

-- Definition for bishops not attacking each other
def bishops_safe (positions : List (ℕ × ℕ)) : Prop :=
  ∀ (b1 b2 : ℕ × ℕ), b1 ∈ positions → b2 ∈ positions → b1 ≠ b2 → 
    (b1.1 + b1.2 ≠ b2.1 + b2.2) ∧ (b1.1 - b1.2 ≠ b2.1 - b2.2)

-- Part (a): 14 bishops on an 8x8 chessboard such that no two attack each other
theorem part_a : ∃ (positions : List (ℕ × ℕ)), positions.length = 14 ∧ bishops_safe positions := 
by
  sorry

-- Part (b): It is impossible to place 15 bishops on an 8x8 chessboard without them attacking each other
theorem part_b : ¬ ∃ (positions : List (ℕ × ℕ)), positions.length = 15 ∧ bishops_safe positions :=
by 
  sorry

end part_a_part_b_l1125_112553


namespace complement_A_union_B_l1125_112507

def is_positive_integer_less_than_9 (n : ℕ) : Prop :=
  n > 0 ∧ n < 9

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_multiple_of_3 (n : ℕ) : Prop :=
  n % 3 = 0

noncomputable def U := {n : ℕ | is_positive_integer_less_than_9 n}
noncomputable def A := {n ∈ U | is_odd n}
noncomputable def B := {n ∈ U | is_multiple_of_3 n}

theorem complement_A_union_B :
  (U \ (A ∪ B)) = {2, 4, 8} :=
sorry

end complement_A_union_B_l1125_112507


namespace geometric_arithmetic_sum_l1125_112539

open Real

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

noncomputable def arithmetic_mean (x y a b c : ℝ) : Prop :=
  2 * x = a + b ∧ 2 * y = b + c

theorem geometric_arithmetic_sum
  (a b c x y : ℝ)
  (habc : geometric_sequence a b c)
  (hxy : arithmetic_mean x y a b c)
  (hx_ne_zero : x ≠ 0)
  (hy_ne_zero : y ≠ 0) :
  (a / x) + (c / y) = 2 := 
by {
  sorry -- Proof omitted as per the prompt
}

end geometric_arithmetic_sum_l1125_112539


namespace edric_days_per_week_l1125_112578

variable (monthly_salary : ℝ) (hours_per_day : ℝ) (hourly_rate : ℝ) (weeks_per_month : ℝ)
variable (days_per_week : ℝ)

-- Defining the conditions
def monthly_salary_condition : Prop := monthly_salary = 576
def hours_per_day_condition : Prop := hours_per_day = 8
def hourly_rate_condition : Prop := hourly_rate = 3
def weeks_per_month_condition : Prop := weeks_per_month = 4

-- Correct answer
def correct_answer : Prop := days_per_week = 6

-- Proof problem statement
theorem edric_days_per_week :
  monthly_salary_condition monthly_salary ∧
  hours_per_day_condition hours_per_day ∧
  hourly_rate_condition hourly_rate ∧
  weeks_per_month_condition weeks_per_month →
  correct_answer days_per_week :=
by
  sorry

end edric_days_per_week_l1125_112578


namespace discountIs50Percent_l1125_112569

noncomputable def promotionalPrice (originalPrice : ℝ) : ℝ :=
  (2/3) * originalPrice

noncomputable def finalPrice (originalPrice : ℝ) : ℝ :=
  0.75 * promotionalPrice originalPrice

theorem discountIs50Percent (originalPrice : ℝ) (h₁ : originalPrice > 0) :
  finalPrice originalPrice = 0.5 * originalPrice := by
  sorry

end discountIs50Percent_l1125_112569


namespace integers_between_sqrt7_and_sqrt77_l1125_112508

theorem integers_between_sqrt7_and_sqrt77 : 
  2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 ∧ 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 →
  ∃ (n : ℕ), n = 6 ∧ ∀ (k : ℕ), (3 ≤ k ∧ k ≤ 8) ↔ (2 < Real.sqrt 7 ∧ Real.sqrt 77 < 9) :=
by sorry

end integers_between_sqrt7_and_sqrt77_l1125_112508


namespace friends_contribution_l1125_112592

theorem friends_contribution (x : ℝ) 
  (h1 : 4 * (x - 5) = 0.75 * 4 * x) : 
  0.75 * 4 * x = 60 :=
by 
  sorry

end friends_contribution_l1125_112592


namespace remainder_div_150_by_4_eq_2_l1125_112582

theorem remainder_div_150_by_4_eq_2 :
  (∃ k : ℕ, k > 0 ∧ 120 % k^2 = 24) → 150 % 4 = 2 :=
by
  intro h
  sorry

end remainder_div_150_by_4_eq_2_l1125_112582


namespace distance_between_cities_l1125_112573

theorem distance_between_cities (d : ℝ)
  (meeting_point1 : d - 437 + 437 = d)
  (meeting_point2 : 3 * (d - 437) = 2 * d - 237) :
  d = 1074 :=
by
  sorry

end distance_between_cities_l1125_112573


namespace problem_statement_l1125_112572

/-- 
  Theorem: If the solution set of the inequality (ax-1)(x+2) > 0 is -3 < x < -2, 
  then a equals -1/3 
--/
theorem problem_statement (a : ℝ) :
  (forall x, (ax-1)*(x+2) > 0 -> -3 < x ∧ x < -2) → a = -1/3 := 
by
  sorry

end problem_statement_l1125_112572


namespace trapezium_other_parallel_side_l1125_112545

theorem trapezium_other_parallel_side (a b h : ℝ) (area : ℝ) (h_area : area = (1 / 2) * (a + b) * h) (h_a : a = 18) (h_h : h = 20) (h_area_val : area = 380) :
  b = 20 :=
by 
  sorry

end trapezium_other_parallel_side_l1125_112545


namespace find_value_of_k_l1125_112595

theorem find_value_of_k (k x : ℝ) 
  (h : 1 / (4 - x ^ 2) + 2 = k / (x - 2)) : 
  k = -1 / 4 :=
by
  sorry

end find_value_of_k_l1125_112595


namespace total_cups_l1125_112556

theorem total_cups (b f s : ℕ) (ratio_bt_f_s : b / s = 1 / 5) (ratio_fl_b_s : f / s = 8 / 5) (sugar_cups : s = 10) :
  b + f + s = 28 :=
sorry

end total_cups_l1125_112556


namespace avg_chem_math_l1125_112504

-- Given conditions
variables (P C M : ℕ)
axiom total_marks : P + C + M = P + 130

-- The proof problem
theorem avg_chem_math : (C + M) / 2 = 65 :=
by sorry

end avg_chem_math_l1125_112504


namespace alpha_beta_value_l1125_112540

noncomputable def alpha_beta_sum : ℝ := 75

theorem alpha_beta_value (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : |Real.sin α - (1 / 2)| + Real.sqrt (Real.tan β - 1) = 0) :
  α + β = α_beta_sum := 
  sorry

end alpha_beta_value_l1125_112540


namespace minimum_value_l1125_112536

theorem minimum_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_eq : 2 * m + n = 4) : 
  ∃ (x : ℝ), (x = 2) ∧ (∀ (p q : ℝ), q > 0 → p > 0 → 2 * p + q = 4 → x ≤ (1 / p + 2 / q)) := 
sorry

end minimum_value_l1125_112536


namespace adam_total_cost_l1125_112577

theorem adam_total_cost 
    (sandwiches_count : ℕ)
    (sandwiches_price : ℝ)
    (chips_count : ℕ)
    (chips_price : ℝ)
    (water_count : ℕ)
    (water_price : ℝ)
    (sandwich_discount : sandwiches_count = 4 ∧ sandwiches_price = 4 ∧ sandwiches_count = 3 + 1)
    (tax_rate : ℝ)
    (initial_tax_rate : tax_rate = 0.10)
    (chips_cost : chips_count = 3 ∧ chips_price = 3.50)
    (water_cost : water_count = 2 ∧ water_price = 2) : 
  (3 * sandwiches_price + chips_count * chips_price + water_count * water_price) * (1 + tax_rate) = 29.15 := 
by
  sorry

end adam_total_cost_l1125_112577


namespace capacity_of_other_bottle_l1125_112549

theorem capacity_of_other_bottle (x : ℝ) :
  (16 / 3) * (x / 8) + (16 / 3) = 8 → x = 4 := by
  -- the proof will go here
  sorry

end capacity_of_other_bottle_l1125_112549


namespace tennis_tournament_matches_l1125_112580

noncomputable def total_matches (players: ℕ) : ℕ :=
  players - 1

theorem tennis_tournament_matches :
  total_matches 104 = 103 :=
by
  sorry

end tennis_tournament_matches_l1125_112580


namespace correct_calculation_l1125_112541

theorem correct_calculation (m n : ℝ) : -m^2 * n - 2 * m^2 * n = -3 * m^2 * n :=
by
  sorry

end correct_calculation_l1125_112541


namespace students_liked_strawberries_l1125_112552

theorem students_liked_strawberries : 
  let total_students := 450 
  let students_oranges := 70 
  let students_pears := 120 
  let students_apples := 147 
  let students_strawberries := total_students - (students_oranges + students_pears + students_apples)
  students_strawberries = 113 :=
by
  sorry

end students_liked_strawberries_l1125_112552


namespace magic_8_ball_probability_l1125_112547

theorem magic_8_ball_probability :
  let num_questions := 7
  let num_positive := 3
  let positive_probability := 3 / 7
  let negative_probability := 4 / 7
  let binomial_coefficient := Nat.choose num_questions num_positive
  let total_probability := binomial_coefficient * (positive_probability ^ num_positive) * (negative_probability ^ (num_questions - num_positive))
  total_probability = 242112 / 823543 :=
by
  sorry

end magic_8_ball_probability_l1125_112547


namespace total_suitcases_l1125_112548

-- Definitions based on the conditions in a)
def siblings : ℕ := 4
def suitcases_per_sibling : ℕ := 2
def parents : ℕ := 2
def suitcases_per_parent : ℕ := 3
def suitcases_per_Lily : ℕ := 0

-- The statement to be proved
theorem total_suitcases : (siblings * suitcases_per_sibling) + (parents * suitcases_per_parent) + suitcases_per_Lily = 14 :=
by
  sorry

end total_suitcases_l1125_112548


namespace soda_cost_is_20_l1125_112535

noncomputable def cost_of_soda (b s : ℕ) : Prop :=
  4 * b + 3 * s = 500 ∧ 3 * b + 2 * s = 370

theorem soda_cost_is_20 {b s : ℕ} (h : cost_of_soda b s) : s = 20 :=
  by sorry

end soda_cost_is_20_l1125_112535


namespace num_positive_integers_condition_l1125_112528

theorem num_positive_integers_condition : 
  ∃! n : ℤ, 0 < n ∧ n < 50 ∧ (n + 2) % (50 - n) = 0 :=
by
  sorry

end num_positive_integers_condition_l1125_112528


namespace weeks_to_cover_expense_l1125_112524

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end weeks_to_cover_expense_l1125_112524


namespace max_tiles_l1125_112586

/--
Given a rectangular floor of size 180 cm by 120 cm
and rectangular tiles of size 25 cm by 16 cm, prove that the maximum number of tiles
that can be accommodated on the floor without overlapping, where the tiles' edges
are parallel and abutting the edges of the floor and with no tile overshooting the edges,
is 49 tiles.
-/
theorem max_tiles (floor_len floor_wid tile_len tile_wid : ℕ) (h1 : floor_len = 180)
  (h2 : floor_wid = 120) (h3 : tile_len = 25) (h4 : tile_wid = 16) :
  ∃ max_tiles : ℕ, max_tiles = 49 :=
by
  sorry

end max_tiles_l1125_112586


namespace quadratic_radical_simplified_l1125_112565

theorem quadratic_radical_simplified (a : ℕ) : 
  (∃ (b : ℕ), a = 3 * b^2) -> a = 3 := 
by
  sorry

end quadratic_radical_simplified_l1125_112565


namespace a_beats_b_by_32_meters_l1125_112514

-- Define the known conditions.
def distance_a_in_t : ℕ := 224 -- Distance A runs in 28 seconds
def time_a : ℕ := 28 -- Time A takes to run 224 meters
def distance_b_in_t : ℕ := 224 -- Distance B runs in 32 seconds
def time_b : ℕ := 32 -- Time B takes to run 224 meters

-- Define the speeds.
def speed_a : ℕ := distance_a_in_t / time_a
def speed_b : ℕ := distance_b_in_t / time_b

-- Define the distances each runs in 32 seconds.
def distance_a_in_32_sec : ℕ := speed_a * 32
def distance_b_in_32_sec : ℕ := speed_b * 32

-- The proof statement
theorem a_beats_b_by_32_meters :
  distance_a_in_32_sec - distance_b_in_32_sec = 32 := 
sorry

end a_beats_b_by_32_meters_l1125_112514


namespace four_planes_divide_space_into_fifteen_parts_l1125_112559

-- Define the function that calculates the number of parts given the number of planes.
def parts_divided_by_planes (x : ℕ) : ℕ :=
  (x^3 + 5 * x + 6) / 6

-- Prove that four planes divide the space into 15 parts.
theorem four_planes_divide_space_into_fifteen_parts : parts_divided_by_planes 4 = 15 :=
by sorry

end four_planes_divide_space_into_fifteen_parts_l1125_112559


namespace max_area_proof_l1125_112518

-- Define the original curve
def original_curve (x : ℝ) : ℝ := x^2 + x - 2

-- Reflective symmetry curve about point (p, 2p)
def transformed_curve (p x : ℝ) : ℝ := -x^2 + (4 * p + 1) * x - 4 * p^2 + 2 * p + 2

-- Intersection conditions
def intersecting_curves (p x : ℝ) : Prop :=
original_curve x = transformed_curve p x

-- Range for valid p values
def valid_p (p : ℝ) : Prop := -1 ≤ p ∧ p ≤ 2

-- Prove the problem statement which involves ensuring the curves intersect in the range
theorem max_area_proof :
  ∀ (p : ℝ), valid_p p → ∀ (x : ℝ), intersecting_curves p x →
  ∃ (A : ℝ), A = abs (original_curve x - transformed_curve p x) :=
by
  intros p hp x hx
  sorry

end max_area_proof_l1125_112518


namespace axis_angle_set_l1125_112558

def is_x_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def is_y_axis_angle (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

def is_axis_angle (α : ℝ) : Prop := ∃ n : ℤ, α = (n * Real.pi) / 2

theorem axis_angle_set : 
  (∀ α : ℝ, is_x_axis_angle α ∨ is_y_axis_angle α ↔ is_axis_angle α) :=
by 
  sorry

end axis_angle_set_l1125_112558


namespace value_of_z_l1125_112510

theorem value_of_z (z y : ℝ) (h1 : (12)^3 * z^3 / 432 = y) (h2 : y = 864) : z = 6 :=
by
  sorry

end value_of_z_l1125_112510


namespace doctors_assignment_l1125_112588

theorem doctors_assignment :
  ∃ (assignments : Finset (Fin 3 → Finset (Fin 5))),
    (∀ h ∈ assignments, (∀ i, ∃ j ∈ h i, True) ∧
      ¬(∃ i j, (A ∈ h i ∧ B ∈ h j ∨ A ∈ h j ∧ B ∈ h i)) ∧
      ¬(∃ i j, (C ∈ h i ∧ D ∈ h j ∨ C ∈ h j ∧ D ∈ h i))) ∧
    assignments.card = 84 :=
sorry

end doctors_assignment_l1125_112588


namespace total_ice_cream_amount_l1125_112542

theorem total_ice_cream_amount (ice_cream_friday ice_cream_saturday : ℝ) 
  (h1 : ice_cream_friday = 3.25)
  (h2 : ice_cream_saturday = 0.25) : 
  ice_cream_friday + ice_cream_saturday = 3.50 :=
by
  rw [h1, h2]
  norm_num

end total_ice_cream_amount_l1125_112542


namespace town_population_growth_is_62_percent_l1125_112520

noncomputable def population_growth_proof : ℕ := 
  let p := 22
  let p_square := p * p
  let pop_1991 := p_square
  let pop_2001 := pop_1991 + 150
  let pop_2011 := pop_2001 + 150
  let k := 28  -- Given that 784 = 28^2
  let pop_2011_is_perfect_square := k * k = pop_2011
  let percentage_increase := ((pop_2011 - pop_1991) * 100) / pop_1991
  if pop_2011_is_perfect_square then percentage_increase 
  else 0

theorem town_population_growth_is_62_percent :
  population_growth_proof = 62 :=
by
  sorry

end town_population_growth_is_62_percent_l1125_112520


namespace susan_can_drive_with_50_l1125_112522

theorem susan_can_drive_with_50 (car_efficiency : ℕ) (gas_price : ℕ) (money_available : ℕ) 
  (h1 : car_efficiency = 40) (h2 : gas_price = 5) (h3 : money_available = 50) : 
  car_efficiency * (money_available / gas_price) = 400 :=
by
  sorry

end susan_can_drive_with_50_l1125_112522


namespace call_charge_ratio_l1125_112590

def elvin_jan_total_bill : ℕ := 46
def elvin_feb_total_bill : ℕ := 76
def elvin_internet_charge : ℕ := 16
def elvin_call_charge_ratio : ℕ := 2

theorem call_charge_ratio : 
  (elvin_feb_total_bill - elvin_internet_charge) / (elvin_jan_total_bill - elvin_internet_charge) = elvin_call_charge_ratio := 
by
  sorry

end call_charge_ratio_l1125_112590


namespace Q_is_perfect_square_trinomial_l1125_112531

def is_perfect_square_trinomial (p : ℤ → ℤ) :=
∃ (b : ℤ), ∀ a : ℤ, p a = (a + b) * (a + b)

def P (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2
def Q (a : ℤ) : ℤ := a^2 + 2 * a + 1
def R (a b : ℤ) : ℤ := a^2 + a * b + b^2
def S (a : ℤ) : ℤ := a^2 + 2 * a - 1

theorem Q_is_perfect_square_trinomial : is_perfect_square_trinomial Q :=
sorry -- Proof goes here

end Q_is_perfect_square_trinomial_l1125_112531


namespace days_B_can_finish_alone_l1125_112597

theorem days_B_can_finish_alone (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / x) = (1 / 2 : ℚ) → x = 6 := 
by
  sorry

end days_B_can_finish_alone_l1125_112597


namespace common_ratio_is_2_l1125_112554

noncomputable def common_ratio_of_increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n+1) = a n * q) ∧ (∀ m n, m < n → a m < a n)

theorem common_ratio_is_2
  (a : ℕ → ℝ) (q : ℝ)
  (hgeo : common_ratio_of_increasing_geometric_sequence a q)
  (h1 : a 1 + a 5 = 17)
  (h2 : a 2 * a 4 = 16) :
  q = 2 :=
sorry

end common_ratio_is_2_l1125_112554


namespace solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l1125_112581

theorem solve_quadratic_eq1 : ∀ x : ℝ, 2 * x^2 + 5 * x + 3 = 0 → (x = -3/2 ∨ x = -1) :=
by
  intro x
  intro h
  sorry

theorem solve_quadratic_eq2_complete_square : ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_complete_square_l1125_112581


namespace greatest_integer_for_prime_abs_expression_l1125_112561

open Int

-- Define the quadratic expression and the prime condition
def quadratic_expression (x : ℤ) : ℤ := 6 * x^2 - 47 * x + 15

-- Statement that |quadratic_expression x| is prime
def is_prime_quadratic_expression (x : ℤ) : Prop :=
  Prime (abs (quadratic_expression x))

-- Prove that the greatest integer x such that |quadratic_expression x| is prime is 8
theorem greatest_integer_for_prime_abs_expression :
  ∃ (x : ℤ), is_prime_quadratic_expression x ∧ (∀ (y : ℤ), is_prime_quadratic_expression y → y ≤ x) → x = 8 :=
by
  sorry

end greatest_integer_for_prime_abs_expression_l1125_112561


namespace general_formula_a_n_general_formula_b_n_l1125_112516

-- Prove general formula for the sequence a_n
theorem general_formula_a_n (S : Nat → Nat) (a : Nat → Nat) (h₁ : ∀ n, S n = 2^(n+1) - 2) :
  (∀ n, a n = S n - S (n - 1)) → ∀ n, a n = 2^n :=
by
  sorry

-- Prove general formula for the sequence b_n
theorem general_formula_b_n (a b : Nat → Nat) (h₁ : ∀ n, a n = 2^n) :
  (∀ n, b n = a n + a (n + 1)) → ∀ n, b n = 3 * 2^n :=
by
  sorry

end general_formula_a_n_general_formula_b_n_l1125_112516


namespace highest_degree_has_asymptote_l1125_112589

noncomputable def highest_degree_of_px (denom : ℕ → ℕ) (n : ℕ) : ℕ :=
  let deg := denom n
  deg

theorem highest_degree_has_asymptote (p : ℕ → ℕ) (denom : ℕ → ℕ) (n : ℕ)
  (h_denom : denom n = 6) :
  highest_degree_of_px denom n = 6 := by
  sorry

end highest_degree_has_asymptote_l1125_112589
