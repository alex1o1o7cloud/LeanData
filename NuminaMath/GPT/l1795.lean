import Mathlib

namespace line_passes_through_fixed_point_l1795_179597

theorem line_passes_through_fixed_point (p q : ℝ) (h : 3 * p - 2 * q = 1) :
  p * (-3 / 2) + 3 * (1 / 6) + q = 0 :=
by 
  sorry

end line_passes_through_fixed_point_l1795_179597


namespace trig_identity_l1795_179526

open Real

theorem trig_identity (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 6) (h : sin α ^ 6 + cos α ^ 6 = 7 / 12) : 1998 * cos α = 333 * Real.sqrt 30 :=
sorry

end trig_identity_l1795_179526


namespace pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l1795_179579

-- Definitions based on the problem's conditions
def a (n : Nat) : Nat := n * n

def pos_count (n : Nat) : Nat :=
  List.length (List.filter (λ m : Nat => a m < n) (List.range (n + 1)))

def pos_pos_count (n : Nat) : Nat :=
  pos_count (pos_count n)

-- Theorem statements
theorem pos_count_a5_eq_2 : pos_count 5 = 2 := 
by
  -- Proof would go here
  sorry

theorem pos_pos_count_an_eq_n2 (n : Nat) : pos_pos_count n = n * n :=
by
  -- Proof would go here
  sorry

end pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l1795_179579


namespace jony_stop_block_correct_l1795_179532

-- Jony's walk parameters
def start_time : ℕ := 7 -- In hours, but it is not used directly
def start_block : ℕ := 10
def end_block : ℕ := 90
def stop_time : ℕ := 40 -- Jony stops walking after 40 minutes starting from 07:00
def speed : ℕ := 100 -- meters per minute
def block_length : ℕ := 40 -- meters

-- Function to calculate the stop block given the parameters
def stop_block (start_block end_block stop_time speed block_length : ℕ) : ℕ :=
  let total_distance := stop_time * speed
  let outbound_distance := (end_block - start_block) * block_length
  let remaining_distance := total_distance - outbound_distance
  let blocks_walked_back := remaining_distance / block_length
  end_block - blocks_walked_back

-- The statement to prove
theorem jony_stop_block_correct :
  stop_block start_block end_block stop_time speed block_length = 70 :=
by
  sorry

end jony_stop_block_correct_l1795_179532


namespace sqrt_simplification_l1795_179590

theorem sqrt_simplification : Real.sqrt 360000 = 600 :=
by 
  sorry

end sqrt_simplification_l1795_179590


namespace not_right_angled_triangle_l1795_179561

theorem not_right_angled_triangle 
  (m n : ℝ) 
  (h1 : m > n) 
  (h2 : n > 0)
  : ¬ (m^2 + n^2)^2 = (mn)^2 + (m^2 - n^2)^2 :=
sorry

end not_right_angled_triangle_l1795_179561


namespace trig_identity_l1795_179543

theorem trig_identity : 
  (2 * Real.sin (80 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l1795_179543


namespace solution_is_permutations_l1795_179559

noncomputable def solve_system (x y z : ℤ) : Prop :=
  x^2 = y * z + 1 ∧ y^2 = z * x + 1 ∧ z^2 = x * y + 1

theorem solution_is_permutations (x y z : ℤ) :
  solve_system x y z ↔ (x, y, z) = (1, 0, -1) ∨ (x, y, z) = (1, -1, 0) ∨ (x, y, z) = (0, 1, -1) ∨ (x, y, z) = (0, -1, 1) ∨ (x, y, z) = (-1, 1, 0) ∨ (x, y, z) = (-1, 0, 1) :=
by sorry

end solution_is_permutations_l1795_179559


namespace speed_of_second_train_l1795_179521

/-- Given:
1. The first train has a length of 220 meters.
2. The speed of the first train is 120 kilometers per hour.
3. The time taken to cross each other is 9 seconds.
4. The length of the second train is 280.04 meters.

Prove the speed of the second train is 80 kilometers per hour. -/
theorem speed_of_second_train
    (len_first_train : ℝ := 220)
    (speed_first_train_kmph : ℝ := 120)
    (time_to_cross : ℝ := 9)
    (len_second_train : ℝ := 280.04) 
  : (len_first_train / time_to_cross + len_second_train / time_to_cross - (speed_first_train_kmph * 1000 / 3600)) * (3600 / 1000) = 80 := 
by
  sorry

end speed_of_second_train_l1795_179521


namespace problem_statement_l1795_179536

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (a + 1 / a) ^ 2 + (b + 1 / b) ^ 2 ≥ 25 / 2 := 
by
  sorry

end problem_statement_l1795_179536


namespace range_of_m_l1795_179507

theorem range_of_m (m x1 x2 y1 y2 : ℝ) (h1 : y1 = (1 + 2 * m) / x1) (h2 : y2 = (1 + 2 * m) / x2)
    (hx : x1 < 0 ∧ 0 < x2) (hy : y1 < y2) : m > -1 / 2 :=
sorry

end range_of_m_l1795_179507


namespace calc_expression_l1795_179585

theorem calc_expression :
  (- (2 / 5) : ℝ)^0 - (0.064 : ℝ)^(1/3) + 3^(Real.log (2 / 5) / Real.log 3) + Real.log 2 / Real.log 10 - Real.log (1 / 5) / Real.log 10 = 2 := 
by
  sorry

end calc_expression_l1795_179585


namespace old_selling_price_l1795_179513

theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) :
  C + 0.10 * C = 88 :=
by
  sorry

end old_selling_price_l1795_179513


namespace quadratic_solution_identity_l1795_179515

theorem quadratic_solution_identity (a b : ℤ) (h : (1 : ℤ)^2 + a * 1 + 2 * b = 0) : 2 * a + 4 * b = -2 := by
  sorry

end quadratic_solution_identity_l1795_179515


namespace milk_per_cow_per_day_l1795_179529

-- Define the conditions
def num_cows := 52
def weekly_milk_production := 364000 -- ounces

-- State the theorem
theorem milk_per_cow_per_day :
  (weekly_milk_production / 7 / num_cows) = 1000 := 
by
  -- Here we would include the proof, so we use sorry as placeholder
  sorry

end milk_per_cow_per_day_l1795_179529


namespace original_number_is_400_l1795_179502

theorem original_number_is_400 (x : ℝ) (h : 1.20 * x = 480) : x = 400 :=
sorry

end original_number_is_400_l1795_179502


namespace elephant_entry_rate_l1795_179522

-- Define the variables and constants
def initial_elephants : ℕ := 30000
def exit_rate : ℕ := 2880
def exit_time : ℕ := 4
def enter_time : ℕ := 7
def final_elephants : ℕ := 28980

-- Prove the rate of new elephants entering the park
theorem elephant_entry_rate :
  (final_elephants - (initial_elephants - exit_rate * exit_time)) / enter_time = 1500 :=
by
  sorry -- placeholder for the proof

end elephant_entry_rate_l1795_179522


namespace percentage_employees_four_years_or_more_l1795_179537

theorem percentage_employees_four_years_or_more 
  (x : ℝ) 
  (less_than_one_year : ℝ := 6 * x)
  (one_to_two_years : ℝ := 4 * x)
  (two_to_three_years : ℝ := 7 * x)
  (three_to_four_years : ℝ := 3 * x)
  (four_to_five_years : ℝ := 3 * x)
  (five_to_six_years : ℝ := 1 * x)
  (six_to_seven_years : ℝ := 1 * x)
  (seven_to_eight_years : ℝ := 2 * x)
  (total_employees : ℝ := 27 * x)
  (employees_four_years_or_more : ℝ := 7 * x) : 
  (employees_four_years_or_more / total_employees) * 100 = 25.93 := 
by
  sorry

end percentage_employees_four_years_or_more_l1795_179537


namespace smallest_x_y_z_sum_l1795_179538

theorem smallest_x_y_z_sum :
  ∃ x y z : ℝ, x + 3*y + 6*z = 1 ∧ x*y + 2*x*z + 6*y*z = -8 ∧ x*y*z = 2 ∧ x + y + z = -(8/3) := 
sorry

end smallest_x_y_z_sum_l1795_179538


namespace length_of_plot_is_60_l1795_179531

noncomputable def plot_length (b : ℝ) : ℝ :=
  b + 20

noncomputable def plot_perimeter (b : ℝ) : ℝ :=
  2 * (plot_length b + b)

noncomputable def plot_cost_eq (b : ℝ) : Prop :=
  26.50 * plot_perimeter b = 5300

theorem length_of_plot_is_60 : ∃ b : ℝ, plot_cost_eq b ∧ plot_length b = 60 :=
sorry

end length_of_plot_is_60_l1795_179531


namespace triangle_integer_solutions_l1795_179584

theorem triangle_integer_solutions (x : ℕ) (h1 : 13 < x) (h2 : x < 43) : 
  ∃ (n : ℕ), n = 29 :=
by 
  sorry

end triangle_integer_solutions_l1795_179584


namespace circle_passing_points_l1795_179558

theorem circle_passing_points :
  ∃ (D E F : ℝ), 
    (25 + 1 + 5 * D + E + F = 0) ∧ 
    (36 + 6 * D + F = 0) ∧ 
    (1 + 1 - D + E + F = 0) ∧ 
    (∀ x y : ℝ, (x, y) = (5, 1) ∨ (x, y) = (6, 0) ∨ (x, y) = (-1, 1) → x^2 + y^2 + D * x + E * y + F = 0) → 
  x^2 + y^2 - 4 * x + 6 * y - 12 = 0 :=
by
  sorry

end circle_passing_points_l1795_179558


namespace parabola_tangent_parameter_l1795_179542

theorem parabola_tangent_parameter (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hp : p ≠ 0) :
  ∃ p : ℝ, (∀ y, y^2 + (2 * p * b / a) * y + (2 * p * c^2 / a) = 0) ↔ (p = 2 * a * c^2 / b^2) := 
by
  sorry

end parabola_tangent_parameter_l1795_179542


namespace tan_eq_tan_x2_sol_count_l1795_179580

noncomputable def arctan1000 := Real.arctan 1000

theorem tan_eq_tan_x2_sol_count :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℝ, 
    0 ≤ x ∧ x ≤ arctan1000 ∧ Real.tan x = Real.tan (x^2) →
    ∃ k : ℕ, k < n ∧ x = Real.sqrt (k * Real.pi + x) :=
sorry

end tan_eq_tan_x2_sol_count_l1795_179580


namespace same_speed_4_l1795_179503

theorem same_speed_4 {x : ℝ} (hx : x ≠ -7)
  (H1 : ∀ (x : ℝ), (x^2 - 7*x - 60)/(x + 7) = x - 12) 
  (H2 : ∀ (x : ℝ), x^3 - 5*x^2 - 14*x + 104 = x - 12) :
  ∃ (speed : ℝ), speed = 4 :=
by
  sorry

end same_speed_4_l1795_179503


namespace triangle_angle_l1795_179555

theorem triangle_angle (A B C : ℝ) (h1 : A - C = B) (h2 : A + B + C = 180) : A = 90 :=
by
  sorry

end triangle_angle_l1795_179555


namespace land_for_cattle_l1795_179547

-- Define the conditions as Lean definitions
def total_land : ℕ := 150
def house_and_machinery : ℕ := 25
def future_expansion : ℕ := 15
def crop_production : ℕ := 70

-- Statement to prove
theorem land_for_cattle : total_land - (house_and_machinery + future_expansion + crop_production) = 40 :=
by
  sorry

end land_for_cattle_l1795_179547


namespace ratio_PM_MQ_eq_1_l1795_179533

theorem ratio_PM_MQ_eq_1
  (A B C D E M P Q : ℝ × ℝ)
  (square_side : ℝ)
  (h_square_side : square_side = 15)
  (hA : A = (0, square_side))
  (hB : B = (square_side, square_side))
  (hC : C = (square_side, 0))
  (hD : D = (0, 0))
  (hE : E = (8, 0))
  (hM : M = ((A.1 + E.1) / 2, (A.2 + E.2) / 2))
  (h_slope_AE : E.2 - A.2 = (E.1 - A.1) * -15 / 8)
  (h_P_on_AD : P.2 = 15)
  (h_Q_on_BC : Q.2 = 0)
  (h_PM_len : dist M P = dist M Q) :
  dist P M = dist M Q :=
by sorry

end ratio_PM_MQ_eq_1_l1795_179533


namespace number_of_ordered_triples_l1795_179541

theorem number_of_ordered_triples :
  let b := 2023
  let n := (b ^ 2)
  ∀ (a c : ℕ), a * c = n ∧ a ≤ b ∧ b ≤ c → (∃ (k : ℕ), k = 7) :=
by
  sorry

end number_of_ordered_triples_l1795_179541


namespace sams_charge_per_sheet_is_1_5_l1795_179574

variable (x : ℝ)
variable (a : ℝ) -- John's Photo World's charge per sheet
variable (b : ℝ) -- Sam's Picture Emporium's one-time sitting fee
variable (c : ℝ) -- John's Photo World's one-time sitting fee
variable (n : ℕ) -- Number of sheets

def johnsCost (n : ℕ) (a c : ℝ) := n * a + c
def samsCost (n : ℕ) (x b : ℝ) := n * x + b

theorem sams_charge_per_sheet_is_1_5 :
  ∀ (a b c : ℝ) (n : ℕ), a = 2.75 → b = 140 → c = 125 → n = 12 →
  johnsCost n a c = samsCost n x b → x = 1.50 := by
  intros a b c n ha hb hc hn h
  sorry

end sams_charge_per_sheet_is_1_5_l1795_179574


namespace tim_paid_correct_amount_l1795_179583

-- Define the conditions given in the problem
def mri_cost : ℝ := 1200
def doctor_hourly_rate : ℝ := 300
def doctor_time_hours : ℝ := 0.5 -- 30 minutes is half an hour
def fee_for_being_seen : ℝ := 150
def insurance_coverage_rate : ℝ := 0.80

-- Total amount Tim paid calculation
def total_cost_before_insurance : ℝ :=
  mri_cost + (doctor_hourly_rate * doctor_time_hours) + fee_for_being_seen

def insurance_coverage : ℝ :=
  total_cost_before_insurance * insurance_coverage_rate

def amount_tim_paid : ℝ :=
  total_cost_before_insurance - insurance_coverage

-- Prove that Tim paid $300
theorem tim_paid_correct_amount : amount_tim_paid = 300 :=
by
  sorry

end tim_paid_correct_amount_l1795_179583


namespace product_telescope_identity_l1795_179593

theorem product_telescope_identity :
  (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) * (1 + (1 / 7)) = 8 :=
by
  sorry

end product_telescope_identity_l1795_179593


namespace general_formula_l1795_179598

def sequence_a (n : ℕ) : ℕ :=
by sorry

def partial_sum (n : ℕ) : ℕ :=
by sorry

axiom base_case : partial_sum 1 = 5

axiom recurrence_relation (n : ℕ) (h : 2 ≤ n) : partial_sum (n - 1) = sequence_a n

theorem general_formula (n : ℕ) : partial_sum n = 5 * 2^(n-1) :=
by
-- Proof will be provided here
sorry

end general_formula_l1795_179598


namespace solve_inequality_l1795_179504

noncomputable def rational_inequality_solution (x : ℝ) : Prop :=
  3 - (x^2 - 4 * x - 5) / (3 * x + 2) > 1

theorem solve_inequality (x : ℝ) :
  rational_inequality_solution x ↔ (x > -2 / 3 ∧ x < 9) :=
by
  sorry

end solve_inequality_l1795_179504


namespace floor_x_floor_x_eq_42_l1795_179527

theorem floor_x_floor_x_eq_42 (x : ℝ) : (⌊x * ⌊x⌋⌋ = 42) ↔ (7 ≤ x ∧ x < 43 / 6) :=
by sorry

end floor_x_floor_x_eq_42_l1795_179527


namespace other_train_length_l1795_179501

noncomputable def length_of_other_train
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ) : ℝ :=
  let v1 := (v1_kmph * 1000) / 3600
  let v2 := (v2_kmph * 1000) / 3600
  let relative_speed := v1 + v2
  let total_distance := relative_speed * t
  total_distance - l1

theorem other_train_length
  (l1 : ℝ) (v1_kmph : ℝ) (v2_kmph : ℝ) (t : ℝ)
  (hl1 : l1 = 230)
  (hv1 : v1_kmph = 120)
  (hv2 : v2_kmph = 80)
  (ht : t = 9) :
  length_of_other_train l1 v1_kmph v2_kmph t = 269.95 :=
by
  rw [hl1, hv1, hv2, ht]
  -- Proof steps skipped
  sorry

end other_train_length_l1795_179501


namespace sparrow_swallow_equations_l1795_179569

theorem sparrow_swallow_equations (x y : ℝ) : 
  (5 * x + 6 * y = 16) ∧ (4 * x + y = 5 * y + x) :=
  sorry

end sparrow_swallow_equations_l1795_179569


namespace total_passengers_l1795_179506

theorem total_passengers (P : ℕ)
  (h1 : P / 12 + P / 8 + P / 3 + P / 6 + 35 = P) : 
  P = 120 :=
by
  sorry

end total_passengers_l1795_179506


namespace theresa_crayons_count_l1795_179553

noncomputable def crayons_teresa (initial_teresa_crayons : Nat) 
                                 (initial_janice_crayons : Nat) 
                                 (shared_with_nancy : Nat)
                                 (given_to_mark : Nat)
                                 (received_from_nancy : Nat) : Nat := 
  initial_teresa_crayons + received_from_nancy

theorem theresa_crayons_count : crayons_teresa 32 12 (12 / 2) 3 8 = 40 := by
  -- Given: Theresa initially has 32 crayons.
  -- Janice initially has 12 crayons.
  -- Janice shares half of her crayons with Nancy: 12 / 2 = 6 crayons.
  -- Janice gives 3 crayons to Mark.
  -- Theresa receives 8 crayons from Nancy.
  -- Therefore: Theresa will have 32 + 8 = 40 crayons.
  sorry

end theresa_crayons_count_l1795_179553


namespace problem_1_problem_2_l1795_179508

theorem problem_1 (n : ℕ) (h : n > 0) (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, (n > 0) → 
    (∃ α β, α + β = β * α + 1 ∧ 
            α * β = 1 / a n ∧ 
            a n * α^2 - a (n+1) * α + 1 = 0 ∧ 
            a n * β^2 - a (n+1) * β + 1 = 0)) :
  a (n + 1) = a n + 1 := sorry

theorem problem_2 (n : ℕ) (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n, (n > 0) → a (n+1) = a n + 1) :
  a n = n := sorry

end problem_1_problem_2_l1795_179508


namespace rice_wheat_ratio_l1795_179571

theorem rice_wheat_ratio (total_shi : ℕ) (sample_size : ℕ) (wheat_in_sample : ℕ) (total_sample : ℕ) : 
  total_shi = 1512 ∧ sample_size = 216 ∧ wheat_in_sample = 27 ∧ total_sample = 1512 * (wheat_in_sample / sample_size) →
  total_sample = 189 :=
by
  intros h
  sorry

end rice_wheat_ratio_l1795_179571


namespace number_of_negative_x_l1795_179592

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l1795_179592


namespace village_population_equal_in_15_years_l1795_179575

theorem village_population_equal_in_15_years :
  ∀ n : ℕ, (72000 - 1200 * n = 42000 + 800 * n) → n = 15 :=
by
  intros n h
  sorry

end village_population_equal_in_15_years_l1795_179575


namespace last_digit_of_large_prime_l1795_179544

theorem last_digit_of_large_prime : 
  (859433 = 214858 * 4 + 1) → 
  (∃ d, (2 ^ 859433 - 1) % 10 = d ∧ d = 1) :=
by
  intro h
  sorry

end last_digit_of_large_prime_l1795_179544


namespace range_of_m_is_leq_3_l1795_179578

noncomputable def is_range_of_m (m : ℝ) : Prop :=
  ∀ x : ℝ, 5^x + 3 > m

theorem range_of_m_is_leq_3 (m : ℝ) : is_range_of_m m ↔ m ≤ 3 :=
by
  sorry

end range_of_m_is_leq_3_l1795_179578


namespace bread_cooling_time_l1795_179599

theorem bread_cooling_time 
  (dough_room_temp : ℕ := 60)   -- 1 hour in minutes
  (shape_dough : ℕ := 15)       -- 15 minutes
  (proof_dough : ℕ := 120)      -- 2 hours in minutes
  (bake_bread : ℕ := 30)        -- 30 minutes
  (start_time : ℕ := 2 * 60)    -- 2:00 am in minutes
  (end_time : ℕ := 6 * 60)      -- 6:00 am in minutes
  : (end_time - start_time) - (dough_room_temp + shape_dough + proof_dough + bake_bread) = 15 := 
  by
  sorry

end bread_cooling_time_l1795_179599


namespace division_by_ab_plus_one_is_perfect_square_l1795_179587

theorem division_by_ab_plus_one_is_perfect_square
    (a b : ℕ) (h : 0 < a ∧ 0 < b)
    (hab : (ab + 1) ∣ (a^2 + b^2)) :
    ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) := 
sorry

end division_by_ab_plus_one_is_perfect_square_l1795_179587


namespace car_speed_return_trip_l1795_179557

noncomputable def speed_return_trip (d : ℕ) (v_ab : ℕ) (v_avg : ℕ) : ℕ := 
  (2 * d * v_avg) / (2 * v_avg - v_ab)

theorem car_speed_return_trip :
  let d := 180
  let v_ab := 90
  let v_avg := 60
  speed_return_trip d v_ab v_avg = 45 :=
by
  simp [speed_return_trip]
  sorry

end car_speed_return_trip_l1795_179557


namespace probability_age_21_to_30_l1795_179545

theorem probability_age_21_to_30 : 
  let total_people := 160 
  let people_10_to_20 := 40
  let people_21_to_30 := 70
  let people_31_to_40 := 30
  let people_41_to_50 := 20
  (people_21_to_30 / total_people : ℚ) = 7 / 16 := by
  sorry

end probability_age_21_to_30_l1795_179545


namespace relationship_between_P_and_Q_l1795_179563

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem relationship_between_P_and_Q : 
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
sorry

end relationship_between_P_and_Q_l1795_179563


namespace ratio_wrong_to_correct_l1795_179514

theorem ratio_wrong_to_correct (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36) (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 :=
by {
  -- Proof will go here
  sorry
}

end ratio_wrong_to_correct_l1795_179514


namespace field_trip_people_per_bus_l1795_179594

def number_of_people_on_each_bus (vans buses people_per_van total_people : ℕ) : ℕ :=
  (total_people - (vans * people_per_van)) / buses

theorem field_trip_people_per_bus :
  let vans := 9
  let buses := 10
  let people_per_van := 8
  let total_people := 342
  number_of_people_on_each_bus vans buses people_per_van total_people = 27 :=
by
  sorry

end field_trip_people_per_bus_l1795_179594


namespace michael_cleanings_total_l1795_179581

theorem michael_cleanings_total (baths_per_week : ℕ) (showers_per_week : ℕ) (weeks_in_year : ℕ) 
  (h_baths : baths_per_week = 2) (h_showers : showers_per_week = 1) (h_weeks : weeks_in_year = 52) :
  (baths_per_week + showers_per_week) * weeks_in_year = 156 :=
by 
  -- Omitting proof as instructed.
  sorry

end michael_cleanings_total_l1795_179581


namespace perimeter_of_stadium_l1795_179530

-- Define the length and breadth as given conditions.
def length : ℕ := 100
def breadth : ℕ := 300

-- Define the perimeter function for a rectangle.
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Prove that the perimeter of the stadium is 800 meters given the length and breadth.
theorem perimeter_of_stadium : perimeter length breadth = 800 := 
by
  -- Placeholder for the formal proof.
  sorry

end perimeter_of_stadium_l1795_179530


namespace volume_frustum_correct_l1795_179510

noncomputable def volume_of_frustum : ℚ :=
  let V_original := (1 / 3 : ℚ) * (16^2) * 10
  let V_smaller := (1 / 3 : ℚ) * (8^2) * 5
  V_original - V_smaller

theorem volume_frustum_correct :
  volume_of_frustum = 2240 / 3 :=
by
  sorry

end volume_frustum_correct_l1795_179510


namespace second_number_in_set_l1795_179500

theorem second_number_in_set (avg1 avg2 n1 n2 n3 : ℕ) (h1 : avg1 = (10 + 70 + 19) / 3) (h2 : avg2 = avg1 + 7) (h3 : n1 = 20) (h4 : n3 = 60) :
  n2 = n3 := 
  sorry

end second_number_in_set_l1795_179500


namespace robotics_club_neither_l1795_179562

theorem robotics_club_neither (total_students cs_students e_students both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 52)
  (h3 : e_students = 45)
  (h4 : both_students = 32) :
  total_students - (cs_students - both_students + e_students - both_students + both_students) = 15 :=
by
  sorry

end robotics_club_neither_l1795_179562


namespace geometric_representation_l1795_179549

variables (a : ℝ)

-- Definition of the area of the figure
def total_area := a^2 + 1.5 * a

-- Definition of the perimeter of the figure
def total_perimeter := 4 * a + 3

theorem geometric_representation :
  total_area a = a^2 + 1.5 * a ∧ total_perimeter a = 4 * a + 3 :=
by
  exact ⟨rfl, rfl⟩

end geometric_representation_l1795_179549


namespace Ms_Hatcher_total_students_l1795_179505

noncomputable def number_of_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) : ℕ :=
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem Ms_Hatcher_total_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) 
  (h1 : third_graders = 20)
  (h2 : fourth_graders = 2 * third_graders) 
  (h3 : fifth_graders = third_graders / 2) 
  (h4 : sixth_graders = 3 * (third_graders + fourth_graders) / 4) : 
  number_of_students third_graders fourth_graders fifth_graders sixth_graders = 115 :=
by
  sorry

end Ms_Hatcher_total_students_l1795_179505


namespace probability_less_than_8_rings_l1795_179528

def P_10_ring : ℝ := 0.20
def P_9_ring : ℝ := 0.30
def P_8_ring : ℝ := 0.10

theorem probability_less_than_8_rings : 
  (1 - (P_10_ring + P_9_ring + P_8_ring)) = 0.40 :=
by
  sorry

end probability_less_than_8_rings_l1795_179528


namespace martha_cards_l1795_179560

theorem martha_cards :
  let initial_cards := 3
  let emily_cards := 25
  let alex_cards := 43
  let jenny_cards := 58
  let sam_cards := 14
  initial_cards + emily_cards + alex_cards + jenny_cards - sam_cards = 115 := 
by
  sorry

end martha_cards_l1795_179560


namespace sqrt_fraction_simplified_l1795_179518

theorem sqrt_fraction_simplified :
  Real.sqrt (4 / 3) = 2 * Real.sqrt 3 / 3 :=
by sorry

end sqrt_fraction_simplified_l1795_179518


namespace root_polynomial_satisfies_expression_l1795_179588

noncomputable def roots_of_polynomial (x : ℕ) : Prop :=
  x^3 - 15 * x^2 + 25 * x - 10 = 0

theorem root_polynomial_satisfies_expression (p q r : ℕ) 
    (h1 : roots_of_polynomial p)
    (h2 : roots_of_polynomial q)
    (h3 : roots_of_polynomial r)
    (h_sum : p + q + r = 15)
    (h_prod : p*q + q*r + r*p = 25) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by sorry

end root_polynomial_satisfies_expression_l1795_179588


namespace bike_ride_distance_l1795_179572

-- Definitions for conditions from a)
def speed_out := 24 -- miles per hour
def speed_back := 18 -- miles per hour
def total_time := 7 -- hours

-- Problem statement for the proof problem
theorem bike_ride_distance :
  ∃ (D : ℝ), (D / speed_out) + (D / speed_back) = total_time ∧ 2 * D = 144 :=
by {
  sorry
}

end bike_ride_distance_l1795_179572


namespace encoded_integer_one_less_l1795_179539

theorem encoded_integer_one_less (BDF BEA BFB EAB : ℕ)
  (hBDF : BDF = 1 * 7^2 + 3 * 7 + 6)
  (hBEA : BEA = 1 * 7^2 + 5 * 7 + 0)
  (hBFB : BFB = 1 * 7^2 + 5 * 7 + 1)
  (hEAB : EAB = 5 * 7^2 + 0 * 7 + 1)
  : EAB - 1 = 245 :=
by
  sorry

end encoded_integer_one_less_l1795_179539


namespace total_earnings_of_a_b_c_l1795_179586

theorem total_earnings_of_a_b_c 
  (days_a days_b days_c : ℕ)
  (ratio_a ratio_b ratio_c : ℕ)
  (wage_c : ℕ) 
  (h_ratio : ratio_a * wage_c = 3 * (3 + 4 + 5))
  (h_ratio_a_b : ratio_b = 4 * wage_c / 5 * ratio_a / 60)
  (h_ratio_b_c : ratio_b = 4 * wage_c / 5 * ratio_c / 60):
  (ratio_a * days_a + ratio_b * days_b + ratio_c * days_c) = 1480 := 
  by
    sorry

end total_earnings_of_a_b_c_l1795_179586


namespace min_value_of_quadratic_expression_l1795_179523

theorem min_value_of_quadratic_expression : ∃ x : ℝ, ∀ y : ℝ, y = x^2 + 12*x + 9 → y ≥ -27 :=
sorry

end min_value_of_quadratic_expression_l1795_179523


namespace circle_diameter_given_area_l1795_179565

theorem circle_diameter_given_area : 
  (∃ (r : ℝ), 81 * Real.pi = Real.pi * r^2 ∧ 2 * r = d) → d = 18 := by
  sorry

end circle_diameter_given_area_l1795_179565


namespace correct_sum_is_1826_l1795_179556

-- Define the four-digit number representation
def four_digit (A B C D : ℕ) := 1000 * A + 100 * B + 10 * C + D

-- Condition: Yoongi confused the units digit (9 as 6)
-- The incorrect number Yoongi used
def incorrect_number (A B C : ℕ) := four_digit A B C 6

-- The correct number
def correct_number (A B C : ℕ) := four_digit A B C 9

-- The sum obtained by Yoongi
def yoongi_sum (A B C : ℕ) := incorrect_number A B C + 57

-- The correct sum 
def correct_sum (A B C : ℕ) := correct_number A B C + 57

-- Condition: Yoongi's sum is 1823
axiom yoongi_sum_is_1823 (A B C: ℕ) : yoongi_sum A B C = 1823

-- Proof Problem: Prove that the correct sum is 1826
theorem correct_sum_is_1826 (A B C : ℕ) : correct_sum A B C = 1826 := by
  -- The proof goes here
  sorry

end correct_sum_is_1826_l1795_179556


namespace worker_usual_time_l1795_179551

theorem worker_usual_time (T : ℝ) (S : ℝ) (h₀ : S > 0) (h₁ : (4 / 5) * S * (T + 10) = S * T) : T = 40 :=
sorry

end worker_usual_time_l1795_179551


namespace max_sum_l1795_179548

open Real

theorem max_sum (a b c : ℝ) (h : a^2 + (b^2) / 4 + (c^2) / 9 = 1) : a + b + c ≤ sqrt 14 :=
sorry

end max_sum_l1795_179548


namespace binomial_coefficient_8_5_l1795_179595

theorem binomial_coefficient_8_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_coefficient_8_5_l1795_179595


namespace directrix_of_parabola_l1795_179568

theorem directrix_of_parabola (y : ℝ) : 
  (∃ y : ℝ, x = 1) ↔ (x = (1 / 4 : ℝ) * y^2) := 
sorry

end directrix_of_parabola_l1795_179568


namespace selling_prices_l1795_179566

theorem selling_prices {x y : ℝ} (h1 : y - x = 10) (h2 : (y - 5) - 1.10 * x = 1) :
  x = 40 ∧ y = 50 := by
  sorry

end selling_prices_l1795_179566


namespace probability_individual_selected_l1795_179591

/-- Given a population of 8 individuals, the probability that each 
individual is selected in a simple random sample of size 4 is 1/2. -/
theorem probability_individual_selected :
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  probability = (1 : ℚ) / 2 :=
by
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  sorry

end probability_individual_selected_l1795_179591


namespace value_of_a_plus_b_l1795_179519

theorem value_of_a_plus_b (a b : Int) (h1 : |a| = 1) (h2 : b = -2) : a + b = -1 ∨ a + b = -3 := 
by
  sorry

end value_of_a_plus_b_l1795_179519


namespace quadrilateral_diagonals_l1795_179570

theorem quadrilateral_diagonals (a b c d e f : ℝ) 
  (hac : a > c) 
  (hbd : b ≥ d) 
  (hapc : a = c) 
  (hdiag1 : e^2 = (a - b)^2 + b^2) 
  (hdiag2 : f^2 = (c + b)^2 + b^2) :
  e^4 - f^4 = (a + c) / (a - c) * (d^2 * (2 * a * c + d^2) - b^2 * (2 * a * c + b^2)) :=
by
  sorry

end quadrilateral_diagonals_l1795_179570


namespace distance_between_stations_l1795_179517

theorem distance_between_stations (x : ℕ) 
  (h1 : ∃ (x : ℕ), ∀ t : ℕ, (t * 16 = x ∧ t * 21 = x + 60)) :
  2 * x + 60 = 444 :=
by sorry

end distance_between_stations_l1795_179517


namespace money_given_to_cashier_l1795_179525

theorem money_given_to_cashier (regular_ticket_cost : ℕ) (discount : ℕ) 
  (age1 : ℕ) (age2 : ℕ) (change : ℕ) 
  (h1 : regular_ticket_cost = 109)
  (h2 : discount = 5)
  (h3 : age1 = 6)
  (h4 : age2 = 10)
  (h5 : change = 74)
  (h6 : age1 < 12)
  (h7 : age2 < 12) :
  regular_ticket_cost + regular_ticket_cost + (regular_ticket_cost - discount) + (regular_ticket_cost - discount) + change = 500 :=
by
  sorry

end money_given_to_cashier_l1795_179525


namespace factor_transformation_option_C_l1795_179550

theorem factor_transformation_option_C (y : ℝ) : 
  4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 :=
sorry

end factor_transformation_option_C_l1795_179550


namespace custom_op_4_3_l1795_179516

-- Define the custom operation a * b
def custom_op (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the theorem to be proven
theorem custom_op_4_3 : custom_op 4 3 = 19 := 
by
sorry

end custom_op_4_3_l1795_179516


namespace inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l1795_179534

-- Proof Problem 1
theorem inverse_proportional_t (t : ℝ) (h1 : 1 ≤ t ∧ t ≤ 2023) : t = 1 :=
sorry

-- Proof Problem 2
theorem no_linear_function_2k_times (k : ℝ) (h_pos : 0 < k) : ¬ ∃ a b : ℝ, (a < b) ∧ (∀ x, a ≤ x ∧ x ≤ b → (2 * k * a ≤ k * x + 2 ∧ k * x + 2 ≤ 2 * k * b)) :=
sorry

-- Proof Problem 3
theorem quadratic_function_5_times (a b : ℝ) (h_ab : a < b) (h_quad : ∀ x, a ≤ x ∧ x ≤ b → (5 * a ≤ x^2 - 4 * x - 7 ∧ x^2 - 4 * x - 7 ≤ 5 * b)) :
  (a = -2 ∧ b = 1) ∨ (a = -(11/5) ∧ b = (9 + Real.sqrt 109) / 2) :=
sorry

end inverse_proportional_t_no_linear_function_2k_times_quadratic_function_5_times_l1795_179534


namespace fraction_is_correct_l1795_179576

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem fraction_is_correct : (f (g (f 3))) / (g (f (g 3))) = 59 / 19 :=
by
  sorry

end fraction_is_correct_l1795_179576


namespace ellipse_m_gt_5_l1795_179577

theorem ellipse_m_gt_5 (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m > 5 :=
by
  intros h
  sorry

end ellipse_m_gt_5_l1795_179577


namespace min_value_frac_l1795_179589

variable (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1)

theorem min_value_frac : (1 / a + 4 / b) = 9 :=
by sorry

end min_value_frac_l1795_179589


namespace frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l1795_179567

-- Definitions based on the problem conditions
def machineA_first_class := 150
def machineA_total := 200
def machineB_first_class := 120
def machineB_total := 200
def total_products := machineA_total + machineB_total

-- Frequencies of first-class products
def frequency_machineA : ℚ := machineA_first_class / machineA_total
def frequency_machineB : ℚ := machineB_first_class / machineB_total

-- Values for chi-squared formula
def a := machineA_first_class
def b := machineA_total - machineA_first_class
def c := machineB_first_class
def d := machineB_total - machineB_first_class

-- Given formula for K^2
def K_squared : ℚ := (total_products * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Proof problem statements
theorem frequency_first_class_machineA_is_3_over_4 : frequency_machineA = 3 / 4 := by
  sorry

theorem frequency_first_class_machineB_is_3_over_5 : frequency_machineB = 3 / 5 := by
  sorry

theorem significant_quality_difference : K_squared > 6.635 := by
  sorry

end frequency_first_class_machineA_is_3_over_4_frequency_first_class_machineB_is_3_over_5_significant_quality_difference_l1795_179567


namespace factorize_def_l1795_179512

def factorize_polynomial (p q r : Polynomial ℝ) : Prop :=
  p = q * r

theorem factorize_def (p q r : Polynomial ℝ) :
  factorize_polynomial p q r → p = q * r :=
  sorry

end factorize_def_l1795_179512


namespace find_variable_l1795_179554

def expand : ℤ → ℤ := 3*2*6
    
theorem find_variable (a n some_variable : ℤ) (h : (3 - 7 + a = 3)):
  some_variable = -17 :=
sorry

end find_variable_l1795_179554


namespace determine_S_l1795_179573

theorem determine_S :
  (∃ k : ℝ, (∀ S R T : ℝ, R = k * (S / T)) ∧ (∃ S R T : ℝ, R = 2 ∧ S = 6 ∧ T = 3 ∧ 2 = k * (6 / 3))) →
  (∀ S R T : ℝ, R = 8 ∧ T = 2 → S = 16) :=
by
  sorry

end determine_S_l1795_179573


namespace find_three_digit_number_l1795_179546

theorem find_three_digit_number (A B C : ℕ) (h1 : A + B + C = 10) (h2 : B = A + C) (h3 : 100 * C + 10 * B + A = 100 * A + 10 * B + C + 99) : 100 * A + 10 * B + C = 253 :=
by {
  sorry
}

end find_three_digit_number_l1795_179546


namespace ball_arrangement_l1795_179540

theorem ball_arrangement : ∃ (n : ℕ), n = 120 ∧
  (∀ (ball_count : ℕ), ball_count = 20 → ∃ (box1 box2 box3 : ℕ), 
    box1 ≥ 1 ∧ box2 ≥ 2 ∧ box3 ≥ 3 ∧ box1 + box2 + box3 = ball_count) :=
by
  sorry

end ball_arrangement_l1795_179540


namespace sin_alpha_beta_value_l1795_179520

theorem sin_alpha_beta_value (α β : ℝ) (h1 : 13 * Real.sin α + 5 * Real.cos β = 9) (h2 : 13 * Real.cos α + 5 * Real.sin β = 15) : 
  Real.sin (α + β) = 56 / 65 :=
by
  sorry

end sin_alpha_beta_value_l1795_179520


namespace max_U_value_l1795_179509

noncomputable def maximum_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) : ℝ :=
  x + y

theorem max_U_value (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  maximum_value x y h ≤ Real.sqrt 13 :=
  sorry

end max_U_value_l1795_179509


namespace boy_reaches_early_l1795_179596

theorem boy_reaches_early (usual_rate new_rate : ℝ) (Usual_Time New_Time : ℕ) 
  (Hrate : new_rate = 9/8 * usual_rate) (Htime : Usual_Time = 36) :
  New_Time = 32 → Usual_Time - New_Time = 4 :=
by
  intros
  subst_vars
  sorry

end boy_reaches_early_l1795_179596


namespace seven_people_different_rolls_l1795_179582

def rolls_different (rolls : Fin 7 -> Fin 6) : Prop :=
  ∀ i : Fin 7, rolls i ≠ rolls ⟨(i + 1) % 7, sorry⟩

def probability_rolls_different : ℚ :=
  (625 : ℚ) / 2799

theorem seven_people_different_rolls (rolls : Fin 7 -> Fin 6) :
  (∃ rolls, rolls_different rolls) ->
  probability_rolls_different = 625 / 2799 :=
sorry

end seven_people_different_rolls_l1795_179582


namespace ratio_of_x_to_y_l1795_179524

variable {x y : ℝ}

theorem ratio_of_x_to_y (h1 : (3 * x - 2 * y) / (2 * x + 3 * y) = 5 / 4) (h2 : x + y = 5) : x / y = 23 / 2 := 
by {
  sorry
}

end ratio_of_x_to_y_l1795_179524


namespace x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l1795_179535

theorem x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 3) ∧ ¬(∀ x : ℝ, x > 3 → x > 5) :=
by 
  -- Prove implications with provided conditions
  sorry

end x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l1795_179535


namespace area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l1795_179564

theorem area_of_inscribed_rectangle_not_square (s : ℝ) : 
  (s > 0) ∧ (s < 1 / 2) :=
sorry

theorem area_of_inscribed_rectangle_is_square (s : ℝ) : 
  (s >= 1 / 2) ∧ (s < 1) :=
sorry

end area_of_inscribed_rectangle_not_square_area_of_inscribed_rectangle_is_square_l1795_179564


namespace find_length_of_side_c_find_measure_of_angle_B_l1795_179552

variable {A B C a b c : ℝ}

def triangle_problem (a b c A B C : ℝ) :=
  a * Real.cos B = 3 ∧
  b * Real.cos A = 1 ∧
  A - B = Real.pi / 6 ∧
  a^2 + c^2 - b^2 - 6 * c = 0 ∧
  b^2 + c^2 - a^2 - 2 * c = 0

theorem find_length_of_side_c (h : triangle_problem a b c A B C) :
  c = 4 :=
sorry

theorem find_measure_of_angle_B (h : triangle_problem a b c A B C) :
  B = Real.pi / 6 :=
sorry

end find_length_of_side_c_find_measure_of_angle_B_l1795_179552


namespace most_likely_outcomes_l1795_179511

noncomputable def probability_boy_or_girl : ℚ := 1 / 2

noncomputable def probability_all_boys (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def probability_all_girls (n : ℕ) : ℚ := probability_boy_or_girl^n

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_3_girls_2_boys : ℚ := binom 5 3 * probability_boy_or_girl^5

noncomputable def probability_3_boys_2_girls : ℚ := binom 5 2 * probability_boy_or_girl^5

theorem most_likely_outcomes :
  probability_3_girls_2_boys = 5/16 ∧
  probability_3_boys_2_girls = 5/16 ∧
  probability_all_boys 5 = 1/32 ∧
  probability_all_girls 5 = 1/32 ∧
  (5/16 > 1/32) :=
by
  sorry

end most_likely_outcomes_l1795_179511
