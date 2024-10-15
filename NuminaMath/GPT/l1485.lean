import Mathlib

namespace NUMINAMATH_GPT_analyze_a_b_m_n_l1485_148549

theorem analyze_a_b_m_n (a b m n : ℕ) (ha : 1 < a) (hb : 1 < b) (hm : 1 < m) (hn : 1 < n)
  (h1 : Prime (a^n - 1))
  (h2 : Prime (b^m + 1)) :
  n = 2 ∧ ∃ k : ℕ, m = 2^k :=
by
  sorry

end NUMINAMATH_GPT_analyze_a_b_m_n_l1485_148549


namespace NUMINAMATH_GPT_sum_of_values_l1485_148502

theorem sum_of_values :
  1 + 0.01 + 0.0001 = 1.0101 :=
by sorry

end NUMINAMATH_GPT_sum_of_values_l1485_148502


namespace NUMINAMATH_GPT_lizz_team_loses_by_8_points_l1485_148590

-- Definitions of the given conditions
def initial_deficit : ℕ := 20
def free_throw_points : ℕ := 5 * 1
def three_pointer_points : ℕ := 3 * 3
def jump_shot_points : ℕ := 4 * 2
def liz_points : ℕ := free_throw_points + three_pointer_points + jump_shot_points
def other_team_points : ℕ := 10
def points_caught_up : ℕ := liz_points - other_team_points
def final_deficit : ℕ := initial_deficit - points_caught_up

-- Theorem proving Liz's team loses by 8 points
theorem lizz_team_loses_by_8_points : final_deficit = 8 :=
  by
    -- Proof will be here
    sorry

end NUMINAMATH_GPT_lizz_team_loses_by_8_points_l1485_148590


namespace NUMINAMATH_GPT_distance_from_ground_at_speed_25_is_137_5_l1485_148536
noncomputable section

-- Define the initial conditions and givens
def buildingHeight : ℝ := 200
def speedProportionalityConstant : ℝ := 10
def distanceProportionalityConstant : ℝ := 10

-- Define the speed function and distance function
def speed (t : ℝ) : ℝ := speedProportionalityConstant * t
def distance (t : ℝ) : ℝ := distanceProportionalityConstant * (t * t)

-- Define the specific time when speed is 25 m/sec
def timeWhenSpeedIs25 : ℝ := 25 / speedProportionalityConstant

-- Define the distance traveled at this specific time
def distanceTraveledAtTime : ℝ := distance timeWhenSpeedIs25

-- Calculate the distance from the ground
def distanceFromGroundAtSpeed25 : ℝ := buildingHeight - distanceTraveledAtTime

-- State the theorem
theorem distance_from_ground_at_speed_25_is_137_5 :
  distanceFromGroundAtSpeed25 = 137.5 :=
sorry

end NUMINAMATH_GPT_distance_from_ground_at_speed_25_is_137_5_l1485_148536


namespace NUMINAMATH_GPT_sequence_length_l1485_148507

theorem sequence_length :
  ∀ (a d n : ℤ), a = -6 → d = 4 → (a + (n - 1) * d = 50) → n = 15 :=
by
  intros a d n ha hd h_seq
  sorry

end NUMINAMATH_GPT_sequence_length_l1485_148507


namespace NUMINAMATH_GPT_factor_81_minus_27_x_cubed_l1485_148594

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end NUMINAMATH_GPT_factor_81_minus_27_x_cubed_l1485_148594


namespace NUMINAMATH_GPT_white_tulips_multiple_of_seven_l1485_148572

/-- Let R be the number of red tulips, which is given as 91. 
    We also know that the greatest number of identical bouquets that can be made without 
    leaving any flowers out is 7.
    Prove that the number of white tulips W is a multiple of 7. -/
theorem white_tulips_multiple_of_seven (R : ℕ) (g : ℕ) (W : ℕ) (hR : R = 91) (hg : g = 7) :
  ∃ w : ℕ, W = 7 * w :=
by
  sorry

end NUMINAMATH_GPT_white_tulips_multiple_of_seven_l1485_148572


namespace NUMINAMATH_GPT_probability_interval_l1485_148520

variable (P_A P_B q : ℚ)

axiom prob_A : P_A = 5/6
axiom prob_B : P_B = 3/4
axiom prob_A_and_B : q = P_A + P_B - 1

theorem probability_interval :
  7/12 ≤ q ∧ q ≤ 3/4 :=
by
  sorry

end NUMINAMATH_GPT_probability_interval_l1485_148520


namespace NUMINAMATH_GPT_Black_Queen_thought_Black_King_asleep_l1485_148556

theorem Black_Queen_thought_Black_King_asleep (BK_awake : Prop) (BQ_awake : Prop) :
  (∃ t : ℕ, t = 10 * 60 + 55 → 
  ∀ (BK : Prop) (BQ : Prop),
    ((BK_awake ↔ ¬BK) ∧ (BQ_awake ↔ ¬BQ)) ∧
    (BK → BQ → BQ_awake) ∧
    (¬BK → ¬BQ → BK_awake)) →
  ((BQ ↔ BK) ∧ (BQ_awake ↔ ¬BQ)) →
  (∃ (BQ_thought : Prop), BQ_thought ↔ BK) := 
sorry

end NUMINAMATH_GPT_Black_Queen_thought_Black_King_asleep_l1485_148556


namespace NUMINAMATH_GPT_validate_equation_l1485_148504

variable (x : ℝ)

def price_of_notebook : ℝ := x - 2
def price_of_pen : ℝ := x

def total_cost (x : ℝ) : ℝ := 5 * price_of_notebook x + 3 * price_of_pen x

theorem validate_equation (x : ℝ) : total_cost x = 14 :=
by
  unfold total_cost
  unfold price_of_notebook
  unfold price_of_pen
  sorry

end NUMINAMATH_GPT_validate_equation_l1485_148504


namespace NUMINAMATH_GPT_num_ways_choose_officers_8_l1485_148589

def numWaysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem num_ways_choose_officers_8 : numWaysToChooseOfficers 8 = 336 := by
  sorry

end NUMINAMATH_GPT_num_ways_choose_officers_8_l1485_148589


namespace NUMINAMATH_GPT_overall_loss_percentage_l1485_148542

theorem overall_loss_percentage
  (cost_price : ℝ)
  (discount : ℝ)
  (sales_tax : ℝ)
  (depreciation : ℝ)
  (final_selling_price : ℝ) :
  cost_price = 1900 →
  discount = 0.15 →
  sales_tax = 0.12 →
  depreciation = 0.05 →
  final_selling_price = 1330 →
  ((cost_price - (discount * cost_price)) * (1 + sales_tax) * (1 - depreciation) - final_selling_price) / cost_price * 100 = 20.44 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_overall_loss_percentage_l1485_148542


namespace NUMINAMATH_GPT_calculate_power_expr_l1485_148548

theorem calculate_power_expr :
  let a := (-8 : ℝ)
  let b := (0.125 : ℝ)
  a^2023 * b^2024 = -0.125 :=
by
  sorry

end NUMINAMATH_GPT_calculate_power_expr_l1485_148548


namespace NUMINAMATH_GPT_nesbitts_inequality_l1485_148582

variable (a b c : ℝ)

theorem nesbitts_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) >= 3 / 2 := 
sorry

end NUMINAMATH_GPT_nesbitts_inequality_l1485_148582


namespace NUMINAMATH_GPT_flight_duration_NY_to_CT_l1485_148506

theorem flight_duration_NY_to_CT :
  let departure_London_to_NY : Nat := 6 -- time in ET on Monday
  let arrival_NY_later_hours : Nat := 18 -- hours after departure
  let arrival_NY : Nat := (departure_London_to_NY + arrival_NY_later_hours) % 24 -- time in ET on Tuesday
  let arrival_CapeTown : Nat := 10 -- time in ET on Tuesday
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24 -- duration calculation
  duration_flight_NY_to_CT = 10 :=
by
  let departure_London_to_NY := 6
  let arrival_NY_later_hours := 18
  let arrival_NY := (departure_London_to_NY + arrival_NY_later_hours) % 24
  let arrival_CapeTown := 10
  let duration_flight_NY_to_CT := (arrival_CapeTown + 24 - arrival_NY) % 24
  show duration_flight_NY_to_CT = 10
  sorry

end NUMINAMATH_GPT_flight_duration_NY_to_CT_l1485_148506


namespace NUMINAMATH_GPT_probability_diff_colors_l1485_148595

/-!
There are 5 identical balls, including 3 white balls and 2 black balls. 
If 2 balls are drawn at once, the probability of the event "the 2 balls have different colors" 
occurring is \( \frac{3}{5} \).
-/

theorem probability_diff_colors 
    (white_balls : ℕ) (black_balls : ℕ) (total_balls : ℕ) (drawn_balls : ℕ) 
    (h_white : white_balls = 3) (h_black : black_balls = 2) (h_total : total_balls = 5) (h_drawn : drawn_balls = 2) :
    let total_ways := Nat.choose total_balls drawn_balls
    let diff_color_ways := (Nat.choose white_balls 1) * (Nat.choose black_balls 1)
    (diff_color_ways : ℚ) / (total_ways : ℚ) = 3 / 5 := 
by
    -- Step 1: Calculate total ways to draw 2 balls out of 5
    -- total_ways = 10 (by binomial coefficient)
    -- Step 2: Calculate favorable outcomes (1 white, 1 black)
    -- diff_color_ways = 6
    -- Step 3: Calculate probability
    -- Probability = 6 / 10 = 3 / 5
    sorry

end NUMINAMATH_GPT_probability_diff_colors_l1485_148595


namespace NUMINAMATH_GPT_square_side_length_l1485_148564

theorem square_side_length (A : ℝ) (s : ℝ) (hA : A = 64) (h_s : A = s * s) : s = 8 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l1485_148564


namespace NUMINAMATH_GPT_range_of_a_l1485_148568

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = abs (x - 2) + abs (x + a) ∧ f x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1485_148568


namespace NUMINAMATH_GPT_max_value_M_l1485_148599

def J_k (k : ℕ) : ℕ := 10^(k + 3) + 1600

def M (k : ℕ) : ℕ := (J_k k).factors.count 2

theorem max_value_M : ∃ k > 0, (M k) = 7 ∧ ∀ m > 0, M m ≤ 7 :=
by 
  sorry

end NUMINAMATH_GPT_max_value_M_l1485_148599


namespace NUMINAMATH_GPT_trig_identity_solution_l1485_148541

noncomputable def solve_trig_identity (x : ℝ) : Prop :=
  (∃ k : ℤ, x = (Real.pi / 8 * (4 * k + 1))) ∧
  (Real.sin (2 * x))^4 + (Real.cos (2 * x))^4 = Real.sin (2 * x) * Real.cos (2 * x)

theorem trig_identity_solution (x : ℝ) :
  solve_trig_identity x :=
sorry

end NUMINAMATH_GPT_trig_identity_solution_l1485_148541


namespace NUMINAMATH_GPT_catch_up_distance_l1485_148513

/-- 
  Assume that A walks at 10 km/h, starts at time 0, and B starts cycling at 20 km/h, 
  6 hours after A starts. Prove that B catches up with A 120 km from the start.
-/
theorem catch_up_distance (speed_A speed_B : ℕ) (initial_delay : ℕ) (distance : ℕ) : 
  initial_delay = 6 →
  speed_A = 10 →
  speed_B = 20 →
  distance = 120 →
  distance = speed_B * (initial_delay * speed_A / (speed_B - speed_A)) :=
by sorry

end NUMINAMATH_GPT_catch_up_distance_l1485_148513


namespace NUMINAMATH_GPT_scale_length_l1485_148586

theorem scale_length (num_parts : ℕ) (part_length : ℕ) (total_length : ℕ) 
  (h1 : num_parts = 5) (h2 : part_length = 16) : total_length = 80 :=
by
  sorry

end NUMINAMATH_GPT_scale_length_l1485_148586


namespace NUMINAMATH_GPT_mass_percentage_C_in_CO_l1485_148593

noncomputable def atomic_mass_C : ℚ := 12.01
noncomputable def atomic_mass_O : ℚ := 16.00
noncomputable def molecular_mass_CO : ℚ := atomic_mass_C + atomic_mass_O

theorem mass_percentage_C_in_CO : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 :=
by
  have atomic_mass_C_div_total : atomic_mass_C / molecular_mass_CO = 12.01 / 28.01 := sorry
  have mass_percentage : (atomic_mass_C / molecular_mass_CO) * 100 = 42.88 := sorry
  exact mass_percentage

end NUMINAMATH_GPT_mass_percentage_C_in_CO_l1485_148593


namespace NUMINAMATH_GPT_max_y_value_l1485_148581

noncomputable def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_y_value : ∃ α, (∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α) ∧ α = 3 := by
  sorry

end NUMINAMATH_GPT_max_y_value_l1485_148581


namespace NUMINAMATH_GPT_specific_value_is_165_l1485_148514

-- Declare x as a specific number and its value
def x : ℕ := 11

-- Declare the specific value as 15 times x
def specific_value : ℕ := 15 * x

-- The theorem to prove
theorem specific_value_is_165 : specific_value = 165 := by
  sorry

end NUMINAMATH_GPT_specific_value_is_165_l1485_148514


namespace NUMINAMATH_GPT_circle_problems_satisfy_conditions_l1485_148523

noncomputable def circle1_center_x := 11
noncomputable def circle1_center_y := 8
noncomputable def circle1_radius_squared := 87

noncomputable def circle2_center_x := 14
noncomputable def circle2_center_y := -3
noncomputable def circle2_radius_squared := 168

theorem circle_problems_satisfy_conditions :
  (∀ x y, (x-11)^2 + (y-8)^2 = 87 ∨ (x-14)^2 + (y+3)^2 = 168) := sorry

end NUMINAMATH_GPT_circle_problems_satisfy_conditions_l1485_148523


namespace NUMINAMATH_GPT_flat_fee_first_night_l1485_148525

theorem flat_fee_first_night :
  ∃ f n : ℚ, (f + 3 * n = 195) ∧ (f + 6 * n = 350) ∧ (f = 40) :=
by
  -- Skipping the detailed proof:
  sorry

end NUMINAMATH_GPT_flat_fee_first_night_l1485_148525


namespace NUMINAMATH_GPT_find_x_l1485_148516

-- Define the vectors and collinearity condition
def vector_a : ℝ × ℝ := (3, 6)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 8)

def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (b.1 = k * a.1) ∧ (b.2 = k * a.2)

-- Define the proof problem
theorem find_x (x : ℝ) (h : collinear vector_a (vector_b x)) : x = 4 :=
  sorry

end NUMINAMATH_GPT_find_x_l1485_148516


namespace NUMINAMATH_GPT_battery_charging_budget_l1485_148501

def cost_per_charge : ℝ := 3.5
def charges : ℕ := 4
def leftover : ℝ := 6
def budget : ℝ := 20

theorem battery_charging_budget :
  (charges : ℝ) * cost_per_charge + leftover = budget :=
by
  sorry

end NUMINAMATH_GPT_battery_charging_budget_l1485_148501


namespace NUMINAMATH_GPT_measure_of_angle_D_l1485_148550

theorem measure_of_angle_D 
  (A B C D E F : ℝ)
  (h1 : A = B) (h2 : B = C) (h3 : C = F)
  (h4 : D = E) (h5 : A = D - 30) 
  (sum_angles : A + B + C + D + E + F = 720) : 
  D = 140 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_D_l1485_148550


namespace NUMINAMATH_GPT_rectangle_area_error_percent_l1485_148576

theorem rectangle_area_error_percent 
  (L W : ℝ)
  (hL: L > 0)
  (hW: W > 0) :
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  error_percent = 0.7 := by
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  sorry

end NUMINAMATH_GPT_rectangle_area_error_percent_l1485_148576


namespace NUMINAMATH_GPT_smallest_b_for_perfect_square_l1485_148558

theorem smallest_b_for_perfect_square : ∃ (b : ℕ), b > 4 ∧ (∃ k, (2 * b + 4) = k * k) ∧
                                             ∀ (b' : ℕ), b' > 4 ∧ (∃ k, (2 * b' + 4) = k * k) → b ≤ b' :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_for_perfect_square_l1485_148558


namespace NUMINAMATH_GPT_find_value_of_expression_l1485_148524

theorem find_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 + 4 ≤ ab + 3 * b + 2 * c) :
  200 * a + 9 * b + c = 219 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l1485_148524


namespace NUMINAMATH_GPT_negative_values_of_x_l1485_148566

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end NUMINAMATH_GPT_negative_values_of_x_l1485_148566


namespace NUMINAMATH_GPT_tangents_to_discriminant_parabola_l1485_148510

variable (a : ℝ) (p q : ℝ)

theorem tangents_to_discriminant_parabola :
  (a^2 + a * p + q = 0) ↔ (p^2 - 4 * q = 0) :=
sorry

end NUMINAMATH_GPT_tangents_to_discriminant_parabola_l1485_148510


namespace NUMINAMATH_GPT_blending_marker_drawings_correct_l1485_148597

-- Define the conditions
def total_drawings : ℕ := 25
def colored_pencil_drawings : ℕ := 14
def charcoal_drawings : ℕ := 4

-- Define the target proof statement
def blending_marker_drawings : ℕ := total_drawings - (colored_pencil_drawings + charcoal_drawings)

-- Proof goal
theorem blending_marker_drawings_correct : blending_marker_drawings = 7 := by
  sorry

end NUMINAMATH_GPT_blending_marker_drawings_correct_l1485_148597


namespace NUMINAMATH_GPT_minimum_value_2_only_in_option_b_l1485_148539

noncomputable def option_a (x : ℝ) : ℝ := x + 1 / x
noncomputable def option_b (x : ℝ) : ℝ := 3^x + 3^(-x)
noncomputable def option_c (x : ℝ) : ℝ := (Real.log x) + 1 / (Real.log x)
noncomputable def option_d (x : ℝ) : ℝ := (Real.sin x) + 1 / (Real.sin x)

theorem minimum_value_2_only_in_option_b :
  (∀ x > 0, option_a x ≠ 2) ∧
  (∃ x, option_b x = 2) ∧
  (∀ x (h: 0 < x) (h' : x < 1), option_c x ≠ 2) ∧
  (∀ x (h: 0 < x) (h' : x < π / 2), option_d x ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_2_only_in_option_b_l1485_148539


namespace NUMINAMATH_GPT_hall_length_width_difference_l1485_148561

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L)
  (h2 : L * W = 450) :
  L - W = 15 :=
sorry

end NUMINAMATH_GPT_hall_length_width_difference_l1485_148561


namespace NUMINAMATH_GPT_sin_double_alpha_trig_expression_l1485_148546

theorem sin_double_alpha (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin (2 * α) = 4 * Real.sqrt 2 / 9 :=
sorry

theorem trig_expression (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin (α - 2 * π) * Real.cos (2 * π - α)) / (Real.sin (α + π / 2) ^ 2) = Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_sin_double_alpha_trig_expression_l1485_148546


namespace NUMINAMATH_GPT_fraction_division_l1485_148567

theorem fraction_division :
  (3 / 7) / (2 / 5) = (15 / 14) :=
by
  sorry

end NUMINAMATH_GPT_fraction_division_l1485_148567


namespace NUMINAMATH_GPT_tetrahedron_volume_l1485_148526

theorem tetrahedron_volume (S R V : ℝ) (h : V = (1/3) * S * R) : 
  V = (1/3) * S * R := 
by 
  sorry

end NUMINAMATH_GPT_tetrahedron_volume_l1485_148526


namespace NUMINAMATH_GPT_unique_zero_location_l1485_148598

theorem unique_zero_location (f : ℝ → ℝ) (h : ∃! x, f x = 0 ∧ 1 < x ∧ x < 3) :
  ¬ (∃ x, 2 < x ∧ x < 5 ∧ f x = 0) :=
sorry

end NUMINAMATH_GPT_unique_zero_location_l1485_148598


namespace NUMINAMATH_GPT_concave_number_count_l1485_148574

def is_concave_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  n >= 100 ∧ n < 1000 ∧ tens < hundreds ∧ tens < units

theorem concave_number_count : ∃ n : ℕ, 
  (∀ m < 1000, is_concave_number m → m = n) ∧ n = 240 :=
by
  sorry

end NUMINAMATH_GPT_concave_number_count_l1485_148574


namespace NUMINAMATH_GPT_part_one_solution_part_two_solution_l1485_148553

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part (1): When a = 1, solution set of the inequality f(x) > 1 is (1/2, +∞)
theorem part_one_solution (x : ℝ) :
  f x 1 > 1 ↔ x > 1 / 2 := sorry

-- Part (2): If the inequality f(x) > x holds for x ∈ (0,1), range of values for a is (0, 2]
theorem part_two_solution (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → f x a > x) ↔ 0 < a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_part_one_solution_part_two_solution_l1485_148553


namespace NUMINAMATH_GPT_largest_n_divisibility_condition_l1485_148522

def S1 (n : ℕ) : ℕ := (n * (n + 1)) / 2
def S2 (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem largest_n_divisibility_condition : ∀ (n : ℕ), (n = 1) → (S2 n) % (S1 n) = 0 :=
by
  intros n hn
  rw [hn]
  sorry

end NUMINAMATH_GPT_largest_n_divisibility_condition_l1485_148522


namespace NUMINAMATH_GPT_no_same_last_four_digits_pow_l1485_148521

theorem no_same_last_four_digits_pow (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (5^n % 10000) ≠ (6^m % 10000) :=
by sorry

end NUMINAMATH_GPT_no_same_last_four_digits_pow_l1485_148521


namespace NUMINAMATH_GPT_total_flour_needed_l1485_148551

theorem total_flour_needed (Katie_flour : ℕ) (Sheila_flour : ℕ) 
  (h1 : Katie_flour = 3) 
  (h2 : Sheila_flour = Katie_flour + 2) : 
  Katie_flour + Sheila_flour = 8 := 
  by 
  sorry

end NUMINAMATH_GPT_total_flour_needed_l1485_148551


namespace NUMINAMATH_GPT_find_original_price_l1485_148557

theorem find_original_price (sale_price : ℕ) (discount : ℕ) (original_price : ℕ) 
  (h1 : sale_price = 60) 
  (h2 : discount = 40) 
  (h3 : original_price = sale_price / ((100 - discount) / 100)) : original_price = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_original_price_l1485_148557


namespace NUMINAMATH_GPT_chord_constant_l1485_148573

theorem chord_constant (
    d : ℝ
) : (∃ t : ℝ, (∀ A B : ℝ × ℝ,
    A.2 = A.1^3 ∧ B.2 = B.1^3 ∧ d = 1/2 ∧
    (C : ℝ × ℝ) = (0, d) ∧ 
    (∀ (AC BC: ℝ),
        AC = dist A C ∧
        BC = dist B C ∧
        t = (1 / (AC^2) + 1 / (BC^2))
    )) → t = 4) := 
sorry

end NUMINAMATH_GPT_chord_constant_l1485_148573


namespace NUMINAMATH_GPT_compute_value_l1485_148552

theorem compute_value : (142 + 29 + 26 + 14) * 2 = 422 := 
by 
  sorry

end NUMINAMATH_GPT_compute_value_l1485_148552


namespace NUMINAMATH_GPT_geom_progression_sum_ratio_l1485_148545

theorem geom_progression_sum_ratio (a : ℝ) (r : ℝ) (m : ℕ) :
  r = 5 →
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^m) / (1 - r)) = 126 →
  m = 3 := by
  sorry

end NUMINAMATH_GPT_geom_progression_sum_ratio_l1485_148545


namespace NUMINAMATH_GPT_inequalities_hold_l1485_148532

variables {a b c : ℝ}

theorem inequalities_hold (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : 
  (b / a > c / a) ∧ ((b - a) / c > 0) ∧ ((a - c) / (a * c) < 0) := 
  by
    sorry

end NUMINAMATH_GPT_inequalities_hold_l1485_148532


namespace NUMINAMATH_GPT_geometric_sequence_value_a6_l1485_148518

theorem geometric_sequence_value_a6
    (q a1 : ℝ) (a : ℕ → ℝ)
    (h1 : ∀ n, a n = a1 * q ^ (n - 1))
    (h2 : a 2 = 1)
    (h3 : a 8 = a 6 + 2 * a 4)
    (h4 : q > 0)
    (h5 : ∀ n, a n > 0) : 
    a 6 = 4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_a6_l1485_148518


namespace NUMINAMATH_GPT_p_over_q_at_neg1_l1485_148540

-- Definitions of p(x) and q(x) based on given conditions
noncomputable def q (x : ℝ) := (x + 3) * (x - 2)
noncomputable def p (x : ℝ) := 2 * x

-- Define the main function y = p(x) / q(x)
noncomputable def y (x : ℝ) := p x / q x

-- Statement to prove the value of p(-1) / q(-1)
theorem p_over_q_at_neg1 : y (-1) = (1 : ℝ) / 3 :=
by
  sorry

end NUMINAMATH_GPT_p_over_q_at_neg1_l1485_148540


namespace NUMINAMATH_GPT_junk_mail_per_block_l1485_148515

theorem junk_mail_per_block (houses_per_block : ℕ) (mail_per_house : ℕ) (total_mail : ℕ) :
  houses_per_block = 20 → mail_per_house = 32 → total_mail = 640 := by
  intros hpb_price mph_correct
  sorry

end NUMINAMATH_GPT_junk_mail_per_block_l1485_148515


namespace NUMINAMATH_GPT_number_of_ways_to_assign_roles_l1485_148533

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 5
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let total_men := men - male_roles
  let total_women := women - female_roles
  (men.choose male_roles) * (women.choose female_roles) * (total_men + total_women).choose either_gender_roles = 14400 := by 
sorry

end NUMINAMATH_GPT_number_of_ways_to_assign_roles_l1485_148533


namespace NUMINAMATH_GPT_arccos_sin_3_l1485_148529

theorem arccos_sin_3 : Real.arccos (Real.sin 3) = (Real.pi / 2) + 3 := 
by
  sorry

end NUMINAMATH_GPT_arccos_sin_3_l1485_148529


namespace NUMINAMATH_GPT_major_axis_length_l1485_148571

-- Definitions of the given conditions
structure Ellipse :=
  (focus1 focus2 : ℝ × ℝ)
  (tangent_to_x_axis : Bool)

noncomputable def length_of_major_axis (E : Ellipse) : ℝ :=
  let (x1, y1) := E.focus1
  let (x2, y2) := E.focus2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 + y1) ^ 2)

-- The theorem we want to prove given the conditions
theorem major_axis_length (E : Ellipse)
  (h1 : E.focus1 = (9, 20))
  (h2 : E.focus2 = (49, 55))
  (h3 : E.tangent_to_x_axis = true):
  length_of_major_axis E = 85 :=
by
  sorry

end NUMINAMATH_GPT_major_axis_length_l1485_148571


namespace NUMINAMATH_GPT_second_caterer_cheaper_l1485_148534

theorem second_caterer_cheaper (x : ℕ) (h : x > 33) : 200 + 12 * x < 100 + 15 * x := 
by
  sorry

end NUMINAMATH_GPT_second_caterer_cheaper_l1485_148534


namespace NUMINAMATH_GPT_total_people_veg_l1485_148538

def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 8

theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 21 := by
  sorry

end NUMINAMATH_GPT_total_people_veg_l1485_148538


namespace NUMINAMATH_GPT_deepak_age_l1485_148587

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 2 / 5)  -- the ratio condition
  (h2 : A + 10 = 30)   -- Arun’s age after 10 years will be 30
  : D = 50 :=       -- conclusion Deepak is 50 years old
sorry

end NUMINAMATH_GPT_deepak_age_l1485_148587


namespace NUMINAMATH_GPT_cubic_roots_identity_l1485_148579

theorem cubic_roots_identity (x1 x2 p q : ℝ) 
  (h1 : x1^2 + p * x1 + q = 0) 
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1^3 + x2^3 = 3 * p * q - p^3) ∧ 
  (x1^3 - x2^3 = (p^2 - q) * Real.sqrt (p^2 - 4 * q) ∨ 
   x1^3 - x2^3 = -(p^2 - q) * Real.sqrt (p^2 - 4 * q)) :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_identity_l1485_148579


namespace NUMINAMATH_GPT_least_positive_integer_l1485_148543

theorem least_positive_integer (x : ℕ) (h : x + 5600 ≡ 325 [MOD 15]) : x = 5 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_l1485_148543


namespace NUMINAMATH_GPT_complete_square_sum_l1485_148531

theorem complete_square_sum (a h k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) →
  a + h + k = -5 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_complete_square_sum_l1485_148531


namespace NUMINAMATH_GPT_value_of_a_l1485_148584

theorem value_of_a (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) → m > 1 :=
sorry

end NUMINAMATH_GPT_value_of_a_l1485_148584


namespace NUMINAMATH_GPT_gcd_150_m_l1485_148519

theorem gcd_150_m (m : ℕ)
  (h : ∃ d : ℕ, d ∣ 150 ∧ d ∣ m ∧ (∀ x, x ∣ 150 → x ∣ m → x = 1 ∨ x = 5 ∨ x = 25)) :
  gcd 150 m = 25 :=
sorry

end NUMINAMATH_GPT_gcd_150_m_l1485_148519


namespace NUMINAMATH_GPT_triangle_AB_eq_3_halves_CK_l1485_148535

/-- Mathematically equivalent problem:
In an acute triangle ABC, rectangle ACGH is constructed with AC as one side, and CG : AC = 2:1.
A square BCEF is constructed with BC as one side. The height CD from A to B intersects GE at point K.
Prove that AB = 3/2 * CK. -/
theorem triangle_AB_eq_3_halves_CK
  (A B C H G E K : Type)
  (triangle_ABC_acute : ∀(A B C : Type), True) 
  (rectangle_ACGH : ∀(A C G H : Type), True) 
  (square_BCEF : ∀(B C E F : Type), True)
  (H_C_G_collinear : ∀(H C G : Type), True)
  (HCG_ratio : ∀ (AC CG : ℝ), CG / AC = 2 / 1)
  (BC_side : ∀ (BC : ℝ), BC = 1)
  (height_CD_intersection : ∀ (A B C D E G : Type), True)
  (intersection_point_K : ∀ (C D G E K : Type), True) :
  ∃ (AB CK : ℝ), AB = 3 / 2 * CK :=
by sorry

end NUMINAMATH_GPT_triangle_AB_eq_3_halves_CK_l1485_148535


namespace NUMINAMATH_GPT_CindyHomework_l1485_148512

theorem CindyHomework (x : ℤ) (h : (x - 7) * 4 = 48) : (4 * x - 7) = 69 := by
  sorry

end NUMINAMATH_GPT_CindyHomework_l1485_148512


namespace NUMINAMATH_GPT_non_allergic_children_l1485_148577

theorem non_allergic_children (T : ℕ) (h1 : T / 2 = n) (h2 : ∀ m : ℕ, 10 = m) (h3 : ∀ k : ℕ, 10 = k) :
  10 = 10 :=
by
  sorry

end NUMINAMATH_GPT_non_allergic_children_l1485_148577


namespace NUMINAMATH_GPT_stuffed_animals_total_l1485_148547

variable (x y z : ℕ)

theorem stuffed_animals_total :
  let initial := x
  let after_mom := initial + y
  let after_dad := z * after_mom
  let total := after_mom + after_dad
  total = (x + y) * (1 + z) := 
  by 
    let initial := x
    let after_mom := initial + y
    let after_dad := z * after_mom
    let total := after_mom + after_dad
    sorry

end NUMINAMATH_GPT_stuffed_animals_total_l1485_148547


namespace NUMINAMATH_GPT_calculate_expression_l1485_148517

theorem calculate_expression :
  8^8 + 8^8 + 8^8 + 8^8 + 8^5 = 4 * 8^8 + 8^5 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l1485_148517


namespace NUMINAMATH_GPT_blocks_used_for_fenced_area_l1485_148578

theorem blocks_used_for_fenced_area
  (initial_blocks : ℕ) (building_blocks : ℕ) (farmhouse_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 344 →
  building_blocks = 80 →
  farmhouse_blocks = 123 →
  remaining_blocks = 84 →
  initial_blocks - building_blocks - farmhouse_blocks - remaining_blocks = 57 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_blocks_used_for_fenced_area_l1485_148578


namespace NUMINAMATH_GPT_smallest_n_power_2013_ends_001_l1485_148503

theorem smallest_n_power_2013_ends_001 :
  ∃ n : ℕ, n > 0 ∧ 2013^n % 1000 = 1 ∧ ∀ m : ℕ, m > 0 ∧ 2013^m % 1000 = 1 → n ≤ m := 
sorry

end NUMINAMATH_GPT_smallest_n_power_2013_ends_001_l1485_148503


namespace NUMINAMATH_GPT_problem_I_problem_II_l1485_148588

-- Problem (I)
theorem problem_I (a : ℝ) (h : ∀ x : ℝ, x^2 - 3 * a * x + 9 > 0) : -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Problem (II)
theorem problem_II (m : ℝ) 
  (h₁ : ∀ x : ℝ, x^2 + 2 * x - 8 < 0 → x - m > 0)
  (h₂ : ∃ x : ℝ, x^2 + 2 * x - 8 < 0) : m ≤ -4 :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l1485_148588


namespace NUMINAMATH_GPT_right_triangle_area_valid_right_triangle_perimeter_valid_l1485_148580

-- Define the basic setup for the right triangle problem
def hypotenuse : ℕ := 13
def leg1 : ℕ := 5
def leg2 : ℕ := 12  -- Calculated from Pythagorean theorem, but assumed here as condition

-- Define the calculated area and perimeter based on the above definitions
def area (a b : ℕ) : ℕ := (1 / 2) * a * b
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- State the proof goals
theorem right_triangle_area_valid : area leg1 leg2 = 30 :=
  by sorry

theorem right_triangle_perimeter_valid : perimeter leg1 leg2 hypotenuse = 30 :=
  by sorry

end NUMINAMATH_GPT_right_triangle_area_valid_right_triangle_perimeter_valid_l1485_148580


namespace NUMINAMATH_GPT_maximize_profit_l1485_148585

-- Definitions from the conditions
def cost_price : ℝ := 16
def initial_selling_price : ℝ := 20
def initial_sales_volume : ℝ := 80
def price_decrease_per_step : ℝ := 0.5
def sales_increase_per_step : ℝ := 20

def functional_relationship (x : ℝ) : ℝ := -40 * x + 880

-- The main theorem we need to prove
theorem maximize_profit :
  (∀ x, 16 ≤ x → x ≤ 20 → functional_relationship x = -40 * x + 880) ∧
  (∃ x, 16 ≤ x ∧ x ≤ 20 ∧ (∀ y, 16 ≤ y → y ≤ 20 → 
    ((-40 * x + 880) * (x - cost_price) ≥ (-40 * y + 880) * (y - cost_price)) ∧
    (-40 * x + 880) * (x - cost_price) = 360 ∧ x = 19)) :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l1485_148585


namespace NUMINAMATH_GPT_find_m_l1485_148570

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_m (m : ℝ) :
  is_parallel (vector_a.1 + 2 * m, vector_a.2 + 2 * 1) (2 * vector_a.1 - m, 2 * vector_a.2 - 1) ↔ m = -1 / 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_m_l1485_148570


namespace NUMINAMATH_GPT_kostya_initially_planted_l1485_148562

def bulbs_after_planting (n : ℕ) (stages : ℕ) : ℕ :=
  match stages with
  | 0 => n
  | k + 1 => 2 * bulbs_after_planting n k - 1

theorem kostya_initially_planted (n : ℕ) (stages : ℕ) :
  bulbs_after_planting n stages = 113 → n = 15 := 
sorry

end NUMINAMATH_GPT_kostya_initially_planted_l1485_148562


namespace NUMINAMATH_GPT_maria_punch_l1485_148505

variable (L S W : ℕ)

theorem maria_punch (h1 : S = 3 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 36 :=
by
  sorry

end NUMINAMATH_GPT_maria_punch_l1485_148505


namespace NUMINAMATH_GPT_max_true_statements_l1485_148592

theorem max_true_statements :
  ∃ x : ℝ, 
  (0 < x ∧ x < 1) ∧ -- Statement 4
  (0 < x^3 ∧ x^3 < 1) ∧ -- Statement 1
  (0 < x - x^3 ∧ x - x^3 < 1) ∧ -- Statement 5
  ¬(x^3 > 1) ∧ -- Not Statement 2
  ¬(-1 < x ∧ x < 0) := -- Not Statement 3
sorry

end NUMINAMATH_GPT_max_true_statements_l1485_148592


namespace NUMINAMATH_GPT_find_x0_l1485_148591

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 2 then x^2 - 4 else 2 * x

theorem find_x0 (x0 : ℝ) (h : f x0 = 8) : x0 = 4 := by
  sorry

end NUMINAMATH_GPT_find_x0_l1485_148591


namespace NUMINAMATH_GPT_rajesh_monthly_savings_l1485_148563

theorem rajesh_monthly_savings
  (salary : ℝ)
  (percentage_food : ℝ)
  (percentage_medicines : ℝ)
  (percentage_savings : ℝ)
  (amount_food : ℝ := percentage_food * salary)
  (amount_medicines : ℝ := percentage_medicines * salary)
  (remaining_amount : ℝ := salary - (amount_food + amount_medicines))
  (save_amount : ℝ := percentage_savings * remaining_amount)
  (H_salary : salary = 15000)
  (H_percentage_food : percentage_food = 0.40)
  (H_percentage_medicines : percentage_medicines = 0.20)
  (H_percentage_savings : percentage_savings = 0.60) :
  save_amount = 3600 :=
by
  sorry

end NUMINAMATH_GPT_rajesh_monthly_savings_l1485_148563


namespace NUMINAMATH_GPT_ages_of_Mel_and_Lexi_l1485_148569

theorem ages_of_Mel_and_Lexi (M L K : ℤ)
  (h1 : M = K - 3)
  (h2 : L = M + 2)
  (h3 : K = 60) :
  M = 57 ∧ L = 59 :=
  by
    -- Proof steps are omitted.
    sorry

end NUMINAMATH_GPT_ages_of_Mel_and_Lexi_l1485_148569


namespace NUMINAMATH_GPT_abs_inequality_l1485_148508

theorem abs_inequality (x : ℝ) : 
  abs ((3 * x - 2) / (x - 2)) > 3 ↔ 
  (x > 4 / 3 ∧ x < 2) ∨ (x > 2) := 
sorry

end NUMINAMATH_GPT_abs_inequality_l1485_148508


namespace NUMINAMATH_GPT_jellybean_ratio_l1485_148554

theorem jellybean_ratio (L Tino Arnold : ℕ) (h1 : Tino = L + 24) (h2 : Arnold = 5) (h3 : Tino = 34) :
  Arnold / L = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_jellybean_ratio_l1485_148554


namespace NUMINAMATH_GPT_total_feet_in_garden_l1485_148530

def dogs : ℕ := 6
def ducks : ℕ := 2
def cats : ℕ := 4
def birds : ℕ := 7
def insects : ℕ := 10

def feet_per_dog : ℕ := 4
def feet_per_duck : ℕ := 2
def feet_per_cat : ℕ := 4
def feet_per_bird : ℕ := 2
def feet_per_insect : ℕ := 6

theorem total_feet_in_garden :
  dogs * feet_per_dog + 
  ducks * feet_per_duck + 
  cats * feet_per_cat + 
  birds * feet_per_bird + 
  insects * feet_per_insect = 118 := by
  sorry

end NUMINAMATH_GPT_total_feet_in_garden_l1485_148530


namespace NUMINAMATH_GPT_pages_wednesday_l1485_148583

-- Given conditions as definitions
def borrow_books := 3
def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51

-- Prove that Nico read 19 pages on Wednesday
theorem pages_wednesday :
  let pages_wednesday := total_pages - (pages_monday + pages_tuesday)
  pages_wednesday = 19 :=
by
  sorry

end NUMINAMATH_GPT_pages_wednesday_l1485_148583


namespace NUMINAMATH_GPT_tim_books_l1485_148544

def has_some_books (Tim Sam : ℕ) : Prop :=
  Sam = 52 ∧ Tim + Sam = 96

theorem tim_books (Tim : ℕ) :
  has_some_books Tim 52 → Tim = 44 := 
by
  intro h
  obtain ⟨hSam, hTogether⟩ := h
  sorry

end NUMINAMATH_GPT_tim_books_l1485_148544


namespace NUMINAMATH_GPT_deepak_present_age_l1485_148555

/-- Let Rahul and Deepak's current ages be 4x and 3x respectively
  Given that:
  1. The ratio between Rahul and Deepak's ages is 4:3
  2. After 6 years, Rahul's age will be 26 years
  Prove that Deepak's present age is 15 years.
-/
theorem deepak_present_age (x : ℕ) (hx : 4 * x + 6 = 26) : 3 * x = 15 :=
by
  sorry

end NUMINAMATH_GPT_deepak_present_age_l1485_148555


namespace NUMINAMATH_GPT_disjoint_subsets_same_sum_l1485_148528

-- Define the main theorem
theorem disjoint_subsets_same_sum (S : Finset ℕ) (hS_len : S.card = 10) (hS_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id :=
by {
  sorry
}

end NUMINAMATH_GPT_disjoint_subsets_same_sum_l1485_148528


namespace NUMINAMATH_GPT_problem_statement_l1485_148560

theorem problem_statement : ∀ (x y : ℝ), |x - 2| + (y + 3)^2 = 0 → (x + y)^2023 = -1 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_problem_statement_l1485_148560


namespace NUMINAMATH_GPT_sqrt_7_estimate_l1485_148559

theorem sqrt_7_estimate : (2 : Real) < Real.sqrt 7 ∧ Real.sqrt 7 < 3 → (Real.sqrt 7 - 1) / 2 < 1 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_sqrt_7_estimate_l1485_148559


namespace NUMINAMATH_GPT_circle_covers_three_points_l1485_148500

open Real

theorem circle_covers_three_points 
  (points : Finset (ℝ × ℝ))
  (h_points : points.card = 111)
  (triangle_side : ℝ)
  (h_side : triangle_side = 15) :
  ∃ (circle_center : ℝ × ℝ), ∃ (circle_radius : ℝ), circle_radius = sqrt 3 / 2 ∧ 
  ∃ p1 p2 p3, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
              dist circle_center p1 ≤ circle_radius ∧ 
              dist circle_center p2 ≤ circle_radius ∧ 
              dist circle_center p3 ≤ circle_radius :=
by
  sorry

end NUMINAMATH_GPT_circle_covers_three_points_l1485_148500


namespace NUMINAMATH_GPT_modulo_7_example_l1485_148527

def sum := 2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999

theorem modulo_7_example : (sum % 7) = 5 :=
by
  sorry

end NUMINAMATH_GPT_modulo_7_example_l1485_148527


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_17_l1485_148509

theorem smallest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n ∧ n < 10000 ∧ n % 17 = 0) ∧ ∀ m, (1000 ≤ m ∧ m < 10000 ∧ m % 17 = 0 → n ≤ m) ∧ n = 1013 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_17_l1485_148509


namespace NUMINAMATH_GPT_inv_g_of_43_div_16_l1485_148565

noncomputable def g (x : ℚ) : ℚ := (x^3 - 5) / 4

theorem inv_g_of_43_div_16 : g (3 * (↑7)^(1/3) / 2) = 43 / 16 :=
by 
  sorry

end NUMINAMATH_GPT_inv_g_of_43_div_16_l1485_148565


namespace NUMINAMATH_GPT_student_solved_correctly_l1485_148575

-- Problem conditions as definitions
def sums_attempted : Nat := 96

def sums_correct (x : Nat) : Prop :=
  let sums_wrong := 3 * x
  x + sums_wrong = sums_attempted

-- Lean statement to prove
theorem student_solved_correctly (x : Nat) (h : sums_correct x) : x = 24 :=
  sorry

end NUMINAMATH_GPT_student_solved_correctly_l1485_148575


namespace NUMINAMATH_GPT_range_of_m_l1485_148511

open Real

noncomputable def f (x : ℝ) : ℝ := 1 + sin (2 * x)
noncomputable def g (x m : ℝ) : ℝ := 2 * (cos x)^2 + m

theorem range_of_m (x₀ : ℝ) (m : ℝ) (h₀ : 0 ≤ x₀ ∧ x₀ ≤ π / 2) (h₁ : f x₀ ≥ g x₀ m) : m ≤ sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1485_148511


namespace NUMINAMATH_GPT_seq_max_min_terms_l1485_148537

noncomputable def a (n: ℕ) : ℝ := 1 / (2^n - 18)

theorem seq_max_min_terms : (∀ (n : ℕ), n > 5 → a 5 > a n) ∧ (∀ (n : ℕ), n ≠ 4 → a 4 < a n) :=
by 
  sorry

end NUMINAMATH_GPT_seq_max_min_terms_l1485_148537


namespace NUMINAMATH_GPT_history_book_pages_l1485_148596

-- Conditions
def science_pages : ℕ := 600
def novel_pages (science: ℕ) : ℕ := science / 4
def history_pages (novel: ℕ) : ℕ := novel * 2

-- Theorem to prove
theorem history_book_pages : history_pages (novel_pages science_pages) = 300 :=
by
  sorry

end NUMINAMATH_GPT_history_book_pages_l1485_148596
