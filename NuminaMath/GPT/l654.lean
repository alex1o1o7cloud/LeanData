import Mathlib

namespace NUMINAMATH_GPT_minutes_watched_on_Thursday_l654_65451

theorem minutes_watched_on_Thursday 
  (n_total : ℕ) (n_Mon : ℕ) (n_Tue : ℕ) (n_Wed : ℕ) (n_Fri : ℕ) (n_weekend : ℕ)
  (h_total : n_total = 352)
  (h_Mon : n_Mon = 138)
  (h_Tue : n_Tue = 0)
  (h_Wed : n_Wed = 0)
  (h_Fri : n_Fri = 88)
  (h_weekend : n_weekend = 105) :
  n_total - (n_Mon + n_Tue + n_Wed + n_Fri + n_weekend) = 21 := by
  sorry

end NUMINAMATH_GPT_minutes_watched_on_Thursday_l654_65451


namespace NUMINAMATH_GPT_perpendicular_iff_zero_dot_product_l654_65404

open Real

def a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem perpendicular_iff_zero_dot_product (m : ℝ) :
  dot_product (a m) (b m) = 0 → m = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_iff_zero_dot_product_l654_65404


namespace NUMINAMATH_GPT_smallest_is_C_l654_65462

def A : ℚ := 1/2
def B : ℚ := 9/10
def C : ℚ := 2/5

theorem smallest_is_C : min (min A B) C = C := 
by
  sorry

end NUMINAMATH_GPT_smallest_is_C_l654_65462


namespace NUMINAMATH_GPT_factorial_square_ge_power_l654_65498

theorem factorial_square_ge_power (n : ℕ) (h : n ≥ 1) : (n!)^2 ≥ n^n := 
by sorry

end NUMINAMATH_GPT_factorial_square_ge_power_l654_65498


namespace NUMINAMATH_GPT_segment_association_l654_65409

theorem segment_association (x y : ℝ) 
  (h1 : ∃ (D : ℝ), ∀ (P : ℝ), abs (P - D) ≤ 5) 
  (h2 : ∃ (D' : ℝ), ∀ (P' : ℝ), abs (P' - D') ≤ 9)
  (h3 : 3 * x - 2 * y = 6) : 
  x + y = 12 := 
by sorry

end NUMINAMATH_GPT_segment_association_l654_65409


namespace NUMINAMATH_GPT_range_of_m_l654_65469

theorem range_of_m (x m : ℝ) (h1 : (x ≥ 0) ∧ (x ≠ 1) ∧ (x = (6 - m) / 4)) :
    m ≤ 6 ∧ m ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l654_65469


namespace NUMINAMATH_GPT_find_prime_n_l654_65437

def is_prime (p : ℕ) : Prop := 
  p > 1 ∧ (∀ n, n ∣ p → n = 1 ∨ n = p)

def prime_candidates : List ℕ := [11, 17, 23, 29, 41, 47, 53, 59, 61, 71, 83, 89]

theorem find_prime_n (n : ℕ) 
  (h1 : n ∈ prime_candidates) 
  (h2 : is_prime (n)) 
  (h3 : is_prime (n + 20180500)) : 
  n = 61 :=
by sorry

end NUMINAMATH_GPT_find_prime_n_l654_65437


namespace NUMINAMATH_GPT_derivative_at_x_equals_1_l654_65466

variable (x : ℝ)
def y : ℝ := (x + 1) * (x - 1)

theorem derivative_at_x_equals_1 : deriv y 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_x_equals_1_l654_65466


namespace NUMINAMATH_GPT_width_rectangular_box_5_cm_l654_65430

theorem width_rectangular_box_5_cm 
  (W : ℕ)
  (h_dim_wooden_box : (8 * 10 * 6 * 100 ^ 3) = 480000000) -- dimensions of the wooden box in cm³
  (h_dim_rectangular_box : (4 * W * 6) = (24 * W)) -- dimensions of the rectangular box in cm³
  (h_max_boxes : 4000000 * (24 * W) = 480000000) -- max number of boxes that fit in the wooden box
: 
  W = 5 := 
by
  sorry

end NUMINAMATH_GPT_width_rectangular_box_5_cm_l654_65430


namespace NUMINAMATH_GPT_broken_shells_count_l654_65450

-- Definitions from conditions
def total_perfect_shells := 17
def non_spiral_perfect_shells := 12
def extra_broken_spiral_shells := 21

-- Derived definitions
def perfect_spiral_shells : ℕ := total_perfect_shells - non_spiral_perfect_shells
def broken_spiral_shells : ℕ := perfect_spiral_shells + extra_broken_spiral_shells
def broken_shells : ℕ := 2 * broken_spiral_shells

-- The theorem to be proved
theorem broken_shells_count : broken_shells = 52 := by
  sorry

end NUMINAMATH_GPT_broken_shells_count_l654_65450


namespace NUMINAMATH_GPT_initial_apples_l654_65406

-- Definitions of the conditions
def Minseok_ate : Nat := 3
def Jaeyoon_ate : Nat := 3
def apples_left : Nat := 2

-- The proposition we need to prove
theorem initial_apples : Minseok_ate + Jaeyoon_ate + apples_left = 8 := by
  sorry

end NUMINAMATH_GPT_initial_apples_l654_65406


namespace NUMINAMATH_GPT_volume_of_fifth_section_l654_65455

theorem volume_of_fifth_section
  (a : ℕ → ℚ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence constraint
  (h_sum_top_four : a 0 + a 1 + a 2 + a 3 = 3)  -- Sum of the top four sections
  (h_sum_bottom_three : a 6 + a 7 + a 8 = 4)  -- Sum of the bottom three sections
  : a 4 = 67 / 66 := sorry

end NUMINAMATH_GPT_volume_of_fifth_section_l654_65455


namespace NUMINAMATH_GPT_jerry_weekly_earnings_l654_65427

-- Definitions of the given conditions
def pay_per_task : ℕ := 40
def hours_per_task : ℕ := 2
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7

-- Calculated values from the conditions
def tasks_per_day : ℕ := hours_per_day / hours_per_task
def tasks_per_week : ℕ := tasks_per_day * days_per_week
def total_earnings : ℕ := pay_per_task * tasks_per_week

-- Theorem to prove
theorem jerry_weekly_earnings : total_earnings = 1400 := by
  sorry

end NUMINAMATH_GPT_jerry_weekly_earnings_l654_65427


namespace NUMINAMATH_GPT_hugo_probability_l654_65448

noncomputable def P_hugo_first_roll_seven_given_win (P_Hugo_wins : ℚ) (P_first_roll_seven : ℚ)
  (P_all_others_roll_less_than_seven : ℚ) : ℚ :=
(P_first_roll_seven * P_all_others_roll_less_than_seven) / P_Hugo_wins

theorem hugo_probability :
  let P_Hugo_wins := (1 : ℚ) / 4
  let P_first_roll_seven := (1 : ℚ) / 8
  let P_all_others_roll_less_than_seven := (27 : ℚ) / 64
  P_hugo_first_roll_seven_given_win P_Hugo_wins P_first_roll_seven P_all_others_roll_less_than_seven = (27 : ℚ) / 128 :=
by
  sorry

end NUMINAMATH_GPT_hugo_probability_l654_65448


namespace NUMINAMATH_GPT_rose_spent_on_food_l654_65497

theorem rose_spent_on_food (T : ℝ) 
  (h_clothing : 0.5 * T = 0.5 * T)
  (h_other_items : 0.3 * T = 0.3 * T)
  (h_total_tax : 0.044 * T = 0.044 * T)
  (h_tax_clothing : 0.04 * 0.5 * T = 0.02 * T)
  (h_tax_other_items : 0.08 * 0.3 * T = 0.024 * T) :
  (0.2 * T = T - (0.5 * T + 0.3 * T)) :=
by sorry

end NUMINAMATH_GPT_rose_spent_on_food_l654_65497


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l654_65456

def op (a b : ℕ) : ℕ := a ^ b + b ^ a

theorem part_a (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : op a b = op b a :=
by
  dsimp [op]
  rw [add_comm]

theorem part_b (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op a (op b c) = op (op a b) c) :=
by
  -- example counter: a = 2, b = 2, c = 2 
  -- 2 ^ (2^2 + 2^2) + (2^2 + 2^2) ^ 2 ≠ (2^2 + 2 ^ 2) ^ 2 + 8 ^ 2
  sorry

theorem part_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ¬ (op (op a b) (op b c) = op (op b a) (op c b)) :=
by
  -- example counter: a = 2, b = 3, c = 2 
  -- This will involve specific calculations showing the inequality.
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l654_65456


namespace NUMINAMATH_GPT_work_done_isothermal_l654_65458

variable (n : ℕ) (R T : ℝ) (P DeltaV : ℝ)

-- Definitions based on the conditions
def isobaric_work (P DeltaV : ℝ) := P * DeltaV

noncomputable def isobaric_heat (P DeltaV : ℝ) : ℝ :=
  (5 / 2) * P * DeltaV

noncomputable def isothermal_work (Q_iso : ℝ) : ℝ := Q_iso

theorem work_done_isothermal :
  ∃ (n R : ℝ) (P DeltaV : ℝ),
    isobaric_work P DeltaV = 20 ∧
    isothermal_work (isobaric_heat P DeltaV) = 50 :=
by 
  sorry

end NUMINAMATH_GPT_work_done_isothermal_l654_65458


namespace NUMINAMATH_GPT_peter_work_days_l654_65460

variable (W M P : ℝ)
variable (h1 : M + P = W / 20) -- Combined rate of Matt and Peter
variable (h2 : 12 * (W / 20) + 14 * P = W) -- Work done by Matt and Peter for 12 days + Peter's remaining work

theorem peter_work_days :
  P = W / 35 :=
by
  sorry

end NUMINAMATH_GPT_peter_work_days_l654_65460


namespace NUMINAMATH_GPT_perimeter_of_triangle_eq_28_l654_65400

-- Definitions of conditions
variables (p : ℝ)
def inradius : ℝ := 2.0
def area : ℝ := 28

-- Main theorem statement
theorem perimeter_of_triangle_eq_28 : p = 28 :=
  by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_eq_28_l654_65400


namespace NUMINAMATH_GPT_max_students_divide_equal_pen_pencil_l654_65447

theorem max_students_divide_equal_pen_pencil : Nat.gcd 2500 1575 = 25 := 
by
  sorry

end NUMINAMATH_GPT_max_students_divide_equal_pen_pencil_l654_65447


namespace NUMINAMATH_GPT_B_won_third_four_times_l654_65473

noncomputable def first_place := 5
noncomputable def second_place := 2
noncomputable def third_place := 1

structure ContestantScores :=
  (A_score : ℕ)
  (B_score : ℕ)
  (C_score : ℕ)

def competition_results (A B C : ContestantScores) (a b c : ℕ) : Prop :=
  A.A_score = 26 ∧ B.B_score = 11 ∧ C.C_score = 11 ∧ 1 = 1 ∧ -- B won first place once is synonymous to holding true
  a > b ∧ b > c ∧ a = 5 ∧ b = 2 ∧ c = 1

theorem B_won_third_four_times :
  ∃ (A B C : ContestantScores), competition_results A B C first_place second_place third_place → 
  B.B_score = 4 * third_place + first_place := 
sorry

end NUMINAMATH_GPT_B_won_third_four_times_l654_65473


namespace NUMINAMATH_GPT_find_g_l654_65467

-- Define given functions and terms
def f1 (x : ℝ) := 7 * x^4 - 4 * x^3 + 2 * x - 5
def f2 (x : ℝ) := 5 * x^3 - 3 * x^2 + 4 * x - 1
def g (x : ℝ) := -7 * x^4 + 9 * x^3 - 3 * x^2 + 2 * x + 4

-- Theorem to prove that g(x) satisfies the given condition
theorem find_g : ∀ x : ℝ, f1 x + g x = f2 x :=
by 
  -- Alternatively: Proof is required here
  sorry

end NUMINAMATH_GPT_find_g_l654_65467


namespace NUMINAMATH_GPT_min_value_expression_l654_65421

theorem min_value_expression (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  ∃ (z : ℝ), z = (1 / (2 * x) + x / (y + 1)) ∧ z = 5 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l654_65421


namespace NUMINAMATH_GPT_smallest_positive_even_integer_l654_65492

noncomputable def smallest_even_integer (n : ℕ) : ℕ := 
  if 2 * n > 0 ∧ (3^(n * (n + 1) / 8)) > 500 then n else 0

theorem smallest_positive_even_integer :
  smallest_even_integer 6 = 6 :=
by
  -- Skipping the proofs
  sorry

end NUMINAMATH_GPT_smallest_positive_even_integer_l654_65492


namespace NUMINAMATH_GPT_friends_meet_first_time_at_4pm_l654_65486

def lcm_four_times (a b c d : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

def first_meeting_time (start_time_minutes: ℕ) (lap_anna lap_stephanie lap_james lap_carlos: ℕ) : ℕ :=
  start_time_minutes + lcm_four_times lap_anna lap_stephanie lap_james lap_carlos

theorem friends_meet_first_time_at_4pm :
  first_meeting_time 600 5 8 9 12 = 960 :=
by
  -- where 600 represents 10:00 AM in minutes since midnight and 960 represents 4:00 PM
  sorry

end NUMINAMATH_GPT_friends_meet_first_time_at_4pm_l654_65486


namespace NUMINAMATH_GPT_original_hourly_wage_l654_65475

theorem original_hourly_wage 
  (daily_wage_increase : ∀ W : ℝ, 1.60 * W + 10 = 45)
  (work_hours : ℝ := 8) : 
  ∃ W_hourly : ℝ, W_hourly = 2.73 :=
by 
  have W : ℝ := (45 - 10) / 1.60 
  have W_hourly : ℝ := W / work_hours
  use W_hourly 
  sorry

end NUMINAMATH_GPT_original_hourly_wage_l654_65475


namespace NUMINAMATH_GPT_text_messages_relationship_l654_65478

theorem text_messages_relationship (l x : ℕ) (h_l : l = 111) (h_combined : l + x = 283) : x = l + 61 :=
by sorry

end NUMINAMATH_GPT_text_messages_relationship_l654_65478


namespace NUMINAMATH_GPT_range_of_a_l654_65418

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∨ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) ∧ ¬ (∀ x : ℝ, x^2 + 2*a*x + 4 > 0 ∧ (a < 1 ∧ 3 - 2*a > 1 ∧ ∀ x : ℝ, 3 - 2*a > 1)) →
  a ≤ -2 ∨ 1 ≤ a ∧ a < 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l654_65418


namespace NUMINAMATH_GPT_find_linear_function_l654_65476

theorem find_linear_function (a m : ℝ) : 
  (∀ x y : ℝ, (x, y) = (-2, -3) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, m) ∨ (x, y) = (1, 3) ∨ (x, y) = (a, 5) → 
  y = 2 * x + 1) → 
  (m = 1 ∧ a = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_linear_function_l654_65476


namespace NUMINAMATH_GPT_quadratic_complete_square_l654_65420

theorem quadratic_complete_square (c n : ℝ) (h1 : ∀ x : ℝ, x^2 + c * x + 20 = (x + n)^2 + 12) (h2: 0 < c) : 
  c = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l654_65420


namespace NUMINAMATH_GPT_simplify_inv_sum_l654_65449

variables {x y z : ℝ}

theorem simplify_inv_sum (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = xyz / (yz + xz + xy) :=
by
  sorry

end NUMINAMATH_GPT_simplify_inv_sum_l654_65449


namespace NUMINAMATH_GPT_max_drinks_amount_l654_65433

noncomputable def initial_milk : ℚ := 3 / 4
noncomputable def rachel_fraction : ℚ := 1 / 2
noncomputable def max_fraction : ℚ := 1 / 3

def amount_rachel_drinks (initial: ℚ) (fraction: ℚ) : ℚ := initial * fraction
def remaining_milk_after_rachel (initial: ℚ) (amount_rachel: ℚ) : ℚ := initial - amount_rachel
def amount_max_drinks (remaining: ℚ) (fraction: ℚ) : ℚ := remaining * fraction

theorem max_drinks_amount :
  amount_max_drinks (remaining_milk_after_rachel initial_milk (amount_rachel_drinks initial_milk rachel_fraction)) max_fraction = 1 / 8 := 
sorry

end NUMINAMATH_GPT_max_drinks_amount_l654_65433


namespace NUMINAMATH_GPT_sequence_100th_term_eq_l654_65435

-- Definitions for conditions
def numerator (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominator (n : ℕ) : ℕ := 2 + (n - 1) * 3

-- The statement of the problem as a Lean 4 theorem
theorem sequence_100th_term_eq :
  (numerator 100) / (denominator 100) = 199 / 299 :=
by
  sorry

end NUMINAMATH_GPT_sequence_100th_term_eq_l654_65435


namespace NUMINAMATH_GPT_cos_double_angle_sub_pi_six_l654_65453

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π / 3)
variable (h2 : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5)

theorem cos_double_angle_sub_pi_six :
  Real.cos (2 * α - π / 6) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_sub_pi_six_l654_65453


namespace NUMINAMATH_GPT_circle_passing_points_l654_65415

theorem circle_passing_points (x y : ℝ) :
  (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) →
  x^2 + y^2 - 4*x - 6*y = 0 :=
by
  intros h
  cases h
  case inl h₁ => 
    rw [h₁.1, h₁.2]
    ring
  case inr h₁ =>
    cases h₁
    case inl h₂ => 
      rw [h₂.1, h₂.2]
      ring
    case inr h₂ =>
      rw [h₂.1, h₂.2]
      ring

end NUMINAMATH_GPT_circle_passing_points_l654_65415


namespace NUMINAMATH_GPT_calculate_expression_l654_65425

theorem calculate_expression :
  (0.5 ^ 4 / 0.05 ^ 3) = 500 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l654_65425


namespace NUMINAMATH_GPT_distance_house_to_market_l654_65432

-- Define the conditions
def distance_house_to_school := 50
def total_distance_walked := 140

-- Define the question as a theorem with the correct answer
theorem distance_house_to_market : 
  ∀ (house_to_school school_to_house total_distance market : ℕ), 
  house_to_school = distance_house_to_school →
  school_to_house = distance_house_to_school →
  total_distance = total_distance_walked →
  house_to_school + school_to_house + market = total_distance →
  market = 40 :=
by
  intros house_to_school school_to_house total_distance market 
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_distance_house_to_market_l654_65432


namespace NUMINAMATH_GPT_fish_cost_l654_65477

theorem fish_cost (F P : ℝ) (h1 : 4 * F + 2 * P = 530) (h2 : 7 * F + 3 * P = 875) : F = 80 := 
by
  sorry

end NUMINAMATH_GPT_fish_cost_l654_65477


namespace NUMINAMATH_GPT_find_definite_integers_l654_65484

theorem find_definite_integers (n d e f : ℕ) (h₁ : n = d + Int.sqrt (e + Int.sqrt f)) 
    (h₂: ∀ x : ℝ, x = d + Int.sqrt (e + Int.sqrt f) → 
        (4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12 * x - 5))
        : d + e + f = 76 :=
sorry

end NUMINAMATH_GPT_find_definite_integers_l654_65484


namespace NUMINAMATH_GPT_ratio_of_areas_l654_65436

noncomputable def length_field : ℝ := 16
noncomputable def width_field : ℝ := length_field / 2
noncomputable def area_field : ℝ := length_field * width_field
noncomputable def side_pond : ℝ := 4
noncomputable def area_pond : ℝ := side_pond * side_pond
noncomputable def ratio_area_pond_to_field : ℝ := area_pond / area_field

theorem ratio_of_areas :
  ratio_area_pond_to_field = 1 / 8 :=
  by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l654_65436


namespace NUMINAMATH_GPT_HunterScoreIs45_l654_65489

variable (G J H : ℕ)
variable (h1 : G = J + 10)
variable (h2 : J = 2 * H)
variable (h3 : G = 100)

theorem HunterScoreIs45 : H = 45 := by
  sorry

end NUMINAMATH_GPT_HunterScoreIs45_l654_65489


namespace NUMINAMATH_GPT_total_travel_expenses_l654_65419

noncomputable def cost_of_fuel_tank := 45
noncomputable def miles_per_tank := 500
noncomputable def journey_distance := 2000
noncomputable def food_ratio := 3 / 5
noncomputable def hotel_cost_per_night := 80
noncomputable def number_of_hotel_nights := 3
noncomputable def fuel_cost_increase := 5

theorem total_travel_expenses :
  let number_of_refills := journey_distance / miles_per_tank
  let first_refill_cost := cost_of_fuel_tank
  let second_refill_cost := first_refill_cost + fuel_cost_increase
  let third_refill_cost := second_refill_cost + fuel_cost_increase
  let fourth_refill_cost := third_refill_cost + fuel_cost_increase
  let total_fuel_cost := first_refill_cost + second_refill_cost + third_refill_cost + fourth_refill_cost
  let total_food_cost := food_ratio * total_fuel_cost
  let total_hotel_cost := hotel_cost_per_night * number_of_hotel_nights
  let total_expenses := total_fuel_cost + total_food_cost + total_hotel_cost
  total_expenses = 576 := by sorry

end NUMINAMATH_GPT_total_travel_expenses_l654_65419


namespace NUMINAMATH_GPT_find_other_number_l654_65483

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l654_65483


namespace NUMINAMATH_GPT_quadratic_double_root_eq1_quadratic_double_root_eq2_l654_65402

theorem quadratic_double_root_eq1 :
  (∃ r : ℝ , ∃ s : ℝ, (r ≠ s) ∧ (
  (1 : ℝ) * r^2 + (-3 : ℝ) * r + (2 : ℝ) = 0 ∧
  (1 : ℝ) * s^2 + (-3 : ℝ) * s + (2 : ℝ) = 0 ∧
  (r = 2 * s ∨ s = 2 * r) 
  )) := 
  sorry

theorem quadratic_double_root_eq2 :
  (∃ a b : ℝ, a ≠ 0 ∧
  ((∃ r : ℝ, (-b / a = 2 + r) ∧ (-6 / a = 2 * r)) ∨ 
  ((-b / a = 2 + 1) ∧ (-6 / a = 2 * 1))) ∧ 
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9))) :=
  sorry

end NUMINAMATH_GPT_quadratic_double_root_eq1_quadratic_double_root_eq2_l654_65402


namespace NUMINAMATH_GPT_jimmy_irene_total_payment_l654_65401

def cost_jimmy_shorts : ℝ := 3 * 15
def cost_irene_shirts : ℝ := 5 * 17
def total_cost_before_discount : ℝ := cost_jimmy_shorts + cost_irene_shirts
def discount : ℝ := total_cost_before_discount * 0.10
def total_paid : ℝ := total_cost_before_discount - discount

theorem jimmy_irene_total_payment : total_paid = 117 := by
  sorry

end NUMINAMATH_GPT_jimmy_irene_total_payment_l654_65401


namespace NUMINAMATH_GPT_domain_of_composed_function_l654_65416

theorem domain_of_composed_function
  (f : ℝ → ℝ)
  (dom_f : ∀ x, 0 ≤ x ∧ x ≤ 4 → f x ≠ 0) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → f (x^2) ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_composed_function_l654_65416


namespace NUMINAMATH_GPT_election_required_percentage_l654_65434

def votes_cast : ℕ := 10000

def geoff_percentage : ℕ := 5
def geoff_received_votes := (geoff_percentage * votes_cast) / 1000

def extra_votes_needed : ℕ := 5000
def total_votes_needed := geoff_received_votes + extra_votes_needed

def required_percentage := (total_votes_needed * 100) / votes_cast

theorem election_required_percentage : required_percentage = 505 / 10 :=
by
  sorry

end NUMINAMATH_GPT_election_required_percentage_l654_65434


namespace NUMINAMATH_GPT_Danielle_has_6_rooms_l654_65482

axiom Danielle_rooms : ℕ
axiom Heidi_rooms : ℕ
axiom Grant_rooms : ℕ

axiom Heidi_has_3_times_Danielle : Heidi_rooms = 3 * Danielle_rooms
axiom Grant_has_1_9_Heidi : Grant_rooms = Heidi_rooms / 9
axiom Grant_has_2_rooms : Grant_rooms = 2

theorem Danielle_has_6_rooms : Danielle_rooms = 6 :=
by {
  -- proof steps would go here
  sorry
}

end NUMINAMATH_GPT_Danielle_has_6_rooms_l654_65482


namespace NUMINAMATH_GPT_equal_copper_content_alloy_l654_65422

theorem equal_copper_content_alloy (a b : ℝ) :
  ∃ x : ℝ, 0 < x ∧ x < 10 ∧
  (10 - x) * a + x * b = (15 - x) * b + x * a → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_equal_copper_content_alloy_l654_65422


namespace NUMINAMATH_GPT_strudel_price_l654_65442

def initial_price := 80
def first_increment (P0 : ℕ) := P0 * 3 / 2
def second_increment (P1 : ℕ) := P1 * 3 / 2
def final_price (P2 : ℕ) := P2 / 2

theorem strudel_price (P0 : ℕ) (P1 : ℕ) (P2 : ℕ) (Pf : ℕ)
  (h0 : P0 = initial_price)
  (h1 : P1 = first_increment P0)
  (h2 : P2 = second_increment P1)
  (hf : Pf = final_price P2) :
  Pf = 90 :=
sorry

end NUMINAMATH_GPT_strudel_price_l654_65442


namespace NUMINAMATH_GPT_hoseok_wire_length_l654_65440

theorem hoseok_wire_length (side_length : ℕ) (equilateral : Prop) (leftover_wire : ℕ) (total_wire : ℕ)  
  (eq_side : side_length = 19) (eq_leftover : leftover_wire = 15) 
  (eq_equilateral : equilateral) : total_wire = 72 :=
sorry

end NUMINAMATH_GPT_hoseok_wire_length_l654_65440


namespace NUMINAMATH_GPT_ratio_of_amounts_l654_65481

theorem ratio_of_amounts
    (initial_cents : ℕ)
    (given_to_peter_cents : ℕ)
    (remaining_nickels : ℕ)
    (nickel_value : ℕ := 5)
    (nickels_initial := initial_cents / nickel_value)
    (nickels_to_peter := given_to_peter_cents / nickel_value)
    (nickels_remaining := nickels_initial - nickels_to_peter)
    (nickels_given_to_randi := nickels_remaining - remaining_nickels)
    (cents_to_randi := nickels_given_to_randi * nickel_value)
    (cents_initial : initial_cents = 95)
    (cents_peter : given_to_peter_cents = 25)
    (nickels_left : remaining_nickels = 4)
    :
    (cents_to_randi / given_to_peter_cents) = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_amounts_l654_65481


namespace NUMINAMATH_GPT_incorrect_transformation_l654_65494

-- Definitions based on conditions
variable (a b c : ℝ)

-- Conditions
axiom eq_add_six (h : a = b) : a + 6 = b + 6
axiom eq_div_nine (h : a = b) : a / 9 = b / 9
axiom eq_mul_c (h : a / c = b / c) (hc : c ≠ 0) : a = b
axiom eq_div_neg_two (h : -2 * a = -2 * b) : a = b

-- Proving the incorrect transformation statement
theorem incorrect_transformation : ¬ (a = -b) ∧ (-2 * a = -2 * b → a = b) := by
  sorry

end NUMINAMATH_GPT_incorrect_transformation_l654_65494


namespace NUMINAMATH_GPT_typist_original_salary_l654_65461

theorem typist_original_salary (x : ℝ) (h : (x * 1.10 * 0.95 = 4180)) : x = 4000 :=
by sorry

end NUMINAMATH_GPT_typist_original_salary_l654_65461


namespace NUMINAMATH_GPT_computer_price_after_9_years_l654_65464

theorem computer_price_after_9_years 
  (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) 
  (initial_price_eq : initial_price = 8100)
  (decrease_factor_eq : decrease_factor = 1 - 1/3)
  (years_eq : years = 9) :
  initial_price * (decrease_factor ^ (years / 3)) = 2400 := 
by
  sorry

end NUMINAMATH_GPT_computer_price_after_9_years_l654_65464


namespace NUMINAMATH_GPT_derivative_at_x₀_l654_65465

-- Define the function y = (x - 2)^2
def f (x : ℝ) : ℝ := (x - 2) ^ 2

-- Define the point of interest
def x₀ : ℝ := 1

-- State the problem and the correct answer
theorem derivative_at_x₀ : (deriv f x₀) = -2 := by
  sorry

end NUMINAMATH_GPT_derivative_at_x₀_l654_65465


namespace NUMINAMATH_GPT_find_middle_side_length_l654_65444

theorem find_middle_side_length (a b c : ℕ) (h1 : a + b + c = 2022) (h2 : c - b = 1) (h3 : b - a = 2) :
  b = 674 := 
by
  -- The proof goes here, but we skip it using sorry.
  sorry

end NUMINAMATH_GPT_find_middle_side_length_l654_65444


namespace NUMINAMATH_GPT_average_of_first_16_even_numbers_l654_65405

theorem average_of_first_16_even_numbers : 
  (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30 + 32) / 16 = 17 := 
by sorry

end NUMINAMATH_GPT_average_of_first_16_even_numbers_l654_65405


namespace NUMINAMATH_GPT_combined_population_correct_l654_65491

theorem combined_population_correct (W PP LH N : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : LH = 2 * W + 600)
  (hN : N = 3 * (PP - W)) :
  PP + LH + N = 24900 :=
by
  sorry

end NUMINAMATH_GPT_combined_population_correct_l654_65491


namespace NUMINAMATH_GPT_difference_of_numbers_l654_65431

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 22500) (h2 : b = 10 * a + 5) : b - a = 18410 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l654_65431


namespace NUMINAMATH_GPT_sum_is_integer_l654_65493

theorem sum_is_integer (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  x + y + z = 0 :=
  sorry

end NUMINAMATH_GPT_sum_is_integer_l654_65493


namespace NUMINAMATH_GPT_find_zero_function_l654_65487

noncomputable def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x ^ 714 + y) = f (x ^ 2019) + f (y ^ 122)

theorem find_zero_function (f : ℝ → ℝ) (h : satisfiesCondition f) :
  ∀ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_find_zero_function_l654_65487


namespace NUMINAMATH_GPT_discount_calc_l654_65428

noncomputable def discount_percentage 
    (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : ℝ :=
  let marked_price := cost_price + (markup_percentage / 100 * cost_price)
  let discount := marked_price - selling_price
  (discount / marked_price) * 100

theorem discount_calc :
  discount_percentage 540 15 460 = 25.92 :=
by
  sorry

end NUMINAMATH_GPT_discount_calc_l654_65428


namespace NUMINAMATH_GPT_Lacy_correct_percentage_l654_65479

def problems_exam (y : ℕ) := 10 * y
def problems_section1 (y : ℕ) := 6 * y
def problems_section2 (y : ℕ) := 4 * y
def missed_section1 (y : ℕ) := 2 * y
def missed_section2 (y : ℕ) := y
def solved_section1 (y : ℕ) := problems_section1 y - missed_section1 y
def solved_section2 (y : ℕ) := problems_section2 y - missed_section2 y
def total_solved (y : ℕ) := solved_section1 y + solved_section2 y
def percent_correct (y : ℕ) := (total_solved y : ℚ) / (problems_exam y) * 100

theorem Lacy_correct_percentage (y : ℕ) : percent_correct y = 70 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_Lacy_correct_percentage_l654_65479


namespace NUMINAMATH_GPT_fraction_transform_l654_65471

theorem fraction_transform {x : ℤ} :
  (537 - x : ℚ) / (463 + x) = 1 / 9 ↔ x = 437 := by
sorry

end NUMINAMATH_GPT_fraction_transform_l654_65471


namespace NUMINAMATH_GPT_geometric_sequence_sum_l654_65446

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_a1 : a 1 = 3)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 4 + a 5 + a 6 = 168 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l654_65446


namespace NUMINAMATH_GPT_houses_in_block_l654_65417

theorem houses_in_block (junk_per_house : ℕ) (total_junk : ℕ) (h_junk : junk_per_house = 2) (h_total : total_junk = 14) :
  total_junk / junk_per_house = 7 := by
  sorry

end NUMINAMATH_GPT_houses_in_block_l654_65417


namespace NUMINAMATH_GPT_urn_problem_l654_65408

noncomputable def probability_of_two_black_balls : ℚ := (10 / 15) * (9 / 14)

theorem urn_problem : probability_of_two_black_balls = 3 / 7 := 
by
  sorry

end NUMINAMATH_GPT_urn_problem_l654_65408


namespace NUMINAMATH_GPT_negation_proposition_false_l654_65441

variable (a : ℝ)

theorem negation_proposition_false : ¬ (∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4) :=
sorry

end NUMINAMATH_GPT_negation_proposition_false_l654_65441


namespace NUMINAMATH_GPT_probability_units_digit_odd_l654_65407

theorem probability_units_digit_odd :
  (1 / 2 : ℚ) = 5 / 10 :=
by {
  -- This is the equivalent mathematically correct theorem statement
  -- The proof is omitted as per instructions
  sorry
}

end NUMINAMATH_GPT_probability_units_digit_odd_l654_65407


namespace NUMINAMATH_GPT_triangle_tangency_perimeter_l654_65410

def triangle_perimeter (a b c : ℝ) (s : ℝ) (t : ℝ) (u : ℝ) : ℝ :=
  s + t + u

theorem triangle_tangency_perimeter (a b c : ℝ) (D E F : ℝ) (s : ℝ) (t : ℝ) (u : ℝ)
  (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) 
  (h4 : s + t + u = 3) : triangle_perimeter a b c s t u = 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_tangency_perimeter_l654_65410


namespace NUMINAMATH_GPT_find_element_atomic_mass_l654_65403

-- Define the atomic mass of bromine
def atomic_mass_br : ℝ := 79.904

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 267

-- Define the number of bromine atoms in the compound (assuming n = 1)
def n : ℕ := 1

-- Define the atomic mass of the unknown element X
def atomic_mass_x : ℝ := molecular_weight - n * atomic_mass_br

-- State the theorem to prove
theorem find_element_atomic_mass : atomic_mass_x = 187.096 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_element_atomic_mass_l654_65403


namespace NUMINAMATH_GPT_find_f_l654_65445

-- Define the conditions as hypotheses
def cond1 (f : ℕ) (p : ℕ) : Prop := f + p = 75
def cond2 (f : ℕ) (p : ℕ) : Prop := (f + p) + p = 143

-- The theorem stating that given the conditions, f must be 7
theorem find_f (f p : ℕ) (h1 : cond1 f p) (h2 : cond2 f p) : f = 7 := 
  by
  sorry

end NUMINAMATH_GPT_find_f_l654_65445


namespace NUMINAMATH_GPT_no_common_perfect_squares_l654_65452

theorem no_common_perfect_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (∃ m n : ℕ, a^2 + 4 * b = m^2 ∧ b^2 + 4 * a = n^2) :=
by
  sorry

end NUMINAMATH_GPT_no_common_perfect_squares_l654_65452


namespace NUMINAMATH_GPT_dot_product_equivalence_l654_65429

variable (a : ℝ × ℝ) 
variable (b : ℝ × ℝ)

-- Given conditions
def condition_1 : Prop := a = (2, 1)
def condition_2 : Prop := a - b = (-1, 2)

-- Goal
theorem dot_product_equivalence (h1 : condition_1 a) (h2 : condition_2 a b) : a.1 * b.1 + a.2 * b.2 = 5 :=
  sorry

end NUMINAMATH_GPT_dot_product_equivalence_l654_65429


namespace NUMINAMATH_GPT_area_of_rectangle_is_432_l654_65457

/-- Define the width of the rectangle --/
def width : ℕ := 12

/-- Define the length of the rectangle, which is three times the width --/
def length : ℕ := 3 * width

/-- The area of the rectangle is length multiplied by width --/
def area : ℕ := length * width

/-- Proof problem: the area of the rectangle is 432 square meters --/
theorem area_of_rectangle_is_432 :
  area = 432 :=
sorry

end NUMINAMATH_GPT_area_of_rectangle_is_432_l654_65457


namespace NUMINAMATH_GPT_dot_product_result_l654_65468

open Real

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, 2)

def scale_vec (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add_vec (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_result :
  dot_product (add_vec (scale_vec 2 a) b) a = 6 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_result_l654_65468


namespace NUMINAMATH_GPT_order_of_magnitude_l654_65459

noncomputable def a : Real := 70.3
noncomputable def b : Real := 70.2
noncomputable def c : Real := Real.log 0.3

theorem order_of_magnitude : a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_order_of_magnitude_l654_65459


namespace NUMINAMATH_GPT_c_share_l654_65480

theorem c_share (S : ℝ) (b_share_per_rs c_share_per_rs : ℝ)
  (h1 : S = 246)
  (h2 : b_share_per_rs = 0.65)
  (h3 : c_share_per_rs = 0.40) :
  (c_share_per_rs * S) = 98.40 :=
by sorry

end NUMINAMATH_GPT_c_share_l654_65480


namespace NUMINAMATH_GPT_right_triangle_sides_l654_65423

theorem right_triangle_sides (x y z : ℕ) (h_sum : x + y + z = 156) (h_area : x * y = 2028) (h_pythagorean : z^2 = x^2 + y^2) :
  (x = 39 ∧ y = 52 ∧ z = 65) ∨ (x = 52 ∧ y = 39 ∧ z = 65) :=
by
  admit -- proof goes here

-- Additional details for importing required libraries and setting up the environment
-- are intentionally simplified as per instruction to cover a broader import.

end NUMINAMATH_GPT_right_triangle_sides_l654_65423


namespace NUMINAMATH_GPT_solution_set_f_neg_x_l654_65496

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_f_neg_x (a b : ℝ) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x, f a b (-x) < 0 ↔ (x < -3 ∨ x > 1) :=
by
  intro x
  specialize h (-x)
  sorry

end NUMINAMATH_GPT_solution_set_f_neg_x_l654_65496


namespace NUMINAMATH_GPT_distance_between_points_l654_65454

theorem distance_between_points (A B : ℝ) (hA : |A| = 2) (hB : |B| = 7) :
  |A - B| = 5 ∨ |A - B| = 9 := 
sorry

end NUMINAMATH_GPT_distance_between_points_l654_65454


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_l654_65495

noncomputable def geometric_sequence (a₁ r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n-1)

theorem smallest_n_for_divisibility (h₁ : ∀ n : ℕ, geometric_sequence (1/2 : ℚ) 60 n = (1/2 : ℚ) * 60^(n-1))
    (h₂ : (60 : ℚ) * (1 / 2) = 30)
    (n : ℕ) :
  (∃ n : ℕ, n ≥ 1 ∧ (geometric_sequence (1/2 : ℚ) 60 n) ≥ 10^6) ↔ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_l654_65495


namespace NUMINAMATH_GPT_best_model_l654_65474

theorem best_model (R1 R2 R3 R4 : ℝ) :
  R1 = 0.78 → R2 = 0.85 → R3 = 0.61 → R4 = 0.31 →
  (R2 = max R1 (max R2 (max R3 R4))) :=
by
  intros hR1 hR2 hR3 hR4
  sorry

end NUMINAMATH_GPT_best_model_l654_65474


namespace NUMINAMATH_GPT_central_angle_measure_l654_65438

theorem central_angle_measure (p : ℝ) (x : ℝ) (h1 : p = 1 / 8) (h2 : p = x / 360) : x = 45 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_central_angle_measure_l654_65438


namespace NUMINAMATH_GPT_geese_left_in_the_field_l654_65463

theorem geese_left_in_the_field 
  (initial_geese : ℕ) 
  (geese_flew_away : ℕ) 
  (geese_joined : ℕ)
  (h1 : initial_geese = 372)
  (h2 : geese_flew_away = 178)
  (h3 : geese_joined = 57) :
  initial_geese - geese_flew_away + geese_joined = 251 := by
  sorry

end NUMINAMATH_GPT_geese_left_in_the_field_l654_65463


namespace NUMINAMATH_GPT_cookies_per_person_l654_65488

variable (x y z : ℕ)
variable (h_pos_z : z ≠ 0) -- Ensure z is not zero to avoid division by zero

theorem cookies_per_person (h_cookies : x * y / z = 35) : 35 / 5 = 7 := by
  sorry

end NUMINAMATH_GPT_cookies_per_person_l654_65488


namespace NUMINAMATH_GPT_exp_add_l654_65439

theorem exp_add (a : ℝ) (x₁ x₂ : ℝ) : a^(x₁ + x₂) = a^x₁ * a^x₂ :=
sorry

end NUMINAMATH_GPT_exp_add_l654_65439


namespace NUMINAMATH_GPT_ounces_per_gallon_l654_65499

-- conditions
def gallons_of_milk (james : Type) : ℕ := 3
def ounces_drank (james : Type) : ℕ := 13
def ounces_left (james : Type) : ℕ := 371

-- question
def ounces_in_gallon (james : Type) : ℕ := 128

-- proof statement
theorem ounces_per_gallon (james : Type) :
  (gallons_of_milk james) * (ounces_in_gallon james) = (ounces_left james + ounces_drank james) :=
sorry

end NUMINAMATH_GPT_ounces_per_gallon_l654_65499


namespace NUMINAMATH_GPT_cylinder_height_in_hemisphere_l654_65470

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end NUMINAMATH_GPT_cylinder_height_in_hemisphere_l654_65470


namespace NUMINAMATH_GPT_parabola_focus_distance_l654_65411

theorem parabola_focus_distance (A : ℝ × ℝ) (F : ℝ × ℝ := (1, 0)) 
    (h_parabola : A.2^2 = 4 * A.1) (h_distance : dist A F = 3) :
    A = (2, 2 * Real.sqrt 2) ∨ A = (2, -2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l654_65411


namespace NUMINAMATH_GPT_find_a_b_sum_specific_find_a_b_sum_l654_65412

-- Define the sets A and B based on the given inequalities
def set_A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def set_B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Intersect the sets A and B
def set_A_int_B : Set ℝ := set_A ∩ set_B

-- Define the inequality with parameters a and b
def quad_ineq (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 2 > 0}

-- Define the parameters a and b based on the given condition
noncomputable def a : ℝ := -1
noncomputable def b : ℝ := -1

-- The statement to be proved
theorem find_a_b_sum : ∀ a b : ℝ, set_A ∩ set_B = {x | a * x^2 + b * x + 2 > 0} → a + b = -2 :=
by
  sorry

-- Fixing the parameters a and b for our specific proof condition
theorem specific_find_a_b_sum : a + b = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_sum_specific_find_a_b_sum_l654_65412


namespace NUMINAMATH_GPT_responses_needed_l654_65485

theorem responses_needed (p : ℝ) (q : ℕ) (r : ℕ) : 
  p = 0.6 → q = 370 → r = 222 → 
  q * p = r := 
by
  intros hp hq hr
  rw [hp, hq] 
  sorry

end NUMINAMATH_GPT_responses_needed_l654_65485


namespace NUMINAMATH_GPT_frank_completes_book_in_three_days_l654_65443

-- Define the total number of pages in a book
def total_pages : ℕ := 249

-- Define the number of pages Frank reads per day
def pages_per_day : ℕ := 83

-- Define the number of days Frank needs to finish a book
def days_to_finish_book (total_pages pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

-- Theorem statement to prove that Frank finishes a book in 3 days
theorem frank_completes_book_in_three_days : days_to_finish_book total_pages pages_per_day = 3 := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_frank_completes_book_in_three_days_l654_65443


namespace NUMINAMATH_GPT_range_of_g_l654_65424

noncomputable def f (x : ℝ) : ℝ := 2 * x - 3

noncomputable def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 3 → -29 ≤ g x ∧ g x ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_g_l654_65424


namespace NUMINAMATH_GPT_value_of_expression_l654_65426

theorem value_of_expression (x : ℕ) (h : x = 3) : 2 * x + 3 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l654_65426


namespace NUMINAMATH_GPT_first_term_arithmetic_sequence_l654_65413

theorem first_term_arithmetic_sequence
    (a: ℚ)
    (S_n S_2n: ℕ → ℚ)
    (n: ℕ) 
    (h1: ∀ n > 0, S_n n = (n * (2 * a + (n - 1) * 5)) / 2)
    (h2: ∀ n > 0, S_2n (2 * n) = ((2 * n) * (2 * a + ((2 * n) - 1) * 5)) / 2)
    (h3: ∀ n > 0, (S_2n (2 * n)) / (S_n n) = 4) :
  a = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_first_term_arithmetic_sequence_l654_65413


namespace NUMINAMATH_GPT_arithmetic_sqrt_of_13_l654_65414

theorem arithmetic_sqrt_of_13 : Real.sqrt 13 = Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sqrt_of_13_l654_65414


namespace NUMINAMATH_GPT_taller_tree_height_l654_65490

theorem taller_tree_height
  (h : ℕ)
  (h_shorter_ratio : h - 16 = (3 * h) / 4) : h = 64 := by
  sorry

end NUMINAMATH_GPT_taller_tree_height_l654_65490


namespace NUMINAMATH_GPT_f_has_exactly_one_zero_point_a_range_condition_l654_65472

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * Real.log x + 2 / (x + 1)

theorem f_has_exactly_one_zero_point :
  ∃! x : ℝ, 1 < x ∧ x < Real.exp 2 ∧ f x = 0 := sorry

theorem a_range_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) 1 → ∀ t : ℝ, t ∈ Set.Icc (1 / 2) 2 → f x ≥ t^3 - t^2 - 2 * a * t + 2) → a ≥ 5 / 4 := sorry

end NUMINAMATH_GPT_f_has_exactly_one_zero_point_a_range_condition_l654_65472
