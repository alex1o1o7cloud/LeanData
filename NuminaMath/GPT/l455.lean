import Mathlib

namespace NUMINAMATH_GPT_max_value_x_plus_y_l455_45549

theorem max_value_x_plus_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 48) (hx_mult_4 : x % 4 = 0) : x + y ≤ 49 :=
sorry

end NUMINAMATH_GPT_max_value_x_plus_y_l455_45549


namespace NUMINAMATH_GPT_unique_valid_configuration_l455_45542

-- Define the conditions: a rectangular array of chairs organized in rows and columns such that
-- each row contains the same number of chairs as every other row, each column contains the
-- same number of chairs as every other column, with at least two chairs in every row and column.
def valid_array_configuration (rows cols : ℕ) : Prop :=
  2 ≤ rows ∧ 2 ≤ cols ∧ rows * cols = 49

-- The theorem statement: determine how many valid arrays are possible given the conditions.
theorem unique_valid_configuration : ∃! (rows cols : ℕ), valid_array_configuration rows cols :=
sorry

end NUMINAMATH_GPT_unique_valid_configuration_l455_45542


namespace NUMINAMATH_GPT_minister_can_organize_traffic_l455_45517

-- Definition of cities and roads
structure City (α : Type) :=
(road : α → α → Prop)

-- Defining the Minister's goal
def organize_traffic {α : Type} (c : City α) (num_days : ℕ) : Prop :=
∀ x y : α, c.road x y → num_days ≤ 214

theorem minister_can_organize_traffic :
  ∃ (c : City ℕ) (num_days : ℕ), (num_days ≤ 214 ∧ organize_traffic c num_days) :=
by {
  sorry
}

end NUMINAMATH_GPT_minister_can_organize_traffic_l455_45517


namespace NUMINAMATH_GPT_car_2_speed_proof_l455_45513

noncomputable def car_1_speed : ℝ := 30
noncomputable def car_1_start_time : ℝ := 9
noncomputable def car_2_start_delay : ℝ := 10 / 60
noncomputable def catch_up_time : ℝ := 10.5
noncomputable def car_2_start_time : ℝ := car_1_start_time + car_2_start_delay
noncomputable def travel_duration : ℝ := catch_up_time - car_2_start_time
noncomputable def car_1_head_start_distance : ℝ := car_1_speed * car_2_start_delay
noncomputable def car_1_travel_distance : ℝ := car_1_speed * travel_duration
noncomputable def total_distance : ℝ := car_1_head_start_distance + car_1_travel_distance
noncomputable def car_2_speed : ℝ := total_distance / travel_duration

theorem car_2_speed_proof : car_2_speed = 33.75 := 
by 
  sorry

end NUMINAMATH_GPT_car_2_speed_proof_l455_45513


namespace NUMINAMATH_GPT_stock_investment_net_increase_l455_45524

theorem stock_investment_net_increase :
  ∀ (initial_investment : ℝ)
    (increase_first_year : ℝ)
    (decrease_second_year : ℝ)
    (increase_third_year : ℝ),
  initial_investment = 100 → 
  increase_first_year = 0.60 → 
  decrease_second_year = 0.30 → 
  increase_third_year = 0.20 → 
  ((initial_investment * (1 + increase_first_year)) * (1 - decrease_second_year)) * (1 + increase_third_year) - initial_investment = 34.40 :=
by 
  intros initial_investment increase_first_year decrease_second_year increase_third_year 
  intros h_initial_investment h_increase_first_year h_decrease_second_year h_increase_third_year 
  rw [h_initial_investment, h_increase_first_year, h_decrease_second_year, h_increase_third_year]
  sorry

end NUMINAMATH_GPT_stock_investment_net_increase_l455_45524


namespace NUMINAMATH_GPT_petya_run_12_seconds_l455_45580

-- Define the conditions
variable (petya_speed classmates_speed : ℕ → ℕ) -- speeds of Petya and his classmates
variable (total_distance : ℕ := 100) -- each participant needs to run 100 meters
variable (initial_total_distance_run : ℕ := 288) -- total distance run by all in the first 12 seconds
variable (remaining_distance_when_petya_finished : ℕ := 40) -- remaining distance for others when Petya finished
variable (time_to_first_finish : ℕ) -- the time Petya takes to finish the race

-- Assume constant speeds for all participants
axiom constant_speed_petya (t : ℕ) : petya_speed t = petya_speed 0
axiom constant_speed_classmates (t : ℕ) : classmates_speed t = classmates_speed 0

-- Summarized total distances run by participants
axiom total_distance_run_all (t : ℕ) :
  petya_speed t * t + classmates_speed t * t = initial_total_distance_run + remaining_distance_when_petya_finished + (total_distance - remaining_distance_when_petya_finished) * 3

-- Given conditions converted to Lean
axiom initial_distance_run (t : ℕ) :
  t = 12 → petya_speed t * t + classmates_speed t * t = initial_total_distance_run

axiom petya_completion (t : ℕ) :
  t = time_to_first_finish → petya_speed t * t = total_distance

axiom remaining_distance_classmates (t : ℕ) :
  t = time_to_first_finish → classmates_speed t * (t - time_to_first_finish) = remaining_distance_when_petya_finished
  
-- Define the proof goal using the conditions
theorem petya_run_12_seconds (d : ℕ) :
  (∃ t, t = 12 ∧ d = petya_speed t * t) → d = 80 :=
by
  sorry

end NUMINAMATH_GPT_petya_run_12_seconds_l455_45580


namespace NUMINAMATH_GPT_inequality_of_ab_bc_ca_l455_45590

theorem inequality_of_ab_bc_ca (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^4 + b^4 + c^4 = 3) : 
  (1 / (4 - a * b)) + (1 / (4 - b * c)) + (1 / (4 - c * a)) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_ab_bc_ca_l455_45590


namespace NUMINAMATH_GPT_circle_condition_iff_l455_45506

-- Given a condition a < 2, we need to show it is a necessary and sufficient condition
-- for the equation x^2 + y^2 - 2x + 2y + a = 0 to represent a circle.

theorem circle_condition_iff (a : ℝ) :
  (∃ (x y : ℝ), (x - 1) ^ 2 + (y + 1) ^ 2 = 2 - a) ↔ (a < 2) :=
sorry

end NUMINAMATH_GPT_circle_condition_iff_l455_45506


namespace NUMINAMATH_GPT_coordinates_in_second_quadrant_l455_45505

section 
variable (x y : ℝ)
variable (hx : x = -7)
variable (hy : y = 4)
variable (quadrant : x < 0 ∧ y > 0)
variable (distance_x : |y| = 4)
variable (distance_y : |x| = 7)

theorem coordinates_in_second_quadrant :
  (x, y) = (-7, 4) := by
  sorry
end

end NUMINAMATH_GPT_coordinates_in_second_quadrant_l455_45505


namespace NUMINAMATH_GPT_find_breadth_of_wall_l455_45520

theorem find_breadth_of_wall
  (b h l V : ℝ)
  (h1 : V = 12.8)
  (h2 : h = 5 * b)
  (h3 : l = 8 * h) :
  b = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_find_breadth_of_wall_l455_45520


namespace NUMINAMATH_GPT_length_of_PS_l455_45507

noncomputable def triangle_segments : ℝ := 
  let PR := 15
  let ratio_PS_SR := 3 / 4
  let total_length := 15
  let SR := total_length / (1 + ratio_PS_SR)
  let PS := ratio_PS_SR * SR
  PS

theorem length_of_PS :
  triangle_segments = 45 / 7 :=
by
  sorry

end NUMINAMATH_GPT_length_of_PS_l455_45507


namespace NUMINAMATH_GPT_multiple_time_second_artifact_is_three_l455_45558

-- Define the conditions as Lean definitions
def months_in_year : ℕ := 12
def total_time_both_artifacts_years : ℕ := 10
def total_time_first_artifact_months : ℕ := 6 + 24

-- Convert total time of both artifacts from years to months
def total_time_both_artifacts_months : ℕ := total_time_both_artifacts_years * months_in_year

-- Define the time for the second artifact
def time_second_artifact_months : ℕ :=
  total_time_both_artifacts_months - total_time_first_artifact_months

-- Define the sought multiple
def multiple_second_first : ℕ :=
  time_second_artifact_months / total_time_first_artifact_months

-- The theorem stating the required proof
theorem multiple_time_second_artifact_is_three :
  multiple_second_first = 3 :=
by
  sorry

end NUMINAMATH_GPT_multiple_time_second_artifact_is_three_l455_45558


namespace NUMINAMATH_GPT_percentage_calculation_l455_45510

/-- If x % of 375 equals 5.4375, then x % equals 1.45 %. -/
theorem percentage_calculation (x : ℝ) (h : x / 100 * 375 = 5.4375) : x = 1.45 := 
sorry

end NUMINAMATH_GPT_percentage_calculation_l455_45510


namespace NUMINAMATH_GPT_calculate_parallel_segment_length_l455_45579

theorem calculate_parallel_segment_length :
  ∀ (d : ℝ), 
    ∃ (X Y Z P : Type) 
    (XY YZ XZ : ℝ), 
    XY = 490 ∧ 
    YZ = 520 ∧ 
    XZ = 560 ∧ 
    ∃ (D D' E E' F F' : Type),
      (D ≠ E ∧ E ≠ F ∧ F ≠ D') ∧  
      (XZ - (d * (520/490) + d * (520/560))) = d → d = 268.148148 :=
by
  sorry

end NUMINAMATH_GPT_calculate_parallel_segment_length_l455_45579


namespace NUMINAMATH_GPT_distance_focus_directrix_l455_45523

theorem distance_focus_directrix (y x : ℝ) (h : y^2 = 2 * x) : x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_distance_focus_directrix_l455_45523


namespace NUMINAMATH_GPT_least_possible_integral_BC_l455_45565

theorem least_possible_integral_BC :
  ∃ (BC : ℕ), (BC > 0) ∧ (BC ≥ 15) ∧ 
    (7 + BC > 15) ∧ (25 + 10 > BC) ∧ 
    (7 + 15 > BC) ∧ (25 + BC > 10) := by
    sorry

end NUMINAMATH_GPT_least_possible_integral_BC_l455_45565


namespace NUMINAMATH_GPT_inequality_holds_for_any_xyz_l455_45573

theorem inequality_holds_for_any_xyz (x y z : ℝ) : 
  x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_holds_for_any_xyz_l455_45573


namespace NUMINAMATH_GPT_degrees_to_radians_90_l455_45557

theorem degrees_to_radians_90 : (90 : ℝ) * (Real.pi / 180) = (Real.pi / 2) :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_90_l455_45557


namespace NUMINAMATH_GPT_toy_ratio_l455_45598

variable (Jaxon : ℕ) (Gabriel : ℕ) (Jerry : ℕ)

theorem toy_ratio (h1 : Jerry = Gabriel + 8) 
                  (h2 : Jaxon = 15)
                  (h3 : Gabriel + Jerry + Jaxon = 83) :
                  Gabriel / Jaxon = 2 := 
by
  sorry

end NUMINAMATH_GPT_toy_ratio_l455_45598


namespace NUMINAMATH_GPT_roster_representation_of_M_l455_45568

def M : Set ℚ := {x | ∃ m n : ℤ, x = m / n ∧ |m| < 2 ∧ 1 ≤ n ∧ n ≤ 3}

theorem roster_representation_of_M :
  M = {-1, -1/2, -1/3, 0, 1/2, 1/3} :=
by sorry

end NUMINAMATH_GPT_roster_representation_of_M_l455_45568


namespace NUMINAMATH_GPT_monkey_climb_ladder_l455_45500

theorem monkey_climb_ladder (n : ℕ) 
  (h1 : ∀ k, (k % 18 = 0 → (k - 18 + 10) % 26 = 8))
  (h2 : ∀ m, (m % 10 = 0 → (m - 10 + 18) % 26 = 18))
  (h3 : ∀ l, (l % 18 = 0 ∧ l % 10 = 0 → l = 0 ∨ l = 26)):
  n = 26 :=
by
  sorry

end NUMINAMATH_GPT_monkey_climb_ladder_l455_45500


namespace NUMINAMATH_GPT_robot_path_length_l455_45563

/--
A robot moves in the plane in a straight line, but every one meter it turns 90° to the right or to the left. At some point it reaches its starting point without having visited any other point more than once, and stops immediately. Prove that the possible path lengths of the robot are 4k for some integer k with k >= 3.
-/
theorem robot_path_length (n : ℕ) (h : n > 0) (Movement : n % 4 = 0) :
  ∃ k : ℕ, n = 4 * k ∧ k ≥ 3 :=
sorry

end NUMINAMATH_GPT_robot_path_length_l455_45563


namespace NUMINAMATH_GPT_find_sale_in_third_month_l455_45521

def sale_in_first_month := 5700
def sale_in_second_month := 8550
def sale_in_fourth_month := 3850
def sale_in_fifth_month := 14045
def average_sale := 7800
def num_months := 5
def total_sales := average_sale * num_months

theorem find_sale_in_third_month (X : ℕ) 
  (H : total_sales = sale_in_first_month + sale_in_second_month + X + sale_in_fourth_month + sale_in_fifth_month) :
  X = 9455 :=
by
  sorry

end NUMINAMATH_GPT_find_sale_in_third_month_l455_45521


namespace NUMINAMATH_GPT_Susan_ate_six_candies_l455_45572

def candy_consumption_weekly : Prop :=
  ∀ (candies_bought_Tue candies_bought_Wed candies_bought_Thu candies_bought_Fri : ℕ)
    (candies_left : ℕ) (total_spending : ℕ),
    candies_bought_Tue = 3 →
    candies_bought_Wed = 0 →
    candies_bought_Thu = 5 →
    candies_bought_Fri = 2 →
    candies_left = 4 →
    total_spending = 9 →
    candies_bought_Tue + candies_bought_Wed + candies_bought_Thu + candies_bought_Fri - candies_left = 6

theorem Susan_ate_six_candies : candy_consumption_weekly :=
by {
  -- The proof will be filled in later
  sorry
}

end NUMINAMATH_GPT_Susan_ate_six_candies_l455_45572


namespace NUMINAMATH_GPT_grassy_width_excluding_path_l455_45574

theorem grassy_width_excluding_path
  (l : ℝ) (w : ℝ) (p : ℝ)
  (h1: l = 110) (h2: w = 65) (h3: p = 2.5) :
  w - 2 * p = 60 :=
by
  sorry

end NUMINAMATH_GPT_grassy_width_excluding_path_l455_45574


namespace NUMINAMATH_GPT_infinite_series_value_l455_45526

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 4 * n^2 + 8 * n + 8) / (3^n * (n^3 + 5)) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_infinite_series_value_l455_45526


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l455_45553

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) 
    (h1 : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 81)
    (h2 : a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 171) : 
    ∃ d, d = 10 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l455_45553


namespace NUMINAMATH_GPT_probability_all_three_dice_twenty_l455_45564

theorem probability_all_three_dice_twenty (d1 d2 d3 d4 d5 : ℕ)
  (h1 : 1 ≤ d1 ∧ d1 ≤ 20) (h2 : 1 ≤ d2 ∧ d2 ≤ 20) (h3 : 1 ≤ d3 ∧ d3 ≤ 20)
  (h4 : 1 ≤ d4 ∧ d4 ≤ 20) (h5 : 1 ≤ d5 ∧ d5 ≤ 20)
  (h6 : d1 = 20) (h7 : d2 = 19)
  (h8 : (if d1 = 20 then 1 else 0) + (if d2 = 20 then 1 else 0) +
        (if d3 = 20 then 1 else 0) + (if d4 = 20 then 1 else 0) +
        (if d5 = 20 then 1 else 0) ≥ 3) :
  (1 / 58 : ℚ) = (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) /
                 ((if d3 = 20 ∧ d4 = 20 then 19 else 0) +
                  (if d3 = 20 ∧ d5 = 20 then 19 else 0) +
                  (if d4 = 20 ∧ d5 = 20 then 19 else 0) + 
                  (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_all_three_dice_twenty_l455_45564


namespace NUMINAMATH_GPT_smallest_m_l455_45533

noncomputable def fractional_part (x : ℝ) : ℝ :=
  x - ⌊x⌋

noncomputable def f (x : ℝ) : ℝ :=
  abs (3 * fractional_part x - 1.5)

theorem smallest_m (m : ℤ) (h1 : ∀ x : ℝ, m^2 * f (x * f x) = x → True) : ∃ m, m = 8 :=
by
  have h2 : ∀ m : ℤ, (∃ (s : ℕ), s ≥ 1008 ∧ (m^2 * abs (3 * fractional_part (s * abs (1.5 - 3 * (fractional_part s) )) - 1.5) = s)) → m = 8
  {
    sorry
  }
  sorry

end NUMINAMATH_GPT_smallest_m_l455_45533


namespace NUMINAMATH_GPT_tina_more_than_katya_l455_45569

-- Define the number of glasses sold by Katya, Ricky, and the condition for Tina's sales
def katya_sales : ℕ := 8
def ricky_sales : ℕ := 9

def combined_sales : ℕ := katya_sales + ricky_sales
def tina_sales : ℕ := 2 * combined_sales

-- Define the theorem to prove that Tina sold 26 more glasses than Katya
theorem tina_more_than_katya : tina_sales = katya_sales + 26 := by
  sorry

end NUMINAMATH_GPT_tina_more_than_katya_l455_45569


namespace NUMINAMATH_GPT_int_999_column_is_C_l455_45536

def column_of_int (n : ℕ) : String :=
  let m := n - 2
  match (m / 7 % 2, m % 7) with
  | (0, 0) => "A"
  | (0, 1) => "B"
  | (0, 2) => "C"
  | (0, 3) => "D"
  | (0, 4) => "E"
  | (0, 5) => "F"
  | (0, 6) => "G"
  | (1, 0) => "G"
  | (1, 1) => "F"
  | (1, 2) => "E"
  | (1, 3) => "D"
  | (1, 4) => "C"
  | (1, 5) => "B"
  | (1, 6) => "A"
  | _      => "Invalid"

theorem int_999_column_is_C : column_of_int 999 = "C" := by
  sorry

end NUMINAMATH_GPT_int_999_column_is_C_l455_45536


namespace NUMINAMATH_GPT_erasers_given_l455_45541

theorem erasers_given (initial final : ℕ) (h1 : initial = 8) (h2 : final = 11) : (final - initial = 3) :=
by
  sorry

end NUMINAMATH_GPT_erasers_given_l455_45541


namespace NUMINAMATH_GPT_find_angle_B_l455_45501

theorem find_angle_B 
  (A B : ℝ)
  (h1 : B + A = 90)
  (h2 : B = 4 * A) : 
  B = 144 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_B_l455_45501


namespace NUMINAMATH_GPT_minimum_value_of_sum_l455_45548

noncomputable def left_focus (a b c : ℝ) : ℝ := -c 

noncomputable def right_focus (a b c : ℝ) : ℝ := c

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
magnitude (q.1 - p.1, q.2 - p.2)

def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1

def P_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_eq P.1 P.2

theorem minimum_value_of_sum (P : ℝ × ℝ) (A : ℝ × ℝ) (F F' : ℝ × ℝ) (a b c : ℝ)
  (h1 : F = (-c, 0)) (h2 : F' = (c, 0)) (h3 : A = (1, 4)) (h4 : 2 * a = 4)
  (h5 : c^2 = a^2 + b^2) (h6 : P_on_hyperbola P) :
  (|distance P F| + |distance P A|) ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_sum_l455_45548


namespace NUMINAMATH_GPT_pigeons_among_non_sparrows_l455_45566

theorem pigeons_among_non_sparrows (P_total P_parrots P_peacocks P_sparrows : ℝ)
    (h1 : P_total = 20)
    (h2 : P_parrots = 30)
    (h3 : P_peacocks = 15)
    (h4 : P_sparrows = 35) :
    (P_total / (100 - P_sparrows)) * 100 = 30.77 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_pigeons_among_non_sparrows_l455_45566


namespace NUMINAMATH_GPT_bailey_discount_l455_45546

noncomputable def discount_percentage (total_cost_without_discount amount_spent : ℝ) : ℝ :=
  ((total_cost_without_discount - amount_spent) / total_cost_without_discount) * 100

theorem bailey_discount :
  let guest_sets := 2
  let master_sets := 4
  let price_guest := 40
  let price_master := 50
  let amount_spent := 224
  let total_cost_without_discount := (guest_sets * price_guest) + (master_sets * price_master)
  discount_percentage total_cost_without_discount amount_spent = 20 := 
by
  sorry

end NUMINAMATH_GPT_bailey_discount_l455_45546


namespace NUMINAMATH_GPT_complement_intersection_eq_l455_45554

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_eq :
  U \ (A ∩ B) = {1, 4, 5} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_eq_l455_45554


namespace NUMINAMATH_GPT_thomas_friends_fraction_l455_45585

noncomputable def fraction_of_bars_taken (x : ℝ) (initial_bars : ℝ) (returned_bars : ℝ) 
  (piper_bars : ℝ) (remaining_bars : ℝ) : ℝ :=
  x / initial_bars

theorem thomas_friends_fraction 
  (initial_bars : ℝ)
  (total_taken_by_all : ℝ)
  (returned_bars : ℝ)
  (piper_bars : ℝ)
  (remaining_bars : ℝ)
  (h_initial : initial_bars = 200)
  (h_remaining : remaining_bars = 110)
  (h_taken : 200 - 110 = 90)
  (h_total_taken_by_all : total_taken_by_all = 90)
  (h_returned : returned_bars = 5)
  (h_x_calculation : 2 * (total_taken_by_all + returned_bars - initial_bars) + initial_bars = total_taken_by_all + returned_bars)
  : fraction_of_bars_taken ((total_taken_by_all + returned_bars - initial_bars) + 2 * initial_bars) initial_bars returned_bars piper_bars remaining_bars = 21 / 80 :=
  sorry

end NUMINAMATH_GPT_thomas_friends_fraction_l455_45585


namespace NUMINAMATH_GPT_initial_fish_count_l455_45535

-- Definitions based on the given conditions
def Fish_given : ℝ := 22.0
def Fish_now : ℝ := 25.0

-- The goal is to prove the initial number of fish Mrs. Sheridan had.
theorem initial_fish_count : (Fish_given + Fish_now) = 47.0 := by
  sorry

end NUMINAMATH_GPT_initial_fish_count_l455_45535


namespace NUMINAMATH_GPT_xy_sum_square_l455_45551

theorem xy_sum_square (x y : ℕ) (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end NUMINAMATH_GPT_xy_sum_square_l455_45551


namespace NUMINAMATH_GPT_TotalMarks_l455_45528

def AmayaMarks (Arts Maths Music SocialStudies : ℕ) : Prop :=
  Maths = Arts - 20 ∧
  Maths = (9 * Arts) / 10 ∧
  Music = 70 ∧
  Music + 10 = SocialStudies

theorem TotalMarks (Arts Maths Music SocialStudies : ℕ) : 
  AmayaMarks Arts Maths Music SocialStudies → 
  (Arts + Maths + Music + SocialStudies = 530) :=
by
  sorry

end NUMINAMATH_GPT_TotalMarks_l455_45528


namespace NUMINAMATH_GPT_cost_of_popsicle_sticks_l455_45525

theorem cost_of_popsicle_sticks
  (total_money : ℕ)
  (cost_of_molds : ℕ)
  (cost_per_bottle : ℕ)
  (popsicles_per_bottle : ℕ)
  (sticks_used : ℕ)
  (sticks_left : ℕ)
  (number_of_sticks : ℕ)
  (remaining_money : ℕ) :
  total_money = 10 →
  cost_of_molds = 3 →
  cost_per_bottle = 2 →
  popsicles_per_bottle = 20 →
  sticks_left = 40 →
  number_of_sticks = 100 →
  remaining_money = total_money - cost_of_molds - (sticks_used / popsicles_per_bottle * cost_per_bottle) →
  sticks_used = number_of_sticks - sticks_left →
  remaining_money = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_cost_of_popsicle_sticks_l455_45525


namespace NUMINAMATH_GPT_remaining_pictures_l455_45539

theorem remaining_pictures (k m : ℕ) (d1 := 9 * k + 4) (d2 := 9 * m + 6) :
  (d1 * d2) % 9 = 6 → 9 - (d1 * d2 % 9) = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_remaining_pictures_l455_45539


namespace NUMINAMATH_GPT_sqrt_square_eq_14_l455_45504

theorem sqrt_square_eq_14 : Real.sqrt (14 ^ 2) = 14 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_square_eq_14_l455_45504


namespace NUMINAMATH_GPT_melanie_total_dimes_l455_45556

theorem melanie_total_dimes (d_1 d_2 d_3 : ℕ) (h₁ : d_1 = 19) (h₂ : d_2 = 39) (h₃ : d_3 = 25) : d_1 + d_2 + d_3 = 83 := by
  sorry

end NUMINAMATH_GPT_melanie_total_dimes_l455_45556


namespace NUMINAMATH_GPT_greatest_possible_x_l455_45503

theorem greatest_possible_x (x : ℕ) (h : x^3 < 15) : x ≤ 2 := by
  sorry

end NUMINAMATH_GPT_greatest_possible_x_l455_45503


namespace NUMINAMATH_GPT_find_angle_C_find_a_and_b_l455_45555

-- Conditions from the problem
variables {A B C : ℝ} {a b c : ℝ}
variables {m n : ℝ × ℝ}
variables (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
variables (h2 : n = (a - Real.sqrt 3 * b, b + c))
variables (h3 : m.1 * n.1 + m.2 * n.2 = 0)
variables (h4 : ∀ θ ∈ Set.Ioo 0 Real.pi, θ ≠ C → Real.cos θ = (a^2 + b^2 - c^2) / (2 * a * b))

-- Hypotheses for part (2)
variables (circumradius : ℝ) (area : ℝ)
variables (h5 : circumradius = 2)
variables (h6 : area = Real.sqrt 3)
variables (h7 : a > b)

-- Theorem statement for part (1)
theorem find_angle_C (h1 : m = (Real.sin A, Real.sin B - Real.sin C))
  (h2 : n = (a - Real.sqrt 3 * b, b + c))
  (h3 : m.1 * n.1 + m.2 * n.2 = 0)
  (h4 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) : 
  C = Real.pi / 6 := sorry

-- Theorem statement for part (2)
theorem find_a_and_b (circumradius : ℝ) (area : ℝ) (a b : ℝ)
  (h5 : circumradius = 2) (h6 : area = Real.sqrt 3) (h7 : a > b)
  (h8 : ∀ C ∈ Set.Ioo 0 Real.pi, Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b))
  (h9 : Real.sin C ≠ 0): 
  a = 2 * Real.sqrt 3 ∧ b = 2 := sorry

end NUMINAMATH_GPT_find_angle_C_find_a_and_b_l455_45555


namespace NUMINAMATH_GPT_opposite_blue_face_is_white_l455_45584

-- Define colors
inductive Color
| Red
| Blue
| Orange
| Purple
| Green
| Yellow
| White

-- Define the positions of colors on the cube
structure CubeConfig :=
(top : Color)
(front : Color)
(bottom : Color)
(back : Color)
(left : Color)
(right : Color)

-- The given conditions
def cube_conditions (c : CubeConfig) : Prop :=
  c.top = Color.Purple ∧
  c.front = Color.Green ∧
  c.bottom = Color.Yellow ∧
  c.back = Color.Orange ∧
  c.left = Color.Blue ∧
  c.right = Color.White

-- The statement we need to prove
theorem opposite_blue_face_is_white (c : CubeConfig) (h : cube_conditions c) :
  c.right = Color.White :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_opposite_blue_face_is_white_l455_45584


namespace NUMINAMATH_GPT_min_value_of_inverse_sum_l455_45577

noncomputable def minimumValue (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) : ℝ :=
  9 + 6 * Real.sqrt 2

theorem min_value_of_inverse_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 / 3 ∧ (1/x + 1/y) = 9 + 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_min_value_of_inverse_sum_l455_45577


namespace NUMINAMATH_GPT_determine_n_l455_45545

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem determine_n (n : ℕ) (h1 : binom n 2 + binom n 1 = 6) : n = 3 := 
by
  sorry

end NUMINAMATH_GPT_determine_n_l455_45545


namespace NUMINAMATH_GPT_area_of_given_triangle_l455_45522

def point := ℝ × ℝ

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_given_triangle : 
  triangle_area (1, 1) (7, 1) (5, 3) = 6 :=
by
  -- the proof should go here
  sorry

end NUMINAMATH_GPT_area_of_given_triangle_l455_45522


namespace NUMINAMATH_GPT_correct_statements_l455_45578

theorem correct_statements : 
    let statement1 := "The regression effect is characterized by the relevant exponent R^{2}. The larger the R^{2}, the better the fitting effect."
    let statement2 := "The properties of a sphere are inferred from the properties of a circle by analogy."
    let statement3 := "Any two complex numbers cannot be compared in size."
    let statement4 := "Flowcharts are often used to represent some dynamic processes, usually with a 'starting point' and an 'ending point'."
    true -> (statement1 = "correct" ∧ statement2 = "correct" ∧ statement3 = "incorrect" ∧ statement4 = "incorrect") :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_correct_statements_l455_45578


namespace NUMINAMATH_GPT_max_consecutive_sum_l455_45519

theorem max_consecutive_sum (n : ℕ) : 
  (∀ (n : ℕ), (n*(n + 1))/2 ≤ 400 → n ≤ 27) ∧ ((27*(27 + 1))/2 ≤ 400) :=
by
  sorry

end NUMINAMATH_GPT_max_consecutive_sum_l455_45519


namespace NUMINAMATH_GPT_oj_fraction_is_11_over_30_l455_45560

-- Define the capacity of each pitcher
def pitcher_capacity : ℕ := 600

-- Define the fraction of orange juice in each pitcher
def fraction_oj_pitcher1 : ℚ := 1 / 3
def fraction_oj_pitcher2 : ℚ := 2 / 5

-- Define the amount of orange juice in each pitcher
def oj_amount_pitcher1 := pitcher_capacity * fraction_oj_pitcher1
def oj_amount_pitcher2 := pitcher_capacity * fraction_oj_pitcher2

-- Define the total amount of orange juice after both pitchers are poured into the large container
def total_oj_amount := oj_amount_pitcher1 + oj_amount_pitcher2

-- Define the total volume of the mixture in the large container
def total_mixture_volume := 2 * pitcher_capacity

-- Define the fraction of the mixture that is orange juice
def oj_fraction_in_mixture := total_oj_amount / total_mixture_volume

-- Prove that the fraction of the mixture that is orange juice is 11/30
theorem oj_fraction_is_11_over_30 : oj_fraction_in_mixture = 11 / 30 := by
  sorry

end NUMINAMATH_GPT_oj_fraction_is_11_over_30_l455_45560


namespace NUMINAMATH_GPT_piggy_bank_exceed_five_dollars_l455_45559

noncomputable def sequence_sum (n : ℕ) : ℕ := 2^n - 1

theorem piggy_bank_exceed_five_dollars (n : ℕ) (start_day : Nat) (day_of_week : Fin 7) :
  ∃ (n : ℕ), sequence_sum n > 500 ∧ n = 9 ∧ (start_day + n) % 7 = 2 := 
sorry

end NUMINAMATH_GPT_piggy_bank_exceed_five_dollars_l455_45559


namespace NUMINAMATH_GPT_platform_length_is_correct_l455_45502

noncomputable def length_of_platform (T : ℕ) (t_p t_s : ℕ) : ℕ :=
  let speed_of_train := T / t_s
  let distance_when_crossing_platform := speed_of_train * t_p
  distance_when_crossing_platform - T

theorem platform_length_is_correct :
  ∀ (T t_p t_s : ℕ),
  T = 300 → t_p = 33 → t_s = 18 →
  length_of_platform T t_p t_s = 250 :=
by
  intros T t_p t_s hT ht_p ht_s
  simp [length_of_platform, hT, ht_p, ht_s]
  sorry

end NUMINAMATH_GPT_platform_length_is_correct_l455_45502


namespace NUMINAMATH_GPT_ratio_d_s_proof_l455_45530

noncomputable def ratio_d_s (n : ℕ) (s d : ℝ) : ℝ :=
  d / s

theorem ratio_d_s_proof : ∀ (n : ℕ) (s d : ℝ), 
  (n = 30) → 
  ((n ^ 2 * s ^ 2) / (n * s + 2 * n * d) ^ 2 = 0.81) → 
  ratio_d_s n s d = 1 / 18 :=
by
  intros n s d h_n h_area
  sorry

end NUMINAMATH_GPT_ratio_d_s_proof_l455_45530


namespace NUMINAMATH_GPT_roots_of_polynomial_l455_45581

noncomputable def poly (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem roots_of_polynomial :
  ∀ x : ℝ, poly x = 0 ↔ (x = -1 ∨ x = 1 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l455_45581


namespace NUMINAMATH_GPT_solve_for_x_l455_45537

theorem solve_for_x : ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l455_45537


namespace NUMINAMATH_GPT_polynomial_form_l455_45511

noncomputable def polynomial_solution (P : ℝ → ℝ) :=
  ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))

theorem polynomial_form :
  ∀ (P : ℝ → ℝ), polynomial_solution P ↔ ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x^2 + b * x^4 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_form_l455_45511


namespace NUMINAMATH_GPT_total_triangles_in_figure_l455_45583

theorem total_triangles_in_figure :
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  small_triangles + two_small_comb + three_small_comb + all_small_comb = 11 :=
by
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  show small_triangles + two_small_comb + three_small_comb + all_small_comb = 11
  sorry

end NUMINAMATH_GPT_total_triangles_in_figure_l455_45583


namespace NUMINAMATH_GPT_max_earnings_mary_l455_45588

def wage_rate : ℝ := 8
def first_hours : ℕ := 20
def max_hours : ℕ := 80
def regular_tip_rate : ℝ := 2
def overtime_rate_increase : ℝ := 1.25
def overtime_tip_rate : ℝ := 3
def overtime_bonus_threshold : ℕ := 5
def overtime_bonus_amount : ℝ := 20

noncomputable def total_earnings (hours : ℕ) : ℝ :=
  let regular_hours := min hours first_hours
  let overtime_hours := if hours > first_hours then hours - first_hours else 0
  let overtime_blocks := overtime_hours / overtime_bonus_threshold
  let regular_earnings := regular_hours * (wage_rate + regular_tip_rate)
  let overtime_earnings := overtime_hours * (wage_rate * overtime_rate_increase + overtime_tip_rate)
  let bonuses := (overtime_blocks) * overtime_bonus_amount
  regular_earnings + overtime_earnings + bonuses

theorem max_earnings_mary : total_earnings max_hours = 1220 := by
  sorry

end NUMINAMATH_GPT_max_earnings_mary_l455_45588


namespace NUMINAMATH_GPT_team_X_played_24_games_l455_45582

def games_played_X (x : ℕ) : ℕ := x
def games_played_Y (x : ℕ) : ℕ := x + 9
def games_won_X (x : ℕ) : ℚ := 3 / 4 * x
def games_won_Y (x : ℕ) : ℚ := 2 / 3 * (x + 9)

theorem team_X_played_24_games (x : ℕ) 
  (h1 : games_won_Y x = games_won_X x + 4) : games_played_X x = 24 :=
by
  sorry

end NUMINAMATH_GPT_team_X_played_24_games_l455_45582


namespace NUMINAMATH_GPT_smallest_positive_integer_l455_45529

theorem smallest_positive_integer
    (n : ℕ)
    (h : ∀ (a : Fin n → ℤ), ∃ (i j : Fin n), i ≠ j ∧ (2009 ∣ (a i + a j) ∨ 2009 ∣ (a i - a j))) : n = 1006 := by
  -- Proof is required here
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l455_45529


namespace NUMINAMATH_GPT_rectangle_area_l455_45571

theorem rectangle_area (length width : ℝ) 
  (h1 : width = 0.9 * length) 
  (h2 : length = 15) : 
  length * width = 202.5 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l455_45571


namespace NUMINAMATH_GPT_zachary_more_pushups_l455_45599

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := 44

theorem zachary_more_pushups : zachary_pushups - david_pushups = 7 := by
  sorry

end NUMINAMATH_GPT_zachary_more_pushups_l455_45599


namespace NUMINAMATH_GPT_gcd_117_182_l455_45512

theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_117_182_l455_45512


namespace NUMINAMATH_GPT_tenth_term_is_correct_l455_45544

-- Define the first term and common difference for the sequence
def a1 : ℚ := 1 / 2
def d : ℚ := 1 / 3

-- The property that defines the n-th term of the arithmetic sequence
def a (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Statement to prove that the tenth term in the arithmetic sequence is 7 / 2
theorem tenth_term_is_correct : a 10 = 7 / 2 := 
by 
  -- To be filled in with the proof later
  sorry

end NUMINAMATH_GPT_tenth_term_is_correct_l455_45544


namespace NUMINAMATH_GPT_probability_neither_red_blue_purple_l455_45531

def total_balls : ℕ := 240
def white_balls : ℕ := 60
def green_balls : ℕ := 70
def yellow_balls : ℕ := 45
def red_balls : ℕ := 35
def blue_balls : ℕ := 20
def purple_balls : ℕ := 10

theorem probability_neither_red_blue_purple :
  (total_balls - (red_balls + blue_balls + purple_balls)) / total_balls = 35 / 48 := 
by 
  /- Proof details are not necessary -/
  sorry

end NUMINAMATH_GPT_probability_neither_red_blue_purple_l455_45531


namespace NUMINAMATH_GPT_cube_faces_opposite_10_is_8_l455_45575

theorem cube_faces_opposite_10_is_8 (nums : Finset ℕ) (h_nums : nums = {6, 7, 8, 9, 10, 11})
  (sum_lateral_first : ℕ) (h_sum_lateral_first : sum_lateral_first = 36)
  (sum_lateral_second : ℕ) (h_sum_lateral_second : sum_lateral_second = 33)
  (faces_opposite_10 : ℕ) (h_faces_opposite_10 : faces_opposite_10 ∈ nums) :
  faces_opposite_10 = 8 :=
by
  sorry

end NUMINAMATH_GPT_cube_faces_opposite_10_is_8_l455_45575


namespace NUMINAMATH_GPT_strawberry_cake_cost_proof_l455_45540

-- Define the constants
def chocolate_cakes : ℕ := 3
def price_per_chocolate_cake : ℕ := 12
def total_bill : ℕ := 168
def number_of_strawberry_cakes : ℕ := 6

-- Define the calculation for the total cost of chocolate cakes
def total_cost_of_chocolate_cakes : ℕ := chocolate_cakes * price_per_chocolate_cake

-- Define the remaining cost for strawberry cakes
def remaining_cost : ℕ := total_bill - total_cost_of_chocolate_cakes

-- Prove the cost per strawberry cake
def cost_per_strawberry_cake : ℕ := remaining_cost / number_of_strawberry_cakes

theorem strawberry_cake_cost_proof : cost_per_strawberry_cake = 22 := by
  -- We skip the proof here. Detailed proof steps would go in the place of sorry
  sorry

end NUMINAMATH_GPT_strawberry_cake_cost_proof_l455_45540


namespace NUMINAMATH_GPT_find_a3_l455_45509

theorem find_a3 (a : ℕ → ℕ) (h₁ : a 1 = 2)
  (h₂ : ∀ n, (1 + 2 * a (n + 1)) = (1 + 2 * a n) + 1) : a 3 = 3 :=
by
  -- This is where the proof would go, but we'll leave it as sorry for now.
  sorry

end NUMINAMATH_GPT_find_a3_l455_45509


namespace NUMINAMATH_GPT_quadratic_solution_1_quadratic_solution_2_l455_45552

theorem quadratic_solution_1 (x : ℝ) : x^2 - 8 * x + 12 = 0 ↔ x = 2 ∨ x = 6 := 
by
  sorry

theorem quadratic_solution_2 (x : ℝ) : (x - 3)^2 = 2 * x * (x - 3) ↔ x = 3 ∨ x = -3 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_1_quadratic_solution_2_l455_45552


namespace NUMINAMATH_GPT_number_of_nickels_l455_45562

def dimes : ℕ := 10
def pennies_per_dime : ℕ := 10
def pennies_per_nickel : ℕ := 5
def total_pennies : ℕ := 150

theorem number_of_nickels (total_value_dimes : ℕ := dimes * pennies_per_dime)
  (pennies_needed_from_nickels : ℕ := total_pennies - total_value_dimes)
  (n : ℕ) : n = pennies_needed_from_nickels / pennies_per_nickel → n = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_nickels_l455_45562


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l455_45595

noncomputable def tan_sum_formula (α : ℝ) : ℝ :=
  (Real.tan α + Real.tan (Real.pi / 4)) / (1 - Real.tan α * Real.tan (Real.pi / 4))

theorem tan_alpha_plus_pi_over_4 
  (α : ℝ) 
  (h1 : Real.cos (2 * α) + Real.sin α * (2 * Real.sin α - 1) = 2 / 5) 
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  tan_sum_formula α = 1 / 7 := 
sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_over_4_l455_45595


namespace NUMINAMATH_GPT_shaded_quilt_fraction_l455_45570

-- Define the basic structure of the problem using conditions from step a

def is_unit_square (s : ℕ) : Prop := s = 1

def grid_size : ℕ := 4
def total_squares : ℕ := grid_size * grid_size

def shaded_squares : ℕ := 2
def half_shaded_squares : ℕ := 4

def fraction_shaded (shaded: ℕ) (total: ℕ) : ℚ := shaded / total

theorem shaded_quilt_fraction :
  fraction_shaded (shaded_squares + half_shaded_squares / 2) total_squares = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_shaded_quilt_fraction_l455_45570


namespace NUMINAMATH_GPT_number_of_alligators_l455_45589

theorem number_of_alligators (A : ℕ) 
  (num_snakes : ℕ := 18) 
  (total_eyes : ℕ := 56) 
  (eyes_per_snake : ℕ := 2) 
  (eyes_per_alligator : ℕ := 2) 
  (snakes_eyes : ℕ := num_snakes * eyes_per_snake) 
  (alligators_eyes : ℕ := A * eyes_per_alligator) 
  (total_animals_eyes : ℕ := snakes_eyes + alligators_eyes) 
  (total_eyes_eq : total_animals_eyes = total_eyes) 
: A = 10 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_alligators_l455_45589


namespace NUMINAMATH_GPT_maximum_area_of_enclosed_poly_l455_45597

theorem maximum_area_of_enclosed_poly (k : ℕ) : 
  ∃ (A : ℕ), (A = 4 * k + 1) :=
sorry

end NUMINAMATH_GPT_maximum_area_of_enclosed_poly_l455_45597


namespace NUMINAMATH_GPT_audi_crossing_intersection_between_17_and_18_l455_45508

-- Given conditions:
-- Two cars, an Audi and a BMW, are moving along two intersecting roads at equal constant speeds.
-- At both 17:00 and 18:00, the BMW was twice as far from the intersection as the Audi.
-- Let the distance of Audi from the intersection at 17:00 be x and BMW's distance be 2x.
-- Both vehicles travel at a constant speed v.

noncomputable def car_position (initial_distance : ℝ) (velocity : ℝ) (time_elapsed : ℝ) : ℝ :=
  initial_distance + velocity * time_elapsed

theorem audi_crossing_intersection_between_17_and_18 (x v : ℝ) :
  ∃ t : ℝ, (t = 15 ∨ t = 45) ∧
    car_position x (-v) (t/60) = 0 ∧ car_position (2 * x) (-v) (t/60) = 2 * car_position x (-v) (1 - t/60) :=
sorry

end NUMINAMATH_GPT_audi_crossing_intersection_between_17_and_18_l455_45508


namespace NUMINAMATH_GPT_burrito_calories_l455_45516

theorem burrito_calories :
  ∀ (C : ℕ), 
  (10 * C = 6 * (250 - 50)) →
  C = 120 :=
by
  intros C h
  sorry

end NUMINAMATH_GPT_burrito_calories_l455_45516


namespace NUMINAMATH_GPT_candidate_lost_by_2340_votes_l455_45550

theorem candidate_lost_by_2340_votes
  (total_votes : ℝ)
  (candidate_percentage : ℝ)
  (rival_percentage : ℝ)
  (candidate_votes : ℝ)
  (rival_votes : ℝ)
  (votes_difference : ℝ)
  (h1 : total_votes = 7800)
  (h2 : candidate_percentage = 0.35)
  (h3 : rival_percentage = 0.65)
  (h4 : candidate_votes = candidate_percentage * total_votes)
  (h5 : rival_votes = rival_percentage * total_votes)
  (h6 : votes_difference = rival_votes - candidate_votes) :
  votes_difference = 2340 :=
by
  sorry

end NUMINAMATH_GPT_candidate_lost_by_2340_votes_l455_45550


namespace NUMINAMATH_GPT_minimum_positive_Sn_l455_45586

theorem minimum_positive_Sn (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ n, a (n+1) = a n + d) →
  a 11 / a 10 < -1 →
  (∃ N, ∀ n > N, S n < S (n + 1) ∧ S 1 ≤ S n ∧ ∀ n > N, S n < 0) →
  S 19 > 0 ∧ ∀ k < 19, S k > S 19 → S 19 < 0 →
  n = 19 :=
by
  sorry

end NUMINAMATH_GPT_minimum_positive_Sn_l455_45586


namespace NUMINAMATH_GPT_minimum_inverse_sum_l455_45576

theorem minimum_inverse_sum (a b : ℝ) (h1 : (a > 0) ∧ (b > 0)) 
  (h2 : 3 * a + 4 * b = 55) : 
  (1 / a) + (1 / b) ≥ (7 + 4 * Real.sqrt 3) / 55 :=
sorry

end NUMINAMATH_GPT_minimum_inverse_sum_l455_45576


namespace NUMINAMATH_GPT_smallest_possible_value_m_l455_45518

theorem smallest_possible_value_m (r y b : ℕ) (h : 16 * r = 18 * y ∧ 18 * y = 20 * b) : 
  ∃ m : ℕ, 30 * m = 16 * r ∧ 30 * m = 720 ∧ m = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_possible_value_m_l455_45518


namespace NUMINAMATH_GPT_pentagon_rectangle_ratio_l455_45527

theorem pentagon_rectangle_ratio :
  let p : ℝ := 60  -- Perimeter of both the pentagon and the rectangle
  let length_side_pentagon : ℝ := 12
  let w : ℝ := 10
  p / 5 = length_side_pentagon ∧ p/6 = w ∧ length_side_pentagon / w = 6/5 :=
sorry

end NUMINAMATH_GPT_pentagon_rectangle_ratio_l455_45527


namespace NUMINAMATH_GPT_total_time_spent_l455_45591

-- Define the conditions
def t1 : ℝ := 2.5
def t2 : ℝ := 3 * t1

-- Define the theorem to prove
theorem total_time_spent : t1 + t2 = 10 := by
  sorry

end NUMINAMATH_GPT_total_time_spent_l455_45591


namespace NUMINAMATH_GPT_abc_sub_c_minus_2023_eq_2023_l455_45561

theorem abc_sub_c_minus_2023_eq_2023 (a b c : ℝ) (h : a * b = 1) : 
  a * b * c - (c - 2023) = 2023 := 
by sorry

end NUMINAMATH_GPT_abc_sub_c_minus_2023_eq_2023_l455_45561


namespace NUMINAMATH_GPT_percent_c_of_b_l455_45594

variable (a b c : ℝ)

theorem percent_c_of_b (h1 : c = 0.20 * a) (h2 : b = 2 * a) : 
  ∃ x : ℝ, c = (x / 100) * b ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_percent_c_of_b_l455_45594


namespace NUMINAMATH_GPT_number_of_ants_proof_l455_45514

-- Define the conditions
def width_ft := 500
def length_ft := 600
def ants_per_sq_inch := 4
def inches_per_foot := 12

-- Define the calculation to get the number of ants
def number_of_ants (width_ft : ℕ) (length_ft : ℕ) (ants_per_sq_inch : ℕ) (inches_per_foot : ℕ) :=
  let width_inch := width_ft * inches_per_foot
  let length_inch := length_ft * inches_per_foot
  let area_sq_inch := width_inch * length_inch
  ants_per_sq_inch * area_sq_inch

-- Prove the number of ants is approximately 173 million
theorem number_of_ants_proof :
  number_of_ants width_ft length_ft ants_per_sq_inch inches_per_foot = 172800000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ants_proof_l455_45514


namespace NUMINAMATH_GPT_min_C2_D2_at_36_l455_45534

noncomputable def min_value_C2_D2 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : ℝ :=
  let C := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
  let D := (Real.sqrt (x + 1) + Real.sqrt (y + 2) + Real.sqrt (z + 3))
  C^2 - D^2

theorem min_C2_D2_at_36 (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 3) : 
  min_value_C2_D2 x y z hx hy hz = 36 :=
sorry

end NUMINAMATH_GPT_min_C2_D2_at_36_l455_45534


namespace NUMINAMATH_GPT_iterative_average_difference_l455_45592

theorem iterative_average_difference :
  let numbers : List ℕ := [2, 4, 6, 8, 10] 
  let avg2 (a b : ℝ) := (a + b) / 2
  let avg (init : ℝ) (lst : List ℕ) := lst.foldl (λ acc x => avg2 acc x) init
  let max_avg := avg 2 [4, 6, 8, 10]
  let min_avg := avg 10 [8, 6, 4, 2] 
  max_avg - min_avg = 4.25 := 
by
  sorry

end NUMINAMATH_GPT_iterative_average_difference_l455_45592


namespace NUMINAMATH_GPT_misha_students_count_l455_45596

theorem misha_students_count :
  (∀ n : ℕ, n = 60 → (exists better worse : ℕ, better = n - 1 ∧  worse = n - 1)) →
  (∀ n : ℕ, n = 60 → (better + worse + 1 = 119)) :=
by
  sorry

end NUMINAMATH_GPT_misha_students_count_l455_45596


namespace NUMINAMATH_GPT_top_card_is_11_l455_45538

-- Define the initial configuration of cards
def initial_array : List (List Nat) := [
  [1, 2, 3, 4, 5, 6],
  [7, 8, 9, 10, 11, 12],
  [13, 14, 15, 16, 17, 18]
]

-- Perform the described sequence of folds
def fold1 (arr : List (List Nat)) : List (List Nat) := [
  [3, 4, 5, 6],
  [9, 10, 11, 12],
  [15, 16, 17, 18],
  [1, 2],
  [7, 8],
  [13, 14]
]

def fold2 (arr : List (List Nat)) : List (List Nat) := [
  [5, 6],
  [11, 12],
  [17, 18],
  [3, 4, 1, 2],
  [9, 10, 7, 8],
  [15, 16, 13, 14]
]

def fold3 (arr : List (List Nat)) : List (List Nat) := [
  [11, 12, 7, 8],
  [17, 18, 13, 14],
  [5, 6, 1, 2],
  [9, 10, 3, 4],
  [15, 16, 9, 10]
]

-- Define the final array after all the folds
def final_array := fold3 (fold2 (fold1 initial_array))

-- Statement to be proven
theorem top_card_is_11 : (final_array.head!.head!) = 11 := 
  by
    sorry -- Proof to be filled in

end NUMINAMATH_GPT_top_card_is_11_l455_45538


namespace NUMINAMATH_GPT_ratio_of_B_to_C_l455_45593

theorem ratio_of_B_to_C
  (A B C : ℕ) 
  (h1 : A = B + 2) 
  (h2 : A + B + C = 47) 
  (h3 : B = 18) : B / C = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_B_to_C_l455_45593


namespace NUMINAMATH_GPT_min_value_condition_l455_45532

variable (a b : ℝ)

theorem min_value_condition (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 4 * b^2 = 2) :
    (1 / a^2) + (1 / b^2) = 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_condition_l455_45532


namespace NUMINAMATH_GPT_dolly_dresses_shipment_l455_45543

variable (T : ℕ)

/-- Given that 70% of the total number of Dolly Dresses in the shipment is equal to 140,
    prove that the total number of Dolly Dresses in the shipment is 200. -/
theorem dolly_dresses_shipment (h : (7 * T) / 10 = 140) : T = 200 :=
sorry

end NUMINAMATH_GPT_dolly_dresses_shipment_l455_45543


namespace NUMINAMATH_GPT_remaining_wire_in_cm_l455_45547

theorem remaining_wire_in_cm (total_mm : ℝ) (per_mobile_mm : ℝ) (conversion_factor : ℝ) :
  total_mm = 117.6 →
  per_mobile_mm = 4 →
  conversion_factor = 10 →
  ((total_mm % per_mobile_mm) / conversion_factor) = 0.16 :=
by
  intros htotal hmobile hconv
  sorry

end NUMINAMATH_GPT_remaining_wire_in_cm_l455_45547


namespace NUMINAMATH_GPT_rational_expression_l455_45587

theorem rational_expression {x : ℚ} : (∃ a : ℚ, x / (x^2 + x + 1) = a) → (∃ b : ℚ, x^2 / (x^4 + x^2 + 1) = b) := by
  sorry

end NUMINAMATH_GPT_rational_expression_l455_45587


namespace NUMINAMATH_GPT_problem_1_problem_2_l455_45567

def op (x y : ℝ) : ℝ := 3 * x - y

theorem problem_1 (x : ℝ) : op x (op 2 3) = 1 ↔ x = 4 / 3 := by
  -- definitions from conditions
  let def_op_2_3 := op 2 3
  let eq1 := op x def_op_2_3
  -- problem in lean representation
  sorry

theorem problem_2 (x : ℝ) : op (x ^ 2) 2 = 10 ↔ x = 2 ∨ x = -2 := by
  -- problem in lean representation
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l455_45567


namespace NUMINAMATH_GPT_solve_inequality_l455_45515

theorem solve_inequality (a : ℝ) :
  (a < 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ a < x ∧ x < 1 - a) ∨
  (a > 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ 1 - a < x ∧ x < a) ∨
  (a = 1 / 2 ∧ ∀ x : ℝ, x^2 - x + a - a^2 < 0 ↔ false) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l455_45515
