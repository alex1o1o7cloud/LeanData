import Mathlib

namespace tan_diff_l792_79296

theorem tan_diff (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4/3) : Real.tan (α - β) = 1/3 := 
sorry

end tan_diff_l792_79296


namespace number_of_subsets_of_set_A_l792_79228

theorem number_of_subsets_of_set_A : 
  (setOfSubsets : Finset (Finset ℕ)) = Finset.powerset {2, 4, 5} → 
  setOfSubsets.card = 8 :=
by
  sorry

end number_of_subsets_of_set_A_l792_79228


namespace transmission_time_estimation_l792_79221

noncomputable def number_of_blocks := 80
noncomputable def chunks_per_block := 640
noncomputable def transmission_rate := 160 -- chunks per second
noncomputable def seconds_per_minute := 60
noncomputable def total_chunks := number_of_blocks * chunks_per_block
noncomputable def total_time_seconds := total_chunks / transmission_rate
noncomputable def total_time_minutes := total_time_seconds / seconds_per_minute

theorem transmission_time_estimation : total_time_minutes = 5 := 
  sorry

end transmission_time_estimation_l792_79221


namespace divide_rope_length_l792_79209

-- Definitions of variables based on the problem conditions
def rope_length : ℚ := 8 / 15
def num_parts : ℕ := 3

-- Theorem statement
theorem divide_rope_length :
  (1 / num_parts = (1 : ℚ) / 3) ∧ (rope_length * (1 / num_parts) = 8 / 45) :=
by
  sorry

end divide_rope_length_l792_79209


namespace intersection_A_B_l792_79264

open Set

variable (l : ℝ)

def A := {x : ℝ | x > l}
def B := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_A_B (h₁ : l = 1) :
  A l ∩ B = {x : ℝ | 1 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l792_79264


namespace original_rectangle_area_l792_79273

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l792_79273


namespace depth_of_melted_sauce_l792_79249

theorem depth_of_melted_sauce
  (r_sphere : ℝ) (r_cylinder : ℝ) (h_cylinder : ℝ) (volume_conserved : Bool) :
  r_sphere = 3 ∧ r_cylinder = 10 ∧ volume_conserved → h_cylinder = 9/25 :=
by
  -- Explanation of the condition: 
  -- r_sphere is the radius of the original spherical globe (3 inches)
  -- r_cylinder is the radius of the cylindrical puddle (10 inches)
  -- h_cylinder is the depth we need to prove is 9/25 inches
  -- volume_conserved indicates that the volume is conserved
  sorry

end depth_of_melted_sauce_l792_79249


namespace travel_times_l792_79281

variable (t v1 v2 : ℝ)

def conditions := 
  (v1 * 2 = v2 * t) ∧ 
  (v2 * 4.5 = v1 * t)

theorem travel_times (h : conditions t v1 v2) : 
  t = 3 ∧ 
  (t + 2 = 5) ∧ 
  (t + 4.5 = 7.5) := by
  sorry

end travel_times_l792_79281


namespace find_larger_number_l792_79286

theorem find_larger_number :
  ∃ x y : ℤ, x + y = 30 ∧ 2 * y - x = 6 ∧ x > y ∧ x = 18 :=
by
  sorry

end find_larger_number_l792_79286


namespace gcd_12345_6789_eq_3_l792_79260

theorem gcd_12345_6789_eq_3 : Int.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_eq_3_l792_79260


namespace intersection_eq_expected_l792_79239

def setA := { x : ℝ | 0 ≤ x ∧ x ≤ 3 }
def setB := { x : ℝ | 1 ≤ x ∧ x < 4 }
def expectedSet := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }

theorem intersection_eq_expected :
  {x : ℝ | x ∈ setA ∧ x ∈ setB} = expectedSet :=
by
  sorry

end intersection_eq_expected_l792_79239


namespace remainder_when_200_divided_by_k_l792_79235

theorem remainder_when_200_divided_by_k (k : ℕ) (hk_pos : 0 < k)
  (h₁ : 125 % (k^3) = 5) : 200 % k = 0 :=
sorry

end remainder_when_200_divided_by_k_l792_79235


namespace angle_relationship_l792_79256

open Real

variables (A B C D : Point)
variables (AB AC AD : ℝ)
variables (CAB DAC BDC DBC : ℝ)
variables (k : ℝ)

-- Given conditions
axiom h1 : AB = AC
axiom h2 : AC = AD
axiom h3 : DAC = k * CAB

-- Proof to be shown
theorem angle_relationship : DBC = k * BDC :=
  sorry

end angle_relationship_l792_79256


namespace part_a_part_b_l792_79201

def good (p q n : ℕ) : Prop :=
  ∃ x y : ℕ, n = p * x + q * y

def bad (p q n : ℕ) : Prop := 
  ¬ good p q n

theorem part_a (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ A, A = p * q - p - q ∧ ∀ x y, x + y = A → (good p q x ∧ bad p q y) ∨ (bad p q x ∧ good p q y) := by
  sorry

theorem part_b (p q : ℕ) (h : Nat.gcd p q = 1) : ∃ N, N = (p - 1) * (q - 1) / 2 ∧ ∀ n, n < p * q - p - q → bad p q n :=
  sorry

end part_a_part_b_l792_79201


namespace sum_of_digits_of_N_l792_79253

-- Define N
def N := 10^3 + 10^4 + 10^5 + 10^6 + 10^7 + 10^8 + 10^9

-- Define function to calculate sum of digits
def sum_of_digits(n: Nat) : Nat :=
  n.digits 10 |>.sum

-- Theorem statement
theorem sum_of_digits_of_N : sum_of_digits N = 7 :=
  sorry

end sum_of_digits_of_N_l792_79253


namespace determine_a_l792_79246

theorem determine_a 
(h : ∃x, x = -1 ∧ 2 * x ^ 2 + a * x - a ^ 2 = 0) : a = -2 ∨ a = 1 :=
by
  -- Proof omitted
  sorry

end determine_a_l792_79246


namespace number_of_boys_l792_79280

theorem number_of_boys (b g : ℕ) (h1: (3/5 : ℚ) * b = (5/6 : ℚ) * g) (h2: b + g = 30)
  (h3: g = (b * 18) / 25): b = 17 := by
  sorry

end number_of_boys_l792_79280


namespace minimum_value_of_w_l792_79288

noncomputable def w (x y : ℝ) : ℝ := 3 * x ^ 2 + 3 * y ^ 2 + 9 * x - 6 * y + 27

theorem minimum_value_of_w : (∃ x y : ℝ, w x y = 20.25) := sorry

end minimum_value_of_w_l792_79288


namespace A_finishes_remaining_work_in_2_days_l792_79225

/-- 
Given that A's daily work rate is 1/6 of the work and B's daily work rate is 1/15 of the work,
and B has already completed 2/3 of the work, 
prove that A can finish the remaining work in 2 days.
-/
theorem A_finishes_remaining_work_in_2_days :
  let A_work_rate := (1 : ℝ) / 6
  let B_work_rate := (1 : ℝ) / 15
  let B_work_in_10_days := (10 : ℝ) * B_work_rate
  let remaining_work := (1 : ℝ) - B_work_in_10_days
  let days_for_A := remaining_work / A_work_rate
  B_work_in_10_days = 2 / 3 → 
  remaining_work = 1 / 3 → 
  days_for_A = 2 :=
by
  sorry

end A_finishes_remaining_work_in_2_days_l792_79225


namespace find_bigger_number_l792_79270

noncomputable def common_factor (x : ℕ) : Prop :=
  8 * x + 3 * x = 143

theorem find_bigger_number (x : ℕ) (h : common_factor x) : 8 * x = 104 :=
by
  sorry

end find_bigger_number_l792_79270


namespace parallel_vectors_l792_79203

theorem parallel_vectors {m : ℝ} 
  (h : (2 * m + 1) / 2 = 3 / m): m = 3 / 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l792_79203


namespace geometric_progression_common_ratio_l792_79254

theorem geometric_progression_common_ratio (a r : ℝ) 
(h_pos: a > 0)
(h_condition: ∀ n : ℕ, a * r^(n-1) = (a * r^n + a * r^(n+1))^2):
  r = 0.618 :=
sorry

end geometric_progression_common_ratio_l792_79254


namespace car_distance_l792_79234

/-- A car takes 4 hours to cover a certain distance. We are given that the car should maintain a speed of 90 kmph to cover the same distance in (3/2) of the previous time (which is 6 hours). We need to prove that the distance the car needs to cover is 540 km. -/
theorem car_distance (time_initial : ℝ) (speed : ℝ) (time_new : ℝ) (distance : ℝ) 
  (h1 : time_initial = 4) 
  (h2 : speed = 90)
  (h3 : time_new = (3/2) * time_initial)
  (h4 : distance = speed * time_new) : 
  distance = 540 := 
sorry

end car_distance_l792_79234


namespace steve_total_money_l792_79206

theorem steve_total_money
    (nickels : ℕ)
    (dimes : ℕ)
    (nickel_value : ℕ := 5)
    (dime_value : ℕ := 10)
    (cond1 : nickels = 2)
    (cond2 : dimes = nickels + 4) 
    : (nickels * nickel_value + dimes * dime_value) = 70 := by
  sorry

end steve_total_money_l792_79206


namespace simplify_fraction_l792_79294

theorem simplify_fraction (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c ≠ 0) :
  (a^2 + a * b - b^2 + a * c) / (b^2 + b * c - c^2 + b * a) = (a - b) / (b - c) :=
by
  sorry

end simplify_fraction_l792_79294


namespace katelyn_sandwiches_difference_l792_79295

theorem katelyn_sandwiches_difference :
  ∃ (K : ℕ), K - 49 = 47 ∧ (49 + K + K / 4 = 169) := 
sorry

end katelyn_sandwiches_difference_l792_79295


namespace field_length_l792_79259

theorem field_length (w l : ℝ) (A_f A_p : ℝ) 
  (h1 : l = 3 * w)
  (h2 : A_p = 150) 
  (h3 : A_p = 0.4 * A_f)
  (h4 : A_f = l * w) : 
  l = 15 * Real.sqrt 5 :=
by
  sorry

end field_length_l792_79259


namespace point_on_graph_l792_79297

theorem point_on_graph (g : ℝ → ℝ) (h : g 8 = 10) :
  ∃ x y : ℝ, 3 * y = g (3 * x - 1) + 3 ∧ x = 3 ∧ y = 13 / 3 ∧ x + y = 22 / 3 :=
by
  sorry

end point_on_graph_l792_79297


namespace zeros_distance_l792_79262

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3*x^2 + a

theorem zeros_distance (a x1 x2 : ℝ) 
  (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) (h_order: x1 < x2) : 
  x2 - x1 = 3 := 
sorry

end zeros_distance_l792_79262


namespace average_temperature_week_l792_79222

theorem average_temperature_week 
  (T_sun : ℝ := 40)
  (T_mon : ℝ := 50)
  (T_tue : ℝ := 65)
  (T_wed : ℝ := 36)
  (T_thu : ℝ := 82)
  (T_fri : ℝ := 72)
  (T_sat : ℝ := 26) :
  (T_sun + T_mon + T_tue + T_wed + T_thu + T_fri + T_sat) / 7 = 53 :=
by
  sorry

end average_temperature_week_l792_79222


namespace total_trees_after_planting_l792_79277

def current_trees : ℕ := 7
def trees_planted_today : ℕ := 5
def trees_planted_tomorrow : ℕ := 4

theorem total_trees_after_planting : 
  current_trees + trees_planted_today + trees_planted_tomorrow = 16 :=
by
  sorry

end total_trees_after_planting_l792_79277


namespace range_of_f_l792_79213

noncomputable def f (x : ℝ) : ℝ :=
  x + Real.sqrt (x - 2)

theorem range_of_f : Set.range f = {y : ℝ | 2 ≤ y} :=
by
  sorry

end range_of_f_l792_79213


namespace total_volume_collection_l792_79248

-- Define the conditions
def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 15
def cost_per_box : ℚ := 0.5
def minimum_total_cost : ℚ := 255

-- Define the volume of one box
def volume_of_one_box : ℕ := box_length * box_width * box_height

-- Define the number of boxes needed
def number_of_boxes : ℚ := minimum_total_cost / cost_per_box

-- Define the total volume of the collection
def total_volume : ℚ := volume_of_one_box * number_of_boxes

-- The goal is to prove that the total volume of the collection is as calculated
theorem total_volume_collection :
  total_volume = 3060000 := by
  sorry

end total_volume_collection_l792_79248


namespace complement_U_A_eq_l792_79223
noncomputable def U := {x : ℝ | x ≥ -2}
noncomputable def A := {x : ℝ | x > -1}
noncomputable def comp_U_A := {x ∈ U | x ∉ A}

theorem complement_U_A_eq : comp_U_A = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by sorry

end complement_U_A_eq_l792_79223


namespace geometric_sequence_sum_reciprocal_ratio_l792_79293

theorem geometric_sequence_sum_reciprocal_ratio
  (a : ℚ) (r : ℚ) (n : ℕ) (S S' : ℚ)
  (h1 : a = 1/4)
  (h2 : r = 2)
  (h3 : S = a * (1 - r^n) / (1 - r))
  (h4 : S' = (1/a) * (1 - (1/r)^n) / (1 - 1/r)) :
  S / S' = 32 :=
sorry

end geometric_sequence_sum_reciprocal_ratio_l792_79293


namespace factor_difference_of_squares_l792_79207

-- Given: x is a real number.
-- Prove: x^2 - 64 = (x - 8) * (x + 8).
theorem factor_difference_of_squares (x : ℝ) : 
  x^2 - 64 = (x - 8) * (x + 8) :=
by
  sorry

end factor_difference_of_squares_l792_79207


namespace calculate_total_revenue_l792_79217

-- Definitions based on conditions
def apple_pie_slices := 8
def peach_pie_slices := 6
def cherry_pie_slices := 10

def apple_pie_price := 3
def peach_pie_price := 4
def cherry_pie_price := 5

def apple_pie_customers := 88
def peach_pie_customers := 78
def cherry_pie_customers := 45

-- Definition of total revenue
def total_revenue := 
  (apple_pie_customers * apple_pie_price) + 
  (peach_pie_customers * peach_pie_price) + 
  (cherry_pie_customers * cherry_pie_price)

-- Target theorem to prove: total revenue equals 801
theorem calculate_total_revenue : total_revenue = 801 := by
  sorry

end calculate_total_revenue_l792_79217


namespace sum_of_center_coords_l792_79240

theorem sum_of_center_coords (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : 2 + (-3) = -1 :=
by
  sorry

end sum_of_center_coords_l792_79240


namespace minimum_surface_area_l792_79230

def small_cuboid_1_length := 3 -- Edge length of small cuboid
def small_cuboid_2_length := 4 -- Edge length of small cuboid
def small_cuboid_3_length := 5 -- Edge length of small cuboid

def num_small_cuboids := 24 -- Number of small cuboids used to build the large cuboid

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def large_cuboid_length := 15 -- Corrected length dimension
def large_cuboid_width := 10  -- Corrected width dimension
def large_cuboid_height := 16 -- Corrected height dimension

theorem minimum_surface_area : surface_area large_cuboid_length large_cuboid_width large_cuboid_height = 788 := by
  sorry -- Proof to be completed

end minimum_surface_area_l792_79230


namespace max_digit_d_for_number_divisible_by_33_l792_79284

theorem max_digit_d_for_number_divisible_by_33 : ∃ d e : ℕ, d ≤ 9 ∧ e ≤ 9 ∧ 8 * 100000 + d * 10000 + 8 * 1000 + 3 * 100 + 3 * 10 + e % 33 = 0 ∧  d = 8 :=
by {
  sorry
}

end max_digit_d_for_number_divisible_by_33_l792_79284


namespace solve_for_x_l792_79244

theorem solve_for_x (x : ℝ) (h : 0.20 * x = 0.15 * 1500 - 15) : x = 1050 := 
by
  sorry

end solve_for_x_l792_79244


namespace woman_speed_in_still_water_l792_79252

noncomputable def speed_in_still_water (V_c : ℝ) (t : ℝ) (d : ℝ) : ℝ :=
  let V_downstream := d / (t / 3600)
  V_downstream - V_c

theorem woman_speed_in_still_water :
  let V_c := 60
  let t := 9.99920006399488
  let d := 0.5 -- 500 meters converted to kilometers
  speed_in_still_water V_c t d = 120.01800180018 :=
by
  unfold speed_in_still_water
  sorry

end woman_speed_in_still_water_l792_79252


namespace estimate_fish_population_l792_79292

theorem estimate_fish_population :
  ∀ (x : ℕ), (1200 / x = 100 / 1000) → x = 12000 := by
  sorry

end estimate_fish_population_l792_79292


namespace gcd_lcm_product_eq_l792_79289

-- Define the numbers
def a : ℕ := 10
def b : ℕ := 15

-- Define the GCD and LCM
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Proposition that needs to be proved
theorem gcd_lcm_product_eq : gcd_ab * lcm_ab = 150 :=
  by
    -- Proof would go here
    sorry

end gcd_lcm_product_eq_l792_79289


namespace rectangle_width_decrease_l792_79227

theorem rectangle_width_decrease (L W : ℝ) (h1 : 0 < L) (h2 : 0 < W) 
(h3 : ∀ W' : ℝ, 0 < W' → (1.3 * L * W' = L * W) → W' = (100 - 23.077) / 100 * W) : 
  ∃ W' : ℝ, 0 < W' ∧ (1.3 * L * W' = L * W) ∧ ((W - W') / W = 23.077 / 100) :=
by
  sorry

end rectangle_width_decrease_l792_79227


namespace practice_hours_l792_79274

-- Define the starting and ending hours, and the break duration
def start_hour : ℕ := 8
def end_hour : ℕ := 16
def break_duration : ℕ := 2

-- Compute the total practice hours
def total_practice_time : ℕ := (end_hour - start_hour) - break_duration

-- State that the computed practice time is equal to 6 hours
theorem practice_hours :
  total_practice_time = 6 := 
by
  -- Using the definitions provided to state the proof
  sorry

end practice_hours_l792_79274


namespace zero_in_set_zero_l792_79299

-- Define that 0 is an element
def zero_element : Prop := true

-- Define that {0} is a set containing only the element 0
def set_zero : Set ℕ := {0}

-- The main theorem that proves 0 ∈ {0}
theorem zero_in_set_zero (h : zero_element) : 0 ∈ set_zero := 
by sorry

end zero_in_set_zero_l792_79299


namespace intersection_of_sets_l792_79251

-- Define sets A and B as given in the conditions
def A : Set ℝ := { x | -2 < x ∧ x < 2 }

def B : Set ℝ := {0, 1, 2}

-- Define the proposition to be proved
theorem intersection_of_sets : A ∩ B = {0, 1} :=
by
  sorry

end intersection_of_sets_l792_79251


namespace king_total_payment_l792_79220

theorem king_total_payment
  (crown_cost : ℕ)
  (architect_cost : ℕ)
  (chef_cost : ℕ)
  (crown_tip_percent : ℕ)
  (architect_tip_percent : ℕ)
  (chef_tip_percent : ℕ)
  (crown_tip : ℕ)
  (architect_tip : ℕ)
  (chef_tip : ℕ)
  (total_crown_cost : ℕ)
  (total_architect_cost : ℕ)
  (total_chef_cost : ℕ)
  (total_paid : ℕ) :
  crown_cost = 20000 →
  architect_cost = 50000 →
  chef_cost = 10000 →
  crown_tip_percent = 10 →
  architect_tip_percent = 5 →
  chef_tip_percent = 15 →
  crown_tip = crown_cost * crown_tip_percent / 100 →
  architect_tip = architect_cost * architect_tip_percent / 100 →
  chef_tip = chef_cost * chef_tip_percent / 100 →
  total_crown_cost = crown_cost + crown_tip →
  total_architect_cost = architect_cost + architect_tip →
  total_chef_cost = chef_cost + chef_tip →
  total_paid = total_crown_cost + total_architect_cost + total_chef_cost →
  total_paid = 86000 := by
  sorry

end king_total_payment_l792_79220


namespace slow_speed_distance_l792_79216

theorem slow_speed_distance (D : ℝ) (h : (D + 20) / 14 = D / 10) : D = 50 := by
  sorry

end slow_speed_distance_l792_79216


namespace smallest_student_count_l792_79276

theorem smallest_student_count (x y z w : ℕ) 
  (ratio12to10 : x / y = 3 / 2) 
  (ratio12to11 : x / z = 7 / 4) 
  (ratio12to9 : x / w = 5 / 3) : 
  x + y + z + w = 298 :=
by
  sorry

end smallest_student_count_l792_79276


namespace intervals_of_increase_of_f_l792_79263

theorem intervals_of_increase_of_f :
  ∀ k : ℤ,
  ∀ x y : ℝ,
  k * π - (5 / 8) * π ≤ x ∧ x ≤ y ∧ y ≤ k * π - (1 / 8) * π →
  3 * Real.sin ((π / 4) - 2 * x) - 2 ≤ 3 * Real.sin ((π / 4) - 2 * y) - 2 :=
by
  sorry

end intervals_of_increase_of_f_l792_79263


namespace third_box_weight_l792_79238

def b1 : ℕ := 2
def difference := 11

def weight_third_box (b1 b3 difference : ℕ) : Prop :=
  b3 - b1 = difference

theorem third_box_weight : weight_third_box b1 13 difference :=
by
  simp [b1, difference]
  sorry

end third_box_weight_l792_79238


namespace compare_b_d_l792_79258

noncomputable def percentage_increase (x : ℝ) (p : ℝ) := x * (1 + p)
noncomputable def percentage_decrease (x : ℝ) (p : ℝ) := x * (1 - p)

theorem compare_b_d (a b c d : ℝ)
  (h1 : 0 < b)
  (h2 : a = percentage_increase b 0.02)
  (h3 : c = percentage_decrease a 0.01)
  (h4 : d = percentage_decrease c 0.01) :
  b > d :=
sorry

end compare_b_d_l792_79258


namespace min_dSigma_correct_l792_79282

noncomputable def min_dSigma {a r : ℝ} (h : a > r) : ℝ :=
  (a - r) / 2

theorem min_dSigma_correct (a r : ℝ) (h : a > r) :
  min_dSigma h = (a - r) / 2 :=
by 
  unfold min_dSigma
  sorry

end min_dSigma_correct_l792_79282


namespace oak_trees_cut_down_l792_79283

   def number_of_cuts (initial: ℕ) (remaining: ℕ) : ℕ :=
     initial - remaining

   theorem oak_trees_cut_down : number_of_cuts 9 7 = 2 :=
   by
     -- Based on the conditions, we start with 9 and after workers finished, there are 7 oak trees.
     -- We calculate the number of trees cut down:
     -- 9 - 7 = 2
     sorry
   
end oak_trees_cut_down_l792_79283


namespace sufficient_not_necessary_condition_l792_79285

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

theorem sufficient_not_necessary_condition :
  (∀ x, x ∈ P → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P) :=
by
  sorry

end sufficient_not_necessary_condition_l792_79285


namespace cubic_roots_l792_79268

theorem cubic_roots (a x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 1) (h₃ : x₃ = a)
  (cond : (2 / x₁) + (2 / x₂) = (3 / x₃)) :
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = a ∧ (a = 2 ∨ a = 3 / 4)) :=
by
  sorry

end cubic_roots_l792_79268


namespace reduced_flow_rate_is_correct_l792_79257

-- Define the original flow rate
def original_flow_rate : ℝ := 5.0

-- Define the function for the reduced flow rate
def reduced_flow_rate (x : ℝ) : ℝ := 0.6 * x - 1

-- Prove that the reduced flow rate is 2.0 gallons per minute
theorem reduced_flow_rate_is_correct : reduced_flow_rate original_flow_rate = 2.0 := by
  sorry

end reduced_flow_rate_is_correct_l792_79257


namespace find_k_l792_79261

-- Define the conditions and the question
theorem find_k (t k : ℝ) (h1 : t = 50) (h2 : t = (5 / 9) * (k - 32)) : k = 122 := by
  -- Proof will go here
  sorry

end find_k_l792_79261


namespace find_x_l792_79210

noncomputable def inv_cubicroot (y x : ℝ) : ℝ := y * x^(1/3)

theorem find_x (x y : ℝ) (h1 : ∃ k, inv_cubicroot 2 8 = k) (h2 : y = 8) : x = 1 / 8 :=
by
  sorry

end find_x_l792_79210


namespace bella_eats_six_apples_a_day_l792_79266

variable (A : ℕ) -- Number of apples Bella eats per day
variable (G : ℕ) -- Total number of apples Grace picks in 6 weeks
variable (B : ℕ) -- Total number of apples Bella eats in 6 weeks

-- Definitions for the conditions 
def condition1 := B = 42 * A
def condition2 := B = (1 / 3) * G
def condition3 := (2 / 3) * G = 504

-- Final statement that needs to be proved
theorem bella_eats_six_apples_a_day (A G B : ℕ) 
  (h1 : condition1 A B) 
  (h2 : condition2 G B) 
  (h3 : condition3 G) 
  : A = 6 := by sorry

end bella_eats_six_apples_a_day_l792_79266


namespace aram_fraction_of_fine_l792_79250

theorem aram_fraction_of_fine
  (F : ℝ)
  (Joe_payment : ℝ := (1 / 4) * F + 3)
  (Peter_payment : ℝ := (1 / 3) * F - 3)
  (Aram_payment : ℝ := (1 / 2) * F - 4)
  (sum_payments_eq_F : Joe_payment + Peter_payment + Aram_payment = F):
  (Aram_payment / F) = (5 / 12) :=
by
  sorry

end aram_fraction_of_fine_l792_79250


namespace cos_7_theta_l792_79255

variable (θ : Real)

namespace CosineProof

theorem cos_7_theta (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -5669 / 16384 := by
  sorry

end CosineProof

end cos_7_theta_l792_79255


namespace point_b_not_inside_circle_a_l792_79265

theorem point_b_not_inside_circle_a (a : ℝ) : a < 5 → ¬ (1 < a ∧ a < 5) :=
by
  sorry

end point_b_not_inside_circle_a_l792_79265


namespace sorting_five_rounds_l792_79204

def direct_sorting_method (l : List ℕ) : List ℕ := sorry

theorem sorting_five_rounds (initial_seq : List ℕ) :
  initial_seq = [49, 38, 65, 97, 76, 13, 27] →
  (direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method ∘ direct_sorting_method) initial_seq = [97, 76, 65, 49, 38, 13, 27] :=
by
  intros h
  sorry

end sorting_five_rounds_l792_79204


namespace product_of_solutions_l792_79237

theorem product_of_solutions :
  (∀ x : ℝ, |3 * x - 2| + 5 = 23 → x = 20 / 3 ∨ x = -16 / 3) →
  (20 / 3 * -16 / 3 = -320 / 9) :=
by
  intros h
  have h₁ : 20 / 3 * -16 / 3 = -320 / 9 := sorry
  exact h₁

end product_of_solutions_l792_79237


namespace registration_methods_l792_79242

theorem registration_methods :
  ∀ (interns : ℕ) (companies : ℕ), companies = 4 → interns = 5 → companies^interns = 1024 :=
by intros interns companies h1 h2; rw [h1, h2]; exact rfl

end registration_methods_l792_79242


namespace minimum_cuts_l792_79269

theorem minimum_cuts (n : Nat) : n >= 50 :=
by
  sorry

end minimum_cuts_l792_79269


namespace total_volume_of_cubes_l792_79229

theorem total_volume_of_cubes 
  (Carl_cubes : ℕ)
  (Carl_side_length : ℕ)
  (Kate_cubes : ℕ)
  (Kate_side_length : ℕ)
  (h1 : Carl_cubes = 8)
  (h2 : Carl_side_length = 2)
  (h3 : Kate_cubes = 3)
  (h4 : Kate_side_length = 3) :
  Carl_cubes * Carl_side_length ^ 3 + Kate_cubes * Kate_side_length ^ 3 = 145 :=
by
  sorry

end total_volume_of_cubes_l792_79229


namespace find_F_58_59_60_l792_79275

def F : ℤ → ℤ → ℤ → ℝ := sorry

axiom F_scaling (a b c n : ℤ) : F (n * a) (n * b) (n * c) = n * F a b c
axiom F_shift (a b c n : ℤ) : F (a + n) (b + n) (c + n) = F a b c + n
axiom F_symmetry (a b c : ℤ) : F a b c = F c b a

theorem find_F_58_59_60 : F 58 59 60 = 59 :=
sorry

end find_F_58_59_60_l792_79275


namespace pipe_drain_rate_l792_79247

theorem pipe_drain_rate 
(T r_A r_B r_C : ℕ) 
(h₁ : T = 950) 
(h₂ : r_A = 40) 
(h₃ : r_B = 30) 
(h₄ : ∃ m : ℕ, m = 57 ∧ (T = (m / 3) * (r_A + r_B - r_C))) : 
r_C = 20 :=
sorry

end pipe_drain_rate_l792_79247


namespace number_of_real_roots_l792_79287

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 19) ^ x + (5 / 19) ^ x + (11 / 19) ^ x

noncomputable def g (x : ℝ) : ℝ := sqrt (x - 1)

theorem number_of_real_roots : ∃! x : ℝ, 1 ≤ x ∧ f x = g x :=
by
  sorry

end number_of_real_roots_l792_79287


namespace Louisa_total_travel_time_l792_79243

theorem Louisa_total_travel_time :
  ∀ (v : ℝ), v > 0 → (200 / v) + 4 = (360 / v) → (200 / v) + (360 / v) = 14 :=
by
  intros v hv eqn
  sorry

end Louisa_total_travel_time_l792_79243


namespace cistern_problem_l792_79245

theorem cistern_problem (T : ℝ) (h1 : (1 / 2 - 1 / T) = 1 / 2.571428571428571) : T = 9 :=
by
  sorry

end cistern_problem_l792_79245


namespace sarah_meals_count_l792_79272

theorem sarah_meals_count :
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  main_courses * sides * drinks * desserts = 48 := 
by
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  calc
    4 * 3 * 2 * 2 = 48 := sorry

end sarah_meals_count_l792_79272


namespace chosen_number_l792_79226

theorem chosen_number (x : ℝ) (h : 2 * x - 138 = 102) : x = 120 := by
  sorry

end chosen_number_l792_79226


namespace map_to_actual_distance_ratio_l792_79211

def distance_in_meters : ℝ := 250
def distance_on_map_cm : ℝ := 5
def cm_per_meter : ℝ := 100

theorem map_to_actual_distance_ratio :
  distance_on_map_cm / (distance_in_meters * cm_per_meter) = 1 / 5000 :=
by
  sorry

end map_to_actual_distance_ratio_l792_79211


namespace even_function_inequality_l792_79232

variable {α : Type*} [LinearOrderedField α]

def is_even_function (f : α → α) : Prop := ∀ x, f x = f (-x)

-- The hypothesis and the assertion in Lean
theorem even_function_inequality
  (f : α → α)
  (h_even : is_even_function f)
  (h3_gt_1 : f 3 > f 1)
  : f (-1) < f 3 :=
sorry

end even_function_inequality_l792_79232


namespace smallest_square_perimeter_l792_79278

theorem smallest_square_perimeter (P_largest : ℕ) (units_apart : ℕ) (num_squares : ℕ) (H1 : P_largest = 96) (H2 : units_apart = 1) (H3 : num_squares = 8) : 
  ∃ P_smallest : ℕ, P_smallest = 40 := by
  sorry

end smallest_square_perimeter_l792_79278


namespace father_l792_79218

theorem father's_age :
  ∃ (S F : ℕ), 2 * S + F = 70 ∧ S + 2 * F = 95 ∧ F = 40 :=
by
  sorry

end father_l792_79218


namespace lunks_needed_for_bananas_l792_79219

theorem lunks_needed_for_bananas :
  (7 : ℚ) / 4 * (20 * 3 / 5) = 21 :=
by
  sorry

end lunks_needed_for_bananas_l792_79219


namespace length_AE_l792_79214

structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨0, 4⟩
def B : Point := ⟨7, 0⟩
def C : Point := ⟨5, 3⟩
def D : Point := ⟨3, 0⟩

noncomputable def dist (P Q : Point) : ℝ :=
  Real.sqrt (((Q.x - P.x : ℝ) ^ 2) + ((Q.y - P.y : ℝ) ^ 2))

noncomputable def AE_length : ℝ :=
  (5 * (dist A B)) / 9

theorem length_AE :
  ∃ E : Point, AE_length = (5 * Real.sqrt 65) / 9 := by
  sorry

end length_AE_l792_79214


namespace degree_measure_cherry_pie_l792_79208

theorem degree_measure_cherry_pie 
  (total_students : ℕ) 
  (chocolate_pie : ℕ) 
  (apple_pie : ℕ) 
  (blueberry_pie : ℕ) 
  (remaining_students : ℕ)
  (remaining_students_eq_div : remaining_students = (total_students - (chocolate_pie + apple_pie + blueberry_pie))) 
  (equal_division : remaining_students / 2 = 5) 
  : (remaining_students / 2 * 360 / total_students = 45) := 
by 
  sorry

end degree_measure_cherry_pie_l792_79208


namespace cuboid_volume_l792_79290

theorem cuboid_volume (base_area height : ℝ) (h_base_area : base_area = 18) (h_height : height = 8) : 
  base_area * height = 144 :=
by
  rw [h_base_area, h_height]
  norm_num

end cuboid_volume_l792_79290


namespace symmetric_points_power_l792_79224

variables (m n : ℝ)

def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetric_points_power 
  (h : symmetric_y_axis (m, 3) (4, n)) : 
  (m + n) ^ 2023 = -1 :=
by 
  sorry

end symmetric_points_power_l792_79224


namespace composite_sum_of_squares_l792_79236

theorem composite_sum_of_squares (a b : ℤ) (h_roots : ∃ x1 x2 : ℕ, (x1 + x2 : ℤ) = -a ∧ (x1 * x2 : ℤ) = b + 1) :
  ∃ m n : ℕ, a^2 + b^2 = m * n ∧ 1 < m ∧ 1 < n :=
sorry

end composite_sum_of_squares_l792_79236


namespace remainder_when_divided_82_l792_79202

theorem remainder_when_divided_82 (x : ℤ) (k m : ℤ) (R : ℤ) (h1 : 0 ≤ R) (h2 : R < 82)
    (h3 : x = 82 * k + R) (h4 : x + 7 = 41 * m + 12) : R = 5 :=
by
  sorry

end remainder_when_divided_82_l792_79202


namespace smallest_prime_square_mod_six_l792_79271

theorem smallest_prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p^2 % 6 = 1) : p = 5 :=
sorry

end smallest_prime_square_mod_six_l792_79271


namespace packs_per_box_l792_79205

theorem packs_per_box (total_cost : ℝ) (num_boxes : ℕ) (cost_per_pack : ℝ) 
  (num_tissues_per_pack : ℕ) (cost_per_tissue : ℝ) (total_packs : ℕ) :
  total_cost = 1000 ∧ num_boxes = 10 ∧ cost_per_pack = num_tissues_per_pack * cost_per_tissue ∧ 
  num_tissues_per_pack = 100 ∧ cost_per_tissue = 0.05 ∧ total_packs * cost_per_pack = total_cost / num_boxes →
  total_packs = 20 :=
by
  sorry

end packs_per_box_l792_79205


namespace square_implies_increasing_l792_79212

def seq (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n > 1, 
    ((a n - 2 > 0 ∧ ¬(∃ m < n, a m = a n - 2)) → a (n + 1) = a n - 2) ∧
    ((a n - 2 ≤ 0 ∨ ∃ m < n, a m = a n - 2) → a (n + 1) = a n + 3)

theorem square_implies_increasing (a : ℕ → ℤ) (n : ℕ) (h_seq : seq a) 
  (h_square : ∃ k, a n = k^2) (h_n_pos : n > 1) : 
  a n > a (n - 1) :=
sorry

end square_implies_increasing_l792_79212


namespace perimeter_of_triangle_is_13_l792_79200

-- Conditions
noncomputable def perimeter_of_triangle_with_two_sides_and_third_root_of_eq : ℝ :=
  let a := 3
  let b := 6
  let c1 := 2 -- One root of the equation x^2 - 6x + 8 = 0
  let c2 := 4 -- Another root of the equation x^2 - 6x + 8 = 0
  if a + b > c2 ∧ a + c2 > b ∧ b + c2 > a then
    a + b + c2
  else
    0 -- not possible to form a triangle with these sides

-- Assertion
theorem perimeter_of_triangle_is_13 :
  perimeter_of_triangle_with_two_sides_and_third_root_of_eq = 13 := 
sorry

end perimeter_of_triangle_is_13_l792_79200


namespace minimum_value_inequality_l792_79241

theorem minimum_value_inequality (m n : ℝ) (h₁ : m > n) (h₂ : n > 0) : m + (n^2 - mn + 4)/(m - n) ≥ 4 :=
  sorry

end minimum_value_inequality_l792_79241


namespace jessica_saves_l792_79279

-- Define the costs based on the conditions given
def basic_cost : ℕ := 15
def movie_cost : ℕ := 12
def sports_cost : ℕ := movie_cost - 3
def bundle_cost : ℕ := 25

-- Define the total cost when the packages are purchased separately
def separate_cost : ℕ := basic_cost + movie_cost + sports_cost

-- Define the savings when opting for the bundle
def savings : ℕ := separate_cost - bundle_cost

-- The theorem that states the savings are 11 dollars
theorem jessica_saves : savings = 11 :=
by
  sorry

end jessica_saves_l792_79279


namespace min_value_theorem_l792_79233

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x + 2) * (2 * y + 1) / (x * y)

theorem min_value_theorem {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  min_value x y = 19 + 4 * Real.sqrt 15 :=
sorry

end min_value_theorem_l792_79233


namespace union_M_N_l792_79298

def M : Set ℝ := { x | x^2 + 2 * x = 0 }
def N : Set ℝ := { x | x^2 - 2 * x = 0 }

theorem union_M_N : M ∪ N = {0, -2, 2} := by
  sorry

end union_M_N_l792_79298


namespace museum_pictures_l792_79291

theorem museum_pictures (P : ℕ) (h1 : ¬ (∃ k, P = 2 * k)) (h2 : ∃ k, P + 1 = 2 * k) : P = 3 := 
by 
  sorry

end museum_pictures_l792_79291


namespace shadow_length_l792_79267

variable (H h d : ℝ) (h_pos : h > 0) (H_pos : H > 0) (H_neq_h : H ≠ h)

theorem shadow_length (x : ℝ) (hx : x = d * h / (H - h)) :
  x = d * h / (H - h) :=
sorry

end shadow_length_l792_79267


namespace find_duplicate_page_l792_79231

theorem find_duplicate_page (n p : ℕ) (h : (n * (n + 1) / 2) + p = 3005) : p = 2 := 
sorry

end find_duplicate_page_l792_79231


namespace rational_solution_system_l792_79215

theorem rational_solution_system (x y z t w : ℚ) :
  (t^2 - w^2 + z^2 = 2 * x * y) →
  (t^2 - y^2 + w^2 = 2 * x * z) →
  (t^2 - w^2 + x^2 = 2 * y * z) →
  x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intros h1 h2 h3
  sorry

end rational_solution_system_l792_79215
