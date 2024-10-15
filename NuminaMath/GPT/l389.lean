import Mathlib

namespace NUMINAMATH_GPT_number_of_divisors_of_720_l389_38954

theorem number_of_divisors_of_720 : 
  let n := 720
  let prime_factorization := [(2, 4), (3, 2), (5, 1)] 
  let num_divisors := (4 + 1) * (2 + 1) * (1 + 1)
  n = 2^4 * 3^2 * 5^1 →
  num_divisors = 30 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_divisors_of_720_l389_38954


namespace NUMINAMATH_GPT_tetrahedron_ratio_l389_38994

theorem tetrahedron_ratio (a b c d : ℝ) (h₁ : a^2 = b^2 + c^2) (h₂ : b^2 = a^2 + d^2) (h₃ : c^2 = a^2 + b^2) : 
  a / d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_tetrahedron_ratio_l389_38994


namespace NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l389_38997

theorem sum_of_three_consecutive_odd_integers (x : ℤ) 
  (h1 : x % 2 = 1)                          -- x is odd
  (h2 : (x + 4) % 2 = 1)                    -- x+4 is odd
  (h3 : x + (x + 4) = 150)                  -- sum of first and third integers is 150
  : x + (x + 2) + (x + 4) = 225 :=          -- sum of the three integers is 225
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_odd_integers_l389_38997


namespace NUMINAMATH_GPT_lcm_six_ten_fifteen_is_30_l389_38920

-- Define the numbers and their prime factorizations
def six := 6
def ten := 10
def fifteen := 15

noncomputable def lcm_six_ten_fifteen : ℕ :=
  Nat.lcm (Nat.lcm six ten) fifteen

-- The theorem to prove the LCM
theorem lcm_six_ten_fifteen_is_30 : lcm_six_ten_fifteen = 30 :=
  sorry

end NUMINAMATH_GPT_lcm_six_ten_fifteen_is_30_l389_38920


namespace NUMINAMATH_GPT_incorrect_number_read_l389_38904

theorem incorrect_number_read (incorrect_avg correct_avg : ℕ) (n correct_number incorrect_sum correct_sum : ℕ)
  (h1 : incorrect_avg = 17)
  (h2 : correct_avg = 20)
  (h3 : n = 10)
  (h4 : correct_number = 56)
  (h5 : incorrect_sum = n * incorrect_avg)
  (h6 : correct_sum = n * correct_avg)
  (h7 : correct_sum - incorrect_sum = correct_number - X) :
  X = 26 :=
by
  sorry

end NUMINAMATH_GPT_incorrect_number_read_l389_38904


namespace NUMINAMATH_GPT_ratio_of_sphere_surface_areas_l389_38943

noncomputable def inscribed_sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ := a * (Real.sqrt 3) / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem ratio_of_sphere_surface_areas (a : ℝ) (h : 0 < a) : 
  (sphere_surface_area (circumscribed_sphere_radius a)) / (sphere_surface_area (inscribed_sphere_radius a)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_sphere_surface_areas_l389_38943


namespace NUMINAMATH_GPT_smaller_rectangle_perimeter_l389_38990

def problem_conditions (a b : ℝ) : Prop :=
  2 * (a + b) = 96 ∧ 
  8 * b + 11 * a = 342 ∧
  a + b = 48 ∧ 
  (a * (b - 1) <= 0 ∧ b * (a - 1) <= 0 ∧ a > 0 ∧ b > 0)

theorem smaller_rectangle_perimeter (a b : ℝ) (hab : problem_conditions a b) :
  2 * (a / 12 + b / 9) = 9 :=
  sorry

end NUMINAMATH_GPT_smaller_rectangle_perimeter_l389_38990


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l389_38916

theorem necessary_but_not_sufficient (x : ℝ) :
  (x < 1 ∨ x > 4) → (x^2 - 3 * x + 2 > 0) ∧ ¬((x^2 - 3 * x + 2 > 0) → (x < 1 ∨ x > 4)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l389_38916


namespace NUMINAMATH_GPT_probability_x_gt_3y_in_rectangle_l389_38902

noncomputable def probability_of_x_gt_3y :ℝ :=
  let base := 2010
  let height := 2011
  let triangle_height := 670
  (1/2 * base * triangle_height) / (base * height)

theorem probability_x_gt_3y_in_rectangle:
  probability_of_x_gt_3y = 335 / 2011 := 
by
  sorry

end NUMINAMATH_GPT_probability_x_gt_3y_in_rectangle_l389_38902


namespace NUMINAMATH_GPT_sum_of_squares_l389_38955

theorem sum_of_squares (a b : ℕ) (h_side_lengths : 20^2 = a^2 + b^2) : a + b = 28 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l389_38955


namespace NUMINAMATH_GPT_xiao_wang_parts_processed_l389_38927

-- Definitions for the processing rates and conditions
def xiao_wang_rate := 15 -- parts per hour
def xiao_wang_max_continuous_hours := 2
def xiao_wang_break_hours := 1

def xiao_li_rate := 12 -- parts per hour

-- Constants for the problem setup
def xiao_wang_process_time := 4 -- hours including breaks after first cycle
def xiao_li_process_time := 5 -- hours including no breaks

-- Total parts processed by both when they finish simultaneously
def parts_processed_when_finished_simultaneously := 60

theorem xiao_wang_parts_processed :
  (xiao_wang_rate * xiao_wang_max_continuous_hours) * (xiao_wang_process_time / 
  (xiao_wang_max_continuous_hours + xiao_wang_break_hours)) =
  parts_processed_when_finished_simultaneously :=
sorry

end NUMINAMATH_GPT_xiao_wang_parts_processed_l389_38927


namespace NUMINAMATH_GPT_polygon_sides_l389_38932

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l389_38932


namespace NUMINAMATH_GPT_bob_overtime_pay_rate_l389_38929

theorem bob_overtime_pay_rate :
  let regular_pay_rate := 5
  let total_hours := (44, 48)
  let total_pay := 472
  let overtime_hours (hours : Nat) := max 0 (hours - 40)
  let regular_hours (hours : Nat) := min 40 hours
  let total_regular_hours := regular_hours 44 + regular_hours 48
  let total_regular_pay := total_regular_hours * regular_pay_rate
  let total_overtime_hours := overtime_hours 44 + overtime_hours 48
  let total_overtime_pay := total_pay - total_regular_pay
  let overtime_pay_rate := total_overtime_pay / total_overtime_hours
  overtime_pay_rate = 6 := by sorry

end NUMINAMATH_GPT_bob_overtime_pay_rate_l389_38929


namespace NUMINAMATH_GPT_classroom_student_count_l389_38982

theorem classroom_student_count (n : ℕ) (students_avg : ℕ) (teacher_age : ℕ) (combined_avg : ℕ) 
  (h1 : students_avg = 8) (h2 : teacher_age = 32) (h3 : combined_avg = 11) 
  (h4 : (8 * n + 32) / (n + 1) = 11) : n + 1 = 8 :=
by
  sorry

end NUMINAMATH_GPT_classroom_student_count_l389_38982


namespace NUMINAMATH_GPT_functional_equation_solution_l389_38935

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  intro h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l389_38935


namespace NUMINAMATH_GPT_express_in_scientific_notation_l389_38978

def scientific_notation (n : ℤ) (x : ℝ) :=
  ∃ (a : ℝ) (b : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^b

theorem express_in_scientific_notation : scientific_notation (-8206000) (-8.206 * 10^6) :=
by
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l389_38978


namespace NUMINAMATH_GPT_initial_profit_price_reduction_for_target_profit_l389_38984

-- Define given conditions
def purchase_price : ℝ := 280
def initial_selling_price : ℝ := 360
def items_sold_per_month : ℕ := 60
def target_profit : ℝ := 7200
def increment_per_reduced_yuan : ℕ := 5

-- Problem 1: Prove the initial profit per month before the price reduction
theorem initial_profit : 
  items_sold_per_month * (initial_selling_price - purchase_price) = 4800 := by
sorry

-- Problem 2: Prove that reducing the price by 60 yuan achieves the target profit
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, 
    ((initial_selling_price - x) - purchase_price) * (items_sold_per_month + (increment_per_reduced_yuan * x)) = target_profit ∧
    x = 60 := by
sorry

end NUMINAMATH_GPT_initial_profit_price_reduction_for_target_profit_l389_38984


namespace NUMINAMATH_GPT_cars_in_garage_l389_38992

/-
Conditions:
1. Total wheels in the garage: 22
2. Riding lawnmower wheels: 4
3. Timmy's bicycle wheels: 2
4. Each of Timmy's parents' bicycles: 2 wheels, and there are 2 bicycles.
5. Joey's tricycle wheels: 3
6. Timmy's dad's unicycle wheels: 1

Question: How many cars are inside the garage?

Correct Answer: The number of cars is 2.
-/
theorem cars_in_garage (total_wheels : ℕ) (lawnmower_wheels : ℕ)
  (timmy_bicycle_wheels : ℕ) (parents_bicycles_wheels : ℕ)
  (joey_tricycle_wheels : ℕ) (dad_unicycle_wheels : ℕ) 
  (cars_wheels : ℕ) (cars : ℕ) :
  total_wheels = 22 →
  lawnmower_wheels = 4 →
  timmy_bicycle_wheels = 2 →
  parents_bicycles_wheels = 2 * 2 →
  joey_tricycle_wheels = 3 →
  dad_unicycle_wheels = 1 →
  cars_wheels = total_wheels - (lawnmower_wheels + timmy_bicycle_wheels + parents_bicycles_wheels + joey_tricycle_wheels + dad_unicycle_wheels) →
  cars = cars_wheels / 4 →
  cars = 2 := by
  sorry

end NUMINAMATH_GPT_cars_in_garage_l389_38992


namespace NUMINAMATH_GPT_distance_between_5th_and_29th_red_light_in_feet_l389_38912

-- Define the repeating pattern length and individual light distance
def pattern_length := 7
def red_light_positions := {k | k % pattern_length < 3}
def distance_between_lights := 8 / 12  -- converting inches to feet

-- Positions of the 5th and 29th red lights in terms of pattern repetition
def position_of_nth_red_light (n : ℕ) : ℕ :=
  ((n-1) / 3) * pattern_length + (n-1) % 3 + 1

def position_5th_red_light := position_of_nth_red_light 5
def position_29th_red_light := position_of_nth_red_light 29

theorem distance_between_5th_and_29th_red_light_in_feet :
  (position_29th_red_light - position_5th_red_light - 1) * distance_between_lights = 37 := by
  sorry

end NUMINAMATH_GPT_distance_between_5th_and_29th_red_light_in_feet_l389_38912


namespace NUMINAMATH_GPT_find_two_sets_l389_38973

theorem find_two_sets :
  ∃ (a1 a2 a3 a4 a5 b1 b2 b3 b4 b5 : ℕ),
    a1 + a2 + a3 + a4 + a5 = a1 * a2 * a3 * a4 * a5 ∧
    b1 + b2 + b3 + b4 + b5 = b1 * b2 * b3 * b4 * b5 ∧
    (a1, a2, a3, a4, a5) ≠ (b1, b2, b3, b4, b5) := by
  sorry

end NUMINAMATH_GPT_find_two_sets_l389_38973


namespace NUMINAMATH_GPT_smallest_integer_M_exists_l389_38941

theorem smallest_integer_M_exists :
  ∃ (M : ℕ), 
    (M > 0) ∧ 
    (∃ (x y z : ℕ), 
      (x = M ∨ x = M + 1 ∨ x = M + 2) ∧ 
      (y = M ∨ y = M + 1 ∨ y = M + 2) ∧ 
      (z = M ∨ z = M + 1 ∨ z = M + 2) ∧ 
      ((x = M ∨ x = M + 1 ∨ x = M + 2) ∧ x % 8 = 0) ∧ 
      ((y = M ∨ y = M + 1 ∨ y = M + 2) ∧ y % 9 = 0) ∧ 
      ((z = M ∨ z = M + 1 ∨ z = M + 2) ∧ z % 25 = 0) ) ∧ 
    M = 200 := 
by
  sorry

end NUMINAMATH_GPT_smallest_integer_M_exists_l389_38941


namespace NUMINAMATH_GPT_find_principal_amount_l389_38906

theorem find_principal_amount (P r : ℝ) (A2 A3 : ℝ) (n2 n3 : ℕ) 
  (h1 : n2 = 2) (h2 : n3 = 3) 
  (h3 : A2 = 8820) 
  (h4 : A3 = 9261) 
  (h5 : r = 0.05) 
  (h6 : A2 = P * (1 + r)^n2) 
  (h7 : A3 = P * (1 + r)^n3) : 
  P = 8000 := 
by 
  sorry

end NUMINAMATH_GPT_find_principal_amount_l389_38906


namespace NUMINAMATH_GPT_factorial_product_square_root_square_l389_38940

theorem factorial_product_square_root_square :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 3))^2 = 17280 := 
by
  sorry

end NUMINAMATH_GPT_factorial_product_square_root_square_l389_38940


namespace NUMINAMATH_GPT_find_smallest_divisor_l389_38923

theorem find_smallest_divisor {n : ℕ} 
  (h : n = 44402) 
  (hdiv1 : (n + 2) % 30 = 0) 
  (hdiv2 : (n + 2) % 48 = 0) 
  (hdiv3 : (n + 2) % 74 = 0) 
  (hdiv4 : (n + 2) % 100 = 0) : 
  ∃ d, d = 37 ∧ d ∣ (n + 2) :=
sorry

end NUMINAMATH_GPT_find_smallest_divisor_l389_38923


namespace NUMINAMATH_GPT_paint_required_for_frame_l389_38972

theorem paint_required_for_frame :
  ∀ (width height thickness : ℕ) 
    (coverage : ℚ),
  width = 6 →
  height = 9 →
  thickness = 1 →
  coverage = 5 →
  (width * height - (width - 2 * thickness) * (height - 2 * thickness) + 2 * width * thickness + 2 * height * thickness) / coverage = 11.2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_paint_required_for_frame_l389_38972


namespace NUMINAMATH_GPT_wickets_before_last_match_l389_38971

theorem wickets_before_last_match (R W : ℝ) (h1 : R = 12.4 * W) (h2 : R + 26 = 12 * (W + 7)) :
  W = 145 := 
by 
  sorry

end NUMINAMATH_GPT_wickets_before_last_match_l389_38971


namespace NUMINAMATH_GPT_find_x_y_l389_38967

theorem find_x_y 
  (x y : ℚ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x) :
  x = 5 ∧ y = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l389_38967


namespace NUMINAMATH_GPT_kevin_total_distance_l389_38944

noncomputable def kevin_hop_total_distance_after_seven_leaps : ℚ :=
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  a * (1 - r^n) / (1 - r)

theorem kevin_total_distance (total_distance : ℚ) :
  total_distance = kevin_hop_total_distance_after_seven_leaps → 
  total_distance = 14197 / 16384 := by
  intro h
  sorry

end NUMINAMATH_GPT_kevin_total_distance_l389_38944


namespace NUMINAMATH_GPT_salt_weight_l389_38980

theorem salt_weight {S : ℝ} (h1 : 16 + S = 46) : S = 30 :=
by
  sorry

end NUMINAMATH_GPT_salt_weight_l389_38980


namespace NUMINAMATH_GPT_max_value_a_plus_2b_l389_38995

theorem max_value_a_plus_2b {a b : ℝ} (h_positive : 0 < a ∧ 0 < b) (h_eqn : a^2 + 2 * a * b + 4 * b^2 = 6) :
  a + 2 * b ≤ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_value_a_plus_2b_l389_38995


namespace NUMINAMATH_GPT_equal_numbers_l389_38909

namespace MathProblem

theorem equal_numbers 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x^2 / y + y^2 / z + z^2 / x = x^2 / z + z^2 / y + y^2 / x) : 
  x = y ∨ x = z ∨ y = z :=
by
  sorry

end MathProblem

end NUMINAMATH_GPT_equal_numbers_l389_38909


namespace NUMINAMATH_GPT_point_in_second_quadrant_l389_38919

-- Define the point in question
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Given conditions based on the problem statement
def P (x : ℝ) : Point :=
  Point.mk (-2) (x^2 + 1)

-- The theorem we aim to prove
theorem point_in_second_quadrant (x : ℝ) : (P x).x < 0 ∧ (P x).y > 0 → 
  -- This condition means that the point is in the second quadrant
  (P x).x < 0 ∧ (P x).y > 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l389_38919


namespace NUMINAMATH_GPT_rectangular_coordinates_of_polar_2_pi_over_3_l389_38985

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem rectangular_coordinates_of_polar_2_pi_over_3 :
  polar_to_rectangular 2 (Real.pi / 3) = (1, Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_coordinates_of_polar_2_pi_over_3_l389_38985


namespace NUMINAMATH_GPT_portrait_is_in_Silver_l389_38939

def Gold_inscription (located_in : String → Prop) : Prop := located_in "Gold"
def Silver_inscription (located_in : String → Prop) : Prop := ¬located_in "Silver"
def Lead_inscription (located_in : String → Prop) : Prop := ¬located_in "Gold"

def is_true (inscription : Prop) : Prop := inscription
def is_false (inscription : Prop) : Prop := ¬inscription

noncomputable def portrait_in_Silver_Given_Statements : Prop :=
  ∃ located_in : String → Prop,
    (is_true (Gold_inscription located_in) ∨ is_true (Silver_inscription located_in) ∨ is_true (Lead_inscription located_in)) ∧
    (is_false (Gold_inscription located_in) ∨ is_false (Silver_inscription located_in) ∨ is_false (Lead_inscription located_in)) ∧
    located_in "Silver"

theorem portrait_is_in_Silver : portrait_in_Silver_Given_Statements :=
by {
    sorry
}

end NUMINAMATH_GPT_portrait_is_in_Silver_l389_38939


namespace NUMINAMATH_GPT_possible_values_x_l389_38991

theorem possible_values_x : 
  let x := Nat.gcd 112 168 
  ∃ d : Finset ℕ, d.card = 8 ∧ ∀ y ∈ d, y ∣ 112 ∧ y ∣ 168 := 
by
  let x := Nat.gcd 112 168
  have : x = 56 := by norm_num
  use Finset.filter (fun n => 56 % n = 0) (Finset.range 57)
  sorry

end NUMINAMATH_GPT_possible_values_x_l389_38991


namespace NUMINAMATH_GPT_boat_speed_ratio_l389_38952

variable (B S : ℝ)

theorem boat_speed_ratio (h : 1 / (B - S) = 2 * (1 / (B + S))) : B / S = 3 := 
by
  sorry

end NUMINAMATH_GPT_boat_speed_ratio_l389_38952


namespace NUMINAMATH_GPT_total_marks_l389_38926

theorem total_marks (k l d : ℝ) (hk : k = 3.5) (hl : l = 3.2 * k) (hd : d = l + 5.7) : k + l + d = 31.6 :=
by
  rw [hk] at hl
  rw [hl] at hd
  rw [hk, hl, hd]
  sorry

end NUMINAMATH_GPT_total_marks_l389_38926


namespace NUMINAMATH_GPT_least_possible_integral_QR_l389_38988

theorem least_possible_integral_QR (PQ PR SR SQ QR : ℝ) (hPQ : PQ = 7) (hPR : PR = 10) (hSR : SR = 15) (hSQ : SQ = 24) :
  9 ≤ QR ∧ QR < 17 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_integral_QR_l389_38988


namespace NUMINAMATH_GPT_gcd_8p_18q_l389_38963

theorem gcd_8p_18q (p q : ℕ) (hp : p > 0) (hq : q > 0) (hg : Nat.gcd p q = 9) : Nat.gcd (8 * p) (18 * q) = 18 := 
sorry

end NUMINAMATH_GPT_gcd_8p_18q_l389_38963


namespace NUMINAMATH_GPT_cistern_length_l389_38946

-- Definitions of the given conditions
def width : ℝ := 4
def depth : ℝ := 1.25
def total_wet_surface_area : ℝ := 49

-- Mathematical problem: prove the length of the cistern
theorem cistern_length : ∃ (L : ℝ), (L * width + 2 * L * depth + 2 * width * depth = total_wet_surface_area) ∧ L = 6 :=
by
sorry

end NUMINAMATH_GPT_cistern_length_l389_38946


namespace NUMINAMATH_GPT_m_range_l389_38938

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (2018 * Real.pi - x) * Real.sin (3 * Real.pi / 2 + x) 
  - Real.cos x ^ 2 + 1

def valid_m (m : ℝ) : Prop := 
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), abs (f x - m) ≤ 1

theorem m_range : 
  ∀ m : ℝ, valid_m m ↔ (m ∈ Set.Icc (1 / 2) ((3 - Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_GPT_m_range_l389_38938


namespace NUMINAMATH_GPT_find_values_l389_38977

def isInInterval (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem find_values 
  (a b c d e : ℝ)
  (ha : isInInterval a) 
  (hb : isInInterval b) 
  (hc : isInInterval c) 
  (hd : isInInterval d)
  (he : isInInterval e)
  (h1 : a + b + c + d + e = 0)
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 = 0)
  (h3 : a^5 + b^5 + c^5 + d^5 + e^5 = 10) : 
  (a = 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = 2 ∧ c = (Real.sqrt 5 - 1) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = 2 ∧ e = - (1 + Real.sqrt 5) / 2) ∨
  (a = (Real.sqrt 5 - 1) / 2 ∧ b = (Real.sqrt 5 - 1) / 2 ∧ c = - (1 + Real.sqrt 5) / 2 ∧ d = - (1 + Real.sqrt 5) / 2 ∧ e = 2) :=
sorry

end NUMINAMATH_GPT_find_values_l389_38977


namespace NUMINAMATH_GPT_ratio_books_to_pens_l389_38903

-- Define the given ratios and known constants.
def ratio_pencils : ℕ := 14
def ratio_pens : ℕ := 4
def ratio_books : ℕ := 3
def actual_pencils : ℕ := 140

-- Assume the actual number of pens can be calculated from ratio.
def actual_pens : ℕ := (actual_pencils / ratio_pencils) * ratio_pens

-- Prove that the ratio of exercise books to pens is as expected.
theorem ratio_books_to_pens (h1 : actual_pencils = 140) 
                            (h2 : actual_pens = 40) : 
  ((actual_pencils / ratio_pencils) * ratio_books) / actual_pens = 3 / 4 :=
by
  -- The following proof steps are omitted as per instruction
  sorry

end NUMINAMATH_GPT_ratio_books_to_pens_l389_38903


namespace NUMINAMATH_GPT_bake_sale_money_raised_correct_l389_38930

def bake_sale_money_raised : Prop :=
  let chocolate_chip_cookies_baked := 4 * 12
  let oatmeal_raisin_cookies_baked := 6 * 12
  let regular_brownies_baked := 2 * 12
  let sugar_cookies_baked := 6 * 12
  let blondies_baked := 3 * 12
  let cream_cheese_swirled_brownies_baked := 5 * 12
  let chocolate_chip_cookies_price := 1.50
  let oatmeal_raisin_cookies_price := 1.00
  let regular_brownies_price := 2.50
  let sugar_cookies_price := 1.25
  let blondies_price := 2.75
  let cream_cheese_swirled_brownies_price := 3.00
  let chocolate_chip_cookies_sold := 0.75 * chocolate_chip_cookies_baked
  let oatmeal_raisin_cookies_sold := 0.85 * oatmeal_raisin_cookies_baked
  let regular_brownies_sold := 0.60 * regular_brownies_baked
  let sugar_cookies_sold := 0.90 * sugar_cookies_baked
  let blondies_sold := 0.80 * blondies_baked
  let cream_cheese_swirled_brownies_sold := 0.50 * cream_cheese_swirled_brownies_baked
  let total_money_raised := 
    chocolate_chip_cookies_sold * chocolate_chip_cookies_price + 
    oatmeal_raisin_cookies_sold * oatmeal_raisin_cookies_price + 
    regular_brownies_sold * regular_brownies_price + 
    sugar_cookies_sold * sugar_cookies_price + 
    blondies_sold * blondies_price + 
    cream_cheese_swirled_brownies_sold * cream_cheese_swirled_brownies_price
  total_money_raised = 397.00

theorem bake_sale_money_raised_correct : bake_sale_money_raised := by
  sorry

end NUMINAMATH_GPT_bake_sale_money_raised_correct_l389_38930


namespace NUMINAMATH_GPT_solve_diophantine_equations_l389_38956

theorem solve_diophantine_equations :
  { (a, b, c, d) : ℤ × ℤ × ℤ × ℤ |
    a * b - 2 * c * d = 3 ∧
    a * c + b * d = 1 } =
  { (1, 3, 1, 0), (-1, -3, -1, 0), (3, 1, 0, 1), (-3, -1, 0, -1) } :=
by
  sorry

end NUMINAMATH_GPT_solve_diophantine_equations_l389_38956


namespace NUMINAMATH_GPT_evaluate_expression_l389_38905

theorem evaluate_expression : 2 - (-3) - 4 - (-5) - 6 - (-7) - 8 = -1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l389_38905


namespace NUMINAMATH_GPT_number_of_elements_in_sequence_l389_38917

theorem number_of_elements_in_sequence :
  ∀ (a₀ d : ℕ) (n : ℕ), 
  a₀ = 4 →
  d = 2 →
  n = 64 →
  (a₀ + (n - 1) * d = 130) →
  n = 64 := 
by
  -- We will skip the proof steps as indicated
  sorry

end NUMINAMATH_GPT_number_of_elements_in_sequence_l389_38917


namespace NUMINAMATH_GPT_sum_of_a3_a4_a5_l389_38936

def geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a n = 3 * q ^ n

theorem sum_of_a3_a4_a5 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : geometric_sequence_sum a q)
  (h_pos : ∀ n, a n > 0)
  (h_first_term : a 0 = 3)
  (h_sum_first_three : a 0 + a 1 + a 2 = 21) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end NUMINAMATH_GPT_sum_of_a3_a4_a5_l389_38936


namespace NUMINAMATH_GPT_only_a_zero_is_perfect_square_l389_38915

theorem only_a_zero_is_perfect_square (a : ℕ) : (∃ (k : ℕ), a^2 + 2 * a = k^2) → a = 0 := by
  sorry

end NUMINAMATH_GPT_only_a_zero_is_perfect_square_l389_38915


namespace NUMINAMATH_GPT_number_of_valid_N_count_valid_N_is_seven_l389_38951

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end NUMINAMATH_GPT_number_of_valid_N_count_valid_N_is_seven_l389_38951


namespace NUMINAMATH_GPT_next_tutoring_day_lcm_l389_38961

theorem next_tutoring_day_lcm :
  Nat.lcm (Nat.lcm (Nat.lcm 4 5) 6) 8 = 120 := by
  sorry

end NUMINAMATH_GPT_next_tutoring_day_lcm_l389_38961


namespace NUMINAMATH_GPT_house_number_count_l389_38925

noncomputable def count_valid_house_numbers : Nat :=
  let two_digit_primes := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let valid_combinations := two_digit_primes.product two_digit_primes |>.filter (λ (WX, YZ) => WX ≠ YZ)
  valid_combinations.length

theorem house_number_count : count_valid_house_numbers = 110 :=
  by
    sorry

end NUMINAMATH_GPT_house_number_count_l389_38925


namespace NUMINAMATH_GPT_find_original_number_l389_38987

-- Define the given conditions
def increased_by_twenty_percent (x : ℝ) : ℝ := x * 1.20

-- State the theorem
theorem find_original_number (x : ℝ) (h : increased_by_twenty_percent x = 480) : x = 400 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l389_38987


namespace NUMINAMATH_GPT_smallest_value_div_by_13_l389_38965

theorem smallest_value_div_by_13 : 
  ∃ (A B : ℕ), 
    (0 ≤ A ∧ A ≤ 9) ∧ (0 ≤ B ∧ B ≤ 9) ∧ 
    A ≠ B ∧ 
    1001 * A + 110 * B = 1771 ∧ 
    (1001 * A + 110 * B) % 13 = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_div_by_13_l389_38965


namespace NUMINAMATH_GPT_geometric_sequence_sum_is_five_eighths_l389_38937

noncomputable def geometric_sequence_sum (a₁ : ℝ) (q : ℝ) : ℝ :=
  if q = 1 then 4 * a₁ else a₁ * (1 - q^4) / (1 - q)

theorem geometric_sequence_sum_is_five_eighths
  (a₁ q : ℝ)
  (h₀ : q ≠ 1)
  (h₁ : a₁ * (a₁ * q) * (a₁ * q^2) = -1 / 8)
  (h₂ : 2 * (a₁ * q^2) = a₁ * q + a₁ * q^2) :
  geometric_sequence_sum a₁ q = 5 / 8 := by
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_is_five_eighths_l389_38937


namespace NUMINAMATH_GPT_coprime_exponents_iff_l389_38900

theorem coprime_exponents_iff (p q : ℕ) : 
  Nat.gcd (2^p - 1) (2^q - 1) = 1 ↔ Nat.gcd p q = 1 :=
by 
  sorry

end NUMINAMATH_GPT_coprime_exponents_iff_l389_38900


namespace NUMINAMATH_GPT_owen_profit_l389_38933

/-- Given the initial purchases and sales, calculate Owen's overall profit. -/
theorem owen_profit :
  let boxes_9_dollars := 8
  let boxes_12_dollars := 4
  let cost_9_dollars := 9
  let cost_12_dollars := 12
  let masks_per_box := 50
  let packets_25_pieces := 100
  let price_25_pieces := 5
  let packets_100_pieces := 28
  let price_100_pieces := 12
  let remaining_masks1 := 150
  let price_remaining1 := 3
  let remaining_masks2 := 150
  let price_remaining2 := 4
  let total_cost := (boxes_9_dollars * cost_9_dollars) + (boxes_12_dollars * cost_12_dollars)
  let total_repacked_masks := (packets_25_pieces * price_25_pieces) + (packets_100_pieces * price_100_pieces)
  let total_remaining_masks := (remaining_masks1 * price_remaining1) + (remaining_masks2 * price_remaining2)
  let total_revenue := total_repacked_masks + total_remaining_masks
  let overall_profit := total_revenue - total_cost
  overall_profit = 1766 := by
  sorry

end NUMINAMATH_GPT_owen_profit_l389_38933


namespace NUMINAMATH_GPT_mapleton_math_team_combinations_l389_38922

open Nat

theorem mapleton_math_team_combinations (girls boys : ℕ) (team_size girl_on_team boy_on_team : ℕ)
    (h_girls : girls = 4) (h_boys : boys = 5) (h_team_size : team_size = 4)
    (h_girl_on_team : girl_on_team = 3) (h_boy_on_team : boy_on_team = 1) :
    (Nat.choose girls girl_on_team) * (Nat.choose boys boy_on_team) = 20 := by
  sorry

end NUMINAMATH_GPT_mapleton_math_team_combinations_l389_38922


namespace NUMINAMATH_GPT_cost_per_chair_l389_38986

theorem cost_per_chair (total_spent : ℕ) (chairs_bought : ℕ) (cost : ℕ) 
  (h1 : total_spent = 180) 
  (h2 : chairs_bought = 12) 
  (h3 : cost = total_spent / chairs_bought) : 
  cost = 15 :=
by
  -- Proof steps go here (skipped with sorry)
  sorry

end NUMINAMATH_GPT_cost_per_chair_l389_38986


namespace NUMINAMATH_GPT_min_value_frac_inv_l389_38931

theorem min_value_frac_inv (a b : ℝ) (h1: a > 0) (h2: b > 0) (h3: a + 3 * b = 2) : 
  (2 + Real.sqrt 3) ≤ (1 / a + 1 / b) :=
sorry

end NUMINAMATH_GPT_min_value_frac_inv_l389_38931


namespace NUMINAMATH_GPT_cos_value_l389_38901

variable (α : ℝ)

theorem cos_value (h : Real.sin (π / 4 + α) = 2 / 3) : Real.cos (π / 4 - α) = 2 / 3 := 
by 
  sorry 

end NUMINAMATH_GPT_cos_value_l389_38901


namespace NUMINAMATH_GPT_find_x_l389_38921

theorem find_x
  (PQR_straight : ∀ x y : ℝ, x + y = 76 → 3 * x + 2 * y = 180)
  (h : x + y = 76) :
  x = 28 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l389_38921


namespace NUMINAMATH_GPT_sequence_general_formula_l389_38989

theorem sequence_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 2 = 4 →
  S 4 = 30 →
  (∀ n, n ≥ 2 → a (n + 1) + a (n - 1) = 2 * (a n + 1)) →
  ∀ n, a n = n^2 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l389_38989


namespace NUMINAMATH_GPT_oil_bill_january_l389_38958

-- Declare the constants for January and February oil bills
variables (J F : ℝ)

-- State the conditions
def condition_1 : Prop := F / J = 3 / 2
def condition_2 : Prop := (F + 20) / J = 5 / 3

-- State the theorem based on the conditions and the target statement
theorem oil_bill_january (h1 : condition_1 F J) (h2 : condition_2 F J) : J = 120 :=
by
  sorry

end NUMINAMATH_GPT_oil_bill_january_l389_38958


namespace NUMINAMATH_GPT_kiril_konstantinovich_age_is_full_years_l389_38964

theorem kiril_konstantinovich_age_is_full_years
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  (years = 48) →
  (months = 48) →
  (weeks = 48) →
  (days = 48) →
  (hours = 48) →
  Int.floor (
    years + 
    (months / 12 : ℝ) + 
    (weeks * 7 / 365 : ℝ) + 
    (days / 365 : ℝ) + 
    (hours / (24 * 365) : ℝ)
  ) = 53 :=
by
  intro hyears hmonths hweeks hdays hhours
  rw [hyears, hmonths, hweeks, hdays, hhours]
  sorry

end NUMINAMATH_GPT_kiril_konstantinovich_age_is_full_years_l389_38964


namespace NUMINAMATH_GPT_maximum_sphere_radius_squared_l389_38970

def cone_base_radius : ℝ := 4
def cone_height : ℝ := 10
def axes_intersection_distance_from_base : ℝ := 4

theorem maximum_sphere_radius_squared :
  let m : ℕ := 144
  let n : ℕ := 29
  m + n = 173 :=
by
  sorry

end NUMINAMATH_GPT_maximum_sphere_radius_squared_l389_38970


namespace NUMINAMATH_GPT_exam_max_marks_l389_38928

theorem exam_max_marks (M : ℝ) (h1: 0.30 * M = 66) : M = 220 :=
by
  sorry

end NUMINAMATH_GPT_exam_max_marks_l389_38928


namespace NUMINAMATH_GPT_child_tickets_sold_l389_38914

theorem child_tickets_sold
  (A C : ℕ) 
  (h1 : A + C = 900)
  (h2 : 7 * A + 4 * C = 5100) :
  C = 400 :=
by
  sorry

end NUMINAMATH_GPT_child_tickets_sold_l389_38914


namespace NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l389_38960

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line with the coordinate axes
def forms_triangle_with_axes (x y : ℝ) : Prop := 
  line_eq x 0 ∧ line_eq 0 y

-- Prove the sum of the lengths of the altitudes is 511/17
theorem sum_of_altitudes_of_triangle : 
  ∃ x y : ℝ, forms_triangle_with_axes x y → 
  15 + 8 + (120 / 17) = 511 / 17 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_altitudes_of_triangle_l389_38960


namespace NUMINAMATH_GPT_find_x_l389_38942

theorem find_x (x : ℝ) (h : 3 * x = 36 - x + 16) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l389_38942


namespace NUMINAMATH_GPT_appropriate_weight_design_l389_38934

def weight_design (w_l w_s w_r w_w : ℕ) : Prop :=
  w_l > w_s ∧ w_l > w_w ∧ w_w > w_r ∧ w_s = w_w

theorem appropriate_weight_design :
  weight_design 5 2 1 2 :=
by {
  sorry -- skipped proof
}

end NUMINAMATH_GPT_appropriate_weight_design_l389_38934


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l389_38966

theorem simplify_and_evaluate_expression 
  (a b : ℚ) 
  (ha : a = 2) 
  (hb : b = 1 / 3) : 
  (a / (a - b)) * ((1 / b) - (1 / a)) + ((a - 1) / b) = 6 := 
by
  -- Place the steps verifying this here. For now:
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l389_38966


namespace NUMINAMATH_GPT_arithmetic_sequence_c_d_sum_l389_38979

theorem arithmetic_sequence_c_d_sum (c d : ℕ) 
  (h1 : 10 - 3 = 7) 
  (h2 : 17 - 10 = 7) 
  (h3 : c = 17 + 7) 
  (h4 : d = c + 7) 
  (h5 : d + 7 = 38) :
  c + d = 55 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_c_d_sum_l389_38979


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_monotonicity_l389_38969

theorem sufficient_but_not_necessary_condition_for_monotonicity
  (a : ℕ → ℝ)
  (h_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = a n ^ 2)
  (h_initial : a 1 = 2) :
  (∀ n : ℕ, n > 0 → a n > a 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_monotonicity_l389_38969


namespace NUMINAMATH_GPT_division_remainder_l389_38959

theorem division_remainder :
  ∃ (r : ℝ), ∀ (z : ℝ), (4 * z^3 - 5 * z^2 - 17 * z + 4) = (4 * z + 6) * (z^2 - 4 * z + 1/2) + r ∧ r = 1 :=
sorry

end NUMINAMATH_GPT_division_remainder_l389_38959


namespace NUMINAMATH_GPT_find_x_l389_38983

theorem find_x (x y : ℝ) (h1 : 0.65 * x = 0.20 * y)
  (h2 : y = 617.5 ^ 2 - 42) : 
  x = 117374.3846153846 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l389_38983


namespace NUMINAMATH_GPT_range_of_independent_variable_l389_38998

theorem range_of_independent_variable
  (x : ℝ) 
  (h1 : 2 - 3*x ≥ 0) 
  (h2 : x ≠ 0) 
  : x ≤ 2/3 ∧ x ≠ 0 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_independent_variable_l389_38998


namespace NUMINAMATH_GPT_seats_usually_taken_l389_38996

def total_tables : Nat := 15
def seats_per_table : Nat := 10
def proportion_left_unseated : Rat := 1 / 10
def proportion_taken : Rat := 1 - proportion_left_unseated

theorem seats_usually_taken :
  proportion_taken * (total_tables * seats_per_table) = 135 := by
  sorry

end NUMINAMATH_GPT_seats_usually_taken_l389_38996


namespace NUMINAMATH_GPT_valid_combinations_l389_38950

theorem valid_combinations :
  ∀ (x y z : ℕ), 
  10 ≤ x ∧ x ≤ 20 → 
  10 ≤ y ∧ y ≤ 20 →
  10 ≤ z ∧ z ≤ 20 →
  3 * x^2 - y^2 - 7 * z = 99 →
  (x, y, z) = (15, 10, 12) ∨ (x, y, z) = (16, 12, 11) ∨ (x, y, z) = (18, 15, 13) := 
by
  intros x y z hx hy hz h
  sorry

end NUMINAMATH_GPT_valid_combinations_l389_38950


namespace NUMINAMATH_GPT_incorrect_statement_A_l389_38924

theorem incorrect_statement_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) :
  ¬ (a - a^2 > b - b^2) := sorry

end NUMINAMATH_GPT_incorrect_statement_A_l389_38924


namespace NUMINAMATH_GPT_area_ratio_l389_38910

noncomputable def pentagon_area (R s : ℝ) := (5 / 2) * R * s * Real.sin (Real.pi * 2 / 5)
noncomputable def triangle_area (s : ℝ) := (s^2) / 4

theorem area_ratio (R s : ℝ) (h : R = s / (2 * Real.sin (Real.pi / 5))) :
  (pentagon_area R s) / (triangle_area s) = 5 * (Real.sin ((2 * Real.pi) / 5) / Real.sin (Real.pi / 5)) :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_l389_38910


namespace NUMINAMATH_GPT_train_journey_time_l389_38962

theorem train_journey_time {X : ℝ} (h1 : 0 < X) (h2 : X < 60) (h3 : ∀ T_A M_A T_B M_B : ℝ, M_A - T_A = X ∧ M_B - T_B = X) :
    X = 360 / 7 :=
by
  sorry

end NUMINAMATH_GPT_train_journey_time_l389_38962


namespace NUMINAMATH_GPT_count_three_digit_perfect_squares_l389_38993

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem count_three_digit_perfect_squares : 
  ∃ (count : ℕ), count = 22 ∧
  ∀ (n : ℕ), is_three_digit_number n → is_perfect_square n → true :=
sorry

end NUMINAMATH_GPT_count_three_digit_perfect_squares_l389_38993


namespace NUMINAMATH_GPT_stock_price_is_108_l389_38911

noncomputable def dividend_income (FV : ℕ) (D : ℕ) : ℕ :=
  FV * D / 100

noncomputable def face_value_of_stock (I : ℕ) (D : ℕ) : ℕ :=
  I * 100 / D

noncomputable def price_of_stock (Inv : ℕ) (FV : ℕ) : ℕ :=
  Inv * 100 / FV

theorem stock_price_is_108 (I D Inv : ℕ) (hI : I = 450) (hD : D = 10) (hInv : Inv = 4860) :
  price_of_stock Inv (face_value_of_stock I D) = 108 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_stock_price_is_108_l389_38911


namespace NUMINAMATH_GPT_distance_between_lines_l389_38974

/-- Define the lines by their equations -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 6 = 0

/-- Define the simplified form of the second line -/
def simplified_line2 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

/-- Prove the distance between the two lines is 3 -/
theorem distance_between_lines : 
  let A : ℝ := 3
  let B : ℝ := 4
  let C1 : ℝ := -12
  let C2 : ℝ := 3
  (|C2 - C1| / Real.sqrt (A^2 + B^2) = 3) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_lines_l389_38974


namespace NUMINAMATH_GPT_jim_gave_away_cards_l389_38907

theorem jim_gave_away_cards
  (sets_brother : ℕ := 15)
  (sets_sister : ℕ := 8)
  (sets_friend : ℕ := 4)
  (sets_cousin : ℕ := 6)
  (sets_classmate : ℕ := 3)
  (cards_per_set : ℕ := 25) :
  (sets_brother + sets_sister + sets_friend + sets_cousin + sets_classmate) * cards_per_set = 900 :=
by
  sorry

end NUMINAMATH_GPT_jim_gave_away_cards_l389_38907


namespace NUMINAMATH_GPT_find_C_l389_38957

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 320) : 
  C = 20 := 
by 
  sorry

end NUMINAMATH_GPT_find_C_l389_38957


namespace NUMINAMATH_GPT_relatively_prime_divisibility_l389_38948

theorem relatively_prime_divisibility (x y : ℕ) (h1 : Nat.gcd x y = 1) (h2 : y^2 * (y - x)^2 ∣ x^2 * (x + y)) :
  (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_relatively_prime_divisibility_l389_38948


namespace NUMINAMATH_GPT_Robert_books_read_in_six_hours_l389_38999

theorem Robert_books_read_in_six_hours (P H T: ℕ)
    (h1: P = 270)
    (h2: H = 90)
    (h3: T = 6):
    T * H / P = 2 :=
by 
    -- sorry placeholder to indicate that this is where the proof goes.
    sorry

end NUMINAMATH_GPT_Robert_books_read_in_six_hours_l389_38999


namespace NUMINAMATH_GPT_dressing_p_percentage_l389_38918

-- Define the percentages of vinegar and oil in dressings p and q
def vinegar_in_p : ℝ := 0.30
def vinegar_in_q : ℝ := 0.10

-- Define the desired percentage of vinegar in the new dressing
def vinegar_in_new_dressing : ℝ := 0.12

-- Define the total mass of the new dressing
def total_mass_new_dressing : ℝ := 100.0

-- Define the mass of dressing p in the new dressing
def mass_of_p (x : ℝ) : ℝ := x

-- Define the mass of dressing q in the new dressing
def mass_of_q (x : ℝ) : ℝ := total_mass_new_dressing - x

-- Define the amount of vinegar contributed by dressings p and q
def vinegar_from_p (x : ℝ) : ℝ := vinegar_in_p * mass_of_p x
def vinegar_from_q (x : ℝ) : ℝ := vinegar_in_q * mass_of_q x

-- Define the total vinegar in the new dressing
def total_vinegar (x : ℝ) : ℝ := vinegar_from_p x + vinegar_from_q x

-- Problem statement: prove the percentage of dressing p in the new dressing
theorem dressing_p_percentage (x : ℝ) (hx : total_vinegar x = vinegar_in_new_dressing * total_mass_new_dressing) :
  (mass_of_p x / total_mass_new_dressing) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_dressing_p_percentage_l389_38918


namespace NUMINAMATH_GPT_player_1_winning_strategy_l389_38968

-- Define the properties and rules of the game
def valid_pair (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a + b = 2005

def move (current t a b : ℕ) : Prop := 
  current = t - a ∨ current = t - b

def first_player_wins (t a b : ℕ) : Prop :=
  ∀ k : ℕ, t > k * 2005 → ∃ m : ℕ, move (t - m) t a b

-- Main theorem statement
theorem player_1_winning_strategy : ∃ (t : ℕ) (a b : ℕ), valid_pair a b ∧ first_player_wins t a b :=
sorry

end NUMINAMATH_GPT_player_1_winning_strategy_l389_38968


namespace NUMINAMATH_GPT_fraction_of_area_in_triangle_l389_38913

theorem fraction_of_area_in_triangle :
  let vertex1 := (3, 3)
  let vertex2 := (5, 5)
  let vertex3 := (3, 5)
  let base := (5 - 3)
  let height := (5 - 3)
  let area_triangle := (1 / 2) * base * height
  let area_square := 6 * 6
  let fraction := area_triangle / area_square
  fraction = (1 / 18) :=
by 
  sorry

end NUMINAMATH_GPT_fraction_of_area_in_triangle_l389_38913


namespace NUMINAMATH_GPT_circumscribed_circle_diameter_l389_38949

theorem circumscribed_circle_diameter (a : ℝ) (A : ℝ) (h_a : a = 16) (h_A : A = 30) :
    let D := a / Real.sin (A * Real.pi / 180)
    D = 32 := by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_diameter_l389_38949


namespace NUMINAMATH_GPT_pizza_slices_left_l389_38908

theorem pizza_slices_left (initial_slices : ℕ) (people : ℕ) (slices_per_person : ℕ) 
  (h1 : initial_slices = 16) (h2 : people = 6) (h3 : slices_per_person = 2) : 
  initial_slices - people * slices_per_person = 4 := 
by
  sorry

end NUMINAMATH_GPT_pizza_slices_left_l389_38908


namespace NUMINAMATH_GPT_long_show_episode_duration_is_one_hour_l389_38976

-- Definitions for the given conditions
def total_shows : ℕ := 2
def short_show_length : ℕ := 24
def short_show_episode_duration : ℝ := 0.5
def long_show_episodes : ℕ := 12
def total_viewing_time : ℝ := 24

-- Definition of the length of each episode of the longer show
def long_show_episode_length (L : ℝ) : Prop :=
  (short_show_length * short_show_episode_duration) + (long_show_episodes * L) = total_viewing_time

-- Main statement to prove
theorem long_show_episode_duration_is_one_hour : long_show_episode_length 1 :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_long_show_episode_duration_is_one_hour_l389_38976


namespace NUMINAMATH_GPT_comparison_b_a_c_l389_38953

noncomputable def a : ℝ := Real.sqrt 1.2
noncomputable def b : ℝ := Real.exp 0.1
noncomputable def c : ℝ := 1 + Real.log 1.1

theorem comparison_b_a_c : b > a ∧ a > c :=
by
  unfold a b c
  sorry

end NUMINAMATH_GPT_comparison_b_a_c_l389_38953


namespace NUMINAMATH_GPT_find_xy_l389_38945

theorem find_xy (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l389_38945


namespace NUMINAMATH_GPT_laptop_full_price_l389_38947

theorem laptop_full_price (p : ℝ) (deposit : ℝ) (h1 : deposit = 0.25 * p) (h2 : deposit = 400) : p = 1600 :=
by
  sorry

end NUMINAMATH_GPT_laptop_full_price_l389_38947


namespace NUMINAMATH_GPT_value_of_x_plus_y_squared_l389_38981

theorem value_of_x_plus_y_squared (x y : ℝ) 
  (h₁ : x^2 + y^2 = 20) 
  (h₂ : x * y = 6) : 
  (x + y)^2 = 32 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_squared_l389_38981


namespace NUMINAMATH_GPT_smallest_a_exists_l389_38975

theorem smallest_a_exists : ∃ a b c : ℤ, a > 0 ∧ b^2 > 4*a*c ∧ 
  (∀ x : ℝ, x > 0 ∧ x < 1 → (a * x^2 - b * x + c) = 0 → false) 
  ∧ a = 5 :=
by sorry

end NUMINAMATH_GPT_smallest_a_exists_l389_38975
