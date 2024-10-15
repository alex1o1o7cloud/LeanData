import Mathlib

namespace NUMINAMATH_GPT_distance_reflection_x_axis_l830_83026

/--
Given points C and its reflection over the x-axis C',
prove that the distance between C and C' is 6.
-/
theorem distance_reflection_x_axis :
  let C := (-2, 3)
  let C' := (-2, -3)
  dist C C' = 6 := by
  sorry

end NUMINAMATH_GPT_distance_reflection_x_axis_l830_83026


namespace NUMINAMATH_GPT_third_term_arithmetic_sequence_l830_83047

variable (a d : ℤ)
variable (h1 : a + 20 * d = 12)
variable (h2 : a + 21 * d = 15)

theorem third_term_arithmetic_sequence : a + 2 * d = -42 := by
  sorry

end NUMINAMATH_GPT_third_term_arithmetic_sequence_l830_83047


namespace NUMINAMATH_GPT_midpoint_of_segment_l830_83043

theorem midpoint_of_segment (a b : ℝ) : (a + b) / 2 = (a + b) / 2 :=
sorry

end NUMINAMATH_GPT_midpoint_of_segment_l830_83043


namespace NUMINAMATH_GPT_range_of_m_l830_83064

open Set

noncomputable def A (m : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x^2 + m * x - y + 2 = 0} 

noncomputable def B : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ x - y + 1 = 0}

theorem range_of_m (m : ℝ) : (A m ∩ B ≠ ∅) → (m ≤ -1 ∨ m ≥ 3) := 
sorry

end NUMINAMATH_GPT_range_of_m_l830_83064


namespace NUMINAMATH_GPT_complex_solution_l830_83004

theorem complex_solution (z : ℂ) (h : z * (0 + 1 * I) = (0 + 1 * I) - 1) : z = 1 + I :=
by
  sorry

end NUMINAMATH_GPT_complex_solution_l830_83004


namespace NUMINAMATH_GPT_max_value_of_x_minus_y_l830_83031

theorem max_value_of_x_minus_y
  (x y : ℝ)
  (h : 2 * (x ^ 2 + y ^ 2 - x * y) = x + y) :
  x - y ≤ 1 / 2 := 
sorry

end NUMINAMATH_GPT_max_value_of_x_minus_y_l830_83031


namespace NUMINAMATH_GPT_ones_digit_seven_consecutive_integers_l830_83048

theorem ones_digit_seven_consecutive_integers (k : ℕ) (hk : k % 5 = 1) :
  (k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_seven_consecutive_integers_l830_83048


namespace NUMINAMATH_GPT_initial_minutes_under_plan_A_l830_83097

theorem initial_minutes_under_plan_A (x : ℕ) (planA_initial : ℝ) (planA_rate : ℝ) (planB_rate : ℝ) (call_duration : ℕ) :
  planA_initial = 0.60 ∧ planA_rate = 0.06 ∧ planB_rate = 0.08 ∧ call_duration = 3 ∧
  (planA_initial + planA_rate * (call_duration - x) = planB_rate * call_duration) →
  x = 9 := 
by
  intros h
  obtain ⟨h1, h2, h3, h4, heq⟩ := h
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_initial_minutes_under_plan_A_l830_83097


namespace NUMINAMATH_GPT_find_a_sq_plus_b_sq_l830_83044

theorem find_a_sq_plus_b_sq (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) :
  a^2 + b^2 = 29 := by
  sorry

end NUMINAMATH_GPT_find_a_sq_plus_b_sq_l830_83044


namespace NUMINAMATH_GPT_sin2theta_cos2theta_sum_l830_83046

theorem sin2theta_cos2theta_sum (θ : ℝ) (h1 : Real.sin θ = 2 * Real.cos θ) (h2 : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1) : 
  Real.sin (2 * θ) + Real.cos (2 * θ) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin2theta_cos2theta_sum_l830_83046


namespace NUMINAMATH_GPT_total_money_made_l830_83040

structure Building :=
(floors : Nat)
(rooms_per_floor : Nat)

def cleaning_time_per_room : Nat := 8

structure CleaningRates :=
(first_4_hours_rate : Int)
(next_4_hours_rate : Int)
(unpaid_break_hours : Nat)

def supply_cost : Int := 1200

def total_earnings (b : Building) (c : CleaningRates) : Int :=
  let rooms := b.floors * b.rooms_per_floor
  let earnings_per_room := (4 * c.first_4_hours_rate + 4 * c.next_4_hours_rate)
  rooms * earnings_per_room - supply_cost

theorem total_money_made (b : Building) (c : CleaningRates) : 
  b.floors = 12 →
  b.rooms_per_floor = 25 →
  cleaning_time_per_room = 8 →
  c.first_4_hours_rate = 20 →
  c.next_4_hours_rate = 25 →
  c.unpaid_break_hours = 1 →
  total_earnings b c = 52800 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_money_made_l830_83040


namespace NUMINAMATH_GPT_initial_members_count_l830_83016

theorem initial_members_count (n : ℕ) (W : ℕ)
  (h1 : W = n * 48)
  (h2 : W + 171 = (n + 2) * 51) : 
  n = 23 :=
by sorry

end NUMINAMATH_GPT_initial_members_count_l830_83016


namespace NUMINAMATH_GPT_frog_probability_0_4_l830_83042

-- Definitions and conditions
def vertices : List (ℤ × ℤ) := [(1,1), (1,6), (5,6), (5,1)]
def start_position : ℤ × ℤ := (2,3)

-- Probabilities for transition, boundary definitions, this mimics the recursive nature described
def P : ℤ × ℤ → ℝ
| (x, 1) => 1   -- Boundary condition for horizontal sides
| (x, 6) => 1   -- Boundary condition for horizontal sides
| (1, y) => 0   -- Boundary condition for vertical sides
| (5, y) => 0   -- Boundary condition for vertical sides
| (x, y) => sorry  -- General case for other positions

-- The theorem to prove
theorem frog_probability_0_4 : P (2, 3) = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_frog_probability_0_4_l830_83042


namespace NUMINAMATH_GPT_problem_provable_l830_83052

noncomputable def given_expression (a : ℝ) : ℝ :=
  (1 / (a + 2)) / ((a^2 - 4 * a + 4) / (a^2 - 4)) - (2 / (a - 2))

theorem problem_provable : given_expression (Real.sqrt 5 + 2) = - (Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_problem_provable_l830_83052


namespace NUMINAMATH_GPT_diane_trip_length_l830_83011

-- Define constants and conditions
def first_segment_fraction : ℚ := 1 / 4
def middle_segment_length : ℚ := 24
def last_segment_fraction : ℚ := 1 / 3

def total_trip_length (x : ℚ) : Prop :=
  (1 - first_segment_fraction - last_segment_fraction) * x = middle_segment_length

theorem diane_trip_length : ∃ x : ℚ, total_trip_length x ∧ x = 57.6 := by
  sorry

end NUMINAMATH_GPT_diane_trip_length_l830_83011


namespace NUMINAMATH_GPT_area_of_large_square_l830_83002

theorem area_of_large_square (s : ℝ) (h : 2 * s^2 = 14) : 9 * s^2 = 63 := by
  sorry

end NUMINAMATH_GPT_area_of_large_square_l830_83002


namespace NUMINAMATH_GPT_part_1_part_2_l830_83073

-- Definitions for sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 3}
def N (m : ℝ) : Set ℝ := {x | x ≥ m}

-- Proof problem 1: Prove that if M ∪ N = N, then m ≤ -2
theorem part_1 (m : ℝ) : (M ∪ N m = N m) → m ≤ -2 :=
by sorry

-- Proof problem 2: Prove that if M ∩ N = ∅, then m ≥ 3
theorem part_2 (m : ℝ) : (M ∩ N m = ∅) → m ≥ 3 :=
by sorry

end NUMINAMATH_GPT_part_1_part_2_l830_83073


namespace NUMINAMATH_GPT_integral_value_l830_83030

noncomputable def integral_sin_pi_over_2_to_pi : ℝ := ∫ x in (Real.pi / 2)..Real.pi, Real.sin x

theorem integral_value : integral_sin_pi_over_2_to_pi = 1 := by
  sorry

end NUMINAMATH_GPT_integral_value_l830_83030


namespace NUMINAMATH_GPT_paint_cans_used_l830_83090

theorem paint_cans_used (init_rooms : ℕ) (lost_cans : ℕ) (remaining_rooms : ℕ) (final_rooms : ℕ) :
  init_rooms = 50 → lost_cans = 5 → remaining_rooms = 40 → final_rooms = 40 → 
  remaining_rooms / (lost_cans / (init_rooms - remaining_rooms)) = 20 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_paint_cans_used_l830_83090


namespace NUMINAMATH_GPT_cone_radius_l830_83082

theorem cone_radius
  (l : ℝ) (CSA : ℝ) (π : ℝ) (r : ℝ)
  (h_l : l = 15)
  (h_CSA : CSA = 141.3716694115407)
  (h_pi : π = Real.pi) :
  r = 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_radius_l830_83082


namespace NUMINAMATH_GPT_annual_decrease_rate_l830_83098

theorem annual_decrease_rate :
  ∀ (P₀ P₂ : ℕ) (t : ℕ) (rate : ℝ),
    P₀ = 20000 → P₂ = 12800 → t = 2 → P₂ = P₀ * (1 - rate) ^ t → rate = 0.2 :=
by
sorry

end NUMINAMATH_GPT_annual_decrease_rate_l830_83098


namespace NUMINAMATH_GPT_smallest_b_for_fraction_eq_l830_83027

theorem smallest_b_for_fraction_eq (a b : ℕ) (h1 : 1000 ≤ a ∧ a < 10000) (h2 : 100000 ≤ b ∧ b < 1000000)
(h3 : 1/2006 = 1/a + 1/b) : b = 120360 := sorry

end NUMINAMATH_GPT_smallest_b_for_fraction_eq_l830_83027


namespace NUMINAMATH_GPT_exist_non_special_symmetric_concat_l830_83007

-- Define the notion of a binary series being symmetric
def is_symmetric (xs : List Bool) : Prop :=
  ∀ i, i < xs.length → xs.get? i = xs.get? (xs.length - 1 - i)

-- Define the notion of a binary series being special
def is_special (xs : List Bool) : Prop :=
  (∀ x ∈ xs, x) ∨ (∀ x ∈ xs, ¬x)

-- The main theorem statement
theorem exist_non_special_symmetric_concat (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) :
  ∃ (A B : List Bool), A.length = m ∧ B.length = n ∧ ¬is_special A ∧ ¬is_special B ∧ is_symmetric (A ++ B) :=
sorry

end NUMINAMATH_GPT_exist_non_special_symmetric_concat_l830_83007


namespace NUMINAMATH_GPT_slope_angle_of_perpendicular_line_l830_83068

theorem slope_angle_of_perpendicular_line (h : ∀ x, x = (π / 3)) : ∀ θ, θ = (π / 2) := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_slope_angle_of_perpendicular_line_l830_83068


namespace NUMINAMATH_GPT_gcd_problem_l830_83038

theorem gcd_problem 
  (b : ℤ) 
  (hb_odd : b % 2 = 1) 
  (hb_multiples_of_8723 : ∃ (k : ℤ), b = 8723 * k) : 
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 15) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_problem_l830_83038


namespace NUMINAMATH_GPT_surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l830_83089

-- First Problem:
theorem surface_area_cone_first_octant :
  ∃ (surface_area : ℝ), 
    (∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ z^2 = 2*x*y) → surface_area = 16 :=
sorry

-- Second Problem:
theorem surface_area_sphere_inside_cylinder (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 ∧ x^2 + y^2 = R*x) → surface_area = 2 * R^2 * (π - 2) :=
sorry

-- Third Problem:
theorem surface_area_cylinder_inside_sphere (R : ℝ) :
  ∃ (surface_area : ℝ), 
    (∀ (x y z : ℝ), x^2 + y^2 = R*x ∧ x^2 + y^2 + z^2 = R^2) → surface_area = 4 * R^2 :=
sorry

end NUMINAMATH_GPT_surface_area_cone_first_octant_surface_area_sphere_inside_cylinder_surface_area_cylinder_inside_sphere_l830_83089


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l830_83099

theorem arithmetic_sequence_sum {a : ℕ → ℝ} (d a1 : ℝ)
  (h_arith: ∀ n, a n = a1 + (n - 1) * d)
  (h_condition: a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_sum_l830_83099


namespace NUMINAMATH_GPT_cannot_eat_166_candies_l830_83072

-- Define parameters for sandwiches and candies equations
def sandwiches_eq (x y z : ℕ) := x + 2 * y + 3 * z = 100
def candies_eq (x y z : ℕ) := 3 * x + 4 * y + 5 * z = 166

theorem cannot_eat_166_candies (x y z : ℕ) : ¬ (sandwiches_eq x y z ∧ candies_eq x y z) :=
by {
  -- Proof will show impossibility of (x, y, z) as nonnegative integers solution
  sorry
}

end NUMINAMATH_GPT_cannot_eat_166_candies_l830_83072


namespace NUMINAMATH_GPT_monthly_profit_10000_daily_profit_15000_maximize_profit_l830_83095

noncomputable def price_increase (c p: ℕ) (x: ℕ) : ℕ := c + x - p
noncomputable def sales_volume (s d: ℕ) (x: ℕ) : ℕ := s - d * x
noncomputable def monthly_profit (price cost volume: ℕ) : ℕ := (price - cost) * volume
noncomputable def monthly_profit_equation (x: ℕ) : ℕ := (40 + x - 30) * (600 - 10 * x)

theorem monthly_profit_10000 (x: ℕ) : monthly_profit_equation x = 10000 ↔ x = 10 ∨ x = 40 :=
by sorry

theorem daily_profit_15000 (x: ℕ) : ¬∃ x, monthly_profit_equation x = 15000 :=
by sorry

theorem maximize_profit (x p y: ℕ) : (∀ x, monthly_profit (40 + x) 30 (600 - 10 * x) ≤ y) ∧ y = 12250 ∧ x = 65 :=
by sorry

end NUMINAMATH_GPT_monthly_profit_10000_daily_profit_15000_maximize_profit_l830_83095


namespace NUMINAMATH_GPT_polygon_RS_ST_sum_l830_83091

theorem polygon_RS_ST_sum
  (PQ RS ST: ℝ)
  (PQ_eq : PQ = 10)
  (QR_eq : QR = 7)
  (TU_eq : TU = 6)
  (polygon_area : PQ * QR = 70)
  (PQRSTU_area : 70 = 70) :
  RS + ST = 80 :=
by
  sorry

end NUMINAMATH_GPT_polygon_RS_ST_sum_l830_83091


namespace NUMINAMATH_GPT_no_square_ends_with_four_identical_digits_except_0_l830_83083

theorem no_square_ends_with_four_identical_digits_except_0 (n : ℤ) :
  ¬ (∃ k : ℕ, (1 ≤ k ∧ k < 10) ∧ (n^2 % 10000 = k * 1111)) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_square_ends_with_four_identical_digits_except_0_l830_83083


namespace NUMINAMATH_GPT_part1_proof_l830_83019

variable (a r : ℝ) (f : ℝ → ℝ)

axiom a_gt_1 : a > 1
axiom r_gt_1 : r > 1

axiom f_condition : ∀ x > 0, f x * f x ≤ a * x * f (x / a)
axiom f_bound : ∀ x, 0 < x ∧ x < 1 / 2^2005 → f x < 2^2005

theorem part1_proof : ∀ x > 0, f x ≤ a^(1 - r) * x := 
by 
  sorry

end NUMINAMATH_GPT_part1_proof_l830_83019


namespace NUMINAMATH_GPT_eval_expression_l830_83024

theorem eval_expression : 3 * (3 + 3) / 3 = 6 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l830_83024


namespace NUMINAMATH_GPT_final_number_is_correct_l830_83051

def initial_number := 9
def doubled_number (x : ℕ) := x * 2
def added_number (x : ℕ) := x + 13
def trebled_number (x : ℕ) := x * 3

theorem final_number_is_correct : trebled_number (added_number (doubled_number initial_number)) = 93 := by
  sorry

end NUMINAMATH_GPT_final_number_is_correct_l830_83051


namespace NUMINAMATH_GPT_fraction_sum_l830_83087

theorem fraction_sum : (3 / 8) + (9 / 12) + (5 / 6) = 47 / 24 := by
  sorry

end NUMINAMATH_GPT_fraction_sum_l830_83087


namespace NUMINAMATH_GPT_natural_number_1981_l830_83012

theorem natural_number_1981 (x : ℕ) 
  (h1 : ∃ a : ℕ, x - 45 = a^2)
  (h2 : ∃ b : ℕ, x + 44 = b^2) :
  x = 1981 :=
sorry

end NUMINAMATH_GPT_natural_number_1981_l830_83012


namespace NUMINAMATH_GPT_pencils_remaining_in_drawer_l830_83028

-- Definitions of the conditions
def total_pencils_initially : ℕ := 34
def pencils_taken : ℕ := 22

-- The theorem statement with the correct answer
theorem pencils_remaining_in_drawer : total_pencils_initially - pencils_taken = 12 :=
by
  sorry

end NUMINAMATH_GPT_pencils_remaining_in_drawer_l830_83028


namespace NUMINAMATH_GPT_fraction_spent_on_furniture_l830_83013

variable (original_savings : ℕ)
variable (tv_cost : ℕ)
variable (f : ℚ)

-- Defining the conditions
def conditions := original_savings = 500 ∧ tv_cost = 100 ∧
  f = (original_savings - tv_cost) / original_savings

-- The theorem we want to prove
theorem fraction_spent_on_furniture : conditions original_savings tv_cost f → f = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_spent_on_furniture_l830_83013


namespace NUMINAMATH_GPT_relationship_m_n_k_l_l830_83094

-- Definitions based on the conditions
variables (m n k l : ℕ)

-- Condition: Number of teachers (m), Number of students (n)
-- Each teacher teaches exactly k students
-- Any pair of students has exactly l common teachers

theorem relationship_m_n_k_l (h1 : 0 < m) (h2 : 0 < n) (h3 : 0 < k) (h4 : 0 < l)
  (hk : k * (k - 1) / 2 = k * (k - 1) / 2) (hl : n * (n - 1) / 2 = n * (n - 1) / 2) 
  (h5 : m * (k * (k - 1)) = (n * (n - 1)) * l) :
  m * k * (k - 1) = n * (n - 1) * l :=
by 
  sorry

end NUMINAMATH_GPT_relationship_m_n_k_l_l830_83094


namespace NUMINAMATH_GPT_prop_neg_or_not_l830_83022

theorem prop_neg_or_not (p q : Prop) (h : ¬(p ∨ ¬ q)) : ¬ p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_prop_neg_or_not_l830_83022


namespace NUMINAMATH_GPT_largest_sum_of_three_faces_l830_83017

theorem largest_sum_of_three_faces (faces : Fin 6 → ℕ)
  (h_unique : ∀ i j, i ≠ j → faces i ≠ faces j)
  (h_range : ∀ i, 1 ≤ faces i ∧ faces i ≤ 6)
  (h_opposite_sum : ∀ i, faces i + faces (5 - i) = 10) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ faces i + faces j + faces k = 12 :=
by sorry

end NUMINAMATH_GPT_largest_sum_of_three_faces_l830_83017


namespace NUMINAMATH_GPT_star_equiv_zero_l830_83010

-- Define the new operation for real numbers a and b
def star (a b : ℝ) : ℝ := (a^2 - b^2)^2

-- Prove that (x^2 - y^2) star (y^2 - x^2) equals 0
theorem star_equiv_zero (x y : ℝ) : star (x^2 - y^2) (y^2 - x^2) = 0 := 
by sorry

end NUMINAMATH_GPT_star_equiv_zero_l830_83010


namespace NUMINAMATH_GPT_find_AD_l830_83021

noncomputable def A := 0
noncomputable def C := 3
noncomputable def B (x : ℝ) := C - x
noncomputable def D (x : ℝ) := A + 3 + x

-- conditions
def AC := 3
def BD := 4
def ratio_condition (x : ℝ) := (A + C - x - (A + 3)) / x = (A + 3 + x) / x

-- theorem statement
theorem find_AD (x : ℝ) (h1 : AC = 3) (h2 : BD = 4) (h3 : ratio_condition x) :
  D x = 6 :=
sorry

end NUMINAMATH_GPT_find_AD_l830_83021


namespace NUMINAMATH_GPT_length_of_goods_train_l830_83020

/-- The length of the goods train given the conditions of the problem --/
theorem length_of_goods_train
  (speed_passenger_train : ℝ) (speed_goods_train : ℝ) 
  (time_taken_to_pass : ℝ) (length_goods_train : ℝ) :
  speed_passenger_train = 80 / 3.6 →  -- Convert 80 km/h to m/s
  speed_goods_train    = 32 / 3.6 →  -- Convert 32 km/h to m/s
  time_taken_to_pass   = 9 →
  length_goods_train   = 280 → 
  length_goods_train = (speed_passenger_train + speed_goods_train) * time_taken_to_pass := by
    sorry

end NUMINAMATH_GPT_length_of_goods_train_l830_83020


namespace NUMINAMATH_GPT_swim_ratio_l830_83059

theorem swim_ratio
  (V_m : ℝ) (h1 : V_m = 4.5)
  (V_s : ℝ) (h2 : V_s = 1.5)
  (V_u : ℝ) (h3 : V_u = V_m - V_s)
  (V_d : ℝ) (h4 : V_d = V_m + V_s)
  (T_u T_d : ℝ) (h5 : T_u / T_d = V_d / V_u) :
  T_u / T_d = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_swim_ratio_l830_83059


namespace NUMINAMATH_GPT_evaluate_polynomial_at_5_l830_83037

def polynomial (x : ℕ) : ℕ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem evaluate_polynomial_at_5 : polynomial 5 = 7548 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_5_l830_83037


namespace NUMINAMATH_GPT_max_value_of_fraction_l830_83025

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_fraction_l830_83025


namespace NUMINAMATH_GPT_find_a1_a10_value_l830_83034

variable {α : Type} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∃ r : α, ∀ n : ℕ, a (n + 1) = a n * r

theorem find_a1_a10_value (a : ℕ → α) (h1 : is_geometric_sequence a)
    (h2 : a 4 + a 7 = 2) (h3 : a 5 * a 6 = -8) : a 1 + a 10 = -7 := by
  sorry

end NUMINAMATH_GPT_find_a1_a10_value_l830_83034


namespace NUMINAMATH_GPT_conversion_correct_l830_83054

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.reverse.enum.foldl (λ acc ⟨i, digit⟩ => acc + digit * 2^i) 0

def n : List ℕ := [1, 0, 1, 1, 1, 1, 0, 1, 1]

theorem conversion_correct :
  binary_to_decimal n = 379 :=
by 
  sorry

end NUMINAMATH_GPT_conversion_correct_l830_83054


namespace NUMINAMATH_GPT_sum_of_squares_and_product_l830_83067

open Real

theorem sum_of_squares_and_product (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h1 : x^2 + y^2 = 325) (h2 : x * y = 120) :
    x + y = Real.sqrt 565 := by
  sorry

end NUMINAMATH_GPT_sum_of_squares_and_product_l830_83067


namespace NUMINAMATH_GPT_brenda_sally_track_length_l830_83063

theorem brenda_sally_track_length
  (c d : ℝ) 
  (h1 : c / 4 * 3 = d) 
  (h2 : d - 120 = 0.75 * c - 120) 
  (h3 : 0.75 * c + 60 <= 1.25 * c - 180) 
  (h4 : (c - 120 + 0.25 * c - 60) = 1.25 * c - 180):
  c = 766.67 :=
sorry

end NUMINAMATH_GPT_brenda_sally_track_length_l830_83063


namespace NUMINAMATH_GPT_number_of_students_earning_B_l830_83001

variables (a b c : ℕ) -- since we assume we only deal with whole numbers

-- Given conditions:
-- 1. The probability of earning an A is twice the probability of earning a B.
axiom h1 : a = 2 * b
-- 2. The probability of earning a C is equal to the probability of earning a B.
axiom h2 : c = b
-- 3. The only grades are A, B, or C and there are 45 students in the class.
axiom h3 : a + b + c = 45

-- Prove that the number of students earning a B is 11.
theorem number_of_students_earning_B : b = 11 :=
by
    sorry

end NUMINAMATH_GPT_number_of_students_earning_B_l830_83001


namespace NUMINAMATH_GPT_max_value_sum_seq_l830_83055

theorem max_value_sum_seq : 
  ∃ a1 a2 a3 a4 : ℝ, 
    a1 = 0 ∧ 
    |a2| = |a1 - 1| ∧ 
    |a3| = |a2 - 1| ∧ 
    |a4| = |a3 - 1| ∧ 
    a1 + a2 + a3 + a4 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_max_value_sum_seq_l830_83055


namespace NUMINAMATH_GPT_overall_average_correct_l830_83009

noncomputable def overall_average : ℝ :=
  let students1 := 60
  let students2 := 35
  let students3 := 45
  let students4 := 42
  let avgMarks1 := 50
  let avgMarks2 := 60
  let avgMarks3 := 55
  let avgMarks4 := 45
  let total_students := students1 + students2 + students3 + students4
  let total_marks := (students1 * avgMarks1) + (students2 * avgMarks2) + (students3 * avgMarks3) + (students4 * avgMarks4)
  total_marks / total_students

theorem overall_average_correct : overall_average = 52.00 := by
  sorry

end NUMINAMATH_GPT_overall_average_correct_l830_83009


namespace NUMINAMATH_GPT_right_triangle_area_l830_83045

theorem right_triangle_area (a_square_area b_square_area hypotenuse_square_area : ℝ)
  (ha : a_square_area = 36) (hb : b_square_area = 64) (hc : hypotenuse_square_area = 100)
  (leg1 leg2 hypotenuse : ℝ)
  (hleg1 : leg1 * leg1 = a_square_area)
  (hleg2 : leg2 * leg2 = b_square_area)
  (hhyp : hypotenuse * hypotenuse = hypotenuse_square_area) :
  (1/2) * leg1 * leg2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_area_l830_83045


namespace NUMINAMATH_GPT_hardey_fitness_center_ratio_l830_83075

theorem hardey_fitness_center_ratio
  (f m : ℕ)
  (avg_female_weight : ℕ := 140)
  (avg_male_weight : ℕ := 180)
  (avg_overall_weight : ℕ := 160)
  (h1 : avg_female_weight * f + avg_male_weight * m = avg_overall_weight * (f + m)) :
  f = m :=
by
  sorry

end NUMINAMATH_GPT_hardey_fitness_center_ratio_l830_83075


namespace NUMINAMATH_GPT_range_of_b_over_a_l830_83049

-- Define the problem conditions and conclusion
theorem range_of_b_over_a 
  (a b c : ℝ) (A B C : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) 
  (h_sum_angles : A + B + C = π) 
  (h_sides_relation : ∀ x, (x^2 + c^2 - a^2 - ab = 0 ↔ x = 0)) : 
  1 < b / a ∧ b / a < 2 := 
sorry

end NUMINAMATH_GPT_range_of_b_over_a_l830_83049


namespace NUMINAMATH_GPT_store_revenue_is_1210_l830_83057

noncomputable def shirt_price : ℕ := 10
noncomputable def jeans_price : ℕ := 2 * shirt_price
noncomputable def jacket_price : ℕ := 3 * jeans_price
noncomputable def discounted_jacket_price : ℕ := jacket_price - (jacket_price / 10)

noncomputable def total_revenue : ℕ :=
  20 * shirt_price + 10 * jeans_price + 15 * discounted_jacket_price

theorem store_revenue_is_1210 :
  total_revenue = 1210 :=
by
  sorry

end NUMINAMATH_GPT_store_revenue_is_1210_l830_83057


namespace NUMINAMATH_GPT_remainder_div_P_by_D_plus_D_l830_83033

theorem remainder_div_P_by_D_plus_D' 
  (P Q D R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D + D') = R :=
by
  -- Proof is not required.
  sorry

end NUMINAMATH_GPT_remainder_div_P_by_D_plus_D_l830_83033


namespace NUMINAMATH_GPT_remaining_homes_proof_l830_83006

-- Define the total number of homes
def total_homes : ℕ := 200

-- Distributed homes after the first hour
def homes_distributed_first_hour : ℕ := (2 * total_homes) / 5

-- Remaining homes after the first hour
def remaining_homes_first_hour : ℕ := total_homes - homes_distributed_first_hour

-- Distributed homes in the next 2 hours
def homes_distributed_next_two_hours : ℕ := (60 * remaining_homes_first_hour) / 100

-- Remaining homes after the next 2 hours
def homes_remaining : ℕ := remaining_homes_first_hour - homes_distributed_next_two_hours

theorem remaining_homes_proof : homes_remaining = 48 := by
  sorry

end NUMINAMATH_GPT_remaining_homes_proof_l830_83006


namespace NUMINAMATH_GPT_largest_angle_in_pentagon_l830_83074

-- Define the angles of the pentagon
variables (C D E : ℝ) 

-- Given conditions
def is_pentagon (A B C D E : ℝ) : Prop :=
  A = 75 ∧ B = 95 ∧ D = C + 10 ∧ E = 2 * C + 20 ∧ A + B + C + D + E = 540

-- Prove that the measure of the largest angle is 190°
theorem largest_angle_in_pentagon (C D E : ℝ) : 
  is_pentagon 75 95 C D E → max 75 (max 95 (max C (max (C + 10) (2 * C + 20)))) = 190 :=
by 
  sorry

end NUMINAMATH_GPT_largest_angle_in_pentagon_l830_83074


namespace NUMINAMATH_GPT_third_side_integer_lengths_l830_83092

theorem third_side_integer_lengths (a b : Nat) (h1 : a = 8) (h2 : b = 11) : 
  ∃ n, n = 15 :=
by
  sorry

end NUMINAMATH_GPT_third_side_integer_lengths_l830_83092


namespace NUMINAMATH_GPT_average_weight_increase_l830_83014

noncomputable def average_increase (A : ℝ) : ℝ :=
  let initial_total := 10 * A
  let new_total := initial_total + 25
  let new_average := new_total / 10
  new_average - A

theorem average_weight_increase (A : ℝ) : average_increase A = 2.5 := by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l830_83014


namespace NUMINAMATH_GPT_remainder_when_divided_by_2_is_0_l830_83061

theorem remainder_when_divided_by_2_is_0 (n : ℕ)
  (h1 : ∃ r, n % 2 = r)
  (h2 : n % 7 = 5)
  (h3 : ∃ p, p = 5 ∧ (n + p) % 10 = 0) :
  n % 2 = 0 :=
by
  -- skipping the proof steps; hence adding sorry
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_2_is_0_l830_83061


namespace NUMINAMATH_GPT_least_integer_x_l830_83053

theorem least_integer_x (x : ℤ) (h : 240 ∣ x^2) : x = 60 :=
sorry

end NUMINAMATH_GPT_least_integer_x_l830_83053


namespace NUMINAMATH_GPT_l_shape_area_l830_83003

theorem l_shape_area (P : ℝ) (L : ℝ) (x : ℝ)
  (hP : P = 52) 
  (hL : L = 16) 
  (h_x : L + (L - x) + 2 * (16 - x) = P)
  (h_split : 2 * (16 - x) * x = 120) :
  2 * ((16 - x) * x) = 120 :=
by
  -- This is the proof problem statement
  sorry

end NUMINAMATH_GPT_l_shape_area_l830_83003


namespace NUMINAMATH_GPT_initial_cows_l830_83085

theorem initial_cows {D C : ℕ}
  (h1 : C = 2 * D)
  (h2 : 161 = (3 * C) / 4 + D / 4) :
  C = 184 :=
by
  sorry

end NUMINAMATH_GPT_initial_cows_l830_83085


namespace NUMINAMATH_GPT_projects_count_minimize_time_l830_83058

-- Define the conditions as given in the problem
def total_projects := 15
def energy_transfer_condition (x y : ℕ) : Prop := x = 2 * y - 3

-- Define question 1 as a proof problem
theorem projects_count (x y : ℕ) (h1 : x + y = total_projects) (h2 : energy_transfer_condition x y) :
  x = 9 ∧ y = 6 :=
by
  sorry

-- Define conditions for question 2
def average_time (energy_transfer_time leaping_gate_time : ℕ) (m n total_time : ℕ) : Prop :=
  total_time = 6 * m + 8 * n

-- Define additional conditions needed for Question 2 regarding time
theorem minimize_time (m n total_time : ℕ)
  (h1 : m + n = 10)
  (h2 : 10 - m > n)
  (h3 : average_time 6 8 m n total_time)
  (h4 : m = 6) :
  total_time = 68 :=
by
  sorry

end NUMINAMATH_GPT_projects_count_minimize_time_l830_83058


namespace NUMINAMATH_GPT_range_of_a_l830_83076

open Real

noncomputable def C1 (t a : ℝ) : ℝ × ℝ := (2 * t + 2 * a, -t)
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (2 * cos θ, 2 + 2 * sin θ)

theorem range_of_a {a : ℝ} :
  (∃ (t θ : ℝ), C1 t a = C2 θ) ↔ 2 - sqrt 5 ≤ a ∧ a ≤ 2 + sqrt 5 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l830_83076


namespace NUMINAMATH_GPT_yoki_cans_l830_83039

-- Definitions of the conditions
def total_cans_collected : ℕ := 85
def ladonna_cans : ℕ := 25
def prikya_cans : ℕ := 2 * ladonna_cans
def avi_initial_cans : ℕ := 8
def avi_cans := avi_initial_cans / 2

-- Statement that needs to be proved
theorem yoki_cans : ∀ (total_cans_collected ladonna_cans : ℕ) 
  (prikya_cans : ℕ := 2 * ladonna_cans) 
  (avi_initial_cans : ℕ := 8) 
  (avi_cans : ℕ := avi_initial_cans / 2), 
  (total_cans_collected = 85) → 
  (ladonna_cans = 25) → 
  (prikya_cans = 2 * ladonna_cans) →
  (avi_initial_cans = 8) → 
  (avi_cans = avi_initial_cans / 2) → 
  total_cans_collected - (ladonna_cans + prikya_cans + avi_cans) = 6 :=
by
  intros total_cans_collected ladonna_cans prikya_cans avi_initial_cans avi_cans H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_yoki_cans_l830_83039


namespace NUMINAMATH_GPT_inequality_solution_l830_83029

open Set

def f (x : ℝ) : ℝ := |x| + x^2 + 2

def solution_set : Set ℝ := { x | x < -2 ∨ x > 4 / 3 }

theorem inequality_solution :
  { x : ℝ | f (2 * x - 1) > f (3 - x) } = solution_set := by
  sorry

end NUMINAMATH_GPT_inequality_solution_l830_83029


namespace NUMINAMATH_GPT_compare_negatives_l830_83062

theorem compare_negatives : -3.3 < -3.14 :=
sorry

end NUMINAMATH_GPT_compare_negatives_l830_83062


namespace NUMINAMATH_GPT_john_total_spent_l830_83050

/-- John's expenditure calculations -/
theorem john_total_spent :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 5
  let original_video_card_cost := 300
  let upgraded_video_card_cost := original_video_card_cost * 2
  let additional_upgrade_cost := upgraded_video_card_cost - original_video_card_cost
  let total_spent := computer_cost + peripherals_cost + additional_upgrade_cost
  total_spent = 2100 :=
by
  sorry

end NUMINAMATH_GPT_john_total_spent_l830_83050


namespace NUMINAMATH_GPT_arithmetic_sequence_n_15_l830_83081

theorem arithmetic_sequence_n_15 (a : ℕ → ℤ) (n : ℕ)
  (h₁ : a 3 = 5)
  (h₂ : a 2 + a 5 = 12)
  (h₃ : a n = 29) :
  n = 15 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_15_l830_83081


namespace NUMINAMATH_GPT_find_z_value_l830_83080

theorem find_z_value (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (h1 : x = 2 + 1 / z)
  (h2 : z = 3 + 1 / x) : 
  z = (3 + Real.sqrt 15) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_z_value_l830_83080


namespace NUMINAMATH_GPT_tire_mileage_l830_83041

theorem tire_mileage (total_miles_driven : ℕ) (x : ℕ) (spare_tire_miles : ℕ):
  total_miles_driven = 40000 →
  spare_tire_miles = 2 * x →
  4 * x + spare_tire_miles = total_miles_driven →
  x = 6667 := 
by
  intros h_total h_spare h_eq
  sorry

end NUMINAMATH_GPT_tire_mileage_l830_83041


namespace NUMINAMATH_GPT_five_op_two_l830_83005

-- Definition of the operation
def op (a b : ℝ) := 3 * a + 4 * b

-- The theorem statement
theorem five_op_two : op 5 2 = 23 := by
  sorry

end NUMINAMATH_GPT_five_op_two_l830_83005


namespace NUMINAMATH_GPT_repeating_block_length_five_sevenths_l830_83084

theorem repeating_block_length_five_sevenths : 
  ∃ n : ℕ, (∃ k : ℕ, (5 * 10^k - 5) % 7 = 0) ∧ n = 6 :=
sorry

end NUMINAMATH_GPT_repeating_block_length_five_sevenths_l830_83084


namespace NUMINAMATH_GPT_find_scalars_l830_83036

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 2;
    3, 1]

noncomputable def B4 : Matrix (Fin 2) (Fin 2) ℝ :=
  B * B * B * B

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_scalars (r s : ℝ) (hB : B^4 = r • B + s • I) :
  (r, s) = (51, 52) :=
  sorry

end NUMINAMATH_GPT_find_scalars_l830_83036


namespace NUMINAMATH_GPT_otimes_computation_l830_83096

-- Definition of ⊗ given m
def otimes (a b m : ℕ) : ℚ := (m * a + b) / (2 * a * b)

-- The main theorem we need to prove
theorem otimes_computation (m : ℕ) (h : otimes 1 4 m = otimes 2 3 m) :
  otimes 3 4 6 = 11 / 12 :=
sorry

end NUMINAMATH_GPT_otimes_computation_l830_83096


namespace NUMINAMATH_GPT_remainder_product_mod_5_l830_83065

theorem remainder_product_mod_5 (a b c : ℕ) (h_a : a % 5 = 2) (h_b : b % 5 = 3) (h_c : c % 5 = 4) :
  (a * b * c) % 5 = 4 := 
by
  sorry

end NUMINAMATH_GPT_remainder_product_mod_5_l830_83065


namespace NUMINAMATH_GPT_lower_limit_of_arun_weight_l830_83088

-- Given conditions for Arun's weight
variables (W : ℝ)
variables (avg_val : ℝ)

-- Define the conditions
def arun_weight_condition_1 := W < 72
def arun_weight_condition_2 := 60 < W ∧ W < 70
def arun_weight_condition_3 := W ≤ 67
def arun_weight_avg := avg_val = 66

-- The math proof problem statement
theorem lower_limit_of_arun_weight 
  (h1: arun_weight_condition_1 W) 
  (h2: arun_weight_condition_2 W) 
  (h3: arun_weight_condition_3 W) 
  (h4: arun_weight_avg avg_val) :
  ∃ (lower_limit : ℝ), lower_limit = 65 :=
sorry

end NUMINAMATH_GPT_lower_limit_of_arun_weight_l830_83088


namespace NUMINAMATH_GPT_pat_initial_stickers_l830_83071

def initial_stickers (s : ℕ) : ℕ := s  -- Number of stickers Pat had on the first day of the week

def stickers_earned : ℕ := 22  -- Stickers earned during the week

def stickers_end_week (s : ℕ) : ℕ := initial_stickers s + stickers_earned  -- Stickers at the end of the week

theorem pat_initial_stickers (s : ℕ) (h : stickers_end_week s = 61) : s = 39 :=
by
  sorry

end NUMINAMATH_GPT_pat_initial_stickers_l830_83071


namespace NUMINAMATH_GPT_sqrt_condition_l830_83015

theorem sqrt_condition (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_sqrt_condition_l830_83015


namespace NUMINAMATH_GPT_factor_by_resultant_is_three_l830_83093

theorem factor_by_resultant_is_three
  (x : ℕ) (f : ℕ) (h1 : x = 7)
  (h2 : (2 * x + 9) * f = 69) :
  f = 3 :=
sorry

end NUMINAMATH_GPT_factor_by_resultant_is_three_l830_83093


namespace NUMINAMATH_GPT_problem_l830_83035

theorem problem (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + (1/r^4) = 7 := 
by
  sorry

end NUMINAMATH_GPT_problem_l830_83035


namespace NUMINAMATH_GPT_average_weight_whole_class_l830_83018

def sectionA_students : Nat := 36
def sectionB_students : Nat := 44
def avg_weight_sectionA : Float := 40.0 
def avg_weight_sectionB : Float := 35.0
def total_weight_sectionA := avg_weight_sectionA * Float.ofNat sectionA_students
def total_weight_sectionB := avg_weight_sectionB * Float.ofNat sectionB_students
def total_students := sectionA_students + sectionB_students
def total_weight := total_weight_sectionA + total_weight_sectionB
def avg_weight_class := total_weight / Float.ofNat total_students

theorem average_weight_whole_class :
  avg_weight_class = 37.25 := by
  sorry

end NUMINAMATH_GPT_average_weight_whole_class_l830_83018


namespace NUMINAMATH_GPT_solve_for_n_l830_83056

theorem solve_for_n (n : ℕ) : (8 ^ n) * (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 4 → n = 2 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_n_l830_83056


namespace NUMINAMATH_GPT_larger_number_l830_83078

theorem larger_number (x y : ℝ) (h₁ : x + y = 45) (h₂ : x - y = 7) : x = 26 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_l830_83078


namespace NUMINAMATH_GPT_compute_floor_expression_l830_83079

theorem compute_floor_expression : 
  (Int.floor (↑(2025^3) / (2023 * 2024 : ℤ) - ↑(2023^3) / (2024 * 2025 : ℤ)) = 8) := 
sorry

end NUMINAMATH_GPT_compute_floor_expression_l830_83079


namespace NUMINAMATH_GPT_max_sum_arith_seq_l830_83023

theorem max_sum_arith_seq :
  let a1 := 29
  let d := 2
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  S_n 10 = S_n 20 → S_n 20 = 960 := by
sorry

end NUMINAMATH_GPT_max_sum_arith_seq_l830_83023


namespace NUMINAMATH_GPT_nonagon_diagonals_not_parallel_l830_83077

theorem nonagon_diagonals_not_parallel (n : ℕ) (h : n = 9) : 
  ∃ k : ℕ, k = 18 ∧ 
    ∀ v₁ v₂, v₁ ≠ v₂ → (n : ℕ).choose 2 = 27 → 
    (v₂ - v₁) % n ≠ 4 ∧ (v₂ - v₁) % n ≠ n-4 :=
by
  sorry

end NUMINAMATH_GPT_nonagon_diagonals_not_parallel_l830_83077


namespace NUMINAMATH_GPT_man_l830_83069

theorem man's_speed_with_current (v c : ℝ) (h1 : c = 4.3) (h2 : v - c = 12.4) : v + c = 21 :=
by {
  sorry
}

end NUMINAMATH_GPT_man_l830_83069


namespace NUMINAMATH_GPT_calculate_expression_correct_l830_83060

theorem calculate_expression_correct :
  ( (6 + (7 / 8) - (2 + (1 / 2))) * (1 / 4) + (3 + (23 / 24) + 1 + (2 / 3)) / 4 ) / 2.5 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_correct_l830_83060


namespace NUMINAMATH_GPT_brody_battery_fraction_l830_83008

theorem brody_battery_fraction (full_battery : ℕ) (battery_left_after_exam : ℕ) (exam_duration : ℕ) 
  (battery_before_exam : ℕ) (battery_used : ℕ) (fraction_used : ℚ) 
  (h1 : full_battery = 60)
  (h2 : battery_left_after_exam = 13)
  (h3 : exam_duration = 2)
  (h4 : battery_before_exam = battery_left_after_exam + exam_duration)
  (h5 : battery_used = full_battery - battery_before_exam)
  (h6 : fraction_used = battery_used / full_battery) :
  fraction_used = 3 / 4 := 
sorry

end NUMINAMATH_GPT_brody_battery_fraction_l830_83008


namespace NUMINAMATH_GPT_three_digit_integer_equal_sum_factorials_l830_83066

open Nat

theorem three_digit_integer_equal_sum_factorials :
  ∃ (a b c : ℕ), a = 1 ∧ b = 4 ∧ c = 5 ∧ 100 * a + 10 * b + c = a.factorial + b.factorial + c.factorial :=
by
  use 1, 4, 5
  simp
  sorry

end NUMINAMATH_GPT_three_digit_integer_equal_sum_factorials_l830_83066


namespace NUMINAMATH_GPT_work_completion_days_l830_83032

theorem work_completion_days (a b : Type) (T : ℕ) (ha : T = 12) (hb : T = 6) : 
  (T = 4) :=
sorry

end NUMINAMATH_GPT_work_completion_days_l830_83032


namespace NUMINAMATH_GPT_number_of_women_bathing_suits_correct_l830_83070

def men_bathing_suits : ℕ := 14797
def total_bathing_suits : ℕ := 19766

def women_bathing_suits : ℕ :=
  total_bathing_suits - men_bathing_suits

theorem number_of_women_bathing_suits_correct :
  women_bathing_suits = 19669 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_women_bathing_suits_correct_l830_83070


namespace NUMINAMATH_GPT_find_p_l830_83086

-- Definitions and conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_valid_configuration (p q s r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime s ∧ is_prime r ∧ 
  1 < p ∧ p < q ∧ q < s ∧ p + q + s = r

-- The theorem statement
theorem find_p (p q s r : ℕ) (h : is_valid_configuration p q s r) : p = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l830_83086


namespace NUMINAMATH_GPT_area_of_circle_above_below_lines_l830_83000

noncomputable def circle_area : ℝ :=
  40 * Real.pi

theorem area_of_circle_above_below_lines :
  ∃ (x y : ℝ), (x^2 + y^2 - 16*x - 8*y = 0) ∧ (y > x - 4) ∧ (y < -x + 4) ∧
  (circle_area = 40 * Real.pi) :=
  sorry

end NUMINAMATH_GPT_area_of_circle_above_below_lines_l830_83000
