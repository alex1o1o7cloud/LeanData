import Mathlib

namespace NUMINAMATH_GPT_midpoint_fraction_l1800_180009

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (a + b) / 2 = 19/24 := by
  sorry

end NUMINAMATH_GPT_midpoint_fraction_l1800_180009


namespace NUMINAMATH_GPT_parameter_a_range_l1800_180030

def quadratic_function (a x : ℝ) : ℝ := x^2 + 2 * a * x + 2 * a + 1

theorem parameter_a_range :
  (∀ x : ℝ, (-1 ≤ x ∧ x ≤ 1) → quadratic_function a x ≥ 1) ↔ (0 ≤ a) :=
by
  sorry

end NUMINAMATH_GPT_parameter_a_range_l1800_180030


namespace NUMINAMATH_GPT_vertex_of_parabola_l1800_180066

theorem vertex_of_parabola (x y : ℝ) : (y^2 - 4 * y + 3 * x + 7 = 0) → (x, y) = (-1, 2) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1800_180066


namespace NUMINAMATH_GPT_negation_proof_l1800_180061

theorem negation_proof : ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l1800_180061


namespace NUMINAMATH_GPT_line_equation_through_point_with_intercepts_conditions_l1800_180079

theorem line_equation_through_point_with_intercepts_conditions :
  ∃ (a b : ℚ) (m c : ℚ), 
    (-5) * m + c = 2 ∧ -- The line passes through A(-5, 2)
    a = 2 * b ∧       -- x-intercept is twice the y-intercept
    (a * m + c = 0 ∨ ((1/m)*a + (1/m)^2 * c+1 = 0)) :=         -- Equations of the line
sorry

end NUMINAMATH_GPT_line_equation_through_point_with_intercepts_conditions_l1800_180079


namespace NUMINAMATH_GPT_train_passing_time_l1800_180046

-- conditions
def train_length := 490 -- in meters
def train_speed_kmh := 63 -- in kilometers per hour
def conversion_factor := 1000 / 3600 -- to convert km/hr to m/s

-- conversion
def train_speed_ms := train_speed_kmh * conversion_factor -- speed in meters per second

-- expected correct answer
def expected_time := 28 -- in seconds

-- Theorem statement
theorem train_passing_time :
  train_length / train_speed_ms = expected_time :=
by
  sorry

end NUMINAMATH_GPT_train_passing_time_l1800_180046


namespace NUMINAMATH_GPT_smallest_product_of_two_distinct_primes_greater_than_50_l1800_180028

theorem smallest_product_of_two_distinct_primes_greater_than_50 : 
  ∃ (p q : ℕ), p > 50 ∧ q > 50 ∧ Prime p ∧ Prime q ∧ p ≠ q ∧ p * q = 3127 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_product_of_two_distinct_primes_greater_than_50_l1800_180028


namespace NUMINAMATH_GPT_constant_sum_of_distances_l1800_180000

open Real

theorem constant_sum_of_distances (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
    (ellipse_condition : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1 → ∀ A B : ℝ × ℝ, A.2 > 0 ∧ B.2 > 0)
    (foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-c, 0), (c, 0)))
    (points_AB : ∃ (A B : ℝ × ℝ), A.2 > 0 ∧ B.2 > 0 ∧ (A.1 - c)^2 / a^2 + A.2^2 / b^2 = 1 ∧ (B.1 - -c)^2 / a^2 + B.2^2 / b^2 = 1)
    (AF1_parallel_BF2 : ∀ (A B : ℝ × ℝ), (A.1 - -c) * (B.2 - 0) - (A.2 - 0) * (B.1 - c) = 0)
    (intersection_P: ∀ (A B : ℝ × ℝ), ∃ P : ℝ × ℝ, ((A.1 - c) * (B.2 - 0) = (A.2 - 0) * (P.1 - c)) ∧ ((B.1 - -c) * (A.2 - 0) = (B.2 - 0) * (P.1 - -c))) :
    ∃ k : ℝ, ∀ (P : ℝ × ℝ), dist P (foci.fst) + dist P (foci.snd) = k := 
sorry

end NUMINAMATH_GPT_constant_sum_of_distances_l1800_180000


namespace NUMINAMATH_GPT_fg_at_3_l1800_180016

-- Define the functions f and g according to the conditions
def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2)^2

theorem fg_at_3 : f (g 3) = 103 :=
by
  sorry

end NUMINAMATH_GPT_fg_at_3_l1800_180016


namespace NUMINAMATH_GPT_trigonometric_identity_l1800_180033

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π / 4 + α) = 1 / 2) : 
  (Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α)) * Real.cos (7 * π / 4 - α) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1800_180033


namespace NUMINAMATH_GPT_areaOfTangencyTriangle_l1800_180089

noncomputable def semiPerimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

noncomputable def areaABC (a b c : ℝ) : ℝ :=
  let p := semiPerimeter a b c
  Real.sqrt (p * (p - a) * (p - b) * (p - c))

noncomputable def excircleRadius (a b c : ℝ) : ℝ :=
  let S := areaABC a b c
  let p := semiPerimeter a b c
  S / (p - a)

theorem areaOfTangencyTriangle (a b c R : ℝ) :
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  (S * (ra / (2 * R))) = (S ^ 2 / (2 * R * (p - a))) :=
by
  let p := semiPerimeter a b c
  let S := areaABC a b c
  let ra := excircleRadius a b c
  sorry

end NUMINAMATH_GPT_areaOfTangencyTriangle_l1800_180089


namespace NUMINAMATH_GPT_cost_of_one_jacket_l1800_180063

theorem cost_of_one_jacket
  (S J : ℝ)
  (h1 : 10 * S + 20 * J = 800)
  (h2 : 5 * S + 15 * J = 550) : J = 30 :=
sorry

end NUMINAMATH_GPT_cost_of_one_jacket_l1800_180063


namespace NUMINAMATH_GPT_inequality_holds_for_positive_x_l1800_180032

theorem inequality_holds_for_positive_x (x : ℝ) (h : x > 0) : 
  x^8 - x^5 - 1/x + 1/(x^4) ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_holds_for_positive_x_l1800_180032


namespace NUMINAMATH_GPT_container_volume_ratio_l1800_180005

theorem container_volume_ratio
  (A B C : ℝ)
  (h1 : (3 / 4) * A - (5 / 8) * B = (7 / 8) * C - (1 / 2) * C)
  (h2 : B =  (5 / 8) * B)
  (h3 : (5 / 8) * B =  (3 / 8) * C)
  (h4 : A =  (24 / 40) * C) : 
  A / C = 4 / 5 := sorry

end NUMINAMATH_GPT_container_volume_ratio_l1800_180005


namespace NUMINAMATH_GPT_g_value_at_50_l1800_180091

noncomputable def g : ℝ → ℝ := sorry

theorem g_value_at_50 (g : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, 0 < x → 0 < y → x * g y + y * g x = g (x * y)) :
  g 50 = 0 :=
sorry

end NUMINAMATH_GPT_g_value_at_50_l1800_180091


namespace NUMINAMATH_GPT_gain_percent_is_80_l1800_180084

noncomputable def cost_price : ℝ := 600
noncomputable def selling_price : ℝ := 1080
noncomputable def gain : ℝ := selling_price - cost_price
noncomputable def gain_percent : ℝ := (gain / cost_price) * 100

theorem gain_percent_is_80 :
  gain_percent = 80 := by
  sorry

end NUMINAMATH_GPT_gain_percent_is_80_l1800_180084


namespace NUMINAMATH_GPT_words_per_page_is_106_l1800_180040

noncomputable def book_pages := 154
noncomputable def max_words_per_page := 120
noncomputable def total_words_mod := 221
noncomputable def mod_val := 217

def number_of_words_per_page (p : ℕ) : Prop :=
  (book_pages * p ≡ total_words_mod [MOD mod_val]) ∧ (p ≤ max_words_per_page)

theorem words_per_page_is_106 : number_of_words_per_page 106 :=
by
  sorry

end NUMINAMATH_GPT_words_per_page_is_106_l1800_180040


namespace NUMINAMATH_GPT_product_divisible_by_4_l1800_180068

theorem product_divisible_by_4 (a b c d : ℤ) 
    (h : a^2 + b^2 + c^2 = d^2) : 4 ∣ (a * b * c) :=
sorry

end NUMINAMATH_GPT_product_divisible_by_4_l1800_180068


namespace NUMINAMATH_GPT_profit_calculation_l1800_180092

variable (price : ℕ) (cost : ℕ) (exchange_rate : ℕ) (profit_per_bottle : ℚ)

-- Conditions
def conditions := price = 2 ∧ cost = 1 ∧ exchange_rate = 5

-- Profit per bottle is 0.66 yuan considering the exchange policy
theorem profit_calculation (h : conditions price cost exchange_rate) : profit_per_bottle = 0.66 := sorry

end NUMINAMATH_GPT_profit_calculation_l1800_180092


namespace NUMINAMATH_GPT_number_of_paths_A_to_D_l1800_180026

-- Definition of conditions
def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 2
def ways_C_to_D : Nat := 2
def direct_A_to_D : Nat := 1

-- Theorem statement for the total number of paths from A to D
theorem number_of_paths_A_to_D : ways_A_to_B * ways_B_to_C * ways_C_to_D + direct_A_to_D = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_paths_A_to_D_l1800_180026


namespace NUMINAMATH_GPT_inequality_solution_set_l1800_180004

theorem inequality_solution_set :
  {x : ℝ | (x - 5) * (x + 1) > 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 5} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1800_180004


namespace NUMINAMATH_GPT_eq_solution_l1800_180054

theorem eq_solution (x : ℝ) (h : 2 / x = 3 / (x + 1)) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_eq_solution_l1800_180054


namespace NUMINAMATH_GPT_general_term_a_general_term_b_sum_first_n_terms_l1800_180045

def a : Nat → Nat
| 0     => 1
| (n+1) => 2 * a n

def b (n : Nat) : Int :=
  3 * (n + 1) - 2

def S (n : Nat) : Int :=
  2^n - (3 * n^2) / 2 + n / 2 - 1

-- We state the theorems with the conditions included.

theorem general_term_a (n : Nat) : a n = 2^(n - 1) := by
  sorry

theorem general_term_b (n : Nat) : b n = 3 * (n + 1) - 2 := by
  sorry

theorem sum_first_n_terms (n : Nat) : 
  (Finset.range n).sum (λ i => a i - b i) = 2^n - (3 * n^2) / 2 + n / 2 - 1 := by
  sorry

end NUMINAMATH_GPT_general_term_a_general_term_b_sum_first_n_terms_l1800_180045


namespace NUMINAMATH_GPT_gcd_90_405_l1800_180010

theorem gcd_90_405 : Nat.gcd 90 405 = 45 := by
  sorry

end NUMINAMATH_GPT_gcd_90_405_l1800_180010


namespace NUMINAMATH_GPT_evaluate_expression_l1800_180065

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  rw [h]
  norm_num

end NUMINAMATH_GPT_evaluate_expression_l1800_180065


namespace NUMINAMATH_GPT_percent_of_x_is_y_l1800_180015

theorem percent_of_x_is_y (x y : ℝ) (h : 0.25 * (x - y) = 0.15 * (x + y)) : y / x = 0.25 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l1800_180015


namespace NUMINAMATH_GPT_MapleLeafHigh_points_l1800_180064

def MapleLeafHigh (x y : ℕ) : Prop :=
  (1/3 * x + 3/8 * x + 18 + y = x) ∧ (10 ≤ y) ∧ (y ≤ 30)

theorem MapleLeafHigh_points : ∃ y, MapleLeafHigh 104 y ∧ y = 21 := 
by
  use 21
  sorry

end NUMINAMATH_GPT_MapleLeafHigh_points_l1800_180064


namespace NUMINAMATH_GPT_initial_orange_balloons_l1800_180008

-- Definitions
variable (x : ℕ)
variable (h1 : x - 2 = 7)

-- Theorem to prove
theorem initial_orange_balloons (h1 : x - 2 = 7) : x = 9 :=
sorry

end NUMINAMATH_GPT_initial_orange_balloons_l1800_180008


namespace NUMINAMATH_GPT_no_solution_fractions_eq_l1800_180018

open Real

theorem no_solution_fractions_eq (x : ℝ) :
  (x-2)/(2*x-1) + 1 = 3/(2-4*x) → False :=
by
  intro h
  have h1 : ¬ (2*x - 1 = 0) := by
    -- 2*x - 1 ≠ 0
    sorry
  have h2 : ¬ (2 - 4*x = 0) := by
    -- 2 - 4*x ≠ 0
    sorry
  -- Solve the equation and show no solutions exist without contradicting the conditions
  sorry

end NUMINAMATH_GPT_no_solution_fractions_eq_l1800_180018


namespace NUMINAMATH_GPT_arithmetic_expression_eval_l1800_180053

theorem arithmetic_expression_eval : 8 / 4 - 3 - 9 + 3 * 9 = 17 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_eval_l1800_180053


namespace NUMINAMATH_GPT_cube_volume_l1800_180043

variables (x s : ℝ)
theorem cube_volume (h : 6 * s^2 = 6 * x^2) : s^3 = x^3 :=
by sorry

end NUMINAMATH_GPT_cube_volume_l1800_180043


namespace NUMINAMATH_GPT_ratio_MN_l1800_180080

variables (Q P R M N : ℝ)

def satisfies_conditions (Q P R M N : ℝ) : Prop :=
  M = 0.40 * Q ∧
  Q = 0.25 * P ∧
  R = 0.60 * P ∧
  N = 0.50 * R

theorem ratio_MN (Q P R M N : ℝ) (h : satisfies_conditions Q P R M N) : M / N = 1 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_MN_l1800_180080


namespace NUMINAMATH_GPT_angle_A_is_60_degrees_triangle_area_l1800_180027

-- Define the basic setup for the triangle and its angles
variables (a b c : ℝ) -- internal angles of the triangle ABC
variables (B C : ℝ) -- sides opposite to angles b and c respectively

-- Given conditions
axiom equation_1 : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a
axiom perimeter_condition : a + b + c = 8
axiom circumradius_condition : ∃ R : ℝ, R = Real.sqrt 3

-- Question 1: Prove the measure of angle A is 60 degrees
theorem angle_A_is_60_degrees (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a) : 
  a = 60 :=
sorry

-- Question 2: Prove the area of triangle ABC
theorem triangle_area (h : 2 * b * Real.cos a = a * Real.cos C + c * Real.cos a)
(h_perimeter : a + b + c = 8) (h_circumradius : ∃ R : ℝ, R = Real.sqrt 3) :
  ∃ S : ℝ, S = 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_GPT_angle_A_is_60_degrees_triangle_area_l1800_180027


namespace NUMINAMATH_GPT_Andrey_Gleb_distance_l1800_180002

theorem Andrey_Gleb_distance (AB VG : ℕ) (AG : ℕ) (BV : ℕ) (cond1 : AB = 600) (cond2 : VG = 600) (cond3 : AG = 3 * BV) :
  AG = 900 ∨ AG = 1800 := 
sorry

end NUMINAMATH_GPT_Andrey_Gleb_distance_l1800_180002


namespace NUMINAMATH_GPT_range_of_m_l1800_180007

-- Definitions based on the conditions
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 > 0

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) ↔ 0 < m ∧ m ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_m_l1800_180007


namespace NUMINAMATH_GPT_find_abc_value_l1800_180038

noncomputable def given_conditions (a b c : ℝ) : Prop :=
  (a * b / (a + b) = 2) ∧ (b * c / (b + c) = 5) ∧ (c * a / (c + a) = 9)

theorem find_abc_value (a b c : ℝ) (h : given_conditions a b c) :
  a * b * c / (a * b + b * c + c * a) = 90 / 73 :=
sorry

end NUMINAMATH_GPT_find_abc_value_l1800_180038


namespace NUMINAMATH_GPT_quadratic_has_one_positive_and_one_negative_root_l1800_180096

theorem quadratic_has_one_positive_and_one_negative_root
  (a : ℝ) (h₁ : a ≠ 0) (h₂ : a < -1) :
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + 2 * x₁ + 1 = 0) ∧ (a * x₂^2 + 2 * x₂ + 1 = 0) ∧ (x₁ > 0) ∧ (x₂ < 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_positive_and_one_negative_root_l1800_180096


namespace NUMINAMATH_GPT_opposite_of_neg_2022_l1800_180067

theorem opposite_of_neg_2022 : -(-2022) = 2022 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_2022_l1800_180067


namespace NUMINAMATH_GPT_total_fireworks_l1800_180047

-- Definitions based on conditions
def kobys_boxes := 2
def kobys_sparklers_per_box := 3
def kobys_whistlers_per_box := 5
def cheries_boxes := 1
def cheries_sparklers_per_box := 8
def cheries_whistlers_per_box := 9

-- Calculations
def total_kobys_fireworks := kobys_boxes * (kobys_sparklers_per_box + kobys_whistlers_per_box)
def total_cheries_fireworks := cheries_boxes * (cheries_sparklers_per_box + cheries_whistlers_per_box)

-- Theorem
theorem total_fireworks : total_kobys_fireworks + total_cheries_fireworks = 33 := 
by
  -- Can be elaborated and filled in with steps, if necessary.
  sorry

end NUMINAMATH_GPT_total_fireworks_l1800_180047


namespace NUMINAMATH_GPT_find_other_diagonal_l1800_180058

theorem find_other_diagonal (A : ℝ) (d1 : ℝ) (hA : A = 80) (hd1 : d1 = 16) :
  ∃ d2 : ℝ, 2 * A / d1 = d2 :=
by
  use 10
  -- Rest of the proof goes here
  sorry

end NUMINAMATH_GPT_find_other_diagonal_l1800_180058


namespace NUMINAMATH_GPT_marbles_each_friend_gets_l1800_180088

-- Definitions for the conditions
def total_marbles : ℕ := 100
def marbles_kept : ℕ := 20
def number_of_friends : ℕ := 5

-- The math proof problem
theorem marbles_each_friend_gets :
  (total_marbles - marbles_kept) / number_of_friends = 16 :=
by
  -- We include the proof steps within a by notation but stop at sorry for automated completion skipping proof steps
  sorry

end NUMINAMATH_GPT_marbles_each_friend_gets_l1800_180088


namespace NUMINAMATH_GPT_angle_C_in_parallelogram_l1800_180034

theorem angle_C_in_parallelogram (ABCD : Type)
  (angle_A angle_B angle_C angle_D : ℝ)
  (h1 : angle_A = angle_C)
  (h2 : angle_B = angle_D)
  (h3 : angle_A + angle_B = 180)
  (h4 : angle_A / angle_B = 3) :
  angle_C = 135 :=
  sorry

end NUMINAMATH_GPT_angle_C_in_parallelogram_l1800_180034


namespace NUMINAMATH_GPT_determine_h_l1800_180072

-- Define the initial quadratic expression
def quadratic (x : ℝ) : ℝ := 3 * x^2 + 8 * x + 15

-- Define the form we want to prove
def completed_square_form (x h k : ℝ) : ℝ := 3 * (x - h)^2 + k

-- The proof problem translated to Lean 4
theorem determine_h : ∃ k : ℝ, ∀ x : ℝ, quadratic x = completed_square_form x (-4 / 3) k :=
by
  exists (29 / 3)
  intro x
  sorry

end NUMINAMATH_GPT_determine_h_l1800_180072


namespace NUMINAMATH_GPT_vessel_capacity_proof_l1800_180012

variable (V1_capacity : ℕ) (V2_capacity : ℕ) (total_mixture : ℕ) (final_vessel_capacity : ℕ)
variable (A1_percentage : ℕ) (A2_percentage : ℕ)

theorem vessel_capacity_proof
  (h1 : V1_capacity = 2)
  (h2 : A1_percentage = 35)
  (h3 : V2_capacity = 6)
  (h4 : A2_percentage = 50)
  (h5 : total_mixture = 8)
  (h6 : final_vessel_capacity = 10)
  : final_vessel_capacity = 10 := 
by
  sorry

end NUMINAMATH_GPT_vessel_capacity_proof_l1800_180012


namespace NUMINAMATH_GPT_no_common_root_l1800_180086

theorem no_common_root 
  (a b : ℚ) 
  (α : ℂ) 
  (h1 : α^5 = α + 1) 
  (h2 : α^2 = -a * α - b) : 
  False :=
sorry

end NUMINAMATH_GPT_no_common_root_l1800_180086


namespace NUMINAMATH_GPT_solve_for_y_l1800_180048

theorem solve_for_y (y : ℚ) (h : 2 * y + 3 * y = 500 - (4 * y + 6 * y)) : y = 100 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1800_180048


namespace NUMINAMATH_GPT_ratio_of_triangle_side_to_rectangle_width_l1800_180049

theorem ratio_of_triangle_side_to_rectangle_width
  (t w : ℕ)
  (ht : 3 * t = 24)
  (hw : 6 * w = 24) :
  t / w = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_triangle_side_to_rectangle_width_l1800_180049


namespace NUMINAMATH_GPT_equation_has_exactly_one_real_solution_l1800_180082

-- Definitions for the problem setup
def equation (k : ℝ) (x : ℝ) : Prop := (3 * x + 8) * (x - 6) = -54 + k * x

-- The property that we need to prove
theorem equation_has_exactly_one_real_solution (k : ℝ) :
  (∀ x : ℝ, equation k x → ∃! x : ℝ, equation k x) ↔ k = 6 * Real.sqrt 2 - 10 ∨ k = -6 * Real.sqrt 2 - 10 := 
sorry

end NUMINAMATH_GPT_equation_has_exactly_one_real_solution_l1800_180082


namespace NUMINAMATH_GPT_largest_y_l1800_180021

theorem largest_y : ∃ (y : ℤ), (y ≤ 3) ∧ (∀ (z : ℤ), (z > y) → ¬ (z / 4 + 6 / 7 < 7 / 4)) :=
by
  -- There exists an integer y such that y <= 3 and for all integers z greater than y, the inequality does not hold
  sorry

end NUMINAMATH_GPT_largest_y_l1800_180021


namespace NUMINAMATH_GPT_question_equals_answer_l1800_180057

theorem question_equals_answer (x y : ℝ) (h : abs (x - 6) + (y + 4)^2 = 0) : x + y = 2 :=
sorry

end NUMINAMATH_GPT_question_equals_answer_l1800_180057


namespace NUMINAMATH_GPT_g_g_g_of_3_eq_neg_6561_l1800_180023

def g (x : ℤ) : ℤ := -x^2

theorem g_g_g_of_3_eq_neg_6561 : g (g (g 3)) = -6561 := by
  sorry

end NUMINAMATH_GPT_g_g_g_of_3_eq_neg_6561_l1800_180023


namespace NUMINAMATH_GPT_remainder_sum_first_150_div_11300_l1800_180076

theorem remainder_sum_first_150_div_11300 :
  let n := 150
  let S := n * (n + 1) / 2
  S % 11300 = 25 :=
by
  let n := 150
  let S := n * (n + 1) / 2
  show S % 11300 = 25
  sorry

end NUMINAMATH_GPT_remainder_sum_first_150_div_11300_l1800_180076


namespace NUMINAMATH_GPT_sector_area_l1800_180044

theorem sector_area (θ r : ℝ) (hθ : θ = 2) (hr : r = 1) :
  (1 / 2) * r^2 * θ = 1 :=
by
  -- Conditions are instantiated
  rw [hθ, hr]
  -- Simplification is left to the proof
  sorry

end NUMINAMATH_GPT_sector_area_l1800_180044


namespace NUMINAMATH_GPT_decagon_area_l1800_180070

theorem decagon_area 
    (perimeter_square : ℝ) 
    (side_division : ℕ) 
    (side_length : ℝ) 
    (triangle_area : ℝ) 
    (total_triangle_area : ℝ) 
    (square_area : ℝ)
    (decagon_area : ℝ) :
    perimeter_square = 150 →
    side_division = 5 →
    side_length = perimeter_square / 4 →
    triangle_area = 1 / 2 * (side_length / side_division) * (side_length / side_division) →
    total_triangle_area = 8 * triangle_area →
    square_area = side_length * side_length →
    decagon_area = square_area - total_triangle_area →
    decagon_area = 1181.25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_decagon_area_l1800_180070


namespace NUMINAMATH_GPT_sequence_a_n_eq_5050_l1800_180013

theorem sequence_a_n_eq_5050 (a : ℕ → ℕ) (h1 : ∀ n > 1, (n - 1) * a n = (n + 1) * a (n - 1)) (h2 : a 1 = 1) : 
  a 100 = 5050 := 
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_eq_5050_l1800_180013


namespace NUMINAMATH_GPT_unique_line_equation_l1800_180098

theorem unique_line_equation
  (k : ℝ)
  (m b : ℝ)
  (h1 : |(k^2 + 4*k + 3) - (m*k + b)| = 4)
  (h2 : 2*m + b = 8)
  (h3 : b ≠ 0) :
  (m = 6 ∧ b = -4) :=
by
  sorry

end NUMINAMATH_GPT_unique_line_equation_l1800_180098


namespace NUMINAMATH_GPT_initial_budget_calculation_l1800_180075

variable (flaskCost testTubeCost safetyGearCost totalExpenses remainingAmount initialBudget : ℕ)

theorem initial_budget_calculation (h1 : flaskCost = 150)
                               (h2 : testTubeCost = 2 * flaskCost / 3)
                               (h3 : safetyGearCost = testTubeCost / 2)
                               (h4 : totalExpenses = flaskCost + testTubeCost + safetyGearCost)
                               (h5 : remainingAmount = 25)
                               (h6 : initialBudget = totalExpenses + remainingAmount) :
                               initialBudget = 325 := by
  sorry

end NUMINAMATH_GPT_initial_budget_calculation_l1800_180075


namespace NUMINAMATH_GPT_plastic_skulls_number_l1800_180037

-- Define the conditions
def num_broomsticks : ℕ := 4
def num_spiderwebs : ℕ := 12
def num_pumpkins := 2 * num_spiderwebs
def num_cauldron : ℕ := 1
def budget_left_to_buy : ℕ := 20
def num_left_to_put_up : ℕ := 10
def total_decorations : ℕ := 83

-- The number of plastic skulls calculation as a function
def num_other_decorations : ℕ :=
  num_broomsticks + num_spiderwebs + num_pumpkins + num_cauldron + budget_left_to_buy + num_left_to_put_up

def num_plastic_skulls := total_decorations - num_other_decorations

-- The theorem to be proved
theorem plastic_skulls_number : num_plastic_skulls = 12 := by
  sorry

end NUMINAMATH_GPT_plastic_skulls_number_l1800_180037


namespace NUMINAMATH_GPT_temperature_at_midnight_l1800_180055

-- Define temperature in the morning
def T_morning := -2 -- in degrees Celsius

-- Temperature change at noon
def delta_noon := 12 -- in degrees Celsius

-- Temperature change at midnight
def delta_midnight := -8 -- in degrees Celsius

-- Function to compute temperature
def compute_temperature (T : ℤ) (delta1 : ℤ) (delta2 : ℤ) : ℤ :=
  T + delta1 + delta2

-- The proposition to prove
theorem temperature_at_midnight :
  compute_temperature T_morning delta_noon delta_midnight = 2 :=
by
  sorry

end NUMINAMATH_GPT_temperature_at_midnight_l1800_180055


namespace NUMINAMATH_GPT_correct_solutions_l1800_180081

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ (x y : ℝ), f (x * y) = f x * f y - 2 * x * y

theorem correct_solutions :
  (∀ x : ℝ, f x = 2 * x) ∨ (∀ x : ℝ, f x = -x) := sorry

end NUMINAMATH_GPT_correct_solutions_l1800_180081


namespace NUMINAMATH_GPT_required_run_rate_per_batsman_l1800_180059

variable (initial_run_rate : ℝ) (overs_played : ℕ) (remaining_overs : ℕ)
variable (remaining_wickets : ℕ) (total_target : ℕ) 

theorem required_run_rate_per_batsman 
  (h_initial_run_rate : initial_run_rate = 3.4)
  (h_overs_played : overs_played = 10)
  (h_remaining_overs  : remaining_overs = 40)
  (h_remaining_wickets : remaining_wickets = 7)
  (h_total_target : total_target = 282) :
  (total_target - initial_run_rate * overs_played) / remaining_overs = 6.2 :=
by
  sorry

end NUMINAMATH_GPT_required_run_rate_per_batsman_l1800_180059


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l1800_180001

-- Problem 1
theorem isosceles_triangle_perimeter_1 (a b : ℕ) (h1: a = 4 ∨ a = 6) (h2: b = 4 ∨ b = 6) (h3: a ≠ b): 
  (a + b + b = 14 ∨ a + b + b = 16) :=
sorry

-- Problem 2
theorem isosceles_triangle_perimeter_2 (a b : ℕ) (h1: a = 2 ∨ a = 6) (h2: b = 2 ∨ b = 6) (h3: a ≠ b ∨ (a = 2 ∧ 2 + 2 ≥ 6 ∧ 6 = b)):
  (a + b + b = 14) :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_1_isosceles_triangle_perimeter_2_l1800_180001


namespace NUMINAMATH_GPT_r_has_money_l1800_180017

-- Define the variables and the conditions in Lean
variable (p q r : ℝ)
variable (h1 : p + q + r = 4000)
variable (h2 : r = (2/3) * (p + q))

-- Define the proof statement
theorem r_has_money : r = 1600 := 
  by
    sorry

end NUMINAMATH_GPT_r_has_money_l1800_180017


namespace NUMINAMATH_GPT_amusement_park_ticket_length_l1800_180024

theorem amusement_park_ticket_length (Area Width Length : ℝ) (h₀ : Area = 1.77) (h₁ : Width = 3) (h₂ : Area = Width * Length) : Length = 0.59 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_amusement_park_ticket_length_l1800_180024


namespace NUMINAMATH_GPT_integer_solutions_l1800_180042

theorem integer_solutions (n : ℤ) : ∃ m : ℤ, n^2 + 15 = m^2 ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l1800_180042


namespace NUMINAMATH_GPT_total_time_to_clean_and_complete_l1800_180078

def time_to_complete_assignment : Nat := 10
def num_remaining_keys : Nat := 14
def time_per_key : Nat := 3

theorem total_time_to_clean_and_complete :
  time_to_complete_assignment + num_remaining_keys * time_per_key = 52 :=
by
  sorry

end NUMINAMATH_GPT_total_time_to_clean_and_complete_l1800_180078


namespace NUMINAMATH_GPT_steel_bar_lengths_l1800_180019

theorem steel_bar_lengths
  (x y z : ℝ)
  (h1 : 2 * x + y + 3 * z = 23)
  (h2 : x + 4 * y + 5 * z = 36) :
  x + 2 * y + 3 * z = 22 := 
sorry

end NUMINAMATH_GPT_steel_bar_lengths_l1800_180019


namespace NUMINAMATH_GPT_pow_mod_26_l1800_180031

theorem pow_mod_26 (a b n : ℕ) (hn : n = 2023) (h₁ : a = 17) (h₂ : b = 26) :
  a ^ n % b = 7 := by
  sorry

end NUMINAMATH_GPT_pow_mod_26_l1800_180031


namespace NUMINAMATH_GPT_total_toys_l1800_180077

theorem total_toys (bill_toys hana_toys hash_toys: ℕ) 
  (hb: bill_toys = 60)
  (hh: hana_toys = (5 * bill_toys) / 6)
  (hs: hash_toys = (hana_toys / 2) + 9) :
  (bill_toys + hana_toys + hash_toys) = 144 :=
by
  sorry

end NUMINAMATH_GPT_total_toys_l1800_180077


namespace NUMINAMATH_GPT_add_expression_l1800_180014

theorem add_expression {k : ℕ} :
  (2 * k + 2) + (2 * k + 3) = (2 * k + 2) + (2 * k + 3) := sorry

end NUMINAMATH_GPT_add_expression_l1800_180014


namespace NUMINAMATH_GPT_smallest_group_size_l1800_180094

theorem smallest_group_size (n : ℕ) (k : ℕ) (hk : k > 2) (h1 : n % 2 = 0) (h2 : n % k = 0) :
  n = 6 :=
sorry

end NUMINAMATH_GPT_smallest_group_size_l1800_180094


namespace NUMINAMATH_GPT_solve_for_x_l1800_180085

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solve_for_x (x : ℝ) 
  (h1 : infinite_power_tower x = 4) : 
  x = Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l1800_180085


namespace NUMINAMATH_GPT_banana_distribution_correct_l1800_180039

noncomputable def proof_problem : Prop :=
  let bananas := 40
  let marbles := 4
  let boys := 18
  let girls := 12
  let total_friends := 30
  let bananas_for_boys := (3/8 : ℝ) * bananas
  let bananas_for_girls := (1/4 : ℝ) * bananas
  let bananas_left := bananas - (bananas_for_boys + bananas_for_girls)
  let bananas_per_marble := bananas_left / marbles
  bananas_for_boys = 15 ∧ bananas_for_girls = 10 ∧ bananas_per_marble = 3.75

theorem banana_distribution_correct : proof_problem :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_banana_distribution_correct_l1800_180039


namespace NUMINAMATH_GPT_july_16_2010_is_wednesday_l1800_180050

-- Define necessary concepts for the problem

def is_tuesday (d : ℕ) : Prop := (d % 7 = 2)
def day_after_n_days (d n : ℕ) : ℕ := (d + n) % 7

-- The statement we want to prove
theorem july_16_2010_is_wednesday (h : is_tuesday 1) : day_after_n_days 1 15 = 3 := 
sorry

end NUMINAMATH_GPT_july_16_2010_is_wednesday_l1800_180050


namespace NUMINAMATH_GPT_plan_b_more_cost_effective_l1800_180025

theorem plan_b_more_cost_effective (x : ℕ) : 
  (12 * x : ℤ) > (3000 + 8 * x : ℤ) → x ≥ 751 :=
sorry

end NUMINAMATH_GPT_plan_b_more_cost_effective_l1800_180025


namespace NUMINAMATH_GPT_notebook_cost_l1800_180056

theorem notebook_cost (s n c : ℕ) (h1 : s > 25)
                                 (h2 : n % 2 = 1)
                                 (h3 : n > 1)
                                 (h4 : c > n)
                                 (h5 : s * n * c = 2739) :
  c = 7 :=
sorry

end NUMINAMATH_GPT_notebook_cost_l1800_180056


namespace NUMINAMATH_GPT_max_a_for_integer_roots_l1800_180022

theorem max_a_for_integer_roots (a : ℕ) :
  (∀ x : ℤ, x^2 - 2 * (a : ℤ) * x + 64 = 0 → (∃ y : ℤ, x = y)) →
  (∀ x1 x2 : ℤ, x1 * x2 = 64 ∧ x1 + x2 = 2 * (a : ℤ)) →
  a ≤ 17 := 
sorry

end NUMINAMATH_GPT_max_a_for_integer_roots_l1800_180022


namespace NUMINAMATH_GPT_fraction_computation_l1800_180036

theorem fraction_computation : (2 / 3) * (3 / 4 * 40) = 20 := 
by
  -- The proof will go here, for now we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_fraction_computation_l1800_180036


namespace NUMINAMATH_GPT_direction_vector_correct_l1800_180071

open Real

def line_eq (x y : ℝ) : Prop := x - 3 * y + 1 = 0

noncomputable def direction_vector : ℝ × ℝ := (3, 1)

theorem direction_vector_correct (x y : ℝ) (h : line_eq x y) : 
    ∃ k : ℝ, direction_vector = (k * (1 : ℝ), k * (1 / 3)) :=
by
  use 3
  sorry

end NUMINAMATH_GPT_direction_vector_correct_l1800_180071


namespace NUMINAMATH_GPT_probability_both_cards_are_diamonds_l1800_180011

-- Conditions definitions
def total_cards : ℕ := 52
def diamonds_in_deck : ℕ := 13
def two_draws : ℕ := 2

-- Calculation definitions
def total_possible_outcomes : ℕ := (total_cards * (total_cards - 1)) / two_draws
def favorable_outcomes : ℕ := (diamonds_in_deck * (diamonds_in_deck - 1)) / two_draws

-- Definition of the probability asked in the question
def probability_both_diamonds : ℚ := favorable_outcomes / total_possible_outcomes

theorem probability_both_cards_are_diamonds :
  probability_both_diamonds = 1 / 17 := 
sorry

end NUMINAMATH_GPT_probability_both_cards_are_diamonds_l1800_180011


namespace NUMINAMATH_GPT_find_C_l1800_180069

theorem find_C (A B C : ℝ) 
  (h1 : A + B + C = 600) 
  (h2 : A + C = 250) 
  (h3 : B + C = 450) : 
  C = 100 :=
sorry

end NUMINAMATH_GPT_find_C_l1800_180069


namespace NUMINAMATH_GPT_age_difference_l1800_180090

theorem age_difference (P M Mo : ℕ) (h1 : P = (3 * M) / 5) (h2 : Mo = (4 * M) / 3) (h3 : P + M + Mo = 88) : Mo - P = 22 := 
by sorry

end NUMINAMATH_GPT_age_difference_l1800_180090


namespace NUMINAMATH_GPT_second_share_interest_rate_is_11_l1800_180052

noncomputable def calculate_interest_rate 
    (total_investment : ℝ)
    (amount_in_second_share : ℝ)
    (interest_rate_first : ℝ)
    (total_interest : ℝ) : ℝ := 
  let A := total_investment - amount_in_second_share
  let interest_first := (interest_rate_first / 100) * A
  let interest_second := total_interest - interest_first
  (100 * interest_second) / amount_in_second_share

theorem second_share_interest_rate_is_11 :
  calculate_interest_rate 100000 12499.999999999998 9 9250 = 11 := 
by
  sorry

end NUMINAMATH_GPT_second_share_interest_rate_is_11_l1800_180052


namespace NUMINAMATH_GPT_trigonometric_identity_l1800_180035

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (2 * α) + Real.sin (Real.pi / 2 + α) * Real.cos (3 * Real.pi / 2 - α) = -1 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1800_180035


namespace NUMINAMATH_GPT_determine_y_l1800_180051

variable {R : Type} [LinearOrderedField R]
variables {x y : R}

theorem determine_y (h1 : 2 * x - 3 * y = 5) (h2 : 4 * x + 9 * y = 6) : y = -4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_determine_y_l1800_180051


namespace NUMINAMATH_GPT_chuck_play_area_l1800_180029

-- Define the conditions for the problem in Lean
def shed_length1 : ℝ := 3
def shed_length2 : ℝ := 4
def leash_length : ℝ := 4

-- State the theorem we want to prove
theorem chuck_play_area :
  let sector_area1 := (3 / 4) * Real.pi * (leash_length ^ 2)
  let sector_area2 := (1 / 4) * Real.pi * (1 ^ 2)
  sector_area1 + sector_area2 = (49 / 4) * Real.pi := 
by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_GPT_chuck_play_area_l1800_180029


namespace NUMINAMATH_GPT_analogous_to_tetrahedron_is_triangle_l1800_180060

-- Define the objects as types
inductive Object
| Quadrilateral
| Pyramid
| Triangle
| Prism
| Tetrahedron

-- Define the analogous relationship
def analogous (a b : Object) : Prop :=
  (a = Object.Tetrahedron ∧ b = Object.Triangle)
  ∨ (b = Object.Tetrahedron ∧ a = Object.Triangle)

-- The main statement to prove
theorem analogous_to_tetrahedron_is_triangle :
  ∃ (x : Object), analogous Object.Tetrahedron x ∧ x = Object.Triangle :=
by
  sorry

end NUMINAMATH_GPT_analogous_to_tetrahedron_is_triangle_l1800_180060


namespace NUMINAMATH_GPT_find_sum_of_squares_l1800_180093

-- Define the variables x and y as real numbers
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := x + y = 12
def condition2 : Prop := x * y = 50

-- The statement we need to prove
theorem find_sum_of_squares (h1 : condition1 x y) (h2 : condition2 x y) : x^2 + y^2 = 44 := by
  sorry

end NUMINAMATH_GPT_find_sum_of_squares_l1800_180093


namespace NUMINAMATH_GPT_max_competitors_l1800_180095

theorem max_competitors (P1 P2 P3 : ℕ → ℕ → ℕ)
(hP1 : ∀ i, 0 ≤ P1 i ∧ P1 i ≤ 7)
(hP2 : ∀ i, 0 ≤ P2 i ∧ P2 i ≤ 7)
(hP3 : ∀ i, 0 ≤ P3 i ∧ P3 i ≤ 7)
(hDistinct : ∀ i j, i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :
  ∃ n, n ≤ 64 ∧ ∀ k, k < n → (∀ i j, i < k → j < k → i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :=
sorry

end NUMINAMATH_GPT_max_competitors_l1800_180095


namespace NUMINAMATH_GPT_solve_inequality_l1800_180074

open Set

-- Define a predicate for the inequality solution sets
def inequality_solution_set (k : ℝ) : Set ℝ :=
  if h : k = 0 then {x | x < 1}
  else if h : 0 < k ∧ k < 2 then {x | x < 1 ∨ x > 2 / k}
  else if h : k = 2 then {x | True} \ {1}
  else if h : k > 2 then {x | x < 2 / k ∨ x > 1}
  else {x | 2 / k < x ∧ x < 1}

-- The statement of the proof
theorem solve_inequality (k : ℝ) :
  ∀ x : ℝ, k * x^2 - (k + 2) * x + 2 < 0 ↔ x ∈ inequality_solution_set k :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1800_180074


namespace NUMINAMATH_GPT_not_necessarily_divisor_of_44_l1800_180041

theorem not_necessarily_divisor_of_44 {k : ℤ} (h1 : ∃ k, n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) :
  ¬(44 ∣ n) :=
sorry

end NUMINAMATH_GPT_not_necessarily_divisor_of_44_l1800_180041


namespace NUMINAMATH_GPT_trader_profit_percentage_l1800_180087

-- Definitions for the conditions
def original_price (P : ℝ) : ℝ := P
def discount_price (P : ℝ) : ℝ := 0.80 * P
def selling_price (P : ℝ) : ℝ := 0.80 * P * 1.45

-- Theorem statement including the problem's question and the correct answer
theorem trader_profit_percentage (P : ℝ) (hP : 0 < P) : 
  (selling_price P - original_price P) / original_price P * 100 = 16 :=
by
  sorry

end NUMINAMATH_GPT_trader_profit_percentage_l1800_180087


namespace NUMINAMATH_GPT_fraction_of_students_with_partner_l1800_180097

theorem fraction_of_students_with_partner (s t : ℕ) 
  (h : t = (4 * s) / 3) :
  (t / 4 + s / 3) / (t + s) = 2 / 7 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_fraction_of_students_with_partner_l1800_180097


namespace NUMINAMATH_GPT_total_distance_traveled_by_children_l1800_180073

theorem total_distance_traveled_by_children :
  let ap := 50
  let dist_1_vertex_skip := (50 : ℝ) * Real.sqrt 2
  let dist_2_vertices_skip := (50 : ℝ) * Real.sqrt (2 + 2 * Real.sqrt 2)
  let dist_diameter := (2 : ℝ) * 50
  let single_child_distance := 2 * dist_1_vertex_skip + 2 * dist_2_vertices_skip + dist_diameter
  8 * single_child_distance = 800 * Real.sqrt 2 + 800 * Real.sqrt (2 + 2 * Real.sqrt 2) + 800 :=
sorry

end NUMINAMATH_GPT_total_distance_traveled_by_children_l1800_180073


namespace NUMINAMATH_GPT_equilateral_triangle_side_length_l1800_180083

theorem equilateral_triangle_side_length (total_length : ℕ) (h1 : total_length = 78) : (total_length / 3) = 26 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_side_length_l1800_180083


namespace NUMINAMATH_GPT_find_positive_integers_l1800_180099

theorem find_positive_integers (n : ℕ) : 
  (∀ a : ℕ, a.gcd n = 1 → 2 * n * n ∣ a ^ n - 1) ↔ (n = 2 ∨ n = 6 ∨ n = 42 ∨ n = 1806) :=
sorry

end NUMINAMATH_GPT_find_positive_integers_l1800_180099


namespace NUMINAMATH_GPT_dollar_symmetric_l1800_180006

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric {x y : ℝ} : dollar (x + y) (y + x) = 0 :=
by
  sorry

end NUMINAMATH_GPT_dollar_symmetric_l1800_180006


namespace NUMINAMATH_GPT_ironman_age_l1800_180062

theorem ironman_age (T C P I : ℕ) (h1 : T = 13 * C) (h2 : C = 7 * P) (h3 : I = P + 32) (h4 : T = 1456) : I = 48 := 
by
  sorry

end NUMINAMATH_GPT_ironman_age_l1800_180062


namespace NUMINAMATH_GPT_flutes_tried_out_l1800_180003

theorem flutes_tried_out (flutes clarinets trumpets pianists : ℕ) 
  (percent_flutes_in : ℕ → ℕ) (percent_clarinets_in : ℕ → ℕ) 
  (percent_trumpets_in : ℕ → ℕ) (percent_pianists_in : ℕ → ℕ) 
  (total_in_band : ℕ) :
  percent_flutes_in flutes = 80 / 100 * flutes ∧
  percent_clarinets_in clarinets = 30 / 2 ∧
  percent_trumpets_in trumpets = 60 / 3 ∧
  percent_pianists_in pianists = 20 / 10 ∧
  total_in_band = 53 →
  flutes = 20 :=
by
  sorry

end NUMINAMATH_GPT_flutes_tried_out_l1800_180003


namespace NUMINAMATH_GPT_inverse_proportion_decreasing_l1800_180020

theorem inverse_proportion_decreasing (k : ℝ) (x : ℝ) (hx : x > 0) :
  (y = (k - 1) / x) → (k > 1) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_decreasing_l1800_180020
