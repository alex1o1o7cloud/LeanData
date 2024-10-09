import Mathlib

namespace lucas_min_deliveries_l60_6009

theorem lucas_min_deliveries (cost_of_scooter earnings_per_delivery fuel_cost_per_delivery parking_fee_per_delivery : ℕ)
  (cost_eq : cost_of_scooter = 3000)
  (earnings_eq : earnings_per_delivery = 12)
  (fuel_cost_eq : fuel_cost_per_delivery = 4)
  (parking_fee_eq : parking_fee_per_delivery = 1) :
  ∃ d : ℕ, 7 * d ≥ cost_of_scooter ∧ d = 429 := by
  sorry

end lucas_min_deliveries_l60_6009


namespace trigonometric_identity_l60_6043

variable {a b c A B C : ℝ}

theorem trigonometric_identity (h1 : 2 * c^2 - 2 * a^2 = b^2) 
  (cos_A : ℝ) (cos_C : ℝ) 
  (h_cos_A : cos_A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_cos_C : cos_C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  2 * c * cos_A - 2 * a * cos_C = b := 
sorry

end trigonometric_identity_l60_6043


namespace part_I_part_II_part_III_l60_6013

noncomputable def f (x : ℝ) := x / (x^2 - 1)

-- (I) Prove that f(2) = 2/3.
theorem part_I : f 2 = 2 / 3 :=
by sorry

-- (II) Prove that f(x) is decreasing on the interval (-1, 1).
theorem part_II : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 > f x2 :=
by sorry

-- (III) Prove that f(x) is an odd function.
theorem part_III : ∀ x : ℝ, f (-x) = -f x :=
by sorry

end part_I_part_II_part_III_l60_6013


namespace length_of_garden_l60_6070

variables (w l : ℕ)

-- Definitions based on the problem conditions
def length_twice_width := l = 2 * w
def perimeter_eq_900 := 2 * l + 2 * w = 900

-- The statement to be proved
theorem length_of_garden (h1 : length_twice_width w l) (h2 : perimeter_eq_900 w l) : l = 300 :=
sorry

end length_of_garden_l60_6070


namespace apples_and_pears_weight_l60_6041

theorem apples_and_pears_weight (apples pears : ℕ) 
    (h_apples : apples = 240) 
    (h_pears : pears = 3 * apples) : 
    apples + pears = 960 := 
  by
  sorry

end apples_and_pears_weight_l60_6041


namespace percentage_discount_of_retail_price_l60_6052

theorem percentage_discount_of_retail_price {wp rp sp discount : ℝ} (h1 : wp = 99) (h2 : rp = 132) (h3 : sp = wp + 0.20 * wp) (h4 : discount = (rp - sp) / rp * 100) : discount = 10 := 
by 
  sorry

end percentage_discount_of_retail_price_l60_6052


namespace Tim_total_score_l60_6092

/-- Given the following conditions:
1. A single line is worth 1000 points.
2. A tetris is worth 8 times a single line.
3. If a single line and a tetris are made consecutively, the score of the tetris doubles.
4. If two tetrises are scored back to back, an additional 5000-point bonus is awarded.
5. If a player scores a single, double and triple line consecutively, a 3000-point bonus is awarded.
6. Tim scored 6 singles, 4 tetrises, 2 doubles, and 1 triple during his game.
7. He made a single line and a tetris consecutively once, scored 2 tetrises back to back, 
   and scored a single, double and triple consecutively.
Prove that Tim’s total score is 54000 points.
-/
theorem Tim_total_score :
  let single_points := 1000
  let tetris_points := 8 * single_points
  let singles := 6 * single_points
  let tetrises := 4 * tetris_points
  let base_score := singles + tetrises
  let consecutive_tetris_bonus := tetris_points
  let back_to_back_tetris_bonus := 5000
  let consecutive_lines_bonus := 3000
  let total_score := base_score + consecutive_tetris_bonus + back_to_back_tetris_bonus + consecutive_lines_bonus
  total_score = 54000 := by
  sorry

end Tim_total_score_l60_6092


namespace evaluate_expression_l60_6083

theorem evaluate_expression (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 2 * x ^ y + 5 * y ^ x - z ^ 2 = 42 :=
by
  sorry

end evaluate_expression_l60_6083


namespace problem1_statement_problem2_statement_l60_6040

-- Defining the sets A and B
def set_A (x : ℝ) := 2*x^2 - 7*x + 3 ≤ 0
def set_B (x a : ℝ) := x + a < 0

-- Problem 1: Intersection of A and B when a = -2
def question1 (x : ℝ) : Prop := set_A x ∧ set_B x (-2)

-- Problem 2: Range of a for A ∩ B = A
def question2 (a : ℝ) : Prop := ∀ x, set_A x → set_B x a

theorem problem1_statement :
  ∀ x, question1 x ↔ x >= 1/2 ∧ x < 2 :=
by sorry

theorem problem2_statement :
  ∀ a, (∀ x, set_A x → set_B x a) ↔ a < -3 :=
by sorry

end problem1_statement_problem2_statement_l60_6040


namespace proof_problem_l60_6062

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as {y | y = 2^x, x ∈ ℝ}
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

-- Define the set B as {x ∈ ℤ | x^2 - 4 ≤ 0}
def B : Set ℤ := {x | x ∈ Set.Icc (-2 : ℤ) 2}

-- Define the complement of A relative to U (universal set)
def CU_A : Set ℝ := {x | x ≤ 0}

-- Define the proposition to be proved
theorem proof_problem :
  (CU_A ∩ (Set.image (coe : ℤ → ℝ) B)) = {-2.0, 1.0, 0.0} :=
by 
  sorry

end proof_problem_l60_6062


namespace subtraction_correct_l60_6010

theorem subtraction_correct : 900000009000 - 123456789123 = 776543220777 :=
by
  -- Placeholder proof to ensure it compiles
  sorry

end subtraction_correct_l60_6010


namespace rectangle_area_l60_6058

theorem rectangle_area {AB AC BC : ℕ} (hAB : AB = 15) (hAC : AC = 17)
  (hRightTriangle : AC * AC = AB * AB + BC * BC) : AB * BC = 120 := by
  sorry

end rectangle_area_l60_6058


namespace order_of_magnitudes_l60_6064

variable (x : ℝ)
variable (a : ℝ)

theorem order_of_magnitudes (h1 : x < 0) (h2 : a = 2 * x) : x^2 < a * x ∧ a * x < a^2 := 
by
  sorry

end order_of_magnitudes_l60_6064


namespace sequence_is_increasing_l60_6059

-- Define the sequence recurrence property
def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3

-- The theorem statement
theorem sequence_is_increasing (a : ℕ → ℤ) (h : sequence_condition a) : 
  ∀ n : ℕ, a n < a (n + 1) :=
by
  unfold sequence_condition at h
  intro n
  specialize h n
  sorry

end sequence_is_increasing_l60_6059


namespace hydroflow_rate_30_minutes_l60_6081

def hydroflow_pumped (rate_per_hour: ℕ) (minutes: ℕ) : ℕ :=
  let hours := minutes / 60
  rate_per_hour * hours

theorem hydroflow_rate_30_minutes : 
  hydroflow_pumped 500 30 = 250 :=
by 
  -- place the proof here
  sorry

end hydroflow_rate_30_minutes_l60_6081


namespace non_zero_const_c_l60_6078

theorem non_zero_const_c (a b c x1 x2 : ℝ) (h1 : x1 ≠ 0) (h2 : x2 ≠ 0) 
(h3 : (a - 1) * x1 ^ 2 + b * x1 + c = 0) 
(h4 : (a - 1) * x2 ^ 2 + b * x2 + c = 0)
(h5 : x1 * x2 = -1) 
(h6 : x1 ≠ x2) 
(h7 : x1 * x2 < 0): c ≠ 0 :=
sorry

end non_zero_const_c_l60_6078


namespace general_term_formula_l60_6082

theorem general_term_formula (S : ℕ → ℤ) (a : ℕ → ℤ) : 
  (∀ n, S n = 3 * n ^ 2 - 2 * n) → 
  (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ a 1 = S 1 → 
  ∀ n, a n = 6 * n - 5 := 
by
  sorry

end general_term_formula_l60_6082


namespace more_than_half_remains_l60_6028

def cubic_block := { n : ℕ // n > 0 }

noncomputable def total_cubes (b : cubic_block) : ℕ := b.val ^ 3

noncomputable def outer_layer_cubes (b : cubic_block) : ℕ := 6 * (b.val ^ 2) - 12 * b.val + 8

noncomputable def remaining_cubes (b : cubic_block) : ℕ := total_cubes b - outer_layer_cubes b

theorem more_than_half_remains (b : cubic_block) (h : b.val = 10) : remaining_cubes b > total_cubes b / 2 :=
by
  sorry

end more_than_half_remains_l60_6028


namespace savings_for_23_students_is_30_yuan_l60_6076

-- Define the number of students
def number_of_students : ℕ := 23

-- Define the price per ticket in yuan
def price_per_ticket : ℕ := 10

-- Define the discount rate for the group ticket
def discount_rate : ℝ := 0.8

-- Define the group size that is eligible for the discount
def group_size_discount : ℕ := 25

-- Define the cost without ticket discount
def cost_without_discount : ℕ := number_of_students * price_per_ticket

-- Define the cost with the group ticket discount
def cost_with_discount : ℝ := price_per_ticket * discount_rate * group_size_discount

-- Define the expected amount saved by using the group discount
def expected_savings : ℝ := cost_without_discount - cost_with_discount

-- Theorem statement that the expected_savings is 30 yuan
theorem savings_for_23_students_is_30_yuan :
  expected_savings = 30 := 
sorry

end savings_for_23_students_is_30_yuan_l60_6076


namespace sum_of_squares_of_coeffs_l60_6051

   theorem sum_of_squares_of_coeffs :
     let expr := 3 * (X^3 - 4 * X^2 + X) - 5 * (X^3 + 2 * X^2 - 5 * X + 3)
     let simplified_expr := -2 * X^3 - 22 * X^2 + 28 * X - 15
     let coefficients := [-2, -22, 28, -15]
     (coefficients.map (λ a => a^2)).sum = 1497 := 
   by 
     -- expending, simplifying and summing up the coefficients 
     sorry
   
end sum_of_squares_of_coeffs_l60_6051


namespace train_length_l60_6026

theorem train_length
  (V L : ℝ)
  (h1 : L = V * 18)
  (h2 : L + 350 = V * 39) :
  L = 300 := 
by
  sorry

end train_length_l60_6026


namespace find_z_value_l60_6044

-- We will define the variables and the given condition
variables {x y z : ℝ}

-- Translate the given condition into Lean
def given_condition (x y z : ℝ) : Prop := (1 / x^2 - 1 / y^2) = (1 / z)

-- State the theorem to prove
theorem find_z_value (x y z : ℝ) (h : given_condition x y z) : 
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end find_z_value_l60_6044


namespace sum_possible_values_l60_6077

theorem sum_possible_values (M : ℝ) (h : M * (M - 6) = -5) : ∀ x ∈ {M | M * (M - 6) = -5}, x + (-x) = 6 :=
by sorry

end sum_possible_values_l60_6077


namespace sum_greater_l60_6068

theorem sum_greater {a b c d : ℝ} (h1 : b + Real.sin a > d + Real.sin c) (h2 : a + Real.sin b > c + Real.sin d) : a + b > c + d := by
  sorry

end sum_greater_l60_6068


namespace Marty_painting_combinations_l60_6055

theorem Marty_painting_combinations :
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  (parts_of_room * colors * methods) = 30 := 
by
  let parts_of_room := 2
  let colors := 5
  let methods := 3
  show (parts_of_room * colors * methods) = 30
  sorry

end Marty_painting_combinations_l60_6055


namespace g_g_g_g_3_l60_6090

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_g_g_g_3_l60_6090


namespace mobius_trip_proof_l60_6087

noncomputable def mobius_trip_time : ℝ :=
  let speed_no_load := 13
  let speed_light_load := 12
  let speed_typical_load := 11
  let distance_total := 257
  let distance_typical := 120
  let distance_light := distance_total - distance_typical
  let time_typical := distance_typical / speed_typical_load
  let time_light := distance_light / speed_light_load
  let time_return := distance_total / speed_no_load
  let rest_first := (20 + 25 + 35) / 60.0
  let rest_second := (45 + 30) / 60.0
  time_typical + time_light + time_return + rest_first + rest_second

theorem mobius_trip_proof : mobius_trip_time = 44.6783 :=
  by sorry

end mobius_trip_proof_l60_6087


namespace ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l60_6049

variables (x y k t : ℕ)

theorem ratio_brothers_sisters_boys (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  (x / (y+1)) = t := 
by simp [h2]

theorem ratio_brothers_sisters_girls (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  ((x+1) / y) = k := 
by simp [h1]

#check ratio_brothers_sisters_boys    -- Just for verification
#check ratio_brothers_sisters_girls   -- Just for verification

end ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l60_6049


namespace meeting_success_probability_l60_6042

noncomputable def meeting_probability : ℝ :=
  let totalVolume := 1.5 ^ 3
  let z_gt_x_y := (1.5 * 1.5 * 1.5) / 3
  let assistants_leave := 2 * ((1.5 * 0.5 / 2) / 3 * 0.5)
  let effectiveVolume := z_gt_x_y - assistants_leave
  let probability := effectiveVolume / totalVolume
  probability

theorem meeting_success_probability :
  meeting_probability = 8 / 27 := by
  sorry

end meeting_success_probability_l60_6042


namespace triangle_circle_square_value_l60_6001

theorem triangle_circle_square_value (Δ : ℝ) (bigcirc : ℝ) (square : ℝ) 
  (h1 : 2 * Δ + 3 * bigcirc + square = 45)
  (h2 : Δ + 5 * bigcirc + 2 * square = 58)
  (h3 : 3 * Δ + bigcirc + 3 * square = 62) :
  Δ + 2 * bigcirc + square = 35 :=
sorry

end triangle_circle_square_value_l60_6001


namespace sum_of_sequence_l60_6080

-- Definitions based on conditions
def a (n : ℕ) := 2 * n - 1
def b (n : ℕ) := 2^(a n) + n
def S (n : ℕ) := (Finset.range n).sum (λ i => b (i + 1))

-- The theorem assertion / problem statement
theorem sum_of_sequence (n : ℕ) : 
  S n = (2 * (4^n - 1)) / 3 + n * (n + 1) / 2 := 
sorry

end sum_of_sequence_l60_6080


namespace least_integer_solution_l60_6029

theorem least_integer_solution (x : ℤ) : (∀ y : ℤ, |2 * y + 9| <= 20 → x ≤ y) ↔ x = -14 := by
  sorry

end least_integer_solution_l60_6029


namespace endpoint_of_parallel_segment_l60_6002

theorem endpoint_of_parallel_segment (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hA : A = (2, 1)) (h_parallel : B.snd = A.snd) (h_length : abs (B.fst - A.fst) = 5) :
  B = (7, 1) ∨ B = (-3, 1) :=
by
  -- Proof goes here
  sorry

end endpoint_of_parallel_segment_l60_6002


namespace sand_exchange_impossible_to_achieve_l60_6032

-- Let G and P be the initial weights of gold and platinum sand, respectively
def initial_G : ℕ := 1 -- 1 kg
def initial_P : ℕ := 1 -- 1 kg

-- Initial values for g and p
def initial_g : ℕ := 1001
def initial_p : ℕ := 1001

-- Daily reduction of either g or p
axiom decrease_g_or_p (g p : ℕ) : g > 1 ∨ p > 1 → (g = g - 1 ∨ p = p - 1) ∧ (g ≥ 1) ∧ (p ≥ 1)

-- Final condition: after 2000 days, g and p both equal to 1
axiom final_g_p_after_2000_days : ∀ (g p : ℕ), (g = initial_g - 2000) ∧ (p = initial_p - 2000) → g = 1 ∧ p = 1

-- State of the system, defined as S = G * p + P * g
def S (G P g p : ℕ) : ℕ := G * p + P * g

-- Prove that after 2000 days, the banker cannot have at least 2 kg of each type of sand
theorem sand_exchange_impossible_to_achieve (G P g p : ℕ) (h : G = initial_G) (h1 : P = initial_P) 
  (h2 : g = initial_g) (h3 : p = initial_p) : 
  ∀ (d : ℕ), (d = 2000) → (g = 1) ∧ (p = 1) 
    → (S G P g p < 4) :=
by
  sorry

end sand_exchange_impossible_to_achieve_l60_6032


namespace gcd_polynomial_correct_l60_6098

noncomputable def gcd_polynomial (b : ℤ) := 5 * b^3 + b^2 + 8 * b + 38

theorem gcd_polynomial_correct (b : ℤ) (h : 342 ∣ b) : Int.gcd (gcd_polynomial b) b = 38 := by
  sorry

end gcd_polynomial_correct_l60_6098


namespace missing_fraction_correct_l60_6063

theorem missing_fraction_correct : 
  (1 / 2) + (-5 / 6) + (1 / 5) + (1 / 4) + (-9 / 20) + (-2 / 15) + (3 / 5) = 0.13333333333333333 :=
by sorry

end missing_fraction_correct_l60_6063


namespace counterexample_exists_l60_6095

theorem counterexample_exists : 
  ∃ (m : ℤ), (∃ (k1 : ℤ), m = 2 * k1) ∧ ¬(∃ (k2 : ℤ), m = 4 * k2) := 
sorry

end counterexample_exists_l60_6095


namespace price_difference_l60_6084

def P := ℝ

def Coupon_A_savings (P : ℝ) := 0.20 * P
def Coupon_B_savings : ℝ := 40
def Coupon_C_savings (P : ℝ) := 0.30 * (P - 120) + 20

def Coupon_A_geq_Coupon_B (P : ℝ) := Coupon_A_savings P ≥ Coupon_B_savings
def Coupon_A_geq_Coupon_C (P : ℝ) := Coupon_A_savings P ≥ Coupon_C_savings P

noncomputable def x : ℝ := 200
noncomputable def y : ℝ := 300

theorem price_difference (P : ℝ) (h1 : P > 120)
  (h2 : Coupon_A_geq_Coupon_B P)
  (h3 : Coupon_A_geq_Coupon_C P) :
  y - x = 100 := by
  sorry

end price_difference_l60_6084


namespace total_action_figures_l60_6025

def action_figures_per_shelf : ℕ := 11
def number_of_shelves : ℕ := 4

theorem total_action_figures : action_figures_per_shelf * number_of_shelves = 44 := by
  sorry

end total_action_figures_l60_6025


namespace shortest_side_of_triangle_l60_6054

noncomputable def triangle_shortest_side (AB : ℝ) (AD : ℝ) (DB : ℝ) (radius : ℝ) : ℝ :=
  let x := 6
  let y := 5
  2 * y

theorem shortest_side_of_triangle :
  let AB := 16
  let AD := 7
  let DB := 9
  let radius := 5
  AB = AD + DB →
  (AD = 7) ∧ (DB = 9) ∧ (radius = 5) →
  triangle_shortest_side AB AD DB radius = 10 :=
by
  intros h1 h2
  -- proof goes here
  sorry

end shortest_side_of_triangle_l60_6054


namespace product_mb_gt_one_l60_6060

theorem product_mb_gt_one (m b : ℝ) (hm : m = 3 / 4) (hb : b = 2) : m * b = 3 / 2 := by
  sorry

end product_mb_gt_one_l60_6060


namespace volume_to_surface_area_ratio_l60_6074

theorem volume_to_surface_area_ratio (base_layer: ℕ) (top_layer: ℕ) (unit_cube_volume: ℕ) (unit_cube_faces_exposed_base: ℕ) (unit_cube_faces_exposed_top: ℕ) 
  (V : ℕ := base_layer * top_layer * unit_cube_volume) 
  (S : ℕ := base_layer * unit_cube_faces_exposed_base + top_layer * unit_cube_faces_exposed_top) 
  (ratio := V / S) : ratio = 1 / 2 :=
by
  -- Base Layer: 4 cubes, 3 faces exposed per cube
  have base_layer_faces : ℕ := 4 * 3
  -- Top Layer: 4 cubes, 1 face exposed per cube
  have top_layer_faces : ℕ := 4 * 1
  -- Total volume is 8
  have V : ℕ := 4 * 2
  -- Total surface area is 16
  have S : ℕ := base_layer_faces + top_layer_faces
  -- Volume to surface area ratio computation
  have ratio : ℕ := V / S
  sorry

end volume_to_surface_area_ratio_l60_6074


namespace domain_of_function_l60_6088

def function_undefined_at (x : ℝ) : Prop :=
  ∃ y : ℝ, y = (x - 3) / (x - 2)

theorem domain_of_function (x : ℝ) : ¬(x = 2) ↔ function_undefined_at x :=
sorry

end domain_of_function_l60_6088


namespace intersection_P_Q_l60_6036

def set_P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def set_Q : Set ℝ := {x | (x - 1) ^ 2 ≤ 4}

theorem intersection_P_Q :
  {x | x ∈ set_P ∧ x ∈ set_Q} = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_P_Q_l60_6036


namespace petya_coin_difference_20_l60_6045

-- Definitions for the problem conditions
variables (n k : ℕ) -- n: number of 5-ruble coins Petya has, k: number of 2-ruble coins Petya has

-- Condition: Petya has 60 rubles more than Vanya
def petya_has_60_more (n k : ℕ) : Prop := (5 * n + 2 * k = 5 * k + 2 * n + 60)

-- Theorem to prove Petya has 20 more 5-ruble coins than 2-ruble coins
theorem petya_coin_difference_20 (n k : ℕ) (h : petya_has_60_more n k) : n - k = 20 :=
sorry

end petya_coin_difference_20_l60_6045


namespace distance_on_map_is_correct_l60_6030

-- Define the parameters
def time_hours : ℝ := 1.5
def speed_mph : ℝ := 60
def map_scale_inches_per_mile : ℝ := 0.05555555555555555

-- Define the computation of actual distance and distance on the map
def actual_distance_miles : ℝ := speed_mph * time_hours
def distance_on_map_inches : ℝ := actual_distance_miles * map_scale_inches_per_mile

-- Theorem statement
theorem distance_on_map_is_correct :
  distance_on_map_inches = 5 :=
by 
  sorry

end distance_on_map_is_correct_l60_6030


namespace parking_lot_total_spaces_l60_6086

-- Given conditions
def section1_spaces := 320
def section2_spaces := 440
def section3_spaces := section2_spaces - 200
def total_spaces := section1_spaces + section2_spaces + section3_spaces

-- Problem statement to be proved
theorem parking_lot_total_spaces : total_spaces = 1000 :=
by
  sorry

end parking_lot_total_spaces_l60_6086


namespace convert_to_base_k_l60_6018

noncomputable def base_k_eq (k : ℕ) : Prop :=
  4 * k + 4 = 36

theorem convert_to_base_k :
  ∃ k : ℕ, base_k_eq k ∧ (67 / k^2 % k^2 % k = 1 ∧ 67 / k % k = 0 ∧ 67 % k = 3) :=
sorry

end convert_to_base_k_l60_6018


namespace factorize_problem_1_factorize_problem_2_l60_6079

-- Problem 1 Statement
theorem factorize_problem_1 (a m : ℝ) : 2 * a * m^2 - 8 * a = 2 * a * (m + 2) * (m - 2) := 
sorry

-- Problem 2 Statement
theorem factorize_problem_2 (x y : ℝ) : (x - y)^2 + 4 * (x * y) = (x + y)^2 := 
sorry

end factorize_problem_1_factorize_problem_2_l60_6079


namespace count_valid_n_decomposition_l60_6047

theorem count_valid_n_decomposition : 
  ∃ (count : ℕ), count = 108 ∧ 
  ∀ (a b c n : ℕ), 
    8 * a + 88 * b + 888 * c = 8000 → 
    0 ≤ b ∧ b ≤ 90 → 
    0 ≤ c ∧ c ≤ 9 → 
    n = a + 2 * b + 3 * c → 
    n < 1000 :=
sorry

end count_valid_n_decomposition_l60_6047


namespace tangent_line_parabola_l60_6019

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → ∃! x y, y^2 = 12 * x) → d = 3 := 
by
  intro h
  -- Here, "h" would be our hypothesis where we assume the line is tangent to the parabola
  sorry

end tangent_line_parabola_l60_6019


namespace more_than_3000_students_l60_6050

-- Define the conditions
def students_know_secret (n : ℕ) : ℕ :=
  3 ^ (n - 1)

-- Define the statement to prove
theorem more_than_3000_students : ∃ n : ℕ, students_know_secret n > 3000 ∧ n = 9 := by
  sorry

end more_than_3000_students_l60_6050


namespace area_of_EFGH_l60_6069

-- Define the dimensions of the smaller rectangles
def smaller_rectangle_short_side : ℕ := 7
def smaller_rectangle_long_side : ℕ := 2 * smaller_rectangle_short_side

-- Define the configuration of rectangles
def width_EFGH : ℕ := 2 * smaller_rectangle_short_side
def length_EFGH : ℕ := smaller_rectangle_long_side

-- Prove that the area of rectangle EFGH is 196 square feet
theorem area_of_EFGH : width_EFGH * length_EFGH = 196 := by
  sorry

end area_of_EFGH_l60_6069


namespace pirates_divide_coins_l60_6024

theorem pirates_divide_coins (N : ℕ) (hN : 220 ≤ N ∧ N ≤ 300) :
  ∃ n : ℕ, 
    (N - 2 - (N - 2) / 3 - 2 - (2 * ((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3) - 
    2 - (2 * (((N - 2) / 3 - (2 * ((N - 2) / 3) / 3)) / 3)) / 3) / 3 = 84 := 
sorry

end pirates_divide_coins_l60_6024


namespace inequality_solution_exists_l60_6000

theorem inequality_solution_exists (x m : ℝ) (h1: 1 < x) (h2: x ≤ 2) (h3: x > m) : m < 2 :=
sorry

end inequality_solution_exists_l60_6000


namespace homer_total_points_l60_6097

noncomputable def first_try_points : ℕ := 400
noncomputable def second_try_points : ℕ := first_try_points - 70
noncomputable def third_try_points : ℕ := 2 * second_try_points
noncomputable def total_points : ℕ := first_try_points + second_try_points + third_try_points

theorem homer_total_points : total_points = 1390 :=
by
  -- Using the definitions above, we need to show that total_points = 1390
  sorry

end homer_total_points_l60_6097


namespace length_of_segment_XY_l60_6007

noncomputable def rectangle_length (A B C D : ℝ) (BX DY : ℝ) : ℝ :=
  2 * BX + DY

theorem length_of_segment_XY (A B C D : ℝ) (BX DY : ℝ) (h1 : C = 2 * B) (h2 : BX = 4) (h3 : DY = 10) :
  rectangle_length A B C D BX DY = 13 :=
by
  rw [rectangle_length, h2, h3]
  sorry

end length_of_segment_XY_l60_6007


namespace cost_of_drill_bits_l60_6037

theorem cost_of_drill_bits (x : ℝ) (h1 : 5 * x + 0.10 * (5 * x) = 33) : x = 6 :=
sorry

end cost_of_drill_bits_l60_6037


namespace cost_to_fill_half_of_CanB_l60_6021

theorem cost_to_fill_half_of_CanB (r h : ℝ) (C_cost : ℝ) (VC VB : ℝ) 
(h1 : VC = 2 * VB) 
(h2 : VB = Real.pi * r^2 * h) 
(h3 : VC = Real.pi * (2 * r)^2 * (h / 2)) 
(h4 : C_cost = 16):
  C_cost / 4 = 4 :=
by
  sorry

end cost_to_fill_half_of_CanB_l60_6021


namespace rowing_downstream_speed_l60_6023

-- Define the given conditions
def V_u : ℝ := 60  -- speed upstream in kmph
def V_s : ℝ := 75  -- speed in still water in kmph

-- Define the problem statement
theorem rowing_downstream_speed : ∃ (V_d : ℝ), V_s = (V_u + V_d) / 2 ∧ V_d = 90 :=
by
  sorry

end rowing_downstream_speed_l60_6023


namespace squirrel_divides_acorns_l60_6004

theorem squirrel_divides_acorns (total_acorns parts_per_month remaining_acorns month_acorns winter_months spring_acorns : ℕ)
  (h1 : total_acorns = 210)
  (h2 : parts_per_month = 3)
  (h3 : winter_months = 3)
  (h4 : remaining_acorns = 60)
  (h5 : month_acorns = total_acorns / winter_months)
  (h6 : spring_acorns = 30)
  (h7 : month_acorns - remaining_acorns = spring_acorns / parts_per_month) :
  parts_per_month = 3 :=
by
  sorry

end squirrel_divides_acorns_l60_6004


namespace ratio_of_new_time_to_previous_time_l60_6033

-- Given conditions
def distance : ℕ := 288
def initial_time : ℕ := 6
def new_speed : ℕ := 32

-- Question: Prove the ratio of the new time to the previous time is 3:2
theorem ratio_of_new_time_to_previous_time :
  (distance / new_speed) / initial_time = 3 / 2 :=
by
  sorry

end ratio_of_new_time_to_previous_time_l60_6033


namespace simplify_fraction_l60_6094

theorem simplify_fraction:
  ((1/2 - 1/3) / (3/7 + 1/9)) * (1/4) = 21/272 :=
by
  sorry

end simplify_fraction_l60_6094


namespace abs_eq_five_l60_6022

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  sorry

end abs_eq_five_l60_6022


namespace common_difference_and_first_three_terms_l60_6005

-- Given condition that for any n, the sum of the first n terms of an arithmetic progression is equal to 5n^2.
def arithmetic_sum_property (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 5 * n ^ 2

-- Define the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ :=
  a1 + (n-1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_first_n_terms (a1 d n : ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d)/2

-- Conditions and prove that common difference d is 10 and the first three terms are 5, 15, and 25
theorem common_difference_and_first_three_terms :
  (∃ (a1 d : ℕ), arithmetic_sum_property (sum_first_n_terms a1 d) ∧ d = 10 ∧ nth_term a1 d 1 = 5 ∧ nth_term a1 d 2 = 15 ∧ nth_term a1 d 3  = 25) :=
sorry

end common_difference_and_first_three_terms_l60_6005


namespace bananas_per_monkey_l60_6065

-- Define the given conditions
def total_monkeys : ℕ := 12
def piles_with_9hands : ℕ := 6
def hands_per_pile_9hands : ℕ := 9
def bananas_per_hand_9hands : ℕ := 14
def piles_with_12hands : ℕ := 4
def hands_per_pile_12hands : ℕ := 12
def bananas_per_hand_12hands : ℕ := 9

-- Calculate the total number of bananas from each type of pile
def total_bananas_9hands : ℕ := piles_with_9hands * hands_per_pile_9hands * bananas_per_hand_9hands
def total_bananas_12hands : ℕ := piles_with_12hands * hands_per_pile_12hands * bananas_per_hand_12hands

-- Sum the total number of bananas
def total_bananas : ℕ := total_bananas_9hands + total_bananas_12hands

-- Prove that each monkey gets 99 bananas
theorem bananas_per_monkey : total_bananas / total_monkeys = 99 := by
  sorry

end bananas_per_monkey_l60_6065


namespace inequality_proof_l60_6003

theorem inequality_proof (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end inequality_proof_l60_6003


namespace dealer_gross_profit_l60_6031

variable (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ)

def desk_problem (purchase_price : ℝ) (markup_rate : ℝ) (gross_profit : ℝ) : Prop :=
  ∀ (S : ℝ), S = purchase_price + markup_rate * S → gross_profit = S - purchase_price

theorem dealer_gross_profit : desk_problem 150 0.5 150 :=
by 
  sorry

end dealer_gross_profit_l60_6031


namespace quadratic_factors_l60_6057

-- Define the quadratic polynomial
def quadratic (b c x : ℝ) : ℝ := x^2 + b * x + c

-- Define the roots
def root1 : ℝ := -2
def root2 : ℝ := 3

-- Theorem: If the quadratic equation has roots -2 and 3, then it factors as (x + 2)(x - 3)
theorem quadratic_factors (b c : ℝ) (h1 : quadratic b c root1 = 0) (h2 : quadratic b c root2 = 0) :
  ∀ x : ℝ, quadratic b c x = (x + 2) * (x - 3) :=
by
  sorry

end quadratic_factors_l60_6057


namespace georgie_guacamole_servings_l60_6017

-- Define the conditions
def avocados_needed_per_serving : Nat := 3
def initial_avocados : Nat := 5
def additional_avocados : Nat := 4

-- State the target number of servings Georgie can make
def total_avocados := initial_avocados + additional_avocados
def guacamole_servings := total_avocados / avocados_needed_per_serving

-- Lean 4 statement asserting the number of servings equals 3
theorem georgie_guacamole_servings : guacamole_servings = 3 := by
  sorry

end georgie_guacamole_servings_l60_6017


namespace set_intersection_complement_l60_6016

theorem set_intersection_complement (U M N : Set ℤ)
  (hU : U = {0, -1, -2, -3, -4})
  (hM : M = {0, -1, -2})
  (hN : N = {0, -3, -4}) :
  (U \ M) ∩ N = {-3, -4} :=
by
  sorry

end set_intersection_complement_l60_6016


namespace estimate_passed_students_l60_6020

-- Definitions for the given conditions
def total_papers_in_city : ℕ := 5000
def papers_selected : ℕ := 400
def papers_passed : ℕ := 360

-- The theorem stating the problem in Lean
theorem estimate_passed_students : 
    (5000:ℕ) * ((360:ℕ) / (400:ℕ)) = (4500:ℕ) :=
by
  -- Providing a trivial sorry to skip the proof.
  sorry

end estimate_passed_students_l60_6020


namespace no_carry_consecutive_pairs_l60_6012

/-- Consider the range of integers {2000, 2001, ..., 3000}. 
    We determine that the number of pairs of consecutive integers in this range such that their addition requires no carrying is 729. -/
theorem no_carry_consecutive_pairs : 
  ∀ (n : ℕ), (2000 ≤ n ∧ n < 3000) ∧ ((n + 1) ≤ 3000) → 
  ∃ (count : ℕ), count = 729 := 
sorry

end no_carry_consecutive_pairs_l60_6012


namespace total_monthly_bill_working_from_home_l60_6091

-- Definitions based on conditions
def original_bill : ℝ := 60
def increase_rate : ℝ := 0.45
def additional_internet_cost : ℝ := 25
def additional_cloud_cost : ℝ := 15

-- The theorem to prove
theorem total_monthly_bill_working_from_home : 
  original_bill * (1 + increase_rate) + additional_internet_cost + additional_cloud_cost = 127 := by
  sorry

end total_monthly_bill_working_from_home_l60_6091


namespace percentage_difference_l60_6093

theorem percentage_difference (x : ℝ) (h1 : 0.38 * 80 = 30.4) (h2 : 30.4 - (x / 100) * 160 = 11.2) :
    x = 12 :=
by
  sorry

end percentage_difference_l60_6093


namespace find_x_l60_6014

theorem find_x (x y : ℝ) (h1 : y = 1 / (2 * x + 2)) (h2 : y = 2) : x = -3 / 4 :=
by
  sorry

end find_x_l60_6014


namespace pizza_slices_l60_6067

theorem pizza_slices (S L : ℕ) (h1 : S + L = 36) (h2 : L = 2 * S) :
  (8 * S + 12 * L) = 384 :=
by
  sorry

end pizza_slices_l60_6067


namespace solve_for_a_l60_6071

-- Define the line equation and the condition of equal intercepts
def line_eq (a x y : ℝ) : Prop :=
  a * x + y - 2 - a = 0

def equal_intercepts (a : ℝ) : Prop :=
  (∀ x, line_eq a x 0 → x = 2 + a) ∧ (∀ y, line_eq a 0 y → y = 2 + a)

-- State the problem to prove the value of 'a'
theorem solve_for_a (a : ℝ) : equal_intercepts a → (a = -2 ∨ a = 1) :=
by
  sorry

end solve_for_a_l60_6071


namespace rigid_motion_pattern_l60_6056

-- Define the types of transformations
inductive Transformation
| rotation : ℝ → Transformation -- rotation by an angle
| translation : ℝ → Transformation -- translation by a distance
| reflection_across_m : Transformation -- reflection across line m
| reflection_perpendicular_to_m : ℝ → Transformation -- reflective across line perpendicular to m at a point

-- Define the problem statement conditions
def pattern_alternates (line_m : ℝ → ℝ) : Prop := sorry -- This should define the alternating pattern of equilateral triangles and squares along line m

-- Problem statement in Lean
theorem rigid_motion_pattern (line_m : ℝ → ℝ) (p : Transformation → Prop)
    (h1 : p (Transformation.rotation 180)) -- 180-degree rotation is a valid transformation for the pattern
    (h2 : ∀ d, p (Transformation.translation d)) -- any translation by pattern unit length is a valid transformation
    (h3 : p Transformation.reflection_across_m) -- reflection across line m is a valid transformation
    (h4 : ∀ x, p (Transformation.reflection_perpendicular_to_m x)) -- reflection across any perpendicular line is a valid transformation
    : ∃ t : Finset Transformation, t.card = 4 ∧ ∀ t_val, t_val ∈ t → p t_val ∧ t_val ≠ Transformation.rotation 0 := 
sorry

end rigid_motion_pattern_l60_6056


namespace hyperbola_same_foci_as_ellipse_eccentricity_two_l60_6015

theorem hyperbola_same_foci_as_ellipse_eccentricity_two
  (a b c e : ℝ)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (a = 5 ∧ b = 3 ∧ c = 4))
  (eccentricity_eq : e = 2) :
  ∃ x y : ℝ, (x^2 / (c / e)^2 - y^2 / (c^2 - (c / e)^2) = 1) ↔ (x^2 / 4 - y^2 / 12 = 1) :=
by
  sorry

end hyperbola_same_foci_as_ellipse_eccentricity_two_l60_6015


namespace double_root_conditions_l60_6089

theorem double_root_conditions (k : ℝ) :
  (∃ x, (k - 1)/(x^2 - 1) - 1/(x - 1) = k/(x + 1) ∧ (∀ ε > 0, (∃ δ > 0, (∀ y, |y - x| < δ → (k - 1)/(y^2 - 1) - 1/(y - 1) = k/(y + 1)))))
  → k = 3 ∨ k = 1/3 :=
sorry

end double_root_conditions_l60_6089


namespace problem_statement_l60_6072

-- Definitions from the problem conditions
variable (r : ℝ) (A B C : ℝ)

-- Problem condition that A, B are endpoints of the diameter of the circle
-- Defining the length AB being the diameter -> length AB = 2r
def AB := 2 * r

-- Condition that ABC is inscribed in a circle and AB is the diameter implies the angle ACB = 90°
-- Using Thales' theorem we know that A, B, C satisfy certain geometric properties in a right triangle
-- AC and BC are the other two sides with H right angle at C.

-- Proving the target equation
theorem problem_statement (h : C ≠ A ∧ C ≠ B) : (AC + BC)^2 ≤ 8 * r^2 := 
sorry


end problem_statement_l60_6072


namespace ellipse_equation_l60_6053

theorem ellipse_equation (a b : ℝ) (A : ℝ × ℝ)
  (hA : A = (-3, 1.75))
  (he : 0.75 = Real.sqrt (a^2 - b^2) / a) 
  (hcond : (Real.sqrt (a^2 - b^2) / a) = 0.75) :
  (16 = a^2) ∧ (7 = b^2) :=
by
  have h1 : A = (-3, 1.75) := hA
  have h2 : Real.sqrt (a^2 - b^2) / a = 0.75 := hcond
  sorry

end ellipse_equation_l60_6053


namespace inverse_of_g_compose_three_l60_6099

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Assuming g(x) is defined only for x in {1, 2, 3, 4, 5}

noncomputable def g_inv (y : ℕ) : ℕ :=
  match y with
  | 4 => 1
  | 3 => 2
  | 1 => 3
  | 5 => 4
  | 2 => 5
  | _ => 0  -- Assuming g_inv(y) is defined only for y in {1, 3, 1, 5, 2}

theorem inverse_of_g_compose_three : g_inv (g_inv (g_inv 3)) = 4 := by
  sorry

end inverse_of_g_compose_three_l60_6099


namespace greatest_visible_unit_cubes_from_single_point_l60_6073

-- Define the size of the cube
def cube_size : ℕ := 9

-- The total number of unit cubes in the 9x9x9 cube
def total_unit_cubes (n : ℕ) : ℕ := n^3

-- The greatest number of unit cubes visible from a single point
def visible_unit_cubes (n : ℕ) : ℕ := 3 * n^2 - 3 * (n - 1) + 1

-- The given cube size is 9
def given_cube_size : ℕ := cube_size

-- The correct answer for the greatest number of visible unit cubes from a single point
def correct_visible_cubes : ℕ := 220

-- Theorem stating the visibility calculation for a 9x9x9 cube
theorem greatest_visible_unit_cubes_from_single_point :
  visible_unit_cubes cube_size = correct_visible_cubes := by
  sorry

end greatest_visible_unit_cubes_from_single_point_l60_6073


namespace smallest_N_circular_table_l60_6096

theorem smallest_N_circular_table (N chairs : ℕ) (circular_seating : N < chairs) :
  (∀ new_person_reserved : ℕ, new_person_reserved < chairs →
    (∃ i : ℕ, (i < N) ∧ (new_person_reserved = (i + 1) % chairs ∨ 
                           new_person_reserved = (i - 1) % chairs))) ↔ N = 18 := by
sorry

end smallest_N_circular_table_l60_6096


namespace slope_product_is_neg_one_l60_6034

noncomputable def slope_product (m n : ℝ) : ℝ := m * n

theorem slope_product_is_neg_one 
  (m n : ℝ)
  (eqn1 : ∀ x, ∃ y, y = m * x)
  (eqn2 : ∀ x, ∃ y, y = n * x)
  (angle : ∃ θ1 θ2 : ℝ, θ1 = θ2 + π / 4)
  (neg_reciprocal : m = -1 / n):
  slope_product m n = -1 := 
sorry

end slope_product_is_neg_one_l60_6034


namespace diagonal_angle_with_plane_l60_6038

theorem diagonal_angle_with_plane (α : ℝ) {a : ℝ} 
  (h_square: a > 0)
  (θ : ℝ := Real.arcsin ((Real.sin α) / Real.sqrt 2)): 
  ∃ (β : ℝ), β = θ :=
sorry

end diagonal_angle_with_plane_l60_6038


namespace initial_pieces_count_l60_6027

theorem initial_pieces_count (people : ℕ) (pieces_per_person : ℕ) (leftover_pieces : ℕ) :
  people = 6 → pieces_per_person = 7 → leftover_pieces = 3 → people * pieces_per_person + leftover_pieces = 45 :=
by
  intros h_people h_pieces_per_person h_leftover_pieces
  sorry

end initial_pieces_count_l60_6027


namespace intersection_complement_U_l60_6035

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def B_complement_U : Set ℕ := U \ B

theorem intersection_complement_U (hU : U = {1, 3, 5, 7}) 
                                  (hA : A = {3, 5}) 
                                  (hB : B = {1, 3, 7}) : 
  A ∩ (B_complement_U U B) = {5} := by
  sorry

end intersection_complement_U_l60_6035


namespace geometric_sequence_sum_l60_6011

open Nat

noncomputable def geometric_sum (a q n : ℕ) : ℕ :=
  a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (S : ℕ → ℕ) (q a₁ : ℕ)
  (h_q: q = 2)
  (h_S5: S 5 = 1)
  (h_S: ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) :
  S 10 = 33 :=
by
  sorry

end geometric_sequence_sum_l60_6011


namespace price_jemma_sells_each_frame_is_5_l60_6075

noncomputable def jemma_price_per_frame : ℝ :=
  let num_frames_jemma := 400
  let num_frames_dorothy := num_frames_jemma / 2
  let total_income := 2500
  let P_jemma := total_income / (num_frames_jemma + num_frames_dorothy / 2)
  P_jemma

theorem price_jemma_sells_each_frame_is_5 :
  jemma_price_per_frame = 5 := by
  sorry

end price_jemma_sells_each_frame_is_5_l60_6075


namespace negation_of_exists_proposition_l60_6046

theorem negation_of_exists_proposition :
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
sorry

end negation_of_exists_proposition_l60_6046


namespace tangent_line_eq_l60_6085

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 + x + 1) (h_point : x = 1 ∧ y = 3) : 
  y = 4 * x - 1 := 
sorry

end tangent_line_eq_l60_6085


namespace simplify_expression_l60_6039

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  (3 * (a^2 + a * b + b^2) / (4 * (a + b))) * (2 * (a^2 - b^2) / (9 * (a^3 - b^3))) = 
  1 / 6 := 
by
  -- Placeholder for proof steps
  sorry

end simplify_expression_l60_6039


namespace increased_numerator_value_l60_6066

theorem increased_numerator_value (x y a : ℝ) (h1 : x / y = 2 / 5) (h2 : (x + a) / (2 * y) = 1 / 3) (h3 : x + y = 5.25) : a = 1 :=
by
  -- skipped proof: sorry
  sorry

end increased_numerator_value_l60_6066


namespace number_of_people_per_van_l60_6061

theorem number_of_people_per_van (num_students : ℕ) (num_adults : ℕ) (num_vans : ℕ) (total_people : ℕ) (people_per_van : ℕ) :
  num_students = 40 →
  num_adults = 14 →
  num_vans = 6 →
  total_people = num_students + num_adults →
  people_per_van = total_people / num_vans →
  people_per_van = 9 :=
by
  intros h_students h_adults h_vans h_total h_div
  sorry

end number_of_people_per_van_l60_6061


namespace Alice_min_speed_l60_6006

theorem Alice_min_speed (d : ℝ) (v_bob : ℝ) (delta_t : ℝ) (v_alice : ℝ) :
  d = 180 ∧ v_bob = 40 ∧ delta_t = 0.5 ∧ 0 < v_alice ∧ v_alice * (d / v_bob - delta_t) ≥ d →
  v_alice > 45 :=
by
  sorry

end Alice_min_speed_l60_6006


namespace mira_jogs_hours_each_morning_l60_6008

theorem mira_jogs_hours_each_morning 
  (h : ℝ) -- number of hours Mira jogs each morning
  (speed : ℝ) -- Mira's jogging speed in miles per hour
  (days : ℝ) -- number of days Mira jogs
  (total_distance : ℝ) -- total distance Mira jogs

  (H1 : speed = 5) 
  (H2 : days = 5) 
  (H3 : total_distance = 50) 
  (H4 : total_distance = speed * h * days) :

  h = 2 :=
by
  sorry

end mira_jogs_hours_each_morning_l60_6008


namespace find_x_l60_6048

theorem find_x :
    ∃ x : ℚ, (1/7 + 7/x = 15/x + 1/15) ∧ x = 105 := by
  sorry

end find_x_l60_6048
