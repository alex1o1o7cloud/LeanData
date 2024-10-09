import Mathlib

namespace work_done_by_A_alone_l1680_168006

theorem work_done_by_A_alone (Wb : ℝ) (Wa : ℝ) (D : ℝ) :
  Wa = 3 * Wb →
  (Wb + Wa) * 18 = D →
  D = 72 → 
  (D / Wa) = 24 := 
by
  intros h1 h2 h3
  sorry

end work_done_by_A_alone_l1680_168006


namespace total_cost_price_is_584_l1680_168013

-- Define the costs of individual items
def cost_watch : ℕ := 144
def cost_bracelet : ℕ := 250
def cost_necklace : ℕ := 190

-- The proof statement: the total cost price is 584
theorem total_cost_price_is_584 : cost_watch + cost_bracelet + cost_necklace = 584 :=
by
  -- We skip the proof steps here, assuming the above definitions are correct.
  sorry

end total_cost_price_is_584_l1680_168013


namespace division_of_decimals_l1680_168099

theorem division_of_decimals : (0.45 : ℝ) / (0.005 : ℝ) = 90 := 
sorry

end division_of_decimals_l1680_168099


namespace termite_ridden_fraction_l1680_168003

theorem termite_ridden_fraction:
  ∀ T: ℝ, (3 / 4) * T = 1 / 4 → T = 1 / 3 :=
by
  intro T
  intro h
  sorry

end termite_ridden_fraction_l1680_168003


namespace part_a_l1680_168075

theorem part_a (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x * y ≠ 1) :
  (x * y) / (1 - x * y) = x / (1 - x) + y / (1 - y) :=
sorry

end part_a_l1680_168075


namespace smallest_number_divisible_by_conditions_l1680_168083

theorem smallest_number_divisible_by_conditions:
  ∃ n : ℕ, (∀ d ∈ [8, 12, 22, 24], d ∣ (n - 12)) ∧ (n = 252) :=
by
  sorry

end smallest_number_divisible_by_conditions_l1680_168083


namespace meal_combinations_correct_l1680_168066

-- Let E denote the total number of dishes on the menu
def E : ℕ := 12

-- Let V denote the number of vegetarian dishes on the menu
def V : ℕ := 5

-- Define the function that computes the number of different combinations of meals Elena and Nasir can order
def meal_combinations (e : ℕ) (v : ℕ) : ℕ :=
  e * v

-- The theorem to prove that the number of different combinations of meals Elena and Nasir can order is 60
theorem meal_combinations_correct : meal_combinations E V = 60 := by
  sorry

end meal_combinations_correct_l1680_168066


namespace real_root_range_of_a_l1680_168014

theorem real_root_range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ (0 ≤ a ∧ a ≤ 1/4) :=
by
  sorry

end real_root_range_of_a_l1680_168014


namespace geometric_increasing_condition_l1680_168030

structure GeometricSequence (a₁ q : ℝ) (a : ℕ → ℝ) :=
  (rec_rel : ∀ n : ℕ, a (n + 1) = a n * q)

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a₁ q : ℝ) (a : ℕ → ℝ) (h : GeometricSequence a₁ q a) :
  ¬ (q > 1 ↔ is_increasing a) := sorry

end geometric_increasing_condition_l1680_168030


namespace not_divisible_by_n_l1680_168091

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬n ∣ 2^n - 1 :=
by
  -- proof to be filled in
  sorry

end not_divisible_by_n_l1680_168091


namespace find_length_BF_l1680_168007

-- Define the conditions
structure Rectangle :=
  (short_side : ℝ)
  (long_side : ℝ)

def folded_paper (rect : Rectangle) : Prop :=
  rect.short_side = 12

def congruent_triangles (rect : Rectangle) : Prop :=
  rect.short_side = 12

-- Define the length of BF to prove
def length_BF (rect : Rectangle) : ℝ := 10

-- The theorem statement
theorem find_length_BF (rect : Rectangle) (h1 : folded_paper rect) (h2 : congruent_triangles rect) :
  length_BF rect = 10 := 
  sorry

end find_length_BF_l1680_168007


namespace exponential_difference_l1680_168044

theorem exponential_difference (f : ℕ → ℕ) (x : ℕ) (h : f x = 3^x) : f (x + 2) - f x = 8 * f x :=
by sorry

end exponential_difference_l1680_168044


namespace find_m_l1680_168040

theorem find_m 
  (m : ℕ) 
  (hm_pos : 0 < m) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := 
sorry

end find_m_l1680_168040


namespace num_integer_solutions_prime_l1680_168074

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 ∧ m < n → n % m ≠ 0

def integer_solutions : List ℤ := [-1, 3]

theorem num_integer_solutions_prime :
  (∀ x ∈ integer_solutions, is_prime (|15 * x^2 - 32 * x - 28|)) ∧ (integer_solutions.length = 2) :=
by
  sorry

end num_integer_solutions_prime_l1680_168074


namespace find_b_l1680_168018

theorem find_b (g : ℝ → ℝ) (g_inv : ℝ → ℝ) (b : ℝ) (h_g_def : ∀ x, g x = 1 / (3 * x + b)) (h_g_inv_def : ∀ x, g_inv x = (1 - 3 * x) / (3 * x)) :
  b = 3 :=
by
  sorry

end find_b_l1680_168018


namespace pyramid_transport_volume_l1680_168096

-- Define the conditions of the problem
def pyramid_height : ℝ := 15
def pyramid_base_side_length : ℝ := 8
def box_length : ℝ := 10
def box_width : ℝ := 10
def box_height : ℝ := 15

-- Define the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- State the theorem
theorem pyramid_transport_volume : box_volume = 1500 := by
  sorry

end pyramid_transport_volume_l1680_168096


namespace quadratic_real_roots_range_l1680_168037

theorem quadratic_real_roots_range (a : ℝ) : 
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ (a ≤ 2) :=
by
-- Proof outline:
-- Case 1: when a = 1, the equation simplifies to -2x + 1 = 0, which has a real solution x = 1/2.
-- Case 2: when a ≠ 1, the quadratic equation has real roots if the discriminant 8 - 4a ≥ 0, i.e., 2 ≥ a.
sorry

end quadratic_real_roots_range_l1680_168037


namespace chlorine_moles_l1680_168092

theorem chlorine_moles (methane_used chlorine_used chloromethane_formed : ℕ)
  (h_combined_methane : methane_used = 3)
  (h_formed_chloromethane : chloromethane_formed = 3)
  (balanced_eq : methane_used = chloromethane_formed) :
  chlorine_used = 3 :=
by
  have h : chlorine_used = methane_used := by sorry
  rw [h_combined_methane] at h
  exact h

end chlorine_moles_l1680_168092


namespace cos_double_angle_l1680_168089

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3/5) : Real.cos (2 * θ) = -7 / 25 := 
sorry

end cos_double_angle_l1680_168089


namespace find_unknown_number_l1680_168027

theorem find_unknown_number (x : ℤ) :
  (20 + 40 + 60) / 3 = 5 + (20 + 60 + x) / 3 → x = 25 :=
by
  sorry

end find_unknown_number_l1680_168027


namespace no_rational_roots_l1680_168093

theorem no_rational_roots (a b c d : ℕ) (h1 : 1000 * a + 100 * b + 10 * c + d = p) (h2 : Prime p) (h3: Nat.digits 10 p = [a, b, c, d]) : 
  ¬ ∃ x : ℚ, a * x^3 + b * x^2 + c * x + d = 0 :=
by
  sorry

end no_rational_roots_l1680_168093


namespace find_m_l1680_168029

theorem find_m
  (h1 : ∃ (m : ℝ), ∃ (focus_parabola : ℝ × ℝ), focus_parabola = (0, 1/2)
       ∧ ∃ (focus_ellipse : ℝ × ℝ), focus_ellipse = (0, Real.sqrt (m - 2))
       ∧ focus_parabola = focus_ellipse) :
  ∃ (m : ℝ), m = 9/4 :=
by
  sorry

end find_m_l1680_168029


namespace transform_polynomial_l1680_168058

variable (x z : ℝ)

theorem transform_polynomial (h1 : z = x - 1 / x) (h2 : x^4 - 3 * x^3 - 2 * x^2 + 3 * x + 1 = 0) :
  x^2 * (z^2 - 3 * z) = 0 :=
sorry

end transform_polynomial_l1680_168058


namespace coin_stack_count_l1680_168012

theorem coin_stack_count
  (TN : ℝ := 1.95)
  (TQ : ℝ := 1.75)
  (SH : ℝ := 20)
  (n q : ℕ) :
  (n*Tℕ + q*TQ = SH) → (n + q = 10) :=
sorry

end coin_stack_count_l1680_168012


namespace PQ_sum_l1680_168090

-- Define the problem conditions
variable (P Q x : ℝ)
variable (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)))

-- Define the proof goal
theorem PQ_sum (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3))) : P + Q = 52 :=
sorry

end PQ_sum_l1680_168090


namespace least_number_divisible_l1680_168048

theorem least_number_divisible (x : ℕ) (h1 : x = 857) 
  (h2 : (x + 7) % 24 = 0) 
  (h3 : (x + 7) % 36 = 0) 
  (h4 : (x + 7) % 54 = 0) :
  (x + 7) % 32 = 0 := 
sorry

end least_number_divisible_l1680_168048


namespace non_honda_red_percentage_l1680_168004

-- Define the conditions
def total_cars : ℕ := 900
def honda_percentage_red : ℝ := 0.90
def total_percentage_red : ℝ := 0.60
def honda_cars : ℕ := 500

-- The statement to prove
theorem non_honda_red_percentage : 
  (0.60 * 900 - 0.90 * 500) / (900 - 500) * 100 = 22.5 := 
  by sorry

end non_honda_red_percentage_l1680_168004


namespace find_a_if_parallel_l1680_168021

-- Definitions of the vectors and the scalar a
def vector_m : ℝ × ℝ := (2, 1)
def vector_n (a : ℝ) : ℝ × ℝ := (4, a)

-- Condition for parallel vectors
def are_parallel (m n : ℝ × ℝ) : Prop :=
  m.1 / n.1 = m.2 / n.2

-- Lean 4 statement
theorem find_a_if_parallel (a : ℝ) (h : are_parallel vector_m (vector_n a)) : a = 2 :=
by
  sorry

end find_a_if_parallel_l1680_168021


namespace pet_food_cost_is_correct_l1680_168068

-- Define the given conditions
def rabbit_toy_cost := 6.51
def cage_cost := 12.51
def total_cost := 24.81
def found_dollar := 1.00

-- Define the cost of pet food
def pet_food_cost := total_cost - (rabbit_toy_cost + cage_cost) + found_dollar

-- The statement to prove
theorem pet_food_cost_is_correct : pet_food_cost = 6.79 :=
by
  -- proof steps here
  sorry

end pet_food_cost_is_correct_l1680_168068


namespace quadrilateral_EFGH_inscribed_in_circle_l1680_168069

theorem quadrilateral_EFGH_inscribed_in_circle 
  (a b c : ℝ)
  (angle_EFG : ℝ := 60)
  (angle_EHG : ℝ := 50)
  (EH : ℝ := 5)
  (FG : ℝ := 7)
  (EG : ℝ := a)
  (EF : ℝ := b)
  (GH : ℝ := c)
  : EG = 7 * (Real.sin (70 * Real.pi / 180)) / (Real.sin (50 * Real.pi / 180)) :=
by
  sorry

end quadrilateral_EFGH_inscribed_in_circle_l1680_168069


namespace michael_boxes_l1680_168023

theorem michael_boxes (total_blocks boxes_per_box : ℕ) (h1: total_blocks = 16) (h2: boxes_per_box = 2) :
  total_blocks / boxes_per_box = 8 :=
by
  sorry

end michael_boxes_l1680_168023


namespace isosceles_triangle_perimeter_l1680_168039

theorem isosceles_triangle_perimeter (a b : ℕ)
  (h_eqn : ∀ x : ℕ, (x - 4) * (x - 2) = 0 → x = 4 ∨ x = 2)
  (h_isosceles : ∃ a b : ℕ, (a = 4 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 4)) :
  a + a + b = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l1680_168039


namespace graph_passes_through_fixed_point_l1680_168041

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (∀ x y : ℝ, y = a * x + 2 → (x, y) = (-1, 2))

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
sorry

end graph_passes_through_fixed_point_l1680_168041


namespace usual_time_to_catch_bus_l1680_168052

theorem usual_time_to_catch_bus (S T : ℝ) (h1 : S / ((5/4) * S) = (T + 5) / T) : T = 25 :=
by sorry

end usual_time_to_catch_bus_l1680_168052


namespace smallest_number_greater_300_with_remainder_24_l1680_168085

theorem smallest_number_greater_300_with_remainder_24 :
  ∃ n : ℕ, n > 300 ∧ n % 25 = 24 ∧ ∀ k : ℕ, k > 300 ∧ k % 25 = 24 → n ≤ k :=
sorry

end smallest_number_greater_300_with_remainder_24_l1680_168085


namespace find_a_squared_plus_b_squared_and_ab_l1680_168065

theorem find_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b) ^ 2 = 7)
  (h2 : (a - b) ^ 2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by 
  sorry

end find_a_squared_plus_b_squared_and_ab_l1680_168065


namespace correct_exponent_operation_l1680_168047

theorem correct_exponent_operation (a b : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  (6 * a^6 / (2 * a^2) ≠ 3 * a^3) ∧ 
  ((-a^2)^3 = -a^6) ∧ 
  ((-2 * a * b^2)^2 ≠ 2 * a^2 * b^4) :=
by
  sorry

end correct_exponent_operation_l1680_168047


namespace land_profit_each_son_l1680_168016

theorem land_profit_each_son :
  let hectares : ℝ := 3
  let m2_per_hectare : ℝ := 10000
  let total_sons : ℕ := 8
  let area_per_son := (hectares * m2_per_hectare) / total_sons
  let m2_per_portion : ℝ := 750
  let profit_per_portion : ℝ := 500
  let periods_per_year : ℕ := 12 / 3

  (area_per_son / m2_per_portion * profit_per_portion * periods_per_year = 10000) :=
by
  sorry

end land_profit_each_son_l1680_168016


namespace sum_ratio_is_nine_l1680_168072

open Nat

-- Predicate to define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 0 + a (n - 1))) / 2

axiom a : ℕ → ℝ -- The arithmetic sequence
axiom h_arith : is_arithmetic_sequence a
axiom a5_eq_5a3 : a 4 = 5 * a 2

-- Statement of the problem
theorem sum_ratio_is_nine : S 9 a / S 5 a = 9 :=
sorry

end sum_ratio_is_nine_l1680_168072


namespace goods_train_speed_l1680_168000

theorem goods_train_speed (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ) (speed_kmph : ℝ) :
  length_train = 280.04 →
  length_platform = 240 →
  time_seconds = 26 →
  speed_kmph = (length_train + length_platform) / time_seconds * 3.6 →
  speed_kmph = 72 :=
by
  intros h_train h_platform h_time h_speed
  rw [h_train, h_platform, h_time] at h_speed
  sorry

end goods_train_speed_l1680_168000


namespace find_parallel_line_l1680_168088

/-- 
Given a line l with equation 3x - 2y + 1 = 0 and a point A(1,1).
Find the equation of a line that passes through A and is parallel to l.
-/
theorem find_parallel_line (a b c : ℝ) (p_x p_y : ℝ) 
    (h₁ : 3 * p_x - 2 * p_y + c = 0) 
    (h₂ : p_x = 1 ∧ p_y = 1)
    (h₃ : a = 3 ∧ b = -2) :
    3 * x - 2 * y - 1 = 0 := 
by 
  sorry

end find_parallel_line_l1680_168088


namespace coleFenceCostCorrect_l1680_168019

noncomputable def coleFenceCost : ℕ := 455

def woodenFenceCost : ℕ := 15 * 6
def woodenFenceNeighborContribution : ℕ := woodenFenceCost / 3
def coleWoodenFenceCost : ℕ := woodenFenceCost - woodenFenceNeighborContribution

def metalFenceCost : ℕ := 15 * 8
def coleMetalFenceCost : ℕ := metalFenceCost

def hedgeCost : ℕ := 30 * 10
def hedgeNeighborContribution : ℕ := hedgeCost / 2
def coleHedgeCost : ℕ := hedgeCost - hedgeNeighborContribution

def installationFee : ℕ := 75
def soilPreparationFee : ℕ := 50

def totalCost : ℕ := coleWoodenFenceCost + coleMetalFenceCost + coleHedgeCost + installationFee + soilPreparationFee

theorem coleFenceCostCorrect : totalCost = coleFenceCost := by
  -- Skipping the proof steps with sorry
  sorry

end coleFenceCostCorrect_l1680_168019


namespace range_of_a_l1680_168049

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) → (-4 ≤ a ∧ a ≤ 4) :=
by 
  sorry

end range_of_a_l1680_168049


namespace probability_of_break_in_first_50_meters_l1680_168025

theorem probability_of_break_in_first_50_meters (total_length favorable_length : ℝ) 
  (h_total_length : total_length = 320) 
  (h_favorable_length : favorable_length = 50) : 
  (favorable_length / total_length) = 0.15625 := 
sorry

end probability_of_break_in_first_50_meters_l1680_168025


namespace percent_of_x_l1680_168076

variable {x y z : ℝ}

-- Define the given conditions
def cond1 (z y : ℝ) : Prop := 0.45 * z = 0.9 * y
def cond2 (z x : ℝ) : Prop := z = 1.5 * x

-- State the theorem to prove
theorem percent_of_x (h1 : cond1 z y) (h2 : cond2 z x) : y = 0.75 * x :=
sorry

end percent_of_x_l1680_168076


namespace segment_lengths_l1680_168097

noncomputable def radius : ℝ := 5
noncomputable def diameter : ℝ := 2 * radius
noncomputable def chord_length : ℝ := 8

-- The lengths of the segments AK and KB
theorem segment_lengths (x : ℝ) (y : ℝ) 
  (hx : 0 < x ∧ x < diameter) 
  (hy : 0 < y ∧ y < diameter) 
  (h1 : x + y = diameter) 
  (h2 : x * y = (diameter^2) / 4 - 16 / 4) : 
  x = 2.5 ∧ y = 7.5 := 
sorry

end segment_lengths_l1680_168097


namespace giraffe_ratio_l1680_168024

theorem giraffe_ratio (g ng : ℕ) (h1 : g = 300) (h2 : g = ng + 290) : g / ng = 30 :=
by
  sorry

end giraffe_ratio_l1680_168024


namespace horse_revolutions_l1680_168057

noncomputable def carousel_revolutions (r1 r2 d1 : ℝ) : ℝ :=
  (d1 * r1) / r2

theorem horse_revolutions :
  carousel_revolutions 30 10 40 = 120 :=
by
  sorry

end horse_revolutions_l1680_168057


namespace largest_int_starting_with_8_l1680_168008

theorem largest_int_starting_with_8 (n : ℕ) : 
  (n / 100 = 8) ∧ (n >= 800) ∧ (n < 900) ∧ ∀ (d : ℕ), (d ∣ n ∧ d ≠ 0 ∧ d ≠ 7) → d ∣ 864 → (n ≤ 864) :=
sorry

end largest_int_starting_with_8_l1680_168008


namespace no_solution_for_k_l1680_168062

theorem no_solution_for_k 
  (a1 a2 a3 a4 : ℝ) 
  (h_pos1 : 0 < a1) (h_pos2 : a1 < a2) 
  (h_pos3 : a2 < a3) (h_pos4 : a3 < a4) 
  (x1 x2 x3 x4 k : ℝ) 
  (h1 : x1 + x2 + x3 + x4 = 1) 
  (h2 : a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 = k) 
  (h3 : a1^2 * x1 + a2^2 * x2 + a3^2 * x3 + a4^2 * x4 = k^2) 
  (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (hx3 : 0 ≤ x3) (hx4 : 0 ≤ x4) :
  false := 
sorry

end no_solution_for_k_l1680_168062


namespace work_completion_days_l1680_168011

theorem work_completion_days (x : ℕ) (h_ratio : 5 * 18 = 3 * 30) : 30 = 30 :=
by {
    sorry
}

end work_completion_days_l1680_168011


namespace eq_3_solutions_l1680_168067

theorem eq_3_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∃! (x y : ℕ), (0 < x) ∧ (0 < y) ∧ ((1 / x) + (1 / y) = (1 / p)) ∧
  ((x = p + 1 ∧ y = p^2 + p) ∨ (x = p + p ∧ y = p + p) ∨ (x = p^2 + p ∧ y = p + 1)) :=
sorry

end eq_3_solutions_l1680_168067


namespace no_such_increasing_seq_exists_l1680_168053

theorem no_such_increasing_seq_exists :
  ¬(∃ (a : ℕ → ℕ), (∀ m n : ℕ, a (m * n) = a m + a n) ∧ (∀ n : ℕ, a n < a (n + 1))) :=
by
  sorry

end no_such_increasing_seq_exists_l1680_168053


namespace part1_part2_l1680_168055

def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x - 3)
noncomputable def M := 3 / 2

theorem part1 (x : ℝ) (m : ℝ) : (∀ x, f x ≥ abs (m + 1)) → m ≤ M := sorry

theorem part2 (a b c : ℝ) : a > 0 → b > 0 → c > 0 → a + b + c = M →  (b^2 / a + c^2 / b + a^2 / c) ≥ M := sorry

end part1_part2_l1680_168055


namespace tire_price_l1680_168094

theorem tire_price (x : ℝ) (h : 3 * x + 10 = 310) : x = 100 :=
sorry

end tire_price_l1680_168094


namespace club_last_names_l1680_168087

theorem club_last_names :
  ∃ A B C D E F : ℕ,
    A + B + C + D + E + F = 21 ∧
    A^2 + B^2 + C^2 + D^2 + E^2 + F^2 = 91 :=
by {
  sorry
}

end club_last_names_l1680_168087


namespace arithmetic_sequence_sum_l1680_168043

open Nat

theorem arithmetic_sequence_sum (m n : Nat) (d : ℤ) (a_1 : ℤ)
    (hnm : n ≠ m)
    (hSn : (n * (2 * a_1 + (n - 1) * d) / 2) = n / m)
    (hSm : (m * (2 * a_1 + (m - 1) * d) / 2) = m / n) :
  ((m + n) * (2 * a_1 + (m + n - 1) * d) / 2) > 4 := by
  sorry

end arithmetic_sequence_sum_l1680_168043


namespace find_A_salary_l1680_168045

theorem find_A_salary (A B : ℝ) (h1 : A + B = 2000) (h2 : 0.05 * A = 0.15 * B) : A = 1500 :=
sorry

end find_A_salary_l1680_168045


namespace total_floor_area_is_correct_l1680_168010

-- Define the combined area of the three rugs
def combined_area_of_rugs : ℕ := 212

-- Define the area covered by exactly two layers of rug
def area_covered_by_two_layers : ℕ := 24

-- Define the area covered by exactly three layers of rug
def area_covered_by_three_layers : ℕ := 24

-- Define the total floor area covered by the rugs
def total_floor_area_covered : ℕ :=
  combined_area_of_rugs - area_covered_by_two_layers - 2 * area_covered_by_three_layers

-- The theorem stating the total floor area covered
theorem total_floor_area_is_correct : total_floor_area_covered = 140 := by
  sorry

end total_floor_area_is_correct_l1680_168010


namespace numBills_is_9_l1680_168079

-- Define the conditions: Mike has 45 dollars in 5-dollar bills
def totalDollars : ℕ := 45
def billValue : ℕ := 5
def numBills : ℕ := 9

-- Prove that the number of 5-dollar bills Mike has is 9
theorem numBills_is_9 : (totalDollars = billValue * numBills) → (numBills = 9) :=
by
  intro h
  sorry

end numBills_is_9_l1680_168079


namespace czechoslovak_inequality_l1680_168035

-- Define the triangle and the points
structure Triangle (α : Type) [LinearOrderedRing α] :=
(A B C : α × α)

variables {α : Type} [LinearOrderedRing α]

-- Define the condition that O is on the segment AB but is not a vertex
def on_segment (O A B : α × α) : Prop :=
  ∃ x : α, 0 < x ∧ x < 1 ∧ O = (A.1 + x * (B.1 - A.1), A.2 + x * (B.2 - A.2))

-- Define the dot product for vectors
def dot (u v: α × α) : α := u.1 * v.1 + u.2 * v.2

-- Main statement
theorem czechoslovak_inequality (T : Triangle α) (O : α × α) (hO : on_segment O T.A T.B) :
  dot O T.C * dot T.A T.B < dot T.A O * dot T.B T.C + dot T.B O * dot T.A T.C :=
sorry

end czechoslovak_inequality_l1680_168035


namespace FI_squared_correct_l1680_168005

noncomputable def FI_squared : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (4, 4)
  let D : ℝ × ℝ := (0, 4)
  let E : ℝ × ℝ := (3, 0)
  let H : ℝ × ℝ := (0, 1)
  let F : ℝ × ℝ := (4, 1)
  let G : ℝ × ℝ := (1, 4)
  let I : ℝ × ℝ := (3, 0)
  let J : ℝ × ℝ := (0, 1)
  let FI_squared := (4 - 3)^2 + (1 - 0)^2
  FI_squared

theorem FI_squared_correct : FI_squared = 2 :=
by
  sorry

end FI_squared_correct_l1680_168005


namespace benny_number_of_kids_l1680_168077

-- Define the conditions
def benny_has_dollars (d: ℕ): Prop := d = 360
def cost_per_apple (c: ℕ): Prop := c = 4
def apples_shared (num_kids num_apples: ℕ): Prop := num_apples = 5 * num_kids

-- State the main theorem
theorem benny_number_of_kids : 
  ∀ (d c k a : ℕ), benny_has_dollars d → cost_per_apple c → apples_shared k a → k = 18 :=
by
  intros d c k a hd hc ha
  -- The goal is to prove k = 18; use the provided conditions
  sorry

end benny_number_of_kids_l1680_168077


namespace problem_xy_minimized_problem_x_y_minimized_l1680_168050

open Real

theorem problem_xy_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 16 ∧ y = 2 ∧ x * y = 32 := 
sorry

theorem problem_x_y_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 8 + 2 * sqrt 2 ∧ y = 1 + sqrt 2 ∧ x + y = 9 + 4 * sqrt 2 := 
sorry

end problem_xy_minimized_problem_x_y_minimized_l1680_168050


namespace recurring_decimal_sum_as_fraction_l1680_168034

theorem recurring_decimal_sum_as_fraction :
  (0.2 + 0.03 + 0.0004) = 281 / 1111 := by
  sorry

end recurring_decimal_sum_as_fraction_l1680_168034


namespace min_value_of_expression_l1680_168098

noncomputable def min_expression := 4 * (Real.rpow 5 (1/4) - 1)^2

theorem min_value_of_expression (a b c : ℝ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = min_expression :=
sorry

end min_value_of_expression_l1680_168098


namespace lindsey_saved_in_november_l1680_168028

def savings_sept : ℕ := 50
def savings_oct : ℕ := 37
def additional_money : ℕ := 25
def spent_on_video_game : ℕ := 87
def money_left : ℕ := 36

def total_savings_before_november := savings_sept + savings_oct
def total_savings_after_november (N : ℕ) := total_savings_before_november + N + additional_money

theorem lindsey_saved_in_november : ∃ N : ℕ, total_savings_after_november N - spent_on_video_game = money_left ∧ N = 11 :=
by
  sorry

end lindsey_saved_in_november_l1680_168028


namespace find_n_l1680_168033

theorem find_n (n : ℕ) (hn : n * n! - n! = 5040 - n!) : n = 7 :=
by
  sorry

end find_n_l1680_168033


namespace length_of_plot_57_meters_l1680_168081

section RectangleProblem

variable (b : ℝ) -- breadth of the plot
variable (l : ℝ) -- length of the plot
variable (cost_per_meter : ℝ) -- cost per meter
variable (total_cost : ℝ) -- total cost

-- Given conditions
def length_eq_breadth_plus_14 (b l : ℝ) : Prop := l = b + 14
def cost_eq_perimeter_cost_per_meter (cost_per_meter total_cost perimeter : ℝ) : Prop :=
  total_cost = cost_per_meter * perimeter

-- Definition of perimeter
def perimeter (b l : ℝ) : ℝ := 2 * l + 2 * b

-- Problem statement
theorem length_of_plot_57_meters
  (h1 : length_eq_breadth_plus_14 b l)
  (h2 : cost_eq_perimeter_cost_per_meter cost_per_meter total_cost (perimeter b l))
  (h3 : cost_per_meter = 26.50)
  (h4 : total_cost = 5300) :
  l = 57 :=
by
  sorry

end RectangleProblem

end length_of_plot_57_meters_l1680_168081


namespace train_speed_l1680_168042

-- Definition for the given conditions
def distance : ℕ := 240 -- distance in meters
def time_seconds : ℕ := 6 -- time in seconds
def conversion_factor : ℕ := 3600 -- seconds to hour conversion factor
def meters_in_km : ℕ := 1000 -- meters to kilometers conversion factor

-- The proof goal
theorem train_speed (d : ℕ) (t : ℕ) (cf : ℕ) (mk : ℕ) (h1 : d = distance) (h2 : t = time_seconds) (h3 : cf = conversion_factor) (h4 : mk = meters_in_km) :
  (d * cf / t) / mk = 144 :=
by sorry

end train_speed_l1680_168042


namespace abs_value_solutions_l1680_168078

theorem abs_value_solutions (x : ℝ) : abs x = 6.5 ↔ x = 6.5 ∨ x = -6.5 :=
by
  sorry

end abs_value_solutions_l1680_168078


namespace morgan_total_pens_l1680_168001

def initial_red_pens : Nat := 65
def initial_blue_pens : Nat := 45
def initial_black_pens : Nat := 58
def initial_green_pens : Nat := 36
def initial_purple_pens : Nat := 27

def red_pens_given_away : Nat := 15
def blue_pens_given_away : Nat := 20
def green_pens_given_away : Nat := 10

def black_pens_bought : Nat := 12
def purple_pens_bought : Nat := 5

def final_red_pens : Nat := initial_red_pens - red_pens_given_away
def final_blue_pens : Nat := initial_blue_pens - blue_pens_given_away
def final_black_pens : Nat := initial_black_pens + black_pens_bought
def final_green_pens : Nat := initial_green_pens - green_pens_given_away
def final_purple_pens : Nat := initial_purple_pens + purple_pens_bought

def total_pens : Nat := final_red_pens + final_blue_pens + final_black_pens + final_green_pens + final_purple_pens

theorem morgan_total_pens : total_pens = 203 := 
by
  -- final_red_pens = 50
  -- final_blue_pens = 25
  -- final_black_pens = 70
  -- final_green_pens = 26
  -- final_purple_pens = 32
  -- Therefore, total_pens = 203
  sorry

end morgan_total_pens_l1680_168001


namespace parallelogram_area_twice_quadrilateral_area_l1680_168095

theorem parallelogram_area_twice_quadrilateral_area (S : ℝ) (LMNP_area : ℝ) 
  (h : LMNP_area = 2 * S) : LMNP_area = 2 * S := 
by {
  sorry
}

end parallelogram_area_twice_quadrilateral_area_l1680_168095


namespace simplify_expansion_l1680_168020

theorem simplify_expansion (x : ℝ) : 
  (3 * x - 6) * (x + 8) - (x + 6) * (3 * x + 2) = -2 * x - 60 :=
by
  sorry

end simplify_expansion_l1680_168020


namespace minimum_value_expression_l1680_168082

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, y = (1 / a^2 - 1) * (1 / b^2 - 1) → x ≤ y) :=
sorry

end minimum_value_expression_l1680_168082


namespace a11_is_1_l1680_168073

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Condition 1: The sum of the first n terms S_n satisfies S_n + S_m = S_{n+m}
axiom sum_condition (n m : ℕ) : S n + S m = S (n + m)

-- Condition 2: a_1 = 1
axiom a1_condition : a 1 = 1

-- Question: prove a_{11} = 1
theorem a11_is_1 : a 11 = 1 :=
sorry


end a11_is_1_l1680_168073


namespace average_percentage_decrease_l1680_168022

theorem average_percentage_decrease :
  ∃ (x : ℝ), (5000 * (1 - x / 100)^3 = 2560) ∧ x = 20 :=
by
  sorry

end average_percentage_decrease_l1680_168022


namespace bowling_ball_weight_l1680_168002

theorem bowling_ball_weight (b c : ℕ) (h1 : 8 * b = 4 * c) (h2 : 3 * c = 108) : b = 18 := 
by 
  sorry

end bowling_ball_weight_l1680_168002


namespace number_of_y_axis_returns_l1680_168038

-- Definitions based on conditions
noncomputable def unit_length : ℝ := 0.5
noncomputable def diagonal_length : ℝ := Real.sqrt 2 * unit_length
noncomputable def pen_length_cm : ℝ := 8000 * 100 -- converting meters to cm
noncomputable def circle_length (n : ℕ) : ℝ := ((3 + Real.sqrt 2) * n ^ 2 + 2 * n) * unit_length

-- The main theorem
theorem number_of_y_axis_returns : ∃ n : ℕ, circle_length n ≤ pen_length_cm ∧ circle_length (n+1) > pen_length_cm :=
sorry

end number_of_y_axis_returns_l1680_168038


namespace man_walking_time_l1680_168017

theorem man_walking_time
  (T : ℕ) -- Let T be the time (in minutes) the man usually arrives at the station.
  (usual_arrival_home : ℕ) -- The time (in minutes) they usually arrive home, which is T + 30.
  (early_arrival : ℕ) (walking_start_time : ℕ) (early_home_arrival : ℕ)
  (usual_arrival_home_eq : usual_arrival_home = T + 30)
  (early_arrival_eq : early_arrival = T - 60)
  (walking_start_time_eq : walking_start_time = early_arrival)
  (early_home_arrival_eq : early_home_arrival = T)
  (time_saved : ℕ) (half_time_walk : ℕ)
  (time_saved_eq : time_saved = 30)
  (half_time_walk_eq : half_time_walk = time_saved / 2) :
  walking_start_time = half_time_walk := by
  sorry

end man_walking_time_l1680_168017


namespace ashok_borrowed_l1680_168051

theorem ashok_borrowed (P : ℝ) (h : 11400 = P * (6 / 100 * 2 + 9 / 100 * 3 + 14 / 100 * 4)) : P = 12000 :=
by
  sorry

end ashok_borrowed_l1680_168051


namespace sum_of_discount_rates_l1680_168060

theorem sum_of_discount_rates : 
  let fox_price := 15
  let pony_price := 20
  let fox_pairs := 3
  let pony_pairs := 2
  let total_savings := 9
  let pony_discount := 18.000000000000014
  let fox_discount := 4
  let total_discount_rate := fox_discount + pony_discount
  total_discount_rate = 22.000000000000014 := by
sorry

end sum_of_discount_rates_l1680_168060


namespace inequality_solution_l1680_168084

theorem inequality_solution (z : ℝ) : 
  z^2 - 40 * z + 400 ≤ 36 ↔ 14 ≤ z ∧ z ≤ 26 :=
by
  sorry

end inequality_solution_l1680_168084


namespace solve_for_r_l1680_168059

theorem solve_for_r (r : ℝ) (h: (r + 9) / (r - 3) = (r - 2) / (r + 5)) : r = -39 / 19 :=
sorry

end solve_for_r_l1680_168059


namespace johny_distance_l1680_168015

noncomputable def distance_south : ℕ := 40
variable (E : ℕ)
noncomputable def distance_east : ℕ := E
noncomputable def distance_north (E : ℕ) : ℕ := 2 * E
noncomputable def total_distance (E : ℕ) : ℕ := distance_south + distance_east E + distance_north E

theorem johny_distance :
  ∀ E : ℕ, total_distance E = 220 → E - distance_south = 20 :=
by
  intro E
  intro h
  rw [total_distance, distance_north, distance_east, distance_south] at h
  sorry

end johny_distance_l1680_168015


namespace find_numbers_l1680_168054

theorem find_numbers (x y z : ℝ) 
  (h1 : x = 280)
  (h2 : y = 200)
  (h3 : z = 220) :
  (x = 1.4 * y) ∧
  (x / z = 14 / 11) ∧
  (z - y = 0.125 * (x + y) - 40) :=
by
  sorry

end find_numbers_l1680_168054


namespace pasha_encoded_expression_l1680_168036

theorem pasha_encoded_expression :
  2065 + 5 - 47 = 2023 :=
by
  sorry

end pasha_encoded_expression_l1680_168036


namespace pizza_slice_volume_l1680_168063

-- Define the parameters given in the conditions
def pizza_thickness : ℝ := 0.5
def pizza_diameter : ℝ := 16.0
def num_slices : ℝ := 16.0

-- Define the volume of one slice
theorem pizza_slice_volume : (π * (pizza_diameter / 2) ^ 2 * pizza_thickness / num_slices) = 2 * π := by
  sorry

end pizza_slice_volume_l1680_168063


namespace annual_decrease_rate_l1680_168064

theorem annual_decrease_rate (P : ℕ) (P2 : ℕ) (r : ℝ) : 
  (P = 10000) → (P2 = 8100) → (P2 = P * (1 - r / 100)^2) → (r = 10) :=
by
  intro hP hP2 hEq
  sorry

end annual_decrease_rate_l1680_168064


namespace sandy_gain_percent_is_10_l1680_168046

def total_cost (purchase_price repair_costs : ℕ) := purchase_price + repair_costs

def gain (selling_price total_cost : ℕ) := selling_price - total_cost

def gain_percent (gain total_cost : ℕ) := (gain / total_cost : ℚ) * 100

theorem sandy_gain_percent_is_10 
  (purchase_price : ℕ := 900)
  (repair_costs : ℕ := 300)
  (selling_price : ℕ := 1320) :
  gain_percent (gain selling_price (total_cost purchase_price repair_costs)) 
               (total_cost purchase_price repair_costs) = 10 := 
by
  simp [total_cost, gain, gain_percent]
  sorry

end sandy_gain_percent_is_10_l1680_168046


namespace trapezoid_is_proposition_l1680_168056

-- Define what it means to be a proposition
def is_proposition (s : String) : Prop := ∃ b : Bool, (s = "A trapezoid is a quadrilateral" ∨ s = "Construct line AB" ∨ s = "x is an integer" ∨ s = "Will it snow today?") ∧ 
  (b → s = "A trapezoid is a quadrilateral") 

-- Main proof statement
theorem trapezoid_is_proposition : is_proposition "A trapezoid is a quadrilateral" :=
  sorry

end trapezoid_is_proposition_l1680_168056


namespace velocity_of_current_l1680_168026

theorem velocity_of_current
  (v c : ℝ) 
  (h1 : 32 = (v + c) * 6) 
  (h2 : 14 = (v - c) * 6) :
  c = 1.5 :=
by
  sorry

end velocity_of_current_l1680_168026


namespace unique_digit_sum_l1680_168086

theorem unique_digit_sum (X Y M Z F : ℕ) (H1 : X ≠ 0) (H2 : Y ≠ 0) (H3 : M ≠ 0) (H4 : Z ≠ 0) (H5 : F ≠ 0)
  (H6 : X ≠ Y) (H7 : X ≠ M) (H8 : X ≠ Z) (H9 : X ≠ F)
  (H10 : Y ≠ M) (H11 : Y ≠ Z) (H12 : Y ≠ F)
  (H13 : M ≠ Z) (H14 : M ≠ F)
  (H15 : Z ≠ F)
  (H16 : 10 * X + Y ≠ 0) (H17 : 10 * M + Z ≠ 0)
  (H18 : 111 * F = (10 * X + Y) * (10 * M + Z)) :
  X + Y + M + Z + F = 28 := by
  sorry

end unique_digit_sum_l1680_168086


namespace find_rate_of_current_l1680_168031

-- Given speed of the boat in still water (km/hr)
def boat_speed : ℤ := 20

-- Given time of travel downstream (hours)
def time_downstream : ℚ := 24 / 60

-- Given distance travelled downstream (km)
def distance_downstream : ℤ := 10

-- To find: rate of the current (km/hr)
theorem find_rate_of_current (c : ℚ) 
  (h1 : distance_downstream = (boat_speed + c) * time_downstream) : 
  c = 5 := 
by sorry

end find_rate_of_current_l1680_168031


namespace rectangle_area_l1680_168071

theorem rectangle_area (x : ℕ) (hx : x > 0)
  (h₁ : (x + 5) * 2 * (x + 10) = 3 * x * (x + 10))
  (h₂ : (x - 10) = x + 10 - 10) :
  x * (x + 10) = 200 :=
by {
  sorry
}

end rectangle_area_l1680_168071


namespace smaller_cuboid_width_l1680_168061

theorem smaller_cuboid_width
  (length_orig width_orig height_orig : ℕ)
  (length_small height_small : ℕ)
  (num_small_cuboids : ℕ)
  (volume_orig : ℕ := length_orig * width_orig * height_orig)
  (volume_small : ℕ := length_small * width_small * height_small)
  (H1 : length_orig = 18)
  (H2 : width_orig = 15)
  (H3 : height_orig = 2)
  (H4 : length_small = 5)
  (H5 : height_small = 3)
  (H6 : num_small_cuboids = 6)
  (H_volume_match : num_small_cuboids * volume_small = volume_orig)
  : width_small = 6 := by
  sorry

end smaller_cuboid_width_l1680_168061


namespace arithmetic_sequence_term_20_l1680_168009

theorem arithmetic_sequence_term_20
  (a : ℕ := 2)
  (d : ℕ := 4)
  (n : ℕ := 20) :
  a + (n - 1) * d = 78 :=
by
  sorry

end arithmetic_sequence_term_20_l1680_168009


namespace sum_gcd_lcm_l1680_168032

theorem sum_gcd_lcm (a b : ℕ) (ha : a = 45) (hb : b = 4095) :
    Nat.gcd a b + Nat.lcm a b = 4140 :=
by
  sorry

end sum_gcd_lcm_l1680_168032


namespace sandys_average_price_l1680_168080

noncomputable def average_price_per_book (priceA : ℝ) (discountA : ℝ) (booksA : ℕ) (priceB : ℝ) (discountB : ℝ) (booksB : ℕ) (conversion_rate : ℝ) : ℝ :=
  let costA := priceA / (1 - discountA)
  let priceB_in_usd := priceB / conversion_rate
  let costB := priceB_in_usd / (1 - discountB)
  let total_cost := costA + costB
  let total_books := booksA + booksB
  total_cost / total_books

theorem sandys_average_price :
  average_price_per_book 1380 0.15 65 900 0.10 55 0.85 = 23.33 :=
by
  sorry

end sandys_average_price_l1680_168080


namespace pizza_topping_slices_l1680_168070

theorem pizza_topping_slices 
  (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ)
  (pepperoni_slices_has_at_least_one_topping : pepperoni_slices = 8)
  (mushroom_slices_has_at_least_one_topping : mushroom_slices = 12)
  (olive_slices_has_at_least_one_topping : olive_slices = 14)
  (total_slices_has_one_topping : total_slices = 16)
  (slices_with_at_least_one_topping : 8 + 12 + 14 - 2 * x = 16) :
  x = 9 :=
by
  sorry

end pizza_topping_slices_l1680_168070
