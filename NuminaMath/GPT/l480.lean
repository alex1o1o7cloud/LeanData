import Mathlib

namespace NUMINAMATH_GPT_triangle_angle_C_right_l480_48034

theorem triangle_angle_C_right {a b c A B C : ℝ}
  (h1 : a / Real.sin B + b / Real.sin A = 2 * c) 
  (h2 : a / Real.sin A = b / Real.sin B) 
  (h3 : b / Real.sin B = c / Real.sin C) : 
  C = Real.pi / 2 :=
by sorry

end NUMINAMATH_GPT_triangle_angle_C_right_l480_48034


namespace NUMINAMATH_GPT_joe_cars_after_getting_more_l480_48016

-- Defining the initial conditions as Lean variables
def initial_cars : ℕ := 50
def additional_cars : ℕ := 12

-- Stating the proof problem
theorem joe_cars_after_getting_more : initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_GPT_joe_cars_after_getting_more_l480_48016


namespace NUMINAMATH_GPT_calculate_cost_price_l480_48053

/-
Given:
  SP (Selling Price) is 18000
  If a 10% discount is applied on the SP, the effective selling price becomes 16200
  This effective selling price corresponds to an 8% profit over the cost price
  
Prove:
  The cost price (CP) is 15000
-/

theorem calculate_cost_price (SP : ℝ) (d : ℝ) (p : ℝ) (effective_SP : ℝ) (CP : ℝ) :
  SP = 18000 →
  d = 0.1 →
  p = 0.08 →
  effective_SP = SP - (d * SP) →
  effective_SP = CP * (1 + p) →
  CP = 15000 :=
by
  intros _
  sorry

end NUMINAMATH_GPT_calculate_cost_price_l480_48053


namespace NUMINAMATH_GPT_tickets_used_l480_48093

variable (C T : Nat)

theorem tickets_used (h1 : C = 7) (h2 : T = C + 5) : T = 12 := by
  sorry

end NUMINAMATH_GPT_tickets_used_l480_48093


namespace NUMINAMATH_GPT_nearest_integer_power_l480_48052

noncomputable def power_expression := (3 + Real.sqrt 2)^6

theorem nearest_integer_power :
  Int.floor power_expression = 7414 :=
sorry

end NUMINAMATH_GPT_nearest_integer_power_l480_48052


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l480_48072

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ → ℕ)
  (is_arithmetic_seq : ∀ n, a (n + 1) = a n + d n)
  (h : (a 2) + (a 5) + (a 8) = 39) :
  (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) + (a 8) + (a 9) = 117 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l480_48072


namespace NUMINAMATH_GPT_rancher_cows_l480_48010

theorem rancher_cows : ∃ (C H : ℕ), (C = 5 * H) ∧ (C + H = 168) ∧ (C = 140) := by
  sorry

end NUMINAMATH_GPT_rancher_cows_l480_48010


namespace NUMINAMATH_GPT_problem_k_value_l480_48022

theorem problem_k_value (k x1 x2 : ℝ) 
  (h_eq : 8 * x1^2 + 2 * k * x1 + k - 1 = 0) 
  (h_eq2 : 8 * x2^2 + 2 * k * x2 + k - 1 = 0) 
  (h_sum_sq : x1^2 + x2^2 = 1) : 
  k = -2 :=
sorry

end NUMINAMATH_GPT_problem_k_value_l480_48022


namespace NUMINAMATH_GPT_cut_problem_l480_48046

theorem cut_problem (n : ℕ) : (1 / 2 : ℝ) ^ n = 1 / 64 ↔ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_cut_problem_l480_48046


namespace NUMINAMATH_GPT_total_percentage_of_samplers_l480_48023

theorem total_percentage_of_samplers :
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  (pA + pA_not_caught + pB + pB_not_caught + pC + pC_not_caught + pD + pD_not_caught) = 54 :=
by
  let pA := 12
  let pB := 5
  let pC := 9
  let pD := 4
  let pA_not_caught := 7
  let pB_not_caught := 6
  let pC_not_caught := 3
  let pD_not_caught := 8
  sorry

end NUMINAMATH_GPT_total_percentage_of_samplers_l480_48023


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l480_48037

theorem breadth_of_rectangular_plot (b : ℝ) (h1 : 3 * b * b = 972) : b = 18 :=
sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l480_48037


namespace NUMINAMATH_GPT_factorize_expr_l480_48050

theorem factorize_expr (a b : ℝ) : a * b^2 - 8 * a * b + 16 * a = a * (b - 4)^2 := 
by
  sorry

end NUMINAMATH_GPT_factorize_expr_l480_48050


namespace NUMINAMATH_GPT_find_hypotenuse_of_right_angle_triangle_l480_48060

theorem find_hypotenuse_of_right_angle_triangle
  (PR : ℝ) (angle_QPR : ℝ)
  (h1 : PR = 16)
  (h2 : angle_QPR = Real.pi / 4) :
  ∃ PQ : ℝ, PQ = 16 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_hypotenuse_of_right_angle_triangle_l480_48060


namespace NUMINAMATH_GPT_range_of_a_l480_48063

noncomputable def f (a x : ℝ) : ℝ := (2 - a^2) * x + a

theorem range_of_a (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f a x > 0) ↔ (0 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l480_48063


namespace NUMINAMATH_GPT_number_of_intersection_points_l480_48035

-- Definitions of the given lines
def line1 (x y : ℝ) : Prop := 6 * y - 4 * x = 2
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := -4 * x + 6 * y = 3

-- Definitions of the intersection points
def intersection1 (x y : ℝ) : Prop := line1 x y ∧ line2 x y
def intersection2 (x y : ℝ) : Prop := line2 x y ∧ line3 x y

-- Definition of the problem
theorem number_of_intersection_points : 
  (∃ x y : ℝ, intersection1 x y) ∧
  (∃ x y : ℝ, intersection2 x y) ∧
  (¬ ∃ x y : ℝ, line1 x y ∧ line3 x y) →
  (∃ z : ℕ, z = 2) :=
sorry

end NUMINAMATH_GPT_number_of_intersection_points_l480_48035


namespace NUMINAMATH_GPT_inequality_addition_l480_48006

-- Definitions and Conditions
variables (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c > d)

-- Theorem statement: Prove that a + c > b + d
theorem inequality_addition (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := 
sorry

end NUMINAMATH_GPT_inequality_addition_l480_48006


namespace NUMINAMATH_GPT_Telegraph_Road_length_is_162_l480_48068

-- Definitions based on the conditions
def meters_to_kilometers (meters : ℕ) : ℕ := meters / 1000
def Pardee_Road_length_meters : ℕ := 12000
def Telegraph_Road_extra_length_kilometers : ℕ := 150

-- The length of Pardee Road in kilometers
def Pardee_Road_length_kilometers : ℕ := meters_to_kilometers Pardee_Road_length_meters

-- Lean statement to prove the length of Telegraph Road in kilometers
theorem Telegraph_Road_length_is_162 :
  Pardee_Road_length_kilometers + Telegraph_Road_extra_length_kilometers = 162 :=
sorry

end NUMINAMATH_GPT_Telegraph_Road_length_is_162_l480_48068


namespace NUMINAMATH_GPT_eggs_in_larger_omelette_l480_48044

theorem eggs_in_larger_omelette :
  ∀ (total_eggs : ℕ) (orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette : ℕ),
    total_eggs = 84 →
    orders_3_eggs_first_hour = 5 →
    orders_3_eggs_third_hour = 3 →
    orders_large_eggs_second_hour = 7 →
    orders_large_eggs_last_hour = 8 →
    num_eggs_per_3_omelette = 3 →
    (total_eggs - (orders_3_eggs_first_hour * num_eggs_per_3_omelette + orders_3_eggs_third_hour * num_eggs_per_3_omelette)) / (orders_large_eggs_second_hour + orders_large_eggs_last_hour) = 4 :=
by
  intros total_eggs orders_3_eggs_first_hour orders_3_eggs_third_hour orders_large_eggs_second_hour orders_large_eggs_last_hour num_eggs_per_3_omelette
  sorry

end NUMINAMATH_GPT_eggs_in_larger_omelette_l480_48044


namespace NUMINAMATH_GPT_sum_of_squares_l480_48036

theorem sum_of_squares : 
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  (squares.sum = 195) := 
by
  let nums := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
  let squares := nums.map (λ x => x * x)
  have h : squares.sum = 195 := sorry
  exact h

end NUMINAMATH_GPT_sum_of_squares_l480_48036


namespace NUMINAMATH_GPT_total_seats_round_table_l480_48077

theorem total_seats_round_table 
  (a : ℕ) (b : ℕ) 
  (h₀ : a ≠ b)
  (h₁ : a + b = 39) 
  : ∃ n, n = 38 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_seats_round_table_l480_48077


namespace NUMINAMATH_GPT_number_of_kids_at_circus_l480_48041

theorem number_of_kids_at_circus (K A : ℕ) 
(h1 : ∀ x, 5 * x = 1 / 2 * 10 * x)
(h2 : 5 * K + 10 * A = 50) : K = 2 :=
sorry

end NUMINAMATH_GPT_number_of_kids_at_circus_l480_48041


namespace NUMINAMATH_GPT_calculate_x_l480_48054

theorem calculate_x :
  (422 + 404) ^ 2 - (4 * 422 * 404) = 324 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_calculate_x_l480_48054


namespace NUMINAMATH_GPT_exam_scores_l480_48069

theorem exam_scores (A B C D : ℤ) 
  (h1 : A + B = C + D + 17) 
  (h2 : A = B - 4) 
  (h3 : C = D + 5) :
  ∃ highest lowest, (highest - lowest = 13) ∧ 
                   (highest = A ∨ highest = B ∨ highest = C ∨ highest = D) ∧ 
                   (lowest = A ∨ lowest = B ∨ lowest = C ∨ lowest = D) :=
by
  sorry

end NUMINAMATH_GPT_exam_scores_l480_48069


namespace NUMINAMATH_GPT_total_cards_traded_l480_48045

-- Define the total number of cards traded in both trades
def total_traded (p1_t: ℕ) (r1_t: ℕ) (p2_t: ℕ) (r2_t: ℕ): ℕ :=
  (p1_t + r1_t) + (p2_t + r2_t)

-- Given conditions as definitions
def padma_trade1 := 2   -- Cards Padma traded in the first trade
def robert_trade1 := 10  -- Cards Robert traded in the first trade
def padma_trade2 := 15  -- Cards Padma traded in the second trade
def robert_trade2 := 8   -- Cards Robert traded in the second trade

-- Theorem stating the total number of cards traded is 35
theorem total_cards_traded : 
  total_traded padma_trade1 robert_trade1 padma_trade2 robert_trade2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_cards_traded_l480_48045


namespace NUMINAMATH_GPT_find_range_of_function_l480_48026

variable (a : ℝ) (x : ℝ)

def func (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem find_range_of_function (a : ℝ) :
  if a < 0 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -1 ≤ y ∧ y ≤ 3 - 4*a
  else if 0 ≤ a ∧ a ≤ 1 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ 3 - 4*a
  else if 1 < a ∧ a ≤ 2 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ -1
  else
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ 3 - 4*a ≤ y ∧ y ≤ -1
:= sorry

end NUMINAMATH_GPT_find_range_of_function_l480_48026


namespace NUMINAMATH_GPT_fraction_of_paint_used_l480_48007

theorem fraction_of_paint_used 
  (total_paint : ℕ)
  (paint_used_first_week : ℚ)
  (total_paint_used : ℕ)
  (paint_fraction_first_week : ℚ)
  (remaining_paint : ℚ)
  (paint_used_second_week : ℚ)
  (paint_fraction_second_week : ℚ)
  (h1 : total_paint = 360)
  (h2 : paint_fraction_first_week = 2/3)
  (h3 : paint_used_first_week = paint_fraction_first_week * total_paint)
  (h4 : remaining_paint = total_paint - paint_used_first_week)
  (h5 : remaining_paint = 120)
  (h6 : total_paint_used = 264)
  (h7 : paint_used_second_week = total_paint_used - paint_used_first_week)
  (h8 : paint_fraction_second_week = paint_used_second_week / remaining_paint):
  paint_fraction_second_week = 1/5 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_of_paint_used_l480_48007


namespace NUMINAMATH_GPT_correct_quadratic_eq_l480_48082

-- Define the given conditions
def first_student_sum (b : ℝ) : Prop := 5 + 3 = -b
def second_student_product (c : ℝ) : Prop := (-12) * (-4) = c

-- Define the proof statement
theorem correct_quadratic_eq (b c : ℝ) (h1 : first_student_sum b) (h2 : second_student_product c) :
    b = -8 ∧ c = 48 ∧ (∀ x : ℝ, x^2 + b * x + c = 0 → (x=5 ∨ x=3 ∨ x=-12 ∨ x=-4)) :=
by
  sorry

end NUMINAMATH_GPT_correct_quadratic_eq_l480_48082


namespace NUMINAMATH_GPT_simplify_expression_l480_48083

theorem simplify_expression (x y : ℝ) (h1 : x = 1) (h2 : y = 2) : 
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l480_48083


namespace NUMINAMATH_GPT_tire_price_l480_48092

theorem tire_price (payment : ℕ) (price_ratio : ℕ → ℕ → Prop)
  (h1 : payment = 345)
  (h2 : price_ratio 3 1)
  : ∃ x : ℕ, x = 99 := 
sorry

end NUMINAMATH_GPT_tire_price_l480_48092


namespace NUMINAMATH_GPT_solve_inequality_l480_48064

theorem solve_inequality (x : ℝ) (h : 1 / (x - 1) < -1) : 0 < x ∧ x < 1 :=
sorry

end NUMINAMATH_GPT_solve_inequality_l480_48064


namespace NUMINAMATH_GPT_watermelon_price_in_units_of_1000_l480_48043

theorem watermelon_price_in_units_of_1000
  (initial_price discounted_price: ℝ)
  (h_price: initial_price = 5000)
  (h_discount: discounted_price = initial_price - 200) :
  discounted_price / 1000 = 4.8 :=
by
  sorry

end NUMINAMATH_GPT_watermelon_price_in_units_of_1000_l480_48043


namespace NUMINAMATH_GPT_division_of_decimals_l480_48029

theorem division_of_decimals : (0.5 : ℝ) / (0.025 : ℝ) = 20 := 
sorry

end NUMINAMATH_GPT_division_of_decimals_l480_48029


namespace NUMINAMATH_GPT_find_unknown_number_l480_48058

theorem find_unknown_number (x : ℝ) (h : (45 + 23 / x) * x = 4028) : x = 89 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l480_48058


namespace NUMINAMATH_GPT_exotic_meat_original_price_l480_48066

theorem exotic_meat_original_price (y : ℝ) :
  (0.75 * (y / 4) = 4.5) → y = 96 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_exotic_meat_original_price_l480_48066


namespace NUMINAMATH_GPT_nth_equation_proof_l480_48024

theorem nth_equation_proof (n : ℕ) (h : n ≥ 1) :
  1 / (n + 1 : ℚ) + 1 / (n * (n + 1)) = 1 / n := 
sorry

end NUMINAMATH_GPT_nth_equation_proof_l480_48024


namespace NUMINAMATH_GPT_negative_implies_neg_reciprocal_positive_l480_48031

theorem negative_implies_neg_reciprocal_positive {x : ℝ} (h : x < 0) : -x⁻¹ > 0 :=
sorry

end NUMINAMATH_GPT_negative_implies_neg_reciprocal_positive_l480_48031


namespace NUMINAMATH_GPT_correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l480_48048

theorem correct_statement_a (x : ℝ) : x > 1 → x^2 > x :=
by sorry

theorem incorrect_statement_b (x : ℝ) : ¬ (x^2 < 0 → x < 0) :=
by sorry

theorem incorrect_statement_c (x : ℝ) : ¬ (x^2 < x → x < 0) :=
by sorry

theorem incorrect_statement_d (x : ℝ) : ¬ (x^2 < 1 → x < 1) :=
by sorry

theorem incorrect_statement_e (x : ℝ) : ¬ (x > 0 → x^2 > x) :=
by sorry

end NUMINAMATH_GPT_correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l480_48048


namespace NUMINAMATH_GPT_no_solution_a_solution_b_l480_48076

def f (n : ℕ) : ℕ :=
  if n = 0 then
    0
  else
    n / 7 + f (n / 7)

theorem no_solution_a :
  ¬ ∃ n : ℕ, 7 ^ 399 ∣ n! ∧ ¬ 7 ^ 400 ∣ n! := sorry

theorem solution_b :
  {n : ℕ | 7 ^ 400 ∣ n! ∧ ¬ 7 ^ 401 ∣ n!} = {2401, 2402, 2403, 2404, 2405, 2406, 2407} := sorry

end NUMINAMATH_GPT_no_solution_a_solution_b_l480_48076


namespace NUMINAMATH_GPT_cuboid_dimensions_exist_l480_48005

theorem cuboid_dimensions_exist (l w h : ℝ) 
  (h1 : l * w = 5) 
  (h2 : l * h = 8) 
  (h3 : w * h = 10) 
  (h4 : l * w * h = 200) : 
  ∃ (l w h : ℝ), l = 4 ∧ w = 2.5 ∧ h = 2 := 
sorry

end NUMINAMATH_GPT_cuboid_dimensions_exist_l480_48005


namespace NUMINAMATH_GPT_max_value_of_f_l480_48090

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem max_value_of_f : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l480_48090


namespace NUMINAMATH_GPT_remaining_batches_l480_48088

def flour_per_batch : ℕ := 2
def batches_baked : ℕ := 3
def initial_flour : ℕ := 20

theorem remaining_batches : (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 := by
  sorry

end NUMINAMATH_GPT_remaining_batches_l480_48088


namespace NUMINAMATH_GPT_correct_answer_is_C_l480_48080

def exactly_hits_n_times (n k : ℕ) : Prop :=
  n = k

def hits_no_more_than (n k : ℕ) : Prop :=
  n ≤ k

def hits_at_least (n k : ℕ) : Prop :=
  n ≥ k

def is_mutually_exclusive (P Q : Prop) : Prop :=
  ¬ (P ∧ Q)

def is_non_opposing (P Q : Prop) : Prop :=
  ¬ P ∧ ¬ Q

def events_are_mutually_exclusive_and_non_opposing (n : ℕ) : Prop :=
  let event1 := exactly_hits_n_times 5 3
  let event2 := exactly_hits_n_times 5 4
  is_mutually_exclusive event1 event2 ∧ is_non_opposing event1 event2

theorem correct_answer_is_C : events_are_mutually_exclusive_and_non_opposing 5 :=
by
  sorry

end NUMINAMATH_GPT_correct_answer_is_C_l480_48080


namespace NUMINAMATH_GPT_total_toys_given_l480_48039

theorem total_toys_given (toys_for_boys : ℕ) (toys_for_girls : ℕ) (h1 : toys_for_boys = 134) (h2 : toys_for_girls = 269) : 
  toys_for_boys + toys_for_girls = 403 := 
by 
  sorry

end NUMINAMATH_GPT_total_toys_given_l480_48039


namespace NUMINAMATH_GPT_box_tape_length_l480_48004

variable (L S : ℕ)
variable (tape_total : ℕ)
variable (num_boxes : ℕ)
variable (square_side : ℕ)

theorem box_tape_length (h1 : num_boxes = 5) (h2 : square_side = 40) (h3 : tape_total = 540) :
  tape_total = 5 * (L + 2 * S) + 2 * 3 * square_side → L = 60 - 2 * S := 
by
  sorry

end NUMINAMATH_GPT_box_tape_length_l480_48004


namespace NUMINAMATH_GPT_correct_option_l480_48085

def condition_A (a : ℝ) : Prop := a^3 * a^4 = a^12
def condition_B (a b : ℝ) : Prop := (-3 * a * b^3)^2 = -6 * a * b^6
def condition_C (a : ℝ) : Prop := (a - 3)^2 = a^2 - 9
def condition_D (x y : ℝ) : Prop := (-x + y) * (x + y) = y^2 - x^2

theorem correct_option (x y : ℝ) : condition_D x y := by
  sorry

end NUMINAMATH_GPT_correct_option_l480_48085


namespace NUMINAMATH_GPT_least_pos_integer_to_yield_multiple_of_5_l480_48051

theorem least_pos_integer_to_yield_multiple_of_5 (n : ℕ) (h : n > 0) :
  ((567 + n) % 5 = 0) ↔ (n = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_least_pos_integer_to_yield_multiple_of_5_l480_48051


namespace NUMINAMATH_GPT_more_bottle_caps_than_wrappers_l480_48065

namespace DannyCollection

def bottle_caps_found := 50
def wrappers_found := 46

theorem more_bottle_caps_than_wrappers :
  bottle_caps_found - wrappers_found = 4 :=
by
  -- We skip the proof here with "sorry"
  sorry

end DannyCollection

end NUMINAMATH_GPT_more_bottle_caps_than_wrappers_l480_48065


namespace NUMINAMATH_GPT_middle_number_l480_48000

theorem middle_number {a b c : ℕ} (h1 : a + b = 12) (h2 : a + c = 17) (h3 : b + c = 19) (h4 : a < b) (h5 : b < c) : b = 7 :=
sorry

end NUMINAMATH_GPT_middle_number_l480_48000


namespace NUMINAMATH_GPT_solution_l480_48020

-- Definition of the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * m * x + 1

-- Statement of the problem
theorem solution (m : ℝ) (x : ℝ) (h : quadratic_equation m x = (m - 2) * x^2 + 3 * m * x + 1) : m ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_l480_48020


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l480_48091

def q (x D E F : ℝ) : ℝ := D*x^4 + E*x^2 + F*x - 2

theorem remainder_when_divided_by_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 14) : q (-2) D E F = -18 := 
by 
     sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_plus_2_l480_48091


namespace NUMINAMATH_GPT_pentagonal_pyramid_faces_l480_48056

-- Definition of a pentagonal pyramid
structure PentagonalPyramid where
  base_sides : Nat := 5
  triangular_faces : Nat := 5

-- The goal is to prove that the total number of faces is 6
theorem pentagonal_pyramid_faces (P : PentagonalPyramid) : P.base_sides + 1 = 6 :=
  sorry

end NUMINAMATH_GPT_pentagonal_pyramid_faces_l480_48056


namespace NUMINAMATH_GPT_total_sides_of_cookie_cutters_l480_48009

theorem total_sides_of_cookie_cutters :
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  total_sides = 75 :=
by
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  show total_sides = 75
  sorry

end NUMINAMATH_GPT_total_sides_of_cookie_cutters_l480_48009


namespace NUMINAMATH_GPT_isosceles_right_triangle_square_ratio_l480_48030

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := Real.sqrt 2 / 2

theorem isosceles_right_triangle_square_ratio :
  x / y = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_square_ratio_l480_48030


namespace NUMINAMATH_GPT_smallest_product_is_298150_l480_48001

def digits : List ℕ := [5, 6, 7, 8, 9, 0]

theorem smallest_product_is_298150 :
  ∃ (a b c : ℕ), 
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c = 298150) :=
sorry

end NUMINAMATH_GPT_smallest_product_is_298150_l480_48001


namespace NUMINAMATH_GPT_integer_solution_interval_l480_48059

theorem integer_solution_interval {f : ℝ → ℝ} (m : ℝ) :
  (∀ x : ℤ, (-x^2 + x + m + 2 ≥ |x| ↔ (x : ℝ) = n)) ↔ (-2 ≤ m ∧ m < -1) := 
sorry

end NUMINAMATH_GPT_integer_solution_interval_l480_48059


namespace NUMINAMATH_GPT_required_speed_is_85_l480_48003

-- Definitions based on conditions
def speed1 := 60
def time1 := 3
def total_time := 5
def average_speed := 70

-- Derived conditions
def distance1 := speed1 * time1
def total_distance := average_speed * total_time
def remaining_distance := total_distance - distance1
def remaining_time := total_time - time1
def required_speed := remaining_distance / remaining_time

-- Theorem statement
theorem required_speed_is_85 : required_speed = 85 := by
    sorry

end NUMINAMATH_GPT_required_speed_is_85_l480_48003


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l480_48025

theorem inequality_holds_for_all_x (m : ℝ) (h : ∀ x : ℝ, |x + 5| ≥ m + 2) : m ≤ -2 :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l480_48025


namespace NUMINAMATH_GPT_sequence_2018_value_l480_48062

theorem sequence_2018_value :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2018 = -3 :=
sorry

end NUMINAMATH_GPT_sequence_2018_value_l480_48062


namespace NUMINAMATH_GPT_ages_sum_13_and_product_72_l480_48018

theorem ages_sum_13_and_product_72 (g b s : ℕ) (h1 : b < g) (h2 : g < s) (h3 : b * g * s = 72) : b + g + s = 13 :=
sorry

end NUMINAMATH_GPT_ages_sum_13_and_product_72_l480_48018


namespace NUMINAMATH_GPT_court_cost_proof_l480_48086

-- Define all the given conditions
def base_fine : ℕ := 50
def penalty_rate : ℕ := 2
def mark_speed : ℕ := 75
def speed_limit : ℕ := 30
def school_zone_multiplier : ℕ := 2
def lawyer_fee_rate : ℕ := 80
def lawyer_hours : ℕ := 3
def total_owed : ℕ := 820

-- Define the calculation for the additional penalty
def additional_penalty : ℕ := (mark_speed - speed_limit) * penalty_rate

-- Define the calculation for the total fine
def total_fine : ℕ := (base_fine + additional_penalty) * school_zone_multiplier

-- Define the calculation for the lawyer's fee
def lawyer_fee : ℕ := lawyer_fee_rate * lawyer_hours

-- Define the calculation for the total of fine and lawyer's fee
def fine_and_lawyer_fee := total_fine + lawyer_fee

-- Prove the court costs
theorem court_cost_proof : total_owed - fine_and_lawyer_fee = 300 := by
  sorry

end NUMINAMATH_GPT_court_cost_proof_l480_48086


namespace NUMINAMATH_GPT_four_digit_number_exists_l480_48081

theorem four_digit_number_exists :
  ∃ (x1 x2 y1 y2 : ℕ), (x1 > 0) ∧ (x2 > 0) ∧ (y1 > 0) ∧ (y2 > 0) ∧
                       (x2 * y2 - x1 * y1 = 67) ∧ (x2 > y2) ∧ (x1 < y1) ∧
                       (x1 * 10^3 + x2 * 10^2 + y2 * 10 + y1 = 1985) := sorry

end NUMINAMATH_GPT_four_digit_number_exists_l480_48081


namespace NUMINAMATH_GPT_probability_of_disease_given_positive_test_l480_48021

-- Define the probabilities given in the problem
noncomputable def pr_D : ℝ := 1 / 1000
noncomputable def pr_Dc : ℝ := 1 - pr_D
noncomputable def pr_T_given_D : ℝ := 1
noncomputable def pr_T_given_Dc : ℝ := 0.05

-- Define the total probability of a positive test using the law of total probability
noncomputable def pr_T := 
  pr_T_given_D * pr_D + pr_T_given_Dc * pr_Dc

-- Using Bayes' theorem
noncomputable def pr_D_given_T := 
  pr_T_given_D * pr_D / pr_T

-- Theorem to prove the desired probability
theorem probability_of_disease_given_positive_test : 
  pr_D_given_T = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_disease_given_positive_test_l480_48021


namespace NUMINAMATH_GPT_square_area_from_diagonal_l480_48055

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : (d^2 / 2) = 72 :=
by sorry

end NUMINAMATH_GPT_square_area_from_diagonal_l480_48055


namespace NUMINAMATH_GPT_min_value_y_l480_48027

theorem min_value_y (x : ℝ) : ∃ x : ℝ, (y = x^2 + 16 * x + 20) ∧ ∀ z : ℝ, (y = z^2 + 16 * z + 20) → y ≥ -44 := 
sorry

end NUMINAMATH_GPT_min_value_y_l480_48027


namespace NUMINAMATH_GPT_triangle_largest_angle_l480_48098

theorem triangle_largest_angle {k : ℝ} (h1 : k > 0)
  (h2 : k + 2 * k + 3 * k = 180) : 3 * k = 90 := 
sorry

end NUMINAMATH_GPT_triangle_largest_angle_l480_48098


namespace NUMINAMATH_GPT_n_mul_n_plus_one_even_l480_48032

theorem n_mul_n_plus_one_even (n : ℤ) : Even (n * (n + 1)) := 
sorry

end NUMINAMATH_GPT_n_mul_n_plus_one_even_l480_48032


namespace NUMINAMATH_GPT_inequality_of_trig_function_l480_48061

theorem inequality_of_trig_function 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end NUMINAMATH_GPT_inequality_of_trig_function_l480_48061


namespace NUMINAMATH_GPT_point_M_coordinates_l480_48017

open Real

theorem point_M_coordinates (θ : ℝ) (h_tan : tan θ = -4 / 3) (h_theta : π / 2 < θ ∧ θ < π) :
  let x := 5 * cos θ
  let y := 5 * sin θ
  (x, y) = (-3, 4) := 
by 
  sorry

end NUMINAMATH_GPT_point_M_coordinates_l480_48017


namespace NUMINAMATH_GPT_sum_nine_terms_l480_48057

-- Definitions required based on conditions provided in Step a)
variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- The arithmetic sequence condition is encapsulated here
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The definition of S_n being the sum of the first n terms
def sum_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- The given condition from the problem
def given_condition (a : ℕ → ℝ) : Prop :=
  2 * a 8 = 6 + a 1

-- The proof statement to show S_9 = 54 given the above conditions
theorem sum_nine_terms (h_arith : is_arithmetic_sequence a d)
                        (h_sum : sum_first_n a S) 
                        (h_given : given_condition a): 
                        S 9 = 54 :=
  by sorry

end NUMINAMATH_GPT_sum_nine_terms_l480_48057


namespace NUMINAMATH_GPT_inequality_solution_sets_l480_48033

theorem inequality_solution_sets (a b m : ℝ) (h_sol_set : ∀ x, x^2 - a * x - 2 > 0 ↔ x < -1 ∨ x > b) (hb : b > -1) (hm : m > -1 / 2) :
  a = 1 ∧ b = 2 ∧ 
  (if m > 0 then ∀ x, (x < -1/m ∨ x > 2) ↔ (mx + 1) * (x - 2) > 0 
   else if m = 0 then ∀ x, x > 2 ↔ (mx + 1) * (x - 2) > 0 
   else ∀ x, (2 < x ∧ x < -1/m) ↔ (mx + 1) * (x - 2) > 0) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_sets_l480_48033


namespace NUMINAMATH_GPT_Brandy_energy_drinks_l480_48079

theorem Brandy_energy_drinks 
  (maximum_safe_amount : ℕ)
  (caffeine_per_drink : ℕ)
  (extra_safe_caffeine : ℕ)
  (x : ℕ)
  (h1 : maximum_safe_amount = 500)
  (h2 : caffeine_per_drink = 120)
  (h3 : extra_safe_caffeine = 20)
  (h4 : caffeine_per_drink * x + extra_safe_caffeine = maximum_safe_amount) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_Brandy_energy_drinks_l480_48079


namespace NUMINAMATH_GPT_total_earnings_correct_l480_48014

-- Define the conditions as initial parameters

def ticket_price : ℕ := 3
def weekday_visitors_per_day : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300

def total_weekday_visitors : ℕ := 5 * weekday_visitors_per_day
def total_weekend_visitors : ℕ := saturday_visitors + sunday_visitors
def total_visitors : ℕ := total_weekday_visitors + total_weekend_visitors

def total_earnings := total_visitors * ticket_price

-- Prove that the total earnings of the amusement park in a week is $3000
theorem total_earnings_correct : total_earnings = 3000 :=
by
  sorry

end NUMINAMATH_GPT_total_earnings_correct_l480_48014


namespace NUMINAMATH_GPT_area_of_four_triangles_l480_48097

theorem area_of_four_triangles (a b : ℕ) (h1 : 2 * b = 28) (h2 : a + 2 * b = 30) :
    4 * (1 / 2 * a * b) = 56 := by
  sorry

end NUMINAMATH_GPT_area_of_four_triangles_l480_48097


namespace NUMINAMATH_GPT_solve_prime_equation_l480_48028

theorem solve_prime_equation (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
    p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0 ↔ (p = 17 ∧ q = 3) :=
by
  sorry

end NUMINAMATH_GPT_solve_prime_equation_l480_48028


namespace NUMINAMATH_GPT_curve_intersection_one_point_l480_48070

theorem curve_intersection_one_point (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2 ↔ y = x^2 + a) → (x, y) = (0, a)) ↔ (a ≥ -1/2) := 
sorry

end NUMINAMATH_GPT_curve_intersection_one_point_l480_48070


namespace NUMINAMATH_GPT_solve_for_y_l480_48049

/-- Given the equation 7(2y + 3) - 5 = -3(2 - 5y), solve for y. -/
theorem solve_for_y (y : ℤ) : 7 * (2 * y + 3) - 5 = -3 * (2 - 5 * y) → y = 22 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_for_y_l480_48049


namespace NUMINAMATH_GPT_tims_seashells_now_l480_48013

def initial_seashells : ℕ := 679
def seashells_given_away : ℕ := 172

theorem tims_seashells_now : (initial_seashells - seashells_given_away) = 507 :=
by
  sorry

end NUMINAMATH_GPT_tims_seashells_now_l480_48013


namespace NUMINAMATH_GPT_valid_root_l480_48042

theorem valid_root:
  ∃ x : ℚ, 
    (3 * x^2 + 5) / (x - 2) - (3 * x + 10) / 4 + (5 - 9 * x) / (x - 2) + 2 = 0 ∧ x = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_valid_root_l480_48042


namespace NUMINAMATH_GPT_mary_bought_48_cards_l480_48084

variable (M T F C B : ℕ)

theorem mary_bought_48_cards
  (h1 : M = 18)
  (h2 : T = 8)
  (h3 : F = 26)
  (h4 : C = 84) :
  B = C - (M - T + F) :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_mary_bought_48_cards_l480_48084


namespace NUMINAMATH_GPT_village_population_percentage_l480_48094

theorem village_population_percentage 
  (part : ℝ)
  (whole : ℝ)
  (h_part : part = 8100)
  (h_whole : whole = 9000) : 
  (part / whole) * 100 = 90 :=
by
  sorry

end NUMINAMATH_GPT_village_population_percentage_l480_48094


namespace NUMINAMATH_GPT_coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l480_48075

noncomputable def coefficient_of_x4_expansion : ℕ :=
  let r := 2;
  let n := 5;
  let general_term_coefficient := Nat.choose n r * 2^(n-r);
  general_term_coefficient

theorem coefficient_of_x4_in_expansion_of_2x_plus_sqrtx :
  coefficient_of_x4_expansion = 80 :=
by
  -- We can bypass the actual proving steps by
  -- acknowledging that the necessary proof mechanism
  -- will properly verify the calculation:
  sorry

end NUMINAMATH_GPT_coefficient_of_x4_in_expansion_of_2x_plus_sqrtx_l480_48075


namespace NUMINAMATH_GPT_egg_cost_l480_48099

theorem egg_cost (toast_cost : ℝ) (E : ℝ) (total_cost : ℝ)
  (dales_toast : ℝ) (dales_eggs : ℝ) (andrews_toast : ℝ) (andrews_eggs : ℝ) :
  toast_cost = 1 → 
  dales_toast = 2 → 
  dales_eggs = 2 → 
  andrews_toast = 1 → 
  andrews_eggs = 2 → 
  total_cost = 15 →
  total_cost = (dales_toast * toast_cost + dales_eggs * E) + 
               (andrews_toast * toast_cost + andrews_eggs * E) →
  E = 3 :=
by
  sorry

end NUMINAMATH_GPT_egg_cost_l480_48099


namespace NUMINAMATH_GPT_percentage_error_in_area_l480_48073

-- Definitions based on conditions
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := s * 1.01
def actual_area (s : ℝ) := s^2
def calculated_area (s : ℝ) := (measured_side s)^2

-- Theorem statement of the proof problem
theorem percentage_error_in_area (s : ℝ) : 
  (calculated_area s - actual_area s) / actual_area s * 100 = 2.01 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_percentage_error_in_area_l480_48073


namespace NUMINAMATH_GPT_hexagon_area_l480_48096

theorem hexagon_area (ABCDEF : Type) (l : ℕ) (h : l = 3) (p q : ℕ)
  (area_hexagon : ℝ) (area_formula : area_hexagon = Real.sqrt p + Real.sqrt q) :
  p + q = 54 := by
  sorry

end NUMINAMATH_GPT_hexagon_area_l480_48096


namespace NUMINAMATH_GPT_least_multiple_of_17_gt_450_l480_48011

def least_multiple_gt (n x : ℕ) (k : ℕ) : Prop :=
  k * n > x ∧ ∀ m : ℕ, m * n > x → m ≥ k

theorem least_multiple_of_17_gt_450 : ∃ k : ℕ, least_multiple_gt 17 450 k :=
by
  use 27
  sorry

end NUMINAMATH_GPT_least_multiple_of_17_gt_450_l480_48011


namespace NUMINAMATH_GPT_product_of_midpoint_l480_48002

-- Define the coordinates of the endpoints
def x1 := 5
def y1 := -4
def x2 := 1
def y2 := 14

-- Define the formulas for the midpoint coordinates
def xm := (x1 + x2) / 2
def ym := (y1 + y2) / 2

-- Define the product of the midpoint coordinates
def product := xm * ym

-- Now state the theorem
theorem product_of_midpoint :
  product = 15 := 
by
  -- Optional: detailed steps can go here if necessary
  sorry

end NUMINAMATH_GPT_product_of_midpoint_l480_48002


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l480_48067

theorem sum_of_reciprocals_of_roots (s₁ s₂ : ℝ) (h₀ : s₁ + s₂ = 15) (h₁ : s₁ * s₂ = 36) :
  (1 / s₁) + (1 / s₂) = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l480_48067


namespace NUMINAMATH_GPT_half_abs_diff_of_squares_l480_48078

theorem half_abs_diff_of_squares (x y : ℤ) (h1 : x = 21) (h2 : y = 19) :
  (|x^2 - y^2| / 2) = 40 := 
by
  subst h1
  subst h2
  sorry

end NUMINAMATH_GPT_half_abs_diff_of_squares_l480_48078


namespace NUMINAMATH_GPT_moles_of_MgSO4_formed_l480_48071

def moles_of_Mg := 3
def moles_of_H2SO4 := 3

theorem moles_of_MgSO4_formed
  (Mg : ℕ)
  (H2SO4 : ℕ)
  (react : ℕ → ℕ → ℕ × ℕ)
  (initial_Mg : Mg = moles_of_Mg)
  (initial_H2SO4 : H2SO4 = moles_of_H2SO4)
  (balanced_eq : react Mg H2SO4 = (Mg, H2SO4)) :
  (react Mg H2SO4).1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_MgSO4_formed_l480_48071


namespace NUMINAMATH_GPT_g_composition_evaluation_l480_48019

def g (x : ℤ) : ℤ :=
  if x < 5 then x^3 + x^2 - 6 else 2 * x - 18

theorem g_composition_evaluation : g (g (g 16)) = 2 := by
  sorry

end NUMINAMATH_GPT_g_composition_evaluation_l480_48019


namespace NUMINAMATH_GPT_sum_series_eq_4_l480_48095

theorem sum_series_eq_4 : 
  (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 4 := 
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_4_l480_48095


namespace NUMINAMATH_GPT_basis_group1_basis_group2_basis_group3_basis_l480_48038

def vector (α : Type*) := α × α

def is_collinear (v1 v2: vector ℝ) : Prop :=
  v1.1 * v2.2 - v2.1 * v1.2 = 0

def group1_v1 : vector ℝ := (-1, 2)
def group1_v2 : vector ℝ := (5, 7)

def group2_v1 : vector ℝ := (3, 5)
def group2_v2 : vector ℝ := (6, 10)

def group3_v1 : vector ℝ := (2, -3)
def group3_v2 : vector ℝ := (0.5, 0.75)

theorem basis_group1 : ¬ is_collinear group1_v1 group1_v2 :=
by sorry

theorem basis_group2 : is_collinear group2_v1 group2_v2 :=
by sorry

theorem basis_group3 : ¬ is_collinear group3_v1 group3_v2 :=
by sorry

theorem basis : (¬ is_collinear group1_v1 group1_v2) ∧ (is_collinear group2_v1 group2_v2) ∧ (¬ is_collinear group3_v1 group3_v2) :=
by sorry

end NUMINAMATH_GPT_basis_group1_basis_group2_basis_group3_basis_l480_48038


namespace NUMINAMATH_GPT_simplify_fraction_l480_48015

theorem simplify_fraction (a b c : ℕ) (h1 : a = 2^2 * 3^2 * 5) 
  (h2 : b = 2^1 * 3^3 * 5) (h3 : c = (2^1 * 3^2 * 5)) :
  (a / c) / (b / c) = 2 / 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_l480_48015


namespace NUMINAMATH_GPT_min_value_expression_l480_48012

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c + 1) * (1 / (a + b + 1) + 1 / (b + c + 1) + 1 / (c + a + 1)) ≥ 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l480_48012


namespace NUMINAMATH_GPT_toys_per_rabbit_l480_48087

-- Define the conditions
def rabbits : ℕ := 34
def toys_mon : ℕ := 8
def toys_tue : ℕ := 3 * toys_mon
def toys_wed : ℕ := 2 * toys_tue
def toys_thu : ℕ := toys_mon
def toys_fri : ℕ := 5 * toys_mon
def toys_sat : ℕ := toys_wed / 2

-- Define the total number of toys
def total_toys : ℕ := toys_mon + toys_tue + toys_wed + toys_thu + toys_fri + toys_sat

-- Define the proof statement
theorem toys_per_rabbit : total_toys / rabbits = 4 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_toys_per_rabbit_l480_48087


namespace NUMINAMATH_GPT_total_friends_met_l480_48074

def num_friends_with_pears : Nat := 9
def num_friends_with_oranges : Nat := 6

theorem total_friends_met : num_friends_with_pears + num_friends_with_oranges = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_friends_met_l480_48074


namespace NUMINAMATH_GPT_find_a3_plus_a9_l480_48047

noncomputable def arithmetic_sequence (a : ℕ → ℕ) : Prop := 
∀ n m : ℕ, a (n + m) = a n + a m

theorem find_a3_plus_a9 (a : ℕ → ℕ) 
  (is_arithmetic : arithmetic_sequence a)
  (h : a 1 + a 6 + a 11 = 3) : 
  a 3 + a 9 = 2 :=
sorry

end NUMINAMATH_GPT_find_a3_plus_a9_l480_48047


namespace NUMINAMATH_GPT_find_a_from_conditions_l480_48040

theorem find_a_from_conditions (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 9) 
  (h3 : c = 4) : 
  a = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_from_conditions_l480_48040


namespace NUMINAMATH_GPT_union_sets_l480_48089

open Set

variable {α : Type*}

def A : Set ℝ := {x | -2 < x ∧ x < 2}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = 2^x}

theorem union_sets : A ∪ B = {z | -2 < z ∧ z < 4} :=
by sorry

end NUMINAMATH_GPT_union_sets_l480_48089


namespace NUMINAMATH_GPT_second_offset_length_l480_48008

theorem second_offset_length (d h1 area : ℝ) (h_diagonal : d = 28) (h_offset1 : h1 = 8) (h_area : area = 140) :
  ∃ x : ℝ, area = (1/2) * d * (h1 + x) ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_second_offset_length_l480_48008
