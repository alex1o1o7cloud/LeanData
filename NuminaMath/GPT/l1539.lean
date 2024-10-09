import Mathlib

namespace ellipse_eq_find_k_l1539_153932

open Real

-- Part I: Prove the equation of the ellipse
theorem ellipse_eq (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 = 1 :=
by
  sorry

-- Part II: Prove the value of k
theorem find_k (P : ℝ × ℝ) (hP : P = (-2, 1)) (MN_length : ℝ) (hMN : MN_length = 2) 
  (k : ℝ) (a b : ℝ) (h_a : a = 2) (h_b : b = 1) :
  ∃ k : ℝ, k = -4 :=
by
  sorry

end ellipse_eq_find_k_l1539_153932


namespace find_f_of_2_l1539_153956

theorem find_f_of_2 (f g : ℝ → ℝ) (h₁ : ∀ x : ℝ, f (-x) = -f x) 
                    (h₂ : ∀ x : ℝ, g x = f x + 9) (h₃ : g (-2) = 3) :
                    f 2 = 6 :=
by
  sorry

end find_f_of_2_l1539_153956


namespace digit_for_divisibility_by_5_l1539_153987

theorem digit_for_divisibility_by_5 (B : ℕ) (h : B < 10) :
  (∃ (n : ℕ), n = 527 * 10 + B ∧ n % 5 = 0) ↔ (B = 0 ∨ B = 5) :=
by sorry

end digit_for_divisibility_by_5_l1539_153987


namespace sin_60_eq_sqrt3_div_2_l1539_153918

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_60_eq_sqrt3_div_2_l1539_153918


namespace range_of_x_l1539_153917

-- Define the condition: the expression under the square root must be non-negative
def condition (x : ℝ) : Prop := 3 + x ≥ 0

-- Define what we want to prove: the range of x such that the condition holds
theorem range_of_x (x : ℝ) : condition x ↔ x ≥ -3 :=
by
  -- Proof goes here
  sorry

end range_of_x_l1539_153917


namespace rectangle_perimeter_inequality_l1539_153935

-- Define rectilinear perimeters
def perimeter (length : ℝ) (width : ℝ) : ℝ := 2 * (length + width)

-- Definitions for rectangles contained within each other
def rectangle_contained (len1 wid1 len2 wid2 : ℝ) : Prop :=
  len1 ≤ len2 ∧ wid1 ≤ wid2

-- Statement of the problem
theorem rectangle_perimeter_inequality (l1 w1 l2 w2 : ℝ) (h : rectangle_contained l1 w1 l2 w2) :
  perimeter l1 w1 ≤ perimeter l2 w2 :=
sorry

end rectangle_perimeter_inequality_l1539_153935


namespace determine_c_l1539_153947

theorem determine_c (c : ℚ) : (∀ x : ℝ, (x + 7) * (x^2 * c * x + 19 * x^2 - c * x - 49) = 0) → c = 21 / 8 :=
by
  sorry

end determine_c_l1539_153947


namespace find_y_coordinate_l1539_153990

theorem find_y_coordinate (m n : ℝ) 
  (h₁ : m = 2 * n + 5) 
  (h₂ : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := 
sorry

end find_y_coordinate_l1539_153990


namespace max_rabbits_l1539_153955

theorem max_rabbits (N : ℕ) (h1 : 13 ≤ N) (h2 : 17 ≤ N) (h3 : 3 ≤ N) (long_ears : {n // n ≤ N}) (jump_far : {n // n ≤ N}) 
  (h_long_ears : (long_ears.val = 13)) (h_jump_far : (jump_far.val = 17)) (h_both : (long_ears.val + jump_far.val - N ≥ 3)) : 
  N = 27 :=
by
  sorry

end max_rabbits_l1539_153955


namespace find_days_A_alone_works_l1539_153985

-- Given conditions
def A_is_twice_as_fast_as_B (a b : ℕ) : Prop := a = b / 2
def together_complete_in_12_days (a b : ℕ) : Prop := (1 / b + 1 / a) = 1 / 12

-- We need to prove that A alone can finish the work in 18 days.
def A_alone_in_18_days (a : ℕ) : Prop := a = 18

theorem find_days_A_alone_works :
  ∃ (a b : ℕ), A_is_twice_as_fast_as_B a b ∧ together_complete_in_12_days a b ∧ A_alone_in_18_days a :=
sorry

end find_days_A_alone_works_l1539_153985


namespace abs_x_lt_2_sufficient_but_not_necessary_l1539_153948

theorem abs_x_lt_2_sufficient_but_not_necessary (x : ℝ) :
  (|x| < 2) → (x ^ 2 - x - 6 < 0) ∧ ¬ ((x ^ 2 - x - 6 < 0) → (|x| < 2)) := by
  sorry

end abs_x_lt_2_sufficient_but_not_necessary_l1539_153948


namespace smallest_n_inequality_l1539_153922

theorem smallest_n_inequality : 
  ∃ (n : ℕ), (n > 0) ∧ ( ∀ m : ℕ, (m > 0) ∧ ( m < n ) → ¬( ( 1 : ℚ ) / m - ( 1 / ( m + 1 : ℚ ) ) < ( 1 / 15 ) ) ) ∧ ( ( 1 : ℚ ) / n - ( 1 / ( n + 1 : ℚ ) ) < ( 1 / 15 ) ) :=
sorry

end smallest_n_inequality_l1539_153922


namespace parameterized_line_solution_l1539_153936

theorem parameterized_line_solution :
  ∃ (s l : ℚ), 
  (∀ t : ℚ, 
    ∃ x y : ℚ, 
      x = -3 + t * l ∧ 
      y = s + t * (-7) ∧ 
      y = 3 * x + 2
  ) ∧
  s = -7 ∧ l = -7 / 3 := 
sorry

end parameterized_line_solution_l1539_153936


namespace find_a_l1539_153926

variable (a : ℝ)

def augmented_matrix (a : ℝ) :=
  ([1, -1, -3], [a, 3, 4])

def solution := (-1, 2)

theorem find_a (hx : -1 - 2 = -3)
               (hy : a * (-1) + 3 * 2 = 4) :
               a = 2 :=
by
  sorry

end find_a_l1539_153926


namespace veranda_width_l1539_153973

def area_of_veranda (w : ℝ) : ℝ :=
  let room_area := 19 * 12
  let total_area := room_area + 140
  let total_length := 19 + 2 * w
  let total_width := 12 + 2 * w
  total_length * total_width - room_area

theorem veranda_width:
  ∃ w : ℝ, area_of_veranda w = 140 := by
  sorry

end veranda_width_l1539_153973


namespace number_of_solution_values_l1539_153907

theorem number_of_solution_values (c : ℕ) : 
  0 ≤ c ∧ c ≤ 2000 ↔ (∃ x : ℝ, 5 * (⌊x⌋ : ℝ) + 3 * (⌈x⌉ : ℝ) = c) →
  c = 251 := 
sorry

end number_of_solution_values_l1539_153907


namespace f_of_3_eq_11_l1539_153940

theorem f_of_3_eq_11 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1 / x) = x^2 + 1 / x^2) : f 3 = 11 :=
by
  sorry

end f_of_3_eq_11_l1539_153940


namespace evaluate_polynomial_given_condition_l1539_153919

theorem evaluate_polynomial_given_condition :
  ∀ x : ℝ, x > 0 → x^2 - 2 * x - 8 = 0 → (x^3 - 2 * x^2 - 8 * x + 4 = 4) := 
by
  intro x hx hcond
  sorry

end evaluate_polynomial_given_condition_l1539_153919


namespace royalty_amount_l1539_153924

-- Define the conditions and the question proof.
theorem royalty_amount (x : ℝ) :
  (800 ≤ x ∧ x ≤ 4000 → (x - 800) * 0.14 = 420) ∧
  (x > 4000 → x * 0.11 = 420) ∧
  420 = 420 →
  x = 3800 :=
by
  sorry

end royalty_amount_l1539_153924


namespace area_inside_octagon_outside_semicircles_l1539_153992

theorem area_inside_octagon_outside_semicircles :
  let s := 3
  let octagon_area := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area := (1/2) * Real.pi * (s / 2)^2
  let total_semicircle_area := 8 * semicircle_area
  octagon_area - total_semicircle_area = 54 + 24 * Real.sqrt 2 - 9 * Real.pi :=
sorry

end area_inside_octagon_outside_semicircles_l1539_153992


namespace unique_five_digit_integers_l1539_153903

-- Define the problem conditions
def digits := [2, 2, 3, 9, 9]
def total_spots := 5
def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

-- Compute the number of five-digit integers that can be formed
noncomputable def num_unique_permutations : Nat :=
  factorial total_spots / (factorial 2 * factorial 1 * factorial 2)

-- Proof statement
theorem unique_five_digit_integers : num_unique_permutations = 30 := by
  sorry

end unique_five_digit_integers_l1539_153903


namespace cost_for_15_pounds_of_apples_l1539_153997

-- Axiom stating the cost of apples per weight
axiom cost_of_apples (pounds : ℕ) : ℕ

-- Condition given in the problem
def rate_apples : Prop := cost_of_apples 5 = 4

-- Statement of the problem
theorem cost_for_15_pounds_of_apples : rate_apples → cost_of_apples 15 = 12 :=
by
  intro h
  -- Proof to be filled in here
  sorry

end cost_for_15_pounds_of_apples_l1539_153997


namespace evaluate_expression_l1539_153967

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by 
  sorry

end evaluate_expression_l1539_153967


namespace sequence_a5_l1539_153925

theorem sequence_a5 : 
    ∃ (a : ℕ → ℚ), 
    a 1 = 1 / 3 ∧ 
    (∀ (n : ℕ), n ≥ 2 → a n = (-1 : ℚ)^n * 2 * a (n - 1)) ∧ 
    a 5 = -16 / 3 := 
sorry

end sequence_a5_l1539_153925


namespace semicircle_circumference_correct_l1539_153991

noncomputable def perimeter_of_rectangle (l b : ℝ) : ℝ := 2 * (l + b)
noncomputable def side_of_square_by_rectangle (l b : ℝ) : ℝ := perimeter_of_rectangle l b / 4
noncomputable def circumference_of_semicircle (d : ℝ) : ℝ := (Real.pi * (d / 2)) + d

theorem semicircle_circumference_correct :
  let l := 16
  let b := 12
  let d := side_of_square_by_rectangle l b
  circumference_of_semicircle d = 35.98 :=
by
  sorry

end semicircle_circumference_correct_l1539_153991


namespace evaluate_expression_l1539_153937

theorem evaluate_expression : ((2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8) :=
sorry

end evaluate_expression_l1539_153937


namespace set_A_enum_l1539_153915

def A : Set ℤ := {z | ∃ x : ℕ, 6 / (x - 2) = z ∧ 6 % (x - 2) = 0}

theorem set_A_enum : A = {-3, -6, 6, 3, 2, 1} := by
  sorry

end set_A_enum_l1539_153915


namespace tan_alpha_value_l1539_153941

variable (α : Real)
variable (h1 : Real.sin α = 4/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_value : Real.tan α = -4/3 := by
  sorry

end tan_alpha_value_l1539_153941


namespace sufficient_condition_for_P_l1539_153965

noncomputable def increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

theorem sufficient_condition_for_P (f : ℝ → ℝ) (t : ℝ) 
  (h_inc : increasing f) (h_val1 : f (-1) = -4) (h_val2 : f 2 = 2) :
  (∀ x, (x ∈ {x | -1 - t < x ∧ x < 2 - t}) → x < -1) → t ≥ 3 :=
by
  sorry

end sufficient_condition_for_P_l1539_153965


namespace find_tangent_lines_l1539_153989

noncomputable def tangent_lines (x y : ℝ) : Prop :=
  (x = 2 ∨ 3 * x - 4 * y + 10 = 0)

theorem find_tangent_lines :
  ∃ (x y : ℝ), tangent_lines x y ∧ (x^2 + y^2 = 4) ∧ ((x, y) ≠ (2, 4)) :=
by
  sorry

end find_tangent_lines_l1539_153989


namespace smallest_Y_l1539_153916

-- Define the necessary conditions
def is_digits_0_1 (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

def is_divisible_by_15 (n : ℕ) : Prop :=
  n % 15 = 0

-- Define the main problem statement
theorem smallest_Y (S Y : ℕ) (hS_pos : S > 0) (hS_digits : is_digits_0_1 S) (hS_div_15 : is_divisible_by_15 S) (hY : Y = S / 15) :
  Y = 74 :=
sorry

end smallest_Y_l1539_153916


namespace ellipse_parabola_intersection_l1539_153908

theorem ellipse_parabola_intersection (a b k m : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = Real.sqrt 2) (h4 : c^2 = a^2 - b^2)
    (h5 : (1 / 2) * 2 * a * 2 * b = 2 * Real.sqrt 3) (h6 : k ≠ 0) :
    (∃ (m: ℝ), (1 / 2) < m ∧ m < 2) :=
sorry

end ellipse_parabola_intersection_l1539_153908


namespace julia_tag_kids_monday_l1539_153931

-- Definitions based on conditions
def total_tag_kids (M T : ℕ) : Prop := M + T = 20
def tag_kids_Tuesday := 13

-- Problem statement
theorem julia_tag_kids_monday (M : ℕ) : total_tag_kids M tag_kids_Tuesday → M = 7 := 
by
  intro h
  sorry

end julia_tag_kids_monday_l1539_153931


namespace isosceles_triangle_apex_angle_l1539_153986

theorem isosceles_triangle_apex_angle (base_angle : ℝ) (h_base_angle : base_angle = 42) : 
  180 - 2 * base_angle = 96 :=
by
  sorry

end isosceles_triangle_apex_angle_l1539_153986


namespace son_age_l1539_153912

theorem son_age (M S : ℕ) (h1: M = S + 26) (h2: M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end son_age_l1539_153912


namespace range_of_smallest_nonprime_with_condition_l1539_153949

def smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 : ℕ :=
121

theorem range_of_smallest_nonprime_with_condition :
  120 < smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ∧ 
  smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10 ≤ 130 :=
by
  unfold smallest_nonprime_greater_than_1_with_no_prime_factors_less_than_10
  exact ⟨by norm_num, by norm_num⟩

end range_of_smallest_nonprime_with_condition_l1539_153949


namespace crease_length_l1539_153914

theorem crease_length 
  (AB AC : ℝ) (BC : ℝ) (BA' : ℝ) (A'C : ℝ)
  (h1 : AB = 10) (h2 : AC = 10) (h3 : BC = 8) (h4 : BA' = 3) (h5 : A'C = 5) :
  ∃ PQ : ℝ, PQ = (Real.sqrt 7393) / 15 := by
  sorry

end crease_length_l1539_153914


namespace max_sides_subdivision_13_max_sides_subdivision_1950_l1539_153909

-- Part (a)
theorem max_sides_subdivision_13 (n : ℕ) (h : n = 13) : 
  ∃ p : ℕ, p ≤ n ∧ p = 13 := 
sorry

-- Part (b)
theorem max_sides_subdivision_1950 (n : ℕ) (h : n = 1950) : 
  ∃ p : ℕ, p ≤ n ∧ p = 1950 := 
sorry

end max_sides_subdivision_13_max_sides_subdivision_1950_l1539_153909


namespace kate_needs_more_money_for_trip_l1539_153981

theorem kate_needs_more_money_for_trip:
  let kate_money_base6 := 3 * 6^3 + 2 * 6^2 + 4 * 6^1 + 2 * 6^0
  let ticket_cost := 1000
  kate_money_base6 - ticket_cost = -254 :=
by
  -- Proving the theorem, steps will go here.
  sorry

end kate_needs_more_money_for_trip_l1539_153981


namespace james_total_distance_l1539_153952

structure Segment where
  speed : ℝ -- speed in mph
  time : ℝ -- time in hours

def totalDistance (segments : List Segment) : ℝ :=
  segments.foldr (λ seg acc => seg.speed * seg.time + acc) 0

theorem james_total_distance :
  let segments := [
    Segment.mk 30 0.5,
    Segment.mk 60 0.75,
    Segment.mk 75 1.5,
    Segment.mk 60 2
  ]
  totalDistance segments = 292.5 :=
by
  sorry

end james_total_distance_l1539_153952


namespace count_twelfth_power_l1539_153910

-- Define the conditions under which a number must meet the criteria of being a square, a cube, and a fourth power
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m^2 = n
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m^3 = n
def is_fourth_power (n : ℕ) : Prop := ∃ m : ℕ, m^4 = n

-- Define the main theorem, which proves the count of numbers less than 1000 meeting all criteria
theorem count_twelfth_power (h : ∀ n, is_square n → is_cube n → is_fourth_power n → n < 1000) :
  ∃! x : ℕ, x < 1000 ∧ ∃ k : ℕ, k^12 = x := 
sorry

end count_twelfth_power_l1539_153910


namespace well_rate_correct_l1539_153938

noncomputable def well_rate (depth : ℝ) (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  let r := diameter / 2
  let volume := Real.pi * r^2 * depth
  total_cost / volume

theorem well_rate_correct :
  well_rate 14 3 1583.3626974092558 = 15.993 :=
by
  sorry

end well_rate_correct_l1539_153938


namespace inequality_sum_l1539_153933

theorem inequality_sum (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a ^ 2 + b ^ 2 + c ^ 2 = 1) :
  (a / (a ^ 3 + b * c) + b / (b ^ 3 + c * a) + c / (c ^ 3 + a * b)) > 3 :=
by
  sorry

end inequality_sum_l1539_153933


namespace no_linear_term_l1539_153993

theorem no_linear_term (m : ℝ) (x : ℝ) : 
  (x + m) * (x + 3) - (x * x + 3 * m) = 0 → m = -3 :=
by
  sorry

end no_linear_term_l1539_153993


namespace find_cubic_polynomial_l1539_153982

theorem find_cubic_polynomial (a b c d : ℚ) :
  (a + b + c + d = -5) →
  (8 * a + 4 * b + 2 * c + d = -8) →
  (27 * a + 9 * b + 3 * c + d = -17) →
  (64 * a + 16 * b + 4 * c + d = -34) →
  a = -1/3 ∧ b = -1 ∧ c = -2/3 ∧ d = -3 :=
by
  intros h1 h2 h3 h4
  sorry

end find_cubic_polynomial_l1539_153982


namespace a2_equals_3_l1539_153942

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem a2_equals_3 (a : ℕ → ℕ) (S3 : ℕ) (h1 : a 1 = 1) (h2 : a 1 + a 2 + a 3 = 9) : a 2 = 3 :=
by
  sorry

end a2_equals_3_l1539_153942


namespace probability_all_same_color_is_correct_l1539_153927

-- Definitions of quantities
def yellow_marbles := 3
def green_marbles := 7
def purple_marbles := 5
def total_marbles := yellow_marbles + green_marbles + purple_marbles

-- Calculation of drawing 4 marbles all the same color
def probability_all_yellow : ℚ := (yellow_marbles / total_marbles) * ((yellow_marbles - 1) / (total_marbles - 1)) * ((yellow_marbles - 2) / (total_marbles - 2)) * ((yellow_marbles - 3) / (total_marbles - 3))
def probability_all_green : ℚ := (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) * ((green_marbles - 2) / (total_marbles - 2)) * ((green_marbles - 3) / (total_marbles - 3))
def probability_all_purple : ℚ := (purple_marbles / total_marbles) * ((purple_marbles - 1) / (total_marbles - 1)) * ((purple_marbles - 2) / (total_marbles - 2)) * ((purple_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles all the same color
def total_probability_same_color : ℚ := probability_all_yellow + probability_all_green + probability_all_purple

-- Theorem statement
theorem probability_all_same_color_is_correct : total_probability_same_color = 532 / 4095 :=
by
  sorry

end probability_all_same_color_is_correct_l1539_153927


namespace sum_of_coordinates_of_B_l1539_153980

def point := (ℝ × ℝ)

noncomputable def point_A : point := (0, 0)

def line_y_equals_6 (B : point) : Prop := B.snd = 6

def slope_AB (A B : point) (m : ℝ) : Prop := (B.snd - A.snd) / (B.fst - A.fst) = m

theorem sum_of_coordinates_of_B (B : point) 
  (h1 : B.snd = 6)
  (h2 : slope_AB point_A B (3/5)) :
  B.fst + B.snd = 16 :=
sorry

end sum_of_coordinates_of_B_l1539_153980


namespace twice_shorter_vs_longer_l1539_153975

-- Definitions and conditions
def total_length : ℝ := 20
def shorter_length : ℝ := 8
def longer_length : ℝ := total_length - shorter_length

-- Statement to prove
theorem twice_shorter_vs_longer :
  2 * shorter_length - longer_length = 4 :=
by
  sorry

end twice_shorter_vs_longer_l1539_153975


namespace number_of_pizzas_l1539_153921

-- Define the conditions
def slices_per_pizza := 8
def total_slices := 168

-- Define the statement we want to prove
theorem number_of_pizzas : total_slices / slices_per_pizza = 21 :=
by
  -- Proof goes here
  sorry

end number_of_pizzas_l1539_153921


namespace calc1_calc2_l1539_153994

theorem calc1 : (1 * -11 + 8 + (-14) = -17) := by
  sorry

theorem calc2 : (13 - (-12) + (-21) = 4) := by
  sorry

end calc1_calc2_l1539_153994


namespace gcd_228_2008_l1539_153945

theorem gcd_228_2008 : Int.gcd 228 2008 = 4 := by
  sorry

end gcd_228_2008_l1539_153945


namespace tan_alpha_minus_beta_l1539_153970

theorem tan_alpha_minus_beta (α β : ℝ) (hα : Real.tan α = 8) (hβ : Real.tan β = 7) :
  Real.tan (α - β) = 1 / 57 := 
sorry

end tan_alpha_minus_beta_l1539_153970


namespace number_of_perfect_squares_and_cubes_l1539_153974

theorem number_of_perfect_squares_and_cubes (n m k : ℕ) (hn : n^2 < 1000) (hn' : (n + 1)^2 ≥ 1000) (hm : m^3 < 1000) (hm' : (m + 1)^3 ≥ 1000) (hk : k^6 < 1000) (hk' : (k + 1)^6 ≥ 1000) :
  (n + m - k) = 38 :=
sorry

end number_of_perfect_squares_and_cubes_l1539_153974


namespace distance_from_x_axis_l1539_153998

theorem distance_from_x_axis (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end distance_from_x_axis_l1539_153998


namespace loss_record_l1539_153900

-- Conditions: a profit of 25 yuan is recorded as +25 yuan.
def profit_record (profit : Int) : Int :=
  profit

-- Statement we need to prove: A loss of 30 yuan is recorded as -30 yuan.
theorem loss_record : profit_record (-30) = -30 :=
by
  sorry

end loss_record_l1539_153900


namespace task1_on_time_task2_not_on_time_l1539_153963

/-- Define the probabilities for task 1 and task 2 -/
def P_A : ℚ := 3 / 8
def P_B : ℚ := 3 / 5

/-- The probability that task 1 will be completed on time but task 2 will not is 3 / 20. -/
theorem task1_on_time_task2_not_on_time (P_A : ℚ) (P_B : ℚ) : P_A = 3 / 8 → P_B = 3 / 5 → P_A * (1 - P_B) = 3 / 20 :=
by
  intros hPA hPB
  rw [hPA, hPB]
  norm_num

end task1_on_time_task2_not_on_time_l1539_153963


namespace farm_horse_food_needed_l1539_153966

-- Definitions given in the problem
def sheep_count : ℕ := 16
def sheep_to_horse_ratio : ℕ × ℕ := (2, 7)
def food_per_horse_per_day : ℕ := 230

-- The statement we want to prove
theorem farm_horse_food_needed : 
  ∃ H : ℕ, (sheep_count * sheep_to_horse_ratio.2 = sheep_to_horse_ratio.1 * H) ∧ 
           (H * food_per_horse_per_day = 12880) :=
sorry

end farm_horse_food_needed_l1539_153966


namespace product_of_variables_l1539_153978

variables (a b c d : ℚ)

theorem product_of_variables :
  4 * a + 5 * b + 7 * c + 9 * d = 82 →
  d + c = 2 * b →
  2 * b + 2 * c = 3 * a →
  c - 2 = d →
  a * b * c * d = 276264960 / 14747943 := by
  sorry

end product_of_variables_l1539_153978


namespace value_of_x_l1539_153977

/-
Given the following conditions:
  x = a + 7,
  a = b + 9,
  b = c + 15,
  c = d + 25,
  d = 60,
Prove that x = 116.
-/

theorem value_of_x (a b c d x : ℤ) 
    (h1 : x = a + 7)
    (h2 : a = b + 9)
    (h3 : b = c + 15)
    (h4 : c = d + 25)
    (h5 : d = 60) : x = 116 := 
  sorry

end value_of_x_l1539_153977


namespace sqrt_2700_minus_37_form_l1539_153904

theorem sqrt_2700_minus_37_form (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : (Int.sqrt 2700 - 37) = Int.sqrt a - b ^ 3) : a + b = 13 :=
sorry

end sqrt_2700_minus_37_form_l1539_153904


namespace intersection_A_B_eq_complement_union_eq_subset_condition_l1539_153901

open Set

noncomputable def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
noncomputable def B : Set ℝ := {x : ℝ | x > 3 / 2}
noncomputable def C (a : ℝ) : Set ℝ := {x : ℝ | 1 < x ∧ x < a}

theorem intersection_A_B_eq : A ∩ B = {x : ℝ | 3 / 2 < x ∧ x ≤ 3} :=
by sorry

theorem complement_union_eq : (univ \ B) ∪ A = {x : ℝ | x ≤ 3} :=
by sorry

theorem subset_condition (a : ℝ) : (C a ⊆ A) → (a ≤ 3) :=
by sorry

end intersection_A_B_eq_complement_union_eq_subset_condition_l1539_153901


namespace base_conversion_403_base_6_eq_223_base_8_l1539_153984

theorem base_conversion_403_base_6_eq_223_base_8 :
  (6^2 * 4 + 6^1 * 0 + 6^0 * 3 : ℕ) = (8^2 * 2 + 8^1 * 2 + 8^0 * 3 : ℕ) :=
by
  sorry

end base_conversion_403_base_6_eq_223_base_8_l1539_153984


namespace tangent_k_value_one_common_point_range_l1539_153983

namespace Geometry

-- Definitions:
def line (k : ℝ) : ℝ → ℝ := λ x => k * x - 3 * k + 2
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4
def is_tangent (k : ℝ) : Prop := |-2 * k + 3| / (Real.sqrt (k^2 + 1)) = 2
def has_only_one_common_point (k : ℝ) : Prop :=
  (1 / 2 < k ∧ k <= 5 / 2) ∨ (k = 5 / 12)

-- Theorem statements:
theorem tangent_k_value : ∀ k : ℝ, is_tangent k → k = 5 / 12 := sorry

theorem one_common_point_range : ∀ k : ℝ, has_only_one_common_point k → k ∈
  Set.union (Set.Ioc (1 / 2) (5 / 2)) {5 / 12} := sorry

end Geometry

end tangent_k_value_one_common_point_range_l1539_153983


namespace fraction_eval_l1539_153971

theorem fraction_eval : 
  ((12^4 + 324) * (24^4 + 324) * (36^4 + 324)) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324)) = (84 / 35) :=
by
  sorry

end fraction_eval_l1539_153971


namespace find_x_value_l1539_153950

theorem find_x_value
  (y₁ y₂ z₁ z₂ x₁ x w k : ℝ)
  (h₁ : y₁ = 3) (h₂ : z₁ = 2) (h₃ : x₁ = 1)
  (h₄ : y₂ = 6) (h₅ : z₂ = 5)
  (inv_rel : ∀ y z k, x = k * (z / y^2))
  (const_prod : ∀ x w, x * w = 1) :
  x = 5 / 8 :=
by
  -- omitted proof steps
  sorry

end find_x_value_l1539_153950


namespace faye_coloring_books_l1539_153911

theorem faye_coloring_books (initial_books : ℕ) (gave_away : ℕ) (bought_more : ℕ) (h1 : initial_books = 34) (h2 : gave_away = 3) (h3 : bought_more = 48) : 
  initial_books - gave_away + bought_more = 79 :=
by
  sorry

end faye_coloring_books_l1539_153911


namespace smallest_angle_pentagon_l1539_153988

theorem smallest_angle_pentagon (x : ℝ) (h : 16 * x = 540) : 2 * x = 67.5 := 
by 
  sorry

end smallest_angle_pentagon_l1539_153988


namespace charlene_initial_necklaces_l1539_153962

-- Definitions for the conditions.
def necklaces_sold : ℕ := 16
def necklaces_giveaway : ℕ := 18
def necklaces_left : ℕ := 26

-- Statement to prove that the initial number of necklaces is 60.
theorem charlene_initial_necklaces : necklaces_sold + necklaces_giveaway + necklaces_left = 60 := by
  sorry

end charlene_initial_necklaces_l1539_153962


namespace complex_fraction_simplification_l1539_153976

open Complex

theorem complex_fraction_simplification : (1 + 2 * I) / (1 - I)^2 = 1 - 1 / 2 * I :=
by
  -- Proof omitted
  sorry

end complex_fraction_simplification_l1539_153976


namespace sally_pokemon_cards_l1539_153996

variable (X : ℤ)

theorem sally_pokemon_cards : X + 41 + 20 = 34 → X = -27 :=
by
  sorry

end sally_pokemon_cards_l1539_153996


namespace find_smaller_number_l1539_153972

theorem find_smaller_number (a b : ℕ) (h1 : b = 2 * a - 3) (h2 : a + b = 39) : a = 14 :=
by
  -- Sorry to skip the proof
  sorry

end find_smaller_number_l1539_153972


namespace escalator_steps_l1539_153964

theorem escalator_steps
  (x : ℕ)
  (time_me : ℕ := 60)
  (steps_me : ℕ := 20)
  (time_wife : ℕ := 72)
  (steps_wife : ℕ := 16)
  (escalator_speed_me : x - steps_me = 60 * (x - 20) / 72)
  (escalator_speed_wife : x - steps_wife = 72 * (x - 16) / 60) :
  x = 40 := by
  sorry

end escalator_steps_l1539_153964


namespace height_of_david_l1539_153995

theorem height_of_david
  (building_height : ℕ)
  (building_shadow : ℕ)
  (david_shadow : ℕ)
  (ratio : ℕ)
  (h1 : building_height = 50)
  (h2 : building_shadow = 25)
  (h3 : david_shadow = 18)
  (h4 : ratio = building_height / building_shadow) :
  david_shadow * ratio = 36 := sorry

end height_of_david_l1539_153995


namespace intersection_of_M_and_N_l1539_153959

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}
def intersection := {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7}

theorem intersection_of_M_and_N : M ∩ N = intersection := by
  sorry

end intersection_of_M_and_N_l1539_153959


namespace problem_statement_l1539_153968

theorem problem_statement (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end problem_statement_l1539_153968


namespace X_Y_Z_sum_eq_17_l1539_153979

variable {X Y Z : ℤ}

def base_ten_representation_15_fac (X Y Z : ℤ) : Prop :=
  Z = 0 ∧ (28 + X + Y) % 9 = 8 ∧ (X - Y) % 11 = 11

theorem X_Y_Z_sum_eq_17 (X Y Z : ℤ) (h : base_ten_representation_15_fac X Y Z) : X + Y + Z = 17 :=
by
  sorry

end X_Y_Z_sum_eq_17_l1539_153979


namespace line_equation_l1539_153934

theorem line_equation (p : ℝ × ℝ) (a : ℝ × ℝ) :
  p = (4, -4) →
  a = (1, 2 / 7) →
  ∃ (m b : ℝ), m = 2 / 7 ∧ b = -36 / 7 ∧ ∀ x y : ℝ, y = m * x + b :=
by
  intros hp ha
  sorry

end line_equation_l1539_153934


namespace bridge_length_increase_l1539_153946

open Real

def elevation_change : ℝ := 800
def original_gradient : ℝ := 0.02
def new_gradient : ℝ := 0.015

theorem bridge_length_increase :
  let original_length := elevation_change / original_gradient
  let new_length := elevation_change / new_gradient
  new_length - original_length = 13333 := by
  sorry

end bridge_length_increase_l1539_153946


namespace exists_nat_sol_x9_eq_2013y10_l1539_153905

theorem exists_nat_sol_x9_eq_2013y10 : ∃ (x y : ℕ), x^9 = 2013 * y^10 :=
by {
  -- Assume x and y are natural numbers, and prove that x^9 = 2013 y^10 has a solution
  sorry
}

end exists_nat_sol_x9_eq_2013y10_l1539_153905


namespace value_of_c7_l1539_153999

theorem value_of_c7 
  (a : ℕ → ℕ)
  (b : ℕ → ℕ)
  (c : ℕ → ℕ)
  (h1 : ∀ n, a n = n)
  (h2 : ∀ n, b n = 2^(n-1))
  (h3 : ∀ n, c n = a n * b n) :
  c 7 = 448 :=
by
  sorry

end value_of_c7_l1539_153999


namespace union_of_A_and_B_l1539_153960

namespace SetUnionProof

def A : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | x ≤ 2 }
def C : Set ℝ := { x | x ≤ 2 }

theorem union_of_A_and_B : A ∪ B = C := by
  -- proof goes here
  sorry

end SetUnionProof

end union_of_A_and_B_l1539_153960


namespace greatest_possible_n_l1539_153951

theorem greatest_possible_n (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 :=
by {
  sorry
}

end greatest_possible_n_l1539_153951


namespace sum_of_coefficients_l1539_153957

def polynomial (x : ℤ) : ℤ := 3 * (x^8 - 2 * x^5 + 4 * x^3 - 7) - 5 * (2 * x^4 - 3 * x^2 + 8) + 6 * (x^6 - 3)

theorem sum_of_coefficients : polynomial 1 = -59 := 
by
  sorry

end sum_of_coefficients_l1539_153957


namespace longest_side_triangle_l1539_153930

theorem longest_side_triangle (x : ℝ) 
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) : 
  max 7 (max (x + 4) (2 * x + 1)) = 17 :=
by sorry

end longest_side_triangle_l1539_153930


namespace total_cost_l1539_153943

-- Definition: Cost of first 100 notebooks
def cost_first_100_notebooks : ℕ := 230

-- Definition: Cost per notebook beyond the first 100 notebooks
def cost_additional_notebooks (n : ℕ) : ℕ := n * 2

-- Theorem: Total cost given a > 100 notebooks
theorem total_cost (a : ℕ) (h : a > 100) : (cost_first_100_notebooks + cost_additional_notebooks (a - 100) = 2 * a + 30) := by
  sorry

end total_cost_l1539_153943


namespace trajectory_of_M_l1539_153906

variable (P : ℝ × ℝ) (A : ℝ × ℝ := (4, 0))
variable (M : ℝ × ℝ)

theorem trajectory_of_M (hP : P.1^2 + 4 * P.2^2 = 4) (hM : M = ((P.1 + 4) / 2, P.2 / 2)) :
  (M.1 - 2)^2 + 4 * M.2^2 = 1 :=
by
  sorry

end trajectory_of_M_l1539_153906


namespace boat_speed_upstream_l1539_153944

noncomputable def V_b : ℝ := 11
noncomputable def V_down : ℝ := 15
noncomputable def V_s : ℝ := V_down - V_b
noncomputable def V_up : ℝ := V_b - V_s

theorem boat_speed_upstream :
  V_up = 7 := by
  sorry

end boat_speed_upstream_l1539_153944


namespace smallest_x_l1539_153969

theorem smallest_x 
  (x : ℝ)
  (h : ( ( (5 * x - 20) / (4 * x - 5) ) ^ 2 + ( (5 * x - 20) / (4 * x - 5) ) ) = 6 ) :
  x = -10 / 3 := sorry

end smallest_x_l1539_153969


namespace compare_final_values_l1539_153954

noncomputable def final_value_Almond (initial: ℝ): ℝ := (initial * 1.15) * 0.85
noncomputable def final_value_Bean (initial: ℝ): ℝ := (initial * 0.80) * 1.20
noncomputable def final_value_Carrot (initial: ℝ): ℝ := (initial * 1.10) * 0.90

theorem compare_final_values (initial: ℝ) (h_positive: 0 < initial):
  final_value_Almond initial < final_value_Bean initial ∧ 
  final_value_Bean initial < final_value_Carrot initial := by
  sorry

end compare_final_values_l1539_153954


namespace wickets_before_last_match_l1539_153958

theorem wickets_before_last_match (W : ℕ) (avg_before : ℝ) (wickets_taken : ℕ) (runs_conceded : ℝ) (avg_drop : ℝ) :
  avg_before = 12.4 → wickets_taken = 4 → runs_conceded = 26 → avg_drop = 0.4 →
  (avg_before - avg_drop) * (W + wickets_taken) = avg_before * W + runs_conceded →
  W = 55 :=
by
  intros
  sorry

end wickets_before_last_match_l1539_153958


namespace observed_wheels_l1539_153928

theorem observed_wheels (num_cars wheels_per_car : ℕ) (h1 : num_cars = 12) (h2 : wheels_per_car = 4) : num_cars * wheels_per_car = 48 := by
  sorry

end observed_wheels_l1539_153928


namespace periodic_length_le_T_l1539_153920

noncomputable def purely_periodic (a : ℚ) (T : ℕ) : Prop :=
∃ p : ℤ, a = p / (10^T - 1)

theorem periodic_length_le_T {a b : ℚ} {T : ℕ} 
  (ha : purely_periodic a T) 
  (hb : purely_periodic b T) 
  (hab_sum : purely_periodic (a + b) T)
  (hab_prod : purely_periodic (a * b) T) :
  ∃ Ta Tb : ℕ, Ta ≤ T ∧ Tb ≤ T ∧ purely_periodic a Ta ∧ purely_periodic b Tb := 
sorry

end periodic_length_le_T_l1539_153920


namespace calc_angle_CAB_l1539_153939

theorem calc_angle_CAB (α β γ ε : ℝ) (hα : α = 79) (hβ : β = 63) (hγ : γ = 131) (hε : ε = 123.5) : 
  ∃ φ : ℝ, φ = 24 + 52 / 60 :=
by
  sorry

end calc_angle_CAB_l1539_153939


namespace parabola_min_value_sum_abc_zero_l1539_153961

theorem parabola_min_value_sum_abc_zero
  (a b c : ℝ)
  (h1 : ∀ x y : ℝ, y = ax^2 + bx + c → y = 0 ∧ ((x = 1) ∨ (x = 5)))
  (h2 : ∃ x : ℝ, (∀ t : ℝ, ax^2 + bx + c ≤ t^2) ∧ (ax^2 + bx + c = 36)) :
  a + b + c = 0 :=
sorry

end parabola_min_value_sum_abc_zero_l1539_153961


namespace f_fe_eq_neg1_f_x_gt_neg1_solution_l1539_153923

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (1 / x)
  else if x < 0 then 1 / x
  else 0  -- handle the case for x = 0 explicitly if needed

theorem f_fe_eq_neg1 : 
  f (f (Real.exp 1)) = -1 := 
by
  -- proof to be filled in
  sorry

theorem f_x_gt_neg1_solution :
  {x : ℝ | f x > -1} = {x : ℝ | (x < -1) ∨ (0 < x ∧ x < Real.exp 1)} :=
by
  -- proof to be filled in
  sorry

end f_fe_eq_neg1_f_x_gt_neg1_solution_l1539_153923


namespace maximum_value_40_l1539_153929

theorem maximum_value_40 (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 10) :
  (a-b)^2 + (a-c)^2 + (a-d)^2 + (b-c)^2 + (b-d)^2 + (c-d)^2 ≤ 40 :=
sorry

end maximum_value_40_l1539_153929


namespace hyperbola_asymptotes_l1539_153953

theorem hyperbola_asymptotes:
  (∀ x y : Real, (x^2 / 16 - y^2 / 9 = 1) → (y = 3 / 4 * x ∨ y = -3 / 4 * x)) :=
by {
  sorry
}

end hyperbola_asymptotes_l1539_153953


namespace remainder_of_power_modulo_l1539_153902

theorem remainder_of_power_modulo (n : ℕ) (h : n = 2040) : 
  3^2040 % 5 = 1 :=
by
  sorry

end remainder_of_power_modulo_l1539_153902


namespace fewer_cucumbers_than_potatoes_l1539_153913

theorem fewer_cucumbers_than_potatoes :
  ∃ C : ℕ, 237 + C + 2 * C = 768 ∧ 237 - C = 60 :=
by
  sorry

end fewer_cucumbers_than_potatoes_l1539_153913
