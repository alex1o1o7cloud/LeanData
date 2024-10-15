import Mathlib

namespace NUMINAMATH_GPT_sum_of_coeffs_eq_one_l79_7929

theorem sum_of_coeffs_eq_one (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) (x : ℝ) :
  (1 - 2 * x) ^ 10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + 
                    a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1 :=
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_eq_one_l79_7929


namespace NUMINAMATH_GPT_correct_quadratic_equation_l79_7954

-- The main statement to prove.
theorem correct_quadratic_equation :
  (∀ (x y a : ℝ), (3 * x + 2 * y - 1 ≠ 0) ∧ (5 * x^2 - 6 * y - 3 ≠ 0) ∧ (a * x^2 - x + 2 ≠ 0) ∧ (x^2 - 1 = 0) → (x^2 - 1 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_correct_quadratic_equation_l79_7954


namespace NUMINAMATH_GPT_probability_even_sum_of_spins_l79_7994

theorem probability_even_sum_of_spins :
  let prob_even_first := 3 / 6
  let prob_odd_first := 3 / 6
  let prob_even_second := 2 / 5
  let prob_odd_second := 3 / 5
  let prob_both_even := prob_even_first * prob_even_second
  let prob_both_odd := prob_odd_first * prob_odd_second
  prob_both_even + prob_both_odd = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_probability_even_sum_of_spins_l79_7994


namespace NUMINAMATH_GPT_find_reciprocal_l79_7969

open Real

theorem find_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^3 + y^3 + 1 / 27 = x * y) : 1 / x = 3 := 
sorry

end NUMINAMATH_GPT_find_reciprocal_l79_7969


namespace NUMINAMATH_GPT_student_b_speed_l79_7987

theorem student_b_speed (distance : ℝ) (rate_A : ℝ) (time_diff : ℝ) (rate_B : ℝ) : 
  distance = 12 → rate_A = 1.2 * rate_B → time_diff = 1 / 6 → rate_B = 12 :=
by
  intros h_dist h_rateA h_time_diff
  sorry

end NUMINAMATH_GPT_student_b_speed_l79_7987


namespace NUMINAMATH_GPT_frequency_of_zero_in_3021004201_l79_7917

def digit_frequency (n : Nat) (d : Nat) :  Rat :=
  let digits := n.digits 10
  let count_d := digits.count d
  (count_d : Rat) / digits.length

theorem frequency_of_zero_in_3021004201 : 
  digit_frequency 3021004201 0 = 0.4 := 
by 
  sorry

end NUMINAMATH_GPT_frequency_of_zero_in_3021004201_l79_7917


namespace NUMINAMATH_GPT_range_of_a_l79_7901

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l79_7901


namespace NUMINAMATH_GPT_sum_dihedral_angles_gt_360_l79_7990

-- Define the structure Tetrahedron
structure Tetrahedron (α : Type*) :=
  (A B C D : α)

-- Define the dihedral angles function
noncomputable def sum_dihedral_angles {α : Type*} (T : Tetrahedron α) : ℝ := 
  -- Placeholder for the actual sum of dihedral angles of T
  sorry

-- Statement of the problem
theorem sum_dihedral_angles_gt_360 {α : Type*} (T : Tetrahedron α) :
  sum_dihedral_angles T > 360 := 
sorry

end NUMINAMATH_GPT_sum_dihedral_angles_gt_360_l79_7990


namespace NUMINAMATH_GPT_find_a_even_function_l79_7942

theorem find_a_even_function (a : ℝ) :
  (∀ x : ℝ, (x ^ 2 + a * x - 4) = ((-x) ^ 2 + a * (-x) - 4)) → a = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_even_function_l79_7942


namespace NUMINAMATH_GPT_sam_last_30_minutes_speed_l79_7900

/-- 
Given the total distance of 96 miles driven in 1.5 hours, 
with the first 30 minutes at an average speed of 60 mph, 
and the second 30 minutes at an average speed of 65 mph,
we need to show that the average speed during the last 30 minutes was 67 mph.
-/
theorem sam_last_30_minutes_speed (total_distance : ℤ) (time1 time2 : ℤ) (speed1 speed2 speed_last segment_time : ℤ)
  (h_total_distance : total_distance = 96)
  (h_total_time : time1 + time2 + segment_time = 90)
  (h_segment_time : segment_time = 30)
  (convert_time1 : time1 = 30)
  (convert_time2 : time2 = 30)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 65)
  (h_average_speed : ((60 + 65 + speed_last) / 3) = 64) :
  speed_last = 67 := 
sorry

end NUMINAMATH_GPT_sam_last_30_minutes_speed_l79_7900


namespace NUMINAMATH_GPT_range_of_a_l79_7981

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -x^2 + 2 * x + 3 ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l79_7981


namespace NUMINAMATH_GPT_value_of_x_y_squared_l79_7920

theorem value_of_x_y_squared (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : (x - y)^2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_y_squared_l79_7920


namespace NUMINAMATH_GPT_compound_interest_calculation_l79_7933

noncomputable def compoundInterest (P r t : ℝ) : ℝ :=
  P * (1 + r)^t - P

noncomputable def simpleInterest (P r t : ℝ) : ℝ :=
  P * r * t

theorem compound_interest_calculation :
  ∃ P : ℝ, simpleInterest P 0.10 2 = 600 ∧ compoundInterest P 0.10 2 = 630 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_calculation_l79_7933


namespace NUMINAMATH_GPT_tank_capacity_l79_7974

theorem tank_capacity :
  ∀ (T : ℚ), (3 / 4) * T + 4 = (7 / 8) * T → T = 32 :=
by
  intros T h
  sorry

end NUMINAMATH_GPT_tank_capacity_l79_7974


namespace NUMINAMATH_GPT_scientific_notation_600_million_l79_7989

theorem scientific_notation_600_million : (600000000 : ℝ) = 6 * 10^8 := 
by 
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_scientific_notation_600_million_l79_7989


namespace NUMINAMATH_GPT_quadratic_equal_roots_l79_7950

theorem quadratic_equal_roots :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 → (0 ≤ 0) ∧ 
  (∀ a b : ℝ, 0 = b^2 - 4 * a * 1 → (x = -b / (2 * a))) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equal_roots_l79_7950


namespace NUMINAMATH_GPT_binomial_coeffs_not_arith_seq_l79_7948

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def are_pos_integer (n : ℕ) : Prop := n > 0

def is_arith_seq (a b c d : ℕ) : Prop := 
  2 * b = a + c ∧ 2 * c = b + d 

theorem binomial_coeffs_not_arith_seq (n r : ℕ) : 
  are_pos_integer n → are_pos_integer r → n ≥ r + 3 → ¬ is_arith_seq (binomial n r) (binomial n (r+1)) (binomial n (r+2)) (binomial n (r+3)) :=
by
  sorry

end NUMINAMATH_GPT_binomial_coeffs_not_arith_seq_l79_7948


namespace NUMINAMATH_GPT_circle_numbers_exist_l79_7939

theorem circle_numbers_exist :
  ∃ (a b c d e f : ℚ),
    a = 2 ∧
    b = 3 ∧
    c = 3 / 2 ∧
    d = 1 / 2 ∧
    e = 1 / 3 ∧
    f = 2 / 3 ∧
    a = b * f ∧
    b = a * c ∧
    c = b * d ∧
    d = c * e ∧
    e = d * f ∧
    f = e * a := by
  sorry

end NUMINAMATH_GPT_circle_numbers_exist_l79_7939


namespace NUMINAMATH_GPT_jason_tattoos_on_each_leg_l79_7955

-- Define the basic setup
variable (x : ℕ)

-- Define the number of tattoos Jason has on each leg
def tattoos_on_each_leg := x

-- Define the total number of tattoos Jason has
def total_tattoos_jason := 2 + 2 + 2 * x

-- Define the total number of tattoos Adam has
def total_tattoos_adam := 23

-- Define the relation between Adam's and Jason's tattoos
def relation := 2 * total_tattoos_jason + 3 = total_tattoos_adam

-- The proof statement we need to show
theorem jason_tattoos_on_each_leg : tattoos_on_each_leg = 3  :=
by
  sorry

end NUMINAMATH_GPT_jason_tattoos_on_each_leg_l79_7955


namespace NUMINAMATH_GPT_puppies_per_cage_l79_7940

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (remaining_puppies : ℕ)
  (cages : ℕ)
  (puppies_per_cage : ℕ)
  (h1 : initial_puppies = 78)
  (h2 : sold_puppies = 30)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 6)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 8 := by
  sorry

end NUMINAMATH_GPT_puppies_per_cage_l79_7940


namespace NUMINAMATH_GPT_find_q_l79_7931

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 6) : q = 3 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l79_7931


namespace NUMINAMATH_GPT_sequence_diff_l79_7946

theorem sequence_diff (x : ℕ → ℕ)
  (h1 : ∀ n, x n < x (n + 1))
  (h2 : ∀ n, 2 * n + 1 ≤ x (2 * n + 1)) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end NUMINAMATH_GPT_sequence_diff_l79_7946


namespace NUMINAMATH_GPT_truncated_quadrilateral_pyramid_exists_l79_7932

theorem truncated_quadrilateral_pyramid_exists :
  ∃ (x y z u r s t : ℤ),
    x = 4 * r * t ∧
    y = 4 * s * t ∧
    z = (r - s)^2 - 2 * t^2 ∧
    u = (r - s)^2 + 2 * t^2 ∧
    (x - y)^2 + 2 * z^2 = 2 * u^2 :=
by
  sorry

end NUMINAMATH_GPT_truncated_quadrilateral_pyramid_exists_l79_7932


namespace NUMINAMATH_GPT_complement_intersection_l79_7983

noncomputable def M : Set ℝ := {x | |x| > 2}
noncomputable def N : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | 1 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l79_7983


namespace NUMINAMATH_GPT_closest_point_on_line_l79_7958

theorem closest_point_on_line (x y: ℚ) (h1: y = -4 * x + 3) (h2: ∀ p q: ℚ, y = -4 * p + 3 ∧ y = q * (-4 * p) - q * (-4 * 1 + 0)): (x, y) = (-1 / 17, 55 / 17) :=
sorry

end NUMINAMATH_GPT_closest_point_on_line_l79_7958


namespace NUMINAMATH_GPT_right_triangle_area_l79_7921

theorem right_triangle_area (h : Real) (a : Real) (b : Real) (c : Real) (h_is_hypotenuse : h = 13) (a_is_leg : a = 5) (pythagorean_theorem : a^2 + b^2 = h^2) : (1 / 2) * a * b = 30 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_area_l79_7921


namespace NUMINAMATH_GPT_sum_of_squares_of_two_numbers_l79_7985

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) :
  x^2 + y^2 = 289 := 
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_two_numbers_l79_7985


namespace NUMINAMATH_GPT_dan_spent_amount_l79_7902

-- Defining the prices of items
def candy_bar_price : ℝ := 7
def chocolate_price : ℝ := 6
def gum_price : ℝ := 3
def chips_price : ℝ := 4

-- Defining the discount and tax rates
def candy_bar_discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Defining the steps to calculate the total price including discount and tax
def total_before_discount_and_tax := candy_bar_price + chocolate_price + gum_price + chips_price
def candy_bar_discount := candy_bar_discount_rate * candy_bar_price
def candy_bar_after_discount := candy_bar_price - candy_bar_discount
def total_after_discount := candy_bar_after_discount + chocolate_price + gum_price + chips_price
def tax := tax_rate * total_after_discount
def total_with_discount_and_tax := total_after_discount + tax

theorem dan_spent_amount : total_with_discount_and_tax = 20.27 :=
by sorry

end NUMINAMATH_GPT_dan_spent_amount_l79_7902


namespace NUMINAMATH_GPT_find_b_for_parallel_lines_l79_7922

theorem find_b_for_parallel_lines :
  (∀ (b : ℝ), (∃ (f g : ℝ → ℝ),
  (∀ x, f x = 3 * x + b) ∧
  (∀ x, g x = (b + 9) * x - 2) ∧
  (∀ x, f x = g x → False)) →
  b = -6) :=
sorry

end NUMINAMATH_GPT_find_b_for_parallel_lines_l79_7922


namespace NUMINAMATH_GPT_total_earnings_from_peaches_l79_7908

-- Definitions of the conditions
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def price_per_peach_friends : ℝ := 2
def peaches_sold_to_relatives : ℕ :=  4
def price_per_peach_relatives : ℝ := 1.25
def peaches_for_self : ℕ := 1

-- We aim to prove the following statement
theorem total_earnings_from_peaches :
  (peaches_sold_to_friends * price_per_peach_friends) +
  (peaches_sold_to_relatives * price_per_peach_relatives) = 25 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_earnings_from_peaches_l79_7908


namespace NUMINAMATH_GPT_derivative_at_neg_one_l79_7982

def f (x : ℝ) : ℝ := x^3 + 2*x^2 - 1

theorem derivative_at_neg_one : deriv f (-1) = -1 :=
by
  -- definition of the function
  -- proof of the statement
  sorry

end NUMINAMATH_GPT_derivative_at_neg_one_l79_7982


namespace NUMINAMATH_GPT_least_number_to_multiply_for_multiple_of_112_l79_7926

theorem least_number_to_multiply_for_multiple_of_112 (n : ℕ) : 
  (Nat.lcm 72 112) / 72 = 14 := 
sorry

end NUMINAMATH_GPT_least_number_to_multiply_for_multiple_of_112_l79_7926


namespace NUMINAMATH_GPT_largest_value_among_given_numbers_l79_7988

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem largest_value_among_given_numbers :
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20 
  b > a ∧ b > c ∧ b > d :=
by
  let a := Real.log (Real.sqrt 2)
  let b := 1 / Real.exp 1
  let c := (Real.log Real.pi) / Real.pi
  let d := (Real.sqrt 10 * Real.log 10) / 20
  -- Add the necessary steps to show that b is the largest value
  sorry

end NUMINAMATH_GPT_largest_value_among_given_numbers_l79_7988


namespace NUMINAMATH_GPT_part1_part2_l79_7906

open Complex

-- Define the first proposition p
def p (m : ℝ) : Prop :=
  (m - 1 < 0) ∧ (m + 3 > 0)

-- Define the second proposition q
def q (m : ℝ) : Prop :=
  abs (Complex.mk 1 (m - 2)) ≤ Real.sqrt 10

-- Prove the first part of the problem
theorem part1 (m : ℝ) (hp : p m) : -3 < m ∧ m < 1 :=
sorry

-- Prove the second part of the problem
theorem part2 (m : ℝ) (h : ¬ (p m ∧ q m) ∧ (p m ∨ q m)) : (-3 < m ∧ m < -1) ∨ (1 ≤ m ∧ m ≤ 5) :=
sorry

end NUMINAMATH_GPT_part1_part2_l79_7906


namespace NUMINAMATH_GPT_sin_cos_identity_l79_7998

theorem sin_cos_identity {x : Real} 
    (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
    Real.sin x ^ 12 + Real.cos x ^ 12 = 5 / 18 :=
sorry

end NUMINAMATH_GPT_sin_cos_identity_l79_7998


namespace NUMINAMATH_GPT_rectangle_length_l79_7949

theorem rectangle_length {b l : ℝ} (h1 : 2 * (l + b) = 5 * b) (h2 : l * b = 216) : l = 18 := by
    sorry

end NUMINAMATH_GPT_rectangle_length_l79_7949


namespace NUMINAMATH_GPT_range_of_p_l79_7996

-- Definitions of A and B
def A (p : ℝ) := {x : ℝ | x^2 + (p + 2) * x + 1 = 0}
def B := {x : ℝ | x > 0}

-- Condition of the problem: A ∩ B = ∅
def condition (p : ℝ) := ∀ x ∈ A p, x ∉ B

-- The statement to prove: p > -4
theorem range_of_p (p : ℝ) : condition p → p > -4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_p_l79_7996


namespace NUMINAMATH_GPT_molecular_weight_of_3_moles_HBrO3_l79_7968

-- Definitions from the conditions
def mol_weight_H : ℝ := 1.01  -- atomic weight of H
def mol_weight_Br : ℝ := 79.90  -- atomic weight of Br
def mol_weight_O : ℝ := 16.00  -- atomic weight of O

-- Definition of molecular weight of HBrO3
def mol_weight_HBrO3 : ℝ := mol_weight_H + mol_weight_Br + 3 * mol_weight_O

-- The goal: The molecular weight of 3 moles of HBrO3 is 386.73 grams
theorem molecular_weight_of_3_moles_HBrO3 : 3 * mol_weight_HBrO3 = 386.73 :=
by
  -- We will insert the proof here later
  sorry

end NUMINAMATH_GPT_molecular_weight_of_3_moles_HBrO3_l79_7968


namespace NUMINAMATH_GPT_triangle_with_positive_area_l79_7971

noncomputable def num_triangles_with_A (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) : ℕ :=
  let points_excluding_A := total_points.erase A
  let total_pairs := points_excluding_A.card.choose 2
  let collinear_pairs := 20  -- Derived from the problem; in practice this would be calculated
  total_pairs - collinear_pairs

theorem triangle_with_positive_area (total_points : Finset (ℕ × ℕ)) (A : ℕ × ℕ) (h : total_points.card = 25):
  num_triangles_with_A total_points A = 256 :=
by
  sorry

end NUMINAMATH_GPT_triangle_with_positive_area_l79_7971


namespace NUMINAMATH_GPT_binary_representation_of_23_l79_7995

theorem binary_representation_of_23 : 23 = 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end NUMINAMATH_GPT_binary_representation_of_23_l79_7995


namespace NUMINAMATH_GPT_fraction_product_l79_7970

theorem fraction_product :
  ((1: ℚ) / 2) * (3 / 5) * (7 / 11) = 21 / 110 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_product_l79_7970


namespace NUMINAMATH_GPT_carla_total_students_l79_7975

-- Defining the conditions
def students_in_restroom : Nat := 2
def absent_students : Nat := (3 * students_in_restroom) - 1
def total_desks : Nat := 4 * 6
def occupied_desks : Nat := total_desks * 2 / 3
def students_present : Nat := occupied_desks

-- The target is to prove the total number of students Carla teaches
theorem carla_total_students : students_in_restroom + absent_students + students_present = 23 := by
  sorry

end NUMINAMATH_GPT_carla_total_students_l79_7975


namespace NUMINAMATH_GPT_change_received_proof_l79_7965

-- Define the costs and amounts
def regular_ticket_cost : ℕ := 9
def children_ticket_discount : ℕ := 2
def amount_given : ℕ := 2 * 20

-- Define the number of people
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 3

-- Define the costs calculations
def child_ticket_cost := regular_ticket_cost - children_ticket_discount
def total_adults_cost := number_of_adults * regular_ticket_cost
def total_children_cost := number_of_children * child_ticket_cost
def total_cost := total_adults_cost + total_children_cost
def change_received := amount_given - total_cost

-- Lean statement to prove the change received
theorem change_received_proof : change_received = 1 := by
  sorry

end NUMINAMATH_GPT_change_received_proof_l79_7965


namespace NUMINAMATH_GPT_problem_statement_l79_7967

def f (x : ℝ) : ℝ := x^6 + x^2 + 7 * x

theorem problem_statement : f 3 - f (-3) = 42 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l79_7967


namespace NUMINAMATH_GPT_opposite_of_neg_3_is_3_l79_7997

theorem opposite_of_neg_3_is_3 : -(-3) = 3 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_3_is_3_l79_7997


namespace NUMINAMATH_GPT_greatest_savings_option2_l79_7956

-- Define the initial price
def initial_price : ℝ := 15000

-- Define the discounts for each option
def discounts_option1 : List ℝ := [0.75, 0.85, 0.95]
def discounts_option2 : List ℝ := [0.65, 0.90, 0.95]
def discounts_option3 : List ℝ := [0.70, 0.90, 0.90]

-- Define a function to compute the final price after successive discounts
def final_price (initial : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (λ acc d => acc * d) initial

-- Define the savings for each option
def savings_option1 : ℝ := initial_price - (final_price initial_price discounts_option1)
def savings_option2 : ℝ := initial_price - (final_price initial_price discounts_option2)
def savings_option3 : ℝ := initial_price - (final_price initial_price discounts_option3)

-- Formulate the proof
theorem greatest_savings_option2 :
  max (max savings_option1 savings_option2) savings_option3 = savings_option2 :=
by
  sorry

end NUMINAMATH_GPT_greatest_savings_option2_l79_7956


namespace NUMINAMATH_GPT_jacks_remaining_capacity_l79_7916

noncomputable def jacks_basket_full_capacity : ℕ := 12
noncomputable def jills_basket_full_capacity : ℕ := 2 * jacks_basket_full_capacity
noncomputable def jacks_current_apples (x : ℕ) : Prop := 3 * x = jills_basket_full_capacity

theorem jacks_remaining_capacity {x : ℕ} (hx : jacks_current_apples x) :
  jacks_basket_full_capacity - x = 4 :=
by sorry

end NUMINAMATH_GPT_jacks_remaining_capacity_l79_7916


namespace NUMINAMATH_GPT_B_completes_remaining_work_in_2_days_l79_7941

theorem B_completes_remaining_work_in_2_days 
  (A_work_rate : ℝ) (B_work_rate : ℝ) (total_work : ℝ) 
  (A_days_to_complete : A_work_rate = 1 / 2) 
  (B_days_to_complete : B_work_rate = 1 / 6) 
  (combined_work_1_day : A_work_rate + B_work_rate = 2 / 3) : 
  (total_work - (A_work_rate + B_work_rate)) / B_work_rate = 2 := 
by
  sorry

end NUMINAMATH_GPT_B_completes_remaining_work_in_2_days_l79_7941


namespace NUMINAMATH_GPT_four_integers_product_sum_l79_7937

theorem four_integers_product_sum (a b c d : ℕ) (h1 : a * b * c * d = 2002) (h2 : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end NUMINAMATH_GPT_four_integers_product_sum_l79_7937


namespace NUMINAMATH_GPT_gain_per_year_is_correct_l79_7911

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem gain_per_year_is_correct :
  let borrowed_amount := 7000
  let borrowed_rate := 0.04
  let borrowed_time := 2
  let borrowed_compound_freq := 1 -- annually
  
  let lent_amount := 7000
  let lent_rate := 0.06
  let lent_time := 2
  let lent_compound_freq := 2 -- semi-annually
  
  let amount_owed := compound_interest borrowed_amount borrowed_rate borrowed_compound_freq borrowed_time
  let amount_received := compound_interest lent_amount lent_rate lent_compound_freq lent_time
  let total_gain := amount_received - amount_owed
  let gain_per_year := total_gain / lent_time
  
  gain_per_year = 153.65 :=
by
  sorry

end NUMINAMATH_GPT_gain_per_year_is_correct_l79_7911


namespace NUMINAMATH_GPT_root_of_linear_equation_l79_7912

theorem root_of_linear_equation (b c : ℝ) (hb : b ≠ 0) :
  ∃ x : ℝ, 0 * x^2 + b * x + c = 0 → x = -c / b :=
by
  -- The proof steps would typically go here
  sorry

end NUMINAMATH_GPT_root_of_linear_equation_l79_7912


namespace NUMINAMATH_GPT_sum_of_divisors_of_11_squared_l79_7944

theorem sum_of_divisors_of_11_squared (a b c : ℕ) (h1 : a ∣ 11^2) (h2 : b ∣ 11^2) (h3 : c ∣ 11^2) (h4 : a * b * c = 11^2) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) :
  a + b + c = 23 :=
sorry

end NUMINAMATH_GPT_sum_of_divisors_of_11_squared_l79_7944


namespace NUMINAMATH_GPT_union_M_N_l79_7947

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_M_N : M ∪ N = {x | -1 < x ∧ x < 3} := 
by 
  sorry

end NUMINAMATH_GPT_union_M_N_l79_7947


namespace NUMINAMATH_GPT_factor_congruence_l79_7936

theorem factor_congruence (n : ℕ) (hn : n ≠ 0) :
  ∀ p : ℕ, p ∣ (2 * n)^(2^n) + 1 → p ≡ 1 [MOD 2^(n+1)] :=
sorry

end NUMINAMATH_GPT_factor_congruence_l79_7936


namespace NUMINAMATH_GPT_area_increase_of_square_garden_l79_7991

theorem area_increase_of_square_garden
  (length : ℝ) (width : ℝ)
  (h_length : length = 60)
  (h_width : width = 20) :
  let perimeter := 2 * (length + width)
  let side_length := perimeter / 4
  let initial_area := length * width
  let square_area := side_length ^ 2
  square_area - initial_area = 400 :=
by
  sorry

end NUMINAMATH_GPT_area_increase_of_square_garden_l79_7991


namespace NUMINAMATH_GPT_find_total_photos_l79_7913

noncomputable def total_photos (T : ℕ) (Paul Tim Tom : ℕ) : Prop :=
  Tim = T - 100 ∧ Paul = Tim + 10 ∧ Tom = 38 ∧ Tom + Tim + Paul = T

theorem find_total_photos : ∃ T, total_photos T (T - 90) (T - 100) 38 :=
sorry

end NUMINAMATH_GPT_find_total_photos_l79_7913


namespace NUMINAMATH_GPT_quadrilateral_perimeter_proof_l79_7918

noncomputable def perimeter_quadrilateral (AB BC CD AD : ℝ) : ℝ :=
  AB + BC + CD + AD

theorem quadrilateral_perimeter_proof
  (AB BC CD AD : ℝ)
  (h1 : AB = 15)
  (h2 : BC = 10)
  (h3 : CD = 6)
  (h4 : AB = AD)
  (h5 : AD = Real.sqrt 181)
  : perimeter_quadrilateral AB BC CD AD = 31 + Real.sqrt 181 := by
  unfold perimeter_quadrilateral
  rw [h1, h2, h3, h5]
  sorry

end NUMINAMATH_GPT_quadrilateral_perimeter_proof_l79_7918


namespace NUMINAMATH_GPT_integer_solutions_pxy_eq_xy_l79_7910

theorem integer_solutions_pxy_eq_xy (p : ℤ) (hp : Prime p) :
  ∃ x y : ℤ, p * (x + y) = x * y ∧ 
  ((x, y) = (2 * p, 2 * p) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (p + 1, p + p^2) ∨ 
  (x, y) = (p - 1, p - p^2) ∨ 
  (x, y) = (p + p^2, p + 1) ∨ 
  (x, y) = (p - p^2, p - 1)) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_pxy_eq_xy_l79_7910


namespace NUMINAMATH_GPT_hadley_total_distance_l79_7928

def distance_to_grocery := 2
def distance_to_pet_store := 2 - 1
def distance_back_home := 4 - 1

theorem hadley_total_distance : distance_to_grocery + distance_to_pet_store + distance_back_home = 6 :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_hadley_total_distance_l79_7928


namespace NUMINAMATH_GPT_josephine_milk_containers_l79_7979

theorem josephine_milk_containers :
  3 * 2 + 2 * 0.75 + 5 * x = 10 → x = 0.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_josephine_milk_containers_l79_7979


namespace NUMINAMATH_GPT_inequality_solution_set_l79_7972

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := 
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l79_7972


namespace NUMINAMATH_GPT_principal_amount_borrowed_l79_7978

theorem principal_amount_borrowed
  (R : ℝ) (T : ℝ) (SI : ℝ) (P : ℝ) 
  (hR : R = 12) 
  (hT : T = 20) 
  (hSI : SI = 2100) 
  (hFormula : SI = (P * R * T) / 100) : 
  P = 875 := 
by 
  -- Assuming the initial steps 
  sorry

end NUMINAMATH_GPT_principal_amount_borrowed_l79_7978


namespace NUMINAMATH_GPT_geometric_sequence_problem_l79_7957

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Given condition for the geometric sequence
variables {a : ℕ → ℝ} (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27)

-- Theorem to be proven
theorem geometric_sequence_problem (h_geometric : is_geometric_sequence a) (h_condition : a 4 * a 5 * a 6 = 27) : a 1 * a 9 = 9 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l79_7957


namespace NUMINAMATH_GPT_total_amount_to_be_divided_l79_7984

theorem total_amount_to_be_divided
  (k m x : ℕ)
  (h1 : 18 * k = x)
  (h2 : 20 * m = x)
  (h3 : 13 * m = 11 * k + 1400) :
  x = 36000 := 
sorry

end NUMINAMATH_GPT_total_amount_to_be_divided_l79_7984


namespace NUMINAMATH_GPT_choose_president_and_vice_president_l79_7938

theorem choose_president_and_vice_president :
  let total_members := 24
  let boys := 8
  let girls := 16
  let senior_members := 4
  let senior_boys := 2
  let senior_girls := 2
  let president_choices := senior_members
  let vice_president_choices_boy_pres := girls
  let vice_president_choices_girl_pres := boys - senior_boys
  let total_ways :=
    (senior_boys * vice_president_choices_boy_pres) + 
    (senior_girls * vice_president_choices_girl_pres)
  total_ways = 44 := 
by
  sorry

end NUMINAMATH_GPT_choose_president_and_vice_president_l79_7938


namespace NUMINAMATH_GPT_correct_calculation_result_l79_7977

theorem correct_calculation_result (x : ℤ) (h : x + 63 = 8) : x * 36 = -1980 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l79_7977


namespace NUMINAMATH_GPT_age_problem_contradiction_l79_7962

theorem age_problem_contradiction (C1 C2 : ℕ) (k : ℕ)
  (h1 : 15 = k * (C1 + C2))
  (h2 : 20 = 2 * (C1 + 5 + C2 + 5)) : false :=
by
  sorry

end NUMINAMATH_GPT_age_problem_contradiction_l79_7962


namespace NUMINAMATH_GPT_max_edges_in_8_points_graph_no_square_l79_7976

open Finset

-- Define what a graph is and the properties needed for the problem
structure Graph (V : Type*) :=
  (edges : Finset (V × V))
  (sym : ∀ {x y : V}, (x, y) ∈ edges ↔ (y, x) ∈ edges)
  (irrefl : ∀ {x : V}, ¬ (x, x) ∈ edges)

-- Define the conditions of the problem
def no_square {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c d : V), 
    (a, b) ∈ G.edges → (b, c) ∈ G.edges → (c, d) ∈ G.edges → (d, a) ∈ G.edges →
    (a, c) ∈ G.edges → (b, d) ∈ G.edges → False

-- Define 8 vertices
inductive Vertices
| A | B | C | D | E | F | G | H

-- Define the number of edges
noncomputable def max_edges_no_square : ℕ :=
  11

-- Define the final theorem
theorem max_edges_in_8_points_graph_no_square :
  ∃ (G : Graph Vertices), 
    no_square G ∧ (G.edges.card = max_edges_no_square) :=
sorry

end NUMINAMATH_GPT_max_edges_in_8_points_graph_no_square_l79_7976


namespace NUMINAMATH_GPT_saturday_earnings_l79_7935

-- Lean 4 Statement

theorem saturday_earnings 
  (S Wednesday_earnings : ℝ)
  (h1 : S + Wednesday_earnings = 5182.50)
  (h2 : Wednesday_earnings = S - 142.50) 
  : S = 2662.50 := 
by
  sorry

end NUMINAMATH_GPT_saturday_earnings_l79_7935


namespace NUMINAMATH_GPT_large_cube_side_length_painted_blue_l79_7909

   theorem large_cube_side_length_painted_blue (n : ℕ) (h : 6 * n^2 = (1 / 3) * 6 * n^3) : n = 3 :=
   by
     sorry
   
end NUMINAMATH_GPT_large_cube_side_length_painted_blue_l79_7909


namespace NUMINAMATH_GPT_slope_divides_polygon_area_l79_7953

structure Point where
  x : ℝ
  y : ℝ

noncomputable def polygon_vertices : List Point :=
  [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩]

-- Define the area calculation and conditions needed 
noncomputable def area_of_polygon (vertices : List Point) : ℝ :=
  -- Assuming here that a function exists to calculate the area given the vertices
  sorry

def line_through_origin (slope : ℝ) (x : ℝ) : Point :=
  ⟨x, slope * x⟩

theorem slope_divides_polygon_area :
  let line := line_through_origin (2 / 7)
  ∀ x : ℝ, ∃ (G : Point), 
  polygon_vertices = [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩] →
  area_of_polygon polygon_vertices / 2 = 
  area_of_polygon [⟨0, 0⟩, line x, G] :=
sorry

end NUMINAMATH_GPT_slope_divides_polygon_area_l79_7953


namespace NUMINAMATH_GPT_cheesecake_factory_working_days_l79_7919

-- Define the savings rates
def robby_saves := 2 / 5
def jaylen_saves := 3 / 5
def miranda_saves := 1 / 2

-- Define their hourly rate and daily working hours
def hourly_rate := 10 -- dollars per hour
def work_hours_per_day := 10 -- hours per day

-- Define their combined savings after four weeks and the combined savings target
def four_weeks := 4 * 7
def combined_savings_target := 3000 -- dollars

-- Question: Prove that the number of days they work per week is 7
theorem cheesecake_factory_working_days (d : ℕ) (h : d * 400 = combined_savings_target / 4) : d = 7 := sorry

end NUMINAMATH_GPT_cheesecake_factory_working_days_l79_7919


namespace NUMINAMATH_GPT_unique_function_solution_l79_7904

variable (f : ℝ → ℝ)

theorem unique_function_solution :
  (∀ x y : ℝ, f (f x - y^2) = f x ^ 2 - 2 * f x * y^2 + f (f y))
  → (∀ x : ℝ, f x = x^2) :=
by
  sorry

end NUMINAMATH_GPT_unique_function_solution_l79_7904


namespace NUMINAMATH_GPT_smallest_positive_period_and_axis_of_symmetry_l79_7959

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 4)

theorem smallest_positive_period_and_axis_of_symmetry :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∃ k : ℤ, ∀ x, 2 * x - Real.pi / 4 = k * Real.pi + Real.pi / 2 → x = k * Real.pi / 2 - Real.pi / 8) :=
  sorry

end NUMINAMATH_GPT_smallest_positive_period_and_axis_of_symmetry_l79_7959


namespace NUMINAMATH_GPT_solve_for_a_and_b_l79_7924

theorem solve_for_a_and_b (a b : ℤ) (h1 : 5 + a = 6 - b) (h2 : 6 + b = 9 + a) : 5 - a = 6 := 
sorry

end NUMINAMATH_GPT_solve_for_a_and_b_l79_7924


namespace NUMINAMATH_GPT_original_number_of_people_l79_7986

-- Defining the conditions
variable (n : ℕ) -- number of people originally
variable (total_cost : ℕ := 375)
variable (equal_cost_split : n > 0 ∧ total_cost = 375) -- total cost is $375 and n > 0
variable (cost_condition : 375 / n + 50 = 375 / 5)

-- The proof statement
theorem original_number_of_people (h1 : total_cost = 375) (h2 : 375 / n + 50 = 375 / 5) : n = 15 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_people_l79_7986


namespace NUMINAMATH_GPT_isosceles_triangle_height_l79_7993

theorem isosceles_triangle_height (s h : ℝ) (eq_areas : (2 * s * s) = (1/2 * s * h)) : h = 4 * s :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_height_l79_7993


namespace NUMINAMATH_GPT_vector_subtraction_scalar_mul_l79_7952

theorem vector_subtraction_scalar_mul :
  let v₁ := (3, -8) 
  let scalar := -5 
  let v₂ := (4, 6)
  v₁.1 - scalar * v₂.1 = 23 ∧ v₁.2 - scalar * v₂.2 = 22 := by
    sorry

end NUMINAMATH_GPT_vector_subtraction_scalar_mul_l79_7952


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l79_7915

theorem sum_of_other_endpoint_coordinates (x y : ℤ) :
  (7 + x) / 2 = 5 ∧ (4 + y) / 2 = -8 → x + y = -17 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l79_7915


namespace NUMINAMATH_GPT_systematic_sampling_correct_l79_7943

-- Definitions for the conditions
def total_products := 60
def group_count := 5
def products_per_group := total_products / group_count

-- systematic sampling condition: numbers are in increments of products_per_group
def systematic_sample (start : ℕ) (count : ℕ) : List ℕ := List.range' start products_per_group count

-- Given sequences
def A : List ℕ := [5, 10, 15, 20, 25]
def B : List ℕ := [5, 12, 31, 39, 57]
def C : List ℕ := [5, 17, 29, 41, 53]
def D : List ℕ := [5, 15, 25, 35, 45]

-- Correct solution defined
def correct_solution := [5, 17, 29, 41, 53]

-- Problem Statement
theorem systematic_sampling_correct :
  systematic_sample 5 group_count = correct_solution :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_correct_l79_7943


namespace NUMINAMATH_GPT_frog_jump_problem_l79_7951

theorem frog_jump_problem (A B C : ℝ) (PA PB PC : ℝ) 
  (H1: PA' = (PB + PC) / 2)
  (H2: jump_distance_B = 60)
  (H3: jump_distance_B = 2 * abs ((PB - (PB + PC) / 2))) :
  third_jump_distance = 30 := sorry

end NUMINAMATH_GPT_frog_jump_problem_l79_7951


namespace NUMINAMATH_GPT_digits_product_l79_7945

-- Define the conditions
variables (A B : ℕ)

-- Define the main problem statement using the conditions and expected answer
theorem digits_product (h1 : A + B = 12) (h2 : (10 * A + B) % 3 = 0) : A * B = 35 := 
by
  sorry

end NUMINAMATH_GPT_digits_product_l79_7945


namespace NUMINAMATH_GPT_largest_divisor_8_l79_7923

theorem largest_divisor_8 (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) : 
  8 ∣ (p^2 - q^2 + 2*p - 2*q) := 
sorry

end NUMINAMATH_GPT_largest_divisor_8_l79_7923


namespace NUMINAMATH_GPT_triangle_inequality_a2_a3_a4_l79_7966

variables {a1 a2 a3 a4 d : ℝ}

def is_arithmetic_sequence (a1 a2 a3 a4 : ℝ) (d : ℝ) : Prop :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℝ) : Prop :=
  0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4

theorem triangle_inequality_a2_a3_a4 (h1: positive_terms a1 a2 a3 a4)
  (h2: is_arithmetic_sequence a1 a2 a3 a4 d) (h3: d > 0) :
  (a2 + a3 > a4) ∧ (a2 + a4 > a3) ∧ (a3 + a4 > a2) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_a2_a3_a4_l79_7966


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l79_7980

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l79_7980


namespace NUMINAMATH_GPT_elmer_saves_14_3_percent_l79_7960

-- Define the problem statement conditions and goal
theorem elmer_saves_14_3_percent (old_efficiency new_efficiency : ℝ) (old_cost new_cost : ℝ) :
  new_efficiency = 1.75 * old_efficiency →
  new_cost = 1.5 * old_cost →
  (500 / old_efficiency * old_cost - 500 / new_efficiency * new_cost) / (500 / old_efficiency * old_cost) * 100 = 14.3 := by
  -- sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_elmer_saves_14_3_percent_l79_7960


namespace NUMINAMATH_GPT_nested_fraction_eval_l79_7961

theorem nested_fraction_eval : (1 / (1 + (1 / (2 + (1 / (1 + (1 / 4))))))) = (14 / 19) :=
by
  sorry

end NUMINAMATH_GPT_nested_fraction_eval_l79_7961


namespace NUMINAMATH_GPT_minimum_value_of_a_l79_7973

theorem minimum_value_of_a :
  (∀ x : ℝ, x > 0 → (a : ℝ) * x * Real.exp x - x - Real.log x ≥ 0) → a ≥ 1 / Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_a_l79_7973


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l79_7992

theorem arithmetic_sequence_sum :
  ∀ {a : ℕ → ℕ} {S : ℕ → ℕ},
  (∀ n, a (n + 1) - a n = a 1 - a 0) →
  (∀ n, S n = n * (a 1 + a n) / 2) →
  a 1 + a 9 = 18 →
  a 4 = 7 →
  S 8 = 64 :=
by
  intros a S h_arith_seq h_sum_formula h_a1_a9 h_a4
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l79_7992


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l79_7930

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∃ A, ∀ n : ℕ, a n = A * (q ^ (n - 1))

theorem arithmetic_sequence_problem
  (q : ℝ) 
  (h1 : q > 1)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h_seq : is_arithmetic_sequence a q) : 
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l79_7930


namespace NUMINAMATH_GPT_percentage_of_men_l79_7934

variables {M W : ℝ}
variables (h1 : M + W = 100)
variables (h2 : 0.20 * M + 0.40 * W = 34)

theorem percentage_of_men :
  M = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_l79_7934


namespace NUMINAMATH_GPT_polygon_sides_l79_7927

theorem polygon_sides (n : ℕ) (hn : 3 ≤ n) (H : (n * (n - 3)) / 2 = 15) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l79_7927


namespace NUMINAMATH_GPT_volume_of_soil_extracted_l79_7907

-- Definition of the conditions
def Length : ℝ := 20
def Width : ℝ := 10
def Depth : ℝ := 8

-- Statement of the proof problem
theorem volume_of_soil_extracted : Length * Width * Depth = 1600 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_volume_of_soil_extracted_l79_7907


namespace NUMINAMATH_GPT_sufficient_condition_l79_7903

theorem sufficient_condition (a b : ℝ) (h : a > b ∧ b > 0) : a + a^2 > b + b^2 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l79_7903


namespace NUMINAMATH_GPT_arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l79_7963

theorem arrangement_two_rows :
  ∃ (ways : ℕ), ways = 5040 := by
  sorry

theorem arrangement_no_head_tail (A : ℕ):
  ∃ (ways : ℕ), ways = 3600 := by
  sorry

theorem arrangement_girls_together :
  ∃ (ways : ℕ), ways = 576 := by
  sorry

theorem arrangement_no_boys_next :
  ∃ (ways : ℕ), ways = 1440 := by
  sorry

end NUMINAMATH_GPT_arrangement_two_rows_arrangement_no_head_tail_arrangement_girls_together_arrangement_no_boys_next_l79_7963


namespace NUMINAMATH_GPT_maximum_reflections_l79_7964

theorem maximum_reflections (θ : ℕ) (h : θ = 10) (max_angle : ℕ) (h_max : max_angle = 180) : 
∃ n : ℕ, n ≤ max_angle / θ ∧ n = 18 := by
  sorry

end NUMINAMATH_GPT_maximum_reflections_l79_7964


namespace NUMINAMATH_GPT_moles_CO2_is_one_l79_7914

noncomputable def moles_CO2_formed (moles_HNO3 moles_NaHCO3 : ℕ) : ℕ :=
  if moles_HNO3 = 1 ∧ moles_NaHCO3 = 1 then 1 else 0

theorem moles_CO2_is_one :
  moles_CO2_formed 1 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_CO2_is_one_l79_7914


namespace NUMINAMATH_GPT_equation_in_terms_of_y_l79_7999

theorem equation_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : y = 5 - 2 * x :=
sorry

end NUMINAMATH_GPT_equation_in_terms_of_y_l79_7999


namespace NUMINAMATH_GPT_jack_needs_more_money_l79_7925

-- Definitions based on given conditions
def cost_per_sock_pair : ℝ := 9.50
def num_sock_pairs : ℕ := 2
def cost_per_shoe : ℝ := 92
def jack_money : ℝ := 40

-- Theorem statement
theorem jack_needs_more_money (cost_per_sock_pair num_sock_pairs cost_per_shoe jack_money : ℝ) : 
  ((cost_per_sock_pair * num_sock_pairs) + cost_per_shoe) - jack_money = 71 := by
  sorry

end NUMINAMATH_GPT_jack_needs_more_money_l79_7925


namespace NUMINAMATH_GPT_plan_y_cheaper_than_plan_x_l79_7905

def cost_plan_x (z : ℕ) : ℕ := 15 * z

def cost_plan_y (z : ℕ) : ℕ :=
  if z > 500 then 3000 + 7 * z - 1000 else 3000 + 7 * z

theorem plan_y_cheaper_than_plan_x (z : ℕ) (h : z > 500) : cost_plan_y z < cost_plan_x z :=
by
  sorry

end NUMINAMATH_GPT_plan_y_cheaper_than_plan_x_l79_7905
