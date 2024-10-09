import Mathlib

namespace sum_of_tangent_points_l1220_122068

noncomputable def f (x : ℝ) : ℝ := 
  max (max (-7 * x - 19) (3 * x - 1)) (5 * x + 3)

theorem sum_of_tangent_points :
  ∃ x4 x5 x6 : ℝ, 
  (∃ q : ℝ → ℝ, 
    (∀ x, q x = f x ∨ (q x - (-7 * x - 19)) = b * (x - x4)^2
    ∨ (q x - (3 * x - 1)) = b * (x - x5)^2 
    ∨ (q x - (5 * x + 3)) = b * (x - x6)^2)) ∧
  x4 + x5 + x6 = -3.2 :=
sorry

end sum_of_tangent_points_l1220_122068


namespace perpendicular_lines_foot_l1220_122053

variables (a b c : ℝ)

theorem perpendicular_lines_foot (h1 : a * -2/20 = -1)
  (h2_foot_l1 : a * 1 + 4 * c - 2 = 0)
  (h3_foot_l2 : 2 * 1 - 5 * c + b = 0) :
  a + b + c = -4 :=
sorry

end perpendicular_lines_foot_l1220_122053


namespace equal_divide_remaining_amount_all_girls_l1220_122082

theorem equal_divide_remaining_amount_all_girls 
    (debt : ℕ) (savings_lulu : ℕ) (savings_nora : ℕ) (savings_tamara : ℕ)
    (total_savings : ℕ) (remaining_amount : ℕ)
    (each_girl_gets : ℕ)
    (Lulu_saved : savings_lulu = 6)
    (Nora_saved_multiple_of_Lulu : savings_nora = 5 * savings_lulu)
    (Nora_saved_multiple_of_Tamara : savings_nora = 3 * savings_tamara)
    (total_saved_calculated : total_savings = savings_nora + savings_tamara + savings_lulu)
    (debt_value : debt = 40)
    (remaining_calculated : remaining_amount = total_savings - debt)
    (division_among_girls : each_girl_gets = remaining_amount / 3) :
  each_girl_gets = 2 := 
sorry

end equal_divide_remaining_amount_all_girls_l1220_122082


namespace range_of_a_l1220_122022

theorem range_of_a (a : ℝ) : (∀ x > 1, x^2 ≥ a) ↔ (a ≤ 1) :=
by {
  sorry
}

end range_of_a_l1220_122022


namespace solution_l1220_122054

noncomputable def x : ℕ := 13

theorem solution : (3 * x) - (36 - x) = 16 := by
  sorry

end solution_l1220_122054


namespace product_of_two_numbers_l1220_122072

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 11) (h2 : x^2 + y^2 = 205) : x * y = 42 :=
by
  sorry

end product_of_two_numbers_l1220_122072


namespace bees_leg_count_l1220_122001

-- Define the number of legs per bee
def legsPerBee : Nat := 6

-- Define the number of bees
def numberOfBees : Nat := 8

-- Calculate the total number of legs for 8 bees
def totalLegsForEightBees : Nat := 48

-- The theorem statement
theorem bees_leg_count : (legsPerBee * numberOfBees) = totalLegsForEightBees := 
by
  -- Skipping the proof by using sorry
  sorry

end bees_leg_count_l1220_122001


namespace roots_of_transformed_quadratic_l1220_122002

theorem roots_of_transformed_quadratic (a b c d x : ℝ) :
  (∀ x, (x - a) * (x - b) - x = 0 → x = c ∨ x = d) →
  (x - c) * (x - d) + x = 0 → x = a ∨ x = b :=
by
  sorry

end roots_of_transformed_quadratic_l1220_122002


namespace number_of_people_who_purchased_only_book_A_l1220_122040

-- Define the conditions and the problem
theorem number_of_people_who_purchased_only_book_A 
    (total_A : ℕ) (total_B : ℕ) (both_AB : ℕ) (only_B : ℕ) :
    (total_A = 2 * total_B) → 
    (both_AB = 500) → 
    (both_AB = 2 * only_B) → 
    (total_B = only_B + both_AB) → 
    (total_A - both_AB = 1000) :=
by
  sorry

end number_of_people_who_purchased_only_book_A_l1220_122040


namespace seashells_problem_l1220_122033

theorem seashells_problem
  (F : ℕ)
  (h : (150 - F) / 2 = 55) :
  F = 40 :=
  sorry

end seashells_problem_l1220_122033


namespace factorial_expression_calculation_l1220_122021

theorem factorial_expression_calculation :
  7 * (Nat.factorial 7) + 5 * (Nat.factorial 6) - 6 * (Nat.factorial 5) = 7920 :=
by
  sorry

end factorial_expression_calculation_l1220_122021


namespace calculate_expression_l1220_122038

theorem calculate_expression : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := 
by 
  /-
  In Lean, we typically perform arithmetic simplifications step by step;
  however, for the purpose of this example, only stating the goal:
  -/
  sorry

end calculate_expression_l1220_122038


namespace candy_store_revenue_l1220_122017

def fudge_revenue : ℝ := 20 * 2.50
def truffles_revenue : ℝ := 5 * 12 * 1.50
def pretzels_revenue : ℝ := 3 * 12 * 2.00
def total_revenue : ℝ := fudge_revenue + truffles_revenue + pretzels_revenue

theorem candy_store_revenue :
  total_revenue = 212.00 :=
sorry

end candy_store_revenue_l1220_122017


namespace inequality_proof_l1220_122030

-- Conditions: a > b and c > d
variables {a b c d : ℝ}

-- The main statement to prove: d - a < c - b with given conditions
theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := 
sorry

end inequality_proof_l1220_122030


namespace adult_tickets_l1220_122093

theorem adult_tickets (A C : ℕ) (h1 : A + C = 130) (h2 : 12 * A + 4 * C = 840) : A = 40 :=
by {
  -- Proof omitted
  sorry
}

end adult_tickets_l1220_122093


namespace final_position_total_distance_l1220_122092

-- Define the movements as a list
def movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

-- Prove that the final position of the turtle is 5 meters north of the starting point
theorem final_position (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum movements = 5 :=
by
  rw [h]
  sorry

-- Prove that the total distance crawled by the turtle is 47 meters
theorem total_distance (movements : List Int) (h : movements = [-8, 7, -3, 9, -6, -4, 10]) : List.sum (List.map Int.natAbs movements) = 47 :=
by
  rw [h]
  sorry

end final_position_total_distance_l1220_122092


namespace find_multiplier_l1220_122075

theorem find_multiplier (x : ℕ) (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 :=
sorry

end find_multiplier_l1220_122075


namespace option_b_not_valid_l1220_122083

theorem option_b_not_valid (a b c d : ℝ) (h_arith_seq : b - a = d ∧ c - b = d ∧ d ≠ 0) : 
  a^3 * b + b^3 * c + c^3 * a < a^4 + b^4 + c^4 :=
by sorry

end option_b_not_valid_l1220_122083


namespace chord_length_l1220_122003

theorem chord_length (r d AB : ℝ) (hr : r = 5) (hd : d = 4) : AB = 6 :=
by
  -- Given
  -- r = radius = 5
  -- d = distance from center to chord = 4

  -- prove AB = 6
  sorry

end chord_length_l1220_122003


namespace average_sale_over_six_months_l1220_122045

theorem average_sale_over_six_months : 
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  average_sale = 3500 :=
by
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  show average_sale = 3500
  sorry

end average_sale_over_six_months_l1220_122045


namespace find_f_l1220_122099

theorem find_f (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : ∀ x : ℝ, f x = x + 1 :=
by
  sorry

end find_f_l1220_122099


namespace dog_total_bones_l1220_122089

-- Define the number of original bones and dug up bones as constants
def original_bones : ℕ := 493
def dug_up_bones : ℕ := 367

-- Define the total bones the dog has now
def total_bones : ℕ := original_bones + dug_up_bones

-- State and prove the theorem
theorem dog_total_bones : total_bones = 860 := by
  -- placeholder for the proof
  sorry

end dog_total_bones_l1220_122089


namespace solution_set_of_inequality_l1220_122067

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Problem conditions
theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_neg : ∀ x, x < 0 → f x = x + 2) :
  { x : ℝ | 2 * f x - 1 < 0 } = { x : ℝ | x < -3/2 ∨ (0 ≤ x ∧ x < 5/2) } :=
by
  sorry

end solution_set_of_inequality_l1220_122067


namespace line_circle_no_intersect_l1220_122037

theorem line_circle_no_intersect :
  ∀ (x y : ℝ), (3 * x + 4 * y = 12) → (x^2 + y^2 = 4) → false :=
by
  -- use the given equations to demonstrate the lack of intersection points
  sorry

end line_circle_no_intersect_l1220_122037


namespace simplify_decimal_l1220_122062

theorem simplify_decimal : (3416 / 1000 : ℚ) = 427 / 125 := by
  sorry

end simplify_decimal_l1220_122062


namespace no_natural_number_n_exists_l1220_122056

theorem no_natural_number_n_exists :
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end no_natural_number_n_exists_l1220_122056


namespace sum_of_squares_l1220_122044

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end sum_of_squares_l1220_122044


namespace initial_number_is_12_l1220_122005

theorem initial_number_is_12 {x : ℤ} (h : ∃ k : ℤ, x + 17 = 29 * k) : x = 12 :=
by
  sorry

end initial_number_is_12_l1220_122005


namespace distance_to_water_source_l1220_122077

theorem distance_to_water_source (d : ℝ) :
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 5)) → 7 < d ∧ d < 8 :=
by
  sorry

end distance_to_water_source_l1220_122077


namespace spell_casting_contest_orders_l1220_122020

-- Definition for factorial
def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Theorem statement: number of ways to order 4 contestants is 4!
theorem spell_casting_contest_orders : factorial 4 = 24 := by
  sorry

end spell_casting_contest_orders_l1220_122020


namespace new_percentage_of_water_l1220_122073

noncomputable def initial_weight : ℝ := 100
noncomputable def initial_percentage_water : ℝ := 99 / 100
noncomputable def initial_weight_water : ℝ := initial_weight * initial_percentage_water
noncomputable def initial_weight_non_water : ℝ := initial_weight - initial_weight_water
noncomputable def new_weight : ℝ := 25

theorem new_percentage_of_water :
  ((new_weight - initial_weight_non_water) / new_weight) * 100 = 96 :=
by
  sorry

end new_percentage_of_water_l1220_122073


namespace space_taken_by_files_l1220_122016

-- Definitions/Conditions
def total_space : ℕ := 28
def space_left : ℕ := 2

-- Statement of the theorem
theorem space_taken_by_files : total_space - space_left = 26 := by sorry

end space_taken_by_files_l1220_122016


namespace triangle_trig_problems_l1220_122018

open Real

-- Define the main theorem
theorem triangle_trig_problems (A B C a b c : ℝ) (h1: b ≠ 0) 
  (h2: cos A - 2 * cos C ≠ 0) 
  (h3 : (cos A - 2 * cos C) / cos B = (2 * c - a) / b) 
  (h4 : cos B = 1/4)
  (h5 : b = 2) :
  (sin C / sin A = 2) ∧ 
  (2 * a * c * sqrt 15 / 4 = sqrt 15 / 4) :=
by 
  sorry

end triangle_trig_problems_l1220_122018


namespace decimal_to_fraction_sum_l1220_122013

def recurring_decimal_fraction_sum : Prop :=
  ∃ (a b : ℕ), b ≠ 0 ∧ gcd a b = 1 ∧ (a / b : ℚ) = (0.345345345 : ℚ) ∧ a + b = 226

theorem decimal_to_fraction_sum :
  recurring_decimal_fraction_sum :=
sorry

end decimal_to_fraction_sum_l1220_122013


namespace arithmetic_sequence_problem_l1220_122046

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Use the given specific conditions
theorem arithmetic_sequence_problem 
  (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 2 * a 3 = 21) : 
  a 1 * a 4 = -11 :=
sorry

end arithmetic_sequence_problem_l1220_122046


namespace smallest_palindrome_not_five_digit_l1220_122086

theorem smallest_palindrome_not_five_digit (n : ℕ) (h : 100 ≤ n ∧ n ≤ 999 ∧ n % 10 = n / 100 ∧ n / 10 % 10 = n / 100 ∧ 103 * n < 10000) :
  n = 707 := by
sorry

end smallest_palindrome_not_five_digit_l1220_122086


namespace circle_radius_l1220_122008

theorem circle_radius (A : ℝ) (k : ℝ) (r : ℝ) (h : A = k * π * r^2) (hA : A = 225 * π) (hk : k = 4) : 
  r = 7.5 :=
by 
  sorry

end circle_radius_l1220_122008


namespace manufacturer_l1220_122064

-- Let x be the manufacturer's suggested retail price
variable (x : ℝ)

-- Regular discount range from 10% to 30%
def regular_discount (d : ℝ) : Prop := d >= 0.10 ∧ d <= 0.30

-- Additional discount during sale 
def additional_discount : ℝ := 0.20

-- The final discounted price is $16.80
def final_price (x : ℝ) : Prop := ∃ d, regular_discount d ∧ 0.80 * ((1 - d) * x) = 16.80

theorem manufacturer's_suggested_retail_price :
  final_price x → x = 30 := by
  sorry

end manufacturer_l1220_122064


namespace find_number_l1220_122028

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l1220_122028


namespace lucy_picked_more_l1220_122061

variable (Mary Peter Lucy : ℕ)
variable (Mary_amt Peter_amt Lucy_amt : ℕ)

-- Conditions
def mary_amount : Mary_amt = 12 := sorry
def twice_as_peter : Mary_amt = 2 * Peter_amt := sorry
def total_picked : Mary_amt + Peter_amt + Lucy_amt = 26 := sorry

-- Statement to Prove
theorem lucy_picked_more (h1: Mary_amt = 12) (h2: Mary_amt = 2 * Peter_amt) (h3: Mary_amt + Peter_amt + Lucy_amt = 26) :
  Lucy_amt - Peter_amt = 2 := 
sorry

end lucy_picked_more_l1220_122061


namespace weight_of_new_student_l1220_122023

theorem weight_of_new_student (W : ℝ) (x : ℝ) (h1 : 5 * W - 92 + x = 5 * (W - 4)) : x = 72 :=
sorry

end weight_of_new_student_l1220_122023


namespace fraction_goldfish_preference_l1220_122034

theorem fraction_goldfish_preference
  (students_per_class : ℕ)
  (students_prefer_golfish_miss_johnson : ℕ)
  (students_prefer_golfish_ms_henderson : ℕ)
  (students_prefer_goldfish_total : ℕ)
  (miss_johnson_fraction : ℚ)
  (ms_henderson_fraction : ℚ)
  (total_students_prefer_goldfish_feldstein : ℕ)
  (feldstein_fraction : ℚ) :
  miss_johnson_fraction = 1/6 ∧
  ms_henderson_fraction = 1/5 ∧
  students_per_class = 30 ∧
  students_prefer_golfish_miss_johnson = miss_johnson_fraction * students_per_class ∧
  students_prefer_golfish_ms_henderson = ms_henderson_fraction * students_per_class ∧
  students_prefer_goldfish_total = 31 ∧
  students_prefer_goldfish_total = students_prefer_golfish_miss_johnson + students_prefer_golfish_ms_henderson + total_students_prefer_goldfish_feldstein ∧
  feldstein_fraction * students_per_class = total_students_prefer_goldfish_feldstein
  →
  feldstein_fraction = 2 / 3 :=
by 
  sorry

end fraction_goldfish_preference_l1220_122034


namespace min_x2_plus_y2_l1220_122042

theorem min_x2_plus_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_x2_plus_y2_l1220_122042


namespace solve_fraction_eq_zero_l1220_122015

theorem solve_fraction_eq_zero (x : ℝ) (h : x ≠ 0) : 
  (x^2 - 4*x + 3) / (5*x) = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end solve_fraction_eq_zero_l1220_122015


namespace year_2024_AD_representation_l1220_122098

def year_representation (y: Int) : Int :=
  if y > 0 then y else -y

theorem year_2024_AD_representation : year_representation 2024 = 2024 :=
by sorry

end year_2024_AD_representation_l1220_122098


namespace proof_of_expression_l1220_122085

theorem proof_of_expression (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 :=
by {
  sorry
}

end proof_of_expression_l1220_122085


namespace total_length_of_figure_2_segments_l1220_122060

-- Definitions based on conditions
def rectangle_length : ℕ := 10
def rectangle_breadth : ℕ := 6
def square_side : ℕ := 4
def interior_segment : ℕ := rectangle_breadth / 2

-- Summing up the lengths of segments in Figure 2
def total_length_of_segments : ℕ :=
  square_side + 2 * rectangle_length + interior_segment

-- Mathematical proof problem statement
theorem total_length_of_figure_2_segments :
  total_length_of_segments = 27 :=
sorry

end total_length_of_figure_2_segments_l1220_122060


namespace quadratic_equal_real_roots_l1220_122059

theorem quadratic_equal_real_roots (m : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + m = 1 ∧ 
                              (∀ y : ℝ, y ≠ x → y^2 - 4 * y + m ≠ 1)) : m = 5 :=
by sorry

end quadratic_equal_real_roots_l1220_122059


namespace initial_number_of_kids_l1220_122000

theorem initial_number_of_kids (joined kids_total initial : ℕ) (h1 : joined = 22) (h2 : kids_total = 36) (h3 : kids_total = initial + joined) : initial = 14 :=
by 
  -- Proof goes here
  sorry

end initial_number_of_kids_l1220_122000


namespace increase_75_by_150_percent_l1220_122081

noncomputable def original_number : Real := 75
noncomputable def percentage_increase : Real := 1.5
noncomputable def increase_amount : Real := original_number * percentage_increase
noncomputable def result : Real := original_number + increase_amount

theorem increase_75_by_150_percent : result = 187.5 := by
  sorry

end increase_75_by_150_percent_l1220_122081


namespace poles_needed_to_enclose_plot_l1220_122087

-- Defining the lengths of the sides
def side1 : ℕ := 15
def side2 : ℕ := 22
def side3 : ℕ := 40
def side4 : ℕ := 30
def side5 : ℕ := 18

-- Defining the distance between poles
def dist_first_three_sides : ℕ := 4
def dist_last_two_sides : ℕ := 5

-- Defining the function to calculate required poles for a side
def calculate_poles (length : ℕ) (distance : ℕ) : ℕ :=
  (length / distance) + 1

-- Total poles needed before adjustment
def total_poles_before_adjustment : ℕ :=
  calculate_poles side1 dist_first_three_sides +
  calculate_poles side2 dist_first_three_sides +
  calculate_poles side3 dist_first_three_sides +
  calculate_poles side4 dist_last_two_sides +
  calculate_poles side5 dist_last_two_sides

-- Adjustment for shared poles at corners
def total_poles : ℕ :=
  total_poles_before_adjustment - 5

-- The theorem to prove
theorem poles_needed_to_enclose_plot : total_poles = 29 := by
  sorry

end poles_needed_to_enclose_plot_l1220_122087


namespace car_catches_up_in_6_hours_l1220_122088

-- Conditions
def speed_truck := 40 -- km/h
def speed_car_initial := 50 -- km/h
def speed_car_increment := 5 -- km/h
def distance_between := 135 -- km

-- Solution: car catches up in 6 hours
theorem car_catches_up_in_6_hours : 
  ∃ n : ℕ, n = 6 ∧ (n * speed_truck + distance_between) ≤ (n * speed_car_initial + (n * (n - 1) / 2 * speed_car_increment)) := 
by
  sorry

end car_catches_up_in_6_hours_l1220_122088


namespace tuition_fee_l1220_122025

theorem tuition_fee (R T : ℝ) (h1 : T + R = 2584) (h2 : T = R + 704) : T = 1644 := by sorry

end tuition_fee_l1220_122025


namespace total_value_of_item_l1220_122063

variable (V : ℝ) -- Total value of the item

def import_tax (V : ℝ) := 0.07 * (V - 1000) -- Definition of import tax

theorem total_value_of_item
  (htax_paid : import_tax V = 112.70) :
  V = 2610 := 
by
  sorry

end total_value_of_item_l1220_122063


namespace find_g7_l1220_122012

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_value : g 6 = 7

theorem find_g7 : g 7 = 49 / 6 := by
  sorry

end find_g7_l1220_122012


namespace initial_overs_played_l1220_122084

-- Define the conditions
def initial_run_rate : ℝ := 6.2
def remaining_overs : ℝ := 40
def remaining_run_rate : ℝ := 5.5
def target_runs : ℝ := 282

-- Define what we seek to prove
theorem initial_overs_played :
  ∃ x : ℝ, (6.2 * x) + (5.5 * 40) = 282 ∧ x = 10 :=
by
  sorry

end initial_overs_played_l1220_122084


namespace remainder_when_divided_by_x_minus_2_l1220_122039

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 + 12 * x^2 + 20 * x - 10

-- State the theorem about the remainder when f(x) is divided by x-2
theorem remainder_when_divided_by_x_minus_2 : f 2 = 30 := by
  -- This is where the proof would go, but we use sorry to skip the proof.
  sorry

end remainder_when_divided_by_x_minus_2_l1220_122039


namespace percentage_water_fresh_fruit_l1220_122091

-- Definitions of the conditions
def weight_dried_fruit : ℝ := 12
def water_content_dried_fruit : ℝ := 0.15
def weight_fresh_fruit : ℝ := 101.99999999999999

-- Derived definitions based on the conditions
def weight_non_water_dried_fruit : ℝ := weight_dried_fruit - (water_content_dried_fruit * weight_dried_fruit)
def weight_non_water_fresh_fruit : ℝ := weight_non_water_dried_fruit
def weight_water_fresh_fruit : ℝ := weight_fresh_fruit - weight_non_water_fresh_fruit

-- Proof statement
theorem percentage_water_fresh_fruit :
  (weight_water_fresh_fruit / weight_fresh_fruit) * 100 = 90 :=
sorry

end percentage_water_fresh_fruit_l1220_122091


namespace solve_quadratic_inequality_l1220_122095

theorem solve_quadratic_inequality (a : ℝ) (x : ℝ) :
  (x^2 - a * x + a - 1 ≤ 0) ↔
  (a < 2 ∧ a - 1 ≤ x ∧ x ≤ 1) ∨
  (a = 2 ∧ x = 1) ∨
  (a > 2 ∧ 1 ≤ x ∧ x ≤ a - 1) := 
by
  sorry

end solve_quadratic_inequality_l1220_122095


namespace simplify_expr1_simplify_expr2_l1220_122009

variable (a b m n : ℝ)

theorem simplify_expr1 : 2 * a - 6 * b - 3 * a + 9 * b = -a + 3 * b := by
  sorry

theorem simplify_expr2 : 2 * (3 * m^2 - m * n) - m * n + m^2 = 7 * m^2 - 3 * m * n := by
  sorry

end simplify_expr1_simplify_expr2_l1220_122009


namespace smallest_norwegian_is_1344_l1220_122011

def is_norwegian (n : ℕ) : Prop :=
  ∃ d1 d2 d3 : ℕ, n > 0 ∧ d1 < d2 ∧ d2 < d3 ∧ d1 * d2 * d3 = n ∧ d1 + d2 + d3 = 2022

theorem smallest_norwegian_is_1344 : ∀ m : ℕ, (is_norwegian m) → m ≥ 1344 :=
by
  sorry

end smallest_norwegian_is_1344_l1220_122011


namespace binom_18_6_eq_13260_l1220_122050

/-- The binomial coefficient formula. -/
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The specific proof problem: compute binom(18, 6) and show that it equals 13260. -/
theorem binom_18_6_eq_13260 : binom 18 6 = 13260 :=
by
  sorry

end binom_18_6_eq_13260_l1220_122050


namespace eggs_collection_l1220_122071

theorem eggs_collection (b : ℕ) (c : ℕ) (t : ℕ) 
  (h₁ : b = 6) 
  (h₂ : c = 3 * b) 
  (h₃ : t = b - 4) : 
  b + c + t = 26 :=
by
  sorry

end eggs_collection_l1220_122071


namespace volleyball_ranking_l1220_122029

-- Define type for place
inductive Place where
  | first : Place
  | second : Place
  | third : Place

-- Define type for teams
inductive Team where
  | A : Team
  | B : Team
  | C : Team

open Place Team

-- Given conditions as hypotheses
def LiMing_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p first A ∨ p third A) ∧ (p first B ∨ p third B) ∧ 
  ¬ (p first A ∧ p third A) ∧ ¬ (p first B ∧ p third B)

def ZhangHua_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p third A ∨ p first C) ∧ (p third A ∨ p first A) ∧ 
  ¬ (p third A ∧ p first A) ∧ ¬ (p first C ∧ p third C)

def WangQiang_prediction_half_correct (p : Place → Team → Prop) : Prop :=
  (p second C ∨ p third B) ∧ (p second C ∨ p third C) ∧ 
  ¬ (p second C ∧ p third C) ∧ ¬ (p third B ∧ p second B)

-- Final proof problem
theorem volleyball_ranking (p : Place → Team → Prop) :
    (LiMing_prediction_half_correct p) →
    (ZhangHua_prediction_half_correct p) →
    (WangQiang_prediction_half_correct p) →
    p first C ∧ p second A ∧ p third B :=
  by
    sorry

end volleyball_ranking_l1220_122029


namespace min_value_expression_l1220_122080

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 48) :
  x^2 + 4 * x * y + 4 * y^2 + 3 * z^2 ≥ 144 :=
sorry

end min_value_expression_l1220_122080


namespace part1_part2_l1220_122094

theorem part1 (m : ℝ) :
  ∀ x : ℝ, x^2 + ( (2 * m - 1) : ℝ) * x + m^2 = 0 → m ≤ 1 / 4 :=
sorry

theorem part2 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, (x1^2 + (2*m -1)*x1 + m^2 = 0) ∧ (x2^2 + (2*m -1)*x2 + m^2 = 0) ∧ (x1*x2 + x1 + x2 = 4)) :
    m = -1 :=
sorry

end part1_part2_l1220_122094


namespace percentage_markup_l1220_122065

theorem percentage_markup (P : ℝ) : 
  (∀ (n : ℕ) (cost price total_earned : ℝ),
    n = 50 →
    cost = 1 →
    price = 1 + P / 100 →
    total_earned = 60 →
    n * price = total_earned) →
  P = 20 :=
by
  intro h
  have h₁ := h 50 1 (1 + P / 100) 60 rfl rfl rfl rfl
  sorry  -- Placeholder for proof steps

end percentage_markup_l1220_122065


namespace eval_otimes_l1220_122036

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem eval_otimes : otimes 4 2 = 18 :=
by
  sorry

end eval_otimes_l1220_122036


namespace payment_is_variable_l1220_122079

variable (x y : ℕ)

def price_of_pen : ℕ := 3

theorem payment_is_variable (x y : ℕ) (h : y = price_of_pen * x) : 
  (price_of_pen = 3) ∧ (∃ n : ℕ, y = 3 * n) :=
by 
  sorry

end payment_is_variable_l1220_122079


namespace polynomial_remainder_l1220_122026

theorem polynomial_remainder (x : ℝ) : 
  (x - 1)^100 + (x - 2)^200 = (x^2 - 3 * x + 2) * (some_q : ℝ) + 1 :=
sorry

end polynomial_remainder_l1220_122026


namespace cans_in_third_bin_l1220_122069

noncomputable def num_cans_in_bin (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | 4 => 11
  | 5 => 16
  | _ => sorry

theorem cans_in_third_bin :
  num_cans_in_bin 3 = 7 :=
sorry

end cans_in_third_bin_l1220_122069


namespace negation_of_proposition_l1220_122097

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l1220_122097


namespace prob_teamB_wins_first_game_l1220_122070
-- Import the necessary library

-- Define the conditions and the question in a Lean theorem statement
theorem prob_teamB_wins_first_game :
  (∀ (win_A win_B : ℕ), win_A < 4 ∧ win_B = 4) →
  (∀ (team_wins_game : ℕ → Prop), (team_wins_game 2 = false) ∧ (team_wins_game 3 = true)) →
  (∀ (team_wins_series : Prop), team_wins_series = (win_B ≥ 4 ∧ win_A < 4)) →
  (∀ (game_outcome_distribution : ℕ → ℕ → ℕ → ℕ → ℚ), game_outcome_distribution 4 4 2 2 = 1 / 2) →
  (∀ (first_game_outcome : Prop), first_game_outcome = true) →
  true :=
sorry

end prob_teamB_wins_first_game_l1220_122070


namespace necessary_but_not_sufficient_condition_l1220_122010

theorem necessary_but_not_sufficient_condition
    {a b : ℕ} :
    (¬ (a = 1) ∨ ¬ (b = 2)) ↔ (a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2) :=
by
    sorry

end necessary_but_not_sufficient_condition_l1220_122010


namespace total_students_appeared_l1220_122048

variable (T : ℝ) -- total number of students

def fraction_failed := 0.65
def num_failed := 546

theorem total_students_appeared :
  0.65 * T = 546 → T = 840 :=
by
  intro h
  sorry

end total_students_appeared_l1220_122048


namespace equation_no_solution_at_5_l1220_122049

theorem equation_no_solution_at_5 :
  ∀ (some_expr : ℝ), ¬(1 / (5 + 5) + some_expr = 1 / (5 - 5)) :=
by
  intro some_expr
  sorry

end equation_no_solution_at_5_l1220_122049


namespace find_u_l1220_122047

theorem find_u (u : ℝ) : (∃ x : ℝ, x = ( -15 - Real.sqrt 145 ) / 8 ∧ 4 * x^2 + 15 * x + u = 0) ↔ u = 5 := by
  sorry

end find_u_l1220_122047


namespace consistent_scale_l1220_122057

-- Conditions definitions

def dist_gardensquare_newtonsville : ℕ := 3  -- in inches
def dist_newtonsville_madison : ℕ := 4  -- in inches
def speed_gardensquare_newtonsville : ℕ := 50  -- mph
def time_gardensquare_newtonsville : ℕ := 2  -- hours
def speed_newtonsville_madison : ℕ := 60  -- mph
def time_newtonsville_madison : ℕ := 3  -- hours

-- Actual distances calculated
def actual_distance_gardensquare_newtonsville : ℕ := speed_gardensquare_newtonsville * time_gardensquare_newtonsville
def actual_distance_newtonsville_madison : ℕ := speed_newtonsville_madison * time_newtonsville_madison

-- Prove the scale is consistent across the map
theorem consistent_scale :
  actual_distance_gardensquare_newtonsville / dist_gardensquare_newtonsville =
  actual_distance_newtonsville_madison / dist_newtonsville_madison :=
by
  sorry

end consistent_scale_l1220_122057


namespace find_k_l1220_122035

theorem find_k (k : ℚ) : 
  ((3, -8) ≠ (k, 20)) ∧ 
  (∃ m, (4 * m = -3) ∧ (20 - (-8) = m * (k - 3))) → 
  k = -103/3 := 
by
  sorry

end find_k_l1220_122035


namespace largest_room_width_l1220_122078

theorem largest_room_width (w : ℕ) :
  (w * 30 - 15 * 8 = 1230) → (w = 45) :=
by
  intro h
  sorry

end largest_room_width_l1220_122078


namespace andy_solves_49_problems_l1220_122004

theorem andy_solves_49_problems : ∀ (a b : ℕ), a = 78 → b = 125 → b - a + 1 = 49 :=
by
  introv ha hb
  rw [ha, hb]
  norm_num
  sorry

end andy_solves_49_problems_l1220_122004


namespace count_five_digit_multiples_of_5_l1220_122019

-- Define the range of five-digit positive integers
def lower_bound : ℕ := 10000
def upper_bound : ℕ := 99999

-- Define the divisor
def divisor : ℕ := 5

-- Define the count of multiples of 5 in the range
def count_multiples_of_5 : ℕ :=
  (upper_bound / divisor) - (lower_bound / divisor) + 1

-- The main statement: The number of five-digit multiples of 5 is 18000
theorem count_five_digit_multiples_of_5 : count_multiples_of_5 = 18000 :=
  sorry

end count_five_digit_multiples_of_5_l1220_122019


namespace zero_x_intersections_l1220_122014

theorem zero_x_intersections 
  (a b c : ℝ) 
  (h_geom_seq : b^2 = a * c) 
  (h_ac_pos : a * c > 0) : 
  ∀ x : ℝ, ¬(ax^2 + bx + c = 0) := 
by 
  sorry

end zero_x_intersections_l1220_122014


namespace arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l1220_122007

open Real

theorem arctan_sum_lt_pi_div_two_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  arctan x + arctan y < (π / 2) ↔ x * y < 1 :=
sorry

theorem arctan_sum_lt_pi_iff (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  arctan x + arctan y + arctan z < π ↔ x * y * z < x + y + z :=
sorry

end arctan_sum_lt_pi_div_two_iff_arctan_sum_lt_pi_iff_l1220_122007


namespace random_event_is_option_D_l1220_122096

-- Definitions based on conditions
def rains_without_clouds : Prop := false
def like_charges_repel : Prop := true
def seeds_germinate_without_moisture : Prop := false
def draw_card_get_1 : Prop := true

-- Proof statement
theorem random_event_is_option_D : 
  (¬ rains_without_clouds ∧ like_charges_repel ∧ ¬ seeds_germinate_without_moisture ∧ draw_card_get_1) →
  (draw_card_get_1 = true) :=
by sorry

end random_event_is_option_D_l1220_122096


namespace quadratic_m_value_l1220_122052

theorem quadratic_m_value (m : ℤ) (hm1 : |m| = 2) (hm2 : m ≠ 2) : m = -2 :=
sorry

end quadratic_m_value_l1220_122052


namespace principle_calculation_l1220_122041

noncomputable def calculate_principal (A : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  A / (1 + (R * T))

theorem principle_calculation :
  calculate_principal 1456 0.05 2.4 = 1300 :=
by
  sorry

end principle_calculation_l1220_122041


namespace rat_to_chihuahua_ratio_is_six_to_one_l1220_122031

noncomputable def chihuahuas_thought_to_be : ℕ := 70
noncomputable def actual_rats : ℕ := 60

theorem rat_to_chihuahua_ratio_is_six_to_one
    (h : chihuahuas_thought_to_be - actual_rats = 10) :
    actual_rats / (chihuahuas_thought_to_be - actual_rats) = 6 :=
by
  sorry

end rat_to_chihuahua_ratio_is_six_to_one_l1220_122031


namespace yellow_tint_percentage_l1220_122027

theorem yellow_tint_percentage {V₀ V₁ V_t red_pct yellow_pct : ℝ} 
  (hV₀ : V₀ = 40)
  (hRed : red_pct = 0.20)
  (hYellow : yellow_pct = 0.25)
  (hAdd : V₁ = 10) :
  (yellow_pct * V₀ + V₁) / (V₀ + V₁) = 0.40 :=
by
  sorry

end yellow_tint_percentage_l1220_122027


namespace part_a_part_b_l1220_122090

theorem part_a (x y : ℕ) (h : x^3 + 5 * y = y^3 + 5 * x) : x = y :=
sorry

theorem part_b : ∃ (x y : ℝ), x ≠ y ∧ x > 0 ∧ y > 0 ∧ (x^3 + 5 * y = y^3 + 5 * x) :=
sorry

end part_a_part_b_l1220_122090


namespace product_of_integers_whose_cubes_sum_to_189_l1220_122055

theorem product_of_integers_whose_cubes_sum_to_189 :
  ∃ (a b : ℤ), a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  sorry

end product_of_integers_whose_cubes_sum_to_189_l1220_122055


namespace nails_sum_is_correct_l1220_122074

-- Define the fractions for sizes 2d, 3d, 5d, and 8d
def fraction_2d : ℚ := 1 / 6
def fraction_3d : ℚ := 2 / 15
def fraction_5d : ℚ := 1 / 10
def fraction_8d : ℚ := 1 / 8

-- Define the expected answer
def expected_fraction : ℚ := 21 / 40

-- The theorem to prove
theorem nails_sum_is_correct : fraction_2d + fraction_3d + fraction_5d + fraction_8d = expected_fraction :=
by
  -- The proof is not required as per the instructions
  sorry

end nails_sum_is_correct_l1220_122074


namespace find_principal_amount_l1220_122058

-- Definitions of the conditions
def rate_of_interest : ℝ := 0.20
def time_period : ℕ := 2
def interest_difference : ℝ := 144

-- Definitions for Simple Interest (SI) and Compound Interest (CI)
def simple_interest (P : ℝ) : ℝ := P * rate_of_interest * time_period
def compound_interest (P : ℝ) : ℝ := P * (1 + rate_of_interest)^time_period - P

-- Statement to prove the principal amount given the conditions
theorem find_principal_amount (P : ℝ) : 
    compound_interest P - simple_interest P = interest_difference → P = 3600 := by
    sorry

end find_principal_amount_l1220_122058


namespace sprint_team_total_miles_l1220_122066

-- Define the number of people and miles per person as constants
def numberOfPeople : ℕ := 250
def milesPerPerson : ℝ := 7.5

-- Assertion to prove the total miles
def totalMilesRun : ℝ := numberOfPeople * milesPerPerson

-- Proof statement
theorem sprint_team_total_miles : totalMilesRun = 1875 := 
by 
  -- Proof to be filled in
  sorry

end sprint_team_total_miles_l1220_122066


namespace quadratic_eq_real_roots_l1220_122076

theorem quadratic_eq_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + 2*m = 0) → m ≤ 9 / 8 :=
by
  sorry

end quadratic_eq_real_roots_l1220_122076


namespace abs_inequality_solution_l1220_122051

theorem abs_inequality_solution (x : ℝ) : 
  (|2 * x + 1| > 3) ↔ (x > 1 ∨ x < -2) :=
sorry

end abs_inequality_solution_l1220_122051


namespace trig_identity_l1220_122024

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
    Real.cos (2 * α) - Real.sin α * Real.cos α = -1 := 
by 
  sorry

end trig_identity_l1220_122024


namespace solution_m_value_l1220_122043

theorem solution_m_value (m : ℝ) : 
  (m^2 - 5*m + 4 > 0) ∧ (m^2 - 2*m = 0) ↔ m = 0 :=
by
  sorry

end solution_m_value_l1220_122043


namespace percentage_students_taking_music_l1220_122006

theorem percentage_students_taking_music
  (total_students : ℕ)
  (students_take_dance : ℕ)
  (students_take_art : ℕ)
  (students_take_music : ℕ)
  (percentage_students_taking_music : ℕ) :
  total_students = 400 →
  students_take_dance = 120 →
  students_take_art = 200 →
  students_take_music = total_students - students_take_dance - students_take_art →
  percentage_students_taking_music = (students_take_music * 100) / total_students →
  percentage_students_taking_music = 20 :=
by
  sorry

end percentage_students_taking_music_l1220_122006


namespace total_houses_is_160_l1220_122032

namespace MariamNeighborhood

-- Define the given conditions as variables in Lean.
def houses_on_one_side : ℕ := 40
def multiplier : ℕ := 3

-- Define the number of houses on the other side of the road.
def houses_on_other_side : ℕ := multiplier * houses_on_one_side

-- Define the total number of houses in Mariam's neighborhood.
def total_houses : ℕ := houses_on_one_side + houses_on_other_side

-- Prove that the total number of houses is 160.
theorem total_houses_is_160 : total_houses = 160 :=
by
  -- Placeholder for proof
  sorry

end MariamNeighborhood

end total_houses_is_160_l1220_122032
