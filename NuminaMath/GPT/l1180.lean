import Mathlib

namespace NUMINAMATH_GPT_solve_equation_l1180_118091

theorem solve_equation (a : ℝ) (x : ℝ) : (2 * a * x + 3) / (a - x) = 3 / 4 → x = 1 → a = -3 :=
by
  intros h h1
  rw [h1] at h
  sorry

end NUMINAMATH_GPT_solve_equation_l1180_118091


namespace NUMINAMATH_GPT_problem1_line_equation_problem2_circle_equation_l1180_118045

-- Problem 1: Equation of a specific line
def line_intersection (x y : ℝ) : Prop := 
  2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0

def line_perpendicular (x y : ℝ) : Prop :=
  6 * x - 8 * y + 3 = 0

noncomputable def find_line (x y : ℝ) : Prop :=
  ∃ (l : ℝ), (8 * x + 6 * y + l = 0) ∧ 
  line_intersection x y ∧ line_perpendicular x y

theorem problem1_line_equation : ∃ (x y : ℝ), find_line x y :=
sorry

-- Problem 2: Equation of a specific circle
def point_A (x y : ℝ) : Prop := 
  x = 5 ∧ y = 2

def point_B (x y : ℝ) : Prop := 
  x = 3 ∧ y = -2

def center_on_line (x y : ℝ) : Prop :=
  2 * x - y = 3

noncomputable def find_circle (x y r : ℝ) : Prop :=
  ((x - 2)^2 + (y - 1)^2 = r) ∧
  ∃ x1 y1 x2 y2, point_A x1 y1 ∧ point_B x2 y2 ∧ center_on_line x y ∧ ((x1 - x)^2 + (y1 - y)^2 = r)

theorem problem2_circle_equation : ∃ (x y r : ℝ), find_circle x y 10 :=
sorry

end NUMINAMATH_GPT_problem1_line_equation_problem2_circle_equation_l1180_118045


namespace NUMINAMATH_GPT_find_a_l1180_118070

theorem find_a (a r s : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 24) (h3 : s^2 = 9) : a = 16 :=
sorry

end NUMINAMATH_GPT_find_a_l1180_118070


namespace NUMINAMATH_GPT_minimum_b_value_l1180_118021

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2 * a))^2

theorem minimum_b_value (a : ℝ) : ∃ x_0 > 0, f x_0 a ≤ (4 / 5) :=
sorry

end NUMINAMATH_GPT_minimum_b_value_l1180_118021


namespace NUMINAMATH_GPT_count_two_digit_primes_with_given_conditions_l1180_118062

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def sum_of_digits_is_nine (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens + units = 9

def tens_greater_than_units (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens > units

theorem count_two_digit_primes_with_given_conditions :
  ∃ count : ℕ, count = 0 ∧ ∀ n, is_two_digit_prime n ∧ sum_of_digits_is_nine n ∧ tens_greater_than_units n → false :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_count_two_digit_primes_with_given_conditions_l1180_118062


namespace NUMINAMATH_GPT_quadratic_has_real_root_for_any_t_l1180_118020

theorem quadratic_has_real_root_for_any_t (s : ℝ) :
  (∀ t : ℝ, ∃ x : ℝ, s * x^2 + t * x + s - 1 = 0) ↔ (0 < s ∧ s ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_root_for_any_t_l1180_118020


namespace NUMINAMATH_GPT_intersection_count_is_one_l1180_118033

theorem intersection_count_is_one :
  (∀ x y : ℝ, y = 2 * x^3 + 6 * x + 1 → y = -3 / x^2) → ∃! p : ℝ × ℝ, p.2 = 2 * p.1^3 + 6 * p.1 + 1 ∧ p.2 = -3 / p.1 :=
sorry

end NUMINAMATH_GPT_intersection_count_is_one_l1180_118033


namespace NUMINAMATH_GPT_correspond_half_l1180_118086

theorem correspond_half (m n : ℕ) 
  (H : ∀ h : Fin m, ∃ g_set : Finset (Fin n), (g_set.card = n / 2) ∧ (∀ g : Fin n, g ∈ g_set))
  (G : ∀ g : Fin n, ∃ h_set : Finset (Fin m), (h_set.card ≤ m / 2) ∧ (∀ h : Fin m, h ∈ h_set)) :
  (∀ h : Fin m, ∀ g_set : Finset (Fin n), g_set.card = n / 2) ∧ (∀ g : Fin n, ∀ h_set : Finset (Fin m), h_set.card = m / 2) :=
by
  sorry

end NUMINAMATH_GPT_correspond_half_l1180_118086


namespace NUMINAMATH_GPT_proof_problem_l1180_118037

theorem proof_problem :
  ∀ (X : ℝ), 213 * 16 = 3408 → (213 * 16) + (1.6 * 2.13) = X → X - (5 / 2) * 1.25 = 3408.283 :=
by
  intros X h1 h2
  sorry

end NUMINAMATH_GPT_proof_problem_l1180_118037


namespace NUMINAMATH_GPT_length_more_than_breadth_l1180_118052

theorem length_more_than_breadth (b x : ℕ) 
  (h1 : 60 = b + x) 
  (h2 : 4 * b + 2 * x = 200) : x = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_more_than_breadth_l1180_118052


namespace NUMINAMATH_GPT_hotel_room_friends_distribution_l1180_118009

theorem hotel_room_friends_distribution 
    (rooms : ℕ)
    (friends : ℕ)
    (min_friends_per_room : ℕ)
    (max_friends_per_room : ℕ)
    (unique_ways : ℕ) :
    rooms = 6 →
    friends = 10 →
    min_friends_per_room = 1 →
    max_friends_per_room = 3 →
    unique_ways = 1058400 :=
by
  intros h_rooms h_friends h_min_friends h_max_friends
  sorry

end NUMINAMATH_GPT_hotel_room_friends_distribution_l1180_118009


namespace NUMINAMATH_GPT_swapped_digit_number_l1180_118000

theorem swapped_digit_number (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  10 * b + a = new_number :=
sorry

end NUMINAMATH_GPT_swapped_digit_number_l1180_118000


namespace NUMINAMATH_GPT_floor_x_floor_x_eq_20_l1180_118048

theorem floor_x_floor_x_eq_20 (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := 
sorry

end NUMINAMATH_GPT_floor_x_floor_x_eq_20_l1180_118048


namespace NUMINAMATH_GPT_fraction_sum_l1180_118097

theorem fraction_sum :
  (3 / 30 : ℝ) + (5 / 300) + (7 / 3000) = 0.119 := by
  sorry

end NUMINAMATH_GPT_fraction_sum_l1180_118097


namespace NUMINAMATH_GPT_find_A_for_diamondsuit_l1180_118081

-- Define the operation
def diamondsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- Define the specific instance of the operation equated to 57
theorem find_A_for_diamondsuit :
  ∃ A : ℝ, diamondsuit A 10 = 57 ↔ A = 20 := by
  sorry

end NUMINAMATH_GPT_find_A_for_diamondsuit_l1180_118081


namespace NUMINAMATH_GPT_sequence_general_formula_l1180_118069

theorem sequence_general_formula (n : ℕ) (h : n ≥ 1) :
  ∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n ≥ 1, a (n + 1) = a n / (1 + a n)) ∧ a n = (1 : ℝ) / n :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1180_118069


namespace NUMINAMATH_GPT_distance_between_foci_is_six_l1180_118083

-- Lean 4 Statement
noncomputable def distance_between_foci (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  if (p1 = (1, 3) ∧ p2 = (6, -1) ∧ p3 = (11, 3)) then 6 else 0

theorem distance_between_foci_is_six : distance_between_foci (1, 3) (6, -1) (11, 3) = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_is_six_l1180_118083


namespace NUMINAMATH_GPT_average_marks_of_passed_l1180_118076

theorem average_marks_of_passed
  (total_boys : ℕ)
  (average_all : ℕ)
  (average_failed : ℕ)
  (passed_boys : ℕ)
  (num_boys := 120)
  (avg_all := 37)
  (avg_failed := 15)
  (passed := 110)
  (failed_boys := total_boys - passed_boys)
  (total_marks_all := average_all * total_boys)
  (total_marks_failed := average_failed * failed_boys)
  (total_marks_passed := total_marks_all - total_marks_failed)
  (average_passed := total_marks_passed / passed_boys) :
  average_passed = 39 :=
by
  -- start of proof
  sorry

end NUMINAMATH_GPT_average_marks_of_passed_l1180_118076


namespace NUMINAMATH_GPT_fraction_product_l1180_118005

theorem fraction_product :
  (3 / 7) * (5 / 8) * (9 / 13) * (11 / 17) = 1485 / 12376 := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_l1180_118005


namespace NUMINAMATH_GPT_correct_calculation_l1180_118066

theorem correct_calculation :
  (∀ x : ℤ, x^5 + x^3 ≠ x^8) ∧
  (∀ x : ℤ, x^5 - x^3 ≠ x^2) ∧
  (∀ x : ℤ, x^5 * x^3 = x^8) ∧
  (∀ x : ℤ, (-3 * x)^3 ≠ -9 * x^3) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1180_118066


namespace NUMINAMATH_GPT_polynomial_divisible_by_x_minus_2_l1180_118065

theorem polynomial_divisible_by_x_minus_2 (k : ℝ) :
  (2 * (2 : ℝ)^3 - 8 * (2 : ℝ)^2 + k * (2 : ℝ) - 10 = 0) → 
  k = 13 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_divisible_by_x_minus_2_l1180_118065


namespace NUMINAMATH_GPT_students_with_uncool_parents_l1180_118057

theorem students_with_uncool_parents (class_size : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ) : 
  class_size = 40 → cool_dads = 18 → cool_moms = 20 → both_cool_parents = 10 → 
  (class_size - (cool_dads - both_cool_parents + cool_moms - both_cool_parents + both_cool_parents) = 12) :=
by
  sorry

end NUMINAMATH_GPT_students_with_uncool_parents_l1180_118057


namespace NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1180_118028

theorem hydrogen_atoms_in_compound : 
  ∀ (Al_weight O_weight H_weight : ℕ) (total_weight : ℕ) (num_Al num_O num_H : ℕ),
  Al_weight = 27 →
  O_weight = 16 →
  H_weight = 1 →
  total_weight = 78 →
  num_Al = 1 →
  num_O = 3 →
  (num_Al * Al_weight + num_O * O_weight + num_H * H_weight = total_weight) →
  num_H = 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1180_118028


namespace NUMINAMATH_GPT_correct_operation_l1180_118051

theorem correct_operation (a : ℝ) : (-a^3)^4 = a^12 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1180_118051


namespace NUMINAMATH_GPT_ticket_cost_l1180_118001

theorem ticket_cost (a : ℝ)
  (h1 : ∀ c : ℝ, c = a / 3)
  (h2 : 3 * a + 5 * (a / 3) = 27.75) :
  6 * a + 9 * (a / 3) = 53.52 := 
sorry

end NUMINAMATH_GPT_ticket_cost_l1180_118001


namespace NUMINAMATH_GPT_GCF_seven_eight_factorial_l1180_118095

-- Given conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Calculating 7! and 8!
def seven_factorial := factorial 7
def eight_factorial := factorial 8

-- Proof statement
theorem GCF_seven_eight_factorial : ∃ g, g = seven_factorial ∧ g = Nat.gcd seven_factorial eight_factorial ∧ g = 5040 :=
by sorry

end NUMINAMATH_GPT_GCF_seven_eight_factorial_l1180_118095


namespace NUMINAMATH_GPT_smallest_part_2340_division_l1180_118003

theorem smallest_part_2340_division :
  ∃ (A B C : ℕ), (A + B + C = 2340) ∧ 
                 (A / 5 = B / 7) ∧ 
                 (B / 7 = C / 11) ∧ 
                 (A = 510) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_part_2340_division_l1180_118003


namespace NUMINAMATH_GPT_intersection_of_sets_l1180_118050

def setM : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }
def setN : Set ℝ := { x | Real.log x ≥ 0 }

theorem intersection_of_sets : (setM ∩ setN) = { x | 1 ≤ x ∧ x ≤ 4 } := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_sets_l1180_118050


namespace NUMINAMATH_GPT_proof_problem_l1180_118072

-- Definitions of the conditions
def cond1 (r : ℕ) : Prop := 2^r = 16
def cond2 (s : ℕ) : Prop := 5^s = 25

-- Statement of the problem
theorem proof_problem (r s : ℕ) (h₁ : cond1 r) (h₂ : cond2 s) : r + s = 6 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1180_118072


namespace NUMINAMATH_GPT_electrical_appliance_supermarket_l1180_118014

-- Define the known quantities and conditions
def purchase_price_A : ℝ := 140
def purchase_price_B : ℝ := 100
def week1_sales_A : ℕ := 4
def week1_sales_B : ℕ := 3
def week1_revenue : ℝ := 1250
def week2_sales_A : ℕ := 5
def week2_sales_B : ℕ := 5
def week2_revenue : ℝ := 1750
def total_units : ℕ := 50
def budget : ℝ := 6500
def profit_goal : ℝ := 2850

-- Define the unknown selling prices
noncomputable def selling_price_A : ℝ := 200
noncomputable def selling_price_B : ℝ := 150

-- Define the constraints
def cost_constraint (m : ℕ) : Prop := 140 * m + 100 * (50 - m) ≤ 6500
def profit_exceeds_goal (m : ℕ) : Prop := (200 - 140) * m + (150 - 100) * (50 - m) > 2850

-- The main theorem stating the results
theorem electrical_appliance_supermarket :
  (4 * selling_price_A + 3 * selling_price_B = week1_revenue)
  ∧ (5 * selling_price_A + 5 * selling_price_B = week2_revenue)
  ∧ (∃ m : ℕ, m ≤ 37 ∧ cost_constraint m)
  ∧ (∃ m : ℕ, m > 35 ∧ m ≤ 37 ∧ profit_exceeds_goal m) :=
sorry

end NUMINAMATH_GPT_electrical_appliance_supermarket_l1180_118014


namespace NUMINAMATH_GPT_bluegrass_percentage_l1180_118027

theorem bluegrass_percentage (rx : ℝ) (ry : ℝ) (f : ℝ) (rm : ℝ) (wx : ℝ) (wy : ℝ) (B : ℝ) :
  rx = 0.4 →
  ry = 0.25 →
  f = 0.75 →
  rm = 0.35 →
  wx = 0.6667 →
  wy = 0.3333 →
  (wx * rx + wy * ry = rm) →
  B = 1.0 - rx →
  B = 0.6 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_bluegrass_percentage_l1180_118027


namespace NUMINAMATH_GPT_shirt_ratio_l1180_118004

theorem shirt_ratio
  (A B S : ℕ)
  (h1 : A = 6 * B)
  (h2 : B = 3)
  (h3 : S = 72) :
  S / A = 4 :=
by
  sorry

end NUMINAMATH_GPT_shirt_ratio_l1180_118004


namespace NUMINAMATH_GPT_expression_positive_intervals_l1180_118015

theorem expression_positive_intervals :
  {x : ℝ | (x + 2) * (x - 3) > 0} = {x | x < -2} ∪ {x | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_expression_positive_intervals_l1180_118015


namespace NUMINAMATH_GPT_min_value_a_is_1_or_100_l1180_118039

noncomputable def f (x : ℝ) : ℝ := x + 100 / x

theorem min_value_a_is_1_or_100 (a : ℝ) (m1 m2 : ℝ) 
  (h1 : a > 0) 
  (h_m1 : ∀ x, 0 < x ∧ x ≤ a → f x ≥ m1)
  (h_m1_min : ∃ x, 0 < x ∧ x ≤ a ∧ f x = m1)
  (h_m2 : ∀ x, a ≤ x → f x ≥ m2)
  (h_m2_min : ∃ x, a ≤ x ∧ f x = m2)
  (h_prod : m1 * m2 = 2020) : 
  a = 1 ∨ a = 100 :=
sorry

end NUMINAMATH_GPT_min_value_a_is_1_or_100_l1180_118039


namespace NUMINAMATH_GPT_geometric_sequence_S9_l1180_118043

theorem geometric_sequence_S9 (S : ℕ → ℝ) (S3_eq : S 3 = 2) (S6_eq : S 6 = 6) : S 9 = 14 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_S9_l1180_118043


namespace NUMINAMATH_GPT_investment_three_years_ago_l1180_118011

noncomputable def initial_investment (final_amount : ℝ) : ℝ :=
  final_amount / (1.08 ^ 3)

theorem investment_three_years_ago :
  abs (initial_investment 439.23 - 348.68) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_investment_three_years_ago_l1180_118011


namespace NUMINAMATH_GPT_correct_inequality_l1180_118087

variables {a b c : ℝ}
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem correct_inequality (h_a_pos : a > 0) (h_discriminant_pos : b^2 - 4 * a * c > 0) (h_c_neg : c < 0) (h_b_neg : b < 0) :
  a * b * c > 0 :=
sorry

end NUMINAMATH_GPT_correct_inequality_l1180_118087


namespace NUMINAMATH_GPT_shelves_used_l1180_118047

-- Define the initial conditions
def initial_stock : Float := 40.0
def additional_stock : Float := 20.0
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_stock + additional_stock

-- Define the number of shelves
def number_of_shelves : Float := total_books / books_per_shelf

-- The proof statement that needs to be proven
theorem shelves_used : number_of_shelves = 15.0 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_shelves_used_l1180_118047


namespace NUMINAMATH_GPT_negation_example_l1180_118082

open Classical
variable (x : ℝ)

theorem negation_example :
  (¬ (∀ x : ℝ, 2 * x - 1 > 0)) ↔ (∃ x : ℝ, 2 * x - 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1180_118082


namespace NUMINAMATH_GPT_total_sticks_needed_l1180_118059

theorem total_sticks_needed :
  let simon_sticks := 36
  let gerry_sticks := 2 * (simon_sticks / 3)
  let total_simon_and_gerry := simon_sticks + gerry_sticks
  let micky_sticks := total_simon_and_gerry + 9
  total_simon_and_gerry + micky_sticks = 129 :=
by
  sorry

end NUMINAMATH_GPT_total_sticks_needed_l1180_118059


namespace NUMINAMATH_GPT_convert_to_scientific_notation_l1180_118092

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_convert_to_scientific_notation_l1180_118092


namespace NUMINAMATH_GPT_smallest_percentage_all_correct_l1180_118063

theorem smallest_percentage_all_correct (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.8)
  (h3 : p3 = 0.7) :
  ∃ x, x = 0.4 ∧ (x ≤ 1 - ((1 - p1) + (1 - p2) + (1 - p3))) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_percentage_all_correct_l1180_118063


namespace NUMINAMATH_GPT_percent_increase_in_area_l1180_118064

theorem percent_increase_in_area (s : ℝ) (h_s : s > 0) :
  let medium_area := s^2
  let large_length := 1.20 * s
  let large_width := 1.25 * s
  let large_area := large_length * large_width 
  let percent_increase := ((large_area - medium_area) / medium_area) * 100
  percent_increase = 50 := by
    sorry

end NUMINAMATH_GPT_percent_increase_in_area_l1180_118064


namespace NUMINAMATH_GPT_problem_part_I_problem_part_II_l1180_118018

-- Define the function f(x) given by the problem
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Define the conditions for part (Ⅰ)
def conditions_part_I (a x : ℝ) : Prop :=
  (1 ≤ x ∧ x ≤ a) ∧ (1 ≤ f x a ∧ f x a ≤ a)

-- Lean statement for part (Ⅰ)
theorem problem_part_I (a : ℝ) (h : a > 1) :
  (∀ x, conditions_part_I a x) → a = 2 := by sorry

-- Define the conditions for part (Ⅱ)
def decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 2 → f x a ≥ f y a

def abs_difference_condition (a : ℝ) : Prop :=
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ a + 1 ∧ 1 ≤ x2 ∧ x2 ≤ a + 1 → |f x1 a - f x2 a| ≤ 4

-- Lean statement for part (Ⅱ)
theorem problem_part_II (a : ℝ) (h : a > 1) :
  (decreasing_on_interval a) ∧ (abs_difference_condition a) → (2 ≤ a ∧ a ≤ 3) := by sorry

end NUMINAMATH_GPT_problem_part_I_problem_part_II_l1180_118018


namespace NUMINAMATH_GPT_solve_triple_l1180_118013

theorem solve_triple (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c + a * b + c = a^3) : 
  (b = a - 1 ∧ c = a) ∨ (b = 1 ∧ c = a * (a - 1)) :=
by 
  sorry

end NUMINAMATH_GPT_solve_triple_l1180_118013


namespace NUMINAMATH_GPT_max_profit_price_range_for_minimum_profit_l1180_118053

noncomputable def functional_relationship (x : ℝ) : ℝ :=
-10 * x^2 + 2000 * x - 84000

theorem max_profit :
  ∃ x, (∀ x₀, x₀ ≠ x → functional_relationship x₀ < functional_relationship x) ∧
  functional_relationship x = 16000 := 
sorry

theorem price_range_for_minimum_profit :
  ∀ (x : ℝ), 
  -10 * (x - 100)^2 + 16000 - 1750 ≥ 12000 → 
  85 ≤ x ∧ x ≤ 115 :=
sorry

end NUMINAMATH_GPT_max_profit_price_range_for_minimum_profit_l1180_118053


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_subset_condition_l1180_118089

open Set

variable (a : ℝ)
def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | -1-2*a ≤ x ∧ x ≤ a-2}

theorem sufficient_but_not_necessary_condition (H : ∃ x ∈ A, x ∉ B a) : a ≥ 7 := sorry

theorem subset_condition (H : B a ⊆ A) : a < 1/3 := sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_subset_condition_l1180_118089


namespace NUMINAMATH_GPT_find_c_l1180_118040

def p (x : ℝ) := 4 * x - 9
def q (x : ℝ) (c : ℝ) := 5 * x - c

theorem find_c : ∃ (c : ℝ), p (q 3 c) = 14 ∧ c = 9.25 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1180_118040


namespace NUMINAMATH_GPT_crayons_lost_l1180_118007

theorem crayons_lost (initial_crayons ending_crayons : ℕ) (h_initial : initial_crayons = 253) (h_ending : ending_crayons = 183) : (initial_crayons - ending_crayons) = 70 :=
by
  sorry

end NUMINAMATH_GPT_crayons_lost_l1180_118007


namespace NUMINAMATH_GPT_cost_per_bag_of_potatoes_l1180_118035

variable (x : ℕ)

def chickens_cost : ℕ := 5 * 3
def celery_cost : ℕ := 4 * 2
def total_paid : ℕ := 35
def potatoes_cost (x : ℕ) : ℕ := 2 * x

theorem cost_per_bag_of_potatoes : 
  chickens_cost + celery_cost + potatoes_cost x = total_paid → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_bag_of_potatoes_l1180_118035


namespace NUMINAMATH_GPT_inequality_div_c_squared_l1180_118023

theorem inequality_div_c_squared (a b c : ℝ) (h : a > b) : (a / (c^2 + 1) > b / (c^2 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_div_c_squared_l1180_118023


namespace NUMINAMATH_GPT_solve_inequality_l1180_118077

theorem solve_inequality (x : ℝ) :
  (4 * x^4 + x^2 + 4 * x - 5 * x^2 * |x + 2| + 4) ≥ 0 ↔ 
  x ∈ Set.Iic (-1) ∪ Set.Icc ((1 - Real.sqrt 33) / 8) ((1 + Real.sqrt 33) / 8) ∪ Set.Ici 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1180_118077


namespace NUMINAMATH_GPT_tangent_circle_radius_l1180_118010

theorem tangent_circle_radius (O A B C : ℝ) (r1 r2 : ℝ) :
  (O = 5) →
  (abs (A - B) = 8) →
  (C = (2 * A + B) / 3) →
  r1 = 8 / 9 ∨ r2 = 32 / 9 :=
sorry

end NUMINAMATH_GPT_tangent_circle_radius_l1180_118010


namespace NUMINAMATH_GPT_intersection_M_N_l1180_118024

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_M_N : M ∩ N = Set.Ico 1 3 := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1180_118024


namespace NUMINAMATH_GPT_probability_red_or_white_ball_l1180_118044

theorem probability_red_or_white_ball :
  let red_balls := 3
  let yellow_balls := 2
  let white_balls := 1
  let total_balls := red_balls + yellow_balls + white_balls
  let favorable_outcomes := red_balls + white_balls
  (favorable_outcomes / total_balls : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_probability_red_or_white_ball_l1180_118044


namespace NUMINAMATH_GPT_xy_value_l1180_118016

theorem xy_value (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = -5) : x + y = 1 := 
sorry

end NUMINAMATH_GPT_xy_value_l1180_118016


namespace NUMINAMATH_GPT_length_to_width_ratio_l1180_118078

/-- Let the perimeter of the rectangular sandbox be 30 feet,
    the width be 5 feet, and the length be some multiple of the width.
    Prove that the ratio of the length to the width is 2:1. -/
theorem length_to_width_ratio (P w : ℕ) (h1 : P = 30) (h2 : w = 5) (h3 : ∃ k, l = k * w) : 
  ∃ l, (P = 2 * (l + w)) ∧ (l / w = 2) := 
sorry

end NUMINAMATH_GPT_length_to_width_ratio_l1180_118078


namespace NUMINAMATH_GPT_least_positive_integer_greater_than_100_l1180_118073

theorem least_positive_integer_greater_than_100 : ∃ n : ℕ, n > 100 ∧ (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) ∧ n = 2521 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_greater_than_100_l1180_118073


namespace NUMINAMATH_GPT_problem_inequality_l1180_118096

theorem problem_inequality 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_le_a : a ≤ 1)
  (h_pos_b : 0 < b) (h_le_b : b ≤ 1)
  (h_pos_c : 0 < c) (h_le_c : c ≤ 1)
  (h_pos_d : 0 < d) (h_le_d : d ≤ 1) :
  (1 / (a^2 + b^2 + c^2 + d^2)) ≥ (1 / 4) + (1 - a) * (1 - b) * (1 - c) * (1 - d) :=
by
  sorry

end NUMINAMATH_GPT_problem_inequality_l1180_118096


namespace NUMINAMATH_GPT_quadratic_roots_ratio_l1180_118054

theorem quadratic_roots_ratio (m n p : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : p ≠ 0)
    (h₄ : ∀ (s₁ s₂ : ℝ), s₁ + s₂ = -p ∧ s₁ * s₂ = m ∧ 3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n) :
    n / p = 27 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_ratio_l1180_118054


namespace NUMINAMATH_GPT_five_a_plus_five_b_eq_neg_twenty_five_thirds_l1180_118088

variable (g f : ℝ → ℝ)
variable (a b : ℝ)
axiom g_def : ∀ x, g x = 3 * x + 5
axiom g_inv_rel : ∀ x, g x = (f⁻¹ x) - 1
axiom f_def : ∀ x, f x = a * x + b
axiom f_inv_def : ∀ x, f⁻¹ (f x) = x

theorem five_a_plus_five_b_eq_neg_twenty_five_thirds :
    5 * a + 5 * b = -25 / 3 :=
sorry

end NUMINAMATH_GPT_five_a_plus_five_b_eq_neg_twenty_five_thirds_l1180_118088


namespace NUMINAMATH_GPT_largest_divisible_by_88_l1180_118085

theorem largest_divisible_by_88 (n : ℕ) (h₁ : n = 9999) (h₂ : n % 88 = 55) : n - 55 = 9944 := by
  sorry

end NUMINAMATH_GPT_largest_divisible_by_88_l1180_118085


namespace NUMINAMATH_GPT_stratified_sampling_grade11_l1180_118036

noncomputable def g10 : ℕ := 500
noncomputable def total_students : ℕ := 1350
noncomputable def g10_sample : ℕ := 120
noncomputable def ratio : ℚ := g10_sample / g10
noncomputable def g11 : ℕ := 450
noncomputable def g12 : ℕ := g11 - 50

theorem stratified_sampling_grade11 :
  g10 + g11 + g12 = total_students →
  (g10_sample / g10) = ratio →
  sample_g11 = g11 * ratio →
  sample_g11 = 108 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_grade11_l1180_118036


namespace NUMINAMATH_GPT_a4_is_5_l1180_118046

-- Define the condition x^5 = a_n + a_1(x-1) + a_2(x-1)^2 + a_3(x-1)^3 + a_4(x-1)^4 + a_5(x-1)^5
noncomputable def polynomial_identity (x a_n a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5

-- Define the theorem statement
theorem a4_is_5 (x a_n a_1 a_2 a_3 a_5 : ℝ) (h : polynomial_identity x a_n a_1 a_2 a_3 5 a_5) : a_4 = 5 :=
 by
 sorry

end NUMINAMATH_GPT_a4_is_5_l1180_118046


namespace NUMINAMATH_GPT_transform_quadratic_l1180_118056

theorem transform_quadratic (x m n : ℝ) 
  (h : x^2 - 6 * x - 1 = 0) : 
  (x + m)^2 = n ↔ (m = 3 ∧ n = 10) :=
by sorry

end NUMINAMATH_GPT_transform_quadratic_l1180_118056


namespace NUMINAMATH_GPT_sin_alpha_minus_pi_over_6_l1180_118022

open Real

theorem sin_alpha_minus_pi_over_6 (α : ℝ) (h : sin (α + π / 6) + 2 * sin (α / 2) ^ 2 = 1 - sqrt 2 / 2) : 
  sin (α - π / 6) = -sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_sin_alpha_minus_pi_over_6_l1180_118022


namespace NUMINAMATH_GPT_Linda_has_24_classmates_l1180_118084

theorem Linda_has_24_classmates 
  (cookies_per_student : ℕ := 10)
  (cookies_per_batch : ℕ := 48)
  (chocolate_chip_batches : ℕ := 2)
  (oatmeal_raisin_batches : ℕ := 1)
  (additional_batches : ℕ := 2) : 
  (chocolate_chip_batches * cookies_per_batch + oatmeal_raisin_batches * cookies_per_batch + additional_batches * cookies_per_batch) / cookies_per_student = 24 := 
by 
  sorry

end NUMINAMATH_GPT_Linda_has_24_classmates_l1180_118084


namespace NUMINAMATH_GPT_slope_of_AB_l1180_118038

theorem slope_of_AB (A B : (ℕ × ℕ)) (hA : A = (3, 4)) (hB : B = (2, 3)) : 
  (B.2 - A.2) / (B.1 - A.1) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_slope_of_AB_l1180_118038


namespace NUMINAMATH_GPT_ratio_of_areas_l1180_118079

theorem ratio_of_areas (len_rect width_rect area_tri : ℝ) (h1 : len_rect = 6) (h2 : width_rect = 4) (h3 : area_tri = 60) :
    (len_rect * width_rect) / area_tri = 2 / 5 :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_ratio_of_areas_l1180_118079


namespace NUMINAMATH_GPT_impossible_to_empty_pile_l1180_118042

theorem impossible_to_empty_pile (a b c : ℕ) (h : a = 1993 ∧ b = 199 ∧ c = 19) : 
  ¬ (∃ x y z : ℕ, (x + y + z = 0) ∧ (x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧ z = a ∨ z = b ∨ z = c)) := 
sorry

end NUMINAMATH_GPT_impossible_to_empty_pile_l1180_118042


namespace NUMINAMATH_GPT_alyssa_cookie_count_l1180_118094

variable (Aiyanna_cookies Alyssa_cookies : ℕ)
variable (h1 : Aiyanna_cookies = 140)
variable (h2 : Aiyanna_cookies = Alyssa_cookies + 11)

theorem alyssa_cookie_count : Alyssa_cookies = 129 := by
  -- We can use the given conditions to prove the theorem
  sorry

end NUMINAMATH_GPT_alyssa_cookie_count_l1180_118094


namespace NUMINAMATH_GPT_solve_for_x_l1180_118093

theorem solve_for_x (x : ℂ) (i : ℂ) (h : i ^ 2 = -1) (eqn : 3 + i * x = 5 - 2 * i * x) : x = i / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1180_118093


namespace NUMINAMATH_GPT_total_flowers_in_3_hours_l1180_118019

-- Constants representing the number of each type of flower
def roses : ℕ := 12
def sunflowers : ℕ := 15
def tulips : ℕ := 9
def daisies : ℕ := 18
def orchids : ℕ := 6
def total_flowers : ℕ := 60

-- Number of flowers each bee can pollinate in an hour
def bee_A_rate (roses sunflowers tulips: ℕ) : ℕ := 2 + 3 + 1
def bee_B_rate (daisies orchids: ℕ) : ℕ := 4 + 1
def bee_C_rate (roses sunflowers tulips daisies orchids: ℕ) : ℕ := 1 + 2 + 2 + 3 + 1

-- Total number of flowers pollinated by all bees in an hour
def total_bees_rate (bee_A_rate bee_B_rate bee_C_rate: ℕ) : ℕ := bee_A_rate + bee_B_rate + bee_C_rate

-- Proving the total flowers pollinated in 3 hours
theorem total_flowers_in_3_hours : total_bees_rate 6 5 9 * 3 = total_flowers := 
by {
  sorry
}

end NUMINAMATH_GPT_total_flowers_in_3_hours_l1180_118019


namespace NUMINAMATH_GPT_minimum_students_for_same_vote_l1180_118026

theorem minimum_students_for_same_vote (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 2) :
  ∃ m, m = 46 ∧ ∀ (students : Finset (Finset ℕ)), students.card = m → 
    (∃ s1 s2, s1 ≠ s2 ∧ s1.card = k ∧ s2.card = k ∧ s1 ⊆ (Finset.range n) ∧ s2 ⊆ (Finset.range n) ∧ s1 = s2) :=
by 
  sorry

end NUMINAMATH_GPT_minimum_students_for_same_vote_l1180_118026


namespace NUMINAMATH_GPT_seq_diff_five_consec_odd_avg_55_l1180_118017

theorem seq_diff_five_consec_odd_avg_55 {a b c d e : ℤ} 
    (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) (h5: e % 2 = 1)
    (h6: b = a + 2) (h7: c = a + 4) (h8: d = a + 6) (h9: e = a + 8)
    (avg_5_seq : (a + b + c + d + e) / 5 = 55) : 
    e - a = 8 := 
by
    -- proof part can be skipped with sorry
    sorry

end NUMINAMATH_GPT_seq_diff_five_consec_odd_avg_55_l1180_118017


namespace NUMINAMATH_GPT_percent_first_question_l1180_118075

variable (A B : ℝ) (A_inter_B : ℝ) (A_union_B : ℝ)

-- Given conditions
def condition1 : B = 0.49 := sorry
def condition2 : A_inter_B = 0.32 := sorry
def condition3 : A_union_B = 0.80 := sorry
def union_formula : A_union_B = A + B - A_inter_B := 
by sorry

-- Prove that A = 0.63
theorem percent_first_question (h1 : B = 0.49) 
                               (h2 : A_inter_B = 0.32) 
                               (h3 : A_union_B = 0.80) 
                               (h4 : A_union_B = A + B - A_inter_B) : 
                               A = 0.63 :=
by sorry

end NUMINAMATH_GPT_percent_first_question_l1180_118075


namespace NUMINAMATH_GPT_sqrt_54_sub_sqrt_6_l1180_118071

theorem sqrt_54_sub_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_54_sub_sqrt_6_l1180_118071


namespace NUMINAMATH_GPT_select_four_person_committee_l1180_118012

open Nat

theorem select_four_person_committee 
  (n : ℕ)
  (h1 : (n * (n - 1) * (n - 2)) / 6 = 21) 
  : (n = 9) → Nat.choose n 4 = 126 :=
by
  sorry

end NUMINAMATH_GPT_select_four_person_committee_l1180_118012


namespace NUMINAMATH_GPT_sin_double_angle_values_l1180_118029

theorem sin_double_angle_values (α : ℝ) (hα : 0 < α ∧ α < π) (h : 3 * (Real.cos α)^2 = Real.sin ((π / 4) - α)) :
  Real.sin (2 * α) = 1 ∨ Real.sin (2 * α) = -17 / 18 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_values_l1180_118029


namespace NUMINAMATH_GPT_students_more_than_pets_l1180_118006

theorem students_more_than_pets :
  let students_per_classroom := 15
  let rabbits_per_classroom := 1
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 6
  let total_students := students_per_classroom * number_of_classrooms
  let total_pets := (rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms
  total_students - total_pets = 66 :=
by
  sorry

end NUMINAMATH_GPT_students_more_than_pets_l1180_118006


namespace NUMINAMATH_GPT_simplify_expression_l1180_118099

theorem simplify_expression (x : ℝ) : (3 * x + 15) + (100 * x + 15) + (10 * x - 5) = 113 * x + 25 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1180_118099


namespace NUMINAMATH_GPT_range_of_a_l1180_118060

theorem range_of_a (a : ℝ) (h : (∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2)) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1180_118060


namespace NUMINAMATH_GPT_third_chapter_pages_l1180_118090

theorem third_chapter_pages (x : ℕ) (h : 18 = x + 15) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_third_chapter_pages_l1180_118090


namespace NUMINAMATH_GPT_hostel_cost_for_23_days_l1180_118055

theorem hostel_cost_for_23_days :
  let first_week_days := 7
  let additional_days := 23 - first_week_days
  let cost_first_week := 18 * first_week_days
  let cost_additional_weeks := 11 * additional_days
  23 * ((cost_first_week + cost_additional_weeks) / 23) = 302 :=
by sorry

end NUMINAMATH_GPT_hostel_cost_for_23_days_l1180_118055


namespace NUMINAMATH_GPT_div_by_6_l1180_118098

theorem div_by_6 (m : ℕ) : 6 ∣ (m^3 + 11 * m) :=
sorry

end NUMINAMATH_GPT_div_by_6_l1180_118098


namespace NUMINAMATH_GPT_lin_reg_proof_l1180_118074

variable (x y : List ℝ)
variable (n : ℝ := 10)
variable (sum_x : ℝ := 80)
variable (sum_y : ℝ := 20)
variable (sum_xy : ℝ := 184)
variable (sum_x2 : ℝ := 720)

noncomputable def mean (lst: List ℝ) (n: ℝ) : ℝ := (List.sum lst) / n

noncomputable def lin_reg_slope (n sum_x sum_y sum_xy sum_x2 : ℝ) : ℝ :=
  (sum_xy - n * (sum_x / n) * (sum_y / n)) / (sum_x2 - n * (sum_x / n) ^ 2)

noncomputable def lin_reg_intercept (sum_x sum_y : ℝ) (slope : ℝ) (n : ℝ) : ℝ :=
  (sum_y / n) - slope * (sum_x / n)

theorem lin_reg_proof :
  lin_reg_slope n sum_x sum_y sum_xy sum_x2 = 0.3 ∧ 
  lin_reg_intercept sum_x sum_y 0.3 n = -0.4 ∧ 
  (0.3 * 7 - 0.4 = 1.7) :=
by
  sorry

end NUMINAMATH_GPT_lin_reg_proof_l1180_118074


namespace NUMINAMATH_GPT_breaststroke_speed_correct_l1180_118034

-- Defining the given conditions
def total_distance : ℕ := 500
def front_crawl_speed : ℕ := 45
def front_crawl_time : ℕ := 8
def total_time : ℕ := 12

-- Definition of the breaststroke speed given the conditions
def breaststroke_speed : ℕ :=
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  breaststroke_distance / breaststroke_time

-- Theorem to prove the breaststroke speed is 35 yards per minute
theorem breaststroke_speed_correct : breaststroke_speed = 35 :=
  sorry

end NUMINAMATH_GPT_breaststroke_speed_correct_l1180_118034


namespace NUMINAMATH_GPT_simple_interest_years_l1180_118031

variable (P R T : ℕ)
variable (deltaI : ℕ := 400)
variable (P_value : P = 800)

theorem simple_interest_years 
  (h : (800 * (R + 5) * T / 100) = (800 * R * T / 100) + 400) :
  T = 10 :=
by sorry

end NUMINAMATH_GPT_simple_interest_years_l1180_118031


namespace NUMINAMATH_GPT_handshake_count_250_l1180_118041

theorem handshake_count_250 (n m : ℕ) (h1 : n = 5) (h2 : m = 5) :
  (n * m * (n * m - 1 - (n - 1))) / 2 = 250 :=
by
  -- Traditionally the theorem proof part goes here but it is omitted
  sorry

end NUMINAMATH_GPT_handshake_count_250_l1180_118041


namespace NUMINAMATH_GPT_solve_for_x_l1180_118061

theorem solve_for_x (x : ℝ) : (x - 55) / 3 = (2 - 3*x + x^2) / 4 → (x = 20 / 3 ∨ x = -11) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1180_118061


namespace NUMINAMATH_GPT_term_position_in_sequence_l1180_118008

theorem term_position_in_sequence (n : ℕ) (h1 : n > 0) (h2 : 3 * n + 1 = 40) : n = 13 :=
by
  sorry

end NUMINAMATH_GPT_term_position_in_sequence_l1180_118008


namespace NUMINAMATH_GPT_ajith_rana_meet_l1180_118002

/--
Ajith and Rana walk around a circular course 115 km in circumference, starting together from the same point.
Ajith walks at 4 km/h, and Rana walks at 5 km/h in the same direction.
Prove that they will meet after 115 hours.
-/
theorem ajith_rana_meet 
  (course_circumference : ℕ)
  (ajith_speed : ℕ)
  (rana_speed : ℕ)
  (relative_speed : ℕ)
  (time : ℕ)
  (start_point : Point)
  (ajith : Person)
  (rana : Person)
  (walk_in_same_direction : Prop)
  (start_time : ℕ)
  (meet_time : ℕ) :
  course_circumference = 115 →
  ajith_speed = 4 →
  rana_speed = 5 →
  relative_speed = rana_speed - ajith_speed →
  time = course_circumference / relative_speed →
  meet_time = start_time + time →
  meet_time = 115 :=
by
  sorry

end NUMINAMATH_GPT_ajith_rana_meet_l1180_118002


namespace NUMINAMATH_GPT_max_bishops_on_chessboard_l1180_118032

theorem max_bishops_on_chessboard : ∃ n : ℕ, n = 14 ∧ (∃ k : ℕ, n * n = k^2) := 
by {
  sorry
}

end NUMINAMATH_GPT_max_bishops_on_chessboard_l1180_118032


namespace NUMINAMATH_GPT_sin_of_right_triangle_l1180_118030

open Real

theorem sin_of_right_triangle (Q : ℝ) (h : 3 * sin Q = 4 * cos Q) : sin Q = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_of_right_triangle_l1180_118030


namespace NUMINAMATH_GPT_hexagon_ratio_l1180_118049

noncomputable def ratio_of_hexagon_areas (s : ℝ) : ℝ :=
  let area_ABCDEF := (3 * Real.sqrt 3 / 2) * s^2
  let side_smaller := (3 * s) / 2
  let area_smaller := (3 * Real.sqrt 3 / 2) * side_smaller^2
  area_smaller / area_ABCDEF

theorem hexagon_ratio (s : ℝ) : ratio_of_hexagon_areas s = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_ratio_l1180_118049


namespace NUMINAMATH_GPT_min_cuts_for_100_quadrilaterals_l1180_118080

theorem min_cuts_for_100_quadrilaterals : ∃ n : ℕ, (∃ q : ℕ, q = 100 ∧ n + 1 = q + 99) ∧ n = 1699 :=
sorry

end NUMINAMATH_GPT_min_cuts_for_100_quadrilaterals_l1180_118080


namespace NUMINAMATH_GPT_find_m_n_condition_l1180_118068

theorem find_m_n_condition (m n : ℕ) :
  m ≥ 1 ∧ n > m ∧ (42 ^ n ≡ 42 ^ m [MOD 100]) ∧ m + n = 24 :=
sorry

end NUMINAMATH_GPT_find_m_n_condition_l1180_118068


namespace NUMINAMATH_GPT_zero_in_interval_l1180_118058

noncomputable def f (x : ℝ) : ℝ := 2 * x - 8 + Real.logb 3 x

theorem zero_in_interval : 
  (0 < 3) ∧ (3 < 4) → (f 3 < 0) ∧ (f 4 > 0) → ∃ x, 3 < x ∧ x < 4 ∧ f x = 0 :=
by
  intro h1 h2
  obtain ⟨h3, h4⟩ := h2
  sorry

end NUMINAMATH_GPT_zero_in_interval_l1180_118058


namespace NUMINAMATH_GPT_ab_greater_than_a_plus_b_l1180_118067

theorem ab_greater_than_a_plus_b (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b :=
by
  sorry

end NUMINAMATH_GPT_ab_greater_than_a_plus_b_l1180_118067


namespace NUMINAMATH_GPT_find_missing_fraction_l1180_118025

theorem find_missing_fraction :
  ∃ (x : ℚ), (1/2 + -5/6 + 1/5 + 1/4 + -9/20 + -9/20 + x = 9/20) :=
  by
  sorry

end NUMINAMATH_GPT_find_missing_fraction_l1180_118025
