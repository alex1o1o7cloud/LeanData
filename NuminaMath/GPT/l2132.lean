import Mathlib

namespace NUMINAMATH_GPT_find_speed_l2132_213277

-- Definitions corresponding to conditions
def JacksSpeed (x : ℝ) : ℝ := x^2 - 7 * x - 12
def JillsDistance (x : ℝ) : ℝ := x^2 - 3 * x - 10
def JillsTime (x : ℝ) : ℝ := x + 2

-- Theorem statement
theorem find_speed (x : ℝ) (hx : x ≠ -2) (h_speed_eq : JacksSpeed x = (JillsDistance x) / (JillsTime x)) : JacksSpeed x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_l2132_213277


namespace NUMINAMATH_GPT_geometric_sequence_property_l2132_213252

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geom: ∀ n, a (n + 1) = a n * r) 
  (h_pos: ∀ n, a n > 0)
  (h_root1: a 3 * a 15 = 8)
  (h_root2: a 3 + a 15 = 6) :
  a 1 * a 17 / a 9 = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_property_l2132_213252


namespace NUMINAMATH_GPT_graph_of_equation_is_two_intersecting_lines_l2132_213219

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x - 2 * y)^2 = x^2 + y^2 ↔ (y = 0 ∨ y = 4 / 3 * x) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_intersecting_lines_l2132_213219


namespace NUMINAMATH_GPT_number_of_triangles_l2132_213235

-- Definition of given conditions
def original_wire_length : ℝ := 84
def remaining_wire_length : ℝ := 12
def wire_per_triangle : ℝ := 3

-- The goal is to prove that the number of triangles that can be made is 24
theorem number_of_triangles : (original_wire_length - remaining_wire_length) / wire_per_triangle = 24 := by
  sorry

end NUMINAMATH_GPT_number_of_triangles_l2132_213235


namespace NUMINAMATH_GPT_price_of_table_l2132_213289

-- Given the conditions:
def chair_table_eq1 (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
def chair_table_eq2 (C T : ℝ) : Prop := C + T = 72

-- Prove that the price of one table is $63
theorem price_of_table (C T : ℝ) (h1 : chair_table_eq1 C T) (h2 : chair_table_eq2 C T) : T = 63 := by
  sorry

end NUMINAMATH_GPT_price_of_table_l2132_213289


namespace NUMINAMATH_GPT_problem_solution_l2132_213220

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

theorem problem_solution : a^10 + b^10 = 123 := sorry

end NUMINAMATH_GPT_problem_solution_l2132_213220


namespace NUMINAMATH_GPT_number_division_l2132_213226

theorem number_division (x : ℤ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
by 
  sorry

end NUMINAMATH_GPT_number_division_l2132_213226


namespace NUMINAMATH_GPT_car_speed_kmph_l2132_213257

noncomputable def speed_of_car (d : ℝ) (t : ℝ) : ℝ :=
  (d / t) * 3.6

theorem car_speed_kmph : speed_of_car 10 0.9999200063994881 = 36000.29 := by
  sorry

end NUMINAMATH_GPT_car_speed_kmph_l2132_213257


namespace NUMINAMATH_GPT_expression_takes_many_different_values_l2132_213213

theorem expression_takes_many_different_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) : 
  ∃ v : ℝ, ∀ x, x ≠ 3 → x ≠ -2 → v = (3*x^2 - 2*x + 3)/((x - 3)*(x + 2)) - (5*x - 6)/((x - 3)*(x + 2)) := 
sorry

end NUMINAMATH_GPT_expression_takes_many_different_values_l2132_213213


namespace NUMINAMATH_GPT_candies_per_house_l2132_213230

theorem candies_per_house (candies_per_block : ℕ) (houses_per_block : ℕ) 
  (h1 : candies_per_block = 35) (h2 : houses_per_block = 5) :
  candies_per_block / houses_per_block = 7 := by
  sorry

end NUMINAMATH_GPT_candies_per_house_l2132_213230


namespace NUMINAMATH_GPT_find_length_of_c_find_measure_of_B_l2132_213217

-- Definition of the conditions
def triangle (A B C a b c : ℝ) : Prop :=
  c - b = 2 * b * Real.cos A

noncomputable def value_c (a b : ℝ) : ℝ := sorry

noncomputable def value_B (A B : ℝ) : ℝ := sorry

-- Statement for problem (I)
theorem find_length_of_c (a b : ℝ) (h1 : a = 2 * Real.sqrt 6) (h2 : b = 3) (h3 : ∀ A B C, triangle A B C a b (value_c a b)) : 
  value_c a b = 5 :=
by 
  sorry

-- Statement for problem (II)
theorem find_measure_of_B (B : ℝ) (h1 : ∀ A, A + B = Real.pi / 2) (h2 : B = value_B A B) : 
  value_B A B = Real.pi / 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_length_of_c_find_measure_of_B_l2132_213217


namespace NUMINAMATH_GPT_y1_increasing_on_0_1_l2132_213206

noncomputable def y1 (x : ℝ) : ℝ := |x|
noncomputable def y2 (x : ℝ) : ℝ := 3 - x
noncomputable def y3 (x : ℝ) : ℝ := 1 / x
noncomputable def y4 (x : ℝ) : ℝ := -x^2 + 4

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

theorem y1_increasing_on_0_1 :
  is_increasing_on y1 0 1 ∧
  ¬ is_increasing_on y2 0 1 ∧
  ¬ is_increasing_on y3 0 1 ∧
  ¬ is_increasing_on y4 0 1 :=
by
  sorry

end NUMINAMATH_GPT_y1_increasing_on_0_1_l2132_213206


namespace NUMINAMATH_GPT_find_F_l2132_213287

theorem find_F (C : ℝ) (F : ℝ) (h₁ : C = 35) (h₂ : C = 4 / 7 * (F - 40)) : F = 101.25 := by
  sorry

end NUMINAMATH_GPT_find_F_l2132_213287


namespace NUMINAMATH_GPT_joe_average_test_score_l2132_213293

theorem joe_average_test_score 
  (A B C : ℕ) 
  (Hsum : A + B + C = 135) 
  : (A + B + C + 25) / 4 = 40 :=
by
  sorry

end NUMINAMATH_GPT_joe_average_test_score_l2132_213293


namespace NUMINAMATH_GPT_growth_rate_of_yield_l2132_213225

-- Let x be the growth rate of the average yield per acre
variable (x : ℝ)

-- Initial conditions
def initial_acres := 10
def initial_yield := 20000
def final_yield := 60000

-- Relationship between the growth rates
def growth_relation := x * initial_acres * (1 + 2 * x) * (1 + x) = final_yield / initial_yield

theorem growth_rate_of_yield (h : growth_relation x) : x = 0.5 :=
  sorry

end NUMINAMATH_GPT_growth_rate_of_yield_l2132_213225


namespace NUMINAMATH_GPT_find_x_l2132_213281

def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

theorem find_x : ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2132_213281


namespace NUMINAMATH_GPT_largest_five_digit_palindromic_number_l2132_213255

def is_five_digit_palindrome (n : ℕ) : Prop := n / 10000 = n % 10 ∧ (n / 1000) % 10 = (n / 10) % 10

def is_four_digit_palindrome (n : ℕ) : Prop := n / 1000 = n % 10 ∧ (n / 100) % 10 = (n / 10) % 10

theorem largest_five_digit_palindromic_number :
  ∃ (abcba deed : ℕ), is_five_digit_palindrome abcba ∧ 10000 ≤ abcba ∧ abcba < 100000 ∧ is_four_digit_palindrome deed ∧ 1000 ≤ deed ∧ deed < 10000 ∧ abcba = 45 * deed ∧ abcba = 59895 :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_palindromic_number_l2132_213255


namespace NUMINAMATH_GPT_parabola_properties_l2132_213291

theorem parabola_properties :
  ∀ x : ℝ, (x - 3)^2 + 5 = (x-3)^2 + 5 ∧ 
  (x - 3)^2 + 5 > 0 ∧ 
  (∃ h : ℝ, h = 3 ∧ ∀ x1 x2 : ℝ, (x1 - h)^2 <= (x2 - h)^2) ∧ 
  (∃ h k : ℝ, h = 3 ∧ k = 5) := 
by 
  sorry

end NUMINAMATH_GPT_parabola_properties_l2132_213291


namespace NUMINAMATH_GPT_cookie_ratio_l2132_213273

theorem cookie_ratio (cookies_monday cookies_tuesday cookies_wednesday final_cookies : ℕ)
  (h1 : cookies_monday = 32)
  (h2 : cookies_tuesday = cookies_monday / 2)
  (h3 : final_cookies = 92)
  (h4 : cookies_wednesday = final_cookies + 4 - cookies_monday - cookies_tuesday) :
  cookies_wednesday / cookies_tuesday = 3 :=
by
  sorry

end NUMINAMATH_GPT_cookie_ratio_l2132_213273


namespace NUMINAMATH_GPT_sodium_chloride_formed_l2132_213244

section 

-- Definitions based on the conditions
def hydrochloric_acid_moles : ℕ := 2
def sodium_bicarbonate_moles : ℕ := 2

-- Balanced chemical equation represented as a function (1:1 reaction ratio)
def reaction (hcl_moles naHCO3_moles : ℕ) : ℕ := min hcl_moles naHCO3_moles

-- Theorem stating the reaction outcome
theorem sodium_chloride_formed : reaction hydrochloric_acid_moles sodium_bicarbonate_moles = 2 :=
by
  -- Proof is omitted
  sorry

end

end NUMINAMATH_GPT_sodium_chloride_formed_l2132_213244


namespace NUMINAMATH_GPT_number_of_tables_cost_price_l2132_213246

theorem number_of_tables_cost_price
  (C S : ℝ)
  (N : ℝ)
  (h1 : N * C = 20 * S)
  (h2 : S = 0.75 * C) :
  N = 15 := by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_number_of_tables_cost_price_l2132_213246


namespace NUMINAMATH_GPT_range_of_a_l2132_213299

theorem range_of_a (a : ℝ) (h : ¬ ∃ t : ℝ, t^2 - a * t - a < 0) : -4 ≤ a ∧ a ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l2132_213299


namespace NUMINAMATH_GPT_speedster_convertibles_l2132_213247

theorem speedster_convertibles 
  (T : ℕ) 
  (h1 : T > 0)
  (h2 : 30 = (2/3 : ℚ) * T)
  (h3 : ∀ n, n = (1/3 : ℚ) * T → ∃ m, m = (4/5 : ℚ) * n) :
  ∃ m, m = 12 := 
sorry

end NUMINAMATH_GPT_speedster_convertibles_l2132_213247


namespace NUMINAMATH_GPT_smallest_divisor_sum_of_squares_of_1_to_7_l2132_213218

def is_divisor (a b : ℕ) : Prop := ∃ k, b = k * a

theorem smallest_divisor_sum_of_squares_of_1_to_7 (S : ℕ) (h : S = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) :
  ∃ m, is_divisor m S ∧ (∀ d, is_divisor d S → 2 ≤ d) :=
sorry

end NUMINAMATH_GPT_smallest_divisor_sum_of_squares_of_1_to_7_l2132_213218


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2132_213261

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (h_arith : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) (h_incr : ∀ n, a (n + 1) > a n) (h_prod : a 4 * a 5 = 13) : a 3 * a 6 = -275 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2132_213261


namespace NUMINAMATH_GPT_fido_reach_fraction_simplified_l2132_213254

noncomputable def fidoReach (s r : ℝ) : ℝ :=
  let octagonArea := 2 * (1 + Real.sqrt 2) * s^2
  let circleArea := Real.pi * (s / Real.sqrt (2 + Real.sqrt 2))^2
  circleArea / octagonArea

theorem fido_reach_fraction_simplified (s : ℝ) :
  (∃ a b : ℕ, fidoReach s (s / Real.sqrt (2 + Real.sqrt 2)) = (Real.sqrt a / b) * Real.pi ∧ a * b = 16) :=
  sorry

end NUMINAMATH_GPT_fido_reach_fraction_simplified_l2132_213254


namespace NUMINAMATH_GPT_cut_difference_l2132_213204

-- define the conditions
def skirt_cut : ℝ := 0.75
def pants_cut : ℝ := 0.5

-- theorem to prove the correctness of the difference
theorem cut_difference : (skirt_cut - pants_cut = 0.25) :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_cut_difference_l2132_213204


namespace NUMINAMATH_GPT_probability_sector_F_l2132_213209

theorem probability_sector_F (prob_D prob_E prob_F : ℚ)
    (hD : prob_D = 1/4) 
    (hE : prob_E = 1/3) 
    (hSum : prob_D + prob_E + prob_F = 1) :
    prob_F = 5/12 := by
  sorry

end NUMINAMATH_GPT_probability_sector_F_l2132_213209


namespace NUMINAMATH_GPT_point_P_in_first_quadrant_l2132_213264

def pointInFirstQuadrant (x y : Int) : Prop := x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : pointInFirstQuadrant 2 3 :=
by
  sorry

end NUMINAMATH_GPT_point_P_in_first_quadrant_l2132_213264


namespace NUMINAMATH_GPT_zero_in_interval_l2132_213237

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 2 * x^2 - 4 * x

theorem zero_in_interval : ∃ (c : ℝ), 1 < c ∧ c < Real.exp 1 ∧ f c = 0 := sorry

end NUMINAMATH_GPT_zero_in_interval_l2132_213237


namespace NUMINAMATH_GPT_inequality_solution_l2132_213249

theorem inequality_solution (a b : ℝ) :
  (∀ x : ℝ, (-1/2 < x ∧ x < 2) → (ax^2 + bx + 2 > 0)) →
  a + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2132_213249


namespace NUMINAMATH_GPT_fencing_cost_l2132_213268

noncomputable def diameter : ℝ := 14
noncomputable def cost_per_meter : ℝ := 2.50
noncomputable def pi := Real.pi

noncomputable def circumference (d : ℝ) : ℝ := pi * d

noncomputable def total_cost (c : ℝ) (r : ℝ) : ℝ := r * c

theorem fencing_cost : total_cost (circumference diameter) cost_per_meter = 109.95 := by
  sorry

end NUMINAMATH_GPT_fencing_cost_l2132_213268


namespace NUMINAMATH_GPT_no_solutions_in_natural_numbers_l2132_213200

theorem no_solutions_in_natural_numbers (x y : ℕ) : x^2 + x * y + y^2 ≠ x^2 * y^2 :=
  sorry

end NUMINAMATH_GPT_no_solutions_in_natural_numbers_l2132_213200


namespace NUMINAMATH_GPT_total_ticket_income_l2132_213269

-- All given conditions as definitions/assumptions
def total_seats : ℕ := 200
def children_tickets : ℕ := 60
def adult_ticket_price : ℝ := 3.00
def children_ticket_price : ℝ := 1.50
def adult_tickets : ℕ := total_seats - children_tickets

-- The claim we need to prove
theorem total_ticket_income :
  (adult_tickets * adult_ticket_price + children_tickets * children_ticket_price) = 510.00 :=
by
  -- Placeholder to complete proof later
  sorry

end NUMINAMATH_GPT_total_ticket_income_l2132_213269


namespace NUMINAMATH_GPT_maximum_M_value_l2132_213276

noncomputable def max_value_of_M : ℝ :=
  Real.sqrt 2 + 1 

theorem maximum_M_value {x y z : ℝ} (hx : 0 ≤ x) (hx1 : x ≤ 1) (hy : 0 ≤ y) (hy1 : y ≤ 1) (hz : 0 ≤ z) (hz1 : z ≤ 1) :
  Real.sqrt (abs (x - y)) + Real.sqrt (abs (y - z)) + Real.sqrt (abs (z - x)) ≤ max_value_of_M :=
by
  sorry

end NUMINAMATH_GPT_maximum_M_value_l2132_213276


namespace NUMINAMATH_GPT_proof_problem_l2132_213205

variable (a b c : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem proof_problem 
  (h0 : f a b c 0 = f a b c 4)
  (h1 : f a b c 0 > f a b c 1) : 
  a > 0 ∧ 4 * a + b = 0 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2132_213205


namespace NUMINAMATH_GPT_setD_is_empty_l2132_213242

-- Definitions of sets A, B, C, D
def setA : Set ℝ := {x | x + 3 = 3}
def setB : Set (ℝ × ℝ) := {(x, y) | y^2 ≠ -x^2}
def setC : Set ℝ := {x | x^2 ≤ 0}
def setD : Set ℝ := {x | x^2 - x + 1 = 0}

-- Theorem stating that set D is the empty set
theorem setD_is_empty : setD = ∅ := 
by 
  sorry

end NUMINAMATH_GPT_setD_is_empty_l2132_213242


namespace NUMINAMATH_GPT_problem_thre_is_15_and_10_percent_l2132_213262

theorem problem_thre_is_15_and_10_percent (x y : ℝ) 
  (h1 : 3 = 0.15 * x) 
  (h2 : 3 = 0.10 * y) : 
  x - y = -10 := 
by 
  sorry

end NUMINAMATH_GPT_problem_thre_is_15_and_10_percent_l2132_213262


namespace NUMINAMATH_GPT_pages_left_to_read_l2132_213258

def total_pages : ℕ := 17
def pages_read : ℕ := 11

theorem pages_left_to_read : total_pages - pages_read = 6 := by
  sorry

end NUMINAMATH_GPT_pages_left_to_read_l2132_213258


namespace NUMINAMATH_GPT_license_plate_combinations_l2132_213229

open Nat

theorem license_plate_combinations : 
  (∃ (choose_two_letters: ℕ) (place_first_letter: ℕ) (place_second_letter: ℕ) (choose_non_repeated: ℕ)
     (first_digit: ℕ) (second_digit: ℕ) (third_digit: ℕ),
    choose_two_letters = choose 26 2 ∧
    place_first_letter = choose 5 2 ∧
    place_second_letter = choose 3 2 ∧
    choose_non_repeated = 24 ∧
    first_digit = 10 ∧
    second_digit = 9 ∧
    third_digit = 8 ∧
    choose_two_letters * place_first_letter * place_second_letter * choose_non_repeated * first_digit * second_digit * third_digit = 56016000) :=
sorry

end NUMINAMATH_GPT_license_plate_combinations_l2132_213229


namespace NUMINAMATH_GPT_manuscript_typing_cost_l2132_213288

-- Defining the conditions as per our problem
def first_time_typing_rate : ℕ := 5 -- $5 per page for first-time typing
def revision_rate : ℕ := 3 -- $3 per page per revision

def num_pages : ℕ := 100 -- total number of pages
def revised_once : ℕ := 30 -- number of pages revised once
def revised_twice : ℕ := 20 -- number of pages revised twice
def no_revision := num_pages - (revised_once + revised_twice) -- pages with no revisions

-- Defining the cost function to calculate the total cost of typing
noncomputable def total_typing_cost : ℕ :=
  (num_pages * first_time_typing_rate) + (revised_once * revision_rate) + (revised_twice * revision_rate * 2)

-- Lean theorem statement to prove the total cost is $710
theorem manuscript_typing_cost :
  total_typing_cost = 710 := by
  sorry

end NUMINAMATH_GPT_manuscript_typing_cost_l2132_213288


namespace NUMINAMATH_GPT_circle_equation_l2132_213260

theorem circle_equation
  (a b r : ℝ)
  (ha : (4 - a)^2 + (1 - b)^2 = r^2)
  (hb : (2 - a)^2 + (1 - b)^2 = r^2)
  (ht : (b - 1) / (a - 2) = -1) :
  (a = 3) ∧ (b = 0) ∧ (r = 2) :=
by {
  sorry
}

-- Given the above values for a, b, r
def circle_equation_verified : Prop :=
  (∀ (x y : ℝ), ((x - 3)^2 + y^2) = 4)

example : circle_equation_verified :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_equation_l2132_213260


namespace NUMINAMATH_GPT_eight_div_repeating_three_l2132_213296

theorem eight_div_repeating_three : (8 / (1 / 3)) = 24 := by
  sorry

end NUMINAMATH_GPT_eight_div_repeating_three_l2132_213296


namespace NUMINAMATH_GPT_neg_three_is_square_mod_p_l2132_213224

theorem neg_three_is_square_mod_p (q : ℤ) (p : ℕ) (prime_p : Nat.Prime p) (condition : p = 3 * q + 1) :
  ∃ x : ℤ, (x^2 ≡ -3 [ZMOD p]) :=
sorry

end NUMINAMATH_GPT_neg_three_is_square_mod_p_l2132_213224


namespace NUMINAMATH_GPT_alcohol_percentage_l2132_213215

theorem alcohol_percentage (P : ℝ) : 
  (0.10 * 300) + (P / 100 * 450) = 0.22 * 750 → P = 30 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_alcohol_percentage_l2132_213215


namespace NUMINAMATH_GPT_remainder_7325_mod_11_l2132_213265

theorem remainder_7325_mod_11 : 7325 % 11 = 6 := sorry

end NUMINAMATH_GPT_remainder_7325_mod_11_l2132_213265


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l2132_213294

theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd 279 372) 465 = 93 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l2132_213294


namespace NUMINAMATH_GPT_original_price_of_wand_l2132_213203

theorem original_price_of_wand (P : ℝ) (h1 : 8 = P / 8) : P = 64 :=
by sorry

end NUMINAMATH_GPT_original_price_of_wand_l2132_213203


namespace NUMINAMATH_GPT_sum_of_intercepts_l2132_213210

theorem sum_of_intercepts (x y : ℝ) 
  (h_eq : y - 3 = -3 * (x - 5)) 
  (hx_intercept : y = 0 ∧ x = 6) 
  (hy_intercept : x = 0 ∧ y = 18) : 
  6 + 18 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_intercepts_l2132_213210


namespace NUMINAMATH_GPT_time_to_fill_one_barrel_with_leak_l2132_213284

-- Define the conditions
def normal_time_per_barrel := 3
def time_to_fill_12_barrels_no_leak := normal_time_per_barrel * 12
def additional_time_due_to_leak := 24
def time_to_fill_12_barrels_with_leak (t : ℕ) := 12 * t

-- Define the theorem
theorem time_to_fill_one_barrel_with_leak :
  ∃ t : ℕ, time_to_fill_12_barrels_with_leak t = time_to_fill_12_barrels_no_leak + additional_time_due_to_leak ∧ t = 5 :=
by {
  use 5, 
  sorry
}

end NUMINAMATH_GPT_time_to_fill_one_barrel_with_leak_l2132_213284


namespace NUMINAMATH_GPT_no_zero_root_l2132_213245

theorem no_zero_root (x : ℝ) :
  (¬ (∃ x : ℝ, (4 * x ^ 2 - 3 = 49) ∧ x = 0)) ∧
  (¬ (∃ x : ℝ, (x ^ 2 - x - 20 = 0) ∧ x = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_zero_root_l2132_213245


namespace NUMINAMATH_GPT_focus_of_parabola_y_eq_x_sq_l2132_213263

theorem focus_of_parabola_y_eq_x_sq : ∃ (f : ℝ × ℝ), f = (0, 1/4) ∧ (∃ (p : ℝ), p = 1/2 ∧ ∀ x, y = x^2 → y = 2 * p * (0, y).snd) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_y_eq_x_sq_l2132_213263


namespace NUMINAMATH_GPT_division_by_3_l2132_213222

theorem division_by_3 (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := 
sorry

end NUMINAMATH_GPT_division_by_3_l2132_213222


namespace NUMINAMATH_GPT_find_a8_a12_l2132_213275

noncomputable def geometric_sequence_value_8_12 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then a 0 else a 0 * q^n

theorem find_a8_a12 (a : ℕ → ℝ) (q : ℝ) (terms_geometric : ∀ n, a n = a 0 * q^n)
  (h2_6 : a 2 + a 6 = 3) (h6_10 : a 6 + a 10 = 12) :
  a 8 + a 12 = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_a8_a12_l2132_213275


namespace NUMINAMATH_GPT_maximum_profit_at_110_l2132_213251

noncomputable def profit (x : ℕ) : ℝ := 
if x > 0 ∧ x < 100 then 
  -0.5 * (x : ℝ)^2 + 90 * (x : ℝ) - 600 
else if x ≥ 100 then 
  -2 * (x : ℝ) - 24200 / (x : ℝ) + 4100 
else 
  0 -- To ensure totality, although this won't match the problem's condition that x is always positive

theorem maximum_profit_at_110 :
  ∃ (y_max : ℝ), ∀ (x : ℕ), profit 110 = y_max ∧ (∀ x ≠ 0, profit 110 ≥ profit x) :=
sorry

end NUMINAMATH_GPT_maximum_profit_at_110_l2132_213251


namespace NUMINAMATH_GPT_stationary_train_length_l2132_213232

-- Definitions
def speed_km_per_h := 72
def speed_m_per_s := speed_km_per_h * (1000 / 3600) -- conversion from km/h to m/s
def time_to_pass_pole := 10 -- in seconds
def time_to_cross_stationary_train := 35 -- in seconds
def speed := 20 -- speed in m/s, 72 km/h = 20 m/s, can be inferred from conversion

-- Length of moving train
def length_of_moving_train := speed * time_to_pass_pole

-- Total distance in crossing stationary train
def total_distance := speed * time_to_cross_stationary_train

-- Length of stationary train
def length_of_stationary_train := total_distance - length_of_moving_train

-- Proof statement
theorem stationary_train_length :
  length_of_stationary_train = 500 := by
  sorry

end NUMINAMATH_GPT_stationary_train_length_l2132_213232


namespace NUMINAMATH_GPT_production_rate_is_constant_l2132_213272

def drum_rate := 6 -- drums per day

def days_needed_to_produce (n : ℕ) : ℕ := n / drum_rate

theorem production_rate_is_constant (n : ℕ) : days_needed_to_produce n = n / drum_rate :=
by
  sorry

end NUMINAMATH_GPT_production_rate_is_constant_l2132_213272


namespace NUMINAMATH_GPT_marbles_leftover_l2132_213236

theorem marbles_leftover (r p g : ℕ) (hr : r % 7 = 5) (hp : p % 7 = 4) (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_marbles_leftover_l2132_213236


namespace NUMINAMATH_GPT_greatest_triangle_perimeter_l2132_213202

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end NUMINAMATH_GPT_greatest_triangle_perimeter_l2132_213202


namespace NUMINAMATH_GPT_graveling_cost_is_correct_l2132_213240

noncomputable def graveling_cost (lawn_length lawn_breadth road_width cost_per_sqm : ℝ) : ℝ :=
  let road1_area := road_width * lawn_breadth
  let road2_area := road_width * lawn_length
  let intersection_area := road_width * road_width
  let total_area := road1_area + road2_area - intersection_area
  total_area * cost_per_sqm

theorem graveling_cost_is_correct :
  graveling_cost 80 60 10 2 = 2600 := by
  sorry

end NUMINAMATH_GPT_graveling_cost_is_correct_l2132_213240


namespace NUMINAMATH_GPT_f_of_f_3_eq_3_l2132_213201

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 1 - Real.logb 2 (2 - x) else 2^(1 - x) + 3 / 2

theorem f_of_f_3_eq_3 : f (f 3) = 3 := by
  sorry

end NUMINAMATH_GPT_f_of_f_3_eq_3_l2132_213201


namespace NUMINAMATH_GPT_minimum_value_of_a_l2132_213295

def is_prime (n : ℕ) : Prop := sorry  -- Provide the definition of a prime number

def is_perfect_square (n : ℕ) : Prop := sorry  -- Provide the definition of a perfect square

theorem minimum_value_of_a 
  (a b : ℕ) 
  (h1 : is_prime (a - b)) 
  (h2 : is_perfect_square (a * b)) 
  (h3 : a ≥ 2012) : 
  a = 2025 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l2132_213295


namespace NUMINAMATH_GPT_expression_range_l2132_213248

theorem expression_range (a b c d : ℝ) 
    (ha : 0 ≤ a) (ha' : a ≤ 2)
    (hb : 0 ≤ b) (hb' : b ≤ 2)
    (hc : 0 ≤ c) (hc' : c ≤ 2)
    (hd : 0 ≤ d) (hd' : d ≤ 2) :
  4 + 2 * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) 
  ∧ Real.sqrt (a^2 + (2-b)^2) 
    + Real.sqrt (b^2 + (2-c)^2) 
    + Real.sqrt (c^2 + (2-d)^2) 
    + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := 
sorry

end NUMINAMATH_GPT_expression_range_l2132_213248


namespace NUMINAMATH_GPT_hyperbola_focal_length_l2132_213286

def is_hyperbola (x y a : ℝ) : Prop := (x^2) / (a^2) - (y^2) = 1
def is_perpendicular_asymptote (slope_asymptote slope_line : ℝ) : Prop := slope_asymptote * slope_line = -1

theorem hyperbola_focal_length {a : ℝ} (h1 : is_hyperbola x y a)
  (h2 : is_perpendicular_asymptote (1 / a) (-1)) : 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l2132_213286


namespace NUMINAMATH_GPT_james_driving_speed_l2132_213282

theorem james_driving_speed
  (distance : ℝ)
  (total_time : ℝ)
  (stop_time : ℝ)
  (driving_time : ℝ)
  (speed : ℝ)
  (h1 : distance = 360)
  (h2 : total_time = 7)
  (h3 : stop_time = 1)
  (h4 : driving_time = total_time - stop_time)
  (h5 : speed = distance / driving_time) :
  speed = 60 := by
  -- Here you would put the detailed proof.
  sorry

end NUMINAMATH_GPT_james_driving_speed_l2132_213282


namespace NUMINAMATH_GPT_min_value_of_exponential_l2132_213290

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 1) : 
  2^x + 4^y ≥ 2 * Real.sqrt 2 ∧ 
  (∀ a, (2^x + 4^y = a) → a ≥ 2 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_exponential_l2132_213290


namespace NUMINAMATH_GPT_slope_of_line_between_solutions_l2132_213231

theorem slope_of_line_between_solutions (x1 y1 x2 y2 : ℝ) (h1 : 3 / x1 + 4 / y1 = 0) (h2 : 3 / x2 + 4 / y2 = 0) (h3 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -4 / 3 := 
sorry

end NUMINAMATH_GPT_slope_of_line_between_solutions_l2132_213231


namespace NUMINAMATH_GPT_smallest_x_l2132_213228

theorem smallest_x (x : ℕ) : (x + 3457) % 15 = 1537 % 15 → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l2132_213228


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2132_213221

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h1 : a 1 + a 2 = -1)
  (h2 : a 3 = 4)
  (h3 : ∀ n, a (n + 1) - a n = a 2 - a 1) :
  a 4 + a 5 = 17 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2132_213221


namespace NUMINAMATH_GPT_number_of_children_l2132_213208

theorem number_of_children (total_passengers men women : ℕ) (h1 : total_passengers = 54) (h2 : men = 18) (h3 : women = 26) : 
  total_passengers - men - women = 10 :=
by sorry

end NUMINAMATH_GPT_number_of_children_l2132_213208


namespace NUMINAMATH_GPT_gcd_all_abc_plus_cba_l2132_213292

noncomputable def gcd_of_abc_cba (a : ℕ) (b : ℕ := 2 * a) (c : ℕ := 3 * a) : ℕ :=
  let abc := 64 * a + 8 * b + c
  let cba := 64 * c + 8 * b + a
  Nat.gcd (abc + cba) 300

theorem gcd_all_abc_plus_cba (a : ℕ) : gcd_of_abc_cba a = 300 :=
  sorry

end NUMINAMATH_GPT_gcd_all_abc_plus_cba_l2132_213292


namespace NUMINAMATH_GPT_parabola_constant_unique_l2132_213278

theorem parabola_constant_unique (b c : ℝ) :
  (∀ x y : ℝ, (x = 2 ∧ y = 20) → y = x^2 + b * x + c) →
  (∀ x y : ℝ, (x = -2 ∧ y = -4) → y = x^2 + b * x + c) →
  c = 4 :=
by
    sorry

end NUMINAMATH_GPT_parabola_constant_unique_l2132_213278


namespace NUMINAMATH_GPT_third_measurement_multiple_of_one_l2132_213283

-- Define the lengths in meters
def length1_meter : ℕ := 6
def length2_meter : ℕ := 5

-- Convert lengths to centimeters
def length1_cm := length1_meter * 100
def length2_cm := length2_meter * 100

-- Define that the greatest common divisor (gcd) of lengths in cm is 100 cm
def gcd_length : ℕ := Nat.gcd length1_cm length2_cm

-- Given that the gcd is 100 cm
theorem third_measurement_multiple_of_one
  (h1 : gcd_length = 100) :
  ∃ n : ℕ, n = 1 :=
sorry

end NUMINAMATH_GPT_third_measurement_multiple_of_one_l2132_213283


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2132_213211

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 1) * (2 - x) > 0} = {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2132_213211


namespace NUMINAMATH_GPT_mason_car_nuts_l2132_213243

def busy_squirrels_num := 2
def busy_squirrel_nuts_per_day := 30
def sleepy_squirrel_num := 1
def sleepy_squirrel_nuts_per_day := 20
def days := 40

theorem mason_car_nuts : 
  busy_squirrels_num * busy_squirrel_nuts_per_day * days + sleepy_squirrel_nuts_per_day * days = 3200 :=
  by
    sorry

end NUMINAMATH_GPT_mason_car_nuts_l2132_213243


namespace NUMINAMATH_GPT_range_of_m_l2132_213241

noncomputable def f (x : ℝ) : ℝ :=
  if x < -2 then 3 + 3 * x
  else if x <= 3 then -1
  else x + 5

theorem range_of_m (m : ℝ) (x : ℝ) (hx : f x ≥ 1 / m - 4) :
  m < 0 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2132_213241


namespace NUMINAMATH_GPT_total_time_equiv_l2132_213280

-- Define the number of chairs
def chairs := 7

-- Define the number of tables
def tables := 3

-- Define the time spent on each piece of furniture in minutes
def time_per_piece := 4

-- Prove the total time taken to assemble all furniture
theorem total_time_equiv : chairs + tables = 10 ∧ 4 * 10 = 40 := by
  sorry

end NUMINAMATH_GPT_total_time_equiv_l2132_213280


namespace NUMINAMATH_GPT_more_karabases_than_barabases_l2132_213239

/-- In the fairy-tale land of Perra-Terra, each Karabas is acquainted with nine Barabases, 
    and each Barabas is acquainted with ten Karabases. We aim to prove that there are more Karabases than Barabases. -/
theorem more_karabases_than_barabases (K B : ℕ) (h1 : 9 * K = 10 * B) : K > B := 
by {
    -- Following the conditions and conclusion
    sorry
}

end NUMINAMATH_GPT_more_karabases_than_barabases_l2132_213239


namespace NUMINAMATH_GPT_find_ab_l2132_213297

theorem find_ab (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 30) : a * b = 32 :=
by
  -- We will complete the proof in this space
  sorry

end NUMINAMATH_GPT_find_ab_l2132_213297


namespace NUMINAMATH_GPT_shifted_function_correct_l2132_213253

variable (x : ℝ)

/-- The original function -/
def original_function : ℝ := 3 * x - 4

/-- The function after shifting up by 2 units -/
def shifted_function : ℝ := original_function x + 2

theorem shifted_function_correct :
  shifted_function x = 3 * x - 2 :=
by
  sorry

end NUMINAMATH_GPT_shifted_function_correct_l2132_213253


namespace NUMINAMATH_GPT_repeating_decimal_subtraction_l2132_213207

noncomputable def x := (0.246 : Real)
noncomputable def y := (0.135 : Real)
noncomputable def z := (0.579 : Real)

theorem repeating_decimal_subtraction :
  x - y - z = (-156 : ℚ) / 333 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_subtraction_l2132_213207


namespace NUMINAMATH_GPT_not_set_of_difficult_problems_l2132_213223

-- Define the context and entities
inductive Exercise
| ex (n : Nat) : Exercise  -- Example definition for exercises, assumed to be numbered

def is_difficult (ex : Exercise) : Prop := sorry  -- Placeholder for the subjective predicate

-- Define the main problem statement
theorem not_set_of_difficult_problems
  (Difficult : Exercise → Prop) -- Subjective predicate defining difficult problems
  (H_subj : ∀ (e : Exercise), (Difficult e ↔ is_difficult e)) :
  ¬(∃ (S : Set Exercise), ∀ e, e ∈ S ↔ Difficult e) :=
sorry

end NUMINAMATH_GPT_not_set_of_difficult_problems_l2132_213223


namespace NUMINAMATH_GPT_area_hexagon_STUVWX_l2132_213256

noncomputable def area_of_hexagon (area_PQR : ℕ) (small_area : ℕ) : ℕ := 
  area_PQR - (3 * small_area)

theorem area_hexagon_STUVWX : 
  let area_PQR := 45
  let small_area := 1 
  ∃ area_hexagon, area_hexagon = 42 := 
by
  let area_PQR := 45
  let small_area := 1
  let area_hexagon := area_of_hexagon area_PQR small_area
  use area_hexagon
  sorry

end NUMINAMATH_GPT_area_hexagon_STUVWX_l2132_213256


namespace NUMINAMATH_GPT_probability_of_all_co_captains_l2132_213227

def team_sizes : List ℕ := [6, 8, 9, 10]

def captains_per_team : ℕ := 3

noncomputable def probability_all_co_captains (s : ℕ) : ℚ :=
  1 / (Nat.choose s 3 : ℚ)

noncomputable def total_probability : ℚ :=
  (1 / 4 : ℚ) * 
  (probability_all_co_captains 6 + 
   probability_all_co_captains 8 +
   probability_all_co_captains 9 +
   probability_all_co_captains 10)

theorem probability_of_all_co_captains : total_probability = 1 / 84 :=
  sorry

end NUMINAMATH_GPT_probability_of_all_co_captains_l2132_213227


namespace NUMINAMATH_GPT_least_positive_multiple_of_24_gt_450_l2132_213267

theorem least_positive_multiple_of_24_gt_450 : 
  ∃ n : ℕ, n > 450 ∧ (∃ k : ℕ, n = 24 * k) → n = 456 :=
by 
  sorry

end NUMINAMATH_GPT_least_positive_multiple_of_24_gt_450_l2132_213267


namespace NUMINAMATH_GPT_polygon_sides_l2132_213270

theorem polygon_sides (x : ℕ) 
  (h1 : 180 * (x - 2) = 3 * 360) 
  : x = 8 := 
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l2132_213270


namespace NUMINAMATH_GPT_bread_consumption_l2132_213274

-- Definitions using conditions
def members := 4
def slices_snacks := 2
def slices_per_loaf := 12
def total_loaves := 5
def total_days := 3

-- The main theorem to prove
theorem bread_consumption :
  (3 * members * (B + slices_snacks) = total_loaves * slices_per_loaf) → B = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_bread_consumption_l2132_213274


namespace NUMINAMATH_GPT_highest_place_joker_can_achieve_is_6_l2132_213233

-- Define the total number of teams
def total_teams : ℕ := 16

-- Define conditions for points in football
def points_win : ℕ := 3
def points_draw : ℕ := 1
def points_loss : ℕ := 0

-- Condition definitions for Joker's performance in the tournament
def won_against_strong_teams (j k : ℕ) : Prop := j < k
def lost_against_weak_teams (j k : ℕ) : Prop := j > k

-- Define the performance of all teams
def teams (t : ℕ) := {n // n < total_teams}

-- Function to calculate Joker's points based on position k
def joker_points (k : ℕ) : ℕ := (total_teams - k) * points_win

theorem highest_place_joker_can_achieve_is_6 : ∃ k, k = 6 ∧ 
  (∀ j, 
    (j < k → won_against_strong_teams j k) ∧ 
    (j > k → lost_against_weak_teams j k) ∧
    (∃! p, p = joker_points k)) :=
by
  sorry

end NUMINAMATH_GPT_highest_place_joker_can_achieve_is_6_l2132_213233


namespace NUMINAMATH_GPT_volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l2132_213214

noncomputable def volume_of_reservoir (drain_rate : ℝ) (time_to_drain : ℝ) : ℝ :=
  drain_rate * time_to_drain

theorem volume_of_reservoir_proof :
  volume_of_reservoir 8 6 = 48 :=
by
  sorry

noncomputable def relationship_Q_t (volume : ℝ) (t : ℝ) : ℝ :=
  volume / t

theorem relationship_Q_t_proof :
  ∀ (t : ℝ), relationship_Q_t 48 t = 48 / t :=
by
  intro t
  sorry

noncomputable def min_hourly_drainage (volume : ℝ) (time : ℝ) : ℝ :=
  volume / time

theorem min_hourly_drainage_proof :
  min_hourly_drainage 48 5 = 9.6 :=
by
  sorry

theorem min_time_to_drain_proof :
  ∀ (max_capacity : ℝ), relationship_Q_t 48 max_capacity = 12 → 48 / 12 = 4 :=
by
  intro max_capacity h
  sorry

end NUMINAMATH_GPT_volume_of_reservoir_proof_relationship_Q_t_proof_min_hourly_drainage_proof_min_time_to_drain_proof_l2132_213214


namespace NUMINAMATH_GPT_hundredth_term_sequence_l2132_213259

def numerators (n : ℕ) : ℕ := 1 + (n - 1) * 2
def denominators (n : ℕ) : ℕ := 2 + (n - 1) * 3

theorem hundredth_term_sequence : numerators 100 / denominators 100 = 199 / 299 := by
  sorry

end NUMINAMATH_GPT_hundredth_term_sequence_l2132_213259


namespace NUMINAMATH_GPT_complex_number_quadrant_l2132_213279

theorem complex_number_quadrant :
  let z := (2 * Complex.I) / (1 - Complex.I)
  Complex.re z < 0 ∧ Complex.im z > 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_number_quadrant_l2132_213279


namespace NUMINAMATH_GPT_line_tangent_to_circle_l2132_213250

theorem line_tangent_to_circle (x y : ℝ) :
  (3 * x - 4 * y + 25 = 0) ∧ (x^2 + y^2 = 25) → (x = -3 ∧ y = 4) :=
by sorry

end NUMINAMATH_GPT_line_tangent_to_circle_l2132_213250


namespace NUMINAMATH_GPT_largest_root_of_quadratic_l2132_213271

theorem largest_root_of_quadratic :
  ∀ (x : ℝ), x^2 - 9*x - 22 = 0 → x ≤ 11 :=
by
  sorry

end NUMINAMATH_GPT_largest_root_of_quadratic_l2132_213271


namespace NUMINAMATH_GPT_flowers_are_55_percent_daisies_l2132_213298

noncomputable def percent_daisies (F : ℝ) (yellow : ℝ) (white_daisies : ℝ) (yellow_daisies : ℝ) : ℝ :=
  (yellow_daisies + white_daisies) / F * 100

theorem flowers_are_55_percent_daisies (F : ℝ) (yellow_t : ℝ) (yellow_d : ℝ) (white : ℝ) (white_d : ℝ) :
    yellow_t = 0.5 * yellow →
    yellow_d = yellow - yellow_t →
    white_d = (2 / 3) * white →
    yellow = (7 / 10) * F →
    white = F - yellow →
    percent_daisies F yellow white_d yellow_d = 55 :=
by
  sorry

end NUMINAMATH_GPT_flowers_are_55_percent_daisies_l2132_213298


namespace NUMINAMATH_GPT_bankers_discount_is_correct_l2132_213266

-- Define the given conditions
def TD := 45   -- True discount in Rs.
def FV := 270  -- Face value in Rs.

-- Calculate Present Value based on the given conditions
def PV := FV - TD

-- Define the formula for Banker's Discount
def BD := TD + (TD ^ 2 / PV)

-- Prove that the Banker's Discount is Rs. 54 given the conditions
theorem bankers_discount_is_correct : BD = 54 :=
by
  -- Steps to prove the theorem can be filled here
  -- Add "sorry" to skip the actual proof
  sorry

end NUMINAMATH_GPT_bankers_discount_is_correct_l2132_213266


namespace NUMINAMATH_GPT_lowest_dropped_score_l2132_213234

theorem lowest_dropped_score (A B C D : ℕ) 
  (h1 : (A + B + C + D) / 4 = 90)
  (h2 : (A + B + C) / 3 = 85) :
  D = 105 :=
by
  sorry

end NUMINAMATH_GPT_lowest_dropped_score_l2132_213234


namespace NUMINAMATH_GPT_max_sum_of_multiplication_table_l2132_213216

-- Define primes and their sums
def primes : List ℕ := [2, 3, 5, 7, 17, 19]

noncomputable def sum_primes := primes.sum -- 2 + 3 + 5 + 7 + 17 + 19 = 53

-- Define two groups of primes to maximize the product of their sums
def group1 : List ℕ := [2, 3, 17]
def group2 : List ℕ := [5, 7, 19]

noncomputable def sum_group1 := group1.sum -- 2 + 3 + 17 = 22
noncomputable def sum_group2 := group2.sum -- 5 + 7 + 19 = 31

-- Formulate the proof problem
theorem max_sum_of_multiplication_table : 
  ∃ a b c d e f : ℕ, 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) ∧ 
    (a ∈ primes ∧ b ∈ primes ∧ c ∈ primes ∧ d ∈ primes ∧ e ∈ primes ∧ f ∈ primes) ∧ 
    (a + b + c = sum_group1 ∨ a + b + c = sum_group2) ∧ 
    (d + e + f = sum_group1 ∨ d + e + f = sum_group2) ∧ 
    (a + b + c) ≠ (d + e + f) ∧ 
    ((a + b + c) * (d + e + f) = 682) := 
by
  use 2, 3, 17, 5, 7, 19
  sorry

end NUMINAMATH_GPT_max_sum_of_multiplication_table_l2132_213216


namespace NUMINAMATH_GPT_coordinates_of_P_with_respect_to_y_axis_l2132_213212

-- Define the coordinates of point P
def P_x : ℝ := 5
def P_y : ℝ := -1

-- Define the point P
def P : Prod ℝ ℝ := (P_x, P_y)

-- State the theorem
theorem coordinates_of_P_with_respect_to_y_axis :
  (P.1, P.2) = (-P_x, P_y) :=
sorry

end NUMINAMATH_GPT_coordinates_of_P_with_respect_to_y_axis_l2132_213212


namespace NUMINAMATH_GPT_expected_value_correct_l2132_213285

-- Define the probability distribution of the user's score in the first round
noncomputable def first_round_prob (X : ℕ) : ℚ :=
  if X = 3 then 1 / 4
  else if X = 2 then 1 / 2
  else if X = 1 then 1 / 4
  else 0

-- Define the conditional probability of the user's score in the second round given the first round score
noncomputable def second_round_prob (X Y : ℕ) : ℚ :=
  if X = 3 then
    if Y = 2 then 1 / 5
    else if Y = 1 then 4 / 5
    else 0
  else
    if Y = 2 then 1 / 3
    else if Y = 1 then 2 / 3
    else 0

-- Define the total score probability
noncomputable def total_score_prob (X Y : ℕ) : ℚ :=
  first_round_prob X * second_round_prob X Y

-- Compute the expected value of the user's total score
noncomputable def expected_value : ℚ :=
  (5 * (total_score_prob 3 2) +
   4 * (total_score_prob 3 1 + total_score_prob 2 2) +
   3 * (total_score_prob 2 1 + total_score_prob 1 2) +
   2 * (total_score_prob 1 1))

-- The theorem to be proven
theorem expected_value_correct : expected_value = 3.3 := 
by sorry

end NUMINAMATH_GPT_expected_value_correct_l2132_213285


namespace NUMINAMATH_GPT_find_sets_l2132_213238

open Set

noncomputable def U := ℝ
def A := {x : ℝ | Real.log x / Real.log 2 <= 2}
def B := {x : ℝ | x ≥ 1}

theorem find_sets (x : ℝ) :
  (A = {x : ℝ | -1 ≤ x ∧ x < 3}) ∧
  (B = {x : ℝ | -2 < x ∧ x ≤ 3}) ∧
  (compl A ∩ B = {x : ℝ | (-2 < x ∧ x < -1) ∨ x = 3}) :=
  sorry

end NUMINAMATH_GPT_find_sets_l2132_213238
