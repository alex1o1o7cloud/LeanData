import Mathlib

namespace daisy_germination_rate_theorem_l2176_217673

-- Define the conditions of the problem
variables (daisySeeds sunflowerSeeds : ℕ) (sunflowerGermination flowerProduction finalFlowerPlants : ℝ)
def conditions : Prop :=
  daisySeeds = 25 ∧ sunflowerSeeds = 25 ∧ sunflowerGermination = 0.80 ∧ flowerProduction = 0.80 ∧ finalFlowerPlants = 28

-- Define the statement that the germination rate of the daisy seeds is 60%
def germination_rate_of_daisy_seeds : Prop :=
  ∃ (daisyGerminationRate : ℝ), (conditions daisySeeds sunflowerSeeds sunflowerGermination flowerProduction finalFlowerPlants) →
  daisyGerminationRate = 0.60

-- The proof is omitted - note this is just the statement
theorem daisy_germination_rate_theorem : germination_rate_of_daisy_seeds 25 25 0.80 0.80 28 :=
sorry

end daisy_germination_rate_theorem_l2176_217673


namespace number_of_dogs_l2176_217663

theorem number_of_dogs (cost_price selling_price total_amount : ℝ) (profit_percentage : ℝ)
    (h1 : cost_price = 1000)
    (h2 : profit_percentage = 0.30)
    (h3 : total_amount = 2600)
    (h4 : selling_price = cost_price + (profit_percentage * cost_price)) :
    total_amount / selling_price = 2 :=
by
  sorry

end number_of_dogs_l2176_217663


namespace xiao_gao_actual_score_l2176_217697

-- Definitions from the conditions:
def standard_score : ℕ := 80
def xiao_gao_recorded_score : ℤ := 12

-- Proof problem statement:
theorem xiao_gao_actual_score : (standard_score : ℤ) + xiao_gao_recorded_score = 92 :=
by
  sorry

end xiao_gao_actual_score_l2176_217697


namespace count_pos_integers_three_digits_l2176_217675

/-- The number of positive integers less than 50,000 having at most three distinct digits equals 7862. -/
theorem count_pos_integers_three_digits : 
  ∃ n : ℕ, n < 50000 ∧ (∀ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∨ d1 ≠ d3 ∨ d1 ≠ d4 ∨ d1 ≠ d5 ∨ d2 ≠ d3 ∨ d2 ≠ d4 ∨ d2 ≠ d5 ∨ d3 ≠ d4 ∨ d3 ≠ d5 ∨ d4 ≠ d5) ∧ n = 7862 :=
sorry

end count_pos_integers_three_digits_l2176_217675


namespace correct_sequence_of_linear_regression_analysis_l2176_217671

def linear_regression_steps : List ℕ := [2, 4, 3, 1]

theorem correct_sequence_of_linear_regression_analysis :
  linear_regression_steps = [2, 4, 3, 1] :=
by
  sorry

end correct_sequence_of_linear_regression_analysis_l2176_217671


namespace balance_scale_weights_part_a_balance_scale_weights_part_b_l2176_217616

-- Part (a)
theorem balance_scale_weights_part_a (w : List ℕ) (h : w = List.range (90 + 1) \ List.range 1) :
  ¬ ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

-- Part (b)
theorem balance_scale_weights_part_b (w : List ℕ) (h : w = List.range (99 + 1) \ List.range 1) :
  ∃ (A B : List ℕ), A.length = 2 * B.length ∧ A.sum = B.sum :=
sorry

end balance_scale_weights_part_a_balance_scale_weights_part_b_l2176_217616


namespace max_min_value_function_l2176_217664

noncomputable def given_function (x : ℝ) : ℝ :=
  (Real.sin x) ^ 2 + Real.cos x + 1

theorem max_min_value_function :
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≤ 9 / 4) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 9 / 4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 2) := by
  sorry

end max_min_value_function_l2176_217664


namespace ensemble_average_age_l2176_217661

theorem ensemble_average_age (female_avg_age : ℝ) (num_females : ℕ) (male_avg_age : ℝ) (num_males : ℕ)
  (h1 : female_avg_age = 32) (h2 : num_females = 12) (h3 : male_avg_age = 40) (h4 : num_males = 18) :
  (num_females * female_avg_age + num_males * male_avg_age) / (num_females + num_males) =  36.8 :=
by sorry

end ensemble_average_age_l2176_217661


namespace cost_of_birthday_gift_l2176_217646

theorem cost_of_birthday_gift 
  (boss_contrib : ℕ)
  (todd_contrib : ℕ)
  (employee_contrib : ℕ)
  (num_employees : ℕ)
  (h1 : boss_contrib = 15)
  (h2 : todd_contrib = 2 * boss_contrib)
  (h3 : employee_contrib = 11)
  (h4 : num_employees = 5) :
  boss_contrib + todd_contrib + num_employees * employee_contrib = 100 := by
  sorry

end cost_of_birthday_gift_l2176_217646


namespace regular_triangular_prism_cosine_l2176_217686

-- Define the regular triangular prism and its properties
structure RegularTriangularPrism :=
  (side : ℝ) -- the side length of the base and the lateral edge

-- Define the vertices of the prism
structure Vertices :=
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ)
  (A1 : ℝ × ℝ × ℝ)
  (B1 : ℝ × ℝ × ℝ)
  (C1 : ℝ × ℝ × ℝ)

-- Define the cosine calculation
def cos_angle (prism : RegularTriangularPrism) (v : Vertices) : ℝ := sorry

-- Prove that the cosine of the angle between diagonals AB1 and BC1 is 1/4
theorem regular_triangular_prism_cosine (prism : RegularTriangularPrism) (v : Vertices)
  : cos_angle prism v = 1 / 4 :=
sorry

end regular_triangular_prism_cosine_l2176_217686


namespace arithmetic_geometric_mean_l2176_217678

theorem arithmetic_geometric_mean (a b m : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : (a + b) / 2 = m * Real.sqrt (a * b)) :
  a / b = (m + Real.sqrt (m^2 + 1)) / (m - Real.sqrt (m^2 - 1)) :=
by
  sorry

end arithmetic_geometric_mean_l2176_217678


namespace february_five_sundays_in_twenty_first_century_l2176_217680

/-- 
  Define a function to check if a year is a leap year
-/
def is_leap_year (year : ℕ) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

/-- 
  Define the specific condition for the problem: 
  Given a year, whether February 1st for that year is a Sunday
-/
def february_first_is_sunday (year : ℕ) : Prop :=
  -- This is a placeholder logic. In real applications, you would
  -- calculate the exact weekday of February 1st for the provided year.
  sorry

/-- 
  The list of years in the 21st century where February has 5 Sundays is 
  exactly {2004, 2032, 2060, and 2088}.
-/
theorem february_five_sundays_in_twenty_first_century :
  {year : ℕ | is_leap_year year ∧ february_first_is_sunday year ∧ (2001 ≤ year ∧ year ≤ 2100)} =
  {2004, 2032, 2060, 2088} := sorry

end february_five_sundays_in_twenty_first_century_l2176_217680


namespace find_m_value_l2176_217692

def quadratic_inequality_solution_set (a b c : ℝ) (m : ℝ) := {x : ℝ | 0 < x ∧ x < 2}

theorem find_m_value (a b c : ℝ) (m : ℝ) 
  (h1 : a = -1/2) 
  (h2 : b = 2) 
  (h3 : c = m) 
  (h4 : quadratic_inequality_solution_set a b c m = {x : ℝ | 0 < x ∧ x < 2}) : 
  m = 1 := 
sorry

end find_m_value_l2176_217692


namespace range_of_p_l2176_217628

def A := {x : ℝ | x^2 - x - 2 > 0}
def B := {x : ℝ | (3 / x) - 1 ≥ 0}
def intersection := {x : ℝ | x ∈ A ∧ x ∈ B}
def C (p : ℝ) := {x : ℝ | 2 * x + p ≤ 0}

theorem range_of_p (p : ℝ) : (∀ x : ℝ, x ∈ intersection → x ∈ C p) → p < -6 := by
  sorry

end range_of_p_l2176_217628


namespace tangent_line_at_slope_two_l2176_217666

noncomputable def curve (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_at_slope_two :
  ∃ (x₀ y₀ : ℝ), (deriv curve x₀ = 2) ∧ (curve x₀ = y₀) ∧ (∀ x, (2 * (x - x₀) + y₀) = (2 * x)) :=
by 
  sorry

end tangent_line_at_slope_two_l2176_217666


namespace trig_identity_example_l2176_217623

theorem trig_identity_example (α : Real) (h : Real.cos α = 3 / 5) : Real.cos (2 * α) + Real.sin α ^ 2 = 9 / 25 := by
  sorry

end trig_identity_example_l2176_217623


namespace plates_count_l2176_217660

variable (x : ℕ)
variable (first_taken : ℕ)
variable (second_taken : ℕ)
variable (remaining_plates : ℕ := 9)

noncomputable def plates_initial : ℕ :=
  let first_batch := (x - 2) / 3
  let remaining_after_first := x - 2 - first_batch
  let second_batch := remaining_after_first / 2
  let remaining_after_second := remaining_after_first - second_batch
  remaining_after_second

theorem plates_count (x : ℕ) (h : plates_initial x = remaining_plates) : x = 29 := sorry

end plates_count_l2176_217660


namespace min_students_green_eyes_backpack_no_glasses_l2176_217613

theorem min_students_green_eyes_backpack_no_glasses
  (S G B Gl : ℕ)
  (h_S : S = 25)
  (h_G : G = 15)
  (h_B : B = 18)
  (h_Gl : Gl = 6)
  : ∃ x, x ≥ 8 ∧ x + Gl ≤ S ∧ x ≤ min G B :=
sorry

end min_students_green_eyes_backpack_no_glasses_l2176_217613


namespace find_grade_2_l2176_217687

-- Definitions for the problem
def grade_1 := 78
def weight_1 := 20
def weight_2 := 30
def grade_3 := 90
def weight_3 := 10
def grade_4 := 85
def weight_4 := 40
def overall_average := 83

noncomputable def calc_weighted_average (G : ℕ) : ℝ :=
  (grade_1 * weight_1 + G * weight_2 + grade_3 * weight_3 + grade_4 * weight_4) / (weight_1 + weight_2 + weight_3 + weight_4)

theorem find_grade_2 (G : ℕ) : calc_weighted_average G = overall_average → G = 81 := sorry

end find_grade_2_l2176_217687


namespace remainder_of_large_product_mod_17_l2176_217614

theorem remainder_of_large_product_mod_17 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end remainder_of_large_product_mod_17_l2176_217614


namespace directrix_of_parabola_l2176_217620

theorem directrix_of_parabola (x y : ℝ) (h : y = (1/4) * x^2) : y = -1 :=
sorry

end directrix_of_parabola_l2176_217620


namespace change_calculation_l2176_217633

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple = 4.25) := by
  sorry

end change_calculation_l2176_217633


namespace find_value_of_x2001_plus_y2001_l2176_217653

theorem find_value_of_x2001_plus_y2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
x ^ 2001 + y ^ 2001 = 2 ^ 2001 ∨ x ^ 2001 + y ^ 2001 = -2 ^ 2001 := by
  sorry

end find_value_of_x2001_plus_y2001_l2176_217653


namespace abby_bridget_chris_probability_l2176_217651

noncomputable def seatingProbability : ℚ :=
  let totalArrangements := 720
  let favorableArrangements := 114
  favorableArrangements / totalArrangements

theorem abby_bridget_chris_probability :
  seatingProbability = 19 / 120 :=
by
  simp [seatingProbability]
  sorry

end abby_bridget_chris_probability_l2176_217651


namespace flour_needed_l2176_217624

theorem flour_needed (sugar flour : ℕ) (h1 : sugar = 50) (h2 : sugar / 10 = flour) : flour = 5 :=
by
  sorry

end flour_needed_l2176_217624


namespace avg_remaining_two_l2176_217603

theorem avg_remaining_two (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 8) (h2 : (a + b + c) / 3 = 4) :
  (d + e) / 2 = 14 := by
  sorry

end avg_remaining_two_l2176_217603


namespace sum_of_consecutive_page_numbers_l2176_217685

def consecutive_page_numbers_product_and_sum (n m : ℤ) :=
  n * m = 20412

theorem sum_of_consecutive_page_numbers (n : ℤ) (h1 : consecutive_page_numbers_product_and_sum n (n + 1)) : n + (n + 1) = 285 :=
by
  sorry

end sum_of_consecutive_page_numbers_l2176_217685


namespace arithmetic_sequence_value_l2176_217622

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_value (h : a 3 + a 5 + a 11 + a 13 = 80) : a 8 = 20 :=
sorry

end arithmetic_sequence_value_l2176_217622


namespace jelly_bean_problem_l2176_217615

theorem jelly_bean_problem 
  (x y : ℕ) 
  (h1 : x + y = 1200) 
  (h2 : x = 3 * y - 400) :
  x = 800 := 
sorry

end jelly_bean_problem_l2176_217615


namespace evaluate_expression_l2176_217643

theorem evaluate_expression :
  200 * (200 - 3) + (200 ^ 2 - 8 ^ 2) = 79336 :=
by
  sorry

end evaluate_expression_l2176_217643


namespace total_worth_correct_l2176_217669

def row1_gold_bars : ℕ := 5
def row1_weight_per_bar : ℕ := 2
def row1_cost_per_kg : ℕ := 20000

def row2_gold_bars : ℕ := 8
def row2_weight_per_bar : ℕ := 3
def row2_cost_per_kg : ℕ := 18000

def row3_gold_bars : ℕ := 3
def row3_weight_per_bar : ℕ := 5
def row3_cost_per_kg : ℕ := 22000

def row4_gold_bars : ℕ := 4
def row4_weight_per_bar : ℕ := 4
def row4_cost_per_kg : ℕ := 25000

def total_worth : ℕ :=
  (row1_gold_bars * row1_weight_per_bar * row1_cost_per_kg)
  + (row2_gold_bars * row2_weight_per_bar * row2_cost_per_kg)
  + (row3_gold_bars * row3_weight_per_bar * row3_cost_per_kg)
  + (row4_gold_bars * row4_weight_per_bar * row4_cost_per_kg)

theorem total_worth_correct : total_worth = 1362000 := by
  sorry

end total_worth_correct_l2176_217669


namespace area_of_ring_between_outermost_and_middle_circle_l2176_217641

noncomputable def pi : ℝ := Real.pi

theorem area_of_ring_between_outermost_and_middle_circle :
  let r_outermost := 12
  let r_middle := 8
  let A_outermost := pi * r_outermost^2
  let A_middle := pi * r_middle^2
  A_outermost - A_middle = 80 * pi :=
by 
  sorry

end area_of_ring_between_outermost_and_middle_circle_l2176_217641


namespace ratio_of_edges_l2176_217634

noncomputable def cube_volume (edge : ℝ) : ℝ := edge^3

theorem ratio_of_edges 
  {a b : ℝ} 
  (h : cube_volume a / cube_volume b = 27) : 
  a / b = 3 :=
by
  sorry

end ratio_of_edges_l2176_217634


namespace geometric_sequence_sixth_term_l2176_217642

variable (q : ℕ) (a_2 a_6 : ℕ)

-- Given conditions:
axiom h1 : q = 2
axiom h2 : a_2 = 8

-- Prove that a_6 = 128 where a_n = a_2 * q^(n-2)
theorem geometric_sequence_sixth_term : a_6 = a_2 * q^4 → a_6 = 128 :=
by sorry

end geometric_sequence_sixth_term_l2176_217642


namespace ball_bounce_height_l2176_217647

noncomputable def height_after_bounces (h₀ : ℝ) (r : ℝ) (b : ℕ) : ℝ :=
  h₀ * (r ^ b)

theorem ball_bounce_height
  (h₀ : ℝ) (r : ℝ) (hb : ℕ) (h₀_pos : h₀ > 0) (r_pos : 0 < r ∧ r < 1) (h₀_val : h₀ = 320) (r_val : r = 3 / 4) (height_limit : ℝ) (height_limit_val : height_limit = 40):
  (hb ≥ 6) ∧ height_after_bounces h₀ r hb < height_limit :=
by
  sorry

end ball_bounce_height_l2176_217647


namespace football_outcomes_l2176_217606

theorem football_outcomes : 
  ∃ (W D L : ℕ), (3 * W + D = 19) ∧ (W + D + L = 14) ∧ 
  ((W = 3 ∧ D = 10 ∧ L = 1) ∨ 
   (W = 4 ∧ D = 7 ∧ L = 3) ∨ 
   (W = 5 ∧ D = 4 ∧ L = 5) ∨ 
   (W = 6 ∧ D = 1 ∧ L = 7)) ∧
  (∀ W' D' L' : ℕ, (3 * W' + D' = 19) → (W' + D' + L' = 14) → 
    (W' = 3 ∧ D' = 10 ∧ L' = 1) ∨ 
    (W' = 4 ∧ D' = 7 ∧ L' = 3) ∨ 
    (W' = 5 ∧ D' = 4 ∧ L' = 5) ∨ 
    (W' = 6 ∧ D' = 1 ∧ L' = 7)) := 
sorry

end football_outcomes_l2176_217606


namespace arrange_books_l2176_217601

noncomputable def numberOfArrangements : Nat :=
  4 * 3 * 6 * (Nat.factorial 9)

theorem arrange_books :
  numberOfArrangements = 26210880 := by
  sorry

end arrange_books_l2176_217601


namespace tan_triple_angle_l2176_217655

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l2176_217655


namespace not_necessarily_divisible_by_66_l2176_217648

open Nat

-- Definition of what it means to be the product of four consecutive integers
def product_of_four_consecutive_integers (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (k * (k + 1) * (k + 2) * (k + 3))

-- Lean theorem statement for the proof problem
theorem not_necessarily_divisible_by_66 (n : ℕ) 
  (h1 : product_of_four_consecutive_integers n) 
  (h2 : 11 ∣ n) : ¬ (66 ∣ n) :=
sorry

end not_necessarily_divisible_by_66_l2176_217648


namespace remainder_2519_div_7_l2176_217694

theorem remainder_2519_div_7 : 2519 % 7 = 6 :=
by
  sorry

end remainder_2519_div_7_l2176_217694


namespace find_minimum_value_l2176_217602

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + a * |x - 1| + 1

-- The statement of the proof problem
theorem find_minimum_value (a : ℝ) (h : a ≥ 0) :
  (a = 0 → ∀ x, f x a ≥ 1 ∧ ∃ x, f x a = 1) ∧
  ((0 < a ∧ a < 2) → ∀ x, f x a ≥ -a^2 / 4 + a + 1 ∧ ∃ x, f x a = -a^2 / 4 + a + 1) ∧
  (a ≥ 2 → ∀ x, f x a ≥ 2 ∧ ∃ x, f x a = 2) := 
by
  sorry

end find_minimum_value_l2176_217602


namespace problem_gcf_lcm_sum_l2176_217698

-- Let A be the GCF of {15, 20, 30}
def A : ℕ := Nat.gcd (Nat.gcd 15 20) 30

-- Let B be the LCM of {15, 20, 30}
def B : ℕ := Nat.lcm (Nat.lcm 15 20) 30

-- We need to prove that A + B = 65
theorem problem_gcf_lcm_sum :
  A + B = 65 :=
by
  sorry

end problem_gcf_lcm_sum_l2176_217698


namespace original_price_l2176_217689

theorem original_price (x : ℝ) (h1 : 0.75 * x + 12 = x - 12) (h2 : 0.90 * x - 42 = x - 12) : x = 360 :=
by
  sorry

end original_price_l2176_217689


namespace age_difference_l2176_217640

variable (E Y : ℕ)

theorem age_difference (hY : Y = 35) (hE : E - 15 = 2 * (Y - 15)) : E - Y = 20 := by
  -- Assertions and related steps could be handled subsequently.
  sorry

end age_difference_l2176_217640


namespace factorize_expression_l2176_217625

variable (a b c : ℝ)

theorem factorize_expression : 
  (a - 2 * b) * (a - 2 * b - 4) + 4 - c ^ 2 = ((a - 2 * b) - 2 + c) * ((a - 2 * b) - 2 - c) := 
by
  sorry

end factorize_expression_l2176_217625


namespace a_gt_b_neither_sufficient_nor_necessary_l2176_217691

theorem a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) := 
sorry

end a_gt_b_neither_sufficient_nor_necessary_l2176_217691


namespace ad_space_length_l2176_217677

theorem ad_space_length 
  (num_companies : ℕ)
  (ads_per_company : ℕ)
  (width : ℝ)
  (cost_per_sq_ft : ℝ)
  (total_cost : ℝ) 
  (H1 : num_companies = 3)
  (H2 : ads_per_company = 10)
  (H3 : width = 5)
  (H4 : cost_per_sq_ft = 60)
  (H5 : total_cost = 108000) :
  ∃ L : ℝ, (num_companies * ads_per_company * width * L * cost_per_sq_ft = total_cost) ∧ (L = 12) :=
by
  sorry

end ad_space_length_l2176_217677


namespace imaginary_unit_real_part_eq_l2176_217658

theorem imaginary_unit_real_part_eq (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (∃ r : ℝ, ((3 + i) * (a + 2 * i) / (1 + i) = r)) → a = 4 :=
by
  sorry

end imaginary_unit_real_part_eq_l2176_217658


namespace evaluate_expression_l2176_217621

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 8 = 43046724 := 
by
  sorry

end evaluate_expression_l2176_217621


namespace multiplicative_inverse_185_mod_341_l2176_217626

theorem multiplicative_inverse_185_mod_341 :
  ∃ (b: ℕ), b ≡ 74466 [MOD 341] ∧ 185 * b ≡ 1 [MOD 341] :=
sorry

end multiplicative_inverse_185_mod_341_l2176_217626


namespace ratio_five_to_one_l2176_217690

theorem ratio_five_to_one (x : ℕ) (h : 5 * 12 = x) : x = 60 :=
by
  sorry

end ratio_five_to_one_l2176_217690


namespace least_integer_value_abs_l2176_217618

theorem least_integer_value_abs (x : ℤ) : 
  (∃ x : ℤ, (abs (3 * x + 5) ≤ 20) ∧ (∀ y : ℤ, (abs (3 * y + 5) ≤ 20) → x ≤ y)) ↔ x = -8 :=
by
  sorry

end least_integer_value_abs_l2176_217618


namespace log_base_half_cuts_all_horizontal_lines_l2176_217644

theorem log_base_half_cuts_all_horizontal_lines (x : ℝ) (k : ℝ) (h_pos : x > 0) (h_eq : y = Real.logb 0.5 x) : ∃ x, ∀ k, k = Real.logb 0.5 x ↔ x > 0 := 
sorry

end log_base_half_cuts_all_horizontal_lines_l2176_217644


namespace tan_function_constants_l2176_217629

theorem tan_function_constants (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_period : b ≠ 0 ∧ ∃ k : ℤ, b * (3 / 2) = k * π) 
(h_pass : a * Real.tan (b * (π / 4)) = 3) : a * b = 2 * Real.sqrt 3 :=
by 
  sorry

end tan_function_constants_l2176_217629


namespace five_fold_function_application_l2176_217637

def f (x : ℤ) : ℤ :=
if x ≥ 0 then -x^2 + 1 else x + 9

theorem five_fold_function_application : f (f (f (f (f 2)))) = -17 :=
by
  sorry

end five_fold_function_application_l2176_217637


namespace union_M_N_eq_N_l2176_217674

def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | -3 < x ∧ x < 3}

theorem union_M_N_eq_N : M ∪ N = N := by
  sorry

end union_M_N_eq_N_l2176_217674


namespace blue_apples_l2176_217696

theorem blue_apples (B : ℕ) (h : (12 / 5) * B = 12) : B = 5 :=
by
  sorry

end blue_apples_l2176_217696


namespace solve_inequality_l2176_217688

theorem solve_inequality : 
  {x : ℝ | -x^2 - 2*x + 3 ≤ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end solve_inequality_l2176_217688


namespace arithmetic_sequence_a11_l2176_217617

theorem arithmetic_sequence_a11 (a : ℕ → ℤ) (h_arithmetic : ∀ n, a (n + 1) = a n + (a 2 - a 1))
  (h_a3 : a 3 = 4) (h_a5 : a 5 = 8) : a 11 = 12 :=
by
  sorry

end arithmetic_sequence_a11_l2176_217617


namespace cos_alpha_value_l2176_217627

-- Definitions for conditions and theorem statement

def condition_1 (α : ℝ) : Prop :=
  0 < α ∧ α < Real.pi / 2

def condition_2 (α : ℝ) : Prop :=
  Real.cos (Real.pi / 3 + α) = 1 / 3

theorem cos_alpha_value (α : ℝ) (h1 : condition_1 α) (h2 : condition_2 α) :
  Real.cos α = (1 + 2 * Real.sqrt 6) / 6 := sorry

end cos_alpha_value_l2176_217627


namespace combination_x_l2176_217654
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

theorem combination_x (x : ℕ) (H : C 25 (2 * x) = C 25 (x + 4)) : x = 4 ∨ x = 7 :=
by sorry

end combination_x_l2176_217654


namespace smallest_sum_arith_geo_sequence_l2176_217607

theorem smallest_sum_arith_geo_sequence :
  ∃ (X Y Z W : ℕ),
    X < Y ∧ Y < Z ∧ Z < W ∧
    (2 * Y = X + Z) ∧
    (Y ^ 2 = Z * X) ∧
    (Z / Y = 7 / 4) ∧
    (X + Y + Z + W = 97) :=
by
  sorry

end smallest_sum_arith_geo_sequence_l2176_217607


namespace initial_goldfish_eq_15_l2176_217676

-- Let's define our setup as per the conditions provided
def fourGoldfishLeft := 4
def elevenGoldfishDisappeared := 11

-- Our main statement that we need to prove
theorem initial_goldfish_eq_15 : fourGoldfishLeft + elevenGoldfishDisappeared = 15 := by
  sorry

end initial_goldfish_eq_15_l2176_217676


namespace sum_of_specific_terms_in_arithmetic_sequence_l2176_217600

theorem sum_of_specific_terms_in_arithmetic_sequence
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h_S11 : S 11 = 44) :
  a 4 + a 6 + a 8 = 12 :=
sorry

end sum_of_specific_terms_in_arithmetic_sequence_l2176_217600


namespace third_number_in_first_set_l2176_217604

theorem third_number_in_first_set (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 70 + 13) / 3 + 9 → x = 60 := by
  sorry

end third_number_in_first_set_l2176_217604


namespace prod_is_96_l2176_217605

noncomputable def prod_of_nums (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : ℝ := x * y

theorem prod_is_96 (x y : ℝ) (h1 : x + y = 20) (h2 : (x - y)^2 = 16) : prod_of_nums x y h1 h2 = 96 :=
by
  sorry

end prod_is_96_l2176_217605


namespace smallest_n_for_violet_candy_l2176_217693

theorem smallest_n_for_violet_candy (p y o n : Nat) (h : 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) :
  n = 8 :=
by 
  sorry

end smallest_n_for_violet_candy_l2176_217693


namespace power_mod_l2176_217668

theorem power_mod : (5 ^ 2023) % 11 = 4 := 
by 
  sorry

end power_mod_l2176_217668


namespace sequence_correct_l2176_217699

def seq_formula (n : ℕ) : ℚ := 3/2 + (-1)^n * 11/2

theorem sequence_correct (n : ℕ) :
  (n % 2 = 0 ∧ seq_formula n = 7) ∨ (n % 2 = 1 ∧ seq_formula n = -4) :=
by
  sorry

end sequence_correct_l2176_217699


namespace triangle_angle_sum_l2176_217652

theorem triangle_angle_sum (x : ℝ) (h1 : 70 + 50 + x = 180) : x = 60 := by
  -- proof goes here
  sorry

end triangle_angle_sum_l2176_217652


namespace find_line_eqn_from_bisected_chord_l2176_217635

noncomputable def line_eqn_from_bisected_chord (x y : ℝ) : Prop :=
  2 * x + y - 3 = 0

theorem find_line_eqn_from_bisected_chord (
  A B : ℝ × ℝ) 
  (hA :  (A.1^2) / 2 + (A.2^2) / 4 = 1)
  (hB :  (B.1^2) / 2 + (B.2^2) / 4 = 1)
  (h_mid : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) :
  line_eqn_from_bisected_chord 1 1 :=
by 
  sorry

end find_line_eqn_from_bisected_chord_l2176_217635


namespace ceil_square_of_neg_fraction_l2176_217695

theorem ceil_square_of_neg_fraction : 
  (Int.ceil ((-7 / 4 : ℚ)^2 : ℚ)).toNat = 4 := by
  sorry

end ceil_square_of_neg_fraction_l2176_217695


namespace union_sets_l2176_217683

-- Define the sets A and B as conditions
def A : Set ℝ := {0, 1}  -- Since lg 1 = 0
def B : Set ℝ := {-1, 0}

-- Define that A union B equals {-1, 0, 1}
theorem union_sets : A ∪ B = {-1, 0, 1} := by
  sorry

end union_sets_l2176_217683


namespace part1_part2_part3_l2176_217684

-- Definitions for conditions used in the proof problems
def eq1 (a b : ℝ) : Prop := 2 * a + b = 0
def eq2 (a x : ℝ) : Prop := x = a ^ 2

-- Part 1: Prove b = 4 and x = 4 given a = -2
theorem part1 (a b x : ℝ) (h1 : a = -2) (h2 : eq1 a b) (h3 : eq2 a x) : b = 4 ∧ x = 4 :=
by sorry

-- Part 2: Prove a = -3 and x = 9 given b = 6
theorem part2 (a b x : ℝ) (h1 : b = 6) (h2 : eq1 a b) (h3 : eq2 a x) : a = -3 ∧ x = 9 :=
by sorry

-- Part 3: Prove x = 2 given a^2*x + (a + b)^2*x = 8
theorem part3 (a b x : ℝ) (h : a^2 * x + (a + b)^2 * x = 8) : x = 2 :=
by sorry

end part1_part2_part3_l2176_217684


namespace scientific_notation_36000_l2176_217630

theorem scientific_notation_36000 : 36000 = 3.6 * (10^4) := 
by 
  -- Skipping the proof by adding sorry
  sorry

end scientific_notation_36000_l2176_217630


namespace sequence_properties_l2176_217645

noncomputable def a (n : ℕ) : ℕ := 3 * n - 1

def S (n : ℕ) : ℕ := n * (2 + 3 * n - 1) / 2

theorem sequence_properties :
  a 5 + a 7 = 34 ∧ ∀ n, S n = (3 * n ^ 2 + n) / 2 :=
by
  sorry

end sequence_properties_l2176_217645


namespace completed_shape_perimeter_602_l2176_217619

noncomputable def rectangle_perimeter (length width : ℝ) : ℝ :=
  2 * (length + width)

noncomputable def total_perimeter_no_overlap (n : ℕ) (length width : ℝ) : ℝ :=
  n * rectangle_perimeter length width

noncomputable def total_reduction (n : ℕ) (overlap : ℝ) : ℝ :=
  (n - 1) * overlap

noncomputable def overall_perimeter (n : ℕ) (length width overlap : ℝ) : ℝ :=
  total_perimeter_no_overlap n length width - total_reduction n overlap

theorem completed_shape_perimeter_602 :
  overall_perimeter 100 3 1 2 = 602 :=
by
  sorry

end completed_shape_perimeter_602_l2176_217619


namespace cube_cross_section_area_l2176_217659

def cube_edge_length (a : ℝ) := a > 0

def plane_perpendicular_body_diagonal := 
  ∃ (p : ℝ × ℝ × ℝ), ∀ (x y z : ℝ), 
  p = (x / 2, y / 2, z / 2) ∧ 
  (x + y + z) = (1 : ℝ)

theorem cube_cross_section_area
  (a : ℝ) 
  (h : cube_edge_length a) 
  (plane : plane_perpendicular_body_diagonal) : 
  ∃ (A : ℝ), 
  A = (3 * a^2 * Real.sqrt 3 / 4) := sorry

end cube_cross_section_area_l2176_217659


namespace kelvin_classes_l2176_217681

theorem kelvin_classes (c : ℕ) (h1 : Grant = 4 * c) (h2 : c + Grant = 450) : c = 90 :=
by sorry

end kelvin_classes_l2176_217681


namespace sum_of_factors_is_17_l2176_217657

theorem sum_of_factors_is_17 :
  ∃ (a b c d e f g : ℤ), 
  (16 * x^4 - 81 * y^4) =
    (a * x + b * y) * 
    (c * x^2 + d * x * y + e * y^2) * 
    (f * x + g * y) ∧ 
    a + b + c + d + e + f + g = 17 :=
by
  sorry

end sum_of_factors_is_17_l2176_217657


namespace will_jogged_for_30_minutes_l2176_217670

theorem will_jogged_for_30_minutes 
  (calories_before : ℕ)
  (calories_per_minute : ℕ)
  (net_calories_after : ℕ)
  (h1 : calories_before = 900)
  (h2 : calories_per_minute = 10)
  (h3 : net_calories_after = 600) :
  let calories_burned := calories_before - net_calories_after
  let jogging_time := calories_burned / calories_per_minute
  jogging_time = 30 := by
  sorry

end will_jogged_for_30_minutes_l2176_217670


namespace fraction_subtraction_l2176_217650

theorem fraction_subtraction : (5 / 6) - (1 / 12) = (3 / 4) := 
by 
  sorry

end fraction_subtraction_l2176_217650


namespace simplified_form_l2176_217649

def simplify_expression (x : ℝ) : ℝ :=
  (3 * x - 2) * (6 * x ^ 8 + 3 * x ^ 7 - 2 * x ^ 3 + x)

theorem simplified_form (x : ℝ) : 
  simplify_expression x = 18 * x ^ 9 - 3 * x ^ 8 - 6 * x ^ 7 - 6 * x ^ 4 - 4 * x ^ 3 + x :=
by
  sorry

end simplified_form_l2176_217649


namespace school_raised_amount_correct_l2176_217612

def school_fundraising : Prop :=
  let mrsJohnson := 2300
  let mrsSutton := mrsJohnson / 2
  let missRollin := mrsSutton * 8
  let topThreeTotal := missRollin * 3
  let mrEdward := missRollin * 0.75
  let msAndrea := mrEdward * 1.5
  let totalRaised := mrsJohnson + mrsSutton + missRollin + mrEdward + msAndrea
  let adminFee := totalRaised * 0.02
  let maintenanceExpense := totalRaised * 0.05
  let totalDeductions := adminFee + maintenanceExpense
  let finalAmount := totalRaised - totalDeductions
  finalAmount = 28737

theorem school_raised_amount_correct : school_fundraising := 
by 
  sorry

end school_raised_amount_correct_l2176_217612


namespace generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l2176_217610

-- Define the number five as 4, as we are using five 4s
def four := 4

-- Now prove that each number from 1 to 22 can be generated using the conditions
theorem generate_1 : 1 = (4 / 4) * (4 / 4) := sorry
theorem generate_2 : 2 = (4 / 4) + (4 / 4) := sorry
theorem generate_3 : 3 = ((4 + 4 + 4) / 4) - (4 / 4) := sorry
theorem generate_4 : 4 = 4 * (4 - 4) + 4 := sorry
theorem generate_5 : 5 = 4 + (4 / 4) := sorry
theorem generate_6 : 6 = 4 + 4 - (4 / 4) := sorry
theorem generate_7 : 7 = 4 + 4 - (4 / 4) := sorry
theorem generate_8 : 8 = 4 + 4 := sorry
theorem generate_9 : 9 = 4 + 4 + (4 / 4) := sorry
theorem generate_10 : 10 = 4 * (2 + 4 / 4) := sorry
theorem generate_11 : 11 = 4 * (3 - 1 / 4) := sorry
theorem generate_12 : 12 = 4 + 4 + 4 := sorry
theorem generate_13 : 13 = (4 * 4) - (4 / 4) - 4 := sorry
theorem generate_14 : 14 = 4 * (4 - 1 / 4) := sorry
theorem generate_15 : 15 = 4 * 4 - (4 / 4) - 1 := sorry
theorem generate_16 : 16 = 4 * (4 - (4 - 4) / 4) := sorry
theorem generate_17 : 17 = 4 * (4 + 4 / 4) := sorry
theorem generate_18 : 18 = 4 * 4 + 4 - 4 / 4 := sorry
theorem generate_19 : 19 = 4 + 4 + 4 + 4 + 3 := sorry
theorem generate_20 : 20 = 4 + 4 + 4 + 4 + 4 := sorry
theorem generate_21 : 21 = 4 * 4 + (4 - 1) / 4 := sorry
theorem generate_22 : 22 = (4 * 4 + 4) / 4 := sorry

end generate_1_generate_2_generate_3_generate_4_generate_5_generate_6_generate_7_generate_8_generate_9_generate_10_generate_11_generate_12_generate_13_generate_14_generate_15_generate_16_generate_17_generate_18_generate_19_generate_20_generate_21_generate_22_l2176_217610


namespace share_of_b_l2176_217631

theorem share_of_b (x : ℝ) (h : 3300 / ((7/2) * x) = 2 / 7) :  
   let total_profit := 3300
   let B_share := (x / ((7/2) * x)) * total_profit
   B_share = 942.86 :=
by sorry

end share_of_b_l2176_217631


namespace price_reduction_for_desired_profit_l2176_217608

def profit_per_piece (x : ℝ) : ℝ := 40 - x
def pieces_sold_per_day (x : ℝ) : ℝ := 20 + 2 * x

theorem price_reduction_for_desired_profit (x : ℝ) :
  (profit_per_piece x) * (pieces_sold_per_day x) = 1200 ↔ (x = 10 ∨ x = 20) := by
  sorry

end price_reduction_for_desired_profit_l2176_217608


namespace phones_left_is_7500_l2176_217611

def last_year_production : ℕ := 5000
def this_year_production : ℕ := 2 * last_year_production
def sold_phones : ℕ := this_year_production / 4
def phones_left : ℕ := this_year_production - sold_phones

theorem phones_left_is_7500 : phones_left = 7500 :=
by
  sorry

end phones_left_is_7500_l2176_217611


namespace b_should_pay_348_48_l2176_217638

/-- Definitions for the given conditions --/

def horses_a : ℕ := 12
def months_a : ℕ := 8

def horses_b : ℕ := 16
def months_b : ℕ := 9

def horses_c : ℕ := 18
def months_c : ℕ := 6

def total_rent : ℕ := 841

/-- Calculate the individual and total contributions in horse-months --/

def contribution_a : ℕ := horses_a * months_a
def contribution_b : ℕ := horses_b * months_b
def contribution_c : ℕ := horses_c * months_c

def total_contributions : ℕ := contribution_a + contribution_b + contribution_c

/-- Calculate cost per horse-month and b's share of the rent --/

def cost_per_horse_month : ℚ := total_rent / total_contributions
def b_share : ℚ := contribution_b * cost_per_horse_month

/-- Lean statement to check b's share --/

theorem b_should_pay_348_48 : b_share = 348.48 := by
  sorry

end b_should_pay_348_48_l2176_217638


namespace linear_combination_solution_l2176_217632

theorem linear_combination_solution :
  ∃ a b c : ℚ, 
    a • (⟨1, -2, 3⟩ : ℚ × ℚ × ℚ) + b • (⟨4, 1, -1⟩ : ℚ × ℚ × ℚ) + c • (⟨-3, 2, 1⟩ : ℚ × ℚ × ℚ) = ⟨0, 1, 4⟩ ∧
    a = -491/342 ∧
    b = 233/342 ∧
    c = 49/38 :=
by
  sorry

end linear_combination_solution_l2176_217632


namespace find_value_of_a_l2176_217672

noncomputable def f (x a : ℝ) : ℝ := (3 * Real.log x - x^2 - a - 2)^2 + (x - a)^2

theorem find_value_of_a (a : ℝ) : (∃ x : ℝ, f x a ≤ 8) ↔ a = -1 :=
by
  sorry

end find_value_of_a_l2176_217672


namespace max_profit_l2176_217665

/-- Define the cost and price of device A and device B -/
def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8

/-- Define the total number of devices -/
def total_devices : ℝ := 50

/-- Define the profits per device -/
def profit_per_A : ℝ := price_A - cost_A -- 0.3
def profit_per_B : ℝ := price_B - cost_B -- 0.4

/-- Define the function for total profit -/
def total_profit (x : ℝ) : ℝ :=
  profit_per_A * x + profit_per_B * (total_devices - x)

/-- Define the constraint -/
def constraint (x : ℝ) : Prop := 4 * x ≥ total_devices - x -- x ≥ 10

/-- The statement of the problem that needs to be proven -/
theorem max_profit :
  (total_profit x = -0.1 * x + 20) ∧ 
  ( ∀ x, constraint x → x ≥ 10 → x = 10 ∧ total_profit x = 19) :=
by
  sorry

end max_profit_l2176_217665


namespace ratio_of_areas_l2176_217679

-- Definitions based on the conditions given
variables (A B M N P Q O : Type) 
variables (AB BM BP : ℝ)

-- Assumptions
axiom hAB : AB = 6
axiom hBM : BM = 9
axiom hBP : BP = 5

-- Theorem statement
theorem ratio_of_areas (hMN : M ≠ N) (hPQ : P ≠ Q) :
  (1 / 121 : ℝ) = sorry :=
by sorry

end ratio_of_areas_l2176_217679


namespace tomatoes_initially_l2176_217662

-- Conditions
def tomatoes_picked_yesterday : ℕ := 56
def tomatoes_picked_today : ℕ := 41
def tomatoes_left_after_yesterday : ℕ := 104

-- The statement to prove
theorem tomatoes_initially : tomatoes_left_after_yesterday + tomatoes_picked_yesterday + tomatoes_picked_today = 201 :=
  by
  -- Proof steps would go here
  sorry

end tomatoes_initially_l2176_217662


namespace cos_A_side_c_l2176_217667

-- helper theorem for cosine rule usage
theorem cos_A (a b c : ℝ) (cosA cosB cosC : ℝ) (h : 3 * a * cosA = c * cosB + b * cosC) : cosA = 1 / 3 :=
by
  sorry

-- main statement combining conditions 1 and 2 with side value results
theorem side_c (a b c : ℝ) (cosA cosB cosC : ℝ) (h1 : 3 * a * cosA = c * cosB + b * cosC) (h2 : cosB + cosC = 0) (h3 : a = 1) : c = 2 :=
by
  have h_cosA : cosA = 1 / 3 := cos_A a b c cosA cosB cosC h1
  sorry

end cos_A_side_c_l2176_217667


namespace largest_int_less_than_100_with_remainder_5_l2176_217639

theorem largest_int_less_than_100_with_remainder_5 (x : ℤ) (n : ℤ) (h₁ : x = 7 * n + 5) (h₂ : x < 100) : 
  x = 96 := by
  sorry

end largest_int_less_than_100_with_remainder_5_l2176_217639


namespace triangular_number_difference_30_28_l2176_217636

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_number_difference_30_28 : triangular_number 30 - triangular_number 28 = 59 := 
by
  sorry

end triangular_number_difference_30_28_l2176_217636


namespace gold_copper_alloy_ratio_l2176_217609

theorem gold_copper_alloy_ratio 
  (water : ℝ) 
  (G : ℝ) 
  (C : ℝ) 
  (H1 : G = 10 * water)
  (H2 : C = 6 * water)
  (H3 : 10 * G + 6 * C = 8 * (G + C)) : 
  G / C = 1 :=
by
  sorry

end gold_copper_alloy_ratio_l2176_217609


namespace retail_price_of_machine_l2176_217682

theorem retail_price_of_machine 
  (wholesale_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) 
  (P : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.10)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = wholesale_price * (1 + profit_rate))
  (h5 : (P * (1 - discount_rate)) = selling_price) : 
  P = 120 := by
  sorry

end retail_price_of_machine_l2176_217682


namespace sum_of_extremes_of_g_l2176_217656

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - abs (2 * x - 8)

theorem sum_of_extremes_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≤ g 4) ∧ (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≥ g 1) → g 4 + g 1 = 2 :=
by
  sorry

end sum_of_extremes_of_g_l2176_217656
