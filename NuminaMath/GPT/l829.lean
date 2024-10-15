import Mathlib

namespace NUMINAMATH_GPT_max_value_t_min_value_y_l829_82929

open Real

-- Maximum value of t for ∀ x ∈ ℝ, |3x + 2| + |3x - 1| ≥ t
theorem max_value_t :
  ∃ t, (∀ x : ℝ, |3 * x + 2| + |3 * x - 1| ≥ t) ∧ t = 3 :=
by
  sorry

-- Minimum value of y for 4m + 5n = 3
theorem min_value_y (m n: ℝ) (hm : m > 0) (hn: n > 0) (h: 4 * m + 5 * n = 3) :
  ∃ y, (y = (1 / (m + 2 * n)) + (4 / (3 * m + 3 * n))) ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_t_min_value_y_l829_82929


namespace NUMINAMATH_GPT_robin_piano_highest_before_lowest_l829_82992

def probability_reach_highest_from_middle_C : ℚ :=
  let p_k (k : ℕ) (p_prev : ℚ) (p_next : ℚ) : ℚ := (1/2 : ℚ) * p_prev + (1/2 : ℚ) * p_next
  let p_1 := 0
  let p_88 := 1
  let A := -1/87
  let B := 1/87
  A + B * 40

theorem robin_piano_highest_before_lowest :
  probability_reach_highest_from_middle_C = 13 / 29 :=
by
  sorry

end NUMINAMATH_GPT_robin_piano_highest_before_lowest_l829_82992


namespace NUMINAMATH_GPT_minimum_bailing_rate_l829_82956

-- Conditions as formal definitions.
def distance_from_shore : ℝ := 3
def intake_rate : ℝ := 20 -- gallons per minute
def sinking_threshold : ℝ := 120 -- gallons
def speed_first_half : ℝ := 6 -- miles per hour
def speed_second_half : ℝ := 3 -- miles per hour

-- Formal translation of the problem using definitions.
theorem minimum_bailing_rate : (distance_from_shore = 3) →
                             (intake_rate = 20) →
                             (sinking_threshold = 120) →
                             (speed_first_half = 6) →
                             (speed_second_half = 3) →
                             (∃ r : ℝ, 18 ≤ r) :=
by
  sorry

end NUMINAMATH_GPT_minimum_bailing_rate_l829_82956


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l829_82951

-- Problem 1
theorem simplify_expression1 (a b : ℝ) : 
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := 
sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := 
sorry

-- Problem 3
theorem simplify_expression3 (a b : ℝ) : 
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := 
sorry

-- Problem 4
theorem simplify_expression4 (x y : ℝ) : 
  6 * x * y^2 - (2 * x - (1 / 2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := 
sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_simplify_expression3_simplify_expression4_l829_82951


namespace NUMINAMATH_GPT_compute_fraction_l829_82991

theorem compute_fraction : (2015 : ℝ) / ((2015 : ℝ)^2 - (2016 : ℝ) * (2014 : ℝ)) = 2015 :=
by {
  sorry
}

end NUMINAMATH_GPT_compute_fraction_l829_82991


namespace NUMINAMATH_GPT_fraction_value_l829_82986

theorem fraction_value (a b c d : ℕ)
  (h₁ : a = 4 * b)
  (h₂ : b = 3 * c)
  (h₃ : c = 5 * d) :
  (a * c) / (b * d) = 20 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l829_82986


namespace NUMINAMATH_GPT_find_f_60_l829_82995

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition.

axiom functional_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_48 : f 48 = 36

theorem find_f_60 : f 60 = 28.8 := by 
  sorry

end NUMINAMATH_GPT_find_f_60_l829_82995


namespace NUMINAMATH_GPT_expression_value_l829_82909

theorem expression_value (a b c : ℕ) (h1 : 25^a * 5^(2*b) = 5^6) (h2 : 4^b / 4^c = 4) : a^2 + a * b + 3 * c = 6 := by
  sorry

end NUMINAMATH_GPT_expression_value_l829_82909


namespace NUMINAMATH_GPT_samuel_faster_l829_82969

theorem samuel_faster (S T_h : ℝ) (hT_h : T_h = 1.3) (hS : S = 30) :
  (T_h * 60) - S = 48 :=
by
  sorry

end NUMINAMATH_GPT_samuel_faster_l829_82969


namespace NUMINAMATH_GPT_gift_wrapping_combinations_l829_82954

theorem gift_wrapping_combinations 
  (wrapping_varieties : ℕ)
  (ribbon_colors : ℕ)
  (gift_card_types : ℕ)
  (H_wrapping_varieties : wrapping_varieties = 8)
  (H_ribbon_colors : ribbon_colors = 3)
  (H_gift_card_types : gift_card_types = 4) : 
  wrapping_varieties * ribbon_colors * gift_card_types = 96 := 
by
  sorry

end NUMINAMATH_GPT_gift_wrapping_combinations_l829_82954


namespace NUMINAMATH_GPT_Homer_first_try_points_l829_82935

variable (x : ℕ)
variable (h1 : x + (x - 70) + 2 * (x - 70) = 1390)

theorem Homer_first_try_points : x = 400 := by
  sorry

end NUMINAMATH_GPT_Homer_first_try_points_l829_82935


namespace NUMINAMATH_GPT_pears_equivalence_l829_82947

theorem pears_equivalence :
  (3 / 4 : ℚ) * 16 * (5 / 6) = 10 → 
  (2 / 5 : ℚ) * 20 * (5 / 6) = 20 / 3 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_pears_equivalence_l829_82947


namespace NUMINAMATH_GPT_least_pos_int_with_ten_factors_l829_82990

theorem least_pos_int_with_ten_factors : ∃ (n : ℕ), n > 0 ∧ (∀ m, (m > 0 ∧ ∃ d : ℕ, d∣n → d = 1 ∨ d = n) → m < n) ∧ ( ∃! n, ∃ d : ℕ, d∣n ) := sorry

end NUMINAMATH_GPT_least_pos_int_with_ten_factors_l829_82990


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l829_82959

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h_pos : x > 0) :
  (a = 4 → x + a / x ≥ 4) ∧ (∃ b : ℝ, b ≠ 4 ∧ ∃ x : ℝ, x > 0 ∧ x + b / x ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l829_82959


namespace NUMINAMATH_GPT_percent_decrease_area_pentagon_l829_82931

open Real

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * sqrt 3 / 2) * s ^ 2

noncomputable def area_pentagon (s : ℝ) : ℝ :=
  (sqrt (5 * (5 + 2 * sqrt 5)) / 4) * s ^ 2

noncomputable def diagonal_pentagon (s : ℝ) : ℝ :=
  (1 + sqrt 5) / 2 * s

theorem percent_decrease_area_pentagon :
  let s_p := sqrt (400 / sqrt (5 * (5 + 2 * sqrt 5)))
  let d := diagonal_pentagon s_p
  let new_d := 0.9 * d
  let new_s := new_d / ((1 + sqrt 5) / 2)
  let new_area := area_pentagon new_s
  (100 - new_area) / 100 * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percent_decrease_area_pentagon_l829_82931


namespace NUMINAMATH_GPT_problem1_problem2_l829_82903

variable {A B C a b c : ℝ}

-- Problem (1)
theorem problem1 (h : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B) : b = 2 * c := 
sorry

-- Problem (2)
theorem problem2 (a_eq : a = 1) (tanA_eq : Real.tan A = 2 * Real.sqrt 2) (b_eq_c : b = 2 * c): 
  Real.sqrt (c^2 * (1 - (Real.cos (A + B)))) = 2 * Real.sqrt 2 * b :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l829_82903


namespace NUMINAMATH_GPT_numberOfAntiPalindromes_l829_82944

-- Define what it means for a number to be an anti-palindrome in base 3
def isAntiPalindrome (n : ℕ) : Prop :=
  ∀ (a b : ℕ), a + b = 2 → a ≠ b

-- Define the constraint of no two consecutive digits being the same
def noConsecutiveDigits (digits : List ℕ) : Prop :=
  ∀ (i : ℕ), i < digits.length - 1 → digits.nthLe i sorry ≠ digits.nthLe (i + 1) sorry

-- We want to find the number of anti-palindromes less than 3^12 fulfilling both conditions
def countAntiPalindromes (m : ℕ) (base : ℕ) : ℕ :=
  sorry -- Placeholder definition for the count, to be implemented

-- The main theorem to prove
theorem numberOfAntiPalindromes : countAntiPalindromes (3^12) 3 = 126 :=
  sorry -- Proof to be filled

end NUMINAMATH_GPT_numberOfAntiPalindromes_l829_82944


namespace NUMINAMATH_GPT_union_sets_l829_82910

theorem union_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} :=
sorry

end NUMINAMATH_GPT_union_sets_l829_82910


namespace NUMINAMATH_GPT_range_of_a_l829_82923

noncomputable def f (x : ℝ) : ℝ := Real.log (2 + 3 * x) - (3 / 2) * x^2
noncomputable def f' (x : ℝ) : ℝ := (3 / (2 + 3 * x)) - 3 * x
noncomputable def valid_range (a : ℝ) : Prop := 
∀ x : ℝ, (1 / 6) ≤ x ∧ x ≤ (1 / 3) → |a - Real.log x| + Real.log (f' x + 3 * x) > 0

theorem range_of_a : { a : ℝ | valid_range a } = { a : ℝ | a ≠ Real.log (1 / 3) } := 
sorry

end NUMINAMATH_GPT_range_of_a_l829_82923


namespace NUMINAMATH_GPT_find_N_l829_82934

variable (a b c N : ℕ)

theorem find_N (h1 : a + b + c = 90) (h2 : a - 7 = N) (h3 : b + 7 = N) (h4 : 5 * c = N) : N = 41 := 
by
  sorry

end NUMINAMATH_GPT_find_N_l829_82934


namespace NUMINAMATH_GPT_total_hotdogs_sold_l829_82968

-- Define the number of small and large hotdogs
def small_hotdogs : ℕ := 58
def large_hotdogs : ℕ := 21

-- Define the total hotdogs
def total_hotdogs : ℕ := small_hotdogs + large_hotdogs

-- The Main Statement to prove the total number of hotdogs sold
theorem total_hotdogs_sold : total_hotdogs = 79 :=
by
  -- Proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_total_hotdogs_sold_l829_82968


namespace NUMINAMATH_GPT_men_in_first_group_l829_82924

theorem men_in_first_group (M : ℕ) (h1 : 20 * 30 * (480 / (20 * 30)) = 480) (h2 : M * 15 * (120 / (M * 15)) = 120) :
  M = 10 :=
by sorry

end NUMINAMATH_GPT_men_in_first_group_l829_82924


namespace NUMINAMATH_GPT_ratio_adults_children_l829_82932

-- Definitions based on conditions
def children := 45
def total_adults (A : ℕ) : Prop := (2 / 3 : ℚ) * A = 10

-- The theorem stating the problem
theorem ratio_adults_children :
  ∃ A, total_adults A ∧ (A : ℚ) / children = (1 / 3 : ℚ) :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_adults_children_l829_82932


namespace NUMINAMATH_GPT_borrowed_nickels_l829_82941

def n_original : ℕ := 87
def n_left : ℕ := 12
def n_borrowed : ℕ := n_original - n_left

theorem borrowed_nickels : n_borrowed = 75 := by
  sorry

end NUMINAMATH_GPT_borrowed_nickels_l829_82941


namespace NUMINAMATH_GPT_find_three_digit_integers_mod_l829_82979

theorem find_three_digit_integers_mod (n : ℕ) :
  (n % 7 = 3) ∧ (n % 8 = 6) ∧ (n % 5 = 2) ∧ (100 ≤ n) ∧ (n < 1000) :=
sorry

end NUMINAMATH_GPT_find_three_digit_integers_mod_l829_82979


namespace NUMINAMATH_GPT_a6_value_l829_82983

variable (a_n : ℕ → ℤ)

/-- Given conditions in the arithmetic sequence -/
def arithmetic_sequence_property (a_n : ℕ → ℤ) :=
  ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0)

/-- Given sum condition a_4 + a_5 + a_6 + a_7 + a_8 = 150 -/
def sum_condition :=
  a_n 4 + a_n 5 + a_n 6 + a_n 7 + a_n 8 = 150

theorem a6_value (h : arithmetic_sequence_property a_n) (hsum : sum_condition a_n) :
  a_n 6 = 30 := 
by
  sorry

end NUMINAMATH_GPT_a6_value_l829_82983


namespace NUMINAMATH_GPT_tech_gadget_cost_inr_l829_82989

def conversion_ratio (a b : ℝ) : Prop := a = b

theorem tech_gadget_cost_inr :
  (forall a b c : ℝ, conversion_ratio (a / b) c) →
  (forall a b c d : ℝ, conversion_ratio (a / b) c → conversion_ratio (a / d) c) →
  ∀ (n_usd : ℝ) (n_inr : ℝ) (cost_n : ℝ), 
    n_usd = 8 →
    n_inr = 5 →
    cost_n = 160 →
    cost_n / n_usd * n_inr = 100 :=
by
  sorry

end NUMINAMATH_GPT_tech_gadget_cost_inr_l829_82989


namespace NUMINAMATH_GPT_positive_difference_of_complementary_angles_l829_82904

-- Define the conditions
variables (a b : ℝ)
variable (h1 : a + b = 90)
variable (h2 : 5 * b = a)

-- Define the theorem we are proving
theorem positive_difference_of_complementary_angles (a b : ℝ) (h1 : a + b = 90) (h2 : 5 * b = a) :
  |a - b| = 60 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_complementary_angles_l829_82904


namespace NUMINAMATH_GPT_rectangle_ratio_l829_82996

theorem rectangle_ratio (A L : ℝ) (hA : A = 100) (hL : L = 20) :
  ∃ W : ℝ, A = L * W ∧ (L / W) = 4 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l829_82996


namespace NUMINAMATH_GPT_vicente_total_spent_l829_82927

def kilograms_of_rice := 5
def cost_per_kilogram_of_rice := 2
def pounds_of_meat := 3
def cost_per_pound_of_meat := 5

def total_spent := kilograms_of_rice * cost_per_kilogram_of_rice + pounds_of_meat * cost_per_pound_of_meat

theorem vicente_total_spent : total_spent = 25 := 
by
  sorry -- Proof would go here

end NUMINAMATH_GPT_vicente_total_spent_l829_82927


namespace NUMINAMATH_GPT_final_answer_l829_82971

theorem final_answer : (848 / 8) - 100 = 6 := 
by
  sorry

end NUMINAMATH_GPT_final_answer_l829_82971


namespace NUMINAMATH_GPT_convert_to_general_form_l829_82928

theorem convert_to_general_form (x : ℝ) :
  5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_convert_to_general_form_l829_82928


namespace NUMINAMATH_GPT_negation_of_p_l829_82905

open Real

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, exp x > log x

-- Theorem stating that the negation of p is as described
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, exp x ≤ log x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l829_82905


namespace NUMINAMATH_GPT_shadow_length_building_l829_82980

theorem shadow_length_building:
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  (height_flagstaff / shadow_flagstaff = height_building / expected_shadow_building) := by
  let height_flagstaff := 17.5
  let shadow_flagstaff := 40.25
  let height_building := 12.5
  let expected_shadow_building := 28.75
  sorry

end NUMINAMATH_GPT_shadow_length_building_l829_82980


namespace NUMINAMATH_GPT_range_of_a_l829_82926
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 1| + |2 * x - a|

theorem range_of_a (a : ℝ)
  (h : ∀ x : ℝ, f x a ≥ (1 / 4) * a ^ 2 + 1) : -2 ≤ a ∧ a ≤ 0 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l829_82926


namespace NUMINAMATH_GPT_surface_area_of_larger_prism_l829_82900

def volume_of_brick := 288
def number_of_bricks := 11
def target_surface_area := 1368

theorem surface_area_of_larger_prism
    (vol: ℕ := volume_of_brick)
    (num: ℕ := number_of_bricks)
    (target: ℕ := target_surface_area)
    (exists_a_b_h : ∃ (a b h : ℕ), a = 12 ∧ b = 8 ∧ h = 3)
    (large_prism_dimensions : ∃ (L W H : ℕ), L = 24 ∧ W = 12 ∧ H = 11):
    2 * (24 * 12 + 24 * 11 + 12 * 11) = target :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_larger_prism_l829_82900


namespace NUMINAMATH_GPT_Nina_total_problems_l829_82978

def Ruby_math_problems := 12
def Ruby_reading_problems := 4
def Ruby_science_problems := 5

def Nina_math_problems := 5 * Ruby_math_problems
def Nina_reading_problems := 9 * Ruby_reading_problems
def Nina_science_problems := 3 * Ruby_science_problems

def total_problems := Nina_math_problems + Nina_reading_problems + Nina_science_problems

theorem Nina_total_problems : total_problems = 111 :=
by
  sorry

end NUMINAMATH_GPT_Nina_total_problems_l829_82978


namespace NUMINAMATH_GPT_speed_of_current_l829_82998

variables (b c : ℝ)

theorem speed_of_current (h1 : b + c = 12) (h2 : b - c = 4) : c = 4 :=
sorry

end NUMINAMATH_GPT_speed_of_current_l829_82998


namespace NUMINAMATH_GPT_delta_f_l829_82942

open BigOperators

def f (n : ℕ) : ℕ := ∑ i in Finset.range n, (i + 1) * (n - i)

theorem delta_f (k : ℕ) : f (k + 1) - f k = ∑ i in Finset.range (k + 1), (i + 1) :=
by
  sorry

end NUMINAMATH_GPT_delta_f_l829_82942


namespace NUMINAMATH_GPT_linear_equation_solution_l829_82950

theorem linear_equation_solution (x : ℝ) (h : 1 - x = -3) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_solution_l829_82950


namespace NUMINAMATH_GPT_union_A_B_complement_intersection_A_B_l829_82913

-- Define universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | -5 ≤ x ∧ x ≤ -1 }

-- Define set B
def B : Set ℝ := { x | x ≥ -4 }

-- Prove A ∪ B = [-5, +∞)
theorem union_A_B : A ∪ B = { x : ℝ | -5 ≤ x } :=
by {
  sorry
}

-- Prove complement of A ∩ B with respect to U = (-∞, -4) ∪ (-1, +∞)
theorem complement_intersection_A_B : U \ (A ∩ B) = { x : ℝ | x < -4 } ∪ { x : ℝ | x > -1 } :=
by {
  sorry
}

end NUMINAMATH_GPT_union_A_B_complement_intersection_A_B_l829_82913


namespace NUMINAMATH_GPT_calum_disco_ball_budget_l829_82984

-- Defining the conditions
def n_d : ℕ := 4  -- Number of disco balls
def n_f : ℕ := 10  -- Number of food boxes
def p_f : ℕ := 25  -- Price per food box in dollars
def B : ℕ := 330  -- Total budget in dollars

-- Defining the expected result
def p_d : ℕ := 20  -- Cost per disco ball in dollars

-- Proof statement (no proof, just the statement)
theorem calum_disco_ball_budget :
  (10 * p_f + 4 * p_d = B) → (p_d = 20) :=
by
  sorry

end NUMINAMATH_GPT_calum_disco_ball_budget_l829_82984


namespace NUMINAMATH_GPT_total_bill_cost_l829_82975

-- Definitions of costs and conditions
def curtis_meal_cost : ℝ := 16.00
def rob_meal_cost : ℝ := 18.00
def total_cost_before_discount : ℝ := curtis_meal_cost + rob_meal_cost
def discount_rate : ℝ := 0.5
def time_of_meal : ℝ := 3.0

-- Condition for discount applicability
def discount_applicable : Prop := 2.0 ≤ time_of_meal ∧ time_of_meal ≤ 4.0

-- Total cost with discount applied
def cost_with_discount (total_cost : ℝ) (rate : ℝ) : ℝ := total_cost * rate

-- Theorem statement we need to prove
theorem total_bill_cost :
  discount_applicable →
  cost_with_discount total_cost_before_discount discount_rate = 17.00 :=
by
  sorry

end NUMINAMATH_GPT_total_bill_cost_l829_82975


namespace NUMINAMATH_GPT_sum_of_areas_of_circles_l829_82961

theorem sum_of_areas_of_circles 
  (r s t : ℝ) 
  (h1 : r + s = 6) 
  (h2 : r + t = 8) 
  (h3 : s + t = 10) 
  : π * r^2 + π * s^2 + π * t^2 = 56 * π :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_areas_of_circles_l829_82961


namespace NUMINAMATH_GPT_all_positive_l829_82974

theorem all_positive (a1 a2 a3 a4 a5 a6 a7 : ℝ)
  (h1 : a1 + a2 + a3 + a4 > a5 + a6 + a7)
  (h2 : a1 + a2 + a3 + a5 > a4 + a6 + a7)
  (h3 : a1 + a2 + a3 + a6 > a4 + a5 + a7)
  (h4 : a1 + a2 + a3 + a7 > a4 + a5 + a6)
  (h5 : a1 + a2 + a4 + a5 > a3 + a6 + a7)
  (h6 : a1 + a2 + a4 + a6 > a3 + a5 + a7)
  (h7 : a1 + a2 + a4 + a7 > a3 + a5 + a6)
  (h8 : a1 + a2 + a5 + a6 > a3 + a4 + a7)
  (h9 : a1 + a2 + a5 + a7 > a3 + a4 + a6)
  (h10 : a1 + a2 + a6 + a7 > a3 + a4 + a5)
  (h11 : a1 + a3 + a4 + a5 > a2 + a6 + a7)
  (h12 : a1 + a3 + a4 + a6 > a2 + a5 + a7)
  (h13 : a1 + a3 + a4 + a7 > a2 + a5 + a6)
  (h14 : a1 + a3 + a5 + a6 > a2 + a4 + a7)
  (h15 : a1 + a3 + a5 + a7 > a2 + a4 + a6)
  (h16 : a1 + a3 + a6 + a7 > a2 + a4 + a5)
  (h17 : a1 + a4 + a5 + a6 > a2 + a3 + a7)
  (h18 : a1 + a4 + a5 + a7 > a2 + a3 + a6)
  (h19 : a1 + a4 + a6 + a7 > a2 + a3 + a5)
  (h20 : a1 + a5 + a6 + a7 > a2 + a3 + a4)
  (h21 : a2 + a3 + a4 + a5 > a1 + a6 + a7)
  (h22 : a2 + a3 + a4 + a6 > a1 + a5 + a7)
  (h23 : a2 + a3 + a4 + a7 > a1 + a5 + a6)
  (h24 : a2 + a3 + a5 + a6 > a1 + a4 + a7)
  (h25 : a2 + a3 + a5 + a7 > a1 + a4 + a6)
  (h26 : a2 + a3 + a6 + a7 > a1 + a4 + a5)
  (h27 : a2 + a4 + a5 + a6 > a1 + a3 + a7)
  (h28 : a2 + a4 + a5 + a7 > a1 + a3 + a6)
  (h29 : a2 + a4 + a6 + a7 > a1 + a3 + a5)
  (h30 : a2 + a5 + a6 + a7 > a1 + a3 + a4)
  (h31 : a3 + a4 + a5 + a6 > a1 + a2 + a7)
  (h32 : a3 + a4 + a5 + a7 > a1 + a2 + a6)
  (h33 : a3 + a4 + a6 + a7 > a1 + a2 + a5)
  (h34 : a3 + a5 + a6 + a7 > a1 + a2 + a4)
  (h35 : a4 + a5 + a6 + a7 > a1 + a2 + a3)
: a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0 ∧ a5 > 0 ∧ a6 > 0 ∧ a7 > 0 := 
sorry

end NUMINAMATH_GPT_all_positive_l829_82974


namespace NUMINAMATH_GPT_paving_stone_proof_l829_82966

noncomputable def paving_stone_width (length_court : ℝ) (width_court : ℝ) 
                                      (num_stones: ℕ) (stone_length: ℝ) : ℝ :=
  let area_court := length_court * width_court
  let area_stone := stone_length * (area_court / (num_stones * stone_length))
  area_court / area_stone

theorem paving_stone_proof :
  paving_stone_width 50 16.5 165 2.5 = 2 :=
sorry

end NUMINAMATH_GPT_paving_stone_proof_l829_82966


namespace NUMINAMATH_GPT_gcd_98_63_l829_82999

-- Definition of gcd
def gcd_euclidean := ∀ (a b : ℕ), ∃ (g : ℕ), gcd a b = g

-- Statement of the problem using Lean
theorem gcd_98_63 : gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_GPT_gcd_98_63_l829_82999


namespace NUMINAMATH_GPT_hazel_salmon_caught_l829_82952

-- Define the conditions
def father_salmon_caught : Nat := 27
def total_salmon_caught : Nat := 51

-- Define the main statement to be proved
theorem hazel_salmon_caught : total_salmon_caught - father_salmon_caught = 24 := by
  sorry

end NUMINAMATH_GPT_hazel_salmon_caught_l829_82952


namespace NUMINAMATH_GPT_smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l829_82933

theorem smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 
             (n % 9 = 0) ∧ 
             (∃ d1 d2 d3 d4 : ℕ, 
               d1 * 1000 + d2 * 100 + d3 * 10 + d4 = n ∧ 
               d1 % 2 = 1 ∧ 
               d2 % 2 = 0 ∧ 
               d3 % 2 = 0 ∧ 
               d4 % 2 = 0) ∧ 
             (∀ m : ℕ, (1000 ≤ m ∧ m < 10000 ∧ m % 9 = 0 ∧ 
               ∃ e1 e2 e3 e4 : ℕ, 
                 e1 * 1000 + e2 * 100 + e3 * 10 + e4 = m ∧ 
                 e1 % 2 = 1 ∧ 
                 e2 % 2 = 0 ∧ 
                 e3 % 2 = 0 ∧ 
                 e4 % 2 = 0) → n ≤ m) ∧ 
             n = 1026 :=
sorry

end NUMINAMATH_GPT_smallest_positive_four_digit_number_divisible_by_9_with_even_and_odd_l829_82933


namespace NUMINAMATH_GPT_inequality_problem_l829_82902

theorem inequality_problem {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a = a + b + c) :
  a^2 + b^2 + c^2 + 2 * a * b * c ≥ 5 :=
sorry

end NUMINAMATH_GPT_inequality_problem_l829_82902


namespace NUMINAMATH_GPT_min_value_f_l829_82997

def f (x : ℝ) : ℝ := |2 * x - 1| + |3 * x - 2| + |4 * x - 3| + |5 * x - 4|

theorem min_value_f : (∃ x : ℝ, ∀ y : ℝ, f y ≥ f x) := 
sorry

end NUMINAMATH_GPT_min_value_f_l829_82997


namespace NUMINAMATH_GPT_max_band_members_l829_82963

theorem max_band_members (n : ℤ) (h1 : 20 * n % 31 = 11) (h2 : 20 * n < 1200) : 20 * n = 1100 :=
sorry

end NUMINAMATH_GPT_max_band_members_l829_82963


namespace NUMINAMATH_GPT_evaluate_expression_l829_82962

theorem evaluate_expression : 
  - (16 / 2 * 8 - 72 + 4^2) = -8 :=
by 
  -- here, the proof would typically go
  sorry

end NUMINAMATH_GPT_evaluate_expression_l829_82962


namespace NUMINAMATH_GPT_no_values_less_than_180_l829_82921

/-- Given that w and n are positive integers less than 180 
    such that w % 13 = 2 and n % 8 = 5, 
    prove that there are no such values for w and n. -/
theorem no_values_less_than_180 (w n : ℕ) (hw : w < 180) (hn : n < 180) 
  (h1 : w % 13 = 2) (h2 : n % 8 = 5) : false :=
by
  sorry

end NUMINAMATH_GPT_no_values_less_than_180_l829_82921


namespace NUMINAMATH_GPT_max_min_P_l829_82930

theorem max_min_P (a b c : ℝ) (h : |a + b| + |b + c| + |c + a| = 8) :
  (a^2 + b^2 + c^2 = 48) ∨ (a^2 + b^2 + c^2 = 16 / 3) :=
sorry

end NUMINAMATH_GPT_max_min_P_l829_82930


namespace NUMINAMATH_GPT_kyler_wins_zero_l829_82960

-- Definitions based on conditions provided
def peter_games_won : ℕ := 5
def peter_games_lost : ℕ := 3
def emma_games_won : ℕ := 4
def emma_games_lost : ℕ := 4
def kyler_games_lost : ℕ := 4

-- Number of games each player played
def peter_total_games : ℕ := peter_games_won + peter_games_lost
def emma_total_games : ℕ := emma_games_won + emma_games_lost
def kyler_total_games (k : ℕ) : ℕ := k + kyler_games_lost

-- Step 1: total number of games in the tournament
def total_games (k : ℕ) : ℕ := (peter_total_games + emma_total_games + kyler_total_games k) / 2

-- Step 2: Total games equation
def games_equation (k : ℕ) : Prop := 
  (peter_games_won + emma_games_won + k = total_games k)

-- The proof problem, we need to prove Kyler's wins
theorem kyler_wins_zero : games_equation 0 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_kyler_wins_zero_l829_82960


namespace NUMINAMATH_GPT_minimum_value_of_k_l829_82976

theorem minimum_value_of_k (x y : ℝ) (h : x * (x - 1) ≤ y * (1 - y)) : x^2 + y^2 ≤ 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_k_l829_82976


namespace NUMINAMATH_GPT_valid_5_digit_numbers_l829_82973

noncomputable def num_valid_numbers (d : ℕ) (h : d ≠ 7) (h_valid : d < 10) (h_pos : d ≠ 0) : ℕ :=
  let choices_first_place := 7   -- choices for the first digit (1-9, excluding d and 7)
  let choices_other_places := 8  -- choices for other digits (0-9, excluding d and 7)
  choices_first_place * choices_other_places ^ 4

theorem valid_5_digit_numbers (d : ℕ) (h_d_ne_7 : d ≠ 7) (h_d_valid : d < 10) (h_d_pos : d ≠ 0) :
  num_valid_numbers d h_d_ne_7 h_d_valid h_d_pos = 28672 := sorry

end NUMINAMATH_GPT_valid_5_digit_numbers_l829_82973


namespace NUMINAMATH_GPT_purchasing_plans_count_l829_82940

theorem purchasing_plans_count :
  (∃ (x y : ℕ), 15 * x + 20 * y = 360) ∧ ∀ (x y : ℕ), 15 * x + 20 * y = 360 → (x % 4 = 0) ∧ (y = 18 - (3 / 4) * x) := sorry

end NUMINAMATH_GPT_purchasing_plans_count_l829_82940


namespace NUMINAMATH_GPT_parabola_vertex_x_coordinate_l829_82948

theorem parabola_vertex_x_coordinate (a b c : ℝ) (h1 : c = 0) (h2 : 16 * a + 4 * b = 0) (h3 : 9 * a + 3 * b = 9) : 
    -b / (2 * a) = 2 :=
by 
  -- You can start by adding a proof here
  sorry

end NUMINAMATH_GPT_parabola_vertex_x_coordinate_l829_82948


namespace NUMINAMATH_GPT_football_games_total_l829_82955

def total_football_games_per_season (games_per_month : ℝ) (num_months : ℝ) : ℝ :=
  games_per_month * num_months

theorem football_games_total (games_per_month : ℝ) (num_months : ℝ) (total_games : ℝ) :
  games_per_month = 323.0 ∧ num_months = 17.0 ∧ total_games = 5491.0 →
  total_football_games_per_season games_per_month num_months = total_games :=
by
  intros h
  have h1 : games_per_month = 323.0 := h.1
  have h2 : num_months = 17.0 := h.2.1
  have h3 : total_games = 5491.0 := h.2.2
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_football_games_total_l829_82955


namespace NUMINAMATH_GPT_choosing_officers_l829_82922

noncomputable def total_ways_to_choose_officers (members : List String) (boys : ℕ) (girls : ℕ) : ℕ :=
  let total_members := boys + girls
  let president_choices := total_members
  let vice_president_choices := boys - 1 + girls - 1
  let remaining_members := total_members - 2
  president_choices * vice_president_choices * remaining_members

theorem choosing_officers (members : List String) (boys : ℕ) (girls : ℕ) :
  boys = 15 → girls = 15 → members.length = 30 → total_ways_to_choose_officers members boys girls = 11760 :=
by
  intros hboys hgirls htotal
  rw [hboys, hgirls]
  sorry

end NUMINAMATH_GPT_choosing_officers_l829_82922


namespace NUMINAMATH_GPT_initial_roses_l829_82916

theorem initial_roses (R : ℕ) (initial_orchids : ℕ) (current_orchids : ℕ) (current_roses : ℕ) (added_orchids : ℕ) (added_roses : ℕ) :
  initial_orchids = 84 →
  current_orchids = 91 →
  current_roses = 14 →
  added_orchids = current_orchids - initial_orchids →
  added_roses = added_orchids →
  (R + added_roses = current_roses) →
  R = 7 :=
by
  sorry

end NUMINAMATH_GPT_initial_roses_l829_82916


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l829_82970

section
  variable (a b c d : Int)

  theorem problem1 : -27 + (-32) + (-8) + 72 = 5 := by
    sorry

  theorem problem2 : -4 - 2 * 32 + (-2 * 32) = -132 := by
    sorry

  theorem problem3 : (-48 : Int) / (-2 : Int)^3 - (-25 : Int) * (-4 : Int) + (-2 : Int)^3 = -102 := by
    sorry

  theorem problem4 : (-3 : Int)^2 - (3 / 2)^3 * (2 / 9) - 6 / (-(2 / 3))^3 = -12 := by
    sorry
end

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l829_82970


namespace NUMINAMATH_GPT_find_k_l829_82906

def f (x : ℤ) : ℤ := 3*x^2 - 2*x + 4
def g (x : ℤ) (k : ℤ) : ℤ := x^2 - k * x - 6

theorem find_k : 
  ∃ k : ℤ, f 10 - g 10 k = 10 ∧ k = -18 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_l829_82906


namespace NUMINAMATH_GPT_mod_remainder_1287_1499_l829_82920

theorem mod_remainder_1287_1499 : (1287 * 1499) % 300 = 213 := 
by 
  sorry

end NUMINAMATH_GPT_mod_remainder_1287_1499_l829_82920


namespace NUMINAMATH_GPT_average_goals_per_game_l829_82901

theorem average_goals_per_game
  (slices_per_pizza : ℕ := 12)
  (total_pizzas : ℕ := 6)
  (total_games : ℕ := 8)
  (total_slices : ℕ := total_pizzas * slices_per_pizza)
  (total_goals : ℕ := total_slices)
  (average_goals : ℕ := total_goals / total_games) :
  average_goals = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_goals_per_game_l829_82901


namespace NUMINAMATH_GPT_trig_identity_t_half_l829_82953

theorem trig_identity_t_half (a t : ℝ) (ht : t = Real.tan (a / 2)) :
  Real.sin a = (2 * t) / (1 + t^2) ∧
  Real.cos a = (1 - t^2) / (1 + t^2) ∧
  Real.tan a = (2 * t) / (1 - t^2) := 
sorry

end NUMINAMATH_GPT_trig_identity_t_half_l829_82953


namespace NUMINAMATH_GPT_sock_problem_l829_82988

def sock_pair_count (total_socks : Nat) (socks_distribution : List (String × Nat)) (target_color : String) (different_color : String) : Nat :=
  if target_color = different_color then 0
  else match socks_distribution with
    | [] => 0
    | (color, count) :: tail =>
        if color = target_color then count * socks_distribution.foldl (λ acc (col_count : String × Nat) =>
          if col_count.fst ≠ target_color then acc + col_count.snd else acc) 0
        else sock_pair_count total_socks tail target_color different_color

theorem sock_problem : sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "white" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "brown" +
                        sock_pair_count 16 [("white", 4), ("brown", 4), ("blue", 4), ("red", 4)] "red" "blue" =
                        48 :=
by sorry

end NUMINAMATH_GPT_sock_problem_l829_82988


namespace NUMINAMATH_GPT_johns_out_of_pocket_expense_l829_82907

-- Define the conditions given in the problem
def old_system_cost : ℤ := 250
def old_system_trade_in_value : ℤ := (80 * old_system_cost) / 100
def new_system_initial_cost : ℤ := 600
def new_system_discount : ℤ := (25 * new_system_initial_cost) / 100
def new_system_final_cost : ℤ := new_system_initial_cost - new_system_discount

-- Define the amount of money that came out of John's pocket
def out_of_pocket_expense : ℤ := new_system_final_cost - old_system_trade_in_value

-- State the theorem that needs to be proven
theorem johns_out_of_pocket_expense : out_of_pocket_expense = 250 := by
  sorry

end NUMINAMATH_GPT_johns_out_of_pocket_expense_l829_82907


namespace NUMINAMATH_GPT_probability_manu_wins_l829_82994

theorem probability_manu_wins :
  ∑' (n : ℕ), (1 / 2)^(4 * (n + 1)) = 1 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_manu_wins_l829_82994


namespace NUMINAMATH_GPT_total_whipped_cream_l829_82985

theorem total_whipped_cream (cream_from_farm : ℕ) (cream_to_buy : ℕ) (total_cream : ℕ) 
  (h1 : cream_from_farm = 149) 
  (h2 : cream_to_buy = 151) 
  (h3 : total_cream = cream_from_farm + cream_to_buy) : 
  total_cream = 300 :=
sorry

end NUMINAMATH_GPT_total_whipped_cream_l829_82985


namespace NUMINAMATH_GPT_intersection_of_function_and_inverse_l829_82918

theorem intersection_of_function_and_inverse (m : ℝ) :
  (∀ x y : ℝ, y = Real.sqrt (x - m) ↔ x = y^2 + m) →
  (∃ x : ℝ, Real.sqrt (x - m) = x) ↔ (m ≤ 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_function_and_inverse_l829_82918


namespace NUMINAMATH_GPT_inequality_greater_sqrt_two_l829_82917

theorem inequality_greater_sqrt_two (x y : ℝ) (hx : x > Real.sqrt 2) (hy : y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_greater_sqrt_two_l829_82917


namespace NUMINAMATH_GPT_max_value_arithmetic_sequence_l829_82914

theorem max_value_arithmetic_sequence
  (a : ℕ → ℝ)
  (a1 d : ℝ)
  (h1 : a 1 = a1)
  (h_diff : ∀ n : ℕ, a (n + 1) = a n + d)
  (ha1_pos : a1 > 0)
  (hd_pos : d > 0)
  (h1_2 : a1 + (a1 + d) ≤ 60)
  (h2_3 : (a1 + d) + (a1 + 2 * d) ≤ 100) :
  5 * a1 + (a1 + 4 * d) ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_value_arithmetic_sequence_l829_82914


namespace NUMINAMATH_GPT_intersection_complement_P_Q_l829_82981

def P (x : ℝ) : Prop := x - 1 ≤ 0
def Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

def complement_P (x : ℝ) : Prop := ¬ P x

theorem intersection_complement_P_Q :
  {x : ℝ | complement_P x} ∩ {x : ℝ | Q x} = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_P_Q_l829_82981


namespace NUMINAMATH_GPT_opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l829_82912

theorem opposite_number_of_2_eq_neg2 : -2 = -2 := by
  sorry

theorem abs_val_eq_2_iff_eq_2_or_neg2 (x : ℝ) : abs x = 2 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_GPT_opposite_number_of_2_eq_neg2_abs_val_eq_2_iff_eq_2_or_neg2_l829_82912


namespace NUMINAMATH_GPT_simplify_sum_of_squares_roots_l829_82965

theorem simplify_sum_of_squares_roots :
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_sum_of_squares_roots_l829_82965


namespace NUMINAMATH_GPT_sum_is_odd_square_expression_is_odd_l829_82919

theorem sum_is_odd_square_expression_is_odd (a b c : ℤ) (h : (a + b + c) % 2 = 1) : 
  (a^2 + b^2 - c^2 + 2 * a * b) % 2 = 1 :=
sorry

end NUMINAMATH_GPT_sum_is_odd_square_expression_is_odd_l829_82919


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t3_l829_82967

open Real

noncomputable def displacement (t : ℝ) : ℝ := 4 - 2 * t + t ^ 2

theorem instantaneous_velocity_at_t3 : deriv displacement 3 = 4 := 
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t3_l829_82967


namespace NUMINAMATH_GPT_vasya_max_pencils_l829_82958

theorem vasya_max_pencils (money_for_pencils : ℕ) (rebate_20 : ℕ) (rebate_5 : ℕ) :
  money_for_pencils = 30 → rebate_20 = 25 → rebate_5 = 10 → ∃ max_pencils, max_pencils = 36 :=
by
  intros h_money h_r20 h_r5
  sorry

end NUMINAMATH_GPT_vasya_max_pencils_l829_82958


namespace NUMINAMATH_GPT_percentage_difference_l829_82972

-- Define the quantities involved
def milk_in_A : ℕ := 1264
def transferred_milk : ℕ := 158

-- Define the quantities of milk in container B and C after transfer
noncomputable def quantity_in_B : ℕ := milk_in_A / 2
noncomputable def quantity_in_C : ℕ := quantity_in_B

-- Prove that the percentage difference between the quantity of milk in container B
-- and the capacity of container A is 50%
theorem percentage_difference :
  ((milk_in_A - quantity_in_B) * 100 / milk_in_A) = 50 := sorry

end NUMINAMATH_GPT_percentage_difference_l829_82972


namespace NUMINAMATH_GPT_parabola_vertex_x_coord_l829_82993

theorem parabola_vertex_x_coord (a b c : ℝ)
  (h1 : 5 = a * 2^2 + b * 2 + c)
  (h2 : 5 = a * 8^2 + b * 8 + c)
  (h3 : 11 = a * 9^2 + b * 9 + c) :
  5 = (2 + 8) / 2 := 
sorry

end NUMINAMATH_GPT_parabola_vertex_x_coord_l829_82993


namespace NUMINAMATH_GPT_last_two_digits_of_large_exponent_l829_82987

theorem last_two_digits_of_large_exponent :
  (9 ^ (8 ^ (7 ^ (6 ^ (5 ^ (4 ^ (3 ^ 2))))))) % 100 = 21 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_large_exponent_l829_82987


namespace NUMINAMATH_GPT_unique_sequence_l829_82982

theorem unique_sequence (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 1) ^ 2 = 1 + (n + 2021) * a n) →
  (∀ n : ℕ, a n = n + 2019) :=
by
  sorry

end NUMINAMATH_GPT_unique_sequence_l829_82982


namespace NUMINAMATH_GPT_root_at_neg_x0_l829_82925

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom x0_root : ∃ x0, f x0 = Real.exp x0

-- Theorem
theorem root_at_neg_x0 : 
  (∃ x0, (f (-x0) * Real.exp (-x0) + 1 = 0))
  → (∃ x0, (f x0 * Real.exp x0 + 1 = 0)) := 
sorry

end NUMINAMATH_GPT_root_at_neg_x0_l829_82925


namespace NUMINAMATH_GPT_bridgette_has_4_birds_l829_82945

/-
Conditions:
1. Bridgette has 2 dogs.
2. Bridgette has 3 cats.
3. Bridgette has some birds.
4. She gives the dogs a bath twice a month.
5. She gives the cats a bath once a month.
6. She gives the birds a bath once every 4 months.
7. In a year, she gives a total of 96 baths.
-/

def num_birds (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ) : ℕ :=
  let yearly_dog_baths := num_dogs * dog_baths_per_month * 12
  let yearly_cat_baths := num_cats * cat_baths_per_month * 12
  let birds_baths := total_baths_per_year - (yearly_dog_baths + yearly_cat_baths)
  let baths_per_bird_per_year := 12 / bird_baths_per_4_months
  birds_baths / baths_per_bird_per_year

theorem bridgette_has_4_birds :
  ∀ (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ),
    num_dogs = 2 →
    num_cats = 3 →
    dog_baths_per_month = 2 →
    cat_baths_per_month = 1 →
    bird_baths_per_4_months = 4 →
    total_baths_per_year = 96 →
    num_birds num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year = 4 :=
by
  intros
  sorry


end NUMINAMATH_GPT_bridgette_has_4_birds_l829_82945


namespace NUMINAMATH_GPT_factor_expression_l829_82911

theorem factor_expression (x : ℝ) : 
  5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l829_82911


namespace NUMINAMATH_GPT_ratio_sandra_amy_ruth_l829_82946

/-- Given the amounts received by Sandra and Amy, and an unknown amount received by Ruth,
    the ratio of the money shared between Sandra, Amy, and Ruth is 2:1:R/50. -/
theorem ratio_sandra_amy_ruth (R : ℝ) (hAmy : 50 > 0) (hSandra : 100 > 0) :
  (100 : ℝ) / 50 = 2 ∧ (50 : ℝ) / 50 = 1 ∧ ∃ (R : ℝ), (100/50 : ℝ) = 2 ∧ (50/50 : ℝ) = 1 ∧ (R / 50 : ℝ) = (R / 50 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_ratio_sandra_amy_ruth_l829_82946


namespace NUMINAMATH_GPT_divisor_increase_by_10_5_l829_82938

def condition_one (n t : ℕ) : Prop :=
  n * (t + 7) = t * (n + 2)

def condition_two (n t z : ℕ) : Prop :=
  n * (t + z) = t * (n + 3)

theorem divisor_increase_by_10_5 (n t : ℕ) (hz : ℕ) (nz : n ≠ 0) (tz : t ≠ 0)
  (h1 : condition_one n t) (h2 : condition_two n t hz) : hz = 21 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_divisor_increase_by_10_5_l829_82938


namespace NUMINAMATH_GPT_number_line_steps_l829_82943

theorem number_line_steps (total_steps : ℕ) (total_distance : ℕ) (steps_taken : ℕ) (result_distance : ℕ) 
  (h1 : total_distance = 36) (h2 : total_steps = 9) (h3 : steps_taken = 6) : 
  result_distance = (steps_taken * (total_distance / total_steps)) → result_distance = 24 :=
by
  intros H
  sorry

end NUMINAMATH_GPT_number_line_steps_l829_82943


namespace NUMINAMATH_GPT_quadrilateral_areas_product_l829_82957

noncomputable def areas_product_property (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) : Prop :=
  (S_ADP * S_BCP * S_ABP * S_CDP) % 10000 ≠ 1988
  
theorem quadrilateral_areas_product (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) :
  areas_product_property S_ADP S_ABP S_CDP S_BCP h1 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_areas_product_l829_82957


namespace NUMINAMATH_GPT_car_speed_on_local_roads_l829_82949

theorem car_speed_on_local_roads
    (v : ℝ) -- Speed of the car on local roads
    (h1 : v > 0) -- The speed is positive
    (h2 : 40 / v + 3 = 5) -- Given equation based on travel times and distances
    : v = 20 := 
sorry

end NUMINAMATH_GPT_car_speed_on_local_roads_l829_82949


namespace NUMINAMATH_GPT_number_square_l829_82936

-- Define conditions.
def valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d * d ≤ 9

-- Main statement.
theorem number_square (n : ℕ) (valid_digits : ∀ d, d ∈ [n / 100, (n / 10) % 10, n % 10] → valid_digit d) : 
  n = 233 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_square_l829_82936


namespace NUMINAMATH_GPT_sum_of_first_20_primes_l829_82977

theorem sum_of_first_20_primes :
  ( [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71].sum = 639 ) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_20_primes_l829_82977


namespace NUMINAMATH_GPT_evaluate_g_at_3_l829_82915

def g : ℝ → ℝ := fun x => x^2 - 3 * x + 2

theorem evaluate_g_at_3 : g 3 = 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_3_l829_82915


namespace NUMINAMATH_GPT_redistributed_gnomes_l829_82964

def WestervilleWoods : ℕ := 20
def RavenswoodForest := 4 * WestervilleWoods
def GreenwoodGrove := (5 * RavenswoodForest) / 4
def OwnerTakes (f: ℕ) (p: ℚ) := p * f

def RemainingGnomes (initial: ℕ) (p: ℚ) := initial - (OwnerTakes initial p)

def TotalRemainingGnomes := 
  (RemainingGnomes RavenswoodForest (40 / 100)) + 
  (RemainingGnomes WestervilleWoods (30 / 100)) + 
  (RemainingGnomes GreenwoodGrove (50 / 100))

def GnomesPerForest := TotalRemainingGnomes / 3

theorem redistributed_gnomes : 
  2 * 37 + 38 = TotalRemainingGnomes := by
  sorry

end NUMINAMATH_GPT_redistributed_gnomes_l829_82964


namespace NUMINAMATH_GPT_arithmetic_square_root_of_nine_l829_82908

theorem arithmetic_square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_nine_l829_82908


namespace NUMINAMATH_GPT_problem1_problem2_l829_82937

noncomputable def f (x a b : ℝ) : ℝ := 2 * x ^ 2 - 2 * a * x + b

noncomputable def set_A (a b : ℝ) : Set ℝ := {x | f x a b > 0 }

noncomputable def set_B (t : ℝ) : Set ℝ := {x | |x - t| ≤ 1 }

theorem problem1 (a b : ℝ) (h : f (-1) a b = -8) :
  (∀ x, x ∈ (set_A a b)ᶜ ∪ set_B 1 ↔ -3 ≤ x ∧ x ≤ 2) :=
  sorry

theorem problem2 (a b : ℝ) (t : ℝ) (h : f (-1) a b = -8) (h_not_P : (set_A a b) ∩ (set_B t) = ∅) :
  -2 ≤ t ∧ t ≤ 0 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l829_82937


namespace NUMINAMATH_GPT_problem_proof_l829_82939

-- Problem statement
variable (f : ℕ → ℕ)

-- Condition: if f(k) ≥ k^2 then f(k+1) ≥ (k+1)^2
variable (h : ∀ k, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)

-- Additional condition: f(4) ≥ 25
variable (h₀ : f 4 ≥ 25)

-- To prove: ∀ k ≥ 4, f(k) ≥ k^2
theorem problem_proof : ∀ k ≥ 4, f k ≥ k^2 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l829_82939
