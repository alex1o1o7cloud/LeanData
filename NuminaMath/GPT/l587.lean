import Mathlib

namespace height_of_spherical_cap_case1_height_of_spherical_cap_case2_l587_58748

variable (R : ℝ) (c : ℝ)
variable (h_c_gt_1 : c > 1)

-- Case 1: Not including the circular cap in the surface area
theorem height_of_spherical_cap_case1 : ∃ m : ℝ, m = (2 * R * (c - 1)) / c :=
by
  sorry

-- Case 2: Including the circular cap in the surface area
theorem height_of_spherical_cap_case2 : ∃ m : ℝ, m = (2 * R * (c - 2)) / (c - 1) :=
by
  sorry

end height_of_spherical_cap_case1_height_of_spherical_cap_case2_l587_58748


namespace tan_addition_formula_15_30_l587_58784

-- Define tangent function for angles in degrees.
noncomputable def tanDeg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem for the given problem
theorem tan_addition_formula_15_30 :
  tanDeg 15 + tanDeg 30 + tanDeg 15 * tanDeg 30 = 1 :=
by
  -- Here we use the given conditions and properties in solution
  sorry

end tan_addition_formula_15_30_l587_58784


namespace larger_exceeds_smaller_by_5_l587_58734

-- Define the problem's parameters and conditions.
variables (x n m : ℕ)
variables (subtracted : ℕ := 5)

-- Define the two numbers based on the given ratio.
def larger_number := 6 * x
def smaller_number := 5 * x

-- Condition when a number is subtracted
def new_ratio_condition := (larger_number - subtracted) * 4 = (smaller_number - subtracted) * 5

-- The main goal
theorem larger_exceeds_smaller_by_5 (hx : new_ratio_condition) : larger_number - smaller_number = 5 :=
sorry

end larger_exceeds_smaller_by_5_l587_58734


namespace vasya_days_without_purchase_l587_58741

theorem vasya_days_without_purchase
  (x y z w : ℕ)
  (h1 : x + y + z + w = 15)
  (h2 : 9 * x + 4 * z = 30)
  (h3 : 2 * y + z = 9) :
  w = 7 :=
by
  sorry

end vasya_days_without_purchase_l587_58741


namespace construct_right_triangle_l587_58708

theorem construct_right_triangle (c m n : ℝ) (hc : c > 0) (hm : m > 0) (hn : n > 0) : 
  ∃ a b : ℝ, a^2 + b^2 = c^2 ∧ a / b = m / n :=
by
  sorry

end construct_right_triangle_l587_58708


namespace intersecting_lines_sum_l587_58759

theorem intersecting_lines_sum (a b : ℝ) 
  (h1 : a * 1 + 1 + 1 = 0)
  (h2 : 2 * 1 - b * 1 - 1 = 0) : 
  a + b = -1 := 
by 
  have ha : a = -2 := by linarith [h1]
  have hb : b = 1 := by linarith [h2]
  rw [ha, hb]
  exact by norm_num

end intersecting_lines_sum_l587_58759


namespace brittany_first_test_grade_l587_58793

theorem brittany_first_test_grade (x : ℤ) (h1 : (x + 84) / 2 = 81) : x = 78 :=
by
  sorry

end brittany_first_test_grade_l587_58793


namespace probability_of_problem_being_solved_l587_58766

-- Define the probabilities of solving the problem.
def prob_A_solves : ℚ := 1 / 5
def prob_B_solves : ℚ := 1 / 3

-- Define the proof statement
theorem probability_of_problem_being_solved :
  (1 - ((1 - prob_A_solves) * (1 - prob_B_solves))) = 7 / 15 :=
by
  sorry

end probability_of_problem_being_solved_l587_58766


namespace race_winner_and_liar_l587_58751

def Alyosha_statement (pos : ℕ → Prop) : Prop := ¬ pos 1 ∧ ¬ pos 4
def Borya_statement (pos : ℕ → Prop) : Prop := ¬ pos 4
def Vanya_statement (pos : ℕ → Prop) : Prop := pos 1
def Grisha_statement (pos : ℕ → Prop) : Prop := pos 4

def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop := 
  (s1 ∧ s2 ∧ s3 ∧ ¬ s4) ∨
  (s1 ∧ s2 ∧ ¬ s3 ∧ s4) ∨
  (s1 ∧ ¬ s2 ∧ s3 ∧ s4) ∨
  (¬ s1 ∧ s2 ∧ s3 ∧ s4)

def race_result (pos : ℕ → Prop) : Prop :=
  Vanya_statement pos ∧
  three_true_one_false (Alyosha_statement pos) (Borya_statement pos) (Vanya_statement pos) (Grisha_statement pos) ∧
  Borya_statement pos = false

theorem race_winner_and_liar:
  ∃ (pos : ℕ → Prop), race_result pos :=
sorry

end race_winner_and_liar_l587_58751


namespace min_cubes_required_l587_58743

/--
A lady builds a box with dimensions 10 cm length, 18 cm width, and 4 cm height using 12 cubic cm cubes. Prove that the minimum number of cubes required to build the box is 60.
-/
def min_cubes_for_box (length width height volume_cube : ℕ) : ℕ :=
  (length * width * height) / volume_cube

theorem min_cubes_required :
  min_cubes_for_box 10 18 4 12 = 60 :=
by
  -- The proof details are omitted.
  sorry

end min_cubes_required_l587_58743


namespace interval_necessary_not_sufficient_l587_58716

theorem interval_necessary_not_sufficient :
  (∀ x, x^2 - x - 2 = 0 → (-1 ≤ x ∧ x ≤ 2)) ∧ (∃ x, x^2 - x - 2 = 0 ∧ ¬(-1 ≤ x ∧ x ≤ 2)) → False :=
by
  sorry

end interval_necessary_not_sufficient_l587_58716


namespace remainder_approximately_14_l587_58703

def dividend : ℝ := 14698
def quotient : ℝ := 89
def divisor : ℝ := 164.98876404494382
def remainder : ℝ := dividend - (quotient * divisor)

theorem remainder_approximately_14 : abs (remainder - 14) < 1e-10 := 
by
-- using abs since the problem is numerical/approximate
sorry

end remainder_approximately_14_l587_58703


namespace steve_bought_3_boxes_of_cookies_l587_58727

variable (total_cost : ℝ)
variable (milk_cost : ℝ)
variable (cereal_cost : ℝ)
variable (banana_cost : ℝ)
variable (apple_cost : ℝ)
variable (chicken_cost : ℝ)
variable (peanut_butter_cost : ℝ)
variable (bread_cost : ℝ)
variable (cookie_box_cost : ℝ)
variable (cookie_box_count : ℝ)

noncomputable def proves_steve_cookie_boxes : Prop :=
  total_cost = 50 ∧
  milk_cost = 4 ∧
  cereal_cost = 3 ∧
  banana_cost = 0.2 ∧
  apple_cost = 0.75 ∧
  chicken_cost = 10 ∧
  peanut_butter_cost = 5 ∧
  bread_cost = (2 * cereal_cost) / 2 ∧
  cookie_box_cost = (milk_cost + peanut_butter_cost) / 3 ∧
  cookie_box_count = (total_cost - (milk_cost + 3 * cereal_cost + 6 * banana_cost + 8 * apple_cost + chicken_cost + peanut_butter_cost + bread_cost)) / cookie_box_cost

theorem steve_bought_3_boxes_of_cookies :
  proves_steve_cookie_boxes 50 4 3 0.2 0.75 10 5 3 ((4 + 5) / 3) 3 :=
by
  sorry

end steve_bought_3_boxes_of_cookies_l587_58727


namespace solution_set_a1_range_of_a_l587_58709

def f (x a : ℝ) : ℝ := abs (x - a) * abs (x + abs (x - 2)) * abs (x - a)

theorem solution_set_a1 (x : ℝ) : f x 1 < 0 ↔ x < 1 :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, x < 1 → f x a < 0) ↔ 1 ≤ a :=
by
  sorry

end solution_set_a1_range_of_a_l587_58709


namespace num_special_fractions_eq_one_l587_58736

-- Definitions of relatively prime and positive
def are_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def is_positive (n : ℕ) : Prop := n > 0

-- Statement to prove the number of such fractions
theorem num_special_fractions_eq_one : 
  (∀ (x y : ℕ), is_positive x → is_positive y → are_rel_prime x y → 
    (x + 1) * 10 * y = (y + 1) * 11 * x →
    ((x = 5 ∧ y = 11) ∨ False)) := sorry

end num_special_fractions_eq_one_l587_58736


namespace smallest_option_l587_58754

-- Define the problem with the given condition
def x : ℕ := 10

-- Define all the options in the problem
def option_a := 6 / x
def option_b := 6 / (x + 1)
def option_c := 6 / (x - 1)
def option_d := x / 6
def option_e := (x + 1) / 6
def option_f := (x - 2) / 6

-- The proof problem statement to show that option_b is the smallest
theorem smallest_option :
  option_b < option_a ∧ option_b < option_c ∧ option_b < option_d ∧ option_b < option_e ∧ option_b < option_f :=
by
  sorry

end smallest_option_l587_58754


namespace discount_percentage_l587_58768

theorem discount_percentage (p : ℝ) : 
  (1 + 0.25) * p * (1 - 0.20) = p :=
by
  sorry

end discount_percentage_l587_58768


namespace smallest_positive_integer_l587_58728

def smallest_x (x : ℕ) : Prop :=
  (540 * x) % 800 = 0

theorem smallest_positive_integer (x : ℕ) : smallest_x x → x = 80 :=
by {
  sorry
}

end smallest_positive_integer_l587_58728


namespace sufficient_but_not_necessary_l587_58700

theorem sufficient_but_not_necessary (a b c : ℝ) :
  (b^2 = a * c → (c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) ∨ (b = 0)) ∧ 
  ¬ ((c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) → b^2 = a * c) :=
by
  sorry

end sufficient_but_not_necessary_l587_58700


namespace evaluate_neg_64_exp_4_over_3_l587_58797

theorem evaluate_neg_64_exp_4_over_3 : (-64 : ℝ) ^ (4 / 3) = 256 := 
by
  sorry

end evaluate_neg_64_exp_4_over_3_l587_58797


namespace average_remaining_two_numbers_l587_58740

theorem average_remaining_two_numbers 
  (h1 : (40.5 : ℝ) = 10 * 4.05)
  (h2 : (11.1 : ℝ) = 3 * 3.7)
  (h3 : (11.85 : ℝ) = 3 * 3.95)
  (h4 : (8.6 : ℝ) = 2 * 4.3)
  : (4.475 : ℝ) = (40.5 - (11.1 + 11.85 + 8.6)) / 2 := 
sorry

end average_remaining_two_numbers_l587_58740


namespace sales_quota_50_l587_58750

theorem sales_quota_50 :
  let cars_sold_first_three_days := 5 * 3
  let cars_sold_next_four_days := 3 * 4
  let additional_cars_needed := 23
  let total_quota := cars_sold_first_three_days + cars_sold_next_four_days + additional_cars_needed
  total_quota = 50 :=
by
  -- proof goes here
  sorry

end sales_quota_50_l587_58750


namespace seashells_remaining_l587_58794

def initial_seashells : ℕ := 35
def given_seashells : ℕ := 18

theorem seashells_remaining : initial_seashells - given_seashells = 17 := by
  sorry

end seashells_remaining_l587_58794


namespace price_per_jin_of_tomatoes_is_3yuan_3jiao_l587_58707

/-- Definitions of the conditions --/
def cucumbers_cost_jin : ℕ := 5
def cucumbers_cost_yuan : ℕ := 11
def cucumbers_cost_jiao : ℕ := 8
def tomatoes_cost_jin : ℕ := 4
def difference_cost_yuan : ℕ := 1
def difference_cost_jiao : ℕ := 4

/-- Converting cost in yuan and jiao to decimal yuan --/
def cost_in_yuan (yuan jiao : ℕ) : ℕ := yuan + jiao / 10

/-- Given conditions in decimal --/
def cucumbers_cost := cost_in_yuan cucumbers_cost_yuan cucumbers_cost_jiao
def difference_cost := cost_in_yuan difference_cost_yuan difference_cost_jiao
def tomatoes_cost := cucumbers_cost + difference_cost

/-- Proof statement: price per jin of tomatoes in yuan and jiao --/
theorem price_per_jin_of_tomatoes_is_3yuan_3jiao :
  tomatoes_cost / tomatoes_cost_jin = 3 + 3 / 10 :=
by
  sorry

end price_per_jin_of_tomatoes_is_3yuan_3jiao_l587_58707


namespace number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l587_58706

-- Define the statement about the total number of 5-letter words.
theorem number_of_5_letter_words : 26^5 = 26^5 := by
  sorry

-- Define the statement about the total number of 5-letter words with all different letters.
theorem number_of_5_letter_words_with_all_different_letters : 
  26 * 25 * 24 * 23 * 22 = 26 * 25 * 24 * 23 * 22 := by
  sorry

-- Define the statement about the total number of 5-letter words with no consecutive letters being the same.
theorem number_of_5_letter_words_with_no_consecutive_repeating_letters : 
  26 * 25 * 25 * 25 * 25 = 26 * 25 * 25 * 25 * 25 := by
  sorry

end number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l587_58706


namespace farmer_total_acres_l587_58772

theorem farmer_total_acres (x : ℕ) (H1 : 4 * x = 376) : 
  5 * x + 2 * x + 4 * x = 1034 :=
by
  -- This placeholder is indicating unfinished proof
  sorry

end farmer_total_acres_l587_58772


namespace annual_interest_rate_l587_58787

theorem annual_interest_rate (P A : ℝ) (n t : ℕ) (r : ℝ) 
  (hP : P = 700) 
  (hA : A = 771.75) 
  (hn : n = 2) 
  (ht : t = 1) 
  (h : A = P * (1 + r / n) ^ (n * t)) : 
  r = 0.10 := 
by 
  -- Proof steps go here
  sorry

end annual_interest_rate_l587_58787


namespace smallest_sum_of_squares_l587_58746

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 187) : x^2 + y^2 ≥ 205 := sorry

end smallest_sum_of_squares_l587_58746


namespace rectangle_measurement_error_l587_58726

theorem rectangle_measurement_error
  (L W : ℝ)
  (x : ℝ)
  (h1 : ∀ x, L' = L * (1 + x / 100))
  (h2 : W' = W * 0.9)
  (h3 : A = L * W)
  (h4 : A' = A * 1.08) :
  x = 20 :=
by
  sorry

end rectangle_measurement_error_l587_58726


namespace even_product_when_eight_cards_drawn_l587_58724

theorem even_product_when_eight_cards_drawn :
  ∀ (s : Finset ℕ), (∀ n ∈ s, n ∈ Finset.range 15) →
  s.card ≥ 8 →
  (∃ m ∈ s, Even m) :=
by
  sorry

end even_product_when_eight_cards_drawn_l587_58724


namespace average_temperature_for_july_4th_l587_58735

def avg_temperature_july_4th : ℤ := 
  let temperatures := [90, 90, 90, 79, 71]
  let sum := List.sum temperatures
  sum / temperatures.length

theorem average_temperature_for_july_4th :
  avg_temperature_july_4th = 84 := 
by
  sorry

end average_temperature_for_july_4th_l587_58735


namespace solve_for_y_l587_58752

theorem solve_for_y (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
  sorry

end solve_for_y_l587_58752


namespace calc_305_squared_minus_295_squared_l587_58739

theorem calc_305_squared_minus_295_squared :
  305^2 - 295^2 = 6000 := 
  by
    sorry

end calc_305_squared_minus_295_squared_l587_58739


namespace at_least_one_no_less_than_two_l587_58732

variable (a b c : ℝ)
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)

theorem at_least_one_no_less_than_two :
  ∃ x ∈ ({a + 1/b, b + 1/c, c + 1/a} : Set ℝ), 2 ≤ x := by
  sorry

end at_least_one_no_less_than_two_l587_58732


namespace arithmetic_sequence_product_l587_58777

theorem arithmetic_sequence_product (a : ℕ → ℤ) (d : ℤ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_prod : a 4 * a 5 = 12) : a 2 * a 7 = 6 :=
sorry

end arithmetic_sequence_product_l587_58777


namespace girls_to_boys_ratio_l587_58779

variable (g b : ℕ)
variable (h_total : g + b = 36)
variable (h_diff : g = b + 6)

theorem girls_to_boys_ratio (g b : ℕ) (h_total : g + b = 36) (h_diff : g = b + 6) :
  g / b = 7 / 5 := by
  sorry

end girls_to_boys_ratio_l587_58779


namespace plane_speed_west_l587_58781

theorem plane_speed_west (v t : ℝ) : 
  (300 * t + 300 * t = 1200) ∧ (t = 7 - t) → 
  (v = 300 * t / (7 - t)) ∧ (t = 2) → 
  v = 120 :=
by
  intros h1 h2
  sorry

end plane_speed_west_l587_58781


namespace quadratic_real_roots_l587_58720

theorem quadratic_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 3 * x - 1 = 0) ↔ k ≥ -9 / 4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_real_roots_l587_58720


namespace sum_of_prime_factors_l587_58782

theorem sum_of_prime_factors (x : ℕ) (h1 : x = 2^10 - 1) 
  (h2 : 2^10 - 1 = (2^5 + 1) * (2^5 - 1)) 
  (h3 : 2^5 - 1 = 31) 
  (h4 : 2^5 + 1 = 33) 
  (h5 : 33 = 3 * 11) : 
  (31 + 3 + 11 = 45) := 
  sorry

end sum_of_prime_factors_l587_58782


namespace total_books_is_10033_l587_58701

variable (P C B M H : ℕ)
variable (x : ℕ) (h_P : P = 3 * x) (h_C : C = 2 * x)
variable (h_B : B = (3 / 2) * x)
variable (h_M : M = (3 / 5) * x)
variable (h_H : H = (4 / 5) * x)
variable (total_books : ℕ)
variable (h_total : total_books = P + C + B + M + H)
variable (h_bound : total_books > 10000)

theorem total_books_is_10033 : total_books = 10033 :=
  sorry

end total_books_is_10033_l587_58701


namespace john_new_cards_l587_58715

def cards_per_page : ℕ := 3
def old_cards : ℕ := 16
def pages_used : ℕ := 8

theorem john_new_cards : (pages_used * cards_per_page) - old_cards = 8 := by
  sorry

end john_new_cards_l587_58715


namespace abdul_largest_number_l587_58791

theorem abdul_largest_number {a b c d : ℕ} 
  (h1 : a + (b + c + d) / 3 = 17)
  (h2 : b + (a + c + d) / 3 = 21)
  (h3 : c + (a + b + d) / 3 = 23)
  (h4 : d + (a + b + c) / 3 = 29) :
  d = 21 :=
by sorry

end abdul_largest_number_l587_58791


namespace only_set_d_forms_triangle_l587_58769

/-- Definition of forming a triangle given three lengths -/
def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem only_set_d_forms_triangle :
  ¬ can_form_triangle 3 5 10 ∧ ¬ can_form_triangle 5 4 9 ∧ 
  ¬ can_form_triangle 5 5 10 ∧ can_form_triangle 4 6 9 :=
by {
  sorry
}

end only_set_d_forms_triangle_l587_58769


namespace profit_percentage_l587_58712

theorem profit_percentage (SP CP : ℝ) (h_SP : SP = 150) (h_CP : CP = 120) : 
  ((SP - CP) / CP) * 100 = 25 :=
by {
  sorry
}

end profit_percentage_l587_58712


namespace domain_of_f_l587_58773

noncomputable def f (x : ℝ) : ℝ :=
  (x - 4)^0 + Real.sqrt (2 / (x - 1))

theorem domain_of_f :
  ∀ x : ℝ, (1 < x ∧ x < 4) ∨ (4 < x) ↔
    ∃ y : ℝ, f y = f x :=
sorry

end domain_of_f_l587_58773


namespace part_A_part_B_part_D_l587_58704

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Part A: single transmission probability
theorem part_A (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 :=
by sorry

-- Part B: triple transmission probability
theorem part_B (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  β * (1 - β)^2 = β * (1 - β)^2 :=
by sorry

-- Part D: comparing single and triple transmission
theorem part_D (α β : ℝ) (hα : 0 < α ∧ α < 0.5) (hβ : 0 < β ∧ β < 1) :
  (1 - α) < (1 - α)^3 + 3 * α * (1 - α)^2 :=
by sorry

end part_A_part_B_part_D_l587_58704


namespace proof_C_I_M_cap_N_l587_58786

open Set

variable {𝕜 : Type _} [LinearOrderedField 𝕜]

def I : Set 𝕜 := Set.univ
def M : Set 𝕜 := {x : 𝕜 | -2 ≤ x ∧ x ≤ 2}
def N : Set 𝕜 := {x : 𝕜 | x < 1}
def C_I (A : Set 𝕜) : Set 𝕜 := I \ A

theorem proof_C_I_M_cap_N :
  C_I M ∩ N = {x : 𝕜 | x < -2} := by
  sorry

end proof_C_I_M_cap_N_l587_58786


namespace calculate_A_l587_58710

theorem calculate_A (D B E C A : ℝ) :
  D = 2 * 4 →
  B = 2 * D →
  E = 7 * 2 →
  C = 7 * E →
  A^2 = B * C →
  A = 28 * Real.sqrt 2 :=
by
  sorry

end calculate_A_l587_58710


namespace equation_solution_l587_58711

theorem equation_solution (x : ℝ) : 
  (x - 3)^4 = 16 → x = 5 :=
by
  sorry

end equation_solution_l587_58711


namespace tax_deduction_is_correct_l587_58744

-- Define the hourly wage and tax rate
def hourly_wage_dollars : ℝ := 25
def tax_rate : ℝ := 0.021

-- Define the conversion from dollars to cents
def dollars_to_cents (dollars : ℝ) : ℝ := dollars * 100

-- Calculate the hourly wage in cents
def hourly_wage_cents : ℝ := dollars_to_cents hourly_wage_dollars

-- Calculate the tax deducted in cents per hour
def tax_deduction_cents (wage : ℝ) (rate : ℝ) : ℝ := rate * wage

-- State the theorem that needs to be proven
theorem tax_deduction_is_correct :
  tax_deduction_cents hourly_wage_cents tax_rate = 52.5 :=
by
  sorry

end tax_deduction_is_correct_l587_58744


namespace yellow_beads_needed_l587_58757

variable (Total green yellow : ℕ)

theorem yellow_beads_needed (h_green : green = 4) (h_yellow : yellow = 0) (h_fraction : (4 / 5 : ℚ) = 4 / (green + yellow + 16)) :
    4 + 16 + green = Total := by
  sorry

end yellow_beads_needed_l587_58757


namespace hyperbola_eccentricity_l587_58760

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (e : ℝ) (h3 : e = (Real.sqrt 3) / 2) 
  (h4 : a ^ 2 = b ^ 2 + (Real.sqrt 3) ^ 2) : (Real.sqrt 5) / 2 = 
    (Real.sqrt (a ^ 2 + b ^ 2)) / a :=
by
  sorry

end hyperbola_eccentricity_l587_58760


namespace sum_fractions_l587_58730

theorem sum_fractions :
  (1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)) = (2 / 9) :=
by
  sorry

end sum_fractions_l587_58730


namespace triangle_area_is_64_l587_58761

noncomputable def area_triangle_lines (y : ℝ → ℝ) (x : ℝ → ℝ) (neg_x : ℝ → ℝ) : ℝ :=
  let A := (8, 8)
  let B := (-8, 8)
  let O := (0, 0)
  let base := 16
  let height := 8
  1 / 2 * base * height

theorem triangle_area_is_64 :
  let y := fun x => 8
  let x := fun y => y
  let neg_x := fun y => -y
  area_triangle_lines y x neg_x = 64 := 
by 
  sorry

end triangle_area_is_64_l587_58761


namespace jake_final_bitcoins_l587_58729

def initial_bitcoins : ℕ := 120
def investment_bitcoins : ℕ := 40
def returned_investment : ℕ := investment_bitcoins * 2
def bitcoins_after_investment : ℕ := initial_bitcoins - investment_bitcoins + returned_investment
def first_charity_donation : ℕ := 25
def bitcoins_after_first_donation : ℕ := bitcoins_after_investment - first_charity_donation
def brother_share : ℕ := 67
def bitcoins_after_giving_to_brother : ℕ := bitcoins_after_first_donation - brother_share
def debt_payment : ℕ := 5
def bitcoins_after_taking_back : ℕ := bitcoins_after_giving_to_brother + debt_payment
def quadrupled_bitcoins : ℕ := bitcoins_after_taking_back * 4
def second_charity_donation : ℕ := 15
def final_bitcoins : ℕ := quadrupled_bitcoins - second_charity_donation

theorem jake_final_bitcoins : final_bitcoins = 277 := by
  unfold final_bitcoins
  unfold quadrupled_bitcoins
  unfold bitcoins_after_taking_back
  unfold debt_payment
  unfold bitcoins_after_giving_to_brother
  unfold brother_share
  unfold bitcoins_after_first_donation
  unfold first_charity_donation
  unfold bitcoins_after_investment
  unfold returned_investment
  unfold investment_bitcoins
  unfold initial_bitcoins
  sorry

end jake_final_bitcoins_l587_58729


namespace altitudes_bounded_by_perimeter_l587_58733

theorem altitudes_bounded_by_perimeter (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 2) :
  ¬ (∀ (ha hb hc : ℝ), ha = 2 / a * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hb = 2 / b * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     hc = 2 / c * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ∧ 
                     ha > 1 / Real.sqrt 3 ∧ 
                     hb > 1 / Real.sqrt 3 ∧ 
                     hc > 1 / Real.sqrt 3 ) :=
sorry

end altitudes_bounded_by_perimeter_l587_58733


namespace garden_perimeter_l587_58737

-- formally defining the conditions of the problem
variables (x y : ℝ)
def diagonal_of_garden : Prop := x^2 + y^2 = 900
def area_of_garden : Prop := x * y = 216

-- final statement to prove the perimeter of the garden
theorem garden_perimeter (h1 : diagonal_of_garden x y) (h2 : area_of_garden x y) : 2 * (x + y) = 73 := sorry

end garden_perimeter_l587_58737


namespace problem_l587_58756

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

noncomputable def f_deriv (x : ℝ) : ℝ := - (1 / x^2) * Real.cos x - (1 / x) * Real.sin x

theorem problem (h_pi_ne_zero : Real.pi ≠ 0) (h_pi_div_two_ne_zero : Real.pi / 2 ≠ 0) :
  f Real.pi + f_deriv (Real.pi / 2) = -3 / Real.pi  := by
  sorry

end problem_l587_58756


namespace translated_graph_pass_through_origin_l587_58780

theorem translated_graph_pass_through_origin 
    (φ : ℝ) (h : 0 < φ ∧ φ < π / 2) 
    (passes_through_origin : 0 = Real.sin (-2 * φ + π / 3)) : 
    φ = π / 6 := 
sorry

end translated_graph_pass_through_origin_l587_58780


namespace center_of_symmetry_l587_58723

def symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

/--
  Given the function f(x) := sin x - sqrt(3) * cos x,
  prove that (π/3, 0) is the center of symmetry for f.
-/
theorem center_of_symmetry : symmetry_center (fun x => Real.sin x - Real.sqrt 3 * Real.cos x) (Real.pi / 3, 0) :=
by
  sorry

end center_of_symmetry_l587_58723


namespace mixed_doubles_pairing_l587_58747

def num_ways_to_pair (men women : ℕ) (select_men select_women : ℕ) : ℕ :=
  (Nat.choose men select_men) * (Nat.choose women select_women) * 2

theorem mixed_doubles_pairing : num_ways_to_pair 5 4 2 2 = 120 := by
  sorry

end mixed_doubles_pairing_l587_58747


namespace length_of_platform_l587_58774

theorem length_of_platform {train_length platform_crossing_time signal_pole_crossing_time : ℚ}
  (h_train_length : train_length = 300)
  (h_platform_crossing_time : platform_crossing_time = 40)
  (h_signal_pole_crossing_time : signal_pole_crossing_time = 18) :
  ∃ L : ℚ, L = 1100 / 3 :=
by
  sorry

end length_of_platform_l587_58774


namespace students_going_to_tournament_l587_58749

-- Defining the conditions
def total_students : ℕ := 24
def fraction_in_chess_program : ℚ := 1 / 3
def fraction_going_to_tournament : ℚ := 1 / 2

-- The final goal to prove
theorem students_going_to_tournament : 
  (total_students • fraction_in_chess_program) • fraction_going_to_tournament = 4 := 
by
  sorry

end students_going_to_tournament_l587_58749


namespace raptors_points_l587_58738

theorem raptors_points (x y z : ℕ) (h1 : x + y + z = 48) (h2 : x - y = 18) :
  (z = 0 → y = 15) ∧
  (z = 12 → y = 9) ∧
  (z = 18 → y = 6) ∧
  (z = 30 → y = 0) :=
by sorry

end raptors_points_l587_58738


namespace find_asterisk_value_l587_58755

theorem find_asterisk_value :
  ∃ x : ℤ, (x / 21) * (42 / 84) = 1 ↔ x = 21 :=
by
  sorry

end find_asterisk_value_l587_58755


namespace exists_disjoint_subsets_for_prime_products_l587_58790

theorem exists_disjoint_subsets_for_prime_products :
  ∃ (A : Fin 100 → Set ℕ), (∀ i j, i ≠ j → Disjoint (A i) (A j)) ∧
    (∀ S : Set ℕ, Infinite S → (∃ m : ℕ, ∃ (a : Fin 100 → ℕ),
      (∀ i, a i ∈ A i) ∧ (∀ i, ∃ p : Fin m → ℕ, (∀ k, p k ∈ S) ∧ a i = (List.prod (List.ofFn p))))) :=
sorry

end exists_disjoint_subsets_for_prime_products_l587_58790


namespace simplify_expression_l587_58762

noncomputable def q (x a b c d : ℝ) :=
  (x + a)^4 / ((a - b) * (a - c) * (a - d))
  + (x + b)^4 / ((b - a) * (b - c) * (b - d))
  + (x + c)^4 / ((c - a) * (c - b) * (c - d))
  + (x + d)^4 / ((d - a) * (d - b) * (d - c))

theorem simplify_expression (a b c d x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  q x a b c d = a + b + c + d + 4 * x :=
by
  sorry

end simplify_expression_l587_58762


namespace find_k_l587_58717

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem find_k (k : ℝ) (h_pos : 0 < k) (h_exists : ∃ x₀ : ℝ, 1 ≤ x₀ ∧ g x₀ ≤ k * (-x₀^2 + 3 * x₀)) : 
  k > (1 / 2) * (Real.exp 1 + 1 / Real.exp 1) :=
sorry

end find_k_l587_58717


namespace prime_cube_difference_l587_58767

theorem prime_cube_difference (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) (eqn : p^3 - q^3 = 11 * r) : 
  (p = 13 ∧ q = 2 ∧ r = 199) :=
sorry

end prime_cube_difference_l587_58767


namespace value_range_of_f_in_interval_l587_58796

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

theorem value_range_of_f_in_interval : 
  ∀ x, (2 ≤ x ∧ x ≤ 4) → (1/2 ≤ f x ∧ f x ≤ 2/3) := 
by
  sorry

end value_range_of_f_in_interval_l587_58796


namespace triangle_side_calculation_l587_58714

theorem triangle_side_calculation
  (a : ℝ) (A B : ℝ)
  (ha : a = 3)
  (hA : A = 30)
  (hB : B = 15) :
  let C := 180 - A - B
  let c := a * (Real.sin C) / (Real.sin A)
  c = 3 * Real.sqrt 2 := by
  sorry

end triangle_side_calculation_l587_58714


namespace cost_of_gas_l587_58789

def hoursDriven1 : ℕ := 2
def speed1 : ℕ := 60
def hoursDriven2 : ℕ := 3
def speed2 : ℕ := 50
def milesPerGallon : ℕ := 30
def costPerGallon : ℕ := 2

def totalDistance : ℕ := (hoursDriven1 * speed1) + (hoursDriven2 * speed2)
def gallonsUsed : ℕ := totalDistance / milesPerGallon
def totalCost : ℕ := gallonsUsed * costPerGallon

theorem cost_of_gas : totalCost = 18 := by
  -- You should fill in the proof steps here.
  sorry

end cost_of_gas_l587_58789


namespace allie_betty_total_points_product_l587_58705

def score (n : Nat) : Nat :=
  if n % 3 == 0 then 9
  else if n % 2 == 0 then 3
  else if n % 2 == 1 then 1
  else 0

def allie_points : List Nat := [5, 2, 6, 1, 3]
def betty_points : List Nat := [6, 4, 1, 2, 5]

def total_points (rolls: List Nat) : Nat :=
  rolls.foldl (λ acc n => acc + score n) 0

theorem allie_betty_total_points_product : 
  total_points allie_points * total_points betty_points = 391 := by
  sorry

end allie_betty_total_points_product_l587_58705


namespace ratio_of_investments_l587_58785

-- Define the conditions
def ratio_of_profits (p q : ℝ) : Prop := 7/12 = (p * 5) / (q * 12)

-- Define the problem: given the conditions, prove the ratio of investments is 7/5
theorem ratio_of_investments (P Q : ℝ) (h : ratio_of_profits P Q) : P / Q = 7 / 5 :=
by
  sorry

end ratio_of_investments_l587_58785


namespace initial_number_of_mice_l587_58731

theorem initial_number_of_mice (x : ℕ) 
  (h1 : x % 2 = 0)
  (h2 : (x / 2) % 3 = 0)
  (h3 : (x / 2 - x / 6) % 4 = 0)
  (h4 : (x / 2 - x / 6 - (x / 2 - x / 6) / 4) % 5 = 0)
  (h5 : (x / 5) = (x / 6) + 2) : 
  x = 60 := 
by sorry

end initial_number_of_mice_l587_58731


namespace distinct_cube_arrangements_count_l587_58758

def is_valid_face_sum (face : Finset ℕ) : Prop :=
  face.sum id = 34

def is_valid_opposite_sum (v1 v2 : ℕ) : Prop :=
  v1 + v2 = 16

def is_unique_up_to_rotation (cubes : List (Finset ℕ)) : Prop := sorry -- Define rotational uniqueness check

noncomputable def count_valid_arrangements : ℕ := sorry -- Define counting logic

theorem distinct_cube_arrangements_count : count_valid_arrangements = 3 :=
  sorry

end distinct_cube_arrangements_count_l587_58758


namespace find_sum_l587_58770

variable {f : ℝ → ℝ}

-- Conditions of the problem
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def condition_2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
def condition_3 (f : ℝ → ℝ) : Prop := f 1 = 9

theorem find_sum (h_odd : odd_function f) (h_cond2 : condition_2 f) (h_cond3 : condition_3 f) :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_sum_l587_58770


namespace sofia_running_time_l587_58722

theorem sofia_running_time :
  ∃ t : ℤ, t = 8 * 60 + 20 ∧ 
  (∀ (laps : ℕ) (d1 d2 v1 v2 : ℤ),
    laps = 5 →
    d1 = 200 →
    v1 = 4 →
    d2 = 300 →
    v2 = 6 →
    t = laps * ((d1 / v1 + d2 / v2))) :=
by
  sorry

end sofia_running_time_l587_58722


namespace perpendicular_lines_iff_a_eq_1_l587_58764

theorem perpendicular_lines_iff_a_eq_1 :
  ∀ a : ℝ, (∀ x y, (y = a * x + 1) → (y = (a - 2) * x - 1) → (a = 1)) ↔ (a = 1) :=
by sorry

end perpendicular_lines_iff_a_eq_1_l587_58764


namespace instrument_price_problem_l587_58725

theorem instrument_price_problem (v t p : ℝ) (h1 : 1.5 * v = 0.5 * t + 50) (h2 : 1.5 * t = 0.5 * p + 50) : 
  ∃ m n : ℤ, m = 80 ∧ n = 80 ∧ (100 + m) * v / 100 = n + (100 - m) * p / 100 := 
by
  use 80, 80
  sorry

end instrument_price_problem_l587_58725


namespace modulus_of_complex_l587_58719

open Complex

theorem modulus_of_complex (z : ℂ) (h : (1 + z) / (1 - z) = ⟨0, 1⟩) : abs z = 1 := 
sorry

end modulus_of_complex_l587_58719


namespace skittles_transfer_l587_58771

-- Define the initial number of Skittles Bridget and Henry have
def bridget_initial_skittles := 4
def henry_initial_skittles := 4

-- The main statement we want to prove
theorem skittles_transfer :
  bridget_initial_skittles + henry_initial_skittles = 8 :=
by
  sorry

end skittles_transfer_l587_58771


namespace find_digit_B_l587_58795

theorem find_digit_B (B : ℕ) (h1 : B < 10) : 3 ∣ (5 + 2 + B + 6) → B = 2 :=
by
  sorry

end find_digit_B_l587_58795


namespace arithmetic_geometric_sequence_general_term_l587_58765

theorem arithmetic_geometric_sequence_general_term :
  ∃ q a1 : ℕ, (∀ n : ℕ, a2 = 6 ∧ 6 * a1 + a3 = 30) →
  (∀ n : ℕ, (q = 2 ∧ a1 = 3 → a_n = 3 * 3^(n-1)) ∨ (q = 3 ∧ a1 = 2 → a_n = 2 * 2^(n-1))) :=
by
  sorry

end arithmetic_geometric_sequence_general_term_l587_58765


namespace quadrant_of_angle_l587_58799

theorem quadrant_of_angle (θ : ℝ) (h1 : Real.cos θ = -3 / 5) (h2 : Real.tan θ = 4 / 3) :
    θ ∈ Set.Icc (π : ℝ) (3 * π / 2) := sorry

end quadrant_of_angle_l587_58799


namespace smallest_K_exists_l587_58745

theorem smallest_K_exists (S : Finset ℕ) (h_S : S = (Finset.range 51).erase 0) :
  ∃ K, ∀ (T : Finset ℕ), T ⊆ S ∧ T.card = K → 
  ∃ a b, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b) ∧ K = 39 :=
by
  use 39
  sorry

end smallest_K_exists_l587_58745


namespace spherical_coordinates_equivalence_l587_58792

theorem spherical_coordinates_equivalence :
  ∀ (ρ θ φ : ℝ), 
        ρ = 3 → θ = (2 * Real.pi / 7) → φ = (8 * Real.pi / 5) →
        (0 < ρ) → 
        (0 ≤ (2 * Real.pi / 7) ∧ (2 * Real.pi / 7) < 2 * Real.pi) →
        (0 ≤ (8 * Real.pi / 5) ∧ (8 * Real.pi / 5) ≤ Real.pi) →
      ∃ (ρ' θ' φ' : ℝ), 
        ρ' = ρ ∧ θ' = (9 * Real.pi / 7) ∧ φ' = (2 * Real.pi / 5) :=
by
    sorry

end spherical_coordinates_equivalence_l587_58792


namespace sum_of_squares_eq_frac_squared_l587_58702

theorem sum_of_squares_eq_frac_squared (x y z a b c : ℝ) (hxya : x * y = a) (hxzb : x * z = b) (hyzc : y * z = c)
  (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) (ha0 : a ≠ 0) (hb0 : b ≠ 0) (hc0 : c ≠ 0) :
  x^2 + y^2 + z^2 = ((a * b)^2 + (a * c)^2 + (b * c)^2) / (a * b * c) :=
by
  sorry

end sum_of_squares_eq_frac_squared_l587_58702


namespace rectangular_prism_has_8_vertices_l587_58721

def rectangular_prism_vertices := 8

theorem rectangular_prism_has_8_vertices : rectangular_prism_vertices = 8 := by
  sorry

end rectangular_prism_has_8_vertices_l587_58721


namespace num_perfect_square_factors_1800_l587_58763

theorem num_perfect_square_factors_1800 :
  let factors_1800 := [(2, 3), (3, 2), (5, 2)]
  ∃ n : ℕ, (n = 8) ∧
           (∀ p_k ∈ factors_1800, ∃ (e : ℕ), (e = 0 ∨ e = 2) ∧ n = 2 * 2 * 2 → n = 8) :=
sorry

end num_perfect_square_factors_1800_l587_58763


namespace polynomial_non_negative_for_all_real_iff_l587_58742

theorem polynomial_non_negative_for_all_real_iff (a : ℝ) :
  (∀ x : ℝ, x^4 + (a - 1) * x^2 + 1 ≥ 0) ↔ a ≥ -1 :=
by sorry

end polynomial_non_negative_for_all_real_iff_l587_58742


namespace LCM_180_504_l587_58753

theorem LCM_180_504 : Nat.lcm 180 504 = 2520 := 
by 
  -- We skip the proof.
  sorry

end LCM_180_504_l587_58753


namespace processing_time_l587_58713

theorem processing_time 
  (pictures : ℕ) (minutes_per_picture : ℕ) (minutes_per_hour : ℕ)
  (h1 : pictures = 960) (h2 : minutes_per_picture = 2) (h3 : minutes_per_hour = 60) : 
  (pictures * minutes_per_picture) / minutes_per_hour = 32 :=
by 
  sorry

end processing_time_l587_58713


namespace geometric_series_sixth_term_l587_58775

theorem geometric_series_sixth_term :
  ∃ r : ℝ, r > 0 ∧ (16 * r^7 = 11664) ∧ (16 * r^5 = 3888) :=
by 
  sorry

end geometric_series_sixth_term_l587_58775


namespace julia_drove_214_miles_l587_58718

def daily_rate : ℝ := 29
def cost_per_mile : ℝ := 0.08
def total_cost : ℝ := 46.12

theorem julia_drove_214_miles :
  (total_cost - daily_rate) / cost_per_mile = 214 :=
by
  sorry

end julia_drove_214_miles_l587_58718


namespace line_circle_intersection_l587_58776

theorem line_circle_intersection (a : ℝ) : 
  (∀ x y : ℝ, (4 * x + 3 * y + a = 0) → ((x - 1)^2 + (y - 2)^2 = 9)) ∧
  (∃ A B : ℝ, dist A B = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by 
  sorry

end line_circle_intersection_l587_58776


namespace competition_total_races_l587_58778

theorem competition_total_races (sprinters : ℕ) (sprinters_with_bye : ℕ) (lanes_preliminary : ℕ) (lanes_subsequent : ℕ) 
  (eliminated_per_race : ℕ) (first_round_advance : ℕ) (second_round_advance : ℕ) (third_round_advance : ℕ) 
  : sprinters = 300 → sprinters_with_bye = 16 → lanes_preliminary = 8 → lanes_subsequent = 6 → 
    eliminated_per_race = 7 → first_round_advance = 36 → second_round_advance = 9 → third_round_advance = 2 
    → first_round_races = 36 → second_round_races = 9 → third_round_races = 2 → final_race = 1
    → first_round_races + second_round_races + third_round_races + final_race = 48 :=
by 
  intros sprinters_eq sprinters_with_bye_eq lanes_preliminary_eq lanes_subsequent_eq eliminated_per_race_eq 
         first_round_advance_eq second_round_advance_eq third_round_advance_eq 
         first_round_races_eq second_round_races_eq third_round_races_eq final_race_eq
  sorry

end competition_total_races_l587_58778


namespace diff_is_multiple_of_9_l587_58788

-- Definitions
def orig_num (a b : ℕ) : ℕ := 10 * a + b
def new_num (a b : ℕ) : ℕ := 10 * b + a

-- Statement of the mathematical proof problem
theorem diff_is_multiple_of_9 (a b : ℕ) : 
  9 ∣ (new_num a b - orig_num a b) :=
by
  sorry

end diff_is_multiple_of_9_l587_58788


namespace math_problem_l587_58798

noncomputable def ellipse_standard_equation (a b : ℝ) : Prop :=
  a = 2 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def constant_slope_sum (T R S : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  T = (4, 0) ∧ l (1, 0) ∧ 
  (∀ TR TS : ℝ, (TR = (R.2 / (R.1 - 4)) ∧ TS = (S.2 / (S.1 - 4)) ∧ 
  (TR + TS = 0)))

theorem math_problem 
  {a b : ℝ} {T R S : ℝ × ℝ} {l : ℝ × ℝ → Prop} : 
  ellipse_standard_equation a b ∧ constant_slope_sum T R S l :=
by
  sorry

end math_problem_l587_58798


namespace find_slope_of_line_l_l587_58783

theorem find_slope_of_line_l :
  ∃ k : ℝ, (k = 3 * Real.sqrt 5 / 10 ∨ k = -3 * Real.sqrt 5 / 10) :=
by
  -- Given conditions
  let F1 : ℝ := 6 / 5 * Real.sqrt 5
  let PF : ℝ := 4 / 5 * Real.sqrt 5
  let slope_PQ : ℝ := 1
  let slope_RF1 : ℝ := sorry  -- we need to prove/extract this from the given
  let k := 3 / 2 * slope_RF1
  -- to prove this
  sorry

end find_slope_of_line_l_l587_58783
