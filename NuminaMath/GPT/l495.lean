import Mathlib

namespace NUMINAMATH_GPT_floor_sqrt_120_l495_49543

theorem floor_sqrt_120 : (Int.floor (Real.sqrt 120)) = 10 := by
  have h1 : 10^2 < 120 := by norm_num
  have h2 : 120 < 11^2 := by norm_num
  -- Additional steps to show that Int.floor (Real.sqrt 120) = 10
  sorry

end NUMINAMATH_GPT_floor_sqrt_120_l495_49543


namespace NUMINAMATH_GPT_express_in_scientific_notation_l495_49501

theorem express_in_scientific_notation :
  (2370000 : ℝ) = 2.37 * 10^6 := 
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l495_49501


namespace NUMINAMATH_GPT_tan_315_deg_l495_49519

theorem tan_315_deg : Real.tan (315 * Real.pi / 180) = -1 := sorry

end NUMINAMATH_GPT_tan_315_deg_l495_49519


namespace NUMINAMATH_GPT_B_holds_32_l495_49512

variable (x y z : ℝ)

-- Conditions
def condition1 : Prop := x + 1/2 * (y + z) = 90
def condition2 : Prop := y + 1/2 * (x + z) = 70
def condition3 : Prop := z + 1/2 * (x + y) = 56

-- Theorem to prove
theorem B_holds_32 (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : y = 32 :=
sorry

end NUMINAMATH_GPT_B_holds_32_l495_49512


namespace NUMINAMATH_GPT_adults_at_zoo_l495_49540

theorem adults_at_zoo (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : A = 51 :=
sorry

end NUMINAMATH_GPT_adults_at_zoo_l495_49540


namespace NUMINAMATH_GPT_max_integer_in_form_3_x_3_sub_x_l495_49502

theorem max_integer_in_form_3_x_3_sub_x :
  ∃ x : ℝ, ∀ y : ℝ, y = 3^(x * (3 - x)) → ⌊y⌋ ≤ 11 := 
sorry

end NUMINAMATH_GPT_max_integer_in_form_3_x_3_sub_x_l495_49502


namespace NUMINAMATH_GPT_angle_subtraction_correct_polynomial_simplification_correct_l495_49508

noncomputable def angleSubtraction : Prop :=
  let a1 := 34 * 60 + 26 -- Convert 34°26' to total minutes
  let a2 := 25 * 60 + 33 -- Convert 25°33' to total minutes
  let diff := a1 - a2 -- Subtract in minutes
  let degrees := diff / 60 -- Convert back to degrees
  let minutes := diff % 60 -- Remainder in minutes
  degrees = 8 ∧ minutes = 53 -- Expected result in degrees and minutes

noncomputable def polynomialSimplification (m : Int) : Prop :=
  let expr := 5 * m^2 - (m^2 - 6 * m) - 2 * (-m + 3 * m^2)
  expr = -2 * m^2 + 8 * m -- Simplified form

-- Statements needing proof
theorem angle_subtraction_correct : angleSubtraction := by
  sorry

theorem polynomial_simplification_correct (m : Int) : polynomialSimplification m := by
  sorry

end NUMINAMATH_GPT_angle_subtraction_correct_polynomial_simplification_correct_l495_49508


namespace NUMINAMATH_GPT_expression_meaning_l495_49592

variable (m n : ℤ) -- Assuming m and n are integers for the context.

theorem expression_meaning : 2 * (m - n) = 2 * (m - n) := 
by
  -- It simply follows from the definition of the expression
  sorry

end NUMINAMATH_GPT_expression_meaning_l495_49592


namespace NUMINAMATH_GPT_electronics_weight_l495_49563

theorem electronics_weight (B C E : ℝ) (h1 : B / C = 5 / 4) (h2 : B / E = 5 / 2) (h3 : B / (C - 9) = 10 / 4) : E = 9 := 
by 
  sorry

end NUMINAMATH_GPT_electronics_weight_l495_49563


namespace NUMINAMATH_GPT_egg_hunt_ratio_l495_49577

theorem egg_hunt_ratio :
  ∃ T : ℕ, (3 * T + 30 = 400 ∧ T = 123) ∧ (60 : ℚ) / (T - 20 : ℚ) = 60 / 103 :=
by
  sorry

end NUMINAMATH_GPT_egg_hunt_ratio_l495_49577


namespace NUMINAMATH_GPT_positive_solutions_count_l495_49546

theorem positive_solutions_count :
  ∃ n : ℕ, n = 9 ∧
  (∀ (x y : ℕ), 5 * x + 10 * y = 100 → 0 < x ∧ 0 < y → (∃ k : ℕ, k < 10 ∧ n = 9)) :=
sorry

end NUMINAMATH_GPT_positive_solutions_count_l495_49546


namespace NUMINAMATH_GPT_identity_proof_l495_49537

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a + b)^4 = 2 * (a^2 + a * b + b^2)^2 := 
sorry

end NUMINAMATH_GPT_identity_proof_l495_49537


namespace NUMINAMATH_GPT_product_remainder_l495_49510

-- Define the product of the consecutive numbers
def product := 86 * 87 * 88 * 89 * 90 * 91 * 92

-- Lean statement to state the problem
theorem product_remainder :
  product % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_remainder_l495_49510


namespace NUMINAMATH_GPT_initial_boxes_l495_49587

theorem initial_boxes (x : ℕ) (h1 : 80 + 165 = 245) (h2 : 2000 * 245 = 490000) 
                      (h3 : 4 * 245 * x + 245 * x = 1225 * x) : x = 400 :=
by
  sorry

end NUMINAMATH_GPT_initial_boxes_l495_49587


namespace NUMINAMATH_GPT_no_repetition_five_digit_count_l495_49539

theorem no_repetition_five_digit_count (digits : Finset ℕ) (count : Nat) :
  digits = {0, 1, 2, 3, 4, 5} →
  (∀ n ∈ digits, 0 ≤ n ∧ n ≤ 5) →
  (∃ numbers : Finset ℕ, 
    (∀ x ∈ numbers, (x / 100) % 10 ≠ 3 ∧ x % 5 = 0 ∧ x < 100000 ∧ x ≥ 10000) ∧
    (numbers.card = count)) →
  count = 174 :=
by
  sorry

end NUMINAMATH_GPT_no_repetition_five_digit_count_l495_49539


namespace NUMINAMATH_GPT_always_true_inequality_l495_49549

theorem always_true_inequality (x : ℝ) : x^2 + 1 ≥ 2 * |x| := 
sorry

end NUMINAMATH_GPT_always_true_inequality_l495_49549


namespace NUMINAMATH_GPT_nialls_children_ages_l495_49516

theorem nialls_children_ages : ∃ (a b c d : ℕ), 
  a < 18 ∧ b < 18 ∧ c < 18 ∧ d < 18 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 882 ∧ a + b + c + d = 32 :=
by
  sorry

end NUMINAMATH_GPT_nialls_children_ages_l495_49516


namespace NUMINAMATH_GPT_angle_ne_iff_cos2angle_ne_l495_49550

theorem angle_ne_iff_cos2angle_ne (A B : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) :
  (A ≠ B) ↔ (Real.cos (2 * A) ≠ Real.cos (2 * B)) :=
sorry

end NUMINAMATH_GPT_angle_ne_iff_cos2angle_ne_l495_49550


namespace NUMINAMATH_GPT_comparison_M_N_l495_49507

def M (x : ℝ) : ℝ := x^2 - 3*x + 7
def N (x : ℝ) : ℝ := -x^2 + x + 1

theorem comparison_M_N (x : ℝ) : M x > N x :=
  by sorry

end NUMINAMATH_GPT_comparison_M_N_l495_49507


namespace NUMINAMATH_GPT_bases_for_final_digit_one_l495_49544

noncomputable def numberOfBases : ℕ :=
  (Finset.filter (λ b => ((625 - 1) % b = 0)) (Finset.range 11)).card - 
  (Finset.filter (λ b => b ≤ 2) (Finset.range 11)).card

theorem bases_for_final_digit_one : numberOfBases = 4 :=
by sorry

end NUMINAMATH_GPT_bases_for_final_digit_one_l495_49544


namespace NUMINAMATH_GPT_intersection_M_N_l495_49524

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l495_49524


namespace NUMINAMATH_GPT_set_intersection_l495_49552

def A (x : ℝ) : Prop := -1 ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 3
def B (x : ℝ) : Prop := (x + 1) / x ≤ 0
def C_x_B (x : ℝ) : Prop := x < -1 ∨ x ≥ 0

theorem set_intersection :
  {x : ℝ | A x} ∩ {x : ℝ | C_x_B x} = {x : ℝ | 0 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_set_intersection_l495_49552


namespace NUMINAMATH_GPT_cookies_left_l495_49595

theorem cookies_left (total_cookies : ℕ) (fraction_given : ℚ) (given_cookies : ℕ) (remaining_cookies : ℕ) 
  (h1 : total_cookies = 20)
  (h2 : fraction_given = 2/5)
  (h3 : given_cookies = fraction_given * total_cookies)
  (h4 : remaining_cookies = total_cookies - given_cookies) :
  remaining_cookies = 12 :=
by
  sorry

end NUMINAMATH_GPT_cookies_left_l495_49595


namespace NUMINAMATH_GPT_total_pens_l495_49597

theorem total_pens (r : ℕ) (r_gt_10 : r > 10) (r_div_357 : r ∣ 357) (r_div_441 : r ∣ 441) :
  357 / r + 441 / r = 38 := by
  sorry

end NUMINAMATH_GPT_total_pens_l495_49597


namespace NUMINAMATH_GPT_tom_total_calories_l495_49530

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end NUMINAMATH_GPT_tom_total_calories_l495_49530


namespace NUMINAMATH_GPT_rita_bought_4_pounds_l495_49521

variable (total_amount : ℝ) (cost_per_pound : ℝ) (amount_left : ℝ)

theorem rita_bought_4_pounds (h1 : total_amount = 70)
                             (h2 : cost_per_pound = 8.58)
                             (h3 : amount_left = 35.68) :
  (total_amount - amount_left) / cost_per_pound = 4 := 
  by
  sorry

end NUMINAMATH_GPT_rita_bought_4_pounds_l495_49521


namespace NUMINAMATH_GPT_roots_reciprocal_sum_l495_49566

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) 
    (h_roots : x₁ * x₁ + x₁ - 2 = 0 ∧ x₂ * x₂ + x₂ - 2 = 0):
    x₁ ≠ x₂ → (1 / x₁ + 1 / x₂ = 1 / 2) :=
by
  intro h_neq
  sorry

end NUMINAMATH_GPT_roots_reciprocal_sum_l495_49566


namespace NUMINAMATH_GPT_sphere_cylinder_surface_area_difference_l495_49527

theorem sphere_cylinder_surface_area_difference (R : ℝ) :
  let S_sphere := 4 * Real.pi * R^2
  let S_lateral := 4 * Real.pi * R^2
  S_sphere - S_lateral = 0 :=
by
  sorry

end NUMINAMATH_GPT_sphere_cylinder_surface_area_difference_l495_49527


namespace NUMINAMATH_GPT_range_of_k_l495_49514

theorem range_of_k (k n : ℝ) (h : k ≠ 0) (h_pass : k - n^2 - 2 = k / 2) : k ≥ 4 :=
sorry

end NUMINAMATH_GPT_range_of_k_l495_49514


namespace NUMINAMATH_GPT_exists_composite_expression_l495_49504

-- Define what it means for a number to be composite
def is_composite (m : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = m

-- Main theorem statement
theorem exists_composite_expression :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k > 0 → is_composite (n * 2^k + 1) :=
sorry

end NUMINAMATH_GPT_exists_composite_expression_l495_49504


namespace NUMINAMATH_GPT_field_area_restriction_l495_49586

theorem field_area_restriction (S : ℚ) (b : ℤ) (a : ℚ) (x y : ℚ) 
  (h1 : 10 * 300 * S ≤ 10000)
  (h2 : 2 * a = - b)
  (h3 : abs (6 * y) + 3 ≥ 3)
  (h4 : 2 * abs (2 * x) - abs b ≤ 9)
  (h5 : b ∈ [-4, -3, -2, -1, 0, 1, 2, 3, 4])
: S ≤ 10 / 3 := sorry

end NUMINAMATH_GPT_field_area_restriction_l495_49586


namespace NUMINAMATH_GPT_p_evaluation_l495_49513

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then x + y
  else if x < 0 ∧ y < 0 then x - 3 * y
  else 2 * x + 2 * y

theorem p_evaluation : p (p 3 (-4)) (p (-7) 0) = 40 := by
  sorry

end NUMINAMATH_GPT_p_evaluation_l495_49513


namespace NUMINAMATH_GPT_find_number_l495_49558

theorem find_number (N : ℚ) (h : (5 / 6) * N = (5 / 16) * N + 150) : N = 288 := by
  sorry

end NUMINAMATH_GPT_find_number_l495_49558


namespace NUMINAMATH_GPT_pens_more_than_notebooks_l495_49551

theorem pens_more_than_notebooks
  (N P : ℕ) 
  (h₁ : N = 30) 
  (h₂ : N + P = 110) :
  P - N = 50 := 
by
  sorry

end NUMINAMATH_GPT_pens_more_than_notebooks_l495_49551


namespace NUMINAMATH_GPT_calculate_crayons_lost_l495_49582

def initial_crayons := 440
def given_crayons := 111
def final_crayons := 223

def crayons_left_after_giving := initial_crayons - given_crayons
def crayons_lost := crayons_left_after_giving - final_crayons

theorem calculate_crayons_lost : crayons_lost = 106 :=
  by
    sorry

end NUMINAMATH_GPT_calculate_crayons_lost_l495_49582


namespace NUMINAMATH_GPT_ellipse_focus_value_l495_49578

theorem ellipse_focus_value (m : ℝ) (h1 : m > 0) :
  (∃ (x y : ℝ), (x, y) = (-4, 0) ∧ (25 - m^2 = 16)) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focus_value_l495_49578


namespace NUMINAMATH_GPT_final_price_correct_l495_49569

variable (original_price first_discount second_discount third_discount sales_tax : ℝ)
variable (final_discounted_price final_price: ℝ)

-- Define original price and discounts
def initial_price : ℝ := 20000
def discount1      : ℝ := 0.12
def discount2      : ℝ := 0.10
def discount3      : ℝ := 0.05
def tax_rate       : ℝ := 0.08

def price_after_first_discount : ℝ := initial_price * (1 - discount1)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - discount2)
def price_after_third_discount : ℝ := price_after_second_discount * (1 - discount3)
def final_sale_price : ℝ := price_after_third_discount * (1 + tax_rate)

-- Prove final sale price is 16251.84
theorem final_price_correct : final_sale_price = 16251.84 := by
  sorry

end NUMINAMATH_GPT_final_price_correct_l495_49569


namespace NUMINAMATH_GPT_find_x_l495_49509

-- Let x be a real number such that x > 0 and the area of the given triangle is 180.
theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : 3 * x^2 = 180) : x = 2 * Real.sqrt 15 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_x_l495_49509


namespace NUMINAMATH_GPT_second_parentheses_expression_eq_zero_l495_49561

def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem second_parentheses_expression_eq_zero :
  custom_op (Real.sqrt 6) (Real.sqrt 6) = 0 := by
  sorry

end NUMINAMATH_GPT_second_parentheses_expression_eq_zero_l495_49561


namespace NUMINAMATH_GPT_duration_of_period_l495_49596

/-- The duration of the period at which B gains Rs. 1125 by lending 
Rs. 25000 at rate of 11.5% per annum and borrowing the same 
amount at 10% per annum -/
theorem duration_of_period (principal : ℝ) (rate_borrow : ℝ) (rate_lend : ℝ) (gain : ℝ) : 
  ∃ (t : ℝ), principal = 25000 ∧ rate_borrow = 0.10 ∧ rate_lend = 0.115 ∧ gain = 1125 → 
  t = 3 :=
by
  sorry

end NUMINAMATH_GPT_duration_of_period_l495_49596


namespace NUMINAMATH_GPT_product_of_numbers_l495_49594

theorem product_of_numbers :
  ∃ (x y z : ℚ), (x + y + z = 30) ∧ (x = 3 * (y + z)) ∧ (y = 5 * z) ∧ (x * y * z = 175.78125) :=
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l495_49594


namespace NUMINAMATH_GPT_problem_statement_l495_49589

theorem problem_statement (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
sorry

end NUMINAMATH_GPT_problem_statement_l495_49589


namespace NUMINAMATH_GPT_number_of_integers_l495_49536

theorem number_of_integers (n : ℤ) : 
    25 < n^2 ∧ n^2 < 144 → ∃ l, l = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integers_l495_49536


namespace NUMINAMATH_GPT_probability_three_white_two_black_eq_eight_seventeen_l495_49574
-- Import Mathlib library to access combinatorics functions.

-- Define the total number of white and black balls.
def total_white := 8
def total_black := 7

-- The key function to calculate combinations.
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem conditions as constants.
def total_balls := total_white + total_black
def chosen_balls := 5
def white_balls_chosen := 3
def black_balls_chosen := 2

-- Calculate number of combinations.
noncomputable def total_combinations : ℕ := choose total_balls chosen_balls
noncomputable def white_combinations : ℕ := choose total_white white_balls_chosen
noncomputable def black_combinations : ℕ := choose total_black black_balls_chosen

-- Calculate the probability as a rational number.
noncomputable def probability_exact_three_white_two_black : ℚ :=
  (white_combinations * black_combinations : ℚ) / total_combinations

-- The theorem we want to prove
theorem probability_three_white_two_black_eq_eight_seventeen :
  probability_exact_three_white_two_black = 8 / 17 := by
  sorry

end NUMINAMATH_GPT_probability_three_white_two_black_eq_eight_seventeen_l495_49574


namespace NUMINAMATH_GPT_ashwin_rental_hours_l495_49542

theorem ashwin_rental_hours (x : ℕ) 
  (h1 : 25 + 10 * x = 125) : 1 + x = 11 :=
by
  sorry

end NUMINAMATH_GPT_ashwin_rental_hours_l495_49542


namespace NUMINAMATH_GPT_exp_product_correct_l495_49567

def exp_1 := (2 : ℕ) ^ 4
def exp_2 := (3 : ℕ) ^ 2
def exp_3 := (5 : ℕ) ^ 2
def exp_4 := (7 : ℕ)
def exp_5 := (11 : ℕ)
def final_value := exp_1 * exp_2 * exp_3 * exp_4 * exp_5

theorem exp_product_correct : final_value = 277200 := by
  sorry

end NUMINAMATH_GPT_exp_product_correct_l495_49567


namespace NUMINAMATH_GPT_couples_at_prom_l495_49515

theorem couples_at_prom (total_students attending_alone attending_with_partners couples : ℕ) 
  (h1 : total_students = 123) 
  (h2 : attending_alone = 3) 
  (h3 : attending_with_partners = total_students - attending_alone) 
  (h4 : couples = attending_with_partners / 2) : 
  couples = 60 := 
by 
  sorry

end NUMINAMATH_GPT_couples_at_prom_l495_49515


namespace NUMINAMATH_GPT_total_people_seated_l495_49572

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end NUMINAMATH_GPT_total_people_seated_l495_49572


namespace NUMINAMATH_GPT_renovation_services_are_credence_goods_and_choice_arguments_l495_49545

-- Define what credence goods are and the concept of information asymmetry
structure CredenceGood where
  information_asymmetry : Prop
  unobservable_quality  : Prop

-- Define renovation service as an instance of CredenceGood
def RenovationService : CredenceGood := {
  information_asymmetry := true,
  unobservable_quality := true
}

-- Primary conditions for choosing between construction company and private repair crew
structure ChoiceArgument where
  information_availability     : Prop
  warranty_and_accountability  : Prop
  higher_costs                 : Prop
  potential_bias_in_reviews    : Prop

-- Arguments for using construction company
def ConstructionCompanyArguments : ChoiceArgument := {
  information_availability := true,
  warranty_and_accountability := true,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Arguments against using construction company
def PrivateRepairCrewArguments : ChoiceArgument := {
  information_availability := false,
  warranty_and_accountability := false,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Proof statement to show renovation services are credence goods and economically reasoned arguments for/against
theorem renovation_services_are_credence_goods_and_choice_arguments:
  RenovationService = {
    information_asymmetry := true,
    unobservable_quality := true
  } ∧
  (ConstructionCompanyArguments.information_availability = true ∧
   ConstructionCompanyArguments.warranty_and_accountability = true) ∧
  (ConstructionCompanyArguments.higher_costs = true ∧
   ConstructionCompanyArguments.potential_bias_in_reviews = true) ∧
  (PrivateRepairCrewArguments.higher_costs = true ∧
   PrivateRepairCrewArguments.potential_bias_in_reviews = true) :=
by sorry

end NUMINAMATH_GPT_renovation_services_are_credence_goods_and_choice_arguments_l495_49545


namespace NUMINAMATH_GPT_problem1_problem2_l495_49525

-- Definitions for sets A and S
def setA (x : ℝ) : Prop := -7 ≤ 2 * x - 5 ∧ 2 * x - 5 ≤ 9
def setS (x k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

-- Preliminary ranges for x
lemma range_A : ∀ x, setA x ↔ -1 ≤ x ∧ x ≤ 7 := sorry

noncomputable def k_range1 (k : ℝ) : Prop := 2 ≤ k ∧ k ≤ 4
noncomputable def k_range2 (k : ℝ) : Prop := k < 2 ∨ k > 6

-- Proof problems in Lean 4

-- First problem statement
theorem problem1 (k : ℝ) : (∀ x, setS x k → setA x) ∧ (∃ x, setS x k) → k_range1 k := sorry

-- Second problem statement
theorem problem2 (k : ℝ) : (∀ x, ¬(setA x ∧ setS x k)) → k_range2 k := sorry

end NUMINAMATH_GPT_problem1_problem2_l495_49525


namespace NUMINAMATH_GPT_boys_neither_happy_nor_sad_l495_49517

theorem boys_neither_happy_nor_sad (total_children : ℕ)
  (happy_children sad_children neither_happy_nor_sad total_boys total_girls : ℕ)
  (happy_boys sad_girls : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_sad : sad_children = 10)
  (h_neither : neither_happy_nor_sad = 20)
  (h_boys : total_boys = 17)
  (h_girls : total_girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4) :
  ∃ (boys_neither_happy_nor_sad : ℕ), boys_neither_happy_nor_sad = 5 := by
  sorry

end NUMINAMATH_GPT_boys_neither_happy_nor_sad_l495_49517


namespace NUMINAMATH_GPT_find_n_l495_49529

noncomputable def problem_statement (m n : ℤ) : Prop :=
  (∀ x : ℝ, x^2 - (m + 2) * x + (m - 2) = 0 → ∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 * x2 < 0 ∧ x1 > |x2|) ∧
  (∃ r1 r2 : ℚ, r1 * r2 = 2 ∧ m * (r1 * r1 + r2 * r2) = (n - 2) * (r1 + r2) + m^2 - 3)

theorem find_n (m : ℤ) (hm : -2 < m ∧ m < 2) : 
  problem_statement m 5 ∨ problem_statement m (-1) :=
sorry

end NUMINAMATH_GPT_find_n_l495_49529


namespace NUMINAMATH_GPT_solve_polynomial_l495_49541

theorem solve_polynomial (z : ℂ) : z^6 - 9 * z^3 + 8 = 0 ↔ z = 1 ∨ z = 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_polynomial_l495_49541


namespace NUMINAMATH_GPT_complement_of_A_in_U_l495_49559

def U : Set ℝ := {x | x > 0}
def A : Set ℝ := {x | x ≥ 2}
def complement_U_A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem complement_of_A_in_U :
  (U \ A) = complement_U_A :=
sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l495_49559


namespace NUMINAMATH_GPT_height_of_boxes_l495_49506

theorem height_of_boxes
  (volume_required : ℝ)
  (price_per_box : ℝ)
  (min_expenditure : ℝ)
  (volume_per_box : ∀ n : ℕ, n = min_expenditure / price_per_box -> ℝ) :
  volume_required = 3060000 ->
  price_per_box = 0.50 ->
  min_expenditure = 255 ->
  ∃ h : ℝ, h = 19 := by
  sorry

end NUMINAMATH_GPT_height_of_boxes_l495_49506


namespace NUMINAMATH_GPT_probability_in_interval_l495_49522

theorem probability_in_interval (a b c d : ℝ) (h1 : a = 2) (h2 : b = 10) (h3 : c = 5) (h4 : d = 7) :
  (d - c) / (b - a) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_in_interval_l495_49522


namespace NUMINAMATH_GPT_most_followers_after_three_weeks_l495_49535

def initial_followers_susy := 100
def initial_followers_sarah := 50
def first_week_gain_susy := 40
def second_week_gain_susy := first_week_gain_susy / 2
def third_week_gain_susy := second_week_gain_susy / 2
def first_week_gain_sarah := 90
def second_week_gain_sarah := first_week_gain_sarah / 3
def third_week_gain_sarah := second_week_gain_sarah / 3

def total_followers_susy := initial_followers_susy + first_week_gain_susy + second_week_gain_susy + third_week_gain_susy
def total_followers_sarah := initial_followers_sarah + first_week_gain_sarah + second_week_gain_sarah + third_week_gain_sarah

theorem most_followers_after_three_weeks : max total_followers_susy total_followers_sarah = 180 :=
by
  sorry

end NUMINAMATH_GPT_most_followers_after_three_weeks_l495_49535


namespace NUMINAMATH_GPT_entrance_fee_increase_l495_49511

theorem entrance_fee_increase
  (entrance_fee_under_18 : ℕ)
  (rides_cost : ℕ)
  (num_rides : ℕ)
  (total_spent : ℕ)
  (total_cost_twins : ℕ)
  (total_ride_cost_twins : ℕ)
  (amount_spent_joe : ℕ)
  (total_ride_cost_joe : ℕ)
  (joe_entrance_fee : ℕ)
  (increase : ℕ)
  (percentage_increase : ℕ)
  (h1 : entrance_fee_under_18 = 5)
  (h2 : rides_cost = 50) -- representing $0.50 as 50 cents to maintain integer calculations
  (h3 : num_rides = 3)
  (h4 : total_spent = 2050) -- representing $20.5 as 2050 cents
  (h5 : total_cost_twins = 1300) -- combining entrance fees and cost of rides for the twins in cents
  (h6 : total_ride_cost_twins = 300) -- cost of rides for twins in cents
  (h7 : amount_spent_joe = 750) -- representing $7.5 as 750 cents
  (h8 : total_ride_cost_joe = 150) -- cost of rides for Joe in cents
  (h9 : joe_entrance_fee = 600) -- representing $6 as 600 cents
  (h10 : increase = 100) -- increase in entrance fee in cents
  (h11 : percentage_increase = 20) :
  percentage_increase = ((increase * 100) / entrance_fee_under_18) :=
sorry

end NUMINAMATH_GPT_entrance_fee_increase_l495_49511


namespace NUMINAMATH_GPT_compare_m_n_l495_49557

noncomputable def m (a : ℝ) : ℝ := 6^a / (36^(a + 1) + 1)
noncomputable def n (b : ℝ) : ℝ := (1/3) * b^2 - b + (5/6)

theorem compare_m_n (a b : ℝ) : m a ≤ n b := sorry

end NUMINAMATH_GPT_compare_m_n_l495_49557


namespace NUMINAMATH_GPT_max_value_expression_l495_49500

theorem max_value_expression (k : ℕ) (a b c : ℝ) (h : k > 0) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (habc : a + b + c = 3 * k) :
  a^(3 * k - 1) * b + b^(3 * k - 1) * c + c^(3 * k - 1) * a + k^2 * a^k * b^k * c^k ≤ (3 * k - 1)^(3 * k - 1) :=
sorry

end NUMINAMATH_GPT_max_value_expression_l495_49500


namespace NUMINAMATH_GPT_cereal_original_price_l495_49562

-- Define the known conditions as constants
def initial_money : ℕ := 60
def celery_price : ℕ := 5
def bread_price : ℕ := 8
def milk_full_price : ℕ := 10
def milk_discount : ℕ := 10
def milk_price : ℕ := milk_full_price - (milk_full_price * milk_discount / 100)
def potato_price : ℕ := 1
def potato_quantity : ℕ := 6
def potatoes_total_price : ℕ := potato_price * potato_quantity
def coffee_remaining_money : ℕ := 26
def total_spent_exclude_coffee : ℕ := initial_money - coffee_remaining_money
def spent_on_other_items : ℕ := celery_price + bread_price + milk_price + potatoes_total_price
def spent_on_cereal : ℕ := total_spent_exclude_coffee - spent_on_other_items
def cereal_discount : ℕ := 50

theorem cereal_original_price :
  (spent_on_other_items = celery_price + bread_price + milk_price + potatoes_total_price) →
  (total_spent_exclude_coffee = initial_money - coffee_remaining_money) →
  (spent_on_cereal = total_spent_exclude_coffee - spent_on_other_items) →
  (spent_on_cereal * 2 = 12) :=
by {
  -- proof here
  sorry
}

end NUMINAMATH_GPT_cereal_original_price_l495_49562


namespace NUMINAMATH_GPT_inequality_count_l495_49548

theorem inequality_count {a b : ℝ} (h : 1/a < 1/b ∧ 1/b < 0) :
  (if (|a| > |b|) then 0 else 1) + 
  (if (a + b > ab) then 1 else 0) +
  (if (a / b + b / a > 2) then 1 else 0) + 
  (if (a^2 / b < 2 * a - b) then 1 else 0) = 2 :=
sorry

end NUMINAMATH_GPT_inequality_count_l495_49548


namespace NUMINAMATH_GPT_more_cats_than_dogs_l495_49593

-- Define the number of cats and dogs
def c : ℕ := 23
def d : ℕ := 9

-- The theorem we need to prove
theorem more_cats_than_dogs : c - d = 14 := by
  sorry

end NUMINAMATH_GPT_more_cats_than_dogs_l495_49593


namespace NUMINAMATH_GPT_maximize_profit_l495_49520

-- Definitions
def initial_employees := 320
def profit_per_employee := 200000
def profit_increase_per_layoff := 20000
def expense_per_laid_off_employee := 60000
def min_employees := (3 * initial_employees) / 4
def profit_function (x : ℝ) := -0.2 * x^2 + 38 * x + 6400

-- The main statement
theorem maximize_profit : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 80 ∧ (∀ y : ℝ, 0 ≤ y ∧ y ≤ 80 → profit_function y ≤ profit_function x) ∧ x = 80 :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l495_49520


namespace NUMINAMATH_GPT_james_lifting_ratio_correct_l495_49538

theorem james_lifting_ratio_correct :
  let lt_initial := 2200
  let bw_initial := 245
  let lt_gain_percentage := 0.15
  let bw_gain := 8
  let lt_final := lt_initial + lt_initial * lt_gain_percentage
  let bw_final := bw_initial + bw_gain
  (lt_final / bw_final) = 10 :=
by
  sorry

end NUMINAMATH_GPT_james_lifting_ratio_correct_l495_49538


namespace NUMINAMATH_GPT_log2_15_eq_formula_l495_49505

theorem log2_15_eq_formula (a b : ℝ) (h1 : a = Real.log 6 / Real.log 3) (h2 : b = Real.log 20 / Real.log 5) :
  Real.log 15 / Real.log 2 = (2 * a + b - 3) / ((a - 1) * (b - 1)) :=
by
  sorry

end NUMINAMATH_GPT_log2_15_eq_formula_l495_49505


namespace NUMINAMATH_GPT_like_terms_satisfy_conditions_l495_49518

theorem like_terms_satisfy_conditions (m n : ℤ) (h1 : m - 1 = n) (h2 : m + n = 3) :
  m = 2 ∧ n = 1 := by
  sorry

end NUMINAMATH_GPT_like_terms_satisfy_conditions_l495_49518


namespace NUMINAMATH_GPT_number_of_valid_pairs_l495_49584

-- Definition of the conditions according to step (a)
def perimeter (l w : ℕ) : Prop := 2 * (l + w) = 80
def integer_lengths (l w : ℕ) : Prop := true
def length_greater_than_width (l w : ℕ) : Prop := l > w

-- The mathematical proof problem according to step (c)
theorem number_of_valid_pairs : ∃ n : ℕ, 
  (∀ l w : ℕ, perimeter l w → integer_lengths l w → length_greater_than_width l w → ∃! pair : (ℕ × ℕ), pair = (l, w)) ∧
  n = 19 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l495_49584


namespace NUMINAMATH_GPT_sum_four_digit_even_numbers_l495_49531

-- Define the digits set
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the set of valid units digits for even numbers
def even_units : Finset ℕ := {0, 2, 4}

-- Define the set of all four-digit numbers using the provided digits
def four_digit_even_numbers : Finset ℕ :=
  (Finset.range (10000) \ Finset.range (1000)).filter (λ n =>
    n % 10 ∈ even_units ∧
    (n / 1000) ∈ digits ∧
    ((n / 100) % 10) ∈ digits ∧
    ((n / 10) % 10) ∈ digits)

theorem sum_four_digit_even_numbers :
  (four_digit_even_numbers.sum (λ x => x)) = 1769580 :=
  sorry

end NUMINAMATH_GPT_sum_four_digit_even_numbers_l495_49531


namespace NUMINAMATH_GPT_sum_of_reciprocal_of_roots_l495_49580

theorem sum_of_reciprocal_of_roots :
  ∀ x1 x2 : ℝ, (x1 * x2 = 2) → (x1 + x2 = 3) → (1 / x1 + 1 / x2 = 3 / 2) :=
by
  intros x1 x2 h_prod h_sum
  sorry

end NUMINAMATH_GPT_sum_of_reciprocal_of_roots_l495_49580


namespace NUMINAMATH_GPT_number_of_tiles_l495_49599

noncomputable def tile_count (room_length : ℝ) (room_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) :=
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  room_area / tile_area

theorem number_of_tiles :
  tile_count 10 15 (1 / 4) (5 / 12) = 1440 := by
  sorry

end NUMINAMATH_GPT_number_of_tiles_l495_49599


namespace NUMINAMATH_GPT_programs_produce_same_output_l495_49533

def sum_program_a : ℕ :=
  let S := (Finset.range 1000).sum (λ i => i + 1)
  S

def sum_program_b : ℕ :=
  let S := (Finset.range 1000).sum (λ i => 1000 - i)
  S

theorem programs_produce_same_output :
  sum_program_a = sum_program_b := by
  sorry

end NUMINAMATH_GPT_programs_produce_same_output_l495_49533


namespace NUMINAMATH_GPT_domain_of_g_l495_49564

theorem domain_of_g (t : ℝ) : (t - 1)^2 + (t + 1)^2 + t ≠ 0 :=
  by
  sorry

end NUMINAMATH_GPT_domain_of_g_l495_49564


namespace NUMINAMATH_GPT_number_of_chocolate_bars_by_theresa_l495_49571

-- Define the number of chocolate bars and soda cans that Kayla bought
variables (C S : ℕ)

-- Assume the total number of chocolate bars and soda cans Kayla bought is 15
axiom total_purchased_by_kayla : C + S = 15

-- Define the number of chocolate bars Theresa bought as twice the number Kayla bought
def chocolate_bars_purchased_by_theresa := 2 * C

-- The theorem to prove
theorem number_of_chocolate_bars_by_theresa : chocolate_bars_purchased_by_theresa = 2 * C :=
by
  -- The proof is omitted as instructed
  sorry

end NUMINAMATH_GPT_number_of_chocolate_bars_by_theresa_l495_49571


namespace NUMINAMATH_GPT_find_x_range_l495_49503

def tight_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, 0 < n → 1/2 ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

theorem find_x_range
  (a : ℕ → ℝ)
  (h_tight : tight_sequence a)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3 / 2)
  (h3 : ∃ x, a 3 = x)
  (h4 : a 4 = 4) :
  ∃ x, (2 : ℝ) ≤ x ∧ x ≤ (3 : ℝ) :=
sorry

end NUMINAMATH_GPT_find_x_range_l495_49503


namespace NUMINAMATH_GPT_original_acid_concentration_l495_49532

theorem original_acid_concentration (P : ℝ) (h1 : 0.5 * P + 0.5 * 20 = 35) : P = 50 :=
by
  sorry

end NUMINAMATH_GPT_original_acid_concentration_l495_49532


namespace NUMINAMATH_GPT_common_element_exists_l495_49556

theorem common_element_exists {S : Fin 2011 → Set ℤ}
  (h_nonempty : ∀ (i : Fin 2011), (S i).Nonempty)
  (h_consecutive : ∀ (i : Fin 2011), ∃ a b : ℤ, S i = Set.Icc a b)
  (h_common : ∀ (i j : Fin 2011), (S i ∩ S j).Nonempty) :
  ∃ a : ℤ, 0 < a ∧ ∀ (i : Fin 2011), a ∈ S i := sorry

end NUMINAMATH_GPT_common_element_exists_l495_49556


namespace NUMINAMATH_GPT_asymptote_equation_of_hyperbola_l495_49534

def hyperbola_eccentricity (a : ℝ) (h : a > 0) : Prop :=
  let e := Real.sqrt 2
  e = Real.sqrt (1 + a^2) / a

theorem asymptote_equation_of_hyperbola :
  ∀ (a : ℝ) (h : a > 0), hyperbola_eccentricity a h → (∀ x y : ℝ, (x^2 - y^2 = 1 → y = x ∨ y = -x)) :=
by
  intro a h he
  sorry

end NUMINAMATH_GPT_asymptote_equation_of_hyperbola_l495_49534


namespace NUMINAMATH_GPT_solution_set_inequality_l495_49581

theorem solution_set_inequality (x : ℝ) :
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ (-2 ≤ x ∧ x ≤ 2) ∨ (x = 6) := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l495_49581


namespace NUMINAMATH_GPT_exponential_inequality_l495_49573

variables (x a b : ℝ)

theorem exponential_inequality (h1 : x > 0) (h2 : 1 < b^x) (h3 : b^x < a^x) : 1 < b ∧ b < a :=
by
   sorry

end NUMINAMATH_GPT_exponential_inequality_l495_49573


namespace NUMINAMATH_GPT_arithmetic_progression_rth_term_l495_49553

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 2 * n + 3 * n^2

theorem arithmetic_progression_rth_term : (S r) - (S (r - 1)) = 6 * r - 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_rth_term_l495_49553


namespace NUMINAMATH_GPT_find_a9_l495_49554

theorem find_a9 (a : ℕ → ℕ) 
  (h_add : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
  (h_a2 : a 2 = 4) 
  : a 9 = 18 :=
sorry

end NUMINAMATH_GPT_find_a9_l495_49554


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l495_49547

-- Definitions for conditions
variables (V_b V_s : ℝ)

-- The conditions provided for the problem
def along_stream := V_b + V_s = 13
def against_stream := V_b - V_s = 5

-- The theorem we want to prove
theorem boat_speed_in_still_water (h1 : along_stream V_b V_s) (h2 : against_stream V_b V_s) : V_b = 9 :=
sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l495_49547


namespace NUMINAMATH_GPT_P_ne_77_for_integers_l495_49555

def P (x y : ℤ) : ℤ :=
  x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_ne_77_for_integers (x y : ℤ) : P x y ≠ 77 :=
by
  sorry

end NUMINAMATH_GPT_P_ne_77_for_integers_l495_49555


namespace NUMINAMATH_GPT_meeting_point_distance_l495_49528

theorem meeting_point_distance
  (distance_to_top : ℝ)
  (total_distance : ℝ)
  (jack_start_time : ℝ)
  (jack_uphill_speed : ℝ)
  (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ)
  (jill_downhill_speed : ℝ)
  (meeting_point_distance : ℝ):
  distance_to_top = 5 -> total_distance = 10 -> jack_start_time = 10 / 60 ->
  jack_uphill_speed = 15 -> jack_downhill_speed = 20 ->
  jill_uphill_speed = 16 -> jill_downhill_speed = 22 ->
  meeting_point_distance = 35 / 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_meeting_point_distance_l495_49528


namespace NUMINAMATH_GPT_probability_same_color_l495_49565

-- Define the total number of plates
def totalPlates : ℕ := 6 + 5 + 3

-- Define the number of red plates, blue plates, and green plates
def redPlates : ℕ := 6
def bluePlates : ℕ := 5
def greenPlates : ℕ := 3

-- Define the total number of ways to choose 3 plates from 14
def totalWaysChoose3 : ℕ := Nat.choose totalPlates 3

-- Define the number of ways to choose 3 red plates, 3 blue plates, and 3 green plates
def redWaysChoose3 : ℕ := Nat.choose redPlates 3
def blueWaysChoose3 : ℕ := Nat.choose bluePlates 3
def greenWaysChoose3 : ℕ := Nat.choose greenPlates 3

-- Calculate the total number of favorable combinations (all plates being the same color)
def favorableCombinations : ℕ := redWaysChoose3 + blueWaysChoose3 + greenWaysChoose3

-- State the theorem: the probability that all plates are of the same color.
theorem probability_same_color : (favorableCombinations : ℚ) / (totalWaysChoose3 : ℚ) = 31 / 364 := by sorry

end NUMINAMATH_GPT_probability_same_color_l495_49565


namespace NUMINAMATH_GPT_cyclic_quadrilateral_iff_condition_l495_49570

theorem cyclic_quadrilateral_iff_condition
  (α β γ δ : ℝ)
  (h : α + β + γ + δ = 2 * π) :
  (α * β + α * δ + γ * β + γ * δ = π^2) ↔ (α + γ = π ∧ β + δ = π) :=
by
  sorry

end NUMINAMATH_GPT_cyclic_quadrilateral_iff_condition_l495_49570


namespace NUMINAMATH_GPT_avg_salary_of_Raj_and_Roshan_l495_49576

variable (R S : ℕ)

theorem avg_salary_of_Raj_and_Roshan (h1 : (R + S + 7000) / 3 = 5000) : (R + S) / 2 = 4000 := by
  sorry

end NUMINAMATH_GPT_avg_salary_of_Raj_and_Roshan_l495_49576


namespace NUMINAMATH_GPT_vertex_of_quadratic_function_l495_49583

theorem vertex_of_quadratic_function :
  ∀ x: ℝ, (2 - (x + 1)^2) = 2 - (x + 1)^2 → (∃ h k : ℝ, (h, k) = (-1, 2) ∧ ∀ x: ℝ, (2 - (x + 1)^2) = k - (x - h)^2) :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_quadratic_function_l495_49583


namespace NUMINAMATH_GPT_polynomial_expansion_l495_49560

theorem polynomial_expansion : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24 * x - 36 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l495_49560


namespace NUMINAMATH_GPT_geometric_seq_a5_a7_l495_49575

theorem geometric_seq_a5_a7 (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * q)
  (h3 : a 3 + a 5 = 6)
  (q : ℝ) :
  (a 5 + a 7 = 12) :=
sorry

end NUMINAMATH_GPT_geometric_seq_a5_a7_l495_49575


namespace NUMINAMATH_GPT_S_sum_l495_49588

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -(n / 2)
  else (n + 1) / 2

theorem S_sum :
  S 19 + S 37 + S 52 = 3 :=
by
  sorry

end NUMINAMATH_GPT_S_sum_l495_49588


namespace NUMINAMATH_GPT_probability_of_selecting_cooking_is_one_fourth_l495_49590

-- Define the set of available courses
def courses : List String := ["planting", "cooking", "pottery", "carpentry"]

-- Define the probability of selecting a specific course from the list
def probability_of_selecting_cooking (course : String) (choices : List String) : ℚ :=
  if course ∈ choices then 1 / choices.length else 0

-- Prove that the probability of selecting "cooking" from the four courses is 1/4
theorem probability_of_selecting_cooking_is_one_fourth : 
  probability_of_selecting_cooking "cooking" courses = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_cooking_is_one_fourth_l495_49590


namespace NUMINAMATH_GPT_find_a_l495_49598

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then (1/2)^x - 7 else x^2

theorem find_a (a : ℝ) (h : f a = 1) : a = -3 ∨ a = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l495_49598


namespace NUMINAMATH_GPT_price_of_toy_organizers_is_78_l495_49591

variable (P : ℝ) -- Price per set of toy organizers

-- Conditions
def total_cost_of_toy_organizers (P : ℝ) : ℝ := 3 * P
def total_cost_of_gaming_chairs : ℝ := 2 * 83
def total_sales (P : ℝ) : ℝ := total_cost_of_toy_organizers P + total_cost_of_gaming_chairs
def delivery_fee (P : ℝ) : ℝ := 0.05 * total_sales P
def total_amount_paid (P : ℝ) : ℝ := total_sales P + delivery_fee P

-- Proof statement
theorem price_of_toy_organizers_is_78 (h : total_amount_paid P = 420) : P = 78 :=
by
  sorry

end NUMINAMATH_GPT_price_of_toy_organizers_is_78_l495_49591


namespace NUMINAMATH_GPT_no_x_intersect_one_x_intersect_l495_49523

variable (m : ℝ)

-- Define the original quadratic function
def quadratic_function (x : ℝ) := x^2 - 2 * m * x + m^2 + 3

-- 1. Prove the function does not intersect the x-axis
theorem no_x_intersect : ∀ m, ∀ x : ℝ, quadratic_function m x ≠ 0 := by
  intros
  unfold quadratic_function
  sorry

-- 2. Prove that translating down by 3 units intersects the x-axis at one point
def translated_quadratic (x : ℝ) := (x - m)^2

theorem one_x_intersect : ∃ x : ℝ, translated_quadratic m x = 0 := by
  unfold translated_quadratic
  sorry

end NUMINAMATH_GPT_no_x_intersect_one_x_intersect_l495_49523


namespace NUMINAMATH_GPT_length_of_train_correct_l495_49568

noncomputable def length_of_train (time_pass_man : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - man_speed_kmh
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  relative_speed_ms * time_pass_man

theorem length_of_train_correct :
  length_of_train 29.997600191984642 60 6 = 449.96400287976963 := by
  sorry

end NUMINAMATH_GPT_length_of_train_correct_l495_49568


namespace NUMINAMATH_GPT_storm_deposit_eq_120_billion_gallons_l495_49526

theorem storm_deposit_eq_120_billion_gallons :
  ∀ (initial_content : ℝ) (full_percentage_pre_storm : ℝ) (full_percentage_post_storm : ℝ) (reservoir_capacity : ℝ),
  initial_content = 220 * 10^9 → 
  full_percentage_pre_storm = 0.55 →
  full_percentage_post_storm = 0.85 →
  reservoir_capacity = initial_content / full_percentage_pre_storm →
  (full_percentage_post_storm * reservoir_capacity - initial_content) = 120 * 10^9 :=
by
  intro initial_content full_percentage_pre_storm full_percentage_post_storm reservoir_capacity
  intros h_initial_content h_pre_storm h_post_storm h_capacity
  sorry

end NUMINAMATH_GPT_storm_deposit_eq_120_billion_gallons_l495_49526


namespace NUMINAMATH_GPT_sum_prob_less_one_l495_49579

theorem sum_prob_less_one (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_prob_less_one_l495_49579


namespace NUMINAMATH_GPT_impossible_division_l495_49585

noncomputable def total_matches := 1230

theorem impossible_division :
  ∀ (x y z : ℕ), 
  (x + y + z = total_matches) → 
  (z = (1 / 2) * (x + y + z)) → 
  false :=
by
  sorry

end NUMINAMATH_GPT_impossible_division_l495_49585
