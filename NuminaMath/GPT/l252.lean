import Mathlib

namespace multiplication_expansion_l252_25208

theorem multiplication_expansion (y : ℤ) :
  (y^4 + 9 * y^2 + 81) * (y^2 - 9) = y^6 - 729 :=
by
  sorry

end multiplication_expansion_l252_25208


namespace mod_remainder_l252_25261

theorem mod_remainder (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end mod_remainder_l252_25261


namespace cost_of_1500_pieces_of_gum_in_dollars_l252_25215

theorem cost_of_1500_pieces_of_gum_in_dollars :
  (2 * 1500 * (1 - 0.10) / 100) = 27 := sorry

end cost_of_1500_pieces_of_gum_in_dollars_l252_25215


namespace min_binary_questions_to_determine_number_l252_25283

theorem min_binary_questions_to_determine_number (x : ℕ) (h : 10 ≤ x ∧ x ≤ 19) : 
  ∃ (n : ℕ), n = 3 := 
sorry

end min_binary_questions_to_determine_number_l252_25283


namespace avg_two_ab_l252_25201

-- Defining the weights and conditions
variables (A B C : ℕ)

-- The conditions provided in the problem
def avg_three (A B C : ℕ) := (A + B + C) / 3 = 45
def avg_two_bc (B C : ℕ) := (B + C) / 2 = 43
def weight_b (B : ℕ) := B = 35

-- The target proof statement
theorem avg_two_ab (A B C : ℕ) (h1 : avg_three A B C) (h2 : avg_two_bc B C) (h3 : weight_b B) : (A + B) / 2 = 42 := 
sorry

end avg_two_ab_l252_25201


namespace convex_pentadecagon_diagonals_l252_25214

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem convex_pentadecagon_diagonals :
  number_of_diagonals 15 = 90 :=
by sorry

end convex_pentadecagon_diagonals_l252_25214


namespace non_raining_hours_l252_25229

-- Definitions based on the conditions.
def total_hours := 9
def rained_hours := 4

-- Problem statement: Prove that the non-raining hours equals to 5 given total_hours and rained_hours.
theorem non_raining_hours : (total_hours - rained_hours = 5) :=
by
  -- The proof is omitted with "sorry" to indicate the missing proof.
  sorry

end non_raining_hours_l252_25229


namespace student_score_variance_l252_25273

noncomputable def variance_student_score : ℝ :=
  let number_of_questions := 25
  let probability_correct := 0.8
  let score_correct := 4
  let variance_eta := number_of_questions * probability_correct * (1 - probability_correct)
  let variance_xi := (score_correct ^ 2) * variance_eta
  variance_xi

theorem student_score_variance : variance_student_score = 64 := by
  sorry

end student_score_variance_l252_25273


namespace unique_vector_a_l252_25280

-- Defining the vectors
def vector_a (x y : ℝ) : ℝ × ℝ := (x, y)
def vector_b (x y : ℝ) : ℝ × ℝ := (x^2, y^2)
def vector_c : ℝ × ℝ := (1, 1)
def vector_d : ℝ × ℝ := (2, 2)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The Lean statement to prove
theorem unique_vector_a (x y : ℝ) 
  (h1 : dot_product (vector_a x y) vector_c = 1)
  (h2 : dot_product (vector_b x y) vector_d = 1) : 
  vector_a x y = vector_a (1/2) (1/2) :=
by {
  sorry 
}

end unique_vector_a_l252_25280


namespace area_PQR_l252_25217

-- Define the coordinates of the points
def P : ℝ × ℝ := (-3, 4)
def Q : ℝ × ℝ := (4, 9)
def R : ℝ × ℝ := (5, -3)

-- Function to calculate the area of a triangle given three points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

-- Statement to prove the area of triangle PQR is 44.5
theorem area_PQR : area_of_triangle P Q R = 44.5 := sorry

end area_PQR_l252_25217


namespace sugar_percentage_first_solution_l252_25216

theorem sugar_percentage_first_solution 
  (x : ℝ) (h1 : 0 < x ∧ x < 100) 
  (h2 : 17 = 3 / 4 * x + 1 / 4 * 38) : 
  x = 10 :=
sorry

end sugar_percentage_first_solution_l252_25216


namespace dealer_gross_profit_l252_25291

noncomputable def computeGrossProfit (purchasePrice initialMarkupRate discountRate salesTaxRate: ℝ) : ℝ :=
  let initialSellingPrice := purchasePrice / (1 - initialMarkupRate)
  let discount := discountRate * initialSellingPrice
  let discountedPrice := initialSellingPrice - discount
  let salesTax := salesTaxRate * discountedPrice
  let finalSellingPrice := discountedPrice + salesTax
  finalSellingPrice - purchasePrice - discount

theorem dealer_gross_profit 
  (purchasePrice : ℝ)
  (initialMarkupRate : ℝ)
  (discountRate : ℝ)
  (salesTaxRate : ℝ) 
  (grossProfit : ℝ) :
  purchasePrice = 150 →
  initialMarkupRate = 0.25 →
  discountRate = 0.10 →
  salesTaxRate = 0.05 →
  grossProfit = 19 →
  computeGrossProfit purchasePrice initialMarkupRate discountRate salesTaxRate = grossProfit :=
  by
    intros hp hm hd hs hg
    rw [hp, hm, hd, hs, hg]
    rw [computeGrossProfit]
    sorry

end dealer_gross_profit_l252_25291


namespace smallest_total_books_l252_25237

-- Definitions based on conditions
def physics_books (x : ℕ) := 3 * x
def chemistry_books (x : ℕ) := 2 * x
def biology_books (x : ℕ) := (3 / 2 : ℚ) * x

-- Total number of books
def total_books (x : ℕ) := physics_books x + chemistry_books x + biology_books x

-- Statement of the theorem
theorem smallest_total_books :
  ∃ x : ℕ, total_books x = 15 ∧ 
           (∀ y : ℕ, y < x → total_books y % 1 ≠ 0) :=
sorry

end smallest_total_books_l252_25237


namespace solve_for_x_l252_25277

-- Define the necessary condition
def problem_statement (x : ℚ) : Prop :=
  x / 4 - x - 3 / 6 = 1

-- Prove that if the condition holds, then x = -14/9
theorem solve_for_x (x : ℚ) (h : problem_statement x) : x = -14 / 9 :=
by
  sorry

end solve_for_x_l252_25277


namespace square_side_length_l252_25295

theorem square_side_length (x : ℝ) (h : 4 * x = 8 * Real.pi) : x = 6.28 := 
by {
  -- proof will go here
  sorry
}

end square_side_length_l252_25295


namespace find_wrong_quotient_l252_25259

-- Define the conditions
def correct_divisor : Nat := 21
def correct_quotient : Nat := 24
def mistaken_divisor : Nat := 12
def dividend : Nat := correct_divisor * correct_quotient

-- State the theorem to prove the wrong quotient
theorem find_wrong_quotient : (dividend / mistaken_divisor) = 42 := by
  sorry

end find_wrong_quotient_l252_25259


namespace items_in_descending_order_l252_25218

-- Assume we have four real numbers representing the weights of the items.
variables (C S B K : ℝ)

-- The conditions given in the problem.
axiom h1 : S > B
axiom h2 : C + B > S + K
axiom h3 : K + C = S + B

-- Define a predicate to check if the weights are in descending order.
def DescendingOrder (C S B K : ℝ) : Prop :=
  C > S ∧ S > B ∧ B > K

-- The theorem to prove the descending order of weights.
theorem items_in_descending_order : DescendingOrder C S B K :=
sorry

end items_in_descending_order_l252_25218


namespace peter_needs_5000_for_vacation_l252_25212

variable (currentSavings : ℕ) (monthlySaving : ℕ) (months : ℕ)

-- Conditions
def peterSavings := currentSavings
def monthlySavings := monthlySaving
def savingDuration := months

-- Goal
def vacationFundsRequired (currentSavings monthlySaving months : ℕ) : ℕ :=
  currentSavings + (monthlySaving * months)

theorem peter_needs_5000_for_vacation
  (h1 : currentSavings = 2900)
  (h2 : monthlySaving = 700)
  (h3 : months = 3) :
  vacationFundsRequired currentSavings monthlySaving months = 5000 := by
  sorry

end peter_needs_5000_for_vacation_l252_25212


namespace initial_ratio_of_milk_water_l252_25269

theorem initial_ratio_of_milk_water (M W : ℝ) (H1 : M + W = 85) (H2 : M / (W + 5) = 3) : M / W = 27 / 7 :=
by sorry

end initial_ratio_of_milk_water_l252_25269


namespace sin_ratio_equal_one_or_neg_one_l252_25204

theorem sin_ratio_equal_one_or_neg_one
  (a b : Real)
  (h1 : Real.cos (a + b) = 1/4)
  (h2 : Real.cos (a - b) = 3/4) :
  (Real.sin a) / (Real.sin b) = 1 ∨ (Real.sin a) / (Real.sin b) = -1 :=
sorry

end sin_ratio_equal_one_or_neg_one_l252_25204


namespace theater_total_revenue_l252_25243

theorem theater_total_revenue :
  let seats := 400
  let capacity := 0.8
  let ticket_price := 30
  let days := 3
  seats * capacity * ticket_price * days = 28800 := by
  sorry

end theater_total_revenue_l252_25243


namespace sarah_total_height_in_cm_l252_25279

def sarah_height_in_inches : ℝ := 54
def book_thickness_in_inches : ℝ := 2
def conversion_factor : ℝ := 2.54

def total_height_in_inches : ℝ := sarah_height_in_inches + book_thickness_in_inches
def total_height_in_cm : ℝ := total_height_in_inches * conversion_factor

theorem sarah_total_height_in_cm : total_height_in_cm = 142.2 :=
by
  -- Skip the proof for now
  sorry

end sarah_total_height_in_cm_l252_25279


namespace perfect_square_problem_l252_25245

-- Define the given conditions and question
theorem perfect_square_problem 
  (a b c : ℕ) 
  (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h_cond: 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c + 1) : 
  ∃ k : ℕ, k^2 = a^2 + b^2 - a * b * c := 
sorry

end perfect_square_problem_l252_25245


namespace determine_m_l252_25263

theorem determine_m (a b c m : ℤ) 
  (h1 : c = -4 * a - 2 * b)
  (h2 : 70 < 4 * (8 * a + b) ∧ 4 * (8 * a + b) < 80)
  (h3 : 110 < 5 * (9 * a + b) ∧ 5 * (9 * a + b) < 120)
  (h4 : 2000 * m < (2500 * a + 50 * b + c) ∧ (2500 * a + 50 * b + c) < 2000 * (m + 1)) :
  m = 5 := sorry

end determine_m_l252_25263


namespace prove_inequality_l252_25286

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ (1-Real.sqrt 5)/2 ∧ x ≠ (1+Real.sqrt 5)/2

noncomputable def valid_intervals (x : ℝ) : Prop :=
  (x ≥ -1 ∧ x < (1 - Real.sqrt 5) / 2) ∨
  ((1 - Real.sqrt 5) / 2 < x ∧ x < 0) ∨
  (0 < x ∧ x < (1 + Real.sqrt 5) / 2) ∨
  (x > (1 + Real.sqrt 5) / 2)

theorem prove_inequality (x : ℝ) (hx : valid_x x) :
  (x^2 + x^3 - x^4) / (x + x^2 - x^3) ≥ -1 ↔ valid_intervals x := by
  sorry

end prove_inequality_l252_25286


namespace simplify_expression_l252_25299

theorem simplify_expression (x : ℝ) :
  x * (4 * x^3 - 3 * x + 2) - 6 * (2 * x^3 + x^2 - 3 * x + 4) = 4 * x^4 - 12 * x^3 - 9 * x^2 + 20 * x - 24 :=
by sorry

end simplify_expression_l252_25299


namespace intersection_A_B_l252_25239

-- Define the sets A and B based on given conditions
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {x | -1 < x}

-- The statement to prove
theorem intersection_A_B : (A ∩ B) = {x | -1 < x ∧ x < 4} :=
  sorry

end intersection_A_B_l252_25239


namespace king_luis_courtiers_are_odd_l252_25228

theorem king_luis_courtiers_are_odd (n : ℕ) 
  (h : ∀ i : ℕ, i < n → ∃ j : ℕ, j < n ∧ i ≠ j) : 
  ¬ Even n := 
sorry

end king_luis_courtiers_are_odd_l252_25228


namespace button_remainders_l252_25281

theorem button_remainders 
  (a : ℤ)
  (h1 : a % 2 = 1)
  (h2 : a % 3 = 1)
  (h3 : a % 4 = 3)
  (h4 : a % 5 = 3) :
  a % 12 = 7 := 
sorry

end button_remainders_l252_25281


namespace value_of_f_g6_minus_g_f6_l252_25255

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x + 4

theorem value_of_f_g6_minus_g_f6 : f (g 6) - g (f 6) = 48 :=
by
  sorry

end value_of_f_g6_minus_g_f6_l252_25255


namespace rectangle_area_l252_25264

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 := 
by 
  sorry

end rectangle_area_l252_25264


namespace minimum_value_abs_sum_l252_25219

theorem minimum_value_abs_sum (α β γ : ℝ) (h1 : α + β + γ = 2) (h2 : α * β * γ = 4) : 
  |α| + |β| + |γ| ≥ 6 :=
by
  sorry

end minimum_value_abs_sum_l252_25219


namespace sqrt_6_between_2_and_3_l252_25223

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 :=
by
  sorry

end sqrt_6_between_2_and_3_l252_25223


namespace degrees_to_radians_150_l252_25222

theorem degrees_to_radians_150 :
  (150 : ℝ) * (Real.pi / 180) = (5 * Real.pi) / 6 :=
by
  sorry

end degrees_to_radians_150_l252_25222


namespace fraction_equiv_l252_25252

theorem fraction_equiv (m n : ℚ) (h : m / n = 3 / 4) : (m + n) / n = 7 / 4 :=
sorry

end fraction_equiv_l252_25252


namespace gum_candy_ratio_l252_25293

theorem gum_candy_ratio
  (g c : ℝ)  -- let g be the cost of a stick of gum and c be the cost of a candy bar.
  (hc : c = 1.5)  -- the cost of each candy bar is $1.5
  (h_total_cost : 2 * g + 3 * c = 6)  -- total cost of 2 sticks of gum and 3 candy bars is $6
  : g / c = 1 / 2 := -- the ratio of the cost of gum to candy is 1:2
sorry

end gum_candy_ratio_l252_25293


namespace sum_of_v_values_is_zero_l252_25241

def v (x : ℝ) : ℝ := sorry

theorem sum_of_v_values_is_zero
  (h_odd : ∀ x : ℝ, v (-x) = -v x) :
  v (-3.14) + v (-1.57) + v (1.57) + v (3.14) = 0 :=
by
  sorry

end sum_of_v_values_is_zero_l252_25241


namespace four_digit_numbers_proof_l252_25290

noncomputable def four_digit_numbers_total : ℕ := 9000
noncomputable def two_digit_numbers_total : ℕ := 90
noncomputable def max_distinct_products : ℕ := 4095
noncomputable def cannot_be_expressed_as_product : ℕ := four_digit_numbers_total - max_distinct_products

theorem four_digit_numbers_proof :
  cannot_be_expressed_as_product = 4905 :=
by
  sorry

end four_digit_numbers_proof_l252_25290


namespace basketball_team_heights_l252_25224

theorem basketball_team_heights :
  ∃ (second tallest third fourth shortest : ℝ),
  (tallest = 80.5 ∧
   second = tallest - 6.25 ∧
   third = second - 3.75 ∧
   fourth = third - 5.5 ∧
   shortest = fourth - 4.8 ∧
   second = 74.25 ∧
   third = 70.5 ∧
   fourth = 65 ∧
   shortest = 60.2) := sorry

end basketball_team_heights_l252_25224


namespace determine_z_l252_25213

theorem determine_z (i z : ℂ) (hi : i^2 = -1) (h : i * z = 2 * z + 1) : 
  z = - (2/5 : ℂ) - (1/5 : ℂ) * i := by
  sorry

end determine_z_l252_25213


namespace sequence_general_formula_l252_25234

theorem sequence_general_formula (a : ℕ → ℕ)
    (h1 : a 1 = 1)
    (h2 : a 2 = 2)
    (h3 : ∀ n, a (n + 2) = a n + 2) :
    ∀ n, a n = n := by
  sorry

end sequence_general_formula_l252_25234


namespace remaining_thumbtacks_in_each_can_l252_25226

-- Definitions based on the conditions:
def total_thumbtacks : ℕ := 450
def num_cans : ℕ := 3
def thumbtacks_per_board_tested : ℕ := 1
def total_boards_tested : ℕ := 120

-- Lean 4 Statement

theorem remaining_thumbtacks_in_each_can :
  ∀ (initial_thumbtacks_per_can remaining_thumbtacks_per_can : ℕ),
  initial_thumbtacks_per_can = (total_thumbtacks / num_cans) →
  remaining_thumbtacks_per_can = (initial_thumbtacks_per_can - (thumbtacks_per_board_tested * total_boards_tested)) →
  remaining_thumbtacks_per_can = 30 :=
by
  sorry

end remaining_thumbtacks_in_each_can_l252_25226


namespace ratio_problem_l252_25236

theorem ratio_problem 
  (A B C : ℚ) 
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5 ∧ A / C = 3 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := 
by 
  sorry

end ratio_problem_l252_25236


namespace digital_earth_functionalities_l252_25258

def digital_earth_allows_internet_navigation : Prop := 
  ∀ (f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"]

def digital_earth_does_not_allow_physical_travel : Prop := 
  ¬ (∀ (f : String), f ∈ ["Travel around the world"])

theorem digital_earth_functionalities :
  digital_earth_allows_internet_navigation ∧ digital_earth_does_not_allow_physical_travel →
  ∀(f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"] :=
by
  sorry

end digital_earth_functionalities_l252_25258


namespace minimum_a_div_x_l252_25221

theorem minimum_a_div_x (a x y : ℕ) (h1 : 100 < a) (h2 : 100 < x) (h3 : 100 < y) (h4 : y^2 - 1 = a^2 * (x^2 - 1)) :
  2 ≤ a / x :=
by sorry

end minimum_a_div_x_l252_25221


namespace petya_finishes_earlier_than_masha_l252_25288

variable (t_P t_M t_K : ℕ)

-- Given conditions
def condition1 := t_K = 2 * t_P
def condition2 := t_P + 12 = t_K
def condition3 := t_M = 3 * t_P

-- The proof goal: Petya finishes 24 seconds earlier than Masha
theorem petya_finishes_earlier_than_masha
    (h1 : condition1 t_P t_K)
    (h2 : condition2 t_P t_K)
    (h3 : condition3 t_P t_M) :
    t_M - t_P = 24 := by
  sorry

end petya_finishes_earlier_than_masha_l252_25288


namespace spending_difference_l252_25249

def chocolate_price : ℝ := 7
def candy_bar_price : ℝ := 2
def discount_rate : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def gum_price : ℝ := 3

def discounted_chocolate_price : ℝ := chocolate_price * (1 - discount_rate)
def total_before_tax : ℝ := candy_bar_price + gum_price
def tax_amount : ℝ := total_before_tax * sales_tax_rate
def total_after_tax : ℝ := total_before_tax + tax_amount

theorem spending_difference : 
  discounted_chocolate_price - candy_bar_price = 3.95 :=
by 
  -- Apply the necessary calculations
  have discount_chocolate : ℝ := discounted_chocolate_price
  have candy_bar : ℝ := candy_bar_price
  calc
    discounted_chocolate_price - candy_bar_price = _ := sorry

end spending_difference_l252_25249


namespace fraction_from_condition_l252_25260

theorem fraction_from_condition (x f : ℝ) (h : 0.70 * x = f * x + 110) (hx : x = 300) : f = 1 / 3 :=
by
  sorry

end fraction_from_condition_l252_25260


namespace min_x_y_l252_25244

theorem min_x_y
  (x y : ℝ)
  (h1 : 0 < x)
  (h2 : 0 < y)
  (h3 : x + 2 * y + x * y - 7 = 0) :
  x + y ≥ 3 := by
  sorry

end min_x_y_l252_25244


namespace fraction_neg_range_l252_25240

theorem fraction_neg_range (x : ℝ) : (x ≠ 0 ∧ x < 1) ↔ (x - 1 < 0 ∧ x^2 > 0) := by
  sorry

end fraction_neg_range_l252_25240


namespace count_pairs_divisible_by_nine_l252_25256

open Nat

theorem count_pairs_divisible_by_nine (n : ℕ) (h : n = 528) :
  ∃ (count : ℕ), count = n ∧
  ∀ (a b : ℕ), 1 ≤ a ∧ a < b ∧ b ≤ 100 ∧ (a^2 + b^2 + a * b) % 9 = 0 ↔
  count = 528 :=
by
  sorry

end count_pairs_divisible_by_nine_l252_25256


namespace total_distance_proof_l252_25246

-- Define the conditions
def first_half_time := 20
def second_half_time := 30
def average_time_per_kilometer := 5

-- Calculate the total time
def total_time := first_half_time + second_half_time

-- State the proof problem: prove that the total distance is 10 kilometers
theorem total_distance_proof : 
  (total_time / average_time_per_kilometer) = 10 :=
  by sorry

end total_distance_proof_l252_25246


namespace find_larger_number_l252_25274

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 := by
  sorry

end find_larger_number_l252_25274


namespace set_equality_l252_25251

theorem set_equality : 
  { x : ℕ | ∃ k : ℕ, 6 - x = k ∧ 8 % k = 0 } = { 2, 4, 5 } :=
by
  sorry

end set_equality_l252_25251


namespace abs_neg_2023_l252_25200

theorem abs_neg_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_l252_25200


namespace fraction_order_l252_25205

theorem fraction_order :
  (21:ℚ) / 17 < (23:ℚ) / 18 ∧ (23:ℚ) / 18 < (25:ℚ) / 19 :=
by
  sorry

end fraction_order_l252_25205


namespace cos_double_angle_l252_25268

open Real

-- Define the given conditions
variables {θ : ℝ}
axiom θ_in_interval : 0 < θ ∧ θ < π / 2
axiom sin_minus_cos : sin θ - cos θ = sqrt 2 / 2

-- Create a theorem that reflects the proof problem
theorem cos_double_angle : cos (2 * θ) = - sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_l252_25268


namespace price_of_eraser_l252_25297

variables (x y : ℝ)

theorem price_of_eraser : 
  (3 * x + 5 * y = 10.6) ∧ (4 * x + 4 * y = 12) → x = 2.2 :=
by
  sorry

end price_of_eraser_l252_25297


namespace min_y_value_l252_25298

theorem min_y_value :
  ∃ c : ℝ, ∀ x : ℝ, (5 * x^2 + 20 * x + 25) >= c ∧ (∀ x : ℝ, (5 * x^2 + 20 * x + 25 = c) → x = -2) ∧ c = 5 :=
by
  sorry

end min_y_value_l252_25298


namespace determine_a_l252_25242

theorem determine_a (a x y : ℝ) (h : (a + 1) * x^(|a|) + y = -8) (h_linear : ∀ x y, (a + 1) * x^(|a|) + y = -8 → x ^ 1 = x): a = 1 :=
by 
  sorry

end determine_a_l252_25242


namespace tangent_line_b_value_l252_25253

theorem tangent_line_b_value (b : ℝ) : 
  (∃ pt : ℝ × ℝ, (pt.1)^2 + (pt.2)^2 = 25 ∧ pt.1 - pt.2 + b = 0)
  ↔ b = 5 * Real.sqrt 2 ∨ b = -5 * Real.sqrt 2 :=
by
  sorry

end tangent_line_b_value_l252_25253


namespace net_change_in_price_l252_25272

theorem net_change_in_price (P : ℝ) : 
  ((P * 0.75) * 1.2 = P * 0.9) → 
  ((P * 0.9 - P) / P = -0.1) :=
by
  intro h
  sorry

end net_change_in_price_l252_25272


namespace rahim_average_price_l252_25271

def books_shop1 : ℕ := 50
def cost_shop1 : ℕ := 1000
def books_shop2 : ℕ := 40
def cost_shop2 : ℕ := 800

def total_books : ℕ := books_shop1 + books_shop2
def total_cost : ℕ := cost_shop1 + cost_shop2
def average_price_per_book : ℕ := total_cost / total_books

theorem rahim_average_price :
  average_price_per_book = 20 := by
  sorry

end rahim_average_price_l252_25271


namespace minimum_time_needed_l252_25270

-- Define the task times
def review_time : ℕ := 30
def rest_time : ℕ := 30
def boil_water_time : ℕ := 15
def homework_time : ℕ := 25

-- Define the minimum time required (Xiao Ming can boil water while resting)
theorem minimum_time_needed : review_time + rest_time + homework_time = 85 := by
  -- The proof is omitted with sorry
  sorry

end minimum_time_needed_l252_25270


namespace vector_parallel_dot_product_l252_25248

theorem vector_parallel_dot_product (x : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (x, 1))
  (h2 : b = (4, 2))
  (h3 : x / 4 = 1 / 2) : 
  (a.1 * (b.1 - a.1) + a.2 * (b.2 - a.2)) = 5 := 
by 
  sorry

end vector_parallel_dot_product_l252_25248


namespace eval_expression_l252_25294

theorem eval_expression :
  2^0 + 9^5 / 9^3 = 82 :=
by
  have h1 : 2^0 = 1 := by sorry
  have h2 : 9^5 / 9^3 = 9^(5-3) := by sorry
  have h3 : 9^(5-3) = 9^2 := by sorry
  have h4 : 9^2 = 81 := by sorry
  sorry

end eval_expression_l252_25294


namespace inequality_holds_l252_25210

variable {a b c r : ℝ}
variable (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

/-- 
To prove that the inequality r (ab + bc + ca) + (3 - r) (1/a + 1/b + 1/c) ≥ 9 
is true for all r satisfying 0 < r < 3 and for arbitrary positive reals a, b, c. 
-/
theorem inequality_holds (h : 0 < r ∧ r < 3) : 
  r * (a * b + b * c + c * a) + (3 - r) * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end inequality_holds_l252_25210


namespace area_ratio_correct_l252_25235

noncomputable def ratio_area_MNO_XYZ (s t u : ℝ) (S_XYZ : ℝ) : ℝ := 
  let S_XMO := s * (1 - u) * S_XYZ
  let S_YNM := t * (1 - s) * S_XYZ
  let S_OZN := u * (1 - t) * S_XYZ
  S_XYZ - S_XMO - S_YNM - S_OZN

theorem area_ratio_correct (s t u : ℝ) (h1 : s + t + u = 3 / 4) 
  (h2 : s^2 + t^2 + u^2 = 3 / 8) : 
  ratio_area_MNO_XYZ s t u 1 = 13 / 32 := 
by
  -- Proof omitted
  sorry

end area_ratio_correct_l252_25235


namespace verify_condition_C_l252_25225

variable (x y z : ℤ)

-- Given conditions
def condition_C : Prop := x = y ∧ y = z + 1

-- The theorem/proof problem
theorem verify_condition_C (h : condition_C x y z) : (x - y)^2 + (y - z)^2 + (z - x)^2 = 2 := 
by 
  sorry

end verify_condition_C_l252_25225


namespace circle_and_line_properties_l252_25203

-- Define the circle C with center on the positive x-axis and passing through the origin
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l: y = kx + 2
def line_l (k x y : ℝ) : Prop := y = k * x + 2

-- Statement: the circle and line setup
theorem circle_and_line_properties (k : ℝ) : 
  ∀ (x y : ℝ), 
  circle_C x y → 
  ∃ (x1 y1 x2 y2 : ℝ), 
  line_l k x1 y1 ∧ 
  line_l k x2 y2 ∧ 
  circle_C x1 y1 ∧ 
  circle_C x2 y2 ∧ 
  (x1 ≠ x2 ∧ y1 ≠ y2) → 
  k < -3/4 ∧
  ( (y1 / x1) + (y2 / x2) = 1 ) :=
by
  sorry

end circle_and_line_properties_l252_25203


namespace expectation_is_four_thirds_l252_25206

-- Define the probability function
def P_ξ (k : ℕ) : ℚ :=
  if k = 0 then (1/2)^2 * (2/3)
  else if k = 1 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3)
  else if k = 2 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3) + (1/2) * (1/2) * (1/3)
  else if k = 3 then (1/2) * (1/2) * (1/3)
  else 0

-- Define the expected value function
def E_ξ : ℚ :=
  0 * P_ξ 0 + 1 * P_ξ 1 + 2 * P_ξ 2 + 3 * P_ξ 3

-- Formal statement of the problem
theorem expectation_is_four_thirds : E_ξ = 4 / 3 :=
  sorry

end expectation_is_four_thirds_l252_25206


namespace find_p_l252_25275

theorem find_p (p : ℕ) (hp : Nat.Prime p) (hp2 : Nat.Prime (5 * p^2 - 2)) : p = 3 :=
sorry

end find_p_l252_25275


namespace min_value_fraction_l252_25287

variable (x y : ℝ)

theorem min_value_fraction (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  ∃ (m : ℝ), (∀ z, (z = (1/x) + (9/y)) → z ≥ 16) ∧ ((1/x) + (9/y) = m) :=
sorry

end min_value_fraction_l252_25287


namespace Emily_total_points_l252_25238

-- Definitions of the points scored in each round
def round1_points := 16
def round2_points := 32
def round3_points := -27
def round4_points := 92
def round5_points := 4

-- Total points calculation in Lean
def total_points := round1_points + round2_points + round3_points + round4_points + round5_points

-- Lean statement to prove total points at the end of the game
theorem Emily_total_points : total_points = 117 :=
by 
  -- Unfold the definition of total_points and simplify
  unfold total_points round1_points round2_points round3_points round4_points round5_points
  -- Simplify the expression
  sorry

end Emily_total_points_l252_25238


namespace babblian_word_count_l252_25232

theorem babblian_word_count (n : ℕ) (h1 : n = 6) : ∃ m, m = 258 := by
  sorry

end babblian_word_count_l252_25232


namespace monica_cookies_left_l252_25292

theorem monica_cookies_left 
  (father_cookies : ℕ) 
  (mother_cookies : ℕ) 
  (brother_cookies : ℕ) 
  (sister_cookies : ℕ) 
  (aunt_cookies : ℕ) 
  (cousin_cookies : ℕ) 
  (total_cookies : ℕ)
  (father_cookies_eq : father_cookies = 12)
  (mother_cookies_eq : mother_cookies = father_cookies / 2)
  (brother_cookies_eq : brother_cookies = mother_cookies + 2)
  (sister_cookies_eq : sister_cookies = brother_cookies * 3)
  (aunt_cookies_eq : aunt_cookies = father_cookies * 2)
  (cousin_cookies_eq : cousin_cookies = aunt_cookies - 5)
  (total_cookies_eq : total_cookies = 120) : 
  total_cookies - (father_cookies + mother_cookies + brother_cookies + sister_cookies + aunt_cookies + cousin_cookies) = 27 :=
by
  sorry

end monica_cookies_left_l252_25292


namespace problem_solution_l252_25267

-- Definitions of the conditions as Lean statements:
def condition1 (t : ℝ) : Prop :=
  (1 + Real.sin t) * (1 - Real.cos t) = 1

def condition2 (t : ℝ) (a b c : ℕ) : Prop :=
  (1 - Real.sin t) * (1 + Real.cos t) = (a / b) - Real.sqrt c

def areRelativelyPrime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

-- The proof problem statement:
theorem problem_solution (t : ℝ) (a b c : ℕ) (h1 : condition1 t) (h2 : condition2 t a b c) (h3 : areRelativelyPrime a b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) : a + b + c = 2 := 
sorry

end problem_solution_l252_25267


namespace bush_height_l252_25266

theorem bush_height (h : ℕ → ℕ) (h0 : h 5 = 81) (h1 : ∀ n, h (n + 1) = 3 * h n) :
  h 2 = 3 := 
sorry

end bush_height_l252_25266


namespace number_of_planes_l252_25278

theorem number_of_planes (total_wings: ℕ) (wings_per_plane: ℕ) 
  (h1: total_wings = 50) (h2: wings_per_plane = 2) : 
  total_wings / wings_per_plane = 25 := by 
  sorry

end number_of_planes_l252_25278


namespace gcd_polynomial_multiple_of_532_l252_25233

theorem gcd_polynomial_multiple_of_532 (a : ℤ) (h : ∃ k : ℤ, a = 532 * k) :
  Int.gcd (5 * a ^ 3 + 2 * a ^ 2 + 6 * a + 76) a = 76 :=
by
  sorry

end gcd_polynomial_multiple_of_532_l252_25233


namespace find_x_plus_3y_l252_25276

variables {α : Type*} {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (x y : ℝ)
variables (OA OB OC OD OE : V)

-- Defining the conditions
def condition1 := OA = (1/2) • OB + x • OC + y • OD
def condition2 := OB = 2 • x • OC + (1/3) • OD + y • OE

-- Writing the theorem statement
theorem find_x_plus_3y (h1 : condition1 x y OA OB OC OD) (h2 : condition2 x y OB OC OD OE) : 
  x + 3 * y = 7 / 6 := 
sorry

end find_x_plus_3y_l252_25276


namespace cube_divisibility_l252_25250

theorem cube_divisibility (a : ℤ) (k : ℤ) (h₁ : a > 1) 
(h₂ : (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 4 ∣ a := 
by
  sorry

end cube_divisibility_l252_25250


namespace velma_more_than_veronica_l252_25247

-- Defining the distances each flashlight can be seen
def veronica_distance : ℕ := 1000
def freddie_distance : ℕ := 3 * veronica_distance
def velma_distance : ℕ := 5 * freddie_distance - 2000

-- The proof problem: Prove that Velma's flashlight can be seen 12000 feet farther than Veronica's flashlight.
theorem velma_more_than_veronica : velma_distance - veronica_distance = 12000 := by
  sorry

end velma_more_than_veronica_l252_25247


namespace c_geq_one_l252_25211

theorem c_geq_one {a b : ℕ} {c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : (a + 1) / (b + c) = b / a) : 1 ≤ c :=
  sorry

end c_geq_one_l252_25211


namespace sales_overlap_l252_25262

-- Define the conditions
def bookstore_sale_days : List ℕ := [2, 6, 10, 14, 18, 22, 26, 30]
def shoe_store_sale_days : List ℕ := [1, 8, 15, 22, 29]

-- Define the statement to prove
theorem sales_overlap : (bookstore_sale_days ∩ shoe_store_sale_days).length = 1 := 
by
  sorry

end sales_overlap_l252_25262


namespace arithmetic_geom_sequence_a2_l252_25285

theorem arithmetic_geom_sequence_a2 :
  ∀ (a : ℕ → ℤ), 
    (∀ n, a (n+1) = a n + 2) →  -- Arithmetic sequence with common difference of 2
    a 1 * a 4 = a 3 ^ 2 →  -- Geometric sequence property for a_1, a_3, a_4
    a 2 = -6 :=             -- The value of a_2
by
  intros a h_arith h_geom
  sorry

end arithmetic_geom_sequence_a2_l252_25285


namespace mass_of_15_moles_is_9996_9_l252_25231

/-- Calculation of the molar mass of potassium aluminum sulfate dodecahydrate -/
def KAl_SO4_2_12H2O_molar_mass : ℝ :=
  let K := 39.10
  let Al := 26.98
  let S := 32.07
  let O := 16.00
  let H := 1.01
  K + Al + 2 * S + (8 + 24) * O + 24 * H

/-- Mass calculation for 15 moles of potassium aluminum sulfate dodecahydrate -/
def mass_of_15_moles_KAl_SO4_2_12H2O : ℝ :=
  15 * KAl_SO4_2_12H2O_molar_mass

/-- Proof statement that the mass of 15 moles of potassium aluminum sulfate dodecahydrate is 9996.9 grams -/
theorem mass_of_15_moles_is_9996_9 : mass_of_15_moles_KAl_SO4_2_12H2O = 9996.9 := by
  -- assume KAl_SO4_2_12H2O_molar_mass = 666.46 (from the problem solution steps)
  sorry

end mass_of_15_moles_is_9996_9_l252_25231


namespace total_turnips_l252_25296

-- Conditions
def turnips_keith : ℕ := 6
def turnips_alyssa : ℕ := 9

-- Statement to be proved
theorem total_turnips : turnips_keith + turnips_alyssa = 15 := by
  -- Proof is not required for this prompt, so we use sorry
  sorry

end total_turnips_l252_25296


namespace horse_goat_sheep_consumption_l252_25284

theorem horse_goat_sheep_consumption :
  (1 / (1 / (1 : ℝ) + 1 / 2 + 1 / 3)) = 6 / 11 :=
by
  sorry

end horse_goat_sheep_consumption_l252_25284


namespace construction_work_rate_l252_25207

theorem construction_work_rate (C : ℝ) 
  (h1 : ∀ t1 : ℝ, t1 = 10 → t1 * 8 = 80)
  (h2 : ∀ t2 : ℝ, t2 = 15 → t2 * C + 80 ≥ 300)
  (h3 : ∀ t : ℝ, t = 25 → ∀ t1 t2 : ℝ, t = t1 + t2 → t1 = 10 → t2 = 15)
  : C = 14.67 :=
by
  sorry

end construction_work_rate_l252_25207


namespace max_abs_z_l252_25202

open Complex

theorem max_abs_z (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) : abs z ≤ 1 :=
sorry

end max_abs_z_l252_25202


namespace fraction_of_peaches_l252_25289

-- Define the number of peaches each person has
def Benjy_peaches : ℕ := 5
def Martine_peaches : ℕ := 16
def Gabrielle_peaches : ℕ := 15

-- Condition that Martine has 6 more than twice Benjy's peaches
def Martine_cond : Prop := Martine_peaches = 2 * Benjy_peaches + 6

-- The goal is to prove the fraction of Gabrielle's peaches that Benjy has
theorem fraction_of_peaches :
  Martine_cond → (Benjy_peaches : ℚ) / (Gabrielle_peaches : ℚ) = 1 / 3 :=
by
  -- Assuming the condition holds
  intro h
  rw [Martine_cond] at h
  -- Use the condition directly, since Martine_cond implies Benjy_peaches = 5
  exact sorry

end fraction_of_peaches_l252_25289


namespace solution_l252_25254

theorem solution (A B C : ℚ) (h1 : A + B = 10) (h2 : 2 * A = 3 * B + 5) (h3 : A * B * C = 120) :
  A = 7 ∧ B = 3 ∧ C = 40 / 7 := by
  sorry

end solution_l252_25254


namespace sum_of_fourth_powers_l252_25230

theorem sum_of_fourth_powers (n : ℤ) (h1 : n > 0) (h2 : (n - 1)^2 + n^2 + (n + 1)^2 = 9458) :
  (n - 1)^4 + n^4 + (n + 1)^4 = 30212622 :=
by sorry

end sum_of_fourth_powers_l252_25230


namespace omega_in_abc_l252_25220

variables {R : Type*}
variables [LinearOrderedField R]
variables {a b c ω x y z : R} 

theorem omega_in_abc 
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ω ≠ a ∧ ω ≠ b ∧ ω ≠ c)
  (h1 : x + y + z = 1)
  (h2 : a^2 * x + b^2 * y + c^2 * z = ω^2)
  (h3 : a^3 * x + b^3 * y + c^3 * z = ω^3)
  (h4 : a^4 * x + b^4 * y + c^4 * z = ω^4):
  ω = a ∨ ω = b ∨ ω = c :=
sorry

end omega_in_abc_l252_25220


namespace intersection_unique_element_l252_25282

noncomputable def A := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
noncomputable def B (r : ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem intersection_unique_element (r : ℝ) (hr : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ A ∧ p ∈ B r) → (r = 3 ∨ r = 7) :=
sorry

end intersection_unique_element_l252_25282


namespace necessary_but_not_sufficient_condition_l252_25257

theorem necessary_but_not_sufficient_condition (x : ℝ) : (|x - 1| < 1 → x^2 - 5 * x < 0) ∧ (¬(x^2 - 5 * x < 0 → |x - 1| < 1)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l252_25257


namespace brendan_cuts_84_yards_in_week_with_lawnmower_l252_25227

-- Brendan cuts 8 yards per day
def yards_per_day : ℕ := 8

-- The lawnmower increases his efficiency by fifty percent
def efficiency_increase (yards : ℕ) : ℕ :=
  yards + (yards / 2)

-- Calculate total yards cut in 7 days with the lawnmower
def total_yards_in_week (days : ℕ) (daily_yards : ℕ) : ℕ :=
  days * daily_yards

-- Prove the total yards cut in 7 days with the lawnmower is 84
theorem brendan_cuts_84_yards_in_week_with_lawnmower :
  total_yards_in_week 7 (efficiency_increase yards_per_day) = 84 :=
by
  sorry

end brendan_cuts_84_yards_in_week_with_lawnmower_l252_25227


namespace max_height_reached_l252_25265

def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_reached : ∃ t : ℝ, h t = 161 :=
by
  sorry

end max_height_reached_l252_25265


namespace range_of_m_l252_25209

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) :=
by
  sorry

end range_of_m_l252_25209
