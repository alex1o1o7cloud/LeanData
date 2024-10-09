import Mathlib

namespace sufficient_but_not_necessary_l1011_101142

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 2) : (1/x < 1/2 ∧ (∃ y : ℝ, 1/y < 1/2 ∧ y ≤ 2)) :=
by { sorry }

end sufficient_but_not_necessary_l1011_101142


namespace second_month_sale_l1011_101135

theorem second_month_sale (S : ℝ) :
  (S + 5420 + 6200 + 6350 + 6500 = 30000) → S = 5530 :=
by
  sorry

end second_month_sale_l1011_101135


namespace correct_samples_for_senior_l1011_101180

-- Define the total number of students in each section
def junior_students : ℕ := 400
def senior_students : ℕ := 200
def total_students : ℕ := junior_students + senior_students

-- Define the total number of samples to be drawn
def total_samples : ℕ := 60

-- Calculate the number of samples to be drawn from each section
def junior_samples : ℕ := total_samples * junior_students / total_students
def senior_samples : ℕ := total_samples - junior_samples

-- The theorem to prove
theorem correct_samples_for_senior :
  senior_samples = 20 :=
by
  sorry

end correct_samples_for_senior_l1011_101180


namespace part1_sales_volume_part2_price_reduction_l1011_101193

noncomputable def daily_sales_volume (x : ℝ) : ℝ :=
  100 + 200 * x

noncomputable def profit_eq (x : ℝ) : Prop :=
  (4 - 2 - x) * (100 + 200 * x) = 300

theorem part1_sales_volume (x : ℝ) : daily_sales_volume x = 100 + 200 * x :=
sorry

theorem part2_price_reduction (hx : profit_eq (1 / 2)) : 1 / 2 = 1 / 2 :=
sorry

end part1_sales_volume_part2_price_reduction_l1011_101193


namespace solve_system_equations_l1011_101105

noncomputable def system_equations : Prop :=
  ∃ x y : ℝ,
    (8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0) ∧
    (8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0) ∧
    ((x = 0 ∧ y = 4) ∨ (x = -7.5 ∧ y = 1) ∨ (x = -4.5 ∧ y = 0))

theorem solve_system_equations : system_equations := 
by
  sorry

end solve_system_equations_l1011_101105


namespace inequality_one_inequality_two_l1011_101126

-- First Inequality Problem
theorem inequality_one (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) + (1 / d^2) ≤ (1 / (a^2 * b^2 * c^2 * d^2)) :=
sorry

-- Second Inequality Problem
theorem inequality_two (a b c d : ℝ) (h : a + b + c + d = 3) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) + (1 / d^3) ≤ (1 / (a^3 * b^3 * c^3 * d^3)) :=
sorry

end inequality_one_inequality_two_l1011_101126


namespace option_A_option_C_l1011_101149

variable {a : ℕ → ℝ} (q : ℝ)
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n, a (n + 1) = q * (a n)

def decreasing_sequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a n > a (n + 1)

theorem option_A (h₁ : a 1 > 0) (hq : geometric_sequence a q) : 0 < q ∧ q < 1 → decreasing_sequence a := 
  sorry

theorem option_C (h₁ : a 1 < 0) (hq : geometric_sequence a q) : q > 1 → decreasing_sequence a := 
  sorry

end option_A_option_C_l1011_101149


namespace sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l1011_101110

theorem sqrt_12_eq_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := sorry

theorem sqrt_1_div_2_eq_sqrt_2_div_2 : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 := sorry

end sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l1011_101110


namespace taimour_paints_fence_alone_in_15_hours_l1011_101174

theorem taimour_paints_fence_alone_in_15_hours :
  ∀ (T : ℝ), (∀ (J : ℝ), J = T / 2 → (1 / J + 1 / T = 1 / 5)) → T = 15 :=
by
  intros T h
  have h1 := h (T / 2) rfl
  sorry

end taimour_paints_fence_alone_in_15_hours_l1011_101174


namespace angle_SR_XY_is_70_l1011_101172

-- Define the problem conditions
variables (X Y Z V H S R : Type) 
variables (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ)

-- Set the conditions
def triangleXYZ (X Y Z V H S R : Type) (angleX angleY angleZ angleSRXY : ℝ) (XY XV YH : ℝ) : Prop :=
  angleX = 40 ∧ angleY = 70 ∧ XY = 12 ∧ XV = 2 ∧ YH = 2 ∧
  ∃ S R, S = (XY / 2) ∧ R = ((XV + YH) / 2)

-- Construct the theorem to be proven
theorem angle_SR_XY_is_70 {X Y Z V H S R : Type} 
  {angleX angleY angleZ angleSRXY : ℝ} 
  {XY XV YH : ℝ} : 
  triangleXYZ X Y Z V H S R angleX angleY angleZ angleSRXY XY XV YH →
  angleSRXY = 70 :=
by
  -- Placeholder proof steps
  sorry

end angle_SR_XY_is_70_l1011_101172


namespace hexagon_theorem_l1011_101129

-- Define a structure for the hexagon with its sides
structure Hexagon :=
(side1 side2 side3 side4 side5 side6 : ℕ)

-- Define the conditions of the problem
def hexagon_conditions (h : Hexagon) : Prop :=
  h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧
  (h.side1 + h.side2 + h.side3 + h.side4 + h.side5 + h.side6 = 38)

-- Define the proposition that we need to prove
def hexagon_proposition (h : Hexagon) : Prop :=
  (h.side3 = 7 ∨ h.side4 = 7 ∨ h.side5 = 7 ∨ h.side6 = 7) → 
  (h.side1 = 5 ∧ h.side2 = 6 ∧ h.side3 = 7 ∧ h.side4 = 7 ∧ h.side5 = 7 ∧ h.side6 = 7 → 3 = 3)

-- The proof statement combining conditions and the to-be-proven proposition
theorem hexagon_theorem (h : Hexagon) (hc : hexagon_conditions h) : hexagon_proposition h :=
by
  sorry -- No proof is required

end hexagon_theorem_l1011_101129


namespace grilled_cheese_sandwiches_l1011_101148

theorem grilled_cheese_sandwiches (h g : ℕ) (c_ham c_grilled total_cheese : ℕ)
  (h_count : h = 10)
  (ham_cheese : c_ham = 2)
  (grilled_cheese : c_grilled = 3)
  (cheese_used : total_cheese = 50)
  (sandwich_eq : total_cheese = h * c_ham + g * c_grilled) :
  g = 10 :=
by
  sorry

end grilled_cheese_sandwiches_l1011_101148


namespace abs_ineq_solution_set_l1011_101130

theorem abs_ineq_solution_set (x : ℝ) : |x + 1| - |x - 5| < 4 ↔ x < 4 :=
sorry

end abs_ineq_solution_set_l1011_101130


namespace box_dimensions_correct_l1011_101152

theorem box_dimensions_correct (L W H : ℕ) (L_eq : L = 22) (W_eq : W = 22) (H_eq : H = 11) : 
  let method1 := 2 * L + 2 * W + 4 * H + 24
  let method2 := 2 * L + 4 * W + 2 * H + 24
  method2 - method1 = 22 :=
by
  sorry

end box_dimensions_correct_l1011_101152


namespace integer_solutions_b_l1011_101189

theorem integer_solutions_b (b : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ ∀ x : ℤ, x1 ≤ x ∧ x ≤ x2 → x^2 + b * x + 3 ≤ 0) ↔ b = -4 ∨ b = 4 := 
sorry

end integer_solutions_b_l1011_101189


namespace power_function_not_origin_l1011_101132

theorem power_function_not_origin (m : ℝ) 
  (h1 : m^2 - 3 * m + 3 = 1) 
  (h2 : m^2 - m - 2 ≤ 0) : 
  m = 1 ∨ m = 2 :=
sorry

end power_function_not_origin_l1011_101132


namespace Eliane_schedule_combinations_l1011_101140

def valid_schedule_combinations : ℕ :=
  let mornings := 6 * 3 -- 6 days (Monday to Saturday) each with 3 time slots
  let afternoons := 5 * 2 -- 5 days (Monday to Friday) each with 2 time slots
  let mon_or_fri_comb := 2 * 3 * 3 * 2 -- Morning on Monday or Friday
  let sat_comb := 1 * 3 * 4 * 2 -- Morning on Saturday
  let tue_wed_thu_comb := 3 * 3 * 2 * 2 -- Morning on Tuesday, Wednesday, or Thursday
  mon_or_fri_comb + sat_comb + tue_wed_thu_comb

theorem Eliane_schedule_combinations :
  valid_schedule_combinations = 96 := by
  sorry

end Eliane_schedule_combinations_l1011_101140


namespace shirt_wallet_ratio_l1011_101103

theorem shirt_wallet_ratio
  (F W S : ℕ)
  (hF : F = 30)
  (hW : W = F + 60)
  (h_total : S + W + F = 150) :
  S / W = 1 / 3 := by
  sorry

end shirt_wallet_ratio_l1011_101103


namespace seventh_number_fifth_row_l1011_101138

theorem seventh_number_fifth_row : 
  ∀ (n : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ → ℕ), 
  (∀ i, 1 <= i ∧ i <= n  → b 1 i = 2 * i - 1) →
  (∀ j i, 2 <= j ∧ 1 <= i ∧ i <= n - (j-1)  → b j i = b (j-1) i + b (j-1) (i+1)) →
  (b : ℕ → ℕ → ℕ) →
  b 5 7 = 272 :=
by {
  sorry
}

end seventh_number_fifth_row_l1011_101138


namespace simplify_expression_l1011_101158

theorem simplify_expression (x y : ℝ) :
  4 * x + 8 * x^2 + y^3 + 6 - (3 - 4 * x - 8 * x^2 - y^3) =
  16 * x^2 + 8 * x + 2 * y^3 + 3 :=
by sorry

end simplify_expression_l1011_101158


namespace sum_of_variables_is_16_l1011_101181

theorem sum_of_variables_is_16 (A B C D E : ℕ)
    (h1 : C + E = 4) 
    (h2 : B + E = 7) 
    (h3 : B + D = 6) 
    (h4 : A = 6)
    (hdistinct : ∀ x y, x ≠ y → (x ≠ A ∧ x ≠ B ∧ x ≠ C ∧ x ≠ D ∧ x ≠ E) ∧ (y ≠ A ∧ y ≠ B ∧ y ≠ C ∧ y ≠ D ∧ y ≠ E)) :
    A + B + C + D + E = 16 :=
by
    sorry

end sum_of_variables_is_16_l1011_101181


namespace problem_equivalent_l1011_101146

theorem problem_equivalent : ∀ m : ℝ, 2 * m^2 + m = -1 → 4 * m^2 + 2 * m + 5 = 3 := 
by
  intros m h
  sorry

end problem_equivalent_l1011_101146


namespace mike_passing_percentage_l1011_101119

theorem mike_passing_percentage :
  ∀ (score shortfall max_marks : ℕ), 
    score = 212 ∧ shortfall = 25 ∧ max_marks = 790 →
    (score + shortfall) / max_marks * 100 = 30 :=
by
  intros score shortfall max_marks h
  have h1 : score = 212 := h.1
  have h2 : shortfall = 25 := h.2.1
  have h3 : max_marks = 790 := h.2.2
  rw [h1, h2, h3]
  sorry

end mike_passing_percentage_l1011_101119


namespace problem_statement_l1011_101136

theorem problem_statement (x y : ℝ) (h : x - 2 * y = -2) : 3 + 2 * x - 4 * y = -1 :=
  sorry

end problem_statement_l1011_101136


namespace last_digit_one_over_three_pow_neg_ten_l1011_101127

theorem last_digit_one_over_three_pow_neg_ten : (3^10) % 10 = 9 := by
  sorry

end last_digit_one_over_three_pow_neg_ten_l1011_101127


namespace sequence_value_proof_l1011_101167

theorem sequence_value_proof : 
  (∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n : ℕ, a (2 * n) = 2 * n * a n) ∧ 
    a (2^50) = 2^1276) :=
sorry

end sequence_value_proof_l1011_101167


namespace least_y_l1011_101196

theorem least_y (y : ℝ) : (2 * y ^ 2 + 7 * y + 3 = 5) → y = -2 :=
sorry

end least_y_l1011_101196


namespace greatest_whole_number_satisfying_inequality_l1011_101185

theorem greatest_whole_number_satisfying_inequality :
  ∃ x : ℕ, (∀ y : ℕ, y < 1 → y ≤ x) ∧ 4 * x - 3 < 2 - x :=
sorry

end greatest_whole_number_satisfying_inequality_l1011_101185


namespace tv_sale_increase_l1011_101143

theorem tv_sale_increase (P Q : ℝ) :
  let new_price := 0.9 * P
  let original_sale_value := P * Q
  let increased_percentage := 1.665
  ∃ x : ℝ, (new_price * (1 + x / 100) * Q = increased_percentage * original_sale_value) → x = 85 :=
by
  sorry

end tv_sale_increase_l1011_101143


namespace perimeter_of_square_field_l1011_101195

variable (s a p : ℕ)

-- Given conditions as definitions
def area_eq_side_squared (a s : ℕ) : Prop := a = s^2
def perimeter_eq_four_sides (p s : ℕ) : Prop := p = 4 * s
def given_equation (a p : ℕ) : Prop := 6 * a = 6 * (2 * p + 9)

-- The proof statement
theorem perimeter_of_square_field (s a p : ℕ) 
  (h1 : area_eq_side_squared a s)
  (h2 : perimeter_eq_four_sides p s)
  (h3 : given_equation a p) :
  p = 36 :=
by
  sorry

end perimeter_of_square_field_l1011_101195


namespace car_speed_l1011_101147

def travel_time : ℝ := 5
def travel_distance : ℝ := 300

theorem car_speed :
  travel_distance / travel_time = 60 := sorry

end car_speed_l1011_101147


namespace parallelogram_area_l1011_101184

theorem parallelogram_area (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  let y_top := a
  let y_bottom := -b
  let x_left := -c + 2*y
  let x_right := d - 2*y 
  (d + c) * (a + b) = ad + ac + bd + bc :=
by
  sorry

end parallelogram_area_l1011_101184


namespace simplify_and_evaluate_expression_l1011_101188

variable (x : ℝ) (h : x = Real.sqrt 2 - 1)

theorem simplify_and_evaluate_expression : 
  (1 - 1 / (x + 1)) / (x / (x^2 + 2 * x + 1)) = Real.sqrt 2 :=
by
  -- Using the given definition of x
  have hx : x = Real.sqrt 2 - 1 := h
  
  -- Required proof should go here 
  sorry

end simplify_and_evaluate_expression_l1011_101188


namespace total_books_l1011_101199

-- Define the given conditions
def books_per_shelf : ℕ := 8
def mystery_shelves : ℕ := 12
def picture_shelves : ℕ := 9

-- Define the number of books on each type of shelves
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := picture_shelves * books_per_shelf

-- Define the statement to prove
theorem total_books : total_mystery_books + total_picture_books = 168 := by
  sorry

end total_books_l1011_101199


namespace real_roots_range_of_m_l1011_101116

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem real_roots_range_of_m :
  (∃ x : ℝ, x^2 + 4 * m * x + 4 * m^2 + 2 * m + 3 = 0) ∨ 
  (∃ x : ℝ, x^2 + (2 * m + 1) * x + m^2 = 0) ↔ 
  m ≤ -3 / 2 ∨ m ≥ -1 / 4 :=
by
  sorry

end real_roots_range_of_m_l1011_101116


namespace problem1_problem2_l1011_101177

variable (x : ℝ)

-- Statement for the first problem
theorem problem1 : (-1 + 3 * x) * (-3 * x - 1) = 1 - 9 * x^2 := 
by
  sorry

-- Statement for the second problem
theorem problem2 : (x + 1)^2 - (1 - 3 * x) * (1 + 3 * x) = 10 * x^2 + 2 * x := 
by
  sorry

end problem1_problem2_l1011_101177


namespace area_of_triangle_l1011_101175

theorem area_of_triangle (a b : ℝ) (h1 : a^2 = 25) (h2 : b^2 = 144) : 
  1/2 * a * b = 30 :=
by sorry

end area_of_triangle_l1011_101175


namespace seven_digit_number_subtraction_l1011_101109

theorem seven_digit_number_subtraction 
  (n : ℕ)
  (d1 d2 d3 d4 d5 d6 d7 : ℕ)
  (h1 : n = d1 * 10^6 + d2 * 10^5 + d3 * 10^4 + d4 * 10^3 + d5 * 10^2 + d6 * 10 + d7)
  (h2 : d1 < 10 ∧ d2 < 10 ∧ d3 < 10 ∧ d4 < 10 ∧ d5 < 10 ∧ d6 < 10 ∧ d7 < 10)
  (h3 : n - (d1 + d3 + d4 + d5 + d6 + d7) = 9875352) :
  n - (d1 + d3 + d4 + d5 + d6 + d7 - d2) = 9875357 :=
sorry

end seven_digit_number_subtraction_l1011_101109


namespace inequality_not_always_hold_l1011_101151

variables {a b c d : ℝ}

theorem inequality_not_always_hold 
  (h1 : a > b) 
  (h2 : c > d) 
: ¬ (a + d > b + c) :=
  sorry

end inequality_not_always_hold_l1011_101151


namespace three_distinct_divisors_l1011_101166

theorem three_distinct_divisors (M : ℕ) : (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ∣ M ∧ b ∣ M ∧ c ∣ M ∧ (∀ d, d ≠ a ∧ d ≠ b ∧ d ≠ c → ¬ d ∣ M)) ↔ (∃ p : ℕ, Prime p ∧ M = p^2) := 
by sorry

end three_distinct_divisors_l1011_101166


namespace find_length_of_first_dimension_of_tank_l1011_101156

theorem find_length_of_first_dimension_of_tank 
    (w : ℝ) (h : ℝ) (cost_per_sq_ft : ℝ) (total_cost : ℝ) (l : ℝ) :
    w = 5 → h = 3 → cost_per_sq_ft = 20 → total_cost = 1880 → 
    1880 = (2 * l * w + 2 * l * h + 2 * w * h) * cost_per_sq_ft →
    l = 4 := 
by
  intros hw hh hcost htotal heq
  sorry

end find_length_of_first_dimension_of_tank_l1011_101156


namespace total_gift_money_l1011_101157

-- Definitions based on the conditions given in the problem
def initialAmount : ℕ := 159
def giftFromGrandmother : ℕ := 25
def giftFromAuntAndUncle : ℕ := 20
def giftFromParents : ℕ := 75

-- Lean statement to prove the total amount of money Chris has after receiving his birthday gifts
theorem total_gift_money : 
    initialAmount + giftFromGrandmother + giftFromAuntAndUncle + giftFromParents = 279 := by
sorry

end total_gift_money_l1011_101157


namespace electricity_fee_l1011_101102

theorem electricity_fee (a b : ℝ) : 
  let base_usage := 100
  let additional_usage := 160 - base_usage
  let base_cost := base_usage * a
  let additional_cost := additional_usage * b
  base_cost + additional_cost = 100 * a + 60 * b :=
by
  sorry

end electricity_fee_l1011_101102


namespace greatest_possible_perimeter_l1011_101100

def triangle_side_lengths (x : ℤ) : Prop :=
  (x > 0) ∧ (5 * x > 18) ∧ (x < 6)

def perimeter (x : ℤ) : ℤ :=
  x + 4 * x + 18

theorem greatest_possible_perimeter :
  ∃ x : ℤ, triangle_side_lengths x ∧ (perimeter x = 38) :=
by
  sorry

end greatest_possible_perimeter_l1011_101100


namespace candidate_majority_votes_l1011_101108

theorem candidate_majority_votes (total_votes : ℕ) (candidate_percentage other_percentage : ℕ) 
  (h_total_votes : total_votes = 5200)
  (h_candidate_percentage : candidate_percentage = 60)
  (h_other_percentage : other_percentage = 40) :
  (candidate_percentage * total_votes / 100) - (other_percentage * total_votes / 100) = 1040 := 
by
  sorry

end candidate_majority_votes_l1011_101108


namespace cone_lateral_area_l1011_101150

noncomputable def lateral_area_of_cone (θ : ℝ) (r_base : ℝ) : ℝ :=
  if θ = 120 ∧ r_base = 2 then 
    12 * Real.pi 
  else 
    0 -- default case for the sake of definition, not used in our proof

theorem cone_lateral_area :
  lateral_area_of_cone 120 2 = 12 * Real.pi :=
by
  -- This is where the proof would go
  sorry

end cone_lateral_area_l1011_101150


namespace hexagon_area_l1011_101178

noncomputable def area_of_hexagon (P Q R P' Q' R' : Point) (radius : ℝ) : ℝ :=
  -- a placeholder for the actual area calculation
  sorry 

theorem hexagon_area (P Q R P' Q' R' : Point) 
  (radius : ℝ) (perimeter : ℝ) :
  radius = 9 → perimeter = 42 →
  area_of_hexagon P Q R P' Q' R' radius = 189 := by
  intros h1 h2
  sorry

end hexagon_area_l1011_101178


namespace sum_of_sines_leq_3_sqrt3_over_2_l1011_101173

theorem sum_of_sines_leq_3_sqrt3_over_2 (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Real.sin α + Real.sin β + Real.sin γ ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sum_of_sines_leq_3_sqrt3_over_2_l1011_101173


namespace number_of_terms_in_arithmetic_sequence_l1011_101107

-- Define the first term, common difference, and the nth term of the sequence
def a : ℤ := -3
def d : ℤ := 4
def a_n : ℤ := 45

-- Define the number of terms in the arithmetic sequence
def num_of_terms : ℤ := 13

-- The theorem states that for the given arithmetic sequence, the number of terms n satisfies the sequence equation
theorem number_of_terms_in_arithmetic_sequence :
  a + (num_of_terms - 1) * d = a_n :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l1011_101107


namespace inequality_xyz_l1011_101153

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz / (x^3 + y^3 + xyz) + xyz / (y^3 + z^3 + xyz) + xyz / (z^3 + x^3 + xyz) ≤ 1) := by
  sorry

end inequality_xyz_l1011_101153


namespace lasso_success_probability_l1011_101128

-- Let p be the probability of successfully placing a lasso in a single throw
def p := 1 / 2

-- Let q be the probability of failure in a single throw
def q := 1 - p

-- Let n be the number of attempts
def n := 4

-- The probability of failing all n times
def probFailAll := q ^ n

-- The probability of succeeding at least once
def probSuccessAtLeastOnce := 1 - probFailAll

-- Theorem statement
theorem lasso_success_probability : probSuccessAtLeastOnce = 15 / 16 := by
  sorry

end lasso_success_probability_l1011_101128


namespace inequality_a3_minus_b3_l1011_101124

theorem inequality_a3_minus_b3 (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a^3 - b^3 < 0 :=
by sorry

end inequality_a3_minus_b3_l1011_101124


namespace factor_expression_l1011_101176

theorem factor_expression (x : ℝ) : 9 * x^2 + 3 * x = 3 * x * (3 * x + 1) := 
by
  sorry

end factor_expression_l1011_101176


namespace simplify_fraction_l1011_101192

theorem simplify_fraction (x y : ℝ) (h : x ≠ y) : 
  (x / (x - y) - y / (x + y)) = (x^2 + y^2) / (x^2 - y^2) :=
sorry

end simplify_fraction_l1011_101192


namespace second_day_speed_faster_l1011_101139

def first_day_distance := 18
def first_day_speed := 3
def first_day_time := first_day_distance / first_day_speed
def second_day_time := first_day_time - 1
def third_day_speed := 5
def third_day_time := 3
def third_day_distance := third_day_speed * third_day_time
def total_distance := 53

theorem second_day_speed_faster :
  ∃ r2, (first_day_distance + (second_day_time * r2) + third_day_distance = total_distance) → (r2 - first_day_speed = 1) :=
by
  sorry

end second_day_speed_faster_l1011_101139


namespace machines_complete_job_in_12_days_l1011_101114

-- Given the conditions
variable (D : ℕ) -- The number of days for 12 machines to complete the job
variable (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8))

-- Prove the number of days for 12 machines to complete the job
theorem machines_complete_job_in_12_days (h1 : (1 : ℚ) / ((12 : ℚ) * D) = (1 : ℚ) / ((18 : ℚ) * 8)) : D = 12 :=
by
  sorry

end machines_complete_job_in_12_days_l1011_101114


namespace biquadratic_exactly_two_distinct_roots_l1011_101154

theorem biquadratic_exactly_two_distinct_roots {a : ℝ} :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^4 + a*x1^2 + a - 1 = 0) ∧ (x2^4 + a*x2^2 + a - 1 = 0) ∧
   ∀ x, x^4 + a*x^2 + a - 1 = 0 → (x = x1 ∨ x = x2)) ↔ a < 1 :=
by
  sorry

end biquadratic_exactly_two_distinct_roots_l1011_101154


namespace roof_area_l1011_101145

-- Definitions of the roof's dimensions based on the given conditions.
def length (w : ℝ) := 4 * w
def width (w : ℝ) := w
def difference (l w : ℝ) := l - w
def area (l w : ℝ) := l * w

-- The proof problem: Given the conditions, prove the area is 576 square feet.
theorem roof_area : ∀ w : ℝ, (length w) - (width w) = 36 → area (length w) (width w) = 576 := by
  intro w
  intro h_diff
  sorry

end roof_area_l1011_101145


namespace madhav_rank_from_last_is_15_l1011_101117

-- Defining the conditions
def class_size : ℕ := 31
def madhav_rank_from_start : ℕ := 17

-- Statement to be proved
theorem madhav_rank_from_last_is_15 :
  (class_size - madhav_rank_from_start + 1) = 15 := by
  sorry

end madhav_rank_from_last_is_15_l1011_101117


namespace power_sum_greater_than_linear_l1011_101121

theorem power_sum_greater_than_linear (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0) (hn : n ≥ 2) :
  (1 + x) ^ n > 1 + n * x :=
sorry

end power_sum_greater_than_linear_l1011_101121


namespace range_of_m_l1011_101194

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  ∀ m, m = 9 / 4 → (1 / x + 4 / y) ≥ m := 
by
  sorry

end range_of_m_l1011_101194


namespace least_possible_product_of_primes_l1011_101125

-- Define a prime predicate for a number greater than 20
def is_prime_over_20 (p : Nat) : Prop := Nat.Prime p ∧ p > 20

-- Define the two primes
def prime1 := 23
def prime2 := 29

-- Given the conditions, prove the least possible product of two distinct primes greater than 20 is 667
theorem least_possible_product_of_primes :
  ∃ p1 p2 : Nat, is_prime_over_20 p1 ∧ is_prime_over_20 p2 ∧ p1 ≠ p2 ∧ (p1 * p2 = 667) :=
by
  -- Theorem statement without proof
  existsi (prime1)
  existsi (prime2)
  have h1 : is_prime_over_20 prime1 := by sorry
  have h2 : is_prime_over_20 prime2 := by sorry
  have h3 : prime1 ≠ prime2 := by sorry
  have h4 : prime1 * prime2 = 667 := by sorry
  exact ⟨h1, h2, h3, h4⟩

end least_possible_product_of_primes_l1011_101125


namespace color_pairings_correct_l1011_101190

noncomputable def num_color_pairings (bowls : ℕ) (glasses : ℕ) : ℕ :=
  bowls * glasses

theorem color_pairings_correct : 
  num_color_pairings 4 5 = 20 :=
by 
  -- proof omitted
  sorry

end color_pairings_correct_l1011_101190


namespace part_I_part_II_l1011_101191

-- Part I
theorem part_I (x : ℝ) : (|x + 1| + |x - 4| ≤ 2 * |x - 4|) ↔ (x < 1.5) :=
sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x : ℝ, |x + a| + |x - 4| ≥ 3) → (a ≤ -7 ∨ a ≥ -1) :=
sorry

end part_I_part_II_l1011_101191


namespace amount_of_money_l1011_101115

variable (x : ℝ)

-- Conditions
def condition1 : Prop := x < 2000
def condition2 : Prop := 4 * x > 2000
def condition3 : Prop := 4 * x - 2000 = 2000 - x

theorem amount_of_money (h1 : condition1 x) (h2 : condition2 x) (h3 : condition3 x) : x = 800 :=
by
  sorry

end amount_of_money_l1011_101115


namespace sum_of_angles_l1011_101169

theorem sum_of_angles (ABC ABD : ℝ) (n_octagon n_triangle : ℕ) 
(h1 : n_octagon = 8) 
(h2 : n_triangle = 3) 
(h3 : ABC = 180 * (n_octagon - 2) / n_octagon)
(h4 : ABD = 180 * (n_triangle - 2) / n_triangle) : 
ABC + ABD = 195 :=
by {
  sorry
}

end sum_of_angles_l1011_101169


namespace fair_decision_l1011_101179

def fair_selection (b c : ℕ) : Prop :=
  (b - c)^2 = b + c

theorem fair_decision (b c : ℕ) : fair_selection b c := by
  sorry

end fair_decision_l1011_101179


namespace paint_needed_l1011_101112

-- Definitions from conditions
def total_needed_paint := 70
def initial_paint := 36
def bought_paint := 23

-- The main statement to prove
theorem paint_needed : total_needed_paint - (initial_paint + bought_ppaint) = 11 :=
by
  -- Definitions are already imported and stated
  -- Just need to refer these to the theorem assertion correctly
  sorry

end paint_needed_l1011_101112


namespace problem1_problem2_l1011_101197

-- Define Sn as given
def S (n : ℕ) : ℕ := (n ^ 2 + n) / 2

-- Define a sequence a_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define b_n using a_n = log_2 b_n
def b (n : ℕ) : ℕ := 2 ^ n

-- Define the sum of first n terms of sequence b_n
def T (n : ℕ) : ℕ := (2 ^ (n + 1)) - 2

-- Our main theorem statements
theorem problem1 (n : ℕ) : a n = n := by
  sorry

theorem problem2 (n : ℕ) : (Finset.range n).sum b = T n := by
  sorry

end problem1_problem2_l1011_101197


namespace simplest_square_root_l1011_101183

theorem simplest_square_root (A B C D : Real) 
    (hA : A = Real.sqrt 0.1) 
    (hB : B = 1 / 2) 
    (hC : C = Real.sqrt 30) 
    (hD : D = Real.sqrt 18) : 
    C = Real.sqrt 30 := 
by 
    sorry

end simplest_square_root_l1011_101183


namespace omega_value_l1011_101144

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem omega_value (ω x₁ x₂ : ℝ) (h_ω : ω > 0) (h_x1 : f ω x₁ = -2) (h_x2 : f ω x₂ = 0) (h_min : |x₁ - x₂| = Real.pi) :
  ω = 1 / 2 := 
by 
  sorry

end omega_value_l1011_101144


namespace ratio_of_DE_EC_l1011_101137

noncomputable def ratio_DE_EC (a x : ℝ) : ℝ :=
  let DE := a - x
  x / DE

theorem ratio_of_DE_EC (a : ℝ) (H1 : ∀ x, x = 5 * a / 7) :
  ratio_DE_EC a (5 * a / 7) = 5 / 2 :=
by
  sorry

end ratio_of_DE_EC_l1011_101137


namespace no_three_natural_numbers_l1011_101113

theorem no_three_natural_numbers (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1)
    (h4 : b ∣ a^2 - 1) (h5 : a ∣ c^2 - 1) (h6 : b ∣ c^2 - 1) : false :=
by
  sorry

end no_three_natural_numbers_l1011_101113


namespace a_minus_b_eq_zero_l1011_101101

-- Definitions from the conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

-- The point (0, b)
def point_b (b : ℝ) : (ℝ × ℝ) := (0, b)

-- Slope condition at point (0, b)
def slope_of_f_at_0 (a : ℝ) : ℝ := a
def slope_of_tangent_line : ℝ := 1

-- Prove a - b = 0 given the conditions
theorem a_minus_b_eq_zero (a b : ℝ) 
    (h1 : f 0 a b = b)
    (h2 : tangent_line 0 b) 
    (h3 : slope_of_f_at_0 a = slope_of_tangent_line) : a - b = 0 :=
by
  sorry

end a_minus_b_eq_zero_l1011_101101


namespace rectangle_area_is_correct_l1011_101104

-- Define the conditions
def length : ℕ := 135
def breadth (l : ℕ) : ℕ := l / 3

-- Define the area of the rectangle
def area (l b : ℕ) : ℕ := l * b

-- The statement to prove
theorem rectangle_area_is_correct : area length (breadth length) = 6075 := by
  -- Proof goes here, this is just the statement
  sorry

end rectangle_area_is_correct_l1011_101104


namespace Kyle_monthly_income_l1011_101141

theorem Kyle_monthly_income :
  let rent := 1250
  let utilities := 150
  let retirement_savings := 400
  let groceries_eatingout := 300
  let insurance := 200
  let miscellaneous := 200
  let car_payment := 350
  let gas_maintenance := 350
  rent + utilities + retirement_savings + groceries_eatingout + insurance + miscellaneous + car_payment + gas_maintenance = 3200 :=
by
  -- Informal proof was provided in the solution.
  sorry

end Kyle_monthly_income_l1011_101141


namespace correct_equation_l1011_101106

-- Define conditions as variables in Lean
def cost_price (x : ℝ) : Prop := x > 0
def markup_percentage : ℝ := 0.40
def discount_percentage : ℝ := 0.80
def selling_price : ℝ := 240

-- Define the theorem
theorem correct_equation (x : ℝ) (hx : cost_price x) :
  x * (1 + markup_percentage) * discount_percentage = selling_price :=
by
  sorry

end correct_equation_l1011_101106


namespace distance_between_A_and_B_l1011_101182

theorem distance_between_A_and_B :
  let A := (0, 0)
  let B := (-10, 24)
  dist A B = 26 :=
by
  sorry

end distance_between_A_and_B_l1011_101182


namespace triangle_perimeter_l1011_101118

theorem triangle_perimeter {a b c : ℕ} (ha : a = 10) (hb : b = 6) (hc : c = 7) :
    a + b + c = 23 := by
  sorry

end triangle_perimeter_l1011_101118


namespace range_of_t_l1011_101155

variable (t : ℝ)

def point_below_line (x y a b c : ℝ) : Prop :=
  a * x - b * y + c < 0

theorem range_of_t (t : ℝ) : point_below_line 2 (3 * t) 2 (-1) 6 → t < 10 / 3 :=
  sorry

end range_of_t_l1011_101155


namespace most_stable_student_l1011_101123

-- Define the variances for the four students
def variance_A (SA2 : ℝ) : Prop := SA2 = 0.15
def variance_B (SB2 : ℝ) : Prop := SB2 = 0.32
def variance_C (SC2 : ℝ) : Prop := SC2 = 0.5
def variance_D (SD2 : ℝ) : Prop := SD2 = 0.25

-- Theorem proving that the most stable student is A
theorem most_stable_student {SA2 SB2 SC2 SD2 : ℝ} 
  (hA : variance_A SA2) 
  (hB : variance_B SB2)
  (hC : variance_C SC2)
  (hD : variance_D SD2) : 
  SA2 < SB2 ∧ SA2 < SC2 ∧ SA2 < SD2 :=
by
  rw [variance_A, variance_B, variance_C, variance_D] at *
  sorry

end most_stable_student_l1011_101123


namespace solution_set_l1011_101198

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom monotone_increasing : ∀ x y, x < y → f x ≤ f y
axiom f_at_3 : f 3 = 2

-- Proof statement
theorem solution_set : {x : ℝ | -2 ≤ f (3 - x) ∧ f (3 - x) ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 6} :=
by {
  sorry
}

end solution_set_l1011_101198


namespace chess_competition_players_l1011_101111

theorem chess_competition_players (J H : ℕ) (total_points : ℕ) (junior_points : ℕ) (high_school_points : ℕ → ℕ)
  (plays : ℕ → ℕ)
  (H_junior_points : junior_points = 8)
  (H_total_points : total_points = (J + H) * (J + H - 1) / 2)
  (H_total_points_contribution : total_points = junior_points + H * high_school_points H)
  (H_even_distribution : ∀ x : ℕ, 0 ≤ x ∧ x ≤ J → high_school_points H = x * (x - 1) / 2)
  (H_H_cases : H = 7 ∨ H = 9 ∨ H = 14) :
  H = 7 ∨ H = 14 :=
by
  have H_cases : H = 7 ∨ H = 14 :=
    by
      sorry
  exact H_cases

end chess_competition_players_l1011_101111


namespace fill_in_the_blanks_l1011_101186

theorem fill_in_the_blanks :
  (9 / 18 = 0.5) ∧
  (27 / 54 = 0.5) ∧
  (50 / 100 = 0.5) ∧
  (10 / 20 = 0.5) ∧
  (5 / 10 = 0.5) :=
by
  sorry

end fill_in_the_blanks_l1011_101186


namespace min_value_reciprocal_sum_l1011_101159

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a = 1 ∧ b = 1) → (1 / a + 1 / b = 2) := by
  intros h
  sorry

end min_value_reciprocal_sum_l1011_101159


namespace value_of_S6_l1011_101163

theorem value_of_S6 (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12077 :=
by sorry

end value_of_S6_l1011_101163


namespace value_of_square_of_sum_l1011_101187

theorem value_of_square_of_sum (x y: ℝ) 
(h1: 2 * x * (x + y) = 58) 
(h2: 3 * y * (x + y) = 111):
  (x + y)^2 = (169/5)^2 := by
  sorry

end value_of_square_of_sum_l1011_101187


namespace find_special_numbers_l1011_101131

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Define the main statement to be proved -/
theorem find_special_numbers :
  { n : ℕ | sum_of_digits n * (sum_of_digits n - 1) = n - 1 } = {1, 13, 43, 91, 157} :=
by
  sorry

end find_special_numbers_l1011_101131


namespace fraction_equality_l1011_101171

theorem fraction_equality : (2 + 4) / (1 + 2) = 2 := by
  sorry

end fraction_equality_l1011_101171


namespace tetrahedron_circumscribed_sphere_radius_l1011_101168

open Real

theorem tetrahedron_circumscribed_sphere_radius :
  ∀ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 5 →
    dist C D = 5 →
    dist A C = sqrt 34 →
    dist B D = sqrt 34 →
    dist A D = sqrt 41 →
    dist B C = sqrt 41 →
    ∃ (R : ℝ), R = 5 * sqrt 2 / 2 :=
by
  intros A B C D hAB hCD hAC hBD hAD hBC
  sorry

end tetrahedron_circumscribed_sphere_radius_l1011_101168


namespace treasure_probability_l1011_101164

variable {Island : Type}

-- Define the probabilities.
def prob_treasure : ℚ := 1 / 3
def prob_trap : ℚ := 1 / 6
def prob_neither : ℚ := 1 / 2

-- Define the number of islands.
def num_islands : ℕ := 5

-- Define the probability of encountering exactly 4 islands with treasure and one with neither traps nor treasures.
theorem treasure_probability :
  (num_islands.choose 4) * (prob_ttreasure^4) * (prob_neither^1) = (5 : ℚ) * (1 / 81) * (1 / 2) :=
  by
  sorry

end treasure_probability_l1011_101164


namespace express_inequality_l1011_101161

theorem express_inequality (x : ℝ) : x + 4 ≥ -1 := sorry

end express_inequality_l1011_101161


namespace train_speed_l1011_101134

def train_length : ℕ := 180
def crossing_time : ℕ := 12

theorem train_speed :
  train_length / crossing_time = 15 := sorry

end train_speed_l1011_101134


namespace max_intersections_l1011_101133

-- Define the conditions
def num_points_x : ℕ := 15
def num_points_y : ℕ := 10

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the problem statement
theorem max_intersections (I : ℕ) :
  (15 : ℕ) == num_points_x →
  (10 : ℕ) == num_points_y →
  (I = binom 15 2 * binom 10 2) →
  I = 4725 := by
  -- We add sorry to skip the proof
  sorry

end max_intersections_l1011_101133


namespace smallest_m_plus_n_l1011_101160

theorem smallest_m_plus_n : ∃ (m n : ℕ), m > 1 ∧ 
  (∃ (a b : ℝ), a = (1 : ℝ) / (m * n : ℝ) ∧ b = (m : ℝ) / (n : ℝ) ∧ b - a = (1 : ℝ) / 1007) ∧
  (∀ (k l : ℕ), k > 1 ∧ 
    (∃ (c d : ℝ), c = (1 : ℝ) / (k * l : ℝ) ∧ d = (k : ℝ) / (l : ℝ) ∧ d - c = (1 : ℝ) / 1007) → m + n ≤ k + l) ∧ 
  m + n = 19099 :=
sorry

end smallest_m_plus_n_l1011_101160


namespace minimum_value_l1011_101120

theorem minimum_value (x : ℝ) (hx : 0 ≤ x) : ∃ y : ℝ, y = x^2 - 6 * x + 8 ∧ (∀ t : ℝ, 0 ≤ t → y ≤ t^2 - 6 * t + 8) :=
sorry

end minimum_value_l1011_101120


namespace DongfangElementary_total_students_l1011_101170

theorem DongfangElementary_total_students (x y : ℕ) 
  (h1 : x = y + 2)
  (h2 : 10 * (y + 2) = 22 * 11 * (y - 22))
  (h3 : x - x / 11 = 2 * (y - 22)) :
  x + y = 86 :=
by
  sorry

end DongfangElementary_total_students_l1011_101170


namespace parabola_properties_l1011_101165

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c : ℝ) (h₀ : a ≠ 0)
    (h₁ : parabola a b c (-1) = -1)
    (h₂ : parabola a b c 0 = 1)
    (h₃ : parabola a b c (-2) > 1) :
    (a * b * c > 0) ∧
    (∃ Δ : ℝ, Δ > 0 ∧ (Δ = b^2 - 4*a*c)) ∧
    (a + b + c > 7) :=
sorry

end parabola_properties_l1011_101165


namespace g_at_3_l1011_101162

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = 2

theorem g_at_3 : g 3 = 0 :=
by
  sorry

end g_at_3_l1011_101162


namespace sequence_value_is_correct_l1011_101122

theorem sequence_value_is_correct (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2) : a 8 = 15 :=
sorry

end sequence_value_is_correct_l1011_101122
