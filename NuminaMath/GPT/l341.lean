import Mathlib

namespace NUMINAMATH_GPT_total_newspapers_collected_l341_34139

-- Definitions based on the conditions
def Chris_collected : ℕ := 42
def Lily_collected : ℕ := 23

-- The proof statement
theorem total_newspapers_collected :
  Chris_collected + Lily_collected = 65 := by
  sorry

end NUMINAMATH_GPT_total_newspapers_collected_l341_34139


namespace NUMINAMATH_GPT_neg_proposition_l341_34122

theorem neg_proposition :
  (¬(∀ x : ℕ, x^3 > x^2)) ↔ (∃ x : ℕ, x^3 ≤ x^2) := 
sorry

end NUMINAMATH_GPT_neg_proposition_l341_34122


namespace NUMINAMATH_GPT_dividend_calculation_l341_34141

theorem dividend_calculation :
  ∀ (divisor quotient remainder : ℝ), 
  divisor = 37.2 → 
  quotient = 14.61 → 
  remainder = 0.67 → 
  (divisor * quotient + remainder) = 544.042 :=
by
  intros divisor quotient remainder h_div h_qt h_rm
  sorry

end NUMINAMATH_GPT_dividend_calculation_l341_34141


namespace NUMINAMATH_GPT_hcf_lcm_product_l341_34172

theorem hcf_lcm_product (a b : ℕ) (H : a * b = 45276) (L : Nat.lcm a b = 2058) : Nat.gcd a b = 22 :=
by 
  -- The proof steps go here
  sorry

end NUMINAMATH_GPT_hcf_lcm_product_l341_34172


namespace NUMINAMATH_GPT_ratio_of_areas_l341_34152

theorem ratio_of_areas (Q : Point) (r1 r2 : ℝ) (h : r1 < r2)
  (arc_length_smaller : ℝ) (arc_length_larger : ℝ)
  (h_arc_smaller : arc_length_smaller = (60 / 360) * (2 * r1 * π))
  (h_arc_larger : arc_length_larger = (30 / 360) * (2 * r2 * π))
  (h_equal_arcs : arc_length_smaller = arc_length_larger) :
  (π * r1^2) / (π * r2^2) = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l341_34152


namespace NUMINAMATH_GPT_total_amount_paid_correct_l341_34173

-- Definitions of wholesale costs, retail markups, and employee discounts
def wholesale_cost_video_recorder : ℝ := 200
def retail_markup_video_recorder : ℝ := 0.20
def employee_discount_video_recorder : ℝ := 0.30

def wholesale_cost_digital_camera : ℝ := 150
def retail_markup_digital_camera : ℝ := 0.25
def employee_discount_digital_camera : ℝ := 0.20

def wholesale_cost_smart_tv : ℝ := 800
def retail_markup_smart_tv : ℝ := 0.15
def employee_discount_smart_tv : ℝ := 0.25

-- Calculation of retail prices
def retail_price (wholesale_cost : ℝ) (markup : ℝ) : ℝ :=
  wholesale_cost * (1 + markup)

-- Calculation of employee prices
def employee_price (retail_price : ℝ) (discount : ℝ) : ℝ :=
  retail_price * (1 - discount)

-- Retail prices
def retail_price_video_recorder := retail_price wholesale_cost_video_recorder retail_markup_video_recorder
def retail_price_digital_camera := retail_price wholesale_cost_digital_camera retail_markup_digital_camera
def retail_price_smart_tv := retail_price wholesale_cost_smart_tv retail_markup_smart_tv

-- Employee prices
def employee_price_video_recorder := employee_price retail_price_video_recorder employee_discount_video_recorder
def employee_price_digital_camera := employee_price retail_price_digital_camera employee_discount_digital_camera
def employee_price_smart_tv := employee_price retail_price_smart_tv employee_discount_smart_tv

-- Total amount paid by the employee
def total_amount_paid := 
  employee_price_video_recorder 
  + employee_price_digital_camera 
  + employee_price_smart_tv

theorem total_amount_paid_correct :
  total_amount_paid = 1008 := 
  by 
    sorry

end NUMINAMATH_GPT_total_amount_paid_correct_l341_34173


namespace NUMINAMATH_GPT_committee_count_8_choose_4_l341_34113

theorem committee_count_8_choose_4 : (Nat.choose 8 4) = 70 :=
  by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_committee_count_8_choose_4_l341_34113


namespace NUMINAMATH_GPT_unique_prime_solution_l341_34168

theorem unique_prime_solution :
  ∃! (p : ℕ), Prime p ∧ (∃ (k : ℤ), 2 * (p ^ 4) - 7 * (p ^ 2) + 1 = k ^ 2) := 
sorry

end NUMINAMATH_GPT_unique_prime_solution_l341_34168


namespace NUMINAMATH_GPT_find_a_b_l341_34132

theorem find_a_b (a b : ℤ) (h: 4 * a^2 + 3 * b^2 + 10 * a * b = 144) :
    (a = 2 ∧ b = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_b_l341_34132


namespace NUMINAMATH_GPT_question1_question2_l341_34163

noncomputable def setA := {x : ℝ | -2 < x ∧ x < 4}
noncomputable def setB (m : ℝ) := {x : ℝ | x < -m}

-- (1) If A ∩ B = ∅, find the range of the real number m.
theorem question1 (m : ℝ) (h : setA ∩ setB m = ∅) : 2 ≤ m := by
  sorry

-- (2) If A ⊂ B, find the range of the real number m.
theorem question2 (m : ℝ) (h : setA ⊂ setB m) : m ≤ 4 := by
  sorry

end NUMINAMATH_GPT_question1_question2_l341_34163


namespace NUMINAMATH_GPT_rationalize_denominator_l341_34188

-- Problem statement
theorem rationalize_denominator :
  1 / (Real.sqrt 5 - 2) = Real.sqrt 5 + 2 := by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l341_34188


namespace NUMINAMATH_GPT_area_of_rectangle_ABCD_l341_34125

-- Definitions for the conditions
def small_square_area := 4
def total_small_squares := 2
def large_square_area := (2 * (2 : ℝ)) * (2 * (2 : ℝ))
def total_squares_area := total_small_squares * small_square_area + large_square_area

-- The main proof statement
theorem area_of_rectangle_ABCD : total_squares_area = 24 := 
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_ABCD_l341_34125


namespace NUMINAMATH_GPT_line_always_passes_fixed_point_l341_34103

theorem line_always_passes_fixed_point : ∀ (m : ℝ), (m-1)*(-2) - 1 + (2*m-1) = 0 :=
by
  intro m
  -- Calculations can be done here to prove the theorem straightforwardly.
  sorry

end NUMINAMATH_GPT_line_always_passes_fixed_point_l341_34103


namespace NUMINAMATH_GPT_books_on_shelf_l341_34119

theorem books_on_shelf (original_books : ℕ) (books_added : ℕ) (total_books : ℕ) (h1 : original_books = 38) 
(h2 : books_added = 10) : total_books = 48 :=
by 
  sorry

end NUMINAMATH_GPT_books_on_shelf_l341_34119


namespace NUMINAMATH_GPT_determine_all_functions_l341_34186

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)

theorem determine_all_functions (f : ℝ → ℝ) (h : functional_equation f) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_GPT_determine_all_functions_l341_34186


namespace NUMINAMATH_GPT_sin_identity_cos_identity_l341_34199

-- Define the condition that alpha + beta + gamma = 180 degrees.
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

-- Prove that sin 4α + sin 4β + sin 4γ = -4 sin 2α sin 2β sin 2γ.
theorem sin_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.sin (4 * α) + Real.sin (4 * β) + Real.sin (4 * γ) = -4 * Real.sin (2 * α) * Real.sin (2 * β) * Real.sin (2 * γ) := by
  sorry

-- Prove that cos 4α + cos 4β + cos 4γ = 4 cos 2α cos 2β cos 2γ - 1.
theorem cos_identity (α β γ : ℝ) (h : angles_sum_to_180 α β γ) :
  Real.cos (4 * α) + Real.cos (4 * β) + Real.cos (4 * γ) = 4 * Real.cos (2 * α) * Real.cos (2 * β) * Real.cos (2 * γ) - 1 := by
  sorry

end NUMINAMATH_GPT_sin_identity_cos_identity_l341_34199


namespace NUMINAMATH_GPT_ratio_of_a_over_b_l341_34161

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem ratio_of_a_over_b (a b : ℝ) (h_max : ∀ x : ℝ, f a b x ≤ 10)
  (h_cond1 : f a b 1 = 10) (h_cond2 : (deriv (f a b)) 1 = 0) :
  a / b = -2/3 :=
sorry

end NUMINAMATH_GPT_ratio_of_a_over_b_l341_34161


namespace NUMINAMATH_GPT_largest_expression_is_d_l341_34149

def expr_a := 3 + 0 + 4 + 8
def expr_b := 3 * 0 + 4 + 8
def expr_c := 3 + 0 * 4 + 8
def expr_d := 3 + 0 + 4 * 8
def expr_e := 3 * 0 * 4 * 8
def expr_f := (3 + 0 + 4) / 8

theorem largest_expression_is_d : 
  expr_d = 35 ∧ 
  expr_a = 15 ∧ 
  expr_b = 12 ∧ 
  expr_c = 11 ∧ 
  expr_e = 0 ∧ 
  expr_f = 7 / 8 ∧
  35 > 15 ∧ 
  35 > 12 ∧ 
  35 > 11 ∧ 
  35 > 0 ∧ 
  35 > 7 / 8 := 
by
  sorry

end NUMINAMATH_GPT_largest_expression_is_d_l341_34149


namespace NUMINAMATH_GPT_sean_days_played_is_14_l341_34159

def total_minutes_played : Nat := 1512
def indira_minutes_played : Nat := 812
def sean_minutes_per_day : Nat := 50
def sean_total_minutes : Nat := total_minutes_played - indira_minutes_played
def sean_days_played : Nat := sean_total_minutes / sean_minutes_per_day

theorem sean_days_played_is_14 : sean_days_played = 14 :=
by
  sorry

end NUMINAMATH_GPT_sean_days_played_is_14_l341_34159


namespace NUMINAMATH_GPT_remainder_addition_l341_34140

theorem remainder_addition (k m : ℤ) (x y : ℤ) (h₁ : x = 124 * k + 13) (h₂ : y = 186 * m + 17) :
  ((x + y + 19) % 62) = 49 :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_addition_l341_34140


namespace NUMINAMATH_GPT_find_b_l341_34114

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 15 * b) : b = 147 :=
sorry

end NUMINAMATH_GPT_find_b_l341_34114


namespace NUMINAMATH_GPT_n_value_condition_l341_34194

theorem n_value_condition (n : ℤ) : 
  (3 * (n ^ 2 + n) + 7) % 5 = 0 ↔ n % 5 = 2 := sorry

end NUMINAMATH_GPT_n_value_condition_l341_34194


namespace NUMINAMATH_GPT_chessboard_tiling_l341_34167

theorem chessboard_tiling (chessboard : Fin 8 × Fin 8 → Prop) (colors : Fin 8 × Fin 8 → Bool)
  (removed_squares : (Fin 8 × Fin 8) × (Fin 8 × Fin 8))
  (h_diff_colors : colors removed_squares.1 ≠ colors removed_squares.2) :
  ∃ f : (Fin 8 × Fin 8) → (Fin 8 × Fin 8), ∀ x, chessboard x → chessboard (f x) :=
by
  sorry

end NUMINAMATH_GPT_chessboard_tiling_l341_34167


namespace NUMINAMATH_GPT_incorrect_expression_l341_34184

theorem incorrect_expression :
  ¬((|(-5 : ℤ)|)^2 = 5) :=
by
sorry

end NUMINAMATH_GPT_incorrect_expression_l341_34184


namespace NUMINAMATH_GPT_avg_score_assigned_day_l341_34146

theorem avg_score_assigned_day
  (total_students : ℕ)
  (exam_assigned_day_students_perc : ℕ)
  (exam_makeup_day_students_perc : ℕ)
  (avg_makeup_day_score : ℕ)
  (total_avg_score : ℕ)
  : exam_assigned_day_students_perc = 70 → 
    exam_makeup_day_students_perc = 30 → 
    avg_makeup_day_score = 95 → 
    total_avg_score = 74 → 
    total_students = 100 → 
    (70 * 65 + 30 * 95 = 7400) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_avg_score_assigned_day_l341_34146


namespace NUMINAMATH_GPT_selling_price_correct_l341_34160

theorem selling_price_correct (cost_price : ℝ) (loss_percent : ℝ) (selling_price : ℝ) 
  (h_cost : cost_price = 600) 
  (h_loss : loss_percent = 25)
  (h_selling_price : selling_price = cost_price - (loss_percent / 100) * cost_price) : 
  selling_price = 450 := 
by 
  rw [h_cost, h_loss] at h_selling_price
  norm_num at h_selling_price
  exact h_selling_price

#check selling_price_correct

end NUMINAMATH_GPT_selling_price_correct_l341_34160


namespace NUMINAMATH_GPT_books_on_each_shelf_l341_34187

theorem books_on_each_shelf (M P x : ℕ) (h1 : 3 * M + 5 * P = 72) (h2 : M = x) (h3 : P = x) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_books_on_each_shelf_l341_34187


namespace NUMINAMATH_GPT_find_digit_A_l341_34147

theorem find_digit_A : ∃ A : ℕ, A < 10 ∧ (200 + 10 * A + 4) % 13 = 0 ∧ A = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_digit_A_l341_34147


namespace NUMINAMATH_GPT_b_bounded_l341_34120

open Real

-- Define sequences of real numbers
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define initial conditions and properties
axiom a0_gt_half : a 0 > 1/2
axiom a_non_decreasing : ∀ n : ℕ, a (n + 1) ≥ a n
axiom b_recursive : ∀ n : ℕ, b (n + 1) = a n * (b n + b (n + 2))

-- Prove the sequence (b_n) is bounded
theorem b_bounded : ∃ M : ℝ, ∀ n : ℕ, b n ≤ M :=
by
  sorry

end NUMINAMATH_GPT_b_bounded_l341_34120


namespace NUMINAMATH_GPT_tangent_line_eq_l341_34118

noncomputable def f (x : ℝ) : ℝ := x + Real.log x

theorem tangent_line_eq :
  ∃ (m b : ℝ), (m = (deriv f 1)) ∧ (b = (f 1 - m * 1)) ∧
   (∀ (x y : ℝ), y = m * (x - 1) + b ↔ y = 2 * x - 1) :=
by sorry

end NUMINAMATH_GPT_tangent_line_eq_l341_34118


namespace NUMINAMATH_GPT_subset_exists_l341_34127

theorem subset_exists (p : ℕ) (hp : Nat.Prime p) (A : Finset ℕ) (hA : A.card = p - 1) 
  (hA_div : ∀ a ∈ A, ¬ p ∣ a) :
  ∀ n ∈ Finset.range p, ∃ B ⊆ A, (B.sum id) % p = n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_subset_exists_l341_34127


namespace NUMINAMATH_GPT_restaurant_made_correct_amount_l341_34155

noncomputable def restaurant_revenue : ℝ := 
  let price1 := 8
  let qty1 := 10
  let price2 := 10
  let qty2 := 5
  let price3 := 4
  let qty3 := 20
  let total_sales := qty1 * price1 + qty2 * price2 + qty3 * price3
  let discount := 0.10
  let discounted_total := total_sales * (1 - discount)
  let sales_tax := 0.05
  let final_amount := discounted_total * (1 + sales_tax)
  final_amount

theorem restaurant_made_correct_amount : restaurant_revenue = 198.45 := by
  sorry

end NUMINAMATH_GPT_restaurant_made_correct_amount_l341_34155


namespace NUMINAMATH_GPT_rectangle_lengths_l341_34154

theorem rectangle_lengths (side_length : ℝ) (width1 width2: ℝ) (length1 length2 : ℝ) 
  (h1 : side_length = 6) 
  (h2 : width1 = 4) 
  (h3 : width2 = 3)
  (h_area_square : side_length * side_length = 36)
  (h_area_rectangle1 : width1 * length1 = side_length * side_length)
  (h_area_rectangle2 : width2 * length2 = (1 / 2) * (side_length * side_length)) :
  length1 = 9 ∧ length2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_lengths_l341_34154


namespace NUMINAMATH_GPT_product_of_primes_is_582_l341_34195

-- Define the relevant primes based on the conditions.
def smallest_one_digit_prime_1 := 2
def smallest_one_digit_prime_2 := 3
def largest_two_digit_prime := 97

-- Define the product of these primes as stated in the problem.
def product_of_primes := smallest_one_digit_prime_1 * smallest_one_digit_prime_2 * largest_two_digit_prime

-- Prove that this product equals to 582.
theorem product_of_primes_is_582 : product_of_primes = 582 :=
by {
  sorry
}

end NUMINAMATH_GPT_product_of_primes_is_582_l341_34195


namespace NUMINAMATH_GPT_analytical_expression_maximum_value_l341_34112

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) + 1

theorem analytical_expression (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, abs (x - (x + (Real.pi / (2 * ω)))) = Real.pi / 2) : 
  f x 2 = 2 * Real.sin (2 * x - Real.pi / 6) + 1 :=
sorry

theorem maximum_value (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  2 * Real.sin (2 * x - Real.pi / 6) + 1 ≤ 3 :=
sorry

end NUMINAMATH_GPT_analytical_expression_maximum_value_l341_34112


namespace NUMINAMATH_GPT_total_limes_l341_34148

-- Define the number of limes picked by Alyssa, Mike, and Tom's plums
def alyssa_limes : ℕ := 25
def mike_limes : ℕ := 32
def tom_plums : ℕ := 12

theorem total_limes : alyssa_limes + mike_limes = 57 := by
  -- The proof is omitted as per the instruction
  sorry

end NUMINAMATH_GPT_total_limes_l341_34148


namespace NUMINAMATH_GPT_swim_time_l341_34179

-- Definitions based on conditions:
def speed_in_still_water : ℝ := 6.5 -- speed of the man in still water (km/h)
def distance_downstream : ℝ := 16 -- distance swam downstream (km)
def distance_upstream : ℝ := 10 -- distance swam upstream (km)
def time_downstream := 2 -- time taken to swim downstream (hours)
def time_upstream := 2 -- time taken to swim upstream (hours)

-- Defining the speeds taking the current into account:
def speed_downstream (c : ℝ) : ℝ := speed_in_still_water + c
def speed_upstream (c : ℝ) : ℝ := speed_in_still_water - c

-- Assumption that the time took for both downstream and upstream are equal
def time_eq (c : ℝ) : Prop :=
  distance_downstream / (speed_downstream c) = distance_upstream / (speed_upstream c)

-- The proof we need to establish:
theorem swim_time (c : ℝ) (h : time_eq c) : time_downstream = time_upstream := by
  sorry

end NUMINAMATH_GPT_swim_time_l341_34179


namespace NUMINAMATH_GPT_triangular_number_difference_l341_34198

-- Definition of the nth triangular number
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Theorem stating the problem
theorem triangular_number_difference :
  triangular_number 2010 - triangular_number 2008 = 4019 :=
by
  sorry

end NUMINAMATH_GPT_triangular_number_difference_l341_34198


namespace NUMINAMATH_GPT_union_of_sets_l341_34166

def A : Set ℤ := {1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_sets : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_sets_l341_34166


namespace NUMINAMATH_GPT_average_sales_l341_34101

theorem average_sales
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 90)
  (h2 : a2 = 50)
  (h3 : a3 = 70)
  (h4 : a4 = 110)
  (h5 : a5 = 80) :
  (a1 + a2 + a3 + a4 + a5) / 5 = 80 :=
by
  sorry

end NUMINAMATH_GPT_average_sales_l341_34101


namespace NUMINAMATH_GPT_ezekiel_third_day_hike_l341_34133

-- Ezekiel's total hike distance
def total_distance : ℕ := 50

-- Distance covered on the first day
def first_day_distance : ℕ := 10

-- Distance covered on the second day
def second_day_distance : ℕ := total_distance / 2

-- Distance remaining for the third day
def third_day_distance : ℕ := total_distance - first_day_distance - second_day_distance

-- The distance Ezekiel had to hike on the third day
theorem ezekiel_third_day_hike : third_day_distance = 15 := by
  sorry

end NUMINAMATH_GPT_ezekiel_third_day_hike_l341_34133


namespace NUMINAMATH_GPT_original_pencils_count_l341_34126

theorem original_pencils_count (total_pencils : ℕ) (added_pencils : ℕ) (original_pencils : ℕ) : total_pencils = original_pencils + added_pencils → original_pencils = 2 :=
by
  sorry

end NUMINAMATH_GPT_original_pencils_count_l341_34126


namespace NUMINAMATH_GPT_carpet_dimensions_l341_34138

theorem carpet_dimensions (a b : ℕ) 
  (h1 : a^2 + b^2 = 38^2 + 55^2) 
  (h2 : a^2 + b^2 = 50^2 + 55^2) 
  (h3 : a ≤ b) : 
  (a = 25 ∧ b = 50) ∨ (a = 50 ∧ b = 25) :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_carpet_dimensions_l341_34138


namespace NUMINAMATH_GPT_find_base_l341_34176

theorem find_base (x y : ℕ) (b : ℕ) (h1 : 3 ^ x * b ^ y = 19683) (h2 : x - y = 9) (h3 : x = 9) : b = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_base_l341_34176


namespace NUMINAMATH_GPT_earnings_per_hour_l341_34153

-- Define the conditions
def widgetsProduced : Nat := 750
def hoursWorked : Nat := 40
def totalEarnings : ℝ := 620
def earningsPerWidget : ℝ := 0.16

-- Define the proof goal
theorem earnings_per_hour :
  ∃ H : ℝ, (hoursWorked * H + widgetsProduced * earningsPerWidget = totalEarnings) ∧ H = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_earnings_per_hour_l341_34153


namespace NUMINAMATH_GPT_profit_percentage_l341_34169

theorem profit_percentage (SP CP : ℝ) (hs : SP = 270) (hc : CP = 225) : 
  ((SP - CP) / CP) * 100 = 20 :=
by
  rw [hs, hc]
  sorry  -- The proof will go here

end NUMINAMATH_GPT_profit_percentage_l341_34169


namespace NUMINAMATH_GPT_selling_price_of_car_l341_34193

theorem selling_price_of_car (purchase_price repair_cost : ℝ) (profit_percent : ℝ) 
    (h1 : purchase_price = 42000) (h2 : repair_cost = 8000) (h3 : profit_percent = 29.8) :
    (purchase_price + repair_cost) * (1 + profit_percent / 100) = 64900 := 
by 
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_selling_price_of_car_l341_34193


namespace NUMINAMATH_GPT_geometric_sequence_a7_value_l341_34134

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_a7_value (a : ℕ → ℝ) (r : ℝ) (S : ℕ → ℝ) :
  (∀ n, 0 < a n) →
  (geometric_sequence a r) →
  (S 4 = 3 * S 2) →
  (a 3 = 2) →
  (S n = a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3) →
  a 7 = 8 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a7_value_l341_34134


namespace NUMINAMATH_GPT_sphere_radius_eq_three_l341_34129

theorem sphere_radius_eq_three (r : ℝ) (h : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 := 
sorry

end NUMINAMATH_GPT_sphere_radius_eq_three_l341_34129


namespace NUMINAMATH_GPT_probability_second_marble_purple_correct_l341_34191

/-!
  Bag A has 5 red marbles and 5 green marbles.
  Bag B has 8 purple marbles and 2 orange marbles.
  Bag C has 3 purple marbles and 7 orange marbles.
  Bag D has 4 purple marbles and 6 orange marbles.
  A marble is drawn at random from Bag A.
  If it is red, a marble is drawn at random from Bag B;
  if it is green, a marble is drawn at random from Bag C;
  but if it is neither (an impossible scenario in this setup), a marble would be drawn from Bag D.
  Prove that the probability of the second marble drawn being purple is 11/20.
-/

noncomputable def probability_second_marble_purple : ℚ :=
  let p_red_A := 5 / 10
  let p_green_A := 5 / 10
  let p_purple_B := 8 / 10
  let p_purple_C := 3 / 10
  (p_red_A * p_purple_B) + (p_green_A * p_purple_C)

theorem probability_second_marble_purple_correct :
  probability_second_marble_purple = 11 / 20 := sorry

end NUMINAMATH_GPT_probability_second_marble_purple_correct_l341_34191


namespace NUMINAMATH_GPT_trapezoid_larger_base_length_l341_34136

theorem trapezoid_larger_base_length
  (x : ℝ)
  (h_ratio : 3 = 3 * 1)
  (h_midline : (x + 3 * x) / 2 = 24) :
  3 * x = 36 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_larger_base_length_l341_34136


namespace NUMINAMATH_GPT_angle_triple_complement_l341_34100

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Define the problem statement
theorem angle_triple_complement (x : ℝ) (h : x = 3 * complement x) : x = 67.5 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_angle_triple_complement_l341_34100


namespace NUMINAMATH_GPT_smallest_positive_multiple_l341_34109

theorem smallest_positive_multiple (a : ℕ) (h : a > 0) : ∃ a > 0, (31 * a) % 103 = 7 := 
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_l341_34109


namespace NUMINAMATH_GPT_rectangle_area_l341_34180

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area : area length width = 588 := sorry

end NUMINAMATH_GPT_rectangle_area_l341_34180


namespace NUMINAMATH_GPT_sequence_length_l341_34182

theorem sequence_length 
  (a₁ : ℤ) (d : ℤ) (aₙ : ℤ) (n : ℕ) 
  (h₁ : a₁ = -4) 
  (h₂ : d = 3) 
  (h₃ : aₙ = 32) 
  (h₄ : aₙ = a₁ + (n - 1) * d) : 
  n = 13 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_length_l341_34182


namespace NUMINAMATH_GPT_cost_per_book_l341_34150

theorem cost_per_book (a r n c : ℕ) (h : a - r = n * c) : c = 7 :=
by sorry

end NUMINAMATH_GPT_cost_per_book_l341_34150


namespace NUMINAMATH_GPT_average_marks_110_l341_34196

def marks_problem (P C M B E : ℕ) : Prop :=
  (C = P + 90) ∧
  (M = P + 140) ∧
  (P + C + M + B + E = P + 350) ∧
  (B = E) ∧
  (P ≥ 40) ∧
  (C ≥ 40) ∧
  (M ≥ 40) ∧
  (B ≥ 40) ∧
  (E ≥ 40)

theorem average_marks_110 (P C M B E : ℕ) (h : marks_problem P C M B E) : 
    (B + C + M) / 3 = 110 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_110_l341_34196


namespace NUMINAMATH_GPT_sam_has_8_marbles_l341_34104

theorem sam_has_8_marbles :
  ∀ (steve sam sally : ℕ),
  sam = 2 * steve →
  sally = sam - 5 →
  steve + 3 = 10 →
  sam - 6 = 8 :=
by
  intros steve sam sally
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sam_has_8_marbles_l341_34104


namespace NUMINAMATH_GPT_sum_of_numbers_l341_34105

theorem sum_of_numbers : 145 + 33 + 29 + 13 = 220 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l341_34105


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_S6_l341_34124

noncomputable def S_6 (a : Nat) (q : Nat) : Nat :=
  (q ^ 6 - 1) / (q - 1)

theorem arithmetic_geometric_sequence_S6 (a : Nat) (q : Nat) (h1 : a * q ^ 1 = 2) (h2 : a * q ^ 3 = 8) (hq : q > 0) : S_6 a q = 63 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_S6_l341_34124


namespace NUMINAMATH_GPT_sqrt_of_1024_is_32_l341_34183

theorem sqrt_of_1024_is_32 (y : ℕ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 :=
sorry

end NUMINAMATH_GPT_sqrt_of_1024_is_32_l341_34183


namespace NUMINAMATH_GPT_ducklings_distance_l341_34156

noncomputable def ducklings_swim (r : ℝ) (n : ℕ) : Prop :=
  ∀ (ducklings : Fin n → ℝ × ℝ), (∀ i, (ducklings i).1 ^ 2 + (ducklings i).2 ^ 2 = r ^ 2) →
    ∃ (i j : Fin n), i ≠ j ∧ (ducklings i - ducklings j).1 ^ 2 + (ducklings i - ducklings j).2 ^ 2 ≤ r ^ 2

theorem ducklings_distance :
  ducklings_swim 5 6 :=
by sorry

end NUMINAMATH_GPT_ducklings_distance_l341_34156


namespace NUMINAMATH_GPT_xiaodong_election_l341_34130

theorem xiaodong_election (V : ℕ) (h1 : 0 < V) :
  let total_needed := (3 : ℚ) / 4 * V
  let votes_obtained := (5 : ℚ) / 6 * (2 : ℚ) / 3 * V
  let remaining_votes := V - (2 : ℚ) / 3 * V
  total_needed - votes_obtained = (7 : ℚ) / 12 * remaining_votes :=
by 
  sorry

end NUMINAMATH_GPT_xiaodong_election_l341_34130


namespace NUMINAMATH_GPT_least_common_denominator_l341_34116

-- We first need to define the function to compute the LCM of a list of natural numbers.
def lcm_list (ns : List ℕ) : ℕ :=
ns.foldr Nat.lcm 1

theorem least_common_denominator : 
  lcm_list [3, 4, 5, 8, 9, 11] = 3960 := 
by
  -- Here's where the proof would go
  sorry

end NUMINAMATH_GPT_least_common_denominator_l341_34116


namespace NUMINAMATH_GPT_tablet_screen_area_difference_l341_34131

theorem tablet_screen_area_difference (d1 d2 : ℝ) (A1 A2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 7) :
  A1 - A2 = 7.5 :=
by
  -- Note: The proof is omitted as the prompt requires only the statement.
  sorry

end NUMINAMATH_GPT_tablet_screen_area_difference_l341_34131


namespace NUMINAMATH_GPT_balance_difference_l341_34165

def compounded_balance (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n

def simple_interest_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

/-- Cedric deposits $15,000 into an account that pays 6% interest compounded annually,
    Daniel deposits $15,000 into an account that pays 8% simple annual interest.
    After 10 years, the positive difference between their balances is $137. -/
theorem balance_difference :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 10
  compounded_balance P r_cedric t - simple_interest_balance P r_daniel t = 137 := 
sorry

end NUMINAMATH_GPT_balance_difference_l341_34165


namespace NUMINAMATH_GPT_product_of_five_consecutive_numbers_not_square_l341_34107

theorem product_of_five_consecutive_numbers_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) :=
by
  sorry

end NUMINAMATH_GPT_product_of_five_consecutive_numbers_not_square_l341_34107


namespace NUMINAMATH_GPT_susan_arrives_before_sam_by_14_minutes_l341_34135

theorem susan_arrives_before_sam_by_14_minutes (d : ℝ) (susan_speed sam_speed : ℝ) (h1 : d = 2) (h2 : susan_speed = 12) (h3 : sam_speed = 5) : 
  let susan_time := d / susan_speed
  let sam_time := d / sam_speed
  let susan_minutes := susan_time * 60
  let sam_minutes := sam_time * 60
  sam_minutes - susan_minutes = 14 := 
by
  sorry

end NUMINAMATH_GPT_susan_arrives_before_sam_by_14_minutes_l341_34135


namespace NUMINAMATH_GPT_total_tickets_sold_l341_34189

theorem total_tickets_sold (A C : ℕ) (hC : C = 16) (h1 : 3 * C = 48) (h2 : 5 * A + 3 * C = 178) : 
  A + C = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l341_34189


namespace NUMINAMATH_GPT_total_cost_for_tickets_l341_34143

-- Define the known quantities
def students : Nat := 20
def teachers : Nat := 3
def ticket_cost : Nat := 5

-- Define the total number of people
def total_people : Nat := students + teachers

-- Define the total cost
def total_cost : Nat := total_people * ticket_cost

-- Prove that the total cost is $115
theorem total_cost_for_tickets : total_cost = 115 := by
  -- Sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_total_cost_for_tickets_l341_34143


namespace NUMINAMATH_GPT_train_speed_ratio_l341_34108

theorem train_speed_ratio 
  (v_A v_B : ℝ)
  (h1 : v_A = 2 * v_B)
  (h2 : 27 = L_A / v_A)
  (h3 : 17 = L_B / v_B)
  (h4 : 22 = (L_A + L_B) / (v_A + v_B))
  (h5 : v_A + v_B ≤ 60) :
  v_A / v_B = 2 := by
  sorry

-- Conditions given must be defined properly
variables (L_A L_B : ℝ)

end NUMINAMATH_GPT_train_speed_ratio_l341_34108


namespace NUMINAMATH_GPT_range_of_a_l341_34115

noncomputable def f : ℝ → ℝ → ℝ
| a, x =>
  if x ≥ -1 then a * x ^ 2 + 2 * x 
  else (1 - 3 * a) * x - 3 / 2

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) → 0 < a ∧ a ≤ 1/4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l341_34115


namespace NUMINAMATH_GPT_max_frac_a_c_squared_l341_34106

theorem max_frac_a_c_squared 
  (a b c : ℝ) (y z : ℝ)
  (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order: a ≥ b ∧ b ≥ c)
  (h_system: a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2)
  (h_bounds: 0 ≤ y ∧ y < a ∧ 0 ≤ z ∧ z < c) :
  (a/c)^2 ≤ 4/3 :=
sorry

end NUMINAMATH_GPT_max_frac_a_c_squared_l341_34106


namespace NUMINAMATH_GPT_initial_geese_count_l341_34181

theorem initial_geese_count (G : ℕ) (h1 : G / 2 + 4 = 12) : G = 16 := by
  sorry

end NUMINAMATH_GPT_initial_geese_count_l341_34181


namespace NUMINAMATH_GPT_union_of_M_and_N_l341_34123

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_of_M_and_N : M ∪ N = {x | -1 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_GPT_union_of_M_and_N_l341_34123


namespace NUMINAMATH_GPT_least_number_of_shoes_l341_34151

theorem least_number_of_shoes (num_inhabitants : ℕ) 
  (one_legged_percentage : ℚ) 
  (barefooted_proportion : ℚ) 
  (h_num_inhabitants : num_inhabitants = 10000) 
  (h_one_legged_percentage : one_legged_percentage = 0.05) 
  (h_barefooted_proportion : barefooted_proportion = 0.5) : 
  ∃ (shoes_needed : ℕ), shoes_needed = 10000 := 
by
  sorry

end NUMINAMATH_GPT_least_number_of_shoes_l341_34151


namespace NUMINAMATH_GPT_paperback_copies_sold_l341_34177

theorem paperback_copies_sold
  (H P : ℕ)
  (h1 : H = 36000)
  (h2 : H + P = 440000) :
  P = 404000 :=
by
  rw [h1] at h2
  sorry

end NUMINAMATH_GPT_paperback_copies_sold_l341_34177


namespace NUMINAMATH_GPT_negation_proposition_equivalence_l341_34174

theorem negation_proposition_equivalence :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0)) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negation_proposition_equivalence_l341_34174


namespace NUMINAMATH_GPT_total_work_stations_l341_34170

theorem total_work_stations (total_students : ℕ) (stations_for_2 : ℕ) (stations_for_3 : ℕ)
  (h1 : total_students = 38)
  (h2 : stations_for_2 = 10)
  (h3 : 20 + 3 * stations_for_3 = total_students) :
  stations_for_2 + stations_for_3 = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_work_stations_l341_34170


namespace NUMINAMATH_GPT_T_perimeter_l341_34117

theorem T_perimeter (l w : ℝ) (h1 : l = 4) (h2 : w = 2) :
  let rect_perimeter := 2 * l + 2 * w
  let overlap := 2 * w
  2 * rect_perimeter - overlap = 20 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_T_perimeter_l341_34117


namespace NUMINAMATH_GPT_inequality_proof_l341_34158

open Real

variable (a b c : ℝ)

theorem inequality_proof
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c) :
  sqrt (a * b * c) * (sqrt a + sqrt b + sqrt c) + (a + b + c) ^ 2 ≥ 
  4 * sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l341_34158


namespace NUMINAMATH_GPT_sum_ratio_l341_34175

def arithmetic_sequence (a_1 d : ℚ) (n : ℕ) : ℚ :=
  a_1 + (n - 1) * d

def S_n (a_1 d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a_1 + (n - 1) * d) / 2 -- sum of first n terms of arithmetic sequence

theorem sum_ratio (a_1 d : ℚ) (h : 13 * (a_1 + 6 * d) = 7 * (a_1 + 3 * d)) :
  S_n a_1 d 13 / S_n a_1 d 7 = 1 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_ratio_l341_34175


namespace NUMINAMATH_GPT_janina_must_sell_21_pancakes_l341_34102

/-- The daily rent cost for Janina. -/
def daily_rent := 30

/-- The daily supply cost for Janina. -/
def daily_supplies := 12

/-- The cost of a single pancake. -/
def pancake_price := 2

/-- The total daily expenses for Janina. -/
def total_daily_expenses := daily_rent + daily_supplies

/-- The required number of pancakes Janina needs to sell each day to cover her expenses. -/
def required_pancakes := total_daily_expenses / pancake_price

theorem janina_must_sell_21_pancakes :
  required_pancakes = 21 :=
sorry

end NUMINAMATH_GPT_janina_must_sell_21_pancakes_l341_34102


namespace NUMINAMATH_GPT_part1_part2_l341_34145

noncomputable def f (x : ℝ) : ℝ := Real.cos x * Real.cos (x - Real.pi / 3)

theorem part1 : f (2 * Real.pi / 3) = -1 / 4 :=
by
  sorry

theorem part2 : {x | f x < 1 / 4} = { x | ∃ k : ℤ, k * Real.pi + 5 * Real.pi / 12 < x ∧ x < k * Real.pi + 11 * Real.pi / 12 } :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l341_34145


namespace NUMINAMATH_GPT_adam_earnings_after_taxes_l341_34171

theorem adam_earnings_after_taxes
  (daily_earnings : ℕ) 
  (tax_pct : ℕ)
  (workdays : ℕ)
  (H1 : daily_earnings = 40) 
  (H2 : tax_pct = 10) 
  (H3 : workdays = 30) : 
  (daily_earnings - daily_earnings * tax_pct / 100) * workdays = 1080 := 
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_adam_earnings_after_taxes_l341_34171


namespace NUMINAMATH_GPT_polynomials_equality_l341_34137

open Polynomial

variable {F : Type*} [Field F]

theorem polynomials_equality (P Q : Polynomial F) (h : ∀ x, P.eval (P.eval (P.eval x)) = Q.eval (Q.eval (Q.eval x)) ∧ P.eval (P.eval (P.eval x)) = Q.eval (P.eval (P.eval x))) : 
  P = Q := 
sorry

end NUMINAMATH_GPT_polynomials_equality_l341_34137


namespace NUMINAMATH_GPT_polynomial_coeffs_l341_34197

theorem polynomial_coeffs (a b c d e f : ℤ) :
  (((2 : ℤ) * x - 1) ^ 5 = a * x ^ 5 + b * x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x + f) →
  (a + b + c + d + e + f = 1) ∧ 
  (b + c + d + e = -30) ∧
  (a + c + e = 122) :=
by
  intro h
  sorry  -- Proof omitted

end NUMINAMATH_GPT_polynomial_coeffs_l341_34197


namespace NUMINAMATH_GPT_find_E_coordinates_l341_34192

structure Point :=
(x : ℚ)
(y : ℚ)

def A : Point := { x := -2, y := 1 }
def B : Point := { x := 1, y := 4 }
def C : Point := { x := 4, y := -3 }

def D : Point := 
  let m : ℚ := 1
  let n : ℚ := 2
  let x1 := A.x
  let y1 := A.y
  let x2 := B.x
  let y2 := B.y
  { x := (m * x2 + n * x1) / (m + n), y := (m * y2 + n * y1) / (m + n) }

theorem find_E_coordinates : 
  let k : ℚ := 4
  let x_E : ℚ := (k * C.x + D.x) / (k + 1)
  let y_E : ℚ := (k * C.y + D.y) / (k + 1)
  ∃ E : Point, E.x = (17:ℚ) / 3 ∧ E.y = -(14:ℚ) / 3 :=
sorry

end NUMINAMATH_GPT_find_E_coordinates_l341_34192


namespace NUMINAMATH_GPT_greatest_possible_value_of_n_greatest_possible_value_of_10_l341_34142

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 :=
by
  sorry

theorem greatest_possible_value_of_10 (n : ℤ) (h : 101 * n^2 ≤ 12100) : n = 10 → n = 10 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_of_n_greatest_possible_value_of_10_l341_34142


namespace NUMINAMATH_GPT_range_of_a_l341_34110

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x0 : ℝ, x0^2 + 2 * x0 + a ≤ 0
def q (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a

-- The theorem statement: if p is false and q is true, then 1 < a < 2
theorem range_of_a (a : ℝ) (h1 : ¬ p a) (h2 : q a) : 1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l341_34110


namespace NUMINAMATH_GPT_percentage_calculation_l341_34111

theorem percentage_calculation (Part Whole : ℕ) (h1 : Part = 90) (h2 : Whole = 270) : 
  ((Part : ℝ) / (Whole : ℝ) * 100) = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_percentage_calculation_l341_34111


namespace NUMINAMATH_GPT_simplify_fraction_l341_34128

theorem simplify_fraction (x : ℝ) : (3*x + 2) / 4 + (x - 4) / 3 = (13*x - 10) / 12 := sorry

end NUMINAMATH_GPT_simplify_fraction_l341_34128


namespace NUMINAMATH_GPT_imaginary_unit_calculation_l341_34121

theorem imaginary_unit_calculation (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i := 
by
  sorry

end NUMINAMATH_GPT_imaginary_unit_calculation_l341_34121


namespace NUMINAMATH_GPT_a_2_value_general_terms_T_n_value_l341_34185

-- Definitions based on conditions
def S (n : ℕ) : ℕ := sorry  -- Sum of the first n terms of sequence {a_n}

def a (n : ℕ) : ℕ := (S n + 2) / 2  -- a_n is the arithmetic mean of S_n and 2

def b (n : ℕ) : ℕ := 2 * n - 1  -- Given general term for b_n

-- Prove a_2 = 4
theorem a_2_value : a 2 = 4 := 
by
  sorry

-- Prove the general terms
theorem general_terms (n : ℕ) : a n = 2^n ∧ b n = 2 * n - 1 := 
by
  sorry

-- Definition and sum of the first n terms of c_n
def c (n : ℕ) : ℕ := a n * b n

def T (n : ℕ) : ℕ := (2 * n - 3) * 2^(n + 1) + 6  -- Given sum of the first n terms of {c_n}

-- Prove T_n = (2n - 3)2^(n+1) + 6
theorem T_n_value (n : ℕ) : T n = (2 * n - 3) * 2^(n + 1) + 6 :=
by
  sorry

end NUMINAMATH_GPT_a_2_value_general_terms_T_n_value_l341_34185


namespace NUMINAMATH_GPT_radius_of_small_semicircle_l341_34144

theorem radius_of_small_semicircle
  (radius_big_semicircle : ℝ)
  (radius_inner_circle : ℝ) 
  (pairwise_tangent : ∀ x : ℝ, x = radius_big_semicircle → x = 12 ∧ 
                                x = radius_inner_circle → x = 6 ∧ 
                                true) :
  ∃ (r : ℝ), r = 4 :=
by 
  sorry

end NUMINAMATH_GPT_radius_of_small_semicircle_l341_34144


namespace NUMINAMATH_GPT_lauren_annual_income_l341_34162

open Real

theorem lauren_annual_income (p : ℝ) (A : ℝ) (T : ℝ) :
  (T = (p + 0.45)/100 * A) →
  (T = (p/100) * 20000 + ((p + 1)/100) * 15000 + ((p + 3)/100) * (A - 35000)) →
  A = 36000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lauren_annual_income_l341_34162


namespace NUMINAMATH_GPT_find_constants_l341_34157

theorem find_constants
  (a_1 a_2 : ℚ)
  (h1 : 3 * a_1 - 3 * a_2 = 0)
  (h2 : 4 * a_1 + 7 * a_2 = 5) :
  a_1 = 5 / 11 ∧ a_2 = 5 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l341_34157


namespace NUMINAMATH_GPT_options_not_equal_l341_34164

theorem options_not_equal (a b c d e : ℚ)
  (ha : a = 14 / 10)
  (hb : b = 1 + 2 / 5)
  (hc : c = 1 + 7 / 25)
  (hd : d = 1 + 2 / 10)
  (he : e = 1 + 14 / 70) :
  a = 7 / 5 ∧ b = 7 / 5 ∧ c ≠ 7 / 5 ∧ d ≠ 7 / 5 ∧ e ≠ 7 / 5 :=
by sorry

end NUMINAMATH_GPT_options_not_equal_l341_34164


namespace NUMINAMATH_GPT_quadratic_solution_is_unique_l341_34178

theorem quadratic_solution_is_unique (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : 2 * p + q / 2 = -p)
  (h2 : 2 * p * (q / 2) = q) :
  (p, q) = (1, -6) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_is_unique_l341_34178


namespace NUMINAMATH_GPT_vector_dot_product_identity_l341_34190

-- Define the vectors a, b, and c in ℝ²
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-3, 1)

-- Define vector addition and dot product in ℝ²
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that c · (a + b) = 9
theorem vector_dot_product_identity : dot_product c (vector_add a b) = 9 := 
by 
sorry

end NUMINAMATH_GPT_vector_dot_product_identity_l341_34190
