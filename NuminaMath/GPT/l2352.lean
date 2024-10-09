import Mathlib

namespace ferry_captives_successfully_l2352_235235

-- Definition of conditions
def valid_trip_conditions (trips: ℕ) (captives: ℕ) : Prop :=
  captives = 43 ∧
  (∀ k < trips, k % 2 = 0 ∨ k % 2 = 1) ∧     -- Trips done in pairs or singles
  (∀ k < captives, k > 40)                    -- At least 40 other captives known as werewolves

-- Theorem statement to be proved
theorem ferry_captives_successfully (trips : ℕ) (captives : ℕ) (result : Prop) : 
  valid_trip_conditions trips captives → result = true := by sorry

end ferry_captives_successfully_l2352_235235


namespace garbage_classification_competition_l2352_235240

theorem garbage_classification_competition :
  let boy_rate_seventh := 0.4
  let boy_rate_eighth := 0.5
  let girl_rate_seventh := 0.6
  let girl_rate_eighth := 0.7
  let combined_boy_rate := (boy_rate_seventh + boy_rate_eighth) / 2
  let combined_girl_rate := (girl_rate_seventh + girl_rate_eighth) / 2
  boy_rate_seventh < boy_rate_eighth ∧ combined_boy_rate < combined_girl_rate :=
by {
  sorry
}

end garbage_classification_competition_l2352_235240


namespace mom_twice_alex_l2352_235222

-- Definitions based on the conditions
def alex_age_in_2010 : ℕ := 10
def mom_age_in_2010 : ℕ := 5 * alex_age_in_2010
def future_years_after_2010 (x : ℕ) : ℕ := 2010 + x

-- Defining the ages in the future year
def alex_age_future (x : ℕ) : ℕ := alex_age_in_2010 + x
def mom_age_future (x : ℕ) : ℕ := mom_age_in_2010 + x

-- The theorem to prove
theorem mom_twice_alex (x : ℕ) (h : mom_age_future x = 2 * alex_age_future x) : future_years_after_2010 x = 2040 :=
  by
  sorry

end mom_twice_alex_l2352_235222


namespace combined_books_total_l2352_235200

def keith_books : ℕ := 20
def jason_books : ℕ := 21
def amanda_books : ℕ := 15
def sophie_books : ℕ := 30

def total_books := keith_books + jason_books + amanda_books + sophie_books

theorem combined_books_total : total_books = 86 := 
by sorry

end combined_books_total_l2352_235200


namespace ratio_eq_23_over_28_l2352_235215

theorem ratio_eq_23_over_28 (a b : ℚ) (h : (12 * a - 5 * b) / (14 * a - 3 * b) = 4 / 7) : 
  a / b = 23 / 28 := 
sorry

end ratio_eq_23_over_28_l2352_235215


namespace functional_equation_solution_l2352_235227

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution : 
  (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) 
  → ∃ c : ℝ, (c = 0 ∨ (1 ≤ c ∧ c < 2)) ∧ (∀ x : ℝ, f x = c) :=
by
  intro h
  sorry

end functional_equation_solution_l2352_235227


namespace fn_prime_factor_bound_l2352_235291

theorem fn_prime_factor_bound (n : ℕ) (h : n ≥ 3) : 
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^(2^n) + 1)) ∧ p > 2^(n+2) * (n+1) :=
sorry

end fn_prime_factor_bound_l2352_235291


namespace ordering_of_exponentiations_l2352_235245

def a : ℕ := 3 ^ 34
def b : ℕ := 2 ^ 51
def c : ℕ := 4 ^ 25

theorem ordering_of_exponentiations : c < b ∧ b < a := by
  sorry

end ordering_of_exponentiations_l2352_235245


namespace cos_pi_minus_alpha_cos_double_alpha_l2352_235284

open Real

theorem cos_pi_minus_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (π - α) = - sqrt 7 / 3 :=
by
  sorry

theorem cos_double_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (2 * α) = 5 / 9 :=
by
  sorry

end cos_pi_minus_alpha_cos_double_alpha_l2352_235284


namespace danny_bottle_caps_l2352_235249

theorem danny_bottle_caps 
  (wrappers_park : Nat := 46)
  (caps_park : Nat := 50)
  (wrappers_collection : Nat := 52)
  (more_caps_than_wrappers : Nat := 4)
  (h1 : caps_park = wrappers_park + more_caps_than_wrappers)
  (h2 : wrappers_collection = 52) : 
  (∃ initial_caps : Nat, initial_caps + caps_park = wrappers_collection + more_caps_than_wrappers) :=
by 
  use 6
  sorry

end danny_bottle_caps_l2352_235249


namespace not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l2352_235204

theorem not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles : 
  ¬ ∃ (rectangles : ℕ × ℕ), rectangles.1 = 1 ∧ rectangles.2 = 7 ∧ rectangles.1 * 4 + rectangles.2 * 3 = 25 :=
by
  sorry

end not_possible_to_cut_5x5_square_into_1x4_and_1x3_rectangles_l2352_235204


namespace fiftieth_term_arithmetic_seq_l2352_235282

theorem fiftieth_term_arithmetic_seq : 
  (∀ (n : ℕ), (2 + (n - 1) * 5) = 247) := by
  sorry

end fiftieth_term_arithmetic_seq_l2352_235282


namespace function_passes_through_fixed_point_l2352_235263

variable (a : ℝ)

theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) : (1, 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1) + 1)} :=
by
  sorry

end function_passes_through_fixed_point_l2352_235263


namespace probability_of_sum_odd_is_correct_l2352_235268

noncomputable def probability_sum_odd : ℚ :=
  let total_balls := 13
  let drawn_balls := 7
  let total_ways := Nat.choose total_balls drawn_balls
  let favorable_ways := 
    Nat.choose 7 5 * Nat.choose 6 2 + 
    Nat.choose 7 3 * Nat.choose 6 4 + 
    Nat.choose 7 1 * Nat.choose 6 6
  favorable_ways / total_ways

theorem probability_of_sum_odd_is_correct :
  probability_sum_odd = 847 / 1716 :=
by
  -- Proof goes here
  sorry

end probability_of_sum_odd_is_correct_l2352_235268


namespace probability_sibling_pair_l2352_235260

-- Define the necessary constants for the problem.
def B : ℕ := 500 -- Number of business students
def L : ℕ := 800 -- Number of law students
def S : ℕ := 30  -- Number of sibling pairs

-- State the theorem representing the mathematical proof problem
theorem probability_sibling_pair :
  (S : ℝ) / (B * L) = 0.000075 := sorry

end probability_sibling_pair_l2352_235260


namespace find_number_l2352_235244

theorem find_number (x : ℕ) (h : x = 4) : x + 1 = 5 :=
by
  sorry

end find_number_l2352_235244


namespace isosceles_triangle_base_l2352_235294

theorem isosceles_triangle_base (a b c : ℕ) (h_isosceles : a = b ∨ a = c ∨ b = c)
  (h_perimeter : a + b + c = 29) (h_side : a = 7 ∨ b = 7 ∨ c = 7) : 
  a = 7 ∨ b = 7 ∨ c = 7 ∧ (a = 7 ∨ a = 11) ∧ (b = 7 ∨ b = 11) ∧ (c = 7 ∨ c = 11) ∧ (a ≠ b ∨ c ≠ b) :=
by
  sorry

end isosceles_triangle_base_l2352_235294


namespace fraction_of_Charlie_circumference_l2352_235272

/-- Definitions for the problem conditions -/
def Jack_head_circumference : ℕ := 12
def Charlie_head_circumference : ℕ := 9 + Jack_head_circumference / 2
def Bill_head_circumference : ℕ := 10

/-- Statement of the theorem to be proved -/
theorem fraction_of_Charlie_circumference :
  Bill_head_circumference / Charlie_head_circumference = 2 / 3 :=
sorry

end fraction_of_Charlie_circumference_l2352_235272


namespace smallest_m_l2352_235288

theorem smallest_m (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n - m / n = 2011 / 3) : m = 1120 :=
sorry

end smallest_m_l2352_235288


namespace intersection_A_C_U_B_l2352_235247

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 0}
def C_U_B : Set ℝ := {x | ¬ (Real.log x / Real.log 2 > 0)}

theorem intersection_A_C_U_B :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_A_C_U_B_l2352_235247


namespace quadratic_two_real_roots_find_m_l2352_235226

theorem quadratic_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ * x₂ = 3 * m^2 ∧ x₁ + x₂ = 4 * m :=
by
  sorry

theorem find_m (m : ℝ) (h : m > 0) (h_diff : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ - x₂ = 2) : m = 1 :=
by
  sorry

end quadratic_two_real_roots_find_m_l2352_235226


namespace harold_august_tips_fraction_l2352_235203

noncomputable def tips_fraction : ℚ :=
  let A : ℚ := sorry -- average monthly tips for March to July and September
  let august_tips := 6 * A -- Tips for August
  let total_tips := 6 * A + 6 * A -- Total tips for all months worked
  august_tips / total_tips

theorem harold_august_tips_fraction :
  tips_fraction = 1 / 2 :=
by
  sorry

end harold_august_tips_fraction_l2352_235203


namespace student_incorrect_answer_l2352_235278

theorem student_incorrect_answer (D I : ℕ) (h1 : D / 63 = I) (h2 : D / 36 = 42) : I = 24 := by
  sorry

end student_incorrect_answer_l2352_235278


namespace solve_for_x_l2352_235275

theorem solve_for_x (x : ℝ) (h : 2 * x - 3 = 6 - x) : x = 3 :=
by
  sorry

end solve_for_x_l2352_235275


namespace price_of_first_shirt_l2352_235223

theorem price_of_first_shirt
  (price1 price2 price3 : ℕ)
  (total_shirts : ℕ)
  (min_avg_price_of_remaining : ℕ)
  (total_avg_price_of_all : ℕ)
  (prices_of_first_3 : price1 = 100 ∧ price2 = 90 ∧ price3 = 82)
  (condition1 : total_shirts = 10)
  (condition2 : min_avg_price_of_remaining = 104)
  (condition3 : total_avg_price_of_all > 100) :
  price1 = 100 :=
by
  sorry

end price_of_first_shirt_l2352_235223


namespace base6_divisible_19_l2352_235293

theorem base6_divisible_19 (y : ℤ) : (19 ∣ (615 + 6 * y)) ↔ y = 2 := sorry

end base6_divisible_19_l2352_235293


namespace find_tangent_circle_l2352_235286

-- Define circles and line
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def line_l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the problem statement as a theorem
theorem find_tangent_circle :
  ∃ (x0 y0 : ℝ), (x - x0)^2 + (y - y0)^2 = 5/4 ∧ (x0, y0) = (1/2, 1) ∧
                   ∀ (x y : ℝ), (circle1 x y → circle2 x y → line_l (x0 + x) (y0 + y) ) :=
sorry

end find_tangent_circle_l2352_235286


namespace intersect_lines_l2352_235283

theorem intersect_lines (k : ℝ) :
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 :=
by
  sorry

end intersect_lines_l2352_235283


namespace polynomial_difference_of_squares_l2352_235237

theorem polynomial_difference_of_squares (x y : ℤ) :
  8 * x^2 + 2 * x * y - 3 * y^2 = (3 * x - y)^2 - (x + 2 * y)^2 :=
by
  sorry

end polynomial_difference_of_squares_l2352_235237


namespace value_of_expression_l2352_235277

theorem value_of_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end value_of_expression_l2352_235277


namespace sum_greater_than_3_l2352_235230

theorem sum_greater_than_3 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a > a + b + c) : a + b + c > 3 :=
sorry

end sum_greater_than_3_l2352_235230


namespace range_of_a_l2352_235297

def f (x : ℝ) : ℝ := -x^5 - 3 * x^3 - 5 * x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 :=
by
  -- Here, we would have to show the proof, but we're skipping it
  sorry

end range_of_a_l2352_235297


namespace permutations_of_six_attractions_is_720_l2352_235246

-- Define the number of attractions
def num_attractions : ℕ := 6

-- State the theorem to be proven
theorem permutations_of_six_attractions_is_720 : (num_attractions.factorial = 720) :=
by {
  sorry
}

end permutations_of_six_attractions_is_720_l2352_235246


namespace gcd_1911_1183_l2352_235224

theorem gcd_1911_1183 : gcd 1911 1183 = 91 :=
by sorry

end gcd_1911_1183_l2352_235224


namespace num_letters_dot_not_straight_line_l2352_235251

variable (Total : ℕ)
variable (DS : ℕ)
variable (S_only : ℕ)
variable (D_only : ℕ)

theorem num_letters_dot_not_straight_line 
  (h1 : Total = 40) 
  (h2 : DS = 11) 
  (h3 : S_only = 24) 
  (h4 : Total - S_only - DS = D_only) : 
  D_only = 5 := 
by 
  sorry

end num_letters_dot_not_straight_line_l2352_235251


namespace sale_in_fifth_month_l2352_235270

def sale_first_month : ℝ := 3435
def sale_second_month : ℝ := 3927
def sale_third_month : ℝ := 3855
def sale_fourth_month : ℝ := 4230
def required_avg_sale : ℝ := 3500
def sale_sixth_month : ℝ := 1991

theorem sale_in_fifth_month :
  (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + s + sale_sixth_month) / 6 = required_avg_sale ->
  s = 3562 :=
by
  sorry

end sale_in_fifth_month_l2352_235270


namespace solid_is_triangular_prism_l2352_235258

-- Given conditions as definitions
def front_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the front view is an isosceles triangle
  sorry

def left_view_is_isosceles_triangle (solid : Type) : Prop := 
  -- Define the property that the left view is an isosceles triangle
  sorry

def top_view_is_circle (solid : Type) : Prop := 
   -- Define the property that the top view is a circle
  sorry

-- Define the property of being a triangular prism
def is_triangular_prism (solid : Type) : Prop :=
  -- Define the property that the solid is a triangular prism
  sorry

-- The main theorem: proving that given the conditions, the solid could be a triangular prism
theorem solid_is_triangular_prism (solid : Type) :
  front_view_is_isosceles_triangle solid ∧ 
  left_view_is_isosceles_triangle solid ∧ 
  top_view_is_circle solid →
  is_triangular_prism solid :=
sorry

end solid_is_triangular_prism_l2352_235258


namespace sum_first_10_terms_l2352_235225

def a_n (n : ℕ) : ℤ := (-1)^n * (3 * n - 2)

theorem sum_first_10_terms :
  (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) +
  (a_n 6) + (a_n 7) + (a_n 8) + (a_n 9) + (a_n 10) = 15 :=
by
  sorry

end sum_first_10_terms_l2352_235225


namespace team_a_score_l2352_235280

theorem team_a_score : ∀ (A : ℕ), A + 9 + 4 = 15 → A = 2 :=
by
  intros A h
  sorry

end team_a_score_l2352_235280


namespace problem_circumscribing_sphere_surface_area_l2352_235201

noncomputable def surface_area_of_circumscribing_sphere (a b c : ℕ) :=
  let R := (Real.sqrt (a^2 + b^2 + c^2)) / 2
  4 * Real.pi * R^2

theorem problem_circumscribing_sphere_surface_area
  (a b c : ℕ)
  (ha : (1 / 2 : ℝ) * a * b = 4)
  (hb : (1 / 2 : ℝ) * b * c = 6)
  (hc : (1 / 2: ℝ) * a * c = 12) : 
  surface_area_of_circumscribing_sphere a b c = 56 * Real.pi := 
sorry

end problem_circumscribing_sphere_surface_area_l2352_235201


namespace Dan_gave_Sara_limes_l2352_235208

theorem Dan_gave_Sara_limes : 
  ∀ (original_limes now_limes given_limes : ℕ),
  original_limes = 9 →
  now_limes = 5 →
  given_limes = original_limes - now_limes →
  given_limes = 4 :=
by
  intros original_limes now_limes given_limes h1 h2 h3
  sorry

end Dan_gave_Sara_limes_l2352_235208


namespace red_ants_count_l2352_235242

def total_ants : ℕ := 900
def black_ants : ℕ := 487
def red_ants (r : ℕ) : Prop := r + black_ants = total_ants

theorem red_ants_count : ∃ r : ℕ, red_ants r ∧ r = 413 := 
sorry

end red_ants_count_l2352_235242


namespace simplify_expression_l2352_235231

theorem simplify_expression : 2 - Real.sqrt 3 + 1 / (2 - Real.sqrt 3) + 1 / (Real.sqrt 3 + 2) = 6 :=
by
  sorry

end simplify_expression_l2352_235231


namespace gain_percentage_l2352_235213

theorem gain_percentage (C S : ℝ) (h : 80 * C = 25 * S) : 220 = ((S - C) / C) * 100 :=
by sorry

end gain_percentage_l2352_235213


namespace find_x_l2352_235276

theorem find_x (x y : ℤ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y + x * y = 104) : x = 34 :=
sorry

end find_x_l2352_235276


namespace sum_of_largest_and_smallest_is_correct_l2352_235238

-- Define the set of digits
def digits : Finset ℕ := {2, 0, 4, 1, 5, 8}

-- Define the largest possible number using the digits
def largestNumber : ℕ := 854210

-- Define the smallest possible number using the digits
def smallestNumber : ℕ := 102458

-- Define the sum of largest and smallest possible numbers
def sumOfNumbers : ℕ := largestNumber + smallestNumber

-- Main theorem to prove
theorem sum_of_largest_and_smallest_is_correct : sumOfNumbers = 956668 := by
  sorry

end sum_of_largest_and_smallest_is_correct_l2352_235238


namespace problem_statement_l2352_235234

-- Defining the sets U, M, and N
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

-- Complement of N in U
def complement_U_N : Set ℕ := U \ N

-- Problem statement
theorem problem_statement : M ∩ complement_U_N = {0, 3} :=
by
  sorry

end problem_statement_l2352_235234


namespace triangle_altitude_l2352_235256

theorem triangle_altitude (base side : ℝ) (h : ℝ) : 
  side = 6 → base = 6 → 
  (base * h) / 2 = side ^ 2 → 
  h = 12 :=
by
  intros
  sorry

end triangle_altitude_l2352_235256


namespace greatest_positive_multiple_of_4_l2352_235255

theorem greatest_positive_multiple_of_4 {y : ℕ} (h1 : y % 4 = 0) (h2 : y > 0) (h3 : y^3 < 8000) : y ≤ 16 :=
by {
  -- The proof will go here
  -- Sorry is placed here to skip the proof for now
  sorry
}

end greatest_positive_multiple_of_4_l2352_235255


namespace petya_addition_mistake_l2352_235202

theorem petya_addition_mistake:
  ∃ (x y c : ℕ), x + y = 12345 ∧ (10 * x + c) + y = 44444 ∧ x = 3566 ∧ y = 8779 ∧ c = 5 := by
  sorry

end petya_addition_mistake_l2352_235202


namespace smoothie_cost_l2352_235271

-- Definitions of costs and amounts paid.
def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def amount_paid : ℕ := 20
def change_received : ℕ := 11

-- Define the total cost of the order and the known costs.
def total_order_cost : ℕ := amount_paid - change_received
def known_costs : ℕ := hamburger_cost + onion_rings_cost

-- State the problem: the cost of the smoothie.
theorem smoothie_cost : total_order_cost - known_costs = 3 :=
by 
  sorry

end smoothie_cost_l2352_235271


namespace Punta_position_l2352_235287

theorem Punta_position (N x y p : ℕ) (h1 : N = 36) (h2 : x = y / 4) (h3 : x + y = 35) : p = 8 := by
  sorry

end Punta_position_l2352_235287


namespace complement_union_l2352_235281

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_union :
  U \ (M ∪ N) = {4} :=
by
  sorry

end complement_union_l2352_235281


namespace technicians_count_l2352_235243

-- Define the conditions
def avg_sal_all (total_workers : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 850

def avg_sal_technicians (teches : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 1000

def avg_sal_rest (others : ℕ) (avg_salary : ℕ) : Prop :=
  avg_salary = 780

-- The main theorem to prove
theorem technicians_count (total_workers : ℕ)
  (teches others : ℕ)
  (total_salary : ℕ) :
  total_workers = 22 →
  total_salary = 850 * 22 →
  avg_sal_all total_workers 850 →
  avg_sal_technicians teches 1000 →
  avg_sal_rest others 780 →
  teches + others = total_workers →
  1000 * teches + 780 * others = total_salary →
  teches = 7 :=
by
  intros
  sorry

end technicians_count_l2352_235243


namespace mario_pizza_area_l2352_235254

theorem mario_pizza_area
  (pizza_area : ℝ)
  (cut_distance : ℝ)
  (largest_piece : ℝ)
  (smallest_piece : ℝ)
  (total_pieces : ℕ)
  (pieces_mario_gets_area : ℝ) :
  pizza_area = 4 →
  cut_distance = 0.5 →
  total_pieces = 4 →
  pieces_mario_gets_area = (pizza_area - (largest_piece + smallest_piece)) / 2 →
  pieces_mario_gets_area = 1.5 :=
sorry

end mario_pizza_area_l2352_235254


namespace ben_paperclip_day_l2352_235274

theorem ben_paperclip_day :
  ∃ k : ℕ, k = 6 ∧ (∀ n : ℕ, n = k → 5 * 3^n > 500) :=
sorry

end ben_paperclip_day_l2352_235274


namespace largest_common_term_up_to_150_l2352_235292

theorem largest_common_term_up_to_150 :
  ∃ a : ℕ, a ≤ 150 ∧ (∃ n : ℕ, a = 2 + 8 * n) ∧ (∃ m : ℕ, a = 3 + 9 * m) ∧ (∀ b : ℕ, b ≤ 150 → (∃ n' : ℕ, b = 2 + 8 * n') → (∃ m' : ℕ, b = 3 + 9 * m') → b ≤ a) := 
sorry

end largest_common_term_up_to_150_l2352_235292


namespace high_sulfur_oil_samples_l2352_235239

/-- The number of high-sulfur oil samples in a container with the given conditions. -/
theorem high_sulfur_oil_samples (total_samples : ℕ) 
    (heavy_oil_freq : ℚ) (light_low_sulfur_freq : ℚ)
    (no_heavy_low_sulfur: true) (almost_full : total_samples = 198)
    (heavy_oil_freq_value : heavy_oil_freq = 1 / 9)
    (light_low_sulfur_freq_value : light_low_sulfur_freq = 11 / 18) :
    (22 + 68) = 90 := 
by
  sorry

end high_sulfur_oil_samples_l2352_235239


namespace Amy_crumbs_l2352_235269

variable (z : ℕ)

theorem Amy_crumbs (T C : ℕ) (h1 : T * C = z)
  (h2 : ∃ T_A : ℕ, T_A = 2 * T)
  (h3 : ∃ C_A : ℕ, C_A = (3 * C) / 2) :
  ∃ z_A : ℕ, z_A = 3 * z :=
by
  sorry

end Amy_crumbs_l2352_235269


namespace compute_multiplied_difference_l2352_235221

theorem compute_multiplied_difference (a b : ℕ) (h_a : a = 25) (h_b : b = 15) :
  3 * ((a + b) ^ 2 - (a - b) ^ 2) = 4500 := by
  sorry

end compute_multiplied_difference_l2352_235221


namespace sandy_correct_sums_l2352_235217

theorem sandy_correct_sums :
  ∃ c i : ℤ,
  c + i = 40 ∧
  4 * c - 3 * i = 72 ∧
  c = 27 :=
by 
  sorry

end sandy_correct_sums_l2352_235217


namespace find_k_l2352_235218

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - y = 9 * k) (h3 : x - 2 * y = 22) : k = 2 :=
by
  sorry

end find_k_l2352_235218


namespace smallest_three_digit_number_with_property_l2352_235220

theorem smallest_three_digit_number_with_property : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ 
  (∀ d, (1 ≤ d ∧ d ≤ 1000) → ((d = n + 1 ∨ d = n - 1) → d % 11 = 0)) ∧ 
  n = 120 :=
by
  sorry

end smallest_three_digit_number_with_property_l2352_235220


namespace largest_fraction_l2352_235261

theorem largest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 5)
                                          (h2 : f2 = 3 / 6)
                                          (h3 : f3 = 5 / 10)
                                          (h4 : f4 = 7 / 15)
                                          (h5 : f5 = 8 / 20) : 
  (f2 = 1 / 2 ∨ f3 = 1 / 2) ∧ (f2 ≥ f1 ∧ f2 ≥ f4 ∧ f2 ≥ f5) ∧ (f3 ≥ f1 ∧ f3 ≥ f4 ∧ f3 ≥ f5) := 
by
  sorry

end largest_fraction_l2352_235261


namespace problem_eq_solution_l2352_235296

variables (a b x y : ℝ)

theorem problem_eq_solution
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : a + b + x + y < 2)
  (h6 : a + b^2 = x + y^2)
  (h7 : a^2 + b = x^2 + y) :
  a = x ∧ b = y :=
by
  sorry

end problem_eq_solution_l2352_235296


namespace a_values_l2352_235265

noncomputable def A (a : ℝ) : Set ℝ := {1, a, 5}
noncomputable def B (a : ℝ) : Set ℝ := {2, a^2 + 1}

theorem a_values (a : ℝ) : A a ∩ B a = {x} → (a = 0 ∧ x = 1) ∨ (a = -2 ∧ x = 5) := sorry

end a_values_l2352_235265


namespace complex_multiplication_l2352_235207

theorem complex_multiplication :
  ∀ (i : ℂ), i^2 = -1 → (1 - i) * i = 1 + i :=
by
  sorry

end complex_multiplication_l2352_235207


namespace sum_of_interior_angles_of_polygon_l2352_235266

theorem sum_of_interior_angles_of_polygon (n : ℕ) (h : n - 3 = 3) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l2352_235266


namespace positive_sqrt_729_l2352_235236

theorem positive_sqrt_729 (x : ℝ) (h_pos : 0 < x) (h_eq : x^2 = 729) : x = 27 :=
by
  sorry

end positive_sqrt_729_l2352_235236


namespace Douglas_won_72_percent_of_votes_in_county_X_l2352_235253

/-- Definition of the problem conditions and the goal -/
theorem Douglas_won_72_percent_of_votes_in_county_X
  (V : ℝ)
  (total_votes_ratio : ∀ county_X county_Y, county_X = 2 * county_Y)
  (total_votes_percentage_both_counties : 0.60 = (1.8 * V) / (2 * V + V))
  (votes_percentage_county_Y : 0.36 = (0.36 * V) / V) : 
  ∃ P : ℝ, P = 72 ∧ P = (1.44 * V) / (2 * V) * 100 :=
sorry

end Douglas_won_72_percent_of_votes_in_county_X_l2352_235253


namespace johns_weekly_earnings_after_raise_l2352_235216

theorem johns_weekly_earnings_after_raise 
  (original_weekly_earnings : ℕ) 
  (percentage_increase : ℝ) 
  (new_weekly_earnings : ℕ)
  (h1 : original_weekly_earnings = 60)
  (h2 : percentage_increase = 0.16666666666666664) :
  new_weekly_earnings = 70 :=
sorry

end johns_weekly_earnings_after_raise_l2352_235216


namespace avg_lottery_draws_eq_5232_l2352_235262

def avg_lottery_draws (n m : ℕ) : ℕ :=
  let N := 90 * 89 * 88 * 87 * 86
  let Nk := 25 * 40320
  N / Nk

theorem avg_lottery_draws_eq_5232 : avg_lottery_draws 90 5 = 5232 :=
by 
  unfold avg_lottery_draws
  sorry

end avg_lottery_draws_eq_5232_l2352_235262


namespace attendees_gift_exchange_l2352_235290

theorem attendees_gift_exchange (x : ℕ) (h1 : 56 = x * (x - 1) / 2) : 
  x * (x - 1) = 112 :=
by
  sorry

end attendees_gift_exchange_l2352_235290


namespace functional_equation_solution_l2352_235285

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) = x * f x - y * f y) →
  ∃ m b : ℝ, ∀ t : ℝ, f t = m * t + b :=
by
  intro h
  sorry

end functional_equation_solution_l2352_235285


namespace largest_possible_value_l2352_235205

variable (a b : ℝ)

theorem largest_possible_value (h1 : 4 * a + 3 * b ≤ 10) (h2 : 3 * a + 6 * b ≤ 12) :
  2 * a + b ≤ 5 :=
sorry

end largest_possible_value_l2352_235205


namespace negation_correct_l2352_235279

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  ∀ x > 0, x^2 - 2 * x + 1 ≥ 0

-- Define what it means to negate the proposition
def negated_proposition (x : ℝ) : Prop :=
  ∃ x > 0, x^2 - 2 * x + 1 < 0

-- Main statement: the negation of the original proposition equals the negated proposition
theorem negation_correct : (¬original_proposition x) = (negated_proposition x) :=
  sorry

end negation_correct_l2352_235279


namespace math_pages_l2352_235212

def total_pages := 7
def reading_pages := 2

theorem math_pages : total_pages - reading_pages = 5 := by
  sorry

end math_pages_l2352_235212


namespace product_of_square_and_neighbor_is_divisible_by_12_l2352_235211

theorem product_of_square_and_neighbor_is_divisible_by_12 (n : ℤ) : 12 ∣ (n^2 * (n - 1) * (n + 1)) :=
sorry

end product_of_square_and_neighbor_is_divisible_by_12_l2352_235211


namespace determine_perimeter_of_fourth_shape_l2352_235257

theorem determine_perimeter_of_fourth_shape
  (P_1 P_2 P_3 P_4 : ℝ)
  (h1 : P_1 = 8)
  (h2 : P_2 = 11.4)
  (h3 : P_3 = 14.7)
  (h4 : P_1 + P_2 + P_4 = 2 * P_3) :
  P_4 = 10 := 
by
  -- Proof goes here
  sorry

end determine_perimeter_of_fourth_shape_l2352_235257


namespace subset_A_l2352_235264

open Set

theorem subset_A (A : Set ℝ) (h : A = { x | x > -1 }) : {0} ⊆ A :=
by
  sorry

end subset_A_l2352_235264


namespace gcd_of_all_elements_in_B_is_2_l2352_235214

-- Define the set B as the set of all numbers that can be represented as the sum of four consecutive positive integers.
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2 ∧ x > 0}

-- Translate the question to a Lean statement.
theorem gcd_of_all_elements_in_B_is_2 : ∀ n ∈ B, gcd n 2 = 2 := 
by
  sorry

end gcd_of_all_elements_in_B_is_2_l2352_235214


namespace paul_work_days_l2352_235259

theorem paul_work_days (P : ℕ) (h : 1 / P + 1 / 120 = 1 / 48) : P = 80 := 
by 
  sorry

end paul_work_days_l2352_235259


namespace P_gt_Q_l2352_235267

variable (x : ℝ)

def P := x^2 + 2
def Q := 2 * x

theorem P_gt_Q : P x > Q x := by
  sorry

end P_gt_Q_l2352_235267


namespace mean_combined_l2352_235248

-- Definitions for the two sets and their properties
def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

variables (set₁ set₂ : List ℕ)
-- Conditions based on the problem
axiom h₁ : set₁.length = 7
axiom h₂ : mean set₁ = 15
axiom h₃ : set₂.length = 8
axiom h₄ : mean set₂ = 30

-- Prove that the mean of the combined set is 23
theorem mean_combined (h₁ : set₁.length = 7) (h₂ : mean set₁ = 15)
  (h₃ : set₂.length = 8) (h₄ : mean set₂ = 30) : mean (set₁ ++ set₂) = 23 := 
sorry

end mean_combined_l2352_235248


namespace polygon_sides_l2352_235219

-- Definition of the conditions used in the problem
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Statement of the theorem
theorem polygon_sides (n : ℕ) (h : sum_of_interior_angles n = 1080) : n = 8 :=
by
  sorry  -- Proof placeholder

end polygon_sides_l2352_235219


namespace solve_quadratic_eq_l2352_235233

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2*x + 1 = 0) : x = 1 :=
by
  sorry

end solve_quadratic_eq_l2352_235233


namespace symmetric_circle_eq_l2352_235289

theorem symmetric_circle_eq (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (x^2 + y^2 - 4 * y = 0) :=
sorry

end symmetric_circle_eq_l2352_235289


namespace total_spokes_in_garage_l2352_235273

def bicycle1_front_spokes : ℕ := 16
def bicycle1_back_spokes : ℕ := 18
def bicycle2_front_spokes : ℕ := 20
def bicycle2_back_spokes : ℕ := 22
def bicycle3_front_spokes : ℕ := 24
def bicycle3_back_spokes : ℕ := 26
def bicycle4_front_spokes : ℕ := 28
def bicycle4_back_spokes : ℕ := 30
def tricycle_front_spokes : ℕ := 32
def tricycle_middle_spokes : ℕ := 34
def tricycle_back_spokes : ℕ := 36

theorem total_spokes_in_garage :
  bicycle1_front_spokes + bicycle1_back_spokes +
  bicycle2_front_spokes + bicycle2_back_spokes +
  bicycle3_front_spokes + bicycle3_back_spokes +
  bicycle4_front_spokes + bicycle4_back_spokes +
  tricycle_front_spokes + tricycle_middle_spokes + tricycle_back_spokes = 286 :=
by
  sorry

end total_spokes_in_garage_l2352_235273


namespace johnny_marbles_combination_l2352_235250

theorem johnny_marbles_combination : @Nat.choose 9 4 = 126 := by
  sorry

end johnny_marbles_combination_l2352_235250


namespace club_committee_selections_l2352_235252

theorem club_committee_selections : (Nat.choose 18 3) = 816 := by
  sorry

end club_committee_selections_l2352_235252


namespace min_value_of_quadratic_expression_l2352_235232

theorem min_value_of_quadratic_expression (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) : a^2 + 4 * b^2 + 9 * c^2 ≥ 12 :=
by
  sorry

end min_value_of_quadratic_expression_l2352_235232


namespace evaluate_complex_fraction_l2352_235298

theorem evaluate_complex_fraction : 
  (1 / (2 + (1 / (3 + 1 / 4)))) = (13 / 30) :=
by
  sorry

end evaluate_complex_fraction_l2352_235298


namespace zhao_estimate_larger_l2352_235206

theorem zhao_estimate_larger (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - 2 * ε) > x - y :=
by
  sorry

end zhao_estimate_larger_l2352_235206


namespace pythagorean_triplets_l2352_235229

theorem pythagorean_triplets (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ d p q : ℤ, a = 2 * d * p * q ∧ b = d * (q^2 - p^2) ∧ c = d * (p^2 + q^2) := sorry

end pythagorean_triplets_l2352_235229


namespace yard_flower_beds_fraction_l2352_235299

theorem yard_flower_beds_fraction :
  let yard_length := 30
  let yard_width := 10
  let pool_length := 10
  let pool_width := 4
  let trap_parallel_diff := 22 - 16
  let triangle_leg := trap_parallel_diff / 2
  let triangle_area := (1 / 2) * (triangle_leg ^ 2)
  let total_triangle_area := 2 * triangle_area
  let total_yard_area := yard_length * yard_width
  let pool_area := pool_length * pool_width
  let usable_yard_area := total_yard_area - pool_area
  (total_triangle_area / usable_yard_area) = 9 / 260 :=
by 
  sorry

end yard_flower_beds_fraction_l2352_235299


namespace parallelogram_base_length_l2352_235210

theorem parallelogram_base_length (b : ℝ) (A : ℝ) (h : ℝ)
  (H1 : A = 288) 
  (H2 : h = 2 * b) 
  (H3 : A = b * h) : 
  b = 12 := 
by 
  sorry

end parallelogram_base_length_l2352_235210


namespace evaluate_expression_l2352_235295

theorem evaluate_expression (x y : ℕ) (hx : 2^x ∣ 360 ∧ ¬ 2^(x+1) ∣ 360) (hy : 3^y ∣ 360 ∧ ¬ 3^(y+1) ∣ 360) :
  (3 / 7)^(y - x) = 7 / 3 := by
  sorry

end evaluate_expression_l2352_235295


namespace original_fraction_is_two_thirds_l2352_235209

theorem original_fraction_is_two_thirds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ)/(b + 3) = 2 * (a : ℚ)/b → (a : ℚ)/b = 2/3 :=
by
  sorry

end original_fraction_is_two_thirds_l2352_235209


namespace fraction_zero_solve_l2352_235228

theorem fraction_zero_solve (x : ℝ) (h : (x^2 - 49) / (x + 7) = 0) : x = 7 :=
by
  sorry

end fraction_zero_solve_l2352_235228


namespace negation_of_p_implication_q_l2352_235241

noncomputable def negation_of_conditions : Prop :=
∀ (a : ℝ), (a > 0 → a^2 > a) ∧ (¬(a > 0) ↔ ¬(a^2 > a)) → ¬(a ≤ 0 → a^2 ≤ a)

theorem negation_of_p_implication_q :
  negation_of_conditions :=
by {
  sorry
}

end negation_of_p_implication_q_l2352_235241
