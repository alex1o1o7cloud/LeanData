import Mathlib

namespace kyle_paper_delivery_l1270_127043

-- Define the number of houses Kyle delivers to from Monday through Saturday
def housesDaily : ℕ := 100

-- Define the number of days from Monday to Saturday
def daysWeek : ℕ := 6

-- Define the adjustment for Sunday (10 fewer, 30 additional)
def sundayAdjust : ℕ := 30 - 10 + 100

-- Define the total number of papers delivered from Monday to Saturday
def papersMonToSat : ℕ := daysWeek * housesDaily

-- Define the total number of papers delivered on Sunday
def papersSunday : ℕ := sundayAdjust

-- Define the total number of papers delivered each week
def totalPapers : ℕ := papersMonToSat + papersSunday

-- The theorem we want to prove
theorem kyle_paper_delivery : totalPapers = 720 := by
  -- We are focusing only on the statement here.
  sorry

end kyle_paper_delivery_l1270_127043


namespace smallest_nat_mod_5_6_7_l1270_127033

theorem smallest_nat_mod_5_6_7 (n : ℕ) :
  n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6 → n = 209 :=
sorry

end smallest_nat_mod_5_6_7_l1270_127033


namespace infinite_triangles_with_conditions_l1270_127056

theorem infinite_triangles_with_conditions :
  ∃ (A : ℝ) (B : ℝ) (C : ℝ), 
  (A > 0) ∧ (B > 0) ∧ (C > 0) ∧ (B - A = 2) ∧ (C = 4) ∧ 
  (Δ > 0) := sorry

end infinite_triangles_with_conditions_l1270_127056


namespace largest_integer_solution_l1270_127074

theorem largest_integer_solution (x : ℤ) (h : -x ≥ 2 * x + 3) : x ≤ -1 := sorry

end largest_integer_solution_l1270_127074


namespace total_liquid_poured_out_l1270_127098

noncomputable def capacity1 := 2
noncomputable def capacity2 := 6
noncomputable def percentAlcohol1 := 0.3
noncomputable def percentAlcohol2 := 0.4
noncomputable def totalCapacity := 10
noncomputable def finalConcentration := 0.3

theorem total_liquid_poured_out :
  capacity1 + capacity2 = 8 :=
by
  sorry

end total_liquid_poured_out_l1270_127098


namespace prime_divides_expression_l1270_127048

theorem prime_divides_expression (p : ℕ) (hp : p > 5 ∧ Prime p) : 
  ∃ n : ℕ, p ∣ (20^n + 15^n - 12^n) := 
  by
  use (p - 3)
  sorry

end prime_divides_expression_l1270_127048


namespace MrJones_pants_count_l1270_127084

theorem MrJones_pants_count (P : ℕ) (h1 : 6 * P + P = 280) : P = 40 := by
  sorry

end MrJones_pants_count_l1270_127084


namespace max_value_of_function_l1270_127080

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem max_value_of_function (α : ℝ)
  (h₁ : f 4 α = 2)
  : ∃ a : ℝ, 3 ≤ a ∧ a ≤ 5 ∧ (f (a - 3) (α) + f (5 - a) α = 2) := 
sorry

end max_value_of_function_l1270_127080


namespace contradiction_proof_l1270_127076

theorem contradiction_proof (a b c d : ℝ) (h1 : a = 1) (h2 : b = 1) (h3 : c = 1) (h4 : d = 1) (h5 : a * c + b * d > 1) : ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_proof_l1270_127076


namespace train_speed_ratio_l1270_127061

theorem train_speed_ratio 
  (distance_2nd_train : ℕ)
  (time_2nd_train : ℕ)
  (speed_1st_train : ℚ)
  (H1 : distance_2nd_train = 400)
  (H2 : time_2nd_train = 4)
  (H3 : speed_1st_train = 87.5) :
  distance_2nd_train / time_2nd_train = 100 ∧ 
  (speed_1st_train / (distance_2nd_train / time_2nd_train)) = 7 / 8 :=
by
  sorry

end train_speed_ratio_l1270_127061


namespace minutes_in_3_5_hours_l1270_127047

theorem minutes_in_3_5_hours : 3.5 * 60 = 210 := 
by
  sorry

end minutes_in_3_5_hours_l1270_127047


namespace find_value_of_N_l1270_127086

theorem find_value_of_N (x N : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (N + 3 * x)^4) : N = 1.5 := by
  -- Here we will assume that the proof is filled in and correct.
  sorry

end find_value_of_N_l1270_127086


namespace inequality_problem_l1270_127031

noncomputable def nonneg_real := {x : ℝ // 0 ≤ x}

theorem inequality_problem (x y z : nonneg_real) (h : x.val * y.val + y.val * z.val + z.val * x.val = 1) :
  1 / (x.val + y.val) + 1 / (y.val + z.val) + 1 / (z.val + x.val) ≥ 5 / 2 :=
sorry

end inequality_problem_l1270_127031


namespace problem_statement_l1270_127093

theorem problem_statement (p : ℝ) : 
  (∀ (q : ℝ), q > 0 → (3 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) 
  ↔ (0 ≤ p ∧ p ≤ 7.275) :=
sorry

end problem_statement_l1270_127093


namespace cos_sum_is_zero_l1270_127062

theorem cos_sum_is_zero (x y z : ℝ) 
  (h1: Real.cos (2 * x) + 2 * Real.cos (2 * y) + 3 * Real.cos (2 * z) = 0) 
  (h2: Real.sin (2 * x) + 2 * Real.sin (2 * y) + 3 * Real.sin (2 * z) = 0) : 
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := 
by 
  sorry

end cos_sum_is_zero_l1270_127062


namespace right_triangle_345_l1270_127070

def is_right_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

theorem right_triangle_345 : is_right_triangle 3 4 5 :=
by
  sorry

end right_triangle_345_l1270_127070


namespace sales_on_third_day_l1270_127063

variable (a m : ℕ)

def first_day_sales : ℕ := a
def second_day_sales : ℕ := 3 * a - 3 * m
def third_day_sales : ℕ := (3 * a - 3 * m) + m

theorem sales_on_third_day 
  (a m : ℕ) : third_day_sales a m = 3 * a - 2 * m :=
by
  -- Assuming the conditions as our definitions:
  let fds := first_day_sales a
  let sds := second_day_sales a m
  let tds := third_day_sales a m

  -- Proof direction:
  show tds = 3 * a - 2 * m
  sorry

end sales_on_third_day_l1270_127063


namespace rectangular_field_length_l1270_127081

   theorem rectangular_field_length (w l : ℝ) 
     (h1 : l = 2 * w)
     (h2 : 64 = 8 * 8)
     (h3 : 64 = (1/72) * (l * w)) :
     l = 96 :=
   sorry
   
end rectangular_field_length_l1270_127081


namespace fraction_of_25_exact_value_l1270_127017

-- Define the conditions
def eighty_percent_of_sixty : ℝ := 0.80 * 60
def smaller_by_twenty_eight (x : ℝ) : Prop := x * 25 = eighty_percent_of_sixty - 28

-- The proof problem
theorem fraction_of_25_exact_value (x : ℝ) : smaller_by_twenty_eight x → x = 4 / 5 := by
  intro h
  sorry

end fraction_of_25_exact_value_l1270_127017


namespace t_shirt_sale_revenue_per_minute_l1270_127083

theorem t_shirt_sale_revenue_per_minute (total_tshirts : ℕ) (total_minutes : ℕ)
  (black_tshirts white_tshirts : ℕ) (cost_black cost_white : ℕ) 
  (half_total_tshirts : total_tshirts = black_tshirts + white_tshirts)
  (equal_halves : black_tshirts = white_tshirts)
  (black_price : cost_black = 30) (white_price : cost_white = 25)
  (total_time : total_minutes = 25)
  (total_sold : total_tshirts = 200) :
  ((black_tshirts * cost_black) + (white_tshirts * cost_white)) / total_minutes = 220 := by
  sorry

end t_shirt_sale_revenue_per_minute_l1270_127083


namespace sindbad_can_identify_eight_genuine_dinars_l1270_127099

/--
Sindbad has 11 visually identical dinars in his purse, one of which may be counterfeit and differs in weight from the genuine ones. Using a balance scale twice without weights, it's possible to identify at least 8 genuine dinars.
-/
theorem sindbad_can_identify_eight_genuine_dinars (dinars : Fin 11 → ℝ) (is_genuine : Fin 11 → Prop) :
  (∃! i, ¬ is_genuine i) → 
  (∃ S : Finset (Fin 11), S.card = 8 ∧ S ⊆ (Finset.univ : Finset (Fin 11)) ∧ ∀ i ∈ S, is_genuine i) :=
sorry

end sindbad_can_identify_eight_genuine_dinars_l1270_127099


namespace simplify_expression_l1270_127072

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end simplify_expression_l1270_127072


namespace min_value_of_x_plus_y_l1270_127054

theorem min_value_of_x_plus_y (x y : ℝ) (h1: y ≠ 0) (h2: 1 / y = (x - 1) / 2) : x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_of_x_plus_y_l1270_127054


namespace solution_set_of_inequality_l1270_127004

def fraction_inequality_solution : Set ℝ := {x : ℝ | -4 < x ∧ x < -1}

theorem solution_set_of_inequality (x : ℝ) :
  (2 - x) / (x + 4) > 1 ↔ -4 < x ∧ x < -1 := by
sorry

end solution_set_of_inequality_l1270_127004


namespace tan_sum_l1270_127025

open Real

theorem tan_sum 
  (α β γ θ φ : ℝ)
  (h1 : tan θ = (sin α * cos γ - sin β * sin γ) / (cos α * cos γ - cos β * sin γ))
  (h2 : tan φ = (sin α * sin γ - sin β * cos γ) / (cos α * sin γ - cos β * cos γ)) : 
  tan (θ + φ) = tan (α + β) :=
by
  sorry

end tan_sum_l1270_127025


namespace shifted_parabola_passes_through_neg1_1_l1270_127021

def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

def shifted_parabola (x : ℝ) : ℝ := -(x - 1)^2 + 2

theorem shifted_parabola_passes_through_neg1_1 :
  shifted_parabola (-1) = 1 :=
by 
  -- Proof goes here
  sorry

end shifted_parabola_passes_through_neg1_1_l1270_127021


namespace fraction_of_students_with_buddy_l1270_127092

theorem fraction_of_students_with_buddy (s n : ℕ) (h : n = 4 * s / 3) : 
  (n / 4 + s / 3) / (n + s : ℚ) = 2 / 7 :=
by
  sorry

end fraction_of_students_with_buddy_l1270_127092


namespace ratio_of_coeffs_l1270_127020

theorem ratio_of_coeffs
  (a b c d e : ℝ) 
  (h_poly : ∀ x : ℝ, a * x^4 + b * x^3 + c * x^2 + d * x + e = 0 → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) : 
  d / e = 25 / 12 :=
by
  sorry

end ratio_of_coeffs_l1270_127020


namespace sum_reciprocals_factors_12_l1270_127059

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l1270_127059


namespace greatest_x_value_l1270_127041

theorem greatest_x_value :
  ∃ x, (∀ y, (y ≠ 6) → (y^2 - y - 30) / (y - 6) = 2 / (y + 4) → y ≤ x) ∧ (x^2 - x - 30) / (x - 6) = 2 / (x + 4) ∧ x ≠ 6 ∧ x = -3 :=
sorry

end greatest_x_value_l1270_127041


namespace min_episodes_to_watch_l1270_127089

theorem min_episodes_to_watch (T W H F Sa Su M trip_days total_episodes: ℕ)
  (hW: W = 1) (hTh: H = 1) (hF: F = 1) (hSa: Sa = 2) (hSu: Su = 2) (hMo: M = 0)
  (total_episodes_eq: total_episodes = 60)
  (trip_days_eq: trip_days = 17):
  total_episodes - ((4 * W + 2 * Sa + 1 * M) * (trip_days / 7) + (trip_days % 7) * (W + Sa + Su + Mo)) = 39 := 
by
  sorry

end min_episodes_to_watch_l1270_127089


namespace bryan_total_books_and_magazines_l1270_127058

-- Define the conditions
def books_per_shelf : ℕ := 23
def magazines_per_shelf : ℕ := 61
def bookshelves : ℕ := 29

-- Define the total books and magazines
def total_books : ℕ := books_per_shelf * bookshelves
def total_magazines : ℕ := magazines_per_shelf * bookshelves
def total_books_and_magazines : ℕ := total_books + total_magazines

-- The proof problem statement
theorem bryan_total_books_and_magazines : total_books_and_magazines = 2436 := 
by
  sorry

end bryan_total_books_and_magazines_l1270_127058


namespace smallest_possible_difference_after_101_years_l1270_127075

theorem smallest_possible_difference_after_101_years {D E : ℤ} 
  (init_dollar : D = 6) 
  (init_euro : E = 7)
  (transformations : ∀ D E : ℤ, 
    (D', E') = (D + E, 2 * D + 1) ∨ (D', E') = (D + E, 2 * D - 1) ∨ 
    (D', E') = (D + E, 2 * E + 1) ∨ (D', E') = (D + E, 2 * E - 1)) :
  ∃ n_diff : ℤ, 101 = 2 * n_diff ∧ n_diff = 2 :=
sorry

end smallest_possible_difference_after_101_years_l1270_127075


namespace men_in_second_group_l1270_127026

theorem men_in_second_group (m w : ℝ) (x : ℝ) 
  (h1 : 3 * m + 8 * w = x * m + 2 * w) 
  (h2 : 2 * m + 2 * w = (3 / 7) * (3 * m + 8 * w)) : x = 6 :=
by
  sorry

end men_in_second_group_l1270_127026


namespace points_on_line_with_slope_l1270_127052

theorem points_on_line_with_slope :
  ∃ a b : ℝ, 
  (a - 3) ≠ 0 ∧ (b - 5) ≠ 0 ∧
  (7 - 5) / (a - 3) = 4 ∧ (b - 5) / (-1 - 3) = 4 ∧
  a = 7 / 2 ∧ b = -11 := 
by
  existsi 7 / 2
  existsi -11
  repeat {split}
  all_goals { sorry }

end points_on_line_with_slope_l1270_127052


namespace arithmetic_sequence_sum_l1270_127079

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a2_a5 : a 2 + a 5 = 4
axiom a6_a9 : a 6 + a 9 = 20

theorem arithmetic_sequence_sum : a 4 + a 7 = 12 := by
  sorry

end arithmetic_sequence_sum_l1270_127079


namespace speed_of_second_car_l1270_127087

theorem speed_of_second_car
  (t : ℝ) (d : ℝ) (d1 : ℝ) (d2 : ℝ) (v : ℝ)
  (h1 : t = 2.5)
  (h2 : d = 175)
  (h3 : d1 = 25 * t)
  (h4 : d2 = v * t)
  (h5 : d1 + d2 = d) :
  v = 45 := by sorry

end speed_of_second_car_l1270_127087


namespace sufficient_conditions_for_x_sq_lt_one_l1270_127078

theorem sufficient_conditions_for_x_sq_lt_one
  (x : ℝ) :
  (0 < x ∧ x < 1) ∨ (-1 < x ∧ x < 0) ∨ (-1 < x ∧ x < 1) → x^2 < 1 :=
by
  sorry

end sufficient_conditions_for_x_sq_lt_one_l1270_127078


namespace evaluate_expression_l1270_127057

variable (a : ℝ)
variable (x : ℝ)

theorem evaluate_expression (h : x = a + 9) : x - a + 6 = 15 := by
  sorry

end evaluate_expression_l1270_127057


namespace paper_stars_per_bottle_l1270_127045

theorem paper_stars_per_bottle (a b total_bottles : ℕ) (h1 : a = 33) (h2 : b = 307) (h3 : total_bottles = 4) :
  (a + b) / total_bottles = 85 :=
by
  sorry

end paper_stars_per_bottle_l1270_127045


namespace additional_distance_if_faster_speed_l1270_127044

-- Conditions
def speed_slow := 10 -- km/hr
def speed_fast := 15 -- km/hr
def actual_distance := 30 -- km

-- Question and answer
theorem additional_distance_if_faster_speed : (speed_fast * (actual_distance / speed_slow) - actual_distance) = 15 := by
  sorry

end additional_distance_if_faster_speed_l1270_127044


namespace number_verification_l1270_127018

def is_product_of_four_consecutive (n : ℕ) : Prop :=
  ∃ a : ℕ, n = a * (a + 1) * (a + 2) * (a + 3)

theorem number_verification (h1 : 1680 % 3 = 0) (h2 : ∃ a : ℕ, 1680 = a * (a + 1) * (a + 2) * (a + 3)) : 
  is_product_of_four_consecutive 1680 :=
by
  sorry

end number_verification_l1270_127018


namespace range_of_x_l1270_127010

theorem range_of_x (x : ℝ) : (abs (x + 1) + abs (x - 5) = 6) ↔ (-1 ≤ x ∧ x ≤ 5) :=
by sorry

end range_of_x_l1270_127010


namespace inequality_proof_l1270_127023

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (b+c-a)^2 / ((b+c)^2+a^2) + (c+a-b)^2 / ((c+a)^2+b^2) + (a+b-c)^2 / ((a+b)^2+c^2) ≥ 3 / 5 :=
by sorry

end inequality_proof_l1270_127023


namespace no_positive_real_solutions_l1270_127042

theorem no_positive_real_solutions 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^3 + y^3 + z^3 = x + y + z) (h2 : x^2 + y^2 + z^2 = x * y * z) :
  false :=
by sorry

end no_positive_real_solutions_l1270_127042


namespace digit_sum_of_product_l1270_127053

def digits_after_multiplication (a b : ℕ) : ℕ :=
  let product := a * b
  let units_digit := product % 10
  let tens_digit := (product / 10) % 10
  tens_digit + units_digit

theorem digit_sum_of_product :
  digits_after_multiplication 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909 = 9 :=
by 
  -- proof goes here
sorry

end digit_sum_of_product_l1270_127053


namespace complement_of_M_is_correct_l1270_127077

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - x > 0}
def complement_M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem complement_of_M_is_correct :
  (U \ M) = complement_M :=
by
  sorry

end complement_of_M_is_correct_l1270_127077


namespace square_non_negative_is_universal_l1270_127009

/-- The square of any real number is non-negative, which is a universal proposition. -/
theorem square_non_negative_is_universal : 
  ∀ x : ℝ, x^2 ≥ 0 :=
by
  sorry

end square_non_negative_is_universal_l1270_127009


namespace expand_polynomial_l1270_127097

theorem expand_polynomial :
  (3 * x^2 + 2 * x + 1) * (2 * x^2 + 3 * x + 4) = 6 * x^4 + 13 * x^3 + 20 * x^2 + 11 * x + 4 :=
by
  sorry

end expand_polynomial_l1270_127097


namespace determine_n_l1270_127003

theorem determine_n (x a : ℝ) (n : ℕ)
  (h1 : (n.choose 3) * x^(n-3) * a^3 = 120)
  (h2 : (n.choose 4) * x^(n-4) * a^4 = 360)
  (h3 : (n.choose 5) * x^(n-5) * a^5 = 720) :
  n = 12 :=
sorry

end determine_n_l1270_127003


namespace min_frac_sum_l1270_127012

noncomputable def min_value (x y : ℝ) : ℝ :=
  if (x + y = 1 ∧ x > 0 ∧ y > 0) then 1/x + 4/y else 0

theorem min_frac_sum (x y : ℝ) (h₁ : x + y = 1) (h₂: x > 0) (h₃: y > 0) : 
  min_value x y = 9 :=
sorry

end min_frac_sum_l1270_127012


namespace heads_at_least_twice_in_5_tosses_l1270_127038

noncomputable def probability_at_least_two_heads (n : ℕ) (p : ℚ) : ℚ :=
1 - (n : ℚ) * p^(n : ℕ)

theorem heads_at_least_twice_in_5_tosses :
  probability_at_least_two_heads 5 (1/2) = 13/16 :=
by
  sorry

end heads_at_least_twice_in_5_tosses_l1270_127038


namespace jamie_nickels_l1270_127066

theorem jamie_nickels (x : ℕ) (hx : 5 * x + 10 * x + 25 * x = 1320) : x = 33 :=
sorry

end jamie_nickels_l1270_127066


namespace nancy_more_money_l1270_127091

def jade_available := 1920
def giraffe_jade := 120
def elephant_jade := 2 * giraffe_jade
def giraffe_price := 150
def elephant_price := 350

def num_giraffes := jade_available / giraffe_jade
def num_elephants := jade_available / elephant_jade
def revenue_giraffes := num_giraffes * giraffe_price
def revenue_elephants := num_elephants * elephant_price

def revenue_difference := revenue_elephants - revenue_giraffes

theorem nancy_more_money : revenue_difference = 400 :=
by sorry

end nancy_more_money_l1270_127091


namespace common_difference_of_arithmetic_sequence_l1270_127028

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 0 = 2) 
  (h2 : ∀ n, a (n+1) = a n + d)
  (h3 : a 9 = 20): 
  d = 2 := 
by
  sorry

end common_difference_of_arithmetic_sequence_l1270_127028


namespace range_alpha_minus_beta_l1270_127014

theorem range_alpha_minus_beta (α β : ℝ) (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π / 2) :
  - (3 * π) / 2 ≤ α - β ∧ α - β ≤ 0 :=
sorry

end range_alpha_minus_beta_l1270_127014


namespace tom_total_spent_on_video_games_l1270_127035

-- Conditions
def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

-- Statement to be proved
theorem tom_total_spent_on_video_games : batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end tom_total_spent_on_video_games_l1270_127035


namespace total_weight_of_full_bucket_l1270_127082

variable (a b x y : ℝ)

def bucket_weights :=
  (x + (1/3) * y = a) → (x + (3/4) * y = b) → (x + y = (16/5) * b - (11/5) * a)

theorem total_weight_of_full_bucket :
  bucket_weights a b x y :=
by
  intro h1 h2
  -- proof goes here, can be omitted as per instructions
  sorry

end total_weight_of_full_bucket_l1270_127082


namespace angle_A_measure_l1270_127032

variable {a b c A : ℝ}

def vector_m (b c a : ℝ) : ℝ × ℝ := (b, c - a)
def vector_n (b c a : ℝ) : ℝ × ℝ := (b - c, c + a)

theorem angle_A_measure (h_perpendicular : (vector_m b c a).1 * (vector_n b c a).1 + (vector_m b c a).2 * (vector_n b c a).2 = 0) :
  A = 2 * π / 3 := sorry

end angle_A_measure_l1270_127032


namespace max_candies_ben_eats_l1270_127088

theorem max_candies_ben_eats (total_candies : ℕ) (k : ℕ) (h_pos_k : k > 0) (b : ℕ) 
  (h_total : b + 2 * b + k * b = total_candies) (h_total_candies : total_candies = 30) : b = 6 :=
by
  -- placeholder for proof steps
  sorry

end max_candies_ben_eats_l1270_127088


namespace coefficient_x3_in_expansion_l1270_127085

theorem coefficient_x3_in_expansion : 
  (∃ (r : ℕ), 5 - r / 2 = 3 ∧ 2 * Nat.choose 5 r = 10) :=
by 
  sorry

end coefficient_x3_in_expansion_l1270_127085


namespace find_bicycle_speed_l1270_127029

-- Let's define the conditions first
def distance := 10  -- Distance in km
def time_diff := 1 / 3  -- Time difference in hours
def speed_of_bicycle (x : ℝ) := x
def speed_of_car (x : ℝ) := 2 * x

-- Prove the equation using the given conditions
theorem find_bicycle_speed (x : ℝ) (h : x ≠ 0) :
  (distance / speed_of_bicycle x) = (distance / speed_of_car x) + time_diff :=
by {
  sorry
}

end find_bicycle_speed_l1270_127029


namespace smallest_n_value_l1270_127034

theorem smallest_n_value (n : ℕ) (h : 15 * n - 2 ≡ 0 [MOD 11]) : n = 6 :=
sorry

end smallest_n_value_l1270_127034


namespace gold_coins_percent_l1270_127001

variable (total_objects beads papers coins silver_gold total_gold : ℝ)
variable (h1 : total_objects = 100)
variable (h2 : beads = 15)
variable (h3 : papers = 10)
variable (h4 : silver_gold = 30)
variable (h5 : total_gold = 52.5)

theorem gold_coins_percent : (total_objects - beads - papers) * (100 - silver_gold) / 100 = total_gold :=
by 
  -- Insert proof here
  sorry

end gold_coins_percent_l1270_127001


namespace value_of_p_l1270_127036

theorem value_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (x1 x2 : ℕ), x1 > 0 ∧ x2 > 0 ∧ 67 * x1^2 + p * x1 + q = 0 ∧ 67 * x2^2 + p * x2 + q = 0) : p = -2278 :=
by
  sorry

end value_of_p_l1270_127036


namespace sufficient_condition_implies_range_l1270_127046

theorem sufficient_condition_implies_range {x m : ℝ} : (∀ x, 1 ≤ x ∧ x < 4 → x < m) → 4 ≤ m :=
by
  sorry

end sufficient_condition_implies_range_l1270_127046


namespace calculate_fraction_pow_l1270_127027

theorem calculate_fraction_pow :
  (0.5 ^ 4) / (0.05 ^ 3) = 500 := 
  sorry

end calculate_fraction_pow_l1270_127027


namespace find_n_l1270_127008

theorem find_n 
  (n : ℕ) 
  (h_lcm : Nat.lcm n 16 = 48) 
  (h_gcf : Nat.gcd n 16 = 18) : 
  n = 54 := 
sorry

end find_n_l1270_127008


namespace remainder_of_x50_div_x_minus_1_cubed_l1270_127007

theorem remainder_of_x50_div_x_minus_1_cubed :
  (x : ℝ) → (x ^ 50) % ((x - 1) ^ 3) = 1225 * x ^ 2 - 2400 * x + 1176 := 
by
  sorry

end remainder_of_x50_div_x_minus_1_cubed_l1270_127007


namespace Berry_Temperature_Friday_l1270_127064

theorem Berry_Temperature_Friday (temps : Fin 6 → ℝ) (avg_temp : ℝ) (total_days : ℕ) (friday_temp : ℝ) :
  temps 0 = 99.1 → 
  temps 1 = 98.2 →
  temps 2 = 98.7 →
  temps 3 = 99.3 →
  temps 4 = 99.8 →
  temps 5 = 98.9 →
  avg_temp = 99 →
  total_days = 7 →
  friday_temp = (avg_temp * total_days) - (temps 0 + temps 1 + temps 2 + temps 3 + temps 4 + temps 5) →
  friday_temp = 99 :=
by 
  intros h0 h1 h2 h3 h4 h5 h_avg h_days h_friday
  sorry

end Berry_Temperature_Friday_l1270_127064


namespace dot_product_range_l1270_127065

theorem dot_product_range (a b : ℝ) (θ : ℝ) (h1 : a = 8) (h2 : b = 12)
  (h3 : 30 * (Real.pi / 180) ≤ θ ∧ θ ≤ 60 * (Real.pi / 180)) :
  48 * Real.sqrt 3 ≤ a * b * Real.cos θ ∧ a * b * Real.cos θ ≤ 48 :=
by
  sorry

end dot_product_range_l1270_127065


namespace both_pumps_drain_lake_l1270_127073

theorem both_pumps_drain_lake (T : ℝ) (h₁ : 1 / 9 + 1 / 6 = 5 / 18) : 
  (5 / 18) * T = 1 → T = 18 / 5 := sorry

end both_pumps_drain_lake_l1270_127073


namespace neg_power_of_square_l1270_127095

theorem neg_power_of_square (a : ℝ) : (-a^2)^3 = -a^6 :=
by sorry

end neg_power_of_square_l1270_127095


namespace find_pos_real_nums_l1270_127024

theorem find_pos_real_nums (x y z a b c : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z):
  (x + y + z = a + b + c) ∧ (4 * x * y * z = a^2 * x + b^2 * y + c^2 * z + a * b * c) →
  (a = y + z - x ∧ b = z + x - y ∧ c = x + y - z) :=
by
  sorry

end find_pos_real_nums_l1270_127024


namespace domain_range_sum_l1270_127071

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem domain_range_sum (m n : ℝ) (hmn : ∀ x, m ≤ x ∧ x ≤ n → (f x = 3 * x)) : m + n = -1 :=
by
  sorry

end domain_range_sum_l1270_127071


namespace compute_B_93_l1270_127096

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 0, 0], ![0, 0, -1], ![0, 1, 0]]

theorem compute_B_93 : B^93 = B := by
  sorry

end compute_B_93_l1270_127096


namespace divides_quartic_sum_l1270_127067

theorem divides_quartic_sum (a b c n : ℤ) (h1 : n ∣ (a + b + c)) (h2 : n ∣ (a^2 + b^2 + c^2)) : n ∣ (a^4 + b^4 + c^4) := 
sorry

end divides_quartic_sum_l1270_127067


namespace solution_l1270_127060

noncomputable def problem_statement (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5)) : ℝ :=
  (x^2 * y^2)

theorem solution : ∀ x y : ℝ, x > 1 → y > 1 → (Real.log x / Real.log 3)^4 + (Real.log y / Real.log 5)^4 + 16 = 12 * (Real.log x / Real.log 3) * (Real.log y / Real.log 5) →
  (x^2 * y^2) = 225^(Real.sqrt 2) :=
by
  intros x y hx hy h
  sorry

end solution_l1270_127060


namespace union_of_sets_l1270_127013

open Set

variable (a b : ℕ)

noncomputable def M : Set ℕ := {3, 2 * a}
noncomputable def N : Set ℕ := {a, b}

theorem union_of_sets (h : M a ∩ N a b = {2}) : M a ∪ N a b = {1, 2, 3} :=
by
  -- skipped proof
  sorry

end union_of_sets_l1270_127013


namespace angle_of_rotation_l1270_127069

-- Definitions for the given conditions
def radius_large := 9 -- cm
def radius_medium := 3 -- cm
def radius_small := 1 -- cm
def speed := 1 -- cm/s

-- Definition of the angles calculations
noncomputable def rotations_per_revolution (R1 R2 : ℝ) : ℝ := R1 / R2
noncomputable def total_rotations (R1 R2 R3 : ℝ) : ℝ := 
  let rotations_medium := rotations_per_revolution R1 R2
  let net_rotations_medium := rotations_medium - 1
  net_rotations_medium * rotations_per_revolution R2 R3 + 1

-- Assertion to prove
theorem angle_of_rotation : 
  total_rotations radius_large radius_medium radius_small * 360 = 2520 :=
by 
  simp [total_rotations, rotations_per_revolution]
  exact sorry -- proof placeholder

end angle_of_rotation_l1270_127069


namespace vanya_scores_not_100_l1270_127049

-- Definitions for initial conditions
def score_r (M : ℕ) := M - 14
def score_p (M : ℕ) := M - 9
def score_m (M : ℕ) := M

-- Define the maximum score constraint
def max_score := 100

-- Main statement to be proved
theorem vanya_scores_not_100 (M : ℕ) 
  (hr : score_r M ≤ max_score) 
  (hp : score_p M ≤ max_score) 
  (hm : score_m M ≤ max_score) : 
  ¬(score_r M = max_score ∧ (score_p M = max_score ∨ score_m M = max_score)) ∧
  ¬(score_r M = max_score ∧ score_p M = max_score ∧ score_m M = max_score) :=
sorry

end vanya_scores_not_100_l1270_127049


namespace kite_diagonal_ratio_l1270_127016

theorem kite_diagonal_ratio (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx1 : 0 ≤ x) (hx2 : x < a) (hy1 : 0 ≤ y) (hy2 : y < b)
  (orthogonal_diagonals : a^2 + y^2 = b^2 + x^2) :
  (a / b)^2 = 4 / 3 := 
sorry

end kite_diagonal_ratio_l1270_127016


namespace leap_year_53_sundays_and_february_5_sundays_l1270_127040

theorem leap_year_53_sundays_and_february_5_sundays :
  let Y := 366
  let W := 52
  ∃ (p : ℚ), p = (2/7) * (1/7) → p = 2/49
:=
by
  sorry

end leap_year_53_sundays_and_february_5_sundays_l1270_127040


namespace class_size_l1270_127005

theorem class_size
  (S_society : ℕ) (S_music : ℕ) (S_both : ℕ) (S : ℕ)
  (h_society : S_society = 25)
  (h_music : S_music = 32)
  (h_both : S_both = 27)
  (h_total : S = S_society + S_music - S_both) :
  S = 30 :=
by
  rw [h_society, h_music, h_both] at h_total
  exact h_total

end class_size_l1270_127005


namespace cos_angle_identity_l1270_127094

theorem cos_angle_identity (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) :
  Real.cos (π - 2 * α) = - (5 / 9) := by
sorry

end cos_angle_identity_l1270_127094


namespace age_problem_solution_l1270_127022

namespace AgeProblem

variables (S M : ℕ) (k : ℕ)

-- Condition: The present age of the son is 22
def son_age (S : ℕ) := S = 22

-- Condition: The man is 24 years older than his son
def man_age (M S : ℕ) := M = S + 24

-- Condition: In two years, man's age will be a certain multiple of son's age
def age_multiple (M S k : ℕ) := M + 2 = k * (S + 2)

-- Question: The ratio of man's age to son's age in two years
def age_ratio (M S : ℕ) := (M + 2) / (S + 2)

theorem age_problem_solution (S M : ℕ) (k : ℕ) 
  (h1 : son_age S)
  (h2 : man_age M S)
  (h3 : age_multiple M S k)
  : age_ratio M S = 2 :=
by
  rw [son_age, man_age, age_multiple, age_ratio] at *
  sorry

end AgeProblem

end age_problem_solution_l1270_127022


namespace calories_difference_l1270_127051

def calories_burnt (hours : ℕ) : ℕ := 30 * hours

theorem calories_difference :
  calories_burnt 5 - calories_burnt 2 = 90 :=
by
  sorry

end calories_difference_l1270_127051


namespace tax_collected_from_village_l1270_127000

-- Definitions according to the conditions in the problem
def MrWillamTax : ℝ := 500
def MrWillamPercentage : ℝ := 0.21701388888888893

-- The theorem to prove the total tax collected
theorem tax_collected_from_village : ∃ (total_collected : ℝ), MrWillamPercentage * total_collected = MrWillamTax ∧ total_collected = 2303.7037037037035 :=
sorry

end tax_collected_from_village_l1270_127000


namespace Ryan_reads_more_l1270_127002

theorem Ryan_reads_more 
  (total_pages_Ryan : ℕ)
  (days_in_week : ℕ)
  (pages_per_book_brother : ℕ)
  (books_per_day_brother : ℕ)
  (total_pages_brother : ℕ)
  (Ryan_books : ℕ)
  (Ryan_weeks : ℕ)
  (Brother_weeks : ℕ)
  (days_in_week_def : days_in_week = 7)
  (total_pages_Ryan_def : total_pages_Ryan = 2100)
  (pages_per_book_brother_def : pages_per_book_brother = 200)
  (books_per_day_brother_def : books_per_day_brother = 1)
  (Ryan_weeks_def : Ryan_weeks = 1)
  (Brother_weeks_def : Brother_weeks = 1)
  (total_pages_brother_def : total_pages_brother = pages_per_book_brother * days_in_week)
  : ((total_pages_Ryan / days_in_week) - (total_pages_brother / days_in_week) = 100) :=
by
  -- We provide the proof steps
  sorry

end Ryan_reads_more_l1270_127002


namespace total_money_correct_l1270_127068

def total_money_in_cents : ℕ :=
  let Cindy := 5 * 10 + 3 * 50
  let Eric := 3 * 25 + 2 * 100 + 1 * 50
  let Garrick := 8 * 5 + 7 * 1
  let Ivy := 60 * 1 + 5 * 25
  let TotalBeforeRemoval := Cindy + Eric + Garrick + Ivy
  let BeaumontRemoval := 2 * 10 + 3 * 5 + 10 * 1
  let EricRemoval := 1 * 25 + 1 * 50
  TotalBeforeRemoval - BeaumontRemoval - EricRemoval

theorem total_money_correct : total_money_in_cents = 637 := by
  sorry

end total_money_correct_l1270_127068


namespace number_of_students_with_type_B_l1270_127039

theorem number_of_students_with_type_B
  (total_students : ℕ)
  (students_with_type_A : total_students ≠ 0 ∧ total_students ≠ 0 → 2 * total_students = 90)
  (students_with_type_B : 2 * total_students = 90) :
  2/5 * total_students = 18 :=
by
  sorry

end number_of_students_with_type_B_l1270_127039


namespace tom_sells_games_for_225_42_usd_l1270_127050

theorem tom_sells_games_for_225_42_usd :
  let initial_usd := 200
  let usd_to_eur := 0.85
  let tripled_usd := initial_usd * 3
  let eur_value := tripled_usd * usd_to_eur
  let eur_to_jpy := 130
  let jpy_value := eur_value * eur_to_jpy
  let percent_sold := 0.40
  let sold_jpy_value := jpy_value * percent_sold
  let jpy_to_usd := 0.0085
  let sold_usd_value := sold_jpy_value * jpy_to_usd
  sold_usd_value = 225.42 :=
by
  sorry

end tom_sells_games_for_225_42_usd_l1270_127050


namespace average_weight_a_b_l1270_127030

variables (A B C : ℝ)

theorem average_weight_a_b (h1 : (A + B + C) / 3 = 43)
                          (h2 : (B + C) / 2 = 43)
                          (h3 : B = 37) :
                          (A + B) / 2 = 40 :=
by
  sorry

end average_weight_a_b_l1270_127030


namespace first_trial_addition_amounts_l1270_127055

-- Define the range and conditions for the biological agent addition amount.
def lower_bound : ℝ := 20
def upper_bound : ℝ := 30
def golden_ratio_method : ℝ := 0.618
def first_trial_addition_amount_1 : ℝ := lower_bound + (upper_bound - lower_bound) * golden_ratio_method
def first_trial_addition_amount_2 : ℝ := upper_bound - (upper_bound - lower_bound) * golden_ratio_method

-- Prove that the possible addition amounts for the first trial are 26.18g or 23.82g.
theorem first_trial_addition_amounts :
  (first_trial_addition_amount_1 = 26.18 ∨ first_trial_addition_amount_2 = 23.82) :=
by
  -- Placeholder for the proof.
  sorry

end first_trial_addition_amounts_l1270_127055


namespace amy_books_l1270_127006

theorem amy_books (maddie_books : ℕ) (luisa_books : ℕ) (amy_luisa_more_than_maddie : ℕ) (h1 : maddie_books = 15) (h2 : luisa_books = 18) (h3 : amy_luisa_more_than_maddie = maddie_books + 9) : ∃ (amy_books : ℕ), amy_books = amy_luisa_more_than_maddie - luisa_books ∧ amy_books = 6 :=
by
  have total_books := 24
  sorry

end amy_books_l1270_127006


namespace geometric_sequence_sum_terms_l1270_127019

noncomputable def geom_sequence_sum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a 0 * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_sum_terms
  (a : ℕ → ℝ) (q : ℝ)
  (h_q_nonzero : q ≠ 1)
  (S3_eq : geom_sequence_sum a q 3 = 8)
  (S6_eq : geom_sequence_sum a q 6 = 7)
  : a 6 * q ^ 6 + a 7 * q ^ 7 + a 8 * q ^ 8 = 1 / 8 := sorry

end geometric_sequence_sum_terms_l1270_127019


namespace Michelle_silver_beads_count_l1270_127037

theorem Michelle_silver_beads_count 
  (total_beads : ℕ) (blue_beads : ℕ) (red_beads : ℕ) (white_beads : ℕ) (silver_beads : ℕ)
  (h1 : total_beads = 40)
  (h2 : blue_beads = 5)
  (h3 : red_beads = 2 * blue_beads)
  (h4 : white_beads = blue_beads + red_beads)
  (h5 : silver_beads = total_beads - (blue_beads + red_beads + white_beads)) :
  silver_beads = 10 :=
by
  sorry

end Michelle_silver_beads_count_l1270_127037


namespace find_y_l1270_127011

-- Definitions of angles and the given problem.
def angle_ABC : ℝ := 90
def angle_ABD (y : ℝ) : ℝ := 3 * y
def angle_DBC (y : ℝ) : ℝ := 2 * y

-- The theorem stating the problem
theorem find_y (y : ℝ) (h1 : angle_ABC = 90) (h2 : angle_ABD y + angle_DBC y = angle_ABC) : y = 18 :=
  by 
  sorry

end find_y_l1270_127011


namespace sum_roots_of_quadratic_eq_l1270_127090

theorem sum_roots_of_quadratic_eq (a b c: ℝ) (x: ℝ) :
    (a = 1) →
    (b = -7) →
    (c = -9) →
    (x ^ 2 - 7 * x + 2 = 11) →
    (∃ r1 r2 : ℝ, x ^ 2 - 7 * x - 9 = 0 ∧ r1 + r2 = 7) :=
by
  sorry

end sum_roots_of_quadratic_eq_l1270_127090


namespace average_temperature_Robertson_l1270_127015

def temperatures : List ℝ := [18, 21, 19, 22, 20]

noncomputable def average (temps : List ℝ) : ℝ :=
  (temps.sum) / (temps.length)

theorem average_temperature_Robertson :
  average temperatures = 20.0 :=
by
  sorry

end average_temperature_Robertson_l1270_127015
