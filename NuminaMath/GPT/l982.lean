import Mathlib

namespace NUMINAMATH_GPT_rice_in_each_container_ounces_l982_98293

-- Given conditions
def total_rice_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- Problem statement: proving the amount of rice in each container in ounces
theorem rice_in_each_container_ounces :
  (total_rice_pounds / num_containers) * pounds_to_ounces = 25 :=
by sorry

end NUMINAMATH_GPT_rice_in_each_container_ounces_l982_98293


namespace NUMINAMATH_GPT_range_of_a_l982_98206

theorem range_of_a (a : ℝ) (h1 : a ≤ 1)
(h2 : ∃ n₁ n₂ n₃ : ℤ, a ≤ n₁ ∧ n₁ < n₂ ∧ n₂ < n₃ ∧ n₃ ≤ 2 - a
  ∧ (∀ x : ℤ, a ≤ x ∧ x ≤ 2 - a → x = n₁ ∨ x = n₂ ∨ x = n₃)) :
  -1 < a ∧ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l982_98206


namespace NUMINAMATH_GPT_min_value_expression_l982_98227

theorem min_value_expression (x y : ℝ) (hx : |x| < 1) (hy : |y| < 2) (hxy : x * y = 1) : 
  ∃ k, k = 4 ∧ (∀ z, z = (1 / (1 - x^2) + 4 / (4 - y^2)) → z ≥ k) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l982_98227


namespace NUMINAMATH_GPT_tunnel_length_l982_98259

noncomputable def train_length : Real := 2 -- miles
noncomputable def time_to_exit_tunnel : Real := 4 -- minutes
noncomputable def train_speed : Real := 120 -- miles per hour

theorem tunnel_length : ∃ tunnel_length : Real, tunnel_length = 6 :=
  by
  -- We use the conditions given:
  let speed_in_miles_per_minute := train_speed / 60 -- converting speed from miles per hour to miles per minute
  let distance_travelled_by_front_in_4_min := speed_in_miles_per_minute * time_to_exit_tunnel
  let tunnel_length := distance_travelled_by_front_in_4_min - train_length
  have h : tunnel_length = 6 := by sorry
  exact ⟨tunnel_length, h⟩

end NUMINAMATH_GPT_tunnel_length_l982_98259


namespace NUMINAMATH_GPT_katie_remaining_juice_l982_98229

-- Define the initial condition: Katie initially has 5 gallons of juice
def initial_gallons : ℚ := 5

-- Define the amount of juice given to Mark
def juice_given : ℚ := 18 / 7

-- Define the expected remaining fraction of juice
def expected_remaining_gallons : ℚ := 17 / 7

-- The theorem statement that Katie should have 17/7 gallons of juice left
theorem katie_remaining_juice : initial_gallons - juice_given = expected_remaining_gallons := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_katie_remaining_juice_l982_98229


namespace NUMINAMATH_GPT_annieka_free_throws_l982_98252

theorem annieka_free_throws (deshawn_throws : ℕ) (kayla_factor : ℝ) (annieka_diff : ℕ) (ht1 : deshawn_throws = 12) (ht2 : kayla_factor = 1.5) (ht3 : annieka_diff = 4) :
  ∃ (annieka_throws : ℕ), annieka_throws = (⌊deshawn_throws * kayla_factor⌋.toNat - annieka_diff) :=
by
  sorry

end NUMINAMATH_GPT_annieka_free_throws_l982_98252


namespace NUMINAMATH_GPT_solve_equation_l982_98289

theorem solve_equation (x : ℝ) (h : 3 * x ≠ 0) (h2 : x + 2 ≠ 0) : (2 / (3 * x) = 1 / (x + 2)) ↔ x = 4 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l982_98289


namespace NUMINAMATH_GPT_distance_from_tangency_to_tangent_theorem_l982_98294

noncomputable def distance_from_tangency_to_tangent (R r : ℝ) : ℝ :=
  2 * R * r / (R + r)

theorem distance_from_tangency_to_tangent_theorem (R r : ℝ) :
  ∃ d : ℝ, d = distance_from_tangency_to_tangent R r :=
by
  use 2 * R * r / (R + r)
  sorry

end NUMINAMATH_GPT_distance_from_tangency_to_tangent_theorem_l982_98294


namespace NUMINAMATH_GPT_proof_problem_l982_98234

def operation1 (x : ℝ) := 9 - x
def operation2 (x : ℝ) := x - 9

theorem proof_problem : operation2 (operation1 15) = -15 := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l982_98234


namespace NUMINAMATH_GPT_Janet_sold_six_action_figures_l982_98288

variable {x : ℕ}

theorem Janet_sold_six_action_figures
  (h₁ : 10 - x + 4 + 2 * (10 - x + 4) = 24) :
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_Janet_sold_six_action_figures_l982_98288


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l982_98235

theorem average_of_remaining_numbers 
    (nums : List ℝ) 
    (h_length : nums.length = 12) 
    (h_avg_90 : (nums.sum) / 12 = 90) 
    (h_contains_65_85 : 65 ∈ nums ∧ 85 ∈ nums) 
    (nums' := nums.erase 65)
    (nums'' := nums'.erase 85) : 
   nums''.length = 10 ∧ nums''.sum / 10 = 93 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l982_98235


namespace NUMINAMATH_GPT_voldemort_spending_l982_98241

theorem voldemort_spending :
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  (book_price_paid = (original_book_price / 8)) ∧ (total_spent = 24) :=
by
  let book_price_paid := 8
  let original_book_price := 64
  let journal_price := 2 * book_price_paid
  let total_spent := book_price_paid + journal_price
  have h1 : book_price_paid = (original_book_price / 8) := by
    sorry
  have h2 : total_spent = 24 := by
    sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_voldemort_spending_l982_98241


namespace NUMINAMATH_GPT_simplify_logarithmic_expression_l982_98225

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 3 / Real.log 9 + 1)

theorem simplify_logarithmic_expression :
  simplify_expression = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_logarithmic_expression_l982_98225


namespace NUMINAMATH_GPT_sum_of_two_equal_sides_is_4_l982_98244

noncomputable def isosceles_right_triangle (a c : ℝ) : Prop :=
  c = 2.8284271247461903 ∧ c ^ 2 = 2 * (a ^ 2)

theorem sum_of_two_equal_sides_is_4 :
  ∃ a : ℝ, isosceles_right_triangle a 2.8284271247461903 ∧ 2 * a = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_equal_sides_is_4_l982_98244


namespace NUMINAMATH_GPT_zeros_of_f_l982_98296

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- State the theorem about its roots
theorem zeros_of_f : ∃ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_zeros_of_f_l982_98296


namespace NUMINAMATH_GPT_number_of_classes_l982_98220

theorem number_of_classes
  (s : ℕ)    -- s: number of students in each class
  (bpm : ℕ) -- bpm: books per month per student
  (months : ℕ) -- months: number of months in a year
  (total_books : ℕ) -- total_books: total books read by the entire student body in a year
  (H1 : bpm = 5)
  (H2 : months = 12)
  (H3 : total_books = 60)
  (H4 : total_books = s * bpm * months)
: s = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_classes_l982_98220


namespace NUMINAMATH_GPT_selection_of_representatives_l982_98264

theorem selection_of_representatives 
  (females : ℕ) (males : ℕ)
  (h_females : females = 3) (h_males : males = 4) :
  (females ≥ 1 ∧ males ≥ 1) →
  (females * (males * (males - 1) / 2) + (females * (females - 1) / 2 * males) = 30) := 
by
  sorry

end NUMINAMATH_GPT_selection_of_representatives_l982_98264


namespace NUMINAMATH_GPT_smallest_number_among_10_11_12_l982_98276

theorem smallest_number_among_10_11_12 : min (min 10 11) 12 = 10 :=
by sorry

end NUMINAMATH_GPT_smallest_number_among_10_11_12_l982_98276


namespace NUMINAMATH_GPT_find_percentage_l982_98246

theorem find_percentage (P : ℝ) (h: (20 / 100) * 580 = (P / 100) * 120 + 80) : P = 30 := 
by
  sorry

end NUMINAMATH_GPT_find_percentage_l982_98246


namespace NUMINAMATH_GPT_sixty_percent_of_number_l982_98222

theorem sixty_percent_of_number (N : ℚ) (h : ((1 / 6) * (2 / 3) * (3 / 4) * (5 / 7) * N = 25)) :
  0.60 * N = 252 := sorry

end NUMINAMATH_GPT_sixty_percent_of_number_l982_98222


namespace NUMINAMATH_GPT_solve_for_y_l982_98299

theorem solve_for_y (x y : ℝ) (h : 4 * x + y = 9) : y = 9 - 4 * x :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l982_98299


namespace NUMINAMATH_GPT_least_multiple_72_112_199_is_310_l982_98232

theorem least_multiple_72_112_199_is_310 :
  ∃ k : ℕ, (112 ∣ k * 72) ∧ (199 ∣ k * 72) ∧ k = 310 := 
by
  sorry

end NUMINAMATH_GPT_least_multiple_72_112_199_is_310_l982_98232


namespace NUMINAMATH_GPT_domain_of_f_l982_98257

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f :
  {x : ℝ | x + 1 > 0 ∧ Real.log (x + 1) ≠ 0 ∧ 4 - x^2 ≥ 0} =
  {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l982_98257


namespace NUMINAMATH_GPT_two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l982_98280

noncomputable def rooks_non_attacking : Nat :=
  8 * 8 * 7 * 7 / 2

theorem two_rooks_non_attacking : rooks_non_attacking = 1568 := by
  sorry

noncomputable def kings_non_attacking : Nat :=
  (4 * 60 + 24 * 58 + 36 * 55 + 24 * 55 + 4 * 50) / 2

theorem two_kings_non_attacking : kings_non_attacking = 1806 := by
  sorry

noncomputable def bishops_non_attacking : Nat :=
  (28 * 25 + 20 * 54 + 12 * 52 + 4 * 50) / 2

theorem two_bishops_non_attacking : bishops_non_attacking = 1736 := by
  sorry

noncomputable def knights_non_attacking : Nat :=
  (4 * 61 + 8 * 60 + 20 * 59 + 16 * 57 + 15 * 55) / 2

theorem two_knights_non_attacking : knights_non_attacking = 1848 := by
  sorry

noncomputable def queens_non_attacking : Nat :=
  (28 * 42 + 20 * 40 + 12 * 38 + 4 * 36) / 2

theorem two_queens_non_attacking : queens_non_attacking = 1288 := by
  sorry

end NUMINAMATH_GPT_two_rooks_non_attacking_two_kings_non_attacking_two_bishops_non_attacking_two_knights_non_attacking_two_queens_non_attacking_l982_98280


namespace NUMINAMATH_GPT_ratio_Bipin_Alok_l982_98226

-- Definitions based on conditions
def Alok_age : Nat := 5
def Chandan_age : Nat := 10
def Bipin_age : Nat := 30
def Bipin_age_condition (B C : Nat) : Prop := B + 10 = 2 * (C + 10)

-- Statement to prove
theorem ratio_Bipin_Alok : 
  Bipin_age_condition Bipin_age Chandan_age -> 
  Alok_age = 5 -> 
  Chandan_age = 10 -> 
  Bipin_age / Alok_age = 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_Bipin_Alok_l982_98226


namespace NUMINAMATH_GPT_linear_system_substitution_correct_l982_98267

theorem linear_system_substitution_correct (x y : ℝ)
  (h1 : y = x - 1)
  (h2 : x + 2 * y = 7) :
  x + 2 * x - 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_linear_system_substitution_correct_l982_98267


namespace NUMINAMATH_GPT_books_sold_in_january_l982_98272

theorem books_sold_in_january (J : ℕ) 
  (h_avg : (J + 16 + 17) / 3 = 16) : J = 15 :=
sorry

end NUMINAMATH_GPT_books_sold_in_january_l982_98272


namespace NUMINAMATH_GPT_second_interest_rate_l982_98224

theorem second_interest_rate (P1 P2 : ℝ) (r : ℝ) (total_amount total_income: ℝ) (h1 : total_amount = 2500)
  (h2 : P1 = 1500.0000000000007) (h3 : total_income = 135) :
  P2 = total_amount - P1 →
  P1 * 0.05 = 75 →
  P2 * r = 60 →
  r = 0.06 :=
sorry

end NUMINAMATH_GPT_second_interest_rate_l982_98224


namespace NUMINAMATH_GPT_abs_fraction_inequality_l982_98217

theorem abs_fraction_inequality (x : ℝ) :
  (abs ((3 * x - 4) / (x - 2)) > 3) ↔
  (x ∈ Set.Iio (5 / 3) ∪ Set.Ioo (5 / 3) 2 ∪ Set.Ioi 2) :=
by 
  sorry

end NUMINAMATH_GPT_abs_fraction_inequality_l982_98217


namespace NUMINAMATH_GPT_solution_set_f_pos_min_a2_b2_c2_l982_98277

def f (x : ℝ) : ℝ := |2 * x + 3| - |x - 1|

theorem solution_set_f_pos : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -3 / 2 ∨ -2 / 3 < x } := 
sorry

theorem min_a2_b2_c2 (a b c : ℝ) (h : a + 2 * b + 3 * c = 5) : 
  a^2 + b^2 + c^2 ≥ 25 / 14 :=
sorry

end NUMINAMATH_GPT_solution_set_f_pos_min_a2_b2_c2_l982_98277


namespace NUMINAMATH_GPT_choir_population_l982_98297

theorem choir_population 
  (female_students : ℕ) 
  (male_students : ℕ) 
  (choir_multiple : ℕ) 
  (total_students_orchestra : ℕ := female_students + male_students)
  (total_students_choir : ℕ := choir_multiple * total_students_orchestra)
  (h_females : female_students = 18) 
  (h_males : male_students = 25) 
  (h_multiple : choir_multiple = 3) : 
  total_students_choir = 129 := 
by
  -- The proof of the theorem will be done here.
  sorry

end NUMINAMATH_GPT_choir_population_l982_98297


namespace NUMINAMATH_GPT_product_mn_l982_98215

-- Λet θ1 be the angle L1 makes with the positive x-axis.
-- Λet θ2 be the angle L2 makes with the positive x-axis.
-- Given that θ1 = 3 * θ2 and m = 6 * n.
-- Using the tangent triple angle formula: tan(3θ) = (3 * tan(θ) - tan^3(θ)) / (1 - 3 * tan^2(θ))
-- We need to prove mn = 9/17.

noncomputable def mn_product_condition (θ1 θ2 : ℝ) (m n : ℝ) : Prop :=
θ1 = 3 * θ2 ∧ m = 6 * n ∧ m = Real.tan θ1 ∧ n = Real.tan θ2

theorem product_mn (θ1 θ2 : ℝ) (m n : ℝ) (h : mn_product_condition θ1 θ2 m n) :
  m * n = 9 / 17 :=
sorry

end NUMINAMATH_GPT_product_mn_l982_98215


namespace NUMINAMATH_GPT_range_of_a_for_increasing_l982_98201

noncomputable def f (a x : ℝ) : ℝ := x * abs (2 * a - x) + 2 * x

theorem range_of_a_for_increasing (a : ℝ) :
  -1 ≤ a ∧ a ≤ 1 ↔ ∀ x y : ℝ, x < y → f a x ≤ f a y :=
sorry

end NUMINAMATH_GPT_range_of_a_for_increasing_l982_98201


namespace NUMINAMATH_GPT_maximize_area_center_coordinates_l982_98238

theorem maximize_area_center_coordinates (k : ℝ) :
  (∃ r : ℝ, r^2 = 1 - (3/4) * k^2 ∧ r ≥ 0) →
  ((k = 0) → ∃ a b : ℝ, (a = 0 ∧ b = -1)) :=
by
  sorry

end NUMINAMATH_GPT_maximize_area_center_coordinates_l982_98238


namespace NUMINAMATH_GPT_solve_2019_gon_l982_98245

noncomputable def problem_2019_gon (x : ℕ → ℕ) : Prop :=
  (∀ i : ℕ, (x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) + x (i+5) + x (i+6) + x (i+7) + x (i+8) = 300))
  ∧ (x 18 = 19)
  ∧ (x 19 = 20)

theorem solve_2019_gon :
  ∀ x : ℕ → ℕ,
  problem_2019_gon x →
  x 2018 = 61 :=
by sorry

end NUMINAMATH_GPT_solve_2019_gon_l982_98245


namespace NUMINAMATH_GPT_sum_of_triangles_l982_98243

def triangle (a b c : ℕ) : ℕ := a * b + c

theorem sum_of_triangles :
  triangle 3 2 5 + triangle 4 1 7 = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_triangles_l982_98243


namespace NUMINAMATH_GPT_value_of_a5_l982_98205

theorem value_of_a5 {a_1 a_3 a_5 : ℤ} (n : ℕ) (hn : n = 8) (h1 : (1 - x)^n = 1 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8) (h_ratio : a_1 / a_3 = 1 / 7) :
  a_5 = -56 := 
sorry

end NUMINAMATH_GPT_value_of_a5_l982_98205


namespace NUMINAMATH_GPT_sum_of_ages_is_37_l982_98266

def maries_age : ℕ := 12
def marcos_age (M : ℕ) : ℕ := 2 * M + 1

theorem sum_of_ages_is_37 : maries_age + marcos_age maries_age = 37 := 
by
  -- Inserting the proof details
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_37_l982_98266


namespace NUMINAMATH_GPT_complement_M_l982_98286

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M (U M : Set ℝ) : (U \ M) = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_M_l982_98286


namespace NUMINAMATH_GPT_test_point_selection_l982_98284

theorem test_point_selection (x_1 x_2 : ℝ)
    (interval_begin interval_end : ℝ) (h_interval : interval_begin = 2 ∧ interval_end = 4)
    (h_better_result : x_1 < x_2 ∨ x_1 > x_2)
    (h_test_points : (x_1 = interval_begin + 0.618 * (interval_end - interval_begin) ∧ 
                     x_2 = interval_begin + interval_end - x_1) ∨ 
                    (x_1 = interval_begin + interval_end - (interval_begin + 0.618 * (interval_end - interval_begin)) ∧ 
                     x_2 = interval_begin + 0.618 * (interval_end - interval_begin)))
  : ∃ x_3, x_3 = 3.528 ∨ x_3 = 2.472 := by
    sorry

end NUMINAMATH_GPT_test_point_selection_l982_98284


namespace NUMINAMATH_GPT_range_of_ab_min_value_of_ab_plus_inv_ab_l982_98270

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 0 < a * b ∧ a * b ≤ 1 / 4 :=
sorry

theorem min_value_of_ab_plus_inv_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (∃ ab, ab = a * b ∧ ab + 1 / ab = 17 / 4) :=
sorry

end NUMINAMATH_GPT_range_of_ab_min_value_of_ab_plus_inv_ab_l982_98270


namespace NUMINAMATH_GPT_mean_score_classes_is_82_l982_98240

theorem mean_score_classes_is_82
  (F S : ℕ)
  (f s : ℕ)
  (hF : F = 90)
  (hS : S = 75)
  (hf_ratio : f * 6 = s * 5)
  (hf_total : f + s = 66) :
  ((F * f + S * s) / (f + s) : ℚ) = 82 :=
by
  sorry

end NUMINAMATH_GPT_mean_score_classes_is_82_l982_98240


namespace NUMINAMATH_GPT_negative_x_is_positive_l982_98263

theorem negative_x_is_positive (x : ℝ) (hx : x < 0) : -x > 0 :=
sorry

end NUMINAMATH_GPT_negative_x_is_positive_l982_98263


namespace NUMINAMATH_GPT_total_vases_l982_98291

theorem total_vases (vases_per_day : ℕ) (days : ℕ) (total_vases : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days = 16) 
  (h3 : total_vases = vases_per_day * days) : 
  total_vases = 256 := 
by 
  sorry

end NUMINAMATH_GPT_total_vases_l982_98291


namespace NUMINAMATH_GPT_union_of_A_and_B_l982_98292

variable {α : Type*}

def A (x : ℝ) : Prop := x - 1 > 0
def B (x : ℝ) : Prop := 0 < x ∧ x ≤ 3

theorem union_of_A_and_B : ∀ x : ℝ, (A x ∨ B x) ↔ (0 < x) :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l982_98292


namespace NUMINAMATH_GPT_suzanna_distance_ridden_l982_98239

theorem suzanna_distance_ridden (rate_per_5minutes : ℝ) (time_minutes : ℕ) (total_distance : ℝ) (units_per_interval : ℕ) (interval_distance : ℝ) :
  rate_per_5minutes = 0.75 → time_minutes = 45 → units_per_interval = 5 → interval_distance = 0.75 → total_distance = (time_minutes / units_per_interval) * interval_distance → total_distance = 6.75 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_suzanna_distance_ridden_l982_98239


namespace NUMINAMATH_GPT_max_n_for_coloring_l982_98290

noncomputable def maximum_n : ℕ :=
  11

theorem max_n_for_coloring :
  ∃ n : ℕ, (n = maximum_n) ∧ ∀ k ∈ Finset.range n, 
  (∃ x y : ℕ, 1 ≤ x ∧ x ≤ 14 ∧ 1 ≤ y ∧ y ≤ 14 ∧ (x - y = k ∨ y - x = k) ∧ x ≠ y) ∧
  (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 14 ∧ 1 ≤ b ∧ b ≤ 14 ∧ (a - b = k ∨ b - a = k) ∧ a ≠ b) :=
sorry

end NUMINAMATH_GPT_max_n_for_coloring_l982_98290


namespace NUMINAMATH_GPT_volunteer_hours_per_year_l982_98233

def volunteers_per_month : ℕ := 2
def hours_per_session : ℕ := 3
def months_per_year : ℕ := 12

theorem volunteer_hours_per_year :
  volunteers_per_month * months_per_year * hours_per_session = 72 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_volunteer_hours_per_year_l982_98233


namespace NUMINAMATH_GPT_russian_pairing_probability_l982_98207

-- Definitions based on conditions
def total_players : ℕ := 10
def russian_players : ℕ := 4
def non_russian_players : ℕ := total_players - russian_players

-- Probability calculation as a hypothesis
noncomputable def pairing_probability (rs: ℕ) (ns: ℕ) : ℚ :=
  (rs * (rs - 1)) / (total_players * (total_players - 1))

theorem russian_pairing_probability :
  pairing_probability russian_players non_russian_players = 1 / 21 :=
sorry

end NUMINAMATH_GPT_russian_pairing_probability_l982_98207


namespace NUMINAMATH_GPT_min_equilateral_triangles_l982_98248

theorem min_equilateral_triangles (s : ℝ) (S : ℝ) :
  s = 1 → S = 15 → 
  225 = (S / s) ^ 2 :=
by
  intros hs hS
  rw [hs, hS]
  simp
  sorry

end NUMINAMATH_GPT_min_equilateral_triangles_l982_98248


namespace NUMINAMATH_GPT_ab_cd_not_prime_l982_98223

theorem ab_cd_not_prime (a b c d : ℕ) (ha : a > b) (hb : b > c) (hc : c > d) (hd : d > 0)
  (h : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : ¬ Nat.Prime (a * b + c * d) := 
sorry

end NUMINAMATH_GPT_ab_cd_not_prime_l982_98223


namespace NUMINAMATH_GPT_new_average_of_subtracted_elements_l982_98228

theorem new_average_of_subtracted_elements (a b c d e : ℝ) 
  (h_average : (a + b + c + d + e) / 5 = 5) 
  (new_a : ℝ := a - 2) 
  (new_b : ℝ := b - 2) 
  (new_c : ℝ := c - 2) 
  (new_d : ℝ := d - 2) :
  (new_a + new_b + new_c + new_d + e) / 5 = 3.4 := 
by 
  sorry

end NUMINAMATH_GPT_new_average_of_subtracted_elements_l982_98228


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l982_98218

def point_symmetric_to_x_axis (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match p with
  | (x, y, z) => (x, -y, -z)

theorem symmetric_point_coordinates :
  point_symmetric_to_x_axis (-2, 1, 4) = (-2, -1, -4) := by
  sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l982_98218


namespace NUMINAMATH_GPT_negation_of_abs_x_minus_2_lt_3_l982_98275

theorem negation_of_abs_x_minus_2_lt_3 :
  ¬ (∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_abs_x_minus_2_lt_3_l982_98275


namespace NUMINAMATH_GPT_a_x1_x2_x13_eq_zero_l982_98281

theorem a_x1_x2_x13_eq_zero {a x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 : ℤ}
  (h1: a = (1 + x1) * (1 + x2) * (1 + x3) * (1 + x4) * (1 + x5) * (1 + x6) * (1 + x7) *
             (1 + x8) * (1 + x9) * (1 + x10) * (1 + x11) * (1 + x12) * (1 + x13))
  (h2: a = (1 - x1) * (1 - x2) * (1 - x3) * (1 - x4) * (1 - x5) * (1 - x6) * (1 - x7) *
             (1 - x8) * (1 - x9) * (1 - x10) * (1 - x11) * (1 - x12) * (1 - x13)) :
  a * x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 = 0 := by
  sorry

end NUMINAMATH_GPT_a_x1_x2_x13_eq_zero_l982_98281


namespace NUMINAMATH_GPT_age_of_b_is_6_l982_98273

theorem age_of_b_is_6 (x : ℕ) (h1 : 5 * x / 3 * x = 5 / 3)
                         (h2 : (5 * x + 2) / (3 * x + 2) = 3 / 2) : 3 * x = 6 := 
by
  sorry

end NUMINAMATH_GPT_age_of_b_is_6_l982_98273


namespace NUMINAMATH_GPT_consecutive_numbers_difference_l982_98255

theorem consecutive_numbers_difference :
  ∃ (n : ℕ), (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → (n + 5 - n = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_consecutive_numbers_difference_l982_98255


namespace NUMINAMATH_GPT_minimize_total_price_l982_98242

noncomputable def total_price (a : ℝ) (m x : ℝ) : ℝ :=
  a * ((m / 2 + x)^2 + (m / 2 - x)^2)

theorem minimize_total_price (a m : ℝ) : 
  ∃ y : ℝ, (∀ x, total_price a m x ≥ y) ∧ y = total_price a m 0 :=
by
  sorry

end NUMINAMATH_GPT_minimize_total_price_l982_98242


namespace NUMINAMATH_GPT_max_n_arithmetic_sequences_l982_98204

theorem max_n_arithmetic_sequences (a b : ℕ → ℤ) 
  (ha : ∀ n, a n = 1 + (n - 1) * 1)  -- Assuming x = 1 for simplicity, as per solution x = y = 1
  (hb : ∀ n, b n = 1 + (n - 1) * 1)  -- Assuming y = 1
  (a1 : a 1 = 1)
  (b1 : b 1 = 1)
  (a2_leq_b2 : a 2 ≤ b 2)
  (hn : ∃ n, a n * b n = 1764) :
  ∃ n, n = 44 ∧ a n * b n = 1764 :=
by
  sorry

end NUMINAMATH_GPT_max_n_arithmetic_sequences_l982_98204


namespace NUMINAMATH_GPT_range_of_a_l982_98269

theorem range_of_a (a : ℝ) (h : a ≤ 1) :
  (∃! n : ℕ, n = (2 - a) - a + 1) → -1 < a ∧ a ≤ 0 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l982_98269


namespace NUMINAMATH_GPT_robin_has_43_packages_of_gum_l982_98231

theorem robin_has_43_packages_of_gum (P : ℕ) (h1 : 23 * P + 8 = 997) : P = 43 :=
by
  sorry

end NUMINAMATH_GPT_robin_has_43_packages_of_gum_l982_98231


namespace NUMINAMATH_GPT_pearJuicePercentageCorrect_l982_98209

-- Define the conditions
def dozen : ℕ := 12
def pears := dozen
def oranges := dozen
def pearJuiceFrom3Pears : ℚ := 8
def orangeJuiceFrom2Oranges : ℚ := 10
def juiceBlendPears : ℕ := 4
def juiceBlendOranges : ℕ := 4
def pearJuicePerPear : ℚ := pearJuiceFrom3Pears / 3
def orangeJuicePerOrange : ℚ := orangeJuiceFrom2Oranges / 2
def totalPearJuice : ℚ := juiceBlendPears * pearJuicePerPear
def totalOrangeJuice : ℚ := juiceBlendOranges * orangeJuicePerOrange
def totalJuice : ℚ := totalPearJuice + totalOrangeJuice

-- Prove that the percentage of pear juice in the blend is 34.78%
theorem pearJuicePercentageCorrect : 
  (totalPearJuice / totalJuice) * 100 = 34.78 := by
  sorry

end NUMINAMATH_GPT_pearJuicePercentageCorrect_l982_98209


namespace NUMINAMATH_GPT_problem_1_problem_2_l982_98211

theorem problem_1 (A B C : ℝ) (h_cond : (abs (B - A)) * (abs (C - A)) * (Real.cos A) = 3 * (abs (A - B)) * (abs (C - B)) * (Real.cos B)) : 
  (Real.tan B = 3 * Real.tan A) := 
sorry

theorem problem_2 (A B C : ℝ) (h_cosC : Real.cos C = Real.sqrt 5 / 5) (h_tanB : Real.tan B = 3 * Real.tan A) : 
  (A = Real.pi / 4) := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l982_98211


namespace NUMINAMATH_GPT_age_difference_l982_98237

-- defining the conditions
variable (A B : ℕ)
variable (h1 : B = 35)
variable (h2 : A + 10 = 2 * (B - 10))

-- the proof statement
theorem age_difference : A - B = 5 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l982_98237


namespace NUMINAMATH_GPT_symmetric_point_l982_98251

-- Define the given point M
def point_M : ℝ × ℝ × ℝ := (1, 0, -1)

-- Define the line in parametric form
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (3.5 + 2 * t, 1.5 + 2 * t, 0)

-- Define the symmetric point M'
def point_M' : ℝ × ℝ × ℝ := (2, -1, 1)

-- Statement: Prove that M' is the symmetric point to M with respect to the given line
theorem symmetric_point (M M' : ℝ × ℝ × ℝ) (line : ℝ → ℝ × ℝ × ℝ) :
  M = (1, 0, -1) →
  line (t) = (3.5 + 2 * t, 1.5 + 2 * t, 0) →
  M' = (2, -1, 1) :=
sorry

end NUMINAMATH_GPT_symmetric_point_l982_98251


namespace NUMINAMATH_GPT_parabola_focus_directrix_l982_98271

noncomputable def parabola_distance_property (p : ℝ) (hp : 0 < p) : Prop :=
  let focus := (2 * p, 0)
  let directrix := -2 * p
  let distance := 4 * p
  p = distance / 4

-- Theorem: Given a parabola with equation y^2 = 8px (p > 0), p represents 1/4 of the distance from the focus to the directrix.
theorem parabola_focus_directrix (p : ℝ) (hp : 0 < p) : parabola_distance_property p hp :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_directrix_l982_98271


namespace NUMINAMATH_GPT_fifteenth_even_multiple_of_5_l982_98279

theorem fifteenth_even_multiple_of_5 : 15 * 2 * 5 = 150 := by
  sorry

end NUMINAMATH_GPT_fifteenth_even_multiple_of_5_l982_98279


namespace NUMINAMATH_GPT_count_multiples_of_4_l982_98214

/-- 
Prove that the number of multiples of 4 between 100 and 300 inclusive is 49.
-/
theorem count_multiples_of_4 : 
  ∃ n : ℕ, (∀ k : ℕ, 100 ≤ 4 * k ∧ 4 * k ≤ 300 ↔ k = 26 + n) ∧ n = 48 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_of_4_l982_98214


namespace NUMINAMATH_GPT_arithmetic_progression_sum_at_least_66_l982_98247

-- Define the sum of the first n terms of an arithmetic progression
def sum_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

-- Define the conditions for the arithmetic progression
def arithmetic_prog_conditions (a1 d : ℤ) (n : ℕ) :=
  sum_first_n_terms a1 d n ≥ 66

-- The main theorem to prove
theorem arithmetic_progression_sum_at_least_66 (n : ℕ) :
  (n >= 3 ∧ n <= 14) → arithmetic_prog_conditions 25 (-3) n :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_at_least_66_l982_98247


namespace NUMINAMATH_GPT_find_larger_number_l982_98256

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1311) (h2 : L = 11 * S + 11) : L = 1441 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l982_98256


namespace NUMINAMATH_GPT_binomial_coefficient_30_3_l982_98282

theorem binomial_coefficient_30_3 :
  Nat.choose 30 3 = 4060 := 
by 
  sorry

end NUMINAMATH_GPT_binomial_coefficient_30_3_l982_98282


namespace NUMINAMATH_GPT_largest_multiple_of_45_l982_98212

theorem largest_multiple_of_45 (m : ℕ) 
  (h₁ : m % 45 = 0) 
  (h₂ : ∀ d : ℕ, d ∈ m.digits 10 → d = 8 ∨ d = 0) : 
  m / 45 = 197530 := 
sorry

end NUMINAMATH_GPT_largest_multiple_of_45_l982_98212


namespace NUMINAMATH_GPT_max_plus_min_eq_four_l982_98210

theorem max_plus_min_eq_four {g : ℝ → ℝ} (h_odd_function : ∀ x, g (-x) = -g x)
  (M m : ℝ) (h_f : ∀ x, 2 + g x ≤ M) (h_f' : ∀ x, m ≤ 2 + g x) :
  M + m = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_plus_min_eq_four_l982_98210


namespace NUMINAMATH_GPT_probability_angie_carlos_two_seats_apart_l982_98230

theorem probability_angie_carlos_two_seats_apart :
  let people := ["Angie", "Bridget", "Carlos", "Diego", "Edwin"]
  let table_size := people.length
  let total_arrangements := (Nat.factorial (table_size - 1))
  let favorable_arrangements := 2 * (Nat.factorial (table_size - 2))
  total_arrangements > 0 ∧
  (favorable_arrangements / total_arrangements : ℚ) = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_angie_carlos_two_seats_apart_l982_98230


namespace NUMINAMATH_GPT_paving_cost_l982_98278

variable (L : ℝ) (W : ℝ) (R : ℝ)

def area (L W : ℝ) := L * W
def cost (A R : ℝ) := A * R

theorem paving_cost (hL : L = 5) (hW : W = 4.75) (hR : R = 900) : cost (area L W) R = 21375 :=
by
  sorry

end NUMINAMATH_GPT_paving_cost_l982_98278


namespace NUMINAMATH_GPT_total_contribution_is_1040_l982_98200

-- Definitions of contributions based on conditions.
def Niraj_contribution : ℕ := 80
def Brittany_contribution : ℕ := 3 * Niraj_contribution
def Angela_contribution : ℕ := 3 * Brittany_contribution

-- Statement to prove that total contribution is $1040.
theorem total_contribution_is_1040 : Niraj_contribution + Brittany_contribution + Angela_contribution = 1040 := by
  sorry

end NUMINAMATH_GPT_total_contribution_is_1040_l982_98200


namespace NUMINAMATH_GPT_problem_solution_l982_98261

theorem problem_solution : (90 + 5) * (12 / (180 / (3^2))) = 57 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l982_98261


namespace NUMINAMATH_GPT_downstream_speed_l982_98221

-- Definitions based on the conditions
def V_m : ℝ := 50 -- speed of the man in still water
def V_upstream : ℝ := 45 -- speed of the man when rowing upstream

-- The statement to prove
theorem downstream_speed : ∃ (V_s V_downstream : ℝ), V_upstream = V_m - V_s ∧ V_downstream = V_m + V_s ∧ V_downstream = 55 := 
by
  sorry

end NUMINAMATH_GPT_downstream_speed_l982_98221


namespace NUMINAMATH_GPT_no_such_function_exists_l982_98253

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (f x) = x^2 - 1996 :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l982_98253


namespace NUMINAMATH_GPT_ff_two_eq_three_l982_98285

noncomputable def f (x : ℝ) : ℝ :=
  if x < 6 then x^3 else Real.log x / Real.log x

theorem ff_two_eq_three : f (f 2) = 3 := by
  sorry

end NUMINAMATH_GPT_ff_two_eq_three_l982_98285


namespace NUMINAMATH_GPT_yellow_chip_value_l982_98213

theorem yellow_chip_value
  (y b g : ℕ)
  (hb : b = g)
  (hchips : y^4 * (4 * b)^b * (5 * g)^g = 16000)
  (h4yellow : y = 2) :
  y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_yellow_chip_value_l982_98213


namespace NUMINAMATH_GPT_shortest_distance_between_circles_zero_l982_98268

noncomputable def center_radius_circle1 : (ℝ × ℝ) × ℝ :=
  let c1 := (3, -5)
  let r1 := Real.sqrt 20
  (c1, r1)

noncomputable def center_radius_circle2 : (ℝ × ℝ) × ℝ :=
  let c2 := (-4, 1)
  let r2 := Real.sqrt 1
  (c2, r2)

theorem shortest_distance_between_circles_zero :
  let c1 := center_radius_circle1.1
  let r1 := center_radius_circle1.2
  let c2 := center_radius_circle2.1
  let r2 := center_radius_circle2.2
  let dist := Real.sqrt ((c1.1 - c2.1)^2 + (c1.2 - c2.2)^2)
  dist < r1 + r2 → 0 = 0 :=
by
  intros
  -- Add appropriate steps for the proof (skipping by using sorry for now)
  sorry

end NUMINAMATH_GPT_shortest_distance_between_circles_zero_l982_98268


namespace NUMINAMATH_GPT_shortTreesPlanted_l982_98236

-- Definitions based on conditions
def currentShortTrees : ℕ := 31
def tallTrees : ℕ := 32
def futureShortTrees : ℕ := 95

-- The proposition to be proved
theorem shortTreesPlanted :
  futureShortTrees - currentShortTrees = 64 :=
by
  sorry

end NUMINAMATH_GPT_shortTreesPlanted_l982_98236


namespace NUMINAMATH_GPT_LTE_divisibility_l982_98208

theorem LTE_divisibility (m : ℕ) (h_pos : 0 < m) :
  (∀ k : ℕ, k % 2 = 1 ∧ k ≥ 3 → 2^m ∣ k^m - 1) ↔ m = 1 ∨ m = 2 ∨ m = 4 :=
by
  sorry

end NUMINAMATH_GPT_LTE_divisibility_l982_98208


namespace NUMINAMATH_GPT_find_m_l982_98249

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem find_m (y b : ℝ) (m : ℕ) 
  (h5 : binomial m 4 * y^(m-4) * b^4 = 210) 
  (h6 : binomial m 5 * y^(m-5) * b^5 = 462) 
  (h7 : binomial m 6 * y^(m-6) * b^6 = 792) : 
  m = 7 := 
sorry

end NUMINAMATH_GPT_find_m_l982_98249


namespace NUMINAMATH_GPT_calculate_tough_week_sales_l982_98260

-- Define the conditions
variables (G T : ℝ)
def condition1 := T = G / 2
def condition2 := 5 * G + 3 * T = 10400

-- By substituting and proving
theorem calculate_tough_week_sales (G T : ℝ) (h1 : condition1 G T) (h2 : condition2 G T) : T = 800 := 
by {
  sorry 
}

end NUMINAMATH_GPT_calculate_tough_week_sales_l982_98260


namespace NUMINAMATH_GPT_sum_of_samples_is_six_l982_98202

-- Defining the conditions
def grains_varieties : ℕ := 40
def vegetable_oil_varieties : ℕ := 10
def animal_products_varieties : ℕ := 30
def fruits_and_vegetables_varieties : ℕ := 20
def sample_size : ℕ := 20
def total_varieties : ℕ := grains_varieties + vegetable_oil_varieties + animal_products_varieties + fruits_and_vegetables_varieties

def proportion_sample := (sample_size : ℚ) / total_varieties

-- Definitions for the problem
def vegetable_oil_sampled := (vegetable_oil_varieties : ℚ) * proportion_sample
def fruits_and_vegetables_sampled := (fruits_and_vegetables_varieties : ℚ) * proportion_sample

-- Lean 4 statement for the proof problem
theorem sum_of_samples_is_six :
  vegetable_oil_sampled + fruits_and_vegetables_sampled = 6 := by
  sorry

end NUMINAMATH_GPT_sum_of_samples_is_six_l982_98202


namespace NUMINAMATH_GPT_floor_alpha_six_eq_three_l982_98258

noncomputable def floor_of_alpha_six (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : ℤ :=
  Int.floor (α^6)

theorem floor_alpha_six_eq_three (α : ℝ) (h : α^5 - α^3 + α - 2 = 0) : floor_of_alpha_six α h = 3 :=
sorry

end NUMINAMATH_GPT_floor_alpha_six_eq_three_l982_98258


namespace NUMINAMATH_GPT_find_number_l982_98287

theorem find_number (x : ℝ) (h : 45 * 7 = 0.35 * x) : x = 900 :=
by
  -- Proof (skipped with sorry)
  sorry

end NUMINAMATH_GPT_find_number_l982_98287


namespace NUMINAMATH_GPT_cubic_roots_l982_98219

variable (p q : ℝ)

noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 3)

theorem cubic_roots (y z : ℂ) (h1 : -3 * y * z = p) (h2 : y^3 + z^3 = q) :
  ∃ (x1 x2 x3 : ℂ),
    (x^3 + p * x + q = 0) ∧
    (x1 = -(y + z)) ∧
    (x2 = -(ω * y + ω^2 * z)) ∧
    (x3 = -(ω^2 * y + ω * z)) :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_l982_98219


namespace NUMINAMATH_GPT_initial_video_files_l982_98265

theorem initial_video_files (V : ℕ) (h1 : 26 + V - 48 = 14) : V = 36 := 
by
  sorry

end NUMINAMATH_GPT_initial_video_files_l982_98265


namespace NUMINAMATH_GPT_judy_shopping_total_l982_98254

noncomputable def carrot_price := 1
noncomputable def milk_price := 3
noncomputable def pineapple_price := 4 / 2 -- half price
noncomputable def flour_price := 5
noncomputable def ice_cream_price := 7

noncomputable def carrot_quantity := 5
noncomputable def milk_quantity := 3
noncomputable def pineapple_quantity := 2
noncomputable def flour_quantity := 2
noncomputable def ice_cream_quantity := 1

noncomputable def initial_cost : ℝ := 
  carrot_quantity * carrot_price 
  + milk_quantity * milk_price 
  + pineapple_quantity * pineapple_price 
  + flour_quantity * flour_price 
  + ice_cream_quantity * ice_cream_price

noncomputable def final_cost (initial_cost: ℝ) := if initial_cost ≥ 25 then initial_cost - 5 else initial_cost

theorem judy_shopping_total : final_cost initial_cost = 30 := by
  sorry

end NUMINAMATH_GPT_judy_shopping_total_l982_98254


namespace NUMINAMATH_GPT_least_value_expr_l982_98203

   variable {x y : ℝ}

   theorem least_value_expr : ∃ x y : ℝ, (x^3 * y - 1)^2 + (x + y)^2 = 1 :=
   by
     sorry
   
end NUMINAMATH_GPT_least_value_expr_l982_98203


namespace NUMINAMATH_GPT_fraction_power_rule_l982_98283

theorem fraction_power_rule :
  (5 / 6) ^ 4 = (625 : ℚ) / 1296 := 
by sorry

end NUMINAMATH_GPT_fraction_power_rule_l982_98283


namespace NUMINAMATH_GPT_find_value_l982_98298

def star (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 2010

variable (a b : ℝ)

axiom h1 : 3 * a + 5 * b = 1
axiom h2 : 4 * a + 9 * b = -1

theorem find_value : star a b 1 2 = 2010 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_l982_98298


namespace NUMINAMATH_GPT_person_age_l982_98250

theorem person_age (x : ℕ) (h : 4 * (x + 3) - 4 * (x - 3) = x) : x = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_person_age_l982_98250


namespace NUMINAMATH_GPT_current_speed_l982_98262

theorem current_speed (c : ℝ) :
  (∀ d1 t1 u v, d1 = 20 ∧ t1 = 2 ∧ u = 6 ∧ v = c → d1 = t1 * (u + v))
  ∧ (∀ d2 t2 u w, d2 = 4 ∧ t2 = 2 ∧ u = 6 ∧ w = c → d2 = t2 * (u - w)) 
  → c = 4 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_current_speed_l982_98262


namespace NUMINAMATH_GPT_train_length_l982_98274

theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (h_speed : speed_kmh = 30) (h_time : time_sec = 6) :
  ∃ length_meters : ℝ, abs (length_meters - 50) < 1 :=
by
  -- Converting speed from km/hr to m/s
  let speed_ms := speed_kmh * (1000 / 3600)
  
  -- Calculating length of the train using the distance formula
  let length_meters := speed_ms * time_sec

  use length_meters
  -- Proof would go here showing abs (length_meters - 50) < 1
  sorry

end NUMINAMATH_GPT_train_length_l982_98274


namespace NUMINAMATH_GPT_sophie_total_spend_l982_98216

def total_cost_with_discount_and_tax : ℝ :=
  let cupcakes_price := 5 * 2
  let doughnuts_price := 6 * 1
  let apple_pie_price := 4 * 2
  let cookies_price := 15 * 0.60
  let chocolate_bars_price := 8 * 1.50
  let soda_price := 12 * 1.20
  let gum_price := 3 * 0.80
  let chips_price := 10 * 1.10
  let total_before_discount := cupcakes_price + doughnuts_price + apple_pie_price + cookies_price + chocolate_bars_price + soda_price + gum_price + chips_price
  let discount := 0.10 * total_before_discount
  let subtotal_after_discount := total_before_discount - discount
  let sales_tax := 0.06 * subtotal_after_discount
  let total_cost := subtotal_after_discount + sales_tax
  total_cost

theorem sophie_total_spend :
  total_cost_with_discount_and_tax = 69.45 :=
sorry

end NUMINAMATH_GPT_sophie_total_spend_l982_98216


namespace NUMINAMATH_GPT_sum_of_digits_l982_98295

theorem sum_of_digits (a b : ℕ) (h1 : 10 * a + b + 10 * b + a = 202) (h2 : a < 10) (h3 : b < 10) :
  a + b = 12 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l982_98295
