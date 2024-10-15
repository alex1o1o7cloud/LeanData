import Mathlib

namespace NUMINAMATH_GPT_sphere_surface_area_diameter_4_l2372_237216

noncomputable def sphere_surface_area (d : ℝ) : ℝ :=
  4 * Real.pi * (d / 2) ^ 2

theorem sphere_surface_area_diameter_4 :
  sphere_surface_area 4 = 16 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_surface_area_diameter_4_l2372_237216


namespace NUMINAMATH_GPT_inequality_solution_l2372_237228

theorem inequality_solution (x : ℝ) (h : |(x + 4) / 2| < 3) : -10 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l2372_237228


namespace NUMINAMATH_GPT_find_r_floor_r_add_r_eq_18point2_l2372_237267

theorem find_r_floor_r_add_r_eq_18point2 (r : ℝ) (h : ⌊r⌋ + r = 18.2) : r = 9.2 := 
sorry

end NUMINAMATH_GPT_find_r_floor_r_add_r_eq_18point2_l2372_237267


namespace NUMINAMATH_GPT_point_on_ellipse_l2372_237262

noncomputable def ellipse_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  let x := P.1
  let y := P.2
  let d1 := ((x - F1.1)^2 + (y - F1.2)^2).sqrt
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  x^2 + 4 * y^2 = 16 ∧ d1 = 7

theorem point_on_ellipse (P F1 F2 : ℝ × ℝ)
  (h : ellipse_condition P F1 F2) : 
  let x := P.1
  let y := P.2
  let d2 := ((x - F2.1)^2 + (y - F2.2)^2).sqrt
  d2 = 1 :=
sorry

end NUMINAMATH_GPT_point_on_ellipse_l2372_237262


namespace NUMINAMATH_GPT_value_of_f_l2372_237256

noncomputable
def f (k l m x : ℝ) : ℝ := k + m / (x - l)

theorem value_of_f (k l m : ℝ) (hk : k = -2) (hl : l = 2.5) (hm : m = 12) :
  f k l m (k + l + m) = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_l2372_237256


namespace NUMINAMATH_GPT_min_value_fraction_l2372_237249

variable {a b : ℝ}

theorem min_value_fraction (h₁ : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (1 / a + 4 / b) ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l2372_237249


namespace NUMINAMATH_GPT_density_of_cone_in_mercury_l2372_237292

variable {h : ℝ} -- height of the cone
variable {ρ : ℝ} -- density of the cone
variable {ρ_m : ℝ} -- density of the mercury
variable {k : ℝ} -- proportion factor

-- Archimedes' principle applied to the cone floating in mercury
theorem density_of_cone_in_mercury (stable_eq: ∀ (V V_sub: ℝ), (ρ * V) = (ρ_m * V_sub))
(h_sub: h / k = (k - 1) / k) :
  ρ = ρ_m * ((k - 1)^3 / k^3) :=
by
  sorry

end NUMINAMATH_GPT_density_of_cone_in_mercury_l2372_237292


namespace NUMINAMATH_GPT_greatest_product_sum_300_l2372_237208

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end NUMINAMATH_GPT_greatest_product_sum_300_l2372_237208


namespace NUMINAMATH_GPT_largest_prime_y_in_triangle_l2372_237209

-- Define that a number is prime
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem largest_prime_y_in_triangle : 
  ∃ (x y z : ℕ), is_prime x ∧ is_prime y ∧ is_prime z ∧ x + y + z = 90 ∧ y < x ∧ y > z ∧ y = 47 :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_y_in_triangle_l2372_237209


namespace NUMINAMATH_GPT_sum_of_nine_consecutive_quotients_multiple_of_9_l2372_237286

def a (i : ℕ) : ℕ := (10^(2 * i) - 1) / 9
def q (i : ℕ) : ℕ := a i / 11
def s (i : ℕ) : ℕ := q i + q (i + 1) + q (i + 2) + q (i + 3) + q (i + 4) + q (i + 5) + q (i + 6) + q (i + 7) + q (i + 8)

theorem sum_of_nine_consecutive_quotients_multiple_of_9 (i n : ℕ) (h : n > 8) 
  (h2 : i ≤ n - 8) : s i % 9 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_nine_consecutive_quotients_multiple_of_9_l2372_237286


namespace NUMINAMATH_GPT_gumball_machine_l2372_237290

variable (R B G Y O : ℕ)

theorem gumball_machine : 
  (B = (1 / 2) * R) ∧
  (G = 4 * B) ∧
  (Y = (7 / 2) * B) ∧
  (O = (2 / 3) * (R + B)) ∧
  (R = (3 / 2) * Y) ∧
  (Y = 24) →
  (R + B + G + Y + O = 186) :=
sorry

end NUMINAMATH_GPT_gumball_machine_l2372_237290


namespace NUMINAMATH_GPT_jake_peaches_is_7_l2372_237200

variable (Steven_peaches Jake_peaches Jill_peaches : ℕ)

-- Conditions:
def Steven_has_19_peaches : Steven_peaches = 19 := by sorry

def Jake_has_12_fewer_peaches_than_Steven : Jake_peaches = Steven_peaches - 12 := by sorry

def Jake_has_72_more_peaches_than_Jill : Jake_peaches = Jill_peaches + 72 := by sorry

-- Proof problem:
theorem jake_peaches_is_7 
    (Steven_peaches Jake_peaches Jill_peaches : ℕ)
    (h1 : Steven_peaches = 19)
    (h2 : Jake_peaches = Steven_peaches - 12)
    (h3 : Jake_peaches = Jill_peaches + 72) :
    Jake_peaches = 7 := by sorry

end NUMINAMATH_GPT_jake_peaches_is_7_l2372_237200


namespace NUMINAMATH_GPT_determine_xyz_l2372_237245

theorem determine_xyz (x y z : ℝ) 
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 12) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) : 
  x * y * z = -4 / 3 := 
sorry

end NUMINAMATH_GPT_determine_xyz_l2372_237245


namespace NUMINAMATH_GPT_watch_correction_l2372_237223

noncomputable def correction_time (loss_per_day : ℕ) (start_date : ℕ) (end_date : ℕ) (spring_forward_hour : ℕ) (correction_time_hour : ℕ) : ℝ :=
  let n_days := end_date - start_date
  let total_hours_watch := n_days * 24 + correction_time_hour - spring_forward_hour
  let loss_rate_per_hour := (loss_per_day : ℝ) / 24
  let total_loss := loss_rate_per_hour * total_hours_watch
  total_loss

theorem watch_correction :
  correction_time 3 1 5 1 6 = 6.625 :=
by
  sorry

end NUMINAMATH_GPT_watch_correction_l2372_237223


namespace NUMINAMATH_GPT_max_a2b3c4_l2372_237298

noncomputable def maximum_value (a b c : ℝ) : ℝ := a^2 * b^3 * c^4

theorem max_a2b3c4 (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) :
  maximum_value a b c ≤ 143327232 / 386989855 := sorry

end NUMINAMATH_GPT_max_a2b3c4_l2372_237298


namespace NUMINAMATH_GPT_quadratic_sum_of_squares_l2372_237238

theorem quadratic_sum_of_squares (α β : ℝ) (h1 : α * β = 3) (h2 : α + β = 7) : α^2 + β^2 = 43 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_sum_of_squares_l2372_237238


namespace NUMINAMATH_GPT_flowers_remaining_along_path_after_events_l2372_237251

def total_flowers : ℕ := 30
def total_peonies : ℕ := 15
def total_tulips : ℕ := 15
def unwatered_flowers : ℕ := 10
def tulips_watered_by_sineglazka : ℕ := 10
def tulips_picked_by_neznaika : ℕ := 6
def remaining_flowers : ℕ := 19

theorem flowers_remaining_along_path_after_events :
  total_peonies + total_tulips = total_flowers →
  tulips_watered_by_sineglazka + unwatered_flowers = total_flowers →
  tulips_picked_by_neznaika ≤ total_tulips →
  remaining_flowers = 19 := sorry

end NUMINAMATH_GPT_flowers_remaining_along_path_after_events_l2372_237251


namespace NUMINAMATH_GPT_gcd_288_123_l2372_237289

theorem gcd_288_123 : gcd 288 123 = 3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_288_123_l2372_237289


namespace NUMINAMATH_GPT_fraction_order_l2372_237246

theorem fraction_order :
  (25 / 19 : ℚ) < (21 / 16 : ℚ) ∧ (21 / 16 : ℚ) < (23 / 17 : ℚ) := by
  sorry

end NUMINAMATH_GPT_fraction_order_l2372_237246


namespace NUMINAMATH_GPT_quadratic_y_at_x_5_l2372_237281

-- Define the quadratic function
noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions and question as part of a theorem
theorem quadratic_y_at_x_5 (a b c : ℝ) 
  (h1 : ∀ x, quadratic a b c x ≤ 10) -- Maximum value condition (The maximum value is 10)
  (h2 : (quadratic a b c (-2)) = 10) -- y = 10 when x = -2 (maximum point)
  (h3 : quadratic a b c 0 = -8) -- The first point (0, -8)
  (h4 : quadratic a b c 1 = 0) -- The second point (1, 0)
  : quadratic a b c 5 = -400 / 9 :=
sorry

end NUMINAMATH_GPT_quadratic_y_at_x_5_l2372_237281


namespace NUMINAMATH_GPT_ratio_of_perimeters_l2372_237225

theorem ratio_of_perimeters (s S : ℝ) 
  (h1 : S = 3 * s) : 
  (4 * S) / (4 * s) = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_l2372_237225


namespace NUMINAMATH_GPT_sum_x_coordinates_eq_3_l2372_237240

def f : ℝ → ℝ := sorry -- definition of the function f as given by the five line segments

theorem sum_x_coordinates_eq_3 :
  (∃ x1 x2 x3 : ℝ, (f x1 = x1 + 1 ∧ f x2 = x2 + 1 ∧ f x3 = x3 + 1) ∧ (x1 + x2 + x3 = 3)) :=
sorry

end NUMINAMATH_GPT_sum_x_coordinates_eq_3_l2372_237240


namespace NUMINAMATH_GPT_batsman_average_after_12th_innings_l2372_237296

-- Defining the conditions
def before_12th_innings_average (A : ℕ) : Prop :=
11 * A + 80 = 12 * (A + 2)

-- Defining the question and expected answer
def after_12th_innings_average : ℕ := 58

-- Proving the equivalence
theorem batsman_average_after_12th_innings (A : ℕ) (h : before_12th_innings_average A) : after_12th_innings_average = 58 :=
by
sorry

end NUMINAMATH_GPT_batsman_average_after_12th_innings_l2372_237296


namespace NUMINAMATH_GPT_track_meet_girls_short_hair_l2372_237202

theorem track_meet_girls_short_hair :
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  girls_short_hair = 10 :=
by
  let total_people := 55
  let boys := 30
  let girls := total_people - boys
  let girls_long_hair := (3 / 5 : ℚ) * girls
  let girls_short_hair := girls - girls_long_hair
  sorry

end NUMINAMATH_GPT_track_meet_girls_short_hair_l2372_237202


namespace NUMINAMATH_GPT_usual_time_to_office_l2372_237243

theorem usual_time_to_office (S T : ℝ) (h : T = 4 / 3 * (T + 8)) : T = 24 :=
by
  sorry

end NUMINAMATH_GPT_usual_time_to_office_l2372_237243


namespace NUMINAMATH_GPT_cost_of_seven_books_l2372_237275

theorem cost_of_seven_books (h : 3 * 12 = 36) : 7 * 12 = 84 :=
sorry

end NUMINAMATH_GPT_cost_of_seven_books_l2372_237275


namespace NUMINAMATH_GPT_total_yield_UncleLi_yield_difference_l2372_237272

-- Define the conditions related to Uncle Li and Aunt Lin
def UncleLiAcres : ℕ := 12
def UncleLiYieldPerAcre : ℕ := 660
def AuntLinAcres : ℕ := UncleLiAcres - 2
def AuntLinTotalYield : ℕ := UncleLiYieldPerAcre * UncleLiAcres - 420

-- Prove the total yield of Uncle Li's rice
theorem total_yield_UncleLi : UncleLiYieldPerAcre * UncleLiAcres = 7920 := by
  sorry

-- Prove how much less the yield per acre of Uncle Li's rice is compared to Aunt Lin's
theorem yield_difference :
  UncleLiYieldPerAcre - AuntLinTotalYield / AuntLinAcres = 90 := by
  sorry

end NUMINAMATH_GPT_total_yield_UncleLi_yield_difference_l2372_237272


namespace NUMINAMATH_GPT_negation_of_implication_l2372_237221

theorem negation_of_implication (x : ℝ) :
  ¬ (x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_implication_l2372_237221


namespace NUMINAMATH_GPT_initial_number_of_red_balls_l2372_237237

theorem initial_number_of_red_balls 
  (num_white_balls num_red_balls : ℕ)
  (h1 : num_red_balls = 4 * num_white_balls + 3)
  (num_actions : ℕ)
  (h2 : 4 + 5 * num_actions = num_white_balls)
  (h3 : 34 + 17 * num_actions = num_red_balls) : 
  num_red_balls = 119 := 
by
  sorry

end NUMINAMATH_GPT_initial_number_of_red_balls_l2372_237237


namespace NUMINAMATH_GPT_binomial_sum_l2372_237207

theorem binomial_sum :
  (Nat.choose 10 3) + (Nat.choose 10 4) = 330 :=
by
  sorry

end NUMINAMATH_GPT_binomial_sum_l2372_237207


namespace NUMINAMATH_GPT_trigonometric_expression_value_l2372_237222

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  2 * (Real.sin α)^2 + 4 * Real.sin α * Real.cos α - 9 * (Real.cos α)^2 = 21 / 10 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l2372_237222


namespace NUMINAMATH_GPT_odd_c_perfect_square_no_even_c_infinitely_many_solutions_l2372_237280

open Nat

/-- Problem (1): prove that if c is an odd number, then c is a perfect square given 
    c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem odd_c_perfect_square (a b c : ℕ) (h_eq : c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) (h_odd : Odd c) : ∃ k : ℕ, c = k^2 :=
  sorry

/-- Problem (2): prove that there does not exist an even number c that satisfies 
    c(a c + 1)^2 = (5c + 2b)(2c + b) for some a and b -/
theorem no_even_c (a b : ℕ) : ∀ c : ℕ, Even c → ¬ (c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b)) :=
  sorry

/-- Problem (3): prove that there are infinitely many solutions of positive integers 
    (a, b, c) that satisfy c(a c + 1)^2 = (5c + 2b)(2c + b) -/
theorem infinitely_many_solutions (n : ℕ) : ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  c * (a * c + 1)^2 = (5 * c + 2 * b) * (2 * c + b) :=
  sorry

end NUMINAMATH_GPT_odd_c_perfect_square_no_even_c_infinitely_many_solutions_l2372_237280


namespace NUMINAMATH_GPT_anthony_pencils_l2372_237201

theorem anthony_pencils (P : Nat) (h : P + 56 = 65) : P = 9 :=
by
  sorry

end NUMINAMATH_GPT_anthony_pencils_l2372_237201


namespace NUMINAMATH_GPT_num_of_factorizable_poly_l2372_237204

theorem num_of_factorizable_poly : 
  ∃ (n : ℕ), (1 ≤ n ∧ n ≤ 2023) ∧ 
              (∃ (a : ℤ), n = a * (a + 1)) :=
sorry

end NUMINAMATH_GPT_num_of_factorizable_poly_l2372_237204


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2372_237299

-- 1. Given: ∃ x ∈ ℤ, x^2 - 2x - 3 = 0
--    Show: ∀ x ∈ ℤ, x^2 - 2x - 3 ≠ 0
theorem problem1 : (∃ x : ℤ, x^2 - 2 * x - 3 = 0) ↔ (∀ x : ℤ, x^2 - 2 * x - 3 ≠ 0) := sorry

-- 2. Given: ∀ x ∈ ℝ, x^2 + 3 ≥ 2x
--    Show: ∃ x ∈ ℝ, x^2 + 3 < 2x
theorem problem2 : (∀ x : ℝ, x^2 + 3 ≥ 2 * x) ↔ (∃ x : ℝ, x^2 + 3 < 2 * x) := sorry

-- 3. Given: If x > 1 and y > 1, then x + y > 2
--    Show: If x ≤ 1 or y ≤ 1, then x + y ≤ 2
theorem problem3 : (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ↔ (∀ x y : ℝ, x ≤ 1 ∨ y ≤ 1 → x + y ≤ 2) := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2372_237299


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_holds_l2372_237235

-- Let m be a real number
variable (m : ℝ)

-- Define the conditions
def condition_1 : Prop := (m + 3) * (2 * m + 1) < 0
def condition_2 : Prop := -(2 * m - 1) > m + 2
def condition_3 : Prop := m + 2 > 0

-- Define necessary but not sufficient condition
def necessary_but_not_sufficient : Prop :=
  -2 < m ∧ m < -1 / 3

-- Problem statement
theorem necessary_but_not_sufficient_condition_holds 
  (h1 : condition_1 m) 
  (h2 : condition_2 m) 
  (h3 : condition_3 m) : necessary_but_not_sufficient m :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_holds_l2372_237235


namespace NUMINAMATH_GPT_solve_equation_l2372_237258

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 → (x = 0 ∨ x = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l2372_237258


namespace NUMINAMATH_GPT_general_term_of_sequence_l2372_237248

theorem general_term_of_sequence 
  (a : ℕ → ℝ)
  (log_a : ℕ → ℝ)
  (h1 : ∀ n, log_a n = Real.log (a n)) 
  (h2 : ∃ d, ∀ n, log_a (n + 1) - log_a n = d)
  (h3 : d = Real.log 3)
  (h4 : log_a 0 + log_a 1 + log_a 2 = 6 * Real.log 3) : 
  ∀ n, a n = 3 ^ n :=
by
  sorry

end NUMINAMATH_GPT_general_term_of_sequence_l2372_237248


namespace NUMINAMATH_GPT_line_equation_l2372_237287

theorem line_equation
  (x y : ℝ)
  (h1 : 2 * x + y + 2 = 0)
  (h2 : 2 * x - y + 2 = 0)
  (h3 : ∀ x y, x + y = 0 → x - 1 = y): 
  x - y + 1 = 0 :=
sorry

end NUMINAMATH_GPT_line_equation_l2372_237287


namespace NUMINAMATH_GPT_point_below_line_range_l2372_237231

theorem point_below_line_range (t : ℝ) : (2 * (-2) - 3 * t + 6 > 0) → t < (2 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_point_below_line_range_l2372_237231


namespace NUMINAMATH_GPT_sparrows_among_non_robins_percentage_l2372_237269

-- Define percentages of different birds
def finches_percentage : ℝ := 0.40
def sparrows_percentage : ℝ := 0.20
def owls_percentage : ℝ := 0.15
def robins_percentage : ℝ := 0.25

-- Define the statement to prove 
theorem sparrows_among_non_robins_percentage :
  ((sparrows_percentage / (1 - robins_percentage)) * 100) = 26.67 := by
  -- This is where the proof would go, but it's omitted as per instructions
  sorry

end NUMINAMATH_GPT_sparrows_among_non_robins_percentage_l2372_237269


namespace NUMINAMATH_GPT_tangent_lines_l2372_237268

noncomputable def curve1 (x : ℝ) : ℝ := 2 * x ^ 2 - 5
noncomputable def curve2 (x : ℝ) : ℝ := x ^ 2 - 3 * x + 5

theorem tangent_lines :
  (∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, y = -20 * x - 55 ∨ y = -13 * x - 20 ∨ y = 8 * x - 13 ∨ y = x + 1) ∧ 
    (
      (m₁ = 4 * 2 ∧ b₁ = 3) ∨ 
      (m₁ = 2 * -5 - 3 ∧ b₁ = 45) ∨
      (m₂ = 4 * -5 ∧ b₂ = 45) ∨
      (m₂ = 2 * 2 - 3 ∧ b₂ = 3)
    )) :=
sorry

end NUMINAMATH_GPT_tangent_lines_l2372_237268


namespace NUMINAMATH_GPT_find_l_l2372_237253

variables (a b c l : ℤ)
def g (x : ℤ) : ℤ := a * x^2 + b * x + c

theorem find_l :
  g a b c 2 = 0 →
  60 < g a b c 6 ∧ g a b c 6 < 70 →
  80 < g a b c 9 ∧ g a b c 9 < 90 →
  6000 * l < g a b c 100 ∧ g a b c 100 < 6000 * (l + 1) →
  l = 5 :=
sorry

end NUMINAMATH_GPT_find_l_l2372_237253


namespace NUMINAMATH_GPT_caffeine_per_energy_drink_l2372_237279

variable (amount_of_caffeine_per_drink : ℕ)

def maximum_safe_caffeine_per_day := 500
def drinks_per_day := 4
def additional_safe_amount := 20

theorem caffeine_per_energy_drink :
  4 * amount_of_caffeine_per_drink + additional_safe_amount = maximum_safe_caffeine_per_day →
  amount_of_caffeine_per_drink = 120 :=
by
  sorry

end NUMINAMATH_GPT_caffeine_per_energy_drink_l2372_237279


namespace NUMINAMATH_GPT_angle_B_in_right_triangle_in_degrees_l2372_237274

def angleSum (A B C: ℝ) : Prop := A + B + C = 180

theorem angle_B_in_right_triangle_in_degrees (A B C : ℝ) (h1 : C = 90) (h2 : A = 35.5) (h3 : angleSum A B C) : B = 54.5 := 
by
  sorry

end NUMINAMATH_GPT_angle_B_in_right_triangle_in_degrees_l2372_237274


namespace NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l2372_237282

-- Define the capacity of the can, the amount of milk added, and the ratio when full.
def capacity : ℕ := 72
def additionalMilk : ℕ := 8
def fullRatioNumerator : ℕ := 2
def fullRatioDenominator : ℕ := 1

-- Define the initial amounts of milk and water in the can.
variables (M W : ℕ)

-- Define the conditions given in the problem.
def conditions : Prop :=
  M + W + additionalMilk = capacity ∧
  (M + additionalMilk) * fullRatioDenominator = fullRatioNumerator * W

-- Define the expected result, the initial ratio of milk to water in the can.
def expected_ratio : ℕ × ℕ :=
  (5, 3)

-- The theorem to prove the initial ratio of milk to water given the conditions.
theorem initial_ratio_of_milk_to_water (M W : ℕ) (h : conditions M W) :
  (M / Nat.gcd M W, W / Nat.gcd M W) = expected_ratio :=
sorry

end NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l2372_237282


namespace NUMINAMATH_GPT_part_i_part_ii_l2372_237255

open Real -- Open the Real number space

-- (i) Prove that for any real number x, there exist two points of the same color that are at a distance of x from each other
theorem part_i (color : Real × Real → Bool) :
  ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = color p2 ∧ dist p1 p2 = x :=
by
  sorry

-- (ii) Prove that there exists a color such that for every real number x, 
-- we can find two points of that color that are at a distance of x from each other
theorem part_ii (color : Real × Real → Bool) :
  ∃ c : Bool, ∀ x : ℝ, ∃ p1 p2 : Real × Real, color p1 = c ∧ color p2 = c ∧ dist p1 p2 = x :=
by
  sorry

end NUMINAMATH_GPT_part_i_part_ii_l2372_237255


namespace NUMINAMATH_GPT_selling_price_eq_100_l2372_237212

variable (CP SP : ℝ)

-- Conditions
def gain : ℝ := 20
def gain_percentage : ℝ := 0.25

-- The proof of the selling price
theorem selling_price_eq_100
  (h1 : gain = 20)
  (h2 : gain_percentage = 0.25)
  (h3 : gain = gain_percentage * CP)
  (h4 : SP = CP + gain) :
  SP = 100 := sorry

end NUMINAMATH_GPT_selling_price_eq_100_l2372_237212


namespace NUMINAMATH_GPT_smallest_positive_n_l2372_237226

theorem smallest_positive_n : ∃ n : ℕ, 3 * n ≡ 8 [MOD 26] ∧ n = 20 :=
by 
  use 20
  simp
  sorry

end NUMINAMATH_GPT_smallest_positive_n_l2372_237226


namespace NUMINAMATH_GPT_statement1_statement2_statement3_l2372_237276

variable (P_W P_Z : ℝ)

/-- The conditions of the problem: -/
def conditions : Prop :=
  P_W = 0.4 ∧ P_Z = 0.2

/-- Proof of the first statement -/
theorem statement1 (h : conditions P_W P_Z) : 
  P_W * P_Z = 0.08 := 
by sorry

/-- Proof of the second statement -/
theorem statement2 (h : conditions P_W P_Z) :
  P_W * (1 - P_Z) + (1 - P_W) * P_Z = 0.44 := 
by sorry

/-- Proof of the third statement -/
theorem statement3 (h : conditions P_W P_Z) :
  1 - P_W * P_Z = 0.92 := 
by sorry

end NUMINAMATH_GPT_statement1_statement2_statement3_l2372_237276


namespace NUMINAMATH_GPT_opposite_number_of_sqrt_of_9_is_neg3_l2372_237206

theorem opposite_number_of_sqrt_of_9_is_neg3 :
  - (Real.sqrt 9) = -3 :=
by
  -- The proof is omitted as required.
  sorry

end NUMINAMATH_GPT_opposite_number_of_sqrt_of_9_is_neg3_l2372_237206


namespace NUMINAMATH_GPT_value_of_card_l2372_237283

/-- For this problem: 
    1. Matt has 8 baseball cards worth $6 each.
    2. He trades two of them to Jane in exchange for 3 $2 cards and a card of certain value.
    3. He makes a profit of $3.
    We need to prove that the value of the card that Jane gave to Matt apart from the $2 cards is $9. -/
theorem value_of_card (value_per_card traded_cards received_dollar_cards profit received_total_value : ℤ)
  (h1 : value_per_card = 6)
  (h2 : traded_cards = 2)
  (h3 : received_dollar_cards = 6)
  (h4 : profit = 3)
  (h5 : received_total_value = 15) :
  received_total_value - received_dollar_cards = 9 :=
by {
  -- This is just left as a placeholder to signal that the proof needs to be provided.
  sorry
}

end NUMINAMATH_GPT_value_of_card_l2372_237283


namespace NUMINAMATH_GPT_proof_of_problem_l2372_237297

noncomputable def problem_statement : Prop :=
  ∃ (x y z m : ℝ), (x > 0 ∧ y > 0 ∧ z > 0 ∧ x^3 * y^2 * z = 1 ∧ m = x + 2*y + 3*z ∧ m^3 = 72)

theorem proof_of_problem : problem_statement :=
sorry

end NUMINAMATH_GPT_proof_of_problem_l2372_237297


namespace NUMINAMATH_GPT_problem_f_val_l2372_237291

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f_val (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (-x) = -f x)
  (h2 : ∀ x : ℝ, f (1 + x) = f (1 - x))
  (h3 : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f x = x^3) :
  f 2015 = -1 :=
  sorry

end NUMINAMATH_GPT_problem_f_val_l2372_237291


namespace NUMINAMATH_GPT_manager_salary_l2372_237210

theorem manager_salary (avg_salary_employees : ℝ) (num_employees : ℕ) (salary_increase : ℝ) (manager_salary : ℝ) :
  avg_salary_employees = 1500 →
  num_employees = 24 →
  salary_increase = 400 →
  (num_employees + 1) * (avg_salary_employees + salary_increase) - num_employees * avg_salary_employees = manager_salary →
  manager_salary = 11500 := 
by
  intros h_avg_salary_employees h_num_employees h_salary_increase h_computation
  sorry

end NUMINAMATH_GPT_manager_salary_l2372_237210


namespace NUMINAMATH_GPT_smallest_product_of_digits_l2372_237285

theorem smallest_product_of_digits : 
  ∃ (a b c d : ℕ), 
  (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6) ∧ 
  (∃ x y : ℕ, (x = a * 10 + c ∧ y = b * 10 + d) ∨ (x = a * 10 + d ∧ y = b * 10 + c) ∨ (x = b * 10 + c ∧ y = a * 10 + d) ∨ (x = b * 10 + d ∧ y = a * 10 + c)) ∧
  (∀ x1 y1 x2 y2 : ℕ, ((x1 = 34 ∧ y1 = 56 ∨ x1 = 35 ∧ y1 = 46) ∧ (x2 = 34 ∧ y2 = 56 ∨ x2 = 35 ∧ y2 = 46)) → x1 * y1 ≥ x2 * y2) ∧
  35 * 46 = 1610 :=
sorry

end NUMINAMATH_GPT_smallest_product_of_digits_l2372_237285


namespace NUMINAMATH_GPT_power_sum_eq_nine_l2372_237247

theorem power_sum_eq_nine {m n p q : ℕ} (h : ∀ x > 0, (x + 1)^m / x^n - 1 = (x + 1)^p / x^q) :
  (m^2 + 2 * n + p)^(2 * q) = 9 :=
sorry

end NUMINAMATH_GPT_power_sum_eq_nine_l2372_237247


namespace NUMINAMATH_GPT_brett_blue_marbles_more_l2372_237288

theorem brett_blue_marbles_more (r b : ℕ) (hr : r = 6) (hb : b = 5 * r) : b - r = 24 := by
  rw [hr, hb]
  norm_num
  sorry

end NUMINAMATH_GPT_brett_blue_marbles_more_l2372_237288


namespace NUMINAMATH_GPT_range_f_l2372_237242

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x - Real.cos x

theorem range_f : Set.range f = Set.Icc (-2 : ℝ) 2 := 
by
  sorry

end NUMINAMATH_GPT_range_f_l2372_237242


namespace NUMINAMATH_GPT_increasing_or_decreasing_subseq_l2372_237227

theorem increasing_or_decreasing_subseq {m n : ℕ} (a : Fin (m * n + 1) → ℝ) :
  ∃ (idx_incr : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_incr i) < a (idx_incr j)) ∨ 
  ∃ (idx_decr : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (idx_decr i) > a (idx_decr j)) :=
by
  sorry

end NUMINAMATH_GPT_increasing_or_decreasing_subseq_l2372_237227


namespace NUMINAMATH_GPT_value_of_a_l2372_237284

noncomputable def function_f (x a : ℝ) : ℝ := (x - a) ^ 2 + (Real.log x ^ 2 - 2 * a) ^ 2

theorem value_of_a (x0 : ℝ) (a : ℝ) (h1 : x0 > 0) (h2 : function_f x0 a ≤ 4 / 5) : a = 1 / 5 :=
sorry

end NUMINAMATH_GPT_value_of_a_l2372_237284


namespace NUMINAMATH_GPT_Roshesmina_pennies_l2372_237293

theorem Roshesmina_pennies :
  (∀ compartments : ℕ, compartments = 12 → 
   (∀ initial_pennies : ℕ, initial_pennies = 2 → 
   (∀ additional_pennies : ℕ, additional_pennies = 6 → 
   (compartments * (initial_pennies + additional_pennies) = 96)))) :=
by
  sorry

end NUMINAMATH_GPT_Roshesmina_pennies_l2372_237293


namespace NUMINAMATH_GPT_slope_of_line_l2372_237205

noncomputable def line_eq (x y : ℝ) := x / 4 + y / 5 = 1

theorem slope_of_line : ∀ (x y : ℝ), line_eq x y → (∃ m b : ℝ, y = m * x + b ∧ m = -5 / 4) :=
sorry

end NUMINAMATH_GPT_slope_of_line_l2372_237205


namespace NUMINAMATH_GPT_gcd_10010_15015_l2372_237265

theorem gcd_10010_15015 :
  Int.gcd 10010 15015 = 5005 :=
by 
  sorry

end NUMINAMATH_GPT_gcd_10010_15015_l2372_237265


namespace NUMINAMATH_GPT_pure_imaginary_condition_l2372_237219

theorem pure_imaginary_condition (m : ℝ) (h : (m^2 - 3 * m) = 0) : (m = 0) :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_condition_l2372_237219


namespace NUMINAMATH_GPT_number_of_fish_initially_tagged_l2372_237224

theorem number_of_fish_initially_tagged {N T : ℕ}
  (hN : N = 1250)
  (h_ratio : 2 / 50 = T / N) :
  T = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_of_fish_initially_tagged_l2372_237224


namespace NUMINAMATH_GPT_evaluate_expression_l2372_237234

theorem evaluate_expression {x y : ℕ} (h₁ : 144 = 2^x * 3^y) (hx : x = 4) (hy : y = 2) : (1 / 7) ^ (y - x) = 49 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2372_237234


namespace NUMINAMATH_GPT_find_real_solutions_l2372_237294

theorem find_real_solutions : 
  ∀ x : ℝ, 1 / ((x - 2) * (x - 3)) 
         + 1 / ((x - 3) * (x - 4)) 
         + 1 / ((x - 4) * (x - 5)) 
         = 1 / 8 ↔ x = 7 ∨ x = -2 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_find_real_solutions_l2372_237294


namespace NUMINAMATH_GPT_money_left_after_expenses_l2372_237233

theorem money_left_after_expenses : 
  let salary := 150000.00000000003
  let food := salary * (1 / 5)
  let house_rent := salary * (1 / 10)
  let clothes := salary * (3 / 5)
  let total_spent := food + house_rent + clothes
  let money_left := salary - total_spent
  money_left = 15000.00000000000 :=
by
  sorry

end NUMINAMATH_GPT_money_left_after_expenses_l2372_237233


namespace NUMINAMATH_GPT_probability_of_two_queens_or_at_least_one_king_l2372_237278

def probability_two_queens_or_at_least_one_king : ℚ := 2 / 13

theorem probability_of_two_queens_or_at_least_one_king :
  let probability_two_queens := (4/52) * (3/51)
  let probability_exactly_one_king := (2 * (4/52) * (48/51))
  let probability_two_kings := (4/52) * (3/51)
  let probability_at_least_one_king := probability_exactly_one_king + probability_two_kings
  let total_probability := probability_two_queens + probability_at_least_one_king
  total_probability = probability_two_queens_or_at_least_one_king := 
by
  sorry

end NUMINAMATH_GPT_probability_of_two_queens_or_at_least_one_king_l2372_237278


namespace NUMINAMATH_GPT_geometric_seq_condition_l2372_237230

-- Defining a geometric sequence
def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Defining an increasing sequence
def is_increasing_seq (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

-- The condition to be proved
theorem geometric_seq_condition (a : ℕ → ℝ) (h_geo : is_geometric_seq a) :
  (a 0 < a 1 → is_increasing_seq a) ∧ (is_increasing_seq a → a 0 < a 1) :=
by 
  sorry

end NUMINAMATH_GPT_geometric_seq_condition_l2372_237230


namespace NUMINAMATH_GPT_number_of_zeros_l2372_237266

noncomputable def f (x : ℝ) : ℝ := |2^x - 1| - 3^x

theorem number_of_zeros : ∃! x : ℝ, f x = 0 := sorry

end NUMINAMATH_GPT_number_of_zeros_l2372_237266


namespace NUMINAMATH_GPT_minimum_distance_on_circle_l2372_237254

open Complex

noncomputable def minimum_distance (z : ℂ) : ℝ :=
  abs (z - (1 + 2*I))

theorem minimum_distance_on_circle :
  ∀ z : ℂ, abs (z + 2 - 2*I) = 1 → minimum_distance z = 2 :=
by
  intros z hz
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_minimum_distance_on_circle_l2372_237254


namespace NUMINAMATH_GPT_investment_ratio_l2372_237295

theorem investment_ratio (A_invest B_invest C_invest : ℝ) (F : ℝ) (total_profit B_share : ℝ)
  (h1 : A_invest = 3 * B_invest)
  (h2 : B_invest = F * C_invest)
  (h3 : total_profit = 7700)
  (h4 : B_share = 1400)
  (h5 : (B_invest / (A_invest + B_invest + C_invest)) * total_profit = B_share) :
  (B_invest / C_invest) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_investment_ratio_l2372_237295


namespace NUMINAMATH_GPT_apples_vs_cherries_l2372_237241

def pies_per_day : Nat := 12
def apple_days_per_week : Nat := 3
def cherry_days_per_week : Nat := 2

theorem apples_vs_cherries :
  (apple_days_per_week * pies_per_day) - (cherry_days_per_week * pies_per_day) = 12 := by
  sorry

end NUMINAMATH_GPT_apples_vs_cherries_l2372_237241


namespace NUMINAMATH_GPT_value_of_expression_l2372_237214

theorem value_of_expression (m n : ℝ) (h : m + 2 * n = 1) : 3 * m^2 + 6 * m * n + 6 * n = 3 :=
by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_value_of_expression_l2372_237214


namespace NUMINAMATH_GPT_W_k_two_lower_bound_l2372_237213

-- Define W(k, 2)
def W (k : ℕ) (c : ℕ) : ℕ := -- smallest number such that for every n >= W(k, 2), 
  -- any 2-coloring of the set {1, 2, ..., n} contains a monochromatic arithmetic progression of length k
  sorry 

-- Define the statement to prove
theorem W_k_two_lower_bound (k : ℕ) : ∃ C > 0, W k 2 ≥ C * 2^(k / 2) :=
by
  sorry

end NUMINAMATH_GPT_W_k_two_lower_bound_l2372_237213


namespace NUMINAMATH_GPT_div_z_x_l2372_237261

variables (x y z : ℚ)

theorem div_z_x (h1 : x / y = 3) (h2 : y / z = 5 / 2) : z / x = 2 / 15 :=
sorry

end NUMINAMATH_GPT_div_z_x_l2372_237261


namespace NUMINAMATH_GPT_find_coefficients_l2372_237215

theorem find_coefficients (A B C D : ℚ) :
  (∀ x : ℚ, x ≠ -1 → 
  (A / (x + 1)) + (B / (x + 1)^2) + ((C * x + D) / (x^2 + x + 1)) = 
  1 / ((x + 1)^2 * (x^2 + x + 1))) →
  A = 1 ∧ B = 1 ∧ C = -1 ∧ D = -1 :=
sorry

end NUMINAMATH_GPT_find_coefficients_l2372_237215


namespace NUMINAMATH_GPT_function_solution_l2372_237236

theorem function_solution (f : ℝ → ℝ) (H : ∀ x y : ℝ, 1 < x → 1 < y → f x - f y = (y - x) * f (x * y)) :
  ∃ k : ℝ, ∀ x : ℝ, 1 < x → f x = k / x :=
by
  sorry

end NUMINAMATH_GPT_function_solution_l2372_237236


namespace NUMINAMATH_GPT_find_angle_OD_base_l2372_237203

noncomputable def angle_between_edge_and_base (α β : ℝ): ℝ :=
  Real.arctan ((Real.sin α * Real.sin β) / Real.sqrt (Real.sin (α - β) * Real.sin (α + β)))

theorem find_angle_OD_base (α β : ℝ) :
  ∃ γ : ℝ, γ = angle_between_edge_and_base α β :=
sorry

end NUMINAMATH_GPT_find_angle_OD_base_l2372_237203


namespace NUMINAMATH_GPT_garden_width_l2372_237218

variable (W : ℝ) (L : ℝ := 225) (small_gate : ℝ := 3) (large_gate: ℝ := 10) (total_fencing : ℝ := 687)

theorem garden_width :
  2 * L + 2 * W - (small_gate + large_gate) = total_fencing → W = 125 := 
by
  sorry

end NUMINAMATH_GPT_garden_width_l2372_237218


namespace NUMINAMATH_GPT_enjoyable_gameplay_time_l2372_237244

def total_gameplay_time_base : ℝ := 150
def enjoyable_fraction_base : ℝ := 0.30
def total_gameplay_time_expansion : ℝ := 50
def load_screen_fraction_expansion : ℝ := 0.25
def inventory_management_fraction_expansion : ℝ := 0.25
def mod_skip_fraction : ℝ := 0.15

def enjoyable_time_base : ℝ := total_gameplay_time_base * enjoyable_fraction_base
def not_load_screen_time_expansion : ℝ := total_gameplay_time_expansion * (1 - load_screen_fraction_expansion)
def not_inventory_management_time_expansion : ℝ := not_load_screen_time_expansion * (1 - inventory_management_fraction_expansion)

def tedious_time_base : ℝ := total_gameplay_time_base * (1 - enjoyable_fraction_base)
def tedious_time_expansion : ℝ := total_gameplay_time_expansion - not_inventory_management_time_expansion
def total_tedious_time : ℝ := tedious_time_base + tedious_time_expansion

def time_skipped_by_mod : ℝ := total_tedious_time * mod_skip_fraction

def total_enjoyable_time : ℝ := enjoyable_time_base + not_inventory_management_time_expansion + time_skipped_by_mod

theorem enjoyable_gameplay_time :
  total_enjoyable_time = 92.16 :=     by     simp [total_enjoyable_time, enjoyable_time_base, not_inventory_management_time_expansion, time_skipped_by_mod]; sorry

end NUMINAMATH_GPT_enjoyable_gameplay_time_l2372_237244


namespace NUMINAMATH_GPT_smallest_n_for_cookies_l2372_237257

theorem smallest_n_for_cookies :
  ∃ n : ℕ, 15 * n - 1 % 11 = 0 ∧ (∀ m : ℕ, 15 * m - 1 % 11 = 0 → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_n_for_cookies_l2372_237257


namespace NUMINAMATH_GPT_speed_of_current_eq_l2372_237232

theorem speed_of_current_eq :
  ∃ (m c : ℝ), (m + c = 15) ∧ (m - c = 8.6) ∧ (c = 3.2) :=
by
  sorry

end NUMINAMATH_GPT_speed_of_current_eq_l2372_237232


namespace NUMINAMATH_GPT_original_price_four_pack_l2372_237239

theorem original_price_four_pack (price_with_rush: ℝ) (increase_rate: ℝ) (num_packs: ℕ):
  price_with_rush = 13 → increase_rate = 0.30 → num_packs = 4 → num_packs * (price_with_rush / (1 + increase_rate)) = 40 :=
by
  intros h_price h_rate h_packs
  rw [h_price, h_rate, h_packs]
  sorry

end NUMINAMATH_GPT_original_price_four_pack_l2372_237239


namespace NUMINAMATH_GPT_percentage_of_360_equals_126_l2372_237211

/-- 
  Prove that (126 / 360) * 100 equals 35.
-/
theorem percentage_of_360_equals_126 : (126 / 360 : ℝ) * 100 = 35 := by
  sorry

end NUMINAMATH_GPT_percentage_of_360_equals_126_l2372_237211


namespace NUMINAMATH_GPT_sum_of_invalid_domain_of_g_l2372_237264

noncomputable def g (x : ℝ) : ℝ := 1 / (2 + (1 / (3 + (1 / x))))

theorem sum_of_invalid_domain_of_g : 
  (0 : ℝ) + (-1 / 3) + (-2 / 7) = -13 / 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_invalid_domain_of_g_l2372_237264


namespace NUMINAMATH_GPT_count_ball_distributions_l2372_237217

theorem count_ball_distributions : 
  ∃ (n : ℕ), n = 3 ∧
  (∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → (∀ (dist : ℕ → ℕ), (sorry: Prop))) := sorry

end NUMINAMATH_GPT_count_ball_distributions_l2372_237217


namespace NUMINAMATH_GPT_revenue_change_l2372_237260

theorem revenue_change (T C : ℝ) (T_new C_new : ℝ)
  (h1 : T_new = 0.81 * T)
  (h2 : C_new = 1.15 * C)
  (R : ℝ := T * C) : 
  ((T_new * C_new - R) / R) * 100 = -6.85 :=
by
  sorry

end NUMINAMATH_GPT_revenue_change_l2372_237260


namespace NUMINAMATH_GPT_scientific_notation_15_7_trillion_l2372_237273

theorem scientific_notation_15_7_trillion :
  ∃ n : ℝ, n = 15.7 * 10^12 ∧ n = 1.57 * 10^13 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_15_7_trillion_l2372_237273


namespace NUMINAMATH_GPT_total_revenue_correct_l2372_237271

def small_slices_price := 150
def large_slices_price := 250
def total_slices_sold := 5000
def small_slices_sold := 2000

def large_slices_sold := total_slices_sold - small_slices_sold

def revenue_from_small_slices := small_slices_sold * small_slices_price
def revenue_from_large_slices := large_slices_sold * large_slices_price
def total_revenue := revenue_from_small_slices + revenue_from_large_slices

theorem total_revenue_correct : total_revenue = 1050000 := by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l2372_237271


namespace NUMINAMATH_GPT_white_area_is_69_l2372_237277

def area_of_sign : ℕ := 6 * 20

def area_of_M : ℕ := 2 * (6 * 1) + 2 * 2

def area_of_A : ℕ := 2 * 4 + 1 * 2

def area_of_T : ℕ := 1 * 4 + 6 * 1

def area_of_H : ℕ := 2 * (6 * 1) + 1 * 3

def total_black_area : ℕ := area_of_M + area_of_A + area_of_T + area_of_H

def white_area (sign_area black_area : ℕ) : ℕ := sign_area - black_area

theorem white_area_is_69 : white_area area_of_sign total_black_area = 69 := by
  sorry

end NUMINAMATH_GPT_white_area_is_69_l2372_237277


namespace NUMINAMATH_GPT_Nell_has_123_more_baseball_cards_than_Ace_cards_l2372_237263

def Nell_cards_diff (baseball_cards_new : ℕ) (ace_cards_new : ℕ) : ℕ :=
  baseball_cards_new - ace_cards_new

theorem Nell_has_123_more_baseball_cards_than_Ace_cards:
  (Nell_cards_diff 178 55) = 123 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_Nell_has_123_more_baseball_cards_than_Ace_cards_l2372_237263


namespace NUMINAMATH_GPT_rectangle_divided_into_13_squares_l2372_237250

theorem rectangle_divided_into_13_squares (s a b : ℕ) (h₁ : a * b = 13 * s^2)
  (h₂ : ∃ k l : ℕ, a = k * s ∧ b = l * s ∧ k * l = 13) :
  (a = s ∧ b = 13 * s) ∨ (a = 13 * s ∧ b = s) :=
by
sorry

end NUMINAMATH_GPT_rectangle_divided_into_13_squares_l2372_237250


namespace NUMINAMATH_GPT_monthly_installment_amount_l2372_237259

theorem monthly_installment_amount (total_cost : ℝ) (down_payment_percentage : ℝ) (additional_down_payment : ℝ) 
  (balance_after_months : ℝ) (months : ℕ) (monthly_installment : ℝ) : 
    total_cost = 1000 → 
    down_payment_percentage = 0.20 → 
    additional_down_payment = 20 → 
    balance_after_months = 520 → 
    months = 4 → 
    monthly_installment = 65 :=
by
  intros
  sorry

end NUMINAMATH_GPT_monthly_installment_amount_l2372_237259


namespace NUMINAMATH_GPT_local_odd_function_range_of_a_l2372_237252

variable (f : ℝ → ℝ)
variable (a : ℝ)

def local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (-x₀) = -f x₀

theorem local_odd_function_range_of_a (hf : ∀ x, f x = -a * (2^x) - 4) :
  local_odd_function f → (-4 ≤ a ∧ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_local_odd_function_range_of_a_l2372_237252


namespace NUMINAMATH_GPT_min_value_four_x_plus_one_over_x_l2372_237229

theorem min_value_four_x_plus_one_over_x (x : ℝ) (hx : x > 0) : 4*x + 1/x ≥ 4 := by
  sorry

end NUMINAMATH_GPT_min_value_four_x_plus_one_over_x_l2372_237229


namespace NUMINAMATH_GPT_g_correct_l2372_237220

-- Define the polynomials involved
def p1 (x : ℝ) : ℝ := 2 * x^5 + 4 * x^3 - 3 * x
def p2 (x : ℝ) : ℝ := 7 * x^3 + 5 * x - 2

-- Define g(x) as the polynomial we need to find
def g (x : ℝ) : ℝ := -2 * x^5 + 3 * x^3 + 8 * x - 2

-- Now, state the condition
def condition (x : ℝ) : Prop := p1 x + g x = p2 x

-- Prove the condition holds with the defined polynomials
theorem g_correct (x : ℝ) : condition x :=
by
  change p1 x + g x = p2 x
  sorry

end NUMINAMATH_GPT_g_correct_l2372_237220


namespace NUMINAMATH_GPT_union_of_A_and_B_l2372_237270

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2372_237270
