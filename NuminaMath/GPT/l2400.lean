import Mathlib

namespace NUMINAMATH_GPT_compute_xy_l2400_240090

theorem compute_xy (x y : ℝ) (h1 : x - y = 5) (h2 : x^3 - y^3 = 62) : 
  xy = -126 / 25 ∨ xy = -6 := 
sorry

end NUMINAMATH_GPT_compute_xy_l2400_240090


namespace NUMINAMATH_GPT_carol_ate_12_cakes_l2400_240010

-- Definitions for conditions
def cakes_per_day : ℕ := 10
def days_baking : ℕ := 5
def cans_per_cake : ℕ := 2
def cans_for_remaining_cakes : ℕ := 76

-- Total cakes baked by Sara
def total_cakes_baked (cakes_per_day days_baking : ℕ) : ℕ :=
  cakes_per_day * days_baking

-- Remaining cakes based on frosting cans needed
def remaining_cakes (cans_for_remaining_cakes cans_per_cake : ℕ) : ℕ :=
  cans_for_remaining_cakes / cans_per_cake

-- Cakes Carol ate
def cakes_carol_ate (total_cakes remaining_cakes : ℕ) : ℕ :=
  total_cakes - remaining_cakes

-- Theorem statement
theorem carol_ate_12_cakes :
  cakes_carol_ate (total_cakes_baked cakes_per_day days_baking) (remaining_cakes cans_for_remaining_cakes cans_per_cake) = 12 :=
by
  sorry

end NUMINAMATH_GPT_carol_ate_12_cakes_l2400_240010


namespace NUMINAMATH_GPT_circumradius_of_regular_tetrahedron_l2400_240085

theorem circumradius_of_regular_tetrahedron (a : ℝ) (h : a > 0) :
    ∃ R : ℝ, R = a * (Real.sqrt 6) / 4 :=
by
  sorry

end NUMINAMATH_GPT_circumradius_of_regular_tetrahedron_l2400_240085


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2400_240086

-- Define the lengths of the sides
def side1 : ℕ := 4
def side2 : ℕ := 7

-- Condition: The given sides form an isosceles triangle
def is_isosceles_triangle (a b : ℕ) : Prop := a = b ∨ a = 4 ∧ b = 7 ∨ a = 7 ∧ b = 4

-- Condition: The triangle inequality theorem must be satisfied
def triangle_inequality (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem we want to prove
theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : is_isosceles_triangle a b) (h2 : triangle_inequality a a b ∨ triangle_inequality b b a) :
  a + a + b = 15 ∨ b + b + a = 18 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2400_240086


namespace NUMINAMATH_GPT_area_of_black_region_l2400_240064

theorem area_of_black_region :
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  area_large - total_area_small = 94 :=
by
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  sorry

end NUMINAMATH_GPT_area_of_black_region_l2400_240064


namespace NUMINAMATH_GPT_minimum_cards_to_ensure_60_of_same_color_l2400_240051

-- Define the conditions as Lean definitions
def total_cards : ℕ := 700
def ratio_red_orange_yellow : ℕ × ℕ × ℕ := (1, 3, 4)
def ratio_green_blue_white : ℕ × ℕ × ℕ := (3, 1, 6)
def yellow_more_than_blue : ℕ := 50

-- Define the proof goal
theorem minimum_cards_to_ensure_60_of_same_color :
  ∀ (x y : ℕ),
  (total_cards = (1 * x + 3 * x + 4 * x + 3 * y + y + 6 * y)) ∧
  (4 * x = y + yellow_more_than_blue) →
  min_cards :=
  -- Sorry here to indicate that proof is not provided
  sorry

end NUMINAMATH_GPT_minimum_cards_to_ensure_60_of_same_color_l2400_240051


namespace NUMINAMATH_GPT_expenditure_on_house_rent_l2400_240073

theorem expenditure_on_house_rent
  (income petrol house_rent remaining_income : ℝ)
  (h1 : petrol = 0.30 * income)
  (h2 : petrol = 300)
  (h3 : remaining_income = income - petrol)
  (h4 : house_rent = 0.30 * remaining_income) :
  house_rent = 210 :=
by
  sorry

end NUMINAMATH_GPT_expenditure_on_house_rent_l2400_240073


namespace NUMINAMATH_GPT_total_remaining_macaroons_l2400_240028

-- Define initial macaroons count
def initial_red_macaroons : ℕ := 50
def initial_green_macaroons : ℕ := 40

-- Define macaroons eaten
def eaten_green_macaroons : ℕ := 15
def eaten_red_macaroons : ℕ := 2 * eaten_green_macaroons

-- Define remaining macaroons
def remaining_red_macaroons : ℕ := initial_red_macaroons - eaten_red_macaroons
def remaining_green_macaroons : ℕ := initial_green_macaroons - eaten_green_macaroons

-- Prove the total remaining macaroons
theorem total_remaining_macaroons : remaining_red_macaroons + remaining_green_macaroons = 45 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_remaining_macaroons_l2400_240028


namespace NUMINAMATH_GPT_measure_15_minutes_with_hourglasses_l2400_240070

theorem measure_15_minutes_with_hourglasses (h7 h11 : ℕ) (h7_eq : h7 = 7) (h11_eq : h11 = 11) : ∃ t : ℕ, t = 15 :=
by
  let t := 15
  have h7 : ℕ := 7
  have h11 : ℕ := 11
  exact ⟨t, by norm_num⟩

end NUMINAMATH_GPT_measure_15_minutes_with_hourglasses_l2400_240070


namespace NUMINAMATH_GPT_problem_l2400_240075

noncomputable def investment : ℝ := 13500
noncomputable def total_yield : ℝ := 19000
noncomputable def orchard_price_per_kg : ℝ := 4
noncomputable def market_price_per_kg (x : ℝ) : ℝ := x
noncomputable def daily_sales_rate_market : ℝ := 1000
noncomputable def days_to_sell_all (yield : ℝ) (rate : ℝ) : ℝ := yield / rate

-- Condition that x > 4
axiom x_gt_4 : ∀ (x : ℝ), x > 4

theorem problem (
  x : ℝ
) (hx : x > 4) : 
  -- Part 1
  days_to_sell_all total_yield daily_sales_rate_market = 19 ∧
  -- Part 2
  (total_yield * market_price_per_kg x - total_yield * orchard_price_per_kg) = 19000 * x - 76000 ∧
  -- Part 3
  (6000 * orchard_price_per_kg + (total_yield - 6000) * x - investment) = 13000 * x + 10500 :=
by sorry

end NUMINAMATH_GPT_problem_l2400_240075


namespace NUMINAMATH_GPT_purchase_probability_l2400_240048

/--
A batch of products from a company has packages containing 10 components each.
Each package has either 1 or 2 second-grade components. 10% of the packages
contain 2 second-grade components. Xiao Zhang will decide to purchase
if all 4 randomly selected components from a package are first-grade.

We aim to prove the probability that Xiao Zhang decides to purchase the company's
products is \( \frac{43}{75} \).
-/
theorem purchase_probability : true := sorry

end NUMINAMATH_GPT_purchase_probability_l2400_240048


namespace NUMINAMATH_GPT_min_value_f_min_value_achieved_l2400_240059

noncomputable def f (x y : ℝ) : ℝ :=
  (x^4 / y^4) + (y^4 / x^4) - (x^2 / y^2) - (y^2 / x^2) + (x / y) + (y / x)

theorem min_value_f :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → f x y ≥ 2 :=
sorry

theorem min_value_achieved :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → (f x y = 2) ↔ (x = y) :=
sorry

end NUMINAMATH_GPT_min_value_f_min_value_achieved_l2400_240059


namespace NUMINAMATH_GPT_marbles_in_jar_l2400_240066

theorem marbles_in_jar (T : ℕ) (T_half : T / 2 = 12) (red_marbles : ℕ) (orange_marbles : ℕ) (total_non_blue : red_marbles + orange_marbles = 12) (red_count : red_marbles = 6) (orange_count : orange_marbles = 6) : T = 24 :=
by
  sorry

end NUMINAMATH_GPT_marbles_in_jar_l2400_240066


namespace NUMINAMATH_GPT_shaded_square_percentage_l2400_240006

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 16) (h2 : shaded_squares = 8) : 
  (shaded_squares : ℚ) / total_squares * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_shaded_square_percentage_l2400_240006


namespace NUMINAMATH_GPT_smallest_munificence_monic_cubic_polynomial_l2400_240091

theorem smallest_munificence_monic_cubic_polynomial :
  ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = x^3 + a * x^2 + b * x + c) ∧
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) ∧
  (∀ (M : ℝ), (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |f x| ≤ M) → M ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_munificence_monic_cubic_polynomial_l2400_240091


namespace NUMINAMATH_GPT_range_of_a_l2400_240039

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = Real.sqrt x) :
  (f a < f (a + 1)) ↔ a ∈ Set.Ici (-1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2400_240039


namespace NUMINAMATH_GPT_rhombus_area_l2400_240065

-- Define the given conditions as parameters
variables (EF GH : ℝ) -- Sides of the rhombus
variables (d1 d2 : ℝ) -- Diagonals of the rhombus

-- Statement of the theorem
theorem rhombus_area
  (rhombus_EFGH : ∀ (EF GH : ℝ), EF = GH)
  (perimeter_EFGH : 4 * EF = 40)
  (diagonal_EG_length : d1 = 16)
  (d1_half : d1 / 2 = 8)
  (side_length : EF = 10)
  (pythagorean_theorem : EF^2 = (d1 / 2)^2 + (d2 / 2)^2)
  (calculate_FI : d2 / 2 = 6)
  (diagonal_FG_length : d2 = 12) :
  (1 / 2) * d1 * d2 = 96 :=
sorry

end NUMINAMATH_GPT_rhombus_area_l2400_240065


namespace NUMINAMATH_GPT_problem_b_correct_l2400_240084

theorem problem_b_correct (a b : ℝ) (h₁ : a < 0) (h₂ : 0 < b) (h₃ : b < 1) : (ab^2 > ab ∧ ab > a) :=
by
  sorry

end NUMINAMATH_GPT_problem_b_correct_l2400_240084


namespace NUMINAMATH_GPT_correct_statements_l2400_240050

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

noncomputable def a_n_sequence (n : ℕ) := a n
noncomputable def Sn_sum (n : ℕ) := S n

axiom Sn_2022_lt_zero : S 2022 < 0
axiom Sn_2023_gt_zero : S 2023 > 0

theorem correct_statements :
  (a 1012 > 0) ∧ ( ∀ n, S n >= S 1011 → n = 1011) :=
  sorry

end NUMINAMATH_GPT_correct_statements_l2400_240050


namespace NUMINAMATH_GPT_coin_difference_l2400_240031

theorem coin_difference : ∀ (p : ℕ), 1 ≤ p ∧ p ≤ 999 → (10000 - 9 * 1) - (10000 - 9 * 999) = 8982 :=
by
  intro p
  intro hp
  sorry

end NUMINAMATH_GPT_coin_difference_l2400_240031


namespace NUMINAMATH_GPT_range_of_m_for_circle_l2400_240098

theorem range_of_m_for_circle (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + m*x - 2*y + 4 = 0)  ↔ m < -2*Real.sqrt 3 ∨ m > 2*Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_for_circle_l2400_240098


namespace NUMINAMATH_GPT_proof_problem_l2400_240029

-- Definitions of propositions p and q
def p (a b : ℝ) : Prop := a < b → ∀ c : ℝ, c ≠ 0 → a * c^2 < b * c^2
def q : Prop := ∃ x₀ > 0, x₀ - 1 + Real.log x₀ = 0

-- Conditions for the problem
variable (a b : ℝ)
variable (p_false : ¬ p a b)
variable (q_true : q)

-- Proving which compound proposition is true
theorem proof_problem : (¬ p a b) ∧ q := by
  exact ⟨p_false, q_true⟩

end NUMINAMATH_GPT_proof_problem_l2400_240029


namespace NUMINAMATH_GPT_total_biking_distance_l2400_240002

-- Define the problem conditions 
def shelves := 4
def books_per_shelf := 400
def one_way_distance := shelves * books_per_shelf

-- Prove that the total distance for a round trip is 3200 miles
theorem total_biking_distance : 2 * one_way_distance = 3200 :=
by sorry

end NUMINAMATH_GPT_total_biking_distance_l2400_240002


namespace NUMINAMATH_GPT_find_natural_n_l2400_240030

theorem find_natural_n (a : ℂ) (h₀ : a ≠ 0) (h₁ : a ≠ 1) (h₂ : a ≠ -1)
    (h₃ : a ^ 11 + a ^ 7 + a ^ 3 = 1) : a ^ 4 + a ^ 3 = a ^ 15 + 1 :=
sorry

end NUMINAMATH_GPT_find_natural_n_l2400_240030


namespace NUMINAMATH_GPT_efficiency_ratio_l2400_240033

theorem efficiency_ratio (E_A E_B : ℝ) 
  (h1 : E_B = 1 / 18) 
  (h2 : E_A + E_B = 1 / 6) : 
  E_A / E_B = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_efficiency_ratio_l2400_240033


namespace NUMINAMATH_GPT_initial_black_water_bottles_l2400_240025

-- Define the conditions
variables (red black blue taken left total : ℕ)
variables (hred : red = 2) (hblue : blue = 4) (htaken : taken = 5) (hleft : left = 4)

-- State the theorem with the correct answer given the conditions
theorem initial_black_water_bottles : (red + black + blue = taken + left) → black = 3 :=
by
  intros htotal
  rw [hred, hblue, htaken, hleft] at htotal
  sorry

end NUMINAMATH_GPT_initial_black_water_bottles_l2400_240025


namespace NUMINAMATH_GPT_complement_intersection_l2400_240004

-- Define the universal set U and sets A and B.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 4, 6}
def B : Set ℕ := {4, 5, 7}

-- Define the complements of A and B in U.
def C_UA : Set ℕ := U \ A
def C_UB : Set ℕ := U \ B

-- The proof problem: Prove that the intersection of the complements of A and B 
-- in the universal set U equals {2, 3, 8}.
theorem complement_intersection :
  (C_UA ∩ C_UB = {2, 3, 8}) := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l2400_240004


namespace NUMINAMATH_GPT_space_left_each_side_l2400_240020

theorem space_left_each_side (wall_width : ℕ) (picture_width : ℕ)
  (picture_centered : wall_width = 2 * ((wall_width - picture_width) / 2) + picture_width) :
  (wall_width - picture_width) / 2 = 9 :=
by
  have h : wall_width = 25 := sorry
  have h2 : picture_width = 7 := sorry
  exact sorry

end NUMINAMATH_GPT_space_left_each_side_l2400_240020


namespace NUMINAMATH_GPT_worth_of_presents_is_33536_36_l2400_240046

noncomputable def total_worth_of_presents : ℝ :=
  let ring := 4000
  let car := 2000
  let bracelet := 2 * ring
  let gown := bracelet / 2
  let jewelry := 1.2 * ring
  let painting := 3000 * 1.2
  let honeymoon := 180000 / 110
  let watch := 5500
  ring + car + bracelet + gown + jewelry + painting + honeymoon + watch

theorem worth_of_presents_is_33536_36 : total_worth_of_presents = 33536.36 := by
  sorry

end NUMINAMATH_GPT_worth_of_presents_is_33536_36_l2400_240046


namespace NUMINAMATH_GPT_journey_speed_l2400_240077

theorem journey_speed (t_total : ℝ) (d_total : ℝ) (d_half : ℝ) (v_half2 : ℝ) (time_half2 : ℝ) (time_total : ℝ) (v_half1 : ℝ) :
  t_total = 5 ∧ d_total = 112 ∧ d_half = d_total / 2 ∧ v_half2 = 24 ∧ time_half2 = d_half / v_half2 ∧ time_total = t_total - time_half2 ∧ v_half1 = d_half / time_total → v_half1 = 21 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_journey_speed_l2400_240077


namespace NUMINAMATH_GPT_sum_of_base_areas_eq_5_l2400_240081

-- Define the surface area, lateral area, and the sum of the areas of the two base faces.
def surface_area : ℝ := 30
def lateral_area : ℝ := 25
def sum_base_areas : ℝ := surface_area - lateral_area

-- The theorem statement.
theorem sum_of_base_areas_eq_5 : sum_base_areas = 5 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_base_areas_eq_5_l2400_240081


namespace NUMINAMATH_GPT_result_is_0_85_l2400_240021

noncomputable def calc_expression := 1.85 - 1.85 / 1.85

theorem result_is_0_85 : calc_expression = 0.85 :=
by 
  sorry

end NUMINAMATH_GPT_result_is_0_85_l2400_240021


namespace NUMINAMATH_GPT_anthony_lunch_money_l2400_240096

-- Define the costs as given in the conditions
def juice_box_cost : ℕ := 27
def cupcake_cost : ℕ := 40
def amount_left : ℕ := 8

-- Define the total amount needed for lunch every day
def total_amount_for_lunch : ℕ := juice_box_cost + cupcake_cost + amount_left

theorem anthony_lunch_money : total_amount_for_lunch = 75 := by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_anthony_lunch_money_l2400_240096


namespace NUMINAMATH_GPT_ellipse_hexagon_proof_l2400_240013

noncomputable def m_value : ℝ := 3 + 2 * Real.sqrt 3

theorem ellipse_hexagon_proof (m : ℝ) (k : ℝ) 
  (hk : k ≠ 0) (hm : m > 3) :
  (∀ x y : ℝ, (x / m)^2 + (y / 3)^2 = 1 ∧ (y = k * x ∨ y = -k * x)) →
  k = Real.sqrt 3 →
  (|((4*m)/(m+1)) - (m-3)| = 0) →
  m = m_value :=
by
  sorry

end NUMINAMATH_GPT_ellipse_hexagon_proof_l2400_240013


namespace NUMINAMATH_GPT_savings_percentage_l2400_240014

variables (I S : ℝ)
-- Conditions
-- A man saves a certain portion S of his income I during the first year.
-- He spends the remaining portion (I - S) on his personal expenses.
-- In the second year, his income increases by 50%, so his new income is 1.5I.
-- His savings increase by 100%, so his new savings are 2S.
-- His total expenditure in 2 years is double his expenditure in the first year.

def first_year_expenditure (I S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (S : ℝ) : ℝ := 2 * S
def second_year_expenditure (I S : ℝ) : ℝ := second_year_income I - second_year_savings S
def total_expenditure (I S : ℝ) : ℝ := first_year_expenditure I S + second_year_expenditure I S

theorem savings_percentage :
  total_expenditure I S = 2 * first_year_expenditure I S → S / I = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_savings_percentage_l2400_240014


namespace NUMINAMATH_GPT_find_number_of_students_l2400_240071

-- Definitions for the conditions
def avg_age_students := 14
def teacher_age := 65
def new_avg_age := 15

-- The total age of students is n multiplied by their average age
def total_age_students (n : ℕ) := n * avg_age_students

-- The total age including teacher
def total_age_incl_teacher (n : ℕ) := total_age_students n + teacher_age

-- The new average age when teacher is included
def new_avg_age_incl_teacher (n : ℕ) := total_age_incl_teacher n / (n + 1)

theorem find_number_of_students (n : ℕ) (h₁ : avg_age_students = 14) (h₂ : teacher_age = 65) (h₃ : new_avg_age = 15) 
  (h_averages_eq : new_avg_age_incl_teacher n = new_avg_age) : n = 50 :=
  sorry

end NUMINAMATH_GPT_find_number_of_students_l2400_240071


namespace NUMINAMATH_GPT_union_of_M_and_N_l2400_240054

def M : Set Int := {-1, 0, 1}
def N : Set Int := {0, 1, 2}

theorem union_of_M_and_N :
  M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_union_of_M_and_N_l2400_240054


namespace NUMINAMATH_GPT_longest_perimeter_l2400_240011

theorem longest_perimeter 
  (x : ℝ) (h : x > 1)
  (pA : ℝ := 4 + 6 * x)
  (pB : ℝ := 2 + 10 * x)
  (pC : ℝ := 7 + 5 * x)
  (pD : ℝ := 6 + 6 * x)
  (pE : ℝ := 1 + 11 * x) :
  pE > pA ∧ pE > pB ∧ pE > pC ∧ pE > pD :=
by
  sorry

end NUMINAMATH_GPT_longest_perimeter_l2400_240011


namespace NUMINAMATH_GPT_smallest_m_l2400_240034

-- Let n be a positive integer and r be a positive real number less than 1/5000
def valid_r (r : ℝ) : Prop := 0 < r ∧ r < 1 / 5000

def m (n : ℕ) (r : ℝ) := (n + r)^3

theorem smallest_m : (∃ (n : ℕ) (r : ℝ), valid_r r ∧ n ≥ 41 ∧ m n r = 68922) :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_l2400_240034


namespace NUMINAMATH_GPT_speed_of_river_l2400_240019

theorem speed_of_river :
  ∃ v : ℝ, 
    (∀ d : ℝ, (2 * d = 9.856) → 
              (d = 4.928) ∧ 
              (1 = (d / (10 - v) + d / (10 + v)))) 
    → v = 1.2 :=
sorry

end NUMINAMATH_GPT_speed_of_river_l2400_240019


namespace NUMINAMATH_GPT_average_score_after_19_innings_l2400_240032

/-
  Problem Statement:
  Prove that the cricketer's average score after 19 innings is 24,
  given that scoring 96 runs in the 19th inning increased his average by 4.
-/

theorem average_score_after_19_innings :
  ∀ A : ℕ,
  (18 * A + 96) / 19 = A + 4 → A + 4 = 24 :=
by
  intros A h
  /- Skipping proof by adding "sorry" -/
  sorry

end NUMINAMATH_GPT_average_score_after_19_innings_l2400_240032


namespace NUMINAMATH_GPT_fencing_required_l2400_240012

-- Conditions
def L : ℕ := 20
def A : ℕ := 680

-- Statement to prove
theorem fencing_required : ∃ W : ℕ, A = L * W ∧ 2 * W + L = 88 :=
by
  -- Here you would normally need the logical steps to arrive at the proof
  sorry

end NUMINAMATH_GPT_fencing_required_l2400_240012


namespace NUMINAMATH_GPT_cylinder_volume_l2400_240089

theorem cylinder_volume (V1 V2 : ℝ) (π : ℝ) (r1 r3 h2 h5 : ℝ)
  (h_radii_ratio : r3 = 3 * r1)
  (h_heights_ratio : h5 = 5 / 2 * h2)
  (h_first_volume : V1 = π * r1^2 * h2)
  (h_V1_value : V1 = 40) :
  V2 = 900 :=
by sorry

end NUMINAMATH_GPT_cylinder_volume_l2400_240089


namespace NUMINAMATH_GPT_eulers_formula_l2400_240040

-- Definitions related to simply connected polyhedra
def SimplyConnectedPolyhedron (V E F : ℕ) : Prop := true  -- Genus 0 implies it is simply connected

-- Euler's characteristic property for simply connected polyhedra
theorem eulers_formula (V E F : ℕ) (h : SimplyConnectedPolyhedron V E F) : V - E + F = 2 := 
by
  sorry

end NUMINAMATH_GPT_eulers_formula_l2400_240040


namespace NUMINAMATH_GPT_flour_amount_l2400_240037

theorem flour_amount (a b : ℕ) (h₁ : a = 8) (h₂ : b = 2) : a + b = 10 := by
  sorry

end NUMINAMATH_GPT_flour_amount_l2400_240037


namespace NUMINAMATH_GPT_man_mass_calculation_l2400_240036

/-- A boat has a length of 4 m, a breadth of 2 m, and a weight of 300 kg.
    The density of the water is 1000 kg/m³.
    When the man gets on the boat, it sinks by 1 cm.
    Prove that the mass of the man is 80 kg. -/
theorem man_mass_calculation :
  let length_boat := 4     -- in meters
  let breadth_boat := 2    -- in meters
  let weight_boat := 300   -- in kg
  let density_water := 1000  -- in kg/m³
  let additional_depth := 0.01 -- in meters (1 cm)
  volume_displaced = length_boat * breadth_boat * additional_depth →
  mass_water_displaced = volume_displaced * density_water →
  mass_of_man = mass_water_displaced →
  mass_of_man = 80 :=
by 
  intros length_boat breadth_boat weight_boat density_water additional_depth volume_displaced
  intros mass_water_displaced mass_of_man
  sorry

end NUMINAMATH_GPT_man_mass_calculation_l2400_240036


namespace NUMINAMATH_GPT_determine_ABC_l2400_240022

noncomputable def digits_are_non_zero_distinct_and_not_larger_than_5 (A B C : ℕ) : Prop :=
  0 < A ∧ A ≤ 5 ∧ 0 < B ∧ B ≤ 5 ∧ 0 < C ∧ C ≤ 5 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

noncomputable def first_condition (A B : ℕ) : Prop :=
  A * 6 + B + A = B * 6 + A -- AB_6 + A_6 = BA_6 condition translated into arithmetics

noncomputable def second_condition (A B C : ℕ) : Prop :=
  A * 6 + B + B = C * 6 + 1 -- AB_6 + B_6 = C1_6 condition translated into arithmetics

theorem determine_ABC (A B C : ℕ) (h1 : digits_are_non_zero_distinct_and_not_larger_than_5 A B C)
    (h2 : first_condition A B) (h3 : second_condition A B C) :
    A * 100 + B * 10 + C = 5 * 100 + 1 * 10 + 5 := -- Final transformation of ABC to 515
  sorry

end NUMINAMATH_GPT_determine_ABC_l2400_240022


namespace NUMINAMATH_GPT_find_n_l2400_240074

-- Define the first term a₁, the common ratio q, and the sum Sₙ
def a₁ : ℕ := 2
def q : ℕ := 2
def Sₙ (n : ℕ) : ℕ := 2^(n + 1) - 2

-- The sum of the first n terms is given as 126
def given_sum : ℕ := 126

-- The theorem to be proven
theorem find_n (n : ℕ) (h : Sₙ n = given_sum) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2400_240074


namespace NUMINAMATH_GPT_ratio_both_basketball_volleyball_l2400_240045

variable (total_students : ℕ) (play_basketball : ℕ) (play_volleyball : ℕ) (play_neither : ℕ) (play_both : ℕ)

theorem ratio_both_basketball_volleyball (h1 : total_students = 20)
    (h2 : play_basketball = 20 / 2)
    (h3 : play_volleyball = (2 * 20) / 5)
    (h4 : play_neither = 4)
    (h5 : total_students - play_neither = play_basketball + play_volleyball - play_both) :
    play_both / total_students = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_both_basketball_volleyball_l2400_240045


namespace NUMINAMATH_GPT_scientific_notation_347000_l2400_240047

theorem scientific_notation_347000 :
  347000 = 3.47 * 10^5 :=
by 
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_scientific_notation_347000_l2400_240047


namespace NUMINAMATH_GPT_problem_solution_l2400_240042

theorem problem_solution (a b c : ℝ) (h : b^2 = a * c) :
  (a^2 * b^2 * c^2 / (a^3 + b^3 + c^3)) * (1 / a^3 + 1 / b^3 + 1 / c^3) = 1 :=
  by sorry

end NUMINAMATH_GPT_problem_solution_l2400_240042


namespace NUMINAMATH_GPT_term_37_l2400_240044

section GeometricSequence

variable {a b : ℕ → ℝ}
variable (q p : ℝ)

-- Definition of geometric sequences
def is_geometric_seq (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n ≥ 1, a (n + 1) = r * a n

-- Given conditions
axiom a1_25 : a 1 = 25
axiom b1_4 : b 1 = 4
axiom a2b2_100 : a 2 * b 2 = 100

-- Assume a and b are geometric sequences
axiom a_geom_seq : is_geometric_seq a q
axiom b_geom_seq : is_geometric_seq b p

-- Main theorem to prove
theorem term_37 (n : ℕ) (hn : n = 37) : (a n * b n) = 100 :=
sorry

end GeometricSequence

end NUMINAMATH_GPT_term_37_l2400_240044


namespace NUMINAMATH_GPT_find_value_of_a_l2400_240076

variable (a : ℝ)

def f (x : ℝ) := x^2 + 4
def g (x : ℝ) := x^2 - 2

theorem find_value_of_a (h_pos : a > 0) (h_eq : f (g a) = 12) : a = Real.sqrt (2 * (Real.sqrt 2 + 1)) := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l2400_240076


namespace NUMINAMATH_GPT_time_addition_correct_l2400_240060

theorem time_addition_correct :
  let current_time := (3, 0, 0)  -- Representing 3:00:00 PM as a tuple (hours, minutes, seconds)
  let duration := (313, 45, 56)  -- Duration to be added: 313 hours, 45 minutes, and 56 seconds
  let new_time := ((3 + (313 % 12) + 45 / 60 + (56 / 3600)), (0 + 45 % 60), (0 + 56 % 60))
  let A := (4 : ℕ)  -- Extracted hour part of new_time
  let B := (45 : ℕ)  -- Extracted minute part of new_time
  let C := (56 : ℕ)  -- Extracted second part of new_time
  A + B + C = 105 := 
by
  -- Placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_time_addition_correct_l2400_240060


namespace NUMINAMATH_GPT_car_speed_5_hours_l2400_240080

variable (T : ℝ)
variable (S : ℝ)

theorem car_speed_5_hours (h1 : T > 0) (h2 : 2 * T = S * 5.0) : S = 2 * T / 5.0 :=
sorry

end NUMINAMATH_GPT_car_speed_5_hours_l2400_240080


namespace NUMINAMATH_GPT_above_line_sign_l2400_240001

theorem above_line_sign (A B C x y : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
(h_above : ∃ y₁, Ax + By₁ + C = 0 ∧ y > y₁) : 
  (Ax + By + C > 0 ∧ B > 0) ∨ (Ax + By + C < 0 ∧ B < 0) := 
by
  sorry

end NUMINAMATH_GPT_above_line_sign_l2400_240001


namespace NUMINAMATH_GPT_black_squares_31x31_l2400_240099

-- Definitions to express the checkerboard problem conditions
def isCheckerboard (n : ℕ) : Prop := 
  ∀ i j : ℕ,
    i < n → j < n → 
    ((i + j) % 2 = 0 → (i % 2 = 0 ∧ j % 2 = 0) ∨ (i % 2 = 1 ∧ j % 2 = 1))

def blackCornerSquares (n : ℕ) : Prop :=
  ∀ i j : ℕ,
    (i = 0 ∨ i = n - 1) ∧ (j = 0 ∨ j = n - 1) → (i + j) % 2 = 0

-- The main statement to prove
theorem black_squares_31x31 :
  ∃ (n : ℕ) (count : ℕ), n = 31 ∧ isCheckerboard n ∧ blackCornerSquares n ∧ count = 481 := 
by 
  sorry -- Proof to be provided

end NUMINAMATH_GPT_black_squares_31x31_l2400_240099


namespace NUMINAMATH_GPT_father_age_l2400_240000

variable (F C1 C2 : ℕ)

theorem father_age (h1 : F = 3 * (C1 + C2))
  (h2 : F + 5 = 2 * (C1 + 5 + C2 + 5)) :
  F = 45 := by
  sorry

end NUMINAMATH_GPT_father_age_l2400_240000


namespace NUMINAMATH_GPT_greatest_power_sum_l2400_240061

theorem greatest_power_sum (a b : ℕ) (h1 : 0 < a) (h2 : 2 < b) (h3 : a^b < 500) (h4 : ∀ m n : ℕ, 0 < m → 2 < n → m^n < 500 → a^b ≥ m^n) : a + b = 10 :=
by
  -- Sorry is used to skip the proof steps
  sorry

end NUMINAMATH_GPT_greatest_power_sum_l2400_240061


namespace NUMINAMATH_GPT_quadratic_switch_real_roots_l2400_240055

theorem quadratic_switch_real_roots (a b c u v w : ℝ) (ha : a ≠ u)
  (h_root1 : b^2 - 4 * a * c ≥ 0)
  (h_root2 : v^2 - 4 * u * w ≥ 0)
  (hwc : w * c > 0) :
  (b^2 - 4 * u * c ≥ 0) ∨ (v^2 - 4 * a * w ≥ 0) :=
sorry

end NUMINAMATH_GPT_quadratic_switch_real_roots_l2400_240055


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2400_240083

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + 2 * x - 3 > 0 } = { x : ℝ | x < -3 ∨ x > 1 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2400_240083


namespace NUMINAMATH_GPT_Joan_spent_on_shirt_l2400_240094

/-- Joan spent $15 on shorts, $14.82 on a jacket, and a total of $42.33 on clothing.
    Prove that Joan spent $12.51 on the shirt. -/
theorem Joan_spent_on_shirt (shorts jacket total: ℝ) 
                            (h1: shorts = 15)
                            (h2: jacket = 14.82)
                            (h3: total = 42.33) :
  total - (shorts + jacket) = 12.51 :=
by
  sorry

end NUMINAMATH_GPT_Joan_spent_on_shirt_l2400_240094


namespace NUMINAMATH_GPT_friends_can_reach_destinations_l2400_240018

/-- The distance between Coco da Selva and Quixajuba is 24 km. 
    The walking speed is 6 km/h and the biking speed is 18 km/h. 
    Show that the friends can proceed to reach their destinations in at most 2 hours 40 minutes, with the bicycle initially in Quixajuba. -/
theorem friends_can_reach_destinations (d q c : ℕ) (vw vb : ℕ) (h1 : d = 24) (h2 : vw = 6) (h3 : vb = 18): 
  (∃ ta tb tc : ℕ, ta ≤ 2 * 60 + 40 ∧ tb ≤ 2 * 60 + 40 ∧ tc ≤ 2 * 60 + 40 ∧ 
     True) :=
sorry

end NUMINAMATH_GPT_friends_can_reach_destinations_l2400_240018


namespace NUMINAMATH_GPT_symmetric_to_y_axis_circle_l2400_240056

open Real

-- Definition of the original circle's equation
def original_circle (x y : ℝ) : Prop := x^2 - 2 * x + y^2 = 3

-- Definition of the symmetric circle's equation with respect to the y-axis
def symmetric_circle (x y : ℝ) : Prop := x^2 + 2 * x + y^2 = 3

-- Theorem stating that the symmetric circle has the given equation
theorem symmetric_to_y_axis_circle (x y : ℝ) : 
  (symmetric_circle x y) ↔ (original_circle ((-x) - 2) y) :=
sorry

end NUMINAMATH_GPT_symmetric_to_y_axis_circle_l2400_240056


namespace NUMINAMATH_GPT_number_of_digits_in_x20_l2400_240088

theorem number_of_digits_in_x20 (x : ℝ) (hx1 : 10^(7/4) ≤ x) (hx2 : x < 10^2) :
  10^35 ≤ x^20 ∧ x^20 < 10^36 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_digits_in_x20_l2400_240088


namespace NUMINAMATH_GPT_square_of_sum_l2400_240057

theorem square_of_sum (x y : ℝ) (A B C D : ℝ) :
  A = 2 * x^2 + y^2 →
  B = 2 * (x + y)^2 →
  C = 2 * x + y^2 →
  D = (2 * x + y)^2 →
  D = (2 * x + y)^2 :=
by intros; exact ‹D = (2 * x + y)^2›

end NUMINAMATH_GPT_square_of_sum_l2400_240057


namespace NUMINAMATH_GPT_pieces_length_l2400_240016

theorem pieces_length :
  let total_length_meters := 29.75
  let number_of_pieces := 35
  let length_per_piece_meters := total_length_meters / number_of_pieces
  let length_per_piece_centimeters := length_per_piece_meters * 100
  length_per_piece_centimeters = 85 :=
by
  sorry

end NUMINAMATH_GPT_pieces_length_l2400_240016


namespace NUMINAMATH_GPT_puppies_per_cage_l2400_240053

/-
Theorem: If a pet store had 56 puppies, sold 24 of them, and placed the remaining puppies into 8 cages, then each cage contains 4 puppies.
-/

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (cages : ℕ)
  (remaining_puppies : ℕ)
  (puppies_per_cage : ℕ) :
  initial_puppies = 56 →
  sold_puppies = 24 →
  cages = 8 →
  remaining_puppies = initial_puppies - sold_puppies →
  puppies_per_cage = remaining_puppies / cages →
  puppies_per_cage = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_puppies_per_cage_l2400_240053


namespace NUMINAMATH_GPT_range_fraction_l2400_240093

theorem range_fraction {x y : ℝ} (h : x^2 + y^2 + 2 * x = 0) :
  ∃ a b : ℝ, a = -1 ∧ b = 1 / 3 ∧ ∀ z, z = (y - x) / (x - 1) → a ≤ z ∧ z ≤ b :=
by 
  sorry

end NUMINAMATH_GPT_range_fraction_l2400_240093


namespace NUMINAMATH_GPT_minimum_notes_to_determine_prize_location_l2400_240038

/--
There are 100 boxes, numbered from 1 to 100. A prize is hidden in one of the boxes, 
and the host knows its location. The viewer can send the host a batch of notes 
with questions that require a "yes" or "no" answer. The host shuffles the notes 
in the batch and, without announcing the questions aloud, honestly answers 
all of them. Prove that the minimum number of notes that need to be sent to 
definitely determine where the prize is located is 99.
-/
theorem minimum_notes_to_determine_prize_location : 
  ∀ (boxes : Fin 100 → Prop) (prize_location : ∃ i : Fin 100, boxes i) 
    (batch_size : Nat), 
  (batch_size + 1) ≥ 100 → batch_size = 99 :=
by
  sorry

end NUMINAMATH_GPT_minimum_notes_to_determine_prize_location_l2400_240038


namespace NUMINAMATH_GPT_ladder_base_distance_l2400_240049

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end NUMINAMATH_GPT_ladder_base_distance_l2400_240049


namespace NUMINAMATH_GPT_solve_system_l2400_240015

theorem solve_system : ∀ (a b : ℝ), (∃ (x y : ℝ), x = 5 ∧ y = b ∧ 2 * x + y = a ∧ 2 * x - y = 12) → (a = 8 ∧ b = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2400_240015


namespace NUMINAMATH_GPT_deposit_percentage_l2400_240069

-- Define the conditions of the problem
def amount_deposited : ℕ := 5000
def monthly_income : ℕ := 25000

-- Define the percentage deposited formula
def percentage_deposited (amount_deposited monthly_income : ℕ) : ℚ :=
  (amount_deposited / monthly_income) * 100

-- State the theorem to be proved
theorem deposit_percentage :
  percentage_deposited amount_deposited monthly_income = 20 := by
  sorry

end NUMINAMATH_GPT_deposit_percentage_l2400_240069


namespace NUMINAMATH_GPT_trigonometric_inequality_l2400_240092

theorem trigonometric_inequality (x : ℝ) (n m : ℕ) 
  (hx : 0 < x ∧ x < (Real.pi / 2))
  (hnm : n > m) : 
  2 * |Real.sin x ^ n - Real.cos x ^ n| ≤
  3 * |Real.sin x ^ m - Real.cos x ^ m| := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l2400_240092


namespace NUMINAMATH_GPT_solve_for_x_l2400_240079

theorem solve_for_x :
  ∀ x : ℕ, 100^4 = 5^x → x = 8 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l2400_240079


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l2400_240005

noncomputable def f (x : Real) : Real :=
  Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x) ^ 2

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, ( ∀ x, f (x + T') = f x) → T ≤ T') := by
  sorry

theorem f_ge_negative_sqrt_3_in_interval :
  ∀ x ∈ Set.Icc (-Real.pi / 3) (Real.pi / 6), f x ≥ -Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_f_ge_negative_sqrt_3_in_interval_l2400_240005


namespace NUMINAMATH_GPT_cut_grid_into_six_polygons_with_identical_pair_l2400_240082

noncomputable def totalCells : Nat := 24
def polygonArea : Nat := 4

theorem cut_grid_into_six_polygons_with_identical_pair :
  ∃ (polygons : Fin 6 → Nat → Prop),
  (∀ i, (∃ (cells : Finset (Fin totalCells)), (cells.card = polygonArea ∧ ∀ c ∈ cells, polygons i c))) ∧
  (∃ i j, i ≠ j ∧ ∀ c, polygons i c ↔ polygons j c) :=
sorry

end NUMINAMATH_GPT_cut_grid_into_six_polygons_with_identical_pair_l2400_240082


namespace NUMINAMATH_GPT_jennifer_tanks_l2400_240062

theorem jennifer_tanks (initial_tanks : ℕ) (fish_per_initial_tank : ℕ) (total_fish_needed : ℕ) 
  (additional_tanks : ℕ) (fish_per_additional_tank : ℕ) 
  (initial_calculation : initial_tanks = 3) (fish_per_initial_calculation : fish_per_initial_tank = 15)
  (total_fish_calculation : total_fish_needed = 75) (additional_tanks_calculation : additional_tanks = 3) :
  initial_tanks * fish_per_initial_tank + additional_tanks * fish_per_additional_tank = total_fish_needed 
  → fish_per_additional_tank = 10 := 
by sorry

end NUMINAMATH_GPT_jennifer_tanks_l2400_240062


namespace NUMINAMATH_GPT_Jeffrey_steps_l2400_240067

theorem Jeffrey_steps
  (Andrew_steps : ℕ) (Jeffrey_steps : ℕ) (h_ratio : Andrew_steps / Jeffrey_steps = 3 / 4)
  (h_Andrew : Andrew_steps = 150) :
  Jeffrey_steps = 200 :=
by
  sorry

end NUMINAMATH_GPT_Jeffrey_steps_l2400_240067


namespace NUMINAMATH_GPT_find_number_l2400_240008

theorem find_number : ∃ x : ℝ, 0.35 * x = 0.15 * 40 ∧ x = 120 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2400_240008


namespace NUMINAMATH_GPT_inequality_solution_l2400_240007

theorem inequality_solution (x : ℝ) :
  (x / (x^2 - 4) ≥ 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ico 0 2) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l2400_240007


namespace NUMINAMATH_GPT_students_neither_football_nor_cricket_l2400_240097

theorem students_neither_football_nor_cricket 
  (total_students : ℕ) 
  (football_players : ℕ) 
  (cricket_players : ℕ) 
  (both_players : ℕ) 
  (H1 : total_students = 410) 
  (H2 : football_players = 325) 
  (H3 : cricket_players = 175) 
  (H4 : both_players = 140) :
  total_students - (football_players + cricket_players - both_players) = 50 :=
by
  sorry

end NUMINAMATH_GPT_students_neither_football_nor_cricket_l2400_240097


namespace NUMINAMATH_GPT_digit_sum_subtraction_l2400_240078

theorem digit_sum_subtraction (P Q R S : ℕ) (hQ : Q + P = P) (hP : Q - P = 0) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10) (h4 : S < 10) : S = 0 := by
  sorry

end NUMINAMATH_GPT_digit_sum_subtraction_l2400_240078


namespace NUMINAMATH_GPT_proof_evaluate_expression_l2400_240035

def evaluate_expression : Prop :=
  - (18 / 3 * 8 - 72 + 4 * 8) = 8

theorem proof_evaluate_expression : evaluate_expression :=
by 
  sorry

end NUMINAMATH_GPT_proof_evaluate_expression_l2400_240035


namespace NUMINAMATH_GPT_train_speed_kmph_l2400_240024

def length_of_train : ℝ := 120
def length_of_bridge : ℝ := 255.03
def time_to_cross : ℝ := 30

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross * 3.6 = 45.0036 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_kmph_l2400_240024


namespace NUMINAMATH_GPT_Maggie_apples_l2400_240026

-- Definition of our problem conditions
def K : ℕ := 28 -- Kelsey's apples
def L : ℕ := 22 -- Layla's apples
def avg : ℕ := 30 -- The average number of apples picked

-- Main statement to prove Maggie's apples
theorem Maggie_apples : (A : ℕ) → (A + K + L) / 3 = avg → A = 40 := by
  intros A h
  -- sorry is added to skip the proof since it's not required here.
  sorry

end NUMINAMATH_GPT_Maggie_apples_l2400_240026


namespace NUMINAMATH_GPT_derivative_at_0_5_l2400_240068

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := -2

-- State the theorem
theorem derivative_at_0_5 : f' 0.5 = -2 :=
by {
  -- Proof placeholder
  sorry
}

end NUMINAMATH_GPT_derivative_at_0_5_l2400_240068


namespace NUMINAMATH_GPT_chemical_reaction_produces_l2400_240052

def balanced_equation : Prop :=
  ∀ {CaCO3 HCl CaCl2 CO2 H2O : ℕ},
    (CaCO3 + 2 * HCl = CaCl2 + CO2 + H2O)

def calculate_final_products (initial_CaCO3 initial_HCl final_CaCl2 final_CO2 final_H2O remaining_HCl : ℕ) : Prop :=
  balanced_equation ∧
  initial_CaCO3 = 3 ∧
  initial_HCl = 8 ∧
  final_CaCl2 = 3 ∧
  final_CO2 = 3 ∧
  final_H2O = 3 ∧
  remaining_HCl = 2

theorem chemical_reaction_produces :
  calculate_final_products 3 8 3 3 3 2 :=
by sorry

end NUMINAMATH_GPT_chemical_reaction_produces_l2400_240052


namespace NUMINAMATH_GPT_percentage_of_failed_candidates_l2400_240063

theorem percentage_of_failed_candidates
(total_candidates : ℕ)
(girls : ℕ)
(passed_boys_percentage : ℝ)
(passed_girls_percentage : ℝ)
(h1 : total_candidates = 2000)
(h2 : girls = 900)
(h3 : passed_boys_percentage = 0.28)
(h4 : passed_girls_percentage = 0.32)
: (total_candidates - (passed_boys_percentage * (total_candidates - girls) + passed_girls_percentage * girls)) / total_candidates * 100 = 70.2 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_failed_candidates_l2400_240063


namespace NUMINAMATH_GPT_find_x_l2400_240072

variable (x : ℝ)
variable (h : 0.3 * 100 = 0.5 * x + 10)

theorem find_x : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2400_240072


namespace NUMINAMATH_GPT_route_C_is_quicker_l2400_240043

/-
  Define the conditions based on the problem:
  - Route C: 8 miles at 40 mph.
  - Route D: 5 miles at 35 mph and 2 miles at 25 mph with an additional 3 minutes stop.
-/

def time_route_C : ℚ := (8 : ℚ) / (40 : ℚ) * 60  -- in minutes

def time_route_D : ℚ := ((5 : ℚ) / (35 : ℚ) * 60) + ((2 : ℚ) / (25 : ℚ) * 60) + 3  -- in minutes

def time_difference : ℚ := time_route_D - time_route_C  -- difference in minutes

theorem route_C_is_quicker : time_difference = 4.37 := 
by 
  sorry

end NUMINAMATH_GPT_route_C_is_quicker_l2400_240043


namespace NUMINAMATH_GPT_complex_transformation_result_l2400_240087

theorem complex_transformation_result :
  let z := -1 - 2 * Complex.I 
  let rotation := (1 / 2 : ℂ) + (Complex.I * (Real.sqrt 3) / 2)
  let dilation := 2
  (z * (rotation * dilation)) = (2 * Real.sqrt 3 - 1 - (2 + Real.sqrt 3) * Complex.I) :=
by
  sorry

end NUMINAMATH_GPT_complex_transformation_result_l2400_240087


namespace NUMINAMATH_GPT_height_of_parallelogram_l2400_240023

def area_of_parallelogram (base height : ℝ) : ℝ := base * height

theorem height_of_parallelogram (A B H : ℝ) (hA : A = 33.3) (hB : B = 9) (hAparallelogram : A = area_of_parallelogram B H) :
  H = 3.7 :=
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_height_of_parallelogram_l2400_240023


namespace NUMINAMATH_GPT_good_numbers_characterization_l2400_240017

def is_good (n : ℕ) : Prop :=
  ∀ d, d ∣ n → d + 1 ∣ n + 1

theorem good_numbers_characterization :
  {n : ℕ | is_good n} = {1} ∪ {p | Nat.Prime p ∧ p % 2 = 1} :=
by 
  sorry

end NUMINAMATH_GPT_good_numbers_characterization_l2400_240017


namespace NUMINAMATH_GPT_area_of_rhombus_l2400_240058

-- Defining the conditions
def diagonal1 : ℝ := 20
def diagonal2 : ℝ := 30

-- Proving the area of the rhombus
theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = diagonal1) (h2 : d2 = diagonal2) : 
  (d1 * d2 / 2) = 300 := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l2400_240058


namespace NUMINAMATH_GPT_geometric_sequence_values_l2400_240003

theorem geometric_sequence_values (l a b c : ℝ) (h : ∃ r : ℝ, a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : b = -3 ∧ a * c = 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_values_l2400_240003


namespace NUMINAMATH_GPT_annie_gives_mary_25_crayons_l2400_240027

theorem annie_gives_mary_25_crayons :
  let initial_crayons_given := 21
  let initial_crayons_in_locker := 36
  let bobby_gift := initial_crayons_in_locker / 2
  let total_crayons := initial_crayons_given + initial_crayons_in_locker + bobby_gift
  let mary_share := total_crayons / 3
  mary_share = 25 := 
by
  sorry

end NUMINAMATH_GPT_annie_gives_mary_25_crayons_l2400_240027


namespace NUMINAMATH_GPT_range_of_m_l2400_240095

theorem range_of_m (m : ℝ) :
  (∃! (x : ℤ), (x < 1 ∧ x > m - 1)) →
  (-1 ≤ m ∧ m < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2400_240095


namespace NUMINAMATH_GPT_transport_equivalence_l2400_240041

theorem transport_equivalence (f : ℤ → ℤ) (x y : ℤ) (h : f x = -x) :
  f (-y) = y :=
by
  sorry

end NUMINAMATH_GPT_transport_equivalence_l2400_240041


namespace NUMINAMATH_GPT_books_leftover_l2400_240009

theorem books_leftover (boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) 
  (h1 : boxes = 1575) (h2 : books_per_box = 45) (h3 : new_box_capacity = 50) :
  ((boxes * books_per_box) % new_box_capacity) = 25 :=
by
  sorry

end NUMINAMATH_GPT_books_leftover_l2400_240009
