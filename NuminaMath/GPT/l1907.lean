import Mathlib

namespace NUMINAMATH_GPT_store_profit_l1907_190797

variable (m n : ℝ)
variable (h_mn : m > n)

theorem store_profit : 10 * (m - n) > 0 :=
by
  sorry

end NUMINAMATH_GPT_store_profit_l1907_190797


namespace NUMINAMATH_GPT_initial_percentage_of_water_l1907_190730

variable (V : ℝ) (W : ℝ) (P : ℝ)

theorem initial_percentage_of_water 
  (h1 : V = 120) 
  (h2 : W = 8)
  (h3 : (V + W) * 0.25 = ((P / 100) * V) + W) : 
  P = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_of_water_l1907_190730


namespace NUMINAMATH_GPT_point_location_l1907_190735

variables {A B C m n : ℝ}

theorem point_location (h1 : A > 0) (h2 : B < 0) (h3 : A * m + B * n + C < 0) : 
  -- Statement: the point P(m, n) is on the upper right side of the line Ax + By + C = 0
  true :=
sorry

end NUMINAMATH_GPT_point_location_l1907_190735


namespace NUMINAMATH_GPT_jonathan_daily_burn_l1907_190787

-- Conditions
def daily_calories : ℕ := 2500
def extra_saturday_calories : ℕ := 1000
def weekly_deficit : ℕ := 2500

-- Question and Answer
theorem jonathan_daily_burn :
  let weekly_intake := 6 * daily_calories + (daily_calories + extra_saturday_calories)
  let total_weekly_burn := weekly_intake + weekly_deficit
  total_weekly_burn / 7 = 3000 :=
by
  sorry

end NUMINAMATH_GPT_jonathan_daily_burn_l1907_190787


namespace NUMINAMATH_GPT_probability_2_1_to_2_5_l1907_190786

noncomputable def F (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then (x - 2)^2
else 1

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 0
else if x ≤ 3 then 2 * (x - 2)
else 0

theorem probability_2_1_to_2_5 : 
  (F 2.5 - F 2.1 = 0.24) := 
by
  -- calculations and proof go here, but we skip it with sorry
  sorry

end NUMINAMATH_GPT_probability_2_1_to_2_5_l1907_190786


namespace NUMINAMATH_GPT_inscribed_circle_radius_integer_l1907_190760

theorem inscribed_circle_radius_integer (a b c : ℕ) (h : a^2 + b^2 = c^2) : 
  ∃ (r : ℤ), r = (a + b - c) / 2 := by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_integer_l1907_190760


namespace NUMINAMATH_GPT_angles_relation_l1907_190746

/-- Given angles α and β from two right-angled triangles in a 3x3 grid such that α + β = 90°,
    prove that 2α + β = 90°. -/
theorem angles_relation (α β : ℝ) (h1 : α + β = 90) : 2 * α + β = 90 := by
  sorry

end NUMINAMATH_GPT_angles_relation_l1907_190746


namespace NUMINAMATH_GPT_general_formula_for_sequence_a_l1907_190715

noncomputable def S (n : ℕ) : ℕ := 3^n + 1

def a (n : ℕ) : ℕ :=
if n = 1 then 4 else 2 * 3^(n-1)

theorem general_formula_for_sequence_a (n : ℕ) :
  a n = if n = 1 then 4 else 2 * 3^(n-1) :=
by {
  sorry
}

end NUMINAMATH_GPT_general_formula_for_sequence_a_l1907_190715


namespace NUMINAMATH_GPT_complement_union_l1907_190772

variable (U : Set ℤ) (A : Set ℤ) (B : Set ℤ)

theorem complement_union (hU : U = {-2, -1, 0, 1, 2, 3})
                         (hA : A = {-1, 0, 1})
                         (hB : B = {1, 2}) :
  U \ (A ∪ B) = {-2, 3} := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_complement_union_l1907_190772


namespace NUMINAMATH_GPT_factorize_expression_l1907_190710

theorem factorize_expression (a : ℝ) : 
  (2 * a + 1) * a - 4 * a - 2 = (2 * a + 1) * (a - 2) :=
by 
  -- proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_factorize_expression_l1907_190710


namespace NUMINAMATH_GPT_marching_band_total_weight_l1907_190711

def weight_trumpets := 5
def weight_clarinets := 5
def weight_trombones := 10
def weight_tubas := 20
def weight_drums := 15

def count_trumpets := 6
def count_clarinets := 9
def count_trombones := 8
def count_tubas := 3
def count_drums := 2

theorem marching_band_total_weight :
  (count_trumpets * weight_trumpets) + (count_clarinets * weight_clarinets) + (count_trombones * weight_trombones) + 
  (count_tubas * weight_tubas) + (count_drums * weight_drums) = 245 :=
by
  sorry

end NUMINAMATH_GPT_marching_band_total_weight_l1907_190711


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_zero_l1907_190795

theorem arithmetic_sequence_sum_zero {a1 d n : ℤ} 
(h1 : a1 = 35) 
(h2 : d = -2) 
(h3 : (n * (2 * a1 + (n - 1) * d)) / 2 = 0) : 
n = 36 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_zero_l1907_190795


namespace NUMINAMATH_GPT_cricket_team_members_l1907_190757

theorem cricket_team_members (avg_whole_team: ℕ) (captain_age: ℕ) (wicket_keeper_age: ℕ) 
(remaining_avg_age: ℕ) (n: ℕ):
avg_whole_team = 23 →
captain_age = 25 →
wicket_keeper_age = 30 →
remaining_avg_age = 22 →
(n * avg_whole_team - captain_age - wicket_keeper_age = (n - 2) * remaining_avg_age) →
n = 11 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_cricket_team_members_l1907_190757


namespace NUMINAMATH_GPT_gift_wrapping_combinations_l1907_190703

theorem gift_wrapping_combinations :
  (10 * 4 * 5 * 2 = 400) := by
  sorry

end NUMINAMATH_GPT_gift_wrapping_combinations_l1907_190703


namespace NUMINAMATH_GPT_compound_interest_principal_amount_l1907_190751

theorem compound_interest_principal_amount :
  ∀ (r : ℝ) (n : ℕ) (t : ℕ) (CI : ℝ) (P : ℝ),
    r = 0.04 ∧ n = 1 ∧ t = 2 ∧ CI = 612 →
    (CI = P * (1 + r / n) ^ (n * t) - P) →
    P = 7500 :=
by
  intros r n t CI P h_conditions h_CI
  -- Proof not needed
  sorry

end NUMINAMATH_GPT_compound_interest_principal_amount_l1907_190751


namespace NUMINAMATH_GPT_line_through_intersection_points_of_circles_l1907_190709

theorem line_through_intersection_points_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
    (x - 2*y + 6 = 0) :=
by
  intro x y h
  -- Condition of circle 1
  have circle1 : x^2 + y^2 + 4*x - 4*y - 1 = 0 := h.left
  -- Condition of circle 2
  have circle2 : x^2 + y^2 + 2*x - 13 = 0 := h.right
  sorry

end NUMINAMATH_GPT_line_through_intersection_points_of_circles_l1907_190709


namespace NUMINAMATH_GPT_illegal_simplification_works_for_specific_values_l1907_190770

-- Definitions for the variables
def a : ℕ := 43
def b : ℕ := 17
def c : ℕ := 26

-- Define the sum of cubes
def sum_of_cubes (x y : ℕ) : ℕ := x ^ 3 + y ^ 3

-- Define the illegal simplification fraction
def illegal_simplification_fraction_correct (a b c : ℕ) : Prop :=
  (a^3 + b^3) / (a^3 + c^3) = (a + b) / (a + c)

-- The theorem to prove
theorem illegal_simplification_works_for_specific_values :
  illegal_simplification_fraction_correct a b c :=
by
  -- Proof will reside here
  sorry

end NUMINAMATH_GPT_illegal_simplification_works_for_specific_values_l1907_190770


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l1907_190762

theorem inequality_holds_for_all_x (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3 / 5 < a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l1907_190762


namespace NUMINAMATH_GPT_largest_angle_in_pentagon_l1907_190771

theorem largest_angle_in_pentagon (A B C D E : ℝ) 
  (hA : A = 70) 
  (hB : B = 120) 
  (hCD : C = D) 
  (hE : E = 3 * C - 30) 
  (sum_angles : A + B + C + D + E = 540) :
  E = 198 := 
by 
  sorry

end NUMINAMATH_GPT_largest_angle_in_pentagon_l1907_190771


namespace NUMINAMATH_GPT_solve_equation_nat_numbers_l1907_190790

theorem solve_equation_nat_numbers (a b c d e f g : ℕ) 
  (h : a * b * c * d * e * f * g = a + b + c + d + e + f + g) : 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 2 ∧ g = 7) ∨ (f = 7 ∧ g = 2))) ∨ 
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ ((f = 3 ∧ g = 4) ∨ (f = 4 ∧ g = 3))) :=
sorry

end NUMINAMATH_GPT_solve_equation_nat_numbers_l1907_190790


namespace NUMINAMATH_GPT_handshake_max_participants_l1907_190706

theorem handshake_max_participants (N : ℕ) (hN : 5 < N) (hNotAllShaken: ∃ p1 p2 : ℕ, p1 ≠ p2 ∧ p1 < N ∧ p2 < N ∧ (∀ i : ℕ, i < N → i ≠ p1 → i ≠ p2 → ∃ j : ℕ, j < N ∧ j ≠ i ∧ j ≠ p1 ∧ j ≠ p2)) :
∃ k, k = N - 2 :=
by
  sorry

end NUMINAMATH_GPT_handshake_max_participants_l1907_190706


namespace NUMINAMATH_GPT_frog_arrangement_count_l1907_190793

theorem frog_arrangement_count :
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let frogs := green_frogs + red_frogs + blue_frogs
  -- Descriptions:
  -- 1. green_frogs refuse to sit next to red_frogs
  -- 2. green_frogs and red_frogs are fine sitting next to blue_frogs
  -- 3. blue_frogs can sit next to each other
  frogs = 7 → 
  ∃ arrangements : ℕ, arrangements = 72 :=
by 
  sorry

end NUMINAMATH_GPT_frog_arrangement_count_l1907_190793


namespace NUMINAMATH_GPT_problem1_problem2_l1907_190744

-- Define Set A
def SetA : Set ℝ := { y | ∃ x, (2 ≤ x ∧ x ≤ 3) ∧ y = -2^x }

-- Define Set B parameterized by a
def SetB (a : ℝ) : Set ℝ := { x | x^2 + 3 * x - a^2 - 3 * a > 0 }

-- Problem 1: Prove that when a = 4, A ∩ B = {-8 < x < -7}
theorem problem1 : A ∩ SetB 4 = { x | -8 < x ∧ x < -7 } :=
sorry

-- Problem 2: Prove the range of a for which "x ∈ A" is a sufficient but not necessary condition for "x ∈ B"
theorem problem2 : ∀ a : ℝ, (∀ x, x ∈ SetA → x ∈ SetB a) → -4 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1907_190744


namespace NUMINAMATH_GPT_area_of_triangle_l1907_190767

-- Define the vectors a and b
def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (-3, 3)

-- The goal is to prove the area of the triangle
theorem area_of_triangle (a b : ℝ × ℝ) : 
  a = (4, -1) → b = (-3, 3) → (|4 * 3 - (-1) * (-3)| / 2) = 9 / 2  :=
by
  intros
  sorry

end NUMINAMATH_GPT_area_of_triangle_l1907_190767


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1907_190752

theorem sum_of_three_numbers
  (a b c : ℕ) (h_prime : Prime c)
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + a * c = 50) :
  a + b + c = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1907_190752


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1907_190794

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1/2 → 2 * x^2 + x - 1 > 0) ∧ ¬(2 * x^2 + x - 1 > 0 → x > 1 / 2) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1907_190794


namespace NUMINAMATH_GPT_width_of_smaller_cuboids_is_4_l1907_190785

def length_smaller_cuboid := 5
def height_smaller_cuboid := 3
def length_larger_cuboid := 16
def width_larger_cuboid := 10
def height_larger_cuboid := 12
def num_smaller_cuboids := 32

theorem width_of_smaller_cuboids_is_4 :
  ∃ W : ℝ, W = 4 ∧ (length_smaller_cuboid * W * height_smaller_cuboid) * num_smaller_cuboids = 
            length_larger_cuboid * width_larger_cuboid * height_larger_cuboid :=
by
  sorry

end NUMINAMATH_GPT_width_of_smaller_cuboids_is_4_l1907_190785


namespace NUMINAMATH_GPT_amgm_inequality_abcd_l1907_190799

-- Define the variables and their conditions
variables {a b c d : ℝ}
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)
variable (hd : 0 < d)

-- State the theorem
theorem amgm_inequality_abcd :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) :=
by
  sorry

end NUMINAMATH_GPT_amgm_inequality_abcd_l1907_190799


namespace NUMINAMATH_GPT_girls_in_class_l1907_190798

theorem girls_in_class (B G : ℕ) 
  (h1 : G = B + 3) 
  (h2 : G + B = 41) : 
  G = 22 := 
sorry

end NUMINAMATH_GPT_girls_in_class_l1907_190798


namespace NUMINAMATH_GPT_cake_pieces_in_pan_l1907_190773

theorem cake_pieces_in_pan :
  (24 * 30) / (3 * 2) = 120 := by
  sorry

end NUMINAMATH_GPT_cake_pieces_in_pan_l1907_190773


namespace NUMINAMATH_GPT_paul_earns_from_license_plates_l1907_190734

theorem paul_earns_from_license_plates
  (plates_from_40_states : ℕ)
  (total_50_states : ℕ)
  (reward_per_percentage_point : ℕ)
  (h1 : plates_from_40_states = 40)
  (h2 : total_50_states = 50)
  (h3 : reward_per_percentage_point = 2) :
  (40 / 50) * 100 * 2 = 160 := 
sorry

end NUMINAMATH_GPT_paul_earns_from_license_plates_l1907_190734


namespace NUMINAMATH_GPT_total_cube_volume_l1907_190733

theorem total_cube_volume 
  (carl_cubes : ℕ)
  (carl_cube_side : ℕ)
  (kate_cubes : ℕ)
  (kate_cube_side : ℕ)
  (hcarl : carl_cubes = 4)
  (hcarl_side : carl_cube_side = 3)
  (hkate : kate_cubes = 6)
  (hkate_side : kate_cube_side = 4) :
  (carl_cubes * carl_cube_side ^ 3) + (kate_cubes * kate_cube_side ^ 3) = 492 :=
by
  sorry

end NUMINAMATH_GPT_total_cube_volume_l1907_190733


namespace NUMINAMATH_GPT_sequence_mod_100_repeats_l1907_190720

theorem sequence_mod_100_repeats (a0 : ℕ) : ∃ k l, k ≠ l ∧ (∃ seq : ℕ → ℕ, seq 0 = a0 ∧ (∀ n, seq (n + 1) = seq n + 54 ∨ seq (n + 1) = seq n + 77) ∧ (seq k % 100 = seq l % 100)) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_mod_100_repeats_l1907_190720


namespace NUMINAMATH_GPT_max_a_squared_b_squared_c_squared_l1907_190712

theorem max_a_squared_b_squared_c_squared (a b c : ℤ)
  (h1 : a + b + c = 3)
  (h2 : a^3 + b^3 + c^3 = 3) :
  a^2 + b^2 + c^2 ≤ 57 :=
sorry

end NUMINAMATH_GPT_max_a_squared_b_squared_c_squared_l1907_190712


namespace NUMINAMATH_GPT_kramer_vote_percentage_l1907_190775

def percentage_of_votes_cast (K : ℕ) (V : ℕ) : ℕ :=
  (K * 100) / V

theorem kramer_vote_percentage (K : ℕ) (V : ℕ) (h1 : K = 942568) 
  (h2 : V = 4 * K) : percentage_of_votes_cast K V = 25 := 
by 
  rw [h1, h2, percentage_of_votes_cast]
  sorry

end NUMINAMATH_GPT_kramer_vote_percentage_l1907_190775


namespace NUMINAMATH_GPT_product_of_first_three_terms_of_arithmetic_sequence_l1907_190750

theorem product_of_first_three_terms_of_arithmetic_sequence {a d : ℕ} (ha : a + 6 * d = 20) (hd : d = 2) : a * (a + d) * (a + 2 * d) = 960 := by
  sorry

end NUMINAMATH_GPT_product_of_first_three_terms_of_arithmetic_sequence_l1907_190750


namespace NUMINAMATH_GPT_cauchy_functional_eq_l1907_190721

theorem cauchy_functional_eq
  (f : ℚ → ℚ)
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
sorry

end NUMINAMATH_GPT_cauchy_functional_eq_l1907_190721


namespace NUMINAMATH_GPT_negation_of_proposition_l1907_190737

theorem negation_of_proposition : 
  (¬ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0)) ↔ (∃ x : ℝ, x^2 + 2*x + 5 = 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l1907_190737


namespace NUMINAMATH_GPT_find_b_when_a_is_1600_l1907_190719

variable (a b : ℝ)

def inversely_vary (a b : ℝ) : Prop := a * b = 400

theorem find_b_when_a_is_1600 
  (h1 : inversely_vary 800 0.5)
  (h2 : inversely_vary a b)
  (h3 : a = 1600) :
  b = 0.25 := by
  sorry

end NUMINAMATH_GPT_find_b_when_a_is_1600_l1907_190719


namespace NUMINAMATH_GPT_pete_travel_time_l1907_190756

-- Definitions for the given conditions
def map_distance := 5.0          -- in inches
def scale := 0.05555555555555555 -- in inches per mile
def speed := 60.0                -- in miles per hour
def real_distance := map_distance / scale

-- The theorem to state the proof problem
theorem pete_travel_time : 
  real_distance = 90 → -- Based on condition deduced from earlier
  real_distance / speed = 1.5 := 
by 
  intro h1
  rw[h1]
  norm_num
  sorry

end NUMINAMATH_GPT_pete_travel_time_l1907_190756


namespace NUMINAMATH_GPT_calculate_amount_left_l1907_190725

def base_income : ℝ := 2000
def bonus_percentage : ℝ := 0.15
def public_transport_percentage : ℝ := 0.05
def rent : ℝ := 500
def utilities : ℝ := 100
def food : ℝ := 300
def miscellaneous_percentage : ℝ := 0.10
def savings_percentage : ℝ := 0.07
def investment_percentage : ℝ := 0.05
def medical_expense : ℝ := 250
def tax_percentage : ℝ := 0.15

def total_income (base_income : ℝ) (bonus_percentage : ℝ) : ℝ :=
  base_income + (bonus_percentage * base_income)

def taxes (base_income : ℝ) (tax_percentage : ℝ) : ℝ :=
  tax_percentage * base_income

def total_fixed_expenses (rent : ℝ) (utilities : ℝ) (food : ℝ) : ℝ :=
  rent + utilities + food

def public_transport_expense (total_income : ℝ) (public_transport_percentage : ℝ) : ℝ :=
  public_transport_percentage * total_income

def miscellaneous_expense (total_income : ℝ) (miscellaneous_percentage : ℝ) : ℝ :=
  miscellaneous_percentage * total_income

def variable_expenses (public_transport_expense : ℝ) (miscellaneous_expense : ℝ) : ℝ :=
  public_transport_expense + miscellaneous_expense

def savings (total_income : ℝ) (savings_percentage : ℝ) : ℝ :=
  savings_percentage * total_income

def investment (total_income : ℝ) (investment_percentage : ℝ) : ℝ :=
  investment_percentage * total_income

def total_savings_investments (savings : ℝ) (investment : ℝ) : ℝ :=
  savings + investment

def total_expenses_contributions 
  (fixed_expenses : ℝ) 
  (variable_expenses : ℝ) 
  (medical_expense : ℝ) 
  (total_savings_investments : ℝ) : ℝ :=
  fixed_expenses + variable_expenses + medical_expense + total_savings_investments

def amount_left (income_after_taxes : ℝ) (total_expenses_contributions : ℝ) : ℝ :=
  income_after_taxes - total_expenses_contributions

theorem calculate_amount_left 
  (base_income : ℝ)
  (bonus_percentage : ℝ)
  (public_transport_percentage : ℝ)
  (rent : ℝ)
  (utilities : ℝ)
  (food : ℝ)
  (miscellaneous_percentage : ℝ)
  (savings_percentage : ℝ)
  (investment_percentage : ℝ)
  (medical_expense : ℝ)
  (tax_percentage : ℝ)
  (total_income : ℝ := total_income base_income bonus_percentage)
  (taxes : ℝ := taxes base_income tax_percentage)
  (income_after_taxes : ℝ := total_income - taxes)
  (fixed_expenses : ℝ := total_fixed_expenses rent utilities food)
  (public_transport_expense : ℝ := public_transport_expense total_income public_transport_percentage)
  (miscellaneous_expense : ℝ := miscellaneous_expense total_income miscellaneous_percentage)
  (variable_expenses : ℝ := variable_expenses public_transport_expense miscellaneous_expense)
  (savings : ℝ := savings total_income savings_percentage)
  (investment : ℝ := investment total_income investment_percentage)
  (total_savings_investments : ℝ := total_savings_investments savings investment)
  (total_expenses_contributions : ℝ := total_expenses_contributions fixed_expenses variable_expenses medical_expense total_savings_investments)
  : amount_left income_after_taxes total_expenses_contributions = 229 := 
sorry

end NUMINAMATH_GPT_calculate_amount_left_l1907_190725


namespace NUMINAMATH_GPT_final_answer_is_correct_l1907_190791

-- Define the chosen number
def chosen_number : ℤ := 1376

-- Define the division by 8
def division_result : ℤ := chosen_number / 8

-- Define the final answer
def final_answer : ℤ := division_result - 160

-- Theorem statement
theorem final_answer_is_correct : final_answer = 12 := by
  sorry

end NUMINAMATH_GPT_final_answer_is_correct_l1907_190791


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l1907_190765

def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}
def R : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = R := by
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l1907_190765


namespace NUMINAMATH_GPT_Cherie_boxes_l1907_190707

theorem Cherie_boxes (x : ℕ) :
  (2 * 8 + x * (8 + 9) = 33) → x = 1 :=
by
  intros h
  have h_eq : 16 + 17 * x = 33 := by simp [mul_add, mul_comm, h]
  linarith

end NUMINAMATH_GPT_Cherie_boxes_l1907_190707


namespace NUMINAMATH_GPT_angles_with_same_terminal_side_l1907_190784

theorem angles_with_same_terminal_side (k : ℤ) : 
  (∃ (α : ℝ), α = -437 + k * 360) ↔ (∃ (β : ℝ), β = 283 + k * 360) := 
by
  sorry

end NUMINAMATH_GPT_angles_with_same_terminal_side_l1907_190784


namespace NUMINAMATH_GPT_proof_equivalent_expression_l1907_190718

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2

theorem proof_equivalent_expression (x y : ℝ) :
  (dollar ((x + y) ^ 2) (dollar y x)) - (dollar (dollar x y) (dollar x y)) = 
  4 * (x + y) ^ 2 * ((x + y) ^ 2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_proof_equivalent_expression_l1907_190718


namespace NUMINAMATH_GPT_thabo_hardcover_books_l1907_190763

theorem thabo_hardcover_books:
  ∃ (H P F : ℕ), H + P + F = 280 ∧ P = H + 20 ∧ F = 2 * P ∧ H = 55 := by
  sorry

end NUMINAMATH_GPT_thabo_hardcover_books_l1907_190763


namespace NUMINAMATH_GPT_Miss_Adamson_paper_usage_l1907_190739

-- Definitions from the conditions
def classes : ℕ := 4
def students_per_class : ℕ := 20
def sheets_per_student : ℕ := 5

-- Total number of students
def total_students : ℕ := classes * students_per_class

-- Total number of sheets of paper
def total_sheets : ℕ := total_students * sheets_per_student

-- The proof problem
theorem Miss_Adamson_paper_usage : total_sheets = 400 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_Miss_Adamson_paper_usage_l1907_190739


namespace NUMINAMATH_GPT_moles_of_CaCO3_formed_l1907_190754

theorem moles_of_CaCO3_formed (m n : ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : ∀ m n : ℕ, (m = n) → (m = 3) → (n = 3) → moles_of_CaCO3 = m) : 
  moles_of_CaCO3 = 3 := by
  sorry

end NUMINAMATH_GPT_moles_of_CaCO3_formed_l1907_190754


namespace NUMINAMATH_GPT_cost_price_per_metre_l1907_190788

theorem cost_price_per_metre (total_metres total_sale total_loss_per_metre total_sell_price : ℕ) (h1: total_metres = 500) (h2: total_sell_price = 15000) (h3: total_loss_per_metre = 10) : total_sell_price + (total_loss_per_metre * total_metres) / total_metres = 40 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_per_metre_l1907_190788


namespace NUMINAMATH_GPT_unique_combination_of_segments_l1907_190743

theorem unique_combination_of_segments :
  ∃! (x y : ℤ), 7 * x + 12 * y = 100 := sorry

end NUMINAMATH_GPT_unique_combination_of_segments_l1907_190743


namespace NUMINAMATH_GPT_find_integer_sets_l1907_190716

noncomputable def satisfy_equation (A B C : ℤ) : Prop :=
  A ^ 2 - B ^ 2 - C ^ 2 = 1 ∧ B + C - A = 3

theorem find_integer_sets :
  { (A, B, C) : ℤ × ℤ × ℤ | satisfy_equation A B C } = {(9, 8, 4), (9, 4, 8), (-3, 2, -2), (-3, -2, 2)} :=
  sorry

end NUMINAMATH_GPT_find_integer_sets_l1907_190716


namespace NUMINAMATH_GPT_find_sachin_age_l1907_190700

variables (S R : ℕ)

def sachin_young_than_rahul_by_4_years (S R : ℕ) : Prop := R = S + 4
def ratio_of_ages (S R : ℕ) : Prop := 7 * R = 9 * S

theorem find_sachin_age (S R : ℕ) (h1 : sachin_young_than_rahul_by_4_years S R) (h2 : ratio_of_ages S R) : S = 14 := 
by sorry

end NUMINAMATH_GPT_find_sachin_age_l1907_190700


namespace NUMINAMATH_GPT_S_nine_l1907_190764

noncomputable def S : ℕ → ℚ
| 3 => 8
| 6 => 10
| _ => 0  -- Placeholder for other values, as we're interested in these specific ones

theorem S_nine (S_3_eq : S 3 = 8) (S_6_eq : S 6 = 10) : S 9 = 21 / 2 :=
by
  -- Construct the proof here
  sorry

end NUMINAMATH_GPT_S_nine_l1907_190764


namespace NUMINAMATH_GPT_elementary_schools_in_Lansing_l1907_190741

theorem elementary_schools_in_Lansing (total_students : ℕ) (students_per_school : ℕ) (h1 : total_students = 6175) (h2 : students_per_school = 247) : total_students / students_per_school = 25 := 
by sorry

end NUMINAMATH_GPT_elementary_schools_in_Lansing_l1907_190741


namespace NUMINAMATH_GPT_compute_expression_l1907_190758

noncomputable def roots_exist (P : Polynomial ℝ) (α β γ : ℝ) : Prop :=
  P = Polynomial.C (-13) + Polynomial.X * (Polynomial.C 11 + Polynomial.X * (Polynomial.C (-7) + Polynomial.X))

theorem compute_expression (α β γ : ℝ) (h : roots_exist (Polynomial.X^3 - 7 * Polynomial.X^2 + 11 * Polynomial.X - 13) α β γ) :
  (α ≠ 0) → (β ≠ 0) → (γ ≠ 0) → (α^2 * β^2 + β^2 * γ^2 + γ^2 * α^2 = -61) :=
  sorry

end NUMINAMATH_GPT_compute_expression_l1907_190758


namespace NUMINAMATH_GPT_sport_formulation_water_content_l1907_190713

theorem sport_formulation_water_content :
  ∀ (f_s c_s w_s : ℕ) (f_p c_p w_p : ℕ),
    f_s / c_s = 1 / 12 →
    f_s / w_s = 1 / 30 →
    f_p / c_p = 1 / 4 →
    f_p / w_p = 1 / 60 →
    c_p = 4 →
    w_p = 60 := by
  sorry

end NUMINAMATH_GPT_sport_formulation_water_content_l1907_190713


namespace NUMINAMATH_GPT_paint_grid_condition_l1907_190761

variables {a b c d e A B C D E : ℕ}

def is_valid (n : ℕ) : Prop := n = 2 ∨ n = 3

theorem paint_grid_condition 
  (ha : is_valid a) (hb : is_valid b) (hc : is_valid c) 
  (hd : is_valid d) (he : is_valid e) (hA : is_valid A) 
  (hB : is_valid B) (hC : is_valid C) (hD : is_valid D) 
  (hE : is_valid E) :
  a + b + c + d + e = A + B + C + D + E :=
sorry

end NUMINAMATH_GPT_paint_grid_condition_l1907_190761


namespace NUMINAMATH_GPT_ball_bounce_height_l1907_190702

theorem ball_bounce_height (b : ℕ) : 
  ∃ b : ℕ, 400 * (3 / 4 : ℝ)^b < 50 ∧ ∀ b' < b, 400 * (3 / 4 : ℝ)^b' ≥ 50 :=
sorry

end NUMINAMATH_GPT_ball_bounce_height_l1907_190702


namespace NUMINAMATH_GPT_fraction_ordering_l1907_190740

theorem fraction_ordering:
  (6 / 22) < (5 / 17) ∧ (5 / 17) < (8 / 24) :=
by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l1907_190740


namespace NUMINAMATH_GPT_steve_matching_pairs_l1907_190792

/-- Steve's total number of socks -/
def total_socks : ℕ := 25

/-- Number of Steve's mismatching socks -/
def mismatching_socks : ℕ := 17

/-- Number of Steve's matching socks -/
def matching_socks : ℕ := total_socks - mismatching_socks

/-- Number of pairs of matching socks Steve has -/
def matching_pairs : ℕ := matching_socks / 2

/-- Proof that Steve has 4 pairs of matching socks -/
theorem steve_matching_pairs : matching_pairs = 4 := by
  sorry

end NUMINAMATH_GPT_steve_matching_pairs_l1907_190792


namespace NUMINAMATH_GPT_parabola_directrix_eq_l1907_190705

noncomputable def equation_of_directrix (p : ℝ) : Prop :=
  (p > 0) ∧ (∀ (x y : ℝ), (x ≠ -5 / 4) → ¬ (y ^ 2 = 2 * p * x))

theorem parabola_directrix_eq (A_x A_y : ℝ) (hA : A_x = 2 ∧ A_y = 1)
  (h_perpendicular_bisector_fo : ∃ (f_x f_y : ℝ), f_x = 5 / 4 ∧ f_y = 0) :
  equation_of_directrix (5 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_directrix_eq_l1907_190705


namespace NUMINAMATH_GPT_gumballs_multiple_purchased_l1907_190777

-- Definitions
def joanna_initial : ℕ := 40
def jacques_initial : ℕ := 60
def final_each : ℕ := 250

-- Proof statement
theorem gumballs_multiple_purchased (m : ℕ) :
  (joanna_initial + joanna_initial * m) + (jacques_initial + jacques_initial * m) = 2 * final_each →
  m = 4 :=
by 
  sorry

end NUMINAMATH_GPT_gumballs_multiple_purchased_l1907_190777


namespace NUMINAMATH_GPT_smallest_common_multiple_five_digit_l1907_190748

def is_multiple (a b : ℕ) : Prop := ∃ k, a = k * b

def smallest_five_digit_multiple_of_3_and_5 (x : ℕ) : Prop :=
  is_multiple x 3 ∧ is_multiple x 5 ∧ 10000 ≤ x ∧ x ≤ 99999 ∧ (∀ y, (10000 ≤ y ∧ y ≤ 99999 ∧ is_multiple y 3 ∧ is_multiple y 5) → x ≤ y)

theorem smallest_common_multiple_five_digit : smallest_five_digit_multiple_of_3_and_5 10005 :=
sorry

end NUMINAMATH_GPT_smallest_common_multiple_five_digit_l1907_190748


namespace NUMINAMATH_GPT_alice_outfits_l1907_190729

theorem alice_outfits :
  let trousers := 5
  let shirts := 8
  let jackets := 4
  let shoes := 2
  trousers * shirts * jackets * shoes = 320 :=
by
  sorry

end NUMINAMATH_GPT_alice_outfits_l1907_190729


namespace NUMINAMATH_GPT_distance_Bella_Galya_l1907_190732

theorem distance_Bella_Galya 
    (AB BV VG GD : ℝ)
    (DB : AB + 3 * BV + 2 * VG + GD = 700)
    (DV : AB + 2 * BV + 2 * VG + GD = 600)
    (DG : AB + 2 * BV + 3 * VG + GD = 650) :
    BV + VG = 150 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_distance_Bella_Galya_l1907_190732


namespace NUMINAMATH_GPT_inequality_condition_l1907_190782

noncomputable def f (a b x : ℝ) : ℝ := Real.exp x - (1 + a) * x - b

theorem inequality_condition (a b: ℝ) (h : ∀ x : ℝ, f a b x ≥ 0) : (b * (a + 1)) / 2 < 3 / 4 := 
sorry

end NUMINAMATH_GPT_inequality_condition_l1907_190782


namespace NUMINAMATH_GPT_sequence_a3_l1907_190780

theorem sequence_a3 (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (recursion : ∀ n, a (n + 1) = a n / (1 + a n)) : 
  a 3 = 1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_a3_l1907_190780


namespace NUMINAMATH_GPT_probability_white_marble_l1907_190774

theorem probability_white_marble :
  ∀ (p_blue p_green p_white : ℝ),
    p_blue = 0.25 →
    p_green = 0.4 →
    p_blue + p_green + p_white = 1 →
    p_white = 0.35 :=
by
  intros p_blue p_green p_white h_blue h_green h_total
  sorry

end NUMINAMATH_GPT_probability_white_marble_l1907_190774


namespace NUMINAMATH_GPT_find_d_values_l1907_190769

theorem find_d_values (u v : ℝ) (c d : ℝ)
  (hpu : u^3 + c * u + d = 0)
  (hpv : v^3 + c * v + d = 0)
  (hqu : (u + 2)^3 + c * (u + 2) + d - 120 = 0)
  (hqv : (v - 5)^3 + c * (v - 5) + d - 120 = 0) :
  d = 396 ∨ d = 8 :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_d_values_l1907_190769


namespace NUMINAMATH_GPT_seating_arrangements_l1907_190727

/--
Prove that the number of ways to seat five people in a row of six chairs is 720.
-/
theorem seating_arrangements (people : ℕ) (chairs : ℕ) (h_people : people = 5) (h_chairs : chairs = 6) :
  ∃ (n : ℕ), n = 720 ∧ n = (6 * 5 * 4 * 3 * 2) :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l1907_190727


namespace NUMINAMATH_GPT_log_diff_lt_one_l1907_190778

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log b) / (Real.log a)

theorem log_diff_lt_one
  (b c x : ℝ)
  (h_eq_sym : ∀ (t : ℝ), (t - 2)^2 + b * (t - 2) + c = (t + 2)^2 + b * (t + 2) + c)
  (h_f_zero_pos : (0)^2 + b * (0) + c > 0)
  (m n : ℝ)
  (h_fm_0 : m^2 + b * m + c = 0)
  (h_fn_0 : n^2 + b * n + c = 0)
  (h_m_ne_n : m ≠ n)
  : log_base 4 m - log_base (1/4) n < 1 :=
  sorry

end NUMINAMATH_GPT_log_diff_lt_one_l1907_190778


namespace NUMINAMATH_GPT_min_value_of_expression_l1907_190726

theorem min_value_of_expression (x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x + y + z = 5) : 
  (9 / x + 16 / y + 25 / z) ≥ 28.8 :=
by sorry

end NUMINAMATH_GPT_min_value_of_expression_l1907_190726


namespace NUMINAMATH_GPT_problem_proof_l1907_190755

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + (1 / Real.sqrt (2 - x))
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {y | y ≥ 1}
def CU_B : Set ℝ := {y | y < 1}
def U : Set ℝ := Set.univ

theorem problem_proof :
  (∀ x, x ∈ A ↔ -1 ≤ x ∧ x < 2) ∧
  (∀ y, y ∈ B ↔ y ≥ 1) ∧
  (A ∩ CU_B = {x | -1 ≤ x ∧ x < 1}) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1907_190755


namespace NUMINAMATH_GPT_find_roots_of_polynomial_l1907_190768

noncomputable def polynomial := Polynomial ℝ

theorem find_roots_of_polynomial :
  (∃ (x : ℝ), x^3 + 3 * x^2 - 6 * x - 8 = 0) ↔ (x = -1 ∨ x = 2 ∨ x = -4) :=
sorry

end NUMINAMATH_GPT_find_roots_of_polynomial_l1907_190768


namespace NUMINAMATH_GPT_ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l1907_190742

theorem ellipse_foci_on_x_axis_major_axis_twice_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m * y^2 = 1) → (∃ a b : ℝ, a = 1 ∧ b = Real.sqrt (1 / m) ∧ a = 2 * b) → m = 4 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_on_x_axis_major_axis_twice_minor_axis_l1907_190742


namespace NUMINAMATH_GPT_trapezoid_area_l1907_190747

-- Definitions of the problem's conditions
def a : ℕ := 4
def b : ℕ := 8
def h : ℕ := 3

-- Lean statement to prove the area of the trapezoid is 18 square centimeters
theorem trapezoid_area : (a + b) * h / 2 = 18 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1907_190747


namespace NUMINAMATH_GPT_no_pos_int_squares_l1907_190722

open Nat

theorem no_pos_int_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  ¬(∃ k m : ℕ, k ^ 2 = a ^ 2 + b ∧ m ^ 2 = b ^ 2 + a) :=
sorry

end NUMINAMATH_GPT_no_pos_int_squares_l1907_190722


namespace NUMINAMATH_GPT_savings_relationship_l1907_190714

def combined_salary : ℝ := 3000
def salary_A : ℝ := 2250
def salary_B : ℝ := combined_salary - salary_A
def savings_A : ℝ := 0.05 * salary_A
def savings_B : ℝ := 0.15 * salary_B

theorem savings_relationship : savings_A = 112.5 ∧ savings_B = 112.5 := by
  have h1 : salary_B = 750 := by sorry
  have h2 : savings_A = 0.05 * 2250 := by sorry
  have h3 : savings_B = 0.15 * 750 := by sorry
  have h4 : savings_A = 112.5 := by sorry
  have h5 : savings_B = 112.5 := by sorry
  exact And.intro h4 h5

end NUMINAMATH_GPT_savings_relationship_l1907_190714


namespace NUMINAMATH_GPT_common_non_integer_root_eq_l1907_190704

theorem common_non_integer_root_eq (p1 p2 q1 q2 : ℤ) 
  (x : ℝ) (hx1 : x^2 + p1 * x + q1 = 0) (hx2 : x^2 + p2 * x + q2 = 0) 
  (hnint : ¬ ∃ (n : ℤ), x = n) : p1 = p2 ∧ q1 = q2 :=
sorry

end NUMINAMATH_GPT_common_non_integer_root_eq_l1907_190704


namespace NUMINAMATH_GPT_square_field_area_l1907_190781

theorem square_field_area (s A : ℝ) (h1 : 10 * 4 * s = 9280) (h2 : A = s^2) : A = 53824 :=
by {
  sorry -- The proof goes here
}

end NUMINAMATH_GPT_square_field_area_l1907_190781


namespace NUMINAMATH_GPT_sqrt_40000_eq_200_l1907_190783

theorem sqrt_40000_eq_200 : Real.sqrt 40000 = 200 := 
sorry

end NUMINAMATH_GPT_sqrt_40000_eq_200_l1907_190783


namespace NUMINAMATH_GPT_sum_of_numbers_l1907_190731

theorem sum_of_numbers (x : ℝ) 
  (h_ratio : ∃ x, (2 * x) / x = 2 ∧ (3 * x) / x = 3)
  (h_squares : x^2 + (2 * x)^2 + (3 * x)^2 = 2744) :
  x + 2 * x + 3 * x = 84 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1907_190731


namespace NUMINAMATH_GPT_shortest_chord_line_l1907_190759

theorem shortest_chord_line (x y : ℝ) (P : (ℝ × ℝ)) (C : ℝ → ℝ → Prop) (h₁ : C x y) (hx : P = (1, 1)) (hC : ∀ x y, C x y ↔ x^2 + y^2 = 4) : 
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -2 ∧ a * x + b * y + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_shortest_chord_line_l1907_190759


namespace NUMINAMATH_GPT_smallest_k_for_a_l1907_190723

theorem smallest_k_for_a (a n : ℕ) (h : 10 ^ 2013 ≤ a^n ∧ a^n < 10 ^ 2014) : ∀ k : ℕ, k < 46 → ∃ n : ℕ, (10 ^ (k - 1)) ≤ a ∧ a < 10 ^ k :=
by sorry

end NUMINAMATH_GPT_smallest_k_for_a_l1907_190723


namespace NUMINAMATH_GPT_f_2015_value_l1907_190728

noncomputable def f : ℝ → ℝ := sorry -- Define f with appropriate conditions

theorem f_2015_value :
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) →
  f 2015 = -2 :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_f_2015_value_l1907_190728


namespace NUMINAMATH_GPT_smallest_m_integral_roots_l1907_190789

theorem smallest_m_integral_roots (m : ℕ) : 
  (∃ p q : ℤ, (10 * p * p - ↑m * p + 360 = 0) ∧ (p + q = m / 10) ∧ (p * q = 36) ∧ (p % q = 0 ∨ q % p = 0)) → 
  m = 120 :=
by
sorry

end NUMINAMATH_GPT_smallest_m_integral_roots_l1907_190789


namespace NUMINAMATH_GPT_hardest_vs_least_worked_hours_difference_l1907_190745

-- Let x be the scaling factor for the ratio
-- The times worked are 2x, 3x, and 4x

def project_time_difference (x : ℕ) : Prop :=
  let time1 := 2 * x
  let time2 := 3 * x
  let time3 := 4 * x
  (time1 + time2 + time3 = 90) ∧ ((4 * x - 2 * x) = 20)

theorem hardest_vs_least_worked_hours_difference :
  ∃ x : ℕ, project_time_difference x :=
by
  sorry

end NUMINAMATH_GPT_hardest_vs_least_worked_hours_difference_l1907_190745


namespace NUMINAMATH_GPT_multiplication_24_12_l1907_190717

theorem multiplication_24_12 :
  let a := 24
  let b := 12
  let b1 := 10
  let b2 := 2
  let p1 := a * b2
  let p2 := a * b1
  let sum := p1 + p2
  b = b1 + b2 →
  p1 = a * b2 →
  p2 = a * b1 →
  sum = p1 + p2 →
  a * b = sum :=
by
  intros
  sorry

end NUMINAMATH_GPT_multiplication_24_12_l1907_190717


namespace NUMINAMATH_GPT_weight_difference_l1907_190738

theorem weight_difference (W_A W_B W_C W_D W_E : ℝ)
  (h_avg_ABC : (W_A + W_B + W_C) / 3 = 80)
  (h_WA : W_A = 95)
  (h_avg_ABCD : (W_A + W_B + W_C + W_D) / 4 = 82)
  (h_avg_BCDE : (W_B + W_C + W_D + W_E) / 4 = 81) :
  W_E - W_D = 3 :=
by
  sorry

end NUMINAMATH_GPT_weight_difference_l1907_190738


namespace NUMINAMATH_GPT_shipping_cost_l1907_190753

def total_weight : ℝ := 540
def weight_per_crate : ℝ := 30
def cost_per_crate : ℝ := 1.5

/-- Lizzy's total shipping cost for 540 pounds of fish packed in 30-pound crates at $1.5 per crate is $27. -/
theorem shipping_cost : (total_weight / weight_per_crate) * cost_per_crate = 27 := by
  sorry

end NUMINAMATH_GPT_shipping_cost_l1907_190753


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1907_190708

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 3 + a 4 + a 5 + a 6 + a 7 = 250) : a 2 + a 8 = 100 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1907_190708


namespace NUMINAMATH_GPT_matthew_ate_8_l1907_190776

variable (M P A K : ℕ)

def kimberly_ate_5 : Prop := K = 5
def alvin_eggs : Prop := A = 2 * K - 1
def patrick_eggs : Prop := P = A / 2
def matthew_eggs : Prop := M = 2 * P

theorem matthew_ate_8 (M P A K : ℕ) (h1 : kimberly_ate_5 K) (h2 : alvin_eggs A K) (h3 : patrick_eggs P A) (h4 : matthew_eggs M P) : M = 8 := by
  sorry

end NUMINAMATH_GPT_matthew_ate_8_l1907_190776


namespace NUMINAMATH_GPT_length_of_ab_l1907_190779

variable (a b c d e : ℝ)
variable (bc cd de ac ae ab : ℝ)

axiom bc_eq_3cd : bc = 3 * cd
axiom de_eq_7 : de = 7
axiom ac_eq_11 : ac = 11
axiom ae_eq_20 : ae = 20
axiom ac_def : ac = ab + bc -- Definition of ac
axiom ae_def : ae = ab + bc + cd + de -- Definition of ae

theorem length_of_ab : ab = 5 := by
  sorry

end NUMINAMATH_GPT_length_of_ab_l1907_190779


namespace NUMINAMATH_GPT_solve_inequality_l1907_190701

theorem solve_inequality (x : ℝ) : -7/3 < x ∧ x < 7 → |x+2| + |x-2| < x + 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_inequality_l1907_190701


namespace NUMINAMATH_GPT_joy_remaining_tape_l1907_190796

theorem joy_remaining_tape (total_tape length width : ℕ) (h_total_tape : total_tape = 250) (h_length : length = 60) (h_width : width = 20) :
  total_tape - 2 * (length + width) = 90 :=
by
  sorry

end NUMINAMATH_GPT_joy_remaining_tape_l1907_190796


namespace NUMINAMATH_GPT_double_counted_page_number_l1907_190736

theorem double_counted_page_number (n x : ℕ) 
  (h1: 1 ≤ x ∧ x ≤ n)
  (h2: (n * (n + 1) / 2) + x = 1997) : 
  x = 44 := 
by
  sorry

end NUMINAMATH_GPT_double_counted_page_number_l1907_190736


namespace NUMINAMATH_GPT_family_ages_l1907_190724

theorem family_ages:
  (∀ (Peter Harriet Jane Emily father: ℕ),
  ((Peter + 12 = 2 * (Harriet + 12)) ∧
   (Jane = Emily + 10) ∧
   (Peter = 60 / 3) ∧
   (Peter = Jane + 5) ∧
   (Aunt_Lucy = 52) ∧
   (Aunt_Lucy = 4 + Peter_Jane_mother) ∧
   (father - 20 = Aunt_Lucy)) →
  (Harriet = 4) ∧ (Peter = 20) ∧ (Jane = 15) ∧ (Emily = 5) ∧ (father = 72)) :=
sorry

end NUMINAMATH_GPT_family_ages_l1907_190724


namespace NUMINAMATH_GPT_sin_gt_cos_interval_l1907_190749

theorem sin_gt_cos_interval (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x > Real.cos x) : 
  Real.sin x > Real.cos x ↔ (Real.pi / 4 < x ∧ x < 5 * Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_sin_gt_cos_interval_l1907_190749


namespace NUMINAMATH_GPT_smallest_five_digit_multiple_of_18_correct_l1907_190766

def smallest_five_digit_multiple_of_18 : ℕ := 10008

theorem smallest_five_digit_multiple_of_18_correct :
  (smallest_five_digit_multiple_of_18 >= 10000) ∧ 
  (smallest_five_digit_multiple_of_18 < 100000) ∧ 
  (smallest_five_digit_multiple_of_18 % 18 = 0) :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_multiple_of_18_correct_l1907_190766
