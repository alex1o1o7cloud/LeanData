import Mathlib

namespace NUMINAMATH_GPT_no_real_roots_iff_l1241_124161

theorem no_real_roots_iff (k : ℝ) : (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_iff_l1241_124161


namespace NUMINAMATH_GPT_beer_drawing_time_l1241_124122

theorem beer_drawing_time :
  let rate_A := 1 / 5
  let rate_C := 1 / 4
  let combined_rate := 9 / 20
  let extra_beer := 12
  let total_drawn := 48
  let t := total_drawn / combined_rate
  t = 48 * 20 / 9 :=
by {
  sorry -- proof not required
}

end NUMINAMATH_GPT_beer_drawing_time_l1241_124122


namespace NUMINAMATH_GPT_conic_eccentricity_l1241_124184

theorem conic_eccentricity (m : ℝ) (h : 0 < -m) (h2 : (Real.sqrt (1 + (-1 / m))) = 2) : m = -1/3 := 
by
  -- Proof can be added here
  sorry

end NUMINAMATH_GPT_conic_eccentricity_l1241_124184


namespace NUMINAMATH_GPT_roots_reciprocal_sum_l1241_124106

theorem roots_reciprocal_sum
  (a b c : ℂ)
  (h : Polynomial.roots (Polynomial.C 1 + Polynomial.X - Polynomial.C 1 * Polynomial.X^2 + Polynomial.C 1 * Polynomial.X^3) = {a, b, c}) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 :=
by
  sorry

end NUMINAMATH_GPT_roots_reciprocal_sum_l1241_124106


namespace NUMINAMATH_GPT_linear_function_increasing_l1241_124101

theorem linear_function_increasing (x1 x2 y1 y2 : ℝ) (h1 : y1 = 2 * x1 - 1) (h2 : y2 = 2 * x2 - 1) (h3 : x1 > x2) : y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_increasing_l1241_124101


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l1241_124160

theorem average_gas_mileage_round_trip :
  (300 / ((150 / 28) + (150 / 18))) = 22 := by
sorry

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l1241_124160


namespace NUMINAMATH_GPT_more_crayons_given_to_Lea_than_Mae_l1241_124132

-- Define the initial number of crayons
def initial_crayons : ℕ := 4 * 8

-- Condition: Nori gave 5 crayons to Mae
def crayons_given_to_Mae : ℕ := 5

-- Condition: Nori has 15 crayons left after giving some to Lea
def crayons_left_after_giving_to_Lea : ℕ := 15

-- Define the number of crayons after giving to Mae
def crayons_after_giving_to_Mae : ℕ := initial_crayons - crayons_given_to_Mae

-- Define the number of crayons given to Lea
def crayons_given_to_Lea : ℕ := crayons_after_giving_to_Mae - crayons_left_after_giving_to_Lea

-- Prove the number of more crayons given to Lea than Mae
theorem more_crayons_given_to_Lea_than_Mae : (crayons_given_to_Lea - crayons_given_to_Mae) = 7 := by
  sorry

end NUMINAMATH_GPT_more_crayons_given_to_Lea_than_Mae_l1241_124132


namespace NUMINAMATH_GPT_units_digit_of_square_ne_2_l1241_124142

theorem units_digit_of_square_ne_2 (n : ℕ) : (n * n) % 10 ≠ 2 :=
sorry

end NUMINAMATH_GPT_units_digit_of_square_ne_2_l1241_124142


namespace NUMINAMATH_GPT_expected_winnings_is_350_l1241_124192

noncomputable def expected_winnings : ℝ :=
  (1 / 8) * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0)

theorem expected_winnings_is_350 :
  expected_winnings = 3.5 :=
by sorry

end NUMINAMATH_GPT_expected_winnings_is_350_l1241_124192


namespace NUMINAMATH_GPT_coordinates_provided_l1241_124133

-- Define the coordinates of point P in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the point P with its given coordinates
def P : Point := {x := 3, y := -5}

-- Lean 4 statement for the proof problem
theorem coordinates_provided : (P.x, P.y) = (3, -5) := by
  -- Proof not provided
  sorry

end NUMINAMATH_GPT_coordinates_provided_l1241_124133


namespace NUMINAMATH_GPT_find_a_b_l1241_124113

theorem find_a_b :
  ∃ a b : ℝ, 
    (a = -4) ∧ (b = -9) ∧
    (∀ x : ℝ, |8 * x + 9| < 7 ↔ a * x^2 + b * x - 2 > 0) := 
sorry

end NUMINAMATH_GPT_find_a_b_l1241_124113


namespace NUMINAMATH_GPT_walking_ring_width_l1241_124131

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) :
  r₁ - r₂ = 10 :=
by
  sorry

end NUMINAMATH_GPT_walking_ring_width_l1241_124131


namespace NUMINAMATH_GPT_find_x_minus_y_l1241_124188

def rotated_point (x y h k : ℝ) : ℝ × ℝ := (2 * h - x, 2 * k - y)

def reflected_point (x y : ℝ) : ℝ × ℝ := (y, x)

def transformed_point (x y : ℝ) : ℝ × ℝ :=
  reflected_point (rotated_point x y 2 3).1 (rotated_point x y 2 3).2

theorem find_x_minus_y (x y : ℝ) (h1 : transformed_point x y = (4, -1)) : x - y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l1241_124188


namespace NUMINAMATH_GPT_sum_sequence_correct_l1241_124164

def sequence_term (n : ℕ) : ℕ :=
  if n % 9 = 0 ∧ n % 32 = 0 then 7
  else if n % 7 = 0 ∧ n % 32 = 0 then 9
  else if n % 7 = 0 ∧ n % 9 = 0 then 32
  else 0

def sequence_sum (up_to : ℕ) : ℕ :=
  (Finset.range (up_to + 1)).sum sequence_term

theorem sum_sequence_correct : sequence_sum 2015 = 1106 := by
  sorry

end NUMINAMATH_GPT_sum_sequence_correct_l1241_124164


namespace NUMINAMATH_GPT_calc_miscellaneous_collective_expenses_l1241_124138

def individual_needed_amount : ℕ := 450
def additional_needed_amount : ℕ := 475
def total_students : ℕ := 6
def first_day_amount : ℕ := 600
def second_day_amount : ℕ := 900
def third_day_amount : ℕ := 400
def days : ℕ := 4

def total_individual_goal : ℕ := individual_needed_amount + additional_needed_amount
def total_students_goal : ℕ := total_individual_goal * total_students
def total_first_3_days : ℕ := first_day_amount + second_day_amount + third_day_amount
def total_next_4_days : ℕ := (total_first_3_days / 2) * days
def total_raised : ℕ := total_first_3_days + total_next_4_days

def miscellaneous_collective_expenses : ℕ := total_raised - total_students_goal

theorem calc_miscellaneous_collective_expenses : miscellaneous_collective_expenses = 150 := by
  sorry

end NUMINAMATH_GPT_calc_miscellaneous_collective_expenses_l1241_124138


namespace NUMINAMATH_GPT_count_possible_P_l1241_124109

-- Define the distinct digits with initial conditions
def digits : Type := {n // n ≥ 0 ∧ n ≤ 9}

-- Define the parameters P, Q, R, S as distinct digits
variables (P Q R S : digits)

-- Define the condition that P, Q, R, S are distinct.
def distinct (P Q R S : digits) : Prop := 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S

-- Assertion conditions based on a valid subtraction layout
def valid_subtraction (P Q R S : digits) : Prop :=
  Q.val - P.val = S.val ∧ (P.val - R.val = P.val) ∧ (P.val - Q.val = S.val)

-- Prove that there are exactly 9 possible values for P.
theorem count_possible_P : ∃ n : ℕ, n = 9 ∧ ∀ P Q R S : digits, distinct P Q R S → valid_subtraction P Q R S → n = 9 :=
by sorry

end NUMINAMATH_GPT_count_possible_P_l1241_124109


namespace NUMINAMATH_GPT_length_PQ_l1241_124123

theorem length_PQ (AB BC CA AH : ℝ) (P Q : ℝ) : 
  AB = 7 → BC = 8 → CA = 9 → 
  AH = 3 * Real.sqrt 5 → 
  PQ = AQ - AP → 
  AQ = 7 * (Real.sqrt 5) / 3 → 
  AP = 9 * (Real.sqrt 5) / 5 → 
  PQ = Real.sqrt 5 * 8 / 15 :=
by
  intros hAB hBC hCA hAH hPQ hAQ hAP
  sorry

end NUMINAMATH_GPT_length_PQ_l1241_124123


namespace NUMINAMATH_GPT_num_values_x_satisfying_l1241_124127

theorem num_values_x_satisfying (
  f : ℝ → ℝ → ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (x : ℝ)
  (h_eq : ∀ x, f (cos x) (sin x) = 2 ↔ (cos x) ^ 2 + 3 * (sin x) ^ 2 = 2)
  (h_interval : ∀ x, -20 < x ∧ x < 90)
  (h_cos_sin : ∀ x, cos x = cos (x) ∧ sin x = sin (x)) :
  ∃ n, n = 70 := sorry

end NUMINAMATH_GPT_num_values_x_satisfying_l1241_124127


namespace NUMINAMATH_GPT_no_solution_to_inequalities_l1241_124194

theorem no_solution_to_inequalities : 
  ∀ x : ℝ, ¬ (4 * x - 3 < (x + 2)^2 ∧ (x + 2)^2 < 8 * x - 5) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_to_inequalities_l1241_124194


namespace NUMINAMATH_GPT_robert_time_to_complete_l1241_124120

noncomputable def time_to_complete_semicircle_path (length_mile : ℝ) (width_feet : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
  let diameter_mile := width_feet / mile_to_feet
  let radius_mile := diameter_mile / 2
  let circumference_mile := 2 * Real.pi * radius_mile
  let semicircle_length_mile := circumference_mile / 2
  semicircle_length_mile / speed_mph

theorem robert_time_to_complete :
  time_to_complete_semicircle_path 1 40 5 5280 = Real.pi / 10 :=
by
  sorry

end NUMINAMATH_GPT_robert_time_to_complete_l1241_124120


namespace NUMINAMATH_GPT_avg_weight_section_b_l1241_124128

/-- Definition of the average weight of section B based on given conditions --/
theorem avg_weight_section_b :
  let W_A := 50
  let W_class := 54.285714285714285
  let num_A := 40
  let num_B := 30
  let total_class_weight := (num_A + num_B) * W_class
  let total_A_weight := num_A * W_A
  let total_B_weight := total_class_weight - total_A_weight
  let W_B := total_B_weight / num_B
  W_B = 60 :=
by
  sorry

end NUMINAMATH_GPT_avg_weight_section_b_l1241_124128


namespace NUMINAMATH_GPT_arrange_in_circle_l1241_124104

open Nat

noncomputable def smallest_n := 70

theorem arrange_in_circle (n : ℕ) (h : n = 70) :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n →
    (∀ j : ℕ, 1 ≤ j ∧ j ≤ 40 → k > ((k + j) % n)) ∨
    (∀ p : ℕ, 1 ≤ p ∧ p ≤ 30 → k < ((k + p) % n))) :=
by
  sorry

end NUMINAMATH_GPT_arrange_in_circle_l1241_124104


namespace NUMINAMATH_GPT_rational_numbers_on_circle_l1241_124182

theorem rational_numbers_on_circle (a b c d e f : ℚ)
  (h1 : a = |b - c|)
  (h2 : b = d)
  (h3 : c = |d - e|)
  (h4 : d = |e - f|)
  (h5 : e = f)
  (h6 : a + b + c + d + e + f = 1) :
  [a, b, c, d, e, f] = [1/4, 1/4, 0, 1/4, 1/4, 0] :=
sorry

end NUMINAMATH_GPT_rational_numbers_on_circle_l1241_124182


namespace NUMINAMATH_GPT_find_x_l1241_124171

theorem find_x (x : ℚ) (h : (3 + 1 / (2 + 1 / (3 + 3 / (4 + x)))) = 225 / 68) : 
  x = -50 / 19 := 
sorry

end NUMINAMATH_GPT_find_x_l1241_124171


namespace NUMINAMATH_GPT_exists_integer_lt_sqrt_10_l1241_124180

theorem exists_integer_lt_sqrt_10 : ∃ k : ℤ, k < Real.sqrt 10 := by
  have h_sqrt_bounds : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := by
    -- Proof involving basic properties and calculations
    sorry
  exact ⟨3, h_sqrt_bounds.left⟩

end NUMINAMATH_GPT_exists_integer_lt_sqrt_10_l1241_124180


namespace NUMINAMATH_GPT_km_to_m_is_750_l1241_124126

-- Define 1 kilometer equals 5 hectometers
def km_to_hm := 5

-- Define 1 hectometer equals 10 dekameters
def hm_to_dam := 10

-- Define 1 dekameter equals 15 meters
def dam_to_m := 15

-- Theorem stating that the number of meters in one kilometer is 750
theorem km_to_m_is_750 : 1 * km_to_hm * hm_to_dam * dam_to_m = 750 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_km_to_m_is_750_l1241_124126


namespace NUMINAMATH_GPT_sin_double_angle_pi_six_l1241_124196

theorem sin_double_angle_pi_six (α : ℝ)
  (h : 2 * Real.sin α = 1 + 2 * Real.sqrt 3 * Real.cos α) :
  Real.sin (2 * α - Real.pi / 6) = 7 / 8 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_pi_six_l1241_124196


namespace NUMINAMATH_GPT_max_gcd_13n_plus_4_8n_plus_3_l1241_124165

theorem max_gcd_13n_plus_4_8n_plus_3 : 
  ∀ n : ℕ, n > 0 → ∃ d : ℕ, d = 7 ∧ ∀ k : ℕ, k = gcd (13 * n + 4) (8 * n + 3) → k ≤ d :=
by
  sorry

end NUMINAMATH_GPT_max_gcd_13n_plus_4_8n_plus_3_l1241_124165


namespace NUMINAMATH_GPT_day_of_week_299th_day_2004_l1241_124119

noncomputable def day_of_week (day: ℕ): ℕ := day % 7

theorem day_of_week_299th_day_2004 : 
  ∀ (d: ℕ), day_of_week d = 3 → d = 45 → day_of_week 299 = 5 :=
by
  sorry

end NUMINAMATH_GPT_day_of_week_299th_day_2004_l1241_124119


namespace NUMINAMATH_GPT_pow_neg_one_diff_l1241_124181

theorem pow_neg_one_diff (n : ℤ) (h1 : n = 2010) (h2 : n + 1 = 2011) :
  (-1)^2010 - (-1)^2011 = 2 := 
by
  sorry

end NUMINAMATH_GPT_pow_neg_one_diff_l1241_124181


namespace NUMINAMATH_GPT_casey_stays_for_n_months_l1241_124144

-- Definitions based on conditions.
def weekly_cost : ℕ := 280
def monthly_cost : ℕ := 1000
def weeks_per_month : ℕ := 4
def total_savings : ℕ := 360

-- Calculate monthly cost when paying weekly.
def monthly_cost_weekly := weekly_cost * weeks_per_month

-- Calculate savings per month when paying monthly instead of weekly.
def savings_per_month := monthly_cost_weekly - monthly_cost

-- Define the problem statement.
theorem casey_stays_for_n_months :
  (total_savings / savings_per_month) = 3 := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_casey_stays_for_n_months_l1241_124144


namespace NUMINAMATH_GPT_trapezoid_problem_l1241_124103

theorem trapezoid_problem (b h x : ℝ) 
  (hb : b > 0)
  (hh : h > 0)
  (h_ratio : (b + 90) / (b + 30) = 3 / 4)
  (h_x_def : x = 150 * (h / (x - 90) - 90))
  (hx2 : x^2 = 26100) :
  ⌊x^2 / 120⌋ = 217 := sorry

end NUMINAMATH_GPT_trapezoid_problem_l1241_124103


namespace NUMINAMATH_GPT_greatest_possible_third_side_l1241_124168

theorem greatest_possible_third_side (t : ℕ) (h : 5 < t ∧ t < 15) : t = 14 :=
sorry

end NUMINAMATH_GPT_greatest_possible_third_side_l1241_124168


namespace NUMINAMATH_GPT_inequality_proof_l1241_124136

variable {A B C a b c r : ℝ}

theorem inequality_proof (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hr : 0 < r) :
  (A + a + B + b) / (A + a + B + b + c + r) + (B + b + C + c) / (B + b + C + c + a + r) > (C + c + A + a) / (C + c + A + a + b + r) := 
    sorry

end NUMINAMATH_GPT_inequality_proof_l1241_124136


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1241_124116

theorem solve_eq1 (x : ℝ) : (3 * x + 2) ^ 2 = 25 ↔ (x = 1 ∨ x = -7 / 3) := by
  sorry

theorem solve_eq2 (x : ℝ) : 3 * x ^ 2 - 1 = 4 * x ↔ (x = (2 + Real.sqrt 7) / 3 ∨ x = (2 - Real.sqrt 7) / 3) := by
  sorry

theorem solve_eq3 (x : ℝ) : (2 * x - 1) ^ 2 = 3 * (2 * x + 1) ↔ (x = -1 / 2 ∨ x = 1) := by
  sorry

theorem solve_eq4 (x : ℝ) : x ^ 2 - 7 * x + 10 = 0 ↔ (x = 5 ∨ x = 2) := by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1241_124116


namespace NUMINAMATH_GPT_distance_from_yz_plane_l1241_124114

theorem distance_from_yz_plane (x z : ℝ) : 
  (abs (-6) = (abs x) / 2) → abs x = 12 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_yz_plane_l1241_124114


namespace NUMINAMATH_GPT_max_AB_CD_value_l1241_124158

def is_digit (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 9

noncomputable def max_AB_CD : ℕ :=
  let A := 9
  let B := 8
  let C := 7
  let D := 6
  (A + B) + (C + D)

theorem max_AB_CD_value :
  ∀ (A B C D : ℕ), 
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (A + B) + (C + D) ≤ max_AB_CD :=
by
  sorry

end NUMINAMATH_GPT_max_AB_CD_value_l1241_124158


namespace NUMINAMATH_GPT_find_constants_l1241_124147

noncomputable def f (x : ℕ) (a c : ℕ) : ℝ :=
  if x < a then c / Real.sqrt x else c / Real.sqrt a

theorem find_constants (a c : ℕ) (h₁ : f 4 a c = 30) (h₂ : f a a c = 5) : 
  c = 60 ∧ a = 144 := 
by
  sorry

end NUMINAMATH_GPT_find_constants_l1241_124147


namespace NUMINAMATH_GPT_line_equation_l1241_124175

-- Definitions according to the conditions
def point_P := (3, 4)
def slope_angle_l := 90

-- Statement of the theorem to prove
theorem line_equation (l : ℝ → ℝ) (h1 : l point_P.1 = point_P.2) (h2 : slope_angle_l = 90) :
  ∃ k : ℝ, k = 3 ∧ ∀ x, l x = 3 - x :=
sorry

end NUMINAMATH_GPT_line_equation_l1241_124175


namespace NUMINAMATH_GPT_johns_haircut_tip_percentage_l1241_124163

noncomputable def percent_of_tip (annual_spending : ℝ) (haircut_cost : ℝ) (haircut_frequency : ℕ) : ℝ := 
  ((annual_spending / haircut_frequency - haircut_cost) / haircut_cost) * 100

theorem johns_haircut_tip_percentage : 
  let hair_growth_rate : ℝ := 1.5
  let initial_length : ℝ := 6
  let max_length : ℝ := 9
  let haircut_cost : ℝ := 45
  let annual_spending : ℝ := 324
  let months_in_year : ℕ := 12
  let growth_period := 2 -- months it takes for hair to grow 3 inches
  let haircuts_per_year := months_in_year / growth_period -- number of haircuts per year
  percent_of_tip annual_spending haircut_cost haircuts_per_year = 20 := by
  sorry

end NUMINAMATH_GPT_johns_haircut_tip_percentage_l1241_124163


namespace NUMINAMATH_GPT_students_in_class_l1241_124124

theorem students_in_class
  (S : ℕ)
  (h1 : S / 3 * 4 / 3 = 12) :
  S = 36 := 
sorry

end NUMINAMATH_GPT_students_in_class_l1241_124124


namespace NUMINAMATH_GPT_solve_MQ_above_A_l1241_124162

-- Definitions of the given conditions
def ABCD_side := 8
def MNPQ_length := 16
def MNPQ_width := 8
def area_outer_inner_ratio := 1 / 3

-- Definition to prove
def length_MQ_above_A := 8 / 3

-- The area calculations
def area_MNPQ := MNPQ_length * MNPQ_width
def area_ABCD := ABCD_side * ABCD_side
def area_outer := (area_outer_inner_ratio * area_MNPQ)
def MQ_above_A_calculated := area_outer / MNPQ_length

theorem solve_MQ_above_A :
  MQ_above_A_calculated = length_MQ_above_A := by sorry

end NUMINAMATH_GPT_solve_MQ_above_A_l1241_124162


namespace NUMINAMATH_GPT_remainder_when_divided_by_8_l1241_124102

theorem remainder_when_divided_by_8 :
  (481207 % 8) = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_8_l1241_124102


namespace NUMINAMATH_GPT_union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l1241_124172

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition definitions
def set_A : Set ℝ := {x | x ≤ 3 ∨ x ≥ 6}
def set_B : Set ℝ := {x | -2 < x ∧ x < 9}
def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Proof statement (1)
theorem union_A_B_eq_univ (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  A ∪ B = Set.univ := by sorry

theorem inter_compl_A_B_eq_interval (A B : Set ℝ) (h₁ : A = set_A) (h₂ : B = set_B) :
  (Set.univ \ A) ∩ B = {x | 3 < x ∧ x < 6} := by sorry

-- Proof statement (2)
theorem subset_B_range_of_a (a : ℝ) (h : set_C a ⊆ set_B) :
  -2 ≤ a ∧ a ≤ 8 := by sorry

end NUMINAMATH_GPT_union_A_B_eq_univ_inter_compl_A_B_eq_interval_subset_B_range_of_a_l1241_124172


namespace NUMINAMATH_GPT_brownies_pieces_l1241_124148

theorem brownies_pieces (pan_length pan_width piece_length piece_width : ℕ)
  (h_pan_dims : pan_length = 15) (h_pan_width : pan_width = 25)
  (h_piece_length : piece_length = 3) (h_piece_width : piece_width = 5) :
  (pan_length * pan_width) / (piece_length * piece_width) = 25 :=
by
  sorry

end NUMINAMATH_GPT_brownies_pieces_l1241_124148


namespace NUMINAMATH_GPT_min_combined_number_of_horses_and_ponies_l1241_124130

theorem min_combined_number_of_horses_and_ponies :
  ∃ P H : ℕ, H = P + 4 ∧ (∃ k : ℕ, k = (3 * P) / 10 ∧ k = 16 * (3 * P) / (16 * 10) ∧ H + P = 36) :=
sorry

end NUMINAMATH_GPT_min_combined_number_of_horses_and_ponies_l1241_124130


namespace NUMINAMATH_GPT_Polly_tweets_l1241_124151

theorem Polly_tweets :
  let HappyTweets := 18 * 50
  let HungryTweets := 4 * 35
  let WatchingReflectionTweets := 45 * 30
  let SadTweets := 6 * 20
  let PlayingWithToysTweets := 25 * 75
  HappyTweets + HungryTweets + WatchingReflectionTweets + SadTweets + PlayingWithToysTweets = 4385 :=
by
  sorry

end NUMINAMATH_GPT_Polly_tweets_l1241_124151


namespace NUMINAMATH_GPT_average_age_increase_l1241_124121

theorem average_age_increase 
    (num_students : ℕ) (avg_age_students : ℕ) (age_staff : ℕ)
    (H1: num_students = 32)
    (H2: avg_age_students = 16)
    (H3: age_staff = 49) : 
    ((num_students * avg_age_students + age_staff) / (num_students + 1) - avg_age_students = 1) :=
by
  sorry

end NUMINAMATH_GPT_average_age_increase_l1241_124121


namespace NUMINAMATH_GPT_polynomial_sequence_finite_functions_l1241_124193

theorem polynomial_sequence_finite_functions (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_sequence_finite_functions_l1241_124193


namespace NUMINAMATH_GPT_find_smallest_in_arithmetic_progression_l1241_124170

theorem find_smallest_in_arithmetic_progression (a d : ℝ)
  (h1 : (a-2*d)^3 + (a-d)^3 + a^3 + (a+d)^3 + (a+2*d)^3 = 0)
  (h2 : (a-2*d)^4 + (a-d)^4 + a^4 + (a+d)^4 + (a+2*d)^4 = 136) :
  (a - 2*d) = -2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_smallest_in_arithmetic_progression_l1241_124170


namespace NUMINAMATH_GPT_number_of_dogs_l1241_124189

variable {C D : ℕ}

def ratio_of_dogs_to_cats (D C : ℕ) : Prop := D = (15/7) * C

def ratio_after_additional_cats (D C : ℕ) : Prop :=
  D = 15 * (C + 8) / 11

theorem number_of_dogs (h1 : ratio_of_dogs_to_cats D C) (h2 : ratio_after_additional_cats D C) :
  D = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_dogs_l1241_124189


namespace NUMINAMATH_GPT_binom_12_9_eq_220_l1241_124152

noncomputable def binom (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_12_9_eq_220 : binom 12 9 = 220 :=
sorry

end NUMINAMATH_GPT_binom_12_9_eq_220_l1241_124152


namespace NUMINAMATH_GPT_purchase_price_l1241_124137

theorem purchase_price (P : ℝ)
  (down_payment : ℝ) (monthly_payment : ℝ) (number_of_payments : ℝ)
  (interest_rate : ℝ) (total_paid : ℝ)
  (h1 : down_payment = 12)
  (h2 : monthly_payment = 10)
  (h3 : number_of_payments = 12)
  (h4 : interest_rate = 0.10714285714285714)
  (h5 : total_paid = 132) :
  P = 132 / 1.1071428571428572 :=
by
  sorry

end NUMINAMATH_GPT_purchase_price_l1241_124137


namespace NUMINAMATH_GPT_calculate_lego_set_cost_l1241_124105

variable (total_revenue_after_tax : ℝ) (little_cars_base_price : ℝ)
  (discount_rate : ℝ) (tax_rate : ℝ) (num_little_cars : ℕ)
  (num_action_figures : ℕ) (num_board_games : ℕ)
  (lego_set_cost_before_tax : ℝ)

theorem calculate_lego_set_cost :
  total_revenue_after_tax = 136.50 →
  little_cars_base_price = 5 →
  discount_rate = 0.10 →
  tax_rate = 0.05 →
  num_little_cars = 3 →
  num_action_figures = 2 →
  num_board_games = 1 →
  lego_set_cost_before_tax = 85 :=
by
  sorry

end NUMINAMATH_GPT_calculate_lego_set_cost_l1241_124105


namespace NUMINAMATH_GPT_garden_area_increase_l1241_124174

-- Define the dimensions and perimeter of the rectangular garden
def length_rect : ℕ := 30
def width_rect : ℕ := 12
def area_rect : ℕ := length_rect * width_rect

def perimeter_rect : ℕ := 2 * (length_rect + width_rect)

-- Define the side length and area of the new square garden
def side_square : ℕ := perimeter_rect / 4
def area_square : ℕ := side_square * side_square

-- Define the increase in area
def increase_in_area : ℕ := area_square - area_rect

-- Prove the increase in area is 81 square feet
theorem garden_area_increase : increase_in_area = 81 := by
  sorry

end NUMINAMATH_GPT_garden_area_increase_l1241_124174


namespace NUMINAMATH_GPT_probability_reaching_five_without_returning_to_zero_l1241_124100

def reach_position_without_return_condition (tosses : ℕ) (target : ℤ) (return_limit : ℤ) : ℕ :=
  -- Ideally we should implement the logic to find the number of valid paths here (as per problem constraints)
  sorry

theorem probability_reaching_five_without_returning_to_zero {a b : ℕ} (h_rel_prime : Nat.gcd a b = 1)
    (h_paths_valid : reach_position_without_return_condition 10 5 3 = 15) :
    a = 15 ∧ b = 256 ∧ a + b = 271 :=
by
  sorry

end NUMINAMATH_GPT_probability_reaching_five_without_returning_to_zero_l1241_124100


namespace NUMINAMATH_GPT_projectile_height_at_time_l1241_124134

theorem projectile_height_at_time
  (y : ℝ)
  (t : ℝ)
  (h_eq : y = -16 * t ^ 2 + 64 * t) :
  ∃ t₀ : ℝ, t₀ = 3 ∧ y = 49 :=
by sorry

end NUMINAMATH_GPT_projectile_height_at_time_l1241_124134


namespace NUMINAMATH_GPT_circle_radius_l1241_124150

theorem circle_radius : ∀ (x y : ℝ), x^2 + 10*x + y^2 - 8*y + 25 = 0 → False := sorry

end NUMINAMATH_GPT_circle_radius_l1241_124150


namespace NUMINAMATH_GPT_pascal_row_10_sum_l1241_124198

-- Definition: sum of the numbers in Row n of Pascal's Triangle is 2^n
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- Theorem: sum of the numbers in Row 10 of Pascal's Triangle is 1024
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  sorry

end NUMINAMATH_GPT_pascal_row_10_sum_l1241_124198


namespace NUMINAMATH_GPT_average_salary_of_all_employees_l1241_124176

theorem average_salary_of_all_employees 
    (avg_salary_officers : ℝ)
    (avg_salary_non_officers : ℝ)
    (num_officers : ℕ)
    (num_non_officers : ℕ)
    (h1 : avg_salary_officers = 450)
    (h2 : avg_salary_non_officers = 110)
    (h3 : num_officers = 15)
    (h4 : num_non_officers = 495) :
    (avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers)
    / (num_officers + num_non_officers) = 120 := by
  sorry

end NUMINAMATH_GPT_average_salary_of_all_employees_l1241_124176


namespace NUMINAMATH_GPT_possible_b4b7_products_l1241_124177

theorem possible_b4b7_products (b : ℕ → ℤ) (d : ℤ)
  (h_arith_sequence : ∀ n, b (n + 1) = b n + d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_product_21 : b 5 * b 6 = 21) :
  b 4 * b 7 = -779 ∨ b 4 * b 7 = 21 :=
by
  sorry

end NUMINAMATH_GPT_possible_b4b7_products_l1241_124177


namespace NUMINAMATH_GPT_cost_price_of_article_l1241_124173

theorem cost_price_of_article (C MP SP : ℝ) (h1 : MP = 62.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) :
  C = 47.5 :=
sorry

end NUMINAMATH_GPT_cost_price_of_article_l1241_124173


namespace NUMINAMATH_GPT_cubic_inequality_l1241_124111

theorem cubic_inequality (p q x : ℝ) (h : x^3 + p * x + q = 0) : 4 * q * x ≤ p^2 :=
sorry

end NUMINAMATH_GPT_cubic_inequality_l1241_124111


namespace NUMINAMATH_GPT_valeries_thank_you_cards_l1241_124155

variables (T R J B : ℕ)

theorem valeries_thank_you_cards :
  B = 2 →
  R = B + 3 →
  J = 2 * R →
  T + (B + 1) + R + J = 21 →
  T = 3 :=
by
  intros hB hR hJ hTotal
  sorry

end NUMINAMATH_GPT_valeries_thank_you_cards_l1241_124155


namespace NUMINAMATH_GPT_evaluate_expression_l1241_124110

theorem evaluate_expression : 
  let expr := (15 / 8) ^ 2
  let ceil_expr := Nat.ceil expr
  let mult_expr := ceil_expr * (21 / 5)
  Nat.floor mult_expr = 16 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1241_124110


namespace NUMINAMATH_GPT_time_to_cross_l1241_124199

noncomputable def length_first_train : ℝ := 210
noncomputable def speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
noncomputable def length_second_train : ℝ := 290.04
noncomputable def speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

noncomputable def relative_speed := speed_first_train + speed_second_train
noncomputable def total_length := length_first_train + length_second_train
noncomputable def crossing_time := total_length / relative_speed

theorem time_to_cross : crossing_time = 9 := by
  let length_first_train : ℝ := 210
  let speed_first_train : ℝ := 120 * 1000 / 3600 -- Convert to m/s
  let length_second_train : ℝ := 290.04
  let speed_second_train : ℝ := 80 * 1000 / 3600 -- Convert to m/s

  let relative_speed := speed_first_train + speed_second_train
  let total_length := length_first_train + length_second_train
  let crossing_time := total_length / relative_speed

  show crossing_time = 9
  sorry

end NUMINAMATH_GPT_time_to_cross_l1241_124199


namespace NUMINAMATH_GPT_max_months_with_5_sundays_l1241_124157

theorem max_months_with_5_sundays (months : ℕ) (days_in_year : ℕ) (extra_sundays : ℕ) :
  months = 12 ∧ (days_in_year = 365 ∨ days_in_year = 366) ∧ extra_sundays = days_in_year % 7
  → ∃ max_months_with_5_sundays, max_months_with_5_sundays = 5 := 
by
  sorry

end NUMINAMATH_GPT_max_months_with_5_sundays_l1241_124157


namespace NUMINAMATH_GPT_knights_wins_33_l1241_124191

def sharks_wins : ℕ := sorry
def falcons_wins : ℕ := sorry
def knights_wins : ℕ := sorry
def wolves_wins : ℕ := sorry
def dragons_wins : ℕ := 38 -- Dragons won the most games

-- Condition 1: The Sharks won more games than the Falcons.
axiom sharks_won_more_than_falcons : sharks_wins > falcons_wins

-- Condition 2: The Knights won more games than the Wolves, but fewer than the Dragons.
axiom knights_won_more_than_wolves : knights_wins > wolves_wins
axiom knights_won_less_than_dragons : knights_wins < dragons_wins

-- Condition 3: The Wolves won more than 22 games.
axiom wolves_won_more_than_22 : wolves_wins > 22

-- The possible wins are 24, 27, 33, 36, and 38 and the dragons win 38 (already accounted in dragons_wins)

-- Prove that the Knights won 33 games.
theorem knights_wins_33 : knights_wins = 33 :=
sorry -- proof goes here

end NUMINAMATH_GPT_knights_wins_33_l1241_124191


namespace NUMINAMATH_GPT_remainder_of_s_minus_t_plus_t_minus_u_l1241_124185

theorem remainder_of_s_minus_t_plus_t_minus_u (s t u : ℕ) (hs : s % 12 = 4) (ht : t % 12 = 5) (hu : u % 12 = 7) (h_order : s > t ∧ t > u) :
  ((s - t) + (t - u)) % 12 = 9 :=
by sorry

end NUMINAMATH_GPT_remainder_of_s_minus_t_plus_t_minus_u_l1241_124185


namespace NUMINAMATH_GPT_cdf_from_pdf_l1241_124178

noncomputable def pdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.cos x
  else 0

noncomputable def cdf (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
  else 1

theorem cdf_from_pdf (x : ℝ) : 
  ∀ x : ℝ, cdf x = 
    if x ≤ 0 then 0
    else if 0 < x ∧ x ≤ Real.pi / 2 then Real.sin x
    else 1 :=
by
  sorry

end NUMINAMATH_GPT_cdf_from_pdf_l1241_124178


namespace NUMINAMATH_GPT_binomial_coeff_sum_l1241_124107

theorem binomial_coeff_sum :
  ∀ a b : ℝ, 15 * a^4 * b^2 = 135 ∧ 6 * a^5 * b = -18 →
  (a + b) ^ 6 = 64 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_binomial_coeff_sum_l1241_124107


namespace NUMINAMATH_GPT_complement_A_correct_l1241_124141

-- Define the universal set U
def U : Set ℝ := { x | x ≥ 1 ∨ x ≤ -1 }

-- Define the set A
def A : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := { x | x ≤ -1 ∨ x = 1 ∨ x > 2 }

-- Prove that the complement of A in U is as defined
theorem complement_A_correct : (U \ A) = complement_A_in_U := by
  sorry

end NUMINAMATH_GPT_complement_A_correct_l1241_124141


namespace NUMINAMATH_GPT_rectangle_length_l1241_124125

-- Define a structure for the rectangle.
structure Rectangle where
  breadth : ℝ
  length : ℝ
  area : ℝ

-- Define the given conditions.
def givenConditions (r : Rectangle) : Prop :=
  r.length = 3 * r.breadth ∧ r.area = 6075

-- State the theorem.
theorem rectangle_length (r : Rectangle) (h : givenConditions r) : r.length = 135 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_length_l1241_124125


namespace NUMINAMATH_GPT_abs_equality_holds_if_interval_l1241_124108

noncomputable def quadratic_abs_equality (x : ℝ) : Prop :=
  |x^2 - 8 * x + 12| = x^2 - 8 * x + 12

theorem abs_equality_holds_if_interval (x : ℝ) :
  quadratic_abs_equality x ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end NUMINAMATH_GPT_abs_equality_holds_if_interval_l1241_124108


namespace NUMINAMATH_GPT_share_per_person_l1241_124187

-- Defining the total cost and number of people
def total_cost : ℝ := 12100
def num_people : ℝ := 11

-- The theorem stating that each person's share is $1,100.00
theorem share_per_person : total_cost / num_people = 1100 := by
  sorry

end NUMINAMATH_GPT_share_per_person_l1241_124187


namespace NUMINAMATH_GPT_no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l1241_124166

theorem no_integer_for_58th_power_64_digits : ¬ ∃ n : ℤ, 10^63 ≤ n^58 ∧ n^58 < 10^64 :=
sorry

theorem valid_replacement_for_64_digits (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 81) : 
  ¬ ∃ n : ℤ, 10^(k-1) ≤ n^58 ∧ n^58 < 10^k :=
sorry

end NUMINAMATH_GPT_no_integer_for_58th_power_64_digits_valid_replacement_for_64_digits_l1241_124166


namespace NUMINAMATH_GPT_halfway_between_l1241_124118

-- Definitions based on given conditions
def a : ℚ := 1 / 7
def b : ℚ := 1 / 9

-- Theorem that needs to be proved
theorem halfway_between (h : True) : (a + b) / 2 = 8 / 63 := by
  sorry

end NUMINAMATH_GPT_halfway_between_l1241_124118


namespace NUMINAMATH_GPT_sqrt_meaningful_range_iff_l1241_124135

noncomputable def sqrt_meaningful_range (x : ℝ) : Prop :=
  (∃ r : ℝ, r ≥ 0 ∧ r * r = x - 2023)

theorem sqrt_meaningful_range_iff {x : ℝ} : sqrt_meaningful_range x ↔ x ≥ 2023 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_iff_l1241_124135


namespace NUMINAMATH_GPT_t_over_s_possible_values_l1241_124169

-- Define the initial conditions
variables (n : ℕ) (h : n ≥ 3)

-- The theorem statement
theorem t_over_s_possible_values (s t : ℕ) (h_s : s > 0) (h_t : t > 0) : 
  (∃ r : ℚ, r = t / s ∧ 1 ≤ r ∧ r < (n - 1)) :=
sorry

end NUMINAMATH_GPT_t_over_s_possible_values_l1241_124169


namespace NUMINAMATH_GPT_solve_for_x_l1241_124156

theorem solve_for_x : 
  ∀ x : ℚ, x + 5/6 = 7/18 - 2/9 → x = -2/3 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l1241_124156


namespace NUMINAMATH_GPT_marissa_tied_boxes_l1241_124139

def Total_ribbon : ℝ := 4.5
def Leftover_ribbon : ℝ := 1
def Ribbon_per_box : ℝ := 0.7

theorem marissa_tied_boxes : (Total_ribbon - Leftover_ribbon) / Ribbon_per_box = 5 := by
  sorry

end NUMINAMATH_GPT_marissa_tied_boxes_l1241_124139


namespace NUMINAMATH_GPT_julia_fascinating_last_digits_l1241_124179

theorem julia_fascinating_last_digits : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, (∃ y : ℕ, x = 10 * y) → x % 10 < 10) :=
by
  sorry

end NUMINAMATH_GPT_julia_fascinating_last_digits_l1241_124179


namespace NUMINAMATH_GPT_knowledge_competition_score_l1241_124146

theorem knowledge_competition_score (x : ℕ) (hx : x ≤ 20) : 5 * x - (20 - x) ≥ 88 :=
  sorry

end NUMINAMATH_GPT_knowledge_competition_score_l1241_124146


namespace NUMINAMATH_GPT_correct_statement_l1241_124140

theorem correct_statement : ∀ (a b : ℝ), ((a ≠ b ∧ ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x = a ∨ x = b)) ∧
                                            ¬(∀ p q : ℝ, p = q → p = q) ∧
                                            ¬(∀ a : ℝ, |a| = -a → a < 0) ∧
                                            ¬(∀ a b : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (a = -b)) → (a / b = -1))) :=
by sorry

-- Explanation of conditions:
-- a  ≠ b ensures two distinct points
-- ¬(a < b ∧ ∀ x, (x = a ∨ x = b) → x is between a and b) incorrectly rephrased as shortest distance as a line segment
-- ¬(∀ p q : ℝ, p = q → p = q) is not directly used, a minimum to refute the concept as required.
-- |a| = -a → a < 0 reinterpreted as a ≤ 0 but incorrectly stated as < 0 explicitly refuted
-- ¬(∀ a b : ℝ, a ≠ 0 and/or b ≠ 0 maintained where a / b not strictly required/misinterpreted)

end NUMINAMATH_GPT_correct_statement_l1241_124140


namespace NUMINAMATH_GPT_problem1_l1241_124143

variable (x : ℝ)

theorem problem1 : 5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
  sorry

end NUMINAMATH_GPT_problem1_l1241_124143


namespace NUMINAMATH_GPT_m_necessary_not_sufficient_cond_l1241_124154

theorem m_necessary_not_sufficient_cond (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0) → m ≤ 2 :=
sorry

end NUMINAMATH_GPT_m_necessary_not_sufficient_cond_l1241_124154


namespace NUMINAMATH_GPT_bus_carrying_capacity_l1241_124153

variables (C : ℝ)

theorem bus_carrying_capacity (h1 : ∀ x : ℝ, x = (3 / 5) * C) 
                              (h2 : ∀ y : ℝ, y = 50 - 18)
                              (h3 : ∀ z : ℝ, x + y = C) : C = 80 :=
by
  sorry

end NUMINAMATH_GPT_bus_carrying_capacity_l1241_124153


namespace NUMINAMATH_GPT_no_valid_solutions_l1241_124183

theorem no_valid_solutions (a b : ℝ) (h1 : ∀ x, (a * x + b) ^ 2 = 4 * x^2 + 4 * x + 4) : false :=
  by
  sorry

end NUMINAMATH_GPT_no_valid_solutions_l1241_124183


namespace NUMINAMATH_GPT_intersection_point_exists_l1241_124197

def equation_1 (x y : ℝ) : Prop := 3 * x^2 - 12 * y^2 = 48
def line_eq (x y : ℝ) : Prop := y = - (1 / 3) * x + 5

theorem intersection_point_exists :
  ∃ (x y : ℝ), equation_1 x y ∧ line_eq x y ∧ x = 75 / 8 ∧ y = 15 / 8 :=
sorry

end NUMINAMATH_GPT_intersection_point_exists_l1241_124197


namespace NUMINAMATH_GPT_find_original_selling_price_l1241_124129

variable (SP : ℝ)
variable (CP : ℝ := 10000)
variable (discounted_SP : ℝ := 0.9 * SP)
variable (profit : ℝ := 0.08 * CP)

theorem find_original_selling_price :
  discounted_SP = CP + profit → SP = 12000 := by
sorry

end NUMINAMATH_GPT_find_original_selling_price_l1241_124129


namespace NUMINAMATH_GPT_greatest_teams_l1241_124186

-- Define the number of girls and boys as constants
def numGirls : ℕ := 40
def numBoys : ℕ := 32

-- Define the greatest number of teams possible with equal number of girls and boys as teams.
theorem greatest_teams : Nat.gcd numGirls numBoys = 8 := sorry

end NUMINAMATH_GPT_greatest_teams_l1241_124186


namespace NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_div_by_9_l1241_124112

theorem sum_of_cubes_of_consecutive_integers_div_by_9 (x : ℤ) : 
  let a := (x - 1) ^ 3
  let b := x ^ 3
  let c := (x + 1) ^ 3
  (a + b + c) % 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_consecutive_integers_div_by_9_l1241_124112


namespace NUMINAMATH_GPT_tilly_total_profit_l1241_124117

theorem tilly_total_profit :
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  total_profit = 300 :=
by
  let bags_sold := 100
  let selling_price_per_bag := 10
  let buying_price_per_bag := 7
  let profit_per_bag := selling_price_per_bag - buying_price_per_bag
  let total_profit := bags_sold * profit_per_bag
  sorry

end NUMINAMATH_GPT_tilly_total_profit_l1241_124117


namespace NUMINAMATH_GPT_prob_rain_next_day_given_today_rain_l1241_124190

variable (P_rain : ℝ) (P_rain_2_days : ℝ)
variable (p_given_rain : ℝ)

-- Given conditions
def condition_P_rain : Prop := P_rain = 1/3
def condition_P_rain_2_days : Prop := P_rain_2_days = 1/5

-- The question to prove
theorem prob_rain_next_day_given_today_rain (h1 : condition_P_rain P_rain) (h2 : condition_P_rain_2_days P_rain_2_days) :
  p_given_rain = 3/5 :=
by
  sorry

end NUMINAMATH_GPT_prob_rain_next_day_given_today_rain_l1241_124190


namespace NUMINAMATH_GPT_bmw_length_l1241_124145

theorem bmw_length : 
  let horiz1 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let horiz2 : ℝ := 2 -- Length of each horizontal segment in 'B'
  let vert1  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert2  : ℝ := 2 -- Length of each vertical segment in 'B'
  let vert3  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert4  : ℝ := 2 -- Length of each vertical segment in 'M'
  let vert5  : ℝ := 2 -- Length of each vertical segment in 'W'
  let diag1  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  let diag2  : ℝ := Real.sqrt 2 -- Length of each diagonal segment in 'W'
  (horiz1 + horiz2 + vert1 + vert2 + vert3 + vert4 + vert5 + diag1 + diag2) = 14 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_bmw_length_l1241_124145


namespace NUMINAMATH_GPT_people_in_room_l1241_124115

theorem people_in_room (total_people total_chairs : ℕ) (h1 : (2/3 : ℚ) * total_chairs = 1/2 * total_people)
  (h2 : total_chairs - (2/3 : ℚ) * total_chairs = 8) : total_people = 32 := 
by
  sorry

end NUMINAMATH_GPT_people_in_room_l1241_124115


namespace NUMINAMATH_GPT_matrix_power_101_l1241_124159

noncomputable def matrix_B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0], ![0, 0, 1], ![1, 0, 0]]

theorem matrix_power_101 :
  (matrix_B ^ 101) = ![![0, 0, 1], ![1, 0, 0], ![0, 1, 0]] :=
  sorry

end NUMINAMATH_GPT_matrix_power_101_l1241_124159


namespace NUMINAMATH_GPT_common_ratio_of_sequence_l1241_124149

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) (n1 n2 n3 : ℕ) : Prop :=
  a n2 = a n1 * r ∧ a n3 = a n1 * r^2

theorem common_ratio_of_sequence {a : ℕ → ℝ} {d : ℝ}
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence a ((a 2)/(a 1)) 2 3 6) :
  ((a 3) / (a 2)) = 3 ∨ ((a 3) / (a 2)) = 1 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_sequence_l1241_124149


namespace NUMINAMATH_GPT_triangle_area_PQR_l1241_124167

def point := (ℝ × ℝ)

def P : point := (2, 3)
def Q : point := (7, 3)
def R : point := (4, 10)

noncomputable def triangle_area (A B C : point) : ℝ :=
  (1/2) * ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_PQR : triangle_area P Q R = 17.5 :=
  sorry

end NUMINAMATH_GPT_triangle_area_PQR_l1241_124167


namespace NUMINAMATH_GPT_inequality_holds_if_and_only_if_l1241_124195

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_and_only_if (hx : |x-5| + |x-3| + |x-2| < b) : b > 4 :=
sorry

end NUMINAMATH_GPT_inequality_holds_if_and_only_if_l1241_124195
