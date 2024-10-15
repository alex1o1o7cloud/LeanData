import Mathlib

namespace NUMINAMATH_GPT_rhombus_diagonal_length_l352_35258

theorem rhombus_diagonal_length
  (d2 : ℝ)
  (h1 : d2 = 20)
  (area : ℝ)
  (h2 : area = 150) :
  ∃ d1 : ℝ, d1 = 15 ∧ (area = (d1 * d2) / 2) := by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l352_35258


namespace NUMINAMATH_GPT_circle_condition_l352_35271

noncomputable def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, circle_eq x y m) → m < 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_condition_l352_35271


namespace NUMINAMATH_GPT_swim_meet_time_l352_35215

theorem swim_meet_time {distance : ℕ} (d : distance = 50) (t : ℕ) 
  (meet_first : ∃ t1 : ℕ, t1 = 2 ∧ distance - 20 = 30) 
  (turn : ∀ t1, t1 = 2 → ∀ d1 : ℕ, d1 = 50 → t1 + t1 = 4) :
  t = 4 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_swim_meet_time_l352_35215


namespace NUMINAMATH_GPT_total_respondents_l352_35237

theorem total_respondents (X Y : ℕ) (hX : X = 360) (h_ratio : 9 * Y = X) : X + Y = 400 := by
  sorry

end NUMINAMATH_GPT_total_respondents_l352_35237


namespace NUMINAMATH_GPT_solution_mn_l352_35224

theorem solution_mn (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 5) (h3 : n < 0) : m + n = -1 ∨ m + n = -9 := 
by
  sorry

end NUMINAMATH_GPT_solution_mn_l352_35224


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l352_35200

theorem quadratic_has_distinct_real_roots (m : ℝ) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = m - 1 ∧ (b^2 - 4 * a * c > 0) → (m < 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l352_35200


namespace NUMINAMATH_GPT_tobias_mowed_four_lawns_l352_35225

-- Let’s define the conditions
def shoe_cost : ℕ := 95
def allowance_per_month : ℕ := 5
def savings_months : ℕ := 3
def lawn_mowing_charge : ℕ := 15
def shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def num_driveways_shoveled : ℕ := 5

-- Total money Tobias had before buying the shoes
def total_money : ℕ := shoe_cost + change_after_purchase

-- Money saved from allowance
def money_from_allowance : ℕ := allowance_per_month * savings_months

-- Money earned from shoveling driveways
def money_from_shoveling : ℕ := shoveling_charge * num_driveways_shoveled

-- Money earned from mowing lawns
def money_from_mowing : ℕ := total_money - money_from_allowance - money_from_shoveling

-- Number of lawns mowed
def num_lawns_mowed : ℕ := money_from_mowing / lawn_mowing_charge

-- The theorem stating the number of lawns mowed is 4
theorem tobias_mowed_four_lawns : num_lawns_mowed = 4 :=
by
  sorry

end NUMINAMATH_GPT_tobias_mowed_four_lawns_l352_35225


namespace NUMINAMATH_GPT_common_tangent_lines_l352_35205

theorem common_tangent_lines (m : ℝ) (hm : 0 < m) :
  (∀ x y : ℝ, x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0 →
     (y = 0 ∨ y = 4 / 3 * x - 4 / 3)) :=
by sorry

end NUMINAMATH_GPT_common_tangent_lines_l352_35205


namespace NUMINAMATH_GPT_sum_eq_2184_l352_35256

variable (p q r s : ℝ)

-- Conditions
axiom h1 : r + s = 12 * p
axiom h2 : r * s = 14 * q
axiom h3 : p + q = 12 * r
axiom h4 : p * q = 14 * s
axiom distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

-- Problem: Prove that p + q + r + s = 2184
theorem sum_eq_2184 : p + q + r + s = 2184 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_eq_2184_l352_35256


namespace NUMINAMATH_GPT_biology_marks_correct_l352_35226

-- Define the known marks in other subjects
def math_marks : ℕ := 76
def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 62

-- Define the total number of subjects
def total_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℕ := 74

-- Calculate the total marks of the known four subjects
def total_known_marks : ℕ := math_marks + science_marks + social_studies_marks + english_marks

-- Define a variable to represent the marks in biology
def biology_marks : ℕ := 370 - total_known_marks

-- Statement to prove
theorem biology_marks_correct : biology_marks = 85 := by
  sorry

end NUMINAMATH_GPT_biology_marks_correct_l352_35226


namespace NUMINAMATH_GPT_expected_digits_of_fair_icosahedral_die_l352_35217

noncomputable def expected_num_of_digits : ℚ :=
  (9 / 20) * 1 + (11 / 20) * 2

theorem expected_digits_of_fair_icosahedral_die :
  expected_num_of_digits = 1.55 := by
  sorry

end NUMINAMATH_GPT_expected_digits_of_fair_icosahedral_die_l352_35217


namespace NUMINAMATH_GPT_percentage_off_at_sale_l352_35247

theorem percentage_off_at_sale
  (sale_price original_price : ℝ)
  (h1 : sale_price = 140)
  (h2 : original_price = 350) :
  (original_price - sale_price) / original_price * 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_percentage_off_at_sale_l352_35247


namespace NUMINAMATH_GPT_brown_gumdrops_count_l352_35212

def gumdrops_conditions (total : ℕ) (blue : ℕ) (brown : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ) : Prop :=
  total = blue + brown + red + yellow + green ∧
  blue = total * 25 / 100 ∧
  brown = total * 25 / 100 ∧
  red = total * 20 / 100 ∧
  yellow = total * 15 / 100 ∧
  green = 40 ∧
  green = total * 15 / 100

theorem brown_gumdrops_count: ∃ total blue brown red yellow green new_brown,
  gumdrops_conditions total blue brown red yellow green →
  new_brown = brown + blue / 3 →
  new_brown = 89 :=
by
  sorry

end NUMINAMATH_GPT_brown_gumdrops_count_l352_35212


namespace NUMINAMATH_GPT_area_outside_circle_of_equilateral_triangle_l352_35284

noncomputable def equilateral_triangle_area_outside_circle {a : ℝ} (h : a > 0) : ℝ :=
  let S1 := a^2 * Real.sqrt 3 / 4
  let S2 := Real.pi * (a / 3)^2
  let S3 := (Real.pi * (a / 3)^2 / 6) - (a^2 * Real.sqrt 3 / 36)
  S1 - S2 + 3 * S3

theorem area_outside_circle_of_equilateral_triangle
  (a : ℝ) (h : a > 0) :
  equilateral_triangle_area_outside_circle h = a^2 * (3 * Real.sqrt 3 - Real.pi) / 18 :=
sorry

end NUMINAMATH_GPT_area_outside_circle_of_equilateral_triangle_l352_35284


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l352_35218

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l352_35218


namespace NUMINAMATH_GPT_euro_operation_example_l352_35286

def euro_operation (x y : ℕ) : ℕ := 3 * x * y - x - y

theorem euro_operation_example : euro_operation 6 (euro_operation 4 2) = 300 := by
  sorry

end NUMINAMATH_GPT_euro_operation_example_l352_35286


namespace NUMINAMATH_GPT_average_visitors_other_days_l352_35272

theorem average_visitors_other_days 
  (avg_sunday : ℕ) (avg_day : ℕ)
  (num_days : ℕ) (sunday_offset : ℕ)
  (other_days_count : ℕ) (total_days : ℕ) 
  (total_avg_visitors : ℕ)
  (sunday_avg_visitors : ℕ) :
  avg_sunday = 150 →
  avg_day = 125 →
  num_days = 30 →
  sunday_offset = 5 →
  total_days = 30 →
  total_avg_visitors * total_days =
    (sunday_offset * sunday_avg_visitors) + (other_days_count * avg_sunday) →
  125 = total_avg_visitors →
  150 = sunday_avg_visitors →
  other_days_count = num_days - sunday_offset →
  (125 * 30 = (5 * 150) + (other_days_count * avg_sunday)) →
  avg_sunday = 120 :=
by
  sorry

end NUMINAMATH_GPT_average_visitors_other_days_l352_35272


namespace NUMINAMATH_GPT_domain_of_f_l352_35273

def domain_valid (x : ℝ) :=
  1 - x ≥ 0 ∧ 1 - x ≠ 1

theorem domain_of_f :
  ∀ x : ℝ, domain_valid x ↔ (x ∈ Set.Iio 0 ∪ Set.Ioc 0 1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l352_35273


namespace NUMINAMATH_GPT_gcd_m_n_l352_35235

def m : ℕ := 3333333
def n : ℕ := 66666666

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_m_n_l352_35235


namespace NUMINAMATH_GPT_triangle_angle_B_l352_35206

theorem triangle_angle_B {A B C : ℝ} (h1 : A = 60) (h2 : B = 2 * C) (h3 : A + B + C = 180) : B = 80 :=
sorry

end NUMINAMATH_GPT_triangle_angle_B_l352_35206


namespace NUMINAMATH_GPT_perpendicular_tangents_sum_x1_x2_gt_4_l352_35210

noncomputable def f (x : ℝ) : ℝ := (1 / 6) * x^3 - (1 / 2) * x^2 + (1 / 3)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def F (x : ℝ) : ℝ := (1 / 2) * x^2 - x - 2 * Real.log x

theorem perpendicular_tangents (a : ℝ) (b : ℝ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 1 / 3) (h₃ : c = 0) :
  let f' x := (1 / 2) * x^2 - x
  let g' x := 2 / x
  f' 1 * g' 1 = -1 :=
by sorry

theorem sum_x1_x2_gt_4 (x1 x2 : ℝ) (h₁ : 0 < x1 ∧ x1 < 4) (h₂ : 0 < x2 ∧ x2 < 4) (h₃ : x1 ≠ x2) (h₄ : F x1 = F x2) :
  x1 + x2 > 4 :=
by sorry

end NUMINAMATH_GPT_perpendicular_tangents_sum_x1_x2_gt_4_l352_35210


namespace NUMINAMATH_GPT_students_taking_all_three_classes_l352_35201

variables (total_students Y B P N : ℕ)
variables (X₁ X₂ X₃ X₄ : ℕ)  -- variables representing students taking exactly two classes or all three

theorem students_taking_all_three_classes:
  total_students = 20 →
  Y = 10 →  -- Number of students taking yoga
  B = 13 →  -- Number of students taking bridge
  P = 9 →   -- Number of students taking painting
  N = 9 →   -- Number of students taking at least two classes
  X₂ + X₃ + X₄ = 9 →  -- This equation represents the total number of students taking at least two classes, where \( X₄ \) represents students taking all three (c).
  4 + X₃ + X₄ - (9 - X₃) + 1 + (9 - X₄ - X₂) + X₂ = 11 →
  X₄ = 3 :=                     -- Proving that the number of students taking all three classes is 3.
sorry

end NUMINAMATH_GPT_students_taking_all_three_classes_l352_35201


namespace NUMINAMATH_GPT_result_is_21_l352_35261

theorem result_is_21 (n : ℕ) (h : n = 55) : (n / 5 + 10) = 21 :=
by
  sorry

end NUMINAMATH_GPT_result_is_21_l352_35261


namespace NUMINAMATH_GPT_apples_per_basket_l352_35295

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) (h : total_apples = 629) (k : num_baskets = 37) :
  total_apples / num_baskets = 17 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_apples_per_basket_l352_35295


namespace NUMINAMATH_GPT_slope_l1_parallel_lines_math_proof_problem_l352_35253

-- Define the two lines
def l1 := ∀ x y : ℝ, x + 2 * y + 2 = 0
def l2 (a : ℝ) := ∀ x y : ℝ, a * x + y - 4 = 0

-- Define the assertions
theorem slope_l1 : ∀ x y : ℝ, x + 2 * y + 2 = 0 ↔ y = -1 / 2 * x - 1 := sorry

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) ↔ a = 1 / 2 := sorry

-- Using the assertions to summarize what we need to prove
theorem math_proof_problem (a : ℝ) :
  ((∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) → a = 1 / 2) ∧
  (∀ x y : ℝ, x + 2 * y + 2 = 0 → y = -1 / 2 * x - 1) := sorry

end NUMINAMATH_GPT_slope_l1_parallel_lines_math_proof_problem_l352_35253


namespace NUMINAMATH_GPT_tank_capacity_l352_35279

-- Define the conditions given in the problem.
def tank_full_capacity (x : ℝ) : Prop :=
  (0.25 * x = 60) ∧ (0.15 * x = 36)

-- State the theorem that needs to be proved.
theorem tank_capacity : ∃ x : ℝ, tank_full_capacity x ∧ x = 240 := 
by 
  sorry

end NUMINAMATH_GPT_tank_capacity_l352_35279


namespace NUMINAMATH_GPT_triangle_inequality_l352_35209

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l352_35209


namespace NUMINAMATH_GPT_problem_I_problem_II_l352_35232

def intervalA := { x : ℝ | -2 < x ∧ x < 5 }
def intervalB (m : ℝ) := { x : ℝ | m < x ∧ x < m + 3 }

theorem problem_I (m : ℝ) :
  (intervalB m ⊆ intervalA) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by sorry

theorem problem_II (m : ℝ) :
  (intervalA ∩ intervalB m ≠ ∅) ↔ (-5 < m ∧ m < 2) :=
by sorry

end NUMINAMATH_GPT_problem_I_problem_II_l352_35232


namespace NUMINAMATH_GPT_company_KW_price_l352_35265

theorem company_KW_price (A B : ℝ) (x : ℝ) (h1 : P = x * A) (h2 : P = 2 * B) (h3 : P = (6 / 7) * (A + B)) : x = 1.666666666666667 := 
sorry

end NUMINAMATH_GPT_company_KW_price_l352_35265


namespace NUMINAMATH_GPT_maximum_bottles_l352_35240

-- Definitions for the number of bottles each shop sells
def bottles_from_shop_A : ℕ := 150
def bottles_from_shop_B : ℕ := 180
def bottles_from_shop_C : ℕ := 220

-- The main statement to prove
theorem maximum_bottles : bottles_from_shop_A + bottles_from_shop_B + bottles_from_shop_C = 550 := 
by 
  sorry

end NUMINAMATH_GPT_maximum_bottles_l352_35240


namespace NUMINAMATH_GPT_bicycle_discount_l352_35231

theorem bicycle_discount (original_price : ℝ) (discount : ℝ) (discounted_price : ℝ) :
  original_price = 760 ∧ discount = 0.75 ∧ discounted_price = 570 → 
  original_price * discount = discounted_price := by
  sorry

end NUMINAMATH_GPT_bicycle_discount_l352_35231


namespace NUMINAMATH_GPT_salt_percentage_in_first_solution_l352_35244

variable (S : ℚ)
variable (H : 0 ≤ S ∧ S ≤ 100)  -- percentage constraints

theorem salt_percentage_in_first_solution (h : 0.75 * S / 100 + 7 = 16) : S = 12 :=
by { sorry }

end NUMINAMATH_GPT_salt_percentage_in_first_solution_l352_35244


namespace NUMINAMATH_GPT_find_a_l352_35238

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 :=
sorry

end NUMINAMATH_GPT_find_a_l352_35238


namespace NUMINAMATH_GPT_calculate_fraction_l352_35283

theorem calculate_fraction : (5 / (8 / 13) / (10 / 7) = 91 / 16) :=
by
  sorry

end NUMINAMATH_GPT_calculate_fraction_l352_35283


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000205_l352_35282

noncomputable def scientific_notation (n : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_0_0000205 :
  scientific_notation 0.0000205 = (2.05, -5) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000205_l352_35282


namespace NUMINAMATH_GPT_right_triangle_legs_sum_l352_35207

theorem right_triangle_legs_sum : 
  ∃ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) ∧ (x + (x + 1) = 57) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_legs_sum_l352_35207


namespace NUMINAMATH_GPT_monitor_width_l352_35221

theorem monitor_width (d w h : ℝ) (h_ratio : w / h = 16 / 9) (h_diag : d = 24) :
  w = 384 / Real.sqrt 337 :=
by
  sorry

end NUMINAMATH_GPT_monitor_width_l352_35221


namespace NUMINAMATH_GPT_solution_set_correct_l352_35216

noncomputable def solution_set (x : ℝ) : Prop :=
  x + 2 / (x + 1) > 2

theorem solution_set_correct :
  {x : ℝ | solution_set x} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by sorry

end NUMINAMATH_GPT_solution_set_correct_l352_35216


namespace NUMINAMATH_GPT_gondor_laptop_earning_l352_35250

theorem gondor_laptop_earning :
  ∃ L : ℝ, (3 * 10 + 5 * 10 + 2 * L + 4 * L = 200) → L = 20 :=
by
  use 20
  sorry

end NUMINAMATH_GPT_gondor_laptop_earning_l352_35250


namespace NUMINAMATH_GPT_initial_people_on_train_l352_35287

theorem initial_people_on_train {x y z u v w : ℤ} 
  (h1 : y = 29) (h2 : z = 17) (h3 : u = 27) (h4 : v = 35) (h5 : w = 116) :
  x - (y - z) + (v - u) = w → x = 120 := 
by sorry

end NUMINAMATH_GPT_initial_people_on_train_l352_35287


namespace NUMINAMATH_GPT_pairs_of_mittens_correct_l352_35263

variables (pairs_of_plugs_added pairs_of_plugs_original plugs_total pairs_of_plugs_current pairs_of_mittens : ℕ)

theorem pairs_of_mittens_correct :
  pairs_of_plugs_added = 30 →
  plugs_total = 400 →
  pairs_of_plugs_current = plugs_total / 2 →
  pairs_of_plugs_current = pairs_of_plugs_original + pairs_of_plugs_added →
  pairs_of_mittens = pairs_of_plugs_original - 20 →
  pairs_of_mittens = 150 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_pairs_of_mittens_correct_l352_35263


namespace NUMINAMATH_GPT_total_bad_carrots_and_tomatoes_l352_35251

theorem total_bad_carrots_and_tomatoes 
  (vanessa_carrots : ℕ := 17)
  (vanessa_tomatoes : ℕ := 12)
  (mother_carrots : ℕ := 14)
  (mother_tomatoes : ℕ := 22)
  (brother_carrots : ℕ := 6)
  (brother_tomatoes : ℕ := 8)
  (good_carrots : ℕ := 28)
  (good_tomatoes : ℕ := 35) :
  (vanessa_carrots + mother_carrots + brother_carrots - good_carrots) + 
  (vanessa_tomatoes + mother_tomatoes + brother_tomatoes - good_tomatoes) = 16 := 
by
  sorry

end NUMINAMATH_GPT_total_bad_carrots_and_tomatoes_l352_35251


namespace NUMINAMATH_GPT_sum_lent_is_1100_l352_35290

variables (P : ℝ) (r : ℝ) (t : ℝ) (I : ℝ)

-- Given conditions
def interest_formula := I = P * r * t
def interest_difference := I = P - 572

-- Values
def rate := r = 0.06
def time := t = 8

theorem sum_lent_is_1100 : P = 1100 :=
by
  -- Definitions and axioms
  sorry

end NUMINAMATH_GPT_sum_lent_is_1100_l352_35290


namespace NUMINAMATH_GPT_volume_less_than_1000_l352_35219

noncomputable def volume (x : ℕ) : ℤ :=
(x + 3) * (x - 1) * (x^3 - 20)

theorem volume_less_than_1000 : ∃ (n : ℕ), n = 2 ∧ 
  ∃ x1 x2, x1 ≠ x2 ∧ 0 < x1 ∧ 
  0 < x2 ∧
  volume x1 < 1000 ∧
  volume x2 < 1000 ∧
  ∀ x, 0 < x → volume x < 1000 → (x = x1 ∨ x = x2) :=
by
  sorry

end NUMINAMATH_GPT_volume_less_than_1000_l352_35219


namespace NUMINAMATH_GPT_total_students_high_school_l352_35242

theorem total_students_high_school (s10 s11 s12 total_students sample: ℕ ) 
  (h1 : s10 = 600) 
  (h2 : sample = 45) 
  (h3 : s11 = 20) 
  (h4 : s12 = 10) 
  (h5 : sample = s10 + s11 + s12) : 
  total_students = 1800 :=
by 
  sorry

end NUMINAMATH_GPT_total_students_high_school_l352_35242


namespace NUMINAMATH_GPT_comparison_inequalities_l352_35260

open Real

theorem comparison_inequalities
  (m : ℝ) (h1 : 3 ^ m = Real.exp 1) 
  (a : ℝ) (h2 : a = cos m) 
  (b : ℝ) (h3 : b = 1 - 1/2 * m^2)
  (c : ℝ) (h4 : c = sin m / m) :
  c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_comparison_inequalities_l352_35260


namespace NUMINAMATH_GPT_john_replace_bedroom_doors_l352_35296

variable (B O : ℕ)
variable (cost_outside cost_bedroom total_cost : ℕ)

def john_has_to_replace_bedroom_doors : Prop :=
  let outside_doors_replaced := 2
  let cost_of_outside_door := 20
  let cost_of_bedroom_door := 10
  let total_replacement_cost := 70
  O = outside_doors_replaced ∧
  cost_outside = cost_of_outside_door ∧
  cost_bedroom = cost_of_bedroom_door ∧
  total_cost = total_replacement_cost ∧
  20 * O + 10 * B = total_cost →
  B = 3

theorem john_replace_bedroom_doors : john_has_to_replace_bedroom_doors B O cost_outside cost_bedroom total_cost :=
sorry

end NUMINAMATH_GPT_john_replace_bedroom_doors_l352_35296


namespace NUMINAMATH_GPT_bridget_apples_l352_35222

theorem bridget_apples :
  ∃ x : ℕ, (x - x / 3 - 4) = 6 :=
by
  sorry

end NUMINAMATH_GPT_bridget_apples_l352_35222


namespace NUMINAMATH_GPT_parabola_standard_equation_l352_35274

theorem parabola_standard_equation (x y : ℝ) : 
  (3 * x - 4 * y - 12 = 0) →
  (y = 0 → x = 4 ∨ y = -3 → x = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
by
  intros h_line h_intersect
  sorry

end NUMINAMATH_GPT_parabola_standard_equation_l352_35274


namespace NUMINAMATH_GPT_initial_pokemon_cards_l352_35228

theorem initial_pokemon_cards (x : ℕ) (h : x - 9 = 4) : x = 13 := by
  sorry

end NUMINAMATH_GPT_initial_pokemon_cards_l352_35228


namespace NUMINAMATH_GPT_quadratic_axis_of_symmetry_l352_35297

theorem quadratic_axis_of_symmetry (b c : ℝ) (h : -b / 2 = 3) : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_axis_of_symmetry_l352_35297


namespace NUMINAMATH_GPT_find_x_l352_35259

theorem find_x (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (3, x)) (h : (a.fst * b.fst + a.snd * b.snd) = 3) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l352_35259


namespace NUMINAMATH_GPT_matt_twice_james_age_in_5_years_l352_35288

theorem matt_twice_james_age_in_5_years :
  (∃ x : ℕ, (3 + 27 = 30) ∧ (Matt_current_age = 65) ∧ 
  (Matt_age_in_x_years = Matt_current_age + x) ∧ 
  (James_age_in_x_years = James_current_age + x) ∧ 
  (Matt_age_in_x_years = 2 * James_age_in_x_years) → x = 5) :=
sorry

end NUMINAMATH_GPT_matt_twice_james_age_in_5_years_l352_35288


namespace NUMINAMATH_GPT_composite_number_iff_ge_2_l352_35203

theorem composite_number_iff_ge_2 (n : ℕ) : 
  ¬(Prime (3^(2*n+1) - 2^(2*n+1) - 6^n)) ↔ n ≥ 2 := by
  sorry

end NUMINAMATH_GPT_composite_number_iff_ge_2_l352_35203


namespace NUMINAMATH_GPT_manufacturing_cost_before_decrease_l352_35285

def original_manufacturing_cost (P : ℝ) (C_now : ℝ) (profit_rate_now : ℝ) : ℝ :=
  P - profit_rate_now * P

theorem manufacturing_cost_before_decrease
  (P : ℝ)
  (C_now : ℝ)
  (profit_rate_now : ℝ)
  (profit_rate_original : ℝ)
  (H1 : C_now = P - profit_rate_now * P)
  (H2 : profit_rate_now = 0.50)
  (H3 : profit_rate_original = 0.20)
  (H4 : C_now = 50) :
  original_manufacturing_cost P C_now profit_rate_now = 80 :=
sorry

end NUMINAMATH_GPT_manufacturing_cost_before_decrease_l352_35285


namespace NUMINAMATH_GPT_p_more_than_q_l352_35227

def stamps (p q : ℕ) : Prop :=
  p / q = 7 / 4 ∧ (p - 8) / (q + 8) = 6 / 5

theorem p_more_than_q (p q : ℕ) (h : stamps p q) : p - 8 - (q + 8) = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_p_more_than_q_l352_35227


namespace NUMINAMATH_GPT_alien_run_time_l352_35267

variable (v_r v_f : ℝ) -- velocities in km/h
variable (T_r T_f : ℝ) -- time in hours
variable (D_r D_f : ℝ) -- distances in kilometers

theorem alien_run_time :
  v_r = 15 ∧ v_f = 10 ∧ (T_f = T_r + 0.5) ∧ (D_r = D_f) ∧ (D_r = v_r * T_r) ∧ (D_f = v_f * T_f) → T_f = 1.5 :=
by
  intros h
  rcases h with ⟨_, _, _, _, _, _⟩
  -- proof goes here
  sorry

end NUMINAMATH_GPT_alien_run_time_l352_35267


namespace NUMINAMATH_GPT_Lowella_score_l352_35239

theorem Lowella_score
  (Mandy_score : ℕ)
  (Pamela_score : ℕ)
  (Lowella_score : ℕ)
  (h1 : Mandy_score = 84) 
  (h2 : Mandy_score = 2 * Pamela_score)
  (h3 : Pamela_score = Lowella_score + 20) :
  Lowella_score = 22 := by
  sorry

end NUMINAMATH_GPT_Lowella_score_l352_35239


namespace NUMINAMATH_GPT_bob_25_cent_coins_l352_35243

theorem bob_25_cent_coins (a b c : ℕ)
    (h₁ : a + b + c = 15)
    (h₂ : 15 + 4 * c = 27) : c = 3 := by
  sorry

end NUMINAMATH_GPT_bob_25_cent_coins_l352_35243


namespace NUMINAMATH_GPT_line_through_point_parallel_l352_35289

theorem line_through_point_parallel (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) (hA : A = (2, 3)) (hl : ∀ x y, l x y ↔ 2 * x - 4 * y + 7 = 0) :
  ∃ m, (∀ x y, (2 * x - 4 * y + m = 0) ↔ (x - 2 * y + 4 = 0)) ∧ (2 * (A.1) - 4 * (A.2) + m = 0) := 
sorry

end NUMINAMATH_GPT_line_through_point_parallel_l352_35289


namespace NUMINAMATH_GPT_new_team_average_weight_l352_35202

theorem new_team_average_weight :
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  (new_total_weight / new_player_count) = 92 :=
by
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  sorry

end NUMINAMATH_GPT_new_team_average_weight_l352_35202


namespace NUMINAMATH_GPT_vector_on_line_l352_35230

noncomputable def k_value (a b : Vector ℝ 3) (m : ℝ) : ℝ :=
  if h : m = 5 / 7 then
    (5 / 7 : ℝ)
  else
    0 -- This branch will never be taken because we will assume m = 5 / 7 as a hypothesis.


theorem vector_on_line (a b : Vector ℝ 3) (m k : ℝ) (h : m = 5 / 7) :
  k = k_value a b m :=
by
  sorry

end NUMINAMATH_GPT_vector_on_line_l352_35230


namespace NUMINAMATH_GPT_cylinder_base_radii_l352_35281

theorem cylinder_base_radii {l w : ℝ} (hl : l = 3 * Real.pi) (hw : w = Real.pi) :
  (∃ r : ℝ, l = 2 * Real.pi * r ∧ r = 3 / 2) ∨ (∃ r : ℝ, w = 2 * Real.pi * r ∧ r = 1 / 2) :=
sorry

end NUMINAMATH_GPT_cylinder_base_radii_l352_35281


namespace NUMINAMATH_GPT_triangle_angles_ratio_l352_35293

theorem triangle_angles_ratio (A B C : ℕ) 
  (hA : A = 20)
  (hB : B = 3 * A)
  (hSum : A + B + C = 180) :
  (C / A) = 5 := 
by
  sorry

end NUMINAMATH_GPT_triangle_angles_ratio_l352_35293


namespace NUMINAMATH_GPT_odd_integer_divisibility_l352_35220

theorem odd_integer_divisibility (n : ℕ) (hodd : n % 2 = 1) (hpos : n > 0) : ∃ k : ℕ, n^4 - n^2 - n = n * k := 
sorry

end NUMINAMATH_GPT_odd_integer_divisibility_l352_35220


namespace NUMINAMATH_GPT_second_number_value_l352_35248

-- Definition of the problem conditions
variables (x y z : ℝ)
axiom h1 : z = 4.5 * y
axiom h2 : y = 2.5 * x
axiom h3 : (x + y + z) / 3 = 165

-- The goal is to prove y = 82.5 given the conditions h1, h2, and h3
theorem second_number_value : y = 82.5 :=
by
  sorry

end NUMINAMATH_GPT_second_number_value_l352_35248


namespace NUMINAMATH_GPT_cook_carrots_l352_35276

theorem cook_carrots :
  ∀ (total_carrots : ℕ) (fraction_used_before_lunch : ℚ) (carrots_not_used_end_of_day : ℕ),
    total_carrots = 300 →
    fraction_used_before_lunch = 2 / 5 →
    carrots_not_used_end_of_day = 72 →
    let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
    let carrots_after_lunch := total_carrots - carrots_used_before_lunch
    let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
    (carrots_used_end_of_day / carrots_after_lunch) = 3 / 5 :=
by
  intros total_carrots fraction_used_before_lunch carrots_not_used_end_of_day
  intros h1 h2 h3
  let carrots_used_before_lunch := total_carrots * fraction_used_before_lunch
  let carrots_after_lunch := total_carrots - carrots_used_before_lunch
  let carrots_used_end_of_day := carrots_after_lunch - carrots_not_used_end_of_day
  have h : carrots_used_end_of_day / carrots_after_lunch = 3 / 5 := sorry
  exact h

end NUMINAMATH_GPT_cook_carrots_l352_35276


namespace NUMINAMATH_GPT_factor_fraction_eq_l352_35229

theorem factor_fraction_eq (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) 
  / ((a + b)^3 + (b + c)^3 + (c + a)^3) = 
  ((a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2)) 
  / ((a + b) * (b + c) * (c + a)) :=
by
  sorry

end NUMINAMATH_GPT_factor_fraction_eq_l352_35229


namespace NUMINAMATH_GPT_original_amount_of_money_l352_35245

variable (took : ℕ) (now : ℕ) (initial : ℕ)

-- conditions from the problem
def conditions := (took = 2) ∧ (now = 3)

-- the statement to prove
theorem original_amount_of_money {took now initial : ℕ} (h : conditions took now) :
  initial = now + took ↔ initial = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_amount_of_money_l352_35245


namespace NUMINAMATH_GPT_equivalent_single_discount_l352_35270

variable (x : ℝ)
variable (original_price : ℝ := x)
variable (discount1 : ℝ := 0.15)
variable (discount2 : ℝ := 0.10)
variable (discount3 : ℝ := 0.05)

theorem equivalent_single_discount :
  let final_price := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)
  let equivalent_discount := (1 - final_price / original_price)
  equivalent_discount = 0.27 := 
by 
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l352_35270


namespace NUMINAMATH_GPT_average_track_width_l352_35211

theorem average_track_width (r1 r2 s1 s2 : ℝ) 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : 2 * Real.pi * s1 - 2 * Real.pi * s2 = 30 * Real.pi) :
  (r1 - r2 + (s1 - s2)) / 2 = 12.5 := 
sorry

end NUMINAMATH_GPT_average_track_width_l352_35211


namespace NUMINAMATH_GPT_arithmetic_sum_s6_l352_35249

theorem arithmetic_sum_s6 (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) 
  (h1 : ∀ n, a (n+1) - a n = d)
  (h2 : a 1 = 2)
  (h3 : S 4 = 20)
  (hS : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) :
  S 6 = 42 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sum_s6_l352_35249


namespace NUMINAMATH_GPT_sum_of_roots_eq_five_thirds_l352_35255

-- Define the quadratic equation
def quadratic_eq (n : ℝ) : Prop := 3 * n^2 - 5 * n - 4 = 0

-- Prove that the sum of the solutions to the quadratic equation is 5/3
theorem sum_of_roots_eq_five_thirds :
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 5 / 3) :=
sorry

end NUMINAMATH_GPT_sum_of_roots_eq_five_thirds_l352_35255


namespace NUMINAMATH_GPT_no_rational_roots_of_prime_3_digit_l352_35246

noncomputable def is_prime (n : ℕ) := Nat.Prime n

theorem no_rational_roots_of_prime_3_digit (a b c : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9) 
(h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : 0 ≤ c ∧ c ≤ 9) 
(p := 100 * a + 10 * b + c) (hp : is_prime p) (h₃ : 100 ≤ p ∧ p ≤ 999) :
¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_no_rational_roots_of_prime_3_digit_l352_35246


namespace NUMINAMATH_GPT_pens_sold_l352_35277

theorem pens_sold (C : ℝ) (N : ℝ) (h_gain : 22 * C = 0.25 * N * C) : N = 88 :=
by {
  sorry
}

end NUMINAMATH_GPT_pens_sold_l352_35277


namespace NUMINAMATH_GPT_quadratic_root_shift_c_value_l352_35214

theorem quadratic_root_shift_c_value
  (r s : ℝ)
  (h1 : r + s = 2)
  (h2 : r * s = -5) :
  ∃ b : ℝ, x^2 + b * x - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_shift_c_value_l352_35214


namespace NUMINAMATH_GPT_coordinate_fifth_point_l352_35257

theorem coordinate_fifth_point : 
  ∃ (a : Fin 16 → ℝ), 
    a 0 = 2 ∧ 
    a 15 = 47 ∧ 
    (∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) ∧ 
    a 4 = 14 := 
sorry

end NUMINAMATH_GPT_coordinate_fifth_point_l352_35257


namespace NUMINAMATH_GPT_building_height_l352_35204

theorem building_height (h : ℕ) (flagpole_height : ℕ) (flagpole_shadow : ℕ) (building_shadow : ℕ) :
  flagpole_height = 18 ∧ flagpole_shadow = 45 ∧ building_shadow = 60 → h = 24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_building_height_l352_35204


namespace NUMINAMATH_GPT_area_of_rectangle_l352_35223

noncomputable def leanProblem : Prop :=
  let E := 8
  let F := 2.67
  let BE := E -- length from B to E on AB
  let AF := F -- length from A to F on AD
  let BC := E * (Real.sqrt 3) -- from triangle properties CB is BE * sqrt(3)
  let FD := BC - F -- length from F to D on AD
  let CD := FD * (Real.sqrt 3) -- applying the triangle properties again
  (BC * CD = 192 * (Real.sqrt 3) - 64.08)

theorem area_of_rectangle (E : ℝ) (F : ℝ) 
  (hE : E = 8) 
  (hF : F = 2.67) 
  (BC : ℝ) (CD : ℝ) :
  leanProblem :=
by 
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l352_35223


namespace NUMINAMATH_GPT_problem_solution_l352_35299

theorem problem_solution :
  (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := 
  by 
  sorry

end NUMINAMATH_GPT_problem_solution_l352_35299


namespace NUMINAMATH_GPT_beads_pulled_out_l352_35280

theorem beads_pulled_out (white_beads black_beads : ℕ) (frac_black frac_white : ℚ) (h_black : black_beads = 90) (h_white : white_beads = 51) (h_frac_black : frac_black = (1/6)) (h_frac_white : frac_white = (1/3)) : 
  white_beads * frac_white + black_beads * frac_black = 32 := 
by
  sorry

end NUMINAMATH_GPT_beads_pulled_out_l352_35280


namespace NUMINAMATH_GPT_negation_proposition_l352_35275

theorem negation_proposition : ¬(∀ x : ℝ, x > 0 → x ≥ 1) ↔ ∃ x : ℝ, x > 0 ∧ x < 1 := 
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l352_35275


namespace NUMINAMATH_GPT_perimeter_of_garden_l352_35278

def area (length width : ℕ) : ℕ := length * width

def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

theorem perimeter_of_garden :
  ∀ (l w : ℕ), area l w = 28 ∧ l = 7 → perimeter l w = 22 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_garden_l352_35278


namespace NUMINAMATH_GPT_domain_of_function_l352_35236

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 1 / Real.sqrt (2 - x^2)

theorem domain_of_function : 
  {x : ℝ | x > -1 ∧ x < Real.sqrt 2} = {x : ℝ | x ∈ Set.Ioo (-1) (Real.sqrt 2)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l352_35236


namespace NUMINAMATH_GPT_no_integer_solution_for_Q_square_l352_35234

def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

theorem no_integer_solution_for_Q_square :
  ∀ x : ℤ, ∃ k : ℤ, Q x = k^2 → false :=
by
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_Q_square_l352_35234


namespace NUMINAMATH_GPT_area_of_triangle_l352_35254

variables (yellow_area green_area blue_area : ℝ)
variables (is_equilateral_triangle : Prop)
variables (centered_at_vertices : Prop)
variables (radius_less_than_height : Prop)

theorem area_of_triangle (h_yellow : yellow_area = 1000)
                        (h_green : green_area = 100)
                        (h_blue : blue_area = 1)
                        (h_triangle : is_equilateral_triangle)
                        (h_centered : centered_at_vertices)
                        (h_radius : radius_less_than_height) :
  ∃ (area : ℝ), area = 150 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l352_35254


namespace NUMINAMATH_GPT_range_of_d_l352_35252

noncomputable def sn (n a1 d : ℝ) := (n / 2) * (2 * a1 + (n - 1) * d)

theorem range_of_d (a1 d : ℝ) (h_eq : (sn 2 a1 d) * (sn 4 a1 d) / 2 + (sn 3 a1 d) ^ 2 / 9 + 2 = 0) :
  d ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici (Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_range_of_d_l352_35252


namespace NUMINAMATH_GPT_probability_abs_diff_gt_half_is_7_over_16_l352_35298

noncomputable def probability_abs_diff_gt_half : ℚ :=
  let p_tail := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping tails
  let p_head := (1 : ℚ) / (2 : ℚ)   -- Probability of flipping heads
  let p_x_tail_y_tail := p_tail * p_tail   -- Both first flips tails
  let p_x1_y_tail := p_head * p_tail / 2     -- x = 1, y flip tails
  let p_x_tail_y0 := p_tail * p_head / 2     -- x flip tails, y = 0
  let p_x1_y0 := p_head * p_head / 4         -- x = 1, y = 0
  -- Individual probabilities for x − y > 1/2
  let p_x_tail_y_tail_diff := (1 : ℚ) / (8 : ℚ) * p_x_tail_y_tail
  let p_x1_y_tail_diff := (1 : ℚ) / (2 : ℚ) * p_x1_y_tail
  let p_x_tail_y0_diff := (1 : ℚ) / (2 : ℚ) * p_x_tail_y0
  let p_x1_y0_diff := (1 : ℚ) * p_x1_y0
  -- Combined probability for x − y > 1/2
  let p_x_y_diff_gt_half := p_x_tail_y_tail_diff +
                            p_x1_y_tail_diff +
                            p_x_tail_y0_diff +
                            p_x1_y0_diff
  -- Final probability for |x − y| > 1/2 is twice of x − y > 1/2
  2 * p_x_y_diff_gt_half

theorem probability_abs_diff_gt_half_is_7_over_16 :
  probability_abs_diff_gt_half = (7 : ℚ) / 16 := 
  sorry

end NUMINAMATH_GPT_probability_abs_diff_gt_half_is_7_over_16_l352_35298


namespace NUMINAMATH_GPT_cost_of_sandwiches_and_smoothies_l352_35264

-- Define the cost of sandwiches and smoothies
def sandwich_cost := 4
def smoothie_cost := 3

-- Define the discount applicable
def sandwich_discount := 1
def total_sandwiches := 6
def total_smoothies := 7

-- Calculate the effective cost per sandwich considering discount
def effective_sandwich_cost := if total_sandwiches > 4 then sandwich_cost - sandwich_discount else sandwich_cost

-- Calculate the total cost for sandwiches
def sandwiches_cost := total_sandwiches * effective_sandwich_cost

-- Calculate the total cost for smoothies
def smoothies_cost := total_smoothies * smoothie_cost

-- Calculate the total cost
def total_cost := sandwiches_cost + smoothies_cost

-- The main statement to prove
theorem cost_of_sandwiches_and_smoothies : total_cost = 39 := by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_cost_of_sandwiches_and_smoothies_l352_35264


namespace NUMINAMATH_GPT_min_value_of_m_l352_35268

noncomputable def g (x : ℝ) := (Real.exp x + Real.exp (-x)) / 2
noncomputable def h (x : ℝ) := (Real.exp (-x) - Real.exp x) / 2

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → m * g x + h x ≥ 0) → m ≥ (Real.exp 2 - 1) / (Real.exp 2 + 1) :=
by
  intro h
  have key_ineq : ∀ x, -1 ≤ x ∧ x ≤ 1 → m ≥ 1 - 2 / (Real.exp (2 * x) + 1) := sorry
  sorry

end NUMINAMATH_GPT_min_value_of_m_l352_35268


namespace NUMINAMATH_GPT_algebraic_expression_value_l352_35266

theorem algebraic_expression_value (x : ℝ) (h : 5 * x^2 - x - 2 = 0) :
  (2 * x + 1) * (2 * x - 1) + x * (x - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l352_35266


namespace NUMINAMATH_GPT_Jenny_ate_65_l352_35294

theorem Jenny_ate_65 (mike_squares : ℕ) (jenny_squares : ℕ)
  (h1 : mike_squares = 20)
  (h2 : jenny_squares = 3 * mike_squares + 5) :
  jenny_squares = 65 :=
by
  sorry

end NUMINAMATH_GPT_Jenny_ate_65_l352_35294


namespace NUMINAMATH_GPT_sum_five_consecutive_l352_35208

theorem sum_five_consecutive (n : ℤ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) = 5 * n + 10 := by
  sorry

end NUMINAMATH_GPT_sum_five_consecutive_l352_35208


namespace NUMINAMATH_GPT_veromont_clicked_ads_l352_35292

def ads_on_first_page := 12
def ads_on_second_page := 2 * ads_on_first_page
def ads_on_third_page := ads_on_second_page + 24
def ads_on_fourth_page := (3 / 4) * ads_on_second_page
def total_ads := ads_on_first_page + ads_on_second_page + ads_on_third_page + ads_on_fourth_page
def ads_clicked := (2 / 3) * total_ads

theorem veromont_clicked_ads : ads_clicked = 68 := 
by
  sorry

end NUMINAMATH_GPT_veromont_clicked_ads_l352_35292


namespace NUMINAMATH_GPT_total_cost_is_26_30_l352_35269

open Real

-- Define the costs
def cost_snake_toy : ℝ := 11.76
def cost_cage : ℝ := 14.54

-- Define the total cost of purchases
def total_cost : ℝ := cost_snake_toy + cost_cage

-- Prove the total cost equals $26.30
theorem total_cost_is_26_30 : total_cost = 26.30 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_26_30_l352_35269


namespace NUMINAMATH_GPT_find_n_l352_35233

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = -1 / (a n + 1)

theorem find_n (a : ℕ → ℚ) (h : seq a) : ∃ n : ℕ, a n = 3 ∧ n = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l352_35233


namespace NUMINAMATH_GPT_father_payment_l352_35291

variable (x y : ℤ)

theorem father_payment :
  5 * x - 3 * y = 24 :=
sorry

end NUMINAMATH_GPT_father_payment_l352_35291


namespace NUMINAMATH_GPT_total_words_in_week_l352_35241

def typing_minutes_MWF : ℤ := 260
def typing_minutes_TTh : ℤ := 150
def typing_minutes_Sat : ℤ := 240
def typing_speed_MWF : ℤ := 50
def typing_speed_TTh : ℤ := 40
def typing_speed_Sat : ℤ := 60

def words_per_day_MWF : ℤ := typing_minutes_MWF * typing_speed_MWF
def words_per_day_TTh : ℤ := typing_minutes_TTh * typing_speed_TTh
def words_Sat : ℤ := typing_minutes_Sat * typing_speed_Sat

def total_words_week : ℤ :=
  (words_per_day_MWF * 3) + (words_per_day_TTh * 2) + words_Sat + 0

theorem total_words_in_week :
  total_words_week = 65400 :=
by
  sorry

end NUMINAMATH_GPT_total_words_in_week_l352_35241


namespace NUMINAMATH_GPT_kevin_found_cards_l352_35262

-- Definitions from the conditions
def initial_cards : ℕ := 7
def final_cards : ℕ := 54

-- The proof goal
theorem kevin_found_cards : final_cards - initial_cards = 47 :=
by
  sorry

end NUMINAMATH_GPT_kevin_found_cards_l352_35262


namespace NUMINAMATH_GPT_inequality_solution_set_l352_35213

theorem inequality_solution_set (x : ℝ) : (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l352_35213
