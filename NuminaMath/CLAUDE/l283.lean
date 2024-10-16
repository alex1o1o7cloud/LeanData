import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l283_28344

-- Define a decreasing function on (-2, 2)
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y

-- Define the solution set
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | -2 < x ∧ x < 2 ∧ f x > f (2 - x)}

-- Theorem statement
theorem solution_set_is_open_unit_interval
  (f : ℝ → ℝ) (h : DecreasingFunction f) :
  SolutionSet f = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_unit_interval_l283_28344


namespace NUMINAMATH_CALUDE_cubic_root_sum_l283_28360

theorem cubic_root_sum (p q r : ℂ) : 
  (p^3 - 2*p - 2 = 0) → 
  (q^3 - 2*q - 2 = 0) → 
  (r^3 - 2*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l283_28360


namespace NUMINAMATH_CALUDE_m_values_l283_28380

def A : Set ℝ := {x | x^2 - 3*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x - 1 = 0}

theorem m_values : ∀ m : ℝ, (A ∪ B m = A) ↔ (m = 0 ∨ m = -1/2 ∨ m = 1/5) := by sorry

end NUMINAMATH_CALUDE_m_values_l283_28380


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l283_28370

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = -9 ∧ x₂ = -1 ∧ 
  (x₁^2 + 10*x₁ + 9 = 0) ∧ 
  (x₂^2 + 10*x₂ + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l283_28370


namespace NUMINAMATH_CALUDE_age_sum_from_product_l283_28342

theorem age_sum_from_product (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 144 → a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_from_product_l283_28342


namespace NUMINAMATH_CALUDE_intersection_point_y_value_l283_28366

def f (x : ℝ) := 2 * x^2 - 3 * x + 10

theorem intersection_point_y_value :
  ∀ c : ℝ, f 7 = c → c = 87 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_y_value_l283_28366


namespace NUMINAMATH_CALUDE_group_selection_count_l283_28327

theorem group_selection_count : 
  let total_students : ℕ := 7
  let male_students : ℕ := 4
  let female_students : ℕ := 3
  let group_size : ℕ := 3
  (Nat.choose total_students group_size) - 
  (Nat.choose male_students group_size) - 
  (Nat.choose female_students group_size) = 30 := by
sorry

end NUMINAMATH_CALUDE_group_selection_count_l283_28327


namespace NUMINAMATH_CALUDE_sugar_solution_problem_l283_28397

/-- Calculates the final sugar percentage when replacing part of a solution --/
def finalSugarPercentage (initialPercent : ℝ) (replacementPercent : ℝ) (replacementFraction : ℝ) : ℝ :=
  (initialPercent * (1 - replacementFraction) + replacementPercent * replacementFraction) * 100

/-- Theorem stating the final sugar percentage for the given problem --/
theorem sugar_solution_problem :
  finalSugarPercentage 0.1 0.42 0.25 = 18 := by
  sorry

#eval finalSugarPercentage 0.1 0.42 0.25

end NUMINAMATH_CALUDE_sugar_solution_problem_l283_28397


namespace NUMINAMATH_CALUDE_square_of_105_l283_28349

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by sorry

end NUMINAMATH_CALUDE_square_of_105_l283_28349


namespace NUMINAMATH_CALUDE_heart_calculation_l283_28392

-- Define the ♥ operation
def heart (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heart_calculation : heart 3 (heart 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_heart_calculation_l283_28392


namespace NUMINAMATH_CALUDE_provisions_problem_l283_28389

/-- The initial number of men given the conditions of the problem -/
def initial_men : ℕ := 1000

/-- The number of days the provisions last for the initial group -/
def initial_days : ℕ := 20

/-- The number of additional men that join the group -/
def additional_men : ℕ := 650

/-- The number of days the provisions last after additional men join -/
def final_days : ℚ := 12121212121212121 / 1000000000000000

theorem provisions_problem :
  initial_men * initial_days = (initial_men + additional_men) * final_days :=
sorry

end NUMINAMATH_CALUDE_provisions_problem_l283_28389


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l283_28343

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  a > c →
  (1/2) * a * c * Real.sin B = 3/2 →
  Real.cos B = 4/5 →
  b = 3 * Real.sqrt 2 →
  (a = 5 ∧ c = 1) ∧
  Real.cos (B - C) = (31 * Real.sqrt 2) / 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l283_28343


namespace NUMINAMATH_CALUDE_number_of_true_statements_number_of_true_statements_is_correct_l283_28352

/-- A quadratic equation x^2 + x - m = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + x - m = 0

/-- The number of true propositions among the statement and its variants -/
theorem number_of_true_statements : ℕ :=
  let s1 := ∀ m : ℝ, m > 0 → has_real_roots m
  let s2 := ∀ m : ℝ, has_real_roots m → m > 0
  let s3 := ∀ m : ℝ, m ≤ 0 → ¬has_real_roots m
  let s4 := ∀ m : ℝ, ¬has_real_roots m → m ≤ 0
  2

theorem number_of_true_statements_is_correct :
  number_of_true_statements = 2 :=
by sorry

end NUMINAMATH_CALUDE_number_of_true_statements_number_of_true_statements_is_correct_l283_28352


namespace NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l283_28321

theorem one_fourth_of_eight_point_eight : (8.8 : ℚ) / 4 = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_eight_point_eight_l283_28321


namespace NUMINAMATH_CALUDE_first_five_valid_numbers_l283_28329

def random_table : List (List Nat) := [
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67, 21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 56, 67, 19, 98, 10, 50, 71, 75, 12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38, 15, 51, 00, 13, 42, 99, 66, 02, 79, 54]
]

def start_row : Nat := 8
def start_col : Nat := 7
def max_bag_number : Nat := 799

def is_valid_number (n : Nat) : Bool :=
  n <= max_bag_number

def find_valid_numbers (table : List (List Nat)) (row : Nat) (col : Nat) (count : Nat) : List Nat :=
  sorry

theorem first_five_valid_numbers :
  find_valid_numbers random_table start_row start_col 5 = [785, 667, 199, 507, 175] :=
sorry

end NUMINAMATH_CALUDE_first_five_valid_numbers_l283_28329


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l283_28322

def num_vegetables : ℕ := 4
def num_plots : ℕ := 3
def num_to_select : ℕ := 3

theorem vegetable_planting_methods :
  (num_vegetables - 1).choose (num_to_select - 1) * num_to_select.factorial = 18 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l283_28322


namespace NUMINAMATH_CALUDE_same_color_probability_l283_28376

/-- The probability of drawing two balls of the same color from a bag containing
    8 green balls, 5 red balls, and 3 blue balls, with replacement. -/
theorem same_color_probability (green red blue : ℕ) (total : ℕ) :
  green = 8 → red = 5 → blue = 3 → total = green + red + blue →
  (green^2 + red^2 + blue^2 : ℚ) / total^2 = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l283_28376


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l283_28354

/-- Given a cone with vertex S, prove that its lateral surface area is 40√2π -/
theorem cone_lateral_surface_area (S : Point) (A B : Point) :
  let cos_angle_SA_SB : ℝ := 7/8
  let angle_SA_base : ℝ := π/4  -- 45° in radians
  let area_SAB : ℝ := 5 * Real.sqrt 15
  -- Define the lateral surface area
  let lateral_surface_area : ℝ := 
    let SA : ℝ := 4 * Real.sqrt 5  -- derived from area_SAB and cos_angle_SA_SB
    let base_radius : ℝ := SA * Real.sqrt 2 / 2
    π * base_radius * SA
  lateral_surface_area = 40 * Real.sqrt 2 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l283_28354


namespace NUMINAMATH_CALUDE_smallest_value_l283_28317

def x : ℝ := 4
def y : ℝ := 2

theorem smallest_value : 
  min (x + y) (min (x * y) (min (x - y) (min (x / y) (y / x)))) = y / x :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_l283_28317


namespace NUMINAMATH_CALUDE_bowls_sold_calculation_l283_28394

def total_bowls : ℕ := 114
def cost_per_bowl : ℚ := 13
def sell_price_per_bowl : ℚ := 17
def percentage_gain : ℚ := 23.88663967611336

theorem bowls_sold_calculation :
  ∃ (x : ℕ), 
    x ≤ total_bowls ∧ 
    (x : ℚ) * sell_price_per_bowl = 
      (total_bowls : ℚ) * cost_per_bowl * (1 + percentage_gain / 100) ∧
    x = 108 := by
  sorry

end NUMINAMATH_CALUDE_bowls_sold_calculation_l283_28394


namespace NUMINAMATH_CALUDE_constant_term_zero_implies_m_negative_one_l283_28369

/-- The quadratic equation in x with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * x - m^2 + 1

/-- The constant term of the quadratic equation -/
def constant_term (m : ℝ) : ℝ := quadratic_equation m 0

theorem constant_term_zero_implies_m_negative_one :
  constant_term (-1) = 0 ∧ (∀ m : ℝ, constant_term m = 0 → m = -1) :=
sorry

end NUMINAMATH_CALUDE_constant_term_zero_implies_m_negative_one_l283_28369


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l283_28310

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {3, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l283_28310


namespace NUMINAMATH_CALUDE_german_students_count_l283_28373

theorem german_students_count (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : 
  total = 78 → french = 41 → both = 9 → neither = 24 → 
  ∃ german : ℕ, german = 22 ∧ total = french + german - both + neither :=
by sorry

end NUMINAMATH_CALUDE_german_students_count_l283_28373


namespace NUMINAMATH_CALUDE_arrangements_with_A_must_go_arrangements_A_B_not_Japan_l283_28320

-- Define the number of volunteers
def total_volunteers : ℕ := 6

-- Define the number of people to be selected
def selected_people : ℕ := 4

-- Define the number of pavilions
def num_pavilions : ℕ := 4

-- Function to calculate the number of arrangements when one person must be included
def arrangements_with_one_person (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose (n - 1) (k - 1)) * (Nat.factorial k)

-- Function to calculate the number of arrangements when two people cannot go to a specific pavilion
def arrangements_with_restriction (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose k 1) * (Nat.choose (n - 1) (k - 1)) * (Nat.factorial (k - 1))

-- Theorem for the first question
theorem arrangements_with_A_must_go :
  arrangements_with_one_person total_volunteers selected_people = 240 := by
  sorry

-- Theorem for the second question
theorem arrangements_A_B_not_Japan :
  arrangements_with_restriction total_volunteers selected_people = 240 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_A_must_go_arrangements_A_B_not_Japan_l283_28320


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l283_28314

/-- A hyperbola with foci on the x-axis and asymptotes y = ±√3x has eccentricity 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = Real.sqrt 3) :
  let e := Real.sqrt (1 + (b^2 / a^2))
  e = 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l283_28314


namespace NUMINAMATH_CALUDE_tan_alpha_and_expression_l283_28372

theorem tan_alpha_and_expression (α : Real) 
  (h : Real.tan (π / 4 + α) = 1 / 2) : 
  Real.tan α = -1 / 3 ∧ 
  (Real.sin (2 * α + 2 * π) - Real.sin (π / 2 - α) ^ 2) / 
  (1 - Real.cos (π - 2 * α) + Real.sin α ^ 2) = -15 / 19 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_expression_l283_28372


namespace NUMINAMATH_CALUDE_sean_needs_six_packs_l283_28350

/-- Calculates the number of light bulb packs needed given the number of bulbs required for each room --/
def calculate_packs_needed (bedroom bathroom kitchen basement : ℕ) : ℕ :=
  let other_rooms_total := bedroom + bathroom + kitchen + basement
  let garage := other_rooms_total / 2
  let total_bulbs := other_rooms_total + garage
  (total_bulbs + 1) / 2

/-- Proves that Sean needs 6 packs of light bulbs --/
theorem sean_needs_six_packs :
  calculate_packs_needed 2 1 1 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sean_needs_six_packs_l283_28350


namespace NUMINAMATH_CALUDE_M_in_second_and_fourth_quadrants_l283_28345

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 * p.2 < 0}

theorem M_in_second_and_fourth_quadrants :
  ∀ p ∈ M, (p.1 > 0 ∧ p.2 < 0) ∨ (p.1 < 0 ∧ p.2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_M_in_second_and_fourth_quadrants_l283_28345


namespace NUMINAMATH_CALUDE_age_of_other_man_l283_28337

theorem age_of_other_man (n : ℕ) (avg_increase : ℝ) (age_one_man : ℕ) (avg_age_women : ℝ) :
  n = 8 ∧ 
  avg_increase = 2 ∧ 
  age_one_man = 20 ∧ 
  avg_age_women = 29 →
  ∃ (original_avg : ℝ) (age_other_man : ℕ),
    n * (original_avg + avg_increase) = n * original_avg + 2 * avg_age_women - (age_one_man + age_other_man) ∧
    age_other_man = 22 :=
by sorry

end NUMINAMATH_CALUDE_age_of_other_man_l283_28337


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l283_28339

theorem jelly_bean_probability (red orange blue : ℝ) (h1 : red = 0.25) (h2 : orange = 0.4) (h3 : blue = 0.1) :
  ∃ yellow : ℝ, yellow = 0.25 ∧ red + orange + blue + yellow = 1 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l283_28339


namespace NUMINAMATH_CALUDE_age_difference_l283_28323

theorem age_difference (A B : ℕ) : B = 34 → A + 10 = 2 * (B - 10) → A - B = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l283_28323


namespace NUMINAMATH_CALUDE_equation_has_two_distinct_roots_l283_28391

theorem equation_has_two_distinct_roots (a b : ℝ) (h : a ≠ b) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
  (x₁ + a) * (x₁ + b) = 2 * x₁ + a + b ∧
  (x₂ + a) * (x₂ + b) = 2 * x₂ + a + b :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_distinct_roots_l283_28391


namespace NUMINAMATH_CALUDE_email_difference_l283_28333

def morning_emails : ℕ := 10
def afternoon_emails : ℕ := 7
def evening_emails : ℕ := 17

theorem email_difference : morning_emails - afternoon_emails = 3 := by
  sorry

end NUMINAMATH_CALUDE_email_difference_l283_28333


namespace NUMINAMATH_CALUDE_line_y_intercept_l283_28316

/-- A straight line in the xy-plane with slope 4 passing through (50, 300) has y-intercept 100 -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) : 
  m = 4 → x = 50 → y = 300 → y = m * x + b → b = 100 := by sorry

end NUMINAMATH_CALUDE_line_y_intercept_l283_28316


namespace NUMINAMATH_CALUDE_f_max_value_l283_28359

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := (S n : ℚ) / ((n + 32 : ℚ) * (S (n + 1) : ℚ))

theorem f_max_value :
  (∀ n : ℕ, f n ≤ 1/50) ∧ (∃ n : ℕ, f n = 1/50) := by sorry

end NUMINAMATH_CALUDE_f_max_value_l283_28359


namespace NUMINAMATH_CALUDE_ad_value_l283_28378

/-- Given two-digit numbers ab and cd, and that 1ab is a three-digit number -/
def two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

/-- The theorem statement -/
theorem ad_value (a b c d : ℕ) 
  (h1 : two_digit (10 * a + b))
  (h2 : two_digit (10 * c + d))
  (h3 : three_digit (100 + 10 * a + b))
  (h4 : 10 * a + b = 10 * c + d + 24)
  (h5 : 100 + 10 * a + b = 100 * c + 10 * d + 1 + 15) :
  10 * a + d = 32 := by
sorry

end NUMINAMATH_CALUDE_ad_value_l283_28378


namespace NUMINAMATH_CALUDE_total_cost_for_index_finger_rings_l283_28363

def cost_per_ring : ℕ := 24
def index_fingers_per_person : ℕ := 2

theorem total_cost_for_index_finger_rings :
  cost_per_ring * index_fingers_per_person = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_index_finger_rings_l283_28363


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l283_28308

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 35) (h3 : y = 3 * x) :
  y = -21 → x = -10.9375 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l283_28308


namespace NUMINAMATH_CALUDE_grain_output_scientific_notation_l283_28385

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem grain_output_scientific_notation :
  toScientificNotation 736000000 = ScientificNotation.mk 7.36 8 (by norm_num) := by
  sorry

end NUMINAMATH_CALUDE_grain_output_scientific_notation_l283_28385


namespace NUMINAMATH_CALUDE_interest_rate_is_zero_l283_28346

/-- The interest rate for a TV purchase with installment payments -/
def interest_rate_tv_purchase (tv_price : ℕ) (num_installments : ℕ) (installment_amount : ℕ) (last_installment : ℕ) : ℚ :=
  if tv_price = 60000 ∧ 
     num_installments = 20 ∧ 
     installment_amount = 1000 ∧ 
     last_installment = 59000 ∧
     tv_price - installment_amount = last_installment
  then 0
  else 1 -- arbitrary non-zero value for other cases

/-- Theorem stating that the interest rate is 0% for the given TV purchase conditions -/
theorem interest_rate_is_zero :
  interest_rate_tv_purchase 60000 20 1000 59000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_zero_l283_28346


namespace NUMINAMATH_CALUDE_height_decreases_as_vertex_angle_increases_l283_28353

-- Define an isosceles triangle
structure IsoscelesTriangle where
  a : ℝ  -- Length of equal sides
  φ : ℝ  -- Half of the vertex angle
  h : ℝ  -- Height dropped to the base
  h_eq : h = a * Real.cos φ

-- Theorem statement
theorem height_decreases_as_vertex_angle_increases
  (t1 t2 : IsoscelesTriangle)
  (h_same_side : t1.a = t2.a)
  (h_larger_angle : t1.φ < t2.φ)
  (h_angle_range : 0 < t1.φ ∧ t2.φ < Real.pi / 2) :
  t2.h < t1.h :=
by
  sorry

end NUMINAMATH_CALUDE_height_decreases_as_vertex_angle_increases_l283_28353


namespace NUMINAMATH_CALUDE_fg_length_l283_28330

/-- Represents a parallelogram ABCD and a right triangle DFG with specific properties -/
structure GeometricFigures where
  AB : ℝ
  AD : ℝ
  DG : ℝ
  area_equality : AB * AD = 1/2 * DG * AB

/-- The length of FG in the given geometric configuration is 8 -/
theorem fg_length (figures : GeometricFigures) 
  (h1 : figures.AB = 8)
  (h2 : figures.AD = 3)
  (h3 : figures.DG = 6) :
  figures.AB = 8 := by sorry

end NUMINAMATH_CALUDE_fg_length_l283_28330


namespace NUMINAMATH_CALUDE_trail_distribution_count_l283_28358

/-- The number of ways to distribute 4 people familiar with trails into two groups of 2 each -/
def trail_distribution_ways : ℕ := Nat.choose 4 2

/-- Theorem stating that the number of ways to distribute 4 people familiar with trails
    into two groups of 2 each is equal to 6 -/
theorem trail_distribution_count : trail_distribution_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_trail_distribution_count_l283_28358


namespace NUMINAMATH_CALUDE_xyz_bound_l283_28318

theorem xyz_bound (x y z : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (hz_pos : z > 0)
  (hx_bound : x ≤ 2) (hy_bound : y ≤ 3) (hsum : x + y + z = 11) :
  x * y * z ≤ 36 := by
  sorry

end NUMINAMATH_CALUDE_xyz_bound_l283_28318


namespace NUMINAMATH_CALUDE_total_eggs_supplied_in_week_l283_28315

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days in a week --/
def daysInWeek : ℕ := 7

/-- Represents the number of weekdays in a week --/
def weekdaysInWeek : ℕ := 5

/-- Represents the number of odd days in a week --/
def oddDaysInWeek : ℕ := 3

/-- Represents the number of even days in a week --/
def evenDaysInWeek : ℕ := 4

/-- Represents the daily egg supply to the first store --/
def firstStoreSupply : ℕ := 5 * dozen

/-- Represents the daily egg supply to the second store on weekdays --/
def secondStoreSupply : ℕ := 30

/-- Represents the egg supply to the third store on odd days --/
def thirdStoreOddSupply : ℕ := 25 * dozen

/-- Represents the egg supply to the third store on even days --/
def thirdStoreEvenSupply : ℕ := 15 * dozen

/-- Theorem stating the total number of eggs supplied in a week --/
theorem total_eggs_supplied_in_week :
  firstStoreSupply * daysInWeek +
  secondStoreSupply * weekdaysInWeek +
  thirdStoreOddSupply * oddDaysInWeek +
  thirdStoreEvenSupply * evenDaysInWeek = 2190 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_supplied_in_week_l283_28315


namespace NUMINAMATH_CALUDE_ball_pit_count_l283_28303

theorem ball_pit_count : ∃ (total : ℕ), 
  let red := total / 4
  let non_red := total - red
  let blue := non_red / 5
  let neither_red_nor_blue := total - red - blue
  neither_red_nor_blue = 216 ∧ total = 360 := by
sorry

end NUMINAMATH_CALUDE_ball_pit_count_l283_28303


namespace NUMINAMATH_CALUDE_corn_planting_ratio_l283_28325

/-- Represents the problem of calculating the ratio of dinner cost to total earnings for corn planting kids. -/
theorem corn_planting_ratio :
  -- Define constants based on the problem conditions
  let ears_per_row : ℕ := 70
  let seeds_per_bag : ℕ := 48
  let seeds_per_ear : ℕ := 2
  let pay_per_row : ℚ := 3/2  -- $1.5 expressed as a rational number
  let dinner_cost : ℚ := 36
  let bags_used : ℕ := 140

  -- Calculate total ears planted
  let total_ears : ℕ := (bags_used * seeds_per_bag) / seeds_per_ear

  -- Calculate rows planted
  let rows_planted : ℕ := total_ears / ears_per_row

  -- Calculate total earnings
  let total_earned : ℚ := pay_per_row * rows_planted

  -- The ratio of dinner cost to total earnings is 1/2
  dinner_cost / total_earned = 1/2 :=
by
  sorry  -- Proof is omitted as per instructions

end NUMINAMATH_CALUDE_corn_planting_ratio_l283_28325


namespace NUMINAMATH_CALUDE_tv_selection_combinations_l283_28393

def type_a_count : ℕ := 4
def type_b_count : ℕ := 5
def total_selection : ℕ := 3

theorem tv_selection_combinations : 
  (Nat.choose type_a_count 1 * Nat.choose type_b_count 2) + 
  (Nat.choose type_a_count 2 * Nat.choose type_b_count 1) = 70 := by
  sorry

end NUMINAMATH_CALUDE_tv_selection_combinations_l283_28393


namespace NUMINAMATH_CALUDE_unique_four_digit_int_l283_28307

/-- Represents a four-digit positive integer -/
structure FourDigitInt where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_pos : a > 0
  a_lt_10 : a < 10
  b_lt_10 : b < 10
  c_lt_10 : c < 10
  d_lt_10 : d < 10

def to_int (n : FourDigitInt) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

theorem unique_four_digit_int :
  ∃! (n : FourDigitInt),
    n.a + n.b + n.c + n.d = 16 ∧
    n.b + n.c = 10 ∧
    n.a - n.d = 2 ∧
    (to_int n) % 9 = 0 ∧
    to_int n = 4622 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_int_l283_28307


namespace NUMINAMATH_CALUDE_cloth_cost_theorem_l283_28331

/-- The total cost of cloth given the length and price per meter -/
def total_cost (length : ℝ) (price_per_meter : ℝ) : ℝ :=
  length * price_per_meter

/-- Theorem: The total cost of 9.25 meters of cloth at $44 per meter is $407 -/
theorem cloth_cost_theorem :
  total_cost 9.25 44 = 407 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_theorem_l283_28331


namespace NUMINAMATH_CALUDE_star_minus_emilio_sum_equals_104_l283_28348

def star_list := List.range 40 |>.map (· + 1)

def replace_three_with_two (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list := star_list.map replace_three_with_two

theorem star_minus_emilio_sum_equals_104 :
  star_list.sum - emilio_list.sum = 104 := by
  sorry

end NUMINAMATH_CALUDE_star_minus_emilio_sum_equals_104_l283_28348


namespace NUMINAMATH_CALUDE_min_value_expression_l283_28399

theorem min_value_expression (x y : ℝ) (h1 : x^2 + y^2 = 3) (h2 : |x| ≠ |y|) :
  1 / (2*x + y)^2 + 4 / (x - 2*y)^2 ≥ 3/5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l283_28399


namespace NUMINAMATH_CALUDE_point_on_y_axis_l283_28361

theorem point_on_y_axis (m : ℝ) :
  (m + 1 = 0) → ((m + 1, m + 4) : ℝ × ℝ) = (0, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l283_28361


namespace NUMINAMATH_CALUDE_oranges_per_box_l283_28398

/-- Given 45 oranges and 9 boxes, prove that the number of oranges per box is 5 -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) 
  (h1 : total_oranges = 45) (h2 : num_boxes = 9) : 
  total_oranges / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l283_28398


namespace NUMINAMATH_CALUDE_caramel_candy_probability_l283_28375

/-- The probability of selecting a caramel-flavored candy from a set of candies -/
theorem caramel_candy_probability 
  (total_candies : ℕ) 
  (caramel_candies : ℕ) 
  (lemon_candies : ℕ) 
  (h1 : total_candies = caramel_candies + lemon_candies)
  (h2 : caramel_candies = 3)
  (h3 : lemon_candies = 4) :
  (caramel_candies : ℚ) / total_candies = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_caramel_candy_probability_l283_28375


namespace NUMINAMATH_CALUDE_translation_result_l283_28367

/-- Represents a point in the 2D Cartesian coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point downward by a given number of units -/
def translateDown (p : Point) (units : ℝ) : Point :=
  { x := p.x, y := p.y - units }

/-- Translates a point to the right by a given number of units -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

/-- The main theorem stating the result of the translation -/
theorem translation_result : 
  let A : Point := { x := -2, y := 2 }
  let B : Point := translateRight (translateDown A 4) 3
  B.x = 1 ∧ B.y = -2 := by sorry

end NUMINAMATH_CALUDE_translation_result_l283_28367


namespace NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l283_28312

theorem intersection_locus_is_ellipse :
  ∀ (x y u : ℝ),
  (2 * u * x - 3 * y - 4 * u = 0) →
  (x - 3 * u * y + 4 = 0) →
  (x^2 / 16 + y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_is_ellipse_l283_28312


namespace NUMINAMATH_CALUDE_circumcenter_distance_theorem_l283_28382

-- Define a structure for a triangle with its circumcircle properties
structure TriangleWithCircumcircle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles opposite to sides a, b, c respectively
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Radius of the circumscribed circle
  r : ℝ
  -- Distances from circumcenter to sides a, b, c respectively
  pa : ℝ
  pb : ℝ
  pc : ℝ
  -- Conditions for a valid triangle
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π
  -- Relationship between sides and angles
  sine_law_a : a = 2 * r * Real.sin α
  sine_law_b : b = 2 * r * Real.sin β
  sine_law_c : c = 2 * r * Real.sin γ

-- Theorem statement
theorem circumcenter_distance_theorem (t : TriangleWithCircumcircle) :
  t.pa * Real.sin t.α + t.pb * Real.sin t.β + t.pc * Real.sin t.γ =
  2 * t.r * Real.sin t.α * Real.sin t.β * Real.sin t.γ :=
by sorry

end NUMINAMATH_CALUDE_circumcenter_distance_theorem_l283_28382


namespace NUMINAMATH_CALUDE_volume_ADBFE_l283_28326

-- Define the pyramid ABCD
def Pyramid (A B C D : Point) : Set Point := sorry

-- Define the volume of a set of points
def volume : Set Point → ℝ := sorry

-- Define a median of a triangle
def isMedian (E : Point) (triangle : Set Point) : Prop := sorry

-- Define a midpoint of a line segment
def isMidpoint (F : Point) (segment : Set Point) : Prop := sorry

theorem volume_ADBFE (A B C D E F : Point) 
  (hpyramid : Pyramid A B C D)
  (hmedian : isMedian E {A, B, C})
  (hmidpoint : isMidpoint F {D, C})
  (hvolume : volume (Pyramid A B C D) = 40) :
  volume {A, D, B, F, E} = (3/4) * volume (Pyramid A B C D) := by
  sorry

end NUMINAMATH_CALUDE_volume_ADBFE_l283_28326


namespace NUMINAMATH_CALUDE_first_file_size_is_80_l283_28377

/-- Calculates the size of the first file given the internet speed, total download time, and sizes of two other files. -/
def first_file_size (speed : ℝ) (time : ℝ) (file2 : ℝ) (file3 : ℝ) : ℝ :=
  speed * time * 60 - file2 - file3

/-- Proves that given the specified conditions, the size of the first file is 80 megabits. -/
theorem first_file_size_is_80 :
  first_file_size 2 2 90 70 = 80 := by
  sorry

end NUMINAMATH_CALUDE_first_file_size_is_80_l283_28377


namespace NUMINAMATH_CALUDE_black_pens_count_l283_28396

/-- The number of black pens initially in the jar -/
def initial_black_pens : ℕ := 21

theorem black_pens_count :
  let initial_blue_pens : ℕ := 9
  let initial_red_pens : ℕ := 6
  let removed_blue_pens : ℕ := 4
  let removed_black_pens : ℕ := 7
  let remaining_pens : ℕ := 25
  initial_blue_pens + initial_black_pens + initial_red_pens - 
    (removed_blue_pens + removed_black_pens) = remaining_pens →
  initial_black_pens = 21 := by
sorry

end NUMINAMATH_CALUDE_black_pens_count_l283_28396


namespace NUMINAMATH_CALUDE_sine_graph_shift_l283_28313

theorem sine_graph_shift (x : ℝ) :
  (3 * Real.sin (2 * (x + π / 8))) = (3 * Real.sin (2 * x + π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l283_28313


namespace NUMINAMATH_CALUDE_hyperbola_through_points_l283_28324

/-- The standard form of a hyperbola passing through two given points -/
theorem hyperbola_through_points :
  let P₁ : ℝ × ℝ := (3, -4 * Real.sqrt 2)
  let P₂ : ℝ × ℝ := (9/4, 5)
  let hyperbola (x y : ℝ) := 49 * x^2 - 7 * y^2 = 113
  (hyperbola P₁.1 P₁.2) ∧ (hyperbola P₂.1 P₂.2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_through_points_l283_28324


namespace NUMINAMATH_CALUDE_l_shape_surface_area_l283_28388

/-- Represents the 'L' shaped solid described in the problem -/
structure LShape where
  base_cubes : Nat
  column_cubes : Nat
  base_length : Nat
  base_width : Nat
  extension_length : Nat

/-- Calculates the surface area of the 'L' shaped solid -/
def surface_area (shape : LShape) : Nat :=
  let base_area := shape.base_cubes
  let top_exposed := shape.base_cubes - 1
  let column_sides := 4 * shape.column_cubes
  let column_top := 1
  let base_perimeter := 2 * (shape.base_length + shape.base_width + 2 * shape.extension_length)
  top_exposed + column_sides + column_top + base_perimeter

/-- The specific 'L' shape described in the problem -/
def problem_shape : LShape := {
  base_cubes := 8
  column_cubes := 7
  base_length := 3
  base_width := 2
  extension_length := 2
}

theorem l_shape_surface_area :
  surface_area problem_shape = 58 := by sorry

end NUMINAMATH_CALUDE_l_shape_surface_area_l283_28388


namespace NUMINAMATH_CALUDE_fish_count_after_transfer_l283_28379

/-- The total number of fish after Lilly gives some to Jack -/
def total_fish (lilly_initial : ℕ) (rosy : ℕ) (jack_initial : ℕ) (transfer : ℕ) : ℕ :=
  (lilly_initial - transfer) + rosy + (jack_initial + transfer)

/-- Theorem stating the total number of fish after the transfer -/
theorem fish_count_after_transfer :
  total_fish 10 9 15 2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_after_transfer_l283_28379


namespace NUMINAMATH_CALUDE_first_quarter_spending_river_town_l283_28368

/-- The spending during the first quarter of a year, given the initial and end-of-quarter spending -/
def first_quarter_spending (initial_spending end_of_quarter_spending : ℝ) : ℝ :=
  end_of_quarter_spending - initial_spending

/-- Theorem: The spending during the first quarter is 3.1 million dollars -/
theorem first_quarter_spending_river_town : 
  first_quarter_spending 0 3.1 = 3.1 := by
  sorry

end NUMINAMATH_CALUDE_first_quarter_spending_river_town_l283_28368


namespace NUMINAMATH_CALUDE_money_split_proof_l283_28300

/-- Given a sum of money split in the ratio 2:3:4, where the smallest share is $50, prove that the total amount shared is $225. -/
theorem money_split_proof (total : ℕ) (parker richie jaime : ℕ) : 
  parker + richie + jaime = total →
  parker = 50 →
  2 * richie = 3 * parker →
  2 * jaime = 4 * parker →
  total = 225 := by
sorry

end NUMINAMATH_CALUDE_money_split_proof_l283_28300


namespace NUMINAMATH_CALUDE_expression_range_l283_28338

theorem expression_range (x y : ℝ) (h : (x - 1)^2 + (y - 4)^2 = 1) :
  0 ≤ (x*y - x) / (x^2 + (y - 1)^2) ∧ (x*y - x) / (x^2 + (y - 1)^2) ≤ 12/25 := by
  sorry

end NUMINAMATH_CALUDE_expression_range_l283_28338


namespace NUMINAMATH_CALUDE_stool_sticks_calculation_l283_28362

/-- The number of sticks of wood a chair makes -/
def chair_sticks : ℕ := 6

/-- The number of sticks of wood a table makes -/
def table_sticks : ℕ := 9

/-- The number of sticks Mary needs to burn per hour to stay warm -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped up -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped up -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chopped up -/
def stools_chopped : ℕ := 4

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

/-- The number of sticks of wood a stool makes -/
def stool_sticks : ℕ := 2

theorem stool_sticks_calculation :
  stool_sticks * stools_chopped = 
    hours_warm * sticks_per_hour - 
    (chair_sticks * chairs_chopped + table_sticks * tables_chopped) :=
by sorry

end NUMINAMATH_CALUDE_stool_sticks_calculation_l283_28362


namespace NUMINAMATH_CALUDE_probability_age_20_to_40_l283_28334

theorem probability_age_20_to_40 (total : ℕ) (below_20 : ℕ) (between_20_30 : ℕ) (between_30_40 : ℕ) :
  total = 350 →
  below_20 = 120 →
  between_20_30 = 105 →
  between_30_40 = 85 →
  (between_20_30 + between_30_40 : ℚ) / total = 19 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_age_20_to_40_l283_28334


namespace NUMINAMATH_CALUDE_hypotenuse_squared_length_l283_28395

/-- The ellipse in which the triangle is inscribed -/
def ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The right triangle inscribed in the ellipse -/
structure InscribedRightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A = (0, 1)
  h_B_on_x_axis : B.2 = 0
  h_C_on_x_axis : C.2 = 0
  h_A_on_ellipse : ellipse A.1 A.2
  h_B_on_ellipse : ellipse B.1 B.2
  h_C_on_ellipse : ellipse C.1 C.2
  h_right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

theorem hypotenuse_squared_length 
  (t : InscribedRightTriangle) : (t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_squared_length_l283_28395


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l283_28351

def set_A : Set (ℝ × ℝ) := {p | p.2 = p.1^2}
def set_B : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt p.1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {(0, 0), (1, 1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l283_28351


namespace NUMINAMATH_CALUDE_inequality_solutions_l283_28364

theorem inequality_solutions :
  (∀ x : ℝ, 3 + 2*x > -x - 6 ↔ x > -3) ∧
  (∀ x : ℝ, (2*x + 1 ≤ x + 3 ∧ (2*x + 1) / 3 > 1) ↔ 1 < x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l283_28364


namespace NUMINAMATH_CALUDE_sqrt_2a_plus_b_equals_3_l283_28319

theorem sqrt_2a_plus_b_equals_3 (a b : ℝ) 
  (h1 : (2*a - 1) = 9)
  (h2 : a - 2*b + 1 = 8) :
  Real.sqrt (2*a + b) = 3 := by
sorry

end NUMINAMATH_CALUDE_sqrt_2a_plus_b_equals_3_l283_28319


namespace NUMINAMATH_CALUDE_no_definitive_inference_l283_28347

-- Define the sets
variable (Mem Ens Vee : Set α)

-- Define the conditions
variable (h1 : ∃ x, x ∈ Mem ∧ x ∉ Ens)
variable (h2 : Ens ∩ Vee = ∅)

-- Define the potential inferences
def inference_A := ∃ x, x ∈ Mem ∧ x ∉ Vee
def inference_B := ∃ x, x ∈ Vee ∧ x ∉ Mem
def inference_C := Mem ∩ Vee = ∅
def inference_D := ∃ x, x ∈ Mem ∧ x ∈ Vee

-- The theorem to prove
theorem no_definitive_inference :
  ¬(inference_A Mem Vee) ∧
  ¬(inference_B Mem Vee) ∧
  ¬(inference_C Mem Vee) ∧
  ¬(inference_D Mem Vee) :=
sorry

end NUMINAMATH_CALUDE_no_definitive_inference_l283_28347


namespace NUMINAMATH_CALUDE_counterfeit_coin_weighings_l283_28357

/-- Represents a weighing operation on a balance scale -/
def Weighing := List Nat → List Nat → Bool

/-- Represents a strategy for finding the counterfeit coin -/
def Strategy := List Nat → List (List Nat × List Nat)

/-- The number of coins -/
def n : Nat := 15

/-- The maximum number of weighings needed -/
def max_weighings : Nat := 3

theorem counterfeit_coin_weighings :
  ∃ (s : Strategy),
    ∀ (counterfeit : Fin n),
      ∀ (w : Weighing),
        (∀ i j : Fin n, i ≠ j → w [i.val] [j.val] = true) →
        (∀ i : Fin n, i ≠ counterfeit → w [i.val] [counterfeit.val] = false) →
        (s (List.range n)).length ≤ max_weighings ∧
        ∃ (result : Fin n), result = counterfeit := by sorry

end NUMINAMATH_CALUDE_counterfeit_coin_weighings_l283_28357


namespace NUMINAMATH_CALUDE_cafe_pricing_theorem_l283_28371

/-- Represents the pricing structure of a café -/
structure CafePrices where
  sandwich : ℝ
  coffee : ℝ
  pie : ℝ

/-- The café's pricing satisfies the given conditions -/
def satisfies_conditions (p : CafePrices) : Prop :=
  4 * p.sandwich + 9 * p.coffee + p.pie = 4.30 ∧
  7 * p.sandwich + 14 * p.coffee + p.pie = 7.00

/-- Calculates the total cost for a given order -/
def order_cost (p : CafePrices) (sandwiches coffees pies : ℕ) : ℝ :=
  p.sandwich * sandwiches + p.coffee * coffees + p.pie * pies

/-- Theorem stating that the cost of 11 sandwiches, 23 coffees, and 2 pies is $18.87 -/
theorem cafe_pricing_theorem (p : CafePrices) :
  satisfies_conditions p →
  order_cost p 11 23 2 = 18.87 := by
  sorry

end NUMINAMATH_CALUDE_cafe_pricing_theorem_l283_28371


namespace NUMINAMATH_CALUDE_intersection_point_equality_l283_28306

/-- Given a system of linear equations and its solution, prove that the intersection
    point of two related lines is the same as the solution. -/
theorem intersection_point_equality (x y : ℝ) :
  x - y = -5 →
  x + 2*y = -2 →
  x = -4 →
  y = 1 →
  ∃! (x' y' : ℝ), y' = x' + 5 ∧ y' = -1/2 * x' - 1 ∧ x' = -4 ∧ y' = 1 :=
by sorry


end NUMINAMATH_CALUDE_intersection_point_equality_l283_28306


namespace NUMINAMATH_CALUDE_h_j_h_3_equals_277_l283_28384

def h (x : ℝ) : ℝ := 5 * x + 2

def j (x : ℝ) : ℝ := 3 * x + 4

theorem h_j_h_3_equals_277 : h (j (h 3)) = 277 := by
  sorry

end NUMINAMATH_CALUDE_h_j_h_3_equals_277_l283_28384


namespace NUMINAMATH_CALUDE_sum_26_35_in_base7_l283_28335

/-- Converts a natural number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 26 and 35 in base 10, when converted to base 7, equals 85 -/
theorem sum_26_35_in_base7 : toBase7 (26 + 35) = 85 := by sorry

end NUMINAMATH_CALUDE_sum_26_35_in_base7_l283_28335


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_side_length_l283_28365

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the circle
def Circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }

-- Define the theorem
theorem cyclic_quadrilateral_side_length 
  (ABCD : Quadrilateral) 
  (inscribed : ABCD.A ∈ Circle ∧ ABCD.B ∈ Circle ∧ ABCD.C ∈ Circle ∧ ABCD.D ∈ Circle) 
  (perp_diagonals : (ABCD.A.1 - ABCD.C.1) * (ABCD.B.1 - ABCD.D.1) + 
                    (ABCD.A.2 - ABCD.C.2) * (ABCD.B.2 - ABCD.D.2) = 0)
  (AB_length : (ABCD.A.1 - ABCD.B.1)^2 + (ABCD.A.2 - ABCD.B.2)^2 = 9) :
  (ABCD.C.1 - ABCD.D.1)^2 + (ABCD.C.2 - ABCD.D.2)^2 = 7 :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_side_length_l283_28365


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l283_28356

-- Define set A
def A : Set ℝ := {x | x^2 + x - 12 < 0}

-- Define set B
def B : Set ℝ := {x | Real.sqrt (x + 2) < 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l283_28356


namespace NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l283_28341

/-- Given two hanging spheres with masses m₁ and m₂, where the tension in the upper thread (T_B)
    is three times the tension in the lower thread (T_H), prove that the ratio m₁/m₂ = 2. -/
theorem hanging_spheres_mass_ratio
  (m₁ m₂ : ℝ) -- masses of the spheres
  (T_B T_H : ℝ) -- tensions in the upper and lower threads
  (h1 : T_B = 3 * T_H) -- condition: upper tension is 3 times lower tension
  (h2 : T_H = m₂ * (9.8 : ℝ)) -- force balance for bottom sphere
  (h3 : T_B = m₁ * (9.8 : ℝ) + T_H) -- force balance for top sphere
  : m₁ / m₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l283_28341


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l283_28304

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l283_28304


namespace NUMINAMATH_CALUDE_zero_existence_l283_28336

open Real

-- Define the differential equation
def is_solution (y : ℝ → ℝ) : Prop :=
  ∀ x, (x^2 + 9) * (deriv^[2] y x) + (x^2 + 4) * y x = 0

-- Define the theorem
theorem zero_existence (y : ℝ → ℝ) 
  (h_sol : is_solution y) 
  (h_init1 : y 0 = 0) 
  (h_init2 : deriv y 0 = 1) :
  ∃ x ∈ Set.Icc (Real.sqrt (63/53) * π) (3*π/2), y x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_existence_l283_28336


namespace NUMINAMATH_CALUDE_rectangle_side_length_l283_28355

/-- Given a square with side length 5 and a rectangle with one side 4,
    if they have the same area, then the other side of the rectangle is 6.25 -/
theorem rectangle_side_length (square_side : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side = 5 →
  rectangle_width = 4 →
  square_side * square_side = rectangle_width * rectangle_length →
  rectangle_length = 6.25 := by
sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l283_28355


namespace NUMINAMATH_CALUDE_root_values_l283_28305

theorem root_values (p q r s k : ℂ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) 
  (h1 : p * k^3 + q * k^2 + r * k + s = 0)
  (h2 : q * k^3 + r * k^2 + s * k + p = 0)
  (h3 : 3 * p * k^2 + 2 * q * k + r = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_root_values_l283_28305


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l283_28301

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3  -- 60 degrees in radians
  a = (2, 0) →
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt ((a.1^2 + a.2^2) * (b.1^2 + b.2^2))) = Real.cos angle →
  ‖b‖ = 1 :=
by sorry

#check vector_magnitude_problem

end NUMINAMATH_CALUDE_vector_magnitude_problem_l283_28301


namespace NUMINAMATH_CALUDE_pedro_extra_squares_l283_28381

theorem pedro_extra_squares (jesus_squares linden_squares pedro_squares : ℕ) 
  (h1 : jesus_squares = 60)
  (h2 : linden_squares = 75)
  (h3 : pedro_squares = 200) :
  pedro_squares - (jesus_squares + linden_squares) = 65 := by
  sorry

end NUMINAMATH_CALUDE_pedro_extra_squares_l283_28381


namespace NUMINAMATH_CALUDE_wall_height_proof_l283_28390

/-- Given a wall and bricks with specified dimensions, proves the height of the wall. -/
theorem wall_height_proof (wall_length : Real) (wall_width : Real) (num_bricks : Nat)
  (brick_length : Real) (brick_width : Real) (brick_height : Real)
  (h_wall_length : wall_length = 8)
  (h_wall_width : wall_width = 6)
  (h_num_bricks : num_bricks = 1600)
  (h_brick_length : brick_length = 1)
  (h_brick_width : brick_width = 0.1125)
  (h_brick_height : brick_height = 0.06) :
  ∃ (wall_height : Real),
    wall_height = 0.225 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by
  sorry


end NUMINAMATH_CALUDE_wall_height_proof_l283_28390


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l283_28309

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l283_28309


namespace NUMINAMATH_CALUDE_bobbys_shoe_cost_l283_28386

/-- The total cost for Bobby's handmade shoes -/
def total_cost (mold_cost hourly_rate hours_worked discount_percentage : ℚ) : ℚ :=
  mold_cost + (hourly_rate * hours_worked) * (1 - discount_percentage)

/-- Theorem stating that Bobby's total cost for handmade shoes is $730 -/
theorem bobbys_shoe_cost :
  total_cost 250 75 8 0.2 = 730 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_shoe_cost_l283_28386


namespace NUMINAMATH_CALUDE_evaluate_expression_l283_28332

/-- Given x, y, and z are variables, prove that (25x³y) · (4xy²z) · (1/(5xyz)²) = 4x²y/z -/
theorem evaluate_expression (x y z : ℝ) (h : z ≠ 0) :
  (25 * x^3 * y) * (4 * x * y^2 * z) * (1 / (5 * x * y * z)^2) = 4 * x^2 * y / z := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l283_28332


namespace NUMINAMATH_CALUDE_probability_is_31_over_473_l283_28311

/-- Represents a standard deck of cards --/
def StandardDeck : ℕ := 52

/-- Number of cards per rank in a standard deck --/
def CardsPerRank : ℕ := 4

/-- Number of pairs removed (two pairs of Aces and two pairs of Kings) --/
def PairsRemoved : ℕ := 2

/-- Number of ranks affected by pair removal --/
def RanksAffected : ℕ := 2

/-- Number of unaffected ranks (from Two to Queen) --/
def UnaffectedRanks : ℕ := 11

/-- Calculates the probability of selecting a pair from the modified deck --/
def probability_of_pair (deck : ℕ) (cards_per_rank : ℕ) (pairs_removed : ℕ) (ranks_affected : ℕ) (unaffected_ranks : ℕ) : ℚ :=
  let remaining_cards := deck - 2 * pairs_removed * cards_per_rank
  let total_combinations := remaining_cards.choose 2
  let affected_pairs := ranks_affected
  let unaffected_pairs := unaffected_ranks * (cards_per_rank.choose 2)
  let favorable_outcomes := affected_pairs + unaffected_pairs
  ↑favorable_outcomes / ↑total_combinations

theorem probability_is_31_over_473 :
  probability_of_pair StandardDeck CardsPerRank PairsRemoved RanksAffected UnaffectedRanks = 31 / 473 :=
sorry

end NUMINAMATH_CALUDE_probability_is_31_over_473_l283_28311


namespace NUMINAMATH_CALUDE_susan_age_in_five_years_l283_28387

/-- Represents the current year -/
def current_year : ℕ := 2023

/-- James' age in a given year -/
def james_age (year : ℕ) : ℕ := sorry

/-- Janet's age in a given year -/
def janet_age (year : ℕ) : ℕ := sorry

/-- Susan's age in a given year -/
def susan_age (year : ℕ) : ℕ := sorry

theorem susan_age_in_five_years :
  (∀ year : ℕ, james_age (year - 8) = 2 * janet_age (year - 8)) →
  james_age (current_year + 15) = 37 →
  (∀ year : ℕ, susan_age year = janet_age year - 3) →
  susan_age (current_year + 5) = 17 := by sorry

end NUMINAMATH_CALUDE_susan_age_in_five_years_l283_28387


namespace NUMINAMATH_CALUDE_shadow_boundary_is_constant_l283_28328

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The equation of the shadow boundary for a sphere -/
def shadowBoundary (s : Sphere) (lightSource : Point3D) : ℝ → ℝ := fun x => -2

/-- Theorem stating that the shadow boundary is y = -2 for the given sphere and light source -/
theorem shadow_boundary_is_constant (s : Sphere) (lightSource : Point3D) :
  s.center = Point3D.mk 0 0 2 →
  s.radius = 2 →
  lightSource = Point3D.mk 0 1 3 →
  ∀ x, shadowBoundary s lightSource x = -2 := by
  sorry

#check shadow_boundary_is_constant

end NUMINAMATH_CALUDE_shadow_boundary_is_constant_l283_28328


namespace NUMINAMATH_CALUDE_marbles_remainder_l283_28302

theorem marbles_remainder (r p g : ℕ) 
  (hr : r % 7 = 5) 
  (hp : p % 7 = 4) 
  (hg : g % 7 = 2) : 
  (r + p + g) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_marbles_remainder_l283_28302


namespace NUMINAMATH_CALUDE_perpendicular_lines_min_value_l283_28374

theorem perpendicular_lines_min_value (b : ℝ) (a : ℝ) (h1 : b > 1) :
  ((b^2 + 1) * (-1 / a) * (b - 1) = -1) →
  (∀ a' : ℝ, ((b^2 + 1) * (-1 / a') * (b - 1) = -1) → a ≤ a') →
  a = 2 * Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_min_value_l283_28374


namespace NUMINAMATH_CALUDE_cake_pieces_theorem_l283_28340

/-- The initial number of cake pieces -/
def initial_pieces : ℕ := 240

/-- The percentage of cake pieces eaten -/
def eaten_percentage : ℚ := 60 / 100

/-- The number of people who received the remaining pieces -/
def num_recipients : ℕ := 3

/-- The number of pieces each recipient received -/
def pieces_per_recipient : ℕ := 32

/-- Theorem stating that the initial number of cake pieces is correct -/
theorem cake_pieces_theorem :
  initial_pieces * (1 - eaten_percentage) = num_recipients * pieces_per_recipient := by
  sorry

end NUMINAMATH_CALUDE_cake_pieces_theorem_l283_28340


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l283_28383

theorem smallest_value_complex_sum (a b c d : ℤ) (ω : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) 
  (h_omega_power : ω^4 = 1) 
  (h_omega_neq_one : ω ≠ 1) :
  ∃ (z : ℂ), ∀ (x y u v : ℤ), x ≠ y ∧ x ≠ u ∧ x ≠ v ∧ y ≠ u ∧ y ≠ v ∧ u ≠ v →
    Complex.abs (x + y*ω + u*ω^2 + v*ω^3) ≥ Complex.abs z ∧
    Complex.abs z = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l283_28383
