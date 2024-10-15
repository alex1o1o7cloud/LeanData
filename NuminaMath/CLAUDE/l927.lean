import Mathlib

namespace NUMINAMATH_CALUDE_g_at_one_l927_92798

theorem g_at_one (a b c d : ℝ) (h₁ : 1 < a) (h₂ : a < b) (h₃ : b < c) (h₄ : c < d) :
  let f : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + d
  ∃ g : ℝ → ℝ,
    (∀ x, g x = 0 → ∃ y, f y = 0 ∧ x * y = 1) ∧
    (g 0 = 1) ∧
    (g 1 = (1 + a + b + c + d) / d) :=
by sorry

end NUMINAMATH_CALUDE_g_at_one_l927_92798


namespace NUMINAMATH_CALUDE_arccos_cos_eq_half_x_solution_l927_92715

theorem arccos_cos_eq_half_x_solution (x : Real) :
  -π/3 ≤ x → x ≤ π/3 → Real.arccos (Real.cos x) = x/2 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_half_x_solution_l927_92715


namespace NUMINAMATH_CALUDE_gcf_68_92_l927_92769

theorem gcf_68_92 : Nat.gcd 68 92 = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcf_68_92_l927_92769


namespace NUMINAMATH_CALUDE_first_level_teachers_selected_l927_92753

/-- Represents the number of teachers selected in a stratified sample -/
def stratified_sample (total : ℕ) (senior : ℕ) (first_level : ℕ) (second_level : ℕ) (sample_size : ℕ) : ℕ :=
  (first_level * sample_size) / (senior + first_level + second_level)

/-- Theorem stating that the number of first-level teachers selected in the given scenario is 12 -/
theorem first_level_teachers_selected :
  stratified_sample 380 90 120 170 38 = 12 := by
  sorry

end NUMINAMATH_CALUDE_first_level_teachers_selected_l927_92753


namespace NUMINAMATH_CALUDE_max_non_club_members_in_company_l927_92713

/-- The maximum number of people who did not join any club in a company with 5 clubs -/
def max_non_club_members (total_people : ℕ) (club_a : ℕ) (club_b : ℕ) (club_c : ℕ) (club_d : ℕ) (club_e : ℕ) (c_and_d_overlap : ℕ) (d_and_e_overlap : ℕ) : ℕ :=
  total_people - (club_a + club_b + club_c + (club_d - c_and_d_overlap) + (club_e - d_and_e_overlap))

/-- Theorem stating the maximum number of non-club members in the given scenario -/
theorem max_non_club_members_in_company :
  max_non_club_members 120 25 34 21 16 10 8 4 = 26 :=
by sorry

end NUMINAMATH_CALUDE_max_non_club_members_in_company_l927_92713


namespace NUMINAMATH_CALUDE_log_equality_l927_92794

theorem log_equality : Real.log 16 / Real.log 4096 = Real.log 4 / Real.log 64 := by
  sorry

#check log_equality

end NUMINAMATH_CALUDE_log_equality_l927_92794


namespace NUMINAMATH_CALUDE_billys_age_l927_92728

theorem billys_age (B J S : ℕ) 
  (h1 : B = 2 * J) 
  (h2 : B + J = 3 * S) 
  (h3 : S = 27) : 
  B = 54 := by
  sorry

end NUMINAMATH_CALUDE_billys_age_l927_92728


namespace NUMINAMATH_CALUDE_moon_distance_scientific_notation_l927_92732

theorem moon_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 384000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.84 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_moon_distance_scientific_notation_l927_92732


namespace NUMINAMATH_CALUDE_factorization_x3y_minus_xy_l927_92721

theorem factorization_x3y_minus_xy (x y : ℝ) : x^3*y - x*y = x*y*(x - 1)*(x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x3y_minus_xy_l927_92721


namespace NUMINAMATH_CALUDE_sum_of_symmetric_points_zero_l927_92787

-- Define a function v that is symmetric under 180° rotation around the origin
def v_symmetric (v : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, v (-x) = -v x

-- Theorem statement
theorem sum_of_symmetric_points_zero (v : ℝ → ℝ) (h : v_symmetric v) :
  v (-2) + v (-1) + v 1 + v 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_symmetric_points_zero_l927_92787


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l927_92744

theorem polynomial_value_theorem (P : ℤ → ℤ) : 
  (∃ a b c d e : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                     c ≠ d ∧ c ≠ e ∧ 
                     d ≠ e ∧
                     P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5 ∧ P e = 5) →
  (∀ x : ℤ, P x ≠ 9) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l927_92744


namespace NUMINAMATH_CALUDE_car_race_distance_l927_92778

theorem car_race_distance (v_A v_B : ℝ) (d : ℝ) :
  v_A > 0 ∧ v_B > 0 ∧ d > 0 →
  (v_A / v_B = (2 * v_A) / (2 * v_B)) →
  (d / v_A = (d/2) / (2 * v_A)) →
  15 = 15 := by sorry

end NUMINAMATH_CALUDE_car_race_distance_l927_92778


namespace NUMINAMATH_CALUDE_quadratic_form_value_l927_92724

/-- Given a quadratic function f(x) = 4x^2 - 40x + 100, 
    prove that when written in the form (ax + b)^2 + c,
    the value of 2b - 3c is -20 -/
theorem quadratic_form_value (a b c : ℝ) : 
  (∀ x, 4 * x^2 - 40 * x + 100 = (a * x + b)^2 + c) →
  2 * b - 3 * c = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_value_l927_92724


namespace NUMINAMATH_CALUDE_chinese_count_l927_92751

theorem chinese_count (total : ℕ) (americans : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : americans = 16)
  (h3 : australians = 11) :
  total - (americans + australians) = 22 := by
sorry

end NUMINAMATH_CALUDE_chinese_count_l927_92751


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_l927_92707

/-- A parabola defined by x = -3y^2 + 2y + 2 has exactly one x-intercept. -/
theorem parabola_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_l927_92707


namespace NUMINAMATH_CALUDE_fraction_addition_l927_92786

theorem fraction_addition : (3 / 4) / (5 / 8) + 1 / 8 = 53 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l927_92786


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l927_92719

theorem inequality_proof (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 :=
sorry

theorem equality_condition (x y z : ℝ) 
  (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 3) :
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l927_92719


namespace NUMINAMATH_CALUDE_percentage_problem_l927_92718

theorem percentage_problem (x : ℝ) (P : ℝ) 
  (h1 : x = 180)
  (h2 : P * x = 0.10 * 500 - 5) :
  P = 0.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l927_92718


namespace NUMINAMATH_CALUDE_white_balls_count_l927_92716

theorem white_balls_count (red_balls : ℕ) (ratio_red : ℕ) (ratio_white : ℕ) : 
  red_balls = 16 → ratio_red = 4 → ratio_white = 5 → 
  (red_balls * ratio_white) / ratio_red = 20 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l927_92716


namespace NUMINAMATH_CALUDE_joes_honey_purchase_l927_92700

theorem joes_honey_purchase (orange_price : ℚ) (juice_price : ℚ) (honey_price : ℚ) 
  (plant_price : ℚ) (total_spent : ℚ) (orange_count : ℕ) (juice_count : ℕ) 
  (plant_count : ℕ) :
  orange_price = 9/2 →
  juice_price = 1/2 →
  honey_price = 5 →
  plant_price = 9 →
  total_spent = 68 →
  orange_count = 3 →
  juice_count = 7 →
  plant_count = 4 →
  (total_spent - (orange_price * orange_count + juice_price * juice_count + 
    plant_price * plant_count)) / honey_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_joes_honey_purchase_l927_92700


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l927_92734

/-- A rectangular prism is a three-dimensional geometric shape. -/
structure RectangularPrism where
  edges : ℕ
  corners : ℕ
  faces : ℕ

/-- The properties of a rectangular prism -/
def is_valid_rectangular_prism (rp : RectangularPrism) : Prop :=
  rp.edges = 12 ∧ rp.corners = 8 ∧ rp.faces = 6

/-- The theorem stating that the sum of edges, corners, and faces of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) 
  (h : is_valid_rectangular_prism rp) : 
  rp.edges + rp.corners + rp.faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l927_92734


namespace NUMINAMATH_CALUDE_quadratic_square_completion_l927_92793

theorem quadratic_square_completion (x : ℝ) : 
  (x^2 + 10*x + 9 = 0) → (∃ c d : ℝ, (x + c)^2 = d ∧ d = 16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_completion_l927_92793


namespace NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l927_92742

theorem smallest_n_for_floor_equation : ∃ (x : ℤ), ⌊(10 : ℝ)^7 / x⌋ = 1989 ∧ ∀ (n : ℕ), n < 7 → ¬∃ (x : ℤ), ⌊(10 : ℝ)^n / x⌋ = 1989 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l927_92742


namespace NUMINAMATH_CALUDE_sandwich_composition_ham_cost_is_correct_l927_92764

/-- The cost of a slice of ham in a sandwich -/
def ham_cost : ℚ := 25 / 100

/-- The selling price of a sandwich -/
def sandwich_price : ℚ := 150 / 100

/-- The cost of a slice of bread -/
def bread_cost : ℚ := 15 / 100

/-- The cost of a slice of cheese -/
def cheese_cost : ℚ := 35 / 100

/-- The total cost to make a sandwich -/
def sandwich_cost : ℚ := 90 / 100

/-- A sandwich contains 2 slices of bread, 1 slice of ham, and 1 slice of cheese -/
theorem sandwich_composition (h : ℚ) :
  sandwich_cost = 2 * bread_cost + h + cheese_cost :=
sorry

/-- The cost of a slice of ham is $0.25 -/
theorem ham_cost_is_correct :
  ham_cost = sandwich_cost - 2 * bread_cost - cheese_cost :=
sorry

end NUMINAMATH_CALUDE_sandwich_composition_ham_cost_is_correct_l927_92764


namespace NUMINAMATH_CALUDE_cardinality_of_P_l927_92706

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}
def P : Finset ℕ := M ∩ N

theorem cardinality_of_P : Finset.card P = 2 := by sorry

end NUMINAMATH_CALUDE_cardinality_of_P_l927_92706


namespace NUMINAMATH_CALUDE_butterfly_collection_l927_92785

theorem butterfly_collection (total : ℕ) (black : ℕ) : 
  total = 19 → 
  black = 10 → 
  ∃ (blue yellow : ℕ), 
    blue = 2 * yellow ∧ 
    blue + yellow + black = total ∧ 
    blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterfly_collection_l927_92785


namespace NUMINAMATH_CALUDE_pat_initial_stickers_l927_92709

/-- The number of stickers Pat had at the end of the week -/
def end_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := 22

/-- The number of stickers Pat had on the first day of the week -/
def initial_stickers : ℕ := end_stickers - earned_stickers

theorem pat_initial_stickers : initial_stickers = 39 := by
  sorry

end NUMINAMATH_CALUDE_pat_initial_stickers_l927_92709


namespace NUMINAMATH_CALUDE_jefferson_carriage_cost_l927_92712

/-- Represents the carriage rental cost calculation --/
def carriageRentalCost (
  totalDistance : ℝ)
  (stopDistances : List ℝ)
  (speeds : List ℝ)
  (baseRate : ℝ)
  (flatFee : ℝ)
  (additionalChargeThreshold : ℝ)
  (additionalChargeRate : ℝ)
  (discountRate : ℝ) : ℝ :=
  sorry

/-- Theorem stating the correct total cost for Jefferson's carriage rental --/
theorem jefferson_carriage_cost :
  carriageRentalCost
    20                     -- total distance to church
    [4, 6, 3]              -- distances to each stop
    [8, 12, 10, 15]        -- speeds for each leg
    35                     -- base rate per hour
    20                     -- flat fee
    10                     -- additional charge speed threshold
    5                      -- additional charge rate per mile
    0.1                    -- discount rate
  = 132.15 := by sorry

end NUMINAMATH_CALUDE_jefferson_carriage_cost_l927_92712


namespace NUMINAMATH_CALUDE_sandwich_cookie_cost_l927_92755

theorem sandwich_cookie_cost (s c : ℝ) 
  (eq1 : 3 * s + 4 * c = 4.20)
  (eq2 : 4 * s + 3 * c = 4.50) : 
  4 * s + 5 * c = 5.44 := by
sorry

end NUMINAMATH_CALUDE_sandwich_cookie_cost_l927_92755


namespace NUMINAMATH_CALUDE_zoo_count_difference_l927_92714

theorem zoo_count_difference (zebras camels monkeys giraffes : ℕ) : 
  zebras = 12 →
  camels = zebras / 2 →
  monkeys = 4 * camels →
  giraffes = 2 →
  monkeys - giraffes = 22 := by
sorry

end NUMINAMATH_CALUDE_zoo_count_difference_l927_92714


namespace NUMINAMATH_CALUDE_ellipse_condition_l927_92795

/-- Represents an ellipse with equation ax^2 + by^2 = 1 -/
structure Ellipse (a b : ℝ) where
  equation : ∀ x y : ℝ, a * x^2 + b * y^2 = 1
  is_ellipse : True  -- We assume it's an ellipse
  foci_on_x_axis : True  -- We assume foci are on x-axis

/-- 
If ax^2 + by^2 = 1 represents an ellipse with foci on the x-axis,
where a and b are real numbers, then b > a > 0.
-/
theorem ellipse_condition (a b : ℝ) (e : Ellipse a b) : b > a ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_l927_92795


namespace NUMINAMATH_CALUDE_system_of_equations_solutions_l927_92788

theorem system_of_equations_solutions :
  (∃ x y : ℚ, x + y = 3 ∧ x - y = 1 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℚ, 2*x + y = 3 ∧ x - 2*y = 1 ∧ x = 7/5 ∧ y = 1/5) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solutions_l927_92788


namespace NUMINAMATH_CALUDE_coastal_analysis_uses_gis_l927_92789

-- Define the available technologies
inductive CoastalAnalysisTechnology
  | GPS
  | GIS
  | RemoteSensing
  | GeographicInformationTechnology

-- Define the properties of the analysis
structure CoastalAnalysis where
  involves_sea_level_changes : Bool
  available_technologies : List CoastalAnalysisTechnology

-- Define the main technology used for the analysis
def main_technology_for_coastal_analysis (analysis : CoastalAnalysis) : CoastalAnalysisTechnology :=
  CoastalAnalysisTechnology.GIS

-- Theorem statement
theorem coastal_analysis_uses_gis (analysis : CoastalAnalysis) 
  (h1 : analysis.involves_sea_level_changes = true)
  (h2 : analysis.available_technologies.length ≥ 2) :
  main_technology_for_coastal_analysis analysis = CoastalAnalysisTechnology.GIS := by
  sorry

end NUMINAMATH_CALUDE_coastal_analysis_uses_gis_l927_92789


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l927_92762

theorem fraction_zero_implies_x_negative_two (x : ℝ) : 
  (x^2 - 4) / (x^2 - 4*x + 4) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l927_92762


namespace NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l927_92790

theorem abs_minus_self_nonnegative (a : ℚ) : |a| - a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_minus_self_nonnegative_l927_92790


namespace NUMINAMATH_CALUDE_find_k_value_l927_92796

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k) * (x + 3*k) = x^3 + k*(x^2 - 2*x - 8)) → k = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l927_92796


namespace NUMINAMATH_CALUDE_smallest_c_value_l927_92704

theorem smallest_c_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : ∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin (b * (-π/4) + c))
  (h5 : a = 3) :
  c ≥ 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l927_92704


namespace NUMINAMATH_CALUDE_complex_power_215_36_l927_92702

theorem complex_power_215_36 : (Complex.exp (215 * π / 180 * Complex.I))^36 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_215_36_l927_92702


namespace NUMINAMATH_CALUDE_halloween_candy_weight_l927_92765

/-- Represents the weight of different types of candy in pounds -/
structure CandyWeights where
  chocolate : ℝ
  gummyBears : ℝ
  caramels : ℝ
  hardCandy : ℝ

/-- Calculates the total weight of candy -/
def totalWeight (cw : CandyWeights) : ℝ :=
  cw.chocolate + cw.gummyBears + cw.caramels + cw.hardCandy

/-- Frank's candy weights -/
def frankCandy : CandyWeights := {
  chocolate := 3,
  gummyBears := 2,
  caramels := 1,
  hardCandy := 4
}

/-- Gwen's candy weights -/
def gwenCandy : CandyWeights := {
  chocolate := 2,
  gummyBears := 2.5,
  caramels := 1,
  hardCandy := 1.5
}

/-- Theorem stating that the total combined weight of Frank and Gwen's Halloween candy is 17 pounds -/
theorem halloween_candy_weight :
  totalWeight frankCandy + totalWeight gwenCandy = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_weight_l927_92765


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l927_92780

theorem absolute_value_inequality (x : ℝ) : |x| ≠ 3 → x ≠ 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l927_92780


namespace NUMINAMATH_CALUDE_range_of_a_for_monotone_decreasing_f_l927_92792

/-- A piecewise function f(x) defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x^2 - 6*x + a^2 + 1
  else x^(5 - 2*a)

/-- The theorem stating the range of a for which f is monotonically decreasing -/
theorem range_of_a_for_monotone_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Ioo (5/2) 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotone_decreasing_f_l927_92792


namespace NUMINAMATH_CALUDE_polynomial_monomial_degree_l927_92752

/-- Given a sixth-degree polynomial and a monomial, prove the values of m and n -/
theorem polynomial_monomial_degree (m n : ℕ) : 
  (2 + (m + 1) = 6) ∧ (2*n + (5 - m) = 6) → m = 3 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_monomial_degree_l927_92752


namespace NUMINAMATH_CALUDE_birds_in_tree_l927_92737

theorem birds_in_tree (initial_birds final_birds : ℕ) : 
  initial_birds = 14 → final_birds = 35 → final_birds - initial_birds = 21 := by
  sorry

end NUMINAMATH_CALUDE_birds_in_tree_l927_92737


namespace NUMINAMATH_CALUDE_math_books_count_l927_92733

theorem math_books_count (total_books : ℕ) (math_cost history_cost total_price : ℕ) 
  (h1 : total_books = 90)
  (h2 : math_cost = 4)
  (h3 : history_cost = 5)
  (h4 : total_price = 397) :
  ∃ (math_books : ℕ), 
    math_books * math_cost + (total_books - math_books) * history_cost = total_price ∧ 
    math_books = 53 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l927_92733


namespace NUMINAMATH_CALUDE_constant_term_expansion_l927_92701

/-- The constant term in the expansion of (x^2 + a/sqrt(x))^5 -/
def constantTerm (a : ℝ) : ℝ := 5 * a^4

theorem constant_term_expansion (a : ℝ) (h1 : a > 0) (h2 : constantTerm a = 80) : a = 2 := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_constant_term_expansion_l927_92701


namespace NUMINAMATH_CALUDE_rectangle_area_l927_92750

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l927_92750


namespace NUMINAMATH_CALUDE_sqrt_nine_equals_three_l927_92722

theorem sqrt_nine_equals_three : Real.sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_equals_three_l927_92722


namespace NUMINAMATH_CALUDE_diamond_two_seven_l927_92708

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a^2 * b - a * b^2

-- Theorem statement
theorem diamond_two_seven : diamond 2 7 = -70 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_seven_l927_92708


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l927_92756

theorem min_throws_for_repeated_sum (n : ℕ) (d : ℕ) (s : ℕ) : 
  n = 4 →  -- number of dice
  d = 6 →  -- number of sides on each die
  s = (n * d - n * 1 + 1) →  -- number of possible sums
  s + 1 = 22 →  -- minimum number of throws
  ∀ (throws : ℕ), throws ≥ s + 1 → 
    ∃ (sum1 sum2 : ℕ) (i j : ℕ), 
      i ≠ j ∧ i < throws ∧ j < throws ∧ sum1 = sum2 :=
by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l927_92756


namespace NUMINAMATH_CALUDE_no_solutions_fibonacci_equation_l927_92703

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n+2) => fib (n+1) + fib n

theorem no_solutions_fibonacci_equation :
  ∀ n : ℕ, n * (fib n) * (fib (n+1)) ≠ (fib (n+2) - 1)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_no_solutions_fibonacci_equation_l927_92703


namespace NUMINAMATH_CALUDE_female_students_count_l927_92763

/-- Given a class with n total students and m male students, 
    prove that the number of female students is n - m. -/
theorem female_students_count (n m : ℕ) : ℕ :=
  n - m

#check female_students_count

end NUMINAMATH_CALUDE_female_students_count_l927_92763


namespace NUMINAMATH_CALUDE_product_remainder_is_one_l927_92770

def sequence1 : List Nat := List.range 10 |>.map (fun n => 3 + 10 * n)
def sequence2 : List Nat := List.range 10 |>.map (fun n => 7 + 10 * n)

theorem product_remainder_is_one :
  (sequence1.prod * sequence2.prod) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_is_one_l927_92770


namespace NUMINAMATH_CALUDE_regular_polygon_radius_inequality_l927_92741

/-- For a regular polygon with n sides, n ≥ 3, the circumradius R is at most twice the inradius r. -/
theorem regular_polygon_radius_inequality (n : ℕ) (r R : ℝ) 
  (h_n : n ≥ 3) 
  (h_r : r > 0) 
  (h_R : R > 0) 
  (h_relation : r / R = Real.cos (π / n)) : 
  R ≤ 2 * r := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_radius_inequality_l927_92741


namespace NUMINAMATH_CALUDE_pigeonhole_on_floor_division_l927_92739

theorem pigeonhole_on_floor_division (n : ℕ) (h_n : n > 3) 
  (nums : Finset ℕ) (h_nums_card : nums.card = n) 
  (h_nums_distinct : nums.card = Finset.card (Finset.image id nums))
  (h_nums_bound : ∀ x ∈ nums, x < Nat.factorial (n - 1)) :
  ∃ (a b c d : ℕ), a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧ 
    a > b ∧ c > d ∧ (a ≠ c ∨ b ≠ d) ∧ 
    (a / b : ℕ) = (c / d : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_on_floor_division_l927_92739


namespace NUMINAMATH_CALUDE_price_change_effect_l927_92748

theorem price_change_effect (a : ℝ) (h : a > 0) : a * 1.02 * 0.98 < a := by
  sorry

end NUMINAMATH_CALUDE_price_change_effect_l927_92748


namespace NUMINAMATH_CALUDE_inequality_proof_l927_92773

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l927_92773


namespace NUMINAMATH_CALUDE_multiplicative_inverse_208_mod_307_l927_92740

theorem multiplicative_inverse_208_mod_307 : ∃ x : ℕ, x < 307 ∧ (208 * x) % 307 = 1 :=
by
  use 240
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_208_mod_307_l927_92740


namespace NUMINAMATH_CALUDE_fourth_test_score_l927_92799

theorem fourth_test_score (first_three_average : ℝ) (desired_increase : ℝ) : 
  first_three_average = 85 → 
  desired_increase = 2 → 
  (3 * first_three_average + 93) / 4 = first_three_average + desired_increase := by
sorry

end NUMINAMATH_CALUDE_fourth_test_score_l927_92799


namespace NUMINAMATH_CALUDE_square_99_is_white_l927_92711

def Grid := Fin 9 → Fin 9 → Bool

def is_adjacent (x1 y1 x2 y2 : Fin 9) : Prop :=
  (x1 = x2 ∧ y1.val + 1 = y2.val) ∨
  (x1 = x2 ∧ y2.val + 1 = y1.val) ∨
  (y1 = y2 ∧ x1.val + 1 = x2.val) ∨
  (y1 = y2 ∧ x2.val + 1 = x1.val)

def valid_grid (g : Grid) : Prop :=
  (g 4 4 = true) ∧
  (g 4 9 = true) ∧
  (∀ x y, g x y → (∃! x' y', is_adjacent x y x' y' ∧ g x' y')) ∧
  (∀ x y, ¬g x y → (∃! x' y', is_adjacent x y x' y' ∧ ¬g x' y'))

theorem square_99_is_white (g : Grid) (h : valid_grid g) : g 9 9 = false := by
  sorry

#check square_99_is_white

end NUMINAMATH_CALUDE_square_99_is_white_l927_92711


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l927_92766

/-- Given a cylinder with volume 54π cm³ and height three times its radius,
    prove that the total volume of a cone and a sphere both having the same radius
    as the cylinder is 42π cm³ -/
theorem cylinder_cone_sphere_volume (r : ℝ) (h : ℝ) : 
  (π * r^2 * h = 54 * π) →
  (h = 3 * r) →
  (π * r^2 * r / 3 + 4 * π * r^3 / 3 = 42 * π) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l927_92766


namespace NUMINAMATH_CALUDE_friday_attendance_l927_92774

/-- Calculates the percentage of students present on a given day -/
def students_present (initial_absenteeism : ℝ) (daily_increase : ℝ) (day : ℕ) : ℝ :=
  100 - (initial_absenteeism + daily_increase * day)

/-- Proves that the percentage of students present on Friday is 78% -/
theorem friday_attendance 
  (initial_absenteeism : ℝ) 
  (daily_increase : ℝ) 
  (h1 : initial_absenteeism = 14) 
  (h2 : daily_increase = 2) : 
  students_present initial_absenteeism daily_increase 4 = 78 := by
  sorry

#eval students_present 14 2 4

end NUMINAMATH_CALUDE_friday_attendance_l927_92774


namespace NUMINAMATH_CALUDE_child_cost_age_18_l927_92791

/-- Represents the cost of raising a child --/
structure ChildCost where
  initialYearlyCost : ℕ
  initialYears : ℕ
  laterYearlyCost : ℕ
  tuitionCost : ℕ
  totalCost : ℕ

/-- Calculates the age at which the child stops incurring yearly cost --/
def ageStopCost (c : ChildCost) : ℕ :=
  let initialCost := c.initialYears * c.initialYearlyCost
  let laterYears := (c.totalCost - initialCost - c.tuitionCost) / c.laterYearlyCost
  c.initialYears + laterYears

/-- Theorem stating that given the specific costs, the child stops incurring yearly cost at age 18 --/
theorem child_cost_age_18 :
  let c := ChildCost.mk 5000 8 10000 125000 265000
  ageStopCost c = 18 := by
  sorry

#eval ageStopCost (ChildCost.mk 5000 8 10000 125000 265000)

end NUMINAMATH_CALUDE_child_cost_age_18_l927_92791


namespace NUMINAMATH_CALUDE_assignment_schemes_eq_240_l927_92745

/-- The number of ways to assign 4 out of 6 students to tasks A, B, C, and D,
    given that two specific students can perform task A. -/
def assignment_schemes : ℕ :=
  Nat.descFactorial 6 4 - 2 * Nat.descFactorial 5 3

/-- Theorem stating that the number of assignment schemes is 240. -/
theorem assignment_schemes_eq_240 : assignment_schemes = 240 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_eq_240_l927_92745


namespace NUMINAMATH_CALUDE_bill_difference_is_zero_l927_92705

/-- Given Linda's and Mark's tips and tip percentages, prove that the difference between their bills is 0. -/
theorem bill_difference_is_zero (linda_tip mark_tip : ℝ) (linda_percent mark_percent : ℝ) 
  (h1 : linda_tip = 5)
  (h2 : linda_percent = 0.25)
  (h3 : mark_tip = 3)
  (h4 : mark_percent = 0.15)
  (h5 : linda_tip = linda_percent * linda_bill)
  (h6 : mark_tip = mark_percent * mark_bill)
  : linda_bill - mark_bill = 0 := by
  sorry

#check bill_difference_is_zero

end NUMINAMATH_CALUDE_bill_difference_is_zero_l927_92705


namespace NUMINAMATH_CALUDE_orange_pill_cost_l927_92783

/-- Represents the cost of pills for Alice's treatment --/
structure PillCost where
  orange : ℝ
  blue : ℝ
  duration : ℕ
  daily_intake : ℕ
  total_cost : ℝ

/-- The cost of pills satisfies the given conditions --/
def is_valid_cost (cost : PillCost) : Prop :=
  cost.orange = cost.blue + 2 ∧
  cost.duration = 21 ∧
  cost.daily_intake = 1 ∧
  cost.total_cost = 735 ∧
  cost.duration * cost.daily_intake * (cost.orange + cost.blue) = cost.total_cost

/-- The theorem stating that the cost of one orange pill is $18.5 --/
theorem orange_pill_cost (cost : PillCost) (h : is_valid_cost cost) : cost.orange = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_pill_cost_l927_92783


namespace NUMINAMATH_CALUDE_monday_is_42_l927_92710

/-- Represents the temperature on each day of the week --/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The average temperature for Monday to Thursday is 48 degrees --/
def avg_mon_to_thu (w : WeekTemperatures) : Prop :=
  (w.monday + w.tuesday + w.wednesday + w.thursday) / 4 = 48

/-- The average temperature for Tuesday to Friday is 46 degrees --/
def avg_tue_to_fri (w : WeekTemperatures) : Prop :=
  (w.tuesday + w.wednesday + w.thursday + w.friday) / 4 = 46

/-- The temperature on Friday is 34 degrees --/
def friday_temp (w : WeekTemperatures) : Prop :=
  w.friday = 34

/-- Some day has a temperature of 42 degrees --/
def some_day_42 (w : WeekTemperatures) : Prop :=
  w.monday = 42 ∨ w.tuesday = 42 ∨ w.wednesday = 42 ∨ w.thursday = 42 ∨ w.friday = 42

theorem monday_is_42 (w : WeekTemperatures) 
  (h1 : avg_mon_to_thu w) 
  (h2 : avg_tue_to_fri w) 
  (h3 : friday_temp w) 
  (h4 : some_day_42 w) : 
  w.monday = 42 := by
  sorry

end NUMINAMATH_CALUDE_monday_is_42_l927_92710


namespace NUMINAMATH_CALUDE_fourth_root_of_cubic_l927_92731

theorem fourth_root_of_cubic (c d : ℚ) :
  (∀ x : ℚ, c * x^3 + (c + 3*d) * x^2 + (d - 4*c) * x + (10 - c) = 0 ↔ x = -1 ∨ x = 4 ∨ x = 2 ∨ x = -9/2) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_of_cubic_l927_92731


namespace NUMINAMATH_CALUDE_fox_bridge_crossing_fox_initial_money_unique_l927_92749

/-- The function that doubles the money and subtracts the toll -/
def f (x : ℝ) : ℝ := 2 * x - 40

/-- Theorem stating that applying f three times to 35 results in 0 -/
theorem fox_bridge_crossing :
  f (f (f 35)) = 0 := by sorry

/-- Theorem proving that 35 is the only initial value that results in 0 after three crossings -/
theorem fox_initial_money_unique (x : ℝ) :
  f (f (f x)) = 0 → x = 35 := by sorry

end NUMINAMATH_CALUDE_fox_bridge_crossing_fox_initial_money_unique_l927_92749


namespace NUMINAMATH_CALUDE_f_monotone_range_l927_92777

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem f_monotone_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_range_l927_92777


namespace NUMINAMATH_CALUDE_solution_set_for_even_monotonic_function_l927_92784

-- Define the properties of the function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- Define the set of solutions
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (x + 1) = f (2 * x)}

-- Theorem statement
theorem solution_set_for_even_monotonic_function
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_monotonic : is_monotonic_on_positive f) :
  solution_set f = {1, -1/3} := by
sorry

end NUMINAMATH_CALUDE_solution_set_for_even_monotonic_function_l927_92784


namespace NUMINAMATH_CALUDE_tomato_soup_cans_l927_92746

/-- Proves the number of cans of tomato soup sold for every 4 cans of chili beans -/
theorem tomato_soup_cans (total_cans : ℕ) (chili_beans_cans : ℕ) 
  (h1 : total_cans = 12)
  (h2 : chili_beans_cans = 8)
  (h3 : ∃ (n : ℕ), n * 4 = total_cans - chili_beans_cans) :
  4 = total_cans - chili_beans_cans :=
by sorry

end NUMINAMATH_CALUDE_tomato_soup_cans_l927_92746


namespace NUMINAMATH_CALUDE_polynomial_factorization_l927_92771

/-- Proves that x³ - 2x²y + xy² = x(x-y)² for all real numbers x and y -/
theorem polynomial_factorization (x y : ℝ) : 
  x^3 - 2*x^2*y + x*y^2 = x*(x-y)^2 := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l927_92771


namespace NUMINAMATH_CALUDE_fishing_trip_total_l927_92717

def total_fish (pikes sturgeons herrings : ℕ) : ℕ :=
  pikes + sturgeons + herrings

theorem fishing_trip_total : total_fish 30 40 75 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fishing_trip_total_l927_92717


namespace NUMINAMATH_CALUDE_prob_less_than_8_prob_at_least_7_l927_92729

-- Define the probabilities
def p_9_or_above : ℝ := 0.56
def p_8 : ℝ := 0.22
def p_7 : ℝ := 0.12

-- Theorem for the first question
theorem prob_less_than_8 : 1 - p_9_or_above - p_8 = 0.22 := by sorry

-- Theorem for the second question
theorem prob_at_least_7 : p_9_or_above + p_8 + p_7 = 0.9 := by sorry

end NUMINAMATH_CALUDE_prob_less_than_8_prob_at_least_7_l927_92729


namespace NUMINAMATH_CALUDE_zero_has_square_root_l927_92760

theorem zero_has_square_root : ∃ x : ℝ, x^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_has_square_root_l927_92760


namespace NUMINAMATH_CALUDE_bricks_per_row_l927_92726

theorem bricks_per_row (total_bricks : ℕ) (rows_per_wall : ℕ) (num_walls : ℕ) 
  (h1 : total_bricks = 3000)
  (h2 : rows_per_wall = 50)
  (h3 : num_walls = 2) :
  total_bricks / (rows_per_wall * num_walls) = 30 := by
sorry

end NUMINAMATH_CALUDE_bricks_per_row_l927_92726


namespace NUMINAMATH_CALUDE_inequality_proof_l927_92720

theorem inequality_proof (x a b c : ℝ) 
  (h1 : x ≠ a) (h2 : x ≠ b) (h3 : x ≠ c) 
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) : 
  1 / (x - a) + 1 / (x - b) > 1 / (x - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l927_92720


namespace NUMINAMATH_CALUDE_min_value_expression_l927_92727

theorem min_value_expression (a b c k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (a^2 / (k*b)) + (b^2 / (k*c)) + (c^2 / (k*a)) ≥ 3/k ∧
  ((a^2 / (k*b)) + (b^2 / (k*c)) + (c^2 / (k*a)) = 3/k ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l927_92727


namespace NUMINAMATH_CALUDE_pet_store_kittens_l927_92759

theorem pet_store_kittens (initial : ℕ) : initial + 3 = 9 → initial = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_kittens_l927_92759


namespace NUMINAMATH_CALUDE_lindas_substitution_l927_92730

theorem lindas_substitution (a b c d : ℕ) (e : ℝ) : 
  a = 120 → b = 5 → c = 4 → d = 10 →
  (a / b * c + d - e : ℝ) = (a / (b * (c + (d - e)))) →
  e = 16 := by
sorry

end NUMINAMATH_CALUDE_lindas_substitution_l927_92730


namespace NUMINAMATH_CALUDE_pencil_count_l927_92723

/-- Proves that given the specified costs and quantities, the number of pencils needed is 24 --/
theorem pencil_count (pencil_cost folder_cost total_cost : ℚ) (folder_count : ℕ) : 
  pencil_cost = 1/2 →
  folder_cost = 9/10 →
  folder_count = 20 →
  total_cost = 30 →
  (total_cost - folder_cost * folder_count) / pencil_cost = 24 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l927_92723


namespace NUMINAMATH_CALUDE_square_perimeter_from_rectangle_l927_92797

theorem square_perimeter_from_rectangle (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 32)
  (h2 : rectangle_width = 10)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  ∃ (square_side : ℝ), 
    square_side^2 = 5 * (rectangle_length * rectangle_width) ∧ 
    4 * square_side = 160 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_from_rectangle_l927_92797


namespace NUMINAMATH_CALUDE_derivative_exp_sin_l927_92772

theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_derivative_exp_sin_l927_92772


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l927_92761

theorem mean_of_added_numbers (original_count : ℕ) (original_mean : ℚ) 
  (new_count : ℕ) (new_mean : ℚ) (x y z : ℚ) : 
  original_count = 7 →
  original_mean = 40 →
  new_count = original_count + 3 →
  new_mean = 50 →
  (original_count * original_mean + x + y + z) / new_count = new_mean →
  (x + y + z) / 3 = 220 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l927_92761


namespace NUMINAMATH_CALUDE_translation_result_l927_92782

/-- A line in the xy-plane is represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically by a given amount. -/
def translateLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- The original line y = 2x - 1 -/
def originalLine : Line :=
  { slope := 2, intercept := -1 }

/-- The amount of upward translation -/
def translationAmount : ℝ := 2

/-- The resulting line after translation -/
def resultingLine : Line := translateLine originalLine translationAmount

theorem translation_result :
  resultingLine.slope = 2 ∧ resultingLine.intercept = 1 := by
  sorry

end NUMINAMATH_CALUDE_translation_result_l927_92782


namespace NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l927_92758

theorem smaller_rectangle_dimensions 
  (square_side : ℝ) 
  (h_square_side : square_side = 10) 
  (small_length small_width : ℝ) 
  (h_rectangles : small_length + 2 * small_length = square_side) 
  (h_square : small_width = small_length) : 
  small_length = 10 / 3 ∧ small_width = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l927_92758


namespace NUMINAMATH_CALUDE_arrangement_exists_l927_92735

theorem arrangement_exists (n : ℕ) : 
  ∃ (p : Fin n → ℕ), Function.Injective p ∧ Set.range p = Finset.range n ∧
    ∀ (i j k : Fin n), i < j → j < k → 
      p j ≠ (p i + p k) / 2 := by sorry

end NUMINAMATH_CALUDE_arrangement_exists_l927_92735


namespace NUMINAMATH_CALUDE_soccer_ball_selling_price_l927_92757

theorem soccer_ball_selling_price 
  (num_balls : ℕ) 
  (cost_per_ball : ℚ) 
  (total_profit : ℚ) 
  (h1 : num_balls = 50)
  (h2 : cost_per_ball = 60)
  (h3 : total_profit = 1950) :
  (total_profit / num_balls + cost_per_ball : ℚ) = 99 := by
sorry

end NUMINAMATH_CALUDE_soccer_ball_selling_price_l927_92757


namespace NUMINAMATH_CALUDE_common_solution_z_values_l927_92775

theorem common_solution_z_values : 
  ∃ (z₁ z₂ : ℝ), 
    (∀ x : ℝ, x^2 + z₁^2 - 9 = 0 ∧ x^2 - 4*z₁ + 5 = 0) ∧
    (∀ x : ℝ, x^2 + z₂^2 - 9 = 0 ∧ x^2 - 4*z₂ + 5 = 0) ∧
    z₁ = -2 + 3 * Real.sqrt 2 ∧
    z₂ = -2 - 3 * Real.sqrt 2 ∧
    (∀ z : ℝ, (∃ x : ℝ, x^2 + z^2 - 9 = 0 ∧ x^2 - 4*z + 5 = 0) → (z = z₁ ∨ z = z₂)) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_z_values_l927_92775


namespace NUMINAMATH_CALUDE_triangle_value_l927_92747

theorem triangle_value (q : ℤ) (h1 : ∃ triangle : ℤ, triangle + q = 59) 
  (h2 : ∃ triangle : ℤ, (triangle + q) + q = 106) : 
  ∃ triangle : ℤ, triangle = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l927_92747


namespace NUMINAMATH_CALUDE_range_of_a_l927_92779

/-- Given sets A and B, prove that if A ∪ B = A, then 0 < a ≤ 9/5 -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := { x | 0 < x ∧ x ≤ 3 }
  let B : Set ℝ := { x | x^2 - 2*a*x + a ≤ 0 }
  (A ∪ B = A) → (0 < a ∧ a ≤ 9/5) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l927_92779


namespace NUMINAMATH_CALUDE_circle_area_ratio_l927_92736

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 : ℝ) * (2 * Real.pi * r₁) = (30 / 360 : ℝ) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l927_92736


namespace NUMINAMATH_CALUDE_cost_of_cherries_l927_92776

/-- Given Sally's purchase of peaches and cherries, prove the cost of cherries. -/
theorem cost_of_cherries
  (peaches_after_coupon : ℝ)
  (coupon_value : ℝ)
  (total_cost : ℝ)
  (h1 : peaches_after_coupon = 12.32)
  (h2 : coupon_value = 3)
  (h3 : total_cost = 23.86) :
  total_cost - (peaches_after_coupon + coupon_value) = 8.54 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_cherries_l927_92776


namespace NUMINAMATH_CALUDE_pi_fourth_in_range_of_f_l927_92768

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem pi_fourth_in_range_of_f : ∃ (x : ℝ), f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_pi_fourth_in_range_of_f_l927_92768


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l927_92767

theorem geometric_sequence_product (a b : ℝ) : 
  2 < a ∧ a < b ∧ b < 16 ∧ 
  (∃ r : ℝ, r > 0 ∧ a = 2 * r ∧ b = 2 * r^2 ∧ 16 = 2 * r^3) →
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l927_92767


namespace NUMINAMATH_CALUDE_magazine_clients_count_l927_92725

/-- The number of clients using magazines in an advertising agency --/
def clients_using_magazines (total : ℕ) (tv : ℕ) (radio : ℕ) (tv_mag : ℕ) (tv_radio : ℕ) (radio_mag : ℕ) (all_three : ℕ) : ℕ :=
  total + all_three - (tv + radio - tv_radio)

/-- Theorem stating the number of clients using magazines --/
theorem magazine_clients_count : 
  clients_using_magazines 180 115 110 85 75 95 80 = 130 := by
  sorry

end NUMINAMATH_CALUDE_magazine_clients_count_l927_92725


namespace NUMINAMATH_CALUDE_largest_number_is_482_l927_92743

/-- Given a systematic sample from a set of products, this function calculates the largest number in the sample. -/
def largest_sample_number (total_products : ℕ) (smallest_number : ℕ) (second_smallest : ℕ) : ℕ :=
  let sampling_interval := second_smallest - smallest_number
  let sample_size := total_products / sampling_interval
  smallest_number + sampling_interval * (sample_size - 1)

/-- Theorem stating that for the given conditions, the largest number in the sample is 482. -/
theorem largest_number_is_482 :
  largest_sample_number 500 7 32 = 482 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_is_482_l927_92743


namespace NUMINAMATH_CALUDE_complex_sum_of_sixth_powers_l927_92754

theorem complex_sum_of_sixth_powers : 
  (((1 : ℂ) + Complex.I * Real.sqrt 3) / 2) ^ 6 + 
  (((1 : ℂ) - Complex.I * Real.sqrt 3) / 2) ^ 6 = 
  (1 : ℂ) / 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_sixth_powers_l927_92754


namespace NUMINAMATH_CALUDE_binary_subtraction_l927_92738

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def a : List Bool := [true, true, true, true, true, true, true, true, true, true, true]
def b : List Bool := [true, true, true, true, true, true, true]

theorem binary_subtraction :
  binary_to_decimal a - binary_to_decimal b = 1920 := by sorry

end NUMINAMATH_CALUDE_binary_subtraction_l927_92738


namespace NUMINAMATH_CALUDE_lowry_bonsai_sales_l927_92781

/-- The number of small bonsai sold by Lowry -/
def small_bonsai_sold : ℕ := 3

/-- The cost of a small bonsai in dollars -/
def small_bonsai_cost : ℕ := 30

/-- The cost of a big bonsai in dollars -/
def big_bonsai_cost : ℕ := 20

/-- The number of big bonsai sold -/
def big_bonsai_sold : ℕ := 5

/-- The total earnings in dollars -/
def total_earnings : ℕ := 190

theorem lowry_bonsai_sales :
  small_bonsai_sold * small_bonsai_cost + big_bonsai_sold * big_bonsai_cost = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_lowry_bonsai_sales_l927_92781
