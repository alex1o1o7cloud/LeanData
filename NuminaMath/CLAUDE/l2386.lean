import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l2386_238696

/-- Represents a stratified sampling scenario in a high school -/
structure StratifiedSample where
  total_students : ℕ
  liberal_arts_students : ℕ
  sample_size : ℕ

/-- Calculates the expected number of liberal arts students in the sample -/
def expected_liberal_arts_in_sample (s : StratifiedSample) : ℕ :=
  (s.liberal_arts_students * s.sample_size) / s.total_students

/-- Theorem stating the expected number of liberal arts students in the sample -/
theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.total_students = 1000)
  (h2 : s.liberal_arts_students = 200)
  (h3 : s.sample_size = 100) :
  expected_liberal_arts_in_sample s = 20 := by
  sorry

#eval expected_liberal_arts_in_sample { total_students := 1000, liberal_arts_students := 200, sample_size := 100 }

end NUMINAMATH_CALUDE_stratified_sample_theorem_l2386_238696


namespace NUMINAMATH_CALUDE_paper_width_is_four_l2386_238618

/-- Given a rectangular paper surrounded by a wall photo, this theorem proves
    that the width of the paper is 4 inches under certain conditions. -/
theorem paper_width_is_four 
  (photo_width : ℝ) 
  (paper_length : ℝ) 
  (photo_area : ℝ) 
  (h1 : photo_width = 2)
  (h2 : paper_length = 8)
  (h3 : photo_area = 96)
  (h4 : photo_area = (paper_length + 2 * photo_width) * (paper_width + 2 * photo_width)) :
  paper_width = 4 :=
by
  sorry

#check paper_width_is_four

end NUMINAMATH_CALUDE_paper_width_is_four_l2386_238618


namespace NUMINAMATH_CALUDE_fliers_sent_afternoon_l2386_238661

theorem fliers_sent_afternoon (total : ℕ) (morning_fraction : ℚ) (left_next_day : ℕ) 
  (h1 : total = 1000)
  (h2 : morning_fraction = 1 / 5)
  (h3 : left_next_day = 600) :
  (total - (morning_fraction * total).num - left_next_day) / 
  (total - (morning_fraction * total).num) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fliers_sent_afternoon_l2386_238661


namespace NUMINAMATH_CALUDE_function_equation_solution_l2386_238642

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y) * (x + f y) = x^2 * f y + y^2 * f x) →
  (∀ x : ℝ, f x = 0 ∨ f x = x) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2386_238642


namespace NUMINAMATH_CALUDE_zero_point_in_interval_l2386_238623

def f (x : ℝ) := -x^3 - 3*x + 5

theorem zero_point_in_interval :
  (∀ x y, x < y → f x > f y) →  -- f is monotonically decreasing
  Continuous f →
  f 1 > 0 →
  f 2 < 0 →
  ∃ c, c ∈ Set.Ioo 1 2 ∧ f c = 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_point_in_interval_l2386_238623


namespace NUMINAMATH_CALUDE_left_of_kolya_l2386_238646

/-- The number of people in a line-up -/
def total_people : ℕ := 29

/-- The number of people to the right of Kolya -/
def right_of_kolya : ℕ := 12

/-- The number of people to the left of Sasha -/
def left_of_sasha : ℕ := 20

/-- The number of people to the right of Sasha -/
def right_of_sasha : ℕ := 8

/-- Theorem: The number of people to the left of Kolya is 16 -/
theorem left_of_kolya : total_people - right_of_kolya - 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_left_of_kolya_l2386_238646


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2386_238619

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x + m ≠ 0) → m > 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2386_238619


namespace NUMINAMATH_CALUDE_fifteenth_row_seats_l2386_238610

/-- Represents the number of seats in a row of the stadium -/
def seats (n : ℕ) : ℕ := 5 + 2 * (n - 1)

/-- Theorem stating that the 15th row has 33 seats -/
theorem fifteenth_row_seats :
  seats 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_row_seats_l2386_238610


namespace NUMINAMATH_CALUDE_sum_remainder_six_l2386_238653

theorem sum_remainder_six (m : ℤ) : (9 - m + (m + 5)) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_six_l2386_238653


namespace NUMINAMATH_CALUDE_second_bag_popped_kernels_l2386_238605

/-- Represents a bag of popcorn kernels -/
structure PopcornBag where
  total : ℕ
  popped : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def popPercentage (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem second_bag_popped_kernels 
  (bag1 : PopcornBag)
  (bag2 : PopcornBag)
  (bag3 : PopcornBag)
  (h1 : bag1.total = 75)
  (h2 : bag1.popped = 60)
  (h3 : bag2.total = 50)
  (h4 : bag3.total = 100)
  (h5 : bag3.popped = 82)
  (h6 : (popPercentage bag1 + popPercentage bag2 + popPercentage bag3) / 3 = 82) :
  bag2.popped = 42 := by
  sorry

#eval PopcornBag.popped { total := 50, popped := 42 }

end NUMINAMATH_CALUDE_second_bag_popped_kernels_l2386_238605


namespace NUMINAMATH_CALUDE_local_extremum_sum_l2386_238636

/-- A function f with a local extremum -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_sum (a b : ℝ) :
  f a b 1 = 10 ∧ f' a b 1 = 0 → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_local_extremum_sum_l2386_238636


namespace NUMINAMATH_CALUDE_circle_inequality_l2386_238649

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_l2386_238649


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2386_238677

theorem fruit_seller_apples (initial_apples : ℕ) : 
  (initial_apples : ℝ) * (1 - 0.4) = 420 → initial_apples = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2386_238677


namespace NUMINAMATH_CALUDE_paige_picture_upload_l2386_238668

/-- The number of pictures Paige uploaded to Facebook -/
def total_pictures : ℕ := 35

/-- The number of pictures in the first album -/
def first_album : ℕ := 14

/-- The number of additional albums -/
def additional_albums : ℕ := 3

/-- The number of pictures in each additional album -/
def pictures_per_additional_album : ℕ := 7

/-- Theorem stating that the total number of pictures uploaded is correct -/
theorem paige_picture_upload :
  total_pictures = first_album + additional_albums * pictures_per_additional_album :=
by sorry

end NUMINAMATH_CALUDE_paige_picture_upload_l2386_238668


namespace NUMINAMATH_CALUDE_field_walking_distance_reduction_l2386_238660

theorem field_walking_distance_reduction : 
  let field_width : ℝ := 6
  let field_height : ℝ := 8
  let daniel_distance := field_width + field_height
  let rachel_distance := Real.sqrt (field_width^2 + field_height^2)
  let percentage_reduction := (daniel_distance - rachel_distance) / daniel_distance * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ abs (percentage_reduction - 29) < ε :=
by sorry

end NUMINAMATH_CALUDE_field_walking_distance_reduction_l2386_238660


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2386_238612

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
    a = 1 → 
    b = 3 → 
    c^2 = a^2 + b^2 → 
    c = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2386_238612


namespace NUMINAMATH_CALUDE_simplify_expression_l2386_238608

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x * y - 4 ≠ 0) :
  (x^2 - 4 / y) / (y^2 - 4 / x) = x / y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2386_238608


namespace NUMINAMATH_CALUDE_lindas_savings_l2386_238658

theorem lindas_savings (savings : ℝ) (tv_cost : ℝ) : 
  tv_cost = 240 →
  (1 / 4 : ℝ) * savings = tv_cost →
  savings = 960 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l2386_238658


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2386_238652

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5 ∧ 
  ¬(a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2386_238652


namespace NUMINAMATH_CALUDE_independence_test_most_appropriate_l2386_238647

/-- Represents the survey data in a 2x2 contingency table --/
structure SurveyData where
  male_total : ℕ
  male_doping : ℕ
  female_total : ℕ
  female_framed : ℕ

/-- Represents different statistical methods --/
inductive StatMethod
  | MeanVariance
  | RegressionAnalysis
  | IndependenceTest
  | Probability

/-- Checks if a method is most appropriate for analyzing the given survey data --/
def is_most_appropriate (method : StatMethod) (data : SurveyData) : Prop :=
  method = StatMethod.IndependenceTest

/-- The main theorem stating that the Independence Test is the most appropriate method --/
theorem independence_test_most_appropriate (data : SurveyData) :
  is_most_appropriate StatMethod.IndependenceTest data :=
sorry

end NUMINAMATH_CALUDE_independence_test_most_appropriate_l2386_238647


namespace NUMINAMATH_CALUDE_root_negative_implies_inequality_l2386_238624

theorem root_negative_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 2*a + 4 = 0 ∧ x < 0) → (a - 3) * (a - 4) > 0 := by
  sorry

end NUMINAMATH_CALUDE_root_negative_implies_inequality_l2386_238624


namespace NUMINAMATH_CALUDE_find_S_value_l2386_238680

/-- Represents the relationship between R, S, and T -/
def relationship (R S T : ℝ) : Prop :=
  ∃ (c : ℝ), R = c * S / T

theorem find_S_value (R₁ S₁ T₁ R₂ T₂ : ℝ) :
  relationship R₁ S₁ T₁ →
  R₁ = 4/3 →
  S₁ = 3/7 →
  T₁ = 9/14 →
  R₂ = Real.sqrt 48 →
  T₂ = Real.sqrt 75 →
  ∃ (S₂ : ℝ), relationship R₂ S₂ T₂ ∧ S₂ = 30 :=
by sorry

end NUMINAMATH_CALUDE_find_S_value_l2386_238680


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l2386_238688

-- Define the line l: x - y + a = 0
def line_l (a : ℝ) (x y : ℝ) : Prop := x - y + a = 0

-- Define points M and N
def point_M : ℝ × ℝ := (-2, 0)
def point_N : ℝ × ℝ := (-1, 0)

-- Define the distance ratio condition for point Q
def distance_ratio (x y : ℝ) : Prop :=
  Real.sqrt ((x + 2)^2 + y^2) / Real.sqrt ((x + 1)^2 + y^2) = Real.sqrt 2

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the perpendicularity condition
def perpendicular_vectors (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_intersection :
  -- Part I: Prove that the trajectory of Q is the circle C
  (∀ x y : ℝ, distance_ratio x y ↔ circle_C x y) ∧
  -- Part II: Prove that when l intersects C at two points with perpendicular position vectors, a = ±√2
  (∀ a x₁ y₁ x₂ y₂ : ℝ,
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_l a x₁ y₁ ∧ line_l a x₂ y₂ ∧
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    perpendicular_vectors x₁ y₁ x₂ y₂ →
    a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l2386_238688


namespace NUMINAMATH_CALUDE_tangent_line_implies_k_value_l2386_238631

/-- Given a curve y = 3ln(x) + x + k, where k ∈ ℝ, if there exists a point P(x₀, y₀) on the curve
    such that the tangent line at P has the equation 4x - y - 1 = 0, then k = 2. -/
theorem tangent_line_implies_k_value (k : ℝ) (x₀ y₀ : ℝ) :
  y₀ = 3 * Real.log x₀ + x₀ + k →
  (∀ x y, y = 4 * x - 1 ↔ 4 * x - y - 1 = 0) →
  (∃ m b, ∀ x, 3 / x + 1 = m ∧ y₀ - m * x₀ = b ∧ y₀ = 4 * x₀ - 1) →
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_k_value_l2386_238631


namespace NUMINAMATH_CALUDE_expression_evaluation_l2386_238651

theorem expression_evaluation :
  let x : ℚ := 6
  let y : ℚ := -1/6
  let expr := 7 * x^2 * y - (3*x*y - 2*(x*y - 7/2*x^2*y + 1) + 1/2*x*y)
  expr = 7/2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2386_238651


namespace NUMINAMATH_CALUDE_solve_for_m_l2386_238616

theorem solve_for_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l2386_238616


namespace NUMINAMATH_CALUDE_f_monotonic_k_range_l2386_238656

def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem f_monotonic_k_range :
  ∀ k : ℝ, (monotonic_on (f k) 1 2) → k ≤ 8 ∨ k ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_f_monotonic_k_range_l2386_238656


namespace NUMINAMATH_CALUDE_rationalize_result_l2386_238699

def rationalize_denominator (a b c : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ := sorry

theorem rationalize_result :
  let (A, B, C, D, E) := rationalize_denominator 5 7 13
  A = -4 ∧ B = 7 ∧ C = 3 ∧ D = 13 ∧ E = 1 ∧ B < D ∧
  A * Real.sqrt B + C * Real.sqrt D = 5 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) * E :=
by sorry

end NUMINAMATH_CALUDE_rationalize_result_l2386_238699


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2386_238691

open Set Real

theorem inequality_solution_set : 
  let S := {x : ℝ | (π/2)^((x-1)^2) ≤ (2/π)^(x^2-5*x-5)}
  S = Icc (-1/2) 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2386_238691


namespace NUMINAMATH_CALUDE_linear_function_property_l2386_238674

/-- A linear function is a function of the form f(x) = mx + b for some constants m and b -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hlinear : LinearFunction g) (hcond : g 4 - g 1 = 9) : 
  g 10 - g 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2386_238674


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_two_variables_l2386_238643

theorem arithmetic_geometric_mean_inequality_two_variables
  (a b : ℝ) : (a^2 + b^2) / 2 ≥ a * b ∧ 
  ((a^2 + b^2) / 2 = a * b ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_two_variables_l2386_238643


namespace NUMINAMATH_CALUDE_inequality_preservation_l2386_238698

theorem inequality_preservation (x y : ℝ) (h : x > y) : x / 5 > y / 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2386_238698


namespace NUMINAMATH_CALUDE_sum_of_c_values_l2386_238682

theorem sum_of_c_values (b c : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + b*z + c = 0 ↔ (z = x ∨ z = y)) →
  b = c - 1 →
  ∃ c₁ c₂ : ℝ, (∀ c' : ℝ, (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + (c' - 1)*z + c' = 0 ↔ (z = x ∨ z = y)) ↔ (c' = c₁ ∨ c' = c₂)) ∧
  c₁ + c₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l2386_238682


namespace NUMINAMATH_CALUDE_cone_generatrix_length_l2386_238666

theorem cone_generatrix_length (r : ℝ) (h1 : r = Real.sqrt 2) :
  let l := 2 * Real.sqrt 2
  (2 * Real.pi * r = Real.pi * l) → l = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cone_generatrix_length_l2386_238666


namespace NUMINAMATH_CALUDE_monotonicity_condition_solution_set_l2386_238602

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x - 2*a

-- Theorem for monotonicity condition
theorem monotonicity_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 3, Monotone (f a)) ↔ (a ≥ -1/2 ∨ a ≤ -5/2) := by sorry

-- Theorem for solution set of f(x) < 0
theorem solution_set (a : ℝ) :
  {x : ℝ | f a x < 0} = 
    if a = -1/2 then ∅ 
    else if a < -1/2 then Set.Ioo 1 (-2*a)
    else Set.Ioo (-2*a) 1 := by sorry

end NUMINAMATH_CALUDE_monotonicity_condition_solution_set_l2386_238602


namespace NUMINAMATH_CALUDE_value_of_x_l2386_238686

theorem value_of_x (x y z : ℚ) : 
  x = y / 3 →
  y = z / 4 →
  z = 80 →
  x = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l2386_238686


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l2386_238659

theorem complex_arithmetic_equality : -6 / 2 + (1/3 - 3/4) * 12 + (-3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l2386_238659


namespace NUMINAMATH_CALUDE_band_members_count_l2386_238669

/-- Calculates the number of band members given the earnings per member, total earnings, and number of gigs. -/
def band_members (earnings_per_member : ℕ) (total_earnings : ℕ) (num_gigs : ℕ) : ℕ :=
  (total_earnings / num_gigs) / earnings_per_member

/-- Proves that the number of band members is 4 given the specified conditions. -/
theorem band_members_count : band_members 20 400 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_band_members_count_l2386_238669


namespace NUMINAMATH_CALUDE_poster_area_is_zero_l2386_238687

theorem poster_area_is_zero (x y : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (3 * x + 5) * (y + 3) = x * y + 57) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_poster_area_is_zero_l2386_238687


namespace NUMINAMATH_CALUDE_composite_sum_product_l2386_238611

theorem composite_sum_product (a b c d e : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a^4 + b^4 = c^4 + d^4 ∧
  a^4 + b^4 = e^5 →
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x * y = a * c + b * d :=
by sorry

end NUMINAMATH_CALUDE_composite_sum_product_l2386_238611


namespace NUMINAMATH_CALUDE_tobias_daily_hours_l2386_238638

/-- Proves that Tobias plays 5 hours per day given the conditions of the problem -/
theorem tobias_daily_hours (nathan_daily_hours : ℕ) (nathan_days : ℕ) (tobias_days : ℕ) (total_hours : ℕ) :
  nathan_daily_hours = 3 →
  nathan_days = 14 →
  tobias_days = 7 →
  total_hours = 77 →
  ∃ (tobias_daily_hours : ℕ), 
    tobias_daily_hours * tobias_days + nathan_daily_hours * nathan_days = total_hours ∧
    tobias_daily_hours = 5 :=
by sorry

end NUMINAMATH_CALUDE_tobias_daily_hours_l2386_238638


namespace NUMINAMATH_CALUDE_robins_walk_distance_l2386_238678

/-- The total distance Robin walks given his journey to the city center -/
theorem robins_walk_distance (house_to_center : ℕ) (initial_walk : ℕ) : 
  house_to_center = 500 →
  initial_walk = 200 →
  initial_walk + initial_walk + house_to_center = 900 := by
  sorry

end NUMINAMATH_CALUDE_robins_walk_distance_l2386_238678


namespace NUMINAMATH_CALUDE_tom_speed_l2386_238645

theorem tom_speed (karen_speed : ℝ) (karen_delay : ℝ) (win_distance : ℝ) (tom_distance : ℝ) :
  karen_speed = 60 ∧ 
  karen_delay = 4 / 60 ∧ 
  win_distance = 4 ∧ 
  tom_distance = 24 → 
  ∃ (tom_speed : ℝ), tom_speed = 60 :=
by sorry

end NUMINAMATH_CALUDE_tom_speed_l2386_238645


namespace NUMINAMATH_CALUDE_function_inequality_l2386_238663

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h' : ∀ x, (x - 2) * deriv f x ≤ 0) : 
  f (-3) + f 3 ≤ 2 * f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2386_238663


namespace NUMINAMATH_CALUDE_ratio_problem_l2386_238640

theorem ratio_problem (a b c x : ℝ) 
  (h1 : a / c = 3 / 7)
  (h2 : b / c = x / 7)
  (h3 : (a + b + c) / c = 2) :
  b / (a + c) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2386_238640


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2386_238681

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) = (a^2 - 4*a + 3) + Complex.I * (a - 1)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2386_238681


namespace NUMINAMATH_CALUDE_parabola_coefficient_l2386_238667

/-- A quadratic function of the form y = mx^2 + 2 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 2

/-- The condition for a downward-opening parabola -/
def is_downward_opening (m : ℝ) : Prop := m < 0

theorem parabola_coefficient :
  ∀ m : ℝ, is_downward_opening m → m = -2 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l2386_238667


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2386_238622

theorem simplify_and_evaluate (x y : ℚ) 
  (hx : x = 1) (hy : y = 1/2) : 
  (3*x + 2*y) * (3*x - 2*y) - (x - y)^2 = 31/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2386_238622


namespace NUMINAMATH_CALUDE_triangle_inequality_l2386_238604

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) :
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2386_238604


namespace NUMINAMATH_CALUDE_vector_problem_l2386_238617

/-- Given vectors in 2D space -/
def OA : Fin 2 → ℝ := ![1, -2]
def OB : Fin 2 → ℝ := ![4, -1]
def OC (m : ℝ) : Fin 2 → ℝ := ![m, m + 1]

/-- Vector AB -/
def AB : Fin 2 → ℝ := ![3, 1]

/-- Vector AC -/
def AC (m : ℝ) : Fin 2 → ℝ := ![m - 1, m + 3]

/-- Vector BC -/
def BC (m : ℝ) : Fin 2 → ℝ := ![m - 4, m + 2]

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

/-- Two vectors are perpendicular if their dot product is zero -/
def are_perpendicular (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 0 + v 1 * w 1 = 0

/-- Triangle ABC is right-angled if any two of its sides are perpendicular -/
def is_right_angled (m : ℝ) : Prop :=
  are_perpendicular AB (AC m) ∨ are_perpendicular AB (BC m) ∨ are_perpendicular (AC m) (BC m)

theorem vector_problem (m : ℝ) :
  (are_parallel AB (OC m) → m = -3/2) ∧
  (is_right_angled m → m = 0 ∨ m = 5/2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2386_238617


namespace NUMINAMATH_CALUDE_smallest_dual_palindrome_l2386_238690

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_palindrome : 
  ∀ n : ℕ, n > 8 → isPalindromeInBase n 2 → isPalindromeInBase n 8 → n ≥ 63 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_palindrome_l2386_238690


namespace NUMINAMATH_CALUDE_larry_gave_brother_l2386_238633

def larry_problem (initial_amount lunch_expense final_amount : ℕ) : Prop :=
  initial_amount - lunch_expense - final_amount = 2

theorem larry_gave_brother : 
  larry_problem 22 5 15 := by sorry

end NUMINAMATH_CALUDE_larry_gave_brother_l2386_238633


namespace NUMINAMATH_CALUDE_locus_of_constant_sum_distances_l2386_238684

-- Define a type for lines in a plane
structure Line where
  -- Add necessary fields to represent a line

-- Define a type for points in a plane
structure Point where
  -- Add necessary fields to represent a point

-- Define a function to calculate the distance between a point and a line
def distance (p : Point) (l : Line) : ℝ :=
  sorry

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  sorry

-- Define a type for the locus
inductive Locus
  | Region
  | Parallelogram
  | Octagon

-- State the theorem
theorem locus_of_constant_sum_distances 
  (l1 l2 m1 m2 : Line) 
  (h_parallel1 : are_parallel l1 l2) 
  (h_parallel2 : are_parallel m1 m2) 
  (sum : ℝ) :
  ∃ (locus : Locus),
    ∀ (p : Point),
      distance p l1 + distance p l2 + distance p m1 + distance p m2 = sum →
      (((are_parallel l1 m1) ∧ (locus = Locus.Region)) ∨
       ((¬are_parallel l1 m1) ∧ ((locus = Locus.Parallelogram) ∨ (locus = Locus.Octagon)))) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_constant_sum_distances_l2386_238684


namespace NUMINAMATH_CALUDE_work_completion_men_count_l2386_238685

/-- Given that 42 men can complete a piece of work in 18 days, and the same work
    can be completed by a different number of men in 28 days, prove that the number
    of men in the second group is 27. -/
theorem work_completion_men_count :
  let work_rate_1 : ℚ := 42 / 18  -- Work rate of the first group (men per day)
  let work_rate_2 : ℚ := 1 / 28   -- Work rate of one man in the second group
  ∃ (n : ℕ), n * work_rate_2 = work_rate_1 ∧ n = 27 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_men_count_l2386_238685


namespace NUMINAMATH_CALUDE_zero_point_implies_a_range_l2386_238683

/-- Given a function y = x³ - ax where x ∈ ℝ and y has a zero point at (1, 2),
    prove that a ∈ (1, 4) -/
theorem zero_point_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, x^3 - a*x = 0) →
  a ∈ Set.Ioo 1 4 := by
  sorry

end NUMINAMATH_CALUDE_zero_point_implies_a_range_l2386_238683


namespace NUMINAMATH_CALUDE_games_to_give_away_l2386_238670

def initial_games : ℕ := 50
def desired_games : ℕ := 35

theorem games_to_give_away :
  initial_games - desired_games = 15 :=
by sorry

end NUMINAMATH_CALUDE_games_to_give_away_l2386_238670


namespace NUMINAMATH_CALUDE_floor_sum_eval_l2386_238609

theorem floor_sum_eval : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_eval_l2386_238609


namespace NUMINAMATH_CALUDE_equation_solution_l2386_238676

theorem equation_solution : ∃ x : ℝ, (81 : ℝ) ^ (x - 1) / (9 : ℝ) ^ (x + 1) = (729 : ℝ) ^ (x + 2) ∧ x = -9/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2386_238676


namespace NUMINAMATH_CALUDE_triangle_properties_l2386_238630

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 3 * (t.b^2 + t.c^2) = 3 * t.a^2 + 2 * t.b * t.c)
  (h2 : t.a = 2)
  (h3 : t.b + t.c = 2 * Real.sqrt 2)
  (h4 : Real.sin t.B = Real.sqrt 2 * Real.cos t.C) :
  (∃ S : ℝ, S = Real.sqrt 2 / 2 ∧ S = 1/2 * t.b * t.c * Real.sin t.A) ∧ 
  Real.cos t.C = Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l2386_238630


namespace NUMINAMATH_CALUDE_card_distribution_events_l2386_238607

-- Define the set of cards
inductive Card : Type
| Red : Card
| Yellow : Card
| Blue : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsRed (d : Distribution) : Prop := d Person.B = Card.Red

-- State the theorem
theorem card_distribution_events :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsRed d)) ∧
  -- The events are not opposite (there exists a distribution where neither event occurs)
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsRed d) :=
sorry

end NUMINAMATH_CALUDE_card_distribution_events_l2386_238607


namespace NUMINAMATH_CALUDE_perimeter_of_square_C_l2386_238639

/-- Given squares A, B, and C with specific perimeter relationships, 
    prove that the perimeter of C is 100. -/
theorem perimeter_of_square_C (A B C : ℝ) : 
  (A > 0) →  -- A is positive (side length of a square)
  (B > 0) →  -- B is positive (side length of a square)
  (C > 0) →  -- C is positive (side length of a square)
  (4 * A = 20) →  -- Perimeter of A is 20
  (4 * B = 40) →  -- Perimeter of B is 40
  (C = A + 2 * B) →  -- Side length of C relationship
  (4 * C = 100) :=  -- Perimeter of C is 100
by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_C_l2386_238639


namespace NUMINAMATH_CALUDE_discontinuity_at_three_l2386_238637

/-- The function f(x) = 6 / (x-3)² is discontinuous at x = 3 -/
theorem discontinuity_at_three (f : ℝ → ℝ) (h : ∀ x ≠ 3, f x = 6 / (x - 3)^2) :
  ¬ ContinuousAt f 3 := by
  sorry

end NUMINAMATH_CALUDE_discontinuity_at_three_l2386_238637


namespace NUMINAMATH_CALUDE_tan_three_expression_zero_l2386_238644

theorem tan_three_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (2 - 2 * Real.cos θ) / Real.sin θ - Real.sin θ / (2 + 2 * Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_three_expression_zero_l2386_238644


namespace NUMINAMATH_CALUDE_suzannes_book_pages_l2386_238695

theorem suzannes_book_pages : 
  ∀ (pages_monday pages_tuesday pages_left : ℕ),
    pages_monday = 15 →
    pages_tuesday = pages_monday + 16 →
    pages_left = 18 →
    pages_monday + pages_tuesday + pages_left = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_suzannes_book_pages_l2386_238695


namespace NUMINAMATH_CALUDE_count_divisible_by_five_l2386_238614

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A function to check if a three-digit number is valid (no leading zero) --/
def isValidNumber (n : Nat) : Bool :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 ≠ 0)

/-- A function to check if a number is formed from distinct digits in the given set --/
def isFromDistinctDigits (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The set of valid three-digit numbers formed from the given digits --/
def validNumbers : Finset Nat :=
  Finset.filter (fun n => isValidNumber n ∧ isFromDistinctDigits n) (Finset.range 1000)

/-- The theorem to be proved --/
theorem count_divisible_by_five :
  (validNumbers.filter (fun n => n % 5 = 0)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_five_l2386_238614


namespace NUMINAMATH_CALUDE_movie_ticket_cost_l2386_238657

/-- The cost of movie tickets for a family --/
theorem movie_ticket_cost (C : ℝ) : 
  (∃ (A : ℝ), 
    A = C + 3.25 ∧ 
    2 * A + 4 * C - 2 = 30) → 
  C = 4.25 := by
sorry

end NUMINAMATH_CALUDE_movie_ticket_cost_l2386_238657


namespace NUMINAMATH_CALUDE_quadratic_sum_l2386_238641

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 - 28 * x - 48

-- Define the completed square form
def g (x a b c : ℝ) : ℝ := a * (x + b)^2 + c

-- Theorem statement
theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f x = g x a b c) → a + b + c = -96.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2386_238641


namespace NUMINAMATH_CALUDE_exactly_two_approvals_probability_l2386_238600

/-- The probability of success in a single trial -/
def p : ℝ := 0.6

/-- The number of trials -/
def n : ℕ := 5

/-- The number of desired successes -/
def k : ℕ := 2

/-- The probability of exactly k successes in n independent trials with probability p -/
def binomial_probability (p : ℝ) (n k : ℕ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_approvals_probability :
  binomial_probability p n k = 0.3648 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_approvals_probability_l2386_238600


namespace NUMINAMATH_CALUDE_exam_failure_percentage_l2386_238626

theorem exam_failure_percentage :
  let total_candidates : ℕ := 2000
  let girls : ℕ := 900
  let boys : ℕ := total_candidates - girls
  let boys_pass_rate : ℚ := 34 / 100
  let girls_pass_rate : ℚ := 32 / 100
  let passed_candidates : ℚ := boys_pass_rate * boys + girls_pass_rate * girls
  let failed_candidates : ℚ := total_candidates - passed_candidates
  let failure_percentage : ℚ := failed_candidates / total_candidates * 100
  failure_percentage = 669 / 10 := by sorry

end NUMINAMATH_CALUDE_exam_failure_percentage_l2386_238626


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2386_238625

theorem cubic_equation_solutions :
  let f : ℂ → ℂ := λ x => (x^3 - 4*x^2*(Real.sqrt 3) + 12*x - 8*(Real.sqrt 3)) + (2*x - 2*(Real.sqrt 3))
  ∃ (z₁ z₂ z₃ : ℂ),
    f z₁ = 0 ∧ f z₂ = 0 ∧ f z₃ = 0 ∧
    z₁ = 2 * Real.sqrt 3 ∧
    z₂ = 2 * Real.sqrt 3 + Complex.I * Real.sqrt 2 ∧
    z₃ = 2 * Real.sqrt 3 - Complex.I * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2386_238625


namespace NUMINAMATH_CALUDE_range_of_a_for_p_range_of_a_for_p_or_q_and_not_p_and_q_l2386_238697

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x ≥ 1 ∧ 4^x + 2^(x+1) - 7 - a < 0

-- Theorem for part 1
theorem range_of_a_for_p :
  {a : ℝ | p a} = {a : ℝ | 0 ≤ a ∧ a < 4} :=
sorry

-- Theorem for part 2
theorem range_of_a_for_p_or_q_and_not_p_and_q :
  {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} = {a : ℝ | (0 ≤ a ∧ a ≤ 1) ∨ a ≥ 4} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_p_range_of_a_for_p_or_q_and_not_p_and_q_l2386_238697


namespace NUMINAMATH_CALUDE_water_for_bread_dough_l2386_238673

/-- The amount of water (in mL) needed for a given amount of flour (in mL),
    given a water-to-flour ratio. -/
def water_needed (water_ratio : ℚ) (flour_amount : ℚ) : ℚ :=
  (water_ratio * flour_amount)

/-- Theorem stating that for 1000 mL of flour, given the ratio of 80 mL water
    to 200 mL flour, the amount of water needed is 400 mL. -/
theorem water_for_bread_dough : water_needed (80 / 200) 1000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_water_for_bread_dough_l2386_238673


namespace NUMINAMATH_CALUDE_product_with_zero_is_zero_l2386_238601

theorem product_with_zero_is_zero :
  (-2.5) * 0.37 * 1.25 * (-4) * (-8) * 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_with_zero_is_zero_l2386_238601


namespace NUMINAMATH_CALUDE_proposition_1_proposition_3_l2386_238693

-- Proposition ①
theorem proposition_1 : ∀ a b : ℝ, (a + b ≠ 5) → (a ≠ 2 ∨ b ≠ 3) := by sorry

-- Proposition ③
theorem proposition_3 : 
  (∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ∧ 
  (∀ ε > 0, ∃ x : ℝ, x > 0 ∧ x + 1/x < 2 + ε) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_3_l2386_238693


namespace NUMINAMATH_CALUDE_single_positive_root_condition_l2386_238662

theorem single_positive_root_condition (k : ℝ) :
  (∃! x : ℝ, x > 0 ∧ (x^2 + k*x + 3) / (x - 1) = 3*x + k) ↔
  (k = -33/8 ∨ k = -4 ∨ k ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_single_positive_root_condition_l2386_238662


namespace NUMINAMATH_CALUDE_prime_power_plus_two_l2386_238689

theorem prime_power_plus_two (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → Prime (p^3 + 2) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_plus_two_l2386_238689


namespace NUMINAMATH_CALUDE_sum_of_a_equals_two_l2386_238628

theorem sum_of_a_equals_two (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (eq1 : 2*a₁ + a₂ + a₃ + a₄ + a₅ = 1 + (1/8)*a₄)
  (eq2 : 2*a₂ + a₃ + a₄ + a₅ = 2 + (1/4)*a₃)
  (eq3 : 2*a₃ + a₄ + a₅ = 4 + (1/2)*a₂)
  (eq4 : 2*a₄ + a₅ = 6 + a₁) :
  a₁ + a₂ + a₃ + a₄ + a₅ = 2 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_a_equals_two_l2386_238628


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2386_238627

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2386_238627


namespace NUMINAMATH_CALUDE_smallest_k_remainder_l2386_238629

theorem smallest_k_remainder (k : ℕ) : 
  k > 0 ∧ 
  k % 5 = 2 ∧ 
  k % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 5 = 2 ∧ m % 6 = 5 → k ≤ m) → 
  k % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_k_remainder_l2386_238629


namespace NUMINAMATH_CALUDE_parabola_b_value_l2386_238664

/-- Prove that for a parabola y = 2x^2 + bx + 3 passing through (1, 2) and (-2, -1), b = 11/2 -/
theorem parabola_b_value (b : ℝ) : 
  (2 * (1 : ℝ)^2 + b * 1 + 3 = 2) ∧ 
  (2 * (-2 : ℝ)^2 + b * (-2) + 3 = -1) → 
  b = 11/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l2386_238664


namespace NUMINAMATH_CALUDE_theft_culprits_l2386_238621

-- Define the guilt status of each person
variable (E F G : Prop)

-- E represents "Elise is guilty"
-- F represents "Fred is guilty"
-- G represents "Gaétan is guilty"

-- Define the given conditions
axiom cond1 : ¬G → F
axiom cond2 : ¬E → G
axiom cond3 : G → E
axiom cond4 : E → ¬F

-- Theorem to prove
theorem theft_culprits : E ∧ G ∧ ¬F := by
  sorry

end NUMINAMATH_CALUDE_theft_culprits_l2386_238621


namespace NUMINAMATH_CALUDE_pigeon_problem_solution_l2386_238694

/-- Represents the number of pigeons on the branches and under the tree -/
structure PigeonCount where
  onBranches : ℕ
  underTree : ℕ

/-- The conditions of the pigeon problem -/
def satisfiesPigeonConditions (p : PigeonCount) : Prop :=
  (p.underTree - 1 = (p.onBranches + 1) / 3) ∧
  (p.onBranches - 1 = p.underTree + 1)

/-- The theorem stating the solution to the pigeon problem -/
theorem pigeon_problem_solution :
  ∃ (p : PigeonCount), satisfiesPigeonConditions p ∧ p.onBranches = 7 ∧ p.underTree = 5 := by
  sorry


end NUMINAMATH_CALUDE_pigeon_problem_solution_l2386_238694


namespace NUMINAMATH_CALUDE_sequence_property_l2386_238679

theorem sequence_property (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = 2^n) →
  a 1 = 1 →
  a 101 = 2^5050 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l2386_238679


namespace NUMINAMATH_CALUDE_museum_paintings_l2386_238603

theorem museum_paintings (initial : ℕ) (final : ℕ) (removed : ℕ) : 
  initial = 98 → final = 95 → removed = initial - final → removed = 3 := by
  sorry

end NUMINAMATH_CALUDE_museum_paintings_l2386_238603


namespace NUMINAMATH_CALUDE_restaurant_pie_days_l2386_238654

/-- Given a restaurant that sells a constant number of pies per day,
    calculate the number of days based on the total pies sold. -/
theorem restaurant_pie_days (pies_per_day : ℕ) (total_pies : ℕ) (h1 : pies_per_day = 8) (h2 : total_pies = 56) :
  total_pies / pies_per_day = 7 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_pie_days_l2386_238654


namespace NUMINAMATH_CALUDE_change_in_expression_l2386_238635

theorem change_in_expression (x : ℝ) (b : ℕ+) : 
  let f : ℝ → ℝ := λ t => t^2 - 5*t + 6
  (f (x + b) - f x = 2*b*x + b^2 - 5*b) ∧ 
  (f (x - b) - f x = -2*b*x + b^2 + 5*b) := by
  sorry

end NUMINAMATH_CALUDE_change_in_expression_l2386_238635


namespace NUMINAMATH_CALUDE_determinant_2x2_matrix_l2386_238692

theorem determinant_2x2_matrix (x : ℝ) :
  Matrix.det !![5, x; -3, 9] = 45 + 3 * x := by sorry

end NUMINAMATH_CALUDE_determinant_2x2_matrix_l2386_238692


namespace NUMINAMATH_CALUDE_halloween_candy_problem_l2386_238615

/-- Given Katie's candy count, her sister's candy count, and the number of pieces eaten,
    calculate the remaining candy pieces. -/
theorem halloween_candy_problem (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) :
  katie_candy = 10 →
  sister_candy = 6 →
  eaten_candy = 9 →
  katie_candy + sister_candy - eaten_candy = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_problem_l2386_238615


namespace NUMINAMATH_CALUDE_bulb_arrangement_count_l2386_238606

/-- The number of ways to arrange blue, red, and white bulbs in a garland with no consecutive white bulbs -/
def bulb_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red) blue * Nat.choose (blue + red + 1) white

/-- Theorem: The number of ways to arrange 5 blue, 8 red, and 11 white bulbs in a garland 
    with no consecutive white bulbs is equal to (13 choose 5) * (14 choose 11) -/
theorem bulb_arrangement_count : bulb_arrangements 5 8 11 = 468468 := by
  sorry

#eval bulb_arrangements 5 8 11

end NUMINAMATH_CALUDE_bulb_arrangement_count_l2386_238606


namespace NUMINAMATH_CALUDE_farm_pigs_count_l2386_238650

/-- The number of pigs remaining in a barn after changes -/
def pigs_remaining (initial : ℕ) (joined : ℕ) (moved : ℕ) : ℕ :=
  initial + joined - moved

/-- Theorem stating that given the initial conditions, the number of pigs remaining is 431 -/
theorem farm_pigs_count : pigs_remaining 364 145 78 = 431 := by
  sorry

end NUMINAMATH_CALUDE_farm_pigs_count_l2386_238650


namespace NUMINAMATH_CALUDE_foci_coincide_l2386_238634

/-- The value of m for which the foci of the given hyperbola and ellipse coincide -/
theorem foci_coincide (m : ℝ) : 
  (∀ x y : ℝ, y^2/2 - x^2/m = 1 ↔ (y^2/2 = 1 + x^2/m)) ∧ 
  (∀ x y : ℝ, x^2/4 + y^2/9 = 1) ∧
  (∃ c : ℝ, c^2 = 2 + m ∧ c^2 = 5) →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_foci_coincide_l2386_238634


namespace NUMINAMATH_CALUDE_product_of_fractions_l2386_238675

theorem product_of_fractions : (2 : ℚ) / 3 * 3 / 4 * 4 / 5 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2386_238675


namespace NUMINAMATH_CALUDE_number_difference_l2386_238672

theorem number_difference (x y : ℕ) : 
  x + y = 50 → 
  y = 31 → 
  x < 2 * y → 
  2 * y - x = 43 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2386_238672


namespace NUMINAMATH_CALUDE_square_diff_sum_l2386_238632

theorem square_diff_sum : 1010^2 - 990^2 - 1005^2 + 995^2 + 1012^2 - 988^2 = 68000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_sum_l2386_238632


namespace NUMINAMATH_CALUDE_triangle_rotation_l2386_238671

/-- Triangle OPQ with specific properties -/
structure TriangleOPQ where
  O : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_O : O = (0, 0)
  h_Q : Q = (6, 0)
  h_P_first_quadrant : P.1 > 0 ∧ P.2 > 0
  h_right_angle : (P.1 - Q.1) * (Q.1 - O.1) + (P.2 - Q.2) * (Q.2 - O.2) = 0
  h_45_degree : (P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2) = 
                Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) * Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) / Real.sqrt 2

/-- Rotation of a point 90 degrees counterclockwise about the origin -/
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

/-- The main theorem -/
theorem triangle_rotation (t : TriangleOPQ) : rotate90 t.P = (-6, 6) := by
  sorry


end NUMINAMATH_CALUDE_triangle_rotation_l2386_238671


namespace NUMINAMATH_CALUDE_trigonometric_equation_has_solution_l2386_238655

theorem trigonometric_equation_has_solution :
  ∃ x : ℝ, 2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) -
           3 * Real.sin (Real.pi - x) * Real.cos x +
           Real.sin (Real.pi / 2 + x) * Real.cos x = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_has_solution_l2386_238655


namespace NUMINAMATH_CALUDE_conference_men_count_l2386_238620

/-- The number of men at a climate conference -/
def number_of_men : ℕ := 700

/-- The number of women at the conference -/
def number_of_women : ℕ := 500

/-- The number of children at the conference -/
def number_of_children : ℕ := 800

/-- The percentage of men who were Indian -/
def indian_men_percentage : ℚ := 20 / 100

/-- The percentage of women who were Indian -/
def indian_women_percentage : ℚ := 40 / 100

/-- The percentage of children who were Indian -/
def indian_children_percentage : ℚ := 10 / 100

/-- The percentage of people who were not Indian -/
def non_indian_percentage : ℚ := 79 / 100

theorem conference_men_count :
  let total_people := number_of_men + number_of_women + number_of_children
  let indian_people := (indian_men_percentage * number_of_men) + 
                       (indian_women_percentage * number_of_women) + 
                       (indian_children_percentage * number_of_children)
  (1 - non_indian_percentage) * total_people = indian_people →
  number_of_men = 700 := by sorry

end NUMINAMATH_CALUDE_conference_men_count_l2386_238620


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l2386_238665

/-- Theater ticket sales problem -/
theorem theater_ticket_sales
  (orchestra_price : ℕ)
  (balcony_price : ℕ)
  (total_tickets : ℕ)
  (balcony_orchestra_diff : ℕ)
  (ho : orchestra_price = 12)
  (hb : balcony_price = 8)
  (ht : total_tickets = 340)
  (hd : balcony_orchestra_diff = 40) :
  let orchestra_tickets := (total_tickets - balcony_orchestra_diff) / 2
  let balcony_tickets := total_tickets - orchestra_tickets
  orchestra_tickets * orchestra_price + balcony_tickets * balcony_price = 3320 :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_sales_l2386_238665


namespace NUMINAMATH_CALUDE_complement_of_intersection_l2386_238648

/-- The universal set U -/
def U : Set Nat := {0, 1, 2, 3}

/-- The set M -/
def M : Set Nat := {0, 1, 2}

/-- The set N -/
def N : Set Nat := {1, 2, 3}

/-- Theorem stating that the complement of M ∩ N in U is {0, 3} -/
theorem complement_of_intersection (U M N : Set Nat) (hU : U = {0, 1, 2, 3}) (hM : M = {0, 1, 2}) (hN : N = {1, 2, 3}) :
  (M ∩ N)ᶜ = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l2386_238648


namespace NUMINAMATH_CALUDE_solid_max_volume_l2386_238613

/-- The side length of each cube in centimeters -/
def cube_side_length : ℝ := 3

/-- The number of cubes in the base layer -/
def base_layer_cubes : ℕ := 4 * 4

/-- The number of cubes in the second layer -/
def second_layer_cubes : ℕ := 2 * 2

/-- The total number of cubes in the solid -/
def total_cubes : ℕ := base_layer_cubes + second_layer_cubes

/-- The volume of a single cube in cubic centimeters -/
def single_cube_volume : ℝ := cube_side_length ^ 3

/-- The maximum volume of the solid in cubic centimeters -/
def max_volume : ℝ := (total_cubes : ℝ) * single_cube_volume

theorem solid_max_volume : max_volume = 540 := by sorry

end NUMINAMATH_CALUDE_solid_max_volume_l2386_238613
