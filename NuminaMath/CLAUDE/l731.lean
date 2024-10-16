import Mathlib

namespace NUMINAMATH_CALUDE_square_of_complex_l731_73125

theorem square_of_complex : (3 - Complex.I) ^ 2 = 8 - 6 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_l731_73125


namespace NUMINAMATH_CALUDE_no_possible_values_for_a_l731_73101

def M (a : ℝ) : Set ℝ := {1, 9, a}
def P (a : ℝ) : Set ℝ := {1, a, 2}

theorem no_possible_values_for_a :
  ∀ a : ℝ, (P a) ⊆ (M a) → False :=
sorry

end NUMINAMATH_CALUDE_no_possible_values_for_a_l731_73101


namespace NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l731_73188

/-- Given a line with equation 3x - 4y + 5 = 0, its symmetric line with respect to the x-axis has the equation 3x + 4y + 5 = 0 -/
theorem symmetric_line_wrt_x_axis :
  ∀ (x y : ℝ), 3 * x - 4 * y + 5 = 0 →
  ∃ (x' y' : ℝ), x' = x ∧ y' = -y ∧ 3 * x' + 4 * y' + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_wrt_x_axis_l731_73188


namespace NUMINAMATH_CALUDE_mary_age_proof_l731_73182

/-- Mary's current age -/
def mary_age : ℕ := 2

/-- Jay's current age -/
def jay_age : ℕ := mary_age + 7

theorem mary_age_proof :
  (∃ (j m : ℕ),
    j - 5 = (m - 5) + 7 ∧
    j + 5 = 2 * (m + 5) ∧
    m = mary_age) :=
by sorry

end NUMINAMATH_CALUDE_mary_age_proof_l731_73182


namespace NUMINAMATH_CALUDE_sequence_exists_l731_73160

theorem sequence_exists : ∃ (seq : Fin 2000 → ℝ), 
  (∀ i : Fin 1998, seq i + seq (i + 1) + seq (i + 2) < 0) ∧ 
  (Finset.sum Finset.univ seq > 0) := by
  sorry

end NUMINAMATH_CALUDE_sequence_exists_l731_73160


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l731_73106

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x^2 < 1) → ¬(-1 < x ∧ x < 1)) ↔ ((x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) := by
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l731_73106


namespace NUMINAMATH_CALUDE_other_solution_quadratic_l731_73171

theorem other_solution_quadratic (x : ℚ) :
  (72 * (3/8)^2 + 37 = -95 * (3/8) + 12) →
  (72 * x^2 + 37 = -95 * x + 12) →
  (x ≠ 3/8) →
  x = 5/8 := by
sorry

end NUMINAMATH_CALUDE_other_solution_quadratic_l731_73171


namespace NUMINAMATH_CALUDE_workers_count_l731_73170

/-- Given a group of workers who collectively contribute 300,000 and would contribute 350,000 if each gave 50 more, prove that there are 1000 workers. -/
theorem workers_count (total : ℕ) (extra_total : ℕ) (extra_per_worker : ℕ) : 
  total = 300000 →
  extra_total = 350000 →
  extra_per_worker = 50 →
  ∃ (num_workers : ℕ), num_workers * (total / num_workers + extra_per_worker) = extra_total ∧ 
                        num_workers = 1000 := by
  sorry

end NUMINAMATH_CALUDE_workers_count_l731_73170


namespace NUMINAMATH_CALUDE_abs_plus_square_zero_implies_sum_l731_73147

theorem abs_plus_square_zero_implies_sum (x y : ℝ) :
  |x + 3| + (2*y - 5)^2 = 0 → x + 2*y = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_plus_square_zero_implies_sum_l731_73147


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l731_73109

theorem modulus_of_complex_fraction (z : ℂ) : z = (1 + I) / (1 - I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l731_73109


namespace NUMINAMATH_CALUDE_roots_sum_equation_l731_73142

theorem roots_sum_equation (a b : ℝ) : 
  (a^2 - 4*a + 4 = 0) → 
  (b^2 - 4*b + 4 = 0) → 
  2*(a + b) = 8 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_equation_l731_73142


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l731_73197

theorem baseball_card_value_decrease (x : ℝ) :
  (1 - x / 100) * (1 - 10 / 100) = 1 - 28 / 100 →
  x = 20 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l731_73197


namespace NUMINAMATH_CALUDE_changsha_tourism_l731_73146

/-- The number of visitors (in millions) to Changsha during May Day holiday in 2021 -/
def visitors_2021 : ℝ := 2

/-- The number of visitors (in millions) to Changsha during May Day holiday in 2023 -/
def visitors_2023 : ℝ := 2.88

/-- The amount spent on Youlan Latte -/
def spent_youlan : ℝ := 216

/-- The amount spent on Shengsheng Oolong -/
def spent_oolong : ℝ := 96

/-- The price difference between Youlan Latte and Shengsheng Oolong -/
def price_difference : ℝ := 2

theorem changsha_tourism (r x : ℝ) : 
  ((1 + r)^2 = visitors_2023 / visitors_2021) ∧ 
  (spent_youlan / x = 2 * spent_oolong / (x - price_difference)) → 
  (r = 0.2 ∧ x = 18) := by sorry

end NUMINAMATH_CALUDE_changsha_tourism_l731_73146


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l731_73179

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 2*x + 1 = 0 ∧ a * y^2 - 2*y + 1 = 0) → 
  (a < 1 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l731_73179


namespace NUMINAMATH_CALUDE_ali_bookshelf_problem_l731_73137

theorem ali_bookshelf_problem (x : ℕ) : 
  (x / 2 : ℕ) + (x / 3 : ℕ) + 3 + 7 = x → (x / 2 : ℕ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_ali_bookshelf_problem_l731_73137


namespace NUMINAMATH_CALUDE_post_office_mailing_l731_73118

def total_cost : ℚ := 449/100
def letter_cost : ℚ := 37/100
def package_cost : ℚ := 88/100
def num_letters : ℕ := 5

theorem post_office_mailing :
  ∃ (num_packages : ℕ),
    letter_cost * num_letters + package_cost * num_packages = total_cost ∧
    num_letters - num_packages = 2 :=
by sorry

end NUMINAMATH_CALUDE_post_office_mailing_l731_73118


namespace NUMINAMATH_CALUDE_f_increasing_range_of_a_l731_73122

/-- The function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - a/2)*x + 2

/-- The theorem stating the range of values for a -/
theorem f_increasing_range_of_a :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (8/3) 4 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_range_of_a_l731_73122


namespace NUMINAMATH_CALUDE_real_axis_length_is_six_l731_73136

/-- The hyperbola C with equation x²/a² - y²/b² = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The line l with equation 4x + 3y - 20 = 0 -/
def line_l (x y : ℝ) : Prop := 4*x + 3*y - 20 = 0

/-- The line l passes through one focus of the hyperbola C -/
def passes_through_focus (C : Hyperbola) : Prop :=
  ∃ (x y : ℝ), line_l x y ∧ x^2 - C.a^2 = C.b^2

/-- The line l is parallel to one of the asymptotes of the hyperbola C -/
def parallel_to_asymptote (C : Hyperbola) : Prop :=
  C.b / C.a = 4 / 3

/-- The theorem stating that the length of the real axis of the hyperbola C is 6 -/
theorem real_axis_length_is_six (C : Hyperbola)
  (h1 : passes_through_focus C)
  (h2 : parallel_to_asymptote C) :
  2 * C.a = 6 := by sorry

end NUMINAMATH_CALUDE_real_axis_length_is_six_l731_73136


namespace NUMINAMATH_CALUDE_perpendicular_slope_l731_73104

theorem perpendicular_slope (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let original_slope := a / b
  let perpendicular_slope := -1 / original_slope
  (5 : ℝ) * x - (4 : ℝ) * y = (20 : ℝ) → perpendicular_slope = -(4 : ℝ) / (5 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l731_73104


namespace NUMINAMATH_CALUDE_event_ticket_revenue_l731_73192

theorem event_ticket_revenue :
  ∀ (full_price : ℚ) (full_count half_count : ℕ),
    full_count + half_count = 180 →
    full_price * full_count + (full_price / 2) * half_count = 2652 →
    full_price * full_count = 984 :=
by
  sorry

end NUMINAMATH_CALUDE_event_ticket_revenue_l731_73192


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l731_73183

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y : ℚ) :
  y = 3 * x + 7 ∧ 
  y = -4 * x + 1 ∧ 
  y = 2 * x + k →
  k = 43 / 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l731_73183


namespace NUMINAMATH_CALUDE_factor_difference_of_squares_l731_73162

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_difference_of_squares_l731_73162


namespace NUMINAMATH_CALUDE_b_over_a_range_l731_73161

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if the roots represent eccentricities of conic sections -/
def has_conic_eccentricities (eq : CubicEquation) : Prop :=
  ∃ (e₁ e₂ e₃ : ℝ), 
    e₁^3 + eq.a * e₁^2 + eq.b * e₁ + eq.c = 0 ∧
    e₂^3 + eq.a * e₂^2 + eq.b * e₂ + eq.c = 0 ∧
    e₃^3 + eq.a * e₃^2 + eq.b * e₃ + eq.c = 0 ∧
    (0 ≤ e₁ ∧ e₁ < 1) ∧  -- ellipse eccentricity
    (e₂ > 1) ∧           -- hyperbola eccentricity
    (e₃ = 1)             -- parabola eccentricity

/-- The main theorem stating the range of b/a -/
theorem b_over_a_range (eq : CubicEquation) 
  (h : has_conic_eccentricities eq) : 
  -2 < eq.b / eq.a ∧ eq.b / eq.a < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_b_over_a_range_l731_73161


namespace NUMINAMATH_CALUDE_water_fountain_problem_l731_73100

/-- The number of men needed to build a water fountain of a given length in a given number of days -/
def men_needed (length : ℝ) (days : ℝ) : ℝ :=
  sorry

theorem water_fountain_problem :
  let first_length : ℝ := 56
  let first_days : ℝ := 21
  let second_length : ℝ := 14
  let second_days : ℝ := 3
  let second_men : ℝ := 35

  (men_needed first_length first_days) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_water_fountain_problem_l731_73100


namespace NUMINAMATH_CALUDE_concatenated_number_500_not_divisible_by_9_l731_73127

def concatenated_number (n : ℕ) : ℕ := sorry

theorem concatenated_number_500_not_divisible_by_9 :
  ¬ (9 ∣ concatenated_number 500) := by sorry

end NUMINAMATH_CALUDE_concatenated_number_500_not_divisible_by_9_l731_73127


namespace NUMINAMATH_CALUDE_prob_coprime_with_2015_l731_73143

/-- The probability that gcd(n, 2015) = 1 for a randomly chosen n in [1, 2016] -/
theorem prob_coprime_with_2015 : 
  (Finset.filter (fun n => Nat.gcd n 2015 = 1) (Finset.range 2016)).card / 2016 = 1442 / 2016 := by
  sorry

end NUMINAMATH_CALUDE_prob_coprime_with_2015_l731_73143


namespace NUMINAMATH_CALUDE_evaluate_expression_l731_73167

theorem evaluate_expression : 3 + (-3)^2 = 12 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l731_73167


namespace NUMINAMATH_CALUDE_quadratic_sum_l731_73126

/-- A quadratic function g(x) = dx^2 + ex + f passing through (1, 3) and (2, 0) with vertex at (3, -3) -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := λ x => d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (QuadraticFunction d e f 1 = 3) →
  (QuadraticFunction d e f 2 = 0) →
  (∀ x, QuadraticFunction d e f x ≥ QuadraticFunction d e f 3) →
  (QuadraticFunction d e f 3 = -3) →
  d + e + 2 * f = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l731_73126


namespace NUMINAMATH_CALUDE_train_length_calculation_l731_73148

/-- Represents a train with its length and the time it takes to cross two platforms -/
structure Train where
  length : ℝ
  time_platform1 : ℝ
  time_platform2 : ℝ

/-- The length of the first platform in meters -/
def platform1_length : ℝ := 120

/-- The length of the second platform in meters -/
def platform2_length : ℝ := 250

/-- Theorem stating that a train crossing two platforms of given lengths in specific times has a specific length -/
theorem train_length_calculation (t : Train) 
  (h1 : t.time_platform1 = 15) 
  (h2 : t.time_platform2 = 20) : 
  t.length = 270 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l731_73148


namespace NUMINAMATH_CALUDE_min_value_circle_l731_73103

theorem min_value_circle (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (m : ℝ), (∀ (a b : ℝ), a^2 + b^2 - 4*a + 1 = 0 → x^2 + y^2 ≤ a^2 + b^2) ∧ 
  m = x^2 + y^2 ∧ m = 7 - 4*Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_circle_l731_73103


namespace NUMINAMATH_CALUDE_unique_f_3_l731_73116

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 2 = 3 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- The main theorem -/
theorem unique_f_3 (f : ℝ → ℝ) (hf : special_function f) : f 3 = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_f_3_l731_73116


namespace NUMINAMATH_CALUDE_p_plus_q_equals_30_l731_73190

theorem p_plus_q_equals_30 (P Q : ℝ) :
  (∀ x : ℝ, x ≠ 3 → P / (x - 3) + Q * (x + 2) = (-5 * x^2 + 20 * x + 35) / (x - 3)) →
  P + Q = 30 := by
sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_30_l731_73190


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l731_73180

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 15)
  (product_sum_condition : a * b + a * c + b * c = 40) :
  a^3 + b^3 + c^3 - 3*a*b*c = 1575 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l731_73180


namespace NUMINAMATH_CALUDE_average_mark_calculation_l731_73173

theorem average_mark_calculation (students_class1 students_class2 : ℕ) 
  (avg_class2 avg_total : ℚ) : 
  students_class1 = 20 →
  students_class2 = 50 →
  avg_class2 = 60 →
  avg_total = 54.285714285714285 →
  (students_class1 * (avg_total * (students_class1 + students_class2) - students_class2 * avg_class2)) / 
   (students_class1 * (students_class1 + students_class2)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_calculation_l731_73173


namespace NUMINAMATH_CALUDE_modular_inverse_31_mod_37_l731_73181

theorem modular_inverse_31_mod_37 :
  ∃ x : ℕ, x ≤ 36 ∧ (31 * x) % 37 = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_modular_inverse_31_mod_37_l731_73181


namespace NUMINAMATH_CALUDE_prize_probabilities_l731_73117

/-- Represents the outcome of drawing a ball from a box -/
inductive BallColor
| Red
| White

/-- Represents a box with red and white balls -/
structure Box where
  red : Nat
  white : Nat

/-- Probability of drawing a red ball from a box -/
def probRed (box : Box) : Rat :=
  box.red / (box.red + box.white)

/-- Probability of winning first prize in one draw -/
def probFirstPrize (boxA boxB : Box) : Rat :=
  probRed boxA * probRed boxB

/-- Probability of winning second prize in one draw -/
def probSecondPrize (boxA boxB : Box) : Rat :=
  probRed boxA * (1 - probRed boxB) + (1 - probRed boxA) * probRed boxB

/-- Probability of winning a prize in one draw -/
def probWinPrize (boxA boxB : Box) : Rat :=
  probFirstPrize boxA boxB + probSecondPrize boxA boxB

/-- Expected number of first prizes in n draws -/
def expectedFirstPrizes (boxA boxB : Box) (n : Nat) : Rat :=
  n * probFirstPrize boxA boxB

theorem prize_probabilities (boxA boxB : Box) :
  boxA.red = 4 ∧ boxA.white = 6 ∧ boxB.red = 5 ∧ boxB.white = 5 →
  probWinPrize boxA boxB = 7/10 ∧ expectedFirstPrizes boxA boxB 3 = 3/5 := by
  sorry


end NUMINAMATH_CALUDE_prize_probabilities_l731_73117


namespace NUMINAMATH_CALUDE_largest_angle_in_isosceles_triangle_l731_73178

-- Define an isosceles triangle with one angle of 50°
def IsoscelesTriangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a = b ∧ a = 50

-- Theorem statement
theorem largest_angle_in_isosceles_triangle 
  {a b c : ℝ} (h : IsoscelesTriangle a b c) : 
  max a (max b c) = 80 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_isosceles_triangle_l731_73178


namespace NUMINAMATH_CALUDE_zoo_trip_vans_l731_73107

def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

theorem zoo_trip_vans : vans_needed 4 2 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_vans_l731_73107


namespace NUMINAMATH_CALUDE_acute_triangle_probability_condition_l731_73128

/-- The probability of forming an acute triangle from three random vertices of a regular n-gon --/
def acuteTriangleProbability (n : ℕ) : ℚ :=
  if n % 2 = 0
  then (3 * (n / 2 - 2)) / (2 * (n - 1))
  else (3 * ((n - 1) / 2 - 1)) / (2 * (n - 1))

/-- Theorem stating that the probability of forming an acute triangle is 93/125 
    if and only if n is 376 or 127 --/
theorem acute_triangle_probability_condition (n : ℕ) :
  acuteTriangleProbability n = 93 / 125 ↔ n = 376 ∨ n = 127 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_probability_condition_l731_73128


namespace NUMINAMATH_CALUDE_max_value_inequality_l731_73159

theorem max_value_inequality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : |1/a| + |1/b| + |1/c| ≤ 3) : 
  (a^2 + 4*(b^2 + c^2)) * (b^2 + 4*(a^2 + c^2)) * (c^2 + 4*(a^2 + b^2)) ≥ 729 ∧ 
  ∀ m > 729, ∃ a' b' c' : ℝ, a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧ 
    |1/a'| + |1/b'| + |1/c'| ≤ 3 ∧
    (a'^2 + 4*(b'^2 + c'^2)) * (b'^2 + 4*(a'^2 + c'^2)) * (c'^2 + 4*(a'^2 + b'^2)) < m :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l731_73159


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l731_73175

/-- Represents a trapezoid with given side lengths -/
structure Trapezoid where
  a : ℝ  -- Length of one parallel side
  b : ℝ  -- Length of the other parallel side
  c : ℝ  -- Length of one non-parallel side
  d : ℝ  -- Length of the other non-parallel side

/-- Calculates the area of a trapezoid given its side lengths -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  -- We don't implement the actual calculation here
  sorry

/-- Theorem: The area of the specific trapezoid is 450 -/
theorem specific_trapezoid_area :
  trapezoidArea { a := 16, b := 44, c := 17, d := 25 } = 450 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l731_73175


namespace NUMINAMATH_CALUDE_plate_tower_problem_l731_73189

theorem plate_tower_problem (initial_plates : ℕ) (first_addition : ℕ) (common_difference : ℕ) (total_plates : ℕ) :
  initial_plates = 27 →
  first_addition = 12 →
  common_difference = 3 →
  total_plates = 123 →
  ∃ (n : ℕ) (last_addition : ℕ),
    n = 4 ∧
    last_addition = 21 ∧
    total_plates = initial_plates + n * (2 * first_addition + (n - 1) * common_difference) / 2 :=
by sorry

end NUMINAMATH_CALUDE_plate_tower_problem_l731_73189


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l731_73158

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, 4), and x-intercepts at (1, 0) and (5, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := 4
  intercept1_x : ℝ := 1
  intercept2_x : ℝ := 5

/-- The parabola satisfies its vertex condition -/
axiom vertex_condition (p : Parabola) : p.vertex_y = p.a * p.vertex_x^2 + p.b * p.vertex_x + p.c

/-- The parabola satisfies its first x-intercept condition -/
axiom intercept1_condition (p : Parabola) : 0 = p.a * p.intercept1_x^2 + p.b * p.intercept1_x + p.c

/-- The parabola satisfies its second x-intercept condition -/
axiom intercept2_condition (p : Parabola) : 0 = p.a * p.intercept2_x^2 + p.b * p.intercept2_x + p.c

/-- The sum of coefficients a, b, and c is zero for a parabola satisfying the given conditions -/
theorem sum_of_coefficients_zero (p : Parabola) : p.a + p.b + p.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l731_73158


namespace NUMINAMATH_CALUDE_equation_solution_range_l731_73165

theorem equation_solution_range (x m : ℝ) : 9^x + 4 * 3^x - m = 0 → m > 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_range_l731_73165


namespace NUMINAMATH_CALUDE_people_in_hall_l731_73131

theorem people_in_hall (total_chairs : ℕ) (seated_people : ℕ) (empty_chairs : ℕ) :
  seated_people = (5 : ℕ) * total_chairs / 8 →
  empty_chairs = 8 →
  seated_people = total_chairs - empty_chairs →
  seated_people * 2 = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_hall_l731_73131


namespace NUMINAMATH_CALUDE_nine_chapters_problem_l731_73169

theorem nine_chapters_problem (x y : ℕ) :
  y = 2*x + 9 ∧ y = 3*(x - 2) ↔ 
  (∃ (filled_cars : ℕ), 
    x = filled_cars + 2 ∧ 
    y = 3 * filled_cars) :=
sorry

end NUMINAMATH_CALUDE_nine_chapters_problem_l731_73169


namespace NUMINAMATH_CALUDE_expression_value_at_eight_l731_73185

theorem expression_value_at_eight :
  let x : ℝ := 8
  (x^6 - 64*x^3 + 1024) / (x^3 - 16) = 480 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_eight_l731_73185


namespace NUMINAMATH_CALUDE_stating_time_for_one_click_approx_10_seconds_l731_73199

/-- Represents the length of a rail in feet -/
def rail_length : ℝ := 15

/-- Represents the number of feet in a mile -/
def feet_per_mile : ℝ := 5280

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℝ := 60

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℝ := 60

/-- 
Theorem stating that the time taken to hear one click (passing over one rail joint) 
is approximately 10 seconds for a train traveling at any speed.
-/
theorem time_for_one_click_approx_10_seconds (train_speed : ℝ) : 
  train_speed > 0 → 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
    |((rail_length * minutes_per_hour) / (train_speed * feet_per_mile)) * seconds_per_minute - 10| < ε :=
sorry

end NUMINAMATH_CALUDE_stating_time_for_one_click_approx_10_seconds_l731_73199


namespace NUMINAMATH_CALUDE_lowest_price_option2_l731_73129

def initial_amount : ℝ := 12000

def option1_price : ℝ := initial_amount * (1 - 0.15) * (1 - 0.10) * (1 - 0.05)

def option2_price : ℝ := initial_amount * (1 - 0.25) * (1 - 0.05)

def option3_price : ℝ := initial_amount * (1 - 0.20) - 500

theorem lowest_price_option2 :
  option2_price < option1_price ∧ option2_price < option3_price :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_option2_l731_73129


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l731_73120

/-- 
Given a rectangle with length l and width w, and points P and Q as midpoints of two adjacent sides,
prove that the shaded area is 7/8 of the total area when the triangle formed by P, Q, and the 
vertex at the intersection of uncut sides is unshaded.
-/
theorem shaded_area_fraction (l w : ℝ) (h1 : l > 0) (h2 : w > 0) : 
  let total_area := l * w
  let unshaded_triangle_area := (l / 2) * (w / 2) / 2
  let shaded_area := total_area - unshaded_triangle_area
  (shaded_area / total_area) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l731_73120


namespace NUMINAMATH_CALUDE_proposition_implication_l731_73153

theorem proposition_implication (P : ℕ → Prop) :
  (∀ k : ℕ, k ≥ 1 → (P k → P (k + 1))) →
  (¬ P 10) →
  (¬ P 9) := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l731_73153


namespace NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_l731_73123

/-- A quadrilateral with sides a, b, c, d, diagonals m, n, and distance t between midpoints of diagonals -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  t : ℝ

/-- The sum of squares of sides equals the sum of squares of diagonals plus four times the square of the distance between midpoints of diagonals -/
theorem quadrilateral_sum_of_squares (q : Quadrilateral) :
  q.a^2 + q.b^2 + q.c^2 + q.d^2 = q.m^2 + q.n^2 + 4 * q.t^2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_sum_of_squares_l731_73123


namespace NUMINAMATH_CALUDE_infinitely_many_close_fractions_l731_73164

theorem infinitely_many_close_fractions (x : ℝ) (hx_pos : x > 0) (hx_irrational : ¬ ∃ (a b : ℤ), x = a / b) :
  ∀ n : ℕ, ∃ p q : ℤ, q > n ∧ q > 0 ∧ |x - (p : ℝ) / q| ≤ 1 / q^2 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_close_fractions_l731_73164


namespace NUMINAMATH_CALUDE_plot_length_l731_73166

/-- Proves that the length of a rectangular plot is 55 meters given the specified conditions -/
theorem plot_length (breadth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  (breadth + 10 = breadth + 10) →  -- Length is 10 more than breadth
  (cost_per_meter = 26.5) →        -- Cost per meter is 26.50 rupees
  (total_cost = 5300) →            -- Total cost is 5300 rupees
  (4 * breadth + 20) * cost_per_meter = total_cost →  -- Perimeter calculation
  (breadth + 10 = 55) :=            -- Length of the plot is 55 meters
by sorry

end NUMINAMATH_CALUDE_plot_length_l731_73166


namespace NUMINAMATH_CALUDE_range_of_a_l731_73198

def p (x : ℝ) : Prop := |4*x - 3| ≤ 1

def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a :
  (∀ x a : ℝ, ¬(p x) → ¬(q x a)) ∧
  (∃ x a : ℝ, ¬(q x a) ∧ p x) →
  ∀ a : ℝ, (0 ≤ a ∧ a ≤ 1/2) ↔ (∀ x : ℝ, p x → q x a) ∧ (∃ x : ℝ, q x a ∧ ¬(p x)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l731_73198


namespace NUMINAMATH_CALUDE_num_technicians_correct_l731_73176

/-- Represents the number of technicians in a workshop. -/
def num_technicians : ℕ := 7

/-- Represents the total number of workers in the workshop. -/
def total_workers : ℕ := 49

/-- Represents the average salary of all workers in the workshop. -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians in the workshop. -/
def avg_salary_technicians : ℕ := 20000

/-- Represents the average salary of non-technician workers in the workshop. -/
def avg_salary_rest : ℕ := 6000

/-- Theorem stating that the number of technicians satisfies the given conditions. -/
theorem num_technicians_correct :
  num_technicians * avg_salary_technicians +
  (total_workers - num_technicians) * avg_salary_rest =
  total_workers * avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_num_technicians_correct_l731_73176


namespace NUMINAMATH_CALUDE_descending_order_proof_l731_73172

def original_numbers : List ℝ := [1.64, 2.1, 0.09, 1.2]
def sorted_numbers : List ℝ := [2.1, 1.64, 1.2, 0.09]

theorem descending_order_proof :
  (sorted_numbers.zip (sorted_numbers.tail!)).all (fun (a, b) => a ≥ b) ∧
  sorted_numbers.toFinset = original_numbers.toFinset :=
by sorry

end NUMINAMATH_CALUDE_descending_order_proof_l731_73172


namespace NUMINAMATH_CALUDE_min_value_theorem_l731_73151

theorem min_value_theorem (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 36*c^2 = 4) :
  ∃ (m : ℝ), m = -2 * Real.sqrt 14 ∧ ∀ (x y z : ℝ), 9*x^2 + 4*y^2 + 36*z^2 = 4 → 3*x + 6*y + 12*z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l731_73151


namespace NUMINAMATH_CALUDE_chocolate_theorem_l731_73141

-- Define the parameters of the problem
def chocolate_cost : ℕ := 1
def wrappers_per_exchange : ℕ := 3
def initial_money : ℕ := 15

-- Define a function to calculate the maximum number of chocolates
def max_chocolates (cost : ℕ) (exchange_rate : ℕ) (money : ℕ) : ℕ :=
  -- Implementation details are omitted
  sorry

-- State the theorem
theorem chocolate_theorem :
  max_chocolates chocolate_cost wrappers_per_exchange initial_money = 22 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l731_73141


namespace NUMINAMATH_CALUDE_election_winner_votes_l731_73112

theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℚ) * (58 / 100) - (total_votes : ℚ) * (42 / 100) = 288 →
  (total_votes : ℚ) * (58 / 100) = 1044 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l731_73112


namespace NUMINAMATH_CALUDE_cassidy_grounding_l731_73156

/-- Calculates the number of extra days grounded per grade below B -/
def extraDaysPerGrade (totalDays : ℕ) (baseDays : ℕ) (gradesBelowB : ℕ) : ℕ :=
  if gradesBelowB = 0 then 0 else (totalDays - baseDays) / gradesBelowB

theorem cassidy_grounding (totalDays : ℕ) (baseDays : ℕ) (gradesBelowB : ℕ) 
  (h1 : totalDays = 26)
  (h2 : baseDays = 14)
  (h3 : gradesBelowB = 4) :
  extraDaysPerGrade totalDays baseDays gradesBelowB = 3 := by
  sorry

#eval extraDaysPerGrade 26 14 4

end NUMINAMATH_CALUDE_cassidy_grounding_l731_73156


namespace NUMINAMATH_CALUDE_power_sum_value_l731_73157

theorem power_sum_value (a : ℝ) (x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(x+y) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_value_l731_73157


namespace NUMINAMATH_CALUDE_diamond_eight_five_l731_73138

def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem diamond_eight_five : diamond 8 5 = 160 := by sorry

end NUMINAMATH_CALUDE_diamond_eight_five_l731_73138


namespace NUMINAMATH_CALUDE_tomato_seeds_proof_l731_73115

/-- The number of tomato seeds planted by Mike and Ted -/
def total_seeds (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ) : ℕ :=
  mike_morning + mike_afternoon + ted_morning + ted_afternoon

theorem tomato_seeds_proof :
  ∀ (mike_morning mike_afternoon ted_morning ted_afternoon : ℕ),
    mike_morning = 50 →
    ted_morning = 2 * mike_morning →
    mike_afternoon = 60 →
    ted_afternoon = mike_afternoon - 20 →
    total_seeds mike_morning mike_afternoon ted_morning ted_afternoon = 250 := by
  sorry

end NUMINAMATH_CALUDE_tomato_seeds_proof_l731_73115


namespace NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l731_73119

theorem three_digit_difference_divisible_by_nine :
  ∀ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 →
  0 ≤ b ∧ b ≤ 9 →
  0 ≤ c ∧ c ≤ 9 →
  ∃ (k : ℤ), (100 * a + 10 * b + c) - (a + b + c) = 9 * k :=
by sorry

end NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l731_73119


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l731_73177

def A : Set ℝ := {-1, 1, 3, 5}
def B : Set ℝ := {x | x^2 - 4 < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l731_73177


namespace NUMINAMATH_CALUDE_percentage_spent_l731_73168

theorem percentage_spent (initial_amount remaining_amount : ℝ) 
  (h1 : initial_amount = 1200)
  (h2 : remaining_amount = 840) :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_l731_73168


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l731_73174

theorem smallest_positive_integer_with_given_remainders :
  ∃ n : ℕ, n > 0 ∧
    n % 3 = 1 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    ∀ m : ℕ, m > 0 →
      m % 3 = 1 →
      m % 4 = 2 →
      m % 5 = 3 →
      n ≤ m :=
by
  use 58
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l731_73174


namespace NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l731_73132

theorem cos_six_arccos_one_fourth : 
  Real.cos (6 * Real.arccos (1/4)) = -7/128 := by
  sorry

end NUMINAMATH_CALUDE_cos_six_arccos_one_fourth_l731_73132


namespace NUMINAMATH_CALUDE_min_distance_to_line_l731_73145

/-- The minimum distance from the origin (0,0) to the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : 
  let line := {p : ℝ × ℝ | p.1 + p.2 = 4}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ line → Real.sqrt (p.1^2 + p.2^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l731_73145


namespace NUMINAMATH_CALUDE_sample_size_correct_l731_73154

/-- The sample size that satisfies the given conditions -/
def sample_size : ℕ := 6

/-- The total population size -/
def total_population : ℕ := 36

/-- Theorem stating that the sample size satisfies all conditions -/
theorem sample_size_correct : 
  (sample_size ∣ total_population) ∧ 
  (6 ∣ sample_size) ∧
  (∃ k : ℕ, 35 = k * (sample_size + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sample_size_correct_l731_73154


namespace NUMINAMATH_CALUDE_birds_to_africa_l731_73111

/-- The number of bird families that flew away to Africa -/
def families_to_africa : ℕ := 118 - 80

/-- The number of bird families that flew away to Asia -/
def families_to_asia : ℕ := 80

/-- The total number of bird families that flew away for the winter -/
def total_families_away : ℕ := 118

/-- The number of bird families living near the mountain (not used in the proof) -/
def families_near_mountain : ℕ := 18

theorem birds_to_africa :
  families_to_africa = 38 ∧
  families_to_africa + families_to_asia = total_families_away :=
sorry

end NUMINAMATH_CALUDE_birds_to_africa_l731_73111


namespace NUMINAMATH_CALUDE_book_pages_problem_l731_73113

theorem book_pages_problem (x : ℕ) (h1 : x > 0) (h2 : x + (x + 1) = 125) : x + 1 = 63 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_problem_l731_73113


namespace NUMINAMATH_CALUDE_max_percentage_offering_either_or_both_l731_73195

-- Define the percentage of companies offering wireless internet
def wireless_internet_percentage : ℚ := 20 / 100

-- Define the percentage of companies offering free snacks
def free_snacks_percentage : ℚ := 70 / 100

-- Theorem statement
theorem max_percentage_offering_either_or_both :
  ∃ (max_percentage : ℚ),
    max_percentage = wireless_internet_percentage + free_snacks_percentage ∧
    max_percentage ≤ 1 ∧
    ∀ (actual_percentage : ℚ),
      actual_percentage ≤ max_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_percentage_offering_either_or_both_l731_73195


namespace NUMINAMATH_CALUDE_albert_run_distance_l731_73144

/-- Calculates the total distance run on a circular track -/
def totalDistance (trackLength : ℕ) (lapsRun : ℕ) (additionalLaps : ℕ) : ℕ :=
  trackLength * (lapsRun + additionalLaps)

/-- Proves that running 11 laps on a 9-meter track results in 99 meters total distance -/
theorem albert_run_distance :
  totalDistance 9 6 5 = 99 := by
  sorry

end NUMINAMATH_CALUDE_albert_run_distance_l731_73144


namespace NUMINAMATH_CALUDE_unique_number_with_digit_sum_14_l731_73133

/-- Converts a decimal number to its octal representation -/
def toOctal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits of a natural number in base 10 -/
def sumDigits (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (acc : ℕ) :=
    if m = 0 then acc
    else aux (m / 10) (acc + m % 10)
  aux n 0

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem unique_number_with_digit_sum_14 :
  ∃! n : ℕ,
    n > 0 ∧
    n < 1000 ∧
    (toOctal n).length = 3 ∧
    sumDigits n = 14 ∧
    sumList (toOctal n) = 14 ∧
    n = 455 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_digit_sum_14_l731_73133


namespace NUMINAMATH_CALUDE_sin_cos_difference_equality_l731_73134

theorem sin_cos_difference_equality : 
  Real.sin (7 * π / 180) * Real.cos (37 * π / 180) - 
  Real.sin (83 * π / 180) * Real.sin (37 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equality_l731_73134


namespace NUMINAMATH_CALUDE_work_completion_time_l731_73155

/-- The number of days it takes for a group to complete a work -/
def days_to_complete (women : ℕ) (children : ℕ) : ℚ :=
  1 / ((women / 50 : ℚ) + (children / 100 : ℚ))

/-- The theorem stating that 5 women and 10 children working together will complete the work in 5 days -/
theorem work_completion_time :
  days_to_complete 5 10 = 5 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l731_73155


namespace NUMINAMATH_CALUDE_probability_ascending_rolls_eq_five_fiftyfour_l731_73102

/-- A standard die has faces labeled from 1 to 6 -/
def standardDie : Finset Nat := Finset.range 6

/-- The number of times the die is rolled -/
def numRolls : Nat := 3

/-- The probability of rolling three dice and getting three distinct numbers in ascending order -/
def probabilityAscendingRolls : Rat :=
  (Nat.choose 6 3 : Rat) / (6 ^ numRolls)

theorem probability_ascending_rolls_eq_five_fiftyfour : 
  probabilityAscendingRolls = 5 / 54 := by
  sorry

end NUMINAMATH_CALUDE_probability_ascending_rolls_eq_five_fiftyfour_l731_73102


namespace NUMINAMATH_CALUDE_trailer_homes_proof_l731_73194

/-- Represents the number of new trailer homes added -/
def new_homes : ℕ := 17

/-- Represents the initial number of trailer homes -/
def initial_homes : ℕ := 25

/-- Represents the initial average age of trailer homes in years -/
def initial_avg_age : ℚ := 15

/-- Represents the current average age of all trailer homes in years -/
def current_avg_age : ℚ := 12

/-- Represents the time elapsed since new homes were added, in years -/
def years_passed : ℕ := 3

theorem trailer_homes_proof :
  (initial_homes * (initial_avg_age + years_passed) + new_homes * years_passed) / 
  (initial_homes + new_homes) = current_avg_age :=
sorry

end NUMINAMATH_CALUDE_trailer_homes_proof_l731_73194


namespace NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l731_73114

theorem cubic_equation_integer_solutions :
  (∀ k : ℤ, ∃! x : ℤ, x^3 - 24*x + k = 0) ∧
  (∃! x : ℤ, x^3 + 24*x - 2016 = 0 ∧ x = 12) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_integer_solutions_l731_73114


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l731_73130

/-- Given a parabola y = (1/m)x^2 where m ≠ 0, its focus has coordinates (0, m/4) -/
theorem parabola_focus_coordinates (m : ℝ) (hm : m ≠ 0) :
  let parabola := {(x, y) : ℝ × ℝ | y = (1/m) * x^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (0, m/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l731_73130


namespace NUMINAMATH_CALUDE_fudge_price_per_pound_l731_73187

-- Define the given quantities
def total_revenue : ℚ := 212
def fudge_pounds : ℚ := 20
def truffle_dozens : ℚ := 5
def truffle_price : ℚ := 3/2  -- $1.50 as a rational number
def pretzel_dozens : ℚ := 3
def pretzel_price : ℚ := 2

-- Define the theorem
theorem fudge_price_per_pound :
  (total_revenue - (truffle_dozens * 12 * truffle_price + pretzel_dozens * 12 * pretzel_price)) / fudge_pounds = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_fudge_price_per_pound_l731_73187


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l731_73150

/-- Calculates the percentage profit of a retailer given the wholesale price, retail price, and discount percentage. -/
theorem retailer_profit_percentage 
  (wholesale_price retail_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : wholesale_price = 90) 
  (h2 : retail_price = 120) 
  (h3 : discount_percentage = 0.1) :
  let selling_price := retail_price * (1 - discount_percentage)
  let profit := selling_price - wholesale_price
  let profit_percentage := (profit / wholesale_price) * 100
  profit_percentage = 20 := by
  sorry

end NUMINAMATH_CALUDE_retailer_profit_percentage_l731_73150


namespace NUMINAMATH_CALUDE_dual_colored_cubes_count_l731_73121

/-- Represents a cube painted with two colors on opposite face pairs --/
structure PaintedCube where
  size : ℕ
  color1 : String
  color2 : String

/-- Represents a smaller cube after cutting the original cube --/
structure SmallCube where
  hasColor1 : Bool
  hasColor2 : Bool

/-- Cuts a painted cube into smaller cubes --/
def cutCube (c : PaintedCube) : List SmallCube :=
  sorry

/-- Counts the number of small cubes with both colors --/
def countDualColorCubes (cubes : List SmallCube) : ℕ :=
  sorry

/-- Theorem stating that a cube painted as described and cut into 64 pieces will have 16 dual-colored cubes --/
theorem dual_colored_cubes_count 
  (c : PaintedCube) 
  (h1 : c.size = 4) 
  (h2 : c.color1 ≠ c.color2) : 
  countDualColorCubes (cutCube c) = 16 :=
sorry

end NUMINAMATH_CALUDE_dual_colored_cubes_count_l731_73121


namespace NUMINAMATH_CALUDE_max_trigonometric_product_l731_73135

theorem max_trigonometric_product (x y z : ℝ) : 
  (Real.sin (2 * x) + Real.sin (3 * y) + Real.sin (4 * z)) * 
  (Real.cos (2 * x) + Real.cos (3 * y) + Real.cos (4 * z)) ≤ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_trigonometric_product_l731_73135


namespace NUMINAMATH_CALUDE_seven_c_plus_seven_d_equals_five_l731_73140

-- Define the function h
def h (x : ℝ) : ℝ := 7 * x - 6

-- Define the function f
def f (c d x : ℝ) : ℝ := c * x + d

-- State the theorem
theorem seven_c_plus_seven_d_equals_five 
  (c d : ℝ) 
  (h_def : ∀ x, h x = 7 * x - 6)
  (h_inverse : ∀ x, h x = f c d⁻¹ x - 2)
  (f_inverse : ∀ x, f c d (f c d⁻¹ x) = x) :
  7 * c + 7 * d = 5 := by
sorry

end NUMINAMATH_CALUDE_seven_c_plus_seven_d_equals_five_l731_73140


namespace NUMINAMATH_CALUDE_resort_tips_fraction_l731_73139

theorem resort_tips_fraction (total_months : ℕ) (special_month_factor : ℕ) 
  (h1 : total_months = 7) 
  (h2 : special_month_factor = 4) : 
  (special_month_factor : ℚ) / ((total_months - 1 : ℕ) + special_month_factor : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_resort_tips_fraction_l731_73139


namespace NUMINAMATH_CALUDE_negation_of_proposition_l731_73186

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l731_73186


namespace NUMINAMATH_CALUDE_comparison_and_estimation_l731_73193

theorem comparison_and_estimation : 
  (2 * Real.sqrt 3 < 4) ∧ 
  (4 < Real.sqrt 17) ∧ 
  (Real.sqrt 17 < 5) := by sorry

end NUMINAMATH_CALUDE_comparison_and_estimation_l731_73193


namespace NUMINAMATH_CALUDE_min_perimeter_two_isosceles_triangles_l731_73152

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  side : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.side + t.base

/-- The area of an isosceles triangle -/
def area (t : IsoscelesTriangle) : ℚ :=
  (t.base : ℚ) * (((t.side : ℚ) ^ 2 - ((t.base : ℚ) / 2) ^ 2).sqrt) / 4

theorem min_perimeter_two_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t2.base = 4 * t1.base ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s2.base = 4 * s1.base →
      perimeter t1 ≤ perimeter s1 ∧
      perimeter t1 = 524 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_two_isosceles_triangles_l731_73152


namespace NUMINAMATH_CALUDE_frisbee_deck_difference_l731_73196

/-- Represents the number of items Bella has -/
structure BellasItems where
  marbles : ℕ
  frisbees : ℕ
  deckCards : ℕ

/-- The conditions of the problem -/
def problemConditions (items : BellasItems) : Prop :=
  items.marbles = 2 * items.frisbees ∧
  items.marbles = 60 ∧
  (items.marbles + 2/5 * items.marbles + 
   items.frisbees + 2/5 * items.frisbees + 
   items.deckCards + 2/5 * items.deckCards) = 140

/-- The theorem to prove -/
theorem frisbee_deck_difference (items : BellasItems) 
  (h : problemConditions items) : 
  items.frisbees - items.deckCards = 20 := by
  sorry


end NUMINAMATH_CALUDE_frisbee_deck_difference_l731_73196


namespace NUMINAMATH_CALUDE_calculate_divisor_l731_73124

/-- Given a dividend, quotient, and remainder, calculate the divisor -/
theorem calculate_divisor (dividend : ℝ) (quotient : ℝ) (remainder : ℝ) :
  dividend = 63584 ∧ quotient = 127.8 ∧ remainder = 45.5 →
  ∃ divisor : ℝ, divisor = 497.1 ∧ dividend = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_calculate_divisor_l731_73124


namespace NUMINAMATH_CALUDE_ants_in_field_approx_50_million_l731_73191

/-- Represents the dimensions of a rectangular field in feet -/
structure FieldDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a field in square inches -/
def fieldAreaInSquareInches (d : FieldDimensions) : ℝ :=
  d.width * d.length * 144  -- 144 = 12^2, converting square feet to square inches

/-- Calculates the total number of ants in a field -/
def totalAnts (d : FieldDimensions) (antsPerSquareInch : ℝ) : ℝ :=
  fieldAreaInSquareInches d * antsPerSquareInch

/-- Theorem stating that the number of ants in the given field is approximately 50 million -/
theorem ants_in_field_approx_50_million :
  let d : FieldDimensions := { width := 300, length := 400 }
  let antsPerSquareInch : ℝ := 3
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1000000 ∧ 
    abs (totalAnts d antsPerSquareInch - 50000000) < ε := by
  sorry

end NUMINAMATH_CALUDE_ants_in_field_approx_50_million_l731_73191


namespace NUMINAMATH_CALUDE_prime_divisor_implies_equal_l731_73163

theorem prime_divisor_implies_equal (m n : ℕ) : 
  Prime (m + n + 1) → 
  (m + n + 1) ∣ (2 * (m^2 + n^2) - 1) → 
  m = n :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_implies_equal_l731_73163


namespace NUMINAMATH_CALUDE_wall_building_time_l731_73149

/-- Given that 8 persons can build a 140 m long wall in 42 days, 
    prove that 30 persons can complete a 100 m long wall in 8 days -/
theorem wall_building_time 
  (persons_initial : ℕ) 
  (length_initial : ℕ) 
  (days_initial : ℕ) 
  (persons_new : ℕ) 
  (length_new : ℕ) 
  (h1 : persons_initial = 8) 
  (h2 : length_initial = 140) 
  (h3 : days_initial = 42) 
  (h4 : persons_new = 30) 
  (h5 : length_new = 100) : 
  (persons_initial * days_initial * length_new) / (persons_new * length_initial) = 8 := by
  sorry

end NUMINAMATH_CALUDE_wall_building_time_l731_73149


namespace NUMINAMATH_CALUDE_interview_score_calculation_l731_73184

/-- Calculate the interview score based on individual scores and their proportions -/
theorem interview_score_calculation 
  (basic_knowledge : ℝ) 
  (communication_skills : ℝ) 
  (work_attitude : ℝ) 
  (basic_knowledge_proportion : ℝ) 
  (communication_skills_proportion : ℝ) 
  (work_attitude_proportion : ℝ) 
  (h1 : basic_knowledge = 92) 
  (h2 : communication_skills = 87) 
  (h3 : work_attitude = 94) 
  (h4 : basic_knowledge_proportion = 0.2) 
  (h5 : communication_skills_proportion = 0.3) 
  (h6 : work_attitude_proportion = 0.5) :
  basic_knowledge * basic_knowledge_proportion + 
  communication_skills * communication_skills_proportion + 
  work_attitude * work_attitude_proportion = 91.5 := by
sorry

end NUMINAMATH_CALUDE_interview_score_calculation_l731_73184


namespace NUMINAMATH_CALUDE_cos_103pi_4_l731_73105

theorem cos_103pi_4 : Real.cos (103 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_103pi_4_l731_73105


namespace NUMINAMATH_CALUDE_original_number_proof_l731_73108

theorem original_number_proof : ∃ x : ℝ, x * 0.74 = 1.9832 ∧ x = 2.68 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l731_73108


namespace NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l731_73110

/-- Calculate the number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- The number of trailing zeroes in 500! + 200! is 124 -/
theorem trailing_zeroes_sum_factorials :
  max (trailingZeroes 500) (trailingZeroes 200) = 124 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l731_73110
