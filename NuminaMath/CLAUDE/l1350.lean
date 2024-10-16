import Mathlib

namespace NUMINAMATH_CALUDE_triangle_problem_l1350_135016

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  a^2 + c^2 - b^2 = a * c →
  a = 8 * Real.sqrt 3 →
  Real.cos A = 3 / 5 →
  -- Conclusions
  B = π / 3 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1350_135016


namespace NUMINAMATH_CALUDE_vector_magnitude_difference_l1350_135054

/-- Given two non-zero vectors in ℝ², if their sum is (-3, 6) and their difference is (-3, 2),
    then the difference of their squared magnitudes is 21. -/
theorem vector_magnitude_difference (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) 
    (hsum : a.1 + b.1 = -3 ∧ a.2 + b.2 = 6) (hdiff : a.1 - b.1 = -3 ∧ a.2 - b.2 = 2) :
    a.1^2 + a.2^2 - (b.1^2 + b.2^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_difference_l1350_135054


namespace NUMINAMATH_CALUDE_problem_solution_l1350_135033

theorem problem_solution (a b c : ℝ) (h1 : a = 8 - b) (h2 : c^2 = a*b - 16) : 
  a + c = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1350_135033


namespace NUMINAMATH_CALUDE_solve_equation_l1350_135091

theorem solve_equation : ∃ x : ℝ, 3 * x - 6 = |(-23 + 5)|^2 ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1350_135091


namespace NUMINAMATH_CALUDE_group_b_inspected_products_group_b_inspectors_l1350_135069

-- Define the number of workshops
def num_workshops : ℕ := 9

-- Define the number of inspectors in Group A
def group_a_inspectors : ℕ := 8

-- Define the initial number of finished products per workshop
variable (a : ℕ)

-- Define the daily production of finished products per workshop
variable (b : ℕ)

-- Define the number of days Group A inspects workshops 1 and 2
def days_group_a_1_2 : ℕ := 2

-- Define the number of days Group A inspects workshops 3 and 4
def days_group_a_3_4 : ℕ := 3

-- Define the total number of days for inspection
def total_inspection_days : ℕ := 5

-- Define the number of workshops inspected by Group B
def workshops_group_b : ℕ := 5

-- Theorem for the total number of finished products inspected by Group B
theorem group_b_inspected_products (a b : ℕ) :
  workshops_group_b * a + workshops_group_b * total_inspection_days * b = 5 * a + 25 * b :=
sorry

-- Theorem for the number of inspectors in Group B
theorem group_b_inspectors (a b : ℕ) (h : a = 4 * b) :
  (workshops_group_b * a + workshops_group_b * total_inspection_days * b) /
  ((3 / 4 : ℚ) * b * total_inspection_days) = 12 :=
sorry

end NUMINAMATH_CALUDE_group_b_inspected_products_group_b_inspectors_l1350_135069


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l1350_135065

theorem solve_quadratic_equation (k p : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) :
  let y : ℝ := -(p + k^2) / (2*k)
  (y - 2*k)^2 - (y - 3*k)^2 = 4*k^2 - p := by
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l1350_135065


namespace NUMINAMATH_CALUDE_fraction_simplification_l1350_135041

theorem fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (2 / y) / (3 / x^2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1350_135041


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1350_135074

theorem inequality_not_always_true
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : d ≠ 0) :
  ¬(∀ d, (a + d)^2 > (b + d)^2) ∧
  (a + c * d > b + c * d) ∧
  (a^2 - c * d > b^2 - c * d) ∧
  (a / c > b / c) ∧
  (Real.sqrt a * d^2 > Real.sqrt b * d^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1350_135074


namespace NUMINAMATH_CALUDE_rectangle_thirteen_squares_l1350_135025

/-- A rectangle can be divided into 13 equal squares if and only if its side length ratio is 13:1 -/
theorem rectangle_thirteen_squares (a b : ℕ) (h : a > 0 ∧ b > 0) :
  (∃ (s : ℕ), s > 0 ∧ a * b = 13 * s * s) ↔ (a = 13 * b ∨ b = 13 * a) :=
sorry

end NUMINAMATH_CALUDE_rectangle_thirteen_squares_l1350_135025


namespace NUMINAMATH_CALUDE_units_digit_plus_two_l1350_135027

/-- Given a positive even integer with a positive units digit, 
    if the units digit of its cube minus the units digit of its square is 0, 
    then the units digit of the number plus 2 is 8. -/
theorem units_digit_plus_two (p : ℕ) : 
  p > 0 → 
  Even p → 
  (p % 10 > 0) → 
  ((p^3 % 10) - (p^2 % 10) = 0) → 
  ((p + 2) % 10 = 8) := by
sorry

end NUMINAMATH_CALUDE_units_digit_plus_two_l1350_135027


namespace NUMINAMATH_CALUDE_cat_head_start_l1350_135003

/-- Proves that given a rabbit with speed 25 mph and a cat with speed 20 mph,
    if the rabbit catches up to the cat in 1 hour, then the cat's head start is 15 minutes. -/
theorem cat_head_start (rabbit_speed cat_speed : ℝ) (catch_up_time : ℝ) (head_start : ℝ) :
  rabbit_speed = 25 →
  cat_speed = 20 →
  catch_up_time = 1 →
  rabbit_speed * catch_up_time = cat_speed * (catch_up_time + head_start / 60) →
  head_start = 15 := by
  sorry

#check cat_head_start

end NUMINAMATH_CALUDE_cat_head_start_l1350_135003


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1350_135032

theorem two_digit_number_problem (x y : ℕ) :
  x < 10 ∧ y < 10 ∧ 
  (10 * x + y) - (10 * y + x) = 36 ∧
  x + y = 8 →
  10 * x + y = 62 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1350_135032


namespace NUMINAMATH_CALUDE_least_product_of_primes_above_25_l1350_135086

theorem least_product_of_primes_above_25 (p q : ℕ) : 
  p.Prime → q.Prime → p > 25 → q > 25 → p ≠ q → 
  ∃ (min_product : ℕ), min_product = 899 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r > 25 → s > 25 → r ≠ s → 
      p * q ≤ r * s := by
  sorry

end NUMINAMATH_CALUDE_least_product_of_primes_above_25_l1350_135086


namespace NUMINAMATH_CALUDE_second_quadrant_m_negative_l1350_135090

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The theorem stating that if a point P(m, 2) is in the second quadrant, then m < 0 -/
theorem second_quadrant_m_negative (m : ℝ) :
  SecondQuadrant ⟨m, 2⟩ → m < 0 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_m_negative_l1350_135090


namespace NUMINAMATH_CALUDE_polynomial_remainder_zero_l1350_135007

theorem polynomial_remainder_zero (x : ℝ) : 
  (x^3 - 5*x^2 + 2*x + 8) % (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_zero_l1350_135007


namespace NUMINAMATH_CALUDE_divisor_problem_l1350_135056

theorem divisor_problem (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 17698 →
  quotient = 89 →
  remainder = 14 →
  ∃ (divisor : ℕ), 
    dividend = divisor * quotient + remainder ∧
    divisor = 198 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l1350_135056


namespace NUMINAMATH_CALUDE_inverse_zero_product_l1350_135075

theorem inverse_zero_product (a b : ℝ) : a = 0 → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_zero_product_l1350_135075


namespace NUMINAMATH_CALUDE_heart_op_calculation_l1350_135055

def heart_op (a b : ℤ) : ℤ := Int.natAbs (a^2 - b^2)

theorem heart_op_calculation : heart_op 3 (heart_op 2 5) = 432 := by
  sorry

end NUMINAMATH_CALUDE_heart_op_calculation_l1350_135055


namespace NUMINAMATH_CALUDE_ax5_plus_by5_l1350_135053

theorem ax5_plus_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 4)
  (h2 : a * x^2 + b * y^2 = 10)
  (h3 : a * x^3 + b * y^3 = 28)
  (h4 : a * x^4 + b * y^4 = 60) :
  a * x^5 + b * y^5 = 229 + 1/3 := by
sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_l1350_135053


namespace NUMINAMATH_CALUDE_problem_statement_l1350_135004

theorem problem_statement : (π - 3.14) ^ 0 + (-0.125) ^ 2008 * 8 ^ 2008 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1350_135004


namespace NUMINAMATH_CALUDE_cristinas_pace_cristina_pace_is_3_l1350_135044

/-- Cristina's pace in a race with Nicky -/
theorem cristinas_pace (race_length : ℝ) (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ) : ℝ :=
  let cristinas_distance := nickys_pace * catch_up_time
  cristinas_distance / catch_up_time

/-- The main theorem stating Cristina's pace -/
theorem cristina_pace_is_3 : 
  cristinas_pace 300 12 3 30 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cristinas_pace_cristina_pace_is_3_l1350_135044


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l1350_135018

/-- The slope of a chord in an ellipse --/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 8 + y₁^2 / 6 = 1) →
  (x₂^2 / 8 + y₂^2 / 6 = 1) →
  ((x₁ + x₂) / 2 = 2) →
  ((y₁ + y₂) / 2 = 1) →
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l1350_135018


namespace NUMINAMATH_CALUDE_male_attendee_fraction_l1350_135057

theorem male_attendee_fraction :
  let male_fraction : ℝ → ℝ := λ x => x
  let female_fraction : ℝ → ℝ := λ x => 1 - x
  let male_on_time : ℝ → ℝ := λ x => (7/8) * x
  let female_on_time : ℝ → ℝ := λ x => (9/10) * (1 - x)
  let total_on_time : ℝ := 0.885
  ∀ x : ℝ, male_on_time x + female_on_time x = total_on_time → x = 0.6 :=
by
  sorry

end NUMINAMATH_CALUDE_male_attendee_fraction_l1350_135057


namespace NUMINAMATH_CALUDE_inequality_problem_l1350_135099

theorem inequality_problem (a b c d : ℝ) 
  (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1350_135099


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1350_135077

theorem complex_equation_solution (x : ℂ) : 5 - 2 * Complex.I * x = 7 - 5 * Complex.I * x ↔ x = (2 * Complex.I) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1350_135077


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l1350_135052

theorem trigonometric_inequality (φ : Real) (h : φ ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos φ) < Real.cos φ ∧ Real.cos φ < Real.cos (Real.sin φ) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l1350_135052


namespace NUMINAMATH_CALUDE_equality_of_squared_terms_l1350_135060

theorem equality_of_squared_terms (a b : ℝ) : 7 * a^2 * b - 7 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_squared_terms_l1350_135060


namespace NUMINAMATH_CALUDE_students_behind_minyoung_l1350_135058

/-- Given a line of students with Minyoung, prove the number behind her. -/
theorem students_behind_minyoung 
  (total : ℕ) 
  (in_front : ℕ) 
  (h1 : total = 35) 
  (h2 : in_front = 27) : 
  total - (in_front + 1) = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_minyoung_l1350_135058


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l1350_135015

/-- Proves that for a rectangular roof where the length is 7 times the width
    and the area is 847 square feet, the difference between the length
    and the width is 66 feet. -/
theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  length = 7 * width →
  length * width = 847 →
  length - width = 66 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l1350_135015


namespace NUMINAMATH_CALUDE_only_class_math_scores_comprehensive_l1350_135010

/-- Represents a survey scenario -/
inductive SurveyScenario
  | NationwideVision
  | LightBulbLifespan
  | ClassMathScores
  | DistrictIncome

/-- Determines if a survey scenario is suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  match scenario with
  | .ClassMathScores => true
  | _ => false

/-- The main theorem stating that only ClassMathScores is suitable for a comprehensive survey -/
theorem only_class_math_scores_comprehensive :
  ∀ (scenario : SurveyScenario),
    isSuitableForComprehensiveSurvey scenario ↔ scenario = SurveyScenario.ClassMathScores :=
by
  sorry

/-- Helper lemma: NationwideVision is not suitable for a comprehensive survey -/
lemma nationwide_vision_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.NationwideVision :=
by
  sorry

/-- Helper lemma: LightBulbLifespan is not suitable for a comprehensive survey -/
lemma light_bulb_lifespan_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.LightBulbLifespan :=
by
  sorry

/-- Helper lemma: DistrictIncome is not suitable for a comprehensive survey -/
lemma district_income_not_comprehensive :
  ¬ isSuitableForComprehensiveSurvey SurveyScenario.DistrictIncome :=
by
  sorry

/-- Helper lemma: ClassMathScores is suitable for a comprehensive survey -/
lemma class_math_scores_comprehensive :
  isSuitableForComprehensiveSurvey SurveyScenario.ClassMathScores :=
by
  sorry

end NUMINAMATH_CALUDE_only_class_math_scores_comprehensive_l1350_135010


namespace NUMINAMATH_CALUDE_single_point_equation_l1350_135079

/-- 
Theorem: If the equation 3x^2 + 4y^2 + 12x - 16y + d = 0 represents a single point, then d = 28.
-/
theorem single_point_equation (d : ℝ) : 
  (∃! p : ℝ × ℝ, 3 * p.1^2 + 4 * p.2^2 + 12 * p.1 - 16 * p.2 + d = 0) → 
  d = 28 := by
  sorry

end NUMINAMATH_CALUDE_single_point_equation_l1350_135079


namespace NUMINAMATH_CALUDE_x_minus_y_values_l1350_135097

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 5) (h3 : x * y < 0) :
  x - y = -7 ∨ x - y = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_values_l1350_135097


namespace NUMINAMATH_CALUDE_min_value_on_interval_l1350_135021

def f (x : ℝ) := x^2 - x

theorem min_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ f c = -1/4 ∧ ∀ x ∈ Set.Icc 0 1, f x ≥ f c := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l1350_135021


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1350_135000

theorem solution_set_implies_a_value 
  (h : ∀ x : ℝ, -1 < x ∧ x < 2 ↔ -1/2 * x^2 + a * x > -1) : 
  a = 1/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1350_135000


namespace NUMINAMATH_CALUDE_rectangle_triangle_max_area_and_hypotenuse_l1350_135063

theorem rectangle_triangle_max_area_and_hypotenuse (x y : ℝ) :
  x > 0 → y > 0 →  -- rectangle has positive dimensions
  x + y = 30 →     -- half the perimeter is 30
  (∃ h : ℝ, h^2 = x^2 + y^2) →  -- it's a right triangle
  x * y ≤ 225 ∧    -- max area is 225
  (x * y = 225 → ∃ h : ℝ, h^2 = x^2 + y^2 ∧ h = 15 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_triangle_max_area_and_hypotenuse_l1350_135063


namespace NUMINAMATH_CALUDE_system_solution_unique_l1350_135088

theorem system_solution_unique : ∃! (x y : ℝ), (3 * x - 4 * y = 12) ∧ (9 * x + 6 * y = -18) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1350_135088


namespace NUMINAMATH_CALUDE_paper_division_l1350_135006

/-- Represents the number of pieces after n divisions -/
def pieces (n : ℕ) : ℕ := 3 * n + 1

/-- The main theorem about paper division -/
theorem paper_division :
  (∀ n : ℕ, pieces n = 3 * n + 1) ∧
  (∃ n : ℕ, pieces n = 2011) :=
by sorry

end NUMINAMATH_CALUDE_paper_division_l1350_135006


namespace NUMINAMATH_CALUDE_logans_average_speed_l1350_135059

/-- Prove Logan's average speed given the driving conditions of Tamika and Logan -/
theorem logans_average_speed 
  (tamika_time : ℝ) 
  (tamika_speed : ℝ) 
  (logan_time : ℝ) 
  (distance_difference : ℝ) 
  (h1 : tamika_time = 8) 
  (h2 : tamika_speed = 45) 
  (h3 : logan_time = 5) 
  (h4 : tamika_time * tamika_speed = logan_time * logan_speed + distance_difference) 
  (h5 : distance_difference = 85) : 
  logan_speed = 55 := by
  sorry

end NUMINAMATH_CALUDE_logans_average_speed_l1350_135059


namespace NUMINAMATH_CALUDE_initial_number_equation_l1350_135096

theorem initial_number_equation : ∃ x : ℝ, 3 * (2 * x + 13) = 93 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_equation_l1350_135096


namespace NUMINAMATH_CALUDE_stating_all_magpies_fly_away_l1350_135024

/-- Represents the number of magpies remaining on a tree after a hunting incident -/
def magpies_remaining (initial : ℕ) (killed : ℕ) : ℕ :=
  0

/-- 
Theorem stating that regardless of the initial number of magpies and the number killed,
no magpies remain on the tree after the incident.
-/
theorem all_magpies_fly_away (initial : ℕ) (killed : ℕ) :
  magpies_remaining initial killed = 0 := by
  sorry

end NUMINAMATH_CALUDE_stating_all_magpies_fly_away_l1350_135024


namespace NUMINAMATH_CALUDE_max_pencil_length_in_square_hallway_l1350_135009

/-- Represents the length of a pencil that can navigate a square turn in a hallway -/
def max_pencil_length (L : ℝ) : ℝ := 3 * L

/-- Theorem stating that the maximum length of a pencil that can navigate a square turn
    in a hallway of width and height L is 3L -/
theorem max_pencil_length_in_square_hallway (L : ℝ) (h : L > 0) :
  max_pencil_length L = 3 * L :=
by sorry

end NUMINAMATH_CALUDE_max_pencil_length_in_square_hallway_l1350_135009


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1350_135046

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a + a₂ + a₄ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1350_135046


namespace NUMINAMATH_CALUDE_largest_three_digit_congruent_to_12_mod_15_l1350_135022

theorem largest_three_digit_congruent_to_12_mod_15 : ∃ n : ℕ,
  n = 987 ∧
  n ≥ 100 ∧ n < 1000 ∧
  n % 15 = 12 ∧
  ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 15 = 12 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruent_to_12_mod_15_l1350_135022


namespace NUMINAMATH_CALUDE_linear_function_properties_l1350_135028

-- Define the linear function
def f (k b x : ℝ) : ℝ := k * x + b

-- State the theorem
theorem linear_function_properties
  (k b : ℝ)
  (h1 : f k b 1 = 0)
  (h2 : f k b 0 = 2)
  (m : ℝ)
  (h3 : -2 < m)
  (h4 : m ≤ 3) :
  k = -2 ∧ b = 2 ∧ -4 ≤ f k b m ∧ f k b m < 6 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l1350_135028


namespace NUMINAMATH_CALUDE_proposition_b_l1350_135049

theorem proposition_b (a : ℝ) : 0 < a → a < 1 → a^3 < a := by sorry

end NUMINAMATH_CALUDE_proposition_b_l1350_135049


namespace NUMINAMATH_CALUDE_treasure_points_l1350_135078

theorem treasure_points (total_treasures : ℕ) (total_score : ℕ) 
  (h1 : total_treasures = 7) (h2 : total_score = 35) : 
  (total_score / total_treasures : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_treasure_points_l1350_135078


namespace NUMINAMATH_CALUDE_total_students_l1350_135061

theorem total_students (absent_percentage : ℝ) (present_students : ℕ) 
  (h1 : absent_percentage = 14) 
  (h2 : present_students = 86) : 
  ↑present_students / (1 - absent_percentage / 100) = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l1350_135061


namespace NUMINAMATH_CALUDE_circle_area_and_diameter_l1350_135089

theorem circle_area_and_diameter (C : ℝ) (h : C = 18 * Real.pi) : ∃ (A d : ℝ), A = 81 * Real.pi ∧ d = 18 ∧ A = Real.pi * (d / 2)^2 ∧ C = Real.pi * d := by
  sorry

end NUMINAMATH_CALUDE_circle_area_and_diameter_l1350_135089


namespace NUMINAMATH_CALUDE_inequality_proof_l1350_135042

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1350_135042


namespace NUMINAMATH_CALUDE_expression_value_l1350_135029

theorem expression_value (a b : ℝ) (h : a + b = 1) : a^2 - b^2 + 2*b + 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1350_135029


namespace NUMINAMATH_CALUDE_cos_equation_rational_solution_l1350_135030

theorem cos_equation_rational_solution (a : ℚ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : Real.cos (3 * Real.pi * a) + 2 * Real.cos (2 * Real.pi * a) = 0) : 
  a = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cos_equation_rational_solution_l1350_135030


namespace NUMINAMATH_CALUDE_number_of_gardens_l1350_135005

theorem number_of_gardens (pots_per_garden : ℕ) (flowers_per_pot : ℕ) (total_flowers : ℕ) :
  pots_per_garden = 544 →
  flowers_per_pot = 32 →
  total_flowers = 174080 →
  total_flowers / (pots_per_garden * flowers_per_pot) = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_gardens_l1350_135005


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1350_135045

theorem smallest_next_divisor_after_221 (n : ℕ) :
  (n ≥ 1000 ∧ n ≤ 9999) →  -- n is a 4-digit number
  Even n →                 -- n is even
  221 ∣ n →                -- 221 is a divisor of n
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ d ≤ 238 ∧ ∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1350_135045


namespace NUMINAMATH_CALUDE_totalSavingsIs4440_l1350_135014

-- Define the employees and their properties
structure Employee where
  name : String
  hourlyRate : ℚ
  hoursPerDay : ℚ
  savingRate : ℚ

-- Define the constants
def daysPerWeek : ℚ := 5
def numWeeks : ℚ := 4

-- Define the list of employees
def employees : List Employee := [
  ⟨"Robby", 10, 10, 2/5⟩,
  ⟨"Jaylen", 10, 8, 3/5⟩,
  ⟨"Miranda", 10, 10, 1/2⟩,
  ⟨"Alex", 12, 6, 1/3⟩,
  ⟨"Beth", 15, 4, 1/4⟩,
  ⟨"Chris", 20, 3, 3/4⟩
]

-- Calculate weekly savings for an employee
def weeklySavings (e : Employee) : ℚ :=
  e.hourlyRate * e.hoursPerDay * daysPerWeek * e.savingRate

-- Calculate total savings for all employees over the given number of weeks
def totalSavings : ℚ :=
  (employees.map weeklySavings).sum * numWeeks

-- Theorem statement
theorem totalSavingsIs4440 : totalSavings = 4440 := by
  sorry

end NUMINAMATH_CALUDE_totalSavingsIs4440_l1350_135014


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1350_135012

/-- Given an arithmetic sequence {a_n} with common ratio q ≠ 1,
    if a_1 * a_2 * a_3 = -1/8 and (a_2, a_4, a_3) forms an arithmetic sequence,
    then the sum of the first 4 terms of {a_n} is equal to 5/8. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n, a (n + 1) = a n * q) →
  a 1 * a 2 * a 3 = -1/8 →
  2 * a 4 = a 2 + a 3 →
  (a 1 + a 2 + a 3 + a 4 = 5/8) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1350_135012


namespace NUMINAMATH_CALUDE_max_m_is_zero_l1350_135083

/-- The condition function as described in the problem -/
def condition (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < m ∧ x₂ < m → (x₂ * Real.exp x₁ - x₁ * Real.exp x₂) / (Real.exp x₂ - Real.exp x₁) > 1

/-- The theorem stating that the maximum value of m for which the condition holds is 0 -/
theorem max_m_is_zero :
  ∀ m : ℝ, (∀ m' > m, ¬ condition m') → m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_max_m_is_zero_l1350_135083


namespace NUMINAMATH_CALUDE_max_sum_is_1120_l1350_135084

/-- Represents a splitting operation on a pile of coins -/
structure Split :=
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a > 1)
  (h2 : b ≥ 1)
  (h3 : c ≥ 1)
  (h4 : a = b + c)

/-- Represents the state of the coin piles -/
structure PileState :=
  (piles : List ℕ)
  (board_sum : ℕ)

/-- Performs a single split operation on a pile state -/
def split_pile (state : PileState) (split : Split) : PileState :=
  sorry

/-- Checks if the splitting process is complete -/
def is_complete (state : PileState) : Bool :=
  state.piles.length == 15 && state.piles.all (· == 1)

/-- Finds the maximum possible board sum after splitting 15 coins into 15 piles -/
def max_board_sum : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem max_sum_is_1120 :
  max_board_sum = 1120 :=
sorry

end NUMINAMATH_CALUDE_max_sum_is_1120_l1350_135084


namespace NUMINAMATH_CALUDE_simplify_expression_l1350_135047

theorem simplify_expression (p : ℝ) : ((6*p+2)-3*p*5)^2 + (5-2/4)*(8*p-12) = 81*p^2 - 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1350_135047


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1350_135070

/-- The constant term in the expansion of (1+2x^2)(x-1/x)^8 is -42 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (1 + 2*x^2) * (x - 1/x)^8
  ∃ g : ℝ → ℝ, (∀ x ≠ 0, f x = g x) ∧ g 0 = -42 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1350_135070


namespace NUMINAMATH_CALUDE_correct_inequalities_count_proof_correct_inequalities_count_l1350_135062

theorem correct_inequalities_count : ℕ :=
  let inequality1 := ∀ a : ℝ, a^2 + 1 ≥ 2*a
  let inequality2 := ∀ x : ℝ, x ≥ 2
  let inequality3 := ∀ x : ℝ, x^2 + x ≥ 1
  2

theorem proof_correct_inequalities_count : 
  (inequality1 → True) ∧ (inequality2 → False) ∧ (inequality3 → True) →
  correct_inequalities_count = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_inequalities_count_proof_correct_inequalities_count_l1350_135062


namespace NUMINAMATH_CALUDE_negation_of_implication_l1350_135026

theorem negation_of_implication :
  (¬(∀ x : ℝ, x > 5 → x > 0)) ↔ (∀ x : ℝ, x ≤ 5 → x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1350_135026


namespace NUMINAMATH_CALUDE_basketball_team_wins_l1350_135035

theorem basketball_team_wins (total_games : ℕ) (win_loss_difference : ℕ) 
  (h1 : total_games = 62) 
  (h2 : win_loss_difference = 28) : 
  let games_won := (total_games + win_loss_difference) / 2
  games_won = 45 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_wins_l1350_135035


namespace NUMINAMATH_CALUDE_megan_markers_count_l1350_135008

/-- The number of markers Megan has after receiving and giving away some -/
def final_markers (initial : ℕ) (received : ℕ) (given_away : ℕ) : ℕ :=
  initial + received - given_away

/-- Theorem stating that Megan's final number of markers is correct -/
theorem megan_markers_count :
  final_markers 217 109 35 = 291 :=
by sorry

end NUMINAMATH_CALUDE_megan_markers_count_l1350_135008


namespace NUMINAMATH_CALUDE_janice_starting_sentences_l1350_135072

/-- Represents the typing scenario for Janice --/
structure TypingScenario where
  typing_speed : ℕ
  first_session : ℕ
  second_session : ℕ
  third_session : ℕ
  erased_sentences : ℕ
  final_total : ℕ

/-- Calculates the number of sentences Janice started with today --/
def sentences_at_start (scenario : TypingScenario) : ℕ :=
  scenario.final_total - (scenario.typing_speed * (scenario.first_session + scenario.second_session + scenario.third_session) - scenario.erased_sentences)

/-- Theorem stating that Janice started with 258 sentences --/
theorem janice_starting_sentences (scenario : TypingScenario) 
  (h1 : scenario.typing_speed = 6)
  (h2 : scenario.first_session = 20)
  (h3 : scenario.second_session = 15)
  (h4 : scenario.third_session = 18)
  (h5 : scenario.erased_sentences = 40)
  (h6 : scenario.final_total = 536) :
  sentences_at_start scenario = 258 := by
  sorry

#eval sentences_at_start {
  typing_speed := 6,
  first_session := 20,
  second_session := 15,
  third_session := 18,
  erased_sentences := 40,
  final_total := 536
}

end NUMINAMATH_CALUDE_janice_starting_sentences_l1350_135072


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_ratio_bound_l1350_135082

theorem binomial_coefficient_sum_ratio_bound (n : ℕ+) :
  let a := 2^(n : ℕ)
  let b := 4^(n : ℕ)
  (b / a) + (a / b) ≥ (5 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_ratio_bound_l1350_135082


namespace NUMINAMATH_CALUDE_prove_grade_difference_l1350_135051

-- Define the grades as natural numbers
def jenny_grade : ℕ := 95
def bob_grade : ℕ := 35

-- Define Jason's grade in terms of Bob's
def jason_grade : ℕ := 2 * bob_grade

-- Define the difference between Jenny's and Jason's grades
def grade_difference : ℕ := jenny_grade - jason_grade

-- Theorem to prove
theorem prove_grade_difference : grade_difference = 25 := by
  sorry

end NUMINAMATH_CALUDE_prove_grade_difference_l1350_135051


namespace NUMINAMATH_CALUDE_ratio_of_fourth_power_equality_l1350_135039

theorem ratio_of_fourth_power_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4) :
  b / a = 1 := by sorry

end NUMINAMATH_CALUDE_ratio_of_fourth_power_equality_l1350_135039


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1350_135067

theorem min_value_expression (x y : ℝ) : x^2 + y^2 - 8*x - 6*y + 30 ≥ 5 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, x^2 + y^2 - 8*x - 6*y + 30 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1350_135067


namespace NUMINAMATH_CALUDE_barbara_candies_l1350_135034

/-- The number of candies Barbara has left after using some -/
def candies_left (initial : ℝ) (used : ℝ) : ℝ :=
  initial - used

theorem barbara_candies : 
  let initial_candies : ℝ := 18.0
  let used_candies : ℝ := 9.0
  candies_left initial_candies used_candies = 9.0 := by
sorry

end NUMINAMATH_CALUDE_barbara_candies_l1350_135034


namespace NUMINAMATH_CALUDE_ricky_roses_l1350_135093

def initial_roses : ℕ → ℕ → ℕ → ℕ → Prop
  | total, stolen, people, each =>
    total = stolen + people * each

theorem ricky_roses : initial_roses 40 4 9 4 := by
  sorry

end NUMINAMATH_CALUDE_ricky_roses_l1350_135093


namespace NUMINAMATH_CALUDE_min_blocks_needed_l1350_135073

/-- Represents a three-dimensional structure made of cube blocks -/
structure CubeStructure where
  blocks : ℕ → ℕ → ℕ → Bool

/-- The front view of the structure shows a 2x2 grid -/
def front_view_valid (s : CubeStructure) : Prop :=
  ∃ (i j : Fin 2), s.blocks i.val j.val 0 = true

/-- The left side view of the structure shows a 2x2 grid -/
def left_view_valid (s : CubeStructure) : Prop :=
  ∃ (i k : Fin 2), s.blocks 0 i.val k.val = true

/-- Count the number of blocks in the structure -/
def block_count (s : CubeStructure) : ℕ :=
  (Finset.range 2).sum fun i =>
    (Finset.range 2).sum fun j =>
      (Finset.range 2).sum fun k =>
        if s.blocks i j k then 1 else 0

/-- The main theorem: minimum number of blocks needed is 4 -/
theorem min_blocks_needed (s : CubeStructure) 
  (h_front : front_view_valid s) (h_left : left_view_valid s) :
  block_count s ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_min_blocks_needed_l1350_135073


namespace NUMINAMATH_CALUDE_fifty_second_card_is_ten_l1350_135066

def card_sequence : Fin 14 → String
| 0 => "A"
| 1 => "2"
| 2 => "3"
| 3 => "4"
| 4 => "5"
| 5 => "6"
| 6 => "7"
| 7 => "8"
| 8 => "9"
| 9 => "10"
| 10 => "J"
| 11 => "Q"
| 12 => "K"
| 13 => "Joker"

def nth_card (n : Nat) : String :=
  card_sequence (n % 14)

theorem fifty_second_card_is_ten :
  nth_card 51 = "10" := by
  sorry

end NUMINAMATH_CALUDE_fifty_second_card_is_ten_l1350_135066


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1350_135094

theorem complex_equation_sum (x y : ℝ) :
  (x + 2 * Complex.I = y - 1 + y * Complex.I) → x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1350_135094


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1350_135011

theorem power_fraction_simplification : (4 : ℝ)^800 / (8 : ℝ)^400 = (2 : ℝ)^400 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1350_135011


namespace NUMINAMATH_CALUDE_fraction_equality_l1350_135087

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4*x + 2*y) / (x - 4*y) = -3) : 
  (2*x + 8*y) / (4*x - 2*y) = 38/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1350_135087


namespace NUMINAMATH_CALUDE_recurring_decimal_subtraction_l1350_135020

theorem recurring_decimal_subtraction : 
  (1 : ℚ) / 3 - (2 : ℚ) / 99 = (31 : ℚ) / 99 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_subtraction_l1350_135020


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1350_135037

theorem no_solution_for_equation :
  ¬∃ (x : ℝ), x ≠ 1 ∧ (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1350_135037


namespace NUMINAMATH_CALUDE_inequality_proof_l1350_135071

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2*x + y + z)^2 * (2*x^2 + (y + z)^2) +
  (2*y + z + x)^2 * (2*y^2 + (z + x)^2) +
  (2*z + x + y)^2 * (2*z^2 + (x + y)^2) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1350_135071


namespace NUMINAMATH_CALUDE_square_side_length_when_area_equals_perimeter_l1350_135031

theorem square_side_length_when_area_equals_perimeter :
  ∃ (a : ℝ), a > 0 ∧ a^2 = 4*a :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_when_area_equals_perimeter_l1350_135031


namespace NUMINAMATH_CALUDE_water_jars_problem_l1350_135098

theorem water_jars_problem (S L : ℝ) (h1 : S > 0) (h2 : L > 0) (h3 : S < L) :
  let water_amount := S * (1/3)
  (water_amount = L * (1/2)) →
  (L * (1/2) + water_amount) / L = 1 := by
sorry

end NUMINAMATH_CALUDE_water_jars_problem_l1350_135098


namespace NUMINAMATH_CALUDE_unique_solution_system_l1350_135076

theorem unique_solution_system (x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  (1 / x + 1 / y + 1 / z = 3) →
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) →
  (1 / (x * y * z) = 1) →
  (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1350_135076


namespace NUMINAMATH_CALUDE_shirley_eggs_left_shirley_eggs_problem_l1350_135064

theorem shirley_eggs_left (initial_eggs : ℕ) (bought_eggs : ℕ) 
  (eggs_per_cupcake_batch1 : ℕ) (eggs_per_cupcake_batch2 : ℕ)
  (cupcakes_batch1 : ℕ) (cupcakes_batch2 : ℕ) : ℕ :=
  let total_eggs := initial_eggs + bought_eggs
  let eggs_used_batch1 := eggs_per_cupcake_batch1 * cupcakes_batch1
  let eggs_used_batch2 := eggs_per_cupcake_batch2 * cupcakes_batch2
  let total_eggs_used := eggs_used_batch1 + eggs_used_batch2
  total_eggs - total_eggs_used

theorem shirley_eggs_problem :
  shirley_eggs_left 98 8 5 7 6 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_shirley_eggs_left_shirley_eggs_problem_l1350_135064


namespace NUMINAMATH_CALUDE_school_trip_photos_l1350_135036

theorem school_trip_photos (c : ℕ) : 
  (3 * c = c + 12) →  -- Lisa and Robert have the same number of photos
  c = 6               -- Claire took 6 photos
  := by sorry

end NUMINAMATH_CALUDE_school_trip_photos_l1350_135036


namespace NUMINAMATH_CALUDE_solution_values_l1350_135068

-- Define the solution sets
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Define the equation with parameters a and b
def equation (a b : ℝ) (x : ℝ) : Prop := x^2 + a*x + b = 0

-- Theorem statement
theorem solution_values :
  ∃ (a b : ℝ), A_intersect_B = {x | equation a b x ∧ x^2 + a*x + b < 0} ∧ a = -1 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l1350_135068


namespace NUMINAMATH_CALUDE_beads_taken_out_l1350_135095

/-- Represents the number of beads in a container -/
structure BeadContainer where
  green : Nat
  brown : Nat
  red : Nat

/-- Calculates the total number of beads in a container -/
def totalBeads (container : BeadContainer) : Nat :=
  container.green + container.brown + container.red

theorem beads_taken_out (initial : BeadContainer) (left : Nat) :
  totalBeads initial = 6 → left = 4 → totalBeads initial - left = 2 := by
  sorry

end NUMINAMATH_CALUDE_beads_taken_out_l1350_135095


namespace NUMINAMATH_CALUDE_complex_fraction_sum_simplification_l1350_135023

theorem complex_fraction_sum_simplification :
  let i : ℂ := Complex.I
  ((4 + 7 * i) / (4 - 7 * i) + (4 - 7 * i) / (4 + 7 * i)) = (-66 : ℚ) / 65 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_simplification_l1350_135023


namespace NUMINAMATH_CALUDE_base6_addition_l1350_135080

/-- Converts a base 6 number to base 10 --/
def base6_to_base10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10_to_base6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Addition in base 6 --/
def add_base6 (a b : List Nat) : List Nat :=
  base10_to_base6 (base6_to_base10 a + base6_to_base10 b)

theorem base6_addition :
  add_base6 [2, 5, 4, 1] [4, 5, 3, 2] = [0, 5, 2, 4] := by sorry

end NUMINAMATH_CALUDE_base6_addition_l1350_135080


namespace NUMINAMATH_CALUDE_proposition_analysis_l1350_135013

theorem proposition_analysis (a b c : ℝ) : 
  (∀ x y z : ℝ, (x ≤ y → x*z^2 ≤ y*z^2)) ∧ 
  (∃ x y z : ℝ, (x > y ∧ x*z^2 ≤ y*z^2)) ∧
  (∀ x y z : ℝ, (x*z^2 > y*z^2 → x > y)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l1350_135013


namespace NUMINAMATH_CALUDE_f_properties_l1350_135050

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*(x^2 - 2)

theorem f_properties :
  -- 1. Monotonically increasing intervals
  (∀ x < -Real.sqrt 2, f' x > 0) ∧
  (∀ x > Real.sqrt 2, f' x > 0) ∧
  -- 2. Monotonically decreasing interval
  (∀ x ∈ Set.Ioo (-Real.sqrt 2) (Real.sqrt 2), f' x < 0) ∧
  -- 3. Maximum value
  (f (-Real.sqrt 2) = 5 + 4*Real.sqrt 2) ∧
  (∀ x, f x ≤ 5 + 4*Real.sqrt 2) ∧
  -- 4. Minimum value
  (f (Real.sqrt 2) = 5 - 4*Real.sqrt 2) ∧
  (∀ x, f x ≥ 5 - 4*Real.sqrt 2) ∧
  -- 5. Equation of tangent line at (1, 0)
  (∀ x, f 1 + f' 1 * (x - 1) = -3*x + 3) := by
  sorry


end NUMINAMATH_CALUDE_f_properties_l1350_135050


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1350_135040

/-- The sum of specific fractions is equal to -2/15 -/
theorem sum_of_fractions :
  (1 : ℚ) / 3 + 1 / 2 + (-5) / 6 + 1 / 5 + 1 / 4 + (-9) / 20 + (-2) / 15 = -2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1350_135040


namespace NUMINAMATH_CALUDE_pyramid_vertex_on_face_plane_l1350_135001

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Represents a triangular pyramid in 3D space -/
structure TriangularPyramid where
  v1 : Point3D
  v2 : Point3D
  v3 : Point3D
  v4 : Point3D

/-- Checks if a point lies on a plane defined by three other points -/
def pointLiesOnPlane (p : Point3D) (p1 p2 p3 : Point3D) : Prop := sorry

/-- Main theorem: Each vertex of one pyramid lies on a face plane of the other pyramid -/
theorem pyramid_vertex_on_face_plane (p : Parallelepiped) : 
  let pyramid1 := TriangularPyramid.mk p.A p.B p.D p.D₁
  let pyramid2 := TriangularPyramid.mk p.A₁ p.B₁ p.C₁ p.C
  (pointLiesOnPlane pyramid1.v1 pyramid2.v1 pyramid2.v3 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v2 pyramid2.v2 pyramid2.v3 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v3 pyramid2.v1 pyramid2.v2 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v4 pyramid2.v1 pyramid2.v2 pyramid2.v3) ∧
  (pointLiesOnPlane pyramid2.v1 pyramid1.v1 pyramid1.v3 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v2 pyramid1.v2 pyramid1.v3 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v3 pyramid1.v1 pyramid1.v2 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v4 pyramid1.v1 pyramid1.v2 pyramid1.v3) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_vertex_on_face_plane_l1350_135001


namespace NUMINAMATH_CALUDE_minibus_students_l1350_135019

theorem minibus_students (boys : ℕ) (girls : ℕ) : 
  boys = 8 →
  girls = boys + 2 →
  boys + girls = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_minibus_students_l1350_135019


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l1350_135043

theorem quadratic_inequality_no_solution 
  (a b c : ℝ) (h : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c < 0) ↔ a < 0 ∧ b^2 - 4*a*c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l1350_135043


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1350_135048

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4 * p * x

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / b^2 = 1

/-- The problem statement -/
theorem parabola_hyperbola_intersection (p : Parabola) (h : Hyperbola) : 
  (h.a > 0) →
  (h.b > 0) →
  (p.p = 2 * h.a) →  -- directrix passes through one focus of hyperbola
  (p.eq (3/2) (Real.sqrt 6)) →  -- intersection point
  (h.eq (3/2) (Real.sqrt 6)) →  -- intersection point
  (p.p = 1) ∧ (h.a^2 = 1/4) ∧ (h.b^2 = 3/4) := by
  sorry

#check parabola_hyperbola_intersection

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1350_135048


namespace NUMINAMATH_CALUDE_tangent_circles_diameter_intersection_l1350_135092

/-- Given three circles that are pairwise tangent, the lines connecting
    the tangency points of two circles intersect the third circle at
    the endpoints of its diameter. -/
theorem tangent_circles_diameter_intersection
  (O₁ O₂ O₃ : ℝ × ℝ) -- Centers of the three circles
  (r₁ r₂ r₃ : ℝ) -- Radii of the three circles
  (h_positive : r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) -- Radii are positive
  (h_tangent : -- Circles are pairwise tangent
    (O₁.1 - O₂.1)^2 + (O₁.2 - O₂.2)^2 = (r₁ + r₂)^2 ∧
    (O₂.1 - O₃.1)^2 + (O₂.2 - O₃.2)^2 = (r₂ + r₃)^2 ∧
    (O₃.1 - O₁.1)^2 + (O₃.2 - O₁.2)^2 = (r₃ + r₁)^2)
  (h_distinct : O₁ ≠ O₂ ∧ O₂ ≠ O₃ ∧ O₃ ≠ O₁) -- Centers are distinct
  : ∃ (A B C : ℝ × ℝ), -- Tangency points
    -- A is on circle 1 and 2
    ((A.1 - O₁.1)^2 + (A.2 - O₁.2)^2 = r₁^2 ∧ (A.1 - O₂.1)^2 + (A.2 - O₂.2)^2 = r₂^2) ∧
    -- B is on circle 2 and 3
    ((B.1 - O₂.1)^2 + (B.2 - O₂.2)^2 = r₂^2 ∧ (B.1 - O₃.1)^2 + (B.2 - O₃.2)^2 = r₃^2) ∧
    -- C is on circle 1 and 3
    ((C.1 - O₁.1)^2 + (C.2 - O₁.2)^2 = r₁^2 ∧ (C.1 - O₃.1)^2 + (C.2 - O₃.2)^2 = r₃^2) ∧
    -- Lines AB and AC intersect circle 3 at diameter endpoints
    ∃ (M K : ℝ × ℝ),
      (M.1 - O₃.1)^2 + (M.2 - O₃.2)^2 = r₃^2 ∧
      (K.1 - O₃.1)^2 + (K.2 - O₃.2)^2 = r₃^2 ∧
      (M.1 - K.1)^2 + (M.2 - K.2)^2 = 4 * r₃^2 ∧
      (∃ t : ℝ, M = (1 - t) • A + t • B) ∧
      (∃ s : ℝ, K = (1 - s) • A + s • C) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_diameter_intersection_l1350_135092


namespace NUMINAMATH_CALUDE_product_expansion_sum_l1350_135085

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(9 - x) = a*x^3 + b*x^2 + c*x + d) → 
  9*a + 3*b + c + d = 58 := by
sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l1350_135085


namespace NUMINAMATH_CALUDE_harper_mineral_water_cost_l1350_135002

/-- Harper's mineral water purchase problem -/
theorem harper_mineral_water_cost 
  (daily_consumption : ℚ) 
  (bottles_per_case : ℕ) 
  (cost_per_case : ℚ) 
  (days : ℕ) : 
  daily_consumption = 1/2 → 
  bottles_per_case = 24 → 
  cost_per_case = 12 → 
  days = 240 → 
  (days * daily_consumption / bottles_per_case).ceil * cost_per_case = 60 := by
  sorry

end NUMINAMATH_CALUDE_harper_mineral_water_cost_l1350_135002


namespace NUMINAMATH_CALUDE_shirt_cost_l1350_135038

theorem shirt_cost (initial_amount : ℕ) (change : ℕ) (shirt_cost : ℕ) : 
  initial_amount = 50 → change = 23 → shirt_cost = initial_amount - change → shirt_cost = 27 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l1350_135038


namespace NUMINAMATH_CALUDE_max_lg_product_l1350_135081

open Real

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := log x / log 10

-- State the theorem
theorem max_lg_product (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) 
  (h : lg x ^ 2 + lg y ^ 2 = lg (10 * x ^ 2) + lg (10 * y ^ 2)) :
  ∃ (max : ℝ), max = 2 + 2 * sqrt 2 ∧ lg (x * y) ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_lg_product_l1350_135081


namespace NUMINAMATH_CALUDE_total_fish_l1350_135017

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 14) : 
  lilly_fish + rosy_fish = 24 := by
sorry

end NUMINAMATH_CALUDE_total_fish_l1350_135017
