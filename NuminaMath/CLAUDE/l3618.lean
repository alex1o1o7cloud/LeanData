import Mathlib

namespace NUMINAMATH_CALUDE_existence_of_special_polygon_l3618_361852

-- Define what it means for a polygon to have a center of symmetry
def has_center_of_symmetry (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a set to be a polygon
def is_polygon (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a polygon to be convex
def is_convex (P : Set ℝ × ℝ) : Prop := sorry

-- Define what it means for a polygon to be divided into two parts
def can_be_divided_into (P A B : Set ℝ × ℝ) : Prop := sorry

theorem existence_of_special_polygon : 
  ∃ (P A B : Set ℝ × ℝ), 
    is_polygon P ∧ 
    ¬(has_center_of_symmetry P) ∧
    is_polygon A ∧ 
    is_polygon B ∧
    is_convex A ∧ 
    is_convex B ∧
    can_be_divided_into P A B ∧
    has_center_of_symmetry A ∧
    has_center_of_symmetry B := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_polygon_l3618_361852


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3618_361829

-- Define the coefficients of the two lines as functions of m
def line1_coeff (m : ℝ) : ℝ × ℝ := (m + 2, m)
def line2_coeff (m : ℝ) : ℝ × ℝ := (m - 1, m - 4)

-- Define the perpendicularity condition
def perpendicular (m : ℝ) : Prop :=
  (line1_coeff m).1 * (line2_coeff m).1 + (line1_coeff m).2 * (line2_coeff m).2 = 0

-- State the theorem
theorem perpendicular_lines_m_values :
  ∀ m : ℝ, perpendicular m → m = -1/2 ∨ m = 2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3618_361829


namespace NUMINAMATH_CALUDE_peony_count_l3618_361856

theorem peony_count (n : ℕ) 
  (h1 : ∃ (x : ℕ), n = 4*x + 2*x + 6*x) 
  (h2 : ∃ (y : ℕ), 6*y - 4*y = 30) 
  (h3 : ∃ (z : ℕ), 4 + 2 + 6 = 12) : n = 180 := by
  sorry

end NUMINAMATH_CALUDE_peony_count_l3618_361856


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3618_361834

theorem triangle_angle_measure (a b c d : Real) (h1 : a + b + c = 180) 
  (h2 : b = 68) (h3 : c = 35) : a = 77 := by
  sorry

#check triangle_angle_measure

end NUMINAMATH_CALUDE_triangle_angle_measure_l3618_361834


namespace NUMINAMATH_CALUDE_lamp_probability_l3618_361886

theorem lamp_probability : Real → Prop :=
  fun p =>
    let total_length : Real := 6
    let min_distance : Real := 2
    p = (total_length - 2 * min_distance) / total_length

#check lamp_probability (1/3)

end NUMINAMATH_CALUDE_lamp_probability_l3618_361886


namespace NUMINAMATH_CALUDE_ellipse_inscribed_triangle_uniqueness_l3618_361891

/-- Represents an ellipse with semi-major axis a and semi-minor axis 1 -/
def Ellipse (a : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + p.2^2 = 1}

/-- Represents a right-angled isosceles triangle inscribed in the ellipse -/
def InscribedTriangle (a : ℝ) := 
  {triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) | 
    let (A, B, C) := triangle
    B = (0, 1) ∧ 
    A ∈ Ellipse a ∧ 
    C ∈ Ellipse a ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
    (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0}

/-- The main theorem -/
theorem ellipse_inscribed_triangle_uniqueness (a : ℝ) 
  (h1 : a > 1) 
  (h2 : ∃! triangle, triangle ∈ InscribedTriangle a) : 
  1 < a ∧ a ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_inscribed_triangle_uniqueness_l3618_361891


namespace NUMINAMATH_CALUDE_symmetric_complex_product_l3618_361868

theorem symmetric_complex_product (z₁ z₂ : ℂ) :
  (z₁.re = 2 ∧ z₁.im = 1) →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  z₁ * z₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_product_l3618_361868


namespace NUMINAMATH_CALUDE_sum_vector_magnitude_l3618_361899

/-- Given two vectors a and b in ℝ³, prove that their sum has magnitude √26 -/
theorem sum_vector_magnitude (a b : ℝ × ℝ × ℝ) : 
  a = (1, -1, 0) → b = (3, -2, 1) → ‖a + b‖ = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_vector_magnitude_l3618_361899


namespace NUMINAMATH_CALUDE_survivor_quitters_probability_l3618_361877

/-- The probability of all three quitters being from the same tribe in a Survivor-like scenario -/
theorem survivor_quitters_probability (total_people : ℕ) (tribe_size : ℕ) (quitters : ℕ)
  (h1 : total_people = 20)
  (h2 : tribe_size = 10)
  (h3 : quitters = 3)
  (h4 : total_people = 2 * tribe_size) :
  (Nat.choose tribe_size quitters * 2 : ℚ) / Nat.choose total_people quitters = 20 / 95 := by
  sorry

end NUMINAMATH_CALUDE_survivor_quitters_probability_l3618_361877


namespace NUMINAMATH_CALUDE_binomial_probability_two_successes_l3618_361849

-- Define the parameters of the binomial distribution
def n : ℕ := 6
def p : ℚ := 1/3

-- Define the probability mass function for the binomial distribution
def binomial_pmf (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- State the theorem
theorem binomial_probability_two_successes :
  binomial_pmf 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_successes_l3618_361849


namespace NUMINAMATH_CALUDE_total_score_l3618_361812

-- Define the players and their scores
def Alex : ℕ := 18
def Sam : ℕ := Alex / 2
def Jon : ℕ := 2 * Sam + 3
def Jack : ℕ := Jon + 5
def Tom : ℕ := Jon + Jack - 4

-- State the theorem
theorem total_score : Alex + Sam + Jon + Jack + Tom = 117 := by
  sorry

end NUMINAMATH_CALUDE_total_score_l3618_361812


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3618_361837

theorem quadratic_inequality_range (θ : Real) :
  (∀ m : Real, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (∀ m : Real, m ≥ 4 ∨ m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3618_361837


namespace NUMINAMATH_CALUDE_certain_amount_proof_l3618_361823

theorem certain_amount_proof (x : ℝ) (A : ℝ) : 
  x = 780 → 
  (0.25 * x) = (0.15 * 1500 - A) → 
  A = 30 := by
sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l3618_361823


namespace NUMINAMATH_CALUDE_cone_volume_l3618_361871

/-- The volume of a cone with slant height 5 and lateral area 20π is 16π -/
theorem cone_volume (s : ℝ) (lateral_area : ℝ) (h : s = 5) (h' : lateral_area = 20 * Real.pi) :
  (1 / 3 : ℝ) * Real.pi * (lateral_area / (Real.pi * s))^2 * Real.sqrt (s^2 - (lateral_area / (Real.pi * s))^2) = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3618_361871


namespace NUMINAMATH_CALUDE_unique_score_with_three_combinations_l3618_361860

/-- Represents a scoring combination for the test -/
structure ScoringCombination where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ

/-- Calculates the score for a given scoring combination -/
def calculateScore (sc : ScoringCombination) : ℕ :=
  6 * sc.correct + 3 * sc.unanswered

/-- Checks if a scoring combination is valid (sums to 25 questions) -/
def isValidCombination (sc : ScoringCombination) : Prop :=
  sc.correct + sc.unanswered + sc.incorrect = 25

/-- Theorem: 78 is the only score achievable in exactly three ways -/
theorem unique_score_with_three_combinations :
  ∃! score : ℕ,
    (∃ (combinations : Finset ScoringCombination),
      combinations.card = 3 ∧
      (∀ sc ∈ combinations, isValidCombination sc ∧ calculateScore sc = score) ∧
      (∀ sc : ScoringCombination, isValidCombination sc ∧ calculateScore sc = score → sc ∈ combinations)) ∧
    score = 78 := by
  sorry

end NUMINAMATH_CALUDE_unique_score_with_three_combinations_l3618_361860


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l3618_361858

/-- The standard equation of an ellipse with specific parameters. -/
theorem ellipse_standard_equation 
  (foci_on_y_axis : Bool) 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) : 
  foci_on_y_axis ∧ major_axis_length = 20 ∧ eccentricity = 2/5 → 
  ∃ (x y : ℝ), y^2/100 + x^2/84 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l3618_361858


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3618_361803

def f (x : ℝ) : ℝ := (x - 2)^2 - 3

theorem vertex_of_quadratic :
  ∃ (a b c : ℝ), f x = a * (x - b)^2 + c ∧ f b = c ∧ b = 2 ∧ c = -3 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3618_361803


namespace NUMINAMATH_CALUDE_f_minimized_at_x_min_l3618_361811

/-- The quadratic function we're minimizing -/
def f (x : ℝ) := 2 * x^2 - 8 * x + 6

/-- The value of x that minimizes f -/
def x_min : ℝ := 2

theorem f_minimized_at_x_min :
  ∀ x : ℝ, f x_min ≤ f x :=
sorry

end NUMINAMATH_CALUDE_f_minimized_at_x_min_l3618_361811


namespace NUMINAMATH_CALUDE_tower_height_calculation_l3618_361875

-- Define the tower and measurement points
structure Tower :=
  (height : ℝ)

structure MeasurementPoints :=
  (distanceAD : ℝ)
  (angleA : ℝ)
  (angleD : ℝ)

-- Define the theorem
theorem tower_height_calculation (t : Tower) (m : MeasurementPoints) 
  (h_distanceAD : m.distanceAD = 129)
  (h_angleA : m.angleA = 45)
  (h_angleD : m.angleD = 60) :
  t.height = 305 := by
  sorry


end NUMINAMATH_CALUDE_tower_height_calculation_l3618_361875


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l3618_361846

theorem sqrt_sum_problem (a b : ℝ) 
  (h1 : Real.sqrt 44 = 2 * Real.sqrt a) 
  (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : 
  a + b = 17 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l3618_361846


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3618_361847

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3618_361847


namespace NUMINAMATH_CALUDE_individual_test_scores_l3618_361857

/-- Represents a student's test score -/
structure TestScore where
  value : ℝ

/-- Represents the population of students -/
def Population : Type := Fin 2100

/-- Represents the sample of students -/
def Sample : Type := Fin 100

/-- A function that assigns a test score to each student in the population -/
def scoreAssignment : Population → TestScore := sorry

/-- A function that selects the sample from the population -/
def sampleSelection : Sample → Population := sorry

theorem individual_test_scores 
  (p : Population) 
  (s : Sample) : 
  scoreAssignment p ≠ scoreAssignment (sampleSelection s) → p ≠ sampleSelection s := by
  sorry

end NUMINAMATH_CALUDE_individual_test_scores_l3618_361857


namespace NUMINAMATH_CALUDE_drawing_probability_comparison_l3618_361801

theorem drawing_probability_comparison : 
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let draws : ℕ := 3

  let prob_with_replacement : ℚ := 3 / 8
  let prob_without_replacement : ℚ := 5 / 12

  prob_without_replacement > prob_with_replacement := by
  sorry

end NUMINAMATH_CALUDE_drawing_probability_comparison_l3618_361801


namespace NUMINAMATH_CALUDE_hostel_provisions_l3618_361896

/-- The number of days the provisions would last for the initial number of men -/
def initial_days : ℕ := 28

/-- The number of days the provisions would last if 50 men left -/
def extended_days : ℕ := 35

/-- The number of men that would leave -/
def men_leaving : ℕ := 50

/-- The initial number of men in the hostel -/
def initial_men : ℕ := 250

theorem hostel_provisions :
  initial_men * initial_days = (initial_men - men_leaving) * extended_days := by
  sorry

end NUMINAMATH_CALUDE_hostel_provisions_l3618_361896


namespace NUMINAMATH_CALUDE_expression_evaluation_l3618_361853

theorem expression_evaluation : 8 - 6 / (4 - 2) = 5 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3618_361853


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3618_361827

theorem complex_equation_solution (x y : ℝ) : 
  (x / (1 + Complex.I)) + (y / (1 + 2 * Complex.I)) = 5 / (1 + Complex.I) → y = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3618_361827


namespace NUMINAMATH_CALUDE_zero_points_sum_inequality_l3618_361836

theorem zero_points_sum_inequality (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂)
  (h₄ : Real.log x₁ - a * x₁ = 0) (h₅ : Real.log x₂ - a * x₂ = 0) :
  x₁ + x₂ > 2 / a :=
by sorry

end NUMINAMATH_CALUDE_zero_points_sum_inequality_l3618_361836


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3618_361840

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 5*x + 6 ≤ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l3618_361840


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l3618_361825

/-- The length of a train given specific conditions --/
theorem train_length : ℝ :=
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let passing_time : ℝ := 34 -- seconds

  100 -- meters

/-- Proof that the train length is correct given the conditions --/
theorem train_length_proof :
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let passing_time : ℝ := 34 -- seconds
  train_length = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l3618_361825


namespace NUMINAMATH_CALUDE_r_plus_s_equals_12_l3618_361883

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

-- Define points P and Q
def P : ℝ × ℝ := (16, 0)
def Q : ℝ × ℝ := (0, 8)

-- Define point T
def T (r s : ℝ) : ℝ × ℝ := (r, s)

-- Define that T is on line segment PQ
def T_on_PQ (r s : ℝ) : Prop :=
  line_equation r s ∧ 0 ≤ r ∧ r ≤ 16

-- Define the area of triangle POQ
def area_POQ : ℝ := 64

-- Define the area of triangle TOP
def area_TOP (s : ℝ) : ℝ := 8 * s

-- Theorem statement
theorem r_plus_s_equals_12 (r s : ℝ) :
  T_on_PQ r s → area_POQ = 2 * area_TOP s → r + s = 12 :=
sorry

end NUMINAMATH_CALUDE_r_plus_s_equals_12_l3618_361883


namespace NUMINAMATH_CALUDE_simplify_expression_l3618_361808

theorem simplify_expression (y : ℝ) : y - 3*(2+y) + 4*(2-y) - 5*(2+3*y) = -21*y - 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3618_361808


namespace NUMINAMATH_CALUDE_estimate_proportion_approx_5_7_percent_l3618_361841

/-- Represents the survey data and population information -/
structure SurveyData where
  total_households : ℕ
  ordinary_households : ℕ
  high_income_households : ℕ
  sample_ordinary : ℕ
  sample_high_income : ℕ
  total_with_3plus_housing : ℕ
  ordinary_with_3plus_housing : ℕ
  high_income_with_3plus_housing : ℕ

/-- Calculates the estimated proportion of households with 3+ housing sets -/
def estimate_proportion (data : SurveyData) : ℝ :=
  sorry

/-- Theorem stating that the estimated proportion is approximately 5.7% -/
theorem estimate_proportion_approx_5_7_percent (data : SurveyData)
  (h1 : data.total_households = 100000)
  (h2 : data.ordinary_households = 99000)
  (h3 : data.high_income_households = 1000)
  (h4 : data.sample_ordinary = 990)
  (h5 : data.sample_high_income = 100)
  (h6 : data.total_with_3plus_housing = 120)
  (h7 : data.ordinary_with_3plus_housing = 50)
  (h8 : data.high_income_with_3plus_housing = 70) :
  ∃ ε > 0, |estimate_proportion data - 0.057| < ε :=
sorry

end NUMINAMATH_CALUDE_estimate_proportion_approx_5_7_percent_l3618_361841


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3618_361894

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_4 = 16, prove a_3 = 8 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 2 + a 4 = 16) : 
  a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3618_361894


namespace NUMINAMATH_CALUDE_sum_of_divisors_3k_plus_2_multiple_of_3_l3618_361848

/-- The sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- A number is of the form 3k + 2 -/
def is_3k_plus_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k + 2

theorem sum_of_divisors_3k_plus_2_multiple_of_3 (n : ℕ) (h : is_3k_plus_2 n) :
  3 ∣ sum_of_divisors n :=
sorry

end NUMINAMATH_CALUDE_sum_of_divisors_3k_plus_2_multiple_of_3_l3618_361848


namespace NUMINAMATH_CALUDE_new_tax_rate_is_30_percent_l3618_361850

/-- Calculates the new tax rate given the initial rate, income, and tax savings -/
def calculate_new_tax_rate (initial_rate : ℚ) (income : ℚ) (savings : ℚ) : ℚ :=
  let initial_tax := initial_rate * income
  let new_tax := initial_tax - savings
  new_tax / income

theorem new_tax_rate_is_30_percent :
  let initial_rate : ℚ := 45 / 100
  let income : ℚ := 48000
  let savings : ℚ := 7200
  calculate_new_tax_rate initial_rate income savings = 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_new_tax_rate_is_30_percent_l3618_361850


namespace NUMINAMATH_CALUDE_committee_formation_l3618_361800

theorem committee_formation (total : ℕ) (mathematicians : ℕ) (economists : ℕ) (committee_size : ℕ) :
  total = mathematicians + economists →
  mathematicians = 3 →
  economists = 10 →
  committee_size = 7 →
  (Nat.choose total committee_size) - (Nat.choose economists committee_size) = 1596 :=
by sorry

end NUMINAMATH_CALUDE_committee_formation_l3618_361800


namespace NUMINAMATH_CALUDE_kabulek_numbers_are_correct_l3618_361810

/-- A four-digit number is a Kabulek number if it equals the square of the sum of its first two digits and last two digits. -/
def isKabulek (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧ ∃ a b : ℕ, 
    a ≥ 10 ∧ a < 100 ∧ b ≥ 0 ∧ b < 100 ∧
    n = 100 * a + b ∧ n = (a + b)^2

/-- The set of all four-digit Kabulek numbers. -/
def kabulekNumbers : Set ℕ := {2025, 3025, 9801}

/-- Theorem stating that the set of all four-digit Kabulek numbers is exactly {2025, 3025, 9801}. -/
theorem kabulek_numbers_are_correct : 
  ∀ n : ℕ, isKabulek n ↔ n ∈ kabulekNumbers := by sorry

end NUMINAMATH_CALUDE_kabulek_numbers_are_correct_l3618_361810


namespace NUMINAMATH_CALUDE_greater_sum_from_inequalities_l3618_361864

theorem greater_sum_from_inequalities (a b c d : ℝ) 
  (h1 : a^2 + b > c^2 + d) 
  (h2 : a + b^2 > c + d^2) 
  (h3 : a ≥ (1/2 : ℝ)) 
  (h4 : b ≥ (1/2 : ℝ)) 
  (h5 : c ≥ (1/2 : ℝ)) 
  (h6 : d ≥ (1/2 : ℝ)) : 
  a + b > c + d := by
  sorry

end NUMINAMATH_CALUDE_greater_sum_from_inequalities_l3618_361864


namespace NUMINAMATH_CALUDE_ellipse_intersection_constant_sum_distance_l3618_361892

/-- The slope of a line that intersects an ellipse such that the sum of squared distances
    from any point on the major axis to the intersection points is constant. -/
theorem ellipse_intersection_constant_sum_distance (k : ℝ) : 
  (∀ a : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (A.2 - 0 = k * (A.1 - a)) ∧
    (B.2 - 0 = k * (B.1 - a)) ∧
    ((A.1 - a)^2 + A.2^2 + (B.1 - a)^2 + B.2^2 = (512 - 800 * k^2) / (16 + 25 * k^2))) →
  k = 4/5 ∨ k = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_constant_sum_distance_l3618_361892


namespace NUMINAMATH_CALUDE_star_sum_five_l3618_361893

def star (a b : ℕ) : ℕ := a^b - a*b

theorem star_sum_five :
  ∀ a b : ℕ,
  a ≥ 2 →
  b ≥ 2 →
  star a b = 2 →
  a + b = 5 :=
by sorry

end NUMINAMATH_CALUDE_star_sum_five_l3618_361893


namespace NUMINAMATH_CALUDE_three_integer_chords_l3618_361804

/-- A circle with a point P inside --/
structure CircleWithPoint where
  radius : ℝ
  distanceToCenter : ℝ

/-- Count of integer-length chords through P --/
def integerChordCount (c : CircleWithPoint) : ℕ :=
  sorry

theorem three_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 12)
  (h2 : c.distanceToCenter = 5) : 
  integerChordCount c = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_chords_l3618_361804


namespace NUMINAMATH_CALUDE_tan_product_undefined_l3618_361816

theorem tan_product_undefined : 
  ¬∃ (x : ℝ), Real.tan (π / 6) * Real.tan (π / 3) * Real.tan (π / 2) = x :=
by sorry

end NUMINAMATH_CALUDE_tan_product_undefined_l3618_361816


namespace NUMINAMATH_CALUDE_solution_exists_l3618_361809

theorem solution_exists : ∃ c : ℝ, 
  (∃ x : ℤ, (x = ⌊c⌋ ∧ 3 * (x : ℝ)^2 - 9 * (x : ℝ) - 30 = 0)) ∧
  (∃ y : ℝ, (y = c - ⌊c⌋ ∧ 4 * y^2 - 8 * y + 1 = 0)) ∧
  (c = -1 - Real.sqrt 3 / 2 ∨ c = 6 - Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l3618_361809


namespace NUMINAMATH_CALUDE_heidi_painting_fraction_l3618_361889

/-- Represents the time in minutes it takes Heidi to paint a wall -/
def total_time : ℚ := 45

/-- Represents the time in minutes we want to calculate the painted fraction for -/
def given_time : ℚ := 9

/-- Represents the fraction of the wall painted in the given time -/
def painted_fraction : ℚ := given_time / total_time

theorem heidi_painting_fraction :
  painted_fraction = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_heidi_painting_fraction_l3618_361889


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_one_l3618_361851

theorem tan_sum_product_equals_one :
  let tan15 : ℝ := 2 - Real.sqrt 3
  let tan30 : ℝ := Real.sqrt 3 / 3
  tan15 + tan30 + tan15 * tan30 = 1 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_one_l3618_361851


namespace NUMINAMATH_CALUDE_problem_statement_l3618_361890

theorem problem_statement (n b : ℝ) : 
  n = 2^(7/3) → n^(3*b + 5) = 256 → b = -11/21 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3618_361890


namespace NUMINAMATH_CALUDE_all_greater_than_2ab_in_S_l3618_361832

def is_valid_set (S : Set ℤ) (a b : ℤ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ ∀ x y z, x ∈ S → y ∈ S → z ∈ S → (x + y + z) ∈ S

theorem all_greater_than_2ab_in_S
  (a b : ℤ)
  (ha : a > 0)
  (hb : b > 0)
  (hnot_both_one : ¬(a = 1 ∧ b = 1))
  (hcoprime : Nat.gcd a.natAbs b.natAbs = 1)
  (S : Set ℤ)
  (hS : is_valid_set S a b) :
  ∀ n : ℤ, n > 2 * a * b → n ∈ S :=
sorry

end NUMINAMATH_CALUDE_all_greater_than_2ab_in_S_l3618_361832


namespace NUMINAMATH_CALUDE_anya_lost_games_correct_l3618_361842

/-- Represents a girl in the table tennis game --/
inductive Girl
| Anya
| Bella
| Valya
| Galya
| Dasha

/-- The total number of games played --/
def total_games : ℕ := 19

/-- The number of games each girl played --/
def games_played (g : Girl) : ℕ :=
  match g with
  | Girl.Anya => 4
  | Girl.Bella => 6
  | Girl.Valya => 7
  | Girl.Galya => 10
  | Girl.Dasha => 11

/-- A game is represented by its number and the two girls who played --/
structure Game where
  number : ℕ
  player1 : Girl
  player2 : Girl

/-- The set of all games played --/
def all_games : Set Game := sorry

/-- The set of games where Anya lost --/
def anya_lost_games : Set ℕ := {4, 8, 12, 16}

/-- The main theorem to prove --/
theorem anya_lost_games_correct :
  ∀ (g : Game), g ∈ all_games → 
    (g.player1 = Girl.Anya ∨ g.player2 = Girl.Anya) → 
    g.number ∈ anya_lost_games :=
  sorry

end NUMINAMATH_CALUDE_anya_lost_games_correct_l3618_361842


namespace NUMINAMATH_CALUDE_solution_pairs_l3618_361859

theorem solution_pairs (x y : ℕ+) : 
  let d := Nat.gcd x y
  x * y * d = x + y + d^2 ↔ (x = 2 ∧ y = 2) ∨ (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l3618_361859


namespace NUMINAMATH_CALUDE_abc_sum_product_bound_l3618_361819

theorem abc_sum_product_bound (a b c : ℝ) (h : a + b + c = 3) :
  ∃ (M : ℝ), ∀ (x : ℝ), x ≤ M ∧ (∃ (a' b' c' : ℝ), a' + b' + c' = 3 ∧ a' * b' + a' * c' + b' * c' = x) :=
sorry

end NUMINAMATH_CALUDE_abc_sum_product_bound_l3618_361819


namespace NUMINAMATH_CALUDE_cos_BHD_value_l3618_361813

/-- A rectangular solid with specific angle conditions -/
structure RectangularSolid where
  /-- Angle DHG is 30 degrees -/
  angle_DHG : ℝ
  angle_DHG_eq : angle_DHG = 30 * π / 180
  /-- Angle FHB is 45 degrees -/
  angle_FHB : ℝ
  angle_FHB_eq : angle_FHB = 45 * π / 180

/-- The cosine of angle BHD in the rectangular solid -/
def cos_BHD (solid : RectangularSolid) : ℝ := sorry

/-- Theorem stating that the cosine of angle BHD is 5√2/12 -/
theorem cos_BHD_value (solid : RectangularSolid) : 
  cos_BHD solid = 5 * Real.sqrt 2 / 12 := by sorry

end NUMINAMATH_CALUDE_cos_BHD_value_l3618_361813


namespace NUMINAMATH_CALUDE_complex_geometric_sequence_l3618_361807

theorem complex_geometric_sequence (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 2*a + 2*Complex.I
  let z₃ : ℂ := 3*a + 4*Complex.I
  (∃ r : ℝ, r > 0 ∧ Complex.abs z₂ = r * Complex.abs z₁ ∧ Complex.abs z₃ = r * Complex.abs z₂) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_geometric_sequence_l3618_361807


namespace NUMINAMATH_CALUDE_equilibrium_constant_is_20_l3618_361839

/-- The equilibrium constant for the reaction NH₄I(s) ⇌ NH₃(g) + HI(g) -/
def equilibrium_constant (h2_conc : ℝ) (hi_conc : ℝ) : ℝ :=
  let hi_from_nh4i := hi_conc + 2 * h2_conc
  hi_from_nh4i * hi_conc

/-- Theorem stating that the equilibrium constant is 20 (mol/L)² under given conditions -/
theorem equilibrium_constant_is_20 (h2_conc : ℝ) (hi_conc : ℝ)
  (h2_eq : h2_conc = 0.5)
  (hi_eq : hi_conc = 4) :
  equilibrium_constant h2_conc hi_conc = 20 := by
  sorry

end NUMINAMATH_CALUDE_equilibrium_constant_is_20_l3618_361839


namespace NUMINAMATH_CALUDE_M_intersect_N_l3618_361818

-- Define set M
def M : Set ℝ := {0, 1, 2}

-- Define set N
def N : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}

-- Theorem statement
theorem M_intersect_N : M ∩ N = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_l3618_361818


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l3618_361838

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem: The man's speed against the current is 20 km/hr given the conditions -/
theorem mans_speed_against_current :
  speed_against_current 25 2.5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l3618_361838


namespace NUMINAMATH_CALUDE_divisibility_condition_l3618_361817

theorem divisibility_condition (x y : ℕ+) :
  (∃ k : ℤ, (2 * x * y^2 - y^3 + 1 : ℤ) = k * x^2) ↔
  (∃ t : ℕ+, (x = 2 * t ∧ y = 1) ∨
             (x = t ∧ y = 2 * t) ∨
             (x = 8 * t^4 - t ∧ y = 2 * t)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3618_361817


namespace NUMINAMATH_CALUDE_sum_in_base_b_l3618_361806

/-- Given a base b, this function converts a number from base b to base 10 --/
def toBase10 (b : ℕ) (x : ℕ) : ℕ := sorry

/-- Given a base b, this function converts a number from base 10 to base b --/
def fromBase10 (b : ℕ) (x : ℕ) : ℕ := sorry

/-- The product of 12, 15, and 16 in base b --/
def product (b : ℕ) : ℕ := toBase10 b 12 * toBase10 b 15 * toBase10 b 16

/-- The sum of 12, 15, and 16 in base b --/
def sum (b : ℕ) : ℕ := toBase10 b 12 + toBase10 b 15 + toBase10 b 16

theorem sum_in_base_b (b : ℕ) :
  (product b = toBase10 b 3146) → (fromBase10 b (sum b) = 44) := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base_b_l3618_361806


namespace NUMINAMATH_CALUDE_exists_real_less_than_negative_one_l3618_361830

theorem exists_real_less_than_negative_one : ∃ x : ℝ, x < -1 := by
  sorry

end NUMINAMATH_CALUDE_exists_real_less_than_negative_one_l3618_361830


namespace NUMINAMATH_CALUDE_infinitely_many_numbers_with_property_l3618_361888

/-- A function that returns the number of divisors of a natural number -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- A function that returns the product of prime factors of a natural number -/
def prodPrimeFactors (n : ℕ) : ℕ := sorry

/-- A function that returns the product of exponents in the prime factorization of a natural number -/
def prodExponents (n : ℕ) : ℕ := sorry

/-- The property that we want to prove holds for infinitely many natural numbers -/
def hasProperty (n : ℕ) : Prop :=
  numDivisors n = prodPrimeFactors n - prodExponents n

theorem infinitely_many_numbers_with_property :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, hasProperty n := by sorry

end NUMINAMATH_CALUDE_infinitely_many_numbers_with_property_l3618_361888


namespace NUMINAMATH_CALUDE_sequence_equality_l3618_361831

-- Define the sequence a_n
def a (n : ℕ) (x : ℝ) : ℝ := 1 + x^(n+1) + x^(n+2)

-- State the theorem
theorem sequence_equality (x : ℝ) (h : (a 2 x)^2 = (a 1 x) * (a 3 x)) :
  ∀ n ≥ 3, (a n x)^2 = (a (n-1) x) * (a (n+1) x) :=
by sorry

end NUMINAMATH_CALUDE_sequence_equality_l3618_361831


namespace NUMINAMATH_CALUDE_sin_2x_value_l3618_361833

theorem sin_2x_value (x : ℝ) (h : Real.sin (x + π/4) = 3/5) : 
  Real.sin (2*x) = 8*Real.sqrt 2/25 := by
sorry

end NUMINAMATH_CALUDE_sin_2x_value_l3618_361833


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3618_361824

theorem decimal_to_fraction :
  (3.68 : ℚ) = 92 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3618_361824


namespace NUMINAMATH_CALUDE_quadratic_positive_combination_l3618_361897

/-- A quadratic function is a function of the form ax² + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Two intervals are disjoint if they have no common points -/
def DisjointIntervals (I J : Set ℝ) : Prop :=
  I ∩ J = ∅

/-- A function is negative on an interval if it takes negative values for all points in that interval -/
def NegativeOnInterval (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, f x < 0

theorem quadratic_positive_combination
  (f g : ℝ → ℝ)
  (hf : IsQuadratic f)
  (hg : IsQuadratic g)
  (hfI : ∃ I : Set ℝ, NegativeOnInterval f I)
  (hgJ : ∃ J : Set ℝ, NegativeOnInterval g J)
  (hIJ : ∀ I J, (NegativeOnInterval f I ∧ NegativeOnInterval g J) → DisjointIntervals I J) :
  ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ ∀ x, α * f x + β * g x > 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_positive_combination_l3618_361897


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3618_361867

/-- The number of games played in a round-robin chess tournament. -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 21 participants, where each participant
    plays exactly one game with each of the remaining participants, 
    the total number of games played is 210. -/
theorem chess_tournament_games :
  num_games 21 = 210 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3618_361867


namespace NUMINAMATH_CALUDE_binary_101111_is_47_l3618_361865

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101111_is_47 :
  binary_to_decimal [true, true, true, true, true, false] = 47 := by
  sorry

end NUMINAMATH_CALUDE_binary_101111_is_47_l3618_361865


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3618_361861

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 3 → (|(3 * x + 2) / (x - 3)| < 4 ↔ (10/7 < x ∧ x < 3) ∨ (3 < x ∧ x < 14)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3618_361861


namespace NUMINAMATH_CALUDE_mans_running_speed_l3618_361802

/-- A proof that calculates a man's running speed given his walking speed and times. -/
theorem mans_running_speed (walking_speed : ℝ) (walking_time : ℝ) (running_time : ℝ) :
  walking_speed = 8 →
  walking_time = 3 →
  running_time = 1 →
  walking_speed * walking_time / running_time = 24 := by
  sorry

#check mans_running_speed

end NUMINAMATH_CALUDE_mans_running_speed_l3618_361802


namespace NUMINAMATH_CALUDE_largest_square_tile_l3618_361873

theorem largest_square_tile (board_length board_width tile_size : ℕ) : 
  board_length = 16 →
  board_width = 24 →
  tile_size = Nat.gcd board_length board_width →
  tile_size = 8 := by
sorry

end NUMINAMATH_CALUDE_largest_square_tile_l3618_361873


namespace NUMINAMATH_CALUDE_only_set_B_forms_triangle_l3618_361879

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def set_A : List ℝ := [2, 6, 8]
def set_B : List ℝ := [4, 6, 7]
def set_C : List ℝ := [5, 6, 12]
def set_D : List ℝ := [2, 3, 6]

theorem only_set_B_forms_triangle :
  (¬ triangle_inequality set_A[0] set_A[1] set_A[2]) ∧
  (triangle_inequality set_B[0] set_B[1] set_B[2]) ∧
  (¬ triangle_inequality set_C[0] set_C[1] set_C[2]) ∧
  (¬ triangle_inequality set_D[0] set_D[1] set_D[2]) :=
by sorry

end NUMINAMATH_CALUDE_only_set_B_forms_triangle_l3618_361879


namespace NUMINAMATH_CALUDE_counterexample_existence_l3618_361821

theorem counterexample_existence : ∃ (S : Finset ℝ), 
  (Finset.card S = 25) ∧ 
  (∀ (a b c : ℝ), a ∈ S → b ∈ S → c ∈ S → 
    ∃ (d : ℝ), d ∈ S ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ a + b + c + d > 0) ∧
  (Finset.sum S id ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_existence_l3618_361821


namespace NUMINAMATH_CALUDE_value_added_to_fraction_l3618_361887

theorem value_added_to_fraction : ∀ (N V : ℝ),
  N = 8 →
  0.75 * N + V = 8 →
  V = 2 := by
sorry

end NUMINAMATH_CALUDE_value_added_to_fraction_l3618_361887


namespace NUMINAMATH_CALUDE_union_equality_implies_a_values_l3618_361878

theorem union_equality_implies_a_values (a : ℝ) : 
  ({1, a} : Set ℝ) ∪ {a^2} = {1, a} → a = -1 ∨ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_union_equality_implies_a_values_l3618_361878


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3618_361884

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3618_361884


namespace NUMINAMATH_CALUDE_probability_all_selected_l3618_361895

theorem probability_all_selected (p_ram p_ravi p_raj : ℚ) 
  (h_ram : p_ram = 2/7)
  (h_ravi : p_ravi = 1/5)
  (h_raj : p_raj = 3/8) :
  p_ram * p_ravi * p_raj = 3/140 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_selected_l3618_361895


namespace NUMINAMATH_CALUDE_hyperbola_ratio_l3618_361870

/-- The ratio of a to b for a hyperbola with equation x²/a² - y²/b² = 1 and asymptote angle 45° --/
theorem hyperbola_ratio (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.pi / 4 = Real.arctan ((2 * b / a) / (1 - b^2 / a^2))) →
  a / b = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_ratio_l3618_361870


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l3618_361874

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

-- State the theorem
theorem derivative_f_at_1 : 
  deriv f 1 = 24 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l3618_361874


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3618_361876

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) → 
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 770 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l3618_361876


namespace NUMINAMATH_CALUDE_heart_then_club_probability_l3618_361881

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def numHearts : ℕ := 13

/-- Number of clubs in a standard deck -/
def numClubs : ℕ := 13

/-- Probability of drawing a heart followed by a club from a standard deck -/
def probHeartThenClub : ℚ := numHearts / standardDeck * numClubs / (standardDeck - 1)

theorem heart_then_club_probability :
  probHeartThenClub = 13 / 204 := by sorry

end NUMINAMATH_CALUDE_heart_then_club_probability_l3618_361881


namespace NUMINAMATH_CALUDE_max_product_logarithms_l3618_361843

/-- Given a, b, c > 1 satisfying the given equations, the maximum value of lg a · lg c is 16/3 -/
theorem max_product_logarithms (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq1 : Real.log a / Real.log 10 + Real.log c / Real.log b = 3)
  (eq2 : Real.log b / Real.log 10 + Real.log c / Real.log a = 4) :
  (Real.log a / Real.log 10) * (Real.log c / Real.log 10) ≤ 16/3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_logarithms_l3618_361843


namespace NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l3618_361862

/-- The smallest positive two-digit multiple of 8 -/
def smallest_multiple : ℕ := 16

/-- The largest positive two-digit multiple of 8 -/
def largest_multiple : ℕ := 96

/-- The count of positive two-digit multiples of 8 -/
def count_multiples : ℕ := 11

/-- The sum of all positive two-digit multiples of 8 -/
def sum_multiples : ℕ := 616

/-- Theorem stating that the arithmetic mean of all positive two-digit multiples of 8 is 56 -/
theorem arithmetic_mean_two_digit_multiples_of_8 :
  (sum_multiples : ℚ) / count_multiples = 56 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_two_digit_multiples_of_8_l3618_361862


namespace NUMINAMATH_CALUDE_infinite_capacitor_chain_effective_capacitance_l3618_361866

/-- Given an infinitely long chain of capacitors, each with capacitance C,
    the effective capacitance Ce between any two adjacent points
    is equal to ((1 + √3) * C) / 2. -/
theorem infinite_capacitor_chain_effective_capacitance (C : ℝ) (Ce : ℝ) 
  (h1 : C > 0) -- Capacitance is always positive
  (h2 : Ce = C + Ce / (2 + Ce / C)) -- Relationship derived from the infinite chain
  : Ce = ((1 + Real.sqrt 3) * C) / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_capacitor_chain_effective_capacitance_l3618_361866


namespace NUMINAMATH_CALUDE_like_terms_exponent_product_l3618_361815

/-- Given two algebraic terms are like terms, prove that the product of their exponents is 6 -/
theorem like_terms_exponent_product (x y : ℝ) (m n : ℕ) :
  (∃ (a b : ℝ), a * x^3 * y^n = b * x^m * y^2) → m * n = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_exponent_product_l3618_361815


namespace NUMINAMATH_CALUDE_no_three_primes_sum_squares_l3618_361828

theorem no_three_primes_sum_squares : ¬∃ (p q r : ℕ), 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  ∃ (a b c : ℕ), p + q = a^2 ∧ p + r = b^2 ∧ q + r = c^2 :=
by sorry

end NUMINAMATH_CALUDE_no_three_primes_sum_squares_l3618_361828


namespace NUMINAMATH_CALUDE_fish_problem_l3618_361880

/-- The number of fish originally in the shop -/
def original_fish : ℕ := 36

/-- The number of fish remaining after lunch sale -/
def after_lunch (f : ℕ) : ℕ := f / 2

/-- The number of fish sold for dinner -/
def dinner_sale (f : ℕ) : ℕ := (after_lunch f) / 3

/-- The number of fish remaining after both sales -/
def remaining_fish (f : ℕ) : ℕ := (after_lunch f) - (dinner_sale f)

theorem fish_problem :
  remaining_fish original_fish = 12 :=
by sorry

end NUMINAMATH_CALUDE_fish_problem_l3618_361880


namespace NUMINAMATH_CALUDE_john_average_bottle_price_l3618_361835

/-- The average price of bottles purchased by John -/
def average_price (large_quantity : ℕ) (large_price : ℚ) (small_quantity : ℕ) (small_price : ℚ) : ℚ :=
  (large_quantity * large_price + small_quantity * small_price) / (large_quantity + small_quantity)

/-- The average price of bottles purchased by John is approximately $1.70 -/
theorem john_average_bottle_price :
  let large_quantity : ℕ := 1300
  let large_price : ℚ := 189/100
  let small_quantity : ℕ := 750
  let small_price : ℚ := 138/100
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
    |average_price large_quantity large_price small_quantity small_price - 17/10| < ε :=
sorry

end NUMINAMATH_CALUDE_john_average_bottle_price_l3618_361835


namespace NUMINAMATH_CALUDE_long_division_problem_l3618_361872

theorem long_division_problem (quotient remainder divisor dividend : ℕ) : 
  quotient = 2015 → 
  remainder = 0 → 
  divisor = 105 → 
  dividend = quotient * divisor + remainder → 
  dividend = 20685 := by
sorry

end NUMINAMATH_CALUDE_long_division_problem_l3618_361872


namespace NUMINAMATH_CALUDE_john_burritos_per_day_l3618_361845

theorem john_burritos_per_day 
  (boxes : ℕ) 
  (burritos_per_box : ℕ) 
  (days : ℕ) 
  (remaining : ℕ) 
  (h1 : boxes = 3) 
  (h2 : burritos_per_box = 20) 
  (h3 : days = 10) 
  (h4 : remaining = 10) : 
  (boxes * burritos_per_box - boxes * burritos_per_box / 3 - remaining) / days = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_burritos_per_day_l3618_361845


namespace NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l3618_361814

theorem abs_sum_reciprocals_ge_two (a b : ℝ) (h : a * b ≠ 0) :
  |a / b + b / a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l3618_361814


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l3618_361882

theorem sqrt_sum_fractions : 
  Real.sqrt (4/25 + 9/49) = Real.sqrt 421 / 35 := by sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l3618_361882


namespace NUMINAMATH_CALUDE_problem_solution_l3618_361826

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem problem_solution (x y a : ℕ) : 
  x > 0 → y > 0 → a > 0 → 
  x * y = 32 → 
  sum_of_digits ((10 ^ x) ^ a - 64) = 279 → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3618_361826


namespace NUMINAMATH_CALUDE_log_ratio_squared_l3618_361822

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h2 : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_squared_l3618_361822


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l3618_361855

/-- Given a triangle PQR with angles 3x, x, and 6x, prove that the largest angle is 108° -/
theorem largest_angle_in_triangle (x : ℝ) : 
  x > 0 ∧ 3*x + x + 6*x = 180 → 
  max (3*x) (max x (6*x)) = 108 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l3618_361855


namespace NUMINAMATH_CALUDE_frame_interior_perimeter_l3618_361820

theorem frame_interior_perimeter
  (frame_width : ℝ)
  (frame_area : ℝ)
  (outer_edge : ℝ)
  (h1 : frame_width = 2)
  (h2 : frame_area = 60)
  (h3 : outer_edge = 10) :
  let inner_length := outer_edge - 2 * frame_width
  let inner_width := (frame_area / (outer_edge - inner_length)) - frame_width
  inner_length * 2 + inner_width * 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_frame_interior_perimeter_l3618_361820


namespace NUMINAMATH_CALUDE_fresh_fruit_water_percentage_l3618_361869

theorem fresh_fruit_water_percentage
  (fresh_weight : ℝ)
  (dried_weight : ℝ)
  (dried_water_percentage : ℝ)
  (h1 : fresh_weight = 50)
  (h2 : dried_weight = 5)
  (h3 : dried_water_percentage = 20)
  : (fresh_weight - dried_weight * (1 - dried_water_percentage / 100)) / fresh_weight * 100 = 92 := by
  sorry

end NUMINAMATH_CALUDE_fresh_fruit_water_percentage_l3618_361869


namespace NUMINAMATH_CALUDE_sunglasses_sold_l3618_361863

/-- Proves that the number of pairs of sunglasses sold is 10 -/
theorem sunglasses_sold (selling_price cost_price sign_cost : ℕ) 
  (h1 : selling_price = 30)
  (h2 : cost_price = 26)
  (h3 : sign_cost = 20) :
  (sign_cost * 2) / (selling_price - cost_price) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_sold_l3618_361863


namespace NUMINAMATH_CALUDE_median_interval_is_65_to_69_l3618_361805

/-- Represents a score interval with its lower and upper bounds -/
structure ScoreInterval where
  lower : ℕ
  upper : ℕ

/-- Represents the distribution of scores -/
structure ScoreDistribution where
  intervals : List ScoreInterval
  counts : List ℕ

/-- Finds the interval containing the median score -/
def findMedianInterval (dist : ScoreDistribution) : Option ScoreInterval :=
  sorry

/-- The given score distribution -/
def testScoreDistribution : ScoreDistribution :=
  { intervals := [
      { lower := 50, upper := 54 },
      { lower := 55, upper := 59 },
      { lower := 60, upper := 64 },
      { lower := 65, upper := 69 },
      { lower := 70, upper := 74 }
    ],
    counts := [10, 15, 25, 30, 20]
  }

/-- Theorem: The median score interval for the given distribution is 65-69 -/
theorem median_interval_is_65_to_69 :
  findMedianInterval testScoreDistribution = some { lower := 65, upper := 69 } :=
  sorry

end NUMINAMATH_CALUDE_median_interval_is_65_to_69_l3618_361805


namespace NUMINAMATH_CALUDE_smallest_a_value_l3618_361844

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.cos (a * ↑x + b) = Real.cos (31 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.cos (a' * ↑x + b) = Real.cos (31 * ↑x)) → a' ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3618_361844


namespace NUMINAMATH_CALUDE_characterize_satisfying_polynomials_l3618_361898

/-- A polynomial satisfying the given inequality. -/
structure SatisfyingPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  h_c : |c| ≤ 1
  h_ab : (|a| = 1 ∧ b = 0) ∨ (|a| < 1 ∧ |b| ≤ 2 * Real.sqrt (1 + a * c - |a + c|))

/-- The main theorem statement. -/
theorem characterize_satisfying_polynomials :
  ∀ (P : ℝ → ℝ), (∀ x : ℝ, |P x - x| ≤ x^2 + 1) ↔
    ∃ (p : SatisfyingPolynomial), ∀ x : ℝ, P x = p.a * x^2 + (p.b + 1) * x + p.c :=
sorry

end NUMINAMATH_CALUDE_characterize_satisfying_polynomials_l3618_361898


namespace NUMINAMATH_CALUDE_correct_allocation_count_l3618_361885

def num_volunteers : ℕ := 4
def num_events : ℕ := 3

def allocation_schemes (n_volunteers : ℕ) (n_events : ℕ) : ℕ :=
  if n_volunteers < n_events then 0
  else (n_events.factorial * n_events^(n_volunteers - n_events))

theorem correct_allocation_count :
  allocation_schemes num_volunteers num_events = 18 :=
sorry

end NUMINAMATH_CALUDE_correct_allocation_count_l3618_361885


namespace NUMINAMATH_CALUDE_pet_store_ratios_l3618_361854

/-- Given the ratios of cats to dogs and dogs to parrots, and the number of cats,
    this theorem proves the number of dogs and parrots. -/
theorem pet_store_ratios (cats : ℕ) (dogs : ℕ) (parrots : ℕ) : 
  (3 : ℚ) / 4 = cats / dogs →  -- ratio of cats to dogs
  (2 : ℚ) / 5 = dogs / parrots →  -- ratio of dogs to parrots
  cats = 18 →  -- number of cats
  dogs = 24 ∧ parrots = 60 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_ratios_l3618_361854
