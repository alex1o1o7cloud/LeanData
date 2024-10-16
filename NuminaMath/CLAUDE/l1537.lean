import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l1537_153733

def A : Set ℝ := {x | (2*x - 2)/(x + 1) < 1}

def B (a : ℝ) : Set ℝ := {x | x^2 + x + a - a^2 < 0}

theorem problem_solution :
  (∀ x, x ∈ (B 1 ∪ (Set.univ \ A)) ↔ (x < 0 ∨ x ≥ 3)) ∧
  (∀ a, A = B a ↔ a ∈ Set.Iic (-3) ∪ Set.Ici 4) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1537_153733


namespace NUMINAMATH_CALUDE_art_museum_picture_distribution_l1537_153747

theorem art_museum_picture_distribution (total_pictures : ℕ) (num_exhibits : ℕ) : 
  total_pictures = 154 → num_exhibits = 9 → 
  (∃ (additional_pictures : ℕ), 
    (total_pictures + additional_pictures) % num_exhibits = 0 ∧
    additional_pictures = 8) := by
  sorry

end NUMINAMATH_CALUDE_art_museum_picture_distribution_l1537_153747


namespace NUMINAMATH_CALUDE_sine_inequality_l1537_153744

theorem sine_inequality (y : Real) :
  (y ∈ Set.Icc 0 (Real.pi / 2)) ↔
  (∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), Real.sin (x + y) ≤ Real.sin x + Real.sin y) :=
by sorry

end NUMINAMATH_CALUDE_sine_inequality_l1537_153744


namespace NUMINAMATH_CALUDE_largest_prime_sum_l1537_153708

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the digits of a natural number -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if a list contains exactly the digits 1 to 9 -/
def usesAllDigits (l : List ℕ) : Prop := sorry

theorem largest_prime_sum :
  ∀ (p₁ p₂ p₃ p₄ : ℕ),
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ →
    usesAllDigits (digits p₁ ++ digits p₂ ++ digits p₃ ++ digits p₄) →
    p₁ + p₂ + p₃ + p₄ ≤ 1798 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_sum_l1537_153708


namespace NUMINAMATH_CALUDE_car_journey_speed_l1537_153716

/-- Represents the average speed between two towns -/
structure AverageSpeed where
  value : ℝ
  unit : String

/-- Represents the distance between two towns -/
structure Distance where
  value : ℝ
  unit : String

/-- Theorem: Given the conditions of the car journey, prove that the average speed from Town C to Town D is 36 mph -/
theorem car_journey_speed (d_ab d_bc d_cd : Distance) (s_ab s_bc s_total : AverageSpeed) :
  d_ab.value = 120 ∧ d_ab.unit = "miles" →
  d_bc.value = 60 ∧ d_bc.unit = "miles" →
  d_cd.value = 90 ∧ d_cd.unit = "miles" →
  s_ab.value = 40 ∧ s_ab.unit = "mph" →
  s_bc.value = 30 ∧ s_bc.unit = "mph" →
  s_total.value = 36 ∧ s_total.unit = "mph" →
  ∃ (s_cd : AverageSpeed), s_cd.value = 36 ∧ s_cd.unit = "mph" := by
  sorry


end NUMINAMATH_CALUDE_car_journey_speed_l1537_153716


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1537_153724

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 - 2 * a^2 + 5 * a + 15 = 0) →
  (3 * b^3 - 2 * b^2 + 5 * b + 15 = 0) →
  (3 * c^3 - 2 * c^2 + 5 * c + 15 = 0) →
  a^2 + b^2 + c^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1537_153724


namespace NUMINAMATH_CALUDE_twenty_nine_impossible_l1537_153720

/-- Represents the score for a test with 10 questions. -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  sum_is_ten : correct + unanswered + incorrect = 10

/-- Calculates the total score for a given TestScore. -/
def totalScore (ts : TestScore) : Nat :=
  3 * ts.correct + ts.unanswered

/-- Theorem stating that 29 is not a possible total score. -/
theorem twenty_nine_impossible : ¬∃ (ts : TestScore), totalScore ts = 29 := by
  sorry

end NUMINAMATH_CALUDE_twenty_nine_impossible_l1537_153720


namespace NUMINAMATH_CALUDE_same_result_as_five_minus_seven_l1537_153789

theorem same_result_as_five_minus_seven : 5 - 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_same_result_as_five_minus_seven_l1537_153789


namespace NUMINAMATH_CALUDE_system_equations_properties_l1537_153713

/-- Given a system of equations with parameters x, y, and m, prove properties about the solution and a related expression. -/
theorem system_equations_properties (x y m : ℝ) (h1 : 3 * x + 2 * y = m + 2) (h2 : 2 * x + y = m - 1)
  (hx : x > 0) (hy : y > 0) :
  (x = m - 4 ∧ y = 7 - m) ∧
  (4 < m ∧ m < 7) ∧
  (∀ (m : ℕ), 4 < m → m < 7 → (2 * x - 3 * y + m) ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_system_equations_properties_l1537_153713


namespace NUMINAMATH_CALUDE_polynomial_derivative_sum_l1537_153743

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_derivative_sum_l1537_153743


namespace NUMINAMATH_CALUDE_sqrt_of_square_root_7_minus_3_squared_l1537_153735

theorem sqrt_of_square_root_7_minus_3_squared (x : ℝ) :
  Real.sqrt ((Real.sqrt 7 - 3) ^ 2) = 3 - Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_of_square_root_7_minus_3_squared_l1537_153735


namespace NUMINAMATH_CALUDE_f_bounds_in_R_f_attains_bounds_l1537_153756

/-- The triangular region R with vertices A(4,1), B(-1,-6), C(-3,2) -/
def R : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = (4*a - b - 3*c, a - 6*b + 2*c)}

/-- The function to be maximized and minimized -/
def f (p : ℝ × ℝ) : ℝ := 4 * p.1 - 3 * p.2

theorem f_bounds_in_R :
  ∀ p ∈ R, -18 ≤ f p ∧ f p ≤ 14 :=
by sorry

theorem f_attains_bounds :
  (∃ p ∈ R, f p = -18) ∧ (∃ p ∈ R, f p = 14) :=
by sorry

end NUMINAMATH_CALUDE_f_bounds_in_R_f_attains_bounds_l1537_153756


namespace NUMINAMATH_CALUDE_hendrix_class_size_l1537_153748

theorem hendrix_class_size (initial_students : ℕ) (new_students : ℕ) (transfer_fraction : ℚ) : 
  initial_students = 160 → 
  new_students = 20 → 
  transfer_fraction = 1/3 →
  (initial_students + new_students) - ((initial_students + new_students : ℚ) * transfer_fraction).floor = 120 := by
  sorry

end NUMINAMATH_CALUDE_hendrix_class_size_l1537_153748


namespace NUMINAMATH_CALUDE_number_of_factors_M_l1537_153759

/-- The number of natural-number factors of M, where M = 2^6 · 3^2 · 5^3 · 11^1 -/
theorem number_of_factors_M : 
  let M := 2^6 * 3^2 * 5^3 * 11^1
  (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 168 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_M_l1537_153759


namespace NUMINAMATH_CALUDE_cube_root_of_sum_l1537_153773

theorem cube_root_of_sum (a b : ℝ) : 
  (2*a + 1) + (2*a - 5) = 0 → 
  b^(1/3 : ℝ) = 2 → 
  (a + b)^(1/3 : ℝ) = 9^(1/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_sum_l1537_153773


namespace NUMINAMATH_CALUDE_vowel_word_count_l1537_153753

def vowel_count : ℕ := 5
def word_length : ℕ := 5
def max_vowel_occurrence : ℕ := 3

def total_distributions : ℕ := Nat.choose (word_length + vowel_count - 1) (vowel_count - 1)

def invalid_distributions : ℕ := vowel_count * (vowel_count - 1)

theorem vowel_word_count :
  total_distributions - invalid_distributions = 106 :=
sorry

end NUMINAMATH_CALUDE_vowel_word_count_l1537_153753


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1537_153766

theorem absolute_value_inequality (x : ℝ) :
  abs (x - 2) + abs (x + 3) < 8 ↔ x ∈ Set.Ioo (-6.5) 3.5 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1537_153766


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1537_153796

/-- Given a moving straight line ax + by + c - 2 = 0 where a > 0, c > 0,
    that always passes through point (1, m), and the maximum distance
    from (4, 0) to the line is 3, the minimum value of 1/(2a) + 2/c is 9/4. -/
theorem min_value_of_expression (a b c m : ℝ) : 
  a > 0 → c > 0 → 
  (∀ x y, a * x + b * y + c - 2 = 0 → x = 1 → y = m) →
  (∃ x y, a * x + b * y + c - 2 = 0 ∧ 
    Real.sqrt ((x - 4)^2 + y^2) = 3) →
  (∀ x y, a * x + b * y + c - 2 = 0 → 
    Real.sqrt ((x - 4)^2 + y^2) ≤ 3) →
  (1 / (2 * a) + 2 / c) ≥ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1537_153796


namespace NUMINAMATH_CALUDE_homework_time_ratio_l1537_153701

theorem homework_time_ratio :
  ∀ (geog_time : ℝ) (sci_time : ℝ),
    geog_time > 0 →
    sci_time = (60 + geog_time) / 2 →
    60 + geog_time + sci_time = 135 →
    geog_time / 60 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_homework_time_ratio_l1537_153701


namespace NUMINAMATH_CALUDE_problem_solution_l1537_153799

theorem problem_solution :
  ∀ (a b c : ℕ+) (x y z : ℤ),
    x = -2272 →
    y = 1000 + 100 * c.val + 10 * b.val + a.val →
    z = 1 →
    a.val * x + b.val * y + c.val * z = 1 →
    a < b →
    b < c →
    y = 1987 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1537_153799


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1537_153792

def U : Set Nat := {0, 1, 3, 5, 6, 8}
def A : Set Nat := {1, 5, 8}
def B : Set Nat := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1537_153792


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l1537_153729

theorem three_digit_number_proof :
  ∃! x : ℕ,
    (100 ≤ x ∧ x < 1000) ∧
    (x * (x / 100) = 494) ∧
    (x * ((x / 10) % 10) = 988) ∧
    (x * (x % 10) = 1729) ∧
    x = 247 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l1537_153729


namespace NUMINAMATH_CALUDE_increasing_sequences_with_divisibility_property_l1537_153739

theorem increasing_sequences_with_divisibility_property :
  ∃ (a b : ℕ → ℕ), 
    (∀ n : ℕ, a n < a (n + 1)) ∧ 
    (∀ n : ℕ, b n < b (n + 1)) ∧
    (∀ n : ℕ, (a n * (a n + 1)) ∣ (b n ^ 2 + 1)) :=
by
  let a : ℕ → ℕ := λ n => (2^(2*n) + 1)^2
  let b : ℕ → ℕ := λ n => 2^(n*(2^(2*n) + 1)) + (2^(2*n) + 1)^2 * (2^(n*(2^(2*n)+1)) - (2^(2*n) + 1))
  sorry

end NUMINAMATH_CALUDE_increasing_sequences_with_divisibility_property_l1537_153739


namespace NUMINAMATH_CALUDE_exists_alternating_coloring_l1537_153785

-- Define an ordered set
variable {X : Type*} [PartialOrder X]

-- Define a coloring function
def Coloring (X : Type*) := X → Bool

-- Theorem statement
theorem exists_alternating_coloring :
  ∃ (f : Coloring X), ∀ (x y : X), x < y → f x = f y →
    ∃ (z : X), x < z ∧ z < y ∧ f z ≠ f x := by
  sorry

end NUMINAMATH_CALUDE_exists_alternating_coloring_l1537_153785


namespace NUMINAMATH_CALUDE_decimal_point_problem_l1537_153760

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 / x) : x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l1537_153760


namespace NUMINAMATH_CALUDE_basketball_scores_l1537_153705

/-- The number of different total point scores for a basketball player who made 7 baskets,
    each worth either 2 or 3 points. -/
def differentScores : ℕ := by sorry

theorem basketball_scores :
  let totalBaskets : ℕ := 7
  let twoPointValue : ℕ := 2
  let threePointValue : ℕ := 3
  differentScores = 8 := by sorry

end NUMINAMATH_CALUDE_basketball_scores_l1537_153705


namespace NUMINAMATH_CALUDE_bird_nest_twigs_l1537_153790

theorem bird_nest_twigs (circle_twigs : ℕ) (found_fraction : ℚ) (remaining_twigs : ℕ) :
  circle_twigs = 12 →
  found_fraction = 1 / 3 →
  remaining_twigs = 48 →
  (circle_twigs : ℚ) * (1 - found_fraction) * (circle_twigs : ℚ) = (remaining_twigs : ℚ) →
  circle_twigs * found_fraction * (circle_twigs : ℚ) + (remaining_twigs : ℚ) = 18 * (circle_twigs : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_bird_nest_twigs_l1537_153790


namespace NUMINAMATH_CALUDE_expression_evaluation_l1537_153731

theorem expression_evaluation :
  let x : ℝ := Real.sin (30 * π / 180)
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1537_153731


namespace NUMINAMATH_CALUDE_f_satisfies_properties_l1537_153786

def f (x : ℝ) : ℝ := (x - 2)^2

theorem f_satisfies_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧
  (∀ x y, x < y → x < 2 → y < 2 → f x > f y) ∧
  (∀ x y, x < y → x > 2 → y > 2 → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_satisfies_properties_l1537_153786


namespace NUMINAMATH_CALUDE_sum_of_four_integers_l1537_153762

theorem sum_of_four_integers (m n p q : ℕ+) : 
  m ≠ n ∧ m ≠ p ∧ m ≠ q ∧ n ≠ p ∧ n ≠ q ∧ p ≠ q →
  (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4 →
  m + n + p + q = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_integers_l1537_153762


namespace NUMINAMATH_CALUDE_nested_radical_value_l1537_153704

/-- The value of the infinite nested radical √(15 + √(15 + √(15 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (15 + Real.sqrt (15 + Real.sqrt (15 + Real.sqrt (15 + Real.sqrt 15))))

/-- Theorem stating that the nested radical equals 5 -/
theorem nested_radical_value : nestedRadical = 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1537_153704


namespace NUMINAMATH_CALUDE_organize_four_men_five_women_l1537_153798

/-- The number of ways to organize men and women into groups -/
def organize_groups (num_men : ℕ) (num_women : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of ways to organize the groups -/
theorem organize_four_men_five_women :
  organize_groups 4 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_organize_four_men_five_women_l1537_153798


namespace NUMINAMATH_CALUDE_parabola_vertex_l1537_153726

/-- The vertex of the parabola y = 2(x-5)^2 + 3 has coordinates (5, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2*(x - 5)^2 + 3 → (5, 3) = (x, y) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1537_153726


namespace NUMINAMATH_CALUDE_david_min_score_l1537_153732

def david_scores : List Int := [88, 92, 75, 83, 90]

def current_average : Rat :=
  (david_scores.sum : Rat) / david_scores.length

def target_average : Rat := current_average + 4

def min_score : Int :=
  Int.ceil ((target_average * (david_scores.length + 1) : Rat) - david_scores.sum)

theorem david_min_score :
  min_score = 110 := by sorry

end NUMINAMATH_CALUDE_david_min_score_l1537_153732


namespace NUMINAMATH_CALUDE_shekar_average_marks_l1537_153780

def shekar_marks : List ℕ := [76, 65, 82, 67, 75]

theorem shekar_average_marks :
  (shekar_marks.sum : ℚ) / shekar_marks.length = 73 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l1537_153780


namespace NUMINAMATH_CALUDE_min_points_on_circle_l1537_153771

/-- A type representing a point in a plane -/
def Point : Type := ℝ × ℝ

/-- A type representing a circle in a plane -/
def Circle : Type := Point × ℝ

/-- Check if a point lies on a circle -/
def lies_on (p : Point) (c : Circle) : Prop :=
  let (center, radius) := c
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

/-- Check if four points are concyclic (lie on the same circle) -/
def are_concyclic (p1 p2 p3 p4 : Point) : Prop :=
  ∃ c : Circle, lies_on p1 c ∧ lies_on p2 c ∧ lies_on p3 c ∧ lies_on p4 c

/-- Main theorem -/
theorem min_points_on_circle 
  (points : Finset Point) 
  (h_card : points.card = 10)
  (h_concyclic : ∀ (s : Finset Point), s ⊆ points → s.card = 5 → 
    ∃ (t : Finset Point), t ⊆ s ∧ t.card = 4 ∧ 
    ∃ (p1 p2 p3 p4 : Point), p1 ∈ t ∧ p2 ∈ t ∧ p3 ∈ t ∧ p4 ∈ t ∧ 
    are_concyclic p1 p2 p3 p4) : 
  ∃ (c : Circle) (s : Finset Point), s ⊆ points ∧ s.card = 9 ∧ 
  ∀ p ∈ s, lies_on p c :=
sorry

end NUMINAMATH_CALUDE_min_points_on_circle_l1537_153771


namespace NUMINAMATH_CALUDE_other_sales_percentage_l1537_153774

/-- The percentage of sales for notebooks -/
def notebook_sales : ℝ := 42

/-- The percentage of sales for markers -/
def marker_sales : ℝ := 26

/-- The total percentage of sales -/
def total_sales : ℝ := 100

/-- Theorem: The percentage of sales that were not notebooks or markers is 32% -/
theorem other_sales_percentage :
  total_sales - (notebook_sales + marker_sales) = 32 := by sorry

end NUMINAMATH_CALUDE_other_sales_percentage_l1537_153774


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1537_153737

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  b = 6 →
  c = a + 2 →
  c = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1537_153737


namespace NUMINAMATH_CALUDE_complex_number_location_l1537_153764

theorem complex_number_location :
  let z : ℂ := (3 + Complex.I) / (1 + Complex.I)
  (0 < z.re) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1537_153764


namespace NUMINAMATH_CALUDE_probability_one_unit_apart_l1537_153717

/-- A rectangle with dimensions 3 × 2 -/
structure Rectangle :=
  (length : ℕ := 3)
  (width : ℕ := 2)

/-- Evenly spaced points on the perimeter of the rectangle -/
def PerimeterPoints (r : Rectangle) : ℕ := 15

/-- Number of unit intervals on the perimeter -/
def UnitIntervals (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The probability of selecting two points one unit apart -/
def ProbabilityOneUnitApart (r : Rectangle) : ℚ :=
  16 / (PerimeterPoints r).choose 2

theorem probability_one_unit_apart (r : Rectangle) :
  ProbabilityOneUnitApart r = 16 / 105 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_unit_apart_l1537_153717


namespace NUMINAMATH_CALUDE_car_speed_problem_l1537_153752

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 75 →
  average_speed = 82.5 →
  (speed_second_hour + (average_speed * 2 - speed_second_hour)) / 2 = average_speed →
  average_speed * 2 - speed_second_hour = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1537_153752


namespace NUMINAMATH_CALUDE_binomial_variance_specific_case_l1537_153727

-- Define the parameters
def n : ℕ := 10
def p : ℝ := 0.02

-- Define the variance function for a binomial distribution
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Theorem statement
theorem binomial_variance_specific_case :
  binomial_variance n p = 0.196 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_specific_case_l1537_153727


namespace NUMINAMATH_CALUDE_square_difference_minus_difference_l1537_153712

theorem square_difference_minus_difference (a b : ℤ) : 
  ((a + b)^2 - (a - b)^2) - (a - b) = 4*a*b - (a - b) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_minus_difference_l1537_153712


namespace NUMINAMATH_CALUDE_base_conversion_equality_l1537_153750

theorem base_conversion_equality (b : ℝ) : b > 0 → (4 * 5 + 3 = b^2 + 2) → b = Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l1537_153750


namespace NUMINAMATH_CALUDE_octagon_area_eq_1200_l1537_153784

/-- A regular octagon inscribed in a square with perimeter 160 cm,
    where each side of the square is quadrised by the vertices of the octagon -/
structure InscribedOctagon where
  square_perimeter : ℝ
  square_perimeter_eq : square_perimeter = 160
  is_regular : Bool
  is_inscribed : Bool
  sides_quadrised : Bool

/-- The area of the inscribed octagon -/
def octagon_area (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating that the area of the inscribed octagon is 1200 square centimeters -/
theorem octagon_area_eq_1200 (o : InscribedOctagon) :
  o.is_regular ∧ o.is_inscribed ∧ o.sides_quadrised → octagon_area o = 1200 := by sorry

end NUMINAMATH_CALUDE_octagon_area_eq_1200_l1537_153784


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l1537_153765

/- Define a function to convert binary to decimal -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/- Define the binary number 101 -/
def binary_101 : List Bool := [true, false, true]

/- Theorem statement -/
theorem binary_101_equals_5 :
  binary_to_decimal binary_101 = 5 := by sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l1537_153765


namespace NUMINAMATH_CALUDE_change_in_average_l1537_153781

def scores : List ℝ := [89, 85, 91, 87, 82]

theorem change_in_average (scores : List ℝ) : 
  scores = [89, 85, 91, 87, 82] →
  (scores.sum / scores.length) - ((scores.take 4).sum / 4) = -1.2 := by
  sorry

end NUMINAMATH_CALUDE_change_in_average_l1537_153781


namespace NUMINAMATH_CALUDE_target_hit_probability_l1537_153715

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1537_153715


namespace NUMINAMATH_CALUDE_unique_function_theorem_l1537_153740

-- Define the property that the function must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y + 1) = x + y + 1

-- State the theorem
theorem unique_function_theorem :
  ∃! f : ℝ → ℝ, SatisfiesProperty f ∧ ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l1537_153740


namespace NUMINAMATH_CALUDE_factor_sum_l1537_153775

theorem factor_sum (x y : ℝ) (a b c d e f g : ℤ) :
  16 * x^8 - 256 * y^4 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x^2 + g*y^2) →
  a + b + c + d + e + f + g = 7 := by
  sorry

end NUMINAMATH_CALUDE_factor_sum_l1537_153775


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l1537_153772

theorem polynomial_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + c * x + 7
  (f (-2011) = -17) → (f 2011 = 31) := by
sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l1537_153772


namespace NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l1537_153763

theorem range_of_a_for_quadratic_inequality :
  ∃ (a : ℝ), ∀ (x : ℝ), x^2 + 2*x + a > 0 ↔ a ∈ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_quadratic_inequality_l1537_153763


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1537_153797

theorem isosceles_triangle_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h_equilateral : equilateral_perimeter = 60) 
  (h_isosceles : isosceles_perimeter = 50) 
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) : 
  isosceles_base = 10 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1537_153797


namespace NUMINAMATH_CALUDE_system_solution_l1537_153738

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x - a * y + a^2 * z = a^3)
  (eq2 : x - b * y + b^2 * z = b^3)
  (eq3 : x - c * y + c^2 * z = c^3)
  (hx : x = a * b * c)
  (hy : y = a * b + a * c + b * c)
  (hz : z = a + b + c)
  (ha : a ≠ b)
  (hb : a ≠ c)
  (hc : b ≠ c) :
  x - a * y + a^2 * z = a^3 ∧
  x - b * y + b^2 * z = b^3 ∧
  x - c * y + c^2 * z = c^3 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1537_153738


namespace NUMINAMATH_CALUDE_straight_line_angle_sum_l1537_153745

-- Define the theorem
theorem straight_line_angle_sum 
  (x y : ℝ) 
  (h1 : x + y = 76)  -- Given condition
  (h2 : 3 * x + 2 * y = 180)  -- Straight line segment condition
  : x = 28 := by
  sorry


end NUMINAMATH_CALUDE_straight_line_angle_sum_l1537_153745


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l1537_153723

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_monotonicity 
  (a b c : ℝ) :
  (f_derivative a b (-2/3) = 0 ∧ f_derivative a b 1 = 0) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x, -2/3 < x ∧ x < 1 → f_derivative (-1/2) (-2) x < 0) ∧
  (∀ x, (x < -2/3 ∨ 1 < x) → f_derivative (-1/2) (-2) x > 0) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l1537_153723


namespace NUMINAMATH_CALUDE_log_inequality_sufficiency_not_necessity_l1537_153718

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement of the theorem
theorem log_inequality_sufficiency_not_necessity :
  (∀ a b : ℝ, log10 a > log10 b → a > b) ∧
  (∃ a b : ℝ, a > b ∧ ¬(log10 a > log10 b)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_sufficiency_not_necessity_l1537_153718


namespace NUMINAMATH_CALUDE_tanya_accompanied_twice_l1537_153755

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  anya : ℕ
  tanya : ℕ
  olya : ℕ
  katya : ℕ

/-- Calculates the number of times a girl accompanied given the total songs and her sung songs -/
def timesAccompanied (totalSongs : ℕ) (sungSongs : ℕ) : ℕ :=
  totalSongs - sungSongs

/-- Theorem: Given the song counts, Tanya accompanied 2 times -/
theorem tanya_accompanied_twice (counts : SongCounts)
    (h1 : counts.anya = 8)
    (h2 : counts.tanya = 6)
    (h3 : counts.olya = 3)
    (h4 : counts.katya = 7) :
    timesAccompanied ((counts.anya + counts.tanya + counts.olya + counts.katya) / 3) counts.tanya = 2 := by
  sorry


end NUMINAMATH_CALUDE_tanya_accompanied_twice_l1537_153755


namespace NUMINAMATH_CALUDE_smallest_k_divides_l1537_153754

def f (z : ℂ) : ℂ := z^11 + z^10 + z^7 + z^6 + z^5 + z^2 + 1

theorem smallest_k_divides : 
  ∃ (k : ℕ), k > 0 ∧ (∀ (z : ℂ), f z = 0 → z^k = 1) ∧
  (∀ (m : ℕ), m > 0 ∧ m < k → ∃ (w : ℂ), f w = 0 ∧ w^m ≠ 1) ∧
  k = 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_divides_l1537_153754


namespace NUMINAMATH_CALUDE_cube_root_of_negative_one_eighth_l1537_153721

theorem cube_root_of_negative_one_eighth :
  ∃ y : ℚ, y^3 = -1/8 ∧ y = -1/2 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_one_eighth_l1537_153721


namespace NUMINAMATH_CALUDE_prob_at_least_one_defective_l1537_153795

/-- The probability of drawing a defective box from each large box -/
def p_defective : ℝ := 0.01

/-- The probability of drawing a non-defective box from each large box -/
def p_non_defective : ℝ := 1 - p_defective

/-- The number of boxes drawn -/
def n : ℕ := 3

theorem prob_at_least_one_defective :
  1 - p_non_defective ^ n = 1 - 0.99 ^ 3 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_defective_l1537_153795


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_50_l1537_153788

theorem last_three_digits_of_7_to_50 : 7^50 % 1000 = 991 := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_50_l1537_153788


namespace NUMINAMATH_CALUDE_planar_graph_properties_l1537_153782

structure PlanarGraph where
  s : ℕ  -- number of vertices
  a : ℕ  -- number of edges
  f : ℕ  -- number of faces

def no_triangular_faces (G : PlanarGraph) : Prop :=
  -- This is a placeholder for the condition that no face is a triangle
  True

theorem planar_graph_properties (G : PlanarGraph) :
  (G.s - G.a + G.f = 2) ∧
  (G.a ≤ 3 * G.s - 6) ∧
  (no_triangular_faces G → G.a ≤ 2 * G.s - 4) := by
  sorry

end NUMINAMATH_CALUDE_planar_graph_properties_l1537_153782


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l1537_153702

theorem ice_cream_distribution (nieces : ℚ) (total_sandwiches : ℕ) :
  nieces = 11 ∧ total_sandwiches = 1573 →
  (total_sandwiches : ℚ) / nieces = 143 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l1537_153702


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l1537_153761

theorem binomial_expansion_problem (n : ℕ) (h : (2 : ℝ)^n = 256) :
  n = 8 ∧ (Nat.choose n (n / 2) : ℝ) = 70 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l1537_153761


namespace NUMINAMATH_CALUDE_last_date_divisible_by_101_in_2011_l1537_153779

def is_valid_date (year month day : ℕ) : Prop :=
  year = 2011 ∧ 
  1 ≤ month ∧ month ≤ 12 ∧
  1 ≤ day ∧ day ≤ 31

def date_to_number (year month day : ℕ) : ℕ :=
  year * 10000 + month * 100 + day

theorem last_date_divisible_by_101_in_2011 :
  ∀ (month day : ℕ),
    is_valid_date 2011 month day →
    date_to_number 2011 month day ≤ 20111221 ∨
    ¬(date_to_number 2011 month day % 101 = 0) :=
sorry

end NUMINAMATH_CALUDE_last_date_divisible_by_101_in_2011_l1537_153779


namespace NUMINAMATH_CALUDE_harriet_round_trip_l1537_153719

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_round_trip 
  (speed_to_b : ℝ) 
  (speed_from_b : ℝ) 
  (time_to_b_minutes : ℝ) 
  (h1 : speed_to_b = 110) 
  (h2 : speed_from_b = 140) 
  (h3 : time_to_b_minutes = 168) : 
  let time_to_b := time_to_b_minutes / 60
  let distance := speed_to_b * time_to_b
  let time_from_b := distance / speed_from_b
  time_to_b + time_from_b = 5 := by
  sorry

end NUMINAMATH_CALUDE_harriet_round_trip_l1537_153719


namespace NUMINAMATH_CALUDE_tim_kittens_count_tim_final_kitten_count_l1537_153700

theorem tim_kittens_count : ℕ → ℕ → ℕ → ℕ
  | initial_kittens, sara_kittens, adoption_rate =>
    let kittens_after_jessica := initial_kittens - initial_kittens / 3
    let kittens_before_adoption := kittens_after_jessica + sara_kittens
    let adopted_kittens := sara_kittens * adoption_rate / 100
    kittens_before_adoption - adopted_kittens

theorem tim_final_kitten_count :
  tim_kittens_count 12 14 50 = 15 := by
  sorry

end NUMINAMATH_CALUDE_tim_kittens_count_tim_final_kitten_count_l1537_153700


namespace NUMINAMATH_CALUDE_equation_solution_l1537_153742

theorem equation_solution :
  ∃ (square : ℚ),
    (((13/5 : ℚ) - ((17/2 : ℚ) - square) / (7/2 : ℚ)) / ((2 : ℚ) / 15)) = 2 ∧
    square = (1/3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1537_153742


namespace NUMINAMATH_CALUDE_average_value_iff_m_in_zero_two_l1537_153787

/-- A function f has an average value on [a, b] if there exists x₀ ∈ (a, b) such that
    f(x₀) = (f(b) - f(a)) / (b - a) -/
def has_average_value (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The quadratic function f(x) = -x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x + 1

theorem average_value_iff_m_in_zero_two :
  ∀ m : ℝ, has_average_value (f m) (-1) 1 ↔ 0 < m ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_average_value_iff_m_in_zero_two_l1537_153787


namespace NUMINAMATH_CALUDE_spherical_segment_height_l1537_153711

/-- The height of a spherical segment given a right-angled triangle inscribed in its base -/
theorem spherical_segment_height
  (S : ℝ) -- Area of the inscribed right-angled triangle
  (α : ℝ) -- Acute angle of the inscribed right-angled triangle
  (β : ℝ) -- Central angle of the segment's arc in axial section
  (h_S_pos : S > 0)
  (h_α_pos : 0 < α)
  (h_α_lt_pi_2 : α < π / 2)
  (h_β_pos : 0 < β)
  (h_β_lt_pi : β < π) :
  ∃ (height : ℝ), height = Real.sqrt (S / Real.sin (2 * α)) * Real.tan (β / 4) :=
sorry

end NUMINAMATH_CALUDE_spherical_segment_height_l1537_153711


namespace NUMINAMATH_CALUDE_word_exists_l1537_153793

/-- Represents a word in the Russian language -/
structure RussianWord where
  word : String

/-- Represents a festive dance event -/
structure FestiveDanceEvent where
  name : String

/-- Represents a sport -/
inductive Sport
  | FigureSkating
  | RhythmicGymnastics

/-- Represents the Russian pension system -/
structure RussianPensionSystem where
  startYear : Nat
  calculationMethod : String

/-- The word we're looking for satisfies all conditions -/
def satisfiesAllConditions (w : RussianWord) (f : FestiveDanceEvent) (s : Sport) (p : RussianPensionSystem) : Prop :=
  (w.word.toLower = f.name.toLower) ∧ 
  (match s with
    | Sport.FigureSkating => true
    | Sport.RhythmicGymnastics => true) ∧
  (p.startYear = 2015 ∧ p.calculationMethod = w.word)

theorem word_exists : 
  ∃ (w : RussianWord) (f : FestiveDanceEvent) (s : Sport) (p : RussianPensionSystem), 
    satisfiesAllConditions w f s p :=
sorry

end NUMINAMATH_CALUDE_word_exists_l1537_153793


namespace NUMINAMATH_CALUDE_empty_solution_implies_a_geq_half_l1537_153734

theorem empty_solution_implies_a_geq_half (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x + a ≥ 0) → a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_implies_a_geq_half_l1537_153734


namespace NUMINAMATH_CALUDE_least_number_remainder_l1537_153714

theorem least_number_remainder (n : ℕ) (h1 : n % 20 = 14) (h2 : n % 2535 = 1929) (h3 : n = 1394) : n % 40 = 34 := by
  sorry

end NUMINAMATH_CALUDE_least_number_remainder_l1537_153714


namespace NUMINAMATH_CALUDE_max_common_chord_length_l1537_153746

-- Define the circles
def circle1 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x + 2*a*y + 2*a^2 - 1 = 0

def circle2 (b : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*b*x + 2*b*y + 2*b^2 - 2 = 0

-- Define the common chord
def commonChord (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | circle1 a p.1 p.2 ∧ circle2 b p.1 p.2}

-- Theorem statement
theorem max_common_chord_length (a b : ℝ) :
  ∃ (l : ℝ), l = 2 ∧ ∀ (p q : ℝ × ℝ), p ∈ commonChord a b → q ∈ commonChord a b →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ l :=
by sorry

end NUMINAMATH_CALUDE_max_common_chord_length_l1537_153746


namespace NUMINAMATH_CALUDE_jean_spots_on_sides_l1537_153768

/-- Represents the number of spots on different parts of Jean the jaguar. -/
structure JeanSpots where
  total : ℕ
  upperTorso : ℕ
  backAndHindquarters : ℕ
  sides : ℕ

/-- Theorem stating the number of spots on Jean's sides given the distribution of spots. -/
theorem jean_spots_on_sides (j : JeanSpots) 
  (h1 : j.upperTorso = j.total / 2)
  (h2 : j.backAndHindquarters = j.total / 3)
  (h3 : j.sides = j.total - j.upperTorso - j.backAndHindquarters)
  (h4 : j.upperTorso = 30) :
  j.sides = 10 := by
  sorry

end NUMINAMATH_CALUDE_jean_spots_on_sides_l1537_153768


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1537_153749

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : 1000 ≤ m ∧ m ≤ 9999) 
  (h2 : Even m) (h3 : m % 221 = 0) : 
  ∃ (d : ℕ), d > 221 ∧ m % d = 0 ∧ d = 247 ∧ 
  ∀ (x : ℕ), 221 < x ∧ x < 247 → m % x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_221_l1537_153749


namespace NUMINAMATH_CALUDE_investment_value_after_one_year_l1537_153703

def initial_investment : ℝ := 900
def num_stocks : ℕ := 3
def stock_a_multiplier : ℝ := 2
def stock_b_multiplier : ℝ := 2
def stock_c_multiplier : ℝ := 0.5

theorem investment_value_after_one_year :
  let investment_per_stock := initial_investment / num_stocks
  let stock_a_value := investment_per_stock * stock_a_multiplier
  let stock_b_value := investment_per_stock * stock_b_multiplier
  let stock_c_value := investment_per_stock * stock_c_multiplier
  stock_a_value + stock_b_value + stock_c_value = 1350 := by
  sorry

end NUMINAMATH_CALUDE_investment_value_after_one_year_l1537_153703


namespace NUMINAMATH_CALUDE_cartesian_coordinates_l1537_153791

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define planes and axes
def yOz_plane (p : Point3D) : Prop := p.x = 0
def z_axis (p : Point3D) : Prop := p.x = 0 ∧ p.y = 0
def xOz_plane (p : Point3D) : Prop := p.y = 0

-- Theorem statement
theorem cartesian_coordinates :
  (∃ (p : Point3D), yOz_plane p ∧ ∃ (b c : ℝ), p.y = b ∧ p.z = c) ∧
  (∃ (p : Point3D), z_axis p ∧ ∃ (c : ℝ), p.z = c) ∧
  (∃ (p : Point3D), xOz_plane p ∧ ∃ (a c : ℝ), p.x = a ∧ p.z = c) :=
by sorry

end NUMINAMATH_CALUDE_cartesian_coordinates_l1537_153791


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_l1537_153706

theorem inscribed_triangle_angle (x : ℝ) : 
  let arc_DE := x + 90
  let arc_EF := 2*x + 15
  let arc_FD := 3*x - 30
  -- Sum of arcs is 360°
  arc_DE + arc_EF + arc_FD = 360 →
  -- Triangle inscribed in circle
  -- Interior angles are half the corresponding arc measures
  ∃ (angle : ℝ), (angle = arc_EF / 2 ∨ angle = arc_FD / 2 ∨ angle = arc_DE / 2) ∧ 
  (angle ≥ 68.5 ∧ angle ≤ 69.5) :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_l1537_153706


namespace NUMINAMATH_CALUDE_impossible_grid_arrangement_l1537_153709

/-- A type representing the digits 0, 1, and 2 -/
inductive Digit
  | zero
  | one
  | two

/-- A type representing a 100 x 100 grid filled with Digits -/
def Grid := Fin 100 → Fin 100 → Digit

/-- A function to count the number of a specific digit in a 3 x 4 rectangle -/
def countDigitIn3x4Rectangle (g : Grid) (i j : Fin 100) (d : Digit) : ℕ :=
  sorry

/-- A predicate to check if a 3 x 4 rectangle satisfies the condition -/
def isValid3x4Rectangle (g : Grid) (i j : Fin 100) : Prop :=
  countDigitIn3x4Rectangle g i j Digit.zero = 3 ∧
  countDigitIn3x4Rectangle g i j Digit.one = 4 ∧
  countDigitIn3x4Rectangle g i j Digit.two = 5

/-- The main theorem stating that it's impossible to fill the grid satisfying the conditions -/
theorem impossible_grid_arrangement : ¬ ∃ (g : Grid), ∀ (i j : Fin 100), isValid3x4Rectangle g i j := by
  sorry

end NUMINAMATH_CALUDE_impossible_grid_arrangement_l1537_153709


namespace NUMINAMATH_CALUDE_stock_change_theorem_l1537_153757

/-- The overall percent change in a stock after two days of trading -/
def overall_percent_change (day1_decrease : ℝ) (day2_increase : ℝ) : ℝ :=
  (((1 - day1_decrease) * (1 + day2_increase)) - 1) * 100

/-- Theorem stating the overall percent change for the given scenario -/
theorem stock_change_theorem :
  overall_percent_change 0.25 0.35 = 1.25 := by
  sorry

#eval overall_percent_change 0.25 0.35

end NUMINAMATH_CALUDE_stock_change_theorem_l1537_153757


namespace NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_range_l1537_153710

/-- 
Given an angle θ in standard position with terminal side passing through (x, y),
prove that sin²θ - cos²θ is between -1 and 1, inclusive.
-/
theorem sin_squared_minus_cos_squared_range (θ x y r : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = r^2) →  -- r is the distance from origin to (x, y)
  r > 0 →  -- r is positive (implicitly given in the problem)
  Real.sin θ = y / r → 
  Real.cos θ = x / r → 
  -1 ≤ Real.sin θ^2 - Real.cos θ^2 ∧ Real.sin θ^2 - Real.cos θ^2 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_cos_squared_range_l1537_153710


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2005_unique_position_2005_l1537_153776

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- The theorem stating that the 669th term of the sequence is 2005 -/
theorem arithmetic_sequence_2005 : arithmetic_sequence 669 = 2005 := by
  sorry

/-- The theorem stating that 669 is the unique position where the sequence equals 2005 -/
theorem unique_position_2005 : ∀ n : ℕ, arithmetic_sequence n = 2005 ↔ n = 669 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2005_unique_position_2005_l1537_153776


namespace NUMINAMATH_CALUDE_jacket_cost_is_30_l1537_153722

def calculate_jacket_cost (initial_amount dresses_count pants_count jackets_count dress_cost pants_cost transportation_cost remaining_amount : ℕ) : ℕ :=
  let total_spent := initial_amount - remaining_amount
  let dresses_cost := dresses_count * dress_cost
  let pants_cost := pants_count * pants_cost
  let other_costs := dresses_cost + pants_cost + transportation_cost
  let jackets_total_cost := total_spent - other_costs
  jackets_total_cost / jackets_count

theorem jacket_cost_is_30 :
  calculate_jacket_cost 400 5 3 4 20 12 5 139 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_is_30_l1537_153722


namespace NUMINAMATH_CALUDE_same_color_probability_l1537_153767

/-- The probability of drawing three marbles of the same color from a bag containing
    3 red marbles, 7 white marbles, and 5 blue marbles, without replacement. -/
theorem same_color_probability (red : ℕ) (white : ℕ) (blue : ℕ) 
    (h_red : red = 3) (h_white : white = 7) (h_blue : blue = 5) :
    let total := red + white + blue
    let p_all_red := (red / total) * ((red - 1) / (total - 1)) * ((red - 2) / (total - 2))
    let p_all_white := (white / total) * ((white - 1) / (total - 1)) * ((white - 2) / (total - 2))
    let p_all_blue := (blue / total) * ((blue - 1) / (total - 1)) * ((blue - 2) / (total - 2))
    p_all_red + p_all_white + p_all_blue = 23 / 455 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1537_153767


namespace NUMINAMATH_CALUDE_base7_addition_l1537_153741

/-- Converts a base 7 number represented as a list of digits to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 represented as a list of digits --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem base7_addition :
  toBase7 (toBase10 [3, 5, 4, 1] + toBase10 [4, 1, 6, 3, 2]) = [5, 5, 4, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_base7_addition_l1537_153741


namespace NUMINAMATH_CALUDE_fishing_trip_total_l1537_153794

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  (morning_catch + afternoon_catch - thrown_back) + dad_catch

/-- Theorem stating that the total number of fish caught is 23 -/
theorem fishing_trip_total : 
  total_fish 8 3 5 13 = 23 := by sorry

end NUMINAMATH_CALUDE_fishing_trip_total_l1537_153794


namespace NUMINAMATH_CALUDE_base_conversion_equality_l1537_153778

/-- Given that 32₄ = 120ᵦ, prove that the unique positive integer b satisfying this equation is 2. -/
theorem base_conversion_equality (b : ℕ) : b > 0 ∧ (3 * 4 + 2) = (1 * b^2 + 2 * b + 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_equality_l1537_153778


namespace NUMINAMATH_CALUDE_tom_seashells_l1537_153769

/-- The number of broken seashells Tom found -/
def broken_seashells : ℕ := 4

/-- The number of unbroken seashells Tom found -/
def unbroken_seashells : ℕ := 3

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := broken_seashells + unbroken_seashells

theorem tom_seashells : total_seashells = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l1537_153769


namespace NUMINAMATH_CALUDE_hamburger_combinations_l1537_153707

/-- The number of condiments available for hamburgers -/
def num_condiments : ℕ := 9

/-- The number of choices for meat patties -/
def num_patty_choices : ℕ := 4

/-- The number of bread type choices -/
def num_bread_choices : ℕ := 2

/-- The total number of different hamburger combinations -/
def total_combinations : ℕ := 2^num_condiments * num_patty_choices * num_bread_choices

theorem hamburger_combinations :
  total_combinations = 4096 :=
sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l1537_153707


namespace NUMINAMATH_CALUDE_least_number_of_tiles_l1537_153770

def room_length : ℕ := 672
def room_width : ℕ := 432

theorem least_number_of_tiles (length : ℕ) (width : ℕ) 
  (h1 : length = room_length) (h2 : width = room_width) : 
  ∃ (tile_size : ℕ), tile_size > 0 ∧ 
  length % tile_size = 0 ∧ 
  width % tile_size = 0 ∧
  (length / tile_size) * (width / tile_size) = 126 := by
  sorry

end NUMINAMATH_CALUDE_least_number_of_tiles_l1537_153770


namespace NUMINAMATH_CALUDE_tangent_line_equation_point_B_coordinates_fixed_point_on_AB_l1537_153751

-- Define the parabola Γ
def Γ (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define point D
def D (p x₀ y₀ : ℝ) : Prop := y₀^2 > 2*p*x₀

-- Define tangent line through D intersecting Γ at A and B
def tangent_line (p x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  D p x₀ y₀ ∧ Γ p x₁ y₁ ∧ Γ p x₂ y₂

-- Theorem 1: Line yy₁ = p(x + x₁) is tangent to Γ
theorem tangent_line_equation (p x₀ y₀ x₁ y₁ : ℝ) :
  tangent_line p x₀ y₀ x₁ y₁ x₁ y₁ → ∀ x y, y * y₁ = p * (x + x₁) := by sorry

-- Theorem 2: Coordinates of B when A(4, 4) and D on directrix
theorem point_B_coordinates (p : ℝ) :
  Γ p 4 4 → D p (-p/2) (3/2) → ∃ x₂ y₂, Γ p x₂ y₂ ∧ x₂ = 1/4 ∧ y₂ = -1 := by sorry

-- Theorem 3: AB passes through fixed point when D moves on x + p = 0
theorem fixed_point_on_AB (p x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) :
  tangent_line p x₀ y₀ x₁ y₁ x₂ y₂ → x₀ = -p → 
  ∃ k b, y₁ - y₂ = k * (x₁ - x₂) ∧ y₁ = k * x₁ + b ∧ 0 = k * p + b := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_point_B_coordinates_fixed_point_on_AB_l1537_153751


namespace NUMINAMATH_CALUDE_matrix_property_l1537_153725

/-- A 4x4 complex matrix with the given structure -/
def M (a b c d : ℂ) : Matrix (Fin 4) (Fin 4) ℂ :=
  ![![a, b, c, d],
    ![b, c, d, a],
    ![c, d, a, b],
    ![d, a, b, c]]

theorem matrix_property (a b c d : ℂ) :
  M a b c d ^ 2 = 1 → a * b * c * d = 1 → a^4 + b^4 + c^4 + d^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_property_l1537_153725


namespace NUMINAMATH_CALUDE_jenny_game_ratio_l1537_153728

theorem jenny_game_ratio : 
  ∀ (games_against_mark games_against_jill games_jenny_won : ℕ)
    (mark_wins jill_win_percentage : ℚ),
    games_against_mark = 10 →
    mark_wins = 1 →
    jill_win_percentage = 3/4 →
    games_jenny_won = 14 →
    games_against_jill = (games_jenny_won - (games_against_mark - mark_wins)) / (1 - jill_win_percentage) →
    games_against_jill / games_against_mark = 2 := by
  sorry

end NUMINAMATH_CALUDE_jenny_game_ratio_l1537_153728


namespace NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1537_153783

theorem no_solution_quadratic_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 + 9 * x ≤ -12 := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_inequality_l1537_153783


namespace NUMINAMATH_CALUDE_max_value_of_shui_l1537_153758

def ChineseDigit := Fin 8

structure Phrase :=
  (jin xin li : ChineseDigit)
  (ke ba shan : ChineseDigit)
  (qiong shui : ChineseDigit)

def is_valid_phrase (p : Phrase) : Prop :=
  p.jin.val + p.xin.val + p.jin.val + p.li.val = 19 ∧
  p.li.val + p.ke.val + p.ba.val + p.shan.val = 19 ∧
  p.shan.val + p.qiong.val + p.shui.val + p.jin.val = 19

def all_different (p : Phrase) : Prop :=
  p.jin ≠ p.xin ∧ p.jin ≠ p.li ∧ p.jin ≠ p.ke ∧ p.jin ≠ p.ba ∧ p.jin ≠ p.shan ∧ p.jin ≠ p.qiong ∧ p.jin ≠ p.shui ∧
  p.xin ≠ p.li ∧ p.xin ≠ p.ke ∧ p.xin ≠ p.ba ∧ p.xin ≠ p.shan ∧ p.xin ≠ p.qiong ∧ p.xin ≠ p.shui ∧
  p.li ≠ p.ke ∧ p.li ≠ p.ba ∧ p.li ≠ p.shan ∧ p.li ≠ p.qiong ∧ p.li ≠ p.shui ∧
  p.ke ≠ p.ba ∧ p.ke ≠ p.shan ∧ p.ke ≠ p.qiong ∧ p.ke ≠ p.shui ∧
  p.ba ≠ p.shan ∧ p.ba ≠ p.qiong ∧ p.ba ≠ p.shui ∧
  p.shan ≠ p.qiong ∧ p.shan ≠ p.shui ∧
  p.qiong ≠ p.shui

theorem max_value_of_shui (p : Phrase) 
  (h1 : is_valid_phrase p)
  (h2 : all_different p)
  (h3 : p.jin.val > p.shan.val ∧ p.shan.val > p.li.val) :
  p.shui.val ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_shui_l1537_153758


namespace NUMINAMATH_CALUDE_parabola_point_value_l1537_153736

theorem parabola_point_value (a b : ℝ) : 
  (a * (-2)^2 + b * (-2) + 5 = 9) → (2*a - b + 6 = 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_value_l1537_153736


namespace NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_squares_l1537_153777

theorem arithmetic_progression_reciprocals_squares (a b c : ℝ) :
  (2 / (c + a) = 1 / (b + c) + 1 / (b + a)) →
  (a^2 + c^2 = 2 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_squares_l1537_153777


namespace NUMINAMATH_CALUDE_banana_group_size_l1537_153730

/-- Given a collection of bananas organized into groups, this theorem proves
    the size of each group when the total number of bananas and groups are known. -/
theorem banana_group_size
  (total_bananas : ℕ)
  (num_groups : ℕ)
  (h1 : total_bananas = 203)
  (h2 : num_groups = 7)
  : total_bananas / num_groups = 29 := by
  sorry

#eval 203 / 7  -- This should evaluate to 29

end NUMINAMATH_CALUDE_banana_group_size_l1537_153730
