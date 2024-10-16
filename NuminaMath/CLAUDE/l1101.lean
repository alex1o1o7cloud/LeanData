import Mathlib

namespace NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l1101_110178

theorem remaining_time_for_finger_exerciser 
  (total_time : Nat) 
  (piano_time : Nat) 
  (writing_time : Nat) 
  (reading_time : Nat) 
  (h1 : total_time = 120)
  (h2 : piano_time = 30)
  (h3 : writing_time = 25)
  (h4 : reading_time = 38) :
  total_time - (piano_time + writing_time + reading_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l1101_110178


namespace NUMINAMATH_CALUDE_point_on_y_axis_equal_distance_to_axes_l1101_110104

-- Define point P with parameter a
def P (a : ℝ) : ℝ × ℝ := (2 + a, 3 * a - 6)

-- Theorem for part 1
theorem point_on_y_axis (a : ℝ) :
  P a = (0, -12) ↔ (P a).1 = 0 :=
sorry

-- Theorem for part 2
theorem equal_distance_to_axes (a : ℝ) :
  (P a = (6, 6) ∨ P a = (3, -3)) ↔ abs (P a).1 = abs (P a).2 :=
sorry

end NUMINAMATH_CALUDE_point_on_y_axis_equal_distance_to_axes_l1101_110104


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1101_110168

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5*x + q = 0}

-- State the theorem
theorem union_of_A_and_B (p q : ℝ) :
  (Set.compl (A p) ∩ B q = {2}) →
  (A p ∩ Set.compl (B q) = {4}) →
  (A p ∪ B q = {2, 3, 6}) := by
  sorry


end NUMINAMATH_CALUDE_union_of_A_and_B_l1101_110168


namespace NUMINAMATH_CALUDE_f_one_zero_iff_l1101_110163

/-- A function f(x) = ax^2 - x - 1 where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - 1

/-- The property that f has exactly one zero -/
def has_exactly_one_zero (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that f has exactly one zero iff a = 0 or a = -1/4 -/
theorem f_one_zero_iff (a : ℝ) :
  has_exactly_one_zero a ↔ a = 0 ∨ a = -1/4 := by sorry

end NUMINAMATH_CALUDE_f_one_zero_iff_l1101_110163


namespace NUMINAMATH_CALUDE_first_book_cost_l1101_110189

/-- The cost of Shelby's first book given her spending at the book fair -/
theorem first_book_cost (initial_amount : ℕ) (second_book_cost : ℕ) (poster_cost : ℕ) (num_posters : ℕ) :
  initial_amount = 20 →
  second_book_cost = 4 →
  poster_cost = 4 →
  num_posters = 2 →
  ∃ (first_book_cost : ℕ),
    first_book_cost + second_book_cost + (num_posters * poster_cost) = initial_amount ∧
    first_book_cost = 8 :=
by sorry

end NUMINAMATH_CALUDE_first_book_cost_l1101_110189


namespace NUMINAMATH_CALUDE_candy_difference_is_twenty_l1101_110165

/-- The number of candies Bryan has compared to Ben -/
def candy_difference : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  λ bryan_skittles bryan_gummy bryan_lollipops ben_mm ben_jelly ben_lollipops =>
    (bryan_skittles + bryan_gummy + bryan_lollipops) - (ben_mm + ben_jelly + ben_lollipops)

theorem candy_difference_is_twenty :
  candy_difference 50 30 15 20 45 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_is_twenty_l1101_110165


namespace NUMINAMATH_CALUDE_circular_path_length_l1101_110138

/-- The length of a circular path given specific walking conditions -/
theorem circular_path_length
  (step_length_1 : ℝ)
  (step_length_2 : ℝ)
  (total_footprints : ℕ)
  (h1 : step_length_1 = 0.54)  -- 54 cm in meters
  (h2 : step_length_2 = 0.72)  -- 72 cm in meters
  (h3 : total_footprints = 60)
  (h4 : ∃ (n m : ℕ), n * step_length_1 = m * step_length_2) -- Both complete one lap
  : ∃ (path_length : ℝ), path_length = 21.6 :=
by
  sorry

end NUMINAMATH_CALUDE_circular_path_length_l1101_110138


namespace NUMINAMATH_CALUDE_orange_bin_count_l1101_110117

/-- Given an initial quantity of oranges, a number of oranges removed, and a number of oranges added,
    calculate the final quantity of oranges. -/
def final_orange_count (initial : ℕ) (removed : ℕ) (added : ℕ) : ℕ :=
  initial - removed + added

/-- Theorem stating that given the specific values from the problem,
    the final orange count is 31. -/
theorem orange_bin_count : final_orange_count 5 2 28 = 31 := by
  sorry

end NUMINAMATH_CALUDE_orange_bin_count_l1101_110117


namespace NUMINAMATH_CALUDE_inverse_inequality_l1101_110148

theorem inverse_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 1/x < 1/y := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l1101_110148


namespace NUMINAMATH_CALUDE_three_in_M_l1101_110188

def U : Set ℤ := {x | x^2 - 6*x < 0}

theorem three_in_M (M : Set ℤ) (h : (U \ M) = {1, 2}) : 3 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_three_in_M_l1101_110188


namespace NUMINAMATH_CALUDE_proposition_truth_l1101_110176

theorem proposition_truth : 
  (∀ x : ℝ, x > 0 → (3 : ℝ) ^ x > (2 : ℝ) ^ x) ∧ 
  (∀ x : ℝ, x < 0 → (3 : ℝ) * x ≤ (2 : ℝ) * x) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_l1101_110176


namespace NUMINAMATH_CALUDE_dormitory_students_count_l1101_110110

theorem dormitory_students_count :
  ∃ (x y : ℕ),
    x > 0 ∧
    y > 0 ∧
    x * (x - 1) + x * y + y = 51 ∧
    x = 6 := by
  sorry

end NUMINAMATH_CALUDE_dormitory_students_count_l1101_110110


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1101_110120

theorem quadratic_equation_result (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 2*x^2 - 4*x = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1101_110120


namespace NUMINAMATH_CALUDE_apple_count_theorem_l1101_110112

def is_valid_apple_count (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (n % 6 = 0)

theorem apple_count_theorem :
  ∀ n : ℕ, is_valid_apple_count n ↔ (n = 72 ∨ n = 78) :=
by sorry

end NUMINAMATH_CALUDE_apple_count_theorem_l1101_110112


namespace NUMINAMATH_CALUDE_train_length_approx_100_l1101_110154

/-- Calculates the length of a train given its speed, the time it takes to cross a platform, and the length of the platform. -/
def train_length (speed : ℝ) (time : ℝ) (platform_length : ℝ) : ℝ :=
  speed * time - platform_length

/-- Theorem stating that a train with given parameters has a length of approximately 100 meters. -/
theorem train_length_approx_100 (speed : ℝ) (time : ℝ) (platform_length : ℝ) 
  (h1 : speed = 60 * 1000 / 3600) -- 60 km/hr converted to m/s
  (h2 : time = 14.998800095992321)
  (h3 : platform_length = 150) :
  ∃ ε > 0, |train_length speed time platform_length - 100| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_approx_100_l1101_110154


namespace NUMINAMATH_CALUDE_expression_value_l1101_110124

theorem expression_value : 
  (49 * 91^3 + 338 * 343^2) / (66^3 - 176 * 121) / (39^3 * 7^5 / 1331000) = 5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1101_110124


namespace NUMINAMATH_CALUDE_exam_score_problem_l1101_110183

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 140 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 40 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l1101_110183


namespace NUMINAMATH_CALUDE_parallelogram_may_not_have_symmetry_l1101_110151

-- Define the basic geometric shapes
inductive GeometricShape
  | LineSegment
  | Rectangle
  | Angle
  | Parallelogram

-- Define a property for having an axis of symmetry
def has_axis_of_symmetry (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.LineSegment => True
  | GeometricShape.Rectangle => True
  | GeometricShape.Angle => True
  | GeometricShape.Parallelogram => sorry  -- This can be True or False

-- Theorem: Only parallelograms may not have an axis of symmetry
theorem parallelogram_may_not_have_symmetry :
  ∀ (shape : GeometricShape),
    ¬(has_axis_of_symmetry shape) → shape = GeometricShape.Parallelogram :=
by sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_parallelogram_may_not_have_symmetry_l1101_110151


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1101_110111

theorem arithmetic_sequence_problem (a b c : ℝ) : 
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →  -- arithmetic sequence condition
  a + b + c = 9 →                       -- sum condition
  a * b = 6 * c →                       -- product condition
  a = 4 ∧ b = 3 ∧ c = 2 := by           -- conclusion
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1101_110111


namespace NUMINAMATH_CALUDE_survey_sample_size_l1101_110156

/-- Represents a survey conducted in an urban area -/
structure UrbanSurvey where
  year : Nat
  month : Nat
  investigators : Nat
  households : Nat
  questionnaires : Nat

/-- Definition of sample size for an urban survey -/
def sampleSize (survey : UrbanSurvey) : Nat :=
  survey.questionnaires

/-- Theorem stating that the sample size of the given survey is 30,000 -/
theorem survey_sample_size :
  let survey : UrbanSurvey := {
    year := 2010
    month := 5  -- May
    investigators := 400
    households := 10000
    questionnaires := 30000
  }
  sampleSize survey = 30000 := by
  sorry


end NUMINAMATH_CALUDE_survey_sample_size_l1101_110156


namespace NUMINAMATH_CALUDE_function_1_extrema_function_2_extrema_l1101_110169

-- Function 1
theorem function_1_extrema :
  (∀ x : ℝ, 2 * Real.sin x - 3 ≤ -1) ∧
  (∃ x : ℝ, 2 * Real.sin x - 3 = -1) ∧
  (∀ x : ℝ, 2 * Real.sin x - 3 ≥ -5) ∧
  (∃ x : ℝ, 2 * Real.sin x - 3 = -5) :=
sorry

-- Function 2
theorem function_2_extrema :
  (∀ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 ≤ 2) ∧
  (∃ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 = 2) ∧
  (∀ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 ≥ -1/4) ∧
  (∃ x : ℝ, 7/4 + Real.sin x - (Real.sin x)^2 = -1/4) :=
sorry

end NUMINAMATH_CALUDE_function_1_extrema_function_2_extrema_l1101_110169


namespace NUMINAMATH_CALUDE_new_numbers_average_l1101_110130

theorem new_numbers_average (initial_count : ℕ) (initial_mean : ℝ) 
  (new_count : ℕ) (new_mean : ℝ) : 
  initial_count = 12 →
  initial_mean = 45 →
  new_count = 15 →
  new_mean = 60 →
  (new_count * new_mean - initial_count * initial_mean) / (new_count - initial_count) = 120 :=
by sorry

end NUMINAMATH_CALUDE_new_numbers_average_l1101_110130


namespace NUMINAMATH_CALUDE_plastering_rate_calculation_l1101_110134

/-- Given a tank with specified dimensions and total plastering cost,
    calculate the rate of plastering per square meter in paise. -/
theorem plastering_rate_calculation (length width depth : ℝ) (total_cost : ℝ) :
  length = 25 →
  width = 12 →
  depth = 6 →
  total_cost = 558 →
  let total_area := 2 * (length * depth + width * depth) + length * width
  let rate_per_sqm_rupees := total_cost / total_area
  let rate_per_sqm_paise := rate_per_sqm_rupees * 100
  rate_per_sqm_paise = 75 := by
  sorry

#check plastering_rate_calculation

end NUMINAMATH_CALUDE_plastering_rate_calculation_l1101_110134


namespace NUMINAMATH_CALUDE_elevator_max_weight_next_person_l1101_110159

/-- Given an elevator scenario with adults and children, calculate the maximum weight of the next person that can enter without overloading the elevator. -/
theorem elevator_max_weight_next_person 
  (num_adults : ℕ) 
  (avg_weight_adults : ℝ) 
  (num_children : ℕ) 
  (avg_weight_children : ℝ) 
  (max_elevator_weight : ℝ) 
  (h1 : num_adults = 7) 
  (h2 : avg_weight_adults = 150) 
  (h3 : num_children = 5) 
  (h4 : avg_weight_children = 70) 
  (h5 : max_elevator_weight = 1500) :
  max_elevator_weight - (num_adults * avg_weight_adults + num_children * avg_weight_children) = 100 := by
  sorry

end NUMINAMATH_CALUDE_elevator_max_weight_next_person_l1101_110159


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1101_110125

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1101_110125


namespace NUMINAMATH_CALUDE_unique_natural_pair_l1101_110133

theorem unique_natural_pair : ∃! (a b : ℕ), 
  a ≠ b ∧ 
  (∃ (k : ℕ), ∃ (p : ℕ), Prime p ∧ b^2 + a = p^k) ∧
  (∃ (m : ℕ), (a^2 + b) * m = b^2 + a) ∧
  a = 2 ∧ 
  b = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_natural_pair_l1101_110133


namespace NUMINAMATH_CALUDE_at_most_one_root_l1101_110109

-- Define a monotonically increasing function on ℝ
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Theorem statement
theorem at_most_one_root (f : ℝ → ℝ) (h : MonoIncreasing f) :
  ∃! x, f x = 0 ∨ ∀ x, f x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_at_most_one_root_l1101_110109


namespace NUMINAMATH_CALUDE_min_value_sum_l1101_110140

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' / (3 * b') + b' / (6 * c') + c' / (9 * a') = 3 / Real.rpow 162 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_l1101_110140


namespace NUMINAMATH_CALUDE_original_calculation_l1101_110173

theorem original_calculation (x : ℚ) (h : ((x * 3) + 14) * 2 = 946) : ((x / 3) + 14) * 2 = 130 := by
  sorry

end NUMINAMATH_CALUDE_original_calculation_l1101_110173


namespace NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l1101_110132

def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

def N : Set ℝ := {x | Real.log 2 ^ (1 - x) < 1}

theorem intersection_of_M_and_complement_of_N :
  M ∩ (Set.univ \ N) = Set.Icc 1 2 \ {2} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l1101_110132


namespace NUMINAMATH_CALUDE_lou_senior_first_cookies_l1101_110146

/-- Represents the cookie jar situation --/
structure CookieJar where
  total : ℕ
  louSeniorFirst : ℕ
  louSeniorSecond : ℕ
  louieJunior : ℕ
  remaining : ℕ

/-- The cookie jar problem --/
def cookieJarProblem : CookieJar :=
  { total := 22
  , louSeniorFirst := 3  -- This is what we want to prove
  , louSeniorSecond := 1
  , louieJunior := 7
  , remaining := 11 }

/-- Theorem stating that Lou Senior took 3 cookies the first time --/
theorem lou_senior_first_cookies :
  cookieJarProblem.total - cookieJarProblem.louSeniorFirst - 
  cookieJarProblem.louSeniorSecond - cookieJarProblem.louieJunior = 
  cookieJarProblem.remaining :=
by sorry

end NUMINAMATH_CALUDE_lou_senior_first_cookies_l1101_110146


namespace NUMINAMATH_CALUDE_even_function_implies_c_eq_neg_four_l1101_110131

/-- Given a function f and a constant c, we define g in terms of f and c. -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 3

def g (c : ℝ) (x : ℝ) : ℝ := f x + c*x

/-- A function h is even if h(-x) = h(x) for all x. -/
def IsEven (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = h x

/-- If g is an even function, then c must equal -4. -/
theorem even_function_implies_c_eq_neg_four :
  IsEven (g c) → c = -4 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_c_eq_neg_four_l1101_110131


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1101_110195

open Real
open BigOperators

theorem infinite_series_sum : 
  (∑' n : ℕ, (3 * n + 2) / (n * (n + 1) * (n + 3))) = 10/3 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1101_110195


namespace NUMINAMATH_CALUDE_triangle_proof_l1101_110182

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, BD = b and cos(ABC) = 7/12 -/
theorem triangle_proof (a b c : ℝ) (A B C : ℝ) (D : ℝ × ℝ) :
  b^2 = a * c →
  D.1 ≥ 0 ∧ D.1 ≤ c →  -- D lies on AC
  b * Real.sin B = a * Real.sin C →
  2 * (c - D.1) = D.1 →  -- AD = 2DC
  (b = Real.sqrt (a * c)) ∧
  (Real.cos B = 7 / 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l1101_110182


namespace NUMINAMATH_CALUDE_difference_in_circumferences_l1101_110139

/-- The difference in circumferences of two concentric circular paths -/
theorem difference_in_circumferences 
  (inner_radius : ℝ) 
  (width_difference : ℝ) 
  (h1 : inner_radius = 25) 
  (h2 : width_difference = 15) : 
  2 * π * (inner_radius + width_difference) - 2 * π * inner_radius = 30 * π := by
sorry

end NUMINAMATH_CALUDE_difference_in_circumferences_l1101_110139


namespace NUMINAMATH_CALUDE_average_height_calculation_l1101_110113

theorem average_height_calculation (north_count : ℕ) (south_count : ℕ) 
  (north_avg : ℝ) (south_avg : ℝ) :
  north_count = 300 →
  south_count = 200 →
  north_avg = 1.60 →
  south_avg = 1.50 →
  (north_count * north_avg + south_count * south_avg) / (north_count + south_count) = 1.56 := by
  sorry

end NUMINAMATH_CALUDE_average_height_calculation_l1101_110113


namespace NUMINAMATH_CALUDE_max_x5_value_l1101_110175

theorem max_x5_value (x₁ x₂ x₃ x₄ x₅ : ℕ+) 
  (h : x₁ + x₂ + x₃ + x₄ + x₅ = x₁ * x₂ * x₃ * x₄ * x₅) : 
  x₅ ≤ 5 ∧ ∃ (a b c d : ℕ+), a + b + c + d + 5 = a * b * c * d * 5 := by
  sorry

end NUMINAMATH_CALUDE_max_x5_value_l1101_110175


namespace NUMINAMATH_CALUDE_number_exists_l1101_110171

theorem number_exists : ∃ x : ℝ, 0.75 * x = 0.3 * 1000 + 250 := by
  sorry

end NUMINAMATH_CALUDE_number_exists_l1101_110171


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l1101_110106

theorem difference_of_reciprocals (p q : ℚ) 
  (hp : 4 / p = 8) (hq : 4 / q = 18) : p - q = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l1101_110106


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1101_110197

theorem arithmetic_sequence_sum (a : ℕ → ℕ) : 
  (∀ n, a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
  a 0 = 3 →                            -- first term is 3
  a 1 = 9 →                            -- second term is 9
  a 6 = 33 →                           -- last (seventh) term is 33
  a 4 + a 5 = 60 :=                    -- sum of fifth and sixth terms is 60
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1101_110197


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1101_110155

/-- Proves that given a parabola y^2 = 8x whose latus rectum passes through a focus of a hyperbola
    x^2/a^2 - y^2/b^2 = 1 (a > 0, b > 0), and one asymptote of the hyperbola is x + √3y = 0,
    the equation of the hyperbola is x^2/3 - y^2 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), y^2 = 8*x ∧ x = -2) →  -- Latus rectum of parabola
  (∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  (∃ (x y : ℝ), x + Real.sqrt 3 * y = 0) →  -- Asymptote equation
  (∀ (x y : ℝ), x^2/3 - y^2 = 1) :=  -- Resulting hyperbola equation
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l1101_110155


namespace NUMINAMATH_CALUDE_eight_people_circular_arrangements_l1101_110108

/-- The number of distinct circular arrangements of n people around a round table,
    where rotations are considered identical. -/
def circularArrangements (n : ℕ) : ℕ :=
  Nat.factorial (n - 1)

/-- Theorem stating that the number of distinct circular arrangements
    of 8 people around a round table is 5040. -/
theorem eight_people_circular_arrangements :
  circularArrangements 8 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_circular_arrangements_l1101_110108


namespace NUMINAMATH_CALUDE_binomial_150_1_l1101_110103

theorem binomial_150_1 : Nat.choose 150 1 = 150 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_1_l1101_110103


namespace NUMINAMATH_CALUDE_inequalities_for_x_in_range_l1101_110181

theorem inequalities_for_x_in_range (x : Real) (h : 0 ≤ x ∧ x ≤ π/2) : 
  (1 - Real.cos x ≤ x^2/2) ∧ 
  (x * Real.cos x ≤ Real.sin x ∧ Real.sin x ≤ x * Real.cos (x/2)) := by
sorry

end NUMINAMATH_CALUDE_inequalities_for_x_in_range_l1101_110181


namespace NUMINAMATH_CALUDE_coefficient_a3_equals_84_l1101_110114

theorem coefficient_a3_equals_84 (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (a * x - 1)^9 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + 
                        a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 0) →
  a₃ = 84 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a3_equals_84_l1101_110114


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1101_110172

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 + 48 / 4 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1101_110172


namespace NUMINAMATH_CALUDE_range_of_a_in_system_with_one_integer_solution_l1101_110116

/-- Given a system of inequalities with exactly one integer solution, prove the range of a -/
theorem range_of_a_in_system_with_one_integer_solution :
  ∀ a : ℝ,
  (∃! x : ℤ, (2 * (x : ℝ) + 3 > 5 ∧ (x : ℝ) - a ≤ 0)) →
  (2 ≤ a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_in_system_with_one_integer_solution_l1101_110116


namespace NUMINAMATH_CALUDE_simplify_square_roots_l1101_110164

theorem simplify_square_roots : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l1101_110164


namespace NUMINAMATH_CALUDE_employee_reduction_percentage_l1101_110105

/-- Theorem: Employee Reduction Percentage

Given:
- The number of employees decreased.
- The average salary increased by 10%.
- The total salary remained constant.

Prove:
The percentage decrease in the number of employees is (1 - 1/1.1) * 100%.
-/
theorem employee_reduction_percentage 
  (E : ℝ) -- Initial number of employees
  (E' : ℝ) -- Number of employees after reduction
  (S : ℝ) -- Initial average salary
  (h1 : E' < E) -- Number of employees decreased
  (h2 : E' * (1.1 * S) = E * S) -- Total salary remained constant
  : (E - E') / E * 100 = (1 - 1 / 1.1) * 100 := by
  sorry

#check employee_reduction_percentage

end NUMINAMATH_CALUDE_employee_reduction_percentage_l1101_110105


namespace NUMINAMATH_CALUDE_student_count_equality_l1101_110147

/-- Proves that the number of students in class A equals the number of students in class C
    given the average ages of each class and the overall average age. -/
theorem student_count_equality (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (14 * a + 13 * b + 12 * c) / (a + b + c) = 13 → a = c := by
  sorry

end NUMINAMATH_CALUDE_student_count_equality_l1101_110147


namespace NUMINAMATH_CALUDE_smallest_w_l1101_110157

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 →
  is_factor (2^5) (936 * w) →
  is_factor (3^3) (936 * w) →
  is_factor (12^2) (936 * w) →
  936 = 2^3 * 3^1 * 13^1 →
  (∀ v : ℕ, v > 0 → 
    is_factor (2^5) (936 * v) → 
    is_factor (3^3) (936 * v) → 
    is_factor (12^2) (936 * v) → 
    w ≤ v) →
  w = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_w_l1101_110157


namespace NUMINAMATH_CALUDE_pencils_in_pack_l1101_110136

/-- Given information about Judy's pencil usage and purchases, 
    prove the number of pencils in a pack. -/
theorem pencils_in_pack 
  (pencils_per_week : ℕ)
  (days_per_week : ℕ)
  (pack_cost : ℚ)
  (total_spent : ℚ)
  (total_days : ℕ)
  (h1 : pencils_per_week = 10)
  (h2 : days_per_week = 5)
  (h3 : pack_cost = 4)
  (h4 : total_spent = 12)
  (h5 : total_days = 45) :
  (total_spent / pack_cost) * (pencils_per_week * (total_days / days_per_week)) / 
  (total_spent / pack_cost) = 30 := by
sorry


end NUMINAMATH_CALUDE_pencils_in_pack_l1101_110136


namespace NUMINAMATH_CALUDE_students_left_fourth_grade_students_left_l1101_110152

theorem students_left (initial_students : ℝ) (final_students : ℝ) (transferred_students : ℝ) :
  initial_students ≥ final_students + transferred_students →
  initial_students - (final_students + transferred_students) =
  initial_students - final_students - transferred_students :=
by sorry

def calculate_students_left (initial_students : ℝ) (final_students : ℝ) (transferred_students : ℝ) : ℝ :=
  initial_students - final_students - transferred_students

theorem fourth_grade_students_left :
  let initial_students : ℝ := 42.0
  let final_students : ℝ := 28.0
  let transferred_students : ℝ := 10.0
  calculate_students_left initial_students final_students transferred_students = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_left_fourth_grade_students_left_l1101_110152


namespace NUMINAMATH_CALUDE_f_composition_equals_251_l1101_110107

def f (x : ℝ) : ℝ := 5 * x - 4

theorem f_composition_equals_251 : f (f (f 3)) = 251 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_251_l1101_110107


namespace NUMINAMATH_CALUDE_certain_fraction_proof_l1101_110196

theorem certain_fraction_proof (x y : ℚ) :
  (x / y) / (3 / 7) = 0.46666666666666673 / (1 / 2) →
  x / y = 0.4 := by
sorry

end NUMINAMATH_CALUDE_certain_fraction_proof_l1101_110196


namespace NUMINAMATH_CALUDE_two_tangents_iff_a_in_range_l1101_110126

/-- Definition of the circle C -/
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*y + a^2 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- Condition for exactly two tangents -/
def has_two_tangents (a : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (circle_C a (center.1) (center.2)) ∧
    ((point_A.1 - center.1)^2 + (point_A.2 - center.2)^2 > radius^2) ∧
    (radius^2 > 0)

/-- Main theorem -/
theorem two_tangents_iff_a_in_range :
  ∀ a : ℝ, has_two_tangents a ↔ -2*(3:ℝ).sqrt/3 < a ∧ a < 2*(3:ℝ).sqrt/3 :=
sorry

end NUMINAMATH_CALUDE_two_tangents_iff_a_in_range_l1101_110126


namespace NUMINAMATH_CALUDE_equal_positive_reals_l1101_110192

theorem equal_positive_reals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (x*y + 1) / (x + 1) = (y*z + 1) / (y + 1))
  (h2 : (y*z + 1) / (y + 1) = (z*x + 1) / (z + 1)) :
  x = y ∧ y = z := by sorry

end NUMINAMATH_CALUDE_equal_positive_reals_l1101_110192


namespace NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l1101_110119

theorem three_digit_divisible_by_nine :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  n % 10 = 3 ∧ 
  (n / 100) % 10 = 5 ∧ 
  n % 9 = 0 ∧
  n = 513 :=
sorry

end NUMINAMATH_CALUDE_three_digit_divisible_by_nine_l1101_110119


namespace NUMINAMATH_CALUDE_city_population_theorem_l1101_110194

def city_population (initial_population immigration emigration pregnancy_rate twin_rate : ℕ) : ℕ :=
  let population_after_migration := initial_population + immigration - emigration
  let pregnancies := population_after_migration / 8
  let twin_pregnancies := pregnancies / 4
  let single_pregnancies := pregnancies - twin_pregnancies
  let births := single_pregnancies + 2 * twin_pregnancies
  population_after_migration + births

theorem city_population_theorem :
  city_population 300000 50000 30000 8 4 = 370000 := by
  sorry

end NUMINAMATH_CALUDE_city_population_theorem_l1101_110194


namespace NUMINAMATH_CALUDE_angle_complement_from_supplement_l1101_110158

theorem angle_complement_from_supplement (angle : ℝ) : 
  (180 - angle = 130) → (90 - angle = 40) := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_from_supplement_l1101_110158


namespace NUMINAMATH_CALUDE_friends_not_going_to_movies_l1101_110144

theorem friends_not_going_to_movies (total_friends : ℕ) (friends_going : ℕ) : 
  total_friends = 15 → friends_going = 8 → total_friends - friends_going = 7 := by
  sorry

end NUMINAMATH_CALUDE_friends_not_going_to_movies_l1101_110144


namespace NUMINAMATH_CALUDE_intersecting_circles_sum_l1101_110142

/-- Two circles intersect at points A and B, with their centers on a line -/
structure IntersectingCircles where
  m : ℝ
  n : ℝ
  /-- Point A coordinates -/
  pointA : ℝ × ℝ := (1, 3)
  /-- Point B coordinates -/
  pointB : ℝ × ℝ := (m, n)
  /-- The centers of both circles are on the line x - y - 2 = 0 -/
  centers_on_line : ∀ (x y : ℝ), x - y - 2 = 0

/-- The sum of m and n for the intersecting circles is 4 -/
theorem intersecting_circles_sum (ic : IntersectingCircles) : ic.m + ic.n = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_sum_l1101_110142


namespace NUMINAMATH_CALUDE_existence_of_good_subset_l1101_110191

def M : Finset ℕ := Finset.range 20

def is_valid_function (f : Finset ℕ → ℕ) : Prop :=
  ∀ S : Finset ℕ, S ⊆ M → S.card = 9 → f S ∈ M

theorem existence_of_good_subset (f : Finset ℕ → ℕ) (h : is_valid_function f) :
  ∃ T : Finset ℕ, T ⊆ M ∧ T.card = 10 ∧ ∀ k ∈ T, f (T \ {k}) ≠ k := by
  sorry

#check existence_of_good_subset

end NUMINAMATH_CALUDE_existence_of_good_subset_l1101_110191


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1101_110198

/-- The value of d for which the line y = 3x + d is tangent to the parabola y^2 = 12x -/
theorem tangent_line_to_parabola : ∃ d : ℝ, 
  (∀ x y : ℝ, y = 3*x + d → y^2 = 12*x → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      (y' - (3*x' + d))^2 > ε^2 * ((y')^2 - 12*x')) ∧
  d = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1101_110198


namespace NUMINAMATH_CALUDE_n_fifth_minus_n_divisible_by_30_l1101_110141

theorem n_fifth_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) := by
  sorry

end NUMINAMATH_CALUDE_n_fifth_minus_n_divisible_by_30_l1101_110141


namespace NUMINAMATH_CALUDE_professor_coffee_meeting_l1101_110149

theorem professor_coffee_meeting (n p q r : ℕ) : 
  (∀ (x : ℕ), x > 1 → x.Prime → r % (x ^ 2) ≠ 0) →  -- r is not divisible by the square of any prime
  (n : ℝ) = p - q * Real.sqrt r →  -- n = p - q√r
  (((120 : ℝ) - n) ^ 2 / 14400 = 1 / 2) →  -- probability of meeting is 50%
  p + q + r = 182 := by
  sorry

end NUMINAMATH_CALUDE_professor_coffee_meeting_l1101_110149


namespace NUMINAMATH_CALUDE_unique_solution_complex_magnitude_one_l1101_110184

/-- There exists exactly one real value of x that satisfies |1 - (x/2)i| = 1 -/
theorem unique_solution_complex_magnitude_one :
  ∃! x : ℝ, Complex.abs (1 - (x / 2) * Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_complex_magnitude_one_l1101_110184


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1101_110118

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ 1) (h2 : r ≠ 1) (h3 : p ≠ r) 
  (h4 : k ≠ 0) (h5 : k * p^3 - k * r^3 = 3 * (k * p - k * r)) :
  p + r = Real.sqrt 3 ∨ p + r = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1101_110118


namespace NUMINAMATH_CALUDE_li_shuang_walking_speed_l1101_110161

/-- The problem of finding Li Shuang's walking speed -/
theorem li_shuang_walking_speed 
  (initial_speed : ℝ) 
  (walking_time : ℝ) 
  (repair_distance : ℝ) 
  (repair_time : ℝ) 
  (speed_multiplier : ℝ) 
  (delay : ℝ)
  (h1 : initial_speed = 320)
  (h2 : walking_time = 5)
  (h3 : repair_distance = 1800)
  (h4 : repair_time = 15)
  (h5 : speed_multiplier = 1.5)
  (h6 : delay = 17) :
  ∃ (walking_speed : ℝ), walking_speed = 72 ∧ 
  (∃ (total_distance : ℝ), 
    total_distance / initial_speed + delay = 
    walking_time + repair_time + 
    (total_distance - repair_distance - walking_speed * walking_time) / (initial_speed * speed_multiplier)) := by
  sorry

end NUMINAMATH_CALUDE_li_shuang_walking_speed_l1101_110161


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2009_l1101_110127

/-- Given an arithmetic sequence {a_n} with common difference d and a_k, 
    this function returns a_n -/
def arithmeticSequence (d : ℤ) (k : ℕ) (a_k : ℤ) (n : ℕ) : ℤ :=
  a_k + d * (n - k)

theorem arithmetic_sequence_2009 :
  let d := 2
  let k := 2007
  let a_k := 2007
  let n := 2009
  arithmeticSequence d k a_k n = 2011 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2009_l1101_110127


namespace NUMINAMATH_CALUDE_three_digit_numbers_sum_divisibility_l1101_110174

theorem three_digit_numbers_sum_divisibility :
  ∃ (a b c d : ℕ),
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    100 ≤ c ∧ c < 1000 ∧
    100 ≤ d ∧ d < 1000 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃ (j : ℕ), a / 100 = j ∧ b / 100 = j ∧ c / 100 = j ∧ d / 100 = j) ∧
    (∃ (s : ℕ), s = a + b + c + d ∧ s % a = 0 ∧ s % b = 0 ∧ s % c = 0) ∧
    a = 108 ∧ b = 135 ∧ c = 180 ∧ d = 117 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_sum_divisibility_l1101_110174


namespace NUMINAMATH_CALUDE_debby_water_bottles_l1101_110193

/-- The number of bottles Debby drinks per day -/
def bottles_per_day : ℕ := 5

/-- The number of days the water would last -/
def days_lasting : ℕ := 71

/-- The total number of bottles Debby bought -/
def total_bottles : ℕ := bottles_per_day * days_lasting

theorem debby_water_bottles : total_bottles = 355 := by
  sorry

end NUMINAMATH_CALUDE_debby_water_bottles_l1101_110193


namespace NUMINAMATH_CALUDE_division_simplification_l1101_110100

theorem division_simplification (a : ℝ) (h : a ≠ 0) :
  (21 * a^3 - 7 * a) / (7 * a) = 3 * a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l1101_110100


namespace NUMINAMATH_CALUDE_friends_total_amount_l1101_110199

/-- The total amount of money received by three friends from selling video games -/
def total_amount (zachary_games : ℕ) (price_per_game : ℕ) (jason_percent : ℕ) (ryan_extra : ℕ) : ℕ :=
  let zachary_amount := zachary_games * price_per_game
  let jason_amount := zachary_amount + (jason_percent * zachary_amount) / 100
  let ryan_amount := jason_amount + ryan_extra
  zachary_amount + jason_amount + ryan_amount

/-- Theorem stating that the total amount received by the three friends is $770 -/
theorem friends_total_amount :
  total_amount 40 5 30 50 = 770 := by
  sorry

end NUMINAMATH_CALUDE_friends_total_amount_l1101_110199


namespace NUMINAMATH_CALUDE_selling_price_ratio_l1101_110153

/-- Proves that the ratio of selling prices is 2:1 given different profit percentages -/
theorem selling_price_ratio (cost : ℝ) (profit1 profit2 : ℝ) 
  (h1 : profit1 = 0.20)
  (h2 : profit2 = 1.40) :
  (cost + profit2 * cost) / (cost + profit1 * cost) = 2 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_ratio_l1101_110153


namespace NUMINAMATH_CALUDE_equation_is_quadratic_l1101_110145

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ) (ha : a ≠ 0), ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 2

/-- Theorem: The given equation is a quadratic equation -/
theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_equation_is_quadratic_l1101_110145


namespace NUMINAMATH_CALUDE_line_slope_proof_l1101_110180

/-- Given two points (a, -1) and (2, 3) on a line with slope 2, prove that a = 0 -/
theorem line_slope_proof (a : ℝ) : 
  (3 - (-1)) / (2 - a) = 2 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_proof_l1101_110180


namespace NUMINAMATH_CALUDE_school_bus_seats_l1101_110170

/-- Proves that the number of seats on each school bus is 9, given the conditions of the field trip. -/
theorem school_bus_seats (total_students : ℕ) (num_buses : ℕ) (h1 : total_students = 45) (h2 : num_buses = 5) (h3 : total_students % num_buses = 0) :
  total_students / num_buses = 9 := by
sorry

end NUMINAMATH_CALUDE_school_bus_seats_l1101_110170


namespace NUMINAMATH_CALUDE_intersection_M_N_l1101_110166

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def N : Set ℝ := {x | ∃ y, y = Real.sqrt x + Real.log (1 - x)}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1101_110166


namespace NUMINAMATH_CALUDE_arrangement_theorem_l1101_110185

def number_of_people : ℕ := 6
def people_per_row : ℕ := 3

def arrangement_count : ℕ := 216

theorem arrangement_theorem :
  let total_arrangements := number_of_people.factorial
  let front_row_without_A := (people_per_row - 1).choose 1
  let back_row_without_B := (people_per_row - 1).choose 1
  let remaining_arrangements := (number_of_people - 2).factorial
  front_row_without_A * back_row_without_B * remaining_arrangements = arrangement_count :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l1101_110185


namespace NUMINAMATH_CALUDE_chemical_solution_mixing_l1101_110121

theorem chemical_solution_mixing (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (replaced_portion : ℝ) 
  (resulting_concentration : ℝ) : 
  initial_concentration = 0.85 →
  replacement_concentration = 0.20 →
  replaced_portion = 0.6923076923076923 →
  resulting_concentration = 
    (initial_concentration * (1 - replaced_portion) + 
     replacement_concentration * replaced_portion) →
  resulting_concentration = 0.40 := by
sorry

end NUMINAMATH_CALUDE_chemical_solution_mixing_l1101_110121


namespace NUMINAMATH_CALUDE_remainder_problem_l1101_110135

theorem remainder_problem (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1101_110135


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1101_110179

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ (∀ x, f x ≠ 0) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by
  apply negation_of_existence

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1101_110179


namespace NUMINAMATH_CALUDE_mode_is_97_l1101_110167

/-- Represents a test score with its frequency -/
structure ScoreFrequency where
  score : Nat
  frequency : Nat

/-- Definition of the dataset from the stem-and-leaf plot -/
def testScores : List ScoreFrequency := [
  ⟨75, 2⟩, ⟨81, 2⟩, ⟨82, 3⟩, ⟨89, 2⟩, ⟨93, 1⟩, ⟨94, 2⟩, ⟨97, 4⟩,
  ⟨106, 1⟩, ⟨112, 2⟩, ⟨114, 3⟩, ⟨120, 1⟩
]

/-- Definition of mode: the score with the highest frequency -/
def isMode (s : ScoreFrequency) (scores : List ScoreFrequency) : Prop :=
  ∀ t ∈ scores, s.frequency ≥ t.frequency

/-- Theorem stating that 97 is the mode of the test scores -/
theorem mode_is_97 : ∃ s ∈ testScores, s.score = 97 ∧ isMode s testScores := by
  sorry

end NUMINAMATH_CALUDE_mode_is_97_l1101_110167


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1101_110160

theorem units_digit_of_seven_to_six_to_five (n : ℕ) : n = 7^(6^5) → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_six_to_five_l1101_110160


namespace NUMINAMATH_CALUDE_number_2005_location_l1101_110162

/-- The sum of the first n positive integers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The last number in the nth row of the pattern -/
def last_in_row (n : ℕ) : ℕ := n^2

/-- The first number in the nth row of the pattern -/
def first_in_row (n : ℕ) : ℕ := last_in_row (n - 1) + 1

/-- The number of elements in the nth row of the pattern -/
def elements_in_row (n : ℕ) : ℕ := 2 * n - 1

/-- The position of a number within its row in the pattern -/
def position_in_row (n : ℕ) (target : ℕ) : ℕ :=
  target - first_in_row n + 1

theorem number_2005_location :
  ∃ (i j : ℕ), i = 45 ∧ j = 20 ∧ 
  first_in_row i ≤ 2005 ∧
  2005 ≤ last_in_row i ∧
  position_in_row i 2005 = j :=
sorry

end NUMINAMATH_CALUDE_number_2005_location_l1101_110162


namespace NUMINAMATH_CALUDE_function_properties_no_zeros_l1101_110143

noncomputable section

def f (a : ℝ) (x : ℝ) := a * Real.log x - x
def g (a : ℝ) (x : ℝ) := a * Real.exp x - x

theorem function_properties (a : ℝ) (ha : a > 0) :
  (∀ x > 1, ∀ y > x, f a y < f a x) ∧
  (∃ x > 2, ∀ y > 2, g a x ≤ g a y) →
  a ∈ Set.Ioo 0 (1 / Real.exp 2) :=
sorry

theorem no_zeros (a : ℝ) (ha : a > 0) :
  (∀ x > 0, f a x ≠ 0) ∧ (∀ x, g a x ≠ 0) →
  a ∈ Set.Ioo (1 / Real.exp 1) (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_function_properties_no_zeros_l1101_110143


namespace NUMINAMATH_CALUDE_race_finish_difference_l1101_110128

/-- The time difference between two runners in a race -/
def time_difference (race_distance : ℕ) (speed1 speed2 : ℕ) : ℕ :=
  race_distance * speed2 - race_distance * speed1

/-- Theorem: In a 12-mile race, a runner with 7 min/mile speed finishes 24 minutes 
    after a runner with 5 min/mile speed -/
theorem race_finish_difference :
  time_difference 12 5 7 = 24 := by sorry

end NUMINAMATH_CALUDE_race_finish_difference_l1101_110128


namespace NUMINAMATH_CALUDE_final_ring_count_is_225_l1101_110187

/-- Calculates the final number of ornamental rings in the store after a series of transactions -/
def final_ring_count (initial_purchase : ℕ) (additional_purchase : ℕ) (final_sale : ℕ) : ℕ :=
  let initial_stock := initial_purchase / 2
  let total_stock := initial_purchase + initial_stock
  let remaining_after_first_sale := total_stock - (3 * total_stock / 4)
  let stock_after_additional_purchase := remaining_after_first_sale + additional_purchase
  stock_after_additional_purchase - final_sale

/-- The final number of ornamental rings in the store is 225 -/
theorem final_ring_count_is_225 : final_ring_count 200 300 150 = 225 := by
  sorry

end NUMINAMATH_CALUDE_final_ring_count_is_225_l1101_110187


namespace NUMINAMATH_CALUDE_jogging_track_circumference_jogging_track_circumference_value_l1101_110102

/-- The circumference of a circular jogging track where two people walking in opposite directions meet. -/
theorem jogging_track_circumference (deepak_speed wife_speed : ℝ) (meeting_time : ℝ) : ℝ :=
  let deepak_speed := 4.5
  let wife_speed := 3.75
  let meeting_time := 3.84 / 60
  2 * (deepak_speed + wife_speed) * meeting_time

/-- The circumference of the jogging track is 1.056 km. -/
theorem jogging_track_circumference_value :
  jogging_track_circumference 4.5 3.75 (3.84 / 60) = 1.056 := by
  sorry

end NUMINAMATH_CALUDE_jogging_track_circumference_jogging_track_circumference_value_l1101_110102


namespace NUMINAMATH_CALUDE_inscribed_squares_form_acute_triangle_l1101_110115

/-- Given an acute-angled triangle ABC with sides a, b, c, and inscribed squares
    with sides x, y, z, prove that x, y, z can form an acute-angled triangle. -/
theorem inscribed_squares_form_acute_triangle
  (a b c : ℝ) -- Sides of the original triangle
  (h_acute : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) -- Acute triangle condition
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) -- Positive side lengths
  (x y z : ℝ) -- Sides of the inscribed squares
  (h_x : x = (a^2 * b * c) / (a * d + b * c)) -- Definition of x
  (h_y : y = (b^2 * a * c) / (b * d + c * a)) -- Definition of y
  (h_z : z = (c^2 * a * b) / (c * d + a * b)) -- Definition of z
  (d : ℝ) -- Circumdiameter
  (h_d : d > 0) -- Positive circumdiameter
  : x^2 + y^2 > z^2 := by
  sorry

#check inscribed_squares_form_acute_triangle

end NUMINAMATH_CALUDE_inscribed_squares_form_acute_triangle_l1101_110115


namespace NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_l1101_110177

open Set

theorem solution_set_nonempty_iff_m_in_range (m : ℝ) :
  (∃ x : ℝ, |x - m| + |x + 2| < 4) ↔ m ∈ Ioo (-6) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_iff_m_in_range_l1101_110177


namespace NUMINAMATH_CALUDE_complex_magnitude_l1101_110129

/-- Given that (1+2i)/(a+bi) = 1 - i, where i is the imaginary unit and a and b are real numbers,
    prove that |a+bi| = √10/2 -/
theorem complex_magnitude (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) 
  (h2 : (1 + 2*i) / (a + b*i) = 1 - i) : 
  Complex.abs (a + b*i) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1101_110129


namespace NUMINAMATH_CALUDE_total_value_correct_l1101_110137

/-- The total value of an imported item -/
def total_value : ℝ := 2580

/-- The import tax rate -/
def tax_rate : ℝ := 0.07

/-- The tax-free threshold -/
def tax_free_threshold : ℝ := 1000

/-- The amount of import tax paid -/
def tax_paid : ℝ := 110.60

/-- Theorem stating that the total value is correct given the conditions -/
theorem total_value_correct : 
  tax_rate * (total_value - tax_free_threshold) = tax_paid := by sorry

end NUMINAMATH_CALUDE_total_value_correct_l1101_110137


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1101_110150

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 1 = 0) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1101_110150


namespace NUMINAMATH_CALUDE_no_pairs_50_75_six_pairs_50_600_l1101_110190

-- Define the function to count pairs satisfying the conditions
def countPairs (gcd : Nat) (lcm : Nat) : Nat :=
  (Finset.filter (fun p : Nat × Nat => 
    p.1.gcd p.2 = gcd ∧ p.1.lcm p.2 = lcm) (Finset.product (Finset.range (lcm + 1)) (Finset.range (lcm + 1)))).card

-- Theorem for the first part
theorem no_pairs_50_75 : countPairs 50 75 = 0 := by sorry

-- Theorem for the second part
theorem six_pairs_50_600 : countPairs 50 600 = 6 := by sorry

end NUMINAMATH_CALUDE_no_pairs_50_75_six_pairs_50_600_l1101_110190


namespace NUMINAMATH_CALUDE_company_gender_ratio_l1101_110123

/-- Represents the number of employees of each gender in a company -/
structure Company where
  male : ℕ
  female : ℕ

/-- The ratio of male to female employees -/
def genderRatio (c : Company) : ℚ :=
  c.male / c.female

theorem company_gender_ratio (c : Company) :
  c.male = 189 ∧ 
  genderRatio {male := c.male + 3, female := c.female} = 8 / 9 →
  genderRatio c = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_company_gender_ratio_l1101_110123


namespace NUMINAMATH_CALUDE_sequence_inequality_l1101_110101

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -2 * n.val ^ 2 + 3 * n.val

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := -4 * n.val + 5

/-- Theorem stating the relationship between na_n, S_n, and na_1 for n ≥ 2 -/
theorem sequence_inequality (n : ℕ+) (h : 2 ≤ n.val) :
  (n.val : ℤ) * a n < S n ∧ S n < (n.val : ℤ) * a 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1101_110101


namespace NUMINAMATH_CALUDE_derek_savings_l1101_110122

theorem derek_savings (P : ℚ) : P * 2^11 = 4096 → P = 2 := by
  sorry

end NUMINAMATH_CALUDE_derek_savings_l1101_110122


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1101_110186

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1101_110186
