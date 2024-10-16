import Mathlib

namespace NUMINAMATH_CALUDE_larger_number_proof_l577_57757

/-- Given two positive integers with specific HCF and LCM, prove the larger one is 391 -/
theorem larger_number_proof (a b : ℕ+) 
  (hcf_cond : Nat.gcd a b = 23)
  (lcm_cond : Nat.lcm a b = 23 * 13 * 17) :
  max a b = 391 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l577_57757


namespace NUMINAMATH_CALUDE_rectangular_piece_too_large_l577_57774

theorem rectangular_piece_too_large (square_area : ℝ) (rect_area : ℝ) (ratio_length : ℝ) (ratio_width : ℝ) :
  square_area = 400 →
  rect_area = 300 →
  ratio_length = 3 →
  ratio_width = 2 →
  ∃ (rect_length : ℝ), 
    rect_length * (rect_length * ratio_width / ratio_length) = rect_area ∧
    rect_length > Real.sqrt square_area :=
by sorry

end NUMINAMATH_CALUDE_rectangular_piece_too_large_l577_57774


namespace NUMINAMATH_CALUDE_base_n_representation_l577_57726

theorem base_n_representation (n : ℕ) : 
  n > 0 ∧ 
  (∃ a b c : ℕ, 
    a < n ∧ b < n ∧ c < n ∧ 
    1998 = a * n^2 + b * n + c ∧ 
    a + b + c = 24) → 
  n = 15 ∨ n = 22 ∨ n = 43 := by
sorry

end NUMINAMATH_CALUDE_base_n_representation_l577_57726


namespace NUMINAMATH_CALUDE_smallest_n_inequality_l577_57718

theorem smallest_n_inequality (x y z w : ℝ) : 
  ∃ (n : ℕ), n = 4 ∧ 
  (∀ (m : ℕ), m < n → ∃ (a b c d : ℝ), (a^2 + b^2 + c^2 + d^2)^2 > m*(a^4 + b^4 + c^4 + d^4)) ∧
  (x^2 + y^2 + z^2 + w^2)^2 ≤ n*(x^4 + y^4 + z^4 + w^4) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_inequality_l577_57718


namespace NUMINAMATH_CALUDE_log_product_equal_twelve_l577_57742

theorem log_product_equal_twelve :
  Real.log 9 / Real.log 2 * (Real.log 5 / Real.log 3) * (Real.log 8 / Real.log (Real.sqrt 5)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equal_twelve_l577_57742


namespace NUMINAMATH_CALUDE_equivalent_representations_l577_57761

theorem equivalent_representations : 
  ∀ (a b c d e f : ℚ),
  (a = 9/18) → 
  (b = 1/2) → 
  (c = 27/54) → 
  (d = 1/2) → 
  (e = 1/2) → 
  (f = 1/2) → 
  (a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f) := by
  sorry

#check equivalent_representations

end NUMINAMATH_CALUDE_equivalent_representations_l577_57761


namespace NUMINAMATH_CALUDE_probability_sum_10_l577_57721

/-- The number of faces on a standard die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The target sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (sum of 10) -/
def favorableOutcomes : ℕ := 24

/-- The probability of rolling a sum of 10 with three standard six-sided dice -/
theorem probability_sum_10 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_10_l577_57721


namespace NUMINAMATH_CALUDE_divisibility_by_7_implies_37_l577_57724

/-- Given a natural number n, returns the number consisting of n repeated digits of 1 -/
def repeatedOnes (n : ℕ) : ℕ := 
  (10^n - 1) / 9

/-- Theorem: If a number consisting of n repeated digits of 1 is divisible by 7, 
    then it is also divisible by 37 -/
theorem divisibility_by_7_implies_37 (n : ℕ) :
  (repeatedOnes n) % 7 = 0 → (repeatedOnes n) % 37 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_7_implies_37_l577_57724


namespace NUMINAMATH_CALUDE_flowers_in_basket_is_four_l577_57798

/-- The number of flowers in each basket after planting, growth, and distribution -/
def flowers_per_basket (daughters : ℕ) (flowers_per_daughter : ℕ) (new_flowers : ℕ) (dead_flowers : ℕ) (num_baskets : ℕ) : ℕ :=
  let initial_flowers := daughters * flowers_per_daughter
  let total_flowers := initial_flowers + new_flowers
  let remaining_flowers := total_flowers - dead_flowers
  remaining_flowers / num_baskets

/-- Theorem stating that under the given conditions, each basket will contain 4 flowers -/
theorem flowers_in_basket_is_four :
  flowers_per_basket 2 5 20 10 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_basket_is_four_l577_57798


namespace NUMINAMATH_CALUDE_shipment_size_l577_57741

/-- The total number of novels in the shipment -/
def total_novels : ℕ := 300

/-- The fraction of novels displayed in the storefront -/
def display_fraction : ℚ := 30 / 100

/-- The number of novels in the storage room -/
def storage_novels : ℕ := 210

/-- Theorem stating that the total number of novels is 300 -/
theorem shipment_size :
  total_novels = 300 ∧
  display_fraction = 30 / 100 ∧
  storage_novels = 210 ∧
  (1 - display_fraction) * total_novels = storage_novels :=
by sorry

end NUMINAMATH_CALUDE_shipment_size_l577_57741


namespace NUMINAMATH_CALUDE_divisor_calculation_l577_57729

theorem divisor_calculation (dividend : Float) (quotient : Float) (h1 : dividend = 0.0204) (h2 : quotient = 0.0012000000000000001) :
  dividend / quotient = 17 := by
  sorry

end NUMINAMATH_CALUDE_divisor_calculation_l577_57729


namespace NUMINAMATH_CALUDE_exponent_rule_equality_l577_57700

theorem exponent_rule_equality (x : ℝ) (m : ℤ) (h : x ≠ 0) :
  (x^3)^m / (x^m)^2 = x^m :=
sorry

end NUMINAMATH_CALUDE_exponent_rule_equality_l577_57700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l577_57799

/-- An arithmetic sequence {a_n} -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A monotonically increasing sequence -/
def monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_mono : monotonically_increasing a)
  (h_sum : a 1 + a 2 + a 3 = 21)
  (h_prod : a 1 * a 2 * a 3 = 231) :
  (a 2 = 7) ∧ (∀ n : ℕ, a n = 4 * n - 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l577_57799


namespace NUMINAMATH_CALUDE_consecutive_even_negative_integers_sum_l577_57732

theorem consecutive_even_negative_integers_sum (n m : ℤ) : 
  n < 0 ∧ m < 0 ∧ 
  Even n ∧ Even m ∧ 
  m = n + 2 ∧ 
  n * m = 2496 → 
  n + m = -102 := by sorry

end NUMINAMATH_CALUDE_consecutive_even_negative_integers_sum_l577_57732


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_of_4500_l577_57765

/-- The number of perfect square factors of 4500 -/
def perfectSquareFactorsOf4500 : ℕ :=
  -- Define the number of perfect square factors of 4500
  -- We don't implement the calculation here, just define it
  -- The actual value will be proven to be 8
  sorry

/-- Theorem: The number of perfect square factors of 4500 is 8 -/
theorem count_perfect_square_factors_of_4500 :
  perfectSquareFactorsOf4500 = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_of_4500_l577_57765


namespace NUMINAMATH_CALUDE_square_and_product_l577_57782

theorem square_and_product (x : ℤ) (h : x^2 = 1764) : x = 42 ∧ (x + 2) * (x - 2) = 1760 := by
  sorry

end NUMINAMATH_CALUDE_square_and_product_l577_57782


namespace NUMINAMATH_CALUDE_log_expressibility_l577_57758

-- Define the given logarithm values
noncomputable def log10_5 : ℝ := 0.6990
noncomputable def log10_7 : ℝ := 0.8451

-- Define a function to represent expressibility using given logarithms
def expressible (x : ℝ) : Prop :=
  ∃ (a b c : ℚ), x = a * log10_5 + b * log10_7 + c

-- Theorem statement
theorem log_expressibility :
  (¬ expressible (Real.log 27 / Real.log 10)) ∧
  (¬ expressible (Real.log 21 / Real.log 10)) ∧
  (expressible (Real.log (Real.sqrt 35) / Real.log 10)) ∧
  (expressible (Real.log 1000 / Real.log 10)) ∧
  (expressible (Real.log 0.2 / Real.log 10)) :=
sorry

end NUMINAMATH_CALUDE_log_expressibility_l577_57758


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l577_57728

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 + 2 * Real.sqrt 2 * m + 1 = 0) ∧ 
  (n^2 + 2 * Real.sqrt 2 * n + 1 = 0) → 
  Real.sqrt (m^2 + n^2 + 3*m*n) = 3 := by
  sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l577_57728


namespace NUMINAMATH_CALUDE_arithmetic_sequence_roots_iff_l577_57725

/-- A cubic equation with real coefficients -/
structure CubicEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate for a cubic equation having three real roots in arithmetic sequence -/
def has_arithmetic_sequence_roots (eq : CubicEquation) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧
    x₃ - x₂ = x₂ - x₁ ∧
    x₁^3 + eq.a * x₁^2 + eq.b * x₁ + eq.c = 0 ∧
    x₂^3 + eq.a * x₂^2 + eq.b * x₂ + eq.c = 0 ∧
    x₃^3 + eq.a * x₃^2 + eq.b * x₃ + eq.c = 0

/-- The necessary and sufficient conditions for a cubic equation to have three real roots in arithmetic sequence -/
theorem arithmetic_sequence_roots_iff (eq : CubicEquation) :
  has_arithmetic_sequence_roots eq ↔ 
  (2 * eq.a^3 - 9 * eq.a * eq.b + 27 * eq.c = 0) ∧ (eq.a^2 - 3 * eq.b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_roots_iff_l577_57725


namespace NUMINAMATH_CALUDE_parallelogram_angle_E_l577_57770

structure Parallelogram :=
  (E F G H : Point)

def angle_FGH (p : Parallelogram) : ℝ := sorry
def angle_E (p : Parallelogram) : ℝ := sorry

theorem parallelogram_angle_E (p : Parallelogram) 
  (h : angle_FGH p = 70) : angle_E p = 110 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_E_l577_57770


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l577_57748

/-- A line passing through a fixed point for all values of k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 3 : ℝ) - 1 + 1 = 3 * k := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l577_57748


namespace NUMINAMATH_CALUDE_unique_solution_is_five_l577_57722

/-- The function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The theorem stating that x = 5 is the unique solution to the equation -/
theorem unique_solution_is_five :
  ∃! x : ℝ, 2 * (f x) - 11 = f (x - 2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_five_l577_57722


namespace NUMINAMATH_CALUDE_no_cube_sum_three_consecutive_squares_l577_57783

theorem no_cube_sum_three_consecutive_squares :
  ¬ ∃ (x y : ℤ), x^3 = (y-1)^2 + y^2 + (y+1)^2 :=
by sorry

end NUMINAMATH_CALUDE_no_cube_sum_three_consecutive_squares_l577_57783


namespace NUMINAMATH_CALUDE_train_speed_problem_l577_57701

/-- Represents the speeds of the three trains -/
structure TrainSpeeds where
  slower : ℝ
  faster : ℝ
  perpendicular : ℝ

/-- The problem setup and solution -/
theorem train_speed_problem (total_distance : ℝ) (speed_difference : ℝ) (time : ℝ) 
  (h1 : total_distance = 450)
  (h2 : speed_difference = 6)
  (h3 : time = 5)
  : ∃ (speeds : TrainSpeeds),
    speeds.slower = 42 ∧ 
    speeds.faster = 48 ∧ 
    speeds.perpendicular = 45 ∧
    speeds.faster = speeds.slower + speed_difference ∧
    speeds.slower * time + speeds.faster * time = total_distance ∧
    speeds.perpendicular * time = total_distance / 2 :=
by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l577_57701


namespace NUMINAMATH_CALUDE_sofia_survey_l577_57775

theorem sofia_survey (liked : ℕ) (disliked : ℕ) (h1 : liked = 235) (h2 : disliked = 165) :
  liked + disliked = 400 := by
  sorry

end NUMINAMATH_CALUDE_sofia_survey_l577_57775


namespace NUMINAMATH_CALUDE_kid_ticket_price_l577_57704

theorem kid_ticket_price 
  (total_sales : ℕ) 
  (adult_price : ℕ) 
  (num_adults : ℕ) 
  (total_people : ℕ) : 
  total_sales = 3864 ∧ 
  adult_price = 28 ∧ 
  num_adults = 51 ∧ 
  total_people = 254 → 
  (total_sales - num_adults * adult_price) / (total_people - num_adults) = 12 := by
  sorry

#eval (3864 - 51 * 28) / (254 - 51)

end NUMINAMATH_CALUDE_kid_ticket_price_l577_57704


namespace NUMINAMATH_CALUDE_chalk_pieces_count_l577_57760

/-- Given a box capacity and number of full boxes, calculates the total number of chalk pieces -/
def total_chalk_pieces (box_capacity : ℕ) (full_boxes : ℕ) : ℕ :=
  box_capacity * full_boxes

/-- Proves that the total number of chalk pieces is 3492 -/
theorem chalk_pieces_count :
  let box_capacity := 18
  let full_boxes := 194
  total_chalk_pieces box_capacity full_boxes = 3492 := by
  sorry

end NUMINAMATH_CALUDE_chalk_pieces_count_l577_57760


namespace NUMINAMATH_CALUDE_locus_of_A_is_hyperbola_l577_57737

/-- Triangle ABC with special properties -/
structure SpecialTriangle where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- H is the orthocenter
  H : ℝ × ℝ
  -- G is the centroid
  G : ℝ × ℝ
  -- B and C are fixed points
  B_fixed : B.1 = -a ∧ B.2 = 0
  C_fixed : C.1 = a ∧ C.2 = 0
  -- Midpoint of HG lies on BC
  HG_midpoint_on_BC : ∃ m : ℝ, (H.1 + G.1) / 2 = m ∧ (H.2 + G.2) / 2 = 0
  -- G is the centroid
  G_is_centroid : G.1 = (A.1 + B.1 + C.1) / 3 ∧ G.2 = (A.2 + B.2 + C.2) / 3
  -- H is the orthocenter
  H_is_orthocenter : (A.1 - B.1) * (H.1 - C.1) + (A.2 - B.2) * (H.2 - C.2) = 0 ∧
                     (B.1 - C.1) * (H.1 - A.1) + (B.2 - C.2) * (H.2 - A.2) = 0

/-- The locus of A in a special triangle is a hyperbola -/
theorem locus_of_A_is_hyperbola (t : SpecialTriangle) : 
  t.A.1^2 - t.A.2^2/3 = a^2 := by sorry

end NUMINAMATH_CALUDE_locus_of_A_is_hyperbola_l577_57737


namespace NUMINAMATH_CALUDE_ellipse_properties_l577_57708

-- Define the ellipse and its properties
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_def : c = Real.sqrt (a^2 - b^2)
  h_e_def : e = c / a

-- Define points and line
def F₁ (E : Ellipse) : ℝ × ℝ := (-E.c, 0)
def F₂ (E : Ellipse) : ℝ × ℝ := (E.c, 0)

-- Define the properties we want to prove
def perimeter_ABF₂ (E : Ellipse) (A B : ℝ × ℝ) : ℝ := sorry

def dot_product (v w : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem ellipse_properties (E : Ellipse) (A B : ℝ × ℝ) (h_A_on_C h_B_on_C : (A.1^2 / E.a^2) + (A.2^2 / E.b^2) = 1) 
  (h_l : ∃ (t : ℝ), A = F₁ E + t • (B - F₁ E)) :
  (perimeter_ABF₂ E A B = 4 * E.a) ∧ 
  (dot_product (A - F₁ E) (A - F₂ E) = 5 * E.c^2 → E.e ≥ Real.sqrt 7 / 7) ∧
  (dot_product (A - F₁ E) (A - F₂ E) = 6 * E.c^2 → E.e ≤ Real.sqrt 7 / 7) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l577_57708


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_properties_l577_57735

theorem quadratic_equation_roots_properties : ∃ (r₁ r₂ : ℝ),
  (r₁^2 - 6*r₁ + 8 = 0) ∧
  (r₂^2 - 6*r₂ + 8 = 0) ∧
  (r₁ ≠ r₂) ∧
  (|r₁ - r₂| = 2) ∧
  (|r₁| + |r₂| = 6) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_properties_l577_57735


namespace NUMINAMATH_CALUDE_candy_remaining_l577_57702

theorem candy_remaining (initial : ℕ) (talitha solomon maya : ℕ) 
  (h1 : initial = 572)
  (h2 : talitha = 183)
  (h3 : solomon = 238)
  (h4 : maya = 127) :
  initial - (talitha + solomon + maya) = 24 := by
  sorry

end NUMINAMATH_CALUDE_candy_remaining_l577_57702


namespace NUMINAMATH_CALUDE_exactly_one_prob_l577_57784

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.4

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.5

/-- The events A and B are independent -/
axiom independent : True

/-- The probability that exactly one of A or B occurs -/
def prob_exactly_one : ℝ := (1 - prob_A) * prob_B + prob_A * (1 - prob_B)

/-- Theorem: The probability that exactly one of A or B occurs is 0.5 -/
theorem exactly_one_prob : prob_exactly_one = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_prob_l577_57784


namespace NUMINAMATH_CALUDE_wang_liang_is_president_l577_57790

-- Define the students and positions
inductive Student : Type
| ZhangQiang : Student
| LiMing : Student
| WangLiang : Student

inductive Position : Type
| President : Position
| LifeDelegate : Position
| StudyDelegate : Position

-- Define the council as a function from Position to Student
def Council := Position → Student

-- Define the predictions
def PredictionA (c : Council) : Prop :=
  c Position.President = Student.ZhangQiang ∧ c Position.LifeDelegate = Student.LiMing

def PredictionB (c : Council) : Prop :=
  c Position.President = Student.WangLiang ∧ c Position.LifeDelegate = Student.ZhangQiang

def PredictionC (c : Council) : Prop :=
  c Position.President = Student.LiMing ∧ c Position.StudyDelegate = Student.ZhangQiang

-- Define the condition that each prediction is half correct
def HalfCorrectPredictions (c : Council) : Prop :=
  (PredictionA c = true) = (PredictionA c = false) ∧
  (PredictionB c = true) = (PredictionB c = false) ∧
  (PredictionC c = true) = (PredictionC c = false)

-- Theorem statement
theorem wang_liang_is_president :
  ∀ c : Council, HalfCorrectPredictions c → c Position.President = Student.WangLiang :=
by
  sorry

end NUMINAMATH_CALUDE_wang_liang_is_president_l577_57790


namespace NUMINAMATH_CALUDE_equation_solution_l577_57711

theorem equation_solution (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) 
  (h : 3 / x + 2 / y = 1 / 3) : 
  x = 9 * y / (y - 6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l577_57711


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l577_57716

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  first_term : ℝ
  common_ratio : ℝ

/-- Get the nth term of a geometric progression -/
def GeometricProgression.nth_term (gp : GeometricProgression) (n : ℕ) : ℝ :=
  gp.first_term * gp.common_ratio ^ (n - 1)

theorem geometric_progression_solution :
  ∀ (gp : GeometricProgression),
    (gp.nth_term 1 * gp.nth_term 2 * gp.nth_term 3 = 1728) →
    (gp.nth_term 1 + gp.nth_term 2 + gp.nth_term 3 = 63) →
    ((gp.first_term = 3 ∧ gp.common_ratio = 4) ∨ 
     (gp.first_term = 48 ∧ gp.common_ratio = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l577_57716


namespace NUMINAMATH_CALUDE_smallest_number_in_sample_l577_57755

/-- Systematic sampling function that returns the smallest number given the parameters -/
def systematicSampling (totalSchools : ℕ) (sampleSize : ℕ) (highestDrawn : ℕ) : ℕ :=
  let interval := totalSchools / sampleSize
  highestDrawn - (sampleSize - 1) * interval

/-- Theorem stating the smallest number drawn in the specific scenario -/
theorem smallest_number_in_sample (totalSchools : ℕ) (sampleSize : ℕ) (highestDrawn : ℕ) 
    (h1 : totalSchools = 32)
    (h2 : sampleSize = 8)
    (h3 : highestDrawn = 31) :
  systematicSampling totalSchools sampleSize highestDrawn = 3 := by
  sorry

#eval systematicSampling 32 8 31

end NUMINAMATH_CALUDE_smallest_number_in_sample_l577_57755


namespace NUMINAMATH_CALUDE_sphere_in_cylindrical_hole_l577_57797

theorem sphere_in_cylindrical_hole (r : ℝ) (h : ℝ) :
  h = 2 ∧ 
  6^2 + (r - h)^2 = r^2 →
  r = 10 ∧ 4 * Real.pi * r^2 = 400 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cylindrical_hole_l577_57797


namespace NUMINAMATH_CALUDE_line_direction_vector_l577_57719

/-- Given a line with direction vector (a, -2) passing through points (-3, 7) and (2, -1),
    prove that a = 5/4 -/
theorem line_direction_vector (a : ℝ) : 
  (∃ t : ℝ, (2 : ℝ) = -3 + t * a ∧ (-1 : ℝ) = 7 + t * (-2)) → a = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l577_57719


namespace NUMINAMATH_CALUDE_root_implies_inequality_l577_57745

theorem root_implies_inequality (a b : ℝ) 
  (h : (a + b + a) * (a + b + b) = 9) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_inequality_l577_57745


namespace NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l577_57796

theorem abs_sum_lt_sum_abs_when_product_negative (a b : ℝ) :
  a * b < 0 → |a + b| < |a| + |b| := by sorry

end NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l577_57796


namespace NUMINAMATH_CALUDE_complex_multiplication_l577_57746

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication :
  (1 + i) * i = -1 + i :=
sorry

end NUMINAMATH_CALUDE_complex_multiplication_l577_57746


namespace NUMINAMATH_CALUDE_sum_g_formula_l577_57705

/-- g(n) is the largest odd divisor of the positive integer n -/
def g (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of g(k) for k from 1 to 2^n -/
def sum_g (n : ℕ) : ℕ+ :=
  sorry

/-- Theorem: The sum of g(k) for k from 1 to 2^n equals (4^n + 5) / 3 -/
theorem sum_g_formula (n : ℕ) : 
  (sum_g n : ℚ) = (4^n + 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_formula_l577_57705


namespace NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l577_57730

theorem remainder_31_pow_31_plus_31_mod_32 : (31^31 + 31) % 32 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l577_57730


namespace NUMINAMATH_CALUDE_simplify_expression_l577_57762

theorem simplify_expression :
  ∃ (C : ℝ), C = 2^(1 + Real.sqrt 2) ∧
  (Real.sqrt 3 - 1)^(1 - Real.sqrt 2) / (Real.sqrt 3 + 1)^(1 + Real.sqrt 2) = (4 - 2 * Real.sqrt 3) / C :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l577_57762


namespace NUMINAMATH_CALUDE_product_closed_in_P_l577_57776

/-- The set of perfect squares -/
def P : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^2}

/-- The theorem stating that the product of two elements in P is also in P -/
theorem product_closed_in_P (a b : ℕ) (ha : a ∈ P) (hb : b ∈ P) : a * b ∈ P := by
  sorry

#check product_closed_in_P

end NUMINAMATH_CALUDE_product_closed_in_P_l577_57776


namespace NUMINAMATH_CALUDE_exactly_nine_heads_probability_l577_57751

/-- The probability of getting heads when flipping the biased coin -/
def p : ℚ := 3/4

/-- The number of coin flips -/
def n : ℕ := 12

/-- The number of heads we want to get -/
def k : ℕ := 9

/-- The probability of getting exactly k heads in n flips of a coin with probability p of heads -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem exactly_nine_heads_probability :
  binomial_probability n k p = 4330260/16777216 := by
  sorry

end NUMINAMATH_CALUDE_exactly_nine_heads_probability_l577_57751


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l577_57756

theorem unique_solution_quadratic (a c : ℝ) : 
  (∃! x, a * x^2 + 30 * x + c = 0) →  -- exactly one solution
  (a + c = 35) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = (35 - 5 * Real.sqrt 13) / 2 ∧ c = (35 + 5 * Real.sqrt 13) / 2) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l577_57756


namespace NUMINAMATH_CALUDE_riverside_total_multiple_of_five_l577_57753

/-- Represents the population of animals and people in Riverside --/
structure Riverside where
  people : ℕ
  horses : ℕ
  sheep : ℕ
  cows : ℕ
  ducks : ℕ

/-- The conditions given in the problem --/
def valid_riverside (r : Riverside) : Prop :=
  r.people = 5 * r.horses ∧
  r.sheep = 6 * r.cows ∧
  r.ducks = 4 * r.people ∧
  r.sheep * 2 = r.ducks

/-- The theorem states that the total population in a valid Riverside setup is always a multiple of 5 --/
theorem riverside_total_multiple_of_five (r : Riverside) (h : valid_riverside r) :
  ∃ k : ℕ, r.people + r.horses + r.sheep + r.cows + r.ducks = 5 * k :=
sorry

end NUMINAMATH_CALUDE_riverside_total_multiple_of_five_l577_57753


namespace NUMINAMATH_CALUDE_new_room_size_l577_57734

theorem new_room_size (bedroom_size : ℝ) (bathroom_size : ℝ) 
  (h1 : bedroom_size = 309) 
  (h2 : bathroom_size = 150) : 
  2 * (bedroom_size + bathroom_size) = 918 := by
  sorry

end NUMINAMATH_CALUDE_new_room_size_l577_57734


namespace NUMINAMATH_CALUDE_email_count_theorem_l577_57713

/-- Calculates the total number of emails received in a month with changing email rates --/
def total_emails (days : ℕ) (initial_rate : ℕ) (increase : ℕ) : ℕ :=
  let half_days := days / 2
  let first_half := initial_rate * half_days
  let second_half := (initial_rate + increase) * (days - half_days)
  first_half + second_half

/-- Theorem stating that given the conditions, the total emails received is 675 --/
theorem email_count_theorem :
  total_emails 30 20 5 = 675 := by
  sorry

end NUMINAMATH_CALUDE_email_count_theorem_l577_57713


namespace NUMINAMATH_CALUDE_convex_polygon_diagonals_l577_57786

-- Define a convex polygon type
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool
  interior_angle : ℝ
  all_angles_equal : Bool

-- Theorem statement
theorem convex_polygon_diagonals 
  (p : ConvexPolygon) 
  (h1 : p.is_convex = true) 
  (h2 : p.interior_angle = 150) 
  (h3 : p.all_angles_equal = true) : 
  (p.sides * (p.sides - 3)) / 2 = 54 := by
sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonals_l577_57786


namespace NUMINAMATH_CALUDE_total_figures_is_44_l577_57773

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 11

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 4

/-- The total number of action figures that can fit on all shelves. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

/-- Theorem stating that the total number of action figures is 44. -/
theorem total_figures_is_44 : total_figures = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_figures_is_44_l577_57773


namespace NUMINAMATH_CALUDE_mars_mission_cost_share_l577_57781

/-- The total cost in billions of dollars to send a person to Mars -/
def total_cost : ℝ := 30

/-- The number of people in millions sharing the cost -/
def number_of_people : ℝ := 300

/-- Each person's share of the cost in dollars -/
def cost_per_person : ℝ := 100

/-- Theorem stating that if the total cost in billions of dollars is shared equally among the given number of people in millions, each person's share is the specified amount in dollars -/
theorem mars_mission_cost_share : 
  (total_cost * 1000) / number_of_people = cost_per_person := by
  sorry

end NUMINAMATH_CALUDE_mars_mission_cost_share_l577_57781


namespace NUMINAMATH_CALUDE_triangle_angle_45_l577_57787

/-- Given a triangle with sides a, b, c, perimeter 2s, and area T,
    if T + (ab/2) = s(s-c), then the angle opposite side c is 45°. -/
theorem triangle_angle_45 (a b c s T : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
    (h_perimeter : a + b + c = 2 * s) (h_area : T > 0)
    (h_equation : T + (a * b / 2) = s * (s - c)) :
    let γ := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
    γ = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_45_l577_57787


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l577_57738

def y : ℕ := 2^5 * 3^5 * 4^5 * 5^5 * 6^5 * 7^5 * 8^5 * 9^5

theorem smallest_perfect_square_multiplier (k : ℕ) : 
  (∀ m : ℕ, m < 105 → ¬∃ n : ℕ, m * y = n^2) ∧ 
  (∃ n : ℕ, 105 * y = n^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l577_57738


namespace NUMINAMATH_CALUDE_invisible_dots_count_l577_57739

/-- The total number of dots on a single die -/
def dots_per_die : ℕ := 21

/-- The sum of visible numbers on the stacked dice -/
def visible_sum : ℕ := 2 + 2 + 3 + 4 + 5 + 5 + 6 + 6

/-- The number of dice stacked -/
def num_dice : ℕ := 3

/-- The number of visible faces -/
def visible_faces : ℕ := 8

theorem invisible_dots_count : 
  num_dice * dots_per_die - visible_sum = 30 := by
sorry

end NUMINAMATH_CALUDE_invisible_dots_count_l577_57739


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l577_57771

def newspaper_earnings : ℕ := 16
def car_washing_earnings : ℕ := 74
def lawn_mowing_earnings : ℕ := 45
def lemonade_earnings : ℕ := 22
def yard_work_earnings : ℕ := 30

theorem fred_weekend_earnings :
  newspaper_earnings + car_washing_earnings + lawn_mowing_earnings + lemonade_earnings + yard_work_earnings = 187 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l577_57771


namespace NUMINAMATH_CALUDE_min_distance_sum_l577_57733

/-- Given points A and B, and a point P on a circle, prove the minimum value of |PA|^2 + |PB|^2 -/
theorem min_distance_sum (A B P : ℝ × ℝ) : 
  A = (-2, 0) →
  B = (2, 0) →
  (P.1 - 3)^2 + (P.2 - 4)^2 = 4 →
  (P.1 + 2)^2 + P.2^2 + (P.1 - 2)^2 + P.2^2 ≥ 26 := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_min_distance_sum_l577_57733


namespace NUMINAMATH_CALUDE_ellipse_properties_l577_57749

/-- Properties of the ellipse 9x^2 + y^2 = 81 -/
theorem ellipse_properties :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 81}
  ∃ (major_axis minor_axis eccentricity : ℝ) 
    (foci_y vertex_y vertex_x : ℝ),
    -- Length of major axis
    major_axis = 18 ∧
    -- Length of minor axis
    minor_axis = 6 ∧
    -- Eccentricity
    eccentricity = 2 * Real.sqrt 2 / 3 ∧
    -- Foci coordinates
    foci_y = 6 * Real.sqrt 2 ∧
    (0, foci_y) ∈ ellipse ∧ (0, -foci_y) ∈ ellipse ∧
    -- Vertex coordinates
    vertex_y = 9 ∧ vertex_x = 3 ∧
    (0, vertex_y) ∈ ellipse ∧ (0, -vertex_y) ∈ ellipse ∧
    (vertex_x, 0) ∈ ellipse ∧ (-vertex_x, 0) ∈ ellipse :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l577_57749


namespace NUMINAMATH_CALUDE_july_birth_percentage_l577_57744

def total_people : ℕ := 100
def born_in_july : ℕ := 13

theorem july_birth_percentage :
  (born_in_july : ℚ) / total_people * 100 = 13 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l577_57744


namespace NUMINAMATH_CALUDE_triangle_third_sides_l577_57767

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define similarity between triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.a / t2.a = k ∧ 
    t1.b / t2.b = k ∧ 
    t1.c / t2.c = k

theorem triangle_third_sides 
  (t1 t2 : Triangle) 
  (h_similar : similar t1 t2) 
  (h_not_congruent : t1 ≠ t2) 
  (h_t1_sides : t1.a = 12 ∧ t1.b = 18) 
  (h_t2_sides : t2.a = 12 ∧ t2.b = 18) : 
  (t1.c = 27/2 ∧ t2.c = 8) ∨ (t1.c = 8 ∧ t2.c = 27/2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_sides_l577_57767


namespace NUMINAMATH_CALUDE_intersection_points_range_l577_57794

-- Define the functions
def f (x : ℝ) : ℝ := 2 * x^3 + 1
def g (b : ℝ) (x : ℝ) : ℝ := 3 * x^2 - b

-- Define the property of having three distinct intersection points
def has_three_distinct_intersections (b : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f x₁ = g b x₁ ∧ f x₂ = g b x₂ ∧ f x₃ = g b x₃

-- State the theorem
theorem intersection_points_range :
  ∀ b : ℝ, has_three_distinct_intersections b ↔ -1 < b ∧ b < 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_range_l577_57794


namespace NUMINAMATH_CALUDE_distribute_five_identical_books_to_three_students_l577_57727

/-- The number of ways to distribute n identical objects to k recipients, 
    where each recipient receives exactly one object. -/
def distribute_identical (n k : ℕ) : ℕ :=
  if n = k then 1 else 0

/-- Theorem: There is only one way to distribute 5 identical books to 3 students, 
    with each student receiving one book. -/
theorem distribute_five_identical_books_to_three_students :
  distribute_identical 5 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_identical_books_to_three_students_l577_57727


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l577_57788

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2015th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a5 : a 5 = 6) : 
  a 2015 = 2016 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2015th_term_l577_57788


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l577_57754

theorem mean_proportional_problem (x : ℝ) :
  (Real.sqrt (x * 100) = 90.5) → x = 81.9025 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l577_57754


namespace NUMINAMATH_CALUDE_orthocenter_ratio_l577_57717

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  -- Add any necessary conditions for a valid triangle

-- Define the length of a side
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Define an altitude of a triangle
def altitude (A B C P : ℝ × ℝ) : Prop := sorry

-- Define the orthocenter of a triangle
def orthocenter (X Y Z H : ℝ × ℝ) : Prop := sorry

-- Main theorem
theorem orthocenter_ratio {X Y Z P H : ℝ × ℝ} :
  Triangle X Y Z →
  side_length Y Z = 5 →
  side_length X Z = 4 * Real.sqrt 2 →
  angle_measure X Z Y = π / 4 →
  altitude X Y Z P →
  orthocenter X Y Z H →
  (side_length X H) / (side_length H P) = 3 := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_ratio_l577_57717


namespace NUMINAMATH_CALUDE_cookie_radius_l577_57712

/-- The equation of the cookie boundary -/
def cookie_boundary (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 4

/-- The cookie is a circle -/
def is_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∀ x y : ℝ, cookie_boundary x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem cookie_radius :
  ∃ center : ℝ × ℝ, is_circle center 3 :=
sorry

end NUMINAMATH_CALUDE_cookie_radius_l577_57712


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l577_57778

theorem sin_alpha_plus_pi_fourth (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) (h3 : Real.sin (2 * α) = 1 / 2) : 
  Real.sin (α + Real.pi / 4) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l577_57778


namespace NUMINAMATH_CALUDE_smurf_score_difference_l577_57703

/-- The number of Smurfs in the village -/
def total_smurfs : ℕ := 45

/-- The number of top and bottom Smurfs with known average scores -/
def known_scores_count : ℕ := 25

/-- The average score of the top 25 Smurfs -/
def top_average : ℚ := 93

/-- The average score of the bottom 25 Smurfs -/
def bottom_average : ℚ := 89

/-- The number of top and bottom Smurfs we're comparing -/
def comparison_count : ℕ := 20

/-- The theorem stating the difference between top and bottom scores -/
theorem smurf_score_difference :
  (top_average * known_scores_count - bottom_average * known_scores_count : ℚ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_smurf_score_difference_l577_57703


namespace NUMINAMATH_CALUDE_angle_between_vectors_l577_57795

def a : ℝ × ℝ := (1, 1)

theorem angle_between_vectors (b : ℝ × ℝ) 
  (h : (4 * a.1, 4 * a.2) + b = (4, 2)) : 
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l577_57795


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l577_57706

/-- The length of the real axis of a hyperbola with equation x²/3 - y²/6 = 1 is 2√3 -/
theorem hyperbola_real_axis_length : 
  ∃ (f : ℝ × ℝ → ℝ), 
    (∀ x y, f (x, y) = x^2 / 3 - y^2 / 6) ∧ 
    (∃ a : ℝ, a > 0 ∧ (∀ x y, f (x, y) = 1 → x^2 / a^2 - y^2 / (2*a^2) = 1) ∧ 2*a = 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l577_57706


namespace NUMINAMATH_CALUDE_work_problem_solution_l577_57777

/-- Proves that given the conditions of the work problem, the daily wage of worker c is 115 --/
theorem work_problem_solution (a b c : ℕ) : 
  (a : ℚ) / 3 = (b : ℚ) / 4 ∧ (a : ℚ) / 3 = (c : ℚ) / 5 →  -- daily wages ratio
  6 * a + 9 * b + 4 * c = 1702 →                          -- total earnings
  c = 115 := by
  sorry

end NUMINAMATH_CALUDE_work_problem_solution_l577_57777


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l577_57709

/-- An isosceles trapezoid circumscribed around a circle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- One of the base angles in radians -/
  baseAngle : ℝ
  /-- Condition: The trapezoid is isosceles -/
  isIsosceles : True
  /-- Condition: The trapezoid is circumscribed around a circle -/
  isCircumscribed : True

/-- Calculate the area of the isosceles trapezoid -/
def areaOfTrapezoid (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles trapezoid is 180 -/
theorem area_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := {
    longerBase := 20,
    baseAngle := Real.arctan 1.5,
    isIsosceles := True.intro,
    isCircumscribed := True.intro
  }
  areaOfTrapezoid t = 180 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l577_57709


namespace NUMINAMATH_CALUDE_ones_digit_factorial_sum_10_l577_57766

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def ones_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n + 1 => factorial (n + 1) + factorial_sum n

theorem ones_digit_factorial_sum_10 :
  ones_digit (factorial_sum 10) = 3 := by sorry

end NUMINAMATH_CALUDE_ones_digit_factorial_sum_10_l577_57766


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_is_eight_l577_57759

/-- Given two distinct positive integers that are factors of 60, 
    this function returns the smallest product of these integers 
    that is not a factor of 60. -/
def smallest_non_factor_product : ℕ → ℕ → ℕ :=
  fun x y =>
    if x ≠ y ∧ x > 0 ∧ y > 0 ∧ 60 % x = 0 ∧ 60 % y = 0 ∧ 60 % (x * y) ≠ 0 then
      x * y
    else
      0

theorem smallest_non_factor_product_is_eight :
  ∀ x y : ℕ, x ≠ y → x > 0 → y > 0 → 60 % x = 0 → 60 % y = 0 → 60 % (x * y) ≠ 0 →
  smallest_non_factor_product x y ≥ 8 ∧
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ 60 % a = 0 ∧ 60 % b = 0 ∧ 60 % (a * b) ≠ 0 ∧ a * b = 8 :=
by sorry

#check smallest_non_factor_product_is_eight

end NUMINAMATH_CALUDE_smallest_non_factor_product_is_eight_l577_57759


namespace NUMINAMATH_CALUDE_fourth_difference_zero_third_nonzero_l577_57791

def u (n : ℕ) : ℤ := n^3 + n

def Δ' (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def Δ (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0 => u
  | k + 1 => Δ' (Δ k u)

theorem fourth_difference_zero_third_nonzero :
  (∀ n, Δ 4 u n = 0) ∧ (∃ n, Δ 3 u n ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_fourth_difference_zero_third_nonzero_l577_57791


namespace NUMINAMATH_CALUDE_number_of_rattlesnakes_l577_57750

theorem number_of_rattlesnakes (P B R V : ℕ) : 
  P + B + R + V = 420 →
  P = (3 * B) / 2 →
  V = 8 →
  P + R = 315 →
  R = 162 := by
sorry

end NUMINAMATH_CALUDE_number_of_rattlesnakes_l577_57750


namespace NUMINAMATH_CALUDE_shirt_cost_without_discount_main_theorem_l577_57715

theorem shirt_cost_without_discount (team_size : ℕ) 
  (discounted_shirt_cost discounted_pants_cost discounted_socks_cost : ℚ)
  (total_savings : ℚ) : ℚ :=
  let total_discounted_cost := team_size * (discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost)
  let total_undiscounted_cost := total_discounted_cost + total_savings
  let undiscounted_pants_and_socks_cost := team_size * (discounted_pants_cost + discounted_socks_cost)
  let total_undiscounted_shirts_cost := total_undiscounted_cost - undiscounted_pants_and_socks_cost
  total_undiscounted_shirts_cost / team_size

theorem main_theorem : 
  shirt_cost_without_discount 12 (6.75 : ℚ) (13.50 : ℚ) (3.75 : ℚ) (36 : ℚ) = (9.75 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_without_discount_main_theorem_l577_57715


namespace NUMINAMATH_CALUDE_poster_collection_ratio_l577_57768

theorem poster_collection_ratio : 
  let current_size : ℕ := 22
  let past_size : ℕ := 14
  let gcd := Nat.gcd current_size past_size
  (current_size / gcd) = 11 ∧ (past_size / gcd) = 7 :=
by sorry

end NUMINAMATH_CALUDE_poster_collection_ratio_l577_57768


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l577_57785

def a : ℝ × ℝ := (-5, 6)
def b : ℝ × ℝ := (6, 5)

theorem vectors_perpendicular : a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l577_57785


namespace NUMINAMATH_CALUDE_sum_of_products_l577_57779

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 250)
  (h2 : a + b + c = 16) :
  a*b + b*c + c*a = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_products_l577_57779


namespace NUMINAMATH_CALUDE_school_classes_l577_57720

theorem school_classes (s : ℕ) (h1 : s > 0) : 
  ∃ c : ℕ, c * s * (7 * 12) = 84 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_school_classes_l577_57720


namespace NUMINAMATH_CALUDE_largest_double_after_digit_removal_l577_57747

def is_double_after_digit_removal (x : ℚ) : Prop :=
  ∃ (y : ℚ), (y > 0) ∧ (y < 1) ∧ (x = 0.1 * 3 + y) ∧ (2 * x = 0.1 * 0 + y)

theorem largest_double_after_digit_removal :
  ∀ (x : ℚ), (x > 0) → (x < 1) → is_double_after_digit_removal x → x ≤ 0.375 :=
sorry

end NUMINAMATH_CALUDE_largest_double_after_digit_removal_l577_57747


namespace NUMINAMATH_CALUDE_sin_315_degrees_l577_57743

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l577_57743


namespace NUMINAMATH_CALUDE_exact_money_for_widgets_l577_57769

/-- If a person can buy exactly 6 items at a certain price, and exactly 8 items if the price is reduced by 10%, then the person has exactly $5 to spend. -/
theorem exact_money_for_widgets (price : ℝ) (money : ℝ) 
  (h1 : money = 6 * price) 
  (h2 : money = 8 * (0.9 * price)) : 
  money = 5 := by sorry

end NUMINAMATH_CALUDE_exact_money_for_widgets_l577_57769


namespace NUMINAMATH_CALUDE_range_of_fraction_l577_57789

theorem range_of_fraction (x y : ℝ) (h1 : 2*x + y = 8) (h2 : 2 ≤ x) (h3 : x ≤ 3) :
  3/2 ≤ (y+1)/(x-1) ∧ (y+1)/(x-1) ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_fraction_l577_57789


namespace NUMINAMATH_CALUDE_area_of_triangle_BQW_l577_57710

/-- Given a rectangle ABCD with the following properties:
    - AZ = WC = 8 units
    - AB = 16 units
    - Area of trapezoid ZWCD is 160 square units
    - Q divides ZW in the ratio 1:3 starting from Z
    Prove that the area of triangle BQW is 16 square units. -/
theorem area_of_triangle_BQW (AZ WC AB : ℝ) (area_ZWCD : ℝ) (Q : ℝ) :
  AZ = 8 →
  WC = 8 →
  AB = 16 →
  area_ZWCD = 160 →
  Q = 2 →  -- This represents Q dividing ZW in 1:3 ratio
  (1/2 : ℝ) * AB * Q = 16 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_BQW_l577_57710


namespace NUMINAMATH_CALUDE_passing_marks_l577_57793

theorem passing_marks (T : ℝ) 
  (h1 : 0.3 * T + 60 = 0.4 * T) 
  (h2 : 0.5 * T = 0.4 * T + 40) : 
  0.4 * T = 240 := by
  sorry

end NUMINAMATH_CALUDE_passing_marks_l577_57793


namespace NUMINAMATH_CALUDE_factorial_sum_l577_57772

theorem factorial_sum : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 6 * Nat.factorial 5 = 36600 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_l577_57772


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_126_l577_57763

theorem percentage_of_360_equals_126 : 
  (126 : ℝ) / 360 * 100 = 35 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_126_l577_57763


namespace NUMINAMATH_CALUDE_problem_solution_l577_57714

theorem problem_solution : (-1/2)⁻¹ - 4 * Real.cos (30 * π / 180) - (π + 2013)^0 + Real.sqrt 12 = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l577_57714


namespace NUMINAMATH_CALUDE_meal_profit_and_purchase_theorem_l577_57707

/-- Represents the profit for meals A and B -/
structure MealProfit where
  a : ℝ
  b : ℝ

/-- Represents the purchase quantities for meals A and B -/
structure PurchaseQuantity where
  a : ℝ
  b : ℝ

/-- Conditions for the meal profit problem -/
def meal_profit_conditions (p : MealProfit) : Prop :=
  p.a + 2 * p.b = 35 ∧ 2 * p.a + 3 * p.b = 60

/-- Conditions for the meal purchase problem -/
def meal_purchase_conditions (q : PurchaseQuantity) : Prop :=
  q.a + q.b = 1000 ∧ q.a ≤ 3/2 * q.b

/-- The theorem to be proved -/
theorem meal_profit_and_purchase_theorem 
  (p : MealProfit) 
  (q : PurchaseQuantity) 
  (h1 : meal_profit_conditions p) 
  (h2 : meal_purchase_conditions q) :
  p.a = 15 ∧ 
  p.b = 10 ∧ 
  q.a = 600 ∧ 
  q.b = 400 ∧ 
  p.a * q.a + p.b * q.b = 13000 := by
  sorry

end NUMINAMATH_CALUDE_meal_profit_and_purchase_theorem_l577_57707


namespace NUMINAMATH_CALUDE_y_derivative_l577_57731

noncomputable def y (x : ℝ) : ℝ := Real.sin (2 * x - 1) ^ 2

theorem y_derivative (x : ℝ) : 
  deriv y x = 2 * Real.sin (2 * (2 * x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l577_57731


namespace NUMINAMATH_CALUDE_stock_price_change_l577_57752

def total_stocks : ℕ := 8000

theorem stock_price_change (higher lower : ℕ) 
  (h1 : higher + lower = total_stocks)
  (h2 : higher = lower + lower / 2) :
  higher = 4800 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l577_57752


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l577_57764

theorem intersection_implies_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {-1, 0, 1} →
  B = {0, a, 2} →
  A ∩ B = {-1, 0} →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l577_57764


namespace NUMINAMATH_CALUDE_first_half_chop_count_l577_57792

/-- The number of trees that need to be planted for each tree chopped down -/
def replantRatio : ℕ := 3

/-- The number of trees chopped down in the second half of the year -/
def secondHalfChop : ℕ := 300

/-- The total number of trees that need to be planted -/
def totalPlant : ℕ := 1500

/-- The number of trees chopped down in the first half of the year -/
def firstHalfChop : ℕ := (totalPlant - replantRatio * secondHalfChop) / replantRatio

theorem first_half_chop_count : firstHalfChop = 200 := by
  sorry

end NUMINAMATH_CALUDE_first_half_chop_count_l577_57792


namespace NUMINAMATH_CALUDE_chocolate_vanilla_survey_l577_57780

theorem chocolate_vanilla_survey (total : ℕ) (chocolate : ℕ) (vanilla : ℕ) 
  (h_total : total = 120)
  (h_chocolate : chocolate = 95)
  (h_vanilla : vanilla = 85) :
  (chocolate + vanilla - total : ℕ) ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_vanilla_survey_l577_57780


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l577_57740

theorem product_remainder_by_ten (a b c : ℕ) (ha : a = 3251) (hb : b = 7462) (hc : c = 93419) :
  (a * b * c) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l577_57740


namespace NUMINAMATH_CALUDE_left_number_20th_row_l577_57736

/- Define the sequence of numbers in the array -/
def array_sequence (n : ℕ) : ℕ := n^2

/- Define the sum of numbers in the first n rows -/
def sum_of_rows (n : ℕ) : ℕ := n^2

/- Define the number on the far left of the nth row -/
def left_number (n : ℕ) : ℕ := sum_of_rows (n - 1) + 1

/- Theorem statement -/
theorem left_number_20th_row : left_number 20 = 362 := by
  sorry

end NUMINAMATH_CALUDE_left_number_20th_row_l577_57736


namespace NUMINAMATH_CALUDE_absolute_value_equation_l577_57723

theorem absolute_value_equation (x y : ℝ) :
  |x - Real.log y| = x + Real.log y → x * (y - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l577_57723
