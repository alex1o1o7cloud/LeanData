import Mathlib

namespace NUMINAMATH_CALUDE_min_colors_for_subdivided_rectangle_l2978_297858

/-- Represents an infinitely subdivided rectangle according to the given pattern. -/
structure InfinitelySubdividedRectangle where
  -- Add necessary fields here
  -- This is left abstract as the exact representation is not crucial for the theorem

/-- The minimum number of colors needed so that no two rectangles sharing an edge have the same color. -/
def minEdgeColors (r : InfinitelySubdividedRectangle) : ℕ := 3

/-- The minimum number of colors needed so that no two rectangles sharing a corner have the same color. -/
def minCornerColors (r : InfinitelySubdividedRectangle) : ℕ := 4

/-- Theorem stating the minimum number of colors needed for edge and corner coloring. -/
theorem min_colors_for_subdivided_rectangle (r : InfinitelySubdividedRectangle) :
  (minEdgeColors r, minCornerColors r) = (3, 4) := by sorry

end NUMINAMATH_CALUDE_min_colors_for_subdivided_rectangle_l2978_297858


namespace NUMINAMATH_CALUDE_divisibility_property_l2978_297876

theorem divisibility_property (m n p : ℕ) (h_prime : Nat.Prime p) 
  (h_order : m < n ∧ n < p) (h_div_m : p ∣ m^2 + 1) (h_div_n : p ∣ n^2 + 1) : 
  p ∣ m * n - 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l2978_297876


namespace NUMINAMATH_CALUDE_f_5_solutions_l2978_297868

/-- The function f(x) = x^2 + 12x + 30 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 30

/-- The composition of f with itself 5 times -/
def f_5 (x : ℝ) : ℝ := f (f (f (f (f x))))

/-- Theorem: The solutions to f(f(f(f(f(x))))) = 0 are x = -6 ± 6^(1/32) -/
theorem f_5_solutions :
  ∀ x : ℝ, f_5 x = 0 ↔ x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32) :=
by sorry

end NUMINAMATH_CALUDE_f_5_solutions_l2978_297868


namespace NUMINAMATH_CALUDE_west_8m_is_negative_8m_l2978_297894

/-- Represents the direction of movement --/
inductive Direction
  | East
  | West

/-- Represents a movement with magnitude and direction --/
structure Movement where
  magnitude : ℝ
  direction : Direction

/-- Convention for representing movement as a signed real number --/
def movementValue (m : Movement) : ℝ :=
  match m.direction with
  | Direction.East => m.magnitude
  | Direction.West => -m.magnitude

/-- Theorem stating that moving west 8m is equivalent to -8m --/
theorem west_8m_is_negative_8m :
  let west8m : Movement := { magnitude := 8, direction := Direction.West }
  movementValue west8m = -8 := by
  sorry

end NUMINAMATH_CALUDE_west_8m_is_negative_8m_l2978_297894


namespace NUMINAMATH_CALUDE_sibling_pair_probability_l2978_297846

/-- The probability of selecting a sibling pair when choosing one student
    randomly from each of two schools, given the number of students in each
    school and the number of sibling pairs. -/
theorem sibling_pair_probability
  (business_students : ℕ)
  (law_students : ℕ)
  (sibling_pairs : ℕ)
  (h1 : business_students = 500)
  (h2 : law_students = 800)
  (h3 : sibling_pairs = 30) :
  (sibling_pairs : ℚ) / (business_students * law_students) = 30 / (500 * 800) :=
sorry

end NUMINAMATH_CALUDE_sibling_pair_probability_l2978_297846


namespace NUMINAMATH_CALUDE_cat_or_bird_percentage_l2978_297880

/-- Represents the survey data from a high school -/
structure SurveyData where
  total_students : ℕ
  dog_owners : ℕ
  cat_owners : ℕ
  bird_owners : ℕ

/-- Calculates the percentage of students owning either cats or birds -/
def percentage_cat_or_bird (data : SurveyData) : ℚ :=
  (data.cat_owners + data.bird_owners : ℚ) / data.total_students * 100

/-- The survey data from the high school -/
def high_school_survey : SurveyData :=
  { total_students := 400
  , dog_owners := 80
  , cat_owners := 50
  , bird_owners := 20 }

/-- Theorem stating that the percentage of students owning either cats or birds is 17.5% -/
theorem cat_or_bird_percentage :
  percentage_cat_or_bird high_school_survey = 35/2 := by
  sorry

end NUMINAMATH_CALUDE_cat_or_bird_percentage_l2978_297880


namespace NUMINAMATH_CALUDE_proportion_solution_l2978_297804

theorem proportion_solution (x : ℝ) : (0.75 / x = 10 / 8) → x = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l2978_297804


namespace NUMINAMATH_CALUDE_min_triangle_area_l2978_297831

/-- A point in the 2D Cartesian plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Definition of a rectangle OABC with O at origin and B at (9, 8) -/
def rectangle : Set IntPoint :=
  {p : IntPoint | 0 ≤ p.x ∧ p.x ≤ 9 ∧ 0 ≤ p.y ∧ p.y ≤ 8}

/-- Area of triangle OBX given point X -/
def triangleArea (X : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * |9 * X.y - 8 * X.x|

/-- Theorem stating the minimum area of triangle OBX -/
theorem min_triangle_area :
  ∃ (min_area : ℚ), min_area = 1/2 ∧
  ∀ (X : IntPoint), X ∈ rectangle → triangleArea X ≥ min_area :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l2978_297831


namespace NUMINAMATH_CALUDE_v_1004_eq_3036_l2978_297859

/-- Defines the nth term of the sequence -/
def v (n : ℕ) : ℕ := sorry

/-- The 1004th term of the sequence is 3036 -/
theorem v_1004_eq_3036 : v 1004 = 3036 := by sorry

end NUMINAMATH_CALUDE_v_1004_eq_3036_l2978_297859


namespace NUMINAMATH_CALUDE_quadrilateral_diagonals_theorem_l2978_297883

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : Bool

def diagonals_bisect (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  (d1.1 / 2 = d2.1 / 2) ∧ (d1.2 / 2 = d2.2 / 2)

def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.vertices 1 - q.vertices 0 = q.vertices 3 - q.vertices 2) ∧
  (q.vertices 2 - q.vertices 1 = q.vertices 0 - q.vertices 3)

def diagonals_equal (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  d1.1 * d1.1 + d1.2 * d1.2 = d2.1 * d2.1 + d2.2 * d2.2

def diagonals_perpendicular (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  d1.1 * d2.1 + d1.2 * d2.2 = 0

theorem quadrilateral_diagonals_theorem :
  (∀ q : Quadrilateral, diagonals_bisect q → is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_equal q ∧ ¬is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_perpendicular q ∧ ¬is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_equal q ∧ diagonals_perpendicular q ∧ ¬is_parallelogram q) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonals_theorem_l2978_297883


namespace NUMINAMATH_CALUDE_problem_solution_l2978_297870

noncomputable section

def U : Set ℝ := Set.univ

def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem problem_solution (a : ℝ) :
  (∃ x, x ∈ A a) ∧ (∃ x, x ∈ B a) →
  (a = 1/2 → (U \ B a) ∩ A a = {x | 9/4 ≤ x ∧ x < 5/2}) ∧
  (A a ⊆ B a ↔ a ∈ Set.Icc (-1/2) (1/3) ∪ Set.Ioc (1/3) ((3 - Real.sqrt 5) / 2)) :=
sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2978_297870


namespace NUMINAMATH_CALUDE_log_equation_solution_l2978_297851

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9 →
  x = 2^(54/5) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2978_297851


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l2978_297836

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 7| + 1

-- Theorem for the solution set of f(x) ≤ x
theorem solution_set_f_leq_x :
  {x : ℝ | f x ≤ x} = {x : ℝ | 8/3 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x - 2*|x - 1| ≤ a) ↔ a ≥ -4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_range_of_a_l2978_297836


namespace NUMINAMATH_CALUDE_correct_subtraction_result_l2978_297891

theorem correct_subtraction_result 
  (mistaken_result : ℕ)
  (tens_digit_increase : ℕ)
  (units_digit_increase : ℕ)
  (h1 : mistaken_result = 217)
  (h2 : tens_digit_increase = 3)
  (h3 : units_digit_increase = 4) :
  mistaken_result - (tens_digit_increase * 10 - units_digit_increase) = 191 :=
by sorry

end NUMINAMATH_CALUDE_correct_subtraction_result_l2978_297891


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l2978_297888

theorem president_and_committee_selection (n : ℕ) (k : ℕ) : 
  n = 10 → k = 3 → n * (Nat.choose (n - 1) k) = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_selection_l2978_297888


namespace NUMINAMATH_CALUDE_max_a_bound_l2978_297835

theorem max_a_bound (a : ℝ) : 
  (∀ x > 0, (x^2 + 1) * Real.exp x ≥ a * x^2) ↔ a ≤ 2 * Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_max_a_bound_l2978_297835


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l2978_297823

def n : ℕ := sorry

def total_balls : ℕ := 2 * n

def white_balls : ℕ := n

def black_balls : ℕ := n

def num_people : ℕ := n

def prob_different_colors (n : ℕ) : ℚ :=
  (2^n * (n.factorial)^2 : ℚ) / ((2*n).factorial : ℚ)

def prob_same_colors (k : ℕ) : ℚ :=
  ((2*k).factorial^3 : ℚ) / ((4*k).factorial * (k.factorial)^2 : ℚ)

theorem ball_drawing_probabilities :
  (∀ m : ℕ, prob_different_colors m = (2^m * (m.factorial)^2 : ℚ) / ((2*m).factorial : ℚ)) ∧
  (∀ k : ℕ, prob_same_colors k = ((2*k).factorial^3 : ℚ) / ((4*k).factorial * (k.factorial)^2 : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_ball_drawing_probabilities_l2978_297823


namespace NUMINAMATH_CALUDE_fraction_equality_l2978_297810

theorem fraction_equality (p q : ℝ) (h : (p⁻¹ + q⁻¹) / (p⁻¹ - q⁻¹) = 1009) :
  (p + q) / (p - q) = -1009 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2978_297810


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2978_297852

theorem fraction_evaluation (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  ((1 / y^2) / (1 / x^2))^2 = 256 / 625 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2978_297852


namespace NUMINAMATH_CALUDE_paco_salty_cookies_left_l2978_297898

/-- The number of salty cookies Paco had left after eating some -/
def saltyCookiesLeft (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten

/-- Proof that Paco had 17 salty cookies left -/
theorem paco_salty_cookies_left : 
  saltyCookiesLeft 26 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_paco_salty_cookies_left_l2978_297898


namespace NUMINAMATH_CALUDE_zero_properties_l2978_297824

theorem zero_properties : 
  (0 : ℕ) = 0 ∧ (0 : ℤ) = 0 ∧ (0 : ℝ) = 0 ∧ ¬(0 > 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_properties_l2978_297824


namespace NUMINAMATH_CALUDE_intersection_A_B_l2978_297820

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {-1, 1, 2, 3}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2978_297820


namespace NUMINAMATH_CALUDE_parallel_segment_length_l2978_297813

theorem parallel_segment_length (a b c d : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : a = 300) (h3 : b = 320) (h4 : c = 400) :
  let s := (a + b + c) / 2
  let area_abc := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let area_dpd := area_abc / 4
  ∃ d : ℝ, d > 0 ∧ d^2 / a^2 = area_dpd / area_abc ∧ d = 150 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l2978_297813


namespace NUMINAMATH_CALUDE_wrong_height_calculation_wrong_height_is_176_l2978_297850

/-- Given a class of boys with an incorrect average height and one boy's height recorded incorrectly,
    calculate the wrongly written height of that boy. -/
theorem wrong_height_calculation (n : ℕ) (initial_avg correct_avg actual_height : ℝ) : ℝ :=
  let wrong_height := actual_height + n * (initial_avg - correct_avg)
  wrong_height

/-- Prove that the wrongly written height of a boy is 176 cm given the specified conditions. -/
theorem wrong_height_is_176 :
  wrong_height_calculation 35 180 178 106 = 176 := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_wrong_height_is_176_l2978_297850


namespace NUMINAMATH_CALUDE_larger_divided_by_smaller_l2978_297866

theorem larger_divided_by_smaller : 
  let a := 8
  let b := 22
  let larger := max a b
  let smaller := min a b
  larger / smaller = 2.75 := by sorry

end NUMINAMATH_CALUDE_larger_divided_by_smaller_l2978_297866


namespace NUMINAMATH_CALUDE_mans_downstream_rate_l2978_297837

/-- The man's rate when rowing downstream, given his rate in still water and the current's rate -/
def downstream_rate (still_water_rate current_rate : ℝ) : ℝ :=
  still_water_rate + current_rate

/-- Theorem: The man's rate when rowing downstream is 32 kmph -/
theorem mans_downstream_rate :
  let still_water_rate : ℝ := 24.5
  let current_rate : ℝ := 7.5
  downstream_rate still_water_rate current_rate = 32 := by
  sorry

end NUMINAMATH_CALUDE_mans_downstream_rate_l2978_297837


namespace NUMINAMATH_CALUDE_find_d_when_a_b_c_equal_l2978_297819

theorem find_d_when_a_b_c_equal (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 2 = d + 3 * Real.sqrt (a + b + c - d) →
  a = b →
  b = c →
  d = 5/4 := by
sorry

end NUMINAMATH_CALUDE_find_d_when_a_b_c_equal_l2978_297819


namespace NUMINAMATH_CALUDE_volume_of_region_l2978_297864

-- Define the region
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   |x - y + z| + |x - y - z| ≤ 10 ∧
                   x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0}

-- State the theorem
theorem volume_of_region : 
  MeasureTheory.volume Region = 125 := by sorry

end NUMINAMATH_CALUDE_volume_of_region_l2978_297864


namespace NUMINAMATH_CALUDE_green_ball_probability_l2978_297879

-- Define the containers and their contents
def containerA : ℕ × ℕ := (10, 5)  -- (red balls, green balls)
def containerB : ℕ × ℕ := (3, 6)
def containerC : ℕ × ℕ := (3, 6)

-- Define the probability of selecting each container
def containerProb : ℚ := 1 / 3

-- Define the probability of selecting a green ball from each container
def greenProbA : ℚ := containerA.2 / (containerA.1 + containerA.2)
def greenProbB : ℚ := containerB.2 / (containerB.1 + containerB.2)
def greenProbC : ℚ := containerC.2 / (containerC.1 + containerC.2)

-- Theorem: The probability of selecting a green ball is 5/9
theorem green_ball_probability :
  containerProb * greenProbA +
  containerProb * greenProbB +
  containerProb * greenProbC = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l2978_297879


namespace NUMINAMATH_CALUDE_stock_worth_l2978_297886

theorem stock_worth (X : ℝ) : 
  (0.2 * X * 1.1 + 0.8 * X * 0.95) - X = -250 → X = 12500 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_l2978_297886


namespace NUMINAMATH_CALUDE_vector_inequality_l2978_297853

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given two non-zero vectors a and b satisfying |a + b| = |b|, prove |2b| > |a + 2b| -/
theorem vector_inequality (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (h : ‖a + b‖ = ‖b‖) :
  ‖(2 : ℝ) • b‖ > ‖a + (2 : ℝ) • b‖ := by sorry

end NUMINAMATH_CALUDE_vector_inequality_l2978_297853


namespace NUMINAMATH_CALUDE_probability_white_ball_l2978_297854

/-- The probability of drawing a white ball from a bag with black and white balls -/
theorem probability_white_ball (black_balls white_balls : ℕ) : 
  black_balls = 6 → white_balls = 5 → 
  (white_balls : ℚ) / (black_balls + white_balls : ℚ) = 5 / 11 :=
by
  sorry

#check probability_white_ball

end NUMINAMATH_CALUDE_probability_white_ball_l2978_297854


namespace NUMINAMATH_CALUDE_inequality_proof_l2978_297843

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (b - c) > 1 / (a - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2978_297843


namespace NUMINAMATH_CALUDE_complex_multiplication_result_l2978_297841

theorem complex_multiplication_result (x : ℝ) : 
  let i : ℂ := Complex.I
  let y : ℂ := (2 * x + i) * (1 - i)
  y = 2 := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_result_l2978_297841


namespace NUMINAMATH_CALUDE_arithmetic_sequence_pattern_l2978_297881

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_pattern (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 - 2 * a 2 + a 3 = 0) →
  (a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0) →
  (a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0) →
  (a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_pattern_l2978_297881


namespace NUMINAMATH_CALUDE_larger_number_problem_l2978_297840

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 5) (h2 : x + y = 27) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2978_297840


namespace NUMINAMATH_CALUDE_line_m_equation_l2978_297893

/-- Two distinct lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point about a line -/
def reflect (p : Point) (l : Line) : Point := sorry

/-- The theorem statement -/
theorem line_m_equation (l m : Line) (Q Q'' : Point) :
  l.a = 3 ∧ l.b = -4 ∧ l.c = 0 →  -- Line ℓ: 3x - 4y = 0
  Q.x = 3 ∧ Q.y = -2 →  -- Point Q(3, -2)
  Q''.x = 2 ∧ Q''.y = 5 →  -- Point Q''(2, 5)
  (∃ Q' : Point, reflect Q l = Q' ∧ reflect Q' m = Q'') →  -- Reflection conditions
  m.a = 1 ∧ m.b = 7 ∧ m.c = 0  -- Line m: x + 7y = 0
  := by sorry

end NUMINAMATH_CALUDE_line_m_equation_l2978_297893


namespace NUMINAMATH_CALUDE_complex_equation_first_quadrant_l2978_297878

/-- Given a complex equation, prove the resulting point is in the first quadrant -/
theorem complex_equation_first_quadrant (a b : ℝ) : 
  (2 + a * Complex.I) / (1 + Complex.I) = b + Complex.I → 
  a > 0 ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_first_quadrant_l2978_297878


namespace NUMINAMATH_CALUDE_integer_pair_property_l2978_297808

theorem integer_pair_property (x y : ℤ) (h : x > y) :
  (x * y - (x + y) = Nat.gcd x.natAbs y.natAbs + Nat.lcm x.natAbs y.natAbs) ↔
  ((x = 6 ∧ y = 3) ∨ 
   (x = 6 ∧ y = 4) ∨ 
   (∃ t : ℕ, x = 1 + t ∧ y = -t) ∨
   (∃ t : ℕ, x = 2 ∧ y = -2 * t)) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_property_l2978_297808


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l2978_297814

/-- The number of ordered pairs of positive integers (x,y) satisfying xy = 1944 -/
def num_ordered_pairs : ℕ := 24

/-- The prime factorization of 1944 -/
def prime_factorization_1944 : List (ℕ × ℕ) := [(2, 3), (3, 5)]

/-- Theorem stating that the number of ordered pairs (x,y) of positive integers
    satisfying xy = 1944 is equal to 24, given the prime factorization of 1944 -/
theorem count_ordered_pairs :
  (∀ (x y : ℕ), x * y = 1944 → x > 0 ∧ y > 0) →
  prime_factorization_1944 = [(2, 3), (3, 5)] →
  num_ordered_pairs = 24 := by
  sorry

#check count_ordered_pairs

end NUMINAMATH_CALUDE_count_ordered_pairs_l2978_297814


namespace NUMINAMATH_CALUDE_probability_intersection_l2978_297832

theorem probability_intersection (A B : ℝ) (union : ℝ) (h1 : 0 ≤ A ∧ A ≤ 1) (h2 : 0 ≤ B ∧ B ≤ 1) (h3 : 0 ≤ union ∧ union ≤ 1) :
  ∃ intersection : ℝ, 0 ≤ intersection ∧ intersection ≤ 1 ∧ union = A + B - intersection :=
by sorry

end NUMINAMATH_CALUDE_probability_intersection_l2978_297832


namespace NUMINAMATH_CALUDE_pentagonal_tiles_count_l2978_297861

theorem pentagonal_tiles_count (total_tiles total_edges : ℕ) 
  (h1 : total_tiles = 30)
  (h2 : total_edges = 120) : 
  ∃ (triangular_tiles pentagonal_tiles : ℕ),
    triangular_tiles + pentagonal_tiles = total_tiles ∧
    3 * triangular_tiles + 5 * pentagonal_tiles = total_edges ∧
    pentagonal_tiles = 15 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_tiles_count_l2978_297861


namespace NUMINAMATH_CALUDE_exist_numbers_same_divisors_less_sum_l2978_297849

/-- The number of natural divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The sum of all natural divisors of a natural number -/
def sum_divisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- There exist two natural numbers with the same number of divisors,
    where one is greater than the other, but has a smaller sum of divisors -/
theorem exist_numbers_same_divisors_less_sum :
  ∃ x y : ℕ, x > y ∧ num_divisors x = num_divisors y ∧ sum_divisors x < sum_divisors y :=
sorry

end NUMINAMATH_CALUDE_exist_numbers_same_divisors_less_sum_l2978_297849


namespace NUMINAMATH_CALUDE_factor_theorem_application_l2978_297856

theorem factor_theorem_application (m k : ℝ) : 
  (∃ q : ℝ, m^2 - k*m - 24 = (m - 8) * q) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l2978_297856


namespace NUMINAMATH_CALUDE_proposition_falsity_l2978_297867

theorem proposition_falsity (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 6) : 
  ¬ P 5 := by
sorry

end NUMINAMATH_CALUDE_proposition_falsity_l2978_297867


namespace NUMINAMATH_CALUDE_pawpaw_count_l2978_297829

/-- Represents the contents of fruit baskets -/
structure FruitBaskets where
  total_fruits : Nat
  num_baskets : Nat
  mangoes : Nat
  pears : Nat
  lemons : Nat
  kiwi : Nat
  pawpaws : Nat

/-- Theorem stating the number of pawpaws in one basket -/
theorem pawpaw_count (fb : FruitBaskets) 
  (h1 : fb.total_fruits = 58)
  (h2 : fb.num_baskets = 5)
  (h3 : fb.mangoes = 18)
  (h4 : fb.pears = 10)
  (h5 : fb.lemons = 9)
  (h6 : fb.kiwi = fb.lemons)
  (h7 : fb.total_fruits = fb.mangoes + fb.pears + fb.lemons + fb.kiwi + fb.pawpaws) :
  fb.pawpaws = 12 := by
  sorry

end NUMINAMATH_CALUDE_pawpaw_count_l2978_297829


namespace NUMINAMATH_CALUDE_intersection_distance_zero_l2978_297865

/-- The distance between the intersection points of x^2 + y^2 = 18 and x + y = 6 is 0 -/
theorem intersection_distance_zero : 
  let S := {p : ℝ × ℝ | p.1^2 + p.2^2 = 18 ∧ p.1 + p.2 = 6}
  ∃! p : ℝ × ℝ, p ∈ S :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_zero_l2978_297865


namespace NUMINAMATH_CALUDE_equation_implication_l2978_297863

theorem equation_implication (x y : ℝ) : 
  x^2 - 3*x*y + 2*y^2 + x - y = 0 → 
  x^2 - 2*x*y + y^2 - 5*x + 7*y = 0 → 
  x*y - 12*x + 15*y = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_implication_l2978_297863


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2978_297895

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s * s = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2978_297895


namespace NUMINAMATH_CALUDE_increase_average_grades_l2978_297862

theorem increase_average_grades (group_a_avg : ℝ) (group_b_avg : ℝ) 
  (group_a_size : ℕ) (group_b_size : ℕ) (student1_grade : ℝ) (student2_grade : ℝ) :
  group_a_avg = 44.2 →
  group_b_avg = 38.8 →
  group_a_size = 10 →
  group_b_size = 10 →
  student1_grade = 41 →
  student2_grade = 44 →
  let new_group_a_avg := (group_a_avg * group_a_size - student1_grade - student2_grade) / (group_a_size - 2)
  let new_group_b_avg := (group_b_avg * group_b_size + student1_grade + student2_grade) / (group_b_size + 2)
  new_group_a_avg > group_a_avg ∧ new_group_b_avg > group_b_avg := by
  sorry

end NUMINAMATH_CALUDE_increase_average_grades_l2978_297862


namespace NUMINAMATH_CALUDE_amanda_keeps_121_candy_bars_l2978_297812

/-- The number of candy bars Amanda keeps for herself after four days of transactions --/
def amanda_candy_bars : ℕ :=
  let initial := 7
  let day1_remaining := initial - (initial / 3)
  let day2_total := day1_remaining + 30
  let day2_remaining := day2_total - (day2_total / 4)
  let day3_gift := day2_remaining * 3
  let day3_remaining := day2_remaining + (day3_gift / 2)
  let day4_bought := 20
  let day4_remaining := day3_remaining + (day4_bought / 3)
  day4_remaining

/-- Theorem stating that Amanda keeps 121 candy bars for herself --/
theorem amanda_keeps_121_candy_bars : amanda_candy_bars = 121 := by
  sorry

end NUMINAMATH_CALUDE_amanda_keeps_121_candy_bars_l2978_297812


namespace NUMINAMATH_CALUDE_kitchen_tile_size_l2978_297890

/-- Given a rectangular kitchen floor and the number of tiles needed, 
    calculate the size of each tile. -/
theorem kitchen_tile_size 
  (length : ℕ) 
  (width : ℕ) 
  (num_tiles : ℕ) 
  (h1 : length = 48) 
  (h2 : width = 72) 
  (h3 : num_tiles = 96) : 
  (length * width) / num_tiles = 36 := by
  sorry

#check kitchen_tile_size

end NUMINAMATH_CALUDE_kitchen_tile_size_l2978_297890


namespace NUMINAMATH_CALUDE_subset_sum_theorem_l2978_297887

def A : Finset ℤ := {-3, 0, 2, 6}
def B : Finset ℤ := {-1, 3, 5, 8}

theorem subset_sum_theorem :
  (∀ (s : Finset ℤ), s ⊆ A → s.card = 3 → (s.sum id) ∈ B) ∧
  (∀ b ∈ B, ∃ (s : Finset ℤ), s ⊆ A ∧ s.card = 3 ∧ s.sum id = b) →
  A = {-3, 0, 2, 6} :=
by sorry

end NUMINAMATH_CALUDE_subset_sum_theorem_l2978_297887


namespace NUMINAMATH_CALUDE_vector_addition_l2978_297839

theorem vector_addition :
  let v1 : Fin 2 → ℝ := ![5, -9]
  let v2 : Fin 2 → ℝ := ![-8, 14]
  v1 + v2 = ![(-3), 5] := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l2978_297839


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2978_297821

/-- The area of a square inscribed in the ellipse x^2/4 + y^2 = 1, with its sides parallel to the coordinate axes, is 16/5. -/
theorem inscribed_square_area (x y : ℝ) :
  (∃ t : ℝ, x^2 / 4 + y^2 = 1 ∧ (x = t ∨ x = -t) ∧ (y = t ∨ y = -t)) →
  (∃ s : ℝ, s^2 = 16 / 5 ∧ s^2 = 4 * x^2) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2978_297821


namespace NUMINAMATH_CALUDE_exam_students_count_l2978_297838

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
  T = N * 80 →
  (T - 100) / (N - 5 : ℝ) = 95 →
  N = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l2978_297838


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2978_297827

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1/3) * π * r^2 * h = 30*π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39*π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2978_297827


namespace NUMINAMATH_CALUDE_planted_field_fraction_l2978_297892

theorem planted_field_fraction (a b x : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : x = 3) :
  let total_area := (a * b) / 2
  let square_area := x^2
  let planted_area := total_area - square_area
  planted_area / total_area = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_planted_field_fraction_l2978_297892


namespace NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l2978_297801

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 4

/-- The total number of people -/
def total_people : ℕ := num_teachers + num_students

/-- The number of arrangements of n elements taken r at a time -/
def arrangements (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The number of arrangements where teachers are not adjacent -/
def arrangements_teachers_not_adjacent : ℕ := 
  arrangements num_students num_students * arrangements (num_students + 1) num_teachers

theorem teachers_not_adjacent_arrangements :
  arrangements_teachers_not_adjacent = 480 :=
by sorry

end NUMINAMATH_CALUDE_teachers_not_adjacent_arrangements_l2978_297801


namespace NUMINAMATH_CALUDE_bug_returns_probability_l2978_297803

def bug_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | k + 1 => 1/3 * (1 - bug_probability k)

theorem bug_returns_probability :
  bug_probability 12 = 44287 / 177147 :=
by sorry

end NUMINAMATH_CALUDE_bug_returns_probability_l2978_297803


namespace NUMINAMATH_CALUDE_obtuse_angles_are_second_quadrant_l2978_297899

-- Define angle types
def ObtuseAngle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def SecondQuadrantAngle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180
def FirstQuadrantAngle (θ : ℝ) : Prop := 0 < θ ∧ θ < 90
def ThirdQuadrantAngle (θ : ℝ) : Prop := -180 < θ ∧ θ < -90

-- Theorem statement
theorem obtuse_angles_are_second_quadrant :
  ∀ θ : ℝ, ObtuseAngle θ ↔ SecondQuadrantAngle θ := by
  sorry

end NUMINAMATH_CALUDE_obtuse_angles_are_second_quadrant_l2978_297899


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2978_297897

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) :
  1/m + 1/n ≥ 2 ∧ (1/m + 1/n = 2 ↔ m = 1 ∧ n = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2978_297897


namespace NUMINAMATH_CALUDE_sakshi_work_days_l2978_297847

/-- Proves that if Tanya is 25% more efficient than Sakshi and takes 8 days to complete a piece of work, then Sakshi takes 10 days to complete the same work. -/
theorem sakshi_work_days (sakshi_days : ℝ) (tanya_days : ℝ) : 
  tanya_days = 8 → 
  sakshi_days * 1 = tanya_days * 1.25 → 
  sakshi_days = 10 := by
sorry

end NUMINAMATH_CALUDE_sakshi_work_days_l2978_297847


namespace NUMINAMATH_CALUDE_five_people_seven_chairs_l2978_297833

/-- The number of ways to arrange n people in k chairs, where the first person
    cannot sit in m specific chairs. -/
def seating_arrangements (n k m : ℕ) : ℕ :=
  (k - m) * (k - 1).factorial / (k - n).factorial

/-- The problem statement -/
theorem five_people_seven_chairs : seating_arrangements 5 7 2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_five_people_seven_chairs_l2978_297833


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2978_297860

/-- 
Given a hyperbola with equation x^2 + my^2 = 1, where m is a real number,
if the length of the imaginary axis is twice the length of the real axis,
then m = -1/4.
-/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b = 2*a ∧ 
    ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  m = -1/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2978_297860


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2978_297857

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 / b^2 + b^2 / c^2 + c^2 / a^2 ≥ 3 ∧
  (a^2 / b^2 + b^2 / c^2 + c^2 / a^2 = 3 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2978_297857


namespace NUMINAMATH_CALUDE_root_of_equation_l2978_297834

def combination (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def permutation (n k : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem root_of_equation : ∃ (x : ℕ), 
  x > 6 ∧ 3 * (combination (x - 3) 4) = 5 * (permutation (x - 4) 2) ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_root_of_equation_l2978_297834


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_m_range_l2978_297822

theorem quadratic_two_zeros_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) →
  m < -2 ∨ m > 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_m_range_l2978_297822


namespace NUMINAMATH_CALUDE_i_cubed_eq_neg_i_l2978_297874

-- Define the imaginary unit i
axiom i : ℂ
axiom i_squared : i^2 = -1

-- State the theorem
theorem i_cubed_eq_neg_i : i^3 = -i := by sorry

end NUMINAMATH_CALUDE_i_cubed_eq_neg_i_l2978_297874


namespace NUMINAMATH_CALUDE_division_problem_l2978_297828

theorem division_problem (n : ℕ) : n % 21 = 1 ∧ n / 21 = 9 → n = 190 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2978_297828


namespace NUMINAMATH_CALUDE_number_operation_l2978_297807

theorem number_operation (x : ℝ) : (x - 5) / 7 = 7 → (x - 4) / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l2978_297807


namespace NUMINAMATH_CALUDE_max_b_value_l2978_297811

/-- The volume of the box -/
def box_volume : ℕ := 360

/-- Theorem stating the maximum possible value of b given the conditions -/
theorem max_b_value (a b c : ℕ) 
  (vol_eq : a * b * c = box_volume)
  (int_cond : 1 < c ∧ c < b ∧ b < a) : 
  b ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_max_b_value_l2978_297811


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2978_297842

/-- A hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b
  h_asymptote : a / b = 3 / 4
  h_focus : 5 = Real.sqrt (a^2 + b^2)

/-- The equation of the hyperbola is y²/9 - x²/16 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : h.a = 3 ∧ h.b = 4 := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l2978_297842


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2978_297816

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2978_297816


namespace NUMINAMATH_CALUDE_max_value_constraint_l2978_297806

theorem max_value_constraint (x y z : ℝ) (h : x^2 + y^2 + z^2 = 25) :
  x + 2*y + 2*z ≤ 15 := by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2978_297806


namespace NUMINAMATH_CALUDE_equation_proof_l2978_297869

-- Define the variables and the given equation
theorem equation_proof (x : ℝ) (Q : ℝ) (h : 4 * (5 * x + 7 * Real.pi) = Q) :
  8 * (10 * x + 14 * Real.pi) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2978_297869


namespace NUMINAMATH_CALUDE_stating_head_start_for_tie_l2978_297818

/-- Represents the race scenario -/
structure RaceScenario where
  course_length : ℝ
  speed_ratio : ℝ
  head_start : ℝ

/-- 
Calculates whether the race ends in a tie given a RaceScenario
-/
def is_tie (scenario : RaceScenario) : Prop :=
  scenario.course_length / scenario.speed_ratio = 
  (scenario.course_length - scenario.head_start)

/-- 
Theorem stating that for a 84-meter course where A is 4 times faster than B,
a 63-meter head start results in a tie
-/
theorem head_start_for_tie : 
  let scenario : RaceScenario := {
    course_length := 84,
    speed_ratio := 4,
    head_start := 63
  }
  is_tie scenario := by sorry

end NUMINAMATH_CALUDE_stating_head_start_for_tie_l2978_297818


namespace NUMINAMATH_CALUDE_circle_tangency_radius_sum_l2978_297844

/-- A circle with center D(r, r) is tangent to the positive x and y-axes
    and externally tangent to a circle centered at (5,0) with radius 1.
    The sum of all possible radii of the circle with center D is 12. -/
theorem circle_tangency_radius_sum : 
  ∀ r : ℝ, 
    (r > 0) →
    ((r - 5)^2 + r^2 = (r + 1)^2) →
    (∃ s : ℝ, (s > 0) ∧ ((s - 5)^2 + s^2 = (s + 1)^2) ∧ (r + s = 12)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_radius_sum_l2978_297844


namespace NUMINAMATH_CALUDE_video_game_earnings_l2978_297845

/-- Given the conditions of Mike's video game selling scenario, prove the total earnings. -/
theorem video_game_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : 
  total_games = 16 → non_working_games = 8 → price_per_game = 7 → 
  (total_games - non_working_games) * price_per_game = 56 := by
  sorry

end NUMINAMATH_CALUDE_video_game_earnings_l2978_297845


namespace NUMINAMATH_CALUDE_john_square_calculation_l2978_297882

theorem john_square_calculation (n : ℕ) (h : n = 50) :
  n^2 + 101 = (n + 1)^2 → n^2 - 99 = (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_john_square_calculation_l2978_297882


namespace NUMINAMATH_CALUDE_maria_green_towels_l2978_297877

/-- The number of green towels Maria bought -/
def green_towels : ℕ := sorry

/-- The total number of towels Maria had initially -/
def total_towels : ℕ := green_towels + 44

/-- The number of towels Maria had after giving some away -/
def remaining_towels : ℕ := total_towels - 65

theorem maria_green_towels : green_towels = 40 :=
  by
    have h1 : remaining_towels = 19 := sorry
    sorry

#check maria_green_towels

end NUMINAMATH_CALUDE_maria_green_towels_l2978_297877


namespace NUMINAMATH_CALUDE_right_triangle_tan_G_l2978_297817

theorem right_triangle_tan_G (FG HG FH : ℝ) (h1 : FG = 17) (h2 : HG = 15) 
  (h3 : FG^2 = FH^2 + HG^2) : 
  FH / HG = 8 / 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tan_G_l2978_297817


namespace NUMINAMATH_CALUDE_mult_func_property_l2978_297872

/-- A function satisfying f(a+b) = f(a) * f(b) for all real a, b -/
def MultFunc (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) = f a * f b

theorem mult_func_property (f : ℝ → ℝ) (h1 : MultFunc f) (h2 : f 1 = 2) :
  f 0 + f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mult_func_property_l2978_297872


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l2978_297805

/-- Given uphill speed V₁ and downhill speed V₂, 
    the average speed for a round trip is (2 * V₁ * V₂) / (V₁ + V₂) -/
theorem average_speed_round_trip (V₁ V₂ : ℝ) (h₁ : V₁ > 0) (h₂ : V₂ > 0) :
  let s : ℝ := 1  -- Assume unit distance for simplicity
  let t_up : ℝ := s / V₁
  let t_down : ℝ := s / V₂
  let total_distance : ℝ := 2 * s
  let total_time : ℝ := t_up + t_down
  total_distance / total_time = (2 * V₁ * V₂) / (V₁ + V₂) :=
by sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l2978_297805


namespace NUMINAMATH_CALUDE_max_distinct_permutations_eight_points_l2978_297800

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for directed lines in a plane
def DirectedLine : Type := ℝ × ℝ × ℝ  -- ax + by + c = 0, with (a,b) ≠ (0,0)

-- Define a function to project a point onto a directed line
def project (p : Point) (l : DirectedLine) : ℝ := sorry

-- Define a function to get the permutation from projections
def getPermutation (points : List Point) (l : DirectedLine) : List ℕ := sorry

-- Define a function to count distinct permutations
def countDistinctPermutations (points : List Point) : ℕ := sorry

theorem max_distinct_permutations_eight_points :
  ∀ (points : List Point),
    points.length = 8 →
    points.Nodup →
    (∀ (l : DirectedLine), (getPermutation points l).Nodup) →
    countDistinctPermutations points ≤ 56 ∧
    ∃ (points' : List Point),
      points'.length = 8 ∧
      points'.Nodup ∧
      (∀ (l : DirectedLine), (getPermutation points' l).Nodup) ∧
      countDistinctPermutations points' = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_distinct_permutations_eight_points_l2978_297800


namespace NUMINAMATH_CALUDE_not_p_and_not_q_true_l2978_297871

theorem not_p_and_not_q_true (p q : Prop)
  (h1 : ¬(p ∧ q))
  (h2 : ¬(p ∨ q)) :
  (¬p ∧ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_not_p_and_not_q_true_l2978_297871


namespace NUMINAMATH_CALUDE_restaurant_bill_l2978_297896

theorem restaurant_bill (food_cost : ℝ) (service_fee_percent : ℝ) (tip : ℝ) : 
  food_cost = 50 ∧ service_fee_percent = 12 ∧ tip = 5 →
  food_cost + (service_fee_percent / 100) * food_cost + tip = 61 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_l2978_297896


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l2978_297889

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 5*a + 5 = 0) → (b^2 - 5*b + 5 = 0) → a^2 + b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l2978_297889


namespace NUMINAMATH_CALUDE_rug_area_l2978_297855

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
theorem rug_area (floor_length floor_width strip_width : ℝ) 
  (h_floor_length : floor_length = 10)
  (h_floor_width : floor_width = 8)
  (h_strip_width : strip_width = 2)
  (h_positive_length : floor_length > 0)
  (h_positive_width : floor_width > 0)
  (h_positive_strip : strip_width > 0)
  (h_strip_fits : 2 * strip_width < floor_length ∧ 2 * strip_width < floor_width) :
  (floor_length - 2 * strip_width) * (floor_width - 2 * strip_width) = 24 := by
sorry

end NUMINAMATH_CALUDE_rug_area_l2978_297855


namespace NUMINAMATH_CALUDE_equation_proof_l2978_297815

theorem equation_proof : (8 - 2) + 5 * (3 - 2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2978_297815


namespace NUMINAMATH_CALUDE_equation_solution_l2978_297885

theorem equation_solution : 
  ∃ x : ℚ, (x - 1) / 2 - (2 - x) / 3 = 2 ∧ x = 19 / 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2978_297885


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l2978_297884

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3*x + m = 0 ∧ x = 1) → m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l2978_297884


namespace NUMINAMATH_CALUDE_z_properties_l2978_297809

/-- Complex number z as a function of real number a -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 4) (a + 2)

/-- Condition for z to be purely imaginary -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Condition for z to lie on the line x + 2y + 1 = 0 -/
def on_line (z : ℂ) : Prop := z.re + 2 * z.im + 1 = 0

theorem z_properties (a : ℝ) :
  (is_purely_imaginary (z a) → a = 2) ∧
  (on_line (z a) → a = -1) := by sorry

end NUMINAMATH_CALUDE_z_properties_l2978_297809


namespace NUMINAMATH_CALUDE_sallys_nickels_from_dad_l2978_297802

/-- The number of nickels Sally's dad gave her -/
def dads_nickels (initial_nickels mother_nickels total_nickels : ℕ) : ℕ :=
  total_nickels - (initial_nickels + mother_nickels)

/-- Proof that Sally's dad gave her 9 nickels -/
theorem sallys_nickels_from_dad :
  dads_nickels 7 2 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sallys_nickels_from_dad_l2978_297802


namespace NUMINAMATH_CALUDE_prism_volume_l2978_297825

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 8 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l2978_297825


namespace NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l2978_297826

theorem smallest_addition_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (726 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (726 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l2978_297826


namespace NUMINAMATH_CALUDE_binomial_17_4_l2978_297848

theorem binomial_17_4 : Nat.choose 17 4 = 2380 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_4_l2978_297848


namespace NUMINAMATH_CALUDE_gcd_689_1021_l2978_297875

theorem gcd_689_1021 : Nat.gcd 689 1021 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_689_1021_l2978_297875


namespace NUMINAMATH_CALUDE_sqrt_less_than_y_plus_one_l2978_297873

theorem sqrt_less_than_y_plus_one (y : ℝ) (h : y > 0) : Real.sqrt y < y + 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_less_than_y_plus_one_l2978_297873


namespace NUMINAMATH_CALUDE_combined_boys_avg_is_24_58_l2978_297830

/-- Represents a high school with average test scores -/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  total_avg : ℝ

/-- Calculates the combined average score for boys given two schools and the combined girls' average -/
def combined_boys_avg (lincoln : School) (madison : School) (combined_girls_avg : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the combined boys' average is approximately 24.58 -/
theorem combined_boys_avg_is_24_58 
  (lincoln : School)
  (madison : School)
  (combined_girls_avg : ℝ)
  (h1 : lincoln.boys_avg = 65)
  (h2 : lincoln.girls_avg = 70)
  (h3 : lincoln.total_avg = 68)
  (h4 : madison.boys_avg = 75)
  (h5 : madison.girls_avg = 85)
  (h6 : madison.total_avg = 78)
  (h7 : combined_girls_avg = 80) :
  ∃ ε > 0, |combined_boys_avg lincoln madison combined_girls_avg - 24.58| < ε :=
by sorry

end NUMINAMATH_CALUDE_combined_boys_avg_is_24_58_l2978_297830
