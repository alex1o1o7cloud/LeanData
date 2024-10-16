import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2097_209754

theorem absolute_value_inequality (x : ℝ) :
  (1 < |x - 1| ∧ |x - 1| < 4) ↔ ((-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2097_209754


namespace NUMINAMATH_CALUDE_correct_article_usage_l2097_209706

/-- Represents possible article choices --/
inductive Article
  | A
  | The
  | None

/-- Represents a sentence with two article slots --/
structure Sentence :=
  (first_article : Article)
  (second_article : Article)

/-- Checks if the sentence is grammatically correct --/
def is_grammatically_correct (s : Sentence) : Prop :=
  s.first_article = Article.A ∧ s.second_article = Article.None

/-- The theorem stating the correct article usage --/
theorem correct_article_usage :
  ∃ (s : Sentence), is_grammatically_correct s :=
sorry


end NUMINAMATH_CALUDE_correct_article_usage_l2097_209706


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l2097_209719

theorem soccer_penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : 
  total_players = 16 → goalkeepers = 2 → (total_players - goalkeepers) * goalkeepers = 30 := by
  sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l2097_209719


namespace NUMINAMATH_CALUDE_average_of_numbers_l2097_209786

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

theorem average_of_numbers : (numbers.sum / numbers.length : ℚ) = 1380 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_l2097_209786


namespace NUMINAMATH_CALUDE_four_integers_sum_l2097_209730

theorem four_integers_sum (a b c d : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧
  a + b + c = 6 ∧
  a + b + d = 7 ∧
  a + c + d = 8 ∧
  b + c + d = 9 →
  a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_four_integers_sum_l2097_209730


namespace NUMINAMATH_CALUDE_complex_number_properties_l2097_209757

def z : ℂ := 3 - 4 * Complex.I

theorem complex_number_properties : 
  (Complex.abs z = 5) ∧ 
  (∃ (y : ℝ), z - 3 = y * Complex.I) ∧
  (z.re > 0 ∧ z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2097_209757


namespace NUMINAMATH_CALUDE_running_increase_calculation_l2097_209740

theorem running_increase_calculation 
  (initial_miles : ℕ) 
  (increase_percentage : ℚ) 
  (total_days : ℕ) 
  (days_per_week : ℕ) : 
  initial_miles = 100 →
  increase_percentage = 1/5 →
  total_days = 280 →
  days_per_week = 7 →
  (initial_miles * (1 + increase_percentage) - initial_miles) / (total_days / days_per_week) = 3 :=
by sorry

end NUMINAMATH_CALUDE_running_increase_calculation_l2097_209740


namespace NUMINAMATH_CALUDE_average_marks_chem_math_l2097_209775

/-- Given that the total marks in physics, chemistry, and mathematics is 140 more than
    the marks in physics, prove that the average mark in chemistry and mathematics is 70. -/
theorem average_marks_chem_math (P C M : ℕ) (h : P + C + M = P + 140) :
  (C + M) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chem_math_l2097_209775


namespace NUMINAMATH_CALUDE_one_diagonal_polygon_has_four_edges_edges_equal_vertices_one_diagonal_polygon_four_edges_l2097_209780

/-- A polygon is a shape with straight sides and angles. -/
structure Polygon where
  vertices : ℕ
  vertices_positive : vertices > 0

/-- A diagonal in a polygon is a line segment that connects two non-adjacent vertices. -/
def diagonals_from_vertex (p : Polygon) : ℕ := p.vertices - 3

/-- A polygon where only one diagonal can be drawn from a single vertex has 4 edges. -/
theorem one_diagonal_polygon_has_four_edges (p : Polygon) 
  (h : diagonals_from_vertex p = 1) : p.vertices = 4 := by
  sorry

/-- The number of edges in a polygon is equal to its number of vertices. -/
theorem edges_equal_vertices (p : Polygon) : 
  (number_of_edges : ℕ) → number_of_edges = p.vertices := by
  sorry

/-- A polygon where only one diagonal can be drawn from a single vertex has 4 edges. -/
theorem one_diagonal_polygon_four_edges (p : Polygon) 
  (h : diagonals_from_vertex p = 1) : (number_of_edges : ℕ) → number_of_edges = 4 := by
  sorry

end NUMINAMATH_CALUDE_one_diagonal_polygon_has_four_edges_edges_equal_vertices_one_diagonal_polygon_four_edges_l2097_209780


namespace NUMINAMATH_CALUDE_nell_card_count_l2097_209725

/-- The number of cards Nell has after receiving cards from Jeff -/
def total_cards (initial_cards given_cards : ℝ) : ℝ :=
  initial_cards + given_cards

/-- Theorem stating that Nell's total cards equal the sum of her initial cards and those given by Jeff -/
theorem nell_card_count (initial_cards given_cards : ℝ) :
  total_cards initial_cards given_cards = initial_cards + given_cards :=
by sorry

end NUMINAMATH_CALUDE_nell_card_count_l2097_209725


namespace NUMINAMATH_CALUDE_counterexamples_count_l2097_209714

/-- Definition of digit sum -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Definition to check if a number has zero as a digit -/
def hasZeroDigit (n : ℕ) : Prop := sorry

/-- Definition of a prime number -/
def isPrime (n : ℕ) : Prop := sorry

theorem counterexamples_count :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, n % 2 = 1 ∧ digitSum n = 4 ∧ ¬hasZeroDigit n ∧ ¬isPrime n) ∧
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_counterexamples_count_l2097_209714


namespace NUMINAMATH_CALUDE_quadratic_root_value_l2097_209731

theorem quadratic_root_value (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 7 * x + k = 0 ↔ x = (7 + Real.sqrt 17) / 4 ∨ x = (7 - Real.sqrt 17) / 4) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l2097_209731


namespace NUMINAMATH_CALUDE_shadow_length_indeterminate_l2097_209758

/-- Represents a person's shadow length under different light sources -/
structure Shadow where
  sunLength : ℝ
  streetLightLength : ℝ → ℝ

/-- The theorem states that given Xiao Ming's shadow is longer than Xiao Qiang's under sunlight,
    it's impossible to determine their relative shadow lengths under a streetlight -/
theorem shadow_length_indeterminate 
  (xiaoming xioaqiang : Shadow)
  (h_sun : xiaoming.sunLength > xioaqiang.sunLength) :
  ∃ (d₁ d₂ : ℝ), 
    xiaoming.streetLightLength d₁ > xioaqiang.streetLightLength d₂ ∧
    ∃ (d₃ d₄ : ℝ), 
      xiaoming.streetLightLength d₃ < xioaqiang.streetLightLength d₄ :=
by sorry

end NUMINAMATH_CALUDE_shadow_length_indeterminate_l2097_209758


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l2097_209703

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- Slope of the asymptote -/
  asymptote_slope : ℝ
  /-- Half of the focal length -/
  c : ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  (∃ (x y : ℝ), x^2 / 36 - y^2 / 64 = 1) ∨ 
  (∃ (x y : ℝ), y^2 / 64 - x^2 / 36 = 1)

/-- Theorem stating the standard equation of a hyperbola given its asymptote slope and focal length -/
theorem hyperbola_standard_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 4/3)
  (h_focal_length : h.c = 10) : 
  standard_equation h :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l2097_209703


namespace NUMINAMATH_CALUDE_air_density_scientific_notation_l2097_209745

/-- The mass per unit volume of air in grams per cubic centimeter -/
def air_density : ℝ := 0.00124

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem air_density_scientific_notation :
  to_scientific_notation air_density = ScientificNotation.mk 1.24 (-3) sorry :=
sorry

end NUMINAMATH_CALUDE_air_density_scientific_notation_l2097_209745


namespace NUMINAMATH_CALUDE_charity_event_selection_methods_l2097_209709

def total_students : ℕ := 10
def selected_students : ℕ := 4
def special_students : ℕ := 2  -- A and B

-- Function to calculate the number of ways to select k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem charity_event_selection_methods :
  (choose (total_students - special_students) (selected_students - special_students) +
   choose (total_students - special_students) (selected_students - 1) * special_students) = 140 :=
by sorry

end NUMINAMATH_CALUDE_charity_event_selection_methods_l2097_209709


namespace NUMINAMATH_CALUDE_complex_number_problem_l2097_209705

theorem complex_number_problem (z : ℂ) (h : 2 * z + Complex.abs z = 3 + 6 * Complex.I) :
  z = 3 * Complex.I ∧ 
  ∀ (b c : ℝ), (z ^ 2 + b * z + c = 0) → (b - c = -9) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2097_209705


namespace NUMINAMATH_CALUDE_last_digits_l2097_209782

theorem last_digits (n : ℕ) : 
  (6^811 : ℕ) % 10 = 6 ∧ 
  (2^1000 : ℕ) % 10 = 6 ∧ 
  (3^999 : ℕ) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digits_l2097_209782


namespace NUMINAMATH_CALUDE_noah_holidays_l2097_209756

/-- The number of holidays Noah takes per month -/
def holidays_per_month : ℕ := 3

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Noah takes in a year -/
def total_holidays : ℕ := holidays_per_month * months_in_year

theorem noah_holidays : total_holidays = 36 := by
  sorry

end NUMINAMATH_CALUDE_noah_holidays_l2097_209756


namespace NUMINAMATH_CALUDE_problem_statement_l2097_209720

theorem problem_statement (a b : ℝ) 
  (h1 : a + 1 / (a + 1) = b + 1 / (b - 1) - 2)
  (h2 : a - b + 2 ≠ 0) : 
  a * b - a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2097_209720


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2097_209713

theorem inequality_solution_set :
  ∀ x : ℝ, (x / 4 - 1 ≤ 3 + x ∧ 3 + x < 1 - 3 * (2 + x)) ↔ x ∈ Set.Icc (-16/3) (-2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2097_209713


namespace NUMINAMATH_CALUDE_hyperbola_condition_ellipse_condition_circle_possibility_straight_lines_condition_l2097_209708

-- Define the curve C
def C (m n : ℝ) (x y : ℝ) : Prop := m * x^2 - n * y^2 = 1

-- Statement 1: If mn > 0, then C is a hyperbola
theorem hyperbola_condition (m n : ℝ) (h : m * n > 0) :
  ∃ (a b : ℝ), ∀ (x y : ℝ), C m n x y ↔ (x / a)^2 - (y / b)^2 = 1 :=
sorry

-- Statement 2: If m > 0 and m + n < 0, then C is an ellipse with foci on x-axis
theorem ellipse_condition (m n : ℝ) (h1 : m > 0) (h2 : m + n < 0) :
  ∃ (a b : ℝ), a > b ∧ ∀ (x y : ℝ), C m n x y ↔ (x / a)^2 + (y / b)^2 = 1 :=
sorry

-- Statement 3: It's not always true that if m > 0 and n < 0, then C cannot represent a circle
theorem circle_possibility (m n : ℝ) (h1 : m > 0) (h2 : n < 0) :
  ¬(∀ (r : ℝ), ¬(∀ (x y : ℝ), C m n x y ↔ x^2 + y^2 = r^2)) :=
sorry

-- Statement 4: If m > 0 and n = 0, then C consists of two straight lines
theorem straight_lines_condition (m : ℝ) (h : m > 0) :
  ∃ (k : ℝ), ∀ (x y : ℝ), C m 0 x y ↔ (x = k ∨ x = -k) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_ellipse_condition_circle_possibility_straight_lines_condition_l2097_209708


namespace NUMINAMATH_CALUDE_convex_bodies_with_coinciding_projections_intersect_l2097_209783

/-- A convex body in 3D space -/
structure ConvexBody3D where
  -- Add necessary fields/axioms for a convex body

/-- Projection of a convex body onto a coordinate plane -/
def projection (body : ConvexBody3D) (plane : Fin 3) : Set (Fin 2 → ℝ) :=
  sorry

/-- Two convex bodies intersect if they have a common point -/
def intersect (body1 body2 : ConvexBody3D) : Prop :=
  sorry

/-- Main theorem: If two convex bodies have coinciding projections on all coordinate planes, 
    then they must intersect -/
theorem convex_bodies_with_coinciding_projections_intersect 
  (body1 body2 : ConvexBody3D) 
  (h : ∀ (plane : Fin 3), projection body1 plane = projection body2 plane) : 
  intersect body1 body2 :=
sorry

end NUMINAMATH_CALUDE_convex_bodies_with_coinciding_projections_intersect_l2097_209783


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2097_209702

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 2 < 0 ↔ 1 < x ∧ x < 2) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2097_209702


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2097_209724

def a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def c : Fin 2 → ℝ := ![2, -4]

theorem vector_sum_magnitude : 
  ∃ (x y : ℝ), 
    (∀ i : Fin 2, (a x) i * c i = 0) ∧ 
    (∃ (k : ℝ), ∀ i : Fin 2, (b y) i = k * c i) →
    Real.sqrt ((a x 0 + b y 0)^2 + (a x 1 + b y 1)^2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2097_209724


namespace NUMINAMATH_CALUDE_students_at_higher_fee_l2097_209793

/-- Represents the inverse proportionality between number of students and tuition fee -/
def inverse_proportional (s f : ℝ) : Prop := ∃ k : ℝ, s * f = k

/-- Theorem: Given inverse proportionality and initial conditions, prove the number of students at $2500 -/
theorem students_at_higher_fee 
  (s₁ s₂ f₁ f₂ : ℝ) 
  (h_inverse : inverse_proportional s₁ f₁ ∧ inverse_proportional s₂ f₂)
  (h_initial : s₁ = 40 ∧ f₁ = 2000)
  (h_new_fee : f₂ = 2500) :
  s₂ = 32 := by
  sorry

end NUMINAMATH_CALUDE_students_at_higher_fee_l2097_209793


namespace NUMINAMATH_CALUDE_symmetric_points_x_axis_l2097_209755

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that the y-coordinate of A determines m to be 1. -/
theorem symmetric_points_x_axis (m : ℝ) : 
  let A : ℝ × ℝ := (-3, 2*m - 1)
  let B : ℝ × ℝ := (-3, -1)
  (A.1 = B.1 ∧ A.2 = -B.2) → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_x_axis_l2097_209755


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2097_209766

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  seq_def : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem arithmetic_sequence_12th_term 
  (seq : ArithmeticSequence) 
  (sum7 : sum_n seq 7 = 7)
  (term79 : seq.a 7 + seq.a 9 = 16) : 
  seq.a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2097_209766


namespace NUMINAMATH_CALUDE_five_T_three_equals_38_l2097_209710

-- Define the operation T
def T (a b : ℕ) : ℕ := 4 * a + 6 * b

-- Theorem to prove
theorem five_T_three_equals_38 : T 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_equals_38_l2097_209710


namespace NUMINAMATH_CALUDE_unique_c_for_complex_equation_l2097_209794

theorem unique_c_for_complex_equation : 
  ∃! c : ℝ, Complex.abs (1 - 2*I - (c - 3*I)) = 1 := by sorry

end NUMINAMATH_CALUDE_unique_c_for_complex_equation_l2097_209794


namespace NUMINAMATH_CALUDE_gcd_of_product_3000_not_15_l2097_209737

theorem gcd_of_product_3000_not_15 (a b c : ℕ+) : 
  a * b * c = 3000 → Nat.gcd (a:ℕ) (Nat.gcd (b:ℕ) (c:ℕ)) ≠ 15 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_product_3000_not_15_l2097_209737


namespace NUMINAMATH_CALUDE_circle_change_l2097_209718

/-- Represents the properties of a circle before and after diameter increase -/
structure CircleChange where
  d : ℝ  -- Initial diameter
  Q : ℝ  -- Increase in circumference

/-- Theorem stating the increase in circumference and area when diameter increases by 2π -/
theorem circle_change (c : CircleChange) :
  c.Q = 2 * Real.pi ^ 2 ∧
  (π * ((c.d + 2 * π) / 2) ^ 2 - π * (c.d / 2) ^ 2) = π ^ 2 * c.d + π ^ 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_change_l2097_209718


namespace NUMINAMATH_CALUDE_face_ratio_is_four_thirds_l2097_209751

/-- A polyhedron with triangular and square faces -/
structure Polyhedron where
  triangular_faces : ℕ
  square_faces : ℕ
  no_shared_square_edges : Bool
  no_shared_triangle_edges : Bool

/-- The ratio of triangular faces to square faces in a polyhedron -/
def face_ratio (p : Polyhedron) : ℚ :=
  p.triangular_faces / p.square_faces

/-- Theorem: The ratio of triangular faces to square faces is 4:3 -/
theorem face_ratio_is_four_thirds (p : Polyhedron) 
  (h1 : p.no_shared_square_edges = true) 
  (h2 : p.no_shared_triangle_edges = true) : 
  face_ratio p = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_face_ratio_is_four_thirds_l2097_209751


namespace NUMINAMATH_CALUDE_examination_student_count_l2097_209759

/-- The total number of students who appeared for the examination -/
def total_students : ℕ := 740

/-- The number of students who failed the examination -/
def failed_students : ℕ := 481

/-- The proportion of students who passed the examination -/
def pass_rate : ℚ := 35 / 100

theorem examination_student_count : 
  total_students = failed_students / (1 - pass_rate) := by
  sorry

end NUMINAMATH_CALUDE_examination_student_count_l2097_209759


namespace NUMINAMATH_CALUDE_sunflower_seed_contest_l2097_209789

theorem sunflower_seed_contest (player1 player2 player3 player4 total : ℕ) :
  player1 = 78 →
  player2 = 53 →
  player3 = player2 + 30 →
  player4 = 2 * player3 →
  total = player1 + player2 + player3 + player4 →
  total = 380 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_contest_l2097_209789


namespace NUMINAMATH_CALUDE_remaining_difference_l2097_209768

def recipe_flour : ℕ := 9
def recipe_sugar : ℕ := 6
def sugar_added : ℕ := 4

theorem remaining_difference : 
  recipe_flour - (recipe_sugar - sugar_added) = 7 := by
  sorry

end NUMINAMATH_CALUDE_remaining_difference_l2097_209768


namespace NUMINAMATH_CALUDE_prime_divisor_form_l2097_209752

theorem prime_divisor_form (p q : ℕ) (hp : Prime p) (hp2 : p > 2) (hq : Prime q) 
  (hdiv : q ∣ (2^p - 1)) : ∃ k : ℕ, q = 2*k*p + 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_form_l2097_209752


namespace NUMINAMATH_CALUDE_odd_function_sum_l2097_209723

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_sum (a b c : ℝ) (f : ℝ → ℝ) :
  IsOdd f →
  (∀ x, f x = x^2 * Real.sin x + c - 3) →
  (∀ x, x ∈ Set.Icc (a + 2) b → f x ≠ 0) →
  b > a + 2 →
  a + b + c = 1 := by sorry

end NUMINAMATH_CALUDE_odd_function_sum_l2097_209723


namespace NUMINAMATH_CALUDE_S_min_at_24_l2097_209732

/-- The sequence term a_n -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum S_n of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n ^ 2 - 48 * n

/-- Theorem stating that S_n is minimized when n = 24 -/
theorem S_min_at_24 : ∀ n : ℕ, S 24 ≤ S n := by sorry

end NUMINAMATH_CALUDE_S_min_at_24_l2097_209732


namespace NUMINAMATH_CALUDE_negation_of_p_l2097_209776

/-- Proposition p: a and b are both even numbers -/
def p (a b : ℤ) : Prop := Even a ∧ Even b

/-- The negation of proposition p -/
theorem negation_of_p (a b : ℤ) : ¬(p a b) ↔ ¬(Even a ∧ Even b) := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l2097_209776


namespace NUMINAMATH_CALUDE_problem_solution_l2097_209778

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c + 4 - d) → d = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2097_209778


namespace NUMINAMATH_CALUDE_season_games_count_l2097_209716

/-- Represents a sports league with the given structure -/
structure SportsLeague where
  num_divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games in a complete season -/
def total_games (league : SportsLeague) : Nat :=
  let total_teams := league.num_divisions * league.teams_per_division
  let intra_division_total := league.num_divisions * (league.teams_per_division * (league.teams_per_division - 1) / 2) * league.intra_division_games
  let inter_division_total := (total_teams * (total_teams - league.teams_per_division) / 2) * league.inter_division_games
  intra_division_total + inter_division_total

/-- The theorem to be proved -/
theorem season_games_count : 
  let league := SportsLeague.mk 3 6 3 2
  total_games league = 351 := by
  sorry

end NUMINAMATH_CALUDE_season_games_count_l2097_209716


namespace NUMINAMATH_CALUDE_circle_radius_with_inscribed_dodecagon_l2097_209722

theorem circle_radius_with_inscribed_dodecagon (Q : ℝ) (R : ℝ) : 
  (R > 0) → 
  (π * R^2 = Q + 3 * R^2) → 
  R = Real.sqrt (Q / (π - 3)) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_inscribed_dodecagon_l2097_209722


namespace NUMINAMATH_CALUDE_line_slope_is_three_fifths_l2097_209711

/-- Given two points A and B on a line l, prove that the slope of l is 3/5 -/
theorem line_slope_is_three_fifths (A B : ℝ × ℝ) : 
  (A.2 = 2) →  -- A lies on y = 2
  (B.1 - B.2 - 1 = 0) →  -- B lies on x - y - 1 = 0
  ((A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = -1) →  -- Midpoint of AB is (2, -1)
  (B.2 - A.2) / (B.1 - A.1) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_line_slope_is_three_fifths_l2097_209711


namespace NUMINAMATH_CALUDE_union_eq_P_l2097_209771

def M : Set ℝ := {x : ℝ | x > 1}
def P : Set ℝ := {x : ℝ | x > 1 ∨ x < -1}

theorem union_eq_P : M ∪ P = P := by
  sorry

end NUMINAMATH_CALUDE_union_eq_P_l2097_209771


namespace NUMINAMATH_CALUDE_sum_of_eight_smallest_multiples_of_12_l2097_209744

theorem sum_of_eight_smallest_multiples_of_12 : 
  (Finset.range 8).sum (λ i => 12 * (i + 1)) = 432 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_eight_smallest_multiples_of_12_l2097_209744


namespace NUMINAMATH_CALUDE_field_ratio_l2097_209774

theorem field_ratio (field_length : ℝ) (pond_side : ℝ) (pond_area_ratio : ℝ) :
  field_length = 96 →
  pond_side = 8 →
  pond_area_ratio = 1 / 72 →
  (pond_side * pond_side) * (1 / pond_area_ratio) = field_length * (field_length / 2) →
  field_length / (field_length / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_ratio_l2097_209774


namespace NUMINAMATH_CALUDE_math_textbooks_in_one_box_l2097_209765

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 4
def boxes : ℕ := 3
def books_per_box : ℕ := 5

def probability_all_math_in_one_box : ℚ := 769 / 100947

theorem math_textbooks_in_one_box :
  let total_ways := (total_textbooks.choose books_per_box) * 
                    ((total_textbooks - books_per_box).choose books_per_box) * 
                    ((total_textbooks - 2 * books_per_box).choose books_per_box)
  let favorable_ways := boxes * 
                        ((total_textbooks - math_textbooks).choose 1) * 
                        ((total_textbooks - math_textbooks - 1).choose books_per_box) * 
                        ((total_textbooks - math_textbooks - 1 - books_per_box).choose books_per_box)
  (favorable_ways : ℚ) / total_ways = probability_all_math_in_one_box := by
  sorry

end NUMINAMATH_CALUDE_math_textbooks_in_one_box_l2097_209765


namespace NUMINAMATH_CALUDE_bisector_sum_ratio_bound_bisector_sum_ratio_bound_tight_l2097_209798

/-- A triangle with sides a, b, c and angle bisectors l_a, l_b -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  l_a : ℝ
  l_b : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  bisector_formula_a : l_a = (2 * b * c * Real.sqrt ((1 + (b^2 + c^2 - a^2) / (2 * b * c)) / 2)) / (b + c)
  bisector_formula_b : l_b = (2 * a * c * Real.sqrt ((1 + (a^2 + c^2 - b^2) / (2 * a * c)) / 2)) / (a + c)

/-- The main theorem: the ratio of sum of bisectors to sum of sides is at most 4/3 -/
theorem bisector_sum_ratio_bound (t : Triangle) : (t.l_a + t.l_b) / (t.a + t.b) ≤ 4/3 := by
  sorry

/-- The bound 4/3 is tight -/
theorem bisector_sum_ratio_bound_tight : 
  ∀ ε > 0, ∃ t : Triangle, (t.l_a + t.l_b) / (t.a + t.b) > 4/3 - ε := by
  sorry

end NUMINAMATH_CALUDE_bisector_sum_ratio_bound_bisector_sum_ratio_bound_tight_l2097_209798


namespace NUMINAMATH_CALUDE_group_size_l2097_209749

theorem group_size (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  average_increase = 6 → old_weight = 45 → new_weight = 93 → 
  (new_weight - old_weight) / average_increase = 8 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2097_209749


namespace NUMINAMATH_CALUDE_smallest_value_z_plus_i_l2097_209712

theorem smallest_value_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 4) = Complex.abs (z * (z + 2*I))) :
  ∃ (min_val : ℝ), min_val = 1 ∧ ∀ (w : ℂ), Complex.abs (w^2 + 4) = Complex.abs (w * (w + 2*I)) →
    Complex.abs (w + I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_z_plus_i_l2097_209712


namespace NUMINAMATH_CALUDE_grants_age_l2097_209773

theorem grants_age (hospital_age : ℕ) (grant_age : ℕ) : hospital_age = 40 →
  grant_age + 5 = (2 / 3) * (hospital_age + 5) →
  grant_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_grants_age_l2097_209773


namespace NUMINAMATH_CALUDE_gcd_390_455_l2097_209779

theorem gcd_390_455 : Nat.gcd 390 455 = 65 := by sorry

end NUMINAMATH_CALUDE_gcd_390_455_l2097_209779


namespace NUMINAMATH_CALUDE_lineup_combinations_l2097_209795

def total_players : ℕ := 15
def selected_players : ℕ := 2
def lineup_size : ℕ := 5

theorem lineup_combinations :
  Nat.choose (total_players - selected_players) (lineup_size - selected_players) = 286 := by
  sorry

end NUMINAMATH_CALUDE_lineup_combinations_l2097_209795


namespace NUMINAMATH_CALUDE_range_of_function_l2097_209764

open Real

theorem range_of_function (x : ℝ) (h : 0 < x ∧ x < π/2) :
  let y := sin x - 2 * cos x + 32 / (125 * sin x * (1 - cos x))
  ∀ z, y ≥ z → z ≥ 2/5 := by sorry

end NUMINAMATH_CALUDE_range_of_function_l2097_209764


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2097_209781

theorem complex_fraction_simplification :
  (Complex.I + 1) / (1 - Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2097_209781


namespace NUMINAMATH_CALUDE_two_fifths_divided_by_one_fifth_l2097_209743

theorem two_fifths_divided_by_one_fifth : (2 : ℚ) / 5 / ((1 : ℚ) / 5) = 2 := by sorry

end NUMINAMATH_CALUDE_two_fifths_divided_by_one_fifth_l2097_209743


namespace NUMINAMATH_CALUDE_xiaohui_pe_score_l2097_209760

/-- Calculates the total physical education score based on component scores and weights -/
def calculate_pe_score (max_score : ℝ) (morning_weight : ℝ) (midterm_weight : ℝ) (final_weight : ℝ)
                       (morning_score : ℝ) (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  morning_score * morning_weight + midterm_score * midterm_weight + final_score * final_weight

/-- Theorem stating that Xiaohui's physical education score is 88.5 points -/
theorem xiaohui_pe_score :
  let max_score : ℝ := 100
  let morning_weight : ℝ := 0.2
  let midterm_weight : ℝ := 0.3
  let final_weight : ℝ := 0.5
  let morning_score : ℝ := 95
  let midterm_score : ℝ := 90
  let final_score : ℝ := 85
  calculate_pe_score max_score morning_weight midterm_weight final_weight
                     morning_score midterm_score final_score = 88.5 := by
  sorry

#eval calculate_pe_score 100 0.2 0.3 0.5 95 90 85

end NUMINAMATH_CALUDE_xiaohui_pe_score_l2097_209760


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2097_209733

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only {2, 2, 3} cannot form a right-angled triangle -/
theorem right_triangle_sets :
  is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3) ∧
  is_right_triangle 3 4 5 ∧
  is_right_triangle 5 12 13 ∧
  ¬is_right_triangle 2 2 3 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_sets_l2097_209733


namespace NUMINAMATH_CALUDE_square_value_l2097_209785

/-- Given that square times 3a equals -3a^2b, prove that square equals -ab -/
theorem square_value (a b : ℝ) (square : ℝ) (h : square * 3 * a = -3 * a^2 * b) :
  square = -a * b := by sorry

end NUMINAMATH_CALUDE_square_value_l2097_209785


namespace NUMINAMATH_CALUDE_perimeter_ratio_l2097_209736

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- The original large rectangle -/
def largeRectangle : Rectangle := { width := 4, height := 6 }

/-- One of the small rectangles after folding and cutting -/
def smallRectangle : Rectangle := { width := 2, height := 3 }

/-- Theorem stating the ratio of perimeters -/
theorem perimeter_ratio :
  perimeter smallRectangle / perimeter largeRectangle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_l2097_209736


namespace NUMINAMATH_CALUDE_optimal_laundry_additions_l2097_209704

-- Define constants
def total_capacity : ℝ := 20
def clothes_weight : ℝ := 5
def initial_detergent_scoops : ℝ := 2
def scoop_weight : ℝ := 0.02
def optimal_ratio : ℝ := 0.004  -- 4 g per kg = 0.004 kg per kg

-- Define the problem
theorem optimal_laundry_additions 
  (h1 : total_capacity = 20)
  (h2 : clothes_weight = 5)
  (h3 : initial_detergent_scoops = 2)
  (h4 : scoop_weight = 0.02)
  (h5 : optimal_ratio = 0.004) :
  ∃ (additional_detergent additional_water : ℝ),
    -- The total weight matches the capacity
    clothes_weight + initial_detergent_scoops * scoop_weight + additional_detergent + additional_water = total_capacity ∧
    -- The ratio of total detergent to water is optimal
    (initial_detergent_scoops * scoop_weight + additional_detergent) / additional_water = optimal_ratio ∧
    -- The additional detergent is 0.02 kg
    additional_detergent = 0.02 ∧
    -- The additional water is 14.94 kg
    additional_water = 14.94 :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_laundry_additions_l2097_209704


namespace NUMINAMATH_CALUDE_batsman_running_percentage_l2097_209721

def batsman_score_percentage (total_runs : ℕ) (boundaries : ℕ) (sixes : ℕ) : ℚ :=
  let boundary_runs := 4 * boundaries
  let six_runs := 6 * sixes
  let runs_without_running := boundary_runs + six_runs
  let runs_by_running := total_runs - runs_without_running
  (runs_by_running : ℚ) / total_runs * 100

theorem batsman_running_percentage :
  batsman_score_percentage 120 3 8 = 50 := by sorry

end NUMINAMATH_CALUDE_batsman_running_percentage_l2097_209721


namespace NUMINAMATH_CALUDE_speed_conversion_l2097_209788

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The speed of the train in meters per second -/
def train_speed_mps : ℝ := 20.0016

/-- The speed of the train in kilometers per hour -/
def train_speed_kmph : ℝ := train_speed_mps * mps_to_kmph

theorem speed_conversion :
  train_speed_kmph = 72.00576 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l2097_209788


namespace NUMINAMATH_CALUDE_cube_painting_cost_l2097_209770

/-- The cost of painting a cube's surface area given its volume and paint cost per area unit -/
theorem cube_painting_cost (volume : ℝ) (cost_per_area : ℝ) : 
  volume = 9261 → 
  cost_per_area = 13 / 100 →
  6 * (volume ^ (1/3))^2 * cost_per_area = 344.98 := by
sorry

end NUMINAMATH_CALUDE_cube_painting_cost_l2097_209770


namespace NUMINAMATH_CALUDE_communication_scenarios_10_20_l2097_209799

/-- The number of different possible communication scenarios between two groups of radio operators. -/
def communication_scenarios (operators_a : ℕ) (operators_b : ℕ) : ℕ :=
  2^(operators_a * operators_b)

/-- Theorem stating the number of communication scenarios for 10 operators at A and 20 at B. -/
theorem communication_scenarios_10_20 :
  communication_scenarios 10 20 = 2^200 := by
  sorry

end NUMINAMATH_CALUDE_communication_scenarios_10_20_l2097_209799


namespace NUMINAMATH_CALUDE_f_min_value_l2097_209739

/-- The quadratic function f(x) = 2x^2 - 8x + 9 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 9

/-- The minimum value of f(x) is 1 -/
theorem f_min_value : ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l2097_209739


namespace NUMINAMATH_CALUDE_equal_area_triangles_l2097_209787

theorem equal_area_triangles (b c : ℝ) (h₁ : b > 0) (h₂ : c > 0) :
  let k : ℝ := (Real.sqrt 5 - 1) * c / 2
  let l : ℝ := (Real.sqrt 5 - 1) * b / 2
  let area_ABK : ℝ := b * k / 2
  let area_AKL : ℝ := l * c / 2
  let area_ADL : ℝ := (b * c - k * l) / 2
  area_ABK = area_AKL ∧ area_AKL = area_ADL := by sorry

end NUMINAMATH_CALUDE_equal_area_triangles_l2097_209787


namespace NUMINAMATH_CALUDE_birthday_friends_count_l2097_209746

theorem birthday_friends_count : ∃ (n : ℕ), 
  (12 * (n + 2) = 16 * n) ∧ 
  (∀ m : ℕ, 12 * (m + 2) = 16 * m → m = n) :=
by sorry

end NUMINAMATH_CALUDE_birthday_friends_count_l2097_209746


namespace NUMINAMATH_CALUDE_g_of_seven_l2097_209715

theorem g_of_seven (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 := by
  sorry

end NUMINAMATH_CALUDE_g_of_seven_l2097_209715


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2097_209742

theorem arithmetic_calculations :
  ((1 : ℝ) - 12 + (-6) - (-28) = 10) ∧
  ((2 : ℝ) - 3^2 + (7/8 - 1) * (-2)^2 = -9.5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2097_209742


namespace NUMINAMATH_CALUDE_fifteenSidedFigureArea_l2097_209729

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
def Polygon := List Point

/-- The 15-sided figure described in the problem -/
def fifteenSidedFigure : Polygon := [
  ⟨1, 2⟩, ⟨2, 2⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 4⟩, ⟨5, 5⟩, ⟨6, 5⟩, ⟨7, 4⟩,
  ⟨6, 3⟩, ⟨6, 2⟩, ⟨5, 1⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨1, 2⟩
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ :=
  sorry

/-- Theorem stating that the area of the 15-sided figure is 15 cm² -/
theorem fifteenSidedFigureArea :
  calculateArea fifteenSidedFigure = 15 := by sorry

end NUMINAMATH_CALUDE_fifteenSidedFigureArea_l2097_209729


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l2097_209748

/-- Calculate the total interest earned on an investment with compound interest -/
def totalInterestEarned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- Theorem: The total interest earned on $2,000 at 8% annual interest rate after 5 years is approximately $938.66 -/
theorem investment_interest_calculation :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let years : ℕ := 5
  abs (totalInterestEarned principal rate years - 938.66) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l2097_209748


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l2097_209735

/-- Given two points C(3, 7) and D(8, 10), prove that the sum of the slope and y-intercept
    of the line passing through these points is 29/5 -/
theorem line_slope_intercept_sum (C D : ℝ × ℝ) : 
  C = (3, 7) → D = (8, 10) → 
  let m := (D.2 - C.2) / (D.1 - C.1)
  let b := C.2 - m * C.1
  m + b = 29/5 := by sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l2097_209735


namespace NUMINAMATH_CALUDE_mixed_oil_rate_l2097_209784

/-- Given two oils mixed together, calculate the rate of the mixed oil per litre -/
theorem mixed_oil_rate (volume1 volume2 rate1 rate2 : ℚ) 
  (h1 : volume1 = 10)
  (h2 : volume2 = 5)
  (h3 : rate1 = 40)
  (h4 : rate2 = 66) :
  (volume1 * rate1 + volume2 * rate2) / (volume1 + volume2) = 730 / 15 := by
  sorry

end NUMINAMATH_CALUDE_mixed_oil_rate_l2097_209784


namespace NUMINAMATH_CALUDE_spa_nail_polish_inconsistency_l2097_209767

theorem spa_nail_polish_inconsistency :
  ∀ (n : ℕ), n * 20 ≠ 25 :=
by
  sorry

#check spa_nail_polish_inconsistency

end NUMINAMATH_CALUDE_spa_nail_polish_inconsistency_l2097_209767


namespace NUMINAMATH_CALUDE_e₁_e₂_divisibility_l2097_209750

def e₁ (a : ℕ) : ℕ := a^2 + 3^a + a * 3^((a + 1) / 2)
def e₂ (a : ℕ) : ℕ := a^2 + 3^a - a * 3^((a + 1) / 2)

theorem e₁_e₂_divisibility (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 25) :
  (e₁ a * e₂ a) % 3 = 0 ↔ a % 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_e₁_e₂_divisibility_l2097_209750


namespace NUMINAMATH_CALUDE_circle_centers_characterization_l2097_209707

/-- Given a circle k with radius 1 centered at the origin and diameter e on the x-axis,
    this function characterizes the set of centers (x, y) of circles tangent to e
    with their nearest point to k at a distance equal to their radius -/
def circle_centers (x y : ℝ) : Prop :=
  (9 * (y + 2/3)^2 - 3 * x^2 = 1) ∨ (9 * (y - 2/3)^2 - 3 * x^2 = 1)

/-- Theorem stating that the function circle_centers correctly characterizes
    the set of centers of circles satisfying the given conditions -/
theorem circle_centers_characterization :
  ∀ x y : ℝ, circle_centers x y ↔
    (∃ r : ℝ, r > 0 ∧
      ((x^2 + y^2 = (2*r + 1)^2 ∧ y = r) ∨
       (x^2 + y^2 = (1 - 2*r)^2 ∧ y = r))) :=
  sorry

end NUMINAMATH_CALUDE_circle_centers_characterization_l2097_209707


namespace NUMINAMATH_CALUDE_exists_all_berries_l2097_209726

/-- A binary vector of length 7 -/
def BinaryVector := Fin 7 → Bool

/-- The set of 16 vectors representing the work schedule -/
def WorkSchedule := Fin 16 → BinaryVector

/-- The condition that the first vector is all zeros -/
def firstDayAllMine (schedule : WorkSchedule) : Prop :=
  ∀ i : Fin 7, schedule 0 i = false

/-- The condition that any two vectors differ in at least 3 positions -/
def atLeastThreeDifferences (schedule : WorkSchedule) : Prop :=
  ∀ d1 d2 : Fin 16, d1 ≠ d2 →
    (Finset.filter (fun i => schedule d1 i ≠ schedule d2 i) Finset.univ).card ≥ 3

/-- The theorem to be proved -/
theorem exists_all_berries (schedule : WorkSchedule)
  (h1 : firstDayAllMine schedule)
  (h2 : atLeastThreeDifferences schedule) :
  ∃ d : Fin 16, ∀ i : Fin 7, schedule d i = true := by
  sorry

end NUMINAMATH_CALUDE_exists_all_berries_l2097_209726


namespace NUMINAMATH_CALUDE_sarahs_age_l2097_209763

/-- Given a person (Sarah) who is 18 years younger than her mother, 
    and the sum of their ages is 50 years, Sarah's age is 16 years. -/
theorem sarahs_age (s m : ℕ) : s = m - 18 ∧ s + m = 50 → s = 16 := by
  sorry

end NUMINAMATH_CALUDE_sarahs_age_l2097_209763


namespace NUMINAMATH_CALUDE_cube_sum_equality_l2097_209717

theorem cube_sum_equality (a b : ℝ) (h : a + b = 4) : a^3 + 12*a*b + b^3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l2097_209717


namespace NUMINAMATH_CALUDE_cubic_difference_l2097_209777

theorem cubic_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 - y^3 = 176 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l2097_209777


namespace NUMINAMATH_CALUDE_fathers_age_multiple_l2097_209796

theorem fathers_age_multiple (sons_age : ℕ) (multiple : ℕ) : 
  (44 = multiple * sons_age + 4) →
  (44 + 4 = 2 * (sons_age + 4) + 20) →
  multiple = 4 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_multiple_l2097_209796


namespace NUMINAMATH_CALUDE_book_cost_price_l2097_209753

theorem book_cost_price (final_price : ℝ) (profit_rate : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) 
  (h1 : final_price = 250)
  (h2 : profit_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : discount_rate = 0.05) : 
  ∃ (cost_price : ℝ), cost_price = final_price / ((1 + profit_rate) * (1 - discount_rate) * (1 + tax_rate)) :=
by sorry

end NUMINAMATH_CALUDE_book_cost_price_l2097_209753


namespace NUMINAMATH_CALUDE_mrs_jane_total_coins_l2097_209738

def total_coins (jayden_coins jason_coins : ℕ) : ℕ :=
  jayden_coins + jason_coins

theorem mrs_jane_total_coins : 
  let jayden_coins : ℕ := 300
  let jason_coins : ℕ := jayden_coins + 60
  total_coins jayden_coins jason_coins = 660 := by
  sorry

end NUMINAMATH_CALUDE_mrs_jane_total_coins_l2097_209738


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l2097_209791

/-- Given complex numbers z₁, z₂, z₃ that satisfy certain conditions,
    prove that z₁z₂/z₃ = -5. -/
theorem complex_ratio_theorem (z₁ z₂ z₃ : ℂ)
  (h1 : Complex.abs z₁ = Complex.abs z₂)
  (h2 : Complex.abs z₁ = Real.sqrt 3 * Complex.abs z₃)
  (h3 : z₁ + z₃ = z₂) :
  z₁ * z₂ / z₃ = -5 := by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l2097_209791


namespace NUMINAMATH_CALUDE_max_height_triangle_def_l2097_209797

/-- Triangle DEF with sides a, b, c -/
structure Triangle (a b c : ℝ) where
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The maximum possible height of a table constructed from a triangle -/
def max_table_height (t : Triangle a b c) : ℝ :=
  sorry

theorem max_height_triangle_def (t : Triangle 20 29 35) :
  max_table_height t = 84 * Real.sqrt 2002 / 64 := by
  sorry

end NUMINAMATH_CALUDE_max_height_triangle_def_l2097_209797


namespace NUMINAMATH_CALUDE_third_triangular_square_l2097_209741

/-- A number that is both triangular and square --/
def TriangularSquare (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = a * (a + 1) / 2 ∧ n = b * b

/-- The first two triangular square numbers --/
def FirstTwoTriangularSquares : Prop :=
  TriangularSquare 1 ∧ TriangularSquare 36

/-- Checks if a number is the third triangular square number --/
def IsThirdTriangularSquare (n : ℕ) : Prop :=
  TriangularSquare n ∧
  FirstTwoTriangularSquares ∧
  ∀ m : ℕ, m < n → TriangularSquare m → (m = 1 ∨ m = 36)

/-- 1225 is the third triangular square number --/
theorem third_triangular_square :
  IsThirdTriangularSquare 1225 :=
sorry

end NUMINAMATH_CALUDE_third_triangular_square_l2097_209741


namespace NUMINAMATH_CALUDE_proctoring_arrangements_l2097_209701

theorem proctoring_arrangements (n : ℕ) (k : ℕ) (h : n = 12 ∧ k = 8) :
  (Nat.choose n k) * ((n - k) * (n - k - 1)) = 4455 :=
sorry

end NUMINAMATH_CALUDE_proctoring_arrangements_l2097_209701


namespace NUMINAMATH_CALUDE_gcf_of_45_135_90_l2097_209761

theorem gcf_of_45_135_90 : Nat.gcd 45 (Nat.gcd 135 90) = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_45_135_90_l2097_209761


namespace NUMINAMATH_CALUDE_power_multiplication_l2097_209727

theorem power_multiplication (a : ℝ) : (-a^2)^3 * a^3 = -a^9 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l2097_209727


namespace NUMINAMATH_CALUDE_remainder_property_l2097_209772

theorem remainder_property (N : ℤ) : ∃ (k : ℤ), N = 35 * k + 25 → ∃ (m : ℤ), N = 15 * m + 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_property_l2097_209772


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2097_209790

/-- Given an ellipse and a circle satisfying certain conditions, prove that the eccentricity of the ellipse is 1/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 1 ∧ 
   ((x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 + y^2 = a^2) ∨ 
    (x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 - c^2 = a^2 - c^2 ∧ c^2 = a^2 - b^2))) →
  (a^2 - b^2) / a = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2097_209790


namespace NUMINAMATH_CALUDE_cyclic_iff_arithmetic_progression_l2097_209734

/-- A quadrilateral with sides a, b, d, c (in that order) -/
structure Quadrilateral :=
  (a b d c : ℝ)

/-- The property of sides forming an arithmetic progression -/
def is_arithmetic_progression (q : Quadrilateral) : Prop :=
  ∃ k : ℝ, q.b = q.a + k ∧ q.d = q.a + 2*k ∧ q.c = q.a + 3*k

/-- The property of a quadrilateral being cyclic (inscribable in a circle) -/
def is_cyclic (q : Quadrilateral) : Prop :=
  q.a + q.c = q.b + q.d

/-- Theorem: A quadrilateral is cyclic if and only if its sides form an arithmetic progression -/
theorem cyclic_iff_arithmetic_progression (q : Quadrilateral) :
  is_cyclic q ↔ is_arithmetic_progression q :=
sorry

end NUMINAMATH_CALUDE_cyclic_iff_arithmetic_progression_l2097_209734


namespace NUMINAMATH_CALUDE_impossible_three_coin_piles_l2097_209728

/-- Represents the coin removal and division process -/
def coin_process (initial_coins : ℕ) (steps : ℕ) : Prop :=
  ∃ (final_piles : ℕ),
    (initial_coins - steps = 3 * final_piles) ∧
    (final_piles = steps + 1)

/-- Theorem stating the impossibility of ending with only piles of three coins -/
theorem impossible_three_coin_piles : ¬∃ (steps : ℕ), coin_process 2013 steps :=
  sorry

end NUMINAMATH_CALUDE_impossible_three_coin_piles_l2097_209728


namespace NUMINAMATH_CALUDE_y_value_proof_l2097_209792

theorem y_value_proof (y : ℝ) (h : 9 / (y^3) = y / 81) : y = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2097_209792


namespace NUMINAMATH_CALUDE_S_at_one_l2097_209762

def R (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4

def S (x : ℝ) : ℝ := (3 + 2) * x^3 + (-5 + 2) * x + (4 + 2)

theorem S_at_one : S 1 = 8 := by sorry

end NUMINAMATH_CALUDE_S_at_one_l2097_209762


namespace NUMINAMATH_CALUDE_arithmetic_sequence_special_difference_l2097_209747

/-- An arithmetic sequence with special properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  first_term : a 1 = 1
  arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_mean : a 2 ^ 2 = a 1 * a 6

/-- The common difference of the arithmetic sequence is either 0 or 3 -/
theorem arithmetic_sequence_special_difference (seq : ArithmeticSequence) : 
  seq.d = 0 ∨ seq.d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_special_difference_l2097_209747


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2097_209700

-- Define the square
def Square := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 6}

-- Define the vertical sides of the square
def VerticalSides := {(x, y) : ℝ × ℝ | (x = 0 ∨ x = 6) ∧ 0 ≤ y ∧ y ≤ 6}

-- Define the possible jump directions
inductive Direction
| Up
| Down
| Left
| Right

-- Define a function to represent a single jump
def jump (pos : ℝ × ℝ) (dir : Direction) : ℝ × ℝ :=
  match dir with
  | Direction.Up => (pos.1, pos.2 + 2)
  | Direction.Down => (pos.1, pos.2 - 2)
  | Direction.Left => (pos.1 - 2, pos.2)
  | Direction.Right => (pos.1 + 2, pos.2)

-- Define the probability function
noncomputable def P (pos : ℝ × ℝ) : ℝ :=
  sorry  -- The actual implementation would go here

-- State the theorem
theorem frog_jump_probability :
  P (1, 3) = 2/3 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l2097_209700


namespace NUMINAMATH_CALUDE_banana_permutations_count_l2097_209769

/-- The number of unique permutations of a multiset with 6 elements,
    where one element appears 3 times, another appears 2 times,
    and the third appears once. -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

/-- Theorem stating that the number of unique permutations of "BANANA" is 60. -/
theorem banana_permutations_count : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_count_l2097_209769
