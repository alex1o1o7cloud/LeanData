import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_exterior_120_is_equilateral_equal_angles_is_equilateral_two_angles_70_40_is_isosceles_l2238_223827

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_angles : angle_a + angle_b + angle_c = 180

-- Define an isosceles triangle
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define an equilateral triangle
def EquilateralTriangle (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Statement 1
theorem isosceles_exterior_120_is_equilateral (t : Triangle) (h : IsoscelesTriangle t) :
  ∃ (ext_angle : ℝ), ext_angle = 120 → EquilateralTriangle t :=
sorry

-- Statement 2
theorem equal_angles_is_equilateral (t : Triangle) :
  t.angle_a = t.angle_b ∧ t.angle_b = t.angle_c → EquilateralTriangle t :=
sorry

-- Statement 3
theorem two_angles_70_40_is_isosceles (t : Triangle) :
  t.angle_a = 70 ∧ t.angle_b = 40 → IsoscelesTriangle t :=
sorry

end NUMINAMATH_CALUDE_isosceles_exterior_120_is_equilateral_equal_angles_is_equilateral_two_angles_70_40_is_isosceles_l2238_223827


namespace NUMINAMATH_CALUDE_dana_jayden_pencil_difference_l2238_223818

theorem dana_jayden_pencil_difference :
  ∀ (dana_pencils jayden_pencils marcus_pencils : ℕ),
    jayden_pencils = 20 →
    jayden_pencils = 2 * marcus_pencils →
    dana_pencils = marcus_pencils + 25 →
    dana_pencils - jayden_pencils = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_dana_jayden_pencil_difference_l2238_223818


namespace NUMINAMATH_CALUDE_probability_at_least_two_black_l2238_223811

/-- The number of white balls in the bag -/
def white_balls : ℕ := 5

/-- The number of black balls in the bag -/
def black_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := white_balls + black_balls

/-- The number of balls drawn from the bag -/
def drawn_balls : ℕ := 3

/-- The probability of drawing at least 2 black balls when drawing 3 balls from a bag 
    containing 5 white balls and 3 black balls -/
theorem probability_at_least_two_black : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 1 + Nat.choose black_balls 3) / 
  Nat.choose total_balls drawn_balls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_two_black_l2238_223811


namespace NUMINAMATH_CALUDE_option_C_most_suitable_l2238_223881

/-- Represents a survey option -/
inductive SurveyOption
  | A  -- Understanding the sleep time of middle school students nationwide
  | B  -- Understanding the water quality of a river
  | C  -- Surveying the vision of all classmates
  | D  -- Surveying the number of fish in a pond

/-- Defines what makes a survey comprehensive -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that option C is the most suitable for a comprehensive survey -/
theorem option_C_most_suitable :
  ∀ s : SurveyOption, isComprehensive s → s = SurveyOption.C :=
sorry

end NUMINAMATH_CALUDE_option_C_most_suitable_l2238_223881


namespace NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l2238_223850

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_to_plane_not_always_parallel 
  (l m : Line) (α : Plane) : 
  ¬(∀ l m α, parallel_line_plane l α → parallel_line_plane m α → parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_parallel_to_plane_not_always_parallel_l2238_223850


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2238_223878

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2238_223878


namespace NUMINAMATH_CALUDE_converse_and_inverse_false_l2238_223897

-- Define what it means for a triangle to be equilateral
def is_equilateral (triangle : Type) : Prop := sorry

-- Define what it means for a triangle to be isosceles
def is_isosceles (triangle : Type) : Prop := sorry

-- The original statement (given as true)
axiom original_statement : ∀ (triangle : Type), is_equilateral triangle → is_isosceles triangle

-- Theorem to prove
theorem converse_and_inverse_false :
  (∃ (triangle : Type), is_isosceles triangle ∧ ¬is_equilateral triangle) ∧
  (∃ (triangle : Type), ¬is_equilateral triangle ∧ is_isosceles triangle) :=
by sorry

end NUMINAMATH_CALUDE_converse_and_inverse_false_l2238_223897


namespace NUMINAMATH_CALUDE_floor_minus_x_is_zero_l2238_223893

theorem floor_minus_x_is_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌊x⌋ - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_floor_minus_x_is_zero_l2238_223893


namespace NUMINAMATH_CALUDE_square_root_sum_equality_l2238_223880

theorem square_root_sum_equality (x : ℝ) :
  Real.sqrt (5 + x) + Real.sqrt (20 - x) = 7 →
  (5 + x) * (20 - x) = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equality_l2238_223880


namespace NUMINAMATH_CALUDE_smallest_c_value_l2238_223887

theorem smallest_c_value (c d : ℝ) (h_nonneg_c : c ≥ 0) (h_nonneg_d : d ≥ 0)
  (h_cos_eq : ∀ x : ℤ, Real.cos (c * ↑x - d) = Real.cos (35 * ↑x)) :
  c ≥ 35 ∧ ∀ c' ≥ 0, (∀ x : ℤ, Real.cos (c' * ↑x - d) = Real.cos (35 * ↑x)) → c' ≥ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l2238_223887


namespace NUMINAMATH_CALUDE_m_range_l2238_223835

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*(m+1)*x + m*(m+1) > 0

theorem m_range (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) :
  m > 2 ∨ (-2 ≤ m ∧ m < -1) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2238_223835


namespace NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l2238_223807

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_three_person_subcommittees_from_eight_l2238_223807


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l2238_223870

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 1) : a / b = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l2238_223870


namespace NUMINAMATH_CALUDE_loss_fraction_l2238_223826

theorem loss_fraction (cost_price selling_price : ℚ) 
  (h1 : cost_price = 21)
  (h2 : selling_price = 20) :
  (cost_price - selling_price) / cost_price = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_loss_fraction_l2238_223826


namespace NUMINAMATH_CALUDE_high_school_students_l2238_223886

theorem high_school_students (music : ℕ) (art : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : music = 50)
  (h2 : art = 20)
  (h3 : both = 10)
  (h4 : neither = 440) :
  music + art - both + neither = 500 := by
  sorry

end NUMINAMATH_CALUDE_high_school_students_l2238_223886


namespace NUMINAMATH_CALUDE_miranda_savings_duration_l2238_223856

def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def monthly_saving : ℕ := 70

theorem miranda_savings_duration :
  (total_cost - sister_contribution) / monthly_saving = 3 := by
  sorry

end NUMINAMATH_CALUDE_miranda_savings_duration_l2238_223856


namespace NUMINAMATH_CALUDE_quadratic_roots_farthest_apart_l2238_223851

/-- The quadratic equation x^2 - 4ax + 5a^2 - 6a = 0 has roots that are farthest apart when a = 3 -/
theorem quadratic_roots_farthest_apart (a : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*a*x + 5*a^2 - 6*a
  let discriminant := 4*a*(6 - a)
  (∀ b : ℝ, discriminant ≥ 4*b*(6 - b)) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_farthest_apart_l2238_223851


namespace NUMINAMATH_CALUDE_hyperbola_intersection_trajectory_l2238_223814

theorem hyperbola_intersection_trajectory
  (x1 y1 : ℝ)
  (h_on_hyperbola : x1^2 / 2 - y1^2 = 1)
  (h_distinct : x1 ≠ -Real.sqrt 2 ∧ x1 ≠ Real.sqrt 2)
  (x y : ℝ)
  (h_intersection : ∃ (t s : ℝ),
    x = -Real.sqrt 2 + t * (x1 + Real.sqrt 2) ∧
    y = t * y1 ∧
    x = Real.sqrt 2 + s * (x1 - Real.sqrt 2) ∧
    y = -s * y1) :
  x^2 / 2 + y^2 = 1 ∧ x ≠ 0 ∧ x ≠ -Real.sqrt 2 ∧ x ≠ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_trajectory_l2238_223814


namespace NUMINAMATH_CALUDE_assistant_prof_charts_l2238_223812

theorem assistant_prof_charts (associate_profs assistant_profs : ℕ) 
  (charts_per_assistant : ℕ) :
  associate_profs + assistant_profs = 7 →
  2 * associate_profs + assistant_profs = 10 →
  associate_profs + assistant_profs * charts_per_assistant = 11 →
  charts_per_assistant = 2 :=
by sorry

end NUMINAMATH_CALUDE_assistant_prof_charts_l2238_223812


namespace NUMINAMATH_CALUDE_age_difference_l2238_223890

theorem age_difference (A B : ℕ) : B = 41 → A + 10 = 2 * (B - 10) → A - B = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2238_223890


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2238_223823

/-- Calculates the simple interest rate given the principal, time, and interest amount -/
def simple_interest_rate (principal time interest : ℚ) : ℚ :=
  (interest / (principal * time)) * 100

/-- Theorem stating that for the given conditions, the simple interest rate is 2.5% -/
theorem interest_rate_calculation :
  let principal : ℚ := 700
  let time : ℚ := 4
  let interest : ℚ := 70
  simple_interest_rate principal time interest = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2238_223823


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l2238_223821

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a - b) →
  (10 * a + b) + (10 * b + a) = 33 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l2238_223821


namespace NUMINAMATH_CALUDE_product_mod_seven_l2238_223820

theorem product_mod_seven : (2009 * 2010 * 2011 * 2012) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2238_223820


namespace NUMINAMATH_CALUDE_circle_symmetry_implies_a_value_l2238_223869

/-- A circle C with equation x^2 + y^2 + 2x + ay - 10 = 0, where a is a real number -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + a*p.2 - 10 = 0}

/-- The line l with equation x - y + 2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 2 = 0}

/-- A point is symmetric about a line if the line is the perpendicular bisector of the line segment
    joining the point and its reflection -/
def IsSymmetricAbout (p q : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  q ∈ l ∧ (p.1 + q.1) / 2 = q.1 ∧ (p.2 + q.2) / 2 = q.2

theorem circle_symmetry_implies_a_value (a : ℝ) :
  (∀ p ∈ Circle a, ∃ q, q ∈ Circle a ∧ IsSymmetricAbout p q Line) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_implies_a_value_l2238_223869


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l2238_223817

theorem triangle_abc_problem (A B C : Real) (a b c : Real) 
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2/3) : 
  b = Real.sqrt 6 ∧ Real.sin (2*B - π/3) = (4*Real.sqrt 5 + Real.sqrt 3) / 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l2238_223817


namespace NUMINAMATH_CALUDE_additional_red_flowers_needed_l2238_223828

def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

theorem additional_red_flowers_needed : white_flowers - red_flowers = 208 := by
  sorry

end NUMINAMATH_CALUDE_additional_red_flowers_needed_l2238_223828


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2238_223844

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | x^2 < 9}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2238_223844


namespace NUMINAMATH_CALUDE_inscribed_square_area_l2238_223843

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

-- Define the square
structure InscribedSquare where
  center : ℝ
  side_half : ℝ

-- Theorem statement
theorem inscribed_square_area :
  ∃ (s : InscribedSquare),
    s.center = 5 ∧
    parabola (s.center + s.side_half) = -2 * s.side_half ∧
    (2 * s.side_half)^2 = 24 - 8 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l2238_223843


namespace NUMINAMATH_CALUDE_delta_properties_l2238_223894

def delta (m n : ℚ) : ℚ := (m + n) / (1 + m * n)

theorem delta_properties :
  (delta (-4) 4 = 0) ∧
  (delta (1/3) (1/4) = delta 3 4) ∧
  ∃ (m n : ℚ), delta (-m) n ≠ delta m (-n) := by
  sorry

end NUMINAMATH_CALUDE_delta_properties_l2238_223894


namespace NUMINAMATH_CALUDE_min_dimes_needed_l2238_223837

def jacket_cost : ℚ := 45.50
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 10
def nickels : ℕ := 15

def min_dimes : ℕ := 23

theorem min_dimes_needed (d : ℕ) : 
  (ten_dollar_bills * 10 + quarters * 0.25 + nickels * 0.05 + d * 0.10 : ℚ) ≥ jacket_cost → 
  d ≥ min_dimes := by
sorry

end NUMINAMATH_CALUDE_min_dimes_needed_l2238_223837


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2238_223885

theorem min_value_sum_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 2) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2238_223885


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l2238_223804

/-- A geometric sequence with given third and sixth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_3 : a 3 = 8)
  (h_6 : a 6 = 64) :
  ∃ (q : ℝ), (∀ (n : ℕ), a (n + 1) = a n * q) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l2238_223804


namespace NUMINAMATH_CALUDE_vector_operation_l2238_223815

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (0, -1)) :
  2 • b - a = (-3, -4) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l2238_223815


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2238_223839

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_options : ℕ := 10

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of license plate combinations -/
def license_plate_combinations : ℕ :=
  alphabet_size *
  (choose (alphabet_size - 1) 2) *
  (choose letter_positions 2) *
  (digit_options * (digit_options - 1) * (digit_options - 2))

theorem license_plate_theorem :
  license_plate_combinations = 33696000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2238_223839


namespace NUMINAMATH_CALUDE_equivalent_operations_l2238_223816

theorem equivalent_operations (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (14/5) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operations_l2238_223816


namespace NUMINAMATH_CALUDE_min_sum_fraction_l2238_223858

theorem min_sum_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (7 * a) ≥ 3 / Real.rpow 105 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_fraction_l2238_223858


namespace NUMINAMATH_CALUDE_teacher_wang_travel_time_l2238_223896

theorem teacher_wang_travel_time (bicycle_speed : ℝ) (bicycle_time : ℝ) (walking_speed : ℝ) (max_walking_time : ℝ)
  (h1 : bicycle_speed = 15)
  (h2 : bicycle_time = 0.2)
  (h3 : walking_speed = 5)
  (h4 : max_walking_time = 0.7) :
  (bicycle_speed * bicycle_time) / walking_speed < max_walking_time :=
by sorry

end NUMINAMATH_CALUDE_teacher_wang_travel_time_l2238_223896


namespace NUMINAMATH_CALUDE_unique_sum_product_solution_l2238_223802

theorem unique_sum_product_solution (S P : ℝ) (h : S^2 ≥ 4*P) :
  let x₁ := (S + Real.sqrt (S^2 - 4*P)) / 2
  let y₁ := S - x₁
  let x₂ := (S - Real.sqrt (S^2 - 4*P)) / 2
  let y₂ := S - x₂
  (∀ x y : ℝ, x + y = S ∧ x * y = P ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_unique_sum_product_solution_l2238_223802


namespace NUMINAMATH_CALUDE_chord_diagonal_intersections_collinear_l2238_223829

namespace CircleChords

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a chord as a pair of points
structure Chord where
  p1 : Point
  p2 : Point

-- Define the problem setup
structure ChordConfiguration where
  circle : Circle
  chordAB : Chord
  chordCD : Chord
  chordEF : Chord
  -- Ensure chords are non-intersecting
  non_intersecting : 
    chordAB.p1 ≠ chordCD.p1 ∧ chordAB.p1 ≠ chordCD.p2 ∧
    chordAB.p2 ≠ chordCD.p1 ∧ chordAB.p2 ≠ chordCD.p2 ∧
    chordAB.p1 ≠ chordEF.p1 ∧ chordAB.p1 ≠ chordEF.p2 ∧
    chordAB.p2 ≠ chordEF.p1 ∧ chordAB.p2 ≠ chordEF.p2 ∧
    chordCD.p1 ≠ chordEF.p1 ∧ chordCD.p1 ≠ chordEF.p2 ∧
    chordCD.p2 ≠ chordEF.p1 ∧ chordCD.p2 ≠ chordEF.p2

-- Define the intersection of diagonals
def diagonalIntersection (q1 q2 q3 q4 : Point) : Point :=
  sorry -- Actual implementation would calculate the intersection

-- Define collinearity
def collinear (p1 p2 p3 : Point) : Prop :=
  sorry -- Actual implementation would define collinearity

-- Theorem statement
theorem chord_diagonal_intersections_collinear (config : ChordConfiguration) :
  let M := diagonalIntersection config.chordAB.p1 config.chordAB.p2 config.chordEF.p1 config.chordEF.p2
  let N := diagonalIntersection config.chordCD.p1 config.chordCD.p2 config.chordEF.p1 config.chordEF.p2
  let P := diagonalIntersection config.chordAB.p1 config.chordAB.p2 config.chordCD.p1 config.chordCD.p2
  collinear M N P :=
by
  sorry

end CircleChords

end NUMINAMATH_CALUDE_chord_diagonal_intersections_collinear_l2238_223829


namespace NUMINAMATH_CALUDE_equation_solution_l2238_223877

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (18 + 6*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3*Real.sqrt 3 ∧ 
  x = 31 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2238_223877


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2238_223873

/-- A right-angled triangle with special properties -/
structure SpecialTriangle where
  -- The hypotenuse of the triangle
  hypotenuse : ℝ
  -- The shorter leg of the triangle
  short_leg : ℝ
  -- The longer leg of the triangle
  long_leg : ℝ
  -- The hypotenuse is 1
  hyp_is_one : hypotenuse = 1
  -- The shorter leg is (√5 - 1) / 2
  short_leg_value : short_leg = (Real.sqrt 5 - 1) / 2
  -- The longer leg is the square root of the shorter leg
  long_leg_value : long_leg = Real.sqrt short_leg

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  -- The longer leg is the geometric mean of the hypotenuse and shorter leg
  t.long_leg ^ 2 = t.hypotenuse * t.short_leg ∧
  -- All segments formed by successive altitudes are powers of the longer leg
  ∀ n : ℕ, ∃ segment : ℝ, segment = t.long_leg ^ n ∧ 0 ≤ n ∧ n ≤ 9 :=
by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2238_223873


namespace NUMINAMATH_CALUDE_least_number_divisibility_l2238_223833

theorem least_number_divisibility (n : ℕ) (h1 : (n + 6) % 24 = 0) (h2 : (n + 6) % 32 = 0)
  (h3 : (n + 6) % 36 = 0) (h4 : n + 6 = 858) :
  ∃ p : ℕ, Nat.Prime p ∧ p ≠ 2 ∧ p ≠ 3 ∧ n % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l2238_223833


namespace NUMINAMATH_CALUDE_diagonal_sum_equals_fibonacci_l2238_223859

/-- The sum of binomial coefficients in a diagonal of Pascal's Triangle -/
def diagonalSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose (n - k) k)

/-- The nth Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The main theorem: The diagonal sum equals the (n+1)th Fibonacci number -/
theorem diagonal_sum_equals_fibonacci (n : ℕ) : diagonalSum n = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_sum_equals_fibonacci_l2238_223859


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2238_223853

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2238_223853


namespace NUMINAMATH_CALUDE_unique_solution_condition_l2238_223824

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^4 - a*x^3 - 3*a*x^2 + 2*a^2*x + a^2 - 2 = 0) ↔ 
  a < (3/4)^2 + 3/4 - 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l2238_223824


namespace NUMINAMATH_CALUDE_angle_measure_in_pentagon_and_triangle_l2238_223882

/-- Given a pentagon with angles A, B, C, E, and F, where angles D, E, and F form a triangle,
    this theorem proves that if m∠A = 80°, m∠B = 30°, and m∠C = 20°, then m∠D = 130°. -/
theorem angle_measure_in_pentagon_and_triangle 
  (A B C D E F : Real) 
  (pentagon : A + B + C + E + F = 540) 
  (triangle : D + E + F = 180) 
  (angle_A : A = 80) 
  (angle_B : B = 30) 
  (angle_C : C = 20) : 
  D = 130 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_pentagon_and_triangle_l2238_223882


namespace NUMINAMATH_CALUDE_circle_angle_sum_l2238_223891

/-- Given a circle divided into 12 equal arcs, this theorem proves that the sum of
    half the central angle spanning 2 arcs and half the central angle spanning 4 arcs
    is equal to 90 degrees. -/
theorem circle_angle_sum (α β : Real) : 
  (∀ (n : Nat), n ≤ 12 → 360 / 12 * n = 30 * n) →
  α = (2 * 360 / 12) / 2 →
  β = (4 * 360 / 12) / 2 →
  α + β = 90 := by
sorry

end NUMINAMATH_CALUDE_circle_angle_sum_l2238_223891


namespace NUMINAMATH_CALUDE_max_rectangle_area_l2238_223867

/-- The maximum area of a rectangle given constraints --/
theorem max_rectangle_area (perimeter : ℝ) (min_length min_width : ℝ) :
  perimeter = 400 ∧ min_length = 100 ∧ min_width = 50 →
  ∃ (length width : ℝ),
    length ≥ min_length ∧
    width ≥ min_width ∧
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℝ),
      l ≥ min_length →
      w ≥ min_width →
      2 * (l + w) = perimeter →
      l * w ≤ length * width ∧
      length * width = 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l2238_223867


namespace NUMINAMATH_CALUDE_equation_solution_l2238_223895

theorem equation_solution (a : ℚ) : 
  (∀ x, a * x - 4 * (x - a) = 1) → (a * 2 - 4 * (2 - a) = 1) → a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2238_223895


namespace NUMINAMATH_CALUDE_max_square_sum_l2238_223830

def triangle_numbers : Finset ℕ := {5, 6, 7, 8, 9}

def circle_product (a b c : ℕ) : ℕ := a * b * c

def square_sum (f g h : ℕ) : ℕ := f + g + h

theorem max_square_sum :
  ∃ (a b c d e : ℕ),
    a ∈ triangle_numbers ∧
    b ∈ triangle_numbers ∧
    c ∈ triangle_numbers ∧
    d ∈ triangle_numbers ∧
    e ∈ triangle_numbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    square_sum (circle_product a b c) (circle_product b c d) (circle_product c d e) = 1251 ∧
    ∀ (x y z w v : ℕ),
      x ∈ triangle_numbers →
      y ∈ triangle_numbers →
      z ∈ triangle_numbers →
      w ∈ triangle_numbers →
      v ∈ triangle_numbers →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ v ∧
      w ≠ v →
      square_sum (circle_product x y z) (circle_product y z w) (circle_product z w v) ≤ 1251 :=
sorry

end NUMINAMATH_CALUDE_max_square_sum_l2238_223830


namespace NUMINAMATH_CALUDE_assignment_schemes_l2238_223803

def number_of_roles : ℕ := 5
def number_of_members : ℕ := 5

def roles_for_A : ℕ := number_of_roles - 2
def roles_for_B : ℕ := 1
def remaining_members : ℕ := number_of_members - 2
def remaining_roles : ℕ := number_of_roles - 2

theorem assignment_schemes :
  (roles_for_B) * (roles_for_A) * (remaining_members.factorial) = 18 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_l2238_223803


namespace NUMINAMATH_CALUDE_no_universal_triangle_relation_l2238_223813

/-- A triangle with perimeter, circumradius, and inradius -/
structure Triangle where
  perimeter : ℝ
  circumradius : ℝ
  inradius : ℝ

/-- There is no universal relationship among perimeter, circumradius, and inradius for all triangles -/
theorem no_universal_triangle_relation :
  ¬(∀ t : Triangle,
    (t.perimeter > t.circumradius + t.inradius) ∨
    (t.perimeter ≤ t.circumradius + t.inradius) ∨
    (1/6 < t.circumradius + t.inradius ∧ t.circumradius + t.inradius < 6*t.perimeter)) :=
by sorry

end NUMINAMATH_CALUDE_no_universal_triangle_relation_l2238_223813


namespace NUMINAMATH_CALUDE_least_sum_p_q_l2238_223800

theorem least_sum_p_q (p q : ℕ) (hp : p > 1) (hq : q > 1) 
  (h_eq : 17 * (p + 1) = 25 * (q + 1)) : 
  (∀ p' q' : ℕ, p' > 1 → q' > 1 → 17 * (p' + 1) = 25 * (q' + 1) → p' + q' ≥ p + q) → 
  p + q = 168 := by
sorry

end NUMINAMATH_CALUDE_least_sum_p_q_l2238_223800


namespace NUMINAMATH_CALUDE_books_left_to_read_l2238_223845

def total_books : ℕ := 89
def mcgregor_finished : ℕ := 34
def floyd_finished : ℕ := 32

theorem books_left_to_read :
  total_books - (mcgregor_finished + floyd_finished) = 23 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l2238_223845


namespace NUMINAMATH_CALUDE_rectangular_solid_length_l2238_223857

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (rs : RectangularSolid) : ℝ :=
  2 * (rs.length * rs.width + rs.length * rs.depth + rs.width * rs.depth)

/-- Theorem: The length of a rectangular solid with width 4, depth 1, and surface area 58 is 5 -/
theorem rectangular_solid_length :
  ∃ (rs : RectangularSolid),
    rs.width = 4 ∧
    rs.depth = 1 ∧
    surfaceArea rs = 58 ∧
    rs.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_length_l2238_223857


namespace NUMINAMATH_CALUDE_new_person_weight_l2238_223801

theorem new_person_weight
  (n : ℕ)
  (initial_average : ℝ)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : n = 8)
  (h2 : weight_increase = 4)
  (h3 : replaced_weight = 55)
  : ∃ (new_weight : ℝ),
    n * (initial_average + weight_increase) = (n - 1) * initial_average + new_weight ∧
    new_weight = 87
  := by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2238_223801


namespace NUMINAMATH_CALUDE_street_length_proof_l2238_223889

/-- Proves that the length of a street is 1440 meters, given that a person crosses it in 12 minutes at a speed of 7.2 km per hour. -/
theorem street_length_proof (time : ℝ) (speed : ℝ) (length : ℝ) : 
  time = 12 →
  speed = 7.2 →
  length = speed * 1000 / 60 * time →
  length = 1440 := by
sorry

end NUMINAMATH_CALUDE_street_length_proof_l2238_223889


namespace NUMINAMATH_CALUDE_total_apples_l2238_223819

/-- Proves that the total number of apples given out is 150, given that Harold gave 25 apples to each of 6 people. -/
theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_l2238_223819


namespace NUMINAMATH_CALUDE_average_equals_x_l2238_223849

theorem average_equals_x (x : ℝ) : 
  (2 + 5 + x + 14 + 15) / 5 = x → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_equals_x_l2238_223849


namespace NUMINAMATH_CALUDE_initial_walking_time_l2238_223898

/-- Proves that given a person walking at 5 kilometers per hour, if they need 3 more hours to reach a total of 30 kilometers, then they have already walked for 3 hours. -/
theorem initial_walking_time (speed : ℝ) (additional_hours : ℝ) (total_distance : ℝ) 
  (h1 : speed = 5)
  (h2 : additional_hours = 3)
  (h3 : total_distance = 30) :
  (total_distance - additional_hours * speed) / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_walking_time_l2238_223898


namespace NUMINAMATH_CALUDE_parabola_directrix_l2238_223836

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 2*x) → (∃ (p : ℝ), p = 1/2 ∧ x = -p) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2238_223836


namespace NUMINAMATH_CALUDE_equality_of_negative_powers_l2238_223852

theorem equality_of_negative_powers : -(-1)^99 = (-1)^100 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_negative_powers_l2238_223852


namespace NUMINAMATH_CALUDE_running_increase_per_week_l2238_223854

theorem running_increase_per_week 
  (initial_capacity : ℝ) 
  (increase_percentage : ℝ) 
  (days : ℕ) 
  (h1 : initial_capacity = 100)
  (h2 : increase_percentage = 0.2)
  (h3 : days = 280) :
  let new_capacity := initial_capacity * (1 + increase_percentage)
  let weeks := days / 7
  (new_capacity - initial_capacity) / weeks = 3 := by sorry

end NUMINAMATH_CALUDE_running_increase_per_week_l2238_223854


namespace NUMINAMATH_CALUDE_candidate_a_votes_l2238_223899

/-- Proves that given a ratio of 2:1 for votes between two candidates and a total of 21 votes,
    the candidate with the higher number of votes received 14 votes. -/
theorem candidate_a_votes (total_votes : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) : 
  total_votes = 21 → ratio_a = 2 → ratio_b = 1 → 
  (ratio_a * total_votes) / (ratio_a + ratio_b) = 14 := by
sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l2238_223899


namespace NUMINAMATH_CALUDE_counterexample_exists_l2238_223868

theorem counterexample_exists : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a * b ≤ c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2238_223868


namespace NUMINAMATH_CALUDE_part_one_part_two_l2238_223809

-- Define the function f(x) = |x-a| + 3x
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + 3 * x

theorem part_one :
  let f₁ := f 1
  (∀ x, f₁ x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
sorry

theorem part_two (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≤ 0 ↔ x ≤ -1) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2238_223809


namespace NUMINAMATH_CALUDE_maintenance_check_time_l2238_223848

/-- The initial time between maintenance checks before using the additive -/
def initial_time : ℝ := 20

/-- The new time between maintenance checks after using the additive -/
def new_time : ℝ := 25

/-- The percentage increase in time between maintenance checks -/
def percentage_increase : ℝ := 0.25

theorem maintenance_check_time : 
  initial_time * (1 + percentage_increase) = new_time :=
by sorry

end NUMINAMATH_CALUDE_maintenance_check_time_l2238_223848


namespace NUMINAMATH_CALUDE_percentage_of_330_l2238_223874

theorem percentage_of_330 : (33 + 1/3 : ℚ) / 100 * 330 = 110 := by sorry

end NUMINAMATH_CALUDE_percentage_of_330_l2238_223874


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l2238_223838

/-- Calculates the correct average marks after correcting an error in one student's mark -/
theorem correct_average_after_error_correction 
  (num_students : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark : ℚ) 
  (correct_mark : ℚ) : 
  num_students = 10 → 
  initial_average = 100 → 
  wrong_mark = 60 → 
  correct_mark = 10 → 
  (initial_average * num_students - wrong_mark + correct_mark) / num_students = 95 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l2238_223838


namespace NUMINAMATH_CALUDE_series_sum_ln2_series_sum_1_minus_ln2_l2238_223825

/-- The sum of the series where the nth term is 1/((2n-1)(2n)) converges to ln 2 -/
theorem series_sum_ln2 : ∑' n, 1 / ((2 * n - 1) * (2 * n)) = Real.log 2 := by sorry

/-- The sum of the series where the nth term is 1/((2n)(2n+1)) converges to 1 - ln 2 -/
theorem series_sum_1_minus_ln2 : ∑' n, 1 / ((2 * n) * (2 * n + 1)) = 1 - Real.log 2 := by sorry

end NUMINAMATH_CALUDE_series_sum_ln2_series_sum_1_minus_ln2_l2238_223825


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l2238_223846

/-- Given a function f(x) = ax³ + bx + 1, prove that if f(a) = 8, then f(-a) = -6 -/
theorem function_value_at_negative_a (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + 1
  f a = 8 → f (-a) = -6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l2238_223846


namespace NUMINAMATH_CALUDE_quadratic_sum_constrained_l2238_223832

theorem quadratic_sum_constrained (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10) 
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) : 
  3 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_constrained_l2238_223832


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l2238_223875

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 18) :
  let r := 9 - 9 * Real.sqrt 2 / 2
  (4 / 3 : ℝ) * Real.pi * r^3 = (4 / 3 : ℝ) * Real.pi * (9 - 9 * Real.sqrt 2 / 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l2238_223875


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2238_223808

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let original_area := L * W
  let new_length := L * 1.2
  let new_width := W * 1.2
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 44 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2238_223808


namespace NUMINAMATH_CALUDE_simplify_expression_l2238_223862

theorem simplify_expression (x : ℝ) : (x + 1)^2 + x*(x - 2) = 2*x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2238_223862


namespace NUMINAMATH_CALUDE_like_terms_exponent_sum_l2238_223883

/-- Given two terms 3x^m*y and -5x^2*y^n that are like terms, prove that m + n = 3 -/
theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^m * y = -5 * x^2 * y^n) → m + n = 3 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponent_sum_l2238_223883


namespace NUMINAMATH_CALUDE_root_implies_m_value_l2238_223855

theorem root_implies_m_value (m : ℝ) : 
  (1 : ℝ)^2 + m * (1 : ℝ) - 3 = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l2238_223855


namespace NUMINAMATH_CALUDE_point_outside_circle_l2238_223822

theorem point_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  a^2 + b^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2238_223822


namespace NUMINAMATH_CALUDE_shooting_test_probability_l2238_223805

/-- Represents the probability of hitting a single shot -/
def hit_prob : ℝ := 0.6

/-- Represents the probability of missing a single shot -/
def miss_prob : ℝ := 1 - hit_prob

/-- Calculates the probability of passing the shooting test -/
def pass_prob : ℝ := 
  hit_prob^3 + hit_prob^2 * miss_prob + miss_prob * hit_prob^2

theorem shooting_test_probability : pass_prob = 0.504 := by
  sorry

end NUMINAMATH_CALUDE_shooting_test_probability_l2238_223805


namespace NUMINAMATH_CALUDE_log_division_simplification_l2238_223865

theorem log_division_simplification :
  (Real.log 256 / Real.log 16) / (Real.log (1/256) / Real.log 16) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_division_simplification_l2238_223865


namespace NUMINAMATH_CALUDE_sufficient_condition_l2238_223834

theorem sufficient_condition (a : ℝ) : a > 0 → a^2 + a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_l2238_223834


namespace NUMINAMATH_CALUDE_square_difference_l2238_223842

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2238_223842


namespace NUMINAMATH_CALUDE_floor_product_eq_twenty_l2238_223806

theorem floor_product_eq_twenty (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < (21 / 4) :=
sorry

end NUMINAMATH_CALUDE_floor_product_eq_twenty_l2238_223806


namespace NUMINAMATH_CALUDE_amount_to_find_l2238_223861

def water_bottles : ℕ := 5 * 12
def energy_bars : ℕ := 4 * 12
def original_water_price : ℚ := 2
def original_energy_price : ℚ := 3
def market_water_price : ℚ := 185/100
def market_energy_price : ℚ := 275/100
def discount_rate : ℚ := 1/10

def original_total : ℚ := water_bottles * original_water_price + energy_bars * original_energy_price

def discounted_water_price : ℚ := market_water_price * (1 - discount_rate)
def discounted_energy_price : ℚ := market_energy_price * (1 - discount_rate)

def discounted_total : ℚ := water_bottles * discounted_water_price + energy_bars * discounted_energy_price

theorem amount_to_find : original_total - discounted_total = 453/10 := by sorry

end NUMINAMATH_CALUDE_amount_to_find_l2238_223861


namespace NUMINAMATH_CALUDE_function_behavior_l2238_223888

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) :
  is_even_function f →
  has_period f 2 →
  is_decreasing_on f (-1) 0 →
  (is_increasing_on f 6 7 ∧ is_decreasing_on f 7 8) :=
by sorry

end NUMINAMATH_CALUDE_function_behavior_l2238_223888


namespace NUMINAMATH_CALUDE_double_price_profit_l2238_223864

theorem double_price_profit (cost_price : ℝ) (initial_selling_price : ℝ) :
  initial_selling_price = cost_price * 1.5 →
  let double_price := 2 * initial_selling_price
  (double_price - cost_price) / cost_price = 2 := by
  sorry

end NUMINAMATH_CALUDE_double_price_profit_l2238_223864


namespace NUMINAMATH_CALUDE_pond_water_theorem_l2238_223871

/-- Calculates the amount of water remaining in a pond after a certain number of days,
    given initial water amount, evaporation rate, and rain addition rate. -/
def water_remaining (initial_water : ℝ) (evaporation_rate : ℝ) (rain_rate : ℝ) (days : ℕ) : ℝ :=
  initial_water - (evaporation_rate - rain_rate) * days

theorem pond_water_theorem (initial_water : ℝ) (evaporation_rate : ℝ) (rain_rate : ℝ) (days : ℕ) :
  initial_water = 500 ∧ evaporation_rate = 4 ∧ rain_rate = 2 ∧ days = 40 →
  water_remaining initial_water evaporation_rate rain_rate days = 420 := by
  sorry

#eval water_remaining 500 4 2 40

end NUMINAMATH_CALUDE_pond_water_theorem_l2238_223871


namespace NUMINAMATH_CALUDE_minimum_fraction_ponies_with_horseshoes_l2238_223860

theorem minimum_fraction_ponies_with_horseshoes :
  ∀ (num_ponies num_horses num_ponies_with_horseshoes num_icelandic_ponies_with_horseshoes : ℕ),
  num_horses = num_ponies + 4 →
  num_horses + num_ponies ≥ 164 →
  8 * num_icelandic_ponies_with_horseshoes = 5 * num_ponies_with_horseshoes →
  num_ponies_with_horseshoes ≤ num_ponies →
  (∃ (min_fraction : ℚ), 
    min_fraction = num_ponies_with_horseshoes / num_ponies ∧
    min_fraction = 1 / 10) :=
by sorry

end NUMINAMATH_CALUDE_minimum_fraction_ponies_with_horseshoes_l2238_223860


namespace NUMINAMATH_CALUDE_det_A_eq_90_l2238_223863

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, 0, 2],
    ![8, 5, -2],
    ![3, 3, 6]]

theorem det_A_eq_90 : Matrix.det A = 90 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_90_l2238_223863


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_open_interval_l2238_223810

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then (3 - a) * x - 4 * a else Real.log x / Real.log a

/-- Theorem stating the range of a for which f is increasing on ℝ -/
theorem f_increasing_iff_a_in_open_interval :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_open_interval_l2238_223810


namespace NUMINAMATH_CALUDE_triangle_angle_sum_identity_l2238_223872

theorem triangle_angle_sum_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 
  -4 * Real.cos (3/2 * A) * Real.cos (3/2 * B) * Real.cos (3/2 * C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_identity_l2238_223872


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l2238_223876

theorem x_range_for_inequality (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) → 
  x > 3 ∨ x < -1 := by
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l2238_223876


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l2238_223831

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l2238_223831


namespace NUMINAMATH_CALUDE_apple_picking_theorem_l2238_223840

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- Tom picked twice as many apples as Lexie -/
def tom_apples : ℕ := 2 * lexie_apples

/-- The total number of apples collected -/
def total_apples : ℕ := lexie_apples + tom_apples

theorem apple_picking_theorem : total_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_theorem_l2238_223840


namespace NUMINAMATH_CALUDE_alice_paid_24_percent_l2238_223892

-- Define the suggested retail price
def suggested_retail_price : ℝ := 100

-- Define the marked price as 60% of the suggested retail price
def marked_price : ℝ := 0.6 * suggested_retail_price

-- Define Alice's purchase price as 40% of the marked price
def alice_price : ℝ := 0.4 * marked_price

-- Theorem to prove
theorem alice_paid_24_percent :
  alice_price / suggested_retail_price = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_24_percent_l2238_223892


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l2238_223879

/-- An isosceles triangle with altitude 8 and perimeter 32 has area 48 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 → -- b and s are positive real numbers
  2 * s + 2 * b = 32 → -- perimeter condition
  b ^ 2 + 8 ^ 2 = s ^ 2 → -- Pythagorean theorem for half the triangle
  (2 * b) * 8 / 2 = 48 := by 
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_area_l2238_223879


namespace NUMINAMATH_CALUDE_sector_radius_l2238_223841

theorem sector_radius (area : Real) (angle : Real) (π : Real) (h1 : area = 36.67) (h2 : angle = 42) (h3 : π = 3.14159) :
  ∃ r : Real, r = 10 ∧ area = (angle / 360) * π * r^2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l2238_223841


namespace NUMINAMATH_CALUDE_final_mixture_is_all_x_l2238_223847

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
structure FinalMixture where
  x : ℝ
  y : ℝ

/-- Seed mixture X -/
def X : SeedMixture :=
  { ryegrass := 1 - 0.6
    bluegrass := 0.6
    fescue := 0 }

/-- Seed mixture Y -/
def Y : SeedMixture :=
  { ryegrass := 0.25
    bluegrass := 0
    fescue := 0.75 }

/-- Theorem stating that the percentage of seed mixture X in the final mixture is 100% -/
theorem final_mixture_is_all_x (m : FinalMixture) :
  X.ryegrass * m.x + Y.ryegrass * m.y = 0.4 * (m.x + m.y) →
  m.x + m.y = 1 →
  m.x = 1 := by
  sorry


end NUMINAMATH_CALUDE_final_mixture_is_all_x_l2238_223847


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2238_223866

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + x ≥ 0) ↔ 
  (∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2238_223866


namespace NUMINAMATH_CALUDE_sum_of_squares_l2238_223884

theorem sum_of_squares (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 10)
  (h2 : (a * b * c)^(1/3 : ℝ) = 6)
  (h3 : 3 / (1/a + 1/b + 1/c) = 4) :
  a^2 + b^2 + c^2 = 576 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2238_223884
