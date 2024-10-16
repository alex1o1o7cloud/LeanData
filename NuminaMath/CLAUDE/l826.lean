import Mathlib

namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l826_82654

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let is_odd (n : ℕ) := n % 2 = 1
  let total_products := table_size * table_size
  let odd_products := (table_size / 2) * (table_size / 2)
  odd_products / total_products = (1 : ℚ) / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l826_82654


namespace NUMINAMATH_CALUDE_ten_row_triangle_pieces_l826_82627

/-- The number of rods in the nth row of the triangle -/
def rods_in_row (n : ℕ) : ℕ := 3 * n

/-- The total number of rods in a triangle with n rows -/
def total_rods (n : ℕ) : ℕ := (n * (n + 1) * 3) / 2

/-- The number of connectors in a triangle with n rows of rods -/
def total_connectors (n : ℕ) : ℕ := ((n + 1) * (n + 2)) / 2

/-- The total number of pieces in a triangle with n rows of rods -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

theorem ten_row_triangle_pieces :
  total_pieces 10 = 231 := by sorry

end NUMINAMATH_CALUDE_ten_row_triangle_pieces_l826_82627


namespace NUMINAMATH_CALUDE_f_range_theorem_l826_82614

open Real

noncomputable def f (k x : ℝ) : ℝ := (k * x + 4) * log x - x

def has_unique_integer_root (k : ℝ) : Prop :=
  ∃ s t : ℝ, s < 2 ∧ 2 < t ∧ 
    (∀ x, 1 < x → (s < x ∧ x < t ↔ 0 < f k x)) ∧
    (∀ n : ℤ, (s < ↑n ∧ ↑n < t) → n = 2)

theorem f_range_theorem :
  ∀ k : ℝ, has_unique_integer_root k ↔ 
    (1 / log 2 - 2 < k ∧ k ≤ 1 / log 3 - 4 / 3) :=
sorry

end NUMINAMATH_CALUDE_f_range_theorem_l826_82614


namespace NUMINAMATH_CALUDE_photo_calculation_l826_82648

theorem photo_calculation (total_photos : ℕ) (claire_photos : ℕ) : 
  (claire_photos : ℚ) + 3 * claire_photos + 5/4 * claire_photos + 
  5/2 * 5/4 * claire_photos + (claire_photos + 3 * claire_photos) / 2 + 
  (claire_photos + 3 * claire_photos) / 4 = total_photos ∧ total_photos = 840 → 
  claire_photos = 74 := by
sorry

end NUMINAMATH_CALUDE_photo_calculation_l826_82648


namespace NUMINAMATH_CALUDE_circle_trajectory_and_intersection_line_l826_82678

-- Define the circles E and F
def circle_E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def circle_F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 - y^2 = 1

-- Define the curve C (trajectory of center of circle P)
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 4*y - 5 = 0

-- Define the point M
def point_M : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem circle_trajectory_and_intersection_line :
  ∀ (x_A y_A x_B y_B : ℝ),
  -- Circle P is internally tangent to both E and F
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x_P y_P : ℝ), (x_P - x_A)^2 + (y_P - y_A)^2 = r^2 →
      (∃ (x_E y_E : ℝ), circle_E x_E y_E ∧ (x_P - x_E)^2 + (y_P - y_E)^2 = (5 - r)^2) ∧
      (∃ (x_F y_F : ℝ), circle_F x_F y_F ∧ (x_P - x_F)^2 + (y_P - y_F)^2 = (r - 1)^2))) →
  -- A and B are on curve C
  curve_C x_A y_A →
  curve_C x_B y_B →
  -- M is the midpoint of AB
  point_M = ((x_A + x_B)/2, (y_A + y_B)/2) →
  -- A, B are on line l
  line_l x_A y_A →
  line_l x_B y_B →
  -- The equation of curve C is correct
  (∀ (x y : ℝ), curve_C x y ↔ x^2/4 + y^2 = 1) ∧
  -- The equation of line l is correct
  (∀ (x y : ℝ), line_l x y ↔ x + 4*y - 5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_circle_trajectory_and_intersection_line_l826_82678


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l826_82600

/-- Represents a number in a given base --/
structure BaseNumber where
  digits : List Nat
  base : Nat

/-- Converts a base 2 number to base 10 --/
def binaryToDecimal (bn : BaseNumber) : Nat :=
  bn.digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a base 10 number to base 4 --/
def decimalToQuaternary (n : Nat) : BaseNumber :=
  let rec toDigits (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else toDigits (m / 4) ((m % 4) :: acc)
  { digits := toDigits n [], base := 4 }

/-- The main theorem --/
theorem binary_to_quaternary_conversion :
  let binary := BaseNumber.mk [1,0,1,1,0,0,1,0,1] 2
  let quaternary := BaseNumber.mk [2,3,0,1,1] 4
  decimalToQuaternary (binaryToDecimal binary) = quaternary := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l826_82600


namespace NUMINAMATH_CALUDE_trajectory_of_right_angle_tangents_l826_82610

/-- The trajectory of a point P outside a unit circle, from which two tangent lines to the circle form a right angle. -/
theorem trajectory_of_right_angle_tangents (x y : ℝ) : 
  (x^2 + y^2 > 1) →  -- P is outside the unit circle
  (∃ (m n : ℝ × ℝ), 
    (m.1^2 + m.2^2 = 1) ∧  -- M is on the unit circle
    (n.1^2 + n.2^2 = 1) ∧  -- N is on the unit circle
    ((x - m.1)^2 + (y - m.2)^2) * (m.1^2 + m.2^2) = ((x - m.1) * m.1 + (y - m.2) * m.2)^2 ∧  -- PM is tangent
    ((x - n.1)^2 + (y - n.2)^2) * (n.1^2 + n.2^2) = ((x - n.1) * n.1 + (y - n.2) * n.2)^2 ∧  -- PN is tangent
    ((x - m.1) * (x - n.1) + (y - m.2) * (y - n.2))^2 = 
      ((x - m.1)^2 + (y - m.2)^2) * ((x - n.1)^2 + (y - n.2)^2)) →  -- ∠MPN = 90°
  x^2 + y^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_right_angle_tangents_l826_82610


namespace NUMINAMATH_CALUDE_solve_equation_l826_82656

theorem solve_equation : ∃ a : ℝ, -2 - a = 0 ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l826_82656


namespace NUMINAMATH_CALUDE_cow_distribution_theorem_l826_82664

/-- Represents the distribution of cows among four sons -/
structure CowDistribution where
  total : ℕ
  first_son : ℚ
  second_son : ℚ
  third_son : ℚ
  fourth_son : ℕ

/-- Theorem stating the total number of cows given the distribution -/
theorem cow_distribution_theorem (d : CowDistribution) :
  d.first_son = 1/3 ∧ 
  d.second_son = 1/5 ∧ 
  d.third_son = 1/6 ∧ 
  d.fourth_son = 12 ∧
  d.first_son + d.second_son + d.third_son + (d.fourth_son : ℚ) / d.total = 1 →
  d.total = 40 := by
  sorry

end NUMINAMATH_CALUDE_cow_distribution_theorem_l826_82664


namespace NUMINAMATH_CALUDE_max_angle_between_tangents_l826_82680

/-- The parabola C₁ defined by y² = 4x -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The circle C₂ defined by (x-3)² + y² = 2 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

/-- The angle between two tangents drawn from a point to a circle -/
def angleBetweenTangents (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The maximum angle between tangents theorem -/
theorem max_angle_between_tangents :
  ∃ (θ : ℝ), θ = 60 * π / 180 ∧
  ∀ (p : ℝ × ℝ), p ∈ C₁ →
    angleBetweenTangents p C₂ ≤ θ ∧
    ∃ (q : ℝ × ℝ), q ∈ C₁ ∧ angleBetweenTangents q C₂ = θ :=
  sorry

end NUMINAMATH_CALUDE_max_angle_between_tangents_l826_82680


namespace NUMINAMATH_CALUDE_bacon_count_l826_82671

/-- The number of students who suggested adding mashed potatoes -/
def mashed_potatoes : ℕ := 330

/-- The difference between the number of students who suggested mashed potatoes and bacon -/
def difference : ℕ := 61

/-- The number of students who suggested adding bacon -/
def bacon : ℕ := mashed_potatoes - difference

theorem bacon_count : bacon = 269 := by
  sorry

end NUMINAMATH_CALUDE_bacon_count_l826_82671


namespace NUMINAMATH_CALUDE_condition_relationship_l826_82635

theorem condition_relationship (x : ℝ) : 
  (∀ x, abs x < 1 → x > -1) ∧ 
  (∃ x, x > -1 ∧ ¬(abs x < 1)) :=
sorry

end NUMINAMATH_CALUDE_condition_relationship_l826_82635


namespace NUMINAMATH_CALUDE_range_of_a_when_f_has_four_zeros_l826_82608

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.exp x + a * x^2 else Real.exp (-x) + a * x^2

/-- Theorem stating the range of a when f has four zeros -/
theorem range_of_a_when_f_has_four_zeros :
  ∀ a : ℝ, (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧ f a x₄ = 0) →
  a < -Real.exp 2 / 4 ∧ ∀ y : ℝ, y < -Real.exp 2 / 4 → ∃ x : ℝ, f x y = 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_f_has_four_zeros_l826_82608


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l826_82606

theorem negative_fraction_comparison : -4/5 > -5/6 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l826_82606


namespace NUMINAMATH_CALUDE_oranges_remaining_l826_82686

theorem oranges_remaining (initial_oranges removed_oranges : ℕ) 
  (h1 : initial_oranges = 96)
  (h2 : removed_oranges = 45) :
  initial_oranges - removed_oranges = 51 := by
sorry

end NUMINAMATH_CALUDE_oranges_remaining_l826_82686


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l826_82601

/-- The sum of an arithmetic sequence with first term 3, last term 153, and common difference 5,
    when divided by 24, has a remainder of 0. -/
theorem arithmetic_sequence_sum_remainder (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 3 → aₙ = 153 → d = 5 → aₙ = a₁ + (n - 1) * d →
  (n * (a₁ + aₙ) / 2) % 24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_remainder_l826_82601


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l826_82613

/-- Given two vectors a and b in ℝ², where a is perpendicular to (a + b), prove that the y-coordinate of b is -3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a = (2, 1)) (h' : b.1 = -1) 
  (h'' : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) : ℝ) = 0) : 
  b.2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l826_82613


namespace NUMINAMATH_CALUDE_smallest_repunit_divisible_by_97_l826_82609

theorem smallest_repunit_divisible_by_97 : 
  (∀ k < 96, ∃ r, (10^k - 1) / 9 = 97 * r + 1) ∧ 
  ∃ q, (10^96 - 1) / 9 = 97 * q :=
sorry

end NUMINAMATH_CALUDE_smallest_repunit_divisible_by_97_l826_82609


namespace NUMINAMATH_CALUDE_second_largest_power_of_ten_in_170_factorial_l826_82605

theorem second_largest_power_of_ten_in_170_factorial : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ n → (170 : ℕ).factorial % (10 ^ k) = 0) ∧ 
  (170 : ℕ).factorial % (10 ^ (n + 1)) ≠ 0 ∧ 
  n = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_largest_power_of_ten_in_170_factorial_l826_82605


namespace NUMINAMATH_CALUDE_carly_grape_lollipops_l826_82603

/-- The number of grape lollipops in Carly's collection --/
def grape_lollipops (total : ℕ) (cherry : ℕ) (non_cherry_flavors : ℕ) : ℕ :=
  (total - cherry) / non_cherry_flavors

/-- Theorem stating the number of grape lollipops in Carly's collection --/
theorem carly_grape_lollipops : 
  grape_lollipops 42 (42 / 2) 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_carly_grape_lollipops_l826_82603


namespace NUMINAMATH_CALUDE_lemonade_problem_l826_82658

theorem lemonade_problem (lemons_for_60 : ℕ) (gallons : ℕ) (lemon_cost : ℚ) :
  lemons_for_60 = 36 →
  gallons = 15 →
  lemon_cost = 1/2 →
  (lemons_for_60 * gallons) / 60 = 9 ∧
  (lemons_for_60 * gallons) / 60 * lemon_cost = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_problem_l826_82658


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l826_82694

theorem root_equation_implies_expression_value (x₀ : ℝ) (h : x₀ > 0) :
  x₀^3 * Real.exp (x₀ - 4) + 2 * Real.log x₀ - 4 = 0 →
  Real.exp ((4 - x₀) / 2) + 2 * Real.log x₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l826_82694


namespace NUMINAMATH_CALUDE_line_direction_vector_l826_82652

/-- Given a line passing through two points and a direction vector, prove the scalar value. -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (a : ℝ) :
  p1 = (-3, 2) →
  p2 = (2, -3) →
  (a, -2) = (p2.1 - p1.1, p2.2 - p1.2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l826_82652


namespace NUMINAMATH_CALUDE_local_tax_deduction_l826_82642

-- Define the hourly wage in dollars
def hourly_wage : ℝ := 25

-- Define the local tax rate as a percentage
def tax_rate : ℝ := 2

-- Define the conversion rate from dollars to cents
def dollars_to_cents : ℝ := 100

-- Theorem statement
theorem local_tax_deduction :
  (hourly_wage * dollars_to_cents) * (tax_rate / 100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_local_tax_deduction_l826_82642


namespace NUMINAMATH_CALUDE_meeting_at_64th_lamp_l826_82623

def meet_point (total_intervals : ℕ) (petya_progress : ℕ) (vasya_progress : ℕ) : ℕ :=
  3 * petya_progress + 1

theorem meeting_at_64th_lamp (total_lamps : ℕ) (petya_at : ℕ) (vasya_at : ℕ) 
  (h1 : total_lamps = 100)
  (h2 : petya_at = 22)
  (h3 : vasya_at = 88) :
  meet_point (total_lamps - 1) (petya_at - 1) (total_lamps - vasya_at) = 64 := by
  sorry

end NUMINAMATH_CALUDE_meeting_at_64th_lamp_l826_82623


namespace NUMINAMATH_CALUDE_weight_difference_e_d_l826_82638

/-- Given the weights of individuals A, B, C, D, and E, prove that E weighs 3 kg more than D. -/
theorem weight_difference_e_d (w_a w_b w_c w_d w_e : ℝ) : 
  w_a = 81 →
  (w_a + w_b + w_c) / 3 = 70 →
  (w_a + w_b + w_c + w_d) / 4 = 70 →
  (w_b + w_c + w_d + w_e) / 4 = 68 →
  w_e > w_d →
  w_e - w_d = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_e_d_l826_82638


namespace NUMINAMATH_CALUDE_least_value_x_l826_82675

theorem least_value_x (x y z : ℕ+) (hy : y = 7) (h_least : ∀ (a b c : ℕ+), a - b - c ≥ x - y - z → a - b - c ≥ 17) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_least_value_x_l826_82675


namespace NUMINAMATH_CALUDE_arithmetic_sequence_35th_term_l826_82669

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- The 35th term of the arithmetic sequence is 99. -/
theorem arithmetic_sequence_35th_term 
  (seq : ArithmeticSequence) 
  (h15 : seq.a 15 = 33) 
  (h25 : seq.a 25 = 66) : 
  seq.a 35 = 99 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_35th_term_l826_82669


namespace NUMINAMATH_CALUDE_triangle_side_length_l826_82636

theorem triangle_side_length (X Y Z : Real) (XY : Real) :
  -- XYZ is a triangle
  X + Y + Z = Real.pi →
  -- cos(2X - Y) + sin(X + Y) = 2
  Real.cos (2 * X - Y) + Real.sin (X + Y) = 2 →
  -- XY = 6
  XY = 6 →
  -- Then YZ = 3√3
  ∃ (YZ : Real), YZ = 3 * Real.sqrt 3 := by
    sorry

end NUMINAMATH_CALUDE_triangle_side_length_l826_82636


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l826_82699

/-- A coloring of the edges of a complete graph on 6 vertices -/
def Coloring := Fin 6 → Fin 6 → Fin 5

/-- A valid coloring ensures that each vertex has exactly one edge of each color -/
def is_valid_coloring (c : Coloring) : Prop :=
  ∀ v : Fin 6, ∀ color : Fin 5,
    ∃! w : Fin 6, w ≠ v ∧ c v w = color

/-- There exists a valid coloring of the complete graph K₆ using 5 colors -/
theorem exists_valid_coloring : ∃ c : Coloring, is_valid_coloring c := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l826_82699


namespace NUMINAMATH_CALUDE_point_comparison_l826_82691

/-- Given points in a 2D coordinate system, prove that a > c -/
theorem point_comparison (a b c d e f : ℝ) : 
  b > 0 →  -- (a, b) is above x-axis
  d > 0 →  -- (c, d) is above x-axis
  f < 0 →  -- (e, f) is below x-axis
  a > 0 →  -- (a, b) is to the right of y-axis
  c > 0 →  -- (c, d) is to the right of y-axis
  e < 0 →  -- (e, f) is to the left of y-axis
  a > c →  -- (a, b) is horizontally farther from y-axis than (c, d)
  b > d →  -- (a, b) is vertically farther from x-axis than (c, d)
  a > c :=
by sorry

end NUMINAMATH_CALUDE_point_comparison_l826_82691


namespace NUMINAMATH_CALUDE_unique_k_solution_l826_82626

theorem unique_k_solution (k : ℤ) : 
  (∀ (a b c : ℝ), (a + b + c) * (a * b + b * c + c * a) + k * a * b * c = (a + b) * (b + c) * (c + a)) ↔ 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_unique_k_solution_l826_82626


namespace NUMINAMATH_CALUDE_triangle_properties_l826_82646

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : (Real.cos t.A) / (1 + Real.sin t.A) = (Real.sin (2 * t.B)) / (1 + Real.cos (2 * t.B)))
  (h2 : t.C = 2 * Real.pi / 3)
  (h3 : t.A + t.B + t.C = Real.pi)
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B)
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C)
  : 
  (t.B = Real.pi / 6) ∧ 
  (∀ (x : Triangle), x.A + x.B + x.C = Real.pi → 
    (x.a^2 + x.b^2) / x.c^2 ≥ 4 * Real.sqrt 2 - 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l826_82646


namespace NUMINAMATH_CALUDE_high_school_total_students_l826_82625

/-- Proves that the total number of students in a high school is 1800 given specific sampling conditions --/
theorem high_school_total_students
  (first_grade_students : ℕ)
  (total_sample_size : ℕ)
  (second_grade_sample : ℕ)
  (third_grade_sample : ℕ)
  (h1 : first_grade_students = 600)
  (h2 : total_sample_size = 45)
  (h3 : second_grade_sample = 20)
  (h4 : third_grade_sample = 10)
  (h5 : ∃ (total_students : ℕ), 
    (total_sample_size : ℚ) / total_students = 
    ((total_sample_size - second_grade_sample - third_grade_sample) : ℚ) / first_grade_students) :
  ∃ (total_students : ℕ), total_students = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_high_school_total_students_l826_82625


namespace NUMINAMATH_CALUDE_cd_price_difference_l826_82662

theorem cd_price_difference (album_price book_price : ℝ) (h1 : album_price = 20) (h2 : book_price = 18) : 
  let cd_price := book_price - 4
  (album_price - cd_price) / album_price * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_cd_price_difference_l826_82662


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l826_82696

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 + a 4 + a 7 = 10 →
  a 3 + a 6 + a 9 = 20 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l826_82696


namespace NUMINAMATH_CALUDE_geometric_mean_max_value_l826_82618

theorem geometric_mean_max_value (a b : ℝ) (h : a^2 = (1 + 2*b) * (1 - 2*b)) :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ x, x = (8*a*b)/(|a| + 2*|b|) → x ≤ M :=
sorry

end NUMINAMATH_CALUDE_geometric_mean_max_value_l826_82618


namespace NUMINAMATH_CALUDE_snack_spending_l826_82665

/-- The total amount spent by Robert and Teddy on snacks for their friends -/
def total_spent (pizza_price : ℕ) (pizza_quantity : ℕ) (drink_price : ℕ) (robert_drink_quantity : ℕ) 
  (hamburger_price : ℕ) (hamburger_quantity : ℕ) (teddy_drink_quantity : ℕ) : ℕ :=
  pizza_price * pizza_quantity + 
  drink_price * (robert_drink_quantity + teddy_drink_quantity) + 
  hamburger_price * hamburger_quantity

/-- Theorem stating that Robert and Teddy spend $108 in total -/
theorem snack_spending : 
  total_spent 10 5 2 10 3 6 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_snack_spending_l826_82665


namespace NUMINAMATH_CALUDE_gcd_problem_l826_82698

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, k % 2 = 1 ∧ b = k * 7769) :
  Int.gcd (4 * b^2 + 81 * b + 144) (2 * b + 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l826_82698


namespace NUMINAMATH_CALUDE_rower_upstream_speed_l826_82690

/-- Calculates the upstream speed of a rower given their still water speed and downstream speed -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Proves that given a man's speed in still water is 33 kmph and his downstream speed is 41 kmph, 
    his upstream speed is 25 kmph -/
theorem rower_upstream_speed :
  let still_water_speed := (33 : ℝ)
  let downstream_speed := (41 : ℝ)
  upstream_speed still_water_speed downstream_speed = 25 := by
sorry

#eval upstream_speed 33 41

end NUMINAMATH_CALUDE_rower_upstream_speed_l826_82690


namespace NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l826_82697

/-- The function f(x) defined as x^2 + 4x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The theorem stating that the largest value of c for which -2 is in the range of f is 2 -/
theorem largest_c_for_negative_two_in_range :
  ∀ c : ℝ, (∃ x : ℝ, f c x = -2) ↔ c ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_two_in_range_l826_82697


namespace NUMINAMATH_CALUDE_emily_candy_problem_l826_82660

/-- The number of candy pieces Emily received from neighbors -/
def candy_from_neighbors : ℕ := 5

/-- The number of candy pieces Emily ate per day -/
def candy_eaten_per_day : ℕ := 9

/-- The number of days the candy lasted -/
def days_candy_lasted : ℕ := 2

/-- The number of candy pieces Emily received from her older sister -/
def candy_from_sister : ℕ := 13

theorem emily_candy_problem :
  candy_from_sister = (candy_eaten_per_day * days_candy_lasted) - candy_from_neighbors := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_problem_l826_82660


namespace NUMINAMATH_CALUDE_cos_225_degrees_l826_82640

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l826_82640


namespace NUMINAMATH_CALUDE_min_value_of_a_l826_82645

theorem min_value_of_a (x a : ℝ) : 
  (∃ x, |x - 1| + |x + a| ≤ 8) → a ≥ -9 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l826_82645


namespace NUMINAMATH_CALUDE_swamp_flies_eaten_l826_82657

/-- Represents the number of animals in the swamp ecosystem -/
structure SwampPopulation where
  gharials : ℕ
  herons : ℕ
  caimans : ℕ
  fish : ℕ
  frogs : ℕ

/-- Calculates the total number of flies eaten daily in the swamp -/
def flies_eaten_daily (pop : SwampPopulation) : ℕ :=
  pop.frogs * 30 + pop.herons * 60

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_flies_eaten 
  (pop : SwampPopulation)
  (h_gharials : pop.gharials = 9)
  (h_herons : pop.herons = 12)
  (h_caimans : pop.caimans = 7)
  (h_fish : pop.fish = 20)
  (h_frogs : pop.frogs = 50) :
  flies_eaten_daily pop = 2220 := by
  sorry


end NUMINAMATH_CALUDE_swamp_flies_eaten_l826_82657


namespace NUMINAMATH_CALUDE_trigonometric_identities_l826_82687

theorem trigonometric_identities (α β γ : Real) (h : α + β + γ = Real.pi) :
  (Real.cos α + Real.cos β + Real.cos γ = 4 * Real.sin (α/2) * Real.sin (β/2) * Real.sin (γ/2) + 1) ∧
  (Real.cos α + Real.cos β - Real.cos γ = 4 * Real.cos (α/2) * Real.cos (β/2) * Real.sin (γ/2) - 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l826_82687


namespace NUMINAMATH_CALUDE_distance_to_gate_l826_82621

theorem distance_to_gate (field_side : ℝ) (fence_length : ℝ) (gate_distance : ℝ) :
  field_side = 84 →
  fence_length = 91 →
  gate_distance^2 + field_side^2 = fence_length^2 →
  gate_distance = 35 := by
sorry

end NUMINAMATH_CALUDE_distance_to_gate_l826_82621


namespace NUMINAMATH_CALUDE_count_sevens_to_2017_l826_82615

/-- Count of occurrences of a digit in a range of natural numbers -/
def countDigitOccurrences (digit : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The main theorem stating that the count of 7's from 1 to 2017 is 602 -/
theorem count_sevens_to_2017 : countDigitOccurrences 7 1 2017 = 602 := by
  sorry

end NUMINAMATH_CALUDE_count_sevens_to_2017_l826_82615


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l826_82604

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 2*X^2 + 3 : Polynomial ℝ) = (X^2 - 2*X + 4) * q + (4*X - 13) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l826_82604


namespace NUMINAMATH_CALUDE_triangle_area_l826_82629

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is 4 under the given conditions. -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a = 2 → c = 5 → Real.cos B = 3/5 → 
  (1/2) * a * c * Real.sin B = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l826_82629


namespace NUMINAMATH_CALUDE_intersection_points_l826_82637

-- Define the polar equations
def line_equation (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 4
def curve_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the constraints
def valid_polar_coord (ρ θ : ℝ) : Prop := ρ ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Theorem statement
theorem intersection_points :
  ∀ ρ θ, valid_polar_coord ρ θ →
    (line_equation ρ θ ∧ curve_equation ρ θ) →
    ((ρ = 4 ∧ θ = 0) ∨ (ρ = 2 * Real.sqrt 2 ∧ θ = Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_l826_82637


namespace NUMINAMATH_CALUDE_ramsey_bound_exists_l826_82681

/-- The maximum degree of a graph -/
def maxDegree (G : SimpleGraph α) : ℕ := sorry

/-- The Ramsey number of a graph -/
def ramseyNumber (G : SimpleGraph α) : ℕ := sorry

/-- The order (number of vertices) of a graph -/
def graphOrder (G : SimpleGraph α) : ℕ := sorry

/-- For every positive integer Δ, there exists a constant c such that
    all graphs H with maximum degree at most Δ have R(H) ≤ c|H| -/
theorem ramsey_bound_exists {α : Type*} :
  ∀ Δ : ℕ, Δ > 0 →
  ∃ c : ℝ, c > 0 ∧
  ∀ (H : SimpleGraph α), maxDegree H ≤ Δ →
  (ramseyNumber H : ℝ) ≤ c * (graphOrder H) :=
sorry

end NUMINAMATH_CALUDE_ramsey_bound_exists_l826_82681


namespace NUMINAMATH_CALUDE_simplify_expression_l826_82616

theorem simplify_expression (y : ℝ) : 
  3*y - 7*y^2 + 12 - (5 - 9*y + 4*y^2) = -11*y^2 + 12*y + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l826_82616


namespace NUMINAMATH_CALUDE_cough_ratio_l826_82676

-- Define the number of coughs per minute for Georgia
def georgia_coughs_per_minute : ℕ := 5

-- Define the total number of coughs after 20 minutes
def total_coughs_after_20_minutes : ℕ := 300

-- Define Robert's coughs per minute
def robert_coughs_per_minute : ℕ := (total_coughs_after_20_minutes - georgia_coughs_per_minute * 20) / 20

-- Theorem stating the ratio of Robert's coughs to Georgia's coughs is 2:1
theorem cough_ratio :
  robert_coughs_per_minute / georgia_coughs_per_minute = 2 ∧
  robert_coughs_per_minute > georgia_coughs_per_minute :=
by sorry

end NUMINAMATH_CALUDE_cough_ratio_l826_82676


namespace NUMINAMATH_CALUDE_water_to_pool_volume_l826_82692

/-- Proves that one gallon of water fills 1 cubic foot of Jerry's pool --/
theorem water_to_pool_volume 
  (total_water : ℝ) 
  (drinking_cooking : ℝ) 
  (shower_water : ℝ) 
  (pool_length pool_width pool_height : ℝ) 
  (num_showers : ℕ) 
  (h1 : total_water = 1000) 
  (h2 : drinking_cooking = 100) 
  (h3 : shower_water = 20) 
  (h4 : pool_length = 10 ∧ pool_width = 10 ∧ pool_height = 6) 
  (h5 : num_showers = 15) : 
  (total_water - drinking_cooking - num_showers * shower_water) / (pool_length * pool_width * pool_height) = 1 := by
  sorry

end NUMINAMATH_CALUDE_water_to_pool_volume_l826_82692


namespace NUMINAMATH_CALUDE_soccer_team_selection_l826_82632

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def players_to_select : ℕ := 7
def max_quadruplets : ℕ := 2

theorem soccer_team_selection :
  (Nat.choose total_players players_to_select) -
  ((Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (players_to_select - 3)) +
   (Nat.choose quadruplets 4) * (Nat.choose (total_players - quadruplets) (players_to_select - 4))) = 9240 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l826_82632


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l826_82639

/-- Two 3x3 matrices that are inverses of each other -/
def matrix1 (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![a, 1, b; 2, 2, 3; c, 5, d]
def matrix2 (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![-5, e, -11; f, -13, g; 2, h, 4]

/-- The theorem stating that the sum of all variables is 45 -/
theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  (matrix1 a b c d) * (matrix2 e f g h) = 1 →
  a + b + c + d + e + f + g + h = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l826_82639


namespace NUMINAMATH_CALUDE_q_array_sum_formula_l826_82620

/-- Definition of a 1/q-array sum -/
def qArraySum (q : ℚ) : ℚ :=
  (2 * q^2) / ((2*q - 1) * (q - 1))

/-- Theorem: The sum of all terms in a 1/q-array with the given properties is (2q^2) / ((2q-1)(q-1)) -/
theorem q_array_sum_formula (q : ℚ) (hq : q ≠ 0) (hq1 : q ≠ 1/2) (hq2 : q ≠ 1) : 
  qArraySum q = ∑' (r : ℕ) (c : ℕ), (1 / (2*q)^r) * (1 / q^c) :=
sorry

#eval (qArraySum 1220).num % 1220 + (qArraySum 1220).den % 1220

end NUMINAMATH_CALUDE_q_array_sum_formula_l826_82620


namespace NUMINAMATH_CALUDE_proposition_truth_count_l826_82667

theorem proposition_truth_count : 
  let P1 := ∀ x : ℝ, x > -3 → x > -6
  let P2 := ∀ x : ℝ, x > -6 → x > -3
  let P3 := ∀ x : ℝ, x ≤ -3 → x ≤ -6
  let P4 := ∀ x : ℝ, x ≤ -6 → x ≤ -3
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) ∨
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨
  (P1 ∧ P2 ∧ ¬P3 ∧ ¬P4) ∨
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨
  (¬P1 ∧ ¬P2 ∧ P3 ∧ P4) :=
by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_count_l826_82667


namespace NUMINAMATH_CALUDE_sandy_sums_attempted_sandy_specific_case_l826_82672

theorem sandy_sums_attempted (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) 
  (total_marks : ℕ) (correct_sums : ℕ) : ℕ :=
  let total_sums := correct_sums + (marks_per_correct * correct_sums - total_marks) / marks_per_incorrect
  total_sums

theorem sandy_specific_case : sandy_sums_attempted 3 2 65 25 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_sums_attempted_sandy_specific_case_l826_82672


namespace NUMINAMATH_CALUDE_analysis_time_proof_l826_82661

/-- The number of bones in a human body -/
def num_bones : ℕ := 206

/-- The time (in hours) required to analyze one bone -/
def time_per_bone : ℕ := 1

/-- The total time required to analyze all bones in a human body -/
def total_analysis_time : ℕ := num_bones * time_per_bone

theorem analysis_time_proof : total_analysis_time = 206 := by
  sorry

end NUMINAMATH_CALUDE_analysis_time_proof_l826_82661


namespace NUMINAMATH_CALUDE_virgo_island_trip_duration_l826_82630

/-- The duration of Tom's trip to "Virgo" island -/
theorem virgo_island_trip_duration :
  ∀ (boat_duration : ℝ) (plane_duration : ℝ),
    boat_duration = 2 →
    plane_duration = 4 * boat_duration →
    boat_duration + plane_duration = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_virgo_island_trip_duration_l826_82630


namespace NUMINAMATH_CALUDE_speech_competition_arrangements_l826_82666

/-- The number of students in the speech competition -/
def num_students : ℕ := 6

/-- The total number of arrangements where B and C are adjacent -/
def total_arrangements_bc_adjacent : ℕ := 240

/-- The number of arrangements where A is first or last, and B and C are adjacent -/
def arrangements_a_first_or_last : ℕ := 96

/-- The number of valid arrangements for the speech competition -/
def valid_arrangements : ℕ := total_arrangements_bc_adjacent - arrangements_a_first_or_last

theorem speech_competition_arrangements :
  valid_arrangements = 144 :=
sorry

end NUMINAMATH_CALUDE_speech_competition_arrangements_l826_82666


namespace NUMINAMATH_CALUDE_trisha_annual_take_home_pay_l826_82679

/-- Calculates the annual take-home pay given hourly rate, weekly hours, weeks worked, and withholding rate. -/
def annual_take_home_pay (hourly_rate : ℝ) (weekly_hours : ℝ) (weeks_worked : ℝ) (withholding_rate : ℝ) : ℝ :=
  let gross_pay := hourly_rate * weekly_hours * weeks_worked
  let withheld_amount := withholding_rate * gross_pay
  gross_pay - withheld_amount

/-- Proves that given the specified conditions, Trisha's annual take-home pay is $24,960. -/
theorem trisha_annual_take_home_pay :
  annual_take_home_pay 15 40 52 0.2 = 24960 := by
  sorry

#eval annual_take_home_pay 15 40 52 0.2

end NUMINAMATH_CALUDE_trisha_annual_take_home_pay_l826_82679


namespace NUMINAMATH_CALUDE_divisible_by_twelve_l826_82633

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def last_two_digits (n : ℕ) : ℕ := n % 100

def six_digit_number (a b c d e f : ℕ) : ℕ :=
  100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f

theorem divisible_by_twelve (square : ℕ) :
  is_divisible_by (six_digit_number 4 8 6 3 square 5) 12 ↔ square = 1 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_twelve_l826_82633


namespace NUMINAMATH_CALUDE_fair_cost_calculation_l826_82689

/-- Calculate the total cost for Joe and the twins at the fair -/
theorem fair_cost_calculation (entrance_fee_under_18 : ℚ) 
  (entrance_fee_over_18_multiplier : ℚ) (group_discount : ℚ) 
  (low_thrill_under_18 : ℚ) (low_thrill_over_18 : ℚ)
  (medium_thrill_under_18 : ℚ) (medium_thrill_over_18 : ℚ)
  (high_thrill_under_18 : ℚ) (high_thrill_over_18 : ℚ)
  (joe_age : ℕ) (twin_age : ℕ)
  (joe_low : ℕ) (joe_medium : ℕ) (joe_high : ℕ)
  (twin_a_low : ℕ) (twin_a_medium : ℕ)
  (twin_b_low : ℕ) (twin_b_high : ℕ) :
  entrance_fee_under_18 = 5 →
  entrance_fee_over_18_multiplier = 1.2 →
  group_discount = 0.85 →
  low_thrill_under_18 = 0.5 →
  low_thrill_over_18 = 0.7 →
  medium_thrill_under_18 = 1 →
  medium_thrill_over_18 = 1.2 →
  high_thrill_under_18 = 1.5 →
  high_thrill_over_18 = 1.7 →
  joe_age = 30 →
  twin_age = 6 →
  joe_low = 2 →
  joe_medium = 1 →
  joe_high = 1 →
  twin_a_low = 2 →
  twin_a_medium = 1 →
  twin_b_low = 3 →
  twin_b_high = 2 →
  (entrance_fee_under_18 * entrance_fee_over_18_multiplier * group_discount + 
   2 * entrance_fee_under_18 * group_discount +
   joe_low * low_thrill_over_18 + joe_medium * medium_thrill_over_18 + 
   joe_high * high_thrill_over_18 +
   twin_a_low * low_thrill_under_18 + twin_a_medium * medium_thrill_under_18 +
   twin_b_low * low_thrill_under_18 + twin_b_high * high_thrill_under_18) = 24.4 := by
  sorry

end NUMINAMATH_CALUDE_fair_cost_calculation_l826_82689


namespace NUMINAMATH_CALUDE_total_fruits_in_garden_l826_82663

def papaya_production : List Nat := [10, 12]
def mango_production : List Nat := [18, 20, 22]
def apple_production : List Nat := [14, 15, 16, 17]
def orange_production : List Nat := [20, 23, 25, 27, 30]

theorem total_fruits_in_garden : 
  (papaya_production.sum + mango_production.sum + 
   apple_production.sum + orange_production.sum) = 269 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_in_garden_l826_82663


namespace NUMINAMATH_CALUDE_rectangle_side_difference_l826_82617

theorem rectangle_side_difference (p d : ℝ) (hp : p > 0) (hd : d > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x > y ∧ 
    2 * (x + y) = p ∧ 
    x^2 + y^2 = d^2 ∧
    x - y = (Real.sqrt (8 * d^2 - p^2)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_difference_l826_82617


namespace NUMINAMATH_CALUDE_train_length_l826_82685

/-- Given a train with speed 72 km/hr crossing a 260 m platform in 26 seconds, prove its length is 260 m -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * 1000 / 3600 →
  platform_length = 260 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 260 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l826_82685


namespace NUMINAMATH_CALUDE_samples_left_over_proof_l826_82677

/-- Calculates the number of samples left over given the number of samples per box,
    number of boxes opened, and number of customers who tried a sample. -/
def samples_left_over (samples_per_box : ℕ) (boxes_opened : ℕ) (customers : ℕ) : ℕ :=
  samples_per_box * boxes_opened - customers

/-- Proves that given 20 samples per box, 12 boxes opened, and 235 customers,
    the number of samples left over is 5. -/
theorem samples_left_over_proof :
  samples_left_over 20 12 235 = 5 := by
  sorry

end NUMINAMATH_CALUDE_samples_left_over_proof_l826_82677


namespace NUMINAMATH_CALUDE_jills_uphill_speed_l826_82659

/-- Jill's speed running up the hill -/
def uphill_speed : ℝ := 9

/-- Jill's speed running down the hill -/
def downhill_speed : ℝ := 12

/-- Hill height in feet -/
def hill_height : ℝ := 900

/-- Total time for running up and down the hill in seconds -/
def total_time : ℝ := 175

theorem jills_uphill_speed :
  (hill_height / uphill_speed + hill_height / downhill_speed = total_time) ∧
  (uphill_speed > 0) ∧
  (downhill_speed > 0) ∧
  (hill_height > 0) ∧
  (total_time > 0) := by
  sorry

end NUMINAMATH_CALUDE_jills_uphill_speed_l826_82659


namespace NUMINAMATH_CALUDE_exists_multiplicative_identity_l826_82602

theorem exists_multiplicative_identity : ∃ n : ℝ, ∀ m : ℝ, m * n = m := by
  sorry

end NUMINAMATH_CALUDE_exists_multiplicative_identity_l826_82602


namespace NUMINAMATH_CALUDE_scientific_notation_3650000_l826_82655

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Function to convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_3650000 :
  toScientificNotation 3650000 = ScientificNotation.mk 3.65 6 sorry := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3650000_l826_82655


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l826_82683

theorem min_value_sum_reciprocals (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → a / b + b / a ≤ x / y + y / x) ∧ 
  (a / b + b / a = 2) := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l826_82683


namespace NUMINAMATH_CALUDE_decomposition_count_l826_82624

theorem decomposition_count (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (c d : ℕ), (c, d) ∈ s ↔ 
      c * d = p^2 * q^2 ∧ 
      c < d ∧ 
      d < p * q) := by sorry

end NUMINAMATH_CALUDE_decomposition_count_l826_82624


namespace NUMINAMATH_CALUDE_polynomial_factorization_l826_82644

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 5*x^4 + 10*x^2 - 4 = (x - 1) * (x + 1) * (x^2 - 2)^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l826_82644


namespace NUMINAMATH_CALUDE_perfect_squares_among_expressions_l826_82607

-- Define the expressions
def A : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^10 * 3^12 * 7^14
def B : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^12 * 3^15 * 7^10
def C : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^9 * 3^18 * 7^15
def D : ℕ → ℕ → ℕ → ℕ := λ a b c => 2^20 * 3^16 * 7^12

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

-- Theorem statement
theorem perfect_squares_among_expressions :
  (is_perfect_square (A 2 3 7)) ∧
  (¬ is_perfect_square (B 2 3 7)) ∧
  (¬ is_perfect_square (C 2 3 7)) ∧
  (is_perfect_square (D 2 3 7)) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_among_expressions_l826_82607


namespace NUMINAMATH_CALUDE_quadratic_function_property_l826_82650

/-- A quadratic function with specific properties -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (∀ x, f x ≤ f 3) ∧
    (f 3 = 10) ∧
    (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 4)

/-- Theorem stating that f(5) = 0 for the specified quadratic function -/
theorem quadratic_function_property (f : ℝ → ℝ) (h : quadratic_function f) : f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l826_82650


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l826_82673

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_first : a 1 = 3/5)
  (h_ninth : a 9 = 2/3) :
  a 5 = 19/30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l826_82673


namespace NUMINAMATH_CALUDE_solve_clothing_problem_l826_82619

def clothing_problem (total : ℕ) (num_loads : ℕ) (pieces_per_load : ℕ) : Prop :=
  let remaining := num_loads * pieces_per_load
  let first_load := total - remaining
  first_load = 19

theorem solve_clothing_problem :
  clothing_problem 39 5 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_clothing_problem_l826_82619


namespace NUMINAMATH_CALUDE_binomial_11_choose_9_l826_82670

theorem binomial_11_choose_9 : Nat.choose 11 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_binomial_11_choose_9_l826_82670


namespace NUMINAMATH_CALUDE_exactly_one_first_class_product_l826_82641

theorem exactly_one_first_class_product (p1 p2 : ℝ) 
  (h1 : p1 = 2/3) 
  (h2 : p2 = 3/4) : 
  p1 * (1 - p2) + (1 - p1) * p2 = 5/12 := by
sorry

end NUMINAMATH_CALUDE_exactly_one_first_class_product_l826_82641


namespace NUMINAMATH_CALUDE_shopping_money_calculation_l826_82647

theorem shopping_money_calculation (M : ℚ) : 
  (1 - 4/5 * (1 - 1/3 * (1 - 3/8))) * M = 1200 → M = 14400 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_calculation_l826_82647


namespace NUMINAMATH_CALUDE_inequality_system_solution_l826_82653

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 2 < 2*m ∧ x - m < 0) ↔ x < 2*m - 2) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l826_82653


namespace NUMINAMATH_CALUDE_third_median_length_l826_82631

/-- A triangle with two known medians and area -/
structure Triangle where
  median1 : ℝ
  median2 : ℝ
  area : ℝ

/-- The length of the third median in a triangle -/
def third_median (t : Triangle) : ℝ := sorry

theorem third_median_length (t : Triangle) 
  (h1 : t.median1 = 4)
  (h2 : t.median2 = 5)
  (h3 : t.area = 10 * Real.sqrt 3) :
  third_median t = 3 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_third_median_length_l826_82631


namespace NUMINAMATH_CALUDE_subset_relation_l826_82668

def set_A (a : ℝ) : Set ℝ := {x | 0 < a * x + 1 ∧ a * x + 1 ≤ 5}
def set_B : Set ℝ := {x | -1/2 < x ∧ x ≤ 2}

theorem subset_relation (a : ℝ) :
  (∀ x : ℝ, x ∈ set_B → x ∈ set_A 1) ∧
  (∀ x : ℝ, x ∈ set_A a → x ∈ set_B ↔ a < -8 ∨ a ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_subset_relation_l826_82668


namespace NUMINAMATH_CALUDE_sarah_candy_problem_l826_82611

/-- The number of candy pieces Sarah received from neighbors -/
def candy_from_neighbors : ℕ := 66

/-- The number of candy pieces Sarah ate per day -/
def candy_per_day : ℕ := 9

/-- The number of days the candy lasted -/
def days_lasted : ℕ := 9

/-- The number of candy pieces Sarah received from her older sister -/
def candy_from_sister : ℕ := 15

theorem sarah_candy_problem :
  candy_from_sister = days_lasted * candy_per_day - candy_from_neighbors :=
by sorry

end NUMINAMATH_CALUDE_sarah_candy_problem_l826_82611


namespace NUMINAMATH_CALUDE_coloring_theorem_l826_82693

/-- A coloring of natural numbers using k colors -/
def Coloring (k : ℕ) := ℕ → Fin k

/-- Proposition: For any coloring of natural numbers using k colors,
    there exist four distinct natural numbers a, b, c, d of the same color
    satisfying the required properties. -/
theorem coloring_theorem (k : ℕ) (coloring : Coloring k) :
  ∃ (a b c d : ℕ) (color : Fin k),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    coloring a = color ∧ coloring b = color ∧ coloring c = color ∧ coloring d = color ∧
    a * d = b * c ∧
    (∃ m : ℕ, b = a * 2^m) ∧
    (∃ n : ℕ, c = a * 3^n) :=
  sorry

end NUMINAMATH_CALUDE_coloring_theorem_l826_82693


namespace NUMINAMATH_CALUDE_randy_blocks_problem_l826_82634

theorem randy_blocks_problem (blocks_used blocks_left : ℕ) 
  (h1 : blocks_used = 36)
  (h2 : blocks_left = 23) :
  blocks_used + blocks_left = 59 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_problem_l826_82634


namespace NUMINAMATH_CALUDE_sqrt_meaningful_condition_l826_82643

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_condition_l826_82643


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l826_82695

theorem imaginary_part_of_complex_number : 
  Complex.im ((2 : ℂ) + Complex.I * Complex.I) = 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l826_82695


namespace NUMINAMATH_CALUDE_unique_number_sum_of_digits_l826_82649

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem unique_number_sum_of_digits :
  ∃! N : ℕ, 
    400 < N ∧ N < 600 ∧ 
    N % 2 = 1 ∧ 
    N % 5 = 0 ∧ 
    N % 11 = 0 ∧
    sum_of_digits N = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_sum_of_digits_l826_82649


namespace NUMINAMATH_CALUDE_sum_proper_divisors_81_l826_82628

def proper_divisors (n : ℕ) : Set ℕ :=
  {d : ℕ | d ∣ n ∧ d ≠ n}

theorem sum_proper_divisors_81 :
  (Finset.sum (Finset.filter (· ≠ 81) (Finset.range 82)) (λ x => if x ∣ 81 then x else 0)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_proper_divisors_81_l826_82628


namespace NUMINAMATH_CALUDE_sunglasses_sign_cost_l826_82688

theorem sunglasses_sign_cost (selling_price cost_price : ℕ) (pairs_sold : ℕ) : 
  selling_price = 30 →
  cost_price = 26 →
  pairs_sold = 10 →
  (pairs_sold * (selling_price - cost_price)) / 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_sign_cost_l826_82688


namespace NUMINAMATH_CALUDE_calculate_expression_l826_82684

theorem calculate_expression : 15 * (216 / 3 + 36 / 9 + 16 / 25 + 2^2) = 30240 / 25 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l826_82684


namespace NUMINAMATH_CALUDE_circle_a_l826_82682

theorem circle_a (x y : ℝ) :
  (x - 3)^2 + (y + 2)^2 = 16 → 
  ∃ (center : ℝ × ℝ) (radius : ℝ), center = (3, -2) ∧ radius = 4 :=
by sorry


end NUMINAMATH_CALUDE_circle_a_l826_82682


namespace NUMINAMATH_CALUDE_line_intercepts_l826_82651

/-- Given a line with equation 4x + 6y = 24, prove its x-intercept and y-intercept -/
theorem line_intercepts (x y : ℝ) :
  4 * x + 6 * y = 24 →
  (x = 6 ∧ y = 0) ∨ (x = 0 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_l826_82651


namespace NUMINAMATH_CALUDE_triangle_side_length_l826_82674

theorem triangle_side_length (A B C M : ℝ × ℝ) : 
  -- Triangle ABC is right-angled at C
  (B.1 - A.1) * (C.2 - A.2) = (C.1 - A.1) * (B.2 - A.2) →
  -- AC = BC
  ((C.1 - A.1)^2 + (C.2 - A.2)^2) = ((C.1 - B.1)^2 + (C.2 - B.2)^2) →
  -- M is an interior point (implied by the distances)
  -- MC = 1
  ((M.1 - C.1)^2 + (M.2 - C.2)^2) = 1 →
  -- MA = 2
  ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 4 →
  -- MB = √2
  ((M.1 - B.1)^2 + (M.2 - B.2)^2) = 2 →
  -- AB = √10
  ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l826_82674


namespace NUMINAMATH_CALUDE_independence_test_not_always_correct_l826_82622

-- Define what an independence test is
def IndependenceTest : Type := sorry

-- Define a function that represents the conclusion of an independence test
def conclusion (test : IndependenceTest) : Prop := sorry

-- Theorem stating that the conclusion of an independence test is not always correct
theorem independence_test_not_always_correct :
  ¬ (∀ (test : IndependenceTest), conclusion test) := by sorry

end NUMINAMATH_CALUDE_independence_test_not_always_correct_l826_82622


namespace NUMINAMATH_CALUDE_isosceles_60_similar_l826_82612

-- Define an isosceles triangle with a given angle
structure IsoscelesTriangle :=
  (angle : ℝ)
  (is_isosceles : angle > 0 ∧ angle < π)

-- Define similarity for isosceles triangles
def are_similar (t1 t2 : IsoscelesTriangle) : Prop :=
  t1.angle = t2.angle

-- Theorem: Two isosceles triangles with an internal angle of 60° are similar
theorem isosceles_60_similar (t1 t2 : IsoscelesTriangle) 
  (h1 : t1.angle = π/3) (h2 : t2.angle = π/3) : 
  are_similar t1 t2 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_60_similar_l826_82612
