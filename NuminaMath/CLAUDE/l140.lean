import Mathlib

namespace NUMINAMATH_CALUDE_max_value_of_function_l140_14003

theorem max_value_of_function (x : Real) (h : x ∈ Set.Ioo 0 Real.pi) :
  (2 * Real.sin (x / 2) * (1 - Real.sin (x / 2)) * (1 + Real.sin (x / 2))^2) ≤ (107 + 51 * Real.sqrt 17) / 256 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l140_14003


namespace NUMINAMATH_CALUDE_inequality_system_integer_solutions_l140_14075

theorem inequality_system_integer_solutions :
  let S := {x : ℤ | (x - 1 : ℚ) / 2 ≥ (x - 2 : ℚ) / 3 ∧ (2 * x - 5 : ℤ) < -3 * x}
  S = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_inequality_system_integer_solutions_l140_14075


namespace NUMINAMATH_CALUDE_work_payment_proof_l140_14036

/-- Calculates the total payment for a bricklayer and an electrician's work -/
def total_payment (total_hours : ℝ) (bricklayer_hours : ℝ) (bricklayer_rate : ℝ) (electrician_rate : ℝ) : ℝ :=
  let electrician_hours := total_hours - bricklayer_hours
  bricklayer_hours * bricklayer_rate + electrician_hours * electrician_rate

/-- Proves that the total payment for the given work scenario is $1170 -/
theorem work_payment_proof :
  total_payment 90 67.5 12 16 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_work_payment_proof_l140_14036


namespace NUMINAMATH_CALUDE_stating_count_sequences_l140_14038

/-- 
Given positive integers n and k where 1 ≤ k < n, T(n, k) represents the number of 
sequences of k positive integers that sum to n.
-/
def T (n k : ℕ) : ℕ := sorry

/-- 
Theorem stating that T(n, k) is equal to (n-1) choose (k-1) for 1 ≤ k < n.
-/
theorem count_sequences (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  T n k = Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_stating_count_sequences_l140_14038


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l140_14076

/-- Given a grasshopper's initial position and first jump endpoint, calculate its final position after a second identical jump -/
theorem grasshopper_jumps (initial_pos : ℝ) (first_jump_end : ℝ) : 
  initial_pos = 8 → first_jump_end = 17.5 → 
  let jump_length := first_jump_end - initial_pos
  first_jump_end + jump_length = 27 := by
sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l140_14076


namespace NUMINAMATH_CALUDE_union_of_sets_l140_14083

theorem union_of_sets (a : ℤ) : 
  let A : Set ℤ := {|a + 1|, 3, 5}
  let B : Set ℤ := {2*a + 1, a^2 + 2*a, a^2 + 2*a - 1}
  (A ∩ B = {2, 3}) → (A ∪ B = {-5, 2, 3, 5}) :=
by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l140_14083


namespace NUMINAMATH_CALUDE_quadratic_root_difference_sum_l140_14077

def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 13 * x - 6 = 0

def is_square_free (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p^2 ∣ n) → p = 1

theorem quadratic_root_difference_sum (p q : ℕ) (hp : is_square_free p) :
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ 
    |x₁ - x₂| = (Real.sqrt (p : ℝ)) / (q : ℝ)) →
  p + q = 294 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_sum_l140_14077


namespace NUMINAMATH_CALUDE_incenter_in_triangular_prism_l140_14006

structure TriangularPrism where
  A : Point
  B : Point
  C : Point
  D : Point

def orthogonal_projection (p : Point) (plane : Set Point) : Point :=
  sorry

def distance_to_face (p : Point) (face : Set Point) : ℝ :=
  sorry

def is_incenter (p : Point) (triangle : Set Point) : Prop :=
  sorry

theorem incenter_in_triangular_prism (prism : TriangularPrism) 
  (O : Point) 
  (h1 : O = orthogonal_projection prism.A {prism.B, prism.C, prism.D}) 
  (h2 : distance_to_face O {prism.B, prism.C, prism.D} = 
        distance_to_face O {prism.A, prism.B, prism.D} ∧
        distance_to_face O {prism.B, prism.C, prism.D} = 
        distance_to_face O {prism.A, prism.C, prism.D}) : 
  is_incenter O {prism.B, prism.C, prism.D} :=
sorry

end NUMINAMATH_CALUDE_incenter_in_triangular_prism_l140_14006


namespace NUMINAMATH_CALUDE_lcm_problem_l140_14014

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 16) (h2 : Nat.lcm b c = 21) :
  Nat.lcm a c ≥ 336 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l140_14014


namespace NUMINAMATH_CALUDE_solution_existence_unique_solution_l140_14096

noncomputable def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x ≠ 1 ∧ 2*a - x > 0 ∧
    (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
    (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
    1 / (Real.log 2 / Real.log (a^2 - 1))

noncomputable def has_unique_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ x ≠ 1 ∧ 2*a - x > 0 ∧
    (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
    (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
    1 / (Real.log 2 / Real.log (a^2 - 1))

theorem solution_existence (a : ℝ) :
  has_solution a ↔ (a > 1 ∧ a ≠ Real.sqrt 2) :=
sorry

theorem unique_solution (a : ℝ) :
  has_unique_solution a ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_existence_unique_solution_l140_14096


namespace NUMINAMATH_CALUDE_ecommerce_problem_l140_14085

theorem ecommerce_problem (total_spent : ℝ) (price_difference : ℝ) (total_items : ℕ) 
  (subsidy_rate : ℝ) (max_subsidy : ℝ) 
  (h1 : total_spent = 3000)
  (h2 : price_difference = 600)
  (h3 : total_items = 300)
  (h4 : subsidy_rate = 0.1)
  (h5 : max_subsidy = 50000) :
  ∃ (leather_price sweater_price : ℝ) (min_sweaters : ℕ),
    leather_price = 2600 ∧ 
    sweater_price = 400 ∧ 
    min_sweaters = 128 ∧
    leather_price + sweater_price = total_spent ∧
    leather_price = 5 * sweater_price + price_difference ∧
    (↑min_sweaters : ℝ) ≥ (max_subsidy / subsidy_rate - total_items * leather_price) / (sweater_price - leather_price) := by
  sorry

end NUMINAMATH_CALUDE_ecommerce_problem_l140_14085


namespace NUMINAMATH_CALUDE_acute_triangle_probability_condition_l140_14054

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

end NUMINAMATH_CALUDE_acute_triangle_probability_condition_l140_14054


namespace NUMINAMATH_CALUDE_distance_from_B_to_center_l140_14065

-- Define the circle and points
def circle_radius : ℝ := 10
def vertical_distance : ℝ := 6
def horizontal_distance : ℝ := 4

-- Define the points A, B, and C
def point_B (a b : ℝ) : ℝ × ℝ := (a, b)
def point_A (a b : ℝ) : ℝ × ℝ := (a, b + vertical_distance)
def point_C (a b : ℝ) : ℝ × ℝ := (a + horizontal_distance, b)

-- Define the conditions
def on_circle (x y : ℝ) : Prop := x^2 + y^2 = circle_radius^2
def right_angle (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (x1 - x2) * (x3 - x2) + (y1 - y2) * (y3 - y2) = 0

-- Theorem statement
theorem distance_from_B_to_center (a b : ℝ) :
  on_circle a (b + vertical_distance) →
  on_circle (a + horizontal_distance) b →
  right_angle (point_A a b) (point_B a b) (point_C a b) →
  a^2 + b^2 = 74 :=
sorry

end NUMINAMATH_CALUDE_distance_from_B_to_center_l140_14065


namespace NUMINAMATH_CALUDE_win_sector_area_l140_14040

/-- Given a circular spinner with radius 8 cm and probability of winning 3/8,
    prove that the area of the WIN sector is 24π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_prob : ℝ) (win_area : ℝ) : 
  radius = 8 →
  win_prob = 3 / 8 →
  win_area = win_prob * π * radius^2 →
  win_area = 24 * π := by
  sorry

#check win_sector_area

end NUMINAMATH_CALUDE_win_sector_area_l140_14040


namespace NUMINAMATH_CALUDE_product_of_imaginary_parts_l140_14099

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := z^2 + 3*z = 3 - 4*i

-- Define a function to get the imaginary part of a complex number
def imag (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem product_of_imaginary_parts : 
  ∃ (z₁ z₂ : ℂ), equation z₁ ∧ equation z₂ ∧ z₁ ≠ z₂ ∧ (imag z₁ * imag z₂ = 16/25) :=
sorry

end NUMINAMATH_CALUDE_product_of_imaginary_parts_l140_14099


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l140_14037

theorem expression_simplification_and_evaluation (x : ℝ) 
  (h1 : x ≠ -2) (h2 : x ≠ 0) (h3 : x ≠ 2) :
  (x^2 / (x - 2) + 4 / (2 - x)) / ((x^2 + 4*x + 4) / x) = x / (x + 2) ∧
  (1 : ℝ) / (1 + 2) = (1 : ℝ) / 3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l140_14037


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_stewart_farm_sheep_count_proof_l140_14017

theorem stewart_farm_sheep_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun sheep_count horse_count sheep_ratio horse_ratio =>
    sheep_count * horse_ratio = horse_count * sheep_ratio ∧
    horse_count * 230 = 12880 →
    sheep_count = 40

-- The proof is omitted
theorem stewart_farm_sheep_count_proof : stewart_farm_sheep_count 40 56 5 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_stewart_farm_sheep_count_proof_l140_14017


namespace NUMINAMATH_CALUDE_sequence_general_term_l140_14092

theorem sequence_general_term (n : ℕ) :
  let S : ℕ → ℤ := λ k => 3 * k^2 - 2 * k
  let a : ℕ → ℤ := λ k => S k - S (k - 1)
  a n = 6 * n - 5 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l140_14092


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l140_14008

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

end NUMINAMATH_CALUDE_shaded_area_fraction_l140_14008


namespace NUMINAMATH_CALUDE_largest_inscribed_pentagon_is_regular_l140_14044

/-- A pentagon inscribed in a circle of radius 1 --/
structure InscribedPentagon where
  /-- The vertices of the pentagon --/
  vertices : Fin 5 → ℝ × ℝ
  /-- All vertices lie on the unit circle --/
  on_circle : ∀ i, (vertices i).1^2 + (vertices i).2^2 = 1

/-- The area of an inscribed pentagon --/
def area (p : InscribedPentagon) : ℝ :=
  sorry

/-- A regular pentagon inscribed in a circle of radius 1 --/
def regular_pentagon : InscribedPentagon :=
  sorry

theorem largest_inscribed_pentagon_is_regular :
  ∀ p : InscribedPentagon, area p ≤ area regular_pentagon :=
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_pentagon_is_regular_l140_14044


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l140_14007

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y : ℝ, x ≥ 1 ∧ y ≥ 2 → x + y ≥ 3) ∧
  (∃ x y : ℝ, x + y ≥ 3 ∧ ¬(x ≥ 1 ∧ y ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l140_14007


namespace NUMINAMATH_CALUDE_meatball_fraction_eaten_l140_14046

/-- Given 3 plates with 3 meatballs each, if 3 people eat the same fraction of meatballs from their respective plates and 3 meatballs are left in total, then each person ate 2/3 of the meatballs on their plate. -/
theorem meatball_fraction_eaten (f : ℚ) 
  (h1 : f ≥ 0) 
  (h2 : f ≤ 1) 
  (h3 : 3 * (3 - 3 * f) = 3) : 
  f = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_meatball_fraction_eaten_l140_14046


namespace NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_not_p_sufficient_not_necessary_for_not_q_l140_14082

/-- Proposition p: x^2 - 4ax + 3a^2 < 0 -/
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0 -/
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

theorem range_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

theorem range_a_for_not_p_sufficient_not_necessary_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, (¬p x a → (x^2 - x - 6 > 0 ∨ x^2 + 2*x - 8 ≤ 0)) ∧
    ∃ x : ℝ, (x^2 - x - 6 > 0 ∨ x^2 + 2*x - 8 ≤ 0) ∧ p x a) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_not_p_sufficient_not_necessary_for_not_q_l140_14082


namespace NUMINAMATH_CALUDE_sqrt_77_plus_28sqrt3_l140_14087

theorem sqrt_77_plus_28sqrt3 :
  ∃ (x y z : ℤ), 
    (∀ (k : ℕ), k > 1 → ¬ (∃ (m : ℕ), z = k^2 * m)) →
    (x + y * Real.sqrt z : ℝ) = Real.sqrt (77 + 28 * Real.sqrt 3) ∧
    x = 7 ∧ y = 2 ∧ z = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_77_plus_28sqrt3_l140_14087


namespace NUMINAMATH_CALUDE_certain_number_sum_l140_14027

theorem certain_number_sum (x : ℤ) : x + (-27) = 30 → x = 57 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_sum_l140_14027


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l140_14084

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (s : ℝ) (h : s = 8) :
  let cube_volume := s^3
  let tetrahedron_volume := cube_volume / 3
  tetrahedron_volume = 512 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l140_14084


namespace NUMINAMATH_CALUDE_percentage_relations_l140_14090

theorem percentage_relations (x y z w : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : y = 0.5 * z) 
  (h3 : w = 2 * x) : 
  x = 0.65 * z ∧ y = 0.5 * z ∧ w = 1.3 * z := by
  sorry

end NUMINAMATH_CALUDE_percentage_relations_l140_14090


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l140_14066

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n, a (n + 1) = a n * r

/-- The factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n. -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem geometric_sequence_first_term (a : ℕ → ℚ) :
  IsGeometricSequence a →
  a 7 = factorial 8 →
  a 10 = factorial 11 →
  a 1 = 8 / 245 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l140_14066


namespace NUMINAMATH_CALUDE_concatenated_number_500_not_divisible_by_9_l140_14061

def concatenated_number (n : ℕ) : ℕ := sorry

theorem concatenated_number_500_not_divisible_by_9 :
  ¬ (9 ∣ concatenated_number 500) := by sorry

end NUMINAMATH_CALUDE_concatenated_number_500_not_divisible_by_9_l140_14061


namespace NUMINAMATH_CALUDE_library_visitors_average_l140_14063

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (month_days : ℕ) (sundays_in_month : ℕ) :
  sunday_visitors = 510 →
  other_day_visitors = 240 →
  month_days = 30 →
  sundays_in_month = 5 →
  (sundays_in_month * sunday_visitors + (month_days - sundays_in_month) * other_day_visitors) / month_days = 285 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l140_14063


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l140_14072

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_number (n : ℕ) : ℕ := 
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem four_digit_number_problem (N : ℕ) (hN : is_four_digit N) :
  let M := reverse_number N
  (N + M = 3333 ∧ N - M = 693) → N = 2013 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l140_14072


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l140_14024

theorem problems_left_to_grade (problems_per_worksheet : ℕ) (total_worksheets : ℕ) (graded_worksheets : ℕ) : 
  problems_per_worksheet = 4 →
  total_worksheets = 16 →
  graded_worksheets = 8 →
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 32 := by
  sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l140_14024


namespace NUMINAMATH_CALUDE_school_students_count_l140_14056

theorem school_students_count : ℕ :=
  let below_eight_percent : ℚ := 20 / 100
  let eight_years_count : ℕ := 12
  let above_eight_ratio : ℚ := 2 / 3
  let total_students : ℕ := 40

  have h1 : ↑eight_years_count + (↑eight_years_count * above_eight_ratio) = (1 - below_eight_percent) * total_students := by sorry

  total_students


end NUMINAMATH_CALUDE_school_students_count_l140_14056


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l140_14080

/-- The lateral surface area of a cone with base radius 2 and slant height 4 is 8π -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 4 → π * r * l = 8 * π :=
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l140_14080


namespace NUMINAMATH_CALUDE_sample_size_proof_l140_14094

theorem sample_size_proof (n : ℕ) (f₁ f₂ f₃ f₄ f₅ f₆ : ℕ) : 
  f₁ + f₂ + f₃ + f₄ + f₅ + f₆ = n →
  f₁ + f₂ + f₃ = 27 →
  ∃ (k : ℕ), f₁ = 2*k ∧ f₂ = 3*k ∧ f₃ = 4*k ∧ f₄ = 6*k ∧ f₅ = 4*k ∧ f₆ = k →
  n = 60 := by
sorry

end NUMINAMATH_CALUDE_sample_size_proof_l140_14094


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l140_14048

theorem sum_and_ratio_to_difference (x y : ℝ) 
  (sum_eq : x + y = 780) 
  (ratio_eq : x / y = 1.25) : 
  x - y = 86 + 2/3 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l140_14048


namespace NUMINAMATH_CALUDE_range_of_a_when_p_or_q_false_l140_14001

def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0

def q (a : ℝ) : Prop :=
  ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a_when_p_or_q_false :
  {a : ℝ | ¬(p a ∨ q a)} = Set.Ioo (-1) 0 ∪ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_p_or_q_false_l140_14001


namespace NUMINAMATH_CALUDE_expression_value_l140_14013

theorem expression_value (n m : ℤ) (h : m = 2 * n^2 + n + 1) :
  8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l140_14013


namespace NUMINAMATH_CALUDE_length_A_l140_14030

def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 7)

def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

def on_line_AC (p : ℝ × ℝ) : Prop :=
  (p.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (p.1 - A.1)

def on_line_BC (p : ℝ × ℝ) : Prop :=
  (p.2 - B.2) * (C.1 - B.1) = (C.2 - B.2) * (p.1 - B.1)

def A' : ℝ × ℝ := sorry
def B' : ℝ × ℝ := sorry

theorem length_A'B'_is_4_sqrt_2 :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧ on_line_AC A' ∧ on_line_BC B' →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_length_A_l140_14030


namespace NUMINAMATH_CALUDE_equilateral_triangle_extension_equality_l140_14002

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the equilateral property
def is_equilateral (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- Define points D and E
variable (D E : EuclideanSpace ℝ (Fin 2))

-- Define the conditions for D and E
def D_on_AC_extension (A C D : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = A + t • (C - A)

def E_on_BC_extension (B C E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∃ s : ℝ, s > 1 ∧ E = B + s • (C - B)

-- Define the equality of BD and DE
def BD_equals_DE (B D E : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist B D = dist D E

-- State the theorem
theorem equilateral_triangle_extension_equality
  (h1 : is_equilateral A B C)
  (h2 : D_on_AC_extension A C D)
  (h3 : E_on_BC_extension B C E)
  (h4 : BD_equals_DE B D E) :
  dist A D = dist C E :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_extension_equality_l140_14002


namespace NUMINAMATH_CALUDE_square_side_length_l140_14041

theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) 
  (h1 : rectangle_length = 9) (h2 : rectangle_width = 16) :
  ∃ (square_side : ℝ), square_side ^ 2 = rectangle_length * rectangle_width ∧ square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l140_14041


namespace NUMINAMATH_CALUDE_equal_cost_messages_l140_14010

/-- Represents the cost of a text messaging plan -/
structure TextPlan where
  costPerMessage : ℚ
  monthlyFee : ℚ

/-- Calculates the total cost for a given number of messages -/
def totalCost (plan : TextPlan) (messages : ℚ) : ℚ :=
  plan.costPerMessage * messages + plan.monthlyFee

theorem equal_cost_messages : 
  let planA : TextPlan := ⟨0.25, 9⟩
  let planB : TextPlan := ⟨0.40, 0⟩
  ∃ (x : ℚ), x = 60 ∧ totalCost planA x = totalCost planB x :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_messages_l140_14010


namespace NUMINAMATH_CALUDE_sum_primes_square_bound_l140_14091

/-- S_n is the sum of the first n prime numbers -/
def S (n : ℕ) : ℕ := sorry

/-- The n-th prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem sum_primes_square_bound :
  ∀ n : ℕ, n > 0 → ∃ m : ℕ, S n ≤ m^2 ∧ m^2 ≤ S (n + 1) :=
sorry

end NUMINAMATH_CALUDE_sum_primes_square_bound_l140_14091


namespace NUMINAMATH_CALUDE_book_page_digits_l140_14059

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) +
  2 * (min n 99 - min n 9) +
  3 * (n - min n 99)

/-- Theorem: The total number of digits used in numbering the pages of a book with 356 pages is 960 -/
theorem book_page_digits :
  totalDigits 356 = 960 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l140_14059


namespace NUMINAMATH_CALUDE_max_product_constrained_sum_l140_14042

theorem max_product_constrained_sum (a b : ℝ) : 
  a > 0 → b > 0 → 5 * a + 8 * b = 80 → ab ≤ 40 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 5 * a₀ + 8 * b₀ = 80 ∧ a₀ * b₀ = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_sum_l140_14042


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l140_14053

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

end NUMINAMATH_CALUDE_specific_trapezoid_area_l140_14053


namespace NUMINAMATH_CALUDE_minimum_red_chips_l140_14067

theorem minimum_red_chips (w b r : ℕ) 
  (blue_white : b ≥ (1/3 : ℚ) * w)
  (blue_red : b ≤ (1/4 : ℚ) * r)
  (white_blue_total : w + b ≥ 75) :
  r ≥ 76 :=
sorry

end NUMINAMATH_CALUDE_minimum_red_chips_l140_14067


namespace NUMINAMATH_CALUDE_divisibility_condition_l140_14069

theorem divisibility_condition (a : ℤ) : 
  0 ≤ a ∧ a < 13 ∧ (13 ∣ 51^2022 + a) → a = 12 := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l140_14069


namespace NUMINAMATH_CALUDE_total_profit_is_89_10_l140_14098

def base_price : ℚ := 12
def day1_sales : ℕ := 3
def day2_sales : ℕ := 4
def day3_sales : ℕ := 5
def day1_cost : ℚ := 4
def day2_cost : ℚ := 5
def day3_cost : ℚ := 2
def extra_money : ℚ := 7
def day3_discount : ℚ := 2
def sales_tax_rate : ℚ := 1/10

def day1_profit : ℚ := (day1_sales * base_price + extra_money - day1_sales * day1_cost) * (1 - sales_tax_rate)
def day2_profit : ℚ := (day2_sales * base_price - day2_sales * day2_cost) * (1 - sales_tax_rate)
def day3_profit : ℚ := (day3_sales * (base_price - day3_discount) - day3_sales * day3_cost) * (1 - sales_tax_rate)

theorem total_profit_is_89_10 : 
  day1_profit + day2_profit + day3_profit = 89.1 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_89_10_l140_14098


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l140_14023

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) ≤ 3 * Real.sqrt 8 ∧
  ∃ x y z : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 7 ∧
    Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 1) + Real.sqrt (3 * z + 1) = 3 * Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l140_14023


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l140_14050

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 15)
  (product_sum_condition : a * b + a * c + b * c = 40) :
  a^3 + b^3 + c^3 - 3*a*b*c = 1575 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l140_14050


namespace NUMINAMATH_CALUDE_congruence_problem_l140_14093

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 19 = 3 → (3 * x + 18) % 19 = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l140_14093


namespace NUMINAMATH_CALUDE_dual_colored_cubes_count_l140_14009

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

end NUMINAMATH_CALUDE_dual_colored_cubes_count_l140_14009


namespace NUMINAMATH_CALUDE_saltwater_volume_l140_14022

/-- Proves that the initial volume of a saltwater solution is 200 gallons, given the conditions stated in the problem. -/
theorem saltwater_volume : ∃ (x : ℝ),
  -- Initial solution is 20% salt by volume
  let initial_salt := 0.2 * x
  -- Volume after evaporation (3/4 of initial volume)
  let volume_after_evap := 0.75 * x
  -- New volume after adding water and salt
  let new_volume := volume_after_evap + 10 + 20
  -- New amount of salt
  let new_salt := initial_salt + 20
  -- The resulting mixture is 33 1/3% salt by volume
  new_salt = (1/3) * new_volume ∧ x = 200 :=
sorry

end NUMINAMATH_CALUDE_saltwater_volume_l140_14022


namespace NUMINAMATH_CALUDE_investment_problem_l140_14055

theorem investment_problem (total investment bonds stocks mutual_funds : ℕ) : 
  total = 220000 ∧ 
  stocks = 5 * bonds ∧ 
  mutual_funds = 2 * stocks ∧ 
  total = bonds + stocks + mutual_funds →
  stocks = 68750 := by
sorry

end NUMINAMATH_CALUDE_investment_problem_l140_14055


namespace NUMINAMATH_CALUDE_max_M_value_l140_14015

/-- Definition of J_k -/
def J (k : ℕ) : ℕ := 10^(k+2) + 128

/-- Definition of M(k) -/
def M (k : ℕ) : ℕ := (J k).factors.count 2

/-- Theorem: The maximum value of M(k) for k > 0 is 8 -/
theorem max_M_value : ∃ k > 0, M k = 8 ∧ ∀ n > 0, M n ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_M_value_l140_14015


namespace NUMINAMATH_CALUDE_smallest_candy_count_l140_14088

theorem smallest_candy_count : ∃ n : ℕ,
  (100 ≤ n ∧ n < 1000) ∧
  (n + 7) % 9 = 0 ∧
  (n - 9) % 6 = 0 ∧
  (∀ m : ℕ, 100 ≤ m ∧ m < n → (m + 7) % 9 ≠ 0 ∨ (m - 9) % 6 ≠ 0) ∧
  n = 137 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l140_14088


namespace NUMINAMATH_CALUDE_no_1999_primes_in_ap_l140_14081

theorem no_1999_primes_in_ap (a d : ℕ) (h : a > 0 ∧ d > 0) :
  (∀ k : ℕ, k < 1999 → a + k * d < 12345 ∧ Nat.Prime (a + k * d)) →
  False :=
sorry

end NUMINAMATH_CALUDE_no_1999_primes_in_ap_l140_14081


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l140_14051

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)  -- The sequence
  (h1 : a 1 = 1024)  -- First term is 1024
  (h2 : a 8 = 125)   -- 8th term is 125
  (h3 : ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, a n = a 1 * r^(n-1))  -- Definition of geometric sequence
  : a 6 = 5^(5/7) * 32 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l140_14051


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l140_14034

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = 5)
  (h_a5 : a 5 = 14) :
  (∀ n : ℕ, a n = 3 * n - 1) ∧
  (∃ n : ℕ, n * (a 1 + a n) / 2 = 155 ∧ n = 10) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l140_14034


namespace NUMINAMATH_CALUDE_harry_seed_purchase_cost_l140_14011

/-- The cost of a garden seed purchase --/
def garden_seed_cost (pumpkin_price tomato_price chili_price : ℚ) 
  (pumpkin_qty tomato_qty chili_qty : ℕ) : ℚ :=
  pumpkin_price * pumpkin_qty + tomato_price * tomato_qty + chili_price * chili_qty

/-- Theorem stating the total cost of Harry's seed purchase --/
theorem harry_seed_purchase_cost : 
  garden_seed_cost 2.5 1.5 0.9 3 4 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_harry_seed_purchase_cost_l140_14011


namespace NUMINAMATH_CALUDE_equation_solution_l140_14028

-- Define the equation
def equation (y : ℝ) : Prop :=
  (15 : ℝ)^(3*2) * (7^4 - 3*2) / 5670 = y

-- State the theorem
theorem equation_solution : 
  ∃ y : ℝ, equation y ∧ abs (y - 4812498.20123) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l140_14028


namespace NUMINAMATH_CALUDE_sum_bound_l140_14016

theorem sum_bound (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : |a - b| + |b - c| + |c - a| = 1) : 
  a + b + c ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_bound_l140_14016


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l140_14000

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + f (x + y) - f x * f y = 0

/-- The theorem stating that there is exactly one function satisfying the equation -/
theorem unique_satisfying_function : ∃! f : ℝ → ℝ, SatisfyingFunction f := by sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l140_14000


namespace NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l140_14064

theorem sum_of_angles_two_triangles (A B C D E F : ℝ) :
  (A + B + C = 180) → (D + E + F = 180) → (A + B + C + D + E + F = 360) := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l140_14064


namespace NUMINAMATH_CALUDE_tower_lights_l140_14071

/-- Represents the number of levels in the tower -/
def levels : ℕ := 7

/-- Represents the total number of lights on the tower -/
def totalLights : ℕ := 381

/-- Represents the common ratio between adjacent levels -/
def ratio : ℕ := 2

/-- Calculates the sum of a geometric sequence -/
def geometricSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem tower_lights :
  ∃ (topLights : ℕ), 
    geometricSum topLights ratio levels = totalLights ∧ 
    topLights = 3 := by
  sorry

end NUMINAMATH_CALUDE_tower_lights_l140_14071


namespace NUMINAMATH_CALUDE_initial_bedbug_count_l140_14029

/-- The number of bedbugs after n days, given an initial population -/
def bedbug_population (initial : ℕ) (days : ℕ) : ℕ :=
  initial * (3 ^ days)

/-- Theorem stating the initial number of bedbugs -/
theorem initial_bedbug_count : ∃ (initial : ℕ), 
  bedbug_population initial 4 = 810 ∧ initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_bedbug_count_l140_14029


namespace NUMINAMATH_CALUDE_seven_dots_max_regions_l140_14047

/-- The maximum number of regions formed by connecting n dots on a circle's circumference --/
def max_regions (n : ℕ) : ℕ :=
  1 + (n.choose 2) + (n.choose 4)

/-- Theorem: For 7 dots on a circle's circumference, the maximum number of regions is 57 --/
theorem seven_dots_max_regions :
  max_regions 7 = 57 := by
  sorry

end NUMINAMATH_CALUDE_seven_dots_max_regions_l140_14047


namespace NUMINAMATH_CALUDE_student_response_change_difference_l140_14020

/-- Represents the percentages of student responses --/
structure ResponsePercentages :=
  (yes : ℝ)
  (no : ℝ)
  (undecided : ℝ)

/-- The problem statement --/
theorem student_response_change_difference 
  (initial : ResponsePercentages)
  (final : ResponsePercentages)
  (h_initial_sum : initial.yes + initial.no + initial.undecided = 100)
  (h_final_sum : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 40)
  (h_initial_undecided : initial.undecided = 20)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 30)
  (h_final_undecided : final.undecided = 10) :
  ∃ (min_change max_change : ℝ),
    (∀ (change : ℝ), min_change ≤ change ∧ change ≤ max_change) ∧
    max_change - min_change = 40 :=
sorry

end NUMINAMATH_CALUDE_student_response_change_difference_l140_14020


namespace NUMINAMATH_CALUDE_sequence_recurrence_problem_l140_14039

/-- Given a sequence of positive real numbers {a_n} (n ≥ 0) satisfying the recurrence relation
    a_n = a_{n-1} / (m * a_{n-2}) for n ≥ 2, where m is a real parameter,
    prove that if a_2009 = a_0 / a_1, then m = 1. -/
theorem sequence_recurrence_problem (a : ℕ → ℝ) (m : ℝ) 
    (h_positive : ∀ n, a n > 0)
    (h_recurrence : ∀ n ≥ 2, a n = a (n-1) / (m * a (n-2)))
    (h_equality : a 2009 = a 0 / a 1) :
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_recurrence_problem_l140_14039


namespace NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l140_14079

/-- Given a point C with coordinates (3, y) and its reflection D over the x-axis,
    the sum of all four coordinates of C and D is 6. -/
theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)
  C.1 + C.2 + D.1 + D.2 = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l140_14079


namespace NUMINAMATH_CALUDE_solve_for_m_l140_14031

theorem solve_for_m (x m : ℝ) : 2 * x + m = 1 → x = -1 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l140_14031


namespace NUMINAMATH_CALUDE_inequality_solution_set_l140_14070

theorem inequality_solution_set (x y : ℝ) : 
  (∀ y > 0, (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y)) / (x + y) > 3 * x^2 * y) ↔ 
  x > 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l140_14070


namespace NUMINAMATH_CALUDE_runners_meeting_time_l140_14073

/-- 
Given two runners, Danny and Steve, running towards each other from their respective houses:
* Danny's time to reach Steve's house is t minutes
* Steve's time to reach Danny's house is 2t minutes
* Steve takes 13.5 minutes longer to reach the halfway point than Danny
Prove that t = 27 minutes
-/
theorem runners_meeting_time (t : ℝ) 
  (h1 : t > 0) -- Danny's time is positive
  (h2 : 2 * t - t / 2 = 13.5) -- Difference in time to reach halfway point
  : t = 27 := by
  sorry

end NUMINAMATH_CALUDE_runners_meeting_time_l140_14073


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l140_14068

/-- Given an arithmetic sequence where the first three terms are 3, 16, and 29,
    prove that the 15th term is 185. -/
theorem arithmetic_sequence_15th_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
    a 1 = 3 →                            -- first term
    a 2 = 16 →                           -- second term
    a 3 = 29 →                           -- third term
    a 15 = 185 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l140_14068


namespace NUMINAMATH_CALUDE_m_range_theorem_l140_14095

/-- Proposition p: x^2 - mx + 1 = 0 has no real solutions -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0

/-- Proposition q: x^2/m + y^2 = 1 has its foci on the x-axis -/
def q (m : ℝ) : Prop := m > 1

/-- The range of real values for m given the conditions -/
def m_range (m : ℝ) : Prop := (-2 < m ∧ m ≤ 1) ∨ m ≥ 2

/-- Theorem stating the range of m given the conditions -/
theorem m_range_theorem (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m_range m := by
  sorry

end NUMINAMATH_CALUDE_m_range_theorem_l140_14095


namespace NUMINAMATH_CALUDE_reggie_shopping_spree_l140_14018

def initial_amount : ℕ := 150
def num_books : ℕ := 5
def book_price : ℕ := 12
def game_price : ℕ := 45
def bottle_price : ℕ := 13
def snack_price : ℕ := 7

theorem reggie_shopping_spree :
  initial_amount - (num_books * book_price + game_price + bottle_price + snack_price) = 25 := by
  sorry

end NUMINAMATH_CALUDE_reggie_shopping_spree_l140_14018


namespace NUMINAMATH_CALUDE_opposite_direction_speed_l140_14049

/-- 
Given two people traveling in opposite directions for 1.5 hours, 
with one person traveling at 5 miles per hour and ending up 19.5 miles apart, 
prove that the other person's speed must be 8 miles per hour.
-/
theorem opposite_direction_speed 
  (time : ℝ) 
  (distance : ℝ) 
  (speed_peter : ℝ) 
  (speed_juan : ℝ) : 
  time = 1.5 → 
  distance = 19.5 → 
  speed_peter = 5 → 
  distance = (speed_juan + speed_peter) * time → 
  speed_juan = 8 := by
  sorry

end NUMINAMATH_CALUDE_opposite_direction_speed_l140_14049


namespace NUMINAMATH_CALUDE_numbers_below_nine_and_twenty_four_are_composite_l140_14035

def below_nine (k : ℕ) : ℕ := 4 * k^2 + 5 * k + 1

def below_twenty_four (k : ℕ) : ℕ := 4 * k^2 + 5 * k

theorem numbers_below_nine_and_twenty_four_are_composite :
  (∀ k : ℕ, k ≥ 1 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ below_nine k = a * b) ∧
  (∀ k : ℕ, k ≥ 2 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ below_twenty_four k = a * b) :=
sorry

end NUMINAMATH_CALUDE_numbers_below_nine_and_twenty_four_are_composite_l140_14035


namespace NUMINAMATH_CALUDE_salt_solution_weight_salt_solution_weight_proof_l140_14032

theorem salt_solution_weight (initial_concentration : Real) 
                             (final_concentration : Real) 
                             (added_salt : Real) 
                             (initial_weight : Real) : Prop :=
  initial_concentration = 0.10 ∧
  final_concentration = 0.20 ∧
  added_salt = 12.5 ∧
  initial_weight * initial_concentration + added_salt = 
    (initial_weight + added_salt) * final_concentration →
  initial_weight = 100

-- Proof
theorem salt_solution_weight_proof :
  salt_solution_weight 0.10 0.20 12.5 100 := by
  sorry

end NUMINAMATH_CALUDE_salt_solution_weight_salt_solution_weight_proof_l140_14032


namespace NUMINAMATH_CALUDE_parallel_implies_n_eq_two_transform_implies_m_n_eq_neg_one_l140_14097

-- Define points A and B in the Cartesian coordinate system
def A (m : ℝ) : ℝ × ℝ := (3, 2*m - 1)
def B (n : ℝ) : ℝ × ℝ := (n + 1, -1)

-- Define the condition that A and B are not coincident
def not_coincident (m n : ℝ) : Prop := A m ≠ B n

-- Define what it means for AB to be parallel to y-axis
def parallel_to_y_axis (m n : ℝ) : Prop := (A m).1 = (B n).1

-- Define the transformation of A to B
def transform_A_to_B (m n : ℝ) : Prop :=
  (A m).1 - 3 = (B n).1 ∧ (A m).2 + 2 = (B n).2

-- Theorem 1
theorem parallel_implies_n_eq_two (m n : ℝ) 
  (h1 : not_coincident m n) (h2 : parallel_to_y_axis m n) : n = 2 := by sorry

-- Theorem 2
theorem transform_implies_m_n_eq_neg_one (m n : ℝ) 
  (h1 : not_coincident m n) (h2 : transform_A_to_B m n) : m = -1 ∧ n = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_n_eq_two_transform_implies_m_n_eq_neg_one_l140_14097


namespace NUMINAMATH_CALUDE_projection_result_l140_14086

def v1 : ℝ × ℝ := (3, 2)
def v2 : ℝ × ℝ := (2, 5)

theorem projection_result (u : ℝ × ℝ) (q : ℝ × ℝ) 
  (h1 : ∃ (k1 : ℝ), q = k1 • u ∧ (v1 - q) • u = 0)
  (h2 : ∃ (k2 : ℝ), q = k2 • u ∧ (v2 - q) • u = 0) :
  q = (33/10, 11/10) :=
sorry

end NUMINAMATH_CALUDE_projection_result_l140_14086


namespace NUMINAMATH_CALUDE_inequality_proof_l140_14025

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ (1/2) * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l140_14025


namespace NUMINAMATH_CALUDE_sum_of_ratios_ge_six_l140_14078

theorem sum_of_ratios_ge_six (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / y + y / z + z / x + x / z + z / y + y / x ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ratios_ge_six_l140_14078


namespace NUMINAMATH_CALUDE_unique_abc_sum_l140_14019

theorem unique_abc_sum (x : ℝ) : 
  x = Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2) →
  ∃! (a b c : ℕ+), 
    x^80 = 2*x^78 + 8*x^76 + 9*x^74 - x^40 + (a : ℝ)*x^36 + (b : ℝ)*x^34 + (c : ℝ)*x^30 ∧
    a + b + c = 151 := by
  sorry

end NUMINAMATH_CALUDE_unique_abc_sum_l140_14019


namespace NUMINAMATH_CALUDE_P_equals_Q_l140_14045

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m*x^2 + 4*m*x - 4 < 0}

theorem P_equals_Q : P = Q := by sorry

end NUMINAMATH_CALUDE_P_equals_Q_l140_14045


namespace NUMINAMATH_CALUDE_q_polynomial_form_l140_14060

/-- The function q satisfying the given equation -/
noncomputable def q : ℝ → ℝ := fun x => 4*x^4 + 16*x^3 + 36*x^2 + 10*x + 4 - (2*x^6 + 5*x^4 + 11*x^2 + 6*x)

/-- Theorem stating that q has the specified polynomial form -/
theorem q_polynomial_form : q = fun x => -2*x^6 - x^4 + 16*x^3 + 25*x^2 + 4*x + 4 := by sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l140_14060


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l140_14026

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/5
  let r : ℚ := 2/5
  let n : ℕ := 8
  geometric_sum a r n = 390369/1171875 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l140_14026


namespace NUMINAMATH_CALUDE_fraction_evaluation_l140_14004

theorem fraction_evaluation : (3^4 - 3^3) / (3^(-2) + 3^(-1)) = 121.5 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l140_14004


namespace NUMINAMATH_CALUDE_distribution_plans_count_l140_14043

/-- The number of ways to distribute 3 volunteer teachers among 6 schools, with at most 2 teachers per school -/
def distribution_plans : ℕ := 210

/-- The number of schools -/
def num_schools : ℕ := 6

/-- The number of volunteer teachers -/
def num_teachers : ℕ := 3

/-- The maximum number of teachers allowed per school -/
def max_teachers_per_school : ℕ := 2

theorem distribution_plans_count :
  distribution_plans = 210 :=
sorry

end NUMINAMATH_CALUDE_distribution_plans_count_l140_14043


namespace NUMINAMATH_CALUDE_function_difference_implies_a_range_l140_14033

open Real

theorem function_difference_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ → 
    (a * log x₁ + x₁^2) - (a * log x₂ + x₂^2) > 2) →
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_function_difference_implies_a_range_l140_14033


namespace NUMINAMATH_CALUDE_i_in_first_quadrant_l140_14058

/-- The complex number i corresponds to a point in the first quadrant of the complex plane. -/
theorem i_in_first_quadrant : Complex.I.re = 0 ∧ Complex.I.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_i_in_first_quadrant_l140_14058


namespace NUMINAMATH_CALUDE_coin_problem_l140_14012

def penny : ℕ := 1
def nickel : ℕ := 5
def dime : ℕ := 10
def quarter : ℕ := 25

theorem coin_problem (p d n q : ℕ) : 
  p + n + d + q = 12 →  -- Total number of coins
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 →  -- At least one of each type
  q = 2 * d →  -- Twice as many quarters as dimes
  p * penny + n * nickel + d * dime + q * quarter = 128 →  -- Total value in cents
  n = 3 := by sorry

end NUMINAMATH_CALUDE_coin_problem_l140_14012


namespace NUMINAMATH_CALUDE_x_value_proof_l140_14005

theorem x_value_proof (x : ℝ) : 
  3.5 * ((x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 → x = 3.6 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l140_14005


namespace NUMINAMATH_CALUDE_sarah_copies_360_pages_l140_14057

/-- The number of copies per person -/
def copies_per_person : ℕ := 2

/-- The number of people in the meeting -/
def number_of_people : ℕ := 9

/-- The number of pages in each contract -/
def pages_per_contract : ℕ := 20

/-- The total number of pages Sarah will copy -/
def total_pages : ℕ := copies_per_person * number_of_people * pages_per_contract

theorem sarah_copies_360_pages : total_pages = 360 := by
  sorry

end NUMINAMATH_CALUDE_sarah_copies_360_pages_l140_14057


namespace NUMINAMATH_CALUDE_circle_radius_l140_14021

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 1)^2 = 16

-- Define the ellipse equation (not used in the proof, but included for completeness)
def ellipse_equation (x y : ℝ) : Prop :=
  (x - 2)^2 / 25 + (y - 1)^2 / 9 = 1

-- Theorem: The radius of the circle is 4
theorem circle_radius : ∃ (r : ℝ), r = 4 ∧ ∀ (x y : ℝ), circle_equation x y ↔ (x - 2)^2 + (y - 1)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l140_14021


namespace NUMINAMATH_CALUDE_no_square_from_square_cut_l140_14062

-- Define a square
def Square (s : ℝ) := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

-- Define a straight cut
def StraightCut (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Theorem: It's impossible to create a square from a larger square by a single straight cut
theorem no_square_from_square_cut (s₁ s₂ : ℝ) (h₁ : 0 < s₁) (h₂ : 0 < s₂) (h₃ : s₂ < s₁) :
  ¬∃ (a b c : ℝ), (Square s₁ ∩ StraightCut a b c).Nonempty ∧ 
    (Square s₂).Subset (Square s₁ ∩ StraightCut a b c) :=
sorry

end NUMINAMATH_CALUDE_no_square_from_square_cut_l140_14062


namespace NUMINAMATH_CALUDE_intersection_radius_l140_14052

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the circle in the xz-plane -/
  xz_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in the xz-plane -/
  xz_radius : ℝ
  /-- Center of the circle in the zy-plane -/
  zy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in the zy-plane -/
  zy_radius : ℝ

/-- The radius of the circle where the sphere intersects the zy-plane is 3 -/
theorem intersection_radius (s : IntersectingSphere) 
  (h1 : s.xz_center = (3, 0, 5))
  (h2 : s.xz_radius = 2)
  (h3 : s.zy_center = (0, 3, -4)) :
  s.zy_radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_radius_l140_14052


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l140_14089

/-- Two hyperbolas have the same asymptotes if M = 18 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/32 - x^2/M = 1) →
  M = 18 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l140_14089


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l140_14074

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℕ) 
  (h1 : english = 86)
  (h2 : mathematics = 85)
  (h3 : physics = 82)
  (h4 : biology = 85)
  (h5 : average = 85)
  (h6 : (english + mathematics + physics + biology + chemistry) / 5 = average) :
  chemistry = 87 := by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l140_14074
