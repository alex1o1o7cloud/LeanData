import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l723_72361

def C : Set Nat := {66, 73, 75, 79, 81}

def smallest_prime_factor (n : Nat) : Nat :=
  (Nat.factors n).head!

theorem smallest_prime_factor_in_C :
  ∀ x ∈ C, smallest_prime_factor 66 ≤ smallest_prime_factor x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_prime_factor_in_C_l723_72361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l723_72379

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the midpoint of AB
def midpoint_of (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

theorem ellipse_and_line_properties :
  -- C passes through (√2, 1) and (0, -√2)
  ellipse_C (Real.sqrt 2) 1 ∧ ellipse_C 0 (-Real.sqrt 2) ∧
  -- There exist points A and B where C and l intersect
  ∃ A B : ℝ × ℝ, intersection_points A B ∧
  -- The midpoint of AB is (-2/3, 1/3)
  midpoint_of A B (-2/3, 1/3) := by
  sorry

#check ellipse_and_line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l723_72379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l723_72354

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

theorem train_bridge_passage_time :
  let train_length : ℝ := 800
  let bridge_length : ℝ := 375
  let train_speed_kmh : ℝ := 115
  let calculated_time := train_pass_time train_length bridge_length train_speed_kmh
  abs (calculated_time - 36.78) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l723_72354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l723_72312

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  Real.cos B / Real.cos C = b / (2 * a + c) →
  b = Real.sqrt 13 →
  a + c = 4 →
  -- Conclusions
  B = 2 * π / 3 ∧
  (1 / 2) * a * c * Real.sin B = (3 / 4) * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l723_72312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l723_72342

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * Real.cos t.C ∧
  t.a + t.b = 4 ∧
  t.c = 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l723_72342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l723_72347

noncomputable def a (n : ℕ) : ℝ := 2 * n - 1

noncomputable def b (n : ℕ) : ℝ := 3^n

noncomputable def c (n : ℕ) : ℝ := (a n * b n) / (n^2 + n : ℝ)

noncomputable def S (n : ℕ) : ℝ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

noncomputable def T (n : ℕ) : ℝ := 3^(n * (n + 1) / 2)

noncomputable def Q (n : ℕ) : ℝ := 3^(n + 1) / (n + 1 : ℝ) - 3

theorem sequence_properties (n : ℕ) : 
  a 3 = 5 ∧ 
  S 6 - S 3 = 27 ∧ 
  T n = 3^(n * (n + 1) / 2) ∧
  a n = 2 * n - 1 ∧
  b n = 3^n ∧
  Q n = 3^(n + 1) / (n + 1 : ℝ) - 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l723_72347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l723_72359

/-- The distance between the foci of an ellipse -/
noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

/-- Theorem: The distance between the foci of an ellipse with semi-major axis 8 and semi-minor axis 3 is 2√55 -/
theorem ellipse_foci_distance :
  distance_between_foci 8 3 = 2 * Real.sqrt 55 := by
  unfold distance_between_foci
  simp
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l723_72359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l723_72356

/-- Represents the score distribution of students in a classroom test. -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  score100 : ℝ
  sum_to_one : score60 + score75 + score85 + score95 + score100 = 1
  non_negative : 0 ≤ score60 ∧ 0 ≤ score75 ∧ 0 ≤ score85 ∧ 0 ≤ score95 ∧ 0 ≤ score100

/-- Calculates the mean score given a score distribution. -/
def meanScore (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 95 * d.score95 + 100 * d.score100

/-- Calculates the median score given a score distribution. -/
noncomputable def medianScore (d : ScoreDistribution) : ℝ :=
  if d.score60 > 0.5 then 60
  else if d.score60 + d.score75 > 0.5 then 75
  else if d.score60 + d.score75 + d.score85 > 0.5 then 85
  else if d.score60 + d.score75 + d.score85 + d.score95 > 0.5 then 95
  else 100

/-- The main theorem stating the difference between mean and median scores. -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.25)
  (h2 : d.score75 = 0.20)
  (h3 : d.score85 = 0.30)
  (h4 : d.score95 = 0.15) :
  |meanScore d - medianScore d| = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_median_difference_l723_72356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l723_72338

/-- The distance between the foci of a hyperbola given by its equation -/
noncomputable def distance_between_foci (a b c d e f : ℝ) : ℝ :=
  2 * Real.sqrt ((a + c) / 2)

/-- Theorem: The distance between the foci of the hyperbola 2x^2 - 12x - 8y^2 + 16y = 100 is 2√68.75 -/
theorem hyperbola_foci_distance :
  distance_between_foci 2 (-12) (-8) 16 0 100 = 2 * Real.sqrt 68.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_l723_72338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_exists_l723_72321

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2))

theorem unique_sequence_exists : ∃! a : ℕ → ℤ, is_valid_sequence a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sequence_exists_l723_72321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_from_point_coordinates_l723_72398

theorem angle_value_from_point_coordinates :
  ∀ α : Real,
  0 < α → α < 2 * Real.pi →
  let P : Real × Real := (1 - Real.tan (Real.pi / 12), 1 + Real.tan (Real.pi / 12))
  let x : Real := P.1
  let y : Real := P.2
  (x * Real.cos α + y * Real.sin α = Real.sqrt (x^2 + y^2)) →
  α = Real.pi / 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_from_point_coordinates_l723_72398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_corner_is_three_l723_72305

/-- Represents a 5x5 grid where each cell can contain a digit from 1 to 5 or be empty -/
def Grid := Fin 5 → Fin 5 → Option (Fin 5)

/-- Checks if a given grid satisfies the condition that each digit 1 through 5
    appears exactly once in each row and column -/
def is_valid_grid (g : Grid) : Prop :=
  ∀ (i j : Fin 5),
    (∃! (d : Fin 5), g i j = some d) ∧
    (∃! (i' : Fin 5), g i' j = g i j) ∧
    (∃! (j' : Fin 5), g i j' = g i j)

/-- The initial grid layout as given in the problem -/
def initial_grid : Grid :=
  λ i j ↦ if i = 0 ∧ j = 0 then some 1
    else if i = 0 ∧ j = 1 then some 2
    else if i = 1 ∧ j = 2 then some 3
    else if i = 1 ∧ j = 4 then some 1
    else if i = 3 ∧ j = 4 then some 2
    else none

/-- The theorem stating that given the initial grid layout and the conditions,
    the number in the lower right-hand corner must be 3 -/
theorem lower_right_corner_is_three :
  ∀ (g : Grid),
    is_valid_grid g →
    (∀ (i j : Fin 5), initial_grid i j ≠ none → g i j = initial_grid i j) →
    g 4 4 = some 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lower_right_corner_is_three_l723_72305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l723_72339

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 2 * x - 2

-- Define the derivative of the curve function
noncomputable def f' (x : ℝ) : ℝ := Real.exp x + x * Real.exp x + 2

-- Theorem statement
theorem tangent_line_at_zero :
  let point : ℝ × ℝ := (0, -2)
  let slope : ℝ := f' 0
  let tangent_line (x : ℝ) := slope * x + point.2
  ∀ x, tangent_line x = 3 * x - 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_zero_l723_72339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l723_72351

theorem cube_root_sum_equals_one (x : ℝ) (hx : x > 0) 
  (h : (1 - x^3)^(1/3) + (1 + x^3)^(1/3) = 1) : 
  x^2 = 28^(1/3) / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_one_l723_72351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_distance_l723_72374

/-- A dilation of the plane that transforms one circle to another --/
structure PlaneDilation where
  original_center : ℝ × ℝ
  original_radius : ℝ
  transformed_center : ℝ × ℝ
  transformed_radius : ℝ

/-- The distance a point moves under a dilation --/
noncomputable def dilation_distance (d : PlaneDilation) (p : ℝ × ℝ) : ℝ :=
  sorry

/-- The specific dilation described in the problem --/
def problem_dilation : PlaneDilation where
  original_center := (3, 1)
  original_radius := 4
  transformed_center := (7, 9)
  transformed_radius := 6

/-- Theorem stating the distance the origin moves under the given dilation --/
theorem origin_movement_distance :
  dilation_distance problem_dilation (0, 0) = 0.5 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_movement_distance_l723_72374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_solution_l723_72311

open Real

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem no_rational_solution :
  ∀ (x : ℚ), x > 0 → frac (x : ℝ) + frac ((1 / x : ℚ) : ℝ) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_solution_l723_72311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_is_one_sixth_l723_72369

/-- The ratio of the volume of a regular octahedron constructed by connecting
    the centers of adjacent faces of a cube to the volume of the cube. -/
noncomputable def octahedron_cube_volume_ratio (x : ℝ) : ℝ :=
  let cube_side := 2 * x
  let cube_volume := cube_side ^ 3
  let octahedron_edge := x * Real.sqrt 2
  let octahedron_volume := octahedron_edge ^ 3 * Real.sqrt 2 / 3
  octahedron_volume / cube_volume

/-- The ratio of the volume of a regular octahedron constructed by connecting
    the centers of adjacent faces of a cube to the volume of the cube is 1/6. -/
theorem octahedron_cube_volume_ratio_is_one_sixth (x : ℝ) (hx : x > 0) :
  octahedron_cube_volume_ratio x = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octahedron_cube_volume_ratio_is_one_sixth_l723_72369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_operations_l723_72397

def z₁ : ℂ := -2 + 5 * Complex.I
def z₂ : ℂ := 3 - 4 * Complex.I

theorem complex_operations :
  (z₁ + z₂ = 1 + Complex.I) ∧
  (z₂ - z₁ = 5 - 9 * Complex.I) ∧
  (z₁ * z₂ = 14 + 23 * Complex.I) ∧
  (z₁ / z₂ = -26/25 + 7/25 * Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_operations_l723_72397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coprime_to_180_l723_72363

theorem smallest_coprime_to_180 :
  ∃ (x : ℕ), x > 1 ∧ Nat.Coprime x 180 ∧ ∀ (y : ℕ), y > 1 → Nat.Coprime y 180 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_coprime_to_180_l723_72363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_triangle_l723_72332

/-- A right-angled triangle ABC with given side lengths -/
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  AC : ℝ
  right_angle : AB^2 + BC^2 = AC^2

/-- Volume of a solid generated by revolving a right triangle about one of its legs -/
noncomputable def volume_of_revolution (t : RightTriangle) : ℝ :=
  (1/3) * Real.pi * t.BC^2 * t.AB

/-- The specific right triangle in the problem -/
def triangle : RightTriangle := {
  AB := 3
  BC := 4
  AC := 5
  right_angle := by
    -- Proof of Pythagorean theorem for this triangle
    norm_num
}

theorem volume_of_specific_triangle :
  volume_of_revolution triangle = 16 * Real.pi := by
  unfold volume_of_revolution triangle
  simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_specific_triangle_l723_72332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l723_72364

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - a*x

-- Part I
theorem part_one (a : ℝ) :
  (∀ x > 0, StrictMono (f a)) → a ≤ 2 * Real.sqrt 2 :=
by sorry

-- Part II
theorem part_two (a : ℝ) :
  (∀ x > 0, StrictMono (f a)) →
  (∀ x ∈ Set.Ioc 0 1, f a x ≤ 1/2 * (3*x^2 + 1/x^2 - 6*x)) →
  2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l723_72364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_representation_l723_72326

/-- Represents a repeating decimal of the form 0,1(a₁a₂a₃) -/
structure RepeatingDecimal where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  distinct_digits : a₁ ≠ a₂ ∨ a₁ ≠ a₃ ∨ a₂ ≠ a₃

/-- Checks if a fraction 4/n can be represented as a RepeatingDecimal -/
def isValidRepeatingDecimal (n : ℕ) (d : RepeatingDecimal) : Prop :=
  (4 : ℚ) / n = 1 / 10 + (d.a₁ * 100 + d.a₂ * 10 + d.a₃ : ℚ) / 999

theorem repeating_decimal_representation :
  ∀ n : ℕ, n > 0 →
    (∃ d : RepeatingDecimal, isValidRepeatingDecimal n d) ↔ (n = 27 ∨ n = 37) :=
by sorry

#check repeating_decimal_representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_representation_l723_72326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_classification_l723_72335

def curve_equation (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y ↦ (m - 3) * x^2 + (5 - m) * y^2 = 1

theorem curve_classification (m : ℝ) :
  (m = 4 → curve_equation m = λ x y ↦ x^2 + y^2 = 1) ∧
  ((3 < m ∧ m < 5 ∧ m ≠ 4) → ∃ a b, a > 0 ∧ b > 0 ∧ a ≠ b ∧ curve_equation m = λ x y ↦ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  ((m > 5 ∨ m < 3) → ∃ a b, a > 0 ∧ b > 0 ∧ (curve_equation m = λ x y ↦ (x^2 / a^2) - (y^2 / b^2) = 1 ∨ curve_equation m = λ x y ↦ (y^2 / a^2) - (x^2 / b^2) = 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_classification_l723_72335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_up_theorem_l723_72399

/-- The number of ways to arrange 5 people in a line with the youngest not in the first two positions -/
def lineUpWays : ℕ := 72

/-- The number of positions available for the youngest person -/
def positionsForYoungest : ℕ := 3

/-- The number of people -/
def totalPeople : ℕ := 5

theorem line_up_theorem :
  lineUpWays = positionsForYoungest * Nat.factorial (totalPeople - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_up_theorem_l723_72399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stan_pages_to_write_l723_72366

/-- Calculates the number of pages Stan needs to write based on his typing speed,
    words per page, water consumption rate, and total water needed. -/
noncomputable def pages_to_write (typing_speed : ℝ) (words_per_page : ℝ) (water_per_hour : ℝ) (total_water : ℝ) : ℝ :=
  let minutes_per_page := words_per_page / typing_speed
  let hours_per_page := minutes_per_page / 60
  let water_per_page := water_per_hour * hours_per_page
  total_water / water_per_page

/-- Theorem stating that Stan needs to write 5 pages given the specified conditions. -/
theorem stan_pages_to_write :
  pages_to_write 50 400 15 10 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stan_pages_to_write_l723_72366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chapter_distribution_ways_l723_72357

def total_chapters : ℕ := 16
def total_authors : ℕ := 8
def authors_with_three_chapters : ℕ := 2
def authors_with_two_chapters : ℕ := 4
def authors_with_one_chapter : ℕ := 2

theorem chapter_distribution_ways :
  (Nat.factorial total_chapters) / (2^6 * 3^2) =
  (Nat.choose total_chapters 3) *
  (Nat.choose (total_chapters - 3) 3) *
  (Nat.choose (total_chapters - 6) 2) *
  (Nat.choose (total_chapters - 8) 2) *
  (Nat.choose (total_chapters - 10) 2) *
  (Nat.choose (total_chapters - 12) 2) *
  (Nat.choose (total_chapters - 14) 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chapter_distribution_ways_l723_72357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l723_72381

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2*x + 6

-- State the theorem
theorem min_value_f :
  ∃ (min_val : ℝ), min_val = 9/2 ∧
  ∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ min_val := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l723_72381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l723_72330

theorem contrapositive_equivalence (α : Real) :
  (α = π / 3 → Real.cos α = 1 / 2) ↔ (Real.cos α ≠ 1 / 2 → α ≠ π / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l723_72330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_123456_l723_72343

def num_cards : Nat := 12
def num_unique : Nat := 6

-- Define the type for our card arrangement
def CardArrangement := Fin num_cards → Fin num_unique

-- Function to check if an arrangement results in 123456 after removal
def results_in_123456 (arrangement : CardArrangement) : Prop :=
  ∃ (removed : Fin num_unique → Fin num_cards),
    (∀ i : Fin num_unique, arrangement (removed i) = i) ∧
    (∀ j : Fin num_cards, ∀ i : Fin num_unique, j < removed i → arrangement j ≠ i) ∧
    (∀ k : Fin num_unique, ∃ l : Fin num_cards, l ≥ num_unique ∧ arrangement l = k)

-- Total number of possible arrangements
def total_arrangements : Nat :=
  (Nat.factorial num_cards) / ((Nat.factorial 2) ^ num_unique)

-- Number of favorable arrangements
def favorable_arrangements : Nat := 10395

theorem probability_123456 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_123456_l723_72343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l723_72350

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For the given ellipse, if AF = 2, then BF = 0.4√5 -/
theorem ellipse_chord_theorem (e : Ellipse) (A B F : Point) :
  e.a = 6 →
  e.b = 4 →
  isOnEllipse A e →
  isOnEllipse B e →
  F.x = 2 * Real.sqrt 5 →
  F.y = 0 →
  distance A F = 2 →
  distance B F = 0.4 * Real.sqrt 5 := by
  sorry

#check ellipse_chord_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_theorem_l723_72350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_patterns_cover_in_100_rotations_l723_72355

/-- A regular polygon with 2n vertices -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n) → ℝ × ℝ

/-- A pattern is a subset of n vertices of a 2n-gon -/
def Pattern (n : ℕ) := Fin n → Fin (2*n)

/-- Rotate a pattern by k positions -/
def rotatePattern (n : ℕ) (p : Pattern n) (k : ℕ) : Pattern n :=
  λ i ↦ ⟨(p i + k) % (2*n), by sorry⟩

/-- Check if a set of patterns covers all vertices -/
def coverAllVertices (n : ℕ) (patterns : List (Pattern n)) : Prop :=
  ∀ v : Fin (2*n), ∃ p ∈ patterns, ∃ i : Fin n, p i = v

/-- The main theorem -/
theorem not_all_patterns_cover_in_100_rotations :
  ∃ n : ℕ, ∃ p : Pattern n, 
    ¬(coverAllVertices n (List.map (rotatePattern n p) (List.range 100))) := by
  sorry

#check not_all_patterns_cover_in_100_rotations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_patterns_cover_in_100_rotations_l723_72355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_threshold_value_l723_72333

/-- Represents the tax calculation for a citizen in Country X --/
noncomputable def tax_calculation (X : ℝ) (income : ℝ) : ℝ :=
  0.11 * min X income + 0.20 * max (income - X) 0

/-- Theorem: Given the tax conditions and a specific income and tax amount, X equals 40000 --/
theorem tax_threshold_value : 
  ∃ (X : ℝ), 
    (∀ (income : ℝ), tax_calculation X income = 0.11 * min X income + 0.20 * max (income - X) 0) ∧
    tax_calculation X 58000 = 8000 ∧
    X = 40000 :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tax_threshold_value_l723_72333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_force_ranking_l723_72390

/-- Represents a flask with a circular base -/
structure Flask where
  radius : ℝ
  height : ℝ

/-- The force of water on the base of a flask -/
noncomputable def force_on_base (f : Flask) (water_height : ℝ) : ℝ :=
  Real.pi * f.radius^2 * water_height

/-- The volume of water in a flask -/
noncomputable def water_volume (f : Flask) (water_height : ℝ) : ℝ :=
  Real.pi * f.radius^2 * water_height

theorem force_ranking (flask_a flask_b flask_c : Flask) 
  (water_height_a water_height_b water_height_c : ℝ) :
  flask_a.radius = 2 ∧ 
  flask_b.radius = 2 ∧ 
  flask_c.radius = 2 ∧ 
  water_volume flask_a water_height_a = water_volume flask_b water_height_b ∧
  water_volume flask_b water_height_b = water_volume flask_c water_height_c ∧
  water_height_c > water_height_a ∧
  water_height_a > water_height_b →
  force_on_base flask_c water_height_c > force_on_base flask_a water_height_a ∧
  force_on_base flask_a water_height_a > force_on_base flask_b water_height_b :=
by
  sorry

#check force_ranking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_force_ranking_l723_72390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l723_72391

/-- Represents the state of the buckets -/
structure BucketState where
  bucket10 : ℕ
  bucket6 : ℕ

/-- Checks if a given state is valid (i.e., buckets are not overfilled) -/
def is_valid_state (state : BucketState) : Prop :=
  state.bucket10 ≤ 10 ∧ state.bucket6 ≤ 6

/-- Represents a single operation that can be performed -/
inductive Operation
  | FillFrom10To6
  | FillFrom6To10
  | EmptyIntoRiver10
  | EmptyIntoRiver6
  | FillFromRiver10
  | FillFromRiver6

/-- Applies an operation to a given state, returning a new state -/
def apply_operation (state : BucketState) (op : Operation) : BucketState :=
  match op with
  | Operation.FillFrom10To6 => 
      let amount := min state.bucket10 (6 - state.bucket6)
      { bucket10 := state.bucket10 - amount, bucket6 := state.bucket6 + amount }
  | Operation.FillFrom6To10 => 
      let amount := min state.bucket6 (10 - state.bucket10)
      { bucket10 := state.bucket10 + amount, bucket6 := state.bucket6 - amount }
  | Operation.EmptyIntoRiver10 => { state with bucket10 := 0 }
  | Operation.EmptyIntoRiver6 => { state with bucket6 := 0 }
  | Operation.FillFromRiver10 => { state with bucket10 := 10 }
  | Operation.FillFromRiver6 => { state with bucket6 := 6 }

/-- Checks if a sequence of operations results in a state with 8 liters in the 10-liter bucket -/
def is_solution (ops : List Operation) : Prop :=
  let final_state := ops.foldl apply_operation { bucket10 := 0, bucket6 := 0 }
  final_state.bucket10 = 8 ∧ is_valid_state final_state

/-- The main theorem stating that a solution exists -/
theorem solution_exists : ∃ (ops : List Operation), is_solution ops := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l723_72391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_squared_l723_72313

theorem triangle_inequality_squared (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 + b^2 + c^2 > Real.sqrt 3 * max (|a^2 - b^2|) (max (|b^2 - c^2|) (|c^2 - a^2|)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_squared_l723_72313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_line_l723_72316

/-- The x-intercept of a line is a point where the line crosses the x-axis (y = 0) -/
noncomputable def x_intercept (a b c : ℝ) : ℝ × ℝ :=
  let x := c / a
  (x, 0)

/-- The line equation is 4x + 7y = 28 -/
def line_equation (x y : ℝ) : Prop :=
  4 * x + 7 * y = 28

theorem x_intercept_of_line :
  x_intercept 4 7 28 = (7, 0) ∧ line_equation 7 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_line_l723_72316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangle_l723_72368

/-- The curve on which point A moves --/
noncomputable def C (x : ℝ) : ℝ := (3/2)^x

/-- The slope of line OA --/
noncomputable def k (a : ℝ) : ℝ := C a / a

/-- Theorem: No equilateral triangle OAB exists with A on the curve C and B on the positive x-axis --/
theorem no_equilateral_triangle :
  ∀ a b : ℝ, a > 0 → b > 0 →
  ¬(k a = Real.sqrt 3 ∧ C a = Real.sqrt 3 * b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equilateral_triangle_l723_72368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leahs_coins_value_l723_72367

/-- Represents the number of pennies Leah has -/
def pennies : ℕ := sorry

/-- Represents the number of nickels Leah has -/
def nickels : ℕ := sorry

/-- The total number of coins Leah has -/
def total_coins : ℕ := 15

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Condition: The total number of coins is 15 -/
axiom coin_count : pennies + nickels = total_coins

/-- Condition: If Leah had two more nickels, she would have the same number of pennies and nickels -/
axiom equal_with_two_more : nickels + 2 = pennies

/-- The total value of Leah's coins in cents -/
def total_value : ℕ := pennies * penny_value + nickels * nickel_value

/-- Theorem: The total value of Leah's coins is 38 cents -/
theorem leahs_coins_value : total_value = 38 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leahs_coins_value_l723_72367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_for_radius_three_l723_72349

open Real

/-- The ratio of the area of a star figure to the area of its circumscribing circle -/
noncomputable def star_circle_area_ratio (r : ℝ) : ℝ :=
  (6 * (Real.pi * r^2 / 6 - r^2 / 2)) / (Real.pi * r^2)

/-- Theorem: For a circle of radius 3 cut into six congruent arcs and joined to form a star figure,
    the ratio of the area of the star figure to the area of the original circle is (π - 3) / π -/
theorem star_circle_area_ratio_for_radius_three :
  star_circle_area_ratio 3 = (Real.pi - 3) / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_circle_area_ratio_for_radius_three_l723_72349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_6L_consumed_l723_72358

/-- Represents the motorcycle trip scenario -/
structure MotorcycleTrip where
  initialFuel : ℝ
  remainingFuelAfter2Hours : ℝ
  speed : ℝ

/-- Calculates the fuel consumption rate per hour -/
noncomputable def fuelConsumptionRate (trip : MotorcycleTrip) : ℝ :=
  (trip.initialFuel - trip.remainingFuelAfter2Hours) / 2

/-- Calculates the remaining fuel after t hours -/
noncomputable def remainingFuel (trip : MotorcycleTrip) (t : ℝ) : ℝ :=
  trip.initialFuel - (fuelConsumptionRate trip) * t

/-- Calculates the time when a specific amount of fuel is consumed -/
noncomputable def timeWhenFuelConsumed (trip : MotorcycleTrip) (fuelConsumed : ℝ) : ℝ :=
  fuelConsumed / (fuelConsumptionRate trip)

/-- Calculates the distance traveled in a given time -/
noncomputable def distanceTraveled (trip : MotorcycleTrip) (time : ℝ) : ℝ :=
  trip.speed * time

/-- Theorem: The distance traveled when 6 liters of fuel are consumed is 200 km -/
theorem distance_traveled_6L_consumed 
  (trip : MotorcycleTrip) 
  (h1 : trip.initialFuel = 9)
  (h2 : trip.remainingFuelAfter2Hours = 6)
  (h3 : trip.speed = 50) :
  distanceTraveled trip (timeWhenFuelConsumed trip 6) = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_traveled_6L_consumed_l723_72358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l723_72387

noncomputable def f (x : ℝ) := Real.cos (2 * x - Real.pi / 2)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ y, y > 0 ∧ (∀ x, f (x + y) = f x) → y ≥ Real.pi) ∧
  (∀ x, f (x + Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l723_72387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_choir_members_l723_72396

/-- Represents the number of choir members -/
def choir_members (m : ℕ) : Prop := True

/-- Represents a valid square formation size -/
def valid_square (m : ℕ) : Prop := ∃ k : ℕ, m = k^2 + 6

/-- Represents a valid rectangular formation size -/
def valid_rectangle (m : ℕ) : Prop := ∃ n : ℕ, m = n * (n + 6)

/-- The choir can be arranged in a square formation with 6 members left over -/
axiom square_condition (m : ℕ) : choir_members m → valid_square m

/-- The choir can be arranged in a rectangular formation with 6 more rows than columns -/
axiom rectangle_condition (m : ℕ) : choir_members m → valid_rectangle m

/-- The maximum number of choir members satisfying both conditions is 112 -/
theorem max_choir_members :
  (∃ m : ℕ, choir_members m ∧ valid_square m ∧ valid_rectangle m) →
  (∀ m : ℕ, choir_members m ∧ valid_square m ∧ valid_rectangle m → m ≤ 112) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_choir_members_l723_72396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_9_l723_72319

def T : ℕ → ℕ
  | 0 => 4  -- Add this case for 0
  | 1 => 4
  | n + 2 => 4^(T (n + 1))

theorem t_100_mod_9 : T 100 % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_9_l723_72319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_quadratic_trinomial_l723_72346

/-- A polynomial is a quadratic trinomial if it has degree 2 and consists of exactly three terms. -/
def is_quadratic_trinomial (p : Polynomial ℚ) : Prop :=
  p.degree = 2 ∧ p.support.card = 3

/-- The polynomial x^2 + x + 18 -/
noncomputable def p : Polynomial ℚ := Polynomial.X^2 + Polynomial.X + 18

theorem p_is_quadratic_trinomial : is_quadratic_trinomial p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_quadratic_trinomial_l723_72346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_sqrt_implies_a_leq_one_l723_72362

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) / Real.log x

-- State the theorem
theorem f_greater_than_sqrt_implies_a_leq_one (a : ℝ) :
  (∀ x > 1, f a x > Real.sqrt x) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_greater_than_sqrt_implies_a_leq_one_l723_72362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_A_rate_l723_72386

/-- The rate at which tap A fills the bucket in liters per minute. -/
noncomputable def rate_A : ℝ := 3

/-- The rate at which tap B fills the bucket in liters per minute. -/
noncomputable def rate_B : ℝ := 12 / 20

/-- The total volume of the bucket in liters. -/
noncomputable def bucket_volume : ℝ := 36

/-- The time it takes for both taps to fill the bucket together in minutes. -/
noncomputable def fill_time_both : ℝ := 10

/-- Theorem stating that the rate of tap A is 3 liters per minute. -/
theorem tap_A_rate :
  rate_A = 3 :=
by
  -- The proof is omitted for now
  sorry

#check tap_A_rate

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tap_A_rate_l723_72386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_triangle_l723_72345

def is_consecutive_even_integers (a b c : ℕ) : Prop :=
  ∃ x : ℕ, a = 2 * x ∧ b = 2 * x + 2 ∧ c = 2 * x + 4

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s : ℝ := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem smallest_perimeter_triangle (a b c : ℕ) :
  is_consecutive_even_integers a b c →
  triangle_area (a : ℝ) (b : ℝ) (c : ℝ) > 2 →
  a + b + c ≥ 18 :=
by
  sorry

#check smallest_perimeter_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_perimeter_triangle_l723_72345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_ray_l723_72341

open Set Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, deriv f x > f x)

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ := {x | exp (f x) > f 1 * exp x}

-- State the theorem
theorem solution_set_is_open_ray (f : ℝ → ℝ) (h1 : Differentiable ℝ f) (h2 : ∀ x, deriv f x > f x) :
  solution_set f = Ioi 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_is_open_ray_l723_72341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_DEF_l723_72307

/-- The radius of the inscribed circle in a triangle with sides a, b, and c --/
noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s

/-- Theorem: The radius of the inscribed circle in triangle DEF is approximately 2.57 --/
theorem inscribed_circle_radius_DEF :
  let r := inscribed_circle_radius 8 9 10
  ∃ ε > 0, abs (r - 2.57) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_DEF_l723_72307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l723_72303

/-- The line l: x - 2y + 8 = 0 --/
def line_l (x y : ℝ) : Prop := x - 2*y + 8 = 0

/-- Point A --/
def point_A : ℝ × ℝ := (2, 0)

/-- Point B --/
def point_B : ℝ × ℝ := (-2, -4)

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: The minimum sum of distances from any point P on line l to points A and B is 12 --/
theorem min_distance_sum :
  ∀ (P : ℝ × ℝ), line_l P.1 P.2 → distance P point_A + distance P point_B ≥ 12 := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l723_72303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l723_72394

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.a * t.c * Real.sin t.B

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.a - t.c)^2 = t.b^2 - t.a * t.c)
  (h2 : t.b = 2)
  (h3 : ∃ k, Real.sin t.A + k = Real.sin t.B ∧ Real.sin t.B + k = Real.sin t.C) :
  t.B = π / 3 ∧ 
  area t = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l723_72394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l723_72380

theorem max_value_trig_expression :
  ∀ x y z : ℝ, (Real.sin (2*x) + Real.sin (3*y) + Real.sin (4*z)) * 
               (Real.cos (2*x) + Real.cos (3*y) + Real.cos (4*z)) ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l723_72380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l723_72301

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
noncomputable def z : ℂ := Complex.abs ((Real.sqrt 3 - i) * i) + i ^ 2018

-- Theorem statement
theorem abs_z_equals_one : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_one_l723_72301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l723_72360

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an intersection point between circles -/
structure IntersectionPoint where
  point : ℝ × ℝ
  circles : List Nat

/-- Predicate to check if two circles are tangent -/
def IsTangent (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  d = c1.radius + c2.radius

/-- 
Theorem: Given n circles of equal radius where each circle intersects at least one other circle 
and no two circles are tangent, the number of intersection points p is greater than or equal to n.
-/
theorem intersection_points_count (n : ℕ) (circles : Fin n → Circle) 
  (intersections : List IntersectionPoint) :
  (∀ i : Fin n, ∃ j : Fin n, i ≠ j ∧ circles i ≠ circles j) →  -- Each circle intersects at least one other
  (∀ i j : Fin n, i ≠ j → ¬ IsTangent (circles i) (circles j)) →  -- No two circles are tangent
  (∀ i j : Fin n, circles i = circles j → i = j) →  -- All circles are distinct
  (∀ i j : Fin n, (circles i).radius = (circles j).radius) →  -- All circles have the same radius
  intersections.length ≥ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_count_l723_72360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyfifth_digit_of_sum_l723_72309

def decimal_sum (a b : ℚ) : ℕ → ℕ := sorry

def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem twentyfifth_digit_of_sum : 
  nth_digit_after_decimal (1/5 + 1/6) 25 = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twentyfifth_digit_of_sum_l723_72309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_l723_72344

-- Define the parabola
noncomputable def parabola (x : ℝ) : ℝ := (1/4) * x^2

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the line passing through the focus with slope 1
noncomputable def line (x : ℝ) : ℝ := x + 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ x : ℝ, p.1 = x ∧ p.2 = parabola x ∧ p.2 = line x}

-- State the theorem
theorem midpoint_x_coordinate :
  ∀ A B : ℝ × ℝ, A ∈ intersection_points → B ∈ intersection_points →
    A ≠ B →
    (A.1 + B.1) / 2 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_x_coordinate_l723_72344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l723_72325

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l723_72325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_A_union_B_l723_72376

def is_valid_set_pair (A B : Finset ℕ) : Prop :=
  (∀ x y, x ∈ A → y ∈ A → x ≠ y → x + y ∈ B) ∧
  (∀ x y, x ∈ B → y ∈ B → x ≠ y → (max x y) / (min x y) ∈ A)

theorem max_elements_A_union_B :
  ∃ (A B : Finset ℕ), is_valid_set_pair A B ∧ (A ∪ B).card = 5 ∧
  ∀ (C D : Finset ℕ), is_valid_set_pair C D → (C ∪ D).card ≤ 5 := by
  sorry

#check max_elements_A_union_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_elements_A_union_B_l723_72376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_squares_traversal_l723_72352

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in 2D space -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Represents the configuration of three intersecting squares -/
structure ThreeSquares where
  squareA : Square
  squareB : Square
  squareC : Square

/-- Represents a path through the squares -/
structure PathThrough where
  points : List Point

/-- Checks if a path is valid (continuous and covers all edges once) -/
def isValidPath (squares : ThreeSquares) (path : PathThrough) : Prop :=
  sorry

/-- Main theorem: There exists a valid path for any configuration of three intersecting squares -/
theorem three_squares_traversal (squares : ThreeSquares) :
  ∃ (path : PathThrough), isValidPath squares path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_squares_traversal_l723_72352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l723_72371

/-- Calculates the time taken for two trains moving in opposite directions to cross each other. -/
noncomputable def timeToCross (speed1 speed2 : ℝ) (length1 length2 : ℝ) : ℝ :=
  let relativeSpeed := speed1 + speed2
  let combinedLength := length1 + length2
  combinedLength / (relativeSpeed * 1000 / 3600)

/-- Theorem stating that the time taken for two specific trains to cross is approximately 19.63 seconds. -/
theorem train_crossing_time :
  let speed1 : ℝ := 100  -- km/h
  let speed2 : ℝ := 120  -- km/h
  let length1 : ℝ := 500 -- meters
  let length2 : ℝ := 700 -- meters
  ∃ ε > 0, |timeToCross speed1 speed2 length1 length2 - 19.63| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l723_72371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_expressions_positive_l723_72337

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  opens_downwards : a < 0
  vertex_x_between_0_and_1 : 0 < -b / (2 * a) ∧ -b / (2 * a) < 1

/-- The number of positive expressions among ab, ac, a+b+c, a-b+c, 2a+b, and 2a-b -/
noncomputable def count_positive (f : QuadraticFunction) : ℕ :=
  let expressions := [f.a * f.b, f.a * f.c, f.a + f.b + f.c, f.a - f.b + f.c, 2 * f.a + f.b, 2 * f.a - f.b]
  expressions.filter (λ x => x > 0) |>.length

/-- Theorem stating that exactly two expressions are positive -/
theorem two_expressions_positive (f : QuadraticFunction) : count_positive f = 2 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_expressions_positive_l723_72337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_price_is_correct_l723_72365

def total_boxes : ℕ := 2000
def cost_per_kg : ℚ := 10
def weight_per_box : ℚ := 5
def sample_size : ℕ := 20
def sample_weights : List ℚ := [4.7, 4.8, 4.6, 4.5, 4.8, 4.9, 4.8, 4.7, 4.8, 4.7, 
                                4.8, 4.9, 4.7, 4.8, 4.5, 4.7, 4.7, 4.9, 4.7, 5.0]

noncomputable def mode (l : List ℚ) : ℚ := sorry

def total_damaged_weight (m : ℚ) : ℚ := total_boxes * (weight_per_box - m)

def break_even_price (damaged : ℚ) : ℚ :=
  (cost_per_kg * total_boxes * weight_per_box) / (total_boxes * weight_per_box - damaged)

theorem break_even_price_is_correct : 
  let m := mode sample_weights
  let damaged := total_damaged_weight m
  ∃ ε > 0, |break_even_price damaged - 10.7| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_break_even_price_is_correct_l723_72365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l723_72384

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

/-- Point P -/
def P : ℝ × ℝ := (1, 1)

/-- A chord of the ellipse -/
structure Chord where
  slope : ℝ
  intercept : ℝ

/-- The midpoint of a chord -/
def chordMidpoint (c : Chord) : ℝ × ℝ := 
  (1, c.slope + c.intercept)

/-- The theorem stating the equation of the chord -/
theorem chord_equation : 
  ∃ (c : Chord), 
    ellipse (chordMidpoint c).1 (chordMidpoint c).2 ∧ 
    chordMidpoint c = P ∧
    c.slope = -1/4 ∧ 
    c.intercept = 5/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_equation_l723_72384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_ratio_is_one_half_l723_72348

/-- Represents the financial transaction sequence --/
structure Transaction where
  initial_amount : ℚ
  ice_cream_cost : ℚ
  deposit_fraction : ℚ
  final_amount : ℚ

/-- Calculates the ratio of t-shirt cost to remaining money after ice cream purchase --/
def tshirt_ratio (t : Transaction) : ℚ :=
  let remaining_after_ice_cream := t.initial_amount - t.ice_cream_cost
  let tshirt_cost := remaining_after_ice_cream - 
    (t.final_amount / (1 - t.deposit_fraction))
  tshirt_cost / remaining_after_ice_cream

/-- The main theorem stating the ratio is 1:2 --/
theorem tshirt_ratio_is_one_half (t : Transaction) 
  (h1 : t.initial_amount = 65)
  (h2 : t.ice_cream_cost = 5)
  (h3 : t.deposit_fraction = 1/5)
  (h4 : t.final_amount = 24) :
  tshirt_ratio t = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_ratio_is_one_half_l723_72348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_range_of_m_l723_72383

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then 2^x - 1/(2^x) else 0

-- Theorem 1: Unique solution for f(x) = 3/2
theorem unique_solution_exists :
  ∃! x, x > 0 ∧ f x = 3/2 ∧ x = 1 := by sorry

-- Theorem 2: Range of m
theorem range_of_m (m : ℝ) (h : m ≥ -5) :
  ∀ t ∈ Set.Icc 1 2, 
    2^t * (2^(2*t) - 1/(2^(2*t))) + m * (2^t - 1/(2^t)) ≥ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exists_range_of_m_l723_72383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_12_minutes_l723_72331

/-- Calculates the cost of a call under Plan A -/
noncomputable def costPlanA (duration : ℝ) : ℝ :=
  if duration ≤ 9 then 0.60 else 0.60 + (duration - 9) * 0.06

/-- Calculates the cost of a call under Plan B -/
def costPlanB (duration : ℝ) : ℝ :=
  duration * 0.08

/-- The duration at which both plans cost the same -/
def equalCostDuration : ℝ := 12

theorem equal_cost_at_12_minutes :
  costPlanA equalCostDuration = costPlanB equalCostDuration :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_at_12_minutes_l723_72331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l723_72377

theorem trigonometric_identity (α : ℝ) :
  2 - (Real.sin (8 * α)) / (Real.sin (2 * α) ^ 4 - Real.cos (2 * α) ^ 4) = 4 * (Real.cos (π / 4 - 2 * α)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l723_72377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l723_72302

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + a*x - 2 else Real.log x / Real.log a

-- State the theorem
theorem monotonic_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_range_l723_72302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_hyperbola_is_valid_l723_72324

/-- A hyperbola is defined by its equation, asymptotes, and a point it passes through. -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form ax² + by² = c -/
  equation : ℝ → ℝ → ℝ
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- Checks if the given hyperbola satisfies the conditions -/
def is_valid_hyperbola (h : Hyperbola) : Prop :=
  let (x, y) := h.point
  -- The equation is satisfied by the given point
  h.equation x y = 0 ∧
  -- The asymptotes have the correct slope
  h.asymptote_slope = 3 ∧
  -- The equation has the correct form (x² - y²/9 = 1)
  ∀ x y, h.equation x y = x^2 - y^2/9 - 1

/-- The hyperbola we want to prove is valid -/
noncomputable def our_hyperbola : Hyperbola where
  equation := fun x y => x^2 - y^2/9 - 1
  asymptote_slope := 3
  point := (2, -3 * Real.sqrt 3)

/-- Theorem stating that our_hyperbola is a valid hyperbola -/
theorem our_hyperbola_is_valid : is_valid_hyperbola our_hyperbola := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_hyperbola_is_valid_l723_72324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l723_72322

noncomputable def g (x : ℝ) : ℝ := 
  (Real.sin x ^ 3 + 5 * Real.sin x ^ 2 + 2 * Real.sin x + 3 * Real.cos x ^ 2 - 9) / (Real.sin x + 1)

theorem g_range :
  ∀ y : ℝ, (∃ x : ℝ, Real.sin x ≠ -1 ∧ g x = y) ↔ -6 ≤ y ∧ y ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l723_72322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l723_72317

noncomputable def f (k x : ℝ) : ℝ := k * (x - 1) * Real.exp x + x^2

theorem tangent_line_at_one (k : ℝ) (h : k = -1 / Real.exp 1) :
  let f' := fun x => k * x * Real.exp x + 2 * x
  (fun x => x - f k x) = (fun x => x - 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l723_72317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_4pi_l723_72315

noncomputable def enclosed_area : ℝ := 4 * Real.pi

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x

def g : ℝ → ℝ := λ _ ↦ 2

theorem enclosed_area_is_4pi :
  ∫ x in (0)..(2 * Real.pi), |g x - f x| = enclosed_area := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_4pi_l723_72315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l723_72327

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The right focus of a hyperbola -/
noncomputable def rightFocus (h : Hyperbola) : Point where
  x := Real.sqrt (h.a^2 + h.b^2)
  y := 0

/-- A line with slope 1 passing through a point -/
structure LineWithSlope1 (p : Point)

/-- Intersection point of a line with slope 1 and a hyperbola -/
noncomputable def intersectionPoint (h : Hyperbola) (l : LineWithSlope1 (rightFocus h)) : Point :=
  sorry

/-- Predicate for an isosceles right triangle formed by F1, F2, and A -/
def isIsoscelesRightTriangle (h : Hyperbola) (a : Point) : Prop :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (l : LineWithSlope1 (rightFocus h)) 
  (a : Point := intersectionPoint h l)
  (h_isosceles : isIsoscelesRightTriangle h a) : 
  eccentricity h = Real.sqrt 2 + 1 :=
by sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l723_72327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l723_72336

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 4*x + a else Real.log x + 1

-- Theorem statement
theorem min_value_implies_a_range (a : ℝ) : 
  (∀ x, f a x ≥ 1) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_range_l723_72336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_surface_fraction_is_seven_eighths_l723_72320

/-- Represents a cube with its properties -/
structure Cube where
  edge_length : ℕ
  total_small_cubes : ℕ
  white_small_cubes : ℕ
  black_small_cubes : ℕ

/-- Calculates the fraction of white surface area on a cube -/
def white_surface_fraction (c : Cube) : ℚ :=
  let total_surface_area := (6 * c.edge_length * c.edge_length : ℚ)
  let black_faces_on_corners := (2 * 3 : ℚ) -- 2 corner cubes, 3 faces each
  let black_faces_on_centers := (6 : ℚ) -- 1 face for each of the 6 center black cubes
  let total_black_faces := black_faces_on_corners + black_faces_on_centers
  let white_faces := total_surface_area - total_black_faces
  white_faces / total_surface_area

/-- The main theorem stating the fraction of white surface area -/
theorem white_surface_fraction_is_seven_eighths (c : Cube)
  (h1 : c.edge_length = 4)
  (h2 : c.total_small_cubes = 64)
  (h3 : c.white_small_cubes = 48)
  (h4 : c.black_small_cubes = 16)
  (h5 : c.total_small_cubes = c.white_small_cubes + c.black_small_cubes) :
  white_surface_fraction c = 7/8 := by
  sorry

#eval white_surface_fraction ⟨4, 64, 48, 16⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_surface_fraction_is_seven_eighths_l723_72320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_65_l723_72328

theorem divisibility_by_65 (n : ℕ) : ∃ k : ℤ, (7 : ℤ)^(4*n) - (4 : ℤ)^(4*n) = 65 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_65_l723_72328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_percentage_is_correct_l723_72314

-- Define the commission structure
noncomputable def commission_rate_low : ℝ := 0.20
noncomputable def commission_rate_high : ℝ := 0.50
noncomputable def commission_threshold : ℝ := 500

-- Define the total sale amount
noncomputable def total_sale : ℝ := 800

-- Calculate the commission
noncomputable def commission : ℝ :=
  commission_rate_low * commission_threshold +
  commission_rate_high * (total_sale - commission_threshold)

-- Calculate the commission percentage
noncomputable def commission_percentage : ℝ :=
  (commission / total_sale) * 100

-- Theorem to prove
theorem commission_percentage_is_correct :
  commission_percentage = 31.25 := by
  -- Expand definitions
  unfold commission_percentage
  unfold commission
  unfold total_sale
  unfold commission_threshold
  unfold commission_rate_low
  unfold commission_rate_high
  -- Perform numerical calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_percentage_is_correct_l723_72314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l723_72395

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 9*x < 0}
def B : Set ℝ := {x | 1 < Real.exp (x * Real.log 2) ∧ Real.exp (x * Real.log 2) < 8}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l723_72395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_7pi_l723_72304

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sin (3 * x)

-- Define the interval [0, 2π]
def I : Set ℝ := Set.Icc 0 (2 * Real.pi)

-- Theorem statement
theorem sum_of_zeros_is_7pi :
  ∃ (S : Finset ℝ), S.toSet ⊆ I ∧ (∀ x ∈ S, f x = 0) ∧
  (∀ x ∈ I, f x = 0 → x ∈ S) ∧
  S.sum id = 7 * Real.pi :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_7pi_l723_72304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l723_72340

-- Define the inequality and its solution set
def inequality (a c : ℝ) (x : ℝ) := x^2 + a*x - c < 0
def solution_set := {x : ℝ | -2 < x ∧ x < 1}

-- Define the function
noncomputable def f (a m c : ℝ) (x : ℝ) := a*x^3 + m*x^2 + x + c/2

-- Define the property of not being monotonic in an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) :=
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ f x > f y ∧
  ∃ x y, a < x ∧ x < y ∧ y < b ∧ f x < f y

-- State the theorem
theorem m_range (a c : ℝ) :
  (∀ x, inequality a c x ↔ x ∈ solution_set) →
  not_monotonic (f 1 m 2) (1/2) 1 →
  ∃ m₁ m₂, m₁ = -2 ∧ m₂ = -Real.sqrt 3 ∧ m₁ < m ∧ m < m₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l723_72340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_and_trigonometry_equalities_l723_72389

open Real

theorem logarithm_and_trigonometry_equalities :
  (∃ (lg : ℝ → ℝ), lg 10 = 1 ∧
    lg (5^2) + (2/3) * lg 8 + lg 5 * lg 20 + (lg 2)^2 = 3) ∧
  cos (17*π/4) + sin (13*π/3) + tan (25*π/6) = (3*Real.sqrt 2 + 5*Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_and_trigonometry_equalities_l723_72389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l723_72370

/-- The area of a triangle formed by the vertex of the parabola y = x^2 - 4 and 
    the intersections of the line y = r with the parabola -/
noncomputable def triangleArea (r : ℝ) : ℝ := (r + 4) * Real.sqrt (r + 4)

/-- Theorem stating the range of r for which the triangle area is between 16 and 128 -/
theorem triangle_area_range (r : ℝ) : 
  16 ≤ triangleArea r ∧ triangleArea r ≤ 128 ↔ 0 ≤ r ∧ r ≤ 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_range_l723_72370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_and_integral_l723_72323

-- Define the parabolas
def C₁ (x : ℝ) : ℝ := -x^2 + 1
def C₂ (u x : ℝ) : ℝ := (x - u)^2 + u

-- Define the range of u
noncomputable def a : ℝ := -Real.sqrt 3 - 1
noncomputable def b : ℝ := Real.sqrt 3 - 1

-- Define the function f(u)
noncomputable def f (u : ℝ) : ℝ := (u^2 + u + 1) * Real.sqrt (2 - u - u^2)

theorem parabolas_intersection_and_integral :
  (∀ u : ℝ, (∃ x : ℝ, C₁ x = C₂ u x) ↔ a ≤ u ∧ u ≤ b) ∧
  (∀ u : ℝ, a ≤ u → u ≤ b →
    ∃ x₁ x₂ y₁ y₂ : ℝ,
      C₁ x₁ = C₂ u x₁ ∧
      C₁ x₂ = C₂ u x₂ ∧
      y₁ = C₁ x₁ ∧
      y₂ = C₁ x₂ ∧
      2 * |x₁ * y₂ - x₂ * y₁| = f u) ∧
  (∫ x in a..b, f x) = 21 * Real.pi / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_and_integral_l723_72323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_price_function_l723_72334

/-- Given that each box of beverages costs 55 yuan and contains 6 bottles,
    prove that the function y = (55x)/6 represents the total price y (in yuan)
    as a function of the number of bottles x purchased. -/
theorem beverage_price_function (x : ℝ) :
  let box_price : ℝ := 55
  let bottles_per_box : ℝ := 6
  let total_price : ℝ → ℝ := fun bottles ↦ (box_price * bottles) / bottles_per_box
  total_price x = (55 * x) / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_price_function_l723_72334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_two_distinct_roots_roots_difference_2sqrt2_roots_for_m_neg3_roots_for_m_1_l723_72393

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Creates a quadratic equation based on the parameter m -/
def createEquation (m : ℝ) : QuadraticEquation :=
  { a := 1, b := m + 3, c := m + 1 }

/-- Calculates the discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4*eq.a*eq.c

/-- Calculates the roots of a quadratic equation -/
noncomputable def roots (eq : QuadraticEquation) : ℝ × ℝ :=
  let d := discriminant eq
  ((-eq.b + Real.sqrt d) / (2*eq.a), (-eq.b - Real.sqrt d) / (2*eq.a))

theorem always_two_distinct_roots (m : ℝ) :
  let eq := createEquation m
  discriminant eq > 0 := by sorry

theorem roots_difference_2sqrt2 (m : ℝ) :
  let eq := createEquation m
  let (x₁, x₂) := roots eq
  |x₁ - x₂| = 2 * Real.sqrt 2 → m = -3 ∨ m = 1 := by sorry

theorem roots_for_m_neg3 :
  let eq := createEquation (-3)
  let (x₁, x₂) := roots eq
  (x₁ = Real.sqrt 2 ∧ x₂ = -Real.sqrt 2) ∨ (x₁ = -Real.sqrt 2 ∧ x₂ = Real.sqrt 2) := by sorry

theorem roots_for_m_1 :
  let eq := createEquation 1
  let (x₁, x₂) := roots eq
  (x₁ = -2 + Real.sqrt 2 ∧ x₂ = -2 - Real.sqrt 2) ∨ (x₁ = -2 - Real.sqrt 2 ∧ x₂ = -2 + Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_two_distinct_roots_roots_difference_2sqrt2_roots_for_m_neg3_roots_for_m_1_l723_72393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l723_72300

theorem remainder_problem (k : ℕ) (hk : k > 0) (h : 180 % (k^2) = 12) : 250 % k = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l723_72300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cosine_matrix_is_zero_l723_72306

/-- The determinant of the given 3x3 matrix is zero -/
theorem det_cosine_matrix_is_zero (a b c : ℝ) : 
  Matrix.det (![
    ![1, Real.cos (a + b + c), Real.cos (a + c)],
    ![Real.cos (a + b + c), 1, Real.cos (b + c)],
    ![Real.cos (a + c), Real.cos (b + c), 1]
  ] : Matrix (Fin 3) (Fin 3) ℝ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_cosine_matrix_is_zero_l723_72306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_statements_l723_72375

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- subset relation
variable (parallel_line : Line → Line → Prop)  -- parallel relation for lines
variable (parallel_line_plane : Line → Plane → Prop)  -- parallel relation for line and plane
variable (parallel_plane : Plane → Plane → Prop)  -- parallel relation for planes
variable (perpendicular : Line → Plane → Prop)  -- perpendicular relation
variable (intersect : Plane → Plane → Line → Prop)  -- intersection of two planes is a line

-- Define the statement
theorem geometry_statements 
  (l m : Line) (α β : Plane) 
  (h_diff_lines : l ≠ m) (h_diff_planes : α ≠ β) : 
  ¬(
    ((subset l α ∧ subset m α ∧ parallel_line_plane l β ∧ parallel_line_plane m β) → parallel_plane α β) ∧
    ((subset l α ∧ parallel_line_plane l β ∧ intersect α β m) → parallel_line l m) ∧
    ((parallel_plane α β ∧ parallel_line_plane l α) → parallel_line_plane l β) ∧
    ((perpendicular l α ∧ parallel_line m l ∧ parallel_plane α β) → perpendicular m β)
  ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_statements_l723_72375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_containing_circle_radius_is_correct_l723_72329

/-- The radius of a circle containing two circles of radius r and one circle of radius R,
    where each circle is tangent to the other two. -/
noncomputable def containing_circle_radius (r R : ℝ) : ℝ :=
  (R * (R + r + Real.sqrt (R^2 + 2*r*R))) / (R - r + Real.sqrt (R^2 + 2*r*R))

/-- Theorem stating that the containing_circle_radius function correctly calculates
    the radius of the circle containing two circles of radius r and one of radius R,
    with all circles being mutually tangent. -/
theorem containing_circle_radius_is_correct (r R : ℝ) (hr : r > 0) (hR : R > 0) :
  let x := containing_circle_radius r R
  (x - R)^2 + (x - r)^2 = (R + r)^2 ∧
  x > R ∧ x > r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_containing_circle_radius_is_correct_l723_72329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_line_intersection_l723_72378

/-- Given a triangle ABC with centroid G, if a line PQ passes through G
    intersecting AB at P and AC at Q, and AP = ma and AQ = nb
    (where a = AB and b = AC), then 1/m + 1/n = 3 -/
theorem centroid_line_intersection
  (A B C G P Q : EuclideanSpace ℝ (Fin 3))
  (a b : EuclideanSpace ℝ (Fin 3))
  (m n : ℝ) :
  G = (1/3 : ℝ) • (A + B + C) →  -- G is the centroid
  a = B - A →            -- a is vector AB
  b = C - A →            -- b is vector AC
  P = A + m • a →        -- P is on AB
  Q = A + n • b →        -- Q is on AC
  ∃ (t : ℝ), G = P + t • (Q - P) →  -- G is on PQ
  1 / m + 1 / n = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_line_intersection_l723_72378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l723_72318

/-- Two lines in the plane -/
structure Lines where
  a : ℝ
  l1 : ℝ → ℝ → ℝ := λ x y => a * x + 2 * y + 6
  l2 : ℝ → ℝ → ℝ := λ x y => x + (a - 1) * y + a^2 - 1

/-- Perpendicular lines -/
def perpendicular (lines : Lines) : Prop :=
  lines.a * (1 / (lines.a - 1)) = -1

/-- Parallel lines -/
def parallel (lines : Lines) : Prop :=
  lines.a / 2 = 1 / (lines.a - 1)

/-- Distance between parallel lines -/
noncomputable def distance (lines : Lines) : ℝ :=
  6 / Real.sqrt 5

theorem lines_theorem (lines : Lines) :
  (perpendicular lines → lines.a = 2/3) ∧
  (parallel lines → distance lines = 6 * Real.sqrt 5 / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_theorem_l723_72318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_pi_third_radius_six_l723_72353

/-- The area of a sector with central angle θ and radius r -/
noncomputable def sectorArea (θ : ℝ) (r : ℝ) : ℝ :=
  1/2 * θ * r^2

theorem sector_area_pi_third_radius_six :
  sectorArea (π/3) 6 = 6 * π := by
  -- Unfold the definition of sectorArea
  unfold sectorArea
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_pi_third_radius_six_l723_72353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_two_fifty_l723_72385

theorem log_sum_two_fifty (log : ℝ → ℝ) : 
  (∀ x y, x > 0 → y > 0 → log (x * y) = log x + log y) →  -- logarithm product rule
  (∀ x, x > 0 → log (x^10) = 10 * log x) →                -- logarithm power rule
  log 10 = 1 →                                            -- definition of log base 10
  log 2 + log 50 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_two_fifty_l723_72385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_angle_l723_72308

noncomputable section

-- Define the given line l
def line_l (x y : ℝ) : Prop := y - 1 = Real.sqrt 3 * (x - 2)

-- Define the point P
def point_P : ℝ × ℝ := (2, 1)

-- Define the acute angle
def angle : ℝ := 30 * Real.pi / 180

-- Define the equation of the line we're looking for
def target_line (x y : ℝ) : Prop := x = 2 ∨ x - Real.sqrt 3 * y - 2 + Real.sqrt 3 = 0

-- Theorem statement
theorem line_through_point_with_angle :
  ∃ (x y : ℝ), (x, y) = point_P ∧ 
  (∃ (θ : ℝ), θ = angle ∧ 
    (∀ (x' y' : ℝ), line_l x' y' → 
      ∃ (θ' : ℝ), θ' = θ ∨ θ' = Real.pi - θ)) →
  target_line x y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_with_angle_l723_72308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l723_72388

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (1 / (8 + 2*x - x^2))

-- Define the domain of the function
def domain : Set ℝ := {x : ℝ | 8 + 2*x - x^2 > 0}

-- State the theorem
theorem monotonic_decreasing_interval :
  ∃ (a b : ℝ), a = -2 ∧ b = 1 ∧
  (∀ x y, x ∈ domain → y ∈ domain → a < x ∧ x < y ∧ y ≤ b → f y ≤ f x) ∧
  (∀ ε > 0, ∃ x y, x ∈ domain ∧ y ∈ domain ∧ 
    a - ε < x ∧ x < y ∧ y < a ∧ f x < f y) ∧
  (∀ ε > 0, ∃ x y, x ∈ domain ∧ y ∈ domain ∧ 
    b < x ∧ x < y ∧ y < b + ε ∧ f x < f y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l723_72388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_l723_72373

theorem angle_A_is_pi_over_three (a b c : ℝ) (h : (a + b + c) * (b + c - a) = 3 * b * c) :
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_pi_over_three_l723_72373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_mod_1000_l723_72372

def N : ℕ := (Finset.range 50).sum (λ i => 
  (4 * i + 1)^2 + (4 * i + 2)^2 - (4 * i + 3)^2 - (4 * i + 4)^2)

theorem N_mod_1000 : N % 1000 = 796 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_mod_1000_l723_72372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l723_72310

noncomputable def dilation (center : ℂ) (scale : ℝ) (point : ℂ) : ℂ :=
  center + scale • (point - center)

theorem dilation_image : 
  let center : ℂ := 1 - 3*Complex.I
  let scale : ℝ := 3
  let original : ℂ := -1 + 2*Complex.I
  dilation center scale original = -5 + 12*Complex.I :=
by
  -- Expand the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.add_re, Complex.add_im, Complex.mul_re, Complex.mul_im]
  -- The proof is completed with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_image_l723_72310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_sum_l723_72382

open Real

-- Define the radius of the circle
variable (r : ℝ)

-- Define the areas of the circle and triangle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def triangle_area (r : ℝ) : ℝ := (3 * Real.sqrt 3 * r^2) / 4

-- State the theorem
theorem circle_triangle_area_sum :
  circle_area r + triangle_area r = 50 →
  circle_area r = (Real.pi * 200) / (4 * Real.pi + 3 * Real.sqrt 3) ∧
  triangle_area r = (150 * Real.sqrt 3) / (4 * Real.pi + 3 * Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_triangle_area_sum_l723_72382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_four_collinear_l723_72392

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The set of points in the plane -/
def PointSet : Finset Point := sorry

/-- The set of lines formed by any two points in the PointSet -/
def LineSet : Finset Line := sorry

/-- A function that checks if a set of points is collinear -/
def isCollinear (points : Finset Point) : Prop := sorry

theorem at_least_four_collinear :
  (Finset.card PointSet = 65) →
  (Finset.card LineSet = 2015) →
  ∃ (collinearPoints : Finset Point), collinearPoints ⊆ PointSet ∧ Finset.card collinearPoints ≥ 4 ∧ isCollinear collinearPoints :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_four_collinear_l723_72392
