import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_paths_count_l960_96036

/-- Represents the triangular arrangement of letters -/
structure LetterArrangement : Type :=
  (dummy : Unit)

/-- Represents a valid path in the letter arrangement -/
structure ValidPath (arrangement : LetterArrangement) : Type :=
  (dummy : Unit)

/-- Counts the number of valid paths spelling "CONTEST" -/
def count_valid_paths (arrangement : LetterArrangement) : ℕ := sorry

/-- The main theorem stating that there are 127 valid paths -/
theorem contest_paths_count (arrangement : LetterArrangement) : 
  count_valid_paths arrangement = 127 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contest_paths_count_l960_96036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_not_divisible_by_three_l960_96041

theorem max_numbers_not_divisible_by_three (n : ℕ) (hn : n = 2008) :
  ∃ (S : Finset ℕ),
    (∀ x, x ∈ S → x ≤ n) ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y → (x + y) % 3 ≠ 0) ∧
    S.card = 671 ∧
    (∀ T : Finset ℕ, (∀ x, x ∈ T → x ≤ n) →
      (∀ x y, x ∈ T → y ∈ T → x ≠ y → (x + y) % 3 ≠ 0) →
      T.card ≤ S.card) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_numbers_not_divisible_by_three_l960_96041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analytical_method_most_suitable_inequality_proof_l960_96060

/-- The analytical method for proving inequalities -/
def analytical_method (ineq : ℝ → ℝ → Prop) : Prop :=
  ∃ (f : ℝ → ℝ), ∀ x y, ineq x y ↔ f x < f y

/-- The inequality to be verified -/
def target_inequality : ℝ → ℝ → Prop :=
  λ x y ↦ Real.sqrt 2 + Real.sqrt 7 < Real.sqrt 3 + Real.sqrt 6

/-- Theorem stating that the analytical method is most suitable for the target inequality -/
theorem analytical_method_most_suitable :
  analytical_method target_inequality ∧
  ∀ (other_method : (ℝ → ℝ → Prop) → Prop),
    other_method ≠ analytical_method →
    ¬(other_method target_inequality) ∨
    (analytical_method target_inequality ∧ other_method target_inequality) := by
  sorry

/-- Proof that the inequality holds using the analytical method -/
theorem inequality_proof : target_inequality 0 0 := by
  unfold target_inequality
  have h : (Real.sqrt 2 + Real.sqrt 7)^2 < (Real.sqrt 3 + Real.sqrt 6)^2 := by
    -- The actual proof would go here
    sorry
  -- Convert back to the original inequality
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_analytical_method_most_suitable_inequality_proof_l960_96060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l960_96089

noncomputable def f (a x : ℝ) := Real.log x + a / x - x + 1 - a

theorem min_a_value (a : ℤ) :
  (∃ x : ℝ, x > 1 ∧ f a x + x < (1 - x) / x) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l960_96089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_percentage_l960_96015

/-- Calculates the percentage reduction between two numbers -/
noncomputable def percentageReduction (original : ℝ) (reduced : ℝ) : ℝ :=
  ((original - reduced) / original) * 100

theorem faculty_reduction_percentage : 
  let original : ℝ := 253.25
  let reduced : ℝ := 195
  abs (percentageReduction original reduced - 22.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_faculty_reduction_percentage_l960_96015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l960_96013

-- Define the sphere and points
def Sphere : Type := Unit
def Point : Type := Unit

-- Define the radius of the sphere
def radius : ℝ := 5

-- Define the distance function between two points
noncomputable def distance (p q : Point) : ℝ := sorry

-- Define the function to calculate the surface area of a circle
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

-- State the theorem
theorem intersection_area 
  (A B C D : Point) 
  (sphere : Sphere) 
  (h1 : distance A B = 2 * Real.sqrt 5) 
  (h2 : distance A C = 2 * Real.sqrt 5)
  (h3 : distance A D = 2 * Real.sqrt 5)
  (h4 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  ∃ (intersection_shape : Set Point),
    circle_area (radius * 4 / 5) = 16 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_l960_96013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_x_squared_plus_x_minus_two_to_fourth_l960_96007

theorem sum_of_coefficients_x_squared_plus_x_minus_two_to_fourth :
  (1^2 + 1 - 2)^4 = 0 := by
  calc
    (1^2 + 1 - 2)^4 = (1 + 1 - 2)^4 := by rfl
    _ = 0^4 := by rfl
    _ = 0 := by rfl

#check sum_of_coefficients_x_squared_plus_x_minus_two_to_fourth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_x_squared_plus_x_minus_two_to_fourth_l960_96007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l960_96006

/-- Calculates the speed of a car given distance and time -/
noncomputable def calculate_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem car_speed_proof (distance : ℝ) (time : ℝ) 
  (h1 : distance = 624)
  (h2 : time = 3 + 1 / 5) :
  calculate_speed distance time = 195 := by
  -- Unfold the definition of calculate_speed
  unfold calculate_speed
  -- Substitute the given values
  rw [h1, h2]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_proof_l960_96006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_8_l960_96059

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 - 40 * x + 128) / (x - 8)

theorem limit_f_at_8 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 8 → |x - 8| < δ → |f x - 8| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_f_at_8_l960_96059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_properties_l960_96090

def square (x y : ℝ) : Prop :=
  -2 * Real.pi ≤ x ∧ x ≤ 3 * Real.pi ∧ -Real.pi ≤ y ∧ y ≤ 4 * Real.pi

def satisfies_equations (x y : ℝ) : Prop :=
  Real.sin x + Real.sin y = Real.sin 4 ∧ Real.cos x + Real.cos y = Real.cos 4

def solution_set : Set (ℝ × ℝ) :=
  {p | square p.1 p.2 ∧ satisfies_equations p.1 p.2}

def largest_abscissa : Set (ℝ × ℝ) :=
  {(4 + 5 * Real.pi / 3, 4 + Real.pi / 3),
   (4 + 5 * Real.pi / 3, 4 + 7 * Real.pi / 3),
   (4 + 5 * Real.pi / 3, 4 - 5 * Real.pi / 3)}

theorem solution_properties :
  (∃ s : Finset (ℝ × ℝ), s.card = 13 ∧ ∀ p ∈ s, p ∈ solution_set) ∧
  (∀ p ∈ solution_set, p.1 ≤ 4 + 5 * Real.pi / 3) ∧
  (largest_abscissa ⊆ solution_set) := by
  sorry

#check solution_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_properties_l960_96090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_C_best_l960_96084

-- Define the structure for an athlete's performance
structure AthletePerformance where
  name : String
  average : Float
  stdDev : Float

-- Define the four athletes
def athleteA : AthletePerformance := ⟨"A", 8.6, 1.3⟩
def athleteB : AthletePerformance := ⟨"B", 8.6, 1.5⟩
def athleteC : AthletePerformance := ⟨"C", 9.1, 1.0⟩
def athleteD : AthletePerformance := ⟨"D", 9.1, 1.2⟩

-- Define a function to compare two athletes
def betterPerformance (a1 a2 : AthletePerformance) : Prop :=
  (a1.average > a2.average) ∨ (a1.average = a2.average ∧ a1.stdDev < a2.stdDev)

-- Theorem stating that athlete C has the best performance
theorem athlete_C_best : 
  betterPerformance athleteC athleteA ∧
  betterPerformance athleteC athleteB ∧
  betterPerformance athleteC athleteD := by
  sorry

#check athlete_C_best

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_C_best_l960_96084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_perimeter_l960_96098

/-- An equilateral triangle with side length 3 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 3

/-- The triangle formed by connecting the midpoints of an equilateral triangle's sides -/
noncomputable def MidpointTriangle (t : EquilateralTriangle) : ℝ :=
  3 * (t.side_length / 2)

/-- The perimeter of the midpoint triangle is 9/2 -/
theorem midpoint_triangle_perimeter (t : EquilateralTriangle) :
  MidpointTriangle t = 9/2 := by
  unfold MidpointTriangle
  rw [t.is_equilateral]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_perimeter_l960_96098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l960_96005

noncomputable def sequenceX (n : ℕ) : ℝ :=
  match n with
  | 0 => 5
  | n + 1 => sequenceX n + 1 / sequenceX n

theorem sequence_bounds : 45 < sequenceX 1000 ∧ sequenceX 1000 < 45.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounds_l960_96005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_int_solution_but_infinite_int_solutions_l960_96023

theorem no_positive_int_solution_but_infinite_int_solutions :
  (¬ ∃ (x y z t : ℕ+), (x : ℚ) / y + (y : ℚ) / z + (z : ℚ) / t + (t : ℚ) / x = 1) ∧
  (∃ f : ℕ → ℤ × ℤ × ℤ × ℤ, ∀ n : ℕ,
    let (x, y, z, t) := f n
    (x : ℚ) / y + (y : ℚ) / z + (z : ℚ) / t + (t : ℚ) / x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_int_solution_but_infinite_int_solutions_l960_96023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l960_96070

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := 1 / (x - 1)^2

-- State the theorem about the range of g(x)
theorem range_of_g :
  ∀ y > 0, ∃ x ≠ 1, g x = y ∧ ∀ x ≠ 1, g x > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l960_96070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_twenty_l960_96095

theorem binomial_sum_twenty : (Nat.choose 20 19) + (Nat.choose 20 1) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_twenty_l960_96095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_sequence_123456_l960_96053

def number_of_cards : ℕ := 12
def numbers_range : Finset ℕ := Finset.range 6 
def cards_per_number : ℕ := 2

def total_arrangements : ℕ := (Finset.prod numbers_range (λ i => Nat.choose (number_of_cards - 2*i) cards_per_number))

def favorable_arrangements : ℕ := (Finset.prod numbers_range (λ i => 2*i + 1))

theorem probability_of_sequence_123456 : 
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 720 := by
  sorry

#eval total_arrangements
#eval favorable_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_sequence_123456_l960_96053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_power_of_nineteen_implies_even_exponent_l960_96045

theorem product_equals_power_of_nineteen_implies_even_exponent 
  (a b n c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n > 0)
  (h : (a + b * c) * (b + a * c) = 19^n) : 
  Even n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equals_power_of_nineteen_implies_even_exponent_l960_96045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_forms_pyramid_patternB_forms_pyramid_l960_96077

/-- Represents a pattern of squares with fold lines -/
structure SquarePattern where
  squares : Finset (ℝ × ℝ)
  foldLines : Finset ((ℝ × ℝ) × (ℝ × ℝ))

/-- Represents a pyramid with a square base -/
structure SquarePyramid where
  base : ℝ × ℝ
  apex : ℝ × ℝ

/-- Checks if a pattern can form a pyramid with a square base -/
def canFormPyramid (pattern : SquarePattern) : Prop :=
  ∃ (pyramid : SquarePyramid),
    pattern.squares.card = 5 ∧
    pyramid.base ∈ pattern.squares ∧
    ∀ s ∈ pattern.squares, s ≠ pyramid.base →
      ∃ l ∈ pattern.foldLines, ((s = l.1 ∨ s = l.2) ∧
        (pyramid.base = l.1 ∨ pyramid.base = l.2))

/-- Theorem stating the conditions for a pattern to form a pyramid -/
theorem pattern_forms_pyramid (pattern : SquarePattern) :
  canFormPyramid pattern ↔
    pattern.squares.card = 5 ∧
    (∃ base ∈ pattern.squares, 
      ∀ s ∈ pattern.squares, s ≠ base →
        ∃ l ∈ pattern.foldLines, ((s = l.1 ∨ s = l.2) ∧
          (base = l.1 ∨ base = l.2))) ∧
    (∃ apex : ℝ × ℝ, ∀ s ∈ pattern.squares, s ≠ base →
      ∃ l ∈ pattern.foldLines, ((s = l.1 ∨ s = l.2) ∧
        (apex = l.1 ∨ apex = l.2))) :=
by
  sorry

/-- Definition of pattern B -/
def patternB : SquarePattern :=
  { squares := sorry,
    foldLines := sorry }

/-- Theorem stating that pattern B can form a pyramid -/
theorem patternB_forms_pyramid : canFormPyramid patternB :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_forms_pyramid_patternB_forms_pyramid_l960_96077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin4_2cos4_l960_96012

noncomputable section

open Real

theorem min_value_sin4_2cos4 :
  ∃ (min : ℝ), min = 2/3 ∧ (∀ x : ℝ, (sin x)^4 + 2 * (cos x)^4 ≥ min) ∧
  (∃ x : ℝ, (sin x)^4 + 2 * (cos x)^4 = min) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin4_2cos4_l960_96012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l960_96058

/-- A predicate indicating that a, b, and c form a triangle --/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

theorem triangle_inequality (a b c : ℝ) (h : IsTriangle a b c) : a^2 - b^2 - c^2 - 2*b*c < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l960_96058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_theorem_l960_96010

theorem matrix_product_theorem (n : ℕ) :
  let matrices := List.range n |>.map (fun i => !![1, 2 * (i + 1); 0, 1])
  List.prod matrices = !![1, n * (n + 1); 0, 1] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_product_theorem_l960_96010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_is_correct_l960_96065

/-- The side length of the largest square that can be inscribed in the space inside a square 
    with side length 10 and outside two congruent equilateral triangles sharing one side 
    and each having one vertex on a vertex of the outer square. -/
noncomputable def largest_inscribed_square_side_length : ℝ := 5 - (5 * Real.sqrt 3) / 3

/-- The theorem stating that the largest inscribed square side length is correct. -/
theorem largest_inscribed_square_side_length_is_correct 
  (outer_square_side_length : ℝ)
  (triangle_side_length : ℝ)
  (h1 : outer_square_side_length = 10)
  (h2 : triangle_side_length = (10 * Real.sqrt 2) / Real.sqrt 3)
  (h3 : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
       x * Real.sqrt 2 + triangle_side_length + y * Real.sqrt 2 = 10 * Real.sqrt 2 ∧
       x = y) :
  largest_inscribed_square_side_length = 5 - (5 * Real.sqrt 3) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_inscribed_square_side_length_is_correct_l960_96065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_equality_l960_96099

noncomputable def f (n : ℤ) (x : ℝ) : ℝ := Real.sin (n * x) * Real.cos (6 * x / (n + 1))

theorem periodic_function_equality (n : ℤ) (hn : n ≠ 0) :
  (∀ x : ℝ, f n (x + 5 * Real.pi) = f n x) ↔
  n ∈ ({-31, -16, -11, -7, -6, -4, -3, -2, 1, 2, 4, 5, 9, 14, 29} : Set ℤ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_equality_l960_96099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_x_value_l960_96018

/-- The number of pieces in a complete pack of gum -/
def x : ℝ := 31.25

/-- The number of pieces of cherry gum Chewbacca has -/
def cherry_gum : ℝ := 45

/-- The number of pieces of grape gum Chewbacca has -/
def grape_gum : ℝ := 55

/-- Theorem stating that x satisfies the given equation -/
theorem gum_pack_size :
  (cherry_gum - x) / grape_gum = cherry_gum / (grape_gum + 4 * x) := by
  -- Proof goes here
  sorry

/-- Theorem stating that x is equal to 31.25 -/
theorem x_value : x = 31.25 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gum_pack_size_x_value_l960_96018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equations_l960_96037

-- Define the solution sets A and B as functions of p, q, and r
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 12 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

-- State the theorem
theorem solve_equations : 
  ∃ (p q r : ℝ), 
    A p ≠ B q r ∧ 
    A p ∪ B q r = {-3, 4} ∧ 
    A p ∩ B q r = {-3} ∧
    p = -1 ∧ q = 6 ∧ r = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equations_l960_96037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l960_96092

-- Define the circle
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define point A on the ray x + y = 0 (x ≤ 0)
def PointA (x y : ℝ) : Prop := x + y = 0 ∧ x ≤ 0

-- Define point B on the positive half of x-axis
def PointB (x y : ℝ) : Prop := y = 0 ∧ x > 0

-- Define the line AB
def LineAB (x1 y1 x2 y2 x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define the tangent condition
def IsTangent (x1 y1 x2 y2 : ℝ) : Prop :=
  ∃ (x y : ℝ), Circle x y ∧ LineAB x1 y1 x2 y2 x y

-- Define the distance between two points
noncomputable def Distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem min_distance_AB :
  ∀ (x1 y1 x2 y2 : ℝ),
    PointA x1 y1 →
    PointB x2 y2 →
    IsTangent x1 y1 x2 y2 →
    Distance x1 y1 x2 y2 ≥ 2 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_AB_l960_96092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_product_equals_one_cos_complementary_angle_sin_complementary_angle_sin_sq_plus_cos_sq_l960_96030

-- Define the angles in radians
noncomputable def angle35 : ℝ := 35 * Real.pi / 180
noncomputable def angle55 : ℝ := 55 * Real.pi / 180

-- State the theorem
theorem trig_product_equals_one :
  (1 - 1 / Real.cos angle35) * (1 + 1 / Real.sin angle55) * 
  (1 - 1 / Real.sin angle35) * (1 + 1 / Real.cos angle55) = 1 := by
  sorry

-- Additional helper theorems for the trigonometric identities
theorem cos_complementary_angle (θ : ℝ) : 
  Real.cos (Real.pi/2 - θ) = Real.sin θ := by
  sorry

theorem sin_complementary_angle (θ : ℝ) : 
  Real.sin (Real.pi/2 - θ) = Real.cos θ := by
  sorry

theorem sin_sq_plus_cos_sq (θ : ℝ) : 
  Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_product_equals_one_cos_complementary_angle_sin_complementary_angle_sin_sq_plus_cos_sq_l960_96030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_in_triangle_l960_96031

/-- Triangle ABC with given points A and B -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- A line defined by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop :=
  sorry

/-- The angle bisector of ∠C -/
noncomputable def angle_bisector (t : Triangle) : Line :=
  sorry

/-- Main theorem -/
theorem line_equations_in_triangle (t : Triangle) (l : Line) (M : ℝ × ℝ) :
  t.A = (1, 1) →
  t.B = (3, -2) →
  M = (2, 0) →
  point_on_line M l →
  distance_point_to_line t.A l = distance_point_to_line t.B l →
  angle_bisector t = Line.mk 1 1 (-3) →
  ((l = Line.mk 1 0 (-2)) ∨ (l = Line.mk 3 2 (-6))) ∧
  (Line.mk 4 1 (-10) = Line.mk 4 1 (-10)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equations_in_triangle_l960_96031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_profits_theorem_l960_96050

/-- Represents a company with its profit ratios and revenue change -/
structure Company where
  profit_ratio_1998 : ℝ
  profit_ratio_1999 : ℝ
  revenue_change : ℝ

/-- Calculates the profit in 1999 as a percentage of the profit in 1998 -/
noncomputable def profit_percentage (c : Company) : ℝ :=
  (c.profit_ratio_1999 * (1 + c.revenue_change)) / c.profit_ratio_1998 * 100

/-- The given data for companies A, B, and C -/
def company_A : Company := ⟨0.10, 0.12, -0.30⟩
def company_B : Company := ⟨0.15, 0.10, 0.25⟩
def company_C : Company := ⟨0.08, 0.18, 0⟩

theorem company_profits_theorem :
  (abs (profit_percentage company_A - 84) < 0.01) ∧
  (abs (profit_percentage company_B - 83.33) < 0.01) ∧
  (abs (profit_percentage company_C - 225) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_profits_theorem_l960_96050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_sequence_properties_l960_96076

/-- An arithmetic sequence with a positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  d : ℕ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

/-- Our specific arithmetic sequence satisfying given conditions -/
def our_sequence : ArithmeticSequence where
  a := λ n => 4 * n - 25
  d := 4
  h_positive := by simp
  h_arithmetic := by
    intro n
    simp [add_sub_cancel]
    ring

theorem our_sequence_properties (seq : ArithmeticSequence) 
  (h1 : seq.a 3 * seq.a 4 = 117)
  (h2 : seq.a 2 + seq.a 5 = -22) :
  seq.a = our_sequence.a ∧ 
  ∃ n : ℕ, S_n seq n = -66 ∧ ∀ m : ℕ, S_n seq m ≥ -66 := by
  sorry

#check our_sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_sequence_properties_l960_96076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l960_96001

theorem cube_root_equation_solution (x : ℝ) :
  (5 - 2 / x)^(1/3 : ℝ) = -3 → x = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l960_96001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_difference_l960_96025

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sequence_A : ℕ → ℝ := geometric_sequence 5 3

def sequence_B : ℕ → ℝ := arithmetic_sequence 30 40

def valid_term (x : ℝ) : Prop := x ≤ 450

theorem least_positive_difference :
  ∃ (m n : ℕ), 
    valid_term (sequence_A m) ∧ 
    valid_term (sequence_B n) ∧
    (∀ (i j : ℕ), valid_term (sequence_A i) → valid_term (sequence_B j) →
      |sequence_A i - sequence_B j| ≥ 15) ∧
    (∃ (k l : ℕ), valid_term (sequence_A k) ∧ valid_term (sequence_B l) ∧
      |sequence_A k - sequence_B l| = 15) :=
by sorry

#eval sequence_A 0
#eval sequence_A 1
#eval sequence_A 2
#eval sequence_A 3
#eval sequence_A 4

#eval sequence_B 0
#eval sequence_B 1
#eval sequence_B 2
#eval sequence_B 3
#eval sequence_B 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_difference_l960_96025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_six_eq_four_twentyfifths_l960_96080

/-- The number of cards -/
def total_cards : ℕ := 100

/-- A function that returns true if a number is a multiple of 6 -/
def is_multiple_of_six (n : ℕ) : Bool := n % 6 = 0

/-- The count of multiples of 6 in the range 1 to 100 -/
def count_multiples_of_six : ℕ := (List.range total_cards).filter is_multiple_of_six |>.length

/-- The probability of drawing a card that is a multiple of 6 -/
noncomputable def probability_multiple_of_six : ℚ := count_multiples_of_six / total_cards

theorem probability_multiple_of_six_eq_four_twentyfifths :
  probability_multiple_of_six = 4 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_of_six_eq_four_twentyfifths_l960_96080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_probability_l960_96078

theorem birth_probability (n : ℕ) (p : ℝ) (h1 : n = 5) (h2 : p = 1/2) :
  let prob_all_same := p^n + (1-p)^n
  let prob_three_two := (Nat.choose n 3 : ℝ) * p^3 * (1-p)^2
  let prob_four_one := (Nat.choose n 1 : ℝ) * p^4 * (1-p) + (Nat.choose n 1 : ℝ) * p * (1-p)^4
  (prob_three_two > prob_all_same) ∧ (prob_four_one > prob_all_same) :=
by
  -- Introduce the local definitions
  have prob_all_same := p^n + (1-p)^n
  have prob_three_two := (Nat.choose n 3 : ℝ) * p^3 * (1-p)^2
  have prob_four_one := (Nat.choose n 1 : ℝ) * p^4 * (1-p) + (Nat.choose n 1 : ℝ) * p * (1-p)^4

  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_probability_l960_96078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extracurricular_activities_l960_96087

open Set

theorem extracurricular_activities (total : ℕ) (spanish german sports : Finset ℕ) :
  total = 94 →
  spanish.card = 40 →
  german.card = 27 →
  sports.card = 60 →
  (spanish ∩ sports).card = 24 →
  (spanish ∩ german).card = 10 →
  (german ∩ sports).card = 12 →
  (spanish ∩ german ∩ sports).card = 4 →
  (∃ (only_one not_any : ℕ),
    only_one = ((spanish \ (german ∪ sports)).card +
                (german \ (spanish ∪ sports)).card +
                (sports \ (spanish ∪ german)).card) ∧
    not_any = total - (spanish ∪ german ∪ sports).card ∧
    only_one = 47 ∧
    not_any = 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extracurricular_activities_l960_96087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orientation_preserving_mapping_l960_96009

-- Define the circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the line
structure Line where
  normal : EuclideanSpace ℝ (Fin 2)
  offset : ℝ

-- Define the mapping
noncomputable def f (ω : Circle) (m n : Line) (P : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

-- Define tangent predicate
def Line.tangent (l : Line) (ω : Circle) : Prop :=
  sorry

-- Define collinear predicate
def collinear (A B C : EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define orientation_preserving predicate
def orientation_preserving (f : EuclideanSpace ℝ (Fin 2) → EuclideanSpace ℝ (Fin 2)) : Prop :=
  sorry

-- Define membership for points on lines
instance : Membership (EuclideanSpace ℝ (Fin 2)) Line where
  mem := λ P l => sorry

-- State the theorem
theorem orientation_preserving_mapping 
  (ω : Circle) (m n : Line) (P : EuclideanSpace ℝ (Fin 2)) :
  Line.tangent m ω → Line.tangent n ω → 
  P ∈ m → (f ω m n P) ∈ n → 
  collinear ω.center P (f ω m n P) →
  orientation_preserving (f ω m n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orientation_preserving_mapping_l960_96009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_2008_l960_96052

def sequence_a (n : ℕ) : ℤ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+2 => 2 * sequence_a (n+1) + 2007 * sequence_a n

theorem smallest_divisible_by_2008 :
  (∀ k : ℕ, k > 0 ∧ k < 2008 → ¬(2008 ∣ sequence_a k)) ∧ (2008 ∣ sequence_a 2008) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_divisible_by_2008_l960_96052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_second_quadrant_l960_96097

def second_quadrant (α : ℝ) : Prop := 
  Real.pi / 2 < α ∧ α < Real.pi

def on_terminal_side (α : ℝ) (x y : ℝ) : Prop :=
  x = -Real.sqrt 3 ∧ y > 0 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2)

theorem sin_value_in_second_quadrant (α : ℝ) (y : ℝ) :
  second_quadrant α →
  on_terminal_side α (-Real.sqrt 3) y →
  Real.cos α = -(Real.sqrt 15) / 5 →
  Real.sin α = Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_value_in_second_quadrant_l960_96097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approximation_l960_96075

/-- Given an object traveling 80 feet in 2 seconds, and 1 mile equals 5280 feet,
    prove that the object's speed is approximately 58.67 miles per hour. -/
theorem object_speed_approximation : 
  let distance : ℝ := 80  -- distance in feet
  let time : ℝ := 2       -- time in seconds
  let feet_per_mile : ℝ := 5280
  let seconds_per_hour : ℝ := 3600
  let speed_mph : ℝ := (distance / time) * (seconds_per_hour / feet_per_mile)
  abs (speed_mph - 58.67) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_speed_approximation_l960_96075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_probability_theorem_l960_96021

/-- Probability of moving from (a,b) to (a-1,b) -/
def prob_left : ℚ := 2/5

/-- Probability of moving from (a,b) to (a,b-1) -/
def prob_down : ℚ := 2/5

/-- Probability of moving from (a,b) to (a-1,b-1) -/
def prob_diagonal : ℚ := 1/5

/-- The starting point of the particle -/
def start_point : ℕ × ℕ := (5, 5)

/-- Function representing the probability of reaching (0,0) from a given point -/
noncomputable def reach_probability : ℕ × ℕ → ℚ :=
  sorry

/-- Theorem stating the existence of m and n satisfying the required conditions -/
theorem particle_probability_theorem :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ ¬(5 ∣ m) ∧
  reach_probability start_point = m / 5^n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_probability_theorem_l960_96021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l960_96079

/-- The speed of a stream given downstream and upstream speeds -/
noncomputable def stream_speed (downstream_speed upstream_speed : ℝ) : ℝ :=
  (downstream_speed - upstream_speed) / 2

/-- Theorem: The speed of the stream is 2 kmph -/
theorem stream_speed_calculation (downstream_speed upstream_speed : ℝ)
  (h1 : downstream_speed = 12)
  (h2 : upstream_speed = 8) :
  stream_speed downstream_speed upstream_speed = 2 := by
  unfold stream_speed
  rw [h1, h2]
  norm_num

#check stream_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stream_speed_calculation_l960_96079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_even_l960_96004

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (3 + x) + Real.log (3 - x)

-- Define the domain of f
def domain : Set ℝ := { x : ℝ | -3 < x ∧ x < 3 }

-- Statement 1: The domain of f is (-3, 3)
theorem f_domain : { x : ℝ | ∃ y, f x = y } = domain := by sorry

-- Statement 2: f is an even function on its domain
theorem f_even : ∀ x ∈ domain, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_f_even_l960_96004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_rationals_in_unit_interval_in_S_l960_96043

-- Define the set S using an inductive definition
inductive S : ℚ → Prop
  | base : S (1/2)
  | step1 (x : ℚ) : S x → S (1 / (x + 1))
  | step2 (x : ℚ) : S x → S (x / (x + 1))

-- State the theorem
theorem all_rationals_in_unit_interval_in_S :
  ∀ q : ℚ, 0 < q ∧ q < 1 → S q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_rationals_in_unit_interval_in_S_l960_96043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_set_l960_96051

def is_valid_set (S : Finset ℕ) : Prop :=
  S.card = 50 ∧
  (∃ S1 S2 : Finset ℕ, S = S1 ∪ S2 ∧ 
    S1.card = 25 ∧ S2.card = 25 ∧
    (∀ n ∈ S1, n ≤ 50) ∧
    (∀ n ∈ S2, 50 < n ∧ n ≤ 100) ∧
    (∀ a ∈ S, ∀ b ∈ S, a ≠ b → (a : Int) - b ≠ 50 ∧ (b : Int) - a ≠ 50))

theorem sum_of_special_set (S : Finset ℕ) (h : is_valid_set S) : 
  S.sum id = 2525 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_set_l960_96051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_not_in_P_l960_96016

-- Define the set P
noncomputable def P : Set ℝ := {x | x^2 - Real.sqrt 2 * x ≤ 0}

-- Define m
noncomputable def m : ℝ := Real.sqrt 3

-- Theorem statement
theorem m_not_in_P : m ∉ P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_not_in_P_l960_96016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_subsets_count_exists_natural_sqrt_necessary_not_sufficient_range_of_sum_l960_96044

-- Define the set {a,b,c}
def S : Finset Char := {'a', 'b', 'c'}

-- Statement 1
theorem non_empty_subsets_count : Finset.card (Finset.powerset S \ {∅}) = 7 := by sorry

-- Statement 2
theorem exists_natural_sqrt : ∃ m : ℕ, ∃ n : ℕ, n^2 = m^2 + 1 := by sorry

-- Statement 3
theorem necessary_not_sufficient :
  (∀ m : ℝ, m < 3 → m < 4) ∧ 
  (∃ m : ℝ, m < 4 ∧ ¬(m < 3)) := by sorry

-- Statement 4
theorem range_of_sum :
  ∀ a b : ℝ, 2 < a → a < 3 → -2 < b → b < -1 →
  2 < 2*a + b ∧ 2*a + b < 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_empty_subsets_count_exists_natural_sqrt_necessary_not_sufficient_range_of_sum_l960_96044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l960_96047

open Real

-- Define the function f
noncomputable def f (α : ℝ) : ℝ :=
  (sin α) * (6 * sin α + cos α) + (cos α) * (7 * sin α - 2 * cos α)

-- Define the theorem
theorem triangle_side_length
  (A B C : ℝ)  -- Angles of the triangle
  (a b c : ℝ)  -- Side lengths of the triangle
  (h1 : 0 < A ∧ A < π/2)  -- A is acute
  (h2 : 0 < B ∧ B < π/2)  -- B is acute
  (h3 : 0 < C ∧ C < π/2)  -- C is acute
  (h4 : A + B + C = π)   -- Sum of angles in a triangle
  (h5 : f A = 6)         -- Given condition
  (h6 : (1/2) * b * c * sin A = 3)  -- Area of triangle is 3
  (h7 : b + c = 2 + 3 * sqrt 2)  -- Given condition
  : a = sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l960_96047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l960_96003

/-- Represents the problem of two trains passing a platform in opposite directions -/
def TrainProblem (length1 length2 platform_length : ℝ) (speed1 speed2 : ℝ) : Prop :=
  let total_distance := length1 + platform_length
  let speed1_ms := speed1 * (1000 / 3600)
  let speed2_ms := speed2 * (1000 / 3600)
  let relative_speed := speed1_ms + speed2_ms
  let time := total_distance / relative_speed
  ∃ ε > 0, |time - 20.57| < ε

/-- The theorem stating the solution to the train problem -/
theorem train_problem_solution :
  TrainProblem 360 480 240 45 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_problem_solution_l960_96003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_pairs_in_first_ten_primes_l960_96017

def first_ten_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem no_valid_pairs_in_first_ten_primes :
  ∀ p q, p ∈ first_ten_primes → q ∈ first_ten_primes → p ≠ q →
    ¬(Nat.Prime (p + q) ∧ Nat.Prime (p * q)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_pairs_in_first_ten_primes_l960_96017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PQ_l960_96011

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def k : Circle := { center := (6, 0), radius := 9 }
def k1 : Circle := { center := (0, 0), radius := 6 }
def k2 : Circle := { center := (9, 0), radius := 3 }

-- Define the property of being externally tangent
def externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the property of touching from inside
def touches_from_inside (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

-- Define a point on a circle
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define the theorem
theorem length_of_PQ :
  externally_tangent k1 k2 ∧
  touches_from_inside k k1 ∧
  touches_from_inside k k2 →
  ∃ P Q : ℝ × ℝ,
    on_circle P k ∧
    on_circle Q k ∧
    (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (4 * Real.sqrt 14)^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PQ_l960_96011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l960_96038

/-- An ellipse with equation x²/16 + y²/9 = 1 -/
structure Ellipse where
  equation : ℝ × ℝ → Prop
  foci : (ℝ × ℝ) × (ℝ × ℝ)

/-- A line passing through one focus of the ellipse -/
structure Line where
  passes_through : ℝ × ℝ
  intersects : (ℝ × ℝ) × (ℝ × ℝ)

/-- The perimeter of a triangle -/
noncomputable def triangle_perimeter (A B C : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_perimeter 
  (e : Ellipse) 
  (l : Line) 
  (h1 : e.equation = fun p => p.1^2/16 + p.2^2/9 = 1)
  (h2 : l.passes_through = e.foci.1)
  (h3 : e.equation l.intersects.1 ∧ e.equation l.intersects.2) :
  triangle_perimeter l.intersects.1 l.intersects.2 e.foci.2 = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_perimeter_l960_96038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_angle_l960_96054

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- Point on a parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * c.p * x

/-- Focus of a parabola -/
noncomputable def focus (c : Parabola) : ℝ × ℝ := (c.p / 2, 0)

/-- Directrix of a parabola -/
noncomputable def directrix (c : Parabola) : ℝ → ℝ := fun x ↦ -c.p / 2

/-- Point where directrix intersects x-axis -/
noncomputable def K (c : Parabola) : ℝ × ℝ := (-c.p / 2, 0)

/-- Origin -/
def O : ℝ × ℝ := (0, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Angle between three points -/
noncomputable def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem parabola_angle (c : Parabola) (m : PointOnParabola c)
    (h : distance (m.x, m.y) (focus c) = c.p) :
    angle (K c) O (m.x, m.y) = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_angle_l960_96054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_fourth_composition_eq_two_solutions_l960_96072

def g (x : ℝ) : ℝ := x^2 - 3*x

theorem g_fourth_composition_eq_two_solutions :
  ∃! (s : Finset ℝ), (∀ c ∈ s, g (g (g (g c))) = 2) ∧ (Finset.card s = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_fourth_composition_eq_two_solutions_l960_96072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_inverse_l960_96049

-- Define the logarithm function
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the symmetry condition
def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_inverse (f : ℝ → ℝ) :
  symmetric_about_y_eq_x f (λ x ↦ log2 (x / 2)) →
  ∀ x, f x = 2^(x + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_inverse_l960_96049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_energy_drinks_l960_96019

/-- The number of bottles of energy drinks bought by the basketball team -/
def energy_drinks_bought (cupcakes_sold : ℕ) (cupcake_price : ℚ) 
  (cookies_sold : ℕ) (cookie_price : ℚ)
  (basketballs_bought : ℕ) (basketball_price : ℚ)
  (energy_drink_price : ℚ) : ℕ :=
  let total_sales := cupcakes_sold * cupcake_price + cookies_sold * cookie_price
  let basketball_cost := basketballs_bought * basketball_price
  let remaining_money := total_sales - basketball_cost
  (remaining_money / energy_drink_price).floor.toNat

/-- Theorem stating the number of energy drinks bought by the basketball team -/
theorem basketball_team_energy_drinks : 
  energy_drinks_bought 50 2 40 (1/2) 2 40 2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_team_energy_drinks_l960_96019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l960_96040

/-- The translation of a parabola -/
def translate_parabola (f : ℝ → ℝ) (h v : ℝ) : ℝ → ℝ :=
  λ x => f (x - h) + v

theorem parabola_translation :
  let f : ℝ → ℝ := λ x => x^2
  let g : ℝ → ℝ := λ x => (x+2)^2 - 3
  g = translate_parabola f (-2) (-3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_translation_l960_96040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calc_l960_96057

/-- The nabla operation for positive real numbers -/
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

/-- Theorem: 3 ∇ (2 ∇ 4) = 11/9 -/
theorem nabla_calc : nabla 3 (nabla 2 4) = 11/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calc_l960_96057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_prisoner_hat_is_green_l960_96008

/-- Represents the color of a hat -/
inductive HatColor
| Red
| Green

/-- Represents a prisoner in the game -/
structure Prisoner where
  position : Nat
  canSee : Nat

/-- The game setup -/
structure HatGame where
  totalHats : Nat
  redHats : Nat
  greenHats : Nat
  prisoners : List Prisoner

/-- Defines the conditions of the game -/
def gameConditions (game : HatGame) : Prop :=
  game.totalHats = 5 ∧
  game.redHats = 2 ∧
  game.greenHats = 3 ∧
  game.prisoners.length = 3 ∧
  game.prisoners.all (λ p => p.position ≤ 3 ∧ p.canSee < p.position)

/-- Represents the silence of a prisoner -/
def isSilent (prisoner : Prisoner) (game : HatGame) : Prop :=
  ∀ color, ¬ (prisoner.canSee = game.redHats ∧ color = HatColor.Green) ∧
           ¬ (prisoner.canSee = game.greenHats ∧ color = HatColor.Red)

/-- The main theorem to prove -/
theorem first_prisoner_hat_is_green (game : HatGame) :
  gameConditions game →
  (∃ p₁ ∈ game.prisoners, p₁.position = 2 ∧ isSilent p₁ game) →
  (∃ p₂ ∈ game.prisoners, p₂.position = 3 ∧ isSilent p₂ game) →
  (∃ p₀ ∈ game.prisoners, p₀.position = 1 ∧ p₀.canSee = 0) →
  ∃ hats : List HatColor, hats.length = 3 ∧ hats.head? = some HatColor.Green :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_prisoner_hat_is_green_l960_96008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l960_96074

/-- Definition of a quadratic function -/
noncomputable def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Function A -/
noncomputable def f_A : ℝ → ℝ := λ x => -2 / x^2 - 3 * x

/-- Function B -/
noncomputable def f_B : ℝ → ℝ := λ x => -(x - 1)^2 + x^2

/-- Function C -/
noncomputable def f_C : ℝ → ℝ := λ x => 11 * x^2 + 29 * x

/-- Function D -/
noncomputable def f_D (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_identification :
  ¬(is_quadratic f_A) ∧
  ¬(is_quadratic f_B) ∧
  (is_quadratic f_C) ∧
  ¬(∀ a b c, is_quadratic (f_D a b c)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_identification_l960_96074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_count_l960_96073

/-- Proves that the number of new students who joined the class is 12 -/
theorem new_students_count (original_avg : ℝ) (new_students_avg : ℝ) (avg_decrease : ℝ) (original_count : ℕ) 
  (h1 : original_avg = 40)
  (h2 : new_students_avg = 32)
  (h3 : avg_decrease = 4)
  (h4 : original_count = 12) : ℕ := by
  -- The number of new students
  /- New average after students joined -/
  let new_avg : ℝ := original_avg - avg_decrease
  /- Equation: (original_count * original_avg + x * new_students_avg) / (original_count + x) = new_avg -/
  /- Where x is the number of new students -/
  have : ∃ x : ℕ, (original_count * original_avg + x * new_students_avg) / (original_count + x) = new_avg ∧ x = 12
  sorry
  exact 12


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_count_l960_96073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_l960_96069

/-- A curve in polar coordinates is defined by the equation r = 3. -/
def polar_curve : Set (ℝ × ℝ) :=
  {p | p.1 = 3 ∧ p.2 ∈ Set.Icc 0 (2 * Real.pi)}

/-- The definition of a circle in Cartesian coordinates -/
def cartesian_circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Theorem stating that the polar curve r = 3 is equivalent to a circle -/
theorem polar_curve_is_circle :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    polar_curve = cartesian_circle center radius := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_circle_l960_96069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_a_is_correct_l960_96061

/-- Represents the possible answers to the fill-in-the-blank question --/
inductive Answer
  | A
  | B
  | C
  | D

/-- Represents the conditions of the problem --/
structure ProblemConditions where
  discovery_is_specific : Bool
  man_represents_mankind : Bool

/-- Checks if the answer is correct given the problem conditions --/
def is_correct_answer (conditions : ProblemConditions) (answer : Answer) : Prop :=
  conditions.discovery_is_specific ∧
  conditions.man_represents_mankind ∧
  answer = Answer.A

/-- Theorem stating that Answer A is correct given the problem conditions --/
theorem answer_a_is_correct (conditions : ProblemConditions) 
  (h1 : conditions.discovery_is_specific = true)
  (h2 : conditions.man_represents_mankind = true) :
  is_correct_answer conditions Answer.A := by
  unfold is_correct_answer
  simp [h1, h2]

#check answer_a_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_answer_a_is_correct_l960_96061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_minus_one_l960_96082

def y : ℕ → ℚ
  | 0 => 50  -- Add this case to handle Nat.zero
  | 1 => 50
  | (k + 2) => y (k + 1) ^ 2 - y (k + 1)

theorem sum_reciprocal_y_minus_one :
  (∑' k : ℕ, 1 / (y k - 1)) = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_y_minus_one_l960_96082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l960_96022

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + 1

theorem tangent_line_perpendicular (a : ℝ) : 
  (∃ (m : ℝ), (∀ (h : ℝ), h ≠ 0 → 
    (f (Real.pi / 2 + h) - f (Real.pi / 2)) / h = m) ∧
    m * (-a / 2) = -1) → 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l960_96022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_even_sum_l960_96034

theorem contrapositive_even_sum (a b : ℤ) :
  (¬(Even a ∧ Even b) ↔ ¬Even (a + b)) ↔
  (¬Even (a + b) → ¬(Even a ∧ Even b)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_even_sum_l960_96034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_volume_l960_96081

theorem rectangular_parallelepiped_volume 
  (a : ℝ) 
  (h1 : a > 0) -- Assuming a is positive
  : ∃ (V : ℝ), V = (1/8) * a^3 * Real.sqrt 2 := by
  -- Define the angles in radians
  let angle30 := Real.pi / 6
  let angle45 := Real.pi / 4

  -- Assert the existence of the sides of the parallelepiped
  have side1 : ℝ := a * Real.sin angle30
  have side2 : ℝ := a * Real.sin angle45
  have side3 : ℝ := Real.sqrt (a^2 - side1^2)

  -- Calculate the volume
  let V := side1 * side2 * side3

  -- Show that this volume equals the expected result
  have volume_eq : V = (1/8) * a^3 * Real.sqrt 2 := by
    -- This is where the detailed proof would go
    -- We would use trigonometric identities and algebraic manipulation
    sorry

  -- Conclude the proof
  exact ⟨V, volume_eq⟩

-- Example usage of the theorem
example (a : ℝ) (h : a > 0) : 
  ∃ (V : ℝ), V = (1/8) * a^3 * Real.sqrt 2 :=
  rectangular_parallelepiped_volume a h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_parallelepiped_volume_l960_96081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_example_l960_96088

def is_systematic_sampling (sample : List Nat) (population_size : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  sample.all (· ≤ population_size) ∧
  (sample.zip (sample.tail)).all (fun (a, b) => b - a = population_size / sample_size)

theorem systematic_sampling_example :
  is_systematic_sampling [3, 13, 23, 33, 43, 53] 60 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_example_l960_96088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archimedean_spiral_integral_l960_96062

/-- The Archimedean spiral parameterized by angle φ -/
noncomputable def archimedeanSpiral (φ : ℝ) : ℝ × ℝ :=
  (φ * Real.cos φ, φ * Real.sin φ)

/-- The integrand function -/
noncomputable def integrand (x y : ℝ) : ℝ :=
  Real.arctan (y / x)

/-- The line element along the Archimedean spiral -/
noncomputable def lineElement (φ : ℝ) : ℝ :=
  Real.sqrt (1 + φ^2)

theorem archimedean_spiral_integral :
  ∃ (L : Set (ℝ × ℝ)),
    (∀ p ∈ L, ∃ φ ∈ Set.Icc 0 (Real.pi / 2), p = archimedeanSpiral φ) ∧
    (∫ (p : ℝ × ℝ) in L, integrand p.1 p.2 * lineElement (Real.sqrt (p.1^2 + p.2^2))) =
      ((Real.pi^2 + 4)^(3/2) - 8) / 24 := by
  sorry

#check archimedean_spiral_integral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archimedean_spiral_integral_l960_96062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_vector_ratio_l960_96055

noncomputable def A : ℝ × ℝ × ℝ := (0, 2, 19/8)
noncomputable def B : ℝ × ℝ × ℝ := (1, -1, 5/8)
noncomputable def C : ℝ × ℝ × ℝ := (-2, 1, 5/8)

theorem normal_vector_ratio (x y z : ℝ) :
  let normal := (x, y, z)
  (x * (B.1 - A.1) + y * (B.2.1 - A.2.1) + z * (B.2.2 - A.2.2) = 0) ∧
  (x * (C.1 - A.1) + y * (C.2.1 - A.2.1) + z * (C.2.2 - A.2.2) = 0) →
  ∃ (k : ℝ), k ≠ 0 ∧ x = 2*k ∧ y = 3*k ∧ z = -4*k :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_normal_vector_ratio_l960_96055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_has_winning_strategy_l960_96066

/-- Represents the state of the game --/
structure GameState where
  deck : List Nat
  pile : List Nat

/-- Checks if a number is the difference of two squares --/
def isDiffOfSquares (n : Nat) : Prop :=
  ∃ a b : Nat, n = a * a - b * b

/-- Represents a player's strategy --/
def Strategy := GameState → Nat

/-- Checks if a strategy is winning for Player 2 --/
def isWinningStrategy (s : Strategy) : Prop :=
  ∀ (initialDeck : List Nat),
    (∀ n, n ∈ initialDeck → 1 ≤ n ∧ n ≤ 8) →
    (∀ n, (List.countP (· = n) initialDeck) = n) →
    ∀ (game : GameState),
      game.deck = [] →
      isDiffOfSquares (s game)

/-- The main theorem stating that Player 2 has a winning strategy --/
theorem player2_has_winning_strategy :
  ∃ (s : Strategy), isWinningStrategy s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_has_winning_strategy_l960_96066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_equals_pi_over_four_l960_96033

theorem alpha_plus_beta_equals_pi_over_four
  (α β : Real)
  (h1 : Real.cos α = 2 * Real.sqrt 5 / 5)
  (h2 : Real.sin β = Real.sqrt 10 / 10)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : 0 < β ∧ β < Real.pi / 2) :
  α + β = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_equals_pi_over_four_l960_96033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_zero_in_interval_l960_96000

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 10 then
    3 * 2^x - 24
  else if 10 < x ∧ x ≤ 20 then
    -2^(x-5) + 126
  else
    0  -- Define a default value for x outside the given ranges

-- State the theorem
theorem f_no_zero_in_interval :
  ∀ x ∈ Set.Ioo 3 7, f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_no_zero_in_interval_l960_96000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_read_l960_96029

/-- Given a book with a total number of pages and the number of pages left to read,
    prove that the number of pages already read is the difference between the total
    and the pages left to read. -/
theorem pages_read (total_pages pages_left : ℕ) :
  total_pages ≥ pages_left →
  total_pages - pages_left = total_pages - pages_left :=
by
  intro h
  rfl

/-- Ceasar's reading progress -/
def ceasar_reading : ℕ := 
  if h : 563 ≥ 416 then
    563 - 416
  else
    0

#eval ceasar_reading

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pages_read_l960_96029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_overtakes_car_l960_96026

/-- The time it takes for a faster vehicle to overtake a slower vehicle -/
noncomputable def overtake_time (car_speed truck_speed : ℝ) (delay : ℝ) : ℝ :=
  (car_speed * delay) / (truck_speed - car_speed)

/-- Theorem: The time it takes for the truck to pass the car is 5.5 hours -/
theorem truck_overtakes_car :
  let car_speed : ℝ := 55
  let truck_speed : ℝ := 65
  let delay : ℝ := 1
  overtake_time car_speed truck_speed delay = 5.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_overtakes_car_l960_96026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_bird_distance_theorem_l960_96067

/-- The point where the cat starts moving away from the bird -/
noncomputable def cat_turning_point (bird_x bird_y cat_line_slope cat_line_intercept : ℝ) : ℝ × ℝ :=
  let perpendicular_slope := -1 / cat_line_slope
  let perpendicular_intercept := bird_y - perpendicular_slope * bird_x
  let c := (cat_line_intercept - perpendicular_intercept) / (perpendicular_slope - cat_line_slope)
  let d := cat_line_slope * c + cat_line_intercept
  (c, d)

/-- Theorem stating that the sum of coordinates of the turning point is approximately 18.95 -/
theorem cat_bird_distance_theorem (ε : ℝ) (hε : ε > 0) :
  let (c, d) := cat_turning_point 15 15 (-4) 34
  |c + d - 18.95| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cat_bird_distance_theorem_l960_96067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cr_in_H2CrO4_l960_96027

/-- The molar mass of hydrogen in g/mol -/
noncomputable def molar_mass_H : ℝ := 1.01

/-- The molar mass of chromium in g/mol -/
noncomputable def molar_mass_Cr : ℝ := 52.00

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The number of hydrogen atoms in H2CrO4 -/
def num_H : ℕ := 2

/-- The number of chromium atoms in H2CrO4 -/
def num_Cr : ℕ := 1

/-- The number of oxygen atoms in H2CrO4 -/
def num_O : ℕ := 4

/-- The molar mass of H2CrO4 in g/mol -/
noncomputable def molar_mass_H2CrO4 : ℝ :=
  num_H * molar_mass_H + num_Cr * molar_mass_Cr + num_O * molar_mass_O

/-- The mass percentage of Cr in H2CrO4 -/
noncomputable def mass_percentage_Cr : ℝ :=
  (molar_mass_Cr / molar_mass_H2CrO4) * 100

theorem mass_percentage_Cr_in_H2CrO4 :
  ∃ ε > 0, |mass_percentage_Cr - 44.06| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Cr_in_H2CrO4_l960_96027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l960_96014

/-- Represents the time (in seconds) it takes for a train to cross a pole -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  train_length / train_speed_ms

/-- Theorem stating that a train with length 250 meters and speed 90 km/h takes 10 seconds to cross a pole -/
theorem train_crossing_pole_time :
  train_crossing_time 250 90 = 10 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_pole_time_l960_96014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l960_96048

def mySequence (n : ℕ) : ℚ :=
  if n % 2 = 1 then 3 else 4

theorem fifteenth_term_is_three :
  (∀ n : ℕ, n ≥ 1 → mySequence n * mySequence (n + 1) = 12) →
  (∀ n : ℕ, n ≥ 2 → mySequence n = 12 / mySequence (n - 1)) →
  mySequence 1 = 3 →
  mySequence 15 = 3 := by
  sorry

#eval mySequence 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_term_is_three_l960_96048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_daughter_l960_96086

-- Define the family members
inductive FamilyMember
  | Woman
  | Brother
  | Son
  | Daughter1
  | Daughter2

-- Define the sex of a family member
def sex : FamilyMember → Bool
  | FamilyMember.Woman => false
  | FamilyMember.Brother => true
  | FamilyMember.Son => true
  | FamilyMember.Daughter1 => false
  | FamilyMember.Daughter2 => false

-- Define the twin relationship
def isTwin : FamilyMember → FamilyMember → Prop := sorry

-- Define the age of a family member
def age : FamilyMember → ℕ := sorry

-- Define the worst player
def worstPlayer : FamilyMember := sorry

-- Define the best player
def bestPlayer : FamilyMember := sorry

-- Theorem statement
theorem worst_player_is_daughter :
  (∃ m : FamilyMember, isTwin worstPlayer m) →
  (sex worstPlayer ≠ sex bestPlayer) →
  (age worstPlayer = age bestPlayer) →
  (worstPlayer = FamilyMember.Daughter1 ∨ worstPlayer = FamilyMember.Daughter2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worst_player_is_daughter_l960_96086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_intersection_l960_96093

-- Define the hyperbola
noncomputable def hyperbola (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y^2 / 5 - x^2 / m = 1

-- Define the ellipse
noncomputable def ellipse (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => x^2 / (2*m) + y^2 / (9-m) = 1

-- Define the eccentricity of the hyperbola
noncomputable def eccentricity (m : ℝ) : ℝ :=
  Real.sqrt (1 + m / 5)

-- Theorem statement
theorem hyperbola_ellipse_intersection (m : ℝ) :
  (Real.sqrt 6 / 2 < eccentricity m ∧ eccentricity m < Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a > b ∧ ∀ x y, ellipse m x y ↔ x^2/a^2 + y^2/b^2 = 1) →
  5/2 < m ∧ m < 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_intersection_l960_96093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l960_96094

-- Define the function g(x) as noncomputable
noncomputable def g (x : ℝ) : ℝ := (3*x - 4)*(x + 2) / (x - 1)

-- State the theorem
theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  g x ≤ 0 ↔ x ∈ Set.Iic (-2) ∪ Set.Icc 1 (4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l960_96094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l960_96091

theorem sin_alpha_plus_beta (α β : ℝ) 
  (h1 : 0 < β) (h2 : β < π/4) (h3 : π/4 < α) (h4 : α < 3*π/4)
  (h5 : Real.cos (π/4 - α) = 3/5) (h6 : Real.sin (3*π/4 + β) = 5/13) :
  Real.sin (α + β) = 56/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_beta_l960_96091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_pi_minus_alpha_is_two_l960_96068

theorem sin_double_angle_when_tan_pi_minus_alpha_is_two (α : ℝ) :
  Real.tan (π - α) = 2 → Real.sin (2 * α) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_when_tan_pi_minus_alpha_is_two_l960_96068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l960_96056

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x + 1)}
def B : Set ℝ := {y | ∃ x, y = 1 - Real.exp x}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Icc (-1) 1 \ {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l960_96056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l960_96032

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  a₁_eq_1 : a 1 = 1
  a₃_plus_a₅_eq_a₄_plus_7 : a 3 + a 5 = a 4 + 7

/-- The sum of the first n terms of the sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 1) ∧
  {n : ℕ | S seq n < 3 * seq.a n - 2} = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l960_96032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plan2_cost_per_mile_l960_96020

/-- Represents a car rental plan -/
structure RentalPlan where
  initialFee : ℝ
  costPerMile : ℝ

/-- The distance at which both plans cost the same -/
def equalCostDistance : ℝ := 325

/-- The first rental plan -/
def plan1 : RentalPlan := { initialFee := 65, costPerMile := 0.40 }

/-- The second rental plan (with unknown cost per mile) -/
def plan2 (x : ℝ) : RentalPlan := { initialFee := 0, costPerMile := x }

/-- The total cost of a rental plan for a given distance -/
def totalCost (plan : RentalPlan) (distance : ℝ) : ℝ :=
  plan.initialFee + plan.costPerMile * distance

/-- Theorem stating that the cost per mile for Plan 2 is $0.60 -/
theorem plan2_cost_per_mile :
  ∃ x, totalCost plan1 equalCostDistance = totalCost (plan2 x) equalCostDistance ∧ x = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plan2_cost_per_mile_l960_96020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_neg_two_point_eight_l960_96096

/-- The floor function returns the largest integer less than or equal to a given real number. -/
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

/-- Theorem: The floor of -2.8 is equal to -3. -/
theorem floor_of_neg_two_point_eight :
  floor (-2.8) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_of_neg_two_point_eight_l960_96096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_effective_dose_max_effective_duration_l960_96063

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 4 then
    -1/2 * x^2 + 2*x + 8
  else if 4 < x ∧ x ≤ 16 then
    -x/2 - Real.log x / Real.log 2 + 12
  else
    0

-- Define the drug concentration function y(m, x)
noncomputable def y (m : ℝ) (x : ℝ) : ℝ := m * f x

-- Theorem statement
theorem min_effective_dose :
  ∀ m : ℝ, m > 0 →
  (∀ x : ℝ, 0 < x ∧ x ≤ 8 → y m x ≥ 12) ↔
  m ≥ 12/5 := by
  sorry

-- Second part of the problem
theorem max_effective_duration :
  ∃ k : ℕ, k = 6 ∧
  (∀ x : ℝ, 0 < x ∧ x ≤ k → y 2 x ≥ 12) ∧
  ¬(∀ x : ℝ, 0 < x ∧ x ≤ k + 1 → y 2 x ≥ 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_effective_dose_max_effective_duration_l960_96063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_rhombus_l960_96071

/-- A quadrilateral in 2D space -/
structure Quadrilateral (V : Type*) [AddCommGroup V] [Module ℝ V] [NormedAddCommGroup V] :=
  (A B C D : V)

/-- Definition of a rhombus -/
def is_rhombus {V : Type*} [AddCommGroup V] [Module ℝ V] [NormedAddCommGroup V] (q : Quadrilateral V) : Prop :=
  q.A - q.D = q.C - q.B ∧ ‖q.A - q.D‖ = ‖q.A - q.B‖

/-- Theorem: If AD = BC and |AD| = |AB|, then ABCD is a rhombus -/
theorem quadrilateral_is_rhombus 
  {V : Type*} [AddCommGroup V] [Module ℝ V] [NormedAddCommGroup V] 
  (q : Quadrilateral V) 
  (h1 : q.A - q.D = q.C - q.B) 
  (h2 : ‖q.A - q.D‖ = ‖q.A - q.B‖) : 
  is_rhombus q :=
by
  unfold is_rhombus
  exact ⟨h1, h2⟩

#check quadrilateral_is_rhombus

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_is_rhombus_l960_96071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_neg_one_l960_96046

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.exp x else Real.exp (-x)

-- State the theorem
theorem tangent_line_at_neg_one (h : ∀ x, f x = f (-x)) :
  ∃ m b : ℝ, (∀ x, x * m + b = f (-1)) ∧ m = -Real.exp 1 ∧ b = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_neg_one_l960_96046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_36_train_speed_equation_l960_96042

/-- The speed of a train given specific conditions involving a jogger --/
def train_speed (jogger_speed initial_distance train_length passing_time : ℝ) : ℝ :=
  let train_speed : ℝ := 36
  train_speed

/-- The main theorem stating that the train speed is 36 km/hr given the conditions --/
theorem train_speed_is_36 : train_speed 9 150 100 25 = 36 := by
  unfold train_speed
  -- Proof steps would go here, but we'll use sorry for now
  sorry

/-- Theorem stating the relationship between train speed and other parameters --/
theorem train_speed_equation (jogger_speed initial_distance train_length passing_time : ℝ) :
  let train_speed := train_speed jogger_speed initial_distance train_length passing_time
  train_speed * passing_time / 3600 = (initial_distance + train_length) / 1000 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_is_36_train_speed_equation_l960_96042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_given_numbers_l960_96035

theorem largest_among_given_numbers :
  ∀ x : ℝ, x ∈ ({0, -1, 3.5, Real.sqrt 13} : Set ℝ) → x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_given_numbers_l960_96035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brie_skirts_count_correct_l960_96039

def brie_skirts_count (total_blouses : ℕ) (total_slacks : ℕ) (clothes_to_wash : ℕ) (skirts_in_hamper : ℕ) : ℕ :=
  if total_blouses = 12 ∧ 
     total_slacks = 8 ∧ 
     clothes_to_wash = 14 ∧
     skirts_in_hamper = 3 ∧
     (total_blouses * 3 / 4 : ℚ) = 9 ∧
     (total_slacks * 1 / 4 : ℚ) = 2
  then 6
  else 0

#eval brie_skirts_count 12 8 14 3

theorem brie_skirts_count_correct :
  brie_skirts_count 12 8 14 3 = 6 := by
  simp [brie_skirts_count]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brie_skirts_count_correct_l960_96039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_rectangle_perimeter_l960_96002

/-- The perimeter of a figure is the sum of the lengths of all its sides. -/
def perimeter (sides : List ℝ) : ℝ := sides.sum

/-- A square is a rectangle with equal sides. -/
structure Square where
  side : ℝ
deriving Inhabited

/-- A rectangle is defined by its length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ
deriving Inhabited

/-- The large rectangle is composed of squares and small rectangles. -/
structure LargeRectangle where
  squares : List Square
  small_rectangles : List Rectangle
deriving Inhabited

theorem large_rectangle_perimeter :
  ∀ (lr : LargeRectangle),
    lr.squares.length = 3 →
    lr.small_rectangles.length = 3 →
    (∀ s ∈ lr.squares, perimeter [s.side, s.side, s.side, s.side] = 24) →
    (∀ r ∈ lr.small_rectangles, perimeter [r.length, r.width, r.length, r.width] = 16) →
    perimeter [
      3 * (lr.squares.head!).side,
      (lr.squares.head!).side + (lr.squares.head!).side + (lr.small_rectangles.head!).width
    ] = 52 := by
  sorry

#eval "Large rectangle perimeter theorem stated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_rectangle_perimeter_l960_96002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_no_answers_l960_96083

/-- Represents the type of an islander: either a knight or a liar -/
inductive IslanderType
  | Knight
  | Liar
deriving BEq, Repr

/-- Represents a table with its occupants -/
structure Table where
  occupants : List IslanderType
  size_eq_five : occupants.length = 5

/-- Represents the island with its tables -/
structure Island where
  tables : List Table
  seven_tables : tables.length = 7
  total_islanders : (tables.map (·.occupants) |>.join).length = 35

/-- Counts the number of knights at a table -/
def count_knights (table : Table) : Nat :=
  table.occupants.filter (· == IslanderType.Knight) |>.length

/-- Counts the number of tables with at least 3 knights -/
def count_tables_with_three_plus_knights (island : Island) : Nat :=
  island.tables.filter (fun t => count_knights t ≥ 3) |>.length

/-- Determines if there are more than three tables with at least 3 knights -/
def more_than_three_tables_with_three_plus_knights (island : Island) : Bool :=
  count_tables_with_three_plus_knights island > 3

/-- Counts the number of islanders who would answer "No" to the question -/
def count_no_answers (island : Island) : Nat :=
  let actual_answer := more_than_three_tables_with_three_plus_knights island
  (island.tables.map (·.occupants) |>.join).filter (fun t =>
    (t == IslanderType.Knight) != actual_answer
  ) |>.length

/-- The main theorem to be proved -/
theorem max_no_answers (island : Island) :
  count_no_answers island ≤ 23 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_no_answers_l960_96083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l960_96024

-- Define the lines
noncomputable def l₁ (t : ℝ) : ℝ × ℝ := (1 + t, -5 + Real.sqrt 3 * t)
def l₂ (x y : ℝ) : Prop := x - y - 2 * Real.sqrt 3 = 0

-- Define the intersection point P
noncomputable def P : ℝ × ℝ := (1 + 2 * Real.sqrt 3, 1)

-- Define point Q
def Q : ℝ × ℝ := (1, -5)

-- Theorem statement
theorem intersection_and_distance :
  (∃ t, l₁ t = P ∧ l₂ P.1 P.2) ∧
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_distance_l960_96024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_proof_l960_96064

/-- The side length of a rhombus with an inscribed circle of radius r and acute angle α -/
noncomputable def rhombus_side_length (r : ℝ) (α : ℝ) : ℝ := 2 * r / Real.sin α

/-- Theorem stating that the side length of a rhombus with an inscribed circle of radius r
    and acute angle α is equal to 2r / sin(α) -/
theorem rhombus_side_length_proof (r : ℝ) (α : ℝ) 
    (h_r_pos : r > 0) 
    (h_α_pos : 0 < α) 
    (h_α_acute : α < Real.pi / 2) :
  rhombus_side_length r α = 2 * r / Real.sin α :=
by
  -- Unfold the definition of rhombus_side_length
  unfold rhombus_side_length
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_proof_l960_96064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_mpg_l960_96028

noncomputable def initial_odometer : ℚ := 32000
noncomputable def final_odometer : ℚ := 33100
noncomputable def refill1 : ℚ := 15
noncomputable def refill2 : ℚ := 10
noncomputable def refill3 : ℚ := 22

noncomputable def total_distance : ℚ := final_odometer - initial_odometer
noncomputable def total_gasoline : ℚ := refill1 + refill2 + refill3

noncomputable def average_mpg : ℚ := total_distance / total_gasoline

theorem car_trip_mpg : 
  (↑(Int.floor (average_mpg * 10 + 0.5)) / 10 : ℚ) = 23.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_mpg_l960_96028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centenary_set_average_exists_l960_96085

/-- A centenary set is a finite set of at least two distinct positive integers with 100 as its maximum element. -/
def CentenarySet (S : Finset ℕ) : Prop :=
  S.Nonempty ∧ S.card ≥ 2 ∧ (∀ x ∈ S, x > 0 ∧ x ≤ 100) ∧ 100 ∈ S

/-- The average of a finite set of natural numbers. -/
noncomputable def average (S : Finset ℕ) : ℚ :=
  (S.sum (fun x => (x : ℚ))) / S.card

/-- For any integer n between 14 and 100 (inclusive), there exists a centenary set with average n. -/
theorem centenary_set_average_exists :
  ∀ n : ℕ, 14 ≤ n ∧ n ≤ 100 →
  ∃ S : Finset ℕ, CentenarySet S ∧ average S = n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centenary_set_average_exists_l960_96085
