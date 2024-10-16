import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_factors_imply_absolute_value_l1872_187280

theorem polynomial_factors_imply_absolute_value (h k : ℚ) : 
  (∃ (a b : ℚ), (3 * X^4 - 2*h*X^2 - 5*X + k) = (X - 1) * (X + 3) * (a*X^2 + b*X + 1)) →
  |3*h + 2*k| = 117.75 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factors_imply_absolute_value_l1872_187280


namespace NUMINAMATH_CALUDE_least_rectangle_area_for_two_squares_l1872_187287

theorem least_rectangle_area_for_two_squares :
  ∃ (A : ℝ), A = Real.sqrt 2 ∧
  (∀ (a b : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ a^2 + b^2 = 1 →
    ∃ (w h : ℝ), w ≥ 0 ∧ h ≥ 0 ∧ w * h = A ∧ a ≤ w ∧ b ≤ h) ∧
  (∀ (A' : ℝ), A' < A →
    ∃ (a b : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ a^2 + b^2 = 1 ∧
      ∀ (w h : ℝ), w ≥ 0 ∧ h ≥ 0 ∧ w * h = A' → (a > w ∨ b > h)) :=
by sorry

end NUMINAMATH_CALUDE_least_rectangle_area_for_two_squares_l1872_187287


namespace NUMINAMATH_CALUDE_trigonometric_propositions_l1872_187289

theorem trigonometric_propositions :
  (∃ α : ℝ, Real.sin α + Real.cos α = Real.sqrt 2) ∧
  (∀ x : ℝ, Real.sin (3 * Real.pi / 2 + x) = Real.sin (3 * Real.pi / 2 + (-x))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_propositions_l1872_187289


namespace NUMINAMATH_CALUDE_angle_calculation_l1872_187272

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℤ)
  (minutes : ℤ)

/-- Multiplication of an angle by an integer -/
def Angle.mul (a : Angle) (n : ℤ) : Angle :=
  ⟨a.degrees * n, a.minutes * n⟩

/-- Addition of two angles -/
def Angle.add (a b : Angle) : Angle :=
  ⟨a.degrees + b.degrees, a.minutes + b.minutes⟩

/-- Subtraction of two angles -/
def Angle.sub (a b : Angle) : Angle :=
  ⟨a.degrees - b.degrees, a.minutes - b.minutes⟩

/-- Normalize an angle by converting excess minutes to degrees -/
def Angle.normalize (a : Angle) : Angle :=
  let extraDegrees := a.minutes / 60
  let normalizedMinutes := a.minutes % 60
  ⟨a.degrees + extraDegrees, normalizedMinutes⟩

theorem angle_calculation :
  (Angle.normalize ((Angle.mul ⟨24, 31⟩ 4).sub ⟨62, 10⟩)) = ⟨35, 54⟩ := by
  sorry

end NUMINAMATH_CALUDE_angle_calculation_l1872_187272


namespace NUMINAMATH_CALUDE_certain_value_is_one_l1872_187201

theorem certain_value_is_one (w x : ℝ) (h1 : 13 = 13 * w / x) (h2 : w^2 = 1) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_is_one_l1872_187201


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1872_187259

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 10 ∧ b = 24 ∧ c^2 = a^2 + b^2 → c = 26 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1872_187259


namespace NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l1872_187254

/-- Given the conditions of the police force duty, prove the percentage of female officers on duty -/
theorem percentage_female_officers_on_duty
  (total_on_duty : ℕ)
  (total_female_officers : ℕ)
  (half_on_duty_female : total_on_duty / 2 = total_on_duty - total_on_duty / 2)
  (h_total_on_duty : total_on_duty = 180)
  (h_total_female : total_female_officers = 500) :
  (((total_on_duty / 2 : ℚ) / total_female_officers) * 100 : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_percentage_female_officers_on_duty_l1872_187254


namespace NUMINAMATH_CALUDE_decimal_representation_5_11_l1872_187227

/-- The decimal representation of 5/11 has a repeating sequence of length 2 -/
def repeating_length : ℕ := 2

/-- The 150th decimal place in the representation of 5/11 -/
def decimal_place : ℕ := 150

/-- The result we want to prove -/
def result : ℕ := 5

theorem decimal_representation_5_11 :
  (decimal_place % repeating_length = 0) ∧
  (result = 5) := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_5_11_l1872_187227


namespace NUMINAMATH_CALUDE_fourth_grade_students_l1872_187249

/-- The number of students in fourth grade at the start of the year. -/
def initial_students : ℕ := 33

/-- The number of students who left during the year. -/
def students_left : ℕ := 18

/-- The number of new students who came during the year. -/
def new_students : ℕ := 14

/-- The number of students at the end of the year. -/
def final_students : ℕ := 29

theorem fourth_grade_students : 
  initial_students - students_left + new_students = final_students := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l1872_187249


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1872_187229

/-- An arithmetic sequence with 20 terms -/
structure ArithmeticSequence :=
  (a : ℚ)  -- First term
  (d : ℚ)  -- Common difference

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a + (n - 1) * seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) : 
  sum_n seq 3 = 15 ∧ 
  sum_n seq 3 - 3 * seq.a - 51 * seq.d = 12 → 
  sum_n seq 20 = 90 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1872_187229


namespace NUMINAMATH_CALUDE_rabbit_carrot_problem_l1872_187299

theorem rabbit_carrot_problem :
  ∀ (rabbit_holes hamster_holes : ℕ),
    rabbit_holes = hamster_holes - 3 →
    4 * rabbit_holes = 5 * hamster_holes →
    4 * rabbit_holes = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrot_problem_l1872_187299


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l1872_187200

/-- Represents a triangle with side lengths a, b, and c --/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Generates the next triangle in the sequence based on the incircle tangency points --/
def nextTriangle (t : Triangle) : Triangle := sorry

/-- Checks if a triangle is valid (satisfies the triangle inequality) --/
def isValidTriangle (t : Triangle) : Bool := sorry

/-- The sequence of triangles starting from T₁ --/
def triangleSequence : List Triangle := sorry

/-- The last valid triangle in the sequence --/
def lastValidTriangle : Triangle := sorry

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℚ := sorry

theorem last_triangle_perimeter :
  let t₁ : Triangle := { a := 2011, b := 2012, c := 2013 }
  perimeter (lastValidTriangle) = 1509 / 128 := by sorry

end NUMINAMATH_CALUDE_last_triangle_perimeter_l1872_187200


namespace NUMINAMATH_CALUDE_triangle_incircle_path_length_l1872_187277

theorem triangle_incircle_path_length (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sides : a = 6 ∧ b = 8 ∧ c = 10) : 
  let s := (a + b + c) / 2
  let r := (s * (s - a) * (s - b) * (s - c)).sqrt / s
  (a + b + c) - 2 * r = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_incircle_path_length_l1872_187277


namespace NUMINAMATH_CALUDE_pencil_count_l1872_187248

/-- The total number of colored pencils Cheryl, Cyrus, and Madeline have -/
def total_pencils (cheryl : ℕ) (cyrus : ℕ) (madeline : ℕ) : ℕ :=
  cheryl + cyrus + madeline

/-- Theorem stating the total number of colored pencils given the conditions -/
theorem pencil_count :
  ∀ (cheryl cyrus madeline : ℕ),
    cheryl = 3 * cyrus →
    madeline = 63 →
    cheryl = 2 * madeline →
    total_pencils cheryl cyrus madeline = 231 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l1872_187248


namespace NUMINAMATH_CALUDE_matching_segment_exists_l1872_187224

/-- A 20-digit binary number -/
def BinaryNumber := Fin 20 → Bool

/-- A is a 20-digit binary number with 10 zeros and 10 ones -/
def is_valid_A (A : BinaryNumber) : Prop :=
  (Finset.filter (λ i => A i = false) Finset.univ).card = 10 ∧
  (Finset.filter (λ i => A i = true) Finset.univ).card = 10

/-- B is any 20-digit binary number -/
def B : BinaryNumber := sorry

/-- C is a 40-digit binary number formed by concatenating B with itself -/
def C : Fin 40 → Bool :=
  λ i => B (Fin.val i % 20)

/-- Count matching bits between two binary numbers -/
def count_matches (X Y : BinaryNumber) : Nat :=
  (Finset.filter (λ i => X i = Y i) Finset.univ).card

/-- Theorem: There exists a 20-bit segment of C with at least 10 matching bits with A -/
theorem matching_segment_exists (A : BinaryNumber) (h : is_valid_A A) :
  ∃ k : Fin 21, count_matches A (λ i => C (i + k)) ≥ 10 := by sorry

end NUMINAMATH_CALUDE_matching_segment_exists_l1872_187224


namespace NUMINAMATH_CALUDE_hyperbola_through_point_l1872_187235

/-- A hyperbola with its axes of symmetry along the coordinate axes -/
structure CoordinateAxisHyperbola where
  a : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / a^2 - y^2 / a^2 = 1

/-- The hyperbola passes through the point (3, -1) -/
def passes_through (h : CoordinateAxisHyperbola) : Prop :=
  h.equation (3, -1)

theorem hyperbola_through_point :
  ∃ (h : CoordinateAxisHyperbola), passes_through h ∧ h.a^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_through_point_l1872_187235


namespace NUMINAMATH_CALUDE_min_t_value_l1872_187234

theorem min_t_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) :
  (∀ a b, a > 0 → b > 0 → 2 * a + b = 1 → 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) →
  t ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l1872_187234


namespace NUMINAMATH_CALUDE_train_length_l1872_187230

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 * 1000 / 3600 →
  crossing_time = 16.7986561075114 →
  bridge_length = 170 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.0000000001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1872_187230


namespace NUMINAMATH_CALUDE_circles_intersect_iff_distance_between_radii_sum_and_diff_l1872_187275

/-- Two circles intersect if and only if the distance between their centers
    is greater than the absolute difference of their radii and less than
    the sum of their radii. -/
theorem circles_intersect_iff_distance_between_radii_sum_and_diff
  (R r d : ℝ) (h : R ≥ r) :
  (∃ (p : ℝ × ℝ), (p.1 - 0)^2 + (p.2 - 0)^2 = R^2 ∧ 
                  (p.1 - d)^2 + p.2^2 = r^2) ↔
  (R - r < d ∧ d < R + r) :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_iff_distance_between_radii_sum_and_diff_l1872_187275


namespace NUMINAMATH_CALUDE_terrys_breakfast_spending_l1872_187205

theorem terrys_breakfast_spending (x : ℝ) : 
  x > 0 ∧ x + 2*x + 6*x = 54 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_terrys_breakfast_spending_l1872_187205


namespace NUMINAMATH_CALUDE_pairwise_products_equal_differences_impossible_l1872_187251

theorem pairwise_products_equal_differences_impossible
  (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_order : a < b ∧ b < c ∧ c < d)
  (h_product_order : a * b < a * c ∧ a * c < a * d ∧ a * d < b * c ∧ b * c < b * d ∧ b * d < c * d) :
  ¬∃ k : ℝ, k > 0 ∧
    a * c - a * b = k ∧
    a * d - a * c = k ∧
    b * c - a * d = k ∧
    b * d - b * c = k ∧
    c * d - b * d = k :=
by sorry

end NUMINAMATH_CALUDE_pairwise_products_equal_differences_impossible_l1872_187251


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l1872_187286

open Real

-- Define the equation
def equation (x : ℝ) : Prop :=
  tan (5 * x * π / 180) = (1 - sin (x * π / 180)) / (1 + sin (x * π / 180))

-- State the theorem
theorem smallest_positive_angle :
  ∃ (x : ℝ), x > 0 ∧ x < 10 ∧ equation x ∧ ∀ (y : ℝ), 0 < y ∧ y < x → ¬(equation y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l1872_187286


namespace NUMINAMATH_CALUDE_student_count_proof_l1872_187262

theorem student_count_proof (initial_avg : ℝ) (new_student_weight : ℝ) (final_avg : ℝ) :
  initial_avg = 28 →
  new_student_weight = 7 →
  final_avg = 27.3 →
  (∃ n : ℕ, (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * final_avg ∧ n = 29) :=
by sorry

end NUMINAMATH_CALUDE_student_count_proof_l1872_187262


namespace NUMINAMATH_CALUDE_five_integers_problem_l1872_187223

theorem five_integers_problem : 
  ∃ (a b c d e : ℤ), 
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
      {3, 8, 9, 16, 17, 17, 18, 22, 23, 31} ∧
    a * b * c * d * e = 3360 := by
  sorry

end NUMINAMATH_CALUDE_five_integers_problem_l1872_187223


namespace NUMINAMATH_CALUDE_set_equals_interval_l1872_187267

-- Define the set S as {x | -1 < x ≤ 3}
def S : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- Define the interval (-1,3]
def I : Set ℝ := Set.Ioc (-1) 3

-- Theorem statement
theorem set_equals_interval : S = I := by sorry

end NUMINAMATH_CALUDE_set_equals_interval_l1872_187267


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l1872_187222

/-- The number of lampposts along the alley -/
def total_lampposts : ℕ := 400

/-- The lamppost number where Alla is observed -/
def alla_observed : ℕ := 55

/-- The lamppost number where Boris is observed -/
def boris_observed : ℕ := 321

/-- The function to calculate the meeting point of Alla and Boris -/
def meeting_point : ℕ :=
  let intervals_covered := (alla_observed - 1) + (total_lampposts - boris_observed)
  let total_intervals := total_lampposts - 1
  (intervals_covered * 3) + 1

/-- Theorem stating that Alla and Boris meet at lamppost 163 -/
theorem alla_boris_meeting :
  meeting_point = 163 := by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l1872_187222


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_is_glb_l1872_187204

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := by sorry

theorem min_value_is_glb : ∃ (seq : ℕ → ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  ((seq n)^2 + 9) / Real.sqrt ((seq n)^2 + 5) < 4 + ε := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_is_glb_l1872_187204


namespace NUMINAMATH_CALUDE_line_plane_zero_angle_l1872_187214

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- The angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry

/-- A line is parallel to a plane -/
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line lies within a plane -/
def lies_within (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- If the angle between a line and a plane is 0°, then the line is either parallel to the plane or lies within it -/
theorem line_plane_zero_angle (l : Line3D) (p : Plane3D) :
  angle_line_plane l p = 0 → is_parallel l p ∨ lies_within l p :=
sorry

end NUMINAMATH_CALUDE_line_plane_zero_angle_l1872_187214


namespace NUMINAMATH_CALUDE_chocolates_remaining_on_day_five_l1872_187211

/-- Chocolates eaten per day --/
def chocolates_eaten (day : Nat) : Nat :=
  match day with
  | 1 => 4
  | 2 => 2 * 4 - 3
  | 3 => 4 - 2
  | 4 => (4 - 2) - 1
  | _ => 0

/-- Total chocolates eaten up to a given day --/
def total_eaten (day : Nat) : Nat :=
  match day with
  | 0 => 0
  | n + 1 => total_eaten n + chocolates_eaten (n + 1)

theorem chocolates_remaining_on_day_five : 
  24 - total_eaten 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_remaining_on_day_five_l1872_187211


namespace NUMINAMATH_CALUDE_probability_of_selection_X_l1872_187258

theorem probability_of_selection_X 
  (prob_Y : ℝ) 
  (prob_X_and_Y : ℝ) 
  (h1 : prob_Y = 2/5) 
  (h2 : prob_X_and_Y = 0.05714285714285714) : 
  ∃ (prob_X : ℝ), prob_X = 0.14285714285714285 ∧ prob_X_and_Y = prob_X * prob_Y :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_X_l1872_187258


namespace NUMINAMATH_CALUDE_number_problem_l1872_187213

theorem number_problem (x : ℝ) : (0.6 * (3/5) * x = 36) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1872_187213


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1872_187207

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 4 = 4) →
  (a 3 + a 5 = 10) →
  a 5 + a 7 = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1872_187207


namespace NUMINAMATH_CALUDE_mango_apple_not_orange_count_l1872_187285

/-- Given information about fruit preferences --/
structure FruitPreferences where
  apple : Nat
  orange_mango_not_apple : Nat
  all_fruits : Nat
  total_apple : Nat

/-- Calculate the number of people who like mango and apple and dislike orange --/
def mango_apple_not_orange (prefs : FruitPreferences) : Nat :=
  prefs.total_apple - prefs.all_fruits - prefs.orange_mango_not_apple

/-- Theorem stating the result of the calculation --/
theorem mango_apple_not_orange_count 
  (prefs : FruitPreferences) 
  (h1 : prefs.apple = 40)
  (h2 : prefs.orange_mango_not_apple = 7)
  (h3 : prefs.all_fruits = 4)
  (h4 : prefs.total_apple = 47) :
  mango_apple_not_orange prefs = 36 := by
  sorry

#eval mango_apple_not_orange ⟨40, 7, 4, 47⟩

end NUMINAMATH_CALUDE_mango_apple_not_orange_count_l1872_187285


namespace NUMINAMATH_CALUDE_bobs_number_l1872_187203

theorem bobs_number (alice bob : ℂ) : 
  alice * bob = 48 - 16 * I ∧ alice = 7 + 4 * I → 
  bob = 272/65 - 304/65 * I := by
sorry

end NUMINAMATH_CALUDE_bobs_number_l1872_187203


namespace NUMINAMATH_CALUDE_division_remainder_proof_l1872_187270

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ)
  (h1 : dividend = 176)
  (h2 : divisor = 14)
  (h3 : quotient = 12)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l1872_187270


namespace NUMINAMATH_CALUDE_min_irrational_root_distance_l1872_187212

theorem min_irrational_root_distance (a b c : ℕ+) (h_a : a ≤ 10) :
  let f := fun x : ℝ => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)
  let roots := {x : ℝ | f x = 0}
  let distance := fun (x y : ℝ) => |x - y|
  (∃ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∃ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ Irrational (distance x y)) →
  (∀ (x y : ℝ), x ∈ roots ∧ y ∈ roots ∧ x ≠ y → distance x y ≥ Real.sqrt 13 / 9) :=
by sorry

end NUMINAMATH_CALUDE_min_irrational_root_distance_l1872_187212


namespace NUMINAMATH_CALUDE_painted_cells_count_l1872_187295

-- Define the structure of the grid
structure Grid :=
  (k : ℕ)
  (l : ℕ)

-- Define the properties of the grid
def valid_grid (g : Grid) : Prop :=
  g.k = 2 ∧ g.l = 37

-- Define the number of white cells
def white_cells (g : Grid) : ℕ :=
  g.k * g.l

-- Define the total number of cells
def total_cells (g : Grid) : ℕ :=
  (2 * g.k + 1) * (2 * g.l + 1)

-- Define the number of painted cells
def painted_cells (g : Grid) : ℕ :=
  total_cells g - white_cells g

-- The main theorem
theorem painted_cells_count (g : Grid) :
  valid_grid g → white_cells g = 74 → painted_cells g = 301 :=
by
  sorry


end NUMINAMATH_CALUDE_painted_cells_count_l1872_187295


namespace NUMINAMATH_CALUDE_tan_sum_equals_double_tan_l1872_187255

theorem tan_sum_equals_double_tan (α β : Real) 
  (h : 3 * Real.sin β = Real.sin (2 * α + β)) : 
  Real.tan (α + β) = 2 * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equals_double_tan_l1872_187255


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l1872_187268

-- Problem 1
def solution_set (x : ℝ) : Prop := x < -7 ∨ x > 5/3

theorem inequality_solution : 
  ∀ x : ℝ, |2*x + 1| - |x - 4| > 2 ↔ solution_set x := by sorry

-- Problem 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l1872_187268


namespace NUMINAMATH_CALUDE_max_lateral_surface_area_rectangular_prism_l1872_187233

theorem max_lateral_surface_area_rectangular_prism :
  ∀ l w h : ℕ,
  l + w + h = 88 →
  2 * (l * w + l * h + w * h) ≤ 224 :=
by sorry

end NUMINAMATH_CALUDE_max_lateral_surface_area_rectangular_prism_l1872_187233


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1872_187218

theorem solution_set_quadratic_inequality :
  {x : ℝ | x * (x - 1) > 0} = Set.Iio 0 ∪ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1872_187218


namespace NUMINAMATH_CALUDE_first_stop_students_correct_l1872_187245

/-- The number of students who got on the bus at the first stop -/
def first_stop_students : ℕ := 39

/-- The number of students who got on the bus at the second stop -/
def second_stop_students : ℕ := 29

/-- The total number of students on the bus after the second stop -/
def total_students : ℕ := 68

/-- Theorem stating that the number of students who got on at the first stop is correct -/
theorem first_stop_students_correct :
  first_stop_students + second_stop_students = total_students := by
  sorry

end NUMINAMATH_CALUDE_first_stop_students_correct_l1872_187245


namespace NUMINAMATH_CALUDE_function_property_implies_k_values_l1872_187291

-- Define the function type
def FunctionType := ℕ → ℤ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionType) (k : ℤ) : Prop :=
  f 1995 = 1996 ∧
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

-- Theorem statement
theorem function_property_implies_k_values :
  ∀ f : FunctionType, ∀ k : ℤ,
    SatisfiesProperty f k → (k = -1 ∨ k = 0) :=
sorry

end NUMINAMATH_CALUDE_function_property_implies_k_values_l1872_187291


namespace NUMINAMATH_CALUDE_march_book_sales_l1872_187281

theorem march_book_sales (january_sales february_sales : ℕ) 
  (h1 : january_sales = 15)
  (h2 : february_sales = 16)
  (h3 : (january_sales + february_sales + march_sales) / 3 = 16) :
  march_sales = 17 := by
  sorry

end NUMINAMATH_CALUDE_march_book_sales_l1872_187281


namespace NUMINAMATH_CALUDE_heart_then_king_probability_l1872_187215

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of hearts in a standard deck -/
def num_hearts : ℕ := 13

/-- The number of kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing a heart first and a king second from a standard deck -/
theorem heart_then_king_probability :
  (num_hearts / deck_size) * ((num_kings - 1) / (deck_size - 1)) +
  ((num_hearts - 1) / deck_size) * (num_kings / (deck_size - 1)) =
  1 / deck_size :=
sorry

end NUMINAMATH_CALUDE_heart_then_king_probability_l1872_187215


namespace NUMINAMATH_CALUDE_odot_calculation_l1872_187263

-- Define the ⊙ operation
def odot (a b : ℤ) : ℤ := a * b - (a + b)

-- State the theorem
theorem odot_calculation : odot 6 (odot 5 4) = 49 := by
  sorry

end NUMINAMATH_CALUDE_odot_calculation_l1872_187263


namespace NUMINAMATH_CALUDE_infinitely_many_palindromes_l1872_187225

/-- Arithmetic progression term -/
def a (n : ℕ+) : ℕ := 18 + 19 * (n - 1)

/-- Repunit -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- k values -/
def k (t : ℕ) : ℕ := 18 * t + 6

theorem infinitely_many_palindromes :
  ∀ m : ℕ, ∃ t : ℕ, t > m ∧ ∃ n : ℕ+, R (k t) = a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_palindromes_l1872_187225


namespace NUMINAMATH_CALUDE_angle_sum_proof_l1872_187209

theorem angle_sum_proof (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin α = Real.sqrt 5 / 5) (h4 : Real.sin β = Real.sqrt 10 / 10) :
  α + β = π/4 := by sorry

end NUMINAMATH_CALUDE_angle_sum_proof_l1872_187209


namespace NUMINAMATH_CALUDE_range_of_a_when_complement_subset_l1872_187206

-- Define the sets A, B, and C
def A : Set ℝ := {x | 0 < 2*x + 4 ∧ 2*x + 4 < 10}
def B : Set ℝ := {x | x < -4 ∨ x > 2}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0}

-- State the theorem
theorem range_of_a_when_complement_subset (a : ℝ) :
  (Set.univ \ (A ∪ B) : Set ℝ) ⊆ C a → -2 < a ∧ a < -4/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_when_complement_subset_l1872_187206


namespace NUMINAMATH_CALUDE_g_minus_one_eq_zero_iff_s_eq_neg_six_l1872_187278

/-- The function g(x) as defined in the problem -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 2 * x^2 + x + s

/-- Theorem stating that g(-1) = 0 if and only if s = -6 -/
theorem g_minus_one_eq_zero_iff_s_eq_neg_six :
  ∀ s : ℝ, g s (-1) = 0 ↔ s = -6 := by sorry

end NUMINAMATH_CALUDE_g_minus_one_eq_zero_iff_s_eq_neg_six_l1872_187278


namespace NUMINAMATH_CALUDE_max_x_minus_y_l1872_187256

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l1872_187256


namespace NUMINAMATH_CALUDE_toucan_count_l1872_187298

theorem toucan_count (initial_toucans : ℕ) (joining_toucans : ℕ) : 
  initial_toucans = 2 → joining_toucans = 1 → initial_toucans + joining_toucans = 3 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l1872_187298


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_10000_l1872_187266

-- Define the modulus
def m : ℕ := 10000

-- Define the number we're finding the inverse for
def a : ℕ := 7

-- Define the claimed inverse
def claimed_inverse : ℕ := 8571

-- Theorem statement
theorem modular_inverse_of_7_mod_10000 :
  (a * claimed_inverse) % m = 1 ∧ 0 ≤ claimed_inverse ∧ claimed_inverse < m :=
by sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_10000_l1872_187266


namespace NUMINAMATH_CALUDE_max_M_value_l1872_187279

def J (k : ℕ) : ℕ := 10^(k+2) + 128

def M (k : ℕ) : ℕ := (J k).factors.count 2

theorem max_M_value : ∃ k : ℕ, k > 0 ∧ M k = 8 ∧ ∀ j : ℕ, j > 0 → M j ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_M_value_l1872_187279


namespace NUMINAMATH_CALUDE_square_difference_equality_l1872_187288

theorem square_difference_equality : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1872_187288


namespace NUMINAMATH_CALUDE_people_in_room_l1872_187210

theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs ∧ 
  chairs - (4 : ℚ) / 5 * chairs = 10 →
  people = 67 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l1872_187210


namespace NUMINAMATH_CALUDE_certain_number_equation_l1872_187264

theorem certain_number_equation : ∃ x : ℝ, 
  (3889 + x - 47.95000000000027 = 3854.002) ∧ 
  (x = 12.95200000000054) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1872_187264


namespace NUMINAMATH_CALUDE_box_volume_increase_l1872_187269

/-- 
A rectangular box with length l, width w, and height h.
Given the conditions:
1. Volume is 5400 cubic inches
2. Surface area is 2352 square inches
3. Sum of the lengths of its 12 edges is 240 inches
Prove that increasing each dimension by 1 inch results in a volume of 6637 cubic inches
-/
theorem box_volume_increase (l w h : ℝ) 
  (volume : l * w * h = 5400)
  (surface_area : 2 * l * w + 2 * w * h + 2 * h * l = 2352)
  (edge_sum : 4 * l + 4 * w + 4 * h = 240) :
  (l + 1) * (w + 1) * (h + 1) = 6637 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1872_187269


namespace NUMINAMATH_CALUDE_andy_coat_production_l1872_187273

/-- Given the conditions about minks and coat production, prove that Andy can make 7 coats. -/
theorem andy_coat_production (
  minks_per_coat : ℕ := 15
  ) (
  initial_minks : ℕ := 30
  ) (
  babies_per_mink : ℕ := 6
  ) (
  freed_fraction : ℚ := 1/2
  ) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_andy_coat_production_l1872_187273


namespace NUMINAMATH_CALUDE_unique_m_value_l1872_187242

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l1872_187242


namespace NUMINAMATH_CALUDE_cats_adopted_l1872_187297

/-- Proves the number of cats adopted given the shelter's cat population changes -/
theorem cats_adopted (initial_cats : ℕ) (new_cats : ℕ) (kittens_born : ℕ) (cat_picked_up : ℕ) (final_cats : ℕ) :
  initial_cats = 6 →
  new_cats = 12 →
  kittens_born = 5 →
  cat_picked_up = 1 →
  final_cats = 19 →
  initial_cats + new_cats - (initial_cats + new_cats + kittens_born - cat_picked_up - final_cats) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cats_adopted_l1872_187297


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l1872_187228

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Given conditions for the problem -/
structure ProblemConditions (C : Ellipse) where
  eccentricity : ℝ
  focusDistance : ℝ
  h1 : eccentricity = 1/2
  h2 : focusDistance = 2 * Real.sqrt 2

/-- The equation of line l -/
def line_equation (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 2 * Real.sqrt 3 = 0

/-- Main theorem statement -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (cond : ProblemConditions C)
  (T_y_coord : ℝ)
  (h_T_y : T_y_coord = 6 * Real.sqrt 3) :
  (∀ x y, x^2 / 16 + y^2 / 12 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (∀ x y, line_equation x y) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l1872_187228


namespace NUMINAMATH_CALUDE_train_length_l1872_187292

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 180 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1872_187292


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1872_187282

theorem pure_imaginary_complex_number (x : ℝ) :
  (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I = (0 : ℂ) + (y : ℂ) * Complex.I →
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1872_187282


namespace NUMINAMATH_CALUDE_f_composition_negative_one_l1872_187257

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_one_l1872_187257


namespace NUMINAMATH_CALUDE_club_officer_selection_l1872_187271

/-- Represents the number of ways to choose officers in a club -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  let president_vp_combinations := boys * girls * 2
  let secretary_choices := girls
  president_vp_combinations * secretary_choices

/-- Theorem stating the number of ways to choose officers under given conditions -/
theorem club_officer_selection :
  choose_officers 15 9 6 = 648 :=
by
  sorry


end NUMINAMATH_CALUDE_club_officer_selection_l1872_187271


namespace NUMINAMATH_CALUDE_unique_magnitude_quadratic_l1872_187260

/-- For the quadratic equation z^2 - 10z + 50 = 0, there is only one possible value for |z| -/
theorem unique_magnitude_quadratic : 
  ∃! m : ℝ, ∀ z : ℂ, z^2 - 10*z + 50 = 0 → Complex.abs z = m :=
by sorry

end NUMINAMATH_CALUDE_unique_magnitude_quadratic_l1872_187260


namespace NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_one_l1872_187216

theorem inequality_solution_implies_m_less_than_one :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_m_less_than_one_l1872_187216


namespace NUMINAMATH_CALUDE_babysitter_earnings_correct_l1872_187208

/-- Calculates the babysitter's earnings for a given number of hours worked -/
def babysitter_earnings (regular_rate : ℕ) (regular_hours : ℕ) (overtime_rate : ℕ) (total_hours : ℕ) : ℕ :=
  let regular_pay := min regular_hours total_hours * regular_rate
  let overtime_pay := max 0 (total_hours - regular_hours) * overtime_rate
  regular_pay + overtime_pay

theorem babysitter_earnings_correct :
  let regular_rate : ℕ := 16
  let regular_hours : ℕ := 30
  let overtime_rate : ℕ := 28  -- 16 + (75% of 16)
  let total_hours : ℕ := 40
  babysitter_earnings regular_rate regular_hours overtime_rate total_hours = 760 :=
by sorry

end NUMINAMATH_CALUDE_babysitter_earnings_correct_l1872_187208


namespace NUMINAMATH_CALUDE_second_smallest_divisible_sum_of_digits_l1872_187226

def isDivisibleByAllLessThan8 (n : ℕ) : Prop :=
  ∀ k : ℕ, 0 < k ∧ k < 8 → n % k = 0

def isSecondSmallestDivisible (n : ℕ) : Prop :=
  isDivisibleByAllLessThan8 n ∧
  ∃ m : ℕ, m < n ∧ isDivisibleByAllLessThan8 m ∧
  ∀ k : ℕ, k < n ∧ isDivisibleByAllLessThan8 k → k ≤ m

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem second_smallest_divisible_sum_of_digits :
  ∃ N : ℕ, isSecondSmallestDivisible N ∧ sumOfDigits N = 12 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_divisible_sum_of_digits_l1872_187226


namespace NUMINAMATH_CALUDE_max_value_fraction_l1872_187238

theorem max_value_fraction (a b : ℝ) (h1 : a * b = 1) (h2 : a > b) (h3 : b ≥ 2/3) :
  (a - b) / (a^2 + b^2) ≤ 30/97 := by
  sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1872_187238


namespace NUMINAMATH_CALUDE_max_distribution_girls_l1872_187284

theorem max_distribution_girls (bags : Nat) (eyeliners : Nat) 
  (h1 : bags = 2923) (h2 : eyeliners = 3239) : 
  Nat.gcd bags eyeliners = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_distribution_girls_l1872_187284


namespace NUMINAMATH_CALUDE_angela_beth_age_ratio_l1872_187241

theorem angela_beth_age_ratio :
  ∀ (angela_age beth_age : ℕ),
    (angela_age - 5 + beth_age - 5 = 45) →  -- Five years ago, sum of ages was 45
    (angela_age + 5 = 44) →                 -- In five years, Angela will be 44
    (angela_age : ℚ) / beth_age = 39 / 16   -- Ratio of current ages is 39:16
    :=
by
  sorry

end NUMINAMATH_CALUDE_angela_beth_age_ratio_l1872_187241


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1872_187293

def A : Set Int := {-2, -1, 0, 1, 2}
def B : Set Int := {x | 2 * x - 1 > 0}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1872_187293


namespace NUMINAMATH_CALUDE_container_volume_l1872_187239

/-- Given a cube with surface area 864 square units placed inside a cuboidal container
    with a 1 unit gap on all sides, the volume of the container is 2744 cubic units. -/
theorem container_volume (cube_surface_area : ℝ) (gap : ℝ) :
  cube_surface_area = 864 →
  gap = 1 →
  (cube_surface_area / 6).sqrt + 2 * gap ^ 3 = 2744 :=
by sorry

end NUMINAMATH_CALUDE_container_volume_l1872_187239


namespace NUMINAMATH_CALUDE_max_distance_C₁_intersections_l1872_187240

noncomputable section

-- Define the curves
def C₁ (t α : ℝ) : ℝ × ℝ := (t * Real.cos α, t * Real.sin α)

def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 3 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

def C₃ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the set of valid parameters
def ValidParams : Set ℝ := {α | 0 ≤ α ∧ α ≤ Real.pi}

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem max_distance_C₁_intersections :
  ∃ (max_dist : ℝ), max_dist = 4 ∧
  ∀ (t₁ t₂ θ₁ θ₂ α : ℝ), α ∈ ValidParams →
    distance (C₁ t₁ α) (C₂ θ₁) = 0 →
    distance (C₁ t₂ α) (C₃ θ₂) = 0 →
    distance (C₁ t₁ α) (C₁ t₂ α) ≤ max_dist :=
sorry

end

end NUMINAMATH_CALUDE_max_distance_C₁_intersections_l1872_187240


namespace NUMINAMATH_CALUDE_f_maps_neg_two_three_to_one_neg_six_l1872_187217

/-- The mapping f that transforms a point (x, y) to (x+y, xy) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 * p.2)

/-- Theorem stating that f maps (-2, 3) to (1, -6) -/
theorem f_maps_neg_two_three_to_one_neg_six :
  f (-2, 3) = (1, -6) := by sorry

end NUMINAMATH_CALUDE_f_maps_neg_two_three_to_one_neg_six_l1872_187217


namespace NUMINAMATH_CALUDE_sum_of_perfect_squares_l1872_187261

theorem sum_of_perfect_squares (x : ℕ) (h : ∃ k : ℕ, x = k ^ 2) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m ^ 2) ∧ x + y = 2 * x + 2 * (x.sqrt) + 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_perfect_squares_l1872_187261


namespace NUMINAMATH_CALUDE_reciprocal_pairs_l1872_187221

def are_reciprocals (a b : ℚ) : Prop := a * b = 1

theorem reciprocal_pairs :
  ¬(are_reciprocals 1 (-1)) ∧
  ¬(are_reciprocals (-1/3) 3) ∧
  are_reciprocals (-5) (-1/5) ∧
  ¬(are_reciprocals (-3) (|(-3)|)) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_pairs_l1872_187221


namespace NUMINAMATH_CALUDE_probability_theorem_l1872_187236

/- Define the number of white and black balls -/
def white_balls : ℕ := 2
def black_balls : ℕ := 3
def total_balls : ℕ := white_balls + black_balls

/- Define the probability of drawing a white ball and a black ball -/
def prob_white : ℚ := white_balls / total_balls
def prob_black : ℚ := black_balls / total_balls

/- Part I: Sampling with replacement -/
def prob_different_colors : ℚ := prob_white * prob_black * 2

/- Part II: Sampling without replacement -/
def prob_zero_white : ℚ := (black_balls / total_balls) * ((black_balls - 1) / (total_balls - 1))
def prob_one_white : ℚ := (black_balls / total_balls) * (white_balls / (total_balls - 1)) + 
                          (white_balls / total_balls) * (black_balls / (total_balls - 1))
def prob_two_white : ℚ := (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))

def expectation : ℚ := 0 * prob_zero_white + 1 * prob_one_white + 2 * prob_two_white
def variance : ℚ := (0 - expectation)^2 * prob_zero_white + 
                    (1 - expectation)^2 * prob_one_white + 
                    (2 - expectation)^2 * prob_two_white

theorem probability_theorem :
  prob_different_colors = 12/25 ∧
  prob_zero_white = 3/10 ∧
  prob_one_white = 3/5 ∧
  prob_two_white = 1/10 ∧
  expectation = 4/5 ∧
  variance = 9/25 := by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1872_187236


namespace NUMINAMATH_CALUDE_village_population_problem_l1872_187243

theorem village_population_problem (final_population : ℕ) 
  (h1 : final_population = 3168) : ∃ initial_population : ℕ,
  (initial_population : ℝ) * 0.9 * 0.8 = final_population ∧ 
  initial_population = 4400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l1872_187243


namespace NUMINAMATH_CALUDE_special_sequence_a10_l1872_187244

/-- A sequence with the property that for any p, q ∈ ℕ*, aₚ₊ₖ = aₚ · aₖ -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p * a q

theorem special_sequence_a10 (a : ℕ → ℕ) (h : SpecialSequence a) (h2 : a 2 = 4) :
  a 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a10_l1872_187244


namespace NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l1872_187246

theorem opposite_reciprocal_expression_value (a b c d m : ℝ) :
  a + b = 0 →
  c * d = 1 →
  |m| = 4 →
  (a + b) / (3 * m) + m^2 - 5 * c * d + 6 * m = 35 ∨
  (a + b) / (3 * m) + m^2 - 5 * c * d + 6 * m = -13 :=
by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_expression_value_l1872_187246


namespace NUMINAMATH_CALUDE_second_negative_integer_l1872_187283

theorem second_negative_integer (n : ℤ) : 
  n < 0 → -11 * n + 5 = 93 → n = -8 :=
by sorry

end NUMINAMATH_CALUDE_second_negative_integer_l1872_187283


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1872_187265

theorem regular_polygon_sides (exterior_angle : ℝ) :
  exterior_angle = 18 →
  (360 / exterior_angle : ℝ) = 20 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1872_187265


namespace NUMINAMATH_CALUDE_vectors_perpendicular_l1872_187296

theorem vectors_perpendicular : ∀ (a b : ℝ × ℝ), 
  a = (2, -3) → b = (3, 2) → a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_perpendicular_l1872_187296


namespace NUMINAMATH_CALUDE_shopping_expense_calculation_l1872_187232

theorem shopping_expense_calculation (T : ℝ) (x : ℝ) 
  (h1 : 0 < T) 
  (h2 : 0.5 * T + 0.2 * T + x * T = T) 
  (h3 : 0.04 * 0.5 * T + 0 * 0.2 * T + 0.08 * x * T = 0.044 * T) : 
  x = 0.3 := by
sorry

end NUMINAMATH_CALUDE_shopping_expense_calculation_l1872_187232


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l1872_187252

theorem pentagon_largest_angle (F G H I J : ℝ) : 
  F = 90 → 
  G = 70 → 
  H = I → 
  J = 2 * H + 20 → 
  F + G + H + I + J = 540 → 
  max F (max G (max H (max I J))) = 200 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l1872_187252


namespace NUMINAMATH_CALUDE_triangle_cosB_value_l1872_187219

theorem triangle_cosB_value (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  A = π / 4 →
  c * Real.sin B = Real.sqrt 3 * b * Real.cos C →
  Real.cos B = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry


end NUMINAMATH_CALUDE_triangle_cosB_value_l1872_187219


namespace NUMINAMATH_CALUDE_distance_equals_speed_times_time_l1872_187294

/-- The distance between Patrick's house and Aaron's house -/
def distance : ℝ := 14

/-- The time Patrick spent jogging -/
def time : ℝ := 2

/-- Patrick's jogging speed -/
def speed : ℝ := 7

/-- Theorem stating that the distance is equal to speed multiplied by time -/
theorem distance_equals_speed_times_time : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_distance_equals_speed_times_time_l1872_187294


namespace NUMINAMATH_CALUDE_limit_at_negative_one_l1872_187274

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem limit_at_negative_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |((f (-1 + Δx) - f (-1)) / Δx) - (-2)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_at_negative_one_l1872_187274


namespace NUMINAMATH_CALUDE_total_savings_l1872_187202

/-- The total savings over two months given the savings in September and the difference in October -/
theorem total_savings (september : ℕ) (difference : ℕ) : 
  september = 260 → difference = 30 → september + (september + difference) = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_l1872_187202


namespace NUMINAMATH_CALUDE_spelling_bee_points_l1872_187250

theorem spelling_bee_points (max_points : ℕ) : max_points = 5 := by
  -- Define Dulce's points
  let dulce_points : ℕ := 3

  -- Define Val's points in terms of Max and Dulce's points
  let val_points : ℕ := 2 * (max_points + dulce_points)

  -- Define the total points of Max's team
  let team_points : ℕ := max_points + dulce_points + val_points

  -- Define the opponents' team points
  let opponents_points : ℕ := 40

  -- Express that Max's team is behind by 16 points
  have team_difference : team_points = opponents_points - 16 := by sorry

  -- Prove that max_points = 5
  sorry

end NUMINAMATH_CALUDE_spelling_bee_points_l1872_187250


namespace NUMINAMATH_CALUDE_g_of_5_l1872_187231

def g (x : ℚ) : ℚ := (3 * x + 2) / (x - 2)

theorem g_of_5 : g 5 = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_g_of_5_l1872_187231


namespace NUMINAMATH_CALUDE_river_depth_l1872_187247

/-- Given a river with specified width, flow rate, and volume flow rate, calculate its depth. -/
theorem river_depth (width : ℝ) (flow_rate_kmph : ℝ) (volume_flow_rate : ℝ) :
  width = 75 →
  flow_rate_kmph = 4 →
  volume_flow_rate = 35000 →
  (volume_flow_rate / (flow_rate_kmph * 1000 / 60) / width) = 7 := by
  sorry

#check river_depth

end NUMINAMATH_CALUDE_river_depth_l1872_187247


namespace NUMINAMATH_CALUDE_marble_picking_ways_l1872_187276

/-- The number of ways to pick up at least one marble from a set of marbles -/
def pick_marbles_ways (red green yellow black pink : ℕ) : ℕ :=
  ((red + 1) * (green + 1) * (yellow + 1) * (black + 1) * (pink + 1)) - 1

/-- Theorem: There are 95 ways to pick up at least one marble from a set
    containing 3 red marbles, 2 green marbles, and one each of yellow, black, and pink marbles -/
theorem marble_picking_ways :
  pick_marbles_ways 3 2 1 1 1 = 95 := by
  sorry

end NUMINAMATH_CALUDE_marble_picking_ways_l1872_187276


namespace NUMINAMATH_CALUDE_students_in_both_clubs_count_l1872_187220

/-- Represents the number of students in both drama and art clubs -/
def students_in_both_clubs (total : ℕ) (drama : ℕ) (art : ℕ) (drama_or_art : ℕ) : ℕ :=
  drama + art - drama_or_art

/-- Theorem stating the number of students in both drama and art clubs -/
theorem students_in_both_clubs_count : 
  students_in_both_clubs 300 120 150 220 = 50 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_count_l1872_187220


namespace NUMINAMATH_CALUDE_cost_of_melons_l1872_187290

/-- The cost of a single melon in dollars -/
def cost_per_melon : ℕ := 3

/-- The number of melons we want to calculate the cost for -/
def num_melons : ℕ := 6

/-- Theorem stating that the cost of 6 melons is $18 -/
theorem cost_of_melons : cost_per_melon * num_melons = 18 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_melons_l1872_187290


namespace NUMINAMATH_CALUDE_power_function_increasing_m_l1872_187237

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x^b

-- Define an increasing function on (0, +∞)
def isIncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

-- The main theorem
theorem power_function_increasing_m (m : ℝ) :
  let f := fun x : ℝ => (m^2 - m - 1) * x^m
  isPowerFunction f ∧ isIncreasingOn f → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_m_l1872_187237


namespace NUMINAMATH_CALUDE_min_value_theorem_l1872_187253

/-- Given a function y = a^(1-x) where a > 0 and a ≠ 1, 
    and a point A that lies on both the graph of the function and the line mx + ny - 1 = 0,
    where mn > 0, prove that the minimum value of 1/m + 2/n is 3 + 2√2. -/
theorem min_value_theorem (a : ℝ) (m n : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  (∀ m' n', m' * n' > 0 → m' + n' = 1 → 1 / m' + 2 / n' ≥ 3 + 2 * Real.sqrt 2) ∧
  (∃ m' n', m' * n' > 0 ∧ m' + n' = 1 ∧ 1 / m' + 2 / n' = 3 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1872_187253
