import Mathlib

namespace NUMINAMATH_CALUDE_dihedral_angle_in_unit_cube_l262_26214

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Calculates the dihedral angle between two planes in a cube -/
def dihedralAngle (cube : Cube) : ℝ :=
  sorry

/-- Theorem: The dihedral angle between planes ABD₁ and A₁B₁C₁ in a unit cube is 60° -/
theorem dihedral_angle_in_unit_cube :
  ∀ (cube : Cube),
    (cube.A.x = 0 ∧ cube.A.y = 0 ∧ cube.A.z = 0) →
    (cube.B.x = 1 ∧ cube.B.y = 0 ∧ cube.B.z = 0) →
    (cube.C.x = 1 ∧ cube.C.y = 1 ∧ cube.C.z = 0) →
    (cube.D.x = 0 ∧ cube.D.y = 1 ∧ cube.D.z = 0) →
    (cube.E₁.x = 0 ∧ cube.E₁.y = 0 ∧ cube.E₁.z = 1) →
    (cube.B₁.x = 1 ∧ cube.B₁.y = 0 ∧ cube.B₁.z = 1) →
    (cube.C₁.x = 1 ∧ cube.C₁.y = 1 ∧ cube.C₁.z = 1) →
    (cube.D₁.x = 0 ∧ cube.D₁.y = 1 ∧ cube.D₁.z = 1) →
    dihedralAngle cube = 60 * π / 180 :=
  by sorry

end NUMINAMATH_CALUDE_dihedral_angle_in_unit_cube_l262_26214


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l262_26212

def jeff_scores : List ℝ := [85, 94, 87, 93, 95, 88, 90]

theorem jeff_scores_mean : 
  (jeff_scores.sum / jeff_scores.length : ℝ) = 90.2857142857 := by
  sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l262_26212


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l262_26241

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {2, 3, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l262_26241


namespace NUMINAMATH_CALUDE_acute_slope_implies_a_is_one_l262_26282

/-- The curve C is defined by y = x³ - 2ax² + 2ax -/
def C (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x^2 + 2*a*x

/-- The derivative of C with respect to x -/
def C_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 4*a*x + 2*a

/-- The slope is acute if it's greater than 0 -/
def is_slope_acute (slope : ℝ) : Prop := slope > 0

theorem acute_slope_implies_a_is_one :
  ∀ a : ℤ, (∀ x : ℝ, is_slope_acute (C_derivative a x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_acute_slope_implies_a_is_one_l262_26282


namespace NUMINAMATH_CALUDE_total_skips_is_450_l262_26254

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of times Sally can skip a rock -/
def sally_skips : ℕ := 18

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips for all three people -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped + sally_skips * rocks_skipped

theorem total_skips_is_450 : total_skips = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_skips_is_450_l262_26254


namespace NUMINAMATH_CALUDE_fraction_evaluation_l262_26285

theorem fraction_evaluation : 
  (1 / 5 + 1 / 3) / (3 / 7 - 1 / 4) = 224 / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l262_26285


namespace NUMINAMATH_CALUDE_factorization_of_5a_cubed_minus_125a_l262_26262

theorem factorization_of_5a_cubed_minus_125a (a : ℝ) :
  5 * a^3 - 125 * a = 5 * a * (a + 5) * (a - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_5a_cubed_minus_125a_l262_26262


namespace NUMINAMATH_CALUDE_complement_of_intersection_l262_26250

def U : Finset ℕ := {1, 2, 3, 4, 6}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 3, 4}

theorem complement_of_intersection (U A B : Finset ℕ) 
  (hU : U = {1, 2, 3, 4, 6})
  (hA : A = {1, 2, 3})
  (hB : B = {2, 3, 4}) :
  U \ (A ∩ B) = {1, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l262_26250


namespace NUMINAMATH_CALUDE_parabola_symmetry_condition_l262_26277

theorem parabola_symmetry_condition (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    y₁ = a * x₁^2 - 1 ∧ 
    y₂ = a * x₂^2 - 1 ∧ 
    x₁ + y₁ = -(x₂ + y₂) ∧ 
    x₁ ≠ x₂) → 
  a > 3/4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_condition_l262_26277


namespace NUMINAMATH_CALUDE_baker_earnings_calculation_l262_26232

def cakes_sold : ℕ := 453
def cake_price : ℕ := 12
def pies_sold : ℕ := 126
def pie_price : ℕ := 7

def baker_earnings : ℕ := cakes_sold * cake_price + pies_sold * pie_price

theorem baker_earnings_calculation : baker_earnings = 6318 := by
  sorry

end NUMINAMATH_CALUDE_baker_earnings_calculation_l262_26232


namespace NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l262_26252

/-- Represents the meeting point of Jack and Jill on their hill run. -/
structure MeetingPoint where
  /-- The time at which Jack and Jill meet, measured from Jill's start time. -/
  time : ℝ
  /-- The distance from the start point where Jack and Jill meet. -/
  distance : ℝ

/-- Calculates the meeting point of Jack and Jill given their running conditions. -/
def calculateMeetingPoint (totalDistance jackHeadStart uphillDistance : ℝ)
                          (jackUphillSpeed jackDownhillSpeed : ℝ)
                          (jillUphillSpeed jillDownhillSpeed : ℝ) : MeetingPoint :=
  sorry

/-- Theorem stating that Jack and Jill meet 2 km from the top of the hill. -/
theorem jack_and_jill_meeting_point :
  let meetingPoint := calculateMeetingPoint 12 (2/15) 7 12 18 14 20
  meetingPoint.distance = 5 ∧ uphillDistance - meetingPoint.distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_jack_and_jill_meeting_point_l262_26252


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l262_26278

theorem arithmetic_calculations :
  (4.6 - (1.75 + 2.08) = 0.77) ∧
  (9.5 + 4.85 - 6.36 = 7.99) ∧
  (5.6 + 2.7 + 4.4 = 12.7) ∧
  (13 - 4.85 - 3.15 = 5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l262_26278


namespace NUMINAMATH_CALUDE_complex_parts_of_one_plus_sqrt_three_i_l262_26266

theorem complex_parts_of_one_plus_sqrt_three_i :
  let z : ℂ := Complex.I * (1 + Real.sqrt 3)
  Complex.re z = 0 ∧ Complex.im z = 1 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_parts_of_one_plus_sqrt_three_i_l262_26266


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l262_26251

theorem quadratic_inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - (a + 2)*x + 2*a > 0}
  (a > 2 → solution_set = {x : ℝ | x < 2 ∨ x > a}) ∧
  (a = 2 → solution_set = {x : ℝ | x ≠ 2}) ∧
  (a < 2 → solution_set = {x : ℝ | x < a ∨ x > 2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l262_26251


namespace NUMINAMATH_CALUDE_base_difference_equals_7422_l262_26222

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The main theorem --/
theorem base_difference_equals_7422 :
  let base_7_num := to_base_10 [3, 4, 1, 2, 5] 7
  let base_8_num := to_base_10 [5, 4, 3, 2, 1] 8
  base_7_num - base_8_num = 7422 := by sorry

end NUMINAMATH_CALUDE_base_difference_equals_7422_l262_26222


namespace NUMINAMATH_CALUDE_corner_square_length_l262_26295

/-- Given a rectangular sheet of dimensions 48 m x 36 m, if squares of side length x
    are cut from each corner to form an open box with volume 5120 m³, then x = 8 meters. -/
theorem corner_square_length (x : ℝ) : 
  x > 0 ∧ x < 24 ∧ x < 18 →
  (48 - 2*x) * (36 - 2*x) * x = 5120 →
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_corner_square_length_l262_26295


namespace NUMINAMATH_CALUDE_foreign_language_selection_l262_26223

theorem foreign_language_selection (total : ℕ) (english_speakers : ℕ) (japanese_speakers : ℕ) :
  total = 9 ∧ english_speakers = 5 ∧ japanese_speakers = 4 →
  english_speakers * japanese_speakers = 20 := by
sorry

end NUMINAMATH_CALUDE_foreign_language_selection_l262_26223


namespace NUMINAMATH_CALUDE_base_number_proof_l262_26211

theorem base_number_proof (x : ℝ) : x^8 = 4^16 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l262_26211


namespace NUMINAMATH_CALUDE_quadratic_roots_large_difference_l262_26267

theorem quadratic_roots_large_difference :
  ∃ (p q p' q' u v u' v' : ℝ),
    (u > v) ∧ (u' > v') ∧
    (u^2 + p*u + q = 0) ∧
    (v^2 + p*v + q = 0) ∧
    (u'^2 + p'*u' + q' = 0) ∧
    (v'^2 + p'*v' + q' = 0) ∧
    (|p' - p| < 0.01) ∧
    (|q' - q| < 0.01) ∧
    (|u' - u| > 10000) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_large_difference_l262_26267


namespace NUMINAMATH_CALUDE_hyperbola_foci_l262_26260

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

def is_focus (x y : ℝ) : Prop :=
  (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)

theorem hyperbola_foci :
  ∀ x y : ℝ, hyperbola_equation x y → is_focus x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l262_26260


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l262_26292

theorem cone_lateral_surface_area (slant_height height : Real) 
  (h1 : slant_height = 15)
  (h2 : height = 9) :
  let radius := Real.sqrt (slant_height^2 - height^2)
  π * radius * slant_height = 180 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l262_26292


namespace NUMINAMATH_CALUDE_untouched_produce_count_l262_26229

/-- The number of untouched tomatoes and cucumbers after processing -/
def untouched_produce (tomato_plants : ℕ) (tomatoes_per_plant : ℕ) (cucumbers : ℕ) : ℕ :=
  let total_tomatoes := tomato_plants * tomatoes_per_plant
  let dried_tomatoes := (2 * total_tomatoes) / 3
  let remaining_tomatoes := total_tomatoes - dried_tomatoes
  let sauce_tomatoes := remaining_tomatoes / 2
  let untouched_tomatoes := remaining_tomatoes - sauce_tomatoes
  let pickled_cucumbers := cucumbers / 4
  let untouched_cucumbers := cucumbers - pickled_cucumbers
  untouched_tomatoes + untouched_cucumbers

/-- Theorem stating the number of untouched produce given the conditions -/
theorem untouched_produce_count :
  untouched_produce 50 15 25 = 143 := by
  sorry


end NUMINAMATH_CALUDE_untouched_produce_count_l262_26229


namespace NUMINAMATH_CALUDE_points_on_line_procedure_l262_26249

theorem points_on_line_procedure (n : ℕ) : ∃ n, 9 * n - 8 = 82 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_procedure_l262_26249


namespace NUMINAMATH_CALUDE_equation_solution_l262_26275

theorem equation_solution : ∃ x : ℚ, x - (x + 2) / 2 = (2 * x - 1) / 3 - 1 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l262_26275


namespace NUMINAMATH_CALUDE_floor_length_l262_26200

/-- Represents the dimensions of a rectangular floor -/
structure FloorDimensions where
  breadth : ℝ
  length : ℝ

/-- The properties of the floor as given in the problem -/
def FloorProperties (d : FloorDimensions) : Prop :=
  d.length = 3 * d.breadth ∧ d.length * d.breadth = 156

/-- Theorem stating the length of the floor -/
theorem floor_length (d : FloorDimensions) (h : FloorProperties d) : 
  d.length = 6 * Real.sqrt 13 := by
  sorry


end NUMINAMATH_CALUDE_floor_length_l262_26200


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l262_26264

def fibonacci_factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 6
  | 4 => 24
  | 5 => 120
  | 6 => 720
  | 7 => 5040
  | 8 => 40320
  | 9 => 362880
  | _ => 0  -- For n ≥ 10, we only care about the last two digits, which are 00

def modified_fibonacci_series : List ℕ :=
  [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 55]

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem sum_of_last_two_digits_of_modified_fibonacci_factorial_series :
  (modified_fibonacci_series.map (λ x => last_two_digits (fibonacci_factorial x))).sum % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_modified_fibonacci_factorial_series_l262_26264


namespace NUMINAMATH_CALUDE_ball_distribution_l262_26269

theorem ball_distribution (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 3) :
  (Nat.choose (n + k - 1 - k) (k - 1)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_l262_26269


namespace NUMINAMATH_CALUDE_symmetric_points_product_l262_26284

/-- 
If point A (2008, y) and point B (x, -1) are symmetric about the origin,
then xy = -2008.
-/
theorem symmetric_points_product (x y : ℝ) : 
  (2008 = -x ∧ y = 1) → x * y = -2008 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_product_l262_26284


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l262_26272

theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 240 ∧ 
  6 * a = m ∧ 
  b - 12 = m ∧ 
  c + 12 = m ∧ 
  a ≤ c ∧ 
  c ≤ b → 
  a * b * c = 490108320 / 2197 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l262_26272


namespace NUMINAMATH_CALUDE_girls_from_pine_l262_26270

theorem girls_from_pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_boys : ℕ)
  (h1 : total_students = 120)
  (h2 : total_boys = 70)
  (h3 : total_girls = 50)
  (h4 : maple_students = 50)
  (h5 : pine_students = 70)
  (h6 : maple_boys = 25)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = maple_students + pine_students)
  (h9 : maple_students = maple_boys + (total_girls - (pine_students - (total_boys - maple_boys)))) :
  pine_students - (total_boys - maple_boys) = 25 := by
  sorry

end NUMINAMATH_CALUDE_girls_from_pine_l262_26270


namespace NUMINAMATH_CALUDE_remainder_theorem_l262_26224

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 - 2*x^3 + 4*x^2 + x + 5

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, p = λ x => (x + 2) * q x + 3 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l262_26224


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l262_26268

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 2*x > 0} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l262_26268


namespace NUMINAMATH_CALUDE_inequality_proof_l262_26226

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_sum : a*b + b*c + c*d + d*a = 1) : 
  a^3 / (b+c+d) + b^3 / (a+c+d) + c^3 / (a+b+d) + d^3 / (a+b+c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l262_26226


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l262_26219

theorem unique_solution_for_equation : ∃! (x y z : ℕ), 
  x < 10 ∧ y < 10 ∧ z < 10 ∧ 
  (10 * x + 5) * (300 + 10 * y + z) = 7850 ∧
  x = 2 ∧ y = 1 ∧ z = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l262_26219


namespace NUMINAMATH_CALUDE_certain_number_divisibility_l262_26248

theorem certain_number_divisibility (m : ℕ+) 
  (h1 : ∃ (k : ℕ+), m = 8 * k) 
  (h2 : ∀ (d : ℕ+), d ∣ m → d ≤ 8) : 
  64 ∣ m^2 ∧ ∀ (n : ℕ+), (∀ (k : ℕ+), n ∣ (8*k)^2) → n ≤ 64 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisibility_l262_26248


namespace NUMINAMATH_CALUDE_unique_solution_l262_26255

theorem unique_solution (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq1 : x*y + y*z + z*x = 12)
  (eq2 : x*y*z = 2 + x + y + z) :
  x = 2 ∧ y = 2 ∧ z = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l262_26255


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l262_26242

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, real axis length 2, and focal distance 4,
    prove that its asymptotes are y = ±√3 x -/
theorem hyperbola_asymptotes (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (2 * a = 2) →  -- real axis length is 2
  (4 = 2 * Real.sqrt (a^2 + b^2)) →  -- focal distance is 4
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l262_26242


namespace NUMINAMATH_CALUDE_sum_lower_bound_l262_26246

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : 1/a + 4/b = 1) :
  ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l262_26246


namespace NUMINAMATH_CALUDE_farm_ploughing_problem_l262_26265

/-- Calculates the remaining area to be ploughed given the total area, planned and actual ploughing rates, and additional days worked. -/
def remaining_area (total_area planned_rate actual_rate extra_days : ℕ) : ℕ :=
  let planned_days := total_area / planned_rate
  let actual_days := planned_days + extra_days
  let ploughed_area := actual_rate * actual_days
  total_area - ploughed_area

/-- Theorem stating that given the specific conditions of the farm problem, the remaining area to be ploughed is 40 hectares. -/
theorem farm_ploughing_problem :
  remaining_area 720 120 85 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_farm_ploughing_problem_l262_26265


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l262_26258

/-- RegularOctagon represents a regular octagon with center O and vertices A to H -/
structure RegularOctagon where
  O : Point
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  G : Point
  H : Point

/-- Given a regular octagon, returns the area of the specified region -/
def shaded_area (octagon : RegularOctagon) : ℝ :=
  sorry

/-- The total area of the regular octagon -/
def total_area (octagon : RegularOctagon) : ℝ :=
  sorry

/-- Theorem stating that the shaded area is 5/8 of the total area -/
theorem shaded_area_fraction (octagon : RegularOctagon) :
  shaded_area octagon / total_area octagon = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l262_26258


namespace NUMINAMATH_CALUDE_max_salary_460000_l262_26213

/-- Represents a hockey team -/
structure HockeyTeam where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player in a hockey team -/
def maxPlayerSalary (team : HockeyTeam) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a player in the given conditions -/
theorem max_salary_460000 (team : HockeyTeam) 
  (h1 : team.players = 18)
  (h2 : team.minSalary = 20000)
  (h3 : team.maxTotalSalary = 800000) : 
  maxPlayerSalary team = 460000 := by
sorry

#eval maxPlayerSalary { players := 18, minSalary := 20000, maxTotalSalary := 800000 }

end NUMINAMATH_CALUDE_max_salary_460000_l262_26213


namespace NUMINAMATH_CALUDE_unique_invalid_triangle_l262_26261

/-- Represents the ratio of altitudes of a triangle -/
structure AltitudeRatio where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a triangle with given side lengths satisfies the triangle inequality -/
def satisfiesTriangleInequality (x y z : ℚ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Converts altitude ratios to side length ratios -/
def toSideLengthRatio (ar : AltitudeRatio) : (ℚ × ℚ × ℚ) :=
  (1 / ar.a, 1 / ar.b, 1 / ar.c)

/-- Theorem stating that among the given altitude ratios, only 1:2:3 violates the triangle inequality -/
theorem unique_invalid_triangle (ar : AltitudeRatio) : 
  (ar = ⟨1, 1, 2⟩ ∨ ar = ⟨1, 2, 3⟩ ∨ ar = ⟨2, 3, 4⟩ ∨ ar = ⟨3, 4, 5⟩) →
  (¬satisfiesTriangleInequality (toSideLengthRatio ar).1 (toSideLengthRatio ar).2.1 (toSideLengthRatio ar).2.2 ↔ ar = ⟨1, 2, 3⟩) :=
sorry

end NUMINAMATH_CALUDE_unique_invalid_triangle_l262_26261


namespace NUMINAMATH_CALUDE_sam_and_billy_total_money_l262_26273

/-- Given that Sam has $75 and Billy has $25 less than twice Sam's money, 
    prove that their total money is $200. -/
theorem sam_and_billy_total_money :
  ∀ (sam_money billy_money : ℕ),
    sam_money = 75 →
    billy_money = 2 * sam_money - 25 →
    sam_money + billy_money = 200 := by
sorry

end NUMINAMATH_CALUDE_sam_and_billy_total_money_l262_26273


namespace NUMINAMATH_CALUDE_expression_simplification_l262_26235

theorem expression_simplification (x y z : ℝ) :
  (x - (2 * y + z)) - ((x + 2 * y) - 3 * z) = -4 * y + 2 * z := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l262_26235


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l262_26201

theorem new_average_age_with_teacher (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℕ) :
  num_students = 40 →
  student_avg_age = 15 →
  teacher_age = 56 →
  (num_students : ℝ) * student_avg_age + teacher_age = 16 * (num_students + 1) := by
  sorry


end NUMINAMATH_CALUDE_new_average_age_with_teacher_l262_26201


namespace NUMINAMATH_CALUDE_polynomial_intersection_l262_26227

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- Define the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct
  (∃ x, f a b x ≠ g c d x) →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (b - a^2/4 = d - c^2/4) →
  -- The graphs of f and g intersect at the point (2012, -2012)
  f a b 2012 = -2012 ∧ g c d 2012 = -2012 →
  -- Conclusion
  a + c = -8048 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l262_26227


namespace NUMINAMATH_CALUDE_lucas_chocolate_theorem_l262_26271

/-- Represents the number of pieces of chocolate candy Lucas makes for each student. -/
def pieces_per_student : ℕ := 4

/-- Represents the total number of pieces of chocolate candy Lucas made last Monday. -/
def total_pieces_last_monday : ℕ := 40

/-- Represents the number of students who will not be coming to class this upcoming Monday. -/
def absent_students : ℕ := 3

/-- Calculates the number of pieces of chocolate candy Lucas will make for his class on the upcoming Monday. -/
def pieces_for_upcoming_monday : ℕ :=
  ((total_pieces_last_monday / pieces_per_student) - absent_students) * pieces_per_student

/-- Theorem stating that Lucas will make 28 pieces of chocolate candy for his class on the upcoming Monday. -/
theorem lucas_chocolate_theorem :
  pieces_for_upcoming_monday = 28 := by sorry

end NUMINAMATH_CALUDE_lucas_chocolate_theorem_l262_26271


namespace NUMINAMATH_CALUDE_contest_possible_orders_l262_26233

/-- The number of questions in the contest -/
def num_questions : ℕ := 10

/-- The number of possible orders to answer the questions -/
def num_possible_orders : ℕ := 512

/-- Theorem stating that the number of possible orders is correct -/
theorem contest_possible_orders :
  (2 ^ (num_questions - 1) : ℕ) = num_possible_orders := by
  sorry

end NUMINAMATH_CALUDE_contest_possible_orders_l262_26233


namespace NUMINAMATH_CALUDE_unique_solution_is_negation_f_is_bijective_l262_26293

/-- A function satisfying the given functional equation. -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y + 1) * f x + f (x * f y + f (x + y)) = y

/-- The main theorem stating that f(x) = -x is the unique solution. -/
theorem unique_solution_is_negation (f : ℝ → ℝ) 
    (h : SatisfiesFunctionalEquation f) : 
    f = fun x ↦ -x := by
  sorry

/-- f is bijective -/
theorem f_is_bijective (f : ℝ → ℝ) 
    (h : SatisfiesFunctionalEquation f) : 
    Function.Bijective f := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_negation_f_is_bijective_l262_26293


namespace NUMINAMATH_CALUDE_negative_abs_two_squared_equals_two_l262_26245

theorem negative_abs_two_squared_equals_two : (-|2|)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_two_squared_equals_two_l262_26245


namespace NUMINAMATH_CALUDE_seashells_sum_l262_26296

/-- The number of seashells found by Mary and Keith -/
def total_seashells (mary_seashells keith_seashells : ℕ) : ℕ :=
  mary_seashells + keith_seashells

/-- Theorem stating that the total number of seashells is the sum of Mary's and Keith's seashells -/
theorem seashells_sum (mary_seashells keith_seashells : ℕ) :
  total_seashells mary_seashells keith_seashells = mary_seashells + keith_seashells :=
by sorry

end NUMINAMATH_CALUDE_seashells_sum_l262_26296


namespace NUMINAMATH_CALUDE_first_bakery_sacks_proof_l262_26243

/-- The number of sacks the second bakery needs per week -/
def second_bakery_sacks : ℕ := 4

/-- The number of sacks the third bakery needs per week -/
def third_bakery_sacks : ℕ := 12

/-- The total number of weeks -/
def total_weeks : ℕ := 4

/-- The total number of sacks needed for all bakeries in 4 weeks -/
def total_sacks : ℕ := 72

/-- The number of sacks the first bakery needs per week -/
def first_bakery_sacks : ℕ := 2

theorem first_bakery_sacks_proof :
  first_bakery_sacks * total_weeks + 
  second_bakery_sacks * total_weeks + 
  third_bakery_sacks * total_weeks = total_sacks :=
by sorry

end NUMINAMATH_CALUDE_first_bakery_sacks_proof_l262_26243


namespace NUMINAMATH_CALUDE_triangle_with_given_altitudes_exists_l262_26220

theorem triangle_with_given_altitudes_exists (m_a m_b : ℝ) 
  (h1 : 0 < m_a) (h2 : 0 < m_b) (h3 : m_a ≤ m_b) :
  (∃ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧
    a + b > c ∧ b + c > a ∧ c + a > b ∧
    m_a = (2 * (a * b * c / (a + b + c))) / a ∧
    m_b = (2 * (a * b * c / (a + b + c))) / b ∧
    m_a + m_b = (2 * (a * b * c / (a + b + c))) / c) ↔
  (m_a / m_b)^2 + (m_a / m_b) > 1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_given_altitudes_exists_l262_26220


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l262_26279

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15 ∧ |x₂ + 3| = 15) ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_difference_l262_26279


namespace NUMINAMATH_CALUDE_free_throw_probability_l262_26208

/-- The probability of making a single shot -/
def p : ℝ := sorry

/-- The probability of passing the test (making at least one shot out of three chances) -/
def prob_pass : ℝ := p + p * (1 - p) + p * (1 - p)^2

/-- Theorem stating that if the probability of passing is 0.784, then p is 0.4 -/
theorem free_throw_probability : prob_pass = 0.784 → p = 0.4 := by sorry

end NUMINAMATH_CALUDE_free_throw_probability_l262_26208


namespace NUMINAMATH_CALUDE_difference_calculation_l262_26210

theorem difference_calculation (total : ℝ) (h : total = 6000) : 
  (1 / 10 * total) - (1 / 1000 * total) = 594 := by
  sorry

end NUMINAMATH_CALUDE_difference_calculation_l262_26210


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l262_26209

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), 
    (∀ x : ℤ, (3*(x-1) > x-6 ∧ 8-2*x+2*a ≥ 0) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) ∧
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  -3 ≤ a ∧ a < -2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l262_26209


namespace NUMINAMATH_CALUDE_alice_box_height_l262_26299

/-- The height of the box Alice needs to reach the light bulb -/
def box_height (ceiling_height room_height alice_height alice_reach light_bulb_distance shelf_distance : ℝ) : ℝ :=
  ceiling_height - light_bulb_distance - (alice_height + alice_reach)

/-- Proof that Alice needs a 75 cm box to reach the light bulb -/
theorem alice_box_height :
  let ceiling_height : ℝ := 300  -- cm
  let room_height : ℝ := 300     -- cm
  let alice_height : ℝ := 160    -- cm
  let alice_reach : ℝ := 50      -- cm
  let light_bulb_distance : ℝ := 15  -- cm from ceiling
  let shelf_distance : ℝ := 10   -- cm below light bulb
  box_height ceiling_height room_height alice_height alice_reach light_bulb_distance shelf_distance = 75 := by
  sorry


end NUMINAMATH_CALUDE_alice_box_height_l262_26299


namespace NUMINAMATH_CALUDE_money_distribution_l262_26218

theorem money_distribution (a b c total : ℕ) : 
  a + b + c = total →
  2 * b = 3 * a →
  4 * b = 3 * c →
  b = 600 →
  total = 1800 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l262_26218


namespace NUMINAMATH_CALUDE_library_visitors_l262_26281

/-- Proves that the average number of visitors on Sundays is 510 given the conditions -/
theorem library_visitors (total_days : Nat) (sunday_count : Nat) (avg_visitors : Nat) (non_sunday_visitors : Nat) :
  total_days = 30 ∧ 
  sunday_count = 5 ∧ 
  avg_visitors = 285 ∧ 
  non_sunday_visitors = 240 →
  (sunday_count * (total_days * avg_visitors - (total_days - sunday_count) * non_sunday_visitors)) / 
  (sunday_count * total_days) = 510 := by
  sorry


end NUMINAMATH_CALUDE_library_visitors_l262_26281


namespace NUMINAMATH_CALUDE_fraction_simplification_l262_26259

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (5 - 4 * x) / 3 = (-13 * x + 26) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l262_26259


namespace NUMINAMATH_CALUDE_standard_deviation_decreases_after_correction_l262_26240

/-- Represents a class with test scores -/
structure TestScores where
  size : ℕ
  average : ℝ
  standardDev : ℝ

/-- Represents a score correction -/
structure ScoreCorrection where
  oldScore : ℝ
  newScore : ℝ

/-- The main theorem stating that the original standard deviation is greater than the new one after corrections -/
theorem standard_deviation_decreases_after_correction 
  (original : TestScores)
  (correction1 correction2 : ScoreCorrection)
  (new_std_dev : ℝ)
  (h_size : original.size = 50)
  (h_avg : original.average = 70)
  (h_correction1 : correction1.oldScore = 50 ∧ correction1.newScore = 80)
  (h_correction2 : correction2.oldScore = 100 ∧ correction2.newScore = 70)
  : original.standardDev > new_std_dev := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_decreases_after_correction_l262_26240


namespace NUMINAMATH_CALUDE_rent_increase_effect_rent_problem_l262_26288

theorem rent_increase_effect (num_friends : ℕ) (initial_avg_rent : ℚ) 
  (increased_rent : ℚ) (increase_percentage : ℚ) : ℚ :=
  let total_initial_rent := num_friends * initial_avg_rent
  let rent_increase := increased_rent * increase_percentage
  let new_total_rent := total_initial_rent + rent_increase
  let new_avg_rent := new_total_rent / num_friends
  new_avg_rent

theorem rent_problem :
  rent_increase_effect 4 800 1600 (1/5) = 880 := by sorry

end NUMINAMATH_CALUDE_rent_increase_effect_rent_problem_l262_26288


namespace NUMINAMATH_CALUDE_trivia_team_groups_l262_26221

/-- Given a total number of students, number of students not picked, and students per group,
    calculate the number of groups formed. -/
def calculate_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) : ℕ :=
  (total_students - not_picked) / students_per_group

/-- Theorem stating that with 17 total students, 5 not picked, and 4 per group, 3 groups are formed. -/
theorem trivia_team_groups : calculate_groups 17 5 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l262_26221


namespace NUMINAMATH_CALUDE_surface_area_reduction_approx_l262_26256

/-- The number of faces in a single cube -/
def cube_faces : ℕ := 6

/-- The number of faces lost when splicing two cubes into a cuboid -/
def faces_lost : ℕ := 2

/-- The percentage reduction in surface area when splicing two cubes into a cuboid -/
def surface_area_reduction : ℚ :=
  (faces_lost : ℚ) / (2 * cube_faces : ℚ) * 100

theorem surface_area_reduction_approx :
  ∃ ε > 0, abs (surface_area_reduction - 167/10) < ε ∧ ε < 1/10 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_reduction_approx_l262_26256


namespace NUMINAMATH_CALUDE_correct_inscription_l262_26238

-- Define the type for box makers
inductive BoxMaker
| Bellini
| Cellini
| Other

-- Define a box
structure Box where
  maker : BoxMaker
  inscription : String

-- Define the problem conditions
def validInscription (inscription : String) : Prop :=
  ∃ (box1 box2 : Box),
    (box1.inscription = inscription) ∧
    (box2.inscription = inscription) ∧
    (box1.maker = BoxMaker.Bellini ∧ box2.maker = BoxMaker.Bellini) ∧
    (∀ (b1 b2 : Box), b1.inscription = inscription → b2.inscription = inscription →
      ((b1.maker = BoxMaker.Bellini ∧ b2.maker = BoxMaker.Bellini) ∨
       (b1.maker = BoxMaker.Cellini ∨ b2.maker = BoxMaker.Cellini)))

-- The theorem to be proved
theorem correct_inscription :
  validInscription "Either both caskets are made by Bellini, or at least one of them is made by a member of the Cellini family" :=
sorry

end NUMINAMATH_CALUDE_correct_inscription_l262_26238


namespace NUMINAMATH_CALUDE_line_equation_l262_26290

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + y - 8 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line3 (x y : ℝ) : Prop := 4*x - 3*y - 7 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallelism
def parallel (a b c d e f : ℝ) : Prop := a*e = b*d

-- Define the line l
def line_l (x y : ℝ) : Prop := 4*x - 3*y - 6 = 0

-- Theorem statement
theorem line_equation : 
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧ 
  parallel 4 (-3) (-6) 4 (-3) (-7) ∧
  line_l x₀ y₀ := by sorry

end NUMINAMATH_CALUDE_line_equation_l262_26290


namespace NUMINAMATH_CALUDE_football_game_spectators_l262_26205

theorem football_game_spectators (total_wristbands : ℕ) 
  (wristbands_per_person : ℕ) (h1 : total_wristbands = 234) 
  (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 117 := by
  sorry

end NUMINAMATH_CALUDE_football_game_spectators_l262_26205


namespace NUMINAMATH_CALUDE_stratified_sample_proportion_l262_26203

/-- Calculates the number of teachers under 40 in a stratified sample -/
def teachersUnder40InSample (totalTeachers : ℕ) (under40Teachers : ℕ) (sampleSize : ℕ) : ℕ :=
  (under40Teachers * sampleSize) / totalTeachers

theorem stratified_sample_proportion 
  (totalTeachers : ℕ) 
  (under40Teachers : ℕ) 
  (over40Teachers : ℕ) 
  (sampleSize : ℕ) :
  totalTeachers = 490 →
  under40Teachers = 350 →
  over40Teachers = 140 →
  sampleSize = 70 →
  totalTeachers = under40Teachers + over40Teachers →
  teachersUnder40InSample totalTeachers under40Teachers sampleSize = 50 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_proportion_l262_26203


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_8000_l262_26263

theorem greatest_multiple_of_four_under_cube_root_8000 :
  (∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^3 < 8000 ∧ ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 8000 → y ≤ x) →
  (∃ (x : ℕ), x = 16 ∧ x > 0 ∧ 4 ∣ x ∧ x^3 < 8000 ∧ ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 8000 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_8000_l262_26263


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l262_26298

theorem sine_cosine_inequality (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) :
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l262_26298


namespace NUMINAMATH_CALUDE_sum_of_four_digit_even_and_multiples_of_three_l262_26202

/-- The number of four-digit even numbers -/
def C : ℕ := 4500

/-- The number of four-digit multiples of 3 -/
def D : ℕ := 3000

/-- Theorem stating that the sum of four-digit even numbers and four-digit multiples of 3 is 7500 -/
theorem sum_of_four_digit_even_and_multiples_of_three :
  C + D = 7500 := by sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_even_and_multiples_of_three_l262_26202


namespace NUMINAMATH_CALUDE_moms_approach_is_sampling_survey_l262_26217

/-- Represents a method of data collection. -/
inductive DataCollectionMethod
| Census
| SamplingSurvey

/-- Represents the action of tasting food. -/
structure TastingAction where
  dish : String
  portion : String

/-- Determines the data collection method based on the tasting action. -/
def determineMethod (action : TastingAction) : DataCollectionMethod :=
  if action.portion = "entire" then DataCollectionMethod.Census
  else DataCollectionMethod.SamplingSurvey

theorem moms_approach_is_sampling_survey :
  let momsTasting : TastingAction := { dish := "cooking dish", portion := "little bit" }
  determineMethod momsTasting = DataCollectionMethod.SamplingSurvey := by
  sorry


end NUMINAMATH_CALUDE_moms_approach_is_sampling_survey_l262_26217


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l262_26294

/-- The capacity of a bucket in the first scenario, in litres. -/
def first_bucket_capacity : ℝ := 13.5

/-- The number of buckets required to fill the tank in the first scenario. -/
def first_scenario_buckets : ℕ := 28

/-- The number of buckets required to fill the tank in the second scenario. -/
def second_scenario_buckets : ℕ := 42

/-- The capacity of a bucket in the second scenario, in litres. -/
def second_bucket_capacity : ℝ := 9

theorem bucket_capacity_proof :
  first_bucket_capacity * first_scenario_buckets =
  second_bucket_capacity * second_scenario_buckets := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_proof_l262_26294


namespace NUMINAMATH_CALUDE_circle_area_decrease_l262_26276

theorem circle_area_decrease (r : ℝ) (hr : r > 0) :
  let original_area := π * r^2
  let new_radius := r / 2
  let new_area := π * new_radius^2
  (original_area - new_area) / original_area = 3/4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l262_26276


namespace NUMINAMATH_CALUDE_appetizer_cost_l262_26204

/-- Proves that the cost of the appetizer is $10 given the conditions of the restaurant bill --/
theorem appetizer_cost (entree_cost : ℝ) (entree_count : ℕ) (tip_rate : ℝ) (total_spent : ℝ) :
  entree_cost = 20 →
  entree_count = 4 →
  tip_rate = 0.2 →
  total_spent = 108 →
  ∃ (appetizer_cost : ℝ),
    appetizer_cost + entree_cost * entree_count + tip_rate * (appetizer_cost + entree_cost * entree_count) = total_spent ∧
    appetizer_cost = 10 := by
  sorry


end NUMINAMATH_CALUDE_appetizer_cost_l262_26204


namespace NUMINAMATH_CALUDE_hyperbola_min_value_l262_26283

theorem hyperbola_min_value (x y : ℝ) : 
  x^2 / 4 - y^2 = 1 → (∀ z w : ℝ, z^2 / 4 - w^2 = 1 → 3*x^2 - 2*y ≤ 3*z^2 - 2*w) ∧ (∃ a b : ℝ, a^2 / 4 - b^2 = 1 ∧ 3*a^2 - 2*b = 143/12) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_min_value_l262_26283


namespace NUMINAMATH_CALUDE_jake_read_225_pages_l262_26228

/-- The number of pages Jake read in a week -/
def pages_read : ℕ :=
  let day1 : ℕ := 45
  let day2 : ℕ := day1 / 3
  let day3 : ℕ := 58 - 12
  let day4 : ℕ := (day1 + 1) / 2  -- Rounding up
  let day5 : ℕ := (3 * day3 + 3) / 4  -- Rounding up
  let day6 : ℕ := day2
  let day7 : ℕ := 2 * day4
  day1 + day2 + day3 + day4 + day5 + day6 + day7

/-- Theorem stating that Jake read 225 pages in total -/
theorem jake_read_225_pages : pages_read = 225 := by
  sorry

end NUMINAMATH_CALUDE_jake_read_225_pages_l262_26228


namespace NUMINAMATH_CALUDE_bank_account_withdrawal_l262_26216

theorem bank_account_withdrawal (initial_balance : ℚ) : 
  (initial_balance > 0) →
  (initial_balance - 200 + (1/2) * (initial_balance - 200) = 450) →
  (200 / initial_balance = 2/5) := by
sorry

end NUMINAMATH_CALUDE_bank_account_withdrawal_l262_26216


namespace NUMINAMATH_CALUDE_derivative_of_one_plus_cos_2x_squared_l262_26236

theorem derivative_of_one_plus_cos_2x_squared (x : ℝ) :
  let y : ℝ → ℝ := λ x => (1 + Real.cos (2 * x))^2
  deriv y x = -4 * Real.sin (2 * x) - 2 * Real.sin (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_one_plus_cos_2x_squared_l262_26236


namespace NUMINAMATH_CALUDE_second_wheat_rate_l262_26253

-- Define the quantities and rates
def wheat1_quantity : ℝ := 30
def wheat1_rate : ℝ := 11.50
def wheat2_quantity : ℝ := 20
def profit_percentage : ℝ := 0.10
def mixture_sell_rate : ℝ := 13.86

-- Define the theorem
theorem second_wheat_rate (wheat2_rate : ℝ) : 
  wheat1_quantity * wheat1_rate + wheat2_quantity * wheat2_rate = 
  (wheat1_quantity + wheat2_quantity) * mixture_sell_rate / (1 + profit_percentage) →
  wheat2_rate = 14.25 := by
sorry

end NUMINAMATH_CALUDE_second_wheat_rate_l262_26253


namespace NUMINAMATH_CALUDE_quadratic_function_min_value_l262_26237

/-- Given a quadratic function f(x) = ax^2 + 2x + c with range [0, +∞),
    the minimum value of (a+1)/c + (c+1)/a is 4 -/
theorem quadratic_function_min_value (a c : ℝ) :
  (∀ x, a * x^2 + 2 * x + c ≥ 0) →
  a > 0 →
  c > 0 →
  (∃ x, a * x^2 + 2 * x + c = 0) →
  (a + 1) / c + (c + 1) / a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_min_value_l262_26237


namespace NUMINAMATH_CALUDE_inequality_system_solution_l262_26244

theorem inequality_system_solution (x : ℝ) :
  (5 * x + 1 ≥ 3 * (x - 1)) →
  (1 - (x + 3) / 3 ≤ x) →
  x ≥ 0 := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l262_26244


namespace NUMINAMATH_CALUDE_sally_balloons_l262_26231

def initial_orange_balloons : ℕ := sorry

def lost_balloons : ℕ := 2

def current_orange_balloons : ℕ := 7

theorem sally_balloons : initial_orange_balloons = current_orange_balloons + lost_balloons :=
by sorry

end NUMINAMATH_CALUDE_sally_balloons_l262_26231


namespace NUMINAMATH_CALUDE_fruit_arrangement_unique_l262_26225

-- Define the fruits
inductive Fruit
| Apple
| Pear
| Orange
| Banana

-- Define a type for box numbers
inductive BoxNumber
| One
| Two
| Three
| Four

-- Define a function type for fruit arrangements
def Arrangement := BoxNumber → Fruit

-- Define a predicate for the correctness of labels
def LabelIncorrect (arr : Arrangement) : Prop :=
  arr BoxNumber.One ≠ Fruit.Orange ∧
  arr BoxNumber.Two ≠ Fruit.Pear ∧
  (arr BoxNumber.One = Fruit.Banana → arr BoxNumber.Three ≠ Fruit.Apple ∧ arr BoxNumber.Three ≠ Fruit.Pear) ∧
  arr BoxNumber.Four ≠ Fruit.Apple

-- Define the correct arrangement
def CorrectArrangement : Arrangement :=
  fun b => match b with
  | BoxNumber.One => Fruit.Banana
  | BoxNumber.Two => Fruit.Apple
  | BoxNumber.Three => Fruit.Orange
  | BoxNumber.Four => Fruit.Pear

-- Theorem statement
theorem fruit_arrangement_unique :
  ∀ (arr : Arrangement),
    (∀ (b : BoxNumber), ∃! (f : Fruit), arr b = f) →
    LabelIncorrect arr →
    arr = CorrectArrangement :=
sorry

end NUMINAMATH_CALUDE_fruit_arrangement_unique_l262_26225


namespace NUMINAMATH_CALUDE_hockey_championship_points_l262_26230

/-- Represents the number of points a team receives for winning a game. -/
def win_points : ℕ := 2

/-- Represents the number of games tied. -/
def games_tied : ℕ := 12

/-- Represents the number of games won. -/
def games_won : ℕ := games_tied + 12

/-- Represents the points received for a tie. -/
def tie_points : ℕ := 1

theorem hockey_championship_points :
  win_points * games_won + tie_points * games_tied = 60 :=
sorry

end NUMINAMATH_CALUDE_hockey_championship_points_l262_26230


namespace NUMINAMATH_CALUDE_pyramid_arrangements_10_l262_26291

/-- The number of distinguishable ways to form a pyramid with n distinct pool balls -/
def pyramid_arrangements (n : ℕ) : ℕ :=
  n.factorial / 9

/-- The theorem stating that the number of distinguishable ways to form a pyramid
    with 10 distinct pool balls is 403,200 -/
theorem pyramid_arrangements_10 :
  pyramid_arrangements 10 = 403200 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_arrangements_10_l262_26291


namespace NUMINAMATH_CALUDE_no_solution_for_inequality_l262_26286

theorem no_solution_for_inequality (a b : ℝ) (h : |a - b| > 2) :
  ¬∃ x : ℝ, |x - a| + |x - b| ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_for_inequality_l262_26286


namespace NUMINAMATH_CALUDE_route_upper_bound_l262_26257

/-- Represents the number of possible routes in a grid city -/
def f (m n : ℕ) : ℕ := sorry

/-- Theorem: The number of possible routes in a grid city is at most 2^(m*n) -/
theorem route_upper_bound (m n : ℕ) : f m n ≤ 2^(m*n) := by sorry

end NUMINAMATH_CALUDE_route_upper_bound_l262_26257


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l262_26206

theorem negative_fraction_comparison : -3/5 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l262_26206


namespace NUMINAMATH_CALUDE_difference_of_squares_l262_26207

theorem difference_of_squares (m : ℝ) : m^2 - 9 = (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l262_26207


namespace NUMINAMATH_CALUDE_greatest_solution_sin_cos_equation_l262_26239

theorem greatest_solution_sin_cos_equation :
  ∃ (x : ℝ),
    x ∈ Set.Icc 0 (10 * Real.pi) ∧
    |2 * Real.sin x - 1| + |2 * Real.cos (2 * x) - 1| = 0 ∧
    (∀ (y : ℝ), y ∈ Set.Icc 0 (10 * Real.pi) →
      |2 * Real.sin y - 1| + |2 * Real.cos (2 * y) - 1| = 0 → y ≤ x) ∧
    x = 61 * Real.pi / 6 :=
by sorry

end NUMINAMATH_CALUDE_greatest_solution_sin_cos_equation_l262_26239


namespace NUMINAMATH_CALUDE_quadratic_properties_l262_26287

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) :
  f a b c (-2) = -11 →
  f a b c (-1) = 9 →
  f a b c 0 = 21 →
  f a b c 3 = 9 →
  (∃ (x_max : ℝ), x_max = 0 ∧ ∀ x, f a b c x ≤ f a b c x_max) ∧
  (∃ (x_sym : ℝ), x_sym = 1 ∧ ∀ x, f a b c (x_sym - x) = f a b c (x_sym + x)) ∧
  (∃ (x : ℝ), 3 < x ∧ x < 4 ∧ f a b c x = 0) ∧
  (∀ x, f a b c x > 21 ↔ 0 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l262_26287


namespace NUMINAMATH_CALUDE_border_material_length_l262_26297

/-- Given a circular table top with an area of 616 square inches,
    calculate the length of border material needed to cover the circumference
    plus an additional 3 inches, using π ≈ 22/7. -/
theorem border_material_length : 
  let table_area : ℝ := 616
  let π_approx : ℝ := 22 / 7
  let radius : ℝ := Real.sqrt (table_area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let border_length : ℝ := circumference + 3
  border_length = 91 := by
  sorry

end NUMINAMATH_CALUDE_border_material_length_l262_26297


namespace NUMINAMATH_CALUDE_fraction_relation_l262_26280

theorem fraction_relation (x y z w : ℝ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 4)
  (h3 : z / w = 7) :
  w / x = 4 / 35 := by
sorry

end NUMINAMATH_CALUDE_fraction_relation_l262_26280


namespace NUMINAMATH_CALUDE_rachel_money_theorem_l262_26234

def rachel_money_left (initial_amount : ℚ) (lunch_fraction : ℚ) (dvd_fraction : ℚ) : ℚ :=
  initial_amount - (lunch_fraction * initial_amount) - (dvd_fraction * initial_amount)

theorem rachel_money_theorem :
  rachel_money_left 200 (1/4) (1/2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_rachel_money_theorem_l262_26234


namespace NUMINAMATH_CALUDE_rectangle_width_l262_26289

/-- A rectangle with area 50 square meters and perimeter 30 meters has a width of 5 meters. -/
theorem rectangle_width (length width : ℝ) 
  (area_eq : length * width = 50)
  (perimeter_eq : 2 * length + 2 * width = 30) :
  width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l262_26289


namespace NUMINAMATH_CALUDE_centroid_distance_sum_l262_26274

/-- Given a triangle DEF with centroid G, prove that if the sum of squared distances
    from G to the vertices is 90, then the sum of squared side lengths is 270. -/
theorem centroid_distance_sum (D E F G : ℝ × ℝ) : 
  (G = ((D.1 + E.1 + F.1) / 3, (D.2 + E.2 + F.2) / 3)) →  -- G is the centroid
  ((G.1 - D.1)^2 + (G.2 - D.2)^2 + 
   (G.1 - E.1)^2 + (G.2 - E.2)^2 + 
   (G.1 - F.1)^2 + (G.2 - F.2)^2 = 90) →  -- Sum of squared distances from G to vertices is 90
  ((D.1 - E.1)^2 + (D.2 - E.2)^2 + 
   (D.1 - F.1)^2 + (D.2 - F.2)^2 + 
   (E.1 - F.1)^2 + (E.2 - F.2)^2 = 270)  -- Sum of squared side lengths is 270
:= by sorry

end NUMINAMATH_CALUDE_centroid_distance_sum_l262_26274


namespace NUMINAMATH_CALUDE_correct_negation_l262_26215

theorem correct_negation :
  (¬ ∃ x : ℝ, x^2 < 0) ↔ (∀ x : ℝ, x^2 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_correct_negation_l262_26215


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l262_26247

def selling_price : ℝ := 900
def profit : ℝ := 300

theorem profit_percentage_calculation : 
  (profit / (selling_price - profit)) * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l262_26247
