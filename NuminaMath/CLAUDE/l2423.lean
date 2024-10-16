import Mathlib

namespace NUMINAMATH_CALUDE_parametric_to_cartesian_l2423_242387

theorem parametric_to_cartesian :
  ∀ (x y : ℝ), (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 * Real.pi ∧ x = 2 * Real.cos t ∧ y = 3 * Real.sin t) →
  x^2 / 4 + y^2 / 9 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_parametric_to_cartesian_l2423_242387


namespace NUMINAMATH_CALUDE_smallest_cube_for_cone_l2423_242310

/-- Represents a cone with given height and base diameter -/
structure Cone where
  height : ℝ
  baseDiameter : ℝ

/-- Represents a cube with given side length -/
structure Cube where
  sideLength : ℝ

/-- The volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- A cube contains a cone if its side length is at least as large as both
    the cone's height and base diameter -/
def cubeContainsCone (cube : Cube) (cone : Cone) : Prop :=
  cube.sideLength ≥ cone.height ∧ cube.sideLength ≥ cone.baseDiameter

theorem smallest_cube_for_cone (c : Cone)
    (h1 : c.height = 15)
    (h2 : c.baseDiameter = 8) :
    ∃ (cube : Cube),
      cubeContainsCone cube c ∧
      cubeVolume cube = 3375 ∧
      ∀ (other : Cube), cubeContainsCone other c → cubeVolume other ≥ cubeVolume cube :=
  sorry

end NUMINAMATH_CALUDE_smallest_cube_for_cone_l2423_242310


namespace NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l2423_242309

/-- A regular octagon with vertices A, B, C, D, E, F, G, H -/
structure RegularOctagon where
  vertices : Fin 8 → Point

/-- The angle formed by extending sides AB and GH of a regular octagon to meet at point Q -/
def angle_Q (octagon : RegularOctagon) : ℝ :=
  sorry

theorem regular_octagon_extended_sides_angle (octagon : RegularOctagon) :
  angle_Q octagon = 90 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l2423_242309


namespace NUMINAMATH_CALUDE_car_can_climb_slope_l2423_242372

theorem car_can_climb_slope (car_max_angle : Real) (slope_gradient : Real) : 
  car_max_angle = 60 * Real.pi / 180 →
  slope_gradient = 1.5 →
  Real.tan car_max_angle > slope_gradient := by
  sorry

end NUMINAMATH_CALUDE_car_can_climb_slope_l2423_242372


namespace NUMINAMATH_CALUDE_candy_distribution_l2423_242384

def is_valid_student_count (n : ℕ) : Prop :=
  n > 1 ∧ 129 % n = 0

theorem candy_distribution (total_candies : ℕ) (h_total : total_candies = 130) :
  ∀ n : ℕ, is_valid_student_count n ↔ (n = 3 ∨ n = 43 ∨ n = 129) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l2423_242384


namespace NUMINAMATH_CALUDE_locus_of_point_c_l2423_242316

/-- Given a right triangle ABC with ∠C = 90°, where A is on the positive x-axis and B is on the positive y-axis,
    prove that the locus of point C is described by the equation y = (b/a)x, where ab/c ≤ x ≤ a. -/
theorem locus_of_point_c (a b c : ℝ) (A B C : ℝ × ℝ) :
  a > 0 → b > 0 →
  c^2 = a^2 + b^2 →
  A.1 > 0 → A.2 = 0 →
  B.1 = 0 → B.2 > 0 →
  C.1^2 + C.2^2 = a^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = b^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = a^2 →
  ∃ (x : ℝ), a*b/c ≤ x ∧ x ≤ a ∧ C = (x, b/a * x) :=
sorry


end NUMINAMATH_CALUDE_locus_of_point_c_l2423_242316


namespace NUMINAMATH_CALUDE_rest_area_distance_l2423_242359

theorem rest_area_distance (d : ℝ) : 
  (¬ (d ≥ 8)) →  -- David's statement is false
  (¬ (d ≤ 7)) →  -- Ellen's statement is false
  (¬ (d ≤ 6)) →  -- Frank's statement is false
  (7 < d ∧ d < 8) := by
sorry

end NUMINAMATH_CALUDE_rest_area_distance_l2423_242359


namespace NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5_l2423_242338

theorem largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5 : 
  ∃ (n : ℕ), 
    n > 0 ∧ 
    n % 18 = 0 ∧ 
    (26 : ℝ) < Real.sqrt n ∧ 
    Real.sqrt n ≤ 26.5 ∧
    ∀ (m : ℕ), m > 0 ∧ m % 18 = 0 ∧ (26 : ℝ) < Real.sqrt m ∧ Real.sqrt m ≤ 26.5 → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_divisible_by_18_with_sqrt_between_26_and_26_5_l2423_242338


namespace NUMINAMATH_CALUDE_binomial_6_2_l2423_242331

theorem binomial_6_2 : Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_6_2_l2423_242331


namespace NUMINAMATH_CALUDE_bridget_apples_theorem_l2423_242395

/-- The number of apples Bridget originally bought -/
def original_apples : ℕ := 14

/-- The number of apples Bridget gives to Ann -/
def apples_to_ann : ℕ := original_apples / 2

/-- The number of apples Bridget gives to Cassie -/
def apples_to_cassie : ℕ := 5

/-- The number of apples Bridget keeps for herself -/
def apples_for_bridget : ℕ := 2

theorem bridget_apples_theorem :
  original_apples = apples_to_ann * 2 ∧
  original_apples = apples_to_ann + apples_to_cassie + apples_for_bridget :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_theorem_l2423_242395


namespace NUMINAMATH_CALUDE_sum_of_a_and_c_l2423_242334

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 48) 
  (h2 : b + d = 6) : 
  a + c = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_c_l2423_242334


namespace NUMINAMATH_CALUDE_jayda_spending_l2423_242366

theorem jayda_spending (aitana_spending jayda_spending : ℚ) : 
  aitana_spending = jayda_spending + (2/5 : ℚ) * jayda_spending →
  aitana_spending + jayda_spending = 960 →
  jayda_spending = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_jayda_spending_l2423_242366


namespace NUMINAMATH_CALUDE_election_votes_l2423_242304

theorem election_votes (winning_percentage : ℝ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 0.6 →
  majority = 1380 →
  total_votes * winning_percentage - total_votes * (1 - winning_percentage) = majority →
  total_votes = 6900 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l2423_242304


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2423_242327

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, g (x^2 + y^2 + y * g z) = x * g x + z^2 * g y

/-- The theorem stating that g must be either the zero function or the identity function -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
    (∀ x, g x = 0) ∨ (∀ x, g x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2423_242327


namespace NUMINAMATH_CALUDE_max_intersections_four_circles_l2423_242385

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a plane, represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a line intersects a circle -/
def intersects (l : Line) (c : Circle) : Prop :=
  sorry

/-- The number of intersection points between a line and a circle -/
def numIntersections (l : Line) (c : Circle) : ℕ :=
  sorry

/-- Predicate to check if four circles are coplanar -/
def coplanar (c1 c2 c3 c4 : Circle) : Prop :=
  sorry

/-- Theorem: The maximum number of intersection points between a line and four coplanar circles is 8 -/
theorem max_intersections_four_circles (c1 c2 c3 c4 : Circle) (l : Line) :
  coplanar c1 c2 c3 c4 →
  intersects l c1 →
  intersects l c2 →
  intersects l c3 →
  intersects l c4 →
  numIntersections l c1 + numIntersections l c2 + numIntersections l c3 + numIntersections l c4 ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_four_circles_l2423_242385


namespace NUMINAMATH_CALUDE_water_addition_changes_ratio_l2423_242380

/-- Given a mixture of alcohol and water, prove that adding 2 liters of water
    changes the ratio from 4:3 to 4:5 when the initial amount of alcohol is 4 liters. -/
theorem water_addition_changes_ratio :
  let initial_alcohol : ℝ := 4
  let initial_water : ℝ := 3
  let water_added : ℝ := 2
  let final_water : ℝ := initial_water + water_added
  let initial_ratio : ℝ := initial_alcohol / initial_water
  let final_ratio : ℝ := initial_alcohol / final_water
  initial_ratio = 4/3 ∧ final_ratio = 4/5 := by
  sorry

#check water_addition_changes_ratio

end NUMINAMATH_CALUDE_water_addition_changes_ratio_l2423_242380


namespace NUMINAMATH_CALUDE_sons_age_l2423_242302

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2423_242302


namespace NUMINAMATH_CALUDE_no_solution_implies_n_greater_than_one_l2423_242325

theorem no_solution_implies_n_greater_than_one (n : ℝ) :
  (∀ x : ℝ, ¬(x ≤ 1 ∧ x ≥ n)) → n > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_n_greater_than_one_l2423_242325


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l2423_242314

/-- Given a triangle with side lengths 8, 2x+5, and 3x+2, and a perimeter of 40,
    the longest side of the triangle is 17. -/
theorem longest_side_of_triangle (x : ℝ) : 
  8 + (2*x + 5) + (3*x + 2) = 40 → 
  max 8 (max (2*x + 5) (3*x + 2)) = 17 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l2423_242314


namespace NUMINAMATH_CALUDE_brent_baby_ruths_l2423_242344

/-- The number of Baby Ruths Brent received -/
def baby_ruths : ℕ := sorry

/-- The number of Kit-Kat bars Brent received -/
def kit_kat : ℕ := 5

/-- The number of Hershey kisses Brent received -/
def hershey_kisses : ℕ := 3 * kit_kat

/-- The number of Nerds boxes Brent received -/
def nerds : ℕ := 8

/-- The number of lollipops Brent received -/
def lollipops : ℕ := 11

/-- The number of Reese Peanut butter cups Brent received -/
def reese_cups : ℕ := baby_ruths / 2

/-- The number of lollipops Brent gave to his sister -/
def lollipops_given : ℕ := 5

/-- The total number of candies Brent had left after giving lollipops to his sister -/
def total_left : ℕ := 49

theorem brent_baby_ruths :
  kit_kat + hershey_kisses + nerds + (lollipops - lollipops_given) + baby_ruths + reese_cups = total_left ∧
  baby_ruths = 10 := by sorry

end NUMINAMATH_CALUDE_brent_baby_ruths_l2423_242344


namespace NUMINAMATH_CALUDE_circle_area_theorem_l2423_242397

theorem circle_area_theorem (r : ℝ) (h : r > 0) :
  (2 * (1 / (2 * Real.pi * r)) = r / 2) → (Real.pi * r^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l2423_242397


namespace NUMINAMATH_CALUDE_chord_length_specific_case_l2423_242308

/-- The length of the chord formed by the intersection of a line and a circle -/
def chord_length (line_point : ℝ × ℝ) (line_angle : ℝ) (circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  sorry

theorem chord_length_specific_case :
  let line_point : ℝ × ℝ := (1, 0)
  let line_angle : ℝ := 30 * π / 180  -- 30 degrees in radians
  let circle_center : ℝ × ℝ := (2, 0)
  let circle_radius : ℝ := 1
  chord_length line_point line_angle circle_center circle_radius = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_specific_case_l2423_242308


namespace NUMINAMATH_CALUDE_max_value_of_f_l2423_242352

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2423_242352


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l2423_242389

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  hydrogen : ℕ
  bromine : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (h_weight o_weight br_weight : ℕ) : ℕ :=
  c.hydrogen * h_weight + c.bromine * br_weight + c.oxygen * o_weight

theorem compound_oxygen_count :
  ∀ (c : Compound),
    c.hydrogen = 1 →
    c.bromine = 1 →
    molecularWeight c 1 16 80 = 129 →
    c.oxygen = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_count_l2423_242389


namespace NUMINAMATH_CALUDE_opposite_numbers_pairs_l2423_242306

theorem opposite_numbers_pairs (a b : ℝ) (h1 : a + b = 0) (h2 : a ≠ 0) :
  ((-a) + b ≠ 0) ∧
  ((-a) + (-b) = 0) ∧
  (|a| + |b| ≠ 0) ∧
  (a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_pairs_l2423_242306


namespace NUMINAMATH_CALUDE_remainder_problem_l2423_242305

theorem remainder_problem (N : ℤ) (h : N % 350 = 37) : (2 * N) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2423_242305


namespace NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l2423_242341

theorem units_digit_of_seven_to_sixth (n : ℕ) : n = 7^6 → n % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_to_sixth_l2423_242341


namespace NUMINAMATH_CALUDE_symmetric_function_a_value_inequality_condition_a_range_l2423_242333

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + 2*a

-- Theorem 1
theorem symmetric_function_a_value (a : ℝ) :
  (∀ x : ℝ, f a x = f a (3 - x)) → a = -3 := by sorry

-- Theorem 2
theorem inequality_condition_a_range (a : ℝ) :
  (∃ x : ℝ, f a x ≤ -|2*x - 1| + a) → a ≤ -1/2 := by sorry

end NUMINAMATH_CALUDE_symmetric_function_a_value_inequality_condition_a_range_l2423_242333


namespace NUMINAMATH_CALUDE_A_intersect_B_empty_l2423_242367

def A : Set ℤ := {x | ∃ n : ℕ+, x = 2*n - 1}

def B : Set ℤ := {y | ∃ x ∈ A, y = 3*x - 1}

theorem A_intersect_B_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_empty_l2423_242367


namespace NUMINAMATH_CALUDE_circle_radius_l2423_242320

/-- The radius of a circle satisfying the given condition -/
theorem circle_radius : ∃ (r : ℝ), r > 0 ∧ 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l2423_242320


namespace NUMINAMATH_CALUDE_number_calculation_l2423_242381

theorem number_calculation (x : ℝ) : 
  (1.5 * x) / 7 = 271.07142857142856 → x = 1265 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2423_242381


namespace NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_gt_sqrt_five_l2423_242364

theorem sqrt_two_plus_sqrt_three_gt_sqrt_five :
  Real.sqrt 2 + Real.sqrt 3 > Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_three_gt_sqrt_five_l2423_242364


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l2423_242371

theorem cubic_roots_sum_of_squares (a b c t : ℝ) : 
  (∀ x, x^3 - 8*x^2 + 14*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 16*t^2 - 12*t = -8*Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l2423_242371


namespace NUMINAMATH_CALUDE_inequality_always_true_l2423_242322

theorem inequality_always_true (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l2423_242322


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l2423_242303

theorem pentagon_largest_angle (a b c d e : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 → -- All angles are positive
  b / a = 3 / 2 ∧ c / a = 2 ∧ d / a = 5 / 2 ∧ e / a = 3 → -- Angles are in ratio 2:3:4:5:6
  a + b + c + d + e = 540 → -- Sum of angles in a pentagon
  e = 162 := by
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l2423_242303


namespace NUMINAMATH_CALUDE_triangular_prism_no_body_diagonal_l2423_242321

-- Define what a prism is
structure Prism where
  base : Type
  has_base_diagonal : Bool

-- Define the property of having a body diagonal
def has_body_diagonal (p : Prism) : Bool := p.has_base_diagonal

-- Define specific types of prisms
def triangular_prism : Prism := { base := Unit, has_base_diagonal := false }

-- Theorem statement
theorem triangular_prism_no_body_diagonal : 
  ¬(has_body_diagonal triangular_prism) := by sorry

end NUMINAMATH_CALUDE_triangular_prism_no_body_diagonal_l2423_242321


namespace NUMINAMATH_CALUDE_inverse_proportion_ratio_l2423_242301

/-- Given that x is inversely proportional to y, prove that y₁/y₂ = 5/3 when x₁/x₂ = 3/5 -/
theorem inverse_proportion_ratio (x₁ x₂ y₁ y₂ : ℝ) (hx : x₁ ≠ 0 ∧ x₂ ≠ 0) (hy : y₁ ≠ 0 ∧ y₂ ≠ 0)
  (h_prop : ∃ (k : ℝ), ∀ (x y : ℝ), x * y = k)
  (h_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ratio_l2423_242301


namespace NUMINAMATH_CALUDE_notepad_lasts_four_days_l2423_242390

/-- Calculates the number of days a notepad lasts given the specified conditions -/
def notepadDuration (piecesPerNotepad : ℕ) (folds : ℕ) (notesPerDay : ℕ) : ℕ :=
  let sectionsPerPiece := 2^folds
  let totalNotes := piecesPerNotepad * sectionsPerPiece
  totalNotes / notesPerDay

/-- Theorem stating that under the given conditions, a notepad lasts 4 days -/
theorem notepad_lasts_four_days :
  notepadDuration 5 3 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_notepad_lasts_four_days_l2423_242390


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2423_242312

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) → 
  (a^3 - ((a - 2) * a * (a + 2)) = 16) → 
  (a^3 = 64) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2423_242312


namespace NUMINAMATH_CALUDE_product_of_numbers_with_hcf_l2423_242382

theorem product_of_numbers_with_hcf (A B : ℕ) : 
  A > 0 ∧ B > 0 →
  A > B →
  A = 33 →
  Nat.gcd A B = 11 →
  A * B = 363 :=
by sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_hcf_l2423_242382


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2423_242365

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ ¬(a > 0)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2423_242365


namespace NUMINAMATH_CALUDE_handmade_sweater_cost_l2423_242300

/-- The cost of a handmade sweater given Maria's shopping scenario -/
theorem handmade_sweater_cost 
  (num_sweaters num_scarves : ℕ)
  (scarf_cost : ℚ)
  (initial_savings remaining_savings : ℚ)
  (h1 : num_sweaters = 6)
  (h2 : num_scarves = 6)
  (h3 : scarf_cost = 20)
  (h4 : initial_savings = 500)
  (h5 : remaining_savings = 200) :
  (initial_savings - remaining_savings - num_scarves * scarf_cost) / num_sweaters = 30 := by
  sorry

end NUMINAMATH_CALUDE_handmade_sweater_cost_l2423_242300


namespace NUMINAMATH_CALUDE_area_of_DBCE_l2423_242379

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the trapezoid DBCE in the diagram -/
structure Trapezoid where
  area : ℝ

/-- The isosceles triangle ABC -/
def ABC : Triangle := { area := 96 }

/-- One of the smallest triangles in the diagram -/
def smallTriangle : Triangle := { area := 2 }

/-- The number of smallest triangles in the diagram -/
def numSmallTriangles : ℕ := 12

/-- The triangle ADF formed by 8 smallest triangles -/
def ADF : Triangle := { area := 8 * smallTriangle.area }

/-- The trapezoid DBCE -/
def DBCE : Trapezoid := { area := ABC.area - ADF.area }

theorem area_of_DBCE : DBCE.area = 80 := by
  sorry

end NUMINAMATH_CALUDE_area_of_DBCE_l2423_242379


namespace NUMINAMATH_CALUDE_divisibility_problem_l2423_242388

theorem divisibility_problem (N : ℕ) : 
  N = 7 * 13 + 1 → (N / 8 + N % 8 = 15) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2423_242388


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2423_242355

theorem inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 1| < m) → m > 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2423_242355


namespace NUMINAMATH_CALUDE_average_age_decrease_l2423_242335

theorem average_age_decrease (initial_avg : ℝ) : 
  let initial_total := 10 * initial_avg
  let new_total := initial_total - 45 + 15
  let new_avg := new_total / 10
  initial_avg - new_avg = 3 := by sorry

end NUMINAMATH_CALUDE_average_age_decrease_l2423_242335


namespace NUMINAMATH_CALUDE_dans_purchases_cost_l2423_242360

/-- The total cost of Dan's purchases, given the cost of a snake toy, a cage, and finding a dollar bill. -/
theorem dans_purchases_cost (snake_toy_cost cage_cost found_money : ℚ) : 
  snake_toy_cost = 11.76 →
  cage_cost = 14.54 →
  found_money = 1 →
  snake_toy_cost + cage_cost - found_money = 25.30 := by
  sorry

end NUMINAMATH_CALUDE_dans_purchases_cost_l2423_242360


namespace NUMINAMATH_CALUDE_symmetric_angle_set_l2423_242328

theorem symmetric_angle_set (α β : Real) (k : Int) : 
  α = π / 6 → 
  (∃ (f : Real → Real), f x = x → f α = β) →  -- Symmetry condition
  (β = 2 * k * π + π / 3) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_angle_set_l2423_242328


namespace NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_l2423_242399

theorem largest_binomial_coefficient_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_coefficient_sum_l2423_242399


namespace NUMINAMATH_CALUDE_smallest_y_makes_perfect_cube_no_smaller_y_exists_smallest_y_is_perfect_cube_l2423_242332

def x : ℕ := 7 * 24 * 54

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def smallest_y : ℕ := 1764

theorem smallest_y_makes_perfect_cube :
  (∀ y : ℕ, y < smallest_y → ¬ is_perfect_cube (x * y)) ∧
  is_perfect_cube (x * smallest_y) := by sorry

theorem no_smaller_y_exists (y : ℕ) (h : y < smallest_y) :
  ¬ is_perfect_cube (x * y) := by sorry

theorem smallest_y_is_perfect_cube :
  is_perfect_cube (x * smallest_y) := by sorry

end NUMINAMATH_CALUDE_smallest_y_makes_perfect_cube_no_smaller_y_exists_smallest_y_is_perfect_cube_l2423_242332


namespace NUMINAMATH_CALUDE_triangle_area_l2423_242336

theorem triangle_area (A B C : Real) (angleC : A + B + C = Real.pi) 
  (sideAC sideAB : Real) (h_angleC : C = Real.pi / 6) 
  (h_sideAC : sideAC = 3 * Real.sqrt 3) (h_sideAB : sideAB = 3) :
  let area := (1 / 2) * sideAC * sideAB * Real.sin A
  area = (9 * Real.sqrt 3) / 2 ∨ area = (9 * Real.sqrt 3) / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l2423_242336


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2423_242330

theorem quadratic_factorization (y a b : ℤ) : 
  2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b) → a - b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2423_242330


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2423_242393

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 75500000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 7.55
  exponent := 7
  property := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific : 
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2423_242393


namespace NUMINAMATH_CALUDE_octal_to_binary_l2423_242317

-- Define the octal number
def octal_177 : ℕ := 177

-- Define the binary number
def binary_1111111 : ℕ := 127

-- Theorem statement
theorem octal_to_binary :
  (octal_177 : ℕ) = binary_1111111 := by sorry

end NUMINAMATH_CALUDE_octal_to_binary_l2423_242317


namespace NUMINAMATH_CALUDE_third_term_binomial_expansion_l2423_242354

theorem third_term_binomial_expansion (x : ℝ) : 
  let n : ℕ := 4
  let a : ℝ := x
  let b : ℝ := 2
  let r : ℕ := 2
  let binomial_coeff := Nat.choose n r
  let power_term := a^(n - r) * b^r
  binomial_coeff * power_term = 24 * x^2 := by
sorry


end NUMINAMATH_CALUDE_third_term_binomial_expansion_l2423_242354


namespace NUMINAMATH_CALUDE_function_property_l2423_242353

theorem function_property (A : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, A (x + y) = A x + A y) 
  (h2 : ∀ x y : ℝ, A (x * y) = A x * A y) : 
  (∀ x : ℝ, A x = x) ∨ (∀ x : ℝ, A x = 0) :=
by sorry

end NUMINAMATH_CALUDE_function_property_l2423_242353


namespace NUMINAMATH_CALUDE_parallel_vectors_y_value_l2423_242324

/-- Given two parallel vectors a and b, prove that y = 7 -/
theorem parallel_vectors_y_value (a b : ℝ × ℝ) (y : ℝ) 
  (h1 : a = (2, 3)) 
  (h2 : b = (4, -1 + y)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a = k • b) : 
  y = 7 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_y_value_l2423_242324


namespace NUMINAMATH_CALUDE_shift_f_equals_g_l2423_242315

def f (x : ℝ) : ℝ := -x^2

def g (x : ℝ) : ℝ := -x^2 + 2

def vertical_shift (h : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x => h x + k

theorem shift_f_equals_g : vertical_shift f 2 = g := by sorry

end NUMINAMATH_CALUDE_shift_f_equals_g_l2423_242315


namespace NUMINAMATH_CALUDE_max_value_of_function_l2423_242313

theorem max_value_of_function (x : ℝ) : 
  1 / (x^2 + 2) ≤ 1 / 2 ∧ ∃ y : ℝ, 1 / (y^2 + 2) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l2423_242313


namespace NUMINAMATH_CALUDE_track_meet_seating_l2423_242307

theorem track_meet_seating (children adults seniors pets seats : ℕ) : 
  children = 52 → 
  adults = 29 → 
  seniors = 15 → 
  pets = 3 → 
  seats = 95 → 
  children + adults + seniors + pets - seats = 4 := by
  sorry

end NUMINAMATH_CALUDE_track_meet_seating_l2423_242307


namespace NUMINAMATH_CALUDE_djibos_sister_age_l2423_242348

/-- Given that Djibo is 17 years old and 5 years ago the sum of his and his sister's ages was 35,
    prove that his sister is 28 years old today. -/
theorem djibos_sister_age :
  ∀ (djibo_age sister_age : ℕ),
    djibo_age = 17 →
    djibo_age + sister_age = 35 + 5 →
    sister_age = 28 :=
by sorry

end NUMINAMATH_CALUDE_djibos_sister_age_l2423_242348


namespace NUMINAMATH_CALUDE_hannah_practice_hours_l2423_242350

/-- Hannah's weekend practice hours -/
def weekend_hours : ℕ := sorry

theorem hannah_practice_hours : 
  (weekend_hours + (weekend_hours + 17) = 33) → 
  weekend_hours = 8 := by sorry

end NUMINAMATH_CALUDE_hannah_practice_hours_l2423_242350


namespace NUMINAMATH_CALUDE_limit_fraction_sequence_l2423_242376

theorem limit_fraction_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((n : ℝ) + 20) / (3 * n + 13) - 1/3| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_fraction_sequence_l2423_242376


namespace NUMINAMATH_CALUDE_sets_inclusion_l2423_242326

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem sets_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_sets_inclusion_l2423_242326


namespace NUMINAMATH_CALUDE_pentagon_square_angle_sum_l2423_242374

theorem pentagon_square_angle_sum : 
  ∀ (pentagon_angle square_angle : ℝ),
  (pentagon_angle = 180 * (5 - 2) / 5) →
  (square_angle = 180 * (4 - 2) / 4) →
  pentagon_angle + square_angle = 198 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_square_angle_sum_l2423_242374


namespace NUMINAMATH_CALUDE_expression_simplification_l2423_242392

theorem expression_simplification (x y : ℝ) (h : |x - 2| + (y + 1)^2 = 0) :
  3*x - 2*(x^2 - 1/2*y^2) + (x - 1/2*y^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2423_242392


namespace NUMINAMATH_CALUDE_race_probability_l2423_242342

theorem race_probability (p_x p_y p_z : ℝ) : 
  p_x = 1/7 →
  p_y = 1/3 →
  p_x + p_y + p_z = 0.6761904761904762 →
  p_z = 0.2 := by
sorry

end NUMINAMATH_CALUDE_race_probability_l2423_242342


namespace NUMINAMATH_CALUDE_ribbon_parts_l2423_242343

theorem ribbon_parts (total_length : ℝ) (used_parts : ℕ) (unused_length : ℝ) :
  total_length = 30 ∧ used_parts = 4 ∧ unused_length = 10 →
  ∃ (n : ℕ), n > 0 ∧ n * (total_length - unused_length) / used_parts = total_length / n :=
by sorry

end NUMINAMATH_CALUDE_ribbon_parts_l2423_242343


namespace NUMINAMATH_CALUDE_julia_played_with_34_kids_l2423_242329

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 17

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := 15

/-- The number of kids Julia played with on Wednesday -/
def wednesday_kids : ℕ := 2

/-- The total number of kids Julia played with -/
def total_kids : ℕ := monday_kids + tuesday_kids + wednesday_kids

theorem julia_played_with_34_kids : total_kids = 34 := by
  sorry

end NUMINAMATH_CALUDE_julia_played_with_34_kids_l2423_242329


namespace NUMINAMATH_CALUDE_distance_after_movements_l2423_242347

/-- The distance between two points given a path with specific movements -/
theorem distance_after_movements (south west north east : ℝ) :
  south = 50 ∧ west = 80 ∧ north = 30 ∧ east = 10 →
  Real.sqrt ((south - north)^2 + (west - east)^2) = 50 * Real.sqrt 106 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_after_movements_l2423_242347


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2423_242358

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n + 1) - a n = d) 
  (h2 : a 1 = f (d - 1)) 
  (h3 : a 3 = f (d + 1)) :
  ∀ n, a n = 2 * n + 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2423_242358


namespace NUMINAMATH_CALUDE_evaluate_expression_l2423_242386

theorem evaluate_expression : (36 - 6 * 3) / (6 / 3 * 2) = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2423_242386


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2423_242311

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2423_242311


namespace NUMINAMATH_CALUDE_probability_second_yellow_ball_l2423_242377

def initial_white_balls : ℕ := 5
def initial_yellow_balls : ℕ := 3

def remaining_white_balls : ℕ := initial_white_balls
def remaining_yellow_balls : ℕ := initial_yellow_balls - 1

def total_remaining_balls : ℕ := remaining_white_balls + remaining_yellow_balls

theorem probability_second_yellow_ball :
  (remaining_yellow_balls : ℚ) / total_remaining_balls = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_yellow_ball_l2423_242377


namespace NUMINAMATH_CALUDE_power_sum_equality_l2423_242345

theorem power_sum_equality : 3^3 + 4^3 + 5^3 = 6^3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2423_242345


namespace NUMINAMATH_CALUDE_balloon_arrangements_count_l2423_242323

/-- The number of distinct arrangements of the letters in "BALLOON" -/
def balloon_arrangements : ℕ := 1260

/-- The total number of letters in "BALLOON" -/
def total_letters : ℕ := 7

/-- The number of times 'L' appears in "BALLOON" -/
def l_count : ℕ := 2

/-- The number of times 'O' appears in "BALLOON" -/
def o_count : ℕ := 2

/-- Theorem stating that the number of distinct arrangements of the letters in "BALLOON" is 1260 -/
theorem balloon_arrangements_count :
  balloon_arrangements = (Nat.factorial total_letters) / (Nat.factorial l_count * Nat.factorial o_count) :=
sorry

end NUMINAMATH_CALUDE_balloon_arrangements_count_l2423_242323


namespace NUMINAMATH_CALUDE_product_change_l2423_242383

theorem product_change (a b : ℕ) (h : (a + 3) * (b - 3) - a * b = 600) : 
  a * b - (a - 3) * (b + 3) = 618 := by
sorry

end NUMINAMATH_CALUDE_product_change_l2423_242383


namespace NUMINAMATH_CALUDE_petrol_expense_l2423_242337

def monthly_expenses (rent milk groceries education misc petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + misc + petrol

def savings_percentage : ℚ := 1 / 10

theorem petrol_expense (rent milk groceries education misc savings : ℕ) 
  (h1 : rent = 5000)
  (h2 : milk = 1500)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : misc = 3940)
  (h6 : savings = 2160)
  (h7 : ∃ (salary petrol : ℕ), savings_percentage * salary = savings ∧ 
        monthly_expenses rent milk groceries education misc petrol = salary - savings) :
  ∃ (petrol : ℕ), petrol = 2000 := by
sorry

end NUMINAMATH_CALUDE_petrol_expense_l2423_242337


namespace NUMINAMATH_CALUDE_rebus_puzzle_solution_l2423_242318

theorem rebus_puzzle_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C ∧
    100 * A + 10 * C + C = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 :=
by sorry

end NUMINAMATH_CALUDE_rebus_puzzle_solution_l2423_242318


namespace NUMINAMATH_CALUDE_remainder_problem_l2423_242357

theorem remainder_problem (n : ℕ) : 
  (n / 44 = 432 ∧ n % 44 = 0) → n % 38 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2423_242357


namespace NUMINAMATH_CALUDE_night_heads_count_l2423_242356

/-- Represents the number of animals of each type -/
structure AnimalCounts where
  chickens : ℕ
  rabbits : ℕ
  geese : ℕ

/-- Calculates the total number of legs during the day -/
def totalDayLegs (counts : AnimalCounts) : ℕ :=
  2 * counts.chickens + 4 * counts.rabbits + 2 * counts.geese

/-- Calculates the total number of heads -/
def totalHeads (counts : AnimalCounts) : ℕ :=
  counts.chickens + counts.rabbits + counts.geese

/-- Calculates the total number of legs at night -/
def totalNightLegs (counts : AnimalCounts) : ℕ :=
  2 * counts.chickens + 4 * counts.rabbits + counts.geese

/-- The main theorem to prove -/
theorem night_heads_count (counts : AnimalCounts) 
  (h1 : totalDayLegs counts = 56)
  (h2 : totalDayLegs counts - totalHeads counts = totalNightLegs counts - totalHeads counts) :
  totalHeads counts = 14 := by
  sorry


end NUMINAMATH_CALUDE_night_heads_count_l2423_242356


namespace NUMINAMATH_CALUDE_no_real_solution_l2423_242363

theorem no_real_solution :
  ¬∃ x : ℝ, Real.sqrt (x + 9) - Real.sqrt (x - 2) + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_l2423_242363


namespace NUMINAMATH_CALUDE_tenth_term_value_l2423_242398

/-- An arithmetic sequence with 30 terms, first term 3, and last term 88 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (88 - 3) / 29
  3 + (n - 1) * d

/-- The 10th term of the arithmetic sequence is 852/29 -/
theorem tenth_term_value : arithmetic_sequence 10 = 852 / 29 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_value_l2423_242398


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2423_242369

def f (x : ℝ) := x^3 - x^2 - x

def f_derivative (x : ℝ) := 3*x^2 - 2*x - 1

theorem monotonic_decreasing_interval :
  {x : ℝ | f_derivative x < 0} = {x : ℝ | -1/3 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l2423_242369


namespace NUMINAMATH_CALUDE_glen_village_count_l2423_242378

theorem glen_village_count (p h s c d : ℕ) : 
  p = 2 * h →  -- 2 people for each horse
  s = 5 * c →  -- 5 sheep for each cow
  d = 4 * p →  -- 4 ducks for each person
  p + h + s + c + d ≠ 47 :=
by sorry

end NUMINAMATH_CALUDE_glen_village_count_l2423_242378


namespace NUMINAMATH_CALUDE_student_failed_marks_l2423_242368

def total_marks : ℕ := 300
def passing_percentage : ℚ := 60 / 100
def student_marks : ℕ := 160

theorem student_failed_marks :
  (passing_percentage * total_marks : ℚ).ceil - student_marks = 20 := by
  sorry

end NUMINAMATH_CALUDE_student_failed_marks_l2423_242368


namespace NUMINAMATH_CALUDE_boat_travel_time_l2423_242339

theorem boat_travel_time (v : ℝ) :
  let upstream_speed := v - 4
  let downstream_speed := v + 4
  let distance := 120
  let upstream_time (t : ℝ) := t + 1
  let downstream_time := 1
  (upstream_speed * upstream_time downstream_time = distance) ∧
  (downstream_speed * downstream_time = distance) →
  downstream_time = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_travel_time_l2423_242339


namespace NUMINAMATH_CALUDE_inequality_solution_sets_l2423_242349

theorem inequality_solution_sets (a : ℝ) : 
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_sets_l2423_242349


namespace NUMINAMATH_CALUDE_sphere_properties_l2423_242391

/-- Given a sphere with volume 288π cubic inches, prove its surface area is 144π square inches and its diameter is 12 inches -/
theorem sphere_properties (r : ℝ) (h : (4/3) * Real.pi * r^3 = 288 * Real.pi) :
  (4 * Real.pi * r^2 = 144 * Real.pi) ∧ (2 * r = 12) := by
  sorry

end NUMINAMATH_CALUDE_sphere_properties_l2423_242391


namespace NUMINAMATH_CALUDE_heartsuit_three_five_l2423_242373

-- Define the ⊛ operation
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- State the theorem
theorem heartsuit_three_five : heartsuit 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_five_l2423_242373


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2423_242319

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {x | |x| < 2}

theorem intersection_of_M_and_N :
  M ∩ N = {x | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2423_242319


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l2423_242394

theorem rachel_milk_consumption 
  (bottle1 : ℚ) (bottle2 : ℚ) (rachel_fraction : ℚ) :
  bottle1 = 3/8 →
  bottle2 = 1/4 →
  rachel_fraction = 3/4 →
  rachel_fraction * (bottle1 + bottle2) = 15/32 := by
  sorry

end NUMINAMATH_CALUDE_rachel_milk_consumption_l2423_242394


namespace NUMINAMATH_CALUDE_triangle_area_problem_l2423_242351

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * x * (3*x) = 96) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l2423_242351


namespace NUMINAMATH_CALUDE_characterization_theorem_l2423_242370

/-- A function that checks if a number satisfies the given conditions -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    n ≥ 2 ∧
    n = a^2 + b^2 ∧
    a > 1 ∧
    a ∣ n ∧
    b ∣ n ∧
    ∀ d : ℕ, d > 1 → d ∣ n → d ≥ a

/-- The main theorem stating the characterization of numbers satisfying the condition -/
theorem characterization_theorem :
  ∀ n : ℕ, satisfies_condition n ↔ 
    (n = 4) ∨ 
    (∃ k j : ℕ, k ≥ 2 ∧ j ≥ 1 ∧ j ≤ k ∧ n = 2^k * (2^(k*(j-1)) + 1)) :=
by sorry

end NUMINAMATH_CALUDE_characterization_theorem_l2423_242370


namespace NUMINAMATH_CALUDE_movie_ticket_difference_l2423_242361

theorem movie_ticket_difference (romance_tickets horror_tickets : ℕ) : 
  romance_tickets = 25 → 
  horror_tickets = 93 → 
  horror_tickets - 3 * romance_tickets = 18 := by
sorry

end NUMINAMATH_CALUDE_movie_ticket_difference_l2423_242361


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l2423_242375

theorem right_triangle_leg_length 
  (a b c : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 13) 
  (h_leg : a = 5) : 
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l2423_242375


namespace NUMINAMATH_CALUDE_no_natural_solution_for_x2_plus_y2_eq_7z2_l2423_242340

theorem no_natural_solution_for_x2_plus_y2_eq_7z2 :
  ¬ ∃ (x y z : ℕ), x^2 + y^2 = 7 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_x2_plus_y2_eq_7z2_l2423_242340


namespace NUMINAMATH_CALUDE_power_division_l2423_242346

theorem power_division (m : ℝ) : m^4 / m^2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l2423_242346


namespace NUMINAMATH_CALUDE_trig_inequality_l2423_242362

theorem trig_inequality : ∀ a b c : ℝ,
  a = Real.sin (21 * π / 180) →
  b = Real.cos (72 * π / 180) →
  c = Real.tan (23 * π / 180) →
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l2423_242362


namespace NUMINAMATH_CALUDE_functional_eq_solution_l2423_242396

/-- A function satisfying the given functional equation -/
def FunctionalEq (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x - y) = 2009 * f x * f y

/-- The main theorem -/
theorem functional_eq_solution (f : ℝ → ℝ) 
  (h1 : FunctionalEq f) 
  (h2 : ∀ x : ℝ, f x ≠ 0) : 
  f (Real.sqrt 2009) = 1 / 2009 := by
  sorry

end NUMINAMATH_CALUDE_functional_eq_solution_l2423_242396
