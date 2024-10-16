import Mathlib

namespace NUMINAMATH_CALUDE_selection_probabilities_l462_46250

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- The number of people to be selected -/
def num_selected : ℕ := 3

/-- The total number of ways to select 3 people from 5 people -/
def total_combinations : ℕ := Nat.choose (num_boys + num_girls) num_selected

/-- The probability of selecting all boys -/
def prob_all_boys : ℚ := (Nat.choose num_boys num_selected : ℚ) / total_combinations

/-- The probability of selecting exactly one girl -/
def prob_one_girl : ℚ := (Nat.choose num_boys (num_selected - 1) * Nat.choose num_girls 1 : ℚ) / total_combinations

/-- The probability of selecting at least one girl -/
def prob_at_least_one_girl : ℚ := 1 - prob_all_boys

theorem selection_probabilities :
  prob_all_boys = 1/10 ∧
  prob_one_girl = 6/10 ∧
  prob_at_least_one_girl = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_selection_probabilities_l462_46250


namespace NUMINAMATH_CALUDE_f_f_3_equals_13_9_l462_46249

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2/x

theorem f_f_3_equals_13_9 : f (f 3) = 13/9 := by
  sorry

end NUMINAMATH_CALUDE_f_f_3_equals_13_9_l462_46249


namespace NUMINAMATH_CALUDE_area_is_60_l462_46247

/-- Two perpendicular lines intersecting at point A(6,8) with y-intercepts P and Q -/
structure PerpendicularLines where
  A : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  perpendicular : True  -- Represents that the lines are perpendicular
  intersect_at_A : True -- Represents that the lines intersect at A
  A_coords : A = (6, 8)
  P_is_y_intercept : P.1 = 0
  Q_is_y_intercept : Q.1 = 0
  sum_of_y_intercepts_zero : P.2 + Q.2 = 0

/-- The area of triangle APQ -/
def triangle_area (lines : PerpendicularLines) : ℝ := sorry

/-- Theorem stating that the area of triangle APQ is 60 -/
theorem area_is_60 (lines : PerpendicularLines) : triangle_area lines = 60 := by
  sorry

end NUMINAMATH_CALUDE_area_is_60_l462_46247


namespace NUMINAMATH_CALUDE_smallest_positive_number_l462_46255

theorem smallest_positive_number (a b c d e : ℝ) :
  a = 15 - 4 * Real.sqrt 14 ∧
  b = 4 * Real.sqrt 14 - 15 ∧
  c = 20 - 6 * Real.sqrt 15 ∧
  d = 60 - 12 * Real.sqrt 31 ∧
  e = 12 * Real.sqrt 31 - 60 →
  (0 < a ∧ a ≤ b ∧ a ≤ c ∧ a ≤ d ∧ a ≤ e) ∨
  (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0 ∧ d ≤ 0 ∧ e ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_number_l462_46255


namespace NUMINAMATH_CALUDE_distribute_5_3_l462_46243

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l462_46243


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l462_46237

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement "p or q is false" is sufficient for "not p is true"
def is_sufficient : Prop :=
  (¬(p ∨ q)) → (¬p)

-- Define the statement "p or q is false" is not necessary for "not p is true"
def is_not_necessary : Prop :=
  ∃ (p q : Prop), (¬p) ∧ ¬(¬(p ∨ q))

-- The main theorem stating that "p or q is false" is sufficient but not necessary for "not p is true"
theorem sufficient_but_not_necessary :
  (is_sufficient p q) ∧ is_not_necessary :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l462_46237


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l462_46277

/-- The functional equation that f must satisfy for all x and y -/
def functional_equation (f : ℝ → ℝ) (α : ℝ) : Prop :=
  ∀ x y, f (f (x + y) * f (x - y)) = x^2 + α * y * f y

/-- The theorem stating the only solutions to the functional equation -/
theorem functional_equation_solutions :
  ∀ α : ℝ, ∀ f : ℝ → ℝ,
    functional_equation f α →
    ((α = 1 ∧ ∀ x, f x = -x) ∨ (α = -1 ∧ ∀ x, f x = x)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l462_46277


namespace NUMINAMATH_CALUDE_students_surveyed_students_surveyed_proof_l462_46222

theorem students_surveyed : ℕ :=
  let total_students : ℕ := sorry
  let french_speakers : ℕ := sorry
  let french_english_speakers : ℕ := 10
  let french_only_speakers : ℕ := 40

  have h1 : french_speakers = french_english_speakers + french_only_speakers := by sorry
  have h2 : french_speakers = 50 := by sorry
  have h3 : french_speakers = total_students / 4 := by sorry

  200

theorem students_surveyed_proof : students_surveyed = 200 := by sorry

end NUMINAMATH_CALUDE_students_surveyed_students_surveyed_proof_l462_46222


namespace NUMINAMATH_CALUDE_ladybugs_with_spots_count_l462_46288

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := 67082

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := total_ladybugs - ladybugs_without_spots

theorem ladybugs_with_spots_count : ladybugs_with_spots = 12170 := by
  sorry

end NUMINAMATH_CALUDE_ladybugs_with_spots_count_l462_46288


namespace NUMINAMATH_CALUDE_custom_operations_result_l462_46205

def star (a b : ℤ) : ℤ := a + b - 1

def hash (a b : ℤ) : ℤ := a * b - 1

theorem custom_operations_result : (star (star 6 8) (hash 3 5)) = 26 := by
  sorry

end NUMINAMATH_CALUDE_custom_operations_result_l462_46205


namespace NUMINAMATH_CALUDE_scientific_notation_of_1340000000_l462_46231

theorem scientific_notation_of_1340000000 :
  ∃ (a : ℝ) (n : ℤ), 1340000000 = a * (10 : ℝ)^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.34 ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1340000000_l462_46231


namespace NUMINAMATH_CALUDE_distance_BA_is_54_l462_46244

/-- Represents a circular path with three points -/
structure CircularPath where
  -- Distance from A to B
  dAB : ℝ
  -- Distance from B to C
  dBC : ℝ
  -- Distance from C to A
  dCA : ℝ
  -- Ensure all distances are positive
  all_positive : 0 < dAB ∧ 0 < dBC ∧ 0 < dCA

/-- The distance from B to A in the opposite direction on the circular path -/
def distance_BA (path : CircularPath) : ℝ :=
  path.dBC + path.dCA

/-- Theorem stating the distance from B to A in the opposite direction -/
theorem distance_BA_is_54 (path : CircularPath) 
  (h1 : path.dAB = 30) 
  (h2 : path.dBC = 28) 
  (h3 : path.dCA = 26) : 
  distance_BA path = 54 := by
  sorry

end NUMINAMATH_CALUDE_distance_BA_is_54_l462_46244


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l462_46280

/-- The area of a circle circumscribing an equilateral triangle with side length 4 is 16π/3 -/
theorem circle_area_equilateral_triangle :
  let s : ℝ := 4  -- side length of the equilateral triangle
  let r : ℝ := s / Real.sqrt 3  -- radius of the circumscribed circle
  let A : ℝ := π * r^2  -- area of the circle
  A = 16 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l462_46280


namespace NUMINAMATH_CALUDE_punch_difference_l462_46266

def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def total_punch : ℝ := 21

def apple_juice : ℝ := total_punch - orange_punch - cherry_punch

theorem punch_difference : cherry_punch - apple_juice = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_punch_difference_l462_46266


namespace NUMINAMATH_CALUDE_hash_composition_l462_46219

-- Define the # operation
def hash (a b : ℝ) : ℝ := a * b - b + b^2

-- Theorem statement
theorem hash_composition (z : ℝ) : hash (hash 3 8) z = 79 * z + z^2 := by
  sorry

end NUMINAMATH_CALUDE_hash_composition_l462_46219


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_four_l462_46283

theorem three_digit_perfect_cube_divisible_by_four :
  ∃! n : ℕ, 100 ≤ 8 * n^3 ∧ 8 * n^3 ≤ 999 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_four_l462_46283


namespace NUMINAMATH_CALUDE_hemisphere_intersection_area_l462_46209

/-- Given two hemispheres A and B, where A has a surface area of 50π, and B has twice the surface area of A,
    if B shares 1/4 of its surface area with A, then the surface area of the remainder of hemisphere B
    after the intersection is 75π. -/
theorem hemisphere_intersection_area (A B : ℝ) : 
  A = 50 * Real.pi →
  B = 2 * A →
  let shared := (1/4) * B
  B - shared = 75 * Real.pi := by sorry

end NUMINAMATH_CALUDE_hemisphere_intersection_area_l462_46209


namespace NUMINAMATH_CALUDE_gum_distribution_l462_46271

theorem gum_distribution (john_gum cole_gum aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : aubrey_gum = 0)
  (num_people : ℕ)
  (h4 : num_people = 3) :
  (john_gum + cole_gum + aubrey_gum) / num_people = 33 := by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l462_46271


namespace NUMINAMATH_CALUDE_unique_positive_solution_l462_46276

theorem unique_positive_solution (n : ℕ) (hn : n > 1) :
  ∀ x : ℝ, x > 0 → (x^n - n*x + n - 1 = 0) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l462_46276


namespace NUMINAMATH_CALUDE_f_maximum_l462_46239

/-- The quadratic function f(x) = -3x^2 + 9x + 5 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 5

/-- The value of x that maximizes f(x) -/
def x_max : ℝ := 1.5

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_f_maximum_l462_46239


namespace NUMINAMATH_CALUDE_lawn_length_l462_46257

/-- Given a rectangular lawn with area 20 square feet and width 5 feet, prove its length is 4 feet. -/
theorem lawn_length (area : ℝ) (width : ℝ) (length : ℝ) : 
  area = 20 → width = 5 → area = length * width → length = 4 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_l462_46257


namespace NUMINAMATH_CALUDE_cricket_average_score_l462_46226

theorem cricket_average_score (matches1 matches2 : ℕ) (avg1 avg2 : ℚ) :
  matches1 = 10 →
  matches2 = 15 →
  avg1 = 60 →
  avg2 = 70 →
  (matches1 * avg1 + matches2 * avg2) / (matches1 + matches2) = 66 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l462_46226


namespace NUMINAMATH_CALUDE_area_ratio_extended_triangle_l462_46265

-- Define the triangle ABC and its extensions
structure ExtendedTriangle where
  -- Original equilateral triangle
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Extended sides
  BB' : ℝ
  CC' : ℝ
  AA' : ℝ
  -- Conditions
  equilateral : AB = BC ∧ BC = CA
  extension_BB' : BB' = 2 * AB
  extension_CC' : CC' = 3 * BC
  extension_AA' : AA' = 4 * CA

-- Define the theorem
theorem area_ratio_extended_triangle (t : ExtendedTriangle) :
  (t.AB + t.BB')^2 + (t.BC + t.CC')^2 + (t.CA + t.AA')^2 = 25 * (t.AB^2 + t.BC^2 + t.CA^2) :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_extended_triangle_l462_46265


namespace NUMINAMATH_CALUDE_hope_project_protractors_l462_46207

theorem hope_project_protractors :
  ∀ (x y z : ℕ),
  x > 31 →
  z > 33 →
  10 * x + 15 * y + 20 * z = 1710 →
  8 * x + 2 * y + 8 * z = 664 →
  6 * x + 7 * y + 10 * z = 870 :=
by
  sorry

end NUMINAMATH_CALUDE_hope_project_protractors_l462_46207


namespace NUMINAMATH_CALUDE_fencing_calculation_l462_46295

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area = 50 → uncovered_side = 20 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 25 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l462_46295


namespace NUMINAMATH_CALUDE_race_head_start_l462_46223

theorem race_head_start (v_a v_b L H : ℝ) : 
  v_a = (32/27) * v_b →
  (L / v_a = (L - H) / v_b) →
  H = (5/32) * L :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l462_46223


namespace NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_q_l462_46220

-- Define propositions p and q
def p (x : ℝ) : Prop := -1 < x ∧ x < 3
def q (x : ℝ) : Prop := x > 5

-- Theorem statement
theorem not_p_necessary_not_sufficient_for_q :
  (∀ x, q x → ¬(p x)) ∧ 
  ¬(∀ x, ¬(p x) → q x) :=
by sorry

end NUMINAMATH_CALUDE_not_p_necessary_not_sufficient_for_q_l462_46220


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l462_46201

/-- Given a line expressed as a dot product of vectors, prove it can be rewritten in slope-intercept form -/
theorem line_equation_equivalence (x y : ℝ) : 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y - (-5)) = 0 ↔ y = 2 * x - 11 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l462_46201


namespace NUMINAMATH_CALUDE_seven_n_implies_n_is_sum_of_squares_l462_46204

theorem seven_n_implies_n_is_sum_of_squares (n : ℤ) (A B : ℤ) (h : 7 * n = A^2 + 3 * B^2) :
  ∃ (a b : ℤ), n = a^2 + 3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_seven_n_implies_n_is_sum_of_squares_l462_46204


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l462_46248

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 3/(y+2) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 3/(y₀+2) = 1 ∧ x₀ + y₀ = 2 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l462_46248


namespace NUMINAMATH_CALUDE_unique_be_length_l462_46286

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a unit square ABCD -/
def UnitSquare : (Point × Point × Point × Point) :=
  (⟨0, 0⟩, ⟨1, 0⟩, ⟨1, 1⟩, ⟨0, 1⟩)

/-- Definition of perpendicularity between two line segments -/
def Perpendicular (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y) = 0

/-- Theorem: In a unit square ABCD, with points E on BC, F on CD, and G on DA,
    if AE ⊥ EF, EF ⊥ FG, and GA = 404/1331, then BE = 9/11 -/
theorem unique_be_length (A B C D E F G : Point)
  (square : (A, B, C, D) = UnitSquare)
  (e_on_bc : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = ⟨1, t⟩)
  (f_on_cd : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = ⟨1 - t, 1⟩)
  (g_on_da : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ G = ⟨0, 1 - t⟩)
  (ae_perp_ef : Perpendicular A E E F)
  (ef_perp_fg : Perpendicular E F F G)
  (ga_length : (G.x - A.x)^2 + (G.y - A.y)^2 = (404/1331)^2) :
  (E.x - B.x)^2 + (E.y - B.y)^2 = (9/11)^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_be_length_l462_46286


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l462_46227

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid where
  base_side : ℝ
  is_square_base : base_side > 0
  is_equilateral_lateral : True

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  pyramid : Pyramid
  bottom_on_base : True
  top_on_lateral_faces : True

/-- The volume of an inscribed cube in a pyramid -/
def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

theorem inscribed_cube_volume_in_specific_pyramid :
  ∀ (cube : InscribedCube),
    cube.pyramid.base_side = 2 →
    inscribed_cube_volume cube = 3 * Real.sqrt 6 / 4 :=
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l462_46227


namespace NUMINAMATH_CALUDE_mountain_paths_theorem_l462_46221

/-- The number of paths leading to the summit from the east side -/
def east_paths : ℕ := 3

/-- The number of paths leading to the summit from the west side -/
def west_paths : ℕ := 2

/-- The total number of paths leading to the summit -/
def total_paths : ℕ := east_paths + west_paths

/-- The number of different ways for tourists to go up and come down the mountain -/
def different_ways : ℕ := total_paths * total_paths

theorem mountain_paths_theorem : different_ways = 25 := by
  sorry

end NUMINAMATH_CALUDE_mountain_paths_theorem_l462_46221


namespace NUMINAMATH_CALUDE_train_length_l462_46278

/-- Given a train crossing a bridge, calculate its length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * 1000 / 3600 →
  crossing_time = 30 →
  bridge_length = 235 →
  train_speed * crossing_time - bridge_length = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l462_46278


namespace NUMINAMATH_CALUDE_xy_and_x2y_2xy2_values_l462_46246

theorem xy_and_x2y_2xy2_values (x y : ℝ) 
  (h1 : x - 2*y = 3) 
  (h2 : x^2 - 2*x*y + 4*y^2 = 11) : 
  x * y = 1 ∧ x^2 * y - 2 * x * y^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_xy_and_x2y_2xy2_values_l462_46246


namespace NUMINAMATH_CALUDE_umar_age_is_10_l462_46245

-- Define the ages as natural numbers
def ali_age : ℕ := 8
def age_difference : ℕ := 3
def umar_age_multiplier : ℕ := 2

-- Theorem to prove
theorem umar_age_is_10 :
  let yusaf_age := ali_age - age_difference
  let umar_age := umar_age_multiplier * yusaf_age
  umar_age = 10 := by sorry

end NUMINAMATH_CALUDE_umar_age_is_10_l462_46245


namespace NUMINAMATH_CALUDE_inequality_proof_l462_46273

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  b^2 / a + c^2 / b + a^2 / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l462_46273


namespace NUMINAMATH_CALUDE_income_ratio_l462_46211

theorem income_ratio (A B EA EB : ℚ) 
  (h1 : EA / EB = 3 / 2)
  (h2 : A - EA = 800)
  (h3 : B - EB = 800)
  (h4 : A = 2000) :
  A / B = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_income_ratio_l462_46211


namespace NUMINAMATH_CALUDE_two_pump_fill_time_l462_46291

theorem two_pump_fill_time (small_pump_time large_pump_time : ℝ) 
  (h_small : small_pump_time = 3)
  (h_large : large_pump_time = 1/4)
  (h_positive : small_pump_time > 0 ∧ large_pump_time > 0) :
  1 / (1 / small_pump_time + 1 / large_pump_time) = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_two_pump_fill_time_l462_46291


namespace NUMINAMATH_CALUDE_emily_eggs_collection_l462_46212

theorem emily_eggs_collection (baskets : ℕ) (eggs_per_basket : ℕ) 
  (h1 : baskets = 303) (h2 : eggs_per_basket = 28) : 
  baskets * eggs_per_basket = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_eggs_collection_l462_46212


namespace NUMINAMATH_CALUDE_parallel_lines_a_values_l462_46238

/-- Given two lines l₁ and l₂, if they are parallel, then a = -1 or a = 2 -/
theorem parallel_lines_a_values (a : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | (a - 1) * x + y + 3 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | 2 * x + a * y + 1 = 0}
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l₁ ∧ (x₂, y₂) ∈ l₂ → (a - 1) * (x₂ - x₁) = -(y₂ - y₁)) →
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_values_l462_46238


namespace NUMINAMATH_CALUDE_arithmetic_mean_characterization_l462_46270

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- φ(n) is Euler's totient function -/
def phi (n : ℕ+) : ℕ := sorry

/-- One of n, τ(n), or φ(n) is the arithmetic mean of the other two -/
def is_arithmetic_mean (n : ℕ+) : Prop :=
  (n : ℚ) = (tau n + phi n) / 2 ∨
  (tau n : ℚ) = (n + phi n) / 2 ∨
  (phi n : ℚ) = (n + tau n) / 2

theorem arithmetic_mean_characterization (n : ℕ+) :
  is_arithmetic_mean n ↔ n ∈ ({1, 4, 6, 9} : Set ℕ+) := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_characterization_l462_46270


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l462_46299

theorem trigonometric_inequalities (α β γ : ℝ) : 
  (|Real.cos (α + β)| ≤ |Real.cos α| + |Real.sin β|) ∧ 
  (|Real.sin (α + β)| ≤ |Real.cos α| + |Real.cos β|) ∧ 
  (α + β + γ = 0 → |Real.cos α| + |Real.cos β| + |Real.cos γ| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l462_46299


namespace NUMINAMATH_CALUDE_elena_snow_removal_l462_46236

/-- The volume of snow Elena removes from her pathway -/
def snow_volume (length width depth : ℝ) (compaction_factor : ℝ) : ℝ :=
  length * width * depth * compaction_factor

/-- Theorem stating the volume of snow Elena removes -/
theorem elena_snow_removal :
  snow_volume 30 3 0.75 0.9 = 60.75 := by
  sorry

end NUMINAMATH_CALUDE_elena_snow_removal_l462_46236


namespace NUMINAMATH_CALUDE_production_increase_l462_46233

theorem production_increase (n : ℕ) (old_avg new_avg : ℚ) (today_production : ℚ) : 
  n = 19 →
  old_avg = 50 →
  new_avg = 52 →
  today_production = n * old_avg + today_production →
  (n + 1) * new_avg = n * old_avg + today_production →
  today_production = 90 := by
  sorry

end NUMINAMATH_CALUDE_production_increase_l462_46233


namespace NUMINAMATH_CALUDE_inscribed_hexagon_diagonal_sum_l462_46290

/-- A hexagon inscribed in a circle with five sides of length 90 and one side of length 36 -/
structure InscribedHexagon where
  /-- The length of five sides of the hexagon -/
  regularSideLength : ℝ
  /-- The length of the sixth side of the hexagon -/
  irregularSideLength : ℝ
  /-- The hexagon is inscribed in a circle -/
  inscribed : Bool
  /-- Five sides have the same length -/
  fiveSidesEqual : regularSideLength = 90
  /-- The sixth side has a different length -/
  sixthSideDifferent : irregularSideLength = 36
  /-- The hexagon is actually inscribed in a circle -/
  isInscribed : inscribed = true

/-- The sum of the lengths of the three diagonals drawn from one vertex of the hexagon -/
def diagonalSum (h : InscribedHexagon) : ℝ := 428.4

/-- Theorem: The sum of the lengths of the three diagonals drawn from one vertex
    of the inscribed hexagon with the given properties is 428.4 -/
theorem inscribed_hexagon_diagonal_sum (h : InscribedHexagon) :
  diagonalSum h = 428.4 := by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_diagonal_sum_l462_46290


namespace NUMINAMATH_CALUDE_square_inequality_l462_46296

theorem square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l462_46296


namespace NUMINAMATH_CALUDE_textbook_completion_date_l462_46262

/-- Represents the number of problems solved on a given day -/
def problems_solved (day : ℕ) : ℕ → ℕ
| 0 => day + 1  -- September 6
| n + 1 => day - n  -- Subsequent days

/-- Calculates the total problems solved up to a given day -/
def total_solved (day : ℕ) : ℕ :=
  (List.range (day + 1)).map (problems_solved day) |>.sum

theorem textbook_completion_date 
  (total_problems : ℕ) 
  (problems_left_day3 : ℕ) 
  (h1 : total_problems = 91)
  (h2 : problems_left_day3 = 46)
  (h3 : total_solved 2 = total_problems - problems_left_day3) :
  total_solved 6 = total_problems := by
  sorry

#eval total_solved 6  -- Should output 91

end NUMINAMATH_CALUDE_textbook_completion_date_l462_46262


namespace NUMINAMATH_CALUDE_max_distance_with_swap_20000_30000_l462_46216

/-- Represents the maximum distance a car can travel with one tire swap -/
def maxDistanceWithSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + min frontTireLife (rearTireLife - frontTireLife)

/-- Theorem stating the maximum distance for the given problem -/
theorem max_distance_with_swap_20000_30000 :
  maxDistanceWithSwap 20000 30000 = 30000 := by
  sorry

#eval maxDistanceWithSwap 20000 30000

end NUMINAMATH_CALUDE_max_distance_with_swap_20000_30000_l462_46216


namespace NUMINAMATH_CALUDE_vote_ratio_proof_l462_46254

def candidate_A_votes : ℕ := 14
def total_votes : ℕ := 21

theorem vote_ratio_proof :
  let candidate_B_votes := total_votes - candidate_A_votes
  (candidate_A_votes : ℚ) / candidate_B_votes = 2 := by
  sorry

end NUMINAMATH_CALUDE_vote_ratio_proof_l462_46254


namespace NUMINAMATH_CALUDE_first_negative_term_l462_46224

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem first_negative_term (a₁ d : ℝ) (h₁ : a₁ = 51) (h₂ : d = -4) :
  ∀ k < 14, arithmetic_sequence a₁ d k ≥ 0 ∧
  arithmetic_sequence a₁ d 14 < 0 := by
sorry

end NUMINAMATH_CALUDE_first_negative_term_l462_46224


namespace NUMINAMATH_CALUDE_notebook_buyers_difference_l462_46298

theorem notebook_buyers_difference (notebook_cost : ℕ) 
  (fifth_grade_total : ℕ) (fourth_grade_total : ℕ) 
  (fourth_grade_count : ℕ) :
  notebook_cost > 0 ∧ 
  notebook_cost * 100 ∣ fifth_grade_total ∧ 
  notebook_cost * 100 ∣ fourth_grade_total ∧
  fifth_grade_total = 210 ∧
  fourth_grade_total = 252 ∧
  fourth_grade_count = 28 ∧
  fourth_grade_count ≥ fourth_grade_total / (notebook_cost * 100) →
  (fourth_grade_total / (notebook_cost * 100)) - 
  (fifth_grade_total / (notebook_cost * 100)) = 2 :=
sorry

end NUMINAMATH_CALUDE_notebook_buyers_difference_l462_46298


namespace NUMINAMATH_CALUDE_percent_of_x_l462_46200

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25) / x * 100 = 6 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_l462_46200


namespace NUMINAMATH_CALUDE_time_per_regular_letter_l462_46208

-- Define the given conditions
def days_between_letters : ℕ := 3
def minutes_per_page_regular : ℕ := 10
def minutes_per_page_long : ℕ := 20
def total_minutes_long_letter : ℕ := 80
def total_pages_per_month : ℕ := 24
def days_in_month : ℕ := 30

-- Define the theorem
theorem time_per_regular_letter :
  let pages_long_letter := total_minutes_long_letter / minutes_per_page_long
  let pages_regular_letters := total_pages_per_month - pages_long_letter
  let total_minutes_regular_letters := pages_regular_letters * minutes_per_page_regular
  let num_regular_letters := days_in_month / days_between_letters
  total_minutes_regular_letters / num_regular_letters = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_per_regular_letter_l462_46208


namespace NUMINAMATH_CALUDE_min_disks_for_vincent_l462_46269

/-- Represents the number of disks required to store files -/
def MinDisks (total_files : ℕ) (disk_capacity : ℚ) 
  (files_09 : ℕ) (files_075 : ℕ) (files_05 : ℕ) : ℕ :=
  sorry

theorem min_disks_for_vincent : 
  MinDisks 40 2 5 15 20 = 18 := by sorry

end NUMINAMATH_CALUDE_min_disks_for_vincent_l462_46269


namespace NUMINAMATH_CALUDE_muirhead_inequality_inequality_chain_l462_46267

/-- Symmetric mean function -/
def T (α : List ℝ) (a b c : ℝ) : ℝ := sorry

/-- Majorization relation -/
def Majorizes (α β : List ℝ) : Prop := sorry

theorem muirhead_inequality {α β : List ℝ} {a b c : ℝ} 
  (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : Majorizes α β) :
  T β a b c ≤ T α a b c := sorry

/-- Main theorem to prove -/
theorem inequality_chain (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  T [2, 1, 1] a b c ≤ T [3, 1, 0] a b c ∧ T [3, 1, 0] a b c ≤ T [4, 0, 0] a b c := by
  sorry

end NUMINAMATH_CALUDE_muirhead_inequality_inequality_chain_l462_46267


namespace NUMINAMATH_CALUDE_find_p_l462_46214

theorem find_p (P Q : ℝ) (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l462_46214


namespace NUMINAMATH_CALUDE_volume_maximized_when_perpendicular_l462_46284

/-- A tetrahedron with edge lengths u, v, and w. -/
structure Tetrahedron (u v w : ℝ) where
  edge_u : ℝ := u
  edge_v : ℝ := v
  edge_w : ℝ := w

/-- The volume of a tetrahedron. -/
noncomputable def volume (t : Tetrahedron u v w) : ℝ :=
  sorry

/-- Mutually perpendicular edges of a tetrahedron. -/
def mutually_perpendicular (t : Tetrahedron u v w) : Prop :=
  sorry

/-- Theorem: The volume of a tetrahedron is maximized when its edges are mutually perpendicular. -/
theorem volume_maximized_when_perpendicular (u v w : ℝ) (t : Tetrahedron u v w) :
  mutually_perpendicular t ↔ ∀ (t' : Tetrahedron u v w), volume t ≥ volume t' :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_when_perpendicular_l462_46284


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l462_46285

theorem power_fraction_simplification :
  (16 : ℕ) ^ 24 / (64 : ℕ) ^ 8 = (16 : ℕ) ^ 12 := by sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l462_46285


namespace NUMINAMATH_CALUDE_parallel_plane_sufficient_not_necessary_l462_46240

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Main theorem
theorem parallel_plane_sufficient_not_necessary
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_subset_α : subset m α)
  (h_n_subset_α : subset n α) :
  (∀ l : Line, subset l α → parallel_line_plane l β) ∧
  ∃ m n : Line, subset m α ∧ subset n α ∧ 
    parallel_line_plane m β ∧ parallel_line_plane n β ∧
    ¬ parallel_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_sufficient_not_necessary_l462_46240


namespace NUMINAMATH_CALUDE_inequality_implies_a_bound_l462_46282

theorem inequality_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bound_l462_46282


namespace NUMINAMATH_CALUDE_inverse_function_point_correspondence_l462_46218

theorem inverse_function_point_correspondence 
  (f : ℝ → ℝ) (hf : Function.Bijective f) :
  (1 - f 1 = 2) → (f⁻¹ (-1) - (-1) = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_correspondence_l462_46218


namespace NUMINAMATH_CALUDE_triangle_side_length_l462_46215

/-- Given a triangle ABC where ∠C = 2∠A, a = 34, and c = 60, prove that b = 4352/450 -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (h1 : C = 2 * A) (h2 : a = 34) (h3 : c = 60) : 
  b = 4352 / 450 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l462_46215


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l462_46225

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l462_46225


namespace NUMINAMATH_CALUDE_largest_c_value_l462_46261

theorem largest_c_value : ∃ (c_max : ℚ), c_max = 4 ∧ 
  (∀ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c → c ≤ c_max) ∧
  ((3 * c_max + 4) * (c_max - 2) = 9 * c_max) := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l462_46261


namespace NUMINAMATH_CALUDE_maricela_production_l462_46235

/-- Represents the orange production and juice sale scenario of the Morales sisters. -/
structure OrangeGroveSale where
  trees_per_sister : ℕ
  gabriela_oranges_per_tree : ℕ
  alba_oranges_per_tree : ℕ
  oranges_per_cup : ℕ
  price_per_cup : ℚ
  total_revenue : ℚ

/-- Calculates the number of oranges Maricela's trees must produce per tree. -/
def maricela_oranges_per_tree (sale : OrangeGroveSale) : ℚ :=
  sorry

/-- Theorem stating that given the conditions, Maricela's trees must produce 500 oranges per tree. -/
theorem maricela_production (sale : OrangeGroveSale) 
  (h1 : sale.trees_per_sister = 110)
  (h2 : sale.gabriela_oranges_per_tree = 600)
  (h3 : sale.alba_oranges_per_tree = 400)
  (h4 : sale.oranges_per_cup = 3)
  (h5 : sale.price_per_cup = 4)
  (h6 : sale.total_revenue = 220000) :
  maricela_oranges_per_tree sale = 500 := by
  sorry

end NUMINAMATH_CALUDE_maricela_production_l462_46235


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l462_46253

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smallest circle
  let d₂ : ℝ := 4  -- diameter of middle circle
  let d₃ : ℝ := 6  -- diameter of largest circle
  let r₁ : ℝ := d₁ / 2  -- radius of smallest circle
  let r₂ : ℝ := d₂ / 2  -- radius of middle circle
  let r₃ : ℝ := d₃ / 2  -- radius of largest circle
  let A₁ : ℝ := π * r₁^2  -- area of smallest circle
  let A₂ : ℝ := π * r₂^2  -- area of middle circle
  let A₃ : ℝ := π * r₃^2  -- area of largest circle
  (A₃ - A₂) / A₁ = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l462_46253


namespace NUMINAMATH_CALUDE_ken_kept_pencils_l462_46203

def pencil_distribution (total : ℕ) (manny : ℕ) : Prop :=
  let nilo := 2 * manny
  let carlos := nilo / 2
  let tina := carlos + 10
  let rina := tina - 20
  let given_away := manny + nilo + carlos + tina + rina
  total - given_away = 100

theorem ken_kept_pencils :
  pencil_distribution 250 25 := by sorry

end NUMINAMATH_CALUDE_ken_kept_pencils_l462_46203


namespace NUMINAMATH_CALUDE_cubic_difference_over_difference_l462_46272

theorem cubic_difference_over_difference (r s : ℝ) : 
  3 * r^2 - 4 * r - 12 = 0 →
  3 * s^2 - 4 * s - 12 = 0 →
  (9 * r^3 - 9 * s^3) / (r - s) = 52 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_over_difference_l462_46272


namespace NUMINAMATH_CALUDE_contingency_table_confidence_level_l462_46268

/-- Represents a 2x2 contingency table -/
structure ContingencyTable :=
  (data : Matrix (Fin 2) (Fin 2) ℕ)

/-- Calculates the k^2 value for a contingency table -/
def calculate_k_squared (table : ContingencyTable) : ℝ :=
  sorry

/-- Determines the confidence level based on the k^2 value -/
def confidence_level (k_squared : ℝ) : ℝ :=
  sorry

theorem contingency_table_confidence_level :
  ∀ (table : ContingencyTable),
  calculate_k_squared table = 4.013 →
  confidence_level (calculate_k_squared table) = 0.99 :=
sorry

end NUMINAMATH_CALUDE_contingency_table_confidence_level_l462_46268


namespace NUMINAMATH_CALUDE_sequence_ratio_l462_46251

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ d : ℝ, a₁ = 1 + d ∧ a₂ = 1 + 2*d ∧ 3 = 1 + 3*d

-- Define the geometric sequence
def geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ r : ℝ, b₁ = 1 * r ∧ b₂ = 1 * r^2 ∧ b₃ = 1 * r^3 ∧ 4 = 1 * r^4

theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) 
  (h1 : arithmetic_sequence a₁ a₂) 
  (h2 : geometric_sequence b₁ b₂ b₃) : 
  (a₁ + a₂) / b₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l462_46251


namespace NUMINAMATH_CALUDE_sequence_formula_l462_46228

theorem sequence_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h1 : ∀ n : ℕ+, a n > 0)
  (h2 : ∀ n : ℕ+, S n = (1/2) * (a n + 1 / (a n)))
  (h3 : ∀ n : ℕ+, S n = S (n-1) + a n)
  : ∀ n : ℕ+, a n = Real.sqrt n - Real.sqrt (n-1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l462_46228


namespace NUMINAMATH_CALUDE_range_of_a_range_of_f_when_a_is_2_l462_46206

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- Part 1: Range of a when f(x) ≥ 0 for all x ∈ ℝ
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) → a ∈ Set.Icc (-2) 2 :=
sorry

-- Part 2: Range of f(x) when a = 2 and x ∈ [0, 3]
theorem range_of_f_when_a_is_2 : 
  Set.image (f 2) (Set.Icc 0 3) = Set.Icc 0 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_f_when_a_is_2_l462_46206


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l462_46287

theorem average_of_solutions_is_zero :
  let solutions := {x : ℝ | Real.sqrt (5 * x^2 + 4) = Real.sqrt 29}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solutions ∧ x₂ ∈ solutions ∧ x₁ ≠ x₂ ∧
    (x₁ + x₂) / 2 = 0 ∧
    ∀ (x : ℝ), x ∈ solutions → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l462_46287


namespace NUMINAMATH_CALUDE_division_problem_l462_46259

theorem division_problem (x y : ℤ) (hx : x > 0) : 
  (∃ q : ℤ, x = 11 * y + 4 ∧ q * 11 + 4 = x) →
  (∃ q : ℤ, 2 * x = 6 * (3 * y) + 1 ∧ q * 6 + 1 = 2 * x) →
  7 * y - x = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l462_46259


namespace NUMINAMATH_CALUDE_fitness_center_member_ratio_l462_46242

theorem fitness_center_member_ratio 
  (avg_female : ℝ) 
  (avg_male : ℝ) 
  (avg_total : ℝ) 
  (h1 : avg_female = 140) 
  (h2 : avg_male = 180) 
  (h3 : avg_total = 160) :
  ∃ (f m : ℝ), f > 0 ∧ m > 0 ∧ f / m = 1 ∧
  (f * avg_female + m * avg_male) / (f + m) = avg_total :=
by sorry

end NUMINAMATH_CALUDE_fitness_center_member_ratio_l462_46242


namespace NUMINAMATH_CALUDE_solution_set_empty_l462_46264

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

-- Define the constants a and b
def a : ℝ := 2
def b : ℝ := -3

-- State the theorem
theorem solution_set_empty :
  ∀ x : ℝ, f a (a * x + b) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_empty_l462_46264


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l462_46258

theorem multiplication_puzzle :
  ∀ (G L D E N : ℕ),
    G ≠ L ∧ G ≠ D ∧ G ≠ E ∧ G ≠ N ∧
    L ≠ D ∧ L ≠ E ∧ L ≠ N ∧
    D ≠ E ∧ D ≠ N ∧
    E ≠ N ∧
    1 ≤ G ∧ G ≤ 9 ∧
    1 ≤ L ∧ L ≤ 9 ∧
    1 ≤ D ∧ D ≤ 9 ∧
    1 ≤ E ∧ E ≤ 9 ∧
    1 ≤ N ∧ N ≤ 9 ∧
    100000 * G + 40000 + 1000 * L + 100 * D + 10 * E + N = 
    (100000 * D + 10000 * E + 1000 * N + 100 * G + 40 + L) * 6 →
    G = 1 ∧ L = 2 ∧ D = 8 ∧ E = 5 ∧ N = 7 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l462_46258


namespace NUMINAMATH_CALUDE_highlighted_area_theorem_l462_46260

theorem highlighted_area_theorem (circle_area : ℝ) (angle1 : ℝ) (angle2 : ℝ) :
  circle_area = 20 →
  angle1 = 60 →
  angle2 = 30 →
  (angle1 + angle2) / 360 * circle_area = 5 :=
by sorry

end NUMINAMATH_CALUDE_highlighted_area_theorem_l462_46260


namespace NUMINAMATH_CALUDE_valid_team_combinations_l462_46210

/-- The number of ways to select a team of size k from n people -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of guests -/
def total_guests : ℕ := 5

/-- The number of male guests -/
def male_guests : ℕ := 3

/-- The number of female guests -/
def female_guests : ℕ := 2

/-- The required team size -/
def team_size : ℕ := 3

/-- The number of valid team combinations -/
def valid_combinations : ℕ := 
  choose male_guests 1 * choose female_guests 2 + 
  choose male_guests 2 * choose female_guests 1

theorem valid_team_combinations : valid_combinations = 9 := by sorry

end NUMINAMATH_CALUDE_valid_team_combinations_l462_46210


namespace NUMINAMATH_CALUDE_final_state_is_twelve_and_fourteen_l462_46279

/-- Represents the numbers on the blackboard -/
inductive Number
  | eleven
  | twelve
  | thirteen
  | fourteen
  | fifteen

/-- The state of the blackboard -/
structure BoardState where
  counts : Number → Nat
  total : Nat

/-- The initial state of the blackboard -/
def initial_state : BoardState := {
  counts := λ n => match n with
    | Number.eleven => 11
    | Number.twelve => 12
    | Number.thirteen => 13
    | Number.fourteen => 14
    | Number.fifteen => 15
  total := 65
}

/-- Represents an operation on the board -/
def operation (s : BoardState) : BoardState :=
  sorry  -- Implementation of the operation

/-- Predicate to check if a state has exactly two numbers remaining -/
def has_two_remaining (s : BoardState) : Prop :=
  (s.total = 2) ∧ (∃ a b : Number, a ≠ b ∧ s.counts a > 0 ∧ s.counts b > 0 ∧ 
    ∀ c : Number, c ≠ a ∧ c ≠ b → s.counts c = 0)

/-- The main theorem -/
theorem final_state_is_twelve_and_fourteen :
  ∃ (n : Nat), 
    let final_state := (operation^[n] initial_state)
    has_two_remaining final_state ∧ 
    final_state.counts Number.twelve > 0 ∧ 
    final_state.counts Number.fourteen > 0 :=
  sorry


end NUMINAMATH_CALUDE_final_state_is_twelve_and_fourteen_l462_46279


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l462_46234

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/8))^(1/4))^12 * (((a^16)^(1/4))^(1/8))^12 = a^12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l462_46234


namespace NUMINAMATH_CALUDE_expand_expression_l462_46297

theorem expand_expression (x y : ℝ) : (x + 10) * (2 * y + 10) = 2 * x * y + 10 * x + 20 * y + 100 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l462_46297


namespace NUMINAMATH_CALUDE_sequence_problem_l462_46232

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, b (n + 1) = b n + d

theorem sequence_problem (a b : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2 →
  a 4 = 16 →
  arithmetic_sequence b →
  a 3 = b 3 →
  a 5 = b 5 →
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ n : ℕ, b n = 12*n - 28) ∧
  (∀ n : ℕ, S n = (3*n - 10) * 2^(n+3) - 80) :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l462_46232


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l462_46241

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ

/-- Predicate to check if a cylinder is inscribed in a cone -/
def is_inscribed (cylinder : Cylinder) (cone : Cone) : Prop :=
  -- This is a placeholder for the actual geometric condition
  True

theorem inscribed_cylinder_radius (cone : Cone) (cylinder : Cylinder) :
  cone.diameter = 8 →
  cone.altitude = 10 →
  is_inscribed cylinder cone →
  cylinder.radius * 2 = cylinder.radius * 2 →  -- Diameter equals height
  cylinder.radius = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l462_46241


namespace NUMINAMATH_CALUDE_quadratic_coefficients_from_absolute_value_l462_46293

theorem quadratic_coefficients_from_absolute_value (x : ℝ) :
  (|x - 3| = 4 ↔ x = 7 ∨ x = -1) →
  ∃ d e : ℝ, (∀ x : ℝ, x^2 + d*x + e = 0 ↔ x = 7 ∨ x = -1) ∧ d = -6 ∧ e = -7 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_from_absolute_value_l462_46293


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l462_46230

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 2186 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l462_46230


namespace NUMINAMATH_CALUDE_a_less_than_reciprocal_relationship_l462_46202

theorem a_less_than_reciprocal_relationship (a : ℝ) :
  (a < -1 → a < 1/a) ∧ ¬(a < 1/a → a < -1) :=
by sorry

end NUMINAMATH_CALUDE_a_less_than_reciprocal_relationship_l462_46202


namespace NUMINAMATH_CALUDE_product_of_935421_and_625_l462_46274

theorem product_of_935421_and_625 : 935421 * 625 = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_product_of_935421_and_625_l462_46274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l462_46294

/-- Arithmetic sequence sum -/
def arithmetic_sum (a : ℕ → ℚ) (n : ℕ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n, S n = arithmetic_sum a n) →
  (∀ n, T n = arithmetic_sum b n) →
  (∀ n, S n / T n = (7 * n + 1) / (4 * n + 27)) →
  a 11 / b 11 = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l462_46294


namespace NUMINAMATH_CALUDE_number_calculation_l462_46275

theorem number_calculation (x : ℝ) : (0.1 * 0.3 * 0.5 * x = 90) → x = 6000 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l462_46275


namespace NUMINAMATH_CALUDE_abs_greater_than_y_if_x_greater_than_y_l462_46217

theorem abs_greater_than_y_if_x_greater_than_y (x y : ℝ) (h : x > y) : |x| > y := by
  sorry

end NUMINAMATH_CALUDE_abs_greater_than_y_if_x_greater_than_y_l462_46217


namespace NUMINAMATH_CALUDE_morgan_hula_hoop_time_l462_46292

/-- Given information about hula hooping times for Nancy, Casey, and Morgan,
    prove that Morgan can hula hoop for 21 minutes. -/
theorem morgan_hula_hoop_time :
  ∀ (nancy casey morgan : ℕ),
    nancy = 10 →
    casey = nancy - 3 →
    morgan = 3 * casey →
    morgan = 21 := by
  sorry

end NUMINAMATH_CALUDE_morgan_hula_hoop_time_l462_46292


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l462_46256

/-- The line x + 2y - 6 = 0 is tangent to the circle (x-1)^2 + y^2 = 5 at the point (2, 2) -/
theorem tangent_line_to_circle : 
  let circle : ℝ × ℝ → Prop := λ (x, y) ↦ (x - 1)^2 + y^2 = 5
  let line : ℝ × ℝ → Prop := λ (x, y) ↦ x + 2*y - 6 = 0
  let P : ℝ × ℝ := (2, 2)
  (circle P) ∧ (line P) ∧ 
  (∀ Q : ℝ × ℝ, Q ≠ P → (circle Q ∧ line Q → False)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l462_46256


namespace NUMINAMATH_CALUDE_calzone_time_proof_l462_46281

def calzone_time_calculation (onion_time garlic_time knead_time rest_time assemble_time : ℕ) : Prop :=
  (garlic_time = onion_time / 4) ∧
  (rest_time = 2 * knead_time) ∧
  (assemble_time = (knead_time + rest_time) / 10) ∧
  (onion_time + garlic_time + knead_time + rest_time + assemble_time = 124)

theorem calzone_time_proof :
  ∃ (onion_time garlic_time knead_time rest_time assemble_time : ℕ),
    onion_time = 20 ∧
    knead_time = 30 ∧
    calzone_time_calculation onion_time garlic_time knead_time rest_time assemble_time :=
by
  sorry

end NUMINAMATH_CALUDE_calzone_time_proof_l462_46281


namespace NUMINAMATH_CALUDE_r₂_lower_bound_two_is_greatest_lower_bound_l462_46252

/-- The function f(x) = x² - r₂x + r₃ -/
noncomputable def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂*x + r₃

/-- The sequence {gₙ} defined recursively -/
noncomputable def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The theorem stating the lower bound on |r₂| -/
theorem r₂_lower_bound (r₂ r₃ : ℝ) :
  (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) →
  (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i) →
  (∀ M : ℝ, ∃ n : ℕ, |g r₂ r₃ n| > M) →
  |r₂| > 2 :=
sorry

/-- The theorem stating that 2 is the greatest lower bound -/
theorem two_is_greatest_lower_bound :
  ∀ ε > 0, ∃ r₂ r₃ : ℝ,
    (∀ i : ℕ, i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)) ∧
    (∃ j : ℕ, ∀ i : ℕ, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i) ∧
    (∀ M : ℝ, ∃ n : ℕ, |g r₂ r₃ n| > M) ∧
    |r₂| < 2 + ε :=
sorry

end NUMINAMATH_CALUDE_r₂_lower_bound_two_is_greatest_lower_bound_l462_46252


namespace NUMINAMATH_CALUDE_sandro_children_l462_46263

/-- The number of sons Sandro has -/
def num_sons : ℕ := 3

/-- The ratio of daughters to sons -/
def daughter_son_ratio : ℕ := 6

/-- The number of daughters Sandro has -/
def num_daughters : ℕ := daughter_son_ratio * num_sons

/-- The total number of children Sandro has -/
def total_children : ℕ := num_daughters + num_sons

theorem sandro_children : total_children = 21 := by
  sorry

end NUMINAMATH_CALUDE_sandro_children_l462_46263


namespace NUMINAMATH_CALUDE_max_value_of_function_l462_46213

theorem max_value_of_function :
  let f : ℝ → ℝ := λ x => 3 * Real.sin x + 4 * Real.sqrt (1 + Real.cos (2 * x))
  ∃ M : ℝ, M = Real.sqrt 41 ∧ ∀ x : ℝ, f x ≤ M ∧ ∃ x₀ : ℝ, f x₀ = M := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l462_46213


namespace NUMINAMATH_CALUDE_cosine_function_properties_l462_46229

/-- Given a cosine function f(x) = a * cos(b * x + c) with positive constants a, b, and c,
    if f(x) reaches its first maximum at x = -π/4 and has a maximum value of 3,
    then a = 3, b = 1, and c = π/4. -/
theorem cosine_function_properties (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos (b * x + c)
  (∀ x, f x ≤ 3) ∧ (f (-π/4) = 3) ∧ (∀ x < -π/4, f x < 3) →
  a = 3 ∧ b = 1 ∧ c = π/4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_function_properties_l462_46229


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l462_46289

/-- Proves that in an isosceles triangle where one angle is 40% larger than a right angle,
    the measure of one of the two smallest angles is 27°. -/
theorem isosceles_triangle_angle_measure :
  ∀ (a b c : ℝ),
  -- The triangle is isosceles
  a = b →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- One angle is 40% larger than a right angle (90°)
  c = 90 + 0.4 * 90 →
  -- One of the two smallest angles measures 27°
  a = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l462_46289
