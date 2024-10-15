import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_root_condition_l833_83303

/-- 
If the quadratic function f(x) = -x^2 - 2x + m has a root, 
then m is greater than or equal to 1.
-/
theorem quadratic_root_condition (m : ℝ) : 
  (∃ x, -x^2 - 2*x + m = 0) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l833_83303


namespace NUMINAMATH_CALUDE_negation_of_neither_even_l833_83369

theorem negation_of_neither_even (a b : ℤ) : 
  ¬(¬(Even a) ∧ ¬(Even b)) ↔ (Even a ∨ Even b) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_neither_even_l833_83369


namespace NUMINAMATH_CALUDE_special_numbers_count_l833_83372

/-- Sum of digits of a positive integer -/
def digit_sum (x : ℕ+) : ℕ := sorry

/-- Counts the number of three-digit positive integers satisfying the condition -/
def count_special_numbers : ℕ := sorry

/-- Main theorem -/
theorem special_numbers_count :
  count_special_numbers = 14 := by sorry

end NUMINAMATH_CALUDE_special_numbers_count_l833_83372


namespace NUMINAMATH_CALUDE_surface_area_greater_when_contained_l833_83347

/-- A convex polyhedron in 3D space -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  surfaceArea : ℝ

/-- States that one polyhedron is completely contained within another -/
def IsContainedIn (inner outer : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem: The surface area of the outer polyhedron is greater than
    the surface area of the inner polyhedron when one is contained in the other -/
theorem surface_area_greater_when_contained
  (inner outer : ConvexPolyhedron)
  (h : IsContainedIn inner outer) :
  outer.surfaceArea > inner.surfaceArea :=
sorry

end NUMINAMATH_CALUDE_surface_area_greater_when_contained_l833_83347


namespace NUMINAMATH_CALUDE_line_through_midpoint_of_ellipse_chord_l833_83337

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 16 = 1

/-- The line we're trying to find -/
def line (x y : ℝ) : Prop := x + 8*y - 17 = 0

/-- Theorem stating that the line passing through the midpoint (1, 2) of a chord of the given ellipse
    has the equation x + 8y - 17 = 0 -/
theorem line_through_midpoint_of_ellipse_chord :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧  -- The endpoints of the chord lie on the ellipse
    (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 2 ∧  -- (1, 2) is the midpoint of the chord
    ∀ (x y : ℝ), line x y ↔ y - 2 = (-1/8) * (x - 1) :=  -- The line equation is correct
by sorry

end NUMINAMATH_CALUDE_line_through_midpoint_of_ellipse_chord_l833_83337


namespace NUMINAMATH_CALUDE_equation_one_solutions_l833_83356

theorem equation_one_solutions (x : ℝ) : (5*x + 2) * (4 - x) = 0 ↔ x = -2/5 ∨ x = 4 := by
  sorry

#check equation_one_solutions

end NUMINAMATH_CALUDE_equation_one_solutions_l833_83356


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l833_83371

theorem geometric_sequence_first_term
  (a : ℝ) -- first term
  (r : ℝ) -- common ratio
  (h1 : a * r = 4) -- second term is 4
  (h2 : a * r^3 = 16) -- fourth term is 16
  : a = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l833_83371


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l833_83350

/-- The maximum area of a rectangle with integer side lengths and perimeter 160 feet -/
theorem max_area_rectangle_with_fixed_perimeter :
  ∃ (w h : ℕ), 
    (2 * w + 2 * h = 160) ∧ 
    (∀ (x y : ℕ), (2 * x + 2 * y = 160) → (x * y ≤ w * h)) ∧
    (w * h = 1600) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_fixed_perimeter_l833_83350


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l833_83389

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 22 cm has a base of 8 cm. -/
theorem isosceles_triangle_base_length : ∀ (base congruent_side : ℝ),
  congruent_side = 7 →
  base + 2 * congruent_side = 22 →
  base = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l833_83389


namespace NUMINAMATH_CALUDE_min_value_of_expression_l833_83385

theorem min_value_of_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x*y/z + z*x/y + y*z/x) * (x/(y*z) + y/(z*x) + z/(x*y)) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l833_83385


namespace NUMINAMATH_CALUDE_binary_1010011_conversion_l833_83320

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its hexadecimal representation -/
def decimal_to_hex (n : ℕ) : String :=
  let rec aux (m : ℕ) : List Char :=
    if m = 0 then []
    else
      let digit := m % 16
      let char := if digit < 10 then Char.ofNat (digit + 48) else Char.ofNat (digit + 55)
      char :: aux (m / 16)
  String.mk (aux n).reverse

/-- The binary number 1010011₂ -/
def binary_1010011 : List Bool := [true, true, false, false, true, false, true]

theorem binary_1010011_conversion :
  (binary_to_decimal binary_1010011 = 83) ∧
  (decimal_to_hex (binary_to_decimal binary_1010011) = "53") := by
  sorry

end NUMINAMATH_CALUDE_binary_1010011_conversion_l833_83320


namespace NUMINAMATH_CALUDE_sally_peaches_count_l833_83339

def initial_peaches : ℕ := 13
def first_orchard_peaches : ℕ := 55

def peaches_after_giving : ℕ := initial_peaches - (initial_peaches / 2)
def peaches_after_first_orchard : ℕ := peaches_after_giving + first_orchard_peaches
def second_orchard_peaches : ℕ := 2 * first_orchard_peaches
def total_peaches : ℕ := peaches_after_first_orchard + second_orchard_peaches

theorem sally_peaches_count : total_peaches = 172 := by
  sorry

end NUMINAMATH_CALUDE_sally_peaches_count_l833_83339


namespace NUMINAMATH_CALUDE_hcl_production_l833_83313

-- Define the chemical reaction
structure Reaction where
  reactant1 : ℕ  -- moles of NaCl
  reactant2 : ℕ  -- moles of HNO3
  product : ℕ    -- moles of HCl produced

-- Define the stoichiometric relationship
def stoichiometric_ratio (r : Reaction) : Prop :=
  r.product = min r.reactant1 r.reactant2

-- Theorem statement
theorem hcl_production (nacl_moles hno3_moles : ℕ) 
  (h : nacl_moles = 3 ∧ hno3_moles = 3) : 
  ∃ (r : Reaction), r.reactant1 = nacl_moles ∧ r.reactant2 = hno3_moles ∧ 
  stoichiometric_ratio r ∧ r.product = 3 :=
sorry

end NUMINAMATH_CALUDE_hcl_production_l833_83313


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l833_83317

/-- Represents the color of a marble -/
inductive Color
| Blue
| Yellow

/-- Represents an arrangement of marbles -/
def Arrangement := List Color

/-- Counts the number of adjacent pairs with the same color -/
def countSameColorPairs (arr : Arrangement) : Nat :=
  sorry

/-- Counts the number of adjacent pairs with different colors -/
def countDifferentColorPairs (arr : Arrangement) : Nat :=
  sorry

/-- Checks if an arrangement satisfies the equal pairs condition -/
def isValidArrangement (arr : Arrangement) : Prop :=
  countSameColorPairs arr = countDifferentColorPairs arr

/-- Counts the number of blue marbles in an arrangement -/
def countBlueMarbles (arr : Arrangement) : Nat :=
  sorry

/-- Counts the number of yellow marbles in an arrangement -/
def countYellowMarbles (arr : Arrangement) : Nat :=
  sorry

theorem marble_arrangement_theorem :
  ∃ (validArrangements : List Arrangement),
    (∀ arr ∈ validArrangements, isValidArrangement arr) ∧
    (∀ arr ∈ validArrangements, countBlueMarbles arr = 4) ∧
    (∀ arr ∈ validArrangements, countYellowMarbles arr ≤ 11) ∧
    (validArrangements.length = 35) :=
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l833_83317


namespace NUMINAMATH_CALUDE_max_overlap_area_isosceles_triangles_l833_83387

/-- The maximal area of overlap between two congruent right-angled isosceles triangles -/
theorem max_overlap_area_isosceles_triangles :
  ∃ (overlap_area : ℝ),
    overlap_area = 2/9 ∧
    ∀ (x : ℝ),
      0 ≤ x ∧ x ≤ 1 →
      let triangle_area := 1/4 * (1 - x)^2
      let pentagon_area := 1/4 * (1 - x) * (3*x + 1)
      overlap_area ≥ max triangle_area pentagon_area :=
by sorry

end NUMINAMATH_CALUDE_max_overlap_area_isosceles_triangles_l833_83387


namespace NUMINAMATH_CALUDE_tetrahedron_volume_EFGH_l833_83392

/-- The volume of a tetrahedron given its edge lengths -/
def tetrahedron_volume (EF EG EH FG FH GH : ℝ) : ℝ :=
  sorry

/-- Theorem: The volume of tetrahedron EFGH with given edge lengths is √3/2 -/
theorem tetrahedron_volume_EFGH :
  tetrahedron_volume 5 (3 * Real.sqrt 2) (2 * Real.sqrt 3) 4 (Real.sqrt 37) 3 = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_EFGH_l833_83392


namespace NUMINAMATH_CALUDE_rational_inequality_l833_83390

theorem rational_inequality (a b : ℚ) (h1 : a + b > 0) (h2 : a * b < 0) :
  a > 0 ∧ b < 0 ∧ |a| > |b| := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_l833_83390


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l833_83393

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l833_83393


namespace NUMINAMATH_CALUDE_frog_jump_probability_l833_83326

/-- Represents a jump of the frog -/
structure Jump where
  direction : ℝ × ℝ
  length : ℝ
  random_direction : Bool

/-- Represents the frog's journey -/
structure FrogJourney where
  jumps : List Jump
  final_position : ℝ × ℝ

/-- The probability of the frog's final position being within 1 meter of the start -/
def probability_within_one_meter (journey : FrogJourney) : ℝ :=
  sorry

/-- Theorem stating the probability of the frog's final position being within 1 meter of the start -/
theorem frog_jump_probability :
  ∀ (journey : FrogJourney),
    journey.jumps.length = 4 ∧
    (∀ jump ∈ journey.jumps, jump.length = 1 ∧ jump.random_direction) →
    probability_within_one_meter journey = 1/5 :=
  sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l833_83326


namespace NUMINAMATH_CALUDE_cube_edge_length_specific_l833_83361

/-- The edge length of a cube with the same volume as a rectangular block -/
def cube_edge_length (l w h : ℝ) : ℝ :=
  (l * w * h) ^ (1/3)

/-- Theorem: The edge length of a cube with the same volume as a 50cm × 8cm × 20cm rectangular block is 20cm -/
theorem cube_edge_length_specific : cube_edge_length 50 8 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_specific_l833_83361


namespace NUMINAMATH_CALUDE_mat_length_approximation_l833_83377

/-- Represents the setup of a circular table with place mats -/
structure TableSetup where
  tableRadius : ℝ
  numMats : ℕ
  matWidth : ℝ

/-- Calculates the length of place mats given a table setup -/
def calculateMatLength (setup : TableSetup) : ℝ :=
  sorry

/-- Theorem stating that for the given setup, the mat length is approximately 3.9308 meters -/
theorem mat_length_approximation (setup : TableSetup) 
  (h1 : setup.tableRadius = 6)
  (h2 : setup.numMats = 8)
  (h3 : setup.matWidth = 1.5) :
  abs (calculateMatLength setup - 3.9308) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_mat_length_approximation_l833_83377


namespace NUMINAMATH_CALUDE_unique_solution_condition_l833_83310

theorem unique_solution_condition (j : ℝ) : 
  (∃! x : ℝ, (3 * x + 4) * (x - 6) = -52 + j * x) ↔ 
  (j = -14 + 4 * Real.sqrt 21 ∨ j = -14 - 4 * Real.sqrt 21) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l833_83310


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l833_83305

theorem quadratic_equation_solution (a : ℝ) : 
  ((-1)^2 - 2*(-1) + a = 0) → 
  (3^2 - 2*3 + a = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x + a = 0 → (x = -1 ∨ x = 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l833_83305


namespace NUMINAMATH_CALUDE_ducks_in_lake_l833_83348

/-- The number of ducks initially in the lake -/
def initial_ducks : ℕ := 13

/-- The number of ducks that joined the lake -/
def joining_ducks : ℕ := 20

/-- The total number of ducks in the lake -/
def total_ducks : ℕ := initial_ducks + joining_ducks

theorem ducks_in_lake : total_ducks = 33 := by
  sorry

end NUMINAMATH_CALUDE_ducks_in_lake_l833_83348


namespace NUMINAMATH_CALUDE_speed_in_still_water_l833_83340

theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 26) 
  (h2 : downstream_speed = 30) : 
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l833_83340


namespace NUMINAMATH_CALUDE_complex_equality_implies_ratio_l833_83334

theorem complex_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_implies_ratio_l833_83334


namespace NUMINAMATH_CALUDE_fraction_transformation_l833_83304

theorem fraction_transformation (x : ℝ) (h : x ≠ 1) : -1 / (1 - x) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_transformation_l833_83304


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l833_83352

def ellipse (x y : ℝ) : Prop := y^2/16 + x^2/4 = 1

theorem max_value_on_ellipse :
  ∃ (M : ℝ), ∀ (x y : ℝ), ellipse x y → |2*Real.sqrt 3*x + y - 1| ≤ M ∧
  ∃ (x₀ y₀ : ℝ), ellipse x₀ y₀ ∧ |2*Real.sqrt 3*x₀ + y₀ - 1| = M :=
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l833_83352


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_is_correct_l833_83386

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | abs x ≤ 2}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

-- Define the complement of A ∩ B in ℝ
def complement_A_intersect_B : Set ℝ := {x : ℝ | x < -2 ∨ x > 0}

-- State the theorem
theorem complement_A_intersect_B_is_correct :
  (Set.univ : Set ℝ) \ (A ∩ B) = complement_A_intersect_B := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_is_correct_l833_83386


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l833_83336

-- Problem 1
theorem problem_1 : 
  2 * Real.cos (π / 4) + (3 - Real.pi) ^ 0 - |2 - Real.sqrt 8| - (-1/3)⁻¹ = 6 - Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : 
  let x : ℝ := Real.sqrt 27 + |-2| - 3 * Real.tan (π / 3)
  ((x^2 - 1) / (x^2 - 2*x + 1) - 1 / (x - 1)) / ((x + 2) / (x - 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l833_83336


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l833_83346

theorem chocolate_box_problem (total : ℕ) (remaining : ℕ) : 
  (remaining = 36) →
  (total = (remaining : ℚ) * 3 / (1 - (4/15 : ℚ))) →
  (total = 98) := by
sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l833_83346


namespace NUMINAMATH_CALUDE_largest_non_representable_number_l833_83360

theorem largest_non_representable_number : ∃ (n : ℕ), n > 0 ∧
  (∀ x y : ℕ, x > 0 → y > 0 → 9 * x + 11 * y ≠ n) ∧
  (∀ m : ℕ, m > n → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 9 * x + 11 * y = m) ∧
  (∀ k : ℕ, k > n → ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 9 * x + 11 * y = k) →
  n = 99 := by
sorry

end NUMINAMATH_CALUDE_largest_non_representable_number_l833_83360


namespace NUMINAMATH_CALUDE_justins_dogs_l833_83395

theorem justins_dogs (camden_dogs rico_dogs justin_dogs : ℕ) : 
  camden_dogs = (3 * rico_dogs) / 4 →
  rico_dogs = justin_dogs + 10 →
  camden_dogs * 4 = 72 →
  justin_dogs = 14 := by
  sorry

end NUMINAMATH_CALUDE_justins_dogs_l833_83395


namespace NUMINAMATH_CALUDE_no_quadratic_trinomial_with_odd_coeffs_and_2022th_root_l833_83374

theorem no_quadratic_trinomial_with_odd_coeffs_and_2022th_root :
  ¬ ∃ (a b c : ℤ), 
    (Odd a ∧ Odd b ∧ Odd c) ∧ 
    (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomial_with_odd_coeffs_and_2022th_root_l833_83374


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l833_83345

theorem cube_root_equation_solution :
  ∃! x : ℝ, (4 - x / 3) ^ (1/3 : ℝ) = -2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l833_83345


namespace NUMINAMATH_CALUDE_money_problem_l833_83397

/-- Given three people A, B, and C with certain amounts of money, 
    prove that A and C together have 300 rupees. -/
theorem money_problem (a b c : ℕ) : 
  a + b + c = 700 →
  b + c = 600 →
  c = 200 →
  a + c = 300 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l833_83397


namespace NUMINAMATH_CALUDE_shifted_line_equation_l833_83358

/-- Given a line with equation y = -2x, shifting it one unit upwards
    results in the equation y = -2x + 1 -/
theorem shifted_line_equation (x y : ℝ) :
  (y = -2 * x) → (y + 1 = -2 * x + 1) := by sorry

end NUMINAMATH_CALUDE_shifted_line_equation_l833_83358


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l833_83315

theorem sqrt_x_minus_one_squared (x : ℝ) (h : |2 - x| = 2 + |x|) : 
  Real.sqrt ((x - 1)^2) = 1 - x := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_squared_l833_83315


namespace NUMINAMATH_CALUDE_unique_correct_ranking_l833_83319

/-- Represents the participants in the long jump competition -/
inductive Participant
| Decimals
| Elementary
| Xiaohua
| Xiaoyuan
| Exploration

/-- Represents the ranking of participants -/
def Ranking := Participant → Fin 5

/-- Checks if a ranking satisfies all the given conditions -/
def satisfies_conditions (r : Ranking) : Prop :=
  (r Participant.Decimals < r Participant.Elementary) ∧
  (r Participant.Xiaohua > r Participant.Xiaoyuan) ∧
  (r Participant.Exploration > r Participant.Elementary) ∧
  (r Participant.Elementary < r Participant.Xiaohua) ∧
  (r Participant.Xiaoyuan > r Participant.Exploration)

/-- The correct ranking of participants -/
def correct_ranking : Ranking :=
  fun p => match p with
    | Participant.Decimals => 0
    | Participant.Elementary => 1
    | Participant.Exploration => 2
    | Participant.Xiaoyuan => 3
    | Participant.Xiaohua => 4

/-- Theorem stating that the correct_ranking is the unique ranking that satisfies all conditions -/
theorem unique_correct_ranking :
  satisfies_conditions correct_ranking ∧
  ∀ (r : Ranking), satisfies_conditions r → r = correct_ranking :=
sorry

end NUMINAMATH_CALUDE_unique_correct_ranking_l833_83319


namespace NUMINAMATH_CALUDE_chord_equation_l833_83330

/-- Given positive real numbers m, n, s, t satisfying certain conditions,
    prove that the equation of a line containing a chord of an ellipse is 2x + y - 4 = 0 -/
theorem chord_equation (m n s t : ℝ) (hm : m > 0) (hn : n > 0) (hs : s > 0) (ht : t > 0)
  (h_sum : m + n = 3)
  (h_frac : m / s + n / t = 1)
  (h_order : m < n)
  (h_min : ∀ (s' t' : ℝ), s' > 0 → t' > 0 → m / s' + n / t' = 1 → s' + t' ≥ 3 + 2 * Real.sqrt 2)
  (h_midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2 / 4 + y₁^2 / 16 = 1 ∧
    x₂^2 / 4 + y₂^2 / 16 = 1 ∧
    (x₁ + x₂) / 2 = m ∧
    (y₁ + y₂) / 2 = n) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ 2 * a + b = 0 ∧ a ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_chord_equation_l833_83330


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l833_83349

theorem arithmetic_geometric_sequence (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (2 * b = a + c) →  -- arithmetic sequence condition
  (b ^ 2 = c * (a + 1)) →  -- geometric sequence condition when a is increased by 1
  (b ^ 2 = a * (c + 2)) →  -- geometric sequence condition when c is increased by 2
  b = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l833_83349


namespace NUMINAMATH_CALUDE_candy_distribution_l833_83370

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (chocolate_hearts_bags : ℕ) (chocolate_kisses_bags : ℕ) 
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : chocolate_hearts_bags = 2)
  (h4 : chocolate_kisses_bags = 3)
  (h5 : total_candy % total_bags = 0) :
  let candy_per_bag := total_candy / total_bags
  let chocolate_bags := chocolate_hearts_bags + chocolate_kisses_bags
  let non_chocolate_bags := total_bags - chocolate_bags
  non_chocolate_bags * candy_per_bag = 28 := by
sorry


end NUMINAMATH_CALUDE_candy_distribution_l833_83370


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l833_83357

theorem arithmetic_geometric_sequence_sum (a b c d : ℝ) : 
  (∃ k : ℝ, a = 6 + k ∧ b = 6 + 2*k ∧ 48 = 6 + 3*k) →  -- arithmetic sequence condition
  (∃ q : ℝ, c = 6*q ∧ d = 6*q^2 ∧ 48 = 6*q^3) →        -- geometric sequence condition
  a + b + c + d = 111 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_sum_l833_83357


namespace NUMINAMATH_CALUDE_vector_sum_l833_83309

theorem vector_sum (x : ℝ) : 
  (⟨-3, 4, -2⟩ : ℝ × ℝ × ℝ) + (⟨5, -3, x⟩ : ℝ × ℝ × ℝ) = ⟨2, 1, x - 2⟩ := by
sorry

end NUMINAMATH_CALUDE_vector_sum_l833_83309


namespace NUMINAMATH_CALUDE_square_exterior_points_distance_l833_83391

/-- Given a square ABCD with side length 10 and exterior points E and F,
    prove that EF^2 = 850 + 250√125 when BE = DF = 7 and AE = CF = 15 -/
theorem square_exterior_points_distance (A B C D E F : ℝ × ℝ) : 
  let side_length : ℝ := 10
  let be_df_length : ℝ := 7
  let ae_cf_length : ℝ := 15
  -- Square ABCD definition
  (A = (0, side_length) ∧ 
   B = (side_length, side_length) ∧ 
   C = (side_length, 0) ∧ 
   D = (0, 0)) →
  -- E and F are exterior points
  (E.1 > side_length ∧ E.2 = side_length) →
  (F.1 = 0 ∧ F.2 < 0) →
  -- BE and DF lengths
  (dist B E = be_df_length ∧ dist D F = be_df_length) →
  -- AE and CF lengths
  (dist A E = ae_cf_length ∧ dist C F = ae_cf_length) →
  -- Conclusion: EF^2 = 850 + 250√125
  dist E F ^ 2 = 850 + 250 * Real.sqrt 125 :=
by sorry


end NUMINAMATH_CALUDE_square_exterior_points_distance_l833_83391


namespace NUMINAMATH_CALUDE_equation_equivalence_l833_83376

theorem equation_equivalence : ∃ (b c : ℝ), 
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 1 ∨ x = 7) →
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 7) →
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l833_83376


namespace NUMINAMATH_CALUDE_triangulation_labeling_exists_l833_83333

/-- A convex polygon with n+1 vertices -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin (n+1) → ℝ × ℝ

/-- A triangulation of a convex polygon -/
structure Triangulation (n : ℕ) where
  polygon : ConvexPolygon n
  triangles : Fin (n-1) → Fin 3 → Fin (n+1)

/-- A labeling of triangles in a triangulation -/
def Labeling (n : ℕ) := Fin (n-1) → Fin (n-1)

/-- Predicate to check if a vertex is part of a triangle -/
def isVertexOfTriangle (n : ℕ) (t : Triangulation n) (v : Fin (n+1)) (tri : Fin (n-1)) : Prop :=
  ∃ i : Fin 3, t.triangles tri i = v

/-- Main theorem statement -/
theorem triangulation_labeling_exists (n : ℕ) (t : Triangulation n) :
  ∃ l : Labeling n, ∀ i : Fin (n-1), isVertexOfTriangle n t i (l i) :=
sorry

end NUMINAMATH_CALUDE_triangulation_labeling_exists_l833_83333


namespace NUMINAMATH_CALUDE_smallest_triangle_side_l833_83329

theorem smallest_triangle_side : ∃ (s : ℕ), 
  (s : ℝ) ≥ 4 ∧ 
  (∀ (t : ℕ), (t : ℝ) ≥ 4 → 
    (8.5 + (t : ℝ) > 11.5) ∧
    (8.5 + 11.5 > (t : ℝ)) ∧
    (11.5 + (t : ℝ) > 8.5)) ∧
  (∀ (u : ℕ), (u : ℝ) < 4 → 
    ¬((8.5 + (u : ℝ) > 11.5) ∧
      (8.5 + 11.5 > (u : ℝ)) ∧
      (11.5 + (u : ℝ) > 8.5))) :=
by
  sorry

#check smallest_triangle_side

end NUMINAMATH_CALUDE_smallest_triangle_side_l833_83329


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l833_83331

theorem arithmetic_mean_of_fractions :
  (5/6 : ℚ) = (7/9 + 8/9) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l833_83331


namespace NUMINAMATH_CALUDE_hyperbola_equation_part1_hyperbola_equation_part2_l833_83375

-- Part 1
theorem hyperbola_equation_part1 (c : ℝ) (h1 : c = Real.sqrt 6) :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / a^2 - y^2 / (6 - a^2) = 1) ↔ 
  (x^2 / 5 - y^2 = 1)) ∧
  ((-5)^2 / a^2 - 2^2 / (6 - a^2) = 1) := by
sorry

-- Part 2
theorem hyperbola_equation_part2 (x1 y1 x2 y2 : ℝ) 
  (h1 : x1 = 3 ∧ y1 = -4 * Real.sqrt 2)
  (h2 : x2 = 9/4 ∧ y2 = 5) :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧
  (∀ (x y : ℝ), (m * x^2 - n * y^2 = 1) ↔ 
  (y^2 / 16 - x^2 / 9 = 1)) ∧
  (m * x1^2 - n * y1^2 = 1) ∧
  (m * x2^2 - n * y2^2 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_part1_hyperbola_equation_part2_l833_83375


namespace NUMINAMATH_CALUDE_encryption_theorem_l833_83368

/-- Represents the encryption table --/
def encryption_table : Fin 16 → Fin 16 := sorry

/-- Applies the encryption once to a string of 16 characters --/
def apply_encryption (s : String) : String := sorry

/-- Applies the encryption n times to a string --/
def apply_encryption_n_times (s : String) (n : ℕ) : String := sorry

/-- The last three characters of a string --/
def last_three (s : String) : String := sorry

theorem encryption_theorem :
  ∀ s : String,
  last_three s = "уао" →
  apply_encryption_n_times (apply_encryption s) 2014 = s →
  ∃ t : String, last_three t = "чку" ∧ apply_encryption_n_times t 2015 = s :=
sorry

end NUMINAMATH_CALUDE_encryption_theorem_l833_83368


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l833_83366

theorem tennis_tournament_matches (total_players : ℕ) (bye_players : ℕ) (first_round_players : ℕ) :
  total_players = 128 →
  bye_players = 40 →
  first_round_players = 88 →
  (total_players = bye_players + first_round_players) →
  (∃ (total_matches : ℕ), total_matches = 127 ∧
    total_matches = (first_round_players / 2) + (total_players - 1)) := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l833_83366


namespace NUMINAMATH_CALUDE_b_plus_c_equals_nine_l833_83384

theorem b_plus_c_equals_nine (a b c d : ℤ) 
  (h1 : a + b = 11) 
  (h2 : c + d = 3) 
  (h3 : a + d = 5) : 
  b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_nine_l833_83384


namespace NUMINAMATH_CALUDE_parabola_directrix_l833_83380

/-- For a parabola with equation y = 2x^2, its directrix has the equation y = -1/8 -/
theorem parabola_directrix (x y : ℝ) :
  y = 2 * x^2 → (∃ (k : ℝ), y = k ∧ k = -1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l833_83380


namespace NUMINAMATH_CALUDE_inflation_cost_increase_l833_83344

def original_lumber_cost : ℝ := 450
def original_nails_cost : ℝ := 30
def original_fabric_cost : ℝ := 80
def lumber_inflation_rate : ℝ := 0.20
def nails_inflation_rate : ℝ := 0.10
def fabric_inflation_rate : ℝ := 0.05

def total_increased_cost : ℝ :=
  (original_lumber_cost * lumber_inflation_rate) +
  (original_nails_cost * nails_inflation_rate) +
  (original_fabric_cost * fabric_inflation_rate)

theorem inflation_cost_increase :
  total_increased_cost = 97 := by sorry

end NUMINAMATH_CALUDE_inflation_cost_increase_l833_83344


namespace NUMINAMATH_CALUDE_margarita_ricciana_difference_l833_83302

/-- Ricciana's running distance in feet -/
def ricciana_run : ℝ := 20

/-- Ricciana's jumping distance in feet -/
def ricciana_jump : ℝ := 4

/-- Margarita's running distance in feet -/
def margarita_run : ℝ := 18

/-- Calculates Margarita's jumping distance in feet -/
def margarita_jump : ℝ := 2 * ricciana_jump - 1

/-- Calculates Ricciana's total distance (run + jump) in feet -/
def ricciana_total : ℝ := ricciana_run + ricciana_jump

/-- Calculates Margarita's total distance (run + jump) in feet -/
def margarita_total : ℝ := margarita_run + margarita_jump

/-- Proves that Margarita ran and jumped 1 foot farther than Ricciana -/
theorem margarita_ricciana_difference : margarita_total - ricciana_total = 1 := by
  sorry

end NUMINAMATH_CALUDE_margarita_ricciana_difference_l833_83302


namespace NUMINAMATH_CALUDE_f_definition_l833_83321

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_definition (x : ℝ) : f x = 2 - x :=
  sorry

end NUMINAMATH_CALUDE_f_definition_l833_83321


namespace NUMINAMATH_CALUDE_a_2021_eq_6_l833_83398

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => 
    if n % 3 = 0 then a (n / 3)
    else a (n / 3) + 1

theorem a_2021_eq_6 : a 2021 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_2021_eq_6_l833_83398


namespace NUMINAMATH_CALUDE_sales_growth_rate_l833_83383

theorem sales_growth_rate (initial_sales final_sales : ℝ) 
  (h1 : initial_sales = 2000000)
  (h2 : final_sales = 2880000)
  (h3 : ∃ r : ℝ, initial_sales * (1 + r)^2 = final_sales) :
  ∃ r : ℝ, initial_sales * (1 + r)^2 = final_sales ∧ r = 0.2 :=
sorry

end NUMINAMATH_CALUDE_sales_growth_rate_l833_83383


namespace NUMINAMATH_CALUDE_complex_modulus_product_l833_83364

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l833_83364


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l833_83351

theorem largest_n_divisible_by_seven : 
  ∀ n : ℕ, n < 50000 → 
  (6 * (n - 3)^3 - n^2 + 10*n - 15) % 7 = 0 → 
  n ≤ 49999 ∧ 
  (6 * (49999 - 3)^3 - 49999^2 + 10*49999 - 15) % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_l833_83351


namespace NUMINAMATH_CALUDE_benny_stored_bales_l833_83394

/-- The number of bales Benny stored in the barn -/
def bales_stored (initial_bales final_bales : ℕ) : ℕ :=
  final_bales - initial_bales

/-- Proof that Benny stored 35 bales in the barn -/
theorem benny_stored_bales : 
  let initial_bales : ℕ := 47
  let final_bales : ℕ := 82
  bales_stored initial_bales final_bales = 35 := by
  sorry

end NUMINAMATH_CALUDE_benny_stored_bales_l833_83394


namespace NUMINAMATH_CALUDE_probability_multiple_of_15_l833_83327

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A five-digit number without repeating digits -/
def FiveDigitNumber := {n : Finset Nat // n.card = 5 ∧ n ⊆ digits}

/-- The set of all possible five-digit numbers -/
def allNumbers : Finset FiveDigitNumber := sorry

/-- Predicate to check if a number is a multiple of 15 -/
def isMultipleOf15 (n : FiveDigitNumber) : Prop := sorry

/-- The set of five-digit numbers that are multiples of 15 -/
def multiplesOf15 : Finset FiveDigitNumber := sorry

/-- The probability of drawing a multiple of 15 -/
def probabilityMultipleOf15 : ℚ := (multiplesOf15.card : ℚ) / (allNumbers.card : ℚ)

theorem probability_multiple_of_15 : probabilityMultipleOf15 = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_multiple_of_15_l833_83327


namespace NUMINAMATH_CALUDE_room_width_calculation_l833_83367

/-- Given a rectangular room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 700)
    (h3 : total_cost = 14437.5) :
    total_cost / cost_per_sqm / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l833_83367


namespace NUMINAMATH_CALUDE_prob_two_red_marbles_l833_83323

/-- The probability of selecting two red marbles without replacement from a bag containing 2 red marbles and 3 green marbles is 1/10. -/
theorem prob_two_red_marbles (red : ℕ) (green : ℕ) (h1 : red = 2) (h2 : green = 3) :
  (red / (red + green)) * ((red - 1) / (red + green - 1)) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_marbles_l833_83323


namespace NUMINAMATH_CALUDE_f_increasing_iff_l833_83300

/-- A piecewise function f defined on ℝ --/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x ≥ 1 then a^x else (3-a)*x + 1

/-- Theorem stating the condition for f to be increasing --/
theorem f_increasing_iff (a : ℝ) :
  StrictMono (f a) ↔ 2 ≤ a ∧ a < 3 :=
sorry

#check f_increasing_iff

end NUMINAMATH_CALUDE_f_increasing_iff_l833_83300


namespace NUMINAMATH_CALUDE_percent_of_number_l833_83325

theorem percent_of_number (x : ℝ) : (26 / 100) * x = 93.6 → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l833_83325


namespace NUMINAMATH_CALUDE_min_balls_for_three_colors_l833_83306

/-- Represents the number of balls of a specific color in the box -/
def BallCount := ℕ

/-- Represents the total number of balls in the box -/
def TotalBalls : ℕ := 111

/-- Represents the number of different colors of balls in the box -/
def NumColors : ℕ := 4

/-- Represents the number of balls that guarantees at least four different colors when drawn -/
def GuaranteeFourColors : ℕ := 100

/-- Represents a function that returns the minimum number of balls to draw to ensure at least three different colors -/
def minBallsForThreeColors (total : ℕ) (numColors : ℕ) (guaranteeFour : ℕ) : ℕ := 
  total - guaranteeFour + 1

/-- Theorem stating that the minimum number of balls to draw to ensure at least three different colors is 88 -/
theorem min_balls_for_three_colors : 
  minBallsForThreeColors TotalBalls NumColors GuaranteeFourColors = 88 := by
  sorry

end NUMINAMATH_CALUDE_min_balls_for_three_colors_l833_83306


namespace NUMINAMATH_CALUDE_student_path_probability_l833_83381

/-- Represents the number of paths between two points given the number of eastward and southward moves -/
def num_paths (east south : ℕ) : ℕ := Nat.choose (east + south) east

/-- Represents the total number of paths from A to B -/
def total_paths : ℕ := num_paths 6 5

/-- Represents the number of paths from A to B that pass through C and D -/
def paths_through_C_and_D : ℕ := num_paths 3 2 * num_paths 2 1 * num_paths 1 2

/-- The probability of choosing a specific path given the number of moves -/
def path_probability (moves : ℕ) : ℚ := (1 / 2) ^ moves

theorem student_path_probability : 
  (paths_through_C_and_D : ℚ) / total_paths = 15 / 77 := by sorry

end NUMINAMATH_CALUDE_student_path_probability_l833_83381


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l833_83396

theorem complex_arithmetic_equality : ((-1 : ℤ) ^ 2024) + (-10 : ℤ) / (1 / 2 : ℚ) * 2 + (2 - (-3 : ℤ) ^ 3) = -10 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l833_83396


namespace NUMINAMATH_CALUDE_canoe_row_probability_l833_83324

-- Define the probability of each oar working
def p_left_works : ℚ := 3/5
def p_right_works : ℚ := 3/5

-- Define the event of being able to row the canoe
def can_row : ℚ := 
  p_left_works * p_right_works +  -- both oars work
  p_left_works * (1 - p_right_works) +  -- left works, right breaks
  (1 - p_left_works) * p_right_works  -- left breaks, right works

-- Theorem statement
theorem canoe_row_probability : can_row = 21/25 := by
  sorry

end NUMINAMATH_CALUDE_canoe_row_probability_l833_83324


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l833_83301

theorem sin_two_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * (Real.cos α)^2 = Real.sin (π/4 - α)) :
  Real.sin (2*α) = 1 ∨ Real.sin (2*α) = -17/18 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l833_83301


namespace NUMINAMATH_CALUDE_frank_skee_ball_tickets_l833_83318

/-- The number of tickets Frank won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 33

/-- The cost of one candy in tickets -/
def candy_cost : ℕ := 6

/-- The number of candies Frank can buy with his total tickets -/
def candies_bought : ℕ := 7

/-- The number of tickets Frank won playing 'skee ball' -/
def skee_ball_tickets : ℕ := candies_bought * candy_cost - whack_a_mole_tickets

theorem frank_skee_ball_tickets : skee_ball_tickets = 9 := by
  sorry

end NUMINAMATH_CALUDE_frank_skee_ball_tickets_l833_83318


namespace NUMINAMATH_CALUDE_tunnel_length_l833_83316

/-- The length of a tunnel given a train passing through it. -/
theorem tunnel_length
  (train_length : ℝ)
  (transit_time : ℝ)
  (train_speed : ℝ)
  (h1 : train_length = 2)
  (h2 : transit_time = 4 / 60)  -- 4 minutes converted to hours
  (h3 : train_speed = 90) :
  train_speed * transit_time - train_length = 4 :=
by sorry

end NUMINAMATH_CALUDE_tunnel_length_l833_83316


namespace NUMINAMATH_CALUDE_inequality_proof_l833_83363

theorem inequality_proof (A B C a b c r : ℝ) 
  (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) : 
  (A + a + B + b) / (A + a + B + b + c + r) + 
  (B + b + C + c) / (B + b + C + c + a + r) > 
  (c + c + A + a) / (C + c + A + a + b + r) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l833_83363


namespace NUMINAMATH_CALUDE_train_speed_problem_l833_83338

/-- Calculates the speed of train A given the conditions of the problem -/
theorem train_speed_problem (length_A length_B : ℝ) (speed_B : ℝ) (crossing_time : ℝ) :
  length_A = 150 →
  length_B = 150 →
  speed_B = 36 →
  crossing_time = 12 →
  (length_A + length_B) / crossing_time * 3.6 - speed_B = 54 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l833_83338


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l833_83359

theorem quadratic_root_difference (x : ℝ) : 
  let roots := {r : ℝ | r^2 - 7*r + 11 = 0}
  ∃ (r₁ r₂ : ℝ), r₁ ∈ roots ∧ r₂ ∈ roots ∧ r₁ ≠ r₂ ∧ |r₁ - r₂| = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l833_83359


namespace NUMINAMATH_CALUDE_continuity_properties_l833_83328

theorem continuity_properties :
  (¬ ∀ a b : ℤ, a < b → ∃ c : ℤ, a < c ∧ c < b) ∧
  (¬ ∀ S : Set ℤ, S.Nonempty → (∃ x : ℤ, ∀ y ∈ S, y ≤ x) → ∃ z : ℤ, ∀ y ∈ S, y ≤ z ∧ ∀ w : ℤ, (∀ y ∈ S, y ≤ w) → z ≤ w) ∧
  (∀ a b : ℚ, a < b → ∃ c : ℚ, a < c ∧ c < b) ∧
  (¬ ∀ S : Set ℚ, S.Nonempty → (∃ x : ℚ, ∀ y ∈ S, y ≤ x) → ∃ z : ℚ, ∀ y ∈ S, y ≤ z ∧ ∀ w : ℚ, (∀ y ∈ S, y ≤ w) → z ≤ w) :=
by sorry

end NUMINAMATH_CALUDE_continuity_properties_l833_83328


namespace NUMINAMATH_CALUDE_find_a_l833_83388

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

theorem find_a : ∃ (a : ℝ), A ∩ B a = {3} → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l833_83388


namespace NUMINAMATH_CALUDE_cherry_pricing_and_profit_l833_83355

/-- Represents the cost and quantity of cherries --/
structure CherryData where
  yellow_cost : ℝ
  red_cost : ℝ
  yellow_quantity : ℝ
  red_quantity : ℝ

/-- Represents the sales data for red light cherries --/
structure SalesData where
  week1_price : ℝ
  week1_quantity : ℝ
  week2_price_decrease : ℝ
  week2_quantity : ℝ
  week3_discount : ℝ

/-- Theorem stating the cost price of red light cherries and minimum value of m --/
theorem cherry_pricing_and_profit (data : CherryData) (sales : SalesData) :
  data.yellow_cost = 6000 ∧
  data.red_cost = 1000 ∧
  data.yellow_quantity = data.red_quantity + 100 ∧
  data.yellow_cost / data.yellow_quantity = 2 * (data.red_cost / data.red_quantity) ∧
  sales.week1_price = 40 ∧
  sales.week2_quantity = 20 ∧
  sales.week3_discount = 0.3 →
  data.red_cost / data.red_quantity = 20 ∧
  ∃ m : ℝ,
    m ≥ 5 ∧
    sales.week1_quantity = 3 * m ∧
    sales.week2_price_decrease = 0.5 * m ∧
    (40 - 20) * (3 * m) + 20 * (40 - 0.5 * m - 20) + (40 * 0.7 - 20) * (50 - 3 * m - 20) ≥ 770 ∧
    ∀ m' : ℝ,
      m' < 5 →
      (40 - 20) * (3 * m') + 20 * (40 - 0.5 * m' - 20) + (40 * 0.7 - 20) * (50 - 3 * m' - 20) < 770 :=
by sorry

end NUMINAMATH_CALUDE_cherry_pricing_and_profit_l833_83355


namespace NUMINAMATH_CALUDE_janessa_cards_ordered_l833_83373

/-- The number of cards Janessa ordered from eBay --/
def cards_ordered (initial_cards : ℕ) (father_cards : ℕ) (thrown_cards : ℕ) (given_cards : ℕ) (kept_cards : ℕ) : ℕ :=
  given_cards + kept_cards - (initial_cards + father_cards) + thrown_cards

theorem janessa_cards_ordered :
  cards_ordered 4 13 4 29 20 = 36 := by
  sorry

end NUMINAMATH_CALUDE_janessa_cards_ordered_l833_83373


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l833_83307

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 6) ((Nat.factorial 9) / (Nat.factorial 4)) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l833_83307


namespace NUMINAMATH_CALUDE_range_of_x_l833_83311

theorem range_of_x (x : ℝ) : 2 * x + 1 ≤ 0 → x ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l833_83311


namespace NUMINAMATH_CALUDE_project_completion_time_l833_83399

theorem project_completion_time (a_time b_time total_time : ℕ) 
  (h1 : a_time = 20)
  (h2 : b_time = 20)
  (h3 : total_time = 15) :
  ∃ (x : ℕ), 
    (1 : ℚ) / a_time + (1 : ℚ) / b_time = (1 : ℚ) / (total_time - x) + 
    ((1 : ℚ) / b_time) * (x : ℚ) / total_time ∧ 
    x = 10 := by
  sorry

#check project_completion_time

end NUMINAMATH_CALUDE_project_completion_time_l833_83399


namespace NUMINAMATH_CALUDE_cells_after_three_divisions_l833_83354

/-- The number of cells after n divisions, given that each division doubles the number of cells -/
def num_cells (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of cells after 3 divisions is 8 -/
theorem cells_after_three_divisions : num_cells 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cells_after_three_divisions_l833_83354


namespace NUMINAMATH_CALUDE_twice_gcf_equals_180_l833_83343

def a : ℕ := 180
def b : ℕ := 270
def c : ℕ := 450

theorem twice_gcf_equals_180 : 2 * Nat.gcd a (Nat.gcd b c) = 180 := by
  sorry

end NUMINAMATH_CALUDE_twice_gcf_equals_180_l833_83343


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l833_83365

theorem solution_satisfies_equations :
  let x : ℚ := -5/7
  let y : ℚ := -18/7
  (6 * x + 3 * y = -12) ∧ (4 * x = 5 * y + 10) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l833_83365


namespace NUMINAMATH_CALUDE_marias_reading_capacity_l833_83332

/-- Given Maria's reading speed and available time, prove how many complete books she can read --/
theorem marias_reading_capacity (pages_per_hour : ℕ) (book_pages : ℕ) (available_hours : ℕ) : 
  pages_per_hour = 120 → book_pages = 360 → available_hours = 8 → 
  (available_hours * pages_per_hour) / book_pages = 2 := by
  sorry

#check marias_reading_capacity

end NUMINAMATH_CALUDE_marias_reading_capacity_l833_83332


namespace NUMINAMATH_CALUDE_earrings_sold_count_l833_83378

def necklace_price : ℕ := 25
def bracelet_price : ℕ := 15
def earrings_price : ℕ := 10
def ensemble_price : ℕ := 45

def necklaces_sold : ℕ := 5
def bracelets_sold : ℕ := 10
def ensembles_sold : ℕ := 2

def total_sales : ℕ := 565

theorem earrings_sold_count :
  ∃ (x : ℕ), 
    necklace_price * necklaces_sold + 
    bracelet_price * bracelets_sold + 
    earrings_price * x + 
    ensemble_price * ensembles_sold = total_sales ∧
    x = 20 := by sorry

end NUMINAMATH_CALUDE_earrings_sold_count_l833_83378


namespace NUMINAMATH_CALUDE_cakes_served_today_l833_83312

theorem cakes_served_today (dinner_stock : ℕ) (lunch_served : ℚ) (dinner_percentage : ℚ) :
  dinner_stock = 95 →
  lunch_served = 48.5 →
  dinner_percentage = 62.25 →
  ⌈lunch_served + (dinner_percentage / 100) * dinner_stock⌉ = 108 :=
by sorry

end NUMINAMATH_CALUDE_cakes_served_today_l833_83312


namespace NUMINAMATH_CALUDE_haunted_castle_windows_l833_83308

theorem haunted_castle_windows (n : ℕ) (h : n = 10) : 
  n * (n - 1) * (n - 2) * (n - 3) = 5040 :=
sorry

end NUMINAMATH_CALUDE_haunted_castle_windows_l833_83308


namespace NUMINAMATH_CALUDE_calvin_insect_collection_l833_83362

/-- Calculates the total number of insects in Calvin's collection --/
def total_insects (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := 2 * scorpions
  let beetles := 4 * crickets
  let other_insects := roaches + scorpions + crickets + caterpillars + beetles
  let exotic_insects := 3 * other_insects
  other_insects + exotic_insects

/-- Theorem stating that Calvin has 204 insects in his collection --/
theorem calvin_insect_collection : total_insects 12 3 = 204 := by
  sorry

end NUMINAMATH_CALUDE_calvin_insect_collection_l833_83362


namespace NUMINAMATH_CALUDE_cubic_expression_value_l833_83322

theorem cubic_expression_value (x : ℝ) (h : x^2 + x - 1 = 0) : x^3 + 2*x^2 - 7 = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l833_83322


namespace NUMINAMATH_CALUDE_angle_A_measure_l833_83335

theorem angle_A_measure (A B C : ℝ) (a b c : ℝ) : 
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  A = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_l833_83335


namespace NUMINAMATH_CALUDE_system_solutions_l833_83379

-- Define the system of equations
def system (t y a : ℝ) : Prop :=
  (|t| - y = 1 - a^4 - a^4 * t^4) ∧ (t^2 + y^2 = 1)

-- Define the property of having multiple solutions
def has_multiple_solutions (a : ℝ) : Prop :=
  ∃ (t₁ y₁ t₂ y₂ : ℝ), t₁ ≠ t₂ ∧ system t₁ y₁ a ∧ system t₂ y₂ a

-- Define the property of having a unique solution
def has_unique_solution (a : ℝ) : Prop :=
  ∀ (t y : ℝ), system t y a → t = 0 ∧ y = 1

-- Theorem statement
theorem system_solutions :
  (has_multiple_solutions 0) ∧
  (has_unique_solution (Real.sqrt (Real.sqrt 2))) :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l833_83379


namespace NUMINAMATH_CALUDE_e_value_is_negative_72_l833_83353

/-- A cubic polynomial with specific properties -/
structure SpecialCubicPolynomial where
  d : ℝ
  e : ℝ
  mean_zeros : ℝ
  product_zeros : ℝ
  sum_coefficients : ℝ
  h1 : mean_zeros = 2 * product_zeros
  h2 : mean_zeros = sum_coefficients
  h3 : sum_coefficients = 3 + d + e + 9

/-- The value of e in the special cubic polynomial -/
def find_e (p : SpecialCubicPolynomial) : ℝ := -72

/-- Theorem stating that the value of e is -72 for the given polynomial -/
theorem e_value_is_negative_72 (p : SpecialCubicPolynomial) : find_e p = -72 := by
  sorry

end NUMINAMATH_CALUDE_e_value_is_negative_72_l833_83353


namespace NUMINAMATH_CALUDE_bread_leftover_l833_83382

theorem bread_leftover (total_length : Real) (jimin_eats_cm : Real) (taehyung_eats_m : Real) :
  total_length = 30 ∧ jimin_eats_cm = 150 ∧ taehyung_eats_m = 1.65 →
  total_length - (jimin_eats_cm / 100 + taehyung_eats_m) = 26.85 := by
  sorry

end NUMINAMATH_CALUDE_bread_leftover_l833_83382


namespace NUMINAMATH_CALUDE_exam_success_probability_l833_83342

theorem exam_success_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/2) (h2 : p2 = 1/4) (h3 : p3 = 1/5) :
  let at_least_two_success := 
    p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3 + (1 - p1) * p2 * p3 + p1 * p2 * p3
  at_least_two_success = 9/40 := by
  sorry

end NUMINAMATH_CALUDE_exam_success_probability_l833_83342


namespace NUMINAMATH_CALUDE_second_chapter_length_l833_83341

theorem second_chapter_length (total_pages first_chapter_pages : ℕ) 
  (h1 : total_pages = 94)
  (h2 : first_chapter_pages = 48) :
  total_pages - first_chapter_pages = 46 := by
  sorry

end NUMINAMATH_CALUDE_second_chapter_length_l833_83341


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_nine_l833_83314

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The set of all possible sums when rolling two dice -/
def possible_sums : Set ℕ := {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

/-- The set of sums greater than 9 -/
def sums_greater_than_nine : Set ℕ := {10, 11, 12}

/-- The number of favorable outcomes (sums greater than 9) -/
def favorable_outcomes : ℕ := 6

/-- Theorem: The probability of rolling two dice and getting a sum greater than 9 is 1/6 -/
theorem probability_sum_greater_than_nine :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_nine_l833_83314
