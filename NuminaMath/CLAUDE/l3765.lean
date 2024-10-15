import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_system_l3765_376530

theorem unique_solution_system (x y : ℝ) : 
  (x + 2*y = 4 ∧ 2*x - y = 3) ↔ (x = 2 ∧ y = 1) := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3765_376530


namespace NUMINAMATH_CALUDE_sin_eq_sin_sin_unique_solution_l3765_376552

noncomputable def arcsin_099 : ℝ := Real.arcsin 0.99

theorem sin_eq_sin_sin_unique_solution :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ arcsin_099 ∧ Real.sin x = Real.sin (Real.sin x) :=
sorry

end NUMINAMATH_CALUDE_sin_eq_sin_sin_unique_solution_l3765_376552


namespace NUMINAMATH_CALUDE_equal_integers_from_ratio_l3765_376557

theorem equal_integers_from_ratio (a b : ℕ+) 
  (hK : K = Real.sqrt ((a.val ^ 2 + b.val ^ 2) / 2))
  (hA : A = (a.val + b.val) / 2)
  (hKA : ∃ (n : ℕ+), K / A = n.val) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_equal_integers_from_ratio_l3765_376557


namespace NUMINAMATH_CALUDE_inequalities_hold_l3765_376534

theorem inequalities_hold (a b : ℝ) (h : a * b > 0) :
  (2 * (a^2 + b^2) ≥ (a + b)^2) ∧
  (b / a + a / b ≥ 2) ∧
  ((a + 1 / a) * (b + 1 / b) ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l3765_376534


namespace NUMINAMATH_CALUDE_max_cross_section_area_l3765_376572

/-- A right rectangular prism with a square base and varying height -/
structure Prism where
  base_length : ℝ
  height_a : ℝ
  height_b : ℝ
  height_c : ℝ
  height_d : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The cross-section area formed by the intersection of a prism and a plane -/
def cross_section_area (p : Prism) (pl : Plane) : ℝ := sorry

/-- The theorem stating that the maximal area of the cross-section is 110 -/
theorem max_cross_section_area (p : Prism) (pl : Plane) :
  p.base_length = 8 ∧
  p.height_a = 3 ∧ p.height_b = 2 ∧ p.height_c = 4 ∧ p.height_d = 1 ∧
  pl.a = 3 ∧ pl.b = -5 ∧ pl.c = 3 ∧ pl.d = 24 →
  cross_section_area p pl = 110 := by
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l3765_376572


namespace NUMINAMATH_CALUDE_four_inch_cube_three_painted_faces_l3765_376545

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Represents a smaller cube resulting from cutting a larger cube -/
structure SmallCube where
  paintedFaces : ℕ

/-- The number of small cubes with at least three painted faces in a painted cube -/
def numCubesWithThreePaintedFaces (c : Cube) : ℕ :=
  8

/-- Theorem stating that a 4-inch cube cut into 1-inch cubes has 8 cubes with at least three painted faces -/
theorem four_inch_cube_three_painted_faces :
  ∀ (c : Cube), c.sideLength = 4 → numCubesWithThreePaintedFaces c = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_inch_cube_three_painted_faces_l3765_376545


namespace NUMINAMATH_CALUDE_new_person_age_l3765_376599

/-- Given a group of 10 people, prove that if replacing a 44-year-old person
    with a new person decreases the average age by 3 years, then the age of
    the new person is 14 years. -/
theorem new_person_age (group_size : ℕ) (old_person_age : ℕ) (avg_decrease : ℕ) :
  group_size = 10 →
  old_person_age = 44 →
  avg_decrease = 3 →
  ∃ (new_person_age : ℕ),
    (group_size * (avg_decrease + new_person_age) : ℤ) = old_person_age - new_person_age ∧
    new_person_age = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_age_l3765_376599


namespace NUMINAMATH_CALUDE_point_outside_circle_l3765_376586

/-- The line ax + by = 1 intersects with the circle x^2 + y^2 = 1 -/
def line_intersects_circle (a b : ℝ) : Prop :=
  ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1

theorem point_outside_circle (a b : ℝ) :
  line_intersects_circle a b → a^2 + b^2 > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_circle_l3765_376586


namespace NUMINAMATH_CALUDE_least_value_of_x_l3765_376561

theorem least_value_of_x (x p : ℕ) : 
  x > 0 → 
  Nat.Prime p → 
  ∃ q, Nat.Prime q ∧ q % 2 = 1 ∧ x / (9 * p) = q →
  x ≥ 81 :=
sorry

end NUMINAMATH_CALUDE_least_value_of_x_l3765_376561


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3765_376526

theorem absolute_value_inequality (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3765_376526


namespace NUMINAMATH_CALUDE_problem_solution_l3765_376565

/-- The function g(x) = -|x+m| -/
def g (m : ℝ) (x : ℝ) : ℝ := -|x + m|

/-- The function f(x) = 2|x-1| - a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2*|x - 1| - a

theorem problem_solution :
  (∃! (n : ℤ), g m n > -1) ∧ (∀ (x : ℤ), g m x > -1 → x = -3) →
  m = 3 ∧
  (∀ x, f a x > g 3 x) →
  a < 4 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3765_376565


namespace NUMINAMATH_CALUDE_triangle_area_l3765_376548

/-- The area of a triangle with vertices at (3, -3), (8, 4), and (3, 4) is 17.5 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let v1 : (Real × Real) := (3, -3)
  let v2 : (Real × Real) := (8, 4)
  let v3 : (Real × Real) := (3, 4)

  -- Calculate the area of the triangle
  let area : Real := 17.5

  sorry -- The proof is omitted

#check triangle_area

end NUMINAMATH_CALUDE_triangle_area_l3765_376548


namespace NUMINAMATH_CALUDE_inequality_proof_l3765_376524

theorem inequality_proof (x : ℝ) (n : ℕ) (h : x > 0) :
  x + n^n / x^n ≥ n + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3765_376524


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l3765_376560

theorem sequence_is_arithmetic (a : ℕ+ → ℝ)
  (h : ∀ p q : ℕ+, a p = a q + 2003 * (p - q)) :
  ∃ d : ℝ, ∀ n m : ℕ+, a n = a m + d * (n - m) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l3765_376560


namespace NUMINAMATH_CALUDE_eugene_model_house_l3765_376587

/-- The number of toothpicks Eugene uses for each card -/
def toothpicks_per_card : ℕ := 75

/-- The total number of cards in a deck -/
def cards_in_deck : ℕ := 52

/-- The number of cards Eugene didn't use -/
def unused_cards : ℕ := 16

/-- The number of toothpicks in each box -/
def toothpicks_per_box : ℕ := 450

/-- The number of boxes of toothpicks Eugene used -/
def boxes_used : ℕ := 6

theorem eugene_model_house :
  (cards_in_deck - unused_cards) * toothpicks_per_card / toothpicks_per_box = boxes_used :=
sorry

end NUMINAMATH_CALUDE_eugene_model_house_l3765_376587


namespace NUMINAMATH_CALUDE_scientific_notation_of_120_million_l3765_376504

theorem scientific_notation_of_120_million : 
  ∃ (a : ℝ) (n : ℤ), 120000000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_120_million_l3765_376504


namespace NUMINAMATH_CALUDE_arccos_negative_half_l3765_376509

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_negative_half_l3765_376509


namespace NUMINAMATH_CALUDE_simplify_expression_l3765_376583

theorem simplify_expression (x : ℝ) : 125 * x - 57 * x = 68 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3765_376583


namespace NUMINAMATH_CALUDE_vampire_daily_victims_l3765_376503

-- Define the vampire's weekly blood requirement in gallons
def weekly_blood_requirement : ℚ := 7

-- Define the amount of blood sucked per person in pints
def blood_per_person : ℚ := 2

-- Define the number of days in a week
def days_per_week : ℕ := 7

-- Define the number of pints in a gallon
def pints_per_gallon : ℕ := 8

-- Theorem: The vampire needs to suck blood from 4 people per day
theorem vampire_daily_victims : 
  (weekly_blood_requirement / days_per_week * pints_per_gallon) / blood_per_person = 4 := by
  sorry


end NUMINAMATH_CALUDE_vampire_daily_victims_l3765_376503


namespace NUMINAMATH_CALUDE_jeff_initial_pencils_l3765_376573

theorem jeff_initial_pencils (J : ℝ) : 
  J > 0 →
  (0.7 * J + 0.25 * (2 * J) = 360) →
  J = 300 := by
sorry

end NUMINAMATH_CALUDE_jeff_initial_pencils_l3765_376573


namespace NUMINAMATH_CALUDE_negPowersOfTwo_is_geometric_l3765_376563

/-- A sequence is geometric if it has a constant ratio between consecutive terms. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence of the form a_n = cq^n (where cq ≠ 0) is geometric. -/
axiom geometric_sequence_criterion (c q : ℝ) (hcq : c * q ≠ 0) :
  IsGeometricSequence (fun n => c * q ^ n)

/-- The sequence {-2^n} -/
def negPowersOfTwo (n : ℕ) : ℝ := -2 ^ n

/-- Theorem: The sequence {-2^n} is a geometric sequence -/
theorem negPowersOfTwo_is_geometric : IsGeometricSequence negPowersOfTwo := by
  sorry

end NUMINAMATH_CALUDE_negPowersOfTwo_is_geometric_l3765_376563


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3765_376597

theorem cubic_polynomial_satisfies_conditions :
  let q : ℝ → ℝ := λ x => -(1/3) * x^3 - x^2 - (2/3) * x - 3
  (q 1 = -5) ∧ (q 2 = -8) ∧ (q 3 = -17) ∧ (q 4 = -34) := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l3765_376597


namespace NUMINAMATH_CALUDE_certain_number_problem_l3765_376584

theorem certain_number_problem :
  ∃ x : ℝ, (1/10 : ℝ) * x - (1/1000 : ℝ) * x = 693 ∧ x = 7000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3765_376584


namespace NUMINAMATH_CALUDE_compute_expression_l3765_376528

theorem compute_expression : 9 * (2/3)^4 = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3765_376528


namespace NUMINAMATH_CALUDE_smallest_max_volume_is_500_l3765_376515

/-- Represents a cuboid with integral side lengths -/
structure Cuboid where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- Calculates the volume of a cuboid -/
def Cuboid.volume (c : Cuboid) : ℕ := c.length.val * c.width.val * c.height.val

/-- Represents the result of cutting a cube into three cuboids -/
structure CubeCut where
  cuboid1 : Cuboid
  cuboid2 : Cuboid
  cuboid3 : Cuboid

/-- Checks if a CubeCut is valid for a cube with side length 10 -/
def isValidCubeCut (cut : CubeCut) : Prop :=
  cut.cuboid1.length + cut.cuboid2.length + cut.cuboid3.length = 10 ∧
  cut.cuboid1.width = 10 ∧ cut.cuboid2.width = 10 ∧ cut.cuboid3.width = 10 ∧
  cut.cuboid1.height = 10 ∧ cut.cuboid2.height = 10 ∧ cut.cuboid3.height = 10

/-- The main theorem to prove -/
theorem smallest_max_volume_is_500 :
  ∀ (cut : CubeCut),
    isValidCubeCut cut →
    max cut.cuboid1.volume (max cut.cuboid2.volume cut.cuboid3.volume) ≥ 500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_volume_is_500_l3765_376515


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l3765_376555

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → b (n + 1) - b n = d

theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n ≥ 1 → a n > 0) →
  a 1 = 2 →
  arithmetic_sequence (λ n => match n with
    | 1 => 2 * a 1
    | 2 => a 3
    | 3 => 3 * a 2
    | _ => 0
  ) →
  ∀ n : ℕ, n ≥ 1 → a n = 2^n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l3765_376555


namespace NUMINAMATH_CALUDE_exactly_two_statements_true_l3765_376539

theorem exactly_two_statements_true :
  let statement1 := (¬∀ x : ℝ, x^2 - 3*x - 2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 - 3*x₀ - 2 ≤ 0)
  let statement2 := ∀ P Q : Prop, (P ∨ Q → P ∧ Q) ∧ ¬(P ∧ Q → P ∨ Q)
  let statement3 := ∃ m : ℝ, ∀ x : ℝ, x > 0 → (
    (∃ α : ℝ, ∀ x : ℝ, x > 0 → m * x^(m^2 + 2*m) = x^α) ∧
    (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → m * x₁^(m^2 + 2*m) < m * x₂^(m^2 + 2*m))
  )
  let statement4 := ∀ a b : ℝ, a ≠ 0 ∧ b ≠ 0 →
    (∀ x y : ℝ, x/a + y/b = 1 ↔ ∃ k : ℝ, k ≠ 0 ∧ x = k*a ∧ y = k*b)
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) :=
by sorry

end NUMINAMATH_CALUDE_exactly_two_statements_true_l3765_376539


namespace NUMINAMATH_CALUDE_bean_ratio_l3765_376592

/-- Given a jar of beans with the following properties:
  - There are 572 beans in total
  - One-fourth of the beans are red
  - Half of the remaining beans after removing red are green
  - There are 143 green beans
  This theorem proves that the ratio of white beans to the remaining beans
  after removing red beans is 1:2. -/
theorem bean_ratio (total : ℕ) (red : ℕ) (green : ℕ) (white : ℕ) : 
  total = 572 →
  red = total / 4 →
  green = (total - red) / 2 →
  green = 143 →
  white = total - red - green →
  (white : ℚ) / (total - red - green : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bean_ratio_l3765_376592


namespace NUMINAMATH_CALUDE_exterior_angle_decreases_l3765_376536

theorem exterior_angle_decreases (n : ℕ) (h : n > 2) :
  (360 : ℝ) / (n + 1) < 360 / n := by
sorry

end NUMINAMATH_CALUDE_exterior_angle_decreases_l3765_376536


namespace NUMINAMATH_CALUDE_triangle_side_c_l3765_376554

theorem triangle_side_c (a b c : ℝ) (S : ℝ) (B : ℝ) :
  B = π / 4 →  -- 45° in radians
  a = 4 →
  S = 16 * Real.sqrt 2 →
  S = 1 / 2 * a * c * Real.sin B →
  c = 16 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_c_l3765_376554


namespace NUMINAMATH_CALUDE_subcommittee_count_l3765_376574

theorem subcommittee_count (n m k : ℕ) (hn : n = 12) (hm : m = 5) (hk : k = 5) :
  Nat.choose n k - Nat.choose (n - m) k = 771 :=
sorry

end NUMINAMATH_CALUDE_subcommittee_count_l3765_376574


namespace NUMINAMATH_CALUDE_min_omega_for_cosine_function_l3765_376596

theorem min_omega_for_cosine_function (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.cos (ω * x - π / 6)) →
  (ω > 0) →
  (∀ x, f x ≤ f (π / 4)) →
  (∀ ω' > 0, (∀ x, Real.cos (ω' * x - π / 6) ≤ Real.cos (ω' * π / 4 - π / 6)) → ω' ≥ 2 / 3) →
  ω = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_min_omega_for_cosine_function_l3765_376596


namespace NUMINAMATH_CALUDE_equal_volume_equivalent_by_decomposition_l3765_376598

/-- A type representing geometric shapes (either rectangular parallelepipeds or prisms) -/
structure GeometricShape where
  volume : ℝ

/-- A type representing a decomposition of a geometric shape -/
structure Decomposition (α : Type) where
  parts : List α

/-- A function that checks if two decompositions are equivalent -/
def equivalent_decompositions {α : Type} (d1 d2 : Decomposition α) : Prop :=
  sorry

/-- A function that transforms one shape into another using a decomposition -/
def transform (s1 s2 : GeometricShape) (d : Decomposition GeometricShape) : Prop :=
  sorry

/-- The main theorem stating that equal-volume shapes are equivalent by decomposition -/
theorem equal_volume_equivalent_by_decomposition (s1 s2 : GeometricShape) :
  s1.volume = s2.volume →
  ∃ (d : Decomposition GeometricShape), transform s1 s2 d ∧ transform s2 s1 d :=
sorry

end NUMINAMATH_CALUDE_equal_volume_equivalent_by_decomposition_l3765_376598


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l3765_376531

def sumEven (n : ℕ) : ℕ := 
  (n / 2) * (2 + n)

def sumOdd (n : ℕ) : ℕ := 
  (n / 2) * (1 + (n - 1))

theorem even_odd_sum_difference : 
  sumEven 100 - sumOdd 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l3765_376531


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l3765_376566

theorem arithmetic_sequence_sum_mod_15 : 
  let first_term := 1
  let last_term := 101
  let common_diff := 5
  let num_terms := (last_term - first_term) / common_diff + 1
  ∃ (sum : ℕ), sum = (num_terms * (first_term + last_term)) / 2 ∧ sum ≡ 6 [MOD 15] :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_15_l3765_376566


namespace NUMINAMATH_CALUDE_relationship_abc_l3765_376580

theorem relationship_abc (a b c : ℝ) : 
  a = (1.01 : ℝ) ^ (0.5 : ℝ) →
  b = (1.01 : ℝ) ^ (0.6 : ℝ) →
  c = (0.6 : ℝ) ^ (0.5 : ℝ) →
  b > a ∧ a > c :=
by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3765_376580


namespace NUMINAMATH_CALUDE_earliest_year_500_mismatched_l3765_376559

/-- Number of shoe pairs in Moor's room in a given year -/
def shoe_pairs (year : ℕ) : ℕ := 2^(year - 2013)

/-- Number of mismatched shoe pairs possible with a given number of shoe pairs -/
def mismatched_pairs (pairs : ℕ) : ℕ := pairs * (pairs - 1)

/-- Predicate for whether a year allows at least 500 mismatched pairs -/
def can_wear_500_mismatched (year : ℕ) : Prop :=
  mismatched_pairs (shoe_pairs year) ≥ 500

theorem earliest_year_500_mismatched :
  (∀ y < 2018, ¬ can_wear_500_mismatched y) ∧ can_wear_500_mismatched 2018 := by
  sorry

end NUMINAMATH_CALUDE_earliest_year_500_mismatched_l3765_376559


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l3765_376579

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to (a + mb), then m = 5. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ) (m : ℝ) 
  (h1 : a = (2, -1))
  (h2 : b = (1, 3))
  (h3 : a.1 * (a.1 + m * b.1) + a.2 * (a.2 + m * b.2) = 0) :
  m = 5 := by
  sorry

#check perpendicular_vector_scalar

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l3765_376579


namespace NUMINAMATH_CALUDE_cosine_equation_roots_l3765_376521

theorem cosine_equation_roots :
  ∃ (a b c : ℝ), (∀ x : ℝ, 4 * Real.cos (2007 * x) = 2007 * x ↔ x = a ∨ x = b ∨ x = c) ∧
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
sorry

end NUMINAMATH_CALUDE_cosine_equation_roots_l3765_376521


namespace NUMINAMATH_CALUDE_exponent_division_l3765_376501

theorem exponent_division (a : ℝ) : a^12 / a^6 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l3765_376501


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l3765_376588

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l3765_376588


namespace NUMINAMATH_CALUDE_violinists_count_l3765_376576

/-- Represents the number of people playing each instrument in an orchestra -/
structure Orchestra where
  total : ℕ
  drums : ℕ
  trombone : ℕ
  trumpet : ℕ
  frenchHorn : ℕ
  cello : ℕ
  contrabass : ℕ
  clarinet : ℕ
  flute : ℕ
  maestro : ℕ

/-- Calculates the number of violinists in the orchestra -/
def violinists (o : Orchestra) : ℕ :=
  o.total - (o.drums + o.trombone + o.trumpet + o.frenchHorn + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro)

/-- Theorem stating that the number of violinists in the given orchestra is 3 -/
theorem violinists_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.drums = 1)
  (h3 : o.trombone = 4)
  (h4 : o.trumpet = 2)
  (h5 : o.frenchHorn = 1)
  (h6 : o.cello = 1)
  (h7 : o.contrabass = 1)
  (h8 : o.clarinet = 3)
  (h9 : o.flute = 4)
  (h10 : o.maestro = 1) :
  violinists o = 3 := by
  sorry


end NUMINAMATH_CALUDE_violinists_count_l3765_376576


namespace NUMINAMATH_CALUDE_tuesday_kids_l3765_376569

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 24

/-- The difference in the number of kids Julia played with between Monday and Tuesday -/
def difference : ℕ := 18

/-- Theorem: The number of kids Julia played with on Tuesday is 6 -/
theorem tuesday_kids : monday_kids - difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_kids_l3765_376569


namespace NUMINAMATH_CALUDE_bird_sanctuary_theorem_l3765_376591

def bird_sanctuary_problem (initial_storks initial_herons initial_sparrows : ℕ)
  (storks_left herons_left sparrows_arrived hummingbirds_arrived : ℕ) : ℤ :=
  let final_storks : ℕ := initial_storks - storks_left
  let final_herons : ℕ := initial_herons - herons_left
  let final_sparrows : ℕ := initial_sparrows + sparrows_arrived
  let final_hummingbirds : ℕ := hummingbirds_arrived
  let total_other_birds : ℕ := final_herons + final_sparrows + final_hummingbirds
  (final_storks : ℤ) - (total_other_birds : ℤ)

theorem bird_sanctuary_theorem :
  bird_sanctuary_problem 8 4 5 3 2 4 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_bird_sanctuary_theorem_l3765_376591


namespace NUMINAMATH_CALUDE_salary_calculation_l3765_376500

def initial_salary : ℝ := 5000

def final_salary (s : ℝ) : ℝ :=
  let s1 := s * 1.3
  let s2 := s1 * 0.93
  let s3 := s2 * 0.8
  let s4 := s3 - 100
  let s5 := s4 * 1.1
  let s6 := s5 * 0.9
  s6 * 0.75

theorem salary_calculation :
  final_salary initial_salary = 3516.48 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l3765_376500


namespace NUMINAMATH_CALUDE_tenth_even_term_is_92_l3765_376550

def arithmetic_sequence (n : ℕ) : ℤ := 2 + (n - 1) * 5

def is_even (z : ℤ) : Prop := ∃ k : ℤ, z = 2 * k

def nth_even_term (n : ℕ) : ℕ := 2 * n - 1

theorem tenth_even_term_is_92 :
  arithmetic_sequence (nth_even_term 10) = 92 :=
sorry

end NUMINAMATH_CALUDE_tenth_even_term_is_92_l3765_376550


namespace NUMINAMATH_CALUDE_solution_difference_l3765_376508

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3

-- Define the theorem
theorem solution_difference (r s : ℝ) : 
  equation r ∧ equation s ∧ r ≠ s ∧ r > s → r - s = 3 := by
  sorry


end NUMINAMATH_CALUDE_solution_difference_l3765_376508


namespace NUMINAMATH_CALUDE_trapezoid_gh_length_l3765_376568

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  side_ef : ℝ
  side_gh : ℝ

/-- The area of a trapezoid is equal to the average of its parallel sides multiplied by its altitude -/
axiom trapezoid_area (t : Trapezoid) : t.area = (t.side_ef + t.side_gh) / 2 * t.altitude

theorem trapezoid_gh_length (t : Trapezoid) 
    (h_area : t.area = 250)
    (h_altitude : t.altitude = 10)
    (h_ef : t.side_ef = 15) :
    t.side_gh = 35 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_gh_length_l3765_376568


namespace NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l3765_376543

theorem sqrt_x_plus_inverse (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_inverse_l3765_376543


namespace NUMINAMATH_CALUDE_sum_and_equality_condition_l3765_376540

/-- Given three real numbers x, y, and z satisfying the conditions:
    1. x + y + z = 150
    2. (x + 10) = (y - 10) = 3z
    Prove that x = 380/7 -/
theorem sum_and_equality_condition (x y z : ℝ) 
  (sum_eq : x + y + z = 150)
  (equality_cond : (x + 10) = (y - 10) ∧ (x + 10) = 3*z) :
  x = 380/7 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_equality_condition_l3765_376540


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_problem_solution_l3765_376582

theorem arithmetic_series_sum (a₁ d n : ℕ) (h : n > 0) : 
  (n : ℝ) / 2 * (2 * a₁ + (n - 1) * d) = (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) :=
by sorry

theorem problem_solution : 
  let a₁ : ℕ := 9
  let d : ℕ := 4
  let n : ℕ := 50
  (n : ℝ) / 2 * (a₁ + (a₁ + (n - 1) * d)) = 5350 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_problem_solution_l3765_376582


namespace NUMINAMATH_CALUDE_original_triangle_area_l3765_376590

/-- Given a triangle whose dimensions are quintupled to form a new triangle with an area of 100 square feet,
    the area of the original triangle is 4 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 100 → 
  new = original * 25 → 
  original = 4 := by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3765_376590


namespace NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l3765_376556

theorem range_of_a_for_nonempty_solution_set (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ a ∈ Set.Ici 2 ∪ Set.Iic (-6) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_nonempty_solution_set_l3765_376556


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3765_376517

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r and R are real numbers representing radii
  (h_positive : r > 0) -- r is positive
  (h_ratio : π * R^2 = 4 * π * r^2) -- area ratio is 1:4
  : R - r = r := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3765_376517


namespace NUMINAMATH_CALUDE_remainder_problem_l3765_376513

theorem remainder_problem (n : ℤ) : (3 * n) % 7 = 3 → n % 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3765_376513


namespace NUMINAMATH_CALUDE_journey_final_distance_l3765_376518

-- Define the directions
inductive Direction
| NorthEast
| SouthEast
| SouthWest
| NorthWest

-- Define a leg of the journey
structure Leg where
  distance : ℝ
  direction : Direction

-- Define the journey
def journey : List Leg := [
  { distance := 5, direction := Direction.NorthEast },
  { distance := 15, direction := Direction.SouthEast },
  { distance := 25, direction := Direction.SouthWest },
  { distance := 35, direction := Direction.NorthWest },
  { distance := 20, direction := Direction.NorthEast }
]

-- Function to calculate the final distance
def finalDistance (j : List Leg) : ℝ := sorry

-- Theorem stating that the final distance is 20 miles
theorem journey_final_distance : finalDistance journey = 20 := by sorry

end NUMINAMATH_CALUDE_journey_final_distance_l3765_376518


namespace NUMINAMATH_CALUDE_set_operation_equality_l3765_376514

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, 1, 2}

theorem set_operation_equality : A ∪ (U \ B) = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l3765_376514


namespace NUMINAMATH_CALUDE_evaluate_expression_l3765_376542

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^(y-1) + 2 * y^(x+1) = 647 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3765_376542


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3765_376594

-- Define the rhombus
structure Rhombus :=
  (side_length : ℝ)
  (diagonal_length : ℝ)

-- Define the conditions
def satisfies_equation (y : ℝ) : Prop :=
  y^2 - 7*y + 10 = 0

def is_valid_rhombus (r : Rhombus) : Prop :=
  r.diagonal_length = 6 ∧ satisfies_equation r.side_length

-- Theorem statement
theorem rhombus_perimeter (r : Rhombus) (h : is_valid_rhombus r) : 
  4 * r.side_length = 20 :=
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3765_376594


namespace NUMINAMATH_CALUDE_no_common_solutions_l3765_376553

theorem no_common_solutions : 
  ¬∃ x : ℝ, (|x - 10| = |x + 3| ∧ 2 * x + 6 = 18) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solutions_l3765_376553


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l3765_376567

/-- A line in the 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def Line.isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) is on a line -/
def Line.containsPoint (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point (x₀ y₀ : ℝ) :
  ∃ (l : Line), l.isParallel ⟨1, -2, -2⟩ ∧ l.containsPoint 1 0 ∧ l = ⟨1, -2, -1⟩ := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l3765_376567


namespace NUMINAMATH_CALUDE_mix_alcohol_solutions_l3765_376551

/-- Represents an alcohol solution with a given volume and concentration -/
structure AlcoholSolution where
  volume : ℝ
  concentration : ℝ

/-- Proves that mixing two alcohol solutions results in the desired solution -/
theorem mix_alcohol_solutions
  (solution_a : AlcoholSolution)
  (solution_b : AlcoholSolution)
  (mixed_solution : AlcoholSolution)
  (h1 : solution_a.volume = 10.5)
  (h2 : solution_a.concentration = 0.75)
  (h3 : solution_b.volume = 7.5)
  (h4 : solution_b.concentration = 0.15)
  (h5 : mixed_solution.volume = 18)
  (h6 : mixed_solution.concentration = 0.5)
  : solution_a.volume * solution_a.concentration + solution_b.volume * solution_b.concentration
    = mixed_solution.volume * mixed_solution.concentration :=
by
  sorry

#check mix_alcohol_solutions

end NUMINAMATH_CALUDE_mix_alcohol_solutions_l3765_376551


namespace NUMINAMATH_CALUDE_angle_in_second_quadrant_l3765_376507

/-- Given an angle α in the second quadrant with P(x,4) on its terminal side and cos α = (1/5)x,
    prove that x = -3 and tan α = -4/3 -/
theorem angle_in_second_quadrant (α : Real) (x : Real) 
    (h1 : π / 2 < α ∧ α < π) -- α is in the second quadrant
    (h2 : x < 0) -- P(x,4) is on the terminal side in the second quadrant
    (h3 : Real.cos α = (1/5) * x) -- Given condition
    : x = -3 ∧ Real.tan α = -4/3 := by
  sorry


end NUMINAMATH_CALUDE_angle_in_second_quadrant_l3765_376507


namespace NUMINAMATH_CALUDE_trains_passing_time_l3765_376527

/-- Given two trains with specified characteristics, prove that they will completely pass each other in 11 seconds. -/
theorem trains_passing_time (tunnel_length : ℝ) (tunnel_time : ℝ) (bridge_length : ℝ) (bridge_time : ℝ)
  (freight_train_length : ℝ) (freight_train_speed : ℝ) :
  tunnel_length = 285 →
  tunnel_time = 24 →
  bridge_length = 245 →
  bridge_time = 22 →
  freight_train_length = 135 →
  freight_train_speed = 10 →
  ∃ (train_speed : ℝ) (train_length : ℝ),
    train_speed = (tunnel_length - bridge_length) / (tunnel_time - bridge_time) ∧
    train_length = train_speed * tunnel_time - tunnel_length ∧
    (train_length + freight_train_length) / (train_speed + freight_train_speed) = 11 :=
by sorry

end NUMINAMATH_CALUDE_trains_passing_time_l3765_376527


namespace NUMINAMATH_CALUDE_girls_boys_difference_l3765_376558

theorem girls_boys_difference (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 36 → 
  5 * boys = 4 * girls → 
  total = girls + boys → 
  girls - boys = 4 := by
sorry

end NUMINAMATH_CALUDE_girls_boys_difference_l3765_376558


namespace NUMINAMATH_CALUDE_quadratic_inequality_bc_value_l3765_376529

theorem quadratic_inequality_bc_value 
  (b c : ℝ) 
  (h : ∀ x : ℝ, x^2 + b*x + c < 0 ↔ 2 < x ∧ x < 4) : 
  b * c = -48 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bc_value_l3765_376529


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3765_376541

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the subset relation for a line being contained in a plane
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

theorem sufficient_but_not_necessary 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : subset m β) :
  (∃ (h : parallel α β), ∀ (l m : Line), perpendicular l α → subset m β → perpendicularLines l m) ∧
  (∃ (l m : Line) (α β : Plane), perpendicular l α ∧ subset m β ∧ perpendicularLines l m ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3765_376541


namespace NUMINAMATH_CALUDE_equation_solutions_l3765_376589

theorem equation_solutions : 
  (∃ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧ 
  (∃ x : ℝ, (x + 3)^2 = (2*x - 1)*(x + 3) ↔ x = -3 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3765_376589


namespace NUMINAMATH_CALUDE_max_gold_coins_l3765_376570

theorem max_gold_coins (n : ℕ) : n < 150 ∧ ∃ k : ℕ, n = 13 * k + 3 → n ≤ 146 :=
by
  sorry

end NUMINAMATH_CALUDE_max_gold_coins_l3765_376570


namespace NUMINAMATH_CALUDE_customers_after_family_l3765_376510

/-- Represents the taco truck's sales during lunch rush -/
def taco_truck_sales (soft_taco_price hard_taco_price : ℕ) 
  (family_hard_tacos family_soft_tacos : ℕ)
  (other_customers : ℕ) (total_revenue : ℕ) : Prop :=
  let family_revenue := family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price
  let other_revenue := other_customers * 2 * soft_taco_price
  family_revenue + other_revenue = total_revenue

/-- Theorem stating the number of customers after the family -/
theorem customers_after_family : 
  taco_truck_sales 2 5 4 3 10 66 := by sorry

end NUMINAMATH_CALUDE_customers_after_family_l3765_376510


namespace NUMINAMATH_CALUDE_missing_carton_dimension_l3765_376512

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the carton with one unknown dimension -/
def carton (x : ℝ) : BoxDimensions :=
  { length := 25, width := x, height := 60 }

/-- Represents the soap box dimensions -/
def soapBox : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- The maximum number of soap boxes that can fit in the carton -/
def maxSoapBoxes : ℕ := 300

theorem missing_carton_dimension :
  ∃ x : ℝ, boxVolume (carton x) = (maxSoapBoxes : ℝ) * boxVolume soapBox ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_missing_carton_dimension_l3765_376512


namespace NUMINAMATH_CALUDE_range_of_m_when_S_true_range_of_m_when_p_or_q_and_not_q_l3765_376562

-- Define the propositions
def p (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ x y : ℝ, x^2 / (4 - m) + y^2 / m = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- State the theorems
theorem range_of_m_when_S_true :
  ∀ m : ℝ, S m → m < 0 ∨ m ≥ 1 :=
sorry

theorem range_of_m_when_p_or_q_and_not_q :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_when_S_true_range_of_m_when_p_or_q_and_not_q_l3765_376562


namespace NUMINAMATH_CALUDE_vet_donation_is_78_l3765_376547

/-- Represents the vet fees and adoption numbers for different animal types -/
structure AnimalAdoption where
  dog_fee : ℕ
  cat_fee : ℕ
  rabbit_fee : ℕ
  parrot_fee : ℕ
  dog_adoptions : ℕ
  cat_adoptions : ℕ
  rabbit_adoptions : ℕ
  parrot_adoptions : ℕ

/-- Calculates the total vet fees collected -/
def total_fees (a : AnimalAdoption) : ℕ :=
  a.dog_fee * a.dog_adoptions +
  a.cat_fee * a.cat_adoptions +
  a.rabbit_fee * a.rabbit_adoptions +
  a.parrot_fee * a.parrot_adoptions

/-- Calculates the amount donated by the vet -/
def vet_donation (a : AnimalAdoption) : ℕ :=
  (total_fees a + 1) / 3

/-- Theorem stating that the vet's donation is $78 given the specified conditions -/
theorem vet_donation_is_78 (a : AnimalAdoption) 
  (h1 : a.dog_fee = 15)
  (h2 : a.cat_fee = 13)
  (h3 : a.rabbit_fee = 10)
  (h4 : a.parrot_fee = 12)
  (h5 : a.dog_adoptions = 8)
  (h6 : a.cat_adoptions = 3)
  (h7 : a.rabbit_adoptions = 5)
  (h8 : a.parrot_adoptions = 2) :
  vet_donation a = 78 := by
  sorry


end NUMINAMATH_CALUDE_vet_donation_is_78_l3765_376547


namespace NUMINAMATH_CALUDE_sequence_inequality_l3765_376581

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a (n + 1) = 0)
  (h : ∀ k : ℕ, k ≥ 1 → k ≤ n → |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → |a k| ≤ k * (n + 1 - k) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3765_376581


namespace NUMINAMATH_CALUDE_gold_coins_in_urn_l3765_376525

-- Define the total percentage
def total_percentage : ℝ := 100

-- Define the percentage of beads
def bead_percentage : ℝ := 30

-- Define the percentage of silver coins among all coins
def silver_coin_percentage : ℝ := 50

-- Define the percentage of coins
def coin_percentage : ℝ := total_percentage - bead_percentage

-- Define the percentage of gold coins among all coins
def gold_coin_percentage : ℝ := total_percentage - silver_coin_percentage

-- Theorem to prove
theorem gold_coins_in_urn : 
  (coin_percentage * gold_coin_percentage) / total_percentage = 35 := by
  sorry

end NUMINAMATH_CALUDE_gold_coins_in_urn_l3765_376525


namespace NUMINAMATH_CALUDE_triangle_type_l3765_376564

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B = Real.pi / 6 ∧  -- 30 degrees in radians
  t.c = 15 ∧
  t.b = 5 * Real.sqrt 3

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Define right triangle
def is_right (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_type (t : Triangle) :
  triangle_conditions t → (is_isosceles t ∨ is_right t) :=
sorry

end NUMINAMATH_CALUDE_triangle_type_l3765_376564


namespace NUMINAMATH_CALUDE_wax_remaining_l3765_376533

/-- The amount of wax remaining after detailing vehicles -/
def remaining_wax (initial : ℕ) (spilled : ℕ) (car : ℕ) (suv : ℕ) : ℕ :=
  initial - spilled - car - suv

/-- Theorem stating the remaining wax after detailing vehicles -/
theorem wax_remaining :
  remaining_wax 11 2 3 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_wax_remaining_l3765_376533


namespace NUMINAMATH_CALUDE_sebastians_orchestra_size_l3765_376520

/-- Represents the composition of an orchestra --/
structure Orchestra where
  percussion : Nat
  trombone : Nat
  trumpet : Nat
  french_horn : Nat
  violin : Nat
  cello : Nat
  contrabass : Nat
  clarinet : Nat
  flute : Nat
  maestro : Nat

/-- The total number of people in the orchestra --/
def Orchestra.total (o : Orchestra) : Nat :=
  o.percussion + o.trombone + o.trumpet + o.french_horn +
  o.violin + o.cello + o.contrabass +
  o.clarinet + o.flute + o.maestro

/-- The specific orchestra composition from the problem --/
def sebastians_orchestra : Orchestra :=
  { percussion := 1
  , trombone := 4
  , trumpet := 2
  , french_horn := 1
  , violin := 3
  , cello := 1
  , contrabass := 1
  , clarinet := 3
  , flute := 4
  , maestro := 1
  }

/-- Theorem stating that the total number of people in Sebastian's orchestra is 21 --/
theorem sebastians_orchestra_size :
  sebastians_orchestra.total = 21 := by
  sorry

end NUMINAMATH_CALUDE_sebastians_orchestra_size_l3765_376520


namespace NUMINAMATH_CALUDE_thousand_pow_seven_div_ten_pow_seventeen_l3765_376544

theorem thousand_pow_seven_div_ten_pow_seventeen :
  (1000 : ℕ)^7 / (10 : ℕ)^17 = (10 : ℕ)^4 := by
  sorry

end NUMINAMATH_CALUDE_thousand_pow_seven_div_ten_pow_seventeen_l3765_376544


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3765_376575

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * (1 + y)) = f x * (1 + f y)

/-- The main theorem stating that any function satisfying the functional equation
    is either the identity function or the zero function -/
theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : SatisfiesFunctionalEquation f) : 
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3765_376575


namespace NUMINAMATH_CALUDE_f_properties_l3765_376511

noncomputable section

def f (x : ℝ) : ℝ := Real.log x - (x - 1)^2 / 2

def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem f_properties :
  (∀ x y, 0 < x ∧ x < y ∧ y < phi → f x < f y) ∧
  (∀ x, 1 < x → f x < x - 1) ∧
  (∀ k, (∃ x₀, 1 < x₀ ∧ ∀ x, 1 < x ∧ x < x₀ → k * (x - 1) < f x) → k < 1) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l3765_376511


namespace NUMINAMATH_CALUDE_major_axis_length_is_13_l3765_376585

/-- A configuration of a cylinder and two spheres -/
structure CylinderSphereConfig where
  cylinder_radius : ℝ
  sphere_radius : ℝ
  sphere_centers_distance : ℝ

/-- The length of the major axis of the ellipse formed by the intersection of a plane 
    touching both spheres and the cylinder surface -/
def major_axis_length (config : CylinderSphereConfig) : ℝ :=
  config.sphere_centers_distance

/-- Theorem stating that for the given configuration, the major axis length is 13 -/
theorem major_axis_length_is_13 :
  let config := CylinderSphereConfig.mk 6 6 13
  major_axis_length config = 13 := by
  sorry

#eval major_axis_length (CylinderSphereConfig.mk 6 6 13)

end NUMINAMATH_CALUDE_major_axis_length_is_13_l3765_376585


namespace NUMINAMATH_CALUDE_third_row_sum_is_226_l3765_376535

/-- Represents a position in the grid -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → Nat

/-- The size of the grid -/
def gridSize : Nat := 13

/-- The starting number -/
def startNum : Nat := 100

/-- The ending number -/
def endNum : Nat := 268

/-- The center position of the grid -/
def centerPos : Position :=
  { row := 6, col := 6 }  -- 0-based index

/-- Generates the spiral grid -/
def generateSpiralGrid : SpiralGrid :=
  sorry

/-- Gets the numbers in the third row -/
def getThirdRowNumbers (grid : SpiralGrid) : List Nat :=
  sorry

/-- Theorem: The sum of the greatest and least numbers in the third row is 226 -/
theorem third_row_sum_is_226 (grid : SpiralGrid) :
  grid = generateSpiralGrid →
  let thirdRowNums := getThirdRowNumbers grid
  (List.maximum thirdRowNums).getD 0 + (List.minimum thirdRowNums).getD 0 = 226 :=
sorry

end NUMINAMATH_CALUDE_third_row_sum_is_226_l3765_376535


namespace NUMINAMATH_CALUDE_central_cell_value_l3765_376577

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_prod : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_prod : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_prod : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l3765_376577


namespace NUMINAMATH_CALUDE_euclidean_division_l3765_376578

theorem euclidean_division (a b : ℕ) (hb : b > 0) :
  ∃ q r : ℤ, 0 ≤ r ∧ r < b ∧ (a : ℤ) = b * q + r :=
sorry

end NUMINAMATH_CALUDE_euclidean_division_l3765_376578


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l3765_376516

theorem rectangle_area_ratio : 
  let length_A : ℝ := 48
  let breadth_A : ℝ := 30
  let length_B : ℝ := 60
  let breadth_B : ℝ := 35
  let area_A := length_A * breadth_A
  let area_B := length_B * breadth_B
  (area_A / area_B) = 24 / 35 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l3765_376516


namespace NUMINAMATH_CALUDE_toms_common_cards_l3765_376522

theorem toms_common_cards (rare_count : ℕ) (uncommon_count : ℕ) (rare_cost : ℚ) (uncommon_cost : ℚ) (common_cost : ℚ) (total_cost : ℚ) :
  rare_count = 19 →
  uncommon_count = 11 →
  rare_cost = 1 →
  uncommon_cost = 1/2 →
  common_cost = 1/4 →
  total_cost = 32 →
  (total_cost - (rare_count * rare_cost + uncommon_count * uncommon_cost)) / common_cost = 30 := by
  sorry

#eval (32 : ℚ) - (19 * 1 + 11 * (1/2 : ℚ)) / (1/4 : ℚ)

end NUMINAMATH_CALUDE_toms_common_cards_l3765_376522


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l3765_376593

theorem three_digit_numbers_count : 
  let digits : Finset Nat := {1, 2, 3, 4, 5}
  (digits.card : Nat) ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l3765_376593


namespace NUMINAMATH_CALUDE_power_sum_simplification_l3765_376506

theorem power_sum_simplification :
  (-1)^2006 - (-1)^2007 + 1^2008 + 1^2009 - 1^2010 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_simplification_l3765_376506


namespace NUMINAMATH_CALUDE_irene_income_l3765_376519

/-- Calculates the total income for a given number of hours worked -/
def total_income (regular_income : ℕ) (overtime_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  let regular_hours := 40
  let overtime_hours := max (hours_worked - regular_hours) 0
  regular_income + overtime_rate * overtime_hours

/-- Irene's income calculation theorem -/
theorem irene_income :
  total_income 500 20 50 = 700 := by
  sorry

#eval total_income 500 20 50

end NUMINAMATH_CALUDE_irene_income_l3765_376519


namespace NUMINAMATH_CALUDE_ratio_and_equation_solution_l3765_376571

theorem ratio_and_equation_solution :
  ∀ (x y z b : ℤ),
  (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  y = 15 * b - 5 →
  (b = 3 → (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) ∧ y = 15 * b - 5) :=
by sorry

end NUMINAMATH_CALUDE_ratio_and_equation_solution_l3765_376571


namespace NUMINAMATH_CALUDE_net_growth_rate_calculation_l3765_376502

/-- Given birth and death rates per certain number of people and an initial population,
    calculate the net growth rate as a percentage. -/
theorem net_growth_rate_calculation 
  (birth_rate death_rate : ℕ) 
  (initial_population : ℕ) 
  (birth_rate_val : birth_rate = 32)
  (death_rate_val : death_rate = 11)
  (initial_population_val : initial_population = 1000) :
  (birth_rate - death_rate : ℝ) / initial_population * 100 = 2.1 := by
  sorry

end NUMINAMATH_CALUDE_net_growth_rate_calculation_l3765_376502


namespace NUMINAMATH_CALUDE_unique_solution_triple_l3765_376549

theorem unique_solution_triple (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1)) →
  x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_triple_l3765_376549


namespace NUMINAMATH_CALUDE_line_inclination_angle_l3765_376523

theorem line_inclination_angle (x y : ℝ) :
  x - Real.sqrt 3 * y + 2 = 0 →
  Real.arctan (1 / Real.sqrt 3) = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l3765_376523


namespace NUMINAMATH_CALUDE_simple_interest_problem_l3765_376538

/-- Given a principal P and an interest rate R, if increasing the rate by 15%
    results in $300 more interest over 10 years, then P must equal $200. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) : 
  (P * (R + 15) * 10 / 100 = P * R * 10 / 100 + 300) → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l3765_376538


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3765_376546

theorem decimal_to_fraction_sum (p q : ℕ+) : 
  (p : ℚ) / q = 504/1000 → 
  (∀ (a b : ℕ+), (a : ℚ) / b = p / q → a ≤ p ∧ b ≤ q) → 
  (p : ℕ) + q = 188 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l3765_376546


namespace NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_l3765_376537

theorem abs_sum_inequality_solution_set :
  {x : ℝ | |x - 1| + |x| < 3} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_solution_set_l3765_376537


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3765_376532

theorem fraction_zero_implies_x_negative_two (x : ℝ) :
  ((x - 1) * (x + 2)) / (x^2 - 1) = 0 → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_two_l3765_376532


namespace NUMINAMATH_CALUDE_smaller_number_is_35_l3765_376595

theorem smaller_number_is_35 (x y : ℝ) : 
  x + y = 77 ∧ 
  (x = 42 ∨ y = 42) ∧ 
  (5 * x = 6 * y ∨ 5 * y = 6 * x) →
  min x y = 35 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_is_35_l3765_376595


namespace NUMINAMATH_CALUDE_rhombus_acute_angle_l3765_376505

-- Define a rhombus
structure Rhombus where
  -- We don't need to define all properties of a rhombus, just what we need
  acute_angle : ℝ

-- Define the plane passing through a side
structure Plane where
  -- The angles it forms with the diagonals
  angle1 : ℝ
  angle2 : ℝ

-- The main theorem
theorem rhombus_acute_angle (r : Rhombus) (p : Plane) 
  (h1 : p.angle1 = α)
  (h2 : p.angle2 = 2 * α)
  : r.acute_angle = 2 * Real.arctan (1 / (2 * Real.cos α)) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_acute_angle_l3765_376505
