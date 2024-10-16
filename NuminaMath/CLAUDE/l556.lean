import Mathlib

namespace NUMINAMATH_CALUDE_minimum_score_proof_l556_55670

theorem minimum_score_proof (C ω : ℕ) (S : ℝ) : 
  S = 30 + 4 * C - ω →
  S > 80 →
  C + ω = 26 →
  ω ≤ 3 →
  ∀ (C' ω' : ℕ) (S' : ℝ), 
    (S' = 30 + 4 * C' - ω' ∧ 
     S' > 80 ∧ 
     C' + ω' = 26 ∧ 
     ω' ≤ 3) → 
    S ≤ S' →
  S = 119 :=
sorry

end NUMINAMATH_CALUDE_minimum_score_proof_l556_55670


namespace NUMINAMATH_CALUDE_m_range_theorem_l556_55601

/-- The equation x^2 + mx + 1 = 0 has two real roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- ∀x ∈ ℝ, 4x^2 + 4(m-2)x + 1 ≠ 0 -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The range of values for m is (1, 2) -/
def range_m : Set ℝ := { m | 1 < m ∧ m < 2 }

theorem m_range_theorem (m : ℝ) :
  (¬(p m ∧ q m)) ∧ (¬¬(q m)) → m ∈ range_m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l556_55601


namespace NUMINAMATH_CALUDE_largest_integral_x_l556_55646

theorem largest_integral_x : ∃ (x : ℤ), 
  (1/4 : ℚ) < (x : ℚ)/6 ∧ (x : ℚ)/6 < 2/3 ∧ 
  x < 10 ∧
  x = 3 ∧
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/6 ∧ (y : ℚ)/6 < 2/3 ∧ y < 10) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integral_x_l556_55646


namespace NUMINAMATH_CALUDE_subset_sum_implies_total_sum_l556_55691

theorem subset_sum_implies_total_sum (a₁ a₂ a₃ : ℝ) :
  (a₁ + a₂ + a₁ + a₃ + a₂ + a₃ + (a₁ + a₂) + (a₁ + a₃) + (a₂ + a₃) = 12) →
  (a₁ + a₂ + a₃ = 4) := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_implies_total_sum_l556_55691


namespace NUMINAMATH_CALUDE_factor_implies_h_value_l556_55653

theorem factor_implies_h_value (h : ℝ) (m : ℝ) : 
  (∃ k : ℝ, m^2 - h*m - 24 = (m - 8) * k) → h = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_h_value_l556_55653


namespace NUMINAMATH_CALUDE_hannahs_age_l556_55632

/-- Given the ages of Eliza, Felipe, Gideon, and Hannah, prove Hannah's age -/
theorem hannahs_age 
  (eliza felipe gideon hannah : ℕ)
  (h1 : eliza = felipe - 4)
  (h2 : felipe = gideon + 6)
  (h3 : hannah = gideon + 2)
  (h4 : eliza = 15) :
  hannah = 15 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_age_l556_55632


namespace NUMINAMATH_CALUDE_min_tests_for_passing_probability_l556_55613

theorem min_tests_for_passing_probability (p : ℝ) (threshold : ℝ) : 
  (p = 3/4) → (threshold = 0.99) → 
  (∀ k : ℕ, k < 4 → 1 - (1 - p)^k ≤ threshold) ∧ 
  (1 - (1 - p)^4 > threshold) := by
sorry

end NUMINAMATH_CALUDE_min_tests_for_passing_probability_l556_55613


namespace NUMINAMATH_CALUDE_store_turnover_equation_l556_55617

/-- Represents the equation for the total turnover in the first quarter of a store,
    given an initial turnover and a monthly growth rate. -/
theorem store_turnover_equation (initial_turnover : ℝ) (growth_rate : ℝ) :
  initial_turnover = 50 →
  initial_turnover * (1 + (1 + growth_rate) + (1 + growth_rate)^2) = 600 :=
by sorry

end NUMINAMATH_CALUDE_store_turnover_equation_l556_55617


namespace NUMINAMATH_CALUDE_binary_multiplication_l556_55622

/-- Converts a list of bits to a natural number -/
def bitsToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- The first binary number 11101₂ -/
def num1 : List Bool := [true, true, true, false, true]

/-- The second binary number 1101₂ -/
def num2 : List Bool := [true, true, false, true]

/-- The expected product 1001101101₂ -/
def expectedProduct : List Bool := [true, false, false, true, true, false, true, true, false, true]

theorem binary_multiplication :
  bitsToNat num1 * bitsToNat num2 = bitsToNat expectedProduct := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_l556_55622


namespace NUMINAMATH_CALUDE_rhombus_sides_not_equal_l556_55644

/-- Proves that for a rhombus with given diagonal lengths and perimeter, the sides are not all equal -/
theorem rhombus_sides_not_equal (d1 d2 p : ℝ) (h1 : d1 = 30) (h2 : d2 = 18) (h3 : p = 80) : 
  p ≠ 4 * Real.sqrt (d1^2/4 + d2^2/4) :=
by
  sorry

#check rhombus_sides_not_equal

end NUMINAMATH_CALUDE_rhombus_sides_not_equal_l556_55644


namespace NUMINAMATH_CALUDE_jake_balloons_l556_55637

def total_balloons : ℕ := 3
def allan_balloons : ℕ := 2

theorem jake_balloons : total_balloons - allan_balloons = 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_balloons_l556_55637


namespace NUMINAMATH_CALUDE_locus_and_tangent_l556_55625

-- Define the points and lines
def A : ℝ × ℝ := (1, 0)
def B : ℝ → ℝ × ℝ := λ y ↦ (-1, y)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the locus E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

theorem locus_and_tangent :
  (∀ y : ℝ, ∃ M : ℝ × ℝ, 
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - (B y).1)^2 + (M.2 - (B y).2)^2 ∧
    M ∈ E) ∧
  (P ∈ E ∧ tangent_line ∩ E = {P}) := by sorry

end NUMINAMATH_CALUDE_locus_and_tangent_l556_55625


namespace NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l556_55669

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Defines a similarity transformation with a given ratio -/
def similarityTransform (p : Point) (ratio : ℝ) : Set Point :=
  { p' : Point | p'.x = p.x * ratio ∨ p'.x = p.x * (-ratio) ∧ 
                  p'.y = p.y * ratio ∨ p'.y = p.y * (-ratio) }

theorem similar_triangle_point_coordinates 
  (ABC : Triangle) 
  (C : Point) 
  (h1 : C = ABC.C) 
  (h2 : C.x = 4 ∧ C.y = 1) 
  (ratio : ℝ) 
  (h3 : ratio = 3) :
  ∃ (C' : Point), C' ∈ similarityTransform C ratio ∧ 
    ((C'.x = 12 ∧ C'.y = 3) ∨ (C'.x = -12 ∧ C'.y = -3)) :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_point_coordinates_l556_55669


namespace NUMINAMATH_CALUDE_hainan_scientific_notation_l556_55697

theorem hainan_scientific_notation :
  48500000 = 4.85 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_hainan_scientific_notation_l556_55697


namespace NUMINAMATH_CALUDE_cube_root_of_3375_l556_55694

theorem cube_root_of_3375 (x : ℝ) (h1 : x > 0) (h2 : x^3 = 3375) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_3375_l556_55694


namespace NUMINAMATH_CALUDE_max_value_expression_l556_55695

theorem max_value_expression (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_eq_3 : x + y + z = 3)
  (x_ge_y : x ≥ y) (y_ge_z : y ≥ z) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l556_55695


namespace NUMINAMATH_CALUDE_fraction_sum_l556_55611

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l556_55611


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_three_l556_55629

theorem absolute_value_sqrt_three : 
  |1 - Real.sqrt 3| - (Real.sqrt 3 - 1)^0 = Real.sqrt 3 - 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_three_l556_55629


namespace NUMINAMATH_CALUDE_set_operation_equality_l556_55690

theorem set_operation_equality (M N P : Set ℕ) : 
  M = {1, 2, 3} → N = {2, 3, 4} → P = {3, 5} → 
  (M ∩ N) ∪ P = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_equality_l556_55690


namespace NUMINAMATH_CALUDE_volumes_and_cross_sections_l556_55633

/-- Represents a geometric body -/
structure GeometricBody where
  volume : ℝ
  crossSectionArea : ℝ → ℝ  -- Function mapping height to cross-sectional area

/-- Zu Chongzhi's principle -/
axiom zu_chongzhi_principle (A B : GeometricBody) :
  (∀ h : ℝ, A.crossSectionArea h = B.crossSectionArea h) → A.volume = B.volume

/-- The main theorem to prove -/
theorem volumes_and_cross_sections (A B : GeometricBody) :
  (A.volume ≠ B.volume → ∃ h : ℝ, A.crossSectionArea h ≠ B.crossSectionArea h) ∧
  ∃ C D : GeometricBody, C.volume = D.volume ∧ ∃ h : ℝ, C.crossSectionArea h ≠ D.crossSectionArea h :=
sorry

end NUMINAMATH_CALUDE_volumes_and_cross_sections_l556_55633


namespace NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l556_55603

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The percentage of mixture X in the final blend -/
def x_percentage : ℝ := 13.333333333333332

/-- The percentage of ryegrass in the final blend -/
def final_ryegrass_percentage : ℝ := 27

/-- Seed mixture X -/
def mixture_x : SeedMixture where
  ryegrass := 40  -- This is what we want to prove
  bluegrass := 60
  fescue := 0

/-- Seed mixture Y -/
def mixture_y : SeedMixture where
  ryegrass := 25
  bluegrass := 0
  fescue := 75

theorem ryegrass_percentage_in_x : 
  (mixture_x.ryegrass * x_percentage + mixture_y.ryegrass * (100 - x_percentage)) / 100 = final_ryegrass_percentage := by
  sorry

#check ryegrass_percentage_in_x

end NUMINAMATH_CALUDE_ryegrass_percentage_in_x_l556_55603


namespace NUMINAMATH_CALUDE_archer_weekly_expenditure_is_1056_l556_55614

/-- The archer's weekly expenditure on arrows -/
def archer_weekly_expenditure (shots_per_day : ℕ) (days_per_week : ℕ) 
  (recovery_rate : ℚ) (arrow_cost : ℚ) (team_contribution_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let recovered_arrows := (total_shots : ℚ) * recovery_rate
  let arrows_used := (total_shots : ℚ) - recovered_arrows
  let total_cost := arrows_used * arrow_cost
  let team_contribution := total_cost * team_contribution_rate
  total_cost - team_contribution

/-- Theorem stating the archer's weekly expenditure on arrows -/
theorem archer_weekly_expenditure_is_1056 :
  archer_weekly_expenditure 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end NUMINAMATH_CALUDE_archer_weekly_expenditure_is_1056_l556_55614


namespace NUMINAMATH_CALUDE_class_composition_l556_55624

theorem class_composition (total_students : ℕ) (girls_ratio boys_ratio : ℕ) 
  (h1 : total_students = 56)
  (h2 : girls_ratio = 4)
  (h3 : boys_ratio = 3) :
  ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧ 
    girls * boys_ratio = boys * girls_ratio ∧
    girls = 32 ∧ 
    boys = 24 :=
by sorry

end NUMINAMATH_CALUDE_class_composition_l556_55624


namespace NUMINAMATH_CALUDE_torturie_problem_l556_55678

/-- The number of the last remaining prisoner in the Torturie problem -/
def lastPrisoner (n : ℕ) : ℕ :=
  2 * n - 2^(Nat.log2 n + 1) + 1

/-- The Torturie problem statement -/
theorem torturie_problem (n : ℕ) (h : n > 0) :
  lastPrisoner n = 
    let k := Nat.log2 n
    2 * n - 2^(k + 1) + 1 :=
by sorry

end NUMINAMATH_CALUDE_torturie_problem_l556_55678


namespace NUMINAMATH_CALUDE_alf3_weight_calculation_l556_55688

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_F : ℝ := 19.00

-- Define the number of moles
def num_moles : ℝ := 7

-- Define the molecular weight calculation function
def molecular_weight (al_weight f_weight : ℝ) : ℝ :=
  al_weight + 3 * f_weight

-- Define the total weight calculation function
def total_weight (mol_weight num_mol : ℝ) : ℝ :=
  mol_weight * num_mol

-- Theorem statement
theorem alf3_weight_calculation :
  total_weight (molecular_weight atomic_weight_Al atomic_weight_F) num_moles = 587.86 := by
  sorry


end NUMINAMATH_CALUDE_alf3_weight_calculation_l556_55688


namespace NUMINAMATH_CALUDE_wildcats_panthers_score_difference_l556_55674

/-- The score difference between two teams -/
def scoreDifference (team1Score team2Score : ℕ) : ℕ :=
  team1Score - team2Score

/-- Theorem: The Wildcats scored 19 more points than the Panthers -/
theorem wildcats_panthers_score_difference :
  scoreDifference 36 17 = 19 := by
  sorry

end NUMINAMATH_CALUDE_wildcats_panthers_score_difference_l556_55674


namespace NUMINAMATH_CALUDE_sun_division_l556_55610

theorem sun_division (x y z total : ℝ) : 
  (y = 0.45 * x) →
  (z = 0.3 * x) →
  (y = 36) →
  (total = x + y + z) →
  total = 140 := by
sorry

end NUMINAMATH_CALUDE_sun_division_l556_55610


namespace NUMINAMATH_CALUDE_division_value_proof_l556_55654

theorem division_value_proof (x : ℝ) : (5.5 / x) * 12 = 11 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_proof_l556_55654


namespace NUMINAMATH_CALUDE_sum_of_A_coordinates_l556_55672

/-- Given three points A, B, and C in a plane, where C divides AB in the ratio 1:2,
    and given the coordinates of B and C, prove that the sum of A's coordinates is 16. -/
theorem sum_of_A_coordinates (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (C.2 - A.2) / (B.2 - A.2) = 1/3 →
  B = (2, 5) →
  C = (5, 8) →
  A.1 + A.2 = 16 := by
sorry


end NUMINAMATH_CALUDE_sum_of_A_coordinates_l556_55672


namespace NUMINAMATH_CALUDE_binary_multiplication_addition_l556_55635

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_multiplication_addition :
  let a := [true, false, false, true, true]  -- 11001₂
  let b := [false, true, true]               -- 110₂
  let c := [false, true, false, true]        -- 1010₂
  let result := [false, true, true, true, true, true, false, true]  -- 10111110₂
  (binary_to_nat a * binary_to_nat b + binary_to_nat c) = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_addition_l556_55635


namespace NUMINAMATH_CALUDE_smallest_m_for_identical_digits_l556_55605

theorem smallest_m_for_identical_digits : ∃ (n : ℕ), 
  (∀ (m : ℕ), m < 671 → 
    ¬∃ (k : ℕ), (2015^(3*m+1) - 2015^(6*k+2)) % 10^2014 = 0 ∧ 2015^(3*m+1) < 2015^(6*k+2)) ∧
  ∃ (n : ℕ), (2015^(3*671+1) - 2015^(6*n+2)) % 10^2014 = 0 ∧ 2015^(3*671+1) < 2015^(6*n+2) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_identical_digits_l556_55605


namespace NUMINAMATH_CALUDE_football_team_progress_l556_55628

theorem football_team_progress (yards_lost yards_gained : ℤ) : 
  yards_lost = 5 → yards_gained = 9 → yards_gained - yards_lost = 4 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l556_55628


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l556_55626

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two sides
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define a point on a line segment
def pointOnSegment (p1 p2 : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := sorry

-- Define the intersection of two lines
def lineIntersection (l1p1 l1p2 l2p1 l2p2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the median of a triangle
def median (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem triangle_ratio_theorem (t : Triangle) :
  area t = 2 * Real.sqrt 3 →
  sideLength t.B t.C = 1 →
  angle t.B t.C t.A = π / 3 →
  let D := pointOnSegment t.A t.B 3
  let E := median t t.C
  let M := lineIntersection t.C D t.B E
  ∃ (r : ℝ), r = 3 / 5 ∧ sideLength t.B M = r * sideLength M E :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l556_55626


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_for_inequality_l556_55675

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part I
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem range_of_a_for_inequality :
  (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_for_inequality_l556_55675


namespace NUMINAMATH_CALUDE_negative_division_subtraction_l556_55685

theorem negative_division_subtraction : (-96) / (-24) - 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_subtraction_l556_55685


namespace NUMINAMATH_CALUDE_grandmother_age_is_132_l556_55651

-- Define the ages as natural numbers
def mason_age : ℕ := 20
def sydney_age : ℕ := 3 * mason_age
def father_age : ℕ := sydney_age + 6
def grandmother_age : ℕ := 2 * father_age

-- Theorem to prove
theorem grandmother_age_is_132 : grandmother_age = 132 := by
  sorry


end NUMINAMATH_CALUDE_grandmother_age_is_132_l556_55651


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l556_55681

theorem trigonometric_inequality (a b α β : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ α ∧ α ≤ Real.pi / 2) 
  (h4 : 0 ≤ β ∧ β ≤ Real.pi / 2) 
  (h5 : a * b * Real.cos (α - β) ≤ Real.sqrt ((1 - a^2) * (1 - b^2))) :
  a * Real.cos α + b * Real.sin β ≤ 1 + a * b * Real.sin (β - α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l556_55681


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l556_55698

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Represents the inner dimensions of the prism (not covered by foil) -/
def inner_prism : PrismDimensions :=
  { length := 4,
    width := 8,
    height := 4 }

/-- Represents the outer dimensions of the prism (covered by foil) -/
def outer_prism : PrismDimensions :=
  { length := inner_prism.length + 2,
    width := inner_prism.width + 2,
    height := inner_prism.height + 2 }

/-- The main theorem to prove -/
theorem foil_covered_prism_width :
  (volume inner_prism = 128) →
  (inner_prism.width = 2 * inner_prism.length) →
  (inner_prism.width = 2 * inner_prism.height) →
  (outer_prism.width = 10) := by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l556_55698


namespace NUMINAMATH_CALUDE_inequality_solution_set_l556_55665

theorem inequality_solution_set (x : ℝ) : 
  (1 / (x^2 + 4) > 5/x + 21/10) ↔ (-2 < x ∧ x < 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l556_55665


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l556_55645

theorem equation_has_real_roots (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  ∃ x : ℝ, x ≠ 1 ∧ a^2 / x + b^2 / (x - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l556_55645


namespace NUMINAMATH_CALUDE_friends_attended_reception_l556_55692

/-- The number of friends attending a wedding reception --/
def friends_at_reception (total_guests : ℕ) (family_couples : ℕ) (coworkers : ℕ) (distant_relatives : ℕ) : ℕ :=
  total_guests - (2 * (2 * family_couples + coworkers + distant_relatives))

/-- Theorem: Given the conditions of the wedding reception, 180 friends attended --/
theorem friends_attended_reception :
  friends_at_reception 400 40 10 20 = 180 := by
  sorry

end NUMINAMATH_CALUDE_friends_attended_reception_l556_55692


namespace NUMINAMATH_CALUDE_fourth_power_sum_l556_55638

theorem fourth_power_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 3)
  (sum_cubes : a^3 + b^3 + c^3 = 6) :
  a^4 + b^4 + c^4 = 34/3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l556_55638


namespace NUMINAMATH_CALUDE_fraction_sum_times_two_l556_55640

theorem fraction_sum_times_two : 
  (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_times_two_l556_55640


namespace NUMINAMATH_CALUDE_amount_fraction_is_one_third_l556_55680

/-- Represents the amounts received by A, B, and C in dollars -/
structure Amounts where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The conditions of the problem -/
def satisfies_conditions (x : Amounts) (total : ℝ) (fraction : ℝ) : Prop :=
  x.a + x.b + x.c = total ∧
  x.a = fraction * (x.b + x.c) ∧
  x.b = (2 / 7) * (x.a + x.c) ∧
  x.a = x.b + 10

theorem amount_fraction_is_one_third :
  ∃ (x : Amounts) (fraction : ℝ),
    satisfies_conditions x 360 fraction ∧ fraction = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_amount_fraction_is_one_third_l556_55680


namespace NUMINAMATH_CALUDE_age_difference_l556_55623

/-- Given four people A, B, C, and D with ages a, b, c, and d respectively,
    prove that C is 14 years younger than A under the given conditions. -/
theorem age_difference (a b c d : ℕ) : 
  (a + b = b + c + 14) →
  (b + d = c + a + 10) →
  (d = c + 6) →
  (a = c + 14) := by sorry

end NUMINAMATH_CALUDE_age_difference_l556_55623


namespace NUMINAMATH_CALUDE_points_on_line_l556_55699

theorem points_on_line (t : ℝ) :
  let x := Real.sin t ^ 2
  let y := Real.cos t ^ 2
  x + y = 1 := by
sorry

end NUMINAMATH_CALUDE_points_on_line_l556_55699


namespace NUMINAMATH_CALUDE_probability_three_fourths_radius_l556_55636

/-- A circle concentric with and outside a square --/
structure ConcentricCircleSquare where
  squareSideLength : ℝ
  circleRadius : ℝ
  squareSideLength_pos : 0 < squareSideLength
  circleRadius_gt_squareSideLength : squareSideLength < circleRadius

/-- The probability of seeing two sides of the square from a random point on the circle --/
def probabilityTwoSides (c : ConcentricCircleSquare) : ℝ := sorry

theorem probability_three_fourths_radius (c : ConcentricCircleSquare) 
  (h : c.squareSideLength = 4) 
  (prob : probabilityTwoSides c = 3/4) : 
  c.circleRadius = 8 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_three_fourths_radius_l556_55636


namespace NUMINAMATH_CALUDE_tie_points_in_tournament_l556_55684

def round_robin_tournament (n : ℕ) : ℕ := n * (n - 1) / 2

theorem tie_points_in_tournament (win_points tie_points : ℕ) :
  6 > 1 →
  win_points = 3 →
  round_robin_tournament 6 * win_points - round_robin_tournament 6 * tie_points = 15 →
  tie_points = 2 := by
  sorry

end NUMINAMATH_CALUDE_tie_points_in_tournament_l556_55684


namespace NUMINAMATH_CALUDE_perfect_squares_between_210_and_560_l556_55648

theorem perfect_squares_between_210_and_560 :
  (Finset.filter (fun n => 210 < n^2 ∧ n^2 < 560) (Finset.range 24)).card = 9 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_between_210_and_560_l556_55648


namespace NUMINAMATH_CALUDE_print_shop_charge_l556_55616

/-- The charge per color copy at print shop X -/
def charge_X : ℝ := 1.20

/-- The number of copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X for 40 copies -/
def additional_charge : ℝ := 20

/-- The charge per color copy at print shop Y -/
def charge_Y : ℝ := 1.70

theorem print_shop_charge :
  charge_Y * num_copies = charge_X * num_copies + additional_charge :=
by sorry

end NUMINAMATH_CALUDE_print_shop_charge_l556_55616


namespace NUMINAMATH_CALUDE_equation_simplification_l556_55687

theorem equation_simplification (x : ℝ) : 
  (x / 0.3 = 1 + (1.2 - 0.3 * x) / 0.2) ↔ (10 * x / 3 = 1 + (12 - 3 * x) / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_simplification_l556_55687


namespace NUMINAMATH_CALUDE_daily_production_is_1100_l556_55683

/-- The number of toys produced per week -/
def weekly_production : ℕ := 5500

/-- The number of working days per week -/
def working_days : ℕ := 5

/-- The number of toys produced each day -/
def daily_production : ℕ := weekly_production / working_days

/-- Theorem: The daily production of toys is 1100 -/
theorem daily_production_is_1100 : daily_production = 1100 := by
  sorry

end NUMINAMATH_CALUDE_daily_production_is_1100_l556_55683


namespace NUMINAMATH_CALUDE_no_solution_iff_a_geq_five_l556_55666

theorem no_solution_iff_a_geq_five (a : ℝ) :
  (∀ x : ℝ, ¬(x ≤ 5 ∧ x > a)) ↔ a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_geq_five_l556_55666


namespace NUMINAMATH_CALUDE_concentric_circles_chords_l556_55600

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the number of chords needed to complete
    a full circle is 3. -/
theorem concentric_circles_chords (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chords_l556_55600


namespace NUMINAMATH_CALUDE_town_population_growth_l556_55662

theorem town_population_growth (r : ℕ) (h1 : r^3 + 200 = (r + 1)^3 + 27) 
  (h2 : (r + 1)^3 + 300 = (r + 1)^3) : 
  (((r + 1)^3 - r^3) * 100 : ℚ) / r^3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_town_population_growth_l556_55662


namespace NUMINAMATH_CALUDE_bells_toll_together_l556_55609

theorem bells_toll_together (bell_intervals : List ℕ := [13, 17, 21, 26, 34, 39]) : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 13 17) 21) 26) 34) 39 = 9272 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l556_55609


namespace NUMINAMATH_CALUDE_cubic_tangent_max_value_l556_55642

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_tangent_max_value (a b m : ℝ) (hm : m ≠ 0) :
  (f a b m = 0) →                          -- f(x) is zero at x = m
  (f' a b m = 0) →                         -- f'(x) is zero at x = m
  (∀ x, f a b x ≤ (1/2)) →                 -- maximum value of f(x) is 1/2
  (∃ x, f a b x = (1/2)) →                 -- f(x) achieves the maximum value 1/2
  m = (3/2) := by sorry

end NUMINAMATH_CALUDE_cubic_tangent_max_value_l556_55642


namespace NUMINAMATH_CALUDE_f_composition_negative_one_l556_55631

-- Define the function f
def f (x : ℝ) : ℝ := x + 1

-- State the theorem
theorem f_composition_negative_one : f (f (-1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_one_l556_55631


namespace NUMINAMATH_CALUDE_smallest_perfect_square_factor_l556_55602

def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

theorem smallest_perfect_square_factor (k : ℕ) : 
  (k > 0 ∧ ∃ m : ℕ, k * y = m^2 ∧ ∀ n : ℕ, 0 < n ∧ n < k → ¬∃ m : ℕ, n * y = m^2) ↔ k = 110 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_factor_l556_55602


namespace NUMINAMATH_CALUDE_count_valid_bases_for_216_l556_55618

theorem count_valid_bases_for_216 :
  ∃! (n : ℕ), n > 0 ∧ (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ k : ℕ, k > 0 ∧ b^k = 216) ∧
    S.card = n ∧
    (∀ b : ℕ, b > 0 → (∃ k : ℕ, k > 0 ∧ b^k = 216) → b ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_valid_bases_for_216_l556_55618


namespace NUMINAMATH_CALUDE_coin_sum_theorem_l556_55649

def coin_values : List Nat := [5, 10, 25, 50, 100]

def sum_three_coins (a b c : Nat) : Nat := a + b + c

def is_valid_sum (sum : Nat) : Prop :=
  ∃ (a b c : Nat), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ sum_three_coins a b c = sum

theorem coin_sum_theorem :
  ¬(is_valid_sum 52) ∧
  (is_valid_sum 60) ∧
  (is_valid_sum 115) ∧
  (is_valid_sum 165) ∧
  (is_valid_sum 180) :=
sorry

end NUMINAMATH_CALUDE_coin_sum_theorem_l556_55649


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l556_55679

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 11) 
  (sum_products_eq : a * b + a * c + b * c = 25) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 506 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l556_55679


namespace NUMINAMATH_CALUDE_unique_c_value_l556_55650

/-- The quadratic equation we're considering -/
def quadratic (b : ℝ) (c : ℝ) (x : ℝ) : Prop :=
  x^2 + (b^2 + 1/b^2) * x + c = 3

/-- The condition for the quadratic to have exactly one solution -/
def has_unique_solution (b : ℝ) (c : ℝ) : Prop :=
  ∃! x, quadratic b c x

/-- The main theorem statement -/
theorem unique_c_value : ∃! c : ℝ, c ≠ 0 ∧ 
  (∃! b : ℕ, b > 0 ∧ has_unique_solution (b : ℝ) c) :=
sorry

end NUMINAMATH_CALUDE_unique_c_value_l556_55650


namespace NUMINAMATH_CALUDE_path_length_calculation_l556_55660

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 500

/-- Represents the length of the path on the map in inches -/
def path_length_on_map : ℝ := 3.5

/-- Calculates the actual length of the path in feet -/
def actual_path_length : ℝ := map_scale * path_length_on_map

theorem path_length_calculation :
  actual_path_length = 1750 := by sorry

end NUMINAMATH_CALUDE_path_length_calculation_l556_55660


namespace NUMINAMATH_CALUDE_cube_split_sequence_l556_55686

theorem cube_split_sequence (n : ℕ) : ∃ (k : ℕ), 
  2019 = n^2 - (n - 1) + 2 * k ∧ 
  0 ≤ k ∧ 
  k < n ∧ 
  n = 45 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_sequence_l556_55686


namespace NUMINAMATH_CALUDE_ancient_chinese_gold_tax_l556_55620

theorem ancient_chinese_gold_tax (x : ℚ) : 
  x > 0 ∧ 
  x/2 + x/2 * 1/3 + x/3 * 1/4 + x/4 * 1/5 + x/5 * 1/6 = 1 → 
  x/5 * 1/6 = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_gold_tax_l556_55620


namespace NUMINAMATH_CALUDE_remi_and_father_seedlings_l556_55607

/-- The number of seedlings Remi's father planted -/
def fathers_seedlings (day1 : ℕ) (total : ℕ) : ℕ :=
  total - (day1 + 2 * day1)

theorem remi_and_father_seedlings :
  fathers_seedlings 200 1200 = 600 := by
  sorry

end NUMINAMATH_CALUDE_remi_and_father_seedlings_l556_55607


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l556_55656

/-- Given that i² = -1, prove that (2 - 3i) / (4 - 5i) = 23/41 - (2/41)i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - 3*i) / (4 - 5*i) = 23/41 - (2/41)*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l556_55656


namespace NUMINAMATH_CALUDE_max_value_inequality_l556_55667

theorem max_value_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  2 * x * y * Real.sqrt 8 + 7 * y * z + 5 * x * z ≤ 23.0219 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l556_55667


namespace NUMINAMATH_CALUDE_smallest_angle_measure_l556_55604

theorem smallest_angle_measure (ABC ABD : ℝ) (h1 : ABC = 40) (h2 : ABD = 30) :
  ∃ (CBD : ℝ), CBD = ABC - ABD ∧ CBD = 10 ∧ ∀ (x : ℝ), x ≥ 0 → x ≥ CBD :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_measure_l556_55604


namespace NUMINAMATH_CALUDE_modified_short_bingo_first_column_possibilities_l556_55677

theorem modified_short_bingo_first_column_possibilities : 
  (Finset.univ.filter (λ x : Finset (Fin 12) => x.card = 5)).card = 95040 := by
  sorry

end NUMINAMATH_CALUDE_modified_short_bingo_first_column_possibilities_l556_55677


namespace NUMINAMATH_CALUDE_husk_consumption_rate_l556_55608

/-- Given that 20 cows eat 20 bags of husk in 20 days, prove that 1 cow will eat 1 bag of husk in 20 days -/
theorem husk_consumption_rate (cows bags days : ℕ) (h1 : cows = 20) (h2 : bags = 20) (h3 : days = 20) :
  (1 : ℚ) / cows * bags * days = 20 := by
  sorry

end NUMINAMATH_CALUDE_husk_consumption_rate_l556_55608


namespace NUMINAMATH_CALUDE_olivias_paper_pieces_l556_55621

/-- The number of paper pieces Olivia used -/
def pieces_used : ℕ := 56

/-- The number of paper pieces Olivia has left -/
def pieces_left : ℕ := 25

/-- The initial number of paper pieces Olivia had -/
def initial_pieces : ℕ := pieces_used + pieces_left

theorem olivias_paper_pieces : initial_pieces = 81 := by
  sorry

end NUMINAMATH_CALUDE_olivias_paper_pieces_l556_55621


namespace NUMINAMATH_CALUDE_part_I_part_II_l556_55673

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 3| + |x - a|

-- Part I
theorem part_I : 
  ∀ x : ℝ, f 4 x = 7 → x ∈ Set.Icc (-3) 4 := by sorry

-- Part II
theorem part_II : 
  ∀ a : ℝ, a > 0 → 
  ({x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2}) → 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l556_55673


namespace NUMINAMATH_CALUDE_product_congruence_l556_55630

theorem product_congruence : 66 * 77 * 88 ≡ 16 [ZMOD 25] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l556_55630


namespace NUMINAMATH_CALUDE_complex_equation_solution_l556_55612

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_equation_solution (z : ℂ) (a : ℝ) (h1 : is_pure_imaginary z) 
  (h2 : (2 - Complex.I) * z = a + Complex.I) : a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l556_55612


namespace NUMINAMATH_CALUDE_second_number_proof_l556_55689

theorem second_number_proof (x : ℝ) : 3 + x + 333 + 3.33 = 369.63 → x = 30.3 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l556_55689


namespace NUMINAMATH_CALUDE_angle_decreases_as_n_increases_l556_55671

/-- Given two mutually perpendicular unit vectors i and j, and a vector a_n that satisfies
    the given dot product conditions, theta_n is the angle between i and a_n. -/
theorem angle_decreases_as_n_increases (i j : ℝ × ℝ) (a : ℕ+ → ℝ × ℝ) 
  (h_perp : i.1 * j.1 + i.2 * j.2 = 0)
  (h_unit_i : i.1^2 + i.2^2 = 1)
  (h_unit_j : j.1^2 + j.2^2 = 1)
  (h_dot_i : ∀ n : ℕ+, i.1 * (a n).1 + i.2 * (a n).2 = n)
  (h_dot_j : ∀ n : ℕ+, j.1 * (a n).1 + j.2 * (a n).2 = 2*n + 1)
  (theta : ℕ+ → ℝ)
  (h_theta : ∀ n : ℕ+, theta n = Real.arccos ((i.1 * (a n).1 + i.2 * (a n).2) / 
    (Real.sqrt (i.1^2 + i.2^2) * Real.sqrt ((a n).1^2 + (a n).2^2))))
  (n₁ n₂ : ℕ+) (h_lt : n₁ < n₂) :
  theta n₁ > theta n₂ :=
sorry

end NUMINAMATH_CALUDE_angle_decreases_as_n_increases_l556_55671


namespace NUMINAMATH_CALUDE_profit_for_two_yuan_reduction_selling_price_for_770_profit_no_price_for_880_profit_l556_55668

/-- Represents the supermarket beverage pricing and sales model -/
structure BeverageModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ

/-- Calculates the profit for a given price reduction -/
def profit (model : BeverageModel) (price_reduction : ℝ) : ℝ :=
  let new_price := model.initial_price - price_reduction
  let new_sales := model.initial_sales + model.price_sensitivity * price_reduction
  (new_price - model.cost_price) * new_sales

/-- Theorem: The profit with a 2 yuan price reduction is 800 yuan -/
theorem profit_for_two_yuan_reduction (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  profit model 2 = 800 := by sorry

/-- Theorem: To achieve a profit of 770 yuan, the selling price should be 55 yuan -/
theorem selling_price_for_770_profit (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  ∃ (price_reduction : ℝ), profit model price_reduction = 770 ∧ 
  model.initial_price - price_reduction = 55 := by sorry

/-- Theorem: There is no selling price that can achieve a profit of 880 yuan -/
theorem no_price_for_880_profit (model : BeverageModel) 
  (h1 : model.cost_price = 48)
  (h2 : model.initial_price = 60)
  (h3 : model.initial_sales = 60)
  (h4 : model.price_sensitivity = 10) :
  ¬∃ (price_reduction : ℝ), profit model price_reduction = 880 := by sorry

end NUMINAMATH_CALUDE_profit_for_two_yuan_reduction_selling_price_for_770_profit_no_price_for_880_profit_l556_55668


namespace NUMINAMATH_CALUDE_sallys_class_size_l556_55693

theorem sallys_class_size (school_money : ℕ) (book_cost : ℕ) (out_of_pocket : ℕ) :
  school_money = 320 →
  book_cost = 12 →
  out_of_pocket = 40 →
  (school_money + out_of_pocket) / book_cost = 30 := by
sorry

end NUMINAMATH_CALUDE_sallys_class_size_l556_55693


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l556_55682

/-- The x-intercept of the line 5y - 7x = 35 is (-5, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  5 * y - 7 * x = 35 → y = 0 → x = -5 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l556_55682


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l556_55641

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if three real numbers form a geometric sequence -/
def isGeometricSequence (a b c : ℝ) : Prop :=
  ∃ k : ℝ, b = k * a ∧ c = k * b

/-- Theorem: The root of a specific quadratic equation -/
theorem quadratic_root_theorem (p q r : ℝ) (h1 : isGeometricSequence p q r)
    (h2 : p ≤ q ∧ q ≤ r ∧ r ≤ 0) (h3 : ∃! x : ℝ, p * x^2 + q * x + r = 0) :
    ∃ x : ℝ, p * x^2 + q * x + r = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l556_55641


namespace NUMINAMATH_CALUDE_alpha_plus_beta_eq_115_l556_55652

theorem alpha_plus_beta_eq_115 :
  ∃ (α β : ℝ), (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 116*x + 2783) / (x^2 + 99*x - 4080)) →
  α + β = 115 := by
  sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_eq_115_l556_55652


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l556_55627

theorem defective_shipped_percentage 
  (total_units : ℕ) 
  (defective_percentage : ℝ) 
  (shipped_percentage : ℝ) 
  (h1 : defective_percentage = 7) 
  (h2 : shipped_percentage = 5) : 
  (defective_percentage / 100) * (shipped_percentage / 100) * 100 = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l556_55627


namespace NUMINAMATH_CALUDE_rohan_salary_rohan_salary_proof_l556_55643

/-- Rohan's monthly salary calculation --/
theorem rohan_salary (food_percent : ℝ) (rent_percent : ℝ) (entertainment_percent : ℝ) 
  (conveyance_percent : ℝ) (savings : ℝ) : ℝ :=
  let total_expenses_percent : ℝ := food_percent + rent_percent + entertainment_percent + conveyance_percent
  let savings_percent : ℝ := 1 - total_expenses_percent
  savings / savings_percent

/-- Proof of Rohan's monthly salary --/
theorem rohan_salary_proof :
  rohan_salary 0.4 0.2 0.1 0.1 2500 = 12500 := by
  sorry

end NUMINAMATH_CALUDE_rohan_salary_rohan_salary_proof_l556_55643


namespace NUMINAMATH_CALUDE_ball_drawing_properties_l556_55655

/-- Represents the number of balls drawn -/
def n : ℕ := 3

/-- Represents the initial number of red balls -/
def r : ℕ := 5

/-- Represents the initial number of black balls -/
def b : ℕ := 2

/-- Represents the total number of balls -/
def total : ℕ := r + b

/-- Represents the random variable for the number of red balls drawn without replacement -/
def X : Fin (n + 1) → ℝ := sorry

/-- Represents the random variable for the number of black balls drawn without replacement -/
def Y : Fin (n + 1) → ℝ := sorry

/-- Represents the random variable for the number of red balls drawn with replacement -/
def ξ : Fin (n + 1) → ℝ := sorry

/-- The expected value of X -/
noncomputable def E_X : ℝ := sorry

/-- The expected value of Y -/
noncomputable def E_Y : ℝ := sorry

/-- The expected value of ξ -/
noncomputable def E_ξ : ℝ := sorry

/-- The variance of X -/
noncomputable def D_X : ℝ := sorry

/-- The variance of ξ -/
noncomputable def D_ξ : ℝ := sorry

theorem ball_drawing_properties :
  (E_X / E_Y = r / b) ∧ (E_X = E_ξ) ∧ (D_X < D_ξ) := by sorry

end NUMINAMATH_CALUDE_ball_drawing_properties_l556_55655


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l556_55676

/-- Represents a parabola of the form y = a(x - h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The given parabola y = -2(x-1)^2 + 3 --/
def givenParabola : Parabola :=
  { a := -2
  , h := 1
  , k := 3 }

/-- The axis of symmetry of a parabola --/
def axisOfSymmetry (p : Parabola) : ℝ := p.h

theorem axis_of_symmetry_is_x_equals_one :
  axisOfSymmetry givenParabola = 1 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l556_55676


namespace NUMINAMATH_CALUDE_class_overlap_difference_l556_55661

theorem class_overlap_difference (total students_geometry students_biology : ℕ) 
  (h_total : total = 232)
  (h_geometry : students_geometry = 144)
  (h_biology : students_biology = 119) :
  let max_overlap := min students_geometry students_biology
  let min_overlap := students_geometry + students_biology - total
  max_overlap - min_overlap = 88 := by
sorry

end NUMINAMATH_CALUDE_class_overlap_difference_l556_55661


namespace NUMINAMATH_CALUDE_benny_seashells_l556_55639

/-- The number of seashells Benny has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Benny has 14 seashells after giving away 52 from his initial 66 -/
theorem benny_seashells : remaining_seashells 66 52 = 14 := by
  sorry

end NUMINAMATH_CALUDE_benny_seashells_l556_55639


namespace NUMINAMATH_CALUDE_smallest_with_eight_prime_power_divisors_l556_55615

def is_prime_power (n : ℕ) : Prop := ∃ p k, Prime p ∧ n = p ^ k

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem smallest_with_eight_prime_power_divisors :
  (∀ m : ℕ, m < 24 →
    (divisors m).card ≠ 8 ∨
    ¬(∀ d ∈ divisors m, is_prime_power d)) ∧
  (divisors 24).card = 8 ∧
  (∀ d ∈ divisors 24, is_prime_power d) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_eight_prime_power_divisors_l556_55615


namespace NUMINAMATH_CALUDE_hcf_of_8_and_12_l556_55696

theorem hcf_of_8_and_12 :
  let a : ℕ := 8
  let b : ℕ := 12
  Nat.lcm a b = 24 →
  Nat.gcd a b = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_hcf_of_8_and_12_l556_55696


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l556_55647

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : sides ≥ 3

/-- Represents the configuration of two convex polygons -/
structure TwoPolygons where
  P₁ : ConvexPolygon
  P₂ : ConvexPolygon
  sameplane : True  -- Represents that P₁ and P₂ are on the same plane
  no_overlap : True  -- Represents that P₁ and P₂ do not have overlapping line segments
  size_order : P₁.sides ≤ P₂.sides

/-- The function that calculates the maximum number of intersection points -/
def max_intersections (tp : TwoPolygons) : ℕ := 2 * tp.P₁.sides

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersections_theorem (tp : TwoPolygons) : 
  max_intersections tp = 2 * tp.P₁.sides := by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l556_55647


namespace NUMINAMATH_CALUDE_sequence_increasing_l556_55606

theorem sequence_increasing (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_rel : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a (n + 1) > a n :=
sorry

end NUMINAMATH_CALUDE_sequence_increasing_l556_55606


namespace NUMINAMATH_CALUDE_fraction_meaningful_l556_55619

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l556_55619


namespace NUMINAMATH_CALUDE_rectangle_area_l556_55664

/-- The area of a rectangle with sides 5.9 cm and 3 cm is 17.7 square centimeters. -/
theorem rectangle_area : 
  let side1 : ℝ := 5.9
  let side2 : ℝ := 3
  side1 * side2 = 17.7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l556_55664


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l556_55657

/-- Proves that given the specified blanket purchases and average price, the unknown rate is 285 --/
theorem unknown_blanket_rate (num_blankets_1 num_blankets_2 num_unknown : ℕ)
                              (price_1 price_2 avg_price : ℚ) :
  num_blankets_1 = 3 →
  num_blankets_2 = 5 →
  num_unknown = 2 →
  price_1 = 100 →
  price_2 = 150 →
  avg_price = 162 →
  let total_blankets := num_blankets_1 + num_blankets_2 + num_unknown
  let total_cost := avg_price * total_blankets
  let known_cost := num_blankets_1 * price_1 + num_blankets_2 * price_2
  let unknown_cost := total_cost - known_cost
  unknown_cost / num_unknown = 285 := by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l556_55657


namespace NUMINAMATH_CALUDE_smallest_value_of_y_l556_55663

theorem smallest_value_of_y (x : ℝ) : 
  (17 - x) * (19 - x) * (19 + x) * (17 + x) ≥ -1296 ∧ 
  ∃ x : ℝ, (17 - x) * (19 - x) * (19 + x) * (17 + x) = -1296 :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_y_l556_55663


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l556_55658

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- Given -1 < m < 1, the complex number (1-i) + m(1+i) is in the fourth quadrant. -/
theorem complex_in_fourth_quadrant (m : ℝ) (h : -1 < m ∧ m < 1) :
  in_fourth_quadrant ((1 - Complex.I) + m * (1 + Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l556_55658


namespace NUMINAMATH_CALUDE_skating_speed_ratio_l556_55634

theorem skating_speed_ratio (v_f v_s : ℝ) (h1 : v_f > 0) (h2 : v_s > 0) :
  (v_f + v_s) / (v_f - v_s) = 5 → v_f / v_s = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_skating_speed_ratio_l556_55634


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l556_55659

/-- Represents the number of students to be chosen from a class in stratified sampling -/
def stratified_sample (total_students : ℕ) (class_size : ℕ) (sample_size : ℕ) : ℕ :=
  (class_size * sample_size) / total_students

theorem stratified_sampling_correct (class1_size class2_size sample_size : ℕ) 
  (h1 : class1_size = 54)
  (h2 : class2_size = 42)
  (h3 : sample_size = 16) :
  (stratified_sample (class1_size + class2_size) class1_size sample_size = 9) ∧
  (stratified_sample (class1_size + class2_size) class2_size sample_size = 7) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l556_55659
