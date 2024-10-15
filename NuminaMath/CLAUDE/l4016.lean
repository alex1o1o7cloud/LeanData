import Mathlib

namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l4016_401611

theorem fraction_of_powers_equals_five_fourths :
  (3^10 + 3^8) / (3^10 - 3^8) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_five_fourths_l4016_401611


namespace NUMINAMATH_CALUDE_graham_crackers_leftover_l4016_401690

/-- Represents the number of boxes of Graham crackers Lionel bought -/
def graham_crackers : ℕ := 14

/-- Represents the number of packets of Oreos Lionel bought -/
def oreos : ℕ := 15

/-- Represents the number of boxes of Graham crackers needed for one cheesecake -/
def graham_crackers_per_cake : ℕ := 2

/-- Represents the number of packets of Oreos needed for one cheesecake -/
def oreos_per_cake : ℕ := 3

/-- Calculates the number of boxes of Graham crackers left over after making
    the maximum number of Oreo cheesecakes -/
def graham_crackers_left : ℕ :=
  graham_crackers - graham_crackers_per_cake * (min (graham_crackers / graham_crackers_per_cake) (oreos / oreos_per_cake))

theorem graham_crackers_leftover :
  graham_crackers_left = 4 := by sorry

end NUMINAMATH_CALUDE_graham_crackers_leftover_l4016_401690


namespace NUMINAMATH_CALUDE_author_earnings_l4016_401671

theorem author_earnings (paper_cover_percentage : ℝ) (hardcover_percentage : ℝ)
  (paper_cover_copies : ℕ) (hardcover_copies : ℕ)
  (paper_cover_price : ℝ) (hardcover_price : ℝ) :
  paper_cover_percentage = 0.06 →
  hardcover_percentage = 0.12 →
  paper_cover_copies = 32000 →
  hardcover_copies = 15000 →
  paper_cover_price = 0.20 →
  hardcover_price = 0.40 →
  (paper_cover_percentage * (paper_cover_copies : ℝ) * paper_cover_price) +
  (hardcover_percentage * (hardcover_copies : ℝ) * hardcover_price) = 1104 :=
by sorry

end NUMINAMATH_CALUDE_author_earnings_l4016_401671


namespace NUMINAMATH_CALUDE_factorization_theorem_l4016_401666

theorem factorization_theorem (m n : ℝ) : m^3*n - m*n = m*n*(m-1)*(m+1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l4016_401666


namespace NUMINAMATH_CALUDE_counterexample_exists_l4016_401607

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def given_numbers : List ℕ := [6, 9, 10, 11, 15]

theorem counterexample_exists : ∃ n : ℕ, 
  n ∈ given_numbers ∧
  ¬(is_prime n) ∧ 
  is_prime (n - 2) ∧ 
  is_prime (n + 2) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l4016_401607


namespace NUMINAMATH_CALUDE_root_sum_product_l4016_401651

theorem root_sum_product (p q : ℝ) : 
  (∃ x, x^4 - 6*x - 2 = 0) → 
  (p^4 - 6*p - 2 = 0) →
  (q^4 - 6*q - 2 = 0) →
  p ≠ q →
  pq + p + q = 1 - 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_product_l4016_401651


namespace NUMINAMATH_CALUDE_min_female_participants_l4016_401619

/-- Proves the minimum number of female students participating in community work -/
theorem min_female_participants (male_students female_students : ℕ) 
  (total_participants : ℕ) (h1 : male_students = 22) (h2 : female_students = 18) 
  (h3 : total_participants = ((male_students + female_students) * 6) / 10) : 
  ∃ (female_participants : ℕ), 
    female_participants ≥ 2 ∧ 
    female_participants ≤ female_students ∧
    female_participants + male_students ≥ total_participants :=
by sorry

end NUMINAMATH_CALUDE_min_female_participants_l4016_401619


namespace NUMINAMATH_CALUDE_system_solution_l4016_401620

theorem system_solution (x y k : ℝ) : 
  (2 * x + 3 * y = k) → 
  (x + 4 * y = k - 16) → 
  (x + y = 8) → 
  k = 12 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4016_401620


namespace NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l4016_401625

/-- Represents the dimensions of the wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  height : ℕ
  length : ℕ

/-- Calculates the minimum number of blocks required to build the wall --/
def minBlocksRequired (wall : WallDimensions) (block1 : BlockDimensions) (block2 : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks required for the specific wall --/
theorem min_blocks_for_specific_wall :
  let wall := WallDimensions.mk 120 10
  let block1 := BlockDimensions.mk 1 3
  let block2 := BlockDimensions.mk 1 1
  minBlocksRequired wall block1 block2 = 415 :=
by sorry

end NUMINAMATH_CALUDE_min_blocks_for_specific_wall_l4016_401625


namespace NUMINAMATH_CALUDE_power_simplification_l4016_401642

theorem power_simplification : 10^6 * (10^2)^3 / 10^4 = 10^8 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l4016_401642


namespace NUMINAMATH_CALUDE_tan_series_equality_l4016_401622

theorem tan_series_equality (x : ℝ) (h : |Real.tan x| < 1) :
  8.407 * ((1 - Real.tan x)⁻¹) / ((1 + Real.tan x)⁻¹) = 1 + Real.sin (2 * x) ↔
  ∃ k : ℤ, x = k * Real.pi := by sorry

end NUMINAMATH_CALUDE_tan_series_equality_l4016_401622


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l4016_401613

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l4016_401613


namespace NUMINAMATH_CALUDE_scott_cake_sales_l4016_401644

theorem scott_cake_sales (smoothie_price : ℕ) (cake_price : ℕ) (smoothies_sold : ℕ) (total_revenue : ℕ) :
  smoothie_price = 3 →
  cake_price = 2 →
  smoothies_sold = 40 →
  total_revenue = 156 →
  ∃ (cakes_sold : ℕ), smoothie_price * smoothies_sold + cake_price * cakes_sold = total_revenue ∧ cakes_sold = 18 := by
  sorry

end NUMINAMATH_CALUDE_scott_cake_sales_l4016_401644


namespace NUMINAMATH_CALUDE_batsman_average_l4016_401617

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = 10 * previous_average →
  (previous_total + 69) / 11 = previous_average + 1 →
  (previous_total + 69) / 11 = 59 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l4016_401617


namespace NUMINAMATH_CALUDE_two_points_on_curve_l4016_401643

def point_on_curve (x y : ℝ) : Prop :=
  x^2 - x*y + 2*y + 1 = 0

def point_A : ℝ × ℝ := (1, -2)
def point_B : ℝ × ℝ := (2, -3)
def point_C : ℝ × ℝ := (3, 10)

theorem two_points_on_curve :
  (point_on_curve point_A.1 point_A.2 ∧
   point_on_curve point_C.1 point_C.2 ∧
   ¬point_on_curve point_B.1 point_B.2) ∨
  (point_on_curve point_A.1 point_A.2 ∧
   point_on_curve point_B.1 point_B.2 ∧
   ¬point_on_curve point_C.1 point_C.2) ∨
  (point_on_curve point_B.1 point_B.2 ∧
   point_on_curve point_C.1 point_C.2 ∧
   ¬point_on_curve point_A.1 point_A.2) :=
sorry

end NUMINAMATH_CALUDE_two_points_on_curve_l4016_401643


namespace NUMINAMATH_CALUDE_successive_numbers_product_l4016_401677

theorem successive_numbers_product (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 4160 → n = 64 := by
  sorry

end NUMINAMATH_CALUDE_successive_numbers_product_l4016_401677


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l4016_401604

def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r ^ (n - 1)

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * (1 - r^n) / (1 - r)

theorem geometric_sequence_properties :
  let a : ℚ := 1/5
  let r : ℚ := 1/2
  let n : ℕ := 8
  (geometric_sequence a r n = 1/640) ∧
  (geometric_sum a r n = 255/320) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l4016_401604


namespace NUMINAMATH_CALUDE_inequality_solution_l4016_401638

theorem inequality_solution (x : ℝ) : 
  (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) ↔ x < -4 ∨ x > -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4016_401638


namespace NUMINAMATH_CALUDE_correct_calculation_result_l4016_401695

theorem correct_calculation_result (x : ℝ) (h : 4 * x = 52) : 20 - x = 7 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_result_l4016_401695


namespace NUMINAMATH_CALUDE_set_union_problem_l4016_401656

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {2^a, b}

-- Theorem statement
theorem set_union_problem (a b : ℝ) : 
  A a ∩ B a b = {1} → A a ∪ B a b = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_set_union_problem_l4016_401656


namespace NUMINAMATH_CALUDE_determinant_zero_l4016_401662

theorem determinant_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![1, Real.sin (a + b), Real.sin a],
    ![Real.sin (a + b), 1, Real.sin b],
    ![Real.sin a, Real.sin b, 1]
  ]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_determinant_zero_l4016_401662


namespace NUMINAMATH_CALUDE_banana_count_l4016_401649

/-- Proves that given 8 boxes and 5 bananas per box, the total number of bananas is 40. -/
theorem banana_count (num_boxes : ℕ) (bananas_per_box : ℕ) (total_bananas : ℕ) : 
  num_boxes = 8 → bananas_per_box = 5 → total_bananas = num_boxes * bananas_per_box → total_bananas = 40 :=
by sorry

end NUMINAMATH_CALUDE_banana_count_l4016_401649


namespace NUMINAMATH_CALUDE_simplify_fraction_l4016_401674

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4016_401674


namespace NUMINAMATH_CALUDE_max_point_range_l4016_401624

/-- Given a differentiable function f : ℝ → ℝ and a real number a, 
    if f'(x) = a(x-1)(x-a) for all x and f attains a maximum at x = a, 
    then 0 < a < 1 -/
theorem max_point_range (f : ℝ → ℝ) (a : ℝ) 
    (h1 : Differentiable ℝ f) 
    (h2 : ∀ x, deriv f x = a * (x - 1) * (x - a))
    (h3 : IsLocalMax f a) : 
    0 < a ∧ a < 1 := by
  sorry


end NUMINAMATH_CALUDE_max_point_range_l4016_401624


namespace NUMINAMATH_CALUDE_onion_harvest_weight_l4016_401672

/-- The total weight of onions harvested by Titan's father -/
def total_weight (bags_per_trip : ℕ) (weight_per_bag : ℕ) (num_trips : ℕ) : ℕ :=
  bags_per_trip * weight_per_bag * num_trips

/-- Theorem stating the total weight of onions harvested -/
theorem onion_harvest_weight :
  total_weight 10 50 20 = 10000 := by
  sorry

#eval total_weight 10 50 20

end NUMINAMATH_CALUDE_onion_harvest_weight_l4016_401672


namespace NUMINAMATH_CALUDE_function_composition_result_l4016_401660

theorem function_composition_result (c d : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (hf : ∀ x, f x = 5*x + c) 
  (hg : ∀ x, g x = c*x + 3) 
  (h_comp : ∀ x, f (g x) = 15*x + d) : d = 18 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_result_l4016_401660


namespace NUMINAMATH_CALUDE_roots_of_g_l4016_401667

/-- Given that 2 is a root of f(x) = ax + b, prove that the roots of g(x) = bx² - ax are 0 and -1/2 --/
theorem roots_of_g (a b : ℝ) (h : a * 2 + b = 0) :
  ∃ (x y : ℝ), x = 0 ∧ y = -1/2 ∧ ∀ z : ℝ, b * z^2 - a * z = 0 ↔ z = x ∨ z = y :=
by sorry

end NUMINAMATH_CALUDE_roots_of_g_l4016_401667


namespace NUMINAMATH_CALUDE_angle_measure_problem_l4016_401683

theorem angle_measure_problem (angle_B angle_small_triangle : ℝ) :
  angle_B = 120 →
  angle_small_triangle = 50 →
  ∃ angle_A : ℝ,
    angle_A = 70 ∧
    angle_A + angle_small_triangle + (180 - angle_B) = 180 :=
by sorry

end NUMINAMATH_CALUDE_angle_measure_problem_l4016_401683


namespace NUMINAMATH_CALUDE_division_problem_l4016_401601

theorem division_problem : ∃ (q : ℕ), 
  220080 = (555 + 445) * q + 80 ∧ 
  q = 2 * (555 - 445) := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4016_401601


namespace NUMINAMATH_CALUDE_factorization_equality_l4016_401632

theorem factorization_equality (a b : ℝ) : a^3 - 9*a*b^2 = a*(a+3*b)*(a-3*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4016_401632


namespace NUMINAMATH_CALUDE_binary_11011_is_27_l4016_401681

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11011_is_27 :
  binary_to_decimal [true, true, false, true, true] = 27 := by
  sorry

end NUMINAMATH_CALUDE_binary_11011_is_27_l4016_401681


namespace NUMINAMATH_CALUDE_lcm_equality_pairs_l4016_401654

theorem lcm_equality_pairs (m n : ℕ) : 
  Nat.lcm m n = 3 * m + 2 * n + 1 ↔ (m = 3 ∧ n = 10) ∨ (m = 9 ∧ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_lcm_equality_pairs_l4016_401654


namespace NUMINAMATH_CALUDE_inequality_proof_l4016_401657

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  1/x + 1/y + 1/z + 9/(x+y+z) ≥ 
  3 * ((1/(2*x+y) + 1/(x+2*y)) + (1/(2*y+z) + 1/(y+2*z)) + (1/(2*z+x) + 1/(x+2*z))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4016_401657


namespace NUMINAMATH_CALUDE_right_triangle_area_l4016_401623

/-- 
  Given a right-angled triangle with legs x and y, and hypotenuse z,
  where x:y = 3:4 and x^2 + y^2 = z^2, prove that the area A of the triangle
  is equal to (2/3)x^2 or (6/25)z^2.
-/
theorem right_triangle_area (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x^2 + y^2 = z^2) (h5 : 3 * y = 4 * x) :
  ∃ A : ℝ, A = (2/3) * x^2 ∧ A = (6/25) * z^2 := by
  sorry

#check right_triangle_area

end NUMINAMATH_CALUDE_right_triangle_area_l4016_401623


namespace NUMINAMATH_CALUDE_difference_of_squares_l4016_401678

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l4016_401678


namespace NUMINAMATH_CALUDE_sin_theta_value_l4016_401676

theorem sin_theta_value (θ : Real) (h1 : 0 < θ) (h2 : θ < Real.pi / 2) 
  (h3 : 1 + Real.sin θ = 2 * Real.cos θ) : Real.sin θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l4016_401676


namespace NUMINAMATH_CALUDE_oranges_thrown_away_l4016_401679

theorem oranges_thrown_away (initial : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 34 → new = 13 → final = 27 → initial - (initial - final + new) = 20 := by
  sorry

end NUMINAMATH_CALUDE_oranges_thrown_away_l4016_401679


namespace NUMINAMATH_CALUDE_stating_work_completion_time_l4016_401637

/-- The time it takes for a man and his son to complete a piece of work together -/
def combined_time : ℝ := 6

/-- The time it takes for the son to complete the work alone -/
def son_time : ℝ := 10

/-- The time it takes for the man to complete the work alone -/
def man_time : ℝ := 15

/-- 
Theorem stating that if a man and his son can complete a piece of work together in 6 days, 
and the son can complete the work alone in 10 days, then the man can complete the work 
alone in 15 days.
-/
theorem work_completion_time : 
  (1 / combined_time) = (1 / man_time) + (1 / son_time) :=
sorry

end NUMINAMATH_CALUDE_stating_work_completion_time_l4016_401637


namespace NUMINAMATH_CALUDE_parabola_symmetry_axis_l4016_401628

/-- The axis of symmetry of a parabola y^2 = mx has the equation x = -m/4 -/
def axis_of_symmetry (m : ℝ) : ℝ → Prop :=
  fun x ↦ x = -m/4

/-- A point (x, y) lies on the parabola y^2 = mx -/
def on_parabola (m : ℝ) : ℝ × ℝ → Prop :=
  fun p ↦ p.2^2 = m * p.1

theorem parabola_symmetry_axis (m : ℝ) :
  axis_of_symmetry m (-m^2) →
  on_parabola m (-m^2, 3) →
  m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_axis_l4016_401628


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l4016_401631

theorem imaginary_part_of_reciprocal (a : ℝ) (z : ℂ) : 
  z = a^2 - 1 + (a + 1) * Complex.I → 
  z.re = 0 → 
  z ≠ 0 → 
  (Complex.I * ((z + a)⁻¹)).re = -2/5 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l4016_401631


namespace NUMINAMATH_CALUDE_equation_equivalence_l4016_401603

theorem equation_equivalence (x : ℝ) : (5 = 3 * x - 2) ↔ (5 + 2 = 3 * x) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l4016_401603


namespace NUMINAMATH_CALUDE_equation_roots_problem_l4016_401692

theorem equation_roots_problem (p q : ℝ) 
  (eq1 : ∃ x1 x2 : ℝ, x1^2 - p*x1 + 4 = 0 ∧ x2^2 - p*x2 + 4 = 0 ∧ x1 ≠ x2)
  (eq2 : ∃ x3 x4 : ℝ, 2*x3^2 - 9*x3 + q = 0 ∧ 2*x4^2 - 9*x4 + q = 0 ∧ x3 ≠ x4)
  (root_relation : ∃ x1 x2 x3 x4 : ℝ, 
    x1^2 - p*x1 + 4 = 0 ∧ x2^2 - p*x2 + 4 = 0 ∧ x1 < x2 ∧
    2*x3^2 - 9*x3 + q = 0 ∧ 2*x4^2 - 9*x4 + q = 0 ∧
    ((x3 = x2 + 2 ∧ x4 = x1 - 2) ∨ (x4 = x2 + 2 ∧ x3 = x1 - 2))) :
  q = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_problem_l4016_401692


namespace NUMINAMATH_CALUDE_exists_non_intersecting_line_l4016_401609

/-- Represents a domino on a grid -/
structure Domino where
  x1 : Nat
  y1 : Nat
  x2 : Nat
  y2 : Nat

/-- Represents a 6x6 grid covered by dominoes -/
structure DominoGrid where
  dominoes : List Domino
  domino_count : dominoes.length = 18
  covers_grid : ∀ x y, x < 6 ∧ y < 6 → ∃ d ∈ dominoes, 
    ((d.x1 = x ∧ d.y1 = y) ∨ (d.x2 = x ∧ d.y2 = y))
  valid_dominoes : ∀ d ∈ dominoes, 
    (d.x1 = d.x2 ∧ d.y2 = d.y1 + 1) ∨ (d.y1 = d.y2 ∧ d.x2 = d.x1 + 1)

/-- Main theorem: There exists a grid line not intersecting any domino -/
theorem exists_non_intersecting_line (grid : DominoGrid) :
  (∃ x : Nat, x < 5 ∧ ∀ d ∈ grid.dominoes, d.x1 ≠ x + 1 ∨ d.x2 ≠ x + 1) ∨
  (∃ y : Nat, y < 5 ∧ ∀ d ∈ grid.dominoes, d.y1 ≠ y + 1 ∨ d.y2 ≠ y + 1) :=
sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_line_l4016_401609


namespace NUMINAMATH_CALUDE_interval_cardinality_equal_l4016_401669

/-- Two sets are equinumerous if there exists a bijection between them -/
def Equinumerous (α β : Type*) : Prop :=
  ∃ f : α → β, Function.Bijective f

theorem interval_cardinality_equal (a b : ℝ) (h : a < b) :
  Equinumerous (Set.Icc a b) (Set.Ioo a b) ∧
  Equinumerous (Set.Icc a b) (Set.Ico a b) ∧
  Equinumerous (Set.Icc a b) (Set.Ioc a b) :=
sorry

end NUMINAMATH_CALUDE_interval_cardinality_equal_l4016_401669


namespace NUMINAMATH_CALUDE_inequality_proof_l4016_401697

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8 / (x * y) + y^2 ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4016_401697


namespace NUMINAMATH_CALUDE_max_square_pen_area_l4016_401684

def fencing_length : ℝ := 64

def square_pen_area (side_length : ℝ) : ℝ := side_length ^ 2

def perimeter_constraint (side_length : ℝ) : Prop := 4 * side_length = fencing_length

theorem max_square_pen_area :
  ∃ (side_length : ℝ), perimeter_constraint side_length ∧
    ∀ (x : ℝ), perimeter_constraint x → square_pen_area x ≤ square_pen_area side_length ∧
    square_pen_area side_length = 256 :=
  sorry

end NUMINAMATH_CALUDE_max_square_pen_area_l4016_401684


namespace NUMINAMATH_CALUDE_jorge_total_goals_l4016_401606

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorge_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorge_total_goals_l4016_401606


namespace NUMINAMATH_CALUDE_cauchy_schwarz_2d_l4016_401640

theorem cauchy_schwarz_2d {a b c d : ℝ} :
  (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 ∧
  ((a^2 + b^2) * (c^2 + d^2) = (a*c + b*d)^2 ↔ a*d = b*c) :=
sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_2d_l4016_401640


namespace NUMINAMATH_CALUDE_binomial_60_3_l4016_401675

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l4016_401675


namespace NUMINAMATH_CALUDE_shipwreck_age_conversion_l4016_401699

theorem shipwreck_age_conversion : 
  (7 * 8^2 + 4 * 8^1 + 2 * 8^0 : ℕ) = 482 := by
  sorry

end NUMINAMATH_CALUDE_shipwreck_age_conversion_l4016_401699


namespace NUMINAMATH_CALUDE_simplify_cube_divided_by_base_l4016_401659

theorem simplify_cube_divided_by_base (x y : ℝ) : (x + y)^3 / (x + y) = (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_cube_divided_by_base_l4016_401659


namespace NUMINAMATH_CALUDE_moving_sidewalk_speed_l4016_401661

/-- The speed of a moving sidewalk given a child's running parameters -/
theorem moving_sidewalk_speed
  (child_speed : ℝ)
  (with_distance : ℝ)
  (with_time : ℝ)
  (against_distance : ℝ)
  (against_time : ℝ)
  (h1 : child_speed = 74)
  (h2 : with_distance = 372)
  (h3 : with_time = 4)
  (h4 : against_distance = 165)
  (h5 : against_time = 3)
  : ∃ (sidewalk_speed : ℝ),
    sidewalk_speed = 19 ∧
    with_distance = (child_speed + sidewalk_speed) * with_time ∧
    against_distance = (child_speed - sidewalk_speed) * against_time :=
by sorry

end NUMINAMATH_CALUDE_moving_sidewalk_speed_l4016_401661


namespace NUMINAMATH_CALUDE_money_ratio_proof_l4016_401655

def josh_money (doug_money : ℚ) : ℚ := (3 / 4) * doug_money

theorem money_ratio_proof (doug_money : ℚ) 
  (h1 : josh_money doug_money + doug_money + 12 = 68) : 
  josh_money doug_money / 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l4016_401655


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l4016_401663

theorem parabola_vertex_on_x_axis (a : ℝ) : 
  (∃ x : ℝ, x^2 - (a + 2)*x + 9 = 0 ∧ 
   ∀ y : ℝ, y^2 - (a + 2)*y + 9 ≥ x^2 - (a + 2)*x + 9) →
  a = 4 ∨ a = -8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l4016_401663


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l4016_401696

/-- A rhombus with side length 51 and shorter diagonal 48 has a longer diagonal of 90 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 51 → shorter_diag = 48 → longer_diag = 90 → 
  side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2 := by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l4016_401696


namespace NUMINAMATH_CALUDE_odd_function_property_l4016_401621

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_prop : ∀ x, f (1 + x) = f (-x)) 
  (h_value : f (-1/3) = 1/3) : 
  f (5/3) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_odd_function_property_l4016_401621


namespace NUMINAMATH_CALUDE_max_trig_ratio_l4016_401630

theorem max_trig_ratio (x : ℝ) : 
  (Real.sin x)^2 + (Real.cos x)^2 = 1 → 
  ((Real.sin x)^4 + (Real.cos x)^4 + 1) / ((Real.sin x)^2 + (Real.cos x)^2 + 1) ≤ 7/4 := by
sorry

end NUMINAMATH_CALUDE_max_trig_ratio_l4016_401630


namespace NUMINAMATH_CALUDE_rearrange_three_of_eight_count_l4016_401608

/-- The number of ways to select and rearrange 3 people out of 8 -/
def rearrange_three_of_eight : ℕ :=
  Nat.choose 8 3 * (3 * 2)

/-- Theorem stating that rearranging 3 people out of 8 has C₈₃ * A³₂ ways -/
theorem rearrange_three_of_eight_count :
  rearrange_three_of_eight = Nat.choose 8 3 * (3 * 2) := by
  sorry

end NUMINAMATH_CALUDE_rearrange_three_of_eight_count_l4016_401608


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l4016_401612

/-- The distance between two parallel lines -/
theorem parallel_lines_distance (a b c d e f : ℝ) :
  (a = 1 ∧ b = 2 ∧ c = -1) →
  (d = 2 ∧ e = 4 ∧ f = 3) →
  (∃ (k : ℝ), k ≠ 0 ∧ d = k * a ∧ e = k * b) →
  (abs (f / d - c / a) / Real.sqrt (a^2 + b^2) : ℝ) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l4016_401612


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l4016_401693

theorem square_area_from_perimeter (perimeter : ℝ) (h : perimeter = 52) :
  let side_length := perimeter / 4
  let area := side_length ^ 2
  area = 169 := by sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l4016_401693


namespace NUMINAMATH_CALUDE_least_positive_integer_with_four_distinct_primes_l4016_401665

def is_prime (n : ℕ) : Prop := sorry

def distinct_prime_factors (n : ℕ) : Prop := sorry

theorem least_positive_integer_with_four_distinct_primes :
  ∃ (n : ℕ), n > 0 ∧ distinct_prime_factors n ∧ (∀ m : ℕ, m > 0 → distinct_prime_factors m → n ≤ m) :=
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_four_distinct_primes_l4016_401665


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l4016_401687

theorem sqrt_sum_comparison : Real.sqrt 11 + Real.sqrt 7 > Real.sqrt 13 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l4016_401687


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l4016_401629

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 14 = 56 → Nat.gcd n 14 = 10 → n = 40 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l4016_401629


namespace NUMINAMATH_CALUDE_count_nonzero_terms_l4016_401691

/-- The number of nonzero terms in the simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def nonzero_terms : ℕ := 56883810

/-- The exponent used in the expression -/
def exponent : ℕ := 2008

theorem count_nonzero_terms (a b c : ℤ) :
  nonzero_terms = (exponent + 3).choose 3 := by sorry

end NUMINAMATH_CALUDE_count_nonzero_terms_l4016_401691


namespace NUMINAMATH_CALUDE_f_inequality_solution_comparison_theorem_l4016_401602

def f (x : ℝ) : ℝ := -abs x - abs (x + 2)

theorem f_inequality_solution (x : ℝ) : f x < -4 ↔ x < -3 ∨ x > 1 := by sorry

theorem comparison_theorem (x a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = Real.sqrt 5) :
  a^2 + b^2/4 ≥ f x + 3 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_comparison_theorem_l4016_401602


namespace NUMINAMATH_CALUDE_complex_fraction_squared_difference_l4016_401618

theorem complex_fraction_squared_difference (a b : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I) / (1 + Complex.I) = a + b * Complex.I →
  a^2 - b^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_squared_difference_l4016_401618


namespace NUMINAMATH_CALUDE_second_mission_duration_l4016_401646

def planned_duration : ℕ := 5
def actual_duration_increase : ℚ := 60 / 100
def total_mission_time : ℕ := 11

theorem second_mission_duration :
  let actual_first_mission := planned_duration + (planned_duration * actual_duration_increase).floor
  let second_mission := total_mission_time - actual_first_mission
  second_mission = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_mission_duration_l4016_401646


namespace NUMINAMATH_CALUDE_min_quotient_l4016_401650

/-- A three-digit number with distinct non-zero digits that sum to 10 -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0
  h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c
  h3 : a + b + c = 10

/-- The quotient of the number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : ℚ :=
  (100 * n.a + 10 * n.b + n.c) / (n.a + n.b + n.c)

/-- The minimum value of the quotient is 12.7 -/
theorem min_quotient :
  ∀ n : ThreeDigitNumber, quotient n ≥ 127/10 ∧ ∃ m : ThreeDigitNumber, quotient m = 127/10 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_l4016_401650


namespace NUMINAMATH_CALUDE_edge_probability_is_one_l4016_401615

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Checks if a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop direction -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, staying in bounds -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨min 3 (p.x + 1), p.y⟩
  | Direction.Down => ⟨max 0 (p.x - 1), p.y⟩
  | Direction.Left => ⟨p.x, max 0 (p.y - 1)⟩
  | Direction.Right => ⟨p.x, min 3 (p.y + 1)⟩

/-- The starting position (2,2) -/
def startPos : Position := ⟨2, 2⟩

/-- Theorem: The probability of reaching an edge cell within three hops from (2,2) is 1 -/
theorem edge_probability_is_one :
  ∀ (hops : List Direction),
    hops.length ≤ 3 →
    isEdge (hops.foldl hop startPos) = true :=
by sorry

end NUMINAMATH_CALUDE_edge_probability_is_one_l4016_401615


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l4016_401634

theorem p_necessary_not_sufficient_for_q (a b : ℝ) :
  (∀ a b, a^2 + b^2 ≠ 0 → a * b = 0) ∧
  ¬(∀ a b, a * b = 0 → a^2 + b^2 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l4016_401634


namespace NUMINAMATH_CALUDE_problem_statement_l4016_401673

def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

theorem problem_statement (m : ℝ) (h : m > 0) :
  (∀ x, p x → q x m) → m ∈ Set.Ici 4 ∧
  (m = 5 → ∀ x, (p x ∨ q x m) ∧ ¬(p x ∧ q x m) → 
    x ∈ Set.Ioc (-4) (-1) ∪ Set.Ioo 5 6) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4016_401673


namespace NUMINAMATH_CALUDE_xy_value_l4016_401633

theorem xy_value (x y : ℝ) (h : x^2 + y^2 + 4*x - 6*y + 13 = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l4016_401633


namespace NUMINAMATH_CALUDE_candy_bar_chocolate_cost_difference_l4016_401653

/-- The problem of calculating the difference in cost between a candy bar and chocolate. -/
theorem candy_bar_chocolate_cost_difference :
  let dans_money : ℕ := 2
  let candy_bar_cost : ℕ := 6
  let chocolate_cost : ℕ := 3
  candy_bar_cost - chocolate_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_chocolate_cost_difference_l4016_401653


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l4016_401694

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, 0 < x ∧ x < 5 → -5 < x - 2 ∧ x - 2 < 5) ∧
  (∃ x, -5 < x - 2 ∧ x - 2 < 5 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l4016_401694


namespace NUMINAMATH_CALUDE_sequence_proof_l4016_401614

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sequence_proof
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a2 : a 2 = -1)
  (h_b_diff : ∀ n : ℕ, n ≥ 2 → b n - b (n - 1) = a n)
  (h_b1 : b 1 = 1)
  (h_b3 : b 3 = 1) :
  (a 1 = -3) ∧
  (∀ n : ℕ, n ≥ 1 → b n = n^2 - 4*n + 4) :=
by sorry

end NUMINAMATH_CALUDE_sequence_proof_l4016_401614


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l4016_401600

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 5}
def B : Set ℝ := {x : ℝ | x^2 ≥ 4}

-- State the theorem
theorem complement_of_A_union_B :
  (A ∪ B)ᶜ = Ioc (-2) (-1) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l4016_401600


namespace NUMINAMATH_CALUDE_percentage_increase_l4016_401639

theorem percentage_increase (original : ℝ) (final : ℝ) (percentage : ℝ) : 
  original = 900 →
  final = 1080 →
  percentage = ((final - original) / original) * 100 →
  percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l4016_401639


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4016_401610

def A : Set ℝ := {x : ℝ | |x| ≤ 1}
def B : Set ℝ := {x : ℝ | x ≤ 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 ≤ x ∧ x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4016_401610


namespace NUMINAMATH_CALUDE_prob_five_dice_three_matching_l4016_401627

/-- The probability of rolling at least three matching dice out of five fair six-sided dice -/
def prob_at_least_three_matching (n : ℕ) (s : ℕ) : ℚ :=
  -- n is the number of dice
  -- s is the number of sides on each die
  sorry

/-- Theorem stating that the probability of rolling at least three matching dice
    out of five fair six-sided dice is equal to 23/108 -/
theorem prob_five_dice_three_matching :
  prob_at_least_three_matching 5 6 = 23 / 108 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_dice_three_matching_l4016_401627


namespace NUMINAMATH_CALUDE_min_value_d_l4016_401686

def a (n : ℕ+) : ℚ := 1000 / n
def b (n k : ℕ+) : ℚ := 2000 / (k * n)
def c (n k : ℕ+) : ℚ := 1500 / (200 - n - k * n)
def d (n k : ℕ+) : ℚ := max (a n) (max (b n k) (c n k))

theorem min_value_d (n k : ℕ+) (h : n + k * n < 200) :
  ∃ (n₀ k₀ : ℕ+), d n₀ k₀ = 250 / 11 ∧ ∀ (n' k' : ℕ+), n' + k' * n' < 200 → d n' k' ≥ 250 / 11 :=
sorry

end NUMINAMATH_CALUDE_min_value_d_l4016_401686


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l4016_401648

/-- Represents the price reduction problem for a shopping mall selling shirts. -/
theorem shirt_price_reduction
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (price_reduction_effect : ℝ → ℕ)
  (target_profit : ℝ)
  (price_reduction : ℝ)
  (h1 : initial_sales = 20)
  (h2 : initial_profit = 40)
  (h3 : ∀ x, price_reduction_effect x = initial_sales + 2 * ⌊x⌋)
  (h4 : target_profit = 1200)
  (h5 : price_reduction = 20) :
  (initial_profit - price_reduction) * price_reduction_effect price_reduction = target_profit :=
sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l4016_401648


namespace NUMINAMATH_CALUDE_largest_last_digit_l4016_401680

def is_valid_digit_string (s : List Nat) : Prop :=
  s.length = 1001 ∧ 
  s.head? = some 3 ∧
  ∀ i, i < 1000 → (s[i]! * 10 + s[i+1]!) % 17 = 0 ∨ (s[i]! * 10 + s[i+1]!) % 23 = 0

theorem largest_last_digit (s : List Nat) (h : is_valid_digit_string s) : 
  s[1000]! ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_l4016_401680


namespace NUMINAMATH_CALUDE_external_tangent_y_intercept_l4016_401668

/-- Given two circles with centers and radii as specified, 
    prove that their common external tangent with positive slope 
    has a y-intercept of 740/171 -/
theorem external_tangent_y_intercept :
  let c1 : ℝ × ℝ := (1, 3)  -- Center of circle 1
  let r1 : ℝ := 3           -- Radius of circle 1
  let c2 : ℝ × ℝ := (15, 8) -- Center of circle 2
  let r2 : ℝ := 10          -- Radius of circle 2
  let m : ℝ := (140 : ℝ) / 171 -- Slope of the tangent line (positive)
  let b : ℝ := (740 : ℝ) / 171 -- y-intercept to be proved
  let tangent_line (x : ℝ) := m * x + b -- Equation of the tangent line
  (∀ x y : ℝ, (x - c1.1)^2 + (y - c1.2)^2 = r1^2 → 
    (tangent_line x - y)^2 ≥ (r1 * m)^2) ∧ 
  (∀ x y : ℝ, (x - c2.1)^2 + (y - c2.2)^2 = r2^2 → 
    (tangent_line x - y)^2 ≥ (r2 * m)^2) ∧
  (∃ x1 y1 x2 y2 : ℝ, 
    (x1 - c1.1)^2 + (y1 - c1.2)^2 = r1^2 ∧
    (x2 - c2.1)^2 + (y2 - c2.2)^2 = r2^2 ∧
    tangent_line x1 = y1 ∧ tangent_line x2 = y2) :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_y_intercept_l4016_401668


namespace NUMINAMATH_CALUDE_jelly_bean_probability_l4016_401689

theorem jelly_bean_probability (p_red p_orange p_blue p_yellow : ℝ) :
  p_red = 0.25 →
  p_orange = 0.4 →
  p_blue = 0.15 →
  p_red + p_orange + p_blue + p_yellow = 1 →
  p_yellow = 0.2 := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_probability_l4016_401689


namespace NUMINAMATH_CALUDE_distribute_3_4_l4016_401698

/-- The number of ways to distribute n distinct objects into m distinct containers -/
def distribute (n m : ℕ) : ℕ := m^n

/-- Theorem: Distributing 3 distinct objects into 4 distinct containers results in 64 ways -/
theorem distribute_3_4 : distribute 3 4 = 64 := by
  sorry

end NUMINAMATH_CALUDE_distribute_3_4_l4016_401698


namespace NUMINAMATH_CALUDE_operation_result_l4016_401682

def at_op (a b : ℝ) : ℝ := a * b - b^2 + b^3

def hash_op (a b : ℝ) : ℝ := a + b - a * b^2 + a * b^3

theorem operation_result : (at_op 7 3) / (hash_op 7 3) = 39 / 136 := by
  sorry

end NUMINAMATH_CALUDE_operation_result_l4016_401682


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l4016_401635

/-- Given Ethan's rowing time and the total rowing time for both Ethan and Frank,
    prove that the ratio of Frank's rowing time to Ethan's rowing time is 2:1. -/
theorem rowing_time_ratio 
  (ethan_time : ℕ) 
  (total_time : ℕ) 
  (h1 : ethan_time = 25)
  (h2 : total_time = 75) :
  (total_time - ethan_time) / ethan_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l4016_401635


namespace NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l4016_401616

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 30 sides has 405 diagonals -/
theorem diagonals_30_sided_polygon : num_diagonals 30 = 405 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_30_sided_polygon_l4016_401616


namespace NUMINAMATH_CALUDE_last_digit_3_2004_l4016_401652

/-- The last digit of 3^n -/
def last_digit (n : ℕ) : ℕ :=
  (3^n) % 10

/-- The pattern of last digits repeats every 4 steps -/
axiom last_digit_pattern (n : ℕ) :
  last_digit n = last_digit (n % 4)

/-- The last digits for the first 4 powers of 3 -/
axiom last_digit_base :
  last_digit 0 = 1 ∧ 
  last_digit 1 = 3 ∧ 
  last_digit 2 = 9 ∧ 
  last_digit 3 = 7

theorem last_digit_3_2004 :
  last_digit 2004 = 1 :=
by sorry

end NUMINAMATH_CALUDE_last_digit_3_2004_l4016_401652


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l4016_401688

theorem helga_shoe_shopping (x : ℕ) : 
  (x + (x + 2) + 0 + 2 * (x + (x + 2) + 0) = 48) → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l4016_401688


namespace NUMINAMATH_CALUDE_total_books_bought_l4016_401605

/-- Proves that the total number of books bought is 90 -/
theorem total_books_bought (total_price : ℕ) (math_book_price : ℕ) (history_book_price : ℕ) (math_books_count : ℕ) :
  total_price = 390 →
  math_book_price = 4 →
  history_book_price = 5 →
  math_books_count = 60 →
  math_books_count + (total_price - math_books_count * math_book_price) / history_book_price = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_total_books_bought_l4016_401605


namespace NUMINAMATH_CALUDE_digit_sum_property_l4016_401626

def S (k : ℕ) : ℕ := (k.repr.toList.map (λ c => c.toNat - 48)).sum

theorem digit_sum_property (n : ℕ) : 
  (∃ (a b : ℕ), n = S a ∧ n = S b ∧ n = S (a + b)) ↔ 
  (∃ (k : ℕ+), n = 9 * k) :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l4016_401626


namespace NUMINAMATH_CALUDE_simple_interest_rate_l4016_401685

/-- Given a principal amount and a time period of 10 years, 
    if the simple interest is 7/5 of the principal, 
    then the annual interest rate is 14%. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  (P * 14 * 10) / 100 = (7 / 5) * P := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l4016_401685


namespace NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l4016_401641

theorem complex_magnitude_two_thirds_minus_four_fifths_i :
  Complex.abs (2/3 - 4/5 * Complex.I) = Real.sqrt 244 / 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_two_thirds_minus_four_fifths_i_l4016_401641


namespace NUMINAMATH_CALUDE_range_of_a_l4016_401645

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x > a}
def B : Set ℝ := {-1, 0, 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : A a ∩ B = {0, 1} → a ∈ Set.Icc (-1) 0 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4016_401645


namespace NUMINAMATH_CALUDE_inequality_solution_and_bound_l4016_401636

def f (x : ℝ) := |x - 3|
def g (x : ℝ) := |x - 2|

theorem inequality_solution_and_bound :
  (∀ x, f x + g x < 2 ↔ x ∈ Set.Ioo (3/2) (7/2)) ∧
  (∀ x y, f x ≤ 1 → g y ≤ 1 → |x - 2*y + 1| ≤ 3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_and_bound_l4016_401636


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4016_401664

theorem arithmetic_sequence_problem :
  ∀ (a b c d : ℝ),
    (a + b + c + d = 26) →
    (b * c = 40) →
    (c - b = b - a) →
    (d - c = c - b) →
    ((a = 2 ∧ b = 5 ∧ c = 8 ∧ d = 11) ∨ (a = 11 ∧ b = 8 ∧ c = 5 ∧ d = 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l4016_401664


namespace NUMINAMATH_CALUDE_mets_fan_count_l4016_401658

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  redsox : ℕ

/-- The conditions of the problem -/
def fan_conditions (f : FanCounts) : Prop :=
  (f.yankees : ℚ) / f.mets = 3 / 2 ∧
  (f.mets : ℚ) / f.redsox = 4 / 5 ∧
  f.yankees + f.mets + f.redsox = 330

/-- The theorem to be proved -/
theorem mets_fan_count (f : FanCounts) :
  fan_conditions f → f.mets = 88 := by
  sorry

end NUMINAMATH_CALUDE_mets_fan_count_l4016_401658


namespace NUMINAMATH_CALUDE_free_throw_contest_total_l4016_401647

/-- Given a free throw contest where:
  * Alex made 8 baskets
  * Sandra made three times as many baskets as Alex
  * Hector made two times the number of baskets that Sandra made
  Prove that the total number of baskets made by all three is 80. -/
theorem free_throw_contest_total (alex_baskets : ℕ) (sandra_baskets : ℕ) (hector_baskets : ℕ) 
  (h1 : alex_baskets = 8)
  (h2 : sandra_baskets = 3 * alex_baskets)
  (h3 : hector_baskets = 2 * sandra_baskets) :
  alex_baskets + sandra_baskets + hector_baskets = 80 := by
  sorry

#check free_throw_contest_total

end NUMINAMATH_CALUDE_free_throw_contest_total_l4016_401647


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4016_401670

-- Define the operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the imaginary unit
def i : ℂ := Complex.I

-- Define z using the operation
def z : ℂ := det 1 2 i (i^4)

-- Define the fourth quadrant
def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant : fourth_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4016_401670
