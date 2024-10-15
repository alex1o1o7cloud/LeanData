import Mathlib

namespace NUMINAMATH_CALUDE_cos_45_sin_15_minus_sin_45_cos_15_l3757_375738

theorem cos_45_sin_15_minus_sin_45_cos_15 :
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) - 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_sin_15_minus_sin_45_cos_15_l3757_375738


namespace NUMINAMATH_CALUDE_gauss_family_mean_age_l3757_375724

def gauss_family_ages : List ℕ := [8, 8, 8, 8, 16, 17]

theorem gauss_family_mean_age : 
  (gauss_family_ages.sum : ℚ) / gauss_family_ages.length = 65 / 6 := by
  sorry

end NUMINAMATH_CALUDE_gauss_family_mean_age_l3757_375724


namespace NUMINAMATH_CALUDE_donald_drinks_nine_l3757_375775

/-- The number of juice bottles Paul drinks per day -/
def paul_bottles : ℕ := 3

/-- The number of juice bottles Donald drinks per day -/
def donald_bottles : ℕ := 2 * paul_bottles + 3

/-- Theorem stating that Donald drinks 9 bottles of juice per day -/
theorem donald_drinks_nine : donald_bottles = 9 := by
  sorry

end NUMINAMATH_CALUDE_donald_drinks_nine_l3757_375775


namespace NUMINAMATH_CALUDE_base_conversion_sum_l3757_375704

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_conversion_sum : 
  let base9 := toBase10 [1, 2, 3] 9
  let base8 := toBase10 [6, 5, 2] 8
  let base7 := toBase10 [4, 3, 1] 7
  base9 - base8 + base7 = 162 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l3757_375704


namespace NUMINAMATH_CALUDE_expression_evaluation_l3757_375748

theorem expression_evaluation : 
  let a := (1/4 + 1/12 - 7/18 - 1/36 : ℚ)
  let part1 := (1/36 : ℚ) / a
  let part2 := a / (1/36 : ℚ)
  part1 * part2 = 1 → part1 + part2 = -10/3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3757_375748


namespace NUMINAMATH_CALUDE_fish_tank_problem_l3757_375712

theorem fish_tank_problem (tank1_size : ℚ) (tank2_size : ℚ) (tank1_water : ℚ) 
  (fish2_length : ℚ) (fish_diff : ℕ) :
  tank1_size = 2 * tank2_size →
  tank1_water = 48 →
  fish2_length = 2 →
  fish_diff = 3 →
  ∃ (fish1_length : ℚ),
    fish1_length = 3 ∧
    (tank1_water / fish1_length - 1 = tank2_size / fish2_length + fish_diff) :=
by sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l3757_375712


namespace NUMINAMATH_CALUDE_problem_figure_perimeter_l3757_375772

/-- Represents the figure described in the problem -/
structure SquareFigure where
  stackHeight : Nat
  stackGap : Nat
  topSquares : Nat
  bottomSquares : Nat

/-- Calculates the perimeter of the square figure -/
def perimeterOfSquareFigure (fig : SquareFigure) : Nat :=
  let horizontalSegments := fig.topSquares * 2 + fig.bottomSquares * 2
  let verticalSegments := fig.stackHeight * 2 * 2 + fig.topSquares * 2
  horizontalSegments + verticalSegments

/-- The specific figure described in the problem -/
def problemFigure : SquareFigure :=
  { stackHeight := 3
  , stackGap := 1
  , topSquares := 3
  , bottomSquares := 2 }

theorem problem_figure_perimeter :
  perimeterOfSquareFigure problemFigure = 22 := by
  sorry

end NUMINAMATH_CALUDE_problem_figure_perimeter_l3757_375772


namespace NUMINAMATH_CALUDE_age_proof_l3757_375771

/-- Proves the ages of Desiree and her cousin given the conditions -/
theorem age_proof (desiree_age : ℝ) (cousin_age : ℝ) : 
  desiree_age = 2.99999835 →
  cousin_age = 1.499999175 →
  desiree_age = 2 * cousin_age →
  desiree_age + 30 = 0.6666666 * (cousin_age + 30) + 14 :=
by sorry

end NUMINAMATH_CALUDE_age_proof_l3757_375771


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3757_375781

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3757_375781


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l3757_375716

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  e : ℝ

-- Define the conditions
def hyperbola_conditions (h : Hyperbola) : Prop :=
  h.e = Real.sqrt 3 ∧
  h.c = Real.sqrt 3 * h.a ∧
  h.b = Real.sqrt 2 * h.a ∧
  8 / h.a^2 - 1 / h.a^2 = 1

-- Define the standard equation
def standard_equation (h : Hyperbola) : Prop :=
  ∀ (x y : ℝ), x^2 / 7 - y^2 / 14 = 1 ↔ x^2 / h.a^2 - y^2 / h.b^2 = 1

-- Theorem statement
theorem hyperbola_standard_equation (h : Hyperbola) :
  hyperbola_conditions h → standard_equation h :=
sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l3757_375716


namespace NUMINAMATH_CALUDE_max_slope_product_l3757_375725

theorem max_slope_product (m₁ m₂ : ℝ) : 
  (m₁ = 5 * m₂) →                    -- One slope is 5 times the other
  (|((m₂ - m₁) / (1 + m₁ * m₂))| = 1) →  -- Lines intersect at 45° angle
  (∀ n₁ n₂ : ℝ, (n₁ = 5 * n₂) → (|((n₂ - n₁) / (1 + n₁ * n₂))| = 1) → m₁ * m₂ ≥ n₁ * n₂) →
  m₁ * m₂ = 1.8 :=
by sorry

end NUMINAMATH_CALUDE_max_slope_product_l3757_375725


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l3757_375740

theorem smallest_distance_between_circles (z w : ℂ) :
  Complex.abs (z - (2 + 4*I)) = 2 →
  Complex.abs (w - (5 + 6*I)) = 4 →
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 + 4*I)) = 2 →
      Complex.abs (w' - (5 + 6*I)) = 4 →
      Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l3757_375740


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l3757_375756

theorem greatest_integer_quadratic_inequality :
  ∀ n : ℤ, n^2 - 13*n + 30 < 0 → n ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l3757_375756


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l3757_375711

/-- Jake's current weight in pounds -/
def jakes_weight : ℕ := 156

/-- Jake's weight after losing 20 pounds -/
def jakes_reduced_weight : ℕ := jakes_weight - 20

/-- Jake's sister's weight in pounds -/
def sisters_weight : ℕ := jakes_reduced_weight / 2

/-- The combined weight of Jake and his sister -/
def combined_weight : ℕ := jakes_weight + sisters_weight

theorem jake_and_sister_weight : combined_weight = 224 := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l3757_375711


namespace NUMINAMATH_CALUDE_min_area_triangle_l3757_375776

theorem min_area_triangle (m n : ℝ) : 
  let l := {(x, y) : ℝ × ℝ | m * x + n * y - 1 = 0}
  let A := (1/m, 0)
  let B := (0, 1/n)
  let O := (0, 0)
  (∀ (x y : ℝ), (x, y) ∈ l → |m * x + n * y - 1| / Real.sqrt (m^2 + n^2) = Real.sqrt 3) →
  (∃ (S : ℝ), S = (1/2) * |1/m * 1/n| ∧ 
    (∀ (S' : ℝ), S' = (1/2) * |1/m' * 1/n'| → 
      m'^2 + n'^2 = 1/3 → S ≤ S') ∧ S = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_area_triangle_l3757_375776


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3757_375710

theorem trigonometric_identities (α : Real) 
  (h : (Real.tan α) / (Real.tan α - 1) = -1) : 
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + 2 * Real.cos α) = -1 ∧ 
  (Real.sin (π - α) * Real.cos (π + α) * Real.cos (π/2 + α) * Real.cos (π/2 - α)) / 
  (Real.cos (π - α) * Real.sin (3*π - α) * Real.sin (-π - α) * Real.sin (π/2 + α)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3757_375710


namespace NUMINAMATH_CALUDE_find_k_value_l3757_375713

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) 
  (h2 : ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3)) : 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_value_l3757_375713


namespace NUMINAMATH_CALUDE_red_to_blue_ratio_l3757_375731

/-- Represents the number of beads of each color in Michelle's necklace. -/
structure Necklace where
  total : ℕ
  blue : ℕ
  red : ℕ
  white : ℕ
  silver : ℕ

/-- The conditions of Michelle's necklace. -/
def michelle_necklace : Necklace where
  total := 40
  blue := 5
  red := 10  -- This is derived, not given directly
  white := 15 -- This is derived, not given directly
  silver := 10

/-- The ratio of red beads to blue beads is 2:1. -/
theorem red_to_blue_ratio (n : Necklace) (h1 : n = michelle_necklace) 
    (h2 : n.white = n.blue + n.red) 
    (h3 : n.total = n.blue + n.red + n.white + n.silver) : 
  n.red / n.blue = 2 := by
  sorry

#check red_to_blue_ratio

end NUMINAMATH_CALUDE_red_to_blue_ratio_l3757_375731


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l3757_375785

theorem unique_two_digit_integer (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (11 * t) % 100 = 36 ↔ t = 76 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l3757_375785


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3757_375769

theorem quadratic_equation_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (m - 1) * x^2 - 2*x - 1 = 0 ∧ (m - 1) * y^2 - 2*y - 1 = 0) ↔ 
  (m ≥ 0 ∧ m ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3757_375769


namespace NUMINAMATH_CALUDE_lawn_mowing_problem_l3757_375780

theorem lawn_mowing_problem (mary_time tom_time tom_work_time : ℝ) 
  (h1 : mary_time = 6)
  (h2 : tom_time = 4)
  (h3 : tom_work_time = 3) :
  1 - (tom_work_time / tom_time) = 1/4 := by sorry

end NUMINAMATH_CALUDE_lawn_mowing_problem_l3757_375780


namespace NUMINAMATH_CALUDE_permit_increase_l3757_375722

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of letters in old permits -/
def old_permit_letters : ℕ := 2

/-- The number of digits in old permits -/
def old_permit_digits : ℕ := 3

/-- The number of letters in new permits -/
def new_permit_letters : ℕ := 4

/-- The number of digits in new permits -/
def new_permit_digits : ℕ := 4

/-- The ratio of new permits to old permits -/
def permit_ratio : ℕ := 67600

theorem permit_increase :
  (alphabet_size ^ new_permit_letters * digit_count ^ new_permit_digits) /
  (alphabet_size ^ old_permit_letters * digit_count ^ old_permit_digits) = permit_ratio :=
sorry

end NUMINAMATH_CALUDE_permit_increase_l3757_375722


namespace NUMINAMATH_CALUDE_largest_integer_solution_negative_six_is_largest_largest_integer_is_negative_six_l3757_375773

theorem largest_integer_solution (x : ℤ) : (7 - 3 * x > 22) ↔ (x ≤ -6) :=
  sorry

theorem negative_six_is_largest : ∃ (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22)) :=
  sorry

theorem largest_integer_is_negative_six : (∃! (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22))) ∧ 
  (∀ (x : ℤ), (7 - 3 * x > 22) ∧ (∀ (y : ℤ), y > x → ¬(7 - 3 * y > 22)) → x = -6) :=
  sorry

end NUMINAMATH_CALUDE_largest_integer_solution_negative_six_is_largest_largest_integer_is_negative_six_l3757_375773


namespace NUMINAMATH_CALUDE_fraction_multiplication_addition_l3757_375701

theorem fraction_multiplication_addition : (2 : ℚ) / 9 * 5 / 8 + 1 / 4 = 7 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_addition_l3757_375701


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3757_375715

/-- Given p, q, and r are the roots of x^3 - 8x^2 + 6x - 3 = 0,
    prove that p/(qr - 1) + q/(pr - 1) + r/(pq - 1) = 21.75 -/
theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 6*p - 3 = 0 → 
  q^3 - 8*q^2 + 6*q - 3 = 0 → 
  r^3 - 8*r^2 + 6*r - 3 = 0 → 
  p/(q*r - 1) + q/(p*r - 1) + r/(p*q - 1) = 21.75 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3757_375715


namespace NUMINAMATH_CALUDE_line_projections_parallel_implies_parallel_or_skew_l3757_375759

/-- Two lines in 3D space -/
structure Line3D where
  -- Add necessary fields for a 3D line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a 3D plane

/-- Projection of a line onto a plane -/
def project_line (l : Line3D) (p : Plane3D) : Line3D :=
  sorry

/-- Predicate for parallel lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate for skew lines -/
def skew (l1 l2 : Line3D) : Prop :=
  sorry

theorem line_projections_parallel_implies_parallel_or_skew 
  (a b : Line3D) (α : Plane3D) :
  parallel (project_line a α) (project_line b α) →
  parallel a b ∨ skew a b :=
sorry

end NUMINAMATH_CALUDE_line_projections_parallel_implies_parallel_or_skew_l3757_375759


namespace NUMINAMATH_CALUDE_philip_orange_collection_l3757_375746

/-- The number of oranges in Philip's collection -/
def num_oranges : ℕ := 178 * 2

/-- The number of groups of oranges -/
def orange_groups : ℕ := 178

/-- The number of oranges in each group -/
def oranges_per_group : ℕ := 2

/-- Theorem stating that the number of oranges in Philip's collection is 356 -/
theorem philip_orange_collection : num_oranges = 356 := by
  sorry

#eval num_oranges -- This will output 356

end NUMINAMATH_CALUDE_philip_orange_collection_l3757_375746


namespace NUMINAMATH_CALUDE_equilateral_triangle_between_poles_l3757_375762

theorem equilateral_triangle_between_poles (pole1 pole2 : ℝ) (h1 : pole1 = 11) (h2 : pole2 = 13) :
  let a := 8 * Real.sqrt 3
  (a ^ 2 = pole1 ^ 2 + 2 ^ 2) ∧
  (a ^ 2 = pole2 ^ 2 + 2 ^ 2) ∧
  (Real.sqrt (a ^ 2 - pole1 ^ 2) + Real.sqrt (a ^ 2 - pole2 ^ 2) = 2) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_between_poles_l3757_375762


namespace NUMINAMATH_CALUDE_battle_station_staffing_l3757_375750

theorem battle_station_staffing (total_resumes : ℕ) (suitable_fraction : ℚ) 
  (job_openings : ℕ) (h1 : total_resumes = 30) (h2 : suitable_fraction = 2/3) 
  (h3 : job_openings = 5) :
  (total_resumes : ℚ) * suitable_fraction * 
  (total_resumes : ℚ) * suitable_fraction - 1 * 
  (total_resumes : ℚ) * suitable_fraction - 2 * 
  (total_resumes : ℚ) * suitable_fraction - 3 * 
  (total_resumes : ℚ) * suitable_fraction - 4 = 930240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l3757_375750


namespace NUMINAMATH_CALUDE_max_value_expression_l3757_375749

theorem max_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^2 * y^2 * (x^2 + y^2) ≤ 2 ∧
  (x^2 * y^2 * (x^2 + y^2) = 2 ↔ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l3757_375749


namespace NUMINAMATH_CALUDE_weight_per_rep_l3757_375770

-- Define the given conditions
def reps_per_set : ℕ := 10
def num_sets : ℕ := 3
def total_weight : ℕ := 450

-- Define the theorem to prove
theorem weight_per_rep :
  total_weight / (reps_per_set * num_sets) = 15 := by
  sorry

end NUMINAMATH_CALUDE_weight_per_rep_l3757_375770


namespace NUMINAMATH_CALUDE_complex_number_problem_l3757_375796

-- Define the complex number Z₁
def Z₁ (a : ℝ) : ℂ := 2 + a * Complex.I

-- Main theorem
theorem complex_number_problem (a : ℝ) (ha : a > 0) 
  (h_pure_imag : ∃ b : ℝ, Z₁ a ^ 2 = b * Complex.I) :
  a = 2 ∧ Complex.abs (Z₁ a / (1 - Complex.I)) = 2 := by
  sorry


end NUMINAMATH_CALUDE_complex_number_problem_l3757_375796


namespace NUMINAMATH_CALUDE_division_of_fractions_l3757_375779

theorem division_of_fractions : (5 : ℚ) / 6 / (2 + 2 / 3) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l3757_375779


namespace NUMINAMATH_CALUDE_arithmetic_progression_poly_j_value_l3757_375783

/-- A polynomial of degree 4 with four distinct real roots in arithmetic progression -/
structure ArithmeticProgressionPoly where
  j : ℝ
  k : ℝ
  roots_distinct : True
  roots_real : True
  roots_arithmetic : True

/-- The value of j in the polynomial x^4 + jx^2 + kx + 81 with four distinct real roots in arithmetic progression is -10 -/
theorem arithmetic_progression_poly_j_value (p : ArithmeticProgressionPoly) : p.j = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_poly_j_value_l3757_375783


namespace NUMINAMATH_CALUDE_allocation_methods_six_individuals_l3757_375741

/-- The number of ways to allocate 6 individuals into 2 rooms -/
def allocation_methods (n : ℕ) : ℕ → ℕ
  | 1 => Nat.choose n 3  -- Exactly 3 per room
  | 2 => Nat.choose n 1 * Nat.choose (n-1) (n-1) +  -- 1 in first room
         Nat.choose n 2 * Nat.choose (n-2) (n-2) +  -- 2 in first room
         Nat.choose n 3 * Nat.choose (n-3) (n-3) +  -- 3 in first room
         Nat.choose n 4 * Nat.choose (n-4) (n-4) +  -- 4 in first room
         Nat.choose n 5 * Nat.choose (n-5) (n-5)    -- 5 in first room
  | _ => 0  -- For any other input

theorem allocation_methods_six_individuals :
  allocation_methods 6 1 = 20 ∧ allocation_methods 6 2 = 62 := by
  sorry

#eval allocation_methods 6 1  -- Should output 20
#eval allocation_methods 6 2  -- Should output 62

end NUMINAMATH_CALUDE_allocation_methods_six_individuals_l3757_375741


namespace NUMINAMATH_CALUDE_fraction_simplification_l3757_375791

theorem fraction_simplification : 1000^2 / (252^2 - 248^2) = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3757_375791


namespace NUMINAMATH_CALUDE_greatest_base_nine_digit_sum_l3757_375709

/-- The greatest possible sum of digits in base-nine representation of a positive integer less than 3000 -/
def max_base_nine_digit_sum : ℕ := 24

/-- Converts a natural number to its base-nine representation -/
def to_base_nine (n : ℕ) : List ℕ := sorry

/-- Calculates the sum of digits in a list -/
def digit_sum (digits : List ℕ) : ℕ := sorry

/-- Checks if a number is less than 3000 -/
def less_than_3000 (n : ℕ) : Prop := n < 3000

theorem greatest_base_nine_digit_sum :
  ∀ n : ℕ, less_than_3000 n → digit_sum (to_base_nine n) ≤ max_base_nine_digit_sum :=
by sorry

end NUMINAMATH_CALUDE_greatest_base_nine_digit_sum_l3757_375709


namespace NUMINAMATH_CALUDE_sum_of_numbers_greater_than_1_1_l3757_375797

def numbers : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]

theorem sum_of_numbers_greater_than_1_1 : 
  (numbers.filter (λ x => x > 1.1)).sum = 3.9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_greater_than_1_1_l3757_375797


namespace NUMINAMATH_CALUDE_simplify_expression_l3757_375794

theorem simplify_expression (y : ℝ) : 5*y + 6*y + 7*y + 2 = 18*y + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3757_375794


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3757_375766

theorem sum_of_coefficients (A B : ℝ) :
  (∀ x : ℝ, x ≠ 3 → A / (x - 3) + B * (x + 2) = (-4 * x^2 + 14 * x + 38) / (x - 3)) →
  A + B = 46 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3757_375766


namespace NUMINAMATH_CALUDE_equation_solution_l3757_375708

theorem equation_solution :
  ∃ x : ℚ, x + 5/8 = 2 + 3/16 - 2/3 ∧ x = 43/48 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3757_375708


namespace NUMINAMATH_CALUDE_smallest_valid_n_l3757_375784

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  let tens := n / 10
  let ones := n % 10
  2 * n = 10 * ones + tens + 3

theorem smallest_valid_n :
  is_valid 12 ∧ ∀ m : ℕ, is_valid m → 12 ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l3757_375784


namespace NUMINAMATH_CALUDE_max_balls_for_five_weighings_impossibility_more_than_243_balls_l3757_375795

/-- The number of weighings required to identify the lighter ball -/
def num_weighings : ℕ := 5

/-- The maximum number of balls that can be tested with the given number of weighings -/
def max_balls : ℕ := 3^num_weighings

/-- Theorem stating that the maximum number of balls is 243 given 5 weighings -/
theorem max_balls_for_five_weighings :
  num_weighings = 5 → max_balls = 243 := by
  sorry

/-- Theorem stating that it's impossible to identify the lighter ball among more than 243 balls with 5 weighings -/
theorem impossibility_more_than_243_balls (n : ℕ) :
  num_weighings = 5 → n > 243 → ¬(∃ strategy : Unit, True) := by
  sorry

end NUMINAMATH_CALUDE_max_balls_for_five_weighings_impossibility_more_than_243_balls_l3757_375795


namespace NUMINAMATH_CALUDE_probability_of_a_l3757_375752

theorem probability_of_a (a b : Set α) (p : Set α → ℝ) 
  (h1 : p b = 2/5)
  (h2 : p (a ∩ b) = p a * p b)
  (h3 : p (a ∩ b) = 0.16000000000000003) :
  p a = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_a_l3757_375752


namespace NUMINAMATH_CALUDE_sand_per_lorry_l3757_375760

/-- Calculates the number of tons of sand per lorry given the following conditions:
  * 500 bags of cement are provided
  * Cement costs $10 per bag
  * 20 lorries of sand are received
  * Sand costs $40 per ton
  * Total cost for all materials is $13000
-/
theorem sand_per_lorry (cement_bags : ℕ) (cement_cost : ℚ) (lorries : ℕ) (sand_cost : ℚ) (total_cost : ℚ) :
  cement_bags = 500 →
  cement_cost = 10 →
  lorries = 20 →
  sand_cost = 40 →
  total_cost = 13000 →
  (total_cost - cement_bags * cement_cost) / sand_cost / lorries = 10 := by
  sorry

#check sand_per_lorry

end NUMINAMATH_CALUDE_sand_per_lorry_l3757_375760


namespace NUMINAMATH_CALUDE_john_card_expenditure_l3757_375737

/-- The number of thank you cards John sent for Christmas gifts -/
def christmas_cards : ℕ := 20

/-- The number of thank you cards John sent for birthday gifts -/
def birthday_cards : ℕ := 15

/-- The cost of each thank you card in dollars -/
def card_cost : ℕ := 2

/-- The total cost of all thank you cards John bought -/
def total_cost : ℕ := (christmas_cards + birthday_cards) * card_cost

theorem john_card_expenditure :
  total_cost = 70 := by sorry

end NUMINAMATH_CALUDE_john_card_expenditure_l3757_375737


namespace NUMINAMATH_CALUDE_medicine_price_reduction_l3757_375729

theorem medicine_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 25)
  (h2 : final_price = 16)
  (h3 : final_price = original_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) : 
  x = 0.2 := by sorry

end NUMINAMATH_CALUDE_medicine_price_reduction_l3757_375729


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l3757_375765

-- Define the isosceles triangle
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

-- Define the properties of the triangle
def triangle_properties (t : IsoscelesTriangle) (area1 area2 : ℝ) : Prop :=
  area1 = 6 * 6 / 11 ∧ 
  area2 = 5 * 5 / 11 ∧ 
  area1 + area2 = 1 / 2 * t.base * (t.leg ^ 2 - (t.base / 2) ^ 2).sqrt

-- Theorem statement
theorem isosceles_triangle_sides 
  (t : IsoscelesTriangle) 
  (area1 area2 : ℝ) 
  (h : triangle_properties t area1 area2) : 
  t.base = 6 ∧ t.leg = 5 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_sides_l3757_375765


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3757_375717

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_large_number : sum_of_digits (10^95 - 95 - 2) = 840 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l3757_375717


namespace NUMINAMATH_CALUDE_expected_weekly_rain_l3757_375757

/-- The number of days in the week --/
def days : ℕ := 7

/-- The probability of sun (0 inches of rain) --/
def probSun : ℝ := 0.3

/-- The probability of 5 inches of rain --/
def probRain5 : ℝ := 0.4

/-- The probability of 12 inches of rain --/
def probRain12 : ℝ := 0.3

/-- The amount of rain on a sunny day --/
def rainSun : ℝ := 0

/-- The amount of rain on a day with 5 inches --/
def rain5 : ℝ := 5

/-- The amount of rain on a day with 12 inches --/
def rain12 : ℝ := 12

/-- The expected value of rainfall for one day --/
def expectedDailyRain : ℝ := probSun * rainSun + probRain5 * rain5 + probRain12 * rain12

/-- Theorem: The expected value of total rainfall for the week is 39.2 inches --/
theorem expected_weekly_rain : days * expectedDailyRain = 39.2 := by
  sorry

end NUMINAMATH_CALUDE_expected_weekly_rain_l3757_375757


namespace NUMINAMATH_CALUDE_special_function_values_l3757_375761

/-- A function satisfying f(x + y) = 2 f(x) f(y) for all real x and y -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = 2 * f x * f y

/-- Theorem stating the possible values of f(1) for a SpecialFunction -/
theorem special_function_values (f : ℝ → ℝ) (hf : SpecialFunction f) :
  f 1 = 0 ∨ ∃ r : ℝ, f 1 = r :=
by sorry

end NUMINAMATH_CALUDE_special_function_values_l3757_375761


namespace NUMINAMATH_CALUDE_rectangle_area_l3757_375767

/-- Proves that a rectangular field with sides in ratio 3:4 and perimeter costing 8750 paise
    at 25 paise per metre has an area of 7500 square meters. -/
theorem rectangle_area (length width : ℝ) (h1 : length / width = 3 / 4)
    (h2 : 2 * (length + width) * 25 = 8750) : length * width = 7500 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3757_375767


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l3757_375702

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = a^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = b^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = c^2

-- Define the angle measure function
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_measure_in_triangle (A B C P : ℝ × ℝ) :
  Triangle A B C →
  angle_measure A B C = 40 →
  angle_measure A C B = 40 →
  angle_measure P A C = 20 →
  angle_measure P C B = 30 →
  angle_measure P B C = 20 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l3757_375702


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3757_375700

theorem polynomial_divisibility (n : ℕ) (hn : n > 2) :
  (∃ q : Polynomial ℚ, x^n + x^2 + 1 = (x^2 + x + 1) * q) ↔ 
  (∃ k : ℕ, n = 3 * k + 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3757_375700


namespace NUMINAMATH_CALUDE_age_puzzle_l3757_375751

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 18) (h2 : 3 * (A + x) - 3 * (A - 3) = A) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3757_375751


namespace NUMINAMATH_CALUDE_trig_identity_proof_l3757_375790

theorem trig_identity_proof : 
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / 
  Real.sin (73 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l3757_375790


namespace NUMINAMATH_CALUDE_total_correct_answers_l3757_375778

/-- Given a math test with 40 questions where 75% are answered correctly,
    and an English test with 50 questions where 98% are answered correctly,
    the total number of correctly answered questions is 79. -/
theorem total_correct_answers
  (math_questions : ℕ)
  (math_percentage : ℚ)
  (english_questions : ℕ)
  (english_percentage : ℚ)
  (h1 : math_questions = 40)
  (h2 : math_percentage = 75 / 100)
  (h3 : english_questions = 50)
  (h4 : english_percentage = 98 / 100) :
  ⌊math_questions * math_percentage⌋ + ⌊english_questions * english_percentage⌋ = 79 :=
by sorry

end NUMINAMATH_CALUDE_total_correct_answers_l3757_375778


namespace NUMINAMATH_CALUDE_cube_sum_problem_l3757_375782

theorem cube_sum_problem (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) :
  x^3 + y^3 = 640 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_problem_l3757_375782


namespace NUMINAMATH_CALUDE_book_rearrangement_combinations_l3757_375789

/-- The number of options for each day of the week --/
def daily_options : List Nat := [1, 2, 3, 3, 2]

/-- The total number of combinations --/
def total_combinations : Nat := daily_options.prod

/-- Theorem stating that the total number of combinations is 36 --/
theorem book_rearrangement_combinations :
  total_combinations = 36 := by
  sorry

end NUMINAMATH_CALUDE_book_rearrangement_combinations_l3757_375789


namespace NUMINAMATH_CALUDE_one_fourth_of_8_point_8_l3757_375792

theorem one_fourth_of_8_point_8 : 
  (8.8 / 4 : ℚ) = 11 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_point_8_l3757_375792


namespace NUMINAMATH_CALUDE_quadratic_root_l3757_375764

theorem quadratic_root (a b c : ℝ) (h : a ≠ 0 ∧ b + c ≠ 0) :
  let f : ℝ → ℝ := λ x => a * (b + c) * x^2 - b * (c + a) * x - c * (a + b)
  (f (-1) = 0) → (f (c * (a + b) / (a * (b + c))) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_l3757_375764


namespace NUMINAMATH_CALUDE_dividend_calculation_l3757_375728

theorem dividend_calculation (k : ℕ) (quotient : ℕ) (h1 : k = 8) (h2 : quotient = 8) :
  k * quotient = 64 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3757_375728


namespace NUMINAMATH_CALUDE_polygon_sides_l3757_375718

/-- Theorem: For a polygon with n sides, if the sum of its interior angles is 180° less than three times the sum of its exterior angles, then n = 7. -/
theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3757_375718


namespace NUMINAMATH_CALUDE_kibble_remaining_is_seven_l3757_375736

/-- The amount of kibble remaining in Luna's bag after one day of feeding. -/
def kibble_remaining (initial_amount : ℕ) (mary_morning : ℕ) (mary_evening : ℕ) (frank_afternoon : ℕ) : ℕ :=
  initial_amount - (mary_morning + mary_evening + frank_afternoon + 2 * frank_afternoon)

/-- Theorem stating that the amount of kibble remaining is 7 cups. -/
theorem kibble_remaining_is_seven :
  kibble_remaining 12 1 1 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_kibble_remaining_is_seven_l3757_375736


namespace NUMINAMATH_CALUDE_distance_difference_around_block_l3757_375747

/-- The difference in distance run by two people around a square block -/
def distanceDifference (blockSideLength : ℝ) (streetWidth : ℝ) : ℝ :=
  4 * (2 * streetWidth)

theorem distance_difference_around_block :
  let blockSideLength : ℝ := 400
  let streetWidth : ℝ := 20
  distanceDifference blockSideLength streetWidth = 160 := by sorry

end NUMINAMATH_CALUDE_distance_difference_around_block_l3757_375747


namespace NUMINAMATH_CALUDE_math_books_count_l3757_375799

/-- The number of math books on a shelf with 100 total books, 32 history books, and 25 geography books. -/
def math_books (total : ℕ) (history : ℕ) (geography : ℕ) : ℕ :=
  total - history - geography

/-- Theorem stating that there are 43 math books on the shelf. -/
theorem math_books_count : math_books 100 32 25 = 43 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l3757_375799


namespace NUMINAMATH_CALUDE_bonus_implication_l3757_375742

-- Define the universe of discourse
variable (Employee : Type)

-- Define the predicates
variable (completes_all_projects : Employee → Prop)
variable (receives_bonus : Employee → Prop)

-- Mr. Thompson's statement
variable (thompson_statement : ∀ (e : Employee), completes_all_projects e → receives_bonus e)

-- Theorem to prove
theorem bonus_implication :
  ∀ (e : Employee), ¬(receives_bonus e) → ¬(completes_all_projects e) := by
  sorry

end NUMINAMATH_CALUDE_bonus_implication_l3757_375742


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l3757_375774

/-- Represents the two boxes containing balls -/
inductive Box
| A
| B

/-- Represents the color of the balls -/
inductive Color
| Red
| White

/-- Represents the number of balls in each box before transfer -/
def initial_count : Box → Color → ℕ
| Box.A, Color.Red => 4
| Box.A, Color.White => 2
| Box.B, Color.Red => 2
| Box.B, Color.White => 3

/-- Represents the probability space for this problem -/
structure BallProbability where
  /-- The probability of event A (red ball taken from box A) -/
  prob_A : ℝ
  /-- The probability of event B (white ball taken from box A) -/
  prob_B : ℝ
  /-- The probability of event C (red ball taken from box B after transfer) -/
  prob_C : ℝ
  /-- The conditional probability of C given A -/
  prob_C_given_A : ℝ

/-- The main theorem that encapsulates the problem -/
theorem ball_probability_theorem (p : BallProbability) : 
  p.prob_A + p.prob_B = 1 ∧ 
  p.prob_A * p.prob_B = 0 ∧
  p.prob_C_given_A = 1/2 ∧ 
  p.prob_C = 4/9 := by
  sorry


end NUMINAMATH_CALUDE_ball_probability_theorem_l3757_375774


namespace NUMINAMATH_CALUDE_limit_at_one_l3757_375735

-- Define the function f
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 1| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_at_one_l3757_375735


namespace NUMINAMATH_CALUDE_percentage_problem_l3757_375788

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3757_375788


namespace NUMINAMATH_CALUDE_condition_analysis_l3757_375743

theorem condition_analysis (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a + Real.log b > b + Real.log a) ∧
  (∃ a b : ℝ, a + Real.log b > b + Real.log a ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l3757_375743


namespace NUMINAMATH_CALUDE_two_percent_of_one_l3757_375703

theorem two_percent_of_one : (2 : ℚ) / 100 = (2 : ℚ) / 100 * 1 := by sorry

end NUMINAMATH_CALUDE_two_percent_of_one_l3757_375703


namespace NUMINAMATH_CALUDE_right_triangle_conditions_l3757_375734

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.A + t.B = t.C
def condition2 (t : Triangle) : Prop := ∃ (k : Real), t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k
def condition3 (t : Triangle) : Prop := t.A = 90 - t.B

-- Theorem statement
theorem right_triangle_conditions (t : Triangle) :
  (condition1 t ∨ condition2 t ∨ condition3 t) → isRightTriangle t :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_conditions_l3757_375734


namespace NUMINAMATH_CALUDE_fraction_comparison_l3757_375777

theorem fraction_comparison : (17 : ℚ) / 14 > (31 : ℚ) / 11 := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3757_375777


namespace NUMINAMATH_CALUDE_total_pears_picked_l3757_375730

theorem total_pears_picked (alyssa nancy michael : ℕ) 
  (h1 : alyssa = 42)
  (h2 : nancy = 17)
  (h3 : michael = 31) :
  alyssa + nancy + michael = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l3757_375730


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3757_375720

def vector_a : Fin 2 → ℝ := ![1, 2]
def vector_b : Fin 2 → ℝ := ![-2, 3]

theorem vector_sum_magnitude :
  ‖vector_a + vector_b‖ = Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3757_375720


namespace NUMINAMATH_CALUDE_baker_cakes_remaining_l3757_375793

theorem baker_cakes_remaining (initial_cakes bought_cakes sold_cakes : ℕ) 
  (h1 : initial_cakes = 173)
  (h2 : bought_cakes = 103)
  (h3 : sold_cakes = 86) :
  initial_cakes + bought_cakes - sold_cakes = 190 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_remaining_l3757_375793


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l3757_375732

open Complex

theorem smallest_distance_between_circles (z w : ℂ) : 
  abs (z - (2 + 4*I)) = 2 →
  abs (w - (5 + 2*I)) = 4 →
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), abs (z' - (2 + 4*I)) = 2 → abs (w' - (5 + 2*I)) = 4 → abs (z' - w') ≥ min_dist) ∧
    min_dist = 6 - Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l3757_375732


namespace NUMINAMATH_CALUDE_street_running_distances_l3757_375727

/-- Represents the distance run around a square block -/
def run_distance (block_side : ℝ) (street_width : ℝ) (position : ℕ) : ℝ :=
  match position with
  | 0 => 4 * (block_side - 2 * street_width) -- inner side
  | 1 => 4 * block_side -- block side
  | 2 => 4 * (block_side + 2 * street_width) -- outer side
  | _ => 0 -- invalid position

theorem street_running_distances 
  (block_side : ℝ) (street_width : ℝ) 
  (h1 : block_side = 500) 
  (h2 : street_width = 25) : 
  run_distance block_side street_width 2 - run_distance block_side street_width 1 = 200 ∧
  run_distance block_side street_width 1 - run_distance block_side street_width 0 = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_street_running_distances_l3757_375727


namespace NUMINAMATH_CALUDE_overlap_area_is_one_l3757_375719

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  h_x : x < 3
  h_y : y < 3

/-- Represents a triangle on the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The two specific triangles on the grid -/
def triangle1 : GridTriangle := {
  p1 := ⟨0, 0, by norm_num, by norm_num⟩,
  p2 := ⟨2, 1, by norm_num, by norm_num⟩,
  p3 := ⟨1, 2, by norm_num, by norm_num⟩
}

def triangle2 : GridTriangle := {
  p1 := ⟨2, 2, by norm_num, by norm_num⟩,
  p2 := ⟨0, 1, by norm_num, by norm_num⟩,
  p3 := ⟨1, 0, by norm_num, by norm_num⟩
}

/-- Calculates the area of the overlapping region of two triangles -/
def overlapArea (t1 t2 : GridTriangle) : ℝ := sorry

/-- Theorem stating that the overlap area of the specific triangles is 1 -/
theorem overlap_area_is_one : overlapArea triangle1 triangle2 = 1 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_one_l3757_375719


namespace NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l3757_375745

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 30

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 3.75

/-- Theorem stating that the percentage of water in fresh grapes is 90% -/
theorem water_percentage_in_fresh_grapes :
  water_percentage_fresh = 90 :=
by sorry

end NUMINAMATH_CALUDE_water_percentage_in_fresh_grapes_l3757_375745


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3757_375754

theorem rectangle_perimeter (a b : ℤ) : 
  a ≠ b →  -- non-square condition
  a > 0 →  -- positive dimension
  b > 0 →  -- positive dimension
  a * b + 9 = 2 * a + 2 * b + 9 →  -- area plus 9 equals perimeter plus 9
  2 * (a + b) = 18 :=  -- perimeter equals 18
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3757_375754


namespace NUMINAMATH_CALUDE_negative_sum_l3757_375721

theorem negative_sum (a b c : ℝ) 
  (ha : -2 < a ∧ a < -1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : -1 < c ∧ c < 0) : 
  b + c < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l3757_375721


namespace NUMINAMATH_CALUDE_horner_V₁_eq_22_l3757_375755

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 + 2x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-0.8, 1.7, -2.6, 3.5, 2, 4]

/-- V₁ in Horner's method for f(5) -/
def V₁ : ℝ := 4 * 5 + 2

theorem horner_V₁_eq_22 : V₁ = 22 := by
  sorry

#eval V₁  -- Should output 22

end NUMINAMATH_CALUDE_horner_V₁_eq_22_l3757_375755


namespace NUMINAMATH_CALUDE_bread_distribution_l3757_375744

theorem bread_distribution (a : ℚ) (d : ℚ) :
  (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 100) →
  ((a + 3*d) + (a + 4*d) + (a + 2*d))/7 = a + (a + d) →
  a = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_bread_distribution_l3757_375744


namespace NUMINAMATH_CALUDE_problem_solution_l3757_375798

theorem problem_solution : (29.7 + 83.45) - 0.3 = 112.85 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3757_375798


namespace NUMINAMATH_CALUDE_midpoint_line_slope_l3757_375787

/-- The slope of the line containing the midpoints of two given line segments is -3/7 -/
theorem midpoint_line_slope :
  let midpoint1 := ((1 + 3) / 2, (2 + 6) / 2)
  let midpoint2 := ((4 + 7) / 2, (1 + 4) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = -3 / 7 := by sorry

end NUMINAMATH_CALUDE_midpoint_line_slope_l3757_375787


namespace NUMINAMATH_CALUDE_inverse_and_negation_of_union_subset_inverse_of_divisibility_negation_and_contrapositive_of_inequality_inverse_of_quadratic_inequality_l3757_375753

-- Define the sets A and B
variable (A B : Set α)

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- 1. Inverse and negation of "If x ∈ (A ∪ B), then x ∈ B"
theorem inverse_and_negation_of_union_subset (x : α) :
  (x ∈ B → x ∈ A ∪ B) ∧ (x ∉ A ∪ B → x ∉ B) := by sorry

-- 2. Inverse of "If a natural number is divisible by 6, then it is divisible by 2"
theorem inverse_of_divisibility :
  ¬(∀ n : ℕ, divides 2 n → divides 6 n) := by sorry

-- 3. Negation and contrapositive of "If 0 < x < 5, then |x-2| < 3"
theorem negation_and_contrapositive_of_inequality (x : ℝ) :
  ¬(¬(0 < x ∧ x < 5) → |x - 2| ≥ 3) ∧
  (|x - 2| ≥ 3 → ¬(0 < x ∧ x < 5)) := by sorry

-- 4. Inverse of "If (a-2)x^2 + 2(a-2)x - 4 < 0 holds for all x ∈ ℝ, then a ∈ (-2, 2)"
theorem inverse_of_quadratic_inequality (a : ℝ) :
  a ∈ Set.Ioo (-2) 2 →
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) := by sorry

end NUMINAMATH_CALUDE_inverse_and_negation_of_union_subset_inverse_of_divisibility_negation_and_contrapositive_of_inequality_inverse_of_quadratic_inequality_l3757_375753


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l3757_375786

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- Definition of line l₁ -/
def l₁ (k : ℝ) (x y : ℝ) : Prop :=
  (k - 3) * x + (4 - k) * y + 1 = 0

/-- Definition of line l₂ -/
def l₂ (k : ℝ) (x y : ℝ) : Prop :=
  2 * (k - 3) * x - 2 * y + 3 = 0

theorem parallel_lines_k_values :
  ∀ k : ℝ, (∀ x y : ℝ, are_parallel (k - 3) (4 - k) (2 * (k - 3)) (-2)) →
  k = 3 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_values_l3757_375786


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3757_375733

theorem modulus_of_complex_number : 
  let z : ℂ := (1 + 3 * Complex.I) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3757_375733


namespace NUMINAMATH_CALUDE_pet_store_puppies_l3757_375739

theorem pet_store_puppies (sold : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : sold = 24)
  (h2 : cages = 8)
  (h3 : puppies_per_cage = 4) :
  sold + cages * puppies_per_cage = 56 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l3757_375739


namespace NUMINAMATH_CALUDE_inverse_composition_equality_l3757_375706

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition f⁻¹ ∘ g = λ x, 2*x - 4
variable (h : ∀ x, f⁻¹ (g x) = 2 * x - 4)

-- State the theorem
theorem inverse_composition_equality : g⁻¹ (f (-3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_equality_l3757_375706


namespace NUMINAMATH_CALUDE_system_solution_l3757_375714

theorem system_solution : 
  {(x, y) : ℝ × ℝ | x^2 + y^2 + x + y = 50 ∧ x * y = 20} = 
  {(5, 4), (4, 5), (-5 + Real.sqrt 5, -5 - Real.sqrt 5), (-5 - Real.sqrt 5, -5 + Real.sqrt 5)} := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3757_375714


namespace NUMINAMATH_CALUDE_election_abstention_percentage_l3757_375763

theorem election_abstention_percentage 
  (total_members : ℕ) 
  (votes_cast : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_b_percentage : ℚ) 
  (candidate_c_percentage : ℚ) 
  (candidate_d_percentage : ℚ) 
  (h1 : total_members = 1600) 
  (h2 : votes_cast = 900) 
  (h3 : candidate_a_percentage = 45/100) 
  (h4 : candidate_b_percentage = 35/100) 
  (h5 : candidate_c_percentage = 15/100) 
  (h6 : candidate_d_percentage = 5/100) 
  (h7 : candidate_a_percentage + candidate_b_percentage + candidate_c_percentage + candidate_d_percentage = 1) :
  (total_members - votes_cast : ℚ) / total_members * 100 = 43.75 := by
sorry

end NUMINAMATH_CALUDE_election_abstention_percentage_l3757_375763


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l3757_375707

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_intersection 
  (ABCD : Quadrilateral) 
  (hConvex : isConvex ABCD) 
  (hAB : distance ABCD.A ABCD.B = 10)
  (hCD : distance ABCD.C ABCD.D = 15)
  (hAC : distance ABCD.A ABCD.C = 17)
  (E : Point)
  (hE : E = lineIntersection ABCD.A ABCD.C ABCD.B ABCD.D)
  (hAreas : triangleArea ABCD.A E ABCD.D = triangleArea ABCD.B E ABCD.C) :
  distance ABCD.A E = 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l3757_375707


namespace NUMINAMATH_CALUDE_circle_area_equals_rectangle_area_l3757_375705

theorem circle_area_equals_rectangle_area (R : ℝ) (h : R = 4) :
  π * R^2 = (2 * π * R) * (R / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equals_rectangle_area_l3757_375705


namespace NUMINAMATH_CALUDE_final_book_count_l3757_375723

/-- Represents the number of books in the library system -/
structure LibraryState where
  books : ℕ

/-- Represents a transaction that changes the number of books -/
inductive Transaction
  | TakeOut (n : ℕ)
  | Return (n : ℕ)
  | Withdraw (n : ℕ)

/-- Applies a transaction to the library state -/
def applyTransaction (state : LibraryState) (t : Transaction) : LibraryState :=
  match t with
  | Transaction.TakeOut n => ⟨state.books - n⟩
  | Transaction.Return n => ⟨state.books + n⟩
  | Transaction.Withdraw n => ⟨state.books - n⟩

/-- Applies a list of transactions to the library state -/
def applyTransactions (state : LibraryState) (ts : List Transaction) : LibraryState :=
  ts.foldl applyTransaction state

/-- The initial state of the library -/
def initialState : LibraryState := ⟨250⟩

/-- The transactions that occur over the three weeks -/
def transactions : List Transaction := [
  Transaction.TakeOut 120,  -- Week 1 Tuesday
  Transaction.Return 35,    -- Week 1 Wednesday
  Transaction.Withdraw 15,  -- Week 1 Thursday
  Transaction.TakeOut 42,   -- Week 1 Friday
  Transaction.Return 72,    -- Week 2 Monday (60% of 120)
  Transaction.Return 34,    -- Week 2 Tuesday (80% of 42, rounded)
  Transaction.Withdraw 75,  -- Week 2 Wednesday
  Transaction.TakeOut 40,   -- Week 2 Thursday
  Transaction.Return 20,    -- Week 3 Monday (50% of 40)
  Transaction.TakeOut 20,   -- Week 3 Tuesday
  Transaction.Return 46,    -- Week 3 Wednesday (95% of 48, rounded)
  Transaction.Withdraw 10,  -- Week 3 Thursday
  Transaction.TakeOut 55    -- Week 3 Friday
]

/-- The theorem stating that after applying all transactions, the library has 80 books -/
theorem final_book_count :
  (applyTransactions initialState transactions).books = 80 := by
  sorry

end NUMINAMATH_CALUDE_final_book_count_l3757_375723


namespace NUMINAMATH_CALUDE_marble_difference_l3757_375726

/-- Represents a jar of marbles -/
structure Jar :=
  (blue : ℕ)
  (green : ℕ)

/-- The problem statement -/
theorem marble_difference (jar1 jar2 : Jar) : 
  jar1.blue + jar1.green = jar2.blue + jar2.green →
  7 * jar1.green = 3 * jar1.blue →
  9 * jar2.green = jar2.blue →
  jar1.green + jar2.green = 80 →
  jar2.blue - jar1.blue = 40 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l3757_375726


namespace NUMINAMATH_CALUDE_min_cups_in_boxes_min_cups_for_100_boxes_l3757_375758

theorem min_cups_in_boxes : ℕ → ℕ
  | n => (n * (n + 1)) / 2

theorem min_cups_for_100_boxes :
  min_cups_in_boxes 100 = 5050 := by sorry

end NUMINAMATH_CALUDE_min_cups_in_boxes_min_cups_for_100_boxes_l3757_375758


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l3757_375768

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 3 * a 11 = 16 →
  a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l3757_375768
