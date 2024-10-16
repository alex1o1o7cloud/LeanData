import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l1657_165758

/-- Given a right triangle divided by a point on its hypotenuse and lines parallel to its legs,
    forming a square and two smaller right triangles, this theorem proves the relationship
    between the areas of the smaller triangles and the square. -/
theorem right_triangle_division_area_ratio
  (square_side : ℝ)
  (m : ℝ)
  (h_square_side : square_side = 2)
  (h_small_triangle_area : ∃ (small_triangle_area : ℝ), small_triangle_area = m * square_side^2)
  : ∃ (other_triangle_area : ℝ), other_triangle_area / square_side^2 = 1 / (4 * m) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l1657_165758


namespace NUMINAMATH_CALUDE_midpoint_property_l1657_165788

/-- Given two points A and B in ℝ², prove that if C is their midpoint,
    then 2x - 4y = -22 where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B : ℝ × ℝ) (h1 : A = (15, 10)) (h2 : B = (-5, 6)) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  2 * C.1 - 4 * C.2 = -22 := by
sorry

end NUMINAMATH_CALUDE_midpoint_property_l1657_165788


namespace NUMINAMATH_CALUDE_circle_translation_l1657_165740

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop := (x+1)^2 + (y-2)^2 = 1

-- Define the translation vector
def translation_vector : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem circle_translation :
  ∀ (x y : ℝ),
  original_circle x y ↔ translated_circle (x + translation_vector.1) (y + translation_vector.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_translation_l1657_165740


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_error_l1657_165727

theorem rectangular_prism_volume_error (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let actual_volume := a * b * c
  let measured_volume := (a * 1.08) * (b * 0.90) * (c * 0.94)
  let error_percentage := (measured_volume - actual_volume) / actual_volume * 100
  error_percentage = -2.728 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_error_l1657_165727


namespace NUMINAMATH_CALUDE_smallest_number_l1657_165734

-- Define a function to convert a number from any base to decimal
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [8, 5]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 0]
def base2 : Nat := 6

def num3 : List Nat := [1, 0, 0, 0]
def base3 : Nat := 4

def num4 : List Nat := [1, 1, 1, 1, 1, 1, 1]
def base4 : Nat := 2

-- Theorem statement
theorem smallest_number :
  to_decimal num3 base3 < to_decimal num1 base1 ∧
  to_decimal num3 base3 < to_decimal num2 base2 ∧
  to_decimal num3 base3 < to_decimal num4 base4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1657_165734


namespace NUMINAMATH_CALUDE_westward_movement_l1657_165703

-- Define the direction as an enumeration
inductive Direction
| East
| West

-- Define a function to represent movement
def movement (distance : ℤ) (direction : Direction) : ℤ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem westward_movement :
  movement 1000 Direction.West = -1000 :=
by sorry

end NUMINAMATH_CALUDE_westward_movement_l1657_165703


namespace NUMINAMATH_CALUDE_deepak_age_l1657_165798

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    prove Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 22 = 26 →
  deepak_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_deepak_age_l1657_165798


namespace NUMINAMATH_CALUDE_average_weight_increase_l1657_165772

/-- Proves that replacing a person weighing 60 kg with a person weighing 80 kg
    in a group of 8 people increases the average weight by 2.5 kg -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 8 * initial_average
  let new_total := initial_total - 60 + 80
  let new_average := new_total / 8
  new_average - initial_average = 2.5 := by sorry

end NUMINAMATH_CALUDE_average_weight_increase_l1657_165772


namespace NUMINAMATH_CALUDE_log_ratio_sixteen_four_l1657_165770

theorem log_ratio_sixteen_four : (Real.log 16) / (Real.log 4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_ratio_sixteen_four_l1657_165770


namespace NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1657_165729

theorem rectangle_arrangement_exists : ∃ (a b c d : ℕ+), 
  (a * b + c * d = 81) ∧ 
  ((2 * (a + b) = 4 * (c + d)) ∨ (4 * (a + b) = 2 * (c + d))) :=
sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_exists_l1657_165729


namespace NUMINAMATH_CALUDE_parallel_lines_solution_l1657_165711

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (a b c d e f : ℝ) : Prop :=
  a * e = b * d

/-- The first line: ax + 2y + 6 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + 6 = 0

/-- The second line: x + (a - 1)y + (a^2 - 1) = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  x + (a - 1) * y + (a^2 - 1) = 0

theorem parallel_lines_solution :
  ∀ a : ℝ, parallel_lines a 2 1 (a - 1) 1 1 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_solution_l1657_165711


namespace NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l1657_165735

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence where the third term is 12 and the fourth term is 18, the second term is 8. -/
theorem second_term_of_geometric_sequence
    (a : ℕ → ℚ)
    (h_geometric : IsGeometricSequence a)
    (h_third_term : a 3 = 12)
    (h_fourth_term : a 4 = 18) :
    a 2 = 8 := by
  sorry

#check second_term_of_geometric_sequence

end NUMINAMATH_CALUDE_second_term_of_geometric_sequence_l1657_165735


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l1657_165712

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents a set of parallel lines -/
structure ParallelLines where
  ab : LineSegment
  cd : LineSegment
  ef : LineSegment
  gh : LineSegment

/-- Given conditions for the problem -/
def problem_conditions (lines : ParallelLines) : Prop :=
  lines.ab.length = 300 ∧
  lines.cd.length = 200 ∧
  lines.ef.length = (lines.ab.length + lines.cd.length) / 4 ∧
  lines.gh.length = lines.ef.length - (lines.ef.length - lines.cd.length) / 4

/-- The theorem to be proved -/
theorem parallel_lines_theorem (lines : ParallelLines) 
  (h : problem_conditions lines) : lines.gh.length = 93.75 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l1657_165712


namespace NUMINAMATH_CALUDE_unique_five_digit_sum_l1657_165739

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (λ acc d => acc * 10 + d) 0

theorem unique_five_digit_sum (n : ℕ) : 
  is_five_digit n ∧ 
  (∃ (pos : Fin 5), n + remove_digit n pos = 54321) ↔ 
  n = 49383 :=
sorry

end NUMINAMATH_CALUDE_unique_five_digit_sum_l1657_165739


namespace NUMINAMATH_CALUDE_books_bought_is_difference_melanie_books_bought_l1657_165713

/-- Represents the number of books Melanie bought at the yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem stating that the number of books bought is the difference between final and initial books -/
theorem books_bought_is_difference (initial_books final_books : ℕ) 
  (h : final_books ≥ initial_books) :
  books_bought initial_books final_books = final_books - initial_books :=
by
  sorry

/-- Melanie's initial number of books -/
def melanie_initial_books : ℕ := 41

/-- Melanie's final number of books -/
def melanie_final_books : ℕ := 87

/-- Theorem proving the number of books Melanie bought at the yard sale -/
theorem melanie_books_bought : 
  books_bought melanie_initial_books melanie_final_books = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_books_bought_is_difference_melanie_books_bought_l1657_165713


namespace NUMINAMATH_CALUDE_min_women_proof_l1657_165702

/-- The probability of at least 4 men standing together given x women -/
def probability (x : ℕ) : ℚ :=
  (2 * Nat.choose (x + 1) 2 + (x + 1)) / (Nat.choose (x + 1) 3 + 3 * Nat.choose (x + 1) 2 + (x + 1))

/-- The minimum number of women required -/
def min_women : ℕ := 594

theorem min_women_proof :
  ∀ x : ℕ, x ≥ min_women ↔ probability x ≤ 1/100 := by
  sorry

#check min_women_proof

end NUMINAMATH_CALUDE_min_women_proof_l1657_165702


namespace NUMINAMATH_CALUDE_shelf_rearrangement_l1657_165782

theorem shelf_rearrangement (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 8 → k = 2 → m = 4 →
  (Nat.choose n k) * ((m + 1) * m + Nat.choose (m + 1) k) = 840 := by
  sorry

end NUMINAMATH_CALUDE_shelf_rearrangement_l1657_165782


namespace NUMINAMATH_CALUDE_fibonacci_identity_l1657_165741

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_identity (n : ℕ) (h : n ≥ 1) :
  (fib (2 * n - 1))^2 + (fib (2 * n + 1))^2 + 1 = 3 * (fib (2 * n - 1)) * (fib (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_identity_l1657_165741


namespace NUMINAMATH_CALUDE_mean_of_pencil_sharpening_counts_l1657_165750

def pencil_sharpening_counts : List ℕ := [13, 8, 13, 21, 7, 23]

theorem mean_of_pencil_sharpening_counts :
  (pencil_sharpening_counts.sum : ℚ) / pencil_sharpening_counts.length = 85/6 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_pencil_sharpening_counts_l1657_165750


namespace NUMINAMATH_CALUDE_no_two_cubes_between_squares_l1657_165720

theorem no_two_cubes_between_squares : ¬ ∃ (n a b : ℤ), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_cubes_between_squares_l1657_165720


namespace NUMINAMATH_CALUDE_smallest_divisible_by_5_13_7_l1657_165757

theorem smallest_divisible_by_5_13_7 : ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 13 ∣ n ∧ 7 ∣ n → n ≥ 455 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_5_13_7_l1657_165757


namespace NUMINAMATH_CALUDE_inequality_proof_l1657_165796

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1657_165796


namespace NUMINAMATH_CALUDE_jemma_price_calculation_l1657_165754

/-- The price at which Jemma sells each frame -/
def jemma_price : ℝ := 5

/-- The number of frames Jemma sold -/
def jemma_frames : ℕ := 400

/-- The total revenue made by both Jemma and Dorothy -/
def total_revenue : ℝ := 2500

theorem jemma_price_calculation :
  (jemma_price * jemma_frames : ℝ) + 
  (jemma_price / 2 * (jemma_frames / 2) : ℝ) = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_jemma_price_calculation_l1657_165754


namespace NUMINAMATH_CALUDE_even_function_inequality_l1657_165715

theorem even_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_increasing : ∀ x y, x < y → x < 0 → f x < f y) :
  f (-2) ≥ f (a^2 - 4*a + 6) := by
  sorry

end NUMINAMATH_CALUDE_even_function_inequality_l1657_165715


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1657_165745

theorem quadratic_expression_value (x : ℝ) : x = 2 → x^2 - 3*x + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1657_165745


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l1657_165753

/-- Proves that the number of dogs not doing anything is 10, given the total number of dogs and the number of dogs engaged in each activity. -/
theorem dogs_not_doing_anything (total : ℕ) (running : ℕ) (playing : ℕ) (barking : ℕ) : 
  total = 88 → 
  running = 12 → 
  playing = total / 2 → 
  barking = total / 4 → 
  total - (running + playing + barking) = 10 := by
sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l1657_165753


namespace NUMINAMATH_CALUDE_same_remainder_problem_l1657_165704

theorem same_remainder_problem (x : ℕ+) : 
  (∃ q r : ℕ, 100 = q * x + r ∧ r < x) ∧ 
  (∃ p r : ℕ, 197 = p * x + r ∧ r < x) → 
  (∃ r : ℕ, 100 % x = r ∧ 197 % x = r ∧ r = 3) := by
sorry

end NUMINAMATH_CALUDE_same_remainder_problem_l1657_165704


namespace NUMINAMATH_CALUDE_solve_coloring_books_problem_l1657_165762

def coloring_books_problem (initial_stock : ℝ) (coupons_per_book : ℝ) (total_coupons_used : ℕ) : Prop :=
  initial_stock = 40.0 ∧
  coupons_per_book = 4.0 ∧
  total_coupons_used = 80 →
  initial_stock - (total_coupons_used : ℝ) / coupons_per_book = 20

theorem solve_coloring_books_problem :
  ∃ (initial_stock coupons_per_book : ℝ) (total_coupons_used : ℕ),
    coloring_books_problem initial_stock coupons_per_book total_coupons_used :=
by
  sorry

end NUMINAMATH_CALUDE_solve_coloring_books_problem_l1657_165762


namespace NUMINAMATH_CALUDE_rationality_of_x_not_necessarily_rational_l1657_165714

theorem rationality_of_x (x : ℝ) :
  (∃ (a b : ℚ), x^7 = a ∧ x^12 = b) →
  ∃ (q : ℚ), x = q :=
sorry

theorem not_necessarily_rational (x : ℝ) :
  (∃ (a b : ℚ), x^9 = a ∧ x^12 = b) →
  ¬(∀ x : ℝ, ∃ (q : ℚ), x = q) :=
sorry

end NUMINAMATH_CALUDE_rationality_of_x_not_necessarily_rational_l1657_165714


namespace NUMINAMATH_CALUDE_smallest_a_value_smallest_a_exists_l1657_165707

/-- A three-digit even number -/
def ThreeDigitEven := {n : ℕ // 100 ≤ n ∧ n ≤ 998 ∧ Even n}

/-- The sum of five three-digit even numbers -/
def SumFiveNumbers := 4306

theorem smallest_a_value (A B C D E : ThreeDigitEven) 
  (h_order : A.val < B.val ∧ B.val < C.val ∧ C.val < D.val ∧ D.val < E.val)
  (h_sum : A.val + B.val + C.val + D.val + E.val = SumFiveNumbers) :
  A.val ≥ 326 := by
  sorry

theorem smallest_a_exists :
  ∃ (A B C D E : ThreeDigitEven),
    A.val < B.val ∧ B.val < C.val ∧ C.val < D.val ∧ D.val < E.val ∧
    A.val + B.val + C.val + D.val + E.val = SumFiveNumbers ∧
    A.val = 326 := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_smallest_a_exists_l1657_165707


namespace NUMINAMATH_CALUDE_no_integer_roots_l1657_165723

theorem no_integer_roots : ¬∃ (x : ℤ), x^3 - 4*x^2 - 11*x + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_roots_l1657_165723


namespace NUMINAMATH_CALUDE_fraction_simplification_l1657_165736

theorem fraction_simplification (m : ℝ) (h : m ≠ 0) : 
  (3 * m^3) / (6 * m^2) = m / 2 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1657_165736


namespace NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l1657_165705

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_seven_balls_three_boxes : distribute_balls 7 3 = 365 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_three_boxes_l1657_165705


namespace NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1657_165732

/-- The longest segment that can fit inside a cylinder -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  Real.sqrt ((2 * r) ^ 2 + h ^ 2) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_longest_segment_in_cylinder_l1657_165732


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l1657_165755

theorem smallest_n_satisfying_inequality :
  ∀ n : ℕ, (1/4 : ℚ) + (n : ℚ)/8 > 1 ↔ n ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_inequality_l1657_165755


namespace NUMINAMATH_CALUDE_most_likely_parent_genotypes_l1657_165760

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy
| h  -- Recessive hairy
| S  -- Dominant smooth
| s  -- Recessive smooth

/-- Represents the genotype of a rabbit -/
structure Genotype :=
(allele1 : Allele)
(allele2 : Allele)

/-- Represents the phenotype (observable trait) of a rabbit -/
inductive Phenotype
| Hairy
| Smooth

/-- Function to determine the phenotype from a genotype -/
def phenotypeFromGenotype (g : Genotype) : Phenotype :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => Phenotype.Hairy
  | _, Allele.H => Phenotype.Hairy
  | Allele.S, _ => Phenotype.Smooth
  | _, Allele.S => Phenotype.Smooth
  | Allele.h, Allele.h => Phenotype.Hairy
  | Allele.s, Allele.s => Phenotype.Smooth
  | _, _ => Phenotype.Smooth

/-- The probability of the hairy allele in the population -/
def hairyAlleleProbability : ℝ := 0.1

/-- Theorem stating the most likely genotype combination for parents -/
theorem most_likely_parent_genotypes
  (hairyParent smoothParent : Genotype)
  (allOffspringHairy : ∀ (offspring : Genotype),
    phenotypeFromGenotype offspring = Phenotype.Hairy) :
  (hairyParent = ⟨Allele.H, Allele.H⟩ ∧
   smoothParent = ⟨Allele.S, Allele.h⟩) ∨
  (hairyParent = ⟨Allele.H, Allele.H⟩ ∧
   smoothParent = ⟨Allele.h, Allele.S⟩) :=
sorry


end NUMINAMATH_CALUDE_most_likely_parent_genotypes_l1657_165760


namespace NUMINAMATH_CALUDE_larger_number_proof_l1657_165721

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 24672)
  (h2 : L = 13 * S + 257) :
  L = 26706 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1657_165721


namespace NUMINAMATH_CALUDE_fifteen_ways_to_choose_l1657_165749

/-- Represents the number of ways to choose singers and dancers from a group -/
def choose_performers (total : ℕ) (dancers : ℕ) (singers : ℕ) : ℕ :=
  let both := dancers + singers - total
  let pure_singers := singers - both
  let pure_dancers := dancers - both
  both * pure_dancers + pure_singers * both

/-- Theorem: There are 15 ways to choose performers from the given group -/
theorem fifteen_ways_to_choose : choose_performers 8 6 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_ways_to_choose_l1657_165749


namespace NUMINAMATH_CALUDE_focus_coincidence_l1657_165752

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := (5*x^2)/3 - (5*y^2)/2 = 1

-- Define the focus of a parabola
def parabola_focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Define the right focus of a hyperbola
def hyperbola_right_focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Theorem statement
theorem focus_coincidence :
  ∀ (x y : ℝ), parabola_focus x y ↔ hyperbola_right_focus x y :=
sorry

end NUMINAMATH_CALUDE_focus_coincidence_l1657_165752


namespace NUMINAMATH_CALUDE_ratio_problem_l1657_165778

theorem ratio_problem (a b c : ℝ) (h1 : a / b = 11 / 3) (h2 : a / c = 0.7333333333333333) : 
  b / c = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1657_165778


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1657_165766

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ n : ℕ, (n > 0) ∧ 
  (∃ p₁ p₂ p₃ p₄ : ℕ, Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
   p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
   n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0) ∧
  (∀ m : ℕ, m > 0 → m < n → 
   ¬(∃ q₁ q₂ q₃ q₄ : ℕ, Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ 
     q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
     m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0)) ∧
  n = 210 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l1657_165766


namespace NUMINAMATH_CALUDE_inequality_implies_interval_bound_l1657_165767

theorem inequality_implies_interval_bound 
  (k m a b : ℝ) 
  (h : ∀ x ∈ Set.Icc a b, |x^2 - k*x - m| ≤ 1) : 
  b - a ≤ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implies_interval_bound_l1657_165767


namespace NUMINAMATH_CALUDE_bob_profit_l1657_165718

/-- Calculates the profit from breeding and selling show dogs -/
def dog_breeding_profit (num_dogs : ℕ) (cost_per_dog : ℚ) (num_puppies : ℕ) (price_per_puppy : ℚ) : ℚ :=
  num_puppies * price_per_puppy - num_dogs * cost_per_dog

/-- Bob's profit from breeding and selling show dogs -/
theorem bob_profit : 
  dog_breeding_profit 2 250 6 350 = 1600 :=
by sorry

end NUMINAMATH_CALUDE_bob_profit_l1657_165718


namespace NUMINAMATH_CALUDE_tickets_left_l1657_165775

def tickets_bought : ℕ := 11
def tickets_spent : ℕ := 3

theorem tickets_left : tickets_bought - tickets_spent = 8 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l1657_165775


namespace NUMINAMATH_CALUDE_abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one_l1657_165744

theorem abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one :
  (∀ x : ℝ, |x - 1| > 2 → x^2 > 1) ∧
  (∃ x : ℝ, x^2 > 1 ∧ |x - 1| ≤ 2) := by
sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_gt_two_sufficient_not_necessary_for_x_sq_gt_one_l1657_165744


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1657_165710

theorem condition_sufficient_not_necessary (a b c : ℝ) :
  (∀ a b c : ℝ, a > b ∧ c > 0 → a * c > b * c) ∧
  ¬(∀ a b c : ℝ, a * c > b * c → a > b ∧ c > 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1657_165710


namespace NUMINAMATH_CALUDE_swallow_flock_capacity_l1657_165731

/-- Represents the carrying capacity of different types of swallows -/
structure SwallowCapacity where
  american : ℕ
  european : ℕ
  african : ℕ

/-- Represents the composition of a flock of swallows -/
structure SwallowFlock where
  american : ℕ
  european : ℕ
  african : ℕ

/-- Calculates the total number of swallows in a flock -/
def totalSwallows (flock : SwallowFlock) : ℕ :=
  flock.american + flock.european + flock.african

/-- Calculates the maximum weight a flock can carry -/
def maxCarryWeight (capacity : SwallowCapacity) (flock : SwallowFlock) : ℕ :=
  flock.american * capacity.american +
  flock.european * capacity.european +
  flock.african * capacity.african

/-- Theorem stating the maximum carrying capacity of a specific flock of swallows -/
theorem swallow_flock_capacity
  (capacity : SwallowCapacity)
  (flock : SwallowFlock)
  (h1 : capacity.american = 5)
  (h2 : capacity.european = 10)
  (h3 : capacity.african = 15)
  (h4 : flock.american = 45)
  (h5 : flock.european = 30)
  (h6 : flock.african = 75)
  (h7 : totalSwallows flock = 150)
  (h8 : flock.american * 2 = flock.european * 3)
  (h9 : flock.american * 5 = flock.african * 3) :
  maxCarryWeight capacity flock = 1650 := by
  sorry


end NUMINAMATH_CALUDE_swallow_flock_capacity_l1657_165731


namespace NUMINAMATH_CALUDE_rectangle_circle_union_area_l1657_165756

/-- The area of the union of a rectangle and a circle with specific dimensions -/
theorem rectangle_circle_union_area :
  let rectangle_length : ℝ := 12
  let rectangle_width : ℝ := 8
  let circle_radius : ℝ := 8
  let rectangle_area := rectangle_length * rectangle_width
  let circle_area := π * circle_radius^2
  let overlap_area := (1/4) * circle_area
  rectangle_area + circle_area - overlap_area = 96 + 48 * π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_union_area_l1657_165756


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1657_165765

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / (1 + ω)) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1657_165765


namespace NUMINAMATH_CALUDE_number_of_factors_of_n_l1657_165795

def n : ℕ := 2^2 * 3^2 * 7^2

theorem number_of_factors_of_n : (Finset.card (Nat.divisors n)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_n_l1657_165795


namespace NUMINAMATH_CALUDE_election_votes_calculation_l1657_165719

theorem election_votes_calculation (total_votes : ℕ) : 
  (80 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 380800 →
  total_votes = 560000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l1657_165719


namespace NUMINAMATH_CALUDE_group_meal_cost_l1657_165774

/-- Calculates the total cost for a group meal including tax and tip -/
def calculate_total_cost (vegetarian_price chicken_price steak_price kids_price : ℚ)
                         (tax_rate tip_rate : ℚ)
                         (vegetarian_count chicken_count steak_count kids_count : ℕ) : ℚ :=
  let subtotal := vegetarian_price * vegetarian_count +
                  chicken_price * chicken_count +
                  steak_price * steak_count +
                  kids_price * kids_count
  let tax := subtotal * tax_rate
  let tip := subtotal * tip_rate
  subtotal + tax + tip

/-- Theorem stating that the total cost for the given group is $90 -/
theorem group_meal_cost :
  calculate_total_cost 5 7 10 3 (1/10) (15/100) 3 4 2 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_l1657_165774


namespace NUMINAMATH_CALUDE_subset_polynomial_equivalence_l1657_165781

theorem subset_polynomial_equivalence (n : ℕ) (h : n > 4) :
  (∀ (A B : Set (Fin n)), ∃ (f : Polynomial ℤ),
    (∀ a ∈ A, ∃ b ∈ B, f.eval a ≡ b [ZMOD n]) ∨
    (∀ b ∈ B, ∃ a ∈ A, f.eval b ≡ a [ZMOD n])) ↔
  Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_subset_polynomial_equivalence_l1657_165781


namespace NUMINAMATH_CALUDE_milk_fraction_in_second_cup_l1657_165725

theorem milk_fraction_in_second_cup 
  (V : ℝ) -- Volume of each cup
  (x : ℝ) -- Fraction of milk in the second cup
  (h1 : V > 0) -- Volume is positive
  (h2 : 0 ≤ x ∧ x ≤ 1) -- x is a valid fraction
  : ((2/5 * V + (1 - x) * V) / ((3/5 * V + x * V))) = 3/7 → x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_milk_fraction_in_second_cup_l1657_165725


namespace NUMINAMATH_CALUDE_litter_count_sum_l1657_165794

theorem litter_count_sum : 
  let glass_bottles : ℕ := 25
  let aluminum_cans : ℕ := 18
  let plastic_bags : ℕ := 12
  let paper_cups : ℕ := 7
  let cigarette_packs : ℕ := 5
  let face_masks : ℕ := 3
  glass_bottles + aluminum_cans + plastic_bags + paper_cups + cigarette_packs + face_masks = 70 := by
  sorry

end NUMINAMATH_CALUDE_litter_count_sum_l1657_165794


namespace NUMINAMATH_CALUDE_partnership_profit_l1657_165797

/-- Given the investment ratios and C's profit share, calculate the total profit -/
theorem partnership_profit (a b c : ℚ) (c_profit : ℚ) : 
  a = 6 ∧ b = 2 ∧ c = 9 ∧ c_profit = 6000.000000000001 →
  (a + b + c) * c_profit / c = 11333.333333333336 :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l1657_165797


namespace NUMINAMATH_CALUDE_students_walking_to_school_l1657_165791

theorem students_walking_to_school 
  (total_students : ℕ) 
  (walking_minus_public : ℕ) 
  (h1 : total_students = 41)
  (h2 : walking_minus_public = 3) :
  let walking := (total_students + walking_minus_public) / 2
  walking = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_walking_to_school_l1657_165791


namespace NUMINAMATH_CALUDE_choose_and_assign_roles_l1657_165773

/-- The number of members in the group -/
def group_size : ℕ := 4

/-- The number of roles to be assigned -/
def roles_count : ℕ := 3

/-- The number of ways to choose and assign roles -/
def ways_to_choose_and_assign : ℕ := group_size * (group_size - 1) * (group_size - 2)

theorem choose_and_assign_roles :
  ways_to_choose_and_assign = 24 :=
sorry

end NUMINAMATH_CALUDE_choose_and_assign_roles_l1657_165773


namespace NUMINAMATH_CALUDE_possible_values_of_5x_plus_2_l1657_165785

theorem possible_values_of_5x_plus_2 (x : ℝ) : 
  (x - 4) * (5 * x + 2) = 0 → (5 * x + 2 = 0 ∨ 5 * x + 2 = 22) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_5x_plus_2_l1657_165785


namespace NUMINAMATH_CALUDE_least_divisible_by_1920_eight_divisible_by_1920_eight_is_least_divisible_by_1920_l1657_165792

theorem least_divisible_by_1920 (a : ℕ) : a^6 % 1920 = 0 → a ≥ 8 :=
sorry

theorem eight_divisible_by_1920 : 8^6 % 1920 = 0 :=
sorry

theorem eight_is_least_divisible_by_1920 : ∃ (a : ℕ), a^6 % 1920 = 0 ∧ ∀ (b : ℕ), b < a → b^6 % 1920 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_1920_eight_divisible_by_1920_eight_is_least_divisible_by_1920_l1657_165792


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l1657_165717

theorem binomial_expansion_problem (n : ℕ) : 
  ((-2 : ℤ) ^ n = 64) →
  (n = 6 ∧ Nat.choose n 2 * 9 = 135) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l1657_165717


namespace NUMINAMATH_CALUDE_fish_tank_balls_l1657_165761

/-- The number of goldfish in the tank -/
def num_goldfish : ℕ := 3

/-- The number of platyfish in the tank -/
def num_platyfish : ℕ := 10

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish

theorem fish_tank_balls : total_balls = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_balls_l1657_165761


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1657_165728

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 - 2*x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1657_165728


namespace NUMINAMATH_CALUDE_tims_manicure_cost_l1657_165708

/-- The total cost of a manicure with tip -/
def total_cost (base_cost : ℝ) (tip_percentage : ℝ) : ℝ :=
  base_cost * (1 + tip_percentage)

/-- Theorem: Tim's total payment for a $30 manicure with a 30% tip is $39 -/
theorem tims_manicure_cost :
  total_cost 30 0.3 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tims_manicure_cost_l1657_165708


namespace NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l1657_165701

/-- Represents a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : ℝ
  k : ℝ
  pos_a : a > 0
  pos_b : b > 0

/-- The asymptotes of the hyperbola -/
def asymptotes (slope : ℝ) (y_intercept1 y_intercept2 : ℝ) :=
  (fun x => slope * x + y_intercept1, fun x => -slope * x + y_intercept2)

theorem hyperbola_a_plus_h_value
  (slope : ℝ)
  (y_intercept1 y_intercept2 : ℝ)
  (point_x point_y : ℝ)
  (h : Hyperbola)
  (asym : asymptotes slope y_intercept1 y_intercept2 = 
    (fun x => 3 * x + 4, fun x => -3 * x + 2))
  (point_on_hyperbola : (point_x, point_y) = (1, 8))
  (hyperbola_eq : ∀ x y, 
    (y - h.k)^2 / h.a^2 - (x - h.h)^2 / h.b^2 = 1 ↔ 
    (fun x y => (y - h.k)^2 / h.a^2 - (x - h.h)^2 / h.b^2 = 1) x y) :
  h.a + h.h = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_a_plus_h_value_l1657_165701


namespace NUMINAMATH_CALUDE_domain_exact_domain_contains_l1657_165777

-- Define the function
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + x - a

-- Theorem 1: When the domain is exactly (-2, 3), a = -6
theorem domain_exact (a : ℝ) : 
  (∀ x, -2 < x ∧ x < 3 ↔ f a x > 0) → a = -6 :=
sorry

-- Theorem 2: When the domain contains (-2, 3), a ≤ -6
theorem domain_contains (a : ℝ) :
  (∀ x, -2 < x ∧ x < 3 → f a x > 0) → a ≤ -6 :=
sorry

end NUMINAMATH_CALUDE_domain_exact_domain_contains_l1657_165777


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l1657_165706

theorem sum_of_roots_equation (x : ℝ) : 
  ((-15 * x) / (x^2 - 1) = (3 * x) / (x + 1) - 9 / (x - 1)) →
  (∃ y : ℝ, (-15 * y) / (y^2 - 1) = (3 * y) / (y + 1) - 9 / (y - 1) ∧ y ≠ x) →
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l1657_165706


namespace NUMINAMATH_CALUDE_notebook_count_l1657_165716

theorem notebook_count : ∃ (n : ℕ), n > 0 ∧ n + (n + 50) = 110 ∧ n = 30 := by sorry

end NUMINAMATH_CALUDE_notebook_count_l1657_165716


namespace NUMINAMATH_CALUDE_intersection_and_complement_when_a_is_3_range_of_a_when_M_subset_N_l1657_165783

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def N (a : ℝ) : Set ℝ := {x | 1 - a ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part 1
theorem intersection_and_complement_when_a_is_3 :
  (M ∩ N 3 = {x | -2 ≤ x ∧ x ≤ 6}) ∧
  (Set.univ \ N 3 = {x | x < -2 ∨ x > 7}) := by
  sorry

-- Theorem for part 2
theorem range_of_a_when_M_subset_N :
  (∀ a : ℝ, M ⊆ N a) ↔ (∀ a : ℝ, a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_complement_when_a_is_3_range_of_a_when_M_subset_N_l1657_165783


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1657_165771

theorem perfect_square_condition (m : ℝ) : 
  (∃ x : ℝ, ∃ k : ℝ, x^2 + 2*(m-3)*x + 16 = k^2) → (m = 7 ∨ m = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1657_165771


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1657_165779

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → x^2 + a*x - 3*a < 0) → a > (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1657_165779


namespace NUMINAMATH_CALUDE_mildred_orange_collection_l1657_165784

/-- Mildred's orange collection problem -/
theorem mildred_orange_collection (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 77 → additional = 2 → total = initial + additional → total = 79 := by
  sorry

end NUMINAMATH_CALUDE_mildred_orange_collection_l1657_165784


namespace NUMINAMATH_CALUDE_three_lines_exist_l1657_165724

-- Define the line segment AB
def AB : ℝ := 10

-- Define the distances from points A and B to line l
def distance_A_to_l : ℝ := 6
def distance_B_to_l : ℝ := 4

-- Define a function that counts the number of lines satisfying the conditions
def count_lines : ℕ := sorry

-- Theorem statement
theorem three_lines_exist : count_lines = 3 := by sorry

end NUMINAMATH_CALUDE_three_lines_exist_l1657_165724


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l1657_165747

theorem complex_sum_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1) 
  (h4 : a^3 / (b*c) + b^3 / (a*c) + c^3 / (a*b) = 3) : 
  Complex.abs (a + b + c) = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l1657_165747


namespace NUMINAMATH_CALUDE_amanda_remaining_budget_l1657_165789

/- Define the budgets -/
def samuel_budget : ℚ := 25
def kevin_budget : ℚ := 20
def laura_budget : ℚ := 18
def amanda_budget : ℚ := 15

/- Define the regular ticket prices -/
def samuel_ticket_price : ℚ := 14
def kevin_ticket_price : ℚ := 10
def laura_ticket_price : ℚ := 10
def amanda_ticket_price : ℚ := 8

/- Define the discount rates -/
def general_discount : ℚ := 0.1
def student_discount : ℚ := 0.1

/- Define Samuel's additional expenses -/
def samuel_drink : ℚ := 6
def samuel_popcorn : ℚ := 3
def samuel_candy : ℚ := 1

/- Define Kevin's additional expense -/
def kevin_combo : ℚ := 7

/- Define Laura's additional expenses -/
def laura_popcorn : ℚ := 4
def laura_drink : ℚ := 2

/- Calculate discounted ticket prices -/
def samuel_discounted_ticket : ℚ := samuel_ticket_price * (1 - general_discount)
def kevin_discounted_ticket : ℚ := kevin_ticket_price * (1 - general_discount)
def laura_discounted_ticket : ℚ := laura_ticket_price * (1 - general_discount)
def amanda_discounted_ticket : ℚ := amanda_ticket_price * (1 - general_discount) * (1 - student_discount)

/- Define the theorem -/
theorem amanda_remaining_budget :
  amanda_budget - amanda_discounted_ticket = 8.52 := by sorry

end NUMINAMATH_CALUDE_amanda_remaining_budget_l1657_165789


namespace NUMINAMATH_CALUDE_hikers_meeting_point_l1657_165726

/-- Represents the distance between two hikers at any given time -/
structure HikerDistance where
  total : ℝ := 100
  from_a : ℝ
  from_b : ℝ

/-- Calculates the distance traveled by hiker A in t hours -/
def distance_a (t : ℝ) : ℝ := 5 * t

/-- Calculates the distance traveled by hiker B in t hours -/
def distance_b (t : ℝ) : ℝ := t * (4 + 0.125 * (t - 1))

/-- Represents the meeting point of the two hikers -/
def meeting_point (t : ℝ) : HikerDistance :=
  { total := 100
  , from_a := distance_a t
  , from_b := distance_b t }

/-- The time at which the hikers meet -/
def meeting_time : ℕ := 10

theorem hikers_meeting_point :
  let mp := meeting_point meeting_time
  mp.from_b - mp.from_a = 2.5 := by sorry

end NUMINAMATH_CALUDE_hikers_meeting_point_l1657_165726


namespace NUMINAMATH_CALUDE_total_money_divided_l1657_165780

/-- Proof of the total amount of money divided among three people --/
theorem total_money_divided (A B C : ℕ) : 
  A = 600 →                    -- A's share is 600
  A = (2 / 5) * (B + C) →      -- A receives 2/5 as much as B and C together
  B = (1 / 5) * (A + C) →      -- B receives 1/5 as much as A and C together
  A + B + C = 2100             -- The total amount is 2100
  := by sorry

end NUMINAMATH_CALUDE_total_money_divided_l1657_165780


namespace NUMINAMATH_CALUDE_x_one_value_l1657_165793

theorem x_one_value (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1)
  (h_equation : (1-x₁)^2 + (x₁-x₂)^2 + (x₂-x₃)^2 + (x₃-x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_x_one_value_l1657_165793


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1657_165742

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 15 ∧ b = 36 ∧ c^2 = a^2 + b^2 → c = 39 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1657_165742


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1657_165738

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1657_165738


namespace NUMINAMATH_CALUDE_tim_score_l1657_165743

def single_line_score : ℕ := 1000
def tetris_multiplier : ℕ := 8
def consecutive_multiplier : ℕ := 2

def calculate_score (singles : ℕ) (regular_tetrises : ℕ) (consecutive_tetrises : ℕ) : ℕ :=
  singles * single_line_score +
  regular_tetrises * (single_line_score * tetris_multiplier) +
  consecutive_tetrises * (single_line_score * tetris_multiplier * consecutive_multiplier)

theorem tim_score : calculate_score 6 2 2 = 54000 := by
  sorry

end NUMINAMATH_CALUDE_tim_score_l1657_165743


namespace NUMINAMATH_CALUDE_num_distinguishable_triangles_eq_960_l1657_165751

/-- Represents the number of colors available for the small triangles -/
def num_colors : ℕ := 8

/-- Represents the number of small triangles needed to form a large triangle -/
def triangles_per_large : ℕ := 4

/-- Represents the number of corner triangles in a large triangle -/
def num_corners : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of distinguishable large triangles -/
def num_distinguishable_triangles : ℕ :=
  ((num_colors + 
    num_colors * (num_colors - 1) + 
    choose num_colors num_corners) * num_colors)

/-- The main theorem stating the number of distinguishable large triangles -/
theorem num_distinguishable_triangles_eq_960 :
  num_distinguishable_triangles = 960 := by sorry

end NUMINAMATH_CALUDE_num_distinguishable_triangles_eq_960_l1657_165751


namespace NUMINAMATH_CALUDE_max_plate_valid_l1657_165733

-- Define a custom type for characters that can be on a number plate
inductive PlateChar
| Zero
| Six
| Nine
| H
| O

-- Define a function to check if a character looks the same upside down
def looks_same_upside_down (c : PlateChar) : Bool :=
  match c with
  | PlateChar.Zero => true
  | PlateChar.H => true
  | PlateChar.O => true
  | _ => false

-- Define a function to get the upside down version of a character
def upside_down (c : PlateChar) : PlateChar :=
  match c with
  | PlateChar.Six => PlateChar.Nine
  | PlateChar.Nine => PlateChar.Six
  | c => c

-- Define a number plate as a list of PlateChar
def NumberPlate := List PlateChar

-- Define the specific number plate we want to check
def max_plate : NumberPlate :=
  [PlateChar.Six, PlateChar.Zero, PlateChar.H, PlateChar.O, PlateChar.H, PlateChar.Zero, PlateChar.Nine]

-- Theorem: Max's plate is valid when turned upside down
theorem max_plate_valid : 
  max_plate.reverse.map upside_down = max_plate :=
by sorry

end NUMINAMATH_CALUDE_max_plate_valid_l1657_165733


namespace NUMINAMATH_CALUDE_lucille_earnings_l1657_165737

/-- Calculates the earnings from weeding a specific area -/
def calculate_earnings (small medium large : ℕ) : ℕ :=
  4 * small + 8 * medium + 12 * large

/-- Calculates the total cost of items after discount and tax -/
def calculate_total_cost (price : ℕ) (discount_rate tax_rate : ℚ) : ℕ :=
  let discounted_price := price - (price * discount_rate).floor
  (discounted_price + (discounted_price * tax_rate).ceil).toNat

theorem lucille_earnings : 
  let flower_bed := calculate_earnings 6 3 2
  let vegetable_patch := calculate_earnings 10 2 2
  let half_grass := calculate_earnings 10 5 1
  let new_area := calculate_earnings 7 4 1
  let total_earnings := flower_bed + vegetable_patch + half_grass + new_area
  let soda_snack_cost := calculate_total_cost 149 (1/10) (12/100)
  total_earnings - soda_snack_cost = 166 := by sorry

end NUMINAMATH_CALUDE_lucille_earnings_l1657_165737


namespace NUMINAMATH_CALUDE_equation_solution_l1657_165746

theorem equation_solution : 
  {x : ℝ | (x^3 + 3*x^2 - x) / (x^2 + 4*x + 3) + x = -7} = {-5/2, -4} := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1657_165746


namespace NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l1657_165790

theorem sum_of_fractions : 
  7 / 12 + 11 / 15 = 79 / 60 :=
by sorry

theorem simplest_form : 
  ∀ n m : ℕ, n ≠ 0 → m ≠ 0 → Nat.gcd n m = 1 → (n : ℚ) / m = 79 / 60 → n = 79 ∧ m = 60 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_simplest_form_l1657_165790


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1657_165787

theorem geometric_sequence_common_ratio_sum 
  (k p r : ℝ) 
  (h_nonconstant_p : p ≠ 1) 
  (h_nonconstant_r : r ≠ 1) 
  (h_different_ratios : p ≠ r) 
  (h_relation : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
  p + r = 5 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_sum_l1657_165787


namespace NUMINAMATH_CALUDE_triangle_area_OAB_l1657_165722

/-- Given a line passing through (0, -2) that intersects the parabola y² = 16x at points A and B,
    where the y-coordinates of A and B satisfy y₁² - y₂² = 1, 
    prove that the area of triangle OAB (where O is the origin) is 1/16. -/
theorem triangle_area_OAB : 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (∃ m : ℝ, x₁ = m * y₁ + 2 * m ∧ x₂ = m * y₂ + 2 * m) → -- Line equation
    y₁^2 = 16 * x₁ →                                      -- A satisfies parabola equation
    y₂^2 = 16 * x₂ →                                      -- B satisfies parabola equation
    y₁^2 - y₂^2 = 1 →                                     -- Given condition
    (1/2 : ℝ) * |x₁ * y₂ - x₂ * y₁| = 1/16 :=             -- Area of triangle OAB
by sorry

end NUMINAMATH_CALUDE_triangle_area_OAB_l1657_165722


namespace NUMINAMATH_CALUDE_distance_to_origin_l1657_165759

theorem distance_to_origin (a : ℝ) : |a| = 3 → (a - 2 = 1 ∨ a - 2 = -5) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1657_165759


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_186_l1657_165764

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x + 4

-- Theorem statement
theorem f_g_f_3_equals_186 : f (g (f 3)) = 186 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_186_l1657_165764


namespace NUMINAMATH_CALUDE_intersection_range_distance_when_b_is_one_l1657_165748

/-- The line y = x + b intersects the ellipse x^2/2 + y^2 = 1 at two distinct points -/
def intersects_at_two_points (b : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ + b ∧ y₂ = x₂ + b ∧
    x₁^2/2 + y₁^2 = 1 ∧ x₂^2/2 + y₂^2 = 1

/-- The range of b for which the line intersects the ellipse at two distinct points -/
theorem intersection_range :
  ∀ b : ℝ, intersects_at_two_points b ↔ -Real.sqrt 3 < b ∧ b < Real.sqrt 3 :=
sorry

/-- The distance between intersection points when b = 1 -/
theorem distance_when_b_is_one :
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = x₁ + 1 ∧ y₂ = x₂ + 1 ∧
    x₁^2/2 + y₁^2 = 1 ∧ x₂^2/2 + y₂^2 = 1 ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_distance_when_b_is_one_l1657_165748


namespace NUMINAMATH_CALUDE_archer_weekly_spending_is_1056_l1657_165768

/-- The archer's weekly spending on arrows -/
def archer_weekly_spending (shots_per_day : ℕ) (days_per_week : ℕ) 
  (recovery_rate : ℚ) (arrow_cost : ℚ) (team_payment_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let unrecovered_arrows := total_shots * (1 - recovery_rate)
  let total_cost := unrecovered_arrows * arrow_cost
  total_cost * (1 - team_payment_rate)

/-- Theorem: The archer spends $1056 on arrows per week -/
theorem archer_weekly_spending_is_1056 :
  archer_weekly_spending 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end NUMINAMATH_CALUDE_archer_weekly_spending_is_1056_l1657_165768


namespace NUMINAMATH_CALUDE_f_positive_iff_x_gt_one_l1657_165769

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x, (deriv f) x > f x)
variable (h2 : f 1 = 0)

-- State the theorem
theorem f_positive_iff_x_gt_one :
  (∀ x, f x > 0 ↔ x > 1) :=
sorry

end NUMINAMATH_CALUDE_f_positive_iff_x_gt_one_l1657_165769


namespace NUMINAMATH_CALUDE_track_circumference_l1657_165799

/-- Represents the circular track and the movement of A and B -/
structure TrackSystem where
  /-- Half of the track's circumference in yards -/
  half_circumference : ℝ
  /-- Speed of A in yards per unit time -/
  speed_a : ℝ
  /-- Speed of B in yards per unit time -/
  speed_b : ℝ

/-- The theorem stating the conditions and the result to be proven -/
theorem track_circumference (ts : TrackSystem) 
  (h1 : ts.speed_a > 0 ∧ ts.speed_b > 0)  -- A and B travel at uniform (positive) speeds
  (h2 : ts.speed_a + ts.speed_b = ts.half_circumference / 75)  -- They meet after B travels 150 yards
  (h3 : 2 * ts.half_circumference - 90 = (ts.half_circumference + 90) * (ts.speed_a / ts.speed_b)) 
      -- Second meeting condition
  : ts.half_circumference = 360 :=
sorry

end NUMINAMATH_CALUDE_track_circumference_l1657_165799


namespace NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1657_165709

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^(4*x + 2) * (4 : ℝ)^(3*x + 7) = (8 : ℝ)^(5*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_exponential_equation_l1657_165709


namespace NUMINAMATH_CALUDE_complex_number_problem_l1657_165730

theorem complex_number_problem (z : ℂ) (m n : ℝ) :
  (Complex.abs z = 2 * Real.sqrt 10) →
  (Complex.im ((3 - Complex.I) * z) = 0) →
  (Complex.re z < 0) →
  (2 * z^2 + m * z - n = 0) →
  (∃ (a b : ℝ), z = Complex.mk a b ∧ ((a = 2 ∧ b = -6) ∨ (a = -2 ∧ b = 6))) ∧
  (m + n = -72) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1657_165730


namespace NUMINAMATH_CALUDE_two_tangents_from_origin_l1657_165786

/-- The function f(x) = -x^3 + 3x^2 + 1 -/
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 1

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

/-- A point (t, f(t)) on the curve y = f(x) -/
def point_on_curve (t : ℝ) : ℝ × ℝ := (t, f t)

/-- The slope of the tangent line at point (t, f(t)) -/
def tangent_slope (t : ℝ) : ℝ := f' t

/-- The equation for finding points of tangency -/
def tangency_equation (t : ℝ) : Prop := 2*t^3 - 3*t^2 + 1 = 0

theorem two_tangents_from_origin :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    tangency_equation t₁ ∧ 
    tangency_equation t₂ ∧ 
    (∀ t, tangency_equation t → t = t₁ ∨ t = t₂) :=
sorry

end NUMINAMATH_CALUDE_two_tangents_from_origin_l1657_165786


namespace NUMINAMATH_CALUDE_sqrt_180_equals_6_sqrt_5_l1657_165763

theorem sqrt_180_equals_6_sqrt_5 : Real.sqrt 180 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_180_equals_6_sqrt_5_l1657_165763


namespace NUMINAMATH_CALUDE_points_per_correct_answer_l1657_165700

theorem points_per_correct_answer 
  (total_problems : ℕ) 
  (total_score : ℕ) 
  (wrong_answers : ℕ) 
  (points_per_wrong : ℕ) 
  (h1 : total_problems = 25)
  (h2 : total_score = 85)
  (h3 : wrong_answers = 3)
  (h4 : points_per_wrong = 1) :
  (total_score + wrong_answers * points_per_wrong) / (total_problems - wrong_answers) = 4 := by
sorry

end NUMINAMATH_CALUDE_points_per_correct_answer_l1657_165700


namespace NUMINAMATH_CALUDE_bottles_left_after_purchase_l1657_165776

/-- Given a store shelf with bottles of milk, prove the number of bottles left after purchases. -/
theorem bottles_left_after_purchase (initial : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) 
  (h1 : initial = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6) :
  initial - (jason_buys + harry_buys) = 24 := by
  sorry

#check bottles_left_after_purchase

end NUMINAMATH_CALUDE_bottles_left_after_purchase_l1657_165776
