import Mathlib

namespace NUMINAMATH_CALUDE_stating_bus_students_theorem_l2986_298657

/-- 
Calculates the number of students on a bus after a series of stops where 
students get off and on, given an initial number of students.
-/
def students_after_stops (initial : ℚ) (fraction_off : ℚ) (num_stops : ℕ) (new_students : ℚ) : ℚ :=
  (initial * (1 - fraction_off)^num_stops) + new_students

/-- 
Theorem stating that given 72 initial students, with 1/3 getting off at each of 
the first four stops, and 12 new students boarding at the fifth stop, 
the final number of students is 236/9.
-/
theorem bus_students_theorem : 
  students_after_stops 72 (1/3) 4 12 = 236/9 := by
  sorry

end NUMINAMATH_CALUDE_stating_bus_students_theorem_l2986_298657


namespace NUMINAMATH_CALUDE_scientific_notation_of_14nm_l2986_298654

theorem scientific_notation_of_14nm (nm14 : ℝ) (h : nm14 = 0.000000014) :
  ∃ (a b : ℝ), a = 1.4 ∧ b = -8 ∧ nm14 = a * (10 : ℝ) ^ b :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_14nm_l2986_298654


namespace NUMINAMATH_CALUDE_natural_numbers_less_than_two_l2986_298642

theorem natural_numbers_less_than_two : 
  {n : ℕ | n < 2} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_natural_numbers_less_than_two_l2986_298642


namespace NUMINAMATH_CALUDE_tan_alpha_and_tan_alpha_minus_pi_fourth_l2986_298607

theorem tan_alpha_and_tan_alpha_minus_pi_fourth (α : Real) 
  (h : Real.tan (α / 2) = 1 / 2) : 
  Real.tan α = 4 / 3 ∧ Real.tan (α - π / 4) = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_tan_alpha_minus_pi_fourth_l2986_298607


namespace NUMINAMATH_CALUDE_acute_triangle_condition_l2986_298656

/-- A triangle is represented by its incircle radius and circumcircle radius -/
structure Triangle where
  r : ℝ  -- radius of the incircle
  R : ℝ  -- radius of the circumcircle

/-- A triangle is acute if all its angles are less than 90 degrees -/
def Triangle.isAcute (t : Triangle) : Prop :=
  sorry  -- definition of an acute triangle

/-- The main theorem: if R < r(√2 + 1), then the triangle is acute -/
theorem acute_triangle_condition (t : Triangle) 
  (h : t.R < t.r * (Real.sqrt 2 + 1)) : t.isAcute :=
sorry

end NUMINAMATH_CALUDE_acute_triangle_condition_l2986_298656


namespace NUMINAMATH_CALUDE_five_digit_number_theorem_l2986_298663

def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 6 ∨ d = 8

def are_distinct (p q r s t : ℕ) : Prop :=
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

theorem five_digit_number_theorem (p q r s t : ℕ) :
  is_valid_digit p ∧ is_valid_digit q ∧ is_valid_digit r ∧ is_valid_digit s ∧ is_valid_digit t ∧
  are_distinct p q r s t ∧
  (100 * p + 10 * q + r) % 6 = 0 ∧
  (100 * q + 10 * r + s) % 8 = 0 ∧
  (100 * r + 10 * s + t) % 3 = 0 →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_five_digit_number_theorem_l2986_298663


namespace NUMINAMATH_CALUDE_one_eighth_of_2_36_equals_2_y_l2986_298676

theorem one_eighth_of_2_36_equals_2_y (y : ℕ) : (1 / 8 : ℝ) * 2^36 = 2^y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_2_36_equals_2_y_l2986_298676


namespace NUMINAMATH_CALUDE_powerless_common_divisor_l2986_298639

def is_powerless_digit (d : ℕ) : Prop :=
  d ≤ 9 ∧ d ≠ 0 ∧ d ≠ 1 ∧ d ≠ 4 ∧ d ≠ 8 ∧ d ≠ 9

def is_powerless_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ is_powerless_digit (n / 10) ∧ is_powerless_digit (n % 10)

def smallest_powerless : ℕ := 22
def largest_powerless : ℕ := 77

theorem powerless_common_divisor :
  is_powerless_number smallest_powerless ∧
  is_powerless_number largest_powerless ∧
  smallest_powerless % 11 = 0 ∧
  largest_powerless % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_powerless_common_divisor_l2986_298639


namespace NUMINAMATH_CALUDE_existence_of_special_quadratic_l2986_298662

theorem existence_of_special_quadratic (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧
    (Nat.gcd a n = 1) ∧
    (Nat.gcd b n = 1) ∧
    (n ∣ (a^2 + b)) ∧
    (∀ x : ℕ, x ≥ 1 → ∃ p : ℕ, Prime p ∧ p ∣ ((x + a)^2 + b) ∧ ¬(p ∣ n)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_quadratic_l2986_298662


namespace NUMINAMATH_CALUDE_pi_half_irrational_l2986_298628

theorem pi_half_irrational : Irrational (π / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_pi_half_irrational_l2986_298628


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l2986_298688

theorem smallest_n_congruence : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(503 * m.val ≡ 1019 * m.val [ZMOD 48])) ∧
  (503 * n.val ≡ 1019 * n.val [ZMOD 48]) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l2986_298688


namespace NUMINAMATH_CALUDE_polar_cartesian_equivalence_l2986_298664

/-- The curve C in polar coordinates -/
def polar_equation (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

/-- The curve C in Cartesian coordinates -/
def cartesian_equation (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0

/-- Theorem stating the equivalence of polar and Cartesian equations for curve C -/
theorem polar_cartesian_equivalence :
  ∀ (x y ρ θ : ℝ), 
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (polar_equation ρ θ ↔ cartesian_equation x y) := by
  sorry

end NUMINAMATH_CALUDE_polar_cartesian_equivalence_l2986_298664


namespace NUMINAMATH_CALUDE_simplify_expression_l2986_298632

theorem simplify_expression : (45000 - 32000) * 10 + (2500 / 5) - 21005 * 3 = 67485 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2986_298632


namespace NUMINAMATH_CALUDE_complement_of_A_l2986_298677

-- Define the set A
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_of_A : 
  (Set.univ \ A : Set ℝ) = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2986_298677


namespace NUMINAMATH_CALUDE_sin_product_equality_l2986_298618

theorem sin_product_equality : 
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 
  (Real.cos (20 * π / 180) - 1/2) / 8 := by sorry

end NUMINAMATH_CALUDE_sin_product_equality_l2986_298618


namespace NUMINAMATH_CALUDE_density_of_M_l2986_298681

def M : Set ℝ :=
  {r : ℝ | ∃ (m n : ℕ+), r = (m + n) / Real.sqrt (m^2 + n^2)}

theorem density_of_M : ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y →
  ∃ (z : ℝ), z ∈ M ∧ x < z ∧ z < y :=
by sorry

end NUMINAMATH_CALUDE_density_of_M_l2986_298681


namespace NUMINAMATH_CALUDE_wrapping_paper_usage_l2986_298640

theorem wrapping_paper_usage (total_fraction : ℚ) (num_presents : ℕ) :
  total_fraction = 3 / 10 →
  num_presents = 3 →
  (total_fraction / num_presents : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_usage_l2986_298640


namespace NUMINAMATH_CALUDE_set_inclusion_implies_parameter_range_l2986_298600

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0}

theorem set_inclusion_implies_parameter_range :
  ∀ a : ℝ, A ⊆ B a → a ∈ Set.Iic (-2) := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_implies_parameter_range_l2986_298600


namespace NUMINAMATH_CALUDE_smallest_a1_l2986_298638

/-- A sequence of positive real numbers satisfying the given recurrence relation -/
def SequenceA (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n > 1, a n = 9 * a (n - 1) - 2 * n)

/-- The theorem stating the smallest possible value of a₁ -/
theorem smallest_a1 (a : ℕ → ℝ) (h : SequenceA a) :
  ∀ a1 : ℝ, a 1 ≥ a1 → a1 ≥ 19/36 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a1_l2986_298638


namespace NUMINAMATH_CALUDE_ten_people_prob_l2986_298684

/-- Represents the number of valid arrangements where no two adjacent people are standing
    for n people around a circular table. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The probability of no two adjacent people standing when n people
    each flip a fair coin around a circular table. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / (2 ^ n)

/-- The main theorem stating the probability for 10 people. -/
theorem ten_people_prob : noAdjacentStandingProb 10 = 123 / 1024 := by
  sorry


end NUMINAMATH_CALUDE_ten_people_prob_l2986_298684


namespace NUMINAMATH_CALUDE_equation_holds_for_all_y_l2986_298673

theorem equation_holds_for_all_y (x : ℚ) : 
  (∀ y : ℚ, 8 * x * y - 12 * y + 2 * x - 3 = 0) ↔ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_for_all_y_l2986_298673


namespace NUMINAMATH_CALUDE_adjacent_probability_l2986_298694

/-- The number of students in the arrangement -/
def num_students : ℕ := 9

/-- The number of rows in the seating grid -/
def num_rows : ℕ := 3

/-- The number of columns in the seating grid -/
def num_cols : ℕ := 3

/-- The number of ways two students can be adjacent in a row -/
def row_adjacencies : ℕ := num_rows * (num_cols - 1)

/-- The number of ways two students can be adjacent along a main diagonal -/
def diagonal_adjacencies : ℕ := 2 * (num_rows - 1)

/-- The total number of adjacent positions (row + diagonal) -/
def total_adjacencies : ℕ := row_adjacencies + diagonal_adjacencies

/-- The probability of two specific students being adjacent in a 3x3 grid -/
theorem adjacent_probability :
  (total_adjacencies * 2 : ℚ) / (num_students * (num_students - 1)) = 13 / 36 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_probability_l2986_298694


namespace NUMINAMATH_CALUDE_count_less_than_04_l2986_298651

def numbers : Finset ℚ := {0.8, 1/2, 0.3, 1/3}

theorem count_less_than_04 : Finset.card (numbers.filter (λ x => x < 0.4)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_04_l2986_298651


namespace NUMINAMATH_CALUDE_common_chord_equation_l2986_298680

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 + 4*x - 6*y + 12 = 0) ∧ (x^2 + y^2 - 2*x - 14*y + 15 = 0) →
  (6*x + 8*y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2986_298680


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2986_298635

theorem complex_equation_solution (i : ℂ) (z : ℂ) (h1 : i * i = -1) (h2 : i * z = 1) :
  z = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2986_298635


namespace NUMINAMATH_CALUDE_original_number_proof_l2986_298613

theorem original_number_proof (x : ℝ) : ((3 * x^2 + 8) * 2) / 4 = 56 → x = 2 * Real.sqrt 78 / 3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2986_298613


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2986_298666

theorem complex_modulus_problem (w z : ℂ) :
  w * z = 24 - 10 * I ∧ Complex.abs w = Real.sqrt 29 →
  Complex.abs z = (26 * Real.sqrt 29) / 29 :=
by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2986_298666


namespace NUMINAMATH_CALUDE_friends_bill_split_l2986_298649

/-- The cost each friend pays when splitting a bill equally -/
def split_bill (num_friends : ℕ) (item1_cost : ℚ) (item2_count : ℕ) (item2_cost : ℚ)
                (item3_count : ℕ) (item3_cost : ℚ) (item4_count : ℕ) (item4_cost : ℚ) : ℚ :=
  (item1_cost + item2_count * item2_cost + item3_count * item3_cost + item4_count * item4_cost) / num_friends

/-- Theorem: When 5 friends split a bill with the given items, each pays $11 -/
theorem friends_bill_split :
  split_bill 5 10 5 5 4 (5/2) 5 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_friends_bill_split_l2986_298649


namespace NUMINAMATH_CALUDE_ptolemy_special_cases_l2986_298633

/-- Ptolemy's theorem for cyclic quadrilaterals -/
def ptolemyTheorem (a b c d e f : ℝ) : Prop := a * c + b * d = e * f

/-- A cyclic quadrilateral with one side zero -/
def cyclicQuadrilateralOneSideZero (b c d e f : ℝ) : Prop :=
  ptolemyTheorem 0 b c d e f

/-- A rectangle -/
def rectangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0

/-- An isosceles trapezoid -/
def isoscelesTrapezoid (a b c e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ e > 0

theorem ptolemy_special_cases :
  (∀ b c d e f : ℝ, cyclicQuadrilateralOneSideZero b c d e f → b * d = e * f) ∧
  (∀ a b : ℝ, rectangle a b → 2 * a * b = a^2 + b^2) ∧
  (∀ a b c e : ℝ, isoscelesTrapezoid a b c e → e^2 = c^2 + a * b) :=
sorry

end NUMINAMATH_CALUDE_ptolemy_special_cases_l2986_298633


namespace NUMINAMATH_CALUDE_bacterium_probability_l2986_298614

/-- The probability of selecting a single bacterium in a smaller volume from a larger volume --/
theorem bacterium_probability (total_volume small_volume : ℝ) (h1 : total_volume > 0) 
  (h2 : small_volume > 0) (h3 : small_volume ≤ total_volume) :
  small_volume / total_volume = 0.05 → 
  (total_volume = 2 ∧ small_volume = 0.1) := by
  sorry


end NUMINAMATH_CALUDE_bacterium_probability_l2986_298614


namespace NUMINAMATH_CALUDE_volume_surface_area_ratio_eight_cubes_l2986_298617

/-- A shape created by joining unit cubes in a line -/
structure LineCubes where
  num_cubes : ℕ

/-- Calculate the volume of the shape -/
def volume (shape : LineCubes) : ℕ :=
  shape.num_cubes

/-- Calculate the surface area of the shape -/
def surface_area (shape : LineCubes) : ℕ :=
  2 * shape.num_cubes + 2 * 4

/-- The ratio of volume to surface area for a shape with 8 unit cubes -/
theorem volume_surface_area_ratio_eight_cubes :
  let shape : LineCubes := { num_cubes := 8 }
  (volume shape : ℚ) / (surface_area shape : ℚ) = 4 / 9 := by
  sorry


end NUMINAMATH_CALUDE_volume_surface_area_ratio_eight_cubes_l2986_298617


namespace NUMINAMATH_CALUDE_right_angle_and_trig_relation_l2986_298622

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)
  (sum_angles : A + B + C = 180)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- Define the condition for right angle
def is_right_angled (t : Triangle) : Prop :=
  t.C = 90

-- Define the condition for equal sum of sine and cosine
def equal_sin_cos_sum (t : Triangle) : Prop :=
  Real.cos t.A + Real.sin t.A = Real.cos t.B + Real.sin t.B

-- Theorem statement
theorem right_angle_and_trig_relation (t : Triangle) :
  (is_right_angled t → equal_sin_cos_sum t) ∧
  ∃ t', equal_sin_cos_sum t' ∧ ¬is_right_angled t' :=
sorry

end NUMINAMATH_CALUDE_right_angle_and_trig_relation_l2986_298622


namespace NUMINAMATH_CALUDE_alex_age_l2986_298671

/-- Given the ages of Alex, Bella, and Carlos, prove that Alex is 20 years old. -/
theorem alex_age (bella_age carlos_age alex_age : ℕ) : 
  bella_age = 21 →
  carlos_age = bella_age + 5 →
  alex_age = carlos_age - 6 →
  alex_age = 20 := by
sorry

end NUMINAMATH_CALUDE_alex_age_l2986_298671


namespace NUMINAMATH_CALUDE_expansion_term_count_l2986_298667

/-- The number of terms in the expansion of a product of sums of distinct variables -/
def expansion_terms (x y z : ℕ) : ℕ := x * y * z

/-- The first factor (a+b+c) has 3 terms -/
def factor1_terms : ℕ := 3

/-- The second factor (d+e+f+g) has 4 terms -/
def factor2_terms : ℕ := 4

/-- The third factor (h+i) has 2 terms -/
def factor3_terms : ℕ := 2

theorem expansion_term_count : 
  expansion_terms factor1_terms factor2_terms factor3_terms = 24 := by
  sorry

end NUMINAMATH_CALUDE_expansion_term_count_l2986_298667


namespace NUMINAMATH_CALUDE_calculation_proof_l2986_298641

theorem calculation_proof :
  ((-7) * 5 - (-36) / 4 = -26) ∧
  (-1^4 - (1-0.4) * (1/3) * (2-3^2) = 0.4) := by
sorry

end NUMINAMATH_CALUDE_calculation_proof_l2986_298641


namespace NUMINAMATH_CALUDE_train_speed_l2986_298698

/-- Proves that a train with given length, crossing a bridge of given length in a specific time, has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 →
  bridge_length = 132 →
  crossing_time = 24.198064154867613 →
  (train_length + bridge_length) / crossing_time * 3.6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2986_298698


namespace NUMINAMATH_CALUDE_smallest_p_is_12_l2986_298696

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- The property that for all n ≥ p, there exists z ∈ T such that z^n = 1 -/
def has_root_in_T (p : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ p → ∃ z ∈ T, z^n = 1

/-- 12 is the smallest positive integer satisfying the property -/
theorem smallest_p_is_12 : 
  has_root_in_T 12 ∧ ∀ p : ℕ, 0 < p → p < 12 → ¬has_root_in_T p :=
sorry

end NUMINAMATH_CALUDE_smallest_p_is_12_l2986_298696


namespace NUMINAMATH_CALUDE_unique_integer_pair_solution_l2986_298665

theorem unique_integer_pair_solution : 
  ∃! (x y : ℤ), Real.sqrt (x - Real.sqrt (x + 23)) = 2 * Real.sqrt 2 - y := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_pair_solution_l2986_298665


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l2986_298611

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i :=
by sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l2986_298611


namespace NUMINAMATH_CALUDE_intersection_implies_z_l2986_298602

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, 2, z * i}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem intersection_implies_z (z : ℂ) : M z ∩ N = {4} → z = -4 * i := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_z_l2986_298602


namespace NUMINAMATH_CALUDE_salary_distribution_difference_l2986_298606

/-- Proves that given a salary distribution among A, B, C, D in the proportion of 2 : 3 : 4 : 6,
    where B's share is $1050, the difference between D's and C's share is $700. -/
theorem salary_distribution_difference (total : ℕ) (a b c d : ℕ) : 
  a + b + c + d = total →
  2 * total = 15 * b →
  b = 1050 →
  6 * a = 2 * total →
  6 * b = 3 * total →
  6 * c = 4 * total →
  6 * d = 6 * total →
  d - c = 700 := by
  sorry

end NUMINAMATH_CALUDE_salary_distribution_difference_l2986_298606


namespace NUMINAMATH_CALUDE_jasmine_swimming_totals_l2986_298668

/-- Jasmine's weekly swimming routine -/
structure SwimmingRoutine where
  monday_laps : ℕ
  tuesday_laps : ℕ
  tuesday_aerobics : ℕ
  wednesday_laps : ℕ
  wednesday_time_per_lap : ℕ
  thursday_laps : ℕ
  friday_laps : ℕ

/-- Calculate total laps and partial time for a given number of weeks -/
def calculate_totals (routine : SwimmingRoutine) (weeks : ℕ) :
  (ℕ × ℕ) :=
  let weekly_laps := routine.monday_laps + routine.tuesday_laps +
                     routine.wednesday_laps + routine.thursday_laps +
                     routine.friday_laps
  let weekly_partial_time := routine.tuesday_aerobics +
                             (routine.wednesday_laps * routine.wednesday_time_per_lap)
  (weekly_laps * weeks, weekly_partial_time * weeks)

theorem jasmine_swimming_totals :
  let routine := SwimmingRoutine.mk 10 15 20 12 2 18 20
  let (total_laps, partial_time) := calculate_totals routine 5
  total_laps = 375 ∧ partial_time = 220 := by sorry

end NUMINAMATH_CALUDE_jasmine_swimming_totals_l2986_298668


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2986_298616

theorem price_reduction_percentage (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500)
  (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2986_298616


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2986_298634

theorem tenth_term_of_sequence (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h : ∀ k, S k = k^2 + 2*k) : 
  a 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2986_298634


namespace NUMINAMATH_CALUDE_system_solutions_l2986_298653

/-- The system of equations -/
def system (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₃ + x₄ + x₅)^5 = 3*x₁ ∧
  (x₄ + x₅ + x₁)^5 = 3*x₂ ∧
  (x₅ + x₁ + x₂)^5 = 3*x₃ ∧
  (x₁ + x₂ + x₃)^5 = 3*x₄ ∧
  (x₂ + x₃ + x₄)^5 = 3*x₅

/-- The solutions to the system of equations -/
def solutions : Set (ℝ × ℝ × ℝ × ℝ × ℝ) :=
  {(0, 0, 0, 0, 0), (1/3, 1/3, 1/3, 1/3, 1/3), (-1/3, -1/3, -1/3, -1/3, -1/3)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ, system x₁ x₂ x₃ x₄ x₅ ↔ (x₁, x₂, x₃, x₄, x₅) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2986_298653


namespace NUMINAMATH_CALUDE_inequality_proof_l2986_298672

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) 
  (hb : 0 < b ∧ b ≤ 1) 
  (hc : 0 < c ∧ c ≤ 1) 
  (h_sum : a^2 + b^2 + c^2 = 2) : 
  (1 - b^2) / a + (1 - c^2) / b + (1 - a^2) / c ≤ 5/4 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2986_298672


namespace NUMINAMATH_CALUDE_rotated_square_distance_l2986_298647

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the configuration of four squares -/
structure SquareConfiguration where
  squares : Fin 4 → Square
  aligned : Bool
  rotatedSquareIndex : Fin 4
  rotatedSquareTouching : Bool

/-- The distance from the top vertex of the rotated square to the original line -/
def distanceToOriginalLine (config : SquareConfiguration) : ℝ :=
  sorry

theorem rotated_square_distance
  (config : SquareConfiguration)
  (h1 : ∀ i, (config.squares i).sideLength = 2)
  (h2 : config.aligned)
  (h3 : config.rotatedSquareIndex = 1)
  (h4 : config.rotatedSquareTouching) :
  distanceToOriginalLine config = 2 :=
sorry

end NUMINAMATH_CALUDE_rotated_square_distance_l2986_298647


namespace NUMINAMATH_CALUDE_glass_volume_proof_l2986_298625

theorem glass_volume_proof (V : ℝ) 
  (h1 : 0.4 * V = V - 0.6 * V)  -- pessimist's glass is 60% empty (40% full)
  (h2 : 0.6 * V - 0.4 * V = 46) -- difference between optimist's and pessimist's water volumes
  : V = 230 := by
  sorry

end NUMINAMATH_CALUDE_glass_volume_proof_l2986_298625


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l2986_298601

theorem fenced_area_calculation (length width cutout_side : ℝ) 
  (h1 : length = 18.5)
  (h2 : width = 14)
  (h3 : cutout_side = 3.5) : 
  length * width - cutout_side * cutout_side = 246.75 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l2986_298601


namespace NUMINAMATH_CALUDE_haley_weight_l2986_298619

/-- Given the weights of Verna, Haley, and Sherry, prove Haley's weight -/
theorem haley_weight (V H S : ℝ) 
  (verna_haley : V = H + 17)
  (verna_sherry : V = S / 2)
  (total_weight : V + S = 360) :
  H = 103 := by
  sorry

end NUMINAMATH_CALUDE_haley_weight_l2986_298619


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l2986_298685

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2/3, -1/2; 4, -5/2]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-5/6, 1/4; 3/2, -7/4]

theorem matrix_sum_equality : A + B = !![-1/6, -1/4; 11/2, -17/4] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l2986_298685


namespace NUMINAMATH_CALUDE_kylie_apple_picking_l2986_298609

/-- Represents the number of apples Kylie picked in the first hour -/
def first_hour_apples : ℕ := 66

/-- Theorem stating that given the conditions of Kylie's apple picking,
    she picked 66 apples in the first hour -/
theorem kylie_apple_picking :
  ∃ (x : ℕ), 
    x + 2*x + x/3 = 220 ∧ 
    x = first_hour_apples :=
by sorry

end NUMINAMATH_CALUDE_kylie_apple_picking_l2986_298609


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2986_298674

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x > 2/3 ∨ x < -6} := by sorry

-- Theorem for the range of t
theorem range_of_t :
  {t : ℝ | ∀ x, f x ≥ t^2 - (7/2)*t} = {t : ℝ | 3/2 ≤ t ∧ t ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_2_range_of_t_l2986_298674


namespace NUMINAMATH_CALUDE_pond_length_l2986_298644

theorem pond_length (field_length field_width pond_area : ℝ) : 
  field_length = 28 ∧ 
  field_width = 14 ∧ 
  field_length = 2 * field_width ∧ 
  pond_area = (field_length * field_width) / 8 → 
  Real.sqrt pond_area = 7 := by
  sorry

end NUMINAMATH_CALUDE_pond_length_l2986_298644


namespace NUMINAMATH_CALUDE_total_car_parts_cost_l2986_298603

/-- The amount Mike spent on speakers -/
def speakers_cost : ℚ := 118.54

/-- The amount Mike spent on tires -/
def tires_cost : ℚ := 106.33

/-- The total amount Mike spent on car parts -/
def total_cost : ℚ := speakers_cost + tires_cost

/-- Theorem stating that the total cost of car parts is $224.87 -/
theorem total_car_parts_cost : total_cost = 224.87 := by sorry

end NUMINAMATH_CALUDE_total_car_parts_cost_l2986_298603


namespace NUMINAMATH_CALUDE_leftover_milk_proof_l2986_298652

/-- Represents the amount of milk used for each type of milkshake -/
structure MilkUsage where
  vanilla : ℕ
  chocolate : ℕ

/-- Represents the amount of ice cream used for each type of milkshake -/
structure IceCreamUsage where
  vanilla : ℕ
  chocolate : ℕ

/-- Represents the available ingredients -/
structure Ingredients where
  milk : ℕ
  vanilla_ice_cream : ℕ
  chocolate_ice_cream : ℕ

/-- Represents the number of milkshakes to make -/
structure Milkshakes where
  vanilla : ℕ
  chocolate : ℕ

def milk_usage : MilkUsage := ⟨4, 5⟩
def ice_cream_usage : IceCreamUsage := ⟨12, 10⟩
def available_ingredients : Ingredients := ⟨72, 96, 96⟩
def max_milkshakes : ℕ := 16

def valid_milkshake_count (m : Milkshakes) : Prop :=
  m.vanilla + m.chocolate ≤ max_milkshakes ∧
  2 * m.chocolate = m.vanilla

def enough_ingredients (m : Milkshakes) : Prop :=
  m.vanilla * milk_usage.vanilla + m.chocolate * milk_usage.chocolate ≤ available_ingredients.milk ∧
  m.vanilla * ice_cream_usage.vanilla ≤ available_ingredients.vanilla_ice_cream ∧
  m.chocolate * ice_cream_usage.chocolate ≤ available_ingredients.chocolate_ice_cream

def optimal_milkshakes : Milkshakes := ⟨10, 5⟩

theorem leftover_milk_proof :
  valid_milkshake_count optimal_milkshakes ∧
  enough_ingredients optimal_milkshakes ∧
  ∀ m : Milkshakes, valid_milkshake_count m → enough_ingredients m →
    m.vanilla + m.chocolate ≤ optimal_milkshakes.vanilla + optimal_milkshakes.chocolate →
  available_ingredients.milk - (optimal_milkshakes.vanilla * milk_usage.vanilla + optimal_milkshakes.chocolate * milk_usage.chocolate) = 7 :=
sorry

end NUMINAMATH_CALUDE_leftover_milk_proof_l2986_298652


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2986_298699

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_difference
  (a b : ℕ → ℝ)
  (ha : ArithmeticSequence a)
  (hb : ArithmeticSequence b)
  (ha1 : a 1 = 3)
  (hb1 : b 1 = -3)
  (h19 : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2986_298699


namespace NUMINAMATH_CALUDE_tv_sales_effect_l2986_298627

theorem tv_sales_effect (P Q : ℝ) (h1 : P > 0) (h2 : Q > 0) : 
  let new_price := 0.82 * P
  let new_quantity := 1.72 * Q
  let original_value := P * Q
  let new_value := new_price * new_quantity
  (new_value / original_value - 1) * 100 = 41.04 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_effect_l2986_298627


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2986_298629

theorem absolute_value_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 5 ↔ ((-2 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2986_298629


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2986_298620

theorem arithmetic_calculation : 3 + (12 / 3 - 1)^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2986_298620


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2986_298631

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: Given an arithmetic sequence with S_15 = 30 and a_7 = 1, then S_9 = -9 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h1 : seq.S 15 = 30)
  (h2 : seq.a 7 = 1) :
  seq.S 9 = -9 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2986_298631


namespace NUMINAMATH_CALUDE_blue_yellow_probability_l2986_298675

/-- The probability of drawing a blue chip first and then a yellow chip without replacement -/
def draw_blue_then_yellow (blue : ℕ) (yellow : ℕ) : ℚ :=
  (blue : ℚ) / (blue + yellow) * yellow / (blue + yellow - 1)

/-- Theorem stating the probability of drawing a blue chip first and then a yellow chip
    without replacement from a bag containing 10 blue chips and 5 yellow chips -/
theorem blue_yellow_probability :
  draw_blue_then_yellow 10 5 = 5 / 21 := by
  sorry

#eval draw_blue_then_yellow 10 5

end NUMINAMATH_CALUDE_blue_yellow_probability_l2986_298675


namespace NUMINAMATH_CALUDE_three_digit_sum_property_l2986_298626

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

def is_triple_digit (n : ℕ) : Prop :=
  ∃ a : ℕ, 0 < a ∧ a < 10 ∧ n = 100 * a + 10 * a + a

theorem three_digit_sum_property :
  {n : ℕ | is_three_digit n ∧ is_triple_digit (n + sum_of_digits n)} =
  {105, 324, 429, 543, 648, 762, 867, 981} := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_property_l2986_298626


namespace NUMINAMATH_CALUDE_contrapositive_example_l2986_298646

theorem contrapositive_example :
  (∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∀ x : ℝ, x^2 ≤ 1 → x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_example_l2986_298646


namespace NUMINAMATH_CALUDE_team_transfer_equation_l2986_298670

theorem team_transfer_equation (x : ℤ) : 
  let team_a_initial : ℤ := 37
  let team_b_initial : ℤ := 23
  let team_a_final : ℤ := team_a_initial + x
  let team_b_final : ℤ := team_b_initial - x
  team_a_final = 2 * team_b_final →
  37 + x = 2 * (23 - x) :=
by
  sorry

end NUMINAMATH_CALUDE_team_transfer_equation_l2986_298670


namespace NUMINAMATH_CALUDE_min_framing_feet_l2986_298604

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered photo -/
theorem min_framing_feet (original_width original_height border_width : ℕ) : 
  original_width = 5 →
  original_height = 7 →
  border_width = 3 →
  (((2 * original_width + 2 * border_width) + 
    (2 * original_height + 2 * border_width)) : ℕ) / 12 + 
    (if ((2 * original_width + 2 * border_width) + 
         (2 * original_height + 2 * border_width)) % 12 = 0 then 0 else 1) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_framing_feet_l2986_298604


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2986_298637

theorem fixed_point_of_exponential_function (a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 3) + 3
  f 3 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2986_298637


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2986_298608

/-- The coordinates of the foci of the hyperbola x²/3 - y² = 1 are (-2, 0) and (2, 0) -/
theorem hyperbola_foci_coordinates :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 3 - y^2 = 1
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁ = (-2, 0) ∧ f₂ = (2, 0)) ∧
    (∀ x y, h x y ↔ (x - f₁.1)^2 / (f₂.1 - f₁.1)^2 - (y - f₁.2)^2 / ((f₂.1 - f₁.1)^2 / 3 - (f₂.2 - f₁.2)^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l2986_298608


namespace NUMINAMATH_CALUDE_reciprocal_not_always_plus_minus_one_l2986_298687

theorem reciprocal_not_always_plus_minus_one : 
  ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_not_always_plus_minus_one_l2986_298687


namespace NUMINAMATH_CALUDE_garbage_classification_repost_l2986_298624

theorem garbage_classification_repost (n : ℕ) : 
  (1 + n + n^2 = 111) ↔ (n = 10) :=
sorry

end NUMINAMATH_CALUDE_garbage_classification_repost_l2986_298624


namespace NUMINAMATH_CALUDE_f_sum_negative_l2986_298630

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_odd : ∀ x, f x + f (-x) = 0
axiom f_increasing_neg : ∀ x y, x < y → y ≤ 0 → f x < f y

-- Define the theorem
theorem f_sum_negative (x₁ x₂ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₁ * x₂ < 0) : 
  f x₁ + f x₂ < 0 := by sorry

end NUMINAMATH_CALUDE_f_sum_negative_l2986_298630


namespace NUMINAMATH_CALUDE_circle_radius_from_triangle_l2986_298679

/-- Given a right-angled triangle with area 60 cm² and one side 15 cm that touches a circle,
    prove that the radius of the circle is 20 cm. -/
theorem circle_radius_from_triangle (triangle_area : ℝ) (triangle_side : ℝ) (circle_radius : ℝ) :
  triangle_area = 60 →
  triangle_side = 15 →
  -- Additional properties to define the relationship between the triangle and circle
  -- These are simplified representations of the problem conditions
  ∃ (triangle_height : ℝ) (triangle_hypotenuse : ℝ),
    triangle_area = (1/2) * triangle_side * triangle_height ∧
    triangle_hypotenuse^2 = triangle_side^2 + triangle_height^2 ∧
    circle_radius - triangle_height + circle_radius - triangle_side = triangle_hypotenuse →
  circle_radius = 20 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_triangle_l2986_298679


namespace NUMINAMATH_CALUDE_fourth_quadrant_condition_l2986_298690

theorem fourth_quadrant_condition (m : ℝ) :
  let z := (m + Complex.I) / (1 + Complex.I)
  (z.re > 0 ∧ z.im < 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_quadrant_condition_l2986_298690


namespace NUMINAMATH_CALUDE_scissors_count_l2986_298636

/-- The total number of scissors after adding more -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of scissors is 76 -/
theorem scissors_count : total_scissors 54 22 = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l2986_298636


namespace NUMINAMATH_CALUDE_factor_theorem_l2986_298695

theorem factor_theorem (p q : ℝ) : 
  (∀ x : ℝ, (x - 3) * (x + 5) = x^2 + p*x + q) → p = 2 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_l2986_298695


namespace NUMINAMATH_CALUDE_honey_distribution_l2986_298691

/-- Represents the volume of honey in a barrel -/
structure HoneyVolume where
  volume : ℚ
  positive : volume > 0

/-- The volume of honey in a large barrel -/
def large_barrel : HoneyVolume :=
  { volume := 1, positive := by norm_num }

/-- The volume of honey in a small barrel -/
def small_barrel : HoneyVolume :=
  { volume := 5/9, positive := by norm_num }

/-- The total volume of honey in Winnie-the-Pooh's possession -/
def total_honey : ℚ := 25 * large_barrel.volume

theorem honey_distribution (h : 25 * large_barrel.volume = 45 * small_barrel.volume) :
  total_honey = 20 * large_barrel.volume + 9 * small_barrel.volume :=
by sorry

end NUMINAMATH_CALUDE_honey_distribution_l2986_298691


namespace NUMINAMATH_CALUDE_chocolate_mixture_proof_l2986_298682

theorem chocolate_mixture_proof (initial_weight : ℝ) (initial_percentage : ℝ) 
  (final_weight : ℝ) (final_percentage : ℝ) (added_pure_chocolate : ℝ) : 
  initial_weight = 620 →
  initial_percentage = 0.1 →
  final_weight = 1000 →
  final_percentage = 0.7 →
  added_pure_chocolate = 638 →
  (initial_weight * initial_percentage + added_pure_chocolate) / final_weight = final_percentage :=
by
  sorry

#check chocolate_mixture_proof

end NUMINAMATH_CALUDE_chocolate_mixture_proof_l2986_298682


namespace NUMINAMATH_CALUDE_chord_length_when_a_is_3_2_symmetrical_circle_equation_l2986_298697

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 4*a*y + 4*a^2 + 1 = 0

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop :=
  a*x + y + 2*a = 0

-- Part 1: Length of chord AB when a = 3/2
theorem chord_length_when_a_is_3_2 :
  ∃ (A B : ℝ × ℝ),
    circle_C (3/2) A.1 A.2 ∧
    circle_C (3/2) B.1 B.2 ∧
    line_l (3/2) A.1 A.2 ∧
    line_l (3/2) B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (2*Real.sqrt 39 / 13)^2 :=
sorry

-- Part 2: Equation of symmetrical circle C' when line l is tangent to circle C
theorem symmetrical_circle_equation (a : ℝ) :
  a > 0 →
  (∃ (x₀ y₀ : ℝ), circle_C a x₀ y₀ ∧ line_l a x₀ y₀ ∧
    ∀ (x y : ℝ), circle_C a x y → line_l a x y → x = x₀ ∧ y = y₀) →
  ∃ (x₁ y₁ : ℝ),
    x₁ = -5 ∧ y₁ = Real.sqrt 3 ∧
    ∀ (x y : ℝ), (x - x₁)^2 + (y - y₁)^2 = 3 ↔
      circle_C a (2*x₁ - x) (2*y₁ - y) :=
sorry

end NUMINAMATH_CALUDE_chord_length_when_a_is_3_2_symmetrical_circle_equation_l2986_298697


namespace NUMINAMATH_CALUDE_gmat_test_results_l2986_298661

theorem gmat_test_results (first_correct : ℝ) (second_correct : ℝ) (neither_correct : ℝ)
  (h1 : first_correct = 85)
  (h2 : second_correct = 80)
  (h3 : neither_correct = 5)
  : first_correct + second_correct - (100 - neither_correct) = 70 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_results_l2986_298661


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l2986_298689

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with equation y² = -8x -/
def Parabola := Unit

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a hyperbola and a parabola satisfying certain conditions, 
    the eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two 
  (h : Hyperbola) 
  (p : Parabola) 
  (A B O : Point)
  (h_asymptotes : A.x = 2 ∧ B.x = 2)  -- Asymptotes intersect directrix x = 2
  (h_origin : O.x = 0 ∧ O.y = 0)      -- O is the origin
  (h_area : abs ((A.x - O.x) * (B.y - O.y) - (B.x - O.x) * (A.y - O.y)) / 2 = 4 * Real.sqrt 3)
  : h.a / Real.sqrt (h.a^2 - h.b^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_is_two_l2986_298689


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l2986_298612

theorem arithmetic_mean_difference (p q r : ℝ) (G : ℝ) : 
  G = (p * q * r) ^ (1/3) →
  (p + q) / 2 = 10 →
  (q + r) / 2 = 25 →
  r - p = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l2986_298612


namespace NUMINAMATH_CALUDE_existence_of_n_l2986_298650

theorem existence_of_n (p : ℕ) (a k : ℕ+) (h_prime : Nat.Prime p) 
  (h_bound : p^(a : ℕ) < k ∧ k < 2 * p^(a : ℕ)) : 
  ∃ n : ℕ+, n < p^(2 * (a : ℕ)) ∧ 
    (Nat.choose n k : ZMod (p^(a : ℕ))) = n ∧ 
    (n : ZMod (p^(a : ℕ))) = k :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_l2986_298650


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_congruence_l2986_298659

theorem smallest_four_digit_number_congruence (x : ℕ) : 
  (x ≥ 1000 ∧ x < 10000) →
  (3 * x ≡ 9 [ZMOD 18]) →
  (5 * x + 20 ≡ 30 [ZMOD 15]) →
  (3 * x - 4 ≡ 2 * x [ZMOD 35]) →
  x ≥ 1004 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_congruence_l2986_298659


namespace NUMINAMATH_CALUDE_quadratic_fixed_point_l2986_298623

/-- The quadratic function y = -x² + (m-1)x + m has a fixed point at (-1, 0) for all m -/
theorem quadratic_fixed_point :
  ∀ (m : ℝ), -(-1)^2 + (m - 1)*(-1) + m = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_fixed_point_l2986_298623


namespace NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2986_298648

theorem repeating_decimal_fraction_sum : ∃ (n d : ℕ), 
  (n.gcd d = 1) ∧ 
  (n : ℚ) / d = 45 / 99 ∧ 
  n + d = 16 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_fraction_sum_l2986_298648


namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l2986_298692

theorem children_on_bus_after_stop (initial_children : ℕ) (children_off : ℕ) (extra_children_on : ℕ) : 
  initial_children = 5 →
  children_off = 63 →
  extra_children_on = 9 →
  (initial_children - children_off + (children_off + extra_children_on) : ℤ) = 14 :=
by sorry

end NUMINAMATH_CALUDE_children_on_bus_after_stop_l2986_298692


namespace NUMINAMATH_CALUDE_inverse_of_A_l2986_298669

def A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; -2, 9]

theorem inverse_of_A :
  A⁻¹ = !![9/35, -4/35; 2/35, 3/35] := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l2986_298669


namespace NUMINAMATH_CALUDE_root_exists_and_bisection_applicable_l2986_298655

-- Define the function f(x) = x^2 - 2x - 3
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the interval (-2, 2)
def interval : Set ℝ := Set.Ioo (-2) 2

-- Theorem statement
theorem root_exists_and_bisection_applicable :
  ∃ (x : ℝ), x ∈ interval ∧ f x = 0 ∧
  (∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧ f a * f b ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_root_exists_and_bisection_applicable_l2986_298655


namespace NUMINAMATH_CALUDE_carol_weight_l2986_298693

/-- Given that the sum of Alice's and Carol's weights is 220 pounds, and the difference
    between Carol's and Alice's weights is one-third of Carol's weight plus 10 pounds,
    prove that Carol weighs 138 pounds. -/
theorem carol_weight (alice_weight carol_weight : ℝ) 
  (h1 : alice_weight + carol_weight = 220)
  (h2 : carol_weight - alice_weight = (1/3) * carol_weight + 10) : 
  carol_weight = 138 := by
  sorry

end NUMINAMATH_CALUDE_carol_weight_l2986_298693


namespace NUMINAMATH_CALUDE_student_grade_proof_l2986_298621

def courses_last_year : ℕ := 6
def avg_grade_last_year : ℝ := 100
def courses_year_before : ℕ := 5
def avg_grade_two_years : ℝ := 77
def total_courses : ℕ := courses_last_year + courses_year_before

theorem student_grade_proof :
  ∃ (avg_grade_year_before : ℝ),
    avg_grade_year_before * courses_year_before + avg_grade_last_year * courses_last_year =
    avg_grade_two_years * total_courses ∧
    avg_grade_year_before = 49.4 := by
  sorry

end NUMINAMATH_CALUDE_student_grade_proof_l2986_298621


namespace NUMINAMATH_CALUDE_trip_distance_l2986_298645

theorem trip_distance (total_time hiking_speed canoe_speed hiking_distance : ℝ) 
  (h1 : total_time = 5.5)
  (h2 : hiking_speed = 5)
  (h3 : canoe_speed = 12)
  (h4 : hiking_distance = 27) :
  hiking_distance + (total_time - hiking_distance / hiking_speed) * canoe_speed = 28.2 := by
  sorry

#check trip_distance

end NUMINAMATH_CALUDE_trip_distance_l2986_298645


namespace NUMINAMATH_CALUDE_distance_AB_equals_5_l2986_298660

-- Define the line l₁
def line_l₁ (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Define the curve C
def curve_C (x y φ : ℝ) : Prop :=
  x = 1 + Real.sqrt 3 * Real.cos φ ∧
  y = Real.sqrt 3 * Real.sin φ ∧
  0 ≤ φ ∧ φ ≤ Real.pi

-- Define the line l₂ in polar coordinates
def line_l₂ (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin (θ + Real.pi / 3) + 3 * Real.sqrt 3 = 0

-- Define the intersection point A of l₁ and C
def point_A : ℝ × ℝ := sorry

-- Define the intersection point B of l₁ and l₂
def point_B : ℝ × ℝ := sorry

-- Theorem statement
theorem distance_AB_equals_5 :
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_AB_equals_5_l2986_298660


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2986_298610

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y z : ℝ) :
  S.card = 60 →
  x = 48 ∧ y = 58 ∧ z = 52 →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - (x + y + z)) / (S.card - 3) = 41.4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2986_298610


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2986_298683

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) (h2 : x * y = 16) : x^2 + y^2 = 432 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l2986_298683


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l2986_298678

theorem consecutive_odd_numbers_equation (n : ℕ) : 
  let first := 7
  let second := first + 2
  let third := second + 2
  8 * first = 3 * third + 2 * second + 5 :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l2986_298678


namespace NUMINAMATH_CALUDE_tan_seven_pi_fourth_l2986_298605

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_seven_pi_fourth_l2986_298605


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_two_l2986_298643

/-- The curve function -/
def f (x : ℝ) : ℝ := x^2 + 3*x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 2*x + 3

theorem tangent_slope_at_point_two :
  f' 2 = 7 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_two_l2986_298643


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2986_298686

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 2 + a 9 = 11) →
  (a 4 + a 10 = 14) →
  (a 6 + a 11 = 17) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2986_298686


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2986_298615

theorem sum_of_fractions : (3 : ℚ) / 462 + 17 / 42 + 1 / 11 = 116 / 231 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2986_298615


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2986_298658

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x > 1}
def N : Set ℝ := {x | ∃ y, y = Real.log (3/2 - x)}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2986_298658
