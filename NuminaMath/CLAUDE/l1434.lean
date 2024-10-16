import Mathlib

namespace NUMINAMATH_CALUDE_efficient_coefficient_computation_l1434_143443

/-- Represents a method to compute polynomial coefficients -/
structure ComputationMethod where
  (compute : (ℝ → ℝ) → List ℝ)
  (addition_count : ℕ)
  (multiplication_count : ℕ)

/-- A 6th degree polynomial -/
def Polynomial6 (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : ℝ → ℝ :=
  fun x ↦ x^6 + a₁*x^5 + a₂*x^4 + a₃*x^3 + a₄*x^2 + a₅*x + a₆

/-- Theorem: There exists a method to compute coefficients of a 6th degree polynomial
    using its roots with no more than 15 additions and 15 multiplications -/
theorem efficient_coefficient_computation :
  ∃ (method : ComputationMethod),
    (∀ (p : ℝ → ℝ) (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
      (∀ x, p x = (x + r₁) * (x + r₂) * (x + r₃) * (x + r₄) * (x + r₅) * (x + r₆)) →
      ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ),
        p = Polynomial6 a₁ a₂ a₃ a₄ a₅ a₆ ∧
        method.compute p = [a₁, a₂, a₃, a₄, a₅, a₆]) ∧
    method.addition_count ≤ 15 ∧
    method.multiplication_count ≤ 15 :=
by sorry


end NUMINAMATH_CALUDE_efficient_coefficient_computation_l1434_143443


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1434_143403

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The number of bananas that are worth as much as 12 oranges -/
def bananas_worth_12_oranges : ℚ := 12

/-- The fraction of 16 bananas that are worth as much as 12 oranges -/
def fraction_of_16_bananas : ℚ := 3/4

/-- The number of bananas we're considering in the question -/
def question_bananas : ℚ := 9

/-- The fraction of question_bananas we're considering -/
def fraction_of_question_bananas : ℚ := 2/3

theorem banana_orange_equivalence :
  fraction_of_16_bananas * 16 = bananas_worth_12_oranges →
  fraction_of_question_bananas * question_bananas * banana_value = 6 := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1434_143403


namespace NUMINAMATH_CALUDE_one_fourth_of_7_2_l1434_143457

theorem one_fourth_of_7_2 : 
  (7.2 : ℚ) / 4 = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_one_fourth_of_7_2_l1434_143457


namespace NUMINAMATH_CALUDE_f_domain_correct_l1434_143455

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 * x^2) / Real.sqrt (1 - x) + Real.log (3 * x + 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | -1/3 < x ∧ x < 1}

-- Theorem stating that domain_f is the correct domain for f
theorem f_domain_correct : 
  ∀ x : ℝ, x ∈ domain_f ↔ (∃ y : ℝ, f x = y) :=
sorry

end NUMINAMATH_CALUDE_f_domain_correct_l1434_143455


namespace NUMINAMATH_CALUDE_remaining_land_to_clean_l1434_143453

theorem remaining_land_to_clean (total_land area_lizzie area_other : ℕ) :
  total_land = 900 ∧ area_lizzie = 250 ∧ area_other = 265 →
  total_land - (area_lizzie + area_other) = 385 := by
  sorry

end NUMINAMATH_CALUDE_remaining_land_to_clean_l1434_143453


namespace NUMINAMATH_CALUDE_total_dogs_l1434_143439

/-- Represents the properties of dogs in a kennel -/
structure Kennel where
  longFurred : Nat
  brown : Nat
  longFurredBrown : Nat
  neitherLongFurredNorBrown : Nat

/-- Theorem stating the total number of dogs in the kennel -/
theorem total_dogs (k : Kennel) 
  (h1 : k.longFurred = 26)
  (h2 : k.brown = 22)
  (h3 : k.longFurredBrown = 11)
  (h4 : k.neitherLongFurredNorBrown = 8) :
  k.longFurred + k.brown - k.longFurredBrown + k.neitherLongFurredNorBrown = 45 := by
  sorry

#check total_dogs

end NUMINAMATH_CALUDE_total_dogs_l1434_143439


namespace NUMINAMATH_CALUDE_collinearity_condition_l1434_143436

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points A₁, B₁, C₁
variables (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the R value as in Problem 191
def R (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a point is on a side of the triangle
def onSide (ABC : Triangle) (P : ℝ × ℝ) : Bool := sorry

-- Define a function to count how many points are on the sides of the triangle
def countOnSides (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) : Nat := sorry

-- Define collinearity
def collinear (A₁ B₁ C₁ : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem collinearity_condition (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) :
  collinear A₁ B₁ C₁ ↔ R ABC A₁ B₁ C₁ = 1 ∧ Even (countOnSides ABC A₁ B₁ C₁) :=
sorry

end NUMINAMATH_CALUDE_collinearity_condition_l1434_143436


namespace NUMINAMATH_CALUDE_water_height_in_cone_l1434_143494

theorem water_height_in_cone (r h : ℝ) (water_ratio : ℝ) :
  r = 16 →
  h = 96 →
  water_ratio = 1/4 →
  (water_ratio * (1/3 * π * r^2 * h) = 1/3 * π * r^2 * (48 * Real.rpow 2 (1/3))) :=
by sorry

end NUMINAMATH_CALUDE_water_height_in_cone_l1434_143494


namespace NUMINAMATH_CALUDE_age_difference_proof_l1434_143407

/-- Proves that the difference between twice John's current age and Tim's age is 15 years -/
theorem age_difference_proof (james_age_past : ℕ) (john_age_past : ℕ) (tim_age : ℕ) 
  (h1 : james_age_past = 23)
  (h2 : john_age_past = 35)
  (h3 : tim_age = 79)
  (h4 : ∃ (x : ℕ), tim_age + x = 2 * (john_age_past + (john_age_past - james_age_past))) :
  2 * (john_age_past + (john_age_past - james_age_past)) - tim_age = 15 := by
  sorry


end NUMINAMATH_CALUDE_age_difference_proof_l1434_143407


namespace NUMINAMATH_CALUDE_taehyung_has_most_points_l1434_143476

def yoongi_points : ℕ := 7
def jungkook_points : ℕ := 6
def yuna_points : ℕ := 9
def yoojung_points : ℕ := 8
def taehyung_points : ℕ := 10

theorem taehyung_has_most_points :
  taehyung_points ≥ yoongi_points ∧
  taehyung_points ≥ jungkook_points ∧
  taehyung_points ≥ yuna_points ∧
  taehyung_points ≥ yoojung_points :=
by sorry

end NUMINAMATH_CALUDE_taehyung_has_most_points_l1434_143476


namespace NUMINAMATH_CALUDE_chemistry_marks_proof_l1434_143456

def english_marks : ℕ := 76
def math_marks : ℕ := 60
def physics_marks : ℕ := 72
def biology_marks : ℕ := 82
def average_marks : ℕ := 71
def total_subjects : ℕ := 5

theorem chemistry_marks_proof :
  ∃ (chemistry_marks : ℕ),
    chemistry_marks = average_marks * total_subjects - (english_marks + math_marks + physics_marks + biology_marks) ∧
    chemistry_marks = 65 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_marks_proof_l1434_143456


namespace NUMINAMATH_CALUDE_billy_ate_nine_apples_on_wednesday_l1434_143478

/-- The number of apples Billy ate each day of the week --/
structure WeeklyApples where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  total : ℕ

/-- Billy's apple consumption for the week satisfies the given conditions --/
def satisfiesConditions (w : WeeklyApples) : Prop :=
  w.monday = 2 ∧
  w.tuesday = 2 * w.monday ∧
  w.thursday = 4 * w.friday ∧
  w.friday = w.monday / 2 ∧
  w.total = 20 ∧
  w.total = w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- The theorem stating that Billy ate 9 apples on Wednesday --/
theorem billy_ate_nine_apples_on_wednesday (w : WeeklyApples) 
  (h : satisfiesConditions w) : w.wednesday = 9 := by
  sorry

end NUMINAMATH_CALUDE_billy_ate_nine_apples_on_wednesday_l1434_143478


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l1434_143493

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop :=
  ∃ k > 0, ∀ (s1 s2 : ℝ × ℝ), s1 ∈ t1 → s2 ∈ t2 → ‖s1.1 - s1.2‖ = k * ‖s2.1 - s2.2‖

theorem similar_triangles_side_length 
  (PQR STU : Set (ℝ × ℝ))
  (h_similar : similar_triangles PQR STU)
  (h_PQ : ∃ PQ ∈ PQR, ‖PQ.1 - PQ.2‖ = 7)
  (h_PR : ∃ PR ∈ PQR, ‖PR.1 - PR.2‖ = 9)
  (h_ST : ∃ ST ∈ STU, ‖ST.1 - ST.2‖ = 4.2)
  : ∃ SU ∈ STU, ‖SU.1 - SU.2‖ = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l1434_143493


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l1434_143474

theorem factorial_fraction_equality : (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l1434_143474


namespace NUMINAMATH_CALUDE_odd_quadruple_composition_l1434_143424

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem odd_quadruple_composition (g : ℝ → ℝ) (h : IsOdd g) :
  IsOdd (fun x ↦ g (g (g (g x)))) := by
  sorry

end NUMINAMATH_CALUDE_odd_quadruple_composition_l1434_143424


namespace NUMINAMATH_CALUDE_last_two_digits_sum_l1434_143465

theorem last_two_digits_sum (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_l1434_143465


namespace NUMINAMATH_CALUDE_sum_even_positive_lt_100_l1434_143402

/-- The sum of all even, positive integers less than 100 is 2450 -/
theorem sum_even_positive_lt_100 : 
  (Finset.filter (fun n => n % 2 = 0 ∧ 0 < n) (Finset.range 100)).sum id = 2450 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_positive_lt_100_l1434_143402


namespace NUMINAMATH_CALUDE_no_real_solutions_l1434_143438

theorem no_real_solutions : ∀ x y : ℝ, x^2 + 3*y^2 - 4*x - 12*y + 36 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1434_143438


namespace NUMINAMATH_CALUDE_matthew_crackers_l1434_143437

theorem matthew_crackers (initial : ℕ) (friends : ℕ) (given_each : ℕ) (left : ℕ) : 
  friends = 3 → given_each = 7 → left = 17 → 
  initial = friends * given_each + left → initial = 38 :=
by sorry

end NUMINAMATH_CALUDE_matthew_crackers_l1434_143437


namespace NUMINAMATH_CALUDE_power_multiplication_l1434_143419

theorem power_multiplication (x : ℝ) : x^2 * x^4 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1434_143419


namespace NUMINAMATH_CALUDE_days_to_clear_land_l1434_143444

/-- Represents the number of feet in a yard -/
def feet_per_yard : ℝ := 3

/-- Represents the length of the land in feet -/
def land_length_feet : ℝ := 900

/-- Represents the width of the land in feet -/
def land_width_feet : ℝ := 200

/-- Represents the number of rabbits -/
def num_rabbits : ℕ := 100

/-- Represents the area one rabbit can clear per day in square yards -/
def area_per_rabbit_per_day : ℝ := 10

/-- Theorem stating the number of days needed to clear the land -/
theorem days_to_clear_land : 
  ⌈(land_length_feet / feet_per_yard) * (land_width_feet / feet_per_yard) / 
   (num_rabbits : ℝ) / area_per_rabbit_per_day⌉ = 21 := by sorry

end NUMINAMATH_CALUDE_days_to_clear_land_l1434_143444


namespace NUMINAMATH_CALUDE_symmetry_point_l1434_143491

/-- A point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- A line in the form y = mx + b -/
structure Line where
  m : ℚ
  b : ℚ

/-- Check if two points are symmetric with respect to a line -/
def areSymmetric (P Q : Point) (l : Line) : Prop :=
  -- The product of the slopes of PQ and l is -1
  ((Q.y - P.y) / (Q.x - P.x)) * l.m = -1 ∧
  -- The midpoint of PQ lies on l
  ((Q.y + P.y) / 2) = l.m * ((Q.x + P.x) / 2) + l.b

theorem symmetry_point :
  let P : Point := ⟨-1, 2⟩
  let Q : Point := ⟨7/5, 4/5⟩
  let l : Line := ⟨2, 1⟩  -- y = 2x + 1
  areSymmetric P Q l := by sorry

end NUMINAMATH_CALUDE_symmetry_point_l1434_143491


namespace NUMINAMATH_CALUDE_sin_eighth_integral_l1434_143454

theorem sin_eighth_integral : ∫ x in (0)..(2*Real.pi), (Real.sin x)^8 = (35 * Real.pi) / 64 := by sorry

end NUMINAMATH_CALUDE_sin_eighth_integral_l1434_143454


namespace NUMINAMATH_CALUDE_nancy_files_distribution_l1434_143413

/-- Given the initial number of files, number of deleted files, and number of folders,
    calculate the number of files in each folder after distribution. -/
def filesPerFolder (initialFiles deletedFiles numFolders : ℕ) : ℕ :=
  (initialFiles - deletedFiles) / numFolders

/-- Prove that given 80 initial files, after deleting 31 files and distributing
    the remaining files equally into 7 folders, each folder contains 7 files. -/
theorem nancy_files_distribution :
  filesPerFolder 80 31 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_nancy_files_distribution_l1434_143413


namespace NUMINAMATH_CALUDE_number_of_pairs_l1434_143400

theorem number_of_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pairs_l1434_143400


namespace NUMINAMATH_CALUDE_sally_balloons_l1434_143498

/-- Given the number of blue balloons for Alyssa, Sandy, and the total,
    prove that Sally has the correct number of blue balloons. -/
theorem sally_balloons (alyssa_balloons sandy_balloons total_balloons : ℕ)
  (h1 : alyssa_balloons = 37)
  (h2 : sandy_balloons = 28)
  (h3 : total_balloons = 104) :
  total_balloons - (alyssa_balloons + sandy_balloons) = 39 := by
  sorry

#check sally_balloons

end NUMINAMATH_CALUDE_sally_balloons_l1434_143498


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l1434_143482

/-- Given three square regions I, II, and III, where the perimeter of region I is 12 units,
    the perimeter of region II is 24 units, and the side length of region III is the sum of
    the side lengths of regions I and II, prove that the ratio of the area of region I to
    the area of region III is 1/9. -/
theorem area_ratio_of_squares (side_length_I side_length_II side_length_III : ℝ) :
  side_length_I * 4 = 12 →
  side_length_II * 4 = 24 →
  side_length_III = side_length_I + side_length_II →
  (side_length_I ^ 2) / (side_length_III ^ 2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l1434_143482


namespace NUMINAMATH_CALUDE_line_length_difference_l1434_143415

theorem line_length_difference : 
  let white_line : ℝ := 7.67
  let blue_line : ℝ := 3.33
  white_line - blue_line = 4.34 := by
sorry

end NUMINAMATH_CALUDE_line_length_difference_l1434_143415


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1434_143467

/-- Represents the number of households in each category -/
structure HouseholdCounts where
  farmers : ℕ
  workers : ℕ
  intellectuals : ℕ

/-- Represents the sample sizes -/
structure SampleSizes where
  farmers : ℕ
  total : ℕ

/-- Theorem stating the relationship between the household counts, 
    sample sizes, and the expected total sample size -/
theorem stratified_sampling_theorem 
  (counts : HouseholdCounts) 
  (sample : SampleSizes) : 
  counts.farmers = 1500 →
  counts.workers = 401 →
  counts.intellectuals = 99 →
  sample.farmers = 75 →
  sample.total = 100 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1434_143467


namespace NUMINAMATH_CALUDE_cube_root_of_square_l1434_143401

theorem cube_root_of_square (x : ℝ) : (x^2)^(1/3) = x^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_square_l1434_143401


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l1434_143451

-- Define M as a positive integer
def M : ℕ+ := sorry

-- Define the condition that M^2 = 36^50 * 50^36
axiom M_squared : (M : ℕ).pow 2 = (36 : ℕ).pow 50 * (50 : ℕ).pow 36

-- Define a function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_M : sum_of_digits (M : ℕ) = 344 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l1434_143451


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l1434_143472

def g (x : ℝ) := -3 * x^3 + 5 * x^2 + 4

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
by sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l1434_143472


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1434_143483

theorem perfect_square_binomial :
  ∃ a : ℝ, ∀ x : ℝ, x^2 + 120*x + 3600 = (x + a)^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1434_143483


namespace NUMINAMATH_CALUDE_max_rods_in_box_l1434_143469

/-- A rod with dimensions 1×1×4 -/
structure Rod :=
  (length : ℕ := 4)
  (width : ℕ := 1)
  (height : ℕ := 1)

/-- A cube-shaped box with dimensions 6×6×6 -/
structure Box :=
  (length : ℕ := 6)
  (width : ℕ := 6)
  (height : ℕ := 6)

/-- Predicate to check if a rod can be placed parallel to the box faces -/
def isParallel (r : Rod) (b : Box) : Prop :=
  (r.length ≤ b.length ∧ r.width ≤ b.width ∧ r.height ≤ b.height) ∨
  (r.length ≤ b.width ∧ r.width ≤ b.height ∧ r.height ≤ b.length) ∨
  (r.length ≤ b.height ∧ r.width ≤ b.length ∧ r.height ≤ b.width)

/-- The maximum number of rods that can fit in the box -/
def maxRods (r : Rod) (b : Box) : ℕ := 52

/-- Theorem stating that 52 is the maximum number of 1×1×4 rods that can fit in a 6×6×6 box -/
theorem max_rods_in_box (r : Rod) (b : Box) :
  isParallel r b → maxRods r b = 52 ∧ ¬∃ n : ℕ, n > 52 ∧ n * r.length * r.width * r.height ≤ b.length * b.width * b.height :=
sorry


end NUMINAMATH_CALUDE_max_rods_in_box_l1434_143469


namespace NUMINAMATH_CALUDE_necessary_implies_sufficient_l1434_143489

theorem necessary_implies_sufficient (A B : Prop) :
  (A → B) → (A → B) :=
by
  sorry

end NUMINAMATH_CALUDE_necessary_implies_sufficient_l1434_143489


namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1434_143466

/-- Calculates the number of students in both band and chorus -/
def students_in_both (total : ℕ) (band : ℕ) (chorus : ℕ) (band_or_chorus : ℕ) : ℕ :=
  band + chorus - band_or_chorus

/-- Proves that the number of students in both band and chorus is 30 -/
theorem students_in_both_band_and_chorus :
  students_in_both 300 110 140 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1434_143466


namespace NUMINAMATH_CALUDE_range_of_m_l1434_143433

-- Define the inequality system
def inequality_system (x m : ℝ) : Prop :=
  x + 5 < 5*x + 1 ∧ x - m > 1

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x > 1

-- Theorem statement
theorem range_of_m (m : ℝ) :
  (∀ x, inequality_system x m ↔ solution_set x) →
  m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1434_143433


namespace NUMINAMATH_CALUDE_expand_expression_l1434_143406

theorem expand_expression (y : ℝ) : (11 * y + 18) * (3 * y) = 33 * y^2 + 54 * y := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1434_143406


namespace NUMINAMATH_CALUDE_angle_supplement_l1434_143486

theorem angle_supplement (x : ℝ) : 
  (90 - x = 150) → (180 - x = 60) := by
  sorry

end NUMINAMATH_CALUDE_angle_supplement_l1434_143486


namespace NUMINAMATH_CALUDE_triangle_properties_l1434_143464

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Define the property that a^2 + b^2 - c^2 = ab -/
def satisfiesProperty (t : Triangle) : Prop :=
  t.a^2 + t.b^2 - t.c^2 = t.a * t.b

/-- Define the collinearity of vectors (2sin A, 1) and (cos C, 1/2) -/
def vectorsAreCollinear (t : Triangle) : Prop :=
  2 * Real.sin t.A * (1/2) = Real.cos t.C * 1

/-- Theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : satisfiesProperty t) 
  (h2 : vectorsAreCollinear t) : 
  t.C = π/3 ∧ t.B = π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1434_143464


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1434_143471

theorem no_solution_absolute_value_equation : ¬∃ (x : ℝ), |(-4 * x)| + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1434_143471


namespace NUMINAMATH_CALUDE_average_headcount_spring_terms_l1434_143427

def spring_02_03 : ℕ := 10900
def spring_03_04 : ℕ := 10500
def spring_04_05 : ℕ := 10700
def spring_05_06 : ℕ := 11300

def total_headcount : ℕ := spring_02_03 + spring_03_04 + spring_04_05 + spring_05_06
def num_terms : ℕ := 4

theorem average_headcount_spring_terms :
  (total_headcount : ℚ) / num_terms = 10850 := by sorry

end NUMINAMATH_CALUDE_average_headcount_spring_terms_l1434_143427


namespace NUMINAMATH_CALUDE_simple_interest_problem_l1434_143426

/-- Proves that given the conditions of the problem, the principal amount must be 700 --/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_rate : R > 0) : 
  (P * (R + 2) * 4) / 100 = (P * R * 4) / 100 + 56 → P = 700 := by
  sorry

#check simple_interest_problem

end NUMINAMATH_CALUDE_simple_interest_problem_l1434_143426


namespace NUMINAMATH_CALUDE_greatest_n_value_l1434_143499

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 ∧ ∃ m : ℤ, m = 7 ∧ 101 * m^2 ≤ 6400 :=
sorry

end NUMINAMATH_CALUDE_greatest_n_value_l1434_143499


namespace NUMINAMATH_CALUDE_harvard_mit_puzzle_l1434_143420

/-- Given that the product of letters in "harvard", "mit", and "hmmt" all equal 100,
    prove that the product of letters in "rad" and "trivia" equals 10000. -/
theorem harvard_mit_puzzle (h a r v d m i t : ℕ) : 
  h * a * r * v * a * r * d = 100 →
  m * i * t = 100 →
  h * m * m * t = 100 →
  (r * a * d) * (t * r * i * v * i * a) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_harvard_mit_puzzle_l1434_143420


namespace NUMINAMATH_CALUDE_no_real_solution_system_l1434_143410

theorem no_real_solution_system :
  ¬∃ (x y z : ℝ), (x + y - 2 - 4*x*y = 0) ∧ 
                  (y + z - 2 - 4*y*z = 0) ∧ 
                  (z + x - 2 - 4*z*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_system_l1434_143410


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_l1434_143442

theorem quadratic_perfect_square (p : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 + 27*x + p = (a*x + b)^2) → p = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_l1434_143442


namespace NUMINAMATH_CALUDE_lcm_12_18_l1434_143488

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l1434_143488


namespace NUMINAMATH_CALUDE_sequence_repeat_value_l1434_143490

theorem sequence_repeat_value (p q n : ℕ+) (x : Fin (n + 1) → ℤ)
  (h1 : p + q < n)
  (h2 : x 0 = 0 ∧ x n = 0)
  (h3 : ∀ i : Fin n, x (i + 1) - x i = p ∨ x (i + 1) - x i = -q) :
  ∃ i j : Fin (n + 1), i < j ∧ (i, j) ≠ (0, n) ∧ x i = x j := by
  sorry

end NUMINAMATH_CALUDE_sequence_repeat_value_l1434_143490


namespace NUMINAMATH_CALUDE_cosine_fourth_minus_sine_fourth_l1434_143417

theorem cosine_fourth_minus_sine_fourth (θ : ℝ) : 
  Real.cos θ ^ 4 - Real.sin θ ^ 4 = Real.cos (2 * θ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_fourth_minus_sine_fourth_l1434_143417


namespace NUMINAMATH_CALUDE_helen_cookies_theorem_l1434_143414

/-- The number of cookies Helen baked yesterday -/
def cookies_yesterday : ℕ := 31

/-- The number of cookies Helen baked the day before yesterday -/
def cookies_day_before_yesterday : ℕ := 419

/-- The total number of cookies Helen baked till last night -/
def total_cookies_till_last_night : ℕ := cookies_yesterday + cookies_day_before_yesterday

theorem helen_cookies_theorem : total_cookies_till_last_night = 450 := by
  sorry

end NUMINAMATH_CALUDE_helen_cookies_theorem_l1434_143414


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l1434_143435

theorem unique_number_with_gcd : ∃! n : ℕ, 90 < n ∧ n < 100 ∧ Nat.gcd 35 n = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l1434_143435


namespace NUMINAMATH_CALUDE_swimmers_passing_count_l1434_143459

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℝ
  speedA : ℝ
  speedB : ℝ
  totalTime : ℝ
  turnDelay : ℝ

/-- Calculates the number of times swimmers pass each other -/
def passingCount (s : SwimmingScenario) : ℕ :=
  sorry

/-- The main theorem stating the number of times swimmers pass each other -/
theorem swimmers_passing_count :
  let s : SwimmingScenario := {
    poolLength := 100,
    speedA := 4,
    speedB := 3,
    totalTime := 900,  -- 15 minutes in seconds
    turnDelay := 5
  }
  passingCount s = 63 := by sorry

end NUMINAMATH_CALUDE_swimmers_passing_count_l1434_143459


namespace NUMINAMATH_CALUDE_marathon_theorem_l1434_143418

def marathon_problem (total_miles : ℝ) (day1_percent : ℝ) (day3_miles : ℝ) : Prop :=
  let day1_miles := total_miles * day1_percent / 100
  let remaining_miles := total_miles - day1_miles
  let day2_miles := total_miles - day1_miles - day3_miles
  (day2_miles / remaining_miles) * 100 = 50

theorem marathon_theorem : 
  marathon_problem 70 20 28 := by
  sorry

end NUMINAMATH_CALUDE_marathon_theorem_l1434_143418


namespace NUMINAMATH_CALUDE_sum_equals_five_l1434_143461

/-- The mapping f that transforms (x, y) to (x, x+y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.1 + p.2)

/-- Theorem stating that a + b = 5 given the conditions -/
theorem sum_equals_five (a b : ℝ) (h : f (a, b) = (1, 3)) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_five_l1434_143461


namespace NUMINAMATH_CALUDE_ceiling_sum_equality_l1434_143481

theorem ceiling_sum_equality : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_equality_l1434_143481


namespace NUMINAMATH_CALUDE_max_blocks_fit_l1434_143416

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the box dimensions -/
def box : Dimensions := ⟨5, 4, 6⟩

/-- Represents the block dimensions -/
def block : Dimensions := ⟨3, 3, 2⟩

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := 6

theorem max_blocks_fit :
  (volume box) / (volume block) ≥ max_blocks ∧
  max_blocks * (volume block) ≤ volume box ∧
  max_blocks * block.length ≤ box.length + block.length - 1 ∧
  max_blocks * block.width ≤ box.width + block.width - 1 ∧
  max_blocks * block.height ≤ box.height + block.height - 1 :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l1434_143416


namespace NUMINAMATH_CALUDE_distance_between_points_l1434_143470

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1434_143470


namespace NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1434_143404

theorem intersection_and_union_of_sets (a : ℝ) :
  let A : Set ℝ := {-3, a + 1}
  let B : Set ℝ := {2 * a - 1, a^2 + 1}
  (A ∩ B = {3}) →
  (a = 2 ∧ A ∪ B = {-3, 3, 5}) := by
sorry

end NUMINAMATH_CALUDE_intersection_and_union_of_sets_l1434_143404


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_six_fifths_l1434_143449

def h (x : ℝ) : ℝ := 5 * x + 6

theorem h_zero_iff_b_eq_neg_six_fifths :
  ∀ b : ℝ, h b = 0 ↔ b = -6/5 := by sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_neg_six_fifths_l1434_143449


namespace NUMINAMATH_CALUDE_quadratic_equation_magnitude_l1434_143432

theorem quadratic_equation_magnitude (z : ℂ) : 
  z^2 - 12*z + 157 = 0 → ∃! r : ℝ, (Complex.abs z = r ∧ r = Real.sqrt 157) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_magnitude_l1434_143432


namespace NUMINAMATH_CALUDE_age_difference_l1434_143430

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 15) : A - C = 15 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1434_143430


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l1434_143460

/-- Given a parabola y = ax^2 + bx + c passing through (0,0), (2T,0), and (2T+1,15),
    where a and T are integers and T ≠ 0, prove that the largest possible value of N is -10,
    where N is the sum of the coordinates of the vertex point. -/
theorem parabola_vertex_sum_max (a T : ℤ) (hT : T ≠ 0) : 
  ∃ (b c : ℤ),
    (0 = c) ∧
    (0 = 4*a*T^2 + 2*b*T + c) ∧
    (15 = a*(2*T+1)^2 + b*(2*T+1) + c) →
    (∀ N : ℤ, N = T - a*T^2 → N ≤ -10) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l1434_143460


namespace NUMINAMATH_CALUDE_f_of_3_equals_9_l1434_143463

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_3_equals_9 : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_equals_9_l1434_143463


namespace NUMINAMATH_CALUDE_blake_grocery_change_l1434_143434

/-- Calculates the change Blake receives after purchasing groceries with discounts and sales tax. -/
theorem blake_grocery_change (oranges apples mangoes strawberries bananas : ℚ)
  (strawberry_discount banana_discount sales_tax : ℚ)
  (blake_money : ℚ)
  (h1 : oranges = 40)
  (h2 : apples = 50)
  (h3 : mangoes = 60)
  (h4 : strawberries = 30)
  (h5 : bananas = 20)
  (h6 : strawberry_discount = 10 / 100)
  (h7 : banana_discount = 5 / 100)
  (h8 : sales_tax = 7 / 100)
  (h9 : blake_money = 300) :
  let discounted_strawberries := strawberries * (1 - strawberry_discount)
  let discounted_bananas := bananas * (1 - banana_discount)
  let total_cost := oranges + apples + mangoes + discounted_strawberries + discounted_bananas
  let total_with_tax := total_cost * (1 + sales_tax)
  blake_money - total_with_tax = 90.28 := by
sorry


end NUMINAMATH_CALUDE_blake_grocery_change_l1434_143434


namespace NUMINAMATH_CALUDE_cos_difference_from_sum_l1434_143462

theorem cos_difference_from_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_from_sum_l1434_143462


namespace NUMINAMATH_CALUDE_k_less_than_one_necessary_not_sufficient_l1434_143440

/-- For x in the open interval (0, π/2), "k < 1" is a necessary but not sufficient condition for k*sin(x)*cos(x) < x. -/
theorem k_less_than_one_necessary_not_sufficient :
  ∀ k : ℝ, (∃ x : ℝ, 0 < x ∧ x < π/2 ∧ k * Real.sin x * Real.cos x < x) →
  (k < 1 ∧ ∃ k' ≥ 1, ∃ x : ℝ, 0 < x ∧ x < π/2 ∧ k' * Real.sin x * Real.cos x < x) :=
by sorry

end NUMINAMATH_CALUDE_k_less_than_one_necessary_not_sufficient_l1434_143440


namespace NUMINAMATH_CALUDE_minks_set_free_ratio_l1434_143473

/-- Represents the mink coat problem -/
structure MinkCoatProblem where
  skins_per_coat : ℕ
  initial_minks : ℕ
  babies_per_mink : ℕ
  coats_made : ℕ

/-- Calculates the total number of minks -/
def total_minks (p : MinkCoatProblem) : ℕ :=
  p.initial_minks * (1 + p.babies_per_mink)

/-- Calculates the number of minks used for coats -/
def minks_used_for_coats (p : MinkCoatProblem) : ℕ :=
  p.skins_per_coat * p.coats_made

/-- Calculates the number of minks set free -/
def minks_set_free (p : MinkCoatProblem) : ℕ :=
  total_minks p - minks_used_for_coats p

/-- The main theorem stating the ratio of minks set free to total minks -/
theorem minks_set_free_ratio (p : MinkCoatProblem) 
  (h1 : p.skins_per_coat = 15)
  (h2 : p.initial_minks = 30)
  (h3 : p.babies_per_mink = 6)
  (h4 : p.coats_made = 7) :
  minks_set_free p * 2 = total_minks p :=
sorry

end NUMINAMATH_CALUDE_minks_set_free_ratio_l1434_143473


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1434_143408

theorem expand_and_simplify (x : ℝ) : 
  (x + 2)^2 + x * (3 - x) = 7 * x + 4 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1434_143408


namespace NUMINAMATH_CALUDE_opposite_of_two_thirds_l1434_143448

theorem opposite_of_two_thirds :
  (-(2 : ℚ) / 3) = (-1 : ℚ) * (2 : ℚ) / 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_thirds_l1434_143448


namespace NUMINAMATH_CALUDE_justine_colored_sheets_l1434_143496

/-- Given 2450 sheets of paper evenly split into 5 binders, 
    prove that Justine colors 245 sheets when she colors 
    half the sheets in one binder. -/
theorem justine_colored_sheets : 
  let total_sheets : ℕ := 2450
  let num_binders : ℕ := 5
  let sheets_per_binder : ℕ := total_sheets / num_binders
  let justine_colored : ℕ := sheets_per_binder / 2
  justine_colored = 245 := by
  sorry

#check justine_colored_sheets

end NUMINAMATH_CALUDE_justine_colored_sheets_l1434_143496


namespace NUMINAMATH_CALUDE_avery_donation_l1434_143480

/-- The number of shirts Avery puts in the donation box -/
def num_shirts : ℕ := 4

/-- The number of pants Avery puts in the donation box -/
def num_pants : ℕ := 2 * num_shirts

/-- The number of shorts Avery puts in the donation box -/
def num_shorts : ℕ := num_pants / 2

/-- The total number of clothes Avery is donating -/
def total_clothes : ℕ := num_shirts + num_pants + num_shorts

theorem avery_donation : total_clothes = 16 := by
  sorry

end NUMINAMATH_CALUDE_avery_donation_l1434_143480


namespace NUMINAMATH_CALUDE_four_equal_area_volume_prisms_l1434_143423

/-- A square prism with integer edge lengths where the surface area equals the volume. -/
structure EqualAreaVolumePrism where
  a : ℕ  -- length of the base
  b : ℕ  -- height of the prism
  h : 2 * a^2 + 4 * a * b = a^2 * b

/-- The set of all square prisms with integer edge lengths where the surface area equals the volume. -/
def allEqualAreaVolumePrisms : Set EqualAreaVolumePrism :=
  {p : EqualAreaVolumePrism | True}

/-- The theorem stating that there are only four square prisms with integer edge lengths
    where the surface area equals the volume. -/
theorem four_equal_area_volume_prisms :
  allEqualAreaVolumePrisms = {
    ⟨12, 3, by sorry⟩,
    ⟨8, 4, by sorry⟩,
    ⟨6, 6, by sorry⟩,
    ⟨5, 10, by sorry⟩
  } := by sorry

end NUMINAMATH_CALUDE_four_equal_area_volume_prisms_l1434_143423


namespace NUMINAMATH_CALUDE_carlos_laundry_loads_l1434_143411

theorem carlos_laundry_loads (wash_time_per_load : ℕ) (dry_time : ℕ) (total_time : ℕ) 
  (h1 : wash_time_per_load = 45)
  (h2 : dry_time = 75)
  (h3 : total_time = 165) :
  ∃ n : ℕ, n * wash_time_per_load + dry_time = total_time ∧ n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_carlos_laundry_loads_l1434_143411


namespace NUMINAMATH_CALUDE_profit_percentage_example_l1434_143441

/-- Calculates the profit percentage given the selling price and cost price. -/
def profit_percentage (selling_price cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that for a selling price of 250 and a cost price of 200, 
    the profit percentage is 25%. -/
theorem profit_percentage_example : 
  profit_percentage 250 200 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_example_l1434_143441


namespace NUMINAMATH_CALUDE_john_crab_earnings_l1434_143421

/-- Calculates the weekly earnings from crab sales given the following conditions:
  * Number of baskets reeled in per collection
  * Number of collections per week
  * Number of crabs per basket
  * Price per crab
-/
def weekly_crab_earnings (baskets_per_collection : ℕ) (collections_per_week : ℕ) (crabs_per_basket : ℕ) (price_per_crab : ℕ) : ℕ :=
  baskets_per_collection * collections_per_week * crabs_per_basket * price_per_crab

/-- Theorem stating that under the given conditions, John makes $72 per week from selling crabs -/
theorem john_crab_earnings :
  weekly_crab_earnings 3 2 4 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_john_crab_earnings_l1434_143421


namespace NUMINAMATH_CALUDE_units_digit_of_17_pow_2045_l1434_143475

theorem units_digit_of_17_pow_2045 : (17^2045 : ℕ) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_17_pow_2045_l1434_143475


namespace NUMINAMATH_CALUDE_three_times_work_time_l1434_143495

/-- Given a person can complete a piece of work in a certain number of days,
    this function calculates how many days it will take to complete a multiple of that work. -/
def time_for_multiple_work (days_for_single_work : ℕ) (work_multiple : ℕ) : ℕ :=
  days_for_single_work * work_multiple

/-- Theorem stating that if a person can complete a piece of work in 8 days,
    then they will complete three times the work in 24 days. -/
theorem three_times_work_time :
  time_for_multiple_work 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_three_times_work_time_l1434_143495


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l1434_143405

theorem quadratic_root_condition (a b c : ℝ) (h : a ≠ 0) :
  (∃ r s : ℝ, r * s = -1 ∧ a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) →
  c = -a :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l1434_143405


namespace NUMINAMATH_CALUDE_rectangle_area_l1434_143431

/-- Given a rectangle with diagonal length 2a + b, its area is 2ab -/
theorem rectangle_area (a b : ℝ) (h : a > 0 ∧ b > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = (2*a + b)^2 ∧ x * y = 2*a*b :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1434_143431


namespace NUMINAMATH_CALUDE_select_students_l1434_143479

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem select_students (num_boys num_girls : ℕ) (boys_selected girls_selected : ℕ) : 
  num_boys = 11 → num_girls = 10 → boys_selected = 2 → girls_selected = 3 →
  (choose num_girls girls_selected) * (choose num_boys boys_selected) = 6600 := by
  sorry

end NUMINAMATH_CALUDE_select_students_l1434_143479


namespace NUMINAMATH_CALUDE_base_four_of_85_l1434_143447

def base_four_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_four_of_85 :
  base_four_representation 85 = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_four_of_85_l1434_143447


namespace NUMINAMATH_CALUDE_sewn_fabric_theorem_l1434_143412

/-- The length of a sewn fabric piece given the number of fabric pieces, 
    length of each piece, and length of each joint. -/
def sewn_fabric_length (num_pieces : ℕ) (piece_length : ℝ) (joint_length : ℝ) : ℝ :=
  num_pieces * piece_length - (num_pieces - 1) * joint_length

/-- Theorem stating that 20 pieces of 10 cm fabric sewn with 0.5 cm joints 
    result in a 190.5 cm long piece. -/
theorem sewn_fabric_theorem :
  sewn_fabric_length 20 10 0.5 = 190.5 := by
  sorry

#eval sewn_fabric_length 20 10 0.5

end NUMINAMATH_CALUDE_sewn_fabric_theorem_l1434_143412


namespace NUMINAMATH_CALUDE_dormitory_problem_l1434_143450

theorem dormitory_problem (x : ℕ) 
  (h1 : x > 0) 
  (h2 : 4 * x + 18 < 6 * x) 
  (h3 : 4 * x + 18 > 6 * (x - 1)) : 
  x = 10 ∨ x = 11 := by
sorry

end NUMINAMATH_CALUDE_dormitory_problem_l1434_143450


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1434_143452

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1434_143452


namespace NUMINAMATH_CALUDE_amusement_park_problem_l1434_143477

/-- The number of children who got on the Ferris wheel -/
def ferris_wheel_riders : ℕ := sorry

theorem amusement_park_problem :
  let total_children : ℕ := 5
  let ferris_wheel_cost : ℕ := 5
  let merry_go_round_cost : ℕ := 3
  let ice_cream_cost : ℕ := 8
  let ice_cream_per_child : ℕ := 2
  let total_spent : ℕ := 110
  ferris_wheel_riders * ferris_wheel_cost +
  total_children * merry_go_round_cost +
  total_children * ice_cream_per_child * ice_cream_cost = total_spent ∧
  ferris_wheel_riders = 3 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_problem_l1434_143477


namespace NUMINAMATH_CALUDE_adlai_chickens_l1434_143409

def num_dogs : ℕ := 2
def total_legs : ℕ := 10
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

theorem adlai_chickens :
  (total_legs - num_dogs * legs_per_dog) / legs_per_chicken = 1 := by
  sorry

end NUMINAMATH_CALUDE_adlai_chickens_l1434_143409


namespace NUMINAMATH_CALUDE_stratified_by_educational_stage_is_most_reasonable_l1434_143429

-- Define the different sampling methods
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

-- Define the educational stages
inductive EducationalStage
| Primary
| JuniorHigh
| HighSchool

-- Define the conditions
def significantDifferencesInEducationalStages : Prop := True
def noSignificantDifferencesBetweenGenders : Prop := True
def goalIsUnderstandVisionConditions : Prop := True

-- Define the most reasonable sampling method
def mostReasonableSamplingMethod : SamplingMethod := SamplingMethod.StratifiedByEducationalStage

-- Theorem statement
theorem stratified_by_educational_stage_is_most_reasonable :
  significantDifferencesInEducationalStages →
  noSignificantDifferencesBetweenGenders →
  goalIsUnderstandVisionConditions →
  mostReasonableSamplingMethod = SamplingMethod.StratifiedByEducationalStage :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_by_educational_stage_is_most_reasonable_l1434_143429


namespace NUMINAMATH_CALUDE_triangle_theorem_l1434_143485

theorem triangle_theorem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  c - a = 1 →
  b = Real.sqrt 7 →
  B = π / 3 ∧
  (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1434_143485


namespace NUMINAMATH_CALUDE_distance_AB_is_five_halves_l1434_143487

/-- Line represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Line represented by a linear equation ax + by = c -/
structure LinearLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def l₁ : ParametricLine :=
  { x := λ t => 1 + 3 * t
    y := λ t => 2 - 4 * t }

def l₂ : LinearLine :=
  { a := 2
    b := -4
    c := 5 }

def A : Point :=
  { x := 1
    y := 2 }

/-- Function to find the intersection point of a parametric line and a linear line -/
def intersection (pl : ParametricLine) (ll : LinearLine) : Point :=
  sorry

/-- Function to calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem distance_AB_is_five_halves :
  distance A (intersection l₁ l₂) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_AB_is_five_halves_l1434_143487


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l1434_143422

/-- Represents an equilateral triangle with pegs -/
structure TriangleWithPegs where
  sideLength : ℕ
  pegDistance : ℕ

/-- Counts the number of ways to choose pegs that divide the triangle into 9 regions -/
def countValidPegChoices (t : TriangleWithPegs) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem triangle_division_theorem (t : TriangleWithPegs) :
  t.sideLength = 6 ∧ t.pegDistance = 1 → countValidPegChoices t = 456 :=
sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l1434_143422


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l1434_143458

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec toBits (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBits (m / 2)
  toBits n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, true, true]        -- 110₂
  let product := [false, true, true, true, true, false, true]  -- 1011110₂
  binaryToNat a * binaryToNat b = binaryToNat product :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l1434_143458


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l1434_143497

theorem farmer_land_ownership (T : ℝ) 
  (h1 : T > 0)
  (h2 : 0.8 * T + 0.2 * T = T)
  (h3 : 0.05 * (0.8 * T) + 0.3 * (0.2 * T) = 720) :
  0.8 * T = 5760 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l1434_143497


namespace NUMINAMATH_CALUDE_trevor_coin_count_l1434_143446

/-- Given that Trevor counted 29 quarters and has 48 more coins in total than quarters,
    prove that the total number of coins Trevor counted is 77. -/
theorem trevor_coin_count :
  let quarters : ℕ := 29
  let extra_coins : ℕ := 48
  quarters + extra_coins = 77
  := by sorry

end NUMINAMATH_CALUDE_trevor_coin_count_l1434_143446


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1434_143425

theorem square_root_of_sixteen : ∃ (x : ℝ), x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1434_143425


namespace NUMINAMATH_CALUDE_lottery_cheating_suspicion_l1434_143428

/-- The number of balls in the lottery -/
def total_balls : ℕ := 45

/-- The number of winning balls in each draw -/
def winning_balls : ℕ := 6

/-- The probability of no repetition in a single draw -/
def p : ℚ := (total_balls - winning_balls).choose winning_balls / total_balls.choose winning_balls

/-- The suspicion threshold -/
def threshold : ℚ := 1 / 100

theorem lottery_cheating_suspicion :
  p ^ 6 < threshold ∧ p ^ 5 ≥ threshold :=
sorry

end NUMINAMATH_CALUDE_lottery_cheating_suspicion_l1434_143428


namespace NUMINAMATH_CALUDE_zero_of_f_l1434_143468

def f (x : ℝ) := x + 1

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_zero_of_f_l1434_143468


namespace NUMINAMATH_CALUDE_lindas_furniture_fraction_l1434_143445

/-- Given Linda's original savings and the cost of a TV, prove the fraction spent on furniture. -/
theorem lindas_furniture_fraction (original_savings : ℚ) (tv_cost : ℚ) 
  (h1 : original_savings = 600)
  (h2 : tv_cost = 300) :
  (original_savings - tv_cost) / original_savings = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_lindas_furniture_fraction_l1434_143445


namespace NUMINAMATH_CALUDE_combined_weight_leo_kendra_prove_combined_weight_l1434_143492

/-- The combined weight of Leo and Kendra given Leo's current weight and the condition of their weight relationship after Leo gains 10 pounds. -/
theorem combined_weight_leo_kendra : ℝ → ℝ → Prop :=
  fun leo_weight kendra_weight =>
    (leo_weight = 104) →
    (leo_weight + 10 = 1.5 * kendra_weight) →
    (leo_weight + kendra_weight = 180)

/-- The theorem statement -/
theorem prove_combined_weight : ∃ (leo_weight kendra_weight : ℝ),
  combined_weight_leo_kendra leo_weight kendra_weight :=
sorry

end NUMINAMATH_CALUDE_combined_weight_leo_kendra_prove_combined_weight_l1434_143492


namespace NUMINAMATH_CALUDE_price_difference_per_can_l1434_143484

/-- Proves that the difference in price per can between the local grocery store and the bulk warehouse is 25 cents -/
theorem price_difference_per_can (bulk_price : ℚ) (bulk_cans : ℕ) (grocery_price : ℚ) (grocery_cans : ℕ) 
  (h1 : bulk_price = 12) 
  (h2 : bulk_cans = 48) 
  (h3 : grocery_price = 6) 
  (h4 : grocery_cans = 12) : 
  (grocery_price / grocery_cans - bulk_price / bulk_cans) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_per_can_l1434_143484
