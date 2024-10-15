import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1383_138353

/-- Given an arithmetic sequence where the first term is 2/3 and the 17th term is 5/6,
    the 9th term is 3/4. -/
theorem arithmetic_sequence_ninth_term 
  (a : ℚ) 
  (seq : ℕ → ℚ) 
  (h1 : seq 1 = 2/3) 
  (h2 : seq 17 = 5/6) 
  (h3 : ∀ n : ℕ, seq (n + 1) - seq n = seq 2 - seq 1) : 
  seq 9 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1383_138353


namespace NUMINAMATH_CALUDE_evaluate_expression_l1383_138301

theorem evaluate_expression : 3 * (-3)^4 + 3 * (-3)^3 + 3 * (-3)^2 + 3 * 3^2 + 3 * 3^3 + 3 * 3^4 = 540 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1383_138301


namespace NUMINAMATH_CALUDE_lecture_duration_l1383_138367

/-- 
Given a lecture that lasts for 2 hours and m minutes, where the positions of the
hour and minute hands on the clock at the end of the lecture are exactly swapped
from their positions at the beginning, this theorem states that the integer part
of m is 46.
-/
theorem lecture_duration (m : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t < 120 + m → 
    (5 * (120 + m - t) = (60 * t) % 360 ∨ 5 * (120 + m - t) = ((60 * t) % 360 + 360) % 360)) →
  Int.floor m = 46 := by
  sorry

end NUMINAMATH_CALUDE_lecture_duration_l1383_138367


namespace NUMINAMATH_CALUDE_candy_distribution_l1383_138373

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : total_candy = 344)
  (h2 : num_students = 43)
  (h3 : pieces_per_student * num_students = total_candy) :
  pieces_per_student = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1383_138373


namespace NUMINAMATH_CALUDE_jason_borrowed_amount_l1383_138346

/-- Represents the value of a chore based on its position in the cycle -/
def chore_value (n : ℕ) : ℕ :=
  match n % 6 with
  | 1 => 1
  | 2 => 3
  | 3 => 5
  | 4 => 7
  | 5 => 9
  | 0 => 11
  | _ => 0  -- This case should never occur

/-- Calculates the total value of a complete cycle of 6 chores -/
def cycle_value : ℕ := 
  (chore_value 1) + (chore_value 2) + (chore_value 3) + 
  (chore_value 4) + (chore_value 5) + (chore_value 6)

/-- Theorem: Jason borrowed $288 -/
theorem jason_borrowed_amount : 
  (cycle_value * (48 / 6) = 288) := by
  sorry

end NUMINAMATH_CALUDE_jason_borrowed_amount_l1383_138346


namespace NUMINAMATH_CALUDE_laptop_price_l1383_138386

theorem laptop_price (sticker_price : ℝ) : 
  (sticker_price * 0.8 - 100 = sticker_price * 0.7 - 25) → sticker_price = 750 := by
sorry

end NUMINAMATH_CALUDE_laptop_price_l1383_138386


namespace NUMINAMATH_CALUDE_min_abs_b_minus_c_l1383_138339

/-- Given real numbers a, b, c satisfying (a - 2b - 1)² + (a - c - ln c)² = 0,
    the minimum value of |b - c| is 1. -/
theorem min_abs_b_minus_c (a b c : ℝ) 
    (h : (a - 2*b - 1)^2 + (a - c - Real.log c)^2 = 0) :
    ∀ x : ℝ, |b - c| ≤ x → 1 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_min_abs_b_minus_c_l1383_138339


namespace NUMINAMATH_CALUDE_right_cone_diameter_l1383_138311

theorem right_cone_diameter (h : ℝ) (s : ℝ) (d : ℝ) :
  h = 3 →
  s = 5 →
  s^2 = h^2 + (d/2)^2 →
  d = 8 :=
by sorry

end NUMINAMATH_CALUDE_right_cone_diameter_l1383_138311


namespace NUMINAMATH_CALUDE_M_intersect_P_equals_singleton_l1383_138349

-- Define the sets M and P
def M : Set (ℝ × ℝ) := {(x, y) | 4 * x + y = 6}
def P : Set (ℝ × ℝ) := {(x, y) | 3 * x + 2 * y = 7}

-- Theorem statement
theorem M_intersect_P_equals_singleton : M ∩ P = {(1, 2)} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_P_equals_singleton_l1383_138349


namespace NUMINAMATH_CALUDE_fraction_equivalence_l1383_138306

theorem fraction_equivalence (x : ℝ) : (x + 1) / (x + 3) = 1 / 3 ↔ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l1383_138306


namespace NUMINAMATH_CALUDE_rectangle_area_equals_50_l1383_138374

/-- The area of a rectangle with height x and width 2x, whose perimeter is equal to the perimeter of an equilateral triangle with side length 10, is 50. -/
theorem rectangle_area_equals_50 : ∃ x : ℝ,
  let rectangle_height := x
  let rectangle_width := 2 * x
  let rectangle_perimeter := 2 * (rectangle_height + rectangle_width)
  let triangle_side_length := 10
  let triangle_perimeter := 3 * triangle_side_length
  let rectangle_area := rectangle_height * rectangle_width
  rectangle_perimeter = triangle_perimeter ∧ rectangle_area = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_equals_50_l1383_138374


namespace NUMINAMATH_CALUDE_fencing_calculation_l1383_138362

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area > 0 → uncovered_side > 0 → area / uncovered_side + 2 * uncovered_side = 76 := by
  sorry

#check fencing_calculation

end NUMINAMATH_CALUDE_fencing_calculation_l1383_138362


namespace NUMINAMATH_CALUDE_product_in_second_quadrant_l1383_138357

/-- The complex number representing the product (2+i)(-1+i) -/
def z : ℂ := (2 + Complex.I) * (-1 + Complex.I)

/-- The real part of z -/
def real_part : ℝ := z.re

/-- The imaginary part of z -/
def imag_part : ℝ := z.im

/-- Predicate for a complex number being in the second quadrant -/
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

theorem product_in_second_quadrant : in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_product_in_second_quadrant_l1383_138357


namespace NUMINAMATH_CALUDE_exam_students_count_l1383_138345

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) (excluded_count : ℕ) :
  total_average = 80 →
  excluded_average = 20 →
  remaining_average = 92 →
  excluded_count = 5 →
  ∃ N : ℕ, 
    N * total_average = (N - excluded_count) * remaining_average + excluded_count * excluded_average ∧
    N = 30 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l1383_138345


namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1383_138330

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to check if a line has equal intercepts on both axes
def equalIntercepts (l : Line2D) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.c / l.a = l.c / l.b

-- The main theorem
theorem line_through_point_with_equal_intercepts :
  ∃ (l1 l2 : Line2D),
    (pointOnLine ⟨2, 3⟩ l1 ∧ equalIntercepts l1) ∧
    (pointOnLine ⟨2, 3⟩ l2 ∧ equalIntercepts l2) ∧
    ((l1.a = 1 ∧ l1.b = 1 ∧ l1.c = -5) ∨ (l2.a = 3 ∧ l2.b = -2 ∧ l2.c = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l1383_138330


namespace NUMINAMATH_CALUDE_triangle_top_angle_l1383_138326

theorem triangle_top_angle (total : ℝ) (right : ℝ) (left : ℝ) (top : ℝ) : 
  total = 250 →
  right = 60 →
  left = 2 * right →
  total = left + right + top →
  top = 70 := by
sorry

end NUMINAMATH_CALUDE_triangle_top_angle_l1383_138326


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_prism_volume_l1383_138309

theorem cube_surface_area_equal_prism_volume (a b c : ℝ) (h : a = 6 ∧ b = 3 ∧ c = 36) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 216 * 3 ^ (2/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_prism_volume_l1383_138309


namespace NUMINAMATH_CALUDE_second_shift_size_l1383_138307

/-- The number of members in each shift of a company and their participation in a pension program. -/
structure CompanyShifts where
  first_shift : ℕ
  second_shift : ℕ
  third_shift : ℕ
  first_participation : ℚ
  second_participation : ℚ
  third_participation : ℚ
  total_participation : ℚ

/-- The company shifts satisfy the given conditions -/
def satisfies_conditions (c : CompanyShifts) : Prop :=
  c.first_shift = 60 ∧
  c.third_shift = 40 ∧
  c.first_participation = 1/5 ∧
  c.second_participation = 2/5 ∧
  c.third_participation = 1/10 ∧
  c.total_participation = 6/25 ∧
  (c.first_shift * c.first_participation + c.second_shift * c.second_participation + c.third_shift * c.third_participation : ℚ) = 
    c.total_participation * (c.first_shift + c.second_shift + c.third_shift)

/-- The theorem stating that the second shift has 50 members -/
theorem second_shift_size (c : CompanyShifts) (h : satisfies_conditions c) : c.second_shift = 50 := by
  sorry

end NUMINAMATH_CALUDE_second_shift_size_l1383_138307


namespace NUMINAMATH_CALUDE_greatest_rational_root_l1383_138370

-- Define the quadratic equation type
structure QuadraticEquation where
  a : Nat
  b : Nat
  c : Nat
  h_a : a ≤ 100
  h_b : b ≤ 100
  h_c : c ≤ 100

-- Define a rational root
def RationalRoot (q : QuadraticEquation) (x : ℚ) : Prop :=
  q.a * x^2 + q.b * x + q.c = 0

-- State the theorem
theorem greatest_rational_root (q : QuadraticEquation) :
  ∃ (x : ℚ), RationalRoot q x ∧ 
  ∀ (y : ℚ), RationalRoot q y → y ≤ x ∧ x = -1/99 :=
sorry

end NUMINAMATH_CALUDE_greatest_rational_root_l1383_138370


namespace NUMINAMATH_CALUDE_x_value_for_given_y_z_exists_constant_k_l1383_138397

/-- Given a relationship between x, y, and z, prove that x equals 5/8 for specific values of y and z -/
theorem x_value_for_given_y_z : ∀ (x y z k : ℝ), 
  (x = k * (z / y^2)) →  -- Relationship between x, y, and z
  (1 = k * (2 / 3^2)) →  -- Initial condition
  (y = 6 ∧ z = 5) →      -- New values for y and z
  x = 5/8 := by
    sorry

/-- There exists a constant k that satisfies the given conditions -/
theorem exists_constant_k : ∃ (k : ℝ), 
  (1 = k * (2 / 3^2)) ∧
  (∀ (x y z : ℝ), x = k * (z / y^2)) := by
    sorry

end NUMINAMATH_CALUDE_x_value_for_given_y_z_exists_constant_k_l1383_138397


namespace NUMINAMATH_CALUDE_bike_riding_average_l1383_138359

/-- Calculates the average miles ridden per day given total miles and years --/
def average_miles_per_day (total_miles : ℕ) (years : ℕ) : ℚ :=
  total_miles / (years * 365)

/-- Theorem stating that riding 3,285 miles over 3 years averages to 3 miles per day --/
theorem bike_riding_average :
  average_miles_per_day 3285 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bike_riding_average_l1383_138359


namespace NUMINAMATH_CALUDE_perfect_square_sum_partition_not_perfect_square_sum_partition_l1383_138314

/-- A partition of a set of natural numbers -/
def Partition (n : ℕ) := Fin 2 → Finset ℕ

/-- Predicate to check if a partition satisfies the perfect square sum property -/
def HasPerfectSquareSum (p : Partition n) : Prop :=
  ∃ (i : Fin 2) (a b : ℕ), a ≠ b ∧ a ∈ p i ∧ b ∈ p i ∧ ∃ (k : ℕ), a + b = k^2

/-- The main theorem stating the property holds for all n ≥ 15 -/
theorem perfect_square_sum_partition (n : ℕ) (h : n ≥ 15) :
  ∀ (p : Partition n), HasPerfectSquareSum p :=
sorry

/-- The property does not hold for n < 15 -/
theorem not_perfect_square_sum_partition (n : ℕ) (h : n < 15) :
  ∃ (p : Partition n), ¬HasPerfectSquareSum p :=
sorry

end NUMINAMATH_CALUDE_perfect_square_sum_partition_not_perfect_square_sum_partition_l1383_138314


namespace NUMINAMATH_CALUDE_protein_content_lower_bound_l1383_138324

/-- Represents the protein content of a beverage can -/
structure BeverageCan where
  netWeight : ℝ
  proteinPercentage : ℝ

/-- Theorem: Given a beverage can with net weight 300 grams and protein content ≥ 0.6%,
    the protein content is at least 1.8 grams -/
theorem protein_content_lower_bound (can : BeverageCan)
    (h1 : can.netWeight = 300)
    (h2 : can.proteinPercentage ≥ 0.6) :
    can.netWeight * (can.proteinPercentage / 100) ≥ 1.8 := by
  sorry

#check protein_content_lower_bound

end NUMINAMATH_CALUDE_protein_content_lower_bound_l1383_138324


namespace NUMINAMATH_CALUDE_problem_statement_l1383_138372

theorem problem_statement :
  (∀ x : ℝ, x^2 - 8*x + 17 > 0) ∧
  (∀ x : ℝ, (x + 2)^2 - (x - 3)^2 ≥ 0 → x ≥ 1/2) ∧
  (∃ n : ℕ, 11 ∣ 6*n^2 - 7) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1383_138372


namespace NUMINAMATH_CALUDE_ribbon_cuts_l1383_138316

/-- The number of cuts needed to divide ribbon rolls into smaller pieces -/
def cuts_needed (num_rolls : ℕ) (roll_length : ℕ) (piece_length : ℕ) : ℕ :=
  num_rolls * ((roll_length / piece_length) - 1)

/-- Theorem: The number of cuts needed to divide 5 rolls of 50-meter ribbon into 2-meter pieces is 120 -/
theorem ribbon_cuts : cuts_needed 5 50 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_cuts_l1383_138316


namespace NUMINAMATH_CALUDE_two_valid_configurations_l1383_138383

-- Define a 4x4 table as a function from (Fin 4 × Fin 4) to Char
def Table := Fin 4 → Fin 4 → Char

-- Define the swap operations
def swapFirstTwoRows (t : Table) : Table :=
  fun i j => if i = 0 then t 1 j else if i = 1 then t 0 j else t i j

def swapFirstTwoCols (t : Table) : Table :=
  fun i j => if j = 0 then t i 1 else if j = 1 then t i 0 else t i j

def swapLastTwoCols (t : Table) : Table :=
  fun i j => if j = 2 then t i 3 else if j = 3 then t i 2 else t i j

-- Define the property of identical letters in corresponding quadrants
def maintainsQuadrantProperty (t1 t2 : Table) : Prop :=
  ∀ i j, (t1 i j = t1 (i + 2) j ∧ t1 i j = t1 i (j + 2) ∧ t1 i j = t1 (i + 2) (j + 2)) →
         (t2 i j = t2 (i + 2) j ∧ t2 i j = t2 i (j + 2) ∧ t2 i j = t2 (i + 2) (j + 2))

-- Define the initial table
def initialTable : Table :=
  fun i j => match (i, j) with
  | (0, 0) => 'A' | (0, 1) => 'B' | (0, 2) => 'C' | (0, 3) => 'D'
  | (1, 0) => 'D' | (1, 1) => 'C' | (1, 2) => 'B' | (1, 3) => 'A'
  | (2, 0) => 'C' | (2, 1) => 'A' | (2, 2) => 'C' | (2, 3) => 'A'
  | (3, 0) => 'B' | (3, 1) => 'D' | (3, 2) => 'B' | (3, 3) => 'D'

-- The main theorem
theorem two_valid_configurations :
  ∃! (validConfigs : Finset Table),
    validConfigs.card = 2 ∧
    (∀ t ∈ validConfigs,
      maintainsQuadrantProperty initialTable
        (swapLastTwoCols (swapFirstTwoCols (swapFirstTwoRows t)))) := by
  sorry

end NUMINAMATH_CALUDE_two_valid_configurations_l1383_138383


namespace NUMINAMATH_CALUDE_closest_to_fraction_l1383_138331

def options : List ℝ := [4000, 5000, 6000, 7000, 8000]

theorem closest_to_fraction (x : ℝ) (h : x = 510 / 0.125) :
  4000 ∈ options ∧ ∀ y ∈ options, |x - 4000| ≤ |x - y| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l1383_138331


namespace NUMINAMATH_CALUDE_fifth_score_for_average_85_l1383_138390

/-- Given four test scores and a desired average, calculate the required fifth score -/
def calculate_fifth_score (score1 score2 score3 score4 : ℕ) (desired_average : ℚ) : ℚ :=
  5 * desired_average - (score1 + score2 + score3 + score4)

/-- Theorem: The fifth score needed to achieve an average of 85 given the first four scores -/
theorem fifth_score_for_average_85 :
  calculate_fifth_score 85 79 92 84 85 = 85 := by sorry

end NUMINAMATH_CALUDE_fifth_score_for_average_85_l1383_138390


namespace NUMINAMATH_CALUDE_soybean_experiment_results_l1383_138344

/-- Represents the weight distribution of soybean samples -/
structure WeightDistribution :=
  (low : ℕ) -- count in [100, 150) range
  (mid : ℕ) -- count in [150, 200) range
  (high : ℕ) -- count in [200, 250] range

/-- Represents the experimental setup for soybean fields -/
structure SoybeanExperiment :=
  (field_A : WeightDistribution)
  (field_B : WeightDistribution)
  (sample_size : ℕ)
  (critical_value : ℝ)

/-- Calculates the chi-square statistic for the experiment -/
def calculate_chi_square (exp : SoybeanExperiment) : ℝ :=
  sorry

/-- Calculates the probability of selecting at least one full grain from both fields -/
def probability_full_grain (exp : SoybeanExperiment) : ℚ :=
  sorry

/-- Calculates the expected number of full grains in 100 samples from field A -/
def expected_full_grains (exp : SoybeanExperiment) : ℕ :=
  sorry

/-- Calculates the variance of full grains in 100 samples from field A -/
def variance_full_grains (exp : SoybeanExperiment) : ℚ :=
  sorry

/-- Main theorem about the soybean experiment -/
theorem soybean_experiment_results (exp : SoybeanExperiment) 
  (h1 : exp.field_A = ⟨3, 6, 11⟩)
  (h2 : exp.field_B = ⟨6, 10, 4⟩)
  (h3 : exp.sample_size = 20)
  (h4 : exp.critical_value = 5.024) :
  calculate_chi_square exp > exp.critical_value ∧
  probability_full_grain exp = 89 / 100 ∧
  expected_full_grains exp = 55 ∧
  variance_full_grains exp = 99 / 4 :=
sorry

end NUMINAMATH_CALUDE_soybean_experiment_results_l1383_138344


namespace NUMINAMATH_CALUDE_range_of_t_for_true_proposition_l1383_138304

theorem range_of_t_for_true_proposition (t : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_for_true_proposition_l1383_138304


namespace NUMINAMATH_CALUDE_cube_volume_edge_relation_l1383_138379

theorem cube_volume_edge_relation (a : ℝ) (a' : ℝ) (ha : a > 0) (ha' : a' > 0) :
  (a' ^ 3) = 27 * (a ^ 3) → a' = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_edge_relation_l1383_138379


namespace NUMINAMATH_CALUDE_typist_salary_problem_l1383_138351

theorem typist_salary_problem (S : ℝ) : 
  S * 1.1 * 0.95 = 3135 → S = 3000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l1383_138351


namespace NUMINAMATH_CALUDE_parallel_lines_angle_measure_l1383_138336

-- Define the angle measures as real numbers
variable (angle1 angle2 angle5 : ℝ)

-- State the theorem
theorem parallel_lines_angle_measure :
  -- Conditions
  angle1 = (1 / 4) * angle2 →  -- ∠1 is 1/4 of ∠2
  angle1 = angle5 →            -- ∠1 and ∠5 are alternate angles (implied by parallel lines)
  angle2 + angle5 = 180 →      -- ∠2 and ∠5 form a straight line
  -- Conclusion
  angle5 = 36 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_measure_l1383_138336


namespace NUMINAMATH_CALUDE_second_recipe_amount_is_one_l1383_138332

/-- The amount of lower sodium soy sauce in the second recipe -/
def second_recipe_amount : ℚ :=
  let bottle_ounces : ℚ := 16
  let ounces_per_cup : ℚ := 8
  let first_recipe_cups : ℚ := 2
  let third_recipe_cups : ℚ := 3
  let total_bottles : ℚ := 3
  let total_ounces : ℚ := total_bottles * bottle_ounces
  let total_cups : ℚ := total_ounces / ounces_per_cup
  total_cups - first_recipe_cups - third_recipe_cups

theorem second_recipe_amount_is_one :
  second_recipe_amount = 1 := by sorry

end NUMINAMATH_CALUDE_second_recipe_amount_is_one_l1383_138332


namespace NUMINAMATH_CALUDE_lottery_distribution_l1383_138308

theorem lottery_distribution (lottery_win : ℝ) (num_students : ℕ) : 
  lottery_win = 155250 →
  num_students = 100 →
  (lottery_win / 1000) * num_students = 15525 := by
  sorry

end NUMINAMATH_CALUDE_lottery_distribution_l1383_138308


namespace NUMINAMATH_CALUDE_weight_of_8_moles_AlI3_l1383_138360

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of Aluminum atoms in AlI3 -/
def num_Al_atoms : ℕ := 1

/-- The number of Iodine atoms in AlI3 -/
def num_I_atoms : ℕ := 3

/-- The number of moles of AlI3 -/
def num_moles_AlI3 : ℝ := 8

/-- The molecular weight of AlI3 in g/mol -/
def molecular_weight_AlI3 : ℝ := 
  num_Al_atoms * atomic_weight_Al + num_I_atoms * atomic_weight_I

/-- The weight of a given number of moles of AlI3 in grams -/
def weight_AlI3 (moles : ℝ) : ℝ := moles * molecular_weight_AlI3

theorem weight_of_8_moles_AlI3 : 
  weight_AlI3 num_moles_AlI3 = 3261.44 := by
  sorry


end NUMINAMATH_CALUDE_weight_of_8_moles_AlI3_l1383_138360


namespace NUMINAMATH_CALUDE_all_fruits_fallen_by_day_12_l1383_138352

/-- Represents the number of fruits that fall on a given day -/
def fruits_falling (day : ℕ) : ℕ :=
  if day ≤ 10 then day
  else (day - 10)

/-- Represents the total number of fruits that have fallen up to a given day -/
def total_fruits_fallen (day : ℕ) : ℕ :=
  if day ≤ 10 then day * (day + 1) / 2
  else 55 + (day - 10) * (day - 9) / 2

/-- The theorem stating that all fruits will have fallen by the end of the 12th day -/
theorem all_fruits_fallen_by_day_12 :
  total_fruits_fallen 12 = 58 ∧
  ∀ d : ℕ, d < 12 → total_fruits_fallen d < 58 := by
  sorry


end NUMINAMATH_CALUDE_all_fruits_fallen_by_day_12_l1383_138352


namespace NUMINAMATH_CALUDE_count_multiples_eq_42_l1383_138399

/-- The number of positive integers less than 201 that are multiples of either 6 or 8, but not both -/
def count_multiples : ℕ :=
  (Finset.filter (fun n => n % 6 = 0 ∨ n % 8 = 0) (Finset.range 201)).card -
  (Finset.filter (fun n => n % 6 = 0 ∧ n % 8 = 0) (Finset.range 201)).card

theorem count_multiples_eq_42 : count_multiples = 42 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_eq_42_l1383_138399


namespace NUMINAMATH_CALUDE_angle_measure_in_regular_octagon_l1383_138335

/-- A regular octagon is a polygon with 8 sides of equal length and 8 angles of equal measure. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Given three vertices of a regular octagon with one vertex between each pair -/
def skip_one_vertex (o : RegularOctagon) (i j k : Fin 8) : Prop :=
  (j - i) % 8 = 2 ∧ (k - j) % 8 = 2

/-- The angle between three points in a plane -/
def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_in_regular_octagon (o : RegularOctagon) (i j k : Fin 8) 
  (h : skip_one_vertex o i j k) : 
  angle_measure (o.vertices i) (o.vertices j) (o.vertices k) = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_regular_octagon_l1383_138335


namespace NUMINAMATH_CALUDE_largest_multiple_13_negation_gt_neg150_l1383_138341

theorem largest_multiple_13_negation_gt_neg150 : 
  ∀ n : ℤ, n * 13 > 143 → -(n * 13) ≤ -150 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_13_negation_gt_neg150_l1383_138341


namespace NUMINAMATH_CALUDE_carl_has_more_stamps_l1383_138356

/-- Given that Carl has 89 stamps and Kevin has 57 stamps, 
    prove that Carl has 32 more stamps than Kevin. -/
theorem carl_has_more_stamps (carl_stamps : ℕ) (kevin_stamps : ℕ) 
  (h1 : carl_stamps = 89) (h2 : kevin_stamps = 57) : 
  carl_stamps - kevin_stamps = 32 := by
  sorry

end NUMINAMATH_CALUDE_carl_has_more_stamps_l1383_138356


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1383_138312

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 4 → (x > 3 ∨ x < -1)) ∧ 
  (∃ x : ℝ, (x > 3 ∨ x < -1) ∧ ¬(x > 4)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1383_138312


namespace NUMINAMATH_CALUDE_min_value_theorem_l1383_138393

theorem min_value_theorem (a₁ a₂ : ℝ) 
  (h : (3 / (3 + 2 * Real.sin a₁)) + (2 / (4 - Real.sin (2 * a₂))) = 1) :
  ∃ (m : ℝ), m = π / 4 ∧ ∀ (x : ℝ), |4 * π - a₁ + a₂| ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1383_138393


namespace NUMINAMATH_CALUDE_projection_theorem_l1383_138321

def v : Fin 3 → ℝ := ![1, 2, 3]
def proj_v : Fin 3 → ℝ := ![2, 4, 6]
def u : Fin 3 → ℝ := ![2, 1, -1]

theorem projection_theorem (w : Fin 3 → ℝ) 
  (hw : ∃ (k : ℝ), w = fun i => k * proj_v i) :
  let proj_u := (u • w) / (w • w) • w
  proj_u = fun i => (![1/14, 1/7, 3/14] : Fin 3 → ℝ) i := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_projection_theorem_l1383_138321


namespace NUMINAMATH_CALUDE_odd_product_pattern_l1383_138363

theorem odd_product_pattern (n : ℕ) (h : Odd n) : n * (n + 2) = (n + 1)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_product_pattern_l1383_138363


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l1383_138392

/-- A line passing through two points (2, 9) and (5, 17) intersects the y-axis at (0, 11/3) -/
theorem line_intersection_y_axis : 
  ∃ (m b : ℚ), 
    (9 = m * 2 + b) ∧ 
    (17 = m * 5 + b) ∧ 
    (11/3 = b) := by sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l1383_138392


namespace NUMINAMATH_CALUDE_cone_surface_area_l1383_138350

/-- The surface area of a cone given its slant height and lateral surface central angle -/
theorem cone_surface_area (s : ℝ) (θ : ℝ) (h_s : s = 3) (h_θ : θ = 2 * Real.pi / 3) :
  s * θ / 2 + Real.pi * (s * θ / (2 * Real.pi))^2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1383_138350


namespace NUMINAMATH_CALUDE_lucky_325th_number_l1383_138338

/-- A positive integer is "lucky" if the sum of its digits is 7. -/
def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ (Nat.digits 10 n).sum = 7

/-- The sequence of "lucky" numbers in ascending order. -/
def lucky_seq : ℕ → ℕ := sorry

theorem lucky_325th_number : lucky_seq 325 = 52000 := by sorry

end NUMINAMATH_CALUDE_lucky_325th_number_l1383_138338


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l1383_138325

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l1383_138325


namespace NUMINAMATH_CALUDE_calculation_result_l1383_138355

-- Define the numbers in their respective bases
def num_base_8 : ℚ := 2 * 8^3 + 4 * 8^2 + 6 * 8^1 + 8 * 8^0
def num_base_5 : ℚ := 1 * 5^2 + 2 * 5^1 + 1 * 5^0
def num_base_9 : ℚ := 1 * 9^3 + 3 * 9^2 + 5 * 9^1 + 7 * 9^0
def num_base_10 : ℚ := 2048

-- Define the result of the calculation
def result : ℚ := num_base_8 / num_base_5 - num_base_9 + num_base_10

-- State the theorem
theorem calculation_result : result = 1061.1111 := by sorry

end NUMINAMATH_CALUDE_calculation_result_l1383_138355


namespace NUMINAMATH_CALUDE_multiples_of_15_between_25_and_225_l1383_138343

theorem multiples_of_15_between_25_and_225 : 
  (Finset.range 226 
    |>.filter (fun n => n ≥ 25 ∧ n % 15 = 0)
    |>.card) = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_25_and_225_l1383_138343


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1383_138323

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c →
  a * b * c = 399 →
  2 * (a * b + b * c + c * a) = 422 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1383_138323


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l1383_138303

/-- Given a cubic function f(x) = ax³ + bx + 5 where f(-9) = -7, prove that f(9) = 17 -/
theorem cubic_function_symmetry (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 5)
  (h2 : f (-9) = -7) : 
  f 9 = 17 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l1383_138303


namespace NUMINAMATH_CALUDE_work_completion_proof_l1383_138322

/-- The original number of men working on a project -/
def original_men : ℕ := 48

/-- The number of days it takes the original group to complete the work -/
def original_days : ℕ := 60

/-- The number of additional men added to the group -/
def additional_men : ℕ := 8

/-- The number of days it takes the larger group to complete the work -/
def new_days : ℕ := 50

/-- The amount of work to be completed -/
def work : ℝ := 1

theorem work_completion_proof :
  (original_men : ℝ) * work / original_days = 
  ((original_men + additional_men) : ℝ) * work / new_days :=
by sorry

#check work_completion_proof

end NUMINAMATH_CALUDE_work_completion_proof_l1383_138322


namespace NUMINAMATH_CALUDE_edwin_alvin_age_difference_l1383_138319

/-- Represents the age difference between Edwin and Alvin -/
def ageDifference (edwinAge alvinAge : ℝ) : ℝ := edwinAge - alvinAge

/-- Theorem stating the age difference between Edwin and Alvin -/
theorem edwin_alvin_age_difference :
  ∃ (edwinAge alvinAge : ℝ),
    edwinAge > alvinAge ∧
    edwinAge + 2 = (1/3) * (alvinAge + 2) + 20 ∧
    edwinAge + alvinAge = 30.99999999 ∧
    ageDifference edwinAge alvinAge = 12 := by sorry

end NUMINAMATH_CALUDE_edwin_alvin_age_difference_l1383_138319


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1383_138313

theorem polynomial_remainder (x : ℝ) : (x^14 + 1) % (x + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1383_138313


namespace NUMINAMATH_CALUDE_oil_barrel_difference_l1383_138300

theorem oil_barrel_difference :
  ∀ (a b : ℝ),
  a + b = 100 →
  (a + 15) = 4 * (b - 15) →
  a - b = 30 := by
sorry

end NUMINAMATH_CALUDE_oil_barrel_difference_l1383_138300


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1383_138396

theorem quadratic_root_range (a : ℝ) (α β : ℝ) : 
  (∃ x, x^2 - 2*a*x + a + 2 = 0) ∧ 
  (α^2 - 2*a*α + a + 2 = 0) ∧ 
  (β^2 - 2*a*β + a + 2 = 0) ∧ 
  (1 < α) ∧ (α < 2) ∧ (2 < β) ∧ (β < 3) →
  (2 < a) ∧ (a < 11/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1383_138396


namespace NUMINAMATH_CALUDE_fraction_addition_simplification_l1383_138368

theorem fraction_addition_simplification : 7/8 + 3/5 = 59/40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_simplification_l1383_138368


namespace NUMINAMATH_CALUDE_even_function_sum_l1383_138315

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l1383_138315


namespace NUMINAMATH_CALUDE_calculator_sales_loss_l1383_138389

theorem calculator_sales_loss (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  price = 135 ∧ profit_percent = 25 ∧ loss_percent = 25 →
  ∃ (cost1 cost2 : ℝ),
    cost1 + (profit_percent / 100) * cost1 = price ∧
    cost2 - (loss_percent / 100) * cost2 = price ∧
    2 * price - (cost1 + cost2) = -18 :=
by sorry

end NUMINAMATH_CALUDE_calculator_sales_loss_l1383_138389


namespace NUMINAMATH_CALUDE_race_has_six_laps_l1383_138398

/-- Represents a cyclist in the race -/
structure Cyclist where
  name : String
  lap_time : ℕ

/-- Represents the race setup -/
structure Race where
  total_laps : ℕ
  vasya : Cyclist
  petya : Cyclist
  kolya : Cyclist

/-- The race conditions are satisfied -/
def race_conditions (r : Race) : Prop :=
  r.vasya.lap_time + 2 = r.petya.lap_time ∧
  r.petya.lap_time + 3 = r.kolya.lap_time ∧
  r.vasya.lap_time * r.total_laps = r.petya.lap_time * (r.total_laps - 1) ∧
  r.vasya.lap_time * r.total_laps = r.kolya.lap_time * (r.total_laps - 2)

/-- The theorem stating that the race has 6 laps -/
theorem race_has_six_laps :
  ∃ (r : Race), race_conditions r ∧ r.total_laps = 6 := by sorry

end NUMINAMATH_CALUDE_race_has_six_laps_l1383_138398


namespace NUMINAMATH_CALUDE_bobs_remaining_funds_l1383_138378

/-- Converts a number from octal to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := sorry

/-- Calculates the remaining funds after expenses --/
def remaining_funds (savings : ℕ) (ticket_cost : ℕ) (meal_cost : ℕ) : ℕ :=
  savings - (ticket_cost + meal_cost)

theorem bobs_remaining_funds :
  let bobs_savings : ℕ := octal_to_decimal 7777
  let ticket_cost : ℕ := 1500
  let meal_cost : ℕ := 250
  remaining_funds bobs_savings ticket_cost meal_cost = 2345 := by sorry

end NUMINAMATH_CALUDE_bobs_remaining_funds_l1383_138378


namespace NUMINAMATH_CALUDE_smallest_x_for_540x_perfect_square_l1383_138340

theorem smallest_x_for_540x_perfect_square :
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), ∃ (M : ℤ), 540 * y = M^2 → x ≤ y) ∧
    (∃ (M : ℤ), 540 * x = M^2) ∧
    x = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_540x_perfect_square_l1383_138340


namespace NUMINAMATH_CALUDE_pencil_total_length_l1383_138382

/-- The total length of a pencil with colored sections -/
def pencil_length (purple_length black_length blue_length : ℝ) : ℝ :=
  purple_length + black_length + blue_length

/-- Theorem: The total length of a pencil with specific colored sections is 4 cm -/
theorem pencil_total_length :
  pencil_length 1.5 0.5 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_total_length_l1383_138382


namespace NUMINAMATH_CALUDE_fraction_equality_l1383_138333

theorem fraction_equality (a b : ℝ) (h : b / a = 3 / 5) : (a - b) / a = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1383_138333


namespace NUMINAMATH_CALUDE_tom_candy_pieces_l1383_138377

theorem tom_candy_pieces (initial_boxes : ℕ) (given_away : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → given_away = 8 → pieces_per_box = 3 →
  (initial_boxes - given_away) * pieces_per_box = 18 := by
sorry

end NUMINAMATH_CALUDE_tom_candy_pieces_l1383_138377


namespace NUMINAMATH_CALUDE_harrison_croissant_price_l1383_138381

/-- The price of a regular croissant that Harrison buys -/
def regular_croissant_price : ℝ := 3.50

/-- The price of an almond croissant that Harrison buys -/
def almond_croissant_price : ℝ := 5.50

/-- The total amount Harrison spends on croissants in a year -/
def total_spent : ℝ := 468

/-- The number of weeks in a year -/
def weeks_in_year : ℕ := 52

theorem harrison_croissant_price :
  regular_croissant_price * weeks_in_year + almond_croissant_price * weeks_in_year = total_spent :=
by sorry

end NUMINAMATH_CALUDE_harrison_croissant_price_l1383_138381


namespace NUMINAMATH_CALUDE_rent_split_l1383_138361

theorem rent_split (total_rent : ℕ) (num_people : ℕ) (individual_rent : ℕ) :
  total_rent = 490 →
  num_people = 7 →
  individual_rent = total_rent / num_people →
  individual_rent = 70 := by
  sorry

end NUMINAMATH_CALUDE_rent_split_l1383_138361


namespace NUMINAMATH_CALUDE_candy_probability_contradiction_l1383_138380

theorem candy_probability_contradiction :
  ∀ (packet1_blue packet1_total packet2_blue packet2_total : ℕ),
    packet1_blue ≤ packet1_total →
    packet2_blue ≤ packet2_total →
    (3 : ℚ) / 8 ≤ (packet1_blue + packet2_blue : ℚ) / (packet1_total + packet2_total) →
    (packet1_blue + packet2_blue : ℚ) / (packet1_total + packet2_total) ≤ 2 / 5 →
    ¬((17 : ℚ) / 40 ≥ 3 / 8 ∧ 17 / 40 ≤ 2 / 5) :=
by
  sorry

end NUMINAMATH_CALUDE_candy_probability_contradiction_l1383_138380


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1383_138337

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 15

-- State the theorem
theorem quadratic_minimum :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min) ∧
  (∃ (y_min : ℝ), y_min = f 2 ∧ y_min = 7) ∧
  (∀ (x : ℝ), f x ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1383_138337


namespace NUMINAMATH_CALUDE_crayon_selection_ways_l1383_138328

/-- The number of ways to choose k items from n items, where order doesn't matter -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons to be selected -/
def selected_crayons : ℕ := 5

theorem crayon_selection_ways : 
  choose total_crayons selected_crayons = 3003 := by sorry

end NUMINAMATH_CALUDE_crayon_selection_ways_l1383_138328


namespace NUMINAMATH_CALUDE_sector_area_l1383_138366

theorem sector_area (arc_length : Real) (central_angle : Real) :
  arc_length = π ∧ central_angle = π / 4 →
  let radius := arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1383_138366


namespace NUMINAMATH_CALUDE_volunteer_assignment_l1383_138375

-- Define the number of volunteers
def num_volunteers : ℕ := 5

-- Define the number of venues
def num_venues : ℕ := 3

-- Define the function to calculate the number of ways to assign volunteers
def ways_to_assign (volunteers : ℕ) (venues : ℕ) : ℕ :=
  venues^volunteers - venues * (venues - 1)^volunteers

-- Theorem statement
theorem volunteer_assignment :
  ways_to_assign num_volunteers num_venues = 147 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_assignment_l1383_138375


namespace NUMINAMATH_CALUDE_floor_product_eq_square_l1383_138305

def floor (x : ℚ) : ℤ := Int.floor x

theorem floor_product_eq_square (x : ℤ) : 
  (floor (x / 2 : ℚ)) * (floor (x / 3 : ℚ)) * (floor (x / 4 : ℚ)) = x^2 ↔ x = 0 ∨ x = 24 :=
by sorry

end NUMINAMATH_CALUDE_floor_product_eq_square_l1383_138305


namespace NUMINAMATH_CALUDE_binomial_mode_maximizes_pmf_l1383_138342

/-- The number of trials -/
def n : ℕ := 5

/-- The probability of success -/
def p : ℚ := 3/4

/-- The binomial probability mass function -/
def binomialPMF (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- The mode of the binomial distribution -/
def binomialMode : ℕ := 4

/-- Theorem stating that the binomial mode maximizes the probability mass function -/
theorem binomial_mode_maximizes_pmf :
  ∀ k : ℕ, k ≤ n → binomialPMF binomialMode ≥ binomialPMF k :=
sorry

end NUMINAMATH_CALUDE_binomial_mode_maximizes_pmf_l1383_138342


namespace NUMINAMATH_CALUDE_milk_production_l1383_138347

theorem milk_production (M : ℝ) 
  (h1 : M > 0) 
  (h2 : M * 0.25 * 0.5 = 2) : M = 16 := by
  sorry

end NUMINAMATH_CALUDE_milk_production_l1383_138347


namespace NUMINAMATH_CALUDE_students_yes_R_is_400_l1383_138385

/-- Given information about student responses to subjects M and R -/
structure StudentResponses where
  total : Nat
  yes_only_M : Nat
  no_both : Nat

/-- Calculate the number of students who answered yes for subject R -/
def students_yes_R (responses : StudentResponses) : Nat :=
  responses.total - responses.yes_only_M - responses.no_both

/-- Theorem stating that the number of students who answered yes for R is 400 -/
theorem students_yes_R_is_400 (responses : StudentResponses)
  (h1 : responses.total = 800)
  (h2 : responses.yes_only_M = 170)
  (h3 : responses.no_both = 230) :
  students_yes_R responses = 400 := by
  sorry

#eval students_yes_R ⟨800, 170, 230⟩

end NUMINAMATH_CALUDE_students_yes_R_is_400_l1383_138385


namespace NUMINAMATH_CALUDE_g_range_l1383_138395

noncomputable def g (x : ℝ) : ℝ :=
  (Real.sin x ^ 3 + 4 * Real.sin x ^ 2 - 3 * Real.sin x + 3 * Real.cos x ^ 2 - 9) / (Real.sin x - 1)

theorem g_range :
  Set.range (fun x : ℝ => g x) = Set.Icc 5 9 \ {9} :=
by
  sorry

end NUMINAMATH_CALUDE_g_range_l1383_138395


namespace NUMINAMATH_CALUDE_min_non_red_surface_fraction_for_specific_cube_l1383_138318

/-- Represents a cube with given edge length and colored subcubes -/
structure ColoredCube where
  edge_length : ℕ
  red_cubes : ℕ
  white_cubes : ℕ
  blue_cubes : ℕ

/-- Calculate the minimum non-red surface area fraction of a ColoredCube -/
def min_non_red_surface_fraction (c : ColoredCube) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem min_non_red_surface_fraction_for_specific_cube :
  let c := ColoredCube.mk 4 48 12 4
  min_non_red_surface_fraction c = 1/8 := by sorry

end NUMINAMATH_CALUDE_min_non_red_surface_fraction_for_specific_cube_l1383_138318


namespace NUMINAMATH_CALUDE_x_plus_y_eq_1_is_linear_l1383_138310

/-- A linear equation with two variables is of the form ax + by = c, where a, b, and c are constants. -/
def IsLinearEquationTwoVars (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

/-- The function representing x + y = 1 -/
def f (x y : ℝ) : ℝ := x + y - 1

/-- Theorem stating that x + y = 1 is a linear equation with two variables -/
theorem x_plus_y_eq_1_is_linear : IsLinearEquationTwoVars f := by
  sorry


end NUMINAMATH_CALUDE_x_plus_y_eq_1_is_linear_l1383_138310


namespace NUMINAMATH_CALUDE_flowers_in_vase_l1383_138391

theorem flowers_in_vase (roses : ℕ) (carnations : ℕ) : 
  roses = 5 → carnations = 5 → roses + carnations = 10 := by
  sorry

end NUMINAMATH_CALUDE_flowers_in_vase_l1383_138391


namespace NUMINAMATH_CALUDE_part_one_part_two_l1383_138327

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : ¬¬p m) : m > 2 := by
  sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m ≥ 3 ∨ (1 < m ∧ m ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1383_138327


namespace NUMINAMATH_CALUDE_octagon_pyramid_volume_l1383_138371

/-- A right pyramid with a regular octagon base and one equilateral triangular face --/
structure OctagonPyramid where
  /-- Side length of the equilateral triangular face --/
  side_length : ℝ
  /-- The base is a regular octagon --/
  is_regular_octagon : Bool
  /-- The pyramid is a right pyramid --/
  is_right_pyramid : Bool
  /-- One face is an equilateral triangle --/
  has_equilateral_face : Bool

/-- Calculate the volume of the octagon pyramid --/
noncomputable def volume (p : OctagonPyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific octagon pyramid --/
theorem octagon_pyramid_volume :
  ∀ (p : OctagonPyramid),
    p.side_length = 10 ∧
    p.is_regular_octagon ∧
    p.is_right_pyramid ∧
    p.has_equilateral_face →
    volume p = 1000 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_pyramid_volume_l1383_138371


namespace NUMINAMATH_CALUDE_damage_ratio_is_five_fourths_l1383_138358

/-- The ratio of damages for Winnie-the-Pooh's two falls -/
theorem damage_ratio_is_five_fourths
  (g : ℝ) (H : ℝ) (n : ℝ) (k : ℝ) (M : ℝ) (τ : ℝ)
  (h_pos : 0 < H)
  (n_pos : 0 < n)
  (k_pos : 0 < k)
  (g_pos : 0 < g)
  (h_def : H = n * (H / n)) :
  let h := H / n
  let V_I := Real.sqrt (2 * g * H)
  let V_1 := Real.sqrt (2 * g * h)
  let V_1' := (1 / k) * Real.sqrt (2 * g * h)
  let V_II := Real.sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h))
  let I_I := M * V_I * τ
  let I_II := M * τ * ((V_1 - V_1') + V_II)
  I_II / I_I = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_damage_ratio_is_five_fourths_l1383_138358


namespace NUMINAMATH_CALUDE_sequence_range_l1383_138317

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem sequence_range (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, 2 * a (n + 1) * a n + a (n + 1) - 3 * a n = 0)
  (h2 : a 1 > 0)
  (h3 : is_increasing a) :
  0 < a 1 ∧ a 1 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l1383_138317


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1383_138376

theorem perfect_square_quadratic (c : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 14*x + c = y^2) → c = 49 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1383_138376


namespace NUMINAMATH_CALUDE_correct_ball_placement_count_l1383_138394

/-- The number of ways to place four distinct balls into three boxes, leaving exactly one box empty -/
def ball_placement_count : ℕ := 42

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of boxes -/
def num_boxes : ℕ := 3

/-- Theorem stating that the number of ways to place the balls is correct -/
theorem correct_ball_placement_count :
  (∃ (f : Fin num_balls → Fin num_boxes),
    (∀ i j, i ≠ j → f i ≠ f j) ∧
    (∃ (empty_box : Fin num_boxes), ∀ i, f i ≠ empty_box) ∧
    (∀ box : Fin num_boxes, box ≠ empty_box → ∃ i, f i = box)) →
  ball_placement_count = 42 :=
by sorry

end NUMINAMATH_CALUDE_correct_ball_placement_count_l1383_138394


namespace NUMINAMATH_CALUDE_average_of_arithmetic_sequence_l1383_138302

theorem average_of_arithmetic_sequence (z : ℝ) : 
  let seq := [5, 5 + 3*z, 5 + 6*z, 5 + 9*z, 5 + 12*z]
  (seq.sum / seq.length : ℝ) = 5 + 6*z := by sorry

end NUMINAMATH_CALUDE_average_of_arithmetic_sequence_l1383_138302


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1383_138334

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 4 = 0 ∧ m % 9 = 0 → m ≥ n) ∧
  n = 108 := by
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1383_138334


namespace NUMINAMATH_CALUDE_optimal_solution_l1383_138369

/-- Represents the factory worker allocation problem -/
structure FactoryProblem where
  total_workers : ℕ
  salary_a : ℕ
  salary_b : ℕ
  job_b_constraint : ℕ → Prop

/-- The specific factory problem instance -/
def factory_instance : FactoryProblem where
  total_workers := 120
  salary_a := 800
  salary_b := 1000
  job_b_constraint := fun x => (120 - x) ≥ 3 * x

/-- Calculate the total monthly salary -/
def total_salary (p : FactoryProblem) (workers_a : ℕ) : ℕ :=
  p.salary_a * workers_a + p.salary_b * (p.total_workers - workers_a)

/-- Theorem stating the optimal solution -/
theorem optimal_solution (p : FactoryProblem) :
  p = factory_instance →
  (∀ x : ℕ, x ≤ p.total_workers → p.job_b_constraint x → total_salary p x ≥ 114000) ∧
  total_salary p 30 = 114000 ∧
  p.job_b_constraint 30 := by
  sorry

end NUMINAMATH_CALUDE_optimal_solution_l1383_138369


namespace NUMINAMATH_CALUDE_center_in_triangle_probability_l1383_138387

theorem center_in_triangle_probability (n : ℕ) (hn : n > 0) :
  let sides := 2 * n + 1
  (n + 1 : ℚ) / (4 * n - 2) =
    1 - (sides * (n.choose 2) : ℚ) / (sides.choose 3) :=
by sorry

end NUMINAMATH_CALUDE_center_in_triangle_probability_l1383_138387


namespace NUMINAMATH_CALUDE_gummy_bears_count_l1383_138354

/-- The number of gummy bears produced per minute -/
def production_rate : ℕ := 300

/-- The time taken to produce enough gummy bears to fill the packets (in minutes) -/
def production_time : ℕ := 40

/-- The number of packets filled with the gummy bears produced -/
def num_packets : ℕ := 240

/-- The number of gummy bears in each packet -/
def gummy_bears_per_packet : ℕ := production_rate * production_time / num_packets

theorem gummy_bears_count : gummy_bears_per_packet = 50 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bears_count_l1383_138354


namespace NUMINAMATH_CALUDE_phi_difference_bound_l1383_138384

/-- The n-th iterate of a function -/
def iterate (f : ℝ → ℝ) (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

/-- The main theorem -/
theorem phi_difference_bound
  (f : ℝ → ℝ)
  (h_mono : ∀ x y, x ≤ y → f x ≤ f y)
  (h_period : ∀ x, f (x + 1) = f x + 1)
  (n : ℕ)
  (φ : ℝ → ℝ)
  (h_phi : ∀ x, φ x = iterate f n x - x) :
  ∀ x y, |φ x - φ y| < 1 :=
sorry

end NUMINAMATH_CALUDE_phi_difference_bound_l1383_138384


namespace NUMINAMATH_CALUDE_p_range_nonnegative_reals_l1383_138348

/-- The function p(x) = x^4 - 6x^2 + 9 -/
def p (x : ℝ) : ℝ := x^4 - 6*x^2 + 9

theorem p_range_nonnegative_reals :
  Set.range (fun (x : ℝ) ↦ p x) = Set.Ici (0 : ℝ) := by sorry

end NUMINAMATH_CALUDE_p_range_nonnegative_reals_l1383_138348


namespace NUMINAMATH_CALUDE_max_product_sum_l1383_138388

def values : Finset ℕ := {1, 3, 5, 7}

theorem max_product_sum (a b c d : ℕ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_in_values : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values) :
  (a * b + b * c + c * d + d * a) ≤ 64 :=
sorry

end NUMINAMATH_CALUDE_max_product_sum_l1383_138388


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l1383_138329

theorem hyperbola_k_range (k : ℝ) :
  (∃ x y : ℝ, x^2 / (1 + k) - y^2 / (1 - k) = 1) →
  -1 < k ∧ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l1383_138329


namespace NUMINAMATH_CALUDE_backpack_and_weight_difference_l1383_138365

/-- Given the weights of Bridget and Martha, and their combined weight with a backpack,
    prove the weight of the backpack and the weight difference between Bridget and Martha. -/
theorem backpack_and_weight_difference 
  (bridget_weight : ℕ) 
  (martha_weight : ℕ) 
  (combined_weight_with_backpack : ℕ) 
  (h1 : bridget_weight = 39)
  (h2 : martha_weight = 2)
  (h3 : combined_weight_with_backpack = 60) :
  (∃ backpack_weight : ℕ, 
    backpack_weight = combined_weight_with_backpack - (bridget_weight + martha_weight) ∧ 
    backpack_weight = 19) ∧ 
  (bridget_weight - martha_weight = 37) := by
  sorry

end NUMINAMATH_CALUDE_backpack_and_weight_difference_l1383_138365


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1383_138364

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = 5) 
  (h2 : x * y = -3) : 
  x^2 * y - x * y^2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1383_138364


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1383_138320

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - 25
  ∃ x₁ x₂ : ℝ, x₁ = 5 ∧ x₂ = -5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1383_138320
