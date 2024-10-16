import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3500_350033

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: Given an arithmetic sequence with S_3 = 6 and a_4 = 8, the common difference is 3 -/
theorem arithmetic_sequence_difference (seq : ArithmeticSequence) 
  (h1 : seq.sum 3 = 6) (h2 : seq.a 4 = 8) : seq.d = 3 := by
  sorry

#check arithmetic_sequence_difference

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l3500_350033


namespace NUMINAMATH_CALUDE_solution_sum_l3500_350048

theorem solution_sum (a b : ℚ) : 
  (2 * a + b = 14 ∧ a + 2 * b = 21) → a + b = 35 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l3500_350048


namespace NUMINAMATH_CALUDE_external_tangent_y_intercept_l3500_350010

/-- The y-intercept of the common external tangent line with positive slope to two circles -/
theorem external_tangent_y_intercept 
  (center1 : ℝ × ℝ) (radius1 : ℝ) (center2 : ℝ × ℝ) (radius2 : ℝ) 
  (h1 : center1 = (1, 5)) 
  (h2 : radius1 = 3) 
  (h3 : center2 = (15, 10)) 
  (h4 : radius2 = 10) : 
  ∃ (m b : ℝ), m > 0 ∧ 
    (∀ (x y : ℝ), y = m * x + b → 
      ((x - center1.1)^2 + (y - center1.2)^2 = radius1^2 ∨ 
       (x - center2.1)^2 + (y - center2.2)^2 = radius2^2)) ∧ 
    b = 7416 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_external_tangent_y_intercept_l3500_350010


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l3500_350014

theorem smallest_sum_of_squares (x y : ℕ) : 
  x^2 - y^2 = 187 → ∃ (a b : ℕ), a^2 - b^2 = 187 ∧ a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 205 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l3500_350014


namespace NUMINAMATH_CALUDE_dorothy_doughnuts_l3500_350004

/-- Represents the problem of calculating the number of doughnuts Dorothy made. -/
theorem dorothy_doughnuts (ingredient_cost : ℕ) (selling_price : ℕ) (profit : ℕ) 
  (h1 : ingredient_cost = 53)
  (h2 : selling_price = 3)
  (h3 : profit = 22) :
  ∃ (num_doughnuts : ℕ), 
    selling_price * num_doughnuts = ingredient_cost + profit ∧ 
    num_doughnuts = 25 := by
  sorry


end NUMINAMATH_CALUDE_dorothy_doughnuts_l3500_350004


namespace NUMINAMATH_CALUDE_g_seven_value_l3500_350095

theorem g_seven_value (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x + y) = g x + g y) 
  (h2 : g 6 = 7) : 
  g 7 = 49 / 6 := by
sorry

end NUMINAMATH_CALUDE_g_seven_value_l3500_350095


namespace NUMINAMATH_CALUDE_annes_wandering_time_l3500_350028

/-- Proves that Anne's wandering time is 1.5 hours given her distance and speed -/
theorem annes_wandering_time
  (distance : ℝ) (speed : ℝ)
  (h1 : distance = 3.0)
  (h2 : speed = 2.0)
  : distance / speed = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_annes_wandering_time_l3500_350028


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3500_350067

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ m ∈ Set.Iio (-4) ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3500_350067


namespace NUMINAMATH_CALUDE_distance_between_specific_planes_l3500_350062

/-- The distance between two planes given by their equations -/
def distance_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  sorry

/-- Theorem: The distance between the planes x - 2y + 2z = 9 and 2x - 4y + 4z = 18 is 0 -/
theorem distance_between_specific_planes :
  distance_between_planes 1 (-2) 2 9 2 (-4) 4 18 = 0 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_specific_planes_l3500_350062


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l3500_350058

/-- Proves that a train of given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 250) 
  (h2 : train_speed_kmph = 72) 
  (h3 : bridge_length = 1250) : 
  (train_length + bridge_length) / (train_speed_kmph * 1000 / 3600) = 75 := by
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l3500_350058


namespace NUMINAMATH_CALUDE_counterexample_exists_l3500_350071

theorem counterexample_exists : ∃ n : ℕ, 
  (¬ Nat.Prime n) ∧ (¬ Nat.Prime (n - 1) ∨ Nat.Prime (n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3500_350071


namespace NUMINAMATH_CALUDE_sum_first_last_is_14_l3500_350088

/-- A sequence of seven terms satisfying specific conditions -/
structure SevenTermSequence where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ
  U : ℝ
  V : ℝ
  R_eq_7 : R = 7
  sum_consecutive_3 : ∀ (x y z : ℝ), (x = P ∧ y = Q ∧ z = R) ∨
                                     (x = Q ∧ y = R ∧ z = S) ∨
                                     (x = R ∧ y = S ∧ z = T) ∨
                                     (x = S ∧ y = T ∧ z = U) ∨
                                     (x = T ∧ y = U ∧ z = V) →
                                     x + y + z = 21

/-- The sum of the first and last terms in a seven-term sequence is 14 -/
theorem sum_first_last_is_14 (seq : SevenTermSequence) : seq.P + seq.V = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_last_is_14_l3500_350088


namespace NUMINAMATH_CALUDE_seven_factorial_mod_thirteen_l3500_350037

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem seven_factorial_mod_thirteen : factorial 7 % 13 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_factorial_mod_thirteen_l3500_350037


namespace NUMINAMATH_CALUDE_number_equation_solution_l3500_350020

theorem number_equation_solution : ∃ x : ℝ, x - 3 / (1/3) + 3 = 3 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3500_350020


namespace NUMINAMATH_CALUDE_first_machine_copies_per_minute_l3500_350094

/-- Given two copy machines working together, prove that the first machine makes 25 copies per minute. -/
theorem first_machine_copies_per_minute :
  ∀ (x : ℝ),
  (∃ (rate₁ : ℝ), rate₁ = x) →  -- First machine works at a constant rate x
  (∃ (rate₂ : ℝ), rate₂ = 55) →  -- Second machine works at 55 copies per minute
  (x + 55) * 30 = 2400 →  -- Together they make 2400 copies in 30 minutes
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_first_machine_copies_per_minute_l3500_350094


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_l3500_350027

theorem least_positive_integer_multiple (x : ℕ) : x = 16 ↔ 
  (x > 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → ¬(∃ k : ℤ, (3*y)^2 + 2*58*3*y + 58^2 = 53*k) ∧
   ∃ k : ℤ, (3*x)^2 + 2*58*3*x + 58^2 = 53*k) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_l3500_350027


namespace NUMINAMATH_CALUDE_rectangle_area_l3500_350025

theorem rectangle_area (perimeter width : ℝ) (h1 : perimeter = 52) (h2 : width = 11) :
  (width * (perimeter / 2 - width)) = 165 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3500_350025


namespace NUMINAMATH_CALUDE_white_patterns_count_l3500_350073

/-- The number of different white figures on an n × n board created by k rectangles -/
def whitePatterns (n k : ℕ) : ℕ :=
  (Nat.choose n k) ^ 2

/-- Theorem stating the number of different white figures -/
theorem white_patterns_count (n k : ℕ) (h1 : n > 0) (h2 : k > 0) (h3 : k ≤ n) :
  whitePatterns n k = (Nat.choose n k) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_white_patterns_count_l3500_350073


namespace NUMINAMATH_CALUDE_married_women_fraction_l3500_350092

theorem married_women_fraction (total_men : ℕ) (total_women : ℕ) (single_men : ℕ) :
  (single_men : ℚ) / total_men = 3 / 7 →
  total_women = total_men - single_men →
  (total_women : ℚ) / (total_men + total_women) = 4 / 11 :=
by sorry

end NUMINAMATH_CALUDE_married_women_fraction_l3500_350092


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3500_350034

theorem inequality_equivalence (x : ℝ) : 
  (x ≠ 5) → ((x^2 + 2*x + 1) / ((x-5)^2) ≥ 15 ↔ 
    ((76 - 3*Real.sqrt 60) / 14 ≤ x ∧ x < 5) ∨ 
    (5 < x ∧ x ≤ (76 + 3*Real.sqrt 60) / 14)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3500_350034


namespace NUMINAMATH_CALUDE_no_solution_x6_2y2_plus_2_l3500_350063

theorem no_solution_x6_2y2_plus_2 : ∀ (x y : ℤ), x^6 ≠ 2*y^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_x6_2y2_plus_2_l3500_350063


namespace NUMINAMATH_CALUDE_min_sum_squares_l3500_350075

/-- A random variable with normal distribution N(1, σ²) -/
def X (σ : ℝ) : Type := Unit

/-- The probability that X is less than or equal to a -/
def P_le (σ : ℝ) (X : X σ) (a : ℝ) : ℝ := sorry

/-- The probability that X is greater than or equal to b -/
def P_ge (σ : ℝ) (X : X σ) (b : ℝ) : ℝ := sorry

/-- The theorem stating that the minimum value of a² + b² is 2 -/
theorem min_sum_squares (σ : ℝ) (X : X σ) (a b : ℝ) 
  (h : P_le σ X a = P_ge σ X b) : 
  ∃ (min : ℝ), min = 2 ∧ ∀ (x y : ℝ), P_le σ X x = P_ge σ X y → x^2 + y^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3500_350075


namespace NUMINAMATH_CALUDE_marius_monica_difference_l3500_350021

/-- The number of subjects taken by students Millie, Monica, and Marius. -/
structure SubjectCounts where
  millie : ℕ
  monica : ℕ
  marius : ℕ

/-- The conditions of the problem. -/
def problem_conditions (counts : SubjectCounts) : Prop :=
  counts.millie = counts.marius + 3 ∧
  counts.marius > counts.monica ∧
  counts.monica = 10 ∧
  counts.millie + counts.monica + counts.marius = 41

/-- The theorem stating that Marius takes 4 more subjects than Monica. -/
theorem marius_monica_difference (counts : SubjectCounts) 
  (h : problem_conditions counts) : counts.marius - counts.monica = 4 := by
  sorry

end NUMINAMATH_CALUDE_marius_monica_difference_l3500_350021


namespace NUMINAMATH_CALUDE_table_runner_coverage_l3500_350082

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (two_layer_area : ℝ) (four_layer_area : ℝ) 
  (h1 : total_runner_area = 360)
  (h2 : table_area = 250)
  (h3 : two_layer_area = 35)
  (h4 : four_layer_area = 15)
  (h5 : 0.9 * table_area = two_layer_area + three_layer_area + four_layer_area + one_layer_area)
  (h6 : total_runner_area = one_layer_area + 2 * two_layer_area + 3 * three_layer_area + 4 * four_layer_area) :
  three_layer_area = 65 := by
  sorry


end NUMINAMATH_CALUDE_table_runner_coverage_l3500_350082


namespace NUMINAMATH_CALUDE_matts_future_age_l3500_350093

theorem matts_future_age (bush_age : ℕ) (age_difference : ℕ) (years_from_now : ℕ) :
  bush_age = 12 →
  age_difference = 3 →
  years_from_now = 10 →
  bush_age + age_difference + years_from_now = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_matts_future_age_l3500_350093


namespace NUMINAMATH_CALUDE_empty_lorry_weight_l3500_350003

/-- The weight of an empty lorry given the following conditions:
  * The lorry is loaded with 20 bags of apples.
  * Each bag of apples weighs 60 pounds.
  * The weight of the loaded lorry is 1700 pounds.
-/
theorem empty_lorry_weight : ℕ := by
  sorry

#check empty_lorry_weight

end NUMINAMATH_CALUDE_empty_lorry_weight_l3500_350003


namespace NUMINAMATH_CALUDE_last_two_digits_13_pow_101_base_3_l3500_350099

theorem last_two_digits_13_pow_101_base_3 : ∃ n : ℕ, 13^101 ≡ 21 [MOD 9] ∧ n * 9 + 21 = 13^101 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_13_pow_101_base_3_l3500_350099


namespace NUMINAMATH_CALUDE_common_chords_concur_l3500_350057

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Three pairwise intersecting circles --/
structure ThreeIntersectingCircles where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  intersect_12 : c1.center.1 ^ 2 + c1.center.2 ^ 2 ≠ c2.center.1 ^ 2 + c2.center.2 ^ 2 ∨ c1.center ≠ c2.center
  intersect_23 : c2.center.1 ^ 2 + c2.center.2 ^ 2 ≠ c3.center.1 ^ 2 + c3.center.2 ^ 2 ∨ c2.center ≠ c3.center
  intersect_31 : c3.center.1 ^ 2 + c3.center.2 ^ 2 ≠ c1.center.1 ^ 2 + c1.center.2 ^ 2 ∨ c3.center ≠ c1.center

/-- A line in a plane, represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The common chord of two intersecting circles --/
def commonChord (c1 c2 : Circle) : Line := sorry

/-- Three lines concur if they all pass through a single point --/
def concur (l1 l2 l3 : Line) : Prop := sorry

/-- The theorem: The common chords of three pairwise intersecting circles concur --/
theorem common_chords_concur (circles : ThreeIntersectingCircles) :
  let chord12 := commonChord circles.c1 circles.c2
  let chord23 := commonChord circles.c2 circles.c3
  let chord31 := commonChord circles.c3 circles.c1
  concur chord12 chord23 chord31 := by sorry

end NUMINAMATH_CALUDE_common_chords_concur_l3500_350057


namespace NUMINAMATH_CALUDE_triangle_side_length_l3500_350091

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  a = 3 →
  b - c = 2 →
  Real.cos B = -1/2 →
  b = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3500_350091


namespace NUMINAMATH_CALUDE_line_equation_proof_l3500_350015

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point is on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if two lines are parallel
def linesParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- Theorem statement
theorem line_equation_proof (l : Line) (p : Point) (given_line : Line) :
  pointOnLine p l ∧ 
  p = Point.mk 0 3 ∧ 
  linesParallel l given_line ∧ 
  given_line = Line.mk 1 (-1) (-1) →
  l = Line.mk 1 (-1) 3 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3500_350015


namespace NUMINAMATH_CALUDE_point_c_coordinates_l3500_350085

/-- Given points A, B, and C on a line, where C divides AB in the ratio 2:1,
    prove that C has the specified coordinates. -/
theorem point_c_coordinates (A B C : ℝ × ℝ) : 
  A = (-3, -2) →
  B = (5, 10) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →  -- C is on line segment AB
  dist A C = 2 * dist C B →                             -- AC = 2CB
  C = (11/3, 8) := by
  sorry


end NUMINAMATH_CALUDE_point_c_coordinates_l3500_350085


namespace NUMINAMATH_CALUDE_inequality_proof_l3500_350032

theorem inequality_proof (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ((2 + x) / (1 + x))^2 + ((2 + y) / (1 + y))^2 ≥ 9/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3500_350032


namespace NUMINAMATH_CALUDE_minutes_before_noon_l3500_350084

theorem minutes_before_noon (x : ℕ) : 
  (180 - (x + 40) = 3 * x) →  -- Condition 1 and 3
  x = 35                      -- The result we want to prove
  := by sorry

end NUMINAMATH_CALUDE_minutes_before_noon_l3500_350084


namespace NUMINAMATH_CALUDE_cos_150_degrees_l3500_350000

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l3500_350000


namespace NUMINAMATH_CALUDE_point_on_h_graph_l3500_350074

theorem point_on_h_graph (g : ℝ → ℝ) (h : ℝ → ℝ) : 
  g 4 = 7 → 
  (∀ x, h x = (g x + 1)^2) → 
  ∃ x y, h x = y ∧ x + y = 68 := by
sorry

end NUMINAMATH_CALUDE_point_on_h_graph_l3500_350074


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3500_350086

theorem fraction_meaningful (a : ℝ) : 
  (∃ x : ℝ, x = 1 / (a + 3)) ↔ a ≠ -3 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3500_350086


namespace NUMINAMATH_CALUDE_mean_median_difference_l3500_350087

def frequency_distribution : List ℕ := [2, 3, 4, 5, 2, 1, 1]
def days_missed : List ℕ := [0, 1, 2, 3, 4, 5, 6]
def total_students : ℕ := 18

def median (n : ℕ) (freq : List ℕ) (days : List ℕ) : ℚ := sorry

def mean (freq : List ℕ) (days : List ℕ) (total : ℕ) : ℚ := sorry

theorem mean_median_difference :
  mean frequency_distribution days_missed total_students = 
  median total_students frequency_distribution days_missed - 1/3 := by sorry

end NUMINAMATH_CALUDE_mean_median_difference_l3500_350087


namespace NUMINAMATH_CALUDE_root_sum_sixth_power_l3500_350060

theorem root_sum_sixth_power (r s : ℝ) : 
  r^2 - 2*r + Real.sqrt 2 = 0 → 
  s^2 - 2*s + Real.sqrt 2 = 0 → 
  r^6 + s^6 = 904 - 640 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_root_sum_sixth_power_l3500_350060


namespace NUMINAMATH_CALUDE_sum_of_matching_positions_is_322_l3500_350043

def array_size : Nat × Nat := (16, 10)

def esther_fill (r c : Nat) : Nat :=
  16 * (r - 1) + c

def frida_fill (r c : Nat) : Nat :=
  10 * (c - 1) + r

def is_same_position (r c : Nat) : Prop :=
  esther_fill r c = frida_fill r c

def sum_of_matching_positions : Nat :=
  (esther_fill 1 1) + (esther_fill 4 6) + (esther_fill 7 11) + (esther_fill 10 16)

theorem sum_of_matching_positions_is_322 :
  sum_of_matching_positions = 322 :=
sorry

end NUMINAMATH_CALUDE_sum_of_matching_positions_is_322_l3500_350043


namespace NUMINAMATH_CALUDE_apartment_cost_comparison_l3500_350017

/-- Proves that the average cost per mile driven is $0.58 given the conditions of the apartment comparison problem. -/
theorem apartment_cost_comparison (rent1 rent2 utilities1 utilities2 : ℕ)
  (miles_per_day1 miles_per_day2 work_days_per_month : ℕ)
  (total_cost_difference : ℚ) :
  rent1 = 800 →
  rent2 = 900 →
  utilities1 = 260 →
  utilities2 = 200 →
  miles_per_day1 = 31 →
  miles_per_day2 = 21 →
  work_days_per_month = 20 →
  total_cost_difference = 76 →
  let total_miles1 := miles_per_day1 * work_days_per_month
  let total_miles2 := miles_per_day2 * work_days_per_month
  let cost_per_mile := (rent1 + utilities1 - rent2 - utilities2 + total_cost_difference) / (total_miles1 - total_miles2)
  cost_per_mile = 29/50 := by
  sorry

end NUMINAMATH_CALUDE_apartment_cost_comparison_l3500_350017


namespace NUMINAMATH_CALUDE_flower_bed_dimensions_l3500_350038

theorem flower_bed_dimensions (l w : ℝ) : 
  (l + 3) * (w + 2) = l * w + 64 →
  (l + 2) * (w + 3) = l * w + 68 →
  l = 14 ∧ w = 10 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_dimensions_l3500_350038


namespace NUMINAMATH_CALUDE_expression_simplification_l3500_350009

theorem expression_simplification :
  1 + (1 : ℝ) / (1 + Real.sqrt 2) - 1 / (1 - Real.sqrt 5) =
  1 + (-Real.sqrt 2 - Real.sqrt 5) / (1 + Real.sqrt 2 - Real.sqrt 5 - Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3500_350009


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l3500_350066

def p (x : ℝ) : ℝ := 4 * x^4 + 11 * x^3 - 37 * x^2 + 18 * x

theorem roots_of_polynomial : 
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-6) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l3500_350066


namespace NUMINAMATH_CALUDE_oven_capacity_is_two_l3500_350002

/-- Represents the pizza-making process with given constraints -/
structure PizzaMaking where
  dough_time : ℕ  -- Time to make one batch of dough (in minutes)
  cook_time : ℕ   -- Time to cook pizzas in the oven (in minutes)
  pizzas_per_batch : ℕ  -- Number of pizzas one batch of dough can make
  total_time : ℕ  -- Total time to make all pizzas (in minutes)
  total_pizzas : ℕ  -- Total number of pizzas to be made

/-- Calculates the number of pizzas that can fit in the oven at once -/
def oven_capacity (pm : PizzaMaking) : ℕ :=
  let dough_making_time := (pm.total_pizzas / pm.pizzas_per_batch) * pm.dough_time
  let baking_time := pm.total_time - dough_making_time
  let baking_intervals := baking_time / pm.cook_time
  pm.total_pizzas / baking_intervals

/-- Theorem stating that given the conditions, the oven capacity is 2 pizzas -/
theorem oven_capacity_is_two (pm : PizzaMaking)
  (h1 : pm.dough_time = 30)
  (h2 : pm.cook_time = 30)
  (h3 : pm.pizzas_per_batch = 3)
  (h4 : pm.total_time = 300)  -- 5 hours = 300 minutes
  (h5 : pm.total_pizzas = 12) :
  oven_capacity pm = 2 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_oven_capacity_is_two_l3500_350002


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3500_350072

theorem arithmetic_expression_equality : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3500_350072


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3500_350049

/-- Represents the price reduction in yuan -/
def price_reduction : ℝ := 20

/-- Average daily sale before price reduction -/
def initial_sales : ℝ := 20

/-- Initial profit per piece in yuan -/
def initial_profit_per_piece : ℝ := 40

/-- Increase in sales per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Target daily profit in yuan -/
def target_profit : ℝ := 1200

/-- Theorem stating that the given price reduction achieves the target profit -/
theorem price_reduction_achieves_target_profit :
  (initial_sales + sales_increase_rate * price_reduction) * 
  (initial_profit_per_piece - price_reduction) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3500_350049


namespace NUMINAMATH_CALUDE_worker_completion_time_l3500_350078

/-- Given two workers P and Q, this theorem proves the time taken by Q to complete a task alone,
    given the time taken by P alone and the time taken by P and Q together. -/
theorem worker_completion_time (time_p : ℝ) (time_pq : ℝ) (time_q : ℝ) : 
  time_p = 15 → time_pq = 6 → time_q = 10 → 
  1 / time_pq = 1 / time_p + 1 / time_q :=
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l3500_350078


namespace NUMINAMATH_CALUDE_fraction_of_three_fourths_is_one_fifth_l3500_350024

theorem fraction_of_three_fourths_is_one_fifth (x : ℚ) : x * (3 / 4 : ℚ) = (1 / 5 : ℚ) → x = (4 / 15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_three_fourths_is_one_fifth_l3500_350024


namespace NUMINAMATH_CALUDE_angle_trisection_l3500_350031

/-- Given an angle of 54°, prove that its trisection results in three equal angles of 18° each. -/
theorem angle_trisection (θ : Real) (h : θ = 54) : 
  ∃ (α β γ : Real), α = β ∧ β = γ ∧ α + β + γ = θ ∧ α = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_trisection_l3500_350031


namespace NUMINAMATH_CALUDE_binary_digit_difference_l3500_350081

/-- Returns the number of digits in the base-2 representation of a natural number -/
def numDigitsBinary (n : ℕ) : ℕ :=
  if n = 0 then 1 else (Nat.log2 n).succ

/-- The difference between the number of digits in the base-2 representation of 1500
    and the number of digits in the base-2 representation of 300 is 2 -/
theorem binary_digit_difference :
  numDigitsBinary 1500 - numDigitsBinary 300 = 2 := by
  sorry

#eval numDigitsBinary 1500 - numDigitsBinary 300

end NUMINAMATH_CALUDE_binary_digit_difference_l3500_350081


namespace NUMINAMATH_CALUDE_student_weights_l3500_350035

/-- Theorem: Total and average weight of students
Given 10 students with a base weight and weight deviations, 
prove the total weight and average weight. -/
theorem student_weights (base_weight : ℝ) (weight_deviations : List ℝ) : 
  base_weight = 50 ∧ 
  weight_deviations = [2, 3, -7.5, -3, 5, -8, 3.5, 4.5, 8, -1.5] →
  (List.sum weight_deviations + 10 * base_weight = 509) ∧
  ((List.sum weight_deviations + 10 * base_weight) / 10 = 50.9) := by
  sorry

#check student_weights

end NUMINAMATH_CALUDE_student_weights_l3500_350035


namespace NUMINAMATH_CALUDE_picnic_attendance_theorem_l3500_350013

/-- The percentage of men who attended the picnic -/
def men_attendance_rate : ℝ := 0.20

/-- The percentage of women who attended the picnic -/
def women_attendance_rate : ℝ := 0.40

/-- The percentage of employees who are men -/
def men_employee_rate : ℝ := 0.55

/-- The percentage of all employees who attended the picnic -/
def total_attendance_rate : ℝ := men_employee_rate * men_attendance_rate + (1 - men_employee_rate) * women_attendance_rate

theorem picnic_attendance_theorem :
  total_attendance_rate = 0.29 := by sorry

end NUMINAMATH_CALUDE_picnic_attendance_theorem_l3500_350013


namespace NUMINAMATH_CALUDE_equation_solution_l3500_350022

theorem equation_solution : ∃ x : ℚ, 64 * (2 * x - 1)^3 = 27 ∧ x = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3500_350022


namespace NUMINAMATH_CALUDE_odd_function_extension_l3500_350056

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_positive : ∀ x > 0, f x = Real.exp x) :
  ∀ x < 0, f x = -Real.exp (-x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l3500_350056


namespace NUMINAMATH_CALUDE_seven_people_round_table_l3500_350096

/-- The number of unique seating arrangements for n people around a round table,
    considering rotations as the same arrangement -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique seating arrangements for 7 people
    around a round table is equal to 6! -/
theorem seven_people_round_table :
  roundTableArrangements 7 = Nat.factorial 6 := by sorry

end NUMINAMATH_CALUDE_seven_people_round_table_l3500_350096


namespace NUMINAMATH_CALUDE_peach_difference_l3500_350055

theorem peach_difference (martine_peaches benjy_peaches gabrielle_peaches : ℕ) : 
  martine_peaches > 2 * benjy_peaches →
  benjy_peaches = gabrielle_peaches / 3 →
  martine_peaches = 16 →
  gabrielle_peaches = 15 →
  martine_peaches - 2 * benjy_peaches = 6 := by
sorry

end NUMINAMATH_CALUDE_peach_difference_l3500_350055


namespace NUMINAMATH_CALUDE_mustard_at_third_table_l3500_350098

theorem mustard_at_third_table 
  (first_table : Real) 
  (second_table : Real) 
  (total_mustard : Real) 
  (h1 : first_table = 0.25)
  (h2 : second_table = 0.25)
  (h3 : total_mustard = 0.88) :
  total_mustard - (first_table + second_table) = 0.38 := by
sorry

end NUMINAMATH_CALUDE_mustard_at_third_table_l3500_350098


namespace NUMINAMATH_CALUDE_min_value_x_l3500_350051

theorem min_value_x (x : ℝ) (h1 : x > 0) (h2 : 2 * Real.log x ≥ Real.log 8 + Real.log x) (h3 : x ≤ 32) :
  x ≥ 8 ∧ ∀ y : ℝ, y > 0 → 2 * Real.log y ≥ Real.log 8 + Real.log y → y ≤ 32 → y ≥ x := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_l3500_350051


namespace NUMINAMATH_CALUDE_hexagon_largest_angle_l3500_350047

theorem hexagon_largest_angle (a b c d e f : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  b / a = 3 / 2 →
  c / a = 3 / 2 →
  d / a = 2 →
  e / a = 2 →
  f / a = 5 / 2 →
  a + b + c + d + e + f = 720 →
  f = 1200 / 7 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_largest_angle_l3500_350047


namespace NUMINAMATH_CALUDE_shells_per_friend_eq_l3500_350090

/-- The number of shells each friend gets when Jillian, Savannah, and Clayton
    distribute their shells evenly among F friends. -/
def shellsPerFriend (F : ℕ+) : ℚ :=
  let J : ℕ := 29  -- Jillian's shells
  let S : ℕ := 17  -- Savannah's shells
  let C : ℕ := 8   -- Clayton's shells
  (J + S + C) / F

/-- Theorem stating that the number of shells each friend gets is 54 / F. -/
theorem shells_per_friend_eq (F : ℕ+) : shellsPerFriend F = 54 / F := by
  sorry

end NUMINAMATH_CALUDE_shells_per_friend_eq_l3500_350090


namespace NUMINAMATH_CALUDE_fraction_sum_l3500_350064

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 5) : (a + b) / b = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3500_350064


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3500_350044

theorem quadratic_solution_property (a : ℝ) : 
  a^2 - 2*a - 1 = 0 → 2*a^2 - 4*a + 2023 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3500_350044


namespace NUMINAMATH_CALUDE_sin_C_value_side_lengths_l3500_350007

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 13 ∧ Real.cos t.A = 5/13

-- Theorem 1
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) (ha : t.a = 36) :
  Real.sin t.C = 1/3 := by
  sorry

-- Theorem 2
theorem side_lengths (t : Triangle) (h : triangle_conditions t) (harea : (1/2) * t.b * t.c * Real.sin t.A = 6) :
  t.a = 4 * Real.sqrt 10 ∧ t.b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_C_value_side_lengths_l3500_350007


namespace NUMINAMATH_CALUDE_multiple_is_two_l3500_350097

-- Define the depths of the pools
def johns_pool_depth : ℝ := 15
def sarahs_pool_depth : ℝ := 5

-- Define the relationship between the pool depths
def depth_relation (x : ℝ) : Prop :=
  johns_pool_depth = x * sarahs_pool_depth + 5

-- Theorem statement
theorem multiple_is_two :
  ∃ x : ℝ, depth_relation x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_multiple_is_two_l3500_350097


namespace NUMINAMATH_CALUDE_least_common_time_for_seven_horses_l3500_350061

def horse_times : Finset ℕ := Finset.range 12

theorem least_common_time_for_seven_horses :
  ∃ (S : Finset ℕ), S ⊆ horse_times ∧ S.card = 7 ∧
  (∀ n ∈ S, n > 0) ∧
  (∀ (T : ℕ), (∀ n ∈ S, T % n = 0) → T ≥ 420) ∧
  (∀ n ∈ S, 420 % n = 0) :=
sorry

end NUMINAMATH_CALUDE_least_common_time_for_seven_horses_l3500_350061


namespace NUMINAMATH_CALUDE_democrat_ratio_l3500_350039

theorem democrat_ratio (total_participants male_participants female_participants female_democrats : ℕ)
  (h1 : total_participants = 840)
  (h2 : male_participants + female_participants = total_participants)
  (h3 : 3 * (female_democrats + male_participants / 4) = total_participants)
  (h4 : female_democrats = 140) :
  2 * female_democrats = female_participants :=
by sorry

end NUMINAMATH_CALUDE_democrat_ratio_l3500_350039


namespace NUMINAMATH_CALUDE_function_decreasing_range_l3500_350026

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 * a - 1) * x + 4 * a else a / x

theorem function_decreasing_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₂ - f a x₁) / (x₂ - x₁) < 0) ↔
  a ∈ Set.Icc (1 / 5 : ℝ) (1 / 2 : ℝ) ∧ a ≠ 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_function_decreasing_range_l3500_350026


namespace NUMINAMATH_CALUDE_joe_total_cars_l3500_350029

def initial_cars : ℕ := 50
def additional_cars : ℕ := 12

theorem joe_total_cars : initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_total_cars_l3500_350029


namespace NUMINAMATH_CALUDE_inheritance_solution_l3500_350070

/-- Represents the inheritance problem with given conditions --/
def inheritance_problem (total : ℝ) : Prop :=
  ∃ (x : ℝ),
    x > 0 ∧
    (total - x) > 0 ∧
    0.05 * x + 0.065 * (total - x) = 227 ∧
    total - x = 1800

/-- The theorem stating the solution to the inheritance problem --/
theorem inheritance_solution :
  ∃ (total : ℝ), inheritance_problem total ∧ total = 4000 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_solution_l3500_350070


namespace NUMINAMATH_CALUDE_converse_statement_is_false_l3500_350019

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  eq : ∀ (x y : ℝ), x^2 / a^2 + y^2 / 4 = 1
  foci_on_x : True  -- We assume this property is satisfied

/-- The converse statement is false -/
theorem converse_statement_is_false : 
  ¬(∀ e : Ellipse, e.a = 4) :=
sorry

end NUMINAMATH_CALUDE_converse_statement_is_false_l3500_350019


namespace NUMINAMATH_CALUDE_p_squared_plus_98_composite_l3500_350069

theorem p_squared_plus_98_composite (p : ℕ) (h : Prime p) : ¬ Prime (p^2 + 98) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_98_composite_l3500_350069


namespace NUMINAMATH_CALUDE_rounding_bounds_l3500_350076

def rounded_value : ℕ := 1300000

theorem rounding_bounds :
  ∀ n : ℕ,
  (n + 50000) / 100000 * 100000 = rounded_value →
  n ≤ 1304999 ∧ n ≥ 1295000 :=
by sorry

end NUMINAMATH_CALUDE_rounding_bounds_l3500_350076


namespace NUMINAMATH_CALUDE_square_land_area_l3500_350046

/-- A square land plot with side length 30 units has an area of 900 square units. -/
theorem square_land_area (side_length : ℝ) (h1 : side_length = 30) :
  side_length * side_length = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_land_area_l3500_350046


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3500_350065

theorem marble_fraction_after_tripling (total : ℚ) (h1 : total > 0) : 
  let initial_blue := (4 : ℚ) / 7 * total
  let initial_red := total - initial_blue
  let final_red := 3 * initial_red
  let final_total := initial_blue + final_red
  final_red / final_total = (9 : ℚ) / 13 :=
by sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3500_350065


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l3500_350040

/-- Represents the price of orangeade per glass on a given day -/
structure OrangeadePrice where
  price : ℝ
  day : ℕ

/-- Represents the volume of orangeade made on a given day -/
structure OrangeadeVolume where
  volume : ℝ
  day : ℕ

/-- Represents the revenue from selling orangeade on a given day -/
def revenue (p : OrangeadePrice) (v : OrangeadeVolume) : ℝ :=
  p.price * v.volume

theorem orangeade_price_day2 
  (juice : ℝ) -- Amount of orange juice used (same for both days)
  (v1 : OrangeadeVolume) -- Volume of orangeade on day 1
  (v2 : OrangeadeVolume) -- Volume of orangeade on day 2
  (p1 : OrangeadePrice) -- Price of orangeade on day 1
  (p2 : OrangeadePrice) -- Price of orangeade on day 2
  (h1 : v1.volume = 2 * juice) -- Volume on day 1 is twice the amount of juice
  (h2 : v2.volume = 3 * juice) -- Volume on day 2 is thrice the amount of juice
  (h3 : v1.day = 1 ∧ v2.day = 2) -- Volumes correspond to days 1 and 2
  (h4 : p1.day = 1 ∧ p2.day = 2) -- Prices correspond to days 1 and 2
  (h5 : p1.price = 0.6) -- Price on day 1 is $0.60
  (h6 : revenue p1 v1 = revenue p2 v2) -- Revenue is the same for both days
  : p2.price = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l3500_350040


namespace NUMINAMATH_CALUDE_remainder_problem_l3500_350077

theorem remainder_problem : (123456789012 : ℕ) % 252 = 84 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3500_350077


namespace NUMINAMATH_CALUDE_students_just_passed_l3500_350023

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ)
  (h_total : total = 300)
  (h_first : first_div_percent = 25 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 63 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l3500_350023


namespace NUMINAMATH_CALUDE_sine_amplitude_l3500_350068

/-- Given a sine function y = a * sin(bx + c) + d where a, b, c, and d are positive constants,
    if the graph oscillates between 5 and -3, then a = 4 -/
theorem sine_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by sorry

end NUMINAMATH_CALUDE_sine_amplitude_l3500_350068


namespace NUMINAMATH_CALUDE_condition_equivalent_to_a_range_l3500_350011

/-- The function f(x) = ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1

/-- The function g(x) = -x^2 + 2x + 1 -/
def g (x : ℝ) : ℝ := -x^2 + 2*x + 1

/-- The theorem stating the equivalence between the condition and the range of a -/
theorem condition_equivalent_to_a_range :
  ∀ a : ℝ, (∀ x₁ ∈ Set.Icc (-1) 1, ∃ x₂ ∈ Set.Icc 0 2, f a x₁ < g x₂) ↔ a ∈ Set.Ioo (-3) 3 :=
by sorry

end NUMINAMATH_CALUDE_condition_equivalent_to_a_range_l3500_350011


namespace NUMINAMATH_CALUDE_mean_of_special_set_l3500_350005

def is_valid_set (S : Finset ℝ) : Prop :=
  let n := S.card
  let s := S.sum id
  (s + 1) / (n + 1) = s / n - 13 ∧
  (s + 2001) / (n + 1) = s / n + 27

theorem mean_of_special_set (S : Finset ℝ) (h : is_valid_set S) :
  S.sum id / S.card = 651 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_special_set_l3500_350005


namespace NUMINAMATH_CALUDE_function_equality_implies_a_range_l3500_350006

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a| + |x + 3|
def g (x : ℝ) : ℝ := |x - 1| + 2

-- State the theorem
theorem function_equality_implies_a_range (a : ℝ) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) →
  a ≥ 5 ∨ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_function_equality_implies_a_range_l3500_350006


namespace NUMINAMATH_CALUDE_pineapple_weight_l3500_350059

theorem pineapple_weight (P : ℝ) 
  (h1 : P > 0)
  (h2 : P / 6 + 2 / 5 * (5 / 6 * P) + 2 / 3 * (P / 2) + 120 = P) : 
  P = 720 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_weight_l3500_350059


namespace NUMINAMATH_CALUDE_notebook_cost_l3500_350012

theorem notebook_cost (total_cost cover_cost notebook_cost : ℝ) : 
  total_cost = 3.60 →
  notebook_cost = 1.5 * cover_cost →
  total_cost = notebook_cost + cover_cost →
  notebook_cost = 2.16 := by
sorry

end NUMINAMATH_CALUDE_notebook_cost_l3500_350012


namespace NUMINAMATH_CALUDE_not_parallel_if_intersect_and_contain_perpendicular_if_parallel_and_perpendicular_l3500_350050

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the basic relations
variable (intersects : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Theorem 1
theorem not_parallel_if_intersect_and_contain 
  (a b : Line) (α : Plane) (P : Point) :
  intersects a α ∧ contains α b → ¬ parallel a b := by sorry

-- Theorem 2
theorem perpendicular_if_parallel_and_perpendicular 
  (a b : Line) (α : Plane) :
  parallel a b ∧ perpendicular b α → perpendicular a α := by sorry

end NUMINAMATH_CALUDE_not_parallel_if_intersect_and_contain_perpendicular_if_parallel_and_perpendicular_l3500_350050


namespace NUMINAMATH_CALUDE_gdp_2010_calculation_gdp_2010_l3500_350001

def gdp_2008 : ℝ := 1050
def growth_rate : ℝ := 0.132

theorem gdp_2010_calculation : 
  gdp_2008 * (1 + growth_rate)^2 = gdp_2008 * (1 + growth_rate) * (1 + growth_rate) :=
by sorry

theorem gdp_2010 : ℝ := gdp_2008 * (1 + growth_rate)^2

end NUMINAMATH_CALUDE_gdp_2010_calculation_gdp_2010_l3500_350001


namespace NUMINAMATH_CALUDE_distance_to_xoy_plane_l3500_350080

-- Define a 3D point
def Point3D := ℝ × ℝ × ℝ

-- Define the distance from a point to the xOy plane
def distToXOYPlane (p : Point3D) : ℝ := |p.2.2|

-- Theorem statement
theorem distance_to_xoy_plane :
  let P : Point3D := (1, -3, 2)
  distToXOYPlane P = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_xoy_plane_l3500_350080


namespace NUMINAMATH_CALUDE_floor_minus_y_eq_zero_l3500_350041

theorem floor_minus_y_eq_zero (y : ℝ) (h : ⌊y⌋ + ⌈y⌉ = 2 * y) : ⌊y⌋ - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_floor_minus_y_eq_zero_l3500_350041


namespace NUMINAMATH_CALUDE_solve_equation_l3500_350036

theorem solve_equation (n : ℚ) : 
  (1 : ℚ) / (n + 1) + 2 / (n + 1) + (n + 1) / (n + 1) = 2 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3500_350036


namespace NUMINAMATH_CALUDE_b_value_l3500_350030

/-- A probability distribution for a random variable X -/
structure ProbDist where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_to_one : a + b + c = 1
  b_is_mean : b = (a + c) / 2

/-- The value of b in the probability distribution is 1/3 -/
theorem b_value (p : ProbDist) : p.b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_b_value_l3500_350030


namespace NUMINAMATH_CALUDE_frequency_histogram_height_property_l3500_350054

/-- Represents a frequency distribution histogram --/
structure FrequencyHistogram where
  /-- The height of a bar in the histogram --/
  height : ℝ → ℝ
  /-- The frequency of individuals in a group --/
  frequency : ℝ → ℝ
  /-- The class interval for a group --/
  classInterval : ℝ → ℝ

/-- Theorem stating that the height of a frequency distribution histogram
    represents the ratio of frequency to class interval --/
theorem frequency_histogram_height_property (h : FrequencyHistogram) (x : ℝ) :
  h.height x = h.frequency x / h.classInterval x := by
  sorry

end NUMINAMATH_CALUDE_frequency_histogram_height_property_l3500_350054


namespace NUMINAMATH_CALUDE_unique_divisor_property_l3500_350008

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem unique_divisor_property : ∃! n : ℕ, n > 0 ∧ n = 100 * divisor_count n :=
  sorry

end NUMINAMATH_CALUDE_unique_divisor_property_l3500_350008


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l3500_350083

theorem complex_fraction_equals_i : (Complex.I + 3) / (1 - 3 * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l3500_350083


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3500_350016

theorem arithmetic_calculations :
  (1 - 3 - (-4) = 1) ∧
  (-1/3 + (-4/3) = -5/3) ∧
  ((-2) * (-3) * (-5) = -30) ∧
  (15 / 4 * (-1/4) = -15/16) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3500_350016


namespace NUMINAMATH_CALUDE_percent_relation_l3500_350079

theorem percent_relation (a b : ℝ) (h : a = 1.8 * b) : 
  (4 * b) / a * 100 = 222.22 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3500_350079


namespace NUMINAMATH_CALUDE_sequence_range_l3500_350045

theorem sequence_range (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_recur : ∀ n, a (n + 1) ≥ 2 * a n + 1) 
  (h_bound : ∀ n, a n < 2^(n + 1)) : 
  0 < a 1 ∧ a 1 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_range_l3500_350045


namespace NUMINAMATH_CALUDE_percentage_passed_l3500_350042

def total_students : ℕ := 800
def failed_students : ℕ := 520

theorem percentage_passed : 
  (((total_students - failed_students) : ℚ) / total_students) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_l3500_350042


namespace NUMINAMATH_CALUDE_count_odd_two_digit_integers_l3500_350018

/-- A function that returns true if a natural number is odd, false otherwise -/
def is_odd (n : ℕ) : Bool :=
  n % 2 = 1

/-- The set of odd digits (1, 3, 5, 7, 9) -/
def odd_digits : Finset ℕ :=
  {1, 3, 5, 7, 9}

/-- A function that returns true if a natural number is a two-digit integer, false otherwise -/
def is_two_digit (n : ℕ) : Bool :=
  10 ≤ n ∧ n ≤ 99

/-- The set of two-digit integers where both digits are odd -/
def odd_two_digit_integers : Finset ℕ :=
  Finset.filter (fun n => is_two_digit n ∧ is_odd (n / 10) ∧ is_odd (n % 10)) (Finset.range 100)

theorem count_odd_two_digit_integers : 
  Finset.card odd_two_digit_integers = 25 :=
sorry

end NUMINAMATH_CALUDE_count_odd_two_digit_integers_l3500_350018


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3500_350052

theorem unique_positive_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = z) (h2 : y * z = x) (h3 : z * x = y) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3500_350052


namespace NUMINAMATH_CALUDE_absolute_sum_sequence_minimum_sum_l3500_350089

/-- An absolute sum sequence with given initial term and absolute public sum. -/
def AbsoluteSumSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n => if n = 1 then a₁ else sorry

/-- The sum of the first n terms of an absolute sum sequence. -/
def SequenceSum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem absolute_sum_sequence_minimum_sum :
  ∀ a : ℕ → ℝ,
  (a 1 = 2) →
  (∀ n : ℕ, |a (n + 1)| + |a n| = 3) →
  SequenceSum a 2019 ≥ -3025 ∧
  ∃ a : ℕ → ℝ, (a 1 = 2) ∧ (∀ n : ℕ, |a (n + 1)| + |a n| = 3) ∧ SequenceSum a 2019 = -3025 :=
by sorry

end NUMINAMATH_CALUDE_absolute_sum_sequence_minimum_sum_l3500_350089


namespace NUMINAMATH_CALUDE_wall_width_proof_elijah_wall_width_l3500_350053

theorem wall_width_proof (total_walls : ℕ) (known_wall_width : ℝ) (known_wall_count : ℕ) 
  (total_tape_needed : ℝ) : ℝ :=
  let remaining_walls := total_walls - known_wall_count
  let known_walls_tape := known_wall_width * known_wall_count
  let remaining_tape := total_tape_needed - known_walls_tape
  remaining_tape / remaining_walls

theorem elijah_wall_width : wall_width_proof 4 4 2 20 = 6 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_proof_elijah_wall_width_l3500_350053
