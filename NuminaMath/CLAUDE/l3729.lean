import Mathlib

namespace NUMINAMATH_CALUDE_triangle_side_length_l3729_372934

/-- In a triangle ABC, given specific angle and side length conditions, prove the length of side b. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  C = 4 * A ∧  -- Given angle condition
  a = 20 ∧  -- Given side length
  c = 40 ∧  -- Given side length
  a / Real.sin A = b / Real.sin B ∧  -- Law of Sines
  a / Real.sin A = c / Real.sin C  -- Law of Sines
  →
  b = 20 * (16 * (9 * Real.sqrt 3 / 16) - 20 * (3 * Real.sqrt 3 / 4) + 5 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3729_372934


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3729_372941

theorem quadratic_inequality (x : ℝ) : x^2 + x - 20 < 0 ↔ -5 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3729_372941


namespace NUMINAMATH_CALUDE_snow_on_tuesday_l3729_372955

theorem snow_on_tuesday (monday_snow : ℝ) (total_snow : ℝ) (h1 : monday_snow = 0.32) (h2 : total_snow = 0.53) :
  total_snow - monday_snow = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_snow_on_tuesday_l3729_372955


namespace NUMINAMATH_CALUDE_equation_solution_l3729_372938

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 4) * x = 12 ∧ x = 168 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3729_372938


namespace NUMINAMATH_CALUDE_max_label_proof_l3729_372940

/-- Counts the number of '5' digits used to label boxes from 1 to n --/
def count_fives (n : ℕ) : ℕ := sorry

/-- The maximum number that can be labeled using 50 '5' digits --/
def max_label : ℕ := 235

theorem max_label_proof :
  count_fives max_label ≤ 50 ∧
  ∀ m : ℕ, m > max_label → count_fives m > 50 :=
sorry

end NUMINAMATH_CALUDE_max_label_proof_l3729_372940


namespace NUMINAMATH_CALUDE_pattern_proof_l3729_372997

theorem pattern_proof (n : ℕ) : (2*n - 1) * (2*n + 1) = (2*n)^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_pattern_proof_l3729_372997


namespace NUMINAMATH_CALUDE_people_counting_ratio_l3729_372998

theorem people_counting_ratio :
  ∀ (day1 day2 : ℕ),
  day2 = 500 →
  day1 + day2 = 1500 →
  ∃ (k : ℕ), day1 = k * day2 →
  day1 / day2 = 2 := by
sorry

end NUMINAMATH_CALUDE_people_counting_ratio_l3729_372998


namespace NUMINAMATH_CALUDE_BE_length_l3729_372937

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 3 ∧ dist B C = 4 ∧ dist C A = 5

-- Define points D and E on ray AB
def points_on_ray (A B D E : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ > 1 ∧ t₂ > t₁ ∧ D = A + t₁ • (B - A) ∧ E = A + t₂ • (B - A)

-- Define point F as intersection of circumcircles
def point_F (A B C D E F : ℝ × ℝ) : Prop :=
  F ≠ C ∧
  ∃ r₁ r₂ : ℝ,
    dist A F = r₁ ∧ dist C F = r₁ ∧ dist D F = r₁ ∧
    dist E F = r₂ ∧ dist B F = r₂ ∧ dist C F = r₂

-- Main theorem
theorem BE_length (A B C D E F : ℝ × ℝ) :
  triangle_ABC A B C →
  points_on_ray A B D E →
  point_F A B C D E F →
  dist D F = 3 →
  dist E F = 8 →
  dist B E = 3 + Real.sqrt 34.6 :=
sorry

end NUMINAMATH_CALUDE_BE_length_l3729_372937


namespace NUMINAMATH_CALUDE_george_exchange_rate_l3729_372903

/-- The amount George will receive for each special bill he exchanges on his 25th birthday. -/
def exchange_rate (total_years : ℕ) (spent_percentage : ℚ) (total_exchange_amount : ℚ) : ℚ :=
  let total_bills := total_years
  let remaining_bills := total_bills - (spent_percentage * total_bills)
  total_exchange_amount / remaining_bills

/-- Theorem stating that George will receive $1.50 for each special bill he exchanges. -/
theorem george_exchange_rate :
  exchange_rate 10 (1/5) 12 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_george_exchange_rate_l3729_372903


namespace NUMINAMATH_CALUDE_floor_equality_iff_in_range_l3729_372967

theorem floor_equality_iff_in_range (x : ℝ) : 
  ⌊2 * x + 1/2⌋ = ⌊x + 3⌋ ↔ x ∈ Set.Ici (5/2) ∩ Set.Iio (7/2) := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_iff_in_range_l3729_372967


namespace NUMINAMATH_CALUDE_min_value_theorem_l3729_372917

theorem min_value_theorem (x : ℝ) (h : x > 10) :
  (x^2 + 100) / (x - 10) ≥ 20 + 20 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3729_372917


namespace NUMINAMATH_CALUDE_physics_marks_l3729_372990

theorem physics_marks (P C M : ℝ) 
  (avg_total : (P + C + M) / 3 = 55)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 155 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l3729_372990


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l3729_372957

theorem power_two_greater_than_square_plus_one (n : ℕ) (h : n ≥ 5) :
  2^n > n^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_square_plus_one_l3729_372957


namespace NUMINAMATH_CALUDE_greatest_integer_with_conditions_l3729_372910

theorem greatest_integer_with_conditions : ∃ n : ℕ, 
  n < 150 ∧ 
  (∃ a b : ℕ, n + 2 = 9 * a ∧ n + 3 = 11 * b) ∧
  (∀ m : ℕ, m < 150 → (∃ c d : ℕ, m + 2 = 9 * c ∧ m + 3 = 11 * d) → m ≤ n) ∧
  n = 142 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_conditions_l3729_372910


namespace NUMINAMATH_CALUDE_equation_is_parabola_l3729_372907

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation |y - 3| = √((x+4)² + (y-1)²) -/
def equation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 4)^2 + (p.y - 1)^2)

/-- Represents a parabola in general form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point satisfies the parabola equation -/
def satisfies_parabola (p : Point2D) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- Theorem stating that the given equation represents a parabola -/
theorem equation_is_parabola :
  ∃ (para : Parabola), ∀ (p : Point2D), equation p → satisfies_parabola p para :=
sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l3729_372907


namespace NUMINAMATH_CALUDE_probability_N18_mod7_equals_1_is_2_7_l3729_372946

/-- The probability that N^18 mod 7 = 1, given N is an odd integer randomly chosen from 1 to 2023 -/
def probability_N18_mod7_equals_1 : ℚ :=
  let N := Finset.filter (fun n => n % 2 = 1) (Finset.range 2023)
  let favorable := N.filter (fun n => (n^18) % 7 = 1)
  favorable.card / N.card

theorem probability_N18_mod7_equals_1_is_2_7 :
  probability_N18_mod7_equals_1 = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_probability_N18_mod7_equals_1_is_2_7_l3729_372946


namespace NUMINAMATH_CALUDE_rectangle_from_equal_bisecting_diagonals_parallelogram_from_bisecting_diagonals_square_from_rhombus_equal_diagonals_l3729_372950

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of quadrilaterals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry
def diagonals_bisect_each_other (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorems to prove
theorem rectangle_from_equal_bisecting_diagonals (q : Quadrilateral) :
  has_equal_diagonals q → diagonals_bisect_each_other q → is_rectangle q := by sorry

theorem parallelogram_from_bisecting_diagonals (q : Quadrilateral) :
  diagonals_bisect_each_other q → is_parallelogram q := by sorry

theorem square_from_rhombus_equal_diagonals (q : Quadrilateral) :
  is_rhombus q → has_equal_diagonals q → is_square q := by sorry

end NUMINAMATH_CALUDE_rectangle_from_equal_bisecting_diagonals_parallelogram_from_bisecting_diagonals_square_from_rhombus_equal_diagonals_l3729_372950


namespace NUMINAMATH_CALUDE_towers_count_correct_l3729_372951

def number_of_towers (red green blue : ℕ) (height : ℕ) : ℕ :=
  let total := red + green + blue
  let leftout := total - height
  if leftout ≠ 1 then 0
  else
    (Nat.choose total height) *
    (Nat.factorial height / (Nat.factorial red * Nat.factorial (green - 1) * Nat.factorial blue) +
     Nat.factorial height / (Nat.factorial red * Nat.factorial green * Nat.factorial (blue - 1)) +
     Nat.factorial height / (Nat.factorial (red - 1) * Nat.factorial green * Nat.factorial blue))

theorem towers_count_correct :
  number_of_towers 3 4 4 10 = 26250 := by
  sorry

end NUMINAMATH_CALUDE_towers_count_correct_l3729_372951


namespace NUMINAMATH_CALUDE_homework_difference_l3729_372914

def math_homework_pages : ℕ := 3
def reading_homework_pages : ℕ := 4

theorem homework_difference : reading_homework_pages - math_homework_pages = 1 := by
  sorry

end NUMINAMATH_CALUDE_homework_difference_l3729_372914


namespace NUMINAMATH_CALUDE_semicircle_radius_l3729_372911

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  PR : ℝ
  right_angle : PQ^2 + QR^2 = PR^2

-- Define the theorem
theorem semicircle_radius (t : RightTriangle) 
  (h1 : (1/2) * π * (t.PQ/2)^2 = 18*π) 
  (h2 : π * (t.QR/2) = 10*π) : 
  t.PR/2 = 4*Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l3729_372911


namespace NUMINAMATH_CALUDE_three_digit_number_appended_l3729_372985

theorem three_digit_number_appended (n : ℕ) : 
  100 ≤ n ∧ n < 1000 → 1000 * n + n = 1001 * n := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_appended_l3729_372985


namespace NUMINAMATH_CALUDE_race_equation_theorem_l3729_372971

/-- Represents a runner's performance in a race before and after training -/
structure RunnerPerformance where
  distance : ℝ
  speedIncrease : ℝ
  timeImprovement : ℝ
  initialSpeed : ℝ

/-- Checks if the given runner performance satisfies the race equation -/
def satisfiesRaceEquation (perf : RunnerPerformance) : Prop :=
  perf.distance / perf.initialSpeed - 
  perf.distance / (perf.initialSpeed * (1 + perf.speedIncrease)) = 
  perf.timeImprovement

/-- Theorem stating that a runner with the given performance satisfies the race equation -/
theorem race_equation_theorem (perf : RunnerPerformance) 
  (h1 : perf.distance = 3000)
  (h2 : perf.speedIncrease = 0.25)
  (h3 : perf.timeImprovement = 3) :
  satisfiesRaceEquation perf := by
  sorry

end NUMINAMATH_CALUDE_race_equation_theorem_l3729_372971


namespace NUMINAMATH_CALUDE_triangle_properties_l3729_372973

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.b * Real.sin t.A = 3 * t.c * Real.sin t.B ∧
  t.a = 3 ∧
  Real.cos t.B = 2/3

theorem triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.b = Real.sqrt 6 ∧ 
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3729_372973


namespace NUMINAMATH_CALUDE_trapezium_area_l3729_372926

theorem trapezium_area (a b h θ : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 13) (hθ : θ = 30 * π / 180) :
  (a + b) / 2 * (h * Real.sin θ) = 123.5 :=
sorry

end NUMINAMATH_CALUDE_trapezium_area_l3729_372926


namespace NUMINAMATH_CALUDE_like_term_proof_l3729_372947

def is_like_term (t₁ t₂ : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ), t₁ x y = a * x^5 * y^3 ∧ t₂ x y = b * x^5 * y^3

theorem like_term_proof (a : ℝ) :
  is_like_term (λ x y => -5 * x^5 * y^3) (λ x y => a * x^5 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_like_term_proof_l3729_372947


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3729_372905

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define the set C with parameters a and b
def C (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- Theorem statement
theorem intersection_implies_sum (a b : ℝ) :
  C a b = A ∩ B → a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3729_372905


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_3_sqrt_3_over_4_l3729_372939

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition1 (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - t.b * t.c

def satisfies_condition2 (t : Triangle) : Prop :=
  t.a = Real.sqrt 7

def satisfies_condition3 (t : Triangle) : Prop :=
  t.c - t.b = 2

-- Theorem 1
theorem angle_A_is_pi_over_3 (t : Triangle) (h : satisfies_condition1 t) :
  t.A = π / 3 := by sorry

-- Theorem 2
theorem area_is_3_sqrt_3_over_4 (t : Triangle) 
  (h1 : satisfies_condition1 t) 
  (h2 : satisfies_condition2 t) 
  (h3 : satisfies_condition3 t) :
  (1/2) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_3_area_is_3_sqrt_3_over_4_l3729_372939


namespace NUMINAMATH_CALUDE_sum_every_third_odd_integer_l3729_372929

/-- The sum of every third odd integer between 200 and 500 (inclusive) is 17400 -/
theorem sum_every_third_odd_integer : 
  (Finset.range 50).sum (fun i => 201 + 6 * i) = 17400 := by
  sorry

end NUMINAMATH_CALUDE_sum_every_third_odd_integer_l3729_372929


namespace NUMINAMATH_CALUDE_john_mean_score_l3729_372923

def john_scores : List ℝ := [95, 88, 90, 92, 94, 89]

theorem john_mean_score : 
  (john_scores.sum / john_scores.length : ℝ) = 91.3333 := by
  sorry

end NUMINAMATH_CALUDE_john_mean_score_l3729_372923


namespace NUMINAMATH_CALUDE_max_value_fraction_l3729_372912

theorem max_value_fraction (x y z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) → 
  (10 ≤ y ∧ y ≤ 99) → 
  (10 ≤ z ∧ z ≤ 99) → 
  ((x + y + z) / 3 = 60) → 
  ((x + y) / z : ℚ) ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3729_372912


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3729_372943

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≥ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3729_372943


namespace NUMINAMATH_CALUDE_ones_digit_of_6_power_52_l3729_372964

theorem ones_digit_of_6_power_52 : ∃ n : ℕ, 6^52 = 10 * n + 6 :=
sorry

end NUMINAMATH_CALUDE_ones_digit_of_6_power_52_l3729_372964


namespace NUMINAMATH_CALUDE_first_runner_pace_correct_l3729_372919

/-- The average pace of the first runner in a race with the following conditions:
  * The race is 10 miles long.
  * The second runner's pace is 7 minutes per mile.
  * The second runner stops after 56 minutes.
  * The second runner could remain stopped for 8 minutes before the first runner catches up.
-/
def firstRunnerPace : ℝ :=
  let raceLength : ℝ := 10
  let secondRunnerPace : ℝ := 7
  let secondRunnerStopTime : ℝ := 56
  let catchUpTime : ℝ := 8
  
  4  -- The actual pace, to be proved

theorem first_runner_pace_correct :
  let raceLength : ℝ := 10
  let secondRunnerPace : ℝ := 7
  let secondRunnerStopTime : ℝ := 56
  let catchUpTime : ℝ := 8
  
  firstRunnerPace = 4 := by sorry

end NUMINAMATH_CALUDE_first_runner_pace_correct_l3729_372919


namespace NUMINAMATH_CALUDE_sum_of_six_least_l3729_372916

/-- τ(n) denotes the number of positive integer divisors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The set of positive integers n that satisfy τ(n) + τ(n+1) = 8 -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ tau n + tau (n + 1) = 8}

/-- The six least elements of S -/
def six_least : Finset ℕ := sorry

theorem sum_of_six_least : (six_least.sum id) = 800 := by sorry

end NUMINAMATH_CALUDE_sum_of_six_least_l3729_372916


namespace NUMINAMATH_CALUDE_min_value_expression_l3729_372959

theorem min_value_expression (x : ℝ) (h : x > 1) :
  (x + 12) / Real.sqrt (x - 1) ≥ 2 * Real.sqrt 13 ∧
  ∃ y : ℝ, y > 1 ∧ (y + 12) / Real.sqrt (y - 1) = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3729_372959


namespace NUMINAMATH_CALUDE_office_call_probabilities_l3729_372968

/-- Represents the probability of a call being for a specific person -/
structure CallProbability where
  A : ℚ
  B : ℚ
  C : ℚ
  sum_to_one : A + B + C = 1

/-- Calculates the probability of all three calls being for the same person -/
def prob_all_same (p : CallProbability) : ℚ :=
  p.A^3 + p.B^3 + p.C^3

/-- Calculates the probability of exactly two out of three calls being for A -/
def prob_two_for_A (p : CallProbability) : ℚ :=
  3 * p.A^2 * (1 - p.A)

theorem office_call_probabilities :
  ∃ (p : CallProbability),
    p.A = 1/6 ∧ p.B = 1/3 ∧ p.C = 1/2 ∧
    prob_all_same p = 1/6 ∧
    prob_two_for_A p = 5/72 := by
  sorry

end NUMINAMATH_CALUDE_office_call_probabilities_l3729_372968


namespace NUMINAMATH_CALUDE_unique_prime_power_sum_l3729_372995

theorem unique_prime_power_sum (p q : ℕ) : 
  Prime p → Prime q → Prime (p^q + q^p) → (p = 2 ∧ q = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_prime_power_sum_l3729_372995


namespace NUMINAMATH_CALUDE_solution_to_equation_l3729_372989

theorem solution_to_equation (x : ℝ) (hx : x ≠ 0) :
  (3 * x)^5 = (9 * x)^4 → x = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3729_372989


namespace NUMINAMATH_CALUDE_probability_A_hits_twice_B_hits_thrice_l3729_372933

def probability_A_hits : ℚ := 2/3
def probability_B_hits : ℚ := 3/4
def num_shots : ℕ := 4
def num_A_hits : ℕ := 2
def num_B_hits : ℕ := 3

theorem probability_A_hits_twice_B_hits_thrice : 
  (Nat.choose num_shots num_A_hits * probability_A_hits ^ num_A_hits * (1 - probability_A_hits) ^ (num_shots - num_A_hits)) *
  (Nat.choose num_shots num_B_hits * probability_B_hits ^ num_B_hits * (1 - probability_B_hits) ^ (num_shots - num_B_hits)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_A_hits_twice_B_hits_thrice_l3729_372933


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3729_372969

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3729_372969


namespace NUMINAMATH_CALUDE_worker_daily_rate_l3729_372992

/-- Proves that a worker's daily rate is $150 given the specified conditions -/
theorem worker_daily_rate (daily_rate : ℝ) (overtime_rate : ℝ) (total_days : ℝ) 
  (overtime_hours : ℝ) (total_pay : ℝ) : 
  overtime_rate = 5 →
  total_days = 5 →
  overtime_hours = 4 →
  total_pay = 770 →
  total_pay = daily_rate * total_days + overtime_rate * overtime_hours →
  daily_rate = 150 := by
  sorry

end NUMINAMATH_CALUDE_worker_daily_rate_l3729_372992


namespace NUMINAMATH_CALUDE_women_average_age_l3729_372961

/-- The average age of two women given the following conditions:
    1. There are initially 10 men.
    2. When two women replace two men (aged 10 and 12), the average age increases by 2 years.
    3. The number of people remains 10 after the replacement. -/
theorem women_average_age (T : ℕ) : 
  (T : ℝ) / 10 + 2 = (T - 10 - 12 + 42) / 10 → 21 = 42 / 2 := by
  sorry

end NUMINAMATH_CALUDE_women_average_age_l3729_372961


namespace NUMINAMATH_CALUDE_expression_equals_three_l3729_372920

theorem expression_equals_three : 
  (1/2)⁻¹ + 4 * Real.cos (45 * π / 180) - Real.sqrt 8 + (2023 - Real.pi)^0 = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_three_l3729_372920


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3729_372906

/-- Quadratic function -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

theorem quadratic_function_properties (b c : ℝ) :
  (∀ x, f b c x ≥ f b c 1) →  -- minimum at x = 1
  f b c 1 = 3 →              -- minimum value is 3
  f b c 2 = 4 →              -- f(2) = 4
  b = -2 ∧ c = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3729_372906


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3729_372972

theorem area_between_concentric_circles
  (r_small : ℝ) (r_large : ℝ) (h1 : r_small * 2 = 6)
  (h2 : r_large = 3 * r_small) :
  π * r_large^2 - π * r_small^2 = 72 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3729_372972


namespace NUMINAMATH_CALUDE_min_pumps_needed_l3729_372930

/-- Represents the water pumping scenario -/
structure WaterPumping where
  x : ℝ  -- Amount of water already gushed out before pumping
  a : ℝ  -- Amount of water gushing out per minute
  b : ℝ  -- Amount of water each pump can pump out per minute

/-- The conditions of the water pumping problem -/
def water_pumping_conditions (w : WaterPumping) : Prop :=
  w.x + 40 * w.a = 2 * 40 * w.b ∧
  w.x + 16 * w.a = 4 * 16 * w.b ∧
  w.a > 0 ∧ w.b > 0

/-- The theorem stating the minimum number of pumps needed -/
theorem min_pumps_needed (w : WaterPumping) 
  (h : water_pumping_conditions w) : 
  ∀ n : ℕ, (w.x + 10 * w.a ≤ 10 * n * w.b) → n ≥ 6 := by
  sorry

#check min_pumps_needed

end NUMINAMATH_CALUDE_min_pumps_needed_l3729_372930


namespace NUMINAMATH_CALUDE_volume_ratio_is_twenty_l3729_372952

-- Define the dimensions of the boxes
def sehee_side : ℝ := 1  -- 1 meter
def serin_width : ℝ := 0.5  -- 50 cm in meters
def serin_depth : ℝ := 0.5  -- 50 cm in meters
def serin_height : ℝ := 0.2  -- 20 cm in meters

-- Define the volumes of the boxes
def sehee_volume : ℝ := sehee_side ^ 3
def serin_volume : ℝ := serin_width * serin_depth * serin_height

-- State the theorem
theorem volume_ratio_is_twenty :
  sehee_volume / serin_volume = 20 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_is_twenty_l3729_372952


namespace NUMINAMATH_CALUDE_workers_in_second_group_l3729_372999

theorem workers_in_second_group 
  (wages_group1 : ℕ) 
  (workers_group1 : ℕ) 
  (days_group1 : ℕ) 
  (wages_group2 : ℕ) 
  (days_group2 : ℕ) 
  (h1 : wages_group1 = 9450) 
  (h2 : workers_group1 = 15) 
  (h3 : days_group1 = 6) 
  (h4 : wages_group2 = 9975) 
  (h5 : days_group2 = 5) : 
  (wages_group2 / (wages_group1 / (workers_group1 * days_group1) * days_group2)) = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_workers_in_second_group_l3729_372999


namespace NUMINAMATH_CALUDE_sqrt_trig_identity_l3729_372978

theorem sqrt_trig_identity : 
  Real.sqrt (1 - 2 * Real.cos (π / 2 + 3) * Real.sin (π / 2 - 3)) = -Real.sin 3 - Real.cos 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_trig_identity_l3729_372978


namespace NUMINAMATH_CALUDE_emmas_garden_area_l3729_372901

theorem emmas_garden_area :
  ∀ (short_posts long_posts : ℕ) (short_side long_side : ℝ),
  short_posts > 1 ∧
  long_posts > 1 ∧
  short_posts + long_posts = 12 ∧
  long_posts = 3 * short_posts ∧
  short_side = 6 * (short_posts - 1) ∧
  long_side = 6 * (long_posts - 1) →
  short_side * long_side = 576 := by
sorry

end NUMINAMATH_CALUDE_emmas_garden_area_l3729_372901


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l3729_372913

theorem hyperbola_m_range (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m - y^2 / (2*m - 1) = 1) → 
  (0 < m ∧ m < 1/2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l3729_372913


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3729_372984

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_sum (a₁ r : ℝ) (h₁ : a₁ = 1) (h₂ : r = -3) :
  let a := geometric_sequence a₁ r
  a 1 + |a 2| + a 3 + |a 4| + a 5 = 121 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3729_372984


namespace NUMINAMATH_CALUDE_quinn_reading_challenge_l3729_372958

/-- The number of books Quinn needs to read to get one free donut -/
def books_per_donut (books_per_week : ℕ) (weeks : ℕ) (total_donuts : ℕ) : ℕ :=
  (books_per_week * weeks) / total_donuts

/-- Proof that Quinn needs to read 5 books to get one free donut -/
theorem quinn_reading_challenge :
  books_per_donut 2 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quinn_reading_challenge_l3729_372958


namespace NUMINAMATH_CALUDE_complex_number_additive_inverse_l3729_372944

theorem complex_number_additive_inverse (b : ℝ) : 
  let z : ℂ := (2 - b * Complex.I) / (1 + 2 * Complex.I)
  (z.re = -z.im) → b = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_additive_inverse_l3729_372944


namespace NUMINAMATH_CALUDE_wire_cutting_l3729_372996

/-- Given a wire of length 50 feet cut into three pieces, prove the lengths of the pieces. -/
theorem wire_cutting (x : ℝ) 
  (h1 : x + (x + 2) + (2*x - 3) = 50) -- Total length equation
  (h2 : x > 0) -- Ensure positive length
  : x = 12.75 ∧ x + 2 = 14.75 ∧ 2*x - 3 = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l3729_372996


namespace NUMINAMATH_CALUDE_number_division_remainder_l3729_372960

theorem number_division_remainder (N : ℤ) (D : ℤ) : 
  N % 281 = 160 → N % D = 21 → D = 139 := by
  sorry

end NUMINAMATH_CALUDE_number_division_remainder_l3729_372960


namespace NUMINAMATH_CALUDE_unique_partition_l3729_372975

/-- Represents the number of caps collected by each girl -/
def caps : List Nat := [20, 29, 31, 49, 51]

/-- Represents a partition of the caps into two boxes -/
structure Partition where
  red : List Nat
  blue : List Nat
  sum_red : red.sum = 60
  sum_blue : blue.sum = 120
  partition_complete : red ++ blue = caps

/-- The theorem to be proved -/
theorem unique_partition : ∃! p : Partition, True := by sorry

end NUMINAMATH_CALUDE_unique_partition_l3729_372975


namespace NUMINAMATH_CALUDE_gumball_packages_l3729_372980

theorem gumball_packages (gumballs_per_package : ℕ) (gumballs_eaten : ℕ) : 
  gumballs_per_package = 5 → gumballs_eaten = 20 → 
  (gumballs_eaten / gumballs_per_package : ℕ) = 4 := by
sorry

end NUMINAMATH_CALUDE_gumball_packages_l3729_372980


namespace NUMINAMATH_CALUDE_monkey_peach_problem_l3729_372948

/-- The number of peaches the monkey's mother originally had -/
def mothers_original_peaches (little_monkey_initial : ℕ) (peaches_given : ℕ) (mother_ratio : ℕ) : ℕ :=
  (little_monkey_initial + peaches_given) * mother_ratio + peaches_given

theorem monkey_peach_problem :
  mothers_original_peaches 6 3 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_monkey_peach_problem_l3729_372948


namespace NUMINAMATH_CALUDE_number_ordering_l3729_372993

theorem number_ordering : (6 : ℝ)^10 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l3729_372993


namespace NUMINAMATH_CALUDE_polygon_area_l3729_372956

/-- A polygon on a unit grid with vertices at (0,0), (5,0), (5,5), (0,5), (5,10), (0,10), (0,0) -/
def polygon : List (ℤ × ℤ) := [(0,0), (5,0), (5,5), (0,5), (5,10), (0,10), (0,0)]

/-- The area enclosed by the polygon -/
def enclosed_area (p : List (ℤ × ℤ)) : ℚ := sorry

/-- Theorem stating that the area enclosed by the polygon is 37.5 square units -/
theorem polygon_area : enclosed_area polygon = 37.5 := by sorry

end NUMINAMATH_CALUDE_polygon_area_l3729_372956


namespace NUMINAMATH_CALUDE_sandy_fish_count_l3729_372981

def initial_fish : ℕ := 26
def bought_fish : ℕ := 6

theorem sandy_fish_count : initial_fish + bought_fish = 32 := by
  sorry

end NUMINAMATH_CALUDE_sandy_fish_count_l3729_372981


namespace NUMINAMATH_CALUDE_line_segment_param_sum_of_squares_l3729_372915

/-- Given a line segment connecting (-3,9) and (2,12), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (-3,9), prove that a^2 + b^2 + c^2 + d^2 = 124 -/
theorem line_segment_param_sum_of_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = -3 ∧ d = 9) →
  (a + b = 2 ∧ c + d = 12) →
  a^2 + b^2 + c^2 + d^2 = 124 :=
by sorry


end NUMINAMATH_CALUDE_line_segment_param_sum_of_squares_l3729_372915


namespace NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l3729_372954

def y : ℕ := 3^(4^(5^(6^(7^(8^(9^10))))))

theorem smallest_perfect_square_multiplier :
  ∃ (k : ℕ), k > 0 ∧ 
  (∃ (n : ℕ), k * y = n^2) ∧
  (∀ (m : ℕ), m > 0 → m < k → ¬∃ (n : ℕ), m * y = n^2) ∧
  k = 75 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_multiplier_l3729_372954


namespace NUMINAMATH_CALUDE_closest_to_fraction_l3729_372922

def fraction : ℚ := 501 / (1 / 4)

def options : List ℤ := [1800, 1900, 2000, 2100, 2200]

theorem closest_to_fraction :
  (2000 : ℤ) = (options.argmin (λ x => |↑x - fraction|)).get
    (by sorry) := by sorry

end NUMINAMATH_CALUDE_closest_to_fraction_l3729_372922


namespace NUMINAMATH_CALUDE_sin_cos_sum_20_10_l3729_372963

theorem sin_cos_sum_20_10 : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_20_10_l3729_372963


namespace NUMINAMATH_CALUDE_equation_transformation_l3729_372925

theorem equation_transformation (x y : ℝ) : x = y → -2 * x = -2 * y := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l3729_372925


namespace NUMINAMATH_CALUDE_range_of_f_l3729_372935

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3729_372935


namespace NUMINAMATH_CALUDE_no_zonk_probability_l3729_372994

theorem no_zonk_probability : 
  let num_tables : ℕ := 3
  let boxes_per_table : ℕ := 3
  let prob_no_zonk_per_table : ℚ := 2 / 3
  (prob_no_zonk_per_table ^ num_tables : ℚ) = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_no_zonk_probability_l3729_372994


namespace NUMINAMATH_CALUDE_T_is_far_right_l3729_372982

/-- Represents a rectangle with four integer-labeled sides --/
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

/-- Checks if a rectangle is at the far-right end of the row --/
def is_far_right (r : Rectangle) (others : List Rectangle) : Prop :=
  ∀ other ∈ others, r.y ≥ other.y ∧ (r.y = other.y → r.w ≥ other.w)

/-- The given rectangles --/
def P : Rectangle := ⟨3, 0, 9, 5⟩
def Q : Rectangle := ⟨6, 1, 0, 8⟩
def R : Rectangle := ⟨0, 3, 2, 7⟩
def S : Rectangle := ⟨8, 5, 4, 1⟩
def T : Rectangle := ⟨5, 2, 6, 9⟩

theorem T_is_far_right :
  is_far_right T [P, Q, R, S] :=
sorry

end NUMINAMATH_CALUDE_T_is_far_right_l3729_372982


namespace NUMINAMATH_CALUDE_two_solutions_exist_l3729_372983

/-- A structure representing a triangle with integer side lengths -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a.val < b.val + c.val ∧ b.val < a.val + c.val ∧ c.val < a.val + b.val

/-- The condition from the original problem -/
def satisfies_equation (t : IntegerTriangle) : Prop :=
  (t.a.val * t.b.val * t.c.val : ℕ) = 2 * (t.a.val - 1) * (t.b.val - 1) * (t.c.val - 1)

/-- The main theorem stating that there are exactly two solutions -/
theorem two_solutions_exist : 
  (∃ (t1 t2 : IntegerTriangle), 
    satisfies_equation t1 ∧ 
    satisfies_equation t2 ∧ 
    t1 ≠ t2 ∧ 
    (∀ (t : IntegerTriangle), satisfies_equation t → (t = t1 ∨ t = t2))) ∧
  (∃ (t1 : IntegerTriangle), t1.a = 8 ∧ t1.b = 7 ∧ t1.c = 3 ∧ satisfies_equation t1) ∧
  (∃ (t2 : IntegerTriangle), t2.a = 6 ∧ t2.b = 5 ∧ t2.c = 4 ∧ satisfies_equation t2) :=
by sorry


end NUMINAMATH_CALUDE_two_solutions_exist_l3729_372983


namespace NUMINAMATH_CALUDE_no_integer_square_root_l3729_372924

-- Define the polynomial Q
def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 25

-- Theorem statement
theorem no_integer_square_root : ∀ x : ℤ, ¬∃ y : ℤ, Q x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l3729_372924


namespace NUMINAMATH_CALUDE_digit_sum_difference_l3729_372965

-- Define a function to calculate the sum of digits of a number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := sorry

-- Define the sum of digits for all even numbers from 1 to 1000
def sumEvenDigits : ℕ := 
  (List.range 1000).filter isEven |>.map sumOfDigits |>.sum

-- Define the sum of digits for all odd numbers from 1 to 1000
def sumOddDigits : ℕ := 
  (List.range 1000).filter (λ n => ¬(isEven n)) |>.map sumOfDigits |>.sum

-- Theorem statement
theorem digit_sum_difference :
  sumOddDigits - sumEvenDigits = 499 := by sorry

end NUMINAMATH_CALUDE_digit_sum_difference_l3729_372965


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l3729_372988

theorem stratified_sampling_medium_stores 
  (total_stores : ℕ) 
  (medium_stores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (sample_size * medium_stores) / total_stores = 5 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_l3729_372988


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3729_372953

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_implies_planes_perpendicular_l3729_372953


namespace NUMINAMATH_CALUDE_crickets_to_collect_l3729_372904

theorem crickets_to_collect (collected : ℕ) (target : ℕ) (additional : ℕ) : 
  collected = 7 → target = 11 → additional = target - collected :=
by
  sorry

end NUMINAMATH_CALUDE_crickets_to_collect_l3729_372904


namespace NUMINAMATH_CALUDE_tree_spacing_l3729_372918

/-- Given a road of length 151 feet where 11 trees can be planted, with each tree occupying 1 foot of space, 
    the distance between each tree is 14 feet. -/
theorem tree_spacing (road_length : ℕ) (num_trees : ℕ) (tree_space : ℕ) 
    (h1 : road_length = 151)
    (h2 : num_trees = 11)
    (h3 : tree_space = 1) : 
  (road_length - num_trees * tree_space) / (num_trees - 1) = 14 := by
  sorry


end NUMINAMATH_CALUDE_tree_spacing_l3729_372918


namespace NUMINAMATH_CALUDE_fraction_product_equals_one_over_23426_l3729_372928

def fraction_product : ℕ → ℚ
  | 0 => 1
  | n + 1 => (n + 1 : ℚ) / (n + 5 : ℚ) * fraction_product n

theorem fraction_product_equals_one_over_23426 :
  fraction_product 49 = 1 / 23426 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_one_over_23426_l3729_372928


namespace NUMINAMATH_CALUDE_intersection_M_N_l3729_372949

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x : ℤ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3729_372949


namespace NUMINAMATH_CALUDE_arcsin_symmetry_l3729_372991

theorem arcsin_symmetry (x : ℝ) (h : x ∈ Set.Icc (-1) 1) :
  Real.arcsin (-x) = -Real.arcsin x := by
  sorry

end NUMINAMATH_CALUDE_arcsin_symmetry_l3729_372991


namespace NUMINAMATH_CALUDE_trapezium_height_l3729_372976

theorem trapezium_height (a b h : ℝ) : 
  a = 20 → b = 18 → (1/2) * (a + b) * h = 247 → h = 13 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_height_l3729_372976


namespace NUMINAMATH_CALUDE_tom_initial_dimes_l3729_372909

/-- Represents the number of coins Tom has -/
structure TomCoins where
  initial_pennies : ℕ
  initial_dimes : ℕ
  dad_dimes : ℕ
  dad_nickels : ℕ
  final_dimes : ℕ

/-- The theorem states that given the conditions from the problem,
    Tom's initial number of dimes was 15 -/
theorem tom_initial_dimes (coins : TomCoins)
  (h1 : coins.initial_pennies = 27)
  (h2 : coins.dad_dimes = 33)
  (h3 : coins.dad_nickels = 49)
  (h4 : coins.final_dimes = 48)
  (h5 : coins.final_dimes = coins.initial_dimes + coins.dad_dimes) :
  coins.initial_dimes = 15 := by
  sorry


end NUMINAMATH_CALUDE_tom_initial_dimes_l3729_372909


namespace NUMINAMATH_CALUDE_point_division_l3729_372979

/-- Given two points A and B in a vector space, and a point P on the line segment AB
    such that AP:PB = 4:1, prove that P = (4/5)*A + (1/5)*B -/
theorem point_division (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (A B P : V) (h : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) 
  (h_ratio : ∃ (k : ℝ), k > 0 ∧ P - A = (4 * k) • (B - A) ∧ B - P = k • (B - A)) :
  P = (4/5) • A + (1/5) • B := by
  sorry

end NUMINAMATH_CALUDE_point_division_l3729_372979


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3729_372936

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

/-- Checks if three numbers form an arithmetic progression -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  b - a = c - b

/-- Reverses a three-digit number -/
def reverse_number (n : ℕ) : ℕ :=
  (n % 10) * 100 + ((n / 10) % 10) * 10 + (n / 100)

theorem unique_number_satisfying_conditions : ∃! n : ℕ,
  100 ≤ n ∧ n < 1000 ∧
  is_geometric_progression (n / 100) ((n / 10) % 10) (n % 10) ∧
  n - 792 = reverse_number n ∧
  is_arithmetic_progression ((n / 100) - 4) ((n / 10) % 10) (n % 10) ∧
  n = 931 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3729_372936


namespace NUMINAMATH_CALUDE_triangle_area_l3729_372902

/-- The area of a triangle with vertices at (3, -3), (3, 4), and (8, -3) is 17.5 square units -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (3, -3)
  let v2 : ℝ × ℝ := (3, 4)
  let v3 : ℝ × ℝ := (8, -3)
  let area := abs ((v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)) / 2)
  area = 17.5 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l3729_372902


namespace NUMINAMATH_CALUDE_expression_value_l3729_372908

theorem expression_value (a b m n x : ℝ) 
  (h1 : a = -b) 
  (h2 : m * n = 1) 
  (h3 : |x| = 2) : 
  -2*m*n + 3*(a+b) - x = -4 ∨ -2*m*n + 3*(a+b) - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3729_372908


namespace NUMINAMATH_CALUDE_subtract_negative_l3729_372962

theorem subtract_negative : 2 - (-3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l3729_372962


namespace NUMINAMATH_CALUDE_charlies_laps_l3729_372932

/-- Given Charlie's steps per lap and total steps in a session, calculate the number of complete laps --/
theorem charlies_laps (steps_per_lap : ℕ) (total_steps : ℕ) : 
  steps_per_lap = 5350 → total_steps = 13375 → (total_steps / steps_per_lap : ℕ) = 2 :=
by
  sorry

#eval (13375 / 5350 : ℕ)

end NUMINAMATH_CALUDE_charlies_laps_l3729_372932


namespace NUMINAMATH_CALUDE_second_box_clay_capacity_l3729_372974

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ := d.height * d.width * d.length

/-- The dimensions of the first box -/
def firstBox : BoxDimensions := {
  height := 3,
  width := 4,
  length := 7
}

/-- The dimensions of the second box -/
def secondBox : BoxDimensions := {
  height := 3 * firstBox.height,
  width := 2 * firstBox.width,
  length := firstBox.length
}

/-- The amount of clay the first box can hold in grams -/
def firstBoxClay : ℝ := 70

/-- Theorem: The second box can hold 420 grams of clay -/
theorem second_box_clay_capacity : 
  (boxVolume secondBox / boxVolume firstBox) * firstBoxClay = 420 := by sorry

end NUMINAMATH_CALUDE_second_box_clay_capacity_l3729_372974


namespace NUMINAMATH_CALUDE_read_book_in_seven_weeks_l3729_372945

/-- The number of weeks required to read a book given the total pages and pages read per week. -/
def weeks_to_read (total_pages : ℕ) (pages_per_week : ℕ) : ℕ :=
  (total_pages + pages_per_week - 1) / pages_per_week

/-- Theorem stating that it takes 7 weeks to read a 2100-page book at a rate of 300 pages per week. -/
theorem read_book_in_seven_weeks :
  let total_pages : ℕ := 2100
  let pages_per_day : ℕ := 100
  let days_per_week : ℕ := 3
  let pages_per_week : ℕ := pages_per_day * days_per_week
  weeks_to_read total_pages pages_per_week = 7 := by
  sorry

end NUMINAMATH_CALUDE_read_book_in_seven_weeks_l3729_372945


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3729_372900

theorem complex_magnitude_problem (z : ℂ) (h : (1 - 2*I) * z = 5*I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3729_372900


namespace NUMINAMATH_CALUDE_tuesday_necklaces_l3729_372987

/-- The number of beaded necklaces Kylie made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded bracelets Kylie made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings Kylie made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads Kylie used to make her jewelry -/
def total_beads : ℕ := 325

/-- Theorem: The number of beaded necklaces Kylie made on Tuesday is 2 -/
theorem tuesday_necklaces : 
  (total_beads - (monday_necklaces * beads_per_necklace + 
    wednesday_bracelets * beads_per_bracelet + 
    wednesday_earrings * beads_per_earring)) / beads_per_necklace = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuesday_necklaces_l3729_372987


namespace NUMINAMATH_CALUDE_percentage_of_hindu_boys_l3729_372977

-- Define the total number of boys
def total_boys : ℕ := 700

-- Define the percentage of Muslim boys
def muslim_percentage : ℚ := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage : ℚ := 10 / 100

-- Define the number of boys from other communities
def other_boys : ℕ := 126

-- Define the percentage of Hindu boys
def hindu_percentage : ℚ := 28 / 100

-- Theorem statement
theorem percentage_of_hindu_boys :
  hindu_percentage * total_boys =
    total_boys - (muslim_percentage * total_boys + sikh_percentage * total_boys + other_boys) :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_hindu_boys_l3729_372977


namespace NUMINAMATH_CALUDE_joey_age_l3729_372966

theorem joey_age : 
  let ages : List ℕ := [3, 5, 7, 9, 11, 13]
  let movie_pair : ℕ × ℕ := (3, 13)
  let baseball_pair : ℕ × ℕ := (7, 9)
  let stay_home : ℕ × ℕ := (5, 11)
  (∀ (a b : ℕ), a ∈ ages ∧ b ∈ ages ∧ a + b = 16 → (a, b) = movie_pair) ∧
  (∀ (a b : ℕ), a ∈ ages ∧ b ∈ ages ∧ a < 10 ∧ b < 10 ∧ (a, b) ≠ movie_pair → (a, b) = baseball_pair) ∧
  (∀ (a : ℕ), a ∈ ages ∧ a ∉ [movie_pair.1, movie_pair.2, baseball_pair.1, baseball_pair.2, 5] → a = 11) →
  stay_home.2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_joey_age_l3729_372966


namespace NUMINAMATH_CALUDE_cliff_rock_collection_l3729_372921

theorem cliff_rock_collection :
  let total_rocks : ℕ := 180
  let sedimentary_rocks : ℕ := total_rocks * 2 / 3
  let igneous_rocks : ℕ := sedimentary_rocks / 2
  let shiny_igneous_ratio : ℚ := 2 / 3
  shiny_igneous_ratio * igneous_rocks = 40 := by
  sorry

end NUMINAMATH_CALUDE_cliff_rock_collection_l3729_372921


namespace NUMINAMATH_CALUDE_curve_C_symmetric_about_y_axis_l3729_372927

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | |p.2| = Real.sqrt (p.1^2 + (p.2 - 4)^2)}

-- Define symmetry about y-axis
def symmetric_about_y_axis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ S ↔ (-x, y) ∈ S

-- Theorem statement
theorem curve_C_symmetric_about_y_axis : symmetric_about_y_axis C := by
  sorry

end NUMINAMATH_CALUDE_curve_C_symmetric_about_y_axis_l3729_372927


namespace NUMINAMATH_CALUDE_remainder_theorem_l3729_372931

theorem remainder_theorem (r : ℤ) : (r^17 + 1) % (r - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3729_372931


namespace NUMINAMATH_CALUDE_prime_square_diff_divisibility_l3729_372970

theorem prime_square_diff_divisibility (p q : ℕ) (k : ℤ) : 
  Prime p → Prime q → p > 5 → q > 5 → p ≠ q → 
  (p^2 : ℤ) - (q^2 : ℤ) = 6 * k → 
  (p^2 : ℤ) - (q^2 : ℤ) ≡ 0 [ZMOD 24] :=
sorry

end NUMINAMATH_CALUDE_prime_square_diff_divisibility_l3729_372970


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_estate_area_l3729_372986

/-- Represents the scale of the map --/
def map_scale : ℚ := 500 / 2

/-- Represents the length of the diagonals on the map in inches --/
def diagonal_length : ℚ := 10

/-- Calculates the actual length of the diagonal in miles --/
def actual_diagonal_length : ℚ := diagonal_length * map_scale

/-- Represents an isosceles trapezoid estate --/
structure IsoscelesTrapezoidEstate where
  diagonal : ℚ
  area : ℚ

/-- Theorem stating the area of the isosceles trapezoid estate --/
theorem isosceles_trapezoid_estate_area :
  ∃ (estate : IsoscelesTrapezoidEstate),
    estate.diagonal = actual_diagonal_length ∧
    estate.area = 3125000 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_trapezoid_estate_area_l3729_372986


namespace NUMINAMATH_CALUDE_fuchsia_survey_l3729_372942

/-- Given a survey about the color fuchsia with the following parameters:
  * total_surveyed: Total number of people surveyed
  * mostly_pink: Number of people who believe fuchsia is "mostly pink"
  * both: Number of people who believe fuchsia is both "mostly pink" and "mostly purple"
  * neither: Number of people who believe fuchsia is neither "mostly pink" nor "mostly purple"

  This theorem proves that the number of people who believe fuchsia is "mostly purple"
  is equal to total_surveyed - (mostly_pink - both) - neither.
-/
theorem fuchsia_survey (total_surveyed mostly_pink both neither : ℕ)
  (h1 : total_surveyed = 150)
  (h2 : mostly_pink = 80)
  (h3 : both = 40)
  (h4 : neither = 25) :
  total_surveyed - (mostly_pink - both) - neither = 85 := by
  sorry

#check fuchsia_survey

end NUMINAMATH_CALUDE_fuchsia_survey_l3729_372942
