import Mathlib

namespace NUMINAMATH_CALUDE_brokerage_percentage_calculation_l1763_176350

/-- The brokerage percentage calculation problem -/
theorem brokerage_percentage_calculation
  (cash_realized : ℝ)
  (total_amount : ℝ)
  (h1 : cash_realized = 106.25)
  (h2 : total_amount = 106) :
  let brokerage_amount := cash_realized - total_amount
  let brokerage_percentage := (brokerage_amount / total_amount) * 100
  ∃ ε > 0, abs (brokerage_percentage - 0.236) < ε :=
by sorry

end NUMINAMATH_CALUDE_brokerage_percentage_calculation_l1763_176350


namespace NUMINAMATH_CALUDE_remainder_of_1235678901_mod_101_l1763_176395

theorem remainder_of_1235678901_mod_101 : 1235678901 % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1235678901_mod_101_l1763_176395


namespace NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l1763_176356

theorem factorial_squared_greater_than_power (n : ℕ) (h : n > 2) : (n.factorial ^ 2 : ℝ) > n ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_squared_greater_than_power_l1763_176356


namespace NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l1763_176369

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (planeParallelPlane : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_necessary_not_sufficient
  (α β : Plane) (l : Line) (h : subset l α) :
  (lineParallelPlane l β → planeParallelPlane α β) ∧
  ¬(planeParallelPlane α β → lineParallelPlane l β) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_necessary_not_sufficient_l1763_176369


namespace NUMINAMATH_CALUDE_speed_equivalence_l1763_176355

/-- Proves that a speed of 12/36 m/s is equivalent to 1.2 km/h -/
theorem speed_equivalence : ∀ (x : ℚ), x = 12 / 36 → x * (3600 / 1000) = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l1763_176355


namespace NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l1763_176308

/-- Given a point A with coordinates (3, y) and its reflection B over the x-axis,
    prove that the sum of all coordinates of A and B is 6. -/
theorem sum_of_coordinates_after_reflection (y : ℝ) : 
  let A : ℝ × ℝ := (3, y)
  let B : ℝ × ℝ := (3, -y)  -- reflection of A over x-axis
  (A.1 + A.2 + B.1 + B.2) = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_after_reflection_l1763_176308


namespace NUMINAMATH_CALUDE_total_gulbis_count_l1763_176317

/-- The number of dureums of gulbis -/
def num_dureums : ℕ := 156

/-- The number of gulbis in one dureum -/
def gulbis_per_dureum : ℕ := 20

/-- The total number of gulbis -/
def total_gulbis : ℕ := num_dureums * gulbis_per_dureum

theorem total_gulbis_count : total_gulbis = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_gulbis_count_l1763_176317


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1763_176337

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (3 - 5 * x > 22) → x ≤ -4 ∧ (3 - 5 * (-4) > 22) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1763_176337


namespace NUMINAMATH_CALUDE_offspring_trisomy_is_heritable_variation_l1763_176375

-- Define the genotype structure
structure Genotype where
  allele1 : Char
  allele2 : Char
  allele3 : Char
  allele4 : Char

-- Define the chromosome structure
structure Chromosome where
  gene1 : Char
  gene2 : Char

-- Define the diploid tomato
def diploidTomato : Genotype := { allele1 := 'A', allele2 := 'a', allele3 := 'B', allele4 := 'b' }

-- Define the offspring with trisomy
def offspringTrisomy : Genotype := { allele1 := 'A', allele2 := 'a', allele3 := 'B', allele4 := 'b' }

-- Define the property of genes being on different homologous chromosomes
def genesOnDifferentChromosomes (g : Genotype) : Prop :=
  ∃ (c1 c2 : Chromosome), (c1.gene1 = g.allele1 ∧ c1.gene2 = g.allele3) ∧
                          (c2.gene1 = g.allele2 ∧ c2.gene2 = g.allele4)

-- Define heritable variation
def heritableVariation (parent offspring : Genotype) : Prop :=
  parent ≠ offspring ∧ ∃ (gene : Char), (gene ∈ [parent.allele1, parent.allele2, parent.allele3, parent.allele4]) ∧
                                        (gene ∈ [offspring.allele1, offspring.allele2, offspring.allele3, offspring.allele4])

-- Theorem statement
theorem offspring_trisomy_is_heritable_variation :
  genesOnDifferentChromosomes diploidTomato →
  heritableVariation diploidTomato offspringTrisomy :=
by sorry

end NUMINAMATH_CALUDE_offspring_trisomy_is_heritable_variation_l1763_176375


namespace NUMINAMATH_CALUDE_special_gp_common_ratio_l1763_176336

/-- A geometric progression with positive terms where any term minus the next term 
    equals half the sum of the next two terms. -/
structure SpecialGeometricProgression where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  a_pos : 0 < a
  r_pos : 0 < r
  special_property : ∀ n : ℕ, a * r^n - a * r^(n+1) = (1/2) * (a * r^(n+1) + a * r^(n+2))

/-- The common ratio of a special geometric progression is (√17 - 3) / 2. -/
theorem special_gp_common_ratio (gp : SpecialGeometricProgression) : 
  gp.r = (Real.sqrt 17 - 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_special_gp_common_ratio_l1763_176336


namespace NUMINAMATH_CALUDE_regular_octagon_perimeter_regular_octagon_perimeter_three_l1763_176377

/-- The perimeter of a regular octagon with side length 3 is 24 -/
theorem regular_octagon_perimeter : ℕ → ℕ
  | side_length =>
    8 * side_length

theorem regular_octagon_perimeter_three : regular_octagon_perimeter 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_perimeter_regular_octagon_perimeter_three_l1763_176377


namespace NUMINAMATH_CALUDE_binomial_16_13_l1763_176326

theorem binomial_16_13 : Nat.choose 16 13 = 560 := by sorry

end NUMINAMATH_CALUDE_binomial_16_13_l1763_176326


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1763_176380

/-- The eccentricity of a hyperbola given its equation and asymptote angle -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1 →
  (angle_between_asymptotes : ℝ) → angle_between_asymptotes = π / 3 →
  ∃ (e : ℝ), (e = 2*Real.sqrt 3/3 ∨ e = 2) ∧ 
  e^2 * a^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1763_176380


namespace NUMINAMATH_CALUDE_probability_same_color_is_two_fifths_l1763_176359

/-- Represents the number of white balls in the box -/
def num_white_balls : ℕ := 3

/-- Represents the number of black balls in the box -/
def num_black_balls : ℕ := 2

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- Calculates the number of ways to choose 2 balls from the total number of balls -/
def total_ways_to_choose : ℕ := total_balls.choose 2

/-- Calculates the number of ways to choose 2 white balls -/
def ways_to_choose_white : ℕ := num_white_balls.choose 2

/-- Calculates the number of ways to choose 2 black balls -/
def ways_to_choose_black : ℕ := num_black_balls.choose 2

/-- Calculates the total number of ways to choose 2 balls of the same color -/
def same_color_ways : ℕ := ways_to_choose_white + ways_to_choose_black

/-- The probability of drawing two balls of the same color -/
def probability_same_color : ℚ := same_color_ways / total_ways_to_choose

theorem probability_same_color_is_two_fifths :
  probability_same_color = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_is_two_fifths_l1763_176359


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1763_176357

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The value of k for which the lines y = 5x + 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1763_176357


namespace NUMINAMATH_CALUDE_average_marks_combined_classes_l1763_176320

theorem average_marks_combined_classes (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 24) (h₂ : n₂ = 50) (h₃ : avg₁ = 40) (h₄ : avg₂ = 60) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = 53.51 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_classes_l1763_176320


namespace NUMINAMATH_CALUDE_optimal_sampling_for_populations_l1763_176384

/-- Represents different sampling methods -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- Represents a population with its characteristics -/
structure Population where
  total : ℕ
  subgroups : List ℕ
  has_distinct_subgroups : Bool

/-- Determines the optimal sampling method for a given population -/
def optimal_sampling_method (pop : Population) : SamplingMethod :=
  if pop.has_distinct_subgroups then
    SamplingMethod.Stratified
  else
    SamplingMethod.Random

/-- The main theorem stating the optimal sampling methods for given populations -/
theorem optimal_sampling_for_populations 
  (pop1 : Population) 
  (pop2 : Population) 
  (h1 : pop1.has_distinct_subgroups = true) 
  (h2 : pop2.has_distinct_subgroups = false) :
  (optimal_sampling_method pop1 = SamplingMethod.Stratified) ∧
  (optimal_sampling_method pop2 = SamplingMethod.Random) := by
  sorry

#check optimal_sampling_for_populations

end NUMINAMATH_CALUDE_optimal_sampling_for_populations_l1763_176384


namespace NUMINAMATH_CALUDE_study_days_needed_l1763_176334

/-- Represents the study requirements for a subject --/
structure SubjectRequirements where
  chapters : ℕ
  worksheets : ℕ
  chapterTime : ℚ
  worksheetTime : ℚ

/-- Calculates the total study time for a subject --/
def totalStudyTime (req : SubjectRequirements) : ℚ :=
  req.chapters * req.chapterTime + req.worksheets * req.worksheetTime

/-- Represents the break schedule --/
structure BreakSchedule where
  firstThreeHours : ℚ
  nextThreeHours : ℚ
  lastHour : ℚ
  snackBreaks : ℚ
  lunchBreak : ℚ

/-- Calculates the total break time per day --/
def totalBreakTime (schedule : BreakSchedule) : ℚ :=
  3 * schedule.firstThreeHours + 3 * schedule.nextThreeHours + schedule.lastHour +
  2 * schedule.snackBreaks + schedule.lunchBreak

theorem study_days_needed :
  let math := SubjectRequirements.mk 4 7 (5/2) (3/2)
  let physics := SubjectRequirements.mk 5 9 3 2
  let chemistry := SubjectRequirements.mk 6 8 (7/2) (7/4)
  let breakSchedule := BreakSchedule.mk (1/6) (1/4) (1/3) (1/3) (3/4)
  let totalStudyHours := totalStudyTime math + totalStudyTime physics + totalStudyTime chemistry
  let effectiveStudyHoursPerDay := 7 - totalBreakTime breakSchedule
  ⌈totalStudyHours / effectiveStudyHoursPerDay⌉ = 23 := by
  sorry

end NUMINAMATH_CALUDE_study_days_needed_l1763_176334


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1763_176314

theorem solution_set_inequality (x : ℝ) :
  Set.Icc (-1/2 : ℝ) 1 ∪ Set.Ioo 1 3 =
  {x | (x + 5) / ((x - 1)^2) ≥ 2 ∧ x ≠ 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1763_176314


namespace NUMINAMATH_CALUDE_product_change_theorem_l1763_176372

theorem product_change_theorem (k : ℝ) (x y z : ℝ) (h1 : x * y * z = k) :
  ∃ (p q : ℝ),
    1.805 * (1 - p / 100) * (1 + q / 100) = 1 ∧
    Real.log p - Real.cos q = 0 ∧
    x * 1.805 * y * (1 - p / 100) * z * (1 + q / 100) = k := by
  sorry

end NUMINAMATH_CALUDE_product_change_theorem_l1763_176372


namespace NUMINAMATH_CALUDE_eight_digit_numbers_a_eight_digit_numbers_b_eight_digit_numbers_b_start_with_1_l1763_176333

-- Define the set of digits for part a
def digits_a : Finset ℕ := {0, 1, 2}

-- Define the multiset of digits for part b
def digits_b : Multiset ℕ := {0, 0, 0, 1, 2, 2, 2, 2}

-- Define the number of digits in the numbers we're forming
def num_digits : ℕ := 8

-- Theorem for part a
theorem eight_digit_numbers_a : 
  (Finset.card digits_a ^ num_digits) - (Finset.card digits_a ^ (num_digits - 1)) = 4374 :=
sorry

-- Theorem for part b (total valid numbers)
theorem eight_digit_numbers_b : 
  (Multiset.card digits_b).factorial / ((Multiset.count 0 digits_b).factorial * (Multiset.count 2 digits_b).factorial) - 
  ((Multiset.card digits_b - 1).factorial / ((Multiset.count 0 digits_b - 1).factorial * (Multiset.count 2 digits_b).factorial)) = 175 :=
sorry

-- Theorem for part b (numbers starting with 1)
theorem eight_digit_numbers_b_start_with_1 : 
  (Multiset.card digits_b - 1).factorial / ((Multiset.count 0 digits_b).factorial * (Multiset.count 2 digits_b).factorial) = 35 :=
sorry

end NUMINAMATH_CALUDE_eight_digit_numbers_a_eight_digit_numbers_b_eight_digit_numbers_b_start_with_1_l1763_176333


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1763_176321

/-- Definition of a double subtraction point -/
def is_double_subtraction_point (k b x y : ℝ) : Prop :=
  k ≠ 0 ∧ y = k * x ∧ y = b

/-- The main theorem -/
theorem inequality_system_solution 
  (k : ℝ) 
  (h_k : k ≠ 0)
  (a : ℝ)
  (h_double_sub : is_double_subtraction_point k (a - 2) 3 (3 * k)) :
  {y : ℝ | 2 * (y + 1) < 5 * y - 7 ∧ (y + a) / 2 < 5} = {y : ℝ | 3 < y ∧ y < 8} :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1763_176321


namespace NUMINAMATH_CALUDE_incorrect_equation_property_l1763_176328

theorem incorrect_equation_property : ¬ (∀ a b : ℝ, a * b = a → b = 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_equation_property_l1763_176328


namespace NUMINAMATH_CALUDE_original_candle_length_l1763_176378

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (original_length : ℝ) : 
  current_length = 48 →
  factor = 1.33 →
  original_length = current_length * factor →
  original_length = 63.84 := by
sorry

end NUMINAMATH_CALUDE_original_candle_length_l1763_176378


namespace NUMINAMATH_CALUDE_forty_percent_of_two_l1763_176399

theorem forty_percent_of_two : (40 / 100) * 2 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_forty_percent_of_two_l1763_176399


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_squares_max_value_achieved_l1763_176332

theorem max_value_of_sum_of_squares (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y ≥ x^3 + y^2) : 
  x^2 + y^2 ≤ 2 := by
  sorry

-- The maximum value is indeed achieved
theorem max_value_achieved : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y ≥ x^3 + y^2 ∧ x^2 + y^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_squares_max_value_achieved_l1763_176332


namespace NUMINAMATH_CALUDE_honeycomb_thickness_scientific_notation_l1763_176339

theorem honeycomb_thickness_scientific_notation :
  0.000073 = 7.3 * 10^(-5) := by
  sorry

end NUMINAMATH_CALUDE_honeycomb_thickness_scientific_notation_l1763_176339


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1763_176347

theorem trig_equation_solution (z : ℝ) :
  5 * (Real.sin (2 * z))^4 - 4 * (Real.sin (2 * z))^2 * (Real.cos (2 * z))^2 - (Real.cos (2 * z))^4 + 4 * Real.cos (4 * z) = 0 →
  (∃ k : ℤ, z = π / 8 * (2 * ↑k + 1)) ∨ (∃ n : ℤ, z = π / 6 * (3 * ↑n + 1) ∨ z = π / 6 * (3 * ↑n - 1)) := by
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1763_176347


namespace NUMINAMATH_CALUDE_ratio_problem_l1763_176327

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3) 
  (h2 : c / b = 2) 
  (h3 : d / c = 4) : 
  (a + c) / (b + d) = 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1763_176327


namespace NUMINAMATH_CALUDE_frustum_small_cone_altitude_l1763_176374

-- Define the frustum
structure Frustum where
  altitude : ℝ
  lower_base_area : ℝ
  upper_base_area : ℝ

-- Define the theorem
theorem frustum_small_cone_altitude 
  (f : Frustum) 
  (h1 : f.altitude = 30)
  (h2 : f.lower_base_area = 324 * Real.pi)
  (h3 : f.upper_base_area = 36 * Real.pi) : 
  ∃ (small_cone_altitude : ℝ), small_cone_altitude = 15 := by
  sorry

end NUMINAMATH_CALUDE_frustum_small_cone_altitude_l1763_176374


namespace NUMINAMATH_CALUDE_christine_walking_distance_l1763_176393

/-- Given Christine's walking speed and time spent walking, calculate the distance she wandered. -/
theorem christine_walking_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (h1 : speed = 4) 
  (h2 : time = 5) : 
  speed * time = 20 := by
sorry

end NUMINAMATH_CALUDE_christine_walking_distance_l1763_176393


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1763_176330

theorem remainder_divisibility (n : ℕ) (h : n % 10 = 7) : n % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1763_176330


namespace NUMINAMATH_CALUDE_sum_of_three_integers_2015_l1763_176318

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem sum_of_three_integers_2015 :
  ∃ (a b c : ℕ),
    a + b + c = 2015 ∧
    is_prime a ∧
    b % 3 = 0 ∧
    400 < c ∧ c < 500 ∧
    ¬(c % 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_three_integers_2015_l1763_176318


namespace NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_nine_targets_correct_l1763_176364

/-- Rocket artillery system model -/
structure RocketSystem where
  total_rockets : ℕ
  hit_probability : ℝ

/-- Probability of exactly three unused rockets after firing at five targets -/
def prob_three_unused (system : RocketSystem) : ℝ :=
  10 * system.hit_probability^3 * (1 - system.hit_probability)^2

/-- Expected number of targets hit when firing at nine targets -/
def expected_hits_nine_targets (system : RocketSystem) : ℝ :=
  10 * system.hit_probability - system.hit_probability^10

/-- Theorem: Probability of exactly three unused rockets after firing at five targets -/
theorem prob_three_unused_correct (system : RocketSystem) :
  prob_three_unused system = 10 * system.hit_probability^3 * (1 - system.hit_probability)^2 := by
  sorry

/-- Theorem: Expected number of targets hit when firing at nine targets -/
theorem expected_hits_nine_targets_correct (system : RocketSystem) :
  expected_hits_nine_targets system = 10 * system.hit_probability - system.hit_probability^10 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_unused_correct_expected_hits_nine_targets_correct_l1763_176364


namespace NUMINAMATH_CALUDE_product_equals_square_minus_one_l1763_176323

theorem product_equals_square_minus_one (r : ℕ) (hr : r > 5) :
  let a := r^3 + r^2 + r
  (a) * (a + 1) * (a + 2) * (a + 3) = (r^6 + 2*r^5 + 3*r^4 + 5*r^3 + 4*r^2 + 3*r + 1)^2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_product_equals_square_minus_one_l1763_176323


namespace NUMINAMATH_CALUDE_battery_current_at_12_ohms_l1763_176344

/-- A battery with voltage 48V and a relationship between current and resistance --/
structure Battery where
  voltage : ℝ
  current : ℝ → ℝ
  resistance : ℝ
  h_voltage : voltage = 48
  h_current : ∀ r, current r = voltage / r

/-- The theorem states that for a battery with 48V and the given current-resistance relationship,
    when the resistance is 12Ω, the current is 4A --/
theorem battery_current_at_12_ohms (b : Battery) (h : b.resistance = 12) :
  b.current b.resistance = 4 := by
  sorry

end NUMINAMATH_CALUDE_battery_current_at_12_ohms_l1763_176344


namespace NUMINAMATH_CALUDE_unique_line_divides_triangle_l1763_176331

/-- A triangle in a 2D plane --/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- A line in the form y = mx --/
structure Line where
  m : ℝ

/-- Checks if a line divides a triangle into two equal areas --/
def dividesEqualArea (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The specific triangle in the problem --/
def specificTriangle : Triangle :=
  { v1 := (0, 0),
    v2 := (4, 4),
    v3 := (12, 0) }

theorem unique_line_divides_triangle :
  ∃! m : ℝ, dividesEqualArea specificTriangle { m := m } ∧ m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_line_divides_triangle_l1763_176331


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_l1763_176343

/-- The curve C in rectangular coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 = 4*y

/-- The line l in rectangular coordinates -/
def line_l (x y : ℝ) : Prop := y = x + 1

/-- The intersection points of curve C and line l -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | curve_C p.1 p.2 ∧ line_l p.1 p.2}

theorem distance_between_intersection_points :
  ∃ (p q : ℝ × ℝ), p ∈ intersection_points ∧ q ∈ intersection_points ∧ p ≠ q ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_l1763_176343


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l1763_176315

theorem tangent_product_equals_two (α β : Real) (h : α + β = π / 4) :
  (1 + Real.tan α) * (1 + Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l1763_176315


namespace NUMINAMATH_CALUDE_quadratic_equation_m_value_l1763_176349

theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m - 3) * x^(m^2 - 7) - 4*x - 8 = a*x^2 + b*x + c) →
  (m - 3 ≠ 0) →
  m = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_m_value_l1763_176349


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1763_176352

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 3) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1763_176352


namespace NUMINAMATH_CALUDE_m_range_l1763_176348

theorem m_range : ∀ m : ℝ, m = 5 * Real.sqrt (1/5) - Real.sqrt 45 → -5 < m ∧ m < -4 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1763_176348


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1763_176386

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 - 1) :
  (1 - 1 / (a + 1)) * ((a^2 + 2*a + 1) / a) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1763_176386


namespace NUMINAMATH_CALUDE_tan_two_implies_fraction_l1763_176306

theorem tan_two_implies_fraction (θ : Real) (h : Real.tan θ = 2) :
  (Real.cos θ - Real.sin θ) / (Real.cos θ + Real.sin θ) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_implies_fraction_l1763_176306


namespace NUMINAMATH_CALUDE_gym_cost_comparison_l1763_176366

/-- Represents the cost of gym sessions under two different schemes -/
def gym_cost (x : ℕ) : ℝ × ℝ :=
  let y₁ := 12 * x + 40  -- Scheme 1: 40% discount + membership
  let y₂ := 16 * x       -- Scheme 2: 20% discount, no membership
  (y₁, y₂)

/-- Theorem stating which scheme is cheaper based on the number of sessions -/
theorem gym_cost_comparison (x : ℕ) (h : 5 ≤ x ∧ x ≤ 20) :
  let (y₁, y₂) := gym_cost x
  (x < 10 → y₂ < y₁) ∧
  (x = 10 → y₁ = y₂) ∧
  (10 < x → y₁ < y₂) :=
by sorry

end NUMINAMATH_CALUDE_gym_cost_comparison_l1763_176366


namespace NUMINAMATH_CALUDE_inequality_proof_l1763_176382

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (1+a)/(1-a) + (1+b)/(1-b) + (1+c)/(1-c) ≤ 2*(b/a + c/b + a/c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1763_176382


namespace NUMINAMATH_CALUDE_probability_two_changing_yao_l1763_176346

theorem probability_two_changing_yao (n : Nat) (p : Real) (k : Nat) : 
  n = 6 → p = 1/4 → k = 2 →
  Nat.choose n k * p^k * (1-p)^(n-k) = 1215/4096 := by
sorry

end NUMINAMATH_CALUDE_probability_two_changing_yao_l1763_176346


namespace NUMINAMATH_CALUDE_base8_addition_l1763_176381

/-- Addition in base 8 -/
def base8_add (a b : ℕ) : ℕ := sorry

/-- Conversion from base 10 to base 8 -/
def to_base8 (n : ℕ) : ℕ := sorry

/-- Conversion from base 8 to base 10 -/
def from_base8 (n : ℕ) : ℕ := sorry

theorem base8_addition : base8_add (from_base8 12) (from_base8 157) = from_base8 171 := by sorry

end NUMINAMATH_CALUDE_base8_addition_l1763_176381


namespace NUMINAMATH_CALUDE_stating_standard_polar_coord_example_l1763_176325

/-- 
Given a point in polar coordinates (r, θ) where r can be negative,
this function returns the equivalent standard polar coordinate representation
where r > 0 and 0 ≤ θ < 2π.
-/
def standardPolarCoord (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  sorry

/-- 
Theorem stating that the standard polar coordinate representation
of the point (-3, 5π/6) is (3, 11π/6).
-/
theorem standard_polar_coord_example : 
  standardPolarCoord (-3) (5 * Real.pi / 6) = (3, 11 * Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_stating_standard_polar_coord_example_l1763_176325


namespace NUMINAMATH_CALUDE_minor_axis_length_l1763_176309

def ellipse_equation (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

theorem minor_axis_length :
  ∃ (minor_axis_length : ℝ),
    minor_axis_length = 4 ∧
    ∀ (x y : ℝ), ellipse_equation x y →
      ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
        x^2 / a^2 + y^2 / b^2 = 1 ∧
        minor_axis_length = 2 * b :=
by sorry

end NUMINAMATH_CALUDE_minor_axis_length_l1763_176309


namespace NUMINAMATH_CALUDE_fish_fillet_problem_l1763_176367

theorem fish_fillet_problem (total : ℕ) (team1 : ℕ) (team2 : ℕ) 
  (h1 : total = 500) 
  (h2 : team1 = 189) 
  (h3 : team2 = 131) : 
  total - (team1 + team2) = 180 := by
sorry

end NUMINAMATH_CALUDE_fish_fillet_problem_l1763_176367


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l1763_176389

/-- 
Given an arithmetic sequence with:
  - First term: 156
  - Last term: 36
  - Common difference: -6
This theorem proves that the number of terms in the sequence is 21.
-/
theorem arithmetic_sequence_term_count : 
  let a₁ : ℤ := 156  -- First term
  let aₙ : ℤ := 36   -- Last term
  let d : ℤ := -6    -- Common difference
  ∃ n : ℕ, n > 0 ∧ aₙ = a₁ + (n - 1) * d ∧ n = 21
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l1763_176389


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1763_176338

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 + 15 * x₁ - 20 = 0) → 
  (10 * x₂^2 + 15 * x₂ - 20 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1763_176338


namespace NUMINAMATH_CALUDE_function_value_at_three_l1763_176390

/-- Given a function f: ℝ → ℝ satisfying certain conditions, prove that f(3) = 11 -/
theorem function_value_at_three (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : f 1 = 5)
    (h2 : ∀ x, f x = a * x + b * x + 2) : 
  f 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_three_l1763_176390


namespace NUMINAMATH_CALUDE_irrational_sqrt_three_others_rational_l1763_176385

theorem irrational_sqrt_three_others_rational :
  ¬ (∃ (a b : ℤ), b ≠ 0 ∧ Real.sqrt 3 = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-1 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (1/2 : ℝ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (3.14 : ℝ) = a / b) :=
by sorry

end NUMINAMATH_CALUDE_irrational_sqrt_three_others_rational_l1763_176385


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1763_176360

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 1
  1 - (x / (x + 1)) / (x / (x^2 - 1)) = 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1763_176360


namespace NUMINAMATH_CALUDE_total_snowfall_calculation_l1763_176373

theorem total_snowfall_calculation (monday tuesday wednesday : Real) 
  (h1 : monday = 0.327)
  (h2 : tuesday = 0.216)
  (h3 : wednesday = 0.184) :
  monday + tuesday + wednesday = 0.727 := by
  sorry

end NUMINAMATH_CALUDE_total_snowfall_calculation_l1763_176373


namespace NUMINAMATH_CALUDE_angle_problem_l1763_176322

theorem angle_problem (angle1 angle2 angle3 angle4 angle5 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = angle5 →
  angle3 + angle4 = 180 →
  angle4 = 35 := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l1763_176322


namespace NUMINAMATH_CALUDE_parabola_properties_l1763_176311

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- Theorem about properties of the parabola y = x^2 + 2x - 3 -/
theorem parabola_properties :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (0, -3)
  ∀ x ∈ Set.Icc (-3 : ℝ) 2,
  (f A.1 = A.2 ∧ f B.1 = B.2 ∧ f C.1 = C.2) ∧
  (A.1 < B.1) ∧
  (∃ (y_max y_min : ℝ), 
    (∀ x' ∈ Set.Icc (-3 : ℝ) 2, f x' ≤ y_max ∧ f x' ≥ y_min) ∧
    y_max - y_min = 9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1763_176311


namespace NUMINAMATH_CALUDE_least_nickels_l1763_176383

theorem least_nickels (n : ℕ) : 
  (n > 0) → 
  (n % 7 = 2) → 
  (n % 4 = 3) → 
  (∀ m : ℕ, m > 0 → m % 7 = 2 → m % 4 = 3 → n ≤ m) → 
  n = 23 := by
sorry

end NUMINAMATH_CALUDE_least_nickels_l1763_176383


namespace NUMINAMATH_CALUDE_race_track_width_l1763_176351

/-- The width of a circular race track given its inner circumference and outer radius -/
theorem race_track_width (inner_circumference outer_radius : ℝ) :
  inner_circumference = 440 →
  outer_radius = 84.02817496043394 →
  ∃ width : ℝ, abs (width - 14.02056077700854) < 1e-10 ∧
    width = outer_radius - inner_circumference / (2 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_race_track_width_l1763_176351


namespace NUMINAMATH_CALUDE_find_p_value_l1763_176313

-- Define the polynomial (x+y)^9
def polynomial (x y : ℝ) : ℝ := (x + y)^9

-- Define the second term of the expansion
def second_term (x y : ℝ) : ℝ := 9 * x^8 * y

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := 36 * x^7 * y^2

-- Theorem statement
theorem find_p_value (p q : ℝ) : 
  p > 0 ∧ q > 0 ∧ p + q = 1 ∧ second_term p q = third_term p q → p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_find_p_value_l1763_176313


namespace NUMINAMATH_CALUDE_square_area_on_parabola_and_line_l1763_176312

theorem square_area_on_parabola_and_line : ∃ (a : ℝ), a > 0 ∧ 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 + 2*x₁ + 1 = 8) ∧ 
    (x₂^2 + 2*x₂ + 1 = 8) ∧ 
    a = (x₂ - x₁)^2) ∧ 
  a = 36 := by
sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_and_line_l1763_176312


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l1763_176319

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

-- State the theorem
theorem min_value_and_inequality (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 2) : 
  (a + b = 2) ∧ ¬(a^2 + a > 2 ∧ b^2 + b > 2) := by
  sorry


end NUMINAMATH_CALUDE_min_value_and_inequality_l1763_176319


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_specific_roots_l1763_176302

theorem quadratic_equation_roots (m : ℝ) :
  let equation := fun x => m * x^2 - 2 * x + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
sorry

theorem quadratic_equation_specific_roots (m : ℝ) :
  let equation := fun x => m * x^2 - 2 * x + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ x₁ * x₂ - x₁ - x₂ = 1/2) →
  m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_quadratic_equation_specific_roots_l1763_176302


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l1763_176398

theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k) →
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y - x = 2 * k ∧ x < 0 ∧ y > 0) →
  0 < k ∧ k < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l1763_176398


namespace NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_x_satisfying_inequality_l1763_176362

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) ≥ 3
theorem solution_set_f_geq_3 :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0 ∨ x ≥ 3} := by sorry

-- Theorem for the range of x satisfying the inequality
theorem range_of_x_satisfying_inequality :
  {x : ℝ | ∀ a b : ℝ, a ≠ 0 → |a + b| + |a - b| ≥ a * f x} = 
  {x : ℝ | 1/2 ≤ x ∧ x ≤ 5/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_3_range_of_x_satisfying_inequality_l1763_176362


namespace NUMINAMATH_CALUDE_inequality_chain_l1763_176342

theorem inequality_chain (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / a + 1 / b + 1 / c ≥ 2 / (a + b) + 2 / (b + c) + 2 / (c + a) ∧
  2 / (a + b) + 2 / (b + c) + 2 / (c + a) ≥ 9 / (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l1763_176342


namespace NUMINAMATH_CALUDE_task_assignments_and_arrangements_l1763_176341

def num_volunteers : ℕ := 5
def num_tasks : ℕ := 4

theorem task_assignments_and_arrangements :
  let assign_all_tasks := (num_volunteers.choose 2) * num_tasks.factorial
  let assign_one_task_to_two := (num_volunteers.choose 2) * (num_tasks - 1).factorial
  let photo_arrangement := (num_volunteers.factorial / (num_volunteers - 2).factorial) * 2
  (assign_all_tasks = 240) ∧
  (assign_one_task_to_two = 60) ∧
  (photo_arrangement = 40) := by sorry

end NUMINAMATH_CALUDE_task_assignments_and_arrangements_l1763_176341


namespace NUMINAMATH_CALUDE_shopkeeper_pricing_l1763_176303

theorem shopkeeper_pricing (CP : ℝ) 
  (h1 : 0.75 * CP = 600) : 1.25 * CP = 1000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_pricing_l1763_176303


namespace NUMINAMATH_CALUDE_min_sum_squares_roots_l1763_176392

theorem min_sum_squares_roots (m : ℝ) (α β : ℝ) : 
  (∀ x : ℝ, x^2 - 2*m*x + 2 - m^2 = 0 ↔ x = α ∨ x = β) → 
  ∃ (k : ℝ), ∀ m : ℝ, α^2 + β^2 ≥ k ∧ ∃ m : ℝ, α^2 + β^2 = k :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_roots_l1763_176392


namespace NUMINAMATH_CALUDE_abs_sum_problem_l1763_176310

theorem abs_sum_problem (x y : ℝ) 
  (h1 : |x| + x + y = 8) 
  (h2 : x + |y| - y = 14) : 
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_abs_sum_problem_l1763_176310


namespace NUMINAMATH_CALUDE_tank_capacity_is_33_l1763_176335

/-- Represents the capacity of a water tank with specific filling conditions. -/
def tank_capacity (initial_fraction : ℚ) (added_water : ℕ) (leak_rate : ℕ) (fill_time : ℕ) : ℚ :=
  let total_leak := (leak_rate * fill_time : ℚ)
  let total_added := (added_water : ℚ) + total_leak
  total_added / (1 - initial_fraction)

/-- Theorem stating that under given conditions, the tank capacity is 33 gallons. -/
theorem tank_capacity_is_33 :
  tank_capacity (1/3) 16 2 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_is_33_l1763_176335


namespace NUMINAMATH_CALUDE_cosine_amplitude_l1763_176371

theorem cosine_amplitude (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (∀ x, a * Real.cos (b * x - c) ≤ 3) ∧
  (∃ x, a * Real.cos (b * x - c) = 3) ∧
  (∀ x, a * Real.cos (b * x - c) = a * Real.cos (b * (x + 2 * Real.pi) - c)) →
  a = 3 :=
sorry

end NUMINAMATH_CALUDE_cosine_amplitude_l1763_176371


namespace NUMINAMATH_CALUDE_simplify_expression_l1763_176340

theorem simplify_expression :
  1 / ((3 / (Real.sqrt 2 + 2)) + (4 / (Real.sqrt 5 - 2))) =
  1 / (11 + 4 * Real.sqrt 5 - (3 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1763_176340


namespace NUMINAMATH_CALUDE_expression_simplification_l1763_176376

theorem expression_simplification (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) = (3 * x - 2) / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1763_176376


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l1763_176304

-- Define the original and discounted prices
def original_price : ℚ := 12 / 3
def discounted_price : ℚ := 10 / 4

-- Define the percentage decrease
def percentage_decrease : ℚ := (original_price - discounted_price) / original_price * 100

-- Theorem statement
theorem price_decrease_percentage :
  percentage_decrease = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l1763_176304


namespace NUMINAMATH_CALUDE_fraction_inequality_condition_l1763_176365

theorem fraction_inequality_condition (x : ℝ) : 
  (2 * x + 1) / (1 - x) ≥ 0 ↔ -1/2 ≤ x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_condition_l1763_176365


namespace NUMINAMATH_CALUDE_swimming_frequency_difference_l1763_176387

def camden_total : ℕ := 16
def susannah_total : ℕ := 24
def weeks_in_month : ℕ := 4

theorem swimming_frequency_difference :
  (susannah_total / weeks_in_month) - (camden_total / weeks_in_month) = 2 :=
by sorry

end NUMINAMATH_CALUDE_swimming_frequency_difference_l1763_176387


namespace NUMINAMATH_CALUDE_frustum_central_angle_l1763_176363

/-- Represents a frustum of a cone -/
structure Frustum where
  lateral_area : ℝ
  total_area : ℝ

/-- 
Given a frustum of a cone with lateral surface area 10π and total surface area 19π,
the central angle of the lateral surface when laid flat is 324°.
-/
theorem frustum_central_angle (f : Frustum) 
  (h1 : f.lateral_area = 10 * Real.pi)
  (h2 : f.total_area = 19 * Real.pi) : 
  ∃ (angle : ℝ), angle = 324 ∧ 
  (angle / 360) * Real.pi * ((6 * 360) / angle)^2 = f.lateral_area := by
  sorry


end NUMINAMATH_CALUDE_frustum_central_angle_l1763_176363


namespace NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l1763_176300

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 1)^2
def parabola2 (x y : ℝ) : Prop := x + 4 = (y - 3)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem sum_of_intersection_coordinates :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 8) :=
sorry

end NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l1763_176300


namespace NUMINAMATH_CALUDE_order_of_6_undefined_l1763_176391

def f (x : ℤ) : ℤ := x^2 % 13

def f_iter (n : ℕ) (x : ℤ) : ℤ := 
  match n with
  | 0 => x
  | n+1 => f (f_iter n x)

theorem order_of_6_undefined : ¬ ∃ m : ℕ, m > 0 ∧ f_iter m 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_order_of_6_undefined_l1763_176391


namespace NUMINAMATH_CALUDE_binary_remainder_by_four_l1763_176345

/-- The binary number 111001101101₂ -/
def binary_number : Nat := 3693

/-- Theorem: The remainder when 111001101101₂ is divided by 4 is 1 -/
theorem binary_remainder_by_four :
  binary_number % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binary_remainder_by_four_l1763_176345


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l1763_176379

theorem largest_n_for_equation : 
  (∃ (x y z : ℕ+), 6^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 9) ∧ 
  (∀ (n : ℕ+), n > 6 → ¬∃ (x y z : ℕ+), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 9) :=
by sorry

#check largest_n_for_equation

end NUMINAMATH_CALUDE_largest_n_for_equation_l1763_176379


namespace NUMINAMATH_CALUDE_pta_fundraiser_l1763_176329

theorem pta_fundraiser (initial_amount : ℝ) (school_supplies_fraction : ℝ) (food_fraction : ℝ) : 
  initial_amount = 400 →
  school_supplies_fraction = 1/4 →
  food_fraction = 1/2 →
  initial_amount * (1 - school_supplies_fraction) * (1 - food_fraction) = 150 := by
sorry

end NUMINAMATH_CALUDE_pta_fundraiser_l1763_176329


namespace NUMINAMATH_CALUDE_largest_integer_in_range_l1763_176354

theorem largest_integer_in_range : ∃ (x : ℤ), 
  (1/4 : ℚ) < (x : ℚ)/5 ∧ (x : ℚ)/5 < 2/3 ∧ 
  ∀ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/5 ∧ (y : ℚ)/5 < 2/3 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_range_l1763_176354


namespace NUMINAMATH_CALUDE_photo_collection_inconsistency_l1763_176370

/-- Represents the number of photos each person has --/
structure PhotoCollection where
  tom : ℕ
  tim : ℕ
  paul : ℕ
  jane : ℕ

/-- The problem statement --/
theorem photo_collection_inconsistency 
  (photos : PhotoCollection) 
  (total_photos : photos.tom + photos.tim + photos.paul + photos.jane = 200)
  (paul_more_than_tim : photos.paul = photos.tim + 10)
  (tim_less_than_total : photos.tim = 200 - 100) :
  False :=
by
  sorry


end NUMINAMATH_CALUDE_photo_collection_inconsistency_l1763_176370


namespace NUMINAMATH_CALUDE_bird_nest_problem_l1763_176397

theorem bird_nest_problem (birds : ℕ) (difference : ℕ) (nests : ℕ) : 
  birds = 6 → difference = 3 → birds - nests = difference → nests = 3 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_problem_l1763_176397


namespace NUMINAMATH_CALUDE_fraction_equality_l1763_176388

theorem fraction_equality : 
  (4 + 2/3 + 3 + 1/3) - (2 + 1/2 - 1/2) = 4 + 2/3 - (2 + 1/2) + 1/2 + 3 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1763_176388


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_constant_l1763_176301

/-- A quadratic expression can be expressed as the square of a binomial if and only if its discriminant is zero. -/
def is_perfect_square (a b c : ℝ) : Prop :=
  b ^ 2 - 4 * a * c = 0

theorem quadratic_perfect_square_constant (b : ℝ) :
  is_perfect_square 9 (-24) b → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_constant_l1763_176301


namespace NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l1763_176305

-- Define propositions A and B
def prop_A (a b : ℝ) : Prop := a + b ≠ 4
def prop_B (a b : ℝ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem stating that A is neither sufficient nor necessary for B
theorem A_neither_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, prop_A a b → prop_B a b) ∧
  ¬(∀ a b : ℝ, prop_B a b → prop_A a b) :=
by sorry

end NUMINAMATH_CALUDE_A_neither_sufficient_nor_necessary_l1763_176305


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l1763_176307

theorem greatest_integer_satisfying_conditions : 
  ∃ (n : ℕ), n < 150 ∧ 
  (∃ (k m : ℕ), n = 9 * k - 2 ∧ n = 11 * m - 4) ∧
  (∀ (n' : ℕ), n' < 150 → 
    (∃ (k' m' : ℕ), n' = 9 * k' - 2 ∧ n' = 11 * m' - 4) → 
    n' ≤ n) ∧
  n = 139 := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_conditions_l1763_176307


namespace NUMINAMATH_CALUDE_min_overtakes_is_five_l1763_176353

/-- Represents the girls in the race -/
inductive Girl
| Fiona
| Gertrude
| Hannah
| India
| Janice

/-- Represents the order of girls in the race -/
def RaceOrder := List Girl

/-- The initial order of the girls in the race -/
def initial_order : RaceOrder :=
  [Girl.Fiona, Girl.Gertrude, Girl.Hannah, Girl.India, Girl.Janice]

/-- The final order of the girls in the race -/
def final_order : RaceOrder :=
  [Girl.India, Girl.Gertrude, Girl.Fiona, Girl.Janice, Girl.Hannah]

/-- Calculates the minimum number of overtakes required to transform the initial order to the final order -/
def min_overtakes (initial : RaceOrder) (final : RaceOrder) : Nat :=
  sorry

/-- Theorem stating that the minimum number of overtakes is 5 -/
theorem min_overtakes_is_five :
  min_overtakes initial_order final_order = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_overtakes_is_five_l1763_176353


namespace NUMINAMATH_CALUDE_optimal_well_placement_l1763_176358

/-- Three houses positioned along a straight road -/
structure Village where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The distance between adjacent houses is 50 meters -/
def house_distance : ℝ := 50

/-- A village with houses positioned at the correct intervals -/
def village : Village :=
  { A := 0,
    B := house_distance,
    C := 2 * house_distance }

/-- The sum of distances from a point to all houses -/
def total_distance (x : ℝ) : ℝ :=
  |x - village.A| + |x - village.B| + |x - village.C|

/-- The well position that minimizes the total distance -/
def optimal_well_position : ℝ := village.B

theorem optimal_well_placement :
  ∀ x : ℝ, total_distance optimal_well_position ≤ total_distance x :=
sorry

end NUMINAMATH_CALUDE_optimal_well_placement_l1763_176358


namespace NUMINAMATH_CALUDE_variance_best_stability_measure_l1763_176396

/-- A performance measure is a function that takes a list of real numbers (representing performance data) and returns a real number. -/
def PerformanceMeasure := (List ℝ) → ℝ

/-- Average of a list of real numbers -/
def average : PerformanceMeasure := sorry

/-- Median of a list of real numbers -/
def median : PerformanceMeasure := sorry

/-- Mode of a list of real numbers -/
def mode : PerformanceMeasure := sorry

/-- Variance of a list of real numbers -/
def variance : PerformanceMeasure := sorry

/-- A measure is considered stable if it reflects the spread of the data -/
def reflectsSpread (m : PerformanceMeasure) : Prop := sorry

/-- Theorem stating that variance is the measure that best reflects the stability of performance -/
theorem variance_best_stability_measure :
  reflectsSpread variance ∧
  ¬reflectsSpread average ∧
  ¬reflectsSpread median ∧
  ¬reflectsSpread mode :=
sorry

end NUMINAMATH_CALUDE_variance_best_stability_measure_l1763_176396


namespace NUMINAMATH_CALUDE_distance_driven_l1763_176361

/-- Represents the efficiency of a car in kilometers per gallon -/
def car_efficiency : ℝ := 10

/-- Represents the amount of gas available in gallons -/
def gas_available : ℝ := 10

/-- Theorem stating the distance that can be driven given the car's efficiency and available gas -/
theorem distance_driven : car_efficiency * gas_available = 100 := by sorry

end NUMINAMATH_CALUDE_distance_driven_l1763_176361


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1763_176368

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 5 = 0

-- Define what it means for two circles to be externally tangent
def externally_tangent (C₁ C₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧
  ∀ (x' y' : ℝ), (C₁ x' y' ∧ C₂ x' y') → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_externally_tangent : externally_tangent C₁ C₂ :=
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1763_176368


namespace NUMINAMATH_CALUDE_f_intersects_all_lines_l1763_176394

/-- A function that intersects every line in the coordinate plane at least once -/
def f (x : ℝ) : ℝ := x^3

/-- Proposition: The function f intersects every line in the coordinate plane at least once -/
theorem f_intersects_all_lines :
  ∀ (k b : ℝ), ∃ (x : ℝ), f x = k * x + b :=
sorry

end NUMINAMATH_CALUDE_f_intersects_all_lines_l1763_176394


namespace NUMINAMATH_CALUDE_min_value_theorem_l1763_176324

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4*x + 3*y = 1) :
  ∀ z : ℝ, z = (1 / (2*x - y)) + (2 / (x + 2*y)) → z ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1763_176324


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_two_l1763_176316

theorem sum_of_coefficients_equals_two (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10 + a₁₁*(x - 1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_two_l1763_176316
