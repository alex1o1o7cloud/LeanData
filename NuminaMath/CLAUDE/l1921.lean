import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1921_192145

def g (x : ℝ) : ℝ := -x^3 + x^2 - x + 1

theorem polynomial_value_theorem :
  g 3 = 1 ∧ 12 * (-1) - 6 * 1 + 3 * (-1) - 1 = -22 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1921_192145


namespace NUMINAMATH_CALUDE_expression_evaluation_l1921_192158

theorem expression_evaluation : 120 * (120 - 5) - (120 * 120 - 10 + 2) = -592 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1921_192158


namespace NUMINAMATH_CALUDE_f_at_negative_one_l1921_192136

def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x + 16

def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + 5*x^3 + b*x^2 + 150*x + c

theorem f_at_negative_one (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g a x = 0 ∧ g a y = 0 ∧ g a z = 0) →
  (∀ x : ℝ, g a x = 0 → f b c x = 0) →
  f b c (-1) = -1347 := by sorry

end NUMINAMATH_CALUDE_f_at_negative_one_l1921_192136


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1921_192171

/-- An isosceles trapezoid with specific dimensions and inscribed circles --/
structure IsoscelesTrapezoidWithCircles where
  -- The length of side AB
  ab : ℝ
  -- The length of sides BC and DA
  bc : ℝ
  -- The length of side CD
  cd : ℝ
  -- The radius of circles centered at A and B
  r_ab : ℝ
  -- The radius of circles centered at C and D
  r_cd : ℝ

/-- The theorem stating the radius of the inscribed circle tangent to all four circles --/
theorem inscribed_circle_radius (t : IsoscelesTrapezoidWithCircles)
  (h_ab : t.ab = 10)
  (h_bc : t.bc = 7)
  (h_cd : t.cd = 6)
  (h_r_ab : t.r_ab = 4)
  (h_r_cd : t.r_cd = 3) :
  ∃ r : ℝ, r = (-81 + 57 * Real.sqrt 5) / 23 ∧
    (∃ O : ℝ × ℝ, ∃ A B C D : ℝ × ℝ,
      -- O is the center of the inscribed circle
      -- A, B, C, D are the centers of the given circles
      -- The inscribed circle is tangent to all four given circles
      True) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1921_192171


namespace NUMINAMATH_CALUDE_fraction_equality_l1921_192109

theorem fraction_equality : (1 / 5 - 1 / 6) / (1 / 3 - 1 / 4) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1921_192109


namespace NUMINAMATH_CALUDE_division_result_l1921_192106

theorem division_result : (210 : ℚ) / (15 + 12 * 3 - 6) = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1921_192106


namespace NUMINAMATH_CALUDE_factoring_quadratic_l1921_192193

theorem factoring_quadratic (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) := by
  sorry

end NUMINAMATH_CALUDE_factoring_quadratic_l1921_192193


namespace NUMINAMATH_CALUDE_geometric_sequence_max_product_l1921_192169

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

def product_of_terms (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  (a₁^n) * (q^((n * (n - 1)) / 2))

theorem geometric_sequence_max_product :
  ∃ (q : ℝ) (n : ℕ),
    geometric_sequence (-6) q 4 = (-3/4) ∧
    q = (1/2) ∧
    n = 4 ∧
    ∀ (m : ℕ), m ≠ 0 → product_of_terms (-6) q m ≤ product_of_terms (-6) q n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_max_product_l1921_192169


namespace NUMINAMATH_CALUDE_emily_card_collection_l1921_192121

/-- Emily's card collection problem -/
theorem emily_card_collection (initial_cards : ℕ) (additional_cards : ℕ) :
  initial_cards = 63 → additional_cards = 7 → initial_cards + additional_cards = 70 := by
  sorry

end NUMINAMATH_CALUDE_emily_card_collection_l1921_192121


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1921_192196

/-- A rhombus with given diagonal lengths has a specific perimeter -/
theorem rhombus_perimeter (AC BD : ℝ) (h1 : AC = 8) (h2 : BD = 6) :
  let side_length := Real.sqrt ((AC / 2) ^ 2 + (BD / 2) ^ 2)
  4 * side_length = 20 := by sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l1921_192196


namespace NUMINAMATH_CALUDE_line_charts_reflect_changes_l1921_192127

/-- Represents a line chart --/
structure LineChart where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a change in data over time or across categories --/
structure DataChange where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Defines the property of a line chart being able to reflect changes clearly --/
def clearly_reflects_changes (chart : LineChart) : Prop :=
  ∀ (change : DataChange), ∃ (representation : LineChart → DataChange → Prop),
    representation chart change ∧ 
    (∀ (other_chart : LineChart), representation other_chart change → other_chart = chart)

/-- Theorem stating that line charts can clearly reflect changes in things --/
theorem line_charts_reflect_changes :
  ∀ (chart : LineChart), clearly_reflects_changes chart :=
sorry

end NUMINAMATH_CALUDE_line_charts_reflect_changes_l1921_192127


namespace NUMINAMATH_CALUDE_stratified_sampling_male_count_l1921_192188

theorem stratified_sampling_male_count 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 48) 
  (h2 : total_female = 36) 
  (h3 : sample_size = 21) :
  (sample_size * total_male) / (total_male + total_female) = 12 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_male_count_l1921_192188


namespace NUMINAMATH_CALUDE_factorization_equality_simplification_equality_system_of_inequalities_l1921_192124

-- Problem 1
theorem factorization_equality (x y : ℝ) :
  x^2 * (x - 3) + y^2 * (3 - x) = (x - 3) * (x + y) * (x - y) := by sorry

-- Problem 2
theorem simplification_equality (x : ℝ) (h1 : x ≠ 3/5) (h2 : x ≠ -3/5) :
  (2*x / (5*x - 3)) / (3 / (25*x^2 - 9)) * (x / (5*x + 3)) = 2/3 * x^2 := by sorry

-- Problem 3
theorem system_of_inequalities (x : ℝ) :
  ((x - 3) / 2 + 3 ≥ x + 1) ∧ (1 - 3*(x - 1) < 8 - x) ↔ (-2 < x ∧ x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_factorization_equality_simplification_equality_system_of_inequalities_l1921_192124


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1921_192147

theorem complex_expression_simplification :
  (8 - 5*Complex.I) + 3*(2 - 4*Complex.I) = 14 - 17*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1921_192147


namespace NUMINAMATH_CALUDE_representation_bound_l1921_192148

def f (n : ℕ) : ℕ := sorry

theorem representation_bound (n : ℕ) (h : n ≥ 3) :
  (2 : ℝ) ^ (n^2/4) < (f (2^n) : ℝ) ∧ (f (2^n) : ℝ) < (2 : ℝ) ^ (n^2/2) := by
  sorry

end NUMINAMATH_CALUDE_representation_bound_l1921_192148


namespace NUMINAMATH_CALUDE_probability_ratio_l1921_192134

def total_slips : ℕ := 50
def distinct_numbers : ℕ := 10
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := distinct_numbers / Nat.choose total_slips drawn_slips

def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l1921_192134


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_mixing_l1921_192186

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Represents the result of mixing two mixtures -/
def mix (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk,
    water := m1.water + m2.water }

/-- The ratio of milk to water in a mixture -/
def milkWaterRatio (m : Mixture) : ℚ := m.milk / m.water

theorem milk_water_ratio_after_mixing :
  let m1 : Mixture := { milk := 7, water := 2 }
  let m2 : Mixture := { milk := 8, water := 1 }
  milkWaterRatio (mix m1 m2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_mixing_l1921_192186


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1921_192155

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (1 / x^2 - 2 / x - 3 < 0) ↔ (x < -1 ∨ x > 1/3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1921_192155


namespace NUMINAMATH_CALUDE_flour_already_added_l1921_192146

theorem flour_already_added (total_flour : ℕ) (flour_needed : ℕ) (flour_already_added : ℕ) : 
  total_flour = 9 → flour_needed = 6 → flour_already_added = total_flour - flour_needed → 
  flour_already_added = 3 := by
sorry

end NUMINAMATH_CALUDE_flour_already_added_l1921_192146


namespace NUMINAMATH_CALUDE_birds_joined_birds_joined_fence_l1921_192128

theorem birds_joined (initial_birds : ℕ) (initial_storks : ℕ) (final_total : ℕ) : ℕ :=
  let initial_total := initial_birds + initial_storks
  let birds_joined := final_total - initial_total
  birds_joined

theorem birds_joined_fence : birds_joined 3 2 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_birds_joined_birds_joined_fence_l1921_192128


namespace NUMINAMATH_CALUDE_total_blocks_adolfos_blocks_l1921_192166

/-- Given an initial number of blocks and a number of blocks added, 
    the total number of blocks is equal to the sum of the initial blocks and added blocks. -/
theorem total_blocks (initial_blocks added_blocks : ℕ) :
  initial_blocks + added_blocks = initial_blocks + added_blocks := by
  sorry

/-- Adolfo's block problem -/
theorem adolfos_blocks : 
  let initial_blocks : ℕ := 35
  let added_blocks : ℕ := 30
  initial_blocks + added_blocks = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_blocks_adolfos_blocks_l1921_192166


namespace NUMINAMATH_CALUDE_scout_weekend_earnings_l1921_192150

/-- Scout's weekend earnings calculation --/
theorem scout_weekend_earnings 
  (base_pay : ℝ) 
  (tip_per_customer : ℝ)
  (saturday_hours : ℝ) 
  (saturday_customers : ℕ)
  (sunday_hours : ℝ) 
  (sunday_customers : ℕ)
  (h1 : base_pay = 10)
  (h2 : tip_per_customer = 5)
  (h3 : saturday_hours = 4)
  (h4 : saturday_customers = 5)
  (h5 : sunday_hours = 5)
  (h6 : sunday_customers = 8) :
  base_pay * (saturday_hours + sunday_hours) + 
  tip_per_customer * (saturday_customers + sunday_customers) = 155 := by
sorry

end NUMINAMATH_CALUDE_scout_weekend_earnings_l1921_192150


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1921_192185

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℚ, 
  (∀ x : ℚ, (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1921_192185


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l1921_192123

theorem percentage_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hM : M > 0) (hq_bound : q < 100) :
  M * (1 + p / 100) * (1 - q / 100) = 1.1 * M ↔ 
  p = (10 + 100 * q) / (100 - q) :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_decrease_l1921_192123


namespace NUMINAMATH_CALUDE_number_puzzle_l1921_192195

theorem number_puzzle : ∃ x : ℝ, ((2 * x - 37 + 25) / 8 = 5) ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1921_192195


namespace NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l1921_192156

/-- Given an arithmetic sequence {a_n} with first term a and common difference d,
    prove that the 11th term is 3.5 under the given conditions. -/
theorem arithmetic_sequence_11th_term
  (a d : ℝ)
  (h1 : a + (a + 3 * d) + (a + 6 * d) = 31.5)
  (h2 : 9 * a + (9 * 8 / 2) * d = 85.5) :
  a + 10 * d = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_11th_term_l1921_192156


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l1921_192108

/-- Two points are symmetric about the origin if their coordinates are negatives of each other -/
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_coordinates :
  ∀ (m n : ℝ), symmetric_about_origin (m, 4) (-2, n) → m = 2 ∧ n = -4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l1921_192108


namespace NUMINAMATH_CALUDE_parallelepiped_volume_theorem_l1921_192181

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  -- Side lengths of the base parallelogram
  side1 : ℝ
  side2 : ℝ
  -- Acute angle of the base parallelogram in radians
  angle : ℝ
  -- Length of the longest diagonal
  diagonal : ℝ

/-- Calculate the volume of a right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ :=
  -- This function will be defined in the proof
  sorry

/-- The main theorem to prove -/
theorem parallelepiped_volume_theorem (p : RightParallelepiped) 
  (h1 : p.side1 = 1)
  (h2 : p.side2 = 4)
  (h3 : p.angle = π / 3)  -- 60 degrees in radians
  (h4 : p.diagonal = 5) :
  volume p = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_theorem_l1921_192181


namespace NUMINAMATH_CALUDE_ball_pit_problem_l1921_192138

theorem ball_pit_problem (total_balls : ℕ) (red_fraction : ℚ) (neither_red_nor_blue : ℕ) :
  total_balls = 360 →
  red_fraction = 1/4 →
  neither_red_nor_blue = 216 →
  (total_balls - red_fraction * total_balls - neither_red_nor_blue) / 
  (total_balls - red_fraction * total_balls) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ball_pit_problem_l1921_192138


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1921_192175

/-- The area of a circle with diameter endpoints C(-2,3) and D(6,9) is 25π square units. -/
theorem circle_area_from_diameter_endpoints :
  let c : ℝ × ℝ := (-2, 3)
  let d : ℝ × ℝ := (6, 9)
  let diameter_squared := (d.1 - c.1)^2 + (d.2 - c.2)^2
  let radius := Real.sqrt diameter_squared / 2
  let area := π * radius^2
  area = 25 * π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1921_192175


namespace NUMINAMATH_CALUDE_units_digit_of_17_power_28_l1921_192132

theorem units_digit_of_17_power_28 :
  ∃ n : ℕ, 17^28 ≡ 1 [ZMOD 10] ∧ 17 ≡ 7 [ZMOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_power_28_l1921_192132


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_root_l1921_192141

theorem smallest_n_for_integer_root : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬ ∃ (k : ℕ), k^2 = 2019 - m) ∧
  ∃ (k : ℕ), k^2 = 2019 - n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_root_l1921_192141


namespace NUMINAMATH_CALUDE_both_unsuccessful_correct_both_successful_correct_exactly_one_successful_correct_at_least_one_successful_correct_at_most_one_successful_correct_l1921_192151

-- Define the propositions
variable (p q : Prop)

-- Define the shooting scenarios
def both_unsuccessful : Prop := ¬p ∧ ¬q
def both_successful : Prop := p ∧ q
def exactly_one_successful : Prop := (¬p ∧ q) ∨ (p ∧ ¬q)
def at_least_one_successful : Prop := p ∨ q
def at_most_one_successful : Prop := ¬(p ∧ q)

-- Theorem statements
theorem both_unsuccessful_correct (p q : Prop) : 
  both_unsuccessful p q ↔ ¬p ∧ ¬q := by sorry

theorem both_successful_correct (p q : Prop) : 
  both_successful p q ↔ p ∧ q := by sorry

theorem exactly_one_successful_correct (p q : Prop) : 
  exactly_one_successful p q ↔ (¬p ∧ q) ∨ (p ∧ ¬q) := by sorry

theorem at_least_one_successful_correct (p q : Prop) : 
  at_least_one_successful p q ↔ p ∨ q := by sorry

theorem at_most_one_successful_correct (p q : Prop) : 
  at_most_one_successful p q ↔ ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_both_unsuccessful_correct_both_successful_correct_exactly_one_successful_correct_at_least_one_successful_correct_at_most_one_successful_correct_l1921_192151


namespace NUMINAMATH_CALUDE_maurice_horseback_rides_l1921_192103

theorem maurice_horseback_rides (maurice_visit_rides : ℕ) 
                                (matt_with_maurice : ℕ) 
                                (matt_alone_rides : ℕ) : 
  maurice_visit_rides = 8 →
  matt_with_maurice = 8 →
  matt_alone_rides = 16 →
  matt_with_maurice + matt_alone_rides = 3 * maurice_before_visit →
  maurice_before_visit = 8 := by
  sorry

def maurice_before_visit : ℕ := 8

end NUMINAMATH_CALUDE_maurice_horseback_rides_l1921_192103


namespace NUMINAMATH_CALUDE_test_failure_rate_l1921_192137

/-- The percentage of students who failed a test, given the number of boys and girls
    and their respective pass rates. -/
def percentageFailed (numBoys numGirls : ℕ) (boyPassRate girlPassRate : ℚ) : ℚ :=
  let totalStudents := numBoys + numGirls
  let failedStudents := numBoys * (1 - boyPassRate) + numGirls * (1 - girlPassRate)
  failedStudents / totalStudents

/-- Theorem stating that given 50 boys and 100 girls, with 50% of boys passing
    and 40% of girls passing, the percentage of total students who failed is 56.67%. -/
theorem test_failure_rate : 
  percentageFailed 50 100 (1/2) (2/5) = 8500/15000 := by
  sorry

end NUMINAMATH_CALUDE_test_failure_rate_l1921_192137


namespace NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l1921_192118

/-- The number of diagonals from a single vertex in a polygon -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

theorem hexagon_diagonals_from_vertex :
  diagonals_from_vertex hexagon_sides = 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_from_vertex_l1921_192118


namespace NUMINAMATH_CALUDE_john_scores_42_points_l1921_192180

/-- The number of points John scores in a game, given the scoring pattern and game duration -/
def johnTotalPoints (pointsPer4Min : ℕ) (intervalsPer12Min : ℕ) (numPeriods : ℕ) : ℕ :=
  pointsPer4Min * intervalsPer12Min * numPeriods

/-- Theorem stating that John scores 42 points under the given conditions -/
theorem john_scores_42_points :
  let pointsPer4Min := 2 * 2 + 1 * 3  -- 2 shots worth 2 points and 1 shot worth 3 points
  let intervalsPer12Min := 12 / 4     -- Each period is 12 minutes, divided into 4-minute intervals
  let numPeriods := 2                 -- He plays for 2 periods
  johnTotalPoints pointsPer4Min intervalsPer12Min numPeriods = 42 := by
  sorry

#eval johnTotalPoints (2 * 2 + 1 * 3) (12 / 4) 2

end NUMINAMATH_CALUDE_john_scores_42_points_l1921_192180


namespace NUMINAMATH_CALUDE_eighteen_twelve_over_fiftyfour_six_l1921_192168

theorem eighteen_twelve_over_fiftyfour_six : (18 ^ 12) / (54 ^ 6) = 46656 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_twelve_over_fiftyfour_six_l1921_192168


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_b_zero_l1921_192192

theorem quadratic_inequality_solution_implies_b_zero
  (a b m : ℝ)
  (h : ∀ x, ax^2 - a*x + b < 0 ↔ m < x ∧ x < m + 1) :
  b = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_b_zero_l1921_192192


namespace NUMINAMATH_CALUDE_exists_unreachable_grid_l1921_192183

/-- Represents an 8x8 grid of natural numbers -/
def Grid := Fin 8 → Fin 8 → ℕ

/-- Represents a subgrid selection, either 3x3 or 4x4 -/
inductive Subgrid
| three : Fin 6 → Fin 6 → Subgrid
| four : Fin 5 → Fin 5 → Subgrid

/-- Applies the increment operation to a subgrid -/
def applyOperation (g : Grid) (s : Subgrid) : Grid :=
  sorry

/-- Checks if all numbers in the grid are divisible by 10 -/
def allDivisibleBy10 (g : Grid) : Prop :=
  ∀ i j, (g i j) % 10 = 0

/-- The main theorem statement -/
theorem exists_unreachable_grid :
  ∃ (initial : Grid), ¬∃ (ops : List Subgrid), allDivisibleBy10 (ops.foldl applyOperation initial) :=
sorry

end NUMINAMATH_CALUDE_exists_unreachable_grid_l1921_192183


namespace NUMINAMATH_CALUDE_sin_240_degrees_l1921_192125

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l1921_192125


namespace NUMINAMATH_CALUDE_difference_average_median_l1921_192167

theorem difference_average_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((1 + (a + 1) + (2*a + b) + (a + b + 1)) / 4) - ((a + 1 + (a + b + 1)) / 2)| = 1/4 := by
sorry

end NUMINAMATH_CALUDE_difference_average_median_l1921_192167


namespace NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l1921_192176

theorem complex_product_one_plus_i_one_minus_i : 
  (1 + Complex.I) * (1 - Complex.I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_one_plus_i_one_minus_i_l1921_192176


namespace NUMINAMATH_CALUDE_remainder_2345678901_mod_101_l1921_192119

theorem remainder_2345678901_mod_101 : 2345678901 % 101 = 12 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345678901_mod_101_l1921_192119


namespace NUMINAMATH_CALUDE_random_events_identification_l1921_192178

-- Define the type for events
inductive Event
| CoinToss : Event
| ChargeAttraction : Event
| WaterFreeze : Event
| DieRoll : Event

-- Define a predicate for random events
def isRandomEvent : Event → Prop
| Event.CoinToss => true
| Event.ChargeAttraction => false
| Event.WaterFreeze => false
| Event.DieRoll => true

-- Theorem statement
theorem random_events_identification :
  (isRandomEvent Event.CoinToss ∧ isRandomEvent Event.DieRoll) ∧
  (¬isRandomEvent Event.ChargeAttraction ∧ ¬isRandomEvent Event.WaterFreeze) :=
by sorry

end NUMINAMATH_CALUDE_random_events_identification_l1921_192178


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_l1921_192149

/-- The ellipse C₁ -/
def C₁ (x y a b : ℝ) : Prop := y^2 / a^2 + x^2 / b^2 = 1

/-- The parabola C₂ -/
def C₂ (x y p : ℝ) : Prop := x^2 = 2 * p * y

/-- The directrix l of C₂ -/
def l (y : ℝ) : Prop := y = -2

/-- Intersection point of l and C₁ -/
def intersection_point (x y : ℝ) : Prop := x = Real.sqrt 2 ∧ y = -2

/-- Common focus condition -/
def common_focus (a b p : ℝ) : Prop := sorry

theorem ellipse_parabola_intersection (a b p : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : p > 0)
  (h4 : common_focus a b p)
  (h5 : ∃ x y, C₁ x y a b ∧ l y ∧ intersection_point x y) :
  (∀ x y, C₁ x y a b ↔ y^2 / 8 + x^2 / 4 = 1) ∧
  (∀ x y, C₂ x y p ↔ x^2 = 8 * y) ∧
  (∃ min max : ℝ, min = -8 ∧ max = 2 ∧
    ∀ t : ℝ, ∃ x₃ y₃ x₄ y₄ : ℝ,
      C₁ x₃ y₃ a b ∧ C₁ x₄ y₄ a b ∧
      min < x₃ * x₄ + y₃ * y₄ ∧ x₃ * x₄ + y₃ * y₄ ≤ max) :=
sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_l1921_192149


namespace NUMINAMATH_CALUDE_slope_zero_sufficient_not_necessary_l1921_192177

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a line passing through (-1, 1)
def Line (m : ℝ) := {p : ℝ × ℝ | p.2 - 1 = m * (p.1 + 1)}

-- Define tangency
def IsTangent (l : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ l ∩ Circle ∧ ∀ q : ℝ × ℝ, q ∈ l ∩ Circle → q = p

-- Theorem statement
theorem slope_zero_sufficient_not_necessary :
  (∃ l : Set (ℝ × ℝ), l = Line 0 ∧ IsTangent l) ∧
  (∃ l : Set (ℝ × ℝ), IsTangent l ∧ l ≠ Line 0) :=
sorry

end NUMINAMATH_CALUDE_slope_zero_sufficient_not_necessary_l1921_192177


namespace NUMINAMATH_CALUDE_number_problem_l1921_192194

theorem number_problem (x : ℚ) : 
  (35 / 100 : ℚ) * x = (20 / 100 : ℚ) * 40 → x = 160 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1921_192194


namespace NUMINAMATH_CALUDE_circle_tangency_l1921_192139

structure Circle where
  center : Point
  radius : ℝ

structure Point where
  x : ℝ
  y : ℝ

def touches_internally (c1 c2 : Circle) (p : Point) : Prop := sorry

def center_on_circle (c1 c2 : Circle) : Prop := sorry

def common_chord_intersects (c1 c2 c3 : Circle) (a b : Point) : Prop := sorry

def line_intersects_circle (p1 p2 : Point) (c : Circle) (q : Point) : Prop := sorry

def is_tangent (c : Circle) (p1 p2 : Point) : Prop := sorry

theorem circle_tangency 
  (Ω Ω₁ Ω₂ : Circle) 
  (M N A B C D : Point) :
  touches_internally Ω₁ Ω M →
  touches_internally Ω₂ Ω N →
  center_on_circle Ω₂ Ω₁ →
  common_chord_intersects Ω₁ Ω₂ Ω A B →
  line_intersects_circle M A Ω₁ C →
  line_intersects_circle M B Ω₁ D →
  is_tangent Ω₂ C D := by
    sorry

end NUMINAMATH_CALUDE_circle_tangency_l1921_192139


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l1921_192140

theorem max_value_sqrt_sum (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hsum : a + b + c = 7) :
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ Real.sqrt 69 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l1921_192140


namespace NUMINAMATH_CALUDE_product_sum_not_1001_l1921_192120

theorem product_sum_not_1001 (a b c d : ℤ) (h1 : a + b = 100) (h2 : c + d = 100) : 
  a * b + c * d ≠ 1001 := by
sorry

end NUMINAMATH_CALUDE_product_sum_not_1001_l1921_192120


namespace NUMINAMATH_CALUDE_pickle_ratio_l1921_192104

/-- Prove the ratio of pickle slices Tammy can eat to Sammy can eat -/
theorem pickle_ratio (sammy tammy ron : ℕ) : 
  sammy = 15 → 
  ron = 24 → 
  ron = (80 * tammy) / 100 → 
  tammy / sammy = 2 := by
  sorry

end NUMINAMATH_CALUDE_pickle_ratio_l1921_192104


namespace NUMINAMATH_CALUDE_min_prime_angle_sum_90_l1921_192160

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem min_prime_angle_sum_90 :
  ∀ x y : ℕ,
    isPrime x →
    isPrime y →
    x + y = 90 →
    y ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_min_prime_angle_sum_90_l1921_192160


namespace NUMINAMATH_CALUDE_exist_four_numbers_squares_l1921_192199

theorem exist_four_numbers_squares : 
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 :=
by sorry

end NUMINAMATH_CALUDE_exist_four_numbers_squares_l1921_192199


namespace NUMINAMATH_CALUDE_savings_calculation_l1921_192130

theorem savings_calculation (income : ℕ) (income_ratio expenditure_ratio : ℕ) : 
  income = 19000 → 
  income_ratio = 10 → 
  expenditure_ratio = 4 → 
  income - (income * expenditure_ratio / income_ratio) = 11400 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l1921_192130


namespace NUMINAMATH_CALUDE_wire_length_between_poles_l1921_192159

theorem wire_length_between_poles (base_distance : ℝ) (short_pole_height : ℝ) (tall_pole_height : ℝ) 
  (h1 : base_distance = 20)
  (h2 : short_pole_height = 10)
  (h3 : tall_pole_height = 22) :
  Real.sqrt (base_distance ^ 2 + (tall_pole_height - short_pole_height) ^ 2) = Real.sqrt 544 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_between_poles_l1921_192159


namespace NUMINAMATH_CALUDE_river_flow_rate_l1921_192189

/-- Given a river with specified dimensions and flow rate, calculate its velocity --/
theorem river_flow_rate (depth : ℝ) (width : ℝ) (flow_rate : ℝ) (velocity : ℝ) : 
  depth = 8 →
  width = 25 →
  flow_rate = 26666.666666666668 →
  velocity = flow_rate / (depth * width) →
  velocity = 133.33333333333334 := by
  sorry

#check river_flow_rate

end NUMINAMATH_CALUDE_river_flow_rate_l1921_192189


namespace NUMINAMATH_CALUDE_myopia_functional_relationship_l1921_192197

/-- The functional relationship between the degree of myopia glasses and focal length of lenses -/
def myopia_relationship (y x : ℝ) : Prop :=
  y = 100 / x

/-- y and x are inversely proportional -/
def inverse_proportional (y x : ℝ) : Prop :=
  ∃ k : ℝ, y = k / x

theorem myopia_functional_relationship :
  ∀ y x : ℝ, 
  inverse_proportional y x → 
  (y = 400 ∧ x = 0.25) → 
  myopia_relationship y x :=
sorry

end NUMINAMATH_CALUDE_myopia_functional_relationship_l1921_192197


namespace NUMINAMATH_CALUDE_wood_square_weight_l1921_192165

/-- Represents a square piece of wood -/
structure WoodSquare where
  side_length : ℝ
  weight : ℝ

/-- Calculates the area of a square -/
def square_area (s : ℝ) : ℝ := s * s

/-- Theorem: Given two square pieces of wood with uniform density, 
    where the first piece has a side length of 3 inches and weighs 15 ounces, 
    and the second piece has a side length of 6 inches, 
    the weight of the second piece is 60 ounces. -/
theorem wood_square_weight 
  (first : WoodSquare) 
  (second : WoodSquare) 
  (h1 : first.side_length = 3) 
  (h2 : first.weight = 15) 
  (h3 : second.side_length = 6) : 
  second.weight = 60 := by
  sorry


end NUMINAMATH_CALUDE_wood_square_weight_l1921_192165


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1921_192113

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * x - 1
  f (-1/3) = 0 ∧ f 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1921_192113


namespace NUMINAMATH_CALUDE_parabola_chord_dot_product_l1921_192191

/-- The parabola y^2 = 4x with focus at (1, 0) -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- A chord passing through the focus -/
def Chord (A B : ℝ × ℝ) : Prop :=
  A ∈ Parabola ∧ B ∈ Parabola ∧ ∃ t : ℝ, (1 - t) • A + t • B = Focus

theorem parabola_chord_dot_product (A B : ℝ × ℝ) (h : Chord A B) :
    (A.1 * B.1 + A.2 * B.2 : ℝ) = -3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_chord_dot_product_l1921_192191


namespace NUMINAMATH_CALUDE_range_of_m_l1921_192153

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 + (2*m - 3)*x₁ + 1 = 0 ∧ x₂^2 + (2*m - 3)*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∃ a b, a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∀ x y, x^2/m + y^2/2 = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Define the range of m
def m_range (m : ℝ) : Prop := m < 1/2 ∨ (2 < m ∧ m ≤ 5/2)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1921_192153


namespace NUMINAMATH_CALUDE_xiaoliang_step_count_l1921_192162

/-- Represents the number of steps a person climbs to reach their floor -/
structure StepCount where
  floor : ℕ
  steps : ℕ

/-- Represents the building with information about Xiaoping and Xiaoliang -/
structure Building where
  xiaoping : StepCount
  xiaoliang : StepCount

/-- The theorem stating the number of steps Xiaoliang climbs -/
theorem xiaoliang_step_count (b : Building) 
  (h1 : b.xiaoping.floor = 5)
  (h2 : b.xiaoliang.floor = 4)
  (h3 : b.xiaoping.steps = 80) :
  b.xiaoliang.steps = 60 := by
  sorry

end NUMINAMATH_CALUDE_xiaoliang_step_count_l1921_192162


namespace NUMINAMATH_CALUDE_parabola_with_same_shape_and_vertex_l1921_192154

/-- A parabola with the same shape and opening direction as y = -3x^2 + 1 and vertex at (-1, 2) -/
theorem parabola_with_same_shape_and_vertex (x y : ℝ) : 
  y = -3 * (x + 1)^2 + 2 → 
  (∃ (a b c : ℝ), y = -3 * x^2 + b * x + c) ∧ 
  (y = -3 * (-1)^2 + 2 ∧ ∀ (h : ℝ), y ≤ -3 * (h + 1)^2 + 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_with_same_shape_and_vertex_l1921_192154


namespace NUMINAMATH_CALUDE_goose_egg_count_l1921_192135

/-- The number of goose eggs laid at a pond -/
def num_eggs : ℕ := 650

/-- The fraction of eggs that hatched -/
def hatched_fraction : ℚ := 2/3

/-- The fraction of hatched geese that survived the first month -/
def survived_month_fraction : ℚ := 3/4

/-- The fraction of geese that survived the first month but did not survive the first year -/
def not_survived_year_fraction : ℚ := 3/5

/-- The number of geese that survived the first year -/
def survived_year : ℕ := 130

theorem goose_egg_count :
  (↑num_eggs * hatched_fraction * survived_month_fraction * (1 - not_survived_year_fraction) : ℚ) = survived_year :=
sorry

end NUMINAMATH_CALUDE_goose_egg_count_l1921_192135


namespace NUMINAMATH_CALUDE_simplified_fraction_equals_ten_l1921_192115

theorem simplified_fraction_equals_ten (x y z : ℝ) 
  (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (15 * x^2 * y^4 * z^2) / (9 * x * y^3 * z) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_equals_ten_l1921_192115


namespace NUMINAMATH_CALUDE_most_efficient_numbering_system_l1921_192170

/-- Represents a numbering system for a population --/
inductive NumberingSystem
  | OneToN
  | ZeroToNMinusOne
  | TwoDigitZeroToNMinusOne
  | ThreeDigitZeroToNMinusOne

/-- Determines if a numbering system is most efficient for random number table sampling --/
def is_most_efficient (n : NumberingSystem) (population_size : ℕ) (sample_size : ℕ) : Prop :=
  n = NumberingSystem.ThreeDigitZeroToNMinusOne ∧ 
  population_size = 106 ∧ 
  sample_size = 10

/-- Theorem stating the most efficient numbering system for the given conditions --/
theorem most_efficient_numbering_system :
  ∃ (n : NumberingSystem), is_most_efficient n 106 10 :=
sorry

end NUMINAMATH_CALUDE_most_efficient_numbering_system_l1921_192170


namespace NUMINAMATH_CALUDE_rectangle_area_l1921_192198

theorem rectangle_area (length width : ℝ) (h1 : length = 5) (h2 : width = 17/20) :
  length * width = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1921_192198


namespace NUMINAMATH_CALUDE_smallest_concatenated_multiple_of_2016_l1921_192173

def concatenate_twice (n : ℕ) : ℕ :=
  n * 1000 + n

theorem smallest_concatenated_multiple_of_2016 :
  ∀ A : ℕ, A > 0 →
    (∃ k : ℕ, concatenate_twice A = 2016 * k) →
    A ≥ 288 :=
sorry

end NUMINAMATH_CALUDE_smallest_concatenated_multiple_of_2016_l1921_192173


namespace NUMINAMATH_CALUDE_trig_identity_l1921_192126

theorem trig_identity : 
  Real.sin (46 * π / 180) * Real.cos (16 * π / 180) - 
  Real.cos (314 * π / 180) * Real.sin (16 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1921_192126


namespace NUMINAMATH_CALUDE_inverse_count_mod_eleven_l1921_192131

theorem inverse_count_mod_eleven : 
  ∃ (S : Finset ℕ), 
    S.card = 10 ∧ 
    (∀ a ∈ S, a ≤ 10) ∧
    (∀ a ∈ S, ∃ b : ℕ, (a * b) % 11 = 1) ∧
    (∀ a : ℕ, a ≤ 10 → (∃ b : ℕ, (a * b) % 11 = 1) → a ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_inverse_count_mod_eleven_l1921_192131


namespace NUMINAMATH_CALUDE_gcd_lcm_product_150_180_l1921_192102

theorem gcd_lcm_product_150_180 : Nat.gcd 150 180 * Nat.lcm 150 180 = 27000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_150_180_l1921_192102


namespace NUMINAMATH_CALUDE_modulus_of_complex_l1921_192172

theorem modulus_of_complex (i : ℂ) : i * i = -1 → Complex.abs (2 * i - 5 / (2 - i)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l1921_192172


namespace NUMINAMATH_CALUDE_age_problem_l1921_192133

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l1921_192133


namespace NUMINAMATH_CALUDE_log_stack_sum_l1921_192116

theorem log_stack_sum (n : ℕ) (a l : ℕ) (h1 : n = 12) (h2 : a = 4) (h3 : l = 15) :
  n * (a + l) / 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l1921_192116


namespace NUMINAMATH_CALUDE_expression_value_l1921_192164

theorem expression_value : (2.502 + 0.064)^2 - (2.502 - 0.064)^2 / (2.502 * 0.064) = 4.002 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1921_192164


namespace NUMINAMATH_CALUDE_solution_implies_a_zero_l1921_192174

/-- Given a system of linear equations and an additional equation with parameter a,
    prove that a must be zero if the solution of the system satisfies the additional equation. -/
theorem solution_implies_a_zero (x y a : ℝ) : 
  2 * x + 7 * y = 11 →
  5 * x - 4 * y = 6 →
  3 * x - 6 * y + 2 * a = 0 →
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_zero_l1921_192174


namespace NUMINAMATH_CALUDE_min_time_less_than_3_9_l1921_192142

/-- The walking speed of each person in km/h -/
def walking_speed : ℝ := 6

/-- The speed of the motorcycle in km/h -/
def motorcycle_speed : ℝ := 90

/-- The total distance to be covered in km -/
def total_distance : ℝ := 135

/-- The maximum number of people the motorcycle can carry -/
def max_motorcycle_capacity : ℕ := 2

/-- The number of people travelling -/
def num_people : ℕ := 3

/-- The minimum time required for all people to reach the destination -/
noncomputable def min_time : ℝ := 
  (23 * total_distance) / (9 * motorcycle_speed)

theorem min_time_less_than_3_9 : min_time < 3.9 := by
  sorry

end NUMINAMATH_CALUDE_min_time_less_than_3_9_l1921_192142


namespace NUMINAMATH_CALUDE_unique_n_exists_l1921_192157

theorem unique_n_exists : ∃! n : ℤ,
  50 ≤ n ∧ n ≤ 200 ∧
  8 ∣ n ∧
  n % 6 = 4 ∧
  n % 7 = 3 ∧
  n = 136 := by
sorry

end NUMINAMATH_CALUDE_unique_n_exists_l1921_192157


namespace NUMINAMATH_CALUDE_sin_triple_angle_l1921_192144

theorem sin_triple_angle (θ : ℝ) :
  Real.sin (3 * θ) = 4 * Real.sin θ * Real.sin (π / 3 + θ) * Real.sin (2 * π / 3 + θ) := by
  sorry

end NUMINAMATH_CALUDE_sin_triple_angle_l1921_192144


namespace NUMINAMATH_CALUDE_tobys_remaining_amount_l1921_192117

/-- Calculates the remaining amount for Toby after sharing with his brothers -/
theorem tobys_remaining_amount (initial_amount : ℕ) (num_brothers : ℕ) 
  (h1 : initial_amount = 343)
  (h2 : num_brothers = 2) : 
  initial_amount - num_brothers * (initial_amount / 7) = 245 := by
  sorry

#eval 343 - 2 * (343 / 7)  -- Expected output: 245

end NUMINAMATH_CALUDE_tobys_remaining_amount_l1921_192117


namespace NUMINAMATH_CALUDE_rational_function_decomposition_l1921_192101

theorem rational_function_decomposition :
  ∃ (P Q R : ℝ), 
    (∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
      (-x^3 + 4*x^2 - 5*x + 3) / (x^4 + x^2) = P/x^2 + (Q*x + R)/(x^2 + 1)) ∧
    P = 3 ∧ Q = -1 ∧ R = 1 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_decomposition_l1921_192101


namespace NUMINAMATH_CALUDE_simplify_power_expression_l1921_192187

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^2 = 9 * y^8 := by sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l1921_192187


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l1921_192184

theorem max_value_sin_cos (a b : ℝ) (h : a^2 + b^2 ≥ 1) :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → a * Real.sin θ + b * Real.cos θ ≤ Real.sqrt (a^2 + b^2)) ∧
  (∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ a * Real.sin θ + b * Real.cos θ = Real.sqrt (a^2 + b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l1921_192184


namespace NUMINAMATH_CALUDE_equal_sampling_most_representative_l1921_192105

/-- Represents a school in the survey --/
inductive School
| A
| B
| C
| D

/-- Represents a survey method --/
structure SurveyMethod where
  schools : List School
  studentsPerSchool : Nat

/-- Defines the representativeness of a survey method --/
def representativeness (method : SurveyMethod) : ℝ :=
  sorry

/-- The number of schools in the survey --/
def totalSchools : Nat := 4

/-- The survey method that samples from all schools equally --/
def equalSamplingMethod : SurveyMethod :=
  { schools := [School.A, School.B, School.C, School.D],
    studentsPerSchool := 150 }

/-- Theorem stating that the equal sampling method is the most representative --/
theorem equal_sampling_most_representative :
  ∀ (method : SurveyMethod),
    method.schools.length = totalSchools →
    representativeness equalSamplingMethod ≥ representativeness method :=
  sorry

end NUMINAMATH_CALUDE_equal_sampling_most_representative_l1921_192105


namespace NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l1921_192100

theorem closest_whole_number_to_ratio : 
  let ratio := (10^4000 + 3*10^4002) / (2*10^4001 + 4*10^4001)
  ∃ (n : ℕ), n = 5 ∧ ∀ (m : ℕ), m ≠ n → |ratio - (n : ℝ)| < |ratio - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_closest_whole_number_to_ratio_l1921_192100


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt_181_over_12_l1921_192179

theorem sqrt_sum_fractions_equals_sqrt_181_over_12 :
  Real.sqrt (9/16 + 25/36) = Real.sqrt 181 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_equals_sqrt_181_over_12_l1921_192179


namespace NUMINAMATH_CALUDE_complementary_angles_of_same_angle_are_equal_l1921_192161

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- An angle is the complement of another if together they form 90 degrees -/
def IsComplement (α β : ℝ) : Prop := Complementary α β

theorem complementary_angles_of_same_angle_are_equal 
  (θ α β : ℝ) (h1 : IsComplement θ α) (h2 : IsComplement θ β) : α = β := by
  sorry

#check complementary_angles_of_same_angle_are_equal

end NUMINAMATH_CALUDE_complementary_angles_of_same_angle_are_equal_l1921_192161


namespace NUMINAMATH_CALUDE_custom_mul_five_three_l1921_192143

/-- Custom multiplication operation -/
def custom_mul (a b : ℤ) : ℤ := a^2 - a*b + b^2

/-- Theorem stating that 5*3 = 19 under the custom multiplication -/
theorem custom_mul_five_three : custom_mul 5 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_five_three_l1921_192143


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l1921_192182

theorem min_sum_of_squares (x y z k : ℝ) 
  (h1 : (x + 8) * (y - 8) = 0) 
  (h2 : x + y + z = k) : 
  x^2 + y^2 + z^2 ≥ 64 + k^2/2 - 4*k + 32 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l1921_192182


namespace NUMINAMATH_CALUDE_x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five_l1921_192112

theorem x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five :
  ∀ x : ℝ, x = 1 + Real.sqrt 2 → x^4 - 4*x^3 + 4*x^2 + 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_four_x_cubed_plus_four_x_squared_plus_four_equals_five_l1921_192112


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1921_192110

theorem pure_imaginary_complex_number (a : ℝ) : 
  (((2 : ℂ) - a * Complex.I) / (1 + Complex.I)).re = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1921_192110


namespace NUMINAMATH_CALUDE_sum_ratio_equals_half_l1921_192111

theorem sum_ratio_equals_half
  (a b c x y z : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 10)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 40)
  (sum_products : a*x + b*y + c*z = 20) :
  (a + b + c) / (x + y + z) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_ratio_equals_half_l1921_192111


namespace NUMINAMATH_CALUDE_leos_garden_tulips_leos_garden_tulips_proof_l1921_192129

/-- Calculates the number of tulips in Leo's garden after additions -/
theorem leos_garden_tulips (initial_carnations : ℕ) (added_carnations : ℕ) 
  (tulip_ratio : ℕ) (carnation_ratio : ℕ) : ℕ :=
  let total_carnations := initial_carnations + added_carnations
  let tulips := (total_carnations / carnation_ratio) * tulip_ratio
  tulips

/-- Proves that Leo will have 36 tulips after the additions -/
theorem leos_garden_tulips_proof :
  leos_garden_tulips 49 35 3 7 = 36 := by
  sorry

end NUMINAMATH_CALUDE_leos_garden_tulips_leos_garden_tulips_proof_l1921_192129


namespace NUMINAMATH_CALUDE_sweater_discount_percentage_l1921_192114

theorem sweater_discount_percentage (final_price saved : ℝ) : 
  final_price = 27 → saved = 3 → (saved / (final_price + saved)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_sweater_discount_percentage_l1921_192114


namespace NUMINAMATH_CALUDE_geometric_sum_property_l1921_192163

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sum_property (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  q = 2 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_property_l1921_192163


namespace NUMINAMATH_CALUDE_slope_product_negative_half_exists_line_equal_distances_l1921_192190

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define points A, B, and Q
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (-2, 0)

-- Theorem for part (I)
theorem slope_product_negative_half (x y : ℝ) :
  C x y → x ≠ 0 → (y - A.2) / (x - A.1) * (y - B.2) / (x - B.1) = -1/2 := by sorry

-- Theorem for part (II)
theorem exists_line_equal_distances :
  ∃ (M N : ℝ × ℝ), 
    C M.1 M.2 ∧ C N.1 N.2 ∧ 
    M ≠ N ∧
    M.2 = 0 ∧ N.2 = 0 ∧
    (M.1 - B.1)^2 + (M.2 - B.2)^2 = (N.1 - B.1)^2 + (N.2 - B.2)^2 := by sorry

end NUMINAMATH_CALUDE_slope_product_negative_half_exists_line_equal_distances_l1921_192190


namespace NUMINAMATH_CALUDE_brownie_ratio_l1921_192122

def total_brownies : ℕ := 15
def monday_brownies : ℕ := 5

def tuesday_brownies : ℕ := total_brownies - monday_brownies

theorem brownie_ratio :
  (tuesday_brownies : ℚ) / monday_brownies = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brownie_ratio_l1921_192122


namespace NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l1921_192152

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem tangent_line_and_minimum_value (a : ℝ) :
  (∀ x, x > 0 → f a x = a * x^2 - (a + 2) * x + Real.log x) →
  (a = 1 → ∀ y, y = -2 ↔ y = f 1 1 ∧ (∀ h, h ≠ 0 → (f 1 (1 + h) - f 1 1) / h = 0)) ∧
  (a > 0 → (∀ x, x ∈ Set.Icc 1 (Real.exp 1) → f a x ≥ -2) ∧ 
           (∃ x, x ∈ Set.Icc 1 (Real.exp 1) ∧ f a x = -2) →
           a ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_minimum_value_l1921_192152


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l1921_192107

-- Problem 1
theorem solve_equation_1 : 
  let f : ℝ → ℝ := λ x => -3*x*(2*x-3)+(2*x-3)
  ∃ x₁ x₂ : ℝ, x₁ = 3/2 ∧ x₂ = 1/3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Problem 2
theorem solve_equation_2 :
  let f : ℝ → ℝ := λ x => x^2 - 6*x + 4
  ∃ x₁ x₂ : ℝ, x₁ = 3 + Real.sqrt 5 ∧ x₂ = 3 - Real.sqrt 5 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

-- Problem 3
theorem solve_equation_3 :
  let f : ℝ → ℝ := λ x => 4 / (x^2 - 4) + 1 / (x - 2) + 1
  ∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_l1921_192107
