import Mathlib

namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3409_340905

/-- Expresses the repeating decimal 0.7̄8̄ as a rational number -/
theorem repeating_decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ (0.7 + 0.08 / (1 - 0.1) : ℚ) = n / d ∧ n = 781 ∧ d = 990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3409_340905


namespace NUMINAMATH_CALUDE_gcd_105_90_l3409_340980

theorem gcd_105_90 : Nat.gcd 105 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_90_l3409_340980


namespace NUMINAMATH_CALUDE_range_of_a_l3409_340925

open Set

-- Define sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = ∅ ↔ 1 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3409_340925


namespace NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l3409_340941

theorem same_root_implies_a_equals_three (a : ℝ) :
  (∃ x : ℝ, 3 * x - 2 * a = 0 ∧ 2 * x + 3 * a - 13 = 0) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_same_root_implies_a_equals_three_l3409_340941


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3409_340971

theorem intersection_of_lines :
  ∃! (x y : ℚ), 5 * x - 3 * y = 7 ∧ 4 * x + 2 * y = 18 ∧ x = 34 / 11 ∧ y = 31 / 11 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3409_340971


namespace NUMINAMATH_CALUDE_polynomial_derivative_value_l3409_340939

theorem polynomial_derivative_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (3*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 240 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_value_l3409_340939


namespace NUMINAMATH_CALUDE_quadratic_roots_l3409_340963

theorem quadratic_roots (m : ℝ) : 
  ((-5 : ℝ)^2 + m * (-5) - 10 = 0) → ((2 : ℝ)^2 + m * 2 - 10 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3409_340963


namespace NUMINAMATH_CALUDE_statue_cost_proof_l3409_340978

theorem statue_cost_proof (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) :
  selling_price = 620 →
  profit_percentage = 0.25 →
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 496 :=
by sorry

end NUMINAMATH_CALUDE_statue_cost_proof_l3409_340978


namespace NUMINAMATH_CALUDE_circle_centers_and_m_l3409_340913

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0
def circle_C2 (x y m : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + m = 0

-- Define external tangency
def externally_tangent (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧ 
  ∀ (x' y' : ℝ), (C1 x' y' → (x' - x)^2 + (y' - y)^2 > 0) ∧
                 (C2 x' y' → (x' - x)^2 + (y' - y)^2 > 0)

-- Theorem statement
theorem circle_centers_and_m :
  externally_tangent circle_C1 (circle_C2 · · (-3)) →
  (∃ (x y : ℝ), circle_C1 x y ∧ x = -1 ∧ y = -1) ∧
  (∀ m : ℝ, externally_tangent circle_C1 (circle_C2 · · m) → m = -3) :=
sorry

end NUMINAMATH_CALUDE_circle_centers_and_m_l3409_340913


namespace NUMINAMATH_CALUDE_forty_percent_of_three_fifths_of_150_forty_percent_of_three_fifths_of_150_equals_36_l3409_340900

theorem forty_percent_of_three_fifths_of_150 : ℚ :=
  let number : ℚ := 150
  let three_fifths : ℚ := 3 / 5
  let forty_percent : ℚ := 40 / 100
  forty_percent * (three_fifths * number)
  
-- Prove that the above expression equals 36
theorem forty_percent_of_three_fifths_of_150_equals_36 :
  forty_percent_of_three_fifths_of_150 = 36 := by sorry

end NUMINAMATH_CALUDE_forty_percent_of_three_fifths_of_150_forty_percent_of_three_fifths_of_150_equals_36_l3409_340900


namespace NUMINAMATH_CALUDE_shirt_ironing_time_l3409_340950

/-- The number of days per week Hayden irons his clothes -/
def days_per_week : ℕ := 5

/-- The number of minutes Hayden spends ironing his pants each day -/
def pants_ironing_time : ℕ := 3

/-- The total number of minutes Hayden spends ironing over 4 weeks -/
def total_ironing_time : ℕ := 160

/-- The number of weeks in the period -/
def num_weeks : ℕ := 4

theorem shirt_ironing_time :
  ∃ (shirt_time : ℕ),
    shirt_time * (days_per_week * num_weeks) = 
      total_ironing_time - (pants_ironing_time * days_per_week * num_weeks) ∧
    shirt_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_shirt_ironing_time_l3409_340950


namespace NUMINAMATH_CALUDE_range_of_a_l3409_340999

-- Define the complex number z
def z (a : ℝ) : ℂ := 1 + a * Complex.I

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (Complex.abs (z a) ≤ 2) ↔ (a ≥ -Real.sqrt 3 ∧ a ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3409_340999


namespace NUMINAMATH_CALUDE_adults_attending_concert_concert_attendance_proof_l3409_340932

/-- The number of adults attending a music festival concert, given ticket prices and total revenue --/
theorem adults_attending_concert (adult_price : ℕ) (child_price : ℕ) (num_children : ℕ) (total_revenue : ℕ) : ℕ :=
  let adults : ℕ := (total_revenue - num_children * child_price) / adult_price
  adults

/-- Proof that 183 adults attended the concert given the specific conditions --/
theorem concert_attendance_proof :
  adults_attending_concert 26 13 28 5122 = 183 := by
  sorry

end NUMINAMATH_CALUDE_adults_attending_concert_concert_attendance_proof_l3409_340932


namespace NUMINAMATH_CALUDE_geometric_sequences_exist_and_unique_l3409_340931

/-- Three geometric sequences satisfying the given conditions -/
def geometric_sequences (a q : ℝ) : Fin 3 → ℕ → ℝ
| ⟨0, _⟩ => λ n => a * (q - 2) ^ n
| ⟨1, _⟩ => λ n => 2 * a * (q - 1) ^ n
| ⟨2, _⟩ => λ n => 4 * a * q ^ n

/-- The theorem stating the existence and uniqueness of the geometric sequences -/
theorem geometric_sequences_exist_and_unique :
  ∃ (a q : ℝ),
    (∀ i : Fin 3, geometric_sequences a q i 0 = a * (2 ^ i.val)) ∧
    (geometric_sequences a q 1 1 - geometric_sequences a q 0 1 =
     geometric_sequences a q 2 1 - geometric_sequences a q 1 1) ∧
    (geometric_sequences a q 0 1 + geometric_sequences a q 1 1 + geometric_sequences a q 2 1 = 24) ∧
    (geometric_sequences a q 0 0 + geometric_sequences a q 1 0 + geometric_sequences a q 2 0 = 84) ∧
    ((a = 1 ∧ q = 4) ∨ (a = 192 / 31 ∧ q = 9 / 8)) := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequences_exist_and_unique_l3409_340931


namespace NUMINAMATH_CALUDE_rate_increase_is_33_percent_l3409_340955

/-- Represents the work team's processing scenario -/
structure WorkScenario where
  initial_items : ℕ
  total_time : ℕ
  worked_time : ℕ
  additional_items : ℕ

/-- Calculates the required rate increase percentage -/
def required_rate_increase (scenario : WorkScenario) : ℚ :=
  let initial_rate := scenario.initial_items / scenario.total_time
  let processed_items := initial_rate * scenario.worked_time
  let remaining_items := scenario.initial_items - processed_items + scenario.additional_items
  let remaining_time := scenario.total_time - scenario.worked_time
  let new_rate := remaining_items / remaining_time
  (new_rate - initial_rate) / initial_rate * 100

/-- The main theorem stating that the required rate increase is 33% -/
theorem rate_increase_is_33_percent (scenario : WorkScenario) 
  (h1 : scenario.initial_items = 1250)
  (h2 : scenario.total_time = 10)
  (h3 : scenario.worked_time = 6)
  (h4 : scenario.additional_items = 165) :
  required_rate_increase scenario = 33 := by
  sorry

end NUMINAMATH_CALUDE_rate_increase_is_33_percent_l3409_340955


namespace NUMINAMATH_CALUDE_negative_sum_l3409_340988

theorem negative_sum (x w : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hw : -2 < w ∧ w < -1) : 
  x + w < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_sum_l3409_340988


namespace NUMINAMATH_CALUDE_average_sleep_time_l3409_340906

def sleep_times : List ℝ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_sleep_time_l3409_340906


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l3409_340985

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (nuts_fraction : ℚ) (berries_fraction : ℚ) (cream_fraction : ℚ) (choc_chips_fraction : ℚ)
  (h_total : total_pies = 48)
  (h_nuts : nuts_fraction = 1/3)
  (h_berries : berries_fraction = 1/2)
  (h_cream : cream_fraction = 3/5)
  (h_choc_chips : choc_chips_fraction = 1/4) :
  ∃ (max_without : ℕ), max_without ≤ total_pies ∧ 
  max_without = total_pies - ⌈cream_fraction * total_pies⌉ ∧
  max_without = 19 :=
sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l3409_340985


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3409_340901

theorem quadratic_square_of_binomial (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 116*x + c = (x + a)^2) → c = 3364 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l3409_340901


namespace NUMINAMATH_CALUDE_bobs_weight_l3409_340928

theorem bobs_weight (j b : ℝ) : 
  j + b = 200 → 
  b - 3 * j = b / 4 → 
  b = 2400 / 14 := by
sorry

end NUMINAMATH_CALUDE_bobs_weight_l3409_340928


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l3409_340920

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) :
  let juice_amount : ℝ := (3 / 4) * C
  let cups : ℕ := 8
  let juice_per_cup : ℝ := juice_amount / cups
  let percent_per_cup : ℝ := (juice_per_cup / C) * 100
  percent_per_cup = 9.375 := by
  sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l3409_340920


namespace NUMINAMATH_CALUDE_sequence_properties_l3409_340953

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) (k : ℝ) : ℝ := k * n^2 + n

/-- The nth term of the sequence -/
def a (n : ℕ+) (k : ℝ) : ℝ := k * (2 * n - 1) + 1

theorem sequence_properties (k : ℝ) :
  (∀ n : ℕ+, S n k - S (n-1) k = a n k) ∧
  (∀ m : ℕ+, (a (2*m) k)^2 = (a m k) * (a (4*m) k)) →
  k = 1/3 := by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3409_340953


namespace NUMINAMATH_CALUDE_smallest_of_three_successive_numbers_l3409_340916

theorem smallest_of_three_successive_numbers :
  ∀ n : ℕ, n * (n + 1) * (n + 2) = 1059460 → n = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_successive_numbers_l3409_340916


namespace NUMINAMATH_CALUDE_xy_sum_reciprocals_l3409_340964

theorem xy_sum_reciprocals (x y : ℝ) (θ : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_theta : ∀ (n : ℤ), θ ≠ π / 2 * n)
  (h_eq1 : Real.sin θ / x = Real.cos θ / y)
  (h_eq2 : Real.cos θ ^ 4 / x ^ 4 + Real.sin θ ^ 4 / y ^ 4 = 
           97 * Real.sin (2 * θ) / (x ^ 3 * y + y ^ 3 * x)) :
  x / y + y / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_reciprocals_l3409_340964


namespace NUMINAMATH_CALUDE_problem_solution_l3409_340989

theorem problem_solution (a b : ℝ) (h1 : a - b = 7) (h2 : a * b = 18) :
  a^2 + b^2 = 85 ∧ (a + b)^2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3409_340989


namespace NUMINAMATH_CALUDE_carpet_dimensions_l3409_340915

/-- A rectangular carpet with integer side lengths. -/
structure Carpet where
  length : ℕ
  width : ℕ

/-- A rectangular room. -/
structure Room where
  length : ℕ
  width : ℕ

/-- Predicate to check if a carpet fits perfectly (diagonally) in a room. -/
def fits_perfectly (c : Carpet) (r : Room) : Prop :=
  (c.length ^ 2 + c.width ^ 2 : ℕ) = r.length ^ 2 + r.width ^ 2

/-- The main theorem about the carpet dimensions. -/
theorem carpet_dimensions :
  ∀ (c : Carpet) (r1 r2 : Room),
    r1.width = 50 →
    r2.width = 38 →
    r1.length = r2.length →
    fits_perfectly c r1 →
    fits_perfectly c r2 →
    c.length = 50 ∧ c.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_carpet_dimensions_l3409_340915


namespace NUMINAMATH_CALUDE_vector_sum_components_l3409_340903

/-- Given 2D vectors a, b, and c, prove that 3a - 2b + c is equal to
    (3ax - 2bx + cx, 3ay - 2by + cy) where ax, ay, bx, by, cx, and cy
    are the respective x and y components of vectors a, b, and c. -/
theorem vector_sum_components (a b c : ℝ × ℝ) :
  3 • a - 2 • b + c = (3 * a.1 - 2 * b.1 + c.1, 3 * a.2 - 2 * b.2 + c.2) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_components_l3409_340903


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_m_values_l3409_340907

theorem ellipse_eccentricity_m_values (m : ℝ) :
  (∃ x y : ℝ, x^2 / 9 + y^2 / (m + 9) = 1) →
  (∃ c : ℝ, c^2 / (m + 9) = 1/4) →
  (m = -9/4 ∨ m = 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_m_values_l3409_340907


namespace NUMINAMATH_CALUDE_grey_pairs_coincide_l3409_340949

/-- Represents the number of triangles of each color in one half of the shape -/
structure TriangleCounts where
  orange : Nat
  green : Nat
  grey : Nat

/-- Represents the number of pairs of triangles that coincide when folded -/
structure CoincidingPairs where
  orange : Nat
  green : Nat
  orangeGrey : Nat

theorem grey_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) :
  counts.orange = 4 →
  counts.green = 6 →
  counts.grey = 9 →
  pairs.orange = 3 →
  pairs.green = 4 →
  pairs.orangeGrey = 1 →
  ∃ (grey_pairs : Nat), grey_pairs = 6 ∧ 
    grey_pairs = counts.grey - (pairs.orangeGrey + (counts.green - 2 * pairs.green)) :=
by sorry

end NUMINAMATH_CALUDE_grey_pairs_coincide_l3409_340949


namespace NUMINAMATH_CALUDE_ellipse_equation_l3409_340934

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line with slope m passing through point p -/
structure Line where
  m : ℝ
  p : Point

/-- The theorem statement -/
theorem ellipse_equation (E : Ellipse) (F : Point) (l : Line) (M : Point) :
  F.x = 3 ∧ F.y = 0 ∧  -- Right focus at (3,0)
  l.m = 1/2 ∧ l.p = F ∧  -- Line with slope 1/2 passing through F
  M.x = 1 ∧ M.y = -1 ∧  -- Midpoint at (1,-1)
  (∃ A B : Point, A ≠ B ∧
    (A.x^2 / E.a^2 + A.y^2 / E.b^2 = 1) ∧
    (B.x^2 / E.a^2 + B.y^2 / E.b^2 = 1) ∧
    (A.y - F.y = l.m * (A.x - F.x)) ∧
    (B.y - F.y = l.m * (B.x - F.x)) ∧
    M.x = (A.x + B.x) / 2 ∧
    M.y = (A.y + B.y) / 2) →
  E.a^2 = 18 ∧ E.b^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3409_340934


namespace NUMINAMATH_CALUDE_A_knitting_time_l3409_340944

/-- The number of days it takes person A to knit a pair of socks -/
def days_A : ℝ := by sorry

/-- The number of days it takes person B to knit a pair of socks -/
def days_B : ℝ := 6

/-- The number of days it takes A and B together to knit two pairs of socks -/
def days_together : ℝ := 4

/-- The number of pairs of socks A and B knit together in 4 days -/
def pairs_together : ℝ := 2

theorem A_knitting_time :
  (1 / days_A + 1 / days_B) * days_together = pairs_together ∧ days_A = 3 := by sorry

end NUMINAMATH_CALUDE_A_knitting_time_l3409_340944


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l3409_340986

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/5 + 1/2, 1/5 + 1/6, 1/5 + 1/4, 1/5 + 1/8, 1/5 + 1/9]
  (∀ x ∈ sums, 1/5 + 1/2 ≥ x) ∧ (1/5 + 1/2 = 7/10) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l3409_340986


namespace NUMINAMATH_CALUDE_x_range_l3409_340983

theorem x_range (x : ℝ) (h1 : 1 / x ≤ 3) (h2 : 1 / x ≥ -2) : x ≥ 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3409_340983


namespace NUMINAMATH_CALUDE_drowned_ratio_l3409_340926

/-- Proves the ratio of drowned cows to drowned sheep given the initial conditions -/
theorem drowned_ratio (initial_sheep initial_cows initial_dogs : ℕ)
  (drowned_sheep : ℕ) (total_survived : ℕ) :
  initial_sheep = 20 →
  initial_cows = 10 →
  initial_dogs = 14 →
  drowned_sheep = 3 →
  total_survived = 35 →
  (initial_cows - (total_survived - (initial_sheep - drowned_sheep) - initial_dogs)) /
  drowned_sheep = 2 := by
  sorry

end NUMINAMATH_CALUDE_drowned_ratio_l3409_340926


namespace NUMINAMATH_CALUDE_highest_salary_grade_is_six_l3409_340974

/-- The minimum salary grade -/
def min_grade : ℕ := 1

/-- Function to calculate hourly wage based on salary grade -/
def hourly_wage (s : ℕ) : ℝ := 7.50 + 0.25 * (s - 1)

/-- The difference in hourly wage between the highest and lowest grade -/
def wage_difference : ℝ := 1.25

theorem highest_salary_grade_is_six :
  ∃ (max_grade : ℕ),
    (∀ (s : ℕ), min_grade ≤ s ∧ s ≤ max_grade) ∧
    (hourly_wage max_grade = hourly_wage min_grade + wage_difference) ∧
    max_grade = 6 :=
by sorry

end NUMINAMATH_CALUDE_highest_salary_grade_is_six_l3409_340974


namespace NUMINAMATH_CALUDE_customer_satisfaction_probability_l3409_340970

/-- The probability that a dissatisfied customer leaves an angry review -/
def prob_angry_given_dissatisfied : ℝ := 0.8

/-- The probability that a satisfied customer leaves a positive review -/
def prob_positive_given_satisfied : ℝ := 0.15

/-- The number of angry reviews received -/
def angry_reviews : ℕ := 60

/-- The number of positive reviews received -/
def positive_reviews : ℕ := 20

/-- The probability that a customer is satisfied with the service -/
def prob_satisfied : ℝ := 0.64

theorem customer_satisfaction_probability :
  prob_satisfied = 0.64 :=
sorry

end NUMINAMATH_CALUDE_customer_satisfaction_probability_l3409_340970


namespace NUMINAMATH_CALUDE_candidate_x_wins_by_16_percent_l3409_340997

/-- Represents the election scenario with given conditions -/
structure ElectionScenario where
  repubRatio : ℚ
  demRatio : ℚ
  repubVoteX : ℚ
  demVoteX : ℚ
  (ratio_positive : repubRatio > 0 ∧ demRatio > 0)
  (vote_percentages : repubVoteX ≥ 0 ∧ repubVoteX ≤ 1 ∧ demVoteX ≥ 0 ∧ demVoteX ≤ 1)

/-- Calculates the percentage by which candidate X is expected to win -/
def winPercentage (e : ElectionScenario) : ℚ :=
  let totalVoters := e.repubRatio + e.demRatio
  let votesForX := e.repubRatio * e.repubVoteX + e.demRatio * e.demVoteX
  let votesForY := totalVoters - votesForX
  (votesForX - votesForY) / totalVoters * 100

/-- Theorem stating that under the given conditions, candidate X wins by 16% -/
theorem candidate_x_wins_by_16_percent :
  ∀ e : ElectionScenario,
    e.repubRatio = 3 ∧
    e.demRatio = 2 ∧
    e.repubVoteX = 4/5 ∧
    e.demVoteX = 1/4 →
    winPercentage e = 16 := by
  sorry


end NUMINAMATH_CALUDE_candidate_x_wins_by_16_percent_l3409_340997


namespace NUMINAMATH_CALUDE_work_completion_theorem_l3409_340952

/-- Represents the number of days needed to complete the work -/
def total_days_x : ℝ := 30

/-- Represents the number of days needed to complete the work -/
def total_days_y : ℝ := 15

/-- Represents the number of days x needs to finish the remaining work -/
def remaining_days_x : ℝ := 10.000000000000002

/-- Represents the number of days y worked before leaving -/
def days_y_worked : ℝ := 10

theorem work_completion_theorem :
  days_y_worked * (1 / total_days_y) + remaining_days_x * (1 / total_days_x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l3409_340952


namespace NUMINAMATH_CALUDE_triangle_area_l3409_340987

/-- The area of a triangle with base 9 cm and height 12 cm is 54 cm² -/
theorem triangle_area : 
  let base : ℝ := 9
  let height : ℝ := 12
  (1/2 : ℝ) * base * height = 54
  := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3409_340987


namespace NUMINAMATH_CALUDE_robins_pieces_l3409_340981

theorem robins_pieces (gum_packages : ℕ) (candy_packages : ℕ) (pieces_per_package : ℕ) : 
  gum_packages = 28 → candy_packages = 14 → pieces_per_package = 6 →
  gum_packages * pieces_per_package + candy_packages * pieces_per_package = 252 := by
sorry

end NUMINAMATH_CALUDE_robins_pieces_l3409_340981


namespace NUMINAMATH_CALUDE_prob_sum_8_twice_eq_l3409_340945

/-- The number of sides on each die -/
def num_sides : ℕ := 7

/-- The set of possible outcomes for a single die roll -/
def die_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- The probability of rolling a sum of 8 with two dice -/
def prob_sum_8 : ℚ := 7 / 49

/-- The probability of rolling a sum of 8 twice in a row with two dice -/
def prob_sum_8_twice : ℚ := (prob_sum_8) * (prob_sum_8)

/-- Theorem: The probability of rolling a sum of 8 twice in a row
    with two 7-sided dice (numbered 1 to 7) is equal to 49/2401 -/
theorem prob_sum_8_twice_eq : prob_sum_8_twice = 49 / 2401 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_8_twice_eq_l3409_340945


namespace NUMINAMATH_CALUDE_prism_has_315_edges_l3409_340958

/-- A prism is a polyhedron with two congruent and parallel faces (bases) connected by rectangular faces. -/
structure Prism where
  num_edges : ℕ

/-- The number of edges in a prism is always a multiple of 3. -/
axiom prism_edges_multiple_of_three (p : Prism) : ∃ k : ℕ, p.num_edges = 3 * k

/-- The prism has more than 310 edges. -/
axiom edges_greater_than_310 (p : Prism) : p.num_edges > 310

/-- The prism has fewer than 320 edges. -/
axiom edges_less_than_320 (p : Prism) : p.num_edges < 320

/-- The number of edges in the prism is odd. -/
axiom edges_odd (p : Prism) : Odd p.num_edges

theorem prism_has_315_edges (p : Prism) : p.num_edges = 315 := by
  sorry

end NUMINAMATH_CALUDE_prism_has_315_edges_l3409_340958


namespace NUMINAMATH_CALUDE_squares_after_six_operations_l3409_340973

/-- Calculates the number of squares after n operations -/
def num_squares (n : ℕ) : ℕ := 5 + 3 * n

/-- The number of squares after 6 operations is 29 -/
theorem squares_after_six_operations :
  num_squares 6 = 29 := by
  sorry

end NUMINAMATH_CALUDE_squares_after_six_operations_l3409_340973


namespace NUMINAMATH_CALUDE_equal_difference_implies_square_equal_difference_equal_difference_and_equal_square_difference_implies_constant_l3409_340968

/-- A sequence is an "equal difference" sequence if the difference between consecutive terms is constant. -/
def IsEqualDifference (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is an "equal square difference" sequence if the difference between consecutive squared terms is constant. -/
def IsEqualSquareDifference (a : ℕ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ n : ℕ, (a (n + 1))^2 - (a n)^2 = p

/-- If a sequence is an "equal difference" sequence, then its square is also an "equal difference" sequence. -/
theorem equal_difference_implies_square_equal_difference (a : ℕ → ℝ) :
    IsEqualDifference a → IsEqualDifference (fun n ↦ (a n)^2) := by sorry

/-- If a sequence is both an "equal difference" sequence and an "equal square difference" sequence,
    then it is a constant sequence. -/
theorem equal_difference_and_equal_square_difference_implies_constant (a : ℕ → ℝ) :
    IsEqualDifference a → IsEqualSquareDifference a → ∃ c : ℝ, ∀ n : ℕ, a n = c := by sorry

end NUMINAMATH_CALUDE_equal_difference_implies_square_equal_difference_equal_difference_and_equal_square_difference_implies_constant_l3409_340968


namespace NUMINAMATH_CALUDE_greatest_n_roots_on_unit_circle_l3409_340984

theorem greatest_n_roots_on_unit_circle : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (z : ℂ), z ≠ 0 → (z + 1)^n = z^n + 1 → Complex.abs z = 1) ∧
  (∀ (m : ℕ), m > n → ∃ (w : ℂ), w ≠ 0 ∧ (w + 1)^m = w^m + 1 ∧ Complex.abs w ≠ 1) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_greatest_n_roots_on_unit_circle_l3409_340984


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l3409_340940

/-- A quadrilateral with given side lengths -/
structure Quadrilateral :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)

/-- The largest possible inscribed circle in a quadrilateral -/
def largest_inscribed_circle (q : Quadrilateral) : ℝ := sorry

/-- Theorem: The radius of the largest inscribed circle in the given quadrilateral is 2√6 -/
theorem largest_inscribed_circle_radius :
  ∀ q : Quadrilateral,
    q.AB = 15 ∧ q.BC = 10 ∧ q.CD = 8 ∧ q.DA = 13 →
    largest_inscribed_circle q = 2 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_radius_l3409_340940


namespace NUMINAMATH_CALUDE_increasing_function_condition_l3409_340935

/-- A function f(x) = x - 5/x - a*ln(x) is increasing on [1, +∞) if and only if a ≤ 2√5 -/
theorem increasing_function_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → Monotone (fun x => x - 5 / x - a * Real.log x)) ↔ a ≤ 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l3409_340935


namespace NUMINAMATH_CALUDE_f_composition_at_one_l3409_340917

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^x else Real.log (x - 1)

theorem f_composition_at_one : f (f 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_at_one_l3409_340917


namespace NUMINAMATH_CALUDE_other_items_percentage_correct_l3409_340994

/-- The percentage of money spent on other items in Jill's shopping trip -/
def other_items_percentage : ℝ := 
  let total := 100
  let clothing_percentage := 45
  let food_percentage := 45
  let clothing_tax_rate := 5
  let other_items_tax_rate := 10
  let total_tax_percentage := 3.25
  10

/-- Theorem stating that the percentage spent on other items is correct -/
theorem other_items_percentage_correct : 
  let total := 100
  let clothing_percentage := 45
  let food_percentage := 45
  let clothing_tax_rate := 5
  let other_items_tax_rate := 10
  let total_tax_percentage := 3.25
  (clothing_percentage + food_percentage + other_items_percentage = total) ∧
  (clothing_tax_rate * clothing_percentage / 100 + 
   other_items_tax_rate * other_items_percentage / 100 = total_tax_percentage) := by
  sorry

#check other_items_percentage_correct

end NUMINAMATH_CALUDE_other_items_percentage_correct_l3409_340994


namespace NUMINAMATH_CALUDE_patients_per_doctor_l3409_340947

/-- Given a hospital with 400 patients and 16 doctors, prove that each doctor takes care of 25 patients. -/
theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) :
  total_patients = 400 → total_doctors = 16 →
  total_patients / total_doctors = 25 := by
  sorry

end NUMINAMATH_CALUDE_patients_per_doctor_l3409_340947


namespace NUMINAMATH_CALUDE_range_of_m_l3409_340977

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (1/4 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define the set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x + m^2 ≥ 1}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

-- State the theorem
theorem range_of_m :
  (∀ x, p x → q m x) ↔ (m ≥ 3/4 ∨ m ≤ -3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3409_340977


namespace NUMINAMATH_CALUDE_ratio_calculation_l3409_340910

theorem ratio_calculation (x y a b : ℚ) 
  (h1 : x / y = 3)
  (h2 : (2 * a - x) / (3 * b - y) = 3) :
  a / b = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_calculation_l3409_340910


namespace NUMINAMATH_CALUDE_star_op_result_l3409_340911

/-- The * operation for non-zero integers -/
def star_op (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem stating the result of the star operation given the conditions -/
theorem star_op_result (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_cond : a + b = 15) (prod_cond : a * b = 36) : 
  star_op a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_op_result_l3409_340911


namespace NUMINAMATH_CALUDE_williams_tickets_l3409_340918

theorem williams_tickets (initial_tickets : ℕ) : 
  initial_tickets + 3 = 18 → initial_tickets = 15 := by
  sorry

end NUMINAMATH_CALUDE_williams_tickets_l3409_340918


namespace NUMINAMATH_CALUDE_price_reduction_proof_l3409_340919

theorem price_reduction_proof (current_price : ℝ) (reduction_percentage : ℝ) (claimed_reduction : ℝ) : 
  current_price = 45 ∧ reduction_percentage = 0.1 ∧ claimed_reduction = 10 →
  (100 / (100 - reduction_percentage * 100) * current_price) - current_price ≠ claimed_reduction :=
by
  sorry

end NUMINAMATH_CALUDE_price_reduction_proof_l3409_340919


namespace NUMINAMATH_CALUDE_exponent_rule_l3409_340956

theorem exponent_rule (a : ℝ) (m : ℤ) : a^(2*m + 2) = a^(2*m) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_l3409_340956


namespace NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l3409_340965

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length_line_circle_intersection (x y : ℝ) :
  (3 * x + 4 * y - 5 = 0) →  -- Line equation
  (x^2 + y^2 = 4) →          -- Circle equation
  ∃ (A B : ℝ × ℝ),           -- Intersection points A and B
    (3 * A.1 + 4 * A.2 - 5 = 0) ∧
    (A.1^2 + A.2^2 = 4) ∧
    (3 * B.1 + 4 * B.2 - 5 = 0) ∧
    (B.1^2 + B.2^2 = 4) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_chord_length_line_circle_intersection_l3409_340965


namespace NUMINAMATH_CALUDE_polynomial_multiple_power_coefficients_l3409_340922

theorem polynomial_multiple_power_coefficients 
  (p : Polynomial ℝ) (n : ℕ) (hn : n > 0) :
  ∃ q : Polynomial ℝ, q ≠ 0 ∧ 
  ∀ i : ℕ, (p * q).coeff i ≠ 0 → ∃ k : ℕ, i = n * k :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiple_power_coefficients_l3409_340922


namespace NUMINAMATH_CALUDE_trajectory_of_M_is_ellipse_l3409_340929

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 32 = 0

-- Define point A
def point_A : ℝ × ℝ := (2, 0)

-- Define a moving point P on circle C
def point_P (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of perpendicular bisector of AP and line PC
def point_M (x y : ℝ) : Prop :=
  ∃ (px py : ℝ), point_P px py ∧
  ((x - 2)^2 + y^2 = (x - px)^2 + (y - py)^2) ∧
  ((x - 2) * (px - 2) + y * py = 0)

-- Theorem: The trajectory of point M is an ellipse
theorem trajectory_of_M_is_ellipse :
  ∀ (x y : ℝ), point_M x y ↔ x^2/9 + y^2/5 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_is_ellipse_l3409_340929


namespace NUMINAMATH_CALUDE_problem_statement_l3409_340914

theorem problem_statement : (2112 - 2021)^2 / 169 = 49 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3409_340914


namespace NUMINAMATH_CALUDE_solve_laboratory_budget_l3409_340904

def laboratory_budget_problem (total_budget flask_cost : ℕ) : Prop :=
  let test_tube_cost := (2 * flask_cost) / 3
  let safety_gear_cost := test_tube_cost / 2
  let total_spent := flask_cost + test_tube_cost + safety_gear_cost
  let remaining_budget := total_budget - total_spent
  total_budget = 325 ∧ flask_cost = 150 → remaining_budget = 25

theorem solve_laboratory_budget :
  laboratory_budget_problem 325 150 := by
  sorry

end NUMINAMATH_CALUDE_solve_laboratory_budget_l3409_340904


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3409_340927

theorem sufficient_not_necessary_condition (m : ℝ) : m = 1/2 →
  m > 0 ∧
  (∀ x : ℝ, 0 < x ∧ x < m → x * (x - 1) < 0) ∧
  (∃ x : ℝ, x * (x - 1) < 0 ∧ ¬(0 < x ∧ x < m)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3409_340927


namespace NUMINAMATH_CALUDE_stone_breaking_loss_l3409_340908

/-- Represents the properties of a precious stone -/
structure Stone where
  weight : ℝ
  price : ℝ
  k : ℝ

/-- Calculates the price of a stone given its weight and constant k -/
def calculatePrice (weight : ℝ) (k : ℝ) : ℝ := k * weight^3

/-- Calculates the loss when a stone breaks -/
def calculateLoss (original : Stone) (piece1 : Stone) (piece2 : Stone) : ℝ :=
  original.price - (piece1.price + piece2.price)

theorem stone_breaking_loss (original : Stone) (piece1 : Stone) (piece2 : Stone) :
  original.weight = 28 ∧ 
  original.price = 60000 ∧ 
  original.price = calculatePrice original.weight original.k ∧
  piece1.weight = (17 / 28) * original.weight ∧
  piece2.weight = (11 / 28) * original.weight ∧
  piece1.k = original.k ∧
  piece2.k = original.k ∧
  piece1.price = calculatePrice piece1.weight piece1.k ∧
  piece2.price = calculatePrice piece2.weight piece2.k →
  abs (calculateLoss original piece1 piece2 - 42933.33) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_stone_breaking_loss_l3409_340908


namespace NUMINAMATH_CALUDE_group_photo_arrangements_eq_12_l3409_340943

/-- The number of ways to arrange 1 teacher, 2 female students, and 2 male students in a row,
    where the two female students are separated only by the teacher. -/
def group_photo_arrangements : ℕ :=
  let teacher : ℕ := 1
  let female_students : ℕ := 2
  let male_students : ℕ := 2
  let teacher_and_females : ℕ := 1  -- Treat teacher and females as one unit
  let remaining_elements : ℕ := teacher_and_females + male_students
  (female_students.factorial) * (remaining_elements.factorial)

theorem group_photo_arrangements_eq_12 : group_photo_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_group_photo_arrangements_eq_12_l3409_340943


namespace NUMINAMATH_CALUDE_radical_combination_l3409_340995

theorem radical_combination (x : ℝ) : (2 + x = 5 - 2*x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_radical_combination_l3409_340995


namespace NUMINAMATH_CALUDE_xyz_sum_l3409_340961

theorem xyz_sum (x y z : ℕ+) 
  (eq1 : x * y + z = 47)
  (eq2 : y * z + x = 47)
  (eq3 : z * x + y = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l3409_340961


namespace NUMINAMATH_CALUDE_weight_within_range_l3409_340992

/-- The labeled weight of the flour in kilograms -/
def labeled_weight : ℝ := 25

/-- The tolerance range for the flour weight in kilograms -/
def tolerance : ℝ := 0.2

/-- The actual weight of the flour in kilograms -/
def actual_weight : ℝ := 25.1

/-- Theorem stating that the actual weight is within the acceptable range -/
theorem weight_within_range : 
  labeled_weight - tolerance ≤ actual_weight ∧ actual_weight ≤ labeled_weight + tolerance :=
by sorry

end NUMINAMATH_CALUDE_weight_within_range_l3409_340992


namespace NUMINAMATH_CALUDE_poster_spacing_proof_l3409_340957

/-- Calculates the equal distance between posters and from the ends of the wall -/
def equal_distance (wall_width : ℕ) (num_posters : ℕ) (poster_width : ℕ) : ℕ :=
  (wall_width - num_posters * poster_width) / (num_posters + 1)

/-- Theorem stating that the equal distance is 20 cm given the problem conditions -/
theorem poster_spacing_proof :
  equal_distance 320 6 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_poster_spacing_proof_l3409_340957


namespace NUMINAMATH_CALUDE_apples_per_pie_l3409_340962

theorem apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (num_pies : ℕ) :
  initial_apples = 62 →
  handed_out = 8 →
  num_pies = 6 →
  (initial_apples - handed_out) / num_pies = 9 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_pie_l3409_340962


namespace NUMINAMATH_CALUDE_right_triangle_from_equations_l3409_340976

theorem right_triangle_from_equations (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  x^2 + 2*a*x + b^2 = 0 →
  x^2 + 2*c*x - b^2 = 0 →
  a^2 = b^2 + c^2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_from_equations_l3409_340976


namespace NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l3409_340972

theorem lcm_of_numbers_in_ratio (a b : ℕ) (h_ratio : a * 5 = b * 4) (h_smaller : a = 36) : 
  Nat.lcm a b = 1620 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_in_ratio_l3409_340972


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3409_340979

theorem discount_percentage_proof (coat_price pants_price : ℝ)
  (coat_discount pants_discount : ℝ) :
  coat_price = 100 →
  pants_price = 50 →
  coat_discount = 0.3 →
  pants_discount = 0.4 →
  let total_original := coat_price + pants_price
  let total_savings := coat_price * coat_discount + pants_price * pants_discount
  let savings_percentage := total_savings / total_original * 100
  savings_percentage = 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3409_340979


namespace NUMINAMATH_CALUDE_park_tree_count_l3409_340948

/-- Calculates the final number of trees in a park after cutting --/
def final_tree_count (initial_oak initial_maple oak_cut maple_cut : ℕ) : ℕ × ℕ × ℕ :=
  let final_oak := initial_oak - oak_cut
  let final_maple := initial_maple - maple_cut
  let total := final_oak + final_maple
  (final_oak, final_maple, total)

/-- Theorem stating the final tree count after cutting in the park --/
theorem park_tree_count :
  final_tree_count 57 43 13 8 = (44, 35, 79) := by
  sorry

end NUMINAMATH_CALUDE_park_tree_count_l3409_340948


namespace NUMINAMATH_CALUDE_james_writes_to_fourteen_people_l3409_340951

/-- Represents James' writing habits and calculates the number of people he writes to daily --/
def james_writing (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (hours_per_week : ℕ) : ℕ :=
  (pages_per_hour * hours_per_week) / pages_per_person_per_day

/-- Theorem stating that James writes to 14 people daily --/
theorem james_writes_to_fourteen_people :
  james_writing 10 5 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_james_writes_to_fourteen_people_l3409_340951


namespace NUMINAMATH_CALUDE_bridesmaids_count_l3409_340982

/-- Represents the makeup requirements for bridesmaids --/
structure MakeupRequirements where
  lipGlossPerTube : ℕ
  mascaraPerTube : ℕ
  lipGlossTubs : ℕ
  tubesPerLipGlossTub : ℕ
  mascaraTubs : ℕ
  tubesPerMascaraTub : ℕ

/-- Represents the makeup styles chosen by bridesmaids --/
inductive MakeupStyle
  | Glam
  | Natural

/-- Calculates the total number of bridesmaids given the makeup requirements --/
def totalBridesmaids (req : MakeupRequirements) : ℕ :=
  let totalLipGloss := req.lipGlossTubs * req.tubesPerLipGlossTub * req.lipGlossPerTube
  let totalMascara := req.mascaraTubs * req.tubesPerMascaraTub * req.mascaraPerTube
  let glamBridesmaids := totalLipGloss / 3  -- Each glam bridesmaid needs 2 lip gloss + 1 natural
  min glamBridesmaids (totalMascara / 2)  -- Each bridesmaid needs at least 1 mascara

/-- Proves that given the specific makeup requirements, there are 24 bridesmaids --/
theorem bridesmaids_count (req : MakeupRequirements) 
    (h1 : req.lipGlossPerTube = 3)
    (h2 : req.mascaraPerTube = 5)
    (h3 : req.lipGlossTubs = 6)
    (h4 : req.tubesPerLipGlossTub = 2)
    (h5 : req.mascaraTubs = 4)
    (h6 : req.tubesPerMascaraTub = 3) :
    totalBridesmaids req = 24 := by
  sorry

#eval totalBridesmaids { 
  lipGlossPerTube := 3, 
  mascaraPerTube := 5, 
  lipGlossTubs := 6, 
  tubesPerLipGlossTub := 2, 
  mascaraTubs := 4, 
  tubesPerMascaraTub := 3 
}

end NUMINAMATH_CALUDE_bridesmaids_count_l3409_340982


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3409_340912

theorem right_triangle_ratio (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (a^2 + b^2) / (a^2 + b^2 + c^2) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3409_340912


namespace NUMINAMATH_CALUDE_non_indian_percentage_approx_l3409_340924

/-- Represents the number of attendees in a category and the percentage of Indians in that category -/
structure AttendeeCategory where
  total : ℕ
  indianPercentage : ℚ

/-- Calculates the number of non-Indian attendees in a category -/
def nonIndianCount (category : AttendeeCategory) : ℚ :=
  category.total * (1 - category.indianPercentage)

/-- Data for the climate conference -/
def conferenceData : List AttendeeCategory := [
  ⟨1200, 25/100⟩,  -- Male participants
  ⟨800, 40/100⟩,   -- Male volunteers
  ⟨1000, 35/100⟩,  -- Female participants
  ⟨500, 15/100⟩,   -- Female volunteers
  ⟨1800, 10/100⟩,  -- Children
  ⟨500, 45/100⟩,   -- Male scientists
  ⟨250, 30/100⟩,   -- Female scientists
  ⟨350, 55/100⟩,   -- Male government officials
  ⟨150, 50/100⟩    -- Female government officials
]

/-- Total number of attendees -/
def totalAttendees : ℕ := 6550

/-- Theorem stating that the percentage of non-Indian attendees is approximately 72.61% -/
theorem non_indian_percentage_approx :
  abs ((List.sum (List.map nonIndianCount conferenceData) / totalAttendees) - 72.61/100) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_non_indian_percentage_approx_l3409_340924


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3409_340936

theorem trigonometric_identity (h1 : Real.tan (10 * π / 180) * Real.tan (20 * π / 180) + 
                                     Real.tan (20 * π / 180) * Real.tan (60 * π / 180) + 
                                     Real.tan (60 * π / 180) * Real.tan (10 * π / 180) = 1)
                               (h2 : Real.tan (5 * π / 180) * Real.tan (10 * π / 180) + 
                                     Real.tan (10 * π / 180) * Real.tan (75 * π / 180) + 
                                     Real.tan (75 * π / 180) * Real.tan (5 * π / 180) = 1) :
  Real.tan (8 * π / 180) * Real.tan (12 * π / 180) + 
  Real.tan (12 * π / 180) * Real.tan (70 * π / 180) + 
  Real.tan (70 * π / 180) * Real.tan (8 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3409_340936


namespace NUMINAMATH_CALUDE_organizationalStructureIsCorrect_l3409_340938

-- Define the types of diagrams
inductive Diagram
  | Flowchart
  | ProcessFlow
  | KnowledgeStructure
  | OrganizationalStructure

-- Define the properties a diagram should have
structure DiagramProperties where
  reflectsRelationships : Bool
  showsVerticalHorizontal : Bool
  reflectsOrganizationalStructure : Bool
  interpretsOrganizationalFunctions : Bool

-- Define a function to check if a diagram has the required properties
def hasRequiredProperties (d : Diagram) : DiagramProperties :=
  match d with
  | Diagram.OrganizationalStructure => {
      reflectsRelationships := true,
      showsVerticalHorizontal := true,
      reflectsOrganizationalStructure := true,
      interpretsOrganizationalFunctions := true
    }
  | _ => {
      reflectsRelationships := false,
      showsVerticalHorizontal := false,
      reflectsOrganizationalStructure := false,
      interpretsOrganizationalFunctions := false
    }

-- Theorem: The Organizational Structure Diagram is the correct choice for describing factory composition
theorem organizationalStructureIsCorrect :
  ∀ (d : Diagram),
    (hasRequiredProperties d).reflectsRelationships ∧
    (hasRequiredProperties d).showsVerticalHorizontal ∧
    (hasRequiredProperties d).reflectsOrganizationalStructure ∧
    (hasRequiredProperties d).interpretsOrganizationalFunctions
    →
    d = Diagram.OrganizationalStructure :=
  sorry

end NUMINAMATH_CALUDE_organizationalStructureIsCorrect_l3409_340938


namespace NUMINAMATH_CALUDE_train_speed_l3409_340998

/-- Given a train crossing a bridge, calculate its speed in km/hr -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 150)
  (h2 : bridge_length = 225)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l3409_340998


namespace NUMINAMATH_CALUDE_cone_slant_height_l3409_340942

/-- 
Given a cone whose lateral surface unfolds to a semicircle and whose base radius is 1,
prove that its slant height is 2.
-/
theorem cone_slant_height (r : ℝ) (l : ℝ) : 
  r = 1 → -- radius of the base is 1
  2 * π * r = π * l → -- lateral surface unfolds to a semicircle
  l = 2 := by sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3409_340942


namespace NUMINAMATH_CALUDE_function_range_l3409_340993

theorem function_range (θ : ℝ) : 
  ∀ x : ℝ, 2 - Real.sqrt 3 ≤ (x^2 + 2*x*Real.sin θ + 2) / (x^2 + 2*x*Real.cos θ + 2) 
         ∧ (x^2 + 2*x*Real.sin θ + 2) / (x^2 + 2*x*Real.cos θ + 2) ≤ 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l3409_340993


namespace NUMINAMATH_CALUDE_deleted_pictures_count_l3409_340959

def zoo_pictures : ℕ := 15
def museum_pictures : ℕ := 18
def remaining_pictures : ℕ := 2

theorem deleted_pictures_count :
  zoo_pictures + museum_pictures - remaining_pictures = 31 := by
  sorry

end NUMINAMATH_CALUDE_deleted_pictures_count_l3409_340959


namespace NUMINAMATH_CALUDE_only_14_satisfies_l3409_340933

def is_multiple_of_three (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def satisfies_conditions (n : ℕ) : Prop :=
  ¬(is_multiple_of_three n) ∧
  ¬(is_perfect_square n) ∧
  is_prime (sum_of_digits n)

theorem only_14_satisfies :
  satisfies_conditions 14 ∧
  ¬(satisfies_conditions 12) ∧
  ¬(satisfies_conditions 16) ∧
  ¬(satisfies_conditions 21) ∧
  ¬(satisfies_conditions 26) :=
sorry

end NUMINAMATH_CALUDE_only_14_satisfies_l3409_340933


namespace NUMINAMATH_CALUDE_sqrt_D_rationality_l3409_340990

/-- Given integers a and b where b = a + 2, and c = ab, 
    D is defined as a² + b² + c². This theorem states that 
    √D can be either rational or irrational. -/
theorem sqrt_D_rationality (a : ℤ) : 
  ∃ (D : ℚ), D = (a^2 : ℚ) + ((a+2)^2 : ℚ) + ((a*(a+2))^2 : ℚ) ∧ 
  (∃ (x : ℚ), x^2 = D) ∨ (∀ (x : ℚ), x^2 ≠ D) :=
sorry

end NUMINAMATH_CALUDE_sqrt_D_rationality_l3409_340990


namespace NUMINAMATH_CALUDE_equation_solution_l3409_340921

theorem equation_solution : ∃! x : ℝ, (1 / (x - 1) = 3 / (x - 3)) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3409_340921


namespace NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l3409_340967

theorem max_value_problem (x : ℝ) (h : x < 1) :
  ∃ y : ℝ, y = (4 * x^2 - 3 * x) / (x - 1) ∧ 
  ∀ z : ℝ, z = (4 * x^2 - 3 * x) / (x - 1) → z ≤ y ∧ y = 1 :=
sorry

theorem min_value_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ y : ℝ, y = 4 / (a + 1) + 1 / b ∧
  ∀ z : ℝ, z = 4 / (a + 1) + 1 / b → y ≤ z ∧ y = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l3409_340967


namespace NUMINAMATH_CALUDE_bowling_tournament_orderings_l3409_340909

/-- Represents a tournament with a fixed number of participants and rounds --/
structure Tournament where
  participants : Nat
  rounds : Nat

/-- Calculates the number of possible orderings in a tournament --/
def possibleOrderings (t : Tournament) : Nat :=
  2 ^ t.rounds

/-- The specific tournament described in the problem --/
def bowlingTournament : Tournament :=
  { participants := 6, rounds := 5 }

/-- Theorem stating that the number of possible orderings in the bowling tournament is 32 --/
theorem bowling_tournament_orderings :
  possibleOrderings bowlingTournament = 32 := by
  sorry

#eval possibleOrderings bowlingTournament

end NUMINAMATH_CALUDE_bowling_tournament_orderings_l3409_340909


namespace NUMINAMATH_CALUDE_fraction_subtraction_and_multiplication_l3409_340930

theorem fraction_subtraction_and_multiplication :
  (1 / 2 : ℚ) * ((5 / 6 : ℚ) - (1 / 9 : ℚ)) = 13 / 36 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_and_multiplication_l3409_340930


namespace NUMINAMATH_CALUDE_points_per_game_l3409_340991

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 91 → 
  num_games = 13 → 
  total_points = num_games * points_per_game → 
  points_per_game = 7 := by
sorry

end NUMINAMATH_CALUDE_points_per_game_l3409_340991


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3409_340946

/-- Represents the rate of simple interest per annum -/
def rate : ℚ := 1 / 24

/-- The time period in years -/
def time : ℕ := 12

/-- The ratio of final amount to initial amount -/
def growth_ratio : ℚ := 9 / 6

theorem simple_interest_rate :
  (1 + rate * time) = growth_ratio := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3409_340946


namespace NUMINAMATH_CALUDE_balcony_orchestra_difference_l3409_340923

/-- Represents the number of tickets sold for a theater performance. -/
structure TicketSales where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold. -/
def TicketSales.total (ts : TicketSales) : ℕ :=
  ts.orchestra + ts.balcony

/-- Calculates the total revenue from ticket sales. -/
def TicketSales.revenue (ts : TicketSales) : ℕ :=
  12 * ts.orchestra + 8 * ts.balcony

/-- Theorem stating the difference between balcony and orchestra ticket sales. -/
theorem balcony_orchestra_difference (ts : TicketSales) 
  (h1 : ts.total = 350)
  (h2 : ts.revenue = 3320) :
  ts.balcony - ts.orchestra = 90 := by
  sorry


end NUMINAMATH_CALUDE_balcony_orchestra_difference_l3409_340923


namespace NUMINAMATH_CALUDE_xyz_sum_max_min_l3409_340937

theorem xyz_sum_max_min (x y z : ℝ) (h : 4 * (x + y + z) = x^2 + y^2 + z^2) :
  let f := fun (a b c : ℝ) => a * b + a * c + b * c
  ∃ (M m : ℝ), (∀ (a b c : ℝ), 4 * (a + b + c) = a^2 + b^2 + c^2 → f a b c ≤ M) ∧
               (∀ (a b c : ℝ), 4 * (a + b + c) = a^2 + b^2 + c^2 → m ≤ f a b c) ∧
               M + 10 * m = 28 :=
by sorry

end NUMINAMATH_CALUDE_xyz_sum_max_min_l3409_340937


namespace NUMINAMATH_CALUDE_scarf_sales_with_new_price_and_tax_l3409_340954

/-- Represents the relationship between number of scarves sold and their price -/
def scarfRelation (k : ℝ) (p c : ℝ) : Prop := p * c = k

theorem scarf_sales_with_new_price_and_tax 
  (k : ℝ) 
  (initial_price initial_quantity new_price tax_rate : ℝ) : 
  scarfRelation k initial_quantity initial_price →
  initial_price = 10 →
  initial_quantity = 30 →
  new_price = 15 →
  tax_rate = 0.1 →
  ∃ (new_quantity : ℕ), 
    scarfRelation k (new_quantity : ℝ) (new_price * (1 + tax_rate)) ∧ 
    new_quantity = 18 := by
  sorry

end NUMINAMATH_CALUDE_scarf_sales_with_new_price_and_tax_l3409_340954


namespace NUMINAMATH_CALUDE_smallest_number_of_eggs_l3409_340996

theorem smallest_number_of_eggs : ∀ n : ℕ,
  (∃ c : ℕ, n = 12 * c - 3) →  -- Eggs are in containers of 12, with 3 containers having 11 eggs
  n > 200 →                   -- More than 200 eggs
  n ≥ 201                     -- The smallest possible number is at least 201
:= by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_eggs_l3409_340996


namespace NUMINAMATH_CALUDE_least_satisfying_number_l3409_340902

def is_multiple_of_36 (n : ℕ) : Prop := ∃ k : ℕ, n = 36 * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  is_multiple_of_36 n ∧ is_multiple_of_36 (digit_product n)

theorem least_satisfying_number :
  satisfies_condition 1296 ∧ 
  ∀ m : ℕ, m > 0 ∧ m < 1296 → ¬(satisfies_condition m) :=
by sorry

end NUMINAMATH_CALUDE_least_satisfying_number_l3409_340902


namespace NUMINAMATH_CALUDE_multiples_of_5_ending_in_0_less_than_200_l3409_340966

def count_multiples_of_5_ending_in_0 (upper_bound : ℕ) : ℕ :=
  (upper_bound - 1) / 10

theorem multiples_of_5_ending_in_0_less_than_200 :
  count_multiples_of_5_ending_in_0 200 = 19 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_5_ending_in_0_less_than_200_l3409_340966


namespace NUMINAMATH_CALUDE_correct_operation_result_l3409_340960

theorem correct_operation_result (x : ℝ) : 
  (x / 8 - 12 = 32) → (x * 8 + 12 = 2828) := by sorry

end NUMINAMATH_CALUDE_correct_operation_result_l3409_340960


namespace NUMINAMATH_CALUDE_parabola_coefficient_sum_l3409_340969

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℚ × ℚ := sorry

/-- Check if a point lies on the parabola -/
def containsPoint (p : Parabola) (x y : ℚ) : Prop := sorry

/-- Check if the parabola has a vertical axis of symmetry -/
def hasVerticalAxisOfSymmetry (p : Parabola) : Prop := sorry

theorem parabola_coefficient_sum 
  (p : Parabola) 
  (h1 : vertex p = (5, 3))
  (h2 : hasVerticalAxisOfSymmetry p)
  (h3 : containsPoint p 2 0) :
  p.a + p.b + p.c = -7/3 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_sum_l3409_340969


namespace NUMINAMATH_CALUDE_inequality_holds_l3409_340975

theorem inequality_holds (f : ℝ → ℝ) (a b x : ℝ) 
  (h_f : ∀ x, f x = 4 * x - 1)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_x : |x - 2*b| < b)
  (h_ab : a ≤ 4*b) : 
  (x + a)^2 + |f x - 3*b| < a^2 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_l3409_340975
