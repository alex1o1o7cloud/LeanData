import Mathlib

namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l3746_374650

/-- Represents a normal distribution with mean μ and standard deviation σ -/
structure NormalDistribution where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The cumulative distribution function (CDF) for a normal distribution -/
noncomputable def normalCDF (nd : NormalDistribution) (x : ℝ) : ℝ :=
  sorry

theorem normal_distribution_symmetry 
  (nd : NormalDistribution) 
  (h_mean : nd.μ = 85) 
  (h_cdf : normalCDF nd 122 = 0.96) :
  normalCDF nd 48 = 0.04 :=
sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l3746_374650


namespace NUMINAMATH_CALUDE_mary_has_ten_more_than_marco_l3746_374655

/-- Calculates the difference in money between Mary and Marco after transactions. -/
def moneyDifference (marco_initial : ℕ) (mary_initial : ℕ) (mary_spent : ℕ) : ℕ :=
  let marco_gives := marco_initial / 2
  let marco_final := marco_initial - marco_gives
  let mary_final := mary_initial + marco_gives - mary_spent
  mary_final - marco_final

/-- Proves that Mary has $10 more than Marco after the described transactions. -/
theorem mary_has_ten_more_than_marco :
  moneyDifference 24 15 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_ten_more_than_marco_l3746_374655


namespace NUMINAMATH_CALUDE_complex_magnitude_l3746_374617

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3746_374617


namespace NUMINAMATH_CALUDE_f_has_unique_zero_and_g_max_a_l3746_374666

noncomputable def f (x : ℝ) := (x - 2) * Real.log x + 2 * x - 3

noncomputable def g (a : ℝ) (x : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem f_has_unique_zero_and_g_max_a :
  (∃! x : ℝ, x ≥ 1 ∧ f x = 0) ∧
  (∀ a : ℝ, a > 6 → ∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ g a x₂ < g a x₁) ∧
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → g 6 x₁ ≤ g 6 x₂) :=
by sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_and_g_max_a_l3746_374666


namespace NUMINAMATH_CALUDE_gcd_of_product_form_l3746_374670

def product_form (a b c d : ℤ) : ℤ :=
  (b - a) * (c - b) * (d - c) * (d - a) * (c - a) * (d - b)

theorem gcd_of_product_form :
  ∃ (g : ℤ), g > 0 ∧ 
  (∀ (a b c d : ℤ), g ∣ product_form a b c d) ∧
  (∀ (h : ℤ), h > 0 → (∀ (a b c d : ℤ), h ∣ product_form a b c d) → h ∣ g) ∧
  g = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_product_form_l3746_374670


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3746_374600

/-- Two cars meeting on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (h1 : highway_length = 105)
    (h2 : speed1 = 15) (h3 : speed2 = 20) : 
  (highway_length / (speed1 + speed2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l3746_374600


namespace NUMINAMATH_CALUDE_sin_150_degrees_l3746_374626

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l3746_374626


namespace NUMINAMATH_CALUDE_range_of_m_l3746_374660

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 1 > 0) →
  ((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0) = False) →
  ((m + 1 ≤ 0) ∨ (∀ x : ℝ, x^2 + m*x + 1 > 0) = True) →
  (m ≤ -2 ∨ (-1 < m ∧ m < 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3746_374660


namespace NUMINAMATH_CALUDE_preimage_of_one_two_l3746_374621

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, 2 * p.1 - 3 * p.2)

theorem preimage_of_one_two :
  f (1, 0) = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_one_two_l3746_374621


namespace NUMINAMATH_CALUDE_benny_market_money_l3746_374602

/-- The amount of money Benny took to the market --/
def money_taken : ℕ → ℕ → ℕ → ℕ
  | num_kids, apples_per_kid, cost_per_apple =>
    num_kids * apples_per_kid * cost_per_apple

theorem benny_market_money :
  money_taken 18 5 4 = 360 := by
  sorry

end NUMINAMATH_CALUDE_benny_market_money_l3746_374602


namespace NUMINAMATH_CALUDE_arc_RS_range_l3746_374636

/-- An isosceles triangle with a rolling circle -/
structure RollingCircleTriangle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The altitude of the isosceles triangle -/
  altitude : ℝ
  /-- The radius of the rolling circle -/
  radius : ℝ
  /-- The position of the tangent point P along the base (0 ≤ p ≤ base) -/
  p : ℝ
  /-- The triangle is isosceles -/
  isosceles : altitude = base / 2
  /-- The altitude is twice the radius -/
  altitude_radius : altitude = 2 * radius
  /-- The tangent point is on the base -/
  p_on_base : 0 ≤ p ∧ p ≤ base

/-- The arc RS of the rolling circle -/
def arc_RS (t : RollingCircleTriangle) : ℝ := sorry

/-- Theorem: The arc RS varies from 90° to 180° -/
theorem arc_RS_range (t : RollingCircleTriangle) : 
  90 ≤ arc_RS t ∧ arc_RS t ≤ 180 := by sorry

end NUMINAMATH_CALUDE_arc_RS_range_l3746_374636


namespace NUMINAMATH_CALUDE_paperclip_excess_day_l3746_374662

def paperclip_sequence (k : ℕ) : ℕ := 4 * 3^k

theorem paperclip_excess_day :
  (∀ j : ℕ, j < 6 → paperclip_sequence j ≤ 2000) ∧
  paperclip_sequence 6 > 2000 :=
sorry

end NUMINAMATH_CALUDE_paperclip_excess_day_l3746_374662


namespace NUMINAMATH_CALUDE_x_value_proof_l3746_374647

theorem x_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 8 * x^2 + 16 * x * y = 2 * x^3 + 4 * x^2 * y) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3746_374647


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l3746_374657

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + x + 3 = 7 → 3*x^2 + 3*x + 7 = 19 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l3746_374657


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3746_374633

theorem partial_fraction_decomposition_product (N₁ N₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (35 * x - 29) / (x^2 - 3*x + 2) = N₁ / (x - 1) + N₂ / (x - 2)) →
  N₁ * N₂ = -246 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3746_374633


namespace NUMINAMATH_CALUDE_prob_at_least_one_even_is_five_ninths_l3746_374619

/-- A set of cards labeled 1, 2, and 3 -/
def cards : Finset ℕ := {1, 2, 3}

/-- The event of drawing an even number -/
def is_even (n : ℕ) : Bool := n % 2 = 0

/-- The sample space of two draws with replacement -/
def sample_space : Finset (ℕ × ℕ) :=
  (cards.product cards)

/-- The favorable outcomes (at least one even number) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p => is_even p.1 ∨ is_even p.2)

/-- The probability of drawing at least one even number in two draws -/
def prob_at_least_one_even : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem prob_at_least_one_even_is_five_ninths :
  prob_at_least_one_even = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_even_is_five_ninths_l3746_374619


namespace NUMINAMATH_CALUDE_min_distance_is_8_l3746_374673

-- Define the condition function
def condition (a b c d : ℝ) : Prop :=
  (a - 2 * Real.exp a) / b = (1 - c) / (d - 1) ∧ (a - 2 * Real.exp a) / b = 1

-- Define the distance function
def distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

-- Theorem statement
theorem min_distance_is_8 :
  ∀ a b c d : ℝ, condition a b c d → 
  ∀ x y z w : ℝ, condition x y z w →
  distance a b c d ≥ 8 ∧ (∃ a₀ b₀ c₀ d₀ : ℝ, condition a₀ b₀ c₀ d₀ ∧ distance a₀ b₀ c₀ d₀ = 8) :=
sorry

end NUMINAMATH_CALUDE_min_distance_is_8_l3746_374673


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l3746_374682

theorem similar_triangles_leg_length (a b c d : ℝ) :
  a > 0 → b > 0 → c > 0 → d > 0 →
  a = 12 → b = 9 → c = 7.5 →
  a / c = b / d →
  d = 5.625 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l3746_374682


namespace NUMINAMATH_CALUDE_solve_for_y_l3746_374606

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - 3 * y = 9) (h2 : x + y = 8) : y = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3746_374606


namespace NUMINAMATH_CALUDE_sin_cos_shift_l3746_374601

theorem sin_cos_shift (x : ℝ) : Real.cos (2 * (x - Real.pi / 8) - Real.pi / 4) = Real.sin (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_shift_l3746_374601


namespace NUMINAMATH_CALUDE_population_decreases_below_threshold_l3746_374651

/-- The annual decrease rate of the population -/
def decrease_rate : ℝ := 0.5

/-- The threshold percentage of the initial population -/
def threshold : ℝ := 0.05

/-- The number of years it takes for the population to decrease below the threshold -/
def years_to_threshold : ℕ := 5

/-- The function that calculates the population after a given number of years -/
def population_after_years (initial_population : ℝ) (years : ℕ) : ℝ :=
  initial_population * (decrease_rate ^ years)

theorem population_decreases_below_threshold :
  ∀ initial_population : ℝ,
  initial_population > 0 →
  population_after_years initial_population years_to_threshold < threshold * initial_population ∧
  population_after_years initial_population (years_to_threshold - 1) ≥ threshold * initial_population :=
by sorry

end NUMINAMATH_CALUDE_population_decreases_below_threshold_l3746_374651


namespace NUMINAMATH_CALUDE_digital_root_of_1999_factorial_l3746_374684

/-- The factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The digital root function -/
def digitalRoot (n : ℕ) : ℕ :=
  if n = 0 then 0 else 1 + (n - 1) % 9

/-- Theorem: The digital root of 1999! is 9 -/
theorem digital_root_of_1999_factorial :
  digitalRoot (factorial 1999) = 9 := by
  sorry

end NUMINAMATH_CALUDE_digital_root_of_1999_factorial_l3746_374684


namespace NUMINAMATH_CALUDE_path_count_theorem_l3746_374686

def grid_path (right up : ℕ) : ℕ := Nat.choose (right + up) up

theorem path_count_theorem :
  let right : ℕ := 6
  let up : ℕ := 4
  let total_path_length : ℕ := right + up
  grid_path right up = 210 := by
  sorry

end NUMINAMATH_CALUDE_path_count_theorem_l3746_374686


namespace NUMINAMATH_CALUDE_sqrt_identity_in_range_l3746_374637

theorem sqrt_identity_in_range (θ : Real) (h : θ ∈ Set.Ioo (7 * Real.pi / 4) (2 * Real.pi)) :
  Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) = Real.cos θ - Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_in_range_l3746_374637


namespace NUMINAMATH_CALUDE_cos_75_degrees_l3746_374681

theorem cos_75_degrees : Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l3746_374681


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l3746_374699

/-- Given a circle with polar equation ρ = 4sin(θ) and a line with parametric equation x = √3t, y = t,
    the distance from the center of the circle to the line is √3. -/
theorem distance_circle_center_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4*y}
  let line := {(x, y) : ℝ × ℝ | ∃ t : ℝ, x = Real.sqrt 3 * t ∧ y = t}
  let circle_center := (0, 2)
  ∃ p ∈ line, Real.sqrt ((circle_center.1 - p.1)^2 + (circle_center.2 - p.2)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l3746_374699


namespace NUMINAMATH_CALUDE_complement_of_A_l3746_374696

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := Set.Ioc (-2) 1

-- State the theorem
theorem complement_of_A : 
  Set.compl A = Set.Iic (-2) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3746_374696


namespace NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l3746_374635

/-- The line equation ax - y + 2 + a = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 + a = 0

/-- The line equation 4x + y + 3 = 0 -/
def line_l1 (x y : ℝ) : Prop := 4 * x + y + 3 = 0

/-- The line equation 3x - 5y - 5 = 0 -/
def line_l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 5 = 0

/-- The fixed point P -/
def point_P : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + 1 = 0 -/
def line_m (x y : ℝ) : Prop := 3 * x + y + 1 = 0

theorem fixed_point_and_bisecting_line :
  (∀ a : ℝ, line_l a (point_P.1) (point_P.2)) ∧
  (∀ x y : ℝ, line_m x y ↔ 
    (∃ t : ℝ, line_l1 (point_P.1 - t) (point_P.2 - t) ∧
              line_l2 (point_P.1 + t) (point_P.2 + t))) :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_and_bisecting_line_l3746_374635


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3746_374641

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p (1, 0) = |p.1 + 1|}

-- Define the property of line l intersecting C at M and N
def intersects_at_MN (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ) : Prop :=
  M ∈ C ∧ N ∈ C ∧ M ∈ l ∧ N ∈ l ∧ M ≠ N ∧ M ≠ (0, 0) ∧ N ≠ (0, 0)

-- Define the perpendicularity of OM and ON
def OM_perp_ON (M N : ℝ × ℝ) : Prop :=
  M.1 * N.1 + M.2 * N.2 = 0

-- Theorem statement
theorem line_passes_through_fixed_point :
  ∀ l : Set (ℝ × ℝ), ∀ M N : ℝ × ℝ,
  intersects_at_MN l M N → OM_perp_ON M N →
  (4, 0) ∈ l :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3746_374641


namespace NUMINAMATH_CALUDE_sequence_may_or_may_not_be_arithmetic_l3746_374693

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def is_arithmetic (s : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- The first five terms of the sequence are 1, 2, 3, 4, 5. -/
def first_five_terms (s : Sequence) : Prop :=
  s 0 = 1 ∧ s 1 = 2 ∧ s 2 = 3 ∧ s 3 = 4 ∧ s 4 = 5

theorem sequence_may_or_may_not_be_arithmetic :
  ∃ s₁ s₂ : Sequence, first_five_terms s₁ ∧ first_five_terms s₂ ∧
    is_arithmetic s₁ ∧ ¬is_arithmetic s₂ := by
  sorry

end NUMINAMATH_CALUDE_sequence_may_or_may_not_be_arithmetic_l3746_374693


namespace NUMINAMATH_CALUDE_expenditure_recording_l3746_374628

-- Define a type for financial transactions
inductive Transaction
| Income (amount : ℤ)
| Expenditure (amount : ℤ)

-- Define a function to record transactions
def record_transaction (t : Transaction) : ℤ :=
  match t with
  | Transaction.Income a => a
  | Transaction.Expenditure a => -a

-- Theorem statement
theorem expenditure_recording (income_amount expenditure_amount : ℤ) 
  (h1 : income_amount > 0) (h2 : expenditure_amount > 0) :
  record_transaction (Transaction.Income income_amount) = income_amount ∧
  record_transaction (Transaction.Expenditure expenditure_amount) = -expenditure_amount :=
by sorry

end NUMINAMATH_CALUDE_expenditure_recording_l3746_374628


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3746_374622

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x^2 - 3*x + 2 < 0 → x < 2) ∧
  ¬(∀ x : ℝ, x < 2 → x^2 - 3*x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3746_374622


namespace NUMINAMATH_CALUDE_percent_swap_l3746_374668

theorem percent_swap (x : ℝ) (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 := by
  sorry

end NUMINAMATH_CALUDE_percent_swap_l3746_374668


namespace NUMINAMATH_CALUDE_unique_positive_number_l3746_374646

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x - 4 = 21 * (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l3746_374646


namespace NUMINAMATH_CALUDE_pants_cost_is_6_l3746_374683

/-- The cost of one pair of pants -/
def pants_cost : ℚ := 6

/-- The cost of one shirt -/
def shirt_cost : ℚ := 10

/-- Theorem stating the cost of one pair of pants is $6 -/
theorem pants_cost_is_6 :
  (2 * pants_cost + 5 * shirt_cost = 62) →
  (2 * shirt_cost = 20) →
  pants_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_is_6_l3746_374683


namespace NUMINAMATH_CALUDE_used_cd_cost_correct_l3746_374664

/-- The cost of Lakota's purchase -/
def lakota_cost : ℝ := 127.92

/-- The cost of Mackenzie's purchase -/
def mackenzie_cost : ℝ := 133.89

/-- The number of new CDs Lakota bought -/
def lakota_new : ℕ := 6

/-- The number of used CDs Lakota bought -/
def lakota_used : ℕ := 2

/-- The number of new CDs Mackenzie bought -/
def mackenzie_new : ℕ := 3

/-- The number of used CDs Mackenzie bought -/
def mackenzie_used : ℕ := 8

/-- The cost of a single used CD -/
def used_cd_cost : ℝ := 9.99

theorem used_cd_cost_correct :
  ∃ (new_cd_cost : ℝ),
    lakota_new * new_cd_cost + lakota_used * used_cd_cost = lakota_cost ∧
    mackenzie_new * new_cd_cost + mackenzie_used * used_cd_cost = mackenzie_cost :=
by sorry

end NUMINAMATH_CALUDE_used_cd_cost_correct_l3746_374664


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l3746_374685

/-- The probability of drawing two white balls consecutively without replacement -/
theorem two_white_balls_probability 
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (red_balls : ℕ) 
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 5)
  (h3 : red_balls = 3) : 
  (white_balls : ℚ) / total_balls * ((white_balls - 1) : ℚ) / (total_balls - 1) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l3746_374685


namespace NUMINAMATH_CALUDE_total_students_l3746_374620

/-- Represents the age groups in the school -/
inductive AgeGroup
  | Below8
  | Exactly8
  | Between9And10
  | Above10

/-- Represents the school with its student distribution -/
structure School where
  totalStudents : ℕ
  ageDistribution : AgeGroup → ℚ
  exactly8Count : ℕ

/-- The conditions of the problem -/
def schoolConditions (s : School) : Prop :=
  s.ageDistribution AgeGroup.Below8 = 1/5 ∧
  s.ageDistribution AgeGroup.Exactly8 = 1/4 ∧
  s.ageDistribution AgeGroup.Between9And10 = 7/20 ∧
  s.ageDistribution AgeGroup.Above10 = 1/5 ∧
  s.exactly8Count = 15

/-- The theorem to prove -/
theorem total_students (s : School) (h : schoolConditions s) : s.totalStudents = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l3746_374620


namespace NUMINAMATH_CALUDE_alberts_remaining_laps_l3746_374615

/-- Calculates the remaining laps for Albert's run -/
theorem alberts_remaining_laps 
  (total_distance : ℕ) 
  (track_length : ℕ) 
  (laps_run : ℕ) 
  (h1 : total_distance = 99) 
  (h2 : track_length = 9) 
  (h3 : laps_run = 6) : 
  total_distance / track_length - laps_run = 5 := by
  sorry

#check alberts_remaining_laps

end NUMINAMATH_CALUDE_alberts_remaining_laps_l3746_374615


namespace NUMINAMATH_CALUDE_sum_of_factors_of_30_l3746_374687

def factors (n : ℕ) : Finset ℕ := Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem sum_of_factors_of_30 : (factors 30).sum id = 72 := by sorry

end NUMINAMATH_CALUDE_sum_of_factors_of_30_l3746_374687


namespace NUMINAMATH_CALUDE_front_axle_wheels_l3746_374694

/-- The toll formula for a truck crossing a bridge -/
def toll (x : ℕ) : ℚ :=
  3.5 + 0.5 * (x - 2)

/-- The number of axles for an 18-wheel truck with f wheels on the front axle -/
def num_axles (f : ℕ) : ℕ :=
  1 + (18 - f) / 4

theorem front_axle_wheels :
  ∃ (f : ℕ), f > 0 ∧ f < 18 ∧ 
  toll (num_axles f) = 5 ∧
  f = 2 := by
  sorry

end NUMINAMATH_CALUDE_front_axle_wheels_l3746_374694


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l3746_374609

/-- The amount Mary spent on clothing -/
def total_spent : ℝ := 25.31

/-- The amount Mary spent on the jacket -/
def jacket_cost : ℝ := 12.27

/-- The number of shops Mary visited -/
def shops_visited : ℕ := 2

/-- The amount Mary spent on the shirt -/
def shirt_cost : ℝ := total_spent - jacket_cost

theorem shirt_cost_calculation : 
  shirt_cost = total_spent - jacket_cost :=
by sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l3746_374609


namespace NUMINAMATH_CALUDE_max_value_of_a_l3746_374679

theorem max_value_of_a (a b c : ℝ) 
  (sum_eq : a + b + c = 6)
  (prod_sum_eq : a * b + a * c + b * c = 11) :
  a ≤ 2 + 2 * Real.sqrt 3 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 6 ∧ 
                    a₀ * b₀ + a₀ * c₀ + b₀ * c₀ = 11 ∧ 
                    a₀ = 2 + 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3746_374679


namespace NUMINAMATH_CALUDE_combinations_equal_fifteen_l3746_374634

/-- The number of window treatment types available. -/
def num_treatments : ℕ := 3

/-- The number of colors available. -/
def num_colors : ℕ := 5

/-- The total number of combinations of window treatment type and color. -/
def total_combinations : ℕ := num_treatments * num_colors

/-- Theorem stating that the total number of combinations is 15. -/
theorem combinations_equal_fifteen : total_combinations = 15 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_fifteen_l3746_374634


namespace NUMINAMATH_CALUDE_unique_solution_l3746_374653

structure Grid :=
  (a b c : ℕ)
  (row_sum : ℕ)
  (col_sum : ℕ)

def is_valid_grid (g : Grid) : Prop :=
  g.row_sum = 9 ∧
  g.col_sum = 12 ∧
  g.a + g.b + g.c = g.row_sum ∧
  4 + g.a + 1 + g.b = g.col_sum ∧
  g.a + 2 + 6 = g.col_sum ∧
  3 + 1 + 6 + g.c = g.col_sum ∧
  g.b + 2 + g.c = g.row_sum

theorem unique_solution :
  ∃! g : Grid, is_valid_grid g ∧ g.a = 6 ∧ g.b = 5 ∧ g.c = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l3746_374653


namespace NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l3746_374697

/-- The cost of the largest pot given 6 pots with increasing prices -/
def largest_pot_cost (num_pots : ℕ) (total_cost : ℚ) (price_difference : ℚ) : ℚ :=
  let smallest_pot_cost := (total_cost - (price_difference * (num_pots - 1) * num_pots / 2)) / num_pots
  smallest_pot_cost + price_difference * (num_pots - 1)

/-- Theorem stating the cost of the largest pot -/
theorem largest_pot_cost_is_correct : 
  largest_pot_cost 6 (39/5) (1/4) = 77/40 := by
  sorry

#eval largest_pot_cost 6 (39/5) (1/4)

end NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l3746_374697


namespace NUMINAMATH_CALUDE_larger_integer_value_l3746_374676

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 5 / 2)
  (h_product : (a : ℕ) * (b : ℕ) = 360) :
  max a b = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3746_374676


namespace NUMINAMATH_CALUDE_circle_triangle_area_difference_l3746_374611

/-- Given an equilateral triangle with side length 12 units and its circumscribed circle,
    the difference between the area of the circle and the area of the triangle
    is 144π - 36√3 square units. -/
theorem circle_triangle_area_difference : 
  let s : ℝ := 12 -- side length of the equilateral triangle
  let r : ℝ := s -- radius of the circumscribed circle (equal to side length)
  let circle_area : ℝ := π * r^2
  let triangle_height : ℝ := s * (Real.sqrt 3) / 2
  let triangle_area : ℝ := (1/2) * s * triangle_height
  circle_area - triangle_area = 144 * π - 36 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_difference_l3746_374611


namespace NUMINAMATH_CALUDE_count_integers_eq_880_l3746_374629

/-- Fibonacci sequence with F₁ = 2 and F₂ = 3 -/
def F : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | n + 2 => F (n + 1) + F n

/-- The number of 10-digit integers with digits 1 or 2 and two consecutive 1's -/
def count_integers : ℕ := 2^10 - F 9

theorem count_integers_eq_880 : count_integers = 880 := by
  sorry

#eval count_integers  -- Should output 880

end NUMINAMATH_CALUDE_count_integers_eq_880_l3746_374629


namespace NUMINAMATH_CALUDE_part1_part2_l3746_374669

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

-- Part 1
theorem part1 (a : ℝ) :
  (∀ x, x ∈ Set.Icc (-5 : ℝ) (-1) ↔ f x a - |x - a| ≤ 2) →
  a = 2 := by sorry

-- Part 2
theorem part2 (a m : ℝ) :
  (∃ x₀, f x₀ a < 4 * m + m^2) →
  m ∈ Set.Ioi 1 ∪ Set.Iio (-5) := by sorry

end NUMINAMATH_CALUDE_part1_part2_l3746_374669


namespace NUMINAMATH_CALUDE_total_savings_is_150_l3746_374642

/-- Calculates the total savings for the year based on the given savings pattern. -/
def total_savings (savings_jan_to_jul : ℕ) (savings_aug_to_nov : ℕ) (savings_dec : ℕ) : ℕ :=
  7 * savings_jan_to_jul + 4 * savings_aug_to_nov + savings_dec

/-- Proves that the total savings for the year is $150 given the specified savings pattern. -/
theorem total_savings_is_150 :
  total_savings 10 15 20 = 150 := by sorry

end NUMINAMATH_CALUDE_total_savings_is_150_l3746_374642


namespace NUMINAMATH_CALUDE_one_quarter_of_seven_point_two_l3746_374643

theorem one_quarter_of_seven_point_two : 
  (7.2 / 4 : ℚ) = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_one_quarter_of_seven_point_two_l3746_374643


namespace NUMINAMATH_CALUDE_gear_diameter_relation_l3746_374645

/-- Represents a circular gear with a diameter and revolutions per minute. -/
structure Gear where
  diameter : ℝ
  rpm : ℝ

/-- Represents a system of two interconnected gears. -/
structure GearSystem where
  gearA : Gear
  gearB : Gear
  /-- The gears travel at the same circumferential rate -/
  same_rate : gearA.diameter * gearA.rpm = gearB.diameter * gearB.rpm

/-- Theorem stating the relationship between gear diameters given their rpm ratio -/
theorem gear_diameter_relation (sys : GearSystem) 
  (h1 : sys.gearB.diameter = 50)
  (h2 : sys.gearA.rpm = 5 * sys.gearB.rpm) :
  sys.gearA.diameter = 10 := by
  sorry

end NUMINAMATH_CALUDE_gear_diameter_relation_l3746_374645


namespace NUMINAMATH_CALUDE_function_inequality_l3746_374659

/-- Given a function f(x) = x^2 - (a + 1/a)x + 1, if for any x in (1, 3),
    f(x) + (1/a)x > -3 always holds, then a < 4. -/
theorem function_inequality (a : ℝ) (h : a > 0) : 
  (∀ x ∈ Set.Ioo 1 3, x^2 - (a + 1/a)*x + 1 + (1/a)*x > -3) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3746_374659


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_symmetric_points_l3746_374674

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- Given that point P(x,1) is symmetric to point Q(-3,y) with respect to the origin, prove that x + y = 2 -/
theorem sum_of_coordinates_of_symmetric_points :
  ∀ x y : ℝ, symmetric_wrt_origin (x, 1) (-3, y) → x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_symmetric_points_l3746_374674


namespace NUMINAMATH_CALUDE_min_value_expression_l3746_374612

theorem min_value_expression (a b c : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3746_374612


namespace NUMINAMATH_CALUDE_long_distance_call_cost_decrease_l3746_374627

/-- The percent decrease in cost of a long-distance call --/
def percent_decrease (initial_cost final_cost : ℚ) : ℚ :=
  (initial_cost - final_cost) / initial_cost * 100

/-- Theorem: The percent decrease from 35 cents to 5 cents is approximately 86% --/
theorem long_distance_call_cost_decrease :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |percent_decrease (35/100) (5/100) - 86| < ε :=
sorry

end NUMINAMATH_CALUDE_long_distance_call_cost_decrease_l3746_374627


namespace NUMINAMATH_CALUDE_max_value_theorem_l3746_374677

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

noncomputable def g (x : ℝ) : ℝ := -(Real.log x) / x

theorem max_value_theorem (x₁ x₂ t : ℝ) (h1 : f x₁ = t) (h2 : g x₂ = t) (h3 : t > 0) :
  (∀ y₁ y₂ s : ℝ, f y₁ = s → g y₂ = s → s > 0 → y₁ / (y₂ * Real.exp s) ≤ 1 / Real.exp 1) ∧
  (∃ z₁ z₂ r : ℝ, f z₁ = r ∧ g z₂ = r ∧ r > 0 ∧ z₁ / (z₂ * Real.exp r) = 1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3746_374677


namespace NUMINAMATH_CALUDE_simplify_expression_l3746_374658

theorem simplify_expression : 
  (Real.sqrt 6 - Real.sqrt 18) * Real.sqrt (1/3) + 2 * Real.sqrt 6 = Real.sqrt 2 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3746_374658


namespace NUMINAMATH_CALUDE_ellipse_parameter_range_l3746_374663

/-- The equation of an ellipse with parameter m -/
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (2 + m) + y^2 / (1 - m) = 1

/-- Conditions for the equation to represent an ellipse with foci on the x-axis -/
def is_valid_ellipse (m : ℝ) : Prop :=
  2 + m > 0 ∧ 1 - m > 0 ∧ 2 + m > 1 - m

/-- The range of m for which the equation represents a valid ellipse -/
theorem ellipse_parameter_range :
  ∀ m : ℝ, is_valid_ellipse m ↔ -1/2 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_parameter_range_l3746_374663


namespace NUMINAMATH_CALUDE_lindsay_dolls_theorem_l3746_374625

theorem lindsay_dolls_theorem (blonde : ℕ) (brown : ℕ) (black : ℕ) : 
  blonde = 4 →
  brown = 4 * blonde →
  black = brown - 2 →
  brown + black - blonde = 26 := by
  sorry

end NUMINAMATH_CALUDE_lindsay_dolls_theorem_l3746_374625


namespace NUMINAMATH_CALUDE_x_minus_p_equals_two_l3746_374665

theorem x_minus_p_equals_two (x p : ℝ) (h1 : |x - 2| = p) (h2 : x > 2) : x - p = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_p_equals_two_l3746_374665


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_relation_l3746_374638

/-- A quadrilateral inscribed in a semicircle -/
structure InscribedQuadrilateral where
  /-- Side length a -/
  a : ℝ
  /-- Side length b -/
  b : ℝ
  /-- Side length c -/
  c : ℝ
  /-- Side length d, which is also the diameter of the semicircle -/
  d : ℝ
  /-- All side lengths are positive -/
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  d_pos : 0 < d
  /-- The quadrilateral is inscribed in a semicircle with diameter d -/
  inscribed : True

/-- The main theorem about the relationship between side lengths of an inscribed quadrilateral -/
theorem inscribed_quadrilateral_relation (q : InscribedQuadrilateral) :
  q.d^3 - (q.a^2 + q.b^2 + q.c^2) * q.d - 2 * q.a * q.b * q.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_relation_l3746_374638


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3746_374678

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a + b + 2c = a² and a + b - 2c = -1, then the largest angle is 120°. -/
theorem largest_angle_in_special_triangle (a b c : ℝ) (h1 : a + b + 2*c = a^2) (h2 : a + b - 2*c = -1) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
    A + B + C = Real.pi ∧    -- Sum of angles in a triangle
    max A (max B C) = 2*Real.pi/3 :=  -- Largest angle is 120°
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l3746_374678


namespace NUMINAMATH_CALUDE_unique_x_with_703_factors_l3746_374614

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- x^x has exactly 703 positive factors -/
def has_703_factors (x : ℕ) : Prop :=
  num_factors (x^x) = 703

theorem unique_x_with_703_factors :
  ∃! x : ℕ, x > 0 ∧ has_703_factors x ∧ x = 18 := by sorry

end NUMINAMATH_CALUDE_unique_x_with_703_factors_l3746_374614


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l3746_374632

/-- Given a triangle ABC where angle A is 60° and side a is 4, 
    the maximum perimeter of the triangle is 12. -/
theorem triangle_max_perimeter (b c : ℝ) : 
  let A : ℝ := 60 * π / 180  -- Convert 60° to radians
  let a : ℝ := 4
  b > 0 → c > 0 →   -- Ensure positive side lengths
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →  -- Cosine theorem
  a + b + c ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l3746_374632


namespace NUMINAMATH_CALUDE_time_spent_calculation_susan_time_allocation_l3746_374656

/-- Given a ratio of activities and time spent on one activity, calculate the time spent on another activity -/
theorem time_spent_calculation (reading_ratio : ℕ) (hangout_ratio : ℕ) (reading_hours : ℕ) 
  (h1 : reading_ratio > 0)
  (h2 : hangout_ratio > 0)
  (h3 : reading_hours > 0) :
  (reading_ratio * (hangout_ratio * reading_hours) / reading_ratio) = hangout_ratio * reading_hours :=
by sorry

/-- Susan's time allocation problem -/
theorem susan_time_allocation :
  let reading_ratio : ℕ := 4
  let hangout_ratio : ℕ := 10
  let reading_hours : ℕ := 8
  (reading_ratio * (hangout_ratio * reading_hours) / reading_ratio) = 20 :=
by sorry

end NUMINAMATH_CALUDE_time_spent_calculation_susan_time_allocation_l3746_374656


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3746_374616

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 6 ∧ b = 8 ∧ c > 0 ∧ a^2 + b^2 = c^2 → c = 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3746_374616


namespace NUMINAMATH_CALUDE_parabola_area_and_binomial_expansion_l3746_374603

/-- Given a > 0 and the area enclosed by y² = ax and x = 1 is 4/3, 
    the coefficient of x⁻¹⁸ in the expansion of (x + a/x)²⁰ is 20 -/
theorem parabola_area_and_binomial_expansion (a : ℝ) (h1 : a > 0) 
  (h2 : (2 : ℝ) * ∫ x in (0 : ℝ)..(1 : ℝ), (a * x).sqrt = 4/3) :
  (Finset.range 21).sum (fun k => Nat.choose 20 k * a^k * (-1)^(19 - k)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_parabola_area_and_binomial_expansion_l3746_374603


namespace NUMINAMATH_CALUDE_mrs_hilt_apples_per_hour_l3746_374652

/-- Given a total number of apples and hours, calculate the apples eaten per hour -/
def apples_per_hour (total_apples : ℕ) (total_hours : ℕ) : ℚ :=
  total_apples / total_hours

/-- Theorem: Mrs. Hilt ate 5 apples per hour -/
theorem mrs_hilt_apples_per_hour :
  apples_per_hour 15 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_apples_per_hour_l3746_374652


namespace NUMINAMATH_CALUDE_chairs_in_clubroom_l3746_374624

/-- Represents the number of chairs in the clubroom -/
def num_chairs : ℕ := 17

/-- Represents the number of legs each chair has -/
def legs_per_chair : ℕ := 4

/-- Represents the number of legs the table has -/
def table_legs : ℕ := 3

/-- Represents the number of unoccupied chairs -/
def unoccupied_chairs : ℕ := 2

/-- Represents the total number of legs in the room -/
def total_legs : ℕ := 101

/-- Proves that the number of chairs in the clubroom is correct given the conditions -/
theorem chairs_in_clubroom :
  num_chairs * legs_per_chair + table_legs = total_legs + 2 * (num_chairs - unoccupied_chairs) :=
by sorry

end NUMINAMATH_CALUDE_chairs_in_clubroom_l3746_374624


namespace NUMINAMATH_CALUDE_circle_fixed_points_l3746_374672

theorem circle_fixed_points (m : ℝ) :
  let circle := λ (x y : ℝ) => x^2 + y^2 - 2*m*x - 4*m*y + 6*m - 2
  circle 1 1 = 0 ∧ circle (1/5) (7/5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_fixed_points_l3746_374672


namespace NUMINAMATH_CALUDE_total_combinations_eq_twelve_l3746_374631

/-- The number of paint colors available. -/
def num_colors : ℕ := 4

/-- The number of painting methods available. -/
def num_methods : ℕ := 3

/-- The total number of combinations of paint color and painting method. -/
def total_combinations : ℕ := num_colors * num_methods

/-- Theorem stating that the total number of combinations is 12. -/
theorem total_combinations_eq_twelve : total_combinations = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_combinations_eq_twelve_l3746_374631


namespace NUMINAMATH_CALUDE_distance_from_two_is_six_l3746_374654

theorem distance_from_two_is_six (x : ℝ) : |x - 2| = 6 → x = 8 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_two_is_six_l3746_374654


namespace NUMINAMATH_CALUDE_eric_apples_l3746_374688

theorem eric_apples (r y g : ℕ) : 
  r = y →                           -- Red apples = Yellow apples (in first box)
  r = (1/3 : ℚ) * (r + g : ℚ) →     -- Red apples are 1/3 of second box after moving
  r + y + g = 28 →                  -- Total number of apples
  r = 7 := by sorry

end NUMINAMATH_CALUDE_eric_apples_l3746_374688


namespace NUMINAMATH_CALUDE_sequence_average_bound_l3746_374613

theorem sequence_average_bound (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ k ∈ Finset.range n, k > 1 → |a k| = |a (k-1) + 1|) :
  (Finset.sum (Finset.range n) (λ i => a (i+1))) / n ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_average_bound_l3746_374613


namespace NUMINAMATH_CALUDE_both_languages_difference_l3746_374608

/-- The total number of students in the school -/
def total_students : ℕ := 2500

/-- The minimum percentage of students studying Italian -/
def min_italian_percent : ℚ := 70 / 100

/-- The maximum percentage of students studying Italian -/
def max_italian_percent : ℚ := 75 / 100

/-- The minimum percentage of students studying German -/
def min_german_percent : ℚ := 35 / 100

/-- The maximum percentage of students studying German -/
def max_german_percent : ℚ := 45 / 100

/-- The number of students studying Italian -/
def italian_students (n : ℕ) : Prop :=
  ⌈(min_italian_percent * total_students : ℚ)⌉ ≤ n ∧ n ≤ ⌊(max_italian_percent * total_students : ℚ)⌋

/-- The number of students studying German -/
def german_students (n : ℕ) : Prop :=
  ⌈(min_german_percent * total_students : ℚ)⌉ ≤ n ∧ n ≤ ⌊(max_german_percent * total_students : ℚ)⌋

/-- The theorem stating the difference between max and min number of students studying both languages -/
theorem both_languages_difference :
  ∃ (max min : ℕ),
    (∀ i g b, italian_students i → german_students g → i + g - b = total_students → b ≤ max) ∧
    (∀ i g b, italian_students i → german_students g → i + g - b = total_students → min ≤ b) ∧
    max - min = 375 := by
  sorry

end NUMINAMATH_CALUDE_both_languages_difference_l3746_374608


namespace NUMINAMATH_CALUDE_fox_catches_rabbits_l3746_374605

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the game setup -/
structure GameSetup where
  A : Point
  B : Point
  C : Point
  D : Point
  foxSpeed : ℝ
  rabbitSpeed : ℝ

/-- Checks if the fox can catch both rabbits -/
def canCatchBothRabbits (setup : GameSetup) : Prop :=
  setup.foxSpeed ≥ 1 + Real.sqrt 2

theorem fox_catches_rabbits (setup : GameSetup) 
  (h1 : setup.A = ⟨0, 0⟩) 
  (h2 : setup.B = ⟨1, 0⟩) 
  (h3 : setup.C = ⟨1, 1⟩) 
  (h4 : setup.D = ⟨0, 1⟩)
  (h5 : setup.rabbitSpeed = 1) :
  canCatchBothRabbits setup ↔ 
    ∀ (t : ℝ), t ≥ 0 → 
      ∃ (foxPos : Point),
        (foxPos.x - setup.C.x)^2 + (foxPos.y - setup.C.y)^2 ≤ (setup.foxSpeed * t)^2 ∧
        ((foxPos.x = setup.B.x + t ∧ foxPos.y = 0) ∨
         (foxPos.x = 0 ∧ foxPos.y = setup.D.y + t) ∨
         (foxPos.x = 0 ∧ foxPos.y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_fox_catches_rabbits_l3746_374605


namespace NUMINAMATH_CALUDE_no_two_digit_sum_reverse_21_l3746_374610

theorem no_two_digit_sum_reverse_21 : 
  ¬ ∃ (N : ℕ), 
    10 ≤ N ∧ N < 100 ∧ 
    (N + (10 * (N % 10) + N / 10) = 21) :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_sum_reverse_21_l3746_374610


namespace NUMINAMATH_CALUDE_sorting_abc_l3746_374690

theorem sorting_abc (a b c : Real)
  (ha : 0 ≤ a ∧ a ≤ Real.pi / 2)
  (hb : 0 ≤ b ∧ b ≤ Real.pi / 2)
  (hc : 0 ≤ c ∧ c ≤ Real.pi / 2)
  (ca : Real.cos a = a)
  (sb : Real.sin (Real.cos b) = b)
  (cs : Real.cos (Real.sin c) = c) :
  b < a ∧ a < c := by
sorry

end NUMINAMATH_CALUDE_sorting_abc_l3746_374690


namespace NUMINAMATH_CALUDE_classes_taught_total_l3746_374644

/-- The number of classes Eduardo taught -/
def eduardo_classes : ℕ := 3

/-- The number of classes Frankie taught -/
def frankie_classes : ℕ := 2 * eduardo_classes

/-- The total number of classes taught by Eduardo and Frankie -/
def total_classes : ℕ := eduardo_classes + frankie_classes

theorem classes_taught_total : total_classes = 9 := by
  sorry

end NUMINAMATH_CALUDE_classes_taught_total_l3746_374644


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3746_374671

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ = -10 + 10 * Real.sqrt 2) ∧ 
              (x₂ = -10 - 10 * Real.sqrt 2) ∧ 
              (∀ x : ℝ, (10 - x)^2 = 2*x^2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3746_374671


namespace NUMINAMATH_CALUDE_solution_implies_m_value_l3746_374648

theorem solution_implies_m_value (m : ℚ) :
  (∀ x : ℚ, (m - 2) * x = 5 * (x + 1) → x = 2) →
  m = 19 / 2 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_m_value_l3746_374648


namespace NUMINAMATH_CALUDE_defect_rate_two_procedures_l3746_374692

/-- The defect rate of a product after two independent procedures -/
def overall_defect_rate (a b : ℝ) : ℝ := 1 - (1 - a) * (1 - b)

/-- Theorem: The overall defect rate of a product after two independent procedures
    with defect rates a and b is 1 - (1-a)(1-b) -/
theorem defect_rate_two_procedures
  (a b : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  : overall_defect_rate a b = 1 - (1 - a) * (1 - b) :=
by sorry

end NUMINAMATH_CALUDE_defect_rate_two_procedures_l3746_374692


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l3746_374618

theorem arithmetic_simplification : 4 * (8 - 3) - 6 / 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l3746_374618


namespace NUMINAMATH_CALUDE_min_sum_of_equal_multiples_l3746_374689

theorem min_sum_of_equal_multiples (x y z : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (m : ℕ+), ∀ (a b c : ℕ+), ((4 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val) →
    m.val ≤ a.val + b.val + c.val ∧ m.val = x.val + y.val + z.val ∧ m.val = 37 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_equal_multiples_l3746_374689


namespace NUMINAMATH_CALUDE_intersection_condition_l3746_374623

theorem intersection_condition (a : ℝ) : 
  let A : Set ℝ := {0, 1}
  let B : Set ℝ := {x | x > a}
  (∃! x, x ∈ A ∩ B) → 0 ≤ a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l3746_374623


namespace NUMINAMATH_CALUDE_new_savings_is_200_l3746_374661

/-- Calculates the new monthly savings after an increase in expenses -/
def new_monthly_savings (salary : ℚ) (initial_savings_rate : ℚ) (expense_increase_rate : ℚ) : ℚ :=
  let initial_expenses := salary * (1 - initial_savings_rate)
  let new_expenses := initial_expenses * (1 + expense_increase_rate)
  salary - new_expenses

/-- Proves that the new monthly savings is 200 given the specified conditions -/
theorem new_savings_is_200 :
  new_monthly_savings 5000 (20 / 100) (20 / 100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_new_savings_is_200_l3746_374661


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l3746_374675

theorem no_positive_integer_solutions :
  ¬∃ (x y : ℕ+), x^2 + y^2 = x^4 := by sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l3746_374675


namespace NUMINAMATH_CALUDE_cubic_inequality_l3746_374607

theorem cubic_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3746_374607


namespace NUMINAMATH_CALUDE_backpack_cost_l3746_374640

/-- The cost of backpacks with discount and monogramming -/
theorem backpack_cost (original_price : ℝ) (discount_percent : ℝ) (monogram_fee : ℝ) (quantity : ℕ) :
  original_price = 20 →
  discount_percent = 20 →
  monogram_fee = 12 →
  quantity = 5 →
  quantity * (original_price * (1 - discount_percent / 100) + monogram_fee) = 140 := by
  sorry

end NUMINAMATH_CALUDE_backpack_cost_l3746_374640


namespace NUMINAMATH_CALUDE_not_square_n5_plus_7_l3746_374680

theorem not_square_n5_plus_7 (n : ℤ) (h : n > 1) : ¬ ∃ k : ℤ, n^5 + 7 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_n5_plus_7_l3746_374680


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l3746_374604

theorem count_ordered_pairs : 
  (Finset.filter (fun p : ℕ × ℕ => p.1^2 * p.2 = 20^20) (Finset.product (Finset.range (20^20 + 1)) (Finset.range (20^20 + 1)))).card = 231 :=
sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_l3746_374604


namespace NUMINAMATH_CALUDE_product_minus_third_lower_bound_l3746_374695

theorem product_minus_third_lower_bound 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0) 
  (a : ℝ) 
  (h1 : x * y - z = a) 
  (h2 : y * z - x = a) 
  (h3 : z * x - y = a) : 
  a ≥ -1/4 := by
sorry

end NUMINAMATH_CALUDE_product_minus_third_lower_bound_l3746_374695


namespace NUMINAMATH_CALUDE_base_equivalence_l3746_374667

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (x y : Nat) : Nat :=
  x * 6 + y

/-- Converts a number in base b to base 10 -/
def baseBToBase10 (b x y z : Nat) : Nat :=
  x * b^2 + y * b + z

theorem base_equivalence :
  ∃! (b : Nat), b > 0 ∧ base6ToBase10 5 3 = baseBToBase10 b 1 1 3 :=
by sorry

end NUMINAMATH_CALUDE_base_equivalence_l3746_374667


namespace NUMINAMATH_CALUDE_equation_has_integer_solution_l3746_374630

theorem equation_has_integer_solution (a b : ℤ) : ∃ x : ℤ, (x - a) * (x - b) * (x - 3) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_has_integer_solution_l3746_374630


namespace NUMINAMATH_CALUDE_inverse_between_zero_and_one_l3746_374649

theorem inverse_between_zero_and_one (x : ℝ) : 0 < (1 : ℝ) / x ∧ (1 : ℝ) / x < 1 ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_between_zero_and_one_l3746_374649


namespace NUMINAMATH_CALUDE_population_and_sample_properties_l3746_374698

/-- Represents a student in the seventh grade -/
structure Student where
  id : Nat

/-- Represents a population of students -/
structure Population where
  students : Finset Student
  size : Nat
  h_size : students.card = size

/-- Represents a sample of students -/
structure Sample where
  students : Finset Student
  population : Population
  h_subset : students ⊆ population.students

/-- The main theorem stating properties of the population and sample -/
theorem population_and_sample_properties
  (total_students : Finset Student)
  (h_total : total_students.card = 800)
  (sample_students : Finset Student)
  (h_sample : sample_students ⊆ total_students)
  (h_sample_size : sample_students.card = 50) :
  let pop : Population := ⟨total_students, 800, h_total⟩
  let samp : Sample := ⟨sample_students, pop, h_sample⟩
  (pop.size = 800) ∧
  (samp.students ⊆ pop.students) ∧
  (samp.students.card = 50) := by
  sorry


end NUMINAMATH_CALUDE_population_and_sample_properties_l3746_374698


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3746_374639

theorem base_conversion_problem :
  ∀ c d : ℕ,
  c < 10 → d < 10 →
  (5 * 6^2 + 2 * 6^1 + 4 * 6^0 = 2 * 10^2 + c * 10^1 + d * 10^0) →
  (c * d : ℚ) / 12 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3746_374639


namespace NUMINAMATH_CALUDE_l₁_passes_through_neg_one_neg_one_perpendicular_condition_l3746_374691

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def l₂ (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

-- Theorem 1: l₁ passes through (-1, -1) for all a
theorem l₁_passes_through_neg_one_neg_one (a : ℝ) : l₁ a (-1) (-1) := by sorry

-- Theorem 2: If l₁ ⊥ l₂, then a = 0 or a = -4
theorem perpendicular_condition (a : ℝ) : 
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0) → 
  a = 0 ∨ a = -4 := by sorry

end NUMINAMATH_CALUDE_l₁_passes_through_neg_one_neg_one_perpendicular_condition_l3746_374691
