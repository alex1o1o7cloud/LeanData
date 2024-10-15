import Mathlib

namespace NUMINAMATH_CALUDE_m_plus_e_equals_22_l3176_317691

def base_value (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem m_plus_e_equals_22 (m e : Nat) :
  m > 0 →
  e < 10 →
  base_value [4, 1, e] m = 346 →
  base_value [4, 1, 6] m = base_value [1, 2, e, 1] 7 →
  m + e = 22 :=
by sorry

end NUMINAMATH_CALUDE_m_plus_e_equals_22_l3176_317691


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3176_317635

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ Real.log (a₀ + b₀) = 0 ∧ 1 / a₀ + 1 / b₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3176_317635


namespace NUMINAMATH_CALUDE_no_single_solution_quadratic_inequality_l3176_317620

theorem no_single_solution_quadratic_inequality :
  ¬ ∃ (b : ℝ), ∃! (x : ℝ), |x^2 + 3*b*x + 4*b| ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_no_single_solution_quadratic_inequality_l3176_317620


namespace NUMINAMATH_CALUDE_brendas_age_l3176_317699

theorem brendas_age (addison janet brenda : ℚ) 
  (h1 : addison = 4 * brenda) 
  (h2 : janet = brenda + 8) 
  (h3 : addison = janet) : 
  brenda = 8/3 := by
sorry

end NUMINAMATH_CALUDE_brendas_age_l3176_317699


namespace NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l3176_317679

/-- Represents a parabola of the form y = a(x-h)^2 + k --/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The axis of symmetry of a parabola --/
def axisOfSymmetry (p : Parabola) : ℝ := p.h

/-- The given parabola y = -2(x-1)^2 + 3 --/
def givenParabola : Parabola := ⟨-2, 1, 3⟩

theorem axis_of_symmetry_is_x_equals_one :
  axisOfSymmetry givenParabola = 1 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_is_x_equals_one_l3176_317679


namespace NUMINAMATH_CALUDE_add_same_power_of_x_l3176_317639

theorem add_same_power_of_x (x : ℝ) : x^3 + x^3 = 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_add_same_power_of_x_l3176_317639


namespace NUMINAMATH_CALUDE_certain_number_proof_l3176_317616

theorem certain_number_proof (x y z N : ℤ) : 
  x < y → y < z →
  y - x > N →
  Even x →
  Odd y →
  Odd z →
  (∀ w, w - x ≥ 13 → w ≥ z) →
  (∃ u v, u < v ∧ v < z ∧ v - u > N ∧ Even u ∧ Odd v ∧ v - x < 13) →
  N ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3176_317616


namespace NUMINAMATH_CALUDE_glee_club_gender_ratio_l3176_317661

/-- Given a glee club with total members and female members, 
    prove the ratio of female to male members -/
theorem glee_club_gender_ratio (total : ℕ) (female : ℕ) 
    (h1 : total = 18) (h2 : female = 12) :
    (female : ℚ) / ((total - female) : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_glee_club_gender_ratio_l3176_317661


namespace NUMINAMATH_CALUDE_straight_line_probability_l3176_317641

/-- The number of dots in each row or column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots required to form a line -/
def dotsInLine : ℕ := 4

/-- The number of possible straight lines containing four dots in a 5x5 grid -/
def numStraightLines : ℕ := 16

/-- The total number of ways to choose 4 dots from 25 dots -/
def totalWaysToChoose : ℕ := Nat.choose totalDots dotsInLine

/-- The probability of selecting four dots that form a straight line -/
def probabilityOfStraightLine : ℚ := numStraightLines / totalWaysToChoose

theorem straight_line_probability :
  probabilityOfStraightLine = 16 / 12650 := by sorry

end NUMINAMATH_CALUDE_straight_line_probability_l3176_317641


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3176_317638

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The property that three terms form an arithmetic sequence -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSequence (3 * a 1) ((1/2) * a 3) (2 * a 2) →
  (a 2014 + a 2015) / (a 2012 + a 2013) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3176_317638


namespace NUMINAMATH_CALUDE_even_mono_decreasing_order_l3176_317655

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def isMonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

theorem even_mono_decreasing_order (f : ℝ → ℝ) 
  (h_even : isEven f) 
  (h_mono : isMonoDecreasing f 0 3) : 
  f (-1) > f 2 ∧ f 2 > f 3 := by
  sorry

end NUMINAMATH_CALUDE_even_mono_decreasing_order_l3176_317655


namespace NUMINAMATH_CALUDE_present_age_ratio_l3176_317621

/-- Given A's present age and the future ratio of ages, prove the present ratio of ages -/
theorem present_age_ratio (a b : ℕ) (h1 : a = 15) (h2 : (a + 6) * 5 = (b + 6) * 7) :
  5 * b = 3 * a := by
  sorry

end NUMINAMATH_CALUDE_present_age_ratio_l3176_317621


namespace NUMINAMATH_CALUDE_remi_seedlings_proof_l3176_317636

/-- The number of seedlings Remi planted on the first day -/
def first_day_seedlings : ℕ := 400

/-- The number of seedlings Remi planted on the second day -/
def second_day_seedlings : ℕ := 2 * first_day_seedlings

/-- The total number of seedlings transferred over the two days -/
def total_seedlings : ℕ := 1200

theorem remi_seedlings_proof :
  first_day_seedlings + second_day_seedlings = total_seedlings ∧
  first_day_seedlings = 400 := by
  sorry

end NUMINAMATH_CALUDE_remi_seedlings_proof_l3176_317636


namespace NUMINAMATH_CALUDE_inequality_implication_l3176_317671

theorem inequality_implication (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) :
  y * (y - 1) ≤ x^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3176_317671


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3176_317600

theorem fraction_meaningful (x : ℝ) : 
  (x - 2) / (2 * x - 3) ≠ 0 ↔ x ≠ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3176_317600


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_upper_bound_l3176_317675

theorem quadratic_inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 3 → a ≤ x^2 - 4*x) →
  a ≤ -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_upper_bound_l3176_317675


namespace NUMINAMATH_CALUDE_score_difference_l3176_317651

/-- Represents the runs scored by each batsman -/
structure BatsmanScores where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ

/-- Theorem stating the difference between A's and E's scores -/
theorem score_difference (scores : BatsmanScores) : scores.a - scores.e = 8 :=
  by
  have h1 : scores.a + scores.b + scores.c + scores.d + scores.e = 180 :=
    sorry -- Average score is 36, so total is 5 * 36 = 180
  have h2 : scores.d = scores.e + 5 := sorry -- D scored 5 more than E
  have h3 : scores.e = 20 := sorry -- E scored 20 runs
  have h4 : scores.b = scores.d + scores.e := sorry -- B scored as many as D and E combined
  have h5 : scores.b + scores.c = 107 := sorry -- B and C scored 107 between them
  have h6 : scores.e < scores.a := sorry -- E scored fewer runs than A
  sorry -- Prove that scores.a - scores.e = 8

end NUMINAMATH_CALUDE_score_difference_l3176_317651


namespace NUMINAMATH_CALUDE_third_cube_edge_l3176_317610

theorem third_cube_edge (a b c x : ℝ) (ha : a = 3) (hb : b = 5) (hc : c = 6) :
  a^3 + b^3 + x^3 = c^3 → x = 4 := by sorry

end NUMINAMATH_CALUDE_third_cube_edge_l3176_317610


namespace NUMINAMATH_CALUDE_ben_gross_income_l3176_317663

-- Define Ben's financial situation
def ben_finances (gross_income : ℝ) : Prop :=
  ∃ (after_tax_income : ℝ),
    -- 20% of after-tax income is spent on car
    0.2 * after_tax_income = 400 ∧
    -- 1/3 of gross income is paid in taxes
    gross_income - (1/3) * gross_income = after_tax_income

-- Theorem statement
theorem ben_gross_income :
  ∃ (gross_income : ℝ), ben_finances gross_income ∧ gross_income = 3000 :=
sorry

end NUMINAMATH_CALUDE_ben_gross_income_l3176_317663


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3176_317678

theorem polynomial_simplification (x : ℝ) : 
  (2 * x^6 + x^5 + 3 * x^4 + 7 * x^2 + 2 * x + 25) - 
  (x^6 + 2 * x^5 + x^4 + x^3 + 8 * x^2 + 15) = 
  x^6 - x^5 + 2 * x^4 - x^3 - x^2 + 2 * x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3176_317678


namespace NUMINAMATH_CALUDE_sequence_general_term_l3176_317653

theorem sequence_general_term (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≥ 2 → a n / a (n - 1) = 2^(n - 1)) →
  a 1 = 1 →
  ∀ n : ℕ, n > 0 → a n = 2^(n * (n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3176_317653


namespace NUMINAMATH_CALUDE_no_real_solutions_l3176_317632

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 6)^2 + 4 = -(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3176_317632


namespace NUMINAMATH_CALUDE_alex_total_cost_l3176_317681

/-- Calculates the total cost of a cell phone plan given the usage and rates. -/
def calculate_total_cost (base_cost : ℚ) (included_hours : ℚ) (text_cost : ℚ) 
  (extra_minute_cost : ℚ) (texts_sent : ℕ) (hours_talked : ℚ) : ℚ :=
  let extra_hours := max (hours_talked - included_hours) 0
  let extra_minutes := extra_hours * 60
  base_cost + (text_cost * texts_sent) + (extra_minute_cost * extra_minutes)

/-- Proves that Alex's total cost is $109.00 given the specified plan and usage. -/
theorem alex_total_cost : 
  calculate_total_cost 25 25 0.08 0.15 150 33 = 109 := by
  sorry

end NUMINAMATH_CALUDE_alex_total_cost_l3176_317681


namespace NUMINAMATH_CALUDE_mod_congruence_problem_l3176_317614

theorem mod_congruence_problem (n : ℕ) : 
  (123^2 * 947) % 60 = n ∧ 0 ≤ n ∧ n < 60 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_problem_l3176_317614


namespace NUMINAMATH_CALUDE_craig_travel_difference_l3176_317607

theorem craig_travel_difference : 
  let bus_distance : ℝ := 3.83
  let walk_distance : ℝ := 0.17
  bus_distance - walk_distance = 3.66 := by
  sorry

end NUMINAMATH_CALUDE_craig_travel_difference_l3176_317607


namespace NUMINAMATH_CALUDE_baseball_league_games_played_l3176_317647

/-- Represents a baseball league with a given number of teams and games per pair of teams. -/
structure BaseballLeague where
  num_teams : ℕ
  games_per_pair : ℕ

/-- Calculates the total number of games played in a baseball league with one team forfeiting one game against each other team. -/
def games_actually_played (league : BaseballLeague) : ℕ :=
  let total_scheduled := (league.num_teams * (league.num_teams - 1) * league.games_per_pair) / 2
  let forfeited_games := league.num_teams - 1
  total_scheduled - forfeited_games

/-- Theorem stating that in a baseball league with 10 teams, 4 games per pair, and one team forfeiting one game against each other team, the total number of games actually played is 171. -/
theorem baseball_league_games_played :
  let league : BaseballLeague := { num_teams := 10, games_per_pair := 4 }
  games_actually_played league = 171 := by
  sorry

end NUMINAMATH_CALUDE_baseball_league_games_played_l3176_317647


namespace NUMINAMATH_CALUDE_arrangement_count_l3176_317669

/-- The number of distinct arrangements of 9 indistinguishable objects and 3 indistinguishable objects in a row of 12 positions -/
def distinct_arrangements : ℕ := 220

/-- The total number of positions -/
def total_positions : ℕ := 12

/-- The number of indistinguishable objects of the first type (armchairs) -/
def first_object_count : ℕ := 9

/-- The number of indistinguishable objects of the second type (benches) -/
def second_object_count : ℕ := 3

theorem arrangement_count :
  distinct_arrangements = (total_positions.choose second_object_count) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3176_317669


namespace NUMINAMATH_CALUDE_base_difference_theorem_l3176_317601

/-- Converts a number from base 5 to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

theorem base_difference_theorem :
  let n1 := 543210
  let n2 := 43210
  (base5_to_base10 n1) - (base8_to_base10 n2) = 499 := by sorry

end NUMINAMATH_CALUDE_base_difference_theorem_l3176_317601


namespace NUMINAMATH_CALUDE_extremum_derivative_zero_not_sufficient_l3176_317613

-- Define a differentiable function f on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for a point to be an extremum
def IsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

-- Theorem statement
theorem extremum_derivative_zero
  (x₀ : ℝ)
  (h_extremum : IsExtremum f x₀) :
  deriv f x₀ = 0 :=
sorry

-- Counter-example to show the converse is not always true
def counter_example : ℝ → ℝ := fun x ↦ x^3

theorem not_sufficient
  (h_deriv_zero : deriv counter_example 0 = 0)
  (h_not_extremum : ¬ IsExtremum counter_example 0) :
  ∃ f : ℝ → ℝ, ∃ x₀ : ℝ, deriv f x₀ = 0 ∧ ¬ IsExtremum f x₀ :=
sorry

end NUMINAMATH_CALUDE_extremum_derivative_zero_not_sufficient_l3176_317613


namespace NUMINAMATH_CALUDE_function_properties_and_triangle_l3176_317668

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.cos x

theorem function_properties_and_triangle (m : ℝ) (A : ℝ) :
  f m (π / 2) = 1 →
  f m (π / 12) = Real.sqrt 2 * Real.sin A →
  0 < A ∧ A < π / 2 →
  (3 * Real.sqrt 3) / 2 = 1 / 2 * 2 * 3 * Real.sin A →
  m = 1 ∧
  (∀ x : ℝ, f m (x + 2 * π) = f m x) ∧
  (∀ x : ℝ, f m x ≤ Real.sqrt 2) ∧
  (∀ x : ℝ, f m x ≥ -Real.sqrt 2) ∧
  A = π / 3 ∧
  Real.sqrt 7 = Real.sqrt (3^2 + 2^2 - 2 * 2 * 3 * Real.cos A) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_and_triangle_l3176_317668


namespace NUMINAMATH_CALUDE_viewing_time_calculation_l3176_317611

/-- Represents the duration of a TV show episode in minutes -/
def episode_duration : ℕ := 30

/-- Represents the number of weekdays in a week -/
def weekdays : ℕ := 5

/-- Represents the number of episodes watched -/
def episodes_watched : ℕ := 4

/-- Calculates the total viewing time in hours -/
def total_viewing_time : ℚ :=
  (episode_duration * episodes_watched) / 60

theorem viewing_time_calculation :
  total_viewing_time = 2 :=
sorry

end NUMINAMATH_CALUDE_viewing_time_calculation_l3176_317611


namespace NUMINAMATH_CALUDE_probability_x_more_points_than_y_in_given_tournament_l3176_317652

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  num_games_per_team : Nat
  win_probability : ℚ
  x_beats_y_first : Bool

/-- The probability that team X finishes with more points than team Y -/
def probability_x_more_points_than_y (tournament : SoccerTournament) : ℚ :=
  sorry

/-- The specific tournament described in the problem -/
def given_tournament : SoccerTournament where
  num_teams := 8
  num_games_per_team := 7
  win_probability := 1/2
  x_beats_y_first := true

theorem probability_x_more_points_than_y_in_given_tournament :
  probability_x_more_points_than_y given_tournament = 610/1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_more_points_than_y_in_given_tournament_l3176_317652


namespace NUMINAMATH_CALUDE_consecutive_integers_transformation_l3176_317606

/-- Sum of squares of first m positive integers -/
def sum_of_squares (m : ℕ) : ℕ := m * (m + 1) * (2 * m + 1) / 6

/-- Sum of squares of 2n consecutive integers starting from k -/
def consecutive_sum_of_squares (n k : ℕ) : ℕ :=
  sum_of_squares (k + 2*n - 1) - sum_of_squares (k - 1)

theorem consecutive_integers_transformation (n : ℕ) :
  ∀ k m : ℕ, ∃ t : ℕ, 2^t * consecutive_sum_of_squares n 1 ≠ consecutive_sum_of_squares n k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_transformation_l3176_317606


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2root2_l3176_317686

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  ∀ x y : ℝ, x > 0 → y > 0 → (x + 1)⁻¹ + (y + 1)⁻¹ = 1 → a + 2*b ≤ x + 2*y :=
by
  sorry

theorem min_value_is_2root2 (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a + 1)⁻¹ + (b + 1)⁻¹ = 1) : 
  a + 2*b = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_min_value_is_2root2_l3176_317686


namespace NUMINAMATH_CALUDE_min_value_expression_l3176_317659

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 5) :
  ((x + 1) * (2*y + 1)) / Real.sqrt (x*y) ≥ 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3176_317659


namespace NUMINAMATH_CALUDE_cubic_integer_values_l3176_317690

/-- A cubic polynomial function -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - c * x - d

/-- Theorem: If a cubic polynomial takes integer values at -1, 0, 1, and 2, 
    then it takes integer values for all integer inputs -/
theorem cubic_integer_values 
  (a b c d : ℝ) 
  (h₁ : ∃ n₁ : ℤ, f a b c d (-1) = n₁) 
  (h₂ : ∃ n₂ : ℤ, f a b c d 0 = n₂)
  (h₃ : ∃ n₃ : ℤ, f a b c d 1 = n₃)
  (h₄ : ∃ n₄ : ℤ, f a b c d 2 = n₄) :
  ∀ x : ℤ, ∃ n : ℤ, f a b c d x = n :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_values_l3176_317690


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3176_317664

theorem unique_prime_triple : 
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    (∃ k : ℤ, (p^2 + 2*q : ℤ) = k * (q + r : ℤ)) ∧
    (∃ l : ℤ, (q^2 + 9*r : ℤ) = l * (r + p : ℤ)) ∧
    (∃ m : ℤ, (r^2 + 3*p : ℤ) = m * (p + q : ℤ)) ∧
    p = 2 ∧ q = 3 ∧ r = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3176_317664


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l3176_317667

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (A B : ℝ × ℝ) : Prop :=
  (A.1 - focus.1) * (B.2 - focus.2) = (B.1 - focus.1) * (A.2 - focus.2)

-- Define the condition that A and B are on the parabola
def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

-- Define the sum of x-coordinates condition
def sum_of_x_coordinates (A B : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 3

-- The main theorem
theorem parabola_intersection_length (A B : ℝ × ℝ) :
  line_through_focus A B →
  points_on_parabola A B →
  sum_of_x_coordinates A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l3176_317667


namespace NUMINAMATH_CALUDE_cube_roots_of_negative_one_l3176_317642

theorem cube_roots_of_negative_one :
  let z₁ : ℂ := -1
  let z₂ : ℂ := (1 + Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (1 - Complex.I * Real.sqrt 3) / 2
  (∀ z : ℂ, z^3 = -1 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cube_roots_of_negative_one_l3176_317642


namespace NUMINAMATH_CALUDE_caterpillars_left_tree_l3176_317665

/-- Proves that the number of caterpillars that left the tree is 8 --/
theorem caterpillars_left_tree (initial : ℕ) (hatched : ℕ) (final : ℕ) : 
  initial = 14 → hatched = 4 → final = 10 → initial + hatched - final = 8 := by
  sorry

end NUMINAMATH_CALUDE_caterpillars_left_tree_l3176_317665


namespace NUMINAMATH_CALUDE_g_monotone_decreasing_l3176_317698

/-- The function g(x) defined in terms of the parameter a -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- The derivative of g(x) with respect to x -/
def g' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 4 * (1 - a) * x - 3 * a

/-- Theorem stating the conditions for g(x) to be monotonically decreasing -/
theorem g_monotone_decreasing (a : ℝ) : 
  (∀ x : ℝ, x < a / 3 → g' a x ≤ 0) ↔ a ∈ Set.Iic (-1) ∪ {0} :=
sorry

end NUMINAMATH_CALUDE_g_monotone_decreasing_l3176_317698


namespace NUMINAMATH_CALUDE_square_sum_proof_l3176_317693

theorem square_sum_proof (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_proof_l3176_317693


namespace NUMINAMATH_CALUDE_min_value_theorem_l3176_317602

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 4) :
  (9 / a) + (16 / b) + (25 / c) ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3176_317602


namespace NUMINAMATH_CALUDE_max_notebooks_is_11_l3176_317648

/-- Represents the number of notebooks in a pack -/
inductive PackSize
  | Single
  | Pack4
  | Pack7

/-- Returns the number of notebooks for a given pack size -/
def notebooks (size : PackSize) : ℕ :=
  match size with
  | PackSize.Single => 1
  | PackSize.Pack4 => 4
  | PackSize.Pack7 => 7

/-- Returns the cost in dollars for a given pack size -/
def cost (size : PackSize) : ℕ :=
  match size with
  | PackSize.Single => 2
  | PackSize.Pack4 => 6
  | PackSize.Pack7 => 10

/-- Represents a purchase of notebook packs -/
structure Purchase where
  single : ℕ
  pack4 : ℕ
  pack7 : ℕ

/-- Calculates the total number of notebooks for a given purchase -/
def totalNotebooks (p : Purchase) : ℕ :=
  p.single * notebooks PackSize.Single +
  p.pack4 * notebooks PackSize.Pack4 +
  p.pack7 * notebooks PackSize.Pack7

/-- Calculates the total cost for a given purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * cost PackSize.Single +
  p.pack4 * cost PackSize.Pack4 +
  p.pack7 * cost PackSize.Pack7

/-- Represents the budget constraint -/
def budget : ℕ := 17

/-- Theorem: The maximum number of notebooks that can be purchased within the budget is 11 -/
theorem max_notebooks_is_11 :
  ∀ p : Purchase, totalCost p ≤ budget → totalNotebooks p ≤ 11 ∧
  ∃ p' : Purchase, totalCost p' ≤ budget ∧ totalNotebooks p' = 11 :=
sorry

end NUMINAMATH_CALUDE_max_notebooks_is_11_l3176_317648


namespace NUMINAMATH_CALUDE_football_match_end_time_l3176_317605

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- State the theorem
theorem football_match_end_time :
  let start_time : Time := { hour := 15, minute := 30 }
  let duration : Nat := 145
  let end_time : Time := addMinutes start_time duration
  end_time = { hour := 17, minute := 55 } := by
  sorry

end NUMINAMATH_CALUDE_football_match_end_time_l3176_317605


namespace NUMINAMATH_CALUDE_wanda_walking_distance_l3176_317622

/-- Represents the distance Wanda walks in miles -/
def distance_to_school : ℝ := 0.5

/-- Represents the number of round trips Wanda makes per day -/
def round_trips_per_day : ℕ := 2

/-- Represents the number of days Wanda walks to school per week -/
def school_days_per_week : ℕ := 5

/-- Represents the number of weeks we're considering -/
def weeks : ℕ := 4

/-- Theorem stating that Wanda walks 40 miles after 4 weeks -/
theorem wanda_walking_distance : 
  2 * distance_to_school * round_trips_per_day * school_days_per_week * weeks = 40 := by
  sorry


end NUMINAMATH_CALUDE_wanda_walking_distance_l3176_317622


namespace NUMINAMATH_CALUDE_min_value_f_and_g_inequality_l3176_317619

noncomputable section

variable (t : ℝ)

def f (x : ℝ) := Real.exp (2 * x) - 2 * t * x

def g (x : ℝ) := -x^2 + 2 * t * Real.exp x - 2 * t^2 + 1/2

theorem min_value_f_and_g_inequality :
  (t ≤ 1 → ∀ x ≥ 0, f t x ≥ 1) ∧
  (t > 1 → ∀ x ≥ 0, f t x ≥ t - t * Real.log t) ∧
  (t = 1 → ∀ x ≥ 0, g t x ≥ 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_min_value_f_and_g_inequality_l3176_317619


namespace NUMINAMATH_CALUDE_cubic_from_quadratic_roots_l3176_317626

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots r and s,
    this theorem states the form of the cubic equation
    with roots r^2 + br + a and s^2 + bs + a. -/
theorem cubic_from_quadratic_roots (a b c r s : ℝ) : 
  (a * r^2 + b * r + c = 0) →
  (a * s^2 + b * s + c = 0) →
  (r ≠ s) →
  ∃ (p q : ℝ), 
    (x^3 + (b^2 - a*b^2 + 4*a^3 - 2*a*c)/a^2 * x^2 + p*x + q = 0) ↔ 
    (x = r^2 + b*r + a ∨ x = s^2 + b*s + a ∨ x = -(r^2 + b*r + a + s^2 + b*s + a)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_from_quadratic_roots_l3176_317626


namespace NUMINAMATH_CALUDE_increase_average_by_transfer_l3176_317629

def group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def removeElement (l : List ℕ) (x : ℕ) : List ℕ :=
  l.filter (· ≠ x)

theorem increase_average_by_transfer :
  ∃ g ∈ group1,
    average (removeElement group1 g) > average group1 ∧
    average (g :: group2) > average group2 := by
  sorry

end NUMINAMATH_CALUDE_increase_average_by_transfer_l3176_317629


namespace NUMINAMATH_CALUDE_student_arrangement_count_l3176_317656

/-- Represents the number of students -/
def n : ℕ := 5

/-- Represents the condition that two specific students must be together -/
def must_be_together : Prop := True

/-- Represents the condition that two specific students cannot be together -/
def cannot_be_together : Prop := True

/-- The number of ways to arrange the students -/
def arrangement_count : ℕ := 24

/-- Theorem stating that the number of arrangements satisfying the conditions is 24 -/
theorem student_arrangement_count :
  must_be_together → cannot_be_together → arrangement_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l3176_317656


namespace NUMINAMATH_CALUDE_inequality_range_l3176_317673

theorem inequality_range (t : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → 
    (1/8 * (2*t - t^2) ≤ x^2 - 3*x + 2 ∧ x^2 - 3*x + 2 ≤ 3 - t^2)) ↔ 
  (t ∈ Set.Icc (-1) (1 - Real.sqrt 3)) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l3176_317673


namespace NUMINAMATH_CALUDE_ramsey_r33_l3176_317654

/-- Represents the relationship between two people -/
inductive Relationship
| Acquaintance
| Stranger

/-- A group of people -/
def People := Fin 6

/-- The relationship between each pair of people -/
def RelationshipMap := People → People → Relationship

/-- Checks if three people are mutual acquaintances -/
def areMutualAcquaintances (rel : RelationshipMap) (a b c : People) : Prop :=
  rel a b = Relationship.Acquaintance ∧
  rel a c = Relationship.Acquaintance ∧
  rel b c = Relationship.Acquaintance

/-- Checks if three people are mutual strangers -/
def areMutualStrangers (rel : RelationshipMap) (a b c : People) : Prop :=
  rel a b = Relationship.Stranger ∧
  rel a c = Relationship.Stranger ∧
  rel b c = Relationship.Stranger

/-- Main theorem: In a group of 6 people, there are either 3 mutual acquaintances or 3 mutual strangers -/
theorem ramsey_r33 (rel : RelationshipMap) :
  (∃ a b c : People, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ areMutualAcquaintances rel a b c) ∨
  (∃ a b c : People, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ areMutualStrangers rel a b c) :=
sorry

end NUMINAMATH_CALUDE_ramsey_r33_l3176_317654


namespace NUMINAMATH_CALUDE_polynomial_roots_coefficient_sum_l3176_317689

theorem polynomial_roots_coefficient_sum (p q r : ℝ) : 
  (∃ a b c : ℝ, 0 < a ∧ a < 2 ∧ 0 < b ∧ b < 2 ∧ 0 < c ∧ c < 2 ∧
    ∀ x : ℝ, x^3 + p*x^2 + q*x + r = (x - a) * (x - b) * (x - c)) →
  -2 < p + q + r ∧ p + q + r < 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_coefficient_sum_l3176_317689


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l3176_317612

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| = 3 - x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l3176_317612


namespace NUMINAMATH_CALUDE_arcade_spending_correct_l3176_317676

/-- Calculates the total money spent by John at the arcade --/
def arcade_spending (total_time : ℕ) (break_time : ℕ) (rate1 : ℚ) (interval1 : ℕ)
  (rate2 : ℚ) (interval2 : ℕ) (rate3 : ℚ) (interval3 : ℕ) (rate4 : ℚ) (interval4 : ℕ) : ℚ :=
  let play_time := total_time - break_time
  let hour1 := (60 / interval1) * rate1
  let hour2 := (60 / interval2) * rate2
  let hour3 := (60 / interval3) * rate3
  let hour4 := ((play_time - 180) / interval4) * rate4
  hour1 + hour2 + hour3 + hour4

theorem arcade_spending_correct :
  arcade_spending 275 50 0.5 4 0.75 5 1 3 1.25 7 = 42.75 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spending_correct_l3176_317676


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l3176_317625

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -867 [ZMOD 13] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l3176_317625


namespace NUMINAMATH_CALUDE_jeanne_initial_tickets_l3176_317628

/-- The cost of all three attractions in tickets -/
def total_cost : ℕ := 13

/-- The number of additional tickets Jeanne needs to buy -/
def additional_tickets : ℕ := 8

/-- Jeanne's initial number of tickets -/
def initial_tickets : ℕ := total_cost - additional_tickets

theorem jeanne_initial_tickets : initial_tickets = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeanne_initial_tickets_l3176_317628


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3176_317684

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → a ∈ Set.Ioi 3 ∪ Set.Iio (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3176_317684


namespace NUMINAMATH_CALUDE_student_number_problem_l3176_317624

theorem student_number_problem (x : ℝ) : 2 * x - 152 = 102 → x = 127 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3176_317624


namespace NUMINAMATH_CALUDE_special_function_property_l3176_317696

/-- A continuously differentiable function satisfying f'(t) > f(f(t)) for all t ∈ ℝ -/
structure SpecialFunction where
  f : ℝ → ℝ
  cont_diff : ContDiff ℝ 1 f
  property : ∀ t : ℝ, deriv f t > f (f t)

/-- The main theorem -/
theorem special_function_property (sf : SpecialFunction) :
  ∀ t : ℝ, t ≥ 0 → sf.f (sf.f (sf.f t)) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l3176_317696


namespace NUMINAMATH_CALUDE_midpoint_coordinates_l3176_317662

/-- The midpoint coordinates of the line segment cut by the parabola y^2 = 4x from the line y = x - 1 are (3, 2). -/
theorem midpoint_coordinates (x y : ℝ) : 
  y^2 = 4*x ∧ y = x - 1 → 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
    (x1^2 - 1)^2 = 4*x1 ∧ 
    (x2^2 - 1)^2 = 4*x2 ∧
    ((x1 + x2) / 2 = 3 ∧ ((x1 - 1) + (x2 - 1)) / 2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_coordinates_l3176_317662


namespace NUMINAMATH_CALUDE_duo_ball_playing_time_l3176_317609

theorem duo_ball_playing_time (num_children : ℕ) (total_time : ℕ) (players_per_game : ℕ) :
  num_children = 8 →
  total_time = 120 →
  players_per_game = 2 →
  (total_time * players_per_game) / num_children = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_duo_ball_playing_time_l3176_317609


namespace NUMINAMATH_CALUDE_alices_number_l3176_317682

theorem alices_number (x : ℝ) : 3 * (3 * x - 6) = 141 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_alices_number_l3176_317682


namespace NUMINAMATH_CALUDE_solve_equation_l3176_317685

theorem solve_equation (x : ℚ) : (3 * x + 5) / 5 = 17 → x = 80 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3176_317685


namespace NUMINAMATH_CALUDE_union_equality_implies_range_l3176_317680

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 1}
def M (a : ℝ) : Set ℝ := {a}

-- State the theorem
theorem union_equality_implies_range (a : ℝ) :
  P ∪ M a = P → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_range_l3176_317680


namespace NUMINAMATH_CALUDE_parabola_circle_tangent_l3176_317658

/-- The value of p for a parabola y^2 = 2px (p > 0) whose directrix is tangent to the circle (x-3)^2 + y^2 = 16 -/
theorem parabola_circle_tangent (p : ℝ) : 
  p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x) ∧ 
  (∃ (x y : ℝ), (x - 3)^2 + y^2 = 16) ∧
  (∃ (x : ℝ), x = -p/2 ∧ (x - 3)^2 + (2*p*x) = 16) →
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangent_l3176_317658


namespace NUMINAMATH_CALUDE_intersection_range_l3176_317666

/-- Given two curves C₁ and C₂, prove the range of m for which they intersect at exactly one point above the x-axis. -/
theorem intersection_range (a : ℝ) (h_a : a > 0) :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x^2 / a^2 + y^2 = 1 ∧ y^2 = 2*(x + m) ∧ y > 0) →
    (0 < a ∧ a < 1 → (m = (a^2 + 1) / 2 ∨ (-a < m ∧ m ≤ a))) ∧
    (a ≥ 1 → (-a < m ∧ m < a)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l3176_317666


namespace NUMINAMATH_CALUDE_ship_optimal_speed_and_cost_l3176_317646

/-- The optimal speed and cost for a ship's journey -/
theorem ship_optimal_speed_and_cost (distance : ℝ) (fuel_cost_coeff : ℝ) (fixed_cost : ℝ)
  (h_distance : distance = 100)
  (h_fuel_cost : fuel_cost_coeff = 0.005)
  (h_fixed_cost : fixed_cost = 80) :
  ∃ (optimal_speed : ℝ) (min_cost : ℝ),
    optimal_speed = 20 ∧
    min_cost = 600 ∧
    ∀ (v : ℝ), v > 0 →
      distance / v * (fuel_cost_coeff * v^3 + fixed_cost) ≥ min_cost :=
by sorry

end NUMINAMATH_CALUDE_ship_optimal_speed_and_cost_l3176_317646


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3176_317674

theorem complex_modulus_problem (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3176_317674


namespace NUMINAMATH_CALUDE_opponent_total_score_l3176_317687

def team_scores : List ℕ := [1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def one_run_losses : ℕ := 3
def two_run_losses : ℕ := 2

def remaining_games (total_games : ℕ) : ℕ :=
  total_games - one_run_losses - two_run_losses

theorem opponent_total_score :
  ∃ (one_loss_scores two_loss_scores three_times_scores : List ℕ),
    (one_loss_scores.length = one_run_losses) ∧
    (two_loss_scores.length = two_run_losses) ∧
    (three_times_scores.length = remaining_games team_scores.length) ∧
    (∀ (s : ℕ), s ∈ one_loss_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ two_loss_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ three_times_scores → s ∈ team_scores) ∧
    (∀ (s : ℕ), s ∈ one_loss_scores → ∃ (o : ℕ), o = s + 1) ∧
    (∀ (s : ℕ), s ∈ two_loss_scores → ∃ (o : ℕ), o = s + 2) ∧
    (∀ (s : ℕ), s ∈ three_times_scores → ∃ (o : ℕ), o = s / 3) ∧
    (one_loss_scores.sum + two_loss_scores.sum + three_times_scores.sum = 42) :=
by
  sorry

end NUMINAMATH_CALUDE_opponent_total_score_l3176_317687


namespace NUMINAMATH_CALUDE_expression_simplification_l3176_317694

theorem expression_simplification (p : ℤ) :
  ((7 * p + 2) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 40 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3176_317694


namespace NUMINAMATH_CALUDE_inequality_solution_l3176_317627

theorem inequality_solution (b : ℝ) (h : ∀ x : ℝ, x ∈ Set.Icc 0 1 → ∃ a : ℝ, x * |x - a| + b < 0) :
  ((-1 ≤ b ∧ b < 2 * Real.sqrt 2 - 3 →
    ∃ a : ℝ, a ∈ Set.Ioo (1 + b) (2 * Real.sqrt (-b))) ∧
   (b < -1 →
    ∃ a : ℝ, a ∈ Set.Ioo (1 + b) (1 - b))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3176_317627


namespace NUMINAMATH_CALUDE_sum_of_trapezoid_areas_l3176_317623

/-- Represents a trapezoid with four side lengths --/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the area of a trapezoid given its side lengths --/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

/-- Generates all valid trapezoid configurations with given side lengths --/
def validConfigurations (sides : List ℝ) : List Trapezoid := sorry

/-- Theorem: The sum of areas of all valid trapezoids with side lengths 4, 6, 8, and 10 
    is equal to a specific value --/
theorem sum_of_trapezoid_areas :
  let sides := [4, 6, 8, 10]
  let validTraps := validConfigurations sides
  let areas := validTraps.map trapezoidArea
  ∃ (total : ℝ), areas.sum = total := by sorry

end NUMINAMATH_CALUDE_sum_of_trapezoid_areas_l3176_317623


namespace NUMINAMATH_CALUDE_tube_length_doubles_pressure_l3176_317643

/-- The length of a vertical tube required to double the pressure at the bottom of a water-filled barrel -/
theorem tube_length_doubles_pressure 
  (h₁ : ℝ) -- Initial height of water in the barrel
  (m : ℝ) -- Mass of water in the barrel
  (a : ℝ) -- Cross-sectional area of the tube
  (ρ : ℝ) -- Density of water
  (g : ℝ) -- Acceleration due to gravity
  (h₁_val : h₁ = 1.5) -- Given height of the barrel
  (m_val : m = 1000) -- Given mass of water
  (a_val : a = 1e-4) -- Given cross-sectional area (1 cm² = 1e-4 m²)
  (ρ_val : ρ = 1000) -- Given density of water
  : ∃ (h₂ : ℝ), h₂ = h₁ ∧ ρ * g * (h₁ + h₂) = 2 * (ρ * g * h₁) :=
by sorry

end NUMINAMATH_CALUDE_tube_length_doubles_pressure_l3176_317643


namespace NUMINAMATH_CALUDE_rhombus_area_l3176_317608

/-- The area of a rhombus with side length 4 cm and one angle of 45° is 8√2 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π/4) :
  let area := s * s * Real.sin θ
  area = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l3176_317608


namespace NUMINAMATH_CALUDE_tangent_function_l3176_317631

/-- Given a function f(x) = ax / (x^2 + b), prove that if f(1) = 2 and f'(1) = 0, 
    then f(x) = 4x / (x^2 + 1) -/
theorem tangent_function (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x / (x^2 + b)
  (f 1 = 2) → (deriv f 1 = 0) → ∀ x, f x = 4 * x / (x^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_tangent_function_l3176_317631


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l3176_317645

def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem f_decreasing_on_interval : 
  ∀ x ∈ Set.Ioo (-2 : ℝ) 1, (deriv f) x < 0 := by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l3176_317645


namespace NUMINAMATH_CALUDE_line_intersection_b_range_l3176_317618

theorem line_intersection_b_range (b : ℝ) (h1 : b ≠ 0) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ 2 * x + b = 3) →
  -3 ≤ b ∧ b ≤ 3 ∧ b ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_b_range_l3176_317618


namespace NUMINAMATH_CALUDE_intersection_sum_zero_l3176_317657

theorem intersection_sum_zero (x₁ x₂ : ℝ) :
  (x₁^2 + 9^2 = 169) →
  (x₂^2 + 9^2 = 169) →
  x₁ ≠ x₂ →
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_zero_l3176_317657


namespace NUMINAMATH_CALUDE_transformations_map_correctly_l3176_317637

-- Define points in 2D space
def C : ℝ × ℝ := (3, -2)
def D : ℝ × ℝ := (4, -5)
def C' : ℝ × ℝ := (-3, 2)
def D' : ℝ × ℝ := (-4, 5)

-- Define translation
def translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 6, p.2 + 4)

-- Define 180° clockwise rotation
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem stating that both transformations map C to C' and D to D'
theorem transformations_map_correctly :
  (translate C = C' ∧ translate D = D') ∧
  (rotate180 C = C' ∧ rotate180 D = D') :=
by sorry

end NUMINAMATH_CALUDE_transformations_map_correctly_l3176_317637


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l3176_317604

theorem divisible_by_eleven (n : ℕ) : ∃ k : ℤ, 5^(2*n) + 3^(n+2) + 3^n = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l3176_317604


namespace NUMINAMATH_CALUDE_jens_height_l3176_317603

theorem jens_height (original_height : ℝ) (bakis_growth_rate : ℝ) (jens_growth_ratio : ℝ) (bakis_final_height : ℝ) :
  original_height > 0 ∧ 
  bakis_growth_rate = 0.25 ∧
  jens_growth_ratio = 2/3 ∧
  bakis_final_height = 75 ∧
  bakis_final_height = original_height * (1 + bakis_growth_rate) →
  original_height + jens_growth_ratio * (bakis_final_height - original_height) = 70 :=
by
  sorry

#check jens_height

end NUMINAMATH_CALUDE_jens_height_l3176_317603


namespace NUMINAMATH_CALUDE_quadratic_properties_l3176_317640

-- Define the quadratic function
def quadratic (a b t : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + t - 1

theorem quadratic_properties :
  ∀ (a b t : ℝ), t < 0 →
  -- Part 1
  (quadratic a b (-2) 1 = -4 ∧ quadratic a b (-2) (-1) = 0) → (a = 1 ∧ b = -2) ∧
  -- Part 2
  (2 * a - b = 1) → 
    ∃ (k p : ℝ), k ≠ 0 ∧ 
      ∀ (x : ℝ), (quadratic a b (-2) x = k * x + p) → 
        ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic a b (-2) x1 = k * x1 + p ∧ quadratic a b (-2) x2 = k * x2 + p ∧
  -- Part 3
  ∀ (m n : ℝ), m > 0 ∧ n > 0 →
    quadratic a b t (-1) = t ∧ quadratic a b t m = t - n ∧
    ((1/2) * n - 2 * t = (1/2) * (m + 1) * (quadratic a b t m - quadratic a b t (-1))) →
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ m → quadratic a b t x ≤ quadratic a b t (-1)) →
    ((0 < a ∧ a ≤ 1/3) ∨ (-1 ≤ a ∧ a < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3176_317640


namespace NUMINAMATH_CALUDE_M_equals_interval_inequality_holds_l3176_317670

def f (x : ℝ) := |x + 2| + |x - 2|

def M : Set ℝ := {x | f x ≤ 6}

theorem M_equals_interval : M = Set.Icc (-3) 3 := by sorry

theorem inequality_holds (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  Real.sqrt 3 * |a + b| ≤ |a * b + 3| := by sorry

end NUMINAMATH_CALUDE_M_equals_interval_inequality_holds_l3176_317670


namespace NUMINAMATH_CALUDE_test_score_problem_l3176_317677

/-- Proves that given a test with 25 problems, a scoring system of 4 points for each correct answer
    and -1 point for each wrong answer, and a total score of 85, the number of wrong answers is 3. -/
theorem test_score_problem (total_problems : Nat) (right_points : Int) (wrong_points : Int) (total_score : Int)
    (h1 : total_problems = 25)
    (h2 : right_points = 4)
    (h3 : wrong_points = -1)
    (h4 : total_score = 85) :
    ∃ (right : Nat) (wrong : Nat),
      right + wrong = total_problems ∧
      right_points * right + wrong_points * wrong = total_score ∧
      wrong = 3 :=
by sorry

end NUMINAMATH_CALUDE_test_score_problem_l3176_317677


namespace NUMINAMATH_CALUDE_sin_cos_identity_l3176_317683

theorem sin_cos_identity (α c d : ℝ) (h : c > 0) (k : d > 0) 
  (eq : (Real.sin α)^6 / c + (Real.cos α)^6 / d = 1 / (c + d)) :
  (Real.sin α)^12 / c^5 + (Real.cos α)^12 / d^5 = 1 / (c + d)^5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l3176_317683


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l3176_317644

/-- Represents the number of adults a can of soup can feed -/
def adults_per_can : ℕ := 4

/-- Represents the number of children a can of soup can feed -/
def children_per_can : ℕ := 7

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 10

/-- Represents the number of children fed -/
def children_fed : ℕ := 35

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed_with_remaining_soup : ℕ := 
  adults_per_can * (total_cans - (children_fed / children_per_can))

theorem remaining_soup_feeds_twenty_adults : 
  adults_fed_with_remaining_soup = 20 := by sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_twenty_adults_l3176_317644


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l3176_317672

theorem smallest_number_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 4 = 3) ∧
  (n % 5 = 4) ∧
  (n % 6 = 5) ∧
  (n % 7 = 6) ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → m ≥ n) ∧
  n = 419 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l3176_317672


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3176_317695

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 30 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l3176_317695


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l3176_317660

theorem purely_imaginary_z (α : ℝ) : 
  let z : ℂ := Complex.mk (Real.sin α) (-(1 - Real.cos α))
  z.re = 0 → ∃ k : ℤ, α = (2 * k + 1) * Real.pi := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l3176_317660


namespace NUMINAMATH_CALUDE_ellipse_intersection_parallel_line_l3176_317692

-- Define the ellipse C
def ellipse_C (b : ℝ) (x y : ℝ) : Prop :=
  b > 0 ∧ x^2 / (5 * b^2) + y^2 / b^2 = 1

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a line on the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the focus of the ellipse
def focus : Point := ⟨2, 0⟩

-- Define point E
def point_E : Point := ⟨3, 0⟩

-- Define the line x = 5
def line_x_5 : Line := ⟨1, 0, -5⟩

-- Define the property of line l passing through (1,0) and not coinciding with x-axis
def line_l_property (l : Line) : Prop :=
  l.a * 1 + l.b * 0 + l.c = 0 ∧ l.b ≠ 0

-- Define the intersection of a line and the ellipse
def intersect_line_ellipse (l : Line) (b : ℝ) : Prop :=
  ∃ (M N : Point), 
    line_l_property l ∧
    ellipse_C b M.x M.y ∧ 
    ellipse_C b N.x N.y ∧
    l.a * M.x + l.b * M.y + l.c = 0 ∧
    l.a * N.x + l.b * N.y + l.c = 0

-- Define the intersection of two lines
def intersect_lines (l1 l2 : Line) (P : Point) : Prop :=
  l1.a * P.x + l1.b * P.y + l1.c = 0 ∧
  l2.a * P.x + l2.b * P.y + l2.c = 0

-- Define parallel lines
def parallel_lines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

-- The main theorem
theorem ellipse_intersection_parallel_line (b : ℝ) (l : Line) (M N F : Point) :
  ellipse_C b focus.x focus.y →
  intersect_line_ellipse l b →
  intersect_lines (Line.mk (M.y - point_E.y) (point_E.x - M.x) (M.x * point_E.y - M.y * point_E.x)) line_x_5 F →
  parallel_lines (Line.mk (F.y - N.y) (N.x - F.x) (F.x * N.y - F.y * N.x)) (Line.mk 0 1 0) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_parallel_line_l3176_317692


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3176_317617

theorem necessary_not_sufficient : 
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (3 - x) > 0)) ∧
  (∀ x : ℝ, x * (3 - x) > 0 → |x - 1| < 2) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3176_317617


namespace NUMINAMATH_CALUDE_election_vote_ratio_l3176_317633

theorem election_vote_ratio (marcy_votes barry_votes joey_votes : ℕ) : 
  marcy_votes = 66 →
  marcy_votes = 3 * barry_votes →
  joey_votes = 8 →
  barry_votes ≠ 0 →
  barry_votes / (joey_votes + 3) = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_election_vote_ratio_l3176_317633


namespace NUMINAMATH_CALUDE_binomial_320_320_l3176_317634

theorem binomial_320_320 : Nat.choose 320 320 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_320_320_l3176_317634


namespace NUMINAMATH_CALUDE_cases_in_1975_l3176_317630

/-- Calculates the number of disease cases in a given year, assuming a linear decrease --/
def casesInYear (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let yearlyDecrease := totalDecrease / totalYears
  let targetYearDiff := targetYear - initialYear
  initialCases - (yearlyDecrease * targetYearDiff)

/-- Theorem stating that given the conditions, the number of cases in 1975 would be 300,150 --/
theorem cases_in_1975 :
  casesInYear 1950 600000 2000 300 1975 = 300150 := by
  sorry

end NUMINAMATH_CALUDE_cases_in_1975_l3176_317630


namespace NUMINAMATH_CALUDE_kevin_six_hops_l3176_317649

def kevin_hop (n : ℕ) : ℚ :=
  2 * (1 - (3/4)^n)

theorem kevin_six_hops :
  kevin_hop 6 = 3367 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_kevin_six_hops_l3176_317649


namespace NUMINAMATH_CALUDE_multiple_birth_statistics_l3176_317697

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets : ℕ) : 
  total_babies = 1000 →
  triplets = 4 * quadruplets →
  twins = 3 * triplets →
  2 * twins + 3 * triplets + 4 * quadruplets = total_babies →
  4 * quadruplets = 100 := by
  sorry

end NUMINAMATH_CALUDE_multiple_birth_statistics_l3176_317697


namespace NUMINAMATH_CALUDE_quadratic_binomial_square_l3176_317650

theorem quadratic_binomial_square (b r : ℚ) : 
  (∀ x, b * x^2 + 20 * x + 16 = (r * x - 4)^2) → b = 25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_binomial_square_l3176_317650


namespace NUMINAMATH_CALUDE_rachel_treasures_l3176_317615

theorem rachel_treasures (points_per_treasure : ℕ) (second_level_treasures : ℕ) (total_score : ℕ) :
  points_per_treasure = 9 →
  second_level_treasures = 2 →
  total_score = 63 →
  ∃ (first_level_treasures : ℕ),
    first_level_treasures * points_per_treasure + second_level_treasures * points_per_treasure = total_score ∧
    first_level_treasures = 5 :=
by
  sorry

#check rachel_treasures

end NUMINAMATH_CALUDE_rachel_treasures_l3176_317615


namespace NUMINAMATH_CALUDE_similar_triangles_leg_l3176_317688

/-- Two similar right triangles with legs 12 and 9 in the first triangle,
    and y and 6 in the second triangle, have y = 8 -/
theorem similar_triangles_leg (y : ℝ) : 
  (12 : ℝ) / y = 9 / 6 → y = 8 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_l3176_317688
