import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_breadth_l506_50630

theorem rectangle_breadth (square_area : ℝ) (rectangle_area : ℝ) :
  square_area = 3600 →
  rectangle_area = 240 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2 / 5) * circle_radius
  rectangle_area = rectangle_length * (rectangle_area / rectangle_length) →
  rectangle_area / rectangle_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_breadth_l506_50630


namespace NUMINAMATH_CALUDE_coordinate_sum_of_A_l506_50609

-- Define the points
def B : ℝ × ℝ := (2, 8)
def C : ℝ × ℝ := (0, 2)

-- Define the theorem
theorem coordinate_sum_of_A (A : ℝ × ℝ) :
  (A.1 - C.1) / (B.1 - C.1) = 1/3 ∧
  (A.2 - C.2) / (B.2 - C.2) = 1/3 →
  A.1 + A.2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_A_l506_50609


namespace NUMINAMATH_CALUDE_smallest_integer_l506_50647

theorem smallest_integer (a b : ℕ) (ha : a = 75) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 45) :
  ∃ (m : ℕ), m ≥ b ∧ m = 135 ∧ Nat.lcm a m / Nat.gcd a m = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l506_50647


namespace NUMINAMATH_CALUDE_flower_theorem_l506_50638

def flower_problem (alissa_flowers melissa_flowers flowers_left : ℕ) : Prop :=
  alissa_flowers + melissa_flowers - flowers_left = 18

theorem flower_theorem :
  flower_problem 16 16 14 := by
  sorry

end NUMINAMATH_CALUDE_flower_theorem_l506_50638


namespace NUMINAMATH_CALUDE_product_of_parts_of_complex_square_l506_50639

theorem product_of_parts_of_complex_square : ∃ (a b : ℝ), (Complex.mk 1 2)^2 = Complex.mk a b ∧ a * b = -12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_parts_of_complex_square_l506_50639


namespace NUMINAMATH_CALUDE_tangent_line_equation_l506_50670

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * Real.log x - 1/2

def a : ℝ := 2

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (x + y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l506_50670


namespace NUMINAMATH_CALUDE_equidistant_after_1_min_equidistant_after_5_min_speed_ratio_l506_50608

-- Define the speeds of A and B
def v_A : ℝ := sorry
def v_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := -800

-- Define the equidistant condition after 1 minute
theorem equidistant_after_1_min : v_A = |initial_B_position + v_B| := sorry

-- Define the equidistant condition after 5 minutes
theorem equidistant_after_5_min : 5 * v_A = |initial_B_position + 5 * v_B| := sorry

-- Theorem to prove
theorem speed_ratio : v_A / v_B = 1 / 9 := sorry

end NUMINAMATH_CALUDE_equidistant_after_1_min_equidistant_after_5_min_speed_ratio_l506_50608


namespace NUMINAMATH_CALUDE_farm_area_l506_50607

/-- Proves that a rectangular farm with given conditions has an area of 1200 square meters -/
theorem farm_area (short_side : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) : 
  short_side = 30 →
  cost_per_meter = 14 →
  total_cost = 1680 →
  ∃ (long_side : ℝ),
    long_side > 0 ∧
    cost_per_meter * (long_side + short_side + Real.sqrt (long_side^2 + short_side^2)) = total_cost ∧
    long_side * short_side = 1200 :=
by
  sorry

#check farm_area

end NUMINAMATH_CALUDE_farm_area_l506_50607


namespace NUMINAMATH_CALUDE_total_earnings_before_car_purchase_l506_50624

def monthly_income : ℕ := 4000
def monthly_savings : ℕ := 500
def car_cost : ℕ := 45000

theorem total_earnings_before_car_purchase :
  (car_cost / monthly_savings) * monthly_income = 360000 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_before_car_purchase_l506_50624


namespace NUMINAMATH_CALUDE_cook_sane_cheshire_cat_insane_l506_50600

/-- Represents the sanity status of an individual -/
inductive Sanity
| Sane
| Insane

/-- Represents the characters in the problem -/
inductive Character
| Cook
| CheshireCat

/-- The cook's assertion about the sanity of the characters -/
def cooksAssertion (sanityStatus : Character → Sanity) : Prop :=
  sanityStatus Character.Cook = Sanity.Insane ∨ sanityStatus Character.CheshireCat = Sanity.Insane

/-- The main theorem to prove -/
theorem cook_sane_cheshire_cat_insane :
  ∃ (sanityStatus : Character → Sanity),
    cooksAssertion sanityStatus ∧
    sanityStatus Character.Cook = Sanity.Sane ∧
    sanityStatus Character.CheshireCat = Sanity.Insane :=
sorry

end NUMINAMATH_CALUDE_cook_sane_cheshire_cat_insane_l506_50600


namespace NUMINAMATH_CALUDE_perfect_square_characterization_l506_50678

theorem perfect_square_characterization (A : ℕ+) :
  (∃ (d : ℕ+), A = d ^ 2) ↔
  (∀ (n : ℕ+), ∃ (j : ℕ+), j ≤ n ∧ (n ∣ ((A + j) ^ 2 - A))) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_characterization_l506_50678


namespace NUMINAMATH_CALUDE_parallel_lines_reasoning_is_deductive_l506_50637

-- Define the types of reasoning
inductive ReasoningType
  | Deductive
  | Analogical

-- Define the characteristics of different types of reasoning
def isDeductive (r : ReasoningType) : Prop :=
  r = ReasoningType.Deductive

def isGeneralToSpecific (r : ReasoningType) : Prop :=
  r = ReasoningType.Deductive

-- Define the geometric concept
def sameSideInteriorAngles (a b : ℝ) : Prop :=
  a + b = 180

-- Theorem statement
theorem parallel_lines_reasoning_is_deductive :
  ∀ (A B : ℝ) (r : ReasoningType),
    sameSideInteriorAngles A B →
    isGeneralToSpecific r →
    isDeductive r :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_reasoning_is_deductive_l506_50637


namespace NUMINAMATH_CALUDE_expression_equals_two_l506_50633

theorem expression_equals_two (x : ℝ) (h : x ≠ -1) :
  ((x - 1) / (x + 1) + 1) / (x / (x + 1)) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_equals_two_l506_50633


namespace NUMINAMATH_CALUDE_white_balls_count_prob_after_addition_l506_50657

/-- The total number of balls in the box -/
def total_balls : ℕ := 40

/-- The probability of picking a white ball -/
def prob_white : ℚ := 1/10 * 6

/-- The number of white balls in the box -/
def white_balls : ℕ := 24

/-- The number of additional balls added -/
def additional_balls : ℕ := 10

/-- Theorem stating the relationship between the number of white balls and the probability -/
theorem white_balls_count : white_balls = total_balls * prob_white := by sorry

/-- Theorem proving that adding 10 balls with 1 white results in 50% probability -/
theorem prob_after_addition : 
  (white_balls + 1) / (total_balls + additional_balls) = 1/2 := by sorry

end NUMINAMATH_CALUDE_white_balls_count_prob_after_addition_l506_50657


namespace NUMINAMATH_CALUDE_train_speed_l506_50680

/-- Proves that a train of given length crossing a bridge of given length in a given time travels at a specific speed. -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  bridge_length = 255 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l506_50680


namespace NUMINAMATH_CALUDE_lawnmower_value_drop_l506_50676

/-- Calculates the final value of a lawnmower after three successive value drops -/
theorem lawnmower_value_drop (initial_value : ℝ) (drop1 drop2 drop3 : ℝ) :
  initial_value = 100 →
  drop1 = 0.25 →
  drop2 = 0.20 →
  drop3 = 0.15 →
  initial_value * (1 - drop1) * (1 - drop2) * (1 - drop3) = 51 := by
  sorry

end NUMINAMATH_CALUDE_lawnmower_value_drop_l506_50676


namespace NUMINAMATH_CALUDE_polynomial_simplification_l506_50661

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 3 * q^3 + 7 * q - 8) + (5 - 2 * q^3 + 9 * q^2 - 4 * q) =
  4 * q^4 - 5 * q^3 + 9 * q^2 + 3 * q - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l506_50661


namespace NUMINAMATH_CALUDE_positive_numbers_relation_l506_50627

theorem positive_numbers_relation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 / b = 5) (h2 : b^2 / c = 3) (h3 : c^2 / a = 7) : a = 15 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_relation_l506_50627


namespace NUMINAMATH_CALUDE_change_in_average_l506_50612

def scores : List ℝ := [89, 85, 91, 87, 82]

theorem change_in_average (scores : List ℝ) : 
  scores = [89, 85, 91, 87, 82] →
  (scores.sum / scores.length) - ((scores.take 4).sum / 4) = -1.2 := by
  sorry

end NUMINAMATH_CALUDE_change_in_average_l506_50612


namespace NUMINAMATH_CALUDE_exponent_division_l506_50671

theorem exponent_division (a : ℝ) : a^5 / a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l506_50671


namespace NUMINAMATH_CALUDE_average_of_four_digits_l506_50622

theorem average_of_four_digits 
  (total_digits : Nat)
  (total_average : ℚ)
  (five_digits : Nat)
  (five_average : ℚ)
  (h1 : total_digits = 9)
  (h2 : total_average = 18)
  (h3 : five_digits = 5)
  (h4 : five_average = 26)
  : (total_digits * total_average - five_digits * five_average) / (total_digits - five_digits) = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_digits_l506_50622


namespace NUMINAMATH_CALUDE_mean_median_difference_l506_50651

/-- Represents the distribution of scores in a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_60_percent : ℚ
  score_75_percent : ℚ
  score_82_percent : ℚ
  score_88_percent : ℚ
  score_92_percent : ℚ

/-- Calculates the mean score given a score distribution -/
def mean_score (d : ScoreDistribution) : ℚ :=
  (60 * d.score_60_percent + 75 * d.score_75_percent + 82 * d.score_82_percent +
   88 * d.score_88_percent + 92 * d.score_92_percent) / 1

/-- Calculates the median score given a score distribution -/
def median_score (d : ScoreDistribution) : ℚ := 82

/-- Theorem stating the difference between mean and median scores -/
theorem mean_median_difference (d : ScoreDistribution) 
  (h1 : d.total_students = 30)
  (h2 : d.score_60_percent = 15/100)
  (h3 : d.score_75_percent = 20/100)
  (h4 : d.score_82_percent = 25/100)
  (h5 : d.score_88_percent = 30/100)
  (h6 : d.score_92_percent = 10/100) :
  mean_score d - median_score d = 47/100 := by
  sorry

end NUMINAMATH_CALUDE_mean_median_difference_l506_50651


namespace NUMINAMATH_CALUDE_positive_number_inequality_l506_50699

theorem positive_number_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧
  0 < (x - y) / (1 + x * y) ∧ (x - y) / (1 + x * y) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_positive_number_inequality_l506_50699


namespace NUMINAMATH_CALUDE_song_game_theorem_l506_50620

/-- Represents the "Guess the Song Title" game -/
structure SongGame where
  /-- Probability of passing each level -/
  pass_prob : Fin 3 → ℚ
  /-- Probability of continuing to next level -/
  continue_prob : ℚ
  /-- Reward for passing each level -/
  reward : Fin 3 → ℕ

/-- The specific game instance as described in the problem -/
def game : SongGame :=
  { pass_prob := λ i => [3/4, 2/3, 1/2].get i
    continue_prob := 1/2
    reward := λ i => [1000, 2000, 3000].get i }

/-- Probability of passing first level but receiving zero reward -/
def prob_pass_first_zero_reward (g : SongGame) : ℚ :=
  g.pass_prob 0 * g.continue_prob * (1 - g.pass_prob 1) +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * g.continue_prob * (1 - g.pass_prob 2)

/-- Expected value of total reward -/
def expected_reward (g : SongGame) : ℚ :=
  g.pass_prob 0 * (1 - g.continue_prob) * g.reward 0 +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * (1 - g.continue_prob) * (g.reward 0 + g.reward 1) +
  g.pass_prob 0 * g.continue_prob * g.pass_prob 1 * g.continue_prob * g.pass_prob 2 * (g.reward 0 + g.reward 1 + g.reward 2)

theorem song_game_theorem (g : SongGame) :
  prob_pass_first_zero_reward g = 3/16 ∧ expected_reward g = 1125 :=
sorry

end NUMINAMATH_CALUDE_song_game_theorem_l506_50620


namespace NUMINAMATH_CALUDE_kite_smallest_angle_l506_50604

/-- Represents the angles of a kite in degrees -/
structure KiteAngles where
  a : ℝ  -- smallest angle
  d : ℝ  -- common difference

/-- Conditions for a valid kite with angles in arithmetic sequence -/
def is_valid_kite (k : KiteAngles) : Prop :=
  k.a > 0 ∧ 
  k.a + k.d > 0 ∧ 
  k.a + 2*k.d > 0 ∧ 
  k.a + 3*k.d > 0 ∧
  k.a + (k.a + 3*k.d) = 180 ∧  -- opposite angles are supplementary
  k.a + 3*k.d = 150  -- largest angle is 150°

theorem kite_smallest_angle (k : KiteAngles) (h : is_valid_kite k) : k.a = 15 := by
  sorry

end NUMINAMATH_CALUDE_kite_smallest_angle_l506_50604


namespace NUMINAMATH_CALUDE_agent_percentage_l506_50619

def total_copies : ℕ := 1000000
def earnings_per_copy : ℚ := 2
def steve_kept_earnings : ℚ := 1620000

theorem agent_percentage : 
  let total_earnings := total_copies * earnings_per_copy
  let agent_earnings := total_earnings - steve_kept_earnings
  (agent_earnings / total_earnings) * 100 = 19 := by sorry

end NUMINAMATH_CALUDE_agent_percentage_l506_50619


namespace NUMINAMATH_CALUDE_inequality_solution_l506_50674

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l506_50674


namespace NUMINAMATH_CALUDE_candles_from_leftovers_l506_50632

/-- Represents the number of candles of a certain size --/
structure CandleSet where
  count : ℕ
  size : ℚ

/-- Calculates the total wax from a set of candles --/
def waxFrom (cs : CandleSet) (leftoverRatio : ℚ) : ℚ :=
  cs.count * cs.size * leftoverRatio

/-- The main theorem --/
theorem candles_from_leftovers 
  (leftoverRatio : ℚ)
  (bigCandles smallCandles tinyCandles : CandleSet)
  (newCandleSize : ℚ)
  (h_leftover : leftoverRatio = 1/10)
  (h_big : bigCandles = ⟨5, 20⟩)
  (h_small : smallCandles = ⟨5, 5⟩)
  (h_tiny : tinyCandles = ⟨25, 1⟩)
  (h_new : newCandleSize = 5) :
  (waxFrom bigCandles leftoverRatio + 
   waxFrom smallCandles leftoverRatio + 
   waxFrom tinyCandles leftoverRatio) / newCandleSize = 3 := by
  sorry

end NUMINAMATH_CALUDE_candles_from_leftovers_l506_50632


namespace NUMINAMATH_CALUDE_meetings_percentage_theorem_l506_50694

/-- Calculates the percentage of a workday spent in meetings -/
def percentage_in_meetings (workday_hours : ℕ) (first_meeting_minutes : ℕ) (second_meeting_multiplier : ℕ) : ℚ :=
  let workday_minutes : ℕ := workday_hours * 60
  let second_meeting_minutes : ℕ := first_meeting_minutes * second_meeting_multiplier
  let total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes
  (total_meeting_minutes : ℚ) / (workday_minutes : ℚ) * 100

theorem meetings_percentage_theorem (workday_hours : ℕ) (first_meeting_minutes : ℕ) (second_meeting_multiplier : ℕ)
  (h1 : workday_hours = 10)
  (h2 : first_meeting_minutes = 60)
  (h3 : second_meeting_multiplier = 3) :
  percentage_in_meetings workday_hours first_meeting_minutes second_meeting_multiplier = 40 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_theorem_l506_50694


namespace NUMINAMATH_CALUDE_intersection_of_lines_l506_50625

/-- Given four points in 3D space, this theorem states that the intersection
    of the lines formed by these points is at a specific coordinate. -/
theorem intersection_of_lines (P Q R S : ℝ × ℝ × ℝ) : 
  P = (4, -8, 8) →
  Q = (14, -18, 14) →
  R = (1, 2, -7) →
  S = (3, -6, 9) →
  ∃ t s : ℝ, 
    (4 + 10*t, -8 - 10*t, 8 + 6*t) = (1 + 2*s, 2 - 8*s, -7 + 16*s) ∧
    (4 + 10*t, -8 - 10*t, 8 + 6*t) = (14/3, -22/3, 38/3) :=
by sorry


end NUMINAMATH_CALUDE_intersection_of_lines_l506_50625


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l506_50696

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ (∀ y : ℝ, y^2 + 2*y + k = 0 → y = x)) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l506_50696


namespace NUMINAMATH_CALUDE_salesman_pear_sales_l506_50644

theorem salesman_pear_sales (morning_sales afternoon_sales total_sales : ℕ) :
  afternoon_sales = 2 * morning_sales →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 510 →
  afternoon_sales = 340 :=
by
  sorry

end NUMINAMATH_CALUDE_salesman_pear_sales_l506_50644


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l506_50664

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_problem 
  (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_nonzero : ∀ n : ℕ, a n ≠ 0)
  (h_condition : a 6 - (a 7)^2 + a 8 = 0)
  (h_geometric : geometric_sequence b)
  (h_equal : b 7 = a 7) :
  b 3 * b 8 * b 10 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_problem_l506_50664


namespace NUMINAMATH_CALUDE_intersection_complement_eq_interval_l506_50640

open Set

/-- Given sets A and B, prove that their intersection with the complement of B is [1, 3) -/
theorem intersection_complement_eq_interval :
  let A : Set ℝ := {x | x - 1 ≥ 0}
  let B : Set ℝ := {x | 3 / x ≤ 1}
  A ∩ (univ \ B) = Icc 1 3 ∩ Iio 3 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_interval_l506_50640


namespace NUMINAMATH_CALUDE_adjacent_angle_measure_l506_50668

-- Define the angle type
def Angle : Type := ℝ

-- Define parallel lines
def ParallelLines (m n : Line) : Prop := sorry

-- Define a transversal line
def Transversal (p m n : Line) : Prop := sorry

-- Define the measure of an angle
def AngleMeasure (θ : Angle) : ℝ := sorry

-- Define supplementary angles
def Supplementary (θ₁ θ₂ : Angle) : Prop :=
  AngleMeasure θ₁ + AngleMeasure θ₂ = 180

-- Theorem statement
theorem adjacent_angle_measure
  (m n p : Line)
  (θ₁ θ₂ : Angle)
  (h_parallel : ParallelLines m n)
  (h_transversal : Transversal p m n)
  (h_internal : AngleMeasure θ₁ = 70)
  (h_supplementary : Supplementary θ₁ θ₂) :
  AngleMeasure θ₂ = 110 :=
sorry

end NUMINAMATH_CALUDE_adjacent_angle_measure_l506_50668


namespace NUMINAMATH_CALUDE_probability_sum_thirty_l506_50643

/-- Die A is a 30-faced die numbered 1-25 and 27-31 -/
def DieA : Finset ℕ := Finset.filter (fun n => n ≠ 26) (Finset.range 32 \ Finset.range 1)

/-- Die B is a 30-faced die numbered 1-20 and 26-31 -/
def DieB : Finset ℕ := (Finset.range 21 \ Finset.range 1) ∪ (Finset.range 32 \ Finset.range 26)

/-- The set of all possible outcomes when rolling both dice -/
def AllOutcomes : Finset (ℕ × ℕ) := DieA.product DieB

/-- The set of outcomes where the sum of the rolled numbers is 30 -/
def SumThirty : Finset (ℕ × ℕ) := AllOutcomes.filter (fun p => p.1 + p.2 = 30)

/-- The probability of rolling a sum of 30 with the given dice -/
def ProbabilitySumThirty : ℚ := (SumThirty.card : ℚ) / (AllOutcomes.card : ℚ)

theorem probability_sum_thirty : ProbabilitySumThirty = 59 / 900 := by sorry

end NUMINAMATH_CALUDE_probability_sum_thirty_l506_50643


namespace NUMINAMATH_CALUDE_decimal_to_percentage_example_l506_50682

/-- Converts a decimal fraction to a percentage -/
def decimal_to_percentage (d : ℝ) : ℝ := d * 100

/-- The decimal fraction we're working with -/
def given_decimal : ℝ := 0.01

/-- Theorem stating that converting 0.01 to a percentage results in 1 -/
theorem decimal_to_percentage_example :
  decimal_to_percentage given_decimal = 1 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_example_l506_50682


namespace NUMINAMATH_CALUDE_unique_number_property_l506_50641

theorem unique_number_property : ∃! x : ℝ, 3 * x = x + 18 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l506_50641


namespace NUMINAMATH_CALUDE_circle_radius_is_one_l506_50677

/-- The radius of the circle with equation 16x^2 + 32x + 16y^2 - 48y + 68 = 0 is 1 -/
theorem circle_radius_is_one :
  ∃ (h k r : ℝ), r = 1 ∧
  ∀ (x y : ℝ), 16*x^2 + 32*x + 16*y^2 - 48*y + 68 = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_one_l506_50677


namespace NUMINAMATH_CALUDE_log_inequality_sufficiency_not_necessity_l506_50626

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement of the theorem
theorem log_inequality_sufficiency_not_necessity :
  (∀ a b : ℝ, log10 a > log10 b → a > b) ∧
  (∃ a b : ℝ, a > b ∧ ¬(log10 a > log10 b)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_sufficiency_not_necessity_l506_50626


namespace NUMINAMATH_CALUDE_coupon_value_l506_50686

def vacuum_cost : ℝ := 250
def dishwasher_cost : ℝ := 450
def total_cost_after_coupon : ℝ := 625

theorem coupon_value : 
  vacuum_cost + dishwasher_cost - total_cost_after_coupon = 75 := by
  sorry

end NUMINAMATH_CALUDE_coupon_value_l506_50686


namespace NUMINAMATH_CALUDE_essay_word_count_excess_l506_50690

theorem essay_word_count_excess (word_limit : ℕ) (saturday_words : ℕ) (sunday_words : ℕ) :
  word_limit = 1000 →
  saturday_words = 450 →
  sunday_words = 650 →
  (saturday_words + sunday_words) - word_limit = 100 := by
  sorry

end NUMINAMATH_CALUDE_essay_word_count_excess_l506_50690


namespace NUMINAMATH_CALUDE_circle_tangent_to_axes_l506_50691

/-- A circle in the Cartesian plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the x-axis -/
def Circle.tangentToXAxis (c : Circle) : Prop :=
  c.center.2 = c.radius

/-- A circle is tangent to the y-axis -/
def Circle.tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- The main theorem -/
theorem circle_tangent_to_axes (c : Circle) :
  c.radius = 2 ∧ c.tangentToXAxis ∧ c.tangentToYAxis ↔ 
  ∀ x y : ℝ, c.equation x y ↔ (x - 2)^2 + (y - 2)^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_axes_l506_50691


namespace NUMINAMATH_CALUDE_tangent_line_derivative_l506_50695

variable (f : ℝ → ℝ)

theorem tangent_line_derivative (h : ∀ y, y = (1/2) * 1 + 3 → y = f 1) :
  deriv f 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_derivative_l506_50695


namespace NUMINAMATH_CALUDE_min_value_trigonometric_function_l506_50650

theorem min_value_trigonometric_function :
  ∀ x : ℝ, 0 < x → x < π / 2 →
    1 / (Real.sin x)^2 + 12 * Real.sqrt 3 / Real.cos x ≥ 28 ∧
    ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π / 2 ∧
      1 / (Real.sin x₀)^2 + 12 * Real.sqrt 3 / Real.cos x₀ = 28 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_function_l506_50650


namespace NUMINAMATH_CALUDE_jacket_cost_is_30_l506_50634

def calculate_jacket_cost (initial_amount dresses_count pants_count jackets_count dress_cost pants_cost transportation_cost remaining_amount : ℕ) : ℕ :=
  let total_spent := initial_amount - remaining_amount
  let dresses_cost := dresses_count * dress_cost
  let pants_cost := pants_count * pants_cost
  let other_costs := dresses_cost + pants_cost + transportation_cost
  let jackets_total_cost := total_spent - other_costs
  jackets_total_cost / jackets_count

theorem jacket_cost_is_30 :
  calculate_jacket_cost 400 5 3 4 20 12 5 139 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_is_30_l506_50634


namespace NUMINAMATH_CALUDE_shekar_average_marks_l506_50611

def shekar_marks : List ℕ := [76, 65, 82, 67, 75]

theorem shekar_average_marks :
  (shekar_marks.sum : ℚ) / shekar_marks.length = 73 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l506_50611


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l506_50679

-- Part 1
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
  a ∈ Set.Iic (1/3) :=
sorry

-- Part 2
def f' (x : ℝ) : ℝ := x^2 - 3*x + 2
def g (m : ℝ) (x : ℝ) : ℝ := -x + m

theorem range_of_m :
  ∀ m : ℝ, (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Ioo 1 8, f' x₁ = g m x₂) →
  m ∈ Set.Ioo 7 (31/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l506_50679


namespace NUMINAMATH_CALUDE_square_stack_area_l506_50623

theorem square_stack_area (blue_exposed red_exposed yellow_exposed : ℝ) 
  (h1 : blue_exposed = 25)
  (h2 : red_exposed = 19)
  (h3 : yellow_exposed = 11) :
  let blue_side := Real.sqrt blue_exposed
  let red_uncovered := red_exposed / blue_side
  let large_side := blue_side + red_uncovered
  large_side ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_stack_area_l506_50623


namespace NUMINAMATH_CALUDE_smallest_angle_equation_l506_50684

/-- The smallest positive angle θ in degrees that satisfies the equation
    cos θ = sin 45° + cos 60° - sin 30° - cos 15° -/
theorem smallest_angle_equation : ∃ θ : ℝ,
  θ > 0 ∧
  θ < 360 ∧
  Real.cos (θ * π / 180) = Real.sin (45 * π / 180) + Real.cos (60 * π / 180) - 
                           Real.sin (30 * π / 180) - Real.cos (15 * π / 180) ∧
  ∀ φ, 0 < φ ∧ φ < θ → 
    Real.cos (φ * π / 180) ≠ Real.sin (45 * π / 180) + Real.cos (60 * π / 180) - 
                             Real.sin (30 * π / 180) - Real.cos (15 * π / 180) ∧
  θ = 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_equation_l506_50684


namespace NUMINAMATH_CALUDE_nickels_count_l506_50615

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a penny in cents -/
def penny_value : ℕ := 1

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Calculates the number of nickels given the total value and the number of pennies and dimes -/
def calculate_nickels (total_value : ℕ) (num_pennies : ℕ) (num_dimes : ℕ) : ℕ :=
  let pennies_value := num_pennies * penny_value
  let dimes_value := num_dimes * dime_value
  let nickels_value := total_value - pennies_value - dimes_value
  nickels_value / nickel_value

theorem nickels_count (total_value : ℕ) (num_pennies : ℕ) (num_dimes : ℕ)
    (h1 : total_value = 59)
    (h2 : num_pennies = 9)
    (h3 : num_dimes = 3) :
    calculate_nickels total_value num_pennies num_dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_nickels_count_l506_50615


namespace NUMINAMATH_CALUDE_greatest_fourth_term_l506_50642

/-- An arithmetic sequence of five positive integers with sum 60 -/
structure ArithmeticSequence where
  a : ℕ+  -- first term
  d : ℕ+  -- common difference
  sum_eq_60 : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 60

/-- The fourth term of an arithmetic sequence -/
def fourth_term (seq : ArithmeticSequence) : ℕ := seq.a + 3 * seq.d

/-- The greatest possible fourth term is 34 -/
theorem greatest_fourth_term :
  ∀ seq : ArithmeticSequence, fourth_term seq ≤ 34 ∧ 
  ∃ seq : ArithmeticSequence, fourth_term seq = 34 :=
sorry

end NUMINAMATH_CALUDE_greatest_fourth_term_l506_50642


namespace NUMINAMATH_CALUDE_system_equations_properties_l506_50613

/-- Given a system of equations with parameters x, y, and m, prove properties about the solution and a related expression. -/
theorem system_equations_properties (x y m : ℝ) (h1 : 3 * x + 2 * y = m + 2) (h2 : 2 * x + y = m - 1)
  (hx : x > 0) (hy : y > 0) :
  (x = m - 4 ∧ y = 7 - m) ∧
  (4 < m ∧ m < 7) ∧
  (∀ (m : ℕ), 4 < m → m < 7 → (2 * x - 3 * y + m) ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_system_equations_properties_l506_50613


namespace NUMINAMATH_CALUDE_messages_total_680_l506_50618

/-- Calculates the total number of messages sent by Alina and Lucia over three days -/
def total_messages (lucia_day1 : ℕ) (alina_difference : ℕ) : ℕ :=
  let alina_day1 := lucia_day1 - alina_difference
  let day1_total := lucia_day1 + alina_day1
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := alina_day1 * 2
  let day2_total := lucia_day2 + alina_day2
  day1_total + day2_total + day1_total

theorem messages_total_680 :
  total_messages 120 20 = 680 := by
  sorry

end NUMINAMATH_CALUDE_messages_total_680_l506_50618


namespace NUMINAMATH_CALUDE_factorization_equality_l506_50660

theorem factorization_equality (m n : ℝ) : m^2 * n - m * n = m * n * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l506_50660


namespace NUMINAMATH_CALUDE_correct_divisor_problem_l506_50629

theorem correct_divisor_problem (student_divisor : ℕ) (student_answer : ℕ) (correct_answer : ℕ) : 
  student_divisor = 72 → student_answer = 24 → correct_answer = 48 →
  ∃ (dividend : ℕ) (correct_divisor : ℕ), 
    dividend / student_divisor = student_answer ∧
    dividend / correct_divisor = correct_answer ∧
    correct_divisor = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_problem_l506_50629


namespace NUMINAMATH_CALUDE_square_division_theorem_l506_50666

theorem square_division_theorem (s : ℝ) :
  s > 0 →
  (3 * s = 42) →
  s = 14 :=
by sorry

end NUMINAMATH_CALUDE_square_division_theorem_l506_50666


namespace NUMINAMATH_CALUDE_calculate_expression_l506_50681

theorem calculate_expression : 7 * (9 + 2/5) + 3 = 68.8 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l506_50681


namespace NUMINAMATH_CALUDE_cubic_roots_sum_min_l506_50687

theorem cubic_roots_sum_min (a : ℝ) (x₁ x₂ x₃ : ℝ) (h_pos : a > 0) 
  (h_roots : x₁^3 - a*x₁^2 + a*x₁ - a = 0 ∧ 
             x₂^3 - a*x₂^2 + a*x₂ - a = 0 ∧ 
             x₃^3 - a*x₃^2 + a*x₃ - a = 0) : 
  ∃ (m : ℝ), m = -4 ∧ ∀ (y : ℝ), y ≥ m → x₁^3 + x₂^3 + x₃^3 - 3*x₁*x₂*x₃ ≥ y :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_min_l506_50687


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l506_50653

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x * Real.sin x

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧
  x₂ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ f x = 0 → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l506_50653


namespace NUMINAMATH_CALUDE_unit_circle_representation_l506_50606

theorem unit_circle_representation (x y : ℝ) (n : ℤ) :
  (Real.arcsin x + Real.arccos y = n * π) →
  ((n = 0 → x^2 + y^2 = 1 ∧ x ≤ 0 ∧ y ≥ 0) ∧
   (n = 1 → x^2 + y^2 = 1 ∧ x ≥ 0 ∧ y ≤ 0)) :=
by sorry

end NUMINAMATH_CALUDE_unit_circle_representation_l506_50606


namespace NUMINAMATH_CALUDE_triangle_position_after_two_moves_l506_50603

/-- Represents the sides of a square --/
inductive SquareSide
  | Top
  | Right
  | Bottom
  | Left

/-- Represents a regular octagon --/
structure RegularOctagon where
  inner_angle : ℝ
  inner_angle_eq : inner_angle = 135

/-- Represents a square rolling around an octagon --/
structure RollingSquare where
  octagon : RegularOctagon
  rotation_per_move : ℝ
  rotation_per_move_eq : rotation_per_move = 135

/-- The result of rolling a square around an octagon --/
def roll_square (initial_side : SquareSide) (num_moves : ℕ) : SquareSide :=
  sorry

theorem triangle_position_after_two_moves :
  ∀ (octagon : RegularOctagon) (square : RollingSquare),
    roll_square SquareSide.Bottom 2 = SquareSide.Bottom :=
  sorry

end NUMINAMATH_CALUDE_triangle_position_after_two_moves_l506_50603


namespace NUMINAMATH_CALUDE_chameleon_color_impossibility_l506_50688

/-- Represents the state of chameleons on the island -/
structure ChameleonSystem :=
  (num_chameleons : Nat)
  (num_colors : Nat)
  (color_change : Nat → Nat → Nat)  -- Function representing color change

/-- Represents the property that all chameleons have been all colors -/
def all_chameleons_all_colors (system : ChameleonSystem) : Prop :=
  ∀ c : Nat, c < system.num_chameleons → 
    ∃ t1 t2 t3 : Nat, 
      system.color_change c t1 = 0 ∧ 
      system.color_change c t2 = 1 ∧ 
      system.color_change c t3 = 2

theorem chameleon_color_impossibility :
  ∀ system : ChameleonSystem, 
    system.num_chameleons = 35 → 
    system.num_colors = 3 → 
    ¬(all_chameleons_all_colors system) := by
  sorry

end NUMINAMATH_CALUDE_chameleon_color_impossibility_l506_50688


namespace NUMINAMATH_CALUDE_swimming_club_boys_l506_50628

theorem swimming_club_boys (total_members : ℕ) (total_attendees : ℕ) :
  total_members = 30 →
  total_attendees = 20 →
  ∃ (boys girls : ℕ),
    boys + girls = total_members ∧
    boys + (girls / 3) = total_attendees ∧
    boys = 15 := by
  sorry

end NUMINAMATH_CALUDE_swimming_club_boys_l506_50628


namespace NUMINAMATH_CALUDE_rectangle_area_l506_50646

theorem rectangle_area (y : ℝ) (h : y > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = y^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l506_50646


namespace NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l506_50655

/-- The shortest distance from a point on the line y = x + 1 to a point on the circle x^2 + y^2 + 2x + 4y + 4 = 0 is √2 - 1 -/
theorem shortest_distance_line_to_circle :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + 4 = 0}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      Real.sqrt 2 - 1 ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l506_50655


namespace NUMINAMATH_CALUDE_smallest_cube_volume_for_ziggurat_model_l506_50669

/-- The volume of the smallest cube that can contain a rectangular prism -/
theorem smallest_cube_volume_for_ziggurat_model (h : ℕ) (b : ℕ) : 
  h = 15 → b = 8 → (max h b) ^ 3 = 3375 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_volume_for_ziggurat_model_l506_50669


namespace NUMINAMATH_CALUDE_simplify_expression_l506_50614

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l506_50614


namespace NUMINAMATH_CALUDE_fibonacci_identity_l506_50610

def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_identity (n : ℕ) (h : n ≥ 1) :
  fib (n - 1) * fib (n + 1) - fib n ^ 2 = (-1) ^ n := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_identity_l506_50610


namespace NUMINAMATH_CALUDE_quadrilateral_prism_edges_and_vertices_l506_50631

/-- A prism with a quadrilateral base -/
structure QuadrilateralPrism :=
  (lateral_faces : ℕ)
  (lateral_faces_eq : lateral_faces = 4)

/-- The number of edges in a quadrilateral prism -/
def num_edges (p : QuadrilateralPrism) : ℕ := 12

/-- The number of vertices in a quadrilateral prism -/
def num_vertices (p : QuadrilateralPrism) : ℕ := 8

/-- Theorem stating that a quadrilateral prism has 12 edges and 8 vertices -/
theorem quadrilateral_prism_edges_and_vertices (p : QuadrilateralPrism) :
  num_edges p = 12 ∧ num_vertices p = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_prism_edges_and_vertices_l506_50631


namespace NUMINAMATH_CALUDE_last_two_average_l506_50616

theorem last_two_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / 7 : ℝ) = 60 →
  ((list.take 3).sum / 3 : ℝ) = 50 →
  ((list.drop 3).take 2).sum / 2 = 70 →
  ((list.drop 5).sum / 2 : ℝ) = 65 := by
sorry

end NUMINAMATH_CALUDE_last_two_average_l506_50616


namespace NUMINAMATH_CALUDE_daisy_count_l506_50685

def white_daisies : ℕ := 6

def pink_daisies : ℕ := 9 * white_daisies

def red_daisies : ℕ := 4 * pink_daisies - 3

def total_daisies : ℕ := white_daisies + pink_daisies + red_daisies

theorem daisy_count : total_daisies = 273 := by
  sorry

end NUMINAMATH_CALUDE_daisy_count_l506_50685


namespace NUMINAMATH_CALUDE_amusement_park_optimization_l506_50689

/-- Represents the ticket cost and ride time for an attraction -/
structure Attraction where
  ticketCost : Nat
  rideTime : Float

/-- Represents a ticket purchase option -/
structure TicketOption where
  quantity : Nat
  price : Float

theorem amusement_park_optimization (budget : Float) 
  (ferrisWheel rollerCoaster bumperCars carousel hauntedHouse : Attraction)
  (entranceFee : Float) (initialTickets : Nat)
  (individualTicketPrice : Float) (tenTicketBundle twentyTicketBundle : TicketOption)
  (lunchMinCost lunchMaxCost : Float) (souvenirMinCost souvenirMaxCost : Float)
  (timeBeforeActivity activityDuration : Float) :
  budget = 50 ∧ 
  entranceFee = 10 ∧ initialTickets = 5 ∧
  ferrisWheel = { ticketCost := 5, rideTime := 0.3 } ∧
  rollerCoaster = { ticketCost := 4, rideTime := 0.3 } ∧
  bumperCars = { ticketCost := 4, rideTime := 0.3 } ∧
  carousel = { ticketCost := 3, rideTime := 0.3 } ∧
  hauntedHouse = { ticketCost := 6, rideTime := 0.3 } ∧
  individualTicketPrice = 1.5 ∧
  tenTicketBundle = { quantity := 10, price := 12 } ∧
  twentyTicketBundle = { quantity := 20, price := 22 } ∧
  lunchMinCost = 8 ∧ lunchMaxCost = 15 ∧
  souvenirMinCost = 5 ∧ souvenirMaxCost = 12 ∧
  timeBeforeActivity = 3 ∧ activityDuration = 1 →
  (∃ (optimalPurchase : TicketOption),
    optimalPurchase = twentyTicketBundle ∧
    ferrisWheel.rideTime + rollerCoaster.rideTime + bumperCars.rideTime + 
    carousel.rideTime + hauntedHouse.rideTime = 1.5) := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_optimization_l506_50689


namespace NUMINAMATH_CALUDE_train_tunnel_crossing_time_l506_50652

theorem train_tunnel_crossing_time
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (tunnel_length : ℝ)
  (h1 : train_length = 100)
  (h2 : train_speed_kmph = 72)
  (h3 : tunnel_length = 1400) :
  (train_length + tunnel_length) / (train_speed_kmph * (1000 / 3600)) = 75 :=
by sorry

end NUMINAMATH_CALUDE_train_tunnel_crossing_time_l506_50652


namespace NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l506_50635

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extreme_values_and_monotonicity 
  (a b c : ℝ) :
  (f_derivative a b (-2/3) = 0 ∧ f_derivative a b 1 = 0) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x, -2/3 < x ∧ x < 1 → f_derivative (-1/2) (-2) x < 0) ∧
  (∀ x, (x < -2/3 ∨ 1 < x) → f_derivative (-1/2) (-2) x > 0) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_monotonicity_l506_50635


namespace NUMINAMATH_CALUDE_complement_M_in_U_l506_50698

-- Define the set U
def U : Set ℕ := {1,2,3,4,5,6,7}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 6*x + 5 ≤ 0}

-- State the theorem
theorem complement_M_in_U : (U \ M) = {6,7} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l506_50698


namespace NUMINAMATH_CALUDE_original_number_before_increase_l506_50697

theorem original_number_before_increase (final_number : ℝ) (increase_percentage : ℝ) (original_number : ℝ) : 
  final_number = 90 ∧ 
  increase_percentage = 50 ∧ 
  final_number = original_number * (1 + increase_percentage / 100) → 
  original_number = 60 := by
sorry

end NUMINAMATH_CALUDE_original_number_before_increase_l506_50697


namespace NUMINAMATH_CALUDE_elisa_family_women_without_daughters_l506_50659

/-- Represents a family tree starting from Elisa -/
structure ElisaFamily where
  daughters : Nat
  granddaughters : Nat
  daughters_with_children : Nat

/-- The conditions of Elisa's family -/
def elisa_family : ElisaFamily where
  daughters := 8
  granddaughters := 28
  daughters_with_children := 4

/-- The total number of daughters and granddaughters -/
def total_descendants (f : ElisaFamily) : Nat :=
  f.daughters + f.granddaughters

/-- The number of women (daughters and granddaughters) who have no daughters -/
def women_without_daughters (f : ElisaFamily) : Nat :=
  (f.daughters - f.daughters_with_children) + f.granddaughters

/-- Theorem stating that 32 of Elisa's daughters and granddaughters have no daughters -/
theorem elisa_family_women_without_daughters :
  women_without_daughters elisa_family = 32 := by
  sorry

end NUMINAMATH_CALUDE_elisa_family_women_without_daughters_l506_50659


namespace NUMINAMATH_CALUDE_jillian_shells_l506_50663

theorem jillian_shells (savannah_shells clayton_shells : ℕ) 
  (h1 : savannah_shells = 17)
  (h2 : clayton_shells = 8)
  (h3 : ∃ (total_shells : ℕ), total_shells = 27 * 2)
  (h4 : ∃ (jillian_shells : ℕ), jillian_shells + savannah_shells + clayton_shells = 27 * 2) :
  ∃ (jillian_shells : ℕ), jillian_shells = 29 := by
sorry

end NUMINAMATH_CALUDE_jillian_shells_l506_50663


namespace NUMINAMATH_CALUDE_burj_khalifa_height_l506_50672

theorem burj_khalifa_height (sears_height burj_difference : ℕ) 
  (h1 : sears_height = 527)
  (h2 : burj_difference = 303) : 
  sears_height + burj_difference = 830 := by
sorry

end NUMINAMATH_CALUDE_burj_khalifa_height_l506_50672


namespace NUMINAMATH_CALUDE_expression_equals_36_l506_50656

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l506_50656


namespace NUMINAMATH_CALUDE_prime_sequence_l506_50667

theorem prime_sequence (A : ℕ) : 
  Nat.Prime A ∧ 
  Nat.Prime (A - 4) ∧ 
  Nat.Prime (A - 6) ∧ 
  Nat.Prime (A - 12) ∧ 
  Nat.Prime (A - 18) → 
  A = 23 := by
sorry

end NUMINAMATH_CALUDE_prime_sequence_l506_50667


namespace NUMINAMATH_CALUDE_courtyard_length_l506_50662

/-- The length of a rectangular courtyard given its width and paving stone information. -/
theorem courtyard_length (width : ℚ) (num_stones : ℕ) (stone_length stone_width : ℚ) :
  width = 33 / 2 →
  num_stones = 132 →
  stone_length = 5 / 2 →
  stone_width = 2 →
  (num_stones * stone_length * stone_width) / width = 40 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_length_l506_50662


namespace NUMINAMATH_CALUDE_difference_of_squares_l506_50693

theorem difference_of_squares (x y : ℝ) : (-x + y) * (x + y) = y^2 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l506_50693


namespace NUMINAMATH_CALUDE_coeff_x_squared_in_expansion_l506_50683

/-- The coefficient of x^2 in the expansion of (1+2x)^6 is 60 -/
theorem coeff_x_squared_in_expansion : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * (1^(6-k)) * ((2:ℕ)^k)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_coeff_x_squared_in_expansion_l506_50683


namespace NUMINAMATH_CALUDE_B_equals_zero_one_two_l506_50665

def A : Set ℤ := {1, 0, -1, 2}

def B : Set ℕ := {y | ∃ x ∈ A, y = |x|}

theorem B_equals_zero_one_two : B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_B_equals_zero_one_two_l506_50665


namespace NUMINAMATH_CALUDE_octal_to_binary_conversion_l506_50673

theorem octal_to_binary_conversion :
  (135 : Nat).digits 8 = [1, 3, 5] →
  (135 : Nat).digits 2 = [1, 0, 1, 1, 1, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_octal_to_binary_conversion_l506_50673


namespace NUMINAMATH_CALUDE_inequality_of_means_l506_50636

theorem inequality_of_means (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ (a * b * c) ^ (1/3) > 3 * a * b * c / (a * b + b * c + c * a) :=
sorry

end NUMINAMATH_CALUDE_inequality_of_means_l506_50636


namespace NUMINAMATH_CALUDE_zachary_initial_money_l506_50649

/-- Calculates Zachary's initial money given the costs of items and additional amount needed --/
theorem zachary_initial_money 
  (football_cost shorts_cost shoes_cost additional_needed : ℚ) 
  (h1 : football_cost = 3.75)
  (h2 : shorts_cost = 2.40)
  (h3 : shoes_cost = 11.85)
  (h4 : additional_needed = 8) :
  football_cost + shorts_cost + shoes_cost - additional_needed = 9 := by
sorry

end NUMINAMATH_CALUDE_zachary_initial_money_l506_50649


namespace NUMINAMATH_CALUDE_simplify_expression_l506_50692

theorem simplify_expression (b : ℝ) : (1)*(2*b)*(3*b^2)*(4*b^3)*(5*b^4)*(6*b^5) = 720*b^15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l506_50692


namespace NUMINAMATH_CALUDE_carousel_revolutions_l506_50601

theorem carousel_revolutions (r₁ r₂ : ℝ) (n₁ : ℕ) :
  r₁ = 30 →
  r₂ = 10 →
  n₁ = 40 →
  r₁ * n₁ = r₂ * (120 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_carousel_revolutions_l506_50601


namespace NUMINAMATH_CALUDE_pencil_difference_l506_50621

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The time frame in minutes -/
def time_frame : ℕ := 6

/-- The time it takes for a hand-crank sharpener to sharpen one pencil (in seconds) -/
def hand_crank_time : ℕ := 45

/-- The time it takes for an electric sharpener to sharpen one pencil (in seconds) -/
def electric_time : ℕ := 20

/-- The difference in the number of pencils sharpened between the electric and hand-crank sharpeners -/
theorem pencil_difference : 
  (time_frame * seconds_per_minute) / electric_time - 
  (time_frame * seconds_per_minute) / hand_crank_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_l506_50621


namespace NUMINAMATH_CALUDE_triangle_side_length_l506_50645

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2asinB = √3 * b, b + c = 5, and bc = 6, then a = √7 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →  -- ABC is acute
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  2 * a * Real.sin B = Real.sqrt 3 * b →
  b + c = 5 →
  b * c = 6 →
  a = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l506_50645


namespace NUMINAMATH_CALUDE_monster_perimeter_l506_50648

theorem monster_perimeter (r : ℝ) (θ : ℝ) (h1 : r = 2) (h2 : θ = 2 * π / 3) :
  let arc_length := (2 * π - θ) / (2 * π) * (2 * π * r)
  let chord_length := 2 * r * Real.sin (θ / 2)
  arc_length + chord_length = 8 * π / 3 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_monster_perimeter_l506_50648


namespace NUMINAMATH_CALUDE_existence_of_positive_reals_l506_50675

theorem existence_of_positive_reals : ∃ (x y z : ℝ), 
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^4 + y^4 + z^4 = 13 ∧
  x^3*y^3*z + y^3*z^3*x + z^3*x^3*y = 6*Real.sqrt 3 ∧
  x^3*y*z + y^3*z*x + z^3*x*y = 5*Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_positive_reals_l506_50675


namespace NUMINAMATH_CALUDE_smaller_solid_volume_is_one_sixth_l506_50605

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  edgeLength : ℝ
  vertex : Point3D

/-- Calculates the volume of the smaller solid created by a plane cutting a cube -/
def smallerSolidVolume (cube : Cube) (plane : Plane) : ℝ :=
  sorry

/-- Theorem: The volume of the smaller solid in a cube with edge length 2,
    cut by a plane passing through vertex D and midpoints of AB and CG, is 1/6 -/
theorem smaller_solid_volume_is_one_sixth :
  let cube := Cube.mk 2 (Point3D.mk 0 0 0)
  let plane := Plane.mk 2 (-4) (-8) 0
  smallerSolidVolume cube plane = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_smaller_solid_volume_is_one_sixth_l506_50605


namespace NUMINAMATH_CALUDE_train_length_l506_50654

/-- The length of a train given its speed, platform length, and crossing time -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (5/18) →
  platform_length = 210 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 310 := by
sorry

end NUMINAMATH_CALUDE_train_length_l506_50654


namespace NUMINAMATH_CALUDE_counterfeit_coin_determination_l506_50602

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  containsCounterfeit : Bool

/-- Represents a weighing operation -/
def weighing (left right : CoinGroup) : WeighingResult :=
  sorry

/-- The main theorem stating that it's possible to determine if counterfeit coins are heavier or lighter -/
theorem counterfeit_coin_determination :
  ∀ (coins : List CoinGroup),
    coins.length = 3 →
    (coins.map CoinGroup.size).sum = 103 →
    (coins.filter CoinGroup.containsCounterfeit).length = 2 →
    ∃ (w₁ w₂ w₃ : CoinGroup × CoinGroup),
      (∀ g₁ g₂, weighing g₁ g₂ = WeighingResult.Equal → g₁.containsCounterfeit = g₂.containsCounterfeit) →
      let r₁ := weighing w₁.1 w₁.2
      let r₂ := weighing w₂.1 w₂.2
      let r₃ := weighing w₃.1 w₃.2
      (r₁ ≠ WeighingResult.Equal ∨ r₂ ≠ WeighingResult.Equal ∨ r₃ ≠ WeighingResult.Equal) :=
by
  sorry

end NUMINAMATH_CALUDE_counterfeit_coin_determination_l506_50602


namespace NUMINAMATH_CALUDE_complex_equation_solution_l506_50617

theorem complex_equation_solution (z : ℂ) :
  Complex.abs z = 2 + z + Complex.I * 3 → z = 5 / 4 - Complex.I * 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l506_50617


namespace NUMINAMATH_CALUDE_girls_on_track_l506_50658

/-- Calculates the total number of girls on a track with given specifications -/
def total_girls (track_length : ℕ) (student_spacing : ℕ) : ℕ :=
  let students_per_side := track_length / student_spacing + 1
  let cycles_per_side := students_per_side / 3
  let girls_per_side := cycles_per_side * 2
  girls_per_side * 2

/-- The total number of girls on a 100-meter track with students every 2 meters,
    arranged in a pattern of two girls followed by one boy, is 68 -/
theorem girls_on_track : total_girls 100 2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_girls_on_track_l506_50658
