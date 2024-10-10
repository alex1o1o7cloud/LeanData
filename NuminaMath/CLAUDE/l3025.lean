import Mathlib

namespace driver_weekly_distance_l3025_302521

/-- Represents the driving schedule for a city bus driver --/
structure DrivingSchedule where
  mwf_hours : ℝ
  mwf_speed : ℝ
  tue_hours : ℝ
  tue_speed : ℝ
  thu_hours : ℝ
  thu_speed : ℝ

/-- Calculates the total distance traveled by the driver in a week --/
def totalDistanceTraveled (schedule : DrivingSchedule) : ℝ :=
  3 * (schedule.mwf_hours * schedule.mwf_speed) +
  schedule.tue_hours * schedule.tue_speed +
  schedule.thu_hours * schedule.thu_speed

/-- Theorem stating that the driver travels 148 kilometers in a week --/
theorem driver_weekly_distance (schedule : DrivingSchedule)
  (h1 : schedule.mwf_hours = 3)
  (h2 : schedule.mwf_speed = 12)
  (h3 : schedule.tue_hours = 2.5)
  (h4 : schedule.tue_speed = 9)
  (h5 : schedule.thu_hours = 2.5)
  (h6 : schedule.thu_speed = 7) :
  totalDistanceTraveled schedule = 148 := by
  sorry

#eval totalDistanceTraveled {
  mwf_hours := 3,
  mwf_speed := 12,
  tue_hours := 2.5,
  tue_speed := 9,
  thu_hours := 2.5,
  thu_speed := 7
}

end driver_weekly_distance_l3025_302521


namespace inequality_proof_l3025_302505

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a + 1/b)^2 + (b + 1/c)^2 + (c + 1/a)^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end inequality_proof_l3025_302505


namespace largest_five_digit_number_with_product_180_l3025_302586

/-- Represents a five-digit number as a list of its digits -/
def FiveDigitNumber := List Nat

/-- Checks if a given list represents a valid five-digit number -/
def is_valid_five_digit_number (n : FiveDigitNumber) : Prop :=
  n.length = 5 ∧ n.all (· < 10) ∧ n.head! ≠ 0

/-- Computes the product of the digits of a number -/
def digit_product (n : FiveDigitNumber) : Nat :=
  n.prod

/-- Computes the sum of the digits of a number -/
def digit_sum (n : FiveDigitNumber) : Nat :=
  n.sum

/-- Compares two five-digit numbers -/
def is_greater (a b : FiveDigitNumber) : Prop :=
  a.foldl (fun acc d => acc * 10 + d) 0 > b.foldl (fun acc d => acc * 10 + d) 0

theorem largest_five_digit_number_with_product_180 :
  ∃ (M : FiveDigitNumber),
    is_valid_five_digit_number M ∧
    digit_product M = 180 ∧
    (∀ (N : FiveDigitNumber), is_valid_five_digit_number N → digit_product N = 180 → is_greater M N) ∧
    digit_sum M = 19 :=
  sorry

end largest_five_digit_number_with_product_180_l3025_302586


namespace quadratic_equation_solution_l3025_302578

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => (2*x + 1)^2 - 3*(2*x + 1)
  ∃ (x₁ x₂ : ℝ), x₁ = -1/2 ∧ x₂ = 1 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_equation_solution_l3025_302578


namespace correct_oranges_count_l3025_302517

/-- Calculates the number of oranges needed to reach a desired total fruit count -/
def oranges_needed (total_desired : ℕ) (apples : ℕ) (bananas : ℕ) : ℕ :=
  total_desired - (apples + bananas)

theorem correct_oranges_count : oranges_needed 12 3 4 = 5 := by
  sorry

end correct_oranges_count_l3025_302517


namespace no_three_five_powers_l3025_302508

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem no_three_five_powers (n : ℕ) :
  ∀ α β : ℕ, v n ≠ 3^α * 5^β :=
by sorry

end no_three_five_powers_l3025_302508


namespace probability_relates_to_uncertain_events_l3025_302531

-- Define the basic types of events
inductive Event
  | Certain
  | Impossible
  | Random

-- Define a probability function
def probability (e : Event) : Real :=
  match e with
  | Event.Certain => 1
  | Event.Impossible => 0
  | Event.Random => sorry -- Assumes a value between 0 and 1

-- Define what it means for an event to be uncertain
def is_uncertain (e : Event) : Prop :=
  e = Event.Random

-- State the theorem
theorem probability_relates_to_uncertain_events :
  ∃ (e : Event), is_uncertain e ∧ 0 < probability e ∧ probability e < 1 :=
sorry

end probability_relates_to_uncertain_events_l3025_302531


namespace f_range_l3025_302587

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sin x - 1

theorem f_range : Set.range f = Set.Icc (-5/4 : ℝ) 1 := by
  sorry

end f_range_l3025_302587


namespace total_contribution_proof_l3025_302595

/-- Proves that the total contribution is $1040 given the specified conditions --/
theorem total_contribution_proof (niraj brittany angela : ℕ) : 
  niraj = 80 ∧ 
  brittany = 3 * niraj ∧ 
  angela = 3 * brittany → 
  niraj + brittany + angela = 1040 := by
  sorry

end total_contribution_proof_l3025_302595


namespace range_of_a_for_subset_l3025_302541

-- Define the sets A and B
def A (a : ℝ) := {x : ℝ | 3 + a ≤ x ∧ x ≤ 4 + 3*a}
def B := {x : ℝ | (x + 4) / (5 - x) ≥ 0 ∧ x ≠ 5}

-- State the theorem
theorem range_of_a_for_subset : 
  {a : ℝ | ∀ x, x ∈ A a → x ∈ B} = {a : ℝ | -1/2 ≤ a ∧ a < 1/3} := by sorry

end range_of_a_for_subset_l3025_302541


namespace system_solution_l3025_302509

theorem system_solution : 
  ∀ x y : ℝ, x > 0 → y > 0 →
  (3*y - Real.sqrt (y/x) - 6*Real.sqrt (x*y) + 2 = 0 ∧ 
   x^2 + 81*x^2*y^4 = 2*y^2) →
  ((x = 1/3 ∧ y = 1/3) ∨ 
   (x = Real.sqrt (Real.sqrt 31) / 12 ∧ y = Real.sqrt (Real.sqrt 31) / 3)) :=
by sorry

end system_solution_l3025_302509


namespace inequality_proof_l3025_302569

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h5 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
  sorry

end inequality_proof_l3025_302569


namespace polynomial_remainder_l3025_302577

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15) % (4 * x - 8) = 31 := by
  sorry

end polynomial_remainder_l3025_302577


namespace prob_at_least_one_white_is_seven_tenths_l3025_302580

/-- The probability of drawing at least one white ball when randomly selecting two balls from a bag containing 3 black balls and 2 white balls. -/
def prob_at_least_one_white : ℚ := 7/10

/-- The total number of balls in the bag. -/
def total_balls : ℕ := 5

/-- The number of black balls in the bag. -/
def black_balls : ℕ := 3

/-- The number of white balls in the bag. -/
def white_balls : ℕ := 2

/-- The theorem stating that the probability of drawing at least one white ball
    when randomly selecting two balls from a bag containing 3 black balls and
    2 white balls is equal to 7/10. -/
theorem prob_at_least_one_white_is_seven_tenths :
  prob_at_least_one_white = 7/10 ∧
  total_balls = black_balls + white_balls ∧
  black_balls = 3 ∧
  white_balls = 2 := by
  sorry

end prob_at_least_one_white_is_seven_tenths_l3025_302580


namespace oranges_per_box_l3025_302500

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℚ) 
  (h1 : total_oranges = 72) 
  (h2 : num_boxes = 3) : 
  (total_oranges : ℚ) / num_boxes = 24 := by
sorry

end oranges_per_box_l3025_302500


namespace parabola_directrix_l3025_302590

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * y^2

/-- The directrix equation -/
def directrix_equation (x : ℝ) : Prop := x = 1

/-- Theorem: The directrix of the parabola x = -1/4 * y^2 is x = 1 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola_equation x y → 
  ∃ (d : ℝ), directrix_equation d ∧
  ∀ (p q : ℝ × ℝ), 
    parabola_equation p.1 p.2 →
    (p.1 - d)^2 = (p.1 - q.1)^2 + (p.2 - q.2)^2 →
    q.1 = -1 ∧ q.2 = 0 :=
sorry

end parabola_directrix_l3025_302590


namespace gcd_of_three_numbers_l3025_302520

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end gcd_of_three_numbers_l3025_302520


namespace arcsin_sum_inequality_l3025_302513

theorem arcsin_sum_inequality (x y : ℝ) : 
  Real.arcsin x + Real.arcsin y > π / 2 ↔ 
  x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ x^2 + y^2 > 1 := by
  sorry

end arcsin_sum_inequality_l3025_302513


namespace max_adjusted_employees_range_of_a_l3025_302545

/- Define the total number of employees -/
def total_employees : ℕ := 1000

/- Define the original average profit per employee (in yuan) -/
def original_profit : ℕ := 100000

/- Define the function for adjusted employees' average profit -/
def adjusted_profit (a x : ℝ) : ℝ := 10000 * (a - 0.008 * x)

/- Define the function for remaining employees' average profit -/
def remaining_profit (x : ℝ) : ℝ := original_profit * (1 + 0.004 * x)

/- Theorem for part I -/
theorem max_adjusted_employees :
  ∃ (max_x : ℕ), max_x = 750 ∧
  ∀ (x : ℕ), x > 0 → x ≤ max_x →
  (total_employees - x : ℝ) * remaining_profit x ≥ total_employees * original_profit ∧
  ¬∃ (y : ℕ), y > max_x ∧ y > 0 ∧
  (total_employees - y : ℝ) * remaining_profit y ≥ total_employees * original_profit :=
sorry

/- Theorem for part II -/
theorem range_of_a (x : ℝ) (hx : 0 < x ∧ x ≤ 750) :
  ∃ (lower upper : ℝ), lower = 0 ∧ upper = 7 ∧
  ∀ (a : ℝ), a > lower ∧ a ≤ upper →
  x * adjusted_profit a x ≤ (total_employees - x) * remaining_profit x ∧
  ¬∃ (b : ℝ), b > upper ∧
  x * adjusted_profit b x ≤ (total_employees - x) * remaining_profit x :=
sorry

end max_adjusted_employees_range_of_a_l3025_302545


namespace y_intercept_of_line_l3025_302575

/-- The y-intercept of the line x/a² - y/b² = 1 is -b², where a and b are non-zero real numbers. -/
theorem y_intercept_of_line (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ y : ℝ, (0 : ℝ) / a^2 - y / b^2 = 1 ∧ y = -b^2 := by sorry

end y_intercept_of_line_l3025_302575


namespace equal_sequences_l3025_302532

theorem equal_sequences (n : ℕ) (a b : Fin n → ℕ) 
  (h_gcd : Nat.gcd n 6 = 1)
  (h_a_pos : ∀ i, a i > 0)
  (h_b_pos : ∀ i, b i > 0)
  (h_a_inc : ∀ i j, i < j → a i < a j)
  (h_b_inc : ∀ i j, i < j → b i < b j)
  (h_sum_eq : ∀ j k l, j < k → k < l → a j + a k + a l = b j + b k + b l) :
  ∀ i, a i = b i :=
sorry

end equal_sequences_l3025_302532


namespace smallest_k_for_fifteen_digit_period_l3025_302551

/-- Represents a positive rational number with a decimal representation having a minimal period of 30 digits -/
def RationalWith30DigitPeriod : Type := { q : ℚ // q > 0 ∧ ∃ m : ℕ+, q = m / (10^30 - 1) }

/-- Given two positive rational numbers with 30-digit periods, returns true if their difference has a 15-digit period -/
def hasFifteenDigitPeriodDiff (a b : RationalWith30DigitPeriod) : Prop :=
  ∃ p : ℤ, (a.val - b.val : ℚ) = p / (10^15 - 1)

/-- Given two positive rational numbers with 30-digit periods and a natural number k,
    returns true if their sum with k times the second number has a 15-digit period -/
def hasFifteenDigitPeriodSum (a b : RationalWith30DigitPeriod) (k : ℕ) : Prop :=
  ∃ q : ℤ, (a.val + k * b.val : ℚ) = q / (10^15 - 1)

theorem smallest_k_for_fifteen_digit_period (a b : RationalWith30DigitPeriod)
  (h : hasFifteenDigitPeriodDiff a b) :
  (∀ k < 6, ¬hasFifteenDigitPeriodSum a b k) ∧ hasFifteenDigitPeriodSum a b 6 :=
sorry

end smallest_k_for_fifteen_digit_period_l3025_302551


namespace tangent_line_at_0_1_l3025_302530

/-- A line that is tangent to the unit circle at (0, 1) has the equation y = 1 -/
theorem tangent_line_at_0_1 (l : Set (ℝ × ℝ)) :
  (∀ p ∈ l, p.1^2 + p.2^2 = 1 → p = (0, 1)) →  -- l is tangent to the circle
  (0, 1) ∈ l →                                 -- l passes through (0, 1)
  l = {p : ℝ × ℝ | p.2 = 1} :=                 -- l has the equation y = 1
by sorry

end tangent_line_at_0_1_l3025_302530


namespace product_of_special_numbers_l3025_302529

theorem product_of_special_numbers (a b : ℝ) 
  (ha : a = Real.exp (2 - a)) 
  (hb : 1 + Real.log b = Real.exp (2 - (1 + Real.log b))) : 
  a * b = Real.exp 1 := by
sorry

end product_of_special_numbers_l3025_302529


namespace residue_mod_17_l3025_302510

theorem residue_mod_17 : (243 * 15 - 22 * 8 + 5) % 17 = 5 := by
  sorry

end residue_mod_17_l3025_302510


namespace percentage_problem_l3025_302596

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 + (8 / 100) * 24 = 5.92 ↔ P = 10 := by sorry

end percentage_problem_l3025_302596


namespace work_completion_time_l3025_302501

/-- Given that two workers A and B together complete a work in a certain number of days,
    and one worker alone can complete the work in a different number of days,
    we can determine how long it takes for both workers together to complete the work. -/
theorem work_completion_time
  (days_together : ℝ)
  (days_a_alone : ℝ)
  (h1 : days_together > 0)
  (h2 : days_a_alone > 0)
  (h3 : days_together < days_a_alone)
  (h4 : (1 / days_together) = (1 / days_a_alone) + (1 / days_b_alone))
  (h5 : days_together = 6)
  (h6 : days_a_alone = 10) :
  days_together = 6 :=
by sorry

end work_completion_time_l3025_302501


namespace square_ending_same_nonzero_digits_l3025_302550

theorem square_ending_same_nonzero_digits (n : ℕ) :
  (∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 100 = d * 10 + d) →
  n^2 % 100 = 44 := by
sorry

end square_ending_same_nonzero_digits_l3025_302550


namespace average_words_in_crossword_puzzle_l3025_302552

/-- The number of words needed to use up a pencil -/
def words_per_pencil : ℕ := 1050

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of puzzles completed in two weeks -/
def puzzles_in_two_weeks : ℕ := days_in_two_weeks

/-- The average number of words in each crossword puzzle -/
def average_words_per_puzzle : ℚ := words_per_pencil / puzzles_in_two_weeks

theorem average_words_in_crossword_puzzle :
  average_words_per_puzzle = 75 := by sorry

end average_words_in_crossword_puzzle_l3025_302552


namespace expression_evaluation_l3025_302559

theorem expression_evaluation :
  let a : ℤ := -2
  let expr := 3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a))
  expr = 10 := by sorry

end expression_evaluation_l3025_302559


namespace point_p_properties_l3025_302516

-- Define point P
def P (a : ℝ) : ℝ × ℝ := (2*a - 2, a + 5)

-- Define point Q
def Q : ℝ × ℝ := (4, 5)

theorem point_p_properties (a : ℝ) :
  -- Part 1
  (P a).2 = 0 → P a = (-12, 0) ∧
  -- Part 2
  (P a).1 = Q.1 → P a = (4, 8) ∧
  -- Part 3
  (P a).1 < 0 ∧ (P a).2 > 0 ∧ |(P a).1| = |(P a).2| → a^2022 + 2022 = 2023 :=
by sorry

end point_p_properties_l3025_302516


namespace inequality_range_l3025_302571

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| > a^2 + a + 1) → 
  a > -2 ∧ a < 1 :=
by sorry

end inequality_range_l3025_302571


namespace ball_count_equality_l3025_302592

-- Define the initial state of the urns
def Urn := ℕ → ℕ

-- m: initial number of black balls in the first urn
-- n: initial number of white balls in the second urn
-- k: number of balls transferred between urns
def initial_state (m n k : ℕ) : Urn × Urn :=
  (λ _ => m, λ _ => n)

-- Function to represent the ball transfer process
def transfer_balls (state : Urn × Urn) (k : ℕ) : Urn × Urn :=
  let (urn1, urn2) := state
  let urn1_after := λ color =>
    if color = 0 then urn1 0 - k + (k - (urn2 1 - (urn2 1 - k)))
    else k - (urn2 1 - (urn2 1 - k))
  let urn2_after := λ color =>
    if color = 0 then k - (k - (urn2 1 - (urn2 1 - k)))
    else urn2 1 - k + (urn2 1 - (urn2 1 - k))
  (urn1_after, urn2_after)

theorem ball_count_equality (m n k : ℕ) :
  let (final_urn1, final_urn2) := transfer_balls (initial_state m n k) k
  final_urn1 1 = final_urn2 0 := by
  sorry

end ball_count_equality_l3025_302592


namespace meeting_point_distance_from_top_l3025_302502

-- Define the race parameters
def race_length : ℝ := 12
def uphill_length : ℝ := 6
def downhill_length : ℝ := 6

-- Define Jack's parameters
def jack_start_time : ℝ := 0
def jack_uphill_speed : ℝ := 12
def jack_downhill_speed : ℝ := 18

-- Define Jill's parameters
def jill_start_time : ℝ := 0.25  -- 15 minutes = 0.25 hours
def jill_uphill_speed : ℝ := 14
def jill_downhill_speed : ℝ := 19

-- Define the theorem
theorem meeting_point_distance_from_top : 
  ∃ (meeting_time : ℝ) (meeting_distance : ℝ),
    meeting_time > jack_start_time + (uphill_length / jack_uphill_speed) ∧
    meeting_time > jill_start_time ∧
    meeting_time < jill_start_time + (uphill_length / jill_uphill_speed) ∧
    meeting_distance = uphill_length - (meeting_time - jill_start_time) * jill_uphill_speed ∧
    meeting_distance = downhill_length - (meeting_time - (jack_start_time + uphill_length / jack_uphill_speed)) * jack_downhill_speed ∧
    meeting_distance = 699 / 64 := by
  sorry

end meeting_point_distance_from_top_l3025_302502


namespace complement_intersection_theorem_l3025_302582

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {2, 3} := by sorry

end complement_intersection_theorem_l3025_302582


namespace B_power_100_is_identity_l3025_302588

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]]

theorem B_power_100_is_identity :
  B ^ 100 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end B_power_100_is_identity_l3025_302588


namespace right_triangle_pythagorean_representation_l3025_302536

theorem right_triangle_pythagorean_representation
  (a b c : ℕ)
  (d : ℤ)
  (h_order : a < b ∧ b < c)
  (h_gcd : Nat.gcd (c - a) (c - b) = 1)
  (h_right_triangle : (a + d)^2 + (b + d)^2 = (c + d)^2) :
  ∃ l m : ℤ, (c : ℤ) + d = l^2 + m^2 := by
  sorry

end right_triangle_pythagorean_representation_l3025_302536


namespace sin_cos_product_positive_implies_quadrant_I_or_III_l3025_302543

def is_in_quadrant_I_or_III (θ : Real) : Prop :=
  (0 < θ ∧ θ < Real.pi / 2) ∨ (Real.pi < θ ∧ θ < 3 * Real.pi / 2)

theorem sin_cos_product_positive_implies_quadrant_I_or_III (θ : Real) :
  Real.sin θ * Real.cos θ > 0 → is_in_quadrant_I_or_III θ :=
by
  sorry

end sin_cos_product_positive_implies_quadrant_I_or_III_l3025_302543


namespace max_area_is_one_l3025_302584

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ
  h_nonzero : m ≠ 0

/-- The maximum area of triangle PMN for the given ellipse and line configuration -/
def max_area (e : Ellipse) (l : Line) : ℝ := 1

/-- Theorem stating the maximum area of triangle PMN is 1 -/
theorem max_area_is_one (e : Ellipse) (l : Line) 
  (h_focus : e.a^2 - e.b^2 = 9)
  (h_vertex : e.a^2 = 12)
  (h_line : l.c = 3) :
  max_area e l = 1 := by sorry

end max_area_is_one_l3025_302584


namespace tricycle_count_l3025_302579

theorem tricycle_count (total_children : ℕ) (total_wheels : ℕ) (walking_children : ℕ) :
  total_children = 10 →
  total_wheels = 24 →
  walking_children = 2 →
  ∃ (bicycles tricycles : ℕ),
    bicycles + tricycles + walking_children = total_children ∧
    2 * bicycles + 3 * tricycles = total_wheels ∧
    tricycles = 8 :=
by sorry

end tricycle_count_l3025_302579


namespace f_zero_is_zero_l3025_302581

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x^2 + y) + 4 * (f x) * y

theorem f_zero_is_zero (f : ℝ → ℝ) (h : special_function f) (h2 : f 2 = 4) : f 0 = 0 := by
  sorry

end f_zero_is_zero_l3025_302581


namespace simplify_expression_l3025_302597

theorem simplify_expression (a : ℝ) (ha : a > 0) :
  a^2 / (a^(1/2) * a^(2/3)) = a^(5/6) := by sorry

end simplify_expression_l3025_302597


namespace two_thousand_fourteenth_smallest_perimeter_l3025_302565

/-- A right triangle with integer side lengths forming an arithmetic sequence -/
structure ArithmeticRightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  a_lt_b : a < b
  b_lt_c : b < c
  is_arithmetic : b - a = c - b
  is_right : a^2 + b^2 = c^2

/-- The perimeter of an arithmetic right triangle -/
def perimeter (t : ArithmeticRightTriangle) : ℕ := t.a + t.b + t.c

/-- The theorem stating that the 2014th smallest perimeter of arithmetic right triangles is 24168 -/
theorem two_thousand_fourteenth_smallest_perimeter :
  (ArithmeticRightTriangle.mk 6042 8056 10070 (by sorry) (by sorry) (by sorry) (by sorry) |>
    perimeter) = 24168 := by sorry

end two_thousand_fourteenth_smallest_perimeter_l3025_302565


namespace prob_xi_equals_three_l3025_302557

/-- A random variable following a binomial distribution B(6, 1/2) -/
def ξ : ℕ → ℝ := sorry

/-- The probability mass function for ξ -/
def P (k : ℕ) : ℝ := sorry

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- Theorem: The probability that ξ equals 3 is 5/16 -/
theorem prob_xi_equals_three : P 3 = 5 / 16 := by sorry

end prob_xi_equals_three_l3025_302557


namespace percentage_problem_l3025_302547

theorem percentage_problem : 
  let product := 45 * 8
  let total := 900
  let percentage := (product / total) * 100
  percentage = 40 := by sorry

end percentage_problem_l3025_302547


namespace real_imaginary_intersection_empty_l3025_302518

-- Define the universal set C (complex numbers)
variable (C : Type)

-- Define R (real numbers) and I (pure imaginary numbers) as subsets of C
variable (R I : Set C)

-- Theorem statement
theorem real_imaginary_intersection_empty : R ∩ I = ∅ := by
  sorry

end real_imaginary_intersection_empty_l3025_302518


namespace marks_change_factor_l3025_302589

theorem marks_change_factor (n : ℕ) (initial_avg final_avg : ℝ) (h1 : n = 10) (h2 : initial_avg = 40) (h3 : final_avg = 80) :
  ∃ (factor : ℝ), factor * (n * initial_avg) = n * final_avg ∧ factor = 2 := by
sorry

end marks_change_factor_l3025_302589


namespace set_properties_l3025_302583

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {0, |x|}
def B : Set ℝ := {1, 0, -1}

-- State the theorem
theorem set_properties (x : ℝ) (h : A x ⊆ B) :
  (x = 1 ∨ x = -1) ∧
  (A x ∪ B = {-1, 0, 1}) ∧
  (B \ A x = {-1}) :=
by sorry

end set_properties_l3025_302583


namespace kameron_has_100_kangaroos_l3025_302519

/-- The number of kangaroos Bert currently has -/
def bert_initial : ℕ := 20

/-- The number of days until Bert has the same number of kangaroos as Kameron -/
def days : ℕ := 40

/-- The number of kangaroos Bert buys per day -/
def bert_rate : ℕ := 2

/-- The number of kangaroos Kameron has -/
def kameron_kangaroos : ℕ := bert_initial + days * bert_rate

theorem kameron_has_100_kangaroos : kameron_kangaroos = 100 := by
  sorry

end kameron_has_100_kangaroos_l3025_302519


namespace stratified_sampling_girls_count_l3025_302593

theorem stratified_sampling_girls_count 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girls_boys_diff : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : girls_boys_diff = 6)
  (h4 : sample_size = (sample_size / 2 - girls_boys_diff / 2) * 2 + girls_boys_diff) :
  (sample_size / 2 - girls_boys_diff / 2) * (total_students / sample_size) = 970 := by
  sorry

#check stratified_sampling_girls_count

end stratified_sampling_girls_count_l3025_302593


namespace problem_1_l3025_302558

theorem problem_1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 := by
  sorry

end problem_1_l3025_302558


namespace polynomial_division_remainder_l3025_302540

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^6 - 2*x^5 + x^4 - x^2 - 2*x + 1 = 
  ((x^2 - 1) * (x - 2) * (x + 2)) * q + (2*x^3 - 9*x^2 + 3*x + 2) := by
  sorry

end polynomial_division_remainder_l3025_302540


namespace final_price_after_discounts_l3025_302512

def original_price : ℝ := 200
def weekend_discount : ℝ := 0.4
def wednesday_discount : ℝ := 0.2

theorem final_price_after_discounts :
  (original_price * (1 - weekend_discount)) * (1 - wednesday_discount) = 96 := by
  sorry

end final_price_after_discounts_l3025_302512


namespace secretary_typing_arrangements_l3025_302546

def remaining_letters : Finset Nat := {1, 2, 3, 4, 6, 7, 8, 10}

def possible_arrangements (s : Finset Nat) : Nat :=
  Finset.card s + 2

theorem secretary_typing_arrangements :
  (Finset.powerset remaining_letters).sum (fun s => Nat.choose 8 (Finset.card s) * possible_arrangements s) = 1400 := by
  sorry

end secretary_typing_arrangements_l3025_302546


namespace intersection_value_l3025_302591

theorem intersection_value (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = k * x₁ ∧ y₁ = 1 / x₁ ∧
  y₂ = k * x₂ ∧ y₂ = 1 / x₂ ∧
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₁ ≠ x₂ →
  x₁ * y₂ + x₂ * y₁ = -2 :=
by sorry

end intersection_value_l3025_302591


namespace quadratic_inequality_l3025_302568

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 4 > 0 ↔ x < -1 ∨ x > 4 := by
  sorry

end quadratic_inequality_l3025_302568


namespace distance_to_big_rock_l3025_302528

/-- The distance to Big Rock given the rower's speed, river's speed, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_speed : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 6)
  (h2 : river_speed = 2)
  (h3 : round_trip_time = 1) :
  (rower_speed + river_speed) * (rower_speed - river_speed) * round_trip_time / 
  (rower_speed + river_speed + rower_speed - river_speed) = 8/3 := by
sorry

end distance_to_big_rock_l3025_302528


namespace smallest_n_satisfying_condition_l3025_302576

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

def satisfies_condition (n : ℕ) : Prop :=
  n > 6 ∧ trailing_zeros (3 * n) = 4 * trailing_zeros n

theorem smallest_n_satisfying_condition :
  ∃ (n : ℕ), satisfies_condition n ∧ ∀ m, satisfies_condition m → n ≤ m :=
sorry

end smallest_n_satisfying_condition_l3025_302576


namespace nine_digit_palindromes_l3025_302544

/-- A function that returns the number of n-digit palindromic integers using only the digits 1, 2, and 3 -/
def count_palindromes (n : ℕ) : ℕ :=
  if n % 2 = 0 then 3^(n/2) else 3^((n+1)/2)

/-- The number of positive nine-digit palindromic integers using only the digits 1, 2, and 3 is 243 -/
theorem nine_digit_palindromes : count_palindromes 9 = 243 := by
  sorry

end nine_digit_palindromes_l3025_302544


namespace xiaohong_school_distance_l3025_302594

/-- The distance between Xiaohong's home and school -/
def distance : ℝ := 2880

/-- The scheduled arrival time in minutes -/
def scheduled_time : ℝ := 29

theorem xiaohong_school_distance :
  (∃ t : ℝ, 
    distance = 120 * (t - 5) ∧
    distance = 90 * (t + 3)) →
  distance = 2880 :=
by sorry

end xiaohong_school_distance_l3025_302594


namespace work_completion_time_l3025_302504

/-- The number of days it takes for A and B together to complete the work -/
def total_days : ℕ := 24

/-- The speed ratio of A to B -/
def speed_ratio : ℕ := 3

/-- The number of days it takes for A alone to complete the work -/
def days_for_A : ℕ := 32

theorem work_completion_time :
  speed_ratio * total_days = (speed_ratio + 1) * days_for_A :=
sorry

end work_completion_time_l3025_302504


namespace circle_distance_theorem_l3025_302527

theorem circle_distance_theorem (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → ∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 = 1 ∧ 
    x2^2 + y2^2 = 1 ∧ 
    (x1 - a)^2 + (y1 - 1)^2 = 4 ∧ 
    (x2 - a)^2 + (y2 - 1)^2 = 4 ∧ 
    (x1, y1) ≠ (x2, y2)) → 
  a = -1 :=
sorry

end circle_distance_theorem_l3025_302527


namespace speed_doubling_l3025_302523

theorem speed_doubling (distance : ℝ) (original_time : ℝ) (new_time : ℝ) 
  (h1 : distance = 440)
  (h2 : original_time = 3)
  (h3 : new_time = original_time / 2)
  : (distance / new_time) = 2 * (distance / original_time) := by
  sorry

#check speed_doubling

end speed_doubling_l3025_302523


namespace fraction_enlargement_l3025_302542

theorem fraction_enlargement (x y : ℝ) (h : x + y ≠ 0) :
  (2 * (3 * x) * (3 * y)) / ((3 * x) + (3 * y)) = 3 * ((2 * x * y) / (x + y)) :=
by sorry

end fraction_enlargement_l3025_302542


namespace absolute_value_quadratic_inequality_l3025_302514

theorem absolute_value_quadratic_inequality (x : ℝ) :
  |3 * x^2 - 5 * x - 2| < 5 ↔ x > -1/3 ∧ x < 1/3 :=
by sorry

end absolute_value_quadratic_inequality_l3025_302514


namespace subset_implies_m_value_l3025_302522

def A (m : ℝ) : Set ℝ := {1, 3, 2*m+3}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_value (m : ℝ) : B m ⊆ A m → m = 1 ∨ m = 3 := by
  sorry

end subset_implies_m_value_l3025_302522


namespace apple_savings_proof_l3025_302538

/-- The price in dollars for a pack of apples at Store 1 -/
def store1_price : ℚ := 3

/-- The number of apples in a pack at Store 1 -/
def store1_apples : ℕ := 6

/-- The price in dollars for a pack of apples at Store 2 -/
def store2_price : ℚ := 4

/-- The number of apples in a pack at Store 2 -/
def store2_apples : ℕ := 10

/-- The savings in cents per apple when buying from Store 2 instead of Store 1 -/
def savings_per_apple : ℕ := 10

theorem apple_savings_proof :
  (store1_price / store1_apples - store2_price / store2_apples) * 100 = savings_per_apple := by
  sorry

end apple_savings_proof_l3025_302538


namespace tenth_stage_toothpicks_l3025_302560

/-- The number of toothpicks in the nth stage of the sequence -/
def toothpicks (n : ℕ) : ℕ := 4 + 3 * (n - 1)

/-- The 10th stage of the sequence has 31 toothpicks -/
theorem tenth_stage_toothpicks : toothpicks 10 = 31 := by
  sorry

end tenth_stage_toothpicks_l3025_302560


namespace ascending_order_fractions_l3025_302533

theorem ascending_order_fractions (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) := by
  sorry

end ascending_order_fractions_l3025_302533


namespace absolute_value_simplification_l3025_302507

theorem absolute_value_simplification : |(-6 - 4)| = 6 + 4 := by
  sorry

end absolute_value_simplification_l3025_302507


namespace cube_side_length_l3025_302570

theorem cube_side_length (volume : ℝ) (side : ℝ) : 
  volume = 729 → side^3 = volume → side = 9 := by
  sorry

end cube_side_length_l3025_302570


namespace diophantine_equation_solutions_l3025_302562

theorem diophantine_equation_solutions :
  ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ S → p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≤ 1980 ∧ 4 * p.1^3 - 3 * p.1 + 1 = 2 * p.2^2) ∧
    S.card ≥ 31 :=
by sorry

end diophantine_equation_solutions_l3025_302562


namespace pizza_pieces_l3025_302563

theorem pizza_pieces (total_pizzas : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) : 
  total_pizzas = 4 → total_cost = 80 → cost_per_piece = 4 → 
  (total_cost / total_pizzas) / cost_per_piece = 5 := by
  sorry

end pizza_pieces_l3025_302563


namespace unique_number_l3025_302567

def is_valid_increase (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ a < 10 ∧ b < 10 ∧
  ∃ (c d : ℕ), (c = 2 ∨ c = 4) ∧ (d = 2 ∨ d = 4) ∧
  4 * n = 10 * (a + c) + (b + d)

theorem unique_number : ∀ n : ℕ, is_valid_increase n ↔ n = 14 := by sorry

end unique_number_l3025_302567


namespace triangle_area_l3025_302574

theorem triangle_area (a c : ℝ) (B : ℝ) (h1 : a = 3 * Real.sqrt 3) (h2 : c = 2) (h3 : B = π / 3) :
  (1 / 2) * a * c * Real.sin B = 9 / 2 := by
  sorry

end triangle_area_l3025_302574


namespace car_distance_traveled_l3025_302599

/-- Given a train speed and a car's relative speed to the train, 
    calculate the distance traveled by the car in a given time. -/
theorem car_distance_traveled 
  (train_speed : ℝ) 
  (car_relative_speed : ℝ) 
  (time_minutes : ℝ) : 
  train_speed = 90 →
  car_relative_speed = 2/3 →
  time_minutes = 30 →
  (car_relative_speed * train_speed) * (time_minutes / 60) = 30 := by
  sorry

end car_distance_traveled_l3025_302599


namespace transistor_count_2010_l3025_302537

def initial_year : ℕ := 1985
def final_year : ℕ := 2010
def initial_transistors : ℕ := 500000
def doubling_period : ℕ := 2

def moores_law (t : ℕ) : ℕ := initial_transistors * 2^((t - initial_year) / doubling_period)

theorem transistor_count_2010 : moores_law final_year = 2048000000 := by
  sorry

end transistor_count_2010_l3025_302537


namespace mack_journal_pages_l3025_302534

/-- Calculates the number of pages written given time and rate -/
def pages_written (time minutes_per_page : ℕ) : ℕ :=
  time / minutes_per_page

/-- Represents Mack's journal writing over four days -/
structure JournalWriting where
  monday_time : ℕ
  monday_rate : ℕ
  tuesday_time : ℕ
  tuesday_rate : ℕ
  wednesday_pages : ℕ
  thursday_time1 : ℕ
  thursday_rate1 : ℕ
  thursday_time2 : ℕ
  thursday_rate2 : ℕ

/-- Calculates the total pages written over four days -/
def total_pages (j : JournalWriting) : ℕ :=
  pages_written j.monday_time j.monday_rate +
  pages_written j.tuesday_time j.tuesday_rate +
  j.wednesday_pages +
  pages_written j.thursday_time1 j.thursday_rate1 +
  pages_written j.thursday_time2 j.thursday_rate2

/-- Theorem stating the total pages written is 16 -/
theorem mack_journal_pages :
  ∀ j : JournalWriting,
    j.monday_time = 60 ∧
    j.monday_rate = 30 ∧
    j.tuesday_time = 45 ∧
    j.tuesday_rate = 15 ∧
    j.wednesday_pages = 5 ∧
    j.thursday_time1 = 30 ∧
    j.thursday_rate1 = 10 ∧
    j.thursday_time2 = 60 ∧
    j.thursday_rate2 = 20 →
    total_pages j = 16 := by
  sorry

end mack_journal_pages_l3025_302534


namespace hexalia_base_theorem_l3025_302515

/-- Converts a number from base s to base 10 -/
def toBase10 (digits : List Nat) (s : Nat) : Nat :=
  digits.foldr (fun d acc => d + s * acc) 0

/-- The base s used in Hexalia -/
def s : Nat :=
  sorry

/-- The cost of the computer in base s -/
def cost : List Nat :=
  [5, 3, 0]

/-- The amount paid in base s -/
def paid : List Nat :=
  [1, 2, 0, 0]

/-- The change received in base s -/
def change : List Nat :=
  [4, 5, 5]

/-- Theorem stating that the base s satisfies the transaction equation -/
theorem hexalia_base_theorem :
  toBase10 cost s + toBase10 change s = toBase10 paid s ∧ s = 10 :=
sorry

end hexalia_base_theorem_l3025_302515


namespace inscribed_cube_volume_l3025_302549

/-- The volume of a cube inscribed in a sphere, which is itself inscribed in a larger cube. -/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let sphere_diameter := outer_cube_edge
  let sphere_radius := sphere_diameter / 2
  let inner_cube_diagonal := sphere_diameter
  let inner_cube_edge := inner_cube_diagonal / Real.sqrt 3
  let inner_cube_volume := inner_cube_edge ^ 3
  inner_cube_volume = 192 * Real.sqrt 3 := by
  sorry

end inscribed_cube_volume_l3025_302549


namespace max_knights_is_eight_l3025_302539

/-- Represents a person who can be either a knight or a liar -/
inductive Person
| knight
| liar

/-- The type of statements a person can make about their number -/
inductive Statement
| greater_than (n : ℕ)
| less_than (n : ℕ)

/-- A function that determines if a statement is true for a given number -/
def is_true_statement (s : Statement) (num : ℕ) : Prop :=
  match s with
  | Statement.greater_than n => num > n
  | Statement.less_than n => num < n

/-- A function that determines if a person's statements are consistent with their type -/
def consistent_statements (p : Person) (num : ℕ) (s1 s2 : Statement) : Prop :=
  match p with
  | Person.knight => is_true_statement s1 num ∧ is_true_statement s2 num
  | Person.liar => ¬(is_true_statement s1 num) ∧ ¬(is_true_statement s2 num)

theorem max_knights_is_eight :
  ∃ (people : Fin 10 → Person) (numbers : Fin 10 → ℕ) 
    (statements1 statements2 : Fin 10 → Statement),
    (∀ i : Fin 10, ∃ n : ℕ, statements1 i = Statement.greater_than n ∧ n = i.val + 1) ∧
    (∀ i : Fin 10, ∃ n : ℕ, statements2 i = Statement.less_than n ∧ n ≤ 10) ∧
    (∀ i : Fin 10, consistent_statements (people i) (numbers i) (statements1 i) (statements2 i)) ∧
    (∀ n : ℕ, n > 8 → ¬∃ (people : Fin n → Person) (numbers : Fin n → ℕ) 
      (statements1 statements2 : Fin n → Statement),
      (∀ i : Fin n, ∃ m : ℕ, statements1 i = Statement.greater_than m ∧ m = i.val + 1) ∧
      (∀ i : Fin n, ∃ m : ℕ, statements2 i = Statement.less_than m ∧ m ≤ n) ∧
      (∀ i : Fin n, consistent_statements (people i) (numbers i) (statements1 i) (statements2 i)) ∧
      (∀ i : Fin n, people i = Person.knight)) :=
by sorry

end max_knights_is_eight_l3025_302539


namespace interest_rate_calculation_l3025_302503

/-- Given a principal amount and an interest rate, if the simple interest
    for 2 years is $400 and the compound interest for 2 years is $440,
    then the interest rate is 20%. -/
theorem interest_rate_calculation (P r : ℝ) :
  P * r * 2 = 400 →
  P * ((1 + r)^2 - 1) = 440 →
  r = 0.20 := by sorry

end interest_rate_calculation_l3025_302503


namespace set_equality_gt_one_set_equality_odd_integers_l3025_302585

-- Statement 1
theorem set_equality_gt_one : {x : ℝ | x > 1} = {y : ℝ | y > 1} := by sorry

-- Statement 2
theorem set_equality_odd_integers : {x : ℤ | ∃ k : ℤ, x = 2*k + 1} = {x : ℤ | ∃ k : ℤ, x = 2*k - 1} := by sorry

end set_equality_gt_one_set_equality_odd_integers_l3025_302585


namespace speeding_ticket_problem_l3025_302573

theorem speeding_ticket_problem (total_motorists : ℕ) 
  (h1 : total_motorists > 0) 
  (exceed_limit : ℕ) 
  (receive_tickets : ℕ) 
  (h2 : exceed_limit = total_motorists * 25 / 100) 
  (h3 : receive_tickets = total_motorists * 20 / 100) :
  (exceed_limit - receive_tickets) * 100 / exceed_limit = 20 := by
sorry

end speeding_ticket_problem_l3025_302573


namespace lateral_surface_area_of_truncated_pyramid_l3025_302526

/-- A regular truncated quadrilateral pyramid with an inscribed sphere -/
structure TruncatedPyramid where
  a : ℝ  -- height of the lateral face
  inscribed_sphere : Prop  -- property that a sphere can be inscribed

/-- The lateral surface area of a truncated quadrilateral pyramid -/
def lateral_surface_area (tp : TruncatedPyramid) : ℝ :=
  4 * tp.a^2

/-- Theorem: The lateral surface area of a regular truncated quadrilateral pyramid
    with an inscribed sphere is 4a^2, where a is the height of the lateral face -/
theorem lateral_surface_area_of_truncated_pyramid (tp : TruncatedPyramid) :
  tp.inscribed_sphere → lateral_surface_area tp = 4 * tp.a^2 := by
  sorry

end lateral_surface_area_of_truncated_pyramid_l3025_302526


namespace parallel_lines_m_values_l3025_302572

-- Define the lines l₁ and l₂
def l₁ (m x y : ℝ) : Prop := 2 * x + (m + 1) * y + 4 = 0
def l₂ (m x y : ℝ) : Prop := m * x + 3 * y - 2 = 0

-- Define what it means for two lines to be parallel
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_m_values :
  ∀ (m : ℝ), parallel (l₁ m) (l₂ m) ↔ (m = -3 ∨ m = 2) :=
sorry

end parallel_lines_m_values_l3025_302572


namespace circle_equation_l3025_302555

/-- Given a circle with center (2, 1) and a line containing its common chord
    with the circle x^2 + y^2 - 3x = 0 passing through (5, -2),
    prove that the equation of the circle is (x-2)^2 + (y-1)^2 = 4 -/
theorem circle_equation (x y : ℝ) :
  let center := (2, 1)
  let known_circle := fun (x y : ℝ) => x^2 + y^2 - 3*x = 0
  let common_chord_point := (5, -2)
  let circle_eq := fun (x y : ℝ) => (x - 2)^2 + (y - 1)^2 = 4
  (∃ (line : ℝ → ℝ → Prop),
    (∀ x y, line x y ↔ known_circle x y ∨ circle_eq x y) ∧
    line common_chord_point.1 common_chord_point.2) →
  circle_eq x y :=
by sorry

end circle_equation_l3025_302555


namespace same_color_plate_probability_l3025_302556

/-- The probability of selecting 3 plates of the same color from a set of 6 red plates and 5 blue plates is 2/11. -/
theorem same_color_plate_probability :
  let total_plates : ℕ := 11
  let red_plates : ℕ := 6
  let blue_plates : ℕ := 5
  let selected_plates : ℕ := 3
  let total_combinations := Nat.choose total_plates selected_plates
  let red_combinations := Nat.choose red_plates selected_plates
  let blue_combinations := Nat.choose blue_plates selected_plates
  let same_color_combinations := red_combinations + blue_combinations
  (same_color_combinations : ℚ) / total_combinations = 2 / 11 := by
  sorry

end same_color_plate_probability_l3025_302556


namespace extreme_value_a_1_monotonicity_a_leq_neg_1_monotonicity_a_gt_neg_1_l3025_302553

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + 1 + a) / x - a * Real.log x

-- Theorem for the extreme value when a = 1
theorem extreme_value_a_1 :
  ∃ (x_min : ℝ), x_min > 0 ∧ 
  (∀ x > 0, f 1 x_min ≤ f 1 x) ∧
  f 1 x_min = Real.sqrt 2 + 3/2 - (1/2) * Real.log 2 :=
sorry

-- Theorem for monotonicity when a ≤ -1
theorem monotonicity_a_leq_neg_1 (a : ℝ) (h : a ≤ -1) :
  ∀ x y, 0 < x → 0 < y → x < y → f a x < f a y :=
sorry

-- Theorem for monotonicity when a > -1
theorem monotonicity_a_gt_neg_1 (a : ℝ) (h : a > -1) :
  (∀ x y, 0 < x → x < y → y < 1 + a → f a x > f a y) ∧
  (∀ x y, 1 + a < x → x < y → f a x < f a y) :=
sorry

end extreme_value_a_1_monotonicity_a_leq_neg_1_monotonicity_a_gt_neg_1_l3025_302553


namespace largest_percent_error_circle_area_l3025_302535

/-- The largest possible percent error in the computed area of a circle, given a measurement error in its diameter --/
theorem largest_percent_error_circle_area (actual_diameter : ℝ) (max_error_percent : ℝ) :
  actual_diameter = 20 →
  max_error_percent = 20 →
  let max_measured_diameter := actual_diameter * (1 + max_error_percent / 100)
  let actual_area := Real.pi * (actual_diameter / 2) ^ 2
  let max_computed_area := Real.pi * (max_measured_diameter / 2) ^ 2
  let max_percent_error := (max_computed_area - actual_area) / actual_area * 100
  max_percent_error = 44 := by sorry

end largest_percent_error_circle_area_l3025_302535


namespace num_lineups_eq_1782_l3025_302548

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 5 starters from a team of 15 players,
    including 4 quadruplets, with at most one quadruplet in the starting lineup -/
def num_lineups : ℕ :=
  let total_players : ℕ := 15
  let num_quadruplets : ℕ := 4
  let non_quadruplet_players : ℕ := total_players - num_quadruplets
  let starters : ℕ := 5
  (choose non_quadruplet_players starters) +
  (num_quadruplets * choose non_quadruplet_players (starters - 1))

theorem num_lineups_eq_1782 : num_lineups = 1782 := by
  sorry

end num_lineups_eq_1782_l3025_302548


namespace min_skittles_proof_l3025_302598

def min_skittles : ℕ := 150

theorem min_skittles_proof :
  (∀ n : ℕ, n ≥ min_skittles ∧ n % 19 = 17 → n ≥ min_skittles) ∧
  min_skittles % 19 = 17 :=
sorry

end min_skittles_proof_l3025_302598


namespace rice_trader_problem_l3025_302524

/-- A rice trader problem -/
theorem rice_trader_problem (initial_stock restocked final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  ∃ (sold : ℕ), initial_stock - sold + restocked = final_stock ∧ sold = 23 := by
  sorry

end rice_trader_problem_l3025_302524


namespace polygon_sides_difference_l3025_302566

/-- Number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The problem statement -/
theorem polygon_sides_difference : ∃ (m : ℕ),
  m > 3 ∧
  diagonals m = 3 * diagonals (m - 3) ∧
  ∃ (x : ℕ),
    diagonals (m + x) = 7 * diagonals m ∧
    x = 12 := by
  sorry

end polygon_sides_difference_l3025_302566


namespace unique_solution_cubic_l3025_302511

theorem unique_solution_cubic (c : ℝ) : c = 3/4 ↔ 
  ∃! (b : ℝ), b > 0 ∧ 
    ∃! (x : ℝ), x^3 + x^2 + (b^2 + 1/b^2) * x + c = 0 :=
by sorry

end unique_solution_cubic_l3025_302511


namespace triangle_abc_degenerate_l3025_302561

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parabola defined by y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- A horizontal line defined by y = 2 -/
def HorizontalLine := {p : Point | p.y = 2}

/-- Theorem: The intersection of the horizontal line y = 2 and the parabola y^2 = 4x 
    results in a point that coincides with A(1,2), making triangle ABC degenerate -/
theorem triangle_abc_degenerate (A : Point) (h1 : A.x = 1) (h2 : A.y = 2) :
  ∃ (B : Point), B ∈ Parabola ∧ B ∈ HorizontalLine ∧ B = A :=
sorry

end triangle_abc_degenerate_l3025_302561


namespace two_valid_solutions_l3025_302506

def original_number : ℕ := 20192020

def is_valid (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ (a * 1000000000 + original_number * 10 + b) % 72 = 0

theorem two_valid_solutions :
  ∃! (s : Set (ℕ × ℕ)), s = {(2, 0), (3, 8)} ∧ 
    ∀ (a b : ℕ), (a, b) ∈ s ↔ is_valid a b :=
sorry

end two_valid_solutions_l3025_302506


namespace hyperbola_other_asymptote_l3025_302525

/-- Given a hyperbola with one asymptote y = -2x + 5 and foci with x-coordinate 2,
    prove that the equation of the other asymptote is y = 2x - 3 -/
theorem hyperbola_other_asymptote (x y : ℝ) :
  let asymptote1 : ℝ → ℝ := λ x => -2 * x + 5
  let foci_x : ℝ := 2
  let center_x : ℝ := foci_x
  let center_y : ℝ := asymptote1 center_x
  let asymptote2 : ℝ → ℝ := λ x => 2 * x - 3
  (∀ x, y = asymptote1 x) → (y = asymptote2 x) := by
sorry

end hyperbola_other_asymptote_l3025_302525


namespace imaginary_sum_equals_4_minus_4i_l3025_302564

theorem imaginary_sum_equals_4_minus_4i :
  let i : ℂ := Complex.I
  (i + 2 * i^2 + 3 * i^3 + 4 * i^4 + 5 * i^5 + 6 * i^6 + 7 * i^7 + 8 * i^8) = (4 : ℂ) - 4 * i :=
by sorry

end imaginary_sum_equals_4_minus_4i_l3025_302564


namespace circle_center_radius_sum_l3025_302554

theorem circle_center_radius_sum (x y : ℝ) : 
  x^2 - 16*x + y^2 - 18*y = -81 → 
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ a + b + r = 17 + Real.sqrt 145 := by
sorry

end circle_center_radius_sum_l3025_302554
