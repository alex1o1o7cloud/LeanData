import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l2589_258974

theorem unique_solution_power_equation :
  ∃! (a b c d : ℕ), 7^a = 4^b + 5^c + 6^d :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l2589_258974


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2589_258921

theorem complex_equation_solution :
  ∃ (z : ℂ), (4 : ℂ) - 3 * Complex.I * z = (2 : ℂ) + 5 * Complex.I * z ∧ z = -(1/4) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2589_258921


namespace NUMINAMATH_CALUDE_no_common_root_l2589_258968

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ x : ℝ, x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_root_l2589_258968


namespace NUMINAMATH_CALUDE_strawberry_weight_theorem_l2589_258909

/-- The weight of Marco's strawberries in pounds -/
def marco_weight : ℕ := 19

/-- The difference in weight between Marco's dad's strawberries and Marco's strawberries in pounds -/
def weight_difference : ℕ := 34

/-- The weight of Marco's dad's strawberries in pounds -/
def dad_weight : ℕ := marco_weight + weight_difference

/-- The total weight of Marco's and his dad's strawberries in pounds -/
def total_weight : ℕ := marco_weight + dad_weight

theorem strawberry_weight_theorem :
  total_weight = 72 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_theorem_l2589_258909


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l2589_258931

/-- The number of ways to arrange 5 people in a row with one person between A and B -/
def arrangement_count : ℕ := 36

/-- The total number of people in the arrangement -/
def total_people : ℕ := 5

/-- The number of people between A and B -/
def people_between : ℕ := 1

theorem arrangement_count_is_correct :
  arrangement_count = 
    2 * (total_people - 2) * (Nat.factorial (total_people - 2)) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_correct_l2589_258931


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2589_258900

def quadratic_equation (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x^2 + 3 * x + k^2 - 4

theorem unique_solution_quadratic (k : ℝ) :
  (quadratic_equation k 0 = 0) →
  (∃! x, quadratic_equation k x = 0) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2589_258900


namespace NUMINAMATH_CALUDE_defeated_candidate_vote_percentage_l2589_258971

theorem defeated_candidate_vote_percentage 
  (total_polled_votes : ℕ) 
  (invalid_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_polled_votes = 90083) 
  (h2 : invalid_votes = 83) 
  (h3 : vote_difference = 9000) : 
  let valid_votes := total_polled_votes - invalid_votes
  let defeated_votes := (valid_votes - vote_difference) / 2
  defeated_votes * 100 / valid_votes = 45 := by
sorry

end NUMINAMATH_CALUDE_defeated_candidate_vote_percentage_l2589_258971


namespace NUMINAMATH_CALUDE_not_always_same_digit_sum_l2589_258946

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), ∃ (k : ℕ), sum_of_digits (N + k * M) ≠ sum_of_digits N :=
sorry

end NUMINAMATH_CALUDE_not_always_same_digit_sum_l2589_258946


namespace NUMINAMATH_CALUDE_scores_mode_and_median_l2589_258991

def scores : List ℕ := [80, 85, 85, 85, 90, 90, 90, 90, 95]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem scores_mode_and_median :
  mode scores = 90 ∧ median scores = 90 := by sorry

end NUMINAMATH_CALUDE_scores_mode_and_median_l2589_258991


namespace NUMINAMATH_CALUDE_prob_no_female_ends_correct_l2589_258961

/-- The number of male students -/
def num_male : ℕ := 3

/-- The number of female students -/
def num_female : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The probability that neither end is a female student when arranging the students in a row -/
def prob_no_female_ends : ℚ := 1 / 5

theorem prob_no_female_ends_correct :
  (num_male.choose 2 * (total_students - 2).factorial) / total_students.factorial = prob_no_female_ends :=
sorry

end NUMINAMATH_CALUDE_prob_no_female_ends_correct_l2589_258961


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l2589_258958

/-- Represents a rectangular rug with three nested regions -/
structure Rug where
  inner_width : ℝ
  inner_length : ℝ
  middle_width : ℝ
  middle_length : ℝ
  outer_width : ℝ
  outer_length : ℝ

/-- Calculates the area of a rectangle -/
def area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four (r : Rug) : 
  r.inner_width = 2 ∧ 
  r.middle_width = r.inner_width + 4 ∧ 
  r.outer_width = r.middle_width + 4 ∧
  r.middle_length = r.inner_length + 4 ∧
  r.outer_length = r.middle_length + 4 ∧
  isArithmeticProgression 
    (area r.inner_width r.inner_length)
    (area r.middle_width r.middle_length - area r.inner_width r.inner_length)
    (area r.outer_width r.outer_length - area r.middle_width r.middle_length) →
  r.inner_length = 4 := by
sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l2589_258958


namespace NUMINAMATH_CALUDE_automobile_distance_l2589_258908

/-- Proves that an automobile traveling a/4 feet in r seconds will cover 20a/r yards in 4 minutes if it maintains the same rate. -/
theorem automobile_distance (a r : ℝ) (ha : a > 0) (hr : r > 0) : 
  let rate_feet_per_second : ℝ := a / (4 * r)
  let rate_yards_per_second : ℝ := rate_feet_per_second / 3
  let time_in_seconds : ℝ := 4 * 60
  rate_yards_per_second * time_in_seconds = 20 * a / r :=
by sorry

end NUMINAMATH_CALUDE_automobile_distance_l2589_258908


namespace NUMINAMATH_CALUDE_factorization_equality_l2589_258916

-- Define the theorem
theorem factorization_equality {R : Type*} [Ring R] (a b : R) :
  2 * a^2 - a * b = a * (2 * a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2589_258916


namespace NUMINAMATH_CALUDE_movie_ticket_change_change_is_nine_l2589_258973

/-- The change received by two sisters after buying movie tickets -/
theorem movie_ticket_change (ticket_cost : ℕ) (money_brought : ℕ) : ℕ :=
  let num_sisters : ℕ := 2
  let total_cost : ℕ := num_sisters * ticket_cost
  money_brought - total_cost

/-- Proof that the change received is $9 -/
theorem change_is_nine :
  movie_ticket_change 8 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_change_change_is_nine_l2589_258973


namespace NUMINAMATH_CALUDE_solve_for_a_l2589_258926

theorem solve_for_a : ∃ a : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → 3 * x - a * y = 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l2589_258926


namespace NUMINAMATH_CALUDE_track_circumference_l2589_258934

/-- Represents a circular track with two runners -/
structure CircularTrack where
  /-- The circumference of the track in yards -/
  circumference : ℝ
  /-- The distance B travels before the first meeting in yards -/
  first_meeting_distance : ℝ
  /-- The distance A has left to complete a lap at the second meeting in yards -/
  second_meeting_remaining : ℝ

/-- The theorem stating the circumference of the track given the conditions -/
theorem track_circumference (track : CircularTrack)
  (h1 : track.first_meeting_distance = 150)
  (h2 : track.second_meeting_remaining = 90)
  (h3 : track.first_meeting_distance < track.circumference / 2)
  (h4 : track.second_meeting_remaining < track.circumference) :
  track.circumference = 720 := by
  sorry

#check track_circumference

end NUMINAMATH_CALUDE_track_circumference_l2589_258934


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l2589_258986

theorem smallest_positive_integer_with_given_remainders : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 3 ∧ 
  x % 8 = 5 ∧
  (∀ y : ℕ, y > 0 ∧ y % 6 = 3 ∧ y % 8 = 5 → x ≤ y) ∧
  x = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_given_remainders_l2589_258986


namespace NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l2589_258970

/-- For a complex number z = re^(iθ) where |z| = r ≠ 1 and r > 0, 
    the real part of 1 / (1 - z^2) is (1 - r^2 cos(2θ)) / (1 - 2r^2 cos(2θ) + r^4) -/
theorem real_part_of_inverse_one_minus_z_squared 
  (z : ℂ) (r θ : ℝ) (h1 : z = r * Complex.exp (θ * Complex.I)) 
  (h2 : Complex.abs z = r) (h3 : r ≠ 1) (h4 : r > 0) : 
  (1 / (1 - z^2)).re = (1 - r^2 * Real.cos (2 * θ)) / (1 - 2 * r^2 * Real.cos (2 * θ) + r^4) := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_inverse_one_minus_z_squared_l2589_258970


namespace NUMINAMATH_CALUDE_sum_of_integers_l2589_258943

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2589_258943


namespace NUMINAMATH_CALUDE_triangle_area_72_l2589_258980

theorem triangle_area_72 (x : ℝ) (h1 : x > 0) 
  (h2 : (1/2) * (2*x) * x = 72) : x = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_72_l2589_258980


namespace NUMINAMATH_CALUDE_relay_game_error_l2589_258962

def initial_equation (x : ℝ) : Prop :=
  3 / (x - 1) = 1 - x / (x + 1)

def step1 (x : ℝ) : Prop :=
  3 * (x + 1) = (x + 1) * (x - 1) - x * (x - 1)

def step2 (x : ℝ) : Prop :=
  3 * x + 3 = x^2 + 1 - x^2 + x

def step3 (x : ℝ) : Prop :=
  3 * x - x = 1 - 3

def step4 (x : ℝ) : Prop :=
  x = -1

theorem relay_game_error :
  ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 →
    (initial_equation x ↔ step1 x) ∧
    ¬(initial_equation x ↔ step2 x) ∧
    (initial_equation x ↔ step3 x) ∧
    (initial_equation x ↔ step4 x) :=
by sorry

end NUMINAMATH_CALUDE_relay_game_error_l2589_258962


namespace NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l2589_258933

/-- Given that x varies inversely as the square of y and directly as the cube of z,
    prove that x = 64/243 when y = 6 and z = 4, given the initial conditions x = 1, y = 2, and z = 3. -/
theorem inverse_square_direct_cube_relation
  (k : ℚ)
  (h : ∀ (x y z : ℚ), x = k * z^3 / y^2)
  (h_init : 1 = k * 3^3 / 2^2) :
  k * 4^3 / 6^2 = 64/243 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_direct_cube_relation_l2589_258933


namespace NUMINAMATH_CALUDE_answer_key_combinations_l2589_258989

/-- Represents the number of answer choices for a multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- Represents the number of true-false questions -/
def true_false_questions : ℕ := 3

/-- Represents the number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 3

/-- Calculates the number of valid true-false combinations -/
def valid_true_false_combinations : ℕ := 2^true_false_questions - 2

/-- Calculates the number of multiple-choice combinations -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- Theorem: The number of ways to create an answer key for the quiz is 384 -/
theorem answer_key_combinations : 
  valid_true_false_combinations * multiple_choice_combinations = 384 := by
  sorry


end NUMINAMATH_CALUDE_answer_key_combinations_l2589_258989


namespace NUMINAMATH_CALUDE_gcf_360_180_l2589_258951

theorem gcf_360_180 : Nat.gcd 360 180 = 180 := by
  sorry

end NUMINAMATH_CALUDE_gcf_360_180_l2589_258951


namespace NUMINAMATH_CALUDE_g_of_negative_four_l2589_258957

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem g_of_negative_four : g (-4) = -18 := by
  sorry

end NUMINAMATH_CALUDE_g_of_negative_four_l2589_258957


namespace NUMINAMATH_CALUDE_machine_no_repair_l2589_258911

/-- Represents the state of a portion measuring machine -/
structure PortionMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ
  standard_deviation : ℝ

/-- Determines if a portion measuring machine requires repair -/
def requires_repair (m : PortionMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨
  m.unreadable_deviation_bound ≥ m.max_deviation ∨
  m.standard_deviation > m.max_deviation

/-- Theorem stating that the given machine does not require repair -/
theorem machine_no_repair (m : PortionMachine)
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation)
  (h4 : m.standard_deviation ≤ m.max_deviation) :
  ¬(requires_repair m) :=
sorry

end NUMINAMATH_CALUDE_machine_no_repair_l2589_258911


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2589_258965

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point of tangency
def P : ℝ × ℝ := (1, 1)

-- Define the slope of the tangent line at P
def m : ℝ := 3

-- Statement of the theorem
theorem tangent_line_equation :
  ∀ x y : ℝ, (x - P.1) * m = y - P.2 ↔ 3*x - y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2589_258965


namespace NUMINAMATH_CALUDE_inequality_relation_l2589_258906

theorem inequality_relation : 
  ∃ (x : ℝ), (x^2 - x - 6 > 0 ∧ x ≥ -5) ∧ 
  ∀ (y : ℝ), y < -5 → y^2 - y - 6 > 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l2589_258906


namespace NUMINAMATH_CALUDE_factorial_divisibility_theorem_l2589_258905

def factorial (n : ℕ) : ℕ := Nat.factorial n

def sum_factorials (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => factorial (i + 1))

theorem factorial_divisibility_theorem :
  ∀ n : ℕ, n > 2 → ¬(factorial (n + 1) ∣ sum_factorials n) ∧
  (factorial 2 ∣ sum_factorials 1) ∧
  (factorial 3 ∣ sum_factorials 2) ∧
  ∀ m : ℕ, m ≠ 1 ∧ m ≠ 2 → ¬(factorial (m + 1) ∣ sum_factorials m) :=
by sorry

end NUMINAMATH_CALUDE_factorial_divisibility_theorem_l2589_258905


namespace NUMINAMATH_CALUDE_round_robin_cyclic_triples_l2589_258988

/-- Represents a round-robin tournament. -/
structure Tournament where
  teams : ℕ
  games_won : ℕ
  games_lost : ℕ

/-- Represents a cyclic triple in the tournament. -/
structure CyclicTriple where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The number of cyclic triples in the tournament. -/
def count_cyclic_triples (t : Tournament) : ℕ :=
  sorry

theorem round_robin_cyclic_triples :
  ∀ t : Tournament,
    t.teams = t.games_won + t.games_lost + 1 →
    t.games_won = 12 →
    t.games_lost = 8 →
    count_cyclic_triples t = 144 :=
  sorry

end NUMINAMATH_CALUDE_round_robin_cyclic_triples_l2589_258988


namespace NUMINAMATH_CALUDE_triangle_area_tripled_sides_l2589_258913

/-- Given a triangle with sides a and b and included angle θ,
    if we triple the sides to 3a and 3b while keeping θ unchanged,
    then the new area A' is 9 times the original area A. -/
theorem triangle_area_tripled_sides (a b θ : ℝ) (ha : a > 0) (hb : b > 0) (hθ : 0 < θ ∧ θ < π) :
  let A := (a * b * Real.sin θ) / 2
  let A' := (3 * a * 3 * b * Real.sin θ) / 2
  A' = 9 * A := by sorry

end NUMINAMATH_CALUDE_triangle_area_tripled_sides_l2589_258913


namespace NUMINAMATH_CALUDE_edwards_earnings_l2589_258929

/-- Edward's lawn mowing earnings problem -/
theorem edwards_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  rate = 4 →
  total_lawns = 17 →
  forgotten_lawns = 9 →
  rate * (total_lawns - forgotten_lawns) = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_edwards_earnings_l2589_258929


namespace NUMINAMATH_CALUDE_product_sum_theorem_l2589_258981

theorem product_sum_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = 1)
  (eq2 : a + b + d = 5)
  (eq3 : a + c + d = 20)
  (eq4 : b + c + d = 15) :
  a * b + c * d = 1002 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l2589_258981


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2589_258928

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) :
  a > 1 ∨ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l2589_258928


namespace NUMINAMATH_CALUDE_three_As_theorem_l2589_258940

-- Define the set of students
inductive Student : Type
  | Alan : Student
  | Beth : Student
  | Carlos : Student
  | Diana : Student
  | Emma : Student

-- Define a function to represent whether a student got an A
def gotA : Student → Prop := sorry

-- Define the implications stated by each student
axiom alan_implication : gotA Student.Alan → gotA Student.Beth
axiom beth_implication : gotA Student.Beth → (gotA Student.Carlos ∧ gotA Student.Emma)
axiom carlos_implication : gotA Student.Carlos → gotA Student.Diana
axiom diana_implication : gotA Student.Diana → gotA Student.Emma

-- Define a function to count how many students got an A
def count_A : (Student → Prop) → Nat := sorry

-- State the theorem
theorem three_As_theorem :
  (count_A gotA = 3) →
  ((gotA Student.Beth ∧ gotA Student.Carlos ∧ gotA Student.Emma) ∨
   (gotA Student.Carlos ∧ gotA Student.Diana ∧ gotA Student.Emma)) :=
by sorry

end NUMINAMATH_CALUDE_three_As_theorem_l2589_258940


namespace NUMINAMATH_CALUDE_population_change_l2589_258907

theorem population_change (initial_population : ℕ) 
  (increase_rate : ℚ) (decrease_rate : ℚ) : 
  initial_population = 10000 →
  increase_rate = 20 / 100 →
  decrease_rate = 20 / 100 →
  (initial_population * (1 + increase_rate) * (1 - decrease_rate)).floor = 9600 := by
sorry

end NUMINAMATH_CALUDE_population_change_l2589_258907


namespace NUMINAMATH_CALUDE_race_head_start_l2589_258969

theorem race_head_start (va vb : ℝ) (h : va = 20/15 * vb) :
  let x : ℝ := 1/4
  ∀ L : ℝ, L > 0 → L / va = (L - x * L) / vb :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l2589_258969


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l2589_258994

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_condition (m : ℝ) :
  let z : ℂ := Complex.mk (m + 1) (m - 1)
  is_purely_imaginary z → m = -1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_l2589_258994


namespace NUMINAMATH_CALUDE_no_consecutive_ones_eq_fib_l2589_258953

/-- The number of binary sequences of length n with no two consecutive 1s -/
def no_consecutive_ones (n : ℕ) : ℕ :=
  sorry

/-- The nth Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: The number of binary sequences of length n with no two consecutive 1s
    is equal to the (n+2)th Fibonacci number -/
theorem no_consecutive_ones_eq_fib (n : ℕ) : no_consecutive_ones n = fib (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_ones_eq_fib_l2589_258953


namespace NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l2589_258932

/-- Given a parabola y² = 4x and a line passing through (1,0) intersecting the parabola at A and B,
    prove that OB · OC = -5, where C is symmetric to A with respect to the y-axis -/
theorem parabola_line_intersection_dot_product :
  ∀ (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ),
  -- Line passes through (1,0)
  y₁ = k * (x₁ - 1) →
  y₂ = k * (x₂ - 1) →
  -- A and B are on the parabola
  y₁^2 = 4*x₁ →
  y₂^2 = 4*x₂ →
  -- A and B are distinct points
  x₁ ≠ x₂ →
  -- C is symmetric to A with respect to y-axis
  let xc := -x₁
  let yc := y₁
  -- OB · OC = -5
  x₂ * xc + y₂ * yc = -5 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l2589_258932


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_equals_three_l2589_258984

-- Define a piecewise linear function with five segments
def f : ℝ → ℝ := sorry

-- Theorem statement
theorem sum_of_x_coordinates_equals_three :
  ∃ (x₁ x₂ x₃ : ℝ), 
    (f x₁ = x₁ + 1) ∧ 
    (f x₂ = x₂ + 1) ∧ 
    (f x₃ = x₃ + 1) ∧ 
    (x₁ + x₂ + x₃ = 3) ∧
    (∀ x : ℝ, f x = x + 1 → (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_equals_three_l2589_258984


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l2589_258990

theorem z_in_first_quadrant (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l2589_258990


namespace NUMINAMATH_CALUDE_checkerboard_exists_l2589_258938

/-- Represents a cell on the board -/
inductive Cell
| Black
| White

/-- Represents the board -/
def Board := Fin 100 → Fin 100 → Cell

/-- Checks if a cell is adjacent to the border -/
def isBorderAdjacent (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Cell, 
    board i j = c ∧ board (i+1) j = c ∧ 
    board i (j+1) = c ∧ board (i+1) (j+1) = c

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard (board : Board) (i j : Fin 100) : Prop :=
  (board i j = board (i+1) (j+1) ∧ board (i+1) j = board i (j+1)) ∧
  (board i j ≠ board (i+1) j)

/-- The main theorem -/
theorem checkerboard_exists (board : Board) 
  (border_black : ∀ i j : Fin 100, isBorderAdjacent i j → board i j = Cell.Black)
  (no_monochromatic : ∀ i j : Fin 100, ¬isMonochromatic board i j) :
  ∃ i j : Fin 100, isCheckerboard board i j :=
sorry

end NUMINAMATH_CALUDE_checkerboard_exists_l2589_258938


namespace NUMINAMATH_CALUDE_unique_positive_solution_exists_distinct_real_solution_l2589_258959

-- Define the system of equations
def equation_system (x y z : ℝ) : Prop :=
  x * y + y * z + z * x = 12 ∧ x * y * z - x - y - z = 2

-- Theorem for unique positive solution
theorem unique_positive_solution :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation_system x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

-- Theorem for existence of distinct real solution
theorem exists_distinct_real_solution :
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ equation_system x y z :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_exists_distinct_real_solution_l2589_258959


namespace NUMINAMATH_CALUDE_convex_pentagon_arithmetic_angles_l2589_258955

/-- A convex pentagon with angles in arithmetic progression has each angle greater than 36° -/
theorem convex_pentagon_arithmetic_angles (α γ : ℝ) (h_convex : α + 4*γ < π) 
  (h_sum : 5*α + 10*γ = 3*π) : α > π/5 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_arithmetic_angles_l2589_258955


namespace NUMINAMATH_CALUDE_yellow_leaves_count_l2589_258998

theorem yellow_leaves_count (thursday_leaves friday_leaves saturday_leaves : ℕ)
  (thursday_brown_percent thursday_green_percent : ℚ)
  (friday_brown_percent friday_green_percent : ℚ)
  (saturday_brown_percent saturday_green_percent : ℚ)
  (h1 : thursday_leaves = 15)
  (h2 : friday_leaves = 22)
  (h3 : saturday_leaves = 30)
  (h4 : thursday_brown_percent = 25/100)
  (h5 : thursday_green_percent = 40/100)
  (h6 : friday_brown_percent = 30/100)
  (h7 : friday_green_percent = 20/100)
  (h8 : saturday_brown_percent = 15/100)
  (h9 : saturday_green_percent = 50/100) :
  ⌊thursday_leaves * (1 - thursday_brown_percent - thursday_green_percent)⌋ +
  ⌊friday_leaves * (1 - friday_brown_percent - friday_green_percent)⌋ +
  ⌊saturday_leaves * (1 - saturday_brown_percent - saturday_green_percent)⌋ = 26 := by
sorry

end NUMINAMATH_CALUDE_yellow_leaves_count_l2589_258998


namespace NUMINAMATH_CALUDE_cubic_expression_evaluation_l2589_258941

theorem cubic_expression_evaluation : 
  let a : ℝ := 6
  let b : ℝ := 3
  let c : ℝ := 2
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2) = 10.04 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_evaluation_l2589_258941


namespace NUMINAMATH_CALUDE_line_equation_l2589_258903

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (0, 2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 2

-- Define a point on both the line and the parabola
def intersection_point (k x y : ℝ) : Prop :=
  parabola x y ∧ line k x y

-- Define the condition for a circle passing through three points
def circle_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem line_equation :
  ∀ k : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    x1 ≠ x2 ∧
    intersection_point k x1 y1 ∧
    intersection_point k x2 y2 ∧
    circle_condition x1 y1 x2 y2) →
  k = -1 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l2589_258903


namespace NUMINAMATH_CALUDE_variance_scaling_l2589_258996

-- Define a function to calculate variance
def variance (data : List ℝ) : ℝ := sorry

-- Define a function to scale a list of real numbers
def scaleList (k : ℝ) (data : List ℝ) : List ℝ := sorry

theorem variance_scaling (data : List ℝ) (h : variance data = 0.01) :
  variance (scaleList 10 data) = 1 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l2589_258996


namespace NUMINAMATH_CALUDE_share_calculation_l2589_258919

theorem share_calculation (total_amount : ℕ) (ratio_a ratio_b ratio_c ratio_d : ℕ) : 
  total_amount = 15800 → 
  ratio_a = 5 →
  ratio_b = 9 →
  ratio_c = 6 →
  ratio_d = 5 →
  (ratio_a * total_amount / (ratio_a + ratio_b + ratio_c + ratio_d) + 
   ratio_c * total_amount / (ratio_a + ratio_b + ratio_c + ratio_d)) = 6952 := by
sorry

end NUMINAMATH_CALUDE_share_calculation_l2589_258919


namespace NUMINAMATH_CALUDE_impossible_odd_sum_arrangement_l2589_258967

theorem impossible_odd_sum_arrangement : 
  ¬ ∃ (seq : Fin 2018 → ℕ), 
    (∀ i : Fin 2018, 1 ≤ seq i ∧ seq i ≤ 2018) ∧ 
    (∀ i : Fin 2018, seq i ≠ seq ((i + 1) % 2018)) ∧
    (∀ i : Fin 2018, Odd (seq i + seq ((i + 1) % 2018) + seq ((i + 2) % 2018))) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_odd_sum_arrangement_l2589_258967


namespace NUMINAMATH_CALUDE_class_ratio_theorem_l2589_258982

theorem class_ratio_theorem (boys girls : ℕ) (h : boys * 7 = girls * 8) :
  -- 1. The number of girls is 7/8 of the number of boys
  (girls : ℚ) / boys = 7 / 8 ∧
  -- 2. The number of boys accounts for 8/15 of the total number of students
  (boys : ℚ) / (boys + girls) = 8 / 15 ∧
  -- 3. The number of girls accounts for 7/15 of the total number of students
  (girls : ℚ) / (boys + girls) = 7 / 15 ∧
  -- 4. If there are 45 students in total, there are 24 boys
  (boys + girls = 45 → boys = 24) :=
by sorry

end NUMINAMATH_CALUDE_class_ratio_theorem_l2589_258982


namespace NUMINAMATH_CALUDE_sequence_general_term_l2589_258922

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = (3/2)a_n - 3,
    prove that the general term formula is a_n = 2 * 3^n. -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (3/2) * a n - 3) →
  ∃ C, ∀ n, a n = C * 3^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2589_258922


namespace NUMINAMATH_CALUDE_large_monkey_doll_cost_l2589_258995

/-- The cost of a large monkey doll satisfies the given conditions --/
theorem large_monkey_doll_cost : ∃ (L : ℝ), 
  (L > 0) ∧ 
  (300 / (L - 2) = 300 / L + 25) ∧ 
  (L = 6) := by
sorry

end NUMINAMATH_CALUDE_large_monkey_doll_cost_l2589_258995


namespace NUMINAMATH_CALUDE_third_square_side_length_l2589_258979

/-- Given three squares with perimeters 60 cm, 48 cm, and 36 cm respectively,
    if the area of the third square is equal to the difference of the areas of the first two squares,
    then the side length of the third square is 9 cm. -/
theorem third_square_side_length 
  (s1 s2 s3 : ℝ) 
  (h1 : 4 * s1 = 60) 
  (h2 : 4 * s2 = 48) 
  (h3 : 4 * s3 = 36) 
  (h4 : s3^2 = s1^2 - s2^2) : 
  s3 = 9 := by
sorry

end NUMINAMATH_CALUDE_third_square_side_length_l2589_258979


namespace NUMINAMATH_CALUDE_infinite_pairs_with_difference_one_l2589_258985

-- Define the property of being tuanis
def is_tuanis (a b : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (a + b).digits 10 → d = 0 ∨ d = 1

-- Define the sets A and B
def tuanis_set (A B : Set ℕ) : Prop :=
  (∀ a ∈ A, ∃ b ∈ B, is_tuanis a b) ∧
  (∀ b ∈ B, ∃ a ∈ A, is_tuanis a b)

-- The main theorem
theorem infinite_pairs_with_difference_one
  (A B : Set ℕ) (hA : Set.Infinite A) (hB : Set.Infinite B)
  (h_tuanis : tuanis_set A B) :
  (Set.Infinite {p : ℕ × ℕ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 = 1}) ∨
  (Set.Infinite {p : ℕ × ℕ | p.1 ∈ B ∧ p.2 ∈ B ∧ p.1 - p.2 = 1}) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_with_difference_one_l2589_258985


namespace NUMINAMATH_CALUDE_christen_peeled_24_l2589_258914

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initialPile : ℕ
  homerRate : ℕ
  christenJoinTime : ℕ
  christenRate : ℕ
  alexExtra : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledCount (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 24 potatoes in the given scenario -/
theorem christen_peeled_24 (scenario : PotatoPeeling) 
  (h1 : scenario.initialPile = 60)
  (h2 : scenario.homerRate = 4)
  (h3 : scenario.christenJoinTime = 6)
  (h4 : scenario.christenRate = 6)
  (h5 : scenario.alexExtra = 2) :
  christenPeeledCount scenario = 24 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_24_l2589_258914


namespace NUMINAMATH_CALUDE_add_three_people_to_two_rows_l2589_258948

/-- The number of ways to add three people to two rows of people -/
def add_people_ways (front_row : ℕ) (back_row : ℕ) (people_to_add : ℕ) : ℕ :=
  (people_to_add) * (front_row + 1) * (back_row + 1) * (back_row + 2)

/-- Theorem: The number of ways to add three people to two rows with 3 in front and 4 in back is 360 -/
theorem add_three_people_to_two_rows :
  add_people_ways 3 4 3 = 360 := by
  sorry

end NUMINAMATH_CALUDE_add_three_people_to_two_rows_l2589_258948


namespace NUMINAMATH_CALUDE_store_fruit_cost_l2589_258983

/-- The cost of fruit in a store -/
structure FruitCost where
  banana_to_apple : ℚ  -- Ratio of banana cost to apple cost
  apple_to_orange : ℚ  -- Ratio of apple cost to orange cost

/-- Given the cost ratios, calculate how many oranges cost the same as a given number of bananas -/
def bananas_to_oranges (cost : FruitCost) (num_bananas : ℕ) : ℚ :=
  (num_bananas : ℚ) * cost.apple_to_orange * cost.banana_to_apple

theorem store_fruit_cost (cost : FruitCost) 
  (h1 : cost.banana_to_apple = 3 / 4)
  (h2 : cost.apple_to_orange = 5 / 7) :
  bananas_to_oranges cost 28 = 15 := by
  sorry

end NUMINAMATH_CALUDE_store_fruit_cost_l2589_258983


namespace NUMINAMATH_CALUDE_sum_of_roots_greater_than_four_l2589_258977

/-- Given a function f(x) = x - 1 + a*exp(x), prove that the sum of its roots is greater than 4 -/
theorem sum_of_roots_greater_than_four (a : ℝ) :
  let f := λ x : ℝ => x - 1 + a * Real.exp x
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_greater_than_four_l2589_258977


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2589_258972

theorem complex_magnitude_problem :
  let z : ℂ := ((1 - 4*I) * (1 + I) + 2 + 4*I) / (3 + 4*I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2589_258972


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2589_258924

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 4, 5}

theorem intersection_complement_equality : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2589_258924


namespace NUMINAMATH_CALUDE_min_omega_value_l2589_258904

theorem min_omega_value (f : ℝ → ℝ) (ω φ : ℝ) : 
  (∀ x, f x = Real.sin (ω * x + φ)) →
  ω > 0 →
  abs φ < π / 2 →
  f 0 = 1 / 2 →
  (∀ x, f x ≤ f (π / 12)) →
  ω ≥ 4 ∧ (∀ ω', ω' > 0 ∧ ω' < 4 → 
    ∃ φ', abs φ' < π / 2 ∧ 
    Real.sin φ' = 1 / 2 ∧ 
    ∃ x, Real.sin (ω' * x + φ') > Real.sin (ω' * π / 12 + φ')) :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l2589_258904


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l2589_258949

def gain_percent (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

theorem cycle_gain_percent :
  let cost_price : ℚ := 900
  let selling_price : ℚ := 1150
  gain_percent cost_price selling_price = (1150 - 900) / 900 * 100 := by
  sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l2589_258949


namespace NUMINAMATH_CALUDE_linear_equation_solution_comparison_l2589_258915

theorem linear_equation_solution_comparison
  (c c' d d' : ℝ)
  (hc_pos : c > 0)
  (hc'_pos : c' > 0)
  (hc_gt_c' : c > c') :
  ((-d) / c < (-d') / c') ↔ (c * d' < c' * d) := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_comparison_l2589_258915


namespace NUMINAMATH_CALUDE_carrot_to_green_bean_ratio_l2589_258930

/-- Given a grocery bag with a maximum capacity and known weights of items,
    prove that the ratio of carrots to green beans is 1:2. -/
theorem carrot_to_green_bean_ratio
  (bag_capacity : ℕ)
  (green_beans : ℕ)
  (milk : ℕ)
  (remaining_capacity : ℕ)
  (h1 : bag_capacity = 20)
  (h2 : green_beans = 4)
  (h3 : milk = 6)
  (h4 : remaining_capacity = 2)
  (h5 : green_beans + milk + remaining_capacity = bag_capacity) :
  (bag_capacity - green_beans - milk) / green_beans = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_carrot_to_green_bean_ratio_l2589_258930


namespace NUMINAMATH_CALUDE_one_third_percent_of_150_l2589_258999

theorem one_third_percent_of_150 : (1 / 3 : ℚ) / 100 * 150 = 0.5 := by sorry

end NUMINAMATH_CALUDE_one_third_percent_of_150_l2589_258999


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l2589_258950

theorem cos_four_arccos_two_fifths : 
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l2589_258950


namespace NUMINAMATH_CALUDE_trig_matrix_det_zero_l2589_258942

/-- The determinant of a 3x3 matrix with specific trigonometric entries is zero -/
theorem trig_matrix_det_zero (a b φ : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := λ i j => 
    match i, j with
    | 0, 0 => 1
    | 0, 1 => Real.sin (a - b + φ)
    | 0, 2 => Real.sin a
    | 1, 0 => Real.sin (a - b + φ)
    | 1, 1 => 1
    | 1, 2 => Real.sin b
    | 2, 0 => Real.sin a
    | 2, 1 => Real.sin b
    | 2, 2 => 1
  Matrix.det M = 0 := by sorry

end NUMINAMATH_CALUDE_trig_matrix_det_zero_l2589_258942


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2589_258976

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 ∧
  (∀ n, a (n + 1) = a n + d) ∧
  (a 3)^2 = a 1 * a 9 →
  (a 1 + a 3 + a 6) / (a 2 + a 4 + a 10) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2589_258976


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l2589_258912

theorem not_necessary_not_sufficient_condition (a b : ℝ) : 
  ¬(((a ≠ 5 ∧ b ≠ -5) → (a + b ≠ 0)) ∧ ((a + b ≠ 0) → (a ≠ 5 ∧ b ≠ -5))) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l2589_258912


namespace NUMINAMATH_CALUDE_abs_value_of_root_l2589_258936

theorem abs_value_of_root (z : ℂ) : z^2 - 2*z + 2 = 0 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_root_l2589_258936


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l2589_258901

theorem decimal_equivalent_of_one_fifth_squared :
  (1 / 5 : ℚ) ^ 2 = (4 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_one_fifth_squared_l2589_258901


namespace NUMINAMATH_CALUDE_find_y_l2589_258956

theorem find_y : ∃ y : ℕ, y^3 * 6^4 / 432 = 5184 ∧ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l2589_258956


namespace NUMINAMATH_CALUDE_first_day_distance_l2589_258920

theorem first_day_distance (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  (total_distance * (1 - ratio) / (1 - ratio^days)) = 192 :=
sorry

end NUMINAMATH_CALUDE_first_day_distance_l2589_258920


namespace NUMINAMATH_CALUDE_vector_at_t_5_l2589_258902

def line_parameterization (t : ℝ) : ℝ × ℝ := sorry

theorem vector_at_t_5 (h1 : line_parameterization 1 = (2, 7))
                      (h2 : line_parameterization 4 = (8, -5)) :
  line_parameterization 5 = (10, -9) := by sorry

end NUMINAMATH_CALUDE_vector_at_t_5_l2589_258902


namespace NUMINAMATH_CALUDE_expand_expression_l2589_258939

theorem expand_expression (x : ℝ) : (5*x - 3) * (x^3 + 4*x) = 5*x^4 - 3*x^3 + 20*x^2 - 12*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2589_258939


namespace NUMINAMATH_CALUDE_jaylen_cucumbers_count_l2589_258978

/-- The number of cucumbers Jaylen has -/
def jaylen_cucumbers (jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total : ℕ) : ℕ :=
  jaylen_total - (jaylen_carrots + jaylen_bell_peppers + jaylen_green_beans)

theorem jaylen_cucumbers_count :
  ∀ (jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total : ℕ),
  jaylen_carrots = 5 →
  jaylen_bell_peppers = 2 * kristin_bell_peppers →
  jaylen_green_beans = kristin_green_beans / 2 - 3 →
  kristin_bell_peppers = 2 →
  kristin_green_beans = 20 →
  jaylen_total = 18 →
  jaylen_cucumbers jaylen_carrots jaylen_bell_peppers jaylen_green_beans kristin_bell_peppers kristin_green_beans jaylen_total = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jaylen_cucumbers_count_l2589_258978


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2589_258925

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 2, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_union_theorem : (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2589_258925


namespace NUMINAMATH_CALUDE_chord_length_is_six_l2589_258935

/-- A circle with equation x^2 + y^2 + 8x - 10y + 41 = r^2 that is tangent to the x-axis --/
structure TangentCircle where
  r : ℝ
  tangent_to_x_axis : r = 5

/-- The length of the chord intercepted by the circle on the y-axis --/
def chord_length (c : TangentCircle) : ℝ :=
  let y₁ := 2
  let y₂ := 8
  |y₁ - y₂|

/-- Theorem stating that the length of the chord intercepted by the circle on the y-axis is 6 --/
theorem chord_length_is_six (c : TangentCircle) : chord_length c = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_is_six_l2589_258935


namespace NUMINAMATH_CALUDE_combination_sum_identity_l2589_258927

theorem combination_sum_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 := by
  sorry

end NUMINAMATH_CALUDE_combination_sum_identity_l2589_258927


namespace NUMINAMATH_CALUDE_range_of_c_l2589_258954

-- Define the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := 1 - 2*c < 0

-- State the theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) 
  (h3 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) : 
  (c ∈ Set.Ioc 0 (1/2)) ∨ (c ∈ Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l2589_258954


namespace NUMINAMATH_CALUDE_range_of_m_l2589_258997

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (|1 - (x-1)/2| ≤ 3 → x^2 - 2*x + 1 - m^2 ≤ 0) ∧ 
  (∃ x : ℝ, x^2 - 2*x + 1 - m^2 ≤ 0 ∧ |1 - (x-1)/2| > 3)) ∧ 
  m > 0 → 
  m ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2589_258997


namespace NUMINAMATH_CALUDE_sector_radius_proof_l2589_258944

/-- The area of a circular sector -/
def sectorArea : ℝ := 51.54285714285714

/-- The central angle of the sector in degrees -/
def centralAngle : ℝ := 41

/-- The radius of the circle -/
def radius : ℝ := 12

/-- Theorem stating that the given sector area and central angle result in the specified radius -/
theorem sector_radius_proof : 
  abs (sectorArea - (centralAngle / 360) * Real.pi * radius^2) < 1e-6 := by sorry

end NUMINAMATH_CALUDE_sector_radius_proof_l2589_258944


namespace NUMINAMATH_CALUDE_green_tile_probability_l2589_258963

theorem green_tile_probability :
  let total_tiles := 100
  let is_green (n : ℕ) := n % 5 = 3
  let green_tiles := Finset.filter is_green (Finset.range total_tiles)
  (green_tiles.card : ℚ) / total_tiles = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_green_tile_probability_l2589_258963


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2589_258917

-- Define the basic types
variable (P : Type) -- Type for points
variable (α : Set P) -- Type for planes
variable (l m : Set P) -- Type for lines

-- Define the geometric relations
variable (perpendicular : Set P → Set P → Prop) -- Perpendicular relation for lines and planes
variable (parallel : Set P → Set P → Prop) -- Parallel relation for lines and planes
variable (subset : Set P → Set P → Prop) -- Subset relation for lines and planes

-- State the theorem
theorem perpendicular_necessary_not_sufficient
  (h : perpendicular m α) :
  (∀ l, parallel l α → perpendicular l m) ∧
  ¬(∀ l, perpendicular l m → parallel l α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2589_258917


namespace NUMINAMATH_CALUDE_money_sharing_problem_l2589_258918

theorem money_sharing_problem (john jose binoy : ℕ) 
  (h1 : john + jose + binoy > 0)  -- Ensure total is positive
  (h2 : jose = 2 * john)          -- Ratio condition for Jose
  (h3 : binoy = 3 * john)         -- Ratio condition for Binoy
  (h4 : john = 2200)              -- John's share
  : john + jose + binoy = 13200 := by
  sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l2589_258918


namespace NUMINAMATH_CALUDE_fd_length_l2589_258960

-- Define the triangle and arc
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r = 20 ∧ 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = r^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2

-- Define the semicircle
def Semicircle (A B D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), O = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
  (D.1 - O.1)^2 + (D.2 - O.2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4

-- Define the tangent point
def Tangent (C D O : ℝ × ℝ) : Prop :=
  (C.1 - D.1) * (D.1 - O.1) + (C.2 - D.2) * (D.2 - O.2) = 0

-- Define the intersection point
def Intersect (C D F B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
  F = (C.1 + t*(D.1 - C.1), C.2 + t*(D.2 - C.2)) ∧
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = 20^2

-- Main theorem
theorem fd_length (A B C D F : ℝ × ℝ) :
  Triangle A B C →
  Semicircle A B D →
  Tangent C D ((A.1 + B.1)/2, (A.2 + B.2)/2) →
  Intersect C D F B →
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fd_length_l2589_258960


namespace NUMINAMATH_CALUDE_three_team_leads_per_supervisor_l2589_258910

/-- Represents the organizational structure of a company -/
structure Company where
  workers : ℕ
  team_leads : ℕ
  supervisors : ℕ
  worker_to_lead_ratio : ℕ

/-- Calculates the number of team leads per supervisor -/
def team_leads_per_supervisor (c : Company) : ℚ :=
  c.team_leads / c.supervisors

/-- Theorem: The number of team leads per supervisor is 3 -/
theorem three_team_leads_per_supervisor (c : Company) 
  (h1 : c.worker_to_lead_ratio = 10)
  (h2 : c.supervisors = 13)
  (h3 : c.workers = 390) :
  team_leads_per_supervisor c = 3 := by
  sorry

#check three_team_leads_per_supervisor

end NUMINAMATH_CALUDE_three_team_leads_per_supervisor_l2589_258910


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2589_258993

theorem nested_fraction_equality : 
  2 - (1 / (2 + (1 / (2 - (1 / 2))))) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2589_258993


namespace NUMINAMATH_CALUDE_difference_of_squares_l2589_258923

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m - 1) * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2589_258923


namespace NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_range_l2589_258964

-- Define the property of having no prime factors less than 20
def no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

-- Define the property of being the smallest nonprime with no small prime factors
def smallest_nonprime_no_small_factors (n : ℕ) : Prop :=
  n > 1 ∧ ¬n.Prime ∧ no_small_prime_factors n ∧
  ∀ m, m > 1 → ¬m.Prime → no_small_prime_factors m → n ≤ m

-- State the theorem
theorem smallest_nonprime_no_small_factors_range :
  ∃ n, smallest_nonprime_no_small_factors n ∧ 500 < n ∧ n ≤ 550 := by
  sorry

end NUMINAMATH_CALUDE_smallest_nonprime_no_small_factors_range_l2589_258964


namespace NUMINAMATH_CALUDE_disinfectant_purchase_theorem_l2589_258937

/-- Represents the cost and quantity of disinfectants --/
structure DisinfectantPurchase where
  costA : ℕ  -- Cost of one bottle of Class A disinfectant
  costB : ℕ  -- Cost of one bottle of Class B disinfectant
  quantityA : ℕ  -- Number of bottles of Class A disinfectant
  quantityB : ℕ  -- Number of bottles of Class B disinfectant

/-- Theorem about disinfectant purchase --/
theorem disinfectant_purchase_theorem 
  (purchase : DisinfectantPurchase)
  (total_cost : purchase.costA * purchase.quantityA + purchase.costB * purchase.quantityB = 2250)
  (cost_difference : purchase.costA + 15 = purchase.costB)
  (quantities : purchase.quantityA = 80 ∧ purchase.quantityB = 35)
  (new_total : ℕ)
  (new_budget : new_total * purchase.costA + (50 - new_total) * purchase.costB ≤ 1200)
  : purchase.costA = 15 ∧ purchase.costB = 30 ∧ new_total ≥ 20 := by
  sorry

#check disinfectant_purchase_theorem

end NUMINAMATH_CALUDE_disinfectant_purchase_theorem_l2589_258937


namespace NUMINAMATH_CALUDE_soccer_team_probability_l2589_258952

theorem soccer_team_probability (total_players defenders : ℕ) 
  (h1 : total_players = 12)
  (h2 : defenders = 6) :
  (Nat.choose defenders 2 : ℚ) / (Nat.choose total_players 2) = 5 / 22 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_probability_l2589_258952


namespace NUMINAMATH_CALUDE_M_characterization_a_range_l2589_258992

-- Define the set M
def M : Set ℝ := {m | ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

-- Define the set N
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x + a - 2) < 0}

-- Statement 1
theorem M_characterization : M = {m | -1/4 ≤ m ∧ m < 2} := by sorry

-- Statement 2
theorem a_range (h : ∀ m ∈ M, ∃ x ∈ N a, x^2 - x - m = 0) : 
  a ∈ Set.Iic (-1/4) ∪ Set.Ioi (9/4) := by sorry

end NUMINAMATH_CALUDE_M_characterization_a_range_l2589_258992


namespace NUMINAMATH_CALUDE_unique_solution_cube_difference_l2589_258947

theorem unique_solution_cube_difference (x y : ℤ) :
  (x + 2)^4 - x^4 = y^3 ↔ x = -1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_difference_l2589_258947


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_ratio_l2589_258966

theorem consecutive_odd_numbers_ratio (x : ℝ) (k m : ℝ) : 
  x = 4.2 →                             -- First number is 4.2
  9 * x = k * (x + 4) + m * (x + 2) + 9  -- Equation from the problem
    → (x + 4) / (x + 2) = 41 / 31        -- Ratio of third to second number
  := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_ratio_l2589_258966


namespace NUMINAMATH_CALUDE_parabola_vertex_l2589_258945

/-- The vertex of a parabola defined by y = a(x+1)^2 - 2 is at (-1, -2) --/
theorem parabola_vertex (a : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * (x + 1)^2 - 2
  ∃! p : ℝ × ℝ, p.1 = -1 ∧ p.2 = -2 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2589_258945


namespace NUMINAMATH_CALUDE_total_trophies_is_430_l2589_258987

/-- Calculates the total number of trophies Jack and Michael will have after three years -/
def totalTrophiesAfterThreeYears (michaelCurrentTrophies : ℕ) (michaelTrophyIncrease : ℕ) (jackMultiplier : ℕ) : ℕ :=
  let michaelFutureTrophies := michaelCurrentTrophies + michaelTrophyIncrease
  let jackFutureTrophies := jackMultiplier * michaelCurrentTrophies
  michaelFutureTrophies + jackFutureTrophies

/-- Theorem stating that the total number of trophies after three years is 430 -/
theorem total_trophies_is_430 : 
  totalTrophiesAfterThreeYears 30 100 10 = 430 := by
  sorry

end NUMINAMATH_CALUDE_total_trophies_is_430_l2589_258987


namespace NUMINAMATH_CALUDE_two_digit_numbers_product_sum_l2589_258975

theorem two_digit_numbers_product_sum (x y : ℕ) : 
  (10 ≤ x ∧ x < 100) ∧ 
  (10 ≤ y ∧ y < 100) ∧ 
  (2000 ≤ x * y ∧ x * y < 3000) ∧ 
  (100 ≤ x + y ∧ x + y < 1000) ∧ 
  (x * y = 2000 + (x + y)) →
  ((x = 24 ∧ y = 88) ∨ (x = 88 ∧ y = 24) ∨ (x = 30 ∧ y = 70) ∨ (x = 70 ∧ y = 30)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_product_sum_l2589_258975
