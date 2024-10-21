import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_interest_rate_is_30_percent_l1237_123713

/-- Given an initial deposit amount and two interest scenarios, 
    proves that the initial interest rate is 30% --/
theorem initial_interest_rate_is_30_percent 
  (deposit : ℝ) 
  (initial_interest : ℝ) 
  (additional_interest : ℝ) 
  (h1 : initial_interest = 101.20)
  (h2 : additional_interest = 20.24)
  : deposit * 0.30 = initial_interest := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_interest_rate_is_30_percent_l1237_123713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_cost_proof_l1237_123715

/-- The cost of one pair of jeans -/
noncomputable def jeans_cost : ℝ := 16.85

/-- The cost of one shirt -/
noncomputable def shirt_cost : ℝ := (104.25 - 3 * jeans_cost) / 6

theorem jeans_cost_proof : 
  (3 * jeans_cost + 6 * shirt_cost = 104.25) ∧ 
  (4 * jeans_cost + 5 * shirt_cost = 112.15) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeans_cost_proof_l1237_123715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_complex_sum_product_l1237_123710

/-- Represents that four complex numbers form a rectangle in the complex plane -/
def IsRectangle (p q r s : ℂ) : Prop := sorry

/-- Returns the side lengths of a rectangle formed by four complex numbers -/
def SideLength (p q r s : ℂ) (h : IsRectangle p q r s) : ℝ × ℝ := sorry

/-- Given complex numbers p, q, r, s forming a rectangle with side lengths 15 and 20,
    if |p + q + r + s| = 50, then |pq + pr + ps + qr + qs + rs| = 625 -/
theorem rectangle_complex_sum_product 
  (p q r s : ℂ) 
  (h_rectangle : IsRectangle p q r s)
  (h_sides : SideLength p q r s h_rectangle = (15, 20))
  (h_sum : Complex.abs (p + q + r + s) = 50) :
  Complex.abs (p*q + p*r + p*s + q*r + q*s + r*s) = 625 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_complex_sum_product_l1237_123710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1237_123719

theorem trig_identity (α : ℝ) (m : ℝ) :
  (Real.tan α + 1 / Real.tan α)^2 + (1 / Real.cos α + 1 / Real.sin α)^2 = m + Real.sin α^2 + Real.cos α^2 →
  m = 5 + Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1237_123719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_l1237_123764

theorem point_on_curve : ∃ θ : ℝ, 
  Real.sin (2 * θ) = -3/4 ∧ Real.cos θ + Real.sin θ = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_l1237_123764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_and_crossing_time_l1237_123761

/-- Given a river flowing at speed u and a boat with speed v relative to the water,
    moving at an angle α to the river's direction, this function computes the boat's
    speed relative to the river bank. -/
noncomputable def boatSpeedRelativeToBank (u v α : ℝ) : ℝ :=
  Real.sqrt (u^2 + v^2 + 2*u*v*(Real.cos α))

/-- The angle that minimizes the time taken to cross the river. -/
noncomputable def minCrossingTimeAngle : ℝ := Real.pi / 2

theorem boat_speed_and_crossing_time (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  ∀ α : ℝ, boatSpeedRelativeToBank u v α = Real.sqrt (u^2 + v^2 + 2*u*v*(Real.cos α)) ∧
  (∀ β : ℝ, Real.sin β ≤ Real.sin minCrossingTimeAngle) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_and_crossing_time_l1237_123761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1237_123798

/-- The number of days it takes for two workers to complete a job together,
    given their individual work rates. -/
noncomputable def days_to_complete (gyuri_days : ℚ) (gyuri_portion : ℚ) 
                     (seungyeon_days : ℚ) (seungyeon_portion : ℚ) : ℚ :=
  1 / ((1 / (gyuri_days / gyuri_portion)) + (1 / (seungyeon_days / seungyeon_portion)))

theorem job_completion_time :
  days_to_complete 5 (1/3) 2 (1/5) = 6 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l1237_123798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_completion_time_l1237_123704

/-- Represents the total amount of work to be done -/
noncomputable def total_work : ℝ := 1

/-- The rate at which x works (portion of work done per day) -/
noncomputable def x_rate : ℝ := 1 / 20

/-- The rate at which y works (portion of work done per day) -/
noncomputable def y_rate : ℝ := 1 / 24

/-- The number of days x works initially -/
noncomputable def x_initial_days : ℝ := 10

/-- The number of days y works to finish the job after x's initial work -/
noncomputable def y_finish_days : ℝ := 12

theorem x_completion_time :
  x_rate * (total_work / x_rate) = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_completion_time_l1237_123704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1237_123734

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (2 * Real.sin (x + Real.pi / 3) + Real.sin x) * Real.cos x - Real.sqrt 3 * (Real.sin x)^2

-- State the theorem
theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≥ 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1237_123734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_competition_score_l1237_123749

theorem shooting_competition_score (team_size best_score hypothetical_best_score hypothetical_average : ℕ) 
  (h1 : team_size = 6)
  (h2 : best_score = 85)
  (h3 : hypothetical_best_score = 92)
  (h4 : hypothetical_average = 84) :
  (hypothetical_average * team_size - (hypothetical_best_score - best_score)) = 497 := by
  sorry

#check shooting_competition_score

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_competition_score_l1237_123749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l1237_123791

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- A rectangle in a 2D plane -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside or on a rectangle -/
def isInRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

theorem min_distance_bound (points : Finset Point) (r : Rectangle) :
  r.width = 2 →
  r.height = 1 →
  points.card = 6 →
  (∀ p, p ∈ points → isInRectangle p r) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_bound_l1237_123791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_N_cardinality_l1237_123705

/-- The set of positive integers with n digits of 1, n digits of 2, and no other digits. -/
def M (n : ℕ) : Set ℕ :=
  {x : ℕ | ∃ (digits : List ℕ), 
    (digits.length = 2 * n) ∧ 
    (digits.count 1 = n) ∧ 
    (digits.count 2 = n) ∧ 
    (∀ d ∈ digits, d = 1 ∨ d = 2) ∧
    (x = digits.foldl (fun acc d => acc * 10 + d) 0)}

/-- The set of positive integers with decimal representation containing only digits 1, 2, 3, and 4, 
    with an equal number of 1s and 2s. -/
def N (n : ℕ) : Set ℕ :=
  {x : ℕ | ∃ (digits : List ℕ), 
    (digits.count 1 = n) ∧ 
    (digits.count 2 = n) ∧ 
    (∀ d ∈ digits, d ∈ [1, 2, 3, 4]) ∧
    (x = digits.foldl (fun acc d => acc * 10 + d) 0)}

/-- The cardinality of set M is equal to the cardinality of set N. -/
theorem M_eq_N_cardinality (n : ℕ) : M n ≃ N n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_eq_N_cardinality_l1237_123705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l1237_123799

variable (O A B C D : EuclideanSpace ℝ (Fin 3))
variable (x y z : ℝ)

/-- Four points are coplanar if they lie on the same plane -/
def areCoplanar (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = 0 ∧ (a, b, c) ≠ (0, 0, 0)

/-- Three points are collinear if they lie on the same line -/
def areCollinear (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∃ (t : ℝ), B - A = t • (C - A)

theorem vector_relation (h1 : areCoplanar A B C D)
    (h2 : ¬areCollinear A B C ∧ ¬areCollinear A B D ∧ ¬areCollinear A C D ∧ ¬areCollinear B C D)
    (h3 : A - O = (2 * x) • (B - O) + (3 * y) • (C - O) + (4 * z) • (D - O)) :
    2 * x + 3 * y + 4 * z = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relation_l1237_123799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whack_a_mole_tickets_value_l1237_123739

/-- Represents the number of tickets Luke won playing 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 2

/-- The number of tickets Luke won playing 'skee ball' -/
def skee_ball_tickets : ℕ := 13

/-- The cost of each candy in tickets -/
def candy_cost : ℕ := 3

/-- The number of candies Luke could buy -/
def candies_bought : ℕ := 5

/-- The total number of tickets Luke spent on candy -/
def total_spent : ℕ := candy_cost * candies_bought

theorem whack_a_mole_tickets_value : whack_a_mole_tickets = total_spent - skee_ball_tickets := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whack_a_mole_tickets_value_l1237_123739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_t_value_l1237_123747

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define vectors a and b
variable (a b : V)

-- Define the non-collinearity of a and b
variable (h_non_collinear : a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (k : ℝ), a = k • b)

-- Define vectors AB and AC
variable (t : ℝ)
def AB (t : ℝ) (a b : V) : V := t • a - b
def AC (a b : V) : V := 2 • a + 3 • b

-- Define collinearity of points A, B, and C
def collinear (AB AC : V) : Prop := ∃ (k : ℝ), AB = k • AC

-- Theorem statement
theorem collinear_implies_t_value (a b : V) (t : ℝ) 
  (h_collinear : collinear (AB t a b) (AC a b)) : t = -2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_t_value_l1237_123747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_3755_l1237_123701

/-- A sequence of positive integers that are powers of 5 or sums of distinct powers of 5 -/
def modified_sequence : ℕ → ℕ := sorry

/-- The nth term of the modified sequence -/
def nth_term (n : ℕ) : ℕ := modified_sequence n

/-- The 50th term of the modified sequence -/
def fiftieth_term : ℕ := nth_term 50

/-- Theorem: The 50th term of the modified sequence is 3755 -/
theorem fiftieth_term_is_3755 : fiftieth_term = 3755 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fiftieth_term_is_3755_l1237_123701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_condition_percentage_l1237_123783

theorem fruit_condition_percentage (oranges bananas : ℕ) 
  (rotten_oranges_percent rotten_bananas_percent : ℚ) : 
  oranges = 600 → 
  bananas = 400 → 
  rotten_oranges_percent = 15/100 → 
  rotten_bananas_percent = 6/100 → 
  (((oranges + bananas : ℚ) - 
    (oranges * rotten_oranges_percent + bananas * rotten_bananas_percent)) / 
   (oranges + bananas : ℚ)) * 100 = 886/10 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_condition_percentage_l1237_123783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_questions_count_l1237_123748

/-- Represents the number of questions missed by a person on a test. -/
abbrev QuestionsMissed := Nat

/-- Given that you missed 5 times as many questions as your friend,
    and together you missed 216 questions, prove that you missed 180 questions. -/
theorem missed_questions_count 
  (your_missed : QuestionsMissed) 
  (friend_missed : QuestionsMissed) 
  (h1 : your_missed = 5 * friend_missed) 
  (h2 : your_missed + friend_missed = 216) : 
  your_missed = 180 := by
  sorry

#check missed_questions_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missed_questions_count_l1237_123748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1237_123784

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := 5 * Real.sqrt (x - 1) + Real.sqrt 2 * Real.sqrt (5 - x)

-- State the theorem
theorem f_max_value :
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → f x ≤ 6 * Real.sqrt 3 :=
by
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l1237_123784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_l1237_123741

/-- Represents the race parameters and conditions -/
structure RaceData where
  distance : ℝ
  time_difference : ℝ
  speed_difference : ℝ

/-- Calculates the average speed of the slower team given race data -/
noncomputable def slower_team_speed (data : RaceData) : ℝ :=
  let v := data.distance / (data.distance / (data.distance / (data.distance / data.speed_difference + data.time_difference) + data.speed_difference) + data.time_difference)
  v

/-- Theorem stating that under the given conditions, the slower team's speed is 20 mph -/
theorem race_theorem (data : RaceData) 
  (h_distance : data.distance = 300)
  (h_time_diff : data.time_difference = 3)
  (h_speed_diff : data.speed_difference = 5) :
  slower_team_speed data = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval slower_team_speed { distance := 300, time_difference := 3, speed_difference := 5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_theorem_l1237_123741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_4_l1237_123708

noncomputable def x : ℕ → ℝ
  | 0 => 7
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem x_converges_to_4 : 
  ∃ m : ℕ, 109 ≤ m ∧ m ≤ 324 ∧ x m ≤ 4 + 1 / 2^18 ∧ 
  ∀ k : ℕ, 0 < k ∧ k < 109 → x k > 4 + 1 / 2^18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_4_l1237_123708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1237_123762

/-- Given plane vectors m and n with unit length, prove that if the inequality holds for all real t,
    then the angle between m and n is 60 degrees and the projection of m onto m+n is (m+n)/2 -/
theorem vector_properties (m n : ℝ × ℝ) :
  (‖m‖ = 1) →
  (‖n‖ = 1) →
  (∀ t : ℝ, ‖m - (1/2 : ℝ) • n‖ ≤ ‖m + t • n‖) →
  (m • n = 1/2) ∧
  (m • (m + n) / ‖m + n‖^2 • (m + n) = (1/2 : ℝ) • (m + n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1237_123762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_until_six_l1237_123729

-- Define the probabilities for each outcome
noncomputable def p6 : ℝ := 3/8
noncomputable def p4 : ℝ := 1/4
noncomputable def pOther : ℝ := 1/20

-- Define the expected value of a single roll
noncomputable def expectedValueRoll : ℝ := 1 * pOther + 2 * pOther + 3 * pOther + 4 * p4 + 5 * pOther + 6 * p6

-- Define the expected number of rolls until a 6 is rolled
noncomputable def expectedRolls : ℝ := 1 / p6

-- Theorem statement
theorem expected_sum_until_six :
  expectedRolls * expectedValueRoll = 9.4 := by
  -- Expand definitions
  unfold expectedRolls expectedValueRoll p6 p4 pOther
  -- Perform algebraic simplifications
  simp [mul_add, mul_sub, mul_div]
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sum_until_six_l1237_123729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1237_123760

-- Define the function f
def f (x : ℝ) : ℝ := |x - 10| + |x - 20|

-- Define the set A
def A : Set ℝ := {a | ∃ x, f x < 10 * a + 10}

-- Theorem statement
theorem problem_solution :
  (A = Set.Ioi 0) ∧
  (∀ a b : ℝ, a ∈ A → b ∈ A → a ≠ b → a^a * b^b > a^b * b^a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1237_123760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_allison_wins_prob_l1237_123742

/-- Represents a 6-sided cube with specific face values --/
structure Cube where
  faces : Fin 6 → ℕ

/-- Allison's cube with all faces showing 4 --/
def allison_cube : Cube :=
  { faces := fun _ => 4 }

/-- Brian's cube with faces numbered 1 to 6 --/
def brian_cube : Cube :=
  { faces := fun i => i.val + 1 }

/-- Noah's cube with three faces showing 1 and three faces showing 5 --/
def noah_cube : Cube :=
  { faces := fun i => if i.val < 3 then 1 else 5 }

/-- The probability of rolling a value less than 4 on a given cube --/
def prob_less_than_four (c : Cube) : ℚ :=
  (Finset.filter (fun i : Fin 6 => c.faces i < 4) (Finset.univ)).card / 6

/-- The main theorem stating the probability of Allison's roll being greater than both Brian's and Noah's --/
theorem allison_wins_prob : 
  prob_less_than_four brian_cube * prob_less_than_four noah_cube = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_allison_wins_prob_l1237_123742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1237_123711

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)

-- State the theorem
theorem f_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (x₁ x₂ : ℝ) (h_f_eq : f ω x₁ = 2 ∧ f ω x₂ = 2) 
  (h_min_diff : ∀ y z, f ω y = 2 → f ω z = 2 → |y - z| ≥ 2) 
  (h_exact_diff : |x₁ - x₂| = 2) : 
  (ω = Real.pi) ∧ 
  (∀ x ∈ Set.Ioo 0 1, f ω x ≤ 2) ∧
  (∀ x, f ω (x + 1/6) = f ω (-x + 1/6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1237_123711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_2cos6_l1237_123754

theorem min_sin6_2cos6 :
  ∀ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 ≥ 2/3 ∧
  ∃ x : ℝ, Real.sin x ^ 6 + 2 * Real.cos x ^ 6 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sin6_2cos6_l1237_123754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1237_123776

noncomputable section

-- Define the triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.C = 2 * t.a + t.c) :
  (t.B = 2 * π / 3) ∧ 
  (t.b = 9 → 
    ∃ (M : ℝ), 
      (2 * M = t.c - M) ∧ 
      (Real.sin t.A * (t.b - M) = Real.sin t.B * M) →
        (1/2 * 3 * 3 * Real.sqrt 3 = 9 * Real.sqrt 3 / 2)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1237_123776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_llama_count_l1237_123785

/-- Calculates the final number of llamas after breeding, trading, and selling -/
def final_llama_count (single_pregnant : ℕ) (twin_pregnant : ℕ) (traded_calves : ℕ) (new_adults : ℕ) (sell_fraction : ℚ) : ℕ :=
  let initial_adults := single_pregnant + twin_pregnant
  let calves := single_pregnant + 2 * twin_pregnant
  let after_trade := initial_adults + calves - traded_calves + new_adults
  (after_trade - Int.floor (↑after_trade * sell_fraction)).toNat

/-- Theorem stating that given the specific conditions, the final llama count is 18 -/
theorem jills_llama_count :
  final_llama_count 9 5 8 2 (1/3) = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jills_llama_count_l1237_123785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_range_three_roots_range_lower_bound_range_lower_bound_range_value_l1237_123745

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x^2 - 2*a*x + a
  else if 0 < x then 2*x + a/x
  else 0  -- This case is added to make the function total

-- Theorem 1: Range of a for increasing function
theorem increasing_range (a : ℝ) :
  (∀ x > 0, Monotone (f a)) → a ≤ -1/2 := by
  sorry

-- Theorem 2: Range of a for three distinct roots
theorem three_roots_range (a : ℝ) :
  (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 1 ∧ f a y = 1 ∧ f a z = 1) →
  (0 < a ∧ a < 1/8) := by
  sorry

-- Theorem 3: Range of a for f(x) ≥ x - 2a
theorem lower_bound_range :
  ∃ a : ℝ, ∀ x : ℝ, f a x ≥ x - 2*a := by
  sorry

theorem lower_bound_range_value (a : ℝ) :
  (∀ x : ℝ, f a x ≥ x - 2*a) →
  (0 ≤ a ∧ a ≤ 1 + Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_range_three_roots_range_lower_bound_range_lower_bound_range_value_l1237_123745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_inverse_sqrt_l1237_123770

/-- The domain of the function u(x) = 1/√x is (0, ∞) -/
theorem domain_of_inverse_sqrt (x : ℝ) :
  x ∈ Set.Ioi 0 ↔ ∃ y : ℝ, y = 1 / Real.sqrt x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_inverse_sqrt_l1237_123770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drum_roll_distance_l1237_123796

theorem drum_roll_distance (drum_diameter : ℝ) (initial_angle : ℝ) : 
  drum_diameter = 12 ∧ initial_angle = 30 * π / 180 → 
  2 * π = (π / 3) * drum_diameter := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drum_roll_distance_l1237_123796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l1237_123746

theorem determinant_sum (x y : ℝ) : 
  x ≠ y → 
  Matrix.det !![2, 6, 8; 4, x, y; 4, y, x] = 0 → 
  x + y = 28 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_sum_l1237_123746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_andy_cookies_l1237_123772

/-- Represents the number of cookies eaten by Andy -/
def andy_cookies : ℕ → Prop := sorry

/-- Represents the number of cookies eaten by Alexa -/
def alexa_cookies : ℕ → Prop := sorry

/-- Represents the number of cookies eaten by Ann -/
def ann_cookies : ℕ → Prop := sorry

/-- The total number of cookies shared by the siblings -/
def total_cookies : ℕ := 30

/-- Alexa eats twice the number of cookies eaten by Andy -/
axiom alexa_rule (n : ℕ) : andy_cookies n → alexa_cookies (2 * n)

/-- Ann eats three cookies fewer than Andy -/
axiom ann_rule (n : ℕ) : andy_cookies n → ann_cookies (n - 3)

/-- The siblings finish all the cookies -/
axiom total_rule (a : ℕ) : 
  andy_cookies a → alexa_cookies (2 * a) → ann_cookies (a - 3) → a + 2 * a + (a - 3) = total_cookies

/-- The maximum number of cookies Andy could have eaten is 8 -/
theorem max_andy_cookies : ∃ (n : ℕ), andy_cookies n ∧ n = 8 ∧ 
  ∀ (m : ℕ), andy_cookies m → m ≤ n := by
  sorry

#check max_andy_cookies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_andy_cookies_l1237_123772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1237_123720

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 2)) / (2^x - 1)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ -2 ∧ x ≠ 0}

-- Theorem statement
theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = domain_f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1237_123720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_halves_pi_plus_two_theta_l1237_123723

theorem sin_three_halves_pi_plus_two_theta (θ : ℝ) (h : Real.tan θ = 1/3) : 
  Real.sin (3/2 * π + 2*θ) = -4/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_three_halves_pi_plus_two_theta_l1237_123723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l1237_123740

theorem tan_sum_given_tan_cot_sum (x y : ℝ) 
  (h1 : Real.tan x + Real.tan y = 12)
  (h2 : (Real.tan x)⁻¹ + (Real.tan y)⁻¹ = 7) : 
  Real.tan (x + y) = -84/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_given_tan_cot_sum_l1237_123740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1237_123703

/-- Represents the race distance in yards -/
noncomputable def d : ℝ := sorry

/-- Represents the speed of racer A -/
noncomputable def a : ℝ := sorry

/-- Represents the speed of racer B -/
noncomputable def b : ℝ := sorry

/-- Represents the speed of racer C -/
noncomputable def c : ℝ := sorry

/-- A can beat B by 25 yards -/
axiom A_beats_B : d / a = (d - 25) / b

/-- B can beat C by 15 yards -/
axiom B_beats_C : d / b = (d - 15) / c

/-- A can beat C by 35 yards -/
axiom A_beats_C : d / a = (d - 35) / c

/-- The race distance is 75 yards -/
theorem race_distance : d = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l1237_123703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_cubic_l1237_123758

/-- A cubic polynomial with real coefficients -/
def cubic_polynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_cubic (a b c d : ℝ) :
  (∀ x : ℝ, cubic_polynomial a b c d (x^4 + x) ≥ cubic_polynomial a b c d (x^3 + 1)) →
  cubic_polynomial a b c d 1 = 0 →
  a ≠ 0 →
  -b / a = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_cubic_l1237_123758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_eccentricity_l1237_123755

/-- Hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0) -/
def hyperbola_C (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

/-- Point P -/
def P : ℝ × ℝ := (3, 6)

/-- Midpoint N of AB -/
def N : ℝ × ℝ := (12, 15)

/-- Line l passes through P and intersects C at A and B -/
def line_l (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (1 - t) • P + t • B

/-- Midpoint of AB is N -/
def midpoint_AB_is_N (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = N.1 ∧ (A.2 + B.2) / 2 = N.2

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_C_eccentricity :
  ∀ a b : ℝ,
  ∀ A B : ℝ × ℝ,
  hyperbola_C a b A.1 A.2 →
  hyperbola_C a b B.1 B.2 →
  line_l A B →
  midpoint_AB_is_N A B →
  eccentricity a b = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_eccentricity_l1237_123755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closest_to_D_l1237_123795

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The probability that a random point in the triangle is closer to one specific vertex -/
noncomputable def closestVertexProbability (t : Triangle) : ℝ := sorry

/-- Our specific triangle DEF -/
def triangleDEF : Triangle := { a := 6, b := 8, c := 10 }

theorem probability_closest_to_D (t : Triangle) (h : t = triangleDEF) :
  closestVertexProbability t = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_closest_to_D_l1237_123795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1237_123702

def z (m : ℝ) : ℂ := Complex.ofReal (m^2 - m) + Complex.I * (m - 1)

theorem complex_number_properties (m : ℝ) :
  (z m).im = 0 → m = 1 ∧
  ((z m).re = 0 ∧ (z m).im ≠ 0) → m = 0 ∧
  ((z m).re > 0 ∧ (z m).im > 0) → m > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l1237_123702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_walking_problem_l1237_123789

/-- Two students walking towards each other -/
structure WalkingStudents where
  initial_distance : ℝ
  rate1 : ℝ
  rate2 : ℝ
  time : ℝ

/-- Calculate the distance traveled by the first student -/
noncomputable def distance_traveled (w : WalkingStudents) : ℝ :=
  (w.rate1 / (w.rate1 + w.rate2)) * w.initial_distance

/-- The problem statement -/
theorem student_walking_problem (w : WalkingStudents) 
  (h1 : w.initial_distance = 350)
  (h2 : w.rate1 = 1.6)
  (h3 : w.rate2 = 1.9)
  (h4 : w.time = 100)
  (h5 : w.initial_distance = (w.rate1 + w.rate2) * w.time) :
  ∃ ε > 0, |distance_traveled w - 160| < ε := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_walking_problem_l1237_123789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_W_555_2_mod_1000_l1237_123726

def W : ℕ → ℕ → ℕ
  | n, 0 => n ^ n
  | n, k+1 => W (W n k) k

theorem W_555_2_mod_1000 : W 555 2 ≡ 875 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_W_555_2_mod_1000_l1237_123726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_property_l1237_123797

/-- Recursive definition of the continued fraction -/
def continuedFraction (a : ℕ → ℤ) : ℕ → ℤ → ℚ
  | 0, x => x
  | n+1, x => a (n+1) + 1 / continuedFraction a n x

theorem continued_fraction_property (a : ℕ → ℤ) (n : ℕ) :
  (∀ i : ℕ, i ≤ n → a i ≠ 0) →
  (∀ x : ℤ, x ∈ Set.preimage (continuedFraction a n) (Set.univ : Set ℚ) → continuedFraction a n x = x) →
  Even n ∧ n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_continued_fraction_property_l1237_123797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_probabilities_l1237_123730

-- Define the probabilities
variable (p q : ℝ)

-- Define the conditions
axiom p_gt_q : p > q
axiom prob_at_least_one : 1 - (1 - p) * (1 - q) = 5/6
axiom prob_both : p * q = 1/3

-- Define the probability that A answers fewer questions correctly than B
def prob_A_less_B (p q : ℝ) : ℝ := 
  (1 - p)^2 * 2 * (1 - q) * q + 
  (1 - p)^2 * q^2 + 
  2 * (1 - p) * p * q^2

-- State the theorem
theorem challenge_probabilities : 
  p = 2/3 ∧ q = 1/2 ∧ prob_A_less_B p q = 7/36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_challenge_probabilities_l1237_123730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1237_123782

noncomputable def f (a x : ℝ) : ℝ := a^(2*x) + 3*a^x - 2

theorem min_value_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x ∈ Set.Icc (-1) 1, f a x ≤ 8) 
  (h4 : ∃ x ∈ Set.Icc (-1) 1, f a x = 8) :
  ∃ x ∈ Set.Icc (-1) 1, f a x = -1/4 ∧ ∀ y ∈ Set.Icc (-1) 1, f a y ≥ -1/4 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1237_123782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_product_approx_l1237_123744

/-- Given three equations involving exponents p, r, and s, prove that their product is approximately 40.32 -/
theorem exponent_product_approx (p r s : ℝ) 
  (eq1 : (4 : ℝ)^p + (4 : ℝ)^3 = 272)
  (eq2 : (3 : ℝ)^r + 39 = 120)
  (eq3 : (4 : ℝ)^s + (2 : ℝ)^8 = 302) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |p * r * s - 40.32| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_product_approx_l1237_123744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_iff_a_in_range_l1237_123731

noncomputable def f (x : ℝ) : ℝ := -Real.exp x - x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.cos x

def tangent_perpendicular (a : ℝ) : Prop :=
  ∀ x₁ : ℝ, ∃ x₂ : ℝ, 
    (1 + (deriv f x₁) * (deriv (g a) x₂) = 0)

theorem tangent_perpendicular_iff_a_in_range :
  ∀ a : ℝ, tangent_perpendicular a ↔ -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

#check tangent_perpendicular_iff_a_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_iff_a_in_range_l1237_123731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_on_line_l1237_123765

theorem sin_theta_on_line (θ : Real) : 
  θ ∈ Set.Ioo 0 (π / 2) →  -- θ is in the first quadrant
  (∃ (x y : Real), x > 0 ∧ y > 0 ∧ 5 * y - 3 * x = 0 ∧  -- terminal side of θ lies on 5y - 3x = 0
  x = Real.cos θ ∧ y = Real.sin θ) →  -- (x, y) is on the unit circle
  Real.sin θ = 3 / Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_theta_on_line_l1237_123765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l1237_123756

/-- A function that computes the volume of a tetrahedron. This is assumed to exist for the purpose of the theorem. -/
noncomputable def volume_of_tetrahedron (a b c α β γ : ℝ) : ℝ :=
  sorry

/-- The volume of a tetrahedron given its edge lengths and angles. -/
theorem tetrahedron_volume 
  (a b c α β γ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) :
  let ω := (α + β + γ) / 2
  let V := (a * b * c) / 3 * Real.sqrt (Real.sin ω * Real.sin (ω - α) * Real.sin (ω - β) * Real.sin (ω - γ))
  V = volume_of_tetrahedron a b c α β γ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l1237_123756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1237_123733

open Real

theorem f_value_in_third_quadrant (α : ℝ) : 
  (π < α ∧ α < 3*π/2) →  -- α is in the third quadrant
  cos (3*π/2 - α) = 3/5 →
  let f := λ x => (sin (5*π - x) * cos (π + x) * cos (3*π/2 + x)) / 
              (cos (x + π/2) * tan (3*π - x) * sin (x - 3*π/2))
  f α = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1237_123733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l1237_123774

theorem diophantine_equation_solution (n x y z p : ℕ) (k : ℕ) :
  Nat.Prime p →
  (x^2 + 4*y^2) * (y^2 + 4*z^2) * (z^2 + 4*x^2) = p^n →
  (n = 3 ∧ x = 5^k ∧ y = 5^k ∧ z = 5^k ∧ p = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solution_l1237_123774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1237_123768

-- Define the curve
noncomputable def curve (x : ℝ) : ℝ := Real.log x + x - 1

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- State the theorem
theorem shortest_distance_curve_to_line :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = curve x₀ ∧ 
    (∀ (x y : ℝ), y = curve x → 
      (x - x₀)^2 + (y - y₀)^2 ≥ 
        ((2 * x₀ - y₀ + 3)^2) / (2^2 + (-1)^2)) ∧
    ((2 * x₀ - y₀ + 3)^2) / (2^2 + (-1)^2) = 5 := by
  sorry

#check shortest_distance_curve_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_curve_to_line_l1237_123768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_width_is_four_l1237_123709

/-- The width of a ring formed by two concentric circles -/
noncomputable def ring_width (inner_circumference outer_circumference : ℝ) : ℝ :=
  (outer_circumference - inner_circumference) / (2 * Real.pi)

/-- Theorem: The width of a ring with given inner and outer circumferences is 4 meters -/
theorem ring_width_is_four :
  ring_width (352 / 7) (528 / 7) = 4 := by
  -- Unfold the definition of ring_width
  unfold ring_width
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ring_width_is_four_l1237_123709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_AB_length_right_trapezoid_AB_length_25_l1237_123752

/-- Represents a trapezoid ABCD with right angles at ADC and BCD -/
structure RightTrapezoid where
  /-- Length of side AD -/
  t : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- BC equals 4 -/
  bc_eq : bc = 4
  /-- CD equals t + 13 -/
  cd_eq : cd = t + 13

/-- The length of AB in a right trapezoid ABCD -/
noncomputable def length_AB (trap : RightTrapezoid) : ℝ :=
  Real.sqrt (2 * trap.t^2 + 18 * trap.t + 185)

/-- Theorem: In a right trapezoid ABCD with AD = t, BC = 4, and CD = t + 13,
    the length of AB is √(2t² + 18t + 185) -/
theorem right_trapezoid_AB_length (trap : RightTrapezoid) :
  length_AB trap = Real.sqrt (2 * trap.t^2 + 18 * trap.t + 185) := by
  -- Proof goes here
  sorry

/-- Theorem: When t = 11, the length of AB is 25 -/
theorem right_trapezoid_AB_length_25 (trap : RightTrapezoid) (h : trap.t = 11) :
  length_AB trap = 25 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_trapezoid_AB_length_right_trapezoid_AB_length_25_l1237_123752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_theorem_l1237_123717

/-- Two lines in the plane -/
structure TwoLines where
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → ℝ → Prop

/-- The given two lines -/
def givenLines : TwoLines where
  l₁ := λ x y ↦ 2 * x + y - 2 = 0
  l₂ := λ x y m ↦ 2 * x - m * y + 4 = 0

/-- Perpendicularity condition for the two lines -/
def perpendicular (lines : TwoLines) (m : ℝ) : Prop :=
  4 - m = 0

/-- Parallelism condition for the two lines -/
def parallel (lines : TwoLines) (m : ℝ) : Prop :=
  -2 * m - 2 = 0

/-- Distance between two parallel lines -/
noncomputable def distance (lines : TwoLines) (m : ℝ) : ℝ :=
  |4 + 2| / Real.sqrt (4 + m^2)

theorem two_lines_theorem (lines : TwoLines) :
  (∃ m : ℝ, perpendicular lines m → 
    ∃ x y : ℝ, lines.l₁ x y ∧ lines.l₂ x y m ∧ x = 0.4 ∧ y = 1.2) ∧
  (∃ m : ℝ, parallel lines m → m = -1 ∧ distance lines m = (6 * Real.sqrt 5) / 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_lines_theorem_l1237_123717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minus_one_x_one_x_minus_one_equals_25_l1237_123790

-- Define the operation x
noncomputable def x (a b : ℝ) : ℝ := (b - a)^2 / a^2

-- Theorem statement
theorem minus_one_x_one_x_minus_one_equals_25 :
  x (-1) (x 1 (-1)) = 25 :=
by
  -- Unfold the definition of x
  unfold x
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minus_one_x_one_x_minus_one_equals_25_l1237_123790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1237_123714

theorem sin_cos_difference (θ : ℝ) (h1 : θ ∈ Set.Ioo 0 Real.pi) (h2 : Real.sin θ + Real.cos θ = 7/13) :
  Real.sin θ - Real.cos θ = 17/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1237_123714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sin_plus_one_l1237_123777

/-- The equation of the tangent line to y = sin x + 1 at (0, 1) is y = x + 1 -/
theorem tangent_line_sin_plus_one (x : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.sin t + 1
  let tangent_line : ℝ → ℝ := λ t => t + 1
  (∀ ε > 0, ∃ δ > 0, ∀ t, 0 < |t| → |t| < δ → |(tangent_line t - f t) / t| < ε) ∧
  tangent_line 0 = f 0 := by
  sorry

#check tangent_line_sin_plus_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sin_plus_one_l1237_123777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_term_is_three_l1237_123794

noncomputable def inverse_proportional_sequence (a₁ : ℝ) (a₂ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | 1 => a₂
  | n + 2 => (a₁ * a₂) / (inverse_proportional_sequence a₁ a₂ (n + 1))

theorem thirteenth_term_is_three :
  let seq := inverse_proportional_sequence 3 4
  seq 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thirteenth_term_is_three_l1237_123794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_original_price_l1237_123725

/-- The original price of a shirt given the discount percentage and discounted price. -/
noncomputable def original_price (discount_percent : ℝ) (discounted_price : ℝ) : ℝ :=
  discounted_price / (1 - discount_percent / 100)

/-- Theorem stating that the original price of a shirt with a 32% discount
    and a discounted price of Rs. 650 is approximately Rs. 955.88. -/
theorem shirt_original_price :
  let discount_percent : ℝ := 32
  let discounted_price : ℝ := 650
  abs (original_price discount_percent discounted_price - 955.88) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_original_price_l1237_123725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_volume_l1237_123788

/-- The volume of a cylindrical bucket with given dimensions and water occupancy. -/
theorem bucket_volume (d h : ℝ) (water_occupancy : ℝ) (h_d : d = 4) (h_h : h = 4) (h_wo : water_occupancy = 0.4) :
  let r := d / 2
  let water_volume := Real.pi * r^2 * h
  let bucket_volume := water_volume / water_occupancy
  bucket_volume = 125.6 := by
  sorry

#check bucket_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bucket_volume_l1237_123788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_domain_h_as_intervals_l1237_123792

-- Define the function h(x)
noncomputable def h (x : ℝ) : ℝ := (5 * x - 7) / (x - 5)

-- State the theorem about the domain of h(x)
theorem domain_of_h :
  ∀ x : ℝ, x ≠ 5 ↔ h x ∈ Set.univ := by
  sorry

-- Explicitly state the domain as a set
def domain_h : Set ℝ := {x : ℝ | x ≠ 5}

-- Theorem stating that the domain is equivalent to the union of two intervals
theorem domain_h_as_intervals :
  domain_h = Set.Iio 5 ∪ Set.Ioi 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_domain_h_as_intervals_l1237_123792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_yellow_apples_max_total_apples_l1237_123718

/-- Represents the process of taking apples from a basket --/
structure AppleTaking where
  green : ℕ
  yellow : ℕ
  red : ℕ
  total_apples : ℕ := 38
  initial_green : ℕ := 9
  initial_yellow : ℕ := 12
  initial_red : ℕ := 17
  stop_condition : Prop := green < yellow ∧ yellow < red
  green_constraint : Prop := green ≤ initial_green
  yellow_constraint : Prop := yellow ≤ initial_yellow
  red_constraint : Prop := red ≤ initial_red
  total_constraint : Prop := green + yellow + red ≤ total_apples

/-- The maximum number of yellow apples that can be taken out is 12 --/
theorem max_yellow_apples : 
  ∃ (g r : ℕ), g < 12 ∧ 12 < r ∧ r ≤ 17 := by
  sorry

/-- The maximum total number of apples that can be taken out is 36 --/
theorem max_total_apples :
  ∃ (g y r : ℕ), g < y ∧ y < r ∧ g + y + r = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_yellow_apples_max_total_apples_l1237_123718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_logarithm_l1237_123743

noncomputable def log_base (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

theorem golden_ratio_logarithm (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  log_base 9 p = log_base 12 q ∧ log_base 9 p = log_base 16 (p + q) →
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

#check golden_ratio_logarithm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_logarithm_l1237_123743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1237_123721

/-- Given hyperbola -/
def given_hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

/-- Desired hyperbola -/
def desired_hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 12 = 1

/-- Two hyperbolas have the same asymptotes if their equations are scalar multiples of each other -/
def same_asymptotes (h1 h2 : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, h1 x y ↔ h2 x y = (k = 1)

theorem hyperbola_theorem :
  same_asymptotes given_hyperbola desired_hyperbola ∧
  desired_hyperbola 2 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1237_123721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_root_bounds_and_smallest_upper_bound_l1237_123780

theorem nth_root_bounds_and_smallest_upper_bound :
  (∀ n : ℕ+, 1 ≤ (n : ℝ) ^ ((1 : ℝ) / n) ∧ (n : ℝ) ^ ((1 : ℝ) / n) ≤ 2) ∧
  (∀ k : ℝ, (∀ n : ℕ+, (n : ℝ) ^ ((1 : ℝ) / n) ≤ k) → (3 : ℝ) ^ (1 / 3) ≤ k) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_root_bounds_and_smallest_upper_bound_l1237_123780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1237_123750

theorem negation_of_proposition :
  (¬(∀ x : ℝ, x ≥ 0 → x^3 - 1 ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 - 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_proposition_l1237_123750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_greater_than_eight_factorial_l1237_123781

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def divisors_greater_than (n m : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ d => d ∣ n ∧ d > m)

theorem divisors_of_nine_factorial_greater_than_eight_factorial :
  (divisors_greater_than (factorial 9) (factorial 8)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_nine_factorial_greater_than_eight_factorial_l1237_123781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_from_pencil_l1237_123706

/-- Given two functions f and g of the form x^2 + y^2 + ax + by + c,
    prove that their difference scaled by lambda ≠ 1 forms a circle equation. -/
theorem circle_from_pencil (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (lambda : ℝ) (h : lambda ≠ 1) :
  ∃ D E F : ℝ,
    ∀ x y : ℝ,
      (x^2 + y^2 + a₁*x + b₁*y + c₁) - lambda*(x^2 + y^2 + a₂*x + b₂*y + c₂) =
      (1 - lambda)*(x^2 + y^2 + D*x + E*y + F) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_from_pencil_l1237_123706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_power_plus_one_l1237_123778

theorem composite_power_plus_one :
  ∀ a : ℕ, 1 < a → a ≤ 100 → ∃ n : ℕ, 0 < n ∧ n ≤ 6 ∧ ∃ k : ℕ, k > 1 ∧ k ∣ (a^(2^n) + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_power_plus_one_l1237_123778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_ending_with_1_9_ge_3_7_l1237_123775

/-- For any positive integer, the number of its divisors ending with 1 or 9
    is not less than the number of its divisors ending with 3 or 7. -/
theorem divisors_ending_with_1_9_ge_3_7 (n : ℕ+) : 
  (Finset.filter (fun d ↦ d ∣ n.val ∧ (d % 10 = 1 ∨ d % 10 = 9)) (Finset.range (n.val + 1))).card ≥ 
  (Finset.filter (fun d ↦ d ∣ n.val ∧ (d % 10 = 3 ∨ d % 10 = 7)) (Finset.range (n.val + 1))).card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_ending_with_1_9_ge_3_7_l1237_123775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_bounded_l1237_123766

/-- A triangle with perimeter 1 whose altitudes form another triangle -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_one : a + b + c = 1
  positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  altitudes_form_triangle : 1/a + 1/b > 1/c ∧ 1/b + 1/c > 1/a ∧ 1/c + 1/a > 1/b

/-- The minimum side length of a special triangle is bounded -/
theorem min_side_bounded (t : SpecialTriangle) :
  (3 - Real.sqrt 5) / 4 < min t.a (min t.b t.c) ∧ min t.a (min t.b t.c) ≤ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_side_bounded_l1237_123766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_l1237_123716

theorem fraction_inequality (x : ℝ) (h : x ≠ -4) :
  (x + 1) / (x + 4) ≥ 0 ↔ x ∈ Set.Iic (-4) ∪ Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_inequality_l1237_123716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_between_24_and_28_l1237_123759

theorem number_between_24_and_28 (S : Set ℕ) : S = {20, 23, 26, 29} →
  ∃! x, x ∈ S ∧ 24 < x ∧ x < 28 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_between_24_and_28_l1237_123759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_investment_value_l1237_123757

/-- Represents the investment and profit of a partnership business -/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  a_profit : ℝ

/-- Theorem stating that given the conditions of the problem, A's investment is 6300 -/
theorem a_investment_value (p : Partnership) 
  (hb : p.b_investment = 4200)
  (hc : p.c_investment = 10500)
  (ht : p.total_profit = 12600)
  (ha : p.a_profit = 3780)
  (h_profit_ratio : p.a_profit / p.total_profit = 
    p.a_investment / (p.a_investment + p.b_investment + p.c_investment)) :
  p.a_investment = 6300 := by
  sorry

#check a_investment_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_investment_value_l1237_123757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_5_pow_4_l1237_123779

/-- Decomposition function for power of natural numbers -/
def decomposition (m n i : ℕ) : ℕ := sorry

/-- The pattern of decomposition holds for all m ≥ 2 and n ≥ 2 -/
axiom decomposition_pattern {m n : ℕ} (h1 : m ≥ 2) (h2 : n ≥ 2) :
  m^n = (Finset.range (m - 1)).sum (λ i ↦ decomposition m n i)

/-- The third term in the decomposition of 5^4 is 125 -/
theorem third_term_of_5_pow_4 : decomposition 5 4 2 = 125 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_term_of_5_pow_4_l1237_123779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1237_123767

noncomputable def f (x : ℝ) := 3 * Real.sin (3 * x - Real.pi / 4) + 1

theorem sine_function_properties :
  (∃ (phase_shift : ℝ), phase_shift = -Real.pi / 12 ∧
    ∀ (x : ℝ), f (x + phase_shift) = 3 * Real.sin (3 * x)) ∧
  (∃ (vertical_translation : ℝ), vertical_translation = 1 ∧
    ∀ (x : ℝ), f x - vertical_translation = 3 * Real.sin (3 * x - Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_properties_l1237_123767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_buses_exists_max_system_l1237_123736

/-- Represents a bus route in the city -/
structure BusRoute where
  stops : Finset Nat
  stop_count : stops.card = 3

/-- The city's bus system -/
structure BusSystem where
  total_stops : Nat
  stop_count : total_stops = 9
  routes : Finset BusRoute
  distinct_routes : ∀ r1 r2, r1 ∈ routes → r2 ∈ routes → r1 ≠ r2 → (r1.stops ∩ r2.stops).card ≤ 1

/-- The maximum number of buses in the system is 10 -/
theorem max_buses (system : BusSystem) : system.routes.card ≤ 10 := by
  sorry

/-- There exists a bus system with exactly 10 buses -/
theorem exists_max_system : ∃ system : BusSystem, system.routes.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_buses_exists_max_system_l1237_123736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1237_123751

/-- Calculates the speed of a train crossing a bridge -/
noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / time
  speed_ms * 3.6

/-- Theorem: A train 500 meters long crossing a 200-meter bridge in 60 seconds has a speed of approximately 42.012 km/hr -/
theorem train_speed_calculation :
  let train_length : ℝ := 500
  let bridge_length : ℝ := 200
  let time : ℝ := 60
  abs (train_speed train_length bridge_length time - 42.012) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1237_123751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1237_123738

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  t.A = (3, -1) ∧
  (∃ (x y : ℝ), 6 * x + 10 * y - 59 = 0) ∧ -- Equation of median CM
  (∃ (x y : ℝ), x - 4 * y + 10 = 0)        -- Equation of angle bisector BT

-- Define the line through two points
def line_through (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, (x, y) = (1 - t) • p + t • q}

-- Theorem statement
theorem triangle_theorem (t : Triangle) 
  (h : triangle_properties t) : 
  t.B = (10, 5) ∧ 
  (∀ (x y : ℝ), (x, y) ∈ line_through t.B t.C ↔ 2 * x + 9 * y - 65 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1237_123738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1237_123707

/-- Definition of a hyperbola -/
noncomputable def is_hyperbola (a b : ℝ) (h : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

/-- Condition for a line to be tangent to a circle -/
noncomputable def is_tangent_to_circle (m : ℝ) (a : ℝ) : Prop :=
  abs (m * a) / Real.sqrt (1 + m^2) = Real.sqrt 3 / 2

/-- Theorem: Given conditions imply the specific hyperbola equation -/
theorem hyperbola_equation (a b : ℝ) (h : ℝ → ℝ → Prop) 
  (ha : a > 0) (hb : b > 0)
  (h_hyp : is_hyperbola a b h)
  (h_ecc : eccentricity a (2 * a) = 2)
  (h_tan : is_tangent_to_circle (b / a) a) :
  ∀ x y, h x y ↔ x^2 - y^2 / 3 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1237_123707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_equals_a_over_b_extrema_l1237_123732

theorem ab_equals_a_over_b_extrema (a b p q : ℝ) : 
  p = b + 2*a →
  q = 2*b + a →
  p + q = 6 →
  p > 0 →
  q > 0 →
  ∃ (n m : ℤ), p = ↑n ∧ q = ↑m →
  (∃ (ab_max ab_min a_over_b_max a_over_b_min : ℝ),
    (∀ (a' b' : ℝ), p = b' + 2*a' → q = 2*b' + a' → a'*b' ≤ ab_max) ∧
    (∀ (a' b' : ℝ), p = b' + 2*a' → q = 2*b' + a' → a'*b' ≥ ab_min) ∧
    (∀ (a' b' : ℝ), p = b' + 2*a' → q = 2*b' + a' → b' ≠ 0 → a'/b' ≤ a_over_b_max) ∧
    (∀ (a' b' : ℝ), p = b' + 2*a' → q = 2*b' + a' → b' ≠ 0 → a'/b' ≥ a_over_b_min) ∧
    ab_max = a_over_b_max ∧
    ab_min = a_over_b_min) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_equals_a_over_b_extrema_l1237_123732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1237_123753

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (1, 0)
def center2 : ℝ × ℝ := (0, 1)
def radius1 : ℝ := 1

noncomputable def radius2 : ℝ := Real.sqrt 2

-- Define the distance between the centers
noncomputable def centerDistance : ℝ := Real.sqrt 2

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  centerDistance < radius1 + radius2 ∧
  centerDistance > |radius1 - radius2| := by
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l1237_123753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_one_plus_i_l1237_123786

noncomputable def g (x : ℂ) : ℂ := (x^6 + x^2) / (x + 1)

theorem g_of_one_plus_i : g (1 + Complex.I) = -6/5 - 12/5 * Complex.I := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_one_plus_i_l1237_123786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_salary_is_1000_l1237_123773

/-- Represents the base salary of a car salesman -/
def base_salary : ℕ := sorry

/-- Represents the number of cars sold in January -/
def cars_sold_january : ℕ := sorry

/-- The commission per car sold -/
def commission_per_car : ℕ := 200

/-- Total earnings in January -/
def january_earnings : ℕ := 1800

/-- Number of cars needed to be sold in February to double January earnings -/
def cars_needed_february : ℕ := 13

/-- Double of January earnings -/
def double_january_earnings : ℕ := 3600

theorem base_salary_is_1000 :
  (base_salary + cars_sold_january * commission_per_car = january_earnings) →
  (base_salary + cars_needed_february * commission_per_car = double_january_earnings) →
  base_salary = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_salary_is_1000_l1237_123773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_eq_neg_two_l1237_123724

noncomputable section

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- Line l₁ with equation ax + 2y + a + 3 = 0 -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + a + 3 = 0

/-- Line l₂ with equation x + (a+1)y + 4 = 0 -/
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

/-- The slope of line l₁ -/
noncomputable def slope_l₁ (a : ℝ) : ℝ := -a / 2

/-- The slope of line l₂ -/
noncomputable def slope_l₂ (a : ℝ) : ℝ := -1 / (a + 1)

theorem parallel_lines_imply_a_eq_neg_two :
  ∀ a : ℝ, parallel (slope_l₁ a) (slope_l₂ a) → a = -2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_imply_a_eq_neg_two_l1237_123724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1237_123763

/-- Given two vectors a and b in ℝ², where a = (x-1, 2) and b = (1, x),
    if a is perpendicular to b, then x = 1/3. -/
theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (1, x)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_x_value_l1237_123763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1237_123727

/-- Given vectors OA, OB, OM in ℝ², and point P on line OM, prove properties about OP, angle APB, and minimum distance. -/
theorem vector_properties (OA OB OM : ℝ × ℝ) :
  OA = (-1, -3) →
  OB = (5, 3) →
  OM = (2, 2) →
  ∃ (P : ℝ × ℝ), ∃ (t : ℝ),
    (P = t • OM) ∧
    ((P.1 - OA.1) * (OB.1 - P.1) + (P.2 - OA.2) * (OB.2 - P.2) = -16) →
    (P = (1, 1)) ∧
    (((P.1 - OA.1) * (OB.1 - P.1) + (P.2 - OA.2) * (OB.2 - P.2)) /
      (Real.sqrt ((P.1 - OA.1)^2 + (P.2 - OA.2)^2) *
       Real.sqrt ((OB.1 - P.1)^2 + (OB.2 - P.2)^2)) = -4/5) ∧
    (∀ (s : ℝ), Real.sqrt ((OA.1 + s * P.1)^2 + (OA.2 + s * P.2)^2) ≥ Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l1237_123727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_bounds_l1237_123735

noncomputable def f (x : ℝ) := 2 * Real.sin x

theorem min_distance_between_bounds (x₁ x₂ : ℝ) :
  (∀ x, f x₁ ≤ f x ∧ f x ≤ f x₂) →
  ∃ d, d = |x₁ - x₂| ∧ d ≥ π ∧
  ∀ y₁ y₂, (∀ x, f y₁ ≤ f x ∧ f x ≤ f y₂) → |y₁ - y₂| ≥ d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_bounds_l1237_123735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_term_arithmetic_seq_l1237_123737

/-- Given four positive integers in arithmetic sequence with sum 46,
    the maximum value of the third term is 15 -/
theorem max_third_term_arithmetic_seq : ∀ a d : ℕ,
  a > 0 → d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) = 46 →
  ∀ b e : ℕ,
  b > 0 → e > 0 →
  b + (b + e) + (b + 2*e) + (b + 3*e) = 46 →
  a + 2*d ≤ 15 ∧ b + 2*e ≤ 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_term_arithmetic_seq_l1237_123737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1237_123793

theorem constant_term_expansion (x : ℝ) (x_pos : x > 0) : 
  (Finset.range 7).sum (λ r => (-1)^r * (Nat.choose 6 r) * (2^(6-r)) * x^(3-3*r/2)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1237_123793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l1237_123787

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h1 : θ ∈ Set.Ioo (-π/2) 0)
  (h2 : Real.cos θ = Real.sqrt 17 / 17) :
  Real.tan (θ + π/4) = -3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_fourth_l1237_123787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_20_deg_value_l1237_123728

theorem sin_20_deg_value (f : ℝ → ℝ) :
  (∀ x, f (Real.cos x) = Real.sin (3 * x)) →
  f (Real.sin (20 * π / 180)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_20_deg_value_l1237_123728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colorings_correct_l1237_123769

/-- The number of ways to color the sides of a regular n-gon with k colors,
    such that adjacent sides are colored differently and the polygon cannot be rotated. -/
def colorings (n : ℕ) (k : ℕ) : ℤ :=
  (k - 1)^n + (k - 1) * (-1 : ℤ)^n

/-- Helper function to represent the number of valid colorings (abstract definition) -/
def number_of_valid_colorings (n : ℕ) (k : ℕ) : ℤ :=
  sorry  -- This is left abstract as its actual implementation is what we're proving

/-- Theorem stating that the colorings function gives the correct number of valid colorings
    for a regular n-gon with k colors, where adjacent sides must be different colors. -/
theorem colorings_correct (n : ℕ) (k : ℕ) :
  colorings n k = number_of_valid_colorings n k := by
  sorry

/-- Lemma: The colorings function always returns a non-negative integer for n ≥ 2 and k ≥ 3 -/
lemma colorings_nonneg (n : ℕ) (k : ℕ) (hn : n ≥ 2) (hk : k ≥ 3) :
  colorings n k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_colorings_correct_l1237_123769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_and_equations_l1237_123722

/-- Two parallel lines passing through points A and B -/
structure ParallelLines where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ → ℝ := sorry

/-- The equations of the two parallel lines -/
def line_equations (lines : ParallelLines) : ℝ → (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) := sorry

theorem parallel_lines_distance_and_equations 
  (lines : ParallelLines) 
  (h1 : lines.A = (6, 2)) 
  (h2 : lines.B = (-3, -1)) :
  (∀ d, distance lines d > 0 ∧ distance lines d < 3 * Real.sqrt 10) ∧
  (∃ d_max, ∀ d, distance lines d ≤ distance lines d_max ∧
    let (eq1, eq2) := line_equations lines d_max
    eq1 = (λ x y ↦ 3 * x + y - 20 = 0) ∧
    eq2 = (λ x y ↦ 3 * x + y + 10 = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_and_equations_l1237_123722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_l1237_123700

noncomputable def first_three_scores : List ℚ := [92, 89, 93]
def fourth_score : ℚ := 95

noncomputable def average (scores : List ℚ) : ℚ :=
  scores.sum / scores.length

theorem average_increase :
  average (fourth_score :: first_three_scores) - average first_three_scores = 92/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_l1237_123700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_intercept_sum_horizontal_line_l1237_123712

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the line passing through two points
def Line (p1 p2 : Point) : Set (ℝ × ℝ) :=
  {(m, b) | m * p1.x + b = p1.y ∧ m * p2.x + b = p2.y}

-- Theorem statement
theorem slope_intercept_sum_horizontal_line (C D : Point)
    (h1 : C.y = 17)
    (h2 : D.y = 17)
    (h3 : C.x ≠ D.x) :
    ∃ (m b : ℝ), (m, b) ∈ Line C D ∧ m + b = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_intercept_sum_horizontal_line_l1237_123712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1237_123771

noncomputable section

variable (a b : ℝ)

def ellipse_equation (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

def line_equation (x y : ℝ) : Prop := y = 2 * x + 1

def midpoint_x : ℝ := -1/3

def vector_m (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁/a, y₁)
def vector_n (x₂ y₂ : ℝ) : ℝ × ℝ := (x₂/a, y₂)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def triangle_area (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (1/2) * abs (x₁ * y₂ - x₂ * y₁)

theorem ellipse_problem
  (h_a : a > b)
  (h_a_neq_1 : a ≠ 1)
  (x₁ y₁ x₂ y₂ : ℝ)
  (h_x_neq : x₁ ≠ x₂)
  (h_A_on_E : ellipse_equation a b x₁ y₁)
  (h_B_on_E : ellipse_equation a b x₂ y₂)
  (h_A_on_line : line_equation x₁ y₁)
  (h_B_on_line : line_equation x₂ y₂)
  (h_midpoint : (x₁ + x₂) / 2 = midpoint_x)
  (h_perp : perpendicular (vector_m a x₁ y₁) (vector_n a x₂ y₂)) :
  a = Real.sqrt 2 / 2 ∧
  triangle_area x₁ y₁ x₂ y₂ = a / 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l1237_123771
