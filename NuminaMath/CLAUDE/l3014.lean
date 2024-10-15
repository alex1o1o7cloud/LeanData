import Mathlib

namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3014_301449

theorem cubic_equation_solution (x : ℝ) (h : 9 / x^2 = x / 25) : x = (225 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3014_301449


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l3014_301403

theorem triangle_inequality_range (A B C : ℝ) (t : ℝ) : 
  0 < B → B ≤ π/3 → 
  (∀ x : ℝ, (x + 2 + Real.sin (2*B))^2 + (Real.sqrt 2 * t * Real.sin (B + π/4))^2 ≥ 1) →
  t ∈ Set.Ici 1 ∪ Set.Iic (-1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l3014_301403


namespace NUMINAMATH_CALUDE_part1_part2_l3014_301448

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x - 1

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ x - 3) ↔ (0 ≤ a ∧ a ≤ 8) :=
sorry

-- Part 2
theorem part2 (a : ℝ) (h : a < 0) :
  (∀ x : ℝ, f a x < 0 ↔ 
    (a = -1 ∧ x ≠ 1) ∨
    (-1 < a ∧ a < 0 ∧ (x < 1 ∨ x > -1/a)) ∨
    (a < -1 ∧ (x < -1/a ∨ x > 1))) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3014_301448


namespace NUMINAMATH_CALUDE_droid_coffee_ratio_l3014_301459

/-- The ratio of afternoon to morning coffee bean usage in Droid's coffee shop --/
def afternoon_to_morning_ratio (morning_bags : ℕ) (total_weekly_bags : ℕ) : ℚ :=
  let afternoon_ratio := (total_weekly_bags / 7 - morning_bags - 2 * morning_bags) / morning_bags
  afternoon_ratio

/-- Theorem stating that the ratio of afternoon to morning coffee bean usage is 3 --/
theorem droid_coffee_ratio :
  afternoon_to_morning_ratio 3 126 = 3 := by sorry

end NUMINAMATH_CALUDE_droid_coffee_ratio_l3014_301459


namespace NUMINAMATH_CALUDE_chord_length_concentric_circles_l3014_301444

/-- Given two concentric circles with radii R and r, where R > r, 
    and the area of the ring between them is 16π square inches,
    the length of a chord of the larger circle that is tangent to the smaller circle is 8 inches. -/
theorem chord_length_concentric_circles 
  (R r : ℝ) 
  (h1 : R > r) 
  (h2 : π * R^2 - π * r^2 = 16 * π) : 
  ∃ (c : ℝ), c = 8 ∧ c^2 = 4 * (R^2 - r^2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_concentric_circles_l3014_301444


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3014_301447

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} : 
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- The value of a for which the lines x + ay - 1 = 0 and (a-1)x + ay + 1 = 0 are parallel -/
theorem parallel_lines_a_value : 
  (∀ x y, x + a * y - 1 = 0 ↔ (a - 1) * x + a * y + 1 = 0) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3014_301447


namespace NUMINAMATH_CALUDE_circle_equation_l3014_301429

/-- The equation of a circle with center (1, 2) passing through the origin (0, 0) -/
theorem circle_equation : ∀ x y : ℝ, 
  (x - 1)^2 + (y - 2)^2 = 5 ↔ 
  (x - 1)^2 + (y - 2)^2 = (0 - 1)^2 + (0 - 2)^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_l3014_301429


namespace NUMINAMATH_CALUDE_ned_bomb_diffusal_l3014_301478

/-- Represents the problem of Ned racing to deactivate a time bomb -/
def BombDefusalProblem (total_flights : ℕ) (time_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ) : Prop :=
  let flights_gone := time_spent / time_per_flight
  let flights_left := total_flights - flights_gone
  let time_left := bomb_timer - (flights_left * time_per_flight)
  time_left = 17

/-- Theorem stating that Ned will have 17 seconds to diffuse the bomb -/
theorem ned_bomb_diffusal :
  BombDefusalProblem 20 11 72 165 :=
sorry

end NUMINAMATH_CALUDE_ned_bomb_diffusal_l3014_301478


namespace NUMINAMATH_CALUDE_total_fans_count_l3014_301463

/-- Represents the number of fans for each team -/
structure FanCounts where
  yankees : ℕ
  mets : ℕ
  red_sox : ℕ

/-- Calculates the total number of fans -/
def total_fans (fans : FanCounts) : ℕ :=
  fans.yankees + fans.mets + fans.red_sox

/-- Theorem: Given the ratios and number of Mets fans, prove the total number of fans is 360 -/
theorem total_fans_count (fans : FanCounts) 
  (yankees_mets_ratio : fans.yankees = 3 * fans.mets / 2)
  (mets_redsox_ratio : fans.red_sox = 5 * fans.mets / 4)
  (mets_count : fans.mets = 96) :
  total_fans fans = 360 := by
  sorry

#eval total_fans { yankees := 144, mets := 96, red_sox := 120 }

end NUMINAMATH_CALUDE_total_fans_count_l3014_301463


namespace NUMINAMATH_CALUDE_absolute_value_equation_l3014_301488

theorem absolute_value_equation (x : ℝ) : |x - 1| = 2*x → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l3014_301488


namespace NUMINAMATH_CALUDE_polynomial_value_for_special_x_l3014_301422

theorem polynomial_value_for_special_x :
  let x : ℝ := 1 / (2 - Real.sqrt 3)
  x^6 - 2 * Real.sqrt 3 * x^5 - x^4 + x^3 - 4 * x^2 + 2 * x - Real.sqrt 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_for_special_x_l3014_301422


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3014_301486

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ) ^ (3 * x) = 27 ^ (1/3) ∧ x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3014_301486


namespace NUMINAMATH_CALUDE_angle4_measure_l3014_301476

-- Define the triangle and its angles
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (angle4 : ℝ)
  (angle5 : ℝ)
  (angle6 : ℝ)

-- Define the theorem
theorem angle4_measure (t : Triangle) 
  (h1 : t.angle1 = 76)
  (h2 : t.angle2 = 27)
  (h3 : t.angle3 = 17)
  (h4 : t.angle1 + t.angle2 + t.angle3 + t.angle5 + t.angle6 = 180) -- Sum of angles in the large triangle
  (h5 : t.angle4 + t.angle5 + t.angle6 = 180) -- Sum of angles in the small triangle
  : t.angle4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle4_measure_l3014_301476


namespace NUMINAMATH_CALUDE_jills_number_satisfies_conditions_l3014_301453

def jills_favorite_number := 98

theorem jills_number_satisfies_conditions :
  -- 98 is even
  Even jills_favorite_number ∧
  -- 98 has repeating prime factors
  ∃ p : Nat, Prime p ∧ (jills_favorite_number % (p * p) = 0) ∧
  -- 7 is a prime factor of 98
  jills_favorite_number % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_jills_number_satisfies_conditions_l3014_301453


namespace NUMINAMATH_CALUDE_pole_length_l3014_301485

/-- The length of a pole that fits diagonally in a rectangular opening -/
theorem pole_length (w h : ℝ) (hw : w > 0) (hh : h > 0) : 
  (w + 4)^2 + (h + 2)^2 = 100 → w^2 + h^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_pole_length_l3014_301485


namespace NUMINAMATH_CALUDE_sum_external_angles_hexagon_l3014_301456

/-- A regular hexagon is a polygon with 6 sides of equal length and 6 equal angles -/
def RegularHexagon : Type := Unit

/-- The external angle of a polygon is the angle between one side and the extension of an adjacent side -/
def ExternalAngle (p : RegularHexagon) : ℝ := sorry

/-- The sum of external angles of a regular hexagon -/
def SumExternalAngles (p : RegularHexagon) : ℝ := sorry

/-- Theorem: The sum of the external angles of a regular hexagon is 360° -/
theorem sum_external_angles_hexagon (p : RegularHexagon) :
  SumExternalAngles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_external_angles_hexagon_l3014_301456


namespace NUMINAMATH_CALUDE_quadratic_solution_l3014_301452

theorem quadratic_solution (a b : ℝ) : 
  (1 : ℝ)^2 * a - (1 : ℝ) * b - 5 = 0 → 2023 + a - b = 2028 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l3014_301452


namespace NUMINAMATH_CALUDE_expression_evaluation_l3014_301458

-- Define the expression
def expression : ℕ → ℕ → ℕ := λ a b => (3^a + 7^b)^2 - (3^a - 7^b)^2

-- State the theorem
theorem expression_evaluation :
  expression 1003 1004 = 5292 * 441^500 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3014_301458


namespace NUMINAMATH_CALUDE_expression_evaluation_l3014_301473

theorem expression_evaluation : 
  (150^2 - 12^2) / (90^2 - 21^2) * ((90 + 21) * (90 - 21)) / ((150 + 12) * (150 - 12)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3014_301473


namespace NUMINAMATH_CALUDE_smallest_number_l3014_301421

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

-- Define the numbers in their respective bases
def num1 : List Nat := [5, 8]  -- 85 in base 9
def num2 : List Nat := [0, 1, 2]  -- 210 in base 6
def num3 : List Nat := [0, 0, 0, 1]  -- 1000 in base 4
def num4 : List Nat := [1, 1, 1, 1, 1, 1]  -- 111111 in base 2

-- Theorem statement
theorem smallest_number :
  to_base_10 num4 2 < to_base_10 num1 9 ∧
  to_base_10 num4 2 < to_base_10 num2 6 ∧
  to_base_10 num4 2 < to_base_10 num3 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3014_301421


namespace NUMINAMATH_CALUDE_quiz_contest_orderings_l3014_301439

theorem quiz_contest_orderings (n : ℕ) (h : n = 5) : Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_quiz_contest_orderings_l3014_301439


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3014_301493

/-- Two lines are parallel if their slopes are equal and they are not identical. -/
def are_parallel (m n : ℝ) : Prop :=
  (m = 1 ∧ n ≠ -1) ∨ (m = -1 ∧ n ≠ 1)

/-- The theorem states that two lines mx+y-n=0 and x+my+1=0 are parallel
    if and only if (m=1 and n≠-1) or (m=-1 and n≠1). -/
theorem parallel_lines_condition (m n : ℝ) :
  are_parallel m n ↔ ∀ x y : ℝ, (m * x + y - n = 0 ↔ x + m * y + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3014_301493


namespace NUMINAMATH_CALUDE_book_reading_time_l3014_301433

theorem book_reading_time (total_pages : ℕ) (planned_days : ℕ) (extra_pages : ℕ) 
    (h1 : total_pages = 960)
    (h2 : planned_days = 20)
    (h3 : extra_pages = 12) :
  (total_pages : ℚ) / ((total_pages / planned_days + extra_pages) : ℚ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l3014_301433


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_eight_l3014_301413

theorem sqrt_meaningful_iff_leq_eight (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 8 - x) ↔ x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_leq_eight_l3014_301413


namespace NUMINAMATH_CALUDE_fiftieth_term_l3014_301496

/-- Sequence defined as a_n = (n + 4) * x^(n-1) for n ≥ 1 -/
def a (n : ℕ) (x : ℝ) : ℝ := (n + 4) * x^(n - 1)

/-- The 50th term of the sequence is 54x^49 -/
theorem fiftieth_term (x : ℝ) : a 50 x = 54 * x^49 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_l3014_301496


namespace NUMINAMATH_CALUDE_range_of_a_l3014_301450

theorem range_of_a (a : ℝ) : 
  (∀ x, x > a → x^2 + x - 2 > 0) ∧ 
  (∃ x, x^2 + x - 2 > 0 ∧ x ≤ a) → 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3014_301450


namespace NUMINAMATH_CALUDE_jordan_machine_l3014_301417

theorem jordan_machine (x : ℚ) : ((3 * x - 6) / 2 + 9 = 27) → x = 14 := by
  sorry

end NUMINAMATH_CALUDE_jordan_machine_l3014_301417


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3014_301460

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3014_301460


namespace NUMINAMATH_CALUDE_rational_closure_l3014_301483

theorem rational_closure (x y : ℚ) (h : y ≠ 0) :
  (∃ a b : ℤ, (x + y = a / b ∧ b ≠ 0)) ∧
  (∃ c d : ℤ, (x - y = c / d ∧ d ≠ 0)) ∧
  (∃ e f : ℤ, (x * y = e / f ∧ f ≠ 0)) ∧
  (∃ g h : ℤ, (x / y = g / h ∧ h ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_rational_closure_l3014_301483


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3014_301445

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3014_301445


namespace NUMINAMATH_CALUDE_maria_score_l3014_301477

/-- Represents a math contest scoring system -/
structure ScoringSystem where
  correct_points : ℝ
  incorrect_penalty : ℝ

/-- Represents a contestant's performance in the math contest -/
structure ContestPerformance where
  total_questions : ℕ
  correct_answers : ℕ
  incorrect_answers : ℕ
  unanswered_questions : ℕ

/-- Calculates the total score for a contestant given their performance and the scoring system -/
def calculate_score (performance : ContestPerformance) (system : ScoringSystem) : ℝ :=
  (performance.correct_answers : ℝ) * system.correct_points -
  (performance.incorrect_answers : ℝ) * system.incorrect_penalty

/-- Theorem stating that Maria's score in the contest is 12.5 -/
theorem maria_score :
  let system : ScoringSystem := { correct_points := 1, incorrect_penalty := 0.25 }
  let performance : ContestPerformance := {
    total_questions := 30,
    correct_answers := 15,
    incorrect_answers := 10,
    unanswered_questions := 5
  }
  calculate_score performance system = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_maria_score_l3014_301477


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3014_301426

/-- A geometric sequence with third term 3 and fifth term 27 has first term 1/3 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) : 
  a * r^2 = 3 → a * r^4 = 27 → a = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3014_301426


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l3014_301465

theorem max_sum_given_constraints (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) :
  x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l3014_301465


namespace NUMINAMATH_CALUDE_equation_one_integral_root_l3014_301481

theorem equation_one_integral_root :
  ∃! x : ℤ, x - 9 / (x - 5 : ℚ) = 7 - 9 / (x - 5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equation_one_integral_root_l3014_301481


namespace NUMINAMATH_CALUDE_work_completion_time_l3014_301415

/-- The number of days A takes to complete the work alone -/
def a_days : ℝ := 12

/-- The number of days B takes to complete the work alone -/
def b_days : ℝ := 27.99999999999998

/-- The number of days A worked alone before B joined -/
def a_solo_days : ℝ := 2

/-- The total number of days it takes to complete the work when A and B work together -/
def total_days : ℝ := 9

theorem work_completion_time :
  let a_rate : ℝ := 1 / a_days
  let b_rate : ℝ := 1 / b_days
  let combined_rate : ℝ := a_rate + b_rate
  let work_done_by_a_solo : ℝ := a_rate * a_solo_days
  let remaining_work : ℝ := 1 - work_done_by_a_solo
  remaining_work / combined_rate + a_solo_days = total_days := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3014_301415


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l3014_301420

theorem lucky_larry_coincidence :
  let a : ℚ := 2
  let b : ℚ := 3
  let c : ℚ := 4
  let d : ℚ := 5
  let f : ℚ := 4/5
  (a - b - c + d * f) = (a - (b - (c - (d * f)))) := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l3014_301420


namespace NUMINAMATH_CALUDE_earliest_meeting_time_is_440_l3014_301418

/-- Represents the lap time in minutes for each runner -/
structure LapTime where
  charlie : ℕ
  ben : ℕ
  laura : ℕ

/-- Calculates the earliest meeting time in minutes -/
def earliest_meeting_time (lt : LapTime) : ℕ :=
  Nat.lcm (Nat.lcm lt.charlie lt.ben) lt.laura

/-- Theorem: Given the specific lap times, the earliest meeting time is 440 minutes -/
theorem earliest_meeting_time_is_440 :
  earliest_meeting_time ⟨5, 8, 11⟩ = 440 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_is_440_l3014_301418


namespace NUMINAMATH_CALUDE_three_leaf_clover_count_l3014_301408

theorem three_leaf_clover_count :
  ∀ (total_leaves : ℕ) (three_leaf_count : ℕ),
    total_leaves = 1000 →
    3 * three_leaf_count + 4 = total_leaves →
    three_leaf_count = 332 := by
  sorry

end NUMINAMATH_CALUDE_three_leaf_clover_count_l3014_301408


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3014_301427

-- Define the circumference of the circle
def circumference : ℝ := 36

-- Theorem stating that the area of a circle with circumference 36 cm is 324/π cm²
theorem circle_area_from_circumference :
  (π * (circumference / (2 * π))^2) = 324 / π := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3014_301427


namespace NUMINAMATH_CALUDE_measure_string_l3014_301414

theorem measure_string (string_length : ℚ) (h : string_length = 2/3) :
  string_length - (1/4 * string_length) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_measure_string_l3014_301414


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3014_301498

theorem fraction_multiplication : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5060 = 759 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3014_301498


namespace NUMINAMATH_CALUDE_prob_both_white_is_zero_l3014_301467

/-- Two boxes containing marbles -/
structure TwoBoxes where
  box1 : Finset ℕ
  box2 : Finset ℕ
  total_marbles : box1.card + box2.card = 36
  box1_black : ∀ m ∈ box1, m = 0  -- 0 represents black marbles
  prob_both_black : (box1.card : ℚ) / 36 * (box2.filter (λ m => m = 0)).card / box2.card = 18 / 25

/-- The probability of drawing two white marbles -/
def prob_both_white (boxes : TwoBoxes) : ℚ :=
  (boxes.box1.filter (λ m => m ≠ 0)).card / boxes.box1.card *
  (boxes.box2.filter (λ m => m ≠ 0)).card / boxes.box2.card

theorem prob_both_white_is_zero (boxes : TwoBoxes) : prob_both_white boxes = 0 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_is_zero_l3014_301467


namespace NUMINAMATH_CALUDE_system_solution_l3014_301457

theorem system_solution (a b : ℤ) : 
  (∃ x y : ℤ, a * x + 5 * y = 15 ∧ 4 * x - b * y = -2) →
  (4 * (-3) - b * (-1) = -2) →
  (a * 5 + 5 * 4 = 15) →
  a^2023 + (-1/10 * b : ℚ)^2023 = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3014_301457


namespace NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_unique_max_value_l3014_301474

theorem max_value_constraint (x y z : ℝ) :
  x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 →
  7*x + 10*y + z ≤ 55 :=
by sorry

theorem max_value_attained :
  ∃ x y z : ℝ, x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 ∧ 7*x + 10*y + z = 55 :=
by sorry

theorem unique_max_value (x y z : ℝ) :
  x^2 + 2*x + (1/5)*y^2 + 7*z^2 = 6 ∧ 7*x + 10*y + z = 55 →
  x = -13/62 ∧ y = 175/31 ∧ z = 1/62 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_max_value_attained_unique_max_value_l3014_301474


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_for_given_intersection_l3014_301436

-- Define set A
def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem intersection_A_complement_B_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem m_value_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_when_m_3_m_value_for_given_intersection_l3014_301436


namespace NUMINAMATH_CALUDE_car_average_speed_l3014_301416

theorem car_average_speed 
  (total_time : ℝ) 
  (first_interval : ℝ) 
  (first_speed : ℝ) 
  (second_speed : ℝ) 
  (h1 : total_time = 8) 
  (h2 : first_interval = 4) 
  (h3 : first_speed = 70) 
  (h4 : second_speed = 60) : 
  (first_speed * first_interval + second_speed * (total_time - first_interval)) / total_time = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_average_speed_l3014_301416


namespace NUMINAMATH_CALUDE_triangle_properties_l3014_301442

open Real

structure Triangle (A B C : ℝ) where
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_properties (A B C : ℝ) (h : Triangle A B C) 
  (h1 : A + B = 3 * C) (h2 : 2 * sin (A - C) = sin B) (h3 : ∃ (AB : ℝ), AB = 5) :
  sin A = (3 * sqrt 10) / 10 ∧ 
  ∃ (height : ℝ), height = 6 ∧ 
    height * 5 / 2 = (sqrt 10 * 3 * sqrt 5 * sqrt 2) / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l3014_301442


namespace NUMINAMATH_CALUDE_roots_equation_l3014_301472

theorem roots_equation (c d r s : ℝ) : 
  (c^2 - 7*c + 12 = 0) →
  (d^2 - 7*d + 12 = 0) →
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) →
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) →
  s = 169/12 := by
sorry

end NUMINAMATH_CALUDE_roots_equation_l3014_301472


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l3014_301454

theorem cubic_sum_over_product (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 33 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l3014_301454


namespace NUMINAMATH_CALUDE_smallest_five_digit_base3_palindrome_is_10001_l3014_301431

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Returns the number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_five_digit_base3_palindrome_is_10001 :
  ∃ (otherBase : ℕ),
    otherBase ≠ 3 ∧
    otherBase > 1 ∧
    isPalindrome 10001 3 ∧
    numDigits 10001 3 = 5 ∧
    isPalindrome (baseConvert 10001 3 otherBase) otherBase ∧
    numDigits (baseConvert 10001 3 otherBase) otherBase = 3 ∧
    ∀ (n : ℕ),
      n < 10001 →
      (isPalindrome n 3 ∧ numDigits n 3 = 5) →
      ¬∃ (b : ℕ), b ≠ 3 ∧ b > 1 ∧
        isPalindrome (baseConvert n 3 b) b ∧
        numDigits (baseConvert n 3 b) b = 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_base3_palindrome_is_10001_l3014_301431


namespace NUMINAMATH_CALUDE_student_average_grade_previous_year_l3014_301440

/-- Represents the average grade of a student for a given year -/
structure YearlyAverage where
  courses : ℕ
  average : ℝ

/-- Calculates the total points for a year -/
def totalPoints (ya : YearlyAverage) : ℝ := ya.courses * ya.average

theorem student_average_grade_previous_year 
  (last_year : YearlyAverage)
  (prev_year : YearlyAverage)
  (h1 : last_year.courses = 6)
  (h2 : last_year.average = 100)
  (h3 : prev_year.courses = 5)
  (h4 : (totalPoints last_year + totalPoints prev_year) / (last_year.courses + prev_year.courses) = 81) :
  prev_year.average = 58.2 := by
  sorry


end NUMINAMATH_CALUDE_student_average_grade_previous_year_l3014_301440


namespace NUMINAMATH_CALUDE_base4_calculation_l3014_301482

/-- Convert a number from base 4 to base 10 -/
def base4_to_base10 (n : ℕ) : ℕ := sorry

/-- Convert a number from base 10 to base 4 -/
def base10_to_base4 (n : ℕ) : ℕ := sorry

/-- Multiplication in base 4 -/
def mul_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a * base4_to_base10 b)

/-- Division in base 4 -/
def div_base4 (a b : ℕ) : ℕ := 
  base10_to_base4 (base4_to_base10 a / base4_to_base10 b)

theorem base4_calculation : 
  div_base4 (mul_base4 231 24) 3 = 2310 := by sorry

end NUMINAMATH_CALUDE_base4_calculation_l3014_301482


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l3014_301425

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (math_copies : ℕ) (novel_copies : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial math_copies * Nat.factorial novel_copies)

/-- Theorem stating the number of arrangements for the given problem -/
theorem book_arrangement_problem :
  arrange_books 7 3 2 = 420 := by sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l3014_301425


namespace NUMINAMATH_CALUDE_exclusive_proposition_range_l3014_301434

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

def q (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 - 4 * x + m > 0

-- Define the range of m
def m_range (m : ℝ) : Prop := m < 1/2 ∨ m > 2

-- Theorem statement
theorem exclusive_proposition_range :
  ∀ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_exclusive_proposition_range_l3014_301434


namespace NUMINAMATH_CALUDE_distance_between_cars_l3014_301443

/-- The distance between two cars on a road after they travel towards each other -/
theorem distance_between_cars (initial_distance car1_distance car2_distance : ℝ) :
  initial_distance = 150 ∧ 
  car1_distance = 50 ∧ 
  car2_distance = 35 →
  initial_distance - (car1_distance + car2_distance) = 65 := by
  sorry


end NUMINAMATH_CALUDE_distance_between_cars_l3014_301443


namespace NUMINAMATH_CALUDE_exists_natural_number_with_seventh_eighth_root_natural_l3014_301406

theorem exists_natural_number_with_seventh_eighth_root_natural :
  ∃ (n : ℕ), n > 1 ∧ ∃ (m : ℕ), n^(7/8) = m := by
  sorry

end NUMINAMATH_CALUDE_exists_natural_number_with_seventh_eighth_root_natural_l3014_301406


namespace NUMINAMATH_CALUDE_project_hours_ratio_l3014_301424

/-- Represents the hours charged by Kate -/
def kate_hours : ℕ := sorry

/-- Represents the hours charged by Pat -/
def pat_hours : ℕ := 2 * kate_hours

/-- Represents the hours charged by Mark -/
def mark_hours : ℕ := kate_hours + 110

/-- The total hours charged by all three -/
def total_hours : ℕ := 198

theorem project_hours_ratio :
  pat_hours + kate_hours + mark_hours = total_hours ∧
  pat_hours.gcd mark_hours = pat_hours ∧
  (pat_hours / pat_hours.gcd mark_hours) = 1 ∧
  (mark_hours / pat_hours.gcd mark_hours) = 3 :=
sorry

end NUMINAMATH_CALUDE_project_hours_ratio_l3014_301424


namespace NUMINAMATH_CALUDE_optimal_pen_area_optimal_parallel_side_l3014_301492

/-- The length of the side parallel to the shed that maximizes the rectangular goat pen area -/
def optimal_parallel_side_length : ℝ := 50

/-- The total length of fence available -/
def total_fence_length : ℝ := 100

/-- The length of the shed -/
def shed_length : ℝ := 300

/-- The area of the pen as a function of the perpendicular side length -/
def pen_area (y : ℝ) : ℝ := y * (total_fence_length - 2 * y)

theorem optimal_pen_area :
  ∀ y : ℝ, 0 < y → y < total_fence_length / 2 →
  pen_area y ≤ pen_area (total_fence_length / 4) :=
sorry

theorem optimal_parallel_side :
  optimal_parallel_side_length = total_fence_length / 2 :=
sorry

end NUMINAMATH_CALUDE_optimal_pen_area_optimal_parallel_side_l3014_301492


namespace NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l3014_301490

theorem three_digit_difference_divisible_by_nine :
  ∀ (a b c : ℕ), 
    a ≤ 9 → b ≤ 9 → c ≤ 9 → a ≠ 0 →
    ∃ (k : ℤ), (100 * a + 10 * b + c) - (a + b + c) = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_difference_divisible_by_nine_l3014_301490


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l3014_301441

theorem modular_inverse_of_5_mod_23 :
  ∃ x : ℕ, x ≤ 22 ∧ (5 * x) % 23 = 1 :=
by
  use 14
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_23_l3014_301441


namespace NUMINAMATH_CALUDE_acid_concentration_percentage_l3014_301409

/-- 
Given a solution with 1.6 litres of pure acid in 8 litres of total volume,
prove that the percentage concentration of the acid is 20%.
-/
theorem acid_concentration_percentage (pure_acid : ℝ) (total_volume : ℝ) :
  pure_acid = 1.6 →
  total_volume = 8 →
  (pure_acid / total_volume) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_acid_concentration_percentage_l3014_301409


namespace NUMINAMATH_CALUDE_shrimp_trap_problem_l3014_301497

theorem shrimp_trap_problem (victor_shrimp : ℕ) (austin_shrimp : ℕ) :
  victor_shrimp = 26 →
  (victor_shrimp + austin_shrimp + (victor_shrimp + austin_shrimp) / 2) * 7 / 11 = 42 →
  austin_shrimp + 8 = victor_shrimp :=
by sorry

end NUMINAMATH_CALUDE_shrimp_trap_problem_l3014_301497


namespace NUMINAMATH_CALUDE_equation_unique_solution_l3014_301438

theorem equation_unique_solution :
  ∃! x : ℝ, (3 * x^2 - 18 * x) / (x^2 - 6 * x) = x^2 - 4 * x + 3 ∧ x ≠ 0 ∧ x ≠ 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l3014_301438


namespace NUMINAMATH_CALUDE_sum_of_digits_nine_times_ascending_l3014_301446

/-- A function that checks if a natural number has digits in ascending order -/
def has_ascending_digits (n : ℕ) : Prop :=
  ∀ i j, i < j → (n.digits 10).get i < (n.digits 10).get j

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem: For any number A with digits in ascending order, 
    the sum of digits of 9 * A is always 9 -/
theorem sum_of_digits_nine_times_ascending (A : ℕ) 
  (h : has_ascending_digits A) : sum_of_digits (9 * A) = 9 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_nine_times_ascending_l3014_301446


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3014_301499

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 32) :
  let side_length := face_perimeter / 4
  let volume := side_length ^ 3
  volume = 512 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l3014_301499


namespace NUMINAMATH_CALUDE_animals_food_consumption_l3014_301461

/-- The total food consumption for a group of animals in one month -/
def total_food_consumption (num_animals : ℕ) (food_per_animal : ℕ) : ℕ :=
  num_animals * food_per_animal

/-- Theorem: Given 6 animals, with each animal eating 4 kg of food in one month,
    the total food consumption for all animals in one month is 24 kg -/
theorem animals_food_consumption :
  total_food_consumption 6 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_animals_food_consumption_l3014_301461


namespace NUMINAMATH_CALUDE_min_value_shifted_quadratic_l3014_301469

/-- Given a quadratic function f(x) = x^2 + 4x + 7 - a with minimum value 2,
    prove that g(x) = f(x - 2015) also has minimum value 2 -/
theorem min_value_shifted_quadratic (a : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 + 4*x + 7 - a) ∧ 
   (∃ m, m = 2 ∧ ∀ x, f x ≥ m)) →
  (∃ (g : ℝ → ℝ), (∀ x, g x = (x - 2015)^2 + 4*(x - 2015) + 7 - a) ∧ 
   (∃ m, m = 2 ∧ ∀ x, g x ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_shifted_quadratic_l3014_301469


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l3014_301432

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 43 ∧ 
  corrected_mean = 36.5 →
  (n : ℝ) * ((n * corrected_mean - (correct_value - wrong_value)) / n) = 36.1 * n :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l3014_301432


namespace NUMINAMATH_CALUDE_pencil_distribution_ways_l3014_301471

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) (min_pencils : ℕ) : ℕ :=
  Nat.choose (total_pencils - num_friends * min_pencils + num_friends - 1) (num_friends - 1)

/-- Theorem: There are 6 ways to distribute 8 pencils among 3 friends with at least 2 pencils each -/
theorem pencil_distribution_ways : distribute_pencils 8 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_ways_l3014_301471


namespace NUMINAMATH_CALUDE_existence_of_special_number_l3014_301466

theorem existence_of_special_number : 
  ∃ (n : ℕ) (N : ℕ), n > 2 ∧ 
  N = 2 * 10^(n+1) - 9 ∧ 
  N % 1991 = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_number_l3014_301466


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3014_301484

/-- Given an arithmetic sequence {a_n} where a_1 = 13 and a_4 = 1,
    prove that the common difference d is -4. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)  -- The sequence a_n
  (h1 : a 1 = 13)  -- a_1 = 13
  (h4 : a 4 = 1)   -- a_4 = 1
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = -4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3014_301484


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3014_301435

/-- The line equation parameterized by k -/
def line_equation (x y k : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + (2 - 3*k) = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (0, 1)

/-- Theorem stating that the fixed point is the unique point that satisfies the line equation for all k -/
theorem fixed_point_on_line :
  ∀ (k : ℝ), line_equation (fixed_point.1) (fixed_point.2) k ∧
  ∀ (x y : ℝ), (∀ (k : ℝ), line_equation x y k) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3014_301435


namespace NUMINAMATH_CALUDE_mary_earnings_proof_l3014_301428

/-- Calculates Mary's weekly earnings after deductions --/
def maryWeeklyEarnings (
  maxHours : Nat)
  (regularRate : ℚ)
  (overtimeRateIncrease : ℚ)
  (additionalRateIncrease : ℚ)
  (regularHours : Nat)
  (overtimeHours : Nat)
  (taxRate1 : ℚ)
  (taxRate2 : ℚ)
  (taxRate3 : ℚ)
  (taxThreshold1 : ℚ)
  (taxThreshold2 : ℚ)
  (insuranceFee : ℚ)
  (weekendBonus : ℚ)
  (weekendShiftHours : Nat) : ℚ :=
  sorry

theorem mary_earnings_proof :
  maryWeeklyEarnings 70 10 0.3 0.6 40 20 0.15 0.1 0.25 400 600 50 75 8 = 691.25 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_proof_l3014_301428


namespace NUMINAMATH_CALUDE_max_product_constraint_l3014_301451

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 6 * a + 8 * b = 48) :
  a * b ≤ 12 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 6 * a₀ + 8 * b₀ = 48 ∧ a₀ * b₀ = 12 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l3014_301451


namespace NUMINAMATH_CALUDE_unique_solution_exists_l3014_301480

theorem unique_solution_exists : ∃! x : ℕ, 
  x < 5311735 ∧
  x % 5 = 0 ∧
  x % 715 = 10 ∧
  x % 247 = 140 ∧
  x % 391 = 245 ∧
  x % 187 = 109 ∧
  x = 10020 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l3014_301480


namespace NUMINAMATH_CALUDE_problem_solution_l3014_301475

theorem problem_solution (x y z : ℚ) 
  (hx : x = 1/3) (hy : y = 1/2) (hz : z = 5/8) :
  x * y * (1 - z) = 1/16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3014_301475


namespace NUMINAMATH_CALUDE_integer_triple_solution_l3014_301464

theorem integer_triple_solution (a b c : ℤ) 
  (eq1 : a + b * c = 2017) 
  (eq2 : b + c * a = 8) : 
  c ∈ ({-6, 0, 2, 8} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_integer_triple_solution_l3014_301464


namespace NUMINAMATH_CALUDE_janet_walk_results_l3014_301412

/-- Represents Janet's walk in the city --/
structure JanetWalk where
  blocks_north : ℕ
  blocks_west : ℕ
  blocks_south : ℕ
  walking_speed : ℕ

/-- Calculates the time Janet needs to get home --/
def time_to_home (walk : JanetWalk) : ℚ :=
  (walk.blocks_west : ℚ) / walk.walking_speed

/-- Calculates the ratio of blocks walked east to south --/
def east_south_ratio (walk : JanetWalk) : ℚ × ℚ :=
  (walk.blocks_west, walk.blocks_south)

/-- Theorem stating the results of Janet's walk --/
theorem janet_walk_results (walk : JanetWalk) 
  (h1 : walk.blocks_north = 3)
  (h2 : walk.blocks_west = 7 * walk.blocks_north)
  (h3 : walk.blocks_south = 8)
  (h4 : walk.walking_speed = 2) :
  time_to_home walk = 21/2 ∧ east_south_ratio walk = (21, 8) := by
  sorry

#eval time_to_home { blocks_north := 3, blocks_west := 21, blocks_south := 8, walking_speed := 2 }
#eval east_south_ratio { blocks_north := 3, blocks_west := 21, blocks_south := 8, walking_speed := 2 }

end NUMINAMATH_CALUDE_janet_walk_results_l3014_301412


namespace NUMINAMATH_CALUDE_shipment_arrival_time_l3014_301437

/-- Calculates the number of days until a shipment arrives at a warehouse -/
def daysUntilArrival (daysSinceDeparture : ℕ) (navigationDays : ℕ) (customsDays : ℕ) (warehouseArrivalDay : ℕ) : ℕ :=
  let daysInPort := daysSinceDeparture - navigationDays
  let daysAfterCustoms := daysInPort - customsDays
  warehouseArrivalDay - daysAfterCustoms

theorem shipment_arrival_time :
  daysUntilArrival 30 21 4 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_shipment_arrival_time_l3014_301437


namespace NUMINAMATH_CALUDE_sin_6phi_l3014_301470

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_sin_6phi_l3014_301470


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3014_301487

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := i / (1 - i)
  Complex.im z = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3014_301487


namespace NUMINAMATH_CALUDE_absolute_value_of_z_l3014_301405

theorem absolute_value_of_z (r : ℝ) (z : ℂ) 
  (hr : |r| > 2) 
  (hz : z - 1/z = r) : 
  Complex.abs z = Real.sqrt ((r^2 / 2) + 1) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_z_l3014_301405


namespace NUMINAMATH_CALUDE_knight_moves_equality_on_7x7_l3014_301462

/-- Represents a position on a chessboard --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a knight's move on a chessboard --/
inductive KnightMove : Position → Position → Prop
  | move_1 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 1, y + 2⟩
  | move_2 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 2, y + 1⟩
  | move_3 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 2, y - 1⟩
  | move_4 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x + 1, y - 2⟩
  | move_5 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 1, y - 2⟩
  | move_6 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 2, y - 1⟩
  | move_7 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 2, y + 1⟩
  | move_8 (x y : Nat) : KnightMove ⟨x, y⟩ ⟨x - 1, y + 2⟩

/-- Minimum number of moves for a knight to reach a target position from a start position --/
def minMoves (start target : Position) : Nat :=
  sorry

theorem knight_moves_equality_on_7x7 :
  let start := ⟨0, 0⟩
  let topRight := ⟨6, 6⟩
  let bottomRight := ⟨6, 0⟩
  minMoves start topRight = minMoves start bottomRight :=
by sorry

end NUMINAMATH_CALUDE_knight_moves_equality_on_7x7_l3014_301462


namespace NUMINAMATH_CALUDE_min_value_xy_l3014_301430

theorem min_value_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = x + 4 * y + 5) :
  x * y ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_l3014_301430


namespace NUMINAMATH_CALUDE_functions_same_domain_range_not_necessarily_equal_l3014_301423

theorem functions_same_domain_range_not_necessarily_equal :
  ∃ (A B : Type) (f g : A → B), (∀ x : A, ∃ y : B, f x = y ∧ g x = y) ∧ f ≠ g :=
sorry

end NUMINAMATH_CALUDE_functions_same_domain_range_not_necessarily_equal_l3014_301423


namespace NUMINAMATH_CALUDE_locus_of_centers_l3014_301402

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

-- Define external tangency to C1
def externally_tangent_to_C1 (a b r : ℝ) : Prop := a^2 + b^2 = (r + 2)^2

-- Define internal tangency to C2
def internally_tangent_to_C2 (a b r : ℝ) : Prop := (a - 3)^2 + b^2 = (5 - r)^2

-- Define the locus equation
def locus_equation (a b : ℝ) : Prop := 13 * a^2 + 49 * b^2 - 12 * a - 1 = 0

-- Theorem statement
theorem locus_of_centers :
  ∀ a b : ℝ,
  (∃ r : ℝ, externally_tangent_to_C1 a b r ∧ internally_tangent_to_C2 a b r) ↔
  locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l3014_301402


namespace NUMINAMATH_CALUDE_science_club_officer_selection_l3014_301491

def science_club_officers (n : ℕ) (k : ℕ) (special_members : ℕ) : ℕ :=
  (n - special_members).choose k + special_members * (special_members - 1) * (n - special_members)

theorem science_club_officer_selection :
  science_club_officers 25 3 2 = 10764 :=
by sorry

end NUMINAMATH_CALUDE_science_club_officer_selection_l3014_301491


namespace NUMINAMATH_CALUDE_min_distance_sum_l3014_301494

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- The left focus of the hyperbola -/
def F : ℝ × ℝ := sorry

/-- Point A -/
def A : ℝ × ℝ := (1, 4)

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- A point is on the right branch of the hyperbola -/
def on_right_branch (p : ℝ × ℝ) : Prop :=
  hyperbola p.1 p.2 ∧ p.1 > 0

theorem min_distance_sum :
  ∀ P : ℝ × ℝ, on_right_branch P →
    distance P F + distance P A ≥ 9 ∧
    ∃ Q : ℝ × ℝ, on_right_branch Q ∧ distance Q F + distance Q A = 9 :=
sorry

end NUMINAMATH_CALUDE_min_distance_sum_l3014_301494


namespace NUMINAMATH_CALUDE_divisible_by_120_l3014_301410

theorem divisible_by_120 (n : ℤ) : ∃ k : ℤ, n^5 - 5*n^3 + 4*n = 120*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l3014_301410


namespace NUMINAMATH_CALUDE_girl_walking_distance_l3014_301495

/-- The distance traveled by a girl walking at a constant speed for a given time. -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: A girl walking at 5 kmph for 6 hours travels 30 kilometers. -/
theorem girl_walking_distance :
  let speed : ℝ := 5
  let time : ℝ := 6
  distance_traveled speed time = 30 := by
  sorry

end NUMINAMATH_CALUDE_girl_walking_distance_l3014_301495


namespace NUMINAMATH_CALUDE_more_squirrels_than_nuts_l3014_301455

def num_squirrels : ℕ := 4
def num_nuts : ℕ := 2

theorem more_squirrels_than_nuts : num_squirrels - num_nuts = 2 := by
  sorry

end NUMINAMATH_CALUDE_more_squirrels_than_nuts_l3014_301455


namespace NUMINAMATH_CALUDE_blueberry_pies_l3014_301411

theorem blueberry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 36 →
  apple_ratio = 3 →
  blueberry_ratio = 4 →
  cherry_ratio = 5 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_pies_l3014_301411


namespace NUMINAMATH_CALUDE_cube_color_probability_l3014_301479

/-- Represents the three possible colors for a cube face -/
inductive Color
  | Red
  | Blue
  | Yellow

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Fin 6 → Color

/-- The probability of each color -/
def colorProb : Color → ℚ
  | _ => 1/3

/-- Checks if a cube configuration satisfies the condition -/
def satisfiesCondition (c : Cube) : Bool :=
  sorry -- Implementation details omitted

/-- Calculates the probability of a cube satisfying the condition -/
noncomputable def probabilityOfSatisfyingCondition : ℚ :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem cube_color_probability :
  probabilityOfSatisfyingCondition = 73/243 :=
sorry

end NUMINAMATH_CALUDE_cube_color_probability_l3014_301479


namespace NUMINAMATH_CALUDE_rectangular_enclosure_properties_l3014_301489

/-- Represents the area of a rectangular enclosure with perimeter 32 meters and side length x -/
def area (x : ℝ) : ℝ := -x^2 + 16*x

/-- Theorem stating the properties of the rectangular enclosure -/
theorem rectangular_enclosure_properties :
  ∀ x : ℝ, 0 < x → x < 16 →
  (∀ y : ℝ, y = area x → 
    (y = 60 → (x = 6 ∨ x = 10)) ∧
    (y ≤ 64) ∧
    (y = 64 ↔ x = 8)) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_enclosure_properties_l3014_301489


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3014_301407

def f (x : ℝ) : ℝ := x * abs (x - 2)

theorem inequality_solution_set (x : ℝ) :
  f (Real.sqrt 2 - x) ≤ f 1 ↔ x ≥ -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3014_301407


namespace NUMINAMATH_CALUDE_store_profit_l3014_301401

/-- The profit made by the store selling New Year cards -/
theorem store_profit (cost_price : ℚ) (total_sales : ℚ) (n : ℕ) (selling_price : ℚ) : 
  cost_price = 21/100 ∧ 
  total_sales = 1457/100 ∧ 
  n * selling_price = total_sales ∧ 
  selling_price ≤ 2 * cost_price →
  n * (selling_price - cost_price) = 47/10 := by
  sorry

#check store_profit

end NUMINAMATH_CALUDE_store_profit_l3014_301401


namespace NUMINAMATH_CALUDE_sixteen_horses_walking_legs_l3014_301468

/-- Given a number of horses and an equal number of men, with half riding and half walking,
    calculate the number of legs walking on the ground. -/
def legs_walking (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_walking_men := num_men / 2
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  men_legs + horse_legs

/-- Theorem stating that with 16 horses and men, half riding and half walking,
    there are 80 legs walking on the ground. -/
theorem sixteen_horses_walking_legs :
  legs_walking 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_horses_walking_legs_l3014_301468


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3014_301400

/-- Given a rectangle with specific properties, prove its perimeter is 92 cm -/
theorem rectangle_perimeter (width length : ℕ) : 
  width = 34 ∧ 
  width % 4 = 2 ∧ 
  (width / 4) * length = 24 → 
  2 * (width + length) = 92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3014_301400


namespace NUMINAMATH_CALUDE_a_sequence_property_l3014_301404

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | (n + 3) => a (n + 1) + 1998 * a n

theorem a_sequence_property (n : ℕ) (h : n > 0) :
  a (2 * n - 1) = 2 * a n * a (n + 1) + 1998 * a (n - 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_sequence_property_l3014_301404


namespace NUMINAMATH_CALUDE_least_k_value_l3014_301419

/-- The number of factors in the original equation -/
def n : ℕ := 2016

/-- The total number of factors on both sides of the equation -/
def total_factors : ℕ := 2 * n

/-- A function representing the left-hand side of the equation after erasing factors -/
def left_side (k : ℕ) (x : ℝ) : ℝ := sorry

/-- A function representing the right-hand side of the equation after erasing factors -/
def right_side (k : ℕ) (x : ℝ) : ℝ := sorry

/-- Predicate to check if the equation has no real solutions after erasing k factors -/
def no_real_solutions (k : ℕ) : Prop :=
  ∀ x : ℝ, left_side k x ≠ right_side k x

/-- Predicate to check if at least one factor remains on each side after erasing k factors -/
def factors_remain (k : ℕ) : Prop :=
  k < total_factors

/-- The main theorem stating that 2016 is the least value of k satisfying the conditions -/
theorem least_k_value :
  (∀ k < n, ¬(no_real_solutions k ∧ factors_remain k)) ∧
  (no_real_solutions n ∧ factors_remain n) :=
sorry

end NUMINAMATH_CALUDE_least_k_value_l3014_301419
