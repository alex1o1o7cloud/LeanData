import Mathlib

namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l759_75908

def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l759_75908


namespace NUMINAMATH_CALUDE_nathan_gumballs_l759_75945

/-- Given that Nathan ate a total number of gumballs and finished a certain number of whole boxes
with no gumballs left, this function calculates the number of gumballs in each package. -/
def gumballs_per_package (total_gumballs : ℕ) (boxes_finished : ℕ) : ℕ :=
  total_gumballs / boxes_finished

theorem nathan_gumballs (total_gumballs : ℕ) (boxes_finished : ℕ) 
  (h1 : total_gumballs = 20) 
  (h2 : boxes_finished = 4) 
  (h3 : total_gumballs % boxes_finished = 0) : 
  gumballs_per_package total_gumballs boxes_finished = 5 := by
  sorry

end NUMINAMATH_CALUDE_nathan_gumballs_l759_75945


namespace NUMINAMATH_CALUDE_quadratic_roots_l759_75968

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  point_neg_two : a * (-2)^2 + b * (-2) + c = 12
  point_zero : c = -8
  point_one : a + b + c = -12
  point_three : a * 3^2 + b * 3 + c = -8

/-- The theorem statement -/
theorem quadratic_roots (f : QuadraticFunction) :
  let roots := {x : ℝ | f.a * x^2 + f.b * x + f.c + 8 = 0}
  roots = {0, 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l759_75968


namespace NUMINAMATH_CALUDE_water_remaining_l759_75922

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/8 → remaining = initial - used → remaining = 13/8 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l759_75922


namespace NUMINAMATH_CALUDE_solve_weeks_worked_problem_l759_75980

/-- Represents the problem of calculating the number of weeks worked --/
def WeeksWorkedProblem (regular_days_per_week : ℕ) 
                       (hours_per_day : ℕ) 
                       (regular_pay_rate : ℚ) 
                       (overtime_pay_rate : ℚ) 
                       (total_earnings : ℚ) 
                       (total_hours : ℕ) : Prop :=
  let regular_hours_per_week := regular_days_per_week * hours_per_day
  ∃ (weeks_worked : ℕ),
    let regular_hours := weeks_worked * regular_hours_per_week
    let overtime_hours := total_hours - regular_hours
    regular_hours * regular_pay_rate + overtime_hours * overtime_pay_rate = total_earnings ∧
    weeks_worked = 4

/-- The main theorem stating the solution to the problem --/
theorem solve_weeks_worked_problem :
  WeeksWorkedProblem 6 10 (210/100) (420/100) 525 245 := by
  sorry

#check solve_weeks_worked_problem

end NUMINAMATH_CALUDE_solve_weeks_worked_problem_l759_75980


namespace NUMINAMATH_CALUDE_subset_implies_m_squared_l759_75971

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B : Set ℝ := {3, 4}

-- State the theorem
theorem subset_implies_m_squared (m : ℝ) : B ⊆ A m → (m = 2 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_squared_l759_75971


namespace NUMINAMATH_CALUDE_expand_expression_l759_75936

theorem expand_expression (x y z : ℝ) : 
  (x + 12) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 36 * y + 24 * z + 180 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l759_75936


namespace NUMINAMATH_CALUDE_acquaintance_pigeonhole_l759_75943

theorem acquaintance_pigeonhole (n : ℕ) (h : n ≥ 2) :
  ∃ (i j : Fin n), i ≠ j ∧ 
  ∃ (f : Fin n → Fin n), (∀ k, f k < n) ∧ f i = f j :=
by
  sorry

end NUMINAMATH_CALUDE_acquaintance_pigeonhole_l759_75943


namespace NUMINAMATH_CALUDE_city_inhabitants_problem_l759_75911

theorem city_inhabitants_problem :
  ∃ (n : ℕ), 
    (∃ (m : ℕ), n^2 + 100 = m^2 + 1) ∧ 
    (∃ (k : ℕ), n^2 + 200 = k^2) ∧ 
    n = 49 := by
  sorry

end NUMINAMATH_CALUDE_city_inhabitants_problem_l759_75911


namespace NUMINAMATH_CALUDE_chord_length_when_m_1_shortest_chord_line_equation_l759_75917

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 11 = 0

-- Define the line l
def line_l (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Define the chord length function
noncomputable def chord_length (m : ℝ) : ℝ := sorry

-- Define the shortest chord condition
def is_shortest_chord (m : ℝ) : Prop := 
  ∀ m', chord_length m ≤ chord_length m'

-- Theorem 1: Chord length when m = 1
theorem chord_length_when_m_1 : 
  chord_length 1 = 6 * Real.sqrt 13 / 13 := sorry

-- Theorem 2: Equation of line l for shortest chord
theorem shortest_chord_line_equation :
  ∃ m, is_shortest_chord m ∧ 
    ∀ x y, line_l m x y ↔ x - y - 2 = 0 := sorry

end NUMINAMATH_CALUDE_chord_length_when_m_1_shortest_chord_line_equation_l759_75917


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l759_75910

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (x < -1/3 ∨ x > 1/2) ↔ a*x^2 + b*x + 2 < 0) → 
  a - b = -14 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l759_75910


namespace NUMINAMATH_CALUDE_quadratic_even_deductive_reasoning_l759_75902

-- Definition of an even function
def IsEvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Definition of a quadratic function
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c

-- Definition of deductive reasoning process
structure DeductiveReasoning :=
  (majorPremise : Prop)
  (minorPremise : Prop)
  (conclusion : Prop)

-- Theorem stating that the reasoning process for proving x^2 is even is deductive
theorem quadratic_even_deductive_reasoning :
  ∃ (reasoning : DeductiveReasoning),
    reasoning.majorPremise = (∀ f : ℝ → ℝ, IsEvenFunction f → ∀ x : ℝ, f (-x) = f x) ∧
    reasoning.minorPremise = (∃ f : ℝ → ℝ, QuadraticFunction f ∧ ∀ x : ℝ, f (-x) = f x) ∧
    reasoning.conclusion = (∃ f : ℝ → ℝ, QuadraticFunction f ∧ IsEvenFunction f) :=
  sorry


end NUMINAMATH_CALUDE_quadratic_even_deductive_reasoning_l759_75902


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l759_75937

/-- An isosceles triangle with congruent sides of 7 cm and perimeter of 25 cm has a base of 11 cm. -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 7
    let perimeter := 25
    (2 * congruent_side + base = perimeter) →
    base = 11

-- The proof is omitted
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 11 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l759_75937


namespace NUMINAMATH_CALUDE_mary_candy_count_l759_75949

theorem mary_candy_count (megan_candy : ℕ) (mary_multiplier : ℕ) (mary_additional : ℕ)
  (h1 : megan_candy = 5)
  (h2 : mary_multiplier = 3)
  (h3 : mary_additional = 10) :
  megan_candy * mary_multiplier + mary_additional = 25 :=
by sorry

end NUMINAMATH_CALUDE_mary_candy_count_l759_75949


namespace NUMINAMATH_CALUDE_negation_of_r_not_p_is_true_p_or_r_is_false_p_and_q_is_false_l759_75946

-- Define the propositions
def p : Prop := ∃ x₀ : ℝ, x₀ > -2 ∧ 6 + |x₀| = 5

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4/x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, |x| + |y| ≤ 1 → |y| / (|x| + 2) ≤ 1/2

-- State the theorems to be proved
theorem negation_of_r : (¬r) ↔ ∃ x y : ℝ, |x| + |y| > 1 ∧ |y| / (|x| + 2) > 1/2 := by sorry

theorem not_p_is_true : ¬p := by sorry

theorem p_or_r_is_false : ¬(p ∨ r) := by sorry

theorem p_and_q_is_false : ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_negation_of_r_not_p_is_true_p_or_r_is_false_p_and_q_is_false_l759_75946


namespace NUMINAMATH_CALUDE_haunted_castle_problem_l759_75926

/-- Represents a castle with windows -/
structure Castle where
  totalWindows : Nat
  forbiddenExitWindows : Nat

/-- Calculates the number of ways to enter and exit the castle -/
def waysToEnterAndExit (castle : Castle) : Nat :=
  castle.totalWindows * (castle.totalWindows - 1 - castle.forbiddenExitWindows)

/-- The haunted castle problem -/
theorem haunted_castle_problem :
  let castle : Castle := { totalWindows := 8, forbiddenExitWindows := 2 }
  waysToEnterAndExit castle = 40 := by
  sorry

end NUMINAMATH_CALUDE_haunted_castle_problem_l759_75926


namespace NUMINAMATH_CALUDE_bacteria_after_10_hours_l759_75901

/-- Represents the number of bacteria in the colony after a given number of hours -/
def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

/-- Theorem stating that after 10 hours, the bacteria count is 1024 -/
theorem bacteria_after_10_hours :
  bacteria_count 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_after_10_hours_l759_75901


namespace NUMINAMATH_CALUDE_limit_proof_l759_75998

def a_n (n : ℕ) : ℚ := (7 * n - 1) / (n + 1)

theorem limit_proof (ε : ℚ) (hε : ε > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |a_n n - 7| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l759_75998


namespace NUMINAMATH_CALUDE_initial_to_doubled_ratio_l759_75952

theorem initial_to_doubled_ratio (x : ℝ) (h : 3 * (2 * x + 13) = 93) : 
  x / (2 * x) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_initial_to_doubled_ratio_l759_75952


namespace NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2005_l759_75961

/-- The sequence a_n defined as 1 + 2^n + 3^n + 4^n + 5^n -/
def a (n : ℕ) : ℕ := 1 + 2^n + 3^n + 4^n + 5^n

/-- Theorem stating that there are no 5 consecutive terms in the sequence a_n all divisible by 2005 -/
theorem no_five_consecutive_divisible_by_2005 :
  ∀ m : ℕ, ¬(∀ k : Fin 5, 2005 ∣ a (m + k)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_consecutive_divisible_by_2005_l759_75961


namespace NUMINAMATH_CALUDE_perimeter_of_PQRS_l759_75974

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the perimeter function
def perimeter (quad : Quadrilateral) : ℝ := sorry

-- Define the properties of the quadrilateral
def is_right_angle_at_Q (quad : Quadrilateral) : Prop := sorry
def PR_perpendicular_to_RS (quad : Quadrilateral) : Prop := sorry
def PQ_length (quad : Quadrilateral) : ℝ := sorry
def QR_length (quad : Quadrilateral) : ℝ := sorry
def RS_length (quad : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem perimeter_of_PQRS (quad : Quadrilateral) :
  is_right_angle_at_Q quad →
  PR_perpendicular_to_RS quad →
  PQ_length quad = 24 →
  QR_length quad = 28 →
  RS_length quad = 16 →
  perimeter quad = 68 + Real.sqrt 1616 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_PQRS_l759_75974


namespace NUMINAMATH_CALUDE_multiply_add_distribute_compute_expression_l759_75962

theorem multiply_add_distribute (a b c : ℕ) : a * b + c * a = a * (b + c) := by sorry

theorem compute_expression : 45 * 27 + 73 * 45 = 4500 := by sorry

end NUMINAMATH_CALUDE_multiply_add_distribute_compute_expression_l759_75962


namespace NUMINAMATH_CALUDE_first_quadrant_trig_positivity_l759_75940

theorem first_quadrant_trig_positivity (α : Real) (h : 0 < α ∧ α < Real.pi / 2) :
  0 < Real.sin (2 * α) ∧ 0 < Real.tan (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_first_quadrant_trig_positivity_l759_75940


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l759_75995

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := (a + 1) * x + y - 2 = 0
def l₂ (a x y : ℝ) : Prop := a * x + (2 * a + 2) * y + 1 = 0

-- Define perpendicularity of two lines
def perpendicular (a : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ → l₂ a x₂ y₂ → 
    (a + 1) * a = -(2 * a + 2)

-- State the theorem
theorem perpendicular_lines_a_values (a : ℝ) :
  perpendicular a → a = -1 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l759_75995


namespace NUMINAMATH_CALUDE_amy_red_balloons_l759_75918

theorem amy_red_balloons (total green blue : ℕ) (h1 : total = 67) (h2 : green = 17) (h3 : blue = 21) : total - green - blue = 29 := by
  sorry

end NUMINAMATH_CALUDE_amy_red_balloons_l759_75918


namespace NUMINAMATH_CALUDE_sticker_distribution_theorem_l759_75991

def distribute_stickers (total_stickers : ℕ) (num_sheets : ℕ) : ℕ :=
  sorry

theorem sticker_distribution_theorem :
  distribute_stickers 10 5 = 126 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_theorem_l759_75991


namespace NUMINAMATH_CALUDE_georges_income_proof_l759_75955

/-- George's monthly income in dollars -/
def monthly_income : ℝ := 240

/-- The amount George spent on groceries in dollars -/
def grocery_expense : ℝ := 20

/-- The amount George has left in dollars -/
def amount_left : ℝ := 100

/-- Theorem stating that George's monthly income is correct given the conditions -/
theorem georges_income_proof :
  monthly_income / 2 - grocery_expense = amount_left := by sorry

end NUMINAMATH_CALUDE_georges_income_proof_l759_75955


namespace NUMINAMATH_CALUDE_brick_in_box_probability_l759_75928

/-- A set of six distinct numbers from 1 to 500 -/
def SixNumbers : Type := { s : Finset ℕ // s.card = 6 ∧ ∀ n ∈ s, 1 ≤ n ∧ n ≤ 500 }

/-- The three largest numbers from a set of six numbers -/
def largestThree (s : SixNumbers) : Finset ℕ :=
  sorry

/-- The three smallest numbers from a set of six numbers -/
def smallestThree (s : SixNumbers) : Finset ℕ :=
  sorry

/-- Whether a brick with given dimensions fits in a box with given dimensions -/
def fits (brick box : Finset ℕ) : Prop :=
  sorry

/-- The probability of a brick fitting in a box -/
def fitProbability : ℚ :=
  sorry

theorem brick_in_box_probability :
  fitProbability = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_brick_in_box_probability_l759_75928


namespace NUMINAMATH_CALUDE_indira_cricket_time_l759_75993

/-- Sean's daily cricket playing time in minutes -/
def sean_daily_time : ℕ := 50

/-- Number of days Sean played cricket -/
def sean_days : ℕ := 14

/-- Total time Sean and Indira played cricket together in minutes -/
def total_time : ℕ := 1512

/-- Calculates Indira's cricket playing time in minutes -/
def indira_time : ℕ := total_time - (sean_daily_time * sean_days)

theorem indira_cricket_time : indira_time = 812 := by
  sorry

end NUMINAMATH_CALUDE_indira_cricket_time_l759_75993


namespace NUMINAMATH_CALUDE_special_list_median_l759_75959

/-- Represents the list where each integer n from 1 to 300 appears exactly n times -/
def special_list : List ℕ := sorry

/-- The total number of elements in the special list -/
def total_elements : ℕ := (300 * 301) / 2

/-- The position of the median in the special list -/
def median_position : ℕ × ℕ := (total_elements / 2, total_elements / 2 + 1)

/-- The median value of the special list -/
def median_value : ℕ := 212

theorem special_list_median :
  median_value = 212 := by sorry

end NUMINAMATH_CALUDE_special_list_median_l759_75959


namespace NUMINAMATH_CALUDE_inequality_relation_l759_75976

theorem inequality_relation (a b : ℝ) (ha : a < 0) (hb : -1 < b ∧ b < 0) :
  a * b > a * b^2 ∧ a * b^2 > a := by sorry

end NUMINAMATH_CALUDE_inequality_relation_l759_75976


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l759_75951

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := -5*(x^5 - x^4 + x^3) + 8*(x^5 + 3) - 3*(2*x^5 + x^3 + 2)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (f : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = -3 := by sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l759_75951


namespace NUMINAMATH_CALUDE_simplified_expression_equals_two_thirds_l759_75906

theorem simplified_expression_equals_two_thirds :
  let x : ℚ := 5
  (1 - 1 / (x + 1)) / ((x^2 - x) / (x^2 - 2*x + 1)) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_equals_two_thirds_l759_75906


namespace NUMINAMATH_CALUDE_cost_of_two_pans_l759_75944

/-- The cost of 2 pans given Katerina's purchase information -/
theorem cost_of_two_pans :
  ∀ (pan_cost : ℕ),
  3 * 20 + 4 * pan_cost = 100 →
  2 * pan_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_of_two_pans_l759_75944


namespace NUMINAMATH_CALUDE_camping_hike_distance_l759_75935

/-- Hiking distances on a camping trip -/
theorem camping_hike_distance 
  (total_distance : ℝ)
  (car_to_stream : ℝ)
  (stream_to_meadow : ℝ)
  (h_total : total_distance = 0.7)
  (h_car_stream : car_to_stream = 0.2)
  (h_stream_meadow : stream_to_meadow = 0.4) :
  total_distance - (car_to_stream + stream_to_meadow) = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l759_75935


namespace NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l759_75941

theorem twelve_point_zero_six_million_scientific_notation :
  (12.06 : ℝ) * 1000000 = 1.206 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_twelve_point_zero_six_million_scientific_notation_l759_75941


namespace NUMINAMATH_CALUDE_jony_start_block_l759_75987

/-- Represents Jony's walk along Sunrise Boulevard -/
structure JonyWalk where
  walkTime : ℕ            -- Walking time in minutes
  speed : ℕ               -- Speed in meters per minute
  blockLength : ℕ         -- Length of each block in meters
  turnAroundBlock : ℕ     -- Block number where Jony turns around
  stopBlock : ℕ           -- Block number where Jony stops

/-- Calculates the starting block number for Jony's walk -/
def calculateStartBlock (walk : JonyWalk) : ℕ :=
  sorry

/-- Theorem stating that given the conditions of Jony's walk, his starting block is 10 -/
theorem jony_start_block :
  let walk : JonyWalk := {
    walkTime := 40,
    speed := 100,
    blockLength := 40,
    turnAroundBlock := 90,
    stopBlock := 70
  }
  calculateStartBlock walk = 10 := by
  sorry

end NUMINAMATH_CALUDE_jony_start_block_l759_75987


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l759_75984

-- Define the sets A and B
def A : Set ℝ := { y | ∃ x, y = x^2 - 4*x + 3 }
def B : Set ℝ := { y | ∃ x, y = -x^2 - 2*x }

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = { y | -1 ≤ y ∧ y ≤ 1 } := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_l759_75984


namespace NUMINAMATH_CALUDE_brick_height_proof_l759_75932

/-- Proves that the height of each brick is 6 cm given the wall and brick dimensions --/
theorem brick_height_proof (wall_length wall_width wall_height : ℝ)
                           (brick_length brick_width : ℝ)
                           (num_bricks : ℕ) :
  wall_length = 700 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 5600 →
  ∃ (h : ℝ), h = 6 ∧ 
    wall_length * wall_width * wall_height = 
    num_bricks * brick_length * brick_width * h :=
by
  sorry

end NUMINAMATH_CALUDE_brick_height_proof_l759_75932


namespace NUMINAMATH_CALUDE_pythagorean_number_existence_l759_75992

theorem pythagorean_number_existence (n : ℕ) (hn : n > 12) :
  ∃ (a b c P : ℕ), a > b ∧ b > 0 ∧ c > 0 ∧
  P = a * b * (a^2 - b^2) * c^2 ∧
  n < P ∧ P < 2 * n :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_number_existence_l759_75992


namespace NUMINAMATH_CALUDE_peters_cucumbers_l759_75950

/-- The problem of Peter's grocery shopping -/
theorem peters_cucumbers 
  (initial_amount : ℕ)
  (potato_kilos potato_price : ℕ)
  (tomato_kilos tomato_price : ℕ)
  (banana_kilos banana_price : ℕ)
  (cucumber_price : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_kilos = 6)
  (h3 : potato_price = 2)
  (h4 : tomato_kilos = 9)
  (h5 : tomato_price = 3)
  (h6 : banana_kilos = 3)
  (h7 : banana_price = 5)
  (h8 : cucumber_price = 4)
  (h9 : remaining_amount = 426)
  : ∃ (cucumber_kilos : ℕ), 
    initial_amount - 
    (potato_kilos * potato_price + 
     tomato_kilos * tomato_price + 
     banana_kilos * banana_price + 
     cucumber_kilos * cucumber_price) = remaining_amount ∧ 
    cucumber_kilos = 5 := by
  sorry

end NUMINAMATH_CALUDE_peters_cucumbers_l759_75950


namespace NUMINAMATH_CALUDE_parallel_condition_l759_75919

/-- Two lines in R² are parallel if and only if their slopes are equal -/
def are_parallel (a b c d e f : ℝ) : Prop :=
  (a * f = b * d) ∧ (a * e ≠ b * c ∨ c * f ≠ d * e)

/-- The condition for two lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (∀ x y : ℝ, are_parallel a 1 (-1) 1 a 1) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_condition_l759_75919


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l759_75903

theorem sufficient_not_necessary (a b : ℝ) : 
  (a > b ∧ b > 0) → (1 / a < 1 / b) ∧ 
  ¬(∀ a b : ℝ, (1 / a < 1 / b) → (a > b ∧ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l759_75903


namespace NUMINAMATH_CALUDE_sum_base5_equals_1333_l759_75966

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from decimal to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The sum of 213₅, 324₅, and 141₅ is equal to 1333₅ in base 5 --/
theorem sum_base5_equals_1333 :
  decimalToBase5 (base5ToDecimal [3, 1, 2] + base5ToDecimal [4, 2, 3] + base5ToDecimal [1, 4, 1]) = [3, 3, 3, 1] :=
sorry

end NUMINAMATH_CALUDE_sum_base5_equals_1333_l759_75966


namespace NUMINAMATH_CALUDE_f_prime_minus_g_prime_at_one_l759_75909

-- Define f and g as differentiable functions on ℝ
variable (f g : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : Differentiable ℝ g)
variable (h3 : ∀ x, f x = x * g x + x^2 - 1)
variable (h4 : f 1 = 1)

-- State the theorem
theorem f_prime_minus_g_prime_at_one :
  deriv f 1 - deriv g 1 = 3 := by sorry

end NUMINAMATH_CALUDE_f_prime_minus_g_prime_at_one_l759_75909


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_discriminants_l759_75965

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℝ :=
  p.b ^ 2 - 4 * p.a * p.c

/-- A quadratic polynomial has zero discriminant -/
def hasZeroDiscriminant (p : QuadraticPolynomial) : Prop :=
  discriminant p = 0

/-- The sum of two quadratic polynomials -/
def sumQuadratic (p q : QuadraticPolynomial) : QuadraticPolynomial where
  a := p.a + q.a
  b := p.b + q.b
  c := p.c + q.c

theorem quadratic_sum_zero_discriminants :
  ∀ (p : QuadraticPolynomial),
  ∃ (q r : QuadraticPolynomial),
  hasZeroDiscriminant q ∧
  hasZeroDiscriminant r ∧
  p = sumQuadratic q r :=
sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_discriminants_l759_75965


namespace NUMINAMATH_CALUDE_total_lollipops_eq_twelve_l759_75982

/-- The number of lollipops Sushi's father brought -/
def total_lollipops : ℕ := sorry

/-- The number of lollipops eaten by the children -/
def eaten_lollipops : ℕ := 5

/-- The number of lollipops left -/
def remaining_lollipops : ℕ := 7

/-- Theorem stating that the total number of lollipops equals 12 -/
theorem total_lollipops_eq_twelve :
  total_lollipops = eaten_lollipops + remaining_lollipops ∧
  total_lollipops = 12 := by sorry

end NUMINAMATH_CALUDE_total_lollipops_eq_twelve_l759_75982


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l759_75925

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l759_75925


namespace NUMINAMATH_CALUDE_unique_solution_system_l759_75994

theorem unique_solution_system (x y z : ℝ) :
  x + y + z = 2008 →
  x^2 + y^2 + z^2 = 6024^2 →
  1/x + 1/y + 1/z = 1/2008 →
  (x = 2008 ∧ y = 4016 ∧ z = -4016) ∨
  (x = 2008 ∧ y = -4016 ∧ z = 4016) ∨
  (x = 4016 ∧ y = 2008 ∧ z = -4016) ∨
  (x = 4016 ∧ y = -4016 ∧ z = 2008) ∨
  (x = -4016 ∧ y = 2008 ∧ z = 4016) ∨
  (x = -4016 ∧ y = 4016 ∧ z = 2008) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l759_75994


namespace NUMINAMATH_CALUDE_greatest_three_digit_special_number_l759_75933

/-- A number is a three-digit number if it's between 100 and 999, inclusive -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number is one more than a multiple of 9 if it leaves a remainder of 1 when divided by 9 -/
def is_one_more_than_multiple_of_9 (n : ℕ) : Prop := n % 9 = 1

/-- A number is three more than a multiple of 5 if it leaves a remainder of 3 when divided by 5 -/
def is_three_more_than_multiple_of_5 (n : ℕ) : Prop := n % 5 = 3

theorem greatest_three_digit_special_number : 
  is_three_digit 973 ∧ 
  is_one_more_than_multiple_of_9 973 ∧ 
  is_three_more_than_multiple_of_5 973 ∧ 
  ∀ m : ℕ, (is_three_digit m ∧ 
            is_one_more_than_multiple_of_9 m ∧ 
            is_three_more_than_multiple_of_5 m) → 
           m ≤ 973 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_special_number_l759_75933


namespace NUMINAMATH_CALUDE_math_problem_proof_l759_75960

theorem math_problem_proof :
  let expression1 := -3^2 + 2^2023 * (-1/2)^2022 + (-2024)^0
  let x : ℚ := -1/2
  let y : ℚ := 1
  let expression2 := ((x + 2*y)^2 - (2*x + y)*(2*x - y) - 5*(x^2 + y^2)) / (2*x)
  expression1 = -6 ∧ expression2 = 4 := by sorry

end NUMINAMATH_CALUDE_math_problem_proof_l759_75960


namespace NUMINAMATH_CALUDE_f_properties_l759_75964

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x) + Real.log (1 + x) + x^4 - 2*x^2

theorem f_properties :
  (∀ x, f x ≠ 0 → -1 < x ∧ x < 1) ∧
  (∀ x, f (-x) = f x) ∧
  (∀ y, (∃ x, f x = y) → y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l759_75964


namespace NUMINAMATH_CALUDE_awards_distribution_l759_75957

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution :
  distribute_awards 6 4 = 1560 :=
sorry

end NUMINAMATH_CALUDE_awards_distribution_l759_75957


namespace NUMINAMATH_CALUDE_closest_to_140_l759_75905

def options : List ℝ := [120, 140, 160, 180, 200]

def expression : ℝ := 3.52 * 7.861 * (6.28 - 1.283)

theorem closest_to_140 : 
  ∀ x ∈ options, |expression - 140| ≤ |expression - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_to_140_l759_75905


namespace NUMINAMATH_CALUDE_smith_family_puzzle_l759_75954

def is_valid_license_plate (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧
  ∃ (a b c : ℕ) (d : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (n.digits 10).count a ≥ 1 ∧
    (n.digits 10).count b ≥ 1 ∧
    (n.digits 10).count c ≥ 1 ∧
    (n.digits 10).count d = 3 ∧
    (n.digits 10).sum = 2 * (n % 100)

theorem smith_family_puzzle :
  ∀ (license_plate : ℕ) (children_ages : List ℕ),
    is_valid_license_plate license_plate →
    children_ages.length = 9 →
    children_ages.maximum = some 10 →
    (∀ age ∈ children_ages, age < 10) →
    (∀ age ∈ children_ages, license_plate % age = 0) →
    4 ∉ children_ages :=
by sorry

end NUMINAMATH_CALUDE_smith_family_puzzle_l759_75954


namespace NUMINAMATH_CALUDE_yellow_paint_cans_l759_75947

theorem yellow_paint_cans (yellow green : ℕ) (total : ℕ) : 
  yellow + green = total → 
  yellow = 4 * green / 3 → 
  total = 42 → 
  yellow = 24 := by sorry

end NUMINAMATH_CALUDE_yellow_paint_cans_l759_75947


namespace NUMINAMATH_CALUDE_fayes_age_l759_75996

/-- Given the ages of Diana, Eduardo, Chad, Faye, and Greg, prove Faye's age --/
theorem fayes_age 
  (D E C F G : ℕ) -- Ages of Diana, Eduardo, Chad, Faye, and Greg
  (h1 : D = E - 2)
  (h2 : C = E + 3)
  (h3 : F = C - 1)
  (h4 : D = 16)
  (h5 : G = D - 5) :
  F = 20 := by
  sorry

end NUMINAMATH_CALUDE_fayes_age_l759_75996


namespace NUMINAMATH_CALUDE_random_walk_properties_l759_75967

/-- A random walk on a line with forward probability 3/4 and backward probability 1/4 -/
structure RandomWalk where
  forwardProb : ℝ
  backwardProb : ℝ
  forwardProbEq : forwardProb = 3/4
  backwardProbEq : backwardProb = 1/4
  probSum : forwardProb + backwardProb = 1

/-- The probability of returning to the starting point after n steps -/
def returnProbability (rw : RandomWalk) (n : ℕ) : ℝ :=
  sorry

/-- The probability distribution of the distance from the starting point after n steps -/
def distanceProbability (rw : RandomWalk) (n : ℕ) (d : ℕ) : ℝ :=
  sorry

/-- The expected value of the distance from the starting point after n steps -/
def expectedDistance (rw : RandomWalk) (n : ℕ) : ℝ :=
  sorry

theorem random_walk_properties (rw : RandomWalk) :
  returnProbability rw 4 = 27/128 ∧
  distanceProbability rw 5 1 = 45/128 ∧
  distanceProbability rw 5 3 = 105/256 ∧
  distanceProbability rw 5 5 = 61/256 ∧
  expectedDistance rw 5 = 355/128 := by
  sorry

end NUMINAMATH_CALUDE_random_walk_properties_l759_75967


namespace NUMINAMATH_CALUDE_f_range_l759_75970

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 9 * x^2)

theorem f_range :
  Set.range f = Set.Ioo 0 3 ∪ {3} :=
sorry

end NUMINAMATH_CALUDE_f_range_l759_75970


namespace NUMINAMATH_CALUDE_solve_for_A_l759_75953

theorem solve_for_A : ∃ A : ℝ, 4 * A + 5 = 33 ∧ A = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l759_75953


namespace NUMINAMATH_CALUDE_tessellation_with_squares_and_triangles_l759_75997

theorem tessellation_with_squares_and_triangles :
  ∀ m n : ℕ,
  (60 * m + 90 * n = 360) →
  (m = 3 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_tessellation_with_squares_and_triangles_l759_75997


namespace NUMINAMATH_CALUDE_dolls_made_l759_75978

def accessories_per_doll : ℕ := 2 + 3 + 1 + 5

def time_per_doll_and_accessories : ℕ := 45 + accessories_per_doll * 10

def total_operation_time : ℕ := 1860000

theorem dolls_made : 
  total_operation_time / time_per_doll_and_accessories = 12000 := by sorry

end NUMINAMATH_CALUDE_dolls_made_l759_75978


namespace NUMINAMATH_CALUDE_max_product_constraint_l759_75912

theorem max_product_constraint (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 3) :
  m * n ≤ 9 / 4 ∧ ∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 3 ∧ m₀ * n₀ = 9 / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_product_constraint_l759_75912


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l759_75921

theorem twenty_five_percent_less_than_80 : ∃ x : ℝ, x + (1/4 * x) = 0.75 * 80 ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l759_75921


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l759_75907

theorem sum_of_solutions_is_zero :
  let f : ℝ → ℝ := λ x => Real.sqrt (9 - x^2 / 4)
  (∀ x, f x = 3 → x = 0) ∧ (∃ x, f x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l759_75907


namespace NUMINAMATH_CALUDE_line_equation_l759_75942

/-- A line passing through the point (2, 3) with opposite intercepts on the coordinate axes -/
structure LineWithOppositeIntercepts where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 3)
  point_condition : 3 = m * 2 + b
  -- The line has opposite intercepts on the axes
  opposite_intercepts : ∃ (k : ℝ), k ≠ 0 ∧ (b = k ∨ b = -k) ∧ (b / m = -k ∨ b / m = k)

/-- The equation of the line is either x - y + 1 = 0 or 3x - 2y = 0 -/
theorem line_equation (l : LineWithOppositeIntercepts) :
  (l.m = 1 ∧ l.b = -1) ∨ (l.m = 3/2 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l759_75942


namespace NUMINAMATH_CALUDE_abcd_product_l759_75999

theorem abcd_product (a b c d : ℝ) 
  (ha : a = Real.sqrt (4 + Real.sqrt (5 - a)))
  (hb : b = Real.sqrt (4 + Real.sqrt (5 + b)))
  (hc : c = Real.sqrt (4 - Real.sqrt (5 - c)))
  (hd : d = Real.sqrt (4 - Real.sqrt (5 + d))) :
  a * b * c * d = 11 := by
sorry

end NUMINAMATH_CALUDE_abcd_product_l759_75999


namespace NUMINAMATH_CALUDE_bike_journey_l759_75989

theorem bike_journey (v d : ℝ) 
  (h1 : d / (v - 4) - d / v = 1.2)
  (h2 : d / v - d / (v + 4) = 2) :
  d = 160 := by
  sorry

end NUMINAMATH_CALUDE_bike_journey_l759_75989


namespace NUMINAMATH_CALUDE_inverse_proposition_l759_75963

theorem inverse_proposition : 
  (∀ a b : ℝ, a > b → b - a < 0) ↔ (∀ a b : ℝ, b - a < 0 → a > b) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_l759_75963


namespace NUMINAMATH_CALUDE_triangle_area_implies_cd_one_l759_75914

theorem triangle_area_implies_cd_one (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_line : ∀ x y, 2*c*x + 3*d*y = 12 → x ≥ 0 ∧ y ≥ 0)
  (h_area : (1/2) * (6/c) * (4/d) = 12) : c * d = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_implies_cd_one_l759_75914


namespace NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l759_75969

-- Define the repeating decimal 0.333...
def repeating_third : ℚ := 1/3

-- State the theorem
theorem eight_divided_by_repeating_third (h : repeating_third = 1/3) : 
  8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_eight_divided_by_repeating_third_l759_75969


namespace NUMINAMATH_CALUDE_base_8_to_base_7_l759_75981

def base_8_to_decimal (n : ℕ) : ℕ := n

def decimal_to_base_7 (n : ℕ) : ℕ := n

theorem base_8_to_base_7 :
  decimal_to_base_7 (base_8_to_decimal 536) = 1010 :=
sorry

end NUMINAMATH_CALUDE_base_8_to_base_7_l759_75981


namespace NUMINAMATH_CALUDE_ferris_wheel_theorem_l759_75939

/-- The number of people who can ride a Ferris wheel at the same time -/
def ferris_wheel_capacity (seats : ℕ) (people_per_seat : ℕ) : ℕ :=
  seats * people_per_seat

/-- Theorem: The capacity of a Ferris wheel with 2 seats and 2 people per seat is 4 -/
theorem ferris_wheel_theorem : ferris_wheel_capacity 2 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_theorem_l759_75939


namespace NUMINAMATH_CALUDE_truck_rental_miles_l759_75956

theorem truck_rental_miles (rental_fee charge_per_mile total_paid : ℚ) : 
  rental_fee = 20.99 →
  charge_per_mile = 0.25 →
  total_paid = 95.74 →
  (total_paid - rental_fee) / charge_per_mile = 299 := by
  sorry

end NUMINAMATH_CALUDE_truck_rental_miles_l759_75956


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l759_75990

theorem smallest_solution_absolute_value_equation :
  let f := fun x : ℝ => x * |x| - 3 * x + 2
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x = 0 → x₀ ≤ x ∧ x₀ = (-3 - Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l759_75990


namespace NUMINAMATH_CALUDE_inequality_proof_l759_75985

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1 / (1/a + 1/b) + 1 / (1/c + 1/d) ≤ 1 / (1/(a+c) + 1/(b+d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l759_75985


namespace NUMINAMATH_CALUDE_nine_oclock_right_angle_l759_75986

/-- The angle between clock hands at a given hour -/
def clock_angle (hour : ℕ) : ℝ :=
  sorry

/-- A right angle is 90 degrees -/
def is_right_angle (angle : ℝ) : Prop :=
  angle = 90

theorem nine_oclock_right_angle : is_right_angle (clock_angle 9) := by
  sorry

end NUMINAMATH_CALUDE_nine_oclock_right_angle_l759_75986


namespace NUMINAMATH_CALUDE_exists_2008_acquaintances_l759_75973

/-- Represents a gathering of people -/
structure Gathering where
  people : Finset Nat
  acquaintances : Nat → Finset Nat
  no_common_acquaintances : ∀ x y, x ∈ people → y ∈ people →
    (acquaintances x).card = (acquaintances y).card →
    (acquaintances x ∩ acquaintances y).card ≤ 1

/-- Main theorem: If there's someone with at least 2008 acquaintances,
    then there's someone with exactly 2008 acquaintances -/
theorem exists_2008_acquaintances (g : Gathering) :
  (∃ x ∈ g.people, (g.acquaintances x).card ≥ 2008) →
  (∃ y ∈ g.people, (g.acquaintances y).card = 2008) := by
  sorry

end NUMINAMATH_CALUDE_exists_2008_acquaintances_l759_75973


namespace NUMINAMATH_CALUDE_sum_y_coordinates_on_y_axis_l759_75904

-- Define the circle
def circle_center : ℝ × ℝ := (-4, 3)
def circle_radius : ℝ := 5

-- Define a function to check if a point is on the circle
def on_circle (point : ℝ × ℝ) : Prop :=
  (point.1 - circle_center.1)^2 + (point.2 - circle_center.2)^2 = circle_radius^2

-- Define a function to check if a point is on the y-axis
def on_y_axis (point : ℝ × ℝ) : Prop :=
  point.1 = 0

-- Theorem statement
theorem sum_y_coordinates_on_y_axis :
  ∃ (p1 p2 : ℝ × ℝ),
    on_circle p1 ∧ on_circle p2 ∧
    on_y_axis p1 ∧ on_y_axis p2 ∧
    p1 ≠ p2 ∧
    p1.2 + p2.2 = 6 :=
  sorry

end NUMINAMATH_CALUDE_sum_y_coordinates_on_y_axis_l759_75904


namespace NUMINAMATH_CALUDE_puppies_per_dog_l759_75913

/-- Given information about Chuck's dog breeding operation -/
structure DogBreeding where
  num_pregnant_dogs : ℕ
  shots_per_puppy : ℕ
  cost_per_shot : ℕ
  total_shot_cost : ℕ

/-- Theorem stating the number of puppies per pregnant dog -/
theorem puppies_per_dog (d : DogBreeding)
  (h1 : d.num_pregnant_dogs = 3)
  (h2 : d.shots_per_puppy = 2)
  (h3 : d.cost_per_shot = 5)
  (h4 : d.total_shot_cost = 120) :
  d.total_shot_cost / (d.num_pregnant_dogs * d.shots_per_puppy * d.cost_per_shot) = 4 := by
  sorry

end NUMINAMATH_CALUDE_puppies_per_dog_l759_75913


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l759_75975

theorem triangle_angle_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ A < π ∧ B > 0 ∧ B < π ∧ C > 0 ∧ C < π →
  A + B + C = π →
  b = 2 * a * Real.sin B →
  A = π / 6 ∨ A = 5 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l759_75975


namespace NUMINAMATH_CALUDE_andromeda_distance_scientific_notation_l759_75958

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The distance of the Andromeda galaxy from the Milky Way in light-years -/
def andromeda_distance : ℝ := 2500000

theorem andromeda_distance_scientific_notation :
  to_scientific_notation andromeda_distance = ScientificNotation.mk 2.5 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_andromeda_distance_scientific_notation_l759_75958


namespace NUMINAMATH_CALUDE_cube_side_length_l759_75972

theorem cube_side_length (volume : ℝ) (x : ℝ) : volume = 8 → x^3 = volume → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l759_75972


namespace NUMINAMATH_CALUDE_infinite_solutions_l759_75916

theorem infinite_solutions (a : ℝ) : 
  (a = 5) → (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 9) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l759_75916


namespace NUMINAMATH_CALUDE_range_of_a_l759_75930

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 4*x + a = 0

-- Define the set of valid a values
def valid_a : Set ℝ := {a | a < Real.exp 1 ∨ a > 4}

-- Theorem statement
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) → a ∈ valid_a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l759_75930


namespace NUMINAMATH_CALUDE_triangle_area_l759_75929

/-- Given a triangle ABC with angle A = 60°, side b = 4, and side a = 2√3,
    prove that its area is 2√3. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 →  -- 60° in radians
  b = 4 → 
  a = 2 * Real.sqrt 3 → 
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l759_75929


namespace NUMINAMATH_CALUDE_even_function_derivative_at_zero_l759_75988

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_derivative_at_zero
  (f : ℝ → ℝ)
  (hf : EvenFunction f)
  (hf' : Differentiable ℝ f) :
  deriv f 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_even_function_derivative_at_zero_l759_75988


namespace NUMINAMATH_CALUDE_school_purchase_cost_l759_75915

theorem school_purchase_cost : 
  let projector_count : ℕ := 8
  let computer_count : ℕ := 32
  let projector_cost : ℕ := 7500
  let computer_cost : ℕ := 3600
  (projector_count * projector_cost + computer_count * computer_cost : ℕ) = 175200 := by
  sorry

end NUMINAMATH_CALUDE_school_purchase_cost_l759_75915


namespace NUMINAMATH_CALUDE_smallest_multiple_ten_satisfies_ten_is_smallest_l759_75977

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 500 ∣ (450 * x) → x ≥ 10 := by
  sorry

theorem ten_satisfies : 500 ∣ (450 * 10) := by
  sorry

theorem ten_is_smallest : ∀ y : ℕ, y > 0 ∧ 500 ∣ (450 * y) → y ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_ten_satisfies_ten_is_smallest_l759_75977


namespace NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l759_75920

theorem sum_of_distinct_prime_factors : 
  (let n := 7^6 - 2 * 7^4
   Finset.sum (Finset.filter (fun p => Nat.Prime p ∧ n % p = 0) (Finset.range (n + 1))) id) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distinct_prime_factors_l759_75920


namespace NUMINAMATH_CALUDE_projection_sum_bound_l759_75983

/-- A segment in the plane represented by its length and angle -/
structure Segment where
  length : ℝ
  angle : ℝ

/-- The theorem statement -/
theorem projection_sum_bound (segments : List Segment) 
  (total_length : (segments.map (λ s => s.length)).sum = 1) :
  ∃ θ : ℝ, (segments.map (λ s => s.length * |Real.cos (θ - s.angle)|)).sum < 2 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_projection_sum_bound_l759_75983


namespace NUMINAMATH_CALUDE_largest_value_proof_l759_75948

theorem largest_value_proof (a b c : ℚ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : b < 1) (h5 : a > b) : 
  (max (-1/(a*b)) (max (1/b) (max |a*c| (max (1/b^2) (1/a^2))))) = 1/b^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_value_proof_l759_75948


namespace NUMINAMATH_CALUDE_recycling_project_points_l759_75900

/-- Calculates points earned for white paper -/
def whitePoints (pounds : ℕ) : ℕ := (pounds / 6) * 2

/-- Calculates points earned for colored paper -/
def colorPoints (pounds : ℕ) : ℕ := (pounds / 8) * 3

/-- Represents a person's recycling contribution -/
structure Recycler where
  whitePaper : ℕ
  coloredPaper : ℕ

/-- Calculates total points for a recycler -/
def totalPoints (r : Recycler) : ℕ :=
  whitePoints r.whitePaper + colorPoints r.coloredPaper

theorem recycling_project_points : 
  let paige : Recycler := { whitePaper := 12, coloredPaper := 18 }
  let alex : Recycler := { whitePaper := 26, coloredPaper := 10 }
  let jordan : Recycler := { whitePaper := 30, coloredPaper := 0 }
  totalPoints paige + totalPoints alex + totalPoints jordan = 31 := by
  sorry

end NUMINAMATH_CALUDE_recycling_project_points_l759_75900


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l759_75924

theorem earth_inhabitable_fraction :
  let water_fraction : ℚ := 3/5
  let inhabitable_land_fraction : ℚ := 2/3
  let total_inhabitable_fraction : ℚ := (1 - water_fraction) * inhabitable_land_fraction
  total_inhabitable_fraction = 4/15 := by sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l759_75924


namespace NUMINAMATH_CALUDE_max_product_value_l759_75923

/-- Given two real-valued functions f and g with specified ranges and a condition on their maxima,
    this theorem states that the maximum value of their product is 35. -/
theorem max_product_value (f g : ℝ → ℝ) (hf : ∀ x, 1 ≤ f x ∧ f x ≤ 7) 
    (hg : ∀ x, -3 ≤ g x ∧ g x ≤ 5) 
    (hmax : ∃ x, f x = 7 ∧ g x = 5) : 
    (∃ b, ∀ x, f x * g x ≤ b) ∧ (∀ b, (∀ x, f x * g x ≤ b) → b ≥ 35) :=
sorry

end NUMINAMATH_CALUDE_max_product_value_l759_75923


namespace NUMINAMATH_CALUDE_range_of_r_l759_75931

-- Define the function r(x)
def r (x : ℝ) : ℝ := x^6 + 6*x^3 + 9

-- State the theorem
theorem range_of_r :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ r x = y) ↔ y ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_r_l759_75931


namespace NUMINAMATH_CALUDE_opposite_signs_sum_zero_l759_75927

theorem opposite_signs_sum_zero (a b : ℝ) : a * b < 0 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_signs_sum_zero_l759_75927


namespace NUMINAMATH_CALUDE_power_function_problem_l759_75934

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ α : ℝ, ∀ x > 0, f x = x^α

-- State the theorem
theorem power_function_problem (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2 / 2) : 
  f 9 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_problem_l759_75934


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l759_75979

theorem necessary_but_not_sufficient (a b : ℝ) :
  (((1 / b) < (1 / a) ∧ (1 / a) < 0) → a < b) ∧
  (∃ a b, a < b ∧ ¬((1 / b) < (1 / a) ∧ (1 / a) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l759_75979


namespace NUMINAMATH_CALUDE_quadratic_integer_root_l759_75938

theorem quadratic_integer_root (b : ℤ) : 
  (∃ x : ℤ, x^2 + 4*x + b = 0) ↔ (b = -12 ∨ b = -5 ∨ b = 3 ∨ b = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_l759_75938
