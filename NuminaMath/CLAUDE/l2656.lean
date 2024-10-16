import Mathlib

namespace NUMINAMATH_CALUDE_cubic_expression_equals_three_l2656_265634

theorem cubic_expression_equals_three (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) 
  (h4 : p + 2*q + 3*r = 0) : (p^3 + 2*q^3 + 3*r^3) / (p*q*r) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_equals_three_l2656_265634


namespace NUMINAMATH_CALUDE_counterexample_exponential_inequality_l2656_265653

theorem counterexample_exponential_inequality :
  ∃ (a m n : ℝ), a > 0 ∧ a ≠ 1 ∧ a^m < a^n ∧ m ≥ n := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exponential_inequality_l2656_265653


namespace NUMINAMATH_CALUDE_smallest_equal_distribution_l2656_265670

def apple_per_box : ℕ := 18
def grapes_per_container : ℕ := 9
def orange_per_container : ℕ := 12
def cherries_per_bag : ℕ := 6

theorem smallest_equal_distribution (n : ℕ) :
  (n % apple_per_box = 0) ∧
  (n % grapes_per_container = 0) ∧
  (n % orange_per_container = 0) ∧
  (n % cherries_per_bag = 0) ∧
  (∀ m : ℕ, m < n →
    ¬((m % apple_per_box = 0) ∧
      (m % grapes_per_container = 0) ∧
      (m % orange_per_container = 0) ∧
      (m % cherries_per_bag = 0))) →
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_smallest_equal_distribution_l2656_265670


namespace NUMINAMATH_CALUDE_angle4_measure_l2656_265656

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)

-- State the theorem
theorem angle4_measure
  (h1 : angle1 = 82)
  (h2 : angle2 = 34)
  (h3 : angle3 = 19)
  (h4 : angle5 = angle6 + 10)
  (h5 : angle1 + angle2 + angle3 + angle5 + angle6 = 180)
  (h6 : angle4 + angle5 + angle6 = 180) :
  angle4 = 135 := by
sorry

end NUMINAMATH_CALUDE_angle4_measure_l2656_265656


namespace NUMINAMATH_CALUDE_exists_valid_31_min_students_smallest_total_l2656_265675

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- The ratios between grades are correct --/
def valid_ratios (gc : GradeCount) : Prop :=
  4 * gc.ninth = 3 * gc.eleventh ∧ 6 * gc.tenth = 5 * gc.eleventh

/-- The total number of students --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- There exists a valid configuration with 31 students --/
theorem exists_valid_31 : ∃ gc : GradeCount, valid_ratios gc ∧ total_students gc = 31 := by
  sorry

/-- Any valid configuration has at least 31 students --/
theorem min_students (gc : GradeCount) (h : valid_ratios gc) : total_students gc ≥ 31 := by
  sorry

/-- The smallest possible number of students is 31 --/
theorem smallest_total : (∃ gc : GradeCount, valid_ratios gc ∧ total_students gc = 31) ∧
  (∀ gc : GradeCount, valid_ratios gc → total_students gc ≥ 31) := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_31_min_students_smallest_total_l2656_265675


namespace NUMINAMATH_CALUDE_root_difference_of_cubic_l2656_265631

theorem root_difference_of_cubic (x₁ x₂ x₃ : ℝ) :
  (81 * x₁^3 - 162 * x₁^2 + 108 * x₁ - 18 = 0) →
  (81 * x₂^3 - 162 * x₂^2 + 108 * x₂ - 18 = 0) →
  (81 * x₃^3 - 162 * x₃^2 + 108 * x₃ - 18 = 0) →
  (x₂ - x₁ = x₃ - x₂) →  -- arithmetic progression condition
  (max x₁ (max x₂ x₃) - min x₁ (min x₂ x₃) = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_root_difference_of_cubic_l2656_265631


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2656_265624

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), 
    (x > 0) ∧ 
    (x < 1) ∧ 
    (initial_price * (1 - x)^2 = final_price) ∧
    (x = 1 - (4/5)) := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2656_265624


namespace NUMINAMATH_CALUDE_language_spoken_by_three_scientists_l2656_265622

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  languages : Finset String
  languages_bound : languages.card ≤ 3

/-- The set of all scientists at the conference -/
def Scientists : Finset Scientist :=
  sorry

/-- The number of scientists at the conference -/
axiom scientists_count : Scientists.card = 9

/-- No scientist speaks more than 3 languages -/
axiom max_languages (s : Scientist) : s.languages.card ≤ 3

/-- Among any three scientists, there are two who speak a common language -/
axiom common_language_exists (s1 s2 s3 : Scientist) :
  s1 ∈ Scientists → s2 ∈ Scientists → s3 ∈ Scientists →
  ∃ (l : String), (l ∈ s1.languages ∧ l ∈ s2.languages) ∨
                  (l ∈ s1.languages ∧ l ∈ s3.languages) ∨
                  (l ∈ s2.languages ∧ l ∈ s3.languages)

/-- There exists a language spoken by at least three scientists -/
theorem language_spoken_by_three_scientists :
  ∃ (l : String), (Scientists.filter (fun s => l ∈ s.languages)).card ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_language_spoken_by_three_scientists_l2656_265622


namespace NUMINAMATH_CALUDE_divided_value_problem_l2656_265672

theorem divided_value_problem (x : ℝ) : (6.5 / x) * 12 = 13 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_divided_value_problem_l2656_265672


namespace NUMINAMATH_CALUDE_range_of_a_l2656_265654

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, a - x - |Real.log x| ≤ 0) ↔ 0 < a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2656_265654


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2656_265629

/-- Given a configuration of four congruent rectangles arranged around an inner square,
    where the area of the outer square is 9 times the area of the inner square,
    the ratio of the longer side to the shorter side of each rectangle is 2. -/
theorem rectangle_ratio (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0)
  (h4 : s + 2*y = 3*s) (h5 : x + s = 3*s) : x / y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2656_265629


namespace NUMINAMATH_CALUDE_pencils_left_l2656_265650

/-- Calculates the number of pencils Steve has left after giving some to Matt and Lauren -/
theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (lauren_pencils : ℕ) (matt_extra : ℕ) : 
  boxes * pencils_per_box - (lauren_pencils + (lauren_pencils + matt_extra)) = 9 :=
by
  sorry

#check pencils_left 2 12 6 3

end NUMINAMATH_CALUDE_pencils_left_l2656_265650


namespace NUMINAMATH_CALUDE_base_85_congruence_l2656_265698

/-- Converts a base 85 number to base 10 -/
def base85ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 85^i) 0

/-- The base 85 representation of 746392847₈₅ -/
def num : List Nat := [7, 4, 6, 3, 9, 2, 8, 4, 7]

theorem base_85_congruence : 
  ∃ (b : ℕ), 0 ≤ b ∧ b ≤ 20 ∧ (base85ToBase10 num - b) % 17 = 0 → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_base_85_congruence_l2656_265698


namespace NUMINAMATH_CALUDE_range_of_m_l2656_265691

-- Define the propositions
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x m : ℝ) : Prop := x > m

-- Define the theorem
theorem range_of_m :
  (∀ x m : ℝ, (¬(q x m) → ¬(p x)) ∧ ¬(¬(p x) → ¬(q x m))) →
  (∀ m : ℝ, m ≥ 1 ↔ ∃ x : ℝ, p x ∧ q x m) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2656_265691


namespace NUMINAMATH_CALUDE_tuesday_poodles_l2656_265678

/-- Represents the number of hours it takes to walk a dog of a specific breed --/
def walkTime (breed : String) : ℕ :=
  match breed with
  | "Poodle" => 2
  | "Chihuahua" => 1
  | "Labrador" => 3
  | _ => 0

/-- Represents the schedule for a specific day --/
structure DaySchedule where
  poodles : ℕ
  chihuahuas : ℕ
  labradors : ℕ

def monday : DaySchedule := { poodles := 4, chihuahuas := 2, labradors := 0 }
def wednesday : DaySchedule := { poodles := 0, chihuahuas := 0, labradors := 4 }

def totalHours : ℕ := 32

theorem tuesday_poodles :
  ∃ (tuesday : DaySchedule),
    tuesday.chihuahuas = monday.chihuahuas ∧
    totalHours =
      (monday.poodles * walkTime "Poodle" +
       monday.chihuahuas * walkTime "Chihuahua" +
       wednesday.labradors * walkTime "Labrador" +
       tuesday.poodles * walkTime "Poodle" +
       tuesday.chihuahuas * walkTime "Chihuahua") ∧
    tuesday.poodles = 4 :=
  sorry

end NUMINAMATH_CALUDE_tuesday_poodles_l2656_265678


namespace NUMINAMATH_CALUDE_final_time_sum_l2656_265601

def initial_time : Nat := 15 * 3600  -- 3:00:00 PM in seconds

def elapsed_time : Nat := 317 * 3600 + 15 * 60 + 30  -- 317 hours, 15 minutes, 30 seconds in seconds

def final_time : Nat := (initial_time + elapsed_time) % (24 * 3600)

def hour (t : Nat) : Nat := (t / 3600) % 12

def minute (t : Nat) : Nat := (t / 60) % 60

def second (t : Nat) : Nat := t % 60

theorem final_time_sum :
  hour final_time + minute final_time + second final_time = 53 := by
  sorry

end NUMINAMATH_CALUDE_final_time_sum_l2656_265601


namespace NUMINAMATH_CALUDE_product_equals_three_l2656_265620

theorem product_equals_three : 
  (∀ a b c : ℝ, a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) → 
  6 * 15 * 3 = 3 := by sorry

end NUMINAMATH_CALUDE_product_equals_three_l2656_265620


namespace NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_minus_two_i_l2656_265673

theorem real_part_of_i_squared_times_one_minus_two_i (i : ℂ) : i ^ 2 = -1 → Complex.re (i ^ 2 * (1 - 2 * i)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_squared_times_one_minus_two_i_l2656_265673


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2656_265612

theorem perfect_square_polynomial (m : ℤ) : 
  1 + 2*m + 3*m^2 + 4*m^3 + 5*m^4 + 4*m^5 + 3*m^6 + 2*m^7 + m^8 = (1 + m + m^2 + m^3 + m^4)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2656_265612


namespace NUMINAMATH_CALUDE_geometry_propositions_l2656_265611

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem geometry_propositions 
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n) 
  (h_distinct_planes : α ≠ β) :
  (parallel_plane α β → contains α m → parallel_plane_line β m) ∧
  (parallel_plane_line β m → contains α m → intersect α β n → parallel m n) :=
sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2656_265611


namespace NUMINAMATH_CALUDE_total_jumps_in_3min_l2656_265606

/-- The number of jumps Jung-min can do in 4 minutes -/
def jung_min_jumps_4min : ℕ := 256

/-- The number of jumps Jimin can do in 3 minutes -/
def jimin_jumps_3min : ℕ := 111

/-- The duration we want to calculate the total jumps for (in minutes) -/
def duration : ℕ := 3

/-- Theorem stating that the sum of Jung-min's and Jimin's jumps in 3 minutes is 303 -/
theorem total_jumps_in_3min :
  (jung_min_jumps_4min * duration) / 4 + jimin_jumps_3min = 303 := by
  sorry

#eval (jung_min_jumps_4min * duration) / 4 + jimin_jumps_3min

end NUMINAMATH_CALUDE_total_jumps_in_3min_l2656_265606


namespace NUMINAMATH_CALUDE_rectangle_area_l2656_265669

/-- The area of a rectangle with width 2 feet and length 5 feet is 10 square feet. -/
theorem rectangle_area : 
  let width : ℝ := 2
  let length : ℝ := 5
  width * length = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2656_265669


namespace NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l2656_265626

/-- Given two circles with radius 4 units centered at (0, 20) and (7, 13),
    and a line passing through (4, 0) that divides the total area of both circles equally,
    prove that the absolute value of the slope of this line is 33/15 -/
theorem equal_area_dividing_line_slope (r : ℝ) (c₁ c₂ : ℝ × ℝ) (p : ℝ × ℝ) (m : ℝ) :
  r = 4 →
  c₁ = (0, 20) →
  c₂ = (7, 13) →
  p = (4, 0) →
  (∀ x y, y = m * (x - p.1) + p.2) →
  (∀ x y, (x - c₁.1)^2 + (y - c₁.2)^2 = r^2 → 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1) = 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1)) →
  (∀ x y, (x - c₂.1)^2 + (y - c₂.2)^2 = r^2 → 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1) = 
          abs (y - m * x + m * p.1 - p.2) / Real.sqrt (m^2 + 1)) →
  abs m = 33 / 15 := by
sorry


end NUMINAMATH_CALUDE_equal_area_dividing_line_slope_l2656_265626


namespace NUMINAMATH_CALUDE_marias_minimum_score_l2656_265609

/-- The minimum score needed in the fifth term to achieve a given average -/
def minimum_fifth_score (score1 score2 score3 score4 : ℝ) (required_average : ℝ) : ℝ :=
  5 * required_average - (score1 + score2 + score3 + score4)

/-- Theorem: Maria's minimum required score for the 5th term is 101% -/
theorem marias_minimum_score :
  minimum_fifth_score 84 80 82 78 85 = 101 := by
  sorry

end NUMINAMATH_CALUDE_marias_minimum_score_l2656_265609


namespace NUMINAMATH_CALUDE_arbelos_equal_segments_l2656_265636

/-- Arbelos type representing the geometric figure --/
structure Arbelos where
  -- Define necessary components of an arbelos
  -- (placeholder for actual definition)

/-- Point type representing a point in the plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line in the plane --/
structure Line where
  -- Define necessary components of a line
  -- (placeholder for actual definition)

/-- Function to check if a point is inside an arbelos --/
def isInsideArbelos (a : Arbelos) (p : Point) : Prop :=
  -- Define the condition for a point to be inside an arbelos
  sorry

/-- Function to check if two lines make equal angles with a given line --/
def makeEqualAngles (l1 l2 base : Line) : Prop :=
  -- Define the condition for two lines to make equal angles with a base line
  sorry

/-- Function to get the segment cut by an arbelos on a line --/
def segmentCutByArbelos (a : Arbelos) (l : Line) : ℝ :=
  -- Define how to calculate the segment cut by an arbelos on a line
  sorry

/-- Theorem statement --/
theorem arbelos_equal_segments 
  (a : Arbelos) (ac : Line) (d : Point) (l1 l2 : Line) :
  isInsideArbelos a d →
  makeEqualAngles l1 l2 ac →
  segmentCutByArbelos a l1 = segmentCutByArbelos a l2 :=
sorry

end NUMINAMATH_CALUDE_arbelos_equal_segments_l2656_265636


namespace NUMINAMATH_CALUDE_four_team_hierarchy_exists_l2656_265613

/-- Represents a volleyball team -/
structure Team :=
  (id : Nat)

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a tournament with n teams -/
structure Tournament (n : Nat) :=
  (teams : Fin n → Team)
  (results : Fin n → Fin n → MatchResult)
  (results_valid : ∀ i j, i ≠ j → results i j ≠ results j i)

/-- Theorem stating the existence of four teams with the specified winning relationships -/
theorem four_team_hierarchy_exists (t : Tournament 8) :
  ∃ (a b c d : Fin 8),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    t.results a b = MatchResult.Win ∧
    t.results a c = MatchResult.Win ∧
    t.results a d = MatchResult.Win ∧
    t.results b c = MatchResult.Win ∧
    t.results b d = MatchResult.Win ∧
    t.results c d = MatchResult.Win :=
  sorry

end NUMINAMATH_CALUDE_four_team_hierarchy_exists_l2656_265613


namespace NUMINAMATH_CALUDE_inequality_proof_l2656_265684

theorem inequality_proof (a b : ℝ) (h : (6*a + 9*b)/(a + b) < (4*a - b)/(a - b)) :
  abs b < abs a ∧ abs a < 2 * abs b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2656_265684


namespace NUMINAMATH_CALUDE_system_solution_l2656_265628

theorem system_solution (x y k : ℝ) : 
  x + 2*y = k → 
  2*x + y = 1 → 
  x + y = 3 → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2656_265628


namespace NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2656_265642

/-- Calculates the shaded area of a floor with specific tiling pattern -/
theorem shaded_area_of_tiled_floor (floor_length floor_width tile_size : ℝ)
  (quarter_circle_radius : ℝ) :
  floor_length = 8 →
  floor_width = 10 →
  tile_size = 1 →
  quarter_circle_radius = 1/2 →
  (floor_length * floor_width) * (tile_size^2 - π * quarter_circle_radius^2) = 80 - 20 * π :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2656_265642


namespace NUMINAMATH_CALUDE_max_arrangements_at_six_min_winning_probability_at_six_l2656_265637

/-- The number of cities and days in the championship --/
def n : ℕ := 8

/-- Calculate the number of possible arrangements for k rounds --/
def arrangements (k : ℕ) : ℕ :=
  (Nat.factorial n * Nat.factorial n) / (Nat.factorial (n - k) * Nat.factorial (n - k) * Nat.factorial k)

/-- Theorem stating that 6 rounds maximizes the number of arrangements --/
theorem max_arrangements_at_six :
  ∀ k : ℕ, k ≤ n → arrangements 6 ≥ arrangements k :=
by sorry

/-- Corollary: The probability of winning the grand prize is minimized when there are 6 rounds --/
theorem min_winning_probability_at_six :
  ∀ k : ℕ, k ≤ n → (1 : ℚ) / arrangements 6 ≤ (1 : ℚ) / arrangements k :=
by sorry

end NUMINAMATH_CALUDE_max_arrangements_at_six_min_winning_probability_at_six_l2656_265637


namespace NUMINAMATH_CALUDE_product_in_second_quadrant_l2656_265686

/-- The complex number representing the product (2+i)(-1+i) -/
def z : ℂ := (2 + Complex.I) * (-1 + Complex.I)

/-- The real part of z -/
def real_part : ℝ := z.re

/-- The imaginary part of z -/
def imag_part : ℝ := z.im

/-- Predicate for a complex number being in the second quadrant -/
def in_second_quadrant (w : ℂ) : Prop := w.re < 0 ∧ w.im > 0

theorem product_in_second_quadrant : in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_product_in_second_quadrant_l2656_265686


namespace NUMINAMATH_CALUDE_toms_allowance_l2656_265663

theorem toms_allowance (allowance : ℝ) : 
  (allowance - allowance / 3 - (allowance - allowance / 3) / 4 = 6) → allowance = 12 := by
  sorry

end NUMINAMATH_CALUDE_toms_allowance_l2656_265663


namespace NUMINAMATH_CALUDE_equation_solutions_l2656_265640

theorem equation_solutions : 
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) * (x - 4) * (x - 3) / ((x - 4) * (x - 5)) = -1} = 
  {10/3, 2/3} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2656_265640


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l2656_265649

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, (2 * x - m) / (x + 1) = 1 ∧ x < 0) ↔ (m < -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l2656_265649


namespace NUMINAMATH_CALUDE_jane_finishing_days_l2656_265661

/-- The number of days Jane needs to finish arranging the remaining vases -/
def days_needed (jane_rate mark_rate mark_days total_vases : ℕ) : ℕ :=
  let combined_rate := jane_rate + mark_rate
  let vases_arranged := combined_rate * mark_days
  let remaining_vases := total_vases - vases_arranged
  (remaining_vases + jane_rate - 1) / jane_rate

theorem jane_finishing_days :
  days_needed 16 20 3 248 = 9 := by
  sorry

end NUMINAMATH_CALUDE_jane_finishing_days_l2656_265661


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2656_265664

theorem two_numbers_problem (a b : ℝ) : 
  a + b = 90 ∧ 
  0.4 * a = 0.3 * b + 15 → 
  a = 60 ∧ b = 30 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2656_265664


namespace NUMINAMATH_CALUDE_alex_6_miles_time_l2656_265641

-- Define the running rates for Steve, Jordan, and Alex
def steve_rate : ℚ := 9 / 36

-- Jordan's rate is the same as Steve's
def jordan_rate : ℚ := steve_rate

-- Alex's rate is 2/3 of Jordan's rate
def alex_rate : ℚ := 2 / 3 * jordan_rate

-- Time it takes Steve to run 9 miles
def steve_time : ℚ := 36

-- Time it takes Jordan to run 3 miles
def jordan_time : ℚ := steve_time / 3

-- Time it takes Alex to run 4 miles
def alex_time_4_miles : ℚ := 2 * jordan_time

-- Theorem to prove
theorem alex_6_miles_time :
  6 / alex_rate = 36 := by sorry

end NUMINAMATH_CALUDE_alex_6_miles_time_l2656_265641


namespace NUMINAMATH_CALUDE_expand_product_l2656_265600

theorem expand_product (x y : ℝ) : (3*x - 2) * (2*x + 4*y + 1) = 6*x^2 + 12*x*y - x - 8*y - 2 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2656_265600


namespace NUMINAMATH_CALUDE_fraction_sum_zero_l2656_265697

theorem fraction_sum_zero : 
  (1 / 12 : ℚ) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + 
  (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_zero_l2656_265697


namespace NUMINAMATH_CALUDE_pet_store_dogs_l2656_265665

/-- Given a ratio of cats to dogs and the number of cats, calculate the number of dogs -/
def calculate_dogs (cat_ratio : ℕ) (dog_ratio : ℕ) (num_cats : ℕ) : ℕ :=
  (num_cats / cat_ratio) * dog_ratio

/-- Theorem: Given the ratio of cats to dogs as 3:5 and 18 cats, there are 30 dogs -/
theorem pet_store_dogs : calculate_dogs 3 5 18 = 30 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l2656_265665


namespace NUMINAMATH_CALUDE_intersection_implies_y_zero_l2656_265658

theorem intersection_implies_y_zero (x y : ℝ) : 
  let A : Set ℝ := {2, Real.log x}
  let B : Set ℝ := {x, y}
  A ∩ B = {0} → y = 0 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_y_zero_l2656_265658


namespace NUMINAMATH_CALUDE_distance_between_foci_l2656_265695

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y + 4)^2) + Real.sqrt ((x + 5)^2 + (y - 8)^2) = 20

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (3, -4)
def focus2 : ℝ × ℝ := (-5, 8)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l2656_265695


namespace NUMINAMATH_CALUDE_book_cost_l2656_265627

theorem book_cost : ∃ (x : ℝ), x = 1 + (1/2) * x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_book_cost_l2656_265627


namespace NUMINAMATH_CALUDE_bridget_profit_is_42_l2656_265623

/-- Calculates Bridget's profit from bread sales --/
def bridget_profit (total_loaves : ℕ) (morning_price afternoon_price late_afternoon_price cost_per_loaf : ℚ) : ℚ :=
  let morning_sold := total_loaves / 3
  let morning_revenue := morning_sold * morning_price
  
  let afternoon_remaining := total_loaves - morning_sold
  let afternoon_sold := afternoon_remaining / 2
  let afternoon_revenue := afternoon_sold * afternoon_price
  
  let late_afternoon_remaining := afternoon_remaining - afternoon_sold
  let late_afternoon_sold := late_afternoon_remaining / 4
  let late_afternoon_revenue := late_afternoon_sold * late_afternoon_price
  
  let evening_remaining := late_afternoon_remaining - late_afternoon_sold
  let evening_price := late_afternoon_price / 2
  let evening_revenue := evening_remaining * evening_price
  
  let total_revenue := morning_revenue + afternoon_revenue + late_afternoon_revenue + evening_revenue
  let total_cost := total_loaves * cost_per_loaf
  
  total_revenue - total_cost

/-- Theorem stating Bridget's profit is $42 --/
theorem bridget_profit_is_42 :
  bridget_profit 60 3 (3/2) 1 1 = 42 := by
  sorry


end NUMINAMATH_CALUDE_bridget_profit_is_42_l2656_265623


namespace NUMINAMATH_CALUDE_derivative_of_f_l2656_265688

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) * Real.log x

theorem derivative_of_f (x : ℝ) (h : x > 0) :
  deriv f x = -2 * Real.sin (2 * x) * Real.log x + Real.cos (2 * x) / x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l2656_265688


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2656_265659

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ b : ℝ, (a - 1) * (a + 1 + Complex.I) = Complex.I * b) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2656_265659


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l2656_265655

theorem inequality_not_always_true (a b : ℝ) (h : a > b) :
  ∃ c : ℝ, ¬(a * c^2 > b * c^2) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l2656_265655


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2656_265603

/-- Checks if a natural number is a palindrome when written in the given base. -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Bool :=
  sorry

/-- Finds the smallest base-10 positive integer greater than 15 that is a palindrome 
    when written in both base 2 and base 4. -/
def smallestDualBasePalindrome : ℕ :=
  sorry

theorem smallest_dual_base_palindrome :
  smallestDualBasePalindrome = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2656_265603


namespace NUMINAMATH_CALUDE_map_distance_to_actual_distance_l2656_265644

theorem map_distance_to_actual_distance 
  (map_distance : ℝ) 
  (scale_map : ℝ) 
  (scale_actual : ℝ) 
  (h1 : map_distance = 18) 
  (h2 : scale_map = 0.5) 
  (h3 : scale_actual = 8) : 
  map_distance * (scale_actual / scale_map) = 288 := by
sorry

end NUMINAMATH_CALUDE_map_distance_to_actual_distance_l2656_265644


namespace NUMINAMATH_CALUDE_three_power_plus_five_power_plus_fourteen_equals_factorial_l2656_265662

theorem three_power_plus_five_power_plus_fourteen_equals_factorial :
  ∀ x y z : ℕ, 3^x + 5^y + 14 = z! ↔ (x = 4 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6) := by
  sorry

end NUMINAMATH_CALUDE_three_power_plus_five_power_plus_fourteen_equals_factorial_l2656_265662


namespace NUMINAMATH_CALUDE_door_height_is_eight_l2656_265651

/-- Represents the dimensions of a rectangular door and a pole satisfying specific conditions -/
structure DoorAndPole where
  pole_length : ℝ
  door_width : ℝ
  door_height : ℝ
  horizontal_condition : pole_length = door_width + 4
  vertical_condition : pole_length = door_height + 2
  diagonal_condition : pole_length^2 = door_width^2 + door_height^2

/-- Theorem stating that for any DoorAndPole structure, the door height is 8 -/
theorem door_height_is_eight (d : DoorAndPole) : d.door_height = 8 := by
  sorry

#check door_height_is_eight

end NUMINAMATH_CALUDE_door_height_is_eight_l2656_265651


namespace NUMINAMATH_CALUDE_tenth_row_fifth_column_l2656_265666

/-- Calculates the sum of the first n natural numbers -/
def triangularSum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the triangular arrangement of natural numbers -/
def triangularArrangement (row : ℕ) (col : ℕ) : ℕ :=
  triangularSum (row - 1) + col

/-- The number in the 10th row and 5th column of the triangular arrangement -/
theorem tenth_row_fifth_column :
  triangularArrangement 10 5 = 101 := by
  sorry

end NUMINAMATH_CALUDE_tenth_row_fifth_column_l2656_265666


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2656_265604

theorem polynomial_simplification : 
  2021^4 - 4 * 2023^4 + 6 * 2025^4 - 4 * 2027^4 + 2029^4 = 384 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2656_265604


namespace NUMINAMATH_CALUDE_damage_ratio_is_five_fourths_l2656_265687

/-- The ratio of damages for Winnie-the-Pooh's two falls -/
theorem damage_ratio_is_five_fourths
  (g : ℝ) (H : ℝ) (n : ℝ) (k : ℝ) (M : ℝ) (τ : ℝ)
  (h_pos : 0 < H)
  (n_pos : 0 < n)
  (k_pos : 0 < k)
  (g_pos : 0 < g)
  (h_def : H = n * (H / n)) :
  let h := H / n
  let V_I := Real.sqrt (2 * g * H)
  let V_1 := Real.sqrt (2 * g * h)
  let V_1' := (1 / k) * Real.sqrt (2 * g * h)
  let V_II := Real.sqrt ((1 / k^2) * 2 * g * h + 2 * g * (H - h))
  let I_I := M * V_I * τ
  let I_II := M * τ * ((V_1 - V_1') + V_II)
  I_II / I_I = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_damage_ratio_is_five_fourths_l2656_265687


namespace NUMINAMATH_CALUDE_geometric_sequence_15th_term_l2656_265615

/-- Given a geometric sequence where the 8th term is 8 and the 11th term is 64,
    prove that the 15th term is 1024. -/
theorem geometric_sequence_15th_term (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * (a 11 / a 8)^(1/3)) →
  a 8 = 8 →
  a 11 = 64 →
  a 15 = 1024 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_15th_term_l2656_265615


namespace NUMINAMATH_CALUDE_incident_ray_equation_l2656_265625

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (5, 7)

-- Define the reflection of B across the x-axis
def B_reflected : ℝ × ℝ := (B.1, -B.2)

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  10 * x + 7 * y - 1 = 0

-- Theorem statement
theorem incident_ray_equation :
  line_equation A.1 A.2 ∧
  line_equation B_reflected.1 B_reflected.2 :=
sorry

end NUMINAMATH_CALUDE_incident_ray_equation_l2656_265625


namespace NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2656_265646

def range_size : ℕ := 60
def num_selections : ℕ := 3

def multiples_of_four (n : ℕ) : ℕ := (n + 3) / 4

theorem probability_at_least_one_multiple_of_four :
  let p := 1 - (1 - multiples_of_four range_size / range_size) ^ num_selections
  p = 37 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_multiple_of_four_l2656_265646


namespace NUMINAMATH_CALUDE_sum_first_5_even_numbers_l2656_265616

def first_n_even_numbers (n : ℕ) : List ℕ :=
  List.map (fun i => 2 * i) (List.range n)

theorem sum_first_5_even_numbers :
  (first_n_even_numbers 5).sum = 30 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_5_even_numbers_l2656_265616


namespace NUMINAMATH_CALUDE_special_number_exists_l2656_265639

def is_tail (n m : Nat) : Prop :=
  ∃ k, n = m + k * (10 ^ (Nat.digits 10 m).length)

def has_no_zero_digit (n : Nat) : Prop :=
  ∀ d ∈ Nat.digits 10 n, d ≠ 0

theorem special_number_exists : ∃ n : Nat,
  (Nat.digits 10 n).length = 6 ∧
  has_no_zero_digit n ∧
  ∀ m, is_tail n m → n % m = 0 :=
by
  use 721875
  sorry

#eval Nat.digits 10 721875
#eval 721875 % 21875
#eval 721875 % 1875
#eval 721875 % 875
#eval 721875 % 75
#eval 721875 % 5

end NUMINAMATH_CALUDE_special_number_exists_l2656_265639


namespace NUMINAMATH_CALUDE_number_problem_l2656_265674

theorem number_problem : ∃ x : ℚ, x - (3/5) * x = 60 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2656_265674


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2656_265668

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, (x - 1) * (2 - x) ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2656_265668


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2656_265630

theorem triangle_perimeter (a b c : ℕ) (ha : a = 3) (hb : b = 8) (hc : Odd c) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) :
  a + b + c = 18 ∨ a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2656_265630


namespace NUMINAMATH_CALUDE_earloop_probability_is_0_12_l2656_265619

/-- Represents a mask factory with two types of products -/
structure MaskFactory where
  regularProportion : ℝ
  surgicalProportion : ℝ
  regularEarloopProportion : ℝ
  surgicalEarloopProportion : ℝ

/-- The probability of selecting a mask with ear loops from the factory -/
def earloopProbability (factory : MaskFactory) : ℝ :=
  factory.regularProportion * factory.regularEarloopProportion +
  factory.surgicalProportion * factory.surgicalEarloopProportion

/-- Theorem stating the probability of selecting a mask with ear loops -/
theorem earloop_probability_is_0_12 (factory : MaskFactory)
  (h1 : factory.regularProportion = 0.8)
  (h2 : factory.surgicalProportion = 0.2)
  (h3 : factory.regularEarloopProportion = 0.1)
  (h4 : factory.surgicalEarloopProportion = 0.2) :
  earloopProbability factory = 0.12 := by
  sorry


end NUMINAMATH_CALUDE_earloop_probability_is_0_12_l2656_265619


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l2656_265645

theorem arctan_equation_solution (y : ℝ) :
  4 * Real.arctan (1/5) + Real.arctan (1/25) + Real.arctan (1/y) = π/4 →
  y = 1251 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l2656_265645


namespace NUMINAMATH_CALUDE_xyz_product_l2656_265699

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 200 / 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l2656_265699


namespace NUMINAMATH_CALUDE_marathon_solution_l2656_265694

def marathon_problem (dean_time : ℝ) : Prop :=
  let micah_time := dean_time * (3/2)
  let jake_time := micah_time * (4/3)
  let nia_time := micah_time * 2
  let eliza_time := dean_time * (4/5)
  let average_time := (dean_time + micah_time + jake_time + nia_time + eliza_time) / 5
  dean_time = 9 ∧ average_time = 15.14

theorem marathon_solution :
  ∃ (dean_time : ℝ), marathon_problem dean_time :=
by
  sorry

end NUMINAMATH_CALUDE_marathon_solution_l2656_265694


namespace NUMINAMATH_CALUDE_tangent_line_equations_l2656_265671

/-- The equation of a tangent line to y = x^3 passing through (1, 1) -/
def IsTangentLine (m b : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    (x₀^3 = m * x₀ + b) ∧  -- The line touches the curve at some point (x₀, x₀^3)
    (1 = m * 1 + b) ∧      -- The line passes through (1, 1)
    (m = 3 * x₀^2)         -- The slope of the line equals the derivative of x^3 at x₀

theorem tangent_line_equations :
  ∀ m b : ℝ, IsTangentLine m b ↔ (m = 3 ∧ b = -2) ∨ (m = 3/4 ∧ b = 1/4) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l2656_265671


namespace NUMINAMATH_CALUDE_original_number_proof_l2656_265679

theorem original_number_proof (x : ℝ) (h : 1 - 1/x = 5/2) : x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2656_265679


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2656_265652

def original_price : ℝ := 300
def new_price : ℝ := 390

theorem price_increase_percentage :
  (new_price - original_price) / original_price * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l2656_265652


namespace NUMINAMATH_CALUDE_min_value_expression_l2656_265621

theorem min_value_expression (x : ℝ) :
  ∃ (min : ℝ), min = -1640.25 ∧
  ∀ y : ℝ, (15 - y) * (12 - y) * (15 + y) * (12 + y) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2656_265621


namespace NUMINAMATH_CALUDE_sin_105_cos_105_l2656_265617

theorem sin_105_cos_105 : Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_105_cos_105_l2656_265617


namespace NUMINAMATH_CALUDE_point_on_line_l2656_265689

/-- Given a line with slope 2 and y-intercept 2, the y-coordinate of a point on this line
    with x-coordinate 498 is 998. -/
theorem point_on_line (line : ℝ → ℝ) (x y : ℝ) : 
  (∀ t, line t = 2 * t + 2) →  -- Condition 1 and 2: slope is 2, y-intercept is 2
  x = 498 →                    -- Condition 4: x-coordinate is 498
  y = line x →                 -- Condition 3: the point (x, y) is on the line
  y = 998 := by                -- Question: prove y = 998
sorry


end NUMINAMATH_CALUDE_point_on_line_l2656_265689


namespace NUMINAMATH_CALUDE_problem_solution_l2656_265648

theorem problem_solution (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y * z = 1) (h2 : x + 1 / z = 4) (h3 : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2656_265648


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2656_265638

/-- Given a journey with the following parameters:
  * total_distance: The total distance of the journey in km
  * total_time: The total time of the journey in hours
  * first_half_speed: The speed for the first half of the journey in km/hr
  
  This theorem proves that the speed for the second half of the journey is equal to the second_half_speed. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ)
  (h1 : total_distance = 672)
  (h2 : total_time = 30)
  (h3 : first_half_speed = 21)
  : ∃ second_half_speed : ℝ, second_half_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l2656_265638


namespace NUMINAMATH_CALUDE_five_foxes_weight_l2656_265647

/-- The weight of a single fox in kilograms. -/
def fox_weight : ℝ := sorry

/-- The weight of a single dog in kilograms. -/
def dog_weight : ℝ := fox_weight + 5

/-- The total weight of 3 foxes and 5 dogs in kilograms. -/
def total_weight : ℝ := 65

theorem five_foxes_weight :
  3 * fox_weight + 5 * dog_weight = total_weight →
  5 * fox_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_five_foxes_weight_l2656_265647


namespace NUMINAMATH_CALUDE_john_unanswered_questions_l2656_265696

/-- Represents the scoring systems and John's scores -/
structure ScoringSystem where
  new_correct : ℤ
  new_wrong : ℤ
  new_unanswered : ℤ
  old_start : ℤ
  old_correct : ℤ
  old_wrong : ℤ
  total_questions : ℕ
  new_score : ℤ
  old_score : ℤ

/-- Calculates the number of unanswered questions based on the scoring system -/
def unanswered_questions (s : ScoringSystem) : ℕ :=
  sorry

/-- Theorem stating that for the given scoring system, John left 2 questions unanswered -/
theorem john_unanswered_questions :
  let s : ScoringSystem := {
    new_correct := 6,
    new_wrong := -1,
    new_unanswered := 3,
    old_start := 25,
    old_correct := 5,
    old_wrong := -2,
    total_questions := 30,
    new_score := 105,
    old_score := 95
  }
  unanswered_questions s = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_unanswered_questions_l2656_265696


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2656_265605

theorem coin_flip_probability (p_tails : ℝ) (p_sequence : ℝ) : 
  p_tails = 1/2 → 
  p_sequence = 0.0625 →
  (1 - p_tails) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2656_265605


namespace NUMINAMATH_CALUDE_tan_240_degrees_l2656_265682

theorem tan_240_degrees : Real.tan (240 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_240_degrees_l2656_265682


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2656_265610

/-- The atomic mass of Deuterium (H-2) in atomic mass units (amu) -/
def mass_deuterium : ℝ := 2.014

/-- The atomic mass of Carbon-13 (C-13) in atomic mass units (amu) -/
def mass_carbon13 : ℝ := 13.003

/-- The atomic mass of Oxygen-16 (O-16) in atomic mass units (amu) -/
def mass_oxygen16 : ℝ := 15.995

/-- The atomic mass of Oxygen-18 (O-18) in atomic mass units (amu) -/
def mass_oxygen18 : ℝ := 17.999

/-- The number of Deuterium molecules in the compound -/
def num_deuterium : ℕ := 2

/-- The number of Carbon-13 molecules in the compound -/
def num_carbon13 : ℕ := 1

/-- The number of Oxygen-16 molecules in the compound -/
def num_oxygen16 : ℕ := 1

/-- The number of Oxygen-18 molecules in the compound -/
def num_oxygen18 : ℕ := 2

/-- The molecular weight of the compound -/
def molecular_weight : ℝ :=
  num_deuterium * mass_deuterium +
  num_carbon13 * mass_carbon13 +
  num_oxygen16 * mass_oxygen16 +
  num_oxygen18 * mass_oxygen18

theorem compound_molecular_weight :
  ∃ ε > 0, |molecular_weight - 69.024| < ε :=
sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2656_265610


namespace NUMINAMATH_CALUDE_range_of_a_for_always_negative_l2656_265607

/-- The quadratic function f(x) = ax^2 + ax - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 4

/-- The predicate that f(x) < 0 for all real x -/
def always_negative (a : ℝ) : Prop := ∀ x, f a x < 0

/-- The theorem stating the range of a for which f(x) < 0 holds for all real x -/
theorem range_of_a_for_always_negative :
  ∀ a, always_negative a ↔ -16 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_negative_l2656_265607


namespace NUMINAMATH_CALUDE_slope_range_l2656_265632

theorem slope_range (a : ℝ) :
  let k := -(1 / (a^2 + Real.sqrt 3))
  5 * Real.pi / 6 ≤ Real.arctan k ∧ Real.arctan k < Real.pi :=
by sorry

end NUMINAMATH_CALUDE_slope_range_l2656_265632


namespace NUMINAMATH_CALUDE_final_tomato_count_l2656_265643

def cherry_tomatoes (initial : ℕ) : ℕ :=
  let after_first_birds := initial - (initial / 3)
  let after_second_birds := after_first_birds - (after_first_birds * 2 / 5)
  let after_growth := after_second_birds + (after_second_birds / 2)
  let after_more_growth := after_growth + 4
  after_more_growth - (after_more_growth / 4)

theorem final_tomato_count : cherry_tomatoes 21 = 13 := by
  sorry

end NUMINAMATH_CALUDE_final_tomato_count_l2656_265643


namespace NUMINAMATH_CALUDE_min_sum_box_dimensions_l2656_265633

theorem min_sum_box_dimensions :
  ∀ (l w h : ℕ+),
    l * w * h = 2002 →
    ∀ (a b c : ℕ+),
      a * b * c = 2002 →
      l + w + h ≤ a + b + c →
      l + w + h = 38 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_box_dimensions_l2656_265633


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_count_l2656_265681

theorem arithmetic_sequence_terms_count (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 15 → aₙ = 99 → d = 4 → a₁ + (n - 1) * d = aₙ → n = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_count_l2656_265681


namespace NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l2656_265657

theorem unique_solution_cube_root_equation :
  ∃! x : ℝ, x^(3/5) - 4 = 32 - x^(2/5) := by sorry

end NUMINAMATH_CALUDE_unique_solution_cube_root_equation_l2656_265657


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2656_265683

theorem complex_equation_solution (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (a - Complex.I) * (1 + a * Complex.I) = -4 + 3 * Complex.I →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2656_265683


namespace NUMINAMATH_CALUDE_burger_distance_is_two_l2656_265677

/-- Represents the distance driven to various locations --/
structure Distances where
  school : ℕ
  softball : ℕ
  friend : ℕ
  home : ℕ

/-- Calculates the distance to the burger restaurant given the car's efficiency,
    initial gas, and distances driven to other locations --/
def distance_to_burger (efficiency : ℕ) (initial_gas : ℕ) (distances : Distances) : ℕ :=
  efficiency * initial_gas - (distances.school + distances.softball + distances.friend + distances.home)

/-- Theorem stating that the distance to the burger restaurant is 2 miles --/
theorem burger_distance_is_two :
  let efficiency := 19
  let initial_gas := 2
  let distances := Distances.mk 15 6 4 11
  distance_to_burger efficiency initial_gas distances = 2 := by
  sorry

#check burger_distance_is_two

end NUMINAMATH_CALUDE_burger_distance_is_two_l2656_265677


namespace NUMINAMATH_CALUDE_exam_results_l2656_265635

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 30)
  (h2 : failed_english = 42)
  (h3 : failed_both = 28) :
  100 - (failed_hindi + failed_english - failed_both) = 56 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l2656_265635


namespace NUMINAMATH_CALUDE_factor_and_multiple_greatest_factor_smallest_multiple_smallest_multiple_one_prime_sum_10_product_21_prime_sum_20_product_91_l2656_265608

-- (1)
theorem factor_and_multiple (n : ℕ) : 
  n ∣ 42 ∧ 7 ∣ n ∧ 2 ∣ n ∧ 3 ∣ n → n = 42 := by sorry

-- (2)
theorem greatest_factor_smallest_multiple (n : ℕ) :
  (∀ m : ℕ, m ∣ n → m ≤ 18) ∧ (∀ k : ℕ, n ∣ k → k ≥ 18) → n = 18 := by sorry

-- (3)
theorem smallest_multiple_one (n : ℕ) :
  (∀ k : ℕ, n ∣ k → k ≥ 1) → n = 1 := by sorry

-- (4)
theorem prime_sum_10_product_21 (p q : ℕ) :
  Prime p ∧ Prime q ∧ p + q = 10 ∧ p * q = 21 → (p = 3 ∧ q = 7) ∨ (p = 7 ∧ q = 3) := by sorry

-- (5)
theorem prime_sum_20_product_91 (p q : ℕ) :
  Prime p ∧ Prime q ∧ p + q = 20 ∧ p * q = 91 → (p = 13 ∧ q = 7) ∨ (p = 7 ∧ q = 13) := by sorry

end NUMINAMATH_CALUDE_factor_and_multiple_greatest_factor_smallest_multiple_smallest_multiple_one_prime_sum_10_product_21_prime_sum_20_product_91_l2656_265608


namespace NUMINAMATH_CALUDE_smallest_square_divisible_by_2016_l2656_265614

theorem smallest_square_divisible_by_2016 :
  ∀ n : ℕ, n > 0 → n^2 % 2016 = 0 → n ≥ 168 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_divisible_by_2016_l2656_265614


namespace NUMINAMATH_CALUDE_consecutive_numbers_problem_l2656_265667

/-- Proves that y = 3 given the conditions of the problem -/
theorem consecutive_numbers_problem (x y z : ℤ) 
  (h1 : x = z + 2) 
  (h2 : y = z + 1) 
  (h3 : x > y ∧ y > z) 
  (h4 : 2*x + 3*y + 3*z = 5*y + 8) 
  (h5 : z = 2) : 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_problem_l2656_265667


namespace NUMINAMATH_CALUDE_inner_square_is_square_l2656_265690

/-- A point on a line segment -/
structure PointOnSegment (A B : ℝ × ℝ) where
  point : ℝ × ℝ
  on_segment : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ point = (1 - t) • A + t • B

/-- A square defined by its four vertices -/
structure Square (A B C D : ℝ × ℝ) where
  is_square : sorry  -- Definition of a square

/-- Points dividing sides of a square in the same ratio -/
def divide_sides_equally (ABCD : Square A B C D) 
  (N : PointOnSegment A B) (K : PointOnSegment B C) 
  (L : PointOnSegment C D) (M : PointOnSegment D A) : Prop :=
  sorry  -- Definition of dividing sides equally

theorem inner_square_is_square 
  (ABCD : Square A B C D)
  (N : PointOnSegment A B) (K : PointOnSegment B C) 
  (L : PointOnSegment C D) (M : PointOnSegment D A)
  (h : divide_sides_equally ABCD N K L M) :
  Square N.point K.point L.point M.point :=
sorry

end NUMINAMATH_CALUDE_inner_square_is_square_l2656_265690


namespace NUMINAMATH_CALUDE_power_calculation_l2656_265692

theorem power_calculation : ((5^13 / 5^11)^2 * 5^2) / 2^5 = 15625 / 32 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2656_265692


namespace NUMINAMATH_CALUDE_centroid_vector_sum_centroid_line_ratio_l2656_265660

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variable (A B O G M P Q : V) (a b : V) (m n : ℝ)

/-- G is the centroid of triangle ABO -/
def is_centroid (G A B O : V) : Prop :=
  G = (1/3 : ℝ) • (A + B + O)

/-- M is the midpoint of AB -/
def is_midpoint (M A B : V) : Prop :=
  M = (1/2 : ℝ) • (A + B)

/-- Line PQ passes through G -/
def line_passes_through (P Q G : V) : Prop :=
  ∃ t : ℝ, G = (1 - t) • P + t • Q

theorem centroid_vector_sum 
  (h1 : is_centroid G A B O)
  (h2 : is_midpoint M A B) :
  (G - A) + (G - B) + (G - O) = (0 : V) :=
sorry

theorem centroid_line_ratio
  (h1 : is_centroid G A B O)
  (h2 : O - A = a)
  (h3 : O - B = b)
  (h4 : O - P = m • a)
  (h5 : O - Q = n • b)
  (h6 : line_passes_through P Q G) :
  1/m + 1/n = 3 :=
sorry

end NUMINAMATH_CALUDE_centroid_vector_sum_centroid_line_ratio_l2656_265660


namespace NUMINAMATH_CALUDE_vector_a_start_point_l2656_265676

/-- The endpoint of vector a -/
def B : ℝ × ℝ := (1, 0)

/-- Vector b -/
def b : ℝ × ℝ := (-3, -4)

/-- Vector c -/
def c : ℝ × ℝ := (1, 1)

/-- Vector a in terms of b and c -/
def a : ℝ × ℝ := (3 * b.1 - 2 * c.1, 3 * b.2 - 2 * c.2)

/-- The starting point of vector a -/
def start_point : ℝ × ℝ := (B.1 - a.1, B.2 - a.2)

theorem vector_a_start_point : start_point = (12, 14) := by
  sorry

end NUMINAMATH_CALUDE_vector_a_start_point_l2656_265676


namespace NUMINAMATH_CALUDE_prob_sum_fifteen_is_16_884_l2656_265602

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face
| Ace

/-- A standard 52-card deck --/
def Deck : Finset Card := sorry

/-- The set of number cards (2 through 10) in a standard deck --/
def NumberCards : Finset Card := sorry

/-- The probability of drawing two cards that sum to 15 --/
def probSumFifteen : ℚ := sorry

/-- The main theorem --/
theorem prob_sum_fifteen_is_16_884 : 
  probSumFifteen = 16 / 884 := by sorry

end NUMINAMATH_CALUDE_prob_sum_fifteen_is_16_884_l2656_265602


namespace NUMINAMATH_CALUDE_eat_cereal_together_l2656_265685

/-- The time needed for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate : ℚ) (thin_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (fat_rate + thin_rate)

/-- Theorem stating the time needed for Mr. Fat and Mr. Thin to eat 5 pounds of cereal together -/
theorem eat_cereal_together :
  let fat_rate : ℚ := 1 / 12
  let thin_rate : ℚ := 1 / 40
  let total_amount : ℚ := 5
  time_to_eat_together fat_rate thin_rate total_amount = 600 / 13 := by sorry

end NUMINAMATH_CALUDE_eat_cereal_together_l2656_265685


namespace NUMINAMATH_CALUDE_undefined_expression_l2656_265680

theorem undefined_expression (a : ℝ) : 
  ¬ (∃ x : ℝ, x = (a + 3) / (a^2 - 9)) ↔ a = -3 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_expression_l2656_265680


namespace NUMINAMATH_CALUDE_arithmetic_equality_l2656_265618

theorem arithmetic_equality : 142 + 29 - 32 + 25 = 164 := by sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l2656_265618


namespace NUMINAMATH_CALUDE_last_hour_probability_l2656_265693

/-- The number of attractions available -/
def num_attractions : ℕ := 6

/-- The number of attractions each person chooses -/
def num_chosen : ℕ := 4

/-- The probability of two people being at the same attraction during their last hour -/
def same_attraction_probability : ℚ := 1 / 6

theorem last_hour_probability :
  (num_attractions : ℚ) / (num_attractions * num_attractions) = same_attraction_probability :=
sorry

end NUMINAMATH_CALUDE_last_hour_probability_l2656_265693
