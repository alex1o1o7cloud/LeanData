import Mathlib

namespace NUMINAMATH_CALUDE_inequality_proof_l2992_299241

theorem inequality_proof (a b : ℝ) (n : ℕ) (x₁ y₁ x₂ y₂ A : ℝ) :
  a > 0 →
  b > 0 →
  n > 1 →
  x₁ > 0 →
  y₁ > 0 →
  x₂ > 0 →
  y₂ > 0 →
  x₁^n - a*y₁^n = b →
  x₂^n - a*y₂^n = b →
  y₁ < y₂ →
  A = (1/2) * |x₁*y₂ - x₂*y₁| →
  b*y₂ > 2*n*y₁^(n-1)*a^(1-1/n)*A :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2992_299241


namespace NUMINAMATH_CALUDE_amp_composition_l2992_299289

def amp (x : ℤ) : ℤ := 9 - x
def amp_bar (x : ℤ) : ℤ := x - 9

theorem amp_composition : amp (amp_bar 15) = 15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l2992_299289


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2992_299272

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line L
def line_L (x y m : ℝ) : Prop := x - Real.sqrt 2 * y - m = 0

-- Define the point P
def point_P (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for intersection points A and B
def intersection_condition (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    line_L x₁ y₁ m ∧ line_L x₂ y₂ m ∧
    ((x₁ - m)^2 + y₁^2) * ((x₂ - m)^2 + y₂^2) = 1

-- Theorem statement
theorem intersection_points_theorem :
  ∀ m : ℝ, intersection_condition m ↔ m = 1 + Real.sqrt 7 / 2 ∨ m = 1 - Real.sqrt 7 / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2992_299272


namespace NUMINAMATH_CALUDE_seven_digit_palindromes_count_l2992_299266

/-- Represents a multiset of digits --/
def DigitMultiset := Multiset Nat

/-- Checks if a number is a palindrome --/
def isPalindrome (n : Nat) : Bool := sorry

/-- Counts the number of 7-digit palindromes that can be formed from a given multiset of digits --/
def countSevenDigitPalindromes (digits : DigitMultiset) : Nat := sorry

/-- The specific multiset of digits given in the problem --/
def givenDigits : DigitMultiset := sorry

theorem seven_digit_palindromes_count :
  countSevenDigitPalindromes givenDigits = 18 := by sorry

end NUMINAMATH_CALUDE_seven_digit_palindromes_count_l2992_299266


namespace NUMINAMATH_CALUDE_min_value_on_line_l2992_299201

/-- The minimum value of ((a+1)^2 + b^2) is 3, given that (a,b) is on y = √3x - √3 -/
theorem min_value_on_line :
  ∀ a b : ℝ, b = Real.sqrt 3 * a - Real.sqrt 3 →
  (∀ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 → (x + 1)^2 + y^2 ≥ 3) ∧
  ∃ x y : ℝ, y = Real.sqrt 3 * x - Real.sqrt 3 ∧ (x + 1)^2 + y^2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_line_l2992_299201


namespace NUMINAMATH_CALUDE_sum_difference_1500_l2992_299204

/-- The sum of the first n odd counting numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n * n

/-- The sum of the first n even counting numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even counting numbers
    and the sum of the first n odd counting numbers -/
def sumDifference (n : ℕ) : ℕ := sumEvenNumbers n - sumOddNumbers n

theorem sum_difference_1500 : sumDifference 1500 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_1500_l2992_299204


namespace NUMINAMATH_CALUDE_inequality_proof_l2992_299277

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  (1 / (2 - a)) + (1 / (2 - b)) + (1 / (2 - c)) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2992_299277


namespace NUMINAMATH_CALUDE_inequality_proof_l2992_299225

theorem inequality_proof (x y : ℝ) (h1 : x > -1) (h2 : y > -1) (h3 : x + y = 1) :
  x / (y + 1) + y / (x + 1) ≥ 2/3 ∧ 
  (x / (y + 1) + y / (x + 1) = 2/3 ↔ x = 1/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2992_299225


namespace NUMINAMATH_CALUDE_water_bottle_consumption_l2992_299261

theorem water_bottle_consumption (total_bottles : ℕ) (days : ℕ) (bottles_per_day : ℕ) : 
  total_bottles = 153 → 
  days = 17 → 
  total_bottles = bottles_per_day * days → 
  bottles_per_day = 9 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_consumption_l2992_299261


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2992_299254

/-- The line 3x - 4y - 5 = 0 is tangent to the circle (x - 1)^2 + (y + 3)^2 = 4 -/
theorem line_tangent_to_circle :
  ∃ (x y : ℝ),
    (3 * x - 4 * y - 5 = 0) ∧
    ((x - 1)^2 + (y + 3)^2 = 4) ∧
    (∀ (x' y' : ℝ), (3 * x' - 4 * y' - 5 = 0) → ((x' - 1)^2 + (y' + 3)^2 ≥ 4)) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2992_299254


namespace NUMINAMATH_CALUDE_map_length_l2992_299268

/-- The length of a rectangular map given its area and width -/
theorem map_length (area : ℝ) (width : ℝ) (h1 : area = 10) (h2 : width = 2) :
  area / width = 5 := by
  sorry

end NUMINAMATH_CALUDE_map_length_l2992_299268


namespace NUMINAMATH_CALUDE_expression_evaluation_l2992_299265

theorem expression_evaluation : 
  (0.66 : ℝ)^3 - (0.1 : ℝ)^3 / (0.66 : ℝ)^2 + 0.066 + (0.1 : ℝ)^2 = 0.3612 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2992_299265


namespace NUMINAMATH_CALUDE_max_juggling_time_max_juggling_time_value_l2992_299231

/-- Represents the time in seconds before Bobo drops a cow when juggling n cows -/
def drop_time (n : ℕ) : ℕ :=
  match n with
  | 1 => 64
  | 2 => 55
  | 3 => 47
  | 4 => 40
  | 5 => 33
  | 6 => 27
  | 7 => 22
  | 8 => 18
  | 9 => 14
  | 10 => 13
  | 11 => 12
  | 12 => 11
  | 13 => 10
  | 14 => 9
  | 15 => 8
  | 16 => 7
  | 17 => 6
  | 18 => 5
  | 19 => 4
  | 20 => 3
  | 21 => 2
  | 22 => 1
  | _ => 0

/-- Calculates the total juggling time for n cows -/
def total_time (n : ℕ) : ℕ := n * drop_time n

/-- The maximum number of cows Bobo can juggle -/
def max_cows : ℕ := 22

/-- Theorem: The maximum total juggling time is achieved with 5 cows -/
theorem max_juggling_time :
  ∀ n : ℕ, n ≤ max_cows → total_time 5 ≥ total_time n :=
by
  sorry

/-- Corollary: The maximum total juggling time is 165 seconds -/
theorem max_juggling_time_value : total_time 5 = 165 :=
by
  sorry

end NUMINAMATH_CALUDE_max_juggling_time_max_juggling_time_value_l2992_299231


namespace NUMINAMATH_CALUDE_inequality_solution_l2992_299294

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 4) > 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2992_299294


namespace NUMINAMATH_CALUDE_sign_up_ways_for_six_students_three_projects_l2992_299280

/-- The number of ways students can sign up for projects -/
def signUpWays (numStudents : ℕ) (numProjects : ℕ) : ℕ :=
  numProjects ^ numStudents

/-- Theorem: For 6 students and 3 projects, the number of ways to sign up is 3^6 -/
theorem sign_up_ways_for_six_students_three_projects :
  signUpWays 6 3 = 3^6 := by
  sorry

end NUMINAMATH_CALUDE_sign_up_ways_for_six_students_three_projects_l2992_299280


namespace NUMINAMATH_CALUDE_b_not_played_e_l2992_299247

/-- Represents a soccer team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents the number of matches played by each team -/
def matches_played : Team → Nat
| Team.A => 5
| Team.B => 4
| Team.C => 3
| Team.D => 2
| Team.E => 1
| Team.F => 0  -- Inferred from the problem

/-- Predicate to check if two teams have played against each other -/
def has_played_against : Team → Team → Prop := sorry

/-- The theorem stating that team B has not played against team E -/
theorem b_not_played_e : ¬(has_played_against Team.B Team.E) := by
  sorry

end NUMINAMATH_CALUDE_b_not_played_e_l2992_299247


namespace NUMINAMATH_CALUDE_sample_size_is_450_l2992_299226

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ

/-- Theorem: Given a population of 5000 students and a sample of 450 students,
    the sample size is 450. -/
theorem sample_size_is_450 (pop : Population) (sample : Sample) 
    (h1 : pop.size = 5000) (h2 : sample.size = 450) : 
  sample.size = 450 := by
  sorry

#check sample_size_is_450

end NUMINAMATH_CALUDE_sample_size_is_450_l2992_299226


namespace NUMINAMATH_CALUDE_small_planters_needed_l2992_299256

/-- Represents the types of seeds --/
inductive SeedType
  | Basil
  | Cilantro
  | Parsley

/-- Represents the types of planters --/
inductive PlanterType
  | Large
  | Medium
  | Small

/-- Represents the planting requirements for each seed type --/
def plantingRequirement (s : SeedType) : Set PlanterType :=
  match s with
  | SeedType.Basil => {PlanterType.Large, PlanterType.Medium}
  | SeedType.Cilantro => {PlanterType.Medium}
  | SeedType.Parsley => {PlanterType.Large, PlanterType.Medium, PlanterType.Small}

/-- The capacity of each planter type --/
def planterCapacity (p : PlanterType) : ℕ :=
  match p with
  | PlanterType.Large => 20
  | PlanterType.Medium => 10
  | PlanterType.Small => 4

/-- The number of each planter type available --/
def planterCount (p : PlanterType) : ℕ :=
  match p with
  | PlanterType.Large => 4
  | PlanterType.Medium => 8
  | PlanterType.Small => 0  -- We're solving for this

/-- The number of seeds for each seed type --/
def seedCount (s : SeedType) : ℕ :=
  match s with
  | SeedType.Basil => 200
  | SeedType.Cilantro => 160
  | SeedType.Parsley => 120

theorem small_planters_needed : 
  ∃ (n : ℕ), 
    n * planterCapacity PlanterType.Small = 
      seedCount SeedType.Parsley + 
      (seedCount SeedType.Cilantro - planterCount PlanterType.Medium * planterCapacity PlanterType.Medium) + 
      (seedCount SeedType.Basil - 
        (planterCount PlanterType.Large * planterCapacity PlanterType.Large + 
         (planterCount PlanterType.Medium - 
          (seedCount SeedType.Cilantro / planterCapacity PlanterType.Medium)) * 
           planterCapacity PlanterType.Medium)) ∧ 
    n = 50 := by
  sorry

end NUMINAMATH_CALUDE_small_planters_needed_l2992_299256


namespace NUMINAMATH_CALUDE_slab_rate_calculation_l2992_299232

/-- Given a room with specified dimensions and total flooring cost, 
    calculate the rate per square meter for the slabs. -/
theorem slab_rate_calculation (length width total_cost : ℝ) 
    (h_length : length = 5.5)
    (h_width : width = 3.75)
    (h_total_cost : total_cost = 24750) : 
  total_cost / (length * width) = 1200 := by
  sorry


end NUMINAMATH_CALUDE_slab_rate_calculation_l2992_299232


namespace NUMINAMATH_CALUDE_factorization_equality_l2992_299295

theorem factorization_equality (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2992_299295


namespace NUMINAMATH_CALUDE_a_monotonically_decreasing_iff_t_lt_3_l2992_299298

/-- The sequence a_n defined as -n^2 + tn for positive integers n and constant t -/
def a (n : ℕ+) (t : ℝ) : ℝ := -n.val^2 + t * n.val

/-- A sequence is monotonically decreasing if each term is less than the previous term -/
def monotonically_decreasing (s : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, s (n + 1) < s n

/-- The main theorem: the sequence a_n is monotonically decreasing iff t < 3 -/
theorem a_monotonically_decreasing_iff_t_lt_3 (t : ℝ) :
  monotonically_decreasing (a · t) ↔ t < 3 := by
  sorry

end NUMINAMATH_CALUDE_a_monotonically_decreasing_iff_t_lt_3_l2992_299298


namespace NUMINAMATH_CALUDE_product_of_roots_l2992_299227

theorem product_of_roots (x : ℝ) : 
  (6 = 2 * x^2 + 4 * x) → 
  (let a := 2
   let b := 4
   let c := -6
   c / a = -3) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l2992_299227


namespace NUMINAMATH_CALUDE_inlet_fill_rate_l2992_299249

/-- Given a tank with the following properties:
  * Capacity: 12960 liters
  * Time to empty with leak alone: 9 hours
  * Time to empty with leak and inlet: 12 hours
  Prove that the rate at which the inlet pipe fills water is 2520 liters per hour. -/
theorem inlet_fill_rate 
  (tank_capacity : ℝ) 
  (empty_time_leak : ℝ) 
  (empty_time_leak_and_inlet : ℝ) 
  (h1 : tank_capacity = 12960)
  (h2 : empty_time_leak = 9)
  (h3 : empty_time_leak_and_inlet = 12) :
  (tank_capacity / empty_time_leak) + (tank_capacity / empty_time_leak_and_inlet) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_inlet_fill_rate_l2992_299249


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2992_299287

theorem unknown_number_proof (n : ℕ) : (n ^ 1) * 6 ^ 4 / 432 = 36 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2992_299287


namespace NUMINAMATH_CALUDE_busy_schedule_starts_26th_l2992_299230

/-- Represents the reading schedule for September --/
structure ReadingSchedule where
  total_pages : ℕ
  total_days : ℕ
  busy_days : ℕ
  special_day : ℕ
  special_day_pages : ℕ
  daily_pages : ℕ

/-- Calculates the start day of the busy schedule --/
def busy_schedule_start (schedule : ReadingSchedule) : ℕ :=
  schedule.total_days - 
  ((schedule.total_pages - schedule.special_day_pages) / schedule.daily_pages) - 
  1

/-- Theorem stating that the busy schedule starts on the 26th --/
theorem busy_schedule_starts_26th (schedule : ReadingSchedule) 
  (h1 : schedule.total_pages = 600)
  (h2 : schedule.total_days = 30)
  (h3 : schedule.busy_days = 4)
  (h4 : schedule.special_day = 23)
  (h5 : schedule.special_day_pages = 100)
  (h6 : schedule.daily_pages = 20) :
  busy_schedule_start schedule = 26 := by
  sorry

#eval busy_schedule_start {
  total_pages := 600,
  total_days := 30,
  busy_days := 4,
  special_day := 23,
  special_day_pages := 100,
  daily_pages := 20
}

end NUMINAMATH_CALUDE_busy_schedule_starts_26th_l2992_299230


namespace NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l2992_299207

/-- The number of squares on a chessboard. -/
def chessboardSize : ℕ := 64

/-- The number of rooks placed on the chessboard. -/
def numberOfRooks : ℕ := 3

/-- The probability that a specific square is not attacked by a single rook. -/
def probNotAttackedByOneRook : ℚ := 49 / 64

/-- The expected number of squares under attack when three rooks are randomly placed on a chessboard. -/
def expectedAttackedSquares : ℚ :=
  chessboardSize * (1 - probNotAttackedByOneRook ^ numberOfRooks)

/-- Theorem stating that the expected number of squares under attack is equal to the calculated value. -/
theorem expected_attacked_squares_theorem :
  expectedAttackedSquares = 64 * (1 - (49/64)^3) :=
sorry

end NUMINAMATH_CALUDE_expected_attacked_squares_theorem_l2992_299207


namespace NUMINAMATH_CALUDE_largest_sample_number_l2992_299278

/-- Systematic sampling from a set of numbered items -/
def systematic_sample (total : ℕ) (first : ℕ) (second : ℕ) : ℕ := 
  let interval := second - first
  let sample_size := total / interval
  first + interval * (sample_size - 1)

/-- The largest number in a systematic sample from 500 items -/
theorem largest_sample_number : 
  systematic_sample 500 7 32 = 482 := by
  sorry

end NUMINAMATH_CALUDE_largest_sample_number_l2992_299278


namespace NUMINAMATH_CALUDE_root_property_l2992_299264

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem root_property (f : ℝ → ℝ) (x₀ : ℝ) 
  (h_odd : is_odd f) 
  (h_root : f x₀ = Real.exp x₀) :
  f (-x₀) * Real.exp (-x₀) + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l2992_299264


namespace NUMINAMATH_CALUDE_derivative_f_at_3pi_4_l2992_299257

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.cos x + 1

theorem derivative_f_at_3pi_4 :
  deriv f (3 * Real.pi / 4) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_3pi_4_l2992_299257


namespace NUMINAMATH_CALUDE_lines_intersection_l2992_299284

-- Define the two lines
def line1 (t : ℝ) : ℝ × ℝ := (3 * t, 2 + 4 * t)
def line2 (u : ℝ) : ℝ × ℝ := (1 + u, 1 - u)

-- State the theorem
theorem lines_intersection :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) ∧ p = (0, 2) := by
  sorry

end NUMINAMATH_CALUDE_lines_intersection_l2992_299284


namespace NUMINAMATH_CALUDE_A_intersect_B_l2992_299291

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | ∃ m : ℝ, x = m^2 + 1}

theorem A_intersect_B : A ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2992_299291


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2992_299276

theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  let first_half_distance := total_distance / 2
  let first_half_time := first_half_distance / first_half_speed
  let second_half_time := first_half_time * second_half_time_factor
  let total_time := first_half_time + second_half_time
  (total_distance / total_time) = 40 := by
sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2992_299276


namespace NUMINAMATH_CALUDE_g_f_three_equals_one_l2992_299282

-- Define the domain of x
inductive Domain : Type
| one : Domain
| two : Domain
| three : Domain
| four : Domain

-- Define function f
def f : Domain → Domain
| Domain.one => Domain.three
| Domain.two => Domain.four
| Domain.three => Domain.two
| Domain.four => Domain.one

-- Define function g
def g : Domain → ℕ
| Domain.one => 2
| Domain.two => 1
| Domain.three => 6
| Domain.four => 8

-- Theorem to prove
theorem g_f_three_equals_one : g (f Domain.three) = 1 := by
  sorry

end NUMINAMATH_CALUDE_g_f_three_equals_one_l2992_299282


namespace NUMINAMATH_CALUDE_equation_solution_l2992_299222

theorem equation_solution (x y : ℝ) 
  (h : |x - Real.log y| + Real.sin (π * x) = x + Real.log y) : 
  x = 0 ∧ Real.exp (-1/2) ≤ y ∧ y ≤ Real.exp (1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2992_299222


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l2992_299237

theorem floor_ceiling_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l2992_299237


namespace NUMINAMATH_CALUDE_line_plane_relationships_l2992_299246

-- Define the basic structures
variable (α : Plane) (l m : Line)

-- Define the relationships
def not_contained_in (l : Line) (α : Plane) : Prop := sorry
def contained_in (m : Line) (α : Plane) : Prop := sorry
def perpendicular (l : Line) (α : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry
def perpendicular_lines (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- State the theorem
theorem line_plane_relationships :
  not_contained_in l α →
  contained_in m α →
  ((perpendicular l α → perpendicular_lines l m) ∧
   (parallel_lines l m → parallel_line_plane l α)) :=
by sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l2992_299246


namespace NUMINAMATH_CALUDE_smallest_number_of_groups_l2992_299258

theorem smallest_number_of_groups (total_campers : ℕ) (max_group_size : ℕ) : 
  total_campers = 36 → max_group_size = 12 → 
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) ∧
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_campers ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_campers → k ≥ num_groups) → num_groups = 3 :=
by sorry


end NUMINAMATH_CALUDE_smallest_number_of_groups_l2992_299258


namespace NUMINAMATH_CALUDE_tan_960_degrees_l2992_299209

theorem tan_960_degrees : Real.tan (960 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_960_degrees_l2992_299209


namespace NUMINAMATH_CALUDE_inequality_proof_l2992_299293

theorem inequality_proof (x y z : ℝ) :
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2992_299293


namespace NUMINAMATH_CALUDE_paint_cans_used_paint_cans_theorem_l2992_299271

-- Define the initial number of rooms that could be painted
def initial_rooms : ℕ := 50

-- Define the number of cans lost
def lost_cans : ℕ := 5

-- Define the number of rooms that can be painted after losing cans
def remaining_rooms : ℕ := 38

-- Define the number of small rooms to be painted
def small_rooms : ℕ := 35

-- Define the number of large rooms to be painted
def large_rooms : ℕ := 5

-- Define the paint requirement ratio of large rooms to small rooms
def large_room_ratio : ℕ := 2

-- Theorem to prove
theorem paint_cans_used : ℕ := by
  -- The proof goes here
  sorry

-- Goal: prove that paint_cans_used = 19
theorem paint_cans_theorem : paint_cans_used = 19 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_paint_cans_theorem_l2992_299271


namespace NUMINAMATH_CALUDE_combined_tower_height_l2992_299242

/-- The combined height of four towers given specific conditions -/
theorem combined_tower_height :
  ∀ (clyde grace sarah linda : ℝ),
  grace = 8 * clyde →
  grace = 40.5 →
  sarah = 2 * clyde →
  linda = (clyde + grace + sarah) / 3 →
  clyde + grace + sarah + linda = 74.25 := by
  sorry

end NUMINAMATH_CALUDE_combined_tower_height_l2992_299242


namespace NUMINAMATH_CALUDE_fraction_sum_squared_l2992_299244

theorem fraction_sum_squared (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_squared_l2992_299244


namespace NUMINAMATH_CALUDE_mean_equality_implies_x_value_mean_equality_proof_l2992_299206

theorem mean_equality_implies_x_value : ℝ → Prop :=
  fun x =>
    (11 + 14 + 25) / 3 = (18 + x + 4) / 3 → x = 28

-- Proof
theorem mean_equality_proof : mean_equality_implies_x_value 28 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_x_value_mean_equality_proof_l2992_299206


namespace NUMINAMATH_CALUDE_gary_flour_amount_l2992_299217

/-- Proves that Gary has 6 pounds of flour given the problem conditions -/
theorem gary_flour_amount :
  ∀ (total_flour : ℝ) 
    (cake_flour cupcake_flour : ℝ) 
    (cake_price cupcake_price : ℝ) 
    (total_earnings : ℝ),
  cake_flour = 4 →
  cupcake_flour = total_flour - cake_flour →
  (cake_flour / 0.5) * cake_price + (cupcake_flour / (1/5)) * cupcake_price = total_earnings →
  cake_price = 2.5 →
  cupcake_price = 1 →
  total_earnings = 30 →
  total_flour = 6 := by
sorry

end NUMINAMATH_CALUDE_gary_flour_amount_l2992_299217


namespace NUMINAMATH_CALUDE_greatest_x_value_l2992_299233

theorem greatest_x_value (x : ℝ) : 
  ((5*x - 20)^2 / (4*x - 5)^2 + (5*x - 20) / (4*x - 5) = 12) → 
  x ≤ 40/21 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2992_299233


namespace NUMINAMATH_CALUDE_spherical_coordinate_conversion_l2992_299299

/-- Proves that the given spherical coordinates are equivalent to the standard representation -/
theorem spherical_coordinate_conversion (ρ θ φ : Real) :
  ρ > 0 →
  0 ≤ θ ∧ θ < 2 * π →
  0 ≤ φ ∧ φ ≤ π →
  (ρ, θ, φ) = (4, 4 * π / 3, π / 5) ↔ (ρ, θ, φ) = (4, π / 3, 9 * π / 5) :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_conversion_l2992_299299


namespace NUMINAMATH_CALUDE_max_red_balls_l2992_299248

/-- Given a pile of red and white balls, with the total number not exceeding 50,
    and the number of red balls being three times the number of white balls,
    prove that the maximum number of red balls is 36. -/
theorem max_red_balls (r w : ℕ) : 
  r + w ≤ 50 →  -- Total number of balls not exceeding 50
  r = 3 * w →   -- Number of red balls is three times the number of white balls
  r ≤ 36        -- Maximum number of red balls is 36
  := by sorry

end NUMINAMATH_CALUDE_max_red_balls_l2992_299248


namespace NUMINAMATH_CALUDE_square_field_area_l2992_299292

/-- The area of a square field with side length 25 meters is 625 square meters. -/
theorem square_field_area : 
  let side_length : ℝ := 25
  let area : ℝ := side_length * side_length
  area = 625 := by sorry

end NUMINAMATH_CALUDE_square_field_area_l2992_299292


namespace NUMINAMATH_CALUDE_translated_minimum_point_l2992_299251

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 2

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - 3) + 4

-- State the theorem
theorem translated_minimum_point :
  ∃ (x_min : ℝ), (∀ (x : ℝ), g x_min ≤ g x) ∧ x_min = 2 ∧ g x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_translated_minimum_point_l2992_299251


namespace NUMINAMATH_CALUDE_no_real_solutions_l2992_299262

theorem no_real_solutions : ¬∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -abs x := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2992_299262


namespace NUMINAMATH_CALUDE_number_difference_l2992_299215

theorem number_difference (L S : ℝ) (h1 : L = 1650) (h2 : L = 6 * S + 15) : 
  L - S = 1377.5 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2992_299215


namespace NUMINAMATH_CALUDE_local_max_derivative_condition_l2992_299281

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f attains a local maximum at x = a, then a is in the open interval (-1, 0) -/
theorem local_max_derivative_condition (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h2 : IsLocalMax f a) :
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_local_max_derivative_condition_l2992_299281


namespace NUMINAMATH_CALUDE_y_derivative_l2992_299252

open Real

noncomputable def y (x : ℝ) : ℝ :=
  Real.sqrt (9 * x^2 - 12 * x + 5) * arctan (3 * x - 2) - log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem y_derivative (x : ℝ) :
  deriv y x = ((9 * x - 6) * arctan (3 * x - 2)) / Real.sqrt (9 * x^2 - 12 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l2992_299252


namespace NUMINAMATH_CALUDE_divisibility_property_l2992_299228

theorem divisibility_property (y : ℕ) (hy : y ≠ 0) :
  (y - 1) ∣ (y^(y^2) - 2*y^(y+1) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2992_299228


namespace NUMINAMATH_CALUDE_ordered_pairs_1764_l2992_299288

/-- The number of ordered pairs of positive integers (x,y) that satisfy xy = n,
    where n has the prime factorization p₁^a₁ * p₂^a₂ * ... * pₖ^aₖ -/
def count_ordered_pairs (n : ℕ) (primes : List ℕ) (exponents : List ℕ) : ℕ :=
  sorry

theorem ordered_pairs_1764 :
  count_ordered_pairs 1764 [2, 3, 7] [2, 2, 2] = 27 :=
sorry

end NUMINAMATH_CALUDE_ordered_pairs_1764_l2992_299288


namespace NUMINAMATH_CALUDE_jack_king_ace_probability_l2992_299219

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the event of drawing three specific cards in order -/
def draw_three_cards (d : Deck) (first second third : Fin 52) : ℚ :=
  (4 : ℚ) / 52 * (4 : ℚ) / 51 * (4 : ℚ) / 50

/-- The probability of drawing a Jack, then a King, then an Ace from a standard deck without replacement -/
theorem jack_king_ace_probability (d : Deck) :
  ∃ (j k a : Fin 52), draw_three_cards d j k a = 16 / 33150 :=
sorry

end NUMINAMATH_CALUDE_jack_king_ace_probability_l2992_299219


namespace NUMINAMATH_CALUDE_third_term_of_geometric_series_l2992_299269

/-- Given an infinite geometric series with common ratio 1/4 and sum 16,
    the third term of the series is 3/4. -/
theorem third_term_of_geometric_series :
  ∀ (a : ℝ),
  (a / (1 - (1/4 : ℝ)) = 16) →  -- Sum of infinite geometric series
  (a * (1/4 : ℝ)^2 = 3/4) :=    -- Third term of the series
by sorry

end NUMINAMATH_CALUDE_third_term_of_geometric_series_l2992_299269


namespace NUMINAMATH_CALUDE_min_cost_grass_seed_l2992_299220

/-- Represents a bag of grass seed -/
structure GrassSeedBag where
  weight : Nat
  price : Rat
  deriving Repr

/-- Calculates the total weight of a list of bags -/
def totalWeight (bags : List GrassSeedBag) : Nat :=
  bags.foldl (fun acc bag => acc + bag.weight) 0

/-- Calculates the total cost of a list of bags -/
def totalCost (bags : List GrassSeedBag) : Rat :=
  bags.foldl (fun acc bag => acc + bag.price) 0

/-- Checks if a list of bags satisfies the purchase conditions -/
def isValidPurchase (bags : List GrassSeedBag) : Prop :=
  totalWeight bags ≥ 65 ∧
  totalWeight bags ≤ 80 ∧
  bags.length ≤ 5 ∧
  bags.length ≥ 4 ∧
  (∃ b ∈ bags, b.weight = 5) ∧
  (∃ b ∈ bags, b.weight = 10) ∧
  (∃ b ∈ bags, b.weight = 25) ∧
  (∃ b ∈ bags, b.weight = 40)

theorem min_cost_grass_seed :
  let bags := [
    GrassSeedBag.mk 5 (13.85),
    GrassSeedBag.mk 10 (20.43),
    GrassSeedBag.mk 25 (32.20),
    GrassSeedBag.mk 40 (54.30)
  ]
  ∀ purchase : List GrassSeedBag,
    isValidPurchase purchase →
    totalCost purchase ≥ 120.78 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_grass_seed_l2992_299220


namespace NUMINAMATH_CALUDE_alcohol_concentration_proof_l2992_299267

/-- Proves that adding 3.6 liters of pure alcohol to a 6-liter solution
    containing 20% alcohol results in a solution with 50% alcohol concentration. -/
theorem alcohol_concentration_proof (initial_volume : Real) (initial_concentration : Real)
  (added_alcohol : Real) (final_concentration : Real)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.2)
  (h3 : added_alcohol = 3.6)
  (h4 : final_concentration = 0.5) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_proof_l2992_299267


namespace NUMINAMATH_CALUDE_sequence_fixed_points_l2992_299250

theorem sequence_fixed_points 
  (a b c d : ℝ) 
  (h1 : c ≠ 0) 
  (h2 : a * d - b * c ≠ 0) 
  (a_n : ℕ → ℝ) 
  (h_seq : ∀ n, a_n (n + 1) = (a * a_n n + b) / (c * a_n n + d)) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ 
    (a * x₁ + b) / (c * x₁ + d) = x₁ ∧ 
    (a * x₂ + b) / (c * x₂ + d) = x₂ →
    ∀ n, (a_n (n + 1) - x₁) / (a_n (n + 1) - x₂) = 
         ((a - c * x₁) / (a - c * x₂)) * ((a_n n - x₁) / (a_n n - x₂))) ∧
  (∃ x₀, (a * x₀ + b) / (c * x₀ + d) = x₀ ∧ a ≠ -d →
    ∀ n, 1 / (a_n (n + 1) - x₀) = (2 * c) / (a + d) + 1 / (a_n n - x₀)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_fixed_points_l2992_299250


namespace NUMINAMATH_CALUDE_smallest_three_digit_even_in_pascal_l2992_299273

/-- Pascal's triangle coefficient -/
def pascal (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- Check if a number is in Pascal's triangle -/
def inPascalTriangle (m : ℕ) : Prop :=
  ∃ n k : ℕ, pascal n k = m

/-- The smallest three-digit even number in Pascal's triangle -/
def smallestThreeDigitEvenInPascal : ℕ := 120

theorem smallest_three_digit_even_in_pascal :
  (inPascalTriangle smallestThreeDigitEvenInPascal) ∧
  (smallestThreeDigitEvenInPascal % 2 = 0) ∧
  (smallestThreeDigitEvenInPascal ≥ 100) ∧
  (smallestThreeDigitEvenInPascal < 1000) ∧
  (∀ m : ℕ, m < smallestThreeDigitEvenInPascal →
    m % 2 = 0 → m ≥ 100 → m < 1000 → ¬(inPascalTriangle m)) := by
  sorry

#check smallest_three_digit_even_in_pascal

end NUMINAMATH_CALUDE_smallest_three_digit_even_in_pascal_l2992_299273


namespace NUMINAMATH_CALUDE_crayon_count_l2992_299245

theorem crayon_count (blue : ℕ) (red : ℕ) : 
  blue = 3 → red = 4 * blue → blue + red = 15 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_l2992_299245


namespace NUMINAMATH_CALUDE_courtyard_width_is_20_l2992_299234

/-- Represents a rectangular paving stone -/
structure PavingStone where
  length : ℝ
  width : ℝ

/-- Represents a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a paving stone -/
def area_paving_stone (stone : PavingStone) : ℝ :=
  stone.length * stone.width

/-- Calculates the area of a courtyard -/
def area_courtyard (yard : Courtyard) : ℝ :=
  yard.length * yard.width

/-- Theorem: The width of the courtyard is 20 meters -/
theorem courtyard_width_is_20 (stone : PavingStone) (yard : Courtyard) 
    (h1 : stone.length = 4)
    (h2 : stone.width = 2)
    (h3 : yard.length = 40)
    (h4 : area_courtyard yard = 100 * area_paving_stone stone) :
    yard.width = 20 := by
  sorry

#check courtyard_width_is_20

end NUMINAMATH_CALUDE_courtyard_width_is_20_l2992_299234


namespace NUMINAMATH_CALUDE_max_min_distance_on_sphere_l2992_299279

/-- A point on a unit sphere represented by its coordinates -/
def SpherePoint := ℝ × ℝ × ℝ

/-- The distance between two points on a unit sphere -/
def sphereDistance (p q : SpherePoint) : ℝ := sorry

/-- Checks if a point is on the unit sphere -/
def isOnUnitSphere (p : SpherePoint) : Prop := sorry

/-- Represents a configuration of five points on a unit sphere -/
def Configuration := Fin 5 → SpherePoint

/-- The minimum pairwise distance in a configuration -/
def minDistance (c : Configuration) : ℝ := sorry

/-- Checks if a configuration has two points at opposite poles and three equidistant points on the equator -/
def isOptimalConfiguration (c : Configuration) : Prop := sorry

theorem max_min_distance_on_sphere :
  ∀ c : Configuration, (∀ i, isOnUnitSphere (c i)) →
  minDistance c ≤ Real.sqrt 2 ∧
  (minDistance c = Real.sqrt 2 ↔ isOptimalConfiguration c) := by sorry

end NUMINAMATH_CALUDE_max_min_distance_on_sphere_l2992_299279


namespace NUMINAMATH_CALUDE_max_vertex_sum_l2992_299214

/-- Represents a face of the dice -/
structure Face where
  value : Nat
  deriving Repr

/-- Represents a cubical dice -/
structure Dice where
  faces : List Face
  opposite_sum : Nat

/-- Defines a valid cubical dice where opposite faces sum to 8 -/
def is_valid_dice (d : Dice) : Prop :=
  d.faces.length = 6 ∧
  d.opposite_sum = 8 ∧
  ∀ (f1 f2 : Face), f1 ∈ d.faces → f2 ∈ d.faces → f1 ≠ f2 → f1.value + f2.value = d.opposite_sum

/-- Represents three faces meeting at a vertex -/
structure Vertex where
  f1 : Face
  f2 : Face
  f3 : Face

/-- Calculates the sum of face values at a vertex -/
def vertex_sum (v : Vertex) : Nat :=
  v.f1.value + v.f2.value + v.f3.value

/-- Defines a valid vertex of the dice -/
def is_valid_vertex (d : Dice) (v : Vertex) : Prop :=
  v.f1 ∈ d.faces ∧ v.f2 ∈ d.faces ∧ v.f3 ∈ d.faces ∧
  v.f1 ≠ v.f2 ∧ v.f1 ≠ v.f3 ∧ v.f2 ≠ v.f3

theorem max_vertex_sum (d : Dice) (h : is_valid_dice d) :
  ∀ (v : Vertex), is_valid_vertex d v → vertex_sum v ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l2992_299214


namespace NUMINAMATH_CALUDE_factor_expression_l2992_299285

theorem factor_expression (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a*b^3 + a*c^3 + a*b*c^2 + b^2*c^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2992_299285


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2992_299218

theorem trigonometric_identity : 
  Real.sin (135 * π / 180) * Real.cos (-15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2992_299218


namespace NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l2992_299260

/-- Given a quadrilateral EFGH where ∠E = 2∠F = 3∠G = 6∠H, prove that ∠E = 180° -/
theorem angle_measure_in_special_quadrilateral (E F G H : ℝ) : 
  E + F + G + H = 360 → -- sum of angles in a quadrilateral
  E = 2 * F →           -- given condition
  E = 3 * G →           -- given condition
  E = 6 * H →           -- given condition
  E = 180 :=             -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_quadrilateral_l2992_299260


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l2992_299224

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 0 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l2992_299224


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2992_299239

theorem age_ratio_problem (sam sue kendra : ℕ) : 
  kendra = 3 * sam →
  kendra = 18 →
  (sam + 3) + (sue + 3) + (kendra + 3) = 36 →
  sam / sue = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2992_299239


namespace NUMINAMATH_CALUDE_inequality_proof_l2992_299259

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ b*c/(b+c) + c*a/(c+a) + a*b/(a+b) + (1/2) * (b*c/a + c*a/b + a*b/c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2992_299259


namespace NUMINAMATH_CALUDE_local_tax_deduction_l2992_299296

-- Define Alicia's hourly wage in dollars
def hourly_wage : ℝ := 25

-- Define the tax rate as a percentage
def tax_rate : ℝ := 2

-- Define the conversion rate from dollars to cents
def dollars_to_cents : ℝ := 100

-- Theorem statement
theorem local_tax_deduction :
  (hourly_wage * dollars_to_cents) * (tax_rate / 100) = 50 := by
  sorry

end NUMINAMATH_CALUDE_local_tax_deduction_l2992_299296


namespace NUMINAMATH_CALUDE_overlap_area_is_one_l2992_299221

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its vertices -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- A triangle defined by its vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Calculate the area of overlap between a square and a triangle -/
def areaOfOverlap (s : Square) (t : Triangle) : ℝ := sorry

/-- The main theorem stating that the area of overlap is 1 square unit -/
theorem overlap_area_is_one :
  let s := Square.mk
    (Point.mk 0 0)
    (Point.mk 0 2)
    (Point.mk 2 2)
    (Point.mk 2 0)
  let t := Triangle.mk
    (Point.mk 2 2)
    (Point.mk 0 1)
    (Point.mk 1 0)
  areaOfOverlap s t = 1 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_one_l2992_299221


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l2992_299210

theorem mean_proportional_problem (x : ℝ) :
  (56.5 : ℝ) = Real.sqrt (x * 64) → x = 3192.25 / 64 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l2992_299210


namespace NUMINAMATH_CALUDE_complement_of_P_l2992_299223

-- Define the universal set R as the set of real numbers
def R : Set ℝ := Set.univ

-- Define set P
def P : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_P : 
  Set.compl P = {x : ℝ | x < 1} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l2992_299223


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2992_299255

theorem quadratic_inequality (y : ℝ) : y^2 + 7*y < 12 ↔ -9 < y ∧ y < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2992_299255


namespace NUMINAMATH_CALUDE_book_sale_profit_l2992_299216

/-- Represents the profit calculation for a book sale with and without discount -/
theorem book_sale_profit (cost_price : ℝ) (discount_percent : ℝ) (profit_with_discount_percent : ℝ) :
  discount_percent = 5 →
  profit_with_discount_percent = 23.5 →
  let selling_price_with_discount := cost_price * (1 + profit_with_discount_percent / 100 - discount_percent / 100)
  let selling_price_without_discount := selling_price_with_discount + cost_price * (discount_percent / 100)
  let profit_without_discount_percent := (selling_price_without_discount - cost_price) / cost_price * 100
  profit_without_discount_percent = 23.5 :=
by sorry

end NUMINAMATH_CALUDE_book_sale_profit_l2992_299216


namespace NUMINAMATH_CALUDE_wallet_and_purse_cost_l2992_299270

/-- The combined cost of a wallet and purse, where the wallet costs $22 and the purse costs $3 less than four times the cost of the wallet, is $107. -/
theorem wallet_and_purse_cost : 
  let wallet_cost : ℕ := 22
  let purse_cost : ℕ := 4 * wallet_cost - 3
  wallet_cost + purse_cost = 107 := by sorry

end NUMINAMATH_CALUDE_wallet_and_purse_cost_l2992_299270


namespace NUMINAMATH_CALUDE_power_function_through_point_l2992_299213

theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →  -- f is a power function
  f 2 = 8 →           -- f passes through (2,8)
  n = 3 :=            -- the power must be 3
by
  sorry


end NUMINAMATH_CALUDE_power_function_through_point_l2992_299213


namespace NUMINAMATH_CALUDE_white_lights_replacement_l2992_299212

/-- The number of white lights Malcolm had initially --/
def total_white_lights : ℕ := by sorry

/-- The number of red lights initially purchased --/
def initial_red_lights : ℕ := 16

/-- The number of yellow lights purchased --/
def yellow_lights : ℕ := 4

/-- The number of blue lights initially purchased --/
def initial_blue_lights : ℕ := 2 * yellow_lights

/-- The number of green lights purchased --/
def green_lights : ℕ := 8

/-- The number of purple lights purchased --/
def purple_lights : ℕ := 3

/-- The additional number of red lights needed --/
def additional_red_lights : ℕ := 10

/-- The additional number of blue lights needed --/
def additional_blue_lights : ℕ := initial_blue_lights / 4

theorem white_lights_replacement :
  total_white_lights = 
    initial_red_lights + additional_red_lights +
    yellow_lights +
    initial_blue_lights + additional_blue_lights +
    green_lights +
    purple_lights := by sorry

end NUMINAMATH_CALUDE_white_lights_replacement_l2992_299212


namespace NUMINAMATH_CALUDE_business_investment_l2992_299203

theorem business_investment (A B C : ℕ) (total_profit : ℚ) :
  A = 6000 →
  C = 10000 →
  B * total_profit / (A + B + C) = 1000 →
  C * total_profit / (A + B + C) - A * total_profit / (A + B + C) = 500 →
  B = 8000 := by
sorry

end NUMINAMATH_CALUDE_business_investment_l2992_299203


namespace NUMINAMATH_CALUDE_divisibility_condition_l2992_299238

theorem divisibility_condition (x y z k : ℤ) :
  (∃ q : ℤ, x^3 + y^3 + z^3 + k*x*y*z = (x + y + z) * q) ↔ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2992_299238


namespace NUMINAMATH_CALUDE_larger_bill_value_l2992_299229

/-- Proves that the value of the larger denomination bill is $10 given the problem conditions --/
theorem larger_bill_value (total_bills : ℕ) (total_value : ℕ) (five_dollar_bills : ℕ) (larger_bills : ℕ) :
  total_bills = 5 + larger_bills →
  total_bills = 12 →
  five_dollar_bills = 4 →
  larger_bills = 8 →
  total_value = 100 →
  total_value = 5 * five_dollar_bills + larger_bills * 10 :=
by sorry

end NUMINAMATH_CALUDE_larger_bill_value_l2992_299229


namespace NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_two_root_implies_abs_z_eq_two_sqrt_two_l2992_299200

/-- Given a complex number z = (a^2 - 5a + 6) + (a - 3)i where a ∈ ℝ -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 5*a + 6) (a - 3)

/-- Part 1: If z is purely imaginary, then a = 2 -/
theorem purely_imaginary_implies_a_eq_two (a : ℝ) :
  z a = Complex.I * Complex.im (z a) → a = 2 := by sorry

/-- Part 2: If z is a root of the equation x^2 - 4x + 8 = 0, then |z| = 2√2 -/
theorem root_implies_abs_z_eq_two_sqrt_two (a : ℝ) :
  (z a)^2 - 4*(z a) + 8 = 0 → Complex.abs (z a) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_implies_a_eq_two_root_implies_abs_z_eq_two_sqrt_two_l2992_299200


namespace NUMINAMATH_CALUDE_lights_after_2011_toggles_l2992_299211

/-- Represents the state of a light (on or off) -/
inductive LightState
| On : LightState
| Off : LightState

/-- Represents a row of 7 lights -/
def LightRow := Fin 7 → LightState

def initialState : LightRow := fun i =>
  if i = 0 ∨ i = 2 ∨ i = 4 ∨ i = 6 then LightState.On else LightState.Off

def toggleLights : LightRow → LightRow := sorry

theorem lights_after_2011_toggles (initialState : LightRow) 
  (h1 : ∀ state, (toggleLights^[14]) state = state)
  (h2 : (toggleLights^[9]) initialState 0 = LightState.On ∧ 
        (toggleLights^[9]) initialState 3 = LightState.On ∧ 
        (toggleLights^[9]) initialState 5 = LightState.On) :
  (toggleLights^[2011]) initialState 0 = LightState.On ∧
  (toggleLights^[2011]) initialState 3 = LightState.On ∧
  (toggleLights^[2011]) initialState 5 = LightState.On :=
sorry

end NUMINAMATH_CALUDE_lights_after_2011_toggles_l2992_299211


namespace NUMINAMATH_CALUDE_sin4_tan2_product_positive_l2992_299283

theorem sin4_tan2_product_positive :
  ∀ (sin4 tan2 : ℝ), sin4 < 0 → tan2 < 0 → sin4 * tan2 > 0 := by sorry

end NUMINAMATH_CALUDE_sin4_tan2_product_positive_l2992_299283


namespace NUMINAMATH_CALUDE_set1_equivalence_set2_equivalence_set3_equivalence_set4_equivalence_l2992_299253

-- Define the sets of points for each condition
def set1 : Set (ℝ × ℝ) := {p | p.1 ≥ -2}
def set2 : Set (ℝ × ℝ) := {p | -2 < p.1 ∧ p.1 < 2}
def set3 : Set (ℝ × ℝ) := {p | |p.1| < 2}
def set4 : Set (ℝ × ℝ) := {p | |p.1| ≥ 2}

-- State the theorems to be proved
theorem set1_equivalence : set1 = {p : ℝ × ℝ | p.1 ≥ -2} := by sorry

theorem set2_equivalence : set2 = {p : ℝ × ℝ | -2 < p.1 ∧ p.1 < 2} := by sorry

theorem set3_equivalence : set3 = {p : ℝ × ℝ | -2 < p.1 ∧ p.1 < 2} := by sorry

theorem set4_equivalence : set4 = {p : ℝ × ℝ | p.1 ≤ -2 ∨ p.1 ≥ 2} := by sorry

end NUMINAMATH_CALUDE_set1_equivalence_set2_equivalence_set3_equivalence_set4_equivalence_l2992_299253


namespace NUMINAMATH_CALUDE_fraction_equality_l2992_299243

theorem fraction_equality (a b : ℝ) (h : a / b = 5 / 2) : (a + 2*b) / (a - b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2992_299243


namespace NUMINAMATH_CALUDE_mary_found_47_shells_l2992_299290

/-- The number of seashells Sam found -/
def sam_shells : ℕ := 18

/-- The total number of seashells Sam and Mary found together -/
def total_shells : ℕ := 65

/-- The number of seashells Mary found -/
def mary_shells : ℕ := total_shells - sam_shells

theorem mary_found_47_shells : mary_shells = 47 := by
  sorry

end NUMINAMATH_CALUDE_mary_found_47_shells_l2992_299290


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l2992_299208

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + 2*x + 2 > 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l2992_299208


namespace NUMINAMATH_CALUDE_cost_per_credit_l2992_299205

/-- Calculates the cost per credit given college expenses -/
theorem cost_per_credit
  (total_credits : ℕ)
  (cost_per_textbook : ℕ)
  (num_textbooks : ℕ)
  (facilities_fee : ℕ)
  (total_expenses : ℕ)
  (h1 : total_credits = 14)
  (h2 : cost_per_textbook = 120)
  (h3 : num_textbooks = 5)
  (h4 : facilities_fee = 200)
  (h5 : total_expenses = 7100) :
  (total_expenses - (cost_per_textbook * num_textbooks + facilities_fee)) / total_credits = 450 :=
by sorry

end NUMINAMATH_CALUDE_cost_per_credit_l2992_299205


namespace NUMINAMATH_CALUDE_angle_at_point_l2992_299202

theorem angle_at_point (x : ℝ) : 
  (170 : ℝ) + 3 * x = 360 → x = 190 / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_at_point_l2992_299202


namespace NUMINAMATH_CALUDE_tea_mixture_price_l2992_299240

/-- Given three varieties of tea mixed in a 1:1:2 ratio, with the first two varieties
    costing 126 and 135 rupees per kg respectively, and the mixture worth 152 rupees per kg,
    prove that the third variety costs 173.5 rupees per kg. -/
theorem tea_mixture_price (price1 price2 mixture_price : ℚ) 
    (h1 : price1 = 126)
    (h2 : price2 = 135)
    (h3 : mixture_price = 152) : ∃ price3 : ℚ,
  price3 = 173.5 ∧ 
  (price1 + price2 + 2 * price3) / 4 = mixture_price :=
by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l2992_299240


namespace NUMINAMATH_CALUDE_trailing_zeros_sum_factorials_l2992_299236

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The sum of factorials 60! + 120! -/
def sumFactorials : ℕ := Nat.factorial 60 + Nat.factorial 120

theorem trailing_zeros_sum_factorials :
  trailingZeros sumFactorials = 14 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_sum_factorials_l2992_299236


namespace NUMINAMATH_CALUDE_negative_m_squared_n_identity_l2992_299274

theorem negative_m_squared_n_identity (m n : ℝ) : -m^2*n - 2*m^2*n = -3*m^2*n := by
  sorry

end NUMINAMATH_CALUDE_negative_m_squared_n_identity_l2992_299274


namespace NUMINAMATH_CALUDE_max_students_is_25_l2992_299297

/-- Represents the field trip problem with given conditions --/
structure FieldTrip where
  bus_rental : ℕ
  bus_capacity : ℕ
  admission_cost : ℕ
  total_budget : ℕ

/-- Calculates the maximum number of students that can go on the field trip --/
def max_students (trip : FieldTrip) : ℕ :=
  min
    ((trip.total_budget - trip.bus_rental) / trip.admission_cost)
    trip.bus_capacity

/-- Theorem stating that the maximum number of students for the given conditions is 25 --/
theorem max_students_is_25 :
  let trip : FieldTrip := {
    bus_rental := 100,
    bus_capacity := 25,
    admission_cost := 10,
    total_budget := 350
  }
  max_students trip = 25 := by
  sorry


end NUMINAMATH_CALUDE_max_students_is_25_l2992_299297


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l2992_299275

-- Define the number of workers
def num_workers : ℕ := 8

-- Define the total number of people (workers + supervisor)
def total_people : ℕ := num_workers + 1

-- Define the initial average salary
def initial_average : ℚ := 430

-- Define the old supervisor's salary
def old_supervisor_salary : ℚ := 870

-- Define the new average salary
def new_average : ℚ := 390

-- Theorem to prove
theorem new_supervisor_salary :
  ∃ (workers_total_salary new_supervisor_salary : ℚ),
    (workers_total_salary + old_supervisor_salary) / total_people = initial_average ∧
    workers_total_salary / num_workers ≤ old_supervisor_salary ∧
    (workers_total_salary + new_supervisor_salary) / total_people = new_average ∧
    new_supervisor_salary = 510 :=
sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_l2992_299275


namespace NUMINAMATH_CALUDE_johns_purchase_cost_l2992_299235

/-- Calculates the total cost of John's metal purchase in USD -/
def total_cost (silver_oz : ℝ) (gold_oz : ℝ) (platinum_oz : ℝ) 
                (silver_price_usd : ℝ) (gold_multiplier : ℝ) 
                (platinum_price_gbp : ℝ) (usd_gbp_rate : ℝ) : ℝ :=
  let silver_cost := silver_oz * silver_price_usd
  let gold_cost := gold_oz * (silver_price_usd * gold_multiplier)
  let platinum_cost := platinum_oz * (platinum_price_gbp * usd_gbp_rate)
  silver_cost + gold_cost + platinum_cost

/-- Theorem stating that John's total cost is $5780.5 -/
theorem johns_purchase_cost : 
  total_cost 2.5 3.5 4.5 25 60 80 1.3 = 5780.5 := by
  sorry

end NUMINAMATH_CALUDE_johns_purchase_cost_l2992_299235


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2992_299263

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- The first and seventh terms are roots of x^2 - 10x + 16 = 0 -/
def RootsProperty (a : ℕ → ℝ) : Prop :=
  a 1 ^ 2 - 10 * a 1 + 16 = 0 ∧ a 7 ^ 2 - 10 * a 7 + 16 = 0

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : RootsProperty a) : 
  a 2 + a 4 + a 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2992_299263


namespace NUMINAMATH_CALUDE_exponent_calculation_l2992_299286

theorem exponent_calculation : (-1 : ℝ)^53 + 3^(2^3 + 5^2 - 7^2) = -1 + (1 : ℝ) / 3^16 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l2992_299286
