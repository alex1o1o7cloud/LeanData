import Mathlib

namespace NUMINAMATH_CALUDE_ordering_of_powers_l123_12330

theorem ordering_of_powers : 5^15 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_powers_l123_12330


namespace NUMINAMATH_CALUDE_total_leaves_eq_696_l123_12398

def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6
def cilantro_pots : ℕ := 7
def lavender_pots : ℕ := 4

def basil_leaves_per_plant : ℕ := 4
def rosemary_leaves_per_plant : ℕ := 18
def thyme_leaves_per_plant : ℕ := 30
def cilantro_leaves_per_plant : ℕ := 42
def lavender_leaves_per_plant : ℕ := 12

def total_leaves : ℕ := 
  basil_pots * basil_leaves_per_plant +
  rosemary_pots * rosemary_leaves_per_plant +
  thyme_pots * thyme_leaves_per_plant +
  cilantro_pots * cilantro_leaves_per_plant +
  lavender_pots * lavender_leaves_per_plant

theorem total_leaves_eq_696 : total_leaves = 696 := by
  sorry

end NUMINAMATH_CALUDE_total_leaves_eq_696_l123_12398


namespace NUMINAMATH_CALUDE_percentage_of_1000_l123_12382

theorem percentage_of_1000 : (66.2 / 1000) * 100 = 6.62 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_1000_l123_12382


namespace NUMINAMATH_CALUDE_jude_current_age_l123_12347

/-- Heath's age today -/
def heath_age_today : ℕ := 16

/-- The number of years in the future when the age comparison is made -/
def years_in_future : ℕ := 5

/-- Heath's age in the future -/
def heath_age_future : ℕ := heath_age_today + years_in_future

/-- Jude's age in the future -/
def jude_age_future : ℕ := heath_age_future / 3

/-- Jude's age today -/
def jude_age_today : ℕ := jude_age_future - years_in_future

theorem jude_current_age : jude_age_today = 2 := by
  sorry

end NUMINAMATH_CALUDE_jude_current_age_l123_12347


namespace NUMINAMATH_CALUDE_pi_approximation_l123_12379

theorem pi_approximation (S : ℝ) (h : S > 0) :
  4 * S = (1 + 1/4) * (π * S) → π = 3 := by
sorry

end NUMINAMATH_CALUDE_pi_approximation_l123_12379


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l123_12375

/-- Given a line segment from (1, 3) to (-7, y) with length 12 and y > 0, prove y = 3 + 4√5 -/
theorem line_segment_endpoint (y : ℝ) (h1 : y > 0) 
  (h2 : Real.sqrt (((-7) - 1)^2 + (y - 3)^2) = 12) : y = 3 + 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l123_12375


namespace NUMINAMATH_CALUDE_work_left_after_14_days_l123_12316

/-- The fraction of work left for the first task after 14 days -/
def first_task_left : ℚ := 11/60

/-- The fraction of work left for the second task after 14 days -/
def second_task_left : ℚ := 0

/-- A's work rate per day -/
def rate_A : ℚ := 1/15

/-- B's work rate per day -/
def rate_B : ℚ := 1/20

/-- C's work rate per day -/
def rate_C : ℚ := 1/25

/-- The number of days A and B work on the first task -/
def days_first_task : ℕ := 7

/-- The total number of days -/
def total_days : ℕ := 14

theorem work_left_after_14_days :
  let work_AB_7_days := (rate_A + rate_B) * days_first_task
  let work_C_7_days := rate_C * days_first_task
  let work_ABC_7_days := (rate_A + rate_B + rate_C) * (total_days - days_first_task)
  (1 - work_AB_7_days = first_task_left) ∧
  (max 0 (1 - work_C_7_days - work_ABC_7_days) = second_task_left) := by
  sorry

end NUMINAMATH_CALUDE_work_left_after_14_days_l123_12316


namespace NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l123_12323

theorem complex_cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 3)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) = 81/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l123_12323


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_l123_12311

theorem sum_of_prime_factors (n : ℕ) : 
  n > 0 ∧ n < 1000 ∧ (∃ k : ℤ, 42 * n = 180 * k) →
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (n + 1))) id = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_l123_12311


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l123_12349

/-- The system of equations representing two lines -/
def line_system (x y : ℚ) : Prop :=
  2 * y = -x + 3 ∧ -y = 5 * x + 1

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (-5/9, 16/9)

/-- Theorem stating that the intersection point is the unique solution to the system of equations -/
theorem intersection_point_is_unique_solution :
  line_system intersection_point.1 intersection_point.2 ∧
  ∀ x y, line_system x y → (x, y) = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_solution_l123_12349


namespace NUMINAMATH_CALUDE_total_cases_after_three_days_l123_12387

-- Define the parameters
def initial_cases : ℕ := 2000
def increase_rate : ℚ := 20 / 100
def recovery_rate : ℚ := 2 / 100
def days : ℕ := 3

-- Function to calculate the cases for the next day
def next_day_cases (current_cases : ℚ) : ℚ :=
  current_cases + current_cases * increase_rate - current_cases * recovery_rate

-- Function to calculate cases after n days
def cases_after_days (n : ℕ) : ℚ :=
  match n with
  | 0 => initial_cases
  | n + 1 => next_day_cases (cases_after_days n)

-- Theorem statement
theorem total_cases_after_three_days :
  ⌊cases_after_days days⌋ = 3286 :=
sorry

end NUMINAMATH_CALUDE_total_cases_after_three_days_l123_12387


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l123_12320

-- Define the arithmetic sequence
def arithmetic_seq (A d : ℝ) (k : ℕ) : ℝ := A + k * d

-- Define the geometric sequence
def geometric_seq (B q : ℝ) (k : ℕ) : ℝ := B * q ^ k

-- Main theorem
theorem arithmetic_geometric_sequence_property
  (a b c : ℝ) (m n p : ℕ) (A d B q : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (ha_arith : a = arithmetic_seq A d m)
  (hb_arith : b = arithmetic_seq A d n)
  (hc_arith : c = arithmetic_seq A d p)
  (ha_geom : a = geometric_seq B q m)
  (hb_geom : b = geometric_seq B q n)
  (hc_geom : c = geometric_seq B q p) :
  a ^ (b - c) * b ^ (c - a) * c ^ (a - b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l123_12320


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l123_12358

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_arithmetic_sequence (a b c : ℕ) :
  (fib a < fib b) ∧ (fib b < fib c) ∧  -- Fₐ, Fₑ, Fₒ form an increasing sequence
  (fib (a + 1) < fib (b + 1)) ∧ (fib (b + 1) < fib (c + 1)) ∧  -- Fₐ₊₁, Fₑ₊₁, Fₒ₊₁ form an increasing sequence
  (fib c - fib b = fib b - fib a) ∧  -- Arithmetic sequence condition
  (fib (c + 1) - fib (b + 1) = fib (b + 1) - fib (a + 1)) ∧  -- Arithmetic sequence condition for next terms
  (a + b + c = 3000) →  -- Sum condition
  a = 999 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l123_12358


namespace NUMINAMATH_CALUDE_exam_students_count_l123_12340

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 0 →
    T = 80 * N →
    (T - 350) / (N - 5 : ℝ) = 90 →
    N = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l123_12340


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l123_12392

theorem trigonometric_equation_solutions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc 0 (Real.pi / 2) ∧ y ∈ Set.Icc 0 (Real.pi / 2) ∧ 
   Real.cos (2 * x) + Real.sqrt 3 * Real.sin (2 * x) = a + 1 ∧
   Real.cos (2 * y) + Real.sqrt 3 * Real.sin (2 * y) = a + 1) →
  0 ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l123_12392


namespace NUMINAMATH_CALUDE_only_negative_number_l123_12376

def is_negative (x : ℝ) : Prop := x < 0

theorem only_negative_number (a b c d : ℝ) 
  (ha : a = -2) (hb : b = 0) (hc : c = 1) (hd : d = 3) : 
  is_negative a ∧ ¬is_negative b ∧ ¬is_negative c ∧ ¬is_negative d :=
sorry

end NUMINAMATH_CALUDE_only_negative_number_l123_12376


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l123_12313

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) (h : r = 3 / 2) : 
  (π * r^2) / 2 = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l123_12313


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l123_12367

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l123_12367


namespace NUMINAMATH_CALUDE_language_class_probability_l123_12396

/-- The probability of selecting two students covering both German and Japanese classes -/
theorem language_class_probability (total : ℕ) (german : ℕ) (japanese : ℕ) 
  (h_total : total = 30)
  (h_german : german = 20)
  (h_japanese : japanese = 24)
  : ℚ := by
  sorry

end NUMINAMATH_CALUDE_language_class_probability_l123_12396


namespace NUMINAMATH_CALUDE_cube_side_length_l123_12388

theorem cube_side_length (volume : ℝ) (side : ℝ) : 
  volume = 729 → side^3 = volume → side = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l123_12388


namespace NUMINAMATH_CALUDE_inequality_range_l123_12389

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x + 2| > a^2 + a + 1) → 
  a > -2 ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l123_12389


namespace NUMINAMATH_CALUDE_triangle_ratio_l123_12312

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  A = π / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l123_12312


namespace NUMINAMATH_CALUDE_square_area_not_correlation_l123_12351

/-- A relationship between two variables -/
structure Relationship (α β : Type) where
  relate : α → β → Prop

/-- A correlation is a relationship that is not deterministic -/
def IsCorrelation {α β : Type} (r : Relationship α β) : Prop :=
  ∃ (x : α) (y₁ y₂ : β), y₁ ≠ y₂ ∧ r.relate x y₁ ∧ r.relate x y₂

/-- The relationship between a square's side length and its area -/
def SquareAreaRelationship : Relationship ℝ ℝ :=
  { relate := λ side area => area = side ^ 2 }

/-- Theorem: The relationship between a square's side length and its area is not a correlation -/
theorem square_area_not_correlation : ¬ IsCorrelation SquareAreaRelationship := by
  sorry

end NUMINAMATH_CALUDE_square_area_not_correlation_l123_12351


namespace NUMINAMATH_CALUDE_complex_equation_solution_l123_12348

theorem complex_equation_solution (m : ℝ) : 
  let z₁ : ℂ := m^2 - 3*m + m^2*Complex.I
  let z₂ : ℂ := 4 + (5*m + 6)*Complex.I
  z₁ - z₂ = 0 → m = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l123_12348


namespace NUMINAMATH_CALUDE_fort_sixty_percent_complete_l123_12307

/-- Calculates the percentage of fort completion given the required sticks, 
    sticks collected per week, and number of weeks collecting. -/
def fort_completion_percentage 
  (required_sticks : ℕ) 
  (sticks_per_week : ℕ) 
  (weeks_collecting : ℕ) : ℚ :=
  (sticks_per_week * weeks_collecting : ℚ) / required_sticks * 100

/-- Theorem stating that given the specific conditions, 
    the fort completion percentage is 60%. -/
theorem fort_sixty_percent_complete : 
  fort_completion_percentage 400 3 80 = 60 := by
  sorry

end NUMINAMATH_CALUDE_fort_sixty_percent_complete_l123_12307


namespace NUMINAMATH_CALUDE_worker_a_alone_time_l123_12301

/-- Represents the efficiency of a worker -/
structure WorkerEfficiency where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a job to be completed -/
structure Job where
  total_work : ℝ
  total_work_pos : total_work > 0

theorem worker_a_alone_time 
  (job : Job) 
  (a b : WorkerEfficiency) 
  (h1 : a.rate = 2 * b.rate) 
  (h2 : job.total_work / (a.rate + b.rate) = 20) : 
  job.total_work / a.rate = 30 := by
  sorry

end NUMINAMATH_CALUDE_worker_a_alone_time_l123_12301


namespace NUMINAMATH_CALUDE_power_calculation_l123_12353

theorem power_calculation : (16^4 * 8^6) / 4^14 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l123_12353


namespace NUMINAMATH_CALUDE_special_sequence_has_large_number_l123_12343

/-- A sequence of natural numbers with the given properties -/
def SpecialSequence (seq : Fin 20 → ℕ) : Prop :=
  (∀ i, seq i ≠ seq (i + 1)) ∧  -- distinct numbers
  (∀ i < 19, ∃ k : ℕ, seq i * seq (i + 1) = k * k) ∧  -- product is perfect square
  seq 0 = 42  -- first number is 42

theorem special_sequence_has_large_number (seq : Fin 20 → ℕ) 
  (h : SpecialSequence seq) : 
  ∃ i, seq i > 16000 := by
sorry

end NUMINAMATH_CALUDE_special_sequence_has_large_number_l123_12343


namespace NUMINAMATH_CALUDE_smallest_yummy_number_l123_12383

/-- Definition of a yummy number -/
def is_yummy (A : ℕ) : Prop :=
  ∃ n : ℕ+, n * (2 * A + n - 1) = 2 * 2023

/-- Theorem stating that 1011 is the smallest yummy number -/
theorem smallest_yummy_number :
  is_yummy 1011 ∧ ∀ A : ℕ, A < 1011 → ¬is_yummy A :=
sorry

end NUMINAMATH_CALUDE_smallest_yummy_number_l123_12383


namespace NUMINAMATH_CALUDE_gcd_factorial_plus_two_l123_12345

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_plus_two : 
  Nat.gcd (factorial 6 + 2) (factorial 8 + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_plus_two_l123_12345


namespace NUMINAMATH_CALUDE_sum_expression_l123_12336

-- Define the variables
variable (x y z : ℝ)

-- State the theorem
theorem sum_expression (h1 : y = 3 * x + 1) (h2 : z = y - x) : 
  x + y + z = 6 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_expression_l123_12336


namespace NUMINAMATH_CALUDE_ribbon_length_proof_l123_12318

/-- Calculates the total length of a ribbon before division, given the number of students,
    length per student, and leftover length. -/
def totalRibbonLength (numStudents : ℕ) (lengthPerStudent : ℝ) (leftover : ℝ) : ℝ :=
  (numStudents : ℝ) * lengthPerStudent + leftover

/-- Proves that for 10 students, 0.84 meters per student, and 0.50 meters leftover,
    the total ribbon length before division was 8.9 meters. -/
theorem ribbon_length_proof :
  totalRibbonLength 10 0.84 0.50 = 8.9 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_length_proof_l123_12318


namespace NUMINAMATH_CALUDE_division_remainder_problem_l123_12338

theorem division_remainder_problem : ∃ (A : ℕ), 17 = 5 * 3 + A ∧ A < 5 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l123_12338


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l123_12308

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l123_12308


namespace NUMINAMATH_CALUDE_light_bulb_cost_exceeds_budget_l123_12341

/-- Represents the cost of light bulbs for Valerie's lamps --/
def light_bulb_cost : ℝ :=
  let small_cost : ℝ := 3 * 8.50
  let large_cost : ℝ := 1 * 14.25
  let medium_cost : ℝ := 2 * 10.75
  let extra_small_cost : ℝ := 4 * 6.25
  small_cost + large_cost + medium_cost + extra_small_cost

/-- Valerie's budget for light bulbs --/
def budget : ℝ := 80

/-- Theorem stating that the total cost of light bulbs exceeds Valerie's budget --/
theorem light_bulb_cost_exceeds_budget : light_bulb_cost > budget := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_cost_exceeds_budget_l123_12341


namespace NUMINAMATH_CALUDE_norm_scale_vector_l123_12303

theorem norm_scale_vector (u : ℝ × ℝ) : ‖u‖ = 7 → ‖(5 : ℝ) • u‖ = 35 := by
  sorry

end NUMINAMATH_CALUDE_norm_scale_vector_l123_12303


namespace NUMINAMATH_CALUDE_kannon_oranges_last_night_l123_12372

/-- Represents the number of fruits Kannon ate --/
structure FruitCount where
  apples : ℕ
  bananas : ℕ
  oranges : ℕ

/-- The total number of fruits eaten over two meals --/
def totalFruits : ℕ := 39

/-- Kannon's fruit consumption for last night --/
def lastNight : FruitCount where
  apples := 3
  bananas := 1
  oranges := 4  -- This is what we want to prove

/-- Kannon's fruit consumption for today --/
def today : FruitCount where
  apples := lastNight.apples + 4
  bananas := 10 * lastNight.bananas
  oranges := 2 * (lastNight.apples + 4)

/-- The theorem to prove --/
theorem kannon_oranges_last_night :
  lastNight.oranges = 4 ∧
  lastNight.apples + lastNight.bananas + lastNight.oranges +
  today.apples + today.bananas + today.oranges = totalFruits := by
  sorry


end NUMINAMATH_CALUDE_kannon_oranges_last_night_l123_12372


namespace NUMINAMATH_CALUDE_disc_price_calculation_l123_12369

/-- The price of the other type of compact disc -/
def other_disc_price : ℝ := 10.50

theorem disc_price_calculation (total_discs : ℕ) (total_spent : ℝ) (known_price : ℝ) (known_quantity : ℕ) :
  total_discs = 10 →
  total_spent = 93 →
  known_price = 8.50 →
  known_quantity = 6 →
  other_disc_price = (total_spent - known_price * known_quantity) / (total_discs - known_quantity) :=
by
  sorry

#eval other_disc_price

end NUMINAMATH_CALUDE_disc_price_calculation_l123_12369


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l123_12350

/-- Given vectors a and b, with m > 0, n > 0, and a parallel to b, 
    the minimum value of 1/m + 8/n is 9/2 -/
theorem min_value_parallel_vectors (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a : ℝ × ℝ := (m, 1)
  let b : ℝ × ℝ := (4 - n, 2)
  (∃ k : ℝ, a = k • b) →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 1 / m' + 8 / n' ≥ 9 / 2) ∧
  (∃ m' n' : ℝ, m' > 0 ∧ n' > 0 ∧ 1 / m' + 8 / n' = 9 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l123_12350


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l123_12324

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ,
  x^6 - 2*x^5 + x^4 - x^2 - 2*x + 1 = 
  ((x^2 - 1) * (x - 2) * (x + 2)) * q + (2*x^3 - 9*x^2 + 3*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l123_12324


namespace NUMINAMATH_CALUDE_jimmy_stair_time_l123_12394

/-- The sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing time -/
theorem jimmy_stair_time : arithmeticSum 20 7 7 = 287 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_time_l123_12394


namespace NUMINAMATH_CALUDE_dice_probability_l123_12327

def standard_dice : ℕ := 5
def special_dice : ℕ := 5
def standard_sides : ℕ := 6
def special_sides : ℕ := 3  -- Only even numbers (2, 4, 6)

def probability_standard_one : ℚ := 1 / 6
def probability_standard_not_one : ℚ := 5 / 6
def probability_special_four : ℚ := 1 / 3
def probability_special_not_four : ℚ := 2 / 3

theorem dice_probability : 
  (Nat.choose standard_dice 1 : ℚ) * probability_standard_one * probability_standard_not_one ^ 4 *
  (Nat.choose special_dice 1 : ℚ) * probability_special_four * probability_special_not_four ^ 4 =
  250000 / 1889568 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l123_12327


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l123_12329

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) -- The geometric sequence
  (S : ℕ → ℝ) -- The sum function
  (h_geom : ∀ n, a (n + 1) = a n * (a 1 / a 0)) -- Condition for geometric sequence
  (h_sum : ∀ n, S n = (a 0) * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))) -- Sum formula
  (h_eq : 8 * S 6 = 7 * S 3) -- Given equation
  : a 1 / a 0 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l123_12329


namespace NUMINAMATH_CALUDE_maze_navigation_ways_l123_12302

/-- Converts a list of digits in base 6 to a number in base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The number of ways the dog can navigate through the maze in base 6 -/
def mazeWaysBase6 : List Nat := [4, 1, 2, 5]

/-- Theorem: The number of ways the dog can navigate through the maze
    is 1162 when converted from base 6 to base 10 -/
theorem maze_navigation_ways :
  base6ToBase10 mazeWaysBase6 = 1162 := by
  sorry

end NUMINAMATH_CALUDE_maze_navigation_ways_l123_12302


namespace NUMINAMATH_CALUDE_inequality_proof_l123_12360

theorem inequality_proof (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h5 : a + b + c + d + e = 1) :
  a * d + d * c + c * b + b * e + e * a ≤ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l123_12360


namespace NUMINAMATH_CALUDE_cereal_spending_l123_12386

/-- The amount spent by Pop on cereal -/
def pop_spend : ℝ := 15

/-- The amount spent by Crackle on cereal -/
def crackle_spend : ℝ := 3 * pop_spend

/-- The amount spent by Snap on cereal -/
def snap_spend : ℝ := 2 * crackle_spend

/-- The total amount spent by Snap, Crackle, and Pop on cereal -/
def total_spend : ℝ := snap_spend + crackle_spend + pop_spend

theorem cereal_spending :
  total_spend = 150 := by sorry

end NUMINAMATH_CALUDE_cereal_spending_l123_12386


namespace NUMINAMATH_CALUDE_sum_even_not_square_or_cube_l123_12366

theorem sum_even_not_square_or_cube (n : ℕ+) :
  ∀ k m : ℕ+, (n : ℕ) * (n + 1) ≠ k ^ 2 ∧ (n : ℕ) * (n + 1) ≠ m ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_not_square_or_cube_l123_12366


namespace NUMINAMATH_CALUDE_greatest_sum_on_circle_l123_12364

theorem greatest_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_on_circle_l123_12364


namespace NUMINAMATH_CALUDE_bird_speed_theorem_l123_12395

theorem bird_speed_theorem (d t : ℝ) 
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  d / t = 48 := by
  sorry

end NUMINAMATH_CALUDE_bird_speed_theorem_l123_12395


namespace NUMINAMATH_CALUDE_length_of_AF_l123_12309

/-- Given a plot ABCD with known dimensions, prove the length of AF --/
theorem length_of_AF (CE ED AE : ℝ) (area_ABCD : ℝ) 
  (h1 : CE = 40)
  (h2 : ED = 50)
  (h3 : AE = 120)
  (h4 : area_ABCD = 7200) :
  ∃ AF : ℝ, AF = 128 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AF_l123_12309


namespace NUMINAMATH_CALUDE_hcd_7350_165_minus_15_l123_12361

theorem hcd_7350_165_minus_15 : Nat.gcd 7350 165 - 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_hcd_7350_165_minus_15_l123_12361


namespace NUMINAMATH_CALUDE_first_line_time_l123_12374

/-- Represents the productivity of a production line -/
structure ProductivityRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a production line -/
structure ProductionLine where
  productivity : ProductivityRate

/-- Represents a system of three production lines -/
structure ProductionSystem where
  line1 : ProductionLine
  line2 : ProductionLine
  line3 : ProductionLine
  combined_productivity : ProductivityRate
  first_second_productivity : ProductivityRate
  combined_vs_first_second : combined_productivity.rate = 1.5 * first_second_productivity.rate
  second_faster_than_first : line2.productivity.rate = line1.productivity.rate + (1 / 2)
  second_third_vs_first : 
    1 / line1.productivity.rate - (24 / 5) = 
    1 / (line2.productivity.rate + line3.productivity.rate)

theorem first_line_time (system : ProductionSystem) : 
  1 / system.line1.productivity.rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_line_time_l123_12374


namespace NUMINAMATH_CALUDE_total_cost_calculation_l123_12368

def coffee_maker_price : ℝ := 70
def blender_price : ℝ := 100
def coffee_maker_discount : ℝ := 0.20
def blender_discount : ℝ := 0.15
def sales_tax_rate : ℝ := 0.08
def extended_warranty_cost : ℝ := 25
def shipping_fee : ℝ := 12

def total_cost : ℝ :=
  let discounted_coffee_maker := coffee_maker_price * (1 - coffee_maker_discount)
  let discounted_blender := blender_price * (1 - blender_discount)
  let subtotal := 2 * discounted_coffee_maker + discounted_blender
  let sales_tax := subtotal * sales_tax_rate
  subtotal + sales_tax + extended_warranty_cost + shipping_fee

theorem total_cost_calculation :
  total_cost = 249.76 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l123_12368


namespace NUMINAMATH_CALUDE_certain_percentage_problem_l123_12380

theorem certain_percentage_problem (x : ℝ) : x = 12 → (x / 100) * 24.2 = 0.1 * 14.2 + 1.484 := by
  sorry

end NUMINAMATH_CALUDE_certain_percentage_problem_l123_12380


namespace NUMINAMATH_CALUDE_contractor_payment_l123_12370

/-- Calculates the total amount a contractor receives given the contract terms and attendance. -/
def contractorPay (totalDays : ℕ) (payPerDay : ℚ) (finePerDay : ℚ) (absentDays : ℕ) : ℚ :=
  let workDays := totalDays - absentDays
  let totalPay := (workDays : ℚ) * payPerDay
  let totalFine := (absentDays : ℚ) * finePerDay
  totalPay - totalFine

/-- Proves that under the given conditions, the contractor receives Rs. 425. -/
theorem contractor_payment :
  contractorPay 30 25 7.50 10 = 425 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_l123_12370


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l123_12354

theorem cubic_roots_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2018*x + n = (x - p) * (x - q) * (x - r)) →
  |p| + |q| + |r| = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l123_12354


namespace NUMINAMATH_CALUDE_selection_theorem_l123_12352

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_people : ℕ := num_boys + num_girls
def num_to_select : ℕ := 4

theorem selection_theorem : 
  (Nat.choose total_people num_to_select) - (Nat.choose num_boys num_to_select) = 34 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l123_12352


namespace NUMINAMATH_CALUDE_sam_age_two_years_ago_l123_12381

def john_age (sam_age : ℕ) : ℕ := 3 * sam_age

theorem sam_age_two_years_ago (sam_current_age : ℕ) : 
  john_age sam_current_age = 3 * sam_current_age ∧ 
  john_age sam_current_age + 9 = 2 * (sam_current_age + 9) →
  sam_current_age - 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_sam_age_two_years_ago_l123_12381


namespace NUMINAMATH_CALUDE_min_value_of_rounded_sum_l123_12334

-- Define the rounding functions
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  (roundToNearest (x * 10)) / 10

-- Define the main theorem
theorem min_value_of_rounded_sum (a b : ℝ) 
  (h1 : roundToNearestTenth a + roundToNearest b = 98.6)
  (h2 : roundToNearest a + roundToNearestTenth b = 99.3) :
  roundToNearest (10 * (a + b)) ≥ 988 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_rounded_sum_l123_12334


namespace NUMINAMATH_CALUDE_factorial_square_root_l123_12397

theorem factorial_square_root (n : ℕ) (h : n = 5) : 
  Real.sqrt (n.factorial * (n.factorial ^ 2)) = 240 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_l123_12397


namespace NUMINAMATH_CALUDE_simplify_expression_l123_12377

theorem simplify_expression (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x^2 - 4)) = (x + 2) / (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l123_12377


namespace NUMINAMATH_CALUDE_apple_savings_proof_l123_12344

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

end NUMINAMATH_CALUDE_apple_savings_proof_l123_12344


namespace NUMINAMATH_CALUDE_product_equals_result_l123_12346

theorem product_equals_result : 582964 * 99999 = 58295817036 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_result_l123_12346


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l123_12314

/-- Definition of a hyperbola with foci and points -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The conditions of the problem -/
def hyperbola_conditions (Γ : Hyperbola) : Prop :=
  Γ.a > 0 ∧ Γ.b > 0 ∧
  (Γ.C.2 = 0) ∧  -- C is on x-axis
  (Γ.C.1 - Γ.B.1, Γ.C.2 - Γ.B.2) = 3 • (Γ.F₂.1 - Γ.A.1, Γ.F₂.2 - Γ.A.2) ∧  -- CB = 3F₂A
  (∃ t : ℝ, t > 0 ∧ Γ.B.1 - Γ.F₂.1 = t * (Γ.F₁.1 - Γ.C.1) ∧ Γ.B.2 - Γ.F₂.2 = t * (Γ.F₁.2 - Γ.C.2))  -- BF₂ bisects ∠F₁BC

/-- The theorem to be proved -/
theorem hyperbola_eccentricity (Γ : Hyperbola) :
  hyperbola_conditions Γ → (Real.sqrt ((Γ.F₁.1 - Γ.F₂.1)^2 + (Γ.F₁.2 - Γ.F₂.2)^2) / (2 * Γ.a) = Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l123_12314


namespace NUMINAMATH_CALUDE_fraction_value_l123_12357

theorem fraction_value (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 * c / (a + b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l123_12357


namespace NUMINAMATH_CALUDE_price_change_equivalence_l123_12390

theorem price_change_equivalence (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0)
  (h2 : x > 0 ∧ x < 100) :
  (1.25 * initial_price) * (1 - x / 100) = 1.125 * initial_price → x = 10 := by
sorry

end NUMINAMATH_CALUDE_price_change_equivalence_l123_12390


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_range_of_a_when_A_subset_B_l123_12325

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {x : ℝ | 0 < 2*x + a ∧ 2*x + a ≤ 3}

/-- Definition of set B -/
def B : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 2}

/-- Theorem for the intersection of A and B when a = -1 -/
theorem intersection_A_B_when_a_neg_one :
  A (-1) ∩ B = {x : ℝ | 1/2 < x ∧ x < 2} := by sorry

/-- Theorem for the range of a when A is a subset of B -/
theorem range_of_a_when_A_subset_B :
  ∀ a : ℝ, A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_neg_one_range_of_a_when_A_subset_B_l123_12325


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l123_12363

/-- The area of a square with adjacent vertices at (1, 3) and (5, -1) is 32 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (5, -1)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l123_12363


namespace NUMINAMATH_CALUDE_square_difference_l123_12328

theorem square_difference (a b : ℝ) (h1 : a * b = 2) (h2 : a + b = 3) :
  (a - b)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_square_difference_l123_12328


namespace NUMINAMATH_CALUDE_one_correct_proposition_l123_12319

theorem one_correct_proposition : 
  (∃! n : Nat, n = 1 ∧ 
    ((∀ a b : ℝ, a > abs b → a^2 > b^2) ∧ 
     ¬(∀ a b c d : ℝ, a > b ∧ c > d → a - c > b - d) ∧
     ¬(∀ a b c d : ℝ, a > b ∧ c > d → a * c > b * d) ∧
     ¬(∀ a b c : ℝ, a > b ∧ b > 0 → c / a > c / b))) :=
by sorry

end NUMINAMATH_CALUDE_one_correct_proposition_l123_12319


namespace NUMINAMATH_CALUDE_percentage_less_than_l123_12365

theorem percentage_less_than (w x y z : ℝ) 
  (hw : w = 0.60 * x) 
  (hz1 : z = 0.54 * y) 
  (hz2 : z = 1.50 * w) : 
  x = 0.60 * y := by
sorry

end NUMINAMATH_CALUDE_percentage_less_than_l123_12365


namespace NUMINAMATH_CALUDE_square_shadow_not_trapezoid_l123_12306

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define a shadow as a quadrilateral
structure Shadow where
  vertices : Fin 4 → ℝ × ℝ

-- Define a uniform light source
structure UniformLight where
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Define a trapezoid
def is_trapezoid (s : Shadow) : Prop :=
  ∃ (i j : Fin 4), i ≠ j ∧ 
    (s.vertices i).1 - (s.vertices j).1 ≠ 0 ∧
    (s.vertices ((i + 1) % 4)).1 - (s.vertices ((j + 1) % 4)).1 ≠ 0 ∧
    ((s.vertices i).2 - (s.vertices j).2) / ((s.vertices i).1 - (s.vertices j).1) =
    ((s.vertices ((i + 1) % 4)).2 - (s.vertices ((j + 1) % 4)).2) / 
    ((s.vertices ((i + 1) % 4)).1 - (s.vertices ((j + 1) % 4)).1)

-- State the theorem
theorem square_shadow_not_trapezoid 
  (square : Square) (light : UniformLight) (shadow : Shadow) :
  (∃ (projection : Square → UniformLight → Shadow), 
    projection square light = shadow) →
  ¬ is_trapezoid shadow :=
sorry

end NUMINAMATH_CALUDE_square_shadow_not_trapezoid_l123_12306


namespace NUMINAMATH_CALUDE_unique_number_of_children_l123_12317

theorem unique_number_of_children : ∃! n : ℕ, 
  100 ≤ n ∧ n ≤ 150 ∧ n % 8 = 5 ∧ n % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_unique_number_of_children_l123_12317


namespace NUMINAMATH_CALUDE_highway_vehicle_ratio_l123_12391

theorem highway_vehicle_ratio (total_vehicles : ℕ) (num_trucks : ℕ) : 
  total_vehicles = 300 → 
  num_trucks = 100 → 
  ∃ (k : ℕ), k * num_trucks = total_vehicles - num_trucks → 
  (total_vehicles - num_trucks) / num_trucks = 2 := by
  sorry

end NUMINAMATH_CALUDE_highway_vehicle_ratio_l123_12391


namespace NUMINAMATH_CALUDE_garden_width_is_15_l123_12339

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- The perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.width)

theorem garden_width_is_15 (g : RectangularGarden) 
  (h1 : g.length = 25) 
  (h2 : perimeter g = 80) : 
  g.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_garden_width_is_15_l123_12339


namespace NUMINAMATH_CALUDE_perpendicular_equivalence_l123_12342

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- Define the intersection of planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_equivalence 
  (α β : Plane) (m n l : Line) 
  (h1 : perp_plane α β) 
  (h2 : intersect α β = l) 
  (h3 : subset m α) 
  (h4 : subset n β) : 
  perp_line m n ↔ (perp_line m l ∨ perp_line n l) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_equivalence_l123_12342


namespace NUMINAMATH_CALUDE_sqrt_inequality_l123_12355

theorem sqrt_inequality (n : ℝ) (h : n ≥ 0) :
  Real.sqrt (n + 2) - Real.sqrt (n + 1) ≤ Real.sqrt (n + 1) - Real.sqrt n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l123_12355


namespace NUMINAMATH_CALUDE_sqrt_4_not_plus_minus_2_l123_12384

theorem sqrt_4_not_plus_minus_2 : ¬(Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_not_plus_minus_2_l123_12384


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_k_in_range_l123_12310

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the property of being not monotonic on an interval
def not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

-- Theorem statement
theorem f_not_monotonic_iff_k_in_range (k : ℝ) :
  not_monotonic f (k - 1) (k + 1) ↔ (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_k_in_range_l123_12310


namespace NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l123_12371

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : A → B ∧ ¬(B → A))
variable (h2 : (B ↔ C))
variable (h3 : (D → C) ∧ ¬(C → D))

-- Theorem to prove
theorem A_sufficient_not_necessary_for_D : 
  (A → D) ∧ ¬(D → A) :=
sorry

end NUMINAMATH_CALUDE_A_sufficient_not_necessary_for_D_l123_12371


namespace NUMINAMATH_CALUDE_intersection_of_sets_l123_12362

theorem intersection_of_sets (M N : Set ℝ) : 
  M = {x : ℝ | Real.sqrt (x + 1) ≥ 0} →
  N = {x : ℝ | x^2 + x - 2 < 0} →
  M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l123_12362


namespace NUMINAMATH_CALUDE_population_equal_in_14_years_l123_12305

/-- The number of years it takes for two villages' populations to be equal -/
def years_to_equal_population (initial_x initial_y decline_rate_x growth_rate_y : ℕ) : ℕ :=
  (initial_x - initial_y) / (decline_rate_x + growth_rate_y)

/-- Theorem stating that it takes 14 years for the populations to be equal -/
theorem population_equal_in_14_years :
  years_to_equal_population 70000 42000 1200 800 = 14 := by
  sorry

#eval years_to_equal_population 70000 42000 1200 800

end NUMINAMATH_CALUDE_population_equal_in_14_years_l123_12305


namespace NUMINAMATH_CALUDE_eight_members_prefer_b_first_l123_12321

/-- Represents the number of ballots for each permutation of candidates A, B, C -/
structure BallotCounts where
  abc : ℕ
  acb : ℕ
  cab : ℕ
  cba : ℕ
  bca : ℕ
  bac : ℕ

/-- The committee voting system with given conditions -/
def CommitteeVoting (counts : BallotCounts) : Prop :=
  -- Total number of ballots is 20
  counts.abc + counts.acb + counts.cab + counts.cba + counts.bca + counts.bac = 20 ∧
  -- Each permutation appears at least once
  counts.abc ≥ 1 ∧ counts.acb ≥ 1 ∧ counts.cab ≥ 1 ∧
  counts.cba ≥ 1 ∧ counts.bca ≥ 1 ∧ counts.bac ≥ 1 ∧
  -- 11 members prefer A to B
  counts.abc + counts.acb + counts.cab = 11 ∧
  -- 12 members prefer C to A
  counts.cab + counts.cba + counts.bca = 12 ∧
  -- 14 members prefer B to C
  counts.abc + counts.bca + counts.bac = 14

/-- The theorem stating that 8 members have B as their first choice -/
theorem eight_members_prefer_b_first (counts : BallotCounts) :
  CommitteeVoting counts → counts.bca + counts.bac = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_members_prefer_b_first_l123_12321


namespace NUMINAMATH_CALUDE_birthday_problem_l123_12333

/-- The number of months in the fantasy world -/
def num_months : ℕ := 10

/-- The number of people in the room -/
def num_people : ℕ := 60

/-- The largest number n such that at least n people are guaranteed to have birthdays in the same month -/
def largest_guaranteed_group : ℕ := 6

theorem birthday_problem :
  ∀ (birthday_distribution : Fin num_people → Fin num_months),
  ∃ (month : Fin num_months),
  (Finset.filter (λ person => birthday_distribution person = month) Finset.univ).card ≥ largest_guaranteed_group ∧
  ∀ n > largest_guaranteed_group,
  ∃ (bad_distribution : Fin num_people → Fin num_months),
  ∀ (month : Fin num_months),
  (Finset.filter (λ person => bad_distribution person = month) Finset.univ).card < n :=
sorry

end NUMINAMATH_CALUDE_birthday_problem_l123_12333


namespace NUMINAMATH_CALUDE_f_properties_l123_12356

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + Real.log x

theorem f_properties (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, f a x ≥ 0 → a ≤ 2 + 1/2 * Real.log 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioi 1 ∧ 
    (∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂) →
    f a x₁ - f a x₂ < -3/4 + Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l123_12356


namespace NUMINAMATH_CALUDE_parabola_intersection_fixed_points_l123_12304

/-- The parabola y^2 = 2px -/
def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}

/-- The fixed point A -/
def A (t : ℝ) : ℝ × ℝ := (t, 0)

/-- The line x = -t -/
def VerticalLine (t : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.1 = -t}

/-- The circle with diameter MN -/
def CircleMN (M N : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | (xy.1 - (M.1 + N.1) / 2)^2 + (xy.2 - (M.2 + N.2) / 2)^2 = 
    ((N.1 - M.1)^2 + (N.2 - M.2)^2) / 4}

theorem parabola_intersection_fixed_points (p t : ℝ) (hp : p > 0) (ht : t > 0) :
  ∀ (B C M N : ℝ × ℝ),
    B ∈ Parabola p → C ∈ Parabola p →
    (∃ (k : ℝ), B.1 = k * B.2 + t ∧ C.1 = k * C.2 + t) →
    M ∈ VerticalLine t → N ∈ VerticalLine t →
    (∃ (r : ℝ), M.2 = r * M.1 ∧ B.2 = r * B.1) →
    (∃ (s : ℝ), N.2 = s * N.1 ∧ C.2 = s * C.1) →
    ((-t - Real.sqrt (2 * p * t), 0) ∈ CircleMN M N) ∧
    ((-t + Real.sqrt (2 * p * t), 0) ∈ CircleMN M N) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_fixed_points_l123_12304


namespace NUMINAMATH_CALUDE_sqrt_y_fourth_power_l123_12300

theorem sqrt_y_fourth_power (y : ℝ) (h : (Real.sqrt y) ^ 4 = 256) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_y_fourth_power_l123_12300


namespace NUMINAMATH_CALUDE_fourth_guard_distance_l123_12332

/-- Represents a rectangular facility with guards -/
structure Facility :=
  (length : ℝ)
  (width : ℝ)
  (perimeter : ℝ)
  (three_guards_distance : ℝ)

/-- The theorem to prove -/
theorem fourth_guard_distance (f : Facility) 
  (h1 : f.length = 200)
  (h2 : f.width = 300)
  (h3 : f.perimeter = 2 * (f.length + f.width))
  (h4 : f.three_guards_distance = 850) :
  f.perimeter - f.three_guards_distance = 150 := by
  sorry

#check fourth_guard_distance

end NUMINAMATH_CALUDE_fourth_guard_distance_l123_12332


namespace NUMINAMATH_CALUDE_consecutive_draw_probability_l123_12399

/-- The probability of drawing one red marble and then one blue marble consecutively from a bag of marbles. -/
theorem consecutive_draw_probability
  (red : ℕ) (blue : ℕ) (green : ℕ)
  (h_red : red = 5)
  (h_blue : blue = 4)
  (h_green : green = 6)
  : (red : ℚ) / (red + blue + green) * (blue : ℚ) / (red + blue + green - 1) = 2 / 21 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_draw_probability_l123_12399


namespace NUMINAMATH_CALUDE_triangle_cosine_b_l123_12359

theorem triangle_cosine_b (ω : ℝ) (A B C a b c : ℝ) :
  ω > 0 →
  (∀ x, 2 * Real.sqrt 3 * Real.sin (ω * x / 2) * Real.cos (ω * x / 2) - 2 * Real.sin (ω * x / 2) ^ 2 =
        2 * Real.sin (2 * x / 3 + π / 6) - 1) →
  a < b →
  b < c →
  Real.sqrt 3 * a = 2 * c * Real.sin A →
  2 * Real.sin (A + π / 2) - 1 = 11 / 13 →
  Real.cos B = (5 * Real.sqrt 3 + 12) / 26 := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_b_l123_12359


namespace NUMINAMATH_CALUDE_oranges_per_box_l123_12326

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℚ) 
  (h1 : total_oranges = 72) 
  (h2 : num_boxes = 3) : 
  (total_oranges : ℚ) / num_boxes = 24 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_box_l123_12326


namespace NUMINAMATH_CALUDE_rectangular_window_width_l123_12322

/-- Represents the width of a rectangular window with specific pane arrangements and dimensions. -/
def window_width (pane_width : ℝ) : ℝ :=
  3 * pane_width + 4  -- 3 panes across plus 4 borders

/-- Theorem stating the width of the rectangular window under given conditions. -/
theorem rectangular_window_width :
  ∃ (pane_width : ℝ),
    pane_width > 0 ∧
    (3 : ℝ) / 4 * pane_width = 3 / 4 * pane_width ∧  -- height-to-width ratio of 3:4
    window_width pane_width = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_window_width_l123_12322


namespace NUMINAMATH_CALUDE_airplane_seats_l123_12393

/-- Given an airplane with a total of 387 seats, where the number of coach class seats
    is 2 more than 4 times the number of first-class seats, prove that there are
    77 first-class seats. -/
theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach : ℕ)
    (h1 : total_seats = 387)
    (h2 : coach = 4 * first_class + 2)
    (h3 : total_seats = first_class + coach) :
    first_class = 77 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l123_12393


namespace NUMINAMATH_CALUDE_books_sum_is_67_l123_12373

/-- The total number of books Sandy, Benny, and Tim have together -/
def total_books (sandy_books benny_books tim_books : ℕ) : ℕ :=
  sandy_books + benny_books + tim_books

/-- Theorem stating that the total number of books is 67 -/
theorem books_sum_is_67 :
  total_books 10 24 33 = 67 := by
  sorry

end NUMINAMATH_CALUDE_books_sum_is_67_l123_12373


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l123_12331

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (∃ a b c : ℝ, (4*x + 7)*(3*x - 5) = 15 ∧ a*x^2 + b*x + c = 0) → 
  (∃ x₁ x₂ : ℝ, (4*x₁ + 7)*(3*x₁ - 5) = 15 ∧ 
                (4*x₂ + 7)*(3*x₂ - 5) = 15 ∧ 
                x₁ + x₂ = -1/12) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l123_12331


namespace NUMINAMATH_CALUDE_function_inequality_l123_12335

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : is_periodic f 2)
  (h_monotone : is_monotone_decreasing f (-1) 0)
  (a b c : ℝ)
  (h_a : a = f (-2.8))
  (h_b : b = f (-1.6))
  (h_c : c = f 0.5) :
  a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l123_12335


namespace NUMINAMATH_CALUDE_new_boys_in_classroom_l123_12315

/-- The number of new boys that joined a classroom --/
def new_boys (initial_size : ℕ) (initial_girls_percent : ℚ) (final_girls_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of new boys that joined the classroom --/
theorem new_boys_in_classroom :
  new_boys 20 (40 / 100) (32 / 100) = 5 := by sorry

end NUMINAMATH_CALUDE_new_boys_in_classroom_l123_12315


namespace NUMINAMATH_CALUDE_tangent_line_sum_l123_12337

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the theorem
theorem tangent_line_sum (h : ∀ y, y = f 5 ↔ y = -5 + 8) : f 5 + deriv f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l123_12337


namespace NUMINAMATH_CALUDE_equation_solution_l123_12385

theorem equation_solution : ∃ x : ℝ, (45 / 75 = Real.sqrt (x / 75) + 1 / 5) ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l123_12385


namespace NUMINAMATH_CALUDE_polynomial_root_sum_product_l123_12378

theorem polynomial_root_sum_product (c d : ℂ) : 
  (c^4 - 6*c - 3 = 0) → 
  (d^4 - 6*d - 3 = 0) → 
  (c*d + c + d = 3 + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_product_l123_12378
