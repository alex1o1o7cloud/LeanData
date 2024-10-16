import Mathlib

namespace NUMINAMATH_CALUDE_next_feb29_sunday_l69_6968

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Checks if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  year % 4 == 0 && (year % 100 ≠ 0 || year % 400 == 0)

/-- Advances the day of the week by the given number of days -/
def advanceDayOfWeek (day : DayOfWeek) (days : Nat) : DayOfWeek :=
  match (day, days % 7) with
  | (DayOfWeek.Sunday, 0) => DayOfWeek.Sunday
  | (DayOfWeek.Sunday, 1) => DayOfWeek.Monday
  | (DayOfWeek.Sunday, 2) => DayOfWeek.Tuesday
  | (DayOfWeek.Sunday, 3) => DayOfWeek.Wednesday
  | (DayOfWeek.Sunday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Sunday, 5) => DayOfWeek.Friday
  | (DayOfWeek.Sunday, 6) => DayOfWeek.Saturday
  | _ => DayOfWeek.Sunday  -- Default case, should not occur

/-- Calculates the day of the week for February 29 in the given year, starting from 2004 -/
def feb29DayOfWeek (year : Nat) : DayOfWeek :=
  let daysAdvanced := (year - 2004) / 4 * 2  -- Each leap year advances by 2 days
  advanceDayOfWeek DayOfWeek.Sunday daysAdvanced

/-- Theorem: The next year after 2004 when February 29 falls on a Sunday is 2032 -/
theorem next_feb29_sunday : 
  (∀ y : Nat, 2004 < y → y < 2032 → feb29DayOfWeek y ≠ DayOfWeek.Sunday) ∧ 
  feb29DayOfWeek 2032 = DayOfWeek.Sunday :=
sorry

end NUMINAMATH_CALUDE_next_feb29_sunday_l69_6968


namespace NUMINAMATH_CALUDE_first_term_formula_l69_6965

theorem first_term_formula (p q : ℕ) (hp : p ≥ 2) (hq : q ≥ 2) :
  ∃ (rest : ℕ), p^q = (p^(q-1) - p + 1) + rest :=
sorry

end NUMINAMATH_CALUDE_first_term_formula_l69_6965


namespace NUMINAMATH_CALUDE_salary_relation_l69_6909

theorem salary_relation (A B C : ℝ) :
  A + B + C = 10000 ∧
  A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧
  0.1 * A + 0.15 * B = 0.2 * C →
  A = 20000 / 3 - 7 * B / 6 :=
by sorry

end NUMINAMATH_CALUDE_salary_relation_l69_6909


namespace NUMINAMATH_CALUDE_triangle_properties_l69_6942

/-- Given a triangle ABC with A = 2B, b = 2, and c = 4, prove the following properties -/
theorem triangle_properties (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C)
  (h_A_2B : A = 2 * B)
  (h_b : b = 2)
  (h_c : c = 4) :
  a = 2 * b * Real.cos B ∧ B = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l69_6942


namespace NUMINAMATH_CALUDE_parcel_cost_formula_l69_6935

def parcel_cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem parcel_cost_formula (W : ℕ) :
  (W ≤ 10 → parcel_cost W = 5 * W + 10) ∧
  (W > 10 → parcel_cost W = 7 * W - 10) := by
  sorry

end NUMINAMATH_CALUDE_parcel_cost_formula_l69_6935


namespace NUMINAMATH_CALUDE_minimal_fraction_sum_l69_6959

theorem minimal_fraction_sum (a b : ℕ+) (h : (45:ℚ)/110 < (a:ℚ)/(b:ℚ) ∧ (a:ℚ)/(b:ℚ) < (50:ℚ)/110) :
  (∃ (c d : ℕ+), (45:ℚ)/110 < (c:ℚ)/(d:ℚ) ∧ (c:ℚ)/(d:ℚ) < (50:ℚ)/110 ∧ c+d ≤ a+b) →
  (3:ℚ)/7 = (a:ℚ)/(b:ℚ) :=
sorry

end NUMINAMATH_CALUDE_minimal_fraction_sum_l69_6959


namespace NUMINAMATH_CALUDE_inequality_proof_l69_6973

theorem inequality_proof (a b c : ℝ) : 
  a = Real.sqrt ((1 - Real.cos (110 * π / 180)) / 2) →
  b = (Real.sqrt 2 / 2) * (Real.sin (20 * π / 180) + Real.cos (20 * π / 180)) →
  c = (1 + Real.tan (20 * π / 180)) / (1 - Real.tan (20 * π / 180)) →
  a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l69_6973


namespace NUMINAMATH_CALUDE_scribes_expenditure_change_l69_6972

/-- Proves that reducing the number of scribes by 50% and increasing the salaries
    of the remaining scribes by 50% results in a 25% decrease in total expenditure. -/
theorem scribes_expenditure_change
  (initial_allocation : ℝ)
  (n : ℕ)
  (h1 : initial_allocation > 0)
  (h2 : n > 0) :
  let reduced_scribes := n / 2
  let initial_salary := initial_allocation / n
  let new_salary := initial_salary * 1.5
  let new_expenditure := reduced_scribes * new_salary
  new_expenditure / initial_allocation = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_scribes_expenditure_change_l69_6972


namespace NUMINAMATH_CALUDE_equal_numbers_product_l69_6929

theorem equal_numbers_product (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 22 →
  c = d →
  c * d = 529 := by
sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l69_6929


namespace NUMINAMATH_CALUDE_expansion_simplification_l69_6926

theorem expansion_simplification (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * (4 / x^2 + 5 * x^3 - 2 / 3) = 3 / x^2 + 15 * x^3 / 4 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_simplification_l69_6926


namespace NUMINAMATH_CALUDE_kitty_vacuuming_time_l69_6944

/-- Represents the weekly cleaning routine for a living room -/
structure LivingRoomCleaning where
  pickup_time : ℕ
  window_time : ℕ
  dusting_time : ℕ
  total_time_4weeks : ℕ

/-- Calculates the time spent vacuuming per week -/
def vacuuming_time_per_week (cleaning : LivingRoomCleaning) : ℕ :=
  let other_tasks_time := cleaning.pickup_time + cleaning.window_time + cleaning.dusting_time
  let total_other_tasks_4weeks := other_tasks_time * 4
  let total_vacuuming_4weeks := cleaning.total_time_4weeks - total_other_tasks_4weeks
  total_vacuuming_4weeks / 4

/-- Theorem stating that Kitty spends 20 minutes vacuuming per week -/
theorem kitty_vacuuming_time (cleaning : LivingRoomCleaning)
    (h1 : cleaning.pickup_time = 5)
    (h2 : cleaning.window_time = 15)
    (h3 : cleaning.dusting_time = 10)
    (h4 : cleaning.total_time_4weeks = 200) :
    vacuuming_time_per_week cleaning = 20 := by
  sorry

end NUMINAMATH_CALUDE_kitty_vacuuming_time_l69_6944


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l69_6917

theorem trigonometric_equalities : 
  (Real.sqrt 2 / 2) * (Real.cos (15 * π / 180) - Real.sin (15 * π / 180)) = 1/2 ∧
  Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180)^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l69_6917


namespace NUMINAMATH_CALUDE_bakery_pastries_and_bagels_l69_6998

/-- Proves that the total number of pastries and bagels is 474 given the bakery conditions -/
theorem bakery_pastries_and_bagels :
  let total_items : ℕ := 720
  let bread_rolls : ℕ := 240
  let croissants : ℕ := 75
  let muffins : ℕ := 145
  let cinnamon_rolls : ℕ := 110
  let pastries : ℕ := croissants + muffins + cinnamon_rolls
  let bagels : ℕ := total_items - (bread_rolls + pastries)
  let pastries_per_bread_roll : ℚ := 2.5
  let bagels_per_5_bread_rolls : ℕ := 3

  (pastries : ℚ) / bread_rolls = pastries_per_bread_roll ∧
  (bagels : ℚ) / bread_rolls = (bagels_per_5_bread_rolls : ℚ) / 5 →
  pastries + bagels = 474 := by
sorry

end NUMINAMATH_CALUDE_bakery_pastries_and_bagels_l69_6998


namespace NUMINAMATH_CALUDE_sum_of_digits_of_five_to_23_l69_6995

/-- The sum of the tens digit and the ones digit of (2+3)^23 is 7 -/
theorem sum_of_digits_of_five_to_23 :
  let n : ℕ := (2 + 3)^23
  (n / 10 % 10) + (n % 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_five_to_23_l69_6995


namespace NUMINAMATH_CALUDE_sequence_a_10_l69_6934

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  ∀ m n : ℕ+, a (m + n) = a m * a n

theorem sequence_a_10 (a : ℕ+ → ℝ) (h1 : sequence_property a) (h2 : a 3 = 8) :
  a 10 = 1024 := by sorry

end NUMINAMATH_CALUDE_sequence_a_10_l69_6934


namespace NUMINAMATH_CALUDE_equation_transformation_l69_6963

theorem equation_transformation (x y : ℝ) : x - 3 = y - 3 → x - y = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_transformation_l69_6963


namespace NUMINAMATH_CALUDE_lcm_18_30_l69_6933

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_30_l69_6933


namespace NUMINAMATH_CALUDE_triangle_problem_l69_6970

theorem triangle_problem (AB : ℝ) (sinA sinC : ℝ) :
  AB = 30 →
  sinA = 4/5 →
  sinC = 1/4 →
  ∃ (DC : ℝ), DC = 24 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l69_6970


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l69_6902

theorem sqrt_sum_squares_equals_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a * b + a * c + b * c = 0 ∧ a + b + c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_equals_sum_l69_6902


namespace NUMINAMATH_CALUDE_study_days_needed_l69_6991

/-- Represents the study requirements for a subject --/
structure SubjectRequirements where
  chapters : ℕ
  worksheets : ℕ
  chapterTime : ℚ
  worksheetTime : ℚ

/-- Calculates the total study time for a subject --/
def totalStudyTime (req : SubjectRequirements) : ℚ :=
  req.chapters * req.chapterTime + req.worksheets * req.worksheetTime

/-- Represents the break schedule --/
structure BreakSchedule where
  firstThreeHours : ℚ
  nextThreeHours : ℚ
  lastHour : ℚ
  snackBreaks : ℚ
  lunchBreak : ℚ

/-- Calculates the total break time per day --/
def totalBreakTime (schedule : BreakSchedule) : ℚ :=
  3 * schedule.firstThreeHours + 3 * schedule.nextThreeHours + schedule.lastHour +
  2 * schedule.snackBreaks + schedule.lunchBreak

theorem study_days_needed :
  let math := SubjectRequirements.mk 4 7 (5/2) (3/2)
  let physics := SubjectRequirements.mk 5 9 3 2
  let chemistry := SubjectRequirements.mk 6 8 (7/2) (7/4)
  let breakSchedule := BreakSchedule.mk (1/6) (1/4) (1/3) (1/3) (3/4)
  let totalStudyHours := totalStudyTime math + totalStudyTime physics + totalStudyTime chemistry
  let effectiveStudyHoursPerDay := 7 - totalBreakTime breakSchedule
  ⌈totalStudyHours / effectiveStudyHoursPerDay⌉ = 23 := by
  sorry

end NUMINAMATH_CALUDE_study_days_needed_l69_6991


namespace NUMINAMATH_CALUDE_cuboid_properties_l69_6947

/-- Represents a cuboid with given length, width, and height -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total length of edges of a cuboid -/
def totalEdgeLength (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

/-- Theorem stating the correctness of totalEdgeLength and surfaceArea functions -/
theorem cuboid_properties (c : Cuboid) :
  (totalEdgeLength c = 4 * (c.length + c.width + c.height)) ∧
  (surfaceArea c = 2 * (c.length * c.width + c.width * c.height + c.height * c.length)) := by
  sorry

end NUMINAMATH_CALUDE_cuboid_properties_l69_6947


namespace NUMINAMATH_CALUDE_fraction_inequality_condition_l69_6954

theorem fraction_inequality_condition (x : ℝ) : 
  (2 * x + 1) / (1 - x) ≥ 0 ↔ -1/2 ≤ x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_condition_l69_6954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l69_6993

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the properties of the sequence
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 2 + a 4 = 10)
  (h3 : ∃ r : ℝ, r ≠ 0 ∧ a 2 = a 1 * r ∧ a 5 = a 2 * r)
  (h4 : arithmetic_sequence a d) :
  a 1 = 1 ∧ ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l69_6993


namespace NUMINAMATH_CALUDE_pet_store_combinations_l69_6997

def num_puppies : ℕ := 20
def num_kittens : ℕ := 6
def num_hamsters : ℕ := 8

def alice_choices : ℕ := num_puppies
def bob_pet_type_choices : ℕ := 2
def bob_specific_pet_choices : ℕ := num_kittens
def charlie_choices : ℕ := num_hamsters

theorem pet_store_combinations : 
  alice_choices * bob_pet_type_choices * bob_specific_pet_choices * charlie_choices = 1920 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l69_6997


namespace NUMINAMATH_CALUDE_max_candies_for_class_l69_6948

def max_candies (num_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) : ℕ :=
  (num_students * mean_candies) - (min_candies * (num_students - 1))

theorem max_candies_for_class (num_students : ℕ) (mean_candies : ℕ) (min_candies : ℕ) 
  (h1 : num_students = 24)
  (h2 : mean_candies = 7)
  (h3 : min_candies = 3) :
  max_candies num_students mean_candies min_candies = 99 :=
by
  sorry

#eval max_candies 24 7 3

end NUMINAMATH_CALUDE_max_candies_for_class_l69_6948


namespace NUMINAMATH_CALUDE_crow_votes_l69_6916

def singing_contest (total_judges reported_total : ℕ)
                    (rooster_crow reported_rooster_crow : ℕ)
                    (crow_cuckoo reported_crow_cuckoo : ℕ)
                    (cuckoo_rooster reported_cuckoo_rooster : ℕ)
                    (max_error : ℕ) : Prop :=
  ∃ (rooster crow cuckoo : ℕ),
    -- Actual total of judges
    rooster + crow + cuckoo = total_judges ∧
    -- Reported total within error range
    (reported_total : ℤ) - (total_judges : ℤ) ≤ max_error ∧
    (total_judges : ℤ) - (reported_total : ℤ) ≤ max_error ∧
    -- Reported sums within error range
    (reported_rooster_crow : ℤ) - ((rooster + crow) : ℤ) ≤ max_error ∧
    ((rooster + crow) : ℤ) - (reported_rooster_crow : ℤ) ≤ max_error ∧
    (reported_crow_cuckoo : ℤ) - ((crow + cuckoo) : ℤ) ≤ max_error ∧
    ((crow + cuckoo) : ℤ) - (reported_crow_cuckoo : ℤ) ≤ max_error ∧
    (reported_cuckoo_rooster : ℤ) - ((cuckoo + rooster) : ℤ) ≤ max_error ∧
    ((cuckoo + rooster) : ℤ) - (reported_cuckoo_rooster : ℤ) ≤ max_error ∧
    -- The number of votes for Crow is 13
    crow = 13

theorem crow_votes :
  singing_contest 46 59 15 15 18 18 20 20 13 :=
by sorry

end NUMINAMATH_CALUDE_crow_votes_l69_6916


namespace NUMINAMATH_CALUDE_negation_equivalence_l69_6982

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 1 ≤ 2*x) ↔ (∀ x : ℝ, x^2 + 1 > 2*x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l69_6982


namespace NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l69_6923

theorem line_not_in_fourth_quadrant 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hc : c > 0) : 
  ∀ x y : ℝ, a * x + b * y + c = 0 → ¬(x > 0 ∧ y < 0) := by
sorry

end NUMINAMATH_CALUDE_line_not_in_fourth_quadrant_l69_6923


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l69_6986

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 + 5*x^2 + 2*x - 8

/-- Theorem stating that 1 + √3 and 1 - √3 are the real roots of the polynomial -/
theorem real_roots_of_polynomial :
  (∃ (x : ℝ), f x = 0) ↔ (∃ (x : ℝ), x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l69_6986


namespace NUMINAMATH_CALUDE_smallest_pair_sum_divisible_by_125_l69_6962

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a number is divisible by 125 -/
def divisible_by_125 (n : ℕ) : Prop := n % 125 = 0

/-- The smallest pair of consecutive numbers with sum of digits divisible by 125 -/
def smallest_pair : ℕ × ℕ := (89999999999998, 89999999999999)

theorem smallest_pair_sum_divisible_by_125 :
  let (a, b) := smallest_pair
  divisible_by_125 (sum_of_digits a) ∧
  divisible_by_125 (sum_of_digits b) ∧
  b = a + 1 ∧
  ∀ (x y : ℕ), x < a → y = x + 1 →
    ¬(divisible_by_125 (sum_of_digits x) ∧ divisible_by_125 (sum_of_digits y)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_pair_sum_divisible_by_125_l69_6962


namespace NUMINAMATH_CALUDE_probability_of_event_b_l69_6957

theorem probability_of_event_b 
  (prob_a : ℝ) 
  (prob_a_and_b : ℝ) 
  (prob_neither_a_nor_b : ℝ) 
  (h1 : prob_a = 0.20)
  (h2 : prob_a_and_b = 0.15)
  (h3 : prob_neither_a_nor_b = 0.5499999999999999) :
  ∃ (prob_b : ℝ), prob_b = 0.40 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_event_b_l69_6957


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l69_6940

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l69_6940


namespace NUMINAMATH_CALUDE_part_one_part_two_l69_6914

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0

def q (m x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Part I
theorem part_one (m : ℝ) : 
  (m > 0 ∧ (∀ x, ¬(q m x) → ¬(p x))) ↔ (0 < m ∧ m ≤ 2) :=
sorry

-- Part II
theorem part_two (x : ℝ) :
  ((p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x)) ↔ ((-6 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x < 8)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l69_6914


namespace NUMINAMATH_CALUDE_hexagon_tessellation_l69_6903

-- Define a hexagon
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

-- Define properties of the hexagon
def is_convex (h : Hexagon) : Prop :=
  sorry

def has_parallel_opposite_sides (h : Hexagon) : Prop :=
  sorry

def parallel_sides_length_one (h : Hexagon) : Prop :=
  sorry

-- Define tessellation
def can_tessellate_plane (h : Hexagon) : Prop :=
  sorry

-- Theorem statement
theorem hexagon_tessellation :
  ∃ (h : Hexagon), 
    is_convex h ∧ 
    has_parallel_opposite_sides h ∧ 
    parallel_sides_length_one h ∧ 
    can_tessellate_plane h :=
sorry

end NUMINAMATH_CALUDE_hexagon_tessellation_l69_6903


namespace NUMINAMATH_CALUDE_consecutive_composites_exist_l69_6911

/-- A natural number is composite if it has more than two distinct positive divisors -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

/-- A sequence of n consecutive composite numbers starting from k -/
def ConsecutiveComposites (k n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → IsComposite (k + i)

theorem consecutive_composites_exist :
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 9) ∧
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composites_exist_l69_6911


namespace NUMINAMATH_CALUDE_bank_investment_problem_l69_6983

theorem bank_investment_problem (total_investment interest_rate1 interest_rate2 total_interest : ℝ)
  (h1 : total_investment = 5000)
  (h2 : interest_rate1 = 0.04)
  (h3 : interest_rate2 = 0.065)
  (h4 : total_interest = 282.5)
  (h5 : ∃ x y : ℝ, x + y = total_investment ∧ interest_rate1 * x + interest_rate2 * y = total_interest) :
  ∃ x : ℝ, x = 1700 ∧ interest_rate1 * x + interest_rate2 * (total_investment - x) = total_interest :=
by
  sorry

end NUMINAMATH_CALUDE_bank_investment_problem_l69_6983


namespace NUMINAMATH_CALUDE_range_of_f_l69_6969

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := arctan x + arctan ((x - 2) / (x + 2))

-- Theorem statement
theorem range_of_f :
  ∃ (S : Set ℝ), S = Set.range f ∧ S = {-π/4, arctan 2} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l69_6969


namespace NUMINAMATH_CALUDE_work_completion_time_l69_6905

theorem work_completion_time (p_rate q_rate : ℚ) (work_left : ℚ) : 
  p_rate = 1/20 → q_rate = 1/10 → work_left = 7/10 → 
  (p_rate + q_rate) * 2 = 1 - work_left := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l69_6905


namespace NUMINAMATH_CALUDE_total_cars_count_l69_6980

/-- Represents the number of cars counted by each person -/
structure CarCounts where
  jared : ℕ
  ann : ℕ
  alfred : ℕ
  bella : ℕ

/-- Calculates the total number of cars counted by all people -/
def total_count (counts : CarCounts) : ℕ :=
  counts.jared + counts.ann + counts.alfred + counts.bella

/-- Theorem stating the total count of cars after Alfred's recount -/
theorem total_cars_count (counts : CarCounts) :
  counts.jared = 300 ∧
  counts.ann = counts.jared + counts.jared * 15 / 100 ∧
  counts.alfred = counts.ann - 7 + (counts.ann - 7) * 12 / 100 ∧
  counts.bella = counts.jared + counts.jared * 20 / 100 ∧
  counts.bella = counts.alfred - counts.alfred * 10 / 100 →
  total_count counts = 1365 := by
  sorry

#eval total_count { jared := 300, ann := 345, alfred := 379, bella := 341 }

end NUMINAMATH_CALUDE_total_cars_count_l69_6980


namespace NUMINAMATH_CALUDE_hyperbola_equation_l69_6927

/-- Hyperbola with given properties -/
def Hyperbola (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) ∧
  (∀ x : ℝ, ∃ y : ℝ, y = (Real.sqrt 3 / 3) * x ∨ y = -(Real.sqrt 3 / 3) * x) ∧
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ |Real.sqrt 3 * x - 3 * y| / Real.sqrt 12 = 1)

/-- The equation of the hyperbola with the given properties -/
theorem hyperbola_equation :
  ∀ a b : ℝ, Hyperbola a b → (∀ x y : ℝ, x^2 / 4 - 3 * y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l69_6927


namespace NUMINAMATH_CALUDE_pencil_count_l69_6920

theorem pencil_count :
  ∀ (pens pencils : ℕ),
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 4 →
  pencils = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l69_6920


namespace NUMINAMATH_CALUDE_remainder_of_1235678901_mod_101_l69_6999

theorem remainder_of_1235678901_mod_101 : 1235678901 % 101 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1235678901_mod_101_l69_6999


namespace NUMINAMATH_CALUDE_k_set_characterization_l69_6996

theorem k_set_characterization (r : ℕ) :
  let h := 2^r
  let k_set := {k : ℕ | ∃ (m n : ℕ), 
    Odd m ∧ m > 1 ∧
    k ∣ m^k - 1 ∧
    m ∣ n^((m^k - 1)/k) + 1}
  k_set = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ ¬ Even t} :=
by sorry

end NUMINAMATH_CALUDE_k_set_characterization_l69_6996


namespace NUMINAMATH_CALUDE_sum_positive_if_difference_abs_positive_l69_6958

theorem sum_positive_if_difference_abs_positive (a b : ℝ) :
  a - |b| > 0 → b + a > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_if_difference_abs_positive_l69_6958


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l69_6979

theorem reciprocal_of_negative_two_thirds :
  ((-2 : ℚ) / 3)⁻¹ = -3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_thirds_l69_6979


namespace NUMINAMATH_CALUDE_function_range_l69_6943

theorem function_range (x : ℝ) : 
  (∀ a : ℝ, a ∈ Set.Icc (-1 : ℝ) 1 → 
    (a * x^2 - (2*a + 1) * x + a + 1 < 0)) → 
  (1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_function_range_l69_6943


namespace NUMINAMATH_CALUDE_unscreened_percentage_l69_6936

def tv_width : ℝ := 6
def tv_height : ℝ := 5
def screen_width : ℝ := 5
def screen_height : ℝ := 4

theorem unscreened_percentage :
  (tv_width * tv_height - screen_width * screen_height) / (tv_width * tv_height) * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unscreened_percentage_l69_6936


namespace NUMINAMATH_CALUDE_inequality_proof_l69_6978

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (1+a)/(1-a) + (1+b)/(1-b) + (1+c)/(1-c) ≤ 2*(b/a + c/b + a/c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l69_6978


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l69_6961

theorem bowl_glass_pairings :
  let num_bowls : ℕ := 5
  let num_glasses : ℕ := 4
  num_bowls * num_glasses = 20 :=
by sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l69_6961


namespace NUMINAMATH_CALUDE_equation_solution_l69_6992

theorem equation_solution :
  ∃ x : ℝ, (Real.sqrt (9 + Real.sqrt (15 + 5*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 5 + Real.sqrt 15) ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l69_6992


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l69_6938

theorem quadratic_roots_problem (α β k : ℝ) : 
  (α^2 - α + k - 1 = 0) →
  (β^2 - β + k - 1 = 0) →
  (α^2 - 2*α - β = 4) →
  (k = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l69_6938


namespace NUMINAMATH_CALUDE_characterization_of_f_l69_6925

-- Define the property of being strictly increasing
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define the functional equation
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (f x + y) = f (x + y) + f 0

-- Theorem statement
theorem characterization_of_f :
  ∀ f : ℝ → ℝ, StrictlyIncreasing f → SatisfiesEquation f →
  ∃ k : ℝ, ∀ x, f x = x - k :=
sorry

end NUMINAMATH_CALUDE_characterization_of_f_l69_6925


namespace NUMINAMATH_CALUDE_pig_feed_per_day_l69_6988

/-- Given that Randy has 2 pigs and they are fed 140 pounds of pig feed per week,
    prove that each pig is fed 10 pounds of feed per day. -/
theorem pig_feed_per_day (num_pigs : ℕ) (total_feed_per_week : ℕ) (days_per_week : ℕ) :
  num_pigs = 2 →
  total_feed_per_week = 140 →
  days_per_week = 7 →
  (total_feed_per_week / num_pigs) / days_per_week = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_pig_feed_per_day_l69_6988


namespace NUMINAMATH_CALUDE_equation_solution_l69_6932

theorem equation_solution (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l69_6932


namespace NUMINAMATH_CALUDE_fraction_unchanged_l69_6941

theorem fraction_unchanged (x y : ℝ) : (x + y) / (x - 2*y) = ((-x) + (-y)) / ((-x) - 2*(-y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l69_6941


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l69_6937

theorem cubic_minus_linear_factorization (x : ℝ) : x^3 - x = x*(x+1)*(x-1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l69_6937


namespace NUMINAMATH_CALUDE_initial_books_correct_l69_6984

/-- Calculates the initial number of books in Mary's mystery book library --/
def initial_books : ℕ :=
  let books_received := 12 -- 1 book per month for 12 months
  let books_bought := 5 + 2 -- 5 from bookstore, 2 from yard sales
  let books_gifted := 1 + 4 -- 1 from daughter, 4 from mother
  let books_removed := 12 + 3 -- 12 donated, 3 sold
  let final_books := 81

  final_books - (books_received + books_bought + books_gifted) + books_removed

theorem initial_books_correct :
  initial_books = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_books_correct_l69_6984


namespace NUMINAMATH_CALUDE_log_meaningful_iff_in_range_l69_6912

def meaningful_log (a : ℝ) : Prop :=
  a - 2 > 0 ∧ a - 2 ≠ 1 ∧ 5 - a > 0

theorem log_meaningful_iff_in_range (a : ℝ) :
  meaningful_log a ↔ (a > 2 ∧ a < 3) ∨ (a > 3 ∧ a < 5) :=
sorry

end NUMINAMATH_CALUDE_log_meaningful_iff_in_range_l69_6912


namespace NUMINAMATH_CALUDE_min_value_quadratic_l69_6987

theorem min_value_quadratic (x y : ℝ) :
  2 * x^2 + 2 * y^2 - 8 * x + 6 * y + 25 ≥ 12.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l69_6987


namespace NUMINAMATH_CALUDE_sprint_no_wind_time_l69_6989

/-- A sprinter's performance under different wind conditions -/
structure SprintPerformance where
  with_wind_distance : ℝ
  against_wind_distance : ℝ
  time_with_wind : ℝ
  time_against_wind : ℝ
  wind_speed : ℝ
  no_wind_speed : ℝ

/-- Theorem stating the time taken to run 100 meters in no wind condition -/
theorem sprint_no_wind_time (perf : SprintPerformance) 
  (h1 : perf.with_wind_distance = 90)
  (h2 : perf.against_wind_distance = 70)
  (h3 : perf.time_with_wind = 10)
  (h4 : perf.time_against_wind = 10)
  (h5 : perf.with_wind_distance / (perf.no_wind_speed + perf.wind_speed) = perf.time_with_wind)
  (h6 : perf.against_wind_distance / (perf.no_wind_speed - perf.wind_speed) = perf.time_against_wind)
  : (100 : ℝ) / perf.no_wind_speed = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_sprint_no_wind_time_l69_6989


namespace NUMINAMATH_CALUDE_no_five_solutions_and_divisibility_l69_6952

theorem no_five_solutions_and_divisibility (k : ℤ) :
  (¬ ∃ (x₁ x₂ x₃ x₄ x₅ y₁ : ℤ),
    y₁^2 - k = x₁^3 ∧
    (y₁ - 1)^2 - k = x₂^3 ∧
    (y₁ - 2)^2 - k = x₃^3 ∧
    (y₁ - 3)^2 - k = x₄^3 ∧
    (y₁ - 4)^2 - k = x₅^3) ∧
  (∀ (x₁ x₂ x₃ x₄ y₁ : ℤ),
    y₁^2 - k = x₁^3 ∧
    (y₁ - 1)^2 - k = x₂^3 ∧
    (y₁ - 2)^2 - k = x₃^3 ∧
    (y₁ - 3)^2 - k = x₄^3 →
    63 ∣ (k - 17)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_solutions_and_divisibility_l69_6952


namespace NUMINAMATH_CALUDE_matrix_multiplication_example_l69_6981

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![23, -7; 24, -16]
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_example_l69_6981


namespace NUMINAMATH_CALUDE_max_sum_with_reciprocals_l69_6994

theorem max_sum_with_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 1/x + 1/y = 5) : 
  ∀ a b : ℝ, a > 0 → b > 0 → a + b + 1/a + 1/b = 5 → x + y ≥ a + b :=
by sorry

end NUMINAMATH_CALUDE_max_sum_with_reciprocals_l69_6994


namespace NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l69_6922

/-- Represents a square card with a given side length -/
structure SquareCard where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a checkerboard with squares of a given side length -/
structure Checkerboard where
  square_side_length : ℝ
  square_side_length_pos : square_side_length > 0

/-- The maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : SquareCard) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a 2-inch square card can cover at most 9 one-inch squares on a checkerboard -/
theorem max_squares_covered_two_inch_card :
  ∀ (card : SquareCard) (board : Checkerboard),
    card.side_length = 2 →
    board.square_side_length = 1 →
    max_squares_covered card board = 9 :=
  sorry

end NUMINAMATH_CALUDE_max_squares_covered_two_inch_card_l69_6922


namespace NUMINAMATH_CALUDE_sequence_periodicity_l69_6945

theorem sequence_periodicity (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n) :
  ∃ N : ℕ, ∀ n ≥ N, a (n + 9) = a n :=
sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l69_6945


namespace NUMINAMATH_CALUDE_fraction_sum_l69_6901

theorem fraction_sum (a b : ℚ) (h : a / b = 3 / 2) : (a + b) / b = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l69_6901


namespace NUMINAMATH_CALUDE_rhombus_area_l69_6971

/-- The area of a rhombus with diagonals of length 6 and 10 is 30. -/
theorem rhombus_area (d₁ d₂ : ℝ) (h₁ : d₁ = 6) (h₂ : d₂ = 10) : 
  (1 / 2 : ℝ) * d₁ * d₂ = 30 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l69_6971


namespace NUMINAMATH_CALUDE_largest_difference_l69_6931

def U : ℕ := 2 * 2010^2011
def V : ℕ := 2010^2011
def W : ℕ := 2009 * 2010^2010
def X : ℕ := 2 * 2010^2010
def Y : ℕ := 2010^2010
def Z : ℕ := 2010^2009

theorem largest_difference : 
  (U - V > V - W) ∧ 
  (U - V > W - X + 100) ∧ 
  (U - V > X - Y) ∧ 
  (U - V > Y - Z) := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l69_6931


namespace NUMINAMATH_CALUDE_ceiling_floor_expression_l69_6907

theorem ceiling_floor_expression : 
  ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ - ⌈(2:ℝ)/3⌉ = -1 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_expression_l69_6907


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l69_6919

-- Define the position function
def s (t : ℝ) : ℝ := 3 * t^2

-- Define the velocity function as the derivative of the position function
noncomputable def v (t : ℝ) : ℝ := deriv s t

-- Theorem statement
theorem instantaneous_velocity_at_3 : v 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_l69_6919


namespace NUMINAMATH_CALUDE_crayons_given_away_l69_6949

theorem crayons_given_away (initial_crayons remaining_crayons : ℕ) 
  (h1 : initial_crayons = 106)
  (h2 : remaining_crayons = 52) : 
  initial_crayons - remaining_crayons = 54 := by
  sorry

end NUMINAMATH_CALUDE_crayons_given_away_l69_6949


namespace NUMINAMATH_CALUDE_y_value_l69_6924

theorem y_value (y : ℝ) (h : (9 : ℝ) / (y^2) = y / 81) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l69_6924


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l69_6985

theorem semicircle_area_with_inscribed_rectangle :
  ∀ (r : ℝ),
  r > 0 →
  ∃ (semicircle_area : ℝ),
  (3 : ℝ)^2 + 1^2 = (2 * r)^2 →
  semicircle_area = (13 * π) / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l69_6985


namespace NUMINAMATH_CALUDE_integral_x_squared_l69_6990

theorem integral_x_squared : ∫ x in (0:ℝ)..1, x^2 = (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_l69_6990


namespace NUMINAMATH_CALUDE_running_distance_proof_l69_6908

/-- Calculates the total distance run over a number of days, given a constant daily distance. -/
def totalDistance (dailyDistance : ℕ) (days : ℕ) : ℕ :=
  dailyDistance * days

/-- Proves that running 1700 meters for 6 consecutive days results in a total distance of 10200 meters. -/
theorem running_distance_proof :
  let dailyDistance : ℕ := 1700
  let days : ℕ := 6
  totalDistance dailyDistance days = 10200 := by
sorry

end NUMINAMATH_CALUDE_running_distance_proof_l69_6908


namespace NUMINAMATH_CALUDE_quadratic_monotone_increasing_condition_l69_6974

/-- A quadratic function f(x) = ax^2 + bx + c is monotonically increasing on [1, +∞)
    if and only if b ≥ -2a, where a > 0. -/
theorem quadratic_monotone_increasing_condition (a b c : ℝ) (ha : a > 0) :
  (∀ x y, x ∈ Set.Ici (1 : ℝ) → y ∈ Set.Ici (1 : ℝ) → x ≤ y →
    a * x^2 + b * x + c ≤ a * y^2 + b * y + c) ↔
  b ≥ -2 * a :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotone_increasing_condition_l69_6974


namespace NUMINAMATH_CALUDE_pams_bank_balance_l69_6939

theorem pams_bank_balance (initial_balance : ℕ) (withdrawal : ℕ) (final_balance : ℕ) : 
  initial_balance = 400 →
  withdrawal = 250 →
  final_balance = (initial_balance * 3) - withdrawal →
  final_balance = 950 := by
sorry

end NUMINAMATH_CALUDE_pams_bank_balance_l69_6939


namespace NUMINAMATH_CALUDE_cos_double_angle_problem_l69_6964

theorem cos_double_angle_problem (α : ℝ) (h : Real.cos (π + α) = 2/5) :
  Real.cos (2 * α) = -17/25 := by sorry

end NUMINAMATH_CALUDE_cos_double_angle_problem_l69_6964


namespace NUMINAMATH_CALUDE_find_set_B_l69_6930

theorem find_set_B (a b : ℝ) : 
  let P : Set ℝ := {1, a/b, b}
  let B : Set ℝ := {0, a+b, b^2}
  P = B → B = {0, -1, 1} := by
sorry

end NUMINAMATH_CALUDE_find_set_B_l69_6930


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l69_6904

theorem cube_volume_ratio (e : ℝ) (h : e > 0) :
  let small_cube_volume := e^3
  let large_cube_volume := (3*e)^3
  large_cube_volume = 27 * small_cube_volume := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l69_6904


namespace NUMINAMATH_CALUDE_treasure_trap_probability_l69_6967

/-- The number of islands --/
def num_islands : ℕ := 5

/-- The probability of an island having treasure and no traps --/
def p_treasure : ℚ := 1/5

/-- The probability of an island having traps but no treasure --/
def p_traps : ℚ := 1/5

/-- The probability of an island having neither traps nor treasure --/
def p_neither : ℚ := 3/5

/-- The number of islands with treasure --/
def treasure_islands : ℕ := 2

/-- The number of islands with traps --/
def trap_islands : ℕ := 2

/-- Theorem stating the probability of encountering exactly 2 islands with treasure and 2 with traps --/
theorem treasure_trap_probability : 
  (Nat.choose num_islands treasure_islands) * 
  (Nat.choose (num_islands - treasure_islands) trap_islands) * 
  (p_treasure ^ treasure_islands) * 
  (p_traps ^ trap_islands) * 
  (p_neither ^ (num_islands - treasure_islands - trap_islands)) = 18/625 := by
sorry

end NUMINAMATH_CALUDE_treasure_trap_probability_l69_6967


namespace NUMINAMATH_CALUDE_gym_cost_comparison_l69_6955

/-- Represents the cost of gym sessions under two different schemes -/
def gym_cost (x : ℕ) : ℝ × ℝ :=
  let y₁ := 12 * x + 40  -- Scheme 1: 40% discount + membership
  let y₂ := 16 * x       -- Scheme 2: 20% discount, no membership
  (y₁, y₂)

/-- Theorem stating which scheme is cheaper based on the number of sessions -/
theorem gym_cost_comparison (x : ℕ) (h : 5 ≤ x ∧ x ≤ 20) :
  let (y₁, y₂) := gym_cost x
  (x < 10 → y₂ < y₁) ∧
  (x = 10 → y₁ = y₂) ∧
  (10 < x → y₁ < y₂) :=
by sorry

end NUMINAMATH_CALUDE_gym_cost_comparison_l69_6955


namespace NUMINAMATH_CALUDE_car_travel_time_l69_6906

/-- Given a car's initial travel and additional distance, calculate the total travel time. -/
theorem car_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) 
  (h1 : initial_distance = 180) 
  (h2 : initial_time = 4)
  (h3 : additional_distance = 135) :
  let speed := initial_distance / initial_time
  let additional_time := additional_distance / speed
  initial_time + additional_time = 7 := by sorry

end NUMINAMATH_CALUDE_car_travel_time_l69_6906


namespace NUMINAMATH_CALUDE_sum_reciprocal_pairs_gt_one_l69_6951

theorem sum_reciprocal_pairs_gt_one (a₁ a₂ a₃ : ℝ) (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) :
  let S := a₁ + a₂ + a₃
  (a₁^2 / (a₁ - 1) > S) ∧ (a₂^2 / (a₂ - 1) > S) ∧ (a₃^2 / (a₃ - 1) > S) →
  1 / (a₁ + a₂) + 1 / (a₂ + a₃) + 1 / (a₃ + a₁) > 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_pairs_gt_one_l69_6951


namespace NUMINAMATH_CALUDE_problem_statement_l69_6966

theorem problem_statement : (1 + Real.sqrt 2) ^ 2023 * (1 - Real.sqrt 2) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l69_6966


namespace NUMINAMATH_CALUDE_cubic_log_relationship_l69_6977

theorem cubic_log_relationship (x : ℝ) :
  (x^3 < 27 → Real.log x / Real.log (1/3) > -1) ∧
  ¬(Real.log x / Real.log (1/3) > -1 → x^3 < 27) :=
sorry

end NUMINAMATH_CALUDE_cubic_log_relationship_l69_6977


namespace NUMINAMATH_CALUDE_check_amount_l69_6960

theorem check_amount (total_parts : ℕ) (expensive_parts : ℕ) (cheap_price : ℕ) (expensive_price : ℕ) : 
  total_parts = 59 → 
  expensive_parts = 40 → 
  cheap_price = 20 → 
  expensive_price = 50 → 
  (total_parts - expensive_parts) * cheap_price + expensive_parts * expensive_price = 2380 := by
sorry

end NUMINAMATH_CALUDE_check_amount_l69_6960


namespace NUMINAMATH_CALUDE_complement_of_A_l69_6910

def A : Set ℝ := {x | |x - 1| > 2}

theorem complement_of_A : 
  (Set.univ \ A) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l69_6910


namespace NUMINAMATH_CALUDE_trajectory_is_line_segment_l69_6976

/-- The trajectory of a point M satisfying |MF₁| + |MF₂| = 8, where F₁ and F₂ are fixed points -/
theorem trajectory_is_line_segment (F₁ F₂ M : ℝ × ℝ) : 
  F₁ = (-4, 0) → 
  F₂ = (4, 0) → 
  dist M F₁ + dist M F₂ = 8 → 
  ∃ (a b : ℝ), M.1 = a ∧ M.2 = b ∧ a ∈ [-4, 4] ∧ b = 0 :=
sorry


end NUMINAMATH_CALUDE_trajectory_is_line_segment_l69_6976


namespace NUMINAMATH_CALUDE_cost_difference_formula_option_A_cheaper_at_50_l69_6928

/-- The number of teachers -/
def num_teachers : ℕ := 5

/-- The full ticket price -/
def full_price : ℕ := 40

/-- Cost calculation for Option A -/
def cost_A (x : ℕ) : ℕ := 20 * x + 200

/-- Cost calculation for Option B -/
def cost_B (x : ℕ) : ℕ := 24 * x + 120

/-- The cost difference between Option B and Option A -/
def cost_difference (x : ℕ) : ℤ := (cost_B x : ℤ) - (cost_A x : ℤ)

theorem cost_difference_formula (x : ℕ) : 
  cost_difference x = 4 * x - 80 :=
sorry

theorem option_A_cheaper_at_50 : 
  cost_A 50 < cost_B 50 :=
sorry

end NUMINAMATH_CALUDE_cost_difference_formula_option_A_cheaper_at_50_l69_6928


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l69_6900

theorem binomial_expansion_coefficient (a : ℝ) (b : ℝ) :
  (∃ x, (1 + a * x)^5 = 1 + 10 * x + b * x^2 + x^3 * (1 + a * x)^2) →
  b = 40 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l69_6900


namespace NUMINAMATH_CALUDE_equal_distribution_problem_l69_6921

theorem equal_distribution_problem (earnings : Fin 5 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 27)
  (h3 : earnings 2 = 30)
  (h4 : earnings 3 = 35)
  (h5 : earnings 4 = 50) :
  50 - (earnings 0 + earnings 1 + earnings 2 + earnings 3 + earnings 4) / 5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_problem_l69_6921


namespace NUMINAMATH_CALUDE_f_prime_at_two_l69_6913

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b / x

theorem f_prime_at_two (a b : ℝ) :
  (f a b 1 = -2) →
  ((deriv (f a b)) 1 = 0) →
  ((deriv (f a b)) 2 = -1/2) :=
sorry

end NUMINAMATH_CALUDE_f_prime_at_two_l69_6913


namespace NUMINAMATH_CALUDE_fish_fillet_problem_l69_6956

theorem fish_fillet_problem (total : ℕ) (team1 : ℕ) (team2 : ℕ) 
  (h1 : total = 500) 
  (h2 : team1 = 189) 
  (h3 : team2 = 131) : 
  total - (team1 + team2) = 180 := by
sorry

end NUMINAMATH_CALUDE_fish_fillet_problem_l69_6956


namespace NUMINAMATH_CALUDE_correct_ticket_count_l69_6915

/-- Represents the number of first-class tickets bought -/
def first_class_tickets : ℕ := 20

/-- Represents the number of second-class tickets bought -/
def second_class_tickets : ℕ := 45 - first_class_tickets

/-- The total cost of all tickets -/
def total_cost : ℕ := 400

theorem correct_ticket_count :
  first_class_tickets * 10 + second_class_tickets * 8 = total_cost ∧
  first_class_tickets + second_class_tickets = 45 :=
sorry

end NUMINAMATH_CALUDE_correct_ticket_count_l69_6915


namespace NUMINAMATH_CALUDE_inequality_preservation_l69_6918

theorem inequality_preservation (a b : ℝ) (h : a < b) : a - 5 < b - 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l69_6918


namespace NUMINAMATH_CALUDE_caterpillar_eggs_hatched_l69_6950

theorem caterpillar_eggs_hatched (initial_caterpillars : ℕ) (caterpillars_left : ℕ) (final_caterpillars : ℕ) 
  (h1 : initial_caterpillars = 14)
  (h2 : caterpillars_left = 8)
  (h3 : final_caterpillars = 10) :
  initial_caterpillars + (caterpillars_left + final_caterpillars - initial_caterpillars) - caterpillars_left = final_caterpillars :=
by sorry

end NUMINAMATH_CALUDE_caterpillar_eggs_hatched_l69_6950


namespace NUMINAMATH_CALUDE_circle_sum_problem_l69_6946

theorem circle_sum_problem (a b c d X Y : ℤ) 
  (h1 : a + b + c + d = 40)
  (h2 : X + Y + c + b = 40)
  (h3 : a + b + X = 30)
  (h4 : c + d + Y = 30)
  (h5 : X = 9) :
  Y = 11 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_problem_l69_6946


namespace NUMINAMATH_CALUDE_problem_1_l69_6953

theorem problem_1 (a : ℝ) : a * (2 - a) + (a + 1) * (a - 1) = 2 * a - 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l69_6953


namespace NUMINAMATH_CALUDE_total_tips_proof_l69_6975

/-- Calculates the total tips earned over three days given the tips per customer,
    customer counts for Friday and Sunday, and that Saturday's count is 3 times Friday's. -/
def total_tips (tips_per_customer : ℕ) (friday_customers : ℕ) (sunday_customers : ℕ) : ℕ :=
  let saturday_customers := 3 * friday_customers
  tips_per_customer * (friday_customers + saturday_customers + sunday_customers)

/-- Proves that the total tips earned over three days is $296 -/
theorem total_tips_proof : total_tips 2 28 36 = 296 := by
  sorry

end NUMINAMATH_CALUDE_total_tips_proof_l69_6975
