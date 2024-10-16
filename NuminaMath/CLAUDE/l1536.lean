import Mathlib

namespace NUMINAMATH_CALUDE_question_selection_ways_l1536_153663

theorem question_selection_ways : 
  (Nat.choose 10 8) * (Nat.choose 10 5) = 11340 := by sorry

end NUMINAMATH_CALUDE_question_selection_ways_l1536_153663


namespace NUMINAMATH_CALUDE_intersection_complement_l1536_153670

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement : M ∩ (U \ N) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_l1536_153670


namespace NUMINAMATH_CALUDE_four_carpenters_in_five_hours_l1536_153602

/-- Represents the number of desks built by a given number of carpenters in a specific time -/
def desks_built (carpenters : ℕ) (hours : ℚ) : ℚ :=
  sorry

/-- Two carpenters can build 2 desks in 2.5 hours -/
axiom two_carpenters_rate : desks_built 2 (5/2) = 2

/-- All carpenters work at the same pace -/
axiom same_pace (c₁ c₂ : ℕ) (h : ℚ) :
  c₁ * desks_built c₂ h = c₂ * desks_built c₁ h

theorem four_carpenters_in_five_hours :
  desks_built 4 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_carpenters_in_five_hours_l1536_153602


namespace NUMINAMATH_CALUDE_initial_men_is_four_l1536_153639

/-- The number of men initially checking exam papers -/
def initial_men : ℕ := 4

/-- The number of days for the initial group to check papers -/
def initial_days : ℕ := 8

/-- The number of hours per day for the initial group -/
def initial_hours_per_day : ℕ := 5

/-- The number of men in the second group -/
def second_men : ℕ := 2

/-- The number of days for the second group to check papers -/
def second_days : ℕ := 20

/-- The number of hours per day for the second group -/
def second_hours_per_day : ℕ := 8

/-- Theorem stating that the initial number of men is 4 -/
theorem initial_men_is_four :
  initial_men * initial_days * initial_hours_per_day = 
  (second_men * second_days * second_hours_per_day) / 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_men_is_four_l1536_153639


namespace NUMINAMATH_CALUDE_digit_sum_problem_l1536_153608

theorem digit_sum_problem (P Q R S : ℕ) : 
  P < 10 → Q < 10 → R < 10 → S < 10 →
  P * 100 + 45 + Q * 10 + R + S = 654 →
  P + Q + R + S = 15 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l1536_153608


namespace NUMINAMATH_CALUDE_b_current_age_l1536_153632

/-- Given two people A and B, where:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is now 5 years older than B.
    Prove that B's current age is 35 years. -/
theorem b_current_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 5) : 
  b = 35 := by
  sorry

end NUMINAMATH_CALUDE_b_current_age_l1536_153632


namespace NUMINAMATH_CALUDE_freyja_age_l1536_153688

/-- Represents the ages of the people in the problem -/
structure Ages where
  kaylin : ℕ
  sarah : ℕ
  eli : ℕ
  freyja : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.kaylin = ages.sarah - 5 ∧
  ages.sarah = 2 * ages.eli ∧
  ages.eli = ages.freyja + 9 ∧
  ages.kaylin = 33

/-- The theorem stating Freyja's age given the problem conditions -/
theorem freyja_age (ages : Ages) (h : problem_conditions ages) : ages.freyja = 10 := by
  sorry


end NUMINAMATH_CALUDE_freyja_age_l1536_153688


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_mutually_exclusive_but_not_opposite_l1536_153643

/-- Represents the contents of a pencil case -/
structure PencilCase where
  pencils : ℕ
  pens : ℕ

/-- Represents the outcome of selecting two items -/
inductive Selection
  | TwoPencils
  | OnePencilOnePen
  | TwoPens

/-- Defines the pencil case with 2 pencils and 2 pens -/
def case : PencilCase := ⟨2, 2⟩

/-- Predicate for exactly one pen being selected -/
def exactlyOnePen (s : Selection) : Prop :=
  s = Selection.OnePencilOnePen

/-- Predicate for exactly two pencils being selected -/
def exactlyTwoPencils (s : Selection) : Prop :=
  s = Selection.TwoPencils

/-- Theorem stating that "Exactly 1 pen" and "Exactly 2 pencils" are mutually exclusive -/
theorem mutually_exclusive :
  ∀ s : Selection, ¬(exactlyOnePen s ∧ exactlyTwoPencils s) :=
sorry

/-- Theorem stating that "Exactly 1 pen" and "Exactly 2 pencils" are not opposite events -/
theorem not_opposite :
  ∃ s : Selection, ¬(exactlyOnePen s ∨ exactlyTwoPencils s) :=
sorry

/-- Main theorem combining the above results -/
theorem mutually_exclusive_but_not_opposite :
  (∀ s : Selection, ¬(exactlyOnePen s ∧ exactlyTwoPencils s)) ∧
  (∃ s : Selection, ¬(exactlyOnePen s ∨ exactlyTwoPencils s)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposite_mutually_exclusive_but_not_opposite_l1536_153643


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1536_153622

theorem roots_sum_of_squares (p q r s : ℝ) : 
  (r^2 - 3*p*r + 2*q = 0) → 
  (s^2 - 3*p*s + 2*q = 0) → 
  (r^2 + s^2 = 9*p^2 - 4*q) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1536_153622


namespace NUMINAMATH_CALUDE_joey_sneaker_purchase_l1536_153627

/-- The number of collectible figures Joey needs to sell to buy sneakers -/
def figures_to_sell (sneaker_cost lawn_count lawn_pay job_hours job_pay figure_price : ℕ) : ℕ :=
  let lawn_earnings := lawn_count * lawn_pay
  let job_earnings := job_hours * job_pay
  let total_earnings := lawn_earnings + job_earnings
  let remaining_amount := sneaker_cost - total_earnings
  (remaining_amount + figure_price - 1) / figure_price

theorem joey_sneaker_purchase :
  figures_to_sell 92 3 8 10 5 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joey_sneaker_purchase_l1536_153627


namespace NUMINAMATH_CALUDE_cycling_equipment_cost_l1536_153675

theorem cycling_equipment_cost (bike_initial : ℝ) (helmet_initial : ℝ) (gloves_initial : ℝ)
  (bike_increase : ℝ) (helmet_increase : ℝ) (gloves_increase : ℝ) (discount : ℝ)
  (h1 : bike_initial = 250)
  (h2 : helmet_initial = 60)
  (h3 : gloves_initial = 30)
  (h4 : bike_increase = 0.08)
  (h5 : helmet_increase = 0.15)
  (h6 : gloves_increase = 0.10)
  (h7 : discount = 0.05) :
  let bike_new := bike_initial * (1 + bike_increase)
  let helmet_new := helmet_initial * (1 + helmet_increase)
  let gloves_new := gloves_initial * (1 + gloves_increase)
  let total_before_discount := bike_new + helmet_new + gloves_new
  total_before_discount * (1 - discount) = 353.4 := by
  sorry

end NUMINAMATH_CALUDE_cycling_equipment_cost_l1536_153675


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_10_l1536_153680

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isLucky (n : ℕ) : Prop := 
  n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf10 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 10 * k

theorem least_non_lucky_multiple_of_10 : 
  (∀ m : ℕ, m < 110 → isMultipleOf10 m → isLucky m) ∧ 
  isMultipleOf10 110 ∧ 
  ¬isLucky 110 := by
sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_10_l1536_153680


namespace NUMINAMATH_CALUDE_total_trees_in_park_l1536_153698

theorem total_trees_in_park (ancient_oaks : ℕ) (fir_trees : ℕ) (saplings : ℕ)
  (h1 : ancient_oaks = 15)
  (h2 : fir_trees = 23)
  (h3 : saplings = 58) :
  ancient_oaks + fir_trees + saplings = 96 :=
by sorry

end NUMINAMATH_CALUDE_total_trees_in_park_l1536_153698


namespace NUMINAMATH_CALUDE_constant_value_proof_l1536_153645

/-- The coefficient of x in the expansion of (x - a/x)(1 - √x)^6 -/
def coefficient_of_x (a : ℝ) : ℝ := 1 - 15 * a

/-- The theorem stating that a = -2 when the coefficient of x is 31 -/
theorem constant_value_proof (a : ℝ) : coefficient_of_x a = 31 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l1536_153645


namespace NUMINAMATH_CALUDE_planting_area_difference_l1536_153693

/-- Given a village with wheat, rice, and corn planting areas, prove the difference between rice and corn areas. -/
theorem planting_area_difference (m : ℝ) : 
  let wheat_area : ℝ := m
  let rice_area : ℝ := 2 * wheat_area + 3
  let corn_area : ℝ := wheat_area - 5
  rice_area - corn_area = m + 8 := by
  sorry

end NUMINAMATH_CALUDE_planting_area_difference_l1536_153693


namespace NUMINAMATH_CALUDE_words_with_a_count_l1536_153605

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E'}

def word_length : Nat := 3

def words_with_a (s : Finset Char) (n : Nat) : Nat :=
  s.card ^ n - (s.erase 'A').card ^ n

theorem words_with_a_count :
  words_with_a alphabet word_length = 61 := by
  sorry

end NUMINAMATH_CALUDE_words_with_a_count_l1536_153605


namespace NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l1536_153673

theorem angle_with_supplement_four_times_complement (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l1536_153673


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l1536_153619

/-- 
Given an arithmetic sequence with:
  - First term: 156
  - Last term: 36
  - Common difference: -6
This theorem proves that the number of terms in the sequence is 21.
-/
theorem arithmetic_sequence_term_count : 
  let a₁ : ℤ := 156  -- First term
  let aₙ : ℤ := 36   -- Last term
  let d : ℤ := -6    -- Common difference
  ∃ n : ℕ, n > 0 ∧ aₙ = a₁ + (n - 1) * d ∧ n = 21
  := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l1536_153619


namespace NUMINAMATH_CALUDE_katya_magic_pen_problem_l1536_153699

theorem katya_magic_pen_problem (total_problems : ℕ) 
  (katya_prob : ℚ) (pen_prob : ℚ) (min_correct : ℕ) :
  total_problems = 20 →
  katya_prob = 4/5 →
  pen_prob = 1/2 →
  min_correct = 13 →
  ∃ (x : ℕ), x ≥ 10 ∧ 
    (x : ℚ) * katya_prob + (total_problems - x : ℚ) * pen_prob ≥ min_correct :=
by sorry

end NUMINAMATH_CALUDE_katya_magic_pen_problem_l1536_153699


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1536_153613

theorem divisibility_theorem (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1536_153613


namespace NUMINAMATH_CALUDE_no_solution_steers_cows_l1536_153676

theorem no_solution_steers_cows : ¬∃ (s c : ℕ), 
  30 * s + 32 * c = 1200 ∧ c > s ∧ s > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_steers_cows_l1536_153676


namespace NUMINAMATH_CALUDE_travel_distance_l1536_153665

theorem travel_distance (d : ℝ) (h1 : d > 0) :
  d / 4 + d / 8 + d / 12 = 11 / 60 →
  3 * d = 1.2 := by
sorry

end NUMINAMATH_CALUDE_travel_distance_l1536_153665


namespace NUMINAMATH_CALUDE_complex_sum_parts_zero_l1536_153641

theorem complex_sum_parts_zero (b : ℝ) : 
  let z : ℂ := 2 - b * I
  (z.re + z.im = 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_parts_zero_l1536_153641


namespace NUMINAMATH_CALUDE_disco_ball_max_cost_l1536_153659

def disco_ball_cost (total_budget : ℕ) (food_boxes : ℕ) (food_cost : ℕ) (disco_balls : ℕ) : ℕ :=
  (total_budget - food_boxes * food_cost) / disco_balls

theorem disco_ball_max_cost : disco_ball_cost 330 10 25 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_disco_ball_max_cost_l1536_153659


namespace NUMINAMATH_CALUDE_tom_hockey_games_attendance_l1536_153646

/-- The number of hockey games Tom attended over six years -/
def total_games_attended (year1 year2 year3 year4 year5 year6 : ℕ) : ℕ :=
  year1 + year2 + year3 + year4 + year5 + year6

/-- Theorem stating that Tom attended 41 hockey games over six years -/
theorem tom_hockey_games_attendance :
  total_games_attended 4 9 5 10 6 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_tom_hockey_games_attendance_l1536_153646


namespace NUMINAMATH_CALUDE_second_day_distance_l1536_153652

-- Define the constants
def first_day_distance : ℝ := 250
def average_speed : ℝ := 33.333333333333336
def time_difference : ℝ := 3

-- Define the theorem
theorem second_day_distance :
  let first_day_time := first_day_distance / average_speed
  let second_day_time := first_day_time + time_difference
  second_day_time * average_speed = 350 := by
  sorry

end NUMINAMATH_CALUDE_second_day_distance_l1536_153652


namespace NUMINAMATH_CALUDE_ceiling_minus_x_equals_half_l1536_153667

theorem ceiling_minus_x_equals_half (x : ℝ) (h : x - ⌊x⌋ = 0.5) : ⌈x⌉ - x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_equals_half_l1536_153667


namespace NUMINAMATH_CALUDE_alice_number_problem_l1536_153655

theorem alice_number_problem (x : ℝ) : ((x + 3) * 3 - 5) / 3 = 10 → x = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_alice_number_problem_l1536_153655


namespace NUMINAMATH_CALUDE_bus_driver_regular_rate_l1536_153671

/-- Represents the bus driver's compensation structure and work hours -/
structure BusDriverCompensation where
  regularRate : ℝ
  overtimeRate : ℝ
  regularHours : ℝ
  overtimeHours : ℝ
  totalCompensation : ℝ

/-- Theorem stating the bus driver's regular rate given the compensation conditions -/
theorem bus_driver_regular_rate 
  (comp : BusDriverCompensation)
  (h1 : comp.regularHours = 40)
  (h2 : comp.overtimeHours = 17)
  (h3 : comp.overtimeRate = comp.regularRate * 1.75)
  (h4 : comp.totalCompensation = 1116)
  (h5 : comp.totalCompensation = comp.regularRate * comp.regularHours + 
        comp.overtimeRate * comp.overtimeHours) : 
  comp.regularRate = 16 := by
  sorry

#check bus_driver_regular_rate

end NUMINAMATH_CALUDE_bus_driver_regular_rate_l1536_153671


namespace NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l1536_153625

theorem quadratic_roots_always_positive_implies_a_zero
  (a b c : ℝ)
  (h : ∀ p : ℝ, p > 0 →
    ∀ x : ℝ, a * x^2 + b * x + c + p = 0 →
      x > 0 ∧ (∃ y : ℝ, y ≠ x ∧ a * y^2 + b * y + c + p = 0 ∧ y > 0)) :
  a = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_always_positive_implies_a_zero_l1536_153625


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l1536_153689

def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3

def red_balls : ℕ := total_balls - yellow_balls - green_balls

def probability_red_ball : ℚ := red_balls / total_balls

theorem probability_of_red_ball :
  probability_red_ball = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l1536_153689


namespace NUMINAMATH_CALUDE_probability_log_integer_l1536_153692

def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 15 ∧ n = 3^k}

def is_valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ k : ℕ, b = a^k

def total_pairs : ℕ := Nat.choose 15 2

def valid_pairs : ℕ := 30

theorem probability_log_integer :
  (valid_pairs : ℚ) / total_pairs = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_log_integer_l1536_153692


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_sum_of_radii_l1536_153636

/-- For a right-angled triangle, the perimeter equals the sum of radii of inscribed and excircles -/
theorem right_triangle_perimeter_equals_sum_of_radii 
  (a b c ρ ρ_a ρ_b ρ_c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Pythagorean theorem for right-angled triangle
  (h_ρ : ρ = (a + b - c) / 2)  -- Formula for inscribed circle radius
  (h_ρ_a : ρ_a = (a + b + c) / 2 - a)  -- Formula for excircle radius opposite to side a
  (h_ρ_b : ρ_b = (a + b + c) / 2 - b)  -- Formula for excircle radius opposite to side b
  (h_ρ_c : ρ_c = (a + b + c) / 2)  -- Formula for excircle radius opposite to side c
  : a + b + c = ρ + ρ_a + ρ_b + ρ_c := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_sum_of_radii_l1536_153636


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1536_153691

/-- An ellipse with axes parallel to the coordinate axes passing through 
    the given points has a minor axis length of 4. -/
theorem ellipse_minor_axis_length : 
  ∀ (e : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ e ↔ ((x - 3/2)^2 / 3^2) + ((y - 1)^2 / b^2) = 1) →
  (0, 0) ∈ e →
  (0, 2) ∈ e →
  (3, 0) ∈ e →
  (3, 2) ∈ e →
  (3/2, 3) ∈ e →
  ∃ (b : ℝ), b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l1536_153691


namespace NUMINAMATH_CALUDE_smallest_integer_l1536_153695

theorem smallest_integer (a b : ℕ) (ha : a = 75) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 45) :
  ∃ (m : ℕ), m ≥ b ∧ m = 135 ∧ Nat.lcm a m / Nat.gcd a m = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_l1536_153695


namespace NUMINAMATH_CALUDE_least_six_digit_multiple_l1536_153637

theorem least_six_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  (12 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (23 ∣ n) ∧ (29 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100000 ∧ m < n → ¬((12 ∣ m) ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (23 ∣ m) ∧ (29 ∣ m))) :=
by
  use 120060
  sorry

end NUMINAMATH_CALUDE_least_six_digit_multiple_l1536_153637


namespace NUMINAMATH_CALUDE_eight_digit_numbers_a_eight_digit_numbers_b_eight_digit_numbers_b_start_with_1_l1536_153617

-- Define the set of digits for part a
def digits_a : Finset ℕ := {0, 1, 2}

-- Define the multiset of digits for part b
def digits_b : Multiset ℕ := {0, 0, 0, 1, 2, 2, 2, 2}

-- Define the number of digits in the numbers we're forming
def num_digits : ℕ := 8

-- Theorem for part a
theorem eight_digit_numbers_a : 
  (Finset.card digits_a ^ num_digits) - (Finset.card digits_a ^ (num_digits - 1)) = 4374 :=
sorry

-- Theorem for part b (total valid numbers)
theorem eight_digit_numbers_b : 
  (Multiset.card digits_b).factorial / ((Multiset.count 0 digits_b).factorial * (Multiset.count 2 digits_b).factorial) - 
  ((Multiset.card digits_b - 1).factorial / ((Multiset.count 0 digits_b - 1).factorial * (Multiset.count 2 digits_b).factorial)) = 175 :=
sorry

-- Theorem for part b (numbers starting with 1)
theorem eight_digit_numbers_b_start_with_1 : 
  (Multiset.card digits_b - 1).factorial / ((Multiset.count 0 digits_b).factorial * (Multiset.count 2 digits_b).factorial) = 35 :=
sorry

end NUMINAMATH_CALUDE_eight_digit_numbers_a_eight_digit_numbers_b_eight_digit_numbers_b_start_with_1_l1536_153617


namespace NUMINAMATH_CALUDE_circle_triangle_area_relation_l1536_153623

theorem circle_triangle_area_relation :
  ∀ (A B C : ℝ),
  (15 : ℝ)^2 + 20^2 = 25^2 →  -- Right triangle condition
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Areas are positive
  C ≥ A ∧ C ≥ B →  -- C is the largest area
  A + B + (1/2 * 15 * 20) = (π * 25^2) / 8 →  -- Area relation
  A + B + 150 = C :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_relation_l1536_153623


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1536_153660

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 27) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1536_153660


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1536_153642

theorem quadratic_equation_solutions (x : ℝ) :
  x^2 = 8*x - 15 →
  (∃ s p : ℝ, s = 8 ∧ p = 15 ∧
    (∀ x₁ x₂ : ℝ, x₁^2 = 8*x₁ - 15 ∧ x₂^2 = 8*x₂ - 15 → x₁ + x₂ = s ∧ x₁ * x₂ = p)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1536_153642


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1536_153626

theorem inequality_equivalence (x : ℝ) :
  (1 / x > -4 ∧ 1 / x < 3) ↔ (x > 1 / 3 ∨ x < -1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1536_153626


namespace NUMINAMATH_CALUDE_flower_bed_circumference_l1536_153682

/-- Given a square garden with a circular flower bed, prove the circumference of the flower bed -/
theorem flower_bed_circumference 
  (a p t : ℝ) 
  (h1 : a > 0) 
  (h2 : p > 0) 
  (h3 : t > 0) 
  (h4 : a = 2 * p + 14.25) 
  (h5 : ∃ s : ℝ, s > 0 ∧ a = s^2 ∧ p = 4 * s) 
  (h6 : ∃ r : ℝ, r > 0 ∧ r = s / 4 ∧ t = a + π * r^2) : 
  ∃ C : ℝ, C = 4.75 * π := by sorry

end NUMINAMATH_CALUDE_flower_bed_circumference_l1536_153682


namespace NUMINAMATH_CALUDE_houses_around_square_l1536_153610

/-- The number of houses around the square. -/
def n : ℕ := 32

/-- Maria's starting position relative to João's. -/
def m_start : ℕ := 8

/-- Proposition that the given conditions imply there are 32 houses around the square. -/
theorem houses_around_square :
  (∀ k : ℕ, (k + 5 - m_start) % n = (k + 12) % n) ∧
  (∀ k : ℕ, (k + 30 - m_start) % n = (k + 5) % n) →
  n = 32 :=
by sorry

end NUMINAMATH_CALUDE_houses_around_square_l1536_153610


namespace NUMINAMATH_CALUDE_five_power_sum_of_squares_l1536_153697

theorem five_power_sum_of_squares (n : ℕ) : ∃ (a b : ℕ), 5^n = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_five_power_sum_of_squares_l1536_153697


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1536_153604

theorem least_subtraction_for_divisibility : 
  ∃ (x : ℕ), x = 26 ∧ 
  (99 ∣ (12702 - x)) ∧ 
  (∀ (y : ℕ), y < x → ¬(99 ∣ (12702 - y))) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1536_153604


namespace NUMINAMATH_CALUDE_can_capacity_proof_l1536_153611

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 8

theorem can_capacity_proof (initial : CanContents) (final : CanContents) :
  -- Initial ratio of milk to water is 1:5
  initial.milk / initial.water = 1 / 5 →
  -- Final contents after adding 2 liters of milk
  final.milk = initial.milk + 2 ∧
  final.water = initial.water →
  -- New ratio of milk to water is 3:5
  final.milk / final.water = 3 / 5 →
  -- The can is full after adding 2 liters of milk
  final.milk + final.water = canCapacity :=
by sorry


end NUMINAMATH_CALUDE_can_capacity_proof_l1536_153611


namespace NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1536_153648

theorem fixed_point_of_linear_function (k : ℝ) : 
  2 = k * 1 - k + 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_linear_function_l1536_153648


namespace NUMINAMATH_CALUDE_canoe_production_sum_l1536_153607

theorem canoe_production_sum : ∀ (a₁ r n : ℕ), 
  a₁ = 5 → r = 3 → n = 4 → 
  a₁ * (r^n - 1) / (r - 1) = 200 := by sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l1536_153607


namespace NUMINAMATH_CALUDE_rectangular_lot_area_l1536_153603

/-- Represents a rectangular lot with given properties -/
structure RectangularLot where
  width : ℝ
  length : ℝ
  length_constraint : length = 2 * width + 35
  perimeter_constraint : 2 * (width + length) = 850

/-- The area of a rectangular lot -/
def area (lot : RectangularLot) : ℝ := lot.width * lot.length

/-- Theorem stating that a rectangular lot with the given properties has an area of 38350 square feet -/
theorem rectangular_lot_area : 
  ∀ (lot : RectangularLot), area lot = 38350 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_lot_area_l1536_153603


namespace NUMINAMATH_CALUDE_expressions_equal_thirty_l1536_153685

theorem expressions_equal_thirty : 
  (6 * 6 - 6 = 30) ∧ 
  (5 * 5 + 5 = 30) ∧ 
  (33 - 3 = 30) ∧ 
  (3^3 + 3 = 30) := by
  sorry

#check expressions_equal_thirty

end NUMINAMATH_CALUDE_expressions_equal_thirty_l1536_153685


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l1536_153633

theorem smallest_multiple_of_seven (x : ℕ) : 
  (∃ k : ℕ, x = 7 * k) ∧ 
  (x^2 > 150) ∧ 
  (x < 40) → 
  x = 14 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l1536_153633


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_l1536_153687

-- Define the parabola
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the circle equation
def circle_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + b * x + c + a * y^2 = 0

theorem parabola_circle_theorem (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (x1 x2 : ℝ), 
    parabola a b c x1 = 0 ∧ 
    parabola a b c x2 = 0 ∧ 
    x1 ≠ x2 → 
    ∀ (x y : ℝ), circle_equation a b c x y ↔ 
      (x - (x1 + x2) / 2)^2 + y^2 = ((x2 - x1) / 2)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_l1536_153687


namespace NUMINAMATH_CALUDE_fractional_equation_m_range_l1536_153668

theorem fractional_equation_m_range :
  ∀ m x : ℝ,
  (m / (1 - x) - 2 / (x - 1) = 1) →
  (x ≥ 0) →
  (x ≠ 1) →
  (m ≤ -1 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_m_range_l1536_153668


namespace NUMINAMATH_CALUDE_f_one_ge_six_l1536_153630

/-- A quadratic function f(x) = x^2 + 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 3

/-- Theorem: If f(x) is increasing on (-1, +∞), then f(1) ≥ 6 -/
theorem f_one_ge_six (a : ℝ) 
  (h : ∀ x y, -1 < x ∧ x < y → f a x < f a y) : 
  f a 1 ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_f_one_ge_six_l1536_153630


namespace NUMINAMATH_CALUDE_james_number_problem_l1536_153683

theorem james_number_problem (x : ℝ) : 3 * ((3 * x + 15) - 5) = 141 → x = 37 / 3 := by
  sorry

end NUMINAMATH_CALUDE_james_number_problem_l1536_153683


namespace NUMINAMATH_CALUDE_two_segment_trip_average_speed_l1536_153629

/-- Calculates the average speed of a two-segment trip -/
def average_speed (d1 d2 v1 v2 : ℚ) : ℚ :=
  (d1 + d2) / (d1 / v1 + d2 / v2)

theorem two_segment_trip_average_speed :
  let d1 : ℚ := 50
  let d2 : ℚ := 25
  let v1 : ℚ := 15
  let v2 : ℚ := 45
  average_speed d1 d2 v1 v2 = 675 / 35 := by
  sorry

end NUMINAMATH_CALUDE_two_segment_trip_average_speed_l1536_153629


namespace NUMINAMATH_CALUDE_rectangle_garden_length_l1536_153664

/-- The perimeter of a rectangle -/
def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: For a rectangular garden with perimeter 1800 m and breadth 400 m, the length is 500 m -/
theorem rectangle_garden_length (p b : ℝ) (h1 : p = 1800) (h2 : b = 400) :
  ∃ l : ℝ, perimeter l b = p ∧ l = 500 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_garden_length_l1536_153664


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1536_153666

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 (where a > b > 0),
    with foci F₁ and F₂, and points A and B on the ellipse satisfying
    AF₁ = 3F₁B and ∠BAF₂ = 90°, prove that the eccentricity of the ellipse
    is √2/2. -/
theorem ellipse_eccentricity (a b : ℝ) (A B F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
  (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
  (F₁.1^2 + F₁.2^2 = (a^2 - b^2)) ∧
  (F₂.1^2 + F₂.2^2 = (a^2 - b^2)) ∧
  (A - F₁ = 3 • (F₁ - B)) ∧
  ((A - B) • (A - F₂) = 0) →
  Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1536_153666


namespace NUMINAMATH_CALUDE_fraction_equality_l1536_153618

theorem fraction_equality : 
  (4 + 2/3 + 3 + 1/3) - (2 + 1/2 - 1/2) = 4 + 2/3 - (2 + 1/2) + 1/2 + 3 + 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1536_153618


namespace NUMINAMATH_CALUDE_equation_solution_l1536_153658

theorem equation_solution : 
  {x : ℝ | (x^3 - 5*x^2 + 6*x)*(x - 5) = 0} = {0, 2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1536_153658


namespace NUMINAMATH_CALUDE_largest_five_digit_congruent_to_15_mod_17_l1536_153651

theorem largest_five_digit_congruent_to_15_mod_17 :
  ∀ n : ℕ, n < 100000 → n ≡ 15 [MOD 17] → n ≤ 99977 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_congruent_to_15_mod_17_l1536_153651


namespace NUMINAMATH_CALUDE_fraction_problem_l1536_153681

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (2 * n + 4) = 3 / 7 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1536_153681


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_product_l1536_153690

theorem simplify_sqrt_sum_product (m n a b : ℝ) 
  (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hb : 0 < b) (hab : a > b)
  (hsum : a + b = m) (hprod : a * b = n) :
  Real.sqrt (m + 2 * Real.sqrt n) = Real.sqrt a + Real.sqrt b ∧
  Real.sqrt (m - 2 * Real.sqrt n) = Real.sqrt a - Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_product_l1536_153690


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l1536_153609

/-- Proves that the length of a rectangular plot is 63 meters given the specified conditions -/
theorem rectangular_plot_length : 
  ∀ (breadth length : ℝ),
  length = breadth + 26 →
  2 * (length + breadth) * 26.5 = 5300 →
  length = 63 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l1536_153609


namespace NUMINAMATH_CALUDE_sum_of_squares_l1536_153653

theorem sum_of_squares (a b c : ℝ) : 
  a * b + b * c + a * c = 131 → a + b + c = 19 → a^2 + b^2 + c^2 = 99 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1536_153653


namespace NUMINAMATH_CALUDE_function_equality_and_sine_value_l1536_153600

/-- Given a function f and angle θ, prove the equality of f and the sine of θ - 5π/6. -/
theorem function_equality_and_sine_value (ω : ℝ) (θ : ℝ) :
  (ω > 0) →
  (∀ x, 2 * Real.cos (ω / 2 * x) * Real.sin (ω / 2 * x - π / 3) + Real.sqrt 3 / 2 =
    Real.sin (2 * x - π / 3)) →
  (θ ∈ Set.Icc (-π / 6) (5 * π / 6)) →
  (2 * Real.cos (ω / 2 * (θ / 2)) * Real.sin (ω / 2 * (θ / 2) - π / 3) + Real.sqrt 3 / 2 = -3 / 5) →
  Real.sin (θ - 5 * π / 6) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_and_sine_value_l1536_153600


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l1536_153661

theorem opposite_of_negative_two_thirds :
  -(-(2/3 : ℚ)) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l1536_153661


namespace NUMINAMATH_CALUDE_largest_integer_in_range_l1536_153679

theorem largest_integer_in_range : ∃ (x : ℤ), 
  (1/4 : ℚ) < (x : ℚ)/5 ∧ (x : ℚ)/5 < 2/3 ∧ 
  ∀ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/5 ∧ (y : ℚ)/5 < 2/3 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_range_l1536_153679


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l1536_153669

theorem sum_of_four_numbers : 1256 + 2561 + 5612 + 6125 = 15554 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l1536_153669


namespace NUMINAMATH_CALUDE_table_sum_difference_l1536_153612

/-- Represents a cell in the N × N table -/
structure Cell (N : ℕ) where
  row : Fin N
  col : Fin N

/-- The rule for placing numbers in the table -/
def placeNumber (N : ℕ) (n : Fin (N^2)) : Cell N → Prop :=
  sorry

/-- The sum of numbers in a given column -/
def columnSum (N : ℕ) (col : Fin N) : ℕ :=
  sorry

/-- The sum of numbers in a given row -/
def rowSum (N : ℕ) (row : Fin N) : ℕ :=
  sorry

/-- The column containing N² -/
def lastColumn (N : ℕ) : Fin N :=
  sorry

/-- The row containing 1 -/
def firstRow (N : ℕ) : Fin N :=
  sorry

theorem table_sum_difference (N : ℕ) :
  columnSum N (lastColumn N) - rowSum N (firstRow N) = N^2 - N :=
sorry

end NUMINAMATH_CALUDE_table_sum_difference_l1536_153612


namespace NUMINAMATH_CALUDE_interval_property_l1536_153684

def f (x : ℝ) : ℝ := |x - 1|

theorem interval_property (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc a b ∧ x₂ ∈ Set.Icc a b ∧ x₁ < x₂ ∧ f x₁ ≥ f x₂) →
  a < 1 := by
  sorry

end NUMINAMATH_CALUDE_interval_property_l1536_153684


namespace NUMINAMATH_CALUDE_product_maximization_second_factor_expression_analogous_product_maximization_l1536_153650

theorem product_maximization (a b : ℝ) :
  a ≥ 0 → b ≥ 0 → a + b = 10 → a * b ≤ 25 := by sorry

theorem second_factor_expression (a b : ℝ) :
  a + b = 10 → b = 10 - a := by sorry

theorem analogous_product_maximization (x y : ℝ) :
  x ≥ 0 → y ≥ 0 → x + y = 36 → x * y ≤ 324 := by sorry

end NUMINAMATH_CALUDE_product_maximization_second_factor_expression_analogous_product_maximization_l1536_153650


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1536_153677

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ / r₂ = 1 / 3) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1536_153677


namespace NUMINAMATH_CALUDE_y_derivative_l1536_153635

noncomputable def y (x : ℝ) : ℝ := 4 * Real.arcsin (4 / (2 * x + 3)) + Real.sqrt (4 * x^2 + 12 * x - 7)

theorem y_derivative (x : ℝ) (h : 2 * x + 3 > 0) :
  deriv y x = (2 * Real.sqrt (4 * x^2 + 12 * x - 7)) / (2 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l1536_153635


namespace NUMINAMATH_CALUDE_rectangle_area_l1536_153694

theorem rectangle_area (y : ℝ) (h : y > 0) : ∃ w l : ℝ,
  w > 0 ∧ l > 0 ∧ l = 3 * w ∧ w^2 + l^2 = y^2 ∧ w * l = (3 * y^2) / 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1536_153694


namespace NUMINAMATH_CALUDE_second_monday_watching_time_l1536_153621

def day_hours : ℕ := 24

def monday_hours : ℕ := day_hours / 2
def tuesday_hours : ℕ := 4
def wednesday_hours : ℕ := day_hours / 4
def thursday_hours : ℕ := day_hours / 3
def friday_hours : ℕ := 2 * wednesday_hours
def saturday_hours : ℕ := 0

def week_total : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours + saturday_hours

def sunday_hours : ℕ := week_total / 2

def total_watched : ℕ := week_total + sunday_hours

def show_length : ℕ := 75

theorem second_monday_watching_time :
  show_length - total_watched = 12 := by sorry

end NUMINAMATH_CALUDE_second_monday_watching_time_l1536_153621


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_l1536_153644

/-- Two lines in the plane given by their equations -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_k (k : ℝ) :
  let l1 : Line := { a := k, b := -1, c := -3 }
  let l2 : Line := { a := 1, b := 2*k+3, c := -2 }
  perpendicular l1 l2 → k = -3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_l1536_153644


namespace NUMINAMATH_CALUDE_xyz_value_l1536_153606

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 9)
  (eq5 : x + y + z = 6) :
  x * y * z = -10 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l1536_153606


namespace NUMINAMATH_CALUDE_evaluate_expression_l1536_153616

theorem evaluate_expression (b : ℕ) (h : b = 2) : (b^3 * b^4) - 10 = 118 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1536_153616


namespace NUMINAMATH_CALUDE_inequality_range_l1536_153631

-- Define the inequality function
def f (x a : ℝ) : Prop := x^2 + a*x > 4*x + a - 3

-- State the theorem
theorem inequality_range (a : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  ∀ x, f x a ↔ x < -1 ∨ x > 3 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l1536_153631


namespace NUMINAMATH_CALUDE_savings_comparison_l1536_153624

theorem savings_comparison (last_year_salary : ℝ) 
  (last_year_savings_rate : ℝ) (salary_increase : ℝ) (this_year_savings_rate : ℝ) :
  last_year_savings_rate = 0.06 →
  salary_increase = 0.20 →
  this_year_savings_rate = 0.05 →
  (this_year_savings_rate * (1 + salary_increase) * last_year_salary) / 
  (last_year_savings_rate * last_year_salary) = 1 := by
  sorry

#check savings_comparison

end NUMINAMATH_CALUDE_savings_comparison_l1536_153624


namespace NUMINAMATH_CALUDE_male_non_listeners_l1536_153628

/-- Radio station survey data -/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Theorem: The number of males who do not listen to the radio station is 105 -/
theorem male_non_listeners (data : SurveyData)
  (h1 : data.total_listeners = 160)
  (h2 : data.total_non_listeners = 200)
  (h3 : data.female_listeners = 75)
  (h4 : data.male_non_listeners = 105) :
  data.male_non_listeners = 105 := by
  sorry


end NUMINAMATH_CALUDE_male_non_listeners_l1536_153628


namespace NUMINAMATH_CALUDE_miss_molly_class_size_l1536_153657

/-- The number of students in Miss Molly's class -/
def total_students : ℕ := 30

/-- The number of girls in the class -/
def num_girls : ℕ := 18

/-- The number of students who like yellow -/
def yellow_fans : ℕ := 9

/-- Theorem: The total number of students in Miss Molly's class is 30 -/
theorem miss_molly_class_size :
  (total_students / 2 = total_students - (num_girls / 3 + yellow_fans)) ∧
  (num_girls = 18) ∧
  (yellow_fans = 9) →
  total_students = 30 := by
sorry

end NUMINAMATH_CALUDE_miss_molly_class_size_l1536_153657


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1536_153662

theorem arithmetic_mean_of_fractions (x a b : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) :
  (((x + a + b) / x + (x - a - b) / x) / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l1536_153662


namespace NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_legs_l1536_153634

/-- Represents a right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of one leg -/
  a : ℝ
  /-- Length of the other leg -/
  b : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Distance from center of inscribed circle to one acute angle vertex -/
  d1 : ℝ
  /-- Distance from center of inscribed circle to other acute angle vertex -/
  d2 : ℝ
  /-- a and b are positive -/
  ha : 0 < a
  hb : 0 < b
  /-- r is positive -/
  hr : 0 < r
  /-- d1 and d2 are positive -/
  hd1 : 0 < d1
  hd2 : 0 < d2
  /-- Relationship between leg length and distance to vertex -/
  h1 : d1^2 = r^2 + (a - r)^2
  h2 : d2^2 = r^2 + (b - r)^2

/-- The main theorem -/
theorem right_triangle_with_inscribed_circle_legs
  (t : RightTriangleWithInscribedCircle)
  (h1 : t.d1 = Real.sqrt 5)
  (h2 : t.d2 = Real.sqrt 10) :
  t.a = 4 ∧ t.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_legs_l1536_153634


namespace NUMINAMATH_CALUDE_christopher_stroll_l1536_153672

theorem christopher_stroll (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 4 → time = 1.25 → distance = speed * time → distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_christopher_stroll_l1536_153672


namespace NUMINAMATH_CALUDE_min_overtakes_is_five_l1536_153678

/-- Represents the girls in the race -/
inductive Girl
| Fiona
| Gertrude
| Hannah
| India
| Janice

/-- Represents the order of girls in the race -/
def RaceOrder := List Girl

/-- The initial order of the girls in the race -/
def initial_order : RaceOrder :=
  [Girl.Fiona, Girl.Gertrude, Girl.Hannah, Girl.India, Girl.Janice]

/-- The final order of the girls in the race -/
def final_order : RaceOrder :=
  [Girl.India, Girl.Gertrude, Girl.Fiona, Girl.Janice, Girl.Hannah]

/-- Calculates the minimum number of overtakes required to transform the initial order to the final order -/
def min_overtakes (initial : RaceOrder) (final : RaceOrder) : Nat :=
  sorry

/-- Theorem stating that the minimum number of overtakes is 5 -/
theorem min_overtakes_is_five :
  min_overtakes initial_order final_order = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_overtakes_is_five_l1536_153678


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1536_153620

/-- Represents the number of wrapping paper varieties -/
def wrapping_paper : Nat := 10

/-- Represents the number of ribbon colors -/
def ribbons : Nat := 3

/-- Represents the number of gift card types -/
def gift_cards : Nat := 4

/-- Represents the number of gift tag types -/
def gift_tags : Nat := 5

/-- Calculates the total number of gift wrapping combinations -/
def total_combinations : Nat := wrapping_paper * ribbons * gift_cards * gift_tags

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem gift_wrapping_combinations :
  total_combinations = 600 := by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1536_153620


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l1536_153614

theorem roots_of_polynomials (r : ℝ) : 
  r^2 - 2*r - 1 = 0 → r^5 - 12*r^4 - 29*r - 12 = 0 := by
  sorry

#check roots_of_polynomials

end NUMINAMATH_CALUDE_roots_of_polynomials_l1536_153614


namespace NUMINAMATH_CALUDE_nine_digit_sum_exists_l1536_153601

def is_nine_digit_permutation (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  ∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → ∃ k : ℕ, n / 10^k % 10 = d

theorem nine_digit_sum_exists : ∃ a b : ℕ, 
  is_nine_digit_permutation a ∧ 
  is_nine_digit_permutation b ∧ 
  a + b = 987654321 :=
sorry

end NUMINAMATH_CALUDE_nine_digit_sum_exists_l1536_153601


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1536_153640

/-- The line equation passes through a fixed point for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11) = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1536_153640


namespace NUMINAMATH_CALUDE_p_value_l1536_153649

theorem p_value (p : ℝ) : (∀ x : ℝ, (x - 1) * (x + 2) = x^2 + p*x - 2) → p = 1 := by
  sorry

end NUMINAMATH_CALUDE_p_value_l1536_153649


namespace NUMINAMATH_CALUDE_f_is_even_l1536_153686

/-- Given a > 0 and f(x) = x^4 - a|x| + 4, f(x) is an even function -/
theorem f_is_even (a : ℝ) (ha : a > 0) : 
  let f : ℝ → ℝ := λ x => x^4 - a * |x| + 4
  ∀ x, f x = f (-x) := by sorry

end NUMINAMATH_CALUDE_f_is_even_l1536_153686


namespace NUMINAMATH_CALUDE_bee_count_l1536_153674

theorem bee_count (flower_count : ℕ) (bee_difference : ℕ) : 
  flower_count = 5 → 
  bee_difference = 2 → 
  flower_count - bee_difference = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_bee_count_l1536_153674


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l1536_153696

open Real

def isSolutionSet (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, f x < exp x ↔ x ∈ S

theorem function_inequality_solution_set
  (f : ℝ → ℝ)
  (hf_diff : Differentiable ℝ f)
  (hf_deriv : ∀ x, deriv f x < f x)
  (hf_even : ∀ x, f (x + 2) = f (-x + 2))
  (hf_init : f 0 = exp 4) :
  isSolutionSet f (Set.Ici 4) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l1536_153696


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l1536_153647

def F (p q : ℕ) : ℚ :=
  let p1 := p / 10
  let p2 := p % 10
  let q1 := q / 10
  let q2 := q % 10
  let sum := (1000 * p1 + 100 * q1 + 10 * q2 + p2) + (1000 * q1 + 100 * p1 + 10 * p2 + q2)
  (sum : ℚ) / 11

theorem two_digit_numbers_problem (m n : ℕ) 
  (hm : m ≤ 9) (hn : 1 ≤ n ∧ n ≤ 9) :
  let a := 10 + m
  let b := 10 * n + 5
  150 * F a 18 + F b 26 = 32761 →
  m + n = 12 ∨ m + n = 11 ∨ m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l1536_153647


namespace NUMINAMATH_CALUDE_x_plus_2y_squared_value_l1536_153656

theorem x_plus_2y_squared_value (x y : ℝ) :
  8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1 →
  x + 2 * y^2 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_squared_value_l1536_153656


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1536_153638

/-- The slope of the tangent line to y = x³ at x = 1 -/
def tangent_slope : ℝ := 3

/-- The line equation ax - by - 2 = 0 -/
def line_equation (a b : ℝ) (x y : ℝ) : Prop :=
  a * x - b * y - 2 = 0

/-- The curve equation y = x³ -/
def curve_equation (x y : ℝ) : Prop :=
  y = x^3

theorem perpendicular_tangents_ratio (a b : ℝ) :
  line_equation a b 1 1 ∧ 
  curve_equation 1 1 ∧
  (a / b) * tangent_slope = -1 →
  a / b = -1/3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l1536_153638


namespace NUMINAMATH_CALUDE_zongzi_probability_theorem_l1536_153615

/-- Given a set of 6 items where 2 are of type A and 4 are of type B -/
def total_items : ℕ := 6
def type_A_items : ℕ := 2
def type_B_items : ℕ := 4
def selected_items : ℕ := 3

/-- Probability of selecting at least one item of type A -/
def prob_at_least_one_A : ℚ := 4/5

/-- Probability distribution of X (number of type A items selected) -/
def prob_dist_X : List (ℕ × ℚ) := [(0, 1/5), (1, 3/5), (2, 1/5)]

/-- Mathematical expectation of X -/
def expectation_X : ℚ := 1

/-- Main theorem -/
theorem zongzi_probability_theorem :
  (total_items = type_A_items + type_B_items) →
  (prob_at_least_one_A = 4/5) ∧
  (prob_dist_X = [(0, 1/5), (1, 3/5), (2, 1/5)]) ∧
  (expectation_X = 1) := by
  sorry


end NUMINAMATH_CALUDE_zongzi_probability_theorem_l1536_153615


namespace NUMINAMATH_CALUDE_smallest_subtrahend_for_multiple_of_five_l1536_153654

theorem smallest_subtrahend_for_multiple_of_five :
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → m < n → ¬(∃ k : ℤ, 425 - m = 5 * k)) ∧ (∃ k : ℤ, 425 - n = 5 * k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_subtrahend_for_multiple_of_five_l1536_153654
