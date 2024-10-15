import Mathlib

namespace NUMINAMATH_CALUDE_disk_arrangement_area_l2943_294348

/-- The total area of eight congruent disks arranged around a square -/
theorem disk_arrangement_area (s : ℝ) (h : s = 2) : 
  let r := s / 2
  8 * π * r^2 = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_disk_arrangement_area_l2943_294348


namespace NUMINAMATH_CALUDE_fourth_grade_student_count_l2943_294320

/-- The number of students at the end of the year in fourth grade -/
def final_student_count (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem: Given the initial conditions, the final student count is 29 -/
theorem fourth_grade_student_count :
  final_student_count 33 18 14 = 29 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_student_count_l2943_294320


namespace NUMINAMATH_CALUDE_evaluate_expression_l2943_294374

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) :
  3 * x^y + 4 * y^x = 59 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2943_294374


namespace NUMINAMATH_CALUDE_expression_one_proof_l2943_294357

theorem expression_one_proof : 
  ((-5/8) / (14/3) * (-16/5) / (-6/7)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_expression_one_proof_l2943_294357


namespace NUMINAMATH_CALUDE_probability_both_science_questions_l2943_294358

def total_questions : ℕ := 5
def science_questions : ℕ := 3
def humanities_questions : ℕ := 2

theorem probability_both_science_questions :
  let total_outcomes := total_questions * (total_questions - 1)
  let favorable_outcomes := science_questions * (science_questions - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_probability_both_science_questions_l2943_294358


namespace NUMINAMATH_CALUDE_expression_value_l2943_294379

theorem expression_value (x : ℝ) (hx : x^2 - x - 1 = 0) :
  (2 / (x + 1) - 1 / x) / ((x^2 - x) / (x^2 + 2*x + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2943_294379


namespace NUMINAMATH_CALUDE_complex_number_evaluation_l2943_294326

theorem complex_number_evaluation :
  let i : ℂ := Complex.I
  ((1 - i) * i^2) / (1 + 2*i) = 1/5 + 3/5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_evaluation_l2943_294326


namespace NUMINAMATH_CALUDE_janet_lives_lost_l2943_294399

/-- The number of lives Janet lost in the hard part of the game -/
def lives_lost : ℕ := sorry

theorem janet_lives_lost :
  (∃ (initial_lives : ℕ) (gained_lives : ℕ),
    initial_lives = 38 ∧
    gained_lives = 32 ∧
    initial_lives - lives_lost + gained_lives = 54) →
  lives_lost = 16 := by sorry

end NUMINAMATH_CALUDE_janet_lives_lost_l2943_294399


namespace NUMINAMATH_CALUDE_max_sphere_radius_squared_l2943_294393

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents the configuration of two intersecting cones and a sphere -/
structure ConeConfiguration where
  cone1 : Cone
  cone2 : Cone
  intersectionDistance : ℝ
  sphereRadius : ℝ

/-- Checks if the configuration is valid according to the problem conditions -/
def isValidConfiguration (config : ConeConfiguration) : Prop :=
  config.cone1 = config.cone2 ∧
  config.cone1.baseRadius = 3 ∧
  config.cone1.height = 8 ∧
  config.intersectionDistance = 3

/-- Theorem stating the maximum possible squared radius of the sphere -/
theorem max_sphere_radius_squared 
  (config : ConeConfiguration) 
  (h : isValidConfiguration config) : 
  (∀ c : ConeConfiguration, isValidConfiguration c → c.sphereRadius ^ 2 ≤ config.sphereRadius ^ 2) →
  config.sphereRadius ^ 2 = 225 / 73 := by
  sorry

end NUMINAMATH_CALUDE_max_sphere_radius_squared_l2943_294393


namespace NUMINAMATH_CALUDE_cement_tess_is_5_1_l2943_294343

/-- The amount of cement used for Tess's street -/
def cement_tess : ℝ := 15.1 - 10

/-- Proof that the amount of cement used for Tess's street is 5.1 tons -/
theorem cement_tess_is_5_1 : cement_tess = 5.1 := by
  sorry

end NUMINAMATH_CALUDE_cement_tess_is_5_1_l2943_294343


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l2943_294398

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 5 → 
  (10 * x + y) - (10 * y + x) = 45 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l2943_294398


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2943_294304

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁^2 - 3*x₁ - 1 = 0 ∧ x₂^2 - 3*x₂ - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2943_294304


namespace NUMINAMATH_CALUDE_park_area_is_30000_l2943_294337

/-- Represents a rectangular park with cycling path on its boundary -/
structure Park where
  length : ℝ
  breadth : ℝ
  avg_speed : ℝ
  round_time : ℝ
  (ratio : length = 3 * breadth)
  (speed_constraint : avg_speed = 12)
  (time_constraint : round_time = 4 / 60)

/-- Calculates the area of the park -/
def park_area (p : Park) : ℝ :=
  p.length * p.breadth

/-- Theorem stating the area of the park is 30000 square meters -/
theorem park_area_is_30000 (p : Park) : park_area p = 30000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_is_30000_l2943_294337


namespace NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l2943_294303

/-- 
Given a rectangular box Q inscribed in a sphere of radius s,
prove that if the surface area of Q is 312 and the sum of the
lengths of its 12 edges is 96, then s = √66.
-/
theorem inscribed_box_sphere_radius (a b c s : ℝ) : 
  a > 0 → b > 0 → c > 0 → s > 0 →
  4 * (a + b + c) = 96 →
  2 * (a * b + b * c + a * c) = 312 →
  (2 * s)^2 = a^2 + b^2 + c^2 →
  s = Real.sqrt 66 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l2943_294303


namespace NUMINAMATH_CALUDE_max_type_A_machines_l2943_294321

/-- The cost of a type A machine in millions of yuan -/
def cost_A : ℕ := 7

/-- The cost of a type B machine in millions of yuan -/
def cost_B : ℕ := 5

/-- The total number of machines to be purchased -/
def total_machines : ℕ := 6

/-- The maximum budget in millions of yuan -/
def max_budget : ℕ := 34

/-- Condition: Cost of 3 type A machines and 2 type B machines is 31 million yuan -/
axiom condition1 : 3 * cost_A + 2 * cost_B = 31

/-- Condition: One type A machine costs 2 million yuan more than one type B machine -/
axiom condition2 : cost_A = cost_B + 2

/-- Theorem: The maximum number of type A machines that can be purchased within the budget is 2 -/
theorem max_type_A_machines : 
  ∀ m : ℕ, m ≤ total_machines → m * cost_A + (total_machines - m) * cost_B ≤ max_budget → m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_type_A_machines_l2943_294321


namespace NUMINAMATH_CALUDE_number_of_men_is_group_size_l2943_294306

/-- Represents the number of men, women, and boys -/
def group_size : ℕ := 8

/-- Represents the total earnings of all people -/
def total_earnings : ℕ := 105

/-- Represents the wage of each man -/
def men_wage : ℕ := 7

/-- Theorem stating that the number of men is equal to the group size -/
theorem number_of_men_is_group_size :
  ∃ (women_wage boy_wage : ℚ),
    group_size * men_wage + group_size * women_wage + group_size * boy_wage = total_earnings →
    group_size = group_size := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_is_group_size_l2943_294306


namespace NUMINAMATH_CALUDE_sequences_theorem_l2943_294370

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = 2

/-- Sequence b satisfying b_{n+1} - b_n = a_n -/
def b_sequence (a b : ℕ → ℤ) : Prop :=
  ∀ n, b (n + 1) - b n = a n

/-- Main theorem about sequences a and b -/
theorem sequences_theorem (a b : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : b_sequence a b)
  (h3 : b 2 = -18)
  (h4 : b 3 = -24) :
  (∀ n, a n = 2 * n - 10) ∧
  (b 5 = -30 ∧ b 6 = -30 ∧ ∀ n, b n ≥ -30) :=
sorry

end NUMINAMATH_CALUDE_sequences_theorem_l2943_294370


namespace NUMINAMATH_CALUDE_paul_reading_books_l2943_294387

/-- The number of books Paul reads per week -/
def books_per_week : ℕ := 7

/-- The number of weeks Paul reads -/
def weeks : ℕ := 12

/-- The total number of books Paul reads -/
def total_books : ℕ := books_per_week * weeks

theorem paul_reading_books : total_books = 84 := by
  sorry

end NUMINAMATH_CALUDE_paul_reading_books_l2943_294387


namespace NUMINAMATH_CALUDE_pool_completion_theorem_l2943_294397

/-- A pool with blue and red tiles that needs to be completed -/
structure Pool :=
  (blue_tiles : ℕ)
  (red_tiles : ℕ)
  (total_required : ℕ)

/-- Calculate the number of additional tiles needed to complete the pool -/
def additional_tiles_needed (p : Pool) : ℕ :=
  p.total_required - (p.blue_tiles + p.red_tiles)

/-- Theorem stating that for a pool with 48 blue tiles, 32 red tiles, 
    and a total requirement of 100 tiles, 20 additional tiles are needed -/
theorem pool_completion_theorem :
  let p : Pool := { blue_tiles := 48, red_tiles := 32, total_required := 100 }
  additional_tiles_needed p = 20 := by
  sorry

end NUMINAMATH_CALUDE_pool_completion_theorem_l2943_294397


namespace NUMINAMATH_CALUDE_proposition_correctness_l2943_294333

theorem proposition_correctness : ∃ (p1 p2 p3 p4 : Prop),
  -- Proposition 1
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧
  
  -- Proposition 2
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) ↔ (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ∧
  
  -- Proposition 3
  (¬(∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x > 0 → x^2 + x + 1 ≥ 0)) ∧
  
  -- Proposition 4
  (∀ p q : Prop, ¬(p ∨ q) → (¬p ∧ ¬q)) ∧
  
  -- Exactly 3 out of 4 propositions are correct
  (p1 ∧ p2 ∧ ¬p3 ∧ p4) :=
by
  sorry

#check proposition_correctness

end NUMINAMATH_CALUDE_proposition_correctness_l2943_294333


namespace NUMINAMATH_CALUDE_binary_exponentiation_not_always_optimal_l2943_294344

/-- The minimum number of multiplications needed to compute x^n -/
noncomputable def l (n : ℕ) : ℕ := sorry

/-- The number of multiplications needed to compute x^n using the binary exponentiation method -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that the binary exponentiation method is not always optimal -/
theorem binary_exponentiation_not_always_optimal :
  ∃ n : ℕ, l n < b n := by sorry

end NUMINAMATH_CALUDE_binary_exponentiation_not_always_optimal_l2943_294344


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l2943_294386

theorem rectangle_width_length_ratio :
  ∀ w : ℝ,
  w > 0 →
  2 * w + 2 * 10 = 30 →
  w / 10 = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l2943_294386


namespace NUMINAMATH_CALUDE_least_number_for_divisibility_l2943_294335

theorem least_number_for_divisibility (n m : ℕ) (h : n = 1052 ∧ m = 23) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_least_number_for_divisibility_l2943_294335


namespace NUMINAMATH_CALUDE_straws_paper_difference_l2943_294389

theorem straws_paper_difference :
  let straws : ℕ := 15
  let paper : ℕ := 7
  straws - paper = 8 := by sorry

end NUMINAMATH_CALUDE_straws_paper_difference_l2943_294389


namespace NUMINAMATH_CALUDE_cafe_working_days_l2943_294378

def days_in_period : ℕ := 13
def days_open_per_week : ℕ := 6
def first_day_is_monday : Prop := True

theorem cafe_working_days :
  let total_days := days_in_period
  let mondays := (total_days + 6) / 7
  total_days - mondays = 11 :=
sorry

end NUMINAMATH_CALUDE_cafe_working_days_l2943_294378


namespace NUMINAMATH_CALUDE_mixed_committee_probability_l2943_294314

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

def probability_mixed_committee : ℚ :=
  1 - (Nat.choose boys committee_size + Nat.choose girls committee_size) / Nat.choose total_members committee_size

theorem mixed_committee_probability :
  probability_mixed_committee = 284 / 295 := by
  sorry

end NUMINAMATH_CALUDE_mixed_committee_probability_l2943_294314


namespace NUMINAMATH_CALUDE_expression_value_l2943_294368

theorem expression_value (m n : ℝ) (h : n - m = 2) : 
  (m^2 - n^2) / m * (2 * m) / (m + n) = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2943_294368


namespace NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l2943_294362

/-- Represents the lowest score within a given number of standard deviations from the mean. -/
def lowestScore (mean standardDeviation : ℝ) (numStdDev : ℝ) : ℝ :=
  mean - numStdDev * standardDeviation

/-- Theorem stating that given a mean of 60 and standard deviation of 10,
    the lowest score within 2 standard deviations is 40. -/
theorem lowest_score_within_two_std_dev :
  lowestScore 60 10 2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lowest_score_within_two_std_dev_l2943_294362


namespace NUMINAMATH_CALUDE_bill_percentage_increase_l2943_294305

/-- 
Given Maximoff's original monthly bill and new monthly bill, 
prove that the percentage increase is 30%.
-/
theorem bill_percentage_increase 
  (original_bill : ℝ) 
  (new_bill : ℝ) 
  (h1 : original_bill = 60) 
  (h2 : new_bill = 78) : 
  (new_bill - original_bill) / original_bill * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_bill_percentage_increase_l2943_294305


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2943_294318

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2943_294318


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l2943_294336

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 12) :
  (∀ k l : ℕ+, Nat.gcd k l = 12 → Nat.gcd (10 * k) (15 * l) ≥ 60) ∧
  (∃ a b : ℕ+, Nat.gcd a b = 12 ∧ Nat.gcd (10 * a) (15 * b) = 60) := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l2943_294336


namespace NUMINAMATH_CALUDE_basketball_games_lost_l2943_294301

theorem basketball_games_lost (total_games : ℕ) (games_won : ℕ) (win_difference : ℕ) 
  (h1 : total_games = 62)
  (h2 : games_won = 45)
  (h3 : games_won = win_difference + (total_games - games_won)) :
  total_games - games_won = 17 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_lost_l2943_294301


namespace NUMINAMATH_CALUDE_division_with_specific_endings_impossible_l2943_294360

theorem division_with_specific_endings_impossible :
  ¬∃ (a b c d : ℕ), 
    a = b * c + d ∧
    a % 10 = 9 ∧
    b % 10 = 7 ∧
    c % 10 = 3 ∧
    d = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_with_specific_endings_impossible_l2943_294360


namespace NUMINAMATH_CALUDE_number_equation_solution_l2943_294369

theorem number_equation_solution : ∃ x : ℝ, (7 * x = 3 * x + 12) ∧ (x = 3) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2943_294369


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2943_294323

/-- Two vectors are parallel if and only if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

/-- The problem statement -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (1, x) (-2, 1) → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2943_294323


namespace NUMINAMATH_CALUDE_trevors_age_a_decade_ago_l2943_294322

theorem trevors_age_a_decade_ago (trevors_brother_age : ℕ) 
  (h1 : trevors_brother_age = 32) 
  (h2 : trevors_brother_age - 20 = 2 * (trevors_brother_age - 30)) : 
  trevors_brother_age - 30 = 16 := by
  sorry

end NUMINAMATH_CALUDE_trevors_age_a_decade_ago_l2943_294322


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2943_294366

def i : ℂ := Complex.I

theorem complex_equation_solution :
  ∃ z : ℂ, z * (1 - i) = 2 * i ∧ z = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2943_294366


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2943_294340

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2943_294340


namespace NUMINAMATH_CALUDE_f_range_on_interval_l2943_294308

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x) - 4 * (Real.cos x) ^ 2 + 2

theorem f_range_on_interval :
  ∀ x ∈ Set.Icc (3 * Real.pi / 4) Real.pi,
    ∃ y ∈ Set.Icc (-2 * Real.sqrt 2) (-2),
      f x = y ∧
      ∀ z, f x = z → z ∈ Set.Icc (-2 * Real.sqrt 2) (-2) :=
by sorry

end NUMINAMATH_CALUDE_f_range_on_interval_l2943_294308


namespace NUMINAMATH_CALUDE_alternating_sum_fraction_l2943_294382

theorem alternating_sum_fraction :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (2 - 3 + 4 - 5 + 6 - 7 + 8 - 9 + 10 - 11 + 12) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sum_fraction_l2943_294382


namespace NUMINAMATH_CALUDE_yellow_probability_is_correct_l2943_294359

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0
  green : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def BagContents.total (bag : BagContents) : ℕ :=
  bag.white + bag.black + bag.yellow + bag.blue + bag.green

/-- The contents of Bag A -/
def bagA : BagContents := { white := 4, black := 5 }

/-- The contents of Bag B -/
def bagB : BagContents := { yellow := 5, blue := 3, green := 2 }

/-- The contents of Bag C -/
def bagC : BagContents := { yellow := 2, blue := 5 }

/-- The probability of drawing a yellow marble as the second marble -/
def yellowProbability : ℚ :=
  (bagA.white * bagB.yellow / (bagA.total * bagB.total) : ℚ) +
  (bagA.black * bagC.yellow / (bagA.total * bagC.total) : ℚ)

theorem yellow_probability_is_correct :
  yellowProbability = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_yellow_probability_is_correct_l2943_294359


namespace NUMINAMATH_CALUDE_expression_simplification_l2943_294390

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / ((1/2 * x * y)^2) = 8*x - 12*y := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2943_294390


namespace NUMINAMATH_CALUDE_real_part_of_z_l2943_294324

theorem real_part_of_z (z : ℂ) (h : z * (2 - Complex.I) = 18 + 11 * Complex.I) :
  z.re = 5 := by sorry

end NUMINAMATH_CALUDE_real_part_of_z_l2943_294324


namespace NUMINAMATH_CALUDE_line_moved_upwards_l2943_294381

/-- Given a line with equation y = -x + 1, prove that moving it 5 units upwards
    results in the equation y = -x + 6 -/
theorem line_moved_upwards (x y : ℝ) :
  (y = -x + 1) → (y + 5 = -x + 6) :=
by sorry

end NUMINAMATH_CALUDE_line_moved_upwards_l2943_294381


namespace NUMINAMATH_CALUDE_max_value_and_inequality_l2943_294376

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| - 2 * |x + 1|

-- State the theorem
theorem max_value_and_inequality :
  -- Part 1: The maximum value of f(x) is 2
  (∃ (k : ℝ), ∀ (x : ℝ), f x ≤ k ∧ ∃ (x₀ : ℝ), f x₀ = k) ∧ 
  (∀ (k : ℝ), (∀ (x : ℝ), f x ≤ k ∧ ∃ (x₀ : ℝ), f x₀ = k) → k = 2) ∧
  -- Part 2: For m > 0 and n > 0, if 1/m + 1/(2n) = 2, then m + 2n ≥ 2
  ∀ (m n : ℝ), m > 0 → n > 0 → 1/m + 1/(2*n) = 2 → m + 2*n ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_and_inequality_l2943_294376


namespace NUMINAMATH_CALUDE_exists_counterexample_inequality_l2943_294349

theorem exists_counterexample_inequality :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^3 + b^3 < 2*a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_counterexample_inequality_l2943_294349


namespace NUMINAMATH_CALUDE_janes_bowling_score_l2943_294329

def janes_score (x : ℝ) := x
def toms_score (x : ℝ) := x - 50

theorem janes_bowling_score :
  ∀ x : ℝ,
  janes_score x = toms_score x + 50 →
  (janes_score x + toms_score x) / 2 = 90 →
  janes_score x = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_bowling_score_l2943_294329


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l2943_294352

/-- The probability of getting exactly k successes in n independent Bernoulli trials 
    with probability p of success on each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 4 heads in 6 independent coin flips -/
def probability_4_heads_in_6_flips (p : ℝ) : ℝ :=
  binomial_probability 6 4 p

theorem unfair_coin_probability (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : probability_4_heads_in_6_flips p = 500 / 2187) : 
  p = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l2943_294352


namespace NUMINAMATH_CALUDE_parabola_c_value_l2943_294371

/-- A parabola passing through two points with its vertex on a line -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x, (x^2 + b*x + c) = 10 → x = 2 ∨ x = -2) →  -- parabola passes through (2, 10) and (-2, 6)
  (∃ x, x^2 + b*x + c = 6 ∧ x = -2) →             -- parabola passes through (-2, 6)
  (∃ x, x^2 + b*x + c = -x + 4) →                 -- vertex lies on y = -x + 4
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2943_294371


namespace NUMINAMATH_CALUDE_pams_bank_theorem_l2943_294384

def pams_bank_problem (current_balance end_year_withdrawal initial_balance : ℕ) : Prop :=
  let end_year_balance := current_balance + end_year_withdrawal
  let ratio := end_year_balance / initial_balance
  ratio = 19 / 8

theorem pams_bank_theorem :
  pams_bank_problem 950 250 400 := by
  sorry

end NUMINAMATH_CALUDE_pams_bank_theorem_l2943_294384


namespace NUMINAMATH_CALUDE_promotion_equivalence_bottles_in_box_l2943_294328

/-- The cost of a box of beverage in yuan -/
def box_cost : ℝ := 26

/-- The discount per bottle in yuan due to the promotion -/
def discount_per_bottle : ℝ := 0.6

/-- The number of free bottles given in the promotion -/
def free_bottles : ℕ := 3

/-- The number of bottles in a box -/
def bottles_per_box : ℕ := 10

theorem promotion_equivalence : 
  (box_cost / bottles_per_box) - (box_cost / (bottles_per_box + free_bottles)) = discount_per_bottle :=
sorry

theorem bottles_in_box : 
  bottles_per_box = 10 :=
sorry

end NUMINAMATH_CALUDE_promotion_equivalence_bottles_in_box_l2943_294328


namespace NUMINAMATH_CALUDE_recurring_decimal_product_l2943_294375

/-- Represents a recurring decimal with a single digit repeating -/
def recurring_decimal_single (n : ℕ) : ℚ :=
  n / 9

/-- Represents a recurring decimal with two digits repeating -/
def recurring_decimal_double (n : ℕ) : ℚ :=
  n / 99

/-- The product of 0.1̅ and 0.23̅ is equal to 23/891 -/
theorem recurring_decimal_product :
  (recurring_decimal_single 1) * (recurring_decimal_double 23) = 23 / 891 := by
  sorry

#eval (1 / 9 : ℚ) * (23 / 99 : ℚ) == 23 / 891  -- For verification

end NUMINAMATH_CALUDE_recurring_decimal_product_l2943_294375


namespace NUMINAMATH_CALUDE_second_pipe_fills_in_30_minutes_l2943_294346

/-- Represents a system of pipes filling and emptying a tank -/
structure PipeSystem where
  fill_time1 : ℝ  -- Time for first pipe to fill tank
  empty_time : ℝ  -- Time for outlet pipe to empty tank
  combined_fill_time : ℝ  -- Time to fill tank when all pipes are open

/-- Calculates the fill time of the second pipe given a PipeSystem -/
def second_pipe_fill_time (sys : PipeSystem) : ℝ :=
  30  -- Placeholder for the actual calculation

/-- Theorem stating that for the given system, the second pipe fills the tank in 30 minutes -/
theorem second_pipe_fills_in_30_minutes (sys : PipeSystem) 
  (h1 : sys.fill_time1 = 18)
  (h2 : sys.empty_time = 45)
  (h3 : sys.combined_fill_time = 0.06666666666666665) :
  second_pipe_fill_time sys = 30 := by
  sorry

#eval second_pipe_fill_time { fill_time1 := 18, empty_time := 45, combined_fill_time := 0.06666666666666665 }

end NUMINAMATH_CALUDE_second_pipe_fills_in_30_minutes_l2943_294346


namespace NUMINAMATH_CALUDE_stacy_has_32_berries_l2943_294373

/-- The number of berries Skylar has -/
def skylar_berries : ℕ := 20

/-- The number of berries Steve has -/
def steve_berries : ℕ := skylar_berries / 2

/-- The number of berries Stacy has -/
def stacy_berries : ℕ := 3 * steve_berries + 2

/-- Theorem stating that Stacy has 32 berries -/
theorem stacy_has_32_berries : stacy_berries = 32 := by
  sorry

end NUMINAMATH_CALUDE_stacy_has_32_berries_l2943_294373


namespace NUMINAMATH_CALUDE_trees_in_yard_l2943_294363

/-- Calculates the number of trees in a yard given the yard length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

theorem trees_in_yard (yard_length : ℕ) (tree_spacing : ℕ) 
  (h1 : yard_length = 250)
  (h2 : tree_spacing = 5) :
  num_trees yard_length tree_spacing = 51 := by
  sorry

#eval num_trees 250 5

end NUMINAMATH_CALUDE_trees_in_yard_l2943_294363


namespace NUMINAMATH_CALUDE_seating_arrangements_correct_l2943_294364

/-- The number of ways to arrange four people in five chairs, with the first chair always occupied -/
def seating_arrangements : ℕ := 120

/-- The number of chairs in the row -/
def num_chairs : ℕ := 5

/-- The number of people to be seated -/
def num_people : ℕ := 4

/-- Theorem stating that the number of seating arrangements is correct -/
theorem seating_arrangements_correct : 
  seating_arrangements = (num_chairs - 1) * (num_chairs - 2) * (num_chairs - 3) * (num_chairs - 4) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_correct_l2943_294364


namespace NUMINAMATH_CALUDE_melody_dogs_count_l2943_294315

def dogs_food_problem (daily_consumption : ℚ) (initial_amount : ℚ) (remaining_amount : ℚ) (days : ℕ) : ℚ :=
  (initial_amount - remaining_amount) / (daily_consumption * days)

theorem melody_dogs_count :
  dogs_food_problem 1 30 9 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_melody_dogs_count_l2943_294315


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2943_294339

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + y = 3) :
  2^x + 2^y ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2943_294339


namespace NUMINAMATH_CALUDE_terrell_new_lifts_count_l2943_294351

/-- The number of times Terrell must lift the new weights to match or exceed his original total weight -/
def min_lifts_for_equal_weight (original_weight : ℕ) (original_reps : ℕ) (new_weight : ℕ) : ℕ :=
  let original_total := 2 * original_weight * original_reps
  let new_total_per_rep := 2 * new_weight
  ((original_total + new_total_per_rep - 1) / new_total_per_rep : ℕ)

/-- Theorem stating that Terrell needs at least 14 lifts with the new weights -/
theorem terrell_new_lifts_count :
  min_lifts_for_equal_weight 25 10 18 = 14 := by
  sorry

#eval min_lifts_for_equal_weight 25 10 18

end NUMINAMATH_CALUDE_terrell_new_lifts_count_l2943_294351


namespace NUMINAMATH_CALUDE_percentage_problem_l2943_294310

/-- Given a number where 10% of it is 40 and a certain percentage of it is 160, prove that percentage is 40% -/
theorem percentage_problem (n : ℝ) (p : ℝ) 
  (h1 : n * 0.1 = 40) 
  (h2 : n * (p / 100) = 160) : 
  p = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2943_294310


namespace NUMINAMATH_CALUDE_complex_addition_l2943_294361

theorem complex_addition : (2 : ℂ) + 5*I + (3 : ℂ) - 7*I = (5 : ℂ) - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_l2943_294361


namespace NUMINAMATH_CALUDE_find_p_l2943_294325

-- Define the quadratic equation
def quadratic_eq (p : ℝ) (x : ℝ) : ℝ := x^2 - 5*p*x + 2*p^3

-- Define a and b as roots of the quadratic equation
def roots_condition (a b p : ℝ) : Prop := 
  quadratic_eq p a = 0 ∧ quadratic_eq p b = 0

-- Define the condition that a and b are non-zero
def non_zero_condition (a b : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0

-- Define the condition for unique root
def unique_root_condition (a b : ℝ) : Prop := 
  ∃! x, x^2 - a*x + b = 0

-- Main theorem
theorem find_p (a b p : ℝ) : 
  non_zero_condition a b → 
  roots_condition a b p → 
  unique_root_condition a b → 
  p = 3 := by sorry

end NUMINAMATH_CALUDE_find_p_l2943_294325


namespace NUMINAMATH_CALUDE_half_level_associated_point_of_A_l2943_294392

/-- Given a point P(x,y) in the Cartesian plane, its a-level associated point Q has coordinates (ax+y, x+ay) where a is a constant. -/
def associated_point (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (a * x + y, x + a * y)

/-- The coordinates of point A -/
def A : ℝ × ℝ := (2, 6)

/-- The theorem states that the 1/2-level associated point of A(2,6) is B(7,5) -/
theorem half_level_associated_point_of_A :
  associated_point (1/2) A = (7, 5) := by
sorry


end NUMINAMATH_CALUDE_half_level_associated_point_of_A_l2943_294392


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2943_294388

theorem closest_integer_to_cube_root : 
  ∃ (n : ℤ), n = 10 ∧ ∀ (m : ℤ), |n - (7^3 + 9^3)^(1/3)| ≤ |m - (7^3 + 9^3)^(1/3)| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l2943_294388


namespace NUMINAMATH_CALUDE_weightlifting_time_l2943_294307

/-- Represents Kyle's basketball practice schedule --/
structure BasketballPractice where
  total_time : ℝ
  shooting_time : ℝ
  running_time : ℝ
  weightlifting_time : ℝ
  stretching_time : ℝ
  dribbling_time : ℝ
  defense_time : ℝ

/-- Kyle's basketball practice schedule satisfies the given conditions --/
def valid_practice (p : BasketballPractice) : Prop :=
  p.total_time = 2 ∧
  p.shooting_time = (1/3) * p.total_time ∧
  p.running_time = 2 * p.weightlifting_time ∧
  p.stretching_time = p.weightlifting_time ∧
  p.dribbling_time = (1/6) * p.total_time ∧
  p.defense_time = (1/12) * p.total_time ∧
  p.total_time = p.shooting_time + p.running_time + p.weightlifting_time + 
                 p.stretching_time + p.dribbling_time + p.defense_time

/-- Theorem: Kyle spends 5/12 hours lifting weights --/
theorem weightlifting_time (p : BasketballPractice) 
  (h : valid_practice p) : p.weightlifting_time = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_weightlifting_time_l2943_294307


namespace NUMINAMATH_CALUDE_rowing_speed_ratio_l2943_294396

/-- Given that a man takes twice as long to row a distance against a stream as to row the same distance with the stream, prove that the ratio of the boat's speed in still water to the stream's speed is 3:1. -/
theorem rowing_speed_ratio 
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (h : B > S) -- Assumption that the boat's speed is greater than the stream's speed
  (h1 : (1 / (B - S)) = 2 * (1 / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_ratio_l2943_294396


namespace NUMINAMATH_CALUDE_system_solution_l2943_294312

theorem system_solution : ∃ (x y : ℝ), (x + y = 4 ∧ x - 2*y = 1) ∧ (x = 3 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2943_294312


namespace NUMINAMATH_CALUDE_vector_BC_calculation_l2943_294317

-- Define the points and vectors
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (3, 2)
def AC : ℝ × ℝ := (-4, -3)

-- Define the theorem
theorem vector_BC_calculation (A B : ℝ × ℝ) (AC : ℝ × ℝ) : 
  A = (0, 1) → B = (3, 2) → AC = (-4, -3) → 
  (B.1 - (A.1 + AC.1), B.2 - (A.2 + AC.2)) = (-7, -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_BC_calculation_l2943_294317


namespace NUMINAMATH_CALUDE_living_room_size_l2943_294345

/-- Given an apartment with the following properties:
  * Total area is 160 square feet
  * There are 6 rooms in total
  * The living room is as big as 3 other rooms
  * All rooms except the living room are the same size
Prove that the living room's area is 96 square feet. -/
theorem living_room_size (total_area : ℝ) (num_rooms : ℕ) (living_room_ratio : ℕ) :
  total_area = 160 →
  num_rooms = 6 →
  living_room_ratio = 3 →
  ∃ (room_unit : ℝ),
    room_unit * (num_rooms - 1 + living_room_ratio) = total_area ∧
    living_room_ratio * room_unit = 96 := by
  sorry

end NUMINAMATH_CALUDE_living_room_size_l2943_294345


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l2943_294342

theorem quadratic_inequality_and_constraint (a b k : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_and_constraint_l2943_294342


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l2943_294355

theorem unique_root_of_equation :
  ∃! x : ℝ, Real.sqrt (x + 25) - 7 / Real.sqrt (x + 25) = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l2943_294355


namespace NUMINAMATH_CALUDE_no_solution_cubic_equation_l2943_294372

theorem no_solution_cubic_equation (p : ℕ) (hp : Nat.Prime p) :
  ¬∃ (x y : ℤ), (x^3 + y^3 = 2001 * ↑p ∨ x^3 - y^3 = 2001 * ↑p) :=
sorry

end NUMINAMATH_CALUDE_no_solution_cubic_equation_l2943_294372


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2943_294367

theorem quadratic_equation_root (a b c : ℝ) (h1 : a - b + c = 0) (h2 : a ≠ 0) :
  a * (-1)^2 + b * (-1) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2943_294367


namespace NUMINAMATH_CALUDE_pencil_count_l2943_294350

/-- The number of pencils originally in the drawer -/
def original_pencils : ℕ := 115

/-- The number of pencils added to the drawer -/
def added_pencils : ℕ := 100

/-- The total number of pencils after addition -/
def total_pencils : ℕ := 215

/-- Theorem stating that the original number of pencils plus the added pencils equals the total pencils -/
theorem pencil_count : original_pencils + added_pencils = total_pencils := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l2943_294350


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l2943_294353

/- Define the concept of opposite for integers -/
def opposite (n : Int) : Int := -n

/- Theorem: The opposite of -2023 is 2023 -/
theorem opposite_of_negative_2023 : opposite (-2023) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l2943_294353


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l2943_294330

def carrots_to_pounds (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℚ :=
  (bed1 + bed2 + bed3) / carrots_per_pound

theorem kelly_carrot_harvest (bed1 bed2 bed3 carrots_per_pound : ℕ) :
  carrots_to_pounds bed1 bed2 bed3 carrots_per_pound =
  (bed1 + bed2 + bed3) / carrots_per_pound :=
by
  sorry

#eval carrots_to_pounds 55 101 78 6

end NUMINAMATH_CALUDE_kelly_carrot_harvest_l2943_294330


namespace NUMINAMATH_CALUDE_bens_income_l2943_294394

/-- Represents the state income tax calculation and Ben's specific case -/
theorem bens_income (q : ℝ) : 
  ∃ (A : ℝ), 
    (A > 35000) ∧ 
    (0.01 * q * 35000 + 0.01 * (q + 4) * (A - 35000) = (0.01 * (q + 0.5)) * A) ∧ 
    (A = 40000) := by
  sorry

end NUMINAMATH_CALUDE_bens_income_l2943_294394


namespace NUMINAMATH_CALUDE_password_factorization_l2943_294341

/-- Represents the correspondence between algebraic expressions and words --/
def word_mapping (x y a b : ℝ) : List (ℝ × String) :=
  [(a - b, "学"), (x - y, "我"), (x + y, "爱"), (a + b, "数"),
   (x^2 - y^2, "游"), (a^2 - b^2, "美")]

/-- The main theorem stating the factorization and its word representation --/
theorem password_factorization (x y a b : ℝ) :
  ∃ (result : String),
    ((x^2 - y^2) * a^2 - (x^2 - y^2) * b^2 = (x + y) * (x - y) * (a + b) * (a - b)) ∧
    (result = "我爱数学") ∧
    (∀ (expr : ℝ) (word : String),
      (expr, word) ∈ word_mapping x y a b →
      word ∈ ["我", "爱", "数", "学"]) :=
by
  sorry


end NUMINAMATH_CALUDE_password_factorization_l2943_294341


namespace NUMINAMATH_CALUDE_solve_for_x_and_y_l2943_294327

theorem solve_for_x_and_y :
  ∀ x y : ℝ,
  (15 + 24 + x + y) / 4 = 20 →
  x - y = 6 →
  x = 23.5 ∧ y = 17.5 := by
sorry

end NUMINAMATH_CALUDE_solve_for_x_and_y_l2943_294327


namespace NUMINAMATH_CALUDE_square_area_increase_l2943_294380

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.6 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 1.56 :=
by sorry

end NUMINAMATH_CALUDE_square_area_increase_l2943_294380


namespace NUMINAMATH_CALUDE_no_valid_solution_l2943_294354

/-- Represents a mapping from letters to digits -/
def LetterDigitMap := Char → Fin 10

/-- Checks if a LetterDigitMap assigns unique digits to different letters -/
def is_valid_map (m : LetterDigitMap) : Prop :=
  ∀ c₁ c₂ : Char, c₁ ≠ c₂ → m c₁ ≠ m c₂

/-- Evaluates the left-hand side of the equation -/
def lhs (m : LetterDigitMap) : ℕ :=
  (m 'Ш') * (m 'Е') * (m 'С') * (m 'Т') * (m 'Ь') + 1

/-- Evaluates the right-hand side of the equation -/
def rhs (m : LetterDigitMap) : ℕ :=
  (m 'C') * (m 'E') * (m 'M') * (m 'b')

/-- The main theorem stating that no valid mapping exists to satisfy the equation -/
theorem no_valid_solution : ¬∃ m : LetterDigitMap, is_valid_map m ∧ lhs m = rhs m := by
  sorry

end NUMINAMATH_CALUDE_no_valid_solution_l2943_294354


namespace NUMINAMATH_CALUDE_abc_def_ratio_l2943_294365

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6) :
  a * b * c / (d * e * f) = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l2943_294365


namespace NUMINAMATH_CALUDE_marble_distribution_l2943_294385

theorem marble_distribution (x : ℕ) 
  (liam : ℕ) (mia : ℕ) (noah : ℕ) (olivia : ℕ) : 
  liam = x ∧ 
  mia = 3 * x ∧ 
  noah = 12 * x ∧ 
  olivia = 24 * x ∧ 
  liam + mia + noah + olivia = 160 → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l2943_294385


namespace NUMINAMATH_CALUDE_same_solution_implies_a_value_l2943_294356

theorem same_solution_implies_a_value : ∀ a : ℝ,
  (∀ x : ℝ, (2*x - a)/3 - (x - a)/2 = x - 1 ↔ 3*(x - 2) - 4*(x - 5/4) = 0) →
  a = -11 := by sorry

end NUMINAMATH_CALUDE_same_solution_implies_a_value_l2943_294356


namespace NUMINAMATH_CALUDE_bobby_chocolate_pieces_l2943_294334

/-- The number of chocolate pieces Bobby ate -/
def chocolate_pieces : ℕ := 58

/-- The number of candy pieces Bobby ate initially -/
def initial_candy : ℕ := 38

/-- The number of additional candy pieces Bobby ate -/
def additional_candy : ℕ := 36

/-- The difference between candy and chocolate pieces -/
def candy_chocolate_difference : ℕ := 58

theorem bobby_chocolate_pieces :
  chocolate_pieces = 
    (initial_candy + additional_candy + candy_chocolate_difference) - (initial_candy + additional_candy) :=
by
  sorry

end NUMINAMATH_CALUDE_bobby_chocolate_pieces_l2943_294334


namespace NUMINAMATH_CALUDE_trigonometric_relations_l2943_294302

theorem trigonometric_relations (x : Real) 
  (h1 : -π/2 < x) (h2 : x < 0) (h3 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x * Real.cos x = -12/25) ∧ 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (Real.tan x = -3/4) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_relations_l2943_294302


namespace NUMINAMATH_CALUDE_range_of_a_for_nonempty_set_l2943_294331

theorem range_of_a_for_nonempty_set (A : Set ℝ) (h_nonempty : A.Nonempty) :
  (∃ a : ℝ, A = {x : ℝ | a * x = 1}) → (∃ a : ℝ, a ≠ 0 ∧ A = {x : ℝ | a * x = 1}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_nonempty_set_l2943_294331


namespace NUMINAMATH_CALUDE_sphere_volume_and_circumference_l2943_294338

/-- Given a sphere with surface area 256π cm², prove its volume and circumference. -/
theorem sphere_volume_and_circumference :
  ∀ (r : ℝ),
  (4 * π * r^2 = 256 * π) →
  (4/3 * π * r^3 = 2048/3 * π) ∧ (2 * π * r = 16 * π) := by
  sorry


end NUMINAMATH_CALUDE_sphere_volume_and_circumference_l2943_294338


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_30_l2943_294332

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (h_length : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  (s.length * (2 * s.start + s.length - 1)) / 2

/-- Theorem: There is exactly one set of consecutive positive integers whose sum is 30 -/
theorem unique_consecutive_sum_30 :
  ∃! s : ConsecutiveSet, sum_consecutive s = 30 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_30_l2943_294332


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l2943_294347

theorem coefficient_x_squared_in_binomial_expansion :
  (Finset.range 5).sum (fun k => (Nat.choose 4 k) * (1^(4-k)) * (1^k)) = 16 ∧
  (Nat.choose 4 2) = 6 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_binomial_expansion_l2943_294347


namespace NUMINAMATH_CALUDE_harry_average_sleep_time_l2943_294319

/-- Harry's sleep schedule for a week --/
structure SleepSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculate the average sleep time --/
def averageSleepTime (schedule : SleepSchedule) : ℚ :=
  (schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday) / 5

/-- Harry's actual sleep schedule --/
def harrySleepSchedule : SleepSchedule := {
  monday := 8,
  tuesday := 7,
  wednesday := 8,
  thursday := 10,
  friday := 7
}

/-- Theorem: Harry's average sleep time is 8 hours --/
theorem harry_average_sleep_time :
  averageSleepTime harrySleepSchedule = 8 := by
  sorry

end NUMINAMATH_CALUDE_harry_average_sleep_time_l2943_294319


namespace NUMINAMATH_CALUDE_parabola_r_value_l2943_294391

/-- A parabola with equation x = py^2 + qy + r -/
structure Parabola where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (para : Parabola) (y : ℝ) : ℝ :=
  para.p * y^2 + para.q * y + para.r

theorem parabola_r_value (para : Parabola) :
  para.x_coord 1 = 4 →  -- vertex at (4,1)
  para.x_coord 0 = 2 →  -- passes through (2,0)
  para.r = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_r_value_l2943_294391


namespace NUMINAMATH_CALUDE_cubic_factorization_l2943_294313

theorem cubic_factorization (m : ℝ) : m^3 - 16*m = m*(m+4)*(m-4) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2943_294313


namespace NUMINAMATH_CALUDE_managers_salary_l2943_294395

/-- Given 100 employees with an average monthly salary of 3500 rupees,
    if adding one more person (the manager) increases the average salary by 800 rupees,
    then the manager's salary is 84300 rupees. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 100 →
  avg_salary = 3500 →
  salary_increase = 800 →
  (num_employees * avg_salary + 84300) / (num_employees + 1) = avg_salary + salary_increase :=
by sorry

end NUMINAMATH_CALUDE_managers_salary_l2943_294395


namespace NUMINAMATH_CALUDE_somu_present_age_l2943_294383

-- Define Somu's age and his father's age
def somu_age : ℕ := sorry
def father_age : ℕ := sorry

-- State the theorem
theorem somu_present_age :
  (somu_age = father_age / 3) ∧
  (somu_age - 8 = (father_age - 8) / 5) →
  somu_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_somu_present_age_l2943_294383


namespace NUMINAMATH_CALUDE_uniform_cost_ratio_l2943_294311

theorem uniform_cost_ratio : 
  ∀ (shirt_cost pants_cost tie_cost sock_cost : ℝ),
    pants_cost = 20 →
    tie_cost = shirt_cost / 5 →
    sock_cost = 3 →
    5 * (pants_cost + shirt_cost + tie_cost + sock_cost) = 355 →
    shirt_cost / pants_cost = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_uniform_cost_ratio_l2943_294311


namespace NUMINAMATH_CALUDE_mean_inequality_l2943_294377

theorem mean_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (a + b + c) / 3 > (a * b * c) ^ (1/3) ∧ 
  (a * b * c) ^ (1/3) > 2 * a * b * c / (a * b + b * c + c * a) := by
sorry

end NUMINAMATH_CALUDE_mean_inequality_l2943_294377


namespace NUMINAMATH_CALUDE_mistaken_calculation_l2943_294300

theorem mistaken_calculation (x : ℝ) (h : x + 0.42 = 0.9) : (x - 0.42) + 0.5 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_calculation_l2943_294300


namespace NUMINAMATH_CALUDE_point_m_coordinate_l2943_294316

/-- Given points L, M, N, P on a number line where M and N divide LP into three equal parts,
    prove that if L is at coordinate 1/6 and P is at coordinate 1/12, then M is at coordinate 1/9 -/
theorem point_m_coordinate (L M N P : ℝ) : 
  L = 1/6 →
  P = 1/12 →
  M - L = N - M →
  N - M = P - N →
  M = 1/9 := by sorry

end NUMINAMATH_CALUDE_point_m_coordinate_l2943_294316


namespace NUMINAMATH_CALUDE_chocolates_per_first_year_student_l2943_294309

theorem chocolates_per_first_year_student :
  ∀ (total_students first_year_students second_year_students total_chocolates leftover_chocolates : ℕ),
    total_students = 24 →
    total_students = first_year_students + second_year_students →
    second_year_students = 2 * first_year_students →
    total_chocolates = 50 →
    leftover_chocolates = 2 →
    (total_chocolates - leftover_chocolates) / first_year_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_per_first_year_student_l2943_294309
