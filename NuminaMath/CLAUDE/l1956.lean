import Mathlib

namespace NUMINAMATH_CALUDE_converse_square_and_intersection_subset_l1956_195665

-- Define the proposition for statement 1
def converse_square_sum_zero (x y : ℝ) : Prop :=
  x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Define the proposition for statement 2
def intersection_subset (A B : Set α) : Prop :=
  A ∩ B = A → A ⊆ B

-- Theorem combining both statements
theorem converse_square_and_intersection_subset :
  (∀ x y : ℝ, converse_square_sum_zero x y) ∧
  (∀ A B : Set α, intersection_subset A B) :=
sorry

end NUMINAMATH_CALUDE_converse_square_and_intersection_subset_l1956_195665


namespace NUMINAMATH_CALUDE_unique_solution_set_l1956_195604

-- Define the set A
def A : Set ℝ := {a | ∃! x, (x^2 - 4) / (x + a) = 1}

-- Theorem statement
theorem unique_solution_set : A = {-17/4, -2, 2} := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_set_l1956_195604


namespace NUMINAMATH_CALUDE_cafe_tables_needed_l1956_195601

def base5ToDecimal (n : Nat) : Nat :=
  (n / 100) * 25 + ((n / 10) % 10) * 5 + (n % 10)

def customersPerTable : Nat := 3

def cafeCapacity : Nat := 123

theorem cafe_tables_needed :
  let decimalCapacity := base5ToDecimal cafeCapacity
  ⌈(decimalCapacity : ℚ) / customersPerTable⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_cafe_tables_needed_l1956_195601


namespace NUMINAMATH_CALUDE_domain_of_f_squared_l1956_195635

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_of_f_squared :
  {x : ℝ | ∃ y ∈ dom_f, x^2 = y} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_f_squared_l1956_195635


namespace NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l1956_195644

def required_average : ℚ := 85
def num_quarters : ℕ := 4
def first_quarter : ℚ := 82
def second_quarter : ℚ := 77
def third_quarter : ℚ := 78

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let sum_first_three := first_quarter + second_quarter + third_quarter
  let minimum_fourth := total_required - sum_first_three
  minimum_fourth = 103 ∧
  (first_quarter + second_quarter + third_quarter + minimum_fourth) / num_quarters ≥ required_average :=
by sorry

end NUMINAMATH_CALUDE_minimum_fourth_quarter_score_l1956_195644


namespace NUMINAMATH_CALUDE_benjamin_egg_collection_l1956_195606

/-- Proves that Benjamin collects 6 dozen eggs a day given the conditions of the problem -/
theorem benjamin_egg_collection :
  ∀ (benjamin_eggs : ℕ),
  (∃ (carla_eggs trisha_eggs : ℕ),
    carla_eggs = 3 * benjamin_eggs ∧
    trisha_eggs = benjamin_eggs - 4 ∧
    benjamin_eggs + carla_eggs + trisha_eggs = 26) →
  benjamin_eggs = 6 := by
sorry

end NUMINAMATH_CALUDE_benjamin_egg_collection_l1956_195606


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1956_195629

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1956_195629


namespace NUMINAMATH_CALUDE_prism_volume_proof_l1956_195668

/-- The volume of a right rectangular prism with face areas 28, 45, and 63 square centimeters -/
def prism_volume : ℝ := 282

theorem prism_volume_proof (x y z : ℝ) 
  (face1 : x * y = 28)
  (face2 : x * z = 45)
  (face3 : y * z = 63) :
  x * y * z = prism_volume := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_proof_l1956_195668


namespace NUMINAMATH_CALUDE_melanie_dimes_l1956_195682

/-- Calculates the total number of dimes Melanie has after receiving dimes from her parents. -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Proves that Melanie has 19 dimes in total. -/
theorem melanie_dimes : total_dimes 7 8 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1956_195682


namespace NUMINAMATH_CALUDE_jane_reading_pages_l1956_195659

/-- Calculates the number of pages Jane reads in a week -/
def pages_read_in_week (morning_pages : ℕ) (evening_pages : ℕ) (days_in_week : ℕ) : ℕ :=
  (morning_pages + evening_pages) * days_in_week

theorem jane_reading_pages : pages_read_in_week 5 10 7 = 105 := by
  sorry

end NUMINAMATH_CALUDE_jane_reading_pages_l1956_195659


namespace NUMINAMATH_CALUDE_total_cost_theorem_l1956_195634

/-- The cost of cherries per pound in yuan -/
def cherry_cost : ℝ := sorry

/-- The cost of apples per pound in yuan -/
def apple_cost : ℝ := sorry

/-- The total cost of 2 pounds of cherries and 3 pounds of apples is 58 yuan -/
axiom condition1 : 2 * cherry_cost + 3 * apple_cost = 58

/-- The total cost of 3 pounds of cherries and 2 pounds of apples is 72 yuan -/
axiom condition2 : 3 * cherry_cost + 2 * apple_cost = 72

/-- The theorem states that the total cost of 3 pounds of cherries and 3 pounds of apples is 78 yuan -/
theorem total_cost_theorem : 3 * cherry_cost + 3 * apple_cost = 78 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l1956_195634


namespace NUMINAMATH_CALUDE_line_slope_one_m_value_l1956_195617

/-- Given a line passing through points P(-2, m) and Q(m, 4) with a slope of 1, prove that m = 1 -/
theorem line_slope_one_m_value (m : ℝ) : 
  (4 - m) / (m + 2) = 1 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_one_m_value_l1956_195617


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l1956_195670

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) → n = 18 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l1956_195670


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l1956_195672

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : fridayCount = 5
  validDayCount : dayCount = 30 ∨ dayCount = 31

/-- Function to get the day of the week for a given day number -/
def getDayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that the 12th day is a Monday -/
theorem twelfth_day_is_monday (m : Month) : 
  getDayOfWeek m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l1956_195672


namespace NUMINAMATH_CALUDE_root_condition_l1956_195641

open Real

theorem root_condition (a : ℝ) :
  (∃ x : ℝ, x ≥ (exp 1) ∧ a + log x = 0) →
  a ≤ -1 ∧
  (∃ a : ℝ, a ≤ -1 ∧ ∃ x : ℝ, x ≥ (exp 1) ∧ a + log x = 0) ∧
  (∃ a : ℝ, a > -1 ∧ ∀ x : ℝ, x ≥ (exp 1) → a + log x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_root_condition_l1956_195641


namespace NUMINAMATH_CALUDE_f_even_not_odd_implies_a_gt_one_l1956_195687

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 1) + Real.sqrt (a - x^2)

theorem f_even_not_odd_implies_a_gt_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ∧ 
  (∃ x, f a x ≠ -(f a (-x))) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_f_even_not_odd_implies_a_gt_one_l1956_195687


namespace NUMINAMATH_CALUDE_cube_inequality_l1956_195690

theorem cube_inequality (x y a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l1956_195690


namespace NUMINAMATH_CALUDE_sandcastle_ratio_l1956_195602

theorem sandcastle_ratio : 
  ∀ (j : ℕ), 
    20 + 200 + j + 5 * j = 580 →
    j / 20 = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_ratio_l1956_195602


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1956_195663

theorem solution_set_inequality (x : ℝ) : 
  (x + 1/2) * (3/2 - x) ≥ 0 ↔ -1/2 ≤ x ∧ x ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1956_195663


namespace NUMINAMATH_CALUDE_special_1992_gon_exists_l1956_195685

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : Fin n → ℝ
  convex : sorry -- Condition for convexity

/-- An inscribed circle in a polygon -/
structure InscribedCircle {n : ℕ} (p : ConvexPolygon n) where
  center : ℝ × ℝ
  radius : ℝ
  touches_all_sides : sorry -- Condition that the circle touches all sides

/-- The theorem stating the existence of the special 1992-gon -/
theorem special_1992_gon_exists : ∃ (p : ConvexPolygon 1992),
  (∃ (σ : Equiv (Fin 1992) (Fin 1992)), ∀ i, p.sides i = σ i + 1) ∧
  ∃ (c : InscribedCircle p), True :=
sorry

end NUMINAMATH_CALUDE_special_1992_gon_exists_l1956_195685


namespace NUMINAMATH_CALUDE_odometer_puzzle_l1956_195637

theorem odometer_puzzle (a b c : ℕ) 
  (h1 : a ≥ 1) 
  (h2 : 100 ≤ a * b * c ∧ a * b * c ≤ 300)
  (h3 : 75 ∣ b)
  (h4 : (a * b * c) + b - a * b * c = b) :
  a^2 + b^2 + c^2 = 5635 := by
sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l1956_195637


namespace NUMINAMATH_CALUDE_count_integer_solutions_l1956_195698

/-- The number of integer values of a for which x^2 + ax + 9a = 0 has integer solutions for x -/
def integerSolutionCount : ℕ := 6

/-- The quadratic equation in question -/
def hasIntegerSolution (a : ℤ) : Prop :=
  ∃ x : ℤ, x^2 + a*x + 9*a = 0

/-- The theorem stating that there are exactly 6 integer values of a for which
    the equation x^2 + ax + 9a = 0 has integer solutions for x -/
theorem count_integer_solutions :
  (∃! (s : Finset ℤ), s.card = integerSolutionCount ∧ ∀ a : ℤ, a ∈ s ↔ hasIntegerSolution a) :=
sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l1956_195698


namespace NUMINAMATH_CALUDE_arctg_sum_eq_pi_fourth_l1956_195616

theorem arctg_sum_eq_pi_fourth (x : ℝ) (h : x > -1) : 
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctg_sum_eq_pi_fourth_l1956_195616


namespace NUMINAMATH_CALUDE_quadratic_value_theorem_l1956_195694

theorem quadratic_value_theorem (m : ℝ) (h : m^2 - 2*m - 1 = 0) :
  3*m^2 - 6*m + 2020 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_theorem_l1956_195694


namespace NUMINAMATH_CALUDE_quadratic_root_sqrt_5_minus_3_l1956_195655

theorem quadratic_root_sqrt_5_minus_3 :
  ∃ (a b c : ℚ), a ≠ 0 ∧
  (a * (Real.sqrt 5 - 3)^2 + b * (Real.sqrt 5 - 3) + c = 0) ∧
  (a = 1 ∧ b = 6 ∧ c = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_sqrt_5_minus_3_l1956_195655


namespace NUMINAMATH_CALUDE_problem_solution_l1956_195684

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : 
  x = 50 ∨ x = 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1956_195684


namespace NUMINAMATH_CALUDE_build_time_relation_l1956_195646

/-- Represents the time taken to build a cottage given the number of builders and their rate -/
def build_time (builders : ℕ) (rate : ℚ) : ℚ :=
  1 / (builders.cast * rate)

/-- Theorem stating the relationship between build times for different numbers of builders -/
theorem build_time_relation (n : ℕ) (rate : ℚ) :
  n > 0 → 6 > 0 → build_time n rate = 8 → 
  build_time 6 rate = (n.cast / 6 : ℚ) * 8 := by
  sorry

#check build_time_relation

end NUMINAMATH_CALUDE_build_time_relation_l1956_195646


namespace NUMINAMATH_CALUDE_fraction_product_sum_l1956_195621

theorem fraction_product_sum : (1/3 : ℚ) * (17/6 : ℚ) * (3/7 : ℚ) + (1/4 : ℚ) * (1/8 : ℚ) = 101/672 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_sum_l1956_195621


namespace NUMINAMATH_CALUDE_probability_is_five_twelfths_l1956_195697

/-- Represents a person with 6 differently colored blocks -/
structure Person :=
  (blocks : Fin 6 → Color)

/-- Represents the colors of blocks -/
inductive Color
  | Red | Blue | Yellow | White | Green | Purple

/-- Represents a box with placed blocks -/
structure Box :=
  (blocks : Fin 3 → Color)

/-- The probability of at least one box receiving blocks of the same color from at least two different people -/
def probability_same_color (people : Fin 3 → Person) (boxes : Fin 5 → Box) : ℚ :=
  sorry

/-- The main theorem stating that the probability is 5/12 -/
theorem probability_is_five_twelfths :
  ∃ (people : Fin 3 → Person) (boxes : Fin 5 → Box),
    probability_same_color people boxes = 5 / 12 :=
  sorry

end NUMINAMATH_CALUDE_probability_is_five_twelfths_l1956_195697


namespace NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l1956_195654

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

theorem tangent_line_at_one (a : ℝ) :
  a = -1 →
  ∃ m b : ℝ, ∀ x : ℝ, (m * (x - 1) + b = f a x) ∧ (m = -Real.log 2) :=
sorry

theorem symmetry_condition (a b : ℝ) :
  (∀ x : ℝ, f a (1/x) = f a (1/(2*b - x))) ↔ (a = 1/2 ∧ b = -1/2) :=
sorry

theorem extreme_values_condition (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≤ f a x) ↔ (0 < a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_symmetry_condition_extreme_values_condition_l1956_195654


namespace NUMINAMATH_CALUDE_cheat_sheet_distribution_l1956_195619

/-- Represents the number of pockets --/
def num_pockets : ℕ := 4

/-- Represents the number of cheat sheets --/
def num_cheat_sheets : ℕ := 6

/-- Represents the number of ways to place cheat sheets 1 and 2 --/
def ways_to_place_1_and_2 : ℕ := num_pockets

/-- Represents the number of ways to place cheat sheets 4 and 5 --/
def ways_to_place_4_and_5 : ℕ := num_pockets - 1

/-- Represents the number of ways to distribute the remaining cheat sheets --/
def ways_to_distribute_remaining : ℕ := 5

/-- Theorem stating the total number of ways to distribute the cheat sheets --/
theorem cheat_sheet_distribution :
  ways_to_place_1_and_2 * ways_to_place_4_and_5 * ways_to_distribute_remaining = 60 := by
  sorry

end NUMINAMATH_CALUDE_cheat_sheet_distribution_l1956_195619


namespace NUMINAMATH_CALUDE_rotation_180_complex_l1956_195610

def rotate_180_degrees (z : ℂ) : ℂ := -z

theorem rotation_180_complex :
  rotate_180_degrees (3 - 4*I) = -3 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_rotation_180_complex_l1956_195610


namespace NUMINAMATH_CALUDE_praveen_initial_investment_l1956_195658

-- Define the initial investment amounts
variable (P : ℚ) -- Praveen's initial investment
def H : ℚ := 8640 -- Hari's investment

-- Define the time periods
def praveen_time : ℚ := 12 -- Praveen's investment time in months
def hari_time : ℚ := 7 -- Hari's investment time in months

-- Define the profit ratio
def profit_ratio : ℚ := 2 / 3 -- Praveen's share : Hari's share

-- Theorem stating Praveen's initial investment
theorem praveen_initial_investment :
  (P * praveen_time) / (H * hari_time) = profit_ratio →
  P = 3360 := by sorry

end NUMINAMATH_CALUDE_praveen_initial_investment_l1956_195658


namespace NUMINAMATH_CALUDE_log_inequality_relationship_l1956_195622

theorem log_inequality_relationship (a b : ℝ) :
  (∀ a b, Real.log a > Real.log b → a > b) ∧
  (∃ a b, a > b ∧ ¬(Real.log a > Real.log b)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_relationship_l1956_195622


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1956_195620

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1956_195620


namespace NUMINAMATH_CALUDE_drums_per_day_l1956_195664

/-- Given that 2916 drums of grapes are filled in 9 days, 
    prove that 324 drums are filled per day. -/
theorem drums_per_day : 
  let total_drums : ℕ := 2916
  let total_days : ℕ := 9
  let drums_per_day : ℕ := total_drums / total_days
  drums_per_day = 324 := by sorry

end NUMINAMATH_CALUDE_drums_per_day_l1956_195664


namespace NUMINAMATH_CALUDE_quadratic_coefficient_positive_l1956_195686

theorem quadratic_coefficient_positive (a c : ℝ) (h₁ : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2*a*x + c
  f (-1) = 1 ∧ f (-5) = 5 → a > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_positive_l1956_195686


namespace NUMINAMATH_CALUDE_books_sold_l1956_195636

theorem books_sold (initial_books : ℕ) (new_books : ℕ) (final_books : ℕ) 
  (h1 : initial_books = 34)
  (h2 : new_books = 7)
  (h3 : final_books = 24) :
  initial_books - (initial_books - new_books - final_books) = 17 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_l1956_195636


namespace NUMINAMATH_CALUDE_next_two_numbers_after_one_l1956_195612

def square_sum (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def satisfies_condition (n : ℕ) : Prop :=
  is_perfect_square (square_sum n / n)

theorem next_two_numbers_after_one (n : ℕ) : 
  (n > 1 ∧ n < 337 → ¬satisfies_condition n) ∧
  satisfies_condition 337 ∧
  (n > 337 ∧ n < 65521 → ¬satisfies_condition n) ∧
  satisfies_condition 65521 :=
sorry

end NUMINAMATH_CALUDE_next_two_numbers_after_one_l1956_195612


namespace NUMINAMATH_CALUDE_lcm_1560_1040_l1956_195679

theorem lcm_1560_1040 : Nat.lcm 1560 1040 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1560_1040_l1956_195679


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l1956_195677

theorem factor_implies_m_value (m : ℤ) : 
  (∃ a : ℤ, ∀ x : ℤ, x^2 - m*x - 15 = (x + 3) * (x - a)) → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l1956_195677


namespace NUMINAMATH_CALUDE_unique_number_with_specific_remainders_l1956_195666

theorem unique_number_with_specific_remainders :
  ∃! m : ℕ+, 
    (m : ℤ) ≡ 8 [ZMOD 13] ∧ 
    (m : ℤ) ≡ 0 [ZMOD 15] ∧
    (m / 13 : ℕ) = (m / 15 : ℕ) ∧
    m = 60 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_specific_remainders_l1956_195666


namespace NUMINAMATH_CALUDE_tom_seashells_l1956_195651

/-- The number of seashells Tom found -/
def total_seashells (broken : ℕ) (unbroken : ℕ) : ℕ :=
  broken + unbroken

/-- Theorem stating that Tom found 7 seashells in total -/
theorem tom_seashells : total_seashells 4 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l1956_195651


namespace NUMINAMATH_CALUDE_gcd_3869_6497_l1956_195633

theorem gcd_3869_6497 : Nat.gcd 3869 6497 = 73 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3869_6497_l1956_195633


namespace NUMINAMATH_CALUDE_f_properties_l1956_195623

/-- The function f(x) -/
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b * x + 1

/-- The derivative of f(x) -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 6 * x^2 + 2 * a * x + b

/-- Theorem stating the properties of f(x) and its extreme values -/
theorem f_properties :
  ∀ a b : ℝ,
  (∀ x : ℝ, f' a b (x + 1/2) = f' a b (-x + 1/2)) →  -- f'(x) is symmetric about x = -1/2
  f' a b 1 = 0 →                                     -- f'(1) = 0
  a = 3 ∧ b = -12 ∧                                  -- Values of a and b
  f a b (-2) = 21 ∧                                  -- Local maximum
  f a b 1 = -6 ∧                                     -- Local minimum
  (∀ x : ℝ, x < -2 → f' a b x > 0) ∧                 -- f(x) increasing on (-∞, -2)
  (∀ x : ℝ, -2 < x ∧ x < 1 → f' a b x < 0) ∧         -- f(x) decreasing on (-2, 1)
  (∀ x : ℝ, x > 1 → f' a b x > 0)                    -- f(x) increasing on (1, ∞)
  := by sorry


end NUMINAMATH_CALUDE_f_properties_l1956_195623


namespace NUMINAMATH_CALUDE_jungkook_boxes_l1956_195643

/-- The number of boxes needed to hold a given number of balls -/
def boxes_needed (total_balls : ℕ) (balls_per_box : ℕ) : ℕ :=
  (total_balls + balls_per_box - 1) / balls_per_box

theorem jungkook_boxes (total_balls : ℕ) (balls_per_box : ℕ) 
  (h1 : total_balls = 10) (h2 : balls_per_box = 5) : 
  boxes_needed total_balls balls_per_box = 2 := by
sorry

end NUMINAMATH_CALUDE_jungkook_boxes_l1956_195643


namespace NUMINAMATH_CALUDE_log_equation_solution_l1956_195676

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) = 1 →
  x = b^((3 + Real.sqrt 5) / 2) ∨ x = b^((3 - Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1956_195676


namespace NUMINAMATH_CALUDE_soccer_tournament_arrangements_l1956_195614

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of matches each team plays -/
def matches_per_team : ℕ := 2

/-- The total number of possible arrangements of matches -/
def total_arrangements : ℕ := 70

/-- Theorem stating the number of possible arrangements for the given conditions -/
theorem soccer_tournament_arrangements :
  ∀ (n : ℕ) (m : ℕ),
    n = num_teams →
    m = matches_per_team →
    (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (k : ℕ), k ≤ 1) →
    (∀ (i : ℕ), i < n → ∃ (s : Finset ℕ), s.card = m ∧ ∀ (j : ℕ), j ∈ s → j < n ∧ j ≠ i) →
    total_arrangements = 70 :=
by sorry

end NUMINAMATH_CALUDE_soccer_tournament_arrangements_l1956_195614


namespace NUMINAMATH_CALUDE_males_in_band_not_orchestra_l1956_195667

/-- Represents the number of students in a group -/
structure GroupCount where
  female : ℕ
  male : ℕ

/-- Represents the counts for band, orchestra, and choir -/
structure MusicGroups where
  band : GroupCount
  orchestra : GroupCount
  choir : GroupCount
  all_three : GroupCount
  total : ℕ

def music_groups : MusicGroups := {
  band := { female := 120, male := 90 },
  orchestra := { female := 90, male := 120 },
  choir := { female := 50, male := 40 },
  all_three := { female := 30, male := 20 },
  total := 250
}

theorem males_in_band_not_orchestra (g : MusicGroups) (h : g = music_groups) :
  g.band.male - (g.band.male + g.orchestra.male + g.choir.male - g.total) = 20 := by
  sorry

end NUMINAMATH_CALUDE_males_in_band_not_orchestra_l1956_195667


namespace NUMINAMATH_CALUDE_yonderland_license_plates_l1956_195669

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits (0-9) -/
def digit_count : ℕ := 10

/-- The number of non-zero digits (1-9) -/
def non_zero_digit_count : ℕ := 9

/-- The number of letters in a license plate -/
def letter_count : ℕ := 3

/-- The number of digits in a license plate -/
def digit_position_count : ℕ := 4

/-- The total number of valid license plates in Yonderland -/
def valid_license_plate_count : ℕ :=
  alphabet_size * (alphabet_size - 1) * (alphabet_size - 2) *
  non_zero_digit_count * digit_count^(digit_position_count - 1)

theorem yonderland_license_plates :
  valid_license_plate_count = 702000000 := by
  sorry

end NUMINAMATH_CALUDE_yonderland_license_plates_l1956_195669


namespace NUMINAMATH_CALUDE_second_derivative_at_x₀_l1956_195611

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the point x₀
def x₀ : ℝ := sorry

-- Define constants a and b
def a : ℝ := sorry
def b : ℝ := sorry

-- State the theorem
theorem second_derivative_at_x₀ (h : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * Δx^2) :
  deriv (deriv f) x₀ = 2 * b := by sorry

end NUMINAMATH_CALUDE_second_derivative_at_x₀_l1956_195611


namespace NUMINAMATH_CALUDE_sarah_cupcake_ratio_l1956_195642

theorem sarah_cupcake_ratio :
  ∀ (michael_cookies sarah_initial_cupcakes sarah_final_desserts : ℕ)
    (sarah_saved_cupcakes : ℕ),
  michael_cookies = 5 →
  sarah_initial_cupcakes = 9 →
  sarah_final_desserts = 11 →
  sarah_final_desserts = sarah_initial_cupcakes - sarah_saved_cupcakes + michael_cookies →
  (sarah_saved_cupcakes : ℚ) / sarah_initial_cupcakes = 1 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sarah_cupcake_ratio_l1956_195642


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l1956_195649

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  (max (ac + bd) (max (ae + bf) (max (ag + bh) (max (ce + df) (max (cg + dh) (eg + fh)))))) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l1956_195649


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l1956_195681

theorem smallest_yellow_marbles (n : ℕ) (h1 : n > 0) 
  (h2 : n % 2 = 0) (h3 : n % 3 = 0) (h4 : 4 ≤ n) : 
  ∃ (y : ℕ), y = n - (n / 2 + n / 3 + 4) ∧ 
  (∀ (m : ℕ), m > 0 → m % 2 = 0 → m % 3 = 0 → 4 ≤ m → 
    m - (m / 2 + m / 3 + 4) ≥ 0 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l1956_195681


namespace NUMINAMATH_CALUDE_distance_between_points_l1956_195689

/-- The distance between the points (2, -1) and (-3, 6) is √74. -/
theorem distance_between_points : Real.sqrt 74 = Real.sqrt ((2 - (-3))^2 + ((-1) - 6)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1956_195689


namespace NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l1956_195652

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 + (2*k + 1) * x + 2

-- Theorem statement
theorem quadratic_roots_and_integer_case :
  (∀ k : ℝ, k ≠ 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) ∧
  (∀ k : ℕ+, (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ quadratic_equation k x₁ = 0 ∧ quadratic_equation k x₂ = 0) → k = 1) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l1956_195652


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1956_195628

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → n + (n + 1) ≥ m + (m + 1)) → 
  n + (n + 1) = 43 :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1956_195628


namespace NUMINAMATH_CALUDE_set_inclusion_equivalence_l1956_195600

-- Define set A
def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ a + 2 }

-- Define set B
def B : Set ℝ := { x | |x - 4| < 1 }

-- Theorem statement
theorem set_inclusion_equivalence (a : ℝ) : A a ⊇ B ↔ 3 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_set_inclusion_equivalence_l1956_195600


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l1956_195696

theorem gcd_of_squares_sum : Nat.gcd (130^2 + 241^2 + 352^2) (129^2 + 240^2 + 353^2 + 2^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l1956_195696


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l1956_195603

theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, 4*x + 3*y + k = 0 → y^2 = 16*x) →
  (∃! p : ℝ × ℝ, (4*(p.1) + 3*(p.2) + k = 0) ∧ (p.2)^2 = 16*(p.1)) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l1956_195603


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1956_195674

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| + |x - 7| ≥ a^2 - 3*a) → 
  a ∈ Set.Icc (-2 : ℝ) 5 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1956_195674


namespace NUMINAMATH_CALUDE_percentage_students_with_cars_l1956_195657

/-- The percentage of all students at Morse High School who have cars -/
theorem percentage_students_with_cars :
  let num_seniors : ℕ := 300
  let percent_seniors_with_cars : ℚ := 40 / 100
  let num_other_students : ℕ := 1500
  let percent_other_with_cars : ℚ := 10 / 100
  let total_students := num_seniors + num_other_students
  let seniors_with_cars := (num_seniors : ℚ) * percent_seniors_with_cars
  let others_with_cars := (num_other_students : ℚ) * percent_other_with_cars
  let total_with_cars := seniors_with_cars + others_with_cars
  (total_with_cars / total_students) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_students_with_cars_l1956_195657


namespace NUMINAMATH_CALUDE_problem_statement_l1956_195653

theorem problem_statement (m n : ℝ) (h : |m - 3| + (n + 2)^2 = 0) : m + 2*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1956_195653


namespace NUMINAMATH_CALUDE_modified_cube_painted_faces_l1956_195615

/-- Represents a cube with its 8 corner small cubes removed and its surface painted -/
structure ModifiedCube where
  size : ℕ
  corner_removed : Bool
  surface_painted : Bool

/-- Counts the number of small cubes with a given number of painted faces -/
def count_painted_faces (c : ModifiedCube) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem about the number of painted faces in a modified cube -/
theorem modified_cube_painted_faces (c : ModifiedCube) 
  (h1 : c.size > 2) 
  (h2 : c.corner_removed = true) 
  (h3 : c.surface_painted = true) : 
  (count_painted_faces c 4 = 12) ∧ 
  (count_painted_faces c 1 = 6) ∧ 
  (count_painted_faces c 0 = 1) :=
sorry

end NUMINAMATH_CALUDE_modified_cube_painted_faces_l1956_195615


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1956_195609

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1956_195609


namespace NUMINAMATH_CALUDE_total_precious_stones_l1956_195692

theorem total_precious_stones :
  let agate : ℕ := 24
  let olivine : ℕ := agate + 5
  let sapphire : ℕ := 2 * olivine
  let diamond : ℕ := olivine + 11
  let amethyst : ℕ := sapphire + diamond
  let ruby : ℕ := (5 * olivine + 1) / 2  -- Rounded up
  let garnet : ℕ := amethyst - ruby - 5
  let topaz : ℕ := garnet / 2
  agate + olivine + sapphire + diamond + amethyst + ruby + garnet + topaz = 352 := by
sorry

end NUMINAMATH_CALUDE_total_precious_stones_l1956_195692


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1956_195695

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 + x - 1 ≤ 0) ↔ (∃ x : ℝ, 2 * x^2 + x - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1956_195695


namespace NUMINAMATH_CALUDE_cherry_cost_weight_relationship_l1956_195648

/-- The relationship between the cost of cherries and their weight -/
theorem cherry_cost_weight_relationship (x y : ℝ) :
  (∀ w, w * 16 = w * (y / x)) → y = 16 * x :=
by sorry

end NUMINAMATH_CALUDE_cherry_cost_weight_relationship_l1956_195648


namespace NUMINAMATH_CALUDE_triangle_area_decomposition_l1956_195625

/-- Given a triangle with area T and a point inside it, through which lines are drawn parallel to each side,
    dividing the triangle into smaller parallelograms and triangles, with the areas of the resulting
    smaller triangles being T₁, T₂, and T₃, prove that √T₁ + √T₂ + √T₃ = √T. -/
theorem triangle_area_decomposition (T T₁ T₂ T₃ : ℝ) 
  (h₁ : T > 0) (h₂ : T₁ > 0) (h₃ : T₂ > 0) (h₄ : T₃ > 0) :
  Real.sqrt T₁ + Real.sqrt T₂ + Real.sqrt T₃ = Real.sqrt T := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_decomposition_l1956_195625


namespace NUMINAMATH_CALUDE_cindys_math_operation_l1956_195693

theorem cindys_math_operation (x : ℝ) : (x - 12) / 2 = 64 → (x - 6) / 4 = 33.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_math_operation_l1956_195693


namespace NUMINAMATH_CALUDE_solution_set_f_leq_0_range_of_m_l1956_195631

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for the solution set of f(x) ≤ 0
theorem solution_set_f_leq_0 :
  {x : ℝ | f x ≤ 0} = {x : ℝ | x ≥ 1/3 ∨ x ≤ -3} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x - 2*m^2 ≤ 4*m} = {m : ℝ | m ≤ -5/2 ∨ m ≥ 1/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_0_range_of_m_l1956_195631


namespace NUMINAMATH_CALUDE_divisible_by_nine_sequence_l1956_195618

theorem divisible_by_nine_sequence (n : ℕ) : 
  (n % 9 = 0) ∧ 
  (n + 54 ≤ 97) ∧ 
  (∀ k : ℕ, k < 7 → (n + 9 * k) % 9 = 0) →
  n = 36 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_sequence_l1956_195618


namespace NUMINAMATH_CALUDE_esteban_exercise_time_l1956_195627

/-- Proves that Esteban exercised for 10 minutes each day given the conditions. -/
theorem esteban_exercise_time :
  -- Natasha's exercise time per day in minutes
  let natasha_daily := 30
  -- Number of days Natasha exercised
  let natasha_days := 7
  -- Number of days Esteban exercised
  let esteban_days := 9
  -- Total exercise time for both in hours
  let total_hours := 5
  -- Calculate Esteban's daily exercise time in minutes
  let esteban_daily := 
    (total_hours * 60 - natasha_daily * natasha_days) / esteban_days
  -- Prove that Esteban's daily exercise time is 10 minutes
  esteban_daily = 10 := by
  sorry

end NUMINAMATH_CALUDE_esteban_exercise_time_l1956_195627


namespace NUMINAMATH_CALUDE_wall_length_is_400cm_l1956_195680

/-- Proves that the length of a wall is 400 cm given the specified conditions --/
theorem wall_length_is_400cm 
  (wall_height : ℝ) 
  (wall_width : ℝ)
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (total_bricks : ℝ)
  (h1 : wall_height = 200)
  (h2 : wall_width = 25)
  (h3 : brick_length = 25)
  (h4 : brick_width = 11.25)
  (h5 : brick_height = 6)
  (h6 : total_bricks = 1185.1851851851852)
  : ∃ (wall_length : ℝ), wall_length = 400 := by
  sorry

#check wall_length_is_400cm

end NUMINAMATH_CALUDE_wall_length_is_400cm_l1956_195680


namespace NUMINAMATH_CALUDE_dealer_purchase_problem_l1956_195639

theorem dealer_purchase_problem (total_cost : ℚ) (selling_price : ℚ) (num_sold : ℕ) (profit_percentage : ℚ) :
  total_cost = 25 →
  selling_price = 32 →
  num_sold = 12 →
  profit_percentage = 60 →
  (∃ (num_purchased : ℕ), 
    num_purchased * (selling_price / num_sold) = total_cost * (1 + profit_percentage / 100) ∧
    num_purchased = 15) :=
by sorry

end NUMINAMATH_CALUDE_dealer_purchase_problem_l1956_195639


namespace NUMINAMATH_CALUDE_teenager_age_problem_l1956_195630

theorem teenager_age_problem (a b : ℕ) (h1 : a > b) (h2 : a^2 - b^2 = 4*(a + b)) (h3 : a + b = 8*(a - b)) : a = 18 := by
  sorry

end NUMINAMATH_CALUDE_teenager_age_problem_l1956_195630


namespace NUMINAMATH_CALUDE_binaryOp_solution_l1956_195613

/-- A binary operation on positive real numbers -/
def binaryOp : (ℝ → ℝ → ℝ) := sorry

/-- The binary operation is continuous -/
axiom binaryOp_continuous : Continuous (Function.uncurry binaryOp)

/-- The binary operation is commutative -/
axiom binaryOp_comm : ∀ a b : ℝ, a > 0 → b > 0 → binaryOp a b = binaryOp b a

/-- The binary operation is distributive across multiplication -/
axiom binaryOp_distrib : ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
  binaryOp a (b * c) = (binaryOp a b) * (binaryOp a c)

/-- The binary operation satisfies 2 ⊗ 2 = 4 -/
axiom binaryOp_two_two : binaryOp 2 2 = 4

/-- The main theorem: if x ⊗ y = x for x > 1, then y = √2 -/
theorem binaryOp_solution {x y : ℝ} (hx : x > 1) (h : binaryOp x y = x) : 
  y = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_binaryOp_solution_l1956_195613


namespace NUMINAMATH_CALUDE_percentage_relation_l1956_195656

theorem percentage_relation (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A = 0.06 * C := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l1956_195656


namespace NUMINAMATH_CALUDE_canada_population_1998_l1956_195608

theorem canada_population_1998 : 
  (30.3 : ℝ) * 1000000 = 30300000 := by
  sorry

end NUMINAMATH_CALUDE_canada_population_1998_l1956_195608


namespace NUMINAMATH_CALUDE_f_sin_pi_12_l1956_195671

theorem f_sin_pi_12 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = - (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sin_pi_12_l1956_195671


namespace NUMINAMATH_CALUDE_circle_through_points_l1956_195650

/-- The general equation of a circle -/
def CircleEquation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle equation we want to prove -/
def SpecificCircle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- Theorem stating that the specific circle equation passes through the given points -/
theorem circle_through_points :
  (∀ D E F : ℝ, CircleEquation 0 0 D E F → CircleEquation 4 0 D E F → CircleEquation (-1) 1 D E F
    → ∀ x y : ℝ, CircleEquation x y D E F ↔ SpecificCircle x y) := by
  sorry

end NUMINAMATH_CALUDE_circle_through_points_l1956_195650


namespace NUMINAMATH_CALUDE_monotone_decreasing_implies_a_bound_l1956_195605

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

theorem monotone_decreasing_implies_a_bound (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y ≤ 4 → f a x > f a y) → a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_monotone_decreasing_implies_a_bound_l1956_195605


namespace NUMINAMATH_CALUDE_f_min_max_l1956_195632

-- Define the function
def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

-- State the theorem
theorem f_min_max :
  (∃ x₁ : ℝ, f x₁ = -1 ∧ ∀ x : ℝ, f x ≥ -1) ∧
  (∃ x₂ : ℝ, f x₂ = 3 ∧ ∀ x : ℝ, f x ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_f_min_max_l1956_195632


namespace NUMINAMATH_CALUDE_number_difference_and_division_l1956_195683

theorem number_difference_and_division (S L : ℕ) : 
  L - S = 8327 → L = 21 * S + 125 → S = 410 ∧ L = 8735 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_and_division_l1956_195683


namespace NUMINAMATH_CALUDE_unique_prime_in_range_l1956_195647

theorem unique_prime_in_range : ∃! (n : ℕ), 
  50 < n ∧ n < 60 ∧ 
  Nat.Prime n ∧ 
  n % 7 = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_in_range_l1956_195647


namespace NUMINAMATH_CALUDE_shirt_price_is_16_30_l1956_195662

/-- Calculates the final price of a shirt given the cost price, profit percentage, discount percentage, tax rate, and packaging fee. -/
def final_shirt_price (cost_price : ℝ) (profit_percentage : ℝ) (discount_percentage : ℝ) (tax_rate : ℝ) (packaging_fee : ℝ) : ℝ :=
  let selling_price := cost_price * (1 + profit_percentage)
  let discounted_price := selling_price * (1 - discount_percentage)
  let price_with_tax := discounted_price * (1 + tax_rate)
  price_with_tax + packaging_fee

/-- Theorem stating that the final price of the shirt is $16.30 given the specific conditions. -/
theorem shirt_price_is_16_30 :
  final_shirt_price 20 0.30 0.50 0.10 2 = 16.30 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_is_16_30_l1956_195662


namespace NUMINAMATH_CALUDE_cubic_difference_l1956_195645

theorem cubic_difference (x : ℝ) (h : x - 1/x = 3) : x^3 - 1/x^3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_l1956_195645


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l1956_195678

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.ierema + 70 = w.yuliy) ∧ (w.foma - 40 = w.yuliy)

/-- The theorem to prove -/
theorem gold_coin_distribution (w : MerchantWealth) 
  (h : problem_conditions w) : w.foma - 55 = w.ierema + 55 := by
  sorry

#check gold_coin_distribution

end NUMINAMATH_CALUDE_gold_coin_distribution_l1956_195678


namespace NUMINAMATH_CALUDE_quadratic_and_inequality_system_solution_l1956_195688

theorem quadratic_and_inequality_system_solution :
  (∃ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  (∀ x : ℝ, 3*x + 5 ≥ 2 ∧ (x - 1) / 2 < (x + 1) / 4 ↔ -1 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_and_inequality_system_solution_l1956_195688


namespace NUMINAMATH_CALUDE_parabola_count_equals_intersection_count_l1956_195691

-- Define the basic geometric objects
structure Line :=
  (a b c : ℝ)

structure Point :=
  (x y : ℝ)

structure Circle :=
  (center : Point)
  (radius : ℝ)

structure Parabola :=
  (focus : Point)
  (directrix : Line)

-- Define the given lines
def t₁ : Line := sorry
def t₂ : Line := sorry
def t₃ : Line := sorry
def e : Line := sorry

-- Define the circumcircle of the triangle formed by t₁, t₂, t₃
def circumcircle : Circle := sorry

-- Function to count intersection points between a circle and a line
def intersectionCount (c : Circle) (l : Line) : Nat := sorry

-- Function to count parabolas touching t₁, t₂, t₃ with focus on e
def parabolaCount : Nat := sorry

-- Theorem statement
theorem parabola_count_equals_intersection_count :
  parabolaCount = intersectionCount circumcircle e :=
sorry

end NUMINAMATH_CALUDE_parabola_count_equals_intersection_count_l1956_195691


namespace NUMINAMATH_CALUDE_hunters_playing_time_l1956_195624

/-- Given Hunter's playing times for football and basketball, prove the total time played in hours. -/
theorem hunters_playing_time (football_minutes basketball_minutes : ℕ) 
  (h1 : football_minutes = 60) 
  (h2 : basketball_minutes = 30) : 
  (football_minutes + basketball_minutes : ℚ) / 60 = 1.5 := by
  sorry

#check hunters_playing_time

end NUMINAMATH_CALUDE_hunters_playing_time_l1956_195624


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l1956_195675

theorem binomial_coefficient_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_four_l1956_195675


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1956_195673

theorem arithmetic_mean_problem (a b c d : ℝ) :
  (a + b + c + d + 120) / 5 = 100 →
  (a + b + c + d) / 4 = 95 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1956_195673


namespace NUMINAMATH_CALUDE_expenditure_ratio_l1956_195699

-- Define the incomes and expenditures
def uma_income : ℚ := 20000
def bala_income : ℚ := 15000
def uma_expenditure : ℚ := 15000
def bala_expenditure : ℚ := 10000
def savings : ℚ := 5000

-- Define the theorem
theorem expenditure_ratio :
  (uma_income / bala_income = 4 / 3) →
  (uma_income = 20000) →
  (uma_income - uma_expenditure = savings) →
  (bala_income - bala_expenditure = savings) →
  (uma_expenditure / bala_expenditure = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l1956_195699


namespace NUMINAMATH_CALUDE_jacket_pricing_l1956_195660

theorem jacket_pricing (x : ℝ) : 
  let marked_price : ℝ := 300
  let discount_rate : ℝ := 0.7
  let profit : ℝ := 20
  (marked_price * discount_rate - x = profit) ↔ 
  (300 * 0.7 - x = 20) :=
by sorry

end NUMINAMATH_CALUDE_jacket_pricing_l1956_195660


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1956_195640

/-- Given a cone with base radius 5 and slant height 15, prove that the central angle of the sector in the unfolded lateral surface is 120 degrees -/
theorem cone_lateral_surface_angle (base_radius : ℝ) (slant_height : ℝ) (central_angle : ℝ) : 
  base_radius = 5 → 
  slant_height = 15 → 
  central_angle * slant_height / 180 * π = 2 * π * base_radius → 
  central_angle = 120 := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l1956_195640


namespace NUMINAMATH_CALUDE_exists_m_for_all_n_l1956_195626

theorem exists_m_for_all_n (n : ℕ+) : ∃ m : ℤ, (2^(2^n.val) - 1) ∣ (m^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_for_all_n_l1956_195626


namespace NUMINAMATH_CALUDE_f_composed_with_g_l1956_195607

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x - 2

theorem f_composed_with_g : f (2 + g 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_composed_with_g_l1956_195607


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l1956_195638

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else (10 : ℝ) ^ x

-- State the theorem
theorem f_composition_negative_two :
  f (f (-2)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l1956_195638


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l1956_195661

/-- A parabola is tangent to a line if and only if their intersection equation has exactly one solution -/
axiom tangent_condition (a : ℝ) : 
  (∃! x, a * x^2 + 4 = 2 * x + 1) ↔ (∃ x, a * x^2 + 4 = 2 * x + 1 ∧ ∀ y, a * y^2 + 4 = 2 * y + 1 → y = x)

/-- The main theorem: if a parabola y = ax^2 + 4 is tangent to the line y = 2x + 1, then a = 1/3 -/
theorem parabola_tangent_line (a : ℝ) : 
  (∃! x, a * x^2 + 4 = 2 * x + 1) → a = 1/3 := by
sorry


end NUMINAMATH_CALUDE_parabola_tangent_line_l1956_195661
