import Mathlib

namespace NUMINAMATH_CALUDE_parallel_plane_sufficient_not_necessary_l1193_119315

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Main theorem
theorem parallel_plane_sufficient_not_necessary
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_subset_α : subset m α)
  (h_n_subset_α : subset n α) :
  (∀ l : Line, subset l α → parallel_line_plane l β) ∧
  ∃ m n : Line, subset m α ∧ subset n α ∧ 
    parallel_line_plane m β ∧ parallel_line_plane n β ∧
    ¬ parallel_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_sufficient_not_necessary_l1193_119315


namespace NUMINAMATH_CALUDE_school_girls_count_l1193_119382

theorem school_girls_count (total_students : ℕ) (boys_girls_difference : ℕ) 
  (h1 : total_students = 1250)
  (h2 : boys_girls_difference = 124) : 
  ∃ (girls : ℕ), girls = 563 ∧ 
  girls + (girls + boys_girls_difference) = total_students :=
sorry

end NUMINAMATH_CALUDE_school_girls_count_l1193_119382


namespace NUMINAMATH_CALUDE_gcd_problem_l1193_119345

/-- The greatest common divisor of (123^2 + 235^2 + 347^2) and (122^2 + 234^2 + 348^2) is 1 -/
theorem gcd_problem : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l1193_119345


namespace NUMINAMATH_CALUDE_min_stamps_theorem_l1193_119342

/-- The minimum number of stamps needed to make 48 cents using only 5 cent and 7 cent stamps -/
def min_stamps : ℕ := 8

/-- The value of stamps in cents -/
def total_value : ℕ := 48

/-- Represents a combination of 5 cent and 7 cent stamps -/
structure StampCombination where
  five_cent : ℕ
  seven_cent : ℕ

/-- Calculates the total value of a stamp combination -/
def combination_value (c : StampCombination) : ℕ :=
  5 * c.five_cent + 7 * c.seven_cent

/-- Calculates the total number of stamps in a combination -/
def total_stamps (c : StampCombination) : ℕ :=
  c.five_cent + c.seven_cent

/-- Predicate for a valid stamp combination that sums to the total value -/
def is_valid_combination (c : StampCombination) : Prop :=
  combination_value c = total_value

theorem min_stamps_theorem :
  ∃ (c : StampCombination), is_valid_combination c ∧
  (∀ (d : StampCombination), is_valid_combination d → total_stamps c ≤ total_stamps d) ∧
  total_stamps c = min_stamps :=
sorry

end NUMINAMATH_CALUDE_min_stamps_theorem_l1193_119342


namespace NUMINAMATH_CALUDE_inequalities_proof_l1193_119373

theorem inequalities_proof (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) : 
  (a * b > a * c) ∧ (a * c < b * c) ∧ (a + c < b + c) ∧ (c / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l1193_119373


namespace NUMINAMATH_CALUDE_call_center_fraction_l1193_119309

/-- Represents the fraction of calls processed by team B given the conditions of the problem -/
theorem call_center_fraction (team_a team_b : ℕ) (calls_a calls_b : ℝ) : 
  team_a = (5 : ℝ) / 8 * team_b →
  calls_a = (1 : ℝ) / 5 * calls_b →
  (team_b * calls_b) / (team_a * calls_a + team_b * calls_b) = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_call_center_fraction_l1193_119309


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1193_119305

def i : ℂ := Complex.I

theorem complex_equation_solution (a : ℝ) :
  (2 + a * i) / (1 + Real.sqrt 2 * i) = -Real.sqrt 2 * i →
  a = -Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1193_119305


namespace NUMINAMATH_CALUDE_owen_june_burger_expense_l1193_119329

/-- The amount Owen spent on burgers in June -/
def owen_burger_expense (burgers_per_day : ℕ) (burger_cost : ℕ) (days_in_june : ℕ) : ℕ :=
  burgers_per_day * days_in_june * burger_cost

/-- Theorem stating that Owen's burger expense in June is 720 dollars -/
theorem owen_june_burger_expense :
  owen_burger_expense 2 12 30 = 720 :=
by sorry

end NUMINAMATH_CALUDE_owen_june_burger_expense_l1193_119329


namespace NUMINAMATH_CALUDE_median_triangle_theorem_l1193_119395

/-- Given a triangle ABC with area 1 and medians s_a, s_b, s_c, there exists a triangle
    with sides s_a, s_b, s_c, and its area is 4/3 times the area of triangle ABC. -/
theorem median_triangle_theorem (A B C : ℝ × ℝ) (s_a s_b s_c : ℝ) :
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  let median_a := ((B.1 + C.1) / 2 - A.1, (B.2 + C.2) / 2 - A.2)
  let median_b := ((A.1 + C.1) / 2 - B.1, (A.2 + C.2) / 2 - B.2)
  let median_c := ((A.1 + B.1) / 2 - C.1, (A.2 + B.2) / 2 - C.2)
  triangle_area = 1 ∧
  s_a = Real.sqrt (median_a.1^2 + median_a.2^2) ∧
  s_b = Real.sqrt (median_b.1^2 + median_b.2^2) ∧
  s_c = Real.sqrt (median_c.1^2 + median_c.2^2) →
  ∃ (D E F : ℝ × ℝ),
    let new_triangle_area := abs ((D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2)) / 2)
    Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) = s_a ∧
    Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2) = s_b ∧
    Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = s_c ∧
    new_triangle_area = 4/3 * triangle_area := by
  sorry


end NUMINAMATH_CALUDE_median_triangle_theorem_l1193_119395


namespace NUMINAMATH_CALUDE_right_triangle_with_constraints_l1193_119364

/-- A right-angled triangle with perimeter 5 and shortest altitude 1 has side lengths 5/3, 5/4, and 25/12. -/
theorem right_triangle_with_constraints (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  a^2 + b^2 = c^2 →  -- right-angled triangle (Pythagorean theorem)
  a + b + c = 5 →  -- perimeter is 5
  min (a*b/c) (min (b*c/a) (c*a/b)) = 1 →  -- shortest altitude is 1
  ((a = 5/3 ∧ b = 5/4 ∧ c = 25/12) ∨ (a = 5/4 ∧ b = 5/3 ∧ c = 25/12)) := by
sorry


end NUMINAMATH_CALUDE_right_triangle_with_constraints_l1193_119364


namespace NUMINAMATH_CALUDE_square_vector_properties_l1193_119389

/-- Given a square ABCD with side length 2 and vectors a and b satisfying the given conditions,
    prove that a · b = 2 and (b - 4a) ⊥ b -/
theorem square_vector_properties (a b : ℝ × ℝ) :
  let A := (0, 0)
  let B := (2, 0)
  let C := (2, 2)
  let D := (0, 2)
  let AB := B - A
  let BC := C - B
  AB = 2 • a →
  BC = b - 2 • a →
  a • b = 2 ∧ (b - 4 • a) • b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_vector_properties_l1193_119389


namespace NUMINAMATH_CALUDE_speed_conversion_l1193_119358

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph_factor : ℝ := 3.6

/-- Given speed in meters per second -/
def given_speed_mps : ℝ := 45

/-- Theorem: Converting 45 meters per second to kilometers per hour results in 162 km/h -/
theorem speed_conversion :
  given_speed_mps * mps_to_kmph_factor = 162 := by sorry

end NUMINAMATH_CALUDE_speed_conversion_l1193_119358


namespace NUMINAMATH_CALUDE_candy_distribution_count_l1193_119361

/-- The number of ways to distribute n distinct items among k bags, where each bag must receive at least one item. -/
def distribute (n k : ℕ) : ℕ := k^n - k * ((k-1)^n - (k-1))

/-- The number of ways to distribute 9 distinct pieces of candy among 3 bags, where each bag must receive at least one piece of candy. -/
def candy_distribution : ℕ := distribute 9 3

theorem candy_distribution_count : candy_distribution = 18921 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_count_l1193_119361


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l1193_119391

theorem simultaneous_equations_solution :
  ∀ x y : ℝ, 
    (2 * x - 3 * y = 0.4 * (x + y)) →
    (5 * y = 1.2 * x) →
    (x = 0 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l1193_119391


namespace NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l1193_119323

theorem min_value_of_a2_plus_b2 (a b : ℝ) :
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) →
  (∀ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) → a'^2 + b'^2 ≥ 4/5) ∧
  (∃ a' b' : ℝ, (∃ x : ℝ, x^4 + a'*x^3 + b'*x^2 + a'*x + 1 = 0) ∧ a'^2 + b'^2 = 4/5) :=
by sorry


end NUMINAMATH_CALUDE_min_value_of_a2_plus_b2_l1193_119323


namespace NUMINAMATH_CALUDE_investment_rate_calculation_investment_rate_proof_l1193_119368

/-- Calculates the required interest rate for the remaining investment --/
theorem investment_rate_calculation 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (first_rate : ℝ) 
  (second_rate : ℝ) 
  (desired_income : ℝ) : ℝ :=
  let remaining_investment := total_investment - first_investment - second_investment
  let first_income := first_investment * first_rate / 100
  let second_income := second_investment * second_rate / 100
  let remaining_income := desired_income - first_income - second_income
  let required_rate := remaining_income / remaining_investment * 100
  required_rate

/-- Proves that the required interest rate is approximately 7.05% --/
theorem investment_rate_proof 
  (h1 : total_investment = 15000)
  (h2 : first_investment = 6000)
  (h3 : second_investment = 4500)
  (h4 : first_rate = 3)
  (h5 : second_rate = 4.5)
  (h6 : desired_income = 700) :
  ∃ ε > 0, |investment_rate_calculation total_investment first_investment second_investment first_rate second_rate desired_income - 7.05| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_investment_rate_calculation_investment_rate_proof_l1193_119368


namespace NUMINAMATH_CALUDE_increasing_quadratic_l1193_119307

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 5

-- State the theorem
theorem increasing_quadratic :
  ∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > x₁ → f x₂ > f x₁ :=
by sorry

end NUMINAMATH_CALUDE_increasing_quadratic_l1193_119307


namespace NUMINAMATH_CALUDE_textbook_completion_date_l1193_119388

/-- Represents the number of problems solved on a given day -/
def problems_solved (day : ℕ) : ℕ → ℕ
| 0 => day + 1  -- September 6
| n + 1 => day - n  -- Subsequent days

/-- Calculates the total problems solved up to a given day -/
def total_solved (day : ℕ) : ℕ :=
  (List.range (day + 1)).map (problems_solved day) |>.sum

theorem textbook_completion_date 
  (total_problems : ℕ) 
  (problems_left_day3 : ℕ) 
  (h1 : total_problems = 91)
  (h2 : problems_left_day3 = 46)
  (h3 : total_solved 2 = total_problems - problems_left_day3) :
  total_solved 6 = total_problems := by
  sorry

#eval total_solved 6  -- Should output 91

end NUMINAMATH_CALUDE_textbook_completion_date_l1193_119388


namespace NUMINAMATH_CALUDE_zeros_before_nonzero_digit_l1193_119363

theorem zeros_before_nonzero_digit (n : ℕ) (m : ℕ) : 
  (Nat.log 10 (2^n * 5^m)).pred = n.max m := by sorry

end NUMINAMATH_CALUDE_zeros_before_nonzero_digit_l1193_119363


namespace NUMINAMATH_CALUDE_triangle_properties_l1193_119360

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b + c = 3 →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  (∃ (R : ℝ), R > 0 ∧ R * (a + b + c) = a * b * Real.sin C) →
  C = π / 3 ∧
  (∀ (S : ℝ), S = π * R^2 → S ≤ π / 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1193_119360


namespace NUMINAMATH_CALUDE_trigonometric_inequalities_l1193_119377

theorem trigonometric_inequalities (α β γ : ℝ) : 
  (|Real.cos (α + β)| ≤ |Real.cos α| + |Real.sin β|) ∧ 
  (|Real.sin (α + β)| ≤ |Real.cos α| + |Real.cos β|) ∧ 
  (α + β + γ = 0 → |Real.cos α| + |Real.cos β| + |Real.cos γ| ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_inequalities_l1193_119377


namespace NUMINAMATH_CALUDE_sum_first_six_primes_gt_10_l1193_119372

def first_six_primes_gt_10 : List Nat :=
  [11, 13, 17, 19, 23, 29]

theorem sum_first_six_primes_gt_10 :
  first_six_primes_gt_10.sum = 112 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_gt_10_l1193_119372


namespace NUMINAMATH_CALUDE_candy_distribution_l1193_119371

theorem candy_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (Nat.choose (n + k - 1) (k - 1)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l1193_119371


namespace NUMINAMATH_CALUDE_expression_value_l1193_119398

theorem expression_value (x y : ℝ) (h : x - 2*y + 3 = 0) : 
  (2*y - x)^2 - 2*x + 4*y - 1 = 14 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1193_119398


namespace NUMINAMATH_CALUDE_blue_chip_value_l1193_119367

theorem blue_chip_value (yellow_value : ℕ) (green_value : ℕ) (yellow_count : ℕ) (blue_count : ℕ) (total_product : ℕ) :
  yellow_value = 2 →
  green_value = 5 →
  yellow_count = 4 →
  blue_count = blue_count →  -- This represents that blue and green chip counts are equal
  total_product = 16000 →
  total_product = yellow_value ^ yellow_count * blue_value ^ blue_count * green_value ^ blue_count →
  blue_value = 8 := by
  sorry

#check blue_chip_value

end NUMINAMATH_CALUDE_blue_chip_value_l1193_119367


namespace NUMINAMATH_CALUDE_ming_ladybugs_l1193_119396

/-- The number of spiders Sami found -/
def spiders : ℕ := 3

/-- The number of ants Hunter saw -/
def ants : ℕ := 12

/-- The number of ladybugs that flew away -/
def flown_ladybugs : ℕ := 2

/-- The number of insects remaining in the playground -/
def remaining_insects : ℕ := 21

/-- The number of ladybugs Ming discovered initially -/
def initial_ladybugs : ℕ := remaining_insects + flown_ladybugs - (spiders + ants)

theorem ming_ladybugs : initial_ladybugs = 8 := by
  sorry

end NUMINAMATH_CALUDE_ming_ladybugs_l1193_119396


namespace NUMINAMATH_CALUDE_lindas_coins_l1193_119341

/-- Represents the number of coins Linda has initially -/
structure InitialCoins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Represents the number of coins Linda's mother gives her -/
structure AdditionalCoins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- The problem statement -/
theorem lindas_coins (initial : InitialCoins) (additional : AdditionalCoins) 
    (h1 : initial.quarters = 6)
    (h2 : initial.nickels = 5)
    (h3 : additional.dimes = 2)
    (h4 : additional.quarters = 10)
    (h5 : additional.nickels = 2 * initial.nickels)
    (h6 : initial.dimes + initial.quarters + initial.nickels + 
          additional.dimes + additional.quarters + additional.nickels = 35) :
    initial.dimes = 4 := by
  sorry


end NUMINAMATH_CALUDE_lindas_coins_l1193_119341


namespace NUMINAMATH_CALUDE_joanne_first_hour_coins_l1193_119336

/-- Represents the number of coins Joanne collected in the first hour -/
def first_hour_coins : ℕ := sorry

/-- Represents the total number of coins collected in the second and third hours -/
def second_third_hour_coins : ℕ := 35

/-- Represents the number of coins collected in the fourth hour -/
def fourth_hour_coins : ℕ := 50

/-- Represents the number of coins given to the coworker -/
def coins_given_away : ℕ := 15

/-- Represents the total number of coins after the fourth hour -/
def total_coins : ℕ := 120

/-- Theorem stating that Joanne collected 15 coins in the first hour -/
theorem joanne_first_hour_coins : 
  first_hour_coins = 15 :=
by
  sorry

#check joanne_first_hour_coins

end NUMINAMATH_CALUDE_joanne_first_hour_coins_l1193_119336


namespace NUMINAMATH_CALUDE_beaver_count_l1193_119312

theorem beaver_count (initial_beavers : Float) (additional_beavers : Float) : 
  initial_beavers = 2.0 → additional_beavers = 1.0 → initial_beavers + additional_beavers = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_beaver_count_l1193_119312


namespace NUMINAMATH_CALUDE_sin_330_degrees_l1193_119379

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l1193_119379


namespace NUMINAMATH_CALUDE_milk_replacement_problem_l1193_119374

theorem milk_replacement_problem (initial_volume : ℝ) (final_pure_milk : ℝ) :
  initial_volume = 45 ∧ final_pure_milk = 28.8 →
  ∃ (x : ℝ), x = 9 ∧ 
  (initial_volume - x) * (initial_volume - x) / initial_volume = final_pure_milk :=
by sorry

end NUMINAMATH_CALUDE_milk_replacement_problem_l1193_119374


namespace NUMINAMATH_CALUDE_playground_boys_count_l1193_119354

theorem playground_boys_count (total_children girls : ℕ) 
  (h1 : total_children = 63) 
  (h2 : girls = 28) : 
  total_children - girls = 35 := by
sorry

end NUMINAMATH_CALUDE_playground_boys_count_l1193_119354


namespace NUMINAMATH_CALUDE_symmetry_implies_difference_l1193_119357

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetry_implies_difference (a b : ℝ) :
  symmetric_wrt_origin (-2, b) (a, 3) → a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_difference_l1193_119357


namespace NUMINAMATH_CALUDE_function_forms_correctness_l1193_119359

-- Define a linear function
def linear_function (a b x : ℝ) : ℝ := a * x + b

-- Define a special case of linear function
def linear_function_special (a x : ℝ) : ℝ := a * x

-- Define a quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define special cases of quadratic function
def quadratic_function_special1 (a c x : ℝ) : ℝ := a * x^2 + c
def quadratic_function_special2 (a x : ℝ) : ℝ := a * x^2

-- Theorem stating the correctness of these function definitions
theorem function_forms_correctness (a b c x : ℝ) (h : a ≠ 0) :
  (∃ y, y = linear_function a b x) ∧
  (∃ y, y = linear_function_special a x) ∧
  (∃ y, y = quadratic_function a b c x) ∧
  (∃ y, y = quadratic_function_special1 a c x) ∧
  (∃ y, y = quadratic_function_special2 a x) :=
sorry

end NUMINAMATH_CALUDE_function_forms_correctness_l1193_119359


namespace NUMINAMATH_CALUDE_highlighted_area_theorem_l1193_119386

theorem highlighted_area_theorem (circle_area : ℝ) (angle1 : ℝ) (angle2 : ℝ) :
  circle_area = 20 →
  angle1 = 60 →
  angle2 = 30 →
  (angle1 + angle2) / 360 * circle_area = 5 :=
by sorry

end NUMINAMATH_CALUDE_highlighted_area_theorem_l1193_119386


namespace NUMINAMATH_CALUDE_a_greater_equal_four_l1193_119311

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x - a < 0}

-- State the theorem
theorem a_greater_equal_four (a : ℝ) : A ⊆ B a → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_equal_four_l1193_119311


namespace NUMINAMATH_CALUDE_average_first_30_multiples_of_29_l1193_119352

theorem average_first_30_multiples_of_29 : 
  let n : ℕ := 30
  let base : ℕ := 29
  let sum : ℕ := n * (base + n * base) / 2
  (sum : ℚ) / n = 449.5 := by sorry

end NUMINAMATH_CALUDE_average_first_30_multiples_of_29_l1193_119352


namespace NUMINAMATH_CALUDE_impossibility_of_triangular_section_l1193_119324

theorem impossibility_of_triangular_section (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = 7^2 →
  b^2 + c^2 = 8^2 →
  c^2 + a^2 = 11^2 →
  False :=
by sorry

end NUMINAMATH_CALUDE_impossibility_of_triangular_section_l1193_119324


namespace NUMINAMATH_CALUDE_unique_number_property_l1193_119383

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 3 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l1193_119383


namespace NUMINAMATH_CALUDE_sqrt_AB_value_l1193_119366

def A : ℕ := 10^9 - 987654321
def B : ℚ := (123456789 + 1) / 10

theorem sqrt_AB_value : Real.sqrt (A * B) = 12345679 := by sorry

end NUMINAMATH_CALUDE_sqrt_AB_value_l1193_119366


namespace NUMINAMATH_CALUDE_mona_unique_players_l1193_119378

/-- Represents the number of groups Mona joined --/
def total_groups : ℕ := 18

/-- Represents the number of groups where Mona encountered 2 previous players --/
def groups_with_two_previous : ℕ := 6

/-- Represents the number of groups where Mona encountered 1 previous player --/
def groups_with_one_previous : ℕ := 4

/-- Represents the number of players in the first large group --/
def first_large_group : ℕ := 9

/-- Represents the number of previous players in the first large group --/
def previous_in_first_large : ℕ := 4

/-- Represents the number of players in the second large group --/
def second_large_group : ℕ := 12

/-- Represents the number of previous players in the second large group --/
def previous_in_second_large : ℕ := 5

/-- Theorem stating that Mona grouped with at least 20 unique players --/
theorem mona_unique_players : ℕ := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l1193_119378


namespace NUMINAMATH_CALUDE_angle_C_measure_l1193_119317

-- Define the angles A, B, and C
variable (A B C : ℝ)

-- Define the parallel lines condition
variable (p_parallel_q : Bool)

-- State the theorem
theorem angle_C_measure :
  p_parallel_q = true →
  A = (1/4) * B →
  B + C = 180 →
  C = 36 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l1193_119317


namespace NUMINAMATH_CALUDE_fermat_like_theorem_l1193_119338

theorem fermat_like_theorem (k : ℕ) : ¬ ∃ (x y z : ℤ), 
  (x^k + y^k = z^k) ∧ (z > 0) ∧ (0 < x) ∧ (x < k) ∧ (0 < y) ∧ (y < k) := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_theorem_l1193_119338


namespace NUMINAMATH_CALUDE_two_books_from_three_genres_l1193_119321

/-- The number of ways to select 2 books of different genres from 3 genres with 4 books each -/
def select_two_books (num_genres : ℕ) (books_per_genre : ℕ) : ℕ :=
  let total_books := num_genres * books_per_genre
  let books_in_other_genres := (num_genres - 1) * books_per_genre
  (total_books * books_in_other_genres) / 2

/-- Theorem stating that selecting 2 books of different genres from 3 genres with 4 books each results in 48 possibilities -/
theorem two_books_from_three_genres : 
  select_two_books 3 4 = 48 := by
  sorry

#eval select_two_books 3 4

end NUMINAMATH_CALUDE_two_books_from_three_genres_l1193_119321


namespace NUMINAMATH_CALUDE_train_length_calculation_l1193_119376

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length_calculation (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 27 → 
  ∃ (length_m : ℝ), abs (length_m - 450.09) < 0.01 ∧ length_m = speed_kmh * (1000 / 3600) * time_s := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1193_119376


namespace NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l1193_119313

theorem max_cars_ac_no_stripes (total_cars : Nat) (cars_no_ac : Nat) (cars_with_stripes : Nat)
  (red_cars : Nat) (red_cars_ac_stripes : Nat) (cars_2000s : Nat) (cars_2010s : Nat)
  (min_new_cars_stripes : Nat) (h1 : total_cars = 150) (h2 : cars_no_ac = 47)
  (h3 : cars_with_stripes = 65) (h4 : red_cars = 25) (h5 : red_cars_ac_stripes = 10)
  (h6 : cars_2000s = 30) (h7 : cars_2010s = 43) (h8 : min_new_cars_stripes = 39)
  (h9 : min_new_cars_stripes ≤ cars_2000s + cars_2010s) :
  (cars_2000s + cars_2010s) - min_new_cars_stripes - red_cars_ac_stripes = 24 :=
by sorry

end NUMINAMATH_CALUDE_max_cars_ac_no_stripes_l1193_119313


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1193_119327

-- Define a geometric sequence
def is_geometric_sequence (x y z : ℝ) : Prop :=
  ∃ q : ℝ, y = x * q ∧ z = y * q

-- Define the problem statement
theorem geometric_sequence_problem (a b c d : ℝ) 
  (h : is_geometric_sequence a c d) :
  (is_geometric_sequence (a*b) (b+c) (c+d) ∨
   is_geometric_sequence (a*b) (b*c) (c*d) ∨
   is_geometric_sequence (a*b) (b-c) (-d)) ∧
  ¬(is_geometric_sequence (a*b) (b+c) (c+d) ∧
    is_geometric_sequence (a*b) (b*c) (c*d)) ∧
  ¬(is_geometric_sequence (a*b) (b+c) (c+d) ∧
    is_geometric_sequence (a*b) (b-c) (-d)) ∧
  ¬(is_geometric_sequence (a*b) (b*c) (c*d) ∧
    is_geometric_sequence (a*b) (b-c) (-d)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1193_119327


namespace NUMINAMATH_CALUDE_problem_solution_l1193_119351

theorem problem_solution (m n : ℝ) (h : |3*m - 15| + ((n/3 + 1)^2) = 0) : 2*m - n = 13 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1193_119351


namespace NUMINAMATH_CALUDE_f_properties_l1193_119314

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^3 - 6*x^2 - 9*x + 3

-- Define the interval
def interval : Set ℝ := Set.Icc (-4) 2

-- Theorem statement
theorem f_properties :
  -- 1. f is strictly decreasing on (-∞, -3) and (-1, +∞)
  (∀ x y, x < y → x < -3 → f y < f x) ∧
  (∀ x y, x < y → -1 < x → f y < f x) ∧
  -- 2. The minimum value of f on [-4, 2] is -47
  (∀ x ∈ interval, f x ≥ -47) ∧
  (∃ x ∈ interval, f x = -47) ∧
  -- 3. The maximum value of f on [-4, 2] is 7
  (∀ x ∈ interval, f x ≤ 7) ∧
  (∃ x ∈ interval, f x = 7) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1193_119314


namespace NUMINAMATH_CALUDE_fruit_baskets_count_l1193_119393

/-- The number of non-empty fruit baskets -/
def num_fruit_baskets (num_apples num_oranges : ℕ) : ℕ :=
  (num_apples + 1) * (num_oranges + 1) - 1

/-- Theorem: The number of non-empty fruit baskets with 6 apples and 12 oranges is 90 -/
theorem fruit_baskets_count :
  num_fruit_baskets 6 12 = 90 := by
sorry

end NUMINAMATH_CALUDE_fruit_baskets_count_l1193_119393


namespace NUMINAMATH_CALUDE_tigers_wins_l1193_119304

theorem tigers_wins (total_games : ℕ) (games_lost_more : ℕ) 
  (h1 : total_games = 120)
  (h2 : games_lost_more = 38) :
  let games_won := (total_games - games_lost_more) / 2
  games_won = 41 := by
sorry

end NUMINAMATH_CALUDE_tigers_wins_l1193_119304


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1193_119320

-- Define sets A and B
def A : Set ℝ := {x | x > 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Define the theorem
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1193_119320


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l1193_119385

/-- Calculates the corrected mean of a set of observations after fixing an error in one observation -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / (n : ℚ)

/-- Theorem stating that the corrected mean for the given problem is 32.5 -/
theorem corrected_mean_problem :
  corrected_mean 50 32 23 48 = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_problem_l1193_119385


namespace NUMINAMATH_CALUDE_max_value_constrained_l1193_119355

/-- Given non-negative real numbers x and y satisfying the constraints
x + 2y ≤ 6 and 2x + y ≤ 6, the maximum value of x + y is 4. -/
theorem max_value_constrained (x y : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) 
  (h1 : x + 2*y ≤ 6) (h2 : 2*x + y ≤ 6) : 
  x + y ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ x₀ + 2*y₀ ≤ 6 ∧ 2*x₀ + y₀ ≤ 6 ∧ x₀ + y₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constrained_l1193_119355


namespace NUMINAMATH_CALUDE_largest_c_value_l1193_119387

theorem largest_c_value : ∃ (c_max : ℚ), c_max = 4 ∧ 
  (∀ c : ℚ, (3 * c + 4) * (c - 2) = 9 * c → c ≤ c_max) ∧
  ((3 * c_max + 4) * (c_max - 2) = 9 * c_max) := by
  sorry

end NUMINAMATH_CALUDE_largest_c_value_l1193_119387


namespace NUMINAMATH_CALUDE_pigeonhole_birthday_l1193_119397

theorem pigeonhole_birthday (n : ℕ) :
  (∀ f : Fin n → Fin 366, ∃ i j, i ≠ j ∧ f i = f j) ↔ n ≥ 367 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_birthday_l1193_119397


namespace NUMINAMATH_CALUDE_train_speed_l1193_119392

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 900) (h2 : time = 12) :
  length / time = 75 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1193_119392


namespace NUMINAMATH_CALUDE_not_equal_implies_not_both_zero_l1193_119365

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem not_equal_implies_not_both_zero (a b : V) (h : a ≠ b) : ¬(a = 0 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_not_equal_implies_not_both_zero_l1193_119365


namespace NUMINAMATH_CALUDE_paint_usage_for_large_canvas_l1193_119308

/-- Given an artist who uses L ounces of paint for every large canvas and 2 ounces for every small canvas,
    prove that L = 3 when the artist has completed 3 large paintings and 4 small paintings,
    using a total of 17 ounces of paint. -/
theorem paint_usage_for_large_canvas (L : ℝ) : 
  (3 * L + 4 * 2 = 17) → L = 3 := by
  sorry

end NUMINAMATH_CALUDE_paint_usage_for_large_canvas_l1193_119308


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l1193_119302

/-- The number of different types of wrapping paper -/
def wrapping_paper_types : ℕ := 10

/-- The number of different colors of ribbon -/
def ribbon_colors : ℕ := 4

/-- The number of different types of gift cards -/
def gift_card_types : ℕ := 5

/-- The number of different styles of gift tags -/
def gift_tag_styles : ℕ := 2

/-- The total number of different combinations for gift wrapping -/
def total_combinations : ℕ := wrapping_paper_types * ribbon_colors * gift_card_types * gift_tag_styles

theorem gift_wrapping_combinations : total_combinations = 400 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l1193_119302


namespace NUMINAMATH_CALUDE_remaining_money_l1193_119332

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := octal_to_decimal 5372

def ticket_cost : ℕ := 1200

theorem remaining_money :
  john_savings - ticket_cost = 1610 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1193_119332


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1193_119348

theorem rectangular_prism_volume 
  (top_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ) 
  (h₁ : top_area = 20) 
  (h₂ : side_area = 15) 
  (h₃ : front_area = 12) : 
  ∃ (x y z : ℝ), 
    x * y = top_area ∧ 
    y * z = side_area ∧ 
    x * z = front_area ∧ 
    x * y * z = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1193_119348


namespace NUMINAMATH_CALUDE_probability_3400_is_3_32_l1193_119330

/-- The number of non-bankrupt outcomes on the spinner -/
def num_outcomes : ℕ := 4

/-- The total number of possible combinations in three spins -/
def total_combinations : ℕ := num_outcomes ^ 3

/-- The number of ways to arrange three specific amounts that sum to $3400 -/
def favorable_arrangements : ℕ := 6

/-- The probability of earning exactly $3400 in three spins -/
def probability_3400 : ℚ := favorable_arrangements / total_combinations

theorem probability_3400_is_3_32 : probability_3400 = 3 / 32 := by
  sorry

end NUMINAMATH_CALUDE_probability_3400_is_3_32_l1193_119330


namespace NUMINAMATH_CALUDE_average_reading_time_l1193_119381

/-- Given that Emery reads 5 times faster than Serena and takes 20 days to read a book,
    prove that the average number of days for both to read the book is 60 days. -/
theorem average_reading_time (emery_days : ℕ) (emery_speed : ℕ) :
  emery_days = 20 →
  emery_speed = 5 →
  (emery_days + emery_speed * emery_days) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_reading_time_l1193_119381


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1193_119319

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1193_119319


namespace NUMINAMATH_CALUDE_percentage_ratio_theorem_l1193_119318

theorem percentage_ratio_theorem (y : ℝ) : 
  let x := 7 * y
  let z := 3 * (x - y)
  let percentage := (x - y) / x * 100
  let ratio := z / (x + y)
  percentage / ratio = 800 / 21 := by
sorry

end NUMINAMATH_CALUDE_percentage_ratio_theorem_l1193_119318


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l1193_119356

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l1193_119356


namespace NUMINAMATH_CALUDE_absolute_value_equation_implies_power_l1193_119328

theorem absolute_value_equation_implies_power (x : ℝ) :
  |x| = 3 * x + 1 → (4 * x + 2)^2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_implies_power_l1193_119328


namespace NUMINAMATH_CALUDE_not_perfect_square_l1193_119384

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n ^ 2)) :
  ¬∃ (x : ℕ), (n : ℝ) ^ 2 + d = (x : ℝ) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1193_119384


namespace NUMINAMATH_CALUDE_cube_surface_area_l1193_119334

/-- The surface area of a cube given the sum of its edge lengths -/
theorem cube_surface_area (sum_of_edges : ℝ) (h : sum_of_edges = 36) : 
  6 * (sum_of_edges / 12)^2 = 54 := by
  sorry

#check cube_surface_area

end NUMINAMATH_CALUDE_cube_surface_area_l1193_119334


namespace NUMINAMATH_CALUDE_min_largest_group_size_l1193_119335

theorem min_largest_group_size (total_boxes : ℕ) (min_apples max_apples : ℕ) : 
  total_boxes = 128 →
  min_apples = 120 →
  max_apples = 144 →
  ∃ (n : ℕ), n = 6 ∧ 
    (∀ (group_size : ℕ), 
      (group_size * (max_apples - min_apples + 1) ≥ total_boxes → group_size ≥ n) ∧
      (∃ (distribution : List ℕ), 
        distribution.length = max_apples - min_apples + 1 ∧
        distribution.sum = total_boxes ∧
        ∀ (x : ℕ), x ∈ distribution → x ≤ n)) :=
by sorry

end NUMINAMATH_CALUDE_min_largest_group_size_l1193_119335


namespace NUMINAMATH_CALUDE_lines_parallel_iff_l1193_119306

/-- Two lines in R² defined by parametric equations -/
structure ParallelLines where
  k : ℝ
  line1 : ℝ → ℝ × ℝ := λ t => (1 + 5*t, 3 - 3*t)
  line2 : ℝ → ℝ × ℝ := λ s => (4 - 2*s, 1 + k*s)

/-- The lines are parallel (do not intersect) if and only if k = 6/5 -/
theorem lines_parallel_iff (pl : ParallelLines) : 
  (∀ t s, pl.line1 t ≠ pl.line2 s) ↔ pl.k = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_l1193_119306


namespace NUMINAMATH_CALUDE_justine_fewer_than_ylona_l1193_119326

/-- The number of rubber bands each person had initially and after Bailey's distribution --/
structure RubberBands where
  justine_initial : ℕ
  bailey_initial : ℕ
  ylona_initial : ℕ
  bailey_final : ℕ

/-- The conditions of the rubber band problem --/
def rubber_band_problem (rb : RubberBands) : Prop :=
  rb.justine_initial = rb.bailey_initial + 10 ∧
  rb.justine_initial < rb.ylona_initial ∧
  rb.bailey_final = rb.bailey_initial - 4 ∧
  rb.bailey_final = 8 ∧
  rb.ylona_initial = 24

/-- Theorem stating that Justine had 2 fewer rubber bands than Ylona initially --/
theorem justine_fewer_than_ylona (rb : RubberBands) 
  (h : rubber_band_problem rb) : 
  rb.ylona_initial - rb.justine_initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_justine_fewer_than_ylona_l1193_119326


namespace NUMINAMATH_CALUDE_vote_ratio_proof_l1193_119350

def candidate_A_votes : ℕ := 14
def total_votes : ℕ := 21

theorem vote_ratio_proof :
  let candidate_B_votes := total_votes - candidate_A_votes
  (candidate_A_votes : ℚ) / candidate_B_votes = 2 := by
  sorry

end NUMINAMATH_CALUDE_vote_ratio_proof_l1193_119350


namespace NUMINAMATH_CALUDE_tree_spacing_l1193_119380

/-- Given a yard of length 300 meters with 26 equally spaced trees, including one at each end,
    the distance between consecutive trees is 12 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) : 
  yard_length = 300 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 12 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l1193_119380


namespace NUMINAMATH_CALUDE_trivia_team_size_l1193_119300

/-- The original number of members in a trivia team -/
def original_members (absent : ℕ) (points_per_member : ℕ) (total_points : ℕ) : ℕ :=
  (total_points / points_per_member) + absent

theorem trivia_team_size :
  original_members 3 2 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_size_l1193_119300


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1193_119362

theorem pure_imaginary_product (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) * (2 - Complex.I) = Complex.I * b) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1193_119362


namespace NUMINAMATH_CALUDE_second_machine_rate_l1193_119316

/-- Represents a copy machine with a constant rate of copies per minute -/
structure CopyMachine where
  copies_per_minute : ℕ

/-- Represents two copy machines working together -/
structure TwoMachines where
  machine1 : CopyMachine
  machine2 : CopyMachine

/-- The total number of copies produced by two machines in a given time -/
def total_copies (machines : TwoMachines) (minutes : ℕ) : ℕ :=
  (machines.machine1.copies_per_minute + machines.machine2.copies_per_minute) * minutes

theorem second_machine_rate (machines : TwoMachines) 
  (h1 : machines.machine1.copies_per_minute = 25)
  (h2 : total_copies machines 30 = 2400) :
  machines.machine2.copies_per_minute = 55 := by
sorry

end NUMINAMATH_CALUDE_second_machine_rate_l1193_119316


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1193_119399

theorem binomial_expansion_sum (x : ℝ) :
  ∃ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
    (2*x - 1)^5 = a*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅ ∧
    a₂ + a₃ = 40 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1193_119399


namespace NUMINAMATH_CALUDE_ellens_snack_calories_l1193_119333

/-- Calculates the calories of an afternoon snack given the total daily allowance and the calories consumed in other meals. -/
def afternoon_snack_calories (daily_allowance breakfast lunch dinner : ℕ) : ℕ :=
  daily_allowance - breakfast - lunch - dinner

/-- Proves that Ellen's afternoon snack was 130 calories given her daily allowance and other meal calorie counts. -/
theorem ellens_snack_calories :
  afternoon_snack_calories 2200 353 885 832 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ellens_snack_calories_l1193_119333


namespace NUMINAMATH_CALUDE_puzzle_solution_l1193_119331

theorem puzzle_solution (A B C : ℤ) 
  (eq1 : A + C = 10)
  (eq2 : A + B + 1 = C + 10)
  (eq3 : A + 1 = B) :
  A = 6 ∧ B = 7 ∧ C = 4 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_solution_l1193_119331


namespace NUMINAMATH_CALUDE_different_ball_counts_l1193_119349

/-- Represents a box in the game -/
structure Box :=
  (id : Nat)

/-- Represents a pair of boxes -/
structure BoxPair :=
  (box1 : Box)
  (box2 : Box)

/-- The game state -/
structure GameState :=
  (boxes : Finset Box)
  (pairs : Finset BoxPair)
  (ballCount : Box → Nat)

/-- The theorem statement -/
theorem different_ball_counts (n : Nat) (h : n = 2018) :
  ∃ (finalState : GameState),
    finalState.boxes.card = n ∧
    finalState.pairs.card = 2 * n - 2 ∧
    ∀ (b1 b2 : Box), b1 ∈ finalState.boxes → b2 ∈ finalState.boxes → b1 ≠ b2 →
      finalState.ballCount b1 ≠ finalState.ballCount b2 :=
by sorry

end NUMINAMATH_CALUDE_different_ball_counts_l1193_119349


namespace NUMINAMATH_CALUDE_faye_halloween_candy_l1193_119339

/-- Represents the number of candy pieces Faye scored on Halloween. -/
def initial_candy : ℕ := 47

/-- Represents the number of candy pieces Faye ate on the first night. -/
def eaten_candy : ℕ := 25

/-- Represents the number of candy pieces Faye's sister gave her. -/
def received_candy : ℕ := 40

/-- Represents the number of candy pieces Faye has now. -/
def current_candy : ℕ := 62

theorem faye_halloween_candy : 
  initial_candy - eaten_candy + received_candy = current_candy := by
  sorry

end NUMINAMATH_CALUDE_faye_halloween_candy_l1193_119339


namespace NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l1193_119375

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost : ℝ),
  (∃ (retail_price employee_price : ℝ),
    retail_price = 1.20 * wholesale_cost ∧
    employee_price = 0.85 * retail_price ∧
    employee_price = 204) →
  wholesale_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_video_recorder_wholesale_cost_l1193_119375


namespace NUMINAMATH_CALUDE_cube_tank_volume_l1193_119340

/-- Represents the number of metal sheets required to make the cube-shaped tank -/
def required_sheets : ℝ := 74.99999999999997

/-- Represents the length of a metal sheet in meters -/
def sheet_length : ℝ := 4

/-- Represents the width of a metal sheet in meters -/
def sheet_width : ℝ := 2

/-- Represents the number of faces in a cube -/
def cube_faces : ℕ := 6

/-- Represents the conversion factor from cubic meters to liters -/
def cubic_meter_to_liter : ℝ := 1000

/-- Theorem stating that the volume of the cube-shaped tank is 1,000,000 liters -/
theorem cube_tank_volume :
  let sheet_area := sheet_length * sheet_width
  let sheets_per_face := required_sheets / cube_faces
  let face_area := sheets_per_face * sheet_area
  let side_length := Real.sqrt face_area
  let volume_cubic_meters := side_length ^ 3
  let volume_liters := volume_cubic_meters * cubic_meter_to_liter
  volume_liters = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cube_tank_volume_l1193_119340


namespace NUMINAMATH_CALUDE_z_as_percentage_of_x_l1193_119301

theorem z_as_percentage_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 0.90 * y) 
  (h2 : y = 0.75 * x) : 
  z = 1.5 * x := by
sorry

end NUMINAMATH_CALUDE_z_as_percentage_of_x_l1193_119301


namespace NUMINAMATH_CALUDE_search_rescue_selection_methods_l1193_119390

def chinese_ships : ℕ := 4
def chinese_planes : ℕ := 3
def foreign_ships : ℕ := 5
def foreign_planes : ℕ := 2

def units_per_side : ℕ := 2
def total_units : ℕ := 4
def required_planes : ℕ := 1

theorem search_rescue_selection_methods :
  (chinese_ships.choose units_per_side * chinese_planes.choose required_planes * foreign_ships.choose units_per_side) +
  (chinese_ships.choose units_per_side * foreign_ships.choose (units_per_side - 1) * foreign_planes.choose required_planes) = 180 := by
  sorry

end NUMINAMATH_CALUDE_search_rescue_selection_methods_l1193_119390


namespace NUMINAMATH_CALUDE_negation_of_existence_is_universal_negation_l1193_119310

theorem negation_of_existence_is_universal_negation :
  (¬ ∃ (x : ℝ), x^2 = 1) ↔ (∀ (x : ℝ), x^2 ≠ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_is_universal_negation_l1193_119310


namespace NUMINAMATH_CALUDE_kevins_calculation_l1193_119322

theorem kevins_calculation (k : ℝ) : 
  (20 + 1) * (6 + k) = 20 + 1 * 6 + k → 20 + 1 * 6 + k = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_kevins_calculation_l1193_119322


namespace NUMINAMATH_CALUDE_two_pump_fill_time_l1193_119353

theorem two_pump_fill_time (small_pump_time large_pump_time : ℝ) 
  (h_small : small_pump_time = 3)
  (h_large : large_pump_time = 1/4)
  (h_positive : small_pump_time > 0 ∧ large_pump_time > 0) :
  1 / (1 / small_pump_time + 1 / large_pump_time) = 3/13 := by
  sorry

end NUMINAMATH_CALUDE_two_pump_fill_time_l1193_119353


namespace NUMINAMATH_CALUDE_pizza_fraction_eaten_l1193_119343

/-- Calculates the fraction of pizza eaten given the calorie information and consumption --/
theorem pizza_fraction_eaten 
  (lettuce_cal : ℕ) 
  (dressing_cal : ℕ) 
  (crust_cal : ℕ) 
  (cheese_cal : ℕ) 
  (total_consumed : ℕ) 
  (h1 : lettuce_cal = 50)
  (h2 : dressing_cal = 210)
  (h3 : crust_cal = 600)
  (h4 : cheese_cal = 400)
  (h5 : total_consumed = 330) :
  (total_consumed - (lettuce_cal + 2 * lettuce_cal + dressing_cal) / 4) / 
  (crust_cal + crust_cal / 3 + cheese_cal) = 1 / 5 := by
  sorry

#check pizza_fraction_eaten

end NUMINAMATH_CALUDE_pizza_fraction_eaten_l1193_119343


namespace NUMINAMATH_CALUDE_isosceles_triangle_lateral_side_length_l1193_119337

/-- Given an isosceles triangle with vertex angle α and the sum of two different heights l,
    the length of a lateral side is l * tan(α/2) / (1 + 2 * sin(α/2)). -/
theorem isosceles_triangle_lateral_side_length
  (α l : ℝ) (h_α : 0 < α ∧ α < π) (h_l : l > 0) :
  ∃ (side_length : ℝ),
    side_length = l * Real.tan (α / 2) / (1 + 2 * Real.sin (α / 2)) ∧
    ∃ (height1 height2 : ℝ),
      height1 + height2 = l ∧
      height1 ≠ height2 ∧
      ∃ (base : ℝ),
        height1 = side_length * Real.cos (α / 2) ∧
        height2 = base / 2 * Real.tan (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_lateral_side_length_l1193_119337


namespace NUMINAMATH_CALUDE_sandwich_percentage_l1193_119394

theorem sandwich_percentage (total_weight : ℝ) (condiment_weight : ℝ) 
  (h1 : total_weight = 150)
  (h2 : condiment_weight = 45) :
  (total_weight - condiment_weight) / total_weight * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_percentage_l1193_119394


namespace NUMINAMATH_CALUDE_salary_after_raise_l1193_119346

/-- 
Given an original salary and a percentage increase, 
calculate the new salary after a raise.
-/
theorem salary_after_raise 
  (original_salary : ℝ) 
  (percentage_increase : ℝ) 
  (new_salary : ℝ) : 
  original_salary = 55 ∧ 
  percentage_increase = 9.090909090909092 ∧
  new_salary = original_salary * (1 + percentage_increase / 100) →
  new_salary = 60 :=
by sorry

end NUMINAMATH_CALUDE_salary_after_raise_l1193_119346


namespace NUMINAMATH_CALUDE_conic_section_properties_l1193_119347

-- Define the equation C
def C (x y k : ℝ) : Prop := x^2 / (16 + k) - y^2 / (9 - k) = 1

-- Theorem statement
theorem conic_section_properties :
  -- The equation cannot represent a circle
  (∀ k : ℝ, ¬∃ r : ℝ, ∀ x y : ℝ, C x y k ↔ x^2 + y^2 = r^2) ∧
  -- When k > 9, the equation represents an ellipse with foci on the x-axis
  (∀ k : ℝ, k > 9 → ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y : ℝ, C x y k ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  -- When -16 < k < 9, the equation represents a hyperbola with foci on the x-axis
  (∀ k : ℝ, -16 < k ∧ k < 9 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, C x y k ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
  -- When the equation represents an ellipse or a hyperbola, the focal distance is always 10
  (∀ k : ℝ, (k > 9 ∨ (-16 < k ∧ k < 9)) → 
    ∃ c : ℝ, c = 5 ∧ 
    (∀ x y : ℝ, C x y k → 
      (k > 9 → ∃ a b : ℝ, a > b ∧ b > 0 ∧ c^2 = a^2 - b^2) ∧
      (-16 < k ∧ k < 9 → ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ c^2 = a^2 + b^2))) :=
sorry

end NUMINAMATH_CALUDE_conic_section_properties_l1193_119347


namespace NUMINAMATH_CALUDE_triangle_midpoint_sum_l1193_119303

theorem triangle_midpoint_sum (a b c : ℝ) (h : a + b + c = 15) :
  (a + b) + (a + c) + (b + c) = 30 := by
  sorry

end NUMINAMATH_CALUDE_triangle_midpoint_sum_l1193_119303


namespace NUMINAMATH_CALUDE_student_score_average_l1193_119369

/-- Given a student's scores in mathematics, physics, and chemistry, prove that the average of mathematics and chemistry scores is 26. -/
theorem student_score_average (math physics chem : ℕ) : 
  math + physics = 32 →
  chem = physics + 20 →
  (math + chem) / 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_student_score_average_l1193_119369


namespace NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l1193_119344

-- Define the propositions
variable (cheap : Prop) (good_quality : Prop)

-- Define the given condition
axiom cheap_implies_not_good : cheap → ¬good_quality

-- Theorem to prove
theorem not_cheap_necessary_for_good_quality :
  good_quality → ¬cheap :=
sorry

end NUMINAMATH_CALUDE_not_cheap_necessary_for_good_quality_l1193_119344


namespace NUMINAMATH_CALUDE_percent_of_300_l1193_119370

theorem percent_of_300 : (22 : ℝ) / 100 * 300 = 66 := by sorry

end NUMINAMATH_CALUDE_percent_of_300_l1193_119370


namespace NUMINAMATH_CALUDE_ant_return_probability_l1193_119325

/-- Represents a vertex in a tetrahedron -/
inductive Vertex : Type
  | A | B | C | D

/-- Represents the state of the ant's position -/
structure AntState :=
  (position : Vertex)
  (distance : ℕ)

/-- The probability of choosing any edge at a vertex -/
def edgeProbability : ℚ := 1 / 3

/-- The total distance the ant needs to travel -/
def totalDistance : ℕ := 4

/-- Function to calculate the probability of the ant being at a specific vertex after a certain distance -/
noncomputable def probabilityAtVertex (v : Vertex) (d : ℕ) : ℚ :=
  sorry

/-- Theorem stating the probability of the ant returning to vertex A after 4 moves -/
theorem ant_return_probability :
  probabilityAtVertex Vertex.A totalDistance = 7 / 27 :=
sorry

end NUMINAMATH_CALUDE_ant_return_probability_l1193_119325
