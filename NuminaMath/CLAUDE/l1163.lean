import Mathlib

namespace NUMINAMATH_CALUDE_monotonic_function_k_range_l1163_116396

theorem monotonic_function_k_range (k : ℝ) :
  (∀ x ≥ 1, Monotone (fun x : ℝ ↦ 4 * x^2 - k * x - 8)) →
  k ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_function_k_range_l1163_116396


namespace NUMINAMATH_CALUDE_race_length_proof_l1163_116314

/-- The race length in meters -/
def race_length : ℕ := 210

/-- Runner A's constant speed in m/s -/
def runner_a_speed : ℕ := 10

/-- Runner B's initial speed in m/s -/
def runner_b_initial_speed : ℕ := 1

/-- Runner B's speed increase per second in m/s -/
def runner_b_speed_increase : ℕ := 1

/-- Time difference between runners at finish in seconds -/
def finish_time_difference : ℕ := 1

/-- Function to calculate the distance covered by Runner B in t seconds -/
def runner_b_distance (t : ℕ) : ℕ := t * (t + 1) / 2

theorem race_length_proof :
  ∃ (t : ℕ), 
    (t * runner_a_speed = race_length) ∧ 
    (runner_b_distance (t - 1) = race_length) ∧ 
    (t > finish_time_difference) :=
by sorry

end NUMINAMATH_CALUDE_race_length_proof_l1163_116314


namespace NUMINAMATH_CALUDE_arithmetic_geometric_properties_l1163_116376

/-- Given an arithmetic progression {a_n} with common difference d,
    where a_3, a_4, and a_8 form a geometric progression,
    prove certain properties about the sequence and its sum. -/
theorem arithmetic_geometric_properties
  (a : ℕ → ℝ)  -- The sequence a_n
  (d : ℝ)      -- Common difference
  (h1 : d ≠ 0) -- d is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d)  -- Arithmetic progression property
  (h3 : (a 4) ^ 2 = a 3 * a 8)     -- Geometric progression property
  : a 1 * d < 0 ∧ 
    d * (4 * a 1 + 6 * d) < 0 ∧
    (a 4 / a 3 = 4) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_properties_l1163_116376


namespace NUMINAMATH_CALUDE_mean_equality_problem_l1163_116338

theorem mean_equality_problem (x y : ℚ) : 
  (((7 : ℚ) + 11 + 19 + 23) / 4 = (14 + x + y) / 3) →
  x = 2 * y →
  x = 62 / 3 ∧ y = 31 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l1163_116338


namespace NUMINAMATH_CALUDE_items_per_crate_l1163_116308

theorem items_per_crate (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) (crates : ℕ) :
  novels = 145 →
  comics = 271 →
  documentaries = 419 →
  albums = 209 →
  crates = 116 →
  (novels + comics + documentaries + albums) / crates = 9 := by
sorry

end NUMINAMATH_CALUDE_items_per_crate_l1163_116308


namespace NUMINAMATH_CALUDE_incorrect_statement_about_real_square_roots_l1163_116347

theorem incorrect_statement_about_real_square_roots :
  ¬ (∀ a b : ℝ, a < b ∧ b < 0 → ¬∃ x y : ℝ, x^2 = a ∧ y^2 = b) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_about_real_square_roots_l1163_116347


namespace NUMINAMATH_CALUDE_division_problem_l1163_116358

theorem division_problem : (88 : ℚ) / ((4 : ℚ) / 2) = 44 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1163_116358


namespace NUMINAMATH_CALUDE_max_value_constrained_l1163_116367

/-- Given non-negative real numbers x and y satisfying the constraints
x + 2y ≤ 6 and 2x + y ≤ 6, the maximum value of x + y is 4. -/
theorem max_value_constrained (x y : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) 
  (h1 : x + 2*y ≤ 6) (h2 : 2*x + y ≤ 6) : 
  x + y ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ x₀ + 2*y₀ ≤ 6 ∧ 2*x₀ + y₀ ≤ 6 ∧ x₀ + y₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constrained_l1163_116367


namespace NUMINAMATH_CALUDE_remainder_sum_l1163_116391

theorem remainder_sum (a b c : ℤ) 
  (ha : a % 80 = 75)
  (hb : b % 120 = 115)
  (hc : c % 160 = 155) : 
  (a + b + c) % 40 = 25 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l1163_116391


namespace NUMINAMATH_CALUDE_greatest_common_factor_of_168_252_315_l1163_116354

theorem greatest_common_factor_of_168_252_315 : Nat.gcd 168 (Nat.gcd 252 315) = 21 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_of_168_252_315_l1163_116354


namespace NUMINAMATH_CALUDE_race_outcomes_l1163_116392

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of podium positions (1st, 2nd, 3rd) -/
def num_positions : ℕ := 3

/-- The number of different podium outcomes in a race with no ties -/
def num_outcomes : ℕ := num_participants * (num_participants - 1) * (num_participants - 2)

theorem race_outcomes :
  num_outcomes = 120 :=
by sorry

end NUMINAMATH_CALUDE_race_outcomes_l1163_116392


namespace NUMINAMATH_CALUDE_special_square_area_l1163_116312

/-- A square with two vertices on a parabola and one side on a line -/
structure SpecialSquare where
  /-- The parabola on which two vertices of the square lie -/
  parabola : ℝ → ℝ
  /-- The line on which one side of the square lies -/
  line : ℝ → ℝ
  /-- Condition that the parabola is y = x^2 -/
  parabola_eq : parabola = fun x ↦ x^2
  /-- Condition that the line is y = 2x - 17 -/
  line_eq : line = fun x ↦ 2*x - 17

/-- The area of the special square is either 80 or 1280 -/
theorem special_square_area (s : SpecialSquare) :
  ∃ (area : ℝ), (area = 80 ∨ area = 1280) ∧ 
  (∃ (side : ℝ), side^2 = area ∧
   ∃ (x₁ y₁ x₂ y₂ : ℝ),
     y₁ = s.parabola x₁ ∧
     y₂ = s.parabola x₂ ∧
     (∃ (x₃ y₃ : ℝ), y₃ = s.line x₃ ∧
      side = ((x₃ - x₁)^2 + (y₃ - y₁)^2)^(1/2))) :=
by sorry

end NUMINAMATH_CALUDE_special_square_area_l1163_116312


namespace NUMINAMATH_CALUDE_not_perfect_square_l1163_116398

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n ^ 2)) :
  ¬∃ (x : ℕ), (n : ℝ) ^ 2 + d = (x : ℝ) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1163_116398


namespace NUMINAMATH_CALUDE_calculate_not_less_than_50_l1163_116393

/-- Represents the frequency of teachers in different age groups -/
structure TeacherAgeFrequency where
  less_than_30 : ℝ
  between_30_and_50 : ℝ
  not_less_than_50 : ℝ

/-- The sum of all frequencies in a probability distribution is 1 -/
axiom sum_of_frequencies (f : TeacherAgeFrequency) : 
  f.less_than_30 + f.between_30_and_50 + f.not_less_than_50 = 1

/-- Theorem: Given the frequencies for two age groups, we can calculate the third -/
theorem calculate_not_less_than_50 (f : TeacherAgeFrequency) 
    (h1 : f.less_than_30 = 0.3) 
    (h2 : f.between_30_and_50 = 0.5) : 
  f.not_less_than_50 = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_calculate_not_less_than_50_l1163_116393


namespace NUMINAMATH_CALUDE_floor_sqrt_18_squared_l1163_116368

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_18_squared_l1163_116368


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1163_116318

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, (1/4 : ℚ) + (x : ℚ)/9 < 1 ↔ x ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1163_116318


namespace NUMINAMATH_CALUDE_shirt_cost_l1163_116336

theorem shirt_cost (initial_amount : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 109 →
  num_shirts = 2 →
  pants_cost = 13 →
  remaining_amount = 74 →
  (initial_amount - remaining_amount - pants_cost) / num_shirts = 11 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_l1163_116336


namespace NUMINAMATH_CALUDE_z_change_l1163_116373

theorem z_change (w h z : ℝ) (z' : ℝ) : 
  let q := 5 * w / (4 * h * z^2)
  let q' := 5 * (4 * w) / (4 * (2 * h) * z'^2)
  q' / q = 2 / 9 →
  z' / z = 3 * Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_z_change_l1163_116373


namespace NUMINAMATH_CALUDE_mean_of_smallest_elements_l1163_116349

/-- F(n, r) represents the arithmetic mean of the smallest elements in all r-element subsets of {1, 2, ..., n} -/
def F (n r : ℕ) : ℚ :=
  sorry

/-- Theorem stating that F(n, r) = (n+1)/(r+1) for 1 ≤ r ≤ n -/
theorem mean_of_smallest_elements (n r : ℕ) (h : 1 ≤ r ∧ r ≤ n) :
  F n r = (n + 1 : ℚ) / (r + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_mean_of_smallest_elements_l1163_116349


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1163_116360

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 12) 
  (h2 : x + |y| - y = 10) : 
  x + y = 26/5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1163_116360


namespace NUMINAMATH_CALUDE_f_lower_bound_f_inequality_solution_l1163_116382

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

theorem f_lower_bound : ∀ x : ℝ, f x ≥ 4 := by sorry

theorem f_inequality_solution : 
  ∀ x : ℝ, f x ≥ x^2 - 2*x + 4 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_f_inequality_solution_l1163_116382


namespace NUMINAMATH_CALUDE_scientific_notation_of_1373100000000_l1163_116328

theorem scientific_notation_of_1373100000000 :
  (1373100000000 : ℝ) = 1.3731 * (10 ^ 12) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1373100000000_l1163_116328


namespace NUMINAMATH_CALUDE_corrected_mean_problem_l1163_116399

/-- Calculates the corrected mean of a set of observations after fixing an error in one observation -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / (n : ℚ)

/-- Theorem stating that the corrected mean for the given problem is 32.5 -/
theorem corrected_mean_problem :
  corrected_mean 50 32 23 48 = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_problem_l1163_116399


namespace NUMINAMATH_CALUDE_combined_length_legs_arms_l1163_116333

/-- Calculates the combined length of legs and arms for two people given their heights and body proportions -/
theorem combined_length_legs_arms 
  (aisha_height : ℝ) 
  (benjamin_height : ℝ) 
  (aisha_legs_ratio : ℝ) 
  (aisha_arms_ratio : ℝ) 
  (benjamin_legs_ratio : ℝ) 
  (benjamin_arms_ratio : ℝ) 
  (h1 : aisha_height = 174) 
  (h2 : benjamin_height = 190) 
  (h3 : aisha_legs_ratio = 1/3) 
  (h4 : aisha_arms_ratio = 1/6) 
  (h5 : benjamin_legs_ratio = 3/7) 
  (h6 : benjamin_arms_ratio = 1/4) : 
  (aisha_legs_ratio * aisha_height + aisha_arms_ratio * aisha_height + 
   benjamin_legs_ratio * benjamin_height + benjamin_arms_ratio * benjamin_height) = 215.93 := by
  sorry

end NUMINAMATH_CALUDE_combined_length_legs_arms_l1163_116333


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1163_116343

theorem inequality_solution_set (x : ℝ) : 
  (1/3 : ℝ) + |x - 11/48| < 1/2 ↔ x ∈ Set.Ioo (1/16 : ℝ) (19/48 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1163_116343


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l1163_116371

/-- Sequence c_n defined recursively -/
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

/-- Sequence a_n defined in terms of c_n -/
def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

/-- Theorem stating that a_n is a perfect square for n > 2 -/
theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l1163_116371


namespace NUMINAMATH_CALUDE_ellipse_a_range_l1163_116395

/-- Represents an ellipse with the given equation and foci on the x-axis -/
structure Ellipse (a : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1)
  (foci_on_x : True)  -- We don't need to formalize this condition for the proof

/-- The range of a for which the given equation represents an ellipse with foci on the x-axis -/
theorem ellipse_a_range (a : ℝ) (e : Ellipse a) : a > 3 ∨ (-6 < a ∧ a < -2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_a_range_l1163_116395


namespace NUMINAMATH_CALUDE_fred_earnings_l1163_116381

/-- The amount of money earned given an hourly rate and number of hours worked -/
def moneyEarned (hourlyRate : ℝ) (hoursWorked : ℝ) : ℝ :=
  hourlyRate * hoursWorked

/-- Proof that working 8 hours at $12.5 per hour results in $100 earned -/
theorem fred_earnings : moneyEarned 12.5 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_fred_earnings_l1163_116381


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1163_116364

theorem binomial_expansion_sum (x : ℝ) :
  ∃ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
    (2*x - 1)^5 = a*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅ ∧
    a₂ + a₃ = 40 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1163_116364


namespace NUMINAMATH_CALUDE_intersection_M_N_l1163_116301

def M : Set ℝ := {x : ℝ | |x + 1| ≤ 1}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1163_116301


namespace NUMINAMATH_CALUDE_triangle_properties_l1163_116319

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a = 2 * Real.sqrt 3 →
  A = π / 3 →
  12 = b^2 + c^2 - b*c →
  (∃ (S : ℝ), S = (Real.sqrt 3 / 4) * b * c ∧ S ≤ 3 * Real.sqrt 3 ∧
    (S = 3 * Real.sqrt 3 → b = c)) ∧
  (a + b + c ≤ 6 * Real.sqrt 3 ∧
    (a + b + c = 6 * Real.sqrt 3 → b = c)) ∧
  (0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
    1/2 < b/c ∧ b/c < 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1163_116319


namespace NUMINAMATH_CALUDE_esperanza_salary_l1163_116323

def gross_salary (rent food mortgage savings taxes : ℚ) : ℚ :=
  rent + food + mortgage + savings + taxes

theorem esperanza_salary :
  let rent : ℚ := 600
  let food : ℚ := (3/5) * rent
  let mortgage : ℚ := 3 * food
  let savings : ℚ := 2000
  let taxes : ℚ := (2/5) * savings
  gross_salary rent food mortgage savings taxes = 4840 := by
  sorry

end NUMINAMATH_CALUDE_esperanza_salary_l1163_116323


namespace NUMINAMATH_CALUDE_hyperbola_equilateral_focus_l1163_116384

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if the distance from the origin to the right focus is 6 and 
    the asymptote forms an equilateral triangle with the origin and the right focus,
    then a = 3 and b = 3√3 -/
theorem hyperbola_equilateral_focus (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := 6  -- distance from origin to right focus
  let slope := b / a  -- slope of asymptote
  c^2 = a^2 + b^2 →  -- focus property of hyperbola
  slope = Real.sqrt 3 →  -- equilateral triangle condition
  (a = 3 ∧ b = 3 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equilateral_focus_l1163_116384


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l1163_116397

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℚ) 
  (h1 : sugar / flour = 5 / 6)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 2000 := by
  sorry

#check bakery_sugar_amount

end NUMINAMATH_CALUDE_bakery_sugar_amount_l1163_116397


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1163_116300

/-- Given a parabola x^2 = 2py (p > 0), if a point on the parabola with ordinate 1 
    is at distance 3 from the focus, then the distance from the focus to the directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) (h1 : p > 0) : 
  (∃ x : ℝ, x^2 = 2*p*1 ∧ 
   ((x - 0)^2 + (1 - p/2)^2)^(1/2) = 3) → 
  (0 - (-p/2)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1163_116300


namespace NUMINAMATH_CALUDE_triangle_area_problem_l1163_116339

/-- Line with slope m passing through point (x0, y0) -/
def Line (m : ℚ) (x0 y0 : ℚ) : ℚ → ℚ → Prop :=
  fun x y => y - y0 = m * (x - x0)

/-- Area of a triangle given coordinates of its vertices -/
def TriangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_problem :
  let line1 := Line (3/4) 1 3
  let line2 := Line (-1/3) 1 3
  let line3 := fun x y => x + y = 8
  let x1 := 1
  let y1 := 3
  let x2 := 21/2
  let y2 := 11/2
  let x3 := 23/7
  let y3 := 32/7
  (∀ x y, line1 x y ↔ y = (3/4) * x + 9/4) ∧
  (∀ x y, line2 x y ↔ y = (-1/3) * x + 10/3) ∧
  line1 x1 y1 ∧
  line2 x1 y1 ∧
  line1 x3 y3 ∧
  line3 x3 y3 ∧
  line2 x2 y2 ∧
  line3 x2 y2 →
  TriangleArea x1 y1 x2 y2 x3 y3 = 475/28 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l1163_116339


namespace NUMINAMATH_CALUDE_savings_proof_l1163_116332

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the savings are 4000 --/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
    (h1 : income = 20000)
    (h2 : income_ratio = 5)
    (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 4000 := by
  sorry

#eval calculate_savings 20000 5 4

end NUMINAMATH_CALUDE_savings_proof_l1163_116332


namespace NUMINAMATH_CALUDE_a_investment_value_l1163_116357

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- Theorem stating that given the conditions of the partnership,
    a's investment is $45,000 --/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 63000)
  (hc : p.c_investment = 72000)
  (hp : p.total_profit = 60000)
  (hcs : p.c_profit_share = 24000) :
  p.a_investment = 45000 :=
sorry

end NUMINAMATH_CALUDE_a_investment_value_l1163_116357


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1163_116309

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem states that in a geometric sequence where the product of the first and fifth terms is 16,
    the third term is either 4 or -4. -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_prod : a 1 * a 5 = 16) :
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1163_116309


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_13_l1163_116327

/-- The area of a quadrilateral with vertices (0, 0), (4, 0), (4, 3), and (2, 5) -/
def quadrilateral_area : ℝ :=
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (4, 3)
  let v4 : ℝ × ℝ := (2, 5)
  -- Define the area calculation here
  0 -- placeholder, replace with actual calculation

/-- Theorem: The area of the quadrilateral is 13 -/
theorem quadrilateral_area_is_13 : quadrilateral_area = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_13_l1163_116327


namespace NUMINAMATH_CALUDE_percentage_increase_l1163_116304

theorem percentage_increase (original : ℝ) (new : ℝ) (percentage : ℝ) : 
  original = 80 →
  new = 88.8 →
  percentage = 11 →
  (new - original) / original * 100 = percentage :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_l1163_116304


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1163_116369

theorem complex_equation_solution (z : ℂ) : (3 - z) * Complex.I = 1 - 3 * Complex.I → z = 6 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1163_116369


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l1163_116344

/-- Given that the cost price of 121 chocolates equals the selling price of 77 chocolates,
    the gain percent is (4400 / 77)%. -/
theorem chocolate_gain_percent :
  ∀ (cost_price selling_price : ℝ),
  cost_price > 0 →
  selling_price > 0 →
  121 * cost_price = 77 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 4400 / 77 := by
sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l1163_116344


namespace NUMINAMATH_CALUDE_point_on_parallel_segment_l1163_116324

/-- Given a point M and a line segment MN parallel to the x-axis, 
    prove that N has specific coordinates -/
theorem point_on_parallel_segment 
  (M : ℝ × ℝ) 
  (length_MN : ℝ) 
  (h_M : M = (2, -4)) 
  (h_length : length_MN = 5) : 
  ∃ (N : ℝ × ℝ), (N = (-3, -4) ∨ N = (7, -4)) ∧ 
                 (N.2 = M.2) ∧ 
                 ((N.1 - M.1)^2 + (N.2 - M.2)^2 = length_MN^2) := by
  sorry

end NUMINAMATH_CALUDE_point_on_parallel_segment_l1163_116324


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l1163_116355

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l1163_116355


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l1163_116351

theorem sum_of_coefficients_equals_one (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                           a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + 
                           a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l1163_116351


namespace NUMINAMATH_CALUDE_fourth_root_sixteen_times_cube_root_eight_times_sqrt_four_l1163_116322

theorem fourth_root_sixteen_times_cube_root_eight_times_sqrt_four : 
  (16 : ℝ) ^ (1/4) * (8 : ℝ) ^ (1/3) * (4 : ℝ) ^ (1/2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_sixteen_times_cube_root_eight_times_sqrt_four_l1163_116322


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1163_116346

theorem partial_fraction_decomposition_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 → 
    (42 * x - 53) / (x^2 - 4*x + 3) = M₁ / (x - 1) + M₂ / (x - 3)) →
  M₁ * M₂ = 200.75 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l1163_116346


namespace NUMINAMATH_CALUDE_function_analysis_l1163_116341

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a*x^2 - 5

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 6*a*x

-- Theorem statement
theorem function_analysis (a : ℝ) :
  (f' a 2 = 0) →  -- x = 2 is a critical point
  (a = 1) ∧       -- The value of a is 1
  (∀ x ∈ Set.Icc (-2 : ℝ) (4 : ℝ), f 1 x ≤ 15) ∧  -- Maximum value on [-2, 4] is 15
  (∀ x ∈ Set.Icc (-2 : ℝ) (4 : ℝ), f 1 x ≥ -21)   -- Minimum value on [-2, 4] is -21
:= by sorry

end NUMINAMATH_CALUDE_function_analysis_l1163_116341


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1163_116342

theorem quadratic_root_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ < 1 ∧ 
   7 * x₁^2 - (m + 13) * x₁ + m^2 - m - 2 = 0 ∧
   7 * x₂^2 - (m + 13) * x₂ + m^2 - m - 2 = 0) →
  -2 < m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1163_116342


namespace NUMINAMATH_CALUDE_playground_boys_count_l1163_116366

theorem playground_boys_count (total_children girls : ℕ) 
  (h1 : total_children = 63) 
  (h2 : girls = 28) : 
  total_children - girls = 35 := by
sorry

end NUMINAMATH_CALUDE_playground_boys_count_l1163_116366


namespace NUMINAMATH_CALUDE_diamonds_G6_l1163_116362

/-- The k-th triangular number -/
def T (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The number of diamonds in the n-th figure -/
def diamonds (n : ℕ) : ℕ :=
  1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ i => T (i + 1)))

/-- The theorem stating that the number of diamonds in G_6 is 141 -/
theorem diamonds_G6 : diamonds 6 = 141 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_G6_l1163_116362


namespace NUMINAMATH_CALUDE_triangle_inequality_l1163_116340

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  2 * Real.sin A * Real.sin B < -Real.cos (2 * B + C) →
  a^2 + b^2 < c^2 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1163_116340


namespace NUMINAMATH_CALUDE_min_travel_time_less_than_3_9_l1163_116353

/-- Represents the problem of three people traveling with a motorcycle --/
structure TravelProblem where
  distance : ℝ
  walkSpeed : ℝ
  motorSpeed : ℝ
  motorCapacity : ℕ

/-- Calculates the minimum time for all three people to reach the destination --/
def minTravelTime (p : TravelProblem) : ℝ :=
  sorry

/-- The main theorem stating that the minimum travel time is less than 3.9 hours --/
theorem min_travel_time_less_than_3_9 :
  let p : TravelProblem := {
    distance := 135,
    walkSpeed := 6,
    motorSpeed := 90,
    motorCapacity := 2
  }
  minTravelTime p < 3.9 := by
  sorry

end NUMINAMATH_CALUDE_min_travel_time_less_than_3_9_l1163_116353


namespace NUMINAMATH_CALUDE_fraction_simplification_l1163_116305

theorem fraction_simplification : (1 : ℚ) / 462 + 23 / 42 = 127 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1163_116305


namespace NUMINAMATH_CALUDE_probability_3400_is_3_32_l1163_116378

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

end NUMINAMATH_CALUDE_probability_3400_is_3_32_l1163_116378


namespace NUMINAMATH_CALUDE_only_valid_root_l1163_116315

def original_equation (x : ℝ) : Prop :=
  (2 * x^2) / (x - 1) - (2 * x + 7) / 3 + (4 - 6 * x) / (x - 1) + 1 = 0

def transformed_equation (x : ℝ) : Prop :=
  x^2 - 5 * x + 4 = 0

theorem only_valid_root :
  (∀ x : ℝ, transformed_equation x ↔ (x = 4 ∨ x = 1)) →
  (∀ x : ℝ, original_equation x ↔ x = 4) :=
sorry

end NUMINAMATH_CALUDE_only_valid_root_l1163_116315


namespace NUMINAMATH_CALUDE_relay_team_selection_l1163_116370

/-- The number of ways to select and arrange 4 sprinters out of 6 for a 4×100m relay, 
    given that one sprinter cannot run the first leg and another cannot run the fourth leg. -/
theorem relay_team_selection (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n.factorial / (n - k).factorial) -     -- Total arrangements without restrictions
  2 * ((n - 1).factorial / (n - k).factorial) +  -- Subtracting arrangements with A or B in wrong position
  ((n - 2).factorial / (n - k).factorial) -- Adding back arrangements with both A and B in wrong positions
  = 252 := by sorry

end NUMINAMATH_CALUDE_relay_team_selection_l1163_116370


namespace NUMINAMATH_CALUDE_inequality_proof_l1163_116334

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : b + 1/a > a + 1/b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1163_116334


namespace NUMINAMATH_CALUDE_largest_expression_l1163_116372

theorem largest_expression : 
  let e1 := 992 * 999 + 999
  let e2 := 993 * 998 + 998
  let e3 := 994 * 997 + 997
  let e4 := 995 * 996 + 996
  (e4 > e1) ∧ (e4 > e2) ∧ (e4 > e3) := by
sorry

end NUMINAMATH_CALUDE_largest_expression_l1163_116372


namespace NUMINAMATH_CALUDE_consecutive_odd_divisibility_l1163_116307

theorem consecutive_odd_divisibility (m n : ℤ) : 
  (∃ k : ℤ, m = 2*k + 1 ∧ n = 2*k + 3) → 
  (∃ l : ℤ, 7*m^2 - 5*n^2 - 2 = 8*l) :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_divisibility_l1163_116307


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l1163_116313

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l1163_116313


namespace NUMINAMATH_CALUDE_coin_count_l1163_116306

/-- Represents the number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Represents the total value of coins in cents -/
def total_value : ℕ := 440

theorem coin_count (n : ℕ) : 
  n * (quarter_value + dime_value + nickel_value) = total_value → 
  n = 11 := by sorry

end NUMINAMATH_CALUDE_coin_count_l1163_116306


namespace NUMINAMATH_CALUDE_triangle_theorem_l1163_116365

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : a^2 = b^2 + c^2 - 2*b*c*Real.cos A

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) : 
  (2 * t.b = t.c + 2 * t.a * Real.cos t.C) → 
  (t.A = π / 3 ∧ 
   (1/2 * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3 ∧ t.a = 3 → 
    t.a + t.b + t.c = 8)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1163_116365


namespace NUMINAMATH_CALUDE_cookie_sales_l1163_116377

theorem cookie_sales (n : ℕ) (a : ℕ) (h1 : n = 10) (h2 : 1 ≤ a) (h3 : a < n) (h4 : 1 + a < n) : a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_l1163_116377


namespace NUMINAMATH_CALUDE_constrained_optimization_l1163_116321

theorem constrained_optimization (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : 3*x + 5*y + 7*z = 10) (h2 : x + 2*y + 5*z = 6) :
  let w := 2*x - 3*y + 4*z
  ∃ (max_w : ℝ), (∀ x' y' z' : ℝ, x' ≥ 0 → y' ≥ 0 → z' ≥ 0 →
    3*x' + 5*y' + 7*z' = 10 → x' + 2*y' + 5*z' = 6 →
    2*x' - 3*y' + 4*z' ≤ max_w) ∧
  max_w = 3 ∧ w ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_constrained_optimization_l1163_116321


namespace NUMINAMATH_CALUDE_teacher_selection_theorem_l1163_116356

/-- The number of male teachers -/
def num_male_teachers : ℕ := 4

/-- The number of female teachers -/
def num_female_teachers : ℕ := 3

/-- The total number of teachers to be selected -/
def num_selected : ℕ := 3

/-- The number of ways to select teachers with both genders represented -/
def num_ways_to_select : ℕ := 30

theorem teacher_selection_theorem :
  (num_ways_to_select = (Nat.choose num_male_teachers 2 * Nat.choose num_female_teachers 1) +
                        (Nat.choose num_male_teachers 1 * Nat.choose num_female_teachers 2)) ∧
  (num_ways_to_select = Nat.choose (num_male_teachers + num_female_teachers) num_selected -
                        Nat.choose num_male_teachers num_selected -
                        Nat.choose num_female_teachers num_selected) := by
  sorry

end NUMINAMATH_CALUDE_teacher_selection_theorem_l1163_116356


namespace NUMINAMATH_CALUDE_no_sequence_with_special_differences_l1163_116310

theorem no_sequence_with_special_differences :
  ¬ ∃ (a : ℕ → ℕ),
    (∀ k : ℕ, ∃! n : ℕ, a (n + 1) - a n = k) ∧
    (∀ k : ℕ, k > 2015 → ∃! n : ℕ, a (n + 2) - a n = k) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_with_special_differences_l1163_116310


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l1163_116394

/-- Represents a line in the coordinate plane --/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Generates horizontal lines y = k for k ∈ [-15, 15] --/
def horizontal_lines : List Line :=
  sorry

/-- Generates sloped lines y = √2x + 3k and y = -√2x + 3k for k ∈ [-15, 15] --/
def sloped_lines : List Line :=
  sorry

/-- All lines in the problem --/
def all_lines : List Line :=
  horizontal_lines ++ sloped_lines

/-- Predicate for an equilateral triangle with side length √2 --/
def is_unit_triangle (p q r : ℝ × ℝ) : Prop :=
  sorry

/-- Count of equilateral triangles formed by the intersection of lines --/
def triangle_count : ℕ :=
  sorry

/-- Main theorem stating the number of equilateral triangles formed --/
theorem equilateral_triangle_count :
  triangle_count = 12336 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l1163_116394


namespace NUMINAMATH_CALUDE_dog_distance_proof_l1163_116320

/-- The distance the dog runs when Ivan travels from work to home -/
def dog_distance (total_distance : ℝ) : ℝ :=
  2 * total_distance

theorem dog_distance_proof (total_distance : ℝ) (h1 : total_distance = 6) :
  dog_distance total_distance = 12 :=
by
  sorry

#check dog_distance_proof

end NUMINAMATH_CALUDE_dog_distance_proof_l1163_116320


namespace NUMINAMATH_CALUDE_translate_right_2_units_l1163_116302

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x + units, y := p.y }

theorem translate_right_2_units (A : Point2D) (h : A = ⟨-2, 3⟩) :
  translateRight A 2 = ⟨0, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_translate_right_2_units_l1163_116302


namespace NUMINAMATH_CALUDE_distance_FM_l1163_116348

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

-- Define the length of AB
def length_AB : ℝ := 6

-- Define the perpendicular bisector of AB
def perp_bisector (k : ℝ) (x y : ℝ) : Prop :=
  y - k = -(1/k) * (x - 2)

-- Define the point M
def point_M (k : ℝ) : ℝ × ℝ :=
  (4, 0)

-- Theorem statement
theorem distance_FM (k : ℝ) :
  let F := focus
  let M := point_M k
  (M.1 - F.1)^2 + (M.2 - F.2)^2 = 3^2 :=
sorry

end NUMINAMATH_CALUDE_distance_FM_l1163_116348


namespace NUMINAMATH_CALUDE_unique_quadratic_function_l1163_116303

/-- A quadratic function satisfying specific conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, 
    (∀ x, f x = a * x^2 + b * x + c) ∧ 
    (f (-1) = 0) ∧ 
    (∀ x, x ≤ f x) ∧
    (∀ x, f x ≤ (1 + x^2) / 2)

/-- The unique quadratic function satisfying the given conditions -/
theorem unique_quadratic_function (f : ℝ → ℝ) (hf : QuadraticFunction f) : 
  ∀ x, f x = (1/4) * x^2 + (1/2) * x + 1/4 :=
sorry

end NUMINAMATH_CALUDE_unique_quadratic_function_l1163_116303


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l1163_116387

theorem smallest_perfect_square_divisible_by_2_and_5 :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, n = m^2) → n % 2 = 0 → n % 5 = 0 → n ≥ 100 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_and_5_l1163_116387


namespace NUMINAMATH_CALUDE_divisibility_implication_l1163_116375

theorem divisibility_implication (x y : ℤ) : 
  (23 ∣ (3 * x + 2 * y)) → (23 ∣ (17 * x + 19 * y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l1163_116375


namespace NUMINAMATH_CALUDE_student_selection_methods_l1163_116386

theorem student_selection_methods (n : ℕ) (h : n = 5) : 
  (n.choose 2) * ((n - 2).choose 1) * ((n - 3).choose 1) = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_methods_l1163_116386


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1163_116363

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 + Complex.I*z) :
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1163_116363


namespace NUMINAMATH_CALUDE_concert_seats_count_l1163_116390

/-- Represents the concert ticket sales scenario -/
structure ConcertSales where
  main_price : ℕ  -- Price of main seat tickets
  back_price : ℕ  -- Price of back seat tickets
  total_revenue : ℕ  -- Total revenue from ticket sales
  back_seats_sold : ℕ  -- Number of back seat tickets sold

/-- Calculates the total number of seats in the arena -/
def total_seats (cs : ConcertSales) : ℕ :=
  let main_seats := (cs.total_revenue - cs.back_price * cs.back_seats_sold) / cs.main_price
  main_seats + cs.back_seats_sold

/-- Theorem stating that the total number of seats is 20,000 -/
theorem concert_seats_count (cs : ConcertSales) 
  (h1 : cs.main_price = 55)
  (h2 : cs.back_price = 45)
  (h3 : cs.total_revenue = 955000)
  (h4 : cs.back_seats_sold = 14500) : 
  total_seats cs = 20000 := by
  sorry

#eval total_seats ⟨55, 45, 955000, 14500⟩

end NUMINAMATH_CALUDE_concert_seats_count_l1163_116390


namespace NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l1163_116385

def geometric_series (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r^n

theorem geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, geometric_series a r n = a / (1 - r) :=
sorry

theorem specific_geometric_series_sum :
  ∑' n, geometric_series 1 (1/3) n = 3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_specific_geometric_series_sum_l1163_116385


namespace NUMINAMATH_CALUDE_books_in_box_l1163_116326

/-- The number of books in a box given the total weight and weight per book -/
def number_of_books (total_weight weight_per_book : ℚ) : ℚ :=
  total_weight / weight_per_book

/-- Theorem stating that a box weighing 42 pounds with books weighing 3 pounds each contains 14 books -/
theorem books_in_box : number_of_books 42 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_books_in_box_l1163_116326


namespace NUMINAMATH_CALUDE_sqrt_450_simplification_l1163_116379

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplification_l1163_116379


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l1163_116311

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l1163_116311


namespace NUMINAMATH_CALUDE_xinyu_taxi_fare_10km_l1163_116350

/-- Calculates the taxi fare in Xinyu city -/
def taxi_fare (distance : ℝ) : ℝ :=
  let base_fare := 5
  let mid_rate := 1.6
  let long_rate := 2.4
  let mid_distance := 6
  let long_distance := 2
  base_fare + mid_rate * mid_distance + long_rate * long_distance

/-- The total taxi fare for a 10 km journey in Xinyu city is 19.4 yuan -/
theorem xinyu_taxi_fare_10km : taxi_fare 10 = 19.4 := by
  sorry

end NUMINAMATH_CALUDE_xinyu_taxi_fare_10km_l1163_116350


namespace NUMINAMATH_CALUDE_smallest_marble_count_l1163_116316

theorem smallest_marble_count : ∃ m : ℕ, 
  m > 0 ∧ 
  m % 9 = 1 ∧ 
  m % 7 = 3 ∧ 
  (∀ n : ℕ, n > 0 ∧ n % 9 = 1 ∧ n % 7 = 3 → m ≤ n) ∧ 
  m = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l1163_116316


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l1163_116335

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 6) ↔ x ≥ 6 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l1163_116335


namespace NUMINAMATH_CALUDE_builders_for_houses_l1163_116359

/-- The number of builders needed to build multiple houses -/
def builders_needed (
  builders_per_floor : ℕ)  -- Number of builders to build one floor
  (days_per_floor : ℕ)     -- Number of days to build one floor
  (pay_per_day : ℕ)        -- Pay per builder per day
  (num_houses : ℕ)         -- Number of houses to build
  (floors_per_house : ℕ)   -- Number of floors per house
  (total_cost : ℕ)         -- Total cost to build all houses
  : ℕ :=
  total_cost / (builders_per_floor * days_per_floor * pay_per_day)

/-- Theorem stating the number of builders needed for the given scenario -/
theorem builders_for_houses : 
  builders_needed 3 30 100 5 6 270000 = 30 := by
  sorry

#eval builders_needed 3 30 100 5 6 270000

end NUMINAMATH_CALUDE_builders_for_houses_l1163_116359


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l1163_116352

theorem positive_real_inequalities (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + 3*b^2 ≥ 2*b*(a + b)) ∧ (a^3 + b^3 ≥ a*b^2 + a^2*b) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l1163_116352


namespace NUMINAMATH_CALUDE_complex_function_inequality_l1163_116345

/-- Given a ∈ (0,1) and f(z) = z^2 - z + a for z ∈ ℂ,
    for any z ∈ ℂ with |z| ≥ 1, there exists z₀ ∈ ℂ with |z₀| = 1
    such that |f(z₀)| ≤ |f(z)| -/
theorem complex_function_inequality (a : ℝ) (ha : 0 < a ∧ a < 1) :
  let f : ℂ → ℂ := fun z ↦ z^2 - z + a
  ∀ z : ℂ, Complex.abs z ≥ 1 →
    ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs (f z₀) ≤ Complex.abs (f z) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_function_inequality_l1163_116345


namespace NUMINAMATH_CALUDE_brianna_marbles_l1163_116389

theorem brianna_marbles (x : ℕ) : 
  x - 4 - (2 * 4) - (4 / 2) = 10 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_brianna_marbles_l1163_116389


namespace NUMINAMATH_CALUDE_even_sum_probability_l1163_116331

/-- Represents a spinner with its possible outcomes -/
structure Spinner :=
  (outcomes : List ℕ)

/-- The probability of getting an even sum from spinning all three spinners -/
def probability_even_sum (s t u : Spinner) : ℚ :=
  sorry

/-- The spinners as defined in the problem -/
def spinner_s : Spinner := ⟨[1, 2, 4]⟩
def spinner_t : Spinner := ⟨[3, 3, 6]⟩
def spinner_u : Spinner := ⟨[2, 4, 6]⟩

/-- The main theorem to prove -/
theorem even_sum_probability :
  probability_even_sum spinner_s spinner_t spinner_u = 5/9 :=
sorry

end NUMINAMATH_CALUDE_even_sum_probability_l1163_116331


namespace NUMINAMATH_CALUDE_handshake_count_l1163_116388

/-- The number of handshakes in a conference of 25 people -/
def conference_handshakes : ℕ := 300

/-- The number of attendees at the conference -/
def num_attendees : ℕ := 25

theorem handshake_count :
  (num_attendees.choose 2 : ℕ) = conference_handshakes :=
sorry

end NUMINAMATH_CALUDE_handshake_count_l1163_116388


namespace NUMINAMATH_CALUDE_total_marbles_l1163_116380

theorem total_marbles (dohyun_pockets : Nat) (dohyun_per_pocket : Nat)
                      (joohyun_bags : Nat) (joohyun_per_bag : Nat) :
  dohyun_pockets = 7 →
  dohyun_per_pocket = 16 →
  joohyun_bags = 6 →
  joohyun_per_bag = 25 →
  dohyun_pockets * dohyun_per_pocket + joohyun_bags * joohyun_per_bag = 262 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_l1163_116380


namespace NUMINAMATH_CALUDE_cos_plus_sin_implies_cos_double_angle_l1163_116383

theorem cos_plus_sin_implies_cos_double_angle 
  (θ : ℝ) (h : Real.cos θ + Real.sin θ = 7/5) : 
  Real.cos (2 * θ) = -527/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_plus_sin_implies_cos_double_angle_l1163_116383


namespace NUMINAMATH_CALUDE_min_value_at_eight_l1163_116317

theorem min_value_at_eight (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 24 / n ≥ 17 / 3 ∧
  ∃ (m : ℕ), m > 0 ∧ (m : ℝ) / 3 + 24 / m = 17 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_at_eight_l1163_116317


namespace NUMINAMATH_CALUDE_min_value_expression_l1163_116337

theorem min_value_expression (a : ℚ) : 
  |2*a + 1| + 1 ≥ 1 ∧ ∃ a : ℚ, |2*a + 1| + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1163_116337


namespace NUMINAMATH_CALUDE_min_value_inequality_l1163_116329

theorem min_value_inequality (a b c d : ℝ) 
  (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (d/c - 1)^2 + (5/d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l1163_116329


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1163_116361

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1163_116361


namespace NUMINAMATH_CALUDE_sum_of_fractions_nonnegative_l1163_116374

theorem sum_of_fractions_nonnegative (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + 
  (33 * b^2 - b) / (33 * b^2 + 1) + 
  (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_nonnegative_l1163_116374


namespace NUMINAMATH_CALUDE_smallest_t_for_no_h_route_l1163_116325

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (h_size : size = 8)

/-- Represents a Horse's move --/
structure HorseMove :=
  (horizontal : Nat)
  (vertical : Nat)

/-- Represents an H Route --/
def HRoute (board : Chessboard) (move : HorseMove) : Prop :=
  ∃ (path : List (Nat × Nat)), 
    path.length = board.size * board.size ∧
    ∀ (pos : Nat × Nat), pos ∈ path → 
      pos.1 ≤ board.size ∧ pos.2 ≤ board.size

/-- The main theorem --/
theorem smallest_t_for_no_h_route : 
  ∀ (board : Chessboard),
    ∀ (t : Nat),
      t > 0 →
      (∀ (start : Nat × Nat), 
        start.1 ≤ board.size ∧ start.2 ≤ board.size →
        ¬ HRoute board ⟨t, t+1⟩) →
      t = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_t_for_no_h_route_l1163_116325


namespace NUMINAMATH_CALUDE_donuts_left_l1163_116330

def initial_donuts : ℕ := 50
def bill_eats : ℕ := 2
def secretary_takes : ℕ := 4

def remaining_donuts : ℕ := 
  let after_bill := initial_donuts - bill_eats
  let after_secretary := after_bill - secretary_takes
  after_secretary / 2

theorem donuts_left : remaining_donuts = 22 := by sorry

end NUMINAMATH_CALUDE_donuts_left_l1163_116330
