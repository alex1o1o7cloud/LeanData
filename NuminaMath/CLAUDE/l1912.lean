import Mathlib

namespace NUMINAMATH_CALUDE_statement_is_proposition_l1912_191273

-- Define what a proposition is
def is_proposition (statement : String) : Prop :=
  ∃ (truth_value : Bool), (statement = "true") ∨ (statement = "false")

-- Define the statement we want to prove is a proposition
def statement : String := "20-5×3=10"

-- Theorem to prove
theorem statement_is_proposition : is_proposition statement := by
  sorry

end NUMINAMATH_CALUDE_statement_is_proposition_l1912_191273


namespace NUMINAMATH_CALUDE_largest_smallest_factor_l1912_191248

theorem largest_smallest_factor (a b c : ℕ+) : 
  a * b * c = 2160 → 
  ∃ (x : ℕ+), x ≤ a ∧ x ≤ b ∧ x ≤ c ∧ 
  (∀ (y : ℕ+), y ≤ a ∧ y ≤ b ∧ y ≤ c → y ≤ x) ∧ 
  x ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_largest_smallest_factor_l1912_191248


namespace NUMINAMATH_CALUDE_quadratic_roots_squared_l1912_191241

theorem quadratic_roots_squared (α β : ℝ) : 
  (α^2 - 3*α - 1 = 0) → 
  (β^2 - 3*β - 1 = 0) → 
  (α + β = 3) →
  (α * β = -1) →
  ((α^2)^2 - 11*(α^2) + 1 = 0) ∧ ((β^2)^2 - 11*(β^2) + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_squared_l1912_191241


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l1912_191229

theorem perfect_square_polynomial (a b : ℝ) : 
  (∃ p q r : ℝ, ∀ x : ℝ, x^4 - x^3 + x^2 + a*x + b = (p*x^2 + q*x + r)^2) → 
  b = 9/64 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l1912_191229


namespace NUMINAMATH_CALUDE_extremum_implies_slope_l1912_191215

-- Define the function f(x)
def f (c : ℝ) (x : ℝ) : ℝ := (x - 2) * (x^2 + c)

-- Define the derivative of f(x)
def f' (c : ℝ) (x : ℝ) : ℝ := (x^2 + c) + (x - 2) * (2 * x)

theorem extremum_implies_slope (c : ℝ) :
  (∃ k, f' c 1 = k ∧ k = 0) → f' c (-1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_slope_l1912_191215


namespace NUMINAMATH_CALUDE_quadratic_function_property_l1912_191223

def repeating_digits (k : ℕ) (p : ℕ) : ℚ := (k : ℚ) / 9 * (10^p - 1)

def f_k (k : ℕ) (x : ℚ) : ℚ := 9 / (k : ℚ) * x^2 + 2 * x

theorem quadratic_function_property (k : ℕ) (p : ℕ) 
  (h1 : 1 ≤ k) (h2 : k ≤ 9) (h3 : 0 < p) :
  f_k k (repeating_digits k p) = repeating_digits k (2 * p) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l1912_191223


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1912_191235

/-- Given an arithmetic sequence {a_n} with a₁ ≠ 0, if S₁, S₂, S₄ form a geometric sequence, 
    then a₂/a₁ = 1 or a₂/a₁ = 3 -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (a 1 ≠ 0) →                          -- first term not zero
  (∀ n, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  (∃ r, S 2 = r * S 1 ∧ S 4 = r * S 2) →  -- geometric sequence condition
  a 2 / a 1 = 1 ∨ a 2 / a 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1912_191235


namespace NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l1912_191275

/-- A function that returns the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def isEven (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has exactly 3 digits -/
def isThreeDigit (n : ℕ) : Prop := sorry

theorem no_even_three_digit_sum_27 :
  ¬∃ n : ℕ, isThreeDigit n ∧ digitSum n = 27 ∧ isEven n :=
sorry

end NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l1912_191275


namespace NUMINAMATH_CALUDE_unique_divisor_1058_l1912_191287

theorem unique_divisor_1058 : ∃! d : ℕ, d ≠ 1 ∧ d ∣ 1058 := by sorry

end NUMINAMATH_CALUDE_unique_divisor_1058_l1912_191287


namespace NUMINAMATH_CALUDE_last_digit_to_appear_is_six_l1912_191225

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define a function to check if a digit has appeared in the sequence up to n
def digitAppearedBy (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ unitsDigit (fib k) = d

-- Main theorem
theorem last_digit_to_appear_is_six :
  ∃ n : ℕ, (∀ d : ℕ, d < 10 → digitAppearedBy d n) ∧
  (∀ m : ℕ, m < n → ¬(∀ d : ℕ, d < 10 → digitAppearedBy d m)) ∧
  unitsDigit (fib n) = 6 :=
sorry

end NUMINAMATH_CALUDE_last_digit_to_appear_is_six_l1912_191225


namespace NUMINAMATH_CALUDE_track_completion_time_l1912_191260

/-- Represents a runner on the circular track -/
structure Runner :=
  (position : ℝ)
  (speed : ℝ)

/-- Represents the circular track -/
structure Track :=
  (circumference : ℝ)
  (runners : List Runner)

/-- Represents a meeting between two runners -/
structure Meeting :=
  (runner1 : Runner)
  (runner2 : Runner)
  (time : ℝ)

/-- The main theorem to be proved -/
theorem track_completion_time 
  (track : Track) 
  (meeting1 : Meeting) 
  (meeting2 : Meeting) 
  (meeting3 : Meeting) :
  meeting1.runner1 = meeting2.runner1 ∧ 
  meeting1.runner2 = meeting2.runner2 ∧
  meeting2.runner2 = meeting3.runner1 ∧
  meeting2.runner1 = meeting3.runner2 ∧
  meeting2.time - meeting1.time = 15 ∧
  meeting3.time - meeting2.time = 25 →
  track.circumference = 80 := by
  sorry

end NUMINAMATH_CALUDE_track_completion_time_l1912_191260


namespace NUMINAMATH_CALUDE_ellipse_equation_l1912_191211

/-- Given an ellipse with the endpoint of its short axis at (3, 0) and focal distance 4,
    prove that its equation is (y²/25) + (x²/9) = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let short_axis_endpoint : ℝ × ℝ := (3, 0)
  let focal_distance : ℝ := 4
  (y^2 / 25) + (x^2 / 9) = 1 := by
sorry


end NUMINAMATH_CALUDE_ellipse_equation_l1912_191211


namespace NUMINAMATH_CALUDE_f_properties_l1912_191264

/-- Definition of the function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a^x + a - 3) / Real.log a

/-- Theorem stating the properties of f(x) -/
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (Function.Injective (f a) ↔ (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1912_191264


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l1912_191227

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^4 + 4*x^2 + 16 : ℝ) = (x^2 + 4) * q x := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l1912_191227


namespace NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1912_191295

theorem max_value_expression (y : ℝ) (h : y > 0) :
  (y^2 + 3 - Real.sqrt (y^4 + 9)) / y ≤ 6 / (2 * Real.sqrt 3 + Real.sqrt 6) :=
sorry

theorem max_value_achievable :
  ∃ y : ℝ, y > 0 ∧ (y^2 + 3 - Real.sqrt (y^4 + 9)) / y = 6 / (2 * Real.sqrt 3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_max_value_achievable_l1912_191295


namespace NUMINAMATH_CALUDE_no_solution_exponential_equation_l1912_191217

theorem no_solution_exponential_equation :
  ¬∃ y : ℝ, (16 : ℝ)^(3*y - 6) = (64 : ℝ)^(2*y + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exponential_equation_l1912_191217


namespace NUMINAMATH_CALUDE_kitchen_broken_fraction_l1912_191283

theorem kitchen_broken_fraction :
  let foyer_broken : ℕ := 10
  let kitchen_total : ℕ := 35
  let total_not_broken : ℕ := 34
  let foyer_total : ℕ := foyer_broken * 3
  let total_bulbs : ℕ := foyer_total + kitchen_total
  let total_broken : ℕ := total_bulbs - total_not_broken
  let kitchen_broken : ℕ := total_broken - foyer_broken
  (kitchen_broken : ℚ) / kitchen_total = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_kitchen_broken_fraction_l1912_191283


namespace NUMINAMATH_CALUDE_rotations_composition_l1912_191265

/-- A rotation in the plane. -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Represents the composition of two rotations. -/
def compose_rotations (r1 r2 : Rotation) : Rotation :=
  sorry

/-- The angles of a triangle formed by the centers of two rotations and their composition. -/
def triangle_angles (r1 r2 : Rotation) : ℝ × ℝ × ℝ :=
  sorry

theorem rotations_composition 
  (O₁ O₂ : ℝ × ℝ) (α β : ℝ) 
  (h1 : 0 ≤ α ∧ α < 2 * π) 
  (h2 : 0 ≤ β ∧ β < 2 * π) 
  (h3 : α + β ≠ 2 * π) :
  let r1 : Rotation := ⟨O₁, α⟩
  let r2 : Rotation := ⟨O₂, β⟩
  let r_composed := compose_rotations r1 r2
  let angles := triangle_angles r1 r2
  (r_composed.angle = α + β) ∧
  ((α + β < 2 * π → angles = (α/2, β/2, π - (α + β)/2)) ∧
   (α + β > 2 * π → angles = (π - α/2, π - β/2, (α + β)/2))) :=
by sorry

end NUMINAMATH_CALUDE_rotations_composition_l1912_191265


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1912_191214

theorem negation_of_proposition (p : ℝ → Prop) :
  (∀ x : ℝ, x^2 > x - 1) ↔ ¬(∃ x : ℝ, x^2 ≤ x - 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1912_191214


namespace NUMINAMATH_CALUDE_snakes_not_hiding_l1912_191205

/-- Given a cage with snakes, some of which are hiding, calculate the number of snakes not hiding. -/
theorem snakes_not_hiding (total_snakes hiding_snakes : ℕ) 
  (h1 : total_snakes = 95)
  (h2 : hiding_snakes = 64) :
  total_snakes - hiding_snakes = 31 := by
  sorry

end NUMINAMATH_CALUDE_snakes_not_hiding_l1912_191205


namespace NUMINAMATH_CALUDE_complete_factorization_l1912_191234

theorem complete_factorization (x : ℝ) : 
  x^8 - 256 = (x^4 + 16) * (x^2 + 4) * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_l1912_191234


namespace NUMINAMATH_CALUDE_distance_to_other_focus_l1912_191250

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

/-- Distance from a point to one focus is 3 -/
def distance_to_one_focus (x y : ℝ) : Prop :=
  ∃ (fx fy : ℝ), (x - fx)^2 + (y - fy)^2 = 3^2

/-- Theorem: If a point is on the ellipse and its distance to one focus is 3,
    then its distance to the other focus is 5 -/
theorem distance_to_other_focus
  (x y : ℝ)
  (h1 : is_on_ellipse x y)
  (h2 : distance_to_one_focus x y) :
  ∃ (gx gy : ℝ), (x - gx)^2 + (y - gy)^2 = 5^2 :=
sorry

end NUMINAMATH_CALUDE_distance_to_other_focus_l1912_191250


namespace NUMINAMATH_CALUDE_max_quarters_sasha_l1912_191201

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100

def total_value : ℚ := 380 / 100

theorem max_quarters_sasha :
  ∃ (q : ℕ), 
    (q * quarter_value + q * nickel_value + 2 * q * dime_value ≤ total_value) ∧
    (∀ (n : ℕ), n > q → 
      n * quarter_value + n * nickel_value + 2 * n * dime_value > total_value) ∧
    q = 7 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_sasha_l1912_191201


namespace NUMINAMATH_CALUDE_shooting_competition_problem_prove_shooting_competition_l1912_191240

/-- Represents the penalty points for misses in a shooting competition -/
def penalty_points (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n : ℚ) / 2 * (2 + (n - 1))

/-- The shooting competition problem -/
theorem shooting_competition_problem 
  (total_shots : ℕ) 
  (total_penalty : ℚ) 
  (hits : ℕ) : Prop :=
  total_shots = 25 ∧ 
  total_penalty = 7 ∧ 
  penalty_points (total_shots - hits) = total_penalty ∧
  hits = 21

/-- Proof of the shooting competition problem -/
theorem prove_shooting_competition : 
  ∃ (hits : ℕ), shooting_competition_problem 25 7 hits :=
sorry

end NUMINAMATH_CALUDE_shooting_competition_problem_prove_shooting_competition_l1912_191240


namespace NUMINAMATH_CALUDE_survey_results_l1912_191285

/-- Represents the survey results for a subject -/
structure SubjectSurvey where
  yes : Nat
  no : Nat
  unsure : Nat

/-- The main theorem about the survey results -/
theorem survey_results 
  (total_students : Nat)
  (subject_m : SubjectSurvey)
  (subject_r : SubjectSurvey)
  (yes_only_m : Nat)
  (h1 : total_students = 800)
  (h2 : subject_m.yes = 500)
  (h3 : subject_m.no = 200)
  (h4 : subject_m.unsure = 100)
  (h5 : subject_r.yes = 400)
  (h6 : subject_r.no = 100)
  (h7 : subject_r.unsure = 300)
  (h8 : yes_only_m = 150)
  (h9 : subject_m.yes + subject_m.no + subject_m.unsure = total_students)
  (h10 : subject_r.yes + subject_r.no + subject_r.unsure = total_students) :
  total_students - (subject_m.yes + subject_r.yes - yes_only_m) = 400 := by
  sorry

end NUMINAMATH_CALUDE_survey_results_l1912_191285


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1912_191263

/-- The coefficient of x^2 in the expansion of (1 + 1/x)(1-x)^7 is -14 -/
theorem coefficient_x_squared_expansion : ℤ := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l1912_191263


namespace NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1912_191270

/-- Given a point (2, 7, z) in 3D space, where z is unknown, and its distance 
    from the origin (0, 0, 0) is 10 units, prove that z = √47. -/
theorem fly_distance_from_ceiling :
  ∀ z : ℝ, (2:ℝ)^2 + 7^2 + z^2 = 10^2 → z = Real.sqrt 47 := by
  sorry

end NUMINAMATH_CALUDE_fly_distance_from_ceiling_l1912_191270


namespace NUMINAMATH_CALUDE_mary_regular_rate_l1912_191218

/-- Represents Mary's work schedule and pay structure --/
structure MaryWork where
  maxHours : ℕ
  regularHours : ℕ
  overtimeRate : ℚ
  weeklyEarnings : ℚ

/-- Calculates Mary's regular hourly rate --/
def regularRate (w : MaryWork) : ℚ :=
  let overtimeHours := w.maxHours - w.regularHours
  w.weeklyEarnings / (w.regularHours + w.overtimeRate * overtimeHours)

/-- Theorem: Mary's regular hourly rate is $8 per hour --/
theorem mary_regular_rate :
  let w : MaryWork := {
    maxHours := 45,
    regularHours := 20,
    overtimeRate := 1.25,
    weeklyEarnings := 410
  }
  regularRate w = 8 := by sorry

end NUMINAMATH_CALUDE_mary_regular_rate_l1912_191218


namespace NUMINAMATH_CALUDE_x_minus_y_equals_eight_l1912_191238

theorem x_minus_y_equals_eight (x y : ℝ) 
  (hx : x + (-3) = 0) 
  (hy : |y| = 5) 
  (hxy : x * y < 0) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_eight_l1912_191238


namespace NUMINAMATH_CALUDE_yoongi_calculation_l1912_191203

theorem yoongi_calculation (x : ℕ) : 
  (x ≥ 10 ∧ x < 100) → (x - 35 = 27) → (x - 53 = 9) := by
  sorry

end NUMINAMATH_CALUDE_yoongi_calculation_l1912_191203


namespace NUMINAMATH_CALUDE_divisibility_by_240_l1912_191299

theorem divisibility_by_240 (p : ℕ) (hp : p.Prime) (hp_gt_7 : p > 7) : 
  240 ∣ (p^4 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_240_l1912_191299


namespace NUMINAMATH_CALUDE_complex_subtraction_l1912_191298

theorem complex_subtraction (a b : ℂ) (ha : a = 6 - 3*I) (hb : b = 2 + 3*I) :
  a - 3*b = -12*I := by sorry

end NUMINAMATH_CALUDE_complex_subtraction_l1912_191298


namespace NUMINAMATH_CALUDE_percentage_calculation_l1912_191284

theorem percentage_calculation (first_number second_number : ℝ) 
  (h1 : first_number = 110)
  (h2 : second_number = 22) :
  (second_number / first_number) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1912_191284


namespace NUMINAMATH_CALUDE_system_solution_system_solution_zero_system_solution_one_l1912_191294

theorem system_solution :
  ∀ x y z : ℝ,
  (2 * y + x - x^2 - y^2 = 0 ∧
   z - x + y - y * (x + z) = 0 ∧
   -2 * y + z - y^2 - z^2 = 0) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

-- Alternatively, we can split it into two theorems for each solution

theorem system_solution_zero :
  2 * 0 + 0 - 0^2 - 0^2 = 0 ∧
  0 - 0 + 0 - 0 * (0 + 0) = 0 ∧
  -2 * 0 + 0 - 0^2 - 0^2 = 0 :=
by
  sorry

theorem system_solution_one :
  2 * 0 + 1 - 1^2 - 0^2 = 0 ∧
  1 - 1 + 0 - 0 * (1 + 1) = 0 ∧
  -2 * 0 + 1 - 0^2 - 1^2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_system_solution_zero_system_solution_one_l1912_191294


namespace NUMINAMATH_CALUDE_divisibility_problem_l1912_191282

theorem divisibility_problem (n : ℕ) (h1 : n > 0) (h2 : (n + 1) % 6 = 4) :
  n % 2 = 1 := by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1912_191282


namespace NUMINAMATH_CALUDE_trees_per_square_meter_l1912_191237

/-- Given a forest and a square-shaped street, calculate the number of trees per square meter in the forest. -/
theorem trees_per_square_meter
  (street_side : ℝ)
  (forest_area_multiplier : ℝ)
  (total_trees : ℕ)
  (h1 : street_side = 100)
  (h2 : forest_area_multiplier = 3)
  (h3 : total_trees = 120000) :
  (total_trees : ℝ) / (forest_area_multiplier * street_side^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_trees_per_square_meter_l1912_191237


namespace NUMINAMATH_CALUDE_square_areas_equality_l1912_191268

theorem square_areas_equality (a : ℝ) :
  let M := a^2 + (a+3)^2 + (a+5)^2 + (a+6)^2
  let N := (a+1)^2 + (a+2)^2 + (a+4)^2 + (a+7)^2
  M = N := by sorry

end NUMINAMATH_CALUDE_square_areas_equality_l1912_191268


namespace NUMINAMATH_CALUDE_jake_has_one_more_balloon_l1912_191274

/-- The number of balloons Allan initially brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The number of additional balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

/-- The difference between Jake's balloons and Allan's total balloons -/
def balloon_difference : ℕ := jake_balloons - allan_total

theorem jake_has_one_more_balloon : balloon_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_one_more_balloon_l1912_191274


namespace NUMINAMATH_CALUDE_apple_ratio_simplification_l1912_191230

def sarah_apples : ℕ := 630
def brother_apples : ℕ := 270
def cousin_apples : ℕ := 540

theorem apple_ratio_simplification :
  ∃ (k : ℕ), k ≠ 0 ∧ 
    sarah_apples / k = 7 ∧ 
    brother_apples / k = 3 ∧ 
    cousin_apples / k = 6 :=
by sorry

end NUMINAMATH_CALUDE_apple_ratio_simplification_l1912_191230


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1912_191206

/-- Given a quadratic equation (m-2)x^2 + 3x + m^2 - 4 = 0 where x = 0 is a solution, prove that m = -2 -/
theorem quadratic_equation_solution (m : ℝ) : 
  ((m - 2) * 0^2 + 3 * 0 + m^2 - 4 = 0) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1912_191206


namespace NUMINAMATH_CALUDE_distance_to_destination_l1912_191276

/-- Proves that the distance to the destination is 2.25 kilometers given the specified conditions. -/
theorem distance_to_destination
  (rowing_speed : ℝ)
  (river_speed : ℝ)
  (round_trip_time : ℝ)
  (h1 : rowing_speed = 4)
  (h2 : river_speed = 2)
  (h3 : round_trip_time = 1.5)
  : ∃ (distance : ℝ),
    distance = 2.25 ∧
    round_trip_time = distance / (rowing_speed + river_speed) + distance / (rowing_speed - river_speed) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_to_destination_l1912_191276


namespace NUMINAMATH_CALUDE_decoration_nail_count_l1912_191244

theorem decoration_nail_count :
  ∀ D : ℕ,
  (D : ℚ) * (21/80) = 20 →
  ⌊(D : ℚ) * (5/8)⌋ = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_decoration_nail_count_l1912_191244


namespace NUMINAMATH_CALUDE_line_transformation_l1912_191221

open Matrix

-- Define the rotation matrix M
def M : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

-- Define the scaling matrix N
def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 3]

-- Define the combined transformation matrix NM
def NM : Matrix (Fin 2) (Fin 2) ℝ := N * M

theorem line_transformation (x y : ℝ) :
  (NM.mulVec ![x, y] = ![x, x]) ↔ (3 * x + 2 * y = 0) := by sorry

end NUMINAMATH_CALUDE_line_transformation_l1912_191221


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l1912_191278

theorem polynomial_coefficient_sums :
  ∀ (a a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (3 - 2 * x)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5) →
  (a₁ + a₂ + a₃ + a₄ + a₅ = -242) ∧
  (|a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 2882) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l1912_191278


namespace NUMINAMATH_CALUDE_lcm_24_30_40_l1912_191279

theorem lcm_24_30_40 : Nat.lcm (Nat.lcm 24 30) 40 = 120 := by sorry

end NUMINAMATH_CALUDE_lcm_24_30_40_l1912_191279


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1912_191224

theorem algebraic_expression_value (a b : ℝ) : 
  (2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18) → 
  (9 * b - 6 * a + 2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1912_191224


namespace NUMINAMATH_CALUDE_percentage_problem_l1912_191212

theorem percentage_problem (N : ℝ) (h : 0.2 * N = 1000) : 1.2 * N = 6000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1912_191212


namespace NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l1912_191288

/-- Prove that given vectors a = (1,2) and b = (-4,m), if a ⊥ b, then m = 2 -/
theorem orthogonal_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-4, m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_m_value_l1912_191288


namespace NUMINAMATH_CALUDE_quadratic_increasing_implies_a_range_l1912_191293

/-- A function f is increasing on an interval [a, +∞) if for all x, y in [a, +∞) with x < y, f(x) < f(y) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y → f x < f y

/-- The quadratic function f(x) = x^2 + (a-1)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + a

theorem quadratic_increasing_implies_a_range (a : ℝ) :
  IncreasingOnInterval (f a) 2 → a ∈ Set.Ici (-3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_implies_a_range_l1912_191293


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l1912_191297

def f (x : ℝ) := 10 * abs x

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l1912_191297


namespace NUMINAMATH_CALUDE_zero_multiple_of_all_primes_l1912_191247

theorem zero_multiple_of_all_primes : ∃! x : ℤ, ∀ p : ℕ, Nat.Prime p → ∃ k : ℤ, x = k * p :=
sorry

end NUMINAMATH_CALUDE_zero_multiple_of_all_primes_l1912_191247


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l1912_191233

theorem largest_stamps_per_page (book1_stamps book2_stamps : ℕ) 
  (h1 : book1_stamps = 924) 
  (h2 : book2_stamps = 1386) : 
  Nat.gcd book1_stamps book2_stamps = 462 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l1912_191233


namespace NUMINAMATH_CALUDE_sequence_is_increasing_l1912_191277

def isIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > a n

theorem sequence_is_increasing (a : ℕ → ℝ) 
    (h : ∀ n, a (n + 1) - a n - 3 = 0) : 
    isIncreasing a := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_increasing_l1912_191277


namespace NUMINAMATH_CALUDE_exp_sum_of_ln_l1912_191255

theorem exp_sum_of_ln (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.exp a + Real.exp b = 5 := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_of_ln_l1912_191255


namespace NUMINAMATH_CALUDE_vector_parallel_problem_l1912_191251

theorem vector_parallel_problem (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![-1, 1]
  let c : Fin 2 → ℝ := ![3, 0]
  (∃ (k : ℝ), a = k • (b + c)) → m = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_problem_l1912_191251


namespace NUMINAMATH_CALUDE_wilfred_carrot_consumption_l1912_191257

/-- The number of carrots Wilfred eats on Tuesday -/
def tuesday_carrots : ℕ := 4

/-- The number of carrots Wilfred eats on Wednesday -/
def wednesday_carrots : ℕ := 6

/-- The number of carrots Wilfred plans to eat on Thursday -/
def thursday_carrots : ℕ := 5

/-- The total number of carrots Wilfred wants to eat from Tuesday to Thursday -/
def total_carrots : ℕ := tuesday_carrots + wednesday_carrots + thursday_carrots

theorem wilfred_carrot_consumption :
  total_carrots = 15 := by sorry

end NUMINAMATH_CALUDE_wilfred_carrot_consumption_l1912_191257


namespace NUMINAMATH_CALUDE_reader_current_page_l1912_191256

/-- Represents a book with a given number of pages -/
structure Book where
  pages : ℕ

/-- Represents a reader with a constant reading rate -/
structure Reader where
  rate : ℕ  -- pages per hour

/-- The current state of reading -/
structure ReadingState where
  book : Book
  reader : Reader
  previousPage : ℕ
  hoursAgo : ℕ
  hoursLeft : ℕ

/-- Calculate the current page number of the reader -/
def currentPage (state : ReadingState) : ℕ :=
  state.previousPage + state.reader.rate * state.hoursAgo

/-- Theorem: Given the conditions, prove that the reader's current page is 90 -/
theorem reader_current_page
  (state : ReadingState)
  (h1 : state.book.pages = 210)
  (h2 : state.previousPage = 60)
  (h3 : state.hoursAgo = 1)
  (h4 : state.hoursLeft = 4)
  (h5 : currentPage state + state.reader.rate * state.hoursLeft = state.book.pages) :
  currentPage state = 90 := by
  sorry


end NUMINAMATH_CALUDE_reader_current_page_l1912_191256


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_quadratic_equation_solutions_l1912_191246

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x => 2 * x^2 + 4 * x - 6
  (f 1 = 0 ∧ f (-3) = 0) ∧ 
  (∀ x : ℝ, f x = 0 → x = 1 ∨ x = -3) :=
sorry

theorem linear_quadratic_equation_solutions :
  let g : ℝ → ℝ := λ x => 2 * (x - 3) - 3 * x * (x - 3)
  (g 3 = 0 ∧ g (2/3) = 0) ∧
  (∀ x : ℝ, g x = 0 → x = 3 ∨ x = 2/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_quadratic_equation_solutions_l1912_191246


namespace NUMINAMATH_CALUDE_triangle_theorem_l1912_191245

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A + B + C = π)

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  2 * t.a^2 = t.b^2 + t.c^2 ∧ 
  (t.a = 5 ∧ Real.cos t.A = 25/31 → t.a + t.b + t.c = 14) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1912_191245


namespace NUMINAMATH_CALUDE_maggi_ate_five_cupcakes_l1912_191239

/-- Calculates the number of cupcakes Maggi ate -/
def cupcakes_eaten (initial_packages : ℕ) (cupcakes_per_package : ℕ) (cupcakes_left : ℕ) : ℕ :=
  initial_packages * cupcakes_per_package - cupcakes_left

/-- Proves that Maggi ate 5 cupcakes -/
theorem maggi_ate_five_cupcakes :
  cupcakes_eaten 3 4 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_maggi_ate_five_cupcakes_l1912_191239


namespace NUMINAMATH_CALUDE_circle_area_increase_l1912_191222

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1912_191222


namespace NUMINAMATH_CALUDE_fourth_power_sum_of_roots_l1912_191210

theorem fourth_power_sum_of_roots (r₁ r₂ r₃ r₄ : ℝ) : 
  (r₁^4 - r₁ - 504 = 0) → 
  (r₂^4 - r₂ - 504 = 0) → 
  (r₃^4 - r₃ - 504 = 0) → 
  (r₄^4 - r₄ - 504 = 0) → 
  r₁^4 + r₂^4 + r₃^4 + r₄^4 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_of_roots_l1912_191210


namespace NUMINAMATH_CALUDE_smallest_box_volume_l1912_191269

/-- Represents a triangular pyramid (tetrahedron) -/
structure Pyramid where
  height : ℝ
  base_side : ℝ

/-- Represents a rectangular prism -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def box_volume (b : Box) : ℝ :=
  b.length * b.width * b.height

/-- Checks if a box can safely contain a pyramid -/
def can_contain (b : Box) (p : Pyramid) : Prop :=
  b.height ≥ p.height ∧ b.length ≥ p.base_side ∧ b.width ≥ p.base_side

/-- The smallest box that can safely contain the pyramid -/
def smallest_box (p : Pyramid) : Box :=
  { length := 10, width := 10, height := p.height }

/-- Theorem: The volume of the smallest box that can safely contain the given pyramid is 1500 cubic inches -/
theorem smallest_box_volume (p : Pyramid) (h1 : p.height = 15) (h2 : p.base_side = 8) :
  box_volume (smallest_box p) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_box_volume_l1912_191269


namespace NUMINAMATH_CALUDE_product_sum_of_digits_base8_l1912_191289

/-- Converts a base-8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base-8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Computes the sum of digits of a base-8 number --/
def sumOfDigitsBase8 (n : ℕ) : ℕ := sorry

/-- Theorem: The base-8 sum of digits of the product of 35₈ and 21₈ is 21₈ --/
theorem product_sum_of_digits_base8 :
  sumOfDigitsBase8 (decimalToBase8 (base8ToDecimal 35 * base8ToDecimal 21)) = 21 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_digits_base8_l1912_191289


namespace NUMINAMATH_CALUDE_parallelogram_x_value_l1912_191231

/-- A parallelogram ABCD with specific properties -/
structure Parallelogram where
  x : ℝ
  area : ℝ
  h : x > 0
  angle : ℝ
  h_angle : angle = 30 * π / 180
  h_area : area = 35

/-- The theorem stating that x = 14 for the given parallelogram -/
theorem parallelogram_x_value (p : Parallelogram) : p.x = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_x_value_l1912_191231


namespace NUMINAMATH_CALUDE_cookie_difference_l1912_191209

def sweet_cookies_initial : ℕ := 37
def salty_cookies_initial : ℕ := 11
def sweet_cookies_eaten : ℕ := 5
def salty_cookies_eaten : ℕ := 2

theorem cookie_difference : sweet_cookies_eaten - salty_cookies_eaten = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l1912_191209


namespace NUMINAMATH_CALUDE_man_to_son_age_ratio_l1912_191243

def son_age : ℕ := 20
def age_difference : ℕ := 22

def man_age : ℕ := son_age + age_difference

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem man_to_son_age_ratio :
  man_age_in_two_years / son_age_in_two_years = 2 ∧
  man_age_in_two_years % son_age_in_two_years = 0 := by
  sorry

#eval man_age_in_two_years / son_age_in_two_years

end NUMINAMATH_CALUDE_man_to_son_age_ratio_l1912_191243


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1912_191290

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1912_191290


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1912_191204

theorem trigonometric_identity : 
  Real.sin (135 * π / 180) * Real.cos (-15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1912_191204


namespace NUMINAMATH_CALUDE_equilateral_triangle_complex_plane_l1912_191242

theorem equilateral_triangle_complex_plane (z : ℂ) (μ : ℝ) : 
  Complex.abs z = 3 →
  μ > 2 →
  (Complex.abs (z^3 - z) = Complex.abs (μ • z - z) ∧
   Complex.abs (z^3 - μ • z) = Complex.abs (μ • z - z) ∧
   Complex.abs (z^3 - μ • z) = Complex.abs (z^3 - z)) →
  μ = 1 + Real.sqrt 82 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_complex_plane_l1912_191242


namespace NUMINAMATH_CALUDE_cos_negative_seventeen_thirds_pi_l1912_191208

theorem cos_negative_seventeen_thirds_pi : 
  Real.cos (-17/3 * Real.pi) = 1/2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_seventeen_thirds_pi_l1912_191208


namespace NUMINAMATH_CALUDE_equation_solution_l1912_191232

theorem equation_solution (x : ℝ) : 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 20) = (x^2 - 3*x - 18) / (x^2 - 2*x - 35) ↔ 
  x = 4 + Real.sqrt 21 ∨ x = 4 - Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1912_191232


namespace NUMINAMATH_CALUDE_andrey_gifts_l1912_191267

theorem andrey_gifts :
  ∃ (n a : ℕ), 
    n > 0 ∧ 
    a > 0 ∧ 
    n * (n - 2) = a * (n - 1) + 16 ∧ 
    n = 18 := by
  sorry

end NUMINAMATH_CALUDE_andrey_gifts_l1912_191267


namespace NUMINAMATH_CALUDE_taxi_fare_100_miles_l1912_191219

/-- The cost of a taxi trip given the distance traveled. -/
noncomputable def taxi_cost (base_fare : ℝ) (rate : ℝ) (distance : ℝ) : ℝ :=
  base_fare + rate * distance

theorem taxi_fare_100_miles :
  let base_fare : ℝ := 40
  let rate : ℝ := (200 - base_fare) / 80
  taxi_cost base_fare rate 100 = 240 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_100_miles_l1912_191219


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l1912_191202

theorem quadratic_equation_with_given_roots :
  ∀ (f : ℝ → ℝ),
  (∀ x, f x = 0 ↔ x = -5 ∨ x = 7) →
  (∀ x, f x = (x + 5) * (x - 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l1912_191202


namespace NUMINAMATH_CALUDE_summer_performs_1300_salutations_l1912_191254

/-- The number of sun salutations Summer performs throughout an entire year. -/
def summer_sun_salutations : ℕ :=
  let poses_per_day : ℕ := 5
  let weekdays_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem stating that Summer performs 1300 sun salutations throughout an entire year. -/
theorem summer_performs_1300_salutations : summer_sun_salutations = 1300 := by
  sorry

end NUMINAMATH_CALUDE_summer_performs_1300_salutations_l1912_191254


namespace NUMINAMATH_CALUDE_equation_solve_for_n_l1912_191280

theorem equation_solve_for_n (s P k c n : ℝ) (h1 : c > 0) (h2 : P = s / (c * (1 + k)^n)) :
  n = Real.log (s / (P * c)) / Real.log (1 + k) := by
  sorry

end NUMINAMATH_CALUDE_equation_solve_for_n_l1912_191280


namespace NUMINAMATH_CALUDE_sandy_age_l1912_191281

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 := by
  sorry

end NUMINAMATH_CALUDE_sandy_age_l1912_191281


namespace NUMINAMATH_CALUDE_furniture_assembly_time_l1912_191271

theorem furniture_assembly_time 
  (num_chairs : ℕ) 
  (num_tables : ℕ) 
  (time_per_piece : ℕ) 
  (h1 : num_chairs = 4) 
  (h2 : num_tables = 4) 
  (h3 : time_per_piece = 6) : 
  (num_chairs + num_tables) * time_per_piece = 48 := by
  sorry

end NUMINAMATH_CALUDE_furniture_assembly_time_l1912_191271


namespace NUMINAMATH_CALUDE_complex_sum_modulus_l1912_191249

theorem complex_sum_modulus : 
  Complex.abs (1/5 - (2/5)*Complex.I) + Complex.abs (3/5 + (4/5)*Complex.I) = (1 + Real.sqrt 5) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_modulus_l1912_191249


namespace NUMINAMATH_CALUDE_jellybeans_count_l1912_191266

/-- The number of jellybeans in a bag with black, green, and orange beans -/
def total_jellybeans (black green orange : ℕ) : ℕ := black + green + orange

/-- Theorem: Given the conditions, the total number of jellybeans is 27 -/
theorem jellybeans_count :
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  orange = green - 1 →
  total_jellybeans black green orange = 27 := by
sorry

end NUMINAMATH_CALUDE_jellybeans_count_l1912_191266


namespace NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l1912_191261

theorem acute_triangle_tangent_inequality (A B C : Real) (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C < π) :
  (1 / 3) * ((Real.tan A)^2 / (Real.tan B * Real.tan C) +
             (Real.tan B)^2 / (Real.tan C * Real.tan A) +
             (Real.tan C)^2 / (Real.tan A * Real.tan B)) +
  3 * (1 / (Real.tan A + Real.tan B + Real.tan C))^(2/3) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_tangent_inequality_l1912_191261


namespace NUMINAMATH_CALUDE_algebra_books_not_unique_l1912_191220

/-- Represents the number of books on a shelf -/
structure ShelfBooks where
  algebra : ℕ+
  geometry : ℕ+

/-- Represents the two shelves in the library -/
structure Library where
  longer_shelf : ShelfBooks
  shorter_shelf : ShelfBooks
  algebra_only : ℕ+

/-- The conditions of the library problem -/
def LibraryProblem (lib : Library) : Prop :=
  lib.longer_shelf.algebra > lib.shorter_shelf.algebra ∧
  lib.longer_shelf.geometry < lib.shorter_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.longer_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.shorter_shelf.algebra ∧
  lib.longer_shelf.algebra ≠ lib.shorter_shelf.geometry ∧
  lib.longer_shelf.geometry ≠ lib.shorter_shelf.algebra ∧
  lib.longer_shelf.geometry ≠ lib.shorter_shelf.geometry ∧
  lib.shorter_shelf.algebra ≠ lib.shorter_shelf.geometry ∧
  lib.longer_shelf.algebra ≠ lib.algebra_only ∧
  lib.longer_shelf.geometry ≠ lib.algebra_only ∧
  lib.shorter_shelf.algebra ≠ lib.algebra_only ∧
  lib.shorter_shelf.geometry ≠ lib.algebra_only

/-- The theorem stating that the number of algebra books to fill the longer shelf cannot be uniquely determined -/
theorem algebra_books_not_unique (lib : Library) (h : LibraryProblem lib) :
  ∃ (lib' : Library), LibraryProblem lib' ∧ lib'.algebra_only ≠ lib.algebra_only :=
sorry

end NUMINAMATH_CALUDE_algebra_books_not_unique_l1912_191220


namespace NUMINAMATH_CALUDE_salary_increase_proof_l1912_191253

theorem salary_increase_proof (original_salary : ℝ) 
  (h1 : original_salary * 1.8 = 25000) 
  (h2 : original_salary > 0) : 
  25000 - original_salary = 11111.11 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l1912_191253


namespace NUMINAMATH_CALUDE_average_value_function_2x_squared_average_value_function_exponential_l1912_191200

/-- Definition of average value function on [a,b] -/
def is_average_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The function f(x) = 2x^2 is an average value function on [-1,1] with average value point 0 -/
theorem average_value_function_2x_squared :
  is_average_value_function (fun x => 2 * x^2) (-1) 1 ∧
  (fun x => 2 * x^2) 0 = ((fun x => 2 * x^2) 1 - (fun x => 2 * x^2) (-1)) / (1 - (-1)) :=
sorry

/-- The function g(x) = -2^(2x+1) + m⋅2^(x+1) + 1 is an average value function on [-1,1]
    if and only if m ∈ (-∞, 13/10) ∪ (17/2, +∞) -/
theorem average_value_function_exponential (m : ℝ) :
  is_average_value_function (fun x => -2^(2*x+1) + m * 2^(x+1) + 1) (-1) 1 ↔
  m < 13/10 ∨ m > 17/2 :=
sorry

end NUMINAMATH_CALUDE_average_value_function_2x_squared_average_value_function_exponential_l1912_191200


namespace NUMINAMATH_CALUDE_distance_A_l1912_191292

/-- Given points A, B, and C in the plane, and points A' and B' on the line y = x,
    prove that the distance between A' and B' is 5√2. -/
theorem distance_A'B' (A B C A' B' : ℝ × ℝ) : 
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (∃ t : ℝ, A + t • (C - A) = A') →
  (∃ s : ℝ, B + s • (C - B) = B') →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_A_l1912_191292


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1912_191286

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℚ, (3 * x - 1)^7 = a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ + a = 128 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1912_191286


namespace NUMINAMATH_CALUDE_impossibleConfig6_impossibleConfig4_impossibleConfig3_l1912_191213

/-- Represents the sign at a vertex -/
inductive Sign
| Positive
| Negative

/-- Represents a dodecagon configuration -/
def DodecagonConfig := Fin 12 → Sign

/-- Initial configuration with A₁ negative and others positive -/
def initialConfig : DodecagonConfig :=
  fun i => if i = 0 then Sign.Negative else Sign.Positive

/-- Applies the sign-flipping operation to n consecutive vertices starting at index i -/
def flipSigns (config : DodecagonConfig) (i : Fin 12) (n : Nat) : DodecagonConfig :=
  fun j => if (j - i) % 12 < n then
    match config j with
    | Sign.Positive => Sign.Negative
    | Sign.Negative => Sign.Positive
    else config j

/-- Checks if only A₂ is negative in the configuration -/
def onlyA₂Negative (config : DodecagonConfig) : Prop :=
  config 1 = Sign.Negative ∧ ∀ i, i ≠ 1 → config i = Sign.Positive

/-- Main theorem for 6 consecutive vertices -/
theorem impossibleConfig6 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 6) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

/-- Main theorem for 4 consecutive vertices -/
theorem impossibleConfig4 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 4) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

/-- Main theorem for 3 consecutive vertices -/
theorem impossibleConfig3 :
  ∀ (ops : List (Fin 12)), 
  let finalConfig := (ops.foldl (fun cfg i => flipSigns cfg i 3) initialConfig)
  ¬(onlyA₂Negative finalConfig) := by sorry

end NUMINAMATH_CALUDE_impossibleConfig6_impossibleConfig4_impossibleConfig3_l1912_191213


namespace NUMINAMATH_CALUDE_sum_of_odd_integers_13_to_41_l1912_191226

theorem sum_of_odd_integers_13_to_41 :
  let first_term : ℕ := 13
  let last_term : ℕ := 41
  let common_difference : ℕ := 2
  let n : ℕ := (last_term - first_term) / common_difference + 1
  (n : ℝ) / 2 * (first_term + last_term) = 405 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_odd_integers_13_to_41_l1912_191226


namespace NUMINAMATH_CALUDE_doris_eggs_l1912_191252

/-- Represents the number of eggs in a package -/
inductive EggPackage
  | small : EggPackage
  | large : EggPackage

/-- Returns the number of eggs in a package -/
def eggs_in_package (p : EggPackage) : Nat :=
  match p with
  | EggPackage.small => 6
  | EggPackage.large => 11

/-- Calculates the total number of eggs bought given the number of large packs -/
def total_eggs (large_packs : Nat) : Nat :=
  large_packs * eggs_in_package EggPackage.large

/-- Proves that Doris bought 55 eggs in total -/
theorem doris_eggs :
  total_eggs 5 = 55 := by sorry

end NUMINAMATH_CALUDE_doris_eggs_l1912_191252


namespace NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l1912_191259

theorem difference_of_odd_squares_divisible_by_eight (a b : Int) 
  (ha : a % 2 = 1) (hb : b % 2 = 1) : 
  ∃ k : Int, a^2 - b^2 = 8 * k := by
sorry

end NUMINAMATH_CALUDE_difference_of_odd_squares_divisible_by_eight_l1912_191259


namespace NUMINAMATH_CALUDE_last_digit_2008_2005_l1912_191262

theorem last_digit_2008_2005 : (2008^2005) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_2008_2005_l1912_191262


namespace NUMINAMATH_CALUDE_total_initials_eq_thousand_l1912_191296

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def initials_length : ℕ := 3

/-- The total number of possible three-letter sets of initials using letters A through J -/
def total_initials : ℕ := num_letters ^ initials_length

/-- Theorem stating that the total number of possible three-letter sets of initials
    using letters A through J is 1000 -/
theorem total_initials_eq_thousand : total_initials = 1000 := by
  sorry

end NUMINAMATH_CALUDE_total_initials_eq_thousand_l1912_191296


namespace NUMINAMATH_CALUDE_fifa_world_cup_2010_matches_l1912_191291

/-- Calculates the number of matches in a round-robin tournament -/
def roundRobinMatches (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the number of matches in a knockout tournament -/
def knockoutMatches (n : Nat) : Nat :=
  n - 1

theorem fifa_world_cup_2010_matches : 
  let totalTeams : Nat := 24
  let groups : Nat := 6
  let teamsPerGroup : Nat := 4
  let knockoutTeams : Nat := 16
  let firstRoundMatches := groups * roundRobinMatches teamsPerGroup
  let knockoutStageMatches := knockoutMatches knockoutTeams
  firstRoundMatches + knockoutStageMatches = 51 := by
  sorry

end NUMINAMATH_CALUDE_fifa_world_cup_2010_matches_l1912_191291


namespace NUMINAMATH_CALUDE_max_product_dice_rolls_l1912_191207

theorem max_product_dice_rolls (rolls : List Nat) : 
  rolls.length = 25 → 
  (∀ x ∈ rolls, 1 ≤ x ∧ x ≤ 20) →
  rolls.sum = 70 →
  rolls.prod ≤ (List.replicate 5 2 ++ List.replicate 20 3).prod :=
sorry

end NUMINAMATH_CALUDE_max_product_dice_rolls_l1912_191207


namespace NUMINAMATH_CALUDE_school_location_minimizes_distance_l1912_191216

/-- Represents the distance between two villages in kilometers -/
def village_distance : ℝ := 3

/-- Represents the number of students in village A -/
def students_A : ℕ := 300

/-- Represents the number of students in village B -/
def students_B : ℕ := 200

/-- Represents the distance from village A to the school -/
def school_distance (x : ℝ) : ℝ := x

/-- Calculates the total distance traveled by all students -/
def total_distance (x : ℝ) : ℝ :=
  students_A * school_distance x + students_B * (village_distance - school_distance x)

/-- Theorem: The total distance is minimized when the school is built in village A -/
theorem school_location_minimizes_distance :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ village_distance →
    total_distance 0 ≤ total_distance x :=
by sorry

end NUMINAMATH_CALUDE_school_location_minimizes_distance_l1912_191216


namespace NUMINAMATH_CALUDE_clothing_distribution_l1912_191272

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 47)
  (h2 : first_load = 17)
  (h3 : num_small_loads = 5) :
  (total - first_load) / num_small_loads = 6 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l1912_191272


namespace NUMINAMATH_CALUDE_volleyball_team_score_l1912_191228

theorem volleyball_team_score :
  let lizzie_score : ℕ := 4
  let nathalie_score : ℕ := lizzie_score + 3
  let aimee_score : ℕ := 2 * (lizzie_score + nathalie_score)
  let teammates_score : ℕ := 17
  lizzie_score + nathalie_score + aimee_score + teammates_score = 50 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_score_l1912_191228


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l1912_191236

theorem basketball_lineup_combinations (n : ℕ) (k₁ k₂ k₃ : ℕ) 
  (h₁ : n = 12) (h₂ : k₁ = 2) (h₃ : k₂ = 2) (h₄ : k₃ = 1) : 
  (n.choose k₁) * ((n - k₁).choose k₂) * ((n - k₁ - k₂).choose k₃) = 23760 :=
by sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l1912_191236


namespace NUMINAMATH_CALUDE_square_ratio_theorem_l1912_191258

theorem square_ratio_theorem :
  let area_ratio : ℚ := 18 / 98
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  ∃ (a b c : ℕ), 
    side_ratio = (a : ℝ) * Real.sqrt b / c ∧
    a = 3 ∧ b = 1 ∧ c = 7 ∧
    a + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_theorem_l1912_191258
