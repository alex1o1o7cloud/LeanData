import Mathlib

namespace NUMINAMATH_CALUDE_family_gathering_handshakes_l1443_144330

theorem family_gathering_handshakes :
  let num_twin_sets : ℕ := 10
  let num_quadruplet_sets : ℕ := 5
  let num_twins : ℕ := num_twin_sets * 2
  let num_quadruplets : ℕ := num_quadruplet_sets * 4
  let twin_handshakes : ℕ := num_twins * (num_twins - 2)
  let quadruplet_handshakes : ℕ := num_quadruplets * (num_quadruplets - 4)
  let twin_to_quadruplet : ℕ := num_twins * (2 * num_quadruplets / 3)
  let quadruplet_to_twin : ℕ := num_quadruplets * (3 * num_twins / 4)
  let total_handshakes : ℕ := (twin_handshakes + quadruplet_handshakes + twin_to_quadruplet + quadruplet_to_twin) / 2
  total_handshakes = 620 :=
by sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_l1443_144330


namespace NUMINAMATH_CALUDE_exam_question_count_l1443_144365

/-- Represents the scoring system and results of an examination. -/
structure ExamResult where
  correct_score : ℕ  -- Score for each correct answer
  wrong_penalty : ℕ  -- Penalty for each wrong answer
  total_score : ℤ    -- Total score achieved
  correct_count : ℕ  -- Number of correctly answered questions
  total_count : ℕ    -- Total number of questions attempted

/-- Theorem stating the relationship between exam parameters and the total number of questions attempted. -/
theorem exam_question_count 
  (exam : ExamResult) 
  (h1 : exam.correct_score = 4)
  (h2 : exam.wrong_penalty = 1)
  (h3 : exam.total_score = 130)
  (h4 : exam.correct_count = 42) :
  exam.total_count = 80 := by
  sorry

#check exam_question_count

end NUMINAMATH_CALUDE_exam_question_count_l1443_144365


namespace NUMINAMATH_CALUDE_fill_measuring_cup_l1443_144304

/-- The capacity of a spoon in milliliters -/
def spoon_capacity : ℝ := 5

/-- The volume of a measuring cup in liters -/
def cup_volume : ℝ := 1

/-- The conversion factor from liters to milliliters -/
def liter_to_ml : ℝ := 1000

/-- The number of spoons needed to fill the measuring cup -/
def spoons_needed : ℕ := 200

theorem fill_measuring_cup : 
  ⌊(cup_volume * liter_to_ml) / spoon_capacity⌋ = spoons_needed := by
  sorry

end NUMINAMATH_CALUDE_fill_measuring_cup_l1443_144304


namespace NUMINAMATH_CALUDE_work_completion_time_l1443_144351

/-- 
Given that Paul completes a piece of work in 80 days and Rose completes the same work in 120 days,
prove that they will complete the work together in 48 days.
-/
theorem work_completion_time 
  (paul_time : ℕ) 
  (rose_time : ℕ) 
  (h_paul : paul_time = 80) 
  (h_rose : rose_time = 120) : 
  (paul_time * rose_time) / (paul_time + rose_time) = 48 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1443_144351


namespace NUMINAMATH_CALUDE_gcd_lcm_888_1147_l1443_144302

theorem gcd_lcm_888_1147 : 
  (Nat.gcd 888 1147 = 37) ∧ (Nat.lcm 888 1147 = 27528) := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_888_1147_l1443_144302


namespace NUMINAMATH_CALUDE_correlation_significance_l1443_144312

/-- The critical value for a 5% significance level -/
def r_0_05 : ℝ := sorry

/-- The observed correlation coefficient -/
def r : ℝ := sorry

/-- An event with a probability of less than 5% -/
def low_probability_event : Prop := sorry

theorem correlation_significance :
  |r| > r_0_05 → low_probability_event := by sorry

end NUMINAMATH_CALUDE_correlation_significance_l1443_144312


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1443_144321

/-- Given an article sold at $1800 with a 20% profit, prove that the cost price is $1500. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 1800)
  (h2 : profit_percentage = 20) :
  selling_price / (1 + profit_percentage / 100) = 1500 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1443_144321


namespace NUMINAMATH_CALUDE_josiah_cookie_spending_l1443_144387

/-- The number of days in March -/
def days_in_march : ℕ := 31

/-- The number of cookies Josiah buys each day -/
def cookies_per_day : ℕ := 2

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 16

/-- Josiah's total spending on cookies in March -/
def total_spending : ℕ := days_in_march * cookies_per_day * cost_per_cookie

/-- Theorem stating that Josiah's total spending on cookies in March is 992 dollars -/
theorem josiah_cookie_spending : total_spending = 992 := by
  sorry

end NUMINAMATH_CALUDE_josiah_cookie_spending_l1443_144387


namespace NUMINAMATH_CALUDE_integral_x_plus_cos_2x_over_symmetric_interval_l1443_144361

theorem integral_x_plus_cos_2x_over_symmetric_interval : 
  ∫ x in (-π/2)..(π/2), (x + Real.cos (2*x)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_plus_cos_2x_over_symmetric_interval_l1443_144361


namespace NUMINAMATH_CALUDE_f_range_l1443_144323

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (Real.log x + 1)

theorem f_range (m n : ℝ) (hm : m > Real.exp 1) (hn : n > Real.exp 1)
  (h : f m = 2 * Real.log (Real.sqrt (Real.exp 1)) - f n) :
  5/7 ≤ f (m * n) ∧ f (m * n) < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l1443_144323


namespace NUMINAMATH_CALUDE_second_day_speed_l1443_144372

/-- Given a journey with specific conditions, prove the speed on the second day -/
theorem second_day_speed 
  (distance : ℝ) 
  (first_day_speed : ℝ) 
  (normal_time : ℝ) 
  (first_day_delay : ℝ) 
  (second_day_early : ℝ) 
  (h1 : distance = 60) 
  (h2 : first_day_speed = 10) 
  (h3 : normal_time = distance / first_day_speed) 
  (h4 : first_day_delay = 2) 
  (h5 : second_day_early = 1) : 
  distance / (normal_time - second_day_early) = 12 := by
sorry

end NUMINAMATH_CALUDE_second_day_speed_l1443_144372


namespace NUMINAMATH_CALUDE_unique_arrangement_l1443_144349

/-- Represents a 4x4 grid with letters A, B, and C --/
def Grid := Fin 4 → Fin 4 → Char

/-- Checks if a given grid satisfies the arrangement conditions --/
def valid_arrangement (g : Grid) : Prop :=
  -- A is in the upper left corner
  g 0 0 = 'A' ∧
  -- Each row contains one of each letter
  (∀ i : Fin 4, ∃ j₁ j₂ j₃ : Fin 4, g i j₁ = 'A' ∧ g i j₂ = 'B' ∧ g i j₃ = 'C') ∧
  -- Each column contains one of each letter
  (∀ j : Fin 4, ∃ i₁ i₂ i₃ : Fin 4, g i₁ j = 'A' ∧ g i₂ j = 'B' ∧ g i₃ j = 'C') ∧
  -- Main diagonal (top-left to bottom-right) contains one of each letter
  (∃ i₁ i₂ i₃ : Fin 4, g i₁ i₁ = 'A' ∧ g i₂ i₂ = 'B' ∧ g i₃ i₃ = 'C') ∧
  -- Anti-diagonal (top-right to bottom-left) contains one of each letter
  (∃ i₁ i₂ i₃ : Fin 4, g i₁ (3 - i₁) = 'A' ∧ g i₂ (3 - i₂) = 'B' ∧ g i₃ (3 - i₃) = 'C')

/-- The main theorem stating there is only one valid arrangement --/
theorem unique_arrangement : ∃! g : Grid, valid_arrangement g :=
  sorry

end NUMINAMATH_CALUDE_unique_arrangement_l1443_144349


namespace NUMINAMATH_CALUDE_radium_decay_heat_equivalence_l1443_144316

/-- The amount of radium in the Earth's crust in kilograms -/
def radium_in_crust : ℝ := 10000000000

/-- The amount of coal in kilograms that releases equivalent heat to 1 kg of radium decay -/
def coal_equivalent : ℝ := 375000

/-- The amount of coal in kilograms that releases equivalent heat to the complete decay of radium in Earth's crust -/
def total_coal_equivalent : ℝ := radium_in_crust * coal_equivalent

theorem radium_decay_heat_equivalence :
  total_coal_equivalent = 3.75 * (10 ^ 15) := by
  sorry

end NUMINAMATH_CALUDE_radium_decay_heat_equivalence_l1443_144316


namespace NUMINAMATH_CALUDE_original_number_is_ten_l1443_144357

theorem original_number_is_ten : ∃ x : ℝ, (2 * x + 5 = x / 2 + 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_ten_l1443_144357


namespace NUMINAMATH_CALUDE_no_solution_l1443_144363

/-- Product of digits of a natural number in base ten -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem: no natural number satisfies the given equation -/
theorem no_solution :
  ∀ x : ℕ, productOfDigits x ≠ x^2 - 10*x - 22 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1443_144363


namespace NUMINAMATH_CALUDE_subset_empty_range_superset_range_l1443_144306

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

-- Theorem for part I
theorem subset_empty_range : ¬∃ a : ℝ, M ⊆ N a := by sorry

-- Theorem for part II
theorem superset_range : {a : ℝ | M ⊇ N a} = {a : ℝ | a ≤ 3} := by sorry

end NUMINAMATH_CALUDE_subset_empty_range_superset_range_l1443_144306


namespace NUMINAMATH_CALUDE_max_n_sin_cos_inequality_l1443_144322

theorem max_n_sin_cos_inequality : 
  (∃ (n : ℕ), n > 0 ∧ ∀ (x : ℝ), (Real.sin x)^n + (Real.cos x)^n ≥ 1 / n) ∧ 
  (∀ (m : ℕ), m > 8 → ∃ (x : ℝ), (Real.sin x)^m + (Real.cos x)^m < 1 / m) :=
by sorry

end NUMINAMATH_CALUDE_max_n_sin_cos_inequality_l1443_144322


namespace NUMINAMATH_CALUDE_find_special_number_l1443_144386

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℤ) : Prop := ∃ k : ℤ, n = k^2

/-- The problem statement -/
theorem find_special_number : 
  ∃ m : ℕ+, 
    is_perfect_square (m.val + 100) ∧ 
    is_perfect_square (m.val + 168) ∧ 
    m.val = 156 := by
  sorry

end NUMINAMATH_CALUDE_find_special_number_l1443_144386


namespace NUMINAMATH_CALUDE_kids_at_reunion_l1443_144364

theorem kids_at_reunion (adults : ℕ) (tables : ℕ) (people_per_table : ℕ) 
  (h1 : adults = 123)
  (h2 : tables = 14)
  (h3 : people_per_table = 12) :
  tables * people_per_table - adults = 45 :=
by sorry

end NUMINAMATH_CALUDE_kids_at_reunion_l1443_144364


namespace NUMINAMATH_CALUDE_sin_2x_value_l1443_144327

theorem sin_2x_value (x : ℝ) (h : Real.tan (π - x) = 3) : Real.sin (2 * x) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_value_l1443_144327


namespace NUMINAMATH_CALUDE_ben_is_25_l1443_144305

/-- Ben's age -/
def ben_age : ℕ := sorry

/-- Dan's age -/
def dan_age : ℕ := sorry

/-- Ben is 3 years younger than Dan -/
axiom age_difference : ben_age = dan_age - 3

/-- The sum of their ages is 53 -/
axiom age_sum : ben_age + dan_age = 53

theorem ben_is_25 : ben_age = 25 := by sorry

end NUMINAMATH_CALUDE_ben_is_25_l1443_144305


namespace NUMINAMATH_CALUDE_root_range_implies_m_range_l1443_144370

theorem root_range_implies_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ x₂ > 1 ∧ 
   x₁^2 + (m^2 - 1)*x₁ + m - 2 = 0 ∧
   x₂^2 + (m^2 - 1)*x₂ + m - 2 = 0) →
  -2 < m ∧ m < 0 := by
sorry

end NUMINAMATH_CALUDE_root_range_implies_m_range_l1443_144370


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1443_144308

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  3 * X^3 - 4 * X^2 - 23 * X + 60 = (X - 3) * q + 36 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1443_144308


namespace NUMINAMATH_CALUDE_remaining_game_price_l1443_144333

def total_games : ℕ := 346
def expensive_games : ℕ := 80
def expensive_price : ℕ := 12
def mid_price : ℕ := 7
def total_spent : ℕ := 2290

theorem remaining_game_price :
  let remaining_games := total_games - expensive_games
  let mid_games := remaining_games / 2
  let cheap_games := remaining_games - mid_games
  let spent_on_expensive := expensive_games * expensive_price
  let spent_on_mid := mid_games * mid_price
  let spent_on_cheap := total_spent - spent_on_expensive - spent_on_mid
  spent_on_cheap / cheap_games = 3 := by
sorry

end NUMINAMATH_CALUDE_remaining_game_price_l1443_144333


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1443_144324

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + m = 0

def line_equation (x y : ℝ) : Prop :=
  x + 2*y - 4 = 0

def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem circle_intersection_theorem :
  ∃ (x₁ y₁ x₂ y₂ m : ℝ),
    circle_equation x₁ y₁ m ∧
    circle_equation x₂ y₂ m ∧
    line_equation x₁ y₁ ∧
    line_equation x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂ →
    m = 8/5 ∧
    ∀ (x y : ℝ), (x - 4/5)^2 + (y - 8/5)^2 = 16/5 ↔
      circle_equation x y (8/5) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1443_144324


namespace NUMINAMATH_CALUDE_unique_valid_assignment_l1443_144319

/-- Represents the possible arithmetic operations --/
inductive Operation
  | Plus
  | Minus
  | Multiply
  | Divide
  | Equal

/-- Represents the assignment of operations to letters --/
structure Assignment :=
  (A B C D E : Operation)

/-- Checks if an assignment is valid according to the problem conditions --/
def is_valid_assignment (a : Assignment) : Prop :=
  a.A ≠ a.B ∧ a.A ≠ a.C ∧ a.A ≠ a.D ∧ a.A ≠ a.E ∧
  a.B ≠ a.C ∧ a.B ≠ a.D ∧ a.B ≠ a.E ∧
  a.C ≠ a.D ∧ a.C ≠ a.E ∧
  a.D ≠ a.E ∧
  (a.A = Operation.Plus ∨ a.B = Operation.Plus ∨ a.C = Operation.Plus ∨ a.D = Operation.Plus ∨ a.E = Operation.Plus) ∧
  (a.A = Operation.Minus ∨ a.B = Operation.Minus ∨ a.C = Operation.Minus ∨ a.D = Operation.Minus ∨ a.E = Operation.Minus) ∧
  (a.A = Operation.Multiply ∨ a.B = Operation.Multiply ∨ a.C = Operation.Multiply ∨ a.D = Operation.Multiply ∨ a.E = Operation.Multiply) ∧
  (a.A = Operation.Divide ∨ a.B = Operation.Divide ∨ a.C = Operation.Divide ∨ a.D = Operation.Divide ∨ a.E = Operation.Divide) ∧
  (a.A = Operation.Equal ∨ a.B = Operation.Equal ∨ a.C = Operation.Equal ∨ a.D = Operation.Equal ∨ a.E = Operation.Equal)

/-- Checks if an assignment satisfies the equations --/
def satisfies_equations (a : Assignment) : Prop :=
  (a.A = Operation.Divide ∧ 4 / 2 = 2) ∧
  (a.B = Operation.Equal) ∧
  (a.C = Operation.Multiply ∧ 4 * 2 = 8) ∧
  (a.D = Operation.Plus ∧ 2 + 3 = 5) ∧
  (a.E = Operation.Minus ∧ 5 - 1 = 4)

/-- The main theorem: there is a unique valid assignment that satisfies the equations --/
theorem unique_valid_assignment :
  ∃! (a : Assignment), is_valid_assignment a ∧ satisfies_equations a :=
sorry

end NUMINAMATH_CALUDE_unique_valid_assignment_l1443_144319


namespace NUMINAMATH_CALUDE_max_value_location_l1443_144360

theorem max_value_location (f : ℝ → ℝ) (a b : ℝ) (h : a < b) :
  Differentiable ℝ f → ∃ x ∈ Set.Icc a b,
    (∀ y ∈ Set.Icc a b, f y ≤ f x) ∧
    (x = a ∨ x = b ∨ deriv f x = 0) :=
sorry

end NUMINAMATH_CALUDE_max_value_location_l1443_144360


namespace NUMINAMATH_CALUDE_lowest_unique_score_above_90_l1443_144359

/-- Represents the scoring system for the modified AHSME exam -/
def score (c w : ℕ) : ℕ := 35 + 4 * c - w

/-- The total number of questions in the exam -/
def total_questions : ℕ := 35

theorem lowest_unique_score_above_90 :
  ∀ s : ℕ,
  s > 90 →
  (∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s) →
  (∀ s' : ℕ, 90 < s' ∧ s' < s → ¬∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s') →
  s = 91 :=
sorry

end NUMINAMATH_CALUDE_lowest_unique_score_above_90_l1443_144359


namespace NUMINAMATH_CALUDE_prove_birds_and_storks_l1443_144382

def birds_and_storks_problem : Prop :=
  let initial_birds : ℕ := 3
  let initial_storks : ℕ := 4
  let birds_arrived : ℕ := 2
  let birds_left : ℕ := 1
  let storks_arrived : ℕ := 3
  let final_birds : ℕ := initial_birds + birds_arrived - birds_left
  let final_storks : ℕ := initial_storks + storks_arrived
  (final_birds : Int) - (final_storks : Int) = -3

theorem prove_birds_and_storks : birds_and_storks_problem := by
  sorry

end NUMINAMATH_CALUDE_prove_birds_and_storks_l1443_144382


namespace NUMINAMATH_CALUDE_b_25_mod_55_l1443_144397

/-- Definition of b_n as a function that concatenates integers from 5 to n+4 -/
def b (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

/-- Theorem stating that b_25 mod 55 = 39 -/
theorem b_25_mod_55 : b 25 % 55 = 39 := by
  sorry

end NUMINAMATH_CALUDE_b_25_mod_55_l1443_144397


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1443_144331

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours -/
def totalDistance (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  (hours * (2 * initialSpeed + (hours - 1) * speedIncrease)) / 2

/-- Theorem stating that a car with given initial speed and speed increase travels a specific distance in 12 hours -/
theorem car_distance_theorem (initialSpeed : ℕ) (speedIncrease : ℕ) (hours : ℕ) :
  initialSpeed = 40 ∧ speedIncrease = 2 ∧ hours = 12 →
  totalDistance initialSpeed speedIncrease hours = 606 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l1443_144331


namespace NUMINAMATH_CALUDE_fourteenSidedPolygonArea_l1443_144383

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a polygon defined by a list of vertices -/
structure Polygon where
  vertices : List Point

/-- Calculates the area of a polygon given its vertices -/
def calculatePolygonArea (p : Polygon) : ℝ := sorry

/-- The fourteen-sided polygon from the problem -/
def fourteenSidedPolygon : Polygon :=
  { vertices := [
      { x := 1, y := 2 }, { x := 2, y := 2 }, { x := 3, y := 3 }, { x := 3, y := 4 },
      { x := 4, y := 5 }, { x := 5, y := 5 }, { x := 6, y := 5 }, { x := 6, y := 4 },
      { x := 5, y := 3 }, { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 3, y := 1 },
      { x := 2, y := 1 }, { x := 1, y := 1 }
    ]
  }

/-- Theorem stating that the area of the fourteen-sided polygon is 14 square centimeters -/
theorem fourteenSidedPolygonArea :
  calculatePolygonArea fourteenSidedPolygon = 14 := by sorry

end NUMINAMATH_CALUDE_fourteenSidedPolygonArea_l1443_144383


namespace NUMINAMATH_CALUDE_tom_initial_investment_l1443_144343

/-- Represents the initial investment of Tom in rupees -/
def tom_investment : ℕ := 30000

/-- Represents Jose's investment in rupees -/
def jose_investment : ℕ := 45000

/-- Represents the total profit after one year in rupees -/
def total_profit : ℕ := 63000

/-- Represents Jose's share of the profit in rupees -/
def jose_profit : ℕ := 35000

/-- Represents the number of months Tom invested -/
def tom_months : ℕ := 12

/-- Represents the number of months Jose invested -/
def jose_months : ℕ := 10

theorem tom_initial_investment :
  tom_investment * tom_months * jose_profit = jose_investment * jose_months * (total_profit - jose_profit) :=
sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l1443_144343


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1443_144352

theorem unique_integer_solution (a b c : ℤ) : 
  a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1443_144352


namespace NUMINAMATH_CALUDE_emily_sixth_score_l1443_144374

def emily_scores : List ℕ := [94, 97, 88, 90, 102]
def target_mean : ℚ := 95
def num_quizzes : ℕ := 6

theorem emily_sixth_score (sixth_score : ℕ) : 
  sixth_score = 99 →
  (emily_scores.sum + sixth_score) / num_quizzes = target_mean := by
sorry

end NUMINAMATH_CALUDE_emily_sixth_score_l1443_144374


namespace NUMINAMATH_CALUDE_ceva_triangle_ratio_product_l1443_144348

/-- Given a triangle ABC with points A', B', C' on sides BC, AC, AB respectively,
    and lines AA', BB', CC' intersecting at point O, if the sum of the ratios
    AO/OA', BO/OB', and CO/OC' is 56, then the square of their product is 2916. -/
theorem ceva_triangle_ratio_product (A B C A' B' C' O : ℝ × ℝ) : 
  let ratio (P Q R : ℝ × ℝ) := dist P Q / dist Q R
  (ratio O A A' + ratio O B B' + ratio O C C' = 56) →
  (ratio O A A' * ratio O B B' * ratio O C C')^2 = 2916 := by
  sorry

end NUMINAMATH_CALUDE_ceva_triangle_ratio_product_l1443_144348


namespace NUMINAMATH_CALUDE_simplify_expression_l1443_144354

theorem simplify_expression (x y : ℝ) : (5 - 2*x) - (8 - 6*x + 3*y) = -3 + 4*x - 3*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1443_144354


namespace NUMINAMATH_CALUDE_system_solution_l1443_144325

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 5) (h2 : x + 2 * y = 6) : x - y = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1443_144325


namespace NUMINAMATH_CALUDE_paint_needed_paint_problem_l1443_144332

theorem paint_needed (initial_paint : ℚ) (day1_fraction : ℚ) (day2_fraction : ℚ) (additional_needed : ℚ) : ℚ :=
  let remaining_after_day1 := initial_paint - day1_fraction * initial_paint
  let remaining_after_day2 := remaining_after_day1 - day2_fraction * remaining_after_day1
  let total_needed := remaining_after_day2 + additional_needed
  total_needed - remaining_after_day2

theorem paint_problem : paint_needed 2 (1/4) (1/2) (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_paint_needed_paint_problem_l1443_144332


namespace NUMINAMATH_CALUDE_line_quadrants_l1443_144369

/-- Given a line ax + by + c = 0 where ab < 0 and bc < 0, 
    the line passes through the first, second, and third quadrants -/
theorem line_quadrants (a b c : ℝ) (hab : a * b < 0) (hbc : b * c < 0) :
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1 > 0 ∧ y1 > 0) ∧  -- First quadrant
    (x2 < 0 ∧ y2 > 0) ∧  -- Second quadrant
    (x3 < 0 ∧ y3 < 0) ∧  -- Third quadrant
    (a * x1 + b * y1 + c = 0) ∧
    (a * x2 + b * y2 + c = 0) ∧
    (a * x3 + b * y3 + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_quadrants_l1443_144369


namespace NUMINAMATH_CALUDE_ellipse_theorem_l1443_144381

-- Define the ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that a > b > 0
def size_condition (a b : ℝ) : Prop := a > b ∧ b > 0

-- Define the angle condition
def angle_condition (PF1F2_angle : ℝ) : Prop := Real.sin PF1F2_angle = 1/3

-- Main theorem
theorem ellipse_theorem (a b : ℝ) (h1 : size_condition a b) 
  (h2 : ∃ P F1 F2 : ℝ × ℝ, 
    ellipse a b (F2.1) (P.2) ∧ 
    angle_condition (Real.arcsin (1/3))) : 
  a = Real.sqrt 2 * b := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l1443_144381


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l1443_144335

/-- The sum of an infinite geometric series with first term h and common ratio 0.8 is equal to 5h -/
theorem ball_bounce_distance (h : ℝ) (h_pos : h > 0) : 
  (∑' n, h * (0.8 ^ n)) = 5 * h := by sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l1443_144335


namespace NUMINAMATH_CALUDE_books_read_during_travel_l1443_144394

theorem books_read_during_travel (total_distance : ℕ) (reading_rate : ℕ) (books_finished : ℕ) : 
  total_distance = 6760 → reading_rate = 450 → books_finished = total_distance / reading_rate → books_finished = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_during_travel_l1443_144394


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_l1443_144338

/-- A rectangle inscribed in a semicircle -/
structure InscribedRectangle where
  /-- The length of side PR of the rectangle -/
  pr : ℝ
  /-- The length of PG and SH, which are equal -/
  pg : ℝ
  /-- Assumption that PR is positive -/
  pr_pos : pr > 0
  /-- Assumption that PG is positive -/
  pg_pos : pg > 0

/-- The theorem stating that the area of the inscribed rectangle is 160√6 -/
theorem inscribed_rectangle_area (rect : InscribedRectangle) 
  (h1 : rect.pr = 20) (h2 : rect.pg = 12) : 
  ∃ (area : ℝ), area = rect.pr * Real.sqrt (rect.pg * (rect.pr + 2 * rect.pg - rect.pg)) ∧ 
  area = 160 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_l1443_144338


namespace NUMINAMATH_CALUDE_base7_25_to_binary_l1443_144326

/-- Converts a number from base 7 to base 10 -/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 2 -/
def decimalToBinary (n : ℕ) : List ℕ := sorry

theorem base7_25_to_binary :
  decimalToBinary (base7ToDecimal 25) = [1, 0, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_base7_25_to_binary_l1443_144326


namespace NUMINAMATH_CALUDE_divisors_of_60_and_84_l1443_144379

theorem divisors_of_60_and_84 : ∃ (n : ℕ), n > 0 ∧ 
  (∀ d : ℕ, d > 0 ∧ (60 % d = 0 ∧ 84 % d = 0) ↔ d ∈ Finset.range n) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_60_and_84_l1443_144379


namespace NUMINAMATH_CALUDE_distribute_teachers_count_l1443_144395

/-- The number of ways to distribute teachers to schools -/
def distribute_teachers : ℕ :=
  let chinese_teachers := 2
  let math_teachers := 4
  let total_teachers := chinese_teachers + math_teachers
  let schools := 2
  let teachers_per_school := 3
  let ways_to_choose_math := Nat.choose math_teachers (teachers_per_school - 1)
  ways_to_choose_math * schools

/-- Theorem stating that the number of ways to distribute teachers is 12 -/
theorem distribute_teachers_count : distribute_teachers = 12 := by
  sorry

end NUMINAMATH_CALUDE_distribute_teachers_count_l1443_144395


namespace NUMINAMATH_CALUDE_ring_ratio_l1443_144362

def ring_problem (first_ring_cost second_ring_cost selling_price out_of_pocket : ℚ) : Prop :=
  first_ring_cost = 10000 ∧
  second_ring_cost = 2 * first_ring_cost ∧
  first_ring_cost + second_ring_cost - selling_price = out_of_pocket ∧
  out_of_pocket = 25000

theorem ring_ratio (first_ring_cost second_ring_cost selling_price out_of_pocket : ℚ) 
  (h : ring_problem first_ring_cost second_ring_cost selling_price out_of_pocket) :
  selling_price / first_ring_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ring_ratio_l1443_144362


namespace NUMINAMATH_CALUDE_hasan_plates_removal_l1443_144344

/-- The weight of each plate in ounces -/
def plate_weight : ℕ := 10

/-- The weight limit for each box in pounds -/
def box_weight_limit : ℕ := 20

/-- The number of plates initially packed in the box -/
def initial_plates : ℕ := 38

/-- The number of ounces in a pound -/
def ounces_per_pound : ℕ := 16

/-- The number of plates Hasan needs to remove from the box -/
def plates_to_remove : ℕ := 6

theorem hasan_plates_removal :
  plates_to_remove = 
    (initial_plates * plate_weight - box_weight_limit * ounces_per_pound) / plate_weight :=
by sorry

end NUMINAMATH_CALUDE_hasan_plates_removal_l1443_144344


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l1443_144373

-- Define the geometric sequence
def geometric_sequence (n : ℕ) : ℚ :=
  (-1/2) ^ (n - 1)

-- Define the sum of the first n terms
def geometric_sum (n : ℕ) : ℚ :=
  (2/3) * (1 - (-1/2)^n)

-- Theorem statement
theorem geometric_sequence_properties :
  (geometric_sequence 3 = 1/4) ∧
  (∀ n : ℕ, geometric_sequence n = (-1/2)^(n-1)) ∧
  (∀ n : ℕ, geometric_sum n = (2/3) * (1 - (-1/2)^n)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l1443_144373


namespace NUMINAMATH_CALUDE_convex_ngon_diagonal_intersections_l1443_144358

/-- The number of intersection points of diagonals in a convex n-gon -/
def diagonalIntersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Theorem: The number of intersection points for diagonals of a convex n-gon,
    where no three diagonals intersect at a single point, is equal to n(n-1)(n-2)(n-3)/24 -/
theorem convex_ngon_diagonal_intersections (n : ℕ) (h1 : n ≥ 4) :
  diagonalIntersections n = (n.choose 4) := by
  sorry

end NUMINAMATH_CALUDE_convex_ngon_diagonal_intersections_l1443_144358


namespace NUMINAMATH_CALUDE_product_of_roots_undefined_expression_l1443_144350

theorem product_of_roots_undefined_expression : ∃ (x y : ℝ),
  x^2 + 4*x - 5 = 0 ∧ 
  y^2 + 4*y - 5 = 0 ∧ 
  x * y = -5 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_undefined_expression_l1443_144350


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l1443_144303

theorem percentage_of_percentage (total : ℝ) (percentage1 : ℝ) (amount : ℝ) (percentage2 : ℝ) :
  total = 500 →
  percentage1 = 50 →
  amount = 25 →
  percentage2 = 10 →
  (amount / (percentage1 / 100 * total)) * 100 = percentage2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l1443_144303


namespace NUMINAMATH_CALUDE_car_uphill_speed_l1443_144388

/-- Given a car's travel information, prove that its uphill speed is 80 km/hour. -/
theorem car_uphill_speed
  (downhill_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (downhill_time : ℝ)
  (uphill_time : ℝ)
  (h1 : downhill_speed = 50)
  (h2 : total_time = 15)
  (h3 : total_distance = 650)
  (h4 : downhill_time = 5)
  (h5 : uphill_time = 5)
  : ∃ (uphill_speed : ℝ), uphill_speed = 80 := by
  sorry

end NUMINAMATH_CALUDE_car_uphill_speed_l1443_144388


namespace NUMINAMATH_CALUDE_q_join_time_l1443_144346

/-- Represents the number of months after which Q joined the business --/
def x : ℕ := sorry

/-- P's initial investment --/
def p_investment : ℕ := 4000

/-- Q's investment --/
def q_investment : ℕ := 9000

/-- Total number of months in a year --/
def total_months : ℕ := 12

/-- Ratio of P's profit share to Q's profit share --/
def profit_ratio : ℚ := 2 / 3

theorem q_join_time :
  (p_investment * total_months) / (q_investment * (total_months - x)) = profit_ratio →
  x = 4 := by sorry

end NUMINAMATH_CALUDE_q_join_time_l1443_144346


namespace NUMINAMATH_CALUDE_smallest_divisor_of_720_two_divides_720_smallest_positive_divisor_of_720_is_two_l1443_144367

theorem smallest_divisor_of_720 : 
  ∀ n : ℕ, n > 0 → n ∣ 720 → n ≥ 2 :=
by
  sorry

theorem two_divides_720 : 2 ∣ 720 :=
by
  sorry

theorem smallest_positive_divisor_of_720_is_two : 
  ∃ (d : ℕ), d > 0 ∧ d ∣ 720 ∧ ∀ n : ℕ, n > 0 → n ∣ 720 → n ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_720_two_divides_720_smallest_positive_divisor_of_720_is_two_l1443_144367


namespace NUMINAMATH_CALUDE_tan_alpha_3_expression_equals_2_l1443_144390

theorem tan_alpha_3_expression_equals_2 (α : Real) (h : Real.tan α = 3) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_3_expression_equals_2_l1443_144390


namespace NUMINAMATH_CALUDE_existence_of_special_n_l1443_144320

theorem existence_of_special_n (t : ℕ) : ∃ n : ℕ, n > 1 ∧ 
  (Nat.gcd n t = 1) ∧ 
  (∀ k x m : ℕ, k ≥ 1 → m > 1 → n^k + t ≠ x^m) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_n_l1443_144320


namespace NUMINAMATH_CALUDE_quiz_competition_participants_l1443_144307

theorem quiz_competition_participants :
  let initial_participants : ℕ := 300
  let first_round_ratio : ℚ := 2/5
  let second_round_ratio : ℚ := 1/4
  let final_participants : ℕ := 30
  (initial_participants : ℚ) * first_round_ratio * second_round_ratio = final_participants := by
sorry

end NUMINAMATH_CALUDE_quiz_competition_participants_l1443_144307


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1443_144384

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perpendicular bisector of a line segment -/
def isPerpBisector (c : ℝ) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  (midpoint.x + midpoint.y = c) ∧ 
  (c - p1.x - p1.y = p2.x + p2.y - c)

/-- The theorem statement -/
theorem perpendicular_bisector_c_value :
  ∀ c : ℝ, isPerpBisector c ⟨2, 5⟩ ⟨8, 11⟩ → c = 13 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l1443_144384


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1443_144399

/-- The volume of a sphere inscribed in a cube with edge length 8 feet -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 8
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = (256 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1443_144399


namespace NUMINAMATH_CALUDE_second_month_sale_l1443_144340

/-- Represents the sales data for a grocery shop over 6 months -/
structure GrocerySales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Calculates the average sale over 6 months -/
def average_sale (sales : GrocerySales) : ℚ :=
  (sales.month1 + sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6) / 6

/-- Theorem stating the conditions and the result to be proved -/
theorem second_month_sale 
  (sales : GrocerySales)
  (h1 : sales.month1 = 6435)
  (h2 : sales.month3 = 7230)
  (h3 : sales.month4 = 6562)
  (h4 : sales.month6 = 4991)
  (h5 : average_sale sales = 6500) :
  sales.month2 = 13782 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l1443_144340


namespace NUMINAMATH_CALUDE_girls_in_school_l1443_144342

theorem girls_in_school (total_students sample_size girls_boys_diff : ℕ) 
  (h1 : total_students = 1600)
  (h2 : sample_size = 200)
  (h3 : girls_boys_diff = 20) : 
  ∃ (girls : ℕ), girls = 860 ∧ 
  girls + (total_students - girls) = total_students ∧
  (girls : ℚ) / total_students * sample_size = 
    (total_students - girls : ℚ) / total_students * sample_size - girls_boys_diff := by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_l1443_144342


namespace NUMINAMATH_CALUDE_sallys_purchase_l1443_144311

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars5 : ℕ
  dollars10 : ℕ

/-- The problem statement -/
theorem sallys_purchase (counts : ItemCounts) : 
  counts.cents50 + counts.dollars5 + counts.dollars10 = 30 →
  50 * counts.cents50 + 500 * counts.dollars5 + 1000 * counts.dollars10 = 10000 →
  counts.cents50 = 20 := by
  sorry


end NUMINAMATH_CALUDE_sallys_purchase_l1443_144311


namespace NUMINAMATH_CALUDE_prob_both_heads_is_one_fourth_l1443_144301

/-- A coin of uniform density -/
structure Coin :=
  (side : Bool)

/-- The sample space of tossing two coins -/
def TwoCoins := Coin × Coin

/-- The event where both coins land heads up -/
def BothHeads (outcome : TwoCoins) : Prop :=
  outcome.1.side ∧ outcome.2.side

/-- The probability measure on the sample space -/
axiom prob : Set TwoCoins → ℝ

/-- The probability measure satisfies basic properties -/
axiom prob_nonneg : ∀ A : Set TwoCoins, 0 ≤ prob A
axiom prob_le_one : ∀ A : Set TwoCoins, prob A ≤ 1
axiom prob_additive : ∀ A B : Set TwoCoins, A ∩ B = ∅ → prob (A ∪ B) = prob A + prob B

/-- The probability of each outcome is equal due to uniform density -/
axiom prob_uniform : ∀ x y : TwoCoins, prob {x} = prob {y}

theorem prob_both_heads_is_one_fourth :
  prob {x : TwoCoins | BothHeads x} = 1/4 := by
  sorry

#check prob_both_heads_is_one_fourth

end NUMINAMATH_CALUDE_prob_both_heads_is_one_fourth_l1443_144301


namespace NUMINAMATH_CALUDE_complementary_event_l1443_144371

-- Define the sample space
def SampleSpace := List Bool

-- Define the event of missing both times
def MissBoth (outcome : SampleSpace) : Prop := outcome = [false, false]

-- Define the event of at least one hit
def AtLeastOneHit (outcome : SampleSpace) : Prop := outcome ≠ [false, false]

-- Theorem statement
theorem complementary_event : 
  ∀ (outcome : SampleSpace), MissBoth outcome ↔ ¬(AtLeastOneHit outcome) :=
sorry

end NUMINAMATH_CALUDE_complementary_event_l1443_144371


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l1443_144380

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 20) :
  Complex.abs (w^3 + z^3) = 56 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l1443_144380


namespace NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1443_144393

theorem greatest_integer_quadratic_inequality :
  ∃ (n : ℤ), n^2 - 13*n + 40 ≤ 0 ∧ n = 8 ∧ ∀ (m : ℤ), m^2 - 13*m + 40 ≤ 0 → m ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_quadratic_inequality_l1443_144393


namespace NUMINAMATH_CALUDE_income_comparison_l1443_144309

theorem income_comparison (juan tim other : ℝ) 
  (h1 : tim = juan * (1 - 0.5))
  (h2 : other = juan * 0.8) :
  other = tim * 1.6 := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l1443_144309


namespace NUMINAMATH_CALUDE_max_identifiable_bulbs_max_identifiable_bulbs_and_switches_l1443_144310

/-- Represents the state of a bulb -/
inductive BulbState
  | On
  | OffWarm
  | OffCold

/-- Represents a trip to the basement -/
def Trip := Nat → BulbState

/-- The number of trips allowed to the basement -/
def numTrips : Nat := 2

/-- The number of possible states for each bulb in a single trip -/
def statesPerTrip : Nat := 3

/-- Theorem: The maximum number of unique bulb configurations identifiable in two trips -/
theorem max_identifiable_bulbs :
  (statesPerTrip ^ numTrips : Nat) = 9 := by
  sorry

/-- Corollary: The maximum number of bulbs and switches that can be identified with each other in two trips -/
theorem max_identifiable_bulbs_and_switches :
  ∃ (n : Nat), n = 9 ∧ n = (statesPerTrip ^ numTrips : Nat) := by
  sorry

end NUMINAMATH_CALUDE_max_identifiable_bulbs_max_identifiable_bulbs_and_switches_l1443_144310


namespace NUMINAMATH_CALUDE_tank_capacity_is_2000_liters_l1443_144315

-- Define the flow rates and time
def inflow_rate : ℚ := 1 / 2 -- kiloliters per minute
def outflow_rate1 : ℚ := 1 / 4 -- kiloliters per minute
def outflow_rate2 : ℚ := 1 / 6 -- kiloliters per minute
def fill_time : ℚ := 12 -- minutes

-- Define the net flow rate
def net_flow_rate : ℚ := inflow_rate - outflow_rate1 - outflow_rate2

-- Define the theorem
theorem tank_capacity_is_2000_liters :
  let volume_added : ℚ := net_flow_rate * fill_time
  let full_capacity_kl : ℚ := 2 * volume_added
  let full_capacity_l : ℚ := 1000 * full_capacity_kl
  full_capacity_l = 2000 := by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_2000_liters_l1443_144315


namespace NUMINAMATH_CALUDE_area_of_triangle_PQR_l1443_144347

/-- Given two lines intersecting at P(2,5) with slopes 3 and 1 respectively,
    and Q and R as the intersections of these lines with the x-axis,
    prove that the area of triangle PQR is 25/3 -/
theorem area_of_triangle_PQR (P Q R : ℝ × ℝ) : 
  P = (2, 5) →
  (∃ m₁ m₂ : ℝ, m₁ = 3 ∧ m₂ = 1 ∧ 
    (∀ x y : ℝ, y - 5 = m₁ * (x - 2) ∨ y - 5 = m₂ * (x - 2))) →
  Q.2 = 0 ∧ R.2 = 0 →
  (∃ x₁ x₂ : ℝ, Q = (x₁, 0) ∧ R = (x₂, 0) ∧ 
    (5 - 0) = 3 * (2 - x₁) ∧ (5 - 0) = 1 * (2 - x₂)) →
  (1/2 : ℝ) * |Q.1 - R.1| * 5 = 25/3 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_PQR_l1443_144347


namespace NUMINAMATH_CALUDE_amanda_pay_l1443_144368

def hourly_rate : ℝ := 50
def hours_worked : ℝ := 10
def withholding_percentage : ℝ := 0.20

def daily_pay : ℝ := hourly_rate * hours_worked
def withheld_amount : ℝ := daily_pay * withholding_percentage
def final_pay : ℝ := daily_pay - withheld_amount

theorem amanda_pay : final_pay = 400 := by
  sorry

end NUMINAMATH_CALUDE_amanda_pay_l1443_144368


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1443_144396

theorem inequality_solution_set (t : ℝ) (a : ℝ) : 
  (∀ x : ℝ, (tx^2 - 6*x + t^2 < 0) ↔ (x < a ∨ x > 1)) → a = -3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1443_144396


namespace NUMINAMATH_CALUDE_annes_cleaning_time_l1443_144355

/-- Represents the time it takes Anne to clean the house individually -/
def annes_individual_time (bruce_rate anne_rate : ℚ) : ℚ :=
  1 / anne_rate

/-- The condition that Bruce and Anne can clean the house in 4 hours together -/
def condition1 (bruce_rate anne_rate : ℚ) : Prop :=
  bruce_rate + anne_rate = 1 / 4

/-- The condition that Bruce and Anne with Anne's doubled speed can clean the house in 3 hours -/
def condition2 (bruce_rate anne_rate : ℚ) : Prop :=
  bruce_rate + 2 * anne_rate = 1 / 3

theorem annes_cleaning_time 
  (bruce_rate anne_rate : ℚ) 
  (h1 : condition1 bruce_rate anne_rate) 
  (h2 : condition2 bruce_rate anne_rate) :
  annes_individual_time bruce_rate anne_rate = 12 := by
sorry

end NUMINAMATH_CALUDE_annes_cleaning_time_l1443_144355


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_7047_l1443_144376

theorem smallest_prime_factor_of_7047 : Nat.minFac 7047 = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_7047_l1443_144376


namespace NUMINAMATH_CALUDE_oil_bottles_total_volume_l1443_144313

theorem oil_bottles_total_volume (total_bottles : ℕ) (small_bottles : ℕ) 
  (small_volume : ℚ) (large_volume : ℚ) :
  total_bottles = 35 →
  small_bottles = 17 →
  small_volume = 250 / 1000 →
  large_volume = 300 / 1000 →
  (small_bottles * small_volume + (total_bottles - small_bottles) * large_volume) = 9.65 := by
sorry

end NUMINAMATH_CALUDE_oil_bottles_total_volume_l1443_144313


namespace NUMINAMATH_CALUDE_estimate_battery_usage_l1443_144392

/-- Estimates the total number of batteries used by a class based on a sample. -/
theorem estimate_battery_usage
  (sample_size : ℕ)
  (sample_total : ℕ)
  (class_size : ℕ)
  (h1 : sample_size = 6)
  (h2 : sample_total = 168)
  (h3 : class_size = 45) :
  (sample_total / sample_size) * class_size = 1260 :=
by sorry

end NUMINAMATH_CALUDE_estimate_battery_usage_l1443_144392


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1443_144366

theorem sum_reciprocal_inequality (a b c : ℝ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (h_sum_squares : a^2 + b^2 + c^2 = 12) : 
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_l1443_144366


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_odd_l1443_144377

theorem sum_of_multiples_is_odd (c d : ℤ) 
  (hc : ∃ m : ℤ, c = 6 * m) 
  (hd : ∃ n : ℤ, d = 9 * n) : 
  Odd (c + d) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_odd_l1443_144377


namespace NUMINAMATH_CALUDE_min_bailing_rate_is_14_l1443_144317

/-- Represents the scenario of Amy and Boris in the leaking boat --/
structure BoatScenario where
  distance_to_shore : Real
  water_intake_rate : Real
  sinking_threshold : Real
  initial_speed : Real
  speed_increase : Real
  speed_increase_interval : Real

/-- Calculates the time taken to reach the shore --/
def time_to_shore (scenario : BoatScenario) : Real :=
  sorry

/-- Calculates the total potential water intake --/
def total_water_intake (scenario : BoatScenario) (time : Real) : Real :=
  scenario.water_intake_rate * time

/-- Calculates the minimum bailing rate required --/
def min_bailing_rate (scenario : BoatScenario) : Real :=
  sorry

/-- The main theorem stating the minimum bailing rate for the given scenario --/
theorem min_bailing_rate_is_14 (scenario : BoatScenario) 
  (h1 : scenario.distance_to_shore = 2)
  (h2 : scenario.water_intake_rate = 15)
  (h3 : scenario.sinking_threshold = 50)
  (h4 : scenario.initial_speed = 2)
  (h5 : scenario.speed_increase = 1)
  (h6 : scenario.speed_increase_interval = 0.5) :
  min_bailing_rate scenario = 14 := by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_is_14_l1443_144317


namespace NUMINAMATH_CALUDE_min_value_sum_of_powers_l1443_144391

theorem min_value_sum_of_powers (a b x y : ℝ) (n : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) (hxy : x + y = 1) :
  a / x^n + b / y^n ≥ (a^(1/(n+1:ℝ)) + b^(1/(n+1:ℝ)))^(n+1) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_powers_l1443_144391


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1443_144353

/-- An arithmetic sequence with first term 2 and the sum of the second and fourth terms equal to the sixth term has a common difference of 2. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 2)  -- First term is 2
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)  -- Definition of arithmetic sequence
  (h3 : a 2 + a 4 = a 6)  -- Sum of second and fourth terms equals sixth term
  : d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1443_144353


namespace NUMINAMATH_CALUDE_shaded_area_in_squares_l1443_144378

/-- The area of the shaded region in a specific geometric configuration -/
theorem shaded_area_in_squares : 
  let small_square_side : ℝ := 4
  let large_square_side : ℝ := 12
  let rectangle_width : ℝ := 2
  let rectangle_height : ℝ := 4
  let total_width : ℝ := small_square_side + large_square_side
  let triangle_height : ℝ := (small_square_side * small_square_side) / total_width
  let triangle_area : ℝ := (1 / 2) * triangle_height * small_square_side
  let small_square_area : ℝ := small_square_side * small_square_side
  let shaded_area : ℝ := small_square_area - triangle_area
  shaded_area = 14 := by
    sorry

end NUMINAMATH_CALUDE_shaded_area_in_squares_l1443_144378


namespace NUMINAMATH_CALUDE_domino_distribution_l1443_144336

theorem domino_distribution (total_dominoes : ℕ) (num_players : ℕ) 
  (h1 : total_dominoes = 28) (h2 : num_players = 4) :
  total_dominoes / num_players = 7 := by
  sorry

end NUMINAMATH_CALUDE_domino_distribution_l1443_144336


namespace NUMINAMATH_CALUDE_boys_playing_both_sports_l1443_144318

theorem boys_playing_both_sports (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) :
  total = 22 →
  basketball = 13 →
  football = 15 →
  neither = 3 →
  ∃ (both : ℕ), both = 9 ∧ total = basketball + football - both + neither :=
by sorry

end NUMINAMATH_CALUDE_boys_playing_both_sports_l1443_144318


namespace NUMINAMATH_CALUDE_waitress_tips_fraction_l1443_144339

theorem waitress_tips_fraction (salary : ℝ) (tips : ℝ) (h : tips = 2/4 * salary) :
  tips / (salary + tips) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_waitress_tips_fraction_l1443_144339


namespace NUMINAMATH_CALUDE_probability_of_three_hits_is_one_fifth_l1443_144329

/-- A set of random numbers -/
structure RandomSet :=
  (numbers : List Nat)

/-- Predicate to check if a number is a hit (1 to 6) -/
def isHit (n : Nat) : Bool :=
  1 ≤ n ∧ n ≤ 6

/-- Count the number of hits in a set -/
def countHits (s : RandomSet) : Nat :=
  s.numbers.filter isHit |>.length

/-- The experiment data -/
def experimentData : List RandomSet := sorry

/-- The number of sets with exactly three hits -/
def setsWithThreeHits : Nat :=
  experimentData.filter (fun s => countHits s = 3) |>.length

/-- Total number of sets in the experiment -/
def totalSets : Nat := 20

theorem probability_of_three_hits_is_one_fifth :
  (setsWithThreeHits : ℚ) / totalSets = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_three_hits_is_one_fifth_l1443_144329


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1443_144300

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1443_144300


namespace NUMINAMATH_CALUDE_sum_of_roots_l1443_144314

theorem sum_of_roots (y₁ y₂ k m : ℝ) (h1 : y₁ ≠ y₂) 
  (h2 : 5 * y₁^2 - k * y₁ = m) (h3 : 5 * y₂^2 - k * y₂ = m) : 
  y₁ + y₂ = k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1443_144314


namespace NUMINAMATH_CALUDE_line_invariant_under_transformation_l1443_144389

def transformation (a b : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (-x + a*y, b*x + 3*y)

theorem line_invariant_under_transformation (a b : ℝ) :
  (∀ x y : ℝ, 2*x - y = 3 → 
    let (x', y') := transformation a b x y
    2*x' - y' = 3) →
  a = 1 ∧ b = -4 := by
sorry

end NUMINAMATH_CALUDE_line_invariant_under_transformation_l1443_144389


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l1443_144375

/-- A rectangle with length 10 and width 6 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_eq : length = 10)
  (width_eq : width = 6)

/-- A circle passing through two vertices of the rectangle and tangent to the opposite side -/
structure CircleTangentToRectangle (rect : Rectangle) :=
  (radius : ℝ)
  (passes_through_vertices : Bool)
  (tangent_to_opposite_side : Bool)

/-- The theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five (rect : Rectangle) (circle : CircleTangentToRectangle rect) :
  circle.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l1443_144375


namespace NUMINAMATH_CALUDE_right_triangle_three_four_five_l1443_144341

theorem right_triangle_three_four_five :
  ∀ (a b c : ℝ),
    a = 3 ∧ b = 4 ∧ c = 5 →
    a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_three_four_five_l1443_144341


namespace NUMINAMATH_CALUDE_sequence_termination_l1443_144328

def b : ℕ → ℚ
  | 0 => 41
  | 1 => 68
  | (k+2) => b k - 5 / b (k+1)

theorem sequence_termination :
  ∃ n : ℕ, n > 0 ∧ b n = 0 ∧ ∀ k < n, b k ≠ 0 ∧ b (k+1) = b (k-1) - 5 / b k :=
by
  use 559
  sorry

#eval b 559

end NUMINAMATH_CALUDE_sequence_termination_l1443_144328


namespace NUMINAMATH_CALUDE_sequence_general_term_l1443_144385

theorem sequence_general_term (a : ℕ → ℕ) (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + n) :
  ∀ n, a n = 2 * n :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1443_144385


namespace NUMINAMATH_CALUDE_no_valid_day_for_statements_l1443_144398

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents whether a statement is true or false on a given day -/
def Statement := Day → Prop

/-- The statement "I lied yesterday" -/
def LiedYesterday : Statement := fun d => 
  match d with
  | Day.Monday => false     -- Sunday's statement
  | Day.Tuesday => false    -- Monday's statement
  | Day.Wednesday => false  -- Tuesday's statement
  | Day.Thursday => false   -- Wednesday's statement
  | Day.Friday => false     -- Thursday's statement
  | Day.Saturday => false   -- Friday's statement
  | Day.Sunday => false     -- Saturday's statement

/-- The statement "I will lie tomorrow" -/
def WillLieTomorrow : Statement := fun d =>
  match d with
  | Day.Monday => false     -- Tuesday's statement
  | Day.Tuesday => false    -- Wednesday's statement
  | Day.Wednesday => false  -- Thursday's statement
  | Day.Thursday => false   -- Friday's statement
  | Day.Friday => false     -- Saturday's statement
  | Day.Saturday => false   -- Sunday's statement
  | Day.Sunday => false     -- Monday's statement

/-- Theorem stating that there is no day where both statements can be made without contradiction -/
theorem no_valid_day_for_statements : ¬∃ (d : Day), LiedYesterday d ∧ WillLieTomorrow d := by
  sorry


end NUMINAMATH_CALUDE_no_valid_day_for_statements_l1443_144398


namespace NUMINAMATH_CALUDE_trig_identities_l1443_144345

theorem trig_identities :
  (Real.sin (15 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4) ∧
  (Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4) ∧
  (Real.sin (18 * π / 180) = (-1 + Real.sqrt 5) / 4) ∧
  (Real.cos (18 * π / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l1443_144345


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_143_l1443_144337

theorem first_nonzero_digit_of_1_over_143 : ∃ (n : ℕ) (d : ℕ), 
  (1 : ℚ) / 143 = (n : ℚ) / 10^d ∧ 
  n % 10 = 7 ∧ 
  ∀ (m : ℕ), m < d → (1 : ℚ) / 143 * 10^m < 1 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_1_over_143_l1443_144337


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1443_144334

noncomputable section

/-- Triangle ABC with given properties -/
structure TriangleABC where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  a_eq : a = Real.sqrt 5
  b_eq : b = 3
  sin_C_eq : Real.sin C = 2 * Real.sin A
  -- Triangle inequality and angle sum
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b
  angle_sum : A + B + C = π

theorem triangle_abc_properties (t : TriangleABC) : 
  t.c = 2 * Real.sqrt 5 ∧ 
  Real.sin (2 * t.A - π/4) = Real.sqrt 2 / 10 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_abc_properties_l1443_144334


namespace NUMINAMATH_CALUDE_square_of_complex_number_l1443_144356

theorem square_of_complex_number :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l1443_144356
