import Mathlib

namespace inequality_proof_l1062_106246

theorem inequality_proof (x : ℝ) : (1 : ℝ) / (x^2 + 1) > (1 : ℝ) / (x^2 + 2) := by
  sorry

end inequality_proof_l1062_106246


namespace quadratic_minimum_quadratic_minimum_attained_l1062_106235

theorem quadratic_minimum (x : ℝ) : 2 * x^2 + 16 * x + 40 ≥ 8 := by sorry

theorem quadratic_minimum_attained : ∃ x : ℝ, 2 * x^2 + 16 * x + 40 = 8 := by sorry

end quadratic_minimum_quadratic_minimum_attained_l1062_106235


namespace problem_solution_l1062_106203

theorem problem_solution (a b c : ℝ) 
  (h1 : (a + b + c)^2 = 3*(a^2 + b^2 + c^2)) 
  (h2 : a + b + c = 12) : 
  a = 4 := by
sorry

end problem_solution_l1062_106203


namespace max_value_of_f_l1062_106225

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x)

theorem max_value_of_f :
  ∃ (M : ℝ), M = (16 * Real.sqrt 3) / 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l1062_106225


namespace total_miles_driven_is_2225_l1062_106269

/-- A structure representing a car's weekly fuel consumption and mileage. -/
structure Car where
  gallons_consumed : ℝ
  average_mpg : ℝ

/-- Calculates the total miles driven by a car given its fuel consumption and average mpg. -/
def miles_driven (car : Car) : ℝ :=
  car.gallons_consumed * car.average_mpg

/-- Represents the family's two cars and their combined mileage. -/
structure FamilyCars where
  car1 : Car
  car2 : Car
  total_average_mpg : ℝ

/-- Theorem stating that under the given conditions, the total miles driven by both cars is 2225. -/
theorem total_miles_driven_is_2225 (family_cars : FamilyCars)
    (h1 : family_cars.car1.gallons_consumed = 25)
    (h2 : family_cars.car2.gallons_consumed = 35)
    (h3 : family_cars.car1.average_mpg = 40)
    (h4 : family_cars.total_average_mpg = 75) :
    miles_driven family_cars.car1 + miles_driven family_cars.car2 = 2225 := by
  sorry


end total_miles_driven_is_2225_l1062_106269


namespace horner_method_evaluation_l1062_106274

def horner_polynomial (x : ℝ) : ℝ := (((((3 * x - 4) * x + 6) * x - 2) * x - 5) * x - 2)

theorem horner_method_evaluation :
  horner_polynomial 5 = 7548 := by
  sorry

end horner_method_evaluation_l1062_106274


namespace two_year_growth_l1062_106281

/-- Calculates the final value after compound growth --/
def compound_growth (initial_value : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ years

/-- Theorem: After two years of 1/8 annual growth, 32000 becomes 40500 --/
theorem two_year_growth :
  compound_growth 32000 (1/8) 2 = 40500 := by
  sorry

end two_year_growth_l1062_106281


namespace problem_solution_l1062_106297

noncomputable def x : ℝ := Real.sqrt (19 - 8 * Real.sqrt 3)

theorem problem_solution : 
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 :=
by sorry

end problem_solution_l1062_106297


namespace function_value_at_two_l1062_106247

/-- Given a function f(x) = ax^5 + bx^3 - x + 2 where a and b are constants,
    and f(-2) = 5, prove that f(2) = -1 -/
theorem function_value_at_two
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^5 + b * x^3 - x + 2)
  (h2 : f (-2) = 5) :
  f 2 = -1 := by
  sorry

end function_value_at_two_l1062_106247


namespace cube_root_8000_l1062_106250

theorem cube_root_8000 (c d : ℕ+) : 
  (c : ℝ) * (d : ℝ)^(1/3 : ℝ) = 20 → 
  (∀ (c' d' : ℕ+), (c' : ℝ) * (d' : ℝ)^(1/3 : ℝ) = 20 → d ≤ d') → 
  c + d = 21 := by
  sorry

end cube_root_8000_l1062_106250


namespace revenue_maximizing_price_l1062_106283

/-- Revenue function for toy sales -/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The price that maximizes revenue is 18.75 -/
theorem revenue_maximizing_price :
  ∃ (p : ℝ), p ≤ 30 ∧ 
  (∀ (q : ℝ), q ≤ 30 → R q ≤ R p) ∧ 
  p = 18.75 := by
sorry

end revenue_maximizing_price_l1062_106283


namespace prohor_receives_all_money_l1062_106245

/-- Represents a person with their initial number of flatbreads -/
structure Person where
  name : String
  flatbreads : ℕ

/-- Represents the situation with the woodcutters and hunter -/
structure WoodcutterSituation where
  ivan : Person
  prohor : Person
  hunter : Person
  total_flatbreads : ℕ
  total_people : ℕ
  hunter_payment : ℕ

/-- Calculates the fair compensation for a person based on shared flatbreads -/
def fair_compensation (situation : WoodcutterSituation) (person : Person) : ℕ :=
  let shared_flatbreads := person.flatbreads - (situation.total_flatbreads / situation.total_people)
  shared_flatbreads * (situation.hunter_payment / situation.total_flatbreads)

/-- Theorem stating that Prohor should receive all the money -/
theorem prohor_receives_all_money (situation : WoodcutterSituation) : 
  situation.ivan.flatbreads = 4 →
  situation.prohor.flatbreads = 8 →
  situation.total_flatbreads = 12 →
  situation.total_people = 3 →
  situation.hunter_payment = 60 →
  fair_compensation situation situation.prohor = situation.hunter_payment :=
sorry

end prohor_receives_all_money_l1062_106245


namespace intersection_of_A_and_B_l1062_106234

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 3} := by
  sorry

end intersection_of_A_and_B_l1062_106234


namespace max_value_of_expression_l1062_106286

theorem max_value_of_expression (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → 
  2*x*y + y*z + 2*z*x ≤ 4/7 ∧ 
  ∃ x' y' z' : ℝ, x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 1 ∧ 2*x'*y' + y'*z' + 2*z'*x' = 4/7 :=
by sorry

end max_value_of_expression_l1062_106286


namespace system_solution_l1062_106299

theorem system_solution (x y z : ℝ) : 
  x = 1 ∧ y = -1 ∧ z = -2 →
  (2 * x + y + z = -1) ∧
  (3 * y - z = -1) ∧
  (3 * x + 2 * y + 3 * z = -5) := by
  sorry

end system_solution_l1062_106299


namespace value_of_a_l1062_106254

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a + 1}

-- State the theorem
theorem value_of_a : 
  ∀ a : ℝ, (A a ⊇ B a) → (a = -1 ∨ a = 2) := by
  sorry

end value_of_a_l1062_106254


namespace smallest_degree_poly_div_by_30_l1062_106209

/-- A polynomial with coefficients in {-1, 0, 1} -/
def RestrictedPoly (k : ℕ) := {f : Polynomial ℤ // ∀ i, i < k → f.coeff i ∈ ({-1, 0, 1} : Set ℤ)}

/-- A polynomial is divisible by 30 for all positive integers -/
def DivisibleBy30 (f : Polynomial ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → (30 : ℤ) ∣ f.eval n

/-- The theorem stating the smallest degree of a polynomial satisfying the conditions -/
theorem smallest_degree_poly_div_by_30 :
  ∃ (k : ℕ) (f : RestrictedPoly k),
    DivisibleBy30 f.val ∧
    (∀ (j : ℕ) (g : RestrictedPoly j), DivisibleBy30 g.val → k ≤ j) ∧
    k = 10 := by sorry

end smallest_degree_poly_div_by_30_l1062_106209


namespace smallest_delicious_integer_l1062_106267

/-- An integer is delicious if there exist several consecutive integers, starting from it, that add up to 2020. -/
def Delicious (n : ℤ) : Prop :=
  ∃ k : ℕ+, (Finset.range k).sum (fun i => n + i) = 2020

/-- The smallest delicious integer less than -2020 is -2021. -/
theorem smallest_delicious_integer :
  (∀ n < -2020, Delicious n → n ≥ -2021) ∧ Delicious (-2021) :=
sorry

end smallest_delicious_integer_l1062_106267


namespace conference_session_duration_l1062_106275

/-- Given a conference duration and break time, calculate the session time in minutes. -/
def conference_session_time (hours minutes break_time : ℕ) : ℕ :=
  hours * 60 + minutes - break_time

/-- Theorem: A conference lasting 8 hours and 45 minutes with a 30-minute break has a session time of 495 minutes. -/
theorem conference_session_duration :
  conference_session_time 8 45 30 = 495 :=
by sorry

end conference_session_duration_l1062_106275


namespace ratio_problem_l1062_106277

theorem ratio_problem (x y z w : ℝ) 
  (h1 : 0.1 * x = 0.2 * y) 
  (h2 : 0.3 * y = 0.4 * z) 
  (h3 : 0.5 * z = 0.6 * w) : 
  ∃ (k : ℝ), k > 0 ∧ x = 8 * k ∧ y = 4 * k ∧ z = 3 * k ∧ w = 2.5 * k :=
sorry

end ratio_problem_l1062_106277


namespace actual_spent_correct_l1062_106287

/-- Represents a project budget with monthly allocations -/
structure ProjectBudget where
  total : ℕ
  months : ℕ
  monthly_allocation : ℕ
  h_allocation : monthly_allocation * months = total

/-- Calculates the actual amount spent given a project budget and over-budget amount -/
def actual_spent (budget : ProjectBudget) (over_budget : ℕ) (months_elapsed : ℕ) : ℕ :=
  budget.monthly_allocation * months_elapsed + over_budget

/-- Proves that the actual amount spent is correct given the project conditions -/
theorem actual_spent_correct (budget : ProjectBudget) 
    (h_total : budget.total = 12600)
    (h_months : budget.months = 12)
    (h_over_budget : over_budget = 280)
    (h_months_elapsed : months_elapsed = 6) :
    actual_spent budget over_budget months_elapsed = 6580 := by
  sorry

#eval actual_spent ⟨12600, 12, 1050, rfl⟩ 280 6

end actual_spent_correct_l1062_106287


namespace max_value_of_f_l1062_106276

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

theorem max_value_of_f (k : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4) →
  k = -3 ∨ k = 3/8 := by sorry

end max_value_of_f_l1062_106276


namespace min_value_product_l1062_106237

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x/y + y/z + z/x + y/x + z/y + x/z = 8) :
  (x/y + y/z + z/x) * (y/x + z/y + x/z) ≥ 22 * Real.sqrt 11 - 57 :=
by sorry

end min_value_product_l1062_106237


namespace medical_team_selection_l1062_106265

theorem medical_team_selection (orthopedic neurosurgeons internists : ℕ) 
  (h1 : orthopedic = 3) 
  (h2 : neurosurgeons = 4) 
  (h3 : internists = 5) 
  (team_size : ℕ) 
  (h4 : team_size = 5) : 
  (Nat.choose (orthopedic + neurosurgeons + internists) team_size) -
  ((Nat.choose (neurosurgeons + internists) team_size - 1) +
   (Nat.choose (orthopedic + internists) team_size - 1) +
   (Nat.choose (orthopedic + neurosurgeons) team_size) +
   1) = 590 := by
  sorry

end medical_team_selection_l1062_106265


namespace equation_solution_l1062_106218

theorem equation_solution : 
  ∃! x : ℝ, x ≠ (1/2) ∧ (5*x + 1) / (2*x^2 + 5*x - 3) = 2*x / (2*x - 1) ∧ x = -1 := by
  sorry

end equation_solution_l1062_106218


namespace weight_sum_l1062_106229

theorem weight_sum (m n o p : ℕ) 
  (h1 : m + n = 320)
  (h2 : n + o = 295)
  (h3 : o + p = 310) :
  m + p = 335 := by
  sorry

end weight_sum_l1062_106229


namespace rook_game_theorem_l1062_106284

/-- Represents the result of the game -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents a chessboard of size K × N with a rook in the upper right corner -/
structure ChessBoard where
  K : Nat
  N : Nat

/-- Determines the winner of the rook game based on the chessboard dimensions -/
def rook_game_winner (board : ChessBoard) : GameResult :=
  if board.K = board.N then
    GameResult.SecondPlayerWins
  else
    GameResult.FirstPlayerWins

/-- Theorem stating the winning condition for the rook game -/
theorem rook_game_theorem (board : ChessBoard) :
  rook_game_winner board =
    if board.K = board.N then
      GameResult.SecondPlayerWins
    else
      GameResult.FirstPlayerWins := by
  sorry

end rook_game_theorem_l1062_106284


namespace expand_expression_l1062_106261

theorem expand_expression (x y : ℝ) : (x + 15) * (3 * y + 20) = 3 * x * y + 20 * x + 45 * y + 300 := by
  sorry

end expand_expression_l1062_106261


namespace parents_age_when_mark_born_l1062_106243

/-- Given the ages of Mark and John, and their relation to their parents' age, 
    proves the age of the parents when Mark was born. -/
theorem parents_age_when_mark_born (mark_age john_age parents_age : ℕ) : 
  mark_age = 18 →
  john_age = mark_age - 10 →
  parents_age = 5 * john_age →
  parents_age - mark_age = 22 :=
by sorry

end parents_age_when_mark_born_l1062_106243


namespace x_minus_q_upper_bound_l1062_106212

theorem x_minus_q_upper_bound (x q : ℝ) (h1 : |x - 3| > q) (h2 : x < 3) :
  x - q < 3 - 2*q := by
sorry

end x_minus_q_upper_bound_l1062_106212


namespace pure_imaginary_condition_l1062_106270

theorem pure_imaginary_condition (a : ℝ) : 
  (∃ (b : ℝ), a^2 - 1 + (a + 1) * Complex.I = Complex.I * b) → a = 1 := by
  sorry

end pure_imaginary_condition_l1062_106270


namespace rectangle_area_l1062_106226

/-- Given a rectangle where the length is 15% more than the breadth and the breadth is 20 meters,
    prove that its area is 460 square meters. -/
theorem rectangle_area (b l a : ℝ) : 
  b = 20 →                  -- The breadth is 20 meters
  l = b * 1.15 →            -- The length is 15% more than the breadth
  a = l * b →               -- Area formula
  a = 460 := by sorry       -- The area is 460 square meters

end rectangle_area_l1062_106226


namespace sparrow_count_l1062_106271

theorem sparrow_count (bluebird_count : ℕ) (ratio_bluebird : ℕ) (ratio_sparrow : ℕ) 
  (h1 : bluebird_count = 28)
  (h2 : ratio_bluebird = 4)
  (h3 : ratio_sparrow = 5) :
  (bluebird_count * ratio_sparrow) / ratio_bluebird = 35 :=
by sorry

end sparrow_count_l1062_106271


namespace complex_modulus_equality_l1062_106279

theorem complex_modulus_equality (m : ℝ) (h : m > 0) :
  Complex.abs (4 + m * Complex.I) = 4 * Real.sqrt 13 → m = 8 * Real.sqrt 3 := by
sorry

end complex_modulus_equality_l1062_106279


namespace problem_statement_l1062_106253

theorem problem_statement (x : ℝ) (h : x = 4) : 5 * x + 7 = 27 := by
  sorry

end problem_statement_l1062_106253


namespace quadratic_equality_implies_coefficient_l1062_106268

theorem quadratic_equality_implies_coefficient (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + 9 = (x - 3)^2) → a = -6 := by
  sorry

end quadratic_equality_implies_coefficient_l1062_106268


namespace calculation_proof_l1062_106216

theorem calculation_proof : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculation_proof_l1062_106216


namespace arithmetic_geometric_sequence_l1062_106201

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- The main theorem -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a)
  (h_geom : geometric_seq (a 1) (a 3) (a 7)) :
  a 5 = 6 := by
sorry

end arithmetic_geometric_sequence_l1062_106201


namespace gcd_of_seven_digit_set_l1062_106291

/-- A function that generates a seven-digit number from a three-digit number -/
def seven_digit_from_three (n : ℕ) : ℕ := 1001 * n

/-- The set of all seven-digit numbers formed by repeating three-digit numbers -/
def seven_digit_set : Set ℕ := {m | ∃ n, 100 ≤ n ∧ n < 1000 ∧ m = seven_digit_from_three n}

/-- The theorem stating that 1001 is the greatest common divisor of all numbers in the set -/
theorem gcd_of_seven_digit_set :
  ∃ d, d > 0 ∧ (∀ m ∈ seven_digit_set, d ∣ m) ∧
  (∀ d' > 0, (∀ m ∈ seven_digit_set, d' ∣ m) → d' ≤ d) ∧
  d = 1001 := by
  sorry

end gcd_of_seven_digit_set_l1062_106291


namespace odd_function_domain_l1062_106238

-- Define the function f
def f (a : ℝ) : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_domain (a : ℝ) : 
  (∀ x, f a x ≠ 0 → x ∈ Set.Ioo (3 - 2*a) (a + 1)) →  -- Domain condition
  (∀ x, f a (x + 1) = -f a (-x - 1)) →                -- Odd function condition
  a = 2 := by sorry

end odd_function_domain_l1062_106238


namespace arcadia_population_growth_l1062_106204

/-- Represents the population of Arcadia at a given year -/
def population (year : ℕ) : ℕ :=
  if year ≤ 2020 then 250
  else 250 * (3 ^ ((year - 2020) / 25))

/-- The year we're trying to prove -/
def target_year : ℕ := 2095

/-- The population threshold we're trying to exceed -/
def population_threshold : ℕ := 6000

theorem arcadia_population_growth :
  (population target_year > population_threshold) ∧
  (∀ y : ℕ, y < target_year → population y ≤ population_threshold) :=
by sorry

end arcadia_population_growth_l1062_106204


namespace smallest_base_perfect_square_l1062_106260

/-- The smallest integer b > 3 for which 34_b is a perfect square -/
theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ 
  (∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (y : ℕ), 3*x + 4 = y^2) ∧
  (∃ (y : ℕ), 3*b + 4 = y^2) :=
by sorry

end smallest_base_perfect_square_l1062_106260


namespace more_boys_probability_l1062_106298

-- Define the possible number of children
inductive ChildCount : Type
  | zero : ChildCount
  | one : ChildCount
  | two : ChildCount
  | three : ChildCount

-- Define the probability distribution for the number of children
def childCountProb : ChildCount → ℚ
  | ChildCount.zero => 1/15
  | ChildCount.one => 6/15
  | ChildCount.two => 6/15
  | ChildCount.three => 2/15

-- Define the probability of a child being a boy
def boyProb : ℚ := 1/2

-- Define the event of having more boys than girls
def moreBoysEvent : ChildCount → ℚ
  | ChildCount.zero => 0
  | ChildCount.one => 1/2
  | ChildCount.two => 1/4
  | ChildCount.three => 1/2

-- State the theorem
theorem more_boys_probability :
  (moreBoysEvent ChildCount.zero * childCountProb ChildCount.zero +
   moreBoysEvent ChildCount.one * childCountProb ChildCount.one +
   moreBoysEvent ChildCount.two * childCountProb ChildCount.two +
   moreBoysEvent ChildCount.three * childCountProb ChildCount.three) = 11/30 := by
  sorry

end more_boys_probability_l1062_106298


namespace fraction_sum_simplification_l1062_106210

theorem fraction_sum_simplification : 
  5 / (1/(1*2) + 1/(2*3) + 1/(3*4) + 1/(4*5) + 1/(5*6)) = 6 := by
  sorry

end fraction_sum_simplification_l1062_106210


namespace smallest_sum_of_a_and_b_l1062_106289

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + 2*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + 2*a = 0) :
  a + b ≥ 2 * Real.sqrt 2 + 4/3 * Real.sqrt (Real.sqrt 2) := by
  sorry

end smallest_sum_of_a_and_b_l1062_106289


namespace right_angled_triangle_l1062_106213

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem right_angled_triangle (t : Triangle) 
  (h : (Real.cos (t.A / 2))^2 = (t.b + t.c) / (2 * t.c)) : 
  t.C = π / 2 := by
  sorry

end right_angled_triangle_l1062_106213


namespace equilateral_triangle_circle_radii_l1062_106233

/-- For an equilateral triangle with side length a, prove the radii of circumscribed and inscribed circles. -/
theorem equilateral_triangle_circle_radii (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ),
    R = a * Real.sqrt 3 / 3 ∧
    r = a * Real.sqrt 3 / 6 ∧
    R = 2 * r :=
by sorry

end equilateral_triangle_circle_radii_l1062_106233


namespace inequality_proof_l1062_106220

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^3 + b^3 = c^3) :
  a^2 + b^2 - c^2 > 6*(c - a)*(c - b) := by
  sorry

end inequality_proof_l1062_106220


namespace always_even_expression_l1062_106222

theorem always_even_expression (x y : ℕ) : 
  x ∈ Finset.range 15 → 
  y ∈ Finset.range 15 → 
  x ≠ y → 
  Even (x * y - 2 * x - 2 * y) := by
  sorry

#check always_even_expression

end always_even_expression_l1062_106222


namespace four_digit_integers_with_five_or_seven_l1062_106239

theorem four_digit_integers_with_five_or_seven (total_four_digit : Nat) 
  (four_digit_without_five_or_seven : Nat) :
  total_four_digit = 9000 →
  four_digit_without_five_or_seven = 3584 →
  total_four_digit - four_digit_without_five_or_seven = 5416 := by
  sorry

end four_digit_integers_with_five_or_seven_l1062_106239


namespace solve_for_a_l1062_106240

theorem solve_for_a (x a : ℝ) : 2 * x + a - 8 = 0 → x = 2 → a = 4 := by
  sorry

end solve_for_a_l1062_106240


namespace alternating_squares_sum_l1062_106207

theorem alternating_squares_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 268 := by
  sorry

end alternating_squares_sum_l1062_106207


namespace percentage_problem_l1062_106259

theorem percentage_problem (x : ℝ) (h : 0.05 * x = 8) : 0.25 * x = 40 := by
  sorry

end percentage_problem_l1062_106259


namespace four_percent_of_fifty_l1062_106258

theorem four_percent_of_fifty : ∃ x : ℝ, x = 50 * (4 / 100) ∧ x = 2 := by
  sorry

end four_percent_of_fifty_l1062_106258


namespace oliver_candy_theorem_l1062_106236

/-- Oliver's Halloween candy problem -/
theorem oliver_candy_theorem (initial_candy : ℕ) (candy_given : ℕ) (remaining_candy : ℕ) :
  initial_candy = 78 →
  candy_given = 10 →
  remaining_candy = initial_candy - candy_given →
  remaining_candy = 68 :=
by
  sorry

end oliver_candy_theorem_l1062_106236


namespace min_omega_for_sine_symmetry_l1062_106296

theorem min_omega_for_sine_symmetry :
  ∀ ω : ℕ+,
  (∀ x : ℝ, ∃ y : ℝ, y = Real.sin (ω * x + π / 6)) →
  (∀ x : ℝ, Real.sin (ω * (π / 3 - x) + π / 6) = Real.sin (ω * x + π / 6)) →
  2 ≤ ω :=
by
  sorry

end min_omega_for_sine_symmetry_l1062_106296


namespace line_equation_proof_l1062_106251

-- Define the circle P
def circle_P (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the parabola S
def parabola_S (x y : ℝ) : Prop := y = x^2 / 8

-- Define a line passing through a point
def line_through_point (k m x y : ℝ) : Prop := y = k*x + m

-- Define the center of the circle
def circle_center : ℝ × ℝ := (0, 2)

-- Define the property of four points being in arithmetic sequence
def arithmetic_sequence (a b c d : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let d2 := Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2)
  let d3 := Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2)
  d2 - d1 = d3 - d2

theorem line_equation_proof :
  ∀ (k m : ℝ) (a b c d : ℝ × ℝ),
    (∀ x y, line_through_point k m x y → (circle_P x y ∨ parabola_S x y)) →
    line_through_point k m circle_center.1 circle_center.2 →
    arithmetic_sequence a b c d →
    (k = -Real.sqrt 2 / 2 ∨ k = Real.sqrt 2 / 2) ∧ m = 2 :=
sorry

end line_equation_proof_l1062_106251


namespace unique_solution_is_zero_function_l1062_106241

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 * y) = f (x * y) + y * f (f x + y)

/-- The theorem stating that the only function satisfying the equation is the zero function -/
theorem unique_solution_is_zero_function
  (f : ℝ → ℝ) (h : SatisfiesFunctionalEquation f) :
  ∀ y : ℝ, f y = 0 := by
  sorry

end unique_solution_is_zero_function_l1062_106241


namespace coin_toss_probability_l1062_106290

theorem coin_toss_probability : 
  let n : ℕ := 5
  let p_tail : ℚ := 1 / 2
  let p_all_tails : ℚ := p_tail ^ n
  let p_at_least_one_head : ℚ := 1 - p_all_tails
  p_at_least_one_head = 31 / 32 := by
sorry

end coin_toss_probability_l1062_106290


namespace q_necessary_not_sufficient_l1062_106252

/-- A function f is monotonically increasing on an interval if for any two points x and y in that interval, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x³ + 2x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The statement p: f(x) is monotonically increasing in (-∞, +∞) -/
def p (m : ℝ) : Prop := MonotonicallyIncreasing (f m)

/-- The statement q: m > 4/3 -/
def q (m : ℝ) : Prop := m > 4/3

/-- Theorem stating that q is a necessary but not sufficient condition for p -/
theorem q_necessary_not_sufficient :
  (∀ m : ℝ, p m → q m) ∧ (∃ m : ℝ, q m ∧ ¬(p m)) := by sorry

end q_necessary_not_sufficient_l1062_106252


namespace always_positive_l1062_106295

-- Define a monotonically increasing odd function
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_f : MonoIncreasingOddFunction f)
  (h_a : ArithmeticSequence a)
  (h_a3 : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end always_positive_l1062_106295


namespace intersection_A_B_union_A_B_complement_A_l1062_106223

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Theorem for the complement of A
theorem complement_A : Aᶜ = {x | x < -1 ∨ 2 ≤ x} := by sorry

end intersection_A_B_union_A_B_complement_A_l1062_106223


namespace vector_inequality_l1062_106278

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_inequality (a b c : V) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖a + b + c‖ ≥ ‖a + b‖ + ‖b + c‖ + ‖c + a‖ := by
  sorry

end vector_inequality_l1062_106278


namespace min_disks_required_l1062_106285

def total_files : ℕ := 40
def disk_capacity : ℚ := 2

def file_sizes : List ℚ := List.replicate 8 0.9 ++ List.replicate 20 0.6 ++ List.replicate 12 0.5

def is_valid_disk_assignment (assignment : List (List ℚ)) : Prop :=
  assignment.all (λ disk => disk.sum ≤ disk_capacity) ∧
  assignment.join.length = total_files ∧
  assignment.join.toFinset = file_sizes.toFinset

theorem min_disks_required :
  ∃ (assignment : List (List ℚ)),
    is_valid_disk_assignment assignment ∧
    assignment.length = 15 ∧
    ∀ (other_assignment : List (List ℚ)),
      is_valid_disk_assignment other_assignment →
      other_assignment.length ≥ 15 :=
by sorry

end min_disks_required_l1062_106285


namespace number_of_pieces_l1062_106215

-- Define the rod length in meters
def rod_length_meters : ℝ := 42.5

-- Define the piece length in centimeters
def piece_length_cm : ℝ := 85

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem to prove
theorem number_of_pieces : 
  ⌊(rod_length_meters * meters_to_cm) / piece_length_cm⌋ = 50 := by
  sorry

end number_of_pieces_l1062_106215


namespace jack_additional_money_l1062_106272

/-- The amount of additional money Jack needs to buy socks and shoes -/
theorem jack_additional_money (sock_cost shoes_cost jack_money : ℚ)
  (h1 : sock_cost = 19)
  (h2 : shoes_cost = 92)
  (h3 : jack_money = 40) :
  sock_cost + shoes_cost - jack_money = 71 := by
  sorry

end jack_additional_money_l1062_106272


namespace final_worker_count_l1062_106230

/-- Represents the number of bees in a hive -/
structure BeeHive where
  workers : ℕ
  drones : ℕ
  queen : ℕ

def initial_hive : BeeHive := { workers := 400, drones := 75, queen := 1 }

def bees_leave (hive : BeeHive) (workers_leaving : ℕ) (drones_leaving : ℕ) : BeeHive :=
  { workers := hive.workers - workers_leaving,
    drones := hive.drones - drones_leaving,
    queen := hive.queen }

def workers_return (hive : BeeHive) (returning_workers : ℕ) : BeeHive :=
  { workers := hive.workers + returning_workers,
    drones := hive.drones,
    queen := hive.queen }

theorem final_worker_count :
  let hive1 := bees_leave initial_hive 28 12
  let hive2 := workers_return hive1 15
  hive2.workers = 387 := by sorry

end final_worker_count_l1062_106230


namespace trigonometric_identity_l1062_106262

theorem trigonometric_identity (α β γ : ℝ) : 
  (Real.sin α + Real.sin β + Real.sin γ - Real.sin (α + β + γ)) / 
  (Real.cos α + Real.cos β + Real.cos γ + Real.cos (α + β + γ)) = 
  Real.tan ((α + β) / 2) * Real.tan ((β + γ) / 2) * Real.tan ((γ + α) / 2) := by
  sorry

end trigonometric_identity_l1062_106262


namespace BC_length_is_580_l1062_106248

-- Define a structure for a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def has_integer_lengths (q : Quadrilateral) : Prop := sorry

def right_angle_at_B_and_D (q : Quadrilateral) : Prop := sorry

def AB_equals_BD (q : Quadrilateral) : Prop := sorry

def CD_equals_41 (q : Quadrilateral) : Prop := sorry

-- Define the length of BC
def BC_length (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem BC_length_is_580 (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_integer : has_integer_lengths q)
  (h_right_angles : right_angle_at_B_and_D q)
  (h_AB_BD : AB_equals_BD q)
  (h_CD_41 : CD_equals_41 q) :
  BC_length q = 580 := by sorry

end BC_length_is_580_l1062_106248


namespace jersey_revenue_proof_l1062_106214

/-- The amount of money made from selling jerseys -/
def jersey_revenue (price_per_jersey : ℕ) (jerseys_sold : ℕ) : ℕ :=
  price_per_jersey * jerseys_sold

/-- Proof that the jersey revenue is $25,740 -/
theorem jersey_revenue_proof :
  jersey_revenue 165 156 = 25740 := by
  sorry

end jersey_revenue_proof_l1062_106214


namespace parallelogram_area_l1062_106205

/-- The area of a parallelogram with vertices at (0, 0), (4, 0), (1, 5), and (5, 5) is 20 square units. -/
theorem parallelogram_area : ℝ := by
  -- Define the vertices of the parallelogram
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (4, 0)
  let v3 : ℝ × ℝ := (1, 5)
  let v4 : ℝ × ℝ := (5, 5)

  -- Calculate the area of the parallelogram
  have area : ℝ := 20

  -- Assert that the calculated area is correct
  exact area

end parallelogram_area_l1062_106205


namespace thumbtacks_total_l1062_106263

/-- Given 3 cans of thumbtacks, where 120 thumbtacks are used from each can
    and 30 thumbtacks remain in each can after use, prove that the total
    number of thumbtacks in the three full cans initially was 450. -/
theorem thumbtacks_total (cans : Nat) (used_per_can : Nat) (remaining_per_can : Nat)
    (h1 : cans = 3)
    (h2 : used_per_can = 120)
    (h3 : remaining_per_can = 30) :
    cans * (used_per_can + remaining_per_can) = 450 := by
  sorry

end thumbtacks_total_l1062_106263


namespace jessie_weight_loss_l1062_106255

/-- Jessie's weight loss calculation -/
theorem jessie_weight_loss 
  (weight_before : ℝ) 
  (weight_after : ℝ) 
  (h1 : weight_before = 192) 
  (h2 : weight_after = 66) : 
  weight_before - weight_after = 126 := by
  sorry

end jessie_weight_loss_l1062_106255


namespace min_distance_sum_l1062_106293

/-- A scalene triangle with sides a, b, c where a > b > c -/
structure ScaleneTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  scalene : a > b ∧ b > c
  positive : a > 0 ∧ b > 0 ∧ c > 0

/-- A point inside or on the boundary of a triangle -/
structure TrianglePoint (t : ScaleneTriangle) where
  x : ℝ
  y : ℝ
  z : ℝ
  in_triangle : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z ≤ t.a

/-- The sum of distances from a point to the sides of the triangle -/
def distance_sum (t : ScaleneTriangle) (p : TrianglePoint t) : ℝ :=
  p.x + p.y + p.z

/-- The vertex opposite to the largest side -/
def opposite_vertex (t : ScaleneTriangle) : TrianglePoint t where
  x := t.a
  y := 0
  z := 0
  in_triangle := by sorry

/-- Theorem: The point that minimizes the sum of distances is the vertex opposite to the largest side -/
theorem min_distance_sum (t : ScaleneTriangle) :
  ∀ p : TrianglePoint t, distance_sum t (opposite_vertex t) ≤ distance_sum t p :=
by sorry

end min_distance_sum_l1062_106293


namespace max_value_range_l1062_106224

/-- The function f(x) = -x^2 + 4x - 3 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 3

/-- The theorem stating the range of m for the maximum value of f(x) -/
theorem max_value_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ f m) →
  0 < m ∧ m ≤ 2 :=
by sorry

end max_value_range_l1062_106224


namespace inscribed_circle_rectangle_area_l1062_106288

/-- A rectangle with a circle inscribed such that the circle is tangent to three sides of the rectangle and its center lies on a diagonal of the rectangle. -/
structure InscribedCircleRectangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The width of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The circle is tangent to three sides of the rectangle -/
  tangent_to_sides : w = 2 * r ∧ h = r
  /-- The center of the circle lies on a diagonal of the rectangle -/
  center_on_diagonal : True

/-- The area of a rectangle with an inscribed circle as described is equal to 2r^2 -/
theorem inscribed_circle_rectangle_area (rect : InscribedCircleRectangle) :
  rect.w * rect.h = 2 * rect.r^2 := by
  sorry

end inscribed_circle_rectangle_area_l1062_106288


namespace rationalize_denominator_l1062_106231

theorem rationalize_denominator :
  ∃ (x : ℝ), x = (Real.sqrt 6 - 1) ∧
  x = (Real.sqrt 8 + Real.sqrt 3) / (Real.sqrt 2 + Real.sqrt 3) := by
  sorry

end rationalize_denominator_l1062_106231


namespace last_digit_of_powers_l1062_106228

theorem last_digit_of_powers (n : Nat) :
  (∃ k : Nat, n = 2^1000 ∧ n % 10 = 6) ∧
  (∃ k : Nat, n = 3^1000 ∧ n % 10 = 1) ∧
  (∃ k : Nat, n = 7^1000 ∧ n % 10 = 1) :=
by sorry

end last_digit_of_powers_l1062_106228


namespace joanne_weekly_earnings_l1062_106211

def main_job_hours : ℝ := 8
def main_job_rate : ℝ := 16
def part_time_hours : ℝ := 2
def part_time_rate : ℝ := 13.5
def days_per_week : ℝ := 5

def weekly_earnings : ℝ := (main_job_hours * main_job_rate + part_time_hours * part_time_rate) * days_per_week

theorem joanne_weekly_earnings : weekly_earnings = 775 := by
  sorry

end joanne_weekly_earnings_l1062_106211


namespace problem_polygon_area_l1062_106266

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.width * r.height

/-- Represents a polygon composed of rectangles -/
structure Polygon where
  rectangles : List Rectangle

/-- Calculates the total area of a polygon -/
def polygonArea (p : Polygon) : ℕ :=
  p.rectangles.map rectangleArea |>.sum

/-- The polygon in the problem -/
def problemPolygon : Polygon :=
  { rectangles := [
      { width := 2, height := 2 },  -- 2x2 square
      { width := 1, height := 2 },  -- 1x2 rectangle
      { width := 1, height := 2 }   -- 1x2 rectangle
    ] 
  }

theorem problem_polygon_area : polygonArea problemPolygon = 8 := by
  sorry

end problem_polygon_area_l1062_106266


namespace football_playtime_l1062_106217

/-- Given a total playtime of 1.5 hours and a basketball playtime of 30 minutes,
    prove that the football playtime is 60 minutes. -/
theorem football_playtime
  (total_time : ℝ)
  (basketball_time : ℕ)
  (h1 : total_time = 1.5)
  (h2 : basketball_time = 30)
  : ↑basketball_time + 60 = total_time * 60 := by
  sorry

end football_playtime_l1062_106217


namespace angle_solution_l1062_106219

def angle_coincides (α : ℝ) : Prop :=
  ∃ k : ℤ, 9 * α = k * 360 + α

theorem angle_solution :
  ∀ α : ℝ, 0 < α → α < 180 → angle_coincides α → (α = 45 ∨ α = 90) :=
by sorry

end angle_solution_l1062_106219


namespace area_not_covered_by_square_l1062_106221

/-- Given a rectangle with dimensions 10 units by 8 units and an inscribed square
    with side length 5 units, the area of the region not covered by the square
    is 55 square units. -/
theorem area_not_covered_by_square (rectangle_length : ℝ) (rectangle_width : ℝ) 
    (square_side : ℝ) (h1 : rectangle_length = 10) (h2 : rectangle_width = 8) 
    (h3 : square_side = 5) : 
    rectangle_length * rectangle_width - square_side^2 = 55 := by
  sorry

end area_not_covered_by_square_l1062_106221


namespace opposite_absolute_values_l1062_106294

theorem opposite_absolute_values (x y : ℝ) : 
  (|x - y + 9| + |2*x + y| = 0) → (x = -3 ∧ y = 6) := by
sorry

end opposite_absolute_values_l1062_106294


namespace wednesday_water_intake_total_water_intake_correct_l1062_106256

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Water intake for a given day -/
def water_intake (d : Day) : ℕ :=
  match d with
  | Day.Monday => 9
  | Day.Tuesday => 8
  | Day.Wednesday => 9  -- This is what we want to prove
  | Day.Thursday => 9
  | Day.Friday => 8
  | Day.Saturday => 9
  | Day.Sunday => 8

/-- Total water intake for the week -/
def total_water_intake : ℕ := 60

/-- Theorem: The water intake on Wednesday is 9 liters -/
theorem wednesday_water_intake :
  water_intake Day.Wednesday = 9 :=
by
  sorry

/-- Theorem: The total water intake for the week is correct -/
theorem total_water_intake_correct :
  (water_intake Day.Monday) +
  (water_intake Day.Tuesday) +
  (water_intake Day.Wednesday) +
  (water_intake Day.Thursday) +
  (water_intake Day.Friday) +
  (water_intake Day.Saturday) +
  (water_intake Day.Sunday) = total_water_intake :=
by
  sorry

end wednesday_water_intake_total_water_intake_correct_l1062_106256


namespace percent_relation_l1062_106282

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) :
  4 * b / a * 100 = 1000 / 3 := by
  sorry

end percent_relation_l1062_106282


namespace unique_prime_sum_10123_l1062_106202

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_sum_10123 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10123 :=
sorry

end unique_prime_sum_10123_l1062_106202


namespace gain_represents_12_meters_l1062_106208

-- Define the total meters of cloth sold
def total_meters : ℝ := 60

-- Define the gain percentage
def gain_percentage : ℝ := 0.20

-- Define the cost price per meter (as a variable)
variable (cost_price : ℝ)

-- Define the selling price per meter
def selling_price (cost_price : ℝ) : ℝ := cost_price * (1 + gain_percentage)

-- Define the total gain
def total_gain (cost_price : ℝ) : ℝ := 
  total_meters * selling_price cost_price - total_meters * cost_price

-- Theorem: The gain represents 12 meters of cloth
theorem gain_represents_12_meters (cost_price : ℝ) (cost_price_pos : cost_price > 0) :
  total_gain cost_price = 12 * cost_price := by
  sorry

end gain_represents_12_meters_l1062_106208


namespace perimeter_gt_four_times_circumradius_l1062_106200

/-- Definition of an acute-angled triangle -/
def IsAcuteAngledTriangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Perimeter of a triangle -/
def Perimeter (a b c : ℝ) : ℝ := a + b + c

/-- Circumradius of a triangle using the formula R = abc / (4A) where A is the area -/
noncomputable def Circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * area)

/-- Theorem: For any acute-angled triangle, its perimeter is greater than 4 times its circumradius -/
theorem perimeter_gt_four_times_circumradius (a b c : ℝ) 
  (h : IsAcuteAngledTriangle a b c) : 
  Perimeter a b c > 4 * Circumradius a b c := by
  sorry


end perimeter_gt_four_times_circumradius_l1062_106200


namespace total_highlighters_l1062_106244

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 6) (h2 : yellow = 2) (h3 : blue = 4) :
  pink + yellow + blue = 12 := by
  sorry

end total_highlighters_l1062_106244


namespace peter_erasers_l1062_106242

theorem peter_erasers (initial_erasers : ℕ) (received_erasers : ℕ) : 
  initial_erasers = 8 → received_erasers = 3 → initial_erasers + received_erasers = 11 := by
  sorry

end peter_erasers_l1062_106242


namespace gcd_of_72_120_168_l1062_106206

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_72_120_168_l1062_106206


namespace burger_share_l1062_106232

theorem burger_share (burger_length : ℕ) (inches_per_foot : ℕ) : 
  burger_length = 1 → 
  inches_per_foot = 12 → 
  (burger_length * inches_per_foot) / 2 = 6 :=
by
  sorry

#check burger_share

end burger_share_l1062_106232


namespace consecutive_lcm_inequality_l1062_106273

theorem consecutive_lcm_inequality : ∃ n : ℕ, 
  Nat.lcm (Nat.lcm n (n + 1)) (n + 2) > Nat.lcm (Nat.lcm (n + 3) (n + 4)) (n + 5) := by
  sorry

end consecutive_lcm_inequality_l1062_106273


namespace equation_solution_l1062_106264

theorem equation_solution (x : ℝ) : 
  x ≠ 2 → ((4*x^2 + 3*x + 2)/(x - 2) = 4*x + 5 ↔ x = -4) := by sorry

end equation_solution_l1062_106264


namespace pool_capacity_l1062_106257

/-- Represents the capacity of a pool and properties of a pump -/
structure Pool :=
  (capacity : ℝ)
  (pumpRate : ℝ)
  (pumpTime : ℝ)
  (remainingWater : ℝ)

/-- Theorem stating the capacity of the pool given the conditions -/
theorem pool_capacity 
  (p : Pool)
  (h1 : p.pumpRate = 2/3)
  (h2 : p.pumpTime = 7.5)
  (h3 : p.pumpTime * 8 = 0.15 * 60)
  (h4 : p.remainingWater = 25)
  (h5 : p.capacity * (1 - p.pumpRate * (0.15 * 60 / p.pumpTime)) = p.remainingWater) :
  p.capacity = 125 := by
  sorry

end pool_capacity_l1062_106257


namespace apple_distribution_l1062_106292

theorem apple_distribution (total_apples : ℕ) (apples_per_classmate : ℕ) 
  (h1 : total_apples = 15) (h2 : apples_per_classmate = 5) : 
  total_apples / apples_per_classmate = 3 := by
  sorry

end apple_distribution_l1062_106292


namespace divisible_by_seven_last_digits_l1062_106249

theorem divisible_by_seven_last_digits :
  ∃ (S : Finset Nat), (∀ n : Nat, n % 10 ∈ S ↔ ∃ m : Nat, m % 7 = 0 ∧ m % 10 = n % 10) ∧ Finset.card S = 2 :=
by sorry

end divisible_by_seven_last_digits_l1062_106249


namespace perpendicular_lines_b_value_l1062_106280

-- Define the slopes of the two lines
def slope1 : ℚ := -2/3
def slope2 (b : ℚ) : ℚ := -b/3

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem perpendicular_lines_b_value :
  ∃ b : ℚ, perpendicular b ∧ b = -9/2 := by sorry

end perpendicular_lines_b_value_l1062_106280


namespace movie_collection_size_l1062_106227

theorem movie_collection_size :
  ∀ (dvd_count blu_count : ℕ),
  (dvd_count : ℚ) / blu_count = 17 / 4 →
  (dvd_count : ℚ) / (blu_count - 4) = 9 / 2 →
  dvd_count + blu_count = 378 :=
by
  sorry

end movie_collection_size_l1062_106227
