import Mathlib

namespace cubic_root_sum_l2933_293378

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 7*p - 1 = 0 ∧ 
  q^3 - 6*q^2 + 7*q - 1 = 0 ∧ 
  r^3 - 6*r^2 + 7*r - 1 = 0 →
  p / (q*r + 1) + q / (p*r + 1) + r / (p*q + 1) = 61/15 :=
by sorry

end cubic_root_sum_l2933_293378


namespace average_weight_increase_l2933_293320

/-- Proves that replacing a person in a group of 5 increases the average weight by 1.5 kg -/
theorem average_weight_increase (group_size : ℕ) (old_weight new_weight : ℝ) :
  group_size = 5 →
  old_weight = 65 →
  new_weight = 72.5 →
  (new_weight - old_weight) / group_size = 1.5 := by
sorry

end average_weight_increase_l2933_293320


namespace inequality_properties_l2933_293344

theorem inequality_properties (a b c : ℝ) : 
  (a^2 > b^2 → abs a > abs b) ∧ 
  (a > b ↔ a + c > b + c) := by sorry

end inequality_properties_l2933_293344


namespace quadratic_equation_solutions_l2933_293355

theorem quadratic_equation_solutions :
  (∀ x, 3 * x^2 - 6 * x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x, x^2 + 4 * x - 1 = 0 ↔ x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end quadratic_equation_solutions_l2933_293355


namespace income_distribution_l2933_293327

theorem income_distribution (total_income : ℝ) 
  (h_total : total_income = 100) 
  (food_percent : ℝ) (h_food : food_percent = 35)
  (education_percent : ℝ) (h_education : education_percent = 25)
  (rent_percent : ℝ) (h_rent : rent_percent = 80) : 
  (total_income - (food_percent + education_percent) * total_income / 100 - 
   rent_percent * (total_income - (food_percent + education_percent) * total_income / 100) / 100) / 
  total_income * 100 = 8 := by
  sorry

end income_distribution_l2933_293327


namespace soccer_balls_count_l2933_293372

/-- The number of soccer balls in the gym. -/
def soccer_balls : ℕ := 20

/-- The number of baseballs in the gym. -/
def baseballs : ℕ := 5 * soccer_balls

/-- The number of volleyballs in the gym. -/
def volleyballs : ℕ := 3 * soccer_balls

/-- Theorem stating that the number of soccer balls is 20, given the conditions of the problem. -/
theorem soccer_balls_count :
  soccer_balls = 20 ∧
  baseballs = 5 * soccer_balls ∧
  volleyballs = 3 * soccer_balls ∧
  baseballs + volleyballs = 160 :=
by sorry

end soccer_balls_count_l2933_293372


namespace zero_point_not_implies_product_negative_l2933_293310

-- Define a continuous function on a closed interval
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOn f (Set.Icc a b)

-- Define the existence of a zero point in an open interval
def HasZeroInOpenInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem zero_point_not_implies_product_negative
  (f : ℝ → ℝ) (a b : ℝ) (h1 : a < b) (h2 : ContinuousOnInterval f a b) :
  HasZeroInOpenInterval f a b → (f a) * (f b) < 0 → False :=
sorry

end zero_point_not_implies_product_negative_l2933_293310


namespace power_of_64_two_thirds_l2933_293381

theorem power_of_64_two_thirds : (64 : ℝ) ^ (2/3) = 16 := by
  sorry

end power_of_64_two_thirds_l2933_293381


namespace unique_solution_l2933_293330

def repeating_decimal_2 (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

def repeating_decimal_3 (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c) / 999

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

theorem unique_solution (a b c : ℕ) :
  is_digit a ∧ is_digit b ∧ is_digit c →
  repeating_decimal_2 a b + repeating_decimal_3 a b c = 35 / 37 →
  a = 5 ∧ b = 3 ∧ c = 0 :=
sorry

end unique_solution_l2933_293330


namespace simplify_expression_l2933_293368

theorem simplify_expression (a : ℝ) (h : a < -3) :
  Real.sqrt ((2 * a - 1)^2) + Real.sqrt ((a + 3)^2) = -3 * a - 2 := by
  sorry

end simplify_expression_l2933_293368


namespace triangle_abc_properties_l2933_293361

/-- Triangle ABC with vertices A(-1,5), B(-2,-1), and C(4,3) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The specific triangle ABC from the problem -/
def triangleABC : Triangle :=
  { A := (-1, 5)
  , B := (-2, -1)
  , C := (4, 3) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Definition of an altitude in a triangle -/
def isAltitude (t : Triangle) (l : Line) : Prop :=
  sorry

/-- The area of a triangle -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Main theorem stating the properties of triangle ABC -/
theorem triangle_abc_properties :
  let t := triangleABC
  let altitude := Line.mk 3 2 (-7)
  isAltitude t altitude ∧ triangleArea t = 5 := by
  sorry

end triangle_abc_properties_l2933_293361


namespace arithmetic_sum_special_case_l2933_293364

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ d n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sum_special_case (k : ℕ) :
  arithmetic_sum (k^2 - k + 1) 1 (2*k + 1) = (2*k + 1) * (k^2 + 1) := by
  sorry

end arithmetic_sum_special_case_l2933_293364


namespace cos_negative_seventy_nine_pi_sixths_l2933_293331

theorem cos_negative_seventy_nine_pi_sixths : 
  Real.cos (-79 * Real.pi / 6) = -Real.sqrt 3 / 2 := by
  sorry

end cos_negative_seventy_nine_pi_sixths_l2933_293331


namespace discounted_price_calculation_l2933_293394

/-- The original price of the coat -/
def original_price : ℝ := 120

/-- The first discount percentage -/
def first_discount : ℝ := 0.25

/-- The second discount percentage -/
def second_discount : ℝ := 0.20

/-- The final price after both discounts -/
def final_price : ℝ := 72

/-- Theorem stating that applying the two discounts sequentially results in the final price -/
theorem discounted_price_calculation :
  (1 - second_discount) * ((1 - first_discount) * original_price) = final_price :=
by sorry

end discounted_price_calculation_l2933_293394


namespace zoe_money_made_l2933_293388

/-- Calculates the money made from selling chocolate bars -/
def money_made (cost_per_bar : ℕ) (total_bars : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * cost_per_bar

/-- Theorem: Zoe made $42 from selling chocolate bars -/
theorem zoe_money_made :
  let cost_per_bar : ℕ := 6
  let total_bars : ℕ := 13
  let unsold_bars : ℕ := 6
  money_made cost_per_bar total_bars unsold_bars = 42 := by
sorry

end zoe_money_made_l2933_293388


namespace min_value_theorem_l2933_293366

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  (1 / x + 1 / (3 * y)) ≥ 4 := by
  sorry

end min_value_theorem_l2933_293366


namespace expression_evaluation_l2933_293324

theorem expression_evaluation (x : ℝ) (h : x = -3) :
  (5 + 2*x*(x+2) - 4^2) / (x - 4 + x^2) = -5/2 := by
  sorry

end expression_evaluation_l2933_293324


namespace min_value_reciprocal_sum_l2933_293386

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x) + (9 / y) ≥ 16 ∧
  ((1 / x) + (9 / y) = 16 ↔ x = 1/4 ∧ y = 3/4) :=
sorry

end min_value_reciprocal_sum_l2933_293386


namespace turtle_theorem_l2933_293328

def turtle_problem (initial : ℕ) : ℕ :=
  let additional := 3 * initial - 2
  let total := initial + additional
  total / 2

theorem turtle_theorem : turtle_problem 9 = 17 := by
  sorry

end turtle_theorem_l2933_293328


namespace darcie_father_age_l2933_293397

def darcie_age : ℕ := 4

theorem darcie_father_age (mother_age father_age : ℕ) 
  (h1 : darcie_age = mother_age / 6)
  (h2 : mother_age * 5 = father_age * 4) : 
  father_age = 30 := by
  sorry

end darcie_father_age_l2933_293397


namespace dad_catch_is_27_l2933_293343

/-- The number of salmons Hazel caught -/
def hazel_catch : ℕ := 24

/-- The total number of salmons caught by Hazel and her dad -/
def total_catch : ℕ := 51

/-- The number of salmons Hazel's dad caught -/
def dad_catch : ℕ := total_catch - hazel_catch

theorem dad_catch_is_27 : dad_catch = 27 := by
  sorry

end dad_catch_is_27_l2933_293343


namespace set_equality_implies_a_values_l2933_293365

theorem set_equality_implies_a_values (a : ℝ) : 
  ({0, -1, 2*a} : Set ℝ) = {a-1, -|a|, a+1} ↔ a = 1 ∨ a = -1 := by
  sorry

end set_equality_implies_a_values_l2933_293365


namespace harmonic_mean_4_5_10_l2933_293346

def harmonic_mean (a b c : ℚ) : ℚ := 3 / (1/a + 1/b + 1/c)

theorem harmonic_mean_4_5_10 :
  harmonic_mean 4 5 10 = 60 / 11 := by
  sorry

end harmonic_mean_4_5_10_l2933_293346


namespace equation_solution_inequalities_solution_l2933_293399

-- Part 1: Equation
theorem equation_solution :
  ∃! x : ℝ, (1 / (x - 3) = 3 + x / (3 - x)) ∧ x = 5 := by sorry

-- Part 2: System of Inequalities
theorem inequalities_solution :
  ∀ x : ℝ, ((x - 1) / 2 < (x + 1) / 3 ∧ x - 3 * (x - 2) ≤ 4) ↔ (1 ≤ x ∧ x < 5) := by sorry

end equation_solution_inequalities_solution_l2933_293399


namespace orchestra_members_count_l2933_293360

theorem orchestra_members_count :
  ∃! n : ℕ, 200 < n ∧ n < 300 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    n % 7 = 4 ∧
    n = 158 :=
by sorry

end orchestra_members_count_l2933_293360


namespace f_one_geq_25_l2933_293363

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_one_geq_25 (m : ℝ) (h : ∀ x ≥ -2, Monotone (f m)) :
  f m 1 ≥ 25 := by sorry

end f_one_geq_25_l2933_293363


namespace rope_problem_l2933_293367

theorem rope_problem (x : ℝ) :
  (8 : ℝ)^2 + (x - 3)^2 = x^2 :=
by sorry

end rope_problem_l2933_293367


namespace ramanujan_number_l2933_293352

theorem ramanujan_number (r h : ℂ) : 
  r * h = 50 - 14 * I →
  h = 7 + 2 * I →
  r = 6 - (198 / 53) * I := by
sorry

end ramanujan_number_l2933_293352


namespace largest_class_size_l2933_293319

theorem largest_class_size (n : ℕ) (total : ℕ) : 
  n = 5 → 
  total = 115 → 
  ∃ x : ℕ, x > 0 ∧ 
    (x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = total) ∧ 
    x = 27 := by
  sorry

end largest_class_size_l2933_293319


namespace absolute_value_implies_inequality_l2933_293359

theorem absolute_value_implies_inequality (x : ℝ) : 
  |x - 1| = 1 - x → x ≤ 1 := by
sorry

end absolute_value_implies_inequality_l2933_293359


namespace completing_square_sum_l2933_293353

theorem completing_square_sum (a b c : ℤ) : 
  (∀ x, 36 * x^2 - 60 * x + 25 = 0 ↔ (a * x + b)^2 = c) →
  a > 0 →
  a + b + c = 26 :=
by sorry

end completing_square_sum_l2933_293353


namespace multiple_of_power_minus_one_l2933_293389

theorem multiple_of_power_minus_one (a b c : ℕ) :
  (∃ k : ℤ, 2^a + 2^b + 1 = k * (2^c - 1)) ↔
  ((a = 0 ∧ b = 0 ∧ c = 2) ∨ (a = 1 ∧ b = 2 ∧ c = 3) ∨ (a = 2 ∧ b = 1 ∧ c = 3)) :=
by sorry

end multiple_of_power_minus_one_l2933_293389


namespace floor_sum_product_l2933_293348

theorem floor_sum_product : 3 * (⌊(12.7 : ℝ)⌋ + ⌊(-12.7 : ℝ)⌋) = -3 := by sorry

end floor_sum_product_l2933_293348


namespace alice_baked_five_more_l2933_293340

/-- The number of additional chocolate chip cookies Alice baked after the accident -/
def additional_cookies (alice_initial bob_initial thrown_away bob_additional final_count : ℕ) : ℕ :=
  final_count - (alice_initial + bob_initial - thrown_away + bob_additional)

/-- Theorem stating that Alice baked 5 more chocolate chip cookies after the accident -/
theorem alice_baked_five_more : additional_cookies 74 7 29 36 93 = 5 := by
  sorry

end alice_baked_five_more_l2933_293340


namespace select_chess_team_l2933_293338

/-- The number of ways to select a team of 4 players from 10, where two are twins and both twins can't be on the team -/
def select_team (total_players : ℕ) (team_size : ℕ) (num_twins : ℕ) : ℕ :=
  Nat.choose total_players team_size - Nat.choose (total_players - num_twins) (team_size - num_twins)

/-- Theorem stating that selecting 4 players from 10, where two are twins and both twins can't be on the team, results in 182 ways -/
theorem select_chess_team : select_team 10 4 2 = 182 := by
  sorry

end select_chess_team_l2933_293338


namespace original_number_proof_l2933_293317

theorem original_number_proof (x : ℝ) : 
  (x * 1.125 - x * 0.75 = 30) → x = 80 := by
  sorry

end original_number_proof_l2933_293317


namespace cat_roaming_area_l2933_293322

/-- The area accessible to a cat tethered to a circular water tank -/
theorem cat_roaming_area (tank_radius rope_length : ℝ) (h1 : tank_radius = 20) (h2 : rope_length = 10) :
  π * (tank_radius + rope_length)^2 - π * tank_radius^2 = 500 * π :=
by sorry

end cat_roaming_area_l2933_293322


namespace negative_one_point_five_less_than_negative_one_and_one_fifth_l2933_293351

theorem negative_one_point_five_less_than_negative_one_and_one_fifth : -1.5 < -(1 + 1/5) := by
  sorry

end negative_one_point_five_less_than_negative_one_and_one_fifth_l2933_293351


namespace bike_riders_count_l2933_293374

theorem bike_riders_count (total : ℕ) (difference : ℕ) :
  total = 676 →
  difference = 178 →
  ∃ (bikers hikers : ℕ),
    total = bikers + hikers ∧
    hikers = bikers + difference ∧
    bikers = 249 := by
  sorry

end bike_riders_count_l2933_293374


namespace square_difference_formula_l2933_293332

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8 / 15) (h2 : x - y = 1 / 45) : 
  x^2 - y^2 = 8 / 675 := by
  sorry

end square_difference_formula_l2933_293332


namespace hyperbola_m_value_l2933_293371

-- Define the hyperbola equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 - y^2/m = 1

-- Define the focus point
def focus : ℝ × ℝ := (-3, 0)

-- Theorem statement
theorem hyperbola_m_value :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), hyperbola_equation x y m → True) ∧ 
    (focus.1^2 = 1 + m) →
    m = 8 := by sorry

end hyperbola_m_value_l2933_293371


namespace koolaid_percentage_l2933_293357

/-- Calculates the percentage of Kool-Aid powder in a mixture after evaporation and water addition -/
theorem koolaid_percentage
  (initial_powder : ℚ)
  (initial_water : ℚ)
  (evaporation_rate : ℚ)
  (water_increase_factor : ℚ)
  (h1 : initial_powder = 3)
  (h2 : initial_water = 20)
  (h3 : evaporation_rate = 1/4)
  (h4 : water_increase_factor = 5) :
  let remaining_water := initial_water * (1 - evaporation_rate)
  let final_water := remaining_water * water_increase_factor
  let final_mixture := initial_powder + final_water
  initial_powder / final_mixture = 1/26 :=
sorry

end koolaid_percentage_l2933_293357


namespace kiley_ate_quarter_cheesecake_l2933_293373

/-- Represents the properties of a cheesecake and Kiley's consumption -/
structure CheesecakeConsumption where
  calories_per_slice : ℕ
  total_calories : ℕ
  slices_eaten : ℕ

/-- Calculates the percentage of cheesecake eaten -/
def percentage_eaten (c : CheesecakeConsumption) : ℚ :=
  (c.calories_per_slice * c.slices_eaten : ℚ) / c.total_calories * 100

/-- Theorem stating that Kiley ate 25% of the cheesecake -/
theorem kiley_ate_quarter_cheesecake :
  let c : CheesecakeConsumption := {
    calories_per_slice := 350,
    total_calories := 2800,
    slices_eaten := 2
  }
  percentage_eaten c = 25 := by
  sorry


end kiley_ate_quarter_cheesecake_l2933_293373


namespace min_correct_answers_is_17_l2933_293342

/-- AMC 12 scoring system and Sarah's strategy -/
structure AMC12 where
  total_questions : Nat
  attempted_questions : Nat
  points_correct : Nat
  points_incorrect : Nat
  points_unanswered : Nat
  min_score : Nat

/-- Calculate the minimum number of correct answers needed -/
def min_correct_answers (amc : AMC12) : Nat :=
  let unanswered := amc.total_questions - amc.attempted_questions
  let points_from_unanswered := unanswered * amc.points_unanswered
  let required_points := amc.min_score - points_from_unanswered
  (required_points + amc.points_correct - 1) / amc.points_correct

/-- Theorem stating the minimum number of correct answers needed -/
theorem min_correct_answers_is_17 (amc : AMC12) 
  (h1 : amc.total_questions = 30)
  (h2 : amc.attempted_questions = 24)
  (h3 : amc.points_correct = 7)
  (h4 : amc.points_incorrect = 0)
  (h5 : amc.points_unanswered = 2)
  (h6 : amc.min_score = 130) : 
  min_correct_answers amc = 17 := by
  sorry

#eval min_correct_answers {
  total_questions := 30,
  attempted_questions := 24,
  points_correct := 7,
  points_incorrect := 0,
  points_unanswered := 2,
  min_score := 130
}

end min_correct_answers_is_17_l2933_293342


namespace min_reciprocal_sum_l2933_293318

theorem min_reciprocal_sum : ∀ a b : ℕ+, 
  4 * a + b = 6 → 
  (1 : ℝ) / 1 + (1 : ℝ) / 2 ≤ (1 : ℝ) / a + (1 : ℝ) / b :=
by sorry

end min_reciprocal_sum_l2933_293318


namespace quadratic_factorization_l2933_293379

theorem quadratic_factorization (a b c : ℤ) : 
  (∀ x, x^2 + 7*x - 18 = (x + a) * (x + b)) →
  (∀ x, x^2 + 11*x + 24 = (x + b) * (x + c)) →
  a + b + c = 20 := by
sorry

end quadratic_factorization_l2933_293379


namespace solution_set_equivalence_a_range_l2933_293349

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a x + x^2 + 1

-- Define the solution set of f(x) ≤ 0
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ 0}

-- Define the condition that g has two distinct zeros in (1,2)
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ g a x₁ = 0 ∧ g a x₂ = 0 ∧ x₁ ≠ x₂

-- Theorem 1
theorem solution_set_equivalence (a : ℝ) :
  solution_set a = Set.Icc 1 2 →
  {x : ℝ | f a x ≥ 1 - x^2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 1} := by sorry

-- Theorem 2
theorem a_range (a : ℝ) :
  has_two_zeros a → -5 < a ∧ a < -2 * Real.sqrt 6 := by sorry

end solution_set_equivalence_a_range_l2933_293349


namespace clairaut_general_solution_l2933_293325

/-- Clairaut's equation -/
def clairaut_equation (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = x * (deriv y x) + 1 / (2 * deriv y x)

/-- General solution of Clairaut's equation -/
def is_general_solution (y : ℝ → ℝ) : Prop :=
  ∃ C : ℝ, (∀ x, y x = C * x + 1 / (2 * C)) ∨ (∀ x, (y x)^2 = 2 * x)

/-- Theorem: The general solution satisfies Clairaut's equation -/
theorem clairaut_general_solution :
  ∀ y : ℝ → ℝ, is_general_solution y → ∀ x : ℝ, clairaut_equation y x :=
sorry

end clairaut_general_solution_l2933_293325


namespace correct_calculation_l2933_293380

theorem correct_calculation (x : ℝ) : (x + 20) * 5 = 225 → (x + 20) / 5 = 9 := by
  sorry

end correct_calculation_l2933_293380


namespace max_beads_is_27_l2933_293306

/-- Represents the maximum number of weighings allowed -/
def max_weighings : ℕ := 3

/-- Represents the number of groups in each weighing -/
def groups_per_weighing : ℕ := 3

/-- Calculates the maximum number of beads that can be in the pile -/
def max_beads : ℕ := groups_per_weighing ^ max_weighings

/-- Theorem stating that the maximum number of beads is 27 -/
theorem max_beads_is_27 : max_beads = 27 := by
  sorry

#eval max_beads -- Should output 27

end max_beads_is_27_l2933_293306


namespace base_conversion_142_to_7_l2933_293362

/-- Converts a natural number from base 10 to base 7 --/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 7 to a natural number in base 10 --/
def fromBase7 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_142_to_7 :
  toBase7 142 = [2, 6, 2] ∧ fromBase7 [2, 6, 2] = 142 := by
  sorry

end base_conversion_142_to_7_l2933_293362


namespace three_heads_after_three_tails_probability_l2933_293313

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a sequence of coin flips -/
def FlipSequence := List CoinFlip

/-- A fair coin has equal probability of heads and tails -/
def isFairCoin (p : CoinFlip → ℝ) : Prop :=
  p CoinFlip.Heads = 1/2 ∧ p CoinFlip.Tails = 1/2

/-- Checks if a sequence ends with three heads in a row -/
def endsWithThreeHeads : FlipSequence → Bool := sorry

/-- Checks if a sequence contains three tails before three heads -/
def hasThreeTailsBeforeThreeHeads : FlipSequence → Bool := sorry

/-- The probability of a specific flip sequence occurring -/
def sequenceProbability (s : FlipSequence) (p : CoinFlip → ℝ) : ℝ := sorry

/-- The main theorem to prove -/
theorem three_heads_after_three_tails_probability 
  (p : CoinFlip → ℝ) (h : isFairCoin p) :
  (∃ s : FlipSequence, endsWithThreeHeads s ∧ hasThreeTailsBeforeThreeHeads s ∧
    sequenceProbability s p = 1/192) :=
by sorry

end three_heads_after_three_tails_probability_l2933_293313


namespace min_value_3x_plus_4y_l2933_293301

theorem min_value_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 := by
sorry

end min_value_3x_plus_4y_l2933_293301


namespace min_probability_bound_l2933_293308

open Real

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The probability function P(k) -/
noncomputable def P (k : ℕ) : ℝ :=
  let count := Finset.filter (fun n : ℕ => 
    floor (n / k) + floor ((200 - n) / k) = floor (200 / k)) 
    (Finset.range 199)
  (count.card : ℝ) / 199

theorem min_probability_bound :
  ∀ k : ℕ, k % 2 = 1 → 1 ≤ k → k ≤ 199 → P k ≥ 50 / 101 := by sorry

end min_probability_bound_l2933_293308


namespace max_a_for_monotonic_f_l2933_293354

/-- Given that f(x) = x^3 - ax is monotonically increasing on [1, +∞), 
    the maximum value of a is 3. -/
theorem max_a_for_monotonic_f (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → (x₁^3 - a*x₁) < (x₂^3 - a*x₂)) →
  a ≤ 3 ∧ ∀ b > a, ∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ (x₁^3 - b*x₁) ≥ (x₂^3 - b*x₂) :=
by sorry

end max_a_for_monotonic_f_l2933_293354


namespace largest_integer_with_conditions_l2933_293321

def digit_sum_of_squares (n : ℕ) : ℕ := sorry

def digits_increasing (n : ℕ) : Prop := sorry

def product_of_digits (n : ℕ) : ℕ := sorry

theorem largest_integer_with_conditions (n : ℕ) :
  (digit_sum_of_squares n = 82) →
  digits_increasing n →
  product_of_digits n ≤ 9 := by sorry

end largest_integer_with_conditions_l2933_293321


namespace sqrt_meaningful_range_l2933_293350

theorem sqrt_meaningful_range (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := by sorry

end sqrt_meaningful_range_l2933_293350


namespace line_parameterization_l2933_293383

/-- Given a line y = 2x - 15 parameterized by (x, y) = (g(t), 10t + 5), prove that g(t) = 5t + 10 -/
theorem line_parameterization (g : ℝ → ℝ) : 
  (∀ t : ℝ, 10 * t + 5 = 2 * (g t) - 15) → 
  (∀ t : ℝ, g t = 5 * t + 10) := by
sorry

end line_parameterization_l2933_293383


namespace smallest_lcm_with_gcd_11_l2933_293311

theorem smallest_lcm_with_gcd_11 :
  ∃ (k l : ℕ), 
    1000 ≤ k ∧ k < 10000 ∧
    1000 ≤ l ∧ l < 10000 ∧
    Nat.gcd k l = 11 ∧
    Nat.lcm k l = 92092 ∧
    ∀ (m n : ℕ), 
      1000 ≤ m ∧ m < 10000 ∧
      1000 ≤ n ∧ n < 10000 ∧
      Nat.gcd m n = 11 →
      Nat.lcm m n ≥ 92092 :=
by sorry

end smallest_lcm_with_gcd_11_l2933_293311


namespace complex_division_equality_l2933_293396

theorem complex_division_equality : (1 + 2*Complex.I) / (2 - Complex.I) = Complex.I := by sorry

end complex_division_equality_l2933_293396


namespace rice_cost_is_ten_cents_l2933_293347

/-- The cost of rice per plate in cents -/
def rice_cost_per_plate (total_plates : ℕ) (chicken_cost_per_plate : ℚ) (total_spent : ℚ) : ℚ :=
  (total_spent - (chicken_cost_per_plate * total_plates)) / total_plates * 100

/-- Theorem: The cost of rice per plate is 10 cents -/
theorem rice_cost_is_ten_cents :
  rice_cost_per_plate 100 0.40 50 = 10 := by
  sorry

end rice_cost_is_ten_cents_l2933_293347


namespace derivative_of_f_l2933_293391

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x + x^2

theorem derivative_of_f :
  deriv f = λ x => 3 / x + 2 * x :=
by sorry

end derivative_of_f_l2933_293391


namespace james_weekly_take_home_pay_l2933_293329

/-- Calculates James' weekly take-home pay given his work and tax conditions --/
def jamesTakeHomePay (mainJobRate hourlyRate : ℝ) 
                     (secondJobRatePercentage : ℝ) 
                     (mainJobHours overtimeHours : ℕ) 
                     (secondJobHours : ℕ) 
                     (weekendDays : ℕ)
                     (weekendRate : ℝ)
                     (taxDeductions : ℝ)
                     (federalTaxRate stateTaxRate : ℝ) : ℝ :=
  let secondJobRate := mainJobRate * (1 - secondJobRatePercentage)
  let regularHours := mainJobHours - overtimeHours
  let mainJobEarnings := regularHours * mainJobRate + overtimeHours * mainJobRate * 1.5
  let secondJobEarnings := secondJobHours * secondJobRate
  let weekendEarnings := weekendDays * weekendRate
  let totalEarnings := mainJobEarnings + secondJobEarnings + weekendEarnings
  let taxableIncome := totalEarnings - taxDeductions
  let federalTax := taxableIncome * federalTaxRate
  let stateTax := taxableIncome * stateTaxRate
  let totalTaxes := federalTax + stateTax
  totalEarnings - totalTaxes

/-- Theorem stating that James' weekly take-home pay is $885.30 --/
theorem james_weekly_take_home_pay :
  jamesTakeHomePay 20 20 0.2 30 5 15 2 100 200 0.18 0.05 = 885.30 := by
  sorry

end james_weekly_take_home_pay_l2933_293329


namespace tangent_line_inclination_l2933_293326

theorem tangent_line_inclination (a : ℝ) : 
  (∀ x : ℝ, (fun x => a * x^3 - 2) x = a * x^3 - 2) →
  (slope_at_neg_one : ℝ) →
  slope_at_neg_one = Real.tan (π / 4) →
  slope_at_neg_one = (fun x => 3 * a * x^2) (-1) →
  a = 1/3 := by
sorry

end tangent_line_inclination_l2933_293326


namespace fifth_day_temperature_l2933_293358

/-- Given the average temperatures and ratio of temperatures for specific days,
    prove that the temperature on the fifth day is 32 degrees. -/
theorem fifth_day_temperature
  (avg_first_four : ℝ)
  (avg_second_to_fifth : ℝ)
  (temp_first : ℝ)
  (temp_fifth : ℝ)
  (h1 : avg_first_four = 58)
  (h2 : avg_second_to_fifth = 59)
  (h3 : temp_fifth = (8 / 7) * temp_first)
  (h4 : temp_first + (avg_first_four * 4 - temp_first) = avg_first_four * 4)
  (h5 : (avg_first_four * 4 - temp_first) + temp_fifth = avg_second_to_fifth * 4) :
  temp_fifth = 32 :=
sorry

end fifth_day_temperature_l2933_293358


namespace no_real_solutions_for_sqrt_equation_l2933_293382

theorem no_real_solutions_for_sqrt_equation :
  ∀ z : ℝ, ¬(Real.sqrt (5 - 4*z) = 7) :=
by
  sorry

end no_real_solutions_for_sqrt_equation_l2933_293382


namespace fraction_decomposition_l2933_293384

theorem fraction_decomposition (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1) = 1 / (x - 1) - 1 / (x + 1)) ∧
  (2 * x / (x^2 - 1) = 1 / (x - 1) + 1 / (x + 1)) := by
  sorry

end fraction_decomposition_l2933_293384


namespace square_area_from_adjacent_points_l2933_293398

theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (5, 7)
  let distance := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := distance^2
  area = 25 := by sorry

end square_area_from_adjacent_points_l2933_293398


namespace sin_cos_sum_11_19_l2933_293392

theorem sin_cos_sum_11_19 : 
  Real.sin (11 * π / 180) * Real.cos (19 * π / 180) + 
  Real.cos (11 * π / 180) * Real.sin (19 * π / 180) = 1 / 2 := by
sorry

end sin_cos_sum_11_19_l2933_293392


namespace exists_universal_program_l2933_293300

/- Define the maze structure -/
def Maze := Fin 10 → Fin 10 → Bool

/- Define the robot's position -/
structure Position where
  x : Fin 10
  y : Fin 10

/- Define the possible robot commands -/
inductive Command
| L
| R
| U
| D

/- Define a program as a list of commands -/
def Program := List Command

/- Function to check if a cell is accessible -/
def isAccessible (maze : Maze) (pos : Position) : Bool :=
  maze pos.x pos.y

/- Function to apply a command to a position -/
def applyCommand (maze : Maze) (pos : Position) (cmd : Command) : Position :=
  sorry

/- Function to check if a program visits all accessible cells -/
def visitsAllCells (maze : Maze) (start : Position) (prog : Program) : Prop :=
  sorry

/- The main theorem -/
theorem exists_universal_program :
  ∃ (prog : Program),
    ∀ (maze : Maze) (start : Position),
      visitsAllCells maze start prog :=
sorry

end exists_universal_program_l2933_293300


namespace range_of_a_for_inequality_l2933_293369

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- State the theorem
theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by
  sorry


end range_of_a_for_inequality_l2933_293369


namespace purchase_equivalence_l2933_293312

/-- Proves that if a person can buy exactly 6 items at price x and exactly 8 items at price (x - 1.5),
    then the total amount of money the person has is 36. -/
theorem purchase_equivalence (x : ℝ) :
  (6 * x = 8 * (x - 1.5)) → 6 * x = 36 := by
  sorry

end purchase_equivalence_l2933_293312


namespace cos_sum_equals_one_l2933_293316

theorem cos_sum_equals_one (α β : Real) 
  (h : (Real.cos α * Real.cos (β/2)) / Real.cos (α - β/2) + 
       (Real.cos β * Real.cos (α/2)) / Real.cos (β - α/2) = 1) : 
  Real.cos α + Real.cos β = 1 := by
  sorry

end cos_sum_equals_one_l2933_293316


namespace consecutive_non_prime_powers_l2933_293339

theorem consecutive_non_prime_powers (n : ℕ+) :
  ∃ x : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬∃ (p : ℕ) (k : ℕ), Prime p ∧ (x + i = p^k) :=
sorry

end consecutive_non_prime_powers_l2933_293339


namespace race_finish_order_l2933_293336

/-- Represents a sprinter in the race -/
inductive Sprinter
  | A
  | B
  | C

/-- Represents the order of sprinters -/
def RaceOrder := List Sprinter

/-- Represents the number of position changes for each sprinter -/
def PositionChanges := Sprinter → Nat

/-- Determines if a sprinter started later than another -/
def StartedLater (s1 s2 : Sprinter) : Prop := sorry

/-- Determines if a sprinter finished before another -/
def FinishedBefore (s1 s2 : Sprinter) : Prop := sorry

/-- Determines if a sprinter was delayed at the start -/
def DelayedAtStart (s : Sprinter) : Prop := sorry

theorem race_finish_order :
  ∀ (changes : PositionChanges),
    changes Sprinter.C = 6 →
    changes Sprinter.A = 5 →
    StartedLater Sprinter.B Sprinter.A →
    FinishedBefore Sprinter.B Sprinter.A →
    DelayedAtStart Sprinter.C →
    ∃ (order : RaceOrder),
      order = [Sprinter.B, Sprinter.A, Sprinter.C] :=
by sorry

end race_finish_order_l2933_293336


namespace cookies_per_batch_is_three_l2933_293393

/-- Given the total number of chocolate chips, number of batches, and chips per cookie,
    calculate the number of cookies in a batch. -/
def cookiesPerBatch (totalChips : ℕ) (numBatches : ℕ) (chipsPerCookie : ℕ) : ℕ :=
  (totalChips / numBatches) / chipsPerCookie

/-- Prove that the number of cookies in a batch is 3 given the problem conditions. -/
theorem cookies_per_batch_is_three :
  cookiesPerBatch 81 3 9 = 3 := by
  sorry

#eval cookiesPerBatch 81 3 9

end cookies_per_batch_is_three_l2933_293393


namespace prime_arithmetic_sequence_difference_l2933_293385

theorem prime_arithmetic_sequence_difference (p₁ p₂ p₃ d : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
  p₁ > 3 ∧ p₂ > 3 ∧ p₃ > 3 ∧
  p₂ = p₁ + d ∧ p₃ = p₂ + d →
  6 ∣ d := by
sorry

end prime_arithmetic_sequence_difference_l2933_293385


namespace berry_ratio_l2933_293333

theorem berry_ratio (total : ℕ) (blueberries : ℕ) : 
  total = 42 →
  blueberries = 7 →
  (total / 2 : ℚ) = (total - blueberries - (total / 2) : ℚ) →
  (total - blueberries - (total / 2) : ℚ) / total = 1 / 3 := by
  sorry

end berry_ratio_l2933_293333


namespace probability_of_sum_22_l2933_293302

/-- A function representing the probability of rolling a specific sum with four standard 6-faced dice -/
def probability_of_sum (sum : ℕ) : ℚ :=
  sorry

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Theorem stating the probability of rolling a sum of 22 with four standard 6-faced dice -/
theorem probability_of_sum_22 : probability_of_sum 22 = 5 / 648 := by
  sorry

end probability_of_sum_22_l2933_293302


namespace intersection_A_B_l2933_293376

def A : Set ℤ := {x | |x| < 3}
def B : Set ℤ := {x | |x| > 1}

theorem intersection_A_B : A ∩ B = {-2, 2} := by
  sorry

end intersection_A_B_l2933_293376


namespace school_survey_methods_l2933_293314

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandomSampling
  | StratifiedSampling
  | SystematicSampling

/-- Represents a survey with two sampling methods --/
structure Survey where
  totalStudents : Nat
  firstMethod : SamplingMethod
  secondMethod : SamplingMethod

/-- Defines the specific survey conducted by the school --/
def schoolSurvey : Survey :=
  { totalStudents := 200,
    firstMethod := SamplingMethod.SimpleRandomSampling,
    secondMethod := SamplingMethod.SystematicSampling }

/-- Theorem stating that the school survey uses Simple Random Sampling for the first method
    and Systematic Sampling for the second method --/
theorem school_survey_methods :
  schoolSurvey.firstMethod = SamplingMethod.SimpleRandomSampling ∧
  schoolSurvey.secondMethod = SamplingMethod.SystematicSampling :=
by sorry

end school_survey_methods_l2933_293314


namespace larger_number_proof_l2933_293315

theorem larger_number_proof (x y : ℝ) (h_diff : x - y = 3) (h_sum : x + y = 31) :
  max x y = 17 := by
sorry

end larger_number_proof_l2933_293315


namespace small_cube_edge_length_small_cube_edge_length_proof_l2933_293305

/-- Given a cube made of 8 smaller cubes with a total volume of 1000 cm³,
    the length of one edge of a smaller cube is 5 cm. -/
theorem small_cube_edge_length : ℝ :=
  let total_volume : ℝ := 1000
  let num_small_cubes : ℕ := 8
  let edge_ratio : ℝ := 2  -- ratio of large cube edge to small cube edge
  
  -- Define the volume of the large cube in terms of the small cube's edge length
  let large_cube_volume (small_edge : ℝ) : ℝ := (edge_ratio * small_edge) ^ 3
  
  -- Define the equation: large cube volume equals total volume
  let volume_equation (small_edge : ℝ) : Prop := large_cube_volume small_edge = total_volume
  
  -- The length of one edge of the smaller cube
  5

/-- Proof of the theorem -/
theorem small_cube_edge_length_proof : small_cube_edge_length = 5 := by
  sorry

end small_cube_edge_length_small_cube_edge_length_proof_l2933_293305


namespace salary_calculation_l2933_293390

/-- Prove that if a salary is first increased by 10% and then decreased by 5%,
    resulting in Rs. 4180, the original salary was Rs. 4000. -/
theorem salary_calculation (original : ℝ) : 
  (original * 1.1 * 0.95 = 4180) → original = 4000 := by
  sorry

end salary_calculation_l2933_293390


namespace train_length_l2933_293356

/-- The length of a train given its speed, the speed of a person running in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 69) 
  (h2 : man_speed = 3) 
  (h3 : passing_time = 10) : 
  train_speed * (5/18) * passing_time + man_speed * (5/18) * passing_time = 200 :=
by sorry

end train_length_l2933_293356


namespace property_square_footage_l2933_293334

/-- Given a property worth $333,200 and a price of $98 per square foot,
    prove that the total square footage is 3400 square feet. -/
theorem property_square_footage :
  let property_value : ℕ := 333200
  let price_per_sqft : ℕ := 98
  let total_sqft : ℕ := property_value / price_per_sqft
  total_sqft = 3400 := by
  sorry

end property_square_footage_l2933_293334


namespace smallest_k_satisfying_inequality_l2933_293375

theorem smallest_k_satisfying_inequality (n m : ℕ) (hn : n > 0) (hm : 0 < m ∧ m ≤ 5) :
  (∀ k : ℕ, k % 3 = 0 → (64^k + 32^m > 4^(16 + n^2) → k ≥ 6)) ∧
  (64^6 + 32^m > 4^(16 + n^2)) := by
  sorry

end smallest_k_satisfying_inequality_l2933_293375


namespace inequality_proof_l2933_293377

theorem inequality_proof (a b x : ℝ) (h1 : a * b > 0) (h2 : 0 < x) (h3 : x < π / 2) :
  (1 + a^2 / Real.sin x) * (1 + b^2 / Real.cos x) ≥ ((1 + Real.sqrt 2 * a * b)^2 * Real.sin (2 * x)) / 2 := by
  sorry

end inequality_proof_l2933_293377


namespace sum_of_roots_l2933_293323

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 5 → b * (b - 4) = 5 → a ≠ b → a + b = 4 := by
  sorry

end sum_of_roots_l2933_293323


namespace marias_number_problem_l2933_293341

theorem marias_number_problem (x : ℝ) : 
  (((x - 3) * 3 + 3) / 3 = 10) → x = 12 := by
  sorry

end marias_number_problem_l2933_293341


namespace largest_lcm_with_15_l2933_293307

theorem largest_lcm_with_15 : 
  (Finset.image (fun x => Nat.lcm 15 x) {3, 5, 9, 12, 10, 15}).max = some 60 := by
  sorry

end largest_lcm_with_15_l2933_293307


namespace email_difference_is_six_l2933_293395

/-- Calculates the difference between morning and afternoon emails --/
def email_difference (early_morning late_morning early_afternoon late_afternoon : ℕ) : ℕ :=
  (early_morning + late_morning) - (early_afternoon + late_afternoon)

/-- Theorem stating the difference between morning and afternoon emails is 6 --/
theorem email_difference_is_six :
  email_difference 10 15 7 12 = 6 := by
  sorry

end email_difference_is_six_l2933_293395


namespace negation_existence_statement_l2933_293309

theorem negation_existence_statement (A : Set ℝ) :
  (¬ ∃ x ∈ A, x^2 - 2*x - 3 > 0) ↔ (∀ x ∈ A, x^2 - 2*x - 3 ≤ 0) :=
sorry

end negation_existence_statement_l2933_293309


namespace range_of_half_difference_l2933_293370

theorem range_of_half_difference (α β : ℝ) 
  (h1 : -π/2 ≤ α) (h2 : α < β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-π/2) 0 ↔ ∃ α β, -π/2 ≤ α ∧ α < β ∧ β ≤ π/2 ∧ x = (α - β)/2 :=
sorry

end range_of_half_difference_l2933_293370


namespace triangle_property_l2933_293335

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) - b*cos(A) = (3/5)*c, then tan(A)/tan(B) = 4 and max(tan(A-B)) = 3/4 -/
theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (a * Real.cos B - b * Real.cos A = (3/5) * c) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (Real.tan A / Real.tan B = 4) ∧
  (∀ x y : ℝ, Real.tan (A - B) ≤ (3/4)) ∧
  (∃ x y : ℝ, Real.tan (A - B) = (3/4)) :=
by sorry

end triangle_property_l2933_293335


namespace nut_mixture_price_l2933_293303

/-- Calculates the total selling price of a nut mixture -/
def total_selling_price (total_weight : ℝ) (cashew_weight : ℝ) (cashew_price : ℝ) (peanut_price : ℝ) : ℝ :=
  cashew_weight * cashew_price + (total_weight - cashew_weight) * peanut_price

/-- The total selling price of the nut mixture is $83.00 -/
theorem nut_mixture_price : total_selling_price 25 11 5 2 = 83 := by
  sorry

end nut_mixture_price_l2933_293303


namespace quadratic_equation_solution_l2933_293387

theorem quadratic_equation_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 9 / 2) : 
  q = 3 := by
sorry

end quadratic_equation_solution_l2933_293387


namespace weight_of_three_moles_l2933_293337

/-- Given a compound with molecular weight of 882 g/mol, 
    prove that 3 moles of this compound weigh 2646 grams. -/
theorem weight_of_three_moles (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 882 → moles = 3 → moles * molecular_weight = 2646 := by
  sorry

end weight_of_three_moles_l2933_293337


namespace parallel_vectors_x_value_l2933_293345

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (x, 1)
  are_parallel a b → x = -1/3 :=
by sorry

end parallel_vectors_x_value_l2933_293345


namespace no_perfect_squares_in_range_l2933_293304

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

def base_n_value (n : ℕ) : ℕ :=
  n^3 + 2*n^2 + 3*n + 4

theorem no_perfect_squares_in_range : 
  ¬ ∃ n : ℕ, 5 ≤ n ∧ n ≤ 20 ∧ is_perfect_square (base_n_value n) := by
  sorry

end no_perfect_squares_in_range_l2933_293304
