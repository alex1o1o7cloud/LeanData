import Mathlib

namespace NUMINAMATH_CALUDE_set_operations_l1063_106301

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | (x - 2) * (x + 5) < 0}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | -5 < x ∧ x ≤ -1}) ∧
  (A ∪ (U \ B) = {x | -5 < x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1063_106301


namespace NUMINAMATH_CALUDE_persimmon_basket_weight_l1063_106304

theorem persimmon_basket_weight (total_weight half_weight : ℝ)
  (h1 : total_weight = 62)
  (h2 : half_weight = 34)
  (h3 : ∃ (basket_weight persimmon_weight : ℝ),
    basket_weight + persimmon_weight = total_weight ∧
    basket_weight + persimmon_weight / 2 = half_weight) :
  ∃ (basket_weight : ℝ), basket_weight = 6 := by
sorry

end NUMINAMATH_CALUDE_persimmon_basket_weight_l1063_106304


namespace NUMINAMATH_CALUDE_dodgeball_assistant_count_l1063_106371

theorem dodgeball_assistant_count :
  ∀ (total_students : ℕ) (boys girls : ℕ),
    total_students = 27 →
    boys + girls < total_students →
    boys % 4 = 0 →
    girls % 6 = 0 →
    boys / 2 + girls / 3 = girls / 2 + boys / 4 →
    (total_students - (boys + girls) = 7) ∨
    (total_students - (boys + girls) = 17) :=
by sorry

end NUMINAMATH_CALUDE_dodgeball_assistant_count_l1063_106371


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l1063_106314

/-- 
Given two lines in the xy-plane:
- Line 1 with equation y = mx + 1
- Line 2 with equation y = 4x - 8
If Line 1 is perpendicular to Line 2, then m = -1/4
-/
theorem perpendicular_lines_slope (m : ℝ) : 
  (∃ (x y : ℝ), y = m * x + 1) →  -- Line 1 exists
  (∃ (x y : ℝ), y = 4 * x - 8) →  -- Line 2 exists
  (∀ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m * x₁ + 1 → y₂ = 4 * x₂ - 8 → 
    (y₂ - y₁) * (x₂ - x₁) = -(x₂ - x₁) * (x₂ - x₁)) →  -- Lines are perpendicular
  m = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l1063_106314


namespace NUMINAMATH_CALUDE_subtract_square_equals_two_square_l1063_106393

theorem subtract_square_equals_two_square (x : ℝ) : 3 * x^2 - x^2 = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_square_equals_two_square_l1063_106393


namespace NUMINAMATH_CALUDE_root_in_interval_l1063_106376

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  ∃ r ∈ Set.Ioo 2 2.5, f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1063_106376


namespace NUMINAMATH_CALUDE_four_Z_three_equals_127_l1063_106364

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^3 - 3*a*b^2 + 3*a^2*b + b^3

-- Theorem statement
theorem four_Z_three_equals_127 : Z 4 3 = 127 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_127_l1063_106364


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1063_106367

theorem geometric_series_sum : 
  let a : ℚ := 3 / 4
  let r : ℚ := 3 / 4
  let n : ℕ := 15
  let series_sum := (a * (1 - r^n)) / (1 - r)
  series_sum = 3216929751 / 1073741824 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1063_106367


namespace NUMINAMATH_CALUDE_polynomial_real_roots_l1063_106396

def polynomial (x : ℝ) : ℝ := x^9 - 37*x^8 - 2*x^7 + 74*x^6 + x^4 - 37*x^3 - 2*x^2 + 74*x

theorem polynomial_real_roots :
  ∃ (s : Finset ℝ), s.card = 5 ∧ (∀ x : ℝ, polynomial x = 0 ↔ x ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_roots_l1063_106396


namespace NUMINAMATH_CALUDE_tank_filling_solution_l1063_106337

/-- Represents the tank filling problem -/
def TankFillingProblem (tankCapacity : Real) (initialFillRatio : Real) 
  (fillingRate : Real) (drain1Rate : Real) (drain2Rate : Real) : Prop :=
  let remainingVolume := tankCapacity * (1 - initialFillRatio)
  let netFlowRate := fillingRate - drain1Rate - drain2Rate
  let timeToFill := remainingVolume / netFlowRate
  timeToFill = 6

/-- The theorem stating the solution to the tank filling problem -/
theorem tank_filling_solution :
  TankFillingProblem 1000 0.5 (1/2) (1/4) (1/6) := by
  sorry

#check tank_filling_solution

end NUMINAMATH_CALUDE_tank_filling_solution_l1063_106337


namespace NUMINAMATH_CALUDE_only_expr1_is_inequality_l1063_106383

-- Define the type for mathematical expressions
inductive MathExpression
  | LessThan : ℝ → ℝ → MathExpression
  | LinearExpr : ℝ → ℝ → MathExpression
  | Equation : ℝ → ℝ → ℝ → ℝ → MathExpression
  | Monomial : ℝ → ℕ → MathExpression

-- Define what it means for an expression to be an inequality
def isInequality : MathExpression → Prop
  | MathExpression.LessThan _ _ => True
  | _ => False

-- Define the given expressions
def expr1 : MathExpression := MathExpression.LessThan 0 19
def expr2 : MathExpression := MathExpression.LinearExpr 1 (-2)
def expr3 : MathExpression := MathExpression.Equation 2 3 (-1) 0
def expr4 : MathExpression := MathExpression.Monomial 1 2

-- Theorem statement
theorem only_expr1_is_inequality :
  isInequality expr1 ∧
  ¬isInequality expr2 ∧
  ¬isInequality expr3 ∧
  ¬isInequality expr4 :=
by sorry

end NUMINAMATH_CALUDE_only_expr1_is_inequality_l1063_106383


namespace NUMINAMATH_CALUDE_five_student_committee_from_eight_l1063_106394

theorem five_student_committee_from_eight (n k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committee_from_eight_l1063_106394


namespace NUMINAMATH_CALUDE_red_white_red_probability_l1063_106310

/-- The probability of drawing a red marble, then a white marble, and finally a red marble
    from a bag containing 4 red marbles and 6 white marbles, without replacement. -/
theorem red_white_red_probability :
  let total_marbles : ℕ := 10
  let red_marbles : ℕ := 4
  let white_marbles : ℕ := 6
  let prob_first_red : ℚ := red_marbles / total_marbles
  let prob_second_white : ℚ := white_marbles / (total_marbles - 1)
  let prob_third_red : ℚ := (red_marbles - 1) / (total_marbles - 2)
  prob_first_red * prob_second_white * prob_third_red = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_red_white_red_probability_l1063_106310


namespace NUMINAMATH_CALUDE_shortest_path_length_on_tetrahedron_l1063_106327

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ

/-- A path on the surface of a regular tetrahedron -/
structure SurfacePath (t : RegularTetrahedron) where
  length : ℝ
  start_vertex : Fin 4
  end_midpoint : Fin 6

/-- The shortest path on the surface of a regular tetrahedron -/
def shortest_path (t : RegularTetrahedron) : SurfacePath t :=
  sorry

theorem shortest_path_length_on_tetrahedron :
  let t : RegularTetrahedron := ⟨2⟩
  (shortest_path t).length = 3 := by sorry

end NUMINAMATH_CALUDE_shortest_path_length_on_tetrahedron_l1063_106327


namespace NUMINAMATH_CALUDE_solution_to_equation_l1063_106344

theorem solution_to_equation : ∃ x : ℕ, 
  (x = 10^2023 - 1) ∧ 
  (567 * x^3 + 171 * x^2 + 15 * x - (3 * x + 5 * x * 10^2023 + 7 * x * 10^(2*2023)) = 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1063_106344


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l1063_106320

theorem opposite_of_negative_three : -((-3) : ℤ) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l1063_106320


namespace NUMINAMATH_CALUDE_planting_schemes_count_l1063_106381

/-- The number of seed types available -/
def total_seeds : ℕ := 5

/-- The number of plots to be planted -/
def plots : ℕ := 4

/-- The number of choices for the first plot -/
def first_plot_choices : ℕ := 2

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

/-- The main theorem stating the total number of planting schemes -/
theorem planting_schemes_count : 
  first_plot_choices * permutations (total_seeds - 1) (plots - 1) = 48 := by
  sorry

end NUMINAMATH_CALUDE_planting_schemes_count_l1063_106381


namespace NUMINAMATH_CALUDE_theater_queue_arrangements_l1063_106343

theorem theater_queue_arrangements :
  let total_people : ℕ := 7
  let pair_size : ℕ := 2
  let units : ℕ := total_people - pair_size + 1
  units.factorial * pair_size.factorial = 1440 :=
by sorry

end NUMINAMATH_CALUDE_theater_queue_arrangements_l1063_106343


namespace NUMINAMATH_CALUDE_solution_set_f_min_value_sum_equality_condition_l1063_106366

-- Define the function f
def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

-- Theorem 1: Solution set of f(x + 3/2) ≥ 0
theorem solution_set_f (x : ℝ) : 
  f (x + 3/2) ≥ 0 ↔ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ) := by sorry

-- Theorem 2: Minimum value of 3p + 2q + r
theorem min_value_sum (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r ≥ 9/4 := by sorry

-- Theorem 3: Condition for equality in Theorem 2
theorem equality_condition (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h : 1/(3*p) + 1/(2*q) + 1/r = 4) : 
  3*p + 2*q + r = 9/4 ↔ p = 1/4 ∧ q = 3/8 ∧ r = 3/4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_min_value_sum_equality_condition_l1063_106366


namespace NUMINAMATH_CALUDE_speed_ratio_is_two_to_one_l1063_106360

-- Define the given constants
def distance : ℝ := 30
def original_speed : ℝ := 5

-- Define Sameer's speed as a variable
variable (sameer_speed : ℝ)

-- Define Abhay's new speed as a variable
variable (new_speed : ℝ)

-- Define the conditions
def condition1 (sameer_speed : ℝ) : Prop :=
  distance / original_speed = distance / sameer_speed + 2

def condition2 (sameer_speed new_speed : ℝ) : Prop :=
  distance / new_speed = distance / sameer_speed - 1

-- Theorem to prove
theorem speed_ratio_is_two_to_one 
  (h1 : condition1 sameer_speed)
  (h2 : condition2 sameer_speed new_speed) :
  new_speed / original_speed = 2 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_is_two_to_one_l1063_106360


namespace NUMINAMATH_CALUDE_pearl_division_l1063_106329

theorem pearl_division (n : ℕ) : 
  (n > 0) →
  (n % 8 = 6) → 
  (n % 7 = 5) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 6 ∨ m % 7 ≠ 5)) →
  (n % 9 = 1) := by
sorry

end NUMINAMATH_CALUDE_pearl_division_l1063_106329


namespace NUMINAMATH_CALUDE_sum_of_products_l1063_106354

theorem sum_of_products (x y z : ℝ) 
  (sum_condition : x + y + z = 20) 
  (sum_squares_condition : x^2 + y^2 + z^2 = 200) : 
  x*y + x*z + y*z = 100 := by sorry

end NUMINAMATH_CALUDE_sum_of_products_l1063_106354


namespace NUMINAMATH_CALUDE_fraction_simplification_l1063_106387

theorem fraction_simplification 
  (x y z u : ℝ) 
  (h1 : x + z ≠ 0) 
  (h2 : y + u ≠ 0) : 
  (x * y^2 + 2 * y * z^2 + y * z * u + 2 * x * y * z + 2 * x * z * u + y^2 * z + 2 * z^2 * u + x * y * u) / 
  (x * u^2 + y * z^2 + y * z * u + x * u * z + x * y * u + u * z^2 + z * u^2 + x * y * z) = 
  (y + 2 * z) / (u + z) := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1063_106387


namespace NUMINAMATH_CALUDE_park_to_restaurant_time_l1063_106347

def park_to_hidden_lake : ℕ := 15
def hidden_lake_to_park : ℕ := 7
def total_time : ℕ := 32

theorem park_to_restaurant_time : 
  total_time - (park_to_hidden_lake + hidden_lake_to_park) = 10 := by
sorry

end NUMINAMATH_CALUDE_park_to_restaurant_time_l1063_106347


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1063_106399

/-- The x-intercept of the line 2x + 3y + 6 = 0 is -3 -/
theorem x_intercept_of_line (x y : ℝ) :
  2 * x + 3 * y + 6 = 0 → y = 0 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1063_106399


namespace NUMINAMATH_CALUDE_video_cassette_cost_l1063_106312

theorem video_cassette_cost (audio_cost video_cost : ℕ) : 
  (7 * audio_cost + 3 * video_cost = 1110) →
  (5 * audio_cost + 4 * video_cost = 1350) →
  video_cost = 300 := by
sorry

end NUMINAMATH_CALUDE_video_cassette_cost_l1063_106312


namespace NUMINAMATH_CALUDE_value_of_expression_l1063_106361

theorem value_of_expression (x : ℝ) (h : 10000 * x + 2 = 4) : 5000 * x + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1063_106361


namespace NUMINAMATH_CALUDE_min_value_xyz_l1063_106311

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  ∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 27 → x + 3 * y + 6 * z ≤ a + 3 * b + 6 * c ∧
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y * z = 27 ∧ x + 3 * y + 6 * z = 27 :=
sorry

end NUMINAMATH_CALUDE_min_value_xyz_l1063_106311


namespace NUMINAMATH_CALUDE_third_level_lamps_l1063_106363

/-- Represents a pagoda with a given number of stories and lamps -/
structure Pagoda where
  stories : ℕ
  total_lamps : ℕ

/-- Calculates the number of lamps on a specific level of the pagoda -/
def lamps_on_level (p : Pagoda) (level : ℕ) : ℕ :=
  let first_level := p.total_lamps * (1 - 1 / 2^p.stories) / (2^p.stories - 1)
  first_level / 2^(level - 1)

theorem third_level_lamps (p : Pagoda) (h1 : p.stories = 7) (h2 : p.total_lamps = 381) :
  lamps_on_level p 5 = 12 := by
  sorry

#eval lamps_on_level ⟨7, 381⟩ 5

end NUMINAMATH_CALUDE_third_level_lamps_l1063_106363


namespace NUMINAMATH_CALUDE_tangent_slope_and_extrema_l1063_106391

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem tangent_slope_and_extrema (a : ℝ) :
  (deriv (f a) 0 = 1) →
  (a = 1) ∧
  (∀ x ∈ Set.Icc 0 2, f 1 0 ≤ f 1 x) ∧
  (∀ x ∈ Set.Icc 0 2, f 1 x ≤ f 1 2) ∧
  (f 1 0 = 0) ∧
  (f 1 2 = 2 * Real.exp 2) :=
by sorry

end

end NUMINAMATH_CALUDE_tangent_slope_and_extrema_l1063_106391


namespace NUMINAMATH_CALUDE_xiaopang_birthday_is_26th_l1063_106397

/-- Represents a day in May -/
def MayDay := Fin 31

/-- Xiaopang's birthday -/
def xiaopang_birthday : MayDay := sorry

/-- Xiaoya's birthday -/
def xiaoya_birthday : MayDay := sorry

/-- Days of the week, represented as integers mod 7 -/
def DayOfWeek := Fin 7

/-- Function to determine the day of the week for a given day in May -/
def day_of_week (d : MayDay) : DayOfWeek := sorry

/-- Wednesday, represented as a specific day of the week -/
def wednesday : DayOfWeek := sorry

theorem xiaopang_birthday_is_26th :
  -- Both birthdays are in May (implied by their types)
  -- Both birthdays fall on a Wednesday
  day_of_week xiaopang_birthday = wednesday ∧
  day_of_week xiaoya_birthday = wednesday ∧
  -- Xiaopang's birthday is later than Xiaoya's
  xiaopang_birthday.val > xiaoya_birthday.val ∧
  -- The sum of their birth dates is 38
  xiaopang_birthday.val + xiaoya_birthday.val = 38 →
  -- Conclusion: Xiaopang's birthday is on the 26th
  xiaopang_birthday.val = 26 := by
  sorry

end NUMINAMATH_CALUDE_xiaopang_birthday_is_26th_l1063_106397


namespace NUMINAMATH_CALUDE_goods_lost_percentage_l1063_106355

-- Define the profit percentage
def profit_percentage : ℝ := 0.10

-- Define the loss percentage on selling price
def loss_percentage_on_selling_price : ℝ := 0.12

-- Theorem to prove
theorem goods_lost_percentage (original_value : ℝ) (original_value_positive : original_value > 0) :
  let selling_price := original_value * (1 + profit_percentage)
  let loss_value := selling_price * loss_percentage_on_selling_price
  let goods_lost_percentage := (loss_value / original_value) * 100
  goods_lost_percentage = 13.2 := by
  sorry

end NUMINAMATH_CALUDE_goods_lost_percentage_l1063_106355


namespace NUMINAMATH_CALUDE_max_spend_amount_l1063_106316

/-- Represents the number of coins of each denomination a person has --/
structure CoinCount where
  coin100 : Nat
  coin50  : Nat
  coin10  : Nat

/-- Calculates the total value in won from a given CoinCount --/
def totalValue (coins : CoinCount) : Nat :=
  100 * coins.coin100 + 50 * coins.coin50 + 10 * coins.coin10

/-- Jimin's coin count --/
def jiminCoins : CoinCount := { coin100 := 5, coin50 := 1, coin10 := 0 }

/-- Seok-jin's coin count --/
def seokJinCoins : CoinCount := { coin100 := 2, coin50 := 0, coin10 := 7 }

/-- The theorem stating the maximum amount Jimin and Seok-jin can spend together --/
theorem max_spend_amount :
  totalValue jiminCoins + totalValue seokJinCoins = 820 := by sorry

end NUMINAMATH_CALUDE_max_spend_amount_l1063_106316


namespace NUMINAMATH_CALUDE_cabbage_production_increase_l1063_106331

theorem cabbage_production_increase (garden_size : ℕ) (this_year_production : ℕ) : 
  garden_size * garden_size = this_year_production →
  this_year_production = 9409 →
  this_year_production - (garden_size - 1) * (garden_size - 1) = 193 := by
sorry

end NUMINAMATH_CALUDE_cabbage_production_increase_l1063_106331


namespace NUMINAMATH_CALUDE_special_square_area_l1063_106353

/-- Square ABCD with points E on AD and F on BC, where BE = EF = FD = 20,
    AE = 2 * ED, and BF = 2 * FC -/
structure SpecialSquare where
  -- Define the side length of the square
  side : ℝ
  -- Define points E and F
  e : ℝ -- distance AE
  f : ℝ -- distance BF
  -- Conditions
  e_on_side : 0 < e ∧ e < side
  f_on_side : 0 < f ∧ f < side
  be_ef_fd : side - f + e = 20 -- BE + EF = 20
  ef_fd : e + side - f = 40 -- EF + FD = 40
  ae_twice_ed : e = 2 * (side - e)
  bf_twice_fc : f = 2 * (side - f)

/-- The area of the SpecialSquare is 720 -/
theorem special_square_area (sq : SpecialSquare) : sq.side ^ 2 = 720 :=
  sorry

end NUMINAMATH_CALUDE_special_square_area_l1063_106353


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1063_106300

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1063_106300


namespace NUMINAMATH_CALUDE_percentage_problem_l1063_106359

theorem percentage_problem (P : ℝ) (h : (P / 4) * 2 = 0.02) : P = 4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1063_106359


namespace NUMINAMATH_CALUDE_consecutive_product_l1063_106330

theorem consecutive_product (t : ℤ) :
  let n : ℤ := t * (t + 1) - 1
  (n^2 - 1 : ℤ) = (t - 1) * t * (t + 1) * (t + 2) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_l1063_106330


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l1063_106372

/-- The inradius of a right triangle with side lengths 9, 12, and 15 is 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →
  a^2 + b^2 = c^2 →
  (a + b + c) / 2 * r = (a * b) / 2 →
  r = 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l1063_106372


namespace NUMINAMATH_CALUDE_fruit_mix_problem_l1063_106370

theorem fruit_mix_problem (total : ℕ) (apples oranges bananas plums : ℕ) : 
  total = 240 →
  oranges = 3 * apples →
  bananas = 2 * oranges →
  plums = 5 * bananas →
  total = apples + oranges + bananas + plums →
  apples = 6 := by
sorry

end NUMINAMATH_CALUDE_fruit_mix_problem_l1063_106370


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l1063_106380

theorem complex_fraction_equals_i (i : ℂ) (hi : i^2 = -1) :
  (2 + i) / (1 - 2*i) = i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l1063_106380


namespace NUMINAMATH_CALUDE_equation_system_has_real_solution_l1063_106323

theorem equation_system_has_real_solution (x y : ℝ) 
  (h1 : 1 ≤ Real.sqrt x) (h2 : Real.sqrt x ≤ y) (h3 : y ≤ x^2) :
  ∃ (a b c : ℝ),
    a + b + c = (x + x^2 + x^4 + y + y^2 + y^4) / 2 ∧
    a * b + a * c + b * c = (x^3 + x^5 + x^6 + y^3 + y^5 + y^6) / 2 ∧
    a * b * c = (x^7 + y^7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_system_has_real_solution_l1063_106323


namespace NUMINAMATH_CALUDE_final_sum_after_operations_l1063_106365

theorem final_sum_after_operations (x y S : ℝ) (h : x + y = S) :
  3 * ((x + 5) + (y + 5)) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_operations_l1063_106365


namespace NUMINAMATH_CALUDE_scale_division_l1063_106350

/-- Proves that dividing a scale of 80 inches into 5 equal parts results in parts of 16 inches each -/
theorem scale_division (scale_length : ℕ) (num_parts : ℕ) (part_length : ℕ) :
  scale_length = 80 ∧ num_parts = 5 → part_length = scale_length / num_parts → part_length = 16 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l1063_106350


namespace NUMINAMATH_CALUDE_jean_initial_candy_l1063_106339

/-- The number of candy pieces Jean had initially -/
def initial_candy : ℕ := sorry

/-- The number of candy pieces Jean gave to her first friend -/
def first_friend : ℕ := 18

/-- The number of candy pieces Jean gave to her second friend -/
def second_friend : ℕ := 12

/-- The number of candy pieces Jean gave to her third friend -/
def third_friend : ℕ := 25

/-- The number of candy pieces Jean bought -/
def bought : ℕ := 10

/-- The number of candy pieces Jean ate -/
def ate : ℕ := 7

/-- The number of candy pieces Jean has left -/
def left : ℕ := 16

theorem jean_initial_candy : 
  initial_candy = first_friend + second_friend + third_friend + left + ate - bought :=
by sorry

end NUMINAMATH_CALUDE_jean_initial_candy_l1063_106339


namespace NUMINAMATH_CALUDE_reciprocal_statements_l1063_106388

def reciprocal (n : ℕ+) : ℚ := 1 / n.val

theorem reciprocal_statements : 
  (¬(reciprocal 4 + reciprocal 8 = reciprocal 12)) ∧
  (¬(reciprocal 9 - reciprocal 3 = reciprocal 6)) ∧
  (reciprocal 3 * reciprocal 9 = reciprocal 27) ∧
  ((reciprocal 15) / (reciprocal 3) = reciprocal 5) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_statements_l1063_106388


namespace NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l1063_106389

theorem exponential_function_sum_of_extrema (a : ℝ) (h : a > 0) : 
  (∀ x ∈ Set.Icc 0 1, ∃ y, a^x = y) → 
  (a^0 + a^1 = 4/3) → 
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_sum_of_extrema_l1063_106389


namespace NUMINAMATH_CALUDE_two_digit_number_subtraction_l1063_106342

theorem two_digit_number_subtraction (b : ℕ) (h1 : b < 9) : 
  (11 * b + 10) - (11 * b + 1) = 9 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_subtraction_l1063_106342


namespace NUMINAMATH_CALUDE_lines_coplanar_conditions_l1063_106373

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a placeholder and should be replaced with actual representation
  dummy : Unit

-- Define what it means for three lines to be coplanar
def coplanar (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- Define the condition of three lines intersecting pairwise but not sharing a common point
def intersect_pairwise_no_common_point (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- Define the condition of two lines being parallel and the third intersecting both
def two_parallel_one_intersecting (l1 l2 l3 : Line3D) : Prop :=
  sorry -- Add the actual definition here

-- The main theorem
theorem lines_coplanar_conditions (l1 l2 l3 : Line3D) :
  (intersect_pairwise_no_common_point l1 l2 l3 ∨ two_parallel_one_intersecting l1 l2 l3) →
  coplanar l1 l2 l3 := by
  sorry


end NUMINAMATH_CALUDE_lines_coplanar_conditions_l1063_106373


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_1073_l1063_106317

theorem max_q_minus_r_for_1073 :
  ∃ (q r : ℕ+), 1073 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1073 = 23 * q' + r' → q - r ≥ q' - r' :=
by
  sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_1073_l1063_106317


namespace NUMINAMATH_CALUDE_running_speed_calculation_l1063_106346

/-- Proves that given the conditions, the running speed must be 8 km/hr -/
theorem running_speed_calculation (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) : 
  walking_speed = 4 →
  total_distance = 16 →
  total_time = 3 →
  (total_distance / 2) / walking_speed + (total_distance / 2) / 8 = total_time :=
by
  sorry

#check running_speed_calculation

end NUMINAMATH_CALUDE_running_speed_calculation_l1063_106346


namespace NUMINAMATH_CALUDE_paint_cost_contribution_l1063_106319

-- Define the given conditions
def wall_area : ℝ := 1600
def paint_coverage : ℝ := 400
def paint_cost_per_gallon : ℝ := 45
def number_of_coats : ℕ := 2

-- Define the theorem
theorem paint_cost_contribution :
  let total_gallons := (wall_area / paint_coverage) * number_of_coats
  let total_cost := total_gallons * paint_cost_per_gallon
  let individual_contribution := total_cost / 2
  individual_contribution = 180 := by sorry

end NUMINAMATH_CALUDE_paint_cost_contribution_l1063_106319


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1063_106315

theorem complex_equation_solution (a : ℝ) : (Complex.I + a) * (1 - a * Complex.I) = 2 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1063_106315


namespace NUMINAMATH_CALUDE_line_vector_at_negative_two_l1063_106382

-- Define the line parameterization
def line_param (t : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem line_vector_at_negative_two :
  (∃ line_param : ℝ → ℝ × ℝ,
    (line_param 1 = (2, 5)) ∧
    (line_param 4 = (5, -7))) →
  (∃ line_param : ℝ → ℝ × ℝ,
    (line_param 1 = (2, 5)) ∧
    (line_param 4 = (5, -7)) ∧
    (line_param (-2) = (-1, 17))) :=
by sorry

end NUMINAMATH_CALUDE_line_vector_at_negative_two_l1063_106382


namespace NUMINAMATH_CALUDE_unique_pairs_from_ten_l1063_106386

theorem unique_pairs_from_ten (n : ℕ) (h : n = 10) : n * (n - 1) / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_unique_pairs_from_ten_l1063_106386


namespace NUMINAMATH_CALUDE_root_sum_squares_l1063_106348

theorem root_sum_squares (a : ℝ) : 
  (∃ x y : ℝ, x^2 - 3*a*x + a^2 = 0 ∧ y^2 - 3*a*y + a^2 = 0 ∧ x^2 + y^2 = 1.75) → 
  a = 0.5 ∨ a = -0.5 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l1063_106348


namespace NUMINAMATH_CALUDE_rope_cutting_l1063_106332

theorem rope_cutting (total_length : ℕ) (equal_pieces : ℕ) (equal_piece_length : ℕ) (remaining_piece_length : ℕ) : 
  total_length = 1165 ∧ 
  equal_pieces = 150 ∧ 
  equal_piece_length = 75 ∧ 
  remaining_piece_length = 100 → 
  (total_length * 10 - equal_pieces * equal_piece_length) / remaining_piece_length + equal_pieces = 154 :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_l1063_106332


namespace NUMINAMATH_CALUDE_machine_work_time_solution_l1063_106324

theorem machine_work_time_solution : ∃ x : ℝ, 
  (x > 0) ∧ 
  (1 / (x + 4) + 1 / (x + 2) + 1 / (2 * x + 6) = 1 / x) ∧ 
  (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_machine_work_time_solution_l1063_106324


namespace NUMINAMATH_CALUDE_nail_trimming_customers_l1063_106334

/-- The number of nails per person -/
def nails_per_person : ℕ := 20

/-- The total number of sounds produced by the nail cutter -/
def total_sounds : ℕ := 100

/-- The number of customers whose nails were trimmed -/
def num_customers : ℕ := total_sounds / nails_per_person

theorem nail_trimming_customers :
  num_customers = 5 :=
sorry

end NUMINAMATH_CALUDE_nail_trimming_customers_l1063_106334


namespace NUMINAMATH_CALUDE_quadratic_function_value_l1063_106306

/-- A quadratic function with specific properties -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := fun x ↦ d * x^2 + e * x + f

theorem quadratic_function_value (d e f : ℝ) :
  (∀ x, QuadraticFunction d e f x = d * x^2 + e * x + f) →
  QuadraticFunction d e f 0 = 2 →
  (∀ x, QuadraticFunction d e f (3.5 + x) = QuadraticFunction d e f (3.5 - x)) →
  ∃ n : ℤ, QuadraticFunction d e f 10 = n →
  QuadraticFunction d e f 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l1063_106306


namespace NUMINAMATH_CALUDE_cos_seven_arccos_two_fifths_l1063_106374

theorem cos_seven_arccos_two_fifths (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, abs (Real.cos (7 * Real.arccos (2/5)) - x) < ε ∧ abs (x + 0.2586) < ε :=
sorry

end NUMINAMATH_CALUDE_cos_seven_arccos_two_fifths_l1063_106374


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1063_106356

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 - b * x - 1

-- Define the solution set of the first inequality
def solution_set (a b : ℝ) := {x : ℝ | f a b x ≥ 0}

-- State the theorem
theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h1 : solution_set a b = Set.Icc (-1/2) (-1/3)) :
  {x : ℝ | x^2 - b*x - a < 0} = Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1063_106356


namespace NUMINAMATH_CALUDE_circle_properties_l1063_106377

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 6*y - 3 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (-2, 3)

/-- The radius of the circle -/
def CircleRadius : ℝ := 4

/-- Theorem: The given equation represents a circle with center (-2, 3) and radius 4 -/
theorem circle_properties :
  ∀ (x y : ℝ),
    CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = CircleRadius^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1063_106377


namespace NUMINAMATH_CALUDE_smallest_positive_integer_2016m_43200n_l1063_106375

theorem smallest_positive_integer_2016m_43200n :
  ∃ (k : ℕ+), (∀ (a : ℕ+), (∃ (m n : ℤ), a = 2016 * m + 43200 * n) → k ≤ a) ∧
  (∃ (m n : ℤ), (k : ℕ) = 2016 * m + 43200 * n) ∧
  k = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_2016m_43200n_l1063_106375


namespace NUMINAMATH_CALUDE_marble_problem_l1063_106362

theorem marble_problem (a : ℚ) : 
  let brian := 3 * a - 4
  let caden := 2 * brian + 2
  let daryl := 4 * caden
  a + brian + caden + daryl = 122 → a = 78 / 17 := by
sorry

end NUMINAMATH_CALUDE_marble_problem_l1063_106362


namespace NUMINAMATH_CALUDE_truncated_cone_radius_l1063_106349

/-- Given three cones touching each other with base radii 6, 24, and 24,
    and a truncated cone sharing a common generator with each,
    the radius of the smaller base of the truncated cone is 2. -/
theorem truncated_cone_radius (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 24) (h₃ : r₃ = 24) :
  ∃ (r : ℝ), r = 2 ∧ 
  (r = r₂ - 24) ∧ 
  (r = r₃ - 24) ∧
  ((24 + r)^2 = 24^2 + (12 - r)^2) :=
by sorry

end NUMINAMATH_CALUDE_truncated_cone_radius_l1063_106349


namespace NUMINAMATH_CALUDE_julian_needs_80_more_legos_l1063_106335

/-- The number of additional legos Julian needs to complete two identical airplane models -/
def additional_legos_needed (total_legos : ℕ) (legos_per_model : ℕ) (num_models : ℕ) : ℕ :=
  max 0 (legos_per_model * num_models - total_legos)

/-- Proof that Julian needs 80 more legos -/
theorem julian_needs_80_more_legos :
  additional_legos_needed 400 240 2 = 80 := by
  sorry

#eval additional_legos_needed 400 240 2

end NUMINAMATH_CALUDE_julian_needs_80_more_legos_l1063_106335


namespace NUMINAMATH_CALUDE_alex_sweaters_l1063_106303

/-- The number of shirts Alex has to wash -/
def num_shirts : ℕ := 18

/-- The number of pants Alex has to wash -/
def num_pants : ℕ := 12

/-- The number of jeans Alex has to wash -/
def num_jeans : ℕ := 13

/-- The maximum number of items the washing machine can wash per cycle -/
def items_per_cycle : ℕ := 15

/-- The duration of each washing cycle in minutes -/
def cycle_duration : ℕ := 45

/-- The total time needed to wash all clothes in minutes -/
def total_wash_time : ℕ := 180

/-- The theorem stating that Alex has 17 sweaters to wash -/
theorem alex_sweaters : 
  ∃ (num_sweaters : ℕ), 
    (num_shirts + num_pants + num_jeans + num_sweaters) = 
    (total_wash_time / cycle_duration * items_per_cycle) ∧ 
    num_sweaters = 17 := by
  sorry

end NUMINAMATH_CALUDE_alex_sweaters_l1063_106303


namespace NUMINAMATH_CALUDE_base_conversion_sum_l1063_106305

-- Define a function to convert a number from base 8 to base 10
def base8To10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

-- Define a function to convert a number from base 13 to base 10
def base13To10 (n : Nat) (c : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds * 13^2 + c * 13^1 + ones * 13^0

theorem base_conversion_sum :
  base8To10 537 + base13To10 405 12 = 1188 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l1063_106305


namespace NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l1063_106352

/-- Given an isosceles triangle with base 2s and height h, and a rectangle with side length s,
    if their areas are equal, then the height of the triangle equals the side length of the rectangle. -/
theorem isosceles_triangle_rectangle_equal_area
  (s h : ℝ) -- s: side length of rectangle, h: height of triangle
  (h_positive : s > 0) -- Ensure s is positive
  (area_equal : s * h = s^2) -- Areas are equal
  : h = s := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_rectangle_equal_area_l1063_106352


namespace NUMINAMATH_CALUDE_juice_cost_calculation_l1063_106318

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℚ := 0.30

/-- The total amount Lyle has in dollars -/
def total_amount : ℚ := 2.50

/-- The number of friends Lyle is buying for -/
def num_friends : ℕ := 4

/-- The cost of a pack of juice in dollars -/
def juice_cost : ℚ := 0.325

theorem juice_cost_calculation : 
  sandwich_cost * num_friends + juice_cost * num_friends = total_amount :=
by sorry

end NUMINAMATH_CALUDE_juice_cost_calculation_l1063_106318


namespace NUMINAMATH_CALUDE_skating_minutes_proof_l1063_106351

/-- The number of minutes Gage skated per day for the first 4 days -/
def minutes_per_day_first_4 : ℕ := 70

/-- The number of minutes Gage skated per day for the next 4 days -/
def minutes_per_day_next_4 : ℕ := 100

/-- The total number of days Gage skated -/
def total_days : ℕ := 9

/-- The desired average number of minutes skated per day -/
def desired_average : ℕ := 100

/-- The number of minutes Gage must skate on the ninth day to achieve the desired average -/
def minutes_on_ninth_day : ℕ := 220

theorem skating_minutes_proof :
  minutes_on_ninth_day = 
    total_days * desired_average - 
    (4 * minutes_per_day_first_4 + 4 * minutes_per_day_next_4) := by
  sorry

end NUMINAMATH_CALUDE_skating_minutes_proof_l1063_106351


namespace NUMINAMATH_CALUDE_ratio_problem_l1063_106390

theorem ratio_problem (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : 
  x / y = 11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1063_106390


namespace NUMINAMATH_CALUDE_optimal_swap_theorem_l1063_106307

/-- The distance at which car tires should be swapped to wear out equally -/
def optimal_swap_distance : ℝ := 9375

/-- The total distance a front tire can travel before wearing out -/
def front_tire_lifespan : ℝ := 25000

/-- The total distance a rear tire can travel before wearing out -/
def rear_tire_lifespan : ℝ := 15000

/-- Theorem stating that swapping tires at the optimal distance results in equal wear -/
theorem optimal_swap_theorem :
  let remaining_front := (3/5) * (front_tire_lifespan - optimal_swap_distance)
  let remaining_rear := (5/3) * (rear_tire_lifespan - optimal_swap_distance)
  remaining_front = remaining_rear := by sorry

end NUMINAMATH_CALUDE_optimal_swap_theorem_l1063_106307


namespace NUMINAMATH_CALUDE_prism_diagonals_l1063_106395

/-- Checks if three numbers can be the lengths of external diagonals of a right regular prism -/
def valid_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  b^2 + c^2 > a^2 ∧
  a^2 + c^2 > b^2

theorem prism_diagonals :
  ¬(valid_diagonals 6 8 11) ∧
  (valid_diagonals 6 8 10) ∧
  (valid_diagonals 6 10 11) ∧
  (valid_diagonals 8 10 11) ∧
  (valid_diagonals 8 11 12) :=
by sorry

end NUMINAMATH_CALUDE_prism_diagonals_l1063_106395


namespace NUMINAMATH_CALUDE_fraction_thousandths_digit_l1063_106325

def fraction : ℚ := 57 / 5000

/-- The thousandths digit of a rational number is the third digit after the decimal point in its decimal representation. -/
def thousandths_digit (q : ℚ) : ℕ :=
  sorry

theorem fraction_thousandths_digit :
  thousandths_digit fraction = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_thousandths_digit_l1063_106325


namespace NUMINAMATH_CALUDE_part_one_part_two_l1063_106384

-- Part I
theorem part_one (t : ℝ) (h1 : t^2 - 5*t + 4 < 0) (h2 : (t-2)*(t-6) < 0) : 
  2 < t ∧ t < 4 := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a ≠ 0) 
  (h_suff : ∀ t : ℝ, 2 < t ∧ t < 6 → t^2 - 5*a*t + 4*a^2 < 0) : 
  3/2 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1063_106384


namespace NUMINAMATH_CALUDE_sweet_potato_problem_l1063_106369

theorem sweet_potato_problem (total : ℕ) (sold_to_adams : ℕ) (sold_to_lenon : ℕ) 
  (h1 : total = 80) 
  (h2 : sold_to_adams = 20) 
  (h3 : sold_to_lenon = 15) : 
  total - (sold_to_adams + sold_to_lenon) = 45 := by
  sorry

end NUMINAMATH_CALUDE_sweet_potato_problem_l1063_106369


namespace NUMINAMATH_CALUDE_square_region_perimeter_l1063_106341

theorem square_region_perimeter (area : ℝ) (num_squares : ℕ) (perimeter : ℝ) :
  area = 392 →
  num_squares = 8 →
  (area / num_squares).sqrt * (2 * num_squares + 2) = perimeter →
  perimeter = 70 := by
  sorry

end NUMINAMATH_CALUDE_square_region_perimeter_l1063_106341


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1063_106336

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (2 - |x|) / (x + 2) = 0 → x = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l1063_106336


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l1063_106385

theorem gcf_lcm_sum_8_12 : 
  Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l1063_106385


namespace NUMINAMATH_CALUDE_netflix_series_episodes_l1063_106358

/-- A TV series with the given properties -/
structure TVSeries where
  seasons : ℕ
  episodes_per_day : ℕ
  days_to_complete : ℕ

/-- Calculate the number of episodes per season -/
def episodes_per_season (series : TVSeries) : ℕ :=
  (series.episodes_per_day * series.days_to_complete) / series.seasons

/-- Theorem stating that for the given TV series, each season has 20 episodes -/
theorem netflix_series_episodes (series : TVSeries) 
  (h1 : series.seasons = 3)
  (h2 : series.episodes_per_day = 2)
  (h3 : series.days_to_complete = 30) :
  episodes_per_season series = 20 := by
  sorry

#check netflix_series_episodes

end NUMINAMATH_CALUDE_netflix_series_episodes_l1063_106358


namespace NUMINAMATH_CALUDE_smallest_equivalent_angle_l1063_106333

theorem smallest_equivalent_angle (x : ℝ) (h : x = -11/4 * Real.pi) :
  ∃ (θ : ℝ) (k : ℤ),
    x = θ + 2 * ↑k * Real.pi ∧
    θ ∈ Set.Icc (-Real.pi) Real.pi ∧
    ∀ (φ : ℝ) (m : ℤ),
      x = φ + 2 * ↑m * Real.pi →
      φ ∈ Set.Icc (-Real.pi) Real.pi →
      |θ| ≤ |φ| ∧
    θ = -3/4 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_smallest_equivalent_angle_l1063_106333


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1063_106313

/-- Given two real numbers are inversely proportional, if one is 40 when the other is 5,
    then the first is 25 when the second is 8. -/
theorem inverse_proportion_problem (r s : ℝ) (h : ∃ k : ℝ, r * s = k) 
    (h1 : ∃ r0 : ℝ, r0 * 5 = 40 ∧ r0 * s = r * s) : 
    r * 8 = 25 * s := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1063_106313


namespace NUMINAMATH_CALUDE_triangle_theorem_l1063_106398

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sin t.A ^ 2 + Real.sin t.A * Real.sin t.B - 6 * Real.sin t.B ^ 2 = 0) :
  (t.a / t.b = 2) ∧ 
  (Real.cos t.C = 3/4 → Real.sin t.B = Real.sqrt 14 / 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l1063_106398


namespace NUMINAMATH_CALUDE_correct_operation_l1063_106340

theorem correct_operation (x : ℝ) : 4 * x^2 * (3 * x) = 12 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1063_106340


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1063_106322

theorem product_from_lcm_gcd (x y : ℕ+) 
  (h_lcm : Nat.lcm x y = 60)
  (h_gcd : Nat.gcd x y = 10) : 
  x * y = 600 := by
sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1063_106322


namespace NUMINAMATH_CALUDE_sqrt_14_between_3_and_4_l1063_106328

theorem sqrt_14_between_3_and_4 : 3 < Real.sqrt 14 ∧ Real.sqrt 14 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_14_between_3_and_4_l1063_106328


namespace NUMINAMATH_CALUDE_sally_quarters_now_l1063_106326

/-- The number of quarters Sally had initially -/
def initial_quarters : ℕ := 760

/-- The number of quarters Sally spent -/
def spent_quarters : ℕ := 418

/-- Theorem: Sally has 342 quarters now -/
theorem sally_quarters_now : initial_quarters - spent_quarters = 342 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_now_l1063_106326


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1063_106345

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1063_106345


namespace NUMINAMATH_CALUDE_m_range_l1063_106309

/-- Two points are on opposite sides of a line if the product of their signed distances from the line is negative -/
def opposite_sides (x₁ y₁ x₂ y₂ : ℝ) (a b c : ℝ) : Prop :=
  (a * x₁ + b * y₁ + c) * (a * x₂ + b * y₂ + c) < 0

/-- The theorem stating the range of m given the conditions -/
theorem m_range (m : ℝ) : 
  opposite_sides m 0 2 m 1 1 (-1) → -1 < m ∧ m < 1 := by
  sorry


end NUMINAMATH_CALUDE_m_range_l1063_106309


namespace NUMINAMATH_CALUDE_solve_equation_l1063_106392

theorem solve_equation : ∃ y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 6 * y) → y = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1063_106392


namespace NUMINAMATH_CALUDE_f_decreasing_intervals_l1063_106379

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem f_decreasing_intervals (ω : ℝ) (h_ω : ω > 0) 
  (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π + π / 6) (k * π + π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_intervals_l1063_106379


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l1063_106308

def A : ℕ := 123456
def B : ℕ := 171717
def M : ℕ := 1000003
def N : ℕ := 538447

theorem multiplicative_inverse_modulo :
  (A * B * N) % M = 1 := by sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l1063_106308


namespace NUMINAMATH_CALUDE_cubic_function_symmetry_l1063_106357

/-- Given a cubic function f(x) = ax³ + bx + 1 where ab ≠ 0,
    if f(2016) = k, then f(-2016) = 2 - k -/
theorem cubic_function_symmetry (a b k : ℝ) (h1 : a * b ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 1
  f 2016 = k → f (-2016) = 2 - k := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_symmetry_l1063_106357


namespace NUMINAMATH_CALUDE_probability_sum_15_l1063_106378

/-- The number of ways to roll a sum of 15 with five six-sided dice -/
def waysToRoll15 : ℕ := 95

/-- The total number of possible outcomes when rolling five six-sided dice -/
def totalOutcomes : ℕ := 6^5

/-- A fair, standard six-sided die -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces = {1, 2, 3, 4, 5, 6})

/-- The probability of rolling a sum of 15 with five fair, standard six-sided dice -/
theorem probability_sum_15 (d1 d2 d3 d4 d5 : Die) :
  (waysToRoll15 : ℚ) / totalOutcomes = 95 / 7776 := by
  sorry


end NUMINAMATH_CALUDE_probability_sum_15_l1063_106378


namespace NUMINAMATH_CALUDE_max_value_of_f_range_of_k_l1063_106368

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x - 9 / x

theorem max_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ f x = (1 : ℝ) / Real.exp 10 ∧ ∀ y > 0, f y ≤ f x :=
sorry

theorem range_of_k :
  ∀ k : ℝ,
  (∀ x : ℝ, x ≥ 1 → x^2 * ((Real.log x) / x - k / x) + 1 / (x + 1) ≥ 0) →
  (∀ x : ℝ, x ≥ 1 → k ≥ (1/2) * x^2 + (Real.exp 2 - 2) * x - Real.exp x - 7) →
  (Real.exp 2 - 9 ≤ k ∧ k ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_range_of_k_l1063_106368


namespace NUMINAMATH_CALUDE_potato_cooking_time_l1063_106338

/-- Given a chef cooking potatoes with the following conditions:
  - Total potatoes to cook is 15
  - Potatoes already cooked is 6
  - Time to cook the remaining potatoes is 72 minutes
  Prove that the time to cook one potato is 8 minutes. -/
theorem potato_cooking_time (total : Nat) (cooked : Nat) (remaining_time : Nat) :
  total = 15 → cooked = 6 → remaining_time = 72 → (remaining_time / (total - cooked) = 8) :=
by sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l1063_106338


namespace NUMINAMATH_CALUDE_adams_change_l1063_106302

def adams_money : ℝ := 5.00
def airplane_cost : ℝ := 4.28

theorem adams_change :
  adams_money - airplane_cost = 0.72 := by sorry

end NUMINAMATH_CALUDE_adams_change_l1063_106302


namespace NUMINAMATH_CALUDE_left_shoe_probability_l1063_106321

/-- The probability of randomly picking a left shoe from a shoe cabinet with 3 pairs of shoes is 1/2. -/
theorem left_shoe_probability (num_pairs : ℕ) (h : num_pairs = 3) :
  (num_pairs : ℚ) / (2 * num_pairs : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_left_shoe_probability_l1063_106321
