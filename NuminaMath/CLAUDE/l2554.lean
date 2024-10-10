import Mathlib

namespace largest_integer_with_conditions_l2554_255437

/-- A function that returns the digits of an integer -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if each digit is twice the previous one -/
def doubling_digits (l : List ℕ) : Prop := sorry

/-- A function that calculates the sum of squares of a list of digits -/
def sum_of_squares (l : List ℕ) : ℕ := sorry

/-- A function that calculates the product of a list of digits -/
def product_of_digits (l : List ℕ) : ℕ := sorry

theorem largest_integer_with_conditions (n : ℕ) :
  (∀ m : ℕ, m > n → ¬(sum_of_squares (digits m) = 65 ∧ doubling_digits (digits m))) →
  sum_of_squares (digits n) = 65 →
  doubling_digits (digits n) →
  product_of_digits (digits n) = 8 := by
  sorry

end largest_integer_with_conditions_l2554_255437


namespace problem_statement_l2554_255467

theorem problem_statement : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end problem_statement_l2554_255467


namespace cloth_sale_total_price_l2554_255442

-- Define the parameters of the problem
def quantity : ℕ := 80
def profit_per_meter : ℕ := 7
def cost_price_per_meter : ℕ := 118

-- Define the theorem
theorem cloth_sale_total_price :
  (quantity * (cost_price_per_meter + profit_per_meter)) = 10000 := by
  sorry

end cloth_sale_total_price_l2554_255442


namespace range_of_m_l2554_255402

def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0

theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 8 :=
by sorry

end range_of_m_l2554_255402


namespace inequality_for_real_numbers_l2554_255413

theorem inequality_for_real_numbers (a b : ℝ) : a * b ≤ ((a + b) / 2) ^ 2 := by
  sorry

end inequality_for_real_numbers_l2554_255413


namespace three_numbers_with_square_sums_l2554_255455

theorem three_numbers_with_square_sums : ∃ (a b c : ℕ+), 
  (∃ (x : ℕ), (a + b + c : ℕ) = x^2) ∧
  (∃ (y : ℕ), (a + b : ℕ) = y^2) ∧
  (∃ (z : ℕ), (b + c : ℕ) = z^2) ∧
  (∃ (w : ℕ), (a + c : ℕ) = w^2) ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by sorry

end three_numbers_with_square_sums_l2554_255455


namespace contrapositive_equivalence_l2554_255447

def divisible_by_2_or_5_ends_with_0 (n : ℕ) : Prop :=
  (n % 2 = 0 ∨ n % 5 = 0) → n % 10 = 0

def last_digit_not_0_not_divisible_by_2_and_5 (n : ℕ) : Prop :=
  n % 10 ≠ 0 → (n % 2 ≠ 0 ∧ n % 5 ≠ 0)

theorem contrapositive_equivalence :
  ∀ n : ℕ, divisible_by_2_or_5_ends_with_0 n ↔ last_digit_not_0_not_divisible_by_2_and_5 n :=
by
  sorry

end contrapositive_equivalence_l2554_255447


namespace sum_nth_from_both_ends_l2554_255441

/-- A set of consecutive integers -/
structure ConsecutiveIntegerSet where
  first : ℤ
  last : ℤ
  h_consecutive : last ≥ first

/-- The median of a set of consecutive integers -/
def median (s : ConsecutiveIntegerSet) : ℚ :=
  (s.first + s.last : ℚ) / 2

/-- The nth number from the beginning of the set -/
def nth_from_beginning (s : ConsecutiveIntegerSet) (n : ℕ) : ℤ :=
  s.first + n - 1

/-- The nth number from the end of the set -/
def nth_from_end (s : ConsecutiveIntegerSet) (n : ℕ) : ℤ :=
  s.last - n + 1

theorem sum_nth_from_both_ends (s : ConsecutiveIntegerSet) (n : ℕ) 
  (h_median : median s = 60) :
  nth_from_beginning s n + nth_from_end s n = 120 := by
  sorry

end sum_nth_from_both_ends_l2554_255441


namespace arithmetic_sequence_terms_l2554_255403

theorem arithmetic_sequence_terms (a₁ aₙ : ℤ) (d : ℤ) (n : ℕ) : 
  a₁ = 1 ∧ aₙ = -89 ∧ d = -2 ∧ aₙ = a₁ + (n - 1) * d → n = 46 := by
  sorry

end arithmetic_sequence_terms_l2554_255403


namespace parabola_properties_l2554_255411

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = x - 4

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through (1, 2)
  parabola 1 2 ∧
  -- If A and B are intersection points of the line and parabola
  ∀ (A B : ℝ × ℝ), 
    (parabola A.1 A.2 ∧ line A.1 A.2) →
    (parabola B.1 B.2 ∧ line B.1 B.2) →
    A ≠ B →
    -- Then OA is perpendicular to OB
    (A.1 * B.1 + A.2 * B.2 = 0) :=
by sorry

end parabola_properties_l2554_255411


namespace f_at_negative_two_l2554_255440

def f (x : ℝ) : ℝ := 2*x^5 + 5*x^4 + 5*x^3 + 10*x^2 + 6*x + 1

theorem f_at_negative_two : f (-2) = 5 := by
  sorry

end f_at_negative_two_l2554_255440


namespace parentheses_removal_l2554_255465

theorem parentheses_removal (a b : ℝ) : a + (5 * a - 3 * b) = 6 * a - 3 * b := by
  sorry

end parentheses_removal_l2554_255465


namespace anna_weekly_salary_l2554_255461

/-- Represents a worker's salary information -/
structure WorkerSalary where
  daysWorkedPerWeek : ℕ
  missedDays : ℕ
  deductionAmount : ℚ

/-- Calculates the usual weekly salary of a worker -/
def usualWeeklySalary (w : WorkerSalary) : ℚ :=
  (w.deductionAmount / w.missedDays) * w.daysWorkedPerWeek

theorem anna_weekly_salary :
  let anna : WorkerSalary := {
    daysWorkedPerWeek := 5,
    missedDays := 2,
    deductionAmount := 985
  }
  usualWeeklySalary anna = 2462.5 := by
  sorry

end anna_weekly_salary_l2554_255461


namespace min_third_altitude_l2554_255423

/-- Represents a scalene triangle with specific altitude properties -/
structure ScaleneTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Altitudes
  h_D : ℝ
  h_E : ℝ
  h_F : ℝ
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- Scalene property
  scalene : a ≠ b ∧ b ≠ c ∧ c ≠ a
  -- Given altitude values
  altitude_D : h_D = 18
  altitude_E : h_E = 8
  -- Relation between sides
  side_relation : b = 2 * a
  -- Area consistency
  area_consistency : a * h_D / 2 = b * h_E / 2

/-- The minimum possible integer length of the third altitude is 17 -/
theorem min_third_altitude (t : ScaleneTriangle) : 
  ∃ (n : ℕ), n ≥ 17 ∧ t.h_F = n ∧ ∀ (m : ℕ), m < 17 → t.h_F ≠ m :=
sorry

end min_third_altitude_l2554_255423


namespace problem_statement_l2554_255489

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a^2 + b^2 ≥ 1/2) ∧ (a*b ≤ 1/4) ∧ (1/a + 1/b > 4) ∧ (Real.sqrt a + Real.sqrt b ≤ Real.sqrt 2) :=
by sorry

end problem_statement_l2554_255489


namespace repeating_decimal_difference_l2554_255415

theorem repeating_decimal_difference : 
  (4 : ℚ) / 11 - (7 : ℚ) / 20 = (3 : ℚ) / 220 := by sorry

end repeating_decimal_difference_l2554_255415


namespace yellow_balls_count_l2554_255472

theorem yellow_balls_count (total : Nat) (white green red purple : Nat) (prob_not_red_purple : Real) :
  total = 60 →
  white = 22 →
  green = 18 →
  red = 15 →
  purple = 3 →
  prob_not_red_purple = 0.7 →
  ∃ yellow : Nat, yellow = 2 ∧ 
    total = white + green + yellow + red + purple ∧
    (white + green + yellow : Real) / total = prob_not_red_purple :=
by sorry

end yellow_balls_count_l2554_255472


namespace vector_projection_and_perpendicular_l2554_255454

/-- Given two vectors a and b in ℝ², and a scalar k, we define vector c and prove properties about their relationships. -/
theorem vector_projection_and_perpendicular (a b : ℝ × ℝ) (k : ℝ) :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (3, 1)
  let c : ℝ × ℝ := b - k • a
  (a.1 * c.1 + a.2 * c.2 = 0) →
  (let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2)
   proj = Real.sqrt 5 ∧ k = 1 ∧ c = (2, -1)) := by
  sorry

end vector_projection_and_perpendicular_l2554_255454


namespace bulb_switch_problem_l2554_255499

theorem bulb_switch_problem :
  let n : Nat := 11
  let target_state : Fin n → Bool := fun i => i.val + 1 == n
  let valid_state (state : Fin n → Bool) :=
    ∃ (k : Nat), k < 2^n ∧ state = fun i => (k.digits 2).get? i.val == some 1
  { count : Nat // ∀ state, valid_state state ∧ state = target_state → count = 2^(n-1) } :=
by
  sorry

#check bulb_switch_problem

end bulb_switch_problem_l2554_255499


namespace set_equality_l2554_255460

theorem set_equality (A B X : Set α) 
  (h1 : A ∩ X = B ∩ X)
  (h2 : A ∩ X = A ∩ B)
  (h3 : A ∪ B ∪ X = A ∪ B) : 
  X = A ∩ B := by
sorry

end set_equality_l2554_255460


namespace quadratic_two_distinct_roots_l2554_255450

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 2 = 0 ∧ y^2 - 2*y + m - 2 = 0) → m < 3 :=
by sorry

end quadratic_two_distinct_roots_l2554_255450


namespace product_of_cubic_fractions_l2554_255498

theorem product_of_cubic_fractions : 
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 168 := by
  sorry

end product_of_cubic_fractions_l2554_255498


namespace T_equals_five_l2554_255431

noncomputable def T : ℝ :=
  1 / (3 - Real.sqrt 8) - 1 / (Real.sqrt 8 - Real.sqrt 7) + 
  1 / (Real.sqrt 7 - Real.sqrt 6) - 1 / (Real.sqrt 6 - Real.sqrt 5) + 
  1 / (Real.sqrt 5 - 2)

theorem T_equals_five : T = 5 := by
  sorry

end T_equals_five_l2554_255431


namespace linear_regression_transformation_l2554_255422

-- Define the variables and functions
variable (a b x : ℝ)
variable (y : ℝ)
variable (μ : ℝ)
variable (c : ℝ)
variable (v : ℝ)

-- Define the conditions
def condition_y : Prop := y = a * Real.exp (b / x)
def condition_μ : Prop := μ = Real.log y
def condition_c : Prop := c = Real.log a
def condition_v : Prop := v = 1 / x

-- State the theorem
theorem linear_regression_transformation 
  (h1 : condition_y a b x y)
  (h2 : condition_μ y μ)
  (h3 : condition_c a c)
  (h4 : condition_v x v) :
  μ = c + b * v :=
by sorry

end linear_regression_transformation_l2554_255422


namespace chess_tournament_players_l2554_255404

/-- The number of chess players in the tournament. -/
def n : ℕ := 21

/-- The score of the winner. -/
def winner_score (n : ℕ) : ℚ := 3/4 * (n - 1)

/-- The total score of all games in the tournament. -/
def total_score (n : ℕ) : ℚ := 1/2 * n * (n - 1)

/-- The main theorem stating the conditions and the result of the chess tournament. -/
theorem chess_tournament_players :
  (∀ (m : ℕ), m > 1 →
    (winner_score m = 1/13 * (total_score m - winner_score m)) →
    m = n) ∧
  n > 1 := by sorry

end chess_tournament_players_l2554_255404


namespace cos_difference_formula_l2554_255449

theorem cos_difference_formula (α β : ℝ) 
  (h1 : Real.sin α + Real.sin β = 1/2) 
  (h2 : Real.cos α + Real.cos β = 1/3) : 
  Real.cos (α - β) = -59/72 := by
sorry

end cos_difference_formula_l2554_255449


namespace quadratic_root_difference_l2554_255476

theorem quadratic_root_difference (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0 ∧ (x₁ - x₂)^2 = 20) → 
  a = -4 := by
sorry

end quadratic_root_difference_l2554_255476


namespace slope_of_line_l2554_255496

/-- The slope of a line given by the equation x/4 + y/3 = 1 is -3/4 -/
theorem slope_of_line (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) :=
by sorry

end slope_of_line_l2554_255496


namespace donnys_remaining_money_l2554_255481

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℕ) (kite_cost : ℕ) (frisbee_cost : ℕ) : ℕ :=
  initial - (kite_cost + frisbee_cost)

/-- Theorem: Donny's remaining money after purchases -/
theorem donnys_remaining_money :
  remaining_money 78 8 9 = 61 := by
  sorry

end donnys_remaining_money_l2554_255481


namespace products_inspected_fraction_l2554_255428

/-- The fraction of products inspected by John, Jane, and Roy is 1 -/
theorem products_inspected_fraction (j n r : ℝ) : 
  j ≥ 0 → n ≥ 0 → r ≥ 0 →
  0.007 * j + 0.008 * n + 0.01 * r = 0.0085 →
  j + n + r = 1 :=
by sorry

end products_inspected_fraction_l2554_255428


namespace equation_solution_l2554_255436

theorem equation_solution :
  ∃ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + n / (n + 2) = 6 - 3 / (n + 2)) ∧ (n = -6/5) := by
  sorry

end equation_solution_l2554_255436


namespace unique_five_digit_divisible_by_72_l2554_255401

def is_divisible_by (n m : ℕ) : Prop := n % m = 0

theorem unique_five_digit_divisible_by_72 :
  ∀ (a b : ℕ), a < 10 → b < 10 →
    (is_divisible_by (a * 10000 + 6790 + b) 72 ↔ a = 3 ∧ b = 2) := by
  sorry

end unique_five_digit_divisible_by_72_l2554_255401


namespace roberto_outfits_l2554_255409

/-- The number of different outfits Roberto can assemble -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ := trousers * shirts * jackets

/-- Theorem stating the number of outfits Roberto can assemble -/
theorem roberto_outfits : number_of_outfits 5 5 3 = 75 := by
  sorry

end roberto_outfits_l2554_255409


namespace abs_inequality_equivalence_l2554_255427

theorem abs_inequality_equivalence (x : ℝ) : 2 ≤ |x - 5| ∧ |x - 5| ≤ 8 ↔ x ∈ Set.Icc (-3) 3 ∪ Set.Icc 7 13 := by
  sorry

end abs_inequality_equivalence_l2554_255427


namespace continuous_function_composition_eq_power_l2554_255480

/-- A continuous function satisfying f(f(x)) = kx^9 exists if and only if k ≥ 0 -/
theorem continuous_function_composition_eq_power (k : ℝ) :
  (∃ f : ℝ → ℝ, Continuous f ∧ ∀ x, f (f x) = k * x^9) ↔ k ≥ 0 :=
sorry

end continuous_function_composition_eq_power_l2554_255480


namespace quadratic_general_form_l2554_255417

/-- Given a quadratic equation x² = 3x + 1, its general form is x² - 3x - 1 = 0 -/
theorem quadratic_general_form :
  (fun x : ℝ => x^2) = (fun x : ℝ => 3*x + 1) →
  (fun x : ℝ => x^2 - 3*x - 1) = (fun x : ℝ => 0) := by
sorry

end quadratic_general_form_l2554_255417


namespace triangle_side_length_l2554_255485

theorem triangle_side_length (X Y Z : ℝ) (x y z : ℝ) :
  y = 7 →
  z = 5 →
  Real.cos (Y - Z) = 21 / 32 →
  x^2 = 47.75 :=
by sorry

end triangle_side_length_l2554_255485


namespace marble_redistribution_l2554_255471

/-- Represents the number of marbles each person has -/
structure Marbles :=
  (dilan : ℕ)
  (martha : ℕ)
  (phillip : ℕ)
  (veronica : ℕ)

/-- The theorem statement -/
theorem marble_redistribution (initial : Marbles) (final : Marbles) :
  initial.dilan = 14 →
  initial.martha = 20 →
  initial.veronica = 7 →
  final.dilan = 15 →
  final.martha = 15 →
  final.phillip = 15 →
  final.veronica = 15 →
  initial.dilan + initial.martha + initial.phillip + initial.veronica =
  final.dilan + final.martha + final.phillip + final.veronica →
  initial.phillip = 19 := by
  sorry

end marble_redistribution_l2554_255471


namespace remainder_sum_l2554_255405

theorem remainder_sum (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5 = 5) := by
  sorry

end remainder_sum_l2554_255405


namespace min_m_value_l2554_255424

/-- The minimum value of m that satisfies the given conditions -/
theorem min_m_value (m : ℝ) (h_m : m > 0) : 
  (∀ x₁ x₂ : ℝ, 
    let y₁ := Real.exp x₁
    let y₂ := 1 + Real.log (x₂ - m)
    y₁ = y₂ → |x₂ - x₁| ≥ Real.exp 1) → 
  m ≥ Real.exp 1 - 1 :=
sorry

end min_m_value_l2554_255424


namespace negation_of_universal_positive_square_plus_one_l2554_255434

theorem negation_of_universal_positive_square_plus_one :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by sorry

end negation_of_universal_positive_square_plus_one_l2554_255434


namespace parabola_intersection_sum_l2554_255414

/-- Given two parabolas that intersect the coordinate axes at four points forming a rectangle with area 36, prove that the sum of their coefficients is 4/27 -/
theorem parabola_intersection_sum (a b : ℝ) : 
  (∃ x y : ℝ, y = a * x^2 + 3 ∧ (x = 0 ∨ y = 0)) ∧ 
  (∃ x y : ℝ, y = 7 - b * x^2 ∧ (x = 0 ∨ y = 0)) ∧
  (∃ x1 x2 y1 y2 : ℝ, 
    (x1 ≠ 0 ∧ y1 = 0 ∧ y1 = a * x1^2 + 3) ∧
    (x2 ≠ 0 ∧ y2 = 0 ∧ y2 = 7 - b * x2^2) ∧
    (x1 * y2 = 36)) →
  a + b = 4/27 := by
sorry

end parabola_intersection_sum_l2554_255414


namespace morning_evening_email_difference_l2554_255494

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 10

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- Theorem stating the difference between morning and evening emails -/
theorem morning_evening_email_difference : 
  morning_emails - evening_emails = 2 := by sorry

end morning_evening_email_difference_l2554_255494


namespace max_value_theorem_max_value_achieved_l2554_255410

theorem max_value_theorem (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : 4 * u + 3 * v < 84) :
  u * v * (84 - 4 * u - 3 * v)^2 ≤ 259308 :=
sorry

theorem max_value_achieved (u v : ℝ) (hu : u > 0) (hv : v > 0) (h : 4 * u + 3 * v < 84) :
  ∃ (u₀ v₀ : ℝ), u₀ > 0 ∧ v₀ > 0 ∧ 4 * u₀ + 3 * v₀ < 84 ∧
    u₀ * v₀ * (84 - 4 * u₀ - 3 * v₀)^2 = 259308 :=
sorry

end max_value_theorem_max_value_achieved_l2554_255410


namespace dark_tile_fraction_is_seven_sixteenths_l2554_255443

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor where
  pattern_size : Nat
  corner_symmetry : Bool
  dark_tiles_in_quadrant : Nat

/-- Calculates the fraction of dark tiles on the floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  sorry

/-- Theorem stating that for a floor with the given properties, 
    the fraction of dark tiles is 7/16 -/
theorem dark_tile_fraction_is_seven_sixteenths 
  (floor : TiledFloor) 
  (h1 : floor.pattern_size = 8) 
  (h2 : floor.corner_symmetry = true) 
  (h3 : floor.dark_tiles_in_quadrant = 7) : 
  dark_tile_fraction floor = 7 / 16 :=
sorry

end dark_tile_fraction_is_seven_sixteenths_l2554_255443


namespace max_notebooks_purchase_l2554_255495

theorem max_notebooks_purchase (total_items : ℕ) (notebook_cost pencil_case_cost max_cost : ℚ) :
  total_items = 10 →
  notebook_cost = 12 →
  pencil_case_cost = 7 →
  max_cost = 100 →
  (∀ x : ℕ, x ≤ total_items →
    x * notebook_cost + (total_items - x) * pencil_case_cost ≤ max_cost →
    x ≤ 6) ∧
  ∃ x : ℕ, x = 6 ∧ x ≤ total_items ∧
    x * notebook_cost + (total_items - x) * pencil_case_cost ≤ max_cost :=
by sorry

end max_notebooks_purchase_l2554_255495


namespace mikes_earnings_l2554_255438

/-- Calculates the total earnings from selling working video games -/
def calculate_earnings (total_games : ℕ) (non_working_games : ℕ) (price_per_game : ℕ) : ℕ :=
  (total_games - non_working_games) * price_per_game

/-- Proves that Mike's earnings from selling his working video games is $56 -/
theorem mikes_earnings : 
  calculate_earnings 16 8 7 = 56 := by
  sorry

end mikes_earnings_l2554_255438


namespace car_speed_increase_l2554_255490

/-- Calculates the final speed of a car after modifications -/
def final_speed (original_speed : ℝ) (supercharge_percentage : ℝ) (weight_cut_increase : ℝ) : ℝ :=
  original_speed * (1 + supercharge_percentage) + weight_cut_increase

/-- Theorem stating that the final speed is 205 mph given the specified conditions -/
theorem car_speed_increase :
  final_speed 150 0.3 10 = 205 := by
  sorry

end car_speed_increase_l2554_255490


namespace integer_roots_of_cubic_l2554_255463

def f (x : ℤ) : ℤ := x^3 - 3*x^2 - 13*x + 15

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {-3, 1, 5} := by sorry

end integer_roots_of_cubic_l2554_255463


namespace student_count_l2554_255451

theorem student_count (total_pencils : ℕ) (pencils_per_student : ℕ) 
  (h1 : total_pencils = 18) 
  (h2 : pencils_per_student = 9) : 
  total_pencils / pencils_per_student = 2 := by
  sorry

end student_count_l2554_255451


namespace ordering_theorem_l2554_255483

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def monotonically_decreasing_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 0 → f y < f x

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem ordering_theorem (h1 : monotonically_decreasing_neg f) (h2 : even_function f) :
  f (-1) < f 9 ∧ f 9 < f 13 := by
  sorry

end ordering_theorem_l2554_255483


namespace local_maximum_value_l2554_255486

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + x - 1

theorem local_maximum_value (x : ℝ) :
  ∃ (a : ℝ), (∀ (y : ℝ), ∃ (ε : ℝ), ε > 0 ∧ ∀ (z : ℝ), |z - a| < ε → f z ≤ f a) ∧
  f a = -23/27 :=
sorry

end local_maximum_value_l2554_255486


namespace half_reporters_not_cover_politics_l2554_255439

/-- Represents the percentage of reporters covering local politics in country X -/
def local_politics_coverage : ℝ := 35

/-- Represents the percentage of political reporters not covering local politics in country X -/
def non_local_political_coverage : ℝ := 30

/-- Theorem stating that 50% of reporters do not cover politics -/
theorem half_reporters_not_cover_politics : 
  local_politics_coverage = 35 ∧ 
  non_local_political_coverage = 30 → 
  (100 : ℝ) - (local_politics_coverage / ((100 : ℝ) - non_local_political_coverage) * 100) = 50 := by
  sorry

end half_reporters_not_cover_politics_l2554_255439


namespace lunch_break_duration_l2554_255446

/-- Represents the painting scenario with Paul and his assistants --/
structure PaintingScenario where
  paul_rate : ℝ
  assistants_rate : ℝ
  lunch_break : ℝ

/-- Checks if the given scenario satisfies all conditions --/
def satisfies_conditions (s : PaintingScenario) : Prop :=
  -- Monday's condition
  (8 - s.lunch_break) * (s.paul_rate + s.assistants_rate) = 0.6 ∧
  -- Tuesday's condition
  (6 - s.lunch_break) * s.assistants_rate = 0.3 ∧
  -- Wednesday's condition
  (4 - s.lunch_break) * s.paul_rate = 0.1

/-- Theorem stating that the lunch break duration is 60 minutes --/
theorem lunch_break_duration :
  ∃ (s : PaintingScenario), s.lunch_break = 1 ∧ satisfies_conditions s :=
sorry

end lunch_break_duration_l2554_255446


namespace kiana_and_twins_ages_l2554_255416

theorem kiana_and_twins_ages (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
  sorry

end kiana_and_twins_ages_l2554_255416


namespace total_sales_given_april_l2554_255435

/-- Bennett's window screen sales pattern --/
structure BennettSales where
  january : ℕ
  february : ℕ := 2 * january
  march : ℕ := (january + february) / 2
  april : ℕ := min (2 * march) 20000

/-- Theorem: Total sales given April sales of 18000 --/
theorem total_sales_given_april (sales : BennettSales) 
  (h_april : sales.april = 18000) : 
  sales.january + sales.february + sales.march + sales.april = 45000 := by
  sorry

end total_sales_given_april_l2554_255435


namespace equal_intercepts_equation_not_in_second_quadrant_range_l2554_255462

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 + a = 0

-- Define the condition for equal intercepts
def equal_intercepts (a : ℝ) : Prop :=
  ∃ k, k = -a - 2 ∧ k = (-a - 2) / (a + 1)

-- Define the condition for not passing through the second quadrant
def not_in_second_quadrant (a : ℝ) : Prop :=
  a = -1 ∨ (-(a + 1) > 0 ∧ -a - 2 ≤ 0)

-- Theorem for equal intercepts
theorem equal_intercepts_equation (a : ℝ) :
  equal_intercepts a → (a = 0 ∨ a = -2) :=
sorry

-- Theorem for not passing through the second quadrant
theorem not_in_second_quadrant_range (a : ℝ) :
  not_in_second_quadrant a → -2 ≤ a ∧ a ≤ -1 :=
sorry

end equal_intercepts_equation_not_in_second_quadrant_range_l2554_255462


namespace imaginary_part_of_z_l2554_255406

theorem imaginary_part_of_z (x y : ℝ) (h : (x - Complex.I) * Complex.I = y + 2 * Complex.I) :
  (x + y * Complex.I).im = 1 := by
  sorry

end imaginary_part_of_z_l2554_255406


namespace circular_garden_ratio_l2554_255425

theorem circular_garden_ratio : 
  let r : ℝ := 8
  let circumference := 2 * Real.pi * r
  let area := Real.pi * r^2
  circumference / area = 1 / 4 := by sorry

end circular_garden_ratio_l2554_255425


namespace circle_C_equation_range_of_a_symmetry_condition_l2554_255453

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25

-- Define the chord line
def chord_line (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop := a * x - y + 5 = 0

-- Define point P
def point_P : ℝ × ℝ := (-2, 4)

-- Theorem 1: Prove that the equation represents circle C
theorem circle_C_equation :
  ∃ (m : ℝ), m > 0 ∧
  (∀ (x y : ℝ), circle_C x y ↔ (x - m)^2 + y^2 = 25) ∧
  (∃ (x y : ℝ), chord_line x y ∧ circle_C x y ∧
    ∃ (x' y' : ℝ), chord_line x' y' ∧ circle_C x' y' ∧
    (x - x')^2 + (y - y')^2 = 4 * 17) :=
sorry

-- Theorem 2: Prove the range of a
theorem range_of_a :
  ∀ (a : ℝ), (∃ (x y : ℝ), intersecting_line a x y ∧ circle_C x y) ↔
  (a < 0 ∨ a > 5/12) :=
sorry

-- Theorem 3: Prove the symmetry condition
theorem symmetry_condition :
  ∃ (a : ℝ), a = 3/4 ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ),
    intersecting_line a x₁ y₁ ∧ circle_C x₁ y₁ ∧
    intersecting_line a x₂ y₂ ∧ circle_C x₂ y₂ ∧
    x₁ ≠ x₂ →
    (x₁ + x₂) * (point_P.1 + 2) + (y₁ + y₂) * (point_P.2 - 4) = 0) :=
sorry

end circle_C_equation_range_of_a_symmetry_condition_l2554_255453


namespace max_books_borrowed_l2554_255491

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (average_books : ℚ) (h1 : total_students = 25) (h2 : zero_books = 3) (h3 : one_book = 10) 
  (h4 : two_books = 4) (h5 : average_books = 5/2) : ℕ :=
  let total_books := (total_students : ℚ) * average_books
  let accounted_students := zero_books + one_book + two_books
  let remaining_students := total_students - accounted_students
  let accounted_books := one_book * 1 + two_books * 2
  let remaining_books := total_books - accounted_books
  let min_books_per_remaining := 3
  24

end max_books_borrowed_l2554_255491


namespace problem_solution_l2554_255448

theorem problem_solution (a b : ℝ) 
  (h1 : Real.log a + b = -2)
  (h2 : a ^ b = 10) : 
  a = (1 : ℝ) / 10 := by
sorry

end problem_solution_l2554_255448


namespace product_of_solutions_l2554_255420

theorem product_of_solutions (x : ℝ) : 
  (3 * x^2 + 5 * x - 40 = 0) → 
  (∃ y : ℝ, 3 * y^2 + 5 * y - 40 = 0 ∧ x * y = -40/3) :=
by sorry

end product_of_solutions_l2554_255420


namespace isosceles_triangle_base_length_l2554_255421

/-- An isosceles triangle with two sides of length 8 cm and perimeter 26 cm has a base of length 10 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 26 →
  base = 10 := by
sorry

end isosceles_triangle_base_length_l2554_255421


namespace tangent_line_circle_l2554_255426

/-- A line is tangent to a circle if the distance from the center of the circle to the line is equal to the radius of the circle -/
def is_tangent (r : ℝ) : Prop :=
  r > 0 ∧ (r / Real.sqrt 2 = 2 * Real.sqrt r)

theorem tangent_line_circle (r : ℝ) : is_tangent r ↔ r = 8 := by
  sorry

end tangent_line_circle_l2554_255426


namespace percent_relation_l2554_255464

theorem percent_relation (a b : ℝ) (h : a = 2 * b) : 4 * b = 2 * a := by sorry

end percent_relation_l2554_255464


namespace photo_exhibition_total_l2554_255492

/-- Represents the number of photographs in various categories -/
structure PhotoExhibition where
  octavia_total : ℕ  -- Total photos taken by Octavia
  jack_octavia : ℕ   -- Photos taken by Octavia and framed by Jack
  jack_others : ℕ    -- Photos taken by others and framed by Jack

/-- Theorem stating the total number of photos either framed by Jack or taken by Octavia -/
theorem photo_exhibition_total (e : PhotoExhibition) 
  (h1 : e.octavia_total = 36)
  (h2 : e.jack_octavia = 24)
  (h3 : e.jack_others = 12) : 
  e.octavia_total + e.jack_others = 48 := by
  sorry


end photo_exhibition_total_l2554_255492


namespace perpendicular_implies_intersects_parallel_perpendicular_transitive_perpendicular_implies_parallel_l2554_255432

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (contains : Plane → Line → Prop)

-- Statement 1
theorem perpendicular_implies_intersects (l : Line) (a : Plane) :
  perpendicular l a → intersects l a :=
sorry

-- Statement 3
theorem parallel_perpendicular_transitive (l m n : Line) (a : Plane) :
  parallel l m → parallel m n → perpendicular l a → perpendicular n a :=
sorry

-- Statement 4
theorem perpendicular_implies_parallel (l m n : Line) (a : Plane) :
  parallel l m → perpendicular m a → perpendicular n a → parallel l n :=
sorry

end perpendicular_implies_intersects_parallel_perpendicular_transitive_perpendicular_implies_parallel_l2554_255432


namespace julies_salary_l2554_255408

/-- Calculates the monthly salary for a worker given specific conditions -/
def monthlySalary (hourlyRate : ℕ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) (missedDays : ℕ) : ℕ :=
  let dailyEarnings := hourlyRate * hoursPerDay
  let weeklyEarnings := dailyEarnings * daysPerWeek
  let monthlyEarnings := weeklyEarnings * 4
  monthlyEarnings - (dailyEarnings * missedDays)

/-- Proves that given the specific conditions, the monthly salary is $920 -/
theorem julies_salary : 
  monthlySalary 5 8 6 1 = 920 := by
  sorry

#eval monthlySalary 5 8 6 1

end julies_salary_l2554_255408


namespace eighth_group_frequency_l2554_255444

theorem eighth_group_frequency 
  (f1 f2 f3 f4 : ℝ) 
  (f5_to_7 : ℝ) 
  (h1 : f1 = 0.15)
  (h2 : f2 = 0.17)
  (h3 : f3 = 0.11)
  (h4 : f4 = 0.13)
  (h5 : f5_to_7 = 0.32)
  (h6 : ∀ f : ℝ, f ≥ 0 → f ≤ 1) -- Assumption: all frequencies are between 0 and 1
  (h7 : f1 + f2 + f3 + f4 + f5_to_7 + (1 - (f1 + f2 + f3 + f4 + f5_to_7)) = 1) -- Sum of all frequencies is 1
  : 1 - (f1 + f2 + f3 + f4 + f5_to_7) = 0.12 := by
  sorry

end eighth_group_frequency_l2554_255444


namespace right_triangle_max_expression_l2554_255466

/-- For a right triangle with legs a and b, and hypotenuse c, 
    the expression (a^2 + b^2 + ab) / c^2 is maximized at 1.5 -/
theorem right_triangle_max_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_triangle : a^2 + b^2 = c^2) :
  (a^2 + b^2 + a*b) / c^2 ≤ (3/2) ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    a'^2 + b'^2 = c'^2 ∧ (a'^2 + b'^2 + a'*b') / c'^2 = (3/2) := by
  sorry

end right_triangle_max_expression_l2554_255466


namespace sphere_radius_equals_seven_l2554_255469

/-- Given a sphere and a right circular cylinder where:
    1. The surface area of the sphere equals the curved surface area of the cylinder
    2. The height of the cylinder is 14 cm
    3. The diameter of the cylinder is 14 cm
    This theorem proves that the radius of the sphere is 7 cm. -/
theorem sphere_radius_equals_seven (r : ℝ) :
  (4 * Real.pi * r^2 = 2 * Real.pi * 7 * 14) →
  r = 7 := by
  sorry

#check sphere_radius_equals_seven

end sphere_radius_equals_seven_l2554_255469


namespace good_numbers_exist_l2554_255493

def has_no_repeating_digits (n : ℕ) : Prop :=
  (n / 10) % 10 ≠ n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def increase_digits (n : ℕ) : ℕ :=
  ((n / 10) + 1) * 10 + ((n % 10) + 1)

theorem good_numbers_exist : ∃ n₁ n₂ : ℕ,
  n₁ ≠ n₂ ∧
  10 ≤ n₁ ∧ n₁ < 100 ∧
  10 ≤ n₂ ∧ n₂ < 100 ∧
  has_no_repeating_digits n₁ ∧
  has_no_repeating_digits n₂ ∧
  n₁ % sum_of_digits n₁ = 0 ∧
  n₂ % sum_of_digits n₂ = 0 ∧
  has_no_repeating_digits (increase_digits n₁) ∧
  has_no_repeating_digits (increase_digits n₂) ∧
  (increase_digits n₁) % sum_of_digits (increase_digits n₁) = 0 ∧
  (increase_digits n₂) % sum_of_digits (increase_digits n₂) = 0 :=
sorry

end good_numbers_exist_l2554_255493


namespace trapezoid_base_lengths_l2554_255474

theorem trapezoid_base_lengths (h : ℝ) (leg1 leg2 larger_base : ℝ) :
  h = 12 ∧ leg1 = 20 ∧ leg2 = 15 ∧ larger_base = 42 →
  ∃ (smaller_base : ℝ), (smaller_base = 17 ∨ smaller_base = 35) ∧
  (∃ (x y : ℝ), x^2 + h^2 = leg1^2 ∧ y^2 + h^2 = leg2^2 ∧
  (larger_base = x + y + smaller_base ∨ larger_base = x - y + smaller_base)) :=
by sorry

end trapezoid_base_lengths_l2554_255474


namespace roots_of_quadratic_l2554_255475

theorem roots_of_quadratic (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 12) :
  x^2 - 10*x - 11 = 0 ∧ y^2 - 10*y - 11 = 0 := by
  sorry

end roots_of_quadratic_l2554_255475


namespace x_squared_plus_y_cubed_eq_neg_seven_l2554_255459

theorem x_squared_plus_y_cubed_eq_neg_seven 
  (x y : ℝ) 
  (h : |x - 1| + (y + 2)^2 = 0) : 
  x^2 + y^3 = -7 := by
  sorry

end x_squared_plus_y_cubed_eq_neg_seven_l2554_255459


namespace equation_solution_l2554_255478

theorem equation_solution : ∃! x : ℚ, x - 5/6 = 7/18 - x/4 ∧ x = 44/45 := by
  sorry

end equation_solution_l2554_255478


namespace max_revenue_l2554_255482

/-- The revenue function for the bookstore --/
def revenue (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- Theorem stating the maximum revenue and the price at which it occurs --/
theorem max_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  revenue p = 140.625 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → revenue q ≤ revenue p :=
by
  sorry

end max_revenue_l2554_255482


namespace area_of_region_T_l2554_255445

/-- Represents a rhombus PQRS -/
structure Rhombus where
  side_length : ℝ
  angle_Q : ℝ

/-- Represents the region T inside the rhombus -/
def region_T (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a region -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem statement -/
theorem area_of_region_T (r : Rhombus) :
  r.side_length = 4 ∧ r.angle_Q = 150 * π / 180 →
  area (region_T r) = 8 * Real.sqrt 3 / 9 :=
sorry

end area_of_region_T_l2554_255445


namespace barbara_tuna_packs_l2554_255457

/-- The number of tuna packs Barbara bought -/
def tuna_packs : ℕ := sorry

/-- The price of each tuna pack in dollars -/
def tuna_price : ℚ := 2

/-- The number of water bottles Barbara bought -/
def water_bottles : ℕ := 4

/-- The price of each water bottle in dollars -/
def water_price : ℚ := (3 : ℚ) / 2

/-- The amount spent on different goods in dollars -/
def different_goods_cost : ℚ := 40

/-- The total amount Barbara paid in dollars -/
def total_paid : ℚ := 56

theorem barbara_tuna_packs : 
  tuna_packs = 5 ∧ 
  (tuna_packs : ℚ) * tuna_price + (water_bottles : ℚ) * water_price + different_goods_cost = total_paid :=
sorry

end barbara_tuna_packs_l2554_255457


namespace geometric_sequence_product_l2554_255433

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_cond : a 1 * a 5 = 4) :
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 ∨ a 1 * a 2 * a 3 * a 4 * a 5 = -32 :=
by
  sorry

end geometric_sequence_product_l2554_255433


namespace percentage_product_theorem_l2554_255470

theorem percentage_product_theorem :
  let p1 : ℝ := 40
  let p2 : ℝ := 35
  let p3 : ℝ := 60
  let p4 : ℝ := 70
  let result : ℝ := p1 * p2 * p3 * p4 / 1000000 * 100
  result = 5.88 := by
sorry

end percentage_product_theorem_l2554_255470


namespace min_value_theorem_l2554_255488

theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 := by
  sorry

end min_value_theorem_l2554_255488


namespace lcm_of_9_12_15_l2554_255456

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end lcm_of_9_12_15_l2554_255456


namespace solution_proof_l2554_255473

noncomputable def f (x : ℝ) : ℝ := 30 / (x + 2)

noncomputable def g (x : ℝ) : ℝ := 4 * (f⁻¹ x)

theorem solution_proof : ∃ x : ℝ, g x = 20 ∧ x = 30 / 7 := by
  sorry

end solution_proof_l2554_255473


namespace function_passes_through_point_l2554_255458

theorem function_passes_through_point 
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) - 2
  f 1 = -1 := by sorry

end function_passes_through_point_l2554_255458


namespace inequality_proof_l2554_255468

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) : a - c > b - d := by
  sorry

end inequality_proof_l2554_255468


namespace square_sum_of_xy_l2554_255487

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 71)
  (h2 : x^2 * y + x * y^2 = 880) : 
  x^2 + y^2 = 146 := by
  sorry

end square_sum_of_xy_l2554_255487


namespace grade_assignment_count_l2554_255412

theorem grade_assignment_count : 
  (Nat.choose 12 2) * (3^10) = 3906234 := by sorry

end grade_assignment_count_l2554_255412


namespace pizza_tip_percentage_l2554_255418

/-- Calculates the tip percentage for Harry's pizza order --/
theorem pizza_tip_percentage
  (large_pizza_cost : ℝ)
  (topping_cost : ℝ)
  (num_pizzas : ℕ)
  (toppings_per_pizza : ℕ)
  (total_cost_with_tip : ℝ)
  (h1 : large_pizza_cost = 14)
  (h2 : topping_cost = 2)
  (h3 : num_pizzas = 2)
  (h4 : toppings_per_pizza = 3)
  (h5 : total_cost_with_tip = 50)
  : (total_cost_with_tip - (num_pizzas * large_pizza_cost + num_pizzas * toppings_per_pizza * topping_cost)) /
    (num_pizzas * large_pizza_cost + num_pizzas * toppings_per_pizza * topping_cost) = 0.25 := by
  sorry


end pizza_tip_percentage_l2554_255418


namespace plane_equation_correct_l2554_255407

/-- A plane equation represented by integers A, B, C, and D -/
structure PlaneEquation where
  A : Int
  B : Int
  C : Int
  D : Int
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- Check if a point (x, y, z) lies on a plane -/
def lies_on_plane (p : PlaneEquation) (x y z : ℝ) : Prop :=
  p.A * x + p.B * y + p.C * z + p.D = 0

/-- Check if two planes are perpendicular -/
def perpendicular_planes (p1 p2 : PlaneEquation) : Prop :=
  p1.A * p2.A + p1.B * p2.B + p1.C * p2.C = 0

theorem plane_equation_correct (p : PlaneEquation) 
  (h1 : p.A = 2 ∧ p.B = -2 ∧ p.C = 1 ∧ p.D = 1) 
  (h2 : lies_on_plane p 0 2 3) 
  (h3 : lies_on_plane p 2 0 3) 
  (h4 : perpendicular_planes p { A := 1, B := -1, C := 4, D := -7, A_pos := by norm_num, gcd_one := by norm_num }) : 
  p.A = 2 ∧ p.B = -2 ∧ p.C = 1 ∧ p.D = 1 := by
  sorry

end plane_equation_correct_l2554_255407


namespace ratio_unchanged_l2554_255430

theorem ratio_unchanged (a b : ℝ) (h : b ≠ 0) :
  (3 * a) / (b / (1 / 3)) = a / b :=
by sorry

end ratio_unchanged_l2554_255430


namespace no_infinite_prime_sequence_with_condition_l2554_255400

theorem no_infinite_prime_sequence_with_condition :
  ¬ ∃ (p : ℕ → ℕ), 
    (∀ n, Prime (p n)) ∧ 
    (∀ n, p n < p (n + 1)) ∧ 
    (∀ k, p (k + 1) = 2 * p k - 1 ∨ p (k + 1) = 2 * p k + 1) :=
by sorry

end no_infinite_prime_sequence_with_condition_l2554_255400


namespace quadratic_inequality_condition_l2554_255429

theorem quadratic_inequality_condition (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c < 0) → (∀ x, a*x^2 + b*x + c > 0) ∧
  ¬(∀ x, a*x^2 + b*x + c > 0 → (a > 0 ∧ b^2 - 4*a*c < 0)) :=
by sorry

end quadratic_inequality_condition_l2554_255429


namespace specific_cube_surface_area_l2554_255477

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a drilled tunnel -/
structure CubeWithTunnel where
  sideLength : ℝ
  pointI : Point3D
  pointJ : Point3D
  pointK : Point3D

/-- Calculates the total surface area of a cube with a drilled tunnel -/
def totalSurfaceArea (cube : CubeWithTunnel) : ℝ :=
  sorry

/-- Theorem stating the total surface area of the specific cube with tunnel -/
theorem specific_cube_surface_area :
  let cube : CubeWithTunnel := {
    sideLength := 10,
    pointI := { x := 3, y := 0, z := 0 },
    pointJ := { x := 0, y := 3, z := 0 },
    pointK := { x := 0, y := 0, z := 3 }
  }
  totalSurfaceArea cube = 630 := by sorry

end specific_cube_surface_area_l2554_255477


namespace f_max_value_l2554_255497

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + 2 * Real.cos x

theorem f_max_value :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ M = 2 := by
  sorry

end f_max_value_l2554_255497


namespace net_gain_proof_l2554_255484

def initial_value : ℝ := 15000

def first_sale (value : ℝ) : ℝ := value * 1.2
def second_sale (value : ℝ) : ℝ := value * 0.85
def third_sale (value : ℝ) : ℝ := value * 1.1
def fourth_sale (value : ℝ) : ℝ := value * 0.95

def total_expense (initial : ℝ) : ℝ :=
  second_sale (first_sale initial) + fourth_sale (third_sale (second_sale (first_sale initial)))

def total_income (initial : ℝ) : ℝ :=
  first_sale initial + third_sale (second_sale (first_sale initial))

theorem net_gain_proof :
  total_income initial_value - total_expense initial_value = 3541.50 := by
  sorry

end net_gain_proof_l2554_255484


namespace square_side_length_from_rectangle_l2554_255452

theorem square_side_length_from_rectangle (width height : ℝ) (h1 : width = 10) (h2 : height = 20) :
  ∃ y : ℝ, y^2 = width * height ∧ y = 10 * Real.sqrt 2 :=
by sorry

end square_side_length_from_rectangle_l2554_255452


namespace quadratic_equation_coefficients_l2554_255419

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 3 * x * (x - 1) = 2 * (x + 2) + 8 ↔ a * x^2 + b * x + c = 0) ∧
    a = 3 ∧ b = -5 ∧ c = -12 := by
  sorry

end quadratic_equation_coefficients_l2554_255419


namespace swimmer_speed_in_still_water_l2554_255479

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions of the swimming problem, the swimmer's speed in still water is 5.5 km/h. -/
theorem swimmer_speed_in_still_water 
  (s : SwimmerSpeed) 
  (h1 : effectiveSpeed s true = 35 / 5)   -- Downstream condition
  (h2 : effectiveSpeed s false = 20 / 5)  -- Upstream condition
  : s.swimmer = 5.5 := by
  sorry

#eval 5.5  -- To check if the value 5.5 is recognized correctly

end swimmer_speed_in_still_water_l2554_255479
