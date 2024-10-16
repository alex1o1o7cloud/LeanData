import Mathlib

namespace NUMINAMATH_CALUDE_constant_remainder_l2995_299596

-- Define the polynomials
def f (b : ℚ) (x : ℚ) : ℚ := 12 * x^3 - 9 * x^2 + b * x + 8
def g (x : ℚ) : ℚ := 3 * x^2 - 4 * x + 2

-- Define the remainder function
def remainder (b : ℚ) (x : ℚ) : ℚ := f b x - g x * ((4 * x + 0) : ℚ)

-- Theorem statement
theorem constant_remainder :
  ∃ (c : ℚ), ∀ (x : ℚ), remainder (-4/3) x = c :=
sorry

end NUMINAMATH_CALUDE_constant_remainder_l2995_299596


namespace NUMINAMATH_CALUDE_number_of_hens_l2995_299513

/-- Given a farm with hens and cows, prove that the number of hens is 24 -/
theorem number_of_hens (hens cows : ℕ) : 
  hens + cows = 44 →  -- Total number of heads
  2 * hens + 4 * cows = 128 →  -- Total number of feet
  hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hens_l2995_299513


namespace NUMINAMATH_CALUDE_ceiling_sum_of_roots_l2995_299508

theorem ceiling_sum_of_roots : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_of_roots_l2995_299508


namespace NUMINAMATH_CALUDE_product_of_numbers_l2995_299522

theorem product_of_numbers (x y : ℝ) : x^2 + y^2 = 289 ∧ x + y = 23 → x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2995_299522


namespace NUMINAMATH_CALUDE_leftover_value_is_seven_l2995_299597

/-- Calculates the value of leftover coins after pooling and rolling --/
def leftover_value (james_quarters james_dimes rebecca_quarters rebecca_dimes : ℕ) 
  (quarters_per_roll dimes_per_roll : ℕ) : ℚ :=
  let total_quarters := james_quarters + rebecca_quarters
  let total_dimes := james_dimes + rebecca_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * (1 / 4) + (leftover_dimes : ℚ) * (1 / 10)

theorem leftover_value_is_seven :
  leftover_value 50 80 170 340 40 50 = 7 := by
  sorry

end NUMINAMATH_CALUDE_leftover_value_is_seven_l2995_299597


namespace NUMINAMATH_CALUDE_snack_pack_distribution_l2995_299517

theorem snack_pack_distribution (pretzels : ℕ) (suckers : ℕ) (kids : ℕ) :
  pretzels = 64 →
  suckers = 32 →
  kids = 16 →
  (pretzels + 4 * pretzels + suckers) / kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_snack_pack_distribution_l2995_299517


namespace NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l2995_299552

theorem log_sin_in_terms_of_m_n (α m n : Real) 
  (h1 : 0 < α) (h2 : α < π/2)
  (h3 : Real.log (1 + Real.cos α) = m)
  (h4 : Real.log (1 / (1 - Real.cos α)) = n) :
  Real.log (Real.sin α) = (1/2) * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_log_sin_in_terms_of_m_n_l2995_299552


namespace NUMINAMATH_CALUDE_line_equation_to_slope_intercept_l2995_299562

/-- Given a line equation, prove it can be expressed in slope-intercept form --/
theorem line_equation_to_slope_intercept :
  ∀ (x y : ℝ),
  3 * (x + 2) - 4 * (y - 8) = 0 →
  y = (3 / 4) * x + (19 / 2) :=
by
  sorry

#check line_equation_to_slope_intercept

end NUMINAMATH_CALUDE_line_equation_to_slope_intercept_l2995_299562


namespace NUMINAMATH_CALUDE_carrie_profit_calculation_l2995_299516

/-- Calculates Carrie's profit from making a wedding cake --/
theorem carrie_profit_calculation :
  let weekday_hours : ℕ := 5 * 4
  let weekend_hours : ℕ := 3 * 4
  let weekday_rate : ℚ := 35
  let weekend_rate : ℚ := 45
  let supply_cost : ℚ := 180
  let supply_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.07

  let total_earnings : ℚ := weekday_hours * weekday_rate + weekend_hours * weekend_rate
  let discounted_supply_cost : ℚ := supply_cost * (1 - supply_discount)
  let sales_tax : ℚ := total_earnings * sales_tax_rate
  let profit : ℚ := total_earnings - discounted_supply_cost - sales_tax

  profit = 991.20 := by sorry

end NUMINAMATH_CALUDE_carrie_profit_calculation_l2995_299516


namespace NUMINAMATH_CALUDE_infinite_primes_4n_plus_3_l2995_299567

theorem infinite_primes_4n_plus_3 :
  ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p ∧ p % 4 = 3) →
  ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_4n_plus_3_l2995_299567


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_intersection_l2995_299529

theorem hyperbola_ellipse_intersection (m : ℝ) : 
  (∃ e : ℝ, e > Real.sqrt 2 ∧ e^2 = (3 + m) / 3) ∧ 
  (m / 2 > m - 2 ∧ m - 2 > 0) → 
  m ∈ Set.Ioo 3 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_intersection_l2995_299529


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l2995_299547

def total_pay : ℝ := 550
def a_percentage : ℝ := 1.2

theorem employee_pay_calculation (b_pay : ℝ) (h1 : b_pay > 0) 
  (h2 : b_pay + a_percentage * b_pay = total_pay) : b_pay = 250 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l2995_299547


namespace NUMINAMATH_CALUDE_skateboard_distance_l2995_299549

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The skateboard problem -/
theorem skateboard_distance : arithmeticSequenceSum 8 9 20 = 1870 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_distance_l2995_299549


namespace NUMINAMATH_CALUDE_currency_exchange_problem_l2995_299540

/-- 
Proves the existence of a positive integer d that satisfies the conditions
of the currency exchange problem and has a digit sum of 3.
-/
theorem currency_exchange_problem : ∃ d : ℕ+, 
  (8 : ℚ) / 5 * d.val - 72 = d.val ∧ 
  (d.val.repr.toList.map (λ c => c.toString.toNat!)).sum = 3 := by
  sorry

end NUMINAMATH_CALUDE_currency_exchange_problem_l2995_299540


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2995_299535

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 3 (-2) 0 →
  point = Point.mk 1 (-1) →
  ∃ (result_line : Line),
    result_line.perpendicular given_line ∧
    point.liesOn result_line ∧
    result_line = Line.mk 2 3 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2995_299535


namespace NUMINAMATH_CALUDE_circle_trajectory_and_min_distance_l2995_299518

-- Define the moving circle
def moving_circle (x y : ℝ) : Prop :=
  y > 0 ∧ Real.sqrt (x^2 + (y - 1)^2) = y + 1

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop :=
  y > 0 ∧ x^2 = 4*y

-- Define points A and B on trajectory E
def point_on_E (x y : ℝ) : Prop :=
  trajectory_E x y

-- Define the perpendicular tangents condition
def perpendicular_tangents (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  point_on_E x₁ y₁ ∧ point_on_E x₂ y₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -4

-- Main theorem
theorem circle_trajectory_and_min_distance :
  (∀ x y, moving_circle x y ↔ trajectory_E x y) ∧
  (∀ x₁ y₁ x₂ y₂, perpendicular_tangents x₁ y₁ x₂ y₂ →
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≥ 4) ∧
  (∃ x₁ y₁ x₂ y₂, perpendicular_tangents x₁ y₁ x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4) :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_and_min_distance_l2995_299518


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l2995_299558

/-- In a right triangle, if the hypotenuse exceeds one leg by 2, then the square of the other leg is 4a + 4 -/
theorem right_triangle_leg_square (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) -- Pythagorean theorem
  (h5 : c = a + 2) : -- Hypotenuse exceeds one leg by 2
  b^2 = 4*a + 4 := by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l2995_299558


namespace NUMINAMATH_CALUDE_quadratic_equation_one_l2995_299583

theorem quadratic_equation_one (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ x = 3/2 ∨ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_one_l2995_299583


namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l2995_299502

theorem quadratic_radicals_combination (x : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 3 * x + 5 = k * (2 * x + 7)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l2995_299502


namespace NUMINAMATH_CALUDE_hyperbola_to_ellipse_l2995_299526

/-- Given a hyperbola with equation y²/12 - x²/4 = 1, prove that the equation of the ellipse
    that has the foci of the hyperbola as its vertices and the vertices of the hyperbola as its foci
    is y²/16 + x²/4 = 1 -/
theorem hyperbola_to_ellipse (x y : ℝ) :
  (y^2 / 12 - x^2 / 4 = 1) →
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b) ∧
    (y^2 / a^2 + x^2 / b^2 = 1) ∧
    a = 4 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_to_ellipse_l2995_299526


namespace NUMINAMATH_CALUDE_cookies_sold_proof_l2995_299521

/-- The number of packs of cookies sold by Robyn -/
def robyn_sales : ℕ := 47

/-- The number of packs of cookies sold by Lucy -/
def lucy_sales : ℕ := 29

/-- The total number of packs of cookies sold by Robyn and Lucy -/
def total_sales : ℕ := robyn_sales + lucy_sales

theorem cookies_sold_proof : total_sales = 76 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_proof_l2995_299521


namespace NUMINAMATH_CALUDE_problem_solution_l2995_299520

def U : Set ℤ := {x | 0 ≤ x ∧ x ≤ 6}

theorem problem_solution (A B : Set ℤ) 
  (h1 : U = A ∪ B) 
  (h2 : A ∩ (U \ B) = {1, 3, 5}) : 
  B = {0, 2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2995_299520


namespace NUMINAMATH_CALUDE_jasmine_purchase_cost_l2995_299586

/-- The cost of Jasmine's purchase -/
def total_cost (coffee_pounds : ℕ) (milk_gallons : ℕ) (coffee_price : ℚ) (milk_price : ℚ) : ℚ :=
  coffee_pounds * coffee_price + milk_gallons * milk_price

/-- Proof that Jasmine's purchase costs $17 -/
theorem jasmine_purchase_cost :
  total_cost 4 2 (5/2) (7/2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_jasmine_purchase_cost_l2995_299586


namespace NUMINAMATH_CALUDE_difference_of_squares_l2995_299591

theorem difference_of_squares (x : ℝ) : 1 - x^2 = (1 - x) * (1 + x) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2995_299591


namespace NUMINAMATH_CALUDE_complex_absolute_value_squared_l2995_299551

theorem complex_absolute_value_squared (z : ℂ) (h : z + Complex.abs z = 1 + 12 * I) : Complex.abs z ^ 2 = 5256 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_squared_l2995_299551


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2995_299525

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ y : ℝ, x = Real.sqrt y ∧ 
  (∀ z : ℝ, z > 0 → z ≠ y → ¬∃ (a b : ℝ), a^2 * b = y ∧ b > 0 ∧ b ≠ 1)

theorem simplest_quadratic_radical : 
  is_simplest_quadratic_radical (Real.sqrt 6) ∧ 
  ¬is_simplest_quadratic_radical (Real.sqrt 27) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 9) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/4)) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2995_299525


namespace NUMINAMATH_CALUDE_shopping_expenditure_l2995_299511

/-- Represents the percentage spent on clothing -/
def clothing_percentage : ℝ := sorry

/-- Represents the percentage spent on food -/
def food_percentage : ℝ := 20

/-- Represents the percentage spent on other items -/
def other_percentage : ℝ := 30

/-- Represents the tax rate on clothing -/
def clothing_tax_rate : ℝ := 4

/-- Represents the tax rate on other items -/
def other_tax_rate : ℝ := 8

/-- Represents the total tax rate as a percentage of pre-tax spending -/
def total_tax_rate : ℝ := 4.4

theorem shopping_expenditure :
  clothing_percentage + food_percentage + other_percentage = 100 ∧
  clothing_percentage * clothing_tax_rate / 100 + other_percentage * other_tax_rate / 100 = total_tax_rate ∧
  clothing_percentage = 50 := by sorry

end NUMINAMATH_CALUDE_shopping_expenditure_l2995_299511


namespace NUMINAMATH_CALUDE_solve_equation_l2995_299573

theorem solve_equation (x : ℚ) : (3 * x + 4) / 7 = 15 → x = 101 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2995_299573


namespace NUMINAMATH_CALUDE_chicken_rabbit_equations_correct_l2995_299578

/-- Represents the "chicken-rabbit in the same cage" problem -/
structure ChickenRabbitProblem where
  total_heads : ℕ
  total_feet : ℕ
  chickens : ℕ
  rabbits : ℕ

/-- The system of equations for the chicken-rabbit problem -/
def correct_equations (problem : ChickenRabbitProblem) : Prop :=
  problem.chickens + problem.rabbits = problem.total_heads ∧
  2 * problem.chickens + 4 * problem.rabbits = problem.total_feet

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem chicken_rabbit_equations_correct (problem : ChickenRabbitProblem) 
  (h1 : problem.total_heads = 35)
  (h2 : problem.total_feet = 94) :
  correct_equations problem :=
sorry

end NUMINAMATH_CALUDE_chicken_rabbit_equations_correct_l2995_299578


namespace NUMINAMATH_CALUDE_x_eighth_power_is_one_l2995_299576

theorem x_eighth_power_is_one (x : ℂ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_eighth_power_is_one_l2995_299576


namespace NUMINAMATH_CALUDE_lemonade_sales_l2995_299598

theorem lemonade_sales (last_week : ℝ) (this_week : ℝ) (total : ℝ) : 
  this_week = 1.3 * last_week →
  total = last_week + this_week →
  total = 46 →
  last_week = 20 := by
sorry

end NUMINAMATH_CALUDE_lemonade_sales_l2995_299598


namespace NUMINAMATH_CALUDE_vacation_cost_difference_l2995_299542

/-- Proves that the difference between Tom's and Dorothy's payments to equalize costs is 20 --/
theorem vacation_cost_difference (tom_paid dorothy_paid sammy_paid : ℕ) 
  (h1 : tom_paid = 105)
  (h2 : dorothy_paid = 125)
  (h3 : sammy_paid = 175) : 
  (((tom_paid + dorothy_paid + sammy_paid) / 3 - tom_paid) - 
   ((tom_paid + dorothy_paid + sammy_paid) / 3 - dorothy_paid)) = 20 := by
  sorry

#eval ((105 + 125 + 175) / 3 - 105) - ((105 + 125 + 175) / 3 - 125)

end NUMINAMATH_CALUDE_vacation_cost_difference_l2995_299542


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2995_299519

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.choose n k else 0

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

theorem unique_solution_for_equation (x : ℕ) :
  x ≥ 7 → (3 * C (x - 3) 4 = 5 * A (x - 4) 2) → x = 11 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2995_299519


namespace NUMINAMATH_CALUDE_distance_between_parallel_lines_l2995_299570

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y - 4 = 0
  ∃ d : ℝ, d = Real.sqrt 5 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₂ x₂ y₂ →
      ((x₂ - x₁)^2 + (y₂ - y₁)^2 : ℝ) ≥ d^2 :=
by sorry


end NUMINAMATH_CALUDE_distance_between_parallel_lines_l2995_299570


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l2995_299560

-- Define the equation
def equation (x : ℝ) : Prop := Real.sqrt (8 - x) = x * Real.sqrt (8 - x)

-- Theorem statement
theorem equation_has_two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ equation a ∧ equation b ∧ ∀ (x : ℝ), equation x → (x = a ∨ x = b) :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l2995_299560


namespace NUMINAMATH_CALUDE_race_length_l2995_299595

theorem race_length (L : ℝ) 
  (h1 : L - 70 = L * ((L - 100) / L))  -- A beats B by 70 m
  (h2 : L - 163 = (L - 100) * ((L - 163) / (L - 100)))  -- B beats C by 100 m
  (h3 : L - 163 = L * ((L - 163) / L))  -- A beats C by 163 m
  : L = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_l2995_299595


namespace NUMINAMATH_CALUDE_area_not_above_y_axis_equals_total_area_l2995_299510

/-- Parallelogram PQRS with given vertices -/
structure Parallelogram where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The specific parallelogram from the problem -/
def PQRS : Parallelogram :=
  { P := (-1, 5)
    Q := (2, -3)
    R := (-5, -3)
    S := (-8, 5) }

/-- Area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  sorry

/-- Area of the part of the parallelogram not above the y-axis -/
def areaNotAboveYAxis (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area not above the y-axis equals the total area -/
theorem area_not_above_y_axis_equals_total_area :
  areaNotAboveYAxis PQRS = parallelogramArea PQRS :=
sorry

end NUMINAMATH_CALUDE_area_not_above_y_axis_equals_total_area_l2995_299510


namespace NUMINAMATH_CALUDE_anna_score_l2995_299548

/-- Calculates the score in a modified contest given the number of correct, incorrect, and unanswered questions -/
def contest_score (correct incorrect unanswered : ℕ) : ℚ :=
  (correct : ℚ) + 0 * (incorrect : ℚ) - 0.5 * (unanswered : ℚ)

theorem anna_score :
  let total_questions : ℕ := 30
  let correct_answers : ℕ := 17
  let incorrect_answers : ℕ := 6
  let unanswered_questions : ℕ := 7
  correct_answers + incorrect_answers + unanswered_questions = total_questions →
  contest_score correct_answers incorrect_answers unanswered_questions = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_anna_score_l2995_299548


namespace NUMINAMATH_CALUDE_sara_movie_expenses_l2995_299537

-- Define the cost of each item
def theater_ticket_cost : ℚ := 10.62
def theater_ticket_count : ℕ := 2
def rented_movie_cost : ℚ := 1.59
def purchased_movie_cost : ℚ := 13.95

-- Define the total spent on movies
def total_spent : ℚ :=
  theater_ticket_cost * theater_ticket_count + rented_movie_cost + purchased_movie_cost

-- Theorem to prove
theorem sara_movie_expenses : total_spent = 36.78 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_expenses_l2995_299537


namespace NUMINAMATH_CALUDE_quadratic_rewrite_product_l2995_299531

/-- Given a quadratic equation 9x^2 - 30x - 42 that can be rewritten as (ax + b)^2 + c
    where a, b, and c are integers, prove that ab = -15 -/
theorem quadratic_rewrite_product (a b c : ℤ) : 
  (∀ x, 9*x^2 - 30*x - 42 = (a*x + b)^2 + c) → a*b = -15 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_product_l2995_299531


namespace NUMINAMATH_CALUDE_specific_arithmetic_sequence_common_difference_l2995_299543

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  first_term : ℚ
  last_term : ℚ
  sum : ℚ
  is_arithmetic : Bool

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ :=
  sorry

/-- Theorem stating the common difference of the specific arithmetic sequence -/
theorem specific_arithmetic_sequence_common_difference :
  let seq := ArithmeticSequence.mk 3 28 186 true
  common_difference seq = 25 / 11 := by
  sorry

end NUMINAMATH_CALUDE_specific_arithmetic_sequence_common_difference_l2995_299543


namespace NUMINAMATH_CALUDE_initial_hats_count_hat_problem_solution_l2995_299564

/-- Represents the hat exchange scenario at the gentlemen's club --/
structure HatExchange where
  total_gentlemen : ℕ
  final_givers : ℕ
  initial_with_hats : ℕ

/-- The hat exchange satisfies the problem conditions --/
def valid_exchange (h : HatExchange) : Prop :=
  h.total_gentlemen = 20 ∧
  h.final_givers = 10 ∧
  h.initial_with_hats = h.total_gentlemen - h.final_givers

/-- Theorem stating that the number of gentlemen who initially wore hats is 10 --/
theorem initial_hats_count (h : HatExchange) (hvalid : valid_exchange h) : 
  h.initial_with_hats = 10 := by
  sorry

/-- Main theorem proving the solution to the problem --/
theorem hat_problem_solution : 
  ∃ (h : HatExchange), valid_exchange h ∧ h.initial_with_hats = 10 := by
  sorry

end NUMINAMATH_CALUDE_initial_hats_count_hat_problem_solution_l2995_299564


namespace NUMINAMATH_CALUDE_quadratic_inequality_relationship_l2995_299505

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

-- Define proposition A
def proposition_A (a : ℝ) : Prop := ∀ x, f a x > 0

-- Define proposition B
def proposition_B (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem statement
theorem quadratic_inequality_relationship :
  (∀ a : ℝ, proposition_A a → proposition_B a) ∧
  (∃ a : ℝ, proposition_B a ∧ ¬proposition_A a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_relationship_l2995_299505


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2995_299559

theorem trigonometric_equation_solution (x : ℝ) (k : ℤ) : 
  8.424 * Real.cos x + Real.sqrt (Real.sin x ^ 2 - 2 * Real.sin (2 * x) + 4 * Real.cos x ^ 2) = 0 ↔ 
  (x = Real.arctan (-6.424) + π * (2 * ↑k + 1) ∨ x = Real.arctan 5.212 + π * (2 * ↑k + 1)) :=
by sorry


end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2995_299559


namespace NUMINAMATH_CALUDE_tax_ratio_is_300_2001_l2995_299504

/-- Represents the lottery winnings and expenses scenario --/
structure LotteryScenario where
  winnings : ℚ
  taxRate : ℚ
  loanRate : ℚ
  savings : ℚ
  investmentRate : ℚ
  funMoney : ℚ

/-- Calculates the tax amount given a lottery scenario --/
def calculateTax (scenario : LotteryScenario) : ℚ :=
  scenario.winnings * scenario.taxRate

/-- Theorem stating that the tax ratio is 300:2001 given the specific scenario --/
theorem tax_ratio_is_300_2001 (scenario : LotteryScenario)
  (h1 : scenario.winnings = 12006)
  (h2 : scenario.loanRate = 1/3)
  (h3 : scenario.savings = 1000)
  (h4 : scenario.investmentRate = 1/5)
  (h5 : scenario.funMoney = 2802)
  (h6 : scenario.winnings * (1 - scenario.taxRate) * (1 - scenario.loanRate) - scenario.savings * (1 + scenario.investmentRate) = 2 * scenario.funMoney) :
  (calculateTax scenario) / scenario.winnings = 300 / 2001 := by
sorry

#eval 300 / 2001

end NUMINAMATH_CALUDE_tax_ratio_is_300_2001_l2995_299504


namespace NUMINAMATH_CALUDE_pencils_distribution_l2995_299555

/-- The number of students who received pencils -/
def num_students : ℕ := 12

/-- The number of pencils each student received -/
def pencils_per_student : ℕ := 3

/-- The total number of pencils given by the teacher -/
def total_pencils : ℕ := num_students * pencils_per_student

theorem pencils_distribution :
  total_pencils = 36 :=
by sorry

end NUMINAMATH_CALUDE_pencils_distribution_l2995_299555


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_even_l2995_299566

theorem sum_of_multiples_is_even (a b : ℤ) (ha : 4 ∣ a) (hb : 6 ∣ b) : Even (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_even_l2995_299566


namespace NUMINAMATH_CALUDE_saras_hourly_wage_l2995_299533

def saras_paycheck (hours_per_week : ℕ) (weeks_worked : ℕ) (tire_cost : ℕ) (money_left : ℕ) : ℚ :=
  let total_earnings := tire_cost + money_left
  let total_hours := hours_per_week * weeks_worked
  (total_earnings : ℚ) / total_hours

theorem saras_hourly_wage :
  saras_paycheck 40 2 410 510 = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_saras_hourly_wage_l2995_299533


namespace NUMINAMATH_CALUDE_parents_disagreeing_with_tuition_increase_l2995_299574

theorem parents_disagreeing_with_tuition_increase 
  (total_parents : ℕ) 
  (agree_percentage : ℚ) 
  (h1 : total_parents = 800) 
  (h2 : agree_percentage = 1/5) : 
  (1 - agree_percentage) * total_parents = 640 := by
  sorry

end NUMINAMATH_CALUDE_parents_disagreeing_with_tuition_increase_l2995_299574


namespace NUMINAMATH_CALUDE_fraction_equality_l2995_299577

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 7 / 5) 
  (h2 : r / t = 8 / 15) : 
  (5 * m * r - 2 * n * t) / (6 * n * t - 8 * m * r) = 65 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2995_299577


namespace NUMINAMATH_CALUDE_journey_distance_l2995_299536

/-- Proves that a journey with given conditions has a total distance of 224 km -/
theorem journey_distance (total_time : ℝ) (speed_first_half : ℝ) (speed_second_half : ℝ)
  (h1 : total_time = 10)
  (h2 : speed_first_half = 21)
  (h3 : speed_second_half = 24) :
  ∃ (distance : ℝ),
    distance = 224 ∧
    total_time = (distance / 2) / speed_first_half + (distance / 2) / speed_second_half :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l2995_299536


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_exists_l2995_299587

theorem geometric_arithmetic_sequence_exists : ∃ (a b : ℝ),
  1 < a ∧ a < b ∧ b < 16 ∧
  (∃ (r : ℝ), a = 1 * r ∧ b = 1 * r^2) ∧
  (∃ (d : ℝ), b = a + d ∧ 16 = b + d) ∧
  a + b = 12.64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_exists_l2995_299587


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l2995_299557

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 64*x^2 + 576 = 0 →
  x ≥ -2 * Real.sqrt 6 ∧
  (∃ y : ℝ, y^4 - 64*y^2 + 576 = 0 ∧ y = -2 * Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l2995_299557


namespace NUMINAMATH_CALUDE_simplify_fraction_l2995_299575

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2995_299575


namespace NUMINAMATH_CALUDE_volume_difference_l2995_299528

/-- The volume of space inside a sphere and outside a combined cylinder and cone -/
theorem volume_difference (r_sphere : ℝ) (r_base : ℝ) (h_cylinder : ℝ) (h_cone : ℝ) 
  (hr_sphere : r_sphere = 6)
  (hr_base : r_base = 4)
  (hh_cylinder : h_cylinder = 10)
  (hh_cone : h_cone = 5) :
  (4 / 3 * π * r_sphere^3) - (π * r_base^2 * h_cylinder + 1 / 3 * π * r_base^2 * h_cone) = 304 / 3 * π :=
sorry

end NUMINAMATH_CALUDE_volume_difference_l2995_299528


namespace NUMINAMATH_CALUDE_survey_respondents_l2995_299530

theorem survey_respondents (preferred_x : ℕ) (ratio_x : ℕ) (ratio_y : ℕ) : 
  preferred_x = 150 → ratio_x = 5 → ratio_y = 1 → 
  ∃ (total : ℕ), total = preferred_x + (preferred_x * ratio_y) / ratio_x ∧ total = 180 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l2995_299530


namespace NUMINAMATH_CALUDE_floor_double_floor_eq_42_l2995_299514

theorem floor_double_floor_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 := by
  sorry

end NUMINAMATH_CALUDE_floor_double_floor_eq_42_l2995_299514


namespace NUMINAMATH_CALUDE_minimum_value_problem_minimum_value_achievable_l2995_299544

theorem minimum_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) ≥ 4 :=
by sorry

theorem minimum_value_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) = 4 :=
by sorry

end NUMINAMATH_CALUDE_minimum_value_problem_minimum_value_achievable_l2995_299544


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2995_299550

theorem cube_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_equality : (a^3 + 8) / a = (b^3 + 8) / b ∧ (b^3 + 8) / b = (c^3 + 8) / c) : 
  a^3 + b^3 + c^3 = -24 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2995_299550


namespace NUMINAMATH_CALUDE_max_value_2a_plus_b_l2995_299546

theorem max_value_2a_plus_b (a b : ℝ) (h : 4 * a^2 + b^2 + a * b = 1) :
  2 * a + b ≤ 2 * Real.sqrt 10 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_2a_plus_b_l2995_299546


namespace NUMINAMATH_CALUDE_line_translation_distance_l2995_299580

/-- Two lines in a 2D Cartesian coordinate system -/
structure Line2D where
  slope : ℝ
  intercept : ℝ

/-- The vertical distance between two parallel lines -/
def vertical_distance (l1 l2 : Line2D) : ℝ :=
  l2.intercept - l1.intercept

/-- Theorem: The vertical distance between l1 and l2 is 6 units -/
theorem line_translation_distance :
  let l1 : Line2D := { slope := -2, intercept := -2 }
  let l2 : Line2D := { slope := -2, intercept := 4 }
  vertical_distance l1 l2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_line_translation_distance_l2995_299580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2995_299501

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_eq : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2995_299501


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l2995_299515

def total_cards : ℕ := 50
def num_range : Set ℕ := Finset.range 10
def cards_per_num : ℕ := 5
def drawn_cards : ℕ := 5

def p : ℚ := 10 / Nat.choose total_cards drawn_cards
def q : ℚ := (10 * 9 * cards_per_num * cards_per_num) / Nat.choose total_cards drawn_cards

theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l2995_299515


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2995_299593

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem decagon_diagonals : 
  num_diagonals 4 = 2 ∧ num_diagonals 5 = 5 → num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2995_299593


namespace NUMINAMATH_CALUDE_function_determination_l2995_299592

/-- Given two functions f and g with specific forms and conditions, prove they have specific expressions. -/
theorem function_determination (a b c : ℝ) : 
  let f := fun (x : ℝ) ↦ 2 * x^3 + a * x
  let g := fun (x : ℝ) ↦ b * x^2 + c
  (f 2 = 0) → 
  (g 2 = 0) → 
  (deriv f 2 = deriv g 2) →
  (f = fun (x : ℝ) ↦ 2 * x^3 - 8 * x) ∧ 
  (g = fun (x : ℝ) ↦ 4 * x^2 - 16) :=
by sorry

end NUMINAMATH_CALUDE_function_determination_l2995_299592


namespace NUMINAMATH_CALUDE_log_product_less_than_one_l2995_299590

theorem log_product_less_than_one : Real.log 9 * Real.log 11 < 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_less_than_one_l2995_299590


namespace NUMINAMATH_CALUDE_max_value_of_f_l2995_299527

-- Define the function
def f (x : ℝ) : ℝ := 12 * x - 4 * x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 11 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2995_299527


namespace NUMINAMATH_CALUDE_juggler_balls_l2995_299582

theorem juggler_balls (total_jugglers : ℕ) (total_balls : ℕ) 
  (h1 : total_jugglers = 378) 
  (h2 : total_balls = 2268) 
  (h3 : total_balls % total_jugglers = 0) : 
  total_balls / total_jugglers = 6 := by
  sorry

end NUMINAMATH_CALUDE_juggler_balls_l2995_299582


namespace NUMINAMATH_CALUDE_function_derivative_value_l2995_299523

/-- Given a function f: ℝ → ℝ such that f(x) = x^2 + 3x * f'(2) for all x ∈ ℝ,
    prove that 1 + f'(1) = -3 -/
theorem function_derivative_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x = x^2 + 3*x*(deriv f 2)) :
  1 + deriv f 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_function_derivative_value_l2995_299523


namespace NUMINAMATH_CALUDE_seven_balls_three_boxes_l2995_299572

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 3 indistinguishable boxes is 301 -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 301 := by sorry

end NUMINAMATH_CALUDE_seven_balls_three_boxes_l2995_299572


namespace NUMINAMATH_CALUDE_inequality_proof_l2995_299556

theorem inequality_proof (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  ¬(abs a + abs b > abs (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2995_299556


namespace NUMINAMATH_CALUDE_product_of_real_parts_l2995_299539

theorem product_of_real_parts (x₁ x₂ : ℂ) : 
  x₁^2 - 4*x₁ = -1 - 3*I → 
  x₂^2 - 4*x₂ = -1 - 3*I → 
  x₁ ≠ x₂ → 
  (x₁.re * x₂.re : ℝ) = (8 - Real.sqrt 6 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_l2995_299539


namespace NUMINAMATH_CALUDE_total_reimbursement_is_correct_l2995_299512

/-- Reimbursement rate for industrial clients on weekdays -/
def industrial_rate : ℚ := 36 / 100

/-- Reimbursement rate for commercial clients on weekdays -/
def commercial_rate : ℚ := 42 / 100

/-- Reimbursement rate for any clients on weekends -/
def weekend_rate : ℚ := 45 / 100

/-- Mileage for industrial clients on Monday -/
def monday_industrial : ℕ := 10

/-- Mileage for commercial clients on Monday -/
def monday_commercial : ℕ := 8

/-- Mileage for industrial clients on Tuesday -/
def tuesday_industrial : ℕ := 12

/-- Mileage for commercial clients on Tuesday -/
def tuesday_commercial : ℕ := 14

/-- Mileage for industrial clients on Wednesday -/
def wednesday_industrial : ℕ := 15

/-- Mileage for commercial clients on Wednesday -/
def wednesday_commercial : ℕ := 5

/-- Mileage for commercial clients on Thursday -/
def thursday_commercial : ℕ := 20

/-- Mileage for industrial clients on Friday -/
def friday_industrial : ℕ := 8

/-- Mileage for commercial clients on Friday -/
def friday_commercial : ℕ := 8

/-- Mileage for commercial clients on Saturday -/
def saturday_commercial : ℕ := 12

/-- Calculate the total reimbursement for the week -/
def total_reimbursement : ℚ :=
  industrial_rate * (monday_industrial + tuesday_industrial + wednesday_industrial + friday_industrial) +
  commercial_rate * (monday_commercial + tuesday_commercial + wednesday_commercial + thursday_commercial + friday_commercial) +
  weekend_rate * saturday_commercial

/-- Theorem stating that the total reimbursement is equal to $44.70 -/
theorem total_reimbursement_is_correct : total_reimbursement = 4470 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_reimbursement_is_correct_l2995_299512


namespace NUMINAMATH_CALUDE_game_theorem_l2995_299568

/-- Represents the outcome of a single round -/
inductive RoundOutcome
| OddDifference
| EvenDifference

/-- Represents the game state -/
structure GameState :=
  (playerAPoints : ℤ)
  (playerBPoints : ℤ)

/-- The game rules -/
def gameRules (n : ℕ+) (outcome : RoundOutcome) (state : GameState) : GameState :=
  match outcome with
  | RoundOutcome.OddDifference  => ⟨state.playerAPoints - 2, state.playerBPoints + 2⟩
  | RoundOutcome.EvenDifference => ⟨state.playerAPoints + n, state.playerBPoints - n⟩

/-- The probability of an odd difference in a single round -/
def probOddDifference : ℚ := 3/5

/-- The probability of an even difference in a single round -/
def probEvenDifference : ℚ := 2/5

/-- The expected value of player A's points after the game -/
def expectedValue (n : ℕ+) : ℚ := (6 * n - 18) / 5

/-- The theorem to be proved -/
theorem game_theorem (n : ℕ+) :
  (∀ m : ℕ+, m < n → expectedValue m ≤ 0) ∧
  expectedValue n > 0 ∧
  n = 4 →
  (probOddDifference^3 + 3 * probOddDifference^2 * probEvenDifference) *
  (3 * probOddDifference * probEvenDifference^2) / 
  (1 - probEvenDifference^3) = 4/13 := by
  sorry


end NUMINAMATH_CALUDE_game_theorem_l2995_299568


namespace NUMINAMATH_CALUDE_three_squares_divisible_to_not_divisible_l2995_299532

theorem three_squares_divisible_to_not_divisible (N : ℕ) :
  (∃ (n : ℕ) (a b c : ℤ), N = 9^n * (a^2 + b^2 + c^2) ∧ 3 ∣ a ∧ 3 ∣ b ∧ 3 ∣ c) →
  (∃ (k m n : ℤ), N = k^2 + m^2 + n^2 ∧ ¬(3 ∣ k) ∧ ¬(3 ∣ m) ∧ ¬(3 ∣ n)) :=
by sorry

end NUMINAMATH_CALUDE_three_squares_divisible_to_not_divisible_l2995_299532


namespace NUMINAMATH_CALUDE_primitive_root_existence_l2995_299571

theorem primitive_root_existence (p : Nat) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ g : Nat, 1 < g ∧ g < p ∧ ∀ n : Nat, n > 0 → IsPrimitiveRoot g (p^n) :=
by sorry

/- Definitions used:
Nat.Prime: Prime number predicate
IsPrimitiveRoot: Predicate for primitive root
-/

end NUMINAMATH_CALUDE_primitive_root_existence_l2995_299571


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2995_299545

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides. -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2995_299545


namespace NUMINAMATH_CALUDE_cubic_three_zeros_l2995_299554

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2 + a

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * a * x

theorem cubic_three_zeros (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  a > 1 ∨ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_l2995_299554


namespace NUMINAMATH_CALUDE_value_of_b_l2995_299561

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2995_299561


namespace NUMINAMATH_CALUDE_rectangle_areas_l2995_299579

theorem rectangle_areas (square_area : ℝ) (ratio1_width ratio1_length ratio2_width ratio2_length : ℕ) :
  square_area = 98 →
  ratio1_width = 2 →
  ratio1_length = 3 →
  ratio2_width = 3 →
  ratio2_length = 8 →
  ∃ (rect1_perim rect2_perim : ℝ),
    4 * Real.sqrt square_area = rect1_perim + rect2_perim ∧
    (rect1_perim * ratio1_width * rect1_perim * ratio1_length) / ((ratio1_width + ratio1_length) ^ 2) =
    (rect2_perim * ratio2_width * rect2_perim * ratio2_length) / ((ratio2_width + ratio2_length) ^ 2) →
  (rect1_perim * ratio1_width * rect1_perim * ratio1_length) / ((ratio1_width + ratio1_length) ^ 2) = 64 / 3 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_areas_l2995_299579


namespace NUMINAMATH_CALUDE_manuscript_pages_count_l2995_299500

/-- The cost structure and revision information for a manuscript typing service. -/
structure ManuscriptTyping where
  first_time_cost : ℕ
  revision_cost : ℕ
  pages_revised_once : ℕ
  pages_revised_twice : ℕ
  total_cost : ℕ

/-- Calculates the total number of pages in a manuscript given the typing costs and revision information. -/
def total_pages (m : ManuscriptTyping) : ℕ :=
  (m.total_cost - (m.pages_revised_once * (m.first_time_cost + m.revision_cost) + 
   m.pages_revised_twice * (m.first_time_cost + 2 * m.revision_cost))) / m.first_time_cost + 
   m.pages_revised_once + m.pages_revised_twice

/-- Theorem stating that for the given manuscript typing scenario, the total number of pages is 100. -/
theorem manuscript_pages_count (m : ManuscriptTyping) 
  (h1 : m.first_time_cost = 6)
  (h2 : m.revision_cost = 4)
  (h3 : m.pages_revised_once = 35)
  (h4 : m.pages_revised_twice = 15)
  (h5 : m.total_cost = 860) :
  total_pages m = 100 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_pages_count_l2995_299500


namespace NUMINAMATH_CALUDE_tangent_line_to_two_curves_l2995_299553

/-- A line y = kx + t is tangent to both curves y = exp x + 2 and y = exp (x + 1) -/
theorem tangent_line_to_two_curves (k t : ℝ) : 
  (∃ x₁ : ℝ, k * x₁ + t = Real.exp x₁ + 2 ∧ k = Real.exp x₁) →
  (∃ x₂ : ℝ, k * x₂ + t = Real.exp (x₂ + 1) ∧ k = Real.exp (x₂ + 1)) →
  t = 4 - 2 * Real.log 2 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_to_two_curves_l2995_299553


namespace NUMINAMATH_CALUDE_sally_bought_twenty_cards_l2995_299534

/-- Calculates the number of Pokemon cards Sally bought -/
def cards_sally_bought (initial : ℕ) (from_dan : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial + from_dan)

/-- Proves that Sally bought 20 Pokemon cards -/
theorem sally_bought_twenty_cards : 
  cards_sally_bought 27 41 88 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_twenty_cards_l2995_299534


namespace NUMINAMATH_CALUDE_prob_exactly_one_red_l2995_299594

structure Box where
  red : ℕ
  black : ℕ

def Box.total (b : Box) : ℕ := b.red + b.black

def prob_red (b : Box) : ℚ := b.red / b.total

def prob_black (b : Box) : ℚ := b.black / b.total

def box_A : Box := ⟨1, 2⟩
def box_B : Box := ⟨2, 2⟩

theorem prob_exactly_one_red : 
  prob_red box_A * prob_black box_B + prob_black box_A * prob_red box_B = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_exactly_one_red_l2995_299594


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l2995_299503

/-- Given an equilateral cone with an inscribed sphere of volume 100 cm³,
    the lateral surface area of the cone is 6π * ∛(5625/π²) cm² -/
theorem cone_lateral_surface_area (v : ℝ) (r : ℝ) (l : ℝ) (P : ℝ) :
  v = 100 →  -- volume of the sphere
  v = (4/3) * π * r^3 →  -- volume formula of a sphere
  l = 2 * Real.sqrt 3 * (75/π)^(1/3) →  -- side length of the cone
  P = 6 * π * ((5625:ℝ)/π^2)^(1/3) →  -- lateral surface area of the cone
  P = 6 * π * ((75:ℝ)^2/π^2)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l2995_299503


namespace NUMINAMATH_CALUDE_nancy_balloons_l2995_299538

theorem nancy_balloons (mary_balloons : ℕ) (nancy_balloons : ℕ) : 
  mary_balloons = 28 → 
  mary_balloons = 4 * nancy_balloons → 
  nancy_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_nancy_balloons_l2995_299538


namespace NUMINAMATH_CALUDE_new_hires_count_l2995_299584

theorem new_hires_count (initial_workers : ℕ) (initial_men_ratio : ℚ) (final_women_percentage : ℚ) : 
  initial_workers = 90 →
  initial_men_ratio = 2/3 →
  final_women_percentage = 40/100 →
  ∃ (new_hires : ℕ), 
    (initial_workers * (1 - initial_men_ratio) + new_hires) / (initial_workers + new_hires) = final_women_percentage ∧
    new_hires = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_hires_count_l2995_299584


namespace NUMINAMATH_CALUDE_correct_calculation_l2995_299524

theorem correct_calculation (x : ℤ) (h : x - 48 = 52) : x + 48 = 148 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2995_299524


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l2995_299506

theorem trigonometric_inequality : 
  let a := (1/2) * Real.cos (6 * π / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * π / 180)
  let b := 2 * Real.sin (13 * π / 180) * Real.cos (13 * π / 180)
  let c := Real.sqrt ((1 - Real.cos (50 * π / 180)) / 2)
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l2995_299506


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2995_299565

theorem smallest_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 6 = 5) ∧ 
  (∀ m : ℕ, m > 0 → m % 4 = 3 → m % 6 = 5 → n ≤ m) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2995_299565


namespace NUMINAMATH_CALUDE_school_visit_arrangements_l2995_299569

/-- Represents the number of days available for scheduling -/
def num_days : ℕ := 5

/-- Represents the number of schools to be scheduled -/
def num_schools : ℕ := 3

/-- Calculates the number of permutations of r items chosen from n items -/
def permutations (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.factorial n / Nat.factorial (n - r)

/-- Calculates the number of valid arrangements for the school visits -/
def count_arrangements : ℕ :=
  permutations 4 2 + permutations 3 2 + permutations 2 2

/-- Theorem stating that the number of valid arrangements is 20 -/
theorem school_visit_arrangements :
  count_arrangements = 20 :=
sorry

end NUMINAMATH_CALUDE_school_visit_arrangements_l2995_299569


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_with_conditions_l2995_299589

theorem greatest_three_digit_number_with_conditions : ∃ n : ℕ, 
  (n ≤ 999 ∧ n ≥ 100) ∧ 
  (∃ k : ℕ, n = 7 * k + 2) ∧ 
  (∃ m : ℕ, n = 6 * m + 4) ∧
  (∀ x : ℕ, (x ≤ 999 ∧ x ≥ 100) → 
    (∃ a : ℕ, x = 7 * a + 2) → 
    (∃ b : ℕ, x = 6 * b + 4) → 
    x ≤ n) ∧
  n = 994 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_with_conditions_l2995_299589


namespace NUMINAMATH_CALUDE_absolute_value_equality_l2995_299581

theorem absolute_value_equality (x : ℚ) :
  (|x + 3| = |x - 4|) ↔ (x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l2995_299581


namespace NUMINAMATH_CALUDE_s_range_l2995_299588

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem s_range :
  {y : ℝ | ∃ x ≠ 2, s x = y} = {y : ℝ | y < 0 ∨ y > 0} :=
by sorry

end NUMINAMATH_CALUDE_s_range_l2995_299588


namespace NUMINAMATH_CALUDE_line_intercept_sum_l2995_299563

/-- Given a line 3x + 5y + c = 0 where the sum of its x-intercept and y-intercept is 16, prove that c = -30 -/
theorem line_intercept_sum (c : ℝ) : 
  (∃ (x y : ℝ), 3 * x + 5 * y + c = 0 ∧ x + y = 16) → c = -30 := by
  sorry

end NUMINAMATH_CALUDE_line_intercept_sum_l2995_299563


namespace NUMINAMATH_CALUDE_gcd_13013_15015_l2995_299509

theorem gcd_13013_15015 : Nat.gcd 13013 15015 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_gcd_13013_15015_l2995_299509


namespace NUMINAMATH_CALUDE_circle_equation_symmetric_center_l2995_299507

/-- A circle C in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The standard equation of a circle. -/
def standardEquation (c : Circle) : ℝ → ℝ → Prop :=
  λ x y => (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Symmetry of two points about the line y = x. -/
def symmetricAboutDiagonal (p q : ℝ × ℝ) : Prop :=
  p.1 + q.2 = p.2 + q.1 ∧ p.1 + p.2 = q.1 + q.2

theorem circle_equation_symmetric_center (c : Circle) :
  c.radius = 1 →
  symmetricAboutDiagonal c.center (1, 0) →
  standardEquation c = λ x y => x^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_symmetric_center_l2995_299507


namespace NUMINAMATH_CALUDE_r_to_s_conversion_l2995_299541

/-- Given a linear relationship between r and s scales, prove that r = 48 corresponds to s = 100 -/
theorem r_to_s_conversion (r s : ℝ → ℝ) : 
  (∃ a b : ℝ, ∀ x, s x = a * x + b) →  -- Linear relationship
  s 6 = 30 →                          -- First given point
  s 24 = 60 →                         -- Second given point
  s 48 = 100 :=                       -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_r_to_s_conversion_l2995_299541


namespace NUMINAMATH_CALUDE_checkerboard_partition_l2995_299585

theorem checkerboard_partition (n : ℕ) : 
  n % 5 = 0 → n % 7 = 0 → n ≤ 200 → n % 6 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_checkerboard_partition_l2995_299585


namespace NUMINAMATH_CALUDE_mocktail_lime_cost_l2995_299599

/-- Represents the cost of limes in dollars for a given number of limes -/
def lime_cost (num_limes : ℕ) : ℚ :=
  (num_limes : ℚ) / 3

/-- Calculates the number of limes needed for a given number of days -/
def limes_needed (days : ℕ) : ℕ :=
  (days + 1) / 2

theorem mocktail_lime_cost : lime_cost (limes_needed 30) = 5 := by
  sorry

end NUMINAMATH_CALUDE_mocktail_lime_cost_l2995_299599
