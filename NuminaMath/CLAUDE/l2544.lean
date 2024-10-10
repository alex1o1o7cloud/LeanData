import Mathlib

namespace integer_roots_of_polynomial_l2544_254493

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x - 11

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | polynomial a₂ a₁ x = 0} ⊆ {-11, -1, 1, 11} :=
by sorry

end integer_roots_of_polynomial_l2544_254493


namespace relationship_p_q_l2544_254497

theorem relationship_p_q (k : ℝ) : ∃ (p₀ q₀ p₁ : ℝ),
  (p₀ * q₀^2 = k) ∧ 
  (p₀ = 16) ∧ 
  (q₀ = 4) ∧ 
  (p₁ * 8^2 = k) → 
  p₁ = 4 := by
sorry

end relationship_p_q_l2544_254497


namespace line_slope_l2544_254421

theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/4) := by
  sorry

end line_slope_l2544_254421


namespace expression_evaluations_l2544_254408

theorem expression_evaluations :
  (3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3) ^ 0 + |Real.sqrt 3 - 2| = 3) ∧
  ((3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / Real.sqrt 3 = 28/3) := by
  sorry

end expression_evaluations_l2544_254408


namespace trees_after_typhoon_l2544_254492

/-- The number of trees Haley initially grew -/
def initial_trees : ℕ := 17

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 5

/-- Theorem stating that the number of trees left after the typhoon is 12 -/
theorem trees_after_typhoon : initial_trees - dead_trees = 12 := by
  sorry

end trees_after_typhoon_l2544_254492


namespace bobs_grade_l2544_254488

theorem bobs_grade (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = jason_grade / 2 →
  bob_grade = 35 := by
  sorry

end bobs_grade_l2544_254488


namespace expression_equality_l2544_254467

theorem expression_equality (x : ℝ) : 
  (3*x + 1)^2 + 2*(3*x + 1)*(x - 3) + (x - 3)^2 = 16*x^2 - 16*x + 4 := by
  sorry

end expression_equality_l2544_254467


namespace laura_shopping_cost_l2544_254438

/-- Calculates the total cost of Laura's shopping trip given the prices and quantities of items. -/
def shopping_cost (salad_price : ℚ) (juice_price : ℚ) : ℚ :=
  let beef_price := 2 * salad_price
  let potato_price := salad_price / 3
  let mixed_veg_price := beef_price / 2 + 0.5
  let tomato_sauce_price := salad_price * 3 / 4
  let pasta_price := juice_price + mixed_veg_price
  2 * salad_price +
  2 * beef_price +
  1 * potato_price +
  2 * juice_price +
  3 * mixed_veg_price +
  5 * tomato_sauce_price +
  4 * pasta_price

theorem laura_shopping_cost :
  shopping_cost 3 1.5 = 63.75 := by
  sorry

end laura_shopping_cost_l2544_254438


namespace equidistant_arrangement_exists_l2544_254454

/-- A move on a circular track -/
structure Move where
  person1 : Fin n
  person2 : Fin n
  distance : ℝ

/-- The state of people on a circular track -/
def TrackState (n : ℕ) := Fin n → ℝ

/-- Apply a move to a track state -/
def applyMove (state : TrackState n) (move : Move) : TrackState n :=
  fun i => if i = move.person1 then state i + move.distance
           else if i = move.person2 then state i - move.distance
           else state i

/-- Check if a track state is equidistant -/
def isEquidistant (state : TrackState n) : Prop :=
  ∀ i j : Fin n, (state i - state j) % 1 = (i - j : ℝ) / n

/-- Main theorem: it's possible to reach an equidistant state in at most n-1 moves -/
theorem equidistant_arrangement_exists (n : ℕ) (initial : TrackState n) :
  ∃ (moves : List Move), moves.length ≤ n - 1 ∧
    isEquidistant (moves.foldl applyMove initial) :=
sorry

end equidistant_arrangement_exists_l2544_254454


namespace probability_three_correct_is_one_sixth_l2544_254461

/-- The probability of exactly 3 out of 5 packages being delivered to the correct houses in a random delivery -/
def probability_three_correct_deliveries : ℚ :=
  (Nat.choose 5 3 * 2) / Nat.factorial 5

/-- Theorem stating that the probability of exactly 3 out of 5 packages being delivered to the correct houses is 1/6 -/
theorem probability_three_correct_is_one_sixth :
  probability_three_correct_deliveries = 1 / 6 := by
  sorry


end probability_three_correct_is_one_sixth_l2544_254461


namespace total_salary_is_7600_l2544_254473

/-- Represents the weekly working hours for each employee -/
structure WeeklyHours where
  fiona : ℕ
  john : ℕ
  jeremy : ℕ

/-- Represents the hourly wage -/
def hourlyWage : ℚ := 20

/-- Represents the number of weeks in a month -/
def weeksPerMonth : ℕ := 4

/-- Calculates the monthly salary for an employee -/
def monthlySalary (hours : ℕ) : ℚ :=
  hours * hourlyWage * weeksPerMonth

/-- Calculates the total monthly expenditure on salaries -/
def totalMonthlyExpenditure (hours : WeeklyHours) : ℚ :=
  monthlySalary hours.fiona + monthlySalary hours.john + monthlySalary hours.jeremy

/-- Theorem stating that the total monthly expenditure on salaries is $7600 -/
theorem total_salary_is_7600 (hours : WeeklyHours)
    (h1 : hours.fiona = 40)
    (h2 : hours.john = 30)
    (h3 : hours.jeremy = 25) :
    totalMonthlyExpenditure hours = 7600 := by
  sorry

end total_salary_is_7600_l2544_254473


namespace intersection_M_N_l2544_254432

def M : Set ℝ := {x | (x + 2) * (x - 2) > 0}
def N : Set ℝ := {-3, -2, 2, 3, 4}

theorem intersection_M_N : M ∩ N = {-3, 3, 4} := by sorry

end intersection_M_N_l2544_254432


namespace unique_set_A_l2544_254448

def M : Set ℤ := {1, 3, 5, 7, 9}

theorem unique_set_A : ∃! A : Set ℤ, A.Nonempty ∧ 
  (∀ a ∈ A, a + 4 ∈ M) ∧ 
  (∀ a ∈ A, a - 4 ∈ M) ∧
  A = {5} := by sorry

end unique_set_A_l2544_254448


namespace functional_equation_solution_l2544_254489

theorem functional_equation_solution (f : ℕ → ℕ) 
  (h : ∀ x y : ℕ, f (x + y) = f x + f y) : 
  ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
sorry

end functional_equation_solution_l2544_254489


namespace train_pass_bridge_time_l2544_254445

/-- Time for a train to pass a bridge -/
theorem train_pass_bridge_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 860)
  (h2 : train_speed_kmh = 85)
  (h3 : bridge_length = 450) :
  ∃ (t : ℝ), abs (t - 55.52) < 0.01 ∧ 
  t = (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) :=
sorry

end train_pass_bridge_time_l2544_254445


namespace jelly_bean_probability_l2544_254411

theorem jelly_bean_probability (p_red p_orange p_yellow p_green : ℝ) :
  p_red = 0.15 →
  p_orange = 0.35 →
  p_yellow = 0.2 →
  p_red + p_orange + p_yellow + p_green = 1 →
  p_green = 0.3 := by
sorry

end jelly_bean_probability_l2544_254411


namespace smallest_fraction_divisible_l2544_254440

theorem smallest_fraction_divisible (f1 f2 f3 : Rat) (h1 : f1 = 6/7) (h2 : f2 = 5/14) (h3 : f3 = 10/21) :
  (∀ q : Rat, (∃ n1 n2 n3 : ℤ, f1 * q = n1 ∧ f2 * q = n2 ∧ f3 * q = n3) →
    (1 : Rat) / 42 ≤ q) ∧
  (∃ n1 n2 n3 : ℤ, f1 * (1/42 : Rat) = n1 ∧ f2 * (1/42 : Rat) = n2 ∧ f3 * (1/42 : Rat) = n3) :=
by sorry

end smallest_fraction_divisible_l2544_254440


namespace equation_solution_l2544_254433

theorem equation_solution : 
  ∃ x : ℝ, (3*x - 5) / (x^2 - 7*x + 12) + (5*x - 1) / (x^2 - 5*x + 6) = (8*x - 13) / (x^2 - 6*x + 8) ∧ x = 5 :=
by
  sorry

end equation_solution_l2544_254433


namespace solve_equation_l2544_254406

theorem solve_equation (y : ℝ) : 7 - y = 10 → y = -3 := by
  sorry

end solve_equation_l2544_254406


namespace percentage_6_plus_years_l2544_254429

-- Define the number of marks for each year range
def marks : List Nat := [10, 4, 6, 5, 8, 3, 5, 4, 2, 2]

-- Define the total number of marks
def total_marks : Nat := marks.sum

-- Define the number of marks for 6 years or more
def marks_6_plus : Nat := (marks.drop 6).sum

-- Theorem to prove
theorem percentage_6_plus_years (ε : Real) (hε : ε > 0) :
  ∃ (p : Real), abs (p - 26.53) < ε ∧ p = (marks_6_plus * 100 : Real) / total_marks :=
sorry

end percentage_6_plus_years_l2544_254429


namespace symmetric_point_coordinates_l2544_254490

/-- A point in the second quadrant with |x| = 2 and |y| = 3 -/
structure PointM where
  x : ℝ
  y : ℝ
  second_quadrant : x < 0 ∧ y > 0
  abs_x_eq_two : |x| = 2
  abs_y_eq_three : |y| = 3

/-- The coordinates of a point symmetric to M with respect to the y-axis -/
def symmetric_point (m : PointM) : ℝ × ℝ := (-m.x, m.y)

theorem symmetric_point_coordinates (m : PointM) : 
  symmetric_point m = (2, 3) := by sorry

end symmetric_point_coordinates_l2544_254490


namespace intersection_equals_interval_l2544_254458

-- Define sets A and B
def A : Set ℝ := {x : ℝ | |x| > 4}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x ≤ 6}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_equals_interval : A_intersect_B = Set.Ioo 4 6 := by sorry

end intersection_equals_interval_l2544_254458


namespace unique_n_value_l2544_254416

/-- Represents a round-robin golf tournament with the given conditions -/
structure GolfTournament where
  /-- Total number of players -/
  T : ℕ
  /-- Number of points scored by each player other than Simon and Garfunkle -/
  n : ℕ
  /-- Condition: Total number of matches equals total points distributed -/
  matches_eq_points : T * (T - 1) / 2 = 16 + n * (T - 2)
  /-- Condition: Tournament has at least 3 players -/
  min_players : T ≥ 3

/-- Theorem stating that the only possible value for n is 17 -/
theorem unique_n_value (tournament : GolfTournament) : tournament.n = 17 := by
  sorry

end unique_n_value_l2544_254416


namespace floor_sqrt_23_squared_l2544_254491

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end floor_sqrt_23_squared_l2544_254491


namespace two_digit_number_value_l2544_254444

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

/-- Theorem: The value of a two-digit number is 10a + b, where a is the tens digit and b is the ones digit -/
theorem two_digit_number_value (n : TwoDigitNumber) : 
  n.value = 10 * n.tens + n.ones := by sorry

end two_digit_number_value_l2544_254444


namespace fixed_point_satisfies_function_l2544_254459

/-- A linear function of the form y = kx + k + 2 -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + k + 2

/-- The fixed point of the linear function -/
def fixedPoint : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the linear function for all k -/
theorem fixed_point_satisfies_function :
  ∀ k : ℝ, linearFunction k (fixedPoint.1) = fixedPoint.2 := by
  sorry

#check fixed_point_satisfies_function

end fixed_point_satisfies_function_l2544_254459


namespace probability_purple_marble_l2544_254430

theorem probability_purple_marble (blue_prob green_prob : ℝ) 
  (h1 : blue_prob = 0.3)
  (h2 : green_prob = 0.4)
  (h3 : ∃ purple_prob : ℝ, blue_prob + green_prob + purple_prob = 1) :
  ∃ purple_prob : ℝ, purple_prob = 0.3 ∧ blue_prob + green_prob + purple_prob = 1 :=
by
  sorry

end probability_purple_marble_l2544_254430


namespace problem_statement_l2544_254435

theorem problem_statement (a b c t : ℝ) 
  (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : t ≥ 1)
  (h5 : a + b + c = 1/2)
  (h6 : Real.sqrt (a + 1/2 * (b - c)^2) + Real.sqrt b + Real.sqrt c = Real.sqrt (6*t) / 2) :
  a^(2*t) + b^(2*t) + c^(2*t) = 1/12 := by
sorry

end problem_statement_l2544_254435


namespace exam_questions_unique_solution_l2544_254452

theorem exam_questions_unique_solution (n : ℕ) : 
  (15 + (n - 20) / 3 : ℚ) / n = 1 / 2 → n = 50 :=
by sorry

end exam_questions_unique_solution_l2544_254452


namespace rectangle_diagonal_maximum_l2544_254420

theorem rectangle_diagonal_maximum (l w : ℝ) : 
  (2 * l + 2 * w = 40) → 
  (∀ l' w' : ℝ, (2 * l' + 2 * w' = 40) → (l'^2 + w'^2 ≤ l^2 + w^2)) →
  l^2 + w^2 = 200 :=
sorry

end rectangle_diagonal_maximum_l2544_254420


namespace inequality_theorem_l2544_254443

theorem inequality_theorem (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : 1/p + 1/q^2 = 1) : 
  1/(p*(p+2)) + 1/(q*(q+2)) ≥ (21*Real.sqrt 21 - 71)/80 ∧
  (1/(p*(p+2)) + 1/(q*(q+2)) = (21*Real.sqrt 21 - 71)/80 ↔ 
    p = 2 + 2*Real.sqrt (7/3) ∧ q = (Real.sqrt 21 + 1)/5) :=
by sorry

end inequality_theorem_l2544_254443


namespace sum_of_coefficients_l2544_254499

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end sum_of_coefficients_l2544_254499


namespace function_property_l2544_254419

/-- Given a function f(x) = ax^3 - bx^(3/5) + 1, if f(-1) = 3, then f(1) = 1 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x^(3/5) + 1
  f (-1) = 3 → f 1 = 1 := by
  sorry

end function_property_l2544_254419


namespace percentage_subtracted_l2544_254472

theorem percentage_subtracted (a : ℝ) (h : ∃ p : ℝ, a - p * a = 0.97 * a) : 
  ∃ p : ℝ, p = 0.03 ∧ a - p * a = 0.97 * a :=
sorry

end percentage_subtracted_l2544_254472


namespace tommys_quarters_l2544_254434

/-- Tommy's coin collection problem -/
theorem tommys_quarters (P D N Q : ℕ) 
  (dimes_pennies : D = P + 10)
  (nickels_dimes : N = 2 * D)
  (pennies_quarters : P = 10 * Q)
  (total_nickels : N = 100) : Q = 4 := by
  sorry

end tommys_quarters_l2544_254434


namespace percentage_relation_l2544_254450

theorem percentage_relation (x y z : ℝ) (h1 : y = 0.6 * z) (h2 : x = 0.78 * z) :
  x = y * (1 + 0.3) :=
by sorry

end percentage_relation_l2544_254450


namespace polynomial_root_relation_l2544_254466

/-- Given real numbers a, b, and c, and polynomials g and f as defined,
    prove that f(2) = 40640 -/
theorem polynomial_root_relation (a b c : ℝ) : 
  let g := fun (x : ℝ) => x^3 + a*x^2 + x + 20
  let f := fun (x : ℝ) => x^4 + x^3 + b*x^2 + 50*x + c
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ g x = 0 ∧ g y = 0 ∧ g z = 0) →
  (∀ (x : ℝ), g x = 0 → f x = 0) →
  f 2 = 40640 := by
  sorry

end polynomial_root_relation_l2544_254466


namespace smallest_non_factor_product_l2544_254476

theorem smallest_non_factor_product (a b : ℕ+) : 
  a ≠ b →
  a ∣ 48 →
  b ∣ 48 →
  ¬(a * b ∣ 48) →
  (∀ (c d : ℕ+), c ≠ d → c ∣ 48 → d ∣ 48 → ¬(c * d ∣ 48) → a * b ≤ c * d) →
  a * b = 18 :=
sorry

end smallest_non_factor_product_l2544_254476


namespace trebled_resultant_l2544_254480

theorem trebled_resultant (x : ℕ) : x = 20 → 3 * ((2 * x) + 5) = 135 := by
  sorry

end trebled_resultant_l2544_254480


namespace matrix_vector_multiplication_l2544_254465

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; -3, 4]
def v : Matrix (Fin 2) (Fin 1) ℝ := !![3; -1]

theorem matrix_vector_multiplication :
  A * v = !![7; -13] := by sorry

end matrix_vector_multiplication_l2544_254465


namespace horner_v1_equals_22_l2544_254441

/-- Horner's Method for polynomial evaluation -/
def horner_step (coeff : ℝ) (x : ℝ) (prev : ℝ) : ℝ :=
  prev * x + coeff

/-- The polynomial f(x) = 4x⁵ + 2x⁴ + 3.5x³ - 2.6x² + 1.7x - 0.8 -/
def f (x : ℝ) : ℝ :=
  4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- Theorem: The value of V₁ when calculating f(5) using Horner's Method is 22 -/
theorem horner_v1_equals_22 :
  let v0 := 4  -- Initialize V₀ with the coefficient of the highest degree term
  let v1 := horner_step 2 5 v0  -- Calculate V₁
  v1 = 22 := by sorry

end horner_v1_equals_22_l2544_254441


namespace reading_time_proof_l2544_254479

/-- Calculates the number of weeks needed to read a series of books -/
def weeks_to_read (total_books : ℕ) (first_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  1 + (total_books - first_week + subsequent_weeks - 1) / subsequent_weeks

/-- Proves that reading 70 books takes 11 weeks when reading 5 books in the first week and 7 books per week thereafter -/
theorem reading_time_proof :
  weeks_to_read 70 5 7 = 11 := by
  sorry

end reading_time_proof_l2544_254479


namespace arithmetic_expression_evaluation_l2544_254405

theorem arithmetic_expression_evaluation :
  (80 / 16) + (100 / 25) + ((6^2) * 3) - 300 - ((324 / 9) * 2) = -255 := by
  sorry

end arithmetic_expression_evaluation_l2544_254405


namespace negation_equivalence_l2544_254446

theorem negation_equivalence (a b : ℝ) : 
  (¬(a + b = 1 → a^2 + b^2 > 1)) ↔ (∃ a b : ℝ, a + b = 1 ∧ a^2 + b^2 ≤ 1) :=
sorry

end negation_equivalence_l2544_254446


namespace quadratic_no_real_roots_l2544_254424

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 3 ≠ 0 := by
sorry

end quadratic_no_real_roots_l2544_254424


namespace geese_count_l2544_254431

/-- Given a marsh with ducks and geese, calculate the number of geese -/
theorem geese_count (total_birds ducks : ℕ) (h1 : total_birds = 95) (h2 : ducks = 37) :
  total_birds - ducks = 58 := by
  sorry

end geese_count_l2544_254431


namespace certain_number_value_l2544_254453

theorem certain_number_value (a : ℝ) (x : ℝ) 
  (h1 : -6 * a^2 = x * (4 * a + 2)) 
  (h2 : -6 * 1^2 = x * (4 * 1 + 2)) : 
  x = -1 := by sorry

end certain_number_value_l2544_254453


namespace number_exists_l2544_254427

theorem number_exists : ∃ N : ℝ, (N / 10 - N / 1000) = 700 := by
  sorry

end number_exists_l2544_254427


namespace rhombus_area_l2544_254464

/-- The area of a rhombus with specific side length and diagonal difference -/
theorem rhombus_area (side : ℝ) (diag_diff : ℝ) (area : ℝ) : 
  side = Real.sqrt 113 →
  diag_diff = 8 →
  area = 6 * Real.sqrt 210 - 12 →
  ∃ (d1 d2 : ℝ), 
    d1 > 0 ∧ d2 > 0 ∧
    d2 - d1 = diag_diff ∧
    d1 * d2 / 2 = area ∧
    d1^2 / 4 + side^2 = (d2 / 2)^2 :=
by sorry

end rhombus_area_l2544_254464


namespace continuity_at_two_l2544_254449

noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x^2 - 4)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 2| < ε :=
by
  sorry

end continuity_at_two_l2544_254449


namespace fixed_point_exponential_function_l2544_254410

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by
  sorry

end fixed_point_exponential_function_l2544_254410


namespace selene_purchase_l2544_254423

theorem selene_purchase (camera_price : ℝ) (frame_price : ℝ) (discount_rate : ℝ) (total_paid : ℝ) :
  camera_price = 110 →
  frame_price = 120 →
  discount_rate = 0.05 →
  total_paid = 551 →
  ∃ num_frames : ℕ,
    (1 - discount_rate) * (2 * camera_price + num_frames * frame_price) = total_paid ∧
    num_frames = 3 :=
by sorry

end selene_purchase_l2544_254423


namespace probability_red_ball_specific_l2544_254417

/-- The probability of drawing a red ball from a bag with specified ball counts. -/
def probability_red_ball (red_count black_count white_count : ℕ) : ℚ :=
  red_count / (red_count + black_count + white_count)

/-- Theorem: The probability of drawing a red ball from a bag with 3 red balls,
    5 black balls, and 4 white balls is 1/4. -/
theorem probability_red_ball_specific : probability_red_ball 3 5 4 = 1/4 := by
  sorry

end probability_red_ball_specific_l2544_254417


namespace students_liking_both_pizza_and_burgers_l2544_254477

theorem students_liking_both_pizza_and_burgers 
  (total : ℕ) 
  (pizza : ℕ) 
  (burgers : ℕ) 
  (neither : ℕ) 
  (h1 : total = 50) 
  (h2 : pizza = 22) 
  (h3 : burgers = 20) 
  (h4 : neither = 14) : 
  pizza + burgers - (total - neither) = 6 := by
sorry

end students_liking_both_pizza_and_burgers_l2544_254477


namespace painted_cube_probability_l2544_254425

/-- The size of the cube's side -/
def cube_side : ℕ := 5

/-- The total number of unit cubes in the larger cube -/
def total_cubes : ℕ := cube_side ^ 3

/-- The number of unit cubes with exactly three painted faces -/
def three_painted_faces : ℕ := 1

/-- The number of unit cubes with no painted faces -/
def no_painted_faces : ℕ := (cube_side - 2) ^ 3

/-- The number of ways to choose two cubes out of the total -/
def total_combinations : ℕ := total_cubes.choose 2

/-- The number of successful outcomes -/
def successful_outcomes : ℕ := three_painted_faces * no_painted_faces

theorem painted_cube_probability :
  (successful_outcomes : ℚ) / total_combinations = 9 / 2583 := by
  sorry

end painted_cube_probability_l2544_254425


namespace longest_wait_time_l2544_254436

def initial_wait : ℕ := 20

def license_renewal_wait (t : ℕ) : ℕ := 2 * t + 8

def registration_update_wait (t : ℕ) : ℕ := 4 * t + 14

def driving_record_wait (t : ℕ) : ℕ := 3 * t - 16

theorem longest_wait_time :
  let tasks := [initial_wait,
                license_renewal_wait initial_wait,
                registration_update_wait initial_wait,
                driving_record_wait initial_wait]
  registration_update_wait initial_wait = 94 ∧
  ∀ t ∈ tasks, t ≤ registration_update_wait initial_wait :=
by sorry

end longest_wait_time_l2544_254436


namespace square_cut_into_three_rectangles_l2544_254487

theorem square_cut_into_three_rectangles (square_side : ℝ) (cut_length : ℝ) : 
  square_side = 36 →
  ∃ (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℝ),
    -- The three rectangles have equal areas
    rect1_width * rect1_height = rect2_width * rect2_height ∧
    rect2_width * rect2_height = rect3_width * rect3_height ∧
    -- The rectangles fit within the square
    rect1_width + rect2_width ≤ square_side ∧
    rect1_height ≤ square_side ∧
    rect2_height ≤ square_side ∧
    rect3_width ≤ square_side ∧
    rect3_height ≤ square_side ∧
    -- The rectangles have common boundaries
    (rect1_width = rect2_width ∨ rect1_height = rect2_height) ∧
    (rect2_width = rect3_width ∨ rect2_height = rect3_height) ∧
    (rect1_width = rect3_width ∨ rect1_height = rect3_height) →
  cut_length = 60 :=
by sorry

end square_cut_into_three_rectangles_l2544_254487


namespace octagon_diagonals_l2544_254495

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon has 8 sides -/
def octagon_sides : ℕ := 8

theorem octagon_diagonals : num_diagonals octagon_sides = 20 := by
  sorry

end octagon_diagonals_l2544_254495


namespace division_remainder_problem_l2544_254414

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1365) (h2 : a = 1620)
  (h3 : ∃ (q : ℕ), q = 6 ∧ a = q * b + (a % b) ∧ a % b < b) : a % b = 90 := by
  sorry

end division_remainder_problem_l2544_254414


namespace egg_count_and_weight_l2544_254422

/-- Conversion factor from ounces to grams -/
def ouncesToGrams : ℝ := 28.3495

/-- Initial number of eggs -/
def initialEggs : ℕ := 47

/-- Number of whole eggs added -/
def addedEggs : ℕ := 5

/-- Total weight of eggs in ounces -/
def totalWeightOunces : ℝ := 143.5

theorem egg_count_and_weight :
  (initialEggs + addedEggs = 52) ∧
  (abs (totalWeightOunces * ouncesToGrams - 4067.86) < 0.01) :=
by sorry

end egg_count_and_weight_l2544_254422


namespace theresa_final_week_hours_l2544_254494

/-- The number of weeks Theresa needs to work -/
def total_weeks : ℕ := 6

/-- The required average number of hours per week -/
def required_average : ℚ := 10

/-- The list of hours worked in the first 5 weeks -/
def hours_worked : List ℚ := [8, 11, 7, 12, 10]

/-- The sum of hours worked in the first 5 weeks -/
def sum_first_five : ℚ := hours_worked.sum

/-- The number of hours Theresa needs to work in the final week -/
def hours_final_week : ℚ := required_average * total_weeks - sum_first_five

theorem theresa_final_week_hours :
  hours_final_week = 12 := by sorry

end theresa_final_week_hours_l2544_254494


namespace election_votes_calculation_l2544_254418

theorem election_votes_calculation (total_votes : ℕ) : 
  (4 : ℕ) ≤ total_votes ∧ 
  (total_votes : ℚ) * (1/2) - (total_votes : ℚ) * (1/4) = 174 →
  total_votes = 696 := by
sorry

end election_votes_calculation_l2544_254418


namespace complex_equation_solution_l2544_254455

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : z = -1 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l2544_254455


namespace circle_center_in_third_quadrant_l2544_254474

/-- A line passes through the first, second, and third quadrants -/
structure LineInQuadrants (a b : ℝ) : Prop :=
  (passes_through_123 : a > 0 ∧ b > 0)

/-- A circle with center (-a, -b) and radius r -/
structure Circle (a b r : ℝ) : Prop :=
  (positive_radius : r > 0)

/-- The third quadrant -/
def ThirdQuadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

theorem circle_center_in_third_quadrant
  (a b r : ℝ) (line : LineInQuadrants a b) (circle : Circle a b r) :
  ThirdQuadrant (-a) (-b) :=
sorry

end circle_center_in_third_quadrant_l2544_254474


namespace age_problem_l2544_254486

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 27 → 
  b = 10 := by
sorry

end age_problem_l2544_254486


namespace ice_cream_sundaes_l2544_254403

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  (n : ℕ) + n.choose 2 = 36 :=
by sorry

end ice_cream_sundaes_l2544_254403


namespace evaluate_expression_l2544_254426

theorem evaluate_expression (a x : ℝ) (h : x = a + 9) : x - a + 5 = 14 := by
  sorry

end evaluate_expression_l2544_254426


namespace min_value_abc_l2544_254428

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 27) :
  a + 3 * b + 9 * c ≥ 27 := by
  sorry

end min_value_abc_l2544_254428


namespace acme_cheaper_at_min_shirts_l2544_254481

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℝ := 60 + 11 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_price (x : ℕ) : ℝ := 20 + 15 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < gamma_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ gamma_price n :=
by sorry

end acme_cheaper_at_min_shirts_l2544_254481


namespace remainder_theorem_l2544_254402

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v := by
sorry

end remainder_theorem_l2544_254402


namespace x_y_negative_l2544_254401

theorem x_y_negative (x y : ℝ) (h1 : x - y > x) (h2 : x + y < y) : x < 0 ∧ y < 0 := by
  sorry

end x_y_negative_l2544_254401


namespace reflection_about_x_axis_l2544_254407

/-- Represents a parabola in the Cartesian coordinate system -/
structure Parabola where
  f : ℝ → ℝ

/-- Reflects a parabola about the x-axis -/
def reflect_x (p : Parabola) : Parabola :=
  { f := λ x => -(p.f x) }

/-- The original parabola y = x^2 + x - 2 -/
def original_parabola : Parabola :=
  { f := λ x => x^2 + x - 2 }

/-- The expected reflected parabola y = -x^2 - x + 2 -/
def expected_reflected_parabola : Parabola :=
  { f := λ x => -x^2 - x + 2 }

theorem reflection_about_x_axis :
  reflect_x original_parabola = expected_reflected_parabola :=
by sorry

end reflection_about_x_axis_l2544_254407


namespace area_under_arcsin_cos_l2544_254475

noncomputable def f (x : ℝ) := Real.arcsin (Real.cos x)

theorem area_under_arcsin_cos : ∫ x in (0)..(3 * Real.pi), |f x| = (3 * Real.pi^2) / 4 := by
  sorry

end area_under_arcsin_cos_l2544_254475


namespace exists_parallel_line_l2544_254460

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (intersects : Plane → Plane → Prop)
variable (not_perpendicular : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem exists_parallel_line
  (α β γ : Plane)
  (h1 : perpendicular β γ)
  (h2 : intersects α γ)
  (h3 : not_perpendicular α γ) :
  ∃ (a : Line), in_plane a α ∧ parallel a γ :=
sorry

end exists_parallel_line_l2544_254460


namespace set_equality_proof_l2544_254404

universe u

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 3, 4}
def N : Set Nat := {1, 2}

theorem set_equality_proof :
  ({2, 3, 4} : Set Nat) = (U \ M) ∪ (U \ N) := by sorry

end set_equality_proof_l2544_254404


namespace parabola_equation_l2544_254400

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  focus_x : ℝ
  focus_x_pos : focus_x > 0

/-- The line y = x -/
def line_y_eq_x (x : ℝ) : ℝ := x

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_equation (C : Parabola) 
  (A B : Point) 
  (P : Point)
  (h1 : line_y_eq_x A.x = A.y ∧ line_y_eq_x B.x = B.y)  -- A and B lie on y = x
  (h2 : P.x = 2 ∧ P.y = 2)  -- P is (2,2)
  (h3 : P.x = (A.x + B.x) / 2 ∧ P.y = (A.y + B.y) / 2)  -- P is midpoint of AB
  : ∀ (x y : ℝ), (y^2 = 4*x) ↔ (∃ (t : ℝ), x = t^2 * C.focus_x ∧ y = 2*t * C.focus_x) :=
sorry

end parabola_equation_l2544_254400


namespace sqrt_sum_simplification_l2544_254482

theorem sqrt_sum_simplification :
  ∃ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
    (∀ (a' b' c' : ℕ), 
      (a' > 0 ∧ b' > 0 ∧ c' > 0) →
      (Real.sqrt 3 + 1 / Real.sqrt 3 + Real.sqrt 11 + 1 / Real.sqrt 11 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c') →
      c ≤ c') ∧
    a + b + c = 113 :=
by sorry

end sqrt_sum_simplification_l2544_254482


namespace range_of_x_minus_sqrt3y_l2544_254471

theorem range_of_x_minus_sqrt3y (x y : ℝ) 
  (h : x^2 + y^2 - 2*x + 2*Real.sqrt 3*y + 3 = 0) :
  ∃ (min max : ℝ), min = 2 ∧ max = 6 ∧ 
    (∀ z, z = x - Real.sqrt 3 * y → min ≤ z ∧ z ≤ max) :=
sorry

end range_of_x_minus_sqrt3y_l2544_254471


namespace angle_measure_l2544_254483

theorem angle_measure (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_measure_l2544_254483


namespace lcm_and_gcd_of_36_and_48_l2544_254415

theorem lcm_and_gcd_of_36_and_48 :
  (Nat.lcm 36 48 = 144) ∧ (Nat.gcd 36 48 = 12) := by
  sorry

end lcm_and_gcd_of_36_and_48_l2544_254415


namespace smallest_m_is_one_l2544_254478

/-- The largest prime with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : q ≥ 10^2022 ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → (p ≥ 10^2022 ∧ p < 10^2023) → p ≤ q

/-- The smallest positive integer m such that q^2 - m is divisible by 15 -/
def m : ℕ := sorry

theorem smallest_m_is_one : m = 1 := by sorry

end smallest_m_is_one_l2544_254478


namespace cylinder_height_relationship_l2544_254468

/-- Theorem: Relationship between heights of two cylinders with equal volume and different radii -/
theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 →
  h₁ > 0 →
  r₂ > 0 →
  h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by
  sorry

end cylinder_height_relationship_l2544_254468


namespace petya_win_probability_l2544_254442

/-- The game "Pile of Stones" --/
structure PileOfStones where
  initialStones : Nat
  minTake : Nat
  maxTake : Nat

/-- The optimal strategy for the game --/
def optimalStrategy (game : PileOfStones) : Nat → Nat :=
  sorry

/-- The probability of winning when playing randomly --/
def randomWinProbability (game : PileOfStones) : ℚ :=
  sorry

/-- The theorem stating the probability of Petya winning --/
theorem petya_win_probability :
  let game : PileOfStones := {
    initialStones := 16,
    minTake := 1,
    maxTake := 4
  }
  randomWinProbability game = 1 / 256 := by
  sorry

end petya_win_probability_l2544_254442


namespace camera_cost_proof_l2544_254498

/-- The cost of the old camera model --/
def old_camera_cost : ℝ := 4000

/-- The cost of the new camera model --/
def new_camera_cost : ℝ := old_camera_cost * 1.3

/-- The original price of the lens --/
def lens_original_price : ℝ := 400

/-- The discount on the lens --/
def lens_discount : ℝ := 200

/-- The discounted price of the lens --/
def lens_discounted_price : ℝ := lens_original_price - lens_discount

/-- The total amount paid for the new camera and the discounted lens --/
def total_paid : ℝ := 5400

theorem camera_cost_proof : 
  new_camera_cost + lens_discounted_price = total_paid ∧ 
  old_camera_cost = 4000 := by
  sorry

end camera_cost_proof_l2544_254498


namespace flower_expense_proof_l2544_254447

/-- Calculates the total expense for flowers given the quantities and price per flower -/
def totalExpense (tulips carnations roses : ℕ) (pricePerFlower : ℕ) : ℕ :=
  (tulips + carnations + roses) * pricePerFlower

/-- Proves that the total expense for the given flower quantities and price is 1890 -/
theorem flower_expense_proof :
  totalExpense 250 375 320 2 = 1890 := by
  sorry

end flower_expense_proof_l2544_254447


namespace second_player_can_prevent_win_l2544_254484

/-- Represents a position on the infinite grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents a move in the game -/
inductive Move
  | X (pos : Position)
  | O (pos : Position)

/-- Represents the game state -/
def GameState := List Move

/-- A strategy for the second player -/
def Strategy := GameState → Position

/-- Checks if a list of positions contains 11 consecutive X's -/
def hasElevenConsecutiveXs (positions : List Position) : Prop :=
  sorry

/-- Checks if a game state has a winning condition for the first player -/
def isWinningState (state : GameState) : Prop :=
  sorry

/-- The main theorem stating that the second player can prevent the first player from winning -/
theorem second_player_can_prevent_win :
  ∃ (strategy : Strategy),
    ∀ (game : GameState),
      ¬(isWinningState game) :=
sorry

end second_player_can_prevent_win_l2544_254484


namespace angle_inequality_l2544_254413

theorem angle_inequality (α β γ : Real) 
  (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : γ < π) :
  Real.sin (α / 2) + Real.sin (β / 2) > Real.sin (γ / 2) := by
  sorry

end angle_inequality_l2544_254413


namespace max_gcd_lcm_product_l2544_254437

theorem max_gcd_lcm_product (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) :
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a₀ b₀ c₀ : ℕ), Nat.gcd (Nat.lcm a₀ b₀) c₀ = 10 ∧
    Nat.gcd (Nat.lcm a₀ b₀) c₀ * Nat.lcm (Nat.gcd a₀ b₀) c₀ = 200 :=
by sorry

end max_gcd_lcm_product_l2544_254437


namespace circle_in_second_quadrant_implies_a_range_l2544_254439

/-- Definition of the circle equation -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0

/-- Definition of a point being in the second quadrant -/
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem stating that if all points on the circle are in the second quadrant,
    then a is between 0 and 3 -/
theorem circle_in_second_quadrant_implies_a_range :
  (∀ x y : ℝ, circle_equation x y a → in_second_quadrant x y) →
  0 < a ∧ a < 3 :=
sorry

end circle_in_second_quadrant_implies_a_range_l2544_254439


namespace library_budget_is_3000_l2544_254496

-- Define the total budget
def total_budget : ℝ := 20000

-- Define the library budget percentage
def library_percentage : ℝ := 0.15

-- Define the parks budget percentage
def parks_percentage : ℝ := 0.24

-- Define the remaining budget
def remaining_budget : ℝ := 12200

-- Theorem to prove
theorem library_budget_is_3000 :
  library_percentage * total_budget = 3000 :=
by
  sorry


end library_budget_is_3000_l2544_254496


namespace toms_hourly_wage_l2544_254457

/-- Tom's hourly wage calculation --/
theorem toms_hourly_wage :
  let item_cost : ℝ := 25.35 + 70.69 + 85.96
  let hours_worked : ℕ := 31
  let savings_rate : ℝ := 0.1
  let hourly_wage : ℝ := item_cost / ((1 - savings_rate) * hours_worked)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |hourly_wage - 6.52| < ε :=
by
  sorry

end toms_hourly_wage_l2544_254457


namespace constant_segments_am_plus_bn_equals_11_am_equals_bn_l2544_254469

-- Define the points on the number line
def A (t : ℝ) : ℝ := -1 + 2*t
def M (t : ℝ) : ℝ := t
def N (t : ℝ) : ℝ := t + 2
def B (t : ℝ) : ℝ := 11 - t

-- Theorem for part 1
theorem constant_segments :
  ∀ x t : ℝ, abs (B t - A t) = 12 ∧ abs (N t - M t) = 2 :=
sorry

-- Theorem for part 2, question 1
theorem am_plus_bn_equals_11 :
  ∃ t : ℝ, abs (M t - A t) + abs (B t - N t) = 11 ∧ t = 9.5 :=
sorry

-- Theorem for part 2, question 2
theorem am_equals_bn :
  ∃ t₁ t₂ : ℝ, 
    abs (M t₁ - A t₁) = abs (B t₁ - N t₁) ∧
    abs (M t₂ - A t₂) = abs (B t₂ - N t₂) ∧
    t₁ = 10/3 ∧ t₂ = 8 :=
sorry

end constant_segments_am_plus_bn_equals_11_am_equals_bn_l2544_254469


namespace continued_fraction_sum_l2544_254485

theorem continued_fraction_sum (v w x y z : ℕ+) : 
  (v : ℚ) + 1 / ((w : ℚ) + 1 / ((x : ℚ) + 1 / ((y : ℚ) + 1 / (z : ℚ)))) = 222 / 155 →
  10^4 * v.val + 10^3 * w.val + 10^2 * x.val + 10 * y.val + z.val = 12354 := by
sorry

end continued_fraction_sum_l2544_254485


namespace census_set_is_population_l2544_254409

/-- The term for the entire set of objects to be investigated in a census -/
def census_set : String := "Population"

/-- Theorem stating that the entire set of objects to be investigated in a census is called "Population" -/
theorem census_set_is_population : census_set = "Population" := by
  sorry

end census_set_is_population_l2544_254409


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l2544_254462

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s, s = -(b / a) ∧ ∀ x y, f x = 0 → f y = 0 → x + y = s) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2023 * x - 2024
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y) →
  (∃ s, s = -2023 ∧ ∀ x y, f x = 0 → f y = 0 → x + y = s) :=
sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l2544_254462


namespace line_intersect_xz_plane_l2544_254456

/-- The line passing through two points intersects the xz-plane at a specific point -/
theorem line_intersect_xz_plane (p₁ p₂ intersection : ℝ × ℝ × ℝ) :
  p₁ = (1, 2, 3) →
  p₂ = (4, 0, -1) →
  intersection = (4, 0, -1) →
  (∃ t : ℝ, intersection = p₁ + t • (p₂ - p₁)) ∧
  (intersection.2 = 0) := by
  sorry

#check line_intersect_xz_plane

end line_intersect_xz_plane_l2544_254456


namespace num_2d_faces_6cube_l2544_254463

/-- The number of 2-D square faces in a 6-dimensional cube of side length 6 -/
def num_2d_faces (n : ℕ) (side_length : ℕ) : ℕ :=
  (Nat.choose n 4) * (side_length + 1)^4 * side_length^2

/-- Theorem stating the number of 2-D square faces in a 6-cube of side length 6 -/
theorem num_2d_faces_6cube :
  num_2d_faces 6 6 = 1296150 := by
  sorry

end num_2d_faces_6cube_l2544_254463


namespace min_value_on_circle_l2544_254451

theorem min_value_on_circle (x y : ℝ) (h : (x - 2)^2 + (y - 1)^2 = 1) :
  ∃ (min : ℝ), (∀ (a b : ℝ), (a - 2)^2 + (b - 1)^2 = 1 → a^2 + b^2 ≥ min) ∧ min = 6 - 2 * Real.sqrt 5 := by
  sorry

end min_value_on_circle_l2544_254451


namespace orange_apple_weight_equivalence_l2544_254470

/-- Given that 9 oranges weigh the same as 6 apples, prove that 36 oranges weigh the same as 24 apples -/
theorem orange_apple_weight_equivalence 
  (orange_weight apple_weight : ℚ) 
  (h : 9 * orange_weight = 6 * apple_weight) : 
  36 * orange_weight = 24 * apple_weight := by
sorry

end orange_apple_weight_equivalence_l2544_254470


namespace least_divisible_by_960_sixty_divisible_by_960_least_value_is_60_l2544_254412

theorem least_divisible_by_960 (a : ℕ) : a^5 % 960 = 0 → a ≥ 60 := by
  sorry

theorem sixty_divisible_by_960 : (60 : ℕ)^5 % 960 = 0 := by
  sorry

theorem least_value_is_60 : ∃ a : ℕ, a^5 % 960 = 0 ∧ ∀ b : ℕ, b^5 % 960 = 0 → b ≥ a := by
  sorry

end least_divisible_by_960_sixty_divisible_by_960_least_value_is_60_l2544_254412
