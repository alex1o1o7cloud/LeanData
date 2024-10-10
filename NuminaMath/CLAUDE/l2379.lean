import Mathlib

namespace sum_of_coefficients_l2379_237902

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, 1 + x^5 = a₀ + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end sum_of_coefficients_l2379_237902


namespace num_factors_of_given_number_l2379_237953

/-- The number of distinct natural-number factors of 8^2 * 9^3 * 7^5 -/
def num_factors : ℕ :=
  (7 : ℕ) * (7 : ℕ) * (6 : ℕ)

/-- The given number 8^2 * 9^3 * 7^5 -/
def given_number : ℕ :=
  (8^2 : ℕ) * (9^3 : ℕ) * (7^5 : ℕ)

theorem num_factors_of_given_number :
  (Finset.filter (fun d => given_number % d = 0) (Finset.range (given_number + 1))).card = num_factors :=
by sorry

end num_factors_of_given_number_l2379_237953


namespace initial_value_problem_l2379_237985

theorem initial_value_problem : ∃! x : ℤ, (x + 82) % 456 = 0 ∧ x = 374 := by sorry

end initial_value_problem_l2379_237985


namespace chocolate_cookie_price_l2379_237972

/-- Given the sale of cookies and total revenue, prove the price of chocolate cookies -/
theorem chocolate_cookie_price
  (chocolate_count : ℕ)
  (vanilla_count : ℕ)
  (vanilla_price : ℚ)
  (total_revenue : ℚ)
  (h1 : chocolate_count = 220)
  (h2 : vanilla_count = 70)
  (h3 : vanilla_price = 2)
  (h4 : total_revenue = 360)
  : ∃ (chocolate_price : ℚ),
    chocolate_price * chocolate_count + vanilla_price * vanilla_count = total_revenue ∧
    chocolate_price = 1 := by
  sorry

end chocolate_cookie_price_l2379_237972


namespace davids_physics_marks_l2379_237951

def marks_english : ℕ := 76
def marks_mathematics : ℕ := 65
def marks_chemistry : ℕ := 67
def marks_biology : ℕ := 85
def average_marks : ℕ := 75
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  ∃ (marks_physics : ℕ),
    marks_physics = average_marks * num_subjects - (marks_english + marks_mathematics + marks_chemistry + marks_biology) ∧
    marks_physics = 82 := by
  sorry

end davids_physics_marks_l2379_237951


namespace triangle_sum_special_case_l2379_237990

def triangle_sum (a b : ℕ) : ℕ :=
  let n_min := a.max b - (a.min b - 1)
  let n_max := a + b - 1
  (n_max - n_min + 1) * (n_max + n_min) / 2

theorem triangle_sum_special_case : triangle_sum 7 10 = 260 := by
  sorry

end triangle_sum_special_case_l2379_237990


namespace circle_symmetry_l2379_237906

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end circle_symmetry_l2379_237906


namespace triangle_side_length_triangle_angle_measure_l2379_237934

-- Part 1
theorem triangle_side_length (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = π / 6 →
  C = 3 * π / 4 →
  a = Real.sqrt 6 - Real.sqrt 2 :=
sorry

-- Part 2
theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  S = (1 / 4) * (a^2 + b^2 - c^2) →
  C = π / 4 :=
sorry

end triangle_side_length_triangle_angle_measure_l2379_237934


namespace system_solution_l2379_237912

-- Define the system of equations
def system (x : Fin 6 → ℚ) : Prop :=
  2 * x 0 + 2 * x 1 - x 2 + x 3 + 4 * x 5 = 0 ∧
  x 0 + 2 * x 1 + 2 * x 2 + 3 * x 4 + x 5 = -2 ∧
  x 0 - 2 * x 1 + x 3 + 2 * x 4 = 0

-- Define the solution
def solution (x : Fin 6 → ℚ) : Prop :=
  x 0 = -1/4 - 5/8 * x 3 - 9/8 * x 4 - 9/8 * x 5 ∧
  x 1 = -1/8 + 3/16 * x 3 - 7/16 * x 4 + 9/16 * x 5 ∧
  x 2 = -3/4 + 1/8 * x 3 - 11/8 * x 4 + 5/8 * x 5

-- Theorem statement
theorem system_solution :
  ∀ x : Fin 6 → ℚ, system x ↔ solution x :=
sorry

end system_solution_l2379_237912


namespace fourth_root_equation_solutions_l2379_237987

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (((64 - 2*x) ^ (1/4) : ℝ) + ((48 + 2*x) ^ (1/4) : ℝ) = 6) ↔ (x = 32 ∨ x = -8) := by
  sorry

end fourth_root_equation_solutions_l2379_237987


namespace octagon_intersection_only_hexagonal_prism_l2379_237996

/-- Represents the possible geometric solids --/
inductive GeometricSolid
  | TriangularPrism
  | RectangularPrism
  | PentagonalPrism
  | HexagonalPrism

/-- Represents the possible shapes resulting from a plane intersection --/
inductive IntersectionShape
  | Triangle
  | Quadrilateral
  | Pentagon
  | Hexagon
  | Heptagon
  | Octagon
  | Rectangle

/-- Returns the possible intersection shapes for a given geometric solid --/
def possibleIntersections (solid : GeometricSolid) : List IntersectionShape :=
  match solid with
  | GeometricSolid.TriangularPrism => [IntersectionShape.Quadrilateral, IntersectionShape.Triangle]
  | GeometricSolid.RectangularPrism => [IntersectionShape.Pentagon, IntersectionShape.Quadrilateral, IntersectionShape.Triangle, IntersectionShape.Rectangle]
  | GeometricSolid.PentagonalPrism => [IntersectionShape.Hexagon, IntersectionShape.Pentagon, IntersectionShape.Rectangle, IntersectionShape.Triangle]
  | GeometricSolid.HexagonalPrism => [IntersectionShape.Octagon, IntersectionShape.Heptagon, IntersectionShape.Rectangle]

/-- Theorem: Only the hexagonal prism can produce an octagonal intersection --/
theorem octagon_intersection_only_hexagonal_prism :
  ∀ (solid : GeometricSolid),
    (IntersectionShape.Octagon ∈ possibleIntersections solid) ↔ (solid = GeometricSolid.HexagonalPrism) :=
by sorry


end octagon_intersection_only_hexagonal_prism_l2379_237996


namespace factorization_equality_l2379_237952

theorem factorization_equality (a b : ℝ) : a^2 * b + 2 * a * b^2 + b^3 = b * (a + b)^2 := by
  sorry

end factorization_equality_l2379_237952


namespace chimney_bricks_proof_l2379_237904

/-- The time it takes Brenda to build the chimney alone (in hours) -/
def brenda_time : ℝ := 9

/-- The time it takes Brandon to build the chimney alone (in hours) -/
def brandon_time : ℝ := 10

/-- The decrease in combined output when working together (in bricks per hour) -/
def output_decrease : ℝ := 10

/-- The time it takes Brenda and Brandon to build the chimney together (in hours) -/
def combined_time : ℝ := 5

/-- The number of bricks in the chimney -/
def chimney_bricks : ℝ := 900

theorem chimney_bricks_proof :
  let brenda_rate := chimney_bricks / brenda_time
  let brandon_rate := chimney_bricks / brandon_time
  let combined_rate := brenda_rate + brandon_rate - output_decrease
  chimney_bricks = combined_rate * combined_time := by
  sorry

end chimney_bricks_proof_l2379_237904


namespace binomial_identity_l2379_237920

theorem binomial_identity (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end binomial_identity_l2379_237920


namespace radiator_problem_l2379_237922

/-- Represents the fraction of water remaining after a number of replacements -/
def water_fraction (initial_volume : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ) : ℚ :=
  (1 - replacement_volume / initial_volume) ^ num_replacements

/-- The problem statement -/
theorem radiator_problem (initial_volume : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ)
    (h1 : initial_volume = 20)
    (h2 : replacement_volume = 5)
    (h3 : num_replacements = 4) :
  water_fraction initial_volume replacement_volume num_replacements = 81 / 256 := by
  sorry

#eval water_fraction 20 5 4

end radiator_problem_l2379_237922


namespace bank_balance_after_two_years_l2379_237983

/-- The amount of money in a bank account after a given number of years,
    given an initial amount and an annual interest rate. -/
def bank_balance (initial_amount : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  initial_amount * (1 + interest_rate) ^ years

/-- Theorem stating that $100 invested for 2 years at 10% annual interest results in $121 -/
theorem bank_balance_after_two_years :
  bank_balance 100 0.1 2 = 121 := by
  sorry

end bank_balance_after_two_years_l2379_237983


namespace seans_fraction_of_fritz_money_l2379_237927

theorem seans_fraction_of_fritz_money (fritz_money sean_money rick_money : ℚ) 
  (x : ℚ) : 
  fritz_money = 40 →
  sean_money = x * fritz_money + 4 →
  rick_money = 3 * sean_money →
  rick_money + sean_money = 96 →
  x = 1/2 := by
sorry

end seans_fraction_of_fritz_money_l2379_237927


namespace voting_ratio_l2379_237998

/-- Given a voting scenario where:
    - 2/9 of the votes have been counted
    - 3/4 of the counted votes are in favor
    - 0.7857142857142856 of the remaining votes are against
    Prove that the ratio of total votes against to total votes in favor is 4:1 -/
theorem voting_ratio (V : ℝ) (hV : V > 0) : 
  let counted := (2/9) * V
  let in_favor := (3/4) * counted
  let remaining := V - counted
  let against_remaining := 0.7857142857142856 * remaining
  let total_against := ((1/4) * counted) + against_remaining
  let total_in_favor := in_favor
  (total_against / total_in_favor) = 4 := by
  sorry


end voting_ratio_l2379_237998


namespace monotonically_decreasing_implies_a_leq_neg_three_l2379_237944

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1

-- State the theorem
theorem monotonically_decreasing_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) → a ≤ -3 := by
  sorry

end monotonically_decreasing_implies_a_leq_neg_three_l2379_237944


namespace normal_transform_theorem_l2379_237938

/-- Transforms a standard normal random variable to a normal distribution with given mean and standard deviation -/
def transform (x : ℝ) (μ σ : ℝ) : ℝ := σ * x + μ

/-- The four standard normal random variables -/
def X₁ : ℝ := 0.06
def X₂ : ℝ := -1.10
def X₃ : ℝ := -1.52
def X₄ : ℝ := 0.83

/-- The mean of the target normal distribution -/
def μ : ℝ := 2

/-- The standard deviation of the target normal distribution -/
def σ : ℝ := 3

/-- Theorem stating that the transformation of the given standard normal random variables
    results in the specified values for the target normal distribution -/
theorem normal_transform_theorem :
  (transform X₁ μ σ, transform X₂ μ σ, transform X₃ μ σ, transform X₄ μ σ) =
  (2.18, -1.3, -2.56, 4.49) := by
  sorry

end normal_transform_theorem_l2379_237938


namespace problem_solution_l2379_237961

theorem problem_solution (x y : ℚ) : 
  x = 103 → x^3 * y - 2 * x^2 * y + x * y = 1060900 → y = 100 / 101 := by
  sorry

end problem_solution_l2379_237961


namespace platform_length_specific_platform_length_l2379_237930

/-- The length of a platform given train parameters -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Proof of the specific platform length problem -/
theorem specific_platform_length : 
  platform_length 360 45 48 = 840 := by
  sorry

end platform_length_specific_platform_length_l2379_237930


namespace cricket_team_average_age_l2379_237981

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age_diff : ℕ) 
  (h1 : team_size = 11)
  (h2 : captain_age = 24)
  (h3 : wicket_keeper_age_diff = 7)
  : ∃ (team_avg_age : ℚ),
    team_avg_age = 23 ∧
    (team_size : ℚ) * team_avg_age = 
      captain_age + (captain_age + wicket_keeper_age_diff) + 
      ((team_size - 2) : ℚ) * (team_avg_age - 1) :=
by sorry

end cricket_team_average_age_l2379_237981


namespace jacket_sale_profit_l2379_237946

/-- Calculates the merchant's gross profit for a jacket sale -/
theorem jacket_sale_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  purchase_price = 60 ∧ 
  markup_percent = 0.25 ∧ 
  discount_percent = 0.20 → 
  let selling_price := purchase_price / (1 - markup_percent)
  let discounted_price := selling_price * (1 - discount_percent)
  discounted_price - purchase_price = 4 :=
by sorry

end jacket_sale_profit_l2379_237946


namespace age_problem_l2379_237929

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 72) : 
  b = 28 := by
  sorry

end age_problem_l2379_237929


namespace discounted_price_calculation_l2379_237960

theorem discounted_price_calculation (original_price discount_percentage : ℝ) :
  original_price = 600 ∧ discount_percentage = 20 →
  original_price * (1 - discount_percentage / 100) = 480 := by
sorry

end discounted_price_calculation_l2379_237960


namespace book_pages_ratio_l2379_237999

theorem book_pages_ratio : 
  let selena_pages : ℕ := 400
  let harry_pages : ℕ := 180
  ∃ (a b : ℕ), (a = 9 ∧ b = 20) ∧ 
    (harry_pages : ℚ) / selena_pages = a / b :=
by sorry

end book_pages_ratio_l2379_237999


namespace project_work_time_l2379_237933

/-- Calculates the time spent working on a project given the number of days,
    number of naps, hours per nap, and hours per day. -/
def time_spent_working (days : ℕ) (num_naps : ℕ) (hours_per_nap : ℕ) (hours_per_day : ℕ) : ℕ :=
  days * hours_per_day - num_naps * hours_per_nap

/-- Proves that given a 4-day project where 6 seven-hour naps are taken,
    and each day has 24 hours, the time spent working on the project is 54 hours. -/
theorem project_work_time :
  time_spent_working 4 6 7 24 = 54 := by
  sorry

end project_work_time_l2379_237933


namespace vector_c_value_l2379_237997

def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (-2, 4)

theorem vector_c_value :
  ∀ c : ℝ × ℝ, (4 • a) + (3 • b - 2 • a) + c = (0, 0) → c = (4, -6) := by
  sorry

end vector_c_value_l2379_237997


namespace min_distance_to_line_l2379_237975

theorem min_distance_to_line : 
  ∀ m n : ℝ, 
  (4 * m - 3 * n - 5 * Real.sqrt 2 = 0) → 
  (∀ x y : ℝ, 4 * x - 3 * y - 5 * Real.sqrt 2 = 0 → m^2 + n^2 ≤ x^2 + y^2) → 
  m^2 + n^2 = 2 := by
sorry

end min_distance_to_line_l2379_237975


namespace positive_expression_l2379_237959

theorem positive_expression (x : ℝ) : x^2 * Real.sin x + x * Real.cos x + x^2 + (1/2 : ℝ) > 0 := by
  sorry

end positive_expression_l2379_237959


namespace tangent_to_both_circumcircles_l2379_237928

-- Define the basic structures
structure Point := (x y : ℝ)

structure Line := (a b : Point)

structure Circle := (center : Point) (radius : ℝ)

-- Define the parallelogram
def Parallelogram (A B C D : Point) : Prop := sorry

-- Define a point between two other points
def PointBetween (E B F : Point) : Prop := sorry

-- Define the intersection of two lines
def Intersect (l₁ l₂ : Line) (O : Point) : Prop := sorry

-- Define a line tangent to a circle
def Tangent (l : Line) (c : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def Circumcircle (A B C : Point) : Circle := sorry

-- Main theorem
theorem tangent_to_both_circumcircles 
  (A B C D E F O : Point) 
  (h1 : Parallelogram A B C D)
  (h2 : PointBetween E B F)
  (h3 : Intersect (Line.mk A C) (Line.mk B D) O)
  (h4 : Tangent (Line.mk A E) (Circumcircle A O D))
  (h5 : Tangent (Line.mk D F) (Circumcircle A O D)) :
  Tangent (Line.mk A E) (Circumcircle E O F) ∧ 
  Tangent (Line.mk D F) (Circumcircle E O F) := by sorry

end tangent_to_both_circumcircles_l2379_237928


namespace grape_bowls_problem_l2379_237923

theorem grape_bowls_problem (n : ℕ) : 
  (8 * 12 = 6 * n) → n = 16 := by
  sorry

end grape_bowls_problem_l2379_237923


namespace smallest_term_of_sequence_l2379_237973

def a (n : ℕ+) : ℤ := n^2 - 9*n - 100

theorem smallest_term_of_sequence (n : ℕ+) :
  ∃ m : ℕ+, (m = 4 ∨ m = 5) ∧ ∀ k : ℕ+, a m ≤ a k :=
sorry

end smallest_term_of_sequence_l2379_237973


namespace triangle_special_condition_right_angle_l2379_237914

/-- Given a triangle ABC, if b cos C + c cos B = a sin A, then angle A is 90° -/
theorem triangle_special_condition_right_angle 
  (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧ 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  b * Real.cos C + c * Real.cos B = a * Real.sin A → 
  A = π / 2 := by
sorry

end triangle_special_condition_right_angle_l2379_237914


namespace proposition_and_negation_l2379_237901

theorem proposition_and_negation :
  (∃ x : ℝ, x^2 - x = 0) ∧
  (¬(∃ x : ℝ, x^2 - x = 0) ↔ (∀ x : ℝ, x^2 - x ≠ 0)) :=
by sorry

end proposition_and_negation_l2379_237901


namespace particle_movement_probability_l2379_237907

/-- The probability of a particle moving from (0, 0) to (2, 3) in 5 steps,
    where each step has an equal probability of 1/2 of moving right or up. -/
theorem particle_movement_probability :
  let n : ℕ := 5  -- Total number of steps
  let k : ℕ := 2  -- Number of steps to the right
  let p : ℚ := 1/2  -- Probability of moving right (or up)
  Nat.choose n k * p^n = (1/2)^5 := by
  sorry

end particle_movement_probability_l2379_237907


namespace tan_one_condition_l2379_237991

theorem tan_one_condition (x : Real) : 
  (∃ k : Int, x = (k * Real.pi) / 4) ∧ 
  (∃ y : Real, (∃ m : Int, y = (m * Real.pi) / 4) ∧ Real.tan y = 1) ∧ 
  (∃ z : Real, (∃ n : Int, z = (n * Real.pi) / 4) ∧ Real.tan z ≠ 1) := by
  sorry

end tan_one_condition_l2379_237991


namespace two_std_dev_below_mean_l2379_237978

def normal_distribution (mean : ℝ) (std_dev : ℝ) := { μ : ℝ // μ = mean }

theorem two_std_dev_below_mean 
  (μ : ℝ) (σ : ℝ) (h_μ : μ = 14.5) (h_σ : σ = 1.7) :
  ∃ (x : ℝ), x = μ - 2 * σ ∧ x = 11.1 :=
sorry

end two_std_dev_below_mean_l2379_237978


namespace inequality_solution_l2379_237957

theorem inequality_solution : 
  {x : ℕ | 3 * x - 2 < 7} = {0, 1, 2} := by sorry

end inequality_solution_l2379_237957


namespace bisection_next_step_l2379_237936

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_continuous : ContinuousOn f (Set.Icc 0 1)
axiom f_neg_zero : f 0 < 0
axiom f_neg_half : f 0.5 < 0
axiom f_pos_one : f 1 > 0

-- Define the theorem
theorem bisection_next_step :
  ∃ x ∈ Set.Ioo 0.5 1, f x = 0 ∧ 
  (∀ y, y ∈ Set.Icc 0 1 → f y = 0 → y ∈ Set.Icc 0.5 1) ∧
  (0.75 = (0.5 + 1) / 2) := by
  sorry

end bisection_next_step_l2379_237936


namespace math_test_results_l2379_237948

/-- Represents the score distribution for a math test -/
structure ScoreDistribution where
  prob_45 : ℚ
  prob_50 : ℚ
  prob_55 : ℚ
  prob_60 : ℚ

/-- Represents the conditions of the math test -/
structure MathTest where
  total_questions : ℕ
  options_per_question : ℕ
  points_per_correct : ℕ
  certain_correct : ℕ
  uncertain_two_eliminated : ℕ
  uncertain_one_eliminated : ℕ

/-- Calculates the probability of scoring 55 points given the test conditions -/
def prob_55 (test : MathTest) : ℚ :=
  sorry

/-- Calculates the score distribution given the test conditions -/
def score_distribution (test : MathTest) : ScoreDistribution :=
  sorry

/-- Calculates the expected value of the score given the score distribution -/
def expected_value (dist : ScoreDistribution) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem math_test_results (test : MathTest) 
  (h1 : test.total_questions = 12)
  (h2 : test.options_per_question = 4)
  (h3 : test.points_per_correct = 5)
  (h4 : test.certain_correct = 9)
  (h5 : test.uncertain_two_eliminated = 2)
  (h6 : test.uncertain_one_eliminated = 1) :
  prob_55 test = 1/3 ∧ 
  expected_value (score_distribution test) = 165/3 :=
sorry

end math_test_results_l2379_237948


namespace cart_distance_proof_l2379_237976

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ → ℕ :=
  fun i => a₁ + (i - 1) * d

def sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem cart_distance_proof (a₁ d n : ℕ) 
  (h₁ : a₁ = 5) 
  (h₂ : d = 7) 
  (h₃ : n = 30) : 
  sequence_sum a₁ d n = 3195 :=
sorry

end cart_distance_proof_l2379_237976


namespace hyperbola_y_axis_condition_l2379_237909

/-- Represents a conic section of the form mx^2 + ny^2 = 1 -/
structure ConicSection (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate for a hyperbola with foci on the y-axis -/
def IsHyperbolaOnYAxis (m n : ℝ) : Prop :=
  m < 0 ∧ n > 0

theorem hyperbola_y_axis_condition (m n : ℝ) :
  (IsHyperbolaOnYAxis m n → m * n < 0) ∧
  ¬(m * n < 0 → IsHyperbolaOnYAxis m n) := by
  sorry

end hyperbola_y_axis_condition_l2379_237909


namespace remainder_proof_l2379_237977

def smallest_prime_greater_than_1000 : ℕ → Prop :=
  λ x => Prime x ∧ x > 1000 ∧ ∀ y, Prime y ∧ y > 1000 → x ≤ y

theorem remainder_proof (x : ℕ) (h : smallest_prime_greater_than_1000 x) :
  (10000 - 999) % x = 945 := by
  sorry

end remainder_proof_l2379_237977


namespace mabel_transactions_l2379_237967

/-- Represents the number of transactions handled by each person -/
structure Transactions where
  mabel : ℕ
  anthony : ℕ
  cal : ℕ
  jade : ℕ

/-- The conditions of the problem -/
def problem_conditions (t : Transactions) : Prop :=
  t.anthony = t.mabel + t.mabel / 10 ∧
  t.cal = (2 * t.anthony) / 3 ∧
  t.jade = t.cal + 16 ∧
  t.jade = 82

/-- The theorem to prove -/
theorem mabel_transactions :
  ∀ t : Transactions, problem_conditions t → t.mabel = 90 := by
  sorry

end mabel_transactions_l2379_237967


namespace inequality_holds_iff_p_in_interval_l2379_237916

theorem inequality_holds_iff_p_in_interval (p q : ℝ) :
  q > 0 →
  2*p + q ≠ 0 →
  (4*(2*p*q^2 + p^2*q + 4*q^2 + 4*p*q) / (2*p + q) > 3*p^2*q) ↔
  (0 ≤ p ∧ p < 4) :=
by sorry

end inequality_holds_iff_p_in_interval_l2379_237916


namespace triangle_third_vertex_l2379_237939

/-- Given a triangle with vertices at (0, 0), (10, 5), and (x, 0) where x < 0,
    if the area of the triangle is 50 square units, then x = -20. -/
theorem triangle_third_vertex (x : ℝ) (h1 : x < 0) :
  (1/2 : ℝ) * |x * 5| = 50 → x = -20 := by sorry

end triangle_third_vertex_l2379_237939


namespace triangle_area_l2379_237940

theorem triangle_area (a b c A B C S : ℝ) : 
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a > 0 →
  a = 4 →
  A = π / 4 →
  B = π / 3 →
  S = (1 / 2) * a * b * Real.sin C →
  S = 6 + 2 * Real.sqrt 3 :=
by sorry

end triangle_area_l2379_237940


namespace average_female_height_l2379_237995

/-- Given the overall average height of students is 180 cm, the average height of male students
    is 182 cm, and the ratio of men to women is 5:1, prove that the average female height is 170 cm. -/
theorem average_female_height
  (overall_avg : ℝ)
  (male_avg : ℝ)
  (ratio : ℕ)
  (h1 : overall_avg = 180)
  (h2 : male_avg = 182)
  (h3 : ratio = 5)
  : ∃ (female_avg : ℝ), female_avg = 170 ∧
    (ratio * male_avg + female_avg) / (ratio + 1) = overall_avg :=
by
  sorry

end average_female_height_l2379_237995


namespace complex_modulus_problem_l2379_237962

theorem complex_modulus_problem (w z : ℂ) : 
  w * z = 18 - 24 * I ∧ Complex.abs w = 3 * Real.sqrt 13 → 
  Complex.abs z = 10 / Real.sqrt 13 := by
sorry

end complex_modulus_problem_l2379_237962


namespace first_square_with_two_twos_l2379_237945

def starts_with_two_twos (n : ℕ) : Prop :=
  (n / 1000 = 2) ∧ ((n / 100) % 10 = 2)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem first_square_with_two_twos : 
  ∃! n : ℕ, 
    (∀ m : ℕ, m < n → ¬(starts_with_two_twos (m^2))) ∧ 
    (starts_with_two_twos (n^2)) ∧
    (∃ k : ℕ, k > n ∧ starts_with_two_twos (k^2) ∧ sum_of_digits (k^2) = 13) ∧
    n = 47 := by sorry

end first_square_with_two_twos_l2379_237945


namespace repeating_decimal_fraction_l2379_237980

-- Define the repeating decimals
def x : ℚ := 0.142857142857142857
def y : ℚ := 2.857142857142857142

-- State the theorem
theorem repeating_decimal_fraction : x / y = 1 / 20 := by
  sorry

end repeating_decimal_fraction_l2379_237980


namespace permutation_inequality_l2379_237986

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (({x, y, z, w} : Finset ℝ) = {a, b, c, d}) ∧
    (2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2)) := by
  sorry

end permutation_inequality_l2379_237986


namespace car_start_time_difference_l2379_237947

/-- Two cars traveling at the same speed with specific distance ratios at different times --/
theorem car_start_time_difference
  (speed : ℝ)
  (distance_ratio_10am : ℝ)
  (distance_ratio_12pm : ℝ)
  (h1 : speed > 0)
  (h2 : distance_ratio_10am = 5)
  (h3 : distance_ratio_12pm = 3)
  : ∃ (start_time_diff : ℝ),
    start_time_diff = 8 ∧
    distance_ratio_10am * (10 - start_time_diff) = 10 ∧
    distance_ratio_12pm * (12 - start_time_diff) = 12 :=
by sorry

end car_start_time_difference_l2379_237947


namespace cone_from_sector_cone_sector_proof_l2379_237949

theorem cone_from_sector (sector_angle : Real) (circle_radius : Real) 
  (base_radius : Real) (slant_height : Real) : Prop :=
  sector_angle = 252 ∧
  circle_radius = 10 ∧
  base_radius = 7 ∧
  slant_height = 10 ∧
  2 * Real.pi * base_radius = (sector_angle / 360) * 2 * Real.pi * circle_radius ∧
  base_radius ^ 2 + (circle_radius ^ 2 - base_radius ^ 2) = slant_height ^ 2

theorem cone_sector_proof : 
  ∃ (sector_angle circle_radius base_radius slant_height : Real),
    cone_from_sector sector_angle circle_radius base_radius slant_height := by
  sorry

end cone_from_sector_cone_sector_proof_l2379_237949


namespace partial_fraction_decomposition_constant_l2379_237941

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 1 → 
    1 / (x^3 - 3*x^2 - 13*x + 15) = A / (x + 3) + B / (x - 1) + C / (x - 1)^2) →
  A = 1/16 := by
sorry

end partial_fraction_decomposition_constant_l2379_237941


namespace adoption_cost_theorem_l2379_237979

def cat_cost : ℕ := 50
def adult_dog_cost : ℕ := 100
def puppy_cost : ℕ := 150

def cats_adopted : ℕ := 2
def adult_dogs_adopted : ℕ := 3
def puppies_adopted : ℕ := 2

def total_cost : ℕ := 
  cat_cost * cats_adopted + 
  adult_dog_cost * adult_dogs_adopted + 
  puppy_cost * puppies_adopted

theorem adoption_cost_theorem : total_cost = 700 := by
  sorry

end adoption_cost_theorem_l2379_237979


namespace equation_solution_l2379_237956

theorem equation_solution :
  let f (y : ℝ) := (8 * y^2 + 40 * y - 48) / (3 * y + 9) - (4 * y - 8)
  ∀ y : ℝ, f y = 0 ↔ y = (7 + Real.sqrt 73) / 2 ∨ y = (7 - Real.sqrt 73) / 2 := by
sorry

end equation_solution_l2379_237956


namespace arithmetic_progression_implies_equal_l2379_237963

theorem arithmetic_progression_implies_equal (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let g := Real.sqrt (a * b)
  let p := (a + b) / 2
  let q := Real.sqrt ((a^2 + b^2) / 2)
  (g + q = 2 * p) → a = b :=
by sorry

end arithmetic_progression_implies_equal_l2379_237963


namespace fraction_sum_equality_l2379_237908

theorem fraction_sum_equality : (3 / 10 : ℚ) + (5 / 100 : ℚ) - (2 / 1000 : ℚ) = (348 / 1000 : ℚ) := by
  sorry

end fraction_sum_equality_l2379_237908


namespace rectangular_solid_surface_area_l2379_237926

/-- The total surface area of a rectangular solid -/
def totalSurfaceArea (length width depth : ℝ) : ℝ :=
  2 * (length * width + width * depth + length * depth)

/-- Theorem: The total surface area of a rectangular solid with length 9 meters, width 8 meters, and depth 5 meters is 314 square meters -/
theorem rectangular_solid_surface_area :
  totalSurfaceArea 9 8 5 = 314 := by
  sorry

end rectangular_solid_surface_area_l2379_237926


namespace complex_z_value_l2379_237958

theorem complex_z_value : ∃ z : ℂ, z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) ∧ 
  z = Complex.mk (1/2) (-Real.sqrt 3 / 2) := by
  sorry

end complex_z_value_l2379_237958


namespace cubic_extrema_difference_l2379_237993

open Real

/-- The cubic function f(x) with parameters a and b. -/
def f (a b x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x

/-- The derivative of f(x) with respect to x. -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_extrema_difference (a b : ℝ) 
  (h1 : f' a b 2 = 0)
  (h2 : f' a b 1 = -3) :
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b x ≤ f a b x_max) ∧ 
    (∀ x, f a b x_min ≤ f a b x) ∧
    (f a b x_max - f a b x_min = 4) := by
  sorry

end cubic_extrema_difference_l2379_237993


namespace binary_10110100_is_180_l2379_237910

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10110100_is_180 :
  binary_to_decimal [false, false, true, false, true, true, false, true] = 180 := by
  sorry

end binary_10110100_is_180_l2379_237910


namespace intersection_theorem_l2379_237943

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2 + 2}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem : A_intersect_B = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_theorem_l2379_237943


namespace sqrt_product_equality_l2379_237954

theorem sqrt_product_equality : Real.sqrt 125 * Real.sqrt 45 * Real.sqrt 10 = 75 * Real.sqrt 10 := by
  sorry

end sqrt_product_equality_l2379_237954


namespace certain_number_proof_l2379_237992

theorem certain_number_proof (x : ℚ) : 
  x^22 * (1/81)^11 = 1/18^22 → x = 1/36 := by
  sorry

end certain_number_proof_l2379_237992


namespace nonempty_set_implies_nonnegative_a_l2379_237915

theorem nonempty_set_implies_nonnegative_a (a : ℝ) :
  (∅ : Set ℝ) ⊂ {x : ℝ | x^2 ≤ a} → a ∈ Set.Ici (0 : ℝ) := by
  sorry

end nonempty_set_implies_nonnegative_a_l2379_237915


namespace multiply_by_point_nine_l2379_237974

theorem multiply_by_point_nine (x : ℝ) : 0.9 * x = 0.0063 → x = 0.007 := by
  sorry

end multiply_by_point_nine_l2379_237974


namespace expression_value_l2379_237971

theorem expression_value (a b c : ℤ) 
  (eq1 : (25 : ℝ) ^ a * 5 ^ (2 * b) = 5 ^ 6)
  (eq2 : (4 : ℝ) ^ b / 4 ^ c = 4) : 
  a ^ 2 + a * b + 3 * c = 6 := by sorry

end expression_value_l2379_237971


namespace picture_frame_interior_edges_sum_l2379_237924

theorem picture_frame_interior_edges_sum 
  (frame_width : ℝ) 
  (frame_area : ℝ) 
  (outer_edge : ℝ) :
  frame_width = 2 →
  frame_area = 68 →
  outer_edge = 15 →
  ∃ (inner_width inner_height : ℝ),
    inner_width = outer_edge - 2 * frame_width ∧
    frame_area = outer_edge * (inner_height + 2 * frame_width) - inner_width * inner_height ∧
    2 * (inner_width + inner_height) = 26 :=
by sorry

end picture_frame_interior_edges_sum_l2379_237924


namespace minimum_satisfying_number_l2379_237994

def is_multiple_of (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def satisfies_conditions (A : ℕ) : Prop :=
  A > 0 ∧
  is_multiple_of A 3 ∧
  ¬is_multiple_of A 9 ∧
  is_multiple_of (A + digit_product A) 9

theorem minimum_satisfying_number :
  satisfies_conditions 138 ∧ ∀ A : ℕ, A < 138 → ¬satisfies_conditions A :=
sorry

end minimum_satisfying_number_l2379_237994


namespace sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l2379_237903

theorem sum_of_quadratic_roots (a b c : ℚ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_specific_quadratic_roots :
  let a : ℚ := -48
  let b : ℚ := 108
  let c : ℚ := 162
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = 9/4 :=
by sorry

end sum_of_quadratic_roots_sum_of_specific_quadratic_roots_l2379_237903


namespace sock_selection_l2379_237932

theorem sock_selection (n m k : ℕ) : 
  n = 8 → m = 4 → k = 1 →
  (Nat.choose n m) - (Nat.choose (n - k) m) = 35 := by
  sorry

end sock_selection_l2379_237932


namespace quadratic_real_roots_l2379_237942

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x - 4*m = 0) ↔ m ≥ -1/16 := by
  sorry

end quadratic_real_roots_l2379_237942


namespace bobs_improvement_percentage_l2379_237905

theorem bobs_improvement_percentage (bob_time sister_time : ℕ) 
  (h1 : bob_time = 640) 
  (h2 : sister_time = 320) : 
  (bob_time - sister_time) / bob_time * 100 = 50 := by
  sorry

end bobs_improvement_percentage_l2379_237905


namespace eunji_reading_pages_l2379_237966

theorem eunji_reading_pages (pages_tuesday pages_thursday total_pages : ℕ) 
  (h1 : pages_tuesday = 18)
  (h2 : pages_thursday = 23)
  (h3 : total_pages = 60)
  (h4 : pages_tuesday + pages_thursday + (total_pages - pages_tuesday - pages_thursday) = total_pages) :
  total_pages - pages_tuesday - pages_thursday = 19 := by
  sorry

end eunji_reading_pages_l2379_237966


namespace curve_and_line_properties_l2379_237937

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x = -2 ∨ 3*x + 4*y - 2 = 0

-- Define the distance ratio condition
def distance_ratio (x y : ℝ) : Prop :=
  (x^2 + y^2) = (1/4) * ((x - 3)^2 + y^2)

-- Define the intersection condition
def intersects_curve (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), l x₁ y₁ ∧ l x₂ y₂ ∧ c x₁ y₁ ∧ c x₂ y₂ ∧ 
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

-- Main theorem
theorem curve_and_line_properties :
  (∀ x y, curve_C x y ↔ distance_ratio x y) ∧
  (line_l (-2) 2) ∧
  (intersects_curve line_l curve_C) ∧
  (∃ x₁ y₁ x₂ y₂, line_l x₁ y₁ ∧ line_l x₂ y₂ ∧ curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) :=
sorry

end curve_and_line_properties_l2379_237937


namespace axis_of_symmetry_l2379_237931

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the property of f
axiom f_property : ∀ x : ℝ, f x = f (3 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry (3/2) f :=
sorry

end axis_of_symmetry_l2379_237931


namespace triangle_proof_l2379_237950

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_proof (t : Triangle) 
  (h1 : (t.a - t.b + t.c) / t.c = t.b / (t.a + t.b - t.c))
  (h2 : t.b - t.c = (Real.sqrt 3 / 3) * t.a) :
  t.A = π / 3 ∧ t.B = π / 2 := by
  sorry


end triangle_proof_l2379_237950


namespace fraction_simplification_l2379_237955

theorem fraction_simplification : 
  (3+6-12+24+48-96+192) / (6+12-24+48+96-192+384) = 1/2 := by
  sorry

end fraction_simplification_l2379_237955


namespace f_properties_imply_l2379_237900

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x > 0 → f x < 0) ∧
  (f 1 = -2)

theorem f_properties_imply (f : ℝ → ℝ) (h : f_properties f) :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ ∀ y ∈ Set.Icc (-3) 3, f y ≤ f x ∧ f x = 6) ∧
  (∃ x : ℝ, x ∈ Set.Icc (-3) 3 ∧ ∀ y ∈ Set.Icc (-3) 3, f y ≥ f x ∧ f x = -6) :=
by sorry

end f_properties_imply_l2379_237900


namespace min_value_quadratic_sum_l2379_237968

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ ∀ (a b c : ℝ), a + 2*b + c = 1 → x^2 + 4*y^2 + z^2 ≥ m :=
sorry

end min_value_quadratic_sum_l2379_237968


namespace arithmetic_evaluation_l2379_237911

theorem arithmetic_evaluation : (7 + 5 + 8) / 3 - 2 / 3 + 1 = 7 := by
  sorry

end arithmetic_evaluation_l2379_237911


namespace age_of_b_l2379_237969

/-- Given three people A, B, and C, their average age, and the average age of A and C, prove the age of B. -/
theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 27 →  -- The average age of A, B, and C is 27
  (a + c) / 2 = 29 →      -- The average age of A and C is 29
  b = 23 :=               -- The age of B is 23
by sorry

end age_of_b_l2379_237969


namespace total_employees_l2379_237918

theorem total_employees (part_time full_time : ℕ) 
  (h1 : part_time = 2041) 
  (h2 : full_time = 63093) : 
  part_time + full_time = 65134 := by
  sorry

end total_employees_l2379_237918


namespace prime_triple_equation_l2379_237919

theorem prime_triple_equation (p q n : ℕ) : 
  p.Prime → q.Prime → p > 0 → q > 0 → n > 0 →
  p * (p + 1) + q * (q + 1) = n * (n + 1) →
  ((p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 3 ∧ q = 5 ∧ n = 6)) :=
by sorry

end prime_triple_equation_l2379_237919


namespace percent_decrease_proof_l2379_237988

theorem percent_decrease_proof (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 50) :
  (original_price - sale_price) / original_price * 100 = 50 := by
  sorry

end percent_decrease_proof_l2379_237988


namespace student_count_l2379_237925

theorem student_count (initial_avg : ℚ) (wrong_mark : ℚ) (correct_mark : ℚ) (final_avg : ℚ) :
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  final_avg = 98 →
  ∃ n : ℕ, n > 0 ∧ n * final_avg = n * initial_avg - (wrong_mark - correct_mark) :=
by
  sorry

end student_count_l2379_237925


namespace problem_solution_l2379_237917

theorem problem_solution (a b c : ℝ) 
  (h1 : a + b + c = 150)
  (h2 : a + 10 = b - 5)
  (h3 : b - 5 = c^2) :
  b = (1322 - 2 * Real.sqrt 1241) / 16 := by
sorry

end problem_solution_l2379_237917


namespace group_bill_proof_l2379_237982

def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

theorem group_bill_proof (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ)
  (h1 : total_people = 13)
  (h2 : num_kids = 9)
  (h3 : adult_meal_cost = 7) :
  restaurant_bill total_people num_kids adult_meal_cost = 28 := by
  sorry

end group_bill_proof_l2379_237982


namespace ratio_odd_even_divisors_M_l2379_237935

def M : ℕ := 33 * 38 * 58 * 462

/-- The sum of odd divisors of a natural number n -/
def sum_odd_divisors (n : ℕ) : ℕ := sorry

/-- The sum of even divisors of a natural number n -/
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors_M :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 14 := by sorry

end ratio_odd_even_divisors_M_l2379_237935


namespace linear_system_solution_l2379_237913

/-- Given a system of linear equations 2x + my = 5 and nx - 3y = 2,
    if the augmented matrix transforms to [[1, 0, 3], [0, 1, 1]],
    then m/n = -3/5 -/
theorem linear_system_solution (m n : ℚ) : 
  (∃ x y : ℚ, 2*x + m*y = 5 ∧ n*x - 3*y = 2) →
  (∃ x y : ℚ, x = 3 ∧ y = 1) →
  m/n = -3/5 := by
  sorry

end linear_system_solution_l2379_237913


namespace addition_subtraction_problem_l2379_237984

theorem addition_subtraction_problem : (5.75 + 3.09) - 1.86 = 6.98 := by
  sorry

end addition_subtraction_problem_l2379_237984


namespace number_multiplied_by_9999_l2379_237989

theorem number_multiplied_by_9999 : ∃ x : ℚ, x * 9999 = 724787425 ∧ x = 72487.5 := by
  sorry

end number_multiplied_by_9999_l2379_237989


namespace count_threes_up_to_80_l2379_237970

/-- Count of digit 3 in a single number -/
def countThreesInNumber (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 3 then 1 + countThreesInNumber (n / 10)
  else countThreesInNumber (n / 10)

/-- Count of digit 3 in numbers from 1 to n -/
def countThreesUpTo (n : ℕ) : ℕ :=
  List.range n |> List.map (fun i => countThreesInNumber (i + 1)) |> List.sum

/-- The count of the digit 3 in the numbers from 1 to 80 (inclusive) is equal to 9 -/
theorem count_threes_up_to_80 : countThreesUpTo 80 = 9 := by
  sorry

end count_threes_up_to_80_l2379_237970


namespace cube_root_equation_product_l2379_237965

theorem cube_root_equation_product (a b : ℤ) : 
  (3 * Real.sqrt (Real.rpow 5 (1/3) - Real.rpow 4 (1/3)) = Real.rpow a (1/3) + Real.rpow b (1/3) + Real.rpow 2 (1/3)) →
  a * b = -500 := by
sorry

end cube_root_equation_product_l2379_237965


namespace total_selling_price_l2379_237964

def bicycle_cost : ℚ := 1600
def scooter_cost : ℚ := 8000
def motorcycle_cost : ℚ := 15000

def bicycle_loss_percent : ℚ := 10
def scooter_loss_percent : ℚ := 5
def motorcycle_loss_percent : ℚ := 8

def selling_price (cost : ℚ) (loss_percent : ℚ) : ℚ :=
  cost - (cost * loss_percent / 100)

theorem total_selling_price :
  selling_price bicycle_cost bicycle_loss_percent +
  selling_price scooter_cost scooter_loss_percent +
  selling_price motorcycle_cost motorcycle_loss_percent = 22840 := by
  sorry

end total_selling_price_l2379_237964


namespace pool_width_calculation_l2379_237921

/-- Represents a rectangular swimming pool with a surrounding deck -/
structure PoolWithDeck where
  poolLength : ℝ
  poolWidth : ℝ
  deckWidth : ℝ

/-- Calculates the total area of the pool and deck -/
def totalArea (p : PoolWithDeck) : ℝ :=
  (p.poolLength + 2 * p.deckWidth) * (p.poolWidth + 2 * p.deckWidth)

/-- Theorem stating the width of the pool given specific conditions -/
theorem pool_width_calculation (p : PoolWithDeck) 
    (h1 : p.poolLength = 20)
    (h2 : p.deckWidth = 3)
    (h3 : totalArea p = 728) :
    p.poolWidth = 572 / 46 := by
  sorry

end pool_width_calculation_l2379_237921
