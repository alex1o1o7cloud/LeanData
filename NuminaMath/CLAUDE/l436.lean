import Mathlib

namespace total_area_parallelogram_triangle_l436_43611

/-- The total area of a shape consisting of a parallelogram and an adjacent right triangle -/
theorem total_area_parallelogram_triangle (angle : Real) (side1 side2 : Real) (leg : Real) : 
  angle = 150 * π / 180 →
  side1 = 10 →
  side2 = 24 →
  leg = 10 →
  (side1 * side2 * Real.sin angle) / 2 + (side2 * leg) / 2 = 170 := by sorry

end total_area_parallelogram_triangle_l436_43611


namespace max_value_of_sum_of_square_roots_l436_43655

theorem max_value_of_sum_of_square_roots (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) 
  (sum_constraint : a + b + c = 8) : 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 26 ∧ 
  ∃ a' b' c' : ℝ, a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 8 ∧
  Real.sqrt (3 * a' + 2) + Real.sqrt (3 * b' + 2) + Real.sqrt (3 * c' + 2) = 3 * Real.sqrt 26 :=
by sorry

end max_value_of_sum_of_square_roots_l436_43655


namespace sum_of_m_values_is_correct_l436_43650

/-- The sum of all possible values of m for which the polynomials x^2 - 6x + 8 and x^2 - 7x + m have a root in common -/
def sum_of_m_values : ℝ := 22

/-- First polynomial: x^2 - 6x + 8 -/
def p1 (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- Second polynomial: x^2 - 7x + m -/
def p2 (x m : ℝ) : ℝ := x^2 - 7*x + m

/-- Theorem stating that the sum of all possible values of m for which p1 and p2 have a common root is equal to sum_of_m_values -/
theorem sum_of_m_values_is_correct : 
  (∃ m1 m2 : ℝ, m1 ≠ m2 ∧ 
    (∃ x1 : ℝ, p1 x1 = 0 ∧ p2 x1 m1 = 0) ∧
    (∃ x2 : ℝ, p1 x2 = 0 ∧ p2 x2 m2 = 0) ∧
    m1 + m2 = sum_of_m_values) :=
by sorry

end sum_of_m_values_is_correct_l436_43650


namespace root_sum_ratio_l436_43649

theorem root_sum_ratio (a b c d : ℝ) (h1 : a ≠ 0) (h2 : d = 0)
  (h3 : a * (4 : ℝ)^3 + b * (4 : ℝ)^2 + c * (4 : ℝ) + d = 0)
  (h4 : a * (-3 : ℝ)^3 + b * (-3 : ℝ)^2 + c * (-3 : ℝ) + d = 0) :
  (b + c) / a = -13 := by
  sorry

end root_sum_ratio_l436_43649


namespace inscribed_squares_ratio_l436_43624

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (c - y) = (a - y) * (b - y) →
  x / y = 5 / 13 := by sorry

end inscribed_squares_ratio_l436_43624


namespace sin_2alpha_value_l436_43621

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos (π - α) = 1/3) :
  Real.sin (2 * α) = 8/9 := by
  sorry

end sin_2alpha_value_l436_43621


namespace sandwich_combinations_l436_43613

theorem sandwich_combinations (n_meat : Nat) (n_cheese : Nat) (n_bread : Nat) :
  n_meat = 12 →
  n_cheese = 11 →
  n_bread = 5 →
  (n_meat.choose 2) * (n_cheese.choose 2) * (n_bread.choose 1) = 18150 := by
  sorry

end sandwich_combinations_l436_43613


namespace complex_magnitude_in_complement_range_l436_43691

theorem complex_magnitude_in_complement_range (a : ℝ) : 
  let z : ℂ := 1 + a * Complex.I
  let M : Set ℝ := {x | x > 2}
  Complex.abs z ∈ (Set.Iic 2) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end complex_magnitude_in_complement_range_l436_43691


namespace number_and_square_average_l436_43612

theorem number_and_square_average (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5*x) : x = 9 := by
  sorry

end number_and_square_average_l436_43612


namespace jiuquan_location_accuracy_l436_43635

-- Define the possible location descriptions
inductive LocationDescription
  | NorthwestOfBeijing
  | LatitudeOnly (lat : Float)
  | LongitudeOnly (long : Float)
  | LatitudeLongitude (lat : Float) (long : Float)

-- Define the accuracy of a location description
def isAccurateLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.LatitudeLongitude _ _ => True
  | _ => False

-- Theorem statement
theorem jiuquan_location_accuracy :
  ∀ (desc : LocationDescription),
    isAccurateLocation desc ↔
      desc = LocationDescription.LatitudeLongitude 39.75 98.52 :=
by sorry

end jiuquan_location_accuracy_l436_43635


namespace circle_center_sum_l436_43651

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the coordinates of its center is -1. -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9) ∧ h + k = -1) :=
by sorry

end circle_center_sum_l436_43651


namespace lexie_age_difference_l436_43656

/-- Given information about Lexie, her sister, and her brother's ages, prove the age difference between Lexie and her brother. -/
theorem lexie_age_difference (lexie_age : ℕ) (sister_age : ℕ) (brother_age : ℕ)
  (h1 : lexie_age = 8)
  (h2 : sister_age = 2 * lexie_age)
  (h3 : sister_age - brother_age = 14) :
  lexie_age - brother_age = 6 := by
  sorry

end lexie_age_difference_l436_43656


namespace discount_calculation_l436_43648

/-- The discount calculation problem --/
theorem discount_calculation (original_cost spent : ℝ) 
  (h1 : original_cost = 35)
  (h2 : spent = 18) : 
  original_cost - spent = 17 := by
  sorry

end discount_calculation_l436_43648


namespace expression_evaluation_l436_43661

theorem expression_evaluation (x : ℝ) (h1 : x^6 ≠ -1) (h2 : x^6 ≠ 1) :
  ((((x^2 + 1)^2 * (x^4 - x^2 + 1)^2) / (x^6 + 1)^2)^2 *
   (((x^2 - 1)^2 * (x^4 + x^2 + 1)^2) / (x^6 - 1)^2)^2) = 1 := by
  sorry


end expression_evaluation_l436_43661


namespace problem_1_problem_2_l436_43674

-- Problem 1
theorem problem_1 : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 6 / Real.sqrt 2 = 2 + Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x = Real.sqrt 2 - 2) : 
  ((1 / (x - 1) - 1 / (x + 1)) / ((x + 2) / (x^2 - 1))) = Real.sqrt 2 := by
  sorry

end problem_1_problem_2_l436_43674


namespace unique_function_solution_l436_43657

theorem unique_function_solution (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (1 + x * y) - f (x + y) = f x * f y) 
  (h2 : f (-1) ≠ 0) : 
  ∀ x : ℝ, f x = x - 1 := by
sorry

end unique_function_solution_l436_43657


namespace partial_fraction_decomposition_l436_43682

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), 
    (∀ x : ℚ, x ≠ 9 ∧ x ≠ -4 →
      (6 * x + 5) / (x^2 - 5*x - 36) = C / (x - 9) + D / (x + 4)) ∧
    C = 59 / 13 ∧
    D = 19 / 13 := by
  sorry

end partial_fraction_decomposition_l436_43682


namespace arithmetic_computation_l436_43668

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 8 * 2 / 4 = 17 := by
  sorry

end arithmetic_computation_l436_43668


namespace good_games_count_l436_43695

def games_from_friend : ℕ := 11
def games_from_garage_sale : ℕ := 22
def non_working_games : ℕ := 19

theorem good_games_count :
  games_from_friend + games_from_garage_sale - non_working_games = 14 := by
  sorry

end good_games_count_l436_43695


namespace extremum_and_monotonicity_l436_43639

/-- The function f(x) defined as e^x - ln(x + m) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (x + m)

/-- The derivative of f(x) with respect to x -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 / (x + m)

theorem extremum_and_monotonicity (m : ℝ) :
  (f_deriv m 0 = 0) →
  (m = 1) ∧
  (∀ x > 0, f_deriv m x > 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f_deriv m x < 0) := by
  sorry

#check extremum_and_monotonicity

end extremum_and_monotonicity_l436_43639


namespace arithmetic_sqrt_of_sqrt_16_l436_43610

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt (Real.sqrt x)

-- State the theorem
theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt 16 = 2 := by sorry

end arithmetic_sqrt_of_sqrt_16_l436_43610


namespace bob_water_percentage_l436_43644

-- Define the water requirements for each crop
def water_corn : ℕ := 20
def water_cotton : ℕ := 80
def water_beans : ℕ := 2 * water_corn

-- Define the acreage for each farmer
def bob_corn : ℕ := 3
def bob_cotton : ℕ := 9
def bob_beans : ℕ := 12

def brenda_corn : ℕ := 6
def brenda_cotton : ℕ := 7
def brenda_beans : ℕ := 14

def bernie_corn : ℕ := 2
def bernie_cotton : ℕ := 12

-- Calculate water usage for each farmer
def bob_water : ℕ := bob_corn * water_corn + bob_cotton * water_cotton + bob_beans * water_beans
def brenda_water : ℕ := brenda_corn * water_corn + brenda_cotton * water_cotton + brenda_beans * water_beans
def bernie_water : ℕ := bernie_corn * water_corn + bernie_cotton * water_cotton

-- Calculate total water usage
def total_water : ℕ := bob_water + brenda_water + bernie_water

-- Define the theorem
theorem bob_water_percentage :
  (bob_water : ℚ) / total_water * 100 = 36 := by sorry

end bob_water_percentage_l436_43644


namespace probability_three_players_complete_theorem_l436_43641

/-- The probability that after N matches in a 5-player round-robin tournament,
    there are at least 3 players who have played all their matches against each other. -/
def probability_three_players_complete (N : ℕ) : ℚ :=
  if N < 3 then 0
  else if N = 3 then 1/12
  else if N = 4 then 1/3
  else if N = 5 then 5/7
  else if N = 6 then 20/21
  else 1

/-- The theorem stating the probability of having at least 3 players who have played
    all their matches against each other after N matches in a 5-player round-robin tournament. -/
theorem probability_three_players_complete_theorem (N : ℕ) (h : 3 ≤ N ∧ N ≤ 10) :
  probability_three_players_complete N =
    if N < 3 then 0
    else if N = 3 then 1/12
    else if N = 4 then 1/3
    else if N = 5 then 5/7
    else if N = 6 then 20/21
    else 1 := by sorry

end probability_three_players_complete_theorem_l436_43641


namespace rhombus_area_l436_43693

/-- The area of a rhombus with vertices at (0, 4.5), (8, 0), (0, -4.5), and (-8, 0) is 72 square units. -/
theorem rhombus_area : ℝ := by
  -- Define the vertices of the rhombus
  let v1 : ℝ × ℝ := (0, 4.5)
  let v2 : ℝ × ℝ := (8, 0)
  let v3 : ℝ × ℝ := (0, -4.5)
  let v4 : ℝ × ℝ := (-8, 0)

  -- Define the diagonals of the rhombus
  let d1 : ℝ := ‖v1.2 - v3.2‖ -- Distance between y-coordinates of v1 and v3
  let d2 : ℝ := ‖v2.1 - v4.1‖ -- Distance between x-coordinates of v2 and v4

  -- Calculate the area of the rhombus
  let area : ℝ := (d1 * d2) / 2

  -- Prove that the area is 72 square units
  sorry

end rhombus_area_l436_43693


namespace movie_theater_revenue_l436_43629

/-- 
Calculates the total revenue from movie ticket sales given the prices and quantities sold.
-/
theorem movie_theater_revenue 
  (matinee_price : ℕ) 
  (evening_price : ℕ) 
  (three_d_price : ℕ)
  (matinee_quantity : ℕ)
  (evening_quantity : ℕ)
  (three_d_quantity : ℕ)
  (h1 : matinee_price = 5)
  (h2 : evening_price = 12)
  (h3 : three_d_price = 20)
  (h4 : matinee_quantity = 200)
  (h5 : evening_quantity = 300)
  (h6 : three_d_quantity = 100) :
  matinee_price * matinee_quantity + 
  evening_price * evening_quantity + 
  three_d_price * three_d_quantity = 6600 :=
by
  sorry

#check movie_theater_revenue

end movie_theater_revenue_l436_43629


namespace binomial_18_10_l436_43671

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 8008) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end binomial_18_10_l436_43671


namespace berry_cobbler_problem_l436_43616

theorem berry_cobbler_problem (total_needed : ℕ) (blueberries : ℕ) (to_buy : ℕ) 
  (h1 : total_needed = 21)
  (h2 : blueberries = 8)
  (h3 : to_buy = 9) :
  total_needed - (blueberries + to_buy) = 4 := by
  sorry

end berry_cobbler_problem_l436_43616


namespace sum_of_digits_N_l436_43672

def N : ℕ := 9 + 99 + 999 + 9999 + 99999 + 999999

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_N : sum_of_digits N = 9 := by
  sorry

end sum_of_digits_N_l436_43672


namespace square_and_circle_measurements_l436_43609

/-- Given a square with side length 70√2 cm and a circle with diameter equal to the square's diagonal,
    prove the square's diagonal length and the circle's circumference. -/
theorem square_and_circle_measurements :
  let square_side : ℝ := 70 * Real.sqrt 2
  let square_diagonal : ℝ := square_side * Real.sqrt 2
  let circle_diameter : ℝ := square_diagonal
  let circle_circumference : ℝ := π * circle_diameter
  (square_diagonal = 140) ∧ (circle_circumference = 140 * π) := by sorry

end square_and_circle_measurements_l436_43609


namespace fraction_problem_l436_43681

theorem fraction_problem (x y : ℕ) (h1 : x + y = 122) (h2 : (x - 19) / (y - 19) = 1 / 5) :
  x / y = 33 / 89 := by
  sorry

end fraction_problem_l436_43681


namespace maximize_sqrt_expression_l436_43618

theorem maximize_sqrt_expression :
  let add := Real.sqrt 8 + Real.sqrt 2
  let mul := Real.sqrt 8 * Real.sqrt 2
  let div := Real.sqrt 8 / Real.sqrt 2
  let sub := Real.sqrt 8 - Real.sqrt 2
  add > mul ∧ add > div ∧ add > sub := by
  sorry

end maximize_sqrt_expression_l436_43618


namespace probability_sum_six_three_dice_l436_43606

/-- A function that returns the number of ways to roll a sum of 6 with three dice -/
def waysToRollSixWithThreeDice : ℕ :=
  -- We don't implement the function, just declare it
  sorry

/-- The total number of possible outcomes when rolling three six-sided dice -/
def totalOutcomes : ℕ := 6^3

/-- The probability of rolling a sum of 6 with three fair six-sided dice -/
theorem probability_sum_six_three_dice :
  (waysToRollSixWithThreeDice : ℚ) / totalOutcomes = 5 / 108 := by
  sorry

end probability_sum_six_three_dice_l436_43606


namespace even_decreasing_function_inequality_l436_43663

noncomputable def e : ℝ := Real.exp 1

theorem even_decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x ≥ 0, ∀ y ≥ x, f x ≥ f y) →
  (∀ x ∈ Set.Icc 1 3, f (-a * x + Real.log x + 1) + f (a * x - Real.log x - 1) ≥ 2 * f 1) →
  a ∈ Set.Icc (1 / e) ((2 + Real.log 3) / 3) :=
sorry

end even_decreasing_function_inequality_l436_43663


namespace simplify_expression_l436_43608

theorem simplify_expression (x y : ℝ) (h : x ≠ y) :
  (x - y)^3 / (x - y)^2 * (y - x) = -(x - y)^2 := by sorry

end simplify_expression_l436_43608


namespace basketball_game_scores_l436_43633

/-- Represents the scores of a team in a basketball game -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a sequence of four numbers is an arithmetic progression -/
def isArithmeticSequence (s : TeamScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3 ∧ s.q2 > s.q1

/-- Checks if a sequence of four numbers is a geometric progression -/
def isGeometricSequence (s : TeamScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- The main theorem statement -/
theorem basketball_game_scores 
  (falcons tigers : TeamScores) 
  (h1 : falcons.q1 = tigers.q1)
  (h2 : isArithmeticSequence falcons)
  (h3 : isGeometricSequence tigers)
  (h4 : falcons.q1 + falcons.q2 + falcons.q3 + falcons.q4 = 
        tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 + 2)
  (h5 : falcons.q1 + falcons.q2 + falcons.q3 + falcons.q4 ≤ 100)
  (h6 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 100) :
  falcons.q1 + falcons.q2 + tigers.q1 + tigers.q2 = 14 :=
by
  sorry


end basketball_game_scores_l436_43633


namespace geometric_sum_2_power_63_l436_43652

theorem geometric_sum_2_power_63 : 
  (Finset.range 64).sum (fun i => 2^i) = 2^64 - 1 :=
by sorry

end geometric_sum_2_power_63_l436_43652


namespace function_satisfies_condition_l436_43667

/-- The function f(x) = 1/x - x satisfies the given condition for all x₁, x₂ in (0, +∞) where x₁ ≠ x₂ -/
theorem function_satisfies_condition :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ →
  (x₁ - x₂) * ((1 / x₁ - x₁) - (1 / x₂ - x₂)) < 0 := by
  sorry


end function_satisfies_condition_l436_43667


namespace isoscelesTriangles29Count_l436_43669

/-- An isosceles triangle with integer side lengths and perimeter 29 -/
structure IsoscelesTriangle29 where
  base : ℕ
  side : ℕ
  isIsosceles : side * 2 + base = 29
  isTriangle : base < side + side

/-- The count of valid isosceles triangles with perimeter 29 -/
def countIsoscelesTriangles29 : ℕ := sorry

/-- Theorem stating that there are exactly 5 isosceles triangles with integer side lengths and perimeter 29 -/
theorem isoscelesTriangles29Count : countIsoscelesTriangles29 = 5 := by sorry

end isoscelesTriangles29Count_l436_43669


namespace complement_of_A_in_U_l436_43658

def U : Set Nat := {2, 4, 5, 7, 8}
def A : Set Nat := {4, 8}

theorem complement_of_A_in_U :
  (U \ A) = {2, 5, 7} := by sorry

end complement_of_A_in_U_l436_43658


namespace product_sum_theorem_l436_43698

theorem product_sum_theorem (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a + b + c = 20) : 
  a*b + b*c + a*c = 131 := by
sorry

end product_sum_theorem_l436_43698


namespace prime_difference_theorem_l436_43676

theorem prime_difference_theorem (x y : ℝ) 
  (h1 : Prime (⌊x - y⌋ : ℤ))
  (h2 : Prime (⌊x^2 - y^2⌋ : ℤ))
  (h3 : Prime (⌊x^3 - y^3⌋ : ℤ)) :
  x - y = 3 := by
  sorry

end prime_difference_theorem_l436_43676


namespace tan_cot_15_sum_even_l436_43640

theorem tan_cot_15_sum_even (n : ℕ+) : 
  ∃ k : ℤ, (2 - Real.sqrt 3) ^ n.val + (2 + Real.sqrt 3) ^ n.val = 2 * k := by
  sorry

end tan_cot_15_sum_even_l436_43640


namespace minimum_loss_for_1997_pills_l436_43697

/-- Represents a bottle of medicine --/
structure Bottle where
  capacity : ℕ
  pills : ℕ

/-- Represents the state of all bottles --/
structure State where
  a : Bottle
  b : Bottle
  c : Bottle
  loss : ℕ

/-- Calculates the minimum total loss of active ingredient --/
def minimumTotalLoss (initialPills : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum total loss for the given problem --/
theorem minimum_loss_for_1997_pills :
  minimumTotalLoss 1997 = 32401 := by
  sorry

#check minimum_loss_for_1997_pills

end minimum_loss_for_1997_pills_l436_43697


namespace quadratic_polynomial_symmetry_l436_43647

theorem quadratic_polynomial_symmetry (P : ℝ → ℝ) (h : ∃ a b c : ℝ, P x = a * x^2 + b * x + c) :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    P (b + c) = P a ∧ P (c + a) = P b ∧ P (a + b) = P c :=
by sorry

end quadratic_polynomial_symmetry_l436_43647


namespace average_first_12_even_numbers_l436_43692

theorem average_first_12_even_numbers : 
  let first_12_even : List ℕ := List.range 12 |>.map (fun n => 2 * (n + 1))
  (first_12_even.sum / first_12_even.length : ℚ) = 13 := by
  sorry

end average_first_12_even_numbers_l436_43692


namespace max_edges_bipartite_graph_l436_43689

/-- 
Given a complete bipartite graph K_{m,n} where m and n are positive integers and m + n = 21,
prove that the maximum number of edges is 110.
-/
theorem max_edges_bipartite_graph : 
  ∀ m n : ℕ+, 
  m + n = 21 → 
  ∃ (max_edges : ℕ), 
    max_edges = m * n ∧ 
    ∀ k l : ℕ+, k + l = 21 → k * l ≤ max_edges :=
by
  sorry

end max_edges_bipartite_graph_l436_43689


namespace final_number_is_one_l436_43642

def initialSum : ℕ := (1988 * 1989) / 2

def operationA (numbers : List ℕ) (d : ℕ) : List ℕ :=
  numbers.map (λ x => if x ≥ d then x - d else 0)

def operationB (numbers : List ℕ) : List ℕ :=
  match numbers with
  | x :: y :: rest => (x + y) :: rest
  | _ => numbers

def performOperations (numbers : List ℕ) (iterations : ℕ) : ℕ :=
  if iterations = 0 then
    match numbers with
    | [x] => x
    | _ => 0
  else
    let numbersAfterA := operationA numbers 1
    let numbersAfterB := operationB numbersAfterA
    performOperations numbersAfterB (iterations - 1)

theorem final_number_is_one :
  performOperations (List.range 1989) 1987 = 1 :=
sorry

end final_number_is_one_l436_43642


namespace binomial_expansion_coefficients_l436_43653

theorem binomial_expansion_coefficients 
  (a b : ℝ) (n : ℕ) 
  (h1 : (1 + b) ^ n = 243)
  (h2 : (1 + |a|) ^ n = 32) : 
  a = 1 ∧ b = 2 ∧ n = 5 := by
sorry

end binomial_expansion_coefficients_l436_43653


namespace probability_of_decagon_side_l436_43677

/-- A regular decagon -/
def RegularDecagon : Type := Unit

/-- A triangle formed by three vertices of a regular decagon -/
def DecagonTriangle : Type := Fin 3 → Fin 10

/-- Predicate to check if a DecagonTriangle has at least one side that is also a side of the decagon -/
def HasDecagonSide (t : DecagonTriangle) : Prop := sorry

/-- The set of all possible DecagonTriangles -/
def AllDecagonTriangles : Finset DecagonTriangle := sorry

/-- The set of DecagonTriangles that have at least one side that is also a side of the decagon -/
def TrianglesWithDecagonSide : Finset DecagonTriangle := sorry

/-- The probability of selecting a DecagonTriangle that has at least one side that is also a side of the decagon -/
def ProbabilityOfDecagonSide : ℚ := Finset.card TrianglesWithDecagonSide / Finset.card AllDecagonTriangles

theorem probability_of_decagon_side :
  ProbabilityOfDecagonSide = 7 / 12 :=
sorry

end probability_of_decagon_side_l436_43677


namespace cubic_function_increasing_iff_a_nonpositive_l436_43631

/-- Theorem: For the function f(x) = x^3 - ax + 1 where a ∈ ℝ, 
    f(x) is increasing in its domain if and only if a ≤ 0 -/
theorem cubic_function_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => x^3 - a*x + 1) (3*x^2 - a) x) →
  (∀ x y : ℝ, x < y → (x^3 - a*x + 1) < (y^3 - a*y + 1)) ↔ a ≤ 0 := by
  sorry

end cubic_function_increasing_iff_a_nonpositive_l436_43631


namespace symbol_equations_l436_43699

theorem symbol_equations :
  ∀ (triangle circle square star : ℤ),
  triangle = circle + 2 →
  square = triangle + triangle →
  star = triangle + square + 5 →
  star = circle + 31 →
  triangle = 12 ∧ circle = 10 ∧ square = 24 ∧ star = 41 := by
sorry

end symbol_equations_l436_43699


namespace triangle_properties_l436_43678

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_acute_triangle (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfies_equation (t : Triangle) : Prop :=
  (Real.sqrt 3 * t.c) / (t.b * Real.cos t.A) = Real.tan t.A + Real.tan t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h_acute : is_acute_triangle t)
  (h_eq : satisfies_equation t) :
  t.B = Real.pi/3 ∧ 
  (t.c = 4 → 2 * Real.sqrt 3 < (1/2 * t.a * t.c * Real.sin t.B) ∧ 
                (1/2 * t.a * t.c * Real.sin t.B) < 8 * Real.sqrt 3) := by
  sorry

end triangle_properties_l436_43678


namespace tom_tickets_l436_43683

/-- The number of tickets Tom has left after playing games and spending some tickets -/
def tickets_left (whack_a_mole skee_ball ring_toss hat plush_toy : ℕ) : ℕ :=
  (whack_a_mole + skee_ball + ring_toss) - (hat + plush_toy)

/-- Theorem stating that Tom is left with 100 tickets -/
theorem tom_tickets : 
  tickets_left 45 38 52 12 23 = 100 := by
  sorry

end tom_tickets_l436_43683


namespace rectangular_plot_perimeter_l436_43632

/-- Given a rectangular plot with length 10 meters more than width,
    and fencing cost of Rs. 910 at Rs. 6.5 per meter,
    prove that the perimeter is 140 meters. -/
theorem rectangular_plot_perimeter : 
  ∀ (width length : ℝ),
  length = width + 10 →
  910 = (2 * (length + width)) * 6.5 →
  2 * (length + width) = 140 :=
by
  sorry

end rectangular_plot_perimeter_l436_43632


namespace unique_solution_floor_ceiling_l436_43688

theorem unique_solution_floor_ceiling (a : ℝ) :
  (⌊a⌋ = 3 * a + 6) ∧ (⌈a⌉ = 4 * a + 9) → a = -3 := by
  sorry

end unique_solution_floor_ceiling_l436_43688


namespace martha_cards_l436_43634

/-- The number of cards Martha ends up with after receiving more cards -/
def final_cards (start : ℕ) (received : ℕ) : ℕ :=
  start + received

/-- Theorem stating that Martha ends up with 79 cards -/
theorem martha_cards : final_cards 3 76 = 79 := by
  sorry

end martha_cards_l436_43634


namespace water_equals_sugar_in_new_recipe_l436_43628

/-- Represents a recipe with ratios of flour, water, and sugar -/
structure Recipe :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Creates a new recipe by doubling the flour to water ratio and halving the flour to sugar ratio -/
def newRecipe (r : Recipe) : Recipe :=
  { flour := r.flour * 2,
    water := r.water,
    sugar := r.sugar / 2 }

/-- Calculates the amount of water needed given the amount of sugar and the recipe ratios -/
def waterNeeded (r : Recipe) (sugarAmount : ℚ) : ℚ :=
  (r.water / r.sugar) * sugarAmount

theorem water_equals_sugar_in_new_recipe (originalRecipe : Recipe) (sugarAmount : ℚ) :
  let newRecipe := newRecipe originalRecipe
  waterNeeded newRecipe sugarAmount = sugarAmount :=
by sorry

#check water_equals_sugar_in_new_recipe

end water_equals_sugar_in_new_recipe_l436_43628


namespace integral_sum_reciprocal_and_semicircle_l436_43685

open Real MeasureTheory

theorem integral_sum_reciprocal_and_semicircle :
  ∫ x in (1 : ℝ)..3, (1 / x + Real.sqrt (1 - (x - 2)^2)) = Real.log 3 + π / 2 := by
  sorry

end integral_sum_reciprocal_and_semicircle_l436_43685


namespace perfect_square_fraction_l436_43627

theorem perfect_square_fraction (m n : ℕ+) : 
  ∃ k : ℕ, (m + n : ℝ)^2 / (4 * (m : ℝ) * (m - n : ℝ)^2 + 4) = (k : ℝ)^2 := by
  sorry

end perfect_square_fraction_l436_43627


namespace quadratic_single_intersection_l436_43686

theorem quadratic_single_intersection (m : ℝ) : 
  (∃! x, (m + 1) * x^2 - 2*(m + 1) * x - 1 = 0) ↔ m = -2 :=
sorry

end quadratic_single_intersection_l436_43686


namespace purchase_system_of_equations_l436_43662

/-- Represents the purchase of basketballs and soccer balls -/
structure PurchaseInfo where
  basketball_price : ℝ
  soccer_ball_price : ℝ
  basketball_count : ℕ
  soccer_ball_count : ℕ
  total_cost : ℝ
  price_difference : ℝ

/-- The system of equations for the purchase -/
def purchase_equations (p : PurchaseInfo) : Prop :=
  p.basketball_count * p.basketball_price + p.soccer_ball_count * p.soccer_ball_price = p.total_cost ∧
  p.basketball_price - p.soccer_ball_price = p.price_difference

theorem purchase_system_of_equations (p : PurchaseInfo) 
  (h1 : p.basketball_count = 3)
  (h2 : p.soccer_ball_count = 2)
  (h3 : p.total_cost = 474)
  (h4 : p.price_difference = 8) :
  purchase_equations p ↔ 
  (3 * p.basketball_price + 2 * p.soccer_ball_price = 474 ∧
   p.basketball_price - p.soccer_ball_price = 8) :=
by sorry

end purchase_system_of_equations_l436_43662


namespace gross_profit_percentage_l436_43645

theorem gross_profit_percentage 
  (selling_price : ℝ) 
  (wholesale_cost : ℝ) 
  (h1 : selling_price = 28) 
  (h2 : wholesale_cost = 25) : 
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 12 := by
sorry

end gross_profit_percentage_l436_43645


namespace range_of_f_l436_43619

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f : Set.range f = Set.Icc (-9 : ℝ) 9 := by sorry

end range_of_f_l436_43619


namespace smallest_number_l436_43660

theorem smallest_number (s : Set ℚ) (h : s = {-1, 0, -3, -2}) : 
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = -3 := by
  sorry

end smallest_number_l436_43660


namespace max_log_sin_l436_43684

open Real

theorem max_log_sin (x : ℝ) (h : 0 < x ∧ x < π) : 
  ∃ c : ℝ, c = 0 ∧ ∀ y : ℝ, 0 < y ∧ y < π → log (sin y) ≤ c :=
by sorry

end max_log_sin_l436_43684


namespace valid_choices_count_l436_43670

/-- The number of elements in the list -/
def n : ℕ := 2016

/-- The number of elements to be shuffled -/
def m : ℕ := 2014

/-- Function to calculate the number of valid ways to choose a and b -/
def count_valid_choices : ℕ := sorry

/-- Theorem stating that the number of valid choices is equal to 508536 -/
theorem valid_choices_count : count_valid_choices = 508536 := by sorry

end valid_choices_count_l436_43670


namespace football_cost_l436_43615

/-- The cost of a football given the total amount paid, change received, and cost of a baseball. -/
theorem football_cost (total_paid : ℝ) (change : ℝ) (baseball_cost : ℝ) 
  (h1 : total_paid = 20)
  (h2 : change = 4.05)
  (h3 : baseball_cost = 6.81) : 
  total_paid - change - baseball_cost = 9.14 := by
  sorry

#check football_cost

end football_cost_l436_43615


namespace problem_statement_l436_43605

theorem problem_statement (d : ℕ) (h : d = 4) :
  (d^d - d*(d-2)^d)^d = 1358954496 := by
  sorry

end problem_statement_l436_43605


namespace smallest_x_satisfying_equation_l436_43646

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 8 → x ≤ y) ∧
    ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧
    x = 89/9 := by
  sorry

end smallest_x_satisfying_equation_l436_43646


namespace quadrilateral_angle_inequality_l436_43622

variable (A B C D A₁ B₁ C₁ D₁ : Point)

-- Define the quadrilaterals
def is_convex_quadrilateral (P Q R S : Point) : Prop := sorry

-- Define the equality of corresponding sides
def equal_corresponding_sides (P Q R S P₁ Q₁ R₁ S₁ : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_angle_inequality
  (h_convex_ABCD : is_convex_quadrilateral A B C D)
  (h_convex_A₁B₁C₁D₁ : is_convex_quadrilateral A₁ B₁ C₁ D₁)
  (h_equal_sides : equal_corresponding_sides A B C D A₁ B₁ C₁ D₁)
  (h_angle_A : angle_measure B A D > angle_measure B₁ A₁ D₁) :
  angle_measure A B C < angle_measure A₁ B₁ C₁ ∧
  angle_measure B C D > angle_measure B₁ C₁ D₁ ∧
  angle_measure C D A < angle_measure C₁ D₁ A₁ :=
by sorry

end quadrilateral_angle_inequality_l436_43622


namespace student_count_l436_43637

/-- The number of students in Elementary and Middle School -/
def total_students (elementary : ℕ) (middle : ℕ) : ℕ :=
  elementary + middle

/-- Theorem stating the total number of students given the conditions -/
theorem student_count : ∃ (elementary : ℕ) (middle : ℕ),
  middle = 50 ∧ 
  elementary = 4 * middle - 3 ∧
  total_students elementary middle = 247 := by
  sorry

end student_count_l436_43637


namespace fraction_multiplication_identity_l436_43630

theorem fraction_multiplication_identity : (5 : ℚ) / 7 * 7 / 5 = 1 := by sorry

end fraction_multiplication_identity_l436_43630


namespace same_terminal_side_as_pi_sixth_l436_43664

def coterminal (θ₁ θ₂ : Real) : Prop :=
  ∃ k : Int, θ₁ = θ₂ + 2 * k * Real.pi

theorem same_terminal_side_as_pi_sixth (θ : Real) : 
  coterminal θ (π/6) ↔ ∃ k : Int, θ = π/6 + 2 * k * π :=
sorry

end same_terminal_side_as_pi_sixth_l436_43664


namespace unique_solution_l436_43643

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  x = 2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1

/-- Theorem stating that 1 is the unique solution to the equation -/
theorem unique_solution :
  ∃! x : ℝ, equation x :=
sorry

end unique_solution_l436_43643


namespace prob_select_two_after_transfer_l436_43625

/-- Represents the label on a ball -/
inductive Label
  | one
  | two
  | three

/-- Represents a bag of balls -/
structure Bag where
  ones : Nat
  twos : Nat
  threes : Nat

/-- Initial state of bag A -/
def bagA : Bag := ⟨3, 2, 1⟩

/-- Initial state of bag B -/
def bagB : Bag := ⟨2, 1, 1⟩

/-- Probability of selecting a ball with a specific label from a bag -/
def probSelect (bag : Bag) (label : Label) : Rat :=
  match label with
  | Label.one => bag.ones / (bag.ones + bag.twos + bag.threes)
  | Label.two => bag.twos / (bag.ones + bag.twos + bag.threes)
  | Label.three => bag.threes / (bag.ones + bag.twos + bag.threes)

/-- Probability of selecting a ball labeled 2 from bag B after transfer -/
def probSelectTwoAfterTransfer : Rat :=
  (probSelect bagA Label.one) * (probSelect ⟨bagB.ones + 1, bagB.twos, bagB.threes⟩ Label.two) +
  (probSelect bagA Label.two) * (probSelect ⟨bagB.ones, bagB.twos + 1, bagB.threes⟩ Label.two) +
  (probSelect bagA Label.three) * (probSelect ⟨bagB.ones, bagB.twos, bagB.threes + 1⟩ Label.two)

theorem prob_select_two_after_transfer :
  probSelectTwoAfterTransfer = 4 / 15 := by
  sorry

end prob_select_two_after_transfer_l436_43625


namespace max_kids_on_bus_l436_43679

/-- Represents the school bus configuration -/
structure SchoolBus where
  lowerDeckRows : Nat
  upperDeckRows : Nat
  lowerDeckCapacity : Nat
  upperDeckCapacity : Nat
  staffMembers : Nat
  reservedSeats : Nat

/-- Calculates the maximum number of kids that can ride the school bus -/
def maxKids (bus : SchoolBus) : Nat :=
  (bus.lowerDeckRows * bus.lowerDeckCapacity + bus.upperDeckRows * bus.upperDeckCapacity)
  - bus.staffMembers - bus.reservedSeats

/-- The theorem stating the maximum number of kids that can ride the school bus -/
theorem max_kids_on_bus :
  let bus : SchoolBus := {
    lowerDeckRows := 15,
    upperDeckRows := 10,
    lowerDeckCapacity := 5,
    upperDeckCapacity := 3,
    staffMembers := 4,
    reservedSeats := 10
  }
  maxKids bus = 91 := by
  sorry

#eval maxKids {
  lowerDeckRows := 15,
  upperDeckRows := 10,
  lowerDeckCapacity := 5,
  upperDeckCapacity := 3,
  staffMembers := 4,
  reservedSeats := 10
}

end max_kids_on_bus_l436_43679


namespace quadratic_intersection_l436_43626

def quadratic (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_intersection (a b c d : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic a b x₁ = quadratic c d x₁ ∧ 
                quadratic a b x₂ = quadratic c d x₂ ∧ 
                x₁ ≠ x₂) →
  (∀ x : ℝ, quadratic a b (-a/2) ≤ quadratic a b x) →
  (∀ x : ℝ, quadratic c d (-c/2) ≤ quadratic c d x) →
  quadratic a b (-a/2) = -200 →
  quadratic c d (-c/2) = -200 →
  (∃ x : ℝ, quadratic c d x = 0 ∧ (-a/2)^2 = x) →
  (∃ x : ℝ, quadratic a b x = 0 ∧ (-c/2)^2 = x) →
  quadratic a b 150 = -200 →
  quadratic c d 150 = -200 →
  a + c = 300 - 4 * Real.sqrt 350 ∨ a + c = 300 + 4 * Real.sqrt 350 :=
by sorry

end quadratic_intersection_l436_43626


namespace system_solution_l436_43659

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 8) (eq2 : 2*x + y = 1) : x + y = 3 := by
  sorry

end system_solution_l436_43659


namespace p_and_q_true_l436_43601

theorem p_and_q_true :
  (∃ x₀ : ℝ, x₀^2 < x₀) ∧ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end p_and_q_true_l436_43601


namespace smaller_solution_quadratic_l436_43623

theorem smaller_solution_quadratic : ∃ (x y : ℝ), 
  x < y ∧ 
  x^2 - 12*x - 28 = 0 ∧ 
  y^2 - 12*y - 28 = 0 ∧
  x = -2 ∧
  ∀ z : ℝ, z^2 - 12*z - 28 = 0 → z = x ∨ z = y := by
  sorry

end smaller_solution_quadratic_l436_43623


namespace smallest_d_value_l436_43690

theorem smallest_d_value (d : ℝ) : 
  (∃ d₀ : ℝ, d₀ > 0 ∧ Real.sqrt (40 + (4 * d₀ - 2)^2) = 10 * d₀ ∧
   ∀ d' : ℝ, d' > 0 → Real.sqrt (40 + (4 * d' - 2)^2) = 10 * d' → d₀ ≤ d') →
  d = (4 + Real.sqrt 940) / 42 :=
sorry

end smallest_d_value_l436_43690


namespace odometer_sum_of_squares_l436_43607

def is_valid_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a + b + c ≤ 10

def circular_shift (a b c : ℕ) : ℕ :=
  100 * c + 10 * a + b

def original_number (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

theorem odometer_sum_of_squares (a b c : ℕ) :
  is_valid_number a b c →
  (circular_shift a b c - original_number a b c) % 60 = 0 →
  a^2 + b^2 + c^2 = 54 := by
sorry

end odometer_sum_of_squares_l436_43607


namespace sixth_term_of_geometric_sequence_l436_43666

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q : q = 2)
  (h_a2 : a 2 = 8) :
  a 6 = 128 := by
sorry

end sixth_term_of_geometric_sequence_l436_43666


namespace typing_orders_count_l436_43620

/-- Represents the order of letters delivered by the boss -/
def letterOrder : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Represents that letter 8 has been typed -/
def letter8Typed : Nat := 8

/-- The number of letters that can be either typed or not typed after lunch -/
def remainingLetters : Nat := 8

/-- Theorem: The number of possible after-lunch typing orders is 2^8 = 256 -/
theorem typing_orders_count : 
  (2 : Nat) ^ remainingLetters = 256 := by
  sorry

#check typing_orders_count

end typing_orders_count_l436_43620


namespace properties_of_negative_2010_l436_43675

theorem properties_of_negative_2010 :
  let n : ℤ := -2010
  (1 / n = 1 / -2010) ∧
  (-n = 2010) ∧
  (abs n = 2010) ∧
  (-(1 / n) = 1 / 2010) := by
sorry

end properties_of_negative_2010_l436_43675


namespace geometry_relations_l436_43600

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (m l : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in l β) :
  (((parallel α β) → (line_perpendicular m l)) ∧
   ((line_parallel m l) → (plane_perpendicular α β))) ∧
  ¬(((plane_perpendicular α β) → (line_parallel m l)) ∧
    ((line_perpendicular m l) → (parallel α β))) :=
sorry

end geometry_relations_l436_43600


namespace largest_triangular_square_under_50_l436_43654

def isTriangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem largest_triangular_square_under_50 :
  ∃ n : ℕ, n ≤ 50 ∧ isTriangular n ∧ isPerfectSquare n ∧
  ∀ m : ℕ, m ≤ 50 → isTriangular m → isPerfectSquare m → m ≤ n :=
by sorry

end largest_triangular_square_under_50_l436_43654


namespace parallel_perpendicular_implication_l436_43614

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (m n : Line) (α : Plane) :
  parallel m n → perpendicular m α → perpendicular n α :=
sorry

end parallel_perpendicular_implication_l436_43614


namespace widget_purchase_theorem_l436_43604

/-- Given a person can buy exactly 6 widgets at price p, and 8 widgets at price (p - 1.15),
    prove that the total amount of money they have is 27.60 -/
theorem widget_purchase_theorem (p : ℝ) (h1 : 6 * p = 8 * (p - 1.15)) : 6 * p = 27.60 := by
  sorry

end widget_purchase_theorem_l436_43604


namespace promotional_price_calculation_l436_43617

/-- The cost of one chocolate at the store with the promotion -/
def promotional_price : ℚ := 2

theorem promotional_price_calculation :
  let chocolates_per_week : ℕ := 2
  let weeks : ℕ := 3
  let local_price : ℚ := 3
  let total_savings : ℚ := 6
  promotional_price = (chocolates_per_week * weeks * local_price - total_savings) / (chocolates_per_week * weeks) :=
by sorry

end promotional_price_calculation_l436_43617


namespace inequality_proof_l436_43638

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end inequality_proof_l436_43638


namespace problem_statement_l436_43602

theorem problem_statement (x y : ℝ) (hx : x = 20) (hy : y = 8) :
  (x - y) * (x + y) = 336 := by
  sorry

end problem_statement_l436_43602


namespace earth_sun_distance_scientific_notation_l436_43636

/-- The distance from Earth to Sun in kilometers -/
def earth_sun_distance : ℕ := 150000000

/-- Represents a number in scientific notation as a pair (coefficient, exponent) -/
def scientific_notation := ℝ × ℤ

/-- Converts a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : scientific_notation :=
  sorry

theorem earth_sun_distance_scientific_notation :
  to_scientific_notation earth_sun_distance = (1.5, 8) :=
sorry

end earth_sun_distance_scientific_notation_l436_43636


namespace sum_of_roots_l436_43696

theorem sum_of_roots (k c : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  (4 * x₁^2 - k * x₁ = c) → 
  (4 * x₂^2 - k * x₂ = c) → 
  x₁ + x₂ = k / 4 := by
sorry

end sum_of_roots_l436_43696


namespace derivative_exp_cos_l436_43665

open Real

theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => exp x * cos x) x = exp x * (cos x - sin x) := by
sorry

end derivative_exp_cos_l436_43665


namespace february_first_is_friday_l436_43603

/-- Represents days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in February -/
structure FebruaryDay where
  date : Nat
  weekday : Weekday

/-- Represents the condition of the student groups visiting Teacher Li -/
structure StudentVisit where
  day : FebruaryDay
  groupSize : Nat

/-- The main theorem -/
theorem february_first_is_friday 
  (visit : StudentVisit)
  (h1 : visit.day.weekday = Weekday.Sunday)
  (h2 : visit.day.date = 3 * visit.groupSize * visit.groupSize)
  (h3 : visit.groupSize > 1)
  : (⟨1, Weekday.Friday⟩ : FebruaryDay) = 
    {date := 1, weekday := Weekday.Friday} :=
by sorry

end february_first_is_friday_l436_43603


namespace investment_scientific_notation_l436_43673

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem investment_scientific_notation :
  toScientificNotation 909000000000 = ScientificNotation.mk 9.09 11 sorry := by
  sorry

end investment_scientific_notation_l436_43673


namespace cubic_sum_minus_product_l436_43680

theorem cubic_sum_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 13) 
  (h2 : x*y + x*z + y*z = 32) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 949 := by
  sorry

end cubic_sum_minus_product_l436_43680


namespace quadratic_vertex_l436_43687

/-- A quadratic function f(x) = x^2 + bx + c -/
def QuadraticFunction (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem quadratic_vertex (b c : ℝ) :
  (QuadraticFunction b c 1 = 0) →
  (∀ x, QuadraticFunction b c (2 + x) = QuadraticFunction b c (2 - x)) →
  (∃ y, QuadraticFunction b c 2 = y ∧ ∀ x, QuadraticFunction b c x ≥ y) →
  (2, -1) = (2, QuadraticFunction b c 2) :=
sorry

end quadratic_vertex_l436_43687


namespace little_john_money_little_john_initial_money_l436_43694

/-- Little John's money problem -/
theorem little_john_money : ℝ → Prop :=
  fun initial_money =>
    let spent_on_sweets : ℝ := 3.25
    let given_to_each_friend : ℝ := 2.20
    let number_of_friends : ℕ := 2
    let money_left : ℝ := 2.45
    initial_money = spent_on_sweets + (given_to_each_friend * number_of_friends) + money_left ∧
    initial_money = 10.10

/-- Proof of Little John's initial money amount -/
theorem little_john_initial_money : ∃ (m : ℝ), little_john_money m :=
  sorry

end little_john_money_little_john_initial_money_l436_43694
