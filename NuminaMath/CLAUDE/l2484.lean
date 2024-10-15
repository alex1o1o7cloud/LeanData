import Mathlib

namespace NUMINAMATH_CALUDE_bella_position_at_102_l2484_248462

/-- Represents a point on a 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents Bella's state at any given point -/
structure BellaState where
  position : Point
  facing : Direction
  lastMove : ℕ

/-- Defines the movement rules for Bella -/
def moveRules (n : ℕ) (state : BellaState) : BellaState :=
  sorry

/-- The main theorem to prove -/
theorem bella_position_at_102 :
  let initialState : BellaState := {
    position := { x := 0, y := 0 },
    facing := Direction.North,
    lastMove := 0
  }
  let finalState := (moveRules 102 initialState)
  finalState.position = { x := -23, y := 29 } :=
sorry

end NUMINAMATH_CALUDE_bella_position_at_102_l2484_248462


namespace NUMINAMATH_CALUDE_martha_coffee_spending_cut_l2484_248425

def coffee_spending_cut_percentage (latte_cost : ℚ) (iced_coffee_cost : ℚ)
  (lattes_per_week : ℕ) (iced_coffees_per_week : ℕ) (weeks_per_year : ℕ)
  (savings_goal : ℚ) : ℚ :=
  let weekly_spending := latte_cost * lattes_per_week + iced_coffee_cost * iced_coffees_per_week
  let annual_spending := weekly_spending * weeks_per_year
  (savings_goal / annual_spending) * 100

theorem martha_coffee_spending_cut (latte_cost : ℚ) (iced_coffee_cost : ℚ)
  (lattes_per_week : ℕ) (iced_coffees_per_week : ℕ) (weeks_per_year : ℕ)
  (savings_goal : ℚ) :
  latte_cost = 4 →
  iced_coffee_cost = 2 →
  lattes_per_week = 5 →
  iced_coffees_per_week = 3 →
  weeks_per_year = 52 →
  savings_goal = 338 →
  coffee_spending_cut_percentage latte_cost iced_coffee_cost lattes_per_week
    iced_coffees_per_week weeks_per_year savings_goal = 25 :=
by sorry

end NUMINAMATH_CALUDE_martha_coffee_spending_cut_l2484_248425


namespace NUMINAMATH_CALUDE_unit_circle_solutions_eq_parameterized_solutions_l2484_248483

noncomputable section

variable (F : Type*) [Field F]

/-- The set of solutions to x^2 + y^2 = 1 in a field F where 1 + 1 ≠ 0 -/
def UnitCircleSolutions (F : Type*) [Field F] (h : (1 : F) + 1 ≠ 0) : Set (F × F) :=
  {p : F × F | p.1^2 + p.2^2 = 1}

/-- The parameterized set of solutions -/
def ParameterizedSolutions (F : Type*) [Field F] : Set (F × F) :=
  {p : F × F | ∃ r : F, r^2 ≠ -1 ∧ 
    p = ((r^2 - 1) / (r^2 + 1), 2*r / (r^2 + 1))} ∪ {(1, 0)}

/-- Theorem stating that the solutions to x^2 + y^2 = 1 are exactly the parameterized solutions -/
theorem unit_circle_solutions_eq_parameterized_solutions 
  (h : (1 : F) + 1 ≠ 0) : 
  UnitCircleSolutions F h = ParameterizedSolutions F :=
by sorry

end

end NUMINAMATH_CALUDE_unit_circle_solutions_eq_parameterized_solutions_l2484_248483


namespace NUMINAMATH_CALUDE_roots_custom_op_result_l2484_248451

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a * b - a - b

-- State the theorem
theorem roots_custom_op_result :
  ∀ x₁ x₂ : ℝ,
  (x₁^2 + x₁ - 1 = 0) →
  (x₂^2 + x₂ - 1 = 0) →
  customOp x₁ x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_custom_op_result_l2484_248451


namespace NUMINAMATH_CALUDE_class_size_is_fifteen_l2484_248416

/-- Given a class of students with the following properties:
  1. The average age of all students is 15 years
  2. The average age of 6 students is 14 years
  3. The average age of 8 students is 16 years
  4. The age of the 15th student is 13 years
  Prove that the total number of students in the class is 15 -/
theorem class_size_is_fifteen (N : ℕ) 
  (h1 : (N : ℚ) * 15 = (6 : ℚ) * 14 + (8 : ℚ) * 16 + 13)
  (h2 : N ≥ 15) : N = 15 := by
  sorry


end NUMINAMATH_CALUDE_class_size_is_fifteen_l2484_248416


namespace NUMINAMATH_CALUDE_yard_sale_books_bought_l2484_248410

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem: The number of books Mike bought at the yard sale is the difference between his final and initial number of books -/
theorem yard_sale_books_bought (initial_books final_books : ℕ) 
  (h : final_books ≥ initial_books) : 
  books_bought initial_books final_books = final_books - initial_books :=
by
  sorry

/-- Given Mike's initial and final number of books, calculate how many he bought -/
def mikes_books : ℕ := 
  books_bought 35 56

#eval mikes_books

end NUMINAMATH_CALUDE_yard_sale_books_bought_l2484_248410


namespace NUMINAMATH_CALUDE_diagonal_cubes_150_324_375_l2484_248456

/-- The number of unit cubes that a diagonal passes through in a rectangular prism -/
def diagonal_cubes (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd a c - Nat.gcd b c + Nat.gcd (Nat.gcd a b) c

/-- Theorem: In a 150 × 324 × 375 rectangular prism, the diagonal passes through 768 unit cubes -/
theorem diagonal_cubes_150_324_375 :
  diagonal_cubes 150 324 375 = 768 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_cubes_150_324_375_l2484_248456


namespace NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l2484_248431

theorem tangent_and_trigonometric_identity (α β : Real) 
  (h1 : Real.tan (α + β) = 2)
  (h2 : Real.tan (Real.pi - β) = 3/2) :
  (Real.tan α = -7/4) ∧ 
  ((Real.sin (Real.pi/2 + α) - Real.sin (Real.pi + α)) / (Real.cos α + 2 * Real.sin α) = 3/10) := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trigonometric_identity_l2484_248431


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_8_is_5_18_l2484_248474

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to get a sum of 8 or less when rolling two dice -/
def sum_8_or_less : ℕ := 26

/-- The probability of rolling two dice and getting a sum greater than eight -/
def prob_sum_greater_than_8 : ℚ :=
  1 - (sum_8_or_less : ℚ) / total_outcomes

theorem prob_sum_greater_than_8_is_5_18 :
  prob_sum_greater_than_8 = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_8_is_5_18_l2484_248474


namespace NUMINAMATH_CALUDE_total_eggs_l2484_248484

def eggs_club_house : ℕ := 12
def eggs_park : ℕ := 5
def eggs_town_hall : ℕ := 3

theorem total_eggs : eggs_club_house + eggs_park + eggs_town_hall = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_eggs_l2484_248484


namespace NUMINAMATH_CALUDE_shortest_distance_point_l2484_248492

/-- Given points A and B, find the point P on the y-axis that minimizes AP + BP -/
theorem shortest_distance_point (A B P : ℝ × ℝ) : 
  A = (3, 2) →
  B = (1, -2) →
  P.1 = 0 →
  P = (0, -1) →
  ∀ Q : ℝ × ℝ, Q.1 = 0 → 
    Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) + Real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2) ≤ 
    Real.sqrt ((A.1 - Q.1)^2 + (A.2 - Q.2)^2) + Real.sqrt ((B.1 - Q.1)^2 + (B.2 - Q.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_shortest_distance_point_l2484_248492


namespace NUMINAMATH_CALUDE_second_metal_cost_l2484_248467

/-- Given two metals mixed in equal proportions, prove the cost of the second metal
    when the cost of the first metal and the resulting alloy are known. -/
theorem second_metal_cost (cost_first : ℝ) (cost_alloy : ℝ) : 
  cost_first = 68 → cost_alloy = 82 → 2 * cost_alloy - cost_first = 96 := by
  sorry

end NUMINAMATH_CALUDE_second_metal_cost_l2484_248467


namespace NUMINAMATH_CALUDE_simplify_expression_l2484_248433

theorem simplify_expression (z : ℝ) : (7 - Real.sqrt (z^2 - 49))^2 = z^2 - 14 * Real.sqrt (z^2 - 49) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2484_248433


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2484_248418

/-- Given a positive arithmetic sequence {a_n} satisfying certain conditions, prove a_10 = 21 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  (∀ n, a n > 0) →  -- positive sequence
  a 1 + a 2 + a 3 = 15 →  -- sum condition
  (a 2 + 5)^2 = (a 1 + 2) * (a 3 + 13) →  -- geometric sequence condition
  a 10 = 21 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2484_248418


namespace NUMINAMATH_CALUDE_exactly_two_correct_l2484_248480

-- Define a mapping
def Mapping (A B : Type) := A → B

-- Define a function
def Function (α : Type) := α → ℝ

-- Define an odd function
def OddFunction (f : Function ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define propositions
def Proposition1 (A B : Type) : Prop :=
  ∃ (f : Mapping A B), ∃ b : B, ∀ a : A, f a ≠ b

def Proposition2 : Prop :=
  ∀ (f : Function ℝ) (t : ℝ), ∃! x : ℝ, f x = t

def Proposition3 (f : Function ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) → OddFunction f

def Proposition4 (f : Function ℝ) : Prop :=
  (∀ x, 0 ≤ f (2*x - 1) ∧ f (2*x - 1) ≤ 1) →
  (∀ x, -1 ≤ f x ∧ f x ≤ 1)

-- Theorem statement
theorem exactly_two_correct :
  (Proposition1 ℝ ℝ) ∧
  (∃ f : Function ℝ, Proposition3 f) ∧
  ¬(Proposition2) ∧
  ¬(∃ f : Function ℝ, Proposition4 f) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_correct_l2484_248480


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2484_248463

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_single : a < 10
  b_single : b < 10
  c_single : c < 10
  d_single : d < 10

/-- Checks if the given four-digit number satisfies the product conditions -/
def satisfies_conditions (n : FourDigitNumber) : Prop :=
  (n.a * n.b = 21 ∧ n.b * n.c = 20) ∨
  (n.a * n.b = 21 ∧ n.c * n.d = 20) ∨
  (n.b * n.c = 21 ∧ n.c * n.d = 20)

/-- The smallest four-digit number satisfying the conditions -/
def smallest_satisfying_number : FourDigitNumber :=
  { a := 3, b := 7, c := 4, d := 5,
    a_single := by norm_num,
    b_single := by norm_num,
    c_single := by norm_num,
    d_single := by norm_num }

theorem smallest_number_proof :
  satisfies_conditions smallest_satisfying_number ∧
  ∀ n : FourDigitNumber, satisfies_conditions n →
    n.a * 1000 + n.b * 100 + n.c * 10 + n.d ≥
    smallest_satisfying_number.a * 1000 +
    smallest_satisfying_number.b * 100 +
    smallest_satisfying_number.c * 10 +
    smallest_satisfying_number.d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2484_248463


namespace NUMINAMATH_CALUDE_no_solution_exists_l2484_248461

theorem no_solution_exists : ¬ ∃ (a b : ℤ), (2006 * 2006) ∣ (a^2006 + b^2006 + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2484_248461


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l2484_248423

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -39 - 52*I ∧ z = 5 - 7*I → (-z = -5 + 7*I ∧ (-z)^2 = -39 - 52*I) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l2484_248423


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_property_l2484_248446

/-- Given an ellipse and a hyperbola with shared foci, prove a property of the ellipse's semi-minor axis --/
theorem ellipse_hyperbola_property (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
   ∃ x' y' : ℝ, x'^2 - y'^2/4 = 1 ∧ 
   ∃ A B : ℝ × ℝ, 
     (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*a)^2 ∧
     (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
       (t*A.1 + (1-t)*B.1)^2/a^2 + (t*A.2 + (1-t)*B.2)^2/b^2 = 1 ∧
       ((1-t)*A.1 + t*B.1)^2/a^2 + ((1-t)*A.2 + t*B.2)^2/b^2 = 1 ∧
       t = 1/3)) →
  b^2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_property_l2484_248446


namespace NUMINAMATH_CALUDE_prime_pair_equation_solution_l2484_248413

theorem prime_pair_equation_solution :
  ∀ p q : ℕ,
  Prime p → Prime q →
  (∃ m : ℕ+, (p * q : ℚ) / (p + q) = (m.val^2 + 6 : ℚ) / (m.val + 1)) →
  (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := by
sorry

end NUMINAMATH_CALUDE_prime_pair_equation_solution_l2484_248413


namespace NUMINAMATH_CALUDE_ln_square_plus_ln_inequality_l2484_248494

theorem ln_square_plus_ln_inequality (x : ℝ) :
  (Real.log x)^2 + Real.log x < 0 ↔ Real.exp (-1) < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_square_plus_ln_inequality_l2484_248494


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2484_248402

theorem complex_equation_solution (m A B : ℝ) : 
  (2 - m * Complex.I) / (1 + 2 * Complex.I) = Complex.mk A B →
  A + B = 0 →
  m = -2/3 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2484_248402


namespace NUMINAMATH_CALUDE_sum_squared_distances_coinciding_centroids_l2484_248469

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- An isosceles right triangle -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  leg_length_pos : leg_length > 0

/-- The sum of squared distances between vertices of two triangles -/
def sum_squared_distances (et : EquilateralTriangle) (irt : IsoscelesRightTriangle) : ℝ := 
  3 * et.side_length^2 + 4 * irt.leg_length^2

theorem sum_squared_distances_coinciding_centroids 
  (et : EquilateralTriangle) 
  (irt : IsoscelesRightTriangle) :
  sum_squared_distances et irt = 3 * et.side_length^2 + 4 * irt.leg_length^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_distances_coinciding_centroids_l2484_248469


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l2484_248465

theorem dress_discount_percentage (d : ℝ) (x : ℝ) (h : d > 0) :
  (d * (100 - x) / 100) * 0.7 = 0.455 * d ↔ x = 35 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_percentage_l2484_248465


namespace NUMINAMATH_CALUDE_lcd_of_fractions_l2484_248403

theorem lcd_of_fractions (a b c d e f : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) (hd : d = 6) (he : e = 8) (hf : f = 9) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e f)))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcd_of_fractions_l2484_248403


namespace NUMINAMATH_CALUDE_basketball_free_throws_l2484_248499

theorem basketball_free_throws (total_score : ℕ) (three_point_shots : ℕ) 
  (h1 : total_score = 79)
  (h2 : 3 * three_point_shots = 2 * (total_score - 3 * three_point_shots - free_throws) / 2)
  (h3 : free_throws = 2 * (total_score - 3 * three_point_shots - free_throws) / 2)
  (h4 : three_point_shots = 4) :
  free_throws = 12 :=
by
  sorry

#check basketball_free_throws

end NUMINAMATH_CALUDE_basketball_free_throws_l2484_248499


namespace NUMINAMATH_CALUDE_root_zero_iff_m_neg_three_l2484_248409

/-- The quadratic equation in x with parameter m -/
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 + (3*m - 1) * x + m^2 - 9

/-- Theorem: One root of the quadratic equation is 0 iff m = -3 -/
theorem root_zero_iff_m_neg_three :
  ∀ m : ℝ, (∃ x : ℝ, quadratic_equation m x = 0 ∧ x = 0) ↔ m = -3 := by sorry

end NUMINAMATH_CALUDE_root_zero_iff_m_neg_three_l2484_248409


namespace NUMINAMATH_CALUDE_linear_function_property_l2484_248404

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g where g(5) - g(1) = 16, prove that g(13) - g(1) = 48. -/
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) 
  (h_given : g 5 - g 1 = 16) : 
  g 13 - g 1 = 48 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_property_l2484_248404


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2484_248482

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, (2 * x₁^2 - 3 * x₁ - 2 = 0 ∧ x₁ = -1/2) ∧
                (2 * x₂^2 - 3 * x₂ - 2 = 0 ∧ x₂ = 2)) ∧
  (∃ y₁ y₂ : ℝ, (2 * y₁^2 - 3 * y₁ - 1 = 0 ∧ y₁ = (3 + Real.sqrt 17) / 4) ∧
                (2 * y₂^2 - 3 * y₂ - 1 = 0 ∧ y₂ = (3 - Real.sqrt 17) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2484_248482


namespace NUMINAMATH_CALUDE_lindsey_squat_weight_l2484_248430

/-- The total weight Lindsey will squat given exercise bands and a dumbbell -/
theorem lindsey_squat_weight (num_bands : ℕ) (band_resistance : ℕ) (dumbbell_weight : ℕ) : 
  num_bands = 2 →
  band_resistance = 5 →
  dumbbell_weight = 10 →
  num_bands * band_resistance + dumbbell_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_lindsey_squat_weight_l2484_248430


namespace NUMINAMATH_CALUDE_exists_permutation_divisible_by_seven_l2484_248421

def digits : List Nat := [1, 3, 7, 9]

def is_permutation (l1 l2 : List Nat) : Prop :=
  l1.length = l2.length ∧ ∀ x, l1.count x = l2.count x

def list_to_number (l : List Nat) : Nat :=
  l.foldl (fun acc d => acc * 10 + d) 0

theorem exists_permutation_divisible_by_seven :
  ∃ perm : List Nat, is_permutation digits perm ∧ 
    (list_to_number perm) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_divisible_by_seven_l2484_248421


namespace NUMINAMATH_CALUDE_expression_evaluation_l2484_248443

theorem expression_evaluation :
  Real.sqrt ((16^6 + 2^18) / (16^3 + 2^21)) = (8 * Real.sqrt 65) / Real.sqrt 513 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2484_248443


namespace NUMINAMATH_CALUDE_complex_root_coefficients_l2484_248426

theorem complex_root_coefficients :
  ∀ (b c : ℝ),
  (Complex.I * Real.sqrt 2 + 1) ^ 2 + b * (Complex.I * Real.sqrt 2 + 1) + c = 0 →
  b = -2 ∧ c = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_coefficients_l2484_248426


namespace NUMINAMATH_CALUDE_necklaces_sold_l2484_248422

theorem necklaces_sold (total : ℕ) (given_away : ℕ) (left : ℕ) (sold : ℕ) : 
  total = 60 → given_away = 18 → left = 26 → sold = total - given_away - left → sold = 16 := by
  sorry

end NUMINAMATH_CALUDE_necklaces_sold_l2484_248422


namespace NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l2484_248496

theorem sqrt_198_between_14_and_15 : 14 < Real.sqrt 198 ∧ Real.sqrt 198 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_198_between_14_and_15_l2484_248496


namespace NUMINAMATH_CALUDE_f_min_value_l2484_248435

/-- The polynomial f(x) defined for a positive integer n and real x -/
def f (n : ℕ+) (x : ℝ) : ℝ :=
  (Finset.range (2*n+1)).sum (fun k => (2*n+1-k) * x^k)

/-- Theorem stating that the minimum value of f(x) is n+1 and occurs at x = -1 -/
theorem f_min_value (n : ℕ+) :
  (∀ x : ℝ, f n x ≥ f n (-1)) ∧ f n (-1) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_value_l2484_248435


namespace NUMINAMATH_CALUDE_ages_solution_l2484_248438

/-- Represents the ages of Ann, Kristine, and Brad -/
structure Ages where
  ann : ℕ
  kristine : ℕ
  brad : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.ann = ages.kristine + 5 ∧
  ages.brad = ages.ann - 3 ∧
  ages.brad = 2 * ages.kristine

/-- The theorem to be proved -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧
    ages.kristine = 2 ∧ ages.ann = 7 ∧ ages.brad = 4 := by
  sorry

end NUMINAMATH_CALUDE_ages_solution_l2484_248438


namespace NUMINAMATH_CALUDE_set_B_equals_expected_l2484_248472

def A : Set Int := {-3, -2, -1, 1, 2, 3, 4}

def f (a : Int) : Int := Int.natAbs a

def B : Set Int := f '' A

theorem set_B_equals_expected : B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_B_equals_expected_l2484_248472


namespace NUMINAMATH_CALUDE_circplus_two_three_one_l2484_248485

/-- Definition of the ⊕ operation -/
def circplus (a b c : ℝ) : ℝ := b^2 - 4*a*c + c^2

/-- Theorem: The value of ⊕(2, 3, 1) is 2 -/
theorem circplus_two_three_one : circplus 2 3 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_circplus_two_three_one_l2484_248485


namespace NUMINAMATH_CALUDE_count_eight_to_800_l2484_248488

/-- Count of digit 8 in a single number -/
def count_eight (n : ℕ) : ℕ := sorry

/-- Sum of count_eight for all numbers from 1 to n -/
def sum_count_eight (n : ℕ) : ℕ := sorry

/-- The count of the digit 8 in all integers from 1 to 800 is 161 -/
theorem count_eight_to_800 : sum_count_eight 800 = 161 := by sorry

end NUMINAMATH_CALUDE_count_eight_to_800_l2484_248488


namespace NUMINAMATH_CALUDE_unique_division_problem_l2484_248441

theorem unique_division_problem :
  ∃! (dividend divisor : ℕ),
    dividend ≥ 1000000 ∧ dividend < 2000000 ∧
    divisor ≥ 300 ∧ divisor < 400 ∧
    (dividend / divisor : ℚ) = 5243 / 1000 ∧
    dividend % divisor = 0 ∧
    ∃ (r1 r2 r3 : ℕ),
      r1 % 10 = 9 ∧
      r2 % 10 = 6 ∧
      r3 % 10 = 3 ∧
      r1 < divisor ∧
      r2 < divisor ∧
      r3 < divisor ∧
      dividend = 1000000 + (dividend / 100000 % 10) * 100000 + 50000 + (dividend % 10000) :=
by sorry

end NUMINAMATH_CALUDE_unique_division_problem_l2484_248441


namespace NUMINAMATH_CALUDE_inequality_proof_l2484_248406

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y ≤ (y^2 / x) + (x^2 / y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2484_248406


namespace NUMINAMATH_CALUDE_battery_life_comparison_l2484_248458

/-- Represents the charge of a battery -/
structure BatteryCharge where
  charge : ℝ
  positive : charge > 0

/-- Represents a clock powered by batteries -/
structure Clock where
  batteries : ℕ
  batteryType : BatteryCharge

/-- The problem statement -/
theorem battery_life_comparison 
  (battery_a battery_b : BatteryCharge)
  (clock_1 clock_2 : Clock)
  (h1 : battery_a.charge = 6 * battery_b.charge)
  (h2 : clock_1.batteries = 4 ∧ clock_1.batteryType = battery_a)
  (h3 : clock_2.batteries = 3 ∧ clock_2.batteryType = battery_b)
  (h4 : (clock_2.batteries : ℝ) * clock_2.batteryType.charge = 2)
  : (clock_1.batteries : ℝ) * clock_1.batteryType.charge - 
    (clock_2.batteries : ℝ) * clock_2.batteryType.charge = 14 := by
  sorry

#check battery_life_comparison

end NUMINAMATH_CALUDE_battery_life_comparison_l2484_248458


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_div_fifth_l2484_248457

/-- Represents a repeating decimal with a two-digit repeat -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (a * 100 + b) / 99

theorem repeating_decimal_sum_div_fifth :
  let x := RepeatingDecimal 8 3
  let y := RepeatingDecimal 1 8
  (x + y) / (1/5) = 505/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_div_fifth_l2484_248457


namespace NUMINAMATH_CALUDE_sequence_a_general_term_sequence_b_general_term_l2484_248437

-- Define the sequences
def sequence_a : ℕ → ℕ
  | 1 => 0
  | 2 => 3
  | 3 => 26
  | 4 => 255
  | 5 => 3124
  | _ => 0  -- Default case, not used in the proof

def sequence_b : ℕ → ℕ
  | 1 => 1
  | 2 => 2
  | 3 => 12
  | 4 => 288
  | 5 => 34560
  | _ => 0  -- Default case, not used in the proof

-- Define the general term for sequence a
def general_term_a (n : ℕ) : ℕ := n^n - 1

-- Define the general term for sequence b
def general_term_b (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc * Nat.factorial (i + 1)) 1

-- Theorem for sequence a
theorem sequence_a_general_term (n : ℕ) (h : n > 0 ∧ n ≤ 5) :
  sequence_a n = general_term_a n := by
  sorry

-- Theorem for sequence b
theorem sequence_b_general_term (n : ℕ) (h : n > 0 ∧ n ≤ 5) :
  sequence_b n = general_term_b n := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_general_term_sequence_b_general_term_l2484_248437


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2484_248417

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) :
  3 * a 9 - a 11 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2484_248417


namespace NUMINAMATH_CALUDE_comparison_of_powers_l2484_248452

theorem comparison_of_powers (a b c : ℝ) : 
  a = 10 ∧ b = -49 ∧ c = -50 → 
  a^b - 2 * a^c = 8 * a^c := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l2484_248452


namespace NUMINAMATH_CALUDE_negation_and_converse_l2484_248442

def last_digit (n : ℤ) : ℕ := (n % 10).natAbs

def divisible_by_five (n : ℤ) : Prop := n % 5 = 0

def statement (n : ℤ) : Prop :=
  (last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n

theorem negation_and_converse :
  (∀ n : ℤ, ¬statement n ↔ (last_digit n = 0 ∨ last_digit n = 5) ∧ ¬(divisible_by_five n)) ∧
  (∀ n : ℤ, (¬(last_digit n = 0 ∨ last_digit n = 5) → ¬(divisible_by_five n)) →
    ((last_digit n = 0 ∨ last_digit n = 5) → divisible_by_five n)) :=
sorry

end NUMINAMATH_CALUDE_negation_and_converse_l2484_248442


namespace NUMINAMATH_CALUDE_equation_solution_l2484_248434

theorem equation_solution (x y : ℝ) 
  (h1 : x + 2 ≠ 0) 
  (h2 : x - y + 1 ≠ 0) 
  (h3 : (x - y) / (x + 2) = y / (x - y + 1)) : 
  x = (y - 1 + Real.sqrt (-3 * y^2 + 10 * y + 1)) / 2 ∨ 
  x = (y - 1 - Real.sqrt (-3 * y^2 + 10 * y + 1)) / 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2484_248434


namespace NUMINAMATH_CALUDE_zoo_animals_l2484_248490

/-- The number of sea lions at the zoo -/
def sea_lions : ℕ := 42

/-- The number of penguins at the zoo -/
def penguins : ℕ := sea_lions + 84

/-- The number of flamingos at the zoo -/
def flamingos : ℕ := penguins + 42

theorem zoo_animals :
  (4 : ℚ) * sea_lions = 11 * sea_lions - 7 * 84 ∧
  7 * penguins = 11 * sea_lions + 7 * 42 ∧
  4 * flamingos = 7 * penguins + 4 * 42 :=
by sorry

#check zoo_animals

end NUMINAMATH_CALUDE_zoo_animals_l2484_248490


namespace NUMINAMATH_CALUDE_angle_inequality_equivalence_l2484_248432

theorem angle_inequality_equivalence (θ : Real) (h1 : 0 ≤ θ) (h2 : θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 2 → x^2 * Real.cos θ - x * (2 - x) + (2 - x)^2 * Real.sin θ > 0) ↔
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_equivalence_l2484_248432


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2484_248447

theorem arithmetic_geometric_sequence_ratio (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ -- positive
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ -- distinct
  2 * a = b + c ∧ -- arithmetic sequence
  a * a = b * c -- geometric sequence
  → ∃ (k : ℝ), k ≠ 0 ∧ a = 2 * k ∧ b = 4 * k ∧ c = k := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l2484_248447


namespace NUMINAMATH_CALUDE_book_length_problem_l2484_248498

/-- Represents the problem of determining book lengths based on reading rates and times -/
theorem book_length_problem (book1_pages book2_pages_read : ℕ) 
  (book1_rate book2_rate : ℕ) (h1 : book1_rate = 40) (h2 : book2_rate = 60) :
  (2 * book1_pages / 3 = book1_pages / 3 + 30) →
  (2 * book1_pages / (3 * book1_rate) = book2_pages_read / book2_rate) →
  book1_pages = 90 ∧ book2_pages_read = 45 := by
  sorry

#check book_length_problem

end NUMINAMATH_CALUDE_book_length_problem_l2484_248498


namespace NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l2484_248424

/-- Triangle PQR with sides PQ = 47, QR = 14, and RP = 50 -/
structure Triangle (P Q R : ℝ × ℝ) :=
  (pq : dist P Q = 47)
  (qr : dist Q R = 14)
  (rp : dist R P = 50)

/-- The circumcircle of triangle PQR -/
def circumcircle (P Q R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {S | dist S P = dist S Q ∧ dist S Q = dist S R}

/-- The perpendicular bisector of RP -/
def perpBisector (R P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {S | dist S R = dist S P ∧ (S.1 - R.1) * (P.1 - R.1) + (S.2 - R.2) * (P.2 - R.2) = 0}

/-- S is on the opposite side of RP from Q -/
def oppositeSide (S Q R P : ℝ × ℝ) : Prop :=
  ((S.1 - R.1) * (P.2 - R.2) - (S.2 - R.2) * (P.1 - R.1)) *
  ((Q.1 - R.1) * (P.2 - R.2) - (Q.2 - R.2) * (P.1 - R.1)) < 0

theorem triangle_circumcircle_intersection
  (P Q R : ℝ × ℝ)
  (tri : Triangle P Q R)
  (S : ℝ × ℝ)
  (h1 : S ∈ circumcircle P Q R)
  (h2 : S ∈ perpBisector R P)
  (h3 : oppositeSide S Q R P) :
  dist P S = 8 * Real.sqrt 47 :=
sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l2484_248424


namespace NUMINAMATH_CALUDE_tan_alpha_problem_l2484_248450

theorem tan_alpha_problem (α : Real) 
  (h1 : Real.tan (α + π/4) = -1/2) 
  (h2 : π/2 < α) 
  (h3 : α < π) : 
  (Real.sin (2*α) - 2*(Real.cos α)^2) / Real.sin (α - π/4) = -2*Real.sqrt 5/5 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_problem_l2484_248450


namespace NUMINAMATH_CALUDE_candy_bar_cost_is_one_l2484_248489

/-- The cost of a candy bar given initial and remaining amounts -/
def candy_bar_cost (initial_amount : ℝ) (remaining_amount : ℝ) : ℝ :=
  initial_amount - remaining_amount

/-- Theorem: The candy bar costs $1 given the conditions -/
theorem candy_bar_cost_is_one :
  let initial_amount : ℝ := 4
  let remaining_amount : ℝ := 3
  candy_bar_cost initial_amount remaining_amount = 1 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_cost_is_one_l2484_248489


namespace NUMINAMATH_CALUDE_function_behavior_l2484_248400

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : is_symmetric_about_one f)
  (h3 : is_decreasing_on f 1 2) :
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l2484_248400


namespace NUMINAMATH_CALUDE_parabola_fv_unique_value_l2484_248477

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ
  F : ℝ × ℝ

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_fv_unique_value (p : Parabola) 
  (A B : PointOnParabola p)
  (h1 : distance A.point p.F = 25)
  (h2 : distance A.point p.V = 24)
  (h3 : distance B.point p.F = 9) :
  distance p.F p.V = 9 := sorry

end NUMINAMATH_CALUDE_parabola_fv_unique_value_l2484_248477


namespace NUMINAMATH_CALUDE_expression_factorization_l2484_248428

theorem expression_factorization (x : ℝ) :
  (20 * x^3 - 100 * x^2 + 30) - (5 * x^3 - 10 * x^2 + 3) = 3 * (5 * x^2 * (x - 6) + 9) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2484_248428


namespace NUMINAMATH_CALUDE_lentil_dishes_count_l2484_248453

/-- Represents the menu of a vegan restaurant -/
structure VeganMenu :=
  (total_dishes : ℕ)
  (beans_and_lentils : ℕ)
  (beans_and_seitan : ℕ)
  (only_beans : ℕ)
  (only_seitan : ℕ)
  (only_lentils : ℕ)

/-- The number of dishes including lentils in a vegan menu -/
def dishes_with_lentils (menu : VeganMenu) : ℕ :=
  menu.beans_and_lentils + menu.only_lentils

/-- Theorem stating the number of dishes including lentils in the given vegan menu -/
theorem lentil_dishes_count (menu : VeganMenu) 
  (h1 : menu.total_dishes = 10)
  (h2 : menu.beans_and_lentils = 2)
  (h3 : menu.beans_and_seitan = 2)
  (h4 : menu.only_beans = (menu.total_dishes - menu.beans_and_lentils - menu.beans_and_seitan) / 2)
  (h5 : menu.only_beans = 3 * menu.only_seitan)
  (h6 : menu.only_lentils = menu.total_dishes - menu.beans_and_lentils - menu.beans_and_seitan - menu.only_beans - menu.only_seitan) :
  dishes_with_lentils menu = 4 := by
  sorry


end NUMINAMATH_CALUDE_lentil_dishes_count_l2484_248453


namespace NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2484_248497

theorem percentage_of_red_non_honda_cars
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (honda_red_percentage : ℚ)
  (total_red_percentage : ℚ)
  (h1 : total_cars = 900)
  (h2 : honda_cars = 500)
  (h3 : honda_red_percentage = 90 / 100)
  (h4 : total_red_percentage = 60 / 100)
  : (((total_red_percentage * total_cars) - (honda_red_percentage * honda_cars)) /
     (total_cars - honda_cars) : ℚ) = 225 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_red_non_honda_cars_l2484_248497


namespace NUMINAMATH_CALUDE_twentieth_meeting_at_D_l2484_248478

/-- Represents a meeting point in the pool lane -/
inductive MeetingPoint
| C
| D

/-- Represents an athlete swimming in the pool lane -/
structure Athlete where
  speed : ℝ
  speed_positive : speed > 0

/-- Represents a swimming scenario with two athletes -/
structure SwimmingScenario where
  athlete1 : Athlete
  athlete2 : Athlete
  different_speeds : athlete1.speed ≠ athlete2.speed
  first_meeting : MeetingPoint
  second_meeting : MeetingPoint
  first_meeting_is_C : first_meeting = MeetingPoint.C
  second_meeting_is_D : second_meeting = MeetingPoint.D

/-- The theorem stating that the 20th meeting occurs at point D -/
theorem twentieth_meeting_at_D (scenario : SwimmingScenario) :
  (fun n => if n % 2 = 0 then MeetingPoint.D else MeetingPoint.C) 20 = MeetingPoint.D :=
sorry

end NUMINAMATH_CALUDE_twentieth_meeting_at_D_l2484_248478


namespace NUMINAMATH_CALUDE_equation_roots_l2484_248405

theorem equation_roots : 
  let f (x : ℝ) := 20 / (x^2 - 9) - 3 / (x + 3) - 2
  let root1 := (-3 + Real.sqrt 385) / 4
  let root2 := (-3 - Real.sqrt 385) / 4
  f root1 = 0 ∧ f root2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_l2484_248405


namespace NUMINAMATH_CALUDE_max_value_on_curve_l2484_248415

theorem max_value_on_curve (b : ℝ) (h : b > 0) :
  let f : ℝ × ℝ → ℝ := λ (x, y) ↦ x^2 + 2*y
  let S : Set (ℝ × ℝ) := {(x, y) | x^2/4 + y^2/b^2 = 1}
  (∃ (M : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) ∧
  (0 < b ∧ b ≤ 4 → ∀ (M : ℝ), (∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) → b^2/4 + 4 ≤ M) ∧
  (b > 4 → ∀ (M : ℝ), (∀ (p : ℝ × ℝ), p ∈ S → f p ≤ M) → 2*b ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l2484_248415


namespace NUMINAMATH_CALUDE_range_theorem_l2484_248449

/-- A monotonically decreasing odd function on ℝ with f(1) = -1 -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonically decreasing -/
axiom f_monotone : ∀ x y, x ≤ y → f y ≤ f x

/-- f is an odd function -/
axiom f_odd : ∀ x, f (-x) = -f x

/-- f(1) = -1 -/
axiom f_one : f 1 = -1

/-- The range of x satisfying -1 ≤ f(x-2) ≤ 1 is [1, 3] -/
theorem range_theorem : Set.Icc 1 3 = {x | -1 ≤ f (x - 2) ∧ f (x - 2) ≤ 1} :=
  sorry

end NUMINAMATH_CALUDE_range_theorem_l2484_248449


namespace NUMINAMATH_CALUDE_total_weight_is_540_l2484_248429

def back_squat_initial : ℝ := 200
def back_squat_increase : ℝ := 50
def front_squat_ratio : ℝ := 0.8
def triple_ratio : ℝ := 0.9
def number_of_triples : ℕ := 3

def calculate_total_weight : ℝ :=
  let back_squat_new := back_squat_initial + back_squat_increase
  let front_squat := back_squat_new * front_squat_ratio
  let triple_weight := front_squat * triple_ratio
  triple_weight * number_of_triples

theorem total_weight_is_540 :
  calculate_total_weight = 540 := by sorry

end NUMINAMATH_CALUDE_total_weight_is_540_l2484_248429


namespace NUMINAMATH_CALUDE_nes_sale_price_l2484_248414

theorem nes_sale_price 
  (snes_value : ℝ)
  (trade_in_percentage : ℝ)
  (additional_cash : ℝ)
  (change : ℝ)
  (game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : additional_cash = 80)
  (h4 : change = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + additional_cash - change - game_value = 160 :=
by
  sorry

#check nes_sale_price

end NUMINAMATH_CALUDE_nes_sale_price_l2484_248414


namespace NUMINAMATH_CALUDE_nanometer_scientific_notation_l2484_248440

/-- Expresses a given decimal number in scientific notation -/
def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem nanometer_scientific_notation :
  scientific_notation 0.000000022 = (2.2, -8) :=
sorry

end NUMINAMATH_CALUDE_nanometer_scientific_notation_l2484_248440


namespace NUMINAMATH_CALUDE_max_product_953_l2484_248470

/-- A type representing a valid digit for our problem -/
inductive Digit
  | three
  | five
  | six
  | eight
  | nine

/-- A function to convert our Digit type to a natural number -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8
  | Digit.nine => 9

/-- A type representing a valid combination of digits -/
structure DigitCombination where
  d1 : Digit
  d2 : Digit
  d3 : Digit
  d4 : Digit
  d5 : Digit
  all_different : d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧
                  d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧
                  d3 ≠ d4 ∧ d3 ≠ d5 ∧
                  d4 ≠ d5

/-- Function to calculate the product of a three-digit and two-digit number from a DigitCombination -/
def calculate_product (dc : DigitCombination) : ℕ :=
  (100 * digit_to_nat dc.d1 + 10 * digit_to_nat dc.d2 + digit_to_nat dc.d3) *
  (10 * digit_to_nat dc.d4 + digit_to_nat dc.d5)

/-- The main theorem stating that 953 yields the maximum product -/
theorem max_product_953 :
  ∀ dc : DigitCombination,
  calculate_product dc ≤ calculate_product
    { d1 := Digit.nine, d2 := Digit.five, d3 := Digit.three,
      d4 := Digit.eight, d5 := Digit.six,
      all_different := by simp } :=
sorry

end NUMINAMATH_CALUDE_max_product_953_l2484_248470


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2484_248420

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the point P
def point_P : ℝ × ℝ := (3, 0)

-- Define a line passing through P
def line_through_P (m : ℝ) (x y : ℝ) : Prop :=
  y = m * (x - point_P.1) + point_P.2

-- Theorem statement
theorem line_intersects_circle :
  ∀ m : ℝ, ∃ x y : ℝ, circle_C x y ∧ line_through_P m x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2484_248420


namespace NUMINAMATH_CALUDE_eight_even_painted_cubes_l2484_248464

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Represents a cube with a certain number of painted faces -/
structure Cube where
  painted_faces : Nat

/-- Function to determine if a number is even -/
def is_even (n : Nat) : Bool :=
  n % 2 = 0

/-- Function to calculate the number of cubes with even painted faces -/
def count_even_painted_cubes (block : Block) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that a 6x3x1 block has 8 cubes with even painted faces -/
theorem eight_even_painted_cubes (block : Block) 
  (h1 : block.length = 6) 
  (h2 : block.width = 3) 
  (h3 : block.height = 1) : 
  count_even_painted_cubes block = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_even_painted_cubes_l2484_248464


namespace NUMINAMATH_CALUDE_national_bank_interest_rate_national_bank_interest_rate_is_five_percent_l2484_248468

theorem national_bank_interest_rate 
  (initial_investment : ℝ) 
  (additional_investment : ℝ) 
  (additional_rate : ℝ) 
  (total_income_rate : ℝ) : ℝ :=
  let total_investment := initial_investment + additional_investment
  let total_income := total_investment * total_income_rate
  let additional_income := additional_investment * additional_rate
  let national_bank_income := total_income - additional_income
  national_bank_income / initial_investment

#check national_bank_interest_rate 2400 600 0.1 0.06 -- Expected output: 0.05

theorem national_bank_interest_rate_is_five_percent :
  national_bank_interest_rate 2400 600 0.1 0.06 = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_national_bank_interest_rate_national_bank_interest_rate_is_five_percent_l2484_248468


namespace NUMINAMATH_CALUDE_can_measure_15_minutes_l2484_248475

/-- Represents an hourglass with a given duration in minutes -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of the timing system -/
structure TimingSystem where
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Defines the initial state of the timing system -/
def initialState : TimingSystem :=
  { hourglass1 := { duration := 7 },
    hourglass2 := { duration := 11 } }

/-- Represents a sequence of operations on the hourglasses -/
inductive Operation
  | FlipHourglass1
  | FlipHourglass2
  | Wait (minutes : ℕ)

/-- Applies a sequence of operations to the timing system -/
def applyOperations (state : TimingSystem) (ops : List Operation) : ℕ :=
  sorry

/-- Theorem stating that 15 minutes can be measured using the given hourglasses -/
theorem can_measure_15_minutes :
  ∃ (ops : List Operation), applyOperations initialState ops = 15 :=
sorry

end NUMINAMATH_CALUDE_can_measure_15_minutes_l2484_248475


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l2484_248407

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  total : ℝ
  coins_and_beads : ℝ
  beads : ℝ
  gold_coins : ℝ

/-- The percentage of gold coins in the urn is 36% -/
theorem gold_coins_percentage (urn : UrnComposition) : 
  urn.coins_and_beads / urn.total = 0.75 →
  urn.beads / urn.total = 0.15 →
  urn.gold_coins / (urn.coins_and_beads - urn.beads) = 0.6 →
  urn.gold_coins / urn.total = 0.36 := by
  sorry

#check gold_coins_percentage

end NUMINAMATH_CALUDE_gold_coins_percentage_l2484_248407


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2484_248495

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 36) : 
  max x (max (x + 1) (x + 2)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2484_248495


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2484_248481

theorem abs_sum_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2484_248481


namespace NUMINAMATH_CALUDE_income_education_relationship_l2484_248436

/-- Represents the linear regression model for annual income and educational expenditure -/
structure IncomeEducationModel where
  -- x: annual income in ten thousand yuan
  -- y: annual educational expenditure in ten thousand yuan
  slope : Real
  intercept : Real
  equation : Real → Real := λ x => slope * x + intercept

/-- Theorem: In the given linear regression model, an increase of 1 in income
    results in an increase of 0.15 in educational expenditure -/
theorem income_education_relationship (model : IncomeEducationModel)
    (h_slope : model.slope = 0.15)
    (h_intercept : model.intercept = 0.2) :
    ∀ x : Real, model.equation (x + 1) - model.equation x = 0.15 := by
  sorry

#check income_education_relationship

end NUMINAMATH_CALUDE_income_education_relationship_l2484_248436


namespace NUMINAMATH_CALUDE_total_sheep_l2484_248460

theorem total_sheep (aaron_sheep beth_sheep : ℕ) 
  (h1 : aaron_sheep = 532)
  (h2 : beth_sheep = 76)
  (h3 : aaron_sheep = 7 * beth_sheep) : 
  aaron_sheep + beth_sheep = 608 := by
sorry

end NUMINAMATH_CALUDE_total_sheep_l2484_248460


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2484_248448

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 75 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2484_248448


namespace NUMINAMATH_CALUDE_cos_20_cos_385_minus_cos_70_sin_155_l2484_248473

theorem cos_20_cos_385_minus_cos_70_sin_155 :
  Real.cos (20 * π / 180) * Real.cos (385 * π / 180) - 
  Real.cos (70 * π / 180) * Real.sin (155 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_20_cos_385_minus_cos_70_sin_155_l2484_248473


namespace NUMINAMATH_CALUDE_number_of_boys_l2484_248479

/-- Proves that the number of boys is 15 given the problem conditions -/
theorem number_of_boys (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  (5 * men = women) → 
  (women = boys) → 
  (total_earnings = 180) → 
  (men_wage = 12) → 
  (5 * men * men_wage + women * (total_earnings - 5 * men * men_wage) / (women + boys) + 
   boys * (total_earnings - 5 * men * men_wage) / (women + boys) = total_earnings) →
  boys = 15 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l2484_248479


namespace NUMINAMATH_CALUDE_sum_of_squares_theorem_l2484_248454

theorem sum_of_squares_theorem (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p) 
  (h_sum : p / (q - r) + q / (r - p) + r / (p - q) = 3) :
  p^2 / (q - r)^2 + q^2 / (r - p)^2 + r^2 / (p - q)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_theorem_l2484_248454


namespace NUMINAMATH_CALUDE_total_strikes_is_180_l2484_248491

/-- Calculates the total number of strikes made by a clock in a 24-hour period. -/
def total_strikes : ℕ :=
  let hourly_strikes := 12 * 13 / 2 * 2  -- Sum of 1 to 12, twice
  let half_hour_strikes := 24            -- One strike every half hour (excluding full hours)
  hourly_strikes + half_hour_strikes

/-- Theorem stating that the total number of strikes in a 24-hour period is 180. -/
theorem total_strikes_is_180 : total_strikes = 180 := by
  sorry

end NUMINAMATH_CALUDE_total_strikes_is_180_l2484_248491


namespace NUMINAMATH_CALUDE_fraction_sum_equals_negative_two_l2484_248476

theorem fraction_sum_equals_negative_two (a b : ℝ) (h1 : a + b = 0) (h2 : a * b ≠ 0) :
  b / a + a / b = -2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_negative_two_l2484_248476


namespace NUMINAMATH_CALUDE_photo_arrangements_l2484_248412

/-- Represents the number of people in each category -/
structure People where
  teacher : Nat
  boys : Nat
  girls : Nat

/-- The total number of people -/
def total_people (p : People) : Nat :=
  p.teacher + p.boys + p.girls

/-- The number of arrangements for the given conditions -/
def arrangements (p : People) : Nat :=
  -- We'll define this function without implementation
  sorry

/-- Theorem stating the number of arrangements for the given problem -/
theorem photo_arrangements :
  let p := People.mk 1 2 2
  arrangements p = 24 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_l2484_248412


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2484_248444

/-- Given that a² and √b vary inversely, prove that b = 16 when a + b = 20 -/
theorem inverse_variation_problem (a b : ℝ) (k : ℝ) : 
  (∀ (a b : ℝ), a^2 * (b^(1/2)) = k) →  -- a² and √b vary inversely
  (4^2 * 16^(1/2) = k) →                -- a = 4 when b = 16
  (a + b = 20) →                        -- condition for the question
  (b = 16) :=                           -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2484_248444


namespace NUMINAMATH_CALUDE_power_expression_equality_l2484_248439

theorem power_expression_equality : (3^5 / 3^2) * 2^7 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_power_expression_equality_l2484_248439


namespace NUMINAMATH_CALUDE_base_2_representation_of_56_l2484_248408

/-- Represents a natural number in base 2 as a list of bits (least significant bit first) -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: go (m / 2)
    go n

/-- Converts a list of bits (least significant bit first) to a natural number -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem base_2_representation_of_56 :
  toBinary 56 = [false, false, false, true, true, true] := by
  sorry

end NUMINAMATH_CALUDE_base_2_representation_of_56_l2484_248408


namespace NUMINAMATH_CALUDE_unique_geometric_sequence_value_l2484_248466

/-- Two geometric sequences with specific conditions -/
structure GeometricSequences (a : ℝ) :=
  (a_seq : ℕ → ℝ)
  (b_seq : ℕ → ℝ)
  (a_positive : a > 0)
  (a_first : a_seq 1 = a)
  (b_minus_a_1 : b_seq 1 - a_seq 1 = 1)
  (b_minus_a_2 : b_seq 2 - a_seq 2 = 2)
  (b_minus_a_3 : b_seq 3 - a_seq 3 = 3)
  (a_geometric : ∀ n : ℕ, a_seq (n + 1) / a_seq n = a_seq 2 / a_seq 1)
  (b_geometric : ∀ n : ℕ, b_seq (n + 1) / b_seq n = b_seq 2 / b_seq 1)

/-- If the a_seq is unique, then a = 1/3 -/
theorem unique_geometric_sequence_value (a : ℝ) (h : GeometricSequences a) 
  (h_unique : ∃! q : ℝ, ∀ n : ℕ, h.a_seq (n + 1) = h.a_seq n * q) : 
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_unique_geometric_sequence_value_l2484_248466


namespace NUMINAMATH_CALUDE_dealer_profit_selling_price_percentage_l2484_248401

theorem dealer_profit (list_price : ℝ) (purchase_price selling_price : ℝ) : 
  purchase_price = 3/4 * list_price →
  selling_price = 2 * purchase_price →
  selling_price = 3/2 * list_price :=
by sorry

theorem selling_price_percentage (list_price : ℝ) (selling_price : ℝ) :
  selling_price = 3/2 * list_price →
  (selling_price - list_price) / list_price = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_dealer_profit_selling_price_percentage_l2484_248401


namespace NUMINAMATH_CALUDE_haley_growth_rate_l2484_248471

/-- Represents Haley's growth over time -/
structure Growth where
  initial_height : ℝ
  final_height : ℝ
  time_period : ℝ
  growth_rate : ℝ

/-- Theorem stating that given the initial conditions, Haley's growth rate is 3 inches per year -/
theorem haley_growth_rate (g : Growth) 
  (h1 : g.initial_height = 20)
  (h2 : g.final_height = 50)
  (h3 : g.time_period = 10)
  (h4 : g.growth_rate = (g.final_height - g.initial_height) / g.time_period) :
  g.growth_rate = 3 := by
  sorry

#check haley_growth_rate

end NUMINAMATH_CALUDE_haley_growth_rate_l2484_248471


namespace NUMINAMATH_CALUDE_sixth_member_income_l2484_248487

theorem sixth_member_income
  (family_size : ℕ)
  (average_income : ℕ)
  (income1 income2 income3 income4 income5 : ℕ)
  (h1 : family_size = 6)
  (h2 : average_income = 12000)
  (h3 : income1 = 11000)
  (h4 : income2 = 15000)
  (h5 : income3 = 10000)
  (h6 : income4 = 9000)
  (h7 : income5 = 13000) :
  average_income * family_size - (income1 + income2 + income3 + income4 + income5) = 14000 := by
  sorry

end NUMINAMATH_CALUDE_sixth_member_income_l2484_248487


namespace NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l2484_248493

theorem a_fourth_zero_implies_a_squared_zero 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : 
  A ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_fourth_zero_implies_a_squared_zero_l2484_248493


namespace NUMINAMATH_CALUDE_farm_animals_l2484_248445

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (cows : ℕ) : 
  total_animals = 120 →
  total_legs = 350 →
  total_animals = chickens + cows →
  total_legs = 2 * chickens + 4 * cows →
  chickens = 65 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2484_248445


namespace NUMINAMATH_CALUDE_urn_probability_l2484_248459

/-- Represents the total number of chips in the urn -/
def total_chips : ℕ := 15

/-- Represents the number of chips of each color -/
def chips_per_color : ℕ := 5

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of chips with each number -/
def chips_per_number : ℕ := 3

/-- Represents the number of different numbers on the chips -/
def num_numbers : ℕ := 5

/-- The probability of drawing two chips with either the same color or the same number -/
theorem urn_probability : 
  (num_colors * (chips_per_color.choose 2) + num_numbers * (chips_per_number.choose 2)) / (total_chips.choose 2) = 3 / 7 :=
by sorry

end NUMINAMATH_CALUDE_urn_probability_l2484_248459


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2484_248455

/-- Two lines are perpendicular if the sum of the products of their coefficients of x and y is zero -/
def are_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

/-- The first line: mx - (m+2)y + 2 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x - (m + 2) * y + 2 = 0

/-- The second line: 3x - my - 1 = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x - m * y - 1 = 0

theorem perpendicular_lines_m_values :
  ∀ m : ℝ, are_perpendicular m (-(m+2)) 3 (-m) → m = 0 ∨ m = -5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l2484_248455


namespace NUMINAMATH_CALUDE_quadratic_equation_value_l2484_248427

theorem quadratic_equation_value (x : ℝ) (h : x = 2) : x^2 + 5*x - 14 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_value_l2484_248427


namespace NUMINAMATH_CALUDE_cistern_leak_time_l2484_248419

/-- Given a cistern with two pipes A and B, this theorem proves the time it takes for pipe B to leak out the full cistern. -/
theorem cistern_leak_time 
  (fill_time_A : ℝ) 
  (fill_time_both : ℝ) 
  (h1 : fill_time_A = 10) 
  (h2 : fill_time_both = 59.999999999999964) : 
  ∃ (leak_time_B : ℝ), leak_time_B = 12 ∧ 
  (1 / fill_time_A - 1 / leak_time_B = 1 / fill_time_both) := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_time_l2484_248419


namespace NUMINAMATH_CALUDE_packages_sold_correct_l2484_248486

/-- The number of packages of gaskets sold during a week -/
def packages_sold : ℕ := 66

/-- The price per package of gaskets -/
def price_per_package : ℚ := 20

/-- The discount factor for packages in excess of 10 -/
def discount_factor : ℚ := 4/5

/-- The total payment received for the gaskets -/
def total_payment : ℚ := 1096

/-- Calculates the total cost for the given number of packages -/
def total_cost (n : ℕ) : ℚ :=
  if n ≤ 10 then n * price_per_package
  else 10 * price_per_package + (n - 10) * (discount_factor * price_per_package)

/-- Theorem stating that the number of packages sold satisfies the given conditions -/
theorem packages_sold_correct : 
  total_cost packages_sold = total_payment := by sorry

end NUMINAMATH_CALUDE_packages_sold_correct_l2484_248486


namespace NUMINAMATH_CALUDE_updated_mean_l2484_248411

theorem updated_mean (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 9 →
  (n * original_mean - n * decrement) / n = 191 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_l2484_248411
