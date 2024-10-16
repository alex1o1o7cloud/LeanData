import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_k_values_l1881_188101

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  isIsosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ (side1 = side3 ∧ side2 ≠ side1) ∨ (side2 = side3 ∧ side1 ≠ side2)

-- Define the quadratic equation
def quadraticRoots (k : ℝ) : Set ℝ :=
  {x : ℝ | x^2 - 4*x + k = 0}

-- Theorem statement
theorem isosceles_triangle_k_values :
  ∀ (t : IsoscelesTriangle) (k : ℝ),
    (t.side1 = 3 ∨ t.side2 = 3 ∨ t.side3 = 3) →
    (∃ (x y : ℝ), x ∈ quadraticRoots k ∧ y ∈ quadraticRoots k ∧ 
      ((t.side1 = x ∧ t.side2 = y) ∨ (t.side1 = x ∧ t.side3 = y) ∨ (t.side2 = x ∧ t.side3 = y))) →
    (k = 3 ∨ k = 4) :=
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_k_values_l1881_188101


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l1881_188112

/-- Calculates the principal amount given the interest rate, time, and total interest. -/
def calculate_principal (rate : ℚ) (time : ℕ) (interest : ℕ) : ℚ :=
  (interest : ℚ) * 100 / (rate * (time : ℚ))

/-- Proves that for a loan with 12% p.a. simple interest, if the interest after 3 years
    is Rs. 5400, then the principal amount borrowed was Rs. 15000. -/
theorem loan_principal_calculation (rate : ℚ) (time : ℕ) (interest : ℕ) :
  rate = 12 → time = 3 → interest = 5400 →
  calculate_principal rate time interest = 15000 := by
  sorry

#eval calculate_principal 12 3 5400

end NUMINAMATH_CALUDE_loan_principal_calculation_l1881_188112


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1881_188165

theorem polynomial_evaluation (w x y z : ℝ) 
  (eq1 : w + x + y + z = 5)
  (eq2 : 2*w + 4*x + 8*y + 16*z = 7)
  (eq3 : 3*w + 9*x + 27*y + 81*z = 11)
  (eq4 : 4*w + 16*x + 64*y + 256*z = 1) :
  5*w + 25*x + 125*y + 625*z = -60 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1881_188165


namespace NUMINAMATH_CALUDE_circle_equation_l1881_188152

/-- Given a circle with center (a, 5-3a) that passes through (0, 0) and (3, -1),
    prove that its equation is (x - 1)^2 + (y - 2)^2 = 5 -/
theorem circle_equation (a : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - (5 - 3*a))^2 = a^2 + (5 - 3*a)^2) →
  (a^2 + (5 - 3*a)^2 = 3^2 + (-1 - (5 - 3*a))^2) →
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1881_188152


namespace NUMINAMATH_CALUDE_tiles_for_taylors_room_l1881_188117

/-- Calculates the total number of tiles needed for a rectangular room with a border of smaller tiles --/
def total_tiles (room_length room_width border_tile_size interior_tile_size : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width) - 4
  let interior_length := room_length - 2 * border_tile_size
  let interior_width := room_width - 2 * border_tile_size
  let interior_area := interior_length * interior_width
  let interior_tiles := interior_area / (interior_tile_size * interior_tile_size)
  border_tiles + interior_tiles

/-- Theorem stating that for a 12x16 room with 1x1 border tiles and 2x2 interior tiles, 87 tiles are needed --/
theorem tiles_for_taylors_room : total_tiles 12 16 1 2 = 87 := by
  sorry

end NUMINAMATH_CALUDE_tiles_for_taylors_room_l1881_188117


namespace NUMINAMATH_CALUDE_john_investment_proof_l1881_188146

/-- The amount John invested in total -/
def total_investment : ℝ := 1200

/-- The annual interest rate for Bank A -/
def rate_A : ℝ := 0.04

/-- The annual interest rate for Bank B -/
def rate_B : ℝ := 0.06

/-- The number of years the money is invested -/
def years : ℕ := 2

/-- The total amount after two years -/
def final_amount : ℝ := 1300.50

/-- The amount John invested in Bank A -/
def investment_A : ℝ := 1138.57

theorem john_investment_proof :
  ∃ (x : ℝ), 
    x = investment_A ∧ 
    x ≥ 0 ∧ 
    x ≤ total_investment ∧
    x * (1 + rate_A) ^ years + (total_investment - x) * (1 + rate_B) ^ years = final_amount :=
by sorry

end NUMINAMATH_CALUDE_john_investment_proof_l1881_188146


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1881_188122

theorem algebraic_expression_value : ∀ x : ℝ, x^2 - 4*x = 5 → 2*x^2 - 8*x - 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1881_188122


namespace NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l1881_188185

theorem largest_n_for_quadratic_equation : 
  (∃ (n : ℕ), ∀ (m : ℕ), 
    (∃ (x y z : ℤ), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) ∧
    (m > n → ¬∃ (x y z : ℤ), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12)) ∧
  (∃ (x y z : ℤ), 10^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_quadratic_equation_l1881_188185


namespace NUMINAMATH_CALUDE_children_retaking_test_l1881_188186

theorem children_retaking_test (total : Float) (passed : Float) 
  (h1 : total = 698.0) (h2 : passed = 105.0) : 
  total - passed = 593.0 := by
  sorry

end NUMINAMATH_CALUDE_children_retaking_test_l1881_188186


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l1881_188189

theorem long_furred_brown_dogs
  (total : ℕ)
  (long_furred : ℕ)
  (brown : ℕ)
  (neither : ℕ)
  (h_total : total = 45)
  (h_long_furred : long_furred = 26)
  (h_brown : brown = 22)
  (h_neither : neither = 8) :
  long_furred + brown - (total - neither) = 11 := by
sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l1881_188189


namespace NUMINAMATH_CALUDE_expansion_equality_l1881_188115

-- Define the left-hand side of the equation
def lhs (x : ℝ) : ℝ := (5 * x^2 + 3 * x - 7) * 4 * x^3

-- Define the right-hand side of the equation
def rhs (x : ℝ) : ℝ := 20 * x^5 + 12 * x^4 - 28 * x^3

-- State the theorem
theorem expansion_equality : ∀ x : ℝ, lhs x = rhs x := by sorry

end NUMINAMATH_CALUDE_expansion_equality_l1881_188115


namespace NUMINAMATH_CALUDE_line_points_l1881_188154

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨4, 8⟩
  let p2 : Point := ⟨1, 2⟩
  let p3 : Point := ⟨3, 6⟩
  let p4 : Point := ⟨2, 4⟩
  let p5 : Point := ⟨5, 10⟩
  collinear p1 p2 p3 ∧ collinear p1 p2 p4 ∧ collinear p1 p2 p5 := by
  sorry

end NUMINAMATH_CALUDE_line_points_l1881_188154


namespace NUMINAMATH_CALUDE_multiply_by_9999_l1881_188184

theorem multiply_by_9999 : ∃! x : ℤ, x * 9999 = 806006795 :=
  by sorry

end NUMINAMATH_CALUDE_multiply_by_9999_l1881_188184


namespace NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l1881_188145

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let remaining_players := total_lives / lives_per_player
  initial_players - remaining_players

theorem correct_players_who_quit :
  players_who_quit 10 8 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_players_who_quit_correct_players_who_quit_l1881_188145


namespace NUMINAMATH_CALUDE_no_integer_b_with_four_integer_solutions_l1881_188121

theorem no_integer_b_with_four_integer_solutions : 
  ¬ ∃ b : ℤ, ∃ x₁ x₂ x₃ x₄ : ℤ, 
    (∀ x : ℤ, x^2 + b*x + 1 ≤ 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_b_with_four_integer_solutions_l1881_188121


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1881_188199

theorem basketball_score_proof (junior_score : ℕ) (percentage_increase : ℚ) : 
  junior_score = 260 → percentage_increase = 20/100 →
  junior_score + (junior_score + junior_score * percentage_increase) = 572 :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l1881_188199


namespace NUMINAMATH_CALUDE_f_equality_f_explicit_formula_l1881_188198

-- Define the function f
def f : ℝ → ℝ := fun x => 2 * x - x^2

-- State the theorem
theorem f_equality (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : 
  f (1 - Real.cos x) = Real.sin x ^ 2 := by
  sorry

-- Prove that f(x) = 2x - x^2 for 0 ≤ x ≤ 2
theorem f_explicit_formula (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  f x = 2 * x - x^2 := by
  sorry

end NUMINAMATH_CALUDE_f_equality_f_explicit_formula_l1881_188198


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_l1881_188168

/-- A function f(x) = 2ax^2 - x - 1 has only one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ 2 * a * x^2 - x - 1 = 0

/-- If f(x) = 2ax^2 - x - 1 has only one zero in the interval (0, 1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_l1881_188168


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1881_188127

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ (x - 1)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1881_188127


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1881_188192

theorem arithmetic_calculation : 10 - 9 + 8 * 7 + 6 - 5 * 4 + 3 - 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1881_188192


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1881_188151

theorem rationalize_denominator :
  ∀ x : ℝ, x > 0 → (30 : ℝ) / (5 - Real.sqrt x) = -30 - 6 * Real.sqrt x → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1881_188151


namespace NUMINAMATH_CALUDE_x_value_proof_l1881_188175

theorem x_value_proof (x : ℝ) (h : 3*x - 4*x + 7*x = 180) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1881_188175


namespace NUMINAMATH_CALUDE_exist_good_numbers_counterexample_l1881_188111

/-- A natural number is "good" if its decimal representation contains only 0s and 1s -/
def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- Sum of digits of a natural number in base 10 -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Statement: There exist two good numbers whose product is good, but the sum of digits
    of their product is not equal to the product of their sums of digits -/
theorem exist_good_numbers_counterexample :
  ∃ (A B : ℕ), is_good A ∧ is_good B ∧ is_good (A * B) ∧
    sum_of_digits (A * B) ≠ sum_of_digits A * sum_of_digits B :=
sorry

end NUMINAMATH_CALUDE_exist_good_numbers_counterexample_l1881_188111


namespace NUMINAMATH_CALUDE_bajazet_winning_strategy_l1881_188166

-- Define a polynomial of degree 4
def polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + 1

-- State the theorem
theorem bajazet_winning_strategy :
  ∀ (a b c : ℝ), ∃ (x : ℝ), polynomial a b c x = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_bajazet_winning_strategy_l1881_188166


namespace NUMINAMATH_CALUDE_anna_win_probability_l1881_188130

-- Define the game state as the sum modulo 4
inductive GameState
| Zero
| One
| Two
| Three

-- Define the die roll
def DieRoll : Type := Fin 6

-- Define the probability of winning for each game state
def winProbability : GameState → ℚ
| GameState.Zero => 0
| GameState.One => 50/99
| GameState.Two => 60/99
| GameState.Three => 62/99

-- Define the transition probability function
def transitionProbability (s : GameState) (r : DieRoll) : GameState :=
  match s, r.val + 1 with
  | GameState.Zero, n => match n % 4 with
    | 0 => GameState.Zero
    | 1 => GameState.One
    | 2 => GameState.Two
    | 3 => GameState.Three
    | _ => GameState.Zero  -- This case should never occur
  | GameState.One, n => match n % 4 with
    | 0 => GameState.One
    | 1 => GameState.Two
    | 2 => GameState.Three
    | 3 => GameState.Zero
    | _ => GameState.One  -- This case should never occur
  | GameState.Two, n => match n % 4 with
    | 0 => GameState.Two
    | 1 => GameState.Three
    | 2 => GameState.Zero
    | 3 => GameState.One
    | _ => GameState.Two  -- This case should never occur
  | GameState.Three, n => match n % 4 with
    | 0 => GameState.Three
    | 1 => GameState.Zero
    | 2 => GameState.One
    | 3 => GameState.Two
    | _ => GameState.Three  -- This case should never occur

-- Theorem statement
theorem anna_win_probability :
  (1 : ℚ) / 6 * (1 - winProbability GameState.Zero) +
  1 / 3 * (1 - winProbability GameState.One) +
  1 / 3 * (1 - winProbability GameState.Two) +
  1 / 6 * (1 - winProbability GameState.Three) = 52 / 99 :=
by sorry


end NUMINAMATH_CALUDE_anna_win_probability_l1881_188130


namespace NUMINAMATH_CALUDE_unique_distribution_l1881_188129

/-- Represents the number of ways to distribute n identical balls into boxes with given capacities -/
def distribution_count (n : ℕ) (capacities : List ℕ) : ℕ :=
  sorry

/-- The capacities of the four boxes -/
def box_capacities : List ℕ := [3, 5, 7, 8]

/-- The total number of balls to distribute -/
def total_balls : ℕ := 19

/-- Theorem stating that there's only one way to distribute the balls -/
theorem unique_distribution : distribution_count total_balls box_capacities = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_distribution_l1881_188129


namespace NUMINAMATH_CALUDE_sum_of_squares_equals_165_l1881_188138

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Property of combination numbers -/
axiom comb_property (n m : ℕ) : binomial n (m-1) + binomial n m = binomial (n+1) m

/-- Special case of combination numbers -/
axiom comb_special_case : binomial 2 2 = binomial 3 3

/-- The sum of squares of binomial coefficients from C(2,2) to C(10,2) -/
def sum_of_squares : ℕ := 
  binomial 2 2 + binomial 3 2 + binomial 4 2 + binomial 5 2 + 
  binomial 6 2 + binomial 7 2 + binomial 8 2 + binomial 9 2 + binomial 10 2

theorem sum_of_squares_equals_165 : sum_of_squares = 165 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_equals_165_l1881_188138


namespace NUMINAMATH_CALUDE_carpet_area_calculation_l1881_188103

/-- Calculates the area of carpet required for a room and corridor -/
theorem carpet_area_calculation (main_length main_width corridor_length corridor_width : ℝ) 
  (h_main_length : main_length = 15)
  (h_main_width : main_width = 12)
  (h_corridor_length : corridor_length = 10)
  (h_corridor_width : corridor_width = 3)
  (h_feet_to_yard : 3 = 1) :
  (main_length * main_width + corridor_length * corridor_width) / 9 = 23.33 := by
sorry

#eval (15 * 12 + 10 * 3) / 9

end NUMINAMATH_CALUDE_carpet_area_calculation_l1881_188103


namespace NUMINAMATH_CALUDE_hawks_score_l1881_188153

theorem hawks_score (total_points margin : ℕ) (h1 : total_points = 50) (h2 : margin = 18) :
  (total_points - margin) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l1881_188153


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l1881_188140

-- Define the function f(x) = x³
def f (x : ℝ) : ℝ := x^3

-- Theorem stating that f is monotonically increasing on ℝ
theorem f_monotone_increasing : 
  ∀ (x y : ℝ), x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l1881_188140


namespace NUMINAMATH_CALUDE_polynomial_factor_sum_l1881_188196

theorem polynomial_factor_sum (a b : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^3 + a*x^2 + b*x + 8 = (x + 1) * (x + 2) * (x + c)) → 
  a + b = 21 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_sum_l1881_188196


namespace NUMINAMATH_CALUDE_de_length_l1881_188183

/-- Triangle ABC with sides AB = 24, AC = 26, and BC = 22 -/
structure Triangle :=
  (AB : ℝ) (AC : ℝ) (BC : ℝ)

/-- Points D and E on sides AB and AC respectively -/
structure PointsDE (T : Triangle) :=
  (D : ℝ) (E : ℝ)
  (hD : D ≥ 0 ∧ D ≤ T.AB)
  (hE : E ≥ 0 ∧ E ≤ T.AC)

/-- DE is parallel to BC and contains the center of the inscribed circle -/
def contains_incenter (T : Triangle) (P : PointsDE T) : Prop :=
  ∃ k : ℝ, P.D / T.AB = P.E / T.AC ∧ k > 0 ∧ k < 1 ∧
    P.D = k * T.AB ∧ P.E = k * T.AC

/-- The main theorem -/
theorem de_length (T : Triangle) (P : PointsDE T) 
    (h1 : T.AB = 24) (h2 : T.AC = 26) (h3 : T.BC = 22)
    (h4 : contains_incenter T P) : 
  P.E - P.D = 275 / 18 := by sorry

end NUMINAMATH_CALUDE_de_length_l1881_188183


namespace NUMINAMATH_CALUDE_cat_collar_nylon_l1881_188107

/-- The number of inches of nylon needed for one dog collar -/
def dog_collar_nylon : ℝ := 18

/-- The total number of inches of nylon needed for all collars -/
def total_nylon : ℝ := 192

/-- The number of dog collars -/
def num_dog_collars : ℕ := 9

/-- The number of cat collars -/
def num_cat_collars : ℕ := 3

/-- Theorem stating that the number of inches of nylon needed for one cat collar is 10 -/
theorem cat_collar_nylon : 
  (total_nylon - dog_collar_nylon * num_dog_collars) / num_cat_collars = 10 := by
sorry

end NUMINAMATH_CALUDE_cat_collar_nylon_l1881_188107


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1881_188133

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Theorem statement
theorem quadratic_function_properties :
  -- Conditions
  (∃ r : ℝ, (∀ x : ℝ, f x = 0 ↔ x = r)) ∧ 
  (∀ x : ℝ, (deriv f) x = 2*x + 2) →
  -- Conclusions
  (∀ x : ℝ, f x = x^2 + 2*x + 1) ∧ 
  (∫ x in (-1)..(0), f x = 1/3) ∧
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    ∫ x in (-1)..(-t), f x = ∫ x in (-t)..0, f x ∧
    t = 1 - 1/32) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1881_188133


namespace NUMINAMATH_CALUDE_curve_is_line_segment_l1881_188141

/-- Parametric curve defined by x = 3t² + 4 and y = t² - 2, where 0 ≤ t ≤ 3 -/
def parametric_curve (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + 4, t^2 - 2)

/-- The range of the parameter t -/
def t_range : Set ℝ := {t : ℝ | 0 ≤ t ∧ t ≤ 3}

/-- The set of points on the curve -/
def curve_points : Set (ℝ × ℝ) :=
  {p | ∃ t ∈ t_range, p = parametric_curve t}

/-- Theorem: The curve is a line segment -/
theorem curve_is_line_segment :
  ∃ a b c : ℝ, a ≠ 0 ∧ curve_points = {p : ℝ × ℝ | a * p.1 + b * p.2 = c} ∩
    {p : ℝ × ℝ | ∃ t ∈ t_range, p = parametric_curve t} :=
by sorry

end NUMINAMATH_CALUDE_curve_is_line_segment_l1881_188141


namespace NUMINAMATH_CALUDE_fraction_of_length_equality_l1881_188159

theorem fraction_of_length_equality : (2 / 7 : ℚ) * 3 = (3 / 7 : ℚ) * 2 := by sorry

end NUMINAMATH_CALUDE_fraction_of_length_equality_l1881_188159


namespace NUMINAMATH_CALUDE_pecan_amount_correct_l1881_188147

/-- Represents the composition of a nut mixture -/
structure NutMixture where
  pecan_pounds : ℝ
  cashew_pounds : ℝ
  pecan_price : ℝ
  mixture_price : ℝ

/-- Verifies if a given nut mixture satisfies the problem conditions -/
def is_valid_mixture (m : NutMixture) : Prop :=
  m.cashew_pounds = 2 ∧
  m.pecan_price = 5.60 ∧
  m.mixture_price = 4.34

/-- Calculates the total value of the mixture -/
def mixture_value (m : NutMixture) : ℝ :=
  (m.pecan_pounds + m.cashew_pounds) * m.mixture_price

/-- Calculates the value of pecans in the mixture -/
def pecan_value (m : NutMixture) : ℝ :=
  m.pecan_pounds * m.pecan_price

/-- The main theorem stating that the mixture with 1.33333333333 pounds of pecans
    satisfies the problem conditions -/
theorem pecan_amount_correct (m : NutMixture) 
  (h_valid : is_valid_mixture m)
  (h_pecan : m.pecan_pounds = 1.33333333333) :
  mixture_value m = pecan_value m + m.cashew_pounds * (mixture_value m / (m.pecan_pounds + m.cashew_pounds)) :=
by
  sorry


end NUMINAMATH_CALUDE_pecan_amount_correct_l1881_188147


namespace NUMINAMATH_CALUDE_exists_composite_evaluation_l1881_188125

/-- A polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Evaluate a polynomial at a given integer -/
def evalPoly (p : IntPolynomial) (x : Int) : Int :=
  p.foldr (fun a b => a + x * b) 0

/-- A number is composite if it has a factor other than 1 and itself -/
def isComposite (n : Int) : Prop :=
  ∃ m, 1 < m ∧ m < n.natAbs ∧ n % m = 0

theorem exists_composite_evaluation (polys : List IntPolynomial) :
  ∃ a : Int, ∀ p ∈ polys, isComposite (evalPoly p a) := by
  sorry

#check exists_composite_evaluation

end NUMINAMATH_CALUDE_exists_composite_evaluation_l1881_188125


namespace NUMINAMATH_CALUDE_committee_size_is_24_l1881_188120

/-- The number of sandwiches per person -/
def sandwiches_per_person : ℕ := 2

/-- The number of croissants per pack -/
def croissants_per_pack : ℕ := 12

/-- The cost of one pack of croissants in cents -/
def cost_per_pack : ℕ := 800

/-- The total amount spent on croissants in cents -/
def total_spent : ℕ := 3200

/-- The number of people on the committee -/
def committee_size : ℕ := total_spent / cost_per_pack * croissants_per_pack / sandwiches_per_person

theorem committee_size_is_24 : committee_size = 24 := by
  sorry

end NUMINAMATH_CALUDE_committee_size_is_24_l1881_188120


namespace NUMINAMATH_CALUDE_factor_expression_l1881_188163

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1881_188163


namespace NUMINAMATH_CALUDE_sum_of_digits_of_a_l1881_188182

def a : ℕ := (10^10) - 47

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_a : sum_of_digits a = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_a_l1881_188182


namespace NUMINAMATH_CALUDE_total_sticks_is_326_l1881_188194

/-- The number of sticks needed for four rafts given specific conditions -/
def total_sticks : ℕ :=
  let simon := 45
  let gerry := (3 * simon) / 5
  let micky := simon + gerry + 15
  let darryl := 2 * micky - 7
  simon + gerry + micky + darryl

/-- Theorem stating that the total number of sticks needed is 326 -/
theorem total_sticks_is_326 : total_sticks = 326 := by
  sorry

end NUMINAMATH_CALUDE_total_sticks_is_326_l1881_188194


namespace NUMINAMATH_CALUDE_f_inequality_l1881_188119

-- Define a function f on the real numbers
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f x < f y

-- State the theorem
theorem f_inequality (h1 : is_even f) (h2 : is_increasing_on_nonneg f) : f (-2) > f 1 ∧ f 1 > f 0 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1881_188119


namespace NUMINAMATH_CALUDE_problem_1_l1881_188191

theorem problem_1 (a : ℚ) (h : a = 4/5) :
  -24.7 * a + 1.3 * a - (33/5) * a = -24 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1881_188191


namespace NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l1881_188135

theorem consecutive_integers_fourth_power_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) = 12 * (3 * n + 3) → 
  n^4 + (n + 1)^4 + (n + 2)^4 = 7793 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l1881_188135


namespace NUMINAMATH_CALUDE_school_trip_buses_l1881_188158

/-- The number of buses for a school trip, given the number of supervisors per bus and the total number of supervisors. -/
def number_of_buses (supervisors_per_bus : ℕ) (total_supervisors : ℕ) : ℕ :=
  total_supervisors / supervisors_per_bus

/-- Theorem stating that the number of buses is 7, given the conditions from the problem. -/
theorem school_trip_buses : number_of_buses 3 21 = 7 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_buses_l1881_188158


namespace NUMINAMATH_CALUDE_blocks_left_in_second_tower_is_two_l1881_188176

/-- The number of blocks left standing in the second tower --/
def blocks_left_in_second_tower (first_stack_height : ℕ) 
                                (second_stack_diff : ℕ) 
                                (third_stack_diff : ℕ) 
                                (blocks_left_in_third : ℕ) 
                                (total_fallen : ℕ) : ℕ :=
  let second_stack_height := first_stack_height + second_stack_diff
  let third_stack_height := second_stack_height + third_stack_diff
  let total_blocks := first_stack_height + second_stack_height + third_stack_height
  let fallen_from_first := first_stack_height
  let fallen_from_third := third_stack_height - blocks_left_in_third
  let fallen_from_second := total_fallen - fallen_from_first - fallen_from_third
  second_stack_height - fallen_from_second

theorem blocks_left_in_second_tower_is_two :
  blocks_left_in_second_tower 7 5 7 3 33 = 2 := by
  sorry

end NUMINAMATH_CALUDE_blocks_left_in_second_tower_is_two_l1881_188176


namespace NUMINAMATH_CALUDE_fraction_sum_equals_62_l1881_188106

theorem fraction_sum_equals_62 (a b : ℝ) : 
  a = (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3) →
  b = (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) →
  b / a + a / b = 62 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_62_l1881_188106


namespace NUMINAMATH_CALUDE_cricket_players_l1881_188172

/-- The number of students who like to play basketball -/
def B : ℕ := 12

/-- The number of students who like to play both basketball and cricket -/
def B_and_C : ℕ := 3

/-- The number of students who like to play basketball or cricket or both -/
def B_or_C : ℕ := 17

/-- The number of students who like to play cricket -/
def C : ℕ := B_or_C - B + B_and_C

theorem cricket_players : C = 8 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_l1881_188172


namespace NUMINAMATH_CALUDE_five_consecutive_not_square_l1881_188180

theorem five_consecutive_not_square (n : ℤ) : ¬∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_not_square_l1881_188180


namespace NUMINAMATH_CALUDE_even_function_composition_l1881_188139

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

theorem even_function_composition (g : ℝ → ℝ) (h : is_even_function g) :
  is_even_function (g ∘ g) :=
by
  sorry

end NUMINAMATH_CALUDE_even_function_composition_l1881_188139


namespace NUMINAMATH_CALUDE_total_students_proof_l1881_188171

/-- The total number of students enrolled at both schools last year -/
def total_students_last_year (x y : ℕ) : ℕ := x + y

/-- The number of students at school XX this year -/
def school_xx_this_year (x : ℕ) : ℕ := x + (7 * x) / 100

/-- The number of students at school YY this year -/
def school_yy_this_year (y : ℕ) : ℕ := y + (3 * y) / 100

/-- The growth difference between schools XX and YY -/
def growth_difference (x y : ℕ) : ℕ := school_xx_this_year x - x - (school_yy_this_year y - y)

theorem total_students_proof (x y : ℕ) 
  (h1 : y = 2400)
  (h2 : growth_difference x y = 40) :
  total_students_last_year x y = 4000 := by
  sorry

end NUMINAMATH_CALUDE_total_students_proof_l1881_188171


namespace NUMINAMATH_CALUDE_cubic_decreasing_iff_l1881_188102

theorem cubic_decreasing_iff (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x) ((3 * a * x^2) - 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x) > (a * y^3 - y)) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_decreasing_iff_l1881_188102


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l1881_188116

theorem complement_intersection_problem (U M N : Set Nat) : 
  U = {1, 2, 3, 4, 5} →
  M = {1, 2, 3} →
  N = {2, 3, 5} →
  (U \ M) ∩ N = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l1881_188116


namespace NUMINAMATH_CALUDE_erased_number_proof_l1881_188144

theorem erased_number_proof (n : ℕ) (x : ℕ) : 
  x ≤ n →
  (n * (n + 1) / 2 - x) / (n - 1) = 866 / 19 →
  x = 326 :=
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l1881_188144


namespace NUMINAMATH_CALUDE_parallelogram_is_rhombus_l1881_188156

/-- A parallelogram ABCD in a 2D Euclidean space. -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- Vector addition -/
def vecAdd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

/-- Vector subtraction -/
def vecSub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Dot product of two vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The zero vector -/
def zeroVec : ℝ × ℝ := (0, 0)

/-- Theorem: A parallelogram is a rhombus if it satisfies certain vector conditions -/
theorem parallelogram_is_rhombus (ABCD : Parallelogram)
  (h1 : vecAdd (vecSub ABCD.B ABCD.A) (vecSub ABCD.D ABCD.C) = zeroVec)
  (h2 : dotProduct (vecSub (vecSub ABCD.B ABCD.A) (vecSub ABCD.D ABCD.A)) (vecSub ABCD.C ABCD.A) = 0) :
  ABCD.A = ABCD.B ∧ ABCD.B = ABCD.C ∧ ABCD.C = ABCD.D ∧ ABCD.D = ABCD.A := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_is_rhombus_l1881_188156


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l1881_188110

theorem females_with_advanced_degrees 
  (total_employees : ℕ) 
  (total_females : ℕ) 
  (employees_with_advanced_degrees : ℕ) 
  (males_with_college_only : ℕ) 
  (h1 : total_employees = 160) 
  (h2 : total_females = 90) 
  (h3 : employees_with_advanced_degrees = 80) 
  (h4 : males_with_college_only = 40) :
  total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 50 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l1881_188110


namespace NUMINAMATH_CALUDE_sequence_general_formula_l1881_188142

/-- Given a sequence {a_n} where a₁ = 6 and aₙ₊₁/aₙ = (n+3)/n for n ≥ 1,
    this theorem states that aₙ = n(n+1)(n+2) for all n ≥ 1 -/
theorem sequence_general_formula (a : ℕ → ℝ) 
    (h1 : a 1 = 6)
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) / a n = (n + 3) / n) :
  ∀ n : ℕ, n ≥ 1 → a n = n * (n + 1) * (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l1881_188142


namespace NUMINAMATH_CALUDE_larger_number_with_given_hcf_lcm_factors_l1881_188118

theorem larger_number_with_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 120 ∧
  ∃ k : ℕ, Nat.lcm a b = 120 * 13 * 17 * 23 * k ∧ k = 1 →
  max a b = 26520 :=
by sorry

end NUMINAMATH_CALUDE_larger_number_with_given_hcf_lcm_factors_l1881_188118


namespace NUMINAMATH_CALUDE_rotation_90_clockwise_l1881_188181

-- Define the possible positions in the circle
inductive Position
  | Top
  | Left
  | Right

-- Define the shapes
inductive Shape
  | Pentagon
  | SmallerCircle
  | Rectangle

-- Define a function to represent the initial configuration
def initial_config : Position → Shape
  | Position.Top => Shape.Pentagon
  | Position.Left => Shape.SmallerCircle
  | Position.Right => Shape.Rectangle

-- Define a function to represent the configuration after 90° clockwise rotation
def rotated_config : Position → Shape
  | Position.Top => Shape.SmallerCircle
  | Position.Right => Shape.Pentagon
  | Position.Left => Shape.Rectangle

-- Theorem stating that the rotated configuration is correct
theorem rotation_90_clockwise :
  ∀ p : Position, rotated_config p = initial_config (match p with
    | Position.Top => Position.Right
    | Position.Right => Position.Left
    | Position.Left => Position.Top
  ) :=
by sorry

end NUMINAMATH_CALUDE_rotation_90_clockwise_l1881_188181


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l1881_188105

theorem regular_polygon_with_150_degree_angles (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : (n : ℝ) * 150 = 180 * (n - 2)) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_l1881_188105


namespace NUMINAMATH_CALUDE_campers_third_week_l1881_188197

/-- Proves the number of campers in the third week given conditions about three consecutive weeks of camping. -/
theorem campers_third_week
  (total : ℕ)
  (second_week : ℕ)
  (h_total : total = 150)
  (h_second : second_week = 40)
  (h_difference : second_week = (second_week - 10) + 10) :
  total - (second_week - 10) - second_week = 80 :=
by sorry

end NUMINAMATH_CALUDE_campers_third_week_l1881_188197


namespace NUMINAMATH_CALUDE_crease_lines_equivalence_l1881_188155

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the ellipse
structure Ellipse where
  focus1 : Point
  focus2 : Point
  majorAxis : ℝ

-- Define the set of points on crease lines
def CreaseLines (c : Circle) (a : Point) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | ∃ (a' : Point), a'.x^2 + a'.y^2 = c.radius^2 ∧ 
    (p.1 - (a.x + a'.x)/2)^2 + (p.2 - (a.y + a'.y)/2)^2 = ((a.x - a'.x)^2 + (a.y - a'.y)^2) / 4 }

-- Define the set of points not on the ellipse
def NotOnEllipse (e : Ellipse) : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (Real.sqrt ((p.1 - e.focus1.x)^2 + (p.2 - e.focus1.y)^2) + 
                 Real.sqrt ((p.1 - e.focus2.x)^2 + (p.2 - e.focus2.y)^2)) ≠ e.majorAxis }

-- Theorem statement
theorem crease_lines_equivalence 
  (c : Circle) (a : Point) (e : Ellipse) 
  (h1 : (a.x - c.center.1)^2 + (a.y - c.center.2)^2 < c.radius^2)  -- A is inside the circle
  (h2 : e.focus1 = Point.mk c.center.1 c.center.2)  -- O is a focus of the ellipse
  (h3 : e.focus2 = a)  -- A is the other focus of the ellipse
  (h4 : e.majorAxis = c.radius) :  -- The major axis of the ellipse is R
  CreaseLines c a = NotOnEllipse e := by
  sorry

end NUMINAMATH_CALUDE_crease_lines_equivalence_l1881_188155


namespace NUMINAMATH_CALUDE_lilia_earnings_l1881_188114

/-- Represents Lilia's peach selling scenario -/
structure PeachSale where
  total : Nat
  sold_to_friends : Nat
  price_friends : Real
  sold_to_relatives : Nat
  price_relatives : Real
  kept : Nat

/-- Calculates the total earnings from selling peaches -/
def total_earnings (sale : PeachSale) : Real :=
  sale.sold_to_friends * sale.price_friends + sale.sold_to_relatives * sale.price_relatives

/-- Theorem stating that Lilia's earnings from selling 14 peaches is $25 -/
theorem lilia_earnings (sale : PeachSale) 
  (h1 : sale.total = 15)
  (h2 : sale.sold_to_friends = 10)
  (h3 : sale.price_friends = 2)
  (h4 : sale.sold_to_relatives = 4)
  (h5 : sale.price_relatives = 1.25)
  (h6 : sale.kept = 1)
  (h7 : sale.sold_to_friends + sale.sold_to_relatives + sale.kept = sale.total) :
  total_earnings sale = 25 := by
  sorry

end NUMINAMATH_CALUDE_lilia_earnings_l1881_188114


namespace NUMINAMATH_CALUDE_square_root_of_square_l1881_188104

theorem square_root_of_square (x : ℝ) : {y : ℝ | y^2 = x^2} = {x, -x} := by sorry

end NUMINAMATH_CALUDE_square_root_of_square_l1881_188104


namespace NUMINAMATH_CALUDE_gcd_of_repeated_digits_l1881_188134

theorem gcd_of_repeated_digits : 
  ∃ (g : ℕ), 
    (∀ n : ℕ, 100 ≤ n → n < 1000 → 
      g ∣ (n * 1000000000 + n * 1000000 + n * 1000 + n)) ∧
    (∀ m : ℕ, 
      (∀ n : ℕ, 100 ≤ n → n < 1000 → 
        m ∣ (n * 1000000000 + n * 1000000 + n * 1000 + n)) → 
      m ∣ g) ∧
    g = 1001001001 :=
by
  sorry

end NUMINAMATH_CALUDE_gcd_of_repeated_digits_l1881_188134


namespace NUMINAMATH_CALUDE_parallel_lines_distance_l1881_188179

/-- The distance between two parallel lines -/
theorem parallel_lines_distance (a b c d e f : ℝ) :
  (a = 1 ∧ b = 2 ∧ c = -1) →
  (d = 2 ∧ e = 4 ∧ f = 3) →
  (∃ (k : ℝ), k ≠ 0 ∧ d = k * a ∧ e = k * b) →
  (abs (f / d - c / a) / Real.sqrt (a^2 + b^2) : ℝ) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_distance_l1881_188179


namespace NUMINAMATH_CALUDE_joans_total_seashells_l1881_188178

/-- The number of seashells Joan found on the beach -/
def initial_seashells : ℕ := 70

/-- The number of seashells Sam gave to Joan -/
def additional_seashells : ℕ := 27

/-- Theorem: Joan's total number of seashells is 97 -/
theorem joans_total_seashells : initial_seashells + additional_seashells = 97 := by
  sorry

end NUMINAMATH_CALUDE_joans_total_seashells_l1881_188178


namespace NUMINAMATH_CALUDE_kenzo_office_chairs_l1881_188143

theorem kenzo_office_chairs :
  ∀ (initial_chairs : ℕ),
    (∃ (chairs_legs tables_legs remaining_chairs_legs : ℕ),
      chairs_legs = 5 * initial_chairs ∧
      tables_legs = 20 * 3 ∧
      remaining_chairs_legs = (6 * chairs_legs) / 10 ∧
      remaining_chairs_legs + tables_legs = 300) →
    initial_chairs = 80 := by
  sorry

end NUMINAMATH_CALUDE_kenzo_office_chairs_l1881_188143


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1881_188195

theorem not_sufficient_nor_necessary : ¬(∀ x : ℝ, (x - 2) * (x - 1) > 0 → (x - 2 > 0 ∨ x - 1 > 0)) ∧
                                       ¬(∀ x : ℝ, (x - 2 > 0 ∨ x - 1 > 0) → (x - 2) * (x - 1) > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l1881_188195


namespace NUMINAMATH_CALUDE_stamp_difference_l1881_188100

theorem stamp_difference (p q : ℕ) (h1 : p * 4 = q * 7) 
  (h2 : (p - 8) * 5 = (q + 8) * 6) : p - q = 8 := by
  sorry

end NUMINAMATH_CALUDE_stamp_difference_l1881_188100


namespace NUMINAMATH_CALUDE_study_tour_students_l1881_188187

/-- Represents the number of students participating in the study tour. -/
def num_students : ℕ := 46

/-- Represents the number of dormitories. -/
def num_dormitories : ℕ := 6

theorem study_tour_students :
  (∃ (n : ℕ), n = num_dormitories ∧
    6 * n + 10 = num_students ∧
    8 * (n - 1) + 4 < num_students ∧
    num_students < 8 * (n - 1) + 8) :=
by sorry

end NUMINAMATH_CALUDE_study_tour_students_l1881_188187


namespace NUMINAMATH_CALUDE_min_width_proof_l1881_188160

/-- The minimum width of a rectangular area satisfying the given conditions -/
def min_width : ℝ := 10

/-- The length of the rectangular area -/
def length (w : ℝ) : ℝ := w + 15

/-- The area of the rectangular region -/
def area (w : ℝ) : ℝ := w * length w

theorem min_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 200 → w ≥ min_width) ∧
  (area min_width ≥ 200) :=
sorry

end NUMINAMATH_CALUDE_min_width_proof_l1881_188160


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1881_188157

/-- Two quantities vary inversely if their product is constant -/
def vary_inversely (a b : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_variation_problem (a b : ℝ → ℝ) 
  (h_inverse : vary_inversely a b)
  (h_initial : b 800 = 0.5) :
  b 3200 = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1881_188157


namespace NUMINAMATH_CALUDE_four_integer_solutions_l1881_188169

def satisfies_equation (a : ℤ) : Prop :=
  |2 * a + 7| + |2 * a - 1| = 8

theorem four_integer_solutions :
  ∃ (S : Finset ℤ), (∀ a ∈ S, satisfies_equation a) ∧ 
                    (∀ a : ℤ, satisfies_equation a → a ∈ S) ∧
                    Finset.card S = 4 :=
sorry

end NUMINAMATH_CALUDE_four_integer_solutions_l1881_188169


namespace NUMINAMATH_CALUDE_tangent_line_sum_l1881_188188

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 2 / x

theorem tangent_line_sum (a b m : ℝ) : 
  (∀ x : ℝ, 3 * x + f a 1 = b) →  -- Tangent line equation
  (∀ x : ℝ, f a x = a * Real.log x + 2 / x) →  -- Function definition
  (f a 1 = m) →  -- Point of tangency
  a + b = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l1881_188188


namespace NUMINAMATH_CALUDE_max_quarters_in_box_l1881_188164

/-- Represents the number of coins of each type in the coin box -/
structure CoinBox where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the box -/
def total_coins (box : CoinBox) : ℕ :=
  box.nickels + box.dimes + box.quarters

/-- The total value of coins in cents -/
def total_value (box : CoinBox) : ℕ :=
  5 * box.nickels + 10 * box.dimes + 25 * box.quarters

/-- Theorem stating the maximum number of quarters possible -/
theorem max_quarters_in_box :
  ∃ (box : CoinBox),
    total_coins box = 120 ∧
    total_value box = 1000 ∧
    (∀ (other_box : CoinBox),
      total_coins other_box = 120 →
      total_value other_box = 1000 →
      other_box.quarters ≤ box.quarters) ∧
    box.quarters = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_quarters_in_box_l1881_188164


namespace NUMINAMATH_CALUDE_inequality_holds_iff_b_greater_than_one_l1881_188150

theorem inequality_holds_iff_b_greater_than_one (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_b_greater_than_one_l1881_188150


namespace NUMINAMATH_CALUDE_value_calculation_l1881_188174

theorem value_calculation (number : ℕ) (value : ℕ) : 
  number = 48 → value = (number / 4 + 15) → value = 27 := by sorry

end NUMINAMATH_CALUDE_value_calculation_l1881_188174


namespace NUMINAMATH_CALUDE_tangent_line_sum_l1881_188108

/-- Given a function f: ℝ → ℝ with a tangent line y = -x + 8 at x = 5, 
    prove that f(5) + f'(5) = 2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 5 + (deriv f 5) * (x - 5) = -x + 8) : 
    f 5 + (deriv f 5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l1881_188108


namespace NUMINAMATH_CALUDE_total_weight_is_675_l1881_188136

/-- The total weight Tom is moving with, given his weight, the weight he holds in each hand, and the weight of his vest. -/
def total_weight_moved (tom_weight : ℝ) (hand_weight_multiplier : ℝ) (vest_weight_multiplier : ℝ) : ℝ :=
  tom_weight + (vest_weight_multiplier * tom_weight) + (2 * hand_weight_multiplier * tom_weight)

/-- Theorem stating that the total weight Tom is moving with is 675 kg -/
theorem total_weight_is_675 :
  total_weight_moved 150 1.5 0.5 = 675 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_675_l1881_188136


namespace NUMINAMATH_CALUDE_wednesday_rainfall_calculation_l1881_188190

/-- Calculates the rainfall on Wednesday given the conditions of the problem -/
def wednesday_rainfall (monday : ℝ) (tuesday_difference : ℝ) : ℝ :=
  2 * (monday + (monday - tuesday_difference))

/-- Theorem stating that given the specific conditions, Wednesday's rainfall is 2.2 inches -/
theorem wednesday_rainfall_calculation :
  wednesday_rainfall 0.9 0.7 = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_rainfall_calculation_l1881_188190


namespace NUMINAMATH_CALUDE_complex_multiplication_l1881_188148

theorem complex_multiplication (i : ℂ) (z₁ z₂ : ℂ) :
  i * i = -1 →
  z₁ = 1 + 2 * i →
  z₂ = -3 * i →
  z₁ * z₂ = 6 - 3 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1881_188148


namespace NUMINAMATH_CALUDE_jamie_rice_purchase_l1881_188131

/-- The price of rice in cents per pound -/
def rice_price : ℚ := 60

/-- The price of flour in cents per pound -/
def flour_price : ℚ := 30

/-- The total amount of rice and flour bought in pounds -/
def total_amount : ℚ := 30

/-- The total amount spent in cents -/
def total_spent : ℚ := 1500

/-- The amount of rice bought in pounds -/
def rice_amount : ℚ := 20

theorem jamie_rice_purchase :
  ∃ (flour_amount : ℚ),
    rice_amount + flour_amount = total_amount ∧
    rice_price * rice_amount + flour_price * flour_amount = total_spent :=
by sorry

end NUMINAMATH_CALUDE_jamie_rice_purchase_l1881_188131


namespace NUMINAMATH_CALUDE_sequence_convergence_l1881_188149

theorem sequence_convergence (a : ℕ → ℚ) :
  a 1 = 3 / 5 →
  (∀ n : ℕ, a (n + 1) = 2 - 1 / (a n)) →
  a 2018 = 4031 / 4029 := by
sorry

end NUMINAMATH_CALUDE_sequence_convergence_l1881_188149


namespace NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1881_188126

/-- A truncated cone with a tangent sphere -/
structure TruncatedConeWithSphere where
  bottomRadius : ℝ
  topRadius : ℝ
  sphereRadius : ℝ
  isTangent : Bool

/-- The theorem stating the radius of the sphere in the given problem -/
theorem sphere_radius_in_truncated_cone 
  (cone : TruncatedConeWithSphere)
  (h1 : cone.bottomRadius = 24)
  (h2 : cone.topRadius = 6)
  (h3 : cone.isTangent = true) :
  cone.sphereRadius = 12 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_in_truncated_cone_l1881_188126


namespace NUMINAMATH_CALUDE_sinusoidal_period_l1881_188161

theorem sinusoidal_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.sin (b * x + c) + d) →
  (∃ n : ℕ, n = 4 ∧ (2 * π) / b = (2 * π) / n) →
  b = 4 :=
by sorry

end NUMINAMATH_CALUDE_sinusoidal_period_l1881_188161


namespace NUMINAMATH_CALUDE_sum_a_b_equals_twelve_l1881_188167

theorem sum_a_b_equals_twelve (a b c d : ℝ) 
  (h1 : b + c = 9) 
  (h2 : c + d = 3) 
  (h3 : a + d = 6) : 
  a + b = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_twelve_l1881_188167


namespace NUMINAMATH_CALUDE_negation_of_implication_l1881_188162

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1881_188162


namespace NUMINAMATH_CALUDE_circle_intersection_range_l1881_188173

noncomputable section

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop :=
  (x - Real.sqrt a)^2 + (y - Real.sqrt a)^2 = 9

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the condition for point P
def exists_P (a : ℝ) : Prop :=
  ∃ P : ℝ × ℝ, circle_M a P.1 P.2 ∧
    (P.1 - point_A.1) * (point_B.1 - point_A.1) +
    (P.2 - point_A.2) * (point_B.2 - point_A.2) = 0

-- State the theorem
theorem circle_intersection_range :
  ∀ a : ℝ, exists_P a ↔ 1/2 ≤ a ∧ a ≤ 25/2 :=
sorry

end

end NUMINAMATH_CALUDE_circle_intersection_range_l1881_188173


namespace NUMINAMATH_CALUDE_range_of_a_proof_l1881_188124

/-- Proposition p: there exists a real x₀ such that x₀² + 2ax₀ - 2a = 0 -/
def p (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*a*x₀ - 2*a = 0

/-- Proposition q: for all real x, ax² + 4x + a > -2x² + 1 -/
def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + 4*x + a > -2*x^2 + 1

/-- The range of a given the conditions -/
def range_of_a : Set ℝ := {a : ℝ | a ≤ -2}

theorem range_of_a_proof (h1 : ∀ a : ℝ, p a ∨ q a) (h2 : ∀ a : ℝ, ¬(p a ∧ q a)) :
  ∀ a : ℝ, a ∈ range_of_a ↔ (p a ∧ ¬q a) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_proof_l1881_188124


namespace NUMINAMATH_CALUDE_school_boys_count_l1881_188113

/-- The percentage of boys who are Muslims -/
def muslim_percentage : ℚ := 44 / 100

/-- The percentage of boys who are Hindus -/
def hindu_percentage : ℚ := 28 / 100

/-- The percentage of boys who are Sikhs -/
def sikh_percentage : ℚ := 10 / 100

/-- The number of boys belonging to other communities -/
def other_communities : ℕ := 153

/-- The total number of boys in the school -/
def total_boys : ℕ := 850

theorem school_boys_count :
  (1 - (muslim_percentage + hindu_percentage + sikh_percentage)) * (total_boys : ℚ) = other_communities := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l1881_188113


namespace NUMINAMATH_CALUDE_candy_bar_chocolate_cost_difference_l1881_188177

/-- The problem of calculating the difference in cost between a candy bar and chocolate. -/
theorem candy_bar_chocolate_cost_difference :
  let dans_money : ℕ := 2
  let candy_bar_cost : ℕ := 6
  let chocolate_cost : ℕ := 3
  candy_bar_cost - chocolate_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_chocolate_cost_difference_l1881_188177


namespace NUMINAMATH_CALUDE_corner_sum_is_164_l1881_188132

/-- Represents a 9x9 checkerboard with numbers 1 to 81 --/
def Checkerboard : Type := Fin 9 → Fin 9 → Fin 81

/-- Returns the number at a given position on the checkerboard --/
def numberAt (board : Checkerboard) (row col : Fin 9) : Fin 81 :=
  if row % 2 = 0 then
    ↑(9 * row + (9 - col))
  else
    ↑(9 * row + col + 1)

/-- The sum of the numbers in the four corners of the checkerboard --/
def cornerSum (board : Checkerboard) : Nat :=
  (numberAt board 0 0).val +
  (numberAt board 0 8).val +
  (numberAt board 8 0).val +
  (numberAt board 8 8).val

theorem corner_sum_is_164 (board : Checkerboard) : cornerSum board = 164 := by
  sorry


end NUMINAMATH_CALUDE_corner_sum_is_164_l1881_188132


namespace NUMINAMATH_CALUDE_existence_of_opposite_colors_l1881_188170

/-- Represents a piece on the circle -/
inductive Piece
| White
| Black

/-- Represents the circle with pieces placed on it -/
structure Circle :=
  (pieces : Fin 40 → Piece)
  (white_count : Nat)
  (black_count : Nat)
  (white_count_eq : white_count = 25)
  (black_count_eq : black_count = 15)
  (total_count : white_count + black_count = 40)

/-- Two points are diametrically opposite if their indices differ by 20 (mod 40) -/
def diametricallyOpposite (i j : Fin 40) : Prop :=
  (i.val + 20) % 40 = j.val ∨ (j.val + 20) % 40 = i.val

/-- Main theorem: There exist diametrically opposite white and black pieces -/
theorem existence_of_opposite_colors (c : Circle) :
  ∃ (i j : Fin 40), diametricallyOpposite i j ∧ 
    c.pieces i = Piece.White ∧ c.pieces j = Piece.Black :=
sorry

end NUMINAMATH_CALUDE_existence_of_opposite_colors_l1881_188170


namespace NUMINAMATH_CALUDE_sheep_to_horse_ratio_l1881_188193

theorem sheep_to_horse_ratio :
  let horse_food_per_day : ℕ := 230
  let total_horse_food : ℕ := 12880
  let num_sheep : ℕ := 16
  let num_horses : ℕ := total_horse_food / horse_food_per_day
  (num_sheep : ℚ) / (num_horses : ℚ) = 2 / 7 :=
by sorry

end NUMINAMATH_CALUDE_sheep_to_horse_ratio_l1881_188193


namespace NUMINAMATH_CALUDE_initial_average_age_l1881_188137

/-- Given a group of people with an unknown initial average age, 
    prove that when a new person joins and changes the average, 
    we can determine the initial average age. -/
theorem initial_average_age 
  (n : ℕ) 
  (new_person_age : ℕ) 
  (new_average : ℝ) 
  (h1 : n = 10)
  (h2 : new_person_age = 37)
  (h3 : new_average = 17) :
  ∃ (initial_average : ℝ),
    n * initial_average + new_person_age = (n + 1) * new_average ∧ 
    initial_average = 15 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_age_l1881_188137


namespace NUMINAMATH_CALUDE_ladder_distance_l1881_188109

theorem ladder_distance (angle : Real) (length : Real) (distance : Real) : 
  angle = 60 * π / 180 →  -- Convert 60° to radians
  length = 19 →
  distance = length * Real.cos angle →
  distance = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l1881_188109


namespace NUMINAMATH_CALUDE_tank_volume_ratio_l1881_188128

theorem tank_volume_ratio :
  ∀ (tank1_volume tank2_volume : ℚ),
  tank1_volume > 0 →
  tank2_volume > 0 →
  (3 / 4 : ℚ) * tank1_volume = (5 / 8 : ℚ) * tank2_volume →
  tank1_volume / tank2_volume = (5 / 6 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_tank_volume_ratio_l1881_188128


namespace NUMINAMATH_CALUDE_missing_number_is_eight_l1881_188123

/-- Represents a 1 × 12 table filled with numbers -/
def Table := Fin 12 → ℝ

/-- The sum of any four adjacent cells in the table is 11 -/
def SumAdjacent (t : Table) : Prop :=
  ∀ i : Fin 9, t i + t (i + 1) + t (i + 2) + t (i + 3) = 11

/-- The table contains the known numbers 4, 1, and 2 -/
def ContainsKnownNumbers (t : Table) : Prop :=
  ∃ (i j k : Fin 12), t i = 4 ∧ t j = 1 ∧ t k = 2

/-- The theorem to be proved -/
theorem missing_number_is_eight
  (t : Table)
  (h1 : SumAdjacent t)
  (h2 : ContainsKnownNumbers t) :
  ∃ (l : Fin 12), t l = 8 :=
sorry

end NUMINAMATH_CALUDE_missing_number_is_eight_l1881_188123
