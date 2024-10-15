import Mathlib

namespace NUMINAMATH_CALUDE_digit_sum_of_predecessor_l2161_216102

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Returns true if all elements in the list are distinct -/
def allDistinct (l : Digits) : Prop := ∀ i j, i ≠ j → l.get? i ≠ l.get? j

/-- Calculates the sum of all elements in the list -/
def digitSum (l : Digits) : Nat := l.sum

/-- Converts a natural number to its digit representation -/
def toDigits (n : Nat) : Digits := sorry

/-- Converts a digit representation back to a natural number -/
def fromDigits (d : Digits) : Nat := sorry

theorem digit_sum_of_predecessor (n : Nat) :
  (∃ d : Digits, fromDigits d = n ∧ allDistinct d ∧ digitSum d = 44) →
  (∃ d' : Digits, fromDigits d' = n - 1 ∧ (digitSum d' = 43 ∨ digitSum d' = 52)) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_of_predecessor_l2161_216102


namespace NUMINAMATH_CALUDE_f_extreme_values_l2161_216198

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x ^ 2) + Real.sin (Real.sin x ^ 2)

theorem f_extreme_values (k : ℤ) :
  ∃ (x : ℝ), x = (k : ℝ) * (Real.pi / 4) ∧ 
  (∀ (y : ℝ), f y ≤ f x ∨ f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_f_extreme_values_l2161_216198


namespace NUMINAMATH_CALUDE_division_problem_l2161_216145

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1345)
  (h2 : a = 1596)
  (h3 : a = b * q + 15) :
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2161_216145


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2161_216109

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def B : Set ℝ := {y | ∃ x, y = -x^2}

-- State the theorem
theorem intersection_complement_equality : A ∩ (U \ B) = {x | x > 0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2161_216109


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l2161_216101

theorem right_triangle_leg_square (a b c : ℝ) (h1 : c = a + 2) (h2 : a^2 + b^2 = c^2) : b^2 = 4*a + 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l2161_216101


namespace NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_42_4_plus_24_4_l2161_216116

theorem units_digit_of_sum (a b : ℕ) : (a^4 + b^4) % 10 = ((a^4 % 10) + (b^4 % 10)) % 10 := by sorry

theorem units_digit_of_42_4_plus_24_4 : (42^4 + 24^4) % 10 = 2 := by
  have h1 : 42^4 % 10 = 6 := by sorry
  have h2 : 24^4 % 10 = 6 := by sorry
  have h3 : (6 + 6) % 10 = 2 := by sorry
  
  calc
    (42^4 + 24^4) % 10 = ((42^4 % 10) + (24^4 % 10)) % 10 := by apply units_digit_of_sum
    _ = (6 + 6) % 10 := by rw [h1, h2]
    _ = 2 := by rw [h3]

end NUMINAMATH_CALUDE_units_digit_of_sum_units_digit_of_42_4_plus_24_4_l2161_216116


namespace NUMINAMATH_CALUDE_average_speed_is_25_l2161_216114

def initial_reading : ℕ := 45654
def final_reading : ℕ := 45854
def total_time : ℕ := 8

def distance : ℕ := final_reading - initial_reading
def average_speed : ℚ := distance / total_time

theorem average_speed_is_25 : average_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_is_25_l2161_216114


namespace NUMINAMATH_CALUDE_f_even_implies_specific_points_l2161_216148

/-- A function f on the real numbers. -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2 * a - b

/-- The domain of f is [2a-1, a^2+1] -/
def domain (a : ℝ) : Set ℝ := Set.Icc (2 * a - 1) (a^2 + 1)

/-- f is an even function -/
def is_even (a b : ℝ) : Prop := ∀ x ∈ domain a, f a b x = f a b (-x)

/-- The theorem stating that given the conditions, (a, b) can only be (0, 0) or (-2, 0) -/
theorem f_even_implies_specific_points :
  ∀ a b : ℝ, is_even a b → (a = 0 ∧ b = 0) ∨ (a = -2 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_f_even_implies_specific_points_l2161_216148


namespace NUMINAMATH_CALUDE_quadratic_inequality_implication_l2161_216124

theorem quadratic_inequality_implication (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implication_l2161_216124


namespace NUMINAMATH_CALUDE_largest_whole_number_solution_l2161_216151

theorem largest_whole_number_solution : 
  (∀ n : ℕ, n > 3 → ¬(1/4 + n/5 < 9/10)) ∧ 
  (1/4 + 3/5 < 9/10) := by
sorry

end NUMINAMATH_CALUDE_largest_whole_number_solution_l2161_216151


namespace NUMINAMATH_CALUDE_f_of_three_equals_seven_l2161_216181

/-- Given a function f(x) = x^7 - ax^5 + bx^3 + cx + 2 where f(-3) = -3, prove that f(3) = 7 -/
theorem f_of_three_equals_seven 
  (a b c : ℝ) 
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = x^7 - a*x^5 + b*x^3 + c*x + 2)
  (h2 : f (-3) = -3) :
  f 3 = 7 := by
sorry

end NUMINAMATH_CALUDE_f_of_three_equals_seven_l2161_216181


namespace NUMINAMATH_CALUDE_faculty_reduction_l2161_216119

theorem faculty_reduction (initial_faculty : ℕ) : 
  (initial_faculty : ℝ) * 0.85 * 0.80 = 195 → 
  initial_faculty = 287 := by
  sorry

end NUMINAMATH_CALUDE_faculty_reduction_l2161_216119


namespace NUMINAMATH_CALUDE_symmetric_circle_correct_l2161_216144

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 6)^2 = 16

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y = 0

-- Define the symmetric circle C
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 6)^2 + (y + 5)^2 = 16

-- Theorem stating that the symmetric circle C is correct
theorem symmetric_circle_correct :
  ∀ (x y : ℝ),
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
   symmetry_line ((x + x₀) / 2) ((y + y₀) / 2)) →
  symmetric_circle x y :=
sorry

end NUMINAMATH_CALUDE_symmetric_circle_correct_l2161_216144


namespace NUMINAMATH_CALUDE_parade_team_size_l2161_216103

theorem parade_team_size : 
  ∃ n : ℕ, 
    n % 5 = 0 ∧ 
    n ≥ 1000 ∧ 
    n % 4 = 3 ∧ 
    n % 3 = 2 ∧ 
    n % 2 = 1 ∧ 
    n = 1045 ∧ 
    ∀ m : ℕ, 
      (m % 5 = 0 ∧ 
       m ≥ 1000 ∧ 
       m % 4 = 3 ∧ 
       m % 3 = 2 ∧ 
       m % 2 = 1) → 
      m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_parade_team_size_l2161_216103


namespace NUMINAMATH_CALUDE_one_incorrect_statement_l2161_216184

-- Define the structure for a statistical statement
structure StatStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the list of statements
def statements : List StatStatement :=
  [
    { id := 1, content := "Residuals can be used to judge the effectiveness of model fitting", isCorrect := true },
    { id := 2, content := "Given a regression equation: ŷ=3-5x, when variable x increases by one unit, y increases by an average of 5 units", isCorrect := false },
    { id := 3, content := "The linear regression line: ŷ=b̂x+â must pass through the point (x̄, ȳ)", isCorrect := true },
    { id := 4, content := "In a 2×2 contingency table, it is calculated that χ²=13.079, thus there is a 99% confidence that there is a relationship between the two variables (where P(χ²≥10.828)=0.001)", isCorrect := true }
  ]

-- Theorem: Exactly one statement is incorrect
theorem one_incorrect_statement : 
  (statements.filter (fun s => !s.isCorrect)).length = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_incorrect_statement_l2161_216184


namespace NUMINAMATH_CALUDE_meaningful_expression_l2161_216129

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / (x - 1)) ↔ x ≥ -3 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2161_216129


namespace NUMINAMATH_CALUDE_complement_of_A_is_closed_ray_l2161_216175

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set A as the domain of log(2-x)
def A : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem complement_of_A_is_closed_ray :
  Set.compl A = Set.Ici (2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_is_closed_ray_l2161_216175


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l2161_216166

theorem cube_sum_theorem (a b c : ℝ) 
  (h1 : a + b + c = 8)
  (h2 : a * b + a * c + b * c = 9)
  (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l2161_216166


namespace NUMINAMATH_CALUDE_range_of_a_l2161_216149

-- Define the conditions
def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x a : ℝ, q x a → p x) :
  ∀ a : ℝ, (∀ x : ℝ, q x a → p x) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2161_216149


namespace NUMINAMATH_CALUDE_logarithm_simplification_l2161_216161

theorem logarithm_simplification : 
  Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 + 4^(-1/2 : ℝ) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l2161_216161


namespace NUMINAMATH_CALUDE_initial_bottles_count_l2161_216158

/-- The number of bottles Maria drank -/
def maria_drank : ℝ := 14.0

/-- The number of bottles Maria's sister drank -/
def sister_drank : ℝ := 8.0

/-- The number of bottles left in the fridge -/
def bottles_left : ℕ := 23

/-- The initial number of bottles in Maria's fridge -/
def initial_bottles : ℝ := maria_drank + sister_drank + bottles_left

theorem initial_bottles_count : initial_bottles = 45.0 := by
  sorry

end NUMINAMATH_CALUDE_initial_bottles_count_l2161_216158


namespace NUMINAMATH_CALUDE_unique_solution_conditions_l2161_216182

theorem unique_solution_conditions (n p : ℕ) :
  (∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p ^ z) ↔
  (p > 1 ∧ 
   (n - 1) % (p - 1) = 0 ∧
   ∀ k : ℕ, n ≠ p ^ k) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_conditions_l2161_216182


namespace NUMINAMATH_CALUDE_tan_equality_implies_negative_thirty_l2161_216120

theorem tan_equality_implies_negative_thirty (n : ℤ) :
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) →
  n = -30 := by sorry

end NUMINAMATH_CALUDE_tan_equality_implies_negative_thirty_l2161_216120


namespace NUMINAMATH_CALUDE_union_of_sets_l2161_216121

theorem union_of_sets : 
  let P : Set ℕ := {1, 2}
  let Q : Set ℕ := {2, 3}
  P ∪ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2161_216121


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l2161_216131

theorem sin_cos_sum_equals_half : 
  Real.sin (21 * π / 180) * Real.cos (9 * π / 180) + 
  Real.sin (69 * π / 180) * Real.sin (9 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_half_l2161_216131


namespace NUMINAMATH_CALUDE_linear_inequality_equivalence_l2161_216168

theorem linear_inequality_equivalence :
  ∀ x : ℝ, (2 * x - 4 > 0) ↔ (x > 2) := by sorry

end NUMINAMATH_CALUDE_linear_inequality_equivalence_l2161_216168


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2161_216117

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b

/-- The point (2,2) lies on the hyperbola -/
def point_on_hyperbola (h : Hyperbola) : Prop :=
  4 / h.a^2 - 4 / h.b^2 = 1

/-- The distance from the foci to the asymptotes equals the length of the real axis -/
def foci_distance_condition (h : Hyperbola) : Prop :=
  h.b = 2 * h.a

theorem hyperbola_equation (h : Hyperbola) 
  (h_point : point_on_hyperbola h) 
  (h_distance : foci_distance_condition h) : 
  h.a = Real.sqrt 3 ∧ h.b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2161_216117


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2161_216196

theorem negation_of_proposition :
  (¬ (∀ n : ℕ, ¬ (Nat.Prime (2^n - 2)))) ↔ (∃ n : ℕ, Nat.Prime (2^n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2161_216196


namespace NUMINAMATH_CALUDE_girls_points_in_checkers_tournament_l2161_216107

theorem girls_points_in_checkers_tournament (x : ℕ) : 
  x > 0 →  -- number of girls is positive
  2 * x * (10 * x - 1) = 18 →  -- derived equation for girls' points
  ∃ (total_games : ℕ) (total_points : ℕ),
    -- total number of games
    total_games = (10 * x) * (10 * x - 1) / 2 ∧
    -- total points distributed
    total_points = 2 * total_games ∧
    -- boys' points are 4 times girls' points
    4 * (2 * x * (10 * x - 1)) = total_points - (2 * x * (10 * x - 1)) :=
by
  sorry

#check girls_points_in_checkers_tournament

end NUMINAMATH_CALUDE_girls_points_in_checkers_tournament_l2161_216107


namespace NUMINAMATH_CALUDE_cheese_division_possible_l2161_216140

/-- Represents the state of the cheese pieces -/
structure CheeseState where
  piece1 : ℕ
  piece2 : ℕ
  piece3 : ℕ

/-- Represents a single cut operation -/
inductive Cut
  | cut12 : Cut  -- Cut 1g from piece1 and piece2
  | cut13 : Cut  -- Cut 1g from piece1 and piece3
  | cut23 : Cut  -- Cut 1g from piece2 and piece3

/-- Applies a single cut to a CheeseState -/
def applyCut (state : CheeseState) (cut : Cut) : CheeseState :=
  match cut with
  | Cut.cut12 => ⟨state.piece1 - 1, state.piece2 - 1, state.piece3⟩
  | Cut.cut13 => ⟨state.piece1 - 1, state.piece2, state.piece3 - 1⟩
  | Cut.cut23 => ⟨state.piece1, state.piece2 - 1, state.piece3 - 1⟩

/-- Checks if all pieces in a CheeseState are equal -/
def allEqual (state : CheeseState) : Prop :=
  state.piece1 = state.piece2 ∧ state.piece2 = state.piece3

/-- The theorem to be proved -/
theorem cheese_division_possible : ∃ (cuts : List Cut), 
  let finalState := cuts.foldl applyCut ⟨5, 8, 11⟩
  allEqual finalState ∧ finalState.piece1 ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_cheese_division_possible_l2161_216140


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2161_216186

theorem complex_modulus_problem (z : ℂ) : z = (2 - I) / (1 + I) → Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2161_216186


namespace NUMINAMATH_CALUDE_third_month_sale_l2161_216179

def average_sale : ℝ := 6500
def num_months : ℕ := 6
def sale_month1 : ℝ := 6435
def sale_month2 : ℝ := 6927
def sale_month4 : ℝ := 7230
def sale_month5 : ℝ := 6562
def sale_month6 : ℝ := 4991

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 6855 := by
  sorry

end NUMINAMATH_CALUDE_third_month_sale_l2161_216179


namespace NUMINAMATH_CALUDE_allen_pizza_change_l2161_216110

def pizza_order (num_boxes : ℕ) (price_per_box : ℚ) (tip_fraction : ℚ) (payment : ℚ) : ℚ :=
  let total_cost := num_boxes * price_per_box
  let tip := total_cost * tip_fraction
  let total_spent := total_cost + tip
  payment - total_spent

theorem allen_pizza_change : 
  pizza_order 5 7 (1/7) 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_allen_pizza_change_l2161_216110


namespace NUMINAMATH_CALUDE_factor_sum_l2161_216142

theorem factor_sum (P Q : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^2 + Q) → 
  P + Q = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l2161_216142


namespace NUMINAMATH_CALUDE_strategy2_is_cheaper_l2161_216160

def original_price : ℝ := 12000

def strategy1_cost (price : ℝ) : ℝ :=
  price * (1 - 0.30) * (1 - 0.15) * (1 - 0.05)

def strategy2_cost (price : ℝ) : ℝ :=
  price * (1 - 0.45) * (1 - 0.10) * (1 - 0.10) + 150

theorem strategy2_is_cheaper :
  strategy2_cost original_price < strategy1_cost original_price :=
by sorry

end NUMINAMATH_CALUDE_strategy2_is_cheaper_l2161_216160


namespace NUMINAMATH_CALUDE_rectangular_garden_length_l2161_216136

/-- Calculates the length of a rectangular garden given its perimeter and breadth. -/
theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 480) 
  (h2 : breadth = 100) : 
  perimeter / 2 - breadth = 140 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_length_l2161_216136


namespace NUMINAMATH_CALUDE_stratified_sample_teachers_l2161_216104

def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

def stratified_sample (total : ℕ) (strata : List ℕ) (sample : ℕ) : List ℕ :=
  strata.map (λ stratum => (stratum * sample) / total)

theorem stratified_sample_teachers :
  stratified_sample total_teachers [senior_teachers, intermediate_teachers, junior_teachers] sample_size = [12, 20, 8] := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_teachers_l2161_216104


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2161_216115

-- Define the hyperbola
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the standard equation of a hyperbola
def standard_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  (17 * x^2) / 4 - (17 * y^2) / 64 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = 4 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem hyperbola_equation (h : Hyperbola) :
  (∃ x y : ℝ, asymptote_equation x y) ∧
  (h.c = parabola_focus.1) →
  ∀ x y : ℝ, standard_equation h x y :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2161_216115


namespace NUMINAMATH_CALUDE_diameter_endpoint2_coordinates_l2161_216170

def circle_center : ℝ × ℝ := (1, 2)
def diameter_endpoint1 : ℝ × ℝ := (4, 6)

theorem diameter_endpoint2_coordinates :
  let midpoint := circle_center
  let endpoint1 := diameter_endpoint1
  let endpoint2 := (2 * midpoint.1 - endpoint1.1, 2 * midpoint.2 - endpoint1.2)
  endpoint2 = (-2, -2) :=
sorry

end NUMINAMATH_CALUDE_diameter_endpoint2_coordinates_l2161_216170


namespace NUMINAMATH_CALUDE_curve_in_second_quadrant_l2161_216172

-- Define the curve C
def C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Theorem statement
theorem curve_in_second_quadrant :
  (∀ a : ℝ, ∀ x y : ℝ, C a x y → second_quadrant x y) →
  (∀ a : ℝ, a ∈ Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_curve_in_second_quadrant_l2161_216172


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l2161_216195

theorem gcd_special_numbers : Nat.gcd 33333333 666666666 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l2161_216195


namespace NUMINAMATH_CALUDE_eighth_term_value_l2161_216156

def is_arithmetic_sequence (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) - s n = d

theorem eighth_term_value (a : ℕ → ℚ) 
  (h1 : a 2 = 3)
  (h2 : a 5 = 1)
  (h3 : is_arithmetic_sequence (fun n ↦ 1 / (a n + 1))) :
  a 8 = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l2161_216156


namespace NUMINAMATH_CALUDE_increasing_implies_a_geq_neg_two_l2161_216177

/-- A quadratic function f(x) = x^2 + 2(a-1)x - 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x - 3

/-- The property of f being increasing on [3, +∞) -/
def is_increasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≥ 3 → y ≥ 3 → x < y → f a x < f a y

/-- Theorem: If f is increasing on [3, +∞), then a ≥ -2 -/
theorem increasing_implies_a_geq_neg_two (a : ℝ) :
  is_increasing_on_interval a → a ≥ -2 :=
by sorry

end NUMINAMATH_CALUDE_increasing_implies_a_geq_neg_two_l2161_216177


namespace NUMINAMATH_CALUDE_cans_first_day_correct_l2161_216143

/-- The number of cans collected on the first day, given the conditions of the problem -/
def cans_first_day : ℕ := 20

/-- The number of days cans are collected -/
def collection_days : ℕ := 5

/-- The daily increase in the number of cans collected -/
def daily_increase : ℕ := 5

/-- The total number of cans collected over the collection period -/
def total_cans : ℕ := 150

/-- Theorem stating that the number of cans collected on the first day is correct -/
theorem cans_first_day_correct : 
  cans_first_day * collection_days + 
  (daily_increase * (collection_days - 1) * collection_days / 2) = total_cans := by
  sorry

end NUMINAMATH_CALUDE_cans_first_day_correct_l2161_216143


namespace NUMINAMATH_CALUDE_frans_original_seat_l2161_216105

/-- Represents the seats in the theater --/
inductive Seat
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the friends --/
inductive Friend
  | ada
  | bea
  | ceci
  | dee
  | edie
  | fran

/-- Represents the direction of movement --/
inductive Direction
  | left
  | right

/-- Represents a movement of a friend --/
structure Movement where
  friend : Friend
  distance : Nat
  direction : Direction

/-- The initial seating arrangement --/
def initialSeating : Friend → Seat := sorry

/-- The movements of the friends --/
def friendMovements : List Movement := [
  ⟨Friend.ada, 3, Direction.right⟩,
  ⟨Friend.bea, 2, Direction.left⟩,
  ⟨Friend.ceci, 0, Direction.right⟩,
  ⟨Friend.dee, 0, Direction.right⟩,
  ⟨Friend.edie, 1, Direction.right⟩
]

/-- Function to apply movements and get the final seating arrangement --/
def applyMovements (initial : Friend → Seat) (movements : List Movement) : Friend → Seat := sorry

/-- Function to find the vacant seat after movements --/
def findVacantSeat (seating : Friend → Seat) : Seat := sorry

/-- Theorem stating Fran's original seat --/
theorem frans_original_seat :
  initialSeating Friend.fran = Seat.three ∧
  (findVacantSeat (applyMovements initialSeating friendMovements) = Seat.one ∨
   findVacantSeat (applyMovements initialSeating friendMovements) = Seat.six) := by
  sorry

end NUMINAMATH_CALUDE_frans_original_seat_l2161_216105


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2161_216152

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  (a^2 - 5*a + 6 = 0) → 
  (b^2 - 5*b + 6 = 0) → 
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2161_216152


namespace NUMINAMATH_CALUDE_son_age_problem_l2161_216192

theorem son_age_problem (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end NUMINAMATH_CALUDE_son_age_problem_l2161_216192


namespace NUMINAMATH_CALUDE_inequality_proof_l2161_216133

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + 
  a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1/9 ∧ 
  (b * c * d / (1 - a)^2 + a * c * d / (1 - b)^2 + 
   a * b * d / (1 - c)^2 + a * b * c / (1 - d)^2 = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2161_216133


namespace NUMINAMATH_CALUDE_rotated_angle_intersection_l2161_216197

/-- 
Given an angle α, when its terminal side is rotated clockwise by π/2,
the intersection of the new angle with the unit circle centered at the origin
has coordinates (sin α, -cos α).
-/
theorem rotated_angle_intersection (α : Real) : 
  let rotated_angle := α - π / 2
  let x := Real.cos rotated_angle
  let y := Real.sin rotated_angle
  (x, y) = (Real.sin α, -Real.cos α) := by
sorry

end NUMINAMATH_CALUDE_rotated_angle_intersection_l2161_216197


namespace NUMINAMATH_CALUDE_inequality_properties_l2161_216128

theorem inequality_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (b / a ≤ (b + c^2) / (a + c^2)) ∧ (a + b < Real.sqrt (2 * (a^2 + b^2))) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l2161_216128


namespace NUMINAMATH_CALUDE_remainder_theorem_l2161_216165

theorem remainder_theorem (T E N S E' N' S' : ℤ)
  (h1 : T = N * E + S)
  (h2 : N = N' * E' + S')
  : T % (E + E') = E * S' + S := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2161_216165


namespace NUMINAMATH_CALUDE_stating_max_perpendicular_diagonals_correct_l2161_216146

/-- 
Given a regular n-gon with n ≥ 3, this function returns the maximum number of diagonals
that can be drawn such that any intersecting pair is perpendicular.
-/
def maxPerpendicularDiagonals (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 2 else n - 3

/-- 
Theorem stating that maxPerpendicularDiagonals correctly computes the maximum number
of diagonals in a regular n-gon (n ≥ 3) such that any intersecting pair is perpendicular.
-/
theorem max_perpendicular_diagonals_correct (n : ℕ) (h : n ≥ 3) :
  maxPerpendicularDiagonals n = 
    if n % 2 = 0 then n - 2 else n - 3 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_perpendicular_diagonals_correct_l2161_216146


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2161_216111

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : is_geometric_sequence a)
  (h_sum : a 2 + a 6 = 10)
  (h_prod : a 3 * a 5 = 16) :
  ∃ q : ℝ, (q = Real.sqrt 2 ∨ q = Real.sqrt 2 / 2) ∧ 
    ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2161_216111


namespace NUMINAMATH_CALUDE_practice_time_difference_l2161_216108

/-- Represents the practice schedule for Carlo's music recital --/
structure PracticeSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem stating the difference in practice time between Wednesday and Thursday --/
theorem practice_time_difference (schedule : PracticeSchedule) : 
  schedule.monday = 2 * schedule.tuesday →
  schedule.tuesday = schedule.wednesday - 10 →
  schedule.wednesday > schedule.thursday →
  schedule.thursday = 50 →
  schedule.friday = 60 →
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday = 300 →
  schedule.wednesday - schedule.thursday = 5 := by
  sorry

end NUMINAMATH_CALUDE_practice_time_difference_l2161_216108


namespace NUMINAMATH_CALUDE_not_octal_7857_l2161_216106

def is_octal_digit (d : Nat) : Prop := d ≤ 7

def is_octal_number (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 8 → is_octal_digit d

theorem not_octal_7857 : ¬ is_octal_number 7857 := by
  sorry

end NUMINAMATH_CALUDE_not_octal_7857_l2161_216106


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2161_216180

theorem stratified_sample_size 
  (total_employees : ℕ) 
  (male_employees : ℕ) 
  (sampled_male : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : male_employees = 90) 
  (h3 : sampled_male = 27) 
  (h4 : male_employees < total_employees) :
  (sampled_male : ℚ) / (male_employees : ℚ) * (total_employees : ℚ) = 36 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2161_216180


namespace NUMINAMATH_CALUDE_count_measurable_weights_l2161_216130

/-- Represents the available weights in grams -/
def available_weights : List ℕ := [1, 2, 6, 26]

/-- Represents a configuration of weights on the balance scale -/
structure WeightConfiguration :=
  (left : List ℕ)
  (right : List ℕ)

/-- Calculates the measurable weight for a given configuration -/
def measurable_weight (config : WeightConfiguration) : ℤ :=
  (config.left.sum : ℤ) - (config.right.sum : ℤ)

/-- Generates all possible weight configurations -/
def all_configurations : List WeightConfiguration :=
  sorry

/-- Calculates all measurable weights -/
def measurable_weights : List ℕ :=
  sorry

/-- The main theorem to prove -/
theorem count_measurable_weights :
  measurable_weights.length = 28 :=
sorry

end NUMINAMATH_CALUDE_count_measurable_weights_l2161_216130


namespace NUMINAMATH_CALUDE_geometric_sum_first_seven_l2161_216174

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_seven :
  geometric_sum (1/4) (1/4) 7 = 16383/49152 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_first_seven_l2161_216174


namespace NUMINAMATH_CALUDE_john_journey_distance_l2161_216147

/-- Calculates the total distance traveled given two journey segments -/
def total_distance (speed1 speed2 : ℝ) (time1 time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem stating that the total distance of John's journey is 240 miles -/
theorem john_journey_distance :
  total_distance 45 50 2 3 = 240 := by
  sorry

end NUMINAMATH_CALUDE_john_journey_distance_l2161_216147


namespace NUMINAMATH_CALUDE_g_five_l2161_216123

/-- A function satisfying g(x+y) = g(x) + g(y) for all real x and y, and g(1) = 2 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The functional equation property of g -/
axiom g_add (x y : ℝ) : g (x + y) = g x + g y

/-- The value of g at 1 -/
axiom g_one : g 1 = 2

/-- The theorem stating that g(5) = 10 -/
theorem g_five : g 5 = 10 := by sorry

end NUMINAMATH_CALUDE_g_five_l2161_216123


namespace NUMINAMATH_CALUDE_mexico_city_car_restriction_l2161_216188

/-- The minimum number of cars needed for a family in Mexico City -/
def min_cars : ℕ := 14

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of restricted days per car per week -/
def restricted_days_per_car : ℕ := 2

/-- The minimum number of cars that must be available each day -/
def min_available_cars : ℕ := 10

theorem mexico_city_car_restriction :
  ∀ n : ℕ,
  n ≥ min_cars →
  (∀ d : ℕ, d < days_in_week →
    n - (n * restricted_days_per_car / days_in_week) ≥ min_available_cars) ∧
  (∀ m : ℕ, m < min_cars →
    ∃ d : ℕ, d < days_in_week ∧
      m - (m * restricted_days_per_car / days_in_week) < min_available_cars) :=
by sorry


end NUMINAMATH_CALUDE_mexico_city_car_restriction_l2161_216188


namespace NUMINAMATH_CALUDE_calculate_expression_l2161_216126

theorem calculate_expression : (8 * 2.25 - 5 * 0.85 / 2.5) = 16.3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2161_216126


namespace NUMINAMATH_CALUDE_max_ab_value_l2161_216176

/-- Given a function f(x) = -a * ln(x) + (a+1)x - (1/2)x^2 where a > 0,
    if f(x) ≥ -(1/2)x^2 + ax + b holds for all x > 0,
    then the maximum value of ab is e/2 -/
theorem max_ab_value (a b : ℝ) (h_a : a > 0) :
  (∀ x > 0, -a * Real.log x + (a + 1) * x - (1/2) * x^2 ≥ -(1/2) * x^2 + a * x + b) →
  (∃ m : ℝ, m = Real.exp 1 / 2 ∧ a * b ≤ m ∧ ∀ c d : ℝ, c > 0 → (∀ x > 0, -c * Real.log x + (c + 1) * x - (1/2) * x^2 ≥ -(1/2) * x^2 + c * x + d) → c * d ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l2161_216176


namespace NUMINAMATH_CALUDE_sheep_in_wilderness_l2161_216125

/-- Given that 90% of sheep are in a pen and there are 81 sheep in the pen,
    prove that there are 9 sheep in the wilderness. -/
theorem sheep_in_wilderness (total : ℕ) (in_pen : ℕ) (h1 : in_pen = 81) 
    (h2 : in_pen = (90 : ℕ) * total / 100) : total - in_pen = 9 := by
  sorry

end NUMINAMATH_CALUDE_sheep_in_wilderness_l2161_216125


namespace NUMINAMATH_CALUDE_cindys_calculation_l2161_216193

theorem cindys_calculation (x : ℝ) (h : (x - 5) / 7 = 15) : (x - 7) / 5 = 20.6 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l2161_216193


namespace NUMINAMATH_CALUDE_evaluate_expression_l2161_216113

theorem evaluate_expression : 5^4 + 5^4 + 5^4 - 5^4 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2161_216113


namespace NUMINAMATH_CALUDE_rain_on_tuesday_l2161_216190

theorem rain_on_tuesday (rain_monday : ℝ) (rain_both : ℝ) (no_rain : ℝ) 
  (h1 : rain_monday = 0.6)
  (h2 : rain_both = 0.4)
  (h3 : no_rain = 0.25) :
  ∃ rain_tuesday : ℝ, rain_tuesday = 0.55 ∧ 
  rain_monday + rain_tuesday - rain_both + no_rain = 1 :=
by sorry

end NUMINAMATH_CALUDE_rain_on_tuesday_l2161_216190


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2161_216141

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 2 = 1) ∧
  (n % 3 ≠ 0) ∧
  (n % 4 = 3) ∧
  (n % 10 = 9) ∧
  (∀ m : ℕ, m > 0 → m % 2 = 1 → m % 3 ≠ 0 → m % 4 = 3 → m % 10 = 9 → m ≥ n) ∧
  n = 59 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2161_216141


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2161_216183

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 5*x^4 + 8*x^2 - 4 = (x-1)*(x+1)*(x^2-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2161_216183


namespace NUMINAMATH_CALUDE_planes_perpendicular_l2161_216112

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (l m : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : parallel l m)
  (h3 : perpendicular m β) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l2161_216112


namespace NUMINAMATH_CALUDE_total_fruits_in_garden_l2161_216167

def papaya_production : List Nat := [10, 12]
def mango_production : List Nat := [18, 20, 22]
def apple_production : List Nat := [14, 15, 16, 17]
def orange_production : List Nat := [20, 23, 25, 27, 30]

theorem total_fruits_in_garden : 
  (papaya_production.sum + mango_production.sum + 
   apple_production.sum + orange_production.sum) = 269 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_in_garden_l2161_216167


namespace NUMINAMATH_CALUDE_baseball_card_value_l2161_216150

def initialValue : ℝ := 100

def yearlyChanges : List ℝ := [-0.10, 0.12, -0.08, 0.05, -0.07]

def applyChange (value : ℝ) (change : ℝ) : ℝ := value * (1 + change)

def finalValue : ℝ := yearlyChanges.foldl applyChange initialValue

theorem baseball_card_value : 
  ∃ ε > 0, |finalValue - 90.56| < ε :=
sorry

end NUMINAMATH_CALUDE_baseball_card_value_l2161_216150


namespace NUMINAMATH_CALUDE_audrey_sleep_time_l2161_216132

/-- Given that Audrey dreamed for 2/5 of her sleep time and was not dreaming for 6 hours,
    prove that she was asleep for 10 hours. -/
theorem audrey_sleep_time :
  ∀ (total_sleep : ℝ),
  (2 / 5 : ℝ) * total_sleep + 6 = total_sleep →
  total_sleep = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_audrey_sleep_time_l2161_216132


namespace NUMINAMATH_CALUDE_spider_web_paths_l2161_216191

theorem spider_web_paths : Nat.choose 11 5 = 462 := by
  sorry

end NUMINAMATH_CALUDE_spider_web_paths_l2161_216191


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l2161_216199

theorem power_mod_seventeen : 4^2023 % 17 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l2161_216199


namespace NUMINAMATH_CALUDE_infinite_equal_pairs_l2161_216137

-- Define the sequence type
def InfiniteSequence := ℤ → ℝ

-- Define the property that each term is 1/4 of the sum of its neighbors
def NeighborSumProperty (a : InfiniteSequence) :=
  ∀ n : ℤ, a n = (1 / 4) * (a (n - 1) + a (n + 1))

-- Define the existence of two equal terms
def HasEqualTerms (a : InfiniteSequence) :=
  ∃ i j : ℤ, i ≠ j ∧ a i = a j

-- Define the existence of infinitely many pairs of equal terms
def HasInfiniteEqualPairs (a : InfiniteSequence) :=
  ∀ N : ℕ, ∃ i j : ℤ, i ≠ j ∧ |i - j| > N ∧ a i = a j

-- The main theorem
theorem infinite_equal_pairs
  (a : InfiniteSequence)
  (h1 : NeighborSumProperty a)
  (h2 : HasEqualTerms a) :
  HasInfiniteEqualPairs a :=
sorry

end NUMINAMATH_CALUDE_infinite_equal_pairs_l2161_216137


namespace NUMINAMATH_CALUDE_cube_surface_area_with_holes_eq_222_l2161_216138

/-- Calculates the entire surface area of a cube with holes, including inside surfaces -/
def cubeSurfaceAreaWithHoles (cubeEdge : ℝ) (holeEdge : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let newExposedArea := 6 * 4 * holeEdge^2
  originalSurface - holeArea + newExposedArea

/-- The entire surface area of the cube with holes is 222 square meters -/
theorem cube_surface_area_with_holes_eq_222 :
  cubeSurfaceAreaWithHoles 5 2 = 222 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_with_holes_eq_222_l2161_216138


namespace NUMINAMATH_CALUDE_solution_system_equations_l2161_216189

theorem solution_system_equations (a : ℝ) (ha : a ≠ 0) :
  let x₁ := a / 2 + (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let y₁ := a / 2 - (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let x₂ := a / 2 - (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  let y₂ := a / 2 + (a / 10) * Real.sqrt (30 * Real.sqrt 5 - 25)
  (x₁ + y₁ = a ∧ x₁^5 + y₁^5 = 2 * a^5) ∧
  (x₂ + y₂ = a ∧ x₂^5 + y₂^5 = 2 * a^5) ∧
  (∀ x y : ℝ, x + y = a ∧ x^5 + y^5 = 2 * a^5 → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_solution_system_equations_l2161_216189


namespace NUMINAMATH_CALUDE_problem_solution_l2161_216118

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

theorem problem_solution :
  (∀ x : ℝ, f 2 x + g x ≤ 7 ↔ -1/2 ≤ x ∧ x ≤ 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) ↔ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2161_216118


namespace NUMINAMATH_CALUDE_car_speed_problem_l2161_216163

/-- Given two cars traveling in opposite directions, prove that the speed of one car is 52 mph -/
theorem car_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed
  3.5 * v + 3.5 * 58 = 385 → 
  v = 52 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l2161_216163


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l2161_216185

/-- Proves that the ratio of the rate down to the rate up is 1.5 given the hiking conditions -/
theorem hiking_rate_ratio 
  (time_equal : ℝ) -- The time for both routes is the same
  (rate_up : ℝ) -- The rate up the mountain
  (time_up : ℝ) -- The time to go up the mountain
  (distance_down : ℝ) -- The distance of the route down the mountain
  (h_rate_up : rate_up = 5) -- The rate up is 5 miles per day
  (h_time_up : time_up = 2) -- It takes 2 days to go up
  (h_distance_down : distance_down = 15) -- The route down is 15 miles long
  : (distance_down / time_equal) / rate_up = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_hiking_rate_ratio_l2161_216185


namespace NUMINAMATH_CALUDE_f_minimum_value_l2161_216178

noncomputable def f (x : ℝ) : ℝ :=
  x + (2*x)/(x^2 + 1) + (x*(x + 5))/(x^2 + 3) + (3*(x + 3))/(x*(x^2 + 3))

theorem f_minimum_value (x : ℝ) (h : x > 0) : f x ≥ 5.5 := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_value_l2161_216178


namespace NUMINAMATH_CALUDE_complement_M_in_U_l2161_216134

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the set M
def M : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_M_in_U : 
  (U \ M) = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l2161_216134


namespace NUMINAMATH_CALUDE_no_two_digit_product_equals_concatenation_l2161_216169

theorem no_two_digit_product_equals_concatenation : ¬∃ (a b c d : ℕ), 
  (0 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (0 ≤ d ∧ d ≤ 9) ∧ 
  ((10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d) :=
by sorry

end NUMINAMATH_CALUDE_no_two_digit_product_equals_concatenation_l2161_216169


namespace NUMINAMATH_CALUDE_computer_price_2004_l2161_216162

/-- The yearly decrease rate of the computer price -/
def yearly_decrease_rate : ℚ := 1 / 3

/-- The initial price of the computer in 2000 -/
def initial_price : ℚ := 8100

/-- The number of years between 2000 and 2004 -/
def years : ℕ := 4

/-- The price of the computer in 2004 -/
def price_2004 : ℚ := initial_price * (1 - yearly_decrease_rate) ^ years

theorem computer_price_2004 : price_2004 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_2004_l2161_216162


namespace NUMINAMATH_CALUDE_extremum_values_l2161_216187

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_values (a b : ℝ) :
  f_deriv a b 1 = 0 ∧ f a b 1 = 10 → a = 4 ∧ b = -11 := by
  sorry

end NUMINAMATH_CALUDE_extremum_values_l2161_216187


namespace NUMINAMATH_CALUDE_parabola_vertex_l2161_216155

/-- A parabola is defined by the equation y = (x - 2)^2 - 1 -/
def parabola (x y : ℝ) : Prop := y = (x - 2)^2 - 1

/-- The vertex of a parabola is the point where it reaches its minimum or maximum -/
def is_vertex (x y : ℝ) : Prop :=
  parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex 2 (-1) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2161_216155


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l2161_216154

theorem sum_of_x_and_y (x y : ℝ) (h1 : y - x = 1) (h2 : y^2 = x^2 + 6) : x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l2161_216154


namespace NUMINAMATH_CALUDE_james_initial_balance_l2161_216100

def ticket_cost_1 : ℚ := 150
def ticket_cost_2 : ℚ := 150
def ticket_cost_3 : ℚ := ticket_cost_1 / 3
def total_cost : ℚ := ticket_cost_1 + ticket_cost_2 + ticket_cost_3
def james_share : ℚ := total_cost / 2
def remaining_balance : ℚ := 325

theorem james_initial_balance :
  ∀ x : ℚ, x - james_share = remaining_balance → x = 500 :=
by sorry

end NUMINAMATH_CALUDE_james_initial_balance_l2161_216100


namespace NUMINAMATH_CALUDE_complex_power_sum_l2161_216153

open Complex

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^1000 + 1 / z^1000 = 2 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2161_216153


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_12_l2161_216171

theorem cos_2alpha_plus_pi_12 (α : Real) (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : Real.sin (α - π/8) = Real.sqrt 3 / 3) : 
  Real.cos (2*α + π/12) = (1 - 2*Real.sqrt 6) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_12_l2161_216171


namespace NUMINAMATH_CALUDE_percentage_calculation_l2161_216135

theorem percentage_calculation (x : ℝ) : 
  (70 / 100 * 600 : ℝ) = (x / 100 * 1050 : ℝ) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2161_216135


namespace NUMINAMATH_CALUDE_complex_power_sum_l2161_216122

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^12 + 1 / z^12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2161_216122


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l2161_216164

theorem sin_plus_two_cos_alpha (α : Real) :
  (∃ P : Real × Real, P.1 = 3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.sin α + 2 * Real.cos α = 2 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l2161_216164


namespace NUMINAMATH_CALUDE_factorial_ratio_l2161_216159

theorem factorial_ratio : Nat.factorial 15 / (Nat.factorial 6 * Nat.factorial 9) = 5005 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l2161_216159


namespace NUMINAMATH_CALUDE_opposite_of_miss_both_is_hit_at_least_once_l2161_216127

-- Define the sample space
def Ω : Type := Unit

-- Define the event of missing the target on both shots
def miss_both : Set Ω := sorry

-- Define the event of hitting the target at least once
def hit_at_least_once : Set Ω := sorry

-- Theorem stating that the complement of missing both shots is hitting at least once
theorem opposite_of_miss_both_is_hit_at_least_once : 
  (miss_both)ᶜ = hit_at_least_once := by sorry

end NUMINAMATH_CALUDE_opposite_of_miss_both_is_hit_at_least_once_l2161_216127


namespace NUMINAMATH_CALUDE_polynomial_sequence_problem_l2161_216157

theorem polynomial_sequence_problem (a b x y : ℝ) 
  (eq1 : a * x + b * y = 3)
  (eq2 : a * x^2 + b * y^2 = 7)
  (eq3 : a * x^3 + b * y^3 = 16)
  (eq4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sequence_problem_l2161_216157


namespace NUMINAMATH_CALUDE_fourth_number_in_proportion_l2161_216173

-- Define the proportion
def proportion (a b c d : ℝ) : Prop := a / b = c / d

-- State the theorem
theorem fourth_number_in_proportion : 
  proportion 0.75 1.35 5 9 := by sorry

end NUMINAMATH_CALUDE_fourth_number_in_proportion_l2161_216173


namespace NUMINAMATH_CALUDE_max_integer_difference_l2161_216194

theorem max_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  ∃ (n : ℤ), n = ⌊y - x⌋ ∧ n ≤ 4 ∧ ∀ (m : ℤ), m = ⌊y - x⌋ → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l2161_216194


namespace NUMINAMATH_CALUDE_average_difference_theorem_l2161_216139

/-- Represents the enrollment of a class -/
structure ClassEnrollment where
  students : ℕ

/-- Represents a school with students, teachers, and class enrollments -/
structure School where
  totalStudents : ℕ
  totalTeachers : ℕ
  classEnrollments : List ClassEnrollment

/-- Calculates the average number of students per teacher -/
def averageStudentsPerTeacher (school : School) : ℚ :=
  school.totalStudents / school.totalTeachers

/-- Calculates the average number of students per student -/
def averageStudentsPerStudent (school : School) : ℚ :=
  (school.classEnrollments.map (λ c => c.students * c.students)).sum / school.totalStudents

/-- The main theorem to prove -/
theorem average_difference_theorem (school : School) 
  (h1 : school.totalStudents = 120)
  (h2 : school.totalTeachers = 6)
  (h3 : school.classEnrollments = [⟨60⟩, ⟨30⟩, ⟨20⟩, ⟨5⟩, ⟨3⟩, ⟨2⟩])
  (h4 : (school.classEnrollments.map (λ c => c.students)).sum = school.totalStudents) :
  averageStudentsPerTeacher school - averageStudentsPerStudent school = -21 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_theorem_l2161_216139
