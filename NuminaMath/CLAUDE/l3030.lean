import Mathlib

namespace fair_coin_probability_difference_l3030_303009

theorem fair_coin_probability_difference : 
  let n : ℕ := 5
  let p : ℚ := 1/2
  let prob_4_heads := (n.choose 4) * p^4 * (1-p)
  let prob_5_heads := p^n
  abs (prob_4_heads - prob_5_heads) = 9/32 := by
sorry

end fair_coin_probability_difference_l3030_303009


namespace division_with_remainder_l3030_303044

theorem division_with_remainder : ∃ (q r : ℤ), 1234567 = 127 * q + r ∧ 0 ≤ r ∧ r < 127 ∧ r = 51 := by
  sorry

end division_with_remainder_l3030_303044


namespace rational_function_zeros_l3030_303086

theorem rational_function_zeros (x : ℝ) : 
  (x^2 - 5*x + 6) / (3*x - 1) = 0 ↔ x = 2 ∨ x = 3 := by
  sorry

end rational_function_zeros_l3030_303086


namespace branch_A_more_profitable_l3030_303037

/-- Represents the grades of products -/
inductive Grade
| A
| B
| C
| D

/-- Represents a branch of the factory -/
structure Branch where
  name : String
  processingCost : ℝ
  gradeDistribution : Grade → ℝ

/-- Calculates the processing fee for a given grade -/
def processingFee (g : Grade) : ℝ :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Calculates the average profit per 100 products for a given branch -/
def averageProfit (b : Branch) : ℝ :=
  (processingFee Grade.A - b.processingCost) * b.gradeDistribution Grade.A +
  (processingFee Grade.B - b.processingCost) * b.gradeDistribution Grade.B +
  (processingFee Grade.C - b.processingCost) * b.gradeDistribution Grade.C +
  (processingFee Grade.D - b.processingCost) * b.gradeDistribution Grade.D

/-- Branch A of the factory -/
def branchA : Branch :=
  { name := "A"
    processingCost := 25
    gradeDistribution := fun g => match g with
      | Grade.A => 0.4
      | Grade.B => 0.2
      | Grade.C => 0.2
      | Grade.D => 0.2 }

/-- Branch B of the factory -/
def branchB : Branch :=
  { name := "B"
    processingCost := 20
    gradeDistribution := fun g => match g with
      | Grade.A => 0.28
      | Grade.B => 0.17
      | Grade.C => 0.34
      | Grade.D => 0.21 }

theorem branch_A_more_profitable :
  averageProfit branchA > averageProfit branchB :=
sorry

end branch_A_more_profitable_l3030_303037


namespace tangent_line_sum_l3030_303092

/-- Given a function f: ℝ → ℝ whose graph is tangent to the line 2x+y-1=0 at the point (1,f(1)),
    prove that f(1) + f'(1) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, 2*x + f x - 1 = 0 ↔ x = 1) : 
    f 1 + deriv f 1 = -3 := by
  sorry

end tangent_line_sum_l3030_303092


namespace imaginary_part_of_two_over_one_plus_i_l3030_303097

theorem imaginary_part_of_two_over_one_plus_i :
  Complex.im (2 / (1 + Complex.I)) = -1 := by
  sorry

end imaginary_part_of_two_over_one_plus_i_l3030_303097


namespace first_week_rate_is_18_l3030_303065

/-- The daily rate for the first week in a student youth hostel -/
def first_week_rate : ℝ := 18

/-- The daily rate for additional weeks in a student youth hostel -/
def additional_week_rate : ℝ := 14

/-- The total number of days stayed -/
def total_days : ℕ := 23

/-- The total cost for the stay -/
def total_cost : ℝ := 350

/-- Theorem stating that the daily rate for the first week is $18.00 -/
theorem first_week_rate_is_18 :
  first_week_rate * 7 + additional_week_rate * (total_days - 7) = total_cost :=
by sorry

end first_week_rate_is_18_l3030_303065


namespace arithmetic_geometric_sequence_l3030_303029

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- a_1, a_3, and a_4 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 3 / a 1) ^ 2 = a 4 / a 1

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_subseq a) : 
  a 2 = -6 := by
  sorry

end arithmetic_geometric_sequence_l3030_303029


namespace min_perimeter_rectangle_l3030_303019

theorem min_perimeter_rectangle (area : Real) (perimeter : Real) : 
  area = 64 → perimeter ≥ 32 := by
  sorry

end min_perimeter_rectangle_l3030_303019


namespace relative_rate_of_change_cubic_parabola_l3030_303013

/-- For a point (x, y) on the cubic parabola 12y = x^3, the relative rate of change between y and x is x^2/4 -/
theorem relative_rate_of_change_cubic_parabola (x y : ℝ) (h : 12 * y = x^3) :
  ∃ (dx dy : ℝ), dy / dx = x^2 / 4 := by
  sorry

end relative_rate_of_change_cubic_parabola_l3030_303013


namespace garden_area_l3030_303053

theorem garden_area (width length perimeter area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  area = width * length →
  area = 243 := by
sorry

end garden_area_l3030_303053


namespace seven_double_prime_l3030_303004

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Theorem statement
theorem seven_double_prime : prime (prime 7) = 51 := by
  sorry

end seven_double_prime_l3030_303004


namespace midpoint_sum_invariant_l3030_303011

/-- A polygon in the Cartesian plane -/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Create a new polygon by connecting midpoints of sides -/
def midpointPolygon (P : Polygon) : Polygon :=
  sorry

/-- Sum of x-coordinates of vertices -/
def sumXCoordinates (P : Polygon) : ℝ :=
  sorry

theorem midpoint_sum_invariant (Q1 : Polygon) 
  (h1 : Q1.vertices.length = 45)
  (h2 : sumXCoordinates Q1 = 135) : 
  let Q2 := midpointPolygon Q1
  let Q3 := midpointPolygon Q2
  sumXCoordinates Q3 = 135 := by
  sorry

end midpoint_sum_invariant_l3030_303011


namespace trackball_mice_count_l3030_303077

theorem trackball_mice_count (total : ℕ) (wireless : ℕ) (optical : ℕ) (trackball : ℕ) : 
  total = 80 →
  wireless = total / 2 →
  optical = total / 4 →
  trackball = total - (wireless + optical) →
  trackball = 20 := by
sorry

end trackball_mice_count_l3030_303077


namespace dave_guitar_strings_l3030_303026

theorem dave_guitar_strings 
  (strings_per_night : ℕ) 
  (shows_per_week : ℕ) 
  (total_weeks : ℕ) 
  (h1 : strings_per_night = 2) 
  (h2 : shows_per_week = 6) 
  (h3 : total_weeks = 12) : 
  strings_per_night * shows_per_week * total_weeks = 144 := by
sorry

end dave_guitar_strings_l3030_303026


namespace set_four_subsets_implies_a_not_zero_or_two_l3030_303095

theorem set_four_subsets_implies_a_not_zero_or_two (a : ℝ) : 
  (Finset.powerset {a, a^2 - a}).card = 4 → a ≠ 0 ∧ a ≠ 2 :=
sorry

end set_four_subsets_implies_a_not_zero_or_two_l3030_303095


namespace some_number_equation_l3030_303081

theorem some_number_equation (n : ℤ) (y : ℤ) : 
  (n * (1 + y) + 17 = n * (-1 + y) - 21) → n = -19 := by
  sorry

end some_number_equation_l3030_303081


namespace difference_of_squares_value_l3030_303062

theorem difference_of_squares_value (x y : ℤ) (hx : x = -5) (hy : y = -10) :
  (y - x) * (y + x) = 75 := by
sorry

end difference_of_squares_value_l3030_303062


namespace diophantine_equation_solutions_l3030_303073

theorem diophantine_equation_solutions :
  ∀ a b c : ℕ+,
  a * b + b * c + c * a = 2 * (a + b + c) ↔
  ((a = 2 ∧ b = 2 ∧ c = 2) ∨
   (a = 1 ∧ b = 2 ∧ c = 4) ∨ (a = 1 ∧ b = 4 ∧ c = 2) ∨
   (a = 2 ∧ b = 1 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 1) ∨
   (a = 4 ∧ b = 1 ∧ c = 2) ∨ (a = 4 ∧ b = 2 ∧ c = 1)) :=
by sorry

end diophantine_equation_solutions_l3030_303073


namespace no_solution_l3030_303079

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

theorem no_solution : ¬∃ x : ℕ, (137 + x = 435) ∧ (reverse_digits x = 672) := by
  sorry

end no_solution_l3030_303079


namespace ribbon_length_proof_l3030_303094

theorem ribbon_length_proof (length1 length2 : ℕ) : 
  length1 = 8 →
  (∃ (piece_length : ℕ), piece_length > 0 ∧ 
    length1 % piece_length = 0 ∧ 
    length2 % piece_length = 0 ∧
    ∀ (l : ℕ), l > piece_length → (length1 % l ≠ 0 ∨ length2 % l ≠ 0)) →
  length2 = 8 := by
sorry

end ribbon_length_proof_l3030_303094


namespace second_player_wins_12_and_11_l3030_303083

/-- Represents the state of the daisy game -/
inductive DaisyState
  | petals (n : Nat)

/-- Represents a move in the daisy game -/
inductive DaisyMove
  | remove_one
  | remove_two

/-- Defines a valid move in the daisy game -/
def valid_move (state : DaisyState) (move : DaisyMove) : Prop :=
  match state, move with
  | DaisyState.petals n, DaisyMove.remove_one => n ≥ 1
  | DaisyState.petals n, DaisyMove.remove_two => n ≥ 2

/-- Applies a move to the current state -/
def apply_move (state : DaisyState) (move : DaisyMove) : DaisyState :=
  match state, move with
  | DaisyState.petals n, DaisyMove.remove_one => DaisyState.petals (n - 1)
  | DaisyState.petals n, DaisyMove.remove_two => DaisyState.petals (n - 2)

/-- Defines a winning strategy for the second player -/
def second_player_wins (initial_petals : Nat) : Prop :=
  ∀ (first_move : DaisyMove),
    valid_move (DaisyState.petals initial_petals) first_move →
    ∃ (strategy : DaisyState → DaisyMove),
      (∀ (state : DaisyState), valid_move state (strategy state)) ∧
      (∀ (game : Nat → DaisyState),
        game 0 = apply_move (DaisyState.petals initial_petals) first_move →
        (∀ n, game (n + 1) = apply_move (game n) (strategy (game n))) →
        ∃ k, ¬∃ move, valid_move (game k) move)

/-- The main theorem stating that the second player wins for both 12 and 11 initial petals -/
theorem second_player_wins_12_and_11 :
  second_player_wins 12 ∧ second_player_wins 11 := by sorry

end second_player_wins_12_and_11_l3030_303083


namespace curvilinearTrapezoidAreaStepsCorrect_l3030_303045

/-- The steps required to calculate the area of a curvilinear trapezoid. -/
inductive CurvilinearTrapezoidAreaStep
  | division
  | approximation
  | summation
  | takingLimit

/-- The list of steps to calculate the area of a curvilinear trapezoid. -/
def curvilinearTrapezoidAreaSteps : List CurvilinearTrapezoidAreaStep :=
  [CurvilinearTrapezoidAreaStep.division,
   CurvilinearTrapezoidAreaStep.approximation,
   CurvilinearTrapezoidAreaStep.summation,
   CurvilinearTrapezoidAreaStep.takingLimit]

/-- Theorem stating that the steps to calculate the area of a curvilinear trapezoid
    are division, approximation, summation, and taking the limit. -/
theorem curvilinearTrapezoidAreaStepsCorrect :
  curvilinearTrapezoidAreaSteps =
    [CurvilinearTrapezoidAreaStep.division,
     CurvilinearTrapezoidAreaStep.approximation,
     CurvilinearTrapezoidAreaStep.summation,
     CurvilinearTrapezoidAreaStep.takingLimit] := by
  sorry

end curvilinearTrapezoidAreaStepsCorrect_l3030_303045


namespace magnified_diameter_calculation_l3030_303099

/-- Given a circular piece of tissue with an actual diameter and a magnification factor,
    calculate the diameter of the magnified image. -/
theorem magnified_diameter_calculation
  (actual_diameter : ℝ)
  (magnification_factor : ℝ)
  (h1 : actual_diameter = 0.0002)
  (h2 : magnification_factor = 1000) :
  actual_diameter * magnification_factor = 0.2 := by
sorry

end magnified_diameter_calculation_l3030_303099


namespace square_side_equations_l3030_303006

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero : a ≠ 0 ∨ b ≠ 0

/-- Represents a square in 2D space --/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ
  parallel_line : Line

/-- Check if two lines are parallel --/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are perpendicular --/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem --/
theorem square_side_equations (s : Square)
  (h1 : s.center = (-3, -4))
  (h2 : s.side_length = 2 * Real.sqrt 5)
  (h3 : s.parallel_line = ⟨2, 1, 3, Or.inl (by norm_num)⟩) :
  ∃ (l1 l2 l3 l4 : Line),
    (l1 = ⟨2, 1, 15, Or.inl (by norm_num)⟩) ∧
    (l2 = ⟨2, 1, 5, Or.inl (by norm_num)⟩) ∧
    (l3 = ⟨1, -2, 0, Or.inr (by norm_num)⟩) ∧
    (l4 = ⟨1, -2, -10, Or.inr (by norm_num)⟩) ∧
    are_parallel l1 s.parallel_line ∧
    are_parallel l2 s.parallel_line ∧
    are_perpendicular l1 l3 ∧
    are_perpendicular l1 l4 ∧
    are_perpendicular l2 l3 ∧
    are_perpendicular l2 l4 :=
  sorry

end square_side_equations_l3030_303006


namespace negation_equivalence_l3030_303080

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end negation_equivalence_l3030_303080


namespace quadratic_function_k_value_l3030_303090

theorem quadratic_function_k_value (a b c k : ℤ) (f : ℝ → ℝ) :
  (∀ x, f x = a * x^2 + b * x + c) →
  f 1 = 0 →
  50 < f 7 ∧ f 7 < 60 →
  70 < f 8 ∧ f 8 < 80 →
  5000 * k < f 100 ∧ f 100 < 5000 * (k + 1) →
  k = 3 := by
sorry


end quadratic_function_k_value_l3030_303090


namespace fraction_equality_l3030_303063

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hyx : y - x^2 ≠ 0) :
  (x^2 - 1/y) / (y - x^2) = (x^2 * y - 1) / (y^2 - x^2 * y) := by
  sorry

end fraction_equality_l3030_303063


namespace parabola_equation_l3030_303046

/-- Given a parabola and a line intersecting it, prove the equation of the parabola. -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y = Real.sqrt 3 * x + (A.2 - Real.sqrt 3 * A.1)) →  -- Line equation
  (∀ x y, x^2 = 2 * p * y) →  -- Parabola equation
  A.1^2 = 2 * p * A.2 →  -- Point A satisfies parabola equation
  B.1^2 = 2 * p * B.2 →  -- Point B satisfies parabola equation
  A.2 = Real.sqrt 3 * A.1 + (A.2 - Real.sqrt 3 * A.1) →  -- Point A satisfies line equation
  B.2 = Real.sqrt 3 * B.1 + (A.2 - Real.sqrt 3 * A.1) →  -- Point B satisfies line equation
  A.1 + B.1 = 3 →  -- Sum of x-coordinates
  (∀ x y, x^2 = Real.sqrt 3 * y) :=  -- Conclusion: equation of the parabola
by sorry

end parabola_equation_l3030_303046


namespace be_length_l3030_303060

structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

def is_right_angle (p q r : ℝ × ℝ) : Prop := sorry

def on_line (p q r : ℝ × ℝ) : Prop := sorry

def perpendicular (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem be_length 
  (ABCD : Quadrilateral)
  (E F : ℝ × ℝ)
  (h1 : is_right_angle ABCD.A ABCD.B ABCD.C)
  (h2 : is_right_angle ABCD.B ABCD.C ABCD.D)
  (h3 : on_line ABCD.A E ABCD.C)
  (h4 : on_line ABCD.A F ABCD.C)
  (h5 : perpendicular (ABCD.D, F) (ABCD.A, ABCD.C))
  (h6 : perpendicular (ABCD.B, E) (ABCD.A, ABCD.C))
  (h7 : distance ABCD.A F = 4)
  (h8 : distance ABCD.D F = 6)
  (h9 : distance ABCD.C F = 8)
  : distance ABCD.B E = 16/3 := sorry

end be_length_l3030_303060


namespace cloth_sold_l3030_303032

/-- Proves the number of meters of cloth sold by a shopkeeper -/
theorem cloth_sold (total_price : ℝ) (loss_per_meter : ℝ) (cost_price : ℝ) :
  total_price = 18000 ∧ loss_per_meter = 5 ∧ cost_price = 50 →
  (total_price / (cost_price - loss_per_meter) : ℝ) = 400 := by
  sorry

end cloth_sold_l3030_303032


namespace initial_money_calculation_l3030_303031

theorem initial_money_calculation (X : ℝ) : 
  X - (X / 2 + 50) = 25 → X = 150 := by
  sorry

end initial_money_calculation_l3030_303031


namespace complex_fraction_equality_l3030_303008

/-- Given that i is the imaginary unit, prove that (1+2i)/(1+i) = (3+i)/2 -/
theorem complex_fraction_equality : (1 + 2 * Complex.I) / (1 + Complex.I) = (3 + Complex.I) / 2 := by
  sorry

end complex_fraction_equality_l3030_303008


namespace rational_term_count_is_seventeen_l3030_303036

/-- The number of terms with rational coefficients in the expansion of (√3x + ∛2)^100 -/
def rationalTermCount : ℕ := 17

/-- The exponent in the binomial expansion -/
def exponent : ℕ := 100

/-- Predicate to check if a number is a multiple of 2 -/
def isMultipleOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Predicate to check if a number is a multiple of 3 -/
def isMultipleOfThree (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

/-- Theorem stating that the number of terms with rational coefficients in the expansion of (√3x + ∛2)^100 is 17 -/
theorem rational_term_count_is_seventeen :
  (∀ r : ℕ, r ≤ exponent →
    (isMultipleOfTwo (exponent - r) ∧ isMultipleOfThree r) ↔
    (∃ n : ℕ, r = 6 * n ∧ n ≤ 16)) ∧
  rationalTermCount = 17 := by sorry

end rational_term_count_is_seventeen_l3030_303036


namespace exponent_equation_l3030_303096

theorem exponent_equation (a b : ℤ) : 3^a * 9^b = (1:ℚ)/3 → a + 2*b = -1 := by
  sorry

end exponent_equation_l3030_303096


namespace age_difference_proof_l3030_303010

def zion_age : ℕ := 8

def dad_age : ℕ := 4 * zion_age + 3

def age_difference_after_10_years : ℕ :=
  (dad_age + 10) - (zion_age + 10)

theorem age_difference_proof :
  age_difference_after_10_years = 27 := by
  sorry

end age_difference_proof_l3030_303010


namespace base5_to_decimal_conversion_l3030_303021

/-- Converts a base-5 digit to its decimal (base-10) value -/
def base5ToDecimal (digit : Nat) : Nat :=
  digit

/-- Converts a base-5 number to its decimal (base-10) equivalent -/
def convertBase5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + (base5ToDecimal d) * (5 ^ i)) 0

/-- The base-5 representation of the number -/
def base5Number : List Nat := [2, 1, 4, 3, 2]

theorem base5_to_decimal_conversion :
  convertBase5ToDecimal base5Number = 1732 := by
  sorry

end base5_to_decimal_conversion_l3030_303021


namespace advanced_tablet_price_relationship_l3030_303076

/-- The price of a smartphone in dollars. -/
def smartphone_price : ℕ := 300

/-- The price difference between a personal computer and a smartphone in dollars. -/
def pc_price_difference : ℕ := 500

/-- The total cost of buying one of each product (smartphone, personal computer, and advanced tablet) in dollars. -/
def total_cost : ℕ := 2200

/-- The price of a personal computer in dollars. -/
def pc_price : ℕ := smartphone_price + pc_price_difference

/-- The price of an advanced tablet in dollars. -/
def advanced_tablet_price : ℕ := total_cost - (smartphone_price + pc_price)

theorem advanced_tablet_price_relationship :
  advanced_tablet_price = smartphone_price + pc_price - 400 := by
  sorry

end advanced_tablet_price_relationship_l3030_303076


namespace tan_alpha_plus_pi_fourth_l3030_303061

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : Real.cos (2 * α) + Real.cos α ^ 2 = 0) : 
  Real.tan (α + Real.pi / 4) = -3 - 2 * Real.sqrt 2 := by
  sorry

end tan_alpha_plus_pi_fourth_l3030_303061


namespace proposition_1_proposition_3_l3030_303052

-- Proposition ①
theorem proposition_1 : ∀ a b : ℝ, (a + b ≠ 5) → (a ≠ 2 ∨ b ≠ 3) := by sorry

-- Proposition ③
theorem proposition_3 : 
  (∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ∧ 
  (∀ ε > 0, ∃ x : ℝ, x > 0 ∧ x + 1/x < 2 + ε) := by sorry

end proposition_1_proposition_3_l3030_303052


namespace fish_ratio_l3030_303087

theorem fish_ratio (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) (total : ℕ) :
  bass = 32 →
  trout = bass / 4 →
  total = 104 →
  total = bass + trout + blue_gill →
  blue_gill / bass = 2 :=
by
  sorry

end fish_ratio_l3030_303087


namespace factory_month_days_l3030_303091

/-- The number of days in a month for a computer factory -/
def days_in_month (computers_per_month : ℕ) (computers_per_half_hour : ℕ) : ℕ :=
  computers_per_month * 30 / (computers_per_half_hour * 24 * 2)

/-- Theorem: Given the production rate, the number of days in the month is 28 -/
theorem factory_month_days :
  days_in_month 4032 3 = 28 := by
  sorry

end factory_month_days_l3030_303091


namespace f_g_f_2_equals_120_l3030_303072

def f (x : ℝ) : ℝ := 3 * x + 3

def g (x : ℝ) : ℝ := 4 * x + 3

theorem f_g_f_2_equals_120 : f (g (f 2)) = 120 := by
  sorry

end f_g_f_2_equals_120_l3030_303072


namespace rectangle_area_increase_l3030_303047

theorem rectangle_area_increase (l w : ℝ) (h_l : l > 0) (h_w : w > 0) :
  let new_area := (1.15 * l) * (1.25 * w)
  let orig_area := l * w
  (new_area - orig_area) / orig_area = 0.4375 := by sorry

end rectangle_area_increase_l3030_303047


namespace original_number_proof_l3030_303066

theorem original_number_proof : ∃ (n : ℕ), n + 859560 ≡ 0 [MOD 456] ∧ n = 696 := by
  sorry

end original_number_proof_l3030_303066


namespace simplify_fraction_l3030_303071

theorem simplify_fraction : (111 : ℚ) / 9999 * 33 = 11 / 3 := by sorry

end simplify_fraction_l3030_303071


namespace bike_travel_time_l3030_303017

-- Constants
def highway_length : Real := 5280  -- in feet
def highway_width : Real := 50     -- in feet
def bike_speed : Real := 6         -- in miles per hour

-- Theorem
theorem bike_travel_time :
  let semicircle_radius : Real := highway_width / 2
  let num_semicircles : Real := highway_length / highway_width
  let total_distance : Real := num_semicircles * (π * semicircle_radius)
  let total_distance_miles : Real := total_distance / 5280
  let time_taken : Real := total_distance_miles / bike_speed
  time_taken = π / 12 := by sorry

end bike_travel_time_l3030_303017


namespace quadratic_one_solution_sum_l3030_303034

theorem quadratic_one_solution_sum (b : ℝ) : 
  let equation := fun (x : ℝ) => 3 * x^2 + b * x + 6 * x + 14
  let discriminant := (b + 6)^2 - 4 * 3 * 14
  (∃! x, equation x = 0) → 
  (∃ b₁ b₂, b = b₁ ∨ b = b₂) ∧ (b₁ + b₂ = -12) :=
by sorry

end quadratic_one_solution_sum_l3030_303034


namespace problem_solution_l3030_303043

theorem problem_solution (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 4*x + 4/x + 1/x^2 = 35)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 23 := by
sorry

end problem_solution_l3030_303043


namespace expansion_terms_count_l3030_303039

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^12 -/
def dissimilarTerms : ℕ :=
  Nat.choose 15 3

/-- The number of ways to distribute 12 indistinguishable objects into 4 distinguishable boxes -/
def distributionWays : ℕ :=
  Nat.choose (12 + 4 - 1) (4 - 1)

theorem expansion_terms_count :
  dissimilarTerms = distributionWays ∧ dissimilarTerms = 455 := by
  sorry

end expansion_terms_count_l3030_303039


namespace canoe_production_sum_l3030_303030

theorem canoe_production_sum : 
  let a : ℕ := 8  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  let sum := a * (r^n - 1) / (r - 1)
  sum = 26240 := by sorry

end canoe_production_sum_l3030_303030


namespace choir_members_count_l3030_303018

theorem choir_members_count : ∃! n : ℕ, 
  200 ≤ n ∧ n ≤ 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by sorry

end choir_members_count_l3030_303018


namespace lcm_of_5_6_10_12_l3030_303054

theorem lcm_of_5_6_10_12 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 12)) = 60 := by
  sorry

end lcm_of_5_6_10_12_l3030_303054


namespace tan_two_alpha_plus_pi_l3030_303056

-- Define the angle α
def α : Real := sorry

-- Define the conditions
axiom vertex_at_origin : True
axiom initial_side_on_x_axis : True
axiom terminal_side_on_line : ∀ (x y : Real), y = Real.sqrt 3 * x → (∃ t : Real, t > 0 ∧ x = t * Real.cos α ∧ y = t * Real.sin α)

-- State the theorem
theorem tan_two_alpha_plus_pi : Real.tan (2 * α + Real.pi) = -Real.sqrt 3 := by sorry

end tan_two_alpha_plus_pi_l3030_303056


namespace max_area_rectangle_with_perimeter_40_l3030_303068

/-- The maximum area of a rectangle with a perimeter of 40 units is 100 square units. -/
theorem max_area_rectangle_with_perimeter_40 :
  ∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    2 * (length + width) = 40 ∧
    length * width = 100 ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → 2 * (l + w) = 40 → l * w ≤ 100 := by
  sorry

end max_area_rectangle_with_perimeter_40_l3030_303068


namespace expression_simplification_and_evaluation_l3030_303027

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ 1 →
  (((1 / x - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = 1 / (x * (x - 1))) ∧
  (((1 / 2 - 1 / 3) / ((2^2 - 1) / (2^2 + 2*2 + 1))) = 1 / 2) :=
by sorry

end expression_simplification_and_evaluation_l3030_303027


namespace meaningful_range_l3030_303082

def is_meaningful (x : ℝ) : Prop :=
  x - 1 ≥ 0 ∧ x ≠ 3

theorem meaningful_range : 
  ∀ x : ℝ, is_meaningful x ↔ x ≥ 1 ∧ x ≠ 3 :=
by sorry

end meaningful_range_l3030_303082


namespace max_three_cards_l3030_303051

theorem max_three_cards (total_cards : ℕ) (sum : ℕ) (cards_chosen : ℕ) : 
  total_cards = 10 →
  sum = 31 →
  cards_chosen = 8 →
  ∃ (threes fours fives : ℕ),
    threes + fours + fives = cards_chosen ∧
    3 * threes + 4 * fours + 5 * fives = sum ∧
    threes ≤ 4 ∧
    ∀ (t f v : ℕ), 
      t + f + v = cards_chosen →
      3 * t + 4 * f + 5 * v = sum →
      t ≤ 4 :=
by sorry

end max_three_cards_l3030_303051


namespace necessary_and_sufficient_condition_l3030_303007

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) ↔
  a ≤ -2 ∨ a = 1 := by
  sorry

end necessary_and_sufficient_condition_l3030_303007


namespace polynomial_inequality_roots_l3030_303067

theorem polynomial_inequality_roots (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c*x - 8 < 0 ↔ x < 2 ∨ x > 6) → c = 8 := by
  sorry

end polynomial_inequality_roots_l3030_303067


namespace wall_length_calculation_l3030_303078

theorem wall_length_calculation (mirror_side : ℝ) (wall_width : ℝ) : 
  mirror_side = 21 →
  wall_width = 28 →
  (mirror_side ^ 2) * 2 = wall_width * (31.5 : ℝ) := by
  sorry

end wall_length_calculation_l3030_303078


namespace intersection_M_N_l3030_303075

def M : Set ℝ := {1, 3, 4}
def N : Set ℝ := {x : ℝ | x^2 - 4*x + 3 = 0}

theorem intersection_M_N : M ∩ N = {1, 3} := by sorry

end intersection_M_N_l3030_303075


namespace divisibility_condition_l3030_303038

theorem divisibility_condition (a b : ℕ+) :
  (a.val * b.val^2 + b.val + 7) ∣ (a.val^2 * b.val + a.val + b.val) →
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
by sorry

end divisibility_condition_l3030_303038


namespace tv_price_increase_l3030_303093

theorem tv_price_increase (x : ℝ) : 
  (((1 + x / 100) * 0.8 - 1) * 100 = 28) → x = 60 := by
  sorry

end tv_price_increase_l3030_303093


namespace geometric_inequalities_l3030_303058

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a point inside a triangle
def InsideTriangle (t : Triangle) (D : ℝ × ℝ) : Prop := sorry

-- Define a point inside a convex quadrilateral
def InsideConvexQuadrilateral (q : Quadrilateral) (E : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle at vertex A of a triangle
def angle_A (t : Triangle) : ℝ := sorry

-- Define the ratio k
def ratio_k (q : Quadrilateral) (E : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem geometric_inequalities 
  (t : Triangle) 
  (D : ℝ × ℝ) 
  (q : Quadrilateral) 
  (E : ℝ × ℝ) 
  (h1 : InsideTriangle t D) 
  (h2 : InsideConvexQuadrilateral q E) : 
  (distance t.B t.C / min (distance t.A D) (min (distance t.B D) (distance t.C D)) ≥ 
    if angle_A t < π/2 then 2 * Real.sin (angle_A t) else 2) ∧
  (ratio_k q E ≥ 2 * Real.sin (70 * π / 180)) := by
  sorry

end geometric_inequalities_l3030_303058


namespace sqrt_144_squared_times_2_l3030_303014

theorem sqrt_144_squared_times_2 : 2 * (Real.sqrt 144)^2 = 288 := by sorry

end sqrt_144_squared_times_2_l3030_303014


namespace sequence_limit_l3030_303049

/-- The sequence defined by the recurrence relation -/
noncomputable def x : ℕ → ℝ
| 0 => sorry -- x₁ is not specified in the original problem
| n + 1 => Real.sqrt (2 * x n + 3)

/-- The theorem stating that the limit of the sequence is 3 -/
theorem sequence_limit : Filter.Tendsto x Filter.atTop (nhds 3) := by sorry

end sequence_limit_l3030_303049


namespace distance_AB_is_5360_l3030_303042

/-- Represents a person in the problem -/
inductive Person
| A
| B
| C

/-- Represents a point on the path -/
structure Point where
  x : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  A : Point
  B : Point
  C : Point
  D : Point
  initialSpeed : Person → ℝ
  returnSpeed : Person → ℝ
  distanceTraveled : Person → Point → ℝ

/-- The main theorem to be proved -/
theorem distance_AB_is_5360 (setup : ProblemSetup) : 
  setup.B.x - setup.A.x = 5360 :=
sorry

end distance_AB_is_5360_l3030_303042


namespace sugar_water_sweetness_l3030_303059

theorem sugar_water_sweetness (a b m : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : m > 0) :
  (a + m) / (b + m) > a / b :=
by sorry

end sugar_water_sweetness_l3030_303059


namespace fraction_sum_inequality_l3030_303001

theorem fraction_sum_inequality (a b c : ℝ) :
  a / (a + 2*b + c) + b / (a + b + 2*c) + c / (2*a + b + c) ≥ 3/4 := by sorry

end fraction_sum_inequality_l3030_303001


namespace mans_speed_in_still_water_l3030_303088

/-- Proves that given a man rowing downstream with a current speed of 3 kmph,
    covering 80 meters in 15.99872010239181 seconds, his speed in still water is 15 kmph. -/
theorem mans_speed_in_still_water
  (current_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : current_speed = 3)
  (h2 : distance = 80)
  (h3 : time = 15.99872010239181)
  : ∃ (speed_still_water : ℝ), speed_still_water = 15 := by
  sorry

#check mans_speed_in_still_water

end mans_speed_in_still_water_l3030_303088


namespace rosie_pies_from_36_apples_l3030_303028

/-- Given that Rosie can make three pies out of twelve apples, 
    this function calculates how many pies she can make from a given number of apples. -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem rosie_pies_from_36_apples : pies_from_apples 36 = 9 := by
  sorry

#eval pies_from_apples 36

end rosie_pies_from_36_apples_l3030_303028


namespace disaster_relief_team_selection_l3030_303040

def internal_medicine_doctors : ℕ := 5
def surgeons : ℕ := 6
def total_doctors : ℕ := internal_medicine_doctors + surgeons
def team_size : ℕ := 4

theorem disaster_relief_team_selection :
  (Nat.choose total_doctors team_size) -
  (Nat.choose internal_medicine_doctors team_size) -
  (Nat.choose surgeons team_size) = 310 := by
  sorry

end disaster_relief_team_selection_l3030_303040


namespace min_good_operations_2009_l3030_303074

/-- Represents the sum of digits in the binary representation of a natural number -/
def S₂ (n : ℕ) : ℕ := sorry

/-- Represents the minimum number of "good" operations required to split a rope of length n into unit lengths -/
def min_good_operations (n : ℕ) : ℕ := sorry

/-- Theorem stating that the minimum number of good operations for a rope of length 2009 
    is equal to S₂(2009) - 1 -/
theorem min_good_operations_2009 : 
  min_good_operations 2009 = S₂ 2009 - 1 := by sorry

end min_good_operations_2009_l3030_303074


namespace rectangle_max_area_l3030_303048

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) (perimeter : x + y = 24) :
  x * y ≤ 144 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 24 ∧ a * b = 144 := by
sorry

end rectangle_max_area_l3030_303048


namespace triangle_inequality_l3030_303016

theorem triangle_inequality (A B C : ℝ) (x y z : ℝ) (n : ℕ) 
  (h_triangle : A + B + C = π) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0) : 
  x^n * Real.cos (A/2) + y^n * Real.cos (B/2) + z^n * Real.cos (C/2) ≥ 
  (y*z)^(n/2) * Real.sin A + (z*x)^(n/2) * Real.sin B + (x*y)^(n/2) * Real.sin C := by
  sorry

end triangle_inequality_l3030_303016


namespace max_clock_digit_sum_l3030_303055

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ := 
  if n < 10 then n else (n % 10 + digit_sum (n / 10))

def clock_digit_sum (h m : ℕ) : ℕ := digit_sum h + digit_sum m

theorem max_clock_digit_sum : 
  ∀ h m, is_valid_hour h → is_valid_minute m → 
  clock_digit_sum h m ≤ 28 ∧ 
  ∃ h' m', is_valid_hour h' ∧ is_valid_minute m' ∧ clock_digit_sum h' m' = 28 := by
  sorry

end max_clock_digit_sum_l3030_303055


namespace jeremy_song_count_l3030_303057

/-- The number of songs Jeremy listened to yesterday -/
def songs_yesterday : ℕ := 9

/-- The difference in songs between today and yesterday -/
def song_difference : ℕ := 5

/-- The number of songs Jeremy listened to today -/
def songs_today : ℕ := songs_yesterday + song_difference

/-- The total number of songs Jeremy listened to in two days -/
def total_songs : ℕ := songs_yesterday + songs_today

theorem jeremy_song_count : total_songs = 23 := by sorry

end jeremy_song_count_l3030_303057


namespace chord_length_squared_l3030_303020

theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 7) (h₃ : R = 10)
  (h₄ : r₁ > 0) (h₅ : r₂ > 0) (h₆ : R > 0) (h₇ : r₁ + r₂ < R) :
  let d := R - r₂
  ∃ x, x^2 = d^2 + (R - r₁)^2 ∧ 4 * x^2 = 364 :=
by sorry

end chord_length_squared_l3030_303020


namespace base_conversion_sum_l3030_303012

def base8_to_10 (n : ℕ) : ℕ := 2 * 8^2 + 5 * 8^1 + 4 * 8^0

def base2_to_10 (n : ℕ) : ℕ := 1 * 2^1 + 1 * 2^0

def base5_to_10 (n : ℕ) : ℕ := 1 * 5^2 + 4 * 5^1 + 4 * 5^0

def base4_to_10 (n : ℕ) : ℕ := 3 * 4^1 + 2 * 4^0

theorem base_conversion_sum :
  (base8_to_10 254 : ℚ) / (base2_to_10 11) + (base5_to_10 144 : ℚ) / (base4_to_10 32) = 57.4 := by
  sorry

end base_conversion_sum_l3030_303012


namespace sum_of_digits_1_to_1000_l3030_303025

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sum_of_digits_up_to (n : ℕ) : ℕ := 
  (Finset.range n).sum (λ i => sum_of_digits (i + 1))

/-- Theorem: The sum of digits of all numbers from 1 to 1000 is 14446 -/
theorem sum_of_digits_1_to_1000 : sum_of_digits_up_to 1000 = 14446 := by sorry

end sum_of_digits_1_to_1000_l3030_303025


namespace quality_related_to_renovation_probability_two_qualified_l3030_303085

-- Define the data from the table
def qualified_before : ℕ := 60
def substandard_before : ℕ := 40
def qualified_after : ℕ := 80
def substandard_after : ℕ := 20
def total_sample : ℕ := 200

-- Define the K^2 statistic
def K_squared (a b c d : ℕ) : ℚ :=
  let n : ℕ := a + b + c + d
  (n : ℚ) * (a * d - b * c : ℚ)^2 / ((a + b : ℚ) * (c + d : ℚ) * (a + c : ℚ) * (b + d : ℚ))

-- Define the critical value for 99% certainty
def critical_value : ℚ := 6635 / 1000

-- Theorem for part 1
theorem quality_related_to_renovation :
  K_squared qualified_before substandard_before qualified_after substandard_after > critical_value := by
  sorry

-- Theorem for part 2
theorem probability_two_qualified :
  (Nat.choose 3 2 : ℚ) / (Nat.choose 5 2 : ℚ) = 3 / 10 := by
  sorry

end quality_related_to_renovation_probability_two_qualified_l3030_303085


namespace jean_card_money_l3030_303002

/-- The amount of money Jean puts in each card for her grandchildren --/
def money_per_card (num_grandchildren : ℕ) (cards_per_grandchild : ℕ) (total_money : ℕ) : ℚ :=
  total_money / (num_grandchildren * cards_per_grandchild)

/-- Theorem: Jean puts $80 in each card for her grandchildren --/
theorem jean_card_money :
  money_per_card 3 2 480 = 80 := by
  sorry

end jean_card_money_l3030_303002


namespace triangle_point_distance_l3030_303024

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  AB = 13 ∧ AC = 13 ∧ BC = 10

-- Define the point P
def PointInside (P A B C : ℝ × ℝ) : Prop :=
  ∃ t u v : ℝ, t > 0 ∧ u > 0 ∧ v > 0 ∧ t + u + v = 1 ∧
  P = (t * A.1 + u * B.1 + v * C.1, t * A.2 + u * B.2 + v * C.2)

-- Define the distances PA and PB
def Distances (P A B : ℝ × ℝ) : Prop :=
  Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 15 ∧
  Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 9

-- Define the angle equality
def AngleEquality (P A B C : ℝ × ℝ) : Prop :=
  let angle (X Y Z : ℝ × ℝ) := Real.arccos (
    ((X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2)) /
    (Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) * Real.sqrt ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2))
  )
  angle A P B = angle B P C ∧ angle B P C = angle C P A

-- Main theorem
theorem triangle_point_distance (A B C P : ℝ × ℝ) :
  Triangle A B C →
  PointInside P A B C →
  Distances P A B →
  AngleEquality P A B C →
  Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) = (-9 + Real.sqrt 157) / 2 := by
  sorry

end triangle_point_distance_l3030_303024


namespace product_of_consecutive_integers_near_twin_primes_divisible_by_240_l3030_303064

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def are_twin_primes (p q : ℕ) : Prop := is_prime p ∧ is_prime q ∧ q = p + 2

theorem product_of_consecutive_integers_near_twin_primes_divisible_by_240 
  (p : ℕ) (h1 : p > 7) (h2 : are_twin_primes p (p + 2)) : 
  240 ∣ ((p - 1) * p * (p + 1)) :=
sorry

end product_of_consecutive_integers_near_twin_primes_divisible_by_240_l3030_303064


namespace continuity_at_one_l3030_303023

def f (x : ℝ) := -5 * x^2 - 7

theorem continuity_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 1| < δ → |f x - f 1| < ε :=
by sorry

end continuity_at_one_l3030_303023


namespace quadratic_coefficient_l3030_303033

/-- A quadratic function with vertex at (-2, 3) passing through (3, -45) has a = -48/25 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →  -- Quadratic function definition
  (3 = a * (-2)^2 + b * (-2) + c) →       -- Vertex condition
  (-45 = a * 3^2 + b * 3 + c) →           -- Point condition
  a = -48/25 := by sorry

end quadratic_coefficient_l3030_303033


namespace complement_of_intersection_l3030_303050

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 3, 4} := by sorry

end complement_of_intersection_l3030_303050


namespace f_neg_two_equals_six_l3030_303070

/-- The quadratic function f(x) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * b * x + c

/-- The quadratic function g(x) -/
def g (a b c : ℝ) (x : ℝ) : ℝ := (a + 1) * x^2 + 2 * (b + 2) * x + (c + 4)

/-- The discriminant of a quadratic function ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem f_neg_two_equals_six (a b c : ℝ) :
  discriminant (a + 1) (b + 2) (c + 4) - discriminant a b c = 24 →
  f a b c (-2) = 6 := by
  sorry

#eval f 1 2 3 (-2)  -- Example usage

end f_neg_two_equals_six_l3030_303070


namespace playground_children_count_l3030_303022

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 27) 
  (h2 : girls = 35) : 
  boys + girls = 62 := by
sorry

end playground_children_count_l3030_303022


namespace tank_fill_time_l3030_303084

/-- Given three pipes with fill rates, calculates the time to fill a tank when all pipes are open -/
theorem tank_fill_time (p q r : ℝ) (hp : p = 1/3) (hq : q = 1/9) (hr : r = 1/18) :
  1 / (p + q + r) = 2 := by
  sorry

end tank_fill_time_l3030_303084


namespace polynomial_coefficient_l3030_303015

theorem polynomial_coefficient (a : Fin 11 → ℝ) :
  (∀ x : ℝ, x^2 + x^10 = a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + 
    a 3 * (x + 1)^3 + a 4 * (x + 1)^4 + a 5 * (x + 1)^5 + 
    a 6 * (x + 1)^6 + a 7 * (x + 1)^7 + a 8 * (x + 1)^8 + 
    a 9 * (x + 1)^9 + a 10 * (x + 1)^10) →
  a 9 = -10 := by
sorry

end polynomial_coefficient_l3030_303015


namespace m_range_theorem_l3030_303089

def is_ellipse_with_y_foci (m : ℝ) : Prop :=
  0 < m ∧ m < 3

def hyperbola_eccentricity_in_range (m : ℝ) : Prop :=
  m > 0 ∧ Real.sqrt 1.5 < (1 + m/5).sqrt ∧ (1 + m/5).sqrt < Real.sqrt 2

def p (m : ℝ) : Prop := is_ellipse_with_y_foci m
def q (m : ℝ) : Prop := hyperbola_eccentricity_in_range m

theorem m_range_theorem (m : ℝ) :
  (0 < m ∧ m < 9) →
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  ((0 < m ∧ m ≤ 5/2) ∨ (3 ≤ m ∧ m < 5)) :=
by
  sorry

end m_range_theorem_l3030_303089


namespace chocolate_ticket_value_l3030_303069

/-- Represents the value of a chocolate box ticket in terms of the box's cost -/
def ticket_value : ℚ := 1 / 9

/-- Represents the number of tickets needed to get a free box -/
def tickets_for_free_box : ℕ := 10

/-- Theorem stating the value of a single ticket -/
theorem chocolate_ticket_value :
  ticket_value = 1 / (tickets_for_free_box - 1) :=
by sorry

end chocolate_ticket_value_l3030_303069


namespace even_function_implies_a_squared_one_l3030_303098

def f (x a : ℝ) : ℝ := x^2 + (a^2 - 1)*x + 6

theorem even_function_implies_a_squared_one (a : ℝ) :
  (∀ x, f x a = f (-x) a) → a = 1 ∨ a = -1 := by
  sorry

end even_function_implies_a_squared_one_l3030_303098


namespace power_of_power_l3030_303000

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by sorry

end power_of_power_l3030_303000


namespace total_ways_eq_17922_l3030_303005

/-- Number of cookie flavors --/
def num_cookie_flavors : ℕ := 7

/-- Number of milk types --/
def num_milk_types : ℕ := 4

/-- Total number of products to purchase --/
def total_products : ℕ := 5

/-- Maximum number of same flavor Alpha can order --/
def alpha_max_same_flavor : ℕ := 2

/-- Function to calculate the number of ways Alpha can choose items --/
def alpha_choices (n : ℕ) : ℕ := sorry

/-- Function to calculate the number of ways Beta can choose cookies --/
def beta_choices (n : ℕ) : ℕ := sorry

/-- The total number of ways Alpha and Beta can purchase 5 products --/
def total_ways : ℕ := sorry

/-- Theorem stating the total number of ways is 17922 --/
theorem total_ways_eq_17922 : total_ways = 17922 := by sorry

end total_ways_eq_17922_l3030_303005


namespace rationalize_result_l3030_303041

def rationalize_denominator (a b c : ℝ) : ℝ × ℝ × ℝ × ℝ × ℝ := sorry

theorem rationalize_result :
  let (A, B, C, D, E) := rationalize_denominator 5 7 13
  A = -4 ∧ B = 7 ∧ C = 3 ∧ D = 13 ∧ E = 1 ∧ B < D ∧
  A * Real.sqrt B + C * Real.sqrt D = 5 / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) * E :=
by sorry

end rationalize_result_l3030_303041


namespace tangent_line_at_P_l3030_303003

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem tangent_line_at_P :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ 
      (y - f P.1 = (deriv f P.1) * (x - P.1) ∧ 
       (x, y) ≠ P)) ∧
    a = 3 ∧ b = -1 ∧ c = -2 := by
  sorry

end tangent_line_at_P_l3030_303003


namespace river_speed_l3030_303035

/-- Proves that the speed of the river is 1.2 kmph given the conditions -/
theorem river_speed (rowing_speed : ℝ) (total_time : ℝ) (total_distance : ℝ)
  (h1 : rowing_speed = 8)
  (h2 : total_time = 1)
  (h3 : total_distance = 7.82) :
  ∃ v : ℝ, v = 1.2 ∧
  (total_distance / 2) / (rowing_speed - v) + (total_distance / 2) / (rowing_speed + v) = total_time :=
by sorry

end river_speed_l3030_303035
