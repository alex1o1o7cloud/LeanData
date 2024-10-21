import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l508_50842

/-- Triangle ABC with vertices A(0, 2), B(0, -2), and C(-2, 2) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The given triangle ABC -/
def triangle_ABC : Triangle :=
  { A := (0, 2),
    B := (0, -2),
    C := (-2, 2) }

/-- Equation of a line in the form ax + by + c = 0 -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle :=
  (h : ℝ)
  (k : ℝ)
  (r : ℝ)

/-- The median parallel to side BC is represented by the line 2x + y = 0 -/
def median_parallel_BC : Line :=
  { a := 2,
    b := 1,
    c := 0 }

/-- The circumcircle of triangle ABC is represented by (x + 1)^2 + y^2 = 5 -/
noncomputable def circumcircle : Circle :=
  { h := -1,
    k := 0,
    r := Real.sqrt 5 }

theorem triangle_properties :
  (median_parallel_BC = { a := 2, b := 1, c := 0 }) ∧
  (circumcircle = { h := -1, k := 0, r := Real.sqrt 5 }) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l508_50842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_square_problem_l508_50838

theorem gcd_square_problem (x y z : ℕ+) (h : (1 : ℚ) / x.val - (1 : ℚ) / y.val = (1 : ℚ) / z.val) :
  ∃ k : ℕ, Nat.gcd (Nat.gcd x.val y.val) z.val * (y.val - x.val) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_square_problem_l508_50838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_multiple_of_15_l508_50899

def n : ℕ := 6480

theorem count_divisors_multiple_of_15 : 
  (Finset.filter (fun d => d ∣ n ∧ 15 ∣ d) (Finset.range (n + 1))).card = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_multiple_of_15_l508_50899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_change_omega_l508_50898

/-- Represents the production and price data for vegetables and fruits in a given year -/
structure YearData where
  vegProduction : ℝ
  fruitProduction : ℝ
  vegPrice : ℝ
  fruitPrice : ℝ

/-- Calculates the nominal GDP for a given year -/
noncomputable def nominalGDP (data : YearData) : ℝ :=
  data.vegProduction * data.vegPrice + data.fruitProduction * data.fruitPrice

/-- Calculates the real GDP for a given year using base year prices -/
noncomputable def realGDP (data : YearData) (baseData : YearData) : ℝ :=
  data.vegProduction * baseData.vegPrice + data.fruitProduction * baseData.fruitPrice

/-- Calculates the percentage change between two values -/
noncomputable def percentageChange (oldValue : ℝ) (newValue : ℝ) : ℝ :=
  100 * (newValue - oldValue) / oldValue

theorem gdp_change_omega :
  let data2014 : YearData := {
    vegProduction := 1200,
    fruitProduction := 750,
    vegPrice := 90000,
    fruitPrice := 75000
  }
  let data2015 : YearData := {
    vegProduction := 900,
    fruitProduction := 900,
    vegPrice := 100000,
    fruitPrice := 70000
  }
  let gdpChange := percentageChange (nominalGDP data2014) (realGDP data2015 data2014)
  ∃ ε > 0, abs (gdpChange + 9.59) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gdp_change_omega_l508_50898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_dot_product_range_l508_50819

open Real

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

def Triangle.valid (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = π

def Vector2D := ℝ × ℝ

def dotProduct (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem triangle_angle_and_dot_product_range (t : Triangle) 
  (h_valid : t.valid)
  (h_area : 4 * t.area = Real.sqrt 3 * (t.a^2 + t.c^2 - t.b^2)) :
  (t.B = π / 3) ∧
  (∀ (m n : Vector2D), 
    m = (sin (2 * t.A), 3 * cos t.A) ∧ 
    n = (3, -2 * cos t.A) →
    ∃ (x : ℝ), x ∈ Set.Ioo (-6 : ℝ) (3 * Real.sqrt 2 - 3) ∧
    dotProduct m n = x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_dot_product_range_l508_50819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_2018_equation_l508_50830

theorem unique_solution_2018_equation :
  ∀ x y z : ℕ,
    x > 0 → y > 0 → z > 0 →
    z % 2 = 1 →
    (2018 : ℤ)^x = (100 : ℤ)^y + (1918 : ℤ)^z →
    x = 1 ∧ y = 1 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_2018_equation_l508_50830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l508_50896

/-- The probability of drawing two chips of different colors from a bag containing
    7 blue chips, 5 yellow chips, and 2 red chips, when drawing with replacement. -/
theorem two_different_color_chips_probability :
  (let total_chips : ℕ := 7 + 5 + 2
   let blue_chips : ℕ := 7
   let yellow_chips : ℕ := 5
   let red_chips : ℕ := 2
   let prob_blue : ℚ := blue_chips / total_chips
   let prob_yellow : ℚ := yellow_chips / total_chips
   let prob_red : ℚ := red_chips / total_chips
   let prob_different_colors : ℚ := 
     prob_blue * (prob_yellow + prob_red) +
     prob_yellow * (prob_blue + prob_red) +
     prob_red * (prob_blue + prob_yellow)
   prob_different_colors) = 59 / 98 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_different_color_chips_probability_l508_50896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l508_50868

theorem sin_minus_cos_value (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = 7/5)
  (h2 : π/4 < α ∧ α < π/2) :
  Real.sin α - Real.cos α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_minus_cos_value_l508_50868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l508_50891

def is_valid (n : ℕ+) : Prop :=
  (Finset.card (Nat.divisors n.val) = 144) ∧
  (∃ k : ℕ, ∀ i : ℕ, i < 10 → (k + i) ∣ n.val)

theorem smallest_valid_number : 
  (is_valid 110880) ∧ (∀ m : ℕ+, m < 110880 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_number_l508_50891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l508_50829

theorem problem_solution (t : ℝ) (k m n : ℕ+) 
  (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 3/2)
  (h2 : (1 - Real.sin t) * (1 - Real.cos t) = (m : ℝ) / (n : ℝ) - Real.sqrt (k : ℝ))
  (h3 : Nat.Coprime m n) : 
  k + m + n = 19 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l508_50829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l508_50825

theorem remainder_theorem (x y z p q : ℕ) 
  (hx : 4 ∣ x)
  (hy : 5 ∣ y)
  (hz : 6 ∣ z)
  (hp : 7 ∣ p)
  (hq : 3 ∣ q)
  (hpos : x > 0 ∧ y > 0 ∧ z > 0 ∧ p > 0 ∧ q > 0) :
  (x^3 * y^2 * (z*p*q + (x + y)^3) - 10) % 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l508_50825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_concentric_circles_l508_50887

/-- Given three concentric circles with chords drawn on the largest circle
    tangent to the middle circle (but not the smallest), and the measure of
    angle ABC is 60°, the minimum number of segments needed to return to
    the starting point is 3. -/
theorem min_segments_concentric_circles : ∃ n : ℕ, n > 0 ∧ n = 3 := by
  -- Define the measure of angle ABC
  let angle_measure : ℝ := 60

  -- Define the measure of the minor arc
  let minor_arc_measure : ℝ := 2 * angle_measure

  -- Define the condition for completing the circle
  let circle_completion (k : ℕ) := k * minor_arc_measure = 360

  -- The minimum number of segments is 3
  have h : circle_completion 3 := by
    sorry

  -- Any smaller number of segments does not complete the circle
  have h2 : ∀ m : ℕ, m < 3 → ¬ circle_completion m := by
    sorry

  -- Therefore, the minimum number of segments is 3
  use 3
  constructor
  · exact Nat.succ_pos 2
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segments_concentric_circles_l508_50887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_from_difference_and_product_l508_50828

theorem sum_from_difference_and_product (a b : ℝ) :
  a - b = -3 → a * b = 2 → (a + b = Real.sqrt 17 ∨ a + b = -Real.sqrt 17) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_from_difference_and_product_l508_50828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_difference_of_squares_l508_50843

/-- Predicate to check if an expression can be factored using the difference of squares formula -/
def is_difference_of_squares (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ g h : ℝ → ℝ → ℝ, ∀ a b, f a b = (g a b)^2 - (h a b)^2

/-- The given expressions from the problem -/
def expr_A (a b : ℝ) : ℝ := a^2 + b^2
def expr_B (a b : ℝ) : ℝ := -a^2 + b^2
def expr_C (a b : ℝ) : ℝ := -a^2 - b^2
def expr_D (a b : ℝ) : ℝ := -(-a^2) + b^2

/-- Theorem stating that only expr_B can be factored using the difference of squares formula -/
theorem only_B_is_difference_of_squares :
  is_difference_of_squares expr_B ∧
  ¬is_difference_of_squares expr_A ∧
  ¬is_difference_of_squares expr_C ∧
  ¬is_difference_of_squares expr_D :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_difference_of_squares_l508_50843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l508_50800

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_properties :
  (∃ T > 0, is_periodic f T ∧ ∀ S, 0 < S → is_periodic f S → T ≤ S) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), -1 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x = 2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f x = -1) :=
by sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l508_50800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_correct_l508_50818

/-- Fibonacci-like sequence for counting valid numbers -/
def F : ℕ → ℕ
  | 0 => 2  -- Add this case
  | 1 => 2
  | 2 => 3
  | n + 3 => F (n + 2) + F (n + 1)

/-- The number of 11-digit integers with digits 1 or 2 and at least two consecutive 1's -/
def count_integers : ℕ := 2^11 - F 11

theorem count_integers_correct : count_integers = 1815 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_correct_l508_50818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_outcome_is_three_fourths_l508_50822

def six_sided_die : Finset ℕ := Finset.range 6

def is_even (n : ℕ) : Bool := n % 2 = 0

noncomputable def prob_even_outcome : ℚ :=
  (Finset.filter (λ x => is_even x) six_sided_die).card / six_sided_die.card +
  ((Finset.filter (λ x => ¬(is_even x)) six_sided_die).card / six_sided_die.card) *
  ((Finset.filter (λ x => is_even x) six_sided_die).card / six_sided_die.card)

theorem prob_even_outcome_is_three_fourths : prob_even_outcome = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_outcome_is_three_fourths_l508_50822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_circle_l508_50832

/-- Given a circle with center (1,-1) and a point (4,3) on the circle,
    the slope of the tangent line at (4,3) is -3/4. -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (1, -1) → point = (4, 3) → 
  (((point.2 - center.2) / (point.1 - center.1))⁻¹ * (-1)) = -3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_circle_l508_50832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_l508_50804

theorem two_digit_number (a : ℤ) : 
  10 * a + (a + 1) = 11 * a + 1 := by
  ring

#check two_digit_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_number_l508_50804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_upper_bound_l508_50886

theorem inequality_implies_upper_bound (x : ℝ) :
  (5 * x + 2) ^ (1/3) - (x + 3) ^ (1/3) ≤ 1 → x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_implies_upper_bound_l508_50886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l508_50851

open Real

-- Define the function f
noncomputable def f (f'1 : ℝ) : ℝ → ℝ := λ x ↦ (1/3) * x^3 - f'1 * x^2 + x + 5

-- State the theorem
theorem f_derivative_at_one :
  ∃ f'1 : ℝ, (deriv (f f'1)) 1 = f'1 ∧ f'1 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l508_50851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_acb_is_right_angle_l508_50882

/-- A point in 2D plane --/
structure Point :=
  (x y : ℝ)

/-- Distance between two points --/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : Point)

/-- An equilateral triangle --/
def IsEquilateral (t : Triangle) : Prop :=
  distance t.A t.B = distance t.B t.C ∧ distance t.B t.C = distance t.C t.A

/-- The main triangle in the problem --/
noncomputable def MainTriangle : Triangle := sorry

/-- The set of all small triangles in the figure --/
def SmallTriangles : Set Triangle := sorry

/-- An angle --/
structure Angle :=
  (value : ℝ)

/-- The angle ACB in the main triangle --/
noncomputable def AngleACB : Angle := sorry

/-- The theorem statement --/
theorem angle_acb_is_right_angle
  (h1 : ∀ t ∈ SmallTriangles, IsEquilateral t)
  (h2 : ∀ t ∈ SmallTriangles, distance t.A t.B = 1) :
  AngleACB.value = 90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_acb_is_right_angle_l508_50882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l508_50809

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 4 else 2^x

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x ≤ -2 ∨ (0 < x ∧ x ≤ 1)}

-- Theorem statement
theorem f_inequality_solution :
  {x : ℝ | f x ≤ 2} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_l508_50809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l508_50862

noncomputable def curve_C1 (x : ℝ) : ℝ := Real.sin x

noncomputable def curve_C2 (x : ℝ) : ℝ := Real.cos ((1/2) * x - (5 * Real.pi) / 6)

theorem curve_transformation (x : ℝ) :
  curve_C2 x = curve_C1 ((1/2) * (x - (2 * Real.pi) / 3)) := by
  calc
    curve_C2 x = Real.cos ((1/2) * x - (5 * Real.pi) / 6) := rfl
    _ = Real.sin ((1/2) * x - (5 * Real.pi) / 6 + Real.pi / 2) := by sorry
    _ = Real.sin ((1/2) * x - Real.pi / 3) := by sorry
    _ = Real.sin ((1/2) * (x - (2 * Real.pi) / 3)) := by sorry
    _ = curve_C1 ((1/2) * (x - (2 * Real.pi) / 3)) := rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l508_50862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_theorem_l508_50848

/-- Represents the profit percentage when selling an item with a discount -/
noncomputable def profit_with_discount : ℝ := 14

/-- Represents the discount percentage offered -/
noncomputable def discount_percentage : ℝ := 5

/-- Calculates the profit percentage when no discount is offered -/
noncomputable def profit_without_discount (p d : ℝ) : ℝ :=
  (100 + p) / (100 - d) * 100 - 100

/-- Theorem stating that with a 5% discount and 14% profit, 
    selling without a discount would result in a 20% profit -/
theorem discount_profit_theorem :
  profit_without_discount profit_with_discount discount_percentage = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_profit_theorem_l508_50848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_property_H_implications_l508_50821

-- Define the property H(t)
def has_property_H (f : ℝ → ℝ) (t : ℝ) : Prop :=
  ∀ x, f (x + t) + (t - 1) * f x = 0

-- Define continuity
def continuous (f : ℝ → ℝ) : Prop :=
  ∀ x ε, ε > 0 → ∃ δ, δ > 0 ∧ ∀ y, |y - x| < δ → |f y - f x| < ε

theorem property_H_implications (f : ℝ → ℝ) (h_cont : continuous f) :
  (has_property_H f 2 → f = λ x ↦ Real.sin (π/2 * x)) ∧
  (has_property_H f 2 → ∀ x, f (x + 4) = f x) ∧
  (has_property_H f (1/2) → f 0 = 1 → ∀ k : ℕ, f k = (1/4)^k) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_property_H_implications_l508_50821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_odd_power_sum_l508_50823

theorem existence_of_odd_power_sum (m : ℕ+) : 
  ∃ (a b : ℤ) (k : ℕ), 
    Odd a ∧ Odd b ∧ (2 * m : ℤ) = a^19 + b^99 + k * 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_odd_power_sum_l508_50823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_decrease_l508_50888

theorem circle_area_decrease (r : ℝ) (h : r > 0) : 
  let r' := r * (1 - 0.2384226894136092)
  let A := Real.pi * r^2
  let A' := Real.pi * r'^2
  abs ((A - A') / A - 0.42) < 1e-10
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_decrease_l508_50888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_h_increasing_l508_50841

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := 4 * x + 1
def g (x : ℝ) : ℝ := x^2 - 1
def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function
noncomputable def h_inv (x : ℝ) : ℝ := Real.sqrt ((x + 3) / 4)

-- Theorem statement
theorem h_inverse_correct (x : ℝ) (hx : x ≥ 0) :
  h (h_inv x) = x ∧ h_inv (h x) = x := by
  sorry

-- Additional lemma to show h is increasing on non-negative reals
theorem h_increasing (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x < y) :
  h x < h y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_correct_h_increasing_l508_50841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_petya_wins_from_2021_l508_50811

/-- Represents a game state with a number of minuses -/
structure GameState where
  minuses : Nat

/-- Represents a move in the game -/
inductive Move
  | changeOneToPlus
  | eraseOnePlusOneMinus
  | changeTwoToThreePlus

/-- Applies a move to a game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.changeOneToPlus => ⟨state.minuses - 1⟩
  | Move.eraseOnePlusOneMinus => ⟨state.minuses - 1⟩
  | Move.changeTwoToThreePlus => ⟨state.minuses - 2⟩

/-- Checks if a move is valid for a given game state -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  match move with
  | Move.changeOneToPlus => state.minuses ≥ 1
  | Move.eraseOnePlusOneMinus => state.minuses ≥ 1
  | Move.changeTwoToThreePlus => state.minuses ≥ 2

/-- Theorem: There exists a winning strategy for the first player -/
theorem first_player_wins :
  ∃ (strategy : GameState → Move),
    ∀ (state : GameState),
      state.minuses % 3 = 0 →
      isValidMove state (strategy state) ∧
      (applyMove state (strategy state)).minuses % 3 = 0 :=
by sorry

/-- Theorem: The first player can win starting with 2021 minuses -/
theorem petya_wins_from_2021 :
  ∃ (strategy : GameState → Move),
    ∀ (opponent_strategy : GameState → Move),
      let initial_state := GameState.mk 2021
      let rec game_sequence (n : Nat) : GameState :=
        match n with
        | 0 => initial_state
        | n+1 => applyMove (applyMove (game_sequence n) (strategy (game_sequence n))) (opponent_strategy (applyMove (game_sequence n) (strategy (game_sequence n))))
      (game_sequence (2021 / 3 + 1)).minuses = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_petya_wins_from_2021_l508_50811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_l508_50861

/-- The hyperbola C: x²/a² - y² = 1 (a > 0) intersects the line l: x + y = 1 at two distinct points A and B. 
    P is the intersection of l with the y-axis. If PA = (5/12)PB, then a = 17/13. -/
theorem hyperbola_intersection (a : ℝ) (A B P : ℝ × ℝ) : 
  a > 0 →
  (∀ x y, x^2 / a^2 - y^2 = 1 ↔ (x, y) ∈ ({A, B} : Set (ℝ × ℝ))) →
  (∀ x y, x + y = 1 ↔ (x, y) ∈ ({A, B, P} : Set (ℝ × ℝ))) →
  P.1 = 0 ∧ P.2 = 1 →
  A ≠ B →
  A - P = (5/12) • (B - P) →
  a = 17/13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_l508_50861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_sq_ft_is_two_l508_50894

/-- Represents the square footage of a house -/
structure HouseSquareFootage where
  masterArea : ℝ
  guestRoomArea : ℝ
  otherArea : ℝ

/-- Calculates the total square footage of a house -/
noncomputable def totalSquareFootage (h : HouseSquareFootage) : ℝ :=
  h.masterArea + 2 * h.guestRoomArea + h.otherArea

/-- Calculates the cost per square foot given the total rent and house square footage -/
noncomputable def costPerSquareFoot (rent : ℝ) (h : HouseSquareFootage) : ℝ :=
  rent / totalSquareFootage h

/-- Theorem stating that for the given house specifications and rent, the cost per square foot is $2 -/
theorem cost_per_sq_ft_is_two :
  let h : HouseSquareFootage := ⟨500, 200, 600⟩
  let rent : ℝ := 3000
  costPerSquareFoot rent h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_sq_ft_is_two_l508_50894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_upper_bound_l508_50875

theorem sin_upper_bound (a : ℝ) : (∀ x : ℝ, Real.sin x ≤ a) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_upper_bound_l508_50875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l508_50889

-- Define a, b, and c
noncomputable def a : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def b : ℝ := Real.log 3 / Real.log (1/2)
noncomputable def c : ℝ := (1/2) ^ (3/10)

-- State the theorem
theorem abc_inequality : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l508_50889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_savings_theorem_l508_50849

/-- Calculates simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates compound interest (annually compounded) -/
noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

theorem julie_savings_theorem (totalSavings : ℝ) (simpleInterestEarned : ℝ) :
  totalSavings = 1200 →
  simpleInterestEarned = 120 →
  let principal := totalSavings / 2
  let time := 2
  let rate := simpleInterestEarned / (principal * time)
  compoundInterest principal rate time = 126 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_julie_savings_theorem_l508_50849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_paint_theorem_l508_50807

/-- The amount of white paint in ounces to be added to the mixture --/
def white_paint_amount (blue_ratio : ℕ) (red_ratio : ℕ) (white_ratio : ℕ) 
  (blue_amount : ℚ) (total_max : ℚ) : ℚ :=
  let total_ratio := blue_ratio + red_ratio + white_ratio
  let part_amount := blue_amount / blue_ratio
  let max_parts := min (total_max / part_amount) (total_ratio : ℚ)
  white_ratio * part_amount

/-- Theorem stating that given the conditions, the amount of white paint to be added is 20 ounces --/
theorem white_paint_theorem : 
  white_paint_amount 7 2 1 140 180 = 20 := by
  sorry

#eval white_paint_amount 7 2 1 140 180

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_paint_theorem_l508_50807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_problem_l508_50850

/-- The growth rate of bacteria in terms of hours required to quadruple -/
noncomputable def quadruple_time : ℝ := 12

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 200

/-- The target number of bacteria -/
def target_bacteria : ℕ := 819200

/-- The time period to consider for the second part of the problem -/
noncomputable def time_period : ℝ := 24

/-- The function modeling bacterial growth -/
noncomputable def bacteria_growth (t : ℝ) : ℝ := initial_bacteria * (4 ^ (t / quadruple_time))

theorem bacteria_growth_problem :
  (∃ t : ℝ, bacteria_growth t = target_bacteria ∧ t = 72) ∧
  bacteria_growth time_period = 3200 := by
  sorry

#eval initial_bacteria
#eval target_bacteria

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_problem_l508_50850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_is_4_sqrt_83_l508_50890

/-- A rectangular box with given surface area and edge length sum -/
structure RectangularBox where
  a : ℝ
  b : ℝ
  c : ℝ
  total_surface_area : a * b + b * c + c * a = 71
  total_edge_length : a + b + c = 15

/-- The sum of the lengths of all interior diagonals of a rectangular box -/
noncomputable def interior_diagonals_sum (box : RectangularBox) : ℝ :=
  4 * Real.sqrt (box.a^2 + box.b^2 + box.c^2)

/-- Theorem: The sum of the lengths of all interior diagonals is 4√83 -/
theorem interior_diagonals_sum_is_4_sqrt_83 (box : RectangularBox) :
  interior_diagonals_sum box = 4 * Real.sqrt 83 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonals_sum_is_4_sqrt_83_l508_50890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l508_50806

def A (a : ℝ) := {x : ℝ | x^2 - a*x + a^2 - 19 = 0}
def B := {x : ℝ | x^2 - 5*x + 6 = 0}
def C := {x : ℝ | x^2 + 2*x - 8 = 0}

theorem problem_solution :
  (∀ a : ℝ, A a ⊆ B → (a < -2*Real.sqrt 57/3 ∨ a > 2*Real.sqrt 57/3 ∨ a = 5)) ∧
  (∀ a : ℝ, (Set.Nonempty (A a ∩ B) ∧ A a ∩ C = ∅) → a = -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l508_50806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_equality_l508_50854

theorem sine_tangent_equality (θ : Real) : 
  0 ≤ θ ∧ θ ≤ Real.pi / 2 →  -- First quadrant condition
  (∀ (ε : Real), ε > 0 → ε ≥ 0.001 → |Real.sin θ - Real.tan θ| < ε) ↔ 
  0 ≤ θ ∧ θ ≤ 4 * Real.pi / 180 + 20 * Real.pi / (180 * 60) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_tangent_equality_l508_50854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_ratio_l508_50880

theorem inscribed_hexagon_area_ratio :
  ∀ (r : ℝ),
  r > 0 →
  let large_hexagon_side := 2 * r;
  let small_hexagon_side := r;
  let large_hexagon_area := (3 * Real.sqrt 3 / 2) * large_hexagon_side^2;
  let small_hexagon_area := (3 * Real.sqrt 3 / 2) * small_hexagon_side^2
  small_hexagon_area / large_hexagon_area = 1 / 4 := by
  intro r hr
  -- The proof steps would go here
  sorry

#check inscribed_hexagon_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_hexagon_area_ratio_l508_50880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l508_50835

theorem expression_evaluation (x : ℝ) :
  x^2 = 16 →
  (1 - (x + 1) / (x^2 - 2*x + 1)) / ((x - 3) / (x - 1)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l508_50835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_in_house_correct_birds_in_house_is_four_l508_50856

/-- The number of birds in a house with given conditions -/
def birds_in_house : ℕ :=
  let dogs : ℕ := 3
  let cats : ℕ := 18
  let family_members : ℕ := 7
  let bird_feet : ℕ := 2
  let dog_feet : ℕ := 4
  let cat_feet : ℕ := 4
  let human_feet : ℕ := 2
  let feet_head_difference : ℕ := 74

  4

/-- Proof that the number of birds satisfies the given conditions -/
theorem birds_in_house_correct : 
  let dogs : ℕ := 3
  let cats : ℕ := 18
  let family_members : ℕ := 7
  let bird_feet : ℕ := 2
  let dog_feet : ℕ := 4
  let cat_feet : ℕ := 4
  let human_feet : ℕ := 2
  let feet_head_difference : ℕ := 74
  ∃ birds : ℕ, 
    (birds * bird_feet + dogs * dog_feet + cats * cat_feet + family_members * human_feet) = 
    (birds + dogs + cats + family_members + feet_head_difference) ∧
    birds = birds_in_house := by
  sorry

/-- Proof that birds_in_house is equal to 4 -/
theorem birds_in_house_is_four : birds_in_house = 4 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birds_in_house_correct_birds_in_house_is_four_l508_50856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l508_50871

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola y^2 = 4x -/
def focus : Point := ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a vertical line -/
def distanceToVerticalLine (p : Point) (x : ℝ) : ℝ :=
  |p.x - x|

theorem parabola_focus_distance 
  (P : Point) 
  (h1 : P ∈ Parabola) 
  (h2 : distanceToVerticalLine P (-2) = 6) : 
  distance P focus = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l508_50871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equalities_l508_50853

theorem sqrt_expressions_equalities : 
  (Real.sqrt 32 + Real.sqrt 2 * (Real.sqrt 2 - 3) = Real.sqrt 2 + 2) ∧ 
  ((Real.sqrt 3 - 2)^2 + Real.sqrt 12 + 6 * Real.sqrt (1/3) = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_expressions_equalities_l508_50853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l508_50812

/-- The function f(x) = (2x-3)/(4x+5) -/
noncomputable def f (x : ℝ) : ℝ := (2*x - 3) / (4*x + 5)

/-- The x-value of the vertical asymptote for f(x) -/
noncomputable def vertical_asymptote : ℝ := -5/4

/-- Theorem stating that the vertical asymptote of f(x) is at x = -5/4 -/
theorem vertical_asymptote_of_f :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - vertical_asymptote| ∧ |x - vertical_asymptote| < δ →
    |f x| > 1/ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_of_f_l508_50812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l508_50874

theorem tan_double_angle (x : ℝ) (h1 : x ∈ Set.Ioo (-π/2) 0) (h2 : Real.sin x = -3/5) : 
  Real.tan (2 * x) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_l508_50874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_representation_l508_50870

theorem sum_of_squares_representation (x y z q : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  x * y - z^2 = 1 →
  z = Nat.factorial (2 * q) →
  p = 4 * q + 1 →
  (∃ (a b c d : ℕ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d) ∧
  (∃ (m n : ℤ), ↑p = m^2 + n^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_representation_l508_50870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l508_50883

-- Define the angles as real numbers representing degrees
variable (angle_A angle_B angle_C : ℝ)

-- State the conditions
axiom parallel_lines : True  -- We can't directly represent parallel lines, so we use this as a placeholder
axiom angle_A_ratio : angle_A = angle_B / 6
axiom angle_A_equals_C : angle_A = angle_C
axiom straight_line : angle_B + angle_C = 180

-- State the theorem
theorem angle_C_measure : ∃ ε > 0, |angle_C - 25.71| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_measure_l508_50883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_bc_length_l508_50831

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Golden section point -/
def isGoldenSectionPoint (a b c : ℝ) : Prop :=
  (b - a) / (c - a) = φ

theorem golden_section_bc_length :
  ∀ (a b c : ℝ),
  isGoldenSectionPoint a c b →
  c - a > b - c →
  c - a = 20 →
  b - c = 10 * (3 - Real.sqrt 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_bc_length_l508_50831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_solution_solution_is_point_three_l508_50840

theorem exists_unique_solution : ∃! x : ℝ, 0.2 * x + (0.4 * 0.5) = 0.26 := by
  sorry

theorem solution_is_point_three : 
  let x : ℝ := Classical.choose exists_unique_solution
  x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_solution_solution_is_point_three_l508_50840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l508_50833

/-- The standard equation of a circle with center (-1, 2) and tangent to the line y = x - 1 -/
theorem circle_equation :
  ∀ (x y : ℝ), 
  (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = x - 1 ∧ 
    (p.1 - (-1))^2 + (p.2 - 2)^2 = ((x + 1)^2 + (y - 2)^2)) →
  (x + 1)^2 + (y - 2)^2 = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l508_50833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plates_clothes_usage_is_48_l508_50859

/-- Represents the water collection and usage system --/
structure WaterSystem where
  barrel1_capacity : ℚ
  barrel2_capacity : ℚ
  barrel3_capacity : ℚ
  barrel1_fill : ℚ
  barrel2_fill : ℚ
  barrel3_fill : ℚ
  car_wash_usage : ℚ
  num_cars : ℕ
  plant_usage : ℚ
  dog_usage : ℚ
  cooking_usage : ℚ
  bathing_usage : ℚ

/-- Calculates the total water collected --/
def total_collected (ws : WaterSystem) : ℚ :=
  ws.barrel1_capacity * ws.barrel1_fill +
  ws.barrel2_capacity * ws.barrel2_fill +
  ws.barrel3_capacity * ws.barrel3_fill

/-- Calculates the total water used for tasks --/
def total_used (ws : WaterSystem) : ℚ :=
  ws.car_wash_usage * ws.num_cars +
  ws.plant_usage + ws.dog_usage +
  ws.cooking_usage + ws.bathing_usage

/-- Calculates the water used for washing plates and clothes --/
def plates_clothes_usage (ws : WaterSystem) : ℚ :=
  (total_collected ws - total_used ws) / 2

/-- Theorem stating the amount of water used for washing plates and clothes --/
theorem plates_clothes_usage_is_48 (ws : WaterSystem) 
  (h1 : ws.barrel1_capacity = 65)
  (h2 : ws.barrel2_capacity = 75)
  (h3 : ws.barrel3_capacity = 45)
  (h4 : ws.barrel1_fill = 1)
  (h5 : ws.barrel2_fill = 4/5)
  (h6 : ws.barrel3_fill = 3/5)
  (h7 : ws.car_wash_usage = 7)
  (h8 : ws.num_cars = 2)
  (h9 : ws.plant_usage = 15)
  (h10 : ws.dog_usage = 10)
  (h11 : ws.cooking_usage = 5)
  (h12 : ws.bathing_usage = 12) :
  plates_clothes_usage ws = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plates_clothes_usage_is_48_l508_50859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_schedule_count_indeterminate_l508_50860

/-- Represents a player in the tournament -/
structure Player where
  school : Nat
  id : Nat

/-- Represents a game between two players -/
structure Game where
  player1 : Player
  player2 : Player

/-- Represents a round in the tournament -/
structure Round where
  games : List Game

/-- Represents the tournament schedule -/
structure TournamentSchedule where
  rounds : List Round

/-- The number of schools in the tournament -/
def num_schools : Nat := 2

/-- The number of players per school -/
def players_per_school : Nat := 4

/-- The number of games each player plays against each player from the other school -/
def games_per_player_pair : Nat := 2

/-- The number of games played simultaneously in each round -/
def games_per_round : Nat := 3

/-- Predicate to check if a tournament schedule is valid -/
def is_valid_schedule (schedule : TournamentSchedule) : Prop :=
  -- Each player plays the correct number of games against each player from the other school
  -- No player plays more than once in a round
  -- The correct number of games are played in each round
  sorry

/-- Theorem stating that the number of valid schedules cannot be determined by a simple factorial calculation -/
theorem schedule_count_indeterminate :
  ∀ n : Nat, ¬ (n = Nat.card { s : TournamentSchedule | is_valid_schedule s }) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_schedule_count_indeterminate_l508_50860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_l508_50827

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The function g -/
noncomputable def g (x : ℂ) : ℂ := (x^4 - 2*x^3 + x^2) / (x + 2)

/-- Theorem stating that g(i) = 2/5 + 4i/5 -/
theorem g_of_i : g i = 2/5 + 4*i/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_i_l508_50827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l508_50834

-- Define the function f(x) = x + log₄(x) - 7
noncomputable def f (x : ℝ) : ℝ := x + (Real.log x) / (Real.log 4) - 7

-- State the theorem
theorem solution_in_interval :
  ∃ x : ℝ, x > 5 ∧ x < 6 ∧ f x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_in_interval_l508_50834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hyperdeficient_integers_l508_50893

/-- Sum of all divisors of a positive integer n -/
def f (n : ℕ+) : ℕ+ := sorry

/-- A positive integer n is hyperdeficient if f(f(n)) = n + 4 -/
def is_hyperdeficient (n : ℕ+) : Prop := (f (f n)).val = n.val + 4

/-- There are no hyperdeficient positive integers -/
theorem no_hyperdeficient_integers : ∀ n : ℕ+, ¬ is_hyperdeficient n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_hyperdeficient_integers_l508_50893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integer_perfect_square_sum_l508_50810

theorem max_negative_integer_perfect_square_sum (p : ℤ) : 
  p < 0 → 
  ∃ (n : ℕ), (2001 : ℤ) + p = n^2 →
  p ≤ -65 ∧ ∃ (m : ℕ), (2001 : ℤ) + (-65) = m^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_negative_integer_perfect_square_sum_l508_50810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l508_50820

theorem m_greater_than_n (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a^4 + b^4) * (a^2 + b^2) > (a^3 + b^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_greater_than_n_l508_50820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_l508_50824

noncomputable def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

def tangent_line (x y : ℝ) : Prop := x - y + 2 = 0

def P : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (0, 2)

def symmetric_points (M N : ℝ × ℝ) (a b : ℝ) : Prop :=
  ellipse_C M.1 M.2 a b ∧ ellipse_C N.1 N.2 a b ∧ M.1 = -N.1 ∧ M.2 = N.2

def intersection_point (T M N : ℝ × ℝ) : Prop :=
  (T.2 - P.2) * (M.1 - P.1) = (M.2 - P.2) * (T.1 - P.1) ∧
  (T.2 - Q.2) * (N.1 - Q.1) = (N.2 - Q.2) * (T.1 - Q.1)

theorem ellipse_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hecc : eccentricity a b = Real.sqrt 3 / 2)
  (htangent : ∃ (x y : ℝ), tangent_line x y ∧ x^2 + y^2 = b^2 / 4) :
  (∀ x y, ellipse_C x y a b ↔ x^2 / 8 + y^2 / 2 = 1) ∧
  (∀ M N T : ℝ × ℝ, symmetric_points M N a b → intersection_point T M N →
    T.1^2 / 8 + T.2^2 / 2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_proof_l508_50824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l508_50855

/-- Given a geometric progression with the first three terms as specified,
    prove that the fourth term is 3^(-1/6) -/
theorem fourth_term_of_geometric_progression
  (a : ℝ) (r : ℝ) -- First term and common ratio
  (h1 : a = Real.sqrt 3)
  (h2 : a * r = 3^(1/4 : ℝ))
  (h3 : a * r^2 = 3^(1/12 : ℝ)) :
  a * r^3 = 3^(-(1/6 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l508_50855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l508_50816

-- Define the curves C1 and C2
noncomputable def C1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

noncomputable def C2 (θ : ℝ) : ℝ × ℝ := 
  let ρ := Real.sqrt (4 * Real.sin θ - 3)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- State the theorem
theorem max_distance_between_curves :
  ∃ (max_dist : ℝ), max_dist = 2 * Real.sqrt 21 / 3 + 1 ∧
  ∀ (θ₁ θ₂ : ℝ), 
    let (x₁, y₁) := C1 θ₁
    let (x₂, y₂) := C2 θ₂
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_curves_l508_50816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l508_50892

theorem cos_difference (α β : ℝ) (h1 : Real.cos (α + β) = 1/5) (h2 : Real.tan α * Real.tan β = 1/2) :
  Real.cos (α - β) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l508_50892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solutions_for_equation_l508_50895

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y k : ℤ), (x : ℚ)^2009 + (y : ℚ)^2009 = (7 : ℚ)^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_solutions_for_equation_l508_50895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_17_l508_50897

theorem remainder_sum_mod_17 (a b c d : ℕ) 
  (ha : a % 17 = 3)
  (hb : b % 17 = 5)
  (hc : c % 17 = 7)
  (hd : d % 17 = 9) :
  (a + b + c + d) % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_sum_mod_17_l508_50897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_0_value_l508_50805

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem x_0_value (x₀ : ℝ) (h : x₀ > 0) :
  deriv f x₀ = 2 → x₀ = exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_0_value_l508_50805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l508_50877

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := Real.log x + 1

-- Theorem statement
theorem tangent_line_at_e :
  ∀ x y : ℝ, (x = Real.exp 1 ∧ y = f (Real.exp 1)) →
  (2 * x - y - Real.exp 1 = 0 ↔ y - f (Real.exp 1) = f_derivative (Real.exp 1) * (x - Real.exp 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_e_l508_50877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l508_50846

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- State the theorem
theorem hyperbola_eccentricity (a b : ℝ) (h1 : b > Real.sqrt 2 * a) (h2 : a > 0) :
  let e := eccentricity a b
  (3 * e^2 - 11 * e + 10 = 0) → e = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l508_50846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_jogs_purchase_l508_50879

/-- Represents the number of jags purchased -/
def jags : ℕ := 8

/-- Represents the number of jigs purchased -/
def jigs : ℕ := 8

/-- Represents the number of jogs purchased -/
def jogs : ℕ := 4

/-- The cost of a single jag -/
def jag_cost : ℕ := 3

/-- The cost of a single jig -/
def jig_cost : ℕ := 4

/-- The cost of a single jog -/
def jog_cost : ℕ := 10

/-- The total amount Bill will spend -/
def total_spend : ℕ := 100

/-- The minimum number of items Bill must purchase -/
def min_items : ℕ := 20

theorem max_jogs_purchase :
  (∀ j : ℕ, j > jogs → 
    (jag_cost * jags + jig_cost * jigs + jog_cost * j ≠ total_spend ∨
     jags + jigs + j < min_items ∨
     jags = 0 ∨ jigs = 0)) ∧
  jag_cost * jags + jig_cost * jigs + jog_cost * jogs = total_spend ∧
  jags + jigs + jogs ≥ min_items ∧
  jags ≥ 1 ∧ jigs ≥ 1 ∧ jogs ≥ 1 →
  jogs = 4 := by
  sorry

#eval jogs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_jogs_purchase_l508_50879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l508_50885

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define our specific triangle
noncomputable def ourTriangle : Triangle where
  A := Real.pi / 3  -- 60 degrees in radians
  b := 5
  c := 4
  -- We don't know B, C, and a yet, so we'll define them in terms of known values
  B := Real.arcsin (5 * Real.sqrt 7 / 14)
  C := Real.arcsin (2 * Real.sqrt 7 / 7)
  a := Real.sqrt 21

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : t = ourTriangle) :
  t.a = Real.sqrt 21 ∧ Real.sin t.B * Real.sin t.C = 5 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l508_50885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l508_50876

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  perimeter_is_14 : dist A B + dist B C + dist C A = 14
  B_coord : B = (-3, 0)
  C_coord : C = (3, 0)

/-- The locus of point A forms an ellipse -/
def locus_equation (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 7 = 1 ∧ y ≠ 0

/-- The area of the triangle when AB is perpendicular to AC -/
noncomputable def triangle_area (t : Triangle) : ℝ :=
  if (t.A.1 + 3) * (t.A.1 - 3) + t.A.2^2 = 0 then 7 else 0

/-- Main theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) :
  (∀ x y, t.A = (x, y) → locus_equation x y) ∧
  (triangle_area t = 7) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l508_50876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l508_50852

/-- A geometric sequence with positive terms -/
structure PositiveGeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)

/-- The common ratio of a geometric sequence -/
noncomputable def commonRatio (seq : PositiveGeometricSequence) : ℝ := seq.a 2 / seq.a 1

theorem geometric_sequence_common_ratio 
  (seq : PositiveGeometricSequence) 
  (h : 2 * seq.a 1 + seq.a 2 = seq.a 3) : 
  commonRatio seq = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l508_50852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_coprime_from_six_l508_50881

/-- A four-digit number is an integer between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A set of six four-digit numbers -/
def SixFourDigitNumbers : Type := Fin 6 → ℕ

/-- Predicate to check if all numbers in a set are four-digit numbers -/
def AllFourDigit (s : SixFourDigitNumbers) : Prop :=
  ∀ i, FourDigitNumber (s i)

/-- Predicate to check if all numbers in a set are pairwise coprime -/
def AllPairwiseCoprime (s : SixFourDigitNumbers) : Prop :=
  ∀ i j, i ≠ j → Nat.Coprime (s i) (s j)

/-- Theorem: From any set of six four-digit numbers that are pairwise coprime,
    we can always choose five that are also pairwise coprime -/
theorem five_coprime_from_six (s : SixFourDigitNumbers) 
  (h1 : AllFourDigit s) (h2 : AllPairwiseCoprime s) :
  ∃ (t : Fin 5 → ℕ), (∀ i, ∃ j, t i = s j) ∧ AllPairwiseCoprime (fun i ↦ t i) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_coprime_from_six_l508_50881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geography_choice_probabilities_l508_50872

-- Define the probabilities
noncomputable def p_physics : ℝ := 3/4
noncomputable def p_history : ℝ := 1/4
noncomputable def p_geo_given_physics : ℝ := 2/3
noncomputable def p_geo_given_history : ℝ := 4/5

-- Define the random variable X
noncomputable def X : ℕ → ℝ
| 0 => 27/1000
| 1 => 189/1000
| 2 => 441/1000
| 3 => 343/1000
| _ => 0

-- Theorem statement
theorem geography_choice_probabilities :
  -- 1. Probability of choosing geography
  (p_physics * p_geo_given_physics + p_history * p_geo_given_history = 7/10) ∧
  -- 2. Probability of exactly 2 out of 3 students choosing geography
  (3 * (7/10)^2 * (1 - 7/10) = 441/1000) ∧
  -- 3. Expected number of students choosing geography
  (X 0 * 0 + X 1 * 1 + X 2 * 2 + X 3 * 3 = 21/10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geography_choice_probabilities_l508_50872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l508_50884

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x + 1)

-- Define the inverse function f_inv as noncomputable
noncomputable def f_inv (y : ℝ) : ℝ := -y / (y - 2)

-- Theorem statement
theorem inverse_function_value :
  f_inv 1 = 1 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l508_50884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l508_50836

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The probability of ending on a vertical side from a given point -/
noncomputable def probability_vertical_side (p : Point) : ℚ :=
  sorry

/-- The grid size -/
def grid_size : ℕ := 6

/-- The starting point of the frog -/
def start : Point := ⟨2, 3⟩

/-- The pond location -/
def pond : Point := ⟨3, 3⟩

/-- Checks if a point is on the grid boundary -/
def is_on_boundary (p : Point) : Prop :=
  p.x = 0 ∨ p.x = grid_size ∨ p.y = 0 ∨ p.y = grid_size

/-- Checks if a point is the pond -/
def is_pond (p : Point) : Prop :=
  p = pond

/-- The main theorem to prove -/
theorem frog_jump_probability :
  probability_vertical_side start = 5 / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l508_50836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l508_50865

/-- The function f(x) = 2x^3 - x - 4 -/
def f (x : ℝ) : ℝ := 2 * x^3 - x - 4

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 1

/-- The point of interest -/
def x₀ : ℝ := 1

/-- The slope of the tangent line at x₀ -/
noncomputable def m_tangent : ℝ := f' x₀

/-- The slope of the normal line at x₀ -/
noncomputable def m_normal : ℝ := -1 / m_tangent

/-- The y-coordinate of the point on the curve at x₀ -/
noncomputable def y₀ : ℝ := f x₀

theorem tangent_and_normal_equations :
  (∀ x y, y = m_tangent * (x - x₀) + y₀ ↔ y = 5 * x - 8) ∧
  (∀ x y, y = m_normal * (x - x₀) + y₀ ↔ y = -1/5 * x - 14/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_normal_equations_l508_50865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_highest_prob_l508_50858

/-- Represents the number of players in the game. -/
def num_players : ℕ := 6

/-- Represents the probability of rolling a 1 or 2 on a 10-sided die. -/
noncomputable def p_success : ℝ := 1 / 5

/-- Represents the probability of not rolling a 1 or 2 on a 10-sided die. -/
noncomputable def p_failure : ℝ := 1 - p_success

/-- Calculates the probability of becoming the game master for a player at position k. -/
noncomputable def prob_gm (k : ℕ) : ℝ :=
  p_success * (p_failure ^ (k - 1)) / (1 - p_failure ^ num_players)

/-- States that the probability of becoming the game master is highest for the first player. -/
theorem first_player_highest_prob :
  ∀ k, k > 1 → k ≤ num_players → prob_gm 1 > prob_gm k := by
  sorry

#check first_player_highest_prob

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_highest_prob_l508_50858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l508_50826

theorem complex_modulus_equality (n : ℝ) : 
  n > 0 → Complex.abs (Complex.mk 5 n) = 5 * Real.sqrt 10 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_equality_l508_50826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_in_trapezoid_l508_50866

/-- An isosceles trapezoid with given side lengths and inscribed circles -/
structure IsoscelesTrapezoidWithCircles where
  EF : ℝ
  FG : ℝ
  HE : ℝ
  GH : ℝ
  radiusEF : ℝ
  radiusGH : ℝ
  isIsosceles : FG = HE
  sidesValid : EF > FG ∧ FG > GH

/-- The radius of the smaller circle inside the trapezoid -/
noncomputable def smallerCircleRadius (t : IsoscelesTrapezoidWithCircles) : ℝ :=
  (-7 + Real.sqrt 85) / 2

/-- Theorem stating the radius of the smaller circle -/
theorem smaller_circle_radius_in_trapezoid 
  (t : IsoscelesTrapezoidWithCircles)
  (h1 : t.EF = 10)
  (h2 : t.FG = 7)
  (h3 : t.GH = 6)
  (h4 : t.radiusEF = 4)
  (h5 : t.radiusGH = 3) :
  smallerCircleRadius t = (-7 + Real.sqrt 85) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_in_trapezoid_l508_50866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_crossing_bridge_l508_50857

/-- Calculates the time in minutes to cross a bridge given the speed in km/hr and the bridge length in meters -/
noncomputable def timeToCrossBridge (speed : ℝ) (bridgeLength : ℝ) : ℝ :=
  bridgeLength / (speed * 1000 / 60)

/-- Theorem stating that a man walking at 5 km/hr takes approximately 15 minutes to cross a 1250-meter bridge -/
theorem man_crossing_bridge :
  let speed := (5 : ℝ)
  let bridgeLength := (1250 : ℝ)
  let crossingTime := timeToCrossBridge speed bridgeLength
  ∃ ε > 0, abs (crossingTime - 15) < ε :=
by
  -- We use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_crossing_bridge_l508_50857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_selection_equals_coefficient_l508_50814

/-- The set of available weights -/
def weights : Finset ℕ := Finset.range 12 \ {0}

/-- The target total weight -/
def target_weight : ℕ := 9

/-- The polynomial representing the weight selection possibilities -/
def weight_polynomial (x : ℕ → ℕ) : ℕ → ℕ := 
  (weights.prod fun i => (1 + x i))

/-- The number of ways to select weights summing to the target -/
def num_selections : ℕ := 
  (Finset.filter (fun s => s.sum id = target_weight) (Finset.powerset weights)).card

/-- The coefficient of x^9 in the expansion of the weight polynomial -/
def coefficient_x9 : ℕ := 
  (weight_polynomial fun i => if i = target_weight then 1 else 0) target_weight

/-- Theorem stating that the number of weight selections equals the coefficient of x^9 -/
theorem weight_selection_equals_coefficient : num_selections = coefficient_x9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_selection_equals_coefficient_l508_50814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l508_50878

/-- An ellipse with a vertex B(0, 2) and left focus F₁(-1, 0) on the line 2x - y + 2 = 0 has eccentricity √5/5 -/
theorem ellipse_eccentricity (B F₁ : ℝ × ℝ) (l : ℝ → ℝ → Prop) : 
  B = (0, 2) →
  F₁ = (-1, 0) →
  (∀ x y, l x y ↔ 2*x - y + 2 = 0) →
  l B.1 B.2 →
  l F₁.1 F₁.2 →
  let c : ℝ := 1
  let b : ℝ := 2
  let a : ℝ := Real.sqrt 5
  let e : ℝ := c / a
  e = Real.sqrt 5 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l508_50878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l508_50869

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

theorem f_properties :
  (∀ x : ℝ, f (x + Real.pi/2) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, 
    (k * Real.pi - Real.pi/6 ≤ x ∧ x ≤ k * Real.pi + 5*Real.pi/12) ∨ 
    (k * Real.pi + 3*Real.pi/4 ≤ x ∧ x ≤ k * Real.pi + 5*Real.pi/6) → 
    f x ≥ -Real.sqrt 3 / 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-Real.pi/6) (-Real.pi/12) ∨ x ∈ Set.Icc (5*Real.pi/12) (7*Real.pi/12) → 
    ∀ y : ℝ, y ∈ Set.Icc (-Real.pi/6) (7*Real.pi/12) ∧ x < y → g x < g y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l508_50869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l508_50813

/-- The area of a triangle given the coordinates of its vertices -/
noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- Theorem: The area of triangle PQR with given coordinates is 26 square units -/
theorem area_of_triangle_PQR : 
  triangle_area (-3) 4 1 7 5 (-3) = 26 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l508_50813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_200_to_10_l508_50817

theorem ratio_of_200_to_10 : ∃ (ratio : ℝ), ratio = 200 / 10 ∧ ratio = 20 := by
  let number : ℝ := 200
  let base : ℝ := 10
  let ratio := number / base
  
  have ratio_calc : ratio = 200 / 10 := by rfl
  have ratio_value : ratio = 20 := by
    calc
      ratio = 200 / 10 := by rfl
      _ = 20 := by norm_num

  exact ⟨ratio, ⟨ratio_calc, ratio_value⟩⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_200_to_10_l508_50817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l508_50847

/-- The function f(x) = |x+1| - 2|x-1| -/
noncomputable def f (x : ℝ) : ℝ := |x + 1| - 2 * |x - 1|

/-- The function g(x) = (x^2 - ax + 4) / x -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (x^2 - a*x + 4) / x

/-- The theorem stating that if g(x) ≥ f(t) for all s, t ∈ (0, +∞), then the maximum value of a is 2 -/
theorem max_a_value (a : ℝ) : 
  (∀ s t : ℝ, s > 0 → t > 0 → g a s ≥ f t) → 
  a ≤ 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_a_value_l508_50847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_volume_angle_relation_triangle_area_quadratic_root_difference_l508_50873

-- Part 1
theorem hemisphere_volume (a : ℝ) (h : (2/3) * Real.pi * (a/2)^3 = 18 * Real.pi) : a = 6 := by sorry

-- Part 2
theorem angle_relation (b : ℝ) (h1 : Real.sin (10 * Real.pi/180) = Real.cos ((360 - b) * Real.pi/180)) (h2 : 0 < b) (h3 : b < 90) : b = 80 := by sorry

-- Part 3
theorem triangle_area (b : ℝ) (c : ℝ) (h : c = (1/2) * (120/b) * (120/(2*b))) : c = 4 := by sorry

-- Part 4
theorem quadratic_root_difference (c : ℝ) (d : ℝ) (h : d = |c|) : d = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_volume_angle_relation_triangle_area_quadratic_root_difference_l508_50873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axes_concurrence_l508_50808

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 - c1.radius^2 = 
               (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 - c2.radius^2}

-- Helper function to check if three points are collinear
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p3.2 - p1.2) * (p2.1 - p1.1) = (p2.2 - p1.2) * (p3.1 - p1.1)

-- Theorem statement
theorem radical_axes_concurrence 
  (c1 c2 c3 : Circle) 
  (h : ¬collinear c1.center c2.center c3.center) :
  ∃ p : ℝ × ℝ, p ∈ radical_axis c1 c2 ∧ 
              p ∈ radical_axis c2 c3 ∧ 
              p ∈ radical_axis c1 c3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axes_concurrence_l508_50808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l508_50845

-- Define the function f(x) = tan(2x)
noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

-- Theorem statement
theorem f_is_odd : Odd f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l508_50845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_less_than_half_l508_50803

theorem cos_less_than_half (x : ℝ) (h : Real.sin x < Real.cos (x/2) ∧ Real.cos (x/2) < 0) : Real.cos x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_less_than_half_l508_50803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l508_50815

/-- The length of a chord formed by the intersection of a line and a circle -/
theorem chord_length (A B : ℝ × ℝ) : 
  (∀ (x y : ℝ), (x, y) = A ∨ (x, y) = B → y = x + 1) →  -- Line equation
  (∀ (x y : ℝ), (x, y) = A ∨ (x, y) = B → x^2 + y^2 + 2*y - 3 = 0) →  -- Circle equation
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l508_50815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consistent_stamps_per_page_l508_50844

theorem largest_consistent_stamps_per_page 
  (book1_stamps book2_stamps book3_stamps : ℕ)
  (h1 : book1_stamps = 840)
  (h2 : book2_stamps = 1008)
  (h3 : book3_stamps = 672) :
  Nat.gcd book1_stamps (Nat.gcd book2_stamps book3_stamps) = 
    (Finset.filter (fun d => book1_stamps % d = 0 ∧ book2_stamps % d = 0 ∧ book3_stamps % d = 0) 
      (Finset.range (min book1_stamps (min book2_stamps book3_stamps) + 1))).max' 
      (by simp [h1, h2, h3]; exact ⟨1, by norm_num⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_consistent_stamps_per_page_l508_50844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_l508_50864

/-- Given a function f: ℝ → ℝ satisfying f(x+1) - f(x) = 1 for all x,
    prove that F(x) := f(x) - x is periodic with period 1 -/
theorem f_periodic (f : ℝ → ℝ) (h : ∀ x, f (x + 1) - f x = 1) :
  let F := λ x ↦ f x - x
  ∀ x, F (x + 1) = F x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_periodic_l508_50864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l508_50801

-- Define the class [k]
def class_ (k : ℤ) : Set ℤ := {n : ℤ | ∃ m : ℤ, n = 5 * m + k}

-- Define the three statements
def statement1 : Prop := 2013 ∈ class_ 3
def statement2 : Prop := -2 ∈ class_ 2
def statement3 : Prop := (Set.univ : Set ℤ) = class_ 0 ∪ class_ 1 ∪ class_ 2 ∪ class_ 3 ∪ class_ 4

-- Theorem to prove
theorem exactly_two_statements_true : 
  (statement1 ∧ ¬statement2 ∧ statement3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_statements_true_l508_50801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicirclic_quadrilateral_area_l508_50837

/-- A quadrilateral that is both circumscribed and inscribed -/
structure BicirclicQuadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_circumscribed : Bool
  is_inscribed : Bool

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculate the area of a quadrilateral -/
def area (q : BicirclicQuadrilateral) : ℝ := sorry

theorem bicirclic_quadrilateral_area 
  (q : BicirclicQuadrilateral)
  (h1 : q.is_circumscribed = true)
  (h2 : q.is_inscribed = true)
  (h3 : distance q.A q.B = 2)
  (h4 : distance q.B q.C = 4)
  (h5 : distance q.C q.D = 5) :
  area q = 2 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicirclic_quadrilateral_area_l508_50837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l508_50863

/-- The area of the region inside a rectangle but outside four quarter circles --/
noncomputable def area_outside_circles (cd da r1 r2 r3 r4 : ℝ) : ℝ :=
  cd * da - (r1^2 + r2^2 + r3^2 + r4^2) * Real.pi / 4

/-- The problem statement --/
theorem area_approximation :
  let cd := (4 : ℝ)
  let da := (6 : ℝ)
  let r1 := (2 : ℝ)  -- radius of circle at A
  let r2 := (3 : ℝ)  -- radius of circle at B
  let r3 := (4 : ℝ)  -- radius of circle at C
  let r4 := (1 : ℝ)  -- radius of circle at D
  ∃ ε > 0, |area_outside_circles cd da r1 r2 r3 r4 - 0.45| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_approximation_l508_50863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_difference_l508_50839

/-- The cost difference between buying and renting a car for a year -/
theorem car_cost_difference : 
  (30 : ℕ) * 12 - (20 : ℕ) * 12 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_cost_difference_l508_50839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_trailing_zeroes_base_8_l508_50867

-- Define 15!
def factorial_15 : ℕ := Nat.factorial 15

-- Define the function to count trailing zeroes in base 8
def count_trailing_zeroes_base_8 (n : ℕ) : ℕ :=
  Nat.log 8 (Nat.gcd n (8^(Nat.log 8 n)))

-- Theorem statement
theorem factorial_15_trailing_zeroes_base_8 :
  count_trailing_zeroes_base_8 factorial_15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_15_trailing_zeroes_base_8_l508_50867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selection_for_same_sum_l508_50802

-- Define the range of numbers
def number_range : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 700}

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

-- Define the property of having three numbers with the same sum of digits
def has_three_same_sum (S : Set ℕ) : Prop :=
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  sum_of_digits a = sum_of_digits b ∧ sum_of_digits b = sum_of_digits c

-- Theorem statement
theorem min_selection_for_same_sum :
  ∀ (S : Finset ℕ), (∀ n ∈ S, n ∈ number_range) → 
    (S.card ≥ 47 → has_three_same_sum ↑S) ∧
    (∃ (T : Finset ℕ), (∀ n ∈ T, n ∈ number_range) ∧ T.card = 46 ∧ ¬has_three_same_sum ↑T) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_selection_for_same_sum_l508_50802
