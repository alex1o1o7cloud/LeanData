import Mathlib

namespace placard_distribution_l427_42707

theorem placard_distribution (total_placards : ℕ) (total_people : ℕ) 
  (h1 : total_placards = 4634) (h2 : total_people = 2317) :
  total_placards / total_people = 2 := by
  sorry

end placard_distribution_l427_42707


namespace tom_marble_pairs_l427_42793

/-- The number of distinct pairs of marbles Tom can choose -/
def distinct_pairs : ℕ := 8

/-- The number of red marbles Tom has -/
def red_marbles : ℕ := 1

/-- The number of blue marbles Tom has -/
def blue_marbles : ℕ := 1

/-- The number of yellow marbles Tom has -/
def yellow_marbles : ℕ := 4

/-- The number of green marbles Tom has -/
def green_marbles : ℕ := 2

/-- Theorem stating that the number of distinct pairs of marbles Tom can choose is 8 -/
theorem tom_marble_pairs :
  distinct_pairs = 8 :=
by sorry

end tom_marble_pairs_l427_42793


namespace nonnegative_fraction_implies_nonnegative_x_l427_42756

theorem nonnegative_fraction_implies_nonnegative_x (x : ℝ) :
  (x^2 + 2*x^4 - 3*x^5) / (x + 2*x^3 - 3*x^4) ≥ 0 → x ≥ 0 := by
  sorry

end nonnegative_fraction_implies_nonnegative_x_l427_42756


namespace standard_pairs_parity_l427_42775

/-- Represents the color of a square on the chessboard -/
inductive Color
| Red
| Blue

/-- Represents a chessboard -/
def Chessboard (m n : ℕ) := Fin m → Fin n → Color

/-- Counts the number of standard pairs on the chessboard -/
def count_standard_pairs (board : Chessboard m n) : ℕ := sorry

/-- Counts the number of blue squares on the edges (excluding corners) -/
def count_blue_edges (board : Chessboard m n) : ℕ := sorry

/-- The main theorem: The parity of standard pairs is equivalent to the parity of blue edge squares -/
theorem standard_pairs_parity (m n : ℕ) (h_m : m ≥ 3) (h_n : n ≥ 3) (board : Chessboard m n) :
  Even (count_standard_pairs board) ↔ Even (count_blue_edges board) := by sorry

end standard_pairs_parity_l427_42775


namespace pencil_cost_l427_42711

theorem pencil_cost (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 340)
  (eq2 : 3 * x + 6 * y = 264) : 
  y = 50 / 3 := by
  sorry

end pencil_cost_l427_42711


namespace watermelon_puzzle_l427_42790

theorem watermelon_puzzle (A B C : ℕ) 
  (h1 : C - (A + B) = 6)
  (h2 : (B + C) - A = 16)
  (h3 : (C + A) - B = 8) :
  A + B + C = 18 := by
sorry

end watermelon_puzzle_l427_42790


namespace geometric_sequence_first_term_l427_42774

theorem geometric_sequence_first_term 
  (a₅ a₆ : ℚ)
  (h₁ : a₅ = 48)
  (h₂ : a₆ = 64)
  : ∃ (a : ℚ), a₅ = a * (a₆ / a₅)^4 ∧ a = 243 / 16 := by
sorry

end geometric_sequence_first_term_l427_42774


namespace min_product_tangents_acute_triangle_l427_42748

theorem min_product_tangents_acute_triangle (α β γ : Real) 
  (h_acute : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α < π/2 ∧ β < π/2 ∧ γ < π/2) 
  (h_sum : α + β + γ = π) : 
  Real.tan α * Real.tan β * Real.tan γ ≥ Real.sqrt 27 ∧ 
  (Real.tan α * Real.tan β * Real.tan γ = Real.sqrt 27 ↔ α = π/3 ∧ β = π/3 ∧ γ = π/3) :=
sorry

end min_product_tangents_acute_triangle_l427_42748


namespace distinct_cuttings_count_l427_42747

/-- Represents a square grid --/
def Square (n : ℕ) := Fin n → Fin n → Bool

/-- Represents an L-shaped piece (corner) --/
structure LPiece where
  size : ℕ
  position : Fin 4 × Fin 4

/-- Represents a cutting of a 4x4 square --/
structure Cutting where
  lpieces : Fin 3 → LPiece
  small_square : Fin 4 × Fin 4

/-- Checks if two cuttings are distinct (considering rotations and reflections) --/
def is_distinct (c1 c2 : Cutting) : Bool := sorry

/-- Counts the number of distinct ways to cut a 4x4 square --/
def count_distinct_cuttings : ℕ := sorry

/-- The main theorem stating that there are 64 distinct ways to cut the 4x4 square --/
theorem distinct_cuttings_count : count_distinct_cuttings = 64 := by sorry

end distinct_cuttings_count_l427_42747


namespace angle_four_value_l427_42735

theorem angle_four_value (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180)
  (h2 : angle3 = angle4)
  (h3 : angle1 + 70 + 40 = 180)
  (h4 : angle2 + angle3 + angle4 = 180) :
  angle4 = 35 := by
sorry

end angle_four_value_l427_42735


namespace right_triangle_with_incircle_legs_l427_42751

/-- A right-angled triangle with an incircle touching the hypotenuse -/
structure RightTriangleWithIncircle where
  -- The lengths of the sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- The lengths of the segments of the hypotenuse
  ap : ℝ
  bp : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  hypotenuse : c = ap + bp
  incircle_property : ap = (a + b + c) / 2 - a ∧ bp = (a + b + c) / 2 - b
  -- Given values
  ap_value : ap = 12
  bp_value : bp = 5

/-- The main theorem -/
theorem right_triangle_with_incircle_legs 
  (triangle : RightTriangleWithIncircle) : 
  triangle.a = 8 ∧ triangle.b = 15 := by
  sorry

end right_triangle_with_incircle_legs_l427_42751


namespace product_of_integers_l427_42750

theorem product_of_integers (A B C D : ℕ+) : 
  (A : ℝ) + (B : ℝ) + (C : ℝ) + (D : ℝ) = 40 →
  (A : ℝ) + 3 = (B : ℝ) - 3 ∧ 
  (A : ℝ) + 3 = (C : ℝ) * 3 ∧ 
  (A : ℝ) + 3 = (D : ℝ) / 3 →
  (A : ℝ) * (B : ℝ) * (C : ℝ) * (D : ℝ) = 2666.25 := by
sorry

end product_of_integers_l427_42750


namespace smallest_AAB_l427_42701

/-- Represents a digit from 1 to 9 -/
def Digit := {n : ℕ | 1 ≤ n ∧ n ≤ 9}

/-- Represents a two-digit integer AB -/
def TwoDigitInt (A B : Digit) : ℕ := 10 * A + B

/-- Represents a three-digit integer AAB -/
def ThreeDigitInt (A B : Digit) : ℕ := 100 * A + 10 * A + B

theorem smallest_AAB : 
  ∃ (A B : Digit), 
    A ≠ B ∧ 
    (TwoDigitInt A B : ℚ) = (1 / 7 : ℚ) * (ThreeDigitInt A B : ℚ) ∧
    ThreeDigitInt A B = 332 ∧
    (∀ (A' B' : Digit), 
      A' ≠ B' → 
      (TwoDigitInt A' B' : ℚ) = (1 / 7 : ℚ) * (ThreeDigitInt A' B' : ℚ) → 
      ThreeDigitInt A' B' ≥ 332) := by
  sorry

end smallest_AAB_l427_42701


namespace dog_toy_cost_l427_42768

/-- The cost of dog toys with a "buy one get one half off" deal -/
theorem dog_toy_cost (regular_price : ℝ) (num_toys : ℕ) : regular_price = 12 → num_toys = 4 →
  let discounted_price := regular_price / 2
  let pair_price := regular_price + discounted_price
  let total_cost := (num_toys / 2 : ℝ) * pair_price
  total_cost = 36 := by
  sorry

end dog_toy_cost_l427_42768


namespace intersection_complement_theorem_l427_42740

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- State the theorem
theorem intersection_complement_theorem : 
  B ∩ (Set.compl A) = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end intersection_complement_theorem_l427_42740


namespace inequality_equivalence_l427_42726

theorem inequality_equivalence (x : ℝ) : 2 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 5 ↔ 7 / 3 < x ∧ x < 14 / 5 := by
  sorry

end inequality_equivalence_l427_42726


namespace min_value_of_sum_l427_42764

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (2 * y + 3) = 1 / 4) : 
  x + 3 * y ≥ 2 + 4 * Real.sqrt 3 := by
  sorry

end min_value_of_sum_l427_42764


namespace equation_solution_l427_42785

theorem equation_solution :
  ∃ x : ℝ, 3 * (16 : ℝ)^x + 37 * (36 : ℝ)^x = 26 * (81 : ℝ)^x ∧ x = (1/2 : ℝ) :=
by
  sorry

end equation_solution_l427_42785


namespace arrangement_exists_for_23_l427_42725

/-- Definition of the Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of an arrangement for P = 23 -/
theorem arrangement_exists_for_23 : ∃ (F : ℕ → ℤ), F 12 % 23 = 0 ∧
  (∀ n, F (n + 2) = 3 * F (n + 1) - F n) ∧ F 0 = 0 ∧ F 1 = 1 := by
  sorry


end arrangement_exists_for_23_l427_42725


namespace problem_solution_l427_42797

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 12) : 
  q = 6 + 2 * Real.sqrt 6 := by
sorry

end problem_solution_l427_42797


namespace expression_evaluation_l427_42739

theorem expression_evaluation :
  let x : ℝ := 3 + Real.sqrt 2
  (1 - 1 / (x + 3)) / ((x + 2) / (x^2 - 9)) = Real.sqrt 2 := by
  sorry

end expression_evaluation_l427_42739


namespace system_solution_l427_42798

theorem system_solution :
  ∃ (x y z t : ℂ),
    x + y = 10 ∧
    z + t = 5 ∧
    x * y = z * t ∧
    x^3 + y^3 + z^3 + t^3 = 1080 ∧
    x = 5 + Real.sqrt 17 ∧
    y = 5 - Real.sqrt 17 ∧
    z = (5 + Complex.I * Real.sqrt 7) / 2 ∧
    t = (5 - Complex.I * Real.sqrt 7) / 2 :=
by sorry

end system_solution_l427_42798


namespace problem_solution_l427_42759

theorem problem_solution : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) * Nat.factorial 7 = 
  (5^128 - 4^128) * 5040 := by
  sorry

end problem_solution_l427_42759


namespace fraction_equality_l427_42733

/-- Given two integers A and B satisfying the equation for all real x except 0, 3, and roots of x^2 + 2x + 1 = 0, prove that B/A = 0 -/
theorem fraction_equality (A B : ℤ) 
  (h : ∀ x : ℝ, x ≠ 0 → x ≠ 3 → x^2 + 2*x + 1 ≠ 0 → 
    (A / (x - 3) : ℝ) + (B / (x^2 + 2*x + 1) : ℝ) = (x^3 - x^2 + 3*x + 1) / (x^3 - x - 3)) : 
  (B : ℚ) / A = 0 := by
  sorry

end fraction_equality_l427_42733


namespace foil_covered_prism_width_l427_42721

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular prism -/
def volume (d : PrismDimensions) : ℝ := d.length * d.width * d.height

/-- Theorem: The width of the foil-covered prism is 10 inches -/
theorem foil_covered_prism_width : 
  ∀ (inner : PrismDimensions),
    volume inner = 128 →
    inner.width = 2 * inner.length →
    inner.width = 2 * inner.height →
    ∃ (outer : PrismDimensions),
      outer.length = inner.length + 2 ∧
      outer.width = inner.width + 2 ∧
      outer.height = inner.height + 2 ∧
      outer.width = 10 := by
  sorry

end foil_covered_prism_width_l427_42721


namespace inverse_evaluation_l427_42741

def problem (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : Prop :=
  Function.Bijective f ∧
  f 4 = 7 ∧
  f 6 = 3 ∧
  f 3 = 6 ∧
  f_inv ∘ f = id ∧
  f ∘ f_inv = id

theorem inverse_evaluation (f : ℝ → ℝ) (f_inv : ℝ → ℝ) 
  (h : problem f f_inv) : 
  f_inv (f_inv 6 + f_inv 7) = 4 := by
  sorry

end inverse_evaluation_l427_42741


namespace absolute_value_plus_power_minus_sqrt_l427_42728

theorem absolute_value_plus_power_minus_sqrt : |-2| + 2023^0 - Real.sqrt 4 = 1 := by
  sorry

end absolute_value_plus_power_minus_sqrt_l427_42728


namespace morning_bikes_count_l427_42794

/-- The number of bikes sold in the morning -/
def morning_bikes : ℕ := 19

/-- The number of bikes sold in the afternoon -/
def afternoon_bikes : ℕ := 27

/-- The number of bike clamps given with each bike -/
def clamps_per_bike : ℕ := 2

/-- The total number of bike clamps given away -/
def total_clamps : ℕ := 92

/-- Theorem stating that the number of bikes sold in the morning is 19 -/
theorem morning_bikes_count : 
  morning_bikes = 19 ∧ 
  clamps_per_bike * (morning_bikes + afternoon_bikes) = total_clamps := by
  sorry

end morning_bikes_count_l427_42794


namespace stock_price_calculation_l427_42744

theorem stock_price_calculation (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) :
  initial_price = 120 →
  first_year_increase = 0.8 →
  second_year_decrease = 0.3 →
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 151.2 :=
by sorry

end stock_price_calculation_l427_42744


namespace ruel_stamps_l427_42705

/-- The number of stamps in a book of 10 stamps -/
def stamps_per_book_10 : ℕ := 10

/-- The number of stamps in a book of 15 stamps -/
def stamps_per_book_15 : ℕ := 15

/-- The number of books with 10 stamps -/
def books_10 : ℕ := 4

/-- The number of books with 15 stamps -/
def books_15 : ℕ := 6

/-- The total number of stamps Ruel has -/
def total_stamps : ℕ := books_10 * stamps_per_book_10 + books_15 * stamps_per_book_15

theorem ruel_stamps : total_stamps = 130 := by
  sorry

end ruel_stamps_l427_42705


namespace employee_price_calculation_l427_42762

/-- Calculates the employee's price for a video recorder given the wholesale cost, markup percentage, and employee discount percentage. -/
theorem employee_price_calculation 
  (wholesale_cost : ℝ) 
  (markup_percentage : ℝ) 
  (employee_discount_percentage : ℝ) : 
  wholesale_cost = 200 ∧ 
  markup_percentage = 20 ∧ 
  employee_discount_percentage = 30 → 
  wholesale_cost * (1 + markup_percentage / 100) * (1 - employee_discount_percentage / 100) = 168 := by
  sorry

#check employee_price_calculation

end employee_price_calculation_l427_42762


namespace opposite_expressions_imply_y_value_l427_42767

theorem opposite_expressions_imply_y_value :
  ∀ y : ℚ, (4 * y + 8) = -(8 * y - 7) → y = -1/12 := by
  sorry

end opposite_expressions_imply_y_value_l427_42767


namespace fourth_hour_highest_speed_l427_42787

def distance_traveled : Fin 7 → ℝ
| 0 => 70
| 1 => 95
| 2 => 85
| 3 => 100
| 4 => 90
| 5 => 85
| 6 => 75

def average_speed (hour : Fin 7) : ℝ := distance_traveled hour

theorem fourth_hour_highest_speed :
  ∀ (hour : Fin 7), average_speed 3 ≥ average_speed hour :=
by sorry

end fourth_hour_highest_speed_l427_42787


namespace sin_even_function_phi_l427_42738

theorem sin_even_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 6)) →
  (0 < φ) →
  (φ < π / 2) →
  (∀ x, f (x - φ) = f (φ - x)) →
  φ = π / 3 := by
sorry

end sin_even_function_phi_l427_42738


namespace f_neg_two_eq_neg_fourteen_l427_42779

/-- A cubic function f(x) with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 4

/-- Theorem stating that f(-2) = -14 given the conditions -/
theorem f_neg_two_eq_neg_fourteen (a b : ℝ) :
  (f a b 2 = 6) → (f a b (-2) = -14) := by
  sorry

end f_neg_two_eq_neg_fourteen_l427_42779


namespace problem_proof_l427_42786

theorem problem_proof (a b : ℝ) (h1 : a > b) (h2 : b > 1/a) (h3 : 1/a > 0) : 
  (a + b > 2) ∧ (a > 1) ∧ (a - 1/b > b - 1/a) := by sorry

end problem_proof_l427_42786


namespace point_moved_right_l427_42784

def move_right (x y d : ℝ) : ℝ × ℝ := (x + d, y)

theorem point_moved_right :
  let A : ℝ × ℝ := (2, -1)
  let d : ℝ := 3
  move_right A.1 A.2 d = (5, -1) := by sorry

end point_moved_right_l427_42784


namespace student_count_l427_42710

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 21)
  (h2 : rank_from_left = 11) :
  rank_from_right + rank_from_left - 1 = 31 := by
  sorry

end student_count_l427_42710


namespace solution_pairs_l427_42769

theorem solution_pairs : 
  ∀ x y : ℝ, 
    (x^2 + y^2 + x + y = x*y*(x + y) - 10/27 ∧ 
     |x*y| ≤ 25/9) ↔ 
    ((x = -1/3 ∧ y = -1/3) ∨ (x = 5/3 ∧ y = 5/3)) := by
  sorry

end solution_pairs_l427_42769


namespace curves_intersect_once_l427_42776

/-- Two curves intersect at exactly one point -/
def intersect_once (f g : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, f x = g x

/-- The first curve -/
def curve1 (b : ℝ) (x : ℝ) : ℝ := b * x^2 - 2 * x + 5

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := 3 * x + 4

/-- The theorem stating the condition for the curves to intersect at exactly one point -/
theorem curves_intersect_once :
  ∀ b : ℝ, intersect_once (curve1 b) curve2 ↔ b = 25/4 := by sorry

end curves_intersect_once_l427_42776


namespace product_of_integers_l427_42719

theorem product_of_integers (a b : ℕ+) : 
  a + b = 30 → 2 * a * b + 12 * a = 3 * b + 240 → a * b = 255 := by
  sorry

end product_of_integers_l427_42719


namespace distance_AB_l427_42788

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (t, -Real.sqrt 3 * t)

noncomputable def curve_C1 (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 1 + Real.sin θ)

noncomputable def curve_C2 (θ : ℝ) : ℝ := 4 * Real.sin (θ - Real.pi / 6)

def intersection_OA (t : ℝ) : Prop :=
  ∃ θ : ℝ, line_l t = curve_C1 θ

def intersection_OB (t : ℝ) : Prop :=
  ∃ θ : ℝ, line_l t = (curve_C2 θ * Real.cos θ, curve_C2 θ * Real.sin θ)

theorem distance_AB :
  ∀ t₁ t₂ : ℝ, intersection_OA t₁ → intersection_OB t₂ →
    Real.sqrt ((t₂ - t₁)^2 + (-Real.sqrt 3 * t₂ + Real.sqrt 3 * t₁)^2) = 4 - Real.sqrt 3 :=
sorry

end distance_AB_l427_42788


namespace tangent_slope_implies_a_l427_42715

/-- Given a quadratic function f(x) = ax^2 + 3x - 2, 
    if the slope of its tangent line at x = 2 is 7, then a = 1 -/
theorem tangent_slope_implies_a (a : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 + 3 * x - 2
  let f' : ℝ → ℝ := λ x => 2 * a * x + 3
  f' 2 = 7 → a = 1 := by
sorry

end tangent_slope_implies_a_l427_42715


namespace coefficient_x6_in_expansion_l427_42713

theorem coefficient_x6_in_expansion : 
  let expansion := (1 + X : Polynomial ℤ)^6 * (1 - X : Polynomial ℤ)^6
  (expansion.coeff 6) = -20 := by sorry

end coefficient_x6_in_expansion_l427_42713


namespace min_value_expression_l427_42777

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 3) : 
  ∃ (min : ℝ), min = 16/9 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
    (x + y) / (x * y * z) ≥ min := by
  sorry

end min_value_expression_l427_42777


namespace championship_completion_impossible_l427_42791

/-- Represents a chess game between two players -/
structure Game where
  player1 : Nat
  player2 : Nat
  deriving Repr

/-- Represents the state of the chess championship -/
structure ChampionshipState where
  numPlayers : Nat
  gamesPlayed : List Game
  deriving Repr

/-- Checks if the championship rules are followed -/
def rulesFollowed (state : ChampionshipState) : Prop :=
  ∀ p1 p2, p1 < state.numPlayers → p2 < state.numPlayers → p1 ≠ p2 →
    let gamesPlayedByP1 := (state.gamesPlayed.filter (λ g => g.player1 = p1 ∨ g.player2 = p1)).length
    let gamesPlayedByP2 := (state.gamesPlayed.filter (λ g => g.player1 = p2 ∨ g.player2 = p2)).length
    (gamesPlayedByP1 : Int) - gamesPlayedByP2 ≤ 1 ∧ gamesPlayedByP2 - gamesPlayedByP1 ≤ 1

/-- Checks if the championship is complete -/
def isComplete (state : ChampionshipState) : Prop :=
  state.gamesPlayed.length = state.numPlayers * (state.numPlayers - 1) / 2

/-- Theorem: There exists a championship state that follows the rules but cannot be completed -/
theorem championship_completion_impossible : ∃ (state : ChampionshipState), 
  rulesFollowed state ∧ ¬∃ (finalState : ChampionshipState), 
    finalState.numPlayers = state.numPlayers ∧ 
    state.gamesPlayed ⊆ finalState.gamesPlayed ∧ 
    rulesFollowed finalState ∧ 
    isComplete finalState :=
sorry

end championship_completion_impossible_l427_42791


namespace line_passes_through_fixed_point_l427_42742

/-- The line equation passing through a fixed point for all real m -/
def line_equation (m x y : ℝ) : Prop :=
  (m + 2) * x + (m - 1) * y - 3 = 0

/-- The theorem stating that the line passes through (1, -1) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m 1 (-1) :=
by sorry

end line_passes_through_fixed_point_l427_42742


namespace infinite_prime_divisors_l427_42730

/-- A sequence of positive integers where no term divides another -/
def NonDivisibleSequence (a : ℕ → ℕ) : Prop :=
  ∀ i j, i ≠ j → ¬(a i ∣ a j)

/-- The set of primes dividing at least one term of the sequence -/
def PrimeDivisorsSet (a : ℕ → ℕ) : Set ℕ :=
  {p : ℕ | Nat.Prime p ∧ ∃ i, p ∣ a i}

theorem infinite_prime_divisors (a : ℕ → ℕ) 
    (h : NonDivisibleSequence a) : Set.Infinite (PrimeDivisorsSet a) := by
  sorry

end infinite_prime_divisors_l427_42730


namespace circle_inequality_l427_42799

theorem circle_inequality (a b c d : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a * b + c * d = 1)
  (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1)
  (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) :
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₄ + b * x₃ + c * x₂ + d * x₁)^2 
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) :=
by sorry

end circle_inequality_l427_42799


namespace triangle_sine_product_inequality_l427_42720

theorem triangle_sine_product_inequality (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 / 8 := by
  sorry

end triangle_sine_product_inequality_l427_42720


namespace room_breadth_calculation_l427_42771

theorem room_breadth_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 18 →
  carpet_width = 0.75 →
  carpet_cost_per_meter = 4.50 →
  total_cost = 810 →
  (total_cost / carpet_cost_per_meter) * carpet_width / room_length = 7.5 := by
  sorry

end room_breadth_calculation_l427_42771


namespace max_value_of_one_minus_sin_l427_42702

theorem max_value_of_one_minus_sin (x : ℝ) : 
  ∃ (M : ℝ), M = 2 ∧ ∀ x, (1 - Real.sin x) ≤ M ∧ ∃ x₀, (1 - Real.sin x₀) = M :=
sorry

end max_value_of_one_minus_sin_l427_42702


namespace range_of_f_l427_42782

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  ∀ y : ℝ, y ≠ 1 → ∃ x : ℝ, x ≠ -2 ∧ f x = y :=
sorry

end range_of_f_l427_42782


namespace function_equality_l427_42731

theorem function_equality (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x + y) + y ≤ f (f (f x))) → 
  ∃ c : ℝ, ∀ x : ℝ, f x = c - x :=
sorry

end function_equality_l427_42731


namespace share_calculation_l427_42792

theorem share_calculation (total : ℕ) (a b c : ℕ) : 
  total = 770 →
  a = b + 40 →
  c = a + 30 →
  total = a + b + c →
  b = 220 := by
sorry

end share_calculation_l427_42792


namespace work_done_circular_path_l427_42753

/-- The work done by a force field on a mass point moving along a circular path -/
theorem work_done_circular_path (m a : ℝ) (h : a > 0) : 
  let force (x y : ℝ) := (x + y, -x)
  let path (t : ℝ) := (a * Real.cos t, -a * Real.sin t)
  let work := ∫ t in (0)..(2 * Real.pi), 
    m * (force (path t).1 (path t).2).1 * (-a * Real.sin t) + 
    m * (force (path t).1 (path t).2).2 * (-a * Real.cos t)
  work = 0 :=
sorry

end work_done_circular_path_l427_42753


namespace candy_bar_total_cost_l427_42765

/-- The cost of a candy bar in dollars -/
def candy_bar_cost : ℕ := 3

/-- The number of candy bars bought -/
def number_of_candy_bars : ℕ := 2

/-- The total cost of candy bars -/
def total_cost : ℕ := candy_bar_cost * number_of_candy_bars

theorem candy_bar_total_cost : total_cost = 6 := by
  sorry

end candy_bar_total_cost_l427_42765


namespace product_of_hash_operations_l427_42722

-- Define the # operation
def hash (a b : ℚ) : ℚ := a + a / b

-- Theorem statement
theorem product_of_hash_operations :
  let x := hash 8 3
  let y := hash 5 4
  x * y = 200 / 3 := by sorry

end product_of_hash_operations_l427_42722


namespace quadrilateral_properties_l427_42754

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : Point)

-- Define properties of a quadrilateral
def has_one_pair_parallel_sides (q : Quadrilateral) : Prop := sorry
def has_one_pair_equal_sides (q : Quadrilateral) : Prop := sorry
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- The main theorem
theorem quadrilateral_properties :
  (∃ q : Quadrilateral, has_one_pair_parallel_sides q ∧ has_one_pair_equal_sides q ∧ ¬is_parallelogram q) ∧
  (∀ q : Quadrilateral, is_parallelogram q → has_one_pair_parallel_sides q ∧ has_one_pair_equal_sides q) :=
sorry

end quadrilateral_properties_l427_42754


namespace larger_solution_of_quadratic_l427_42737

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 11*x - 42 = 0 → 
  (∃ y : ℝ, y ≠ x ∧ y^2 - 11*y - 42 = 0) → 
  (x ≤ 14 ∧ (∀ z : ℝ, z^2 - 11*z - 42 = 0 → z ≤ 14)) :=
sorry

end larger_solution_of_quadratic_l427_42737


namespace x_wins_more_probability_l427_42712

/-- Represents a soccer tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (win_probability : ℚ)

/-- Represents the result of the tournament for two specific teams --/
inductive TournamentResult
  | XWinsMore
  | YWinsMore
  | Tie

/-- The probability of team X finishing with more points than team Y --/
def prob_X_wins_more (t : Tournament) : ℚ :=
  sorry

/-- The main theorem stating the probability of team X finishing with more points than team Y --/
theorem x_wins_more_probability (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.win_probability = 1/2) : 
  prob_X_wins_more t = 1/2 :=
sorry

end x_wins_more_probability_l427_42712


namespace valid_pairings_count_l427_42796

def num_bowls : ℕ := 6
def num_glasses : ℕ := 6

theorem valid_pairings_count :
  (num_bowls * num_glasses : ℕ) = 36 := by
  sorry

end valid_pairings_count_l427_42796


namespace inequality_proof_l427_42773

theorem inequality_proof (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  1 / Real.sqrt (1 + x^2) + 1 / Real.sqrt (1 + y^2) ≤ 2 / Real.sqrt (1 + x*y) :=
by sorry

end inequality_proof_l427_42773


namespace sqrt_3_times_612_times_3_and_half_log_squared_difference_plus_log_l427_42780

-- Part 1
theorem sqrt_3_times_612_times_3_and_half (x : ℝ) :
  x = Real.sqrt 3 * 612 * (3 + 3/2) → x = 3 := by sorry

-- Part 2
theorem log_squared_difference_plus_log (x : ℝ) :
  x = (Real.log 5 / Real.log 10)^2 - (Real.log 2 / Real.log 10)^2 + (Real.log 4 / Real.log 10) → x = 1 := by sorry

end sqrt_3_times_612_times_3_and_half_log_squared_difference_plus_log_l427_42780


namespace only_C_has_inverse_l427_42708

-- Define the set of graph labels
inductive GraphLabel
  | A | B | C | D | E

-- Define a predicate for functions that have inverses
def has_inverse (g : GraphLabel) : Prop :=
  match g with
  | GraphLabel.C => True
  | _ => False

-- Theorem statement
theorem only_C_has_inverse :
  ∀ g : GraphLabel, has_inverse g ↔ g = GraphLabel.C :=
by sorry

end only_C_has_inverse_l427_42708


namespace expression_value_at_two_l427_42703

theorem expression_value_at_two :
  let a : ℝ := 2
  (2 * a⁻¹ + 3 * a^2) / a = 13/2 :=
by sorry

end expression_value_at_two_l427_42703


namespace gcd_of_powers_of_seven_l427_42795

theorem gcd_of_powers_of_seven : Nat.gcd (7^11 + 1) (7^11 + 7^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_seven_l427_42795


namespace cos_squared_pi_sixth_plus_alpha_half_l427_42757

theorem cos_squared_pi_sixth_plus_alpha_half (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end cos_squared_pi_sixth_plus_alpha_half_l427_42757


namespace principal_is_600_l427_42700

/-- Proves that given the conditions of the problem, the principal amount is 600 --/
theorem principal_is_600 (P R : ℝ) (h : (P * (R + 4) * 6) / 100 - (P * R * 6) / 100 = 144) : P = 600 := by
  sorry

#check principal_is_600

end principal_is_600_l427_42700


namespace sum_of_ages_l427_42709

/-- The sum of Jed and Matt's present ages given their age relationship and Jed's future age -/
theorem sum_of_ages (jed_age matt_age : ℕ) : 
  jed_age = matt_age + 10 →  -- Jed is 10 years older than Matt
  jed_age + 10 = 25 →        -- In 10 years, Jed will be 25 years old
  jed_age + matt_age = 20 :=  -- The sum of their present ages is 20
by sorry

end sum_of_ages_l427_42709


namespace cubic_equation_property_l427_42706

theorem cubic_equation_property (p q : ℝ) : 
  (p * 3^3 + q * 3 + 1 = 2018) → (p * (-3)^3 + q * (-3) + 1 = -2016) := by
  sorry

end cubic_equation_property_l427_42706


namespace wood_length_proof_l427_42743

/-- The initial length of the wood Tom cut. -/
def initial_length : ℝ := 143

/-- The length cut off from the initial piece of wood. -/
def cut_length : ℝ := 25

/-- The original length of other boards before cutting. -/
def other_boards_original : ℝ := 125

/-- The length cut off from other boards. -/
def other_boards_cut : ℝ := 7

theorem wood_length_proof :
  initial_length - cut_length > other_boards_original - other_boards_cut ∧
  initial_length = 143 := by
  sorry

#check wood_length_proof

end wood_length_proof_l427_42743


namespace f_at_three_l427_42761

def f (x : ℝ) : ℝ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem f_at_three : f 3 = 181 := by
  sorry

end f_at_three_l427_42761


namespace igor_process_terminates_l427_42755

/-- Appends a digit to make a number divisible by 11 -/
def appendDigit (n : Nat) : Nat :=
  let m := n * 10
  (m + (11 - m % 11) % 11)

/-- Performs one step of Igor's process -/
def igorStep (n : Nat) : Nat :=
  (appendDigit n) / 11

/-- Checks if Igor can continue the process -/
def canContinue (n : Nat) : Bool :=
  ∃ (d : Nat), d < 10 ∧ (n * 10 + d) % 11 = 0

/-- The sequence of numbers generated by Igor's process -/
def igorSequence : Nat → Nat
  | 0 => 2018
  | n + 1 => igorStep (igorSequence n)

theorem igor_process_terminates :
  ∃ (N : Nat), ¬(canContinue (igorSequence N)) :=
sorry

end igor_process_terminates_l427_42755


namespace simplify_radicals_l427_42727

theorem simplify_radicals : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end simplify_radicals_l427_42727


namespace solution_set_when_a_is_3_range_of_a_l427_42752

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 2} = {x : ℝ | x ≤ 2/3 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, f a x ≥ 5 - x) → a ≥ 6 := by sorry

end solution_set_when_a_is_3_range_of_a_l427_42752


namespace dress_price_proof_l427_42772

theorem dress_price_proof (P : ℝ) (Pd : ℝ) (Pf : ℝ) 
  (h1 : Pd = 0.85 * P) 
  (h2 : Pf = 1.25 * Pd) 
  (h3 : P - Pf = 5.25) : 
  Pd = 71.40 := by
  sorry

end dress_price_proof_l427_42772


namespace donut_shop_problem_l427_42781

def donut_combinations (total_donuts : ℕ) (types : ℕ) : ℕ :=
  let remaining := total_donuts - types
  (types.choose 1) * (types.choose 1) * (types.choose 1) + 
  (types.choose 2) * (remaining.choose 2) +
  (types.choose 3) * (remaining.choose 1)

theorem donut_shop_problem :
  donut_combinations 8 5 = 35 := by
  sorry

end donut_shop_problem_l427_42781


namespace radical_conjugate_sum_product_l427_42729

theorem radical_conjugate_sum_product (a b : ℝ) 
  (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = 4)
  (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 9) : 
  a + b = -3 := by
sorry

end radical_conjugate_sum_product_l427_42729


namespace no_intersection_l427_42724

-- Define the two functions
def f (x : ℝ) : ℝ := |3*x + 6|
def g (x : ℝ) : ℝ := -|4*x - 3|

-- Theorem statement
theorem no_intersection :
  ¬ ∃ (x y : ℝ), f x = y ∧ g x = y :=
sorry

end no_intersection_l427_42724


namespace white_balls_count_l427_42723

theorem white_balls_count (total : ℕ) (red : ℕ) (prob : ℚ) : 
  red = 8 → 
  prob = 2/5 → 
  prob = red / total → 
  total - red = 12 := by
sorry

end white_balls_count_l427_42723


namespace punch_mixture_l427_42749

theorem punch_mixture (total_volume : ℕ) (lemonade_parts : ℕ) (extra_cranberry_parts : ℕ) :
  total_volume = 72 →
  lemonade_parts = 3 →
  extra_cranberry_parts = 18 →
  lemonade_parts + (lemonade_parts + extra_cranberry_parts) = total_volume →
  lemonade_parts + extra_cranberry_parts = 21 := by
  sorry

#check punch_mixture

end punch_mixture_l427_42749


namespace angle_BCD_measure_l427_42732

-- Define a pentagon
structure Pentagon :=
  (A B C D E : ℝ)

-- Define the conditions
def pentagon_conditions (p : Pentagon) : Prop :=
  p.A = 100 ∧ p.D = 120 ∧ p.E = 80 ∧ p.A + p.B + p.C + p.D + p.E = 540

-- Theorem statement
theorem angle_BCD_measure (p : Pentagon) :
  pentagon_conditions p → p.B = 140 → p.C = 100 :=
by
  sorry

end angle_BCD_measure_l427_42732


namespace equal_sum_of_squares_8_9_larger_sum_of_squares_12_11_l427_42704

-- Define the polynomial f(x) = 4x³ - 18x² + 24x
def f (x : ℝ) : ℝ := 4 * x^3 - 18 * x^2 + 24 * x

-- Define a function to calculate the sum of squares of roots
def sum_of_squares_of_roots (a b c d : ℝ) : ℝ := sorry

-- Theorem 1: Equal sum of squares of roots for f(x) = 8 and f(x) = 9
theorem equal_sum_of_squares_8_9 :
  sum_of_squares_of_roots 4 (-18) 24 (-8) = sum_of_squares_of_roots 4 (-18) 24 (-9) := by sorry

-- Theorem 2: Larger sum of squares of roots for f(x) = 12 compared to f(x) = 11
theorem larger_sum_of_squares_12_11 :
  sum_of_squares_of_roots 4 (-18) 24 (-12) > sum_of_squares_of_roots 4 (-18) 24 (-11) := by sorry

end equal_sum_of_squares_8_9_larger_sum_of_squares_12_11_l427_42704


namespace garden_flower_distribution_l427_42760

theorem garden_flower_distribution :
  ∀ (total_flowers white_flowers red_flowers white_roses white_tulips red_roses red_tulips : ℕ),
  total_flowers = 100 →
  white_flowers = 60 →
  red_flowers = total_flowers - white_flowers →
  white_roses = (3 * white_flowers) / 5 →
  white_tulips = white_flowers - white_roses →
  red_tulips = red_flowers / 2 →
  red_roses = red_flowers - red_tulips →
  (white_tulips + red_tulips) * 100 / total_flowers = 44 ∧
  (white_roses + red_roses) * 100 / total_flowers = 56 :=
by sorry

end garden_flower_distribution_l427_42760


namespace only_propositions_3_and_4_are_correct_l427_42718

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations between planes and lines
def perpendicular (p q : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def intersection (p q : Plane) : Plane := sorry

-- Define the planes and lines
def α : Plane := sorry
def β : Plane := sorry
def γ : Plane := sorry
def l : Line := sorry
def m : Line := sorry
def n : Line := sorry

-- Define the propositions
def proposition_1 : Prop :=
  (perpendicular α γ ∧ perpendicular β γ) → parallel α β

def proposition_2 : Prop :=
  (parallel_line_plane m β ∧ parallel_line_plane n β) → parallel α β

def proposition_3 : Prop :=
  (line_in_plane l α ∧ parallel α β) → parallel_line_plane l β

def proposition_4 : Prop :=
  (intersection α β = γ ∧ intersection β γ = m ∧ intersection γ α = n ∧ parallel_line_plane l m) →
  parallel_line_plane m n

-- Theorem to prove
theorem only_propositions_3_and_4_are_correct :
  ¬proposition_1 ∧ ¬proposition_2 ∧ proposition_3 ∧ proposition_4 :=
sorry

end only_propositions_3_and_4_are_correct_l427_42718


namespace max_area_triangle_l427_42717

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is √2 + 1 when a*sin(C) = c*cos(A) and a = 2 -/
theorem max_area_triangle (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
  a * Real.sin C = c * Real.cos A ∧  -- Given condition
  a = 2 →  -- Given condition
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧  -- Area formula
              ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) ∧  -- S is maximum
  ((1/2) * b * c * Real.sin A ≤ Real.sqrt 2 + 1) ∧  -- Upper bound
  (∃ b' c', (1/2) * b' * c' * Real.sin A = Real.sqrt 2 + 1)  -- Maximum is achievable
  := by sorry

end max_area_triangle_l427_42717


namespace number_equality_l427_42736

theorem number_equality (T : ℝ) : (1/3 : ℝ) * (1/8 : ℝ) * T = (1/4 : ℝ) * (1/6 : ℝ) * 150 → T = 150 := by
  sorry

end number_equality_l427_42736


namespace arithmetic_sequence_length_l427_42770

theorem arithmetic_sequence_length :
  ∀ (a₁ d : ℤ) (n : ℕ),
    a₁ = -6 →
    d = 5 →
    a₁ + (n - 1) * d = 59 →
    n = 14 :=
by sorry

end arithmetic_sequence_length_l427_42770


namespace security_deposit_percentage_l427_42745

/-- Security deposit calculation for a mountain cabin rental --/
theorem security_deposit_percentage
  (daily_rate : ℚ)
  (duration_days : ℕ)
  (pet_fee : ℚ)
  (service_fee_rate : ℚ)
  (security_deposit : ℚ)
  (h1 : daily_rate = 125)
  (h2 : duration_days = 14)
  (h3 : pet_fee = 100)
  (h4 : service_fee_rate = 1/5)
  (h5 : security_deposit = 1110) :
  security_deposit / (daily_rate * duration_days + pet_fee + service_fee_rate * (daily_rate * duration_days + pet_fee)) = 1/2 := by
  sorry


end security_deposit_percentage_l427_42745


namespace chocolate_division_l427_42758

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_given : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  piles_given = 2 →
  (total_chocolate / num_piles) * piles_given = 30 / 7 := by
  sorry

end chocolate_division_l427_42758


namespace terry_lunch_options_l427_42789

theorem terry_lunch_options :
  ∀ (lettuce_types tomato_types olive_types soup_types : ℕ),
    lettuce_types = 2 →
    tomato_types = 3 →
    olive_types = 4 →
    soup_types = 2 →
    (lettuce_types * tomato_types * olive_types * soup_types) = 48 :=
by
  sorry

end terry_lunch_options_l427_42789


namespace vince_ride_length_l427_42783

-- Define the length of Zachary's bus ride
def zachary_ride : ℝ := 0.5

-- Define how much longer Vince's ride is compared to Zachary's
def difference : ℝ := 0.13

-- Define Vince's bus ride length
def vince_ride : ℝ := zachary_ride + difference

-- Theorem statement
theorem vince_ride_length : vince_ride = 0.63 := by
  sorry

end vince_ride_length_l427_42783


namespace expression_value_l427_42746

theorem expression_value (a b : ℝ) (h : (a - 3)^2 + |b + 2| = 0) :
  (-a^2 + 3*a*b - 3*b^2) - 2*(-1/2*a^2 + 4*a*b - 3/2*b^2) = 30 := by
  sorry

end expression_value_l427_42746


namespace painted_cubes_l427_42763

theorem painted_cubes (n : ℕ) (h : n = 5) : 
  n^3 - (n - 2)^3 = 98 := by
  sorry

end painted_cubes_l427_42763


namespace smallest_element_100th_set_l427_42734

/-- Defines the smallest element of the nth set in the sequence -/
def smallest_element (n : ℕ) : ℕ := 
  (n - 1) * (n + 2) / 2 + 1

/-- The sequence of sets where the nth set contains n+1 consecutive integers -/
def set_sequence (n : ℕ) : Set ℕ :=
  {k : ℕ | smallest_element n ≤ k ∧ k < smallest_element (n + 1)}

/-- Theorem stating that the smallest element of the 100th set is 5050 -/
theorem smallest_element_100th_set : 
  smallest_element 100 = 5050 := by sorry

end smallest_element_100th_set_l427_42734


namespace max_bookshelves_l427_42766

def room_space : ℕ := 400
def shelf_space : ℕ := 80
def reserved_space : ℕ := 160

theorem max_bookshelves : 
  (room_space - reserved_space) / shelf_space = 3 := by
  sorry

end max_bookshelves_l427_42766


namespace apple_cost_is_40_l427_42778

/-- The cost of apples and pears at Clark's Food Store -/
structure FruitCosts where
  pear_cost : ℕ
  apple_cost : ℕ
  apple_quantity : ℕ
  pear_quantity : ℕ
  total_spent : ℕ

/-- Theorem: The cost of a dozen apples is 40 dollars -/
theorem apple_cost_is_40 (fc : FruitCosts) 
  (h1 : fc.pear_cost = 50)
  (h2 : fc.apple_quantity = 14 ∧ fc.pear_quantity = 14)
  (h3 : fc.total_spent = 1260)
  : fc.apple_cost = 40 := by
  sorry

#check apple_cost_is_40

end apple_cost_is_40_l427_42778


namespace increase_by_percentage_increase_80_by_150_percent_l427_42714

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end increase_by_percentage_increase_80_by_150_percent_l427_42714


namespace min_x_minus_y_l427_42716

theorem min_x_minus_y (x y : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) →
  y ∈ Set.Icc 0 (2 * Real.pi) →
  2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1/2 →
  ∃ (z : Real), z = x - y ∧ ∀ (w : Real), w = x - y → z ≤ w ∧ z = -Real.pi/2 :=
by sorry

end min_x_minus_y_l427_42716
