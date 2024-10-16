import Mathlib

namespace NUMINAMATH_CALUDE_unique_solution_l17_1743

/-- Returns the number of digits in a natural number -/
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + digit_count (n / 10)

/-- Represents k as overline(1n) -/
def k (n : ℕ) : ℕ := 10^(digit_count n) + n

/-- The main theorem stating that (11, 7) is the only solution -/
theorem unique_solution :
  ∀ m n : ℕ, m^2 = n * k n + 2 → (m = 11 ∧ n = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l17_1743


namespace NUMINAMATH_CALUDE_vector_properties_l17_1704

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 1)

theorem vector_properties :
  let magnitude_sum := Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2)
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let x := -9/2
  (magnitude_sum = 5) ∧
  (angle = π/4) ∧
  (∃ (k : ℝ), k ≠ 0 ∧ (x * a.1 + 3 * b.1, x * a.2 + 3 * b.2) = (k * (3 * a.1 - 2 * b.1), k * (3 * a.2 - 2 * b.2))) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l17_1704


namespace NUMINAMATH_CALUDE_min_selection_for_sum_multiple_of_10_l17_1701

/-- The set of numbers from 11 to 30 -/
def S : Set ℕ := {n | 11 ≤ n ∧ n ≤ 30}

/-- A function that checks if the sum of two numbers is a multiple of 10 -/
def sumIsMultipleOf10 (a b : ℕ) : Prop := (a + b) % 10 = 0

/-- The theorem stating the minimum number of integers to be selected -/
theorem min_selection_for_sum_multiple_of_10 :
  ∃ (k : ℕ), k = 11 ∧
  (∀ (T : Set ℕ), T ⊆ S → T.ncard ≥ k →
    ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ sumIsMultipleOf10 a b) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (T : Set ℕ), T ⊆ S ∧ T.ncard = k' ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬(sumIsMultipleOf10 a b)) :=
sorry

end NUMINAMATH_CALUDE_min_selection_for_sum_multiple_of_10_l17_1701


namespace NUMINAMATH_CALUDE_distance_between_intersections_l17_1747

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^4
def curve2 (x y : ℝ) : Prop := x - y^2 = 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- Theorem statement
theorem distance_between_intersections :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt (1 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersections_l17_1747


namespace NUMINAMATH_CALUDE_cube_diff_divisibility_l17_1720

theorem cube_diff_divisibility (a b : ℤ) (n : ℕ) 
  (ha : Odd a) (hb : Odd b) : 
  (2^n : ℤ) ∣ (a^3 - b^3) ↔ (2^n : ℤ) ∣ (a - b) := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_divisibility_l17_1720


namespace NUMINAMATH_CALUDE_angle_between_legs_l17_1767

/-- Given two equal right triangles ABC and ADC with common hypotenuse AC,
    where the angle between planes ABC and ADC is α,
    and the angle between equal legs AB and AD is β,
    prove that the angle between legs BC and CD is
    2 * arcsin(sqrt(sin((α + β)/2) * sin((α - β)/2))). -/
theorem angle_between_legs (α β : Real) :
  let angle_between_planes := α
  let angle_between_equal_legs := β
  let angle_between_BC_CD := 2 * Real.arcsin (Real.sqrt (Real.sin ((α + β) / 2) * Real.sin ((α - β) / 2)))
  angle_between_BC_CD = 2 * Real.arcsin (Real.sqrt (Real.sin ((α + β) / 2) * Real.sin ((α - β) / 2))) :=
by sorry

end NUMINAMATH_CALUDE_angle_between_legs_l17_1767


namespace NUMINAMATH_CALUDE_equation_solution_l17_1731

theorem equation_solution : 
  ∃! x : ℝ, x + 36 / (x - 3) = -9 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l17_1731


namespace NUMINAMATH_CALUDE_number_equation_solution_l17_1765

theorem number_equation_solution : 
  ∃ x : ℝ, (3 * x - 5 = 40) ∧ (x = 15) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l17_1765


namespace NUMINAMATH_CALUDE_sample_size_theorem_l17_1780

theorem sample_size_theorem (N : ℕ) (sample_size : ℕ) (probability : ℚ) 
  (h1 : sample_size = 30)
  (h2 : probability = 1/4)
  (h3 : (sample_size : ℚ) / N = probability) : 
  N = 120 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l17_1780


namespace NUMINAMATH_CALUDE_athlete_b_more_stable_l17_1726

/-- Represents an athlete's assessment scores -/
structure AthleteScores where
  scores : Finset ℝ
  count : Nat
  avg : ℝ
  variance : ℝ

/-- Stability of an athlete's scores -/
def moreStable (a b : AthleteScores) : Prop :=
  a.variance < b.variance

theorem athlete_b_more_stable (a b : AthleteScores) 
  (h_count : a.count = 10 ∧ b.count = 10)
  (h_avg : a.avg = b.avg)
  (h_var_a : a.variance = 1.45)
  (h_var_b : b.variance = 0.85) :
  moreStable b a :=
sorry

end NUMINAMATH_CALUDE_athlete_b_more_stable_l17_1726


namespace NUMINAMATH_CALUDE_equation_solution_l17_1790

theorem equation_solution : 
  ∃ x : ℝ, (3 * x - 5 * x = 600 - (4 * x - 6 * x)) ∧ x = -150 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l17_1790


namespace NUMINAMATH_CALUDE_equation_solution_l17_1773

theorem equation_solution : 
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l17_1773


namespace NUMINAMATH_CALUDE_option_c_is_linear_system_l17_1799

-- Define what a linear equation in two variables is
def is_linear_equation (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y - c

-- Define a system of two equations
def is_system_of_two_equations (f g : ℝ → ℝ → ℝ) : Prop :=
  true  -- This is always true as we're given two equations

-- Define the specific equations from Option C
def eq1 (x y : ℝ) : ℝ := x + y - 5
def eq2 (x y : ℝ) : ℝ := 3 * x - 4 * y - 12

-- Theorem stating that eq1 and eq2 form a system of two linear equations
theorem option_c_is_linear_system :
  is_linear_equation eq1 ∧ is_linear_equation eq2 ∧ is_system_of_two_equations eq1 eq2 :=
sorry

end NUMINAMATH_CALUDE_option_c_is_linear_system_l17_1799


namespace NUMINAMATH_CALUDE_eight_power_15_divided_by_64_power_6_l17_1749

theorem eight_power_15_divided_by_64_power_6 : 8^15 / 64^6 = 512 := by sorry

end NUMINAMATH_CALUDE_eight_power_15_divided_by_64_power_6_l17_1749


namespace NUMINAMATH_CALUDE_snow_volume_to_clear_l17_1771

/-- Calculates the volume of snow to be cleared from a driveway -/
theorem snow_volume_to_clear (length width : Real) (depth : Real) (melt_percentage : Real) : 
  length = 30 ∧ width = 3 ∧ depth = 0.5 ∧ melt_percentage = 0.1 → 
  (1 - melt_percentage) * (length * width * depth) / 27 = 1.5 := by
  sorry

#check snow_volume_to_clear

end NUMINAMATH_CALUDE_snow_volume_to_clear_l17_1771


namespace NUMINAMATH_CALUDE_optimal_play_result_l17_1770

/-- Represents a square on the chessboard --/
structure Square where
  x : Fin 8
  y : Fin 8

/-- Represents the state of the chessboard --/
def Chessboard := Square → Bool

/-- Checks if two squares are neighbors --/
def are_neighbors (s1 s2 : Square) : Bool :=
  (s1.x = s2.x ∧ s1.y.val + 1 = s2.y.val) ∨
  (s1.x = s2.x ∧ s1.y.val = s2.y.val + 1) ∨
  (s1.x.val + 1 = s2.x.val ∧ s1.y = s2.y) ∨
  (s1.x.val = s2.x.val + 1 ∧ s1.y = s2.y)

/-- Counts the number of black connected components on the board --/
def count_black_components (board : Chessboard) : Nat :=
  sorry

/-- Represents a move in the game --/
inductive Move
| alice : Square → Move
| bob : Option Square → Move

/-- Applies a move to the chessboard --/
def apply_move (board : Chessboard) (move : Move) : Chessboard :=
  sorry

/-- Represents the game state --/
structure GameState where
  board : Chessboard
  alice_turn : Bool

/-- Checks if the game is over --/
def is_game_over (state : GameState) : Bool :=
  sorry

/-- Returns the optimal move for the current player --/
def optimal_move (state : GameState) : Move :=
  sorry

/-- Plays the game optimally from the given state until it's over --/
def play_game (state : GameState) : Nat :=
  sorry

/-- The main theorem: optimal play results in 16 black connected components --/
theorem optimal_play_result :
  let initial_state : GameState := {
    board := λ _ => false,  -- All squares are initially white
    alice_turn := true
  }
  play_game initial_state = 16 := by
  sorry

end NUMINAMATH_CALUDE_optimal_play_result_l17_1770


namespace NUMINAMATH_CALUDE_arithmetic_sequence_equals_405_l17_1788

theorem arithmetic_sequence_equals_405 : ((306 / 34) * 15) + 270 = 405 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_equals_405_l17_1788


namespace NUMINAMATH_CALUDE_inverse_prop_problem_l17_1763

/-- Two numbers are inversely proportional if their product is constant -/
def inverse_proportional (a b : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ x : ℝ, a x * b x = k

theorem inverse_prop_problem (a b : ℝ → ℝ) 
  (h1 : inverse_proportional a b)
  (h2 : ∃ x : ℝ, a x + b x = 60 ∧ a x = 3 * b x) :
  b (-12) = -56.25 := by
  sorry


end NUMINAMATH_CALUDE_inverse_prop_problem_l17_1763


namespace NUMINAMATH_CALUDE_dans_initial_money_l17_1766

/-- Dan's initial amount of money, given his remaining money and the cost of a candy bar. -/
def initial_money (remaining : ℕ) (candy_cost : ℕ) : ℕ :=
  remaining + candy_cost

theorem dans_initial_money :
  initial_money 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l17_1766


namespace NUMINAMATH_CALUDE_only_345_is_right_triangle_pythagoras_345_l17_1716

/-- A function that checks if three numbers can form a right triangle --/
def isRightTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The theorem stating that only one of the given sets forms a right triangle --/
theorem only_345_is_right_triangle :
  ¬ isRightTriangle 1 1 (Real.sqrt 3) ∧
  isRightTriangle 3 4 5 ∧
  ¬ isRightTriangle 2 3 4 ∧
  ¬ isRightTriangle 5 7 9 :=
by sorry

/-- The specific theorem for the (3, 4, 5) right triangle --/
theorem pythagoras_345 : 3^2 + 4^2 = 5^2 :=
by sorry

end NUMINAMATH_CALUDE_only_345_is_right_triangle_pythagoras_345_l17_1716


namespace NUMINAMATH_CALUDE_laptop_arrangement_impossible_l17_1719

/-- Represents the number of laptops of each type in a row -/
structure LaptopRow :=
  (typeA : ℕ)
  (typeB : ℕ)
  (typeC : ℕ)

/-- The total number of laptops -/
def totalLaptops : ℕ := 44

/-- The number of rows -/
def numRows : ℕ := 5

/-- Checks if a LaptopRow satisfies the ratio condition -/
def satisfiesRatio (row : LaptopRow) : Prop :=
  3 * row.typeA = 2 * row.typeB ∧ 2 * row.typeC = 3 * row.typeB

/-- Checks if a LaptopRow has at least one of each type -/
def hasAllTypes (row : LaptopRow) : Prop :=
  row.typeA > 0 ∧ row.typeB > 0 ∧ row.typeC > 0

/-- Theorem stating the impossibility of the laptop arrangement -/
theorem laptop_arrangement_impossible : 
  ¬ ∃ (row : LaptopRow), 
    (row.typeA + row.typeB + row.typeC) * numRows = totalLaptops ∧
    satisfiesRatio row ∧
    hasAllTypes row :=
by sorry

end NUMINAMATH_CALUDE_laptop_arrangement_impossible_l17_1719


namespace NUMINAMATH_CALUDE_divisibility_by_three_divisibility_by_eleven_l17_1776

-- Part (a)
theorem divisibility_by_three (a : ℤ) (h : ∃ k : ℤ, a + 1 = 3 * k) : ∃ m : ℤ, 4 + 7 * a = 3 * m := by
  sorry

-- Part (b)
theorem divisibility_by_eleven (a b : ℤ) (h1 : ∃ m : ℤ, 2 + a = 11 * m) (h2 : ∃ n : ℤ, 35 - b = 11 * n) : ∃ p : ℤ, a + b = 11 * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_divisibility_by_eleven_l17_1776


namespace NUMINAMATH_CALUDE_extremum_derivative_zero_sufficient_not_necessary_l17_1779

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the property of having an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem extremum_derivative_zero_sufficient_not_necessary (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, (deriv f) x₀ = 0 → HasExtremumAt f x₀) ∧
  ¬(∀ x₀ : ℝ, HasExtremumAt f x₀ → (deriv f) x₀ = 0) :=
sorry

end NUMINAMATH_CALUDE_extremum_derivative_zero_sufficient_not_necessary_l17_1779


namespace NUMINAMATH_CALUDE_alberto_engine_spending_l17_1730

/-- Represents the spending on car maintenance -/
structure CarSpending where
  oil : ℕ
  tires : ℕ
  detailing : ℕ

/-- Calculates the total spending for a CarSpending instance -/
def total_spending (s : CarSpending) : ℕ := s.oil + s.tires + s.detailing

/-- Represents Samara's spending -/
def samara_spending : CarSpending := { oil := 25, tires := 467, detailing := 79 }

/-- The amount Alberto spent more than Samara -/
def alberto_extra_spending : ℕ := 1886

/-- Theorem: Alberto's spending on the new engine is $2457 -/
theorem alberto_engine_spending :
  total_spending samara_spending + alberto_extra_spending = 2457 := by
  sorry

end NUMINAMATH_CALUDE_alberto_engine_spending_l17_1730


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l17_1721

theorem actual_distance_traveled (speed : ℝ) (faster_speed : ℝ) (extra_distance : ℝ) :
  speed = 10 →
  faster_speed = 20 →
  extra_distance = 40 →
  (∃ (time : ℝ), speed * time = faster_speed * time - extra_distance) →
  speed * (extra_distance / (faster_speed - speed)) = 40 :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l17_1721


namespace NUMINAMATH_CALUDE_max_guaranteed_score_is_four_l17_1764

/-- Represents a player in the game -/
inductive Player : Type
| B : Player
| R : Player

/-- Represents a color of a square -/
inductive Color : Type
| White : Color
| Blue : Color
| Red : Color

/-- Represents a square on the infinite grid -/
structure Square :=
  (x : ℤ)
  (y : ℤ)

/-- Represents the game state -/
structure GameState :=
  (grid : Square → Color)
  (currentPlayer : Player)

/-- Represents a simple polygon on the grid -/
structure SimplePolygon :=
  (squares : Set Square)

/-- The score of player B is the area of the largest simple polygon of blue squares -/
def score (state : GameState) : ℕ :=
  sorry

/-- A strategy for player B -/
def Strategy : Type :=
  GameState → Square

/-- The maximum guaranteed score for player B -/
def maxGuaranteedScore : ℕ :=
  sorry

/-- The main theorem stating that the maximum guaranteed score for B is 4 -/
theorem max_guaranteed_score_is_four :
  maxGuaranteedScore = 4 :=
sorry

end NUMINAMATH_CALUDE_max_guaranteed_score_is_four_l17_1764


namespace NUMINAMATH_CALUDE_sin_600_degrees_l17_1793

theorem sin_600_degrees : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l17_1793


namespace NUMINAMATH_CALUDE_olympiad_problem_l17_1797

theorem olympiad_problem (a b c d : ℕ) 
  (h1 : (a * b - c * d) ∣ a)
  (h2 : (a * b - c * d) ∣ b)
  (h3 : (a * b - c * d) ∣ c)
  (h4 : (a * b - c * d) ∣ d) :
  a * b - c * d = 1 := by
sorry

end NUMINAMATH_CALUDE_olympiad_problem_l17_1797


namespace NUMINAMATH_CALUDE_hyperbola_ratio_range_l17_1789

/-- A hyperbola with foci F₁ and F₂, and a point G satisfying specific conditions -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  G : ℝ × ℝ
  hC : G.1^2 / a^2 - G.2^2 / b^2 = 1
  hG : Real.sqrt ((G.1 - F₁.1)^2 + (G.2 - F₁.2)^2) = 7 * Real.sqrt ((G.1 - F₂.1)^2 + (G.2 - F₂.2)^2)

/-- The range of b/a for a hyperbola satisfying the given conditions -/
theorem hyperbola_ratio_range (h : Hyperbola) : 0 < h.b / h.a ∧ h.b / h.a ≤ Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_ratio_range_l17_1789


namespace NUMINAMATH_CALUDE_positive_integer_pairs_eq_enumerated_set_l17_1732

def positive_integer_pairs : Set (ℕ × ℕ) :=
  {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = 4}

def enumerated_set : Set (ℕ × ℕ) :=
  {(1, 3), (2, 2), (3, 1)}

theorem positive_integer_pairs_eq_enumerated_set :
  positive_integer_pairs = enumerated_set := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_pairs_eq_enumerated_set_l17_1732


namespace NUMINAMATH_CALUDE_two_red_two_blue_probability_l17_1757

/-- The probability of selecting two red and two blue marbles from a bag -/
theorem two_red_two_blue_probability (total_marbles : ℕ) (red_marbles : ℕ) (blue_marbles : ℕ)
  (h1 : total_marbles = red_marbles + blue_marbles)
  (h2 : red_marbles = 12)
  (h3 : blue_marbles = 8) :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 2) / Nat.choose total_marbles 4 = 3696 / 9690 := by
  sorry

end NUMINAMATH_CALUDE_two_red_two_blue_probability_l17_1757


namespace NUMINAMATH_CALUDE_cakes_per_person_l17_1786

theorem cakes_per_person (total_cakes : ℕ) (num_friends : ℕ) 
  (h1 : total_cakes = 32) 
  (h2 : num_friends = 8) 
  (h3 : total_cakes % num_friends = 0) : 
  total_cakes / num_friends = 4 := by
sorry

end NUMINAMATH_CALUDE_cakes_per_person_l17_1786


namespace NUMINAMATH_CALUDE_color_guard_row_length_l17_1755

theorem color_guard_row_length 
  (num_students : ℕ) 
  (student_space : ℝ) 
  (gap_space : ℝ) 
  (h1 : num_students = 40)
  (h2 : student_space = 0.4)
  (h3 : gap_space = 0.5) : 
  (num_students : ℝ) * student_space + (num_students - 1 : ℝ) * gap_space = 35.5 :=
by sorry

end NUMINAMATH_CALUDE_color_guard_row_length_l17_1755


namespace NUMINAMATH_CALUDE_product_from_lcm_gcf_l17_1785

theorem product_from_lcm_gcf (a b c : ℕ+) 
  (h1 : Nat.lcm (Nat.lcm a.val b.val) c.val = 2310)
  (h2 : Nat.gcd (Nat.gcd a.val b.val) c.val = 30) :
  a * b * c = 69300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcf_l17_1785


namespace NUMINAMATH_CALUDE_derivative_even_function_at_zero_l17_1794

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem derivative_even_function_at_zero (f : ℝ → ℝ) (hf : even_function f) 
  (hf' : Differentiable ℝ f) : 
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_even_function_at_zero_l17_1794


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l17_1742

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l17_1742


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l17_1761

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 18 ∧ 
       Real.log p / Real.log 8 = Real.log (p + 2*q) / Real.log 24) : 
  q / p = 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l17_1761


namespace NUMINAMATH_CALUDE_lucy_sold_29_packs_l17_1798

/-- The number of packs of cookies sold by Robyn -/
def robyn_packs : ℕ := 47

/-- The total number of packs of cookies sold by Robyn and Lucy -/
def total_packs : ℕ := 76

/-- The number of packs of cookies sold by Lucy -/
def lucy_packs : ℕ := total_packs - robyn_packs

theorem lucy_sold_29_packs : lucy_packs = 29 := by
  sorry

end NUMINAMATH_CALUDE_lucy_sold_29_packs_l17_1798


namespace NUMINAMATH_CALUDE_night_day_crew_ratio_l17_1746

theorem night_day_crew_ratio (D N : ℕ) (B : ℝ) : 
  (D * B = (3/4) * (D * B + N * ((3/4) * B))) →
  (N : ℝ) / D = 4/3 := by
sorry

end NUMINAMATH_CALUDE_night_day_crew_ratio_l17_1746


namespace NUMINAMATH_CALUDE_modular_exponentiation_difference_l17_1738

theorem modular_exponentiation_difference :
  (41^1723 - 18^1723) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modular_exponentiation_difference_l17_1738


namespace NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_sum_of_digits_of_111111111_squared_l17_1723

/-- 
Given a natural number n, we define X as the number consisting of n ones.
For example, if n = 3, then X = 111.
-/
def X (n : ℕ) : ℕ := (10^n - 1) / 9

/-- 
The sum of digits function for a natural number.
-/
def sumOfDigits (m : ℕ) : ℕ := sorry

/-- 
Theorem: For a number X consisting of n ones, the sum of the digits of X^2 is equal to n^2.
-/
theorem sum_of_digits_of_X_squared (n : ℕ) : 
  sumOfDigits ((X n)^2) = n^2 := by sorry

/-- 
Corollary: For the specific case where n = 9 (corresponding to 111111111), 
the sum of the digits of X^2 is 81.
-/
theorem sum_of_digits_of_111111111_squared : 
  sumOfDigits ((X 9)^2) = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_X_squared_sum_of_digits_of_111111111_squared_l17_1723


namespace NUMINAMATH_CALUDE_unique_tuple_existence_l17_1762

theorem unique_tuple_existence 
  (p q : ℝ) 
  (h_pos_p : 0 < p) 
  (h_pos_q : 0 < q) 
  (h_sum : p + q = 1) 
  (y : Fin 2017 → ℝ) : 
  ∃! x : Fin 2018 → ℝ, 
    (∀ i : Fin 2017, p * max (x i) (x (i + 1)) + q * min (x i) (x (i + 1)) = y i) ∧ 
    x 0 = x 2017 := by
  sorry

end NUMINAMATH_CALUDE_unique_tuple_existence_l17_1762


namespace NUMINAMATH_CALUDE_chinese_money_plant_price_is_25_l17_1795

/-- The price of each potted Chinese money plant -/
def chinese_money_plant_price : ℕ := sorry

/-- The number of orchids sold -/
def orchids_sold : ℕ := 20

/-- The price of each orchid -/
def orchid_price : ℕ := 50

/-- The number of potted Chinese money plants sold -/
def chinese_money_plants_sold : ℕ := 15

/-- The payment for each worker -/
def worker_payment : ℕ := 40

/-- The number of workers -/
def number_of_workers : ℕ := 2

/-- The cost of new pots -/
def new_pots_cost : ℕ := 150

/-- The amount left after expenses -/
def amount_left : ℕ := 1145

theorem chinese_money_plant_price_is_25 :
  chinese_money_plant_price = 25 ∧
  orchids_sold * orchid_price + chinese_money_plants_sold * chinese_money_plant_price =
  amount_left + number_of_workers * worker_payment + new_pots_cost :=
sorry

end NUMINAMATH_CALUDE_chinese_money_plant_price_is_25_l17_1795


namespace NUMINAMATH_CALUDE_min_value_implications_l17_1796

/-- Given a > 0, b > 0, and that the function f(x) = |x+a| + |x-b| has a minimum value of 2,
    prove the following inequalities -/
theorem min_value_implications (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) 
    (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
    (3 * a^2 + b^2 ≥ 3) ∧ (4 / (a + 1) + 1 / b ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_implications_l17_1796


namespace NUMINAMATH_CALUDE_visual_range_increase_l17_1715

theorem visual_range_increase (original_range new_range : ℝ) (h1 : original_range = 60) (h2 : new_range = 150) :
  (new_range - original_range) / original_range * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_visual_range_increase_l17_1715


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l17_1782

/-- The function f(x) = x^2 + 2ax - a + 2 --/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - a + 2

/-- Statement 1: For any x ∈ ℝ, f(x) ≥ 0 if and only if a ∈ [-2,1] --/
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ a ∈ Set.Icc (-2) 1 := by sorry

/-- Statement 2: For any x ∈ [-1,1], f(x) ≥ 0 if and only if a ∈ [-3,1] --/
theorem problem_2 (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 0) ↔ a ∈ Set.Icc (-3) 1 := by sorry

/-- Statement 3: For any a ∈ [-1,1], x^2 + 2ax - a + 2 > 0 if and only if x ≠ -1 --/
theorem problem_3 (x : ℝ) :
  (∀ a ∈ Set.Icc (-1) 1, f a x > 0) ↔ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l17_1782


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l17_1729

theorem square_ratio_side_length_sum (area_ratio : ℚ) : 
  area_ratio = 50 / 98 →
  ∃ (a b c : ℕ), 
    (a * (b.sqrt : ℝ) / c : ℝ) ^ 2 = area_ratio ∧
    a = 5 ∧ b = 14 ∧ c = 49 ∧
    a + b + c = 68 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l17_1729


namespace NUMINAMATH_CALUDE_max_value_theorem_l17_1781

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a * Real.sqrt (1 - b^2) - b * Real.sqrt (1 - a^2) = a * b) :
  (a / b + b / a) ≤ Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l17_1781


namespace NUMINAMATH_CALUDE_circle_has_infinite_symmetry_lines_l17_1775

-- Define a circle
def Circle : Type := Unit

-- Define a line of symmetry for a circle
def LineOfSymmetry (c : Circle) : Type := Unit

-- Define the property of having an infinite number of lines of symmetry
def HasInfiniteSymmetryLines (c : Circle) : Prop :=
  ∀ (n : ℕ), ∃ (lines : Fin n → LineOfSymmetry c), Function.Injective lines

-- Theorem statement
theorem circle_has_infinite_symmetry_lines (c : Circle) :
  HasInfiniteSymmetryLines c := by sorry

end NUMINAMATH_CALUDE_circle_has_infinite_symmetry_lines_l17_1775


namespace NUMINAMATH_CALUDE_proposition_implications_l17_1717

theorem proposition_implications (p q : Prop) :
  ¬(¬p ∨ ¬q) → (p ∧ q) ∧ (p ∨ q) :=
by sorry

end NUMINAMATH_CALUDE_proposition_implications_l17_1717


namespace NUMINAMATH_CALUDE_binary_11001_is_25_l17_1702

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11001_is_25 :
  binary_to_decimal [true, false, false, true, true] = 25 := by
  sorry

end NUMINAMATH_CALUDE_binary_11001_is_25_l17_1702


namespace NUMINAMATH_CALUDE_g_of_3_equals_3_over_17_l17_1711

-- Define the function g
def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

-- State the theorem
theorem g_of_3_equals_3_over_17 : g 3 = 3 / 17 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_3_over_17_l17_1711


namespace NUMINAMATH_CALUDE_lava_lamp_probability_l17_1713

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 2
def num_lamps_on : ℕ := 3

def total_arrangements : ℕ := (Nat.choose (num_red_lamps + num_blue_lamps) num_blue_lamps) * 
                               (Nat.choose (num_red_lamps + num_blue_lamps) num_lamps_on)

def constrained_arrangements : ℕ := (Nat.choose (num_red_lamps + num_blue_lamps - 1) (num_blue_lamps - 1)) * 
                                    (Nat.choose (num_red_lamps + num_blue_lamps - 2) (num_lamps_on - 1))

theorem lava_lamp_probability : 
  (constrained_arrangements : ℚ) / total_arrangements = 1 / 10 := by sorry

end NUMINAMATH_CALUDE_lava_lamp_probability_l17_1713


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_solution_set_for_any_a_l17_1759

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for the second part of the problem
theorem solution_set_for_any_a (a : ℝ) :
  {x : ℝ | f a x > 0} = 
    if a > -1 then
      {x : ℝ | x < -1 ∨ x > a}
    else if a = -1 then
      {x : ℝ | x < -1 ∨ x > -1}
    else
      {x : ℝ | x < a ∨ x > -1} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_solution_set_for_any_a_l17_1759


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l17_1727

theorem arithmetic_series_sum : 
  ∀ (a₁ aₙ d : ℚ) (n : ℕ),
    a₁ = 25 →
    aₙ = 50 →
    d = 2/5 →
    aₙ = a₁ + (n - 1) * d →
    (n : ℚ) * (a₁ + aₙ) / 2 = 2400 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l17_1727


namespace NUMINAMATH_CALUDE_smallest_a_value_exists_polynomial_with_61_l17_1768

/-- Represents a polynomial of degree 4 with integer coefficients -/
structure Polynomial4 (α : Type) [Ring α] where
  a : α
  b : α
  c : α

/-- Predicate to check if a list of integers are the roots of a polynomial -/
def are_roots (p : Polynomial4 ℤ) (roots : List ℤ) : Prop :=
  roots.length = 4 ∧
  (∀ x ∈ roots, x > 0) ∧
  (∀ x ∈ roots, x^4 - p.a * x^3 + p.b * x^2 - p.c * x + 5160 = 0)

/-- The main theorem statement -/
theorem smallest_a_value (p : Polynomial4 ℤ) (roots : List ℤ) :
  are_roots p roots → p.a ≥ 61 := by sorry

/-- The existence of a polynomial with a = 61 -/
theorem exists_polynomial_with_61 :
  ∃ (p : Polynomial4 ℤ) (roots : List ℤ), are_roots p roots ∧ p.a = 61 := by sorry

end NUMINAMATH_CALUDE_smallest_a_value_exists_polynomial_with_61_l17_1768


namespace NUMINAMATH_CALUDE_six_digit_number_puzzle_l17_1700

theorem six_digit_number_puzzle :
  ∀ P Q R S T U : ℕ,
    P ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    Q ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    R ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    S ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    T ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    U ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
    Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
    R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
    S ≠ T ∧ S ≠ U ∧
    T ≠ U →
    (100 * P + 10 * Q + R) % 9 = 0 →
    (100 * Q + 10 * R + S) % 4 = 0 →
    (100 * R + 10 * S + T) % 3 = 0 →
    (P + Q + R + S + T + U) % 5 = 0 →
    U = 4 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_puzzle_l17_1700


namespace NUMINAMATH_CALUDE_arcsin_neg_half_eq_neg_pi_sixth_l17_1750

theorem arcsin_neg_half_eq_neg_pi_sixth : 
  Real.arcsin (-1/2) = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_neg_half_eq_neg_pi_sixth_l17_1750


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l17_1769

def college_cost : ℝ := 30000
def savings : ℝ := 10000
def grant_percentage : ℝ := 0.4

theorem loan_amount_calculation : 
  let remainder := college_cost - savings
  let grant_amount := remainder * grant_percentage
  let loan_amount := remainder - grant_amount
  loan_amount = 12000 := by sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l17_1769


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l17_1792

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l17_1792


namespace NUMINAMATH_CALUDE_five_items_four_boxes_l17_1714

/-- The number of ways to distribute n distinct items into k identical boxes --/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 46 ways to distribute 5 distinct items into 4 identical boxes --/
theorem five_items_four_boxes : distribute 5 4 = 46 := by sorry

end NUMINAMATH_CALUDE_five_items_four_boxes_l17_1714


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l17_1751

theorem square_root_fraction_simplification :
  Real.sqrt (7^2 + 24^2) / Real.sqrt (64 + 36) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l17_1751


namespace NUMINAMATH_CALUDE_range_of_f_l17_1708

-- Define the function
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -3 ≤ y ∧ y ≤ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l17_1708


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l17_1748

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (distance_home_to_grocery : ℝ) 
  (distance_grocery_to_gym : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance_home_to_grocery = 100)
  (h2 : distance_grocery_to_gym = 180)
  (h3 : time_difference = 40)
  : ∃ (v : ℝ), 
    (distance_home_to_grocery / v - distance_grocery_to_gym / (2 * v) = time_difference) ∧ 
    (2 * v = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_angelina_walking_speed_l17_1748


namespace NUMINAMATH_CALUDE_binomial_expected_value_l17_1791

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expected_value (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Theorem: The expected value of X ~ B(6, 1/4) is 3/2 -/
theorem binomial_expected_value :
  let X : BinomialDistribution := { n := 6, p := 1/4, h_p := by norm_num }
  expected_value X = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expected_value_l17_1791


namespace NUMINAMATH_CALUDE_five_student_committees_with_two_fixed_l17_1733

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of different five-student committees that can be chosen from a group of 8 students,
    where two specific students must always be included -/
theorem five_student_committees_with_two_fixed (total_students : ℕ) (committee_size : ℕ) (fixed_students : ℕ) :
  total_students = 8 →
  committee_size = 5 →
  fixed_students = 2 →
  choose (total_students - fixed_students) (committee_size - fixed_students) = 20 := by
  sorry


end NUMINAMATH_CALUDE_five_student_committees_with_two_fixed_l17_1733


namespace NUMINAMATH_CALUDE_swimming_pool_length_l17_1707

/-- Given a rectangular swimming pool with width 22 feet, surrounded by a deck of uniform width 3 feet,
    prove that if the total area of the pool and deck is 728 square feet, then the length of the pool is 20 feet. -/
theorem swimming_pool_length (pool_width deck_width total_area : ℝ) : 
  pool_width = 22 →
  deck_width = 3 →
  (pool_width + 2 * deck_width) * (pool_width + 2 * deck_width) = total_area →
  total_area = 728 →
  ∃ pool_length : ℝ, pool_length = 20 ∧ (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = total_area :=
by sorry

end NUMINAMATH_CALUDE_swimming_pool_length_l17_1707


namespace NUMINAMATH_CALUDE_initial_volume_proof_l17_1718

/-- Proves that the initial volume of a solution is 40 liters given the conditions of the problem -/
theorem initial_volume_proof (V : ℝ) : 
  (0.05 * V + 4.5 = 0.13 * (V + 10)) → V = 40 := by
  sorry

end NUMINAMATH_CALUDE_initial_volume_proof_l17_1718


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l17_1722

theorem unique_two_digit_integer (t : ℕ) : 
  (t ≥ 10 ∧ t ≤ 99) ∧ (13 * t) % 100 = 47 ↔ t = 19 :=
by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l17_1722


namespace NUMINAMATH_CALUDE_regular_polygon_pentagon_l17_1705

/-- A regular polygon with side length 25 and perimeter divisible by 5 yielding the side length is a pentagon with perimeter 125. -/
theorem regular_polygon_pentagon (n : ℕ) (perimeter : ℝ) : 
  n > 0 → 
  perimeter > 0 → 
  perimeter / n = 25 → 
  perimeter / 5 = 25 → 
  n = 5 ∧ perimeter = 125 := by
  sorry

#check regular_polygon_pentagon

end NUMINAMATH_CALUDE_regular_polygon_pentagon_l17_1705


namespace NUMINAMATH_CALUDE_closest_point_to_cheese_l17_1787

/-- The point where the mouse starts getting farther from the cheese -/
def closest_point : ℚ × ℚ := (3/17, 141/17)

/-- The location of the cheese -/
def cheese_location : ℚ × ℚ := (15, 12)

/-- The initial location of the mouse -/
def mouse_initial : ℚ × ℚ := (3, -3)

/-- The path of the mouse -/
def mouse_path (x : ℚ) : ℚ := -4 * x + 9

theorem closest_point_to_cheese :
  let (a, b) := closest_point
  (∀ x : ℚ, (x - 15)^2 + (mouse_path x - 12)^2 ≥ (a - 15)^2 + (b - 12)^2) ∧
  mouse_path a = b ∧
  a + b = 144/17 :=
sorry

end NUMINAMATH_CALUDE_closest_point_to_cheese_l17_1787


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l17_1783

theorem units_digit_of_quotient (h : 7 ∣ (4^1985 + 7^1985)) :
  (4^1985 + 7^1985) / 7 % 10 = 2 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l17_1783


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_is_35_l17_1741

/-- Lemonade stand profit calculation -/
def lemonade_stand_profit : ℝ :=
  let small_yield_per_gallon : ℝ := 16
  let medium_yield_per_gallon : ℝ := 10
  let large_yield_per_gallon : ℝ := 6

  let small_cost_per_gallon : ℝ := 2.00
  let medium_cost_per_gallon : ℝ := 3.50
  let large_cost_per_gallon : ℝ := 5.00

  let small_price_per_glass : ℝ := 1.00
  let medium_price_per_glass : ℝ := 1.75
  let large_price_per_glass : ℝ := 2.50

  let gallons_made_each_size : ℝ := 2

  let small_glasses_produced : ℝ := small_yield_per_gallon * gallons_made_each_size
  let medium_glasses_produced : ℝ := medium_yield_per_gallon * gallons_made_each_size
  let large_glasses_produced : ℝ := large_yield_per_gallon * gallons_made_each_size

  let small_glasses_unsold : ℝ := 4
  let medium_glasses_unsold : ℝ := 4
  let large_glasses_unsold : ℝ := 2

  let setup_cost : ℝ := 15.00
  let advertising_cost : ℝ := 10.00

  let small_revenue := (small_glasses_produced - small_glasses_unsold) * small_price_per_glass
  let medium_revenue := (medium_glasses_produced - medium_glasses_unsold) * medium_price_per_glass
  let large_revenue := (large_glasses_produced - large_glasses_unsold) * large_price_per_glass

  let small_cost := gallons_made_each_size * small_cost_per_gallon
  let medium_cost := gallons_made_each_size * medium_cost_per_gallon
  let large_cost := gallons_made_each_size * large_cost_per_gallon

  let total_revenue := small_revenue + medium_revenue + large_revenue
  let total_cost := small_cost + medium_cost + large_cost + setup_cost + advertising_cost

  total_revenue - total_cost

theorem lemonade_stand_profit_is_35 : lemonade_stand_profit = 35 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_is_35_l17_1741


namespace NUMINAMATH_CALUDE_trail_mix_fruit_percentage_l17_1753

-- Define the trail mix compositions
def sue_mix : ℝ := 5
def sue_nuts_percent : ℝ := 0.3
def sue_fruit_percent : ℝ := 0.7

def jane_mix : ℝ := 7
def jane_nuts_percent : ℝ := 0.6

def tom_mix : ℝ := 9
def tom_nuts_percent : ℝ := 0.4
def tom_fruit_percent : ℝ := 0.5

-- Define the combined mixture properties
def combined_nuts_percent : ℝ := 0.45

-- Theorem to prove
theorem trail_mix_fruit_percentage :
  let total_nuts := sue_mix * sue_nuts_percent + jane_mix * jane_nuts_percent + tom_mix * tom_nuts_percent
  let total_weight := total_nuts / combined_nuts_percent
  let total_fruit := sue_mix * sue_fruit_percent + tom_mix * tom_fruit_percent
  let fruit_percentage := total_fruit / total_weight * 100
  abs (fruit_percentage - 38.71) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_trail_mix_fruit_percentage_l17_1753


namespace NUMINAMATH_CALUDE_cost_price_calculation_l17_1735

/-- Proves that the cost price of an article is 40 given specific conditions --/
theorem cost_price_calculation (C M : ℝ) 
  (h1 : 0.95 * M = 1.25 * C)  -- Selling price after 5% discount equals 25% profit on cost
  (h2 : 0.95 * M = 50)        -- Selling price is 50
  : C = 40 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l17_1735


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l17_1736

theorem camping_trip_percentage :
  ∀ (total_percentage : ℝ) 
    (more_than_100 : ℝ) 
    (not_more_than_100 : ℝ),
  more_than_100 = 18 →
  total_percentage = more_than_100 + not_more_than_100 →
  total_percentage = 72 :=
by sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l17_1736


namespace NUMINAMATH_CALUDE_seventh_term_is_24_l17_1703

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  first_term : ℝ
  common_diff : ℝ

/-- The nth term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + seq.common_diff * (n - 1)

theorem seventh_term_is_24 (seq : ArithmeticSequence) 
  (h1 : seq.first_term = 12)
  (h2 : nth_term seq 4 = 18) :
  nth_term seq 7 = 24 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_24_l17_1703


namespace NUMINAMATH_CALUDE_subtraction_decimal_l17_1712

theorem subtraction_decimal : 7.42 - 2.09 = 5.33 := by sorry

end NUMINAMATH_CALUDE_subtraction_decimal_l17_1712


namespace NUMINAMATH_CALUDE_inequality_always_holds_l17_1728

theorem inequality_always_holds (a b c : ℝ) (h : b > c) : a^2 + b > a^2 + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l17_1728


namespace NUMINAMATH_CALUDE_recreation_spending_percentage_l17_1777

/-- Calculates the percentage of this week's recreation spending compared to last week's. -/
theorem recreation_spending_percentage 
  (last_week_wage : ℝ) 
  (last_week_recreation_percent : ℝ) 
  (this_week_wage_reduction : ℝ) 
  (this_week_recreation_percent : ℝ) 
  (h1 : last_week_recreation_percent = 0.40) 
  (h2 : this_week_wage_reduction = 0.05) 
  (h3 : this_week_recreation_percent = 0.50) : 
  (this_week_recreation_percent * (1 - this_week_wage_reduction) * last_week_wage) / 
  (last_week_recreation_percent * last_week_wage) * 100 = 118.75 :=
by sorry

end NUMINAMATH_CALUDE_recreation_spending_percentage_l17_1777


namespace NUMINAMATH_CALUDE_inheritance_division_l17_1752

theorem inheritance_division (total_amount : ℕ) (num_people : ℕ) (amount_per_person : ℕ) :
  total_amount = 527500 →
  num_people = 5 →
  amount_per_person = total_amount / num_people →
  amount_per_person = 105500 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_division_l17_1752


namespace NUMINAMATH_CALUDE_perpendicular_length_l17_1724

/-- Given two oblique lines and their projections, find the perpendicular length -/
theorem perpendicular_length
  (oblique1 oblique2 : ℝ)
  (projection_ratio : ℚ)
  (h1 : oblique1 = 41)
  (h2 : oblique2 = 50)
  (h3 : projection_ratio = 3 / 10) :
  ∃ (perpendicular : ℝ), perpendicular = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_length_l17_1724


namespace NUMINAMATH_CALUDE_race_car_cost_l17_1709

theorem race_car_cost (mater_cost sally_cost race_car_cost : ℝ) : 
  mater_cost = 0.1 * race_car_cost →
  sally_cost = 3 * mater_cost →
  sally_cost = 42000 →
  race_car_cost = 140000 := by
sorry

end NUMINAMATH_CALUDE_race_car_cost_l17_1709


namespace NUMINAMATH_CALUDE_select_one_from_two_sets_l17_1744

theorem select_one_from_two_sets (left_set right_set : Finset ℕ) 
  (h1 : left_set.card = 15) (h2 : right_set.card = 20) 
  (h3 : left_set ∩ right_set = ∅) : 
  (left_set ∪ right_set).card = 35 := by
  sorry

end NUMINAMATH_CALUDE_select_one_from_two_sets_l17_1744


namespace NUMINAMATH_CALUDE_remainder_problem_l17_1710

theorem remainder_problem (N : ℕ) : 
  (∃ R : ℕ, N = 44 * 432 + R ∧ R < 44) → 
  (∃ Q : ℕ, N = 31 * Q + 5) → 
  (∃ R : ℕ, N = 44 * 432 + R ∧ R = 2) := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l17_1710


namespace NUMINAMATH_CALUDE_fraction_meaningful_l17_1706

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (1 - x) / (x + 2)) ↔ x ≠ -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l17_1706


namespace NUMINAMATH_CALUDE_elvis_songwriting_time_l17_1756

theorem elvis_songwriting_time (total_songs : ℕ) (studio_time : ℕ) (recording_time : ℕ) (editing_time : ℕ)
  (h1 : total_songs = 10)
  (h2 : studio_time = 5 * 60)  -- 5 hours in minutes
  (h3 : recording_time = 12)   -- 12 minutes per song
  (h4 : editing_time = 30)     -- 30 minutes for all songs
  : (studio_time - (total_songs * recording_time + editing_time)) / total_songs = 15 := by
  sorry

end NUMINAMATH_CALUDE_elvis_songwriting_time_l17_1756


namespace NUMINAMATH_CALUDE_units_digit_of_product_is_zero_l17_1737

def is_multiple_of_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

def is_in_range (n : ℕ) : Prop := 20 ≤ n ∧ n ≤ 120

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of_4 n ∧ is_in_range n ∧ Even n

theorem units_digit_of_product_is_zero :
  ∃ (product : ℕ),
    (∀ n : ℕ, satisfies_conditions n → n ∣ product) ∧
    (product % 10 = 0) :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_product_is_zero_l17_1737


namespace NUMINAMATH_CALUDE_positive_sum_squares_bound_l17_1725

theorem positive_sum_squares_bound (x y z a : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : a > 0)
  (sum_eq : x + y + z = a)
  (sum_squares_eq : x^2 + y^2 + z^2 = a^2 / 2) :
  x ≤ 2*a/3 ∧ y ≤ 2*a/3 ∧ z ≤ 2*a/3 :=
by sorry

end NUMINAMATH_CALUDE_positive_sum_squares_bound_l17_1725


namespace NUMINAMATH_CALUDE_sector_angle_l17_1772

/-- Given a circular sector with radius 2 and area 8, prove that its central angle is 4 radians -/
theorem sector_angle (r : ℝ) (area : ℝ) (θ : ℝ) 
  (h_radius : r = 2)
  (h_area : area = 8)
  (h_sector_area : area = 1/2 * r^2 * θ) : θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l17_1772


namespace NUMINAMATH_CALUDE_pinball_spending_l17_1758

def half_dollar : ℚ := 0.5

def wednesday_spent : ℕ := 4
def next_day_spent : ℕ := 14

def total_spent : ℚ := (wednesday_spent * half_dollar) + (next_day_spent * half_dollar)

theorem pinball_spending : total_spent = 9 := by sorry

end NUMINAMATH_CALUDE_pinball_spending_l17_1758


namespace NUMINAMATH_CALUDE_min_distance_MN_l17_1734

def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

def point_on_line (x y : ℝ) : Prop := ∃ (m b : ℝ), y = m*x + b ∧ 1 = m*3 + b

def intersect_points (M N : ℝ × ℝ) : Prop :=
  point_on_line M.1 M.2 ∧ point_on_line N.1 N.2 ∧
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2

theorem min_distance_MN :
  ∀ (M N : ℝ × ℝ), intersect_points M N →
  ∃ (MN : ℝ × ℝ → ℝ × ℝ → ℝ),
    (∀ (A B : ℝ × ℝ), MN A B ≥ 0) ∧
    (MN M N ≥ 4) ∧
    (∃ (M' N' : ℝ × ℝ), intersect_points M' N' ∧ MN M' N' = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_MN_l17_1734


namespace NUMINAMATH_CALUDE_no_valid_n_exists_l17_1778

theorem no_valid_n_exists : ¬ ∃ (n : ℕ), 0 < n ∧ n < 200 ∧ 
  ∃ (m : ℕ), 4 ∣ m ∧ ∃ (k : ℕ), m = k^2 ∧
  ∃ (r : ℕ), (r^2 - n*r + m = 0) ∧ ((r+1)^2 - n*(r+1) + m = 0) :=
sorry

end NUMINAMATH_CALUDE_no_valid_n_exists_l17_1778


namespace NUMINAMATH_CALUDE_x_value_l17_1745

theorem x_value : ∃ x : ℝ, (3 * x) / 7 = 21 ∧ x = 49 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l17_1745


namespace NUMINAMATH_CALUDE_cab_journey_time_l17_1740

/-- Given a cab that arrives 12 minutes late when traveling at 5/6th of its usual speed,
    prove that its usual journey time is 60 minutes. -/
theorem cab_journey_time : ℝ := by
  -- Let S be the usual speed and T be the usual time
  let S : ℝ := 1  -- We can set S to any positive real number
  let T : ℝ := 60 -- This is what we want to prove

  -- Define the reduced speed
  let reduced_speed : ℝ := (5 / 6) * S

  -- Define the time taken at reduced speed
  let reduced_time : ℝ := T + 12

  -- Check if the speed-time relation holds
  have h : S * T = reduced_speed * reduced_time := by sorry

  -- Prove that T equals 60
  sorry

end NUMINAMATH_CALUDE_cab_journey_time_l17_1740


namespace NUMINAMATH_CALUDE_connie_needs_4999_l17_1784

/-- Calculates the additional amount Connie needs to buy the items --/
def additional_amount_needed (saved : ℚ) (watch_price : ℚ) (strap_original : ℚ) (strap_discount : ℚ) 
  (case_price : ℚ) (protector_price_eur : ℚ) (tax_rate : ℚ) (exchange_rate : ℚ) : ℚ :=
  let strap_price := strap_original * (1 - strap_discount)
  let protector_price_usd := protector_price_eur * exchange_rate
  let subtotal := watch_price + strap_price + case_price + protector_price_usd
  let total_with_tax := subtotal * (1 + tax_rate)
  (total_with_tax - saved).ceil / 100

/-- The theorem stating the additional amount Connie needs --/
theorem connie_needs_4999 : 
  additional_amount_needed 39 55 20 0.25 10 2 0.08 1.2 = 4999 / 100 := by
  sorry

end NUMINAMATH_CALUDE_connie_needs_4999_l17_1784


namespace NUMINAMATH_CALUDE_solve_for_y_l17_1754

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l17_1754


namespace NUMINAMATH_CALUDE_sum_f_negative_l17_1760

def f (x : ℝ) : ℝ := x + x^3

theorem sum_f_negative (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ < 0) (h₂ : x₂ + x₃ < 0) (h₃ : x₃ + x₁ < 0) : 
  f x₁ + f x₂ + f x₃ < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_negative_l17_1760


namespace NUMINAMATH_CALUDE_expected_straight_flying_airplanes_l17_1774

def flyProbability : ℚ := 3/4
def notStraightProbability : ℚ := 5/6
def totalAirplanes : ℕ := 80

theorem expected_straight_flying_airplanes :
  (totalAirplanes : ℚ) * flyProbability * (1 - notStraightProbability) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expected_straight_flying_airplanes_l17_1774


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l17_1739

/-- The number of points marked on the circle -/
def n : ℕ := 20

/-- The number of sets of three intersecting chords with endpoints chosen from n points on a circle -/
def intersecting_chords_count (n : ℕ) : ℕ :=
  Nat.choose n 3 + 
  8 * Nat.choose n 4 + 
  5 * Nat.choose n 5 + 
  Nat.choose n 6

/-- Theorem stating that the number of sets of three intersecting chords 
    with endpoints chosen from 20 points on a circle is 156180 -/
theorem intersecting_chords_theorem : 
  intersecting_chords_count n = 156180 := by sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l17_1739
