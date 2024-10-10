import Mathlib

namespace croissant_making_time_l2021_202121

-- Define the constants
def fold_time : ℕ := 5
def fold_count : ℕ := 4
def rest_time : ℕ := 75
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

-- Define the theorem
theorem croissant_making_time :
  (fold_time * fold_count + 
   rest_time * fold_count + 
   mixing_time + 
   baking_time) / 60 = 6 := by
  sorry

end croissant_making_time_l2021_202121


namespace lewis_items_found_l2021_202125

theorem lewis_items_found (tanya_items samantha_items lewis_items : ℕ) : 
  tanya_items = 4 →
  samantha_items = 4 * tanya_items →
  lewis_items = samantha_items + 4 →
  lewis_items = 20 := by
  sorry

end lewis_items_found_l2021_202125


namespace largest_lcm_with_15_l2021_202166

theorem largest_lcm_with_15 :
  (List.maximum [Nat.lcm 15 2, Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 6, Nat.lcm 15 9, Nat.lcm 15 10]).get! = 45 := by
  sorry

end largest_lcm_with_15_l2021_202166


namespace inequality_proof_l2021_202197

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) / (x * y * z) ^ (1/3) ≤ x/y + y/z + z/x := by
  sorry

end inequality_proof_l2021_202197


namespace ellipse_standard_equation_hyperbola_standard_equation_l2021_202158

-- Ellipse
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

theorem ellipse_standard_equation
  (major_axis : ℝ)
  (focal_distance : ℝ)
  (h_major_axis : major_axis = 4)
  (h_focal_distance : focal_distance = 2)
  (h_foci_on_x_axis : True) :
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    x^2 / (major_axis^2 / 4) + y^2 / ((major_axis^2 / 4) - focal_distance^2) = 1 :=
sorry

-- Hyperbola
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

theorem hyperbola_standard_equation
  (k : ℝ)
  (d : ℝ)
  (h_asymptote : k = 3/4)
  (h_directrix : d = 16/5) :
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    x^2 / (d^2 / (1 + k^2)) - y^2 / ((d^2 * k^2) / (1 + k^2)) = 1 :=
sorry

end ellipse_standard_equation_hyperbola_standard_equation_l2021_202158


namespace monkey_pole_height_l2021_202126

/-- Calculates the height of a pole given the ascent pattern and time taken by a monkey to reach the top -/
def poleHeight (ascent : ℕ) (descent : ℕ) (totalTime : ℕ) : ℕ :=
  let fullCycles := (totalTime - 1) / 2
  let remainingDistance := ascent
  fullCycles * (ascent - descent) + remainingDistance

/-- The height of the pole given the monkey's climbing pattern and time -/
theorem monkey_pole_height : poleHeight 2 1 17 = 10 := by
  sorry

end monkey_pole_height_l2021_202126


namespace simplify_scientific_notation_l2021_202140

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := by
  sorry

end simplify_scientific_notation_l2021_202140


namespace sqrt_3_times_sqrt_12_l2021_202108

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l2021_202108


namespace inequality_proof_l2021_202103

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end inequality_proof_l2021_202103


namespace roots_of_cubic_polynomial_l2021_202182

theorem roots_of_cubic_polynomial :
  let f : ℝ → ℝ := λ x => x^3 - 2*x^2 - 5*x + 6
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3) := by sorry

end roots_of_cubic_polynomial_l2021_202182


namespace sequence_is_arithmetic_l2021_202154

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) : ℝ := 2 * n + 5

/-- The theorem stating that the given sequence is arithmetic with first term 7 and common difference 2. -/
theorem sequence_is_arithmetic :
  IsArithmeticSequence a ∧ a 1 = 7 ∧ ∀ n : ℕ, a (n + 1) - a n = 2 := by
  sorry

end sequence_is_arithmetic_l2021_202154


namespace polynomial_division_remainder_l2021_202160

theorem polynomial_division_remainder : 
  ∃ (q : Polynomial ℝ), 
    x^4 + 2*x^3 - 3*x^2 + 4*x - 5 = (x^2 - 3*x + 2) * q + (24*x - 25) := by
  sorry

end polynomial_division_remainder_l2021_202160


namespace collinear_probability_is_7_6325_l2021_202161

/-- A 5x5 grid of dots -/
structure Grid :=
  (size : Nat)
  (h_size : size = 5)

/-- The number of collinear sets of four dots in a 5x5 grid -/
def collinear_sets (g : Grid) : Nat := 14

/-- The total number of ways to choose 4 dots from 25 -/
def total_sets (g : Grid) : Nat := 12650

/-- The probability of selecting four collinear dots from a 5x5 grid -/
def collinear_probability (g : Grid) : ℚ :=
  (collinear_sets g : ℚ) / (total_sets g : ℚ)

/-- Theorem stating that the probability of selecting four collinear dots from a 5x5 grid is 7/6325 -/
theorem collinear_probability_is_7_6325 (g : Grid) :
  collinear_probability g = 7 / 6325 := by
  sorry

end collinear_probability_is_7_6325_l2021_202161


namespace equal_split_probability_eight_dice_l2021_202120

theorem equal_split_probability_eight_dice (n : ℕ) (p : ℝ) : 
  n = 8 →
  p = 1 / 2 →
  (n.choose (n / 2)) * p^n = 35 / 128 :=
by sorry

end equal_split_probability_eight_dice_l2021_202120


namespace factors_of_12_and_18_l2021_202178

def factors (n : ℕ) : Set ℕ := {x | x ∣ n}

theorem factors_of_12_and_18 : 
  factors 12 = {1, 2, 3, 4, 6, 12} ∧ factors 18 = {1, 2, 3, 6, 9, 18} := by
  sorry

end factors_of_12_and_18_l2021_202178


namespace fourth_circle_radius_l2021_202149

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def problem_setup (O₁ O₂ O₃ O₄ : Circle) : Prop :=
  -- O₁ and O₂ are externally tangent
  let (x₁, y₁) := O₁.center
  let (x₂, y₂) := O₂.center
  ((x₂ - x₁)^2 + (y₂ - y₁)^2 = (O₁.radius + O₂.radius)^2) ∧
  -- Radii of O₁ and O₂
  (O₁.radius = 7) ∧
  (O₂.radius = 14) ∧
  -- O₃ is tangent to both O₁ and O₂
  let (x₃, y₃) := O₃.center
  ((x₃ - x₁)^2 + (y₃ - y₁)^2 = (O₁.radius + O₃.radius)^2) ∧
  ((x₃ - x₂)^2 + (y₃ - y₂)^2 = (O₂.radius + O₃.radius)^2) ∧
  -- Center of O₃ is on the line connecting centers of O₁ and O₂
  ((y₃ - y₁) * (x₂ - x₁) = (x₃ - x₁) * (y₂ - y₁)) ∧
  -- O₄ is tangent to O₁, O₂, and O₃
  let (x₄, y₄) := O₄.center
  ((x₄ - x₁)^2 + (y₄ - y₁)^2 = (O₁.radius + O₄.radius)^2) ∧
  ((x₄ - x₂)^2 + (y₄ - y₂)^2 = (O₂.radius + O₄.radius)^2) ∧
  ((x₄ - x₃)^2 + (y₄ - y₃)^2 = (O₃.radius - O₄.radius)^2)

theorem fourth_circle_radius (O₁ O₂ O₃ O₄ : Circle) :
  problem_setup O₁ O₂ O₃ O₄ → O₄.radius = 7/10 := by
  sorry

end fourth_circle_radius_l2021_202149


namespace Q_neither_sufficient_nor_necessary_for_P_l2021_202196

/-- Proposition P: The solution sets of two quadratic inequalities are the same -/
def P (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∀ x, (a₁ * x^2 + b₁ * x + c₁ > 0) ↔ (a₂ * x^2 + b₂ * x + c₂ > 0)

/-- Proposition Q: The ratios of corresponding coefficients are equal -/
def Q (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  a₁ / a₂ = b₁ / b₂ ∧ b₁ / b₂ = c₁ / c₂

theorem Q_neither_sufficient_nor_necessary_for_P :
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, Q a₁ b₁ c₁ a₂ b₂ c₂ ∧ ¬P a₁ b₁ c₁ a₂ b₂ c₂) ∧
  (∃ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, P a₁ b₁ c₁ a₂ b₂ c₂ ∧ ¬Q a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end Q_neither_sufficient_nor_necessary_for_P_l2021_202196


namespace first_player_wins_l2021_202147

/-- Represents the game state -/
structure GameState where
  m : Nat
  n : Nat

/-- Represents a move in the game -/
structure Move where
  row : Nat
  col : Nat

/-- Determines if a move is valid for a given game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  1 ≤ move.row ∧ move.row ≤ state.m ∧ 1 ≤ move.col ∧ move.col ≤ state.n

/-- Applies a move to a game state, returning the new state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  { m := move.row - 1, n := move.col - 1 }

/-- Determines if a game state is terminal (i.e., only the losing square remains) -/
def isTerminal (state : GameState) : Prop :=
  state.m = 1 ∧ state.n = 1

/-- Theorem: The first player has a winning strategy in the chocolate bar game -/
theorem first_player_wins (initialState : GameState) : 
  initialState.m ≥ 1 ∧ initialState.n ≥ 1 → 
  ∃ (strategy : GameState → Move), 
    (∀ (state : GameState), isValidMove state (strategy state)) ∧ 
    (∀ (state : GameState), ¬isTerminal state → 
      ¬∃ (counterStrategy : GameState → Move), 
        (∀ (s : GameState), isValidMove s (counterStrategy s)) ∧
        isTerminal (applyMove (applyMove state (strategy state)) (counterStrategy (applyMove state (strategy state))))) :=
by
  sorry

end first_player_wins_l2021_202147


namespace jacobs_age_l2021_202153

/-- Proves Jacob's age given the conditions of the problem -/
theorem jacobs_age :
  ∀ (rehana_age phoebe_age jacob_age : ℕ),
  rehana_age = 25 →
  rehana_age + 5 = 3 * (phoebe_age + 5) →
  jacob_age = (3 * phoebe_age) / 5 →
  jacob_age = 3 :=
by
  sorry

end jacobs_age_l2021_202153


namespace vector_equality_iff_collinear_l2021_202135

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- The theorem states that for arbitrary points O, A, B, C in a vector space and a scalar k,
    the equality OC = k*OA + (1-k)*OB is equivalent to A, B, and C being collinear. -/
theorem vector_equality_iff_collinear 
  (O A B C : V) (k : ℝ) : 
  (C - O = k • (A - O) + (1 - k) • (B - O)) ↔ 
  ∃ t : ℝ, C - B = t • (A - B) :=
sorry

end vector_equality_iff_collinear_l2021_202135


namespace inequality_solution_l2021_202177

theorem inequality_solution (x : ℝ) : 
  (x^2 + 1)/(x-2) ≥ 3/(x+2) + 2/3 ↔ x ∈ Set.Ioo (-2 : ℝ) (5/3 : ℝ) ∪ Set.Ioi (2 : ℝ) :=
sorry

end inequality_solution_l2021_202177


namespace solution_set_l2021_202127

noncomputable def Solutions (a : ℝ) : Set (ℝ × ℝ × ℝ) :=
  { (1, Real.sqrt (-a), -Real.sqrt (-a)),
    (1, -Real.sqrt (-a), Real.sqrt (-a)),
    (Real.sqrt (-a), -Real.sqrt (-a), 1),
    (-Real.sqrt (-a), 1, Real.sqrt (-a)),
    (Real.sqrt (-a), 1, -Real.sqrt (-a)),
    (-Real.sqrt (-a), Real.sqrt (-a), 1) }

theorem solution_set (a : ℝ) :
  ∀ (x y z : ℝ),
    (x + y + z = 1 ∧
     1/x + 1/y + 1/z = 1 ∧
     x*y*z = a) ↔
    (x, y, z) ∈ Solutions a := by
  sorry

end solution_set_l2021_202127


namespace cubic_factorization_l2021_202131

theorem cubic_factorization (m : ℝ) : m^3 - 4*m = m*(m+2)*(m-2) := by
  sorry

end cubic_factorization_l2021_202131


namespace system_solution_l2021_202181

theorem system_solution (x y : ℝ) : x + y = -5 ∧ 2*y = -2 → x = -4 ∧ y = -1 := by
  sorry

end system_solution_l2021_202181


namespace min_value_expression_l2021_202174

theorem min_value_expression (p x : ℝ) (h1 : 0 < p) (h2 : p < 15) (h3 : p ≤ x) (h4 : x ≤ 15) :
  (∀ y, p ≤ y ∧ y ≤ 15 → |x - p| + |x - 15| + |x - p - 15| ≤ |y - p| + |y - 15| + |y - p - 15|) →
  |x - p| + |x - 15| + |x - p - 15| = 15 :=
by sorry

end min_value_expression_l2021_202174


namespace large_cube_surface_area_l2021_202164

theorem large_cube_surface_area (small_cube_volume : ℝ) (num_small_cubes : ℕ) :
  small_cube_volume = 512 →
  num_small_cubes = 8 →
  let small_cube_side := small_cube_volume ^ (1/3)
  let large_cube_side := 2 * small_cube_side
  let large_cube_surface_area := 6 * large_cube_side^2
  large_cube_surface_area = 1536 := by
  sorry

end large_cube_surface_area_l2021_202164


namespace point_on_curve_iff_satisfies_equation_l2021_202168

-- Define a curve C in 2D space
def Curve (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- Define a point P
def Point (a b : ℝ) : ℝ × ℝ := (a, b)

-- Theorem statement
theorem point_on_curve_iff_satisfies_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  Point a b ∈ Curve F ↔ F a b = 0 := by
  sorry

end point_on_curve_iff_satisfies_equation_l2021_202168


namespace at_least_one_fails_l2021_202123

-- Define the propositions
variable (p : Prop) -- "Student A passes the driving test"
variable (q : Prop) -- "Student B passes the driving test"

-- Define the theorem
theorem at_least_one_fails : (¬p ∨ ¬q) ↔ (∃ student, student = p ∨ student = q) ∧ (¬student) :=
sorry

end at_least_one_fails_l2021_202123


namespace cake_brownie_calorie_difference_l2021_202169

def cake_slices : ℕ := 8
def calories_per_cake_slice : ℕ := 347
def brownies : ℕ := 6
def calories_per_brownie : ℕ := 375

theorem cake_brownie_calorie_difference :
  cake_slices * calories_per_cake_slice - brownies * calories_per_brownie = 526 := by
sorry

end cake_brownie_calorie_difference_l2021_202169


namespace committee_formation_count_l2021_202122

def schoolchildren : ℕ := 12
def teachers : ℕ := 3
def committee_size : ℕ := 9

theorem committee_formation_count :
  (Nat.choose (schoolchildren + teachers) committee_size) -
  (Nat.choose schoolchildren committee_size) = 4785 :=
by sorry

end committee_formation_count_l2021_202122


namespace units_digit_sum_factorials_15_l2021_202183

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  (List.range n).map factorial |> List.sum

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_factorials_15 :
  units_digit (sum_factorials 15) = 3 := by
  sorry

end units_digit_sum_factorials_15_l2021_202183


namespace cube_of_product_l2021_202110

theorem cube_of_product (a b : ℝ) : (-2 * a * b^2)^3 = -8 * a^3 * b^6 := by
  sorry

end cube_of_product_l2021_202110


namespace intersection_property_l2021_202173

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in polar form √2ρcos(θ + π/4) = 1 -/
def Line : Type := Unit

/-- Represents a curve in polar form ρ = 2acosθ -/
def Curve (a : ℝ) : Type := Unit

/-- Returns true if the point is on the given line -/
def Point.onLine (p : Point) (l : Line) : Prop := sorry

/-- Returns true if the point is on the given curve -/
def Point.onCurve (p : Point) (c : Curve a) : Prop := sorry

/-- Calculates the squared distance between two points -/
def Point.distanceSquared (p q : Point) : ℝ := sorry

/-- Theorem: Given the conditions, prove that a = 3 -/
theorem intersection_property (a : ℝ) (l : Line) (c : Curve a) (M P Q : Point)
    (h₁ : a > 0)
    (h₂ : M.x = 0 ∧ M.y = -1)
    (h₃ : P.onLine l ∧ P.onCurve c)
    (h₄ : Q.onLine l ∧ Q.onCurve c)
    (h₅ : P.distanceSquared Q = 4 * P.distanceSquared M * Q.distanceSquared M) :
    a = 3 := by sorry

end intersection_property_l2021_202173


namespace evaluate_expression_l2021_202150

theorem evaluate_expression (a b c : ℚ) (ha : a = 1/2) (hb : b = 3/4) (hc : c = 8) :
  a^3 * b^2 * c = 9/16 := by
  sorry

end evaluate_expression_l2021_202150


namespace max_value_a_l2021_202190

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b) 
  (h2 : b < 2 * c) 
  (h3 : c < 5 * d) 
  (h4 : d < 150) : 
  a ≤ 4460 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 4460 ∧ 
    b' = 1487 ∧ 
    c' = 744 ∧ 
    d' = 149 ∧ 
    a' < 3 * b' ∧ 
    b' < 2 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 150 :=
sorry

end max_value_a_l2021_202190


namespace min_difference_of_h_l2021_202199

noncomputable section

variable (a : ℝ) (x₁ x₂ : ℝ)

def h (x : ℝ) : ℝ := x - 1/x + a * Real.log x

theorem min_difference_of_h (ha : a > 0) (hx₁ : 0 < x₁ ∧ x₁ ≤ 1/Real.exp 1)
  (hroots : x₁^2 + a*x₁ + 1 = 0 ∧ x₂^2 + a*x₂ + 1 = 0) :
  ∃ (m : ℝ), m = 4/Real.exp 1 ∧ ∀ y₁ y₂, 
    (0 < y₁ ∧ y₁ ≤ 1/Real.exp 1) → 
    (y₁^2 + a*y₁ + 1 = 0 ∧ y₂^2 + a*y₂ + 1 = 0) → 
    h a y₁ - h a y₂ ≥ m :=
by sorry

end min_difference_of_h_l2021_202199


namespace count_D3_le_200_eq_9_l2021_202145

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- Count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D3_le_200 : ℕ := sorry

theorem count_D3_le_200_eq_9 : count_D3_le_200 = 9 := by sorry

end count_D3_le_200_eq_9_l2021_202145


namespace girls_in_class_l2021_202193

theorem girls_in_class (total : ℕ) (g b t : ℕ) : 
  total = 60 →
  g + b + t = total →
  3 * t = g →
  2 * t = b →
  g = 30 := by
sorry

end girls_in_class_l2021_202193


namespace probability_tamika_greater_carlos_l2021_202184

def tamika_set : Finset ℕ := {10, 11, 12}
def carlos_set : Finset ℕ := {4, 6, 7}

def tamika_sums : Finset ℕ := {21, 22, 23}
def carlos_sums : Finset ℕ := {10, 11, 13}

def favorable_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

def total_outcomes : ℕ := (tamika_sums.card * carlos_sums.card)

theorem probability_tamika_greater_carlos :
  (favorable_outcomes : ℚ) / total_outcomes = 1 := by sorry

end probability_tamika_greater_carlos_l2021_202184


namespace farm_chicken_count_l2021_202101

/-- Represents the number of chickens on a farm -/
structure FarmChickens where
  roosters : ℕ
  hens : ℕ

/-- Given a farm with the specified conditions, proves that the total number of chickens is 75 -/
theorem farm_chicken_count (farm : FarmChickens) 
  (hen_count : farm.hens = 67)
  (rooster_hen_relation : farm.hens = 9 * farm.roosters - 5) :
  farm.roosters + farm.hens = 75 := by
  sorry


end farm_chicken_count_l2021_202101


namespace problem_proof_l2021_202152

theorem problem_proof : 2^0 - |(-3)| + (-1/2) = -5/2 := by
  sorry

end problem_proof_l2021_202152


namespace percentage_problem_l2021_202112

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  (0.5 * x = 200) → (p * x = 160) → p = 0.4 := by
  sorry

end percentage_problem_l2021_202112


namespace inequality_proof_l2021_202106

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) ≤ 8 / 27 := by
  sorry

end inequality_proof_l2021_202106


namespace complement_M_correct_l2021_202119

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}

-- Theorem statement
theorem complement_M_correct : 
  U \ M = complement_M := by sorry

end complement_M_correct_l2021_202119


namespace hyperbola_equation_from_parameters_l2021_202113

/-- Represents a hyperbola with given eccentricity and foci -/
structure Hyperbola where
  eccentricity : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

/-- The equation of a hyperbola given its parameters -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 4 - y^2 / 12 = 1

/-- Theorem stating that a hyperbola with eccentricity 2 and foci at (-4, 0) and (4, 0)
    has the equation x^2/4 - y^2/12 = 1 -/
theorem hyperbola_equation_from_parameters :
  ∀ h : Hyperbola,
    h.eccentricity = 2 ∧
    h.focus1 = (-4, 0) ∧
    h.focus2 = (4, 0) →
    hyperbola_equation h = fun x y => x^2 / 4 - y^2 / 12 = 1 :=
by sorry

end hyperbola_equation_from_parameters_l2021_202113


namespace smallest_number_divisible_by_11_plus_1_divisible_by_13_l2021_202102

theorem smallest_number_divisible_by_11_plus_1_divisible_by_13 :
  ∃ n : ℕ, n = 77 ∧
  (∀ m : ℕ, m < n → ¬(11 ∣ m ∧ 13 ∣ (m + 1))) ∧
  11 ∣ n ∧ 13 ∣ (n + 1) := by
  sorry

end smallest_number_divisible_by_11_plus_1_divisible_by_13_l2021_202102


namespace least_value_theorem_l2021_202148

theorem least_value_theorem (p q : ℕ) (x : ℚ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 17 * (p + 1) = x * (q + 1))
  (h4 : ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = x * (q' + 1) → p' + q' ≥ 40)
  (h5 : p + q = 40) : 
  x = 14 := by
sorry

end least_value_theorem_l2021_202148


namespace base_n_representation_of_b_l2021_202107

/-- Represents a number in base n --/
structure BaseN (n : ℕ) where
  value : ℕ
  is_valid : value < n^2

/-- Convert a decimal number to base n --/
def to_base_n (n : ℕ) (x : ℕ) : BaseN n :=
  ⟨x % (n^2), by sorry⟩

/-- Convert a base n number to decimal --/
def from_base_n {n : ℕ} (x : BaseN n) : ℕ :=
  x.value

theorem base_n_representation_of_b (n m a b : ℕ) : 
  n > 9 →
  n^2 - a*n + b = 0 →
  m^2 - a*m + b = 0 →
  to_base_n n a = ⟨21, by sorry⟩ →
  to_base_n n (n + m) = ⟨30, by sorry⟩ →
  to_base_n n b = ⟨200, by sorry⟩ := by
  sorry


end base_n_representation_of_b_l2021_202107


namespace equal_roots_quadratic_l2021_202151

theorem equal_roots_quadratic (m : ℝ) :
  (∃ x : ℝ, 4 * x^2 - 6 * x + m = 0 ∧
   ∀ y : ℝ, 4 * y^2 - 6 * y + m = 0 → y = x) →
  m = 9/4 := by
sorry

end equal_roots_quadratic_l2021_202151


namespace london_to_edinburgh_distance_l2021_202118

theorem london_to_edinburgh_distance :
  ∀ D : ℝ,
  (∃ x : ℝ, x = 200 ∧ x + 3.5 = D / 2) →
  D = 393 :=
by
  sorry

end london_to_edinburgh_distance_l2021_202118


namespace divides_condition_l2021_202136

theorem divides_condition (p k r : ℕ) : 
  Prime p → 
  k > 0 → 
  r > 0 → 
  p > r → 
  (pk + r) ∣ (p^p + 1) → 
  r ∣ k := by
sorry

end divides_condition_l2021_202136


namespace odd_function_interval_l2021_202175

/-- A function f is odd on an interval [a, b] if and only if
    the interval is symmetric around the origin -/
def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, f (-x) = -f x ∧ -x ∈ Set.Icc a b

theorem odd_function_interval (f : ℝ → ℝ) (b : ℝ) :
  is_odd_on_interval f (b - 1) 2 → b = -1 := by
  sorry

end odd_function_interval_l2021_202175


namespace system_solution_l2021_202100

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 3 ∧ x - y = 1

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {pair | system pair.1 pair.2}

-- Theorem statement
theorem system_solution :
  solution_set = {(2, 1)} := by
  sorry

end system_solution_l2021_202100


namespace intersection_line_equation_l2021_202189

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

-- Define the line
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (A B : ℝ × ℝ),
  (circle1 A.1 A.2 ∧ circle1 B.1 B.2) ∧
  (circle2 A.1 A.2 ∧ circle2 B.1 B.2) ∧
  A ≠ B →
  line A.1 A.2 ∧ line B.1 B.2 :=
sorry

end intersection_line_equation_l2021_202189


namespace swimmer_time_proof_l2021_202128

/-- Proves that a swimmer takes 3 hours for both downstream and upstream swims given specific conditions -/
theorem swimmer_time_proof (downstream_distance upstream_distance still_water_speed : ℝ) 
  (h1 : downstream_distance = 18)
  (h2 : upstream_distance = 12)
  (h3 : still_water_speed = 5)
  (h4 : downstream_distance / (still_water_speed + (downstream_distance - upstream_distance) / 6) = 
        upstream_distance / (still_water_speed - (downstream_distance - upstream_distance) / 6)) :
  downstream_distance / (still_water_speed + (downstream_distance - upstream_distance) / 6) = 3 ∧
  upstream_distance / (still_water_speed - (downstream_distance - upstream_distance) / 6) = 3 := by
  sorry

#check swimmer_time_proof

end swimmer_time_proof_l2021_202128


namespace negative_seven_x_is_product_l2021_202111

theorem negative_seven_x_is_product : ∀ x : ℝ, -7 * x = -7 * x := by sorry

end negative_seven_x_is_product_l2021_202111


namespace stating_no_room_for_other_animals_l2021_202186

/-- Represents the composition of animals in a circus --/
structure CircusAnimals where
  total : ℕ
  lions : ℕ
  tigers : ℕ
  h_lions : lions = (total - lions) / 5
  h_tigers : tigers = total - tigers + 5

/-- 
Theorem stating that in a circus where the number of lions is 1/5 of the number of non-lions, 
and the number of tigers is 5 more than the number of non-tigers, 
there is no room for any other animals.
-/
theorem no_room_for_other_animals (c : CircusAnimals) : 
  c.lions + c.tigers = c.total :=
sorry

end stating_no_room_for_other_animals_l2021_202186


namespace rabbit_count_l2021_202155

/-- The number of ducks in Eunji's house -/
def num_ducks : ℕ := 52

/-- The number of chickens in Eunji's house -/
def num_chickens : ℕ := 78

/-- The number of rabbits in Eunji's house -/
def num_rabbits : ℕ := 38

/-- Theorem stating the relationship between the number of animals and proving the number of rabbits -/
theorem rabbit_count : num_chickens = num_ducks + num_rabbits - 12 := by
  sorry

end rabbit_count_l2021_202155


namespace two_digit_cube_sum_square_l2021_202133

theorem two_digit_cube_sum_square : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ (((n / 10) + (n % 10))^3 = n^2) := by
  sorry

end two_digit_cube_sum_square_l2021_202133


namespace four_digit_numbers_count_l2021_202146

/-- Represents the set of cards with their numbers -/
def cards : Multiset ℕ := {1, 1, 1, 2, 2, 3, 4}

/-- The number of cards drawn -/
def draw_count : ℕ := 4

/-- Calculates the number of different four-digit numbers that can be formed -/
noncomputable def four_digit_numbers (c : Multiset ℕ) (d : ℕ) : ℕ := sorry

/-- Theorem stating that the number of different four-digit numbers is 114 -/
theorem four_digit_numbers_count : four_digit_numbers cards draw_count = 114 := by sorry

end four_digit_numbers_count_l2021_202146


namespace multiple_of_nine_implies_multiple_of_three_l2021_202156

theorem multiple_of_nine_implies_multiple_of_three 
  (h1 : ∀ n : ℕ, 9 ∣ n → 3 ∣ n) 
  (k : ℕ) 
  (h2 : Odd k) 
  (h3 : 9 ∣ k) : 
  3 ∣ k := by
  sorry

end multiple_of_nine_implies_multiple_of_three_l2021_202156


namespace train_length_l2021_202198

/-- The length of a train given its crossing times over a post and a platform -/
theorem train_length (post_time : ℝ) (platform_length platform_time : ℝ) : 
  post_time = 10 →
  platform_length = 150 →
  platform_time = 20 →
  ∃ (train_length : ℝ), 
    train_length / post_time = (train_length + platform_length) / platform_time ∧
    train_length = 150 := by
  sorry

#check train_length

end train_length_l2021_202198


namespace new_york_temperature_l2021_202163

/-- Given temperatures in three cities with specific relationships, prove the temperature in New York --/
theorem new_york_temperature (t_ny : ℝ) : 
  let t_miami := t_ny + 10
  let t_sandiego := t_miami + 25
  (t_ny + t_miami + t_sandiego) / 3 = 95 →
  t_ny = 80 := by
sorry

end new_york_temperature_l2021_202163


namespace expression_one_equality_expression_two_equality_l2021_202130

-- Expression 1
theorem expression_one_equality : 
  0.25 * (-1/2)^(-4) - 4 / 2^0 - (1/16)^(-1/2) = -4 := by sorry

-- Expression 2
theorem expression_two_equality :
  2 * (Real.log 2 / Real.log 3) - 
  (Real.log (32/9) / Real.log 3) + 
  (Real.log 8 / Real.log 3) - 
  ((Real.log 3 / Real.log 4) + (Real.log 3 / Real.log 8)) * 
  ((Real.log 2 / Real.log 3) + (Real.log 2 / Real.log 9)) = 3/4 := by sorry

end expression_one_equality_expression_two_equality_l2021_202130


namespace simplify_trig_expression_l2021_202188

theorem simplify_trig_expression : 
  (Real.sqrt 3 / Real.cos (10 * π / 180)) - (1 / Real.sin (170 * π / 180)) = -4 := by
  sorry

end simplify_trig_expression_l2021_202188


namespace power_of_one_fourth_l2021_202165

theorem power_of_one_fourth (a b : ℕ) : 
  (2^a : ℕ) = (180 / (180 / 2^a : ℕ) : ℕ) →
  (3^b : ℕ) = (180 / (180 / 3^b : ℕ) : ℕ) →
  (1/4 : ℚ)^(b - a) = 1 := by
sorry

end power_of_one_fourth_l2021_202165


namespace rectangle_perimeter_l2021_202129

/-- Given a rectangle with one side of length 18 and the sum of its area and perimeter being 2016,
    the perimeter of the rectangle is 234. -/
theorem rectangle_perimeter : ∀ w : ℝ,
  let l : ℝ := 18
  let area : ℝ := l * w
  let perimeter : ℝ := 2 * (l + w)
  area + perimeter = 2016 →
  perimeter = 234 := by
sorry

end rectangle_perimeter_l2021_202129


namespace carrot_weight_theorem_l2021_202192

/-- Given 30 carrots, where 27 of them have an average weight of 200 grams
    and 3 of them have an average weight of 180 grams,
    the total weight of all 30 carrots is 5.94 kg. -/
theorem carrot_weight_theorem :
  let total_carrots : ℕ := 30
  let remaining_carrots : ℕ := 27
  let removed_carrots : ℕ := 3
  let avg_weight_remaining : ℝ := 200 -- in grams
  let avg_weight_removed : ℝ := 180 -- in grams
  let total_weight_grams : ℝ := remaining_carrots * avg_weight_remaining + removed_carrots * avg_weight_removed
  let total_weight_kg : ℝ := total_weight_grams / 1000
  total_weight_kg = 5.94 := by
  sorry

end carrot_weight_theorem_l2021_202192


namespace purple_or_orange_probability_l2021_202187

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  green : ℕ
  purple : ℕ
  orange : ℕ
  sum_faces : green + purple + orange = sides

/-- The probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

/-- The main theorem -/
theorem purple_or_orange_probability (d : ColoredDie)
    (h : d.sides = 10 ∧ d.green = 5 ∧ d.purple = 3 ∧ d.orange = 2) :
    probability (d.purple + d.orange) d.sides = 1/2 := by
  sorry


end purple_or_orange_probability_l2021_202187


namespace lowest_sale_price_percentage_l2021_202194

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_discount : ℝ) :
  list_price = 80 →
  max_regular_discount = 0.5 →
  summer_discount = 0.2 →
  let regular_discounted_price := list_price * (1 - max_regular_discount)
  let summer_discount_amount := list_price * summer_discount
  let final_sale_price := regular_discounted_price - summer_discount_amount
  (final_sale_price / list_price) * 100 = 30 := by sorry

end lowest_sale_price_percentage_l2021_202194


namespace polynomial_division_remainder_l2021_202167

theorem polynomial_division_remainder : ∃ (r : ℚ),
  ∀ (z : ℚ), 4 * z^3 - 5 * z^2 - 18 * z + 4 = (4 * z + 6) * (z^2 - 4 * z + 2/3) + r :=
by
  use 10/3
  sorry

end polynomial_division_remainder_l2021_202167


namespace cube_volume_from_surface_area_l2021_202105

theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 864 →
  s^3 = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l2021_202105


namespace fraction_equality_l2021_202138

theorem fraction_equality (w x y : ℝ) (hw : w / x = 1 / 3) (hxy : (x + y) / y = 3) :
  w / y = 2 / 3 := by
  sorry

end fraction_equality_l2021_202138


namespace two_part_trip_average_speed_l2021_202134

/-- Calculates the average speed of a two-part trip -/
theorem two_part_trip_average_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (second_part_speed : ℝ)
  (h1 : total_distance = 450)
  (h2 : first_part_distance = 300)
  (h3 : first_part_speed = 20)
  (h4 : second_part_speed = 15)
  (h5 : first_part_distance < total_distance) :
  (total_distance) / ((first_part_distance / first_part_speed) + ((total_distance - first_part_distance) / second_part_speed)) = 18 := by
  sorry

end two_part_trip_average_speed_l2021_202134


namespace heels_cost_calculation_solve_shopping_problem_l2021_202170

def shopping_problem (initial_amount jumper_cost tshirt_cost remaining_amount : ℕ) : Prop :=
  ∃ (heels_cost : ℕ),
    initial_amount = jumper_cost + tshirt_cost + heels_cost + remaining_amount

theorem heels_cost_calculation (initial_amount jumper_cost tshirt_cost remaining_amount : ℕ) 
  (h : shopping_problem initial_amount jumper_cost tshirt_cost remaining_amount) :
  ∃ (heels_cost : ℕ), heels_cost = initial_amount - jumper_cost - tshirt_cost - remaining_amount :=
by
  sorry

#check @heels_cost_calculation

theorem solve_shopping_problem :
  shopping_problem 26 9 4 8 ∧ 
  (∃ (heels_cost : ℕ), heels_cost = 26 - 9 - 4 - 8 ∧ heels_cost = 5) :=
by
  sorry

#check @solve_shopping_problem

end heels_cost_calculation_solve_shopping_problem_l2021_202170


namespace triangle_side_length_l2021_202132

noncomputable def triangleConfiguration (OA OC OD OB BD : ℝ) : ℝ → Prop :=
  λ y => OA = 5 ∧ OC = 12 ∧ OD = 5 ∧ OB = 3 ∧ BD = 6 ∧ 
    y^2 = OA^2 + OC^2 - 2 * OA * OC * ((OD^2 + OB^2 - BD^2) / (2 * OD * OB))

theorem triangle_side_length : 
  ∃ (OA OC OD OB BD : ℝ), triangleConfiguration OA OC OD OB BD (3 * Real.sqrt 67) :=
sorry

end triangle_side_length_l2021_202132


namespace condition_relationship_l2021_202195

theorem condition_relationship (a b : ℝ) : 
  (∀ a b, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧ 
  (∃ a b, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) := by
  sorry

end condition_relationship_l2021_202195


namespace mrs_hilt_apple_pies_l2021_202157

/-- The number of apple pies Mrs. Hilt baked -/
def apple_pies : ℕ := 150 - 16

/-- Theorem: Mrs. Hilt baked 134 apple pies -/
theorem mrs_hilt_apple_pies : apple_pies = 134 := by
  sorry

end mrs_hilt_apple_pies_l2021_202157


namespace no_positive_integer_solution_l2021_202104

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ+), (x^2 + 5*y^2 = z^2) ∧ (5*x^2 + y^2 = t^2) := by
  sorry

end no_positive_integer_solution_l2021_202104


namespace least_four_digit_7_heavy_l2021_202115

def is_7_heavy (n : ℕ) : Prop := n % 7 > 3

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem least_four_digit_7_heavy : 
  (∀ n : ℕ, is_four_digit n → is_7_heavy n → 1000 ≤ n) ∧ 
  is_four_digit 1000 ∧ 
  is_7_heavy 1000 :=
sorry

end least_four_digit_7_heavy_l2021_202115


namespace first_discount_percentage_l2021_202171

theorem first_discount_percentage (original_price : ℝ) (second_discount : ℝ) (final_price : ℝ) :
  original_price = 175 →
  second_discount = 5 →
  final_price = 133 →
  ∃ (first_discount : ℝ),
    first_discount = 20 ∧
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 :=
by sorry

end first_discount_percentage_l2021_202171


namespace complement_of_intersection_l2021_202180

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {1,3,5}
def B : Set Nat := {1,2}

theorem complement_of_intersection (U A B : Set Nat) :
  U = {1,2,3,4,5,6} →
  A = {1,3,5} →
  B = {1,2} →
  (U \ (A ∩ B)) = {2,3,4,5,6} :=
by
  sorry

end complement_of_intersection_l2021_202180


namespace platinum_to_gold_ratio_is_two_to_one_l2021_202139

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

theorem platinum_to_gold_ratio_is_two_to_one 
  (cards : SallysCards)
  (h1 : cards.gold.balance = cards.gold.limit / 3)
  (h2 : cards.platinum.balance = cards.platinum.limit / 6)
  (h3 : cards.platinum.balance + cards.gold.balance = cards.platinum.limit / 3) :
  cards.platinum.limit / cards.gold.limit = 2 := by
  sorry

end platinum_to_gold_ratio_is_two_to_one_l2021_202139


namespace julio_fishing_result_l2021_202142

/-- Calculates the number of fish Julio has after fishing for a given duration and losing some fish. -/
def fish_remaining (rate : ℕ) (duration : ℕ) (loss : ℕ) : ℕ :=
  rate * duration - loss

/-- Proves that Julio has 48 fish after fishing for 9 hours at a rate of 7 fish per hour and losing 15 fish. -/
theorem julio_fishing_result :
  fish_remaining 7 9 15 = 48 := by
  sorry

end julio_fishing_result_l2021_202142


namespace complex_roots_quadratic_l2021_202109

theorem complex_roots_quadratic (b c : ℝ) : 
  (Complex.I + 1) ^ 2 + b * (Complex.I + 1) + c = 0 →
  (b = -2 ∧ c = 2) ∧ 
  ((Complex.I - 1) ^ 2 + b * (Complex.I - 1) + c = 0) :=
by sorry

end complex_roots_quadratic_l2021_202109


namespace winning_strategy_iff_not_div_four_l2021_202143

/-- A game where two players take turns removing stones from a pile. -/
structure StoneGame where
  n : ℕ  -- Initial number of stones

/-- Represents a valid move in the game -/
inductive ValidMove : ℕ → ℕ → Prop where
  | prime_divisor {n m : ℕ} (h : m.Prime) (d : m ∣ n) : ValidMove n m
  | one {n : ℕ} : ValidMove n 1

/-- Defines a winning strategy for the first player -/
def has_winning_strategy (game : StoneGame) : Prop :=
  ∃ (strategy : ℕ → ℕ),
    ∀ (opponent_move : ℕ → ℕ),
      ValidMove game.n (strategy game.n) ∧
      (∀ k, k < game.n →
        ValidMove k (opponent_move k) →
          ValidMove (k - opponent_move k) (strategy (k - opponent_move k)))

/-- The main theorem: The first player has a winning strategy iff n is not divisible by 4 -/
theorem winning_strategy_iff_not_div_four (game : StoneGame) :
  has_winning_strategy game ↔ ¬(4 ∣ game.n) :=
sorry

end winning_strategy_iff_not_div_four_l2021_202143


namespace expected_prize_money_l2021_202114

theorem expected_prize_money (a₁ : ℝ) : 
  a₁ > 0 →  -- Probability should be positive
  a₁ + 2 * a₁ + 4 * a₁ = 1 →  -- Sum of probabilities is 1
  700 * a₁ + 560 * (2 * a₁) + 420 * (4 * a₁) = 500 := by
  sorry

end expected_prize_money_l2021_202114


namespace sin_cos_roots_quadratic_l2021_202162

theorem sin_cos_roots_quadratic (θ : Real) (m : Real) : 
  (4 * (Real.sin θ)^2 + 2 * m * (Real.sin θ) + m = 0) ∧ 
  (4 * (Real.cos θ)^2 + 2 * m * (Real.cos θ) + m = 0) →
  m = 1 - Real.sqrt 5 := by
sorry

end sin_cos_roots_quadratic_l2021_202162


namespace number_division_remainder_l2021_202176

theorem number_division_remainder (n : ℕ) : 
  (n / 8 = 8 ∧ n % 8 = 0) → n % 5 = 4 := by
  sorry

end number_division_remainder_l2021_202176


namespace water_addition_changes_ratio_l2021_202116

/-- Proves that adding 3 litres of water to a 45-litre mixture with initial milk to water ratio of 4:1 results in a new mixture with milk to water ratio of 3:1 -/
theorem water_addition_changes_ratio :
  let initial_volume : ℝ := 45
  let initial_milk_ratio : ℝ := 4
  let initial_water_ratio : ℝ := 1
  let added_water : ℝ := 3
  let final_milk_ratio : ℝ := 3
  let final_water_ratio : ℝ := 1

  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water

  initial_milk / final_water = final_milk_ratio / final_water_ratio :=
by
  sorry

#check water_addition_changes_ratio

end water_addition_changes_ratio_l2021_202116


namespace complex_number_in_fourth_quadrant_l2021_202179

def complex_number : ℂ := 2 - Complex.I

theorem complex_number_in_fourth_quadrant :
  Real.sign (complex_number.re) = 1 ∧ Real.sign (complex_number.im) = -1 :=
by sorry

end complex_number_in_fourth_quadrant_l2021_202179


namespace cos_540_degrees_l2021_202172

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by sorry

end cos_540_degrees_l2021_202172


namespace fraction_problem_l2021_202117

theorem fraction_problem (N : ℝ) (F : ℝ) 
  (h1 : (1/4) * (1/3) * F * N = 15)
  (h2 : 0.40 * N = 180) : 
  F = 2/5 := by sorry

end fraction_problem_l2021_202117


namespace triangle_inequality_third_side_range_l2021_202159

theorem triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a < b + c ∧ b < a + c ∧ c < a + b) := by sorry

theorem third_side_range : 
  ∀ a : ℝ, (∃ (s1 s2 : ℝ), s1 = 3 ∧ s2 = 5 ∧ 0 < a ∧ 
    (a < s1 + s2 ∧ s1 < a + s2 ∧ s2 < a + s1)) → 
  (2 < a ∧ a < 8) := by sorry

end triangle_inequality_third_side_range_l2021_202159


namespace eldoria_population_2070_l2021_202191

/-- The population growth function for Eldoria -/
def eldoria_population (initial_population : ℕ) (years_since_2000 : ℕ) : ℕ :=
  initial_population * (2 ^ (years_since_2000 / 15))

/-- Theorem: The population of Eldoria in 2070 is 8000 -/
theorem eldoria_population_2070 : 
  eldoria_population 500 70 = 8000 := by
  sorry

#eval eldoria_population 500 70

end eldoria_population_2070_l2021_202191


namespace equation_solutions_l2021_202137

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 81 = 0 ↔ x = 9 ∨ x = -9) ∧
  (∀ x : ℝ, x^3 - 3 = 3/8 ↔ x = 3/2) := by
  sorry

end equation_solutions_l2021_202137


namespace sum_coordinates_of_X_l2021_202144

/-- Given three points X, Y, and Z in the plane satisfying certain conditions,
    prove that the sum of the coordinates of X is -28. -/
theorem sum_coordinates_of_X (X Y Z : ℝ × ℝ) : 
  (∃ (k : ℝ), k = 1/2 ∧ Z - X = k • (Y - X) ∧ Y - Z = k • (Y - X)) → 
  Y = (3, 9) →
  Z = (1, -9) →
  X.1 + X.2 = -28 := by
sorry

end sum_coordinates_of_X_l2021_202144


namespace commission_is_25_l2021_202124

/-- Represents the sales data for a salesman selling security systems --/
structure SalesData where
  second_street_sales : Nat
  fourth_street_sales : Nat
  total_commission : Nat

/-- Calculates the total number of security systems sold --/
def total_sales (data : SalesData) : Nat :=
  data.second_street_sales + (data.second_street_sales / 2) + data.fourth_street_sales

/-- Calculates the commission per security system --/
def commission_per_system (data : SalesData) : Nat :=
  data.total_commission / (total_sales data)

/-- Theorem stating that given the sales conditions, the commission per system is $25 --/
theorem commission_is_25 (data : SalesData) 
  (h1 : data.second_street_sales = 4)
  (h2 : data.fourth_street_sales = 1)
  (h3 : data.total_commission = 175) :
  commission_per_system data = 25 := by
  sorry

#eval commission_per_system { second_street_sales := 4, fourth_street_sales := 1, total_commission := 175 }

end commission_is_25_l2021_202124


namespace negation_equivalence_l2021_202141

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end negation_equivalence_l2021_202141


namespace quadratic_equation_general_form_l2021_202185

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 3 * x * (x - 3) = 4 ↔ 3 * x^2 - 9 * x - 4 = 0 := by
  sorry

end quadratic_equation_general_form_l2021_202185
