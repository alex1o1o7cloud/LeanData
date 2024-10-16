import Mathlib

namespace NUMINAMATH_CALUDE_x_range_when_a_is_one_a_range_when_q_necessary_not_sufficient_l3240_324045

-- Define propositions p and q
def p (a x : ℝ) : Prop := a < x ∧ x < 3 * a

def q (x : ℝ) : Prop := 2 < x ∧ x < 3

-- Theorem 1
theorem x_range_when_a_is_one (x : ℝ) (h1 : p 1 x) (h2 : q x) : 2 < x ∧ x < 3 := by
  sorry

-- Theorem 2
theorem a_range_when_q_necessary_not_sufficient (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, q x → p a x)
  (h3 : ∃ x, p a x ∧ ¬q x) :
  1 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_when_a_is_one_a_range_when_q_necessary_not_sufficient_l3240_324045


namespace NUMINAMATH_CALUDE_remainder_1949_1995_mod_7_l3240_324028

theorem remainder_1949_1995_mod_7 : 1949^1995 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1949_1995_mod_7_l3240_324028


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3240_324024

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3240_324024


namespace NUMINAMATH_CALUDE_tan_alpha_2_implications_l3240_324059

theorem tan_alpha_2_implications (α : Real) (h : Real.tan α = 2) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -1/3 ∧
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implications_l3240_324059


namespace NUMINAMATH_CALUDE_a_value_is_negative_six_l3240_324069

/-- The coefficient of x^4 in the expansion of (2+ax)(1-x)^6 -/
def coefficient (a : ℝ) : ℝ := 30 - 20 * a

/-- The theorem stating that a = -6 given the coefficient of x^4 is 150 -/
theorem a_value_is_negative_six : 
  ∃ a : ℝ, coefficient a = 150 ∧ a = -6 :=
sorry

end NUMINAMATH_CALUDE_a_value_is_negative_six_l3240_324069


namespace NUMINAMATH_CALUDE_john_weight_loss_l3240_324062

/-- The number of calories John burns per day -/
def calories_burned_per_day : ℕ := 2300

/-- The number of calories needed to lose 1 pound -/
def calories_per_pound : ℕ := 4000

/-- The number of days it takes John to lose 10 pounds -/
def days_to_lose_10_pounds : ℕ := 80

/-- The number of pounds John wants to lose -/
def pounds_to_lose : ℕ := 10

/-- The number of calories John eats per day -/
def calories_eaten_per_day : ℕ := 1800

theorem john_weight_loss :
  calories_eaten_per_day =
    calories_burned_per_day -
    (pounds_to_lose * calories_per_pound) / days_to_lose_10_pounds :=
by
  sorry

end NUMINAMATH_CALUDE_john_weight_loss_l3240_324062


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_interval_l3240_324050

theorem inequality_holds_iff_m_in_interval :
  ∀ m : ℝ, (∀ x : ℝ, -6 < (2 * x^2 + m * x - 4) / (x^2 - x + 1) ∧ 
    (2 * x^2 + m * x - 4) / (x^2 - x + 1) < 4) ↔ 
  -2 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_interval_l3240_324050


namespace NUMINAMATH_CALUDE_abs_inequality_iff_l3240_324002

theorem abs_inequality_iff (a b : ℝ) : a > b ↔ a * |a| > b * |b| := by sorry

end NUMINAMATH_CALUDE_abs_inequality_iff_l3240_324002


namespace NUMINAMATH_CALUDE_modular_inverse_of_5_mod_24_l3240_324023

theorem modular_inverse_of_5_mod_24 :
  ∃ a : ℕ, a < 24 ∧ (5 * a) % 24 = 1 ∧ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_5_mod_24_l3240_324023


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l3240_324019

/-- Represents a bag of colored balls -/
structure ColoredBalls where
  total : ℕ
  red : ℕ
  black : ℕ

/-- Calculates the probability of drawing two red balls -/
def prob_two_red (bag : ColoredBalls) : ℚ :=
  (bag.red.choose 2 : ℚ) / (bag.total.choose 2)

/-- Calculates the probability of drawing two balls of different colors -/
def prob_different_colors (bag : ColoredBalls) : ℚ :=
  (bag.red * bag.black : ℚ) / (bag.total.choose 2)

/-- The main theorem about probabilities in the ball drawing scenario -/
theorem ball_drawing_probabilities (bag : ColoredBalls) 
    (h_total : bag.total = 6)
    (h_red : bag.red = 4)
    (h_black : bag.black = 2) :
    prob_two_red bag = 2/5 ∧ prob_different_colors bag = 8/15 := by
  sorry


end NUMINAMATH_CALUDE_ball_drawing_probabilities_l3240_324019


namespace NUMINAMATH_CALUDE_unique_m_value_l3240_324094

def A (m : ℝ) : Set ℝ := {1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem unique_m_value : ∀ m : ℝ, B m ⊆ A m → m = -1 := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l3240_324094


namespace NUMINAMATH_CALUDE_koi_fish_count_l3240_324080

theorem koi_fish_count : ∃ k : ℕ, (2 * k - 14 = 64) ∧ (k = 39) := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_count_l3240_324080


namespace NUMINAMATH_CALUDE_meatballs_stolen_l3240_324003

/-- The number of meatballs Hayley initially had -/
def initial_meatballs : ℕ := 25

/-- The number of meatballs Hayley has now -/
def current_meatballs : ℕ := 11

/-- The number of meatballs Kirsten stole -/
def stolen_meatballs : ℕ := initial_meatballs - current_meatballs

theorem meatballs_stolen : stolen_meatballs = 14 := by
  sorry

end NUMINAMATH_CALUDE_meatballs_stolen_l3240_324003


namespace NUMINAMATH_CALUDE_g20_asia_members_l3240_324052

/-- Represents the continents in the G20 --/
inductive Continent
  | Asia
  | Europe
  | Africa
  | Oceania
  | America

/-- Structure representing the G20 membership distribution --/
structure G20 where
  members : Continent → ℕ
  total_twenty : (members Continent.Asia + members Continent.Europe + members Continent.Africa + 
                  members Continent.Oceania + members Continent.America) = 20
  asia_highest : ∀ c : Continent, members Continent.Asia ≥ members c
  africa_oceania_least : members Continent.Africa = members Continent.Oceania ∧ 
                         ∀ c : Continent, members c ≥ members Continent.Africa
  consecutive : ∃ x : ℕ, members Continent.America = x ∧ 
                         members Continent.Europe = x + 1 ∧ 
                         members Continent.Asia = x + 2

theorem g20_asia_members (g : G20) : g.members Continent.Asia = 7 := by
  sorry

end NUMINAMATH_CALUDE_g20_asia_members_l3240_324052


namespace NUMINAMATH_CALUDE_square_difference_l3240_324057

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3240_324057


namespace NUMINAMATH_CALUDE_pencils_left_l3240_324072

def initial_pencils : ℕ := 142
def pencils_given_away : ℕ := 31

theorem pencils_left : initial_pencils - pencils_given_away = 111 := by
  sorry

end NUMINAMATH_CALUDE_pencils_left_l3240_324072


namespace NUMINAMATH_CALUDE_reciprocal_contraction_l3240_324082

open Real

theorem reciprocal_contraction {x₁ x₂ : ℝ} (h₁ : 1 < x₁) (h₂ : x₁ < 2) (h₃ : 1 < x₂) (h₄ : x₂ < 2) (h₅ : x₁ ≠ x₂) :
  |1 / x₁ - 1 / x₂| < |x₂ - x₁| := by
sorry

end NUMINAMATH_CALUDE_reciprocal_contraction_l3240_324082


namespace NUMINAMATH_CALUDE_sum_of_divisors_57_l3240_324048

/-- The sum of all positive divisors of 57 is 80. -/
theorem sum_of_divisors_57 : (Finset.filter (λ x => 57 % x = 0) (Finset.range 58)).sum id = 80 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_57_l3240_324048


namespace NUMINAMATH_CALUDE_fraction_equality_l3240_324043

theorem fraction_equality (a b : ℝ) (h : (1 / a) + (1 / (2 * b)) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3240_324043


namespace NUMINAMATH_CALUDE_alice_bracelet_profit_l3240_324061

/-- Calculates Alice's profit from selling friendship bracelets -/
theorem alice_bracelet_profit :
  let total_design_a : ℕ := 30
  let total_design_b : ℕ := 22
  let cost_design_a : ℚ := 2
  let cost_design_b : ℚ := 4.5
  let given_away_design_a : ℕ := 5
  let given_away_design_b : ℕ := 3
  let bulk_price_design_a : ℚ := 0.2
  let bulk_price_design_b : ℚ := 0.4
  let total_cost := total_design_a * cost_design_a + total_design_b * cost_design_b
  let remaining_design_a := total_design_a - given_away_design_a
  let remaining_design_b := total_design_b - given_away_design_b
  let total_revenue := remaining_design_a * bulk_price_design_a + remaining_design_b * bulk_price_design_b
  let profit := total_revenue - total_cost
  profit = -146.4 := by sorry

end NUMINAMATH_CALUDE_alice_bracelet_profit_l3240_324061


namespace NUMINAMATH_CALUDE_andrews_age_l3240_324056

theorem andrews_age (a g : ℚ) 
  (h1 : g = 10 * a)
  (h2 : g - (a + 2) = 57) :
  a = 59 / 9 := by
  sorry

end NUMINAMATH_CALUDE_andrews_age_l3240_324056


namespace NUMINAMATH_CALUDE_layla_points_difference_l3240_324039

/-- Given a game where the total points scored is 112 and Layla scored 70 points,
    prove that Layla scored 28 more points than Nahima. -/
theorem layla_points_difference (total_points : ℕ) (layla_points : ℕ) : 
  total_points = 112 →
  layla_points = 70 →
  layla_points - (total_points - layla_points) = 28 := by
  sorry

end NUMINAMATH_CALUDE_layla_points_difference_l3240_324039


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l3240_324021

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1/2) * length →
  length - width = 12 →
  length * width = 288 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l3240_324021


namespace NUMINAMATH_CALUDE_cos_equality_proof_l3240_324014

theorem cos_equality_proof (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (1534 * π / 180) → n = 154 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_proof_l3240_324014


namespace NUMINAMATH_CALUDE_sequence_general_term_l3240_324089

/-- Given a sequence {a_n} where S_n is the sum of its first n terms, 
    prove that if S_n + a_n = (n-1) / (n(n+1)) for n ≥ 1, 
    then a_n = 1/(2^n) - 1/(n(n+1)) for all n ≥ 1 -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n ≥ 1 → S n + a n = (n - 1 : ℚ) / (n * (n + 1))) →
  ∀ n : ℕ, n ≥ 1 → a n = 1 / (2 ^ n) - 1 / (n * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3240_324089


namespace NUMINAMATH_CALUDE_largest_x_equation_l3240_324030

theorem largest_x_equation (x : ℝ) : 
  (((14 * x^3 - 40 * x^2 + 20 * x - 4) / (4 * x - 3) + 6 * x = 8 * x - 3) ↔ 
  (14 * x^3 - 48 * x^2 + 38 * x - 13 = 0)) ∧ 
  (∀ y : ℝ, ((14 * y^3 - 40 * y^2 + 20 * y - 4) / (4 * y - 3) + 6 * y = 8 * y - 3) → y ≤ x) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_equation_l3240_324030


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3240_324070

theorem arithmetic_mean_of_special_set (n : ℕ) (hn : n > 2) : 
  let set := [1 - 1 / n, 1 + 1 / n] ++ List.replicate (n - 2) 1
  (List.sum set) / n = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3240_324070


namespace NUMINAMATH_CALUDE_inverse_of_A_l3240_324020

def A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; -2, 9]

theorem inverse_of_A :
  A⁻¹ = !![9/35, -4/35; 2/35, 3/35] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_l3240_324020


namespace NUMINAMATH_CALUDE_area_of_corner_squares_l3240_324022

/-- The total area of four smaller squares inscribed in the corners of a 2x2 square with an inscribed circle -/
theorem area_of_corner_squares (s : ℝ) : 
  s > 0 ∧ 
  s^2 - 4*s + 2 = 0 ∧ 
  (∃ (r : ℝ), r = 1 ∧ r^2 + r^2 = s^2) →
  4 * s^2 = (48 - 32 * Real.sqrt 2) / 18 :=
by sorry

end NUMINAMATH_CALUDE_area_of_corner_squares_l3240_324022


namespace NUMINAMATH_CALUDE_smallest_c_for_inverse_l3240_324013

def f (x : ℝ) := (x - 3)^2 - 4

theorem smallest_c_for_inverse : 
  ∀ c : ℝ, (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ↔ c ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_c_for_inverse_l3240_324013


namespace NUMINAMATH_CALUDE_turnip_bag_weights_l3240_324068

def bag_weights : List ℕ := [13, 15, 16, 17, 21, 24]

def is_valid_turnip_weight (t : ℕ) : Prop :=
  t ∈ bag_weights ∧
  ∃ (onion_weight carrot_weight : ℕ),
    onion_weight + carrot_weight = (bag_weights.sum - t) ∧
    carrot_weight = 2 * onion_weight ∧
    ∃ (onion_bags carrot_bags : List ℕ),
      onion_bags ++ carrot_bags = bag_weights.filter (λ w => w ≠ t) ∧
      onion_bags.sum = onion_weight ∧
      carrot_bags.sum = carrot_weight

theorem turnip_bag_weights :
  ∀ t : ℕ, is_valid_turnip_weight t ↔ t = 13 ∨ t = 16 :=
by sorry

end NUMINAMATH_CALUDE_turnip_bag_weights_l3240_324068


namespace NUMINAMATH_CALUDE_triangular_prism_is_pentahedron_l3240_324031

-- Define the polyhedra types
inductive Polyhedron
| TriangularPyramid
| TriangularPrism
| QuadrangularPrism
| PentagonalPyramid

-- Define the function that returns the number of faces for each polyhedron
def numFaces (p : Polyhedron) : Nat :=
  match p with
  | Polyhedron.TriangularPyramid => 4    -- tetrahedron
  | Polyhedron.TriangularPrism => 5      -- pentahedron
  | Polyhedron.QuadrangularPrism => 6    -- hexahedron
  | Polyhedron.PentagonalPyramid => 6    -- hexahedron

-- Theorem: A triangular prism is a pentahedron
theorem triangular_prism_is_pentahedron :
  numFaces Polyhedron.TriangularPrism = 5 := by sorry

end NUMINAMATH_CALUDE_triangular_prism_is_pentahedron_l3240_324031


namespace NUMINAMATH_CALUDE_staircase_perimeter_l3240_324081

/-- Represents a staircase-shaped region with specific properties -/
structure StaircaseRegion where
  tickMarkSides : ℕ
  tickMarkLength : ℝ
  bottomBaseLength : ℝ
  totalArea : ℝ

/-- Calculates the perimeter of a StaircaseRegion -/
def perimeter (s : StaircaseRegion) : ℝ :=
  s.bottomBaseLength + s.tickMarkSides * s.tickMarkLength

theorem staircase_perimeter (s : StaircaseRegion) 
  (h1 : s.tickMarkSides = 12)
  (h2 : s.tickMarkLength = 1)
  (h3 : s.bottomBaseLength = 12)
  (h4 : s.totalArea = 78) :
  perimeter s = 34.5 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l3240_324081


namespace NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l3240_324074

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the condition for the tangent line being parallel to 4x - y - 1 = 0
def tangent_parallel (x : ℝ) : Prop := f' x = 4

-- Define the third quadrant condition
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Main theorem
theorem tangent_point_and_perpendicular_line :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    tangent_parallel x₀ ∧ 
    third_quadrant x₀ y₀ ∧ 
    x₀ = -1 ∧ 
    y₀ = -4 ∧ 
    ∀ (x y : ℝ), x + 4*y + 17 = 0 ↔ y - y₀ = -(1/4) * (x - x₀) :=
by sorry

end NUMINAMATH_CALUDE_tangent_point_and_perpendicular_line_l3240_324074


namespace NUMINAMATH_CALUDE_same_height_time_l3240_324035

/-- Represents the height of a ball as a function of time -/
def ball_height (a h : ℝ) (t : ℝ) : ℝ := a * (t - 1.2)^2 + h

theorem same_height_time : 
  ∀ (a h : ℝ), a ≠ 0 →
  ∃ (t : ℝ), t > 0 ∧ 
  ball_height a h t = ball_height a h (t - 2) ∧
  t = 2.2 :=
sorry

end NUMINAMATH_CALUDE_same_height_time_l3240_324035


namespace NUMINAMATH_CALUDE_rectangles_4x2_grid_l3240_324017

/-- The number of rectangles that can be formed on a grid of dots -/
def num_rectangles (cols : ℕ) (rows : ℕ) : ℕ :=
  (cols.choose 2) * (rows.choose 2)

/-- Theorem: The number of rectangles on a 4x2 grid is 6 -/
theorem rectangles_4x2_grid : num_rectangles 4 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_4x2_grid_l3240_324017


namespace NUMINAMATH_CALUDE_proposition_induction_l3240_324015

theorem proposition_induction (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  ¬ P 7 →
  ¬ P 6 := by
  sorry

end NUMINAMATH_CALUDE_proposition_induction_l3240_324015


namespace NUMINAMATH_CALUDE_articles_with_equal_price_l3240_324038

/-- Represents the cost price of a single article -/
def cost_price : ℝ := sorry

/-- Represents the selling price of a single article -/
def selling_price : ℝ := sorry

/-- The number of articles whose selling price equals the cost price of 50 articles -/
def N : ℝ := sorry

/-- The gain percentage -/
def gain_percent : ℝ := 100

theorem articles_with_equal_price :
  (50 * cost_price = N * selling_price) →
  (selling_price = 2 * cost_price) →
  (N = 25) :=
by sorry

end NUMINAMATH_CALUDE_articles_with_equal_price_l3240_324038


namespace NUMINAMATH_CALUDE_pigeon_win_conditions_l3240_324064

/-- The game result for the pigeon -/
inductive GameResult
| Win
| Lose

/-- Determines the game result for the pigeon given the board size, egg count, and seagull's square size -/
def pigeonWins (n : ℕ) (m : ℕ) (k : ℕ) : GameResult :=
  if k ≤ n ∧ n ≤ 2 * k - 1 ∧ m ≥ k^2 then GameResult.Win
  else if n ≥ 2 * k ∧ m ≥ k^2 + 1 then GameResult.Win
  else GameResult.Lose

/-- Theorem stating the conditions for the pigeon to win -/
theorem pigeon_win_conditions (n : ℕ) (m : ℕ) (k : ℕ) (h : n ≥ k) :
  (k ≤ n ∧ n ≤ 2 * k - 1 → (pigeonWins n m k = GameResult.Win ↔ m ≥ k^2)) ∧
  (n ≥ 2 * k → (pigeonWins n m k = GameResult.Win ↔ m ≥ k^2 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_pigeon_win_conditions_l3240_324064


namespace NUMINAMATH_CALUDE_terminal_side_point_y_value_l3240_324088

theorem terminal_side_point_y_value (α : Real) (y : Real) :
  let P : Real × Real := (-Real.sqrt 3, y)
  (P.1^2 + P.2^2 ≠ 0) →  -- Ensure the point is not at the origin
  (Real.sin α = Real.sqrt 13 / 13) →
  y = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_terminal_side_point_y_value_l3240_324088


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l3240_324093

/-- Represents a point on the grid --/
structure Point :=
  (x : Nat) (y : Nat)

/-- Represents the initial configuration of shaded squares --/
def initial_shaded : List Point :=
  [⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩]

/-- The dimensions of the grid --/
def grid_width : Nat := 6
def grid_height : Nat := 5

/-- Checks if a point is within the grid --/
def is_valid_point (p : Point) : Bool :=
  p.x > 0 ∧ p.x ≤ grid_width ∧ p.y > 0 ∧ p.y ≤ grid_height

/-- Reflects a point across the vertical line of symmetry --/
def reflect_vertical (p : Point) : Point :=
  ⟨grid_width + 1 - p.x, p.y⟩

/-- Reflects a point across the horizontal line of symmetry --/
def reflect_horizontal (p : Point) : Point :=
  ⟨p.x, grid_height + 1 - p.y⟩

/-- Theorem: The minimum number of additional squares to shade for symmetry is 7 --/
theorem min_additional_squares_for_symmetry :
  ∃ (additional_shaded : List Point),
    (∀ p ∈ additional_shaded, is_valid_point p) ∧
    (∀ p ∈ initial_shaded, reflect_vertical p ∈ additional_shaded ∨ reflect_vertical p ∈ initial_shaded) ∧
    (∀ p ∈ initial_shaded, reflect_horizontal p ∈ additional_shaded ∨ reflect_horizontal p ∈ initial_shaded) ∧
    (∀ p ∈ additional_shaded, reflect_vertical p ∈ additional_shaded ∨ reflect_vertical p ∈ initial_shaded) ∧
    (∀ p ∈ additional_shaded, reflect_horizontal p ∈ additional_shaded ∨ reflect_horizontal p ∈ initial_shaded) ∧
    additional_shaded.length = 7 ∧
    (∀ other_shaded : List Point,
      (∀ p ∈ other_shaded, is_valid_point p) →
      (∀ p ∈ initial_shaded, reflect_vertical p ∈ other_shaded ∨ reflect_vertical p ∈ initial_shaded) →
      (∀ p ∈ initial_shaded, reflect_horizontal p ∈ other_shaded ∨ reflect_horizontal p ∈ initial_shaded) →
      (∀ p ∈ other_shaded, reflect_vertical p ∈ other_shaded ∨ reflect_vertical p ∈ initial_shaded) →
      (∀ p ∈ other_shaded, reflect_horizontal p ∈ other_shaded ∨ reflect_horizontal p ∈ initial_shaded) →
      other_shaded.length ≥ 7) :=
by
  sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l3240_324093


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l3240_324025

/-- Given that 3 dozen apples cost $23.40, prove that 5 dozen apples at the same rate cost $39.00 -/
theorem apple_cost_calculation (cost_three_dozen : ℝ) (h1 : cost_three_dozen = 23.40) :
  let cost_per_dozen : ℝ := cost_three_dozen / 3
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 39.00 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_calculation_l3240_324025


namespace NUMINAMATH_CALUDE_louis_age_l3240_324098

/-- Given the ages of Matilda, Jerica, and Louis, prove Louis' age -/
theorem louis_age (matilda_age jerica_age louis_age : ℕ) : 
  matilda_age = 35 →
  matilda_age = jerica_age + 7 →
  jerica_age = 2 * louis_age →
  louis_age = 14 := by
  sorry

#check louis_age

end NUMINAMATH_CALUDE_louis_age_l3240_324098


namespace NUMINAMATH_CALUDE_probability_under_20_l3240_324007

theorem probability_under_20 (total : ℕ) (over_30 : ℕ) (under_20 : ℕ) :
  total = 100 →
  over_30 = 90 →
  under_20 = total - over_30 →
  (under_20 : ℚ) / (total : ℚ) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_under_20_l3240_324007


namespace NUMINAMATH_CALUDE_lukes_fishing_days_l3240_324032

/-- Proves the number of days Luke catches fish given the conditions -/
theorem lukes_fishing_days (fish_per_day : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ) : 
  fish_per_day = 2 → 
  fillets_per_fish = 2 → 
  total_fillets = 120 → 
  (total_fillets / fillets_per_fish) / fish_per_day = 30 := by
  sorry

end NUMINAMATH_CALUDE_lukes_fishing_days_l3240_324032


namespace NUMINAMATH_CALUDE_product_of_fractions_l3240_324097

theorem product_of_fractions : 
  (1/2 : ℚ) * (9/1 : ℚ) * (1/8 : ℚ) * (64/1 : ℚ) * (1/128 : ℚ) * (729/1 : ℚ) * (1/2187 : ℚ) * (19683/1 : ℚ) = 59049/32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l3240_324097


namespace NUMINAMATH_CALUDE_calculate_expression_l3240_324051

theorem calculate_expression : 10 + 7 * (3 + 8)^2 = 857 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3240_324051


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_max_value_is_five_l3240_324079

theorem max_value_inequality (a : ℝ) : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ a) → a ≤ 5 :=
by
  sorry

theorem max_value_achieved : 
  ∃ x : ℝ, x^2 + |2*x - 6| = 5 :=
by
  sorry

theorem max_value_is_five : 
  (∀ x : ℝ, x^2 + |2*x - 6| ≥ 5) ∧ 
  (∃ x : ℝ, x^2 + |2*x - 6| = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_achieved_max_value_is_five_l3240_324079


namespace NUMINAMATH_CALUDE_total_coins_is_32_l3240_324060

/-- The number of dimes -/
def num_dimes : ℕ := 22

/-- The number of quarters -/
def num_quarters : ℕ := 10

/-- The total number of coins -/
def total_coins : ℕ := num_dimes + num_quarters

/-- Theorem: The total number of coins is 32 -/
theorem total_coins_is_32 : total_coins = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_is_32_l3240_324060


namespace NUMINAMATH_CALUDE_element_in_set_l3240_324034

theorem element_in_set : ∀ (a b : ℕ), 1 ∈ ({a, b, 1} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_element_in_set_l3240_324034


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3240_324067

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 ∧ 
    (∃ (y : ℕ), 5 * x = y^2) ∧ 
    (∃ (z : ℕ), 3 * x = z^3) → 
    x ≥ n) ∧
  n = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3240_324067


namespace NUMINAMATH_CALUDE_modified_car_distance_increase_l3240_324029

/-- Proves the increase in travel distance after modifying a car's fuel efficiency -/
theorem modified_car_distance_increase
  (original_efficiency : ℝ)
  (tank_capacity : ℝ)
  (fuel_reduction_factor : ℝ)
  (h1 : original_efficiency = 32)
  (h2 : tank_capacity = 12)
  (h3 : fuel_reduction_factor = 0.8)
  : (tank_capacity * (original_efficiency / fuel_reduction_factor) - tank_capacity * original_efficiency) = 76.8 := by
  sorry

end NUMINAMATH_CALUDE_modified_car_distance_increase_l3240_324029


namespace NUMINAMATH_CALUDE_unique_six_digit_square_split_l3240_324042

def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2

def contains_no_zero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → d ≠ 10

theorem unique_six_digit_square_split :
  ∃! n : ℕ,
    100000 ≤ n ∧ n ≤ 999999 ∧
    (∃ k : ℕ, n = k^2) ∧
    (∃ a b : ℕ, n = a * 1000 + b ∧
      is_three_digit_square a ∧
      is_three_digit_square b ∧
      contains_no_zero a ∧
      contains_no_zero b) :=
sorry

end NUMINAMATH_CALUDE_unique_six_digit_square_split_l3240_324042


namespace NUMINAMATH_CALUDE_kyle_bottles_l3240_324047

theorem kyle_bottles (bottle_capacity : ℕ) (additional_bottles : ℕ) (total_stars : ℕ) :
  bottle_capacity = 15 →
  additional_bottles = 3 →
  total_stars = 75 →
  ∃ (initial_bottles : ℕ), initial_bottles = 2 ∧ 
    (initial_bottles + additional_bottles) * bottle_capacity = total_stars :=
by sorry

end NUMINAMATH_CALUDE_kyle_bottles_l3240_324047


namespace NUMINAMATH_CALUDE_spinner_probability_divisible_by_3_l3240_324040

/-- A spinner with 8 equal sections numbered from 1 to 8 -/
def Spinner := Finset (Fin 8)

/-- The set of numbers on the spinner that are divisible by 3 -/
def DivisibleBy3 (s : Spinner) : Finset (Fin 8) :=
  s.filter (fun n => n % 3 = 0)

/-- The probability of an event on the spinner -/
def Probability (event : Finset (Fin 8)) (s : Spinner) : ℚ :=
  event.card / s.card

theorem spinner_probability_divisible_by_3 (s : Spinner) :
  Probability (DivisibleBy3 s) s = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_spinner_probability_divisible_by_3_l3240_324040


namespace NUMINAMATH_CALUDE_average_cat_weight_l3240_324086

def cat_weights : List ℝ := [12, 12, 14.7, 9.3]

theorem average_cat_weight :
  (cat_weights.sum / cat_weights.length) = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_cat_weight_l3240_324086


namespace NUMINAMATH_CALUDE_division_and_addition_l3240_324005

theorem division_and_addition : (150 / (10 / 2)) + 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_l3240_324005


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3240_324096

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3240_324096


namespace NUMINAMATH_CALUDE_gcd_65_130_l3240_324077

theorem gcd_65_130 : Nat.gcd 65 130 = 65 := by
  sorry

end NUMINAMATH_CALUDE_gcd_65_130_l3240_324077


namespace NUMINAMATH_CALUDE_fruit_water_content_l3240_324011

theorem fruit_water_content (m : ℝ) : 
  m > 0 ∧ m ≤ 100 →  -- m is a percentage, so it's between 0 and 100
  (100 - m + m * (1 - (m - 5) / 100) = 50) →  -- equation from step 6 in the solution
  m = 80 := by sorry

end NUMINAMATH_CALUDE_fruit_water_content_l3240_324011


namespace NUMINAMATH_CALUDE_inequality_proof_l3240_324026

theorem inequality_proof (n : ℕ) : (2*n + 1)^n ≥ (2*n)^n + (2*n - 1)^n := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3240_324026


namespace NUMINAMATH_CALUDE_number_square_puzzle_l3240_324018

theorem number_square_puzzle : ∃ x : ℝ, x^2 + 95 = (x - 20)^2 ∧ x = 7.625 := by
  sorry

end NUMINAMATH_CALUDE_number_square_puzzle_l3240_324018


namespace NUMINAMATH_CALUDE_cylinder_fill_cost_l3240_324036

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The cost to fill a cylinder with gasoline -/
def fillCost (c : Cylinder) (price : ℝ) : ℝ := c.radius^2 * c.height * price

/-- The theorem statement -/
theorem cylinder_fill_cost 
  (canB canN : Cylinder) 
  (h_radius : canN.radius = 2 * canB.radius) 
  (h_height : canN.height = canB.height / 2) 
  (h_half_cost : fillCost { radius := canB.radius, height := canB.height / 2 } (8 / (π * canB.radius^2 * canB.height)) = 4) :
  fillCost canN (8 / (π * canB.radius^2 * canB.height)) = 16 := by
  sorry


end NUMINAMATH_CALUDE_cylinder_fill_cost_l3240_324036


namespace NUMINAMATH_CALUDE_binary_51_l3240_324004

/-- The binary representation of a natural number -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Convert a list of booleans to a natural number, interpreting it as binary -/
def fromBinary (l : List Bool) : ℕ :=
  l.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem binary_51 :
  toBinary 51 = [true, true, false, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_binary_51_l3240_324004


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3240_324049

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome :
  ∀ k : ℕ,
    k > 8 ∧ 
    isPalindrome k 3 ∧ 
    isPalindrome k 5 →
    k ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l3240_324049


namespace NUMINAMATH_CALUDE_smallest_music_class_size_l3240_324033

theorem smallest_music_class_size :
  ∀ (x : ℕ),
  (∃ (total : ℕ), total = 5 * x + 2 ∧ total > 40) →
  (∀ (y : ℕ), y < x → ¬(∃ (total : ℕ), total = 5 * y + 2 ∧ total > 40)) →
  5 * x + 2 = 42 :=
by sorry

end NUMINAMATH_CALUDE_smallest_music_class_size_l3240_324033


namespace NUMINAMATH_CALUDE_complex_cube_absolute_value_l3240_324044

theorem complex_cube_absolute_value : 
  Complex.abs ((1 + 2 * Complex.I + 3 - Real.sqrt 3 * Complex.I) ^ 3) = 
  (23 - 4 * Real.sqrt 3) ^ (3/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_absolute_value_l3240_324044


namespace NUMINAMATH_CALUDE_sqrt_400_divided_by_2_l3240_324073

theorem sqrt_400_divided_by_2 : Real.sqrt 400 / 2 = 10 := by sorry

end NUMINAMATH_CALUDE_sqrt_400_divided_by_2_l3240_324073


namespace NUMINAMATH_CALUDE_lesser_fraction_l3240_324091

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 13 / 14)
  (prod_eq : x * y = 1 / 8) :
  min x y = (13 - Real.sqrt 113) / 28 := by
  sorry

end NUMINAMATH_CALUDE_lesser_fraction_l3240_324091


namespace NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l3240_324095

theorem range_of_k_for_quadratic_inequality :
  {k : ℝ | ∀ x : ℝ, k * x^2 - k * x - 1 < 0} = {k : ℝ | -4 < k ∧ k ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l3240_324095


namespace NUMINAMATH_CALUDE_simplify_equation_l3240_324000

theorem simplify_equation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 7*x^2 + x + 1 = 0 ↔ x^2*(y^2 + y - 9) = 0 :=
by sorry

end NUMINAMATH_CALUDE_simplify_equation_l3240_324000


namespace NUMINAMATH_CALUDE_fish_catch_calculation_l3240_324027

/-- Prove that given the conditions, Erica caught 80 kg of fish in the past four months --/
theorem fish_catch_calculation (price : ℝ) (total_earnings : ℝ) (past_catch : ℝ) :
  price = 20 →
  total_earnings = 4800 →
  total_earnings = price * (past_catch + 2 * past_catch) →
  past_catch = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_catch_calculation_l3240_324027


namespace NUMINAMATH_CALUDE_average_price_per_book_l3240_324076

theorem average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℕ) 
  (h1 : books1 = 32)
  (h2 : books2 = 60)
  (h3 : price1 = 1500)
  (h4 : price2 = 340) :
  (price1 + price2) / (books1 + books2) = 20 := by
sorry

end NUMINAMATH_CALUDE_average_price_per_book_l3240_324076


namespace NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l3240_324009

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of blue faces after cutting the cube into unit cubes -/
def blue_faces (c : Cube n) : ℕ :=
  6 * n^2

/-- The total number of faces of all unit cubes -/
def total_faces (c : Cube n) : ℕ :=
  6 * n^3

/-- Theorem stating that exactly one-third of the faces are blue iff n = 3 -/
theorem one_third_blue_faces_iff_three {n : ℕ} (c : Cube n) :
  3 * blue_faces c = total_faces c ↔ n = 3 :=
sorry

end NUMINAMATH_CALUDE_one_third_blue_faces_iff_three_l3240_324009


namespace NUMINAMATH_CALUDE_marching_band_composition_l3240_324012

theorem marching_band_composition (total : ℕ) (brass : ℕ) (woodwind : ℕ) (percussion : ℕ)
  (h1 : total = 110)
  (h2 : woodwind = 2 * brass)
  (h3 : percussion = 4 * woodwind)
  (h4 : total = brass + woodwind + percussion) :
  brass = 10 := by
sorry

end NUMINAMATH_CALUDE_marching_band_composition_l3240_324012


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_is_nonzero_constant_l3240_324083

/-- A sequence that is both arithmetic and geometric is non-zero constant -/
theorem arithmetic_and_geometric_is_nonzero_constant (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) →  -- arithmetic sequence condition
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∃ c : ℝ, c ≠ 0 ∧ ∀ n : ℕ, a n = c) :=      -- non-zero constant sequence
by sorry

end NUMINAMATH_CALUDE_arithmetic_and_geometric_is_nonzero_constant_l3240_324083


namespace NUMINAMATH_CALUDE_second_loan_amount_l3240_324087

def initial_loan : ℝ := 40
def final_debt : ℝ := 30

theorem second_loan_amount (half_paid_back : ℝ) (second_loan : ℝ) 
  (h1 : half_paid_back = initial_loan / 2)
  (h2 : final_debt = initial_loan - half_paid_back + second_loan) : 
  second_loan = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_loan_amount_l3240_324087


namespace NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l3240_324055

theorem not_prime_two_pow_plus_one (n m : ℕ) (h1 : m > 1) (h2 : Odd m) (h3 : m ∣ n) :
  ¬ Prime (2^n + 1) := by
sorry

end NUMINAMATH_CALUDE_not_prime_two_pow_plus_one_l3240_324055


namespace NUMINAMATH_CALUDE_problem_statement_l3240_324063

theorem problem_statement (x y : ℝ) (h : (y + 1)^2 + Real.sqrt (x - 2) = 0) : y^x = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3240_324063


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l3240_324066

def product : ℕ := 77 * 79 * 81 * 83

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 5 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l3240_324066


namespace NUMINAMATH_CALUDE_solution_part_I_solution_part_II_l3240_324041

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |x + 1|

-- Theorem for part I
theorem solution_part_I :
  ∀ x : ℝ, f 1 x < 3 ↔ -1 < x ∧ x < 1 :=
by sorry

-- Theorem for part II
theorem solution_part_II :
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x ∧ f a x = 1) ↔ a = -4 ∨ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_solution_part_I_solution_part_II_l3240_324041


namespace NUMINAMATH_CALUDE_range_of_a_l3240_324075

theorem range_of_a (x a : ℝ) : 
  (∀ x, (1/2 ≤ x ∧ x ≤ 1) → ¬((x-a)*(x-a-1) > 0)) ∧ 
  (∃ x, ¬(1/2 ≤ x ∧ x ≤ 1) ∧ ¬((x-a)*(x-a-1) > 0)) →
  (0 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3240_324075


namespace NUMINAMATH_CALUDE_subway_fare_cost_l3240_324016

/-- The cost of the subway fare each way given Brian's spending and constraints -/
theorem subway_fare_cost (apple_cost : ℚ) (kiwi_cost : ℚ) (banana_cost : ℚ) 
  (initial_money : ℚ) (max_apples : ℕ) :
  apple_cost = 14 / 12 →
  kiwi_cost = 10 →
  banana_cost = 5 →
  initial_money = 50 →
  max_apples = 24 →
  (initial_money - kiwi_cost - banana_cost - (↑max_apples * apple_cost)) / 2 = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_subway_fare_cost_l3240_324016


namespace NUMINAMATH_CALUDE_delta_nabla_equality_l3240_324053

/-- Definition of the Δ operation -/
def delta (a b : ℕ) : ℕ := 3 * a + 2 * b

/-- Definition of the ∇ operation -/
def nabla (a b : ℕ) : ℕ := 2 * a + 3 * b

/-- Theorem stating that 3 Δ (2 ∇ 1) = 23 -/
theorem delta_nabla_equality : delta 3 (nabla 2 1) = 23 := by
  sorry

end NUMINAMATH_CALUDE_delta_nabla_equality_l3240_324053


namespace NUMINAMATH_CALUDE_lottery_theorem_l3240_324085

/-- Calculates the remaining amount for fun after deducting taxes, student loan payment, savings, and stock market investment from lottery winnings. -/
def remaining_for_fun (lottery_winnings : ℚ) : ℚ :=
  let after_taxes := lottery_winnings / 2
  let after_student_loans := after_taxes - (after_taxes / 3)
  let after_savings := after_student_loans - 1000
  let stock_investment := 1000 / 5
  after_savings - stock_investment

/-- Theorem stating that given a lottery winning of 12006, the remaining amount for fun is 2802. -/
theorem lottery_theorem : remaining_for_fun 12006 = 2802 := by
  sorry

#eval remaining_for_fun 12006

end NUMINAMATH_CALUDE_lottery_theorem_l3240_324085


namespace NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3240_324006

theorem boys_neither_happy_nor_sad (total_children total_boys total_girls happy_children sad_children neither_children happy_boys sad_girls : ℕ) : 
  total_children = 60 →
  total_boys = 16 →
  total_girls = 44 →
  happy_children = 30 →
  sad_children = 10 →
  neither_children = 20 →
  happy_boys = 6 →
  sad_girls = 4 →
  total_children = total_boys + total_girls →
  happy_children + sad_children + neither_children = total_children →
  (total_boys - happy_boys - (sad_children - sad_girls) = 4) :=
by sorry

end NUMINAMATH_CALUDE_boys_neither_happy_nor_sad_l3240_324006


namespace NUMINAMATH_CALUDE_nina_homework_calculation_l3240_324084

/-- Nina's homework calculation -/
theorem nina_homework_calculation
  (ruby_math : ℕ) (ruby_reading : ℕ)
  (nina_math_multiplier : ℕ) (nina_reading_multiplier : ℕ)
  (h_ruby_math : ruby_math = 6)
  (h_ruby_reading : ruby_reading = 2)
  (h_nina_math : nina_math_multiplier = 4)
  (h_nina_reading : nina_reading_multiplier = 8) :
  (ruby_math * nina_math_multiplier + ruby_math) +
  (ruby_reading * nina_reading_multiplier + ruby_reading) = 48 := by
  sorry

#check nina_homework_calculation

end NUMINAMATH_CALUDE_nina_homework_calculation_l3240_324084


namespace NUMINAMATH_CALUDE_remainder_of_x_plus_one_power_2011_l3240_324046

theorem remainder_of_x_plus_one_power_2011 (x : ℤ) :
  (x + 1)^2011 ≡ x [ZMOD (x^2 - x + 1)] := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_x_plus_one_power_2011_l3240_324046


namespace NUMINAMATH_CALUDE_sum_of_roots_l3240_324010

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x + 14*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 14*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2184 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3240_324010


namespace NUMINAMATH_CALUDE_two_digit_sum_divisibility_l3240_324092

theorem two_digit_sum_divisibility (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ∃ k : Int, (10 * a + b) + (10 * b + a) = 11 * k :=
by sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisibility_l3240_324092


namespace NUMINAMATH_CALUDE_asha_win_probability_l3240_324065

theorem asha_win_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 3/7)
  (h_tie : p_tie = 1/5) :
  1 - p_lose - p_tie = 13/35 := by
  sorry

end NUMINAMATH_CALUDE_asha_win_probability_l3240_324065


namespace NUMINAMATH_CALUDE_sequence_2017_l3240_324054

/-- Property P: If aₚ = aₖ, then aₚ₊₁ = aₖ₊₁ for p, q ∈ ℕ* -/
def PropertyP (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p ≠ 0 → q ≠ 0 → a p = a q → a (p + 1) = a (q + 1)

/-- The sequence satisfying the given conditions -/
def Sequence (a : ℕ → ℕ) : Prop :=
  PropertyP a ∧
  a 1 = 1 ∧
  a 2 = 2 ∧
  a 3 = 3 ∧
  a 5 = 2 ∧
  a 6 + a 7 + a 8 = 21

theorem sequence_2017 (a : ℕ → ℕ) (h : Sequence a) : a 2017 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2017_l3240_324054


namespace NUMINAMATH_CALUDE_tan_half_less_than_x_l3240_324058

theorem tan_half_less_than_x (x : ℝ) (h1 : 0 < x) (h2 : x ≤ π / 2) : Real.tan (x / 2) < x := by
  sorry

end NUMINAMATH_CALUDE_tan_half_less_than_x_l3240_324058


namespace NUMINAMATH_CALUDE_tan_theta_range_l3240_324090

-- Define the condition
def condition (θ : ℝ) : Prop := (Real.sin θ) / (Real.sqrt 3 * Real.cos θ + 1) > 1

-- Define the range of tan θ
def tan_range (x : ℝ) : Prop := x ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ioo (Real.sqrt 3 / 3) (Real.sqrt 2)

-- Theorem statement
theorem tan_theta_range (θ : ℝ) : condition θ → tan_range (Real.tan θ) := by sorry

end NUMINAMATH_CALUDE_tan_theta_range_l3240_324090


namespace NUMINAMATH_CALUDE_thief_speed_l3240_324099

/-- Proves that the thief's speed is 8 km/hr given the problem conditions -/
theorem thief_speed (initial_distance : ℝ) (policeman_speed : ℝ) (thief_distance : ℝ)
  (h1 : initial_distance = 100) -- Initial distance in meters
  (h2 : policeman_speed = 10) -- Policeman's speed in km/hr
  (h3 : thief_distance = 400) -- Distance thief runs before being overtaken in meters
  : ∃ (thief_speed : ℝ), thief_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_thief_speed_l3240_324099


namespace NUMINAMATH_CALUDE_correct_sums_count_l3240_324078

theorem correct_sums_count (total : ℕ) (correct : ℕ) (incorrect : ℕ)
  (h1 : incorrect = 2 * correct)
  (h2 : total = correct + incorrect)
  (h3 : total = 24) :
  correct = 8 := by
  sorry

end NUMINAMATH_CALUDE_correct_sums_count_l3240_324078


namespace NUMINAMATH_CALUDE_intersection_M_N_l3240_324008

def M : Set ℝ := {x | (x - 1)^2 < 4}

def N : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3240_324008


namespace NUMINAMATH_CALUDE_equilateral_triangle_coverage_l3240_324037

theorem equilateral_triangle_coverage (small_side : ℝ) (large_side : ℝ) : 
  small_side = 1 →
  large_side = 15 →
  (large_side / small_side) ^ 2 = 225 :=
by
  sorry

#check equilateral_triangle_coverage

end NUMINAMATH_CALUDE_equilateral_triangle_coverage_l3240_324037


namespace NUMINAMATH_CALUDE_yola_past_weight_l3240_324071

/-- Proves Yola's weight from 2 years ago given current weights and differences -/
theorem yola_past_weight 
  (yola_current : ℕ) 
  (wanda_yola_diff : ℕ) 
  (wanda_yola_past_diff : ℕ) 
  (h1 : yola_current = 220)
  (h2 : wanda_yola_diff = 30)
  (h3 : wanda_yola_past_diff = 80) : 
  yola_current - (wanda_yola_past_diff - wanda_yola_diff) = 170 := by
  sorry

#check yola_past_weight

end NUMINAMATH_CALUDE_yola_past_weight_l3240_324071


namespace NUMINAMATH_CALUDE_min_sum_of_sides_l3240_324001

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a + b)^2 - c^2 = 4 and C = 60°, then the minimum value of a + b is 4√3/3 -/
theorem min_sum_of_sides (a b c : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : Real.cos (Real.pi / 3) = (a^2 + b^2 - c^2) / (2 * a * b)) :
  ∃ (min_sum : ℝ), min_sum = 4 * Real.sqrt 3 / 3 ∧ ∀ x y, (x + y)^2 - c^2 = 4 → x + y ≥ min_sum :=
sorry


end NUMINAMATH_CALUDE_min_sum_of_sides_l3240_324001
