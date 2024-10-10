import Mathlib

namespace system_solution_system_solution_values_no_solution_fractional_equation_l2055_205577

/-- Proves that the system of equations 2x - 7y = 5 and 3x - 8y = 10 has a unique solution -/
theorem system_solution : ∃! (x y : ℝ), 2*x - 7*y = 5 ∧ 3*x - 8*y = 10 := by sorry

/-- Proves that x = 6 and y = 1 is the solution to the system of equations -/
theorem system_solution_values : 
  ∀ (x y : ℝ), (2*x - 7*y = 5 ∧ 3*x - 8*y = 10) → (x = 6 ∧ y = 1) := by sorry

/-- Proves that the equation 3/(x-1) - (x+2)/(x(x-1)) = 0 has no solution -/
theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ 3/(x-1) - (x+2)/(x*(x-1)) = 0 := by sorry

end system_solution_system_solution_values_no_solution_fractional_equation_l2055_205577


namespace factorization_validity_l2055_205560

theorem factorization_validity (x y : ℝ) :
  x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y) := by
  sorry

end factorization_validity_l2055_205560


namespace daniel_purchase_cost_l2055_205523

/-- The total cost of items bought by Daniel -/
def total_cost (tax_amount : ℚ) (tax_rate : ℚ) (tax_free_cost : ℚ) : ℚ :=
  (tax_amount / tax_rate) + tax_free_cost

/-- Theorem stating the total cost of items Daniel bought -/
theorem daniel_purchase_cost :
  let tax_amount : ℚ := 30 / 100  -- 30 paise = 0.30 rupees
  let tax_rate : ℚ := 6 / 100     -- 6%
  let tax_free_cost : ℚ := 347 / 10  -- Rs. 34.7
  total_cost tax_amount tax_rate tax_free_cost = 397 / 10 := by
  sorry

#eval total_cost (30/100) (6/100) (347/10)

end daniel_purchase_cost_l2055_205523


namespace elena_marco_sum_ratio_l2055_205599

def sum_odd_integers (n : ℕ) : ℕ := n * n

def sum_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem elena_marco_sum_ratio :
  (sum_odd_integers 250) / (sum_integers 250) = 2 := by
  sorry

end elena_marco_sum_ratio_l2055_205599


namespace fib_sum_39_40_l2055_205537

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- State the theorem
theorem fib_sum_39_40 : fib 39 + fib 40 = fib 41 := by
  sorry

end fib_sum_39_40_l2055_205537


namespace russian_players_pairing_probability_l2055_205517

/-- The probability of all Russian players being paired exclusively with other Russian players
    in a random pairing of 10 tennis players, where 4 are from Russia. -/
theorem russian_players_pairing_probability :
  let total_players : ℕ := 10
  let russian_players : ℕ := 4
  let probability : ℚ := (russian_players - 1) / (total_players - 1) *
                         (russian_players - 3) / (total_players - 3)
  probability = 1 / 21 := by
  sorry

end russian_players_pairing_probability_l2055_205517


namespace fair_prize_division_l2055_205557

/-- Represents the state of the game --/
structure GameState where
  player1_wins : ℕ
  player2_wins : ℕ

/-- Calculates the probability of a player winning the game from a given state --/
def win_probability (state : GameState) : ℚ :=
  1 - (1/2) ^ (6 - state.player1_wins)

/-- Theorem stating the fair division of the prize --/
theorem fair_prize_division (state : GameState) 
  (h1 : state.player1_wins = 5)
  (h2 : state.player2_wins = 3) :
  let p1_prob := win_probability state
  let p2_prob := 1 - p1_prob
  (p1_prob : ℚ) / p2_prob = 7 / 1 := by sorry

end fair_prize_division_l2055_205557


namespace fixed_point_on_line_l2055_205518

theorem fixed_point_on_line (m : ℝ) : (2 : ℝ) + 1 = m * ((2 : ℝ) - 2) := by sorry

end fixed_point_on_line_l2055_205518


namespace house_height_calculation_l2055_205528

/-- The height of Lily's house in feet -/
def house_height : ℝ := 56.25

/-- The length of the shadow cast by Lily's house in feet -/
def house_shadow : ℝ := 75

/-- The height of the tree in feet -/
def tree_height : ℝ := 15

/-- The length of the shadow cast by the tree in feet -/
def tree_shadow : ℝ := 20

/-- Theorem stating that the calculated house height is correct -/
theorem house_height_calculation :
  house_height = tree_height * (house_shadow / tree_shadow) :=
by sorry

end house_height_calculation_l2055_205528


namespace area_of_quadrilateral_OBEC_area_of_quadrilateral_OBEC_proof_l2055_205530

/-- A line with slope -3 passing through points A, B, and E -/
structure Line1 where
  slope : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  E : ℝ × ℝ

/-- Another line passing through points C, D, and E -/
structure Line2 where
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Definition of the problem setup -/
def ProblemSetup (l1 : Line1) (l2 : Line2) : Prop :=
  l1.slope = -3 ∧
  l1.A.1 > 0 ∧ l1.A.2 = 0 ∧
  l1.B.1 = 0 ∧ l1.B.2 > 0 ∧
  l1.E = (3, 3) ∧
  l2.C = (6, 0) ∧
  l2.D.1 = 0 ∧ l2.D.2 ≠ 0 ∧
  l2.E = (3, 3)

/-- The main theorem to prove -/
theorem area_of_quadrilateral_OBEC (l1 : Line1) (l2 : Line2) 
  (h : ProblemSetup l1 l2) : ℝ :=
  22.5

/-- Proof of the theorem -/
theorem area_of_quadrilateral_OBEC_proof (l1 : Line1) (l2 : Line2) 
  (h : ProblemSetup l1 l2) : 
  area_of_quadrilateral_OBEC l1 l2 h = 22.5 := by
  sorry

end area_of_quadrilateral_OBEC_area_of_quadrilateral_OBEC_proof_l2055_205530


namespace percent_relation_l2055_205527

/-- Given that c is 25% of a and 10% of b, prove that b is 250% of a. -/
theorem percent_relation (a b c : ℝ) 
  (h1 : c = 0.25 * a) 
  (h2 : c = 0.10 * b) : 
  b = 2.5 * a := by
sorry

end percent_relation_l2055_205527


namespace unique_solution_for_reciprocal_squares_sum_l2055_205524

theorem unique_solution_for_reciprocal_squares_sum (x y z t : ℕ+) :
  (1 : ℚ) / x^2 + (1 : ℚ) / y^2 + (1 : ℚ) / z^2 + (1 : ℚ) / t^2 = 1 →
  (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by sorry

end unique_solution_for_reciprocal_squares_sum_l2055_205524


namespace jose_join_time_l2055_205588

theorem jose_join_time (tom_investment : ℕ) (jose_investment : ℕ) (total_profit : ℕ) (jose_profit : ℕ) :
  tom_investment = 30000 →
  jose_investment = 45000 →
  total_profit = 54000 →
  jose_profit = 30000 →
  ∃ x : ℕ,
    x ≤ 12 ∧
    (tom_investment * 12) / (tom_investment * 12 + jose_investment * (12 - x)) =
    (total_profit - jose_profit) / total_profit ∧
    x = 2 :=
by sorry

end jose_join_time_l2055_205588


namespace probability_at_least_one_mistake_l2055_205556

-- Define the probability of making a mistake on a single question
def p_mistake : ℝ := 0.1

-- Define the number of questions
def n_questions : ℕ := 3

-- Theorem statement
theorem probability_at_least_one_mistake :
  1 - (1 - p_mistake) ^ n_questions = 1 - 0.9 ^ 3 := by
  sorry

end probability_at_least_one_mistake_l2055_205556


namespace arithmetic_operations_l2055_205535

theorem arithmetic_operations :
  (12 - (-5) + (-4) - 8 = 5) ∧
  (-1 - (1 + 1/2) * (1/3) / (-4)^2 = -33/32) :=
by
  sorry

end arithmetic_operations_l2055_205535


namespace plumber_distribution_theorem_l2055_205563

/-- The number of ways to distribute n plumbers to k houses -/
def distribute_plumbers (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else Nat.choose n 2 * (Nat.factorial k)

theorem plumber_distribution_theorem :
  distribute_plumbers 4 3 = Nat.choose 4 2 * (Nat.factorial 3) :=
sorry

end plumber_distribution_theorem_l2055_205563


namespace reeya_average_score_l2055_205534

def reeya_scores : List ℝ := [65, 67, 76, 80, 95]

theorem reeya_average_score :
  (reeya_scores.sum / reeya_scores.length : ℝ) = 76.6 := by
  sorry

end reeya_average_score_l2055_205534


namespace f_decreasing_interval_l2055_205505

open Real

/-- The function f(x) = x ln x is monotonically decreasing on the interval (0, 1/e) -/
theorem f_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < (1 : ℝ)/Real.exp 1 →
  x₁ * log x₁ > x₂ * log x₂ := by
  sorry

end f_decreasing_interval_l2055_205505


namespace bucket_weight_l2055_205551

theorem bucket_weight (c d : ℝ) : ℝ :=
  let weight_three_quarters : ℝ := c
  let weight_one_third : ℝ := d
  let weight_full : ℝ := (8 * c / 5) - (3 * d / 5)
  weight_full

#check bucket_weight

end bucket_weight_l2055_205551


namespace bacon_calories_per_strip_l2055_205508

theorem bacon_calories_per_strip 
  (total_calories : ℕ) 
  (bacon_percentage : ℚ) 
  (num_bacon_strips : ℕ) 
  (h1 : total_calories = 1250)
  (h2 : bacon_percentage = 1/5)
  (h3 : num_bacon_strips = 2) :
  (total_calories : ℚ) * bacon_percentage / num_bacon_strips = 125 := by
sorry

end bacon_calories_per_strip_l2055_205508


namespace smallest_n_satisfying_ratio_l2055_205531

def sum_odd_from_3 (n : ℕ) : ℕ := n^2 + 2*n

def sum_even (n : ℕ) : ℕ := n*(n+1)

theorem smallest_n_satisfying_ratio : 
  ∀ n : ℕ, n > 0 → (n < 51 → (sum_odd_from_3 n : ℚ) / sum_even n ≠ 49/50) ∧
  (sum_odd_from_3 51 : ℚ) / sum_even 51 = 49/50 :=
sorry

end smallest_n_satisfying_ratio_l2055_205531


namespace area_between_circles_l2055_205514

-- Define the radius of the inner circle
def inner_radius : ℝ := 2

-- Define the radius of the outer circle
def outer_radius : ℝ := 2 * inner_radius

-- Define the width of the gray region
def width : ℝ := outer_radius - inner_radius

-- Theorem statement
theorem area_between_circles (h : width = 2) : 
  π * outer_radius^2 - π * inner_radius^2 = 12 * π := by
  sorry

end area_between_circles_l2055_205514


namespace one_third_of_1206_percent_of_400_l2055_205597

theorem one_third_of_1206_percent_of_400 : (1206 / 3) / 400 * 100 = 100.5 := by
  sorry

end one_third_of_1206_percent_of_400_l2055_205597


namespace x_minus_y_value_l2055_205594

theorem x_minus_y_value (x y : ℝ) (h1 : |x| = 5) (h2 : |y| = 3) (h3 : x * y > 0) :
  x - y = 2 ∨ x - y = -2 := by
sorry

end x_minus_y_value_l2055_205594


namespace positions_after_179_moves_l2055_205539

/-- Represents the positions of the cat -/
inductive CatPosition
| TopLeft
| TopRight
| BottomRight
| BottomLeft

/-- Represents the positions of the mouse -/
inductive MousePosition
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle
| TopLeft

/-- Calculates the position of the cat after a given number of moves -/
def catPositionAfterMoves (moves : Nat) : CatPosition :=
  match moves % 4 with
  | 0 => CatPosition.BottomLeft
  | 1 => CatPosition.TopLeft
  | 2 => CatPosition.TopRight
  | _ => CatPosition.BottomRight

/-- Calculates the position of the mouse after a given number of moves -/
def mousePositionAfterMoves (moves : Nat) : MousePosition :=
  match moves % 8 with
  | 0 => MousePosition.TopLeft
  | 1 => MousePosition.TopMiddle
  | 2 => MousePosition.TopRight
  | 3 => MousePosition.RightMiddle
  | 4 => MousePosition.BottomRight
  | 5 => MousePosition.BottomMiddle
  | 6 => MousePosition.BottomLeft
  | _ => MousePosition.LeftMiddle

theorem positions_after_179_moves :
  (catPositionAfterMoves 179 = CatPosition.BottomRight) ∧
  (mousePositionAfterMoves 179 = MousePosition.RightMiddle) := by
  sorry

end positions_after_179_moves_l2055_205539


namespace odd_sided_polygon_indivisible_l2055_205507

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry -- Additional conditions to ensure the polygon is regular

/-- The diameter of a polygon -/
def diameter (p : RegularPolygon n) : ℝ := sorry

/-- A division of a polygon into two parts -/
structure Division (p : RegularPolygon n) where
  part1 : Set (Fin n)
  part2 : Set (Fin n)
  is_partition : part1 ∪ part2 = univ ∧ part1 ∩ part2 = ∅

/-- The diameter of a part of a polygon -/
def part_diameter (p : RegularPolygon n) (part : Set (Fin n)) : ℝ := sorry

theorem odd_sided_polygon_indivisible (n : ℕ) (h : Odd n) (p : RegularPolygon n) :
  ∀ d : Division p, 
    part_diameter p d.part1 = diameter p ∨ part_diameter p d.part2 = diameter p := by
  sorry

end odd_sided_polygon_indivisible_l2055_205507


namespace logarithm_sum_simplification_l2055_205565

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + 1) +
  1 / (Real.log 7 / Real.log 14 + 1) = 1.5 := by sorry

end logarithm_sum_simplification_l2055_205565


namespace real_complex_condition_l2055_205550

theorem real_complex_condition (a : ℝ) : 
  (Complex.I * (a - 1)^2 + 4*a).im = 0 → a = 1 := by
  sorry

end real_complex_condition_l2055_205550


namespace f_monotone_and_inequality_l2055_205575

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem f_monotone_and_inequality (a : ℝ) : 
  (a > 0 ∧ a ≤ 2) ↔ 
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x > 0 → (x - 1) * f a x ≥ 0) :=
sorry

end f_monotone_and_inequality_l2055_205575


namespace exists_larger_area_same_perimeter_l2055_205536

/-- A shape in a 2D plane -/
structure Shape where
  area : ℝ
  perimeter : ℝ

/-- Theorem stating the existence of a shape with larger area and same perimeter -/
theorem exists_larger_area_same_perimeter (Φ Φ' : Shape) 
  (h1 : Φ'.area ≥ Φ.area) 
  (h2 : Φ'.perimeter < Φ.perimeter) : 
  ∃ Ψ : Shape, Ψ.perimeter = Φ.perimeter ∧ Ψ.area > Φ.area := by
  sorry

end exists_larger_area_same_perimeter_l2055_205536


namespace sum_first_three_special_sequence_l2055_205520

/-- An arithmetic sequence with given fourth, fifth, and sixth terms -/
def ArithmeticSequence (a₄ a₅ a₆ : ℤ) : ℕ → ℤ :=
  fun n => a₄ + (n - 4) * (a₅ - a₄)

/-- The sum of the first three terms of an arithmetic sequence -/
def SumFirstThree (seq : ℕ → ℤ) : ℤ :=
  seq 1 + seq 2 + seq 3

theorem sum_first_three_special_sequence :
  let seq := ArithmeticSequence 4 7 10
  SumFirstThree seq = -6 := by sorry

end sum_first_three_special_sequence_l2055_205520


namespace function_inequality_l2055_205569

theorem function_inequality (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f y + g x - g y ≥ Real.sin x + Real.cos y) :
  ∃ p q : ℝ → ℝ, 
    (∀ x : ℝ, f x = (Real.sin x + Real.cos x + p x - q x) / 2) ∧
    (∀ x : ℝ, g x = (Real.sin x - Real.cos x + p x + q x) / 2) ∧
    (∀ x y : ℝ, p x ≥ q y) := by
  sorry

end function_inequality_l2055_205569


namespace max_sum_of_shorter_l2055_205574

/-- Represents the configuration of houses -/
structure HouseConfig where
  one_story : ℕ
  two_story : ℕ

/-- The total number of floors in the city -/
def total_floors : ℕ := 30

/-- Calculates the sum of shorter houses seen from each roof -/
def sum_of_shorter (config : HouseConfig) : ℕ :=
  config.one_story * config.two_story

/-- Checks if a configuration is valid (i.e., totals 30 floors) -/
def is_valid_config (config : HouseConfig) : Prop :=
  config.one_story + 2 * config.two_story = total_floors

/-- The theorem to be proved -/
theorem max_sum_of_shorter :
  ∃ (config1 config2 : HouseConfig),
    is_valid_config config1 ∧
    is_valid_config config2 ∧
    sum_of_shorter config1 = 112 ∧
    sum_of_shorter config2 = 112 ∧
    (∀ (config : HouseConfig), is_valid_config config → sum_of_shorter config ≤ 112) ∧
    ((config1.one_story = 16 ∧ config1.two_story = 7) ∨
     (config1.one_story = 14 ∧ config1.two_story = 8)) ∧
    ((config2.one_story = 16 ∧ config2.two_story = 7) ∨
     (config2.one_story = 14 ∧ config2.two_story = 8)) ∧
    config1 ≠ config2 :=
  sorry

end max_sum_of_shorter_l2055_205574


namespace absolute_value_equality_l2055_205545

theorem absolute_value_equality (a b c d e f : ℝ) 
  (h1 : a * c * e ≠ 0)
  (h2 : ∀ x : ℝ, |a * x + b| + |c * x + d| = |e * x + f|) : 
  a * d = b * c := by sorry

end absolute_value_equality_l2055_205545


namespace apple_cost_calculation_l2055_205583

/-- The cost of three dozen apples in dollars -/
def cost_three_dozen : ℚ := 25.20

/-- The number of dozens we want to calculate the cost for -/
def target_dozens : ℕ := 4

/-- The cost of the target number of dozens of apples -/
def cost_target_dozens : ℚ := 33.60

/-- Theorem stating that the cost of the target number of dozens of apples is correct -/
theorem apple_cost_calculation : 
  (cost_three_dozen / 3) * target_dozens = cost_target_dozens := by
  sorry

end apple_cost_calculation_l2055_205583


namespace box_comparison_l2055_205568

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def box_lt (a b : Box) : Prop :=
  (a.x ≤ b.x ∧ a.y ≤ b.y ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.x ∧ a.y ≤ b.z ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.x ∧ a.z ≤ b.z) ∨
  (a.x ≤ b.y ∧ a.y ≤ b.z ∧ a.z ≤ b.x) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.x ∧ a.z ≤ b.y) ∨
  (a.x ≤ b.z ∧ a.y ≤ b.y ∧ a.z ≤ b.x)

def box_gt (a b : Box) : Prop := box_lt b a

theorem box_comparison :
  let A : Box := ⟨5, 6, 3⟩
  let B : Box := ⟨1, 5, 4⟩
  let C : Box := ⟨2, 2, 3⟩
  (box_gt A B) ∧ (box_lt C A) := by sorry

end box_comparison_l2055_205568


namespace shifted_direct_proportion_l2055_205592

/-- Given a direct proportion function y = -3x that is shifted down by 5 units,
    prove that the resulting function is y = -3x - 5 -/
theorem shifted_direct_proportion (x y : ℝ) :
  (y = -3 * x) → (y - 5 = -3 * x - 5) := by
  sorry

end shifted_direct_proportion_l2055_205592


namespace art_kit_student_ratio_is_two_to_one_l2055_205506

/-- Represents the art class scenario --/
structure ArtClass where
  students : ℕ
  art_kits : ℕ
  total_artworks : ℕ

/-- Calculates the ratio of art kits to students --/
def art_kit_student_ratio (ac : ArtClass) : Rat :=
  ac.art_kits / ac.students

/-- Theorem stating the ratio of art kits to students is 2:1 --/
theorem art_kit_student_ratio_is_two_to_one (ac : ArtClass) 
  (h1 : ac.students = 10)
  (h2 : ac.art_kits = 20)
  (h3 : ac.total_artworks = 35)
  (h4 : ∃ (n : ℕ), 2 * n = ac.students ∧ 
       n * 3 + n * 4 = ac.total_artworks) : 
  art_kit_student_ratio ac = 2 := by
  sorry

end art_kit_student_ratio_is_two_to_one_l2055_205506


namespace project_time_calculation_l2055_205595

theorem project_time_calculation (x y z : ℕ) : 
  x > 0 ∧ y > 0 ∧ z > 0 →
  y = (3 * x) / 2 →
  z = 2 * x →
  z = x + 20 →
  x + y + z = 90 :=
by sorry

end project_time_calculation_l2055_205595


namespace joint_investment_l2055_205529

def total_investment : ℝ := 5000

theorem joint_investment (x : ℝ) :
  ∃ (a b : ℝ),
    a + b = total_investment ∧
    a * (1 + x / 100) = 2100 ∧
    b * (1 + (x + 1) / 100) = 3180 ∧
    a = 2000 ∧
    b = 3000 :=
by sorry

end joint_investment_l2055_205529


namespace complex_arithmetic_equality_l2055_205573

theorem complex_arithmetic_equality : (98 * 76 - 679 * 8) / (24 * 6 + 25 * 25 * 3 - 3) = 1 := by
  sorry

end complex_arithmetic_equality_l2055_205573


namespace no_positive_solution_l2055_205500

theorem no_positive_solution :
  ¬ ∃ (a b c d : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
    a * d + b = c ∧
    Real.sqrt a * Real.sqrt d + Real.sqrt b = Real.sqrt c :=
by sorry

end no_positive_solution_l2055_205500


namespace min_value_ab_l2055_205576

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : b > 1)
  (h3 : (1/4 * Real.log a) * (Real.log b) = (1/4)^2) : 
  (∀ x y : ℝ, x > 1 → y > 1 → (1/4 * Real.log x) * (Real.log y) = (1/4)^2 → x * y ≥ a * b) →
  a * b = Real.exp 1 := by
sorry


end min_value_ab_l2055_205576


namespace systematic_sampling_l2055_205513

/-- Systematic sampling problem -/
theorem systematic_sampling
  (total_students : ℕ)
  (num_groups : ℕ)
  (group_size : ℕ)
  (group_16_number : ℕ)
  (h1 : total_students = 160)
  (h2 : num_groups = 20)
  (h3 : group_size = total_students / num_groups)
  (h4 : group_16_number = 126) :
  ∃ (first_group_number : ℕ),
    first_group_number ∈ Finset.range group_size ∧
    first_group_number + (16 - 1) * group_size = group_16_number :=
by sorry

end systematic_sampling_l2055_205513


namespace range_of_m_l2055_205509

/-- Proposition p: The equation x² + mx + 1 = 0 has two different negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition q: The equation 4x² + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The main theorem stating the equivalence of the given conditions and the solution -/
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end range_of_m_l2055_205509


namespace gabby_fruit_count_l2055_205590

/-- The number of fruits Gabby picked in total -/
def total_fruits (watermelons peaches plums : ℕ) : ℕ := watermelons + peaches + plums

/-- The number of watermelons Gabby got -/
def watermelons : ℕ := 1

/-- The number of peaches Gabby got -/
def peaches : ℕ := watermelons + 12

/-- The number of plums Gabby got -/
def plums : ℕ := peaches * 3

theorem gabby_fruit_count :
  total_fruits watermelons peaches plums = 53 := by
  sorry

end gabby_fruit_count_l2055_205590


namespace triangle_cosine_sum_max_l2055_205549

theorem triangle_cosine_sum_max (a b c : ℝ) (x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hxyz : x + y + z = π) : 
  (∃ (x y z : ℝ), x + y + z = π ∧ 
    a * Real.cos x + b * Real.cos y + c * Real.cos z ≤ (1/2) * (a*b/c + a*c/b + b*c/a)) :=
sorry

end triangle_cosine_sum_max_l2055_205549


namespace existence_of_x_y_satisfying_conditions_l2055_205564

theorem existence_of_x_y_satisfying_conditions : ∃ (x y : ℝ), x = y + 1 ∧ x^4 = y^4 := by
  sorry

end existence_of_x_y_satisfying_conditions_l2055_205564


namespace wire_cutting_problem_l2055_205585

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℚ) (shorter_length : ℝ) : 
  total_length = 60 →
  ratio = 2 / 4 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 40 := by
sorry

end wire_cutting_problem_l2055_205585


namespace action_figure_price_l2055_205543

theorem action_figure_price 
  (sneaker_cost : ℕ) 
  (initial_savings : ℕ) 
  (num_figures_sold : ℕ) 
  (money_left : ℕ) : 
  sneaker_cost = 90 →
  initial_savings = 15 →
  num_figures_sold = 10 →
  money_left = 25 →
  (sneaker_cost - initial_savings + money_left) / num_figures_sold = 10 := by
sorry

end action_figure_price_l2055_205543


namespace compound_hydrogen_atoms_l2055_205533

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  al : ℕ
  o : ℕ
  h : ℕ

/-- Represents the atomic weights of elements in g/mol -/
structure AtomicWeights where
  al : ℝ
  o : ℝ
  h : ℝ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (c : Compound) (w : AtomicWeights) : ℝ :=
  c.al * w.al + c.o * w.o + c.h * w.h

/-- The theorem to be proved -/
theorem compound_hydrogen_atoms :
  let c : Compound := { al := 1, o := 3, h := 3 }
  let w : AtomicWeights := { al := 27, o := 16, h := 1 }
  molecularWeight c w = 78 := by
  sorry

end compound_hydrogen_atoms_l2055_205533


namespace sum_of_min_max_T_l2055_205555

theorem sum_of_min_max_T (B M T : ℝ) 
  (h1 : B^2 + M^2 + T^2 = 2022) 
  (h2 : B + M + T = 72) : 
  ∃ (Tmin Tmax : ℝ), 
    (∀ T' : ℝ, (∃ B' M' : ℝ, B'^2 + M'^2 + T'^2 = 2022 ∧ B' + M' + T' = 72) → Tmin ≤ T' ∧ T' ≤ Tmax) ∧
    Tmin + Tmax = 48 :=
sorry

end sum_of_min_max_T_l2055_205555


namespace larger_tv_diagonal_l2055_205593

theorem larger_tv_diagonal (area_diff : ℝ) : 
  area_diff = 40 → 
  let small_tv_diagonal : ℝ := 19
  let small_tv_area : ℝ := (small_tv_diagonal / Real.sqrt 2) ^ 2
  let large_tv_area : ℝ := small_tv_area + area_diff
  let large_tv_diagonal : ℝ := Real.sqrt (2 * large_tv_area)
  large_tv_diagonal = 21 := by
sorry

end larger_tv_diagonal_l2055_205593


namespace transmission_time_is_256_seconds_l2055_205570

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 40

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 1024

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 160

/-- Theorem stating that the transmission time is 256 seconds -/
theorem transmission_time_is_256_seconds :
  (num_blocks * chunks_per_block) / transmission_rate = 256 := by
  sorry

end transmission_time_is_256_seconds_l2055_205570


namespace square_of_one_plus_i_l2055_205548

theorem square_of_one_plus_i (i : ℂ) (h : i^2 = -1) : (1 + i)^2 = 2*i := by
  sorry

end square_of_one_plus_i_l2055_205548


namespace sphere_surface_area_l2055_205566

theorem sphere_surface_area (V : ℝ) (S : ℝ) : 
  V = (32 / 3) * Real.pi → S = 4 * Real.pi * ((3 * V) / (4 * Real.pi))^(2/3) → S = 16 * Real.pi := by
  sorry

end sphere_surface_area_l2055_205566


namespace tan_810_degrees_undefined_l2055_205596

theorem tan_810_degrees_undefined : 
  ¬∃ (x : ℝ), Real.tan (810 * π / 180) = x :=
by
  sorry

end tan_810_degrees_undefined_l2055_205596


namespace no_prime_multiple_of_four_in_range_l2055_205553

theorem no_prime_multiple_of_four_in_range : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 50 → ¬(4 ∣ n ∧ Nat.Prime n ∧ n > 10) :=
by sorry

end no_prime_multiple_of_four_in_range_l2055_205553


namespace property_P_implications_l2055_205511

def has_property_P (f : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)

def d (f : ℕ → ℕ) (x : ℕ) : ℤ :=
  f (x + 1) - f x

theorem property_P_implications (f : ℕ → ℕ) (h : has_property_P f) :
  (∀ x : ℕ, d f x ≥ 0 ∧ d f (x + 1) ≤ d f x) ∧
  (∃ c : ℕ, c ≤ d f 1 ∧ Set.Infinite {n : ℕ | d f n = c}) :=
sorry

end property_P_implications_l2055_205511


namespace rachel_age_when_emily_half_her_age_l2055_205519

def emily_current_age : ℕ := 20
def rachel_current_age : ℕ := 24

theorem rachel_age_when_emily_half_her_age :
  ∃ (x : ℕ), 
    (rachel_current_age - x = 2 * (emily_current_age - x)) ∧
    (rachel_current_age - x = 8) := by
  sorry

end rachel_age_when_emily_half_her_age_l2055_205519


namespace smallest_equal_packages_l2055_205571

theorem smallest_equal_packages (hamburger_pack : ℕ) (bun_pack : ℕ) : 
  hamburger_pack = 10 → bun_pack = 15 → 
  (∃ (h b : ℕ), h * hamburger_pack = b * bun_pack ∧ 
   ∀ (h' b' : ℕ), h' * hamburger_pack = b' * bun_pack → h ≤ h') → 
  (∃ (h : ℕ), h * hamburger_pack = 3 * hamburger_pack ∧ 
   ∃ (b : ℕ), b * bun_pack = 3 * hamburger_pack) :=
by sorry

end smallest_equal_packages_l2055_205571


namespace some_number_exists_l2055_205559

theorem some_number_exists : ∃ N : ℝ, 
  (2 * ((3.6 * 0.48 * N) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002) ∧ 
  (abs (N - 2.5) < 0.0000000000000005) := by
  sorry

end some_number_exists_l2055_205559


namespace max_value_inequality_l2055_205581

theorem max_value_inequality (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) : 
  (a * b / (a + b + 1)) + (b * c / (b + c + 1)) + (c * a / (c + a + 1)) ≤ 2 / 3 := by
  sorry

end max_value_inequality_l2055_205581


namespace probability_of_white_after_red_l2055_205562

/-- Represents the number of balls in the box -/
def total_balls : ℕ := 20

/-- Represents the initial number of red balls -/
def initial_red_balls : ℕ := 10

/-- Represents the initial number of white balls -/
def initial_white_balls : ℕ := 10

/-- Represents that the first person draws a red ball -/
def first_draw_red : Prop := true

/-- The probability of drawing a white ball after a red ball is drawn -/
def prob_white_after_red : ℚ := 10 / 19

theorem probability_of_white_after_red :
  first_draw_red →
  prob_white_after_red = initial_white_balls / (total_balls - 1) :=
by sorry

end probability_of_white_after_red_l2055_205562


namespace intersection_of_modified_functions_l2055_205547

/-- Two functions that intersect at specific points -/
def IntersectingFunctions (p q : ℝ → ℝ) : Prop :=
  p 1 = q 1 ∧ p 1 = 1 ∧
  p 3 = q 3 ∧ p 3 = 3 ∧
  p 5 = q 5 ∧ p 5 = 5 ∧
  p 7 = q 7 ∧ p 7 = 7

/-- Theorem stating that given two functions p and q that intersect at specific points,
    the functions p(2x) and 2q(x) must intersect at (3.5, 7) -/
theorem intersection_of_modified_functions (p q : ℝ → ℝ) 
    (h : IntersectingFunctions p q) : 
    p (2 * 3.5) = 2 * q 3.5 ∧ p (2 * 3.5) = 7 := by
  sorry

end intersection_of_modified_functions_l2055_205547


namespace quadratic_roots_fraction_l2055_205541

theorem quadratic_roots_fraction (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
  sorry

end quadratic_roots_fraction_l2055_205541


namespace total_profit_calculation_l2055_205501

/-- Calculates the total profit given investments and one partner's share --/
def calculate_total_profit (anand_investment deepak_investment deepak_share : ℚ) : ℚ :=
  let total_parts := anand_investment + deepak_investment
  let deepak_parts := deepak_investment
  deepak_share * total_parts / deepak_parts

/-- The total profit is 1380.48 given the investments and Deepak's share --/
theorem total_profit_calculation (anand_investment deepak_investment deepak_share : ℚ) 
  (h1 : anand_investment = 2250)
  (h2 : deepak_investment = 3200)
  (h3 : deepak_share = 810.28) :
  calculate_total_profit anand_investment deepak_investment deepak_share = 1380.48 := by
  sorry

#eval calculate_total_profit 2250 3200 810.28

end total_profit_calculation_l2055_205501


namespace rational_square_fractional_parts_l2055_205522

def fractional_part (x : ℚ) : ℚ :=
  x - ↑(⌊x⌋)

theorem rational_square_fractional_parts (S : Set ℚ) :
  (∀ x ∈ S, fractional_part x ∈ {y | ∃ z ∈ S, fractional_part (z^2) = y}) →
  (∀ x ∈ S, fractional_part (x^2) ∈ {y | ∃ z ∈ S, fractional_part z = y}) →
  ∀ x ∈ S, ∃ n : ℤ, x = n := by
  sorry

end rational_square_fractional_parts_l2055_205522


namespace fraction_of_number_l2055_205561

theorem fraction_of_number (N : ℝ) : (0.4 * N = 204) → ((1/4) * (1/3) * (2/5) * N = 17) := by
  sorry

end fraction_of_number_l2055_205561


namespace train_speed_calculation_l2055_205542

/-- Given a train of length 600 m crossing an overbridge of length 100 m in 70 seconds,
    prove that the speed of the train is 36 km/h. -/
theorem train_speed_calculation (train_length : Real) (overbridge_length : Real) (crossing_time : Real)
    (h1 : train_length = 600)
    (h2 : overbridge_length = 100)
    (h3 : crossing_time = 70) :
    (train_length + overbridge_length) / crossing_time * 3.6 = 36 := by
  sorry

end train_speed_calculation_l2055_205542


namespace cyclic_sum_inequality_l2055_205521

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) := by
  sorry

end cyclic_sum_inequality_l2055_205521


namespace symmetric_points_on_number_line_l2055_205525

/-- Given points A, B, and C on a number line corresponding to real numbers a, b, and c respectively,
    with A and C symmetric with respect to B, a = √5, and b = 3, prove that c = 6 - √5. -/
theorem symmetric_points_on_number_line (a b c : ℝ) 
  (h_symmetric : b = (a + c) / 2) 
  (h_a : a = Real.sqrt 5) 
  (h_b : b = 3) : 
  c = 6 - Real.sqrt 5 := by
sorry

end symmetric_points_on_number_line_l2055_205525


namespace special_number_property_l2055_205586

theorem special_number_property (X : ℕ) : 
  (3 + X % 26 = X / 26) ∧ (X % 29 = X / 29) → X = 270 ∨ X = 540 := by
  sorry

end special_number_property_l2055_205586


namespace max_value_of_m_l2055_205526

theorem max_value_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, a > 0 → b > 0 → m / (3 * a + b) - 3 / a - 1 / b ≤ 0) : 
  m ≤ 16 :=
by sorry

end max_value_of_m_l2055_205526


namespace money_split_l2055_205589

theorem money_split (total : ℝ) (moses_percent : ℝ) (moses_esther_diff : ℝ) : 
  total = 50 ∧ moses_percent = 0.4 ∧ moses_esther_diff = 5 →
  ∃ (tony esther moses : ℝ),
    moses = total * moses_percent ∧
    tony + esther = total - moses ∧
    moses = esther + moses_esther_diff ∧
    tony = 15 ∧ esther = 15 :=
by sorry

end money_split_l2055_205589


namespace unique_quadratic_solution_l2055_205579

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 - 6 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                     -- sum condition
  (a > c) →                          -- inequality condition
  (a = 7 + 2 * Real.sqrt 10 ∧ c = 7 - 2 * Real.sqrt 10) := by
sorry

end unique_quadratic_solution_l2055_205579


namespace trig_identity_l2055_205503

theorem trig_identity (a : ℝ) (h : Real.sin (π / 6 - a) - Real.cos a = 1 / 3) :
  Real.cos (2 * a + π / 3) = 7 / 9 := by
  sorry

end trig_identity_l2055_205503


namespace opposite_of_one_half_l2055_205598

theorem opposite_of_one_half : -(1/2 : ℚ) = -1/2 := by sorry

end opposite_of_one_half_l2055_205598


namespace complex_magnitude_theorem_l2055_205546

theorem complex_magnitude_theorem (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 3 / w = s) : 
  Complex.abs w = (3 : ℝ) / 2 := by
sorry

end complex_magnitude_theorem_l2055_205546


namespace min_value_expression_l2055_205572

theorem min_value_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 ∧
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
    a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b ≥ m) ∧
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a^2 + b^2 + c^2 + 1/a^2 + b/a + c/b = m) :=
by
  sorry

end min_value_expression_l2055_205572


namespace ground_beef_cost_l2055_205580

/-- The price of ground beef per kilogram in dollars -/
def price_per_kg : ℝ := 5.00

/-- The quantity of ground beef in kilograms -/
def quantity : ℝ := 12

/-- The total cost of ground beef -/
def total_cost : ℝ := price_per_kg * quantity

theorem ground_beef_cost : total_cost = 60.00 := by
  sorry

end ground_beef_cost_l2055_205580


namespace hyperbola_eccentricity_l2055_205532

/-- A hyperbola with given parameters a and b -/
structure Hyperbola (a b : ℝ) :=
  (ha : a > 0)
  (hb : b > 0)

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The left focus of a hyperbola -/
def left_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- The asymptotes of a hyperbola -/
def asymptotes (h : Hyperbola a b) : (ℝ × ℝ → Prop) × (ℝ × ℝ → Prop) := sorry

/-- A point is in the first quadrant -/
def is_in_first_quadrant (p : ℝ × ℝ) : Prop := sorry

/-- A point lies on a line -/
def lies_on (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop := sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : ℝ × ℝ → Prop) : Prop := sorry

/-- Two lines are parallel -/
def parallel (l1 l2 : ℝ × ℝ → Prop) : Prop := sorry

/-- The line through two points -/
def line_through (p1 p2 : ℝ × ℝ) : ℝ × ℝ → Prop := sorry

theorem hyperbola_eccentricity (a b : ℝ) (h : Hyperbola a b) 
  (p : ℝ × ℝ) (hp1 : is_in_first_quadrant p) 
  (hp2 : lies_on p (asymptotes h).1) 
  (hp3 : perpendicular (line_through p (left_focus h)) (asymptotes h).2)
  (hp4 : parallel (line_through p (right_focus h)) (asymptotes h).2) :
  eccentricity h = 2 := by sorry

end hyperbola_eccentricity_l2055_205532


namespace teal_color_survey_l2055_205552

theorem teal_color_survey (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_more_blue : more_blue = 90)
  (h_both : both = 45)
  (h_neither : neither = 20) :
  ∃ (more_green : ℕ), more_green = 85 ∧ 
    total = more_blue + more_green - both + neither :=
by sorry

end teal_color_survey_l2055_205552


namespace square_tiles_l2055_205584

theorem square_tiles (n : ℕ) (h : n * n = 81) :
  n * n - n = 72 :=
sorry

end square_tiles_l2055_205584


namespace cube_surface_area_approx_l2055_205502

-- Define the dimensions of the rectangular prism
def prism_length : ℝ := 10
def prism_width : ℝ := 5
def prism_height : ℝ := 24

-- Define the volume of the rectangular prism
def prism_volume : ℝ := prism_length * prism_width * prism_height

-- Define the edge length of the cube with the same volume
def cube_edge : ℝ := (prism_volume) ^ (1/3)

-- Define the surface area of the cube
def cube_surface_area : ℝ := 6 * (cube_edge ^ 2)

-- Theorem stating that the surface area of the cube is approximately 677.76 square inches
theorem cube_surface_area_approx :
  ∃ ε > 0, |cube_surface_area - 677.76| < ε :=
sorry

end cube_surface_area_approx_l2055_205502


namespace power_of_three_squared_cubed_squared_l2055_205512

theorem power_of_three_squared_cubed_squared :
  ((3^2)^3)^2 = 531441 := by
  sorry

end power_of_three_squared_cubed_squared_l2055_205512


namespace range_proof_l2055_205544

theorem range_proof (a b : ℝ) 
  (h1 : 1 ≤ a + b) (h2 : a + b ≤ 5) 
  (h3 : -1 ≤ a - b) (h4 : a - b ≤ 3) : 
  (0 ≤ a ∧ a ≤ 4) ∧ 
  (-1 ≤ b ∧ b ≤ 3) ∧ 
  (-2 ≤ 3*a - 2*b ∧ 3*a - 2*b ≤ 10) := by
  sorry

end range_proof_l2055_205544


namespace brians_trip_distance_l2055_205587

/-- Calculates the distance traveled given car efficiency and fuel consumed -/
def distance_traveled (efficiency : ℝ) (fuel_consumed : ℝ) : ℝ :=
  efficiency * fuel_consumed

/-- Proves that given a car efficiency of 20 miles per gallon and a fuel consumption of 3 gallons, the distance traveled is 60 miles -/
theorem brians_trip_distance :
  distance_traveled 20 3 = 60 := by
  sorry

end brians_trip_distance_l2055_205587


namespace cube_dimension_ratio_l2055_205582

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 64) (h2 : v2 = 512) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end cube_dimension_ratio_l2055_205582


namespace horner_method_result_l2055_205591

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Theorem statement
theorem horner_method_result : f (-2) = 325.4 := by
  sorry

end horner_method_result_l2055_205591


namespace OPRQ_shape_l2055_205540

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral defined by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  R : Point
  Q : Point

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  (quad.P.x - quad.O.x, quad.P.y - quad.O.y) = (quad.R.x - quad.Q.x, quad.R.y - quad.Q.y) ∧
  (quad.Q.x - quad.O.x, quad.Q.y - quad.O.y) = (quad.R.x - quad.P.x, quad.R.y - quad.P.y)

/-- Checks if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Checks if a quadrilateral is a straight line -/
def isStraightLine (quad : Quadrilateral) : Prop :=
  areCollinear quad.O quad.P quad.Q ∧ areCollinear quad.O quad.Q quad.R

/-- Checks if a quadrilateral is a trapezoid -/
def isTrapezoid (quad : Quadrilateral) : Prop :=
  ((quad.P.x - quad.O.x) * (quad.R.y - quad.Q.y) = (quad.R.x - quad.Q.x) * (quad.P.y - quad.O.y) ∧
   (quad.Q.x - quad.O.x) * (quad.R.y - quad.P.y) ≠ (quad.R.x - quad.P.x) * (quad.Q.y - quad.O.y)) ∨
  ((quad.Q.x - quad.O.x) * (quad.R.y - quad.P.y) = (quad.R.x - quad.P.x) * (quad.Q.y - quad.O.y) ∧
   (quad.P.x - quad.O.x) * (quad.R.y - quad.Q.y) ≠ (quad.R.x - quad.Q.x) * (quad.P.y - quad.O.y))

theorem OPRQ_shape (x₁ y₁ x₂ y₂ : ℝ) (h1 : x₁ ≠ x₂) (h2 : y₁ ≠ y₂) :
  let quad := Quadrilateral.mk
    (Point.mk 0 0)
    (Point.mk x₁ y₁)
    (Point.mk (x₁ + 2*x₂) (y₁ + 2*y₂))
    (Point.mk x₂ y₂)
  ¬(isParallelogram quad) ∧ ¬(isStraightLine quad) ∧ (isTrapezoid quad ∨ (¬(isParallelogram quad) ∧ ¬(isStraightLine quad) ∧ ¬(isTrapezoid quad))) := by
  sorry

end OPRQ_shape_l2055_205540


namespace polar_to_cartesian_l2055_205554

/-- 
Given a point P with polar coordinates (r, θ), 
this theorem states that its Cartesian coordinates are (r cos(θ), r sin(θ)).
-/
theorem polar_to_cartesian (r θ : ℝ) : 
  let p : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)
  ∃ (x y : ℝ), p = (x, y) ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ := by
  sorry

end polar_to_cartesian_l2055_205554


namespace simplest_quadratic_radical_value_l2055_205578

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → y ≠ x-5 → ¬∃ (a b : ℝ), b > 0 ∧ b ≠ 1 ∧ x-5 = a^2 * b

theorem simplest_quadratic_radical_value :
  ∀ x : ℝ, x ∈ ({11, 13, 21, 29} : Set ℝ) →
    (is_simplest_quadratic_radical x ↔ x = 11) :=
by sorry

end simplest_quadratic_radical_value_l2055_205578


namespace dogwood_planting_correct_l2055_205516

/-- The number of dogwood trees planted in a park --/
def dogwood_trees_planted (current : ℕ) (total : ℕ) : ℕ :=
  total - current

/-- Theorem stating that the number of dogwood trees planted is correct --/
theorem dogwood_planting_correct (current : ℕ) (total : ℕ) 
  (h : current ≤ total) : 
  dogwood_trees_planted current total = total - current :=
by
  sorry

#eval dogwood_trees_planted 34 83

end dogwood_planting_correct_l2055_205516


namespace line_intersects_y_axis_l2055_205558

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line passing through two points
def Line (p1 p2 : Point2D) :=
  {p : Point2D | ∃ t : ℝ, p.x = p1.x + t * (p2.x - p1.x) ∧ p.y = p1.y + t * (p2.y - p1.y)}

-- The given points
def p1 : Point2D := ⟨2, 9⟩
def p2 : Point2D := ⟨4, 15⟩

-- The y-axis
def yAxis : Set Point2D := {p : Point2D | p.x = 0}

-- The intersection point
def intersectionPoint : Point2D := ⟨0, 3⟩

-- The theorem to prove
theorem line_intersects_y_axis :
  intersectionPoint ∈ Line p1 p2 ∩ yAxis := by sorry

end line_intersects_y_axis_l2055_205558


namespace max_a4_in_geometric_sequence_l2055_205515

theorem max_a4_in_geometric_sequence (a : ℕ → ℝ) :
  (∀ n : ℕ, a n > 0) →  -- positive sequence
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) →  -- geometric sequence
  a 3 + a 5 = 4 →  -- given condition
  ∀ x : ℝ, a 4 ≤ x → x ≤ 2  -- maximum value of a_4 is 2
:= by sorry

end max_a4_in_geometric_sequence_l2055_205515


namespace matrix_multiplication_result_l2055_205567

theorem matrix_multiplication_result : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 5, 2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -2, 1]
  A * B = !![6, 21; -4, 32] := by sorry

end matrix_multiplication_result_l2055_205567


namespace original_price_calculation_l2055_205510

-- Define the discounts
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.05

-- Define the final price
def final_price : ℝ := 266

-- Theorem statement
theorem original_price_calculation :
  ∃ P : ℝ, P * (1 - discount1) * (1 - discount2) = final_price ∧ P = 350 := by
  sorry

end original_price_calculation_l2055_205510


namespace incorrect_observation_value_l2055_205504

/-- Theorem: Given 20 observations with an original mean of 36, if one observation
    is corrected from an unknown value to 25, resulting in a new mean of 34.9,
    then the unknown (incorrect) value must have been 47. -/
theorem incorrect_observation_value
  (n : ℕ) -- number of observations
  (original_mean : ℝ) -- original mean
  (correct_value : ℝ) -- correct value of the observation
  (new_mean : ℝ) -- new mean after correction
  (h_n : n = 20)
  (h_original_mean : original_mean = 36)
  (h_correct_value : correct_value = 25)
  (h_new_mean : new_mean = 34.9)
  : ∃ (incorrect_value : ℝ),
    n * original_mean - incorrect_value + correct_value = n * new_mean ∧
    incorrect_value = 47 :=
sorry

end incorrect_observation_value_l2055_205504


namespace symmetric_complex_numbers_l2055_205538

theorem symmetric_complex_numbers (z₁ z₂ : ℂ) :
  (z₁ = 2 - 3*I) →
  (z₁ = -z₂) →
  (z₂ = -2 + 3*I) := by
sorry

end symmetric_complex_numbers_l2055_205538
