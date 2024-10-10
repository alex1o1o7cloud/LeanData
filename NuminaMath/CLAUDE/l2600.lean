import Mathlib

namespace inverse_sixteen_mod_97_l2600_260000

theorem inverse_sixteen_mod_97 (h : (8⁻¹ : ZMod 97) = 85) : (16⁻¹ : ZMod 97) = 47 := by
  sorry

end inverse_sixteen_mod_97_l2600_260000


namespace prob_three_different_suits_value_l2600_260028

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards per suit in a standard deck -/
def cards_per_suit : ℕ := standard_deck_size / number_of_suits

/-- The probability of picking three cards of different suits from a standard deck without replacement -/
def prob_three_different_suits : ℚ :=
  (cards_per_suit * (number_of_suits - 1) : ℚ) / (standard_deck_size - 1) *
  (cards_per_suit * (number_of_suits - 2) : ℚ) / (standard_deck_size - 2)

theorem prob_three_different_suits_value : 
  prob_three_different_suits = 169 / 425 := by sorry

end prob_three_different_suits_value_l2600_260028


namespace circle_square_area_difference_l2600_260098

theorem circle_square_area_difference :
  let square_side : ℝ := 12
  let circle_diameter : ℝ := 16
  let π : ℝ := 3
  let square_area := square_side ^ 2
  let circle_area := π * (circle_diameter / 2) ^ 2
  circle_area - square_area = 48 := by sorry

end circle_square_area_difference_l2600_260098


namespace rebus_solution_exists_l2600_260019

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def rebus_equation (h1 h2 h3 ch1 ch2 : ℕ) : Prop :=
  10 * h1 + h2 + 4000 + 400 * h3 + ch1 * 10 + ch2 = 4000 + 100 * h1 + 10 * h2 + h3

theorem rebus_solution_exists :
  ∃ (h1 h2 h3 ch1 ch2 : ℕ),
    is_odd h1 ∧ is_odd h2 ∧ is_odd h3 ∧
    is_even ch1 ∧ is_even ch2 ∧
    rebus_equation h1 h2 h3 ch1 ch2 ∧
    h1 = 5 ∧ h2 = 5 ∧ h3 = 5 :=
by sorry

end rebus_solution_exists_l2600_260019


namespace factors_of_2520_l2600_260049

/-- The number of distinct, positive factors of 2520 -/
def num_factors_2520 : ℕ :=
  (Finset.filter (· ∣ 2520) (Finset.range 2521)).card

/-- Theorem stating that the number of distinct, positive factors of 2520 is 48 -/
theorem factors_of_2520 : num_factors_2520 = 48 := by
  sorry

end factors_of_2520_l2600_260049


namespace all_semifinalists_advanced_no_semifinalists_eliminated_l2600_260081

/-- The number of semifinalists -/
def total_semifinalists : ℕ := 8

/-- The number of medal winners in the final round -/
def medal_winners : ℕ := 3

/-- The number of possible groups of medal winners -/
def possible_groups : ℕ := 56

/-- The number of semifinalists who advanced to the final round -/
def advanced_semifinalists : ℕ := total_semifinalists

theorem all_semifinalists_advanced :
  advanced_semifinalists = total_semifinalists ∧
  Nat.choose advanced_semifinalists medal_winners = possible_groups :=
by sorry

theorem no_semifinalists_eliminated :
  total_semifinalists - advanced_semifinalists = 0 :=
by sorry

end all_semifinalists_advanced_no_semifinalists_eliminated_l2600_260081


namespace x_squared_coefficient_is_13_l2600_260075

/-- The coefficient of x^2 in the expansion of ((1-x)^3 * (2x^2+1)^5) is 13 -/
theorem x_squared_coefficient_is_13 : 
  let f : Polynomial ℚ := (1 - X)^3 * (2*X^2 + 1)^5
  (f.coeff 2) = 13 := by sorry

end x_squared_coefficient_is_13_l2600_260075


namespace regular_nonagon_diagonal_sum_l2600_260004

/-- A regular nonagon -/
structure RegularNonagon where
  /-- Side length of the nonagon -/
  side_length : ℝ
  /-- Length of the shortest diagonal -/
  shortest_diagonal : ℝ
  /-- Length of the longest diagonal -/
  longest_diagonal : ℝ
  /-- The side length is positive -/
  side_length_pos : 0 < side_length

/-- 
In a regular nonagon, the length of the longest diagonal 
is equal to the sum of the side length and the shortest diagonal 
-/
theorem regular_nonagon_diagonal_sum (n : RegularNonagon) : 
  n.side_length + n.shortest_diagonal = n.longest_diagonal := by
  sorry


end regular_nonagon_diagonal_sum_l2600_260004


namespace imaginary_part_of_z_l2600_260099

-- Define the complex number -1-2i
def z : ℂ := -1 - 2 * Complex.I

-- Theorem stating that the imaginary part of z is -2
theorem imaginary_part_of_z :
  z.im = -2 := by sorry

end imaginary_part_of_z_l2600_260099


namespace factorization_equality_l2600_260035

theorem factorization_equality (x y : ℝ) : x^2 * y - 4*y = y*(x+2)*(x-2) := by
  sorry

end factorization_equality_l2600_260035


namespace necessary_but_not_sufficient_l2600_260037

def f (a : ℝ) (x : ℝ) := x^2 - 2*a*x + 1

theorem necessary_but_not_sufficient (a : ℝ) :
  (a ≤ 0 → ∀ x y, 1 ≤ x → x < y → f a x < f a y) ∧
  (∃ a > 0, ∀ x y, 1 ≤ x → x < y → f a x < f a y) :=
sorry

end necessary_but_not_sufficient_l2600_260037


namespace rectangles_in_5x4_grid_l2600_260079

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of rectangles in a grid of width w and height h -/
def total_rectangles (w h : ℕ) : ℕ :=
  w * rectangles_in_row h + h * rectangles_in_row w - w * h

theorem rectangles_in_5x4_grid :
  total_rectangles 5 4 = 24 := by sorry

end rectangles_in_5x4_grid_l2600_260079


namespace triangle_count_is_48_l2600_260044

/-- Represents the configuration of the rectangle and its divisions -/
structure RectangleConfig where
  vertical_divisions : Nat
  horizontal_divisions : Nat
  additional_horizontal_divisions : Nat

/-- Calculates the number of triangles in the described figure -/
def count_triangles (config : RectangleConfig) : Nat :=
  let initial_rectangles := config.vertical_divisions * config.horizontal_divisions
  let initial_triangles := 2 * initial_rectangles
  let additional_rectangles := initial_rectangles * config.additional_horizontal_divisions
  let additional_triangles := 2 * additional_rectangles
  initial_triangles + additional_triangles

/-- The specific configuration described in the problem -/
def problem_config : RectangleConfig :=
  { vertical_divisions := 3
  , horizontal_divisions := 2
  , additional_horizontal_divisions := 2 }

/-- Theorem stating that the number of triangles in the described figure is 48 -/
theorem triangle_count_is_48 : count_triangles problem_config = 48 := by
  sorry


end triangle_count_is_48_l2600_260044


namespace concentric_circles_no_common_tangents_l2600_260015

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define concentric circles
def concentric (c1 c2 : Circle) : Prop :=
  c1.center = c2.center ∧ c1.radius ≠ c2.radius

-- Define a tangent line to a circle
def is_tangent_to (line : ℝ × ℝ → ℝ) (c : Circle) : Prop :=
  ∃ (point : ℝ × ℝ), line point = 0 ∧ 
    (point.1 - c.center.1)^2 + (point.2 - c.center.2)^2 = c.radius^2

-- Theorem: Two concentric circles have 0 common tangents
theorem concentric_circles_no_common_tangents (c1 c2 : Circle) 
  (h : concentric c1 c2) : 
  ¬∃ (line : ℝ × ℝ → ℝ), is_tangent_to line c1 ∧ is_tangent_to line c2 :=
sorry

end concentric_circles_no_common_tangents_l2600_260015


namespace inequality_proof_l2600_260039

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  (1/x^2 + x) * (1/y^2 + y) * (1/z^2 + z) ≥ (28/3)^3 := by
  sorry

end inequality_proof_l2600_260039


namespace solution_set_is_correct_l2600_260062

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The solution set of the inequality floor(x)^2 - 5*floor(x) + 6 ≤ 0 -/
def solution_set : Set ℝ :=
  {x : ℝ | (floor x)^2 - 5*(floor x) + 6 ≤ 0}

/-- Theorem stating that the solution set is [2,4) -/
theorem solution_set_is_correct : solution_set = Set.Icc 2 4 := by
  sorry

end solution_set_is_correct_l2600_260062


namespace haley_video_files_l2600_260008

/-- Given the initial number of music files, the number of deleted files,
    and the remaining number of files, calculate the initial number of video files. -/
def initialVideoFiles (initialMusicFiles deletedFiles remainingFiles : ℕ) : ℕ :=
  remainingFiles + deletedFiles - initialMusicFiles

theorem haley_video_files :
  initialVideoFiles 27 11 58 = 42 := by
  sorry

end haley_video_files_l2600_260008


namespace monotonic_decrease_interval_l2600_260012

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |2 * x - 4|

-- State the theorem
theorem monotonic_decrease_interval
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a 1 = 9) :
  ∃ (I : Set ℝ), StrictMonoOn (f a) (Set.Iic 2) ∧ I = Set.Iic 2 := by
  sorry

end monotonic_decrease_interval_l2600_260012


namespace intersectionRangeOfB_l2600_260047

/-- Two lines y = 2x + 1 and y = 3x + b intersect in the third quadrant -/
def linesIntersectInThirdQuadrant (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = 2*x + 1 ∧ y = 3*x + b ∧ x < 0 ∧ y < 0

/-- The range of b for which the lines intersect in the third quadrant -/
theorem intersectionRangeOfB :
  ∀ b : ℝ, linesIntersectInThirdQuadrant b ↔ b > 3/2 :=
sorry

end intersectionRangeOfB_l2600_260047


namespace result_not_divisible_by_1998_l2600_260016

/-- The operation of multiplying by 2 and adding 1 -/
def operation (n : ℕ) : ℕ := 2 * n + 1

/-- The result of applying the operation k times to n -/
def iterate_operation (n k : ℕ) : ℕ :=
  match k with
  | 0 => n
  | k + 1 => operation (iterate_operation n k)

theorem result_not_divisible_by_1998 (n k : ℕ) :
  ¬(1998 ∣ iterate_operation n k) := by
  sorry

#check result_not_divisible_by_1998

end result_not_divisible_by_1998_l2600_260016


namespace largest_number_with_property_l2600_260092

theorem largest_number_with_property : ∃ n : ℕ, 
  (n ≤ 100) ∧ 
  ((n - 2) % 7 = 0) ∧ 
  ((n - 2) % 8 = 0) ∧ 
  (∀ m : ℕ, m ≤ 100 → ((m - 2) % 7 = 0) ∧ ((m - 2) % 8 = 0) → m ≤ n) ∧
  n = 58 := by
sorry

end largest_number_with_property_l2600_260092


namespace opposite_sides_line_range_l2600_260022

/-- Given two points (x₁, y₁) and (x₂, y₂) on opposite sides of the line 3x - 2y + a = 0,
    prove that the range of values for 'a' is -4 < a < 9 -/
theorem opposite_sides_line_range (x₁ y₁ x₂ y₂ : ℝ) (h : (3*x₁ - 2*y₁ + a) * (3*x₂ - 2*y₂ + a) < 0) :
  -4 < a ∧ a < 9 :=
sorry

end opposite_sides_line_range_l2600_260022


namespace greenfield_basketball_league_players_l2600_260003

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 6

/-- The additional cost of a T-shirt compared to a pair of socks in dollars -/
def tshirt_additional_cost : ℕ := 7

/-- The total expenditure on socks and T-shirts for all players in dollars -/
def total_expenditure : ℕ := 4092

/-- The number of pairs of socks required per player -/
def socks_per_player : ℕ := 2

/-- The number of T-shirts required per player -/
def tshirts_per_player : ℕ := 2

/-- The cost of equipment (socks and T-shirts) for one player in dollars -/
def cost_per_player : ℕ := 
  socks_per_player * sock_cost + tshirts_per_player * (sock_cost + tshirt_additional_cost)

/-- The number of players in the Greenfield Basketball League -/
def num_players : ℕ := total_expenditure / cost_per_player

theorem greenfield_basketball_league_players : num_players = 108 := by
  sorry

end greenfield_basketball_league_players_l2600_260003


namespace min_omega_value_l2600_260083

theorem min_omega_value (ω : ℝ) (f g : ℝ → ℝ) : 
  (ω > 0) →
  (∀ x, f x = Real.sin (ω * x)) →
  (∀ x, g x = f (x - π / 12)) →
  (∃ k : ℤ, ω * π / 3 - ω * π / 12 = k * π + π / 2) →
  ω ≥ 2 :=
sorry

end min_omega_value_l2600_260083


namespace quadratic_completion_l2600_260051

theorem quadratic_completion (p : ℝ) (n : ℝ) : 
  (∀ x, x^2 + p*x + 1/4 = (x+n)^2 - 1/16) → 
  p < 0 → 
  p = -Real.sqrt 5 / 2 := by
  sorry

end quadratic_completion_l2600_260051


namespace largest_integer_less_than_95_with_remainder_5_mod_7_l2600_260088

theorem largest_integer_less_than_95_with_remainder_5_mod_7 :
  ∃ n : ℤ, n < 95 ∧ n % 7 = 5 ∧ ∀ m : ℤ, m < 95 ∧ m % 7 = 5 → m ≤ n :=
by
  -- The proof goes here
  sorry

end largest_integer_less_than_95_with_remainder_5_mod_7_l2600_260088


namespace no_real_roots_for_nonzero_k_l2600_260086

theorem no_real_roots_for_nonzero_k (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, x^2 + 2*k*x + 3*k^2 ≠ 0 := by
sorry

end no_real_roots_for_nonzero_k_l2600_260086


namespace box_third_side_length_l2600_260043

/-- Proves that the third side of a rectangular box is 9 cm, given specific conditions. -/
theorem box_third_side_length (num_cubes : ℕ) (cube_volume : ℝ) (side1 side2 : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  side1 = 8 →
  side2 = 9 →
  (num_cubes : ℝ) * cube_volume = side1 * side2 * 9 :=
by sorry

end box_third_side_length_l2600_260043


namespace expression_simplification_l2600_260072

theorem expression_simplification (a x : ℝ) (h : a^2 + x^3 > 0) :
  (Real.sqrt (a^2 + x^3) - (x^3 - a^2) / Real.sqrt (a^2 + x^3)) / (a^2 + x^3) =
  2 * a^2 / (a^2 + x^3)^(3/2) := by
  sorry

end expression_simplification_l2600_260072


namespace product_of_numbers_l2600_260080

theorem product_of_numbers (x y : ℝ) 
  (h1 : x - y = 15) 
  (h2 : x^2 + y^2 = 578) : 
  x * y = (931 - 15 * Real.sqrt 931) / 4 := by
  sorry

end product_of_numbers_l2600_260080


namespace smallest_marble_set_marble_set_existence_l2600_260033

theorem smallest_marble_set (n : ℕ) : n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n → n ≥ 210 := by
  sorry

theorem marble_set_existence : ∃ n : ℕ, n > 0 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 210 := by
  sorry

end smallest_marble_set_marble_set_existence_l2600_260033


namespace list_mean_mode_relation_l2600_260001

theorem list_mean_mode_relation (x : ℕ) (h1 : x ≤ 200) :
  let L := [30, 60, 70, 150, x, x]
  (L.sum / L.length : ℚ) = 2 * x →
  x = 31 := by
sorry

end list_mean_mode_relation_l2600_260001


namespace log_equation_solution_l2600_260007

theorem log_equation_solution :
  ∀ x : ℝ, (x + 5 > 0) ∧ (2*x - 1 > 0) ∧ (3*x^2 - 11*x + 5 > 0) →
  (Real.log (x + 5) + Real.log (2*x - 1) = Real.log (3*x^2 - 11*x + 5)) ↔
  (x = 10 + 3 * Real.sqrt 10 ∨ x = 10 - 3 * Real.sqrt 10) :=
by sorry

end log_equation_solution_l2600_260007


namespace product_b_sample_size_l2600_260005

/-- Represents the number of items drawn from a specific group in stratified sampling -/
def stratifiedSampleSize (totalItems : ℕ) (sampleSize : ℕ) (groupRatio : ℕ) (totalRatio : ℕ) : ℕ :=
  (sampleSize * groupRatio) / totalRatio

/-- Proves that the number of items drawn from product B in the given stratified sampling scenario is 10 -/
theorem product_b_sample_size :
  let totalItems : ℕ := 1200
  let sampleSize : ℕ := 60
  let ratioA : ℕ := 1
  let ratioB : ℕ := 2
  let ratioC : ℕ := 4
  let ratioD : ℕ := 5
  let totalRatio : ℕ := ratioA + ratioB + ratioC + ratioD
  stratifiedSampleSize totalItems sampleSize ratioB totalRatio = 10 := by
  sorry


end product_b_sample_size_l2600_260005


namespace expression_simplification_l2600_260068

theorem expression_simplification (x y : ℝ) 
  (hx : x = (Real.sqrt 5 + 1) / 2) 
  (hy : y = (Real.sqrt 5 - 1) / 2) : 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 := by
  sorry

end expression_simplification_l2600_260068


namespace geometric_mean_minimum_l2600_260054

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (4^a * 2^b)) :
  (2/a + 1/b) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    Real.sqrt 2 = Real.sqrt (4^a₀ * 2^b₀) ∧ 2/a₀ + 1/b₀ = 9 := by
  sorry

end geometric_mean_minimum_l2600_260054


namespace liza_final_balance_l2600_260066

/-- Calculates the final balance in Liza's account after a series of transactions --/
def final_balance (initial_balance rent paycheck electricity internet phone : ℤ) : ℤ :=
  initial_balance - rent + paycheck - electricity - internet - phone

/-- Theorem stating that Liza's final account balance is 1563 --/
theorem liza_final_balance :
  final_balance 800 450 1500 117 100 70 = 1563 := by sorry

end liza_final_balance_l2600_260066


namespace ellipse_foci_l2600_260034

/-- The equation of an ellipse -/
def is_ellipse (x y : ℝ) : Prop := y^2 / 3 + x^2 / 2 = 1

/-- The coordinates of a point -/
def Point := ℝ × ℝ

/-- The foci of an ellipse -/
def are_foci (p1 p2 : Point) : Prop :=
  p1 = (0, -1) ∧ p2 = (0, 1)

/-- Theorem: The foci of the given ellipse are (0, -1) and (0, 1) -/
theorem ellipse_foci :
  ∃ (p1 p2 : Point), (∀ x y : ℝ, is_ellipse x y → are_foci p1 p2) :=
sorry

end ellipse_foci_l2600_260034


namespace max_pangs_proof_l2600_260055

/-- The maximum number of pangs that can be purchased given the constraints -/
def max_pangs : ℕ := 9

/-- The price of a pin in dollars -/
def pin_price : ℕ := 3

/-- The price of a pon in dollars -/
def pon_price : ℕ := 4

/-- The price of a pang in dollars -/
def pang_price : ℕ := 9

/-- The total budget in dollars -/
def total_budget : ℕ := 100

/-- The minimum number of pins that must be purchased -/
def min_pins : ℕ := 2

/-- The minimum number of pons that must be purchased -/
def min_pons : ℕ := 3

theorem max_pangs_proof :
  ∃ (pins pons : ℕ),
    pins ≥ min_pins ∧
    pons ≥ min_pons ∧
    pin_price * pins + pon_price * pons + pang_price * max_pangs = total_budget ∧
    ∀ (pangs : ℕ), pangs > max_pangs →
      ∀ (p q : ℕ),
        p ≥ min_pins →
        q ≥ min_pons →
        pin_price * p + pon_price * q + pang_price * pangs ≠ total_budget :=
by sorry

end max_pangs_proof_l2600_260055


namespace arithmetic_sequence_sum_2017_l2600_260027

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * a 1 + (n * (n - 1) : ℤ) * (a 2 - a 1) / 2

theorem arithmetic_sequence_sum_2017 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_first_term : a 1 = -2015)
  (h_sum_condition : sum_arithmetic_sequence a 6 - 2 * sum_arithmetic_sequence a 3 = 18) :
  sum_arithmetic_sequence a 2017 = 2017 :=
sorry

end arithmetic_sequence_sum_2017_l2600_260027


namespace geometric_sequence_ratio_l2600_260095

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  condition1 : a 3 + a 5 = 8
  condition2 : a 1 * a 5 = 4

/-- The ratio of the 13th term to the 9th term is 9 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : seq.a 13 / seq.a 9 = 9 := by
  sorry

end geometric_sequence_ratio_l2600_260095


namespace six_hours_prep_score_l2600_260018

/-- Represents the relationship between study time and test score -/
structure TestPreparation where
  actualHours : ℝ
  score : ℝ

/-- Calculates effective hours from actual hours -/
def effectiveHours (ah : ℝ) : ℝ := 0.8 * ah

/-- Theorem: Given the conditions, 6 actual hours of preparation results in a score of 96 points -/
theorem six_hours_prep_score :
  ∀ (test1 test2 : TestPreparation),
  test1.actualHours = 5 ∧
  test1.score = 80 ∧
  test2.actualHours = 6 →
  test2.score = 96 := by sorry

end six_hours_prep_score_l2600_260018


namespace system_solution_l2600_260065

theorem system_solution (x y : ℝ) :
  (y = x + 1) ∧ (y = -x + 2) ∧ (x = 1/2) ∧ (y = 3/2) →
  (y - x - 1 = 0) ∧ (y + x - 2 = 0) ∧ (x = 1/2) ∧ (y = 3/2) := by
  sorry

end system_solution_l2600_260065


namespace arrangements_without_adjacent_l2600_260042

def total_people : ℕ := 5

theorem arrangements_without_adjacent (A B : ℕ) (h1 : A ≤ total_people) (h2 : B ≤ total_people) (h3 : A ≠ B) :
  (Nat.factorial total_people) - (2 * Nat.factorial (total_people - 1)) = 72 :=
sorry

end arrangements_without_adjacent_l2600_260042


namespace equation_real_root_l2600_260084

theorem equation_real_root (x m : ℝ) (i : ℂ) : 
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) → m = 1/12 :=
by sorry

end equation_real_root_l2600_260084


namespace three_integers_product_2700_sum_56_l2600_260056

theorem three_integers_product_2700_sum_56 :
  ∃ (a b c : ℕ),
    a > 1 ∧ b > 1 ∧ c > 1 ∧
    Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1 ∧
    a * b * c = 2700 ∧
    a + b + c = 56 := by
  sorry

end three_integers_product_2700_sum_56_l2600_260056


namespace lcm_14_21_l2600_260013

theorem lcm_14_21 : Nat.lcm 14 21 = 42 := by
  sorry

end lcm_14_21_l2600_260013


namespace count_ballpoint_pens_l2600_260077

/-- The total number of school supplies -/
def total_supplies : ℕ := 60

/-- The number of pencils -/
def pencils : ℕ := 5

/-- The number of notebooks -/
def notebooks : ℕ := 10

/-- The number of erasers -/
def erasers : ℕ := 32

/-- The number of ballpoint pens -/
def ballpoint_pens : ℕ := total_supplies - (pencils + notebooks + erasers)

theorem count_ballpoint_pens : ballpoint_pens = 13 := by
  sorry

end count_ballpoint_pens_l2600_260077


namespace intersection_complement_equals_set_l2600_260089

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x | x^2 - 2*x ≥ 3}

theorem intersection_complement_equals_set : M ∩ (Set.univ \ N) = {1, 2} := by
  sorry

end intersection_complement_equals_set_l2600_260089


namespace p_plus_q_equals_21_over_2_l2600_260058

theorem p_plus_q_equals_21_over_2 (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 81 = 0)
  (hq : 9*q^3 - 81*q^2 - 243*q + 3645 = 0) : 
  p + q = 21/2 := by
  sorry

end p_plus_q_equals_21_over_2_l2600_260058


namespace sum_of_squares_equation_l2600_260076

theorem sum_of_squares_equation (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (eq1 : a^2 + a*b + b^2 = 1)
  (eq2 : b^2 + b*c + c^2 = 3)
  (eq3 : c^2 + c*a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := by
  sorry

end sum_of_squares_equation_l2600_260076


namespace prime_exponent_assignment_l2600_260094

theorem prime_exponent_assignment (k : ℕ) (p : Fin k → ℕ) 
  (h_prime : ∀ i, Prime (p i)) 
  (h_distinct : ∀ i j, i ≠ j → p i ≠ p j) :
  (Finset.univ : Finset (Fin k → Fin k)).card = k ^ k :=
sorry

end prime_exponent_assignment_l2600_260094


namespace gcd_with_30_is_6_l2600_260023

theorem gcd_with_30_is_6 : ∃ n : ℕ, 70 < n ∧ n < 80 ∧ Nat.gcd n 30 = 6 := by
  sorry

end gcd_with_30_is_6_l2600_260023


namespace geometric_sequence_third_term_l2600_260029

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a1 : a 1 = 1)
  (h_a5 : a 5 = 4) :
  a 3 = 2 :=
sorry

end geometric_sequence_third_term_l2600_260029


namespace certain_value_proof_l2600_260046

theorem certain_value_proof (n : ℤ) (x : ℤ) 
  (h1 : 101 * n^2 ≤ x)
  (h2 : ∀ m : ℤ, 101 * m^2 ≤ x → m ≤ 7) :
  x = 4979 := by
  sorry

end certain_value_proof_l2600_260046


namespace integral_second_derivative_car_acceleration_l2600_260038

-- Part 1
theorem integral_second_derivative {f : ℝ → ℝ} {a b : ℝ} (h₁ : Continuous (deriv (deriv f))) 
  (h₂ : deriv f a = 0) (h₃ : deriv f b = 0) (h₄ : a < b) :
  f b - f a = ∫ x in a..b, ((a + b) / 2 - x) * deriv (deriv f) x := by sorry

-- Part 2
theorem car_acceleration {f : ℝ → ℝ} {L T : ℝ} (h₁ : f 0 = 0) (h₂ : f T = L) 
  (h₃ : deriv f 0 = 0) (h₄ : deriv f T = 0) (h₅ : T > 0) (h₆ : L > 0) :
  ∃ t : ℝ, t ∈ Set.Icc 0 T ∧ |deriv (deriv f) t| ≥ 4 * L / T^2 := by sorry

end integral_second_derivative_car_acceleration_l2600_260038


namespace barge_unloading_time_l2600_260041

/-- Represents the unloading scenario of a barge with different crane configurations -/
structure BargeUnloading where
  /-- Time (in hours) for one crane of greater capacity to unload the barge alone -/
  x : ℝ
  /-- Time (in hours) for one crane of lesser capacity to unload the barge alone -/
  y : ℝ
  /-- Time (in hours) for one crane of greater capacity and one of lesser capacity to unload together -/
  z : ℝ

/-- The main theorem about the barge unloading scenario -/
theorem barge_unloading_time (b : BargeUnloading) : b.z = 14.4 :=
  sorry

end barge_unloading_time_l2600_260041


namespace subset_implies_a_range_l2600_260026

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + a*x + a + 3 = 0}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) (h : B a ⊆ A) : -2 ≤ a ∧ a < 6 := by
  sorry

end subset_implies_a_range_l2600_260026


namespace sue_made_22_buttons_l2600_260014

def mari_buttons : ℕ := 8

def kendra_buttons (m : ℕ) : ℕ := 5 * m + 4

def sue_buttons (k : ℕ) : ℕ := k / 2

theorem sue_made_22_buttons : 
  sue_buttons (kendra_buttons mari_buttons) = 22 := by
  sorry

end sue_made_22_buttons_l2600_260014


namespace brendas_mother_cookies_l2600_260030

/-- The number of people Brenda's mother made cookies for -/
def num_people (total_cookies : ℕ) (cookies_per_person : ℕ) : ℕ :=
  total_cookies / cookies_per_person

theorem brendas_mother_cookies : num_people 35 7 = 5 := by
  sorry

end brendas_mother_cookies_l2600_260030


namespace simplify_and_evaluate_l2600_260064

theorem simplify_and_evaluate :
  (∀ x : ℝ, x = -1 → (-x^2 + 5*x) - (x - 3) - 4*x = 2) ∧
  (∀ m n : ℝ, m = -1/2 ∧ n = 1/3 → 5*(3*m^2*n - m*n^2) - (m*n^2 + 3*m^2*n) = 4/3) :=
by sorry

end simplify_and_evaluate_l2600_260064


namespace teamA_win_probability_l2600_260069

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  numTeams : Nat
  noTies : Bool
  equalWinChance : Bool
  teamAWonFirst : Bool

/-- Calculates the probability that Team A finishes with more points than Team B -/
def probabilityTeamAWins (tournament : SoccerTournament) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem teamA_win_probability 
  (tournament : SoccerTournament) 
  (h1 : tournament.numTeams = 9)
  (h2 : tournament.noTies = true)
  (h3 : tournament.equalWinChance = true)
  (h4 : tournament.teamAWonFirst = true) :
  probabilityTeamAWins tournament = 9714 / 8192 :=
sorry

end teamA_win_probability_l2600_260069


namespace digit_145_of_49_div_686_l2600_260045

/-- The decimal expansion of 49/686 has a period of 6 -/
def period : ℕ := 6

/-- The repeating sequence in the decimal expansion of 49/686 -/
def repeating_sequence : Fin 6 → ℕ
| 0 => 0
| 1 => 7
| 2 => 1
| 3 => 4
| 4 => 2
| 5 => 8

/-- The 145th digit after the decimal point in the decimal expansion of 49/686 is 8 -/
theorem digit_145_of_49_div_686 : 
  repeating_sequence ((145 - 1) % period) = 8 := by sorry

end digit_145_of_49_div_686_l2600_260045


namespace james_money_l2600_260020

/-- Calculates the total money James has after finding additional bills -/
def total_money (bills_found : ℕ) (bill_value : ℕ) (existing_money : ℕ) : ℕ :=
  bills_found * bill_value + existing_money

/-- Proves that James has $135 after finding 3 $20 bills when he already had $75 -/
theorem james_money :
  total_money 3 20 75 = 135 := by
  sorry

end james_money_l2600_260020


namespace polynomial_root_sum_l2600_260071

theorem polynomial_root_sum (A B C D : ℤ) : 
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℂ, z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36 = 0 ↔ 
      z = r₁ ∨ z = r₂ ∨ z = r₃ ∨ z = r₄ ∨ z = r₅ ∨ z = r₆) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12) →
  B = -76 := by
sorry

end polynomial_root_sum_l2600_260071


namespace largest_perfect_square_factor_of_3465_is_9_l2600_260096

/-- The largest perfect square factor of 3465 -/
def largest_perfect_square_factor_of_3465 : ℕ := 9

/-- Theorem stating that the largest perfect square factor of 3465 is 9 -/
theorem largest_perfect_square_factor_of_3465_is_9 :
  ∀ n : ℕ, n^2 ∣ 3465 → n ≤ 3 :=
sorry

end largest_perfect_square_factor_of_3465_is_9_l2600_260096


namespace inscribed_sphere_sum_l2600_260031

/-- A sphere inscribed in a right cone with base radius 15 cm and height 30 cm -/
structure InscribedSphere where
  base_radius : ℝ
  height : ℝ
  sphere_radius : ℝ
  b : ℝ
  d : ℝ
  base_radius_eq : base_radius = 15
  height_eq : height = 30
  sphere_radius_eq : sphere_radius = b * (Real.sqrt d - 1)

/-- Theorem stating that b + d = 11.75 for the given inscribed sphere -/
theorem inscribed_sphere_sum (s : InscribedSphere) : s.b + s.d = 11.75 := by
  sorry

#check inscribed_sphere_sum

end inscribed_sphere_sum_l2600_260031


namespace tourism_max_value_l2600_260059

noncomputable def f (x : ℝ) : ℝ := (51/50) * x - 0.01 * x^2 - Real.log x + Real.log 10

theorem tourism_max_value (x : ℝ) (h1 : 6 < x) (h2 : x ≤ 12) :
  ∃ (y : ℝ), y = f 12 ∧ ∀ z ∈ Set.Ioo 6 12, f z ≤ y := by
  sorry

end tourism_max_value_l2600_260059


namespace power_product_evaluation_l2600_260070

theorem power_product_evaluation : 
  let a : ℕ := 2
  a^3 * a^4 = 128 := by sorry

end power_product_evaluation_l2600_260070


namespace closest_integer_to_cube_root_150_l2600_260036

theorem closest_integer_to_cube_root_150 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (150 : ℝ)^(1/3)| ≤ |m - (150 : ℝ)^(1/3)| ∧ n = 5 :=
sorry

end closest_integer_to_cube_root_150_l2600_260036


namespace arithmetic_sequence_common_difference_l2600_260085

/-- Given an arithmetic sequence {a_n} with a_3 = 5 and a_15 = 41, 
    prove that the common difference d is equal to 3. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a3 : a 3 = 5) 
  (h_a15 : a 15 = 41) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end arithmetic_sequence_common_difference_l2600_260085


namespace periodic_sequence_quadratic_root_l2600_260052

def is_periodic (x : ℕ → ℝ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, x (n + p) = x n

def sequence_property (x : ℕ → ℝ) : Prop :=
  x 0 > 1 ∧ ∀ n : ℕ, x (n + 1) = 1 / (x n - ⌊x n⌋)

def is_quadratic_root (r : ℝ) : Prop :=
  ∃ a b c : ℤ, a ≠ 0 ∧ a * r^2 + b * r + c = 0

theorem periodic_sequence_quadratic_root (x : ℕ → ℝ) :
  is_periodic x → sequence_property x → is_quadratic_root (x 0) := by
  sorry

end periodic_sequence_quadratic_root_l2600_260052


namespace cyclic_fraction_inequality_l2600_260067

theorem cyclic_fraction_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 2*y) / (z + 2*x + 3*y) + (y + 2*z) / (x + 2*y + 3*z) + (z + 2*x) / (y + 2*z + 3*x) ≤ 3/2 := by
sorry

end cyclic_fraction_inequality_l2600_260067


namespace meeting_attendees_l2600_260053

theorem meeting_attendees (total_handshakes : ℕ) (h : total_handshakes = 91) :
  ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = total_handshakes ∧ n = 14 := by
  sorry

end meeting_attendees_l2600_260053


namespace initial_guinea_fowls_eq_80_l2600_260017

/-- Represents the initial state and daily losses of birds in a poultry farm --/
structure PoultryFarm :=
  (initial_chickens : ℕ)
  (initial_turkeys : ℕ)
  (daily_chicken_loss : ℕ)
  (daily_turkey_loss : ℕ)
  (daily_guinea_fowl_loss : ℕ)
  (disease_duration : ℕ)
  (total_birds_after_disease : ℕ)

/-- Calculates the initial number of guinea fowls in the farm --/
def initial_guinea_fowls (farm : PoultryFarm) : ℕ :=
  let remaining_chickens := farm.initial_chickens - farm.daily_chicken_loss * farm.disease_duration
  let remaining_turkeys := farm.initial_turkeys - farm.daily_turkey_loss * farm.disease_duration
  let remaining_guinea_fowls := farm.total_birds_after_disease - remaining_chickens - remaining_turkeys
  remaining_guinea_fowls + farm.daily_guinea_fowl_loss * farm.disease_duration

/-- Theorem stating that the initial number of guinea fowls is 80 --/
theorem initial_guinea_fowls_eq_80 (farm : PoultryFarm) 
  (h1 : farm.initial_chickens = 300)
  (h2 : farm.initial_turkeys = 200)
  (h3 : farm.daily_chicken_loss = 20)
  (h4 : farm.daily_turkey_loss = 8)
  (h5 : farm.daily_guinea_fowl_loss = 5)
  (h6 : farm.disease_duration = 7)
  (h7 : farm.total_birds_after_disease = 349) :
  initial_guinea_fowls farm = 80 := by
  sorry

#eval initial_guinea_fowls {
  initial_chickens := 300,
  initial_turkeys := 200,
  daily_chicken_loss := 20,
  daily_turkey_loss := 8,
  daily_guinea_fowl_loss := 5,
  disease_duration := 7,
  total_birds_after_disease := 349
}

end initial_guinea_fowls_eq_80_l2600_260017


namespace angle_value_proof_l2600_260024

theorem angle_value_proof (ABC DBC ABD : ℝ) (y : ℝ) : 
  ABC = 90 →
  ABD = 3 * y →
  DBC = 2 * y →
  ABD + DBC = 90 →
  y = 18 := by sorry

end angle_value_proof_l2600_260024


namespace max_d_value_in_multiple_of_13_l2600_260021

theorem max_d_value_in_multiple_of_13 :
  let is_valid : (ℕ → ℕ → Bool) :=
    fun d e => (520000 + 10000 * d + 550 + 10 * e) % 13 = 0 ∧ 
               d < 10 ∧ e < 10
  ∃ d e, is_valid d e ∧ d = 6 ∧ ∀ d' e', is_valid d' e' → d' ≤ d :=
by sorry

end max_d_value_in_multiple_of_13_l2600_260021


namespace sin_300_deg_l2600_260090

theorem sin_300_deg : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_deg_l2600_260090


namespace range_of_x_for_meaningful_sqrt_l2600_260087

theorem range_of_x_for_meaningful_sqrt (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3*x - 2) → x ≥ 2/3 := by
sorry

end range_of_x_for_meaningful_sqrt_l2600_260087


namespace heptagon_triangulation_l2600_260002

-- Define a type for polygons
structure Polygon where
  sides : Nat
  isRegular : Bool

-- Define a triangulation
structure Triangulation where
  polygon : Polygon
  numTriangles : Nat
  usesDiagonals : Bool
  verticesFromPolygon : Bool

-- Define a function to count unique triangulations
def countUniqueTriangulations (p : Polygon) (t : Triangulation) : Nat :=
  sorry

-- Theorem statement
theorem heptagon_triangulation :
  let heptagon : Polygon := { sides := 7, isRegular := true }
  let triangulation : Triangulation := {
    polygon := heptagon,
    numTriangles := 5,
    usesDiagonals := true,
    verticesFromPolygon := true
  }
  countUniqueTriangulations heptagon triangulation = 4 := by
  sorry

end heptagon_triangulation_l2600_260002


namespace participation_related_to_city_probability_one_from_each_city_l2600_260093

-- Define the contingency table
def contingency_table : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![60, 40],
    ![30, 70]]

-- Define the K^2 formula
def K_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for 99.9% certainty
def critical_value : ℚ := 10828 / 1000

-- Theorem for part 1
theorem participation_related_to_city :
  let a := contingency_table 0 0
  let b := contingency_table 0 1
  let c := contingency_table 1 0
  let d := contingency_table 1 1
  K_squared a b c d > critical_value :=
sorry

-- Define the number of people from each city
def city_A_count : ℕ := 4
def city_B_count : ℕ := 2
def total_count : ℕ := city_A_count + city_B_count

-- Theorem for part 2
theorem probability_one_from_each_city :
  (Nat.choose city_A_count 1 * Nat.choose city_B_count 1 : ℚ) / Nat.choose total_count 2 = 8 / 15 :=
sorry

end participation_related_to_city_probability_one_from_each_city_l2600_260093


namespace conic_is_hyperbola_l2600_260078

/-- Defines the equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 16*y^2 - 8*x + 16*y + 32 = 0

/-- Theorem stating that the conic equation represents a hyperbola -/
theorem conic_is_hyperbola :
  ∃ (h k a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ (x y : ℝ), conic_equation x y ↔ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1) :=
sorry

end conic_is_hyperbola_l2600_260078


namespace fourth_number_is_eight_l2600_260009

/-- Given four numbers with an arithmetic mean of 20, where three of the numbers are 12, 24, and 36,
    and the fourth number is the square of another number, prove that the fourth number is 8. -/
theorem fourth_number_is_eight (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 24 →
  c = 36 →
  ∃ x, d = x^2 →
  d = 8 := by
  sorry

end fourth_number_is_eight_l2600_260009


namespace candies_in_box_l2600_260074

def initial_candies : ℕ := 88
def diana_takes : ℕ := 6
def john_adds : ℕ := 12
def sara_takes : ℕ := 20

theorem candies_in_box : 
  initial_candies - diana_takes + john_adds - sara_takes = 74 :=
by sorry

end candies_in_box_l2600_260074


namespace max_triangle_area_l2600_260011

/-- Ellipse E with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : 1/a^2 + 3/(4*b^2) = 1

/-- Line l intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x y : ℝ), x^2/E.a^2 + y^2/E.b^2 = 1 ∧ y = k*x + m

/-- Perpendicular bisector condition -/
def perp_bisector_condition (E : Ellipse) (l : IntersectingLine E) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2/E.a^2 + y₁^2/E.b^2 = 1 ∧
    x₂^2/E.a^2 + y₂^2/E.b^2 = 1 ∧
    y₁ = l.k*x₁ + l.m ∧
    y₂ = l.k*x₂ + l.m ∧
    (x₁ + x₂)/2 = 0 ∧
    (y₁ + y₂)/2 = 1/2

/-- Area of triangle AOB -/
def triangle_area (E : Ellipse) (l : IntersectingLine E) : ℝ :=
  sorry

/-- Theorem: Maximum area of triangle AOB is 1 -/
theorem max_triangle_area (E : Ellipse) :
  ∃ (l : IntersectingLine E),
    perp_bisector_condition E l ∧
    triangle_area E l = 1 ∧
    ∀ (l' : IntersectingLine E),
      perp_bisector_condition E l' →
      triangle_area E l' ≤ 1 :=
sorry

end max_triangle_area_l2600_260011


namespace parabola_directrix_l2600_260006

/-- The directrix of the parabola y² = 4x is the line x = -1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y^2 = 4*x → (∃ (a : ℝ), a = -1 ∧ x = a) := by
  sorry

end parabola_directrix_l2600_260006


namespace sin_90_degrees_l2600_260082

theorem sin_90_degrees : 
  Real.sin (90 * π / 180) = 1 := by
  sorry

end sin_90_degrees_l2600_260082


namespace polynomial_division_quotient_l2600_260063

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 - 8 * X^3 - 12 * X^2 + 5 * X - 9
  let divisor : Polynomial ℚ := 3 * X^2 - 2
  let quotient := dividend / divisor
  (quotient.coeff 2 = 10/3) ∧ (quotient.coeff 1 = -8/3) := by sorry

end polynomial_division_quotient_l2600_260063


namespace exists_valid_division_l2600_260050

/-- A grid-based figure --/
structure GridFigure where
  cells : ℕ

/-- Represents a division of a grid figure --/
structure Division where
  removed : ℕ
  part1 : ℕ
  part2 : ℕ

/-- Checks if a division is valid for a given grid figure --/
def is_valid_division (g : GridFigure) (d : Division) : Prop :=
  d.removed = 1 ∧ 
  d.part1 = d.part2 ∧
  d.part1 + d.part2 + d.removed = g.cells

/-- Theorem stating that a valid division exists for any grid figure --/
theorem exists_valid_division (g : GridFigure) : 
  ∃ (d : Division), is_valid_division g d :=
sorry

end exists_valid_division_l2600_260050


namespace three_digit_number_subtraction_l2600_260061

theorem three_digit_number_subtraction (c : ℕ) 
  (h1 : c < 10) 
  (h2 : 2 * c < 10) 
  (h3 : c + 3 < 10) : 
  (100 * (c + 3) + 10 * (2 * c) + c) - (100 * c + 10 * (2 * c) + (c + 3)) ≡ 7 [ZMOD 10] := by
  sorry

end three_digit_number_subtraction_l2600_260061


namespace trigonometric_identity_l2600_260048

theorem trigonometric_identity : 
  Real.cos (6 * π / 180) * Real.cos (36 * π / 180) + 
  Real.sin (6 * π / 180) * Real.cos (54 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end trigonometric_identity_l2600_260048


namespace bus_speed_with_stoppages_l2600_260073

/-- Given a bus that travels at 32 km/hr excluding stoppages and stops for 30 minutes per hour,
    the speed of the bus including stoppages is 16 km/hr. -/
theorem bus_speed_with_stoppages 
  (speed_without_stoppages : ℝ) 
  (stop_time : ℝ) 
  (h1 : speed_without_stoppages = 32)
  (h2 : stop_time = 0.5) : 
  speed_without_stoppages * (1 - stop_time) = 16 := by
  sorry

#check bus_speed_with_stoppages

end bus_speed_with_stoppages_l2600_260073


namespace lottery_probability_l2600_260040

theorem lottery_probability : 
  let mega_balls : ℕ := 30
  let winner_balls : ℕ := 50
  let picked_winner_balls : ℕ := 5
  let mega_prob : ℚ := 1 / mega_balls
  let winner_prob : ℚ := 1 / (winner_balls.choose picked_winner_balls)
  mega_prob * winner_prob = 1 / 63562800 :=
by sorry

end lottery_probability_l2600_260040


namespace right_triangle_proof_l2600_260097

open Real

theorem right_triangle_proof (A B C : ℝ) (a b c : ℝ) (h1 : b ≠ 1) 
  (h2 : C / A = 2) (h3 : sin B / sin A = 2) (h4 : A + B + C = π) :
  A = π / 6 ∧ B = π / 2 ∧ C = π / 3 := by
  sorry

end right_triangle_proof_l2600_260097


namespace p_necessary_not_sufficient_for_q_l2600_260057

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x^2 + x - 2 < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ x^2 + x - 2 ≥ 0) := by
  sorry

end p_necessary_not_sufficient_for_q_l2600_260057


namespace app_cost_calculation_l2600_260010

/-- Calculates the total cost of an app with online access -/
def total_cost (app_price : ℕ) (monthly_fee : ℕ) (months : ℕ) : ℕ :=
  app_price + monthly_fee * months

/-- Theorem: The total cost for an app with an initial price of $5 and 
    a monthly online access fee of $8, used for 2 months, is $21 -/
theorem app_cost_calculation : total_cost 5 8 2 = 21 := by
  sorry

end app_cost_calculation_l2600_260010


namespace tea_cups_filled_l2600_260032

theorem tea_cups_filled (total_tea : ℕ) (tea_per_cup : ℕ) (h1 : total_tea = 1050) (h2 : tea_per_cup = 65) :
  (total_tea / tea_per_cup : ℕ) = 16 := by
  sorry

end tea_cups_filled_l2600_260032


namespace min_value_theorem_l2600_260091

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x + 3 / (x + 1) ≥ 2 * Real.sqrt 3 - 1 ∧
  (x + 3 / (x + 1) = 2 * Real.sqrt 3 - 1 ↔ x = Real.sqrt 3 - 1) :=
by sorry

end min_value_theorem_l2600_260091


namespace unique_solution_l2600_260060

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (g x * g y - g (x * y)) / 5 = x + y + 2

/-- The main theorem stating that the function g(x) = x + 3 is the unique solution -/
theorem unique_solution :
  ∀ g : ℝ → ℝ, SatisfiesFunctionalEquation g → ∀ x : ℝ, g x = x + 3 := by
  sorry

end unique_solution_l2600_260060


namespace parabola_transformation_l2600_260025

/-- A parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shift a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola where
  f x := p.f (x - h) + v

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola where
  f x := 2 * x^2

/-- The transformed parabola -/
def transformed_parabola : Parabola :=
  shift (shift original_parabola 3 0) 0 (-4)

theorem parabola_transformation :
  ∀ x, transformed_parabola.f x = 2 * (x + 3)^2 - 4 := by sorry

end parabola_transformation_l2600_260025
