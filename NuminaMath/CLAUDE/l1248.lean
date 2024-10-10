import Mathlib

namespace parallel_vectors_sum_magnitude_l1248_124823

/-- Two parallel planar vectors have a specific sum magnitude -/
theorem parallel_vectors_sum_magnitude :
  ∀ (x : ℝ),
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -3]
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →
  ‖(a + b)‖ = 2 * Real.sqrt 5 := by
  sorry

end parallel_vectors_sum_magnitude_l1248_124823


namespace coefficient_x_cubed_l1248_124892

def p (x : ℝ) : ℝ := 3*x^3 + 2*x^2 + 5*x + 4
def q (x : ℝ) : ℝ := 7*x^3 + 5*x^2 + 6*x + 7

theorem coefficient_x_cubed (x : ℝ) : 
  ∃ (a b c d : ℝ), p x * q x = 38*x^3 + a*x^4 + b*x^2 + c*x + d :=
sorry

end coefficient_x_cubed_l1248_124892


namespace quadratic_inequality_no_solution_l1248_124868

theorem quadratic_inequality_no_solution :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 4 ≥ 0 :=
by sorry

end quadratic_inequality_no_solution_l1248_124868


namespace sum_of_numbers_l1248_124828

theorem sum_of_numbers (a b : ℕ) : 
  100 ≤ a ∧ a ≤ 999 →   -- a is a three-digit number
  10 ≤ b ∧ b ≤ 99 →     -- b is a two-digit number
  a - b = 989 →         -- their difference is 989
  a + b = 1009 :=       -- prove their sum is 1009
by
  sorry

end sum_of_numbers_l1248_124828


namespace equivalent_root_equations_l1248_124882

theorem equivalent_root_equations (a : ℝ) :
  ∀ x : ℝ, x = a + Real.sqrt (a + Real.sqrt x) ↔ x = a + Real.sqrt x :=
by sorry

end equivalent_root_equations_l1248_124882


namespace complex_calculation_l1248_124865

theorem complex_calculation (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  2 * (A - M + S - P) = 10 - 4*I :=
by sorry

end complex_calculation_l1248_124865


namespace circle_line_distance_range_l1248_124832

theorem circle_line_distance_range (a : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  let line := {(x, y) : ℝ × ℝ | x + y = a}
  let distance_to_line (p : ℝ × ℝ) := |p.1 + p.2 - a| / Real.sqrt 2
  (∃ p1 p2 : ℝ × ℝ, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2 ∧ 
    distance_to_line p1 = 1 ∧ distance_to_line p2 = 1) →
  a ∈ Set.Ioo (-3 * Real.sqrt 2) (3 * Real.sqrt 2) :=
by sorry

end circle_line_distance_range_l1248_124832


namespace total_tabs_is_322_l1248_124844

def browser1_windows : ℕ := 4
def browser1_tabs_per_window : ℕ := 10

def browser2_windows : ℕ := 5
def browser2_tabs_per_window : ℕ := 12

def browser3_windows : ℕ := 6
def browser3_tabs_per_window : ℕ := 15

def browser4_windows : ℕ := browser1_windows
def browser4_tabs_per_window : ℕ := browser1_tabs_per_window + 5

def browser5_windows : ℕ := browser2_windows
def browser5_tabs_per_window : ℕ := browser2_tabs_per_window - 2

def browser6_windows : ℕ := 3
def browser6_tabs_per_window : ℕ := browser3_tabs_per_window / 2

def total_tabs : ℕ := 
  browser1_windows * browser1_tabs_per_window +
  browser2_windows * browser2_tabs_per_window +
  browser3_windows * browser3_tabs_per_window +
  browser4_windows * browser4_tabs_per_window +
  browser5_windows * browser5_tabs_per_window +
  browser6_windows * browser6_tabs_per_window

theorem total_tabs_is_322 : total_tabs = 322 := by
  sorry

end total_tabs_is_322_l1248_124844


namespace equilateral_triangle_side_length_l1248_124885

/-- An equilateral triangle with perimeter 69 cm has sides of length 23 cm -/
theorem equilateral_triangle_side_length (triangle : Set ℝ) (perimeter : ℝ) :
  perimeter = 69 →
  ∃ (side_length : ℝ), side_length * 3 = perimeter ∧ side_length = 23 := by
  sorry

end equilateral_triangle_side_length_l1248_124885


namespace sum_squares_35_consecutive_divisible_by_35_l1248_124851

theorem sum_squares_35_consecutive_divisible_by_35 (n : ℕ+) :
  ∃ k : ℤ, (((n + 35) * (n + 36) * (2 * (n + 35) + 1)) / 6 -
            (n * (n + 1) * (2 * n + 1)) / 6) = 35 * k :=
sorry

end sum_squares_35_consecutive_divisible_by_35_l1248_124851


namespace sum_first_105_remainder_l1248_124881

theorem sum_first_105_remainder (n : Nat) (d : Nat) : n = 105 → d = 5270 → (n * (n + 1) / 2) % d = 295 := by
  sorry

end sum_first_105_remainder_l1248_124881


namespace house_distance_ratio_l1248_124835

/-- Given three points on a road representing houses, proves the ratio of distances -/
theorem house_distance_ratio (K D M : ℝ) : 
  let KD := |K - D|
  let DM := |D - M|
  KD = 4 → KD + DM + DM + KD = 12 → KD / DM = 2 := by
  sorry

end house_distance_ratio_l1248_124835


namespace intersection_of_sets_l1248_124888

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | |x| ≤ 4} → 
  B = {x : ℝ | 4 ≤ x ∧ x < 5} → 
  A ∩ B = {4} := by
  sorry

end intersection_of_sets_l1248_124888


namespace decimal_to_fraction_l1248_124858

theorem decimal_to_fraction :
  (35 : ℚ) / 100 = 7 / 20 := by sorry

end decimal_to_fraction_l1248_124858


namespace probability_of_specific_pair_l1248_124855

def total_items : ℕ := 4
def items_to_select : ℕ := 2
def favorable_outcomes : ℕ := 1

theorem probability_of_specific_pair :
  (favorable_outcomes : ℚ) / (total_items.choose items_to_select) = 1 / 6 := by
  sorry

end probability_of_specific_pair_l1248_124855


namespace mrs_hilts_snow_amount_l1248_124852

def snow_at_mrs_hilts_house : ℕ := 29
def snow_at_brecknock_school : ℕ := 17

theorem mrs_hilts_snow_amount : snow_at_mrs_hilts_house = 29 := by sorry

end mrs_hilts_snow_amount_l1248_124852


namespace positive_real_inequality_l1248_124812

theorem positive_real_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 1/x + 1/y + 1/z) :
  x + y + z ≥ Real.sqrt ((x*y + 1)/2) + Real.sqrt ((y*z + 1)/2) + Real.sqrt ((z*x + 1)/2) :=
by sorry

end positive_real_inequality_l1248_124812


namespace mateen_backyard_area_l1248_124819

/-- A rectangular backyard with specific walking distances -/
structure Backyard where
  length : ℝ
  width : ℝ
  total_distance : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ

/-- The conditions of Mateen's backyard -/
def mateen_backyard : Backyard where
  length := 40
  width := 10
  total_distance := 1200
  length_walks := 30
  perimeter_walks := 12

/-- Theorem stating the area of Mateen's backyard -/
theorem mateen_backyard_area :
  let b := mateen_backyard
  b.length * b.width = 400 ∧
  b.length_walks * b.length = b.total_distance ∧
  b.perimeter_walks * (2 * b.length + 2 * b.width) = b.total_distance :=
by sorry

end mateen_backyard_area_l1248_124819


namespace cube_center_pyramids_l1248_124821

/-- Given a cube with edge length a, prove the volume and surface area of the pyramids formed by connecting the center to all vertices. -/
theorem cube_center_pyramids (a : ℝ) (h : a > 0) :
  ∃ (volume surface_area : ℝ),
    volume = a^3 / 6 ∧
    surface_area = a^2 * (1 + Real.sqrt 2) :=
by sorry

end cube_center_pyramids_l1248_124821


namespace polynomial_factor_l1248_124860

/-- The polynomial P(x) = x^3 - 3x^2 + cx - 8 -/
def P (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + c*x - 8

theorem polynomial_factor (c : ℝ) : 
  (∀ x, P c x = 0 ↔ (x + 2 = 0 ∨ ∃ q, P c x = (x + 2) * q)) → c = -14 := by
  sorry

end polynomial_factor_l1248_124860


namespace lcm_45_75_l1248_124879

theorem lcm_45_75 : Nat.lcm 45 75 = 225 := by sorry

end lcm_45_75_l1248_124879


namespace fraction_multiplication_l1248_124833

theorem fraction_multiplication : 
  (7 / 8 : ℚ) * (1 / 3 : ℚ) * (3 / 7 : ℚ) = 0.12499999999999997 := by
  sorry

end fraction_multiplication_l1248_124833


namespace sector_area_l1248_124896

theorem sector_area (α l : Real) (h1 : α = π / 6) (h2 : l = π / 3) :
  let r := l / α
  let s := (1 / 2) * l * r
  s = π / 3 := by
  sorry

end sector_area_l1248_124896


namespace coefficient_of_x_cubed_l1248_124875

theorem coefficient_of_x_cubed (x : ℝ) : 
  let expression := 4*(x^3 - 2*x^4) + 3*(x^2 - 3*x^3 + 4*x^6) - (5*x^4 - 2*x^3)
  ∃ (a b c d e : ℝ), expression = -3*x^3 + a*x^2 + b*x^4 + c*x^6 + d*x + e :=
by
  sorry

end coefficient_of_x_cubed_l1248_124875


namespace climb_8_stairs_l1248_124863

/-- The number of ways to climb n stairs, taking 1, 2, 3, or 4 steps at a time. -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | n + 4 => climbStairs n + climbStairs (n + 1) + climbStairs (n + 2) + climbStairs (n + 3)

/-- Theorem stating that there are 108 ways to climb 8 stairs -/
theorem climb_8_stairs : climbStairs 8 = 108 := by
  sorry

end climb_8_stairs_l1248_124863


namespace magnitude_of_vector_sum_l1248_124834

/-- Given vectors a and b in ℝ², prove that the magnitude of 2a+b is √13 -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) :
  a = (-2, 1) →
  b = (1, 0) →
  Real.sqrt ((2 * a.1 + b.1)^2 + (2 * a.2 + b.2)^2) = Real.sqrt 13 := by
  sorry

end magnitude_of_vector_sum_l1248_124834


namespace eggs_for_blueberry_and_pecan_pies_l1248_124859

theorem eggs_for_blueberry_and_pecan_pies 
  (total_eggs : ℕ) 
  (pumpkin_eggs : ℕ) 
  (apple_eggs : ℕ) 
  (cherry_eggs : ℕ) 
  (h1 : total_eggs = 1820)
  (h2 : pumpkin_eggs = 816)
  (h3 : apple_eggs = 384)
  (h4 : cherry_eggs = 120) :
  total_eggs - (pumpkin_eggs + apple_eggs + cherry_eggs) = 500 :=
by sorry

end eggs_for_blueberry_and_pecan_pies_l1248_124859


namespace two_points_explain_phenomena_l1248_124830

-- Define the type for phenomena
inductive Phenomenon : Type
| RiverChannel
| WoodenStrips
| TreePlanting
| WallFixing

-- Define a function to check if a phenomenon can be explained by "two points determine a straight line"
def explainedByTwoPoints : Phenomenon → Prop
| Phenomenon.RiverChannel => false
| Phenomenon.WoodenStrips => true
| Phenomenon.TreePlanting => true
| Phenomenon.WallFixing => true

-- State the theorem
theorem two_points_explain_phenomena :
  (explainedByTwoPoints Phenomenon.RiverChannel = false) ∧
  (explainedByTwoPoints Phenomenon.WoodenStrips = true) ∧
  (explainedByTwoPoints Phenomenon.TreePlanting = true) ∧
  (explainedByTwoPoints Phenomenon.WallFixing = true) :=
by sorry

end two_points_explain_phenomena_l1248_124830


namespace complex_point_in_third_quadrant_l1248_124820

/-- Given that i is the imaginary unit and (x+i)i = y-i where x and y are real numbers,
    prove that the point (x, y) lies in the third quadrant of the complex plane. -/
theorem complex_point_in_third_quadrant (x y : ℝ) (i : ℂ) 
  (h_i : i * i = -1) 
  (h_eq : (x + i) * i = y - i) : 
  x < 0 ∧ y < 0 := by
  sorry

end complex_point_in_third_quadrant_l1248_124820


namespace smallest_common_multiple_9_15_gt_50_l1248_124824

theorem smallest_common_multiple_9_15_gt_50 : ∃ n : ℕ, n = 90 ∧ 
  (∀ m : ℕ, m < n → (m % 9 = 0 ∧ m % 15 = 0 → m ≤ 50)) ∧
  n % 9 = 0 ∧ n % 15 = 0 ∧ n > 50 := by
  sorry

end smallest_common_multiple_9_15_gt_50_l1248_124824


namespace diamond_and_hearts_balance_l1248_124867

-- Define the symbols
variable (triangle diamond heart dot : ℕ)

-- Define the balance relation
def balances (left right : ℕ) : Prop := left = right

-- State the given conditions
axiom balance1 : balances (4 * triangle + 2 * diamond + heart) (21 * dot)
axiom balance2 : balances (2 * triangle) (diamond + heart + 5 * dot)

-- State the theorem to be proved
theorem diamond_and_hearts_balance : balances (diamond + 2 * heart) (11 * dot) := by sorry

end diamond_and_hearts_balance_l1248_124867


namespace min_value_expression_l1248_124800

theorem min_value_expression (m n : ℝ) (h1 : m > 1) (h2 : n > 0) (h3 : m^2 - 3*m + n = 0) :
  ∃ (min_val : ℝ), min_val = 9/2 ∧ ∀ (x : ℝ), (4/(m-1) + m/n) ≥ min_val :=
sorry

end min_value_expression_l1248_124800


namespace income_distribution_equation_l1248_124854

/-- Represents a person's income distribution --/
structure IncomeDistribution where
  total : ℝ
  children_percent : ℝ
  wife_percent : ℝ
  orphan_percent : ℝ
  remaining : ℝ

/-- Theorem stating the relationship between income and its distribution --/
theorem income_distribution_equation (d : IncomeDistribution) 
  (h1 : d.children_percent = 0.1)
  (h2 : d.wife_percent = 0.2)
  (h3 : d.orphan_percent = 0.1)
  (h4 : d.remaining = 500) :
  d.total - (2 * d.children_percent * d.total + 
             d.wife_percent * d.total + 
             d.orphan_percent * (d.total - (2 * d.children_percent * d.total + d.wife_percent * d.total))) = 
  d.remaining := by
  sorry

#eval 500 / 0.54  -- This will output the approximate total income

end income_distribution_equation_l1248_124854


namespace student_tickets_sold_l1248_124878

/-- Proves that the number of student tickets sold is 9 given the specified conditions -/
theorem student_tickets_sold (adult_price : ℝ) (student_price : ℝ) (total_tickets : ℕ) (total_revenue : ℝ)
  (h1 : adult_price = 4)
  (h2 : student_price = 2.5)
  (h3 : total_tickets = 59)
  (h4 : total_revenue = 222.5) :
  ∃ (student_tickets : ℕ), 
    student_tickets = 9 ∧
    (total_tickets - student_tickets : ℝ) * adult_price + (student_tickets : ℝ) * student_price = total_revenue :=
by sorry

end student_tickets_sold_l1248_124878


namespace find_lighter_orange_l1248_124826

/-- Represents a group of objects that can be weighed -/
structure WeightGroup where
  objects : Finset ℕ
  size : ℕ
  h_size : objects.card = size

/-- Represents the result of weighing two groups -/
inductive WeighResult
  | Left
  | Right
  | Equal

/-- Represents a balance scale that can compare two groups -/
def Balance := WeightGroup → WeightGroup → WeighResult

/-- The problem setup with 8 objects, 7 of equal weight and 1 lighter -/
structure OrangeSetup where
  total_objects : ℕ
  h_total : total_objects = 8
  equal_weight_objects : ℕ
  h_equal : equal_weight_objects = 7
  h_lighter : total_objects = equal_weight_objects + 1

/-- The theorem stating that the lighter object can be found in at most 2 measurements -/
theorem find_lighter_orange (setup : OrangeSetup) :
  ∃ (strategy : Balance → Balance → ℕ),
    ∀ (b : Balance), strategy b b < setup.total_objects ∧ 
    (strategy b b) ∈ Finset.range setup.total_objects := by
  sorry


end find_lighter_orange_l1248_124826


namespace puzzle_solution_l1248_124846

/-- Represents the pieces of the puzzle -/
inductive Piece
| Two
| One
| Zero
| Minus

/-- Represents the arrangement of pieces -/
def Arrangement := List Piece

/-- Checks if an arrangement forms a valid subtraction equation -/
def isValidArrangement (arr : Arrangement) : Prop := sorry

/-- Calculates the result of a valid arrangement -/
def calculateResult (arr : Arrangement) : Int := sorry

/-- The main theorem: The correct arrangement results in -100 -/
theorem puzzle_solution :
  ∃ (arr : Arrangement),
    isValidArrangement arr ∧ calculateResult arr = -100 := by
  sorry

end puzzle_solution_l1248_124846


namespace union_M_N_l1248_124898

def M : Set ℝ := {x | 1 / x > 1}
def N : Set ℝ := {x | x^2 + 2*x - 3 < 0}

theorem union_M_N : M ∪ N = Set.Ioo (-3) 1 := by
  sorry

end union_M_N_l1248_124898


namespace largest_distinct_digits_divisible_by_99_l1248_124840

def is_distinct_digits (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n.digits 10).nthLe i (by sorry) ≠ (n.digits 10).nthLe j (by sorry)

theorem largest_distinct_digits_divisible_by_99 :
  ∀ n : ℕ, n > 9876524130 → ¬(is_distinct_digits n ∧ n % 99 = 0) :=
by sorry

end largest_distinct_digits_divisible_by_99_l1248_124840


namespace average_pieces_lost_is_13_4_l1248_124861

/-- The number of games played -/
def num_games : ℕ := 5

/-- Audrey's lost pieces in each game -/
def audrey_lost : List ℕ := [6, 8, 4, 7, 10]

/-- Thomas's lost pieces in each game -/
def thomas_lost : List ℕ := [5, 6, 3, 7, 11]

/-- The average number of pieces lost per game for both players combined -/
def average_pieces_lost : ℚ :=
  (audrey_lost.sum + thomas_lost.sum : ℚ) / num_games

theorem average_pieces_lost_is_13_4 :
  average_pieces_lost = 134/10 := by sorry

end average_pieces_lost_is_13_4_l1248_124861


namespace train_speed_calculation_l1248_124801

/-- Given two trains moving in opposite directions, prove the speed of one train given the lengths, speed of the other train, and time to cross. -/
theorem train_speed_calculation (length1 length2 speed2 time_to_cross : ℝ) 
  (h1 : length1 = 300)
  (h2 : length2 = 200.04)
  (h3 : speed2 = 80)
  (h4 : time_to_cross = 9 / 3600) : 
  ∃ speed1 : ℝ, speed1 = 120.016 ∧ 
  (length1 + length2) / 1000 = (speed1 + speed2) * time_to_cross := by
  sorry

end train_speed_calculation_l1248_124801


namespace arithmetic_sequence_sum_property_l1248_124847

/-- Definition of arithmetic sequence sum -/
def arithmetic_sequence_sum (n : ℕ) : ℝ := sorry

/-- Theorem: For an arithmetic sequence with sum S_n, if S_3 = 15 and S_9 = 153, then S_6 = 66 -/
theorem arithmetic_sequence_sum_property :
  (arithmetic_sequence_sum 3 = 15) →
  (arithmetic_sequence_sum 9 = 153) →
  (arithmetic_sequence_sum 6 = 66) := by
sorry

end arithmetic_sequence_sum_property_l1248_124847


namespace fraction_sum_equation_l1248_124872

theorem fraction_sum_equation (a b : ℝ) (h1 : a ≠ b) 
  (h2 : a / b + (a + 5 * b) / (b + 5 * a) = 3) : a / b = 0.2 := by
  sorry

end fraction_sum_equation_l1248_124872


namespace constant_term_binomial_expansion_l1248_124849

theorem constant_term_binomial_expansion :
  (Finset.sum (Finset.range 10) (fun k => Nat.choose 9 k * (1 : ℝ)^k * (1 : ℝ)^(9 - k))) = 84 := by
  sorry

end constant_term_binomial_expansion_l1248_124849


namespace inequality_condition_l1248_124808

theorem inequality_condition (a b : ℝ) :
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
by sorry

end inequality_condition_l1248_124808


namespace abc_inequality_l1248_124883

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a / (1 + b) + b / (1 + c) + c / (1 + a) ≥ 3 / 2 := by
  sorry

end abc_inequality_l1248_124883


namespace rectangle_perimeter_theorem_l1248_124884

theorem rectangle_perimeter_theorem (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a * b > 2 * (a + b) → 2 * (a + b) > 16 := by
  sorry

end rectangle_perimeter_theorem_l1248_124884


namespace complex_equation_solution_l1248_124871

theorem complex_equation_solution (z : ℂ) :
  z * (1 + 3 * Complex.I) = 4 + Complex.I →
  z = 7/10 - 11/10 * Complex.I := by
  sorry

end complex_equation_solution_l1248_124871


namespace probability_six_spades_correct_l1248_124841

/-- The number of cards in a standard deck of poker cards (excluding jokers) -/
def deck_size : ℕ := 52

/-- The number of spades in a standard deck of poker cards -/
def spades_count : ℕ := 13

/-- The number of cards each player receives when 4 people play -/
def cards_per_player : ℕ := deck_size / 4

/-- The probability of a person getting exactly 6 spades when 4 people play with a standard deck -/
def probability_six_spades : ℚ :=
  (Nat.choose spades_count 6 * Nat.choose (deck_size - spades_count) (cards_per_player - 6)) /
  Nat.choose deck_size cards_per_player

theorem probability_six_spades_correct :
  probability_six_spades = (Nat.choose 13 6 * Nat.choose 39 7) / Nat.choose 52 13 := by
  sorry

end probability_six_spades_correct_l1248_124841


namespace certain_number_calculation_l1248_124814

theorem certain_number_calculation (x y : ℝ) : 
  0.12 / x * 2 = y → x = 0.1 → y = 2.4 := by
  sorry

end certain_number_calculation_l1248_124814


namespace sum_of_q_p_equals_negative_twenty_l1248_124864

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x

def xValues : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_q_p_equals_negative_twenty :
  (xValues.map (λ x => q (p x))).sum = -20 := by sorry

end sum_of_q_p_equals_negative_twenty_l1248_124864


namespace homework_problems_exist_l1248_124809

theorem homework_problems_exist : ∃ (a b c d : ℤ), 
  (a ≤ -1) ∧ (b ≤ -1) ∧ (c ≤ -1) ∧ (d ≤ -1) ∧ 
  (a * b = -(a + b)) ∧ 
  (c * d = -182 * (1 / (c + d))) :=
sorry

end homework_problems_exist_l1248_124809


namespace max_value_of_f_l1248_124813

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.cos x - 3 / 4

theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ ∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ M :=
sorry

end max_value_of_f_l1248_124813


namespace homeless_donation_problem_l1248_124873

/-- The amount given to the last set of homeless families -/
def last_set_amount (total spent_on_first_four : ℝ) : ℝ :=
  total - spent_on_first_four

/-- The problem statement -/
theorem homeless_donation_problem (total first second third fourth : ℝ) 
  (h1 : total = 4500)
  (h2 : first = 725)
  (h3 : second = 1100)
  (h4 : third = 950)
  (h5 : fourth = 815) :
  last_set_amount total (first + second + third + fourth) = 910 := by
  sorry

end homeless_donation_problem_l1248_124873


namespace puzzle_cost_calculation_l1248_124816

def puzzle_cost (initial_money savings comic_cost final_money : ℕ) : ℕ :=
  initial_money + savings - comic_cost - final_money

theorem puzzle_cost_calculation :
  puzzle_cost 8 13 2 1 = 18 := by
  sorry

end puzzle_cost_calculation_l1248_124816


namespace sqrt2_fractional_part_bounds_l1248_124838

theorem sqrt2_fractional_part_bounds :
  (∀ n : ℕ, n * Real.sqrt 2 - ⌊n * Real.sqrt 2⌋ > 1 / (2 * n * Real.sqrt 2)) ∧
  (∀ ε > 0, ∃ n : ℕ, n * Real.sqrt 2 - ⌊n * Real.sqrt 2⌋ < 1 / (2 * n * Real.sqrt 2) + ε) := by
  sorry

end sqrt2_fractional_part_bounds_l1248_124838


namespace trapezoid_perimeter_l1248_124817

/-- The perimeter of a trapezoid JKLM with given coordinates is 34 units. -/
theorem trapezoid_perimeter : 
  let j : ℝ × ℝ := (-2, -4)
  let k : ℝ × ℝ := (-2, 1)
  let l : ℝ × ℝ := (6, 7)
  let m : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := dist j k + dist k l + dist l m + dist m j
  perimeter = 34 := by sorry

end trapezoid_perimeter_l1248_124817


namespace complex_multiplication_l1248_124811

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (2 - i) * (-2 + i) = -3 + 4*i := by
  sorry

end complex_multiplication_l1248_124811


namespace picnic_class_size_l1248_124807

theorem picnic_class_size : ∃ (x : ℕ), 
  x > 0 ∧ 
  (x / 2 + x / 3 + x / 4 : ℚ) = 65 ∧ 
  x = 60 :=
by sorry

end picnic_class_size_l1248_124807


namespace fraction_to_decimal_l1248_124894

theorem fraction_to_decimal : (67 : ℚ) / (2^3 * 5^4) = 0.0134 := by
  sorry

end fraction_to_decimal_l1248_124894


namespace classroom_ratio_l1248_124857

theorem classroom_ratio (num_boys num_girls : ℕ) 
  (h_positive : num_boys > 0 ∧ num_girls > 0) :
  let total := num_boys + num_girls
  let prob_boy := num_boys / total
  let prob_girl := num_girls / total
  prob_boy = (3/4 : ℚ) * prob_girl →
  (num_boys : ℚ) / total = 3/7 := by
sorry

end classroom_ratio_l1248_124857


namespace coin_rotation_theorem_l1248_124845

/-- 
  Represents the number of degrees a coin rotates when rolling around another coin.
  
  coinA : The rolling coin
  coinB : The stationary coin
  radiusRatio : The ratio of coinB's radius to coinA's radius
  rotationDegrees : The number of degrees coinA rotates around its center
-/
def coinRotation (coinA coinB : ℝ) (radiusRatio : ℝ) (rotationDegrees : ℝ) : Prop :=
  coinA > 0 ∧ 
  coinB > 0 ∧ 
  radiusRatio = 2 ∧ 
  rotationDegrees = 3 * 360

theorem coin_rotation_theorem (coinA coinB radiusRatio rotationDegrees : ℝ) :
  coinRotation coinA coinB radiusRatio rotationDegrees →
  rotationDegrees = 1080 :=
by
  sorry

end coin_rotation_theorem_l1248_124845


namespace largest_prime_divisor_l1248_124853

/-- The base 6 number represented as a list of digits -/
def base_6_number : List Nat := [1, 0, 2, 1, 1, 1, 0, 1, 1]

/-- Convert a list of digits in base 6 to a natural number -/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The number we're working with -/
def n : Nat := to_base_10 base_6_number

/-- A number is prime if it has exactly two distinct divisors -/
def is_prime (p : Nat) : Prop :=
  p > 1 ∧ ∀ m : Nat, m > 0 → m < p → (p % m = 0 → m = 1)

/-- p divides n -/
def divides (p n : Nat) : Prop := n % p = 0

theorem largest_prime_divisor :
  ∃ (p : Nat), is_prime p ∧ divides p n ∧
  ∀ (q : Nat), is_prime q → divides q n → q ≤ p :=
by sorry

end largest_prime_divisor_l1248_124853


namespace min_value_complex_sum_l1248_124869

theorem min_value_complex_sum (a b c d : ℤ) (ζ : ℂ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_fourth_root : ζ^4 = 1)
  (h_not_one : ζ ≠ 1) :
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y z w : ℤ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
    Complex.abs (x + y*ζ + z*ζ^2 + w*ζ^3) ≥ m :=
by sorry

end min_value_complex_sum_l1248_124869


namespace rebecca_bought_four_tent_stakes_l1248_124886

/-- The number of tent stakes bought by Rebecca. -/
def tent_stakes : ℕ := sorry

/-- The number of packets of drink mix bought by Rebecca. -/
def drink_mix : ℕ := sorry

/-- The number of bottles of water bought by Rebecca. -/
def water_bottles : ℕ := sorry

/-- The total number of items bought by Rebecca. -/
def total_items : ℕ := 22

/-- Theorem stating that Rebecca bought 4 tent stakes. -/
theorem rebecca_bought_four_tent_stakes :
  (drink_mix = 3 * tent_stakes) ∧
  (water_bottles = tent_stakes + 2) ∧
  (tent_stakes + drink_mix + water_bottles = total_items) →
  tent_stakes = 4 := by
sorry

end rebecca_bought_four_tent_stakes_l1248_124886


namespace pamela_sugar_amount_l1248_124815

/-- The amount of sugar Pamela spilled in ounces -/
def sugar_spilled : ℝ := 5.2

/-- The amount of sugar Pamela has left in ounces -/
def sugar_left : ℝ := 4.6

/-- The initial amount of sugar Pamela bought in ounces -/
def initial_sugar : ℝ := sugar_spilled + sugar_left

theorem pamela_sugar_amount : initial_sugar = 9.8 := by
  sorry

end pamela_sugar_amount_l1248_124815


namespace product_325_3_base7_l1248_124804

-- Define a function to convert from base 7 to base 10
def base7ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 7
def base10ToBase7 (n : ℕ) : ℕ := sorry

-- Define the multiplication operation in base 7
def multBase7 (a b : ℕ) : ℕ := 
  base10ToBase7 (base7ToBase10 a * base7ToBase10 b)

-- State the theorem
theorem product_325_3_base7 : 
  multBase7 325 3 = 3111 := by sorry

end product_325_3_base7_l1248_124804


namespace no_egyptian_fraction_for_seven_seventeenths_l1248_124899

theorem no_egyptian_fraction_for_seven_seventeenths :
  ¬ ∃ (a b : ℕ+), (7 : ℚ) / 17 = 1 / (a : ℚ) + 1 / (b : ℚ) := by
  sorry

end no_egyptian_fraction_for_seven_seventeenths_l1248_124899


namespace a_range_when_f_decreasing_l1248_124890

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 2

-- Define the property of f being decreasing on (-∞, 6)
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 6 → y < 6 → f a x > f a y

-- State the theorem
theorem a_range_when_f_decreasing (a : ℝ) :
  is_decreasing_on_interval a → a ∈ Set.Ici 6 :=
by
  sorry

#check a_range_when_f_decreasing

end a_range_when_f_decreasing_l1248_124890


namespace cube_root_inequality_l1248_124893

theorem cube_root_inequality (x : ℝ) (h : x > 0) :
  Real.rpow x (1/3) < 3 - x ↔ x < 3 := by sorry

end cube_root_inequality_l1248_124893


namespace f_properties_l1248_124866

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem f_properties :
  ∃ (m : ℝ),
    (∀ (a x₁ x₂ : ℝ), x₁ < x₂ → f a x₁ < f a x₂) ∧
    (∀ (x : ℝ), f 1 x = -f 1 (-x)) ∧
    (m = 12/5 ∧ ∀ (x : ℝ), x ∈ Set.Icc 2 3 → f 1 x ≥ m / 2^x) ∧
    (∀ (m' : ℝ), (∀ (x : ℝ), x ∈ Set.Icc 2 3 → f 1 x ≥ m' / 2^x) → m' ≤ m) :=
by sorry

end f_properties_l1248_124866


namespace frame_diameter_l1248_124848

theorem frame_diameter (d_y : ℝ) (uncovered_fraction : ℝ) (d_x : ℝ) : 
  d_y = 12 →
  uncovered_fraction = 0.4375 →
  d_x = 16 →
  (π * (d_x / 2)^2) = (π * (d_y / 2)^2) + uncovered_fraction * (π * (d_x / 2)^2) :=
by sorry

end frame_diameter_l1248_124848


namespace garden_fencing_needed_l1248_124897

/-- Calculates the perimeter of a rectangular garden with given length and width. -/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Proves that a rectangular garden with length 300 yards and width half its length
    requires 900 yards of fencing. -/
theorem garden_fencing_needed :
  let length : ℝ := 300
  let width : ℝ := length / 2
  garden_perimeter length width = 900 := by
sorry

#eval garden_perimeter 300 150

end garden_fencing_needed_l1248_124897


namespace prob_A_at_edge_is_two_thirds_l1248_124829

/-- The number of students -/
def num_students : ℕ := 3

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := num_students.factorial

/-- The number of arrangements with A at the edge -/
def arrangements_with_A_at_edge : ℕ := 2 * (num_students - 1).factorial

/-- The probability of A standing at the edge -/
def prob_A_at_edge : ℚ := arrangements_with_A_at_edge / total_arrangements

theorem prob_A_at_edge_is_two_thirds : prob_A_at_edge = 2/3 := by
  sorry

end prob_A_at_edge_is_two_thirds_l1248_124829


namespace square_area_increase_l1248_124874

theorem square_area_increase (s : ℝ) (k : ℝ) (h1 : s > 0) (h2 : k > 0) :
  (k * s)^2 = 25 * s^2 → k = 5 := by
  sorry

end square_area_increase_l1248_124874


namespace stan_run_time_l1248_124877

/-- Calculates the total run time given the number of 3-minute songs, 2-minute songs, and additional time needed. -/
def total_run_time (three_min_songs : ℕ) (two_min_songs : ℕ) (additional_time : ℕ) : ℕ :=
  three_min_songs * 3 + two_min_songs * 2 + additional_time

/-- Proves that given 10 3-minute songs, 15 2-minute songs, and 40 minutes of additional time, the total run time is 100 minutes. -/
theorem stan_run_time :
  total_run_time 10 15 40 = 100 := by
  sorry

end stan_run_time_l1248_124877


namespace part_one_part_two_l1248_124887

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a*x - 2| - |x + 2|

-- Part 1
theorem part_one : 
  {x : ℝ | f 2 x ≤ 1} = {x : ℝ | -1/3 ≤ x ∧ x ≤ 5} :=
sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4) → (a = 1 ∨ a = -1) :=
sorry

end part_one_part_two_l1248_124887


namespace molecular_weight_BaCl2_calculation_l1248_124891

/-- The molecular weight of 8 moles of BaCl2 -/
def molecular_weight_BaCl2 (atomic_weight_Ba : ℝ) (atomic_weight_Cl : ℝ) : ℝ :=
  8 * (atomic_weight_Ba + 2 * atomic_weight_Cl)

/-- Theorem stating the molecular weight of 8 moles of BaCl2 -/
theorem molecular_weight_BaCl2_calculation :
  molecular_weight_BaCl2 137.33 35.45 = 1665.84 := by
  sorry

#eval molecular_weight_BaCl2 137.33 35.45

end molecular_weight_BaCl2_calculation_l1248_124891


namespace trapezoid_side_length_l1248_124825

/-- Given a square of side length 2, divided into a central square and two congruent trapezoids,
    if the areas are equal, then the longer parallel side of a trapezoid is 1. -/
theorem trapezoid_side_length (s : ℝ) : 
  2 > 0 ∧                             -- Square side length is positive
  s > 0 ∧                             -- Central square side length is positive
  s < 2 ∧                             -- Central square fits inside the larger square
  s^2 = (1 + s) / 2 →                 -- Areas are equal
  s = 1 :=                            -- Longer parallel side of trapezoid is 1
by sorry

end trapezoid_side_length_l1248_124825


namespace cone_sphere_volume_ratio_l1248_124895

theorem cone_sphere_volume_ratio (r : ℝ) (h : r > 0) :
  let cone_volume := (1 / 3) * π * r^3
  let sphere_volume := (4 / 3) * π * r^3
  cone_volume / sphere_volume = 1 / 4 := by sorry

end cone_sphere_volume_ratio_l1248_124895


namespace games_in_own_group_l1248_124889

/-- Represents a baseball league with two groups of teams. -/
structure BaseballLeague where
  n : ℕ  -- Number of games played against each team in own group
  m : ℕ  -- Number of games played against each team in other group

/-- Theorem about the number of games played within a team's own group. -/
theorem games_in_own_group (league : BaseballLeague)
  (h1 : league.n > 2 * league.m)
  (h2 : league.m > 4)
  (h3 : 3 * league.n + 4 * league.m = 76) :
  3 * league.n = 48 := by
sorry

end games_in_own_group_l1248_124889


namespace max_students_distribution_l1248_124822

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 1230) (h2 : pencils = 920) :
  (∃ (students : ℕ), students > 0 ∧ 
   pens % students = 0 ∧ 
   pencils % students = 0 ∧
   ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔ 
  (Nat.gcd pens pencils = 10) :=
sorry

end max_students_distribution_l1248_124822


namespace unique_solution_range_l1248_124818

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (a x : ℝ) : Prop :=
  lg (a * x + 1) = lg (x - 1) + lg (2 - x)

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  (a > -1 ∧ a ≤ -1/2) ∨ a = 3 - 2 * Real.sqrt 3

-- Theorem statement
theorem unique_solution_range :
  ∀ a : ℝ, (∃! x : ℝ, equation a x) ↔ a_range a := by sorry

end unique_solution_range_l1248_124818


namespace octal_55_to_binary_l1248_124836

/-- Converts an octal number to decimal --/
def octal_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 8 + (n % 10)

/-- Converts a decimal number to binary --/
def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

/-- Represents a binary number as a natural number --/
def binary_to_nat (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 2 * acc + d) 0

theorem octal_55_to_binary : 
  binary_to_nat (decimal_to_binary (octal_to_decimal 55)) = binary_to_nat [1,0,1,1,0,1] := by
  sorry

end octal_55_to_binary_l1248_124836


namespace inverse_proportion_problem_l1248_124843

theorem inverse_proportion_problem (x y : ℝ) (h : x * y = 12) :
  x = 5 → y = 2.4 := by
sorry

end inverse_proportion_problem_l1248_124843


namespace income_calculation_l1248_124842

/-- Given a person's income and expenditure ratio, and their savings amount, 
    calculate their income. -/
theorem income_calculation (income_ratio : ℕ) (expenditure_ratio : ℕ) (savings : ℕ) : 
  income_ratio = 5 → expenditure_ratio = 4 → savings = 3200 → 
  income_ratio * savings / (income_ratio - expenditure_ratio) = 16000 := by
  sorry

#check income_calculation

end income_calculation_l1248_124842


namespace wax_sculpture_problem_l1248_124802

theorem wax_sculpture_problem (large_animal_wax : ℕ) (small_animal_wax : ℕ) 
  (small_animal_total_wax : ℕ) (total_wax : ℕ) :
  large_animal_wax = 4 →
  small_animal_wax = 2 →
  small_animal_total_wax = 12 →
  total_wax = 20 →
  total_wax = small_animal_total_wax + (total_wax - small_animal_total_wax) :=
by sorry

end wax_sculpture_problem_l1248_124802


namespace gcd_228_1995_l1248_124810

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l1248_124810


namespace roots_of_equation_l1248_124850

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 5*x + 6)*(x - 3)*(x + 2)
  {x : ℝ | f x = 0} = {2, 3, -2} := by
sorry

end roots_of_equation_l1248_124850


namespace fraction_increase_possible_l1248_124827

theorem fraction_increase_possible : ∃ (a b : ℕ+), (a + 1 : ℚ) / (b + 100) > (a : ℚ) / b := by
  sorry

end fraction_increase_possible_l1248_124827


namespace divisibility_by_11_l1248_124862

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def is_valid_digit (d : ℕ) : Prop :=
  d ≥ 0 ∧ d ≤ 9

def five_digit_number (a b : ℕ) : ℕ :=
  40000 + a * 1000 + b * 100 + 20 + b

theorem divisibility_by_11 (a b : ℕ) 
  (h1 : is_valid_digit a) 
  (h2 : is_valid_digit b) 
  (h3 : is_divisible_by_11 (five_digit_number a b)) : 
  a = 6 ∧ is_valid_digit b := by
  sorry

end divisibility_by_11_l1248_124862


namespace triangular_grid_edges_l1248_124805

theorem triangular_grid_edges (n : ℕ) (h : n = 1001) : 
  let total_squares := n * (n + 1) / 2
  let total_edges_without_sharing := 4 * total_squares
  let shared_edges := (n - 1) * n / 2 - 1
  total_edges_without_sharing - 2 * shared_edges = 1006004 :=
sorry

end triangular_grid_edges_l1248_124805


namespace davids_math_marks_l1248_124837

theorem davids_math_marks (english physics chemistry biology average : ℕ) 
  (h1 : english = 86)
  (h2 : physics = 92)
  (h3 : chemistry = 87)
  (h4 : biology = 95)
  (h5 : average = 89)
  (h6 : (english + physics + chemistry + biology + mathematics) / 5 = average) :
  mathematics = 85 :=
by
  sorry

end davids_math_marks_l1248_124837


namespace x₂_integer_part_sum_of_arctans_l1248_124831

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := x^3 - 17*x - 18

-- Define the roots and their properties
axiom x₁ : ℝ
axiom x₂ : ℝ
axiom x₃ : ℝ
axiom x₁_range : -4 < x₁ ∧ x₁ < -3
axiom x₃_range : 4 < x₃ ∧ x₃ < 5
axiom roots_property : cubic_equation x₁ = 0 ∧ cubic_equation x₂ = 0 ∧ cubic_equation x₃ = 0

-- Theorem for the integer part of x₂
theorem x₂_integer_part : ⌊x₂⌋ = -2 := by sorry

-- Theorem for the sum of arctangents
theorem sum_of_arctans : Real.arctan x₁ + Real.arctan x₂ + Real.arctan x₃ = -π/4 := by sorry

end x₂_integer_part_sum_of_arctans_l1248_124831


namespace cos_315_degrees_l1248_124876

theorem cos_315_degrees :
  Real.cos (315 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_315_degrees_l1248_124876


namespace bagel_store_spending_l1248_124806

/-- The total amount spent by Ben and David in the bagel store -/
def total_spent (b d : ℝ) : ℝ := b + d

/-- Ben's spending is $15 more than David's spending -/
def ben_spent_more (b d : ℝ) : Prop := b = d + 15

/-- David's spending is half of Ben's spending -/
def david_spent_half (b d : ℝ) : Prop := d = b / 2

theorem bagel_store_spending (b d : ℝ) 
  (h1 : david_spent_half b d) 
  (h2 : ben_spent_more b d) : 
  total_spent b d = 45 := by
  sorry

end bagel_store_spending_l1248_124806


namespace x_squared_minus_y_squared_equals_five_l1248_124856

theorem x_squared_minus_y_squared_equals_five 
  (x y : ℝ) 
  (h1 : 23 * x + 977 * y = 2023) 
  (h2 : 977 * x + 23 * y = 2977) : 
  x^2 - y^2 = 5 := by
sorry

end x_squared_minus_y_squared_equals_five_l1248_124856


namespace first_cat_weight_l1248_124880

theorem first_cat_weight (total_weight second_cat_weight third_cat_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : second_cat_weight = 7)
  (h3 : third_cat_weight = 4)
  : total_weight - second_cat_weight - third_cat_weight = 2 := by
  sorry

end first_cat_weight_l1248_124880


namespace waitress_income_fraction_l1248_124803

theorem waitress_income_fraction (salary : ℝ) (tips : ℝ) (income : ℝ) : 
  salary > 0 →
  tips = (7 / 4) * salary →
  income = salary + tips →
  tips / income = 7 / 11 := by
sorry

end waitress_income_fraction_l1248_124803


namespace pairwise_sums_not_distinct_l1248_124839

theorem pairwise_sums_not_distinct (n : ℕ+) (A : Finset (ZMod n)) :
  A.card > 1 + Real.sqrt (n + 4) →
  ∃ (a b c d : ZMod n), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ 
    (a, b) ≠ (c, d) ∧ (a, b) ≠ (d, c) ∧ a + b = c + d :=
by sorry

end pairwise_sums_not_distinct_l1248_124839


namespace father_son_age_ratio_l1248_124870

/-- Proves that given a father who is currently 45 years old, and after 15 years
    will be twice as old as his son, the current ratio of the father's age to
    the son's age is 3:1. -/
theorem father_son_age_ratio :
  ∀ (father_age son_age : ℕ),
    father_age = 45 →
    father_age + 15 = 2 * (son_age + 15) →
    father_age / son_age = 3 :=
by sorry

end father_son_age_ratio_l1248_124870
