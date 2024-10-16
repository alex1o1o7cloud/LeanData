import Mathlib

namespace NUMINAMATH_CALUDE_birds_meeting_time_l2469_246990

/-- The time taken for two birds flying in opposite directions to meet -/
theorem birds_meeting_time 
  (duck_time : ℝ) 
  (goose_time : ℝ) 
  (duck_time_positive : duck_time > 0)
  (goose_time_positive : goose_time > 0) :
  ∃ x : ℝ, x > 0 ∧ (1 / duck_time + 1 / goose_time) * x = 1 :=
sorry

end NUMINAMATH_CALUDE_birds_meeting_time_l2469_246990


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2469_246933

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_difference : 
  let a := -8  -- First term
  let d := 7   -- Common difference (derived from -1 - (-8))
  let seq := arithmeticSequence a d
  (seq 110 - seq 100).natAbs = 70 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l2469_246933


namespace NUMINAMATH_CALUDE_toys_donation_problem_l2469_246999

theorem toys_donation_problem (leila_bags : ℕ) (leila_toys_per_bag : ℕ) 
  (mohamed_bags : ℕ) (extra_toys : ℕ) :
  leila_bags = 2 →
  leila_toys_per_bag = 25 →
  mohamed_bags = 3 →
  extra_toys = 7 →
  (mohamed_bags * ((leila_bags * leila_toys_per_bag + extra_toys) / mohamed_bags) = 
   leila_bags * leila_toys_per_bag + extra_toys) ∧
  ((leila_bags * leila_toys_per_bag + extra_toys) / mohamed_bags = 19) :=
by sorry

end NUMINAMATH_CALUDE_toys_donation_problem_l2469_246999


namespace NUMINAMATH_CALUDE_remainder_after_adding_2025_l2469_246972

theorem remainder_after_adding_2025 (n : ℤ) : n % 5 = 3 → (n + 2025) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_after_adding_2025_l2469_246972


namespace NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l2469_246914

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Part I
theorem solution_set_part_I :
  {x : ℝ | f x 2 + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Part II
theorem range_of_a_part_II :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 3, f x a ≤ 3) ↔ a ∈ Set.Icc (-3) 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_I_range_of_a_part_II_l2469_246914


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2469_246928

theorem fruit_seller_apples : ∀ (original : ℕ),
  (original : ℝ) * (1 - 0.4) = 420 → original = 700 := by
  sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2469_246928


namespace NUMINAMATH_CALUDE_peach_count_theorem_l2469_246995

def audrey_initial : ℕ := 26
def paul_initial : ℕ := 48
def maya_initial : ℕ := 57

def audrey_multiplier : ℕ := 3
def paul_multiplier : ℕ := 2
def maya_additional : ℕ := 20

def total_peaches : ℕ := 
  (audrey_initial + audrey_initial * audrey_multiplier) +
  (paul_initial + paul_initial * paul_multiplier) +
  (maya_initial + maya_additional)

theorem peach_count_theorem : total_peaches = 325 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_theorem_l2469_246995


namespace NUMINAMATH_CALUDE_cookie_theorem_l2469_246913

def cookie_problem (initial_cookies eaten_cookies given_cookies : ℕ) : Prop :=
  initial_cookies = eaten_cookies + given_cookies ∧
  eaten_cookies - given_cookies = 11

theorem cookie_theorem :
  cookie_problem 17 14 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_theorem_l2469_246913


namespace NUMINAMATH_CALUDE_train_length_l2469_246946

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 225 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2469_246946


namespace NUMINAMATH_CALUDE_problem_solution_l2469_246953

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem statement
theorem problem_solution :
  (∀ x : ℝ, f x ≤ 4 ↔ -2 ≤ x ∧ x ≤ 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → |x + 3| + |x + a| < x + 6) ↔ -1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2469_246953


namespace NUMINAMATH_CALUDE_unique_solution_l2469_246904

theorem unique_solution (a b c : ℝ) : 
  a > 4 ∧ b > 4 ∧ c > 4 ∧
  (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 48 →
  a = 9 ∧ b = 8 ∧ c = 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2469_246904


namespace NUMINAMATH_CALUDE_max_value_xy_over_x2_plus_y2_l2469_246922

theorem max_value_xy_over_x2_plus_y2 (x y : ℝ) 
  (hx : 1/4 ≤ x ∧ x ≤ 3/5) (hy : 2/7 ≤ y ∧ y ≤ 1/2) :
  x * y / (x^2 + y^2) ≤ 2/5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xy_over_x2_plus_y2_l2469_246922


namespace NUMINAMATH_CALUDE_polynomial_root_l2469_246903

theorem polynomial_root : ∃ (x : ℝ), 2 * x^5 + x^4 - 20 * x^3 - 10 * x^2 + 2 * x + 1 = 0 ∧ x = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_l2469_246903


namespace NUMINAMATH_CALUDE_unique_digit_for_divisibility_by_nine_l2469_246950

def sum_of_digits (n : ℕ) : ℕ := 8 + 6 + 5 + n + 7 + 4 + 3 + 2

theorem unique_digit_for_divisibility_by_nine :
  ∃! n : ℕ, n ≤ 9 ∧ (sum_of_digits n) % 9 = 0 ∧ n = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_digit_for_divisibility_by_nine_l2469_246950


namespace NUMINAMATH_CALUDE_triangle_side_constraints_l2469_246996

theorem triangle_side_constraints (n : ℕ+) : 
  (2 * n + 10 < 3 * n + 5 ∧ 3 * n + 5 < n + 15) ∧
  (2 * n + 10 + (n + 15) > 3 * n + 5) ∧
  (2 * n + 10 + (3 * n + 5) > n + 15) ∧
  (n + 15 + (3 * n + 5) > 2 * n + 10) ↔
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_constraints_l2469_246996


namespace NUMINAMATH_CALUDE_consecutive_product_divisibility_l2469_246988

theorem consecutive_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 7 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 14 * m) ∧
  (∃ m : ℤ, n = 21 * m) ∧
  (∃ m : ℤ, n = 42 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 28 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_divisibility_l2469_246988


namespace NUMINAMATH_CALUDE_expand_expression_l2469_246935

theorem expand_expression (x : ℝ) : (x + 3) * (4 * x - 8) - 2 * x = 4 * x^2 + 2 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2469_246935


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2469_246989

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 4 ∧
  point.x = -2 ∧ point.y = -3 →
  ∃ (l : Line), 
    pointOnLine point l ∧ 
    perpendicular l given_line ∧
    l.a = 3 ∧ l.b = 2 ∧ l.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2469_246989


namespace NUMINAMATH_CALUDE_quadratic_equivalence_l2469_246949

/-- A quadratic function with vertex form parameters -/
def quad_vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

/-- A quadratic function in standard form -/
def quad_standard_form (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem stating the equivalence of the quadratic function with given properties -/
theorem quadratic_equivalence :
  ∃ (a b c : ℝ),
    (∀ x, quad_vertex_form (1/2) 4 3 x = quad_standard_form a b c x) ∧
    quad_standard_form a b c 2 = 5 ∧
    a = 1/2 ∧ b = -4 ∧ c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalence_l2469_246949


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2469_246966

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 1 = 2 →                    -- a_1 = 2
  a 2 + a 3 = 13 →             -- a_2 + a_3 = 13
  a 4 + a 5 + a 6 = 42 :=      -- conclusion: a_4 + a_5 + a_6 = 42
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2469_246966


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2469_246912

theorem complex_equation_solution (z : ℂ) : z + Complex.abs z = 1 + Complex.I → z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2469_246912


namespace NUMINAMATH_CALUDE_yellow_tiled_area_l2469_246962

theorem yellow_tiled_area (length : ℝ) (width : ℝ) (yellow_ratio : ℝ) : 
  length = 3.6 → 
  width = 2.5 * length → 
  yellow_ratio = 1 / 2 → 
  yellow_ratio * (length * width) = 16.2 := by
sorry

end NUMINAMATH_CALUDE_yellow_tiled_area_l2469_246962


namespace NUMINAMATH_CALUDE_wall_volume_is_12_8_l2469_246981

/-- Calculates the volume of a wall given its dimensions --/
def wall_volume (breadth : ℝ) : ℝ :=
  let height := 5 * breadth
  let length := 8 * height
  breadth * height * length

/-- Theorem stating that the volume of the wall with given dimensions is 12.8 cubic meters --/
theorem wall_volume_is_12_8 :
  wall_volume (40 / 100) = 12.8 := by sorry

end NUMINAMATH_CALUDE_wall_volume_is_12_8_l2469_246981


namespace NUMINAMATH_CALUDE_sector_area_l2469_246955

/-- The area of a sector of a circle with radius 5 cm and arc length 4 cm is 10 cm². -/
theorem sector_area (r : ℝ) (arc_length : ℝ) (h1 : r = 5) (h2 : arc_length = 4) :
  (arc_length / (2 * π * r)) * (π * r^2) = 10 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l2469_246955


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l2469_246980

/-- The number of meals in a day -/
def num_meals : ℕ := 3

/-- The number of fruit types available -/
def num_fruits : ℕ := 4

/-- The probability of choosing a specific fruit at any meal -/
def prob_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruits * (prob_fruit ^ num_meals)

theorem joe_fruit_probability :
  1 - prob_same_fruit = 15 / 16 :=
sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l2469_246980


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2469_246994

theorem negative_fraction_comparison : -3/4 > -5/6 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2469_246994


namespace NUMINAMATH_CALUDE_monic_quadratic_root_l2469_246985

theorem monic_quadratic_root (x : ℂ) : x^2 + 4*x + 9 = 0 ↔ x = -2 - Complex.I * Real.sqrt 5 ∨ x = -2 + Complex.I * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_monic_quadratic_root_l2469_246985


namespace NUMINAMATH_CALUDE_square_sum_and_product_l2469_246975

theorem square_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 49) : 
  x^2 + y^2 = 25 ∧ x * y = -12 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l2469_246975


namespace NUMINAMATH_CALUDE_partner_a_receives_4800_l2469_246945

/-- Calculates the money received by partner a in a business partnership --/
def money_received_by_a (a_investment b_investment total_profit : ℚ) : ℚ :=
  let management_fee := 0.1 * total_profit
  let remaining_profit := total_profit - management_fee
  let total_investment := a_investment + b_investment
  let a_profit_share := (a_investment / total_investment) * remaining_profit
  management_fee + a_profit_share

/-- Theorem stating that given the problem conditions, partner a receives 4800 rs --/
theorem partner_a_receives_4800 :
  money_received_by_a 20000 25000 9600 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_partner_a_receives_4800_l2469_246945


namespace NUMINAMATH_CALUDE_simplify_fraction_l2469_246959

theorem simplify_fraction : (150 : ℚ) / 225 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2469_246959


namespace NUMINAMATH_CALUDE_cistern_leak_emptying_time_l2469_246977

theorem cistern_leak_emptying_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 8) 
  (h2 : leak_fill_time = 10) : 
  (1 / (1 / leak_fill_time - 1 / normal_fill_time)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_emptying_time_l2469_246977


namespace NUMINAMATH_CALUDE_win_probability_comparison_l2469_246919

theorem win_probability_comparison :
  let p : ℝ := 1 / 2  -- Probability of winning a single game
  let n₁ : ℕ := 4     -- Total number of games in scenario 1
  let k₁ : ℕ := 3     -- Number of wins needed in scenario 1
  let n₂ : ℕ := 8     -- Total number of games in scenario 2
  let k₂ : ℕ := 5     -- Number of wins needed in scenario 2
  
  -- Probability of winning exactly k₁ out of n₁ games
  let prob₁ : ℝ := (n₁.choose k₁ : ℝ) * p ^ k₁ * (1 - p) ^ (n₁ - k₁)
  
  -- Probability of winning exactly k₂ out of n₂ games
  let prob₂ : ℝ := (n₂.choose k₂ : ℝ) * p ^ k₂ * (1 - p) ^ (n₂ - k₂)
  
  prob₁ > prob₂ := by sorry

end NUMINAMATH_CALUDE_win_probability_comparison_l2469_246919


namespace NUMINAMATH_CALUDE_linda_furniture_fraction_l2469_246944

def original_savings : ℚ := 1200
def tv_cost : ℚ := 300

def furniture_cost : ℚ := original_savings - tv_cost

theorem linda_furniture_fraction :
  furniture_cost / original_savings = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_linda_furniture_fraction_l2469_246944


namespace NUMINAMATH_CALUDE_percentage_anti_lock_and_cd_l2469_246910

-- Define the percentages as real numbers between 0 and 1
def power_windows : Real := 0.60
def anti_lock_brakes : Real := 0.25
def cd_player : Real := 0.75
def power_windows_and_anti_lock : Real := 0.10
def power_windows_and_cd : Real := 0.22
def cd_player_only : Real := 0.38

-- Define the theorem
theorem percentage_anti_lock_and_cd :
  let anti_lock_and_cd := cd_player - cd_player_only - power_windows_and_cd
  anti_lock_and_cd = 0.15 := by sorry

end NUMINAMATH_CALUDE_percentage_anti_lock_and_cd_l2469_246910


namespace NUMINAMATH_CALUDE_remainder_theorem_l2469_246951

theorem remainder_theorem (y : ℤ) : 
  ∃ (P : ℤ → ℤ), y^50 = (y^2 - 5*y + 6) * P y + (2^50*(y-3) - 3^50*(y-2)) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2469_246951


namespace NUMINAMATH_CALUDE_M_intersect_N_l2469_246967

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem M_intersect_N : M ∩ N = {5, 7, 9} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2469_246967


namespace NUMINAMATH_CALUDE_traced_path_is_asterisk_l2469_246998

/-- A regular n-gon in the plane -/
structure RegularNGon where
  n : ℕ
  center : ℝ × ℝ
  vertices : Fin n → ℝ × ℝ

/-- Triangle formed by two adjacent vertices and the center of a regular n-gon -/
structure TriangleABO (ngon : RegularNGon) where
  A : Fin ngon.n
  B : Fin ngon.n
  hAdjacent : (A.val + 1) % ngon.n = B.val

/-- The path traced by point O when triangle ABO glides around the n-gon -/
def tracedPath (ngon : RegularNGon) : Set (ℝ × ℝ) := sorry

/-- An asterisk consisting of n segments emanating from the center -/
def asterisk (center : ℝ × ℝ) (n : ℕ) (length : ℝ) : Set (ℝ × ℝ) := sorry

/-- Main theorem: The path traced by O forms an asterisk -/
theorem traced_path_is_asterisk (ngon : RegularNGon) :
  ∃ (length : ℝ), tracedPath ngon = asterisk ngon.center ngon.n length := by sorry

end NUMINAMATH_CALUDE_traced_path_is_asterisk_l2469_246998


namespace NUMINAMATH_CALUDE_payment_methods_2005_l2469_246960

/-- The number of ways to pay a given amount using 1 yuan and 2 yuan banknotes -/
def payment_methods (amount : ℕ) : ℕ :=
  if amount % 2 = 0 then
    (amount + 2) / 2
  else
    (amount + 1) / 2

/-- Theorem stating that there are 1003 ways to pay 2005 yuan using 1 yuan and 2 yuan banknotes -/
theorem payment_methods_2005 : payment_methods 2005 = 1003 := by
  sorry

end NUMINAMATH_CALUDE_payment_methods_2005_l2469_246960


namespace NUMINAMATH_CALUDE_min_value_at_four_l2469_246964

/-- The quadratic function we're minimizing -/
def f (x : ℝ) : ℝ := x^2 - 8*x + 15

/-- Theorem stating that f(x) achieves its minimum when x = 4 -/
theorem min_value_at_four :
  ∀ x : ℝ, f x ≥ f 4 := by sorry

end NUMINAMATH_CALUDE_min_value_at_four_l2469_246964


namespace NUMINAMATH_CALUDE_gold_bars_theorem_l2469_246973

/-- Represents the masses of five gold bars -/
structure GoldBars where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  h1 : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e
  h2 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e
  h3 : (a = 1 ∧ b = 2) ∨ (a = 1 ∧ c = 2) ∨ (a = 1 ∧ d = 2) ∨ (a = 1 ∧ e = 2) ∨
       (b = 1 ∧ c = 2) ∨ (b = 1 ∧ d = 2) ∨ (b = 1 ∧ e = 2) ∨
       (c = 1 ∧ d = 2) ∨ (c = 1 ∧ e = 2) ∨ (d = 1 ∧ e = 2)

/-- Condition for equal division of remaining bars -/
def canDivideEqually (bars : GoldBars) : Prop :=
  (bars.c + bars.d + bars.e = bars.a + bars.b) ∧
  (bars.b + bars.d + bars.e = bars.a + bars.c) ∧
  (bars.b + bars.c + bars.e = bars.a + bars.d) ∧
  (bars.a + bars.d + bars.e = bars.b + bars.c) ∧
  (bars.a + bars.c + bars.e = bars.b + bars.d) ∧
  (bars.a + bars.b + bars.e = bars.c + bars.d) ∧
  (bars.a + bars.c + bars.d = bars.b + bars.e) ∧
  (bars.a + bars.b + bars.d = bars.c + bars.e) ∧
  (bars.a + bars.b + bars.c = bars.d + bars.e)

/-- The main theorem -/
theorem gold_bars_theorem (bars : GoldBars) (h : canDivideEqually bars) :
  (bars.a = 1 ∧ bars.b = 1 ∧ bars.c = 2 ∧ bars.d = 2 ∧ bars.e = 2) ∨
  (bars.a = 1 ∧ bars.b = 2 ∧ bars.c = 3 ∧ bars.d = 3 ∧ bars.e = 3) ∨
  (bars.a = 1 ∧ bars.b = 1 ∧ bars.c = 1 ∧ bars.d = 1 ∧ bars.e = 2) :=
by sorry

end NUMINAMATH_CALUDE_gold_bars_theorem_l2469_246973


namespace NUMINAMATH_CALUDE_modulus_of_one_over_one_plus_i_l2469_246965

open Complex

theorem modulus_of_one_over_one_plus_i : 
  let z : ℂ := 1 / (1 + I)
  abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_one_over_one_plus_i_l2469_246965


namespace NUMINAMATH_CALUDE_a_3_value_l2469_246902

/-- Given a polynomial expansion of (1+x)(a-x)^6, prove that a₃ = -5 when the sum of all coefficients is zero. -/
theorem a_3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) : 
  (∀ x, (1 + x) * (a - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) →
  a₃ = -5 := by
sorry

end NUMINAMATH_CALUDE_a_3_value_l2469_246902


namespace NUMINAMATH_CALUDE_lab_items_per_tech_l2469_246915

/-- Proves that each lab tech gets 14 items (coats and uniforms combined) given the problem conditions -/
theorem lab_items_per_tech (uniforms : ℕ) (coats : ℕ) (lab_techs : ℕ) : 
  uniforms = 12 →
  coats = 6 * uniforms →
  lab_techs = uniforms / 2 →
  (coats + uniforms) / lab_techs = 14 :=
by
  sorry

#check lab_items_per_tech

end NUMINAMATH_CALUDE_lab_items_per_tech_l2469_246915


namespace NUMINAMATH_CALUDE_number_ordering_l2469_246957

theorem number_ordering : (6 : ℝ)^10 < 3^20 ∧ 3^20 < 2^30 := by
  sorry

end NUMINAMATH_CALUDE_number_ordering_l2469_246957


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_2160_l2469_246956

theorem least_k_cube_divisible_by_2160 : 
  ∃ k : ℕ+, (k : ℕ)^3 % 2160 = 0 ∧ ∀ m : ℕ+, (m : ℕ)^3 % 2160 = 0 → k ≤ m := by
  sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_2160_l2469_246956


namespace NUMINAMATH_CALUDE_student_rank_theorem_l2469_246936

/-- Given a group of students, calculate the rank from left based on total students and rank from right -/
def rankFromLeft (totalStudents : ℕ) (rankFromRight : ℕ) : ℕ :=
  totalStudents - rankFromRight + 1

/-- Theorem stating that in a group of 10 students, the 6th from right is 5th from left -/
theorem student_rank_theorem :
  let totalStudents : ℕ := 10
  let rankFromRight : ℕ := 6
  rankFromLeft totalStudents rankFromRight = 5 := by
  sorry


end NUMINAMATH_CALUDE_student_rank_theorem_l2469_246936


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2469_246938

theorem sufficient_not_necessary_condition :
  (∀ x > 0, x + (1/18) / (2*x) ≥ 1/3) ∧
  (∃ a ≠ 1/18, ∀ x > 0, x + a / (2*x) ≥ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2469_246938


namespace NUMINAMATH_CALUDE_art_show_sales_l2469_246907

theorem art_show_sales (total : ℕ) (ratio_remaining : ℕ) (ratio_sold : ℕ) (sold : ℕ) : 
  total = 153 →
  ratio_remaining = 9 →
  ratio_sold = 8 →
  (total - sold) * ratio_sold = sold * ratio_remaining →
  sold = 72 := by
sorry

end NUMINAMATH_CALUDE_art_show_sales_l2469_246907


namespace NUMINAMATH_CALUDE_expression_bounds_l2469_246934

theorem expression_bounds (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  let expr := |x + y + z| / (|x| + |y| + |z|)
  (∀ a b c : ℝ, a ≠ 0 → b ≠ 0 → c ≠ 0 → |a + b + c| / (|a| + |b| + |c|) ≤ 1) ∧
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ |a + b + c| / (|a| + |b| + |c|) = 0) ∧
  (1 - 0 = 1) := by
sorry

end NUMINAMATH_CALUDE_expression_bounds_l2469_246934


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_l2469_246984

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Proof that (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean : is_pythagorean_triple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_l2469_246984


namespace NUMINAMATH_CALUDE_f_strictly_increasing_b_range_l2469_246905

/-- Piecewise function f(x) defined by parameter b -/
noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (2*b - 1)*x + b - 1 else -x^2 + (2 - b)*x

/-- Theorem stating the range of b for which f is strictly increasing -/
theorem f_strictly_increasing_b_range :
  {b : ℝ | ∀ (x₁ x₂ : ℝ), x₁ ≠ x₂ → (f b x₁ - f b x₂) / (x₁ - x₂) > 0} = Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_b_range_l2469_246905


namespace NUMINAMATH_CALUDE_inequality_proof_l2469_246906

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a * b) / (a^5 + a * b + b^5) + (b * c) / (b^5 + b * c + c^5) + (c * a) / (c^5 + c * a + a^5) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2469_246906


namespace NUMINAMATH_CALUDE_meeting_distance_l2469_246924

/-- Proves that given two people 75 miles apart, walking towards each other at constant speeds of 4 mph and 6 mph respectively, the person walking at 6 mph will have walked 45 miles when they meet. -/
theorem meeting_distance (initial_distance : ℝ) (speed_fred : ℝ) (speed_sam : ℝ) 
  (h1 : initial_distance = 75)
  (h2 : speed_fred = 4)
  (h3 : speed_sam = 6) :
  let distance_sam := initial_distance * speed_sam / (speed_fred + speed_sam)
  distance_sam = 45 := by
  sorry

#check meeting_distance

end NUMINAMATH_CALUDE_meeting_distance_l2469_246924


namespace NUMINAMATH_CALUDE_square_side_length_l2469_246942

theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ s : ℝ, s > 0 ∧ s * s = d * d / 2 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2469_246942


namespace NUMINAMATH_CALUDE_expression_simplification_l2469_246900

theorem expression_simplification (a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hab : 3 * a - b / 3 ≠ 0) : 
  (3 * a - b / 3)⁻¹ * ((3 * a)⁻¹ - (b / 3)⁻¹) = -(a * b)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2469_246900


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2469_246941

/-- Represents the problem of a train crossing a platform --/
structure TrainProblem where
  train_speed_kmph : ℝ
  train_speed_ms : ℝ
  platform_length : ℝ
  time_to_cross_man : ℝ

/-- The theorem stating the time taken for the train to cross the platform --/
theorem train_platform_crossing_time (p : TrainProblem)
  (h1 : p.train_speed_kmph = 72)
  (h2 : p.train_speed_ms = p.train_speed_kmph / 3.6)
  (h3 : p.platform_length = 300)
  (h4 : p.time_to_cross_man = 15)
  : p.train_speed_ms * p.time_to_cross_man + p.platform_length = p.train_speed_ms * 30 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2469_246941


namespace NUMINAMATH_CALUDE_one_common_root_sum_l2469_246920

theorem one_common_root_sum (a b : ℝ) :
  (∃! x, x^2 + a*x + b = 0 ∧ x^2 + b*x + a = 0) →
  a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_one_common_root_sum_l2469_246920


namespace NUMINAMATH_CALUDE_sqrt_2x_plus_1_domain_l2469_246978

theorem sqrt_2x_plus_1_domain (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x + 1) ↔ x ≥ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_plus_1_domain_l2469_246978


namespace NUMINAMATH_CALUDE_train_length_l2469_246943

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 54 →
  crossing_time = 58.9952803775698 →
  bridge_length = 720 →
  ∃ (train_length : ℝ), abs (train_length - 164.93) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2469_246943


namespace NUMINAMATH_CALUDE_f_2011_is_zero_l2469_246931

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2011_is_zero (f : ℝ → ℝ) (h_odd : is_odd f) (h_period : ∀ x, f (x + 1) = -f x) : f 2011 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2011_is_zero_l2469_246931


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2469_246982

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence {a_n} where a_4 = 5 and a_9 = 17, a_14 = 29 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_a4 : a 4 = 5) 
    (h_a9 : a 9 = 17) : 
  a 14 = 29 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2469_246982


namespace NUMINAMATH_CALUDE_inverse_function_point_sum_l2469_246932

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- Assume f and f_inv are inverse functions
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- Define the condition that (1,2) is on the graph of y = f(x)/2
axiom point_on_graph : f 1 = 4

-- Theorem to prove
theorem inverse_function_point_sum :
  ∃ a b : ℝ, f_inv a = 2*b ∧ a + b = 9/2 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_sum_l2469_246932


namespace NUMINAMATH_CALUDE_nick_babysitting_charge_l2469_246940

/-- Nick's babysitting charge calculation -/
theorem nick_babysitting_charge (y : ℝ) : 
  let travel_cost : ℝ := 7
  let hourly_rate : ℝ := 10
  let total_charge := hourly_rate * y + travel_cost
  total_charge = 10 * y + 7 := by sorry

end NUMINAMATH_CALUDE_nick_babysitting_charge_l2469_246940


namespace NUMINAMATH_CALUDE_messages_cleared_in_29_days_l2469_246917

/-- The number of days required to clear all unread messages -/
def days_to_clear_messages (initial_messages : ℕ) (read_per_day : ℕ) (new_per_day : ℕ) : ℕ :=
  (initial_messages + (read_per_day - new_per_day) - 1) / (read_per_day - new_per_day)

/-- Proof that it takes 29 days to clear all unread messages -/
theorem messages_cleared_in_29_days :
  days_to_clear_messages 198 15 8 = 29 := by
  sorry

#eval days_to_clear_messages 198 15 8

end NUMINAMATH_CALUDE_messages_cleared_in_29_days_l2469_246917


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2469_246925

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  (Real.sqrt (x - 9 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 9 * Real.sqrt (x - 9)) - 3) ↔
  x ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2469_246925


namespace NUMINAMATH_CALUDE_three_friends_came_later_l2469_246961

/-- The number of friends who came over later -/
def friends_came_later (initial_friends final_total : ℕ) : ℕ :=
  final_total - initial_friends

/-- Theorem: Given 4 initial friends and a final total of 7 people,
    prove that 3 friends came over later -/
theorem three_friends_came_later :
  friends_came_later 4 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_friends_came_later_l2469_246961


namespace NUMINAMATH_CALUDE_sallys_gold_card_balance_fraction_l2469_246970

/-- Represents a credit card with a spending limit and balance -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Represents Sally's credit cards -/
structure SallysCards where
  gold : CreditCard
  platinum : CreditCard

/-- The conditions of Sally's credit cards -/
def sallys_cards_conditions (cards : SallysCards) : Prop :=
  cards.platinum.limit = 2 * cards.gold.limit ∧
  cards.platinum.balance = (1 / 6) * cards.platinum.limit ∧
  cards.platinum.balance + cards.gold.balance = (1 / 3) * cards.platinum.limit

/-- The theorem representing the problem -/
theorem sallys_gold_card_balance_fraction (cards : SallysCards) 
  (h : sallys_cards_conditions cards) : 
  cards.gold.balance = (1 / 3) * cards.gold.limit := by
  sorry

end NUMINAMATH_CALUDE_sallys_gold_card_balance_fraction_l2469_246970


namespace NUMINAMATH_CALUDE_cube_root_function_l2469_246923

theorem cube_root_function (k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = k * (x ^ (1/3))) →
  y 64 = 4 →
  y 8 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_function_l2469_246923


namespace NUMINAMATH_CALUDE_min_trips_for_field_trip_l2469_246983

/-- The minimum number of trips required to transport all students -/
def min_trips (total_students : ℕ) (num_buses : ℕ) (bus_capacity : ℕ) : ℕ :=
  (total_students + num_buses * bus_capacity - 1) / (num_buses * bus_capacity)

theorem min_trips_for_field_trip :
  min_trips 520 5 45 = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_trips_for_field_trip_l2469_246983


namespace NUMINAMATH_CALUDE_remaining_amount_is_99_l2469_246979

/-- Calculates the remaining amount in US dollars after transactions --/
def remaining_amount (initial_usd : ℝ) (initial_euro : ℝ) (exchange_rate : ℝ) 
  (supermarket_spend : ℝ) (book_cost_euro : ℝ) (lunch_cost : ℝ) : ℝ :=
  initial_usd + initial_euro * exchange_rate - supermarket_spend - book_cost_euro * exchange_rate - lunch_cost

/-- Proves that the remaining amount is 99 US dollars given the initial amounts and transactions --/
theorem remaining_amount_is_99 :
  remaining_amount 78 50 1.2 15 10 12 = 99 := by
  sorry

#eval remaining_amount 78 50 1.2 15 10 12

end NUMINAMATH_CALUDE_remaining_amount_is_99_l2469_246979


namespace NUMINAMATH_CALUDE_canoe_rowing_probability_l2469_246926

def left_oar_prob : ℚ := 3/5
def right_oar_prob : ℚ := 3/5

theorem canoe_rowing_probability :
  let prob_at_least_one_oar := 
    left_oar_prob * right_oar_prob + 
    left_oar_prob * (1 - right_oar_prob) + 
    (1 - left_oar_prob) * right_oar_prob
  prob_at_least_one_oar = 21/25 := by
sorry

end NUMINAMATH_CALUDE_canoe_rowing_probability_l2469_246926


namespace NUMINAMATH_CALUDE_reciprocal_of_four_l2469_246916

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_four : reciprocal 4 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_four_l2469_246916


namespace NUMINAMATH_CALUDE_system_of_equations_l2469_246993

theorem system_of_equations (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l2469_246993


namespace NUMINAMATH_CALUDE_negative_division_rule_div_negative_64_negative_32_l2469_246937

theorem negative_division_rule (x y : ℤ) (hy : y ≠ 0) : (-x) / (-y) = x / y := by sorry

theorem div_negative_64_negative_32 : (-64) / (-32) = 2 := by sorry

end NUMINAMATH_CALUDE_negative_division_rule_div_negative_64_negative_32_l2469_246937


namespace NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l2469_246976

-- Define a type for data sets
structure DataSet where
  -- Add necessary fields for a data set
  nonEmpty : Bool

-- Define a predicate for whether a regression equation exists for a data set
def hasRegressionEquation (d : DataSet) : Prop := sorry

-- Theorem stating that not every data set has a regression equation
theorem not_all_data_sets_have_regression_equation :
  ¬ ∀ (d : DataSet), hasRegressionEquation d := by
  sorry


end NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l2469_246976


namespace NUMINAMATH_CALUDE_cubic_root_function_l2469_246911

theorem cubic_root_function (k : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = k * x^(1/3)) →
  y 64 = 4 * Real.sqrt 3 →
  y 8 = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_function_l2469_246911


namespace NUMINAMATH_CALUDE_combined_speed_difference_l2469_246921

-- Define the speed functions for each train
def zA (s : ℝ) : ℝ := s^2 + 2*s
def zB (s : ℝ) : ℝ := 2*s^2 + 3*s + 1
def zC (s : ℝ) : ℝ := s^3 - 4*s

-- Define the time constraints for each train
def trainA_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 7
def trainB_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 5
def trainC_time_constraint (s : ℝ) : Prop := 0 ≤ s ∧ s ≤ 4

-- Theorem statement
theorem combined_speed_difference :
  trainA_time_constraint 7 ∧
  trainA_time_constraint 2 ∧
  trainB_time_constraint 5 ∧
  trainB_time_constraint 2 ∧
  trainC_time_constraint 4 ∧
  trainC_time_constraint 2 →
  (zA 7 - zA 2) + (zB 5 - zB 2) + (zC 4 - zC 2) = 154 := by
  sorry

end NUMINAMATH_CALUDE_combined_speed_difference_l2469_246921


namespace NUMINAMATH_CALUDE_evaluate_expression_l2469_246971

theorem evaluate_expression : (4^4 - 4*(4-2)^4)^4 = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2469_246971


namespace NUMINAMATH_CALUDE_inequality_solution_l2469_246929

theorem inequality_solution (x : ℝ) : 
  (6 * x^2 + 12 * x - 35) / ((x - 2) * (3 * x + 6)) < 2 ↔ 
  (x > -2 ∧ x < 11/18) ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2469_246929


namespace NUMINAMATH_CALUDE_vector_problem_l2469_246948

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (c : ℝ), v.1 * w.2 = c * v.2 * w.1

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_problem (k m : ℝ) :
  (parallel (3 • a - b) (a + k • b) → k = -1/3) ∧
  (perpendicular a (m • a - b) → m = -4/5) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l2469_246948


namespace NUMINAMATH_CALUDE_no_zonk_probability_l2469_246958

theorem no_zonk_probability : 
  let num_tables : ℕ := 3
  let boxes_per_table : ℕ := 3
  let prob_no_zonk_per_table : ℚ := 2 / 3
  (prob_no_zonk_per_table ^ num_tables : ℚ) = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_no_zonk_probability_l2469_246958


namespace NUMINAMATH_CALUDE_refrigerator_temperature_l2469_246954

/-- Given an initial temperature, a temperature decrease rate, and elapsed time,
    calculate the final temperature inside a refrigerator. -/
def final_temperature (initial_temp : ℝ) (decrease_rate : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_temp - decrease_rate * elapsed_time

/-- Theorem stating that under the given conditions, the final temperature is -8°C. -/
theorem refrigerator_temperature : 
  final_temperature 12 5 4 = -8 := by
  sorry

#eval final_temperature 12 5 4

end NUMINAMATH_CALUDE_refrigerator_temperature_l2469_246954


namespace NUMINAMATH_CALUDE_staff_meeting_attendance_l2469_246939

theorem staff_meeting_attendance (total_doughnuts served_doughnuts left_doughnuts doughnuts_per_staff : ℕ) :
  served_doughnuts = 50 →
  doughnuts_per_staff = 2 →
  left_doughnuts = 12 →
  total_doughnuts = served_doughnuts - left_doughnuts →
  (total_doughnuts / doughnuts_per_staff : ℕ) = 19 :=
by sorry

end NUMINAMATH_CALUDE_staff_meeting_attendance_l2469_246939


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2469_246991

/-- The complex number z defined as (3 + i)i -/
def z : ℂ := (3 + Complex.I) * Complex.I

/-- Predicate to check if a complex number is in the second quadrant -/
def is_in_second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

/-- Theorem stating that z is in the second quadrant -/
theorem z_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2469_246991


namespace NUMINAMATH_CALUDE_xyz_value_l2469_246987

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 16)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 4) :
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l2469_246987


namespace NUMINAMATH_CALUDE_A_subset_B_A_equals_B_iff_l2469_246930

variable (a : ℝ)

def A : Set ℝ := {x | x^2 + a = x}
def B : Set ℝ := {x | (x^2 + a)^2 + a = x}

axiom A_nonempty : A a ≠ ∅

theorem A_subset_B : A a ⊆ B a := by sorry

theorem A_equals_B_iff : 
  A a = B a ↔ -3/4 ≤ a ∧ a ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_A_subset_B_A_equals_B_iff_l2469_246930


namespace NUMINAMATH_CALUDE_inequalities_theorem_l2469_246952

theorem inequalities_theorem (a b : ℝ) (m n : ℕ) 
    (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n) : 
  (a^n + b^n) * (a^m + b^m) ≤ 2 * (a^(m+n) + b^(m+n)) ∧ 
  (a + b) / 2 * (a^2 + b^2) / 2 * (a^3 + b^3) / 2 ≤ (a^6 + b^6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l2469_246952


namespace NUMINAMATH_CALUDE_difference_ones_zeros_235_l2469_246901

def base_10_to_base_2 (n : ℕ) : List ℕ :=
  sorry

def count_zeros (l : List ℕ) : ℕ :=
  sorry

def count_ones (l : List ℕ) : ℕ :=
  sorry

theorem difference_ones_zeros_235 :
  let binary_235 := base_10_to_base_2 235
  let w := count_ones binary_235
  let z := count_zeros binary_235
  w - z = 2 := by sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_235_l2469_246901


namespace NUMINAMATH_CALUDE_section_b_average_weight_l2469_246969

/-- Given a class with two sections A and B, prove that the average weight of section B is 35 kg. -/
theorem section_b_average_weight
  (students_a : ℕ)
  (students_b : ℕ)
  (total_students : ℕ)
  (avg_weight_a : ℝ)
  (avg_weight_total : ℝ)
  (h1 : students_a = 30)
  (h2 : students_b = 20)
  (h3 : total_students = students_a + students_b)
  (h4 : avg_weight_a = 40)
  (h5 : avg_weight_total = 38)
  : (total_students * avg_weight_total - students_a * avg_weight_a) / students_b = 35 := by
  sorry

#check section_b_average_weight

end NUMINAMATH_CALUDE_section_b_average_weight_l2469_246969


namespace NUMINAMATH_CALUDE_fifteenth_replacement_in_april_l2469_246992

def months : List String := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def monthsAfterFebruary (n : Nat) : Nat :=
  (months.indexOf "February" + n) % months.length

theorem fifteenth_replacement_in_april :
  months[monthsAfterFebruary 98] = "April" := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_replacement_in_april_l2469_246992


namespace NUMINAMATH_CALUDE_candy_cost_l2469_246997

/-- 
Given Chris's babysitting earnings and expenses, prove the cost of the candy assortment.
-/
theorem candy_cost 
  (video_game_cost : ℕ) 
  (hourly_rate : ℕ) 
  (hours_worked : ℕ) 
  (money_left : ℕ) 
  (h1 : video_game_cost = 60)
  (h2 : hourly_rate = 8)
  (h3 : hours_worked = 9)
  (h4 : money_left = 7) :
  video_game_cost + money_left + 5 = hourly_rate * hours_worked :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_l2469_246997


namespace NUMINAMATH_CALUDE_crew_average_weight_increase_l2469_246968

theorem crew_average_weight_increase (initial_average : ℝ) : 
  let initial_total_weight := 20 * initial_average
  let new_total_weight := initial_total_weight + (80 - 40)
  let new_average := new_total_weight / 20
  new_average - initial_average = 2 := by
sorry

end NUMINAMATH_CALUDE_crew_average_weight_increase_l2469_246968


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2469_246974

theorem rationalize_denominator :
  ∃ (A B C D : ℚ),
    (1 / (2 - Real.rpow 7 (1/3 : ℚ)) = Real.rpow A (1/3 : ℚ) + Real.rpow B (1/3 : ℚ) + Real.rpow C (1/3 : ℚ)) ∧
    (A = 4) ∧ (B = 2) ∧ (C = 7) ∧ (D = 1) ∧
    (A + B + C + D = 14) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2469_246974


namespace NUMINAMATH_CALUDE_trip_distance_proof_l2469_246963

/-- Represents the total distance of the trip in miles -/
def total_distance : ℝ := 90

/-- Represents the distance traveled on battery power in miles -/
def battery_distance : ℝ := 30

/-- Represents the gasoline consumption rate after battery power in gallons per mile -/
def gasoline_rate : ℝ := 0.03

/-- Represents the overall average fuel efficiency in miles per gallon -/
def average_efficiency : ℝ := 50

/-- Proves that the total trip distance is correct given the conditions -/
theorem trip_distance_proof :
  (total_distance / (gasoline_rate * (total_distance - battery_distance)) = average_efficiency) ∧
  (total_distance > battery_distance) :=
by sorry

end NUMINAMATH_CALUDE_trip_distance_proof_l2469_246963


namespace NUMINAMATH_CALUDE_five_x_minus_six_greater_than_one_l2469_246918

theorem five_x_minus_six_greater_than_one (x : ℝ) :
  (5 * x - 6 > 1) ↔ (5 * x - 6 > 1) :=
by sorry

end NUMINAMATH_CALUDE_five_x_minus_six_greater_than_one_l2469_246918


namespace NUMINAMATH_CALUDE_smaller_k_implies_smaller_certainty_l2469_246986

/-- Represents the observed value of the random variable K² -/
def observed_value (k : ℝ) : Prop := k ≥ 0

/-- Represents the certainty of the relationship between categorical variables -/
def relationship_certainty (c : ℝ) : Prop := c ≥ 0 ∧ c ≤ 1

/-- Theorem stating the relationship between observed K² value and relationship certainty -/
theorem smaller_k_implies_smaller_certainty 
  (X Y : Type) [Finite X] [Finite Y] 
  (k₁ k₂ c₁ c₂ : ℝ) 
  (hk₁ : observed_value k₁) 
  (hk₂ : observed_value k₂) 
  (hc₁ : relationship_certainty c₁) 
  (hc₂ : relationship_certainty c₂) :
  k₁ < k₂ → c₁ < c₂ :=
sorry

end NUMINAMATH_CALUDE_smaller_k_implies_smaller_certainty_l2469_246986


namespace NUMINAMATH_CALUDE_parabola_translation_l2469_246909

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally -/
def translate_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

/-- Translates a parabola vertically -/
def translate_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b
  , c := p.c + v }

theorem parabola_translation :
  let p1 : Parabola := { a := 2, b := 4, c := -3 }  -- y = 2(x+1)^2 - 3
  let p2 : Parabola := translate_vertical (translate_horizontal p1 1) 3
  p2 = { a := 2, b := 0, c := 0 }  -- y = 2x^2
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2469_246909


namespace NUMINAMATH_CALUDE_min_value_theorem_l2469_246947

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x + 1 / x^2 ≥ 4 ∧ 
  (3 * x + 1 / x^2 = 4 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2469_246947


namespace NUMINAMATH_CALUDE_floor_equality_iff_in_range_l2469_246908

theorem floor_equality_iff_in_range (x : ℝ) : 
  ⌊2 * x + 1/2⌋ = ⌊x + 3⌋ ↔ x ∈ Set.Ici (5/2) ∩ Set.Iio (7/2) := by
  sorry

end NUMINAMATH_CALUDE_floor_equality_iff_in_range_l2469_246908


namespace NUMINAMATH_CALUDE_family_movie_night_l2469_246927

/-- Proves the number of children in a family given ticket prices and payment information --/
theorem family_movie_night (regular_ticket_price : ℕ) 
                            (child_discount : ℕ)
                            (payment : ℕ)
                            (change : ℕ)
                            (num_adults : ℕ) :
  regular_ticket_price = 9 →
  child_discount = 2 →
  payment = 40 →
  change = 1 →
  num_adults = 2 →
  ∃ (num_children : ℕ),
    num_children = 3 ∧
    payment - change = 
      num_adults * regular_ticket_price + 
      num_children * (regular_ticket_price - child_discount) :=
by
  sorry


end NUMINAMATH_CALUDE_family_movie_night_l2469_246927
