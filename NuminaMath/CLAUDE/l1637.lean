import Mathlib

namespace problem_solution_l1637_163750

-- Define the propositions
def P (x : ℝ) : Prop := x^2 - 3*x + 2 = 0
def Q (x : ℝ) : Prop := x = 1
def R (x : ℝ) : Prop := x^2 + x + 1 < 0
def S (x : ℝ) : Prop := x > 2
def T (x : ℝ) : Prop := x^2 - 3*x + 2 > 0

-- Theorem statement
theorem problem_solution :
  (∀ x, (¬Q x → ¬P x) ↔ (P x → Q x)) ∧
  (¬(∃ x, R x) ↔ (∀ x, x^2 + x + 1 ≥ 0)) ∧
  ((∀ x, S x → T x) ∧ ¬(∀ x, T x → S x)) :=
by sorry

end problem_solution_l1637_163750


namespace equation_equivalence_l1637_163730

theorem equation_equivalence (x y : ℝ) :
  y^2 - 2*x*y + x^2 - 1 = 0 ↔ (y = x + 1 ∨ y = x - 1) :=
by sorry

end equation_equivalence_l1637_163730


namespace expression_simplification_l1637_163749

theorem expression_simplification (x : ℝ) :
  14 * (150 / 3 + 35 / 7 + 16 / 32 + x) = 777 + 14 * x := by
  sorry

end expression_simplification_l1637_163749


namespace thanksgiving_turkey_cost_johns_thanksgiving_cost_l1637_163740

/-- Calculates the total cost of John's Thanksgiving turkey surprise for his employees. -/
theorem thanksgiving_turkey_cost 
  (num_employees : ℕ) 
  (turkey_cost : ℝ) 
  (discount_rate : ℝ) 
  (discount_threshold : ℕ) 
  (delivery_flat_fee : ℝ) 
  (delivery_per_turkey : ℝ) 
  (sales_tax_rate : ℝ) : ℝ :=
  let discounted_turkey_cost := 
    if num_employees > discount_threshold
    then num_employees * turkey_cost * (1 - discount_rate)
    else num_employees * turkey_cost
  let delivery_cost := delivery_flat_fee + num_employees * delivery_per_turkey
  let total_before_tax := discounted_turkey_cost + delivery_cost
  let total_cost := total_before_tax * (1 + sales_tax_rate)
  total_cost

/-- The total cost for John's Thanksgiving surprise is $2,188.35. -/
theorem johns_thanksgiving_cost :
  thanksgiving_turkey_cost 85 25 0.15 50 50 2 0.08 = 2188.35 := by
  sorry

end thanksgiving_turkey_cost_johns_thanksgiving_cost_l1637_163740


namespace quadratic_form_equivalence_l1637_163727

theorem quadratic_form_equivalence : ∀ x : ℝ, x^2 + 6*x - 2 = (x + 3)^2 - 11 := by
  sorry

end quadratic_form_equivalence_l1637_163727


namespace parallel_vectors_x_value_l1637_163725

/-- Two 2D vectors are parallel if and only if their determinant is zero -/
def are_parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 - v 1 * w 0 = 0

theorem parallel_vectors_x_value :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-3, x]
  are_parallel a b → x = -6 := by
  sorry

end parallel_vectors_x_value_l1637_163725


namespace redToGreenGrapeRatio_l1637_163777

/-- Represents the composition of a fruit salad --/
structure FruitSalad where
  raspberries : ℕ
  greenGrapes : ℕ
  redGrapes : ℕ

/-- The properties of our specific fruit salad --/
def fruitSaladProperties (fs : FruitSalad) : Prop :=
  fs.raspberries = fs.greenGrapes - 5 ∧
  fs.raspberries + fs.greenGrapes + fs.redGrapes = 102 ∧
  fs.redGrapes = 67

/-- The theorem stating the ratio of red grapes to green grapes --/
theorem redToGreenGrapeRatio (fs : FruitSalad) 
  (h : fruitSaladProperties fs) : 
  fs.redGrapes * 20 = fs.greenGrapes * 67 := by
  sorry

#check redToGreenGrapeRatio

end redToGreenGrapeRatio_l1637_163777


namespace price_difference_l1637_163771

/-- Calculates the difference between the final retail price and the average price
    customers paid for the first 150 garments under a special pricing scheme. -/
theorem price_difference (original_price : ℝ) (first_increase : ℝ) (second_increase : ℝ)
  (special_rate1 : ℝ) (special_rate2 : ℝ) (special_quantity1 : ℕ) (special_quantity2 : ℕ)
  (h1 : original_price = 50)
  (h2 : first_increase = 0.3)
  (h3 : second_increase = 0.15)
  (h4 : special_rate1 = 0.7)
  (h5 : special_rate2 = 0.85)
  (h6 : special_quantity1 = 50)
  (h7 : special_quantity2 = 100) :
  let final_price := original_price * (1 + first_increase) * (1 + second_increase)
  let special_price1 := final_price * special_rate1
  let special_price2 := final_price * special_rate2
  let total_special_price := special_price1 * special_quantity1 + special_price2 * special_quantity2
  let avg_special_price := total_special_price / (special_quantity1 + special_quantity2)
  final_price - avg_special_price = 14.95 := by
sorry

end price_difference_l1637_163771


namespace abc_sum_equals_36_l1637_163729

theorem abc_sum_equals_36 (a b c : ℕ+) 
  (h : (4 : ℕ)^(a.val) * (5 : ℕ)^(b.val) * (6 : ℕ)^(c.val) = (8 : ℕ)^8 * (9 : ℕ)^9 * (10 : ℕ)^10) : 
  a.val + b.val + c.val = 36 := by
sorry

end abc_sum_equals_36_l1637_163729


namespace inequality_solution_range_l1637_163770

theorem inequality_solution_range (a : ℝ) : 
  (∀ x, (a - 1) * x < 1 ↔ x > 1 / (a - 1)) → a < 1 := by
  sorry

end inequality_solution_range_l1637_163770


namespace distance_origin_to_point_l1637_163790

/-- The distance between the origin (0, 0, 0) and the point (1, 2, 3) is √14 -/
theorem distance_origin_to_point :
  Real.sqrt ((1 : ℝ)^2 + 2^2 + 3^2) = Real.sqrt 14 := by
  sorry

end distance_origin_to_point_l1637_163790


namespace polynomial_expansion_l1637_163795

theorem polynomial_expansion (x : ℝ) :
  (3 * x^3 + 4 * x - 7) * (2 * x^4 - 3 * x^2 + 5) =
  6 * x^7 + 12 * x^5 - 9 * x^4 - 21 * x^3 - 11 * x + 35 := by
  sorry

end polynomial_expansion_l1637_163795


namespace problem_I3_1_l1637_163760

theorem problem_I3_1 (w x y z : ℝ) (hw : w > 0) 
  (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) : w = 4 := by
sorry


end problem_I3_1_l1637_163760


namespace uncovered_cells_less_than_mn_div_4_uncovered_cells_less_than_mn_div_5_l1637_163759

/-- Represents a rectangular board with dominoes -/
structure DominoBoard where
  m : ℕ  -- width of the board
  n : ℕ  -- height of the board
  uncovered_cells : ℕ  -- number of uncovered cells

/-- The number of uncovered cells is less than mn/4 -/
theorem uncovered_cells_less_than_mn_div_4 (board : DominoBoard) :
  board.uncovered_cells < (board.m * board.n) / 4 := by
  sorry

/-- The number of uncovered cells is less than mn/5 -/
theorem uncovered_cells_less_than_mn_div_5 (board : DominoBoard) :
  board.uncovered_cells < (board.m * board.n) / 5 := by
  sorry

end uncovered_cells_less_than_mn_div_4_uncovered_cells_less_than_mn_div_5_l1637_163759


namespace repeating_decimal_fraction_product_l1637_163716

theorem repeating_decimal_fraction_product : ∃ (n d : ℕ), 
  (n ≠ 0 ∧ d ≠ 0) ∧ 
  (∀ (k : ℕ), (0.027 + 0.027 / (1000 ^ k - 1) : ℚ) = n / d) ∧
  (∀ (n' d' : ℕ), n' ≠ 0 ∧ d' ≠ 0 → (∀ (k : ℕ), (0.027 + 0.027 / (1000 ^ k - 1) : ℚ) = n' / d') → n ≤ n' ∧ d ≤ d') ∧
  n * d = 37 :=
by sorry

end repeating_decimal_fraction_product_l1637_163716


namespace kevin_kangaroo_hops_l1637_163776

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Kevin's hopping problem -/
theorem kevin_kangaroo_hops :
  geometricSum (1/2) (1/2) 6 = 63/64 := by
  sorry

end kevin_kangaroo_hops_l1637_163776


namespace bicycle_problem_l1637_163796

/-- The time when two people traveling perpendicular to each other at different speeds are 100 miles apart -/
theorem bicycle_problem (jenny_speed mark_speed : ℝ) (h1 : jenny_speed = 10) (h2 : mark_speed = 15) :
  let t := (20 * Real.sqrt 13) / 13
  (t * jenny_speed) ^ 2 + (t * mark_speed) ^ 2 = 100 ^ 2 := by
  sorry

end bicycle_problem_l1637_163796


namespace polynomial_value_constraint_l1637_163793

theorem polynomial_value_constraint 
  (P : ℤ → ℤ) 
  (h_poly : ∀ x y : ℤ, (P x - P y) ∣ (x - y))
  (h_distinct : ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ P a = 2 ∧ P b = 2 ∧ P c = 2) :
  ∀ x : ℤ, P x ≠ 3 :=
by sorry

end polynomial_value_constraint_l1637_163793


namespace square_land_area_l1637_163761

/-- A square land plot with side length 40 units has an area of 1600 square units. -/
theorem square_land_area : 
  ∀ (side_length area : ℝ), 
  side_length = 40 → 
  area = side_length ^ 2 → 
  area = 1600 :=
by sorry

end square_land_area_l1637_163761


namespace equilateral_triangle_count_is_twenty_l1637_163784

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The hexagonal lattice with 19 points -/
def HexagonalLattice : Set LatticePoint :=
  sorry

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- The number of equilateral triangles in the lattice -/
def EquilateralTriangleCount : ℕ :=
  sorry

/-- Theorem stating that there are exactly 20 equilateral triangles in the lattice -/
theorem equilateral_triangle_count_is_twenty :
  EquilateralTriangleCount = 20 :=
sorry

end equilateral_triangle_count_is_twenty_l1637_163784


namespace people_in_room_l1637_163718

/-- Given a room with chairs and people, prove the total number of people -/
theorem people_in_room (chairs : ℕ) (people : ℕ) : 
  (3 : ℚ) / 5 * people = (4 : ℚ) / 5 * chairs ∧ 
  chairs - (4 : ℚ) / 5 * chairs = 8 →
  people = 54 := by sorry

end people_in_room_l1637_163718


namespace max_value_of_h_l1637_163703

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the function h as the sum of f and g
def h (x : ℝ) : ℝ := f x + g x

-- State the theorem
theorem max_value_of_h :
  (∀ x, -7 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ x, h x = 6) ∧ (∀ x, h x ≤ 6) := by
  sorry

end max_value_of_h_l1637_163703


namespace pentagon_area_fraction_is_five_eighths_l1637_163743

/-- Represents the tiling pattern of a large square -/
structure TilingPattern where
  total_divisions : Nat
  pentagon_count : Nat
  square_count : Nat

/-- Calculates the fraction of area covered by pentagons in the tiling pattern -/
def pentagon_area_fraction (pattern : TilingPattern) : Rat :=
  pattern.pentagon_count / pattern.total_divisions

/-- Theorem stating that the fraction of area covered by pentagons is 5/8 -/
theorem pentagon_area_fraction_is_five_eighths (pattern : TilingPattern) :
  pattern.total_divisions = 16 →
  pattern.pentagon_count = 10 →
  pattern.square_count = 6 →
  pentagon_area_fraction pattern = 5 / 8 := by
  sorry

end pentagon_area_fraction_is_five_eighths_l1637_163743


namespace unique_function_solution_l1637_163731

theorem unique_function_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (1 + x * y) - f (x + y) = f x * f y) ∧ 
  (f (-1) ≠ 0) → 
  (∀ x : ℝ, f x = x - 1) :=
sorry

end unique_function_solution_l1637_163731


namespace arithmetic_sequence_14th_term_l1637_163714

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 14th term of an arithmetic sequence given its 5th and 8th terms -/
theorem arithmetic_sequence_14th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_5 : a 5 = 6)
  (h_8 : a 8 = 15) :
  a 14 = 33 := by
sorry

end arithmetic_sequence_14th_term_l1637_163714


namespace smaller_angle_at_4_oclock_l1637_163719

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees between each hour on a clock face -/
def degrees_per_hour : ℕ := full_circle_degrees / clock_hours

/-- The number of hour spaces between 12 and 4 on a clock face -/
def spaces_12_to_4 : ℕ := 4

/-- The smaller angle formed by the hands of a clock at 4 o'clock -/
def clock_angle_at_4 : ℕ := spaces_12_to_4 * degrees_per_hour

theorem smaller_angle_at_4_oclock :
  clock_angle_at_4 = 120 :=
sorry

end smaller_angle_at_4_oclock_l1637_163719


namespace triangle_theorem_l1637_163717

-- Define the triangle ABC
structure Triangle (α : Type*) [Field α] where
  a : α
  b : α
  c : α

-- Define the existence of point P
def exists_unique_point (t : Triangle ℝ) : Prop :=
  t.c ≠ t.a ∧ t.a ≠ t.b

-- Define the angle BAC
noncomputable def angle_BAC (t : Triangle ℝ) : ℝ :=
  Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2 * t.b * t.c))

-- Main theorem
theorem triangle_theorem (t : Triangle ℝ) :
  (exists_unique_point t) ∧
  (∃ (P : ℝ × ℝ), (angle_BAC t < Real.pi / 3)) :=
sorry

end triangle_theorem_l1637_163717


namespace nancy_antacid_intake_l1637_163788

/-- Represents the number of antacids Nancy takes per day for different food types -/
structure AntacidIntake where
  indian : ℕ
  mexican : ℕ
  other : ℝ

/-- Represents Nancy's weekly food consumption -/
structure WeeklyConsumption where
  indian_days : ℕ
  mexican_days : ℕ

/-- Calculates Nancy's monthly antacid intake based on her eating habits -/
def monthly_intake (intake : AntacidIntake) (consumption : WeeklyConsumption) : ℝ :=
  4 * (intake.indian * consumption.indian_days + intake.mexican * consumption.mexican_days) +
  intake.other * (30 - 4 * (consumption.indian_days + consumption.mexican_days))

/-- Theorem stating Nancy's antacid intake for non-Indian and non-Mexican food days -/
theorem nancy_antacid_intake (intake : AntacidIntake) (consumption : WeeklyConsumption) :
  intake.indian = 3 →
  intake.mexican = 2 →
  consumption.indian_days = 3 →
  consumption.mexican_days = 2 →
  monthly_intake intake consumption = 60 →
  intake.other = 0.8 := by
  sorry

end nancy_antacid_intake_l1637_163788


namespace percentage_non_mutated_frogs_l1637_163705

def total_frogs : ℕ := 250
def extra_legs : ℕ := 32
def two_heads : ℕ := 21
def bright_red : ℕ := 16
def skin_abnormalities : ℕ := 12
def extra_eyes : ℕ := 7

theorem percentage_non_mutated_frogs :
  let mutated_frogs := extra_legs + two_heads + bright_red + skin_abnormalities + extra_eyes
  let non_mutated_frogs := total_frogs - mutated_frogs
  (non_mutated_frogs : ℚ) / total_frogs * 100 = 648 / 10 := by
  sorry

end percentage_non_mutated_frogs_l1637_163705


namespace count_numbers_satisfying_conditions_l1637_163728

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def starts_with_nine (n : ℕ) : Prop := ∃ (a b : ℕ), n = 900 + 90 * a + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

def digit_sum (n : ℕ) : ℕ := 
  (n / 100) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_three_digit n ∧
  starts_with_nine n ∧
  digit_sum n = 27 ∧
  Even n

theorem count_numbers_satisfying_conditions : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end count_numbers_satisfying_conditions_l1637_163728


namespace arithmetic_sequence_product_l1637_163701

-- Define the arithmetic sequence
def arithmetic_sequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

-- Define the increasing property
def increasing_sequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  arithmetic_sequence b 2 →
  increasing_sequence b →
  b 4 * b 5 = 15 →
  b 2 * b 7 = -9 :=
by sorry

end arithmetic_sequence_product_l1637_163701


namespace problem_solution_l1637_163742

noncomputable def f (m n x : ℝ) : ℝ := m * x + 1 / (n * x) + 1 / 2

theorem problem_solution (m n : ℝ) 
  (h1 : f m n 1 = 2) 
  (h2 : f m n 2 = 11 / 4) :
  (m = 1 ∧ n = 2) ∧ 
  (∀ x y, 1 ≤ x → x < y → f m n x < f m n y) ∧
  (∀ x : ℝ, f m n (1 + 2 * x^2) > f m n (x^2 - 2 * x + 4) ↔ x < -3 ∨ x > 1) :=
by sorry

end problem_solution_l1637_163742


namespace angle_measure_l1637_163704

theorem angle_measure (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 20) → x = 20 := by
  sorry

end angle_measure_l1637_163704


namespace fertilizer_production_l1637_163739

/-- 
Given:
- m: Initial production in the first quarter (in tons)
- x: Percentage increase in production each quarter (as a decimal)
- n: Production in the third quarter (in tons)

Prove that the production in the third quarter (n) is equal to the initial production (m) 
multiplied by (1 + x)^2.
-/
theorem fertilizer_production (m n : ℝ) (x : ℝ) (h_positive : 0 < x) : 
  m * (1 + x)^2 = n → True :=
by
  sorry

end fertilizer_production_l1637_163739


namespace area_of_polygon15_l1637_163792

/-- A 15-sided polygon on a 1 cm x 1 cm grid -/
def Polygon15 : List (ℤ × ℤ) :=
  [(1,3), (2,4), (2,5), (3,6), (4,6), (5,6), (6,5), (6,4), (5,3), (5,2), (4,1), (3,1), (2,2), (1,2), (1,3)]

/-- The area of a polygon given its vertices -/
def polygonArea (vertices : List (ℤ × ℤ)) : ℚ :=
  sorry

/-- Theorem stating that the area of the 15-sided polygon is 15 cm² -/
theorem area_of_polygon15 : polygonArea Polygon15 = 15 := by
  sorry

end area_of_polygon15_l1637_163792


namespace abs_gt_not_sufficient_nor_necessary_l1637_163772

theorem abs_gt_not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, abs x > abs y ∧ x ≤ y) ∧ 
  (∃ u v : ℝ, u > v ∧ abs u ≤ abs v) := by
sorry

end abs_gt_not_sufficient_nor_necessary_l1637_163772


namespace curve_self_intersection_l1637_163733

/-- A curve in the xy-plane defined by x = t^2 - 4 and y = t^3 - 6t + 4 for all real t. -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 4)

/-- The point where the curve crosses itself. -/
def self_intersection_point : ℝ × ℝ := (2, 4)

/-- Theorem stating that the curve crosses itself at the point (2, 4). -/
theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = self_intersection_point :=
sorry

end curve_self_intersection_l1637_163733


namespace cubic_inequality_solution_set_l1637_163737

-- Define the cubic polynomial function
def f (x : ℝ) : ℝ := -3 * x^3 + 5 * x^2 - 2 * x + 1

-- State the theorem
theorem cubic_inequality_solution_set :
  ∀ x : ℝ, f x > 0 ↔ (x > -1 ∧ x < 1/3) ∨ x > 1 :=
sorry

end cubic_inequality_solution_set_l1637_163737


namespace distance_to_y_axis_l1637_163738

/-- The distance from point A(-2, 1) to the y-axis is 2 -/
theorem distance_to_y_axis : 
  let A : ℝ × ℝ := (-2, 1)
  abs A.1 = 2 := by sorry

end distance_to_y_axis_l1637_163738


namespace arithmetic_series_sum_1_to_21_l1637_163756

/-- The sum of an arithmetic series with first term a, last term l, and n terms -/
def arithmetic_series_sum (a l n : ℕ) : ℕ := n * (a + l) / 2

/-- The number of terms in an arithmetic series with first term a, last term l, and common difference d -/
def arithmetic_series_length (a l d : ℕ) : ℕ := (l - a) / d + 1

theorem arithmetic_series_sum_1_to_21 : 
  let a := 1  -- first term
  let l := 21 -- last term
  let d := 2  -- common difference
  let n := arithmetic_series_length a l d
  arithmetic_series_sum a l n = 121 := by
sorry

end arithmetic_series_sum_1_to_21_l1637_163756


namespace max_scheduling_ways_after_15_games_l1637_163706

/-- Represents a chess tournament between schoolchildren and students. -/
structure ChessTournament where
  schoolchildren : Nat
  students : Nat
  total_games : Nat
  scheduled_games : Nat

/-- The maximum number of ways to schedule one game in the next round. -/
def max_scheduling_ways (tournament : ChessTournament) : Nat :=
  tournament.total_games - tournament.scheduled_games

/-- The theorem stating the maximum number of ways to schedule one game
    after uniquely scheduling 15 games in a tournament with 15 schoolchildren
    and 15 students. -/
theorem max_scheduling_ways_after_15_games
  (tournament : ChessTournament)
  (h1 : tournament.schoolchildren = 15)
  (h2 : tournament.students = 15)
  (h3 : tournament.total_games = tournament.schoolchildren * tournament.students)
  (h4 : tournament.scheduled_games = 15) :
  max_scheduling_ways tournament = 120 := by
  sorry


end max_scheduling_ways_after_15_games_l1637_163706


namespace solve_pocket_money_problem_l1637_163713

def pocket_money_problem (P : ℝ) : Prop :=
  let tteokbokki_cost : ℝ := P / 2
  let remaining_after_tteokbokki : ℝ := P - tteokbokki_cost
  let pencil_cost : ℝ := (3 / 8) * remaining_after_tteokbokki
  let final_remaining : ℝ := remaining_after_tteokbokki - pencil_cost
  (final_remaining = 2500) → (tteokbokki_cost = 4000)

theorem solve_pocket_money_problem :
  ∃ P : ℝ, pocket_money_problem P :=
sorry

end solve_pocket_money_problem_l1637_163713


namespace hash_2_3_4_l1637_163745

/-- The # operation defined on three real numbers -/
def hash (a b c : ℝ) : ℝ := (b + 1)^2 - 4*a*(c - 1)

/-- Theorem stating that #(2, 3, 4) = -8 -/
theorem hash_2_3_4 : hash 2 3 4 = -8 := by
  sorry

end hash_2_3_4_l1637_163745


namespace quadratic_function_properties_l1637_163709

/-- A quadratic function satisfying specific conditions -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The function g defined in terms of f -/
def g (a b c m : ℝ) (x : ℝ) : ℝ := f a b c x - 2 * m * x + 2

/-- The theorem stating the properties of f and g -/
theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c 0 = 0)
  (h2 : ∀ x, f a b c (x + 2) - f a b c x = 4 * x) :
  (∀ x, f a b c x = x^2 - 2 * x) ∧
  (∀ m,
    (m ≤ 0 → ∀ x ≥ 1, g a b c m x ≥ 1 - 2 * m) ∧
    (m > 0 → ∀ x ≥ 1, g a b c m x ≥ -m^2 - 2 * m + 1)) :=
by sorry

end quadratic_function_properties_l1637_163709


namespace triangle_angle_b_l1637_163768

/-- In a triangle ABC, given that a cos B - b cos A = c and C = π/5, prove that B = 3π/10 -/
theorem triangle_angle_b (a b c A B C : ℝ) : 
  a * Real.cos B - b * Real.cos A = c →
  C = π / 5 →
  B = 3 * π / 10 := by
  sorry

end triangle_angle_b_l1637_163768


namespace weight_plates_theorem_l1637_163755

/-- Calculates the effective weight of plates when lowered, considering technology and incline effects -/
def effectiveWeight (numPlates : ℕ) (plateWeight : ℝ) (techIncrease : ℝ) (inclineIncrease : ℝ) : ℝ :=
  let baseWeight := numPlates * plateWeight
  let withTech := baseWeight * (1 + techIncrease)
  withTech * (1 + inclineIncrease)

/-- Theorem: The effective weight of 10 plates of 30 pounds each, with 20% tech increase and 15% incline increase, is 414 pounds -/
theorem weight_plates_theorem :
  effectiveWeight 10 30 0.2 0.15 = 414 := by
  sorry


end weight_plates_theorem_l1637_163755


namespace quiz_probability_theorem_l1637_163715

/-- The number of questions in the quiz -/
def total_questions : ℕ := 30

/-- The number of answer choices for each question -/
def choices_per_question : ℕ := 6

/-- The number of questions Emily guesses randomly -/
def guessed_questions : ℕ := 5

/-- The probability of guessing a single question correctly -/
def prob_correct : ℚ := 1 / choices_per_question

/-- The probability of guessing a single question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of guessing at least two out of five questions correctly -/
def prob_at_least_two_correct : ℚ := 763 / 3888

theorem quiz_probability_theorem :
  (1 : ℚ) - (prob_incorrect ^ guessed_questions + 
    (guessed_questions : ℚ) * prob_correct * prob_incorrect ^ (guessed_questions - 1)) = 
  prob_at_least_two_correct :=
sorry

end quiz_probability_theorem_l1637_163715


namespace quadratic_roots_sum_bound_l1637_163751

theorem quadratic_roots_sum_bound (p : ℝ) (r₁ r₂ : ℝ) : 
  (∀ x : ℝ, x^2 + p*x + 12 = 0 ↔ x = r₁ ∨ x = r₂) →
  |r₁ + r₂| > 4 * Real.sqrt 3 := by
  sorry

end quadratic_roots_sum_bound_l1637_163751


namespace geometric_sequence_partial_sums_zero_property_l1637_163736

/-- A geometric sequence of real numbers -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Partial sums of a sequence -/
def partial_sums (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sums a n + a (n + 1)

/-- The main theorem -/
theorem geometric_sequence_partial_sums_zero_property
  (a : ℕ → ℝ) (h : geometric_sequence a) :
  (∀ n : ℕ, partial_sums a n ≠ 0) ∨
  (∀ m : ℕ, ∃ n : ℕ, n ≥ m ∧ partial_sums a n = 0) :=
sorry

end geometric_sequence_partial_sums_zero_property_l1637_163736


namespace watermelon_weight_calculation_l1637_163766

/-- The weight of a single watermelon in pounds -/
def watermelon_weight : ℝ := 23

/-- The price per pound of watermelon in dollars -/
def price_per_pound : ℝ := 2

/-- The number of watermelons sold -/
def num_watermelons : ℕ := 18

/-- The total revenue from selling the watermelons in dollars -/
def total_revenue : ℝ := 828

theorem watermelon_weight_calculation :
  watermelon_weight = total_revenue / (price_per_pound * num_watermelons) :=
by sorry

end watermelon_weight_calculation_l1637_163766


namespace exists_small_angle_between_diagonals_l1637_163726

/-- A convex dodecagon -/
structure ConvexDodecagon where
  -- We don't need to define the structure explicitly for this problem

/-- A diagonal in a polygon -/
structure Diagonal where
  -- We don't need to define the structure explicitly for this problem

/-- The angle between two diagonals -/
def angle_between_diagonals (d1 d2 : Diagonal) : ℝ := sorry

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The number of diagonals in a dodecagon -/
def dodecagon_diagonals : ℕ := 54

/-- The sum of angles in a plane around a point -/
def sum_of_angles : ℝ := 360

/-- Theorem: In any convex dodecagon, there exist two diagonals forming an angle not exceeding 3° -/
theorem exists_small_angle_between_diagonals (d : ConvexDodecagon) :
  ∃ (d1 d2 : Diagonal), angle_between_diagonals d1 d2 ≤ 3 := by sorry

end exists_small_angle_between_diagonals_l1637_163726


namespace triangle_rectangle_ratio_l1637_163764

theorem triangle_rectangle_ratio : 
  ∀ (t w l : ℝ),
  t > 0 → w > 0 → l > 0 →
  3 * t = 24 →           -- Perimeter of equilateral triangle
  2 * (w + l) = 24 →     -- Perimeter of rectangle
  l = 2 * w →            -- Length is twice the width
  t / w = 2 := by sorry

end triangle_rectangle_ratio_l1637_163764


namespace total_cost_with_tax_l1637_163721

def sandwich_price : ℚ := 4
def soda_price : ℚ := 3
def tax_rate : ℚ := 0.1
def sandwich_quantity : ℕ := 7
def soda_quantity : ℕ := 6

theorem total_cost_with_tax :
  let subtotal := sandwich_price * sandwich_quantity + soda_price * soda_quantity
  let tax := subtotal * tax_rate
  let total := subtotal + tax
  total = 50.6 := by sorry

end total_cost_with_tax_l1637_163721


namespace dentists_age_l1637_163775

theorem dentists_age : ∃ (x : ℕ), 
  (x - 8) / 6 = (x + 8) / 10 ∧ x = 32 := by
  sorry

end dentists_age_l1637_163775


namespace equation_solutions_l1637_163708

theorem equation_solutions :
  let y₁ : ℝ := (3 + Real.sqrt 15) / 2
  let y₂ : ℝ := (3 - Real.sqrt 15) / 2
  (3 - y₁)^2 + y₁^2 = 12 ∧ (3 - y₂)^2 + y₂^2 = 12 := by
  sorry

end equation_solutions_l1637_163708


namespace relationship_between_m_and_a_l1637_163702

theorem relationship_between_m_and_a (m : ℕ) (a : ℝ) 
  (h1 : m > 0) (h2 : a > 0) :
  ((∀ n : ℕ, n > m → (1 : ℝ) / n < a) ∧ 
   (∀ n : ℕ, 0 < n ∧ n ≤ m → (1 : ℝ) / n ≥ a)) ↔ 
  m = ⌊(1 : ℝ) / a⌋ := by
  sorry

end relationship_between_m_and_a_l1637_163702


namespace min_value_reciprocal_sum_l1637_163786

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 9 / b ≥ 16 := by
  sorry

end min_value_reciprocal_sum_l1637_163786


namespace rectangle_sides_l1637_163781

theorem rectangle_sides (S d : ℝ) (h1 : S > 0) (h2 : d ≥ 0) :
  let a := Real.sqrt (S + d^2 / 4) + d / 2
  let b := Real.sqrt (S + d^2 / 4) - d / 2
  a * b = S ∧ a - b = d ∧ a > 0 ∧ b > 0 := by sorry

end rectangle_sides_l1637_163781


namespace real_part_of_z_l1637_163765

theorem real_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 2) : z.re = 1 := by
  sorry

end real_part_of_z_l1637_163765


namespace rectangle_difference_l1637_163787

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  breadth : ℕ

/-- The perimeter of the rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.breadth)

/-- The area of the rectangle -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.breadth

/-- The difference between length and breadth -/
def Rectangle.difference (r : Rectangle) : ℕ := r.length - r.breadth

theorem rectangle_difference (r : Rectangle) :
  r.perimeter = 266 ∧ r.area = 4290 → r.difference = 23 := by
  sorry

#eval Rectangle.difference { length := 78, breadth := 55 }

end rectangle_difference_l1637_163787


namespace four_point_circle_theorem_l1637_163773

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define what it means for three points to be collinear
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

-- Define what it means for a point to be on a circle
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Define what it means for a point to be inside a circle
def insideCircle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 < c.radius^2

-- The main theorem
theorem four_point_circle_theorem (A B C D : Point) 
  (h : ¬collinear A B C ∧ ¬collinear A B D ∧ ¬collinear A C D ∧ ¬collinear B C D) :
  ∃ (c : Circle), 
    (onCircle A c ∧ onCircle B c ∧ onCircle C c ∧ (onCircle D c ∨ insideCircle D c)) ∨
    (onCircle A c ∧ onCircle B c ∧ onCircle D c ∧ (onCircle C c ∨ insideCircle C c)) ∨
    (onCircle A c ∧ onCircle C c ∧ onCircle D c ∧ (onCircle B c ∨ insideCircle B c)) ∨
    (onCircle B c ∧ onCircle C c ∧ onCircle D c ∧ (onCircle A c ∨ insideCircle A c)) :=
  sorry

end four_point_circle_theorem_l1637_163773


namespace crocus_bulbs_count_l1637_163754

/-- Represents the number of crocus bulbs that can be bought given the constraints -/
def crocus_bulbs : ℕ := 22

/-- Represents the number of daffodil bulbs that can be bought given the constraints -/
def daffodil_bulbs : ℕ := 55 - crocus_bulbs

/-- The total number of bulbs -/
def total_bulbs : ℕ := 55

/-- The cost of a single crocus bulb in cents -/
def crocus_cost : ℕ := 35

/-- The cost of a single daffodil bulb in cents -/
def daffodil_cost : ℕ := 65

/-- The total budget in cents -/
def total_budget : ℕ := 2915

theorem crocus_bulbs_count : 
  crocus_bulbs = 22 ∧ 
  crocus_bulbs + daffodil_bulbs = total_bulbs ∧ 
  crocus_bulbs * crocus_cost + daffodil_bulbs * daffodil_cost = total_budget := by
  sorry

end crocus_bulbs_count_l1637_163754


namespace equation_equivalence_l1637_163710

theorem equation_equivalence :
  ∀ (x y : ℝ), (2 * x - y = 3) ↔ (y = 2 * x - 3) := by
  sorry

end equation_equivalence_l1637_163710


namespace remaining_length_is_21_l1637_163762

/-- A polygon with perpendicular adjacent sides -/
structure PerpendicularPolygon where
  left : ℝ
  top : ℝ
  right : ℝ
  bottom_removed : List ℝ

/-- The total length of remaining segments after removal -/
def remaining_length (p : PerpendicularPolygon) : ℝ :=
  p.left + p.top + p.right

theorem remaining_length_is_21 (p : PerpendicularPolygon)
  (h1 : p.left = 10)
  (h2 : p.top = 3)
  (h3 : p.right = 8)
  (h4 : p.bottom_removed = [2, 1, 2]) :
  remaining_length p = 21 := by
  sorry

end remaining_length_is_21_l1637_163762


namespace cubic_equation_integer_solution_l1637_163769

theorem cubic_equation_integer_solution (m : ℤ) :
  (∃ x : ℤ, x^3 - m*x^2 + m*x - (m^2 + 1) = 0) ↔ (m = -3 ∨ m = 0) :=
by sorry

end cubic_equation_integer_solution_l1637_163769


namespace bianca_deleted_files_l1637_163767

/-- The number of pictures Bianca deleted -/
def pictures : ℕ := 5

/-- The number of songs Bianca deleted -/
def songs : ℕ := 12

/-- The number of text files Bianca deleted -/
def text_files : ℕ := 10

/-- The number of video files Bianca deleted -/
def video_files : ℕ := 6

/-- The total number of files Bianca deleted -/
def total_files : ℕ := pictures + songs + text_files + video_files

theorem bianca_deleted_files : total_files = 33 := by
  sorry

end bianca_deleted_files_l1637_163767


namespace last_day_third_quarter_common_year_l1637_163789

/-- Represents a day in a month -/
structure DayInMonth where
  month : Nat
  day : Nat

/-- Definition of a common year -/
def isCommonYear (totalDays : Nat) : Prop := totalDays = 365

/-- Definition of the third quarter -/
def isInThirdQuarter (d : DayInMonth) : Prop :=
  d.month ∈ [7, 8, 9]

/-- The last day of the third quarter in a common year -/
theorem last_day_third_quarter_common_year (totalDays : Nat) 
  (h : isCommonYear totalDays) :
  ∃ (d : DayInMonth), 
    isInThirdQuarter d ∧ 
    d.month = 9 ∧ 
    d.day = 30 ∧ 
    (∀ (d' : DayInMonth), isInThirdQuarter d' → d'.month < d.month ∨ (d'.month = d.month ∧ d'.day ≤ d.day)) :=
sorry

end last_day_third_quarter_common_year_l1637_163789


namespace min_value_of_complex_expression_l1637_163799

open Complex

theorem min_value_of_complex_expression (z : ℂ) (h : abs (z - (3 - 3*I)) = 3) :
  ∃ (min : ℝ), min = 100 ∧ ∀ (w : ℂ), abs (w - (3 - 3*I)) = 3 → 
    abs (w - (2 + 2*I))^2 + abs (w - (6 - 6*I))^2 ≥ min :=
by sorry

end min_value_of_complex_expression_l1637_163799


namespace valid_seating_arrangements_count_l1637_163779

-- Define the people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 5 → Person

-- Define the condition that two people cannot sit next to each other
def CannotSitNextTo (p1 p2 : Person) (arrangement : SeatingArrangement) : Prop :=
  ∀ i : Fin 4, arrangement i ≠ p1 ∨ arrangement (Fin.succ i) ≠ p2

-- Define a valid seating arrangement
def ValidArrangement (arrangement : SeatingArrangement) : Prop :=
  (CannotSitNextTo Person.Alice Person.Bob arrangement) ∧
  (CannotSitNextTo Person.Alice Person.Carla arrangement) ∧
  (CannotSitNextTo Person.Carla Person.Bob arrangement) ∧
  (CannotSitNextTo Person.Carla Person.Derek arrangement) ∧
  (CannotSitNextTo Person.Derek Person.Eric arrangement)

-- The main theorem
theorem valid_seating_arrangements_count :
  ∃ arrangements : Finset SeatingArrangement,
    (∀ arr ∈ arrangements, ValidArrangement arr) ∧
    (∀ arr, ValidArrangement arr → arr ∈ arrangements) ∧
    arrangements.card = 12 :=
sorry

end valid_seating_arrangements_count_l1637_163779


namespace divisor_sum_theorem_l1637_163748

def sum_of_divisors (i j k : ℕ) : ℕ :=
  (2^(i+1) - 1) * (3^(j+1) - 1) * (5^(k+1) - 1) / ((2-1) * (3-1) * (5-1))

theorem divisor_sum_theorem (i j k : ℕ) :
  sum_of_divisors i j k = 1200 → i + j + k = 7 := by
  sorry

end divisor_sum_theorem_l1637_163748


namespace purely_imaginary_implies_m_eq_three_l1637_163734

/-- A complex number z is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- Given that z = (m^2 - 2m - 3) + (m + 1)i is a purely imaginary number, m = 3. -/
theorem purely_imaginary_implies_m_eq_three (m : ℝ) :
  is_purely_imaginary ((m^2 - 2*m - 3 : ℝ) + (m + 1)*I) → m = 3 :=
by sorry

end purely_imaginary_implies_m_eq_three_l1637_163734


namespace age_ratio_proof_l1637_163753

/-- Proves that the ratio of Rommel's age to Tim's age is 3:1 -/
theorem age_ratio_proof (tim_age : ℕ) (rommel_age : ℕ) (jenny_age : ℕ) : 
  tim_age = 5 →
  jenny_age = rommel_age + 2 →
  jenny_age = tim_age + 12 →
  rommel_age / tim_age = 3 := by
  sorry

#check age_ratio_proof

end age_ratio_proof_l1637_163753


namespace fourth_vertex_of_rectangle_l1637_163774

/-- A rectangle in a 2D plane --/
structure Rectangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Check if four points form a rectangle --/
def isRectangle (r : Rectangle) : Prop :=
  let (x1, y1) := r.v1
  let (x2, y2) := r.v2
  let (x3, y3) := r.v3
  let (x4, y4) := r.v4
  (x1 = x3 ∧ x2 = x4 ∧ y1 = y2 ∧ y3 = y4) ∨
  (x1 = x2 ∧ x3 = x4 ∧ y1 = y3 ∧ y2 = y4)

/-- The main theorem --/
theorem fourth_vertex_of_rectangle :
  ∀ (r : Rectangle),
    r.v1 = (1, 1) →
    r.v2 = (5, 1) →
    r.v3 = (1, 7) →
    isRectangle r →
    r.v4 = (5, 7) := by
  sorry


end fourth_vertex_of_rectangle_l1637_163774


namespace sample_size_theorem_l1637_163763

theorem sample_size_theorem (frequency_sum : ℝ) (frequency_ratio : ℝ) 
  (h1 : frequency_sum = 20) 
  (h2 : frequency_ratio = 0.4) : 
  frequency_sum / frequency_ratio = 50 := by
  sorry

end sample_size_theorem_l1637_163763


namespace fraction_equality_l1637_163798

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) : 
  (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := by
  sorry

end fraction_equality_l1637_163798


namespace arithmetic_mean_geq_geometric_mean_fourth_root_l1637_163780

theorem arithmetic_mean_geq_geometric_mean_fourth_root
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) / 4 ≥ (a * b * c * d) ^ (1/4) :=
sorry

end arithmetic_mean_geq_geometric_mean_fourth_root_l1637_163780


namespace overtime_calculation_l1637_163782

/-- A worker's pay structure and hours worked --/
structure WorkerPay where
  ordinary_rate : ℚ  -- Rate for ordinary time in cents per hour
  overtime_rate : ℚ  -- Rate for overtime in cents per hour
  total_pay : ℚ      -- Total pay for the week in cents
  total_hours : ℕ    -- Total hours worked in the week

/-- Calculate the number of overtime hours --/
def overtime_hours (w : WorkerPay) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the overtime hours are 8 --/
theorem overtime_calculation (w : WorkerPay) 
  (h1 : w.ordinary_rate = 60)
  (h2 : w.overtime_rate = 90)
  (h3 : w.total_pay = 3240)
  (h4 : w.total_hours = 50) :
  overtime_hours w = 8 := by
  sorry

end overtime_calculation_l1637_163782


namespace modulus_of_complex_expression_l1637_163785

theorem modulus_of_complex_expression :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I
  Complex.abs z = 1 := by
  sorry

end modulus_of_complex_expression_l1637_163785


namespace parallel_vectors_l1637_163722

/-- Given vectors a and b, if a is parallel to (a + b), then the second component of b is -3. -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : ∃ (k : ℝ), a = k • (a + b)) : 
  a.1 = -1 ∧ a.2 = 1 ∧ b.1 = 3 → b.2 = -3 := by
  sorry

end parallel_vectors_l1637_163722


namespace ariel_birth_year_l1637_163747

/-- Calculates the birth year of a person given their fencing start year, years of fencing, and current age. -/
def birth_year (fencing_start_year : ℕ) (years_fencing : ℕ) (current_age : ℕ) : ℕ :=
  fencing_start_year - (current_age - years_fencing)

/-- Proves that Ariel's birth year is 1992 given the provided conditions. -/
theorem ariel_birth_year :
  let fencing_start_year : ℕ := 2006
  let years_fencing : ℕ := 16
  let current_age : ℕ := 30
  birth_year fencing_start_year years_fencing current_age = 1992 := by
  sorry

#eval birth_year 2006 16 30

end ariel_birth_year_l1637_163747


namespace cindy_calculation_l1637_163791

theorem cindy_calculation (x : ℝ) : (x - 7) / 5 = 15 → (x - 5) / 7 = 11 := by
  sorry

end cindy_calculation_l1637_163791


namespace fraction_sum_equals_decimal_l1637_163732

theorem fraction_sum_equals_decimal : 
  2/10 - 5/100 + 3/1000 + 8/10000 = 0.1538 := by sorry

end fraction_sum_equals_decimal_l1637_163732


namespace line_A2A3_tangent_to_circle_M_l1637_163744

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

-- Define a line tangent to the circle M
def line_tangent_to_circle_M (A B : ℝ × ℝ) : Prop :=
  let d := abs ((B.2 - A.2) * 2 - (B.1 - A.1) * 0 + (A.1 * B.2 - B.1 * A.2)) /
            Real.sqrt ((B.2 - A.2)^2 + (B.1 - A.1)^2)
  d = 1

-- State the theorem
theorem line_A2A3_tangent_to_circle_M 
  (A₁ A₂ A₃ : ℝ × ℝ)
  (h₁ : point_on_parabola A₁)
  (h₂ : point_on_parabola A₂)
  (h₃ : point_on_parabola A₃)
  (h₄ : line_tangent_to_circle_M A₁ A₂)
  (h₅ : line_tangent_to_circle_M A₁ A₃) :
  line_tangent_to_circle_M A₂ A₃ := by sorry

end line_A2A3_tangent_to_circle_M_l1637_163744


namespace trapezoid_side_length_l1637_163752

-- Define the trapezoid ABCD
structure Trapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (BC : ℝ)

-- Define the properties of the trapezoid
def is_valid_trapezoid (t : Trapezoid) : Prop :=
  t.AB = 10 ∧ 
  t.CD = 2 * t.AB ∧ 
  t.AD = t.BC ∧ 
  t.AB + t.BC + t.CD + t.AD = 42

-- Theorem statement
theorem trapezoid_side_length 
  (t : Trapezoid) 
  (h : is_valid_trapezoid t) : 
  t.AD = 6 := by
  sorry

end trapezoid_side_length_l1637_163752


namespace all_statements_equivalent_l1637_163700

-- Define the propositions
variable (P Q : Prop)

-- Define the equivalence of all statements
theorem all_statements_equivalent :
  (P ↔ Q) ↔ (P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end all_statements_equivalent_l1637_163700


namespace set_intersection_union_problem_l1637_163735

theorem set_intersection_union_problem (a b : ℝ) :
  let M : Set ℝ := {3, 2^a}
  let N : Set ℝ := {a, b}
  (M ∩ N = {2}) → (M ∪ N = {1, 2, 3}) := by
  sorry

end set_intersection_union_problem_l1637_163735


namespace largest_five_digit_congruent_to_19_mod_26_l1637_163724

theorem largest_five_digit_congruent_to_19_mod_26 : ∃ (n : ℕ), n = 99989 ∧ 
  (∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 26 = 19 → m ≤ n) ∧ 
  10000 ≤ n ∧ n ≤ 99999 ∧ n % 26 = 19 :=
by sorry

end largest_five_digit_congruent_to_19_mod_26_l1637_163724


namespace pipe_C_rate_l1637_163794

-- Define the rates of the pipes
def rate_A : ℚ := 1 / 60
def rate_B : ℚ := 1 / 80
def rate_combined : ℚ := 1 / 40

-- Define the rate of pipe C
def rate_C : ℚ := rate_A + rate_B - rate_combined

-- Theorem statement
theorem pipe_C_rate : rate_C = 1 / 240 := by
  sorry

end pipe_C_rate_l1637_163794


namespace percentage_of_men_l1637_163778

/-- Represents the composition of employees in a company -/
structure Company where
  men : ℝ
  women : ℝ
  men_french : ℝ
  women_french : ℝ

/-- The company satisfies the given conditions -/
def valid_company (c : Company) : Prop :=
  c.men + c.women = 100 ∧
  c.men_french = 0.6 * c.men ∧
  c.women_french = 0.35 * c.women ∧
  c.men_french + c.women_french = 50

/-- The theorem stating that 60% of the company employees are men -/
theorem percentage_of_men (c : Company) (h : valid_company c) : c.men = 60 := by
  sorry

end percentage_of_men_l1637_163778


namespace total_books_l1637_163797

theorem total_books (tim_books sam_books alice_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : alice_books = 38) :
  tim_books + sam_books + alice_books = 134 := by
  sorry

end total_books_l1637_163797


namespace chocolate_boxes_l1637_163757

theorem chocolate_boxes (pieces_per_box : ℕ) (total_pieces : ℕ) (h1 : pieces_per_box = 500) (h2 : total_pieces = 3000) :
  total_pieces / pieces_per_box = 6 := by
  sorry

end chocolate_boxes_l1637_163757


namespace lemonade_glasses_served_l1637_163741

/-- The number of glasses of lemonade that can be served from a given number of pitchers. -/
def glasses_served (glasses_per_pitcher : ℕ) (num_pitchers : ℕ) : ℕ :=
  glasses_per_pitcher * num_pitchers

/-- Theorem stating that 6 pitchers of lemonade, each serving 5 glasses, can serve 30 glasses in total. -/
theorem lemonade_glasses_served :
  glasses_served 5 6 = 30 := by
  sorry

end lemonade_glasses_served_l1637_163741


namespace two_week_training_hours_l1637_163720

/-- Calculates the total training hours for two weeks given daily maximum hours -/
def totalTrainingHours (week1MaxHours : ℕ) (week2MaxHours : ℕ) : ℕ :=
  7 * week1MaxHours + 7 * week2MaxHours

/-- Proves that training for 2 hours max per day in week 1 and 3 hours max per day in week 2 results in 35 total hours -/
theorem two_week_training_hours : totalTrainingHours 2 3 = 35 := by
  sorry

#eval totalTrainingHours 2 3

end two_week_training_hours_l1637_163720


namespace thirteenth_term_is_15_l1637_163723

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  fifth_term : a 5 = 1
  sum_property : a 8 + a 10 = 16

/-- The 13th term of the arithmetic sequence is 15 -/
theorem thirteenth_term_is_15 (seq : ArithmeticSequence) : seq.a 13 = 15 := by
  sorry

end thirteenth_term_is_15_l1637_163723


namespace small_cube_edge_length_l1637_163746

/-- Given a cube with volume 1000 cm³, prove that cutting off 8 small cubes
    of equal size from its corners, resulting in a remaining volume of 488 cm³,
    yields small cubes with edge length 4 cm. -/
theorem small_cube_edge_length 
  (initial_volume : ℝ) 
  (remaining_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (h_initial : initial_volume = 1000)
  (h_remaining : remaining_volume = 488)
  (h_num_cubes : num_small_cubes = 8) :
  ∃ (edge_length : ℝ), 
    edge_length = 4 ∧ 
    initial_volume - num_small_cubes * edge_length ^ 3 = remaining_volume :=
by sorry

end small_cube_edge_length_l1637_163746


namespace greatest_integer_radius_l1637_163783

theorem greatest_integer_radius (A : ℝ) (h : A < 100 * Real.pi) :
  ∃ (r : ℕ), r^2 * Real.pi ≤ A ∧ ∀ (s : ℕ), s^2 * Real.pi ≤ A → s ≤ r ∧ r = 9 :=
sorry

end greatest_integer_radius_l1637_163783


namespace cyclic_inequality_l1637_163758

theorem cyclic_inequality (x₁ x₂ x₃ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  x₂^2 / x₁ + x₃^2 / x₂ + x₁^2 / x₃ ≥ 1 := by
sorry

end cyclic_inequality_l1637_163758


namespace least_number_of_cans_l1637_163711

theorem least_number_of_cans (maaza pepsi sprite : ℕ) 
  (h_maaza : maaza = 10)
  (h_pepsi : pepsi = 144)
  (h_sprite : sprite = 368) :
  let gcd_all := Nat.gcd maaza (Nat.gcd pepsi sprite)
  ∃ (can_size : ℕ), 
    can_size = gcd_all ∧ 
    can_size > 0 ∧
    maaza % can_size = 0 ∧ 
    pepsi % can_size = 0 ∧ 
    sprite % can_size = 0 ∧
    (maaza / can_size + pepsi / can_size + sprite / can_size) = 261 :=
by sorry

end least_number_of_cans_l1637_163711


namespace water_tank_capacity_l1637_163707

theorem water_tank_capacity : ∃ (C : ℝ), 
  (C > 0) ∧ (0.40 * C - 0.25 * C = 36) ∧ (C = 240) := by
  sorry

end water_tank_capacity_l1637_163707


namespace smallest_advantageous_discount_l1637_163712

def is_more_advantageous (n : ℕ) : Prop :=
  (1 - n / 100 : ℝ) < (1 - 0.2)^2 ∧
  (1 - n / 100 : ℝ) < (1 - 0.15)^3 ∧
  (1 - n / 100 : ℝ) < (1 - 0.3) * (1 - 0.1)

theorem smallest_advantageous_discount : 
  (∀ m : ℕ, m < 39 → ¬(is_more_advantageous m)) ∧ 
  is_more_advantageous 39 := by
  sorry

end smallest_advantageous_discount_l1637_163712
