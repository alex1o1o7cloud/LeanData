import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l620_62051

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂ ∧ b₁ ≠ b₂

/-- The slope-intercept form of a line ax + by + c = 0 is y = -(a/b)x - (c/b) -/
def slope_intercept_form (a b c : ℝ) (hb : b ≠ 0) :
  ∀ x y : ℝ, a * x + b * y + c = 0 ↔ y = -(a/b) * x - (c/b) :=
  sorry

theorem parallel_lines_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, a * x + 2 * y - 1 = 0 ↔ 8 * x + a * y + (2 - a) = 0) →
  a = -4 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l620_62051


namespace NUMINAMATH_CALUDE_infinite_sum_equality_l620_62066

/-- For positive real numbers p and q where p > 2q, the infinite sum
    1/(pq) + 1/(p(3p-2q)) + 1/((3p-2q)(5p-4q)) + 1/((5p-4q)(7p-6q)) + ...
    is equal to 1/((p-2q)p). -/
theorem infinite_sum_equality (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p > 2*q) :
  let f : ℕ → ℝ := λ n => 1 / ((2*n - 1)*p - (2*n - 2)*q) / ((2*n + 1)*p - 2*n*q)
  ∑' n, f n = 1 / ((p - 2*q) * p) := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equality_l620_62066


namespace NUMINAMATH_CALUDE_correct_calculation_l620_62081

theorem correct_calculation (x : ℤ) : x - 32 = 33 → x + 32 = 97 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l620_62081


namespace NUMINAMATH_CALUDE_sequence_properties_l620_62071

theorem sequence_properties :
  (∀ n m : ℕ, (2 * n)^2 + 1 ≠ 3 * m^2) ∧
  (∀ p q : ℕ, p^2 + 1 ≠ 7 * q^2) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l620_62071


namespace NUMINAMATH_CALUDE_sequence_pattern_l620_62065

def sequence_sum (a b : ℕ) : ℕ := a + b - 1

theorem sequence_pattern : 
  (sequence_sum 6 7 = 12) ∧
  (sequence_sum 8 9 = 16) ∧
  (sequence_sum 5 6 = 10) ∧
  (sequence_sum 7 8 = 14) →
  sequence_sum 3 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_pattern_l620_62065


namespace NUMINAMATH_CALUDE_barry_sitting_time_l620_62019

/-- Calculates the sitting time between turns for Barry's head-standing routine -/
def calculate_sitting_time (total_time minutes_per_turn number_of_turns : ℕ) : ℕ :=
  let total_standing_time := minutes_per_turn * number_of_turns
  let total_sitting_time := total_time - total_standing_time
  let number_of_breaks := number_of_turns - 1
  (total_sitting_time + number_of_breaks - 1) / number_of_breaks

theorem barry_sitting_time :
  let total_time : ℕ := 120  -- 2 hours in minutes
  let minutes_per_turn : ℕ := 10
  let number_of_turns : ℕ := 8
  calculate_sitting_time total_time minutes_per_turn number_of_turns = 6 := by
  sorry

end NUMINAMATH_CALUDE_barry_sitting_time_l620_62019


namespace NUMINAMATH_CALUDE_seating_arrangement_theorem_l620_62031

structure SeatingArrangement where
  rows_6 : Nat
  rows_8 : Nat
  rows_9 : Nat
  total_people : Nat
  max_rows : Nat

def is_valid (s : SeatingArrangement) : Prop :=
  s.rows_6 * 6 + s.rows_8 * 8 + s.rows_9 * 9 = s.total_people ∧
  s.rows_6 + s.rows_8 + s.rows_9 ≤ s.max_rows

theorem seating_arrangement_theorem :
  ∃ (s : SeatingArrangement),
    s.total_people = 58 ∧
    s.max_rows = 7 ∧
    is_valid s ∧
    s.rows_9 = 4 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangement_theorem_l620_62031


namespace NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l620_62045

/-- The function f(x) defined as |x - a| + |x - 1| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

/-- Theorem for part (I) of the problem -/
theorem solution_part_i :
  let a : ℝ := 2
  {x : ℝ | f a x < 4} = {x : ℝ | -1/2 < x ∧ x < 7/2} := by sorry

/-- Theorem for part (II) of the problem -/
theorem solution_part_ii :
  {a : ℝ | ∀ x, f a x ≥ 2} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} := by sorry

end NUMINAMATH_CALUDE_solution_part_i_solution_part_ii_l620_62045


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l620_62090

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular
  (l m : Line) (α : Plane)
  (h1 : l ≠ m)
  (h2 : perpendicular l α)
  (h3 : parallel l m) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l620_62090


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l620_62047

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define the property we want to prove
def is_smallest_divisible_by_5_and_6 (n : ℕ) : Prop :=
  is_perfect_square n ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧
  ∀ m : ℕ, m < n → ¬(is_perfect_square m ∧ m % 5 = 0 ∧ m % 6 = 0)

-- State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 :
  is_smallest_divisible_by_5_and_6 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l620_62047


namespace NUMINAMATH_CALUDE_victors_decks_count_l620_62024

/-- The number of decks Victor bought -/
def victors_decks : ℕ := 6

/-- The cost of each trick deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Victor's friend bought -/
def friends_decks : ℕ := 2

/-- The total amount spent by Victor and his friend in dollars -/
def total_spent : ℕ := 64

theorem victors_decks_count :
  victors_decks * deck_cost + friends_decks * deck_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_victors_decks_count_l620_62024


namespace NUMINAMATH_CALUDE_rebecca_eggs_l620_62096

/-- The number of eggs Rebecca has -/
def num_eggs : ℕ := 3 * 6

/-- The number of groups Rebecca will create -/
def num_groups : ℕ := 3

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 6

/-- Theorem stating that Rebecca has 18 eggs -/
theorem rebecca_eggs : num_eggs = 18 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_l620_62096


namespace NUMINAMATH_CALUDE_liz_shopping_cost_l620_62087

def problem (recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount : ℝ) : Prop :=
  let total_cost := 
    recipe_book_cost + 
    baking_dish_cost + 
    (5 * ingredient_cost) + 
    apron_cost + 
    mixer_cost + 
    measuring_cups_cost + 
    (4 * spice_cost) - 
    discount
  total_cost = 84.5 ∧
  recipe_book_cost = 6 ∧
  baking_dish_cost = 2 * recipe_book_cost ∧
  ingredient_cost = 3 ∧
  apron_cost = recipe_book_cost + 1 ∧
  mixer_cost = 3 * baking_dish_cost ∧
  measuring_cups_cost = apron_cost / 2 ∧
  spice_cost = 2 ∧
  discount = 3

theorem liz_shopping_cost : ∃ (recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount : ℝ),
  problem recipe_book_cost baking_dish_cost ingredient_cost apron_cost mixer_cost measuring_cups_cost spice_cost discount :=
by sorry

end NUMINAMATH_CALUDE_liz_shopping_cost_l620_62087


namespace NUMINAMATH_CALUDE_largest_number_bound_l620_62038

theorem largest_number_bound (a b : ℕ+) 
  (hcf_condition : Nat.gcd a b = 143)
  (lcm_condition : ∃ k : ℕ+, Nat.lcm a b = 143 * 17 * 23 * 31 * k) :
  max a b ≤ 143 * 17 * 23 * 31 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_bound_l620_62038


namespace NUMINAMATH_CALUDE_cylinder_not_triangle_l620_62013

-- Define the possible shapes
inductive Shape
  | Cylinder
  | Cone
  | Prism
  | Pyramid

-- Define a function to check if a shape can appear as a triangle
def canAppearAsTriangle (s : Shape) : Prop :=
  match s with
  | Shape.Cylinder => False
  | _ => True

-- Theorem statement
theorem cylinder_not_triangle :
  ∀ s : Shape, canAppearAsTriangle s ↔ s ≠ Shape.Cylinder :=
by
  sorry


end NUMINAMATH_CALUDE_cylinder_not_triangle_l620_62013


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_m_minus_n_equals_8_l620_62018

def vector_a : Fin 3 → ℝ := ![1, 3, -2]
def vector_b (m n : ℝ) : Fin 3 → ℝ := ![2, m + 1, n - 1]

def parallel (u v : Fin 3 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 3, v i = k * u i

theorem parallel_vectors_imply_m_minus_n_equals_8 (m n : ℝ) :
  parallel vector_a (vector_b m n) → m - n = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_m_minus_n_equals_8_l620_62018


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l620_62061

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * (a + b + c) = 140) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 21) : 
  2 * (a*b + b*c + c*a) = 784 := by
sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l620_62061


namespace NUMINAMATH_CALUDE_log_expression_simplification_l620_62074

theorem log_expression_simplification 
  (a b c d x y z w : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) : 
  Real.log (a / b) + Real.log (b / c) + Real.log (c / d) - Real.log (a * y * z / (d * x * w)) = Real.log (x * w / (y * z)) := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l620_62074


namespace NUMINAMATH_CALUDE_circle_passes_through_point_l620_62054

theorem circle_passes_through_point :
  ∀ (a b r : ℝ),
  b^2 = 8*a →                          -- Center (a, b) is on the parabola y² = 8x
  (a + 2)^2 + b^2 = r^2 →              -- Circle is tangent to the line x + 2 = 0
  (2 - a)^2 + b^2 = r^2 :=             -- Circle passes through (2, 0)
by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_point_l620_62054


namespace NUMINAMATH_CALUDE_max_projection_sum_l620_62055

-- Define the plane as ℝ²
def Plane := ℝ × ℝ

-- Define the dot product for vectors in the plane
def dot_product (v w : Plane) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define what it means for a vector to be a unit vector
def is_unit_vector (v : Plane) : Prop := dot_product v v = 1

-- State the theorem
theorem max_projection_sum 
  (a b c : Plane) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hc : is_unit_vector c) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c)
  (hab_dot : dot_product a b = 1/2) 
  (hbc_dot : dot_product b c = 1/2) :
  ∃ (max : ℝ), max = 5 ∧ 
    ∀ (e : Plane), is_unit_vector e → 
      |dot_product a e| + |2 * dot_product b e| + 3 * |dot_product c e| ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_projection_sum_l620_62055


namespace NUMINAMATH_CALUDE_no_integer_solution_l620_62088

theorem no_integer_solution : ¬∃ (n : ℕ+), ∃ (k : ℤ), (n.val^(3*n.val - 2) - 3*n.val + 1) / (3*n.val - 2) = k := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l620_62088


namespace NUMINAMATH_CALUDE_derivative_at_two_l620_62041

/-- Given a function f with the property that f(x) = 2xf'(2) + x^3 for all x,
    prove that f'(2) = -12 -/
theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * (deriv f 2) + x^3) :
  deriv f 2 = -12 := by sorry

end NUMINAMATH_CALUDE_derivative_at_two_l620_62041


namespace NUMINAMATH_CALUDE_sum_nth_group_is_cube_l620_62010

/-- Returns the nth odd number -/
def nthOdd (n : ℕ) : ℕ := 2 * n - 1

/-- Returns the sum of the first n odd numbers -/
def sumFirstNOdds (n : ℕ) : ℕ := n^2

/-- Returns the sum of odd numbers in the nth group -/
def sumNthGroup (n : ℕ) : ℕ :=
  sumFirstNOdds (sumFirstNOdds n) - sumFirstNOdds (sumFirstNOdds (n - 1))

theorem sum_nth_group_is_cube (n : ℕ) (h : 1 ≤ n ∧ n ≤ 5) : sumNthGroup n = n^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_nth_group_is_cube_l620_62010


namespace NUMINAMATH_CALUDE_number_puzzle_l620_62076

theorem number_puzzle (x : ℝ) : (x / 2) / 2 = 85 + 45 → x - 45 = 475 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l620_62076


namespace NUMINAMATH_CALUDE_base_seven_1732_equals_709_l620_62053

def base_seven_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem base_seven_1732_equals_709 :
  base_seven_to_ten [2, 3, 7, 1] = 709 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_1732_equals_709_l620_62053


namespace NUMINAMATH_CALUDE_irrational_sum_product_theorem_l620_62058

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ¬ (∃ (q : ℚ), (q : ℝ) = x)

-- State the theorem
theorem irrational_sum_product_theorem (a : ℝ) (h : IsIrrational a) :
  ∃ (b b' : ℝ), IsIrrational b ∧ IsIrrational b' ∧
    (∃ (q1 q2 : ℚ), (a + b : ℝ) = q1 ∧ (a * b' : ℝ) = q2) ∧
    IsIrrational (a * b) ∧ IsIrrational (a + b') := by
  sorry


end NUMINAMATH_CALUDE_irrational_sum_product_theorem_l620_62058


namespace NUMINAMATH_CALUDE_divisibility_by_eight_l620_62073

theorem divisibility_by_eight (a b c d : ℤ) :
  8 ∣ (1000 * a + 100 * b + 10 * c + d) ↔ 8 ∣ (4 * b + 2 * c + d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eight_l620_62073


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l620_62020

theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l620_62020


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l620_62091

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_neg_first : a 1 < 0)
  (h_increasing : ∀ n : ℕ, a n < a (n + 1)) :
  ∃ q : ℝ, 0 < q ∧ q < 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l620_62091


namespace NUMINAMATH_CALUDE_line_point_distance_l620_62008

/-- Given a point M(a,b) on the line 4x - 3y + c = 0, 
    if the minimum value of (a-1)² + (b-1)² is 4, 
    then c = -11 or c = 9 -/
theorem line_point_distance (a b c : ℝ) : 
  (4 * a - 3 * b + c = 0) → 
  (∃ (d : ℝ), d = 4 ∧ ∀ (x y : ℝ), 4 * x - 3 * y + c = 0 → (x - 1)^2 + (y - 1)^2 ≥ d) →
  (c = -11 ∨ c = 9) :=
sorry

end NUMINAMATH_CALUDE_line_point_distance_l620_62008


namespace NUMINAMATH_CALUDE_cow_starting_weight_l620_62032

/-- The starting weight of a cow, given certain conditions about its weight gain and value increase. -/
theorem cow_starting_weight (W : ℝ) : W = 400 :=
  -- Given conditions
  have weight_increase : W * 1.5 = W + W * 0.5 := by sorry
  have price_per_pound : ℝ := 3
  have value_increase : W * 1.5 * price_per_pound - W * price_per_pound = 600 := by sorry

  -- Proof
  sorry

end NUMINAMATH_CALUDE_cow_starting_weight_l620_62032


namespace NUMINAMATH_CALUDE_square_diagonal_less_than_twice_fg_l620_62092

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    B = (A.1 + s, A.2) ∧
    C = (A.1 + s, A.2 + s) ∧
    D = (A.1, A.2 + s)

-- Define that E is an internal point on side AD
def InternalPointOnSide (E A D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ E = (A.1, A.2 + t * (D.2 - A.2))

-- Define F as the foot of the perpendicular from B to CE
def PerpendicularFoot (F B C E : ℝ × ℝ) : Prop :=
  (F.1 - C.1) * (E.1 - C.1) + (F.2 - C.2) * (E.2 - C.2) = 0 ∧
  (F.1 - B.1) * (E.1 - C.1) + (F.2 - B.2) * (E.2 - C.2) = 0

-- Define that BG = FG
def EqualDistances (B F G : ℝ × ℝ) : Prop :=
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = (G.1 - F.1)^2 + (G.2 - F.2)^2

-- Define that the line through G parallel to BC passes through the midpoint of EF
def ParallelThroughMidpoint (G B C E F : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, G = ((E.1 + F.1)/2 + t*(C.1 - B.1), (E.2 + F.2)/2 + t*(C.2 - B.2))

-- State the theorem
theorem square_diagonal_less_than_twice_fg 
  (A B C D E F G : ℝ × ℝ) : 
  Square A B C D → 
  InternalPointOnSide E A D → 
  PerpendicularFoot F B C E → 
  EqualDistances B F G → 
  ParallelThroughMidpoint G B C E F → 
  (C.1 - A.1)^2 + (C.2 - A.2)^2 < 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_less_than_twice_fg_l620_62092


namespace NUMINAMATH_CALUDE_cos_negative_seventy_nine_sixths_pi_l620_62037

theorem cos_negative_seventy_nine_sixths_pi :
  Real.cos (-79/6 * Real.pi) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_seventy_nine_sixths_pi_l620_62037


namespace NUMINAMATH_CALUDE_max_jogs_is_seven_l620_62042

/-- Represents the number of items Bill buys -/
structure BillsPurchase where
  jags : Nat
  jigs : Nat
  jogs : Nat
  jugs : Nat

/-- Calculates the total cost of Bill's purchase -/
def totalCost (p : BillsPurchase) : Nat :=
  2 * p.jags + 3 * p.jigs + 8 * p.jogs + 5 * p.jugs

/-- Represents a valid purchase satisfying all conditions -/
def isValidPurchase (p : BillsPurchase) : Prop :=
  p.jags ≥ 1 ∧ p.jigs ≥ 1 ∧ p.jogs ≥ 1 ∧ p.jugs ≥ 1 ∧ totalCost p = 72

theorem max_jogs_is_seven :
  ∀ p : BillsPurchase, isValidPurchase p → p.jogs ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_jogs_is_seven_l620_62042


namespace NUMINAMATH_CALUDE_fraction_evaluation_l620_62011

theorem fraction_evaluation : 
  (((1 : ℚ) / 2 + (1 : ℚ) / 5) / ((3 : ℚ) / 7 - (1 : ℚ) / 14)) / ((3 : ℚ) / 4) = (196 : ℚ) / 75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l620_62011


namespace NUMINAMATH_CALUDE_initial_number_of_boys_l620_62057

theorem initial_number_of_boys (initial_days : ℝ) (additional_men : ℕ) (new_days : ℝ) :
  initial_days = 15 ∧ 
  additional_men = 200 ∧ 
  new_days = 12.5 →
  ∃ (B : ℕ), B * initial_days = (B + additional_men) * new_days ∧ B = 1000 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_boys_l620_62057


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l620_62083

/-- Given a man's speed against the current and the speed of the current,
    calculate the man's speed with the current. -/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed with the current is 21 km/hr. -/
theorem mans_speed_with_current :
  let speed_against_current : ℝ := 16
  let current_speed : ℝ := 2.5
  speed_with_current speed_against_current current_speed = 21 := by
  sorry

#eval speed_with_current 16 2.5

end NUMINAMATH_CALUDE_mans_speed_with_current_l620_62083


namespace NUMINAMATH_CALUDE_melanie_dimes_l620_62093

/-- The number of dimes Melanie initially had -/
def initial_dimes : ℕ := 7

/-- The number of dimes Melanie gave to her dad -/
def dimes_to_dad : ℕ := 8

/-- The number of dimes Melanie's mother gave her -/
def dimes_from_mother : ℕ := 4

/-- The number of dimes Melanie has now -/
def current_dimes : ℕ := 3

theorem melanie_dimes : 
  initial_dimes - dimes_to_dad + dimes_from_mother = current_dimes :=
by sorry

end NUMINAMATH_CALUDE_melanie_dimes_l620_62093


namespace NUMINAMATH_CALUDE_order_of_logarithms_l620_62048

theorem order_of_logarithms : 
  let a := (Real.log 3 / Real.log 2) ^ 3
  let b := Real.log 2
  let c := 1 / Real.sqrt 5
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_order_of_logarithms_l620_62048


namespace NUMINAMATH_CALUDE_pug_cleaning_theorem_l620_62067

/-- The number of pugs in the first scenario -/
def num_pugs : ℕ := 4

/-- The time taken by the unknown number of pugs to clean the house -/
def time1 : ℕ := 45

/-- The number of pugs in the second scenario -/
def num_pugs2 : ℕ := 15

/-- The time taken by the known number of pugs to clean the house -/
def time2 : ℕ := 12

/-- The theorem stating that the number of pugs in the first scenario is 4 -/
theorem pug_cleaning_theorem : 
  num_pugs * time1 = num_pugs2 * time2 := by sorry

end NUMINAMATH_CALUDE_pug_cleaning_theorem_l620_62067


namespace NUMINAMATH_CALUDE_f_has_one_zero_l620_62028

/-- The function f(x) defined in terms of the parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * (m + 1) * x - 1

/-- The set of real numbers m for which f(x) has exactly one zero -/
def one_zero_set : Set ℝ := {m : ℝ | m = -3 ∨ m = 0}

/-- Theorem stating that f(x) has exactly one zero if and only if m is in the one_zero_set -/
theorem f_has_one_zero (m : ℝ) : 
  (∃! x : ℝ, f m x = 0) ↔ m ∈ one_zero_set :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_l620_62028


namespace NUMINAMATH_CALUDE_ones_digit_largest_power_of_two_32_factorial_l620_62040

/-- The largest power of 2 that divides n! -/
def largest_power_of_two (n : ℕ) : ℕ := sorry

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_largest_power_of_two_32_factorial :
  ones_digit (2^(largest_power_of_two 32)) = 4 := by sorry

end NUMINAMATH_CALUDE_ones_digit_largest_power_of_two_32_factorial_l620_62040


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l620_62030

def geometric_sequence (a : ℕ → ℝ) := ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 + a 7 = 2) →
  (a 2 * a 9 = -8) →
  (a 1 + a 13 = 17 ∨ a 1 + a 13 = -17/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l620_62030


namespace NUMINAMATH_CALUDE_increasing_function_range_l620_62022

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (6 - a) * x - 4 * a else Real.log x / Real.log a

-- State the theorem
theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (6/5 < a ∧ a < 6) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_range_l620_62022


namespace NUMINAMATH_CALUDE_star_properties_l620_62009

/-- Custom multiplication operation for rational numbers -/
def star (x y : ℚ) : ℚ := x * y + 1

/-- Theorem stating the properties of the star operation -/
theorem star_properties :
  (star 2 3 = 7) ∧
  (star (star 1 4) (-1/2) = -3/2) ∧
  (∀ a b c : ℚ, star a (b + c) + 1 = star a b + star a c) := by
  sorry

end NUMINAMATH_CALUDE_star_properties_l620_62009


namespace NUMINAMATH_CALUDE_cousins_distribution_l620_62044

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to accommodate -/
def num_cousins : ℕ := 5

/-- The number of ways to distribute the cousins is 51 -/
theorem cousins_distribution :
  distribute num_cousins num_rooms = 51 := by sorry

end NUMINAMATH_CALUDE_cousins_distribution_l620_62044


namespace NUMINAMATH_CALUDE_approximate_cost_of_bicycle_and_fan_l620_62056

/-- The cost of a bicycle in yuan -/
def bicycle_cost : ℕ := 389

/-- The cost of an electric fan in yuan -/
def fan_cost : ℕ := 189

/-- The approximate total cost of buying a bicycle and an electric fan -/
def approximate_total_cost : ℕ := 600

/-- Theorem stating that the approximate total cost is 600 yuan -/
theorem approximate_cost_of_bicycle_and_fan :
  ∃ (error : ℕ), bicycle_cost + fan_cost = approximate_total_cost + error ∧ error < 100 := by
  sorry

end NUMINAMATH_CALUDE_approximate_cost_of_bicycle_and_fan_l620_62056


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_100_l620_62089

theorem largest_multiple_of_8_less_than_100 : 
  ∀ n : ℕ, n % 8 = 0 ∧ n < 100 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_less_than_100_l620_62089


namespace NUMINAMATH_CALUDE_jake_has_eight_peaches_l620_62052

-- Define the number of peaches each person has
def steven_peaches : ℕ := 15
def jill_peaches : ℕ := steven_peaches - 14
def jake_peaches : ℕ := steven_peaches - 7

-- Theorem statement
theorem jake_has_eight_peaches : jake_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_eight_peaches_l620_62052


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l620_62078

theorem sufficient_not_necessary (x : ℝ) : 
  (1 / x > 2 → x < 1 / 2) ∧ ¬(x < 1 / 2 → 1 / x > 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l620_62078


namespace NUMINAMATH_CALUDE_bus_stop_time_l620_62059

/-- Calculates the time a bus stops per hour given its speeds with and without stoppages. -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 50) 
  (h2 : speed_with_stops = 43) : ℝ :=
by
  -- The proof goes here
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l620_62059


namespace NUMINAMATH_CALUDE_math_competition_correct_answers_l620_62025

theorem math_competition_correct_answers 
  (total_questions : Nat) 
  (correct_score : Nat) 
  (incorrect_penalty : Nat) 
  (xiao_ming_score : Nat) 
  (xiao_hong_score : Nat) 
  (xiao_hua_score : Nat) :
  total_questions = 10 →
  correct_score = 10 →
  incorrect_penalty = 3 →
  xiao_ming_score = 87 →
  xiao_hong_score = 74 →
  xiao_hua_score = 9 →
  (total_questions - (total_questions * correct_score - xiao_ming_score) / (correct_score + incorrect_penalty)) +
  (total_questions - (total_questions * correct_score - xiao_hong_score) / (correct_score + incorrect_penalty)) +
  (total_questions - (total_questions * correct_score - xiao_hua_score) / (correct_score + incorrect_penalty)) = 20 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_correct_answers_l620_62025


namespace NUMINAMATH_CALUDE_expression_evaluation_l620_62027

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l620_62027


namespace NUMINAMATH_CALUDE_sum_of_four_squares_99_l620_62023

theorem sum_of_four_squares_99 : ∃ (a b c d w x y z : ℕ),
  a^2 + b^2 + c^2 + d^2 = 99 ∧
  w^2 + x^2 + y^2 + z^2 = 99 ∧
  (a, b, c, d) ≠ (w, x, y, z) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_four_squares_99_l620_62023


namespace NUMINAMATH_CALUDE_lcm_36_100_l620_62077

theorem lcm_36_100 : Nat.lcm 36 100 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_100_l620_62077


namespace NUMINAMATH_CALUDE_total_pages_is_281_l620_62063

/-- Calculates the total number of pages read over two months given Janine's reading habits --/
def total_pages_read : ℕ :=
  let last_month_books := 3 + 2
  let last_month_pages := 3 * 12 + 2 * 15
  let this_month_books := 2 * last_month_books
  let this_month_pages := 1 * 20 + 4 * 25 + 2 * 30 + 1 * 35
  last_month_pages + this_month_pages

/-- Proves that the total number of pages read over two months is 281 --/
theorem total_pages_is_281 : total_pages_read = 281 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_is_281_l620_62063


namespace NUMINAMATH_CALUDE_rectangular_field_area_l620_62016

/-- Calculates the area of a rectangular field given its perimeter and width-to-length ratio. -/
theorem rectangular_field_area 
  (perimeter : ℝ) 
  (width_to_length_ratio : ℝ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : width_to_length_ratio = 1/3) : 
  let width := perimeter / (2 * (1 + 1/width_to_length_ratio))
  let length := width / width_to_length_ratio
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l620_62016


namespace NUMINAMATH_CALUDE_negation_of_forall_gt_negation_of_gt_is_le_negation_of_forall_x_squared_gt_1_minus_2x_l620_62099

theorem negation_of_forall_gt (P : ℝ → Prop) :
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_gt_is_le {a b : ℝ} :
  ¬(a > b) ↔ (a ≤ b) :=
by sorry

theorem negation_of_forall_x_squared_gt_1_minus_2x :
  (¬∀ x : ℝ, x^2 > 1 - 2*x) ↔ (∃ x : ℝ, x^2 ≤ 1 - 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_gt_negation_of_gt_is_le_negation_of_forall_x_squared_gt_1_minus_2x_l620_62099


namespace NUMINAMATH_CALUDE_f_minimum_l620_62021

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := x^2 - 2*x*y + 6*y^2 - 14*x - 6*y + 72

/-- Theorem stating that f attains its minimum at (15/2, 1/2) -/
theorem f_minimum : 
  ∀ (x y : ℝ), f x y ≥ f (15/2) (1/2) := by sorry

end NUMINAMATH_CALUDE_f_minimum_l620_62021


namespace NUMINAMATH_CALUDE_fifth_term_ratio_l620_62098

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively -/
def arithmetic_sequences (a b : ℕ → ℝ) (S T : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2 ∧ T n = (n * (b 1 + b n)) / 2

/-- The ratio of sums S_n and T_n is 2n / (3n + 1) -/
def sum_ratio (S T : ℕ → ℝ) : Prop :=
  ∀ n, S n / T n = (2 * n : ℝ) / (3 * n + 1)

/-- The main theorem: given the conditions, prove a_5 / b_5 = 9 / 14 -/
theorem fifth_term_ratio (a b : ℕ → ℝ) (S T : ℕ → ℝ)
    (h1 : arithmetic_sequences a b S T) (h2 : sum_ratio S T) :
    a 5 / b 5 = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_ratio_l620_62098


namespace NUMINAMATH_CALUDE_molecular_weight_CaOH2_is_74_10_l620_62001

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of calcium atoms in Ca(OH)2 -/
def num_Ca : ℕ := 1

/-- The number of oxygen atoms in Ca(OH)2 -/
def num_O : ℕ := 2

/-- The number of hydrogen atoms in Ca(OH)2 -/
def num_H : ℕ := 2

/-- The molecular weight of Ca(OH)2 in g/mol -/
def molecular_weight_CaOH2 : ℝ :=
  num_Ca * atomic_weight_Ca + num_O * atomic_weight_O + num_H * atomic_weight_H

theorem molecular_weight_CaOH2_is_74_10 :
  molecular_weight_CaOH2 = 74.10 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_CaOH2_is_74_10_l620_62001


namespace NUMINAMATH_CALUDE_power_division_simplification_l620_62000

theorem power_division_simplification (a : ℝ) : (2 * a) ^ 7 / (2 * a) ^ 4 = 8 * a ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_simplification_l620_62000


namespace NUMINAMATH_CALUDE_inverse_f_at_120_l620_62080

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 + 9

-- State the theorem
theorem inverse_f_at_120 :
  ∃ (y : ℝ), f y = 120 ∧ y = (37 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_120_l620_62080


namespace NUMINAMATH_CALUDE_unique_base_solution_l620_62094

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (λ (i, d) acc => acc + d * b^i) 0

/-- The equation 142₂ + 163₂ = 315₂ holds in base b --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [1, 4, 2] b + to_decimal [1, 6, 3] b = to_decimal [3, 1, 5] b

theorem unique_base_solution :
  ∃! b : Nat, b > 6 ∧ equation_holds b :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l620_62094


namespace NUMINAMATH_CALUDE_total_gifts_needed_l620_62049

/-- The number of teams participating in the world cup -/
def num_teams : ℕ := 12

/-- The number of invited members per team who receive a gift -/
def members_per_team : ℕ := 4

/-- Theorem stating the total number of gifts needed for the event -/
theorem total_gifts_needed : num_teams * members_per_team = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_gifts_needed_l620_62049


namespace NUMINAMATH_CALUDE_min_value_3m_plus_n_l620_62012

/-- The minimum value of 3m + n given the conditions -/
theorem min_value_3m_plus_n (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) : 
  let f := fun x => a^(x + 3) - 2
  let A := (-3, -1)
  (A.1 / m + A.2 / n + 1 = 0) →
  ∀ m' n', m' > 0 → n' > 0 → 
    (m' / m' + n' / n' + 1 = 0) → 
    (3 * m + n ≤ 3 * m' + n') :=
by sorry

end NUMINAMATH_CALUDE_min_value_3m_plus_n_l620_62012


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_2pi_3_l620_62014

theorem cos_2alpha_plus_2pi_3 (α : Real) (h : Real.sin (α - π/6) = 2/3) :
  Real.cos (2*α + 2*π/3) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_2pi_3_l620_62014


namespace NUMINAMATH_CALUDE_min_covering_size_l620_62029

def X : Finset Nat := {1, 2, 3, 4, 5}

def is_covering (F : Finset (Finset Nat)) : Prop :=
  ∀ B ∈ Finset.powerset X, B.card = 3 → ∃ A ∈ F, A ⊆ B

theorem min_covering_size :
  ∃ F : Finset (Finset Nat),
    (∀ A ∈ F, A ⊆ X ∧ A.card = 2) ∧
    is_covering F ∧
    F.card = 10 ∧
    (∀ G : Finset (Finset Nat),
      (∀ A ∈ G, A ⊆ X ∧ A.card = 2) →
      is_covering G →
      G.card ≥ 10) :=
sorry

end NUMINAMATH_CALUDE_min_covering_size_l620_62029


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l620_62003

/-- An arithmetic sequence {a_n} with a_1 = 2 and a_3 + a_5 = 10 has a common difference of 1. -/
theorem arithmetic_sequence_common_difference : ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term condition
  a 3 + a 5 = 10 →                     -- sum of 3rd and 5th terms condition
  a 2 - a 1 = 1 :=                     -- common difference is 1
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l620_62003


namespace NUMINAMATH_CALUDE_pet_ownership_l620_62069

theorem pet_ownership (total : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total = 50)
  (h2 : dog_owners = 28)
  (h3 : cat_owners = 35)
  (h4 : dog_owners + cat_owners - total ≤ dog_owners)
  (h5 : dog_owners + cat_owners - total ≤ cat_owners) :
  dog_owners + cat_owners - total = 13 := by
sorry

end NUMINAMATH_CALUDE_pet_ownership_l620_62069


namespace NUMINAMATH_CALUDE_trigonometric_simplification_and_evaluation_l620_62002

theorem trigonometric_simplification_and_evaluation (α : Real) :
  (Real.tan (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α - Real.pi) * Real.sin (-Real.pi + α) * Real.cos (α + 5 * Real.pi / 2)) = -1 / Real.sin α ∧
  Real.tan α = 1 / 4 →
  1 / (2 * (Real.cos α)^2 - 3 * Real.sin α * Real.cos α) = 17 / 20 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_and_evaluation_l620_62002


namespace NUMINAMATH_CALUDE_possible_values_of_a_l620_62085

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a^3 - b^3 = 27*x^3) 
  (h2 : a - b = 2*x) : 
  a = x + 5*x/Real.sqrt 6 ∨ a = x - 5*x/Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l620_62085


namespace NUMINAMATH_CALUDE_cricketer_average_score_l620_62075

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_set_matches : ℕ) 
  (second_set_matches : ℕ) 
  (first_set_average : ℝ) 
  (second_set_average : ℝ) 
  (h1 : total_matches = first_set_matches + second_set_matches)
  (h2 : total_matches = 5)
  (h3 : first_set_matches = 2)
  (h4 : second_set_matches = 3)
  (h5 : first_set_average = 40)
  (h6 : second_set_average = 10) :
  (first_set_matches * first_set_average + second_set_matches * second_set_average) / total_matches = 22 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l620_62075


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l620_62017

theorem gcd_of_three_numbers : Nat.gcd 390 (Nat.gcd 455 546) = 13 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l620_62017


namespace NUMINAMATH_CALUDE_exactly_three_valid_combinations_l620_62097

/-- Represents the number of pairs of socks at each price point -/
structure SockCombination :=
  (x : ℕ)  -- Number of 18 yuan socks
  (y : ℕ)  -- Number of 30 yuan socks
  (z : ℕ)  -- Number of 39 yuan socks

/-- Checks if a combination is valid according to the problem constraints -/
def isValidCombination (c : SockCombination) : Prop :=
  18 * c.x + 30 * c.y + 39 * c.z = 100 ∧
  18 * c.x + 30 * c.y + 39 * c.z > 95

/-- The main theorem stating that there are exactly 3 valid combinations -/
theorem exactly_three_valid_combinations :
  ∃! (s : Finset SockCombination), 
    (∀ c ∈ s, isValidCombination c) ∧ 
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_exactly_three_valid_combinations_l620_62097


namespace NUMINAMATH_CALUDE_angle_half_in_second_quadrant_l620_62006

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

def is_in_second_quadrant (θ : Real) : Prop :=
  ∃ k : ℤ, k * Real.pi + Real.pi / 2 < θ ∧ θ < k * Real.pi + Real.pi

theorem angle_half_in_second_quadrant (α : Real) 
  (h1 : is_in_third_quadrant α) 
  (h2 : |Real.cos (α/2)| = -Real.cos (α/2)) : 
  is_in_second_quadrant (α/2) := by
  sorry


end NUMINAMATH_CALUDE_angle_half_in_second_quadrant_l620_62006


namespace NUMINAMATH_CALUDE_exponential_inequality_l620_62036

theorem exponential_inequality (x : ℝ) : 
  Real.exp (2 * x - 1) < 1 ↔ x < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l620_62036


namespace NUMINAMATH_CALUDE_fraction_decomposition_l620_62026

theorem fraction_decomposition : 
  ∃ (A B : ℚ), A = -12/11 ∧ B = 113/11 ∧
  ∀ (x : ℚ), x ≠ 1 ∧ x ≠ -8/3 →
  (7*x - 19) / (3*x^2 + 5*x - 8) = A / (x - 1) + B / (3*x + 8) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l620_62026


namespace NUMINAMATH_CALUDE_f_decreasing_and_k_maximum_l620_62079

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_decreasing_and_k_maximum :
  (∀ x > 0, (deriv f) x < 0) ∧
  (∀ x > 0, f x > 3 / (x + 1)) ∧
  (¬ ∃ k : ℕ, k > 3 ∧ ∀ x > 0, f x > (k : ℝ) / (x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_and_k_maximum_l620_62079


namespace NUMINAMATH_CALUDE_total_red_peaches_l620_62086

/-- The number of baskets of peaches -/
def num_baskets : ℕ := 6

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 16

/-- Theorem: The total number of red peaches in all baskets is 96 -/
theorem total_red_peaches : num_baskets * red_peaches_per_basket = 96 := by
  sorry

end NUMINAMATH_CALUDE_total_red_peaches_l620_62086


namespace NUMINAMATH_CALUDE_surface_polygon_angle_sum_sum_all_defects_l620_62046

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  -- Add necessary fields here
  
/-- An m-gon on the surface of a polyhedron -/
structure SurfacePolygon (P : ConvexPolyhedron) where
  m : ℕ  -- number of sides
  -- Add other necessary fields here

/-- The defect of a polyhedral angle -/
def defect (P : ConvexPolyhedron) (v : ℝ × ℝ × ℝ) : ℝ := sorry

/-- The sum of angles of a surface polygon -/
def sumAngles (P : ConvexPolyhedron) (S : SurfacePolygon P) : ℝ := sorry

/-- The sum of defects of vertices inside a surface polygon -/
def sumDefectsInside (P : ConvexPolyhedron) (S : SurfacePolygon P) : ℝ := sorry

/-- The sum of defects of all vertices of a polyhedron -/
def sumAllDefects (P : ConvexPolyhedron) : ℝ := sorry

theorem surface_polygon_angle_sum (P : ConvexPolyhedron) (S : SurfacePolygon P) :
  sumAngles P S = 2 * Real.pi * (S.m - 2 : ℝ) + sumDefectsInside P S := by sorry

theorem sum_all_defects (P : ConvexPolyhedron) :
  sumAllDefects P = 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_surface_polygon_angle_sum_sum_all_defects_l620_62046


namespace NUMINAMATH_CALUDE_thursday_rainfall_thursday_rainfall_proof_l620_62050

/-- Calculates the total rainfall on Thursday given the rainfall patterns of the week --/
theorem thursday_rainfall (monday_rain : Real) (tuesday_decrease : Real) 
  (wednesday_increase_percent : Real) (thursday_decrease_percent : Real) 
  (thursday_additional_rain : Real) : Real :=
  let tuesday_rain := monday_rain - tuesday_decrease
  let wednesday_rain := tuesday_rain * (1 + wednesday_increase_percent)
  let thursday_rain_before_system := wednesday_rain * (1 - thursday_decrease_percent)
  let thursday_total_rain := thursday_rain_before_system + thursday_additional_rain
  thursday_total_rain

/-- Proves that the total rainfall on Thursday is 0.54 inches given the specific conditions --/
theorem thursday_rainfall_proof :
  thursday_rainfall 0.9 0.7 0.5 0.2 0.3 = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_thursday_rainfall_thursday_rainfall_proof_l620_62050


namespace NUMINAMATH_CALUDE_quadratic_roots_conditions_l620_62004

/-- The quadratic equation x^2 + 2x + 2m = 0 has two distinct real roots -/
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0

/-- The sum of squares of the roots of x^2 + 2x + 2m = 0 is 8 -/
def sum_of_squares_is_8 (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0 ∧ x₁^2 + x₂^2 = 8

theorem quadratic_roots_conditions (m : ℝ) :
  (has_two_distinct_real_roots m ↔ m < 1/2) ∧
  (sum_of_squares_is_8 m → m = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_conditions_l620_62004


namespace NUMINAMATH_CALUDE_solution_range_l620_62095

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, a * x < 6 ∧ (3 * x - 6 * a) / 2 > a / 3 - 1) → 
  a ≤ -3/2 := by
sorry

end NUMINAMATH_CALUDE_solution_range_l620_62095


namespace NUMINAMATH_CALUDE_max_value_wxyz_l620_62043

theorem max_value_wxyz (w x y z : ℝ) 
  (nonneg_w : w ≥ 0) (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_100 : w + x + y + z = 100) : 
  w * x + x * y + y * z ≤ 2500 := by
sorry

end NUMINAMATH_CALUDE_max_value_wxyz_l620_62043


namespace NUMINAMATH_CALUDE_hair_sufficient_for_skin_l620_62039

/-- Represents the state of having skin -/
def HasSkin : Prop := sorry

/-- Represents the state of having hair -/
def HasHair : Prop := sorry

/-- If there is no skin, there cannot be hair -/
axiom no_skin_no_hair : ¬HasSkin → ¬HasHair

/-- Prove that having hair is a sufficient condition for having skin -/
theorem hair_sufficient_for_skin : HasHair → HasSkin := by
  sorry

end NUMINAMATH_CALUDE_hair_sufficient_for_skin_l620_62039


namespace NUMINAMATH_CALUDE_factorization_equality_l620_62082

theorem factorization_equality (a b : ℝ) : 3 * a * b^2 + a^2 * b = a * b * (3 * b + a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l620_62082


namespace NUMINAMATH_CALUDE_sum_bounds_l620_62072

theorem sum_bounds (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h_eq : a^2 + b^2 + c^2 + a*b + 2/3*a*c + 4/3*b*c = 1) : 
  1 ≤ a + b + c ∧ a + b + c ≤ Real.sqrt 345 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_bounds_l620_62072


namespace NUMINAMATH_CALUDE_students_playing_neither_sport_l620_62070

theorem students_playing_neither_sport
  (total : ℕ)
  (football : ℕ)
  (tennis : ℕ)
  (both : ℕ)
  (h1 : total = 50)
  (h2 : football = 32)
  (h3 : tennis = 28)
  (h4 : both = 24) :
  total - (football + tennis - both) = 14 :=
by sorry

end NUMINAMATH_CALUDE_students_playing_neither_sport_l620_62070


namespace NUMINAMATH_CALUDE_total_widgets_sold_is_360_l620_62060

/-- The sum of an arithmetic sequence with first term 3, common difference 3, and 15 terms -/
def widget_sales_sum : ℕ :=
  let first_term := 3
  let common_difference := 3
  let num_days := 15
  (num_days * (2 * first_term + (num_days - 1) * common_difference)) / 2

/-- Theorem stating that the total number of widgets sold is 360 -/
theorem total_widgets_sold_is_360 : widget_sales_sum = 360 := by
  sorry

end NUMINAMATH_CALUDE_total_widgets_sold_is_360_l620_62060


namespace NUMINAMATH_CALUDE_sector_area_l620_62035

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 5) (h2 : θ = 2) :
  (1 / 2) * r^2 * θ = 25 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l620_62035


namespace NUMINAMATH_CALUDE_angle_A_measure_l620_62034

theorem angle_A_measure :
  ∀ (A B C : ℝ) (small_angle : ℝ),
  B = 120 →
  B + C = 180 →
  small_angle = 50 →
  small_angle + C + 70 = 180 →
  A + B = 180 →
  A = 60 :=
by sorry

end NUMINAMATH_CALUDE_angle_A_measure_l620_62034


namespace NUMINAMATH_CALUDE_remainder_2_1000_mod_17_l620_62084

theorem remainder_2_1000_mod_17 (h : Prime 17) : 2^1000 % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2_1000_mod_17_l620_62084


namespace NUMINAMATH_CALUDE_area_of_special_triangle_l620_62064

/-- Given points A and B on the graph of y = 1/x in the first quadrant,
    if ∠OAB = 90° and OA = AB, then the area of triangle OAB is √5/2 -/
theorem area_of_special_triangle (A B : ℝ × ℝ) : 
  (A.2 = 1 / A.1) →  -- A is on y = 1/x
  (B.2 = 1 / B.1) →  -- B is on y = 1/x
  (A.1 > 0 ∧ A.2 > 0) →  -- A is in first quadrant
  (B.1 > 0 ∧ B.2 > 0) →  -- B is in first quadrant
  (A.1 * (B.1 - A.1) + A.2 * (B.2 - A.2) = 0) →  -- ∠OAB = 90°
  (A.1^2 + A.2^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2) →  -- OA = AB
  (1/2 * Real.sqrt (A.1^2 + A.2^2) = Real.sqrt 5 / 2) := by
sorry

end NUMINAMATH_CALUDE_area_of_special_triangle_l620_62064


namespace NUMINAMATH_CALUDE_problem_solution_l620_62007

theorem problem_solution :
  (∃ x : ℝ, x^2 = 81 ∧ (x = 9 ∨ x = -9)) ∧
  |Real.sqrt 15 - 4| = 4 - Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l620_62007


namespace NUMINAMATH_CALUDE_smallest_positive_root_comparison_l620_62015

theorem smallest_positive_root_comparison : ∃ (x₁ x₂ : ℝ), 
  (x₁ > 0 ∧ x₂ > 0) ∧ 
  (x₁^2011 + 2011*x₁ - 1 = 0) ∧
  (x₂^2011 - 2011*x₂ + 1 = 0) ∧
  (∀ y₁ > 0, y₁^2011 + 2011*y₁ - 1 = 0 → y₁ ≥ x₁) ∧
  (∀ y₂ > 0, y₂^2011 - 2011*y₂ + 1 = 0 → y₂ ≥ x₂) ∧
  (x₁ < x₂) := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_root_comparison_l620_62015


namespace NUMINAMATH_CALUDE_segment_bisection_l620_62005

-- Define the angle
structure Angle where
  C : Point
  K : Point
  L : Point

-- Define the condition for a point to be inside an angle
def InsideAngle (α : Angle) (O : Point) : Prop := sorry

-- Define the condition for a point to be on a line
def OnLine (P Q R : Point) : Prop := sorry

-- Define the midpoint of a segment
def Midpoint (M A B : Point) : Prop := sorry

-- Main theorem
theorem segment_bisection (α : Angle) (O : Point) 
  (h : InsideAngle α O) : 
  ∃ (A B : Point), 
    OnLine α.C α.K A ∧ 
    OnLine α.C α.L B ∧ 
    Midpoint O A B :=
  sorry

end NUMINAMATH_CALUDE_segment_bisection_l620_62005


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l620_62033

-- Define a function f with the given symmetry property
def f : ℝ → ℝ := sorry

-- State the symmetry condition
axiom f_symmetry (x : ℝ) : f x = f (5 - x)

-- Define the line of symmetry
def line_of_symmetry : ℝ → Prop := λ x ↦ x = 2.5

-- Theorem stating that the line x = 2.5 is an axis of symmetry
theorem axis_of_symmetry :
  ∀ (x y : ℝ), f x = y → f (5 - x) = y → line_of_symmetry ((x + (5 - x)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l620_62033


namespace NUMINAMATH_CALUDE_hemisphere_chord_length_l620_62068

theorem hemisphere_chord_length (R : ℝ) (h : R = 20) : 
  let chord_length := 2 * R * Real.sqrt 2 / 2
  chord_length = 20 * Real.sqrt 2 := by
  sorry

#check hemisphere_chord_length

end NUMINAMATH_CALUDE_hemisphere_chord_length_l620_62068


namespace NUMINAMATH_CALUDE_lucky_point_properties_l620_62062

/-- Definition of a lucky point -/
def is_lucky_point (m n x y : ℝ) : Prop :=
  2 * m = 4 + n ∧ x = m - 1 ∧ y = (n + 2) / 2

theorem lucky_point_properties :
  -- Part 1: When m = 2, the lucky point is (1, 1)
  (∃ n : ℝ, is_lucky_point 2 n 1 1) ∧
  -- Part 2: Point (3, 3) is a lucky point
  (∃ m n : ℝ, is_lucky_point m n 3 3) ∧
  -- Part 3: If (a, 2a-1) is a lucky point, then it's in the first quadrant
  (∀ a m n : ℝ, is_lucky_point m n a (2*a-1) → a > 0 ∧ 2*a-1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_lucky_point_properties_l620_62062
