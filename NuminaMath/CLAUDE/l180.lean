import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l180_18079

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set of the first inequality
def S₁ : Set ℝ := {x | x < -2 ∨ x > -1/2}

-- State the theorem
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : ∀ x, f a b c x < 0 ↔ x ∈ S₁) :
  ∀ x, f a (-b) c x > 0 ↔ 1/2 < x ∧ x < 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l180_18079


namespace NUMINAMATH_CALUDE_sqrt_3_times_6_plus_sqrt_8_two_divided_by_sqrt_5_minus_2_l180_18084

-- Part 1
theorem sqrt_3_times_6_plus_sqrt_8 : 
  Real.sqrt 3 * Real.sqrt 6 + Real.sqrt 8 = 5 * Real.sqrt 2 := by sorry

-- Part 2
theorem two_divided_by_sqrt_5_minus_2 : 
  2 / (Real.sqrt 5 - 2) = 2 * Real.sqrt 5 + 4 := by sorry

end NUMINAMATH_CALUDE_sqrt_3_times_6_plus_sqrt_8_two_divided_by_sqrt_5_minus_2_l180_18084


namespace NUMINAMATH_CALUDE_economics_class_question_l180_18052

theorem economics_class_question (total_students : ℕ) 
  (q2_correct : ℕ) (not_taken : ℕ) (both_correct : ℕ) :
  total_students = 40 →
  q2_correct = 29 →
  not_taken = 10 →
  both_correct = 29 →
  ∃ (q1_correct : ℕ), q1_correct ≥ 29 :=
by sorry

end NUMINAMATH_CALUDE_economics_class_question_l180_18052


namespace NUMINAMATH_CALUDE_intersection_P_Q_l180_18036

-- Define the sets P and Q
def P : Set ℝ := {x | x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l180_18036


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l180_18012

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by
  sorry

theorem negation_of_quadratic_equation : 
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) ↔ (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l180_18012


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l180_18009

theorem integer_divisibility_problem (a b : ℤ) :
  (a^6 + 1) ∣ (b^11 - 2023*b^3 + 40*b) →
  (a^4 - 1) ∣ (b^10 - 2023*b^2 - 41) →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l180_18009


namespace NUMINAMATH_CALUDE_set_no_duplicate_elements_l180_18029

theorem set_no_duplicate_elements {α : Type*} (S : Set α) :
  ∀ x ∈ S, ∀ y ∈ S, x = y → x = y :=
by sorry

end NUMINAMATH_CALUDE_set_no_duplicate_elements_l180_18029


namespace NUMINAMATH_CALUDE_square_expression_is_perfect_square_l180_18092

theorem square_expression_is_perfect_square (n k l : ℕ) (h : n^2 + k^2 = 2 * l^2) :
  (2 * l - n - k) * (2 * l - n + k) / 2 = (l - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_expression_is_perfect_square_l180_18092


namespace NUMINAMATH_CALUDE_some_beautiful_objects_are_colorful_l180_18019

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Rose Beautiful Colorful : U → Prop)

-- State the theorem
theorem some_beautiful_objects_are_colorful :
  (∀ x, Rose x → Beautiful x) →  -- All roses are beautiful
  (∃ x, Colorful x ∧ Rose x) →   -- Some colorful objects are roses
  (∃ x, Beautiful x ∧ Colorful x) -- Some beautiful objects are colorful
  := by sorry

end NUMINAMATH_CALUDE_some_beautiful_objects_are_colorful_l180_18019


namespace NUMINAMATH_CALUDE_sum_of_powers_eq_7290_l180_18067

/-- The power of a triple of positive integers -/
def power (x y z : ℕ) : ℕ := max x (max y z) + min x (min y z)

/-- The sum of powers of all triples (x,y,z) where x,y,z ≤ 9 -/
def sum_of_powers : ℕ :=
  (Finset.range 9).sum (fun x =>
    (Finset.range 9).sum (fun y =>
      (Finset.range 9).sum (fun z =>
        power (x + 1) (y + 1) (z + 1))))

theorem sum_of_powers_eq_7290 : sum_of_powers = 7290 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_eq_7290_l180_18067


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l180_18034

theorem quadratic_equation_solution :
  let equation := fun x : ℂ => 3 * x^2 + 7 - (6 * x - 4)
  let solution1 := 1 + (2 * Real.sqrt 6 / 3) * I
  let solution2 := 1 - (2 * Real.sqrt 6 / 3) * I
  let a : ℝ := 1
  let b : ℝ := 2 * Real.sqrt 6 / 3
  (equation solution1 = 0) ∧
  (equation solution2 = 0) ∧
  (a + b^2 = 11/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l180_18034


namespace NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l180_18008

/-- The amount of money Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l180_18008


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l180_18001

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let d : ℝ := 10
  let r : ℝ := d / 2
  let area : ℝ := π * r^2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l180_18001


namespace NUMINAMATH_CALUDE_zoo_count_difference_l180_18015

theorem zoo_count_difference (zebras camels monkeys giraffes : ℕ) : 
  zebras = 12 →
  camels = zebras / 2 →
  monkeys = 4 * camels →
  giraffes = 2 →
  monkeys - giraffes = 22 := by
sorry

end NUMINAMATH_CALUDE_zoo_count_difference_l180_18015


namespace NUMINAMATH_CALUDE_difference_equals_three_44ths_l180_18050

/-- The decimal representation of 0.overline{81} -/
def repeating_decimal : ℚ := 9/11

/-- The decimal representation of 0.75 -/
def decimal_75 : ℚ := 3/4

/-- The theorem stating that the difference between 0.overline{81} and 0.75 is 3/44 -/
theorem difference_equals_three_44ths : 
  repeating_decimal - decimal_75 = 3/44 := by sorry

end NUMINAMATH_CALUDE_difference_equals_three_44ths_l180_18050


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l180_18065

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - I) / (1 - 3*I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l180_18065


namespace NUMINAMATH_CALUDE_third_circle_radius_l180_18068

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1) ^ 2 + (y2 - y1) ^ 2 = (c1.radius + c2.radius) ^ 2

/-- Checks if a circle is tangent to the x-axis -/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  c.center.2 = c.radius

/-- The main theorem -/
theorem third_circle_radius 
  (circle_A circle_B circle_C : Circle)
  (h1 : circle_A.radius = 2)
  (h2 : circle_B.radius = 3)
  (h3 : are_externally_tangent circle_A circle_B)
  (h4 : circle_A.center.1 + 6 = circle_B.center.1)
  (h5 : circle_A.center.2 = circle_B.center.2)
  (h6 : are_externally_tangent circle_A circle_C)
  (h7 : are_externally_tangent circle_B circle_C)
  (h8 : is_tangent_to_x_axis circle_C) :
  circle_C.radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l180_18068


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l180_18006

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 6

/-- The number of long diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 3

/-- The total number of elements in set S -/
def total_elements : ℕ := num_sides + num_short_diagonals + num_long_diagonals

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := 11 / 35

theorem hexagon_segment_probability :
  (num_sides * (num_sides - 1) + num_short_diagonals * (num_short_diagonals - 1) + num_long_diagonals * (num_long_diagonals - 1)) / (total_elements * (total_elements - 1)) = prob_same_length :=
sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l180_18006


namespace NUMINAMATH_CALUDE_min_value_on_common_chord_l180_18097

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 4

-- Define the common chord
def common_chord (x y : ℝ) : Prop := circle1 x y ∧ circle2 x y

-- Theorem statement
theorem min_value_on_common_chord :
  ∀ a b : ℝ, a > 0 → b > 0 → common_chord a b →
  (∀ x y : ℝ, x > 0 → y > 0 → common_chord x y → 1/a + 9/b ≤ 1/x + 9/y) →
  1/a + 9/b = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_common_chord_l180_18097


namespace NUMINAMATH_CALUDE_equal_charges_at_60_minutes_l180_18098

/-- United Telephone's base rate in dollars -/
def united_base : ℝ := 9

/-- United Telephone's per-minute rate in dollars -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate in dollars -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate in dollars -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the charges are equal -/
def equal_minutes : ℝ := 60

theorem equal_charges_at_60_minutes :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_charges_at_60_minutes_l180_18098


namespace NUMINAMATH_CALUDE_min_rooks_to_attack_all_white_cells_l180_18032

/-- Represents a cell on the chessboard -/
structure Cell :=
  (row : Fin 9)
  (col : Fin 9)

/-- Determines if a cell is white based on its position -/
def isWhite (c : Cell) : Bool :=
  (c.row.val + c.col.val) % 2 = 0

/-- Represents a rook's position on the board -/
structure Rook :=
  (position : Cell)

/-- Determines if a cell is under attack by a rook -/
def isUnderAttack (c : Cell) (r : Rook) : Bool :=
  c.row = r.position.row ∨ c.col = r.position.col

/-- The main theorem stating the minimum number of rooks required -/
theorem min_rooks_to_attack_all_white_cells :
  ∃ (rooks : List Rook),
    rooks.length = 5 ∧
    (∀ c : Cell, isWhite c → ∃ r ∈ rooks, isUnderAttack c r) ∧
    (∀ (rooks' : List Rook),
      rooks'.length < 5 →
      ¬(∀ c : Cell, isWhite c → ∃ r ∈ rooks', isUnderAttack c r)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rooks_to_attack_all_white_cells_l180_18032


namespace NUMINAMATH_CALUDE_max_groups_equals_gcd_l180_18063

theorem max_groups_equals_gcd (boys girls : ℕ) (h1 : boys = 120) (h2 : girls = 140) :
  let max_groups := Nat.gcd boys girls
  ∀ k : ℕ, k ∣ boys ∧ k ∣ girls → k ≤ max_groups :=
by sorry

end NUMINAMATH_CALUDE_max_groups_equals_gcd_l180_18063


namespace NUMINAMATH_CALUDE_calculate_upstream_speed_l180_18038

/-- Represents the speed of a man rowing in different water conditions -/
structure RowingSpeed where
  still : ℝ  -- Speed in still water
  downstream : ℝ  -- Speed downstream
  upstream : ℝ  -- Speed upstream

/-- Theorem: Given a man's speed in still water and downstream, calculate his upstream speed -/
theorem calculate_upstream_speed (speed : RowingSpeed) 
  (h1 : speed.still = 35)
  (h2 : speed.downstream = 40) : 
  speed.upstream = 30 := by
  sorry

#check calculate_upstream_speed

end NUMINAMATH_CALUDE_calculate_upstream_speed_l180_18038


namespace NUMINAMATH_CALUDE_largest_unformable_amount_correct_l180_18020

/-- Represents the coin denominations in Limonia -/
def coin_denominations (n : ℕ) : Finset ℕ := {3*n - 2, 6*n - 1, 6*n + 2, 6*n + 5}

/-- Predicate to check if an amount can be formed using given coin denominations -/
def is_formable (amount : ℕ) (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), amount = a*(3*n - 2) + b*(6*n - 1) + c*(6*n + 2) + d*(6*n + 5)

/-- The largest amount that cannot be formed using the coin denominations -/
def largest_unformable_amount (n : ℕ) : ℕ := 6*n^2 - 4*n - 3

/-- Main theorem: The largest amount that cannot be formed is 6n^2 - 4n - 3 -/
theorem largest_unformable_amount_correct (n : ℕ) :
  (∀ k > largest_unformable_amount n, is_formable k n) ∧
  ¬is_formable (largest_unformable_amount n) n :=
sorry

end NUMINAMATH_CALUDE_largest_unformable_amount_correct_l180_18020


namespace NUMINAMATH_CALUDE_add_preserves_inequality_l180_18026

theorem add_preserves_inequality (a b : ℝ) (h : a > b) : 2 + a > 2 + b := by
  sorry

end NUMINAMATH_CALUDE_add_preserves_inequality_l180_18026


namespace NUMINAMATH_CALUDE_triangle_area_l180_18051

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (1/2 * b * c * Real.sin A = 4 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l180_18051


namespace NUMINAMATH_CALUDE_toys_per_day_l180_18021

/-- A factory produces toys under the following conditions:
  * The factory produces 3400 toys per week.
  * The workers work 5 days a week.
  * The same number of toys is made every day. -/
def toy_factory (total_toys : ℕ) (work_days : ℕ) (daily_production : ℕ) : Prop :=
  total_toys = 3400 ∧ work_days = 5 ∧ daily_production * work_days = total_toys

/-- The number of toys produced each day is 680. -/
theorem toys_per_day : 
  ∀ (total_toys work_days daily_production : ℕ), 
  toy_factory total_toys work_days daily_production → daily_production = 680 :=
sorry

end NUMINAMATH_CALUDE_toys_per_day_l180_18021


namespace NUMINAMATH_CALUDE_segment_length_is_zero_l180_18081

/-- Triangle with side lengths and an angle -/
structure Triangle :=
  (a b c : ℝ)
  (angle : ℝ)

/-- The problem setup -/
def problem : Prop :=
  ∃ (ABC DEF : Triangle),
    ABC.a = 8 ∧ ABC.b = 12 ∧ ABC.c = 10 ∧
    DEF.a = 4 ∧ DEF.b = 6 ∧ DEF.c = 5 ∧
    ABC.angle = 100 ∧ DEF.angle = 100 ∧
    ∀ (BD : ℝ), BD = 0

/-- The theorem to be proved -/
theorem segment_length_is_zero : problem := by sorry

end NUMINAMATH_CALUDE_segment_length_is_zero_l180_18081


namespace NUMINAMATH_CALUDE_inequality_proof_l180_18003

theorem inequality_proof (x a b c : ℝ) 
  (h1 : x ≠ a) (h2 : x ≠ b) (h3 : x ≠ c) 
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) : 
  1 / (x - a) + 1 / (x - b) > 1 / (x - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l180_18003


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l180_18035

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (2*m - 1, -1)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l180_18035


namespace NUMINAMATH_CALUDE_arithmetic_sequence_perfect_square_sum_l180_18057

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_perfect_square_sum (a d : ℕ) :
  (∀ n : ℕ, is_perfect_square (sum_arithmetic_sequence a d n)) ↔
  (∃ b : ℕ, a = b^2 ∧ d = 2 * b^2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_perfect_square_sum_l180_18057


namespace NUMINAMATH_CALUDE_cost_price_calculation_l180_18061

/-- Proves that the cost price of an article is 95 given the specified conditions -/
theorem cost_price_calculation (marked_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  marked_price = 125 ∧ 
  discount_rate = 0.05 ∧ 
  profit_rate = 0.25 →
  ∃ (cost_price : ℝ), 
    cost_price = 95 ∧ 
    marked_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l180_18061


namespace NUMINAMATH_CALUDE_scout_troop_profit_l180_18062

/-- Calculates the profit of a scout troop selling candy bars -/
theorem scout_troop_profit (num_bars : ℕ) (buy_price : ℚ) (sell_price : ℚ) : 
  num_bars = 1500 → 
  buy_price = 1/3 → 
  sell_price = 2/3 → 
  (sell_price - buy_price) * num_bars = 500 := by
  sorry

#check scout_troop_profit

end NUMINAMATH_CALUDE_scout_troop_profit_l180_18062


namespace NUMINAMATH_CALUDE_volume_of_specific_parallelepiped_l180_18047

/-- A rectangular parallelepiped with vertices A B C D A₁ B₁ C₁ D₁ -/
structure RectangularParallelepiped where
  base_length : ℝ
  base_width : ℝ
  height : ℝ

/-- A plane passing through vertices A, C, and D₁ of the parallelepiped -/
structure DiagonalPlane where
  parallelepiped : RectangularParallelepiped
  dihedral_angle : ℝ

/-- The volume of a rectangular parallelepiped -/
def volume (p : RectangularParallelepiped) : ℝ :=
  p.base_length * p.base_width * p.height

/-- Theorem: Volume of the specific parallelepiped -/
theorem volume_of_specific_parallelepiped (p : RectangularParallelepiped) 
  (d : DiagonalPlane) (h1 : p.base_length = 4) (h2 : p.base_width = 3) 
  (h3 : d.parallelepiped = p) (h4 : d.dihedral_angle = π / 3) :
  volume p = (144 * Real.sqrt 3) / 5 := by
  sorry


end NUMINAMATH_CALUDE_volume_of_specific_parallelepiped_l180_18047


namespace NUMINAMATH_CALUDE_new_student_weight_l180_18069

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℝ) (new_avg : ℝ) :
  initial_count = 19 →
  initial_avg = 15 →
  new_avg = 14.4 →
  (initial_count * initial_avg + (initial_count + 1) * new_avg - initial_count * initial_avg) / (initial_count + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_new_student_weight_l180_18069


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l180_18078

theorem arithmetic_geometric_mean_square_sum (a b : ℝ) :
  (a + b) / 2 = 20 → Real.sqrt (a * b) = Real.sqrt 135 → a^2 + b^2 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_square_sum_l180_18078


namespace NUMINAMATH_CALUDE_intersection_M_N_l180_18002

def M : Set ℝ := {x | 0 < x ∧ x < 3}
def N : Set ℝ := {x | |x| > 2}

theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l180_18002


namespace NUMINAMATH_CALUDE_game_theorems_l180_18058

/-- Game with three possible point values and their probabilities --/
structure Game where
  p : ℝ
  prob_5 : ℝ := 2 * p
  prob_10 : ℝ := p
  prob_20 : ℝ := 1 - 3 * p
  h_p_pos : 0 < p
  h_p_bound : p < 1/3
  h_prob_sum : prob_5 + prob_10 + prob_20 = 1

/-- A round consists of three games --/
def Round := Fin 3 → Game

/-- The probability of total points not exceeding 25 in one round --/
def prob_not_exceed_25 (r : Round) : ℝ := sorry

/-- The expected value of total points in one round --/
def expected_value (r : Round) : ℝ := sorry

theorem game_theorems (r : Round) (h_same_p : ∀ i j : Fin 3, (r i).p = (r j).p) :
  (∃ (p : ℝ), prob_not_exceed_25 r = 26 * p^3) ∧
  (∃ (p : ℝ), p = 1/9 → expected_value r = 140/3) :=
sorry

end NUMINAMATH_CALUDE_game_theorems_l180_18058


namespace NUMINAMATH_CALUDE_factorization_x3y_minus_xy_l180_18004

theorem factorization_x3y_minus_xy (x y : ℝ) : x^3*y - x*y = x*y*(x - 1)*(x + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x3y_minus_xy_l180_18004


namespace NUMINAMATH_CALUDE_trigonometric_identity_l180_18096

theorem trigonometric_identity (α : Real) 
  (h : Real.sqrt 2 * Real.sin (α + π / 4) = 4 * Real.cos α) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l180_18096


namespace NUMINAMATH_CALUDE_division_problem_l180_18011

theorem division_problem : ∃ x : ℝ, 550 - (104 / x) = 545 ∧ x = 20.8 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l180_18011


namespace NUMINAMATH_CALUDE_madeline_sleep_hours_madeline_sleeps_eight_hours_l180_18046

/-- Calculates the number of hours Madeline spends sleeping per day given her weekly schedule. -/
theorem madeline_sleep_hours (class_hours_per_week : ℕ) 
                              (homework_hours_per_day : ℕ) 
                              (work_hours_per_week : ℕ) 
                              (leftover_hours_per_week : ℕ) : ℕ :=
  let total_hours_per_week : ℕ := 24 * 7
  let remaining_hours : ℕ := total_hours_per_week - class_hours_per_week - 
                             (homework_hours_per_day * 7) - work_hours_per_week - 
                             leftover_hours_per_week
  remaining_hours / 7

/-- Proves that Madeline spends 8 hours per day sleeping given her schedule. -/
theorem madeline_sleeps_eight_hours : 
  madeline_sleep_hours 18 4 20 46 = 8 := by
  sorry

end NUMINAMATH_CALUDE_madeline_sleep_hours_madeline_sleeps_eight_hours_l180_18046


namespace NUMINAMATH_CALUDE_debby_pancakes_count_l180_18041

/-- The number of pancakes Debby made with blueberries -/
def blueberry_pancakes : ℕ := 20

/-- The number of pancakes Debby made with bananas -/
def banana_pancakes : ℕ := 24

/-- The number of plain pancakes Debby made -/
def plain_pancakes : ℕ := 23

/-- The total number of pancakes Debby made -/
def total_pancakes : ℕ := blueberry_pancakes + banana_pancakes + plain_pancakes

theorem debby_pancakes_count : total_pancakes = 67 := by
  sorry

end NUMINAMATH_CALUDE_debby_pancakes_count_l180_18041


namespace NUMINAMATH_CALUDE_consecutive_composites_exist_l180_18083

/-- A natural number is composite if it has more than two distinct positive divisors. -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

/-- A sequence of n consecutive composite numbers starting from k. -/
def ConsecutiveComposites (k n : ℕ) : Prop :=
  ∀ i : ℕ, i < n → IsComposite (k + i)

/-- The existence of a sequence of 9 and 11 consecutive composite numbers
    among the first 500 natural numbers. -/
theorem consecutive_composites_exist :
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 9) ∧
  (∃ k : ℕ, k ≤ 500 ∧ ConsecutiveComposites k 11) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_composites_exist_l180_18083


namespace NUMINAMATH_CALUDE_plugs_count_l180_18059

/-- The number of pairs of mittens in the box -/
def mittens_pairs : ℕ := 150

/-- The number of pairs of plugs initially in the box -/
def initial_plugs_pairs : ℕ := mittens_pairs + 20

/-- The number of additional pairs of plugs added -/
def additional_plugs_pairs : ℕ := 30

/-- The total number of plugs after additions -/
def total_plugs : ℕ := 2 * (initial_plugs_pairs + additional_plugs_pairs)

theorem plugs_count : total_plugs = 400 := by
  sorry

end NUMINAMATH_CALUDE_plugs_count_l180_18059


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l180_18022

def p (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 4) ∧
  (∀ x : ℝ, (x - 1) * (x - 2) * (x - 4) = p x) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l180_18022


namespace NUMINAMATH_CALUDE_arccos_cos_eq_half_x_solution_l180_18016

theorem arccos_cos_eq_half_x_solution (x : Real) :
  -π/3 ≤ x → x ≤ π/3 → Real.arccos (Real.cos x) = x/2 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eq_half_x_solution_l180_18016


namespace NUMINAMATH_CALUDE_merchant_discount_l180_18094

/-- Prove that given a 75% markup and a 57.5% profit after discount, the discount offered is 10%. -/
theorem merchant_discount (C : ℝ) (C_pos : C > 0) : 
  let M := 1.75 * C  -- Marked up price (75% markup)
  let S := 1.575 * C -- Selling price (57.5% profit)
  let D := (M - S) / M * 100 -- Discount percentage
  D = 10 := by sorry

end NUMINAMATH_CALUDE_merchant_discount_l180_18094


namespace NUMINAMATH_CALUDE_hilary_corn_shucking_l180_18037

/-- The number of ears of corn per stalk -/
def ears_per_stalk : ℕ := 4

/-- The number of stalks Hilary has -/
def total_stalks : ℕ := 108

/-- The number of kernels on half of the ears -/
def kernels_first_half : ℕ := 500

/-- The additional number of kernels on the other half of the ears -/
def additional_kernels : ℕ := 100

/-- The total number of kernels Hilary has to shuck -/
def total_kernels : ℕ := 
  let total_ears := ears_per_stalk * total_stalks
  let ears_per_half := total_ears / 2
  let kernels_second_half := kernels_first_half + additional_kernels
  ears_per_half * kernels_first_half + ears_per_half * kernels_second_half

theorem hilary_corn_shucking :
  total_kernels = 237600 := by
  sorry

end NUMINAMATH_CALUDE_hilary_corn_shucking_l180_18037


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l180_18090

theorem no_function_satisfies_conditions : ¬∃ (f : ℝ → ℝ), 
  (∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧ 
  (f 1 = 1) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by sorry

end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l180_18090


namespace NUMINAMATH_CALUDE_count_numbers_with_three_is_180_l180_18024

/-- The count of natural numbers from 1 to 1000 that contain the digit 3 at least once -/
def count_numbers_with_three : ℕ :=
  let total_numbers := 1000
  let numbers_without_three := 820
  total_numbers - numbers_without_three

/-- Theorem stating that the count of natural numbers from 1 to 1000 
    containing the digit 3 at least once is equal to 180 -/
theorem count_numbers_with_three_is_180 :
  count_numbers_with_three = 180 := by
  sorry

#eval count_numbers_with_three

end NUMINAMATH_CALUDE_count_numbers_with_three_is_180_l180_18024


namespace NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l180_18055

-- Define a quadratic polynomial
def QuadraticPolynomial (α : Type*) [Field α] := α → α

-- Define the property of having exactly three solutions for (f(x))^3 - 4f(x) = 0
def HasThreeSolutionsCubicMinusFour (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, f x ^ 3 - 4 * f x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Define the property of having exactly two solutions for (f(x))^2 = 1
def HasTwoSolutionsSquaredEqualsOne (f : QuadraticPolynomial ℝ) : Prop :=
  ∃ (y₁ y₂ : ℝ), (∀ y : ℝ, f y ^ 2 = 1 ↔ y = y₁ ∨ y = y₂) ∧ y₁ ≠ y₂

-- State the theorem
theorem quadratic_polynomial_solutions 
  (f : QuadraticPolynomial ℝ) 
  (h : HasThreeSolutionsCubicMinusFour f) : 
  HasTwoSolutionsSquaredEqualsOne f :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_solutions_l180_18055


namespace NUMINAMATH_CALUDE_polynomial_uniqueness_l180_18018

theorem polynomial_uniqueness (P : ℝ → ℝ) : 
  (∀ x, P x = P 0 + P 1 * x + P 3 * x^3) → 
  P (-1) = 3 → 
  ∀ x, P x = 3 + x + x^3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_uniqueness_l180_18018


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l180_18064

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (∀ a b, a > b + 1 → a > b) ∧ 
  (∃ a b, a > b ∧ ¬(a > b + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l180_18064


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l180_18013

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 2 * x^2 - 4 * x - 1
  ∃ x1 x2 : ℝ, x1 = (2 + Real.sqrt 6) / 2 ∧ 
              x2 = (2 - Real.sqrt 6) / 2 ∧ 
              f x1 = 0 ∧ f x2 = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l180_18013


namespace NUMINAMATH_CALUDE_jack_plates_left_l180_18045

def plates_left (flower_plates checked_plates striped_plates : ℕ) : ℕ :=
  let polka_plates := checked_plates ^ 2
  let wave_plates := (4 * checked_plates) / 9
  let smashed_flower := (flower_plates * 10) / 100
  let smashed_checked := (checked_plates * 15) / 100
  let smashed_striped := (striped_plates * 20) / 100
  flower_plates - smashed_flower + checked_plates - smashed_checked + 
  striped_plates - smashed_striped + polka_plates + wave_plates

theorem jack_plates_left : plates_left 6 9 3 = 102 := by
  sorry

end NUMINAMATH_CALUDE_jack_plates_left_l180_18045


namespace NUMINAMATH_CALUDE_borrowed_amounts_proof_l180_18023

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem borrowed_amounts_proof 
  (interest₁ interest₂ interest₃ : ℝ)
  (rate₁ rate₂ rate₃ : ℝ)
  (time₁ time₂ time₃ : ℝ)
  (h₁ : interest₁ = 1500)
  (h₂ : interest₂ = 1500)
  (h₃ : interest₃ = 1500)
  (hr₁ : rate₁ = 0.12)
  (hr₂ : rate₂ = 0.10)
  (hr₃ : rate₃ = 0.05)
  (ht₁ : time₁ = 1)
  (ht₂ : time₂ = 2)
  (ht₃ : time₃ = 3) :
  ∃ (principal₁ principal₂ principal₃ : ℝ),
    simple_interest principal₁ rate₁ time₁ = interest₁ ∧
    simple_interest principal₂ rate₂ time₂ = interest₂ ∧
    simple_interest principal₃ rate₃ time₃ = interest₃ ∧
    principal₁ = 12500 ∧
    principal₂ = 7500 ∧
    principal₃ = 10000 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amounts_proof_l180_18023


namespace NUMINAMATH_CALUDE_pythagorean_diagonal_l180_18093

/-- 
For a right triangle with width 2m (where m ≥ 3 and m is a positive integer) 
and the difference between the diagonal and the height being 2, 
the diagonal is equal to m² - 1.
-/
theorem pythagorean_diagonal (m : ℕ) (h : m ≥ 3) : 
  let width : ℕ := 2 * m
  let diagonal : ℕ := m^2 - 1
  let height : ℕ := diagonal - 2
  width^2 + height^2 = diagonal^2 := by sorry

end NUMINAMATH_CALUDE_pythagorean_diagonal_l180_18093


namespace NUMINAMATH_CALUDE_sum_digits_base7_of_777_l180_18066

-- Define a function to convert a number from base 10 to base 7
def toBase7 (n : ℕ) : List ℕ := sorry

-- Define a function to sum the digits of a number represented as a list
def sumDigits (digits : List ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_digits_base7_of_777 : sumDigits (toBase7 777) = 9 := by sorry

end NUMINAMATH_CALUDE_sum_digits_base7_of_777_l180_18066


namespace NUMINAMATH_CALUDE_kennel_problem_l180_18030

/-- Represents the number of dogs with various accessories in a kennel -/
structure KennelData where
  total : ℕ
  tags : ℕ
  flea_collars : ℕ
  harnesses : ℕ
  tags_and_flea : ℕ
  tags_and_harnesses : ℕ
  flea_and_harnesses : ℕ
  all_three : ℕ

/-- Calculates the number of dogs with no accessories given kennel data -/
def dogs_with_no_accessories (data : KennelData) : ℕ :=
  data.total - (data.tags + data.flea_collars + data.harnesses - 
    data.tags_and_flea - data.tags_and_harnesses - data.flea_and_harnesses + data.all_three)

/-- Theorem stating that given the specific kennel data, 25 dogs have no accessories -/
theorem kennel_problem (data : KennelData) 
    (h1 : data.total = 120)
    (h2 : data.tags = 60)
    (h3 : data.flea_collars = 50)
    (h4 : data.harnesses = 30)
    (h5 : data.tags_and_flea = 20)
    (h6 : data.tags_and_harnesses = 15)
    (h7 : data.flea_and_harnesses = 10)
    (h8 : data.all_three = 5) :
  dogs_with_no_accessories data = 25 := by
  sorry

end NUMINAMATH_CALUDE_kennel_problem_l180_18030


namespace NUMINAMATH_CALUDE_favorite_color_survey_l180_18087

theorem favorite_color_survey (total_students : ℕ) (total_girls : ℕ) 
  (h1 : total_students = 30)
  (h2 : total_girls = 18)
  (h3 : total_students / 2 = total_students - total_girls + total_girls / 3 + 9) :
  9 = total_students - (total_students / 2 + total_girls / 3) :=
by sorry

end NUMINAMATH_CALUDE_favorite_color_survey_l180_18087


namespace NUMINAMATH_CALUDE_special_triangle_properties_l180_18000

/-- A triangle with an inscribed circle of radius 2, where one side is divided into segments of 4 and 6 by the point of tangency -/
structure SpecialTriangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of the first segment of the divided side -/
  a : ℝ
  /-- The length of the second segment of the divided side -/
  b : ℝ
  /-- The radius is 2 -/
  h_r : r = 2
  /-- The first segment is 4 -/
  h_a : a = 4
  /-- The second segment is 6 -/
  h_b : b = 6

/-- The area of the triangle -/
def area (t : SpecialTriangle) : ℝ := 24

/-- The triangle is right-angled -/
def is_right_triangle (t : SpecialTriangle) : Prop :=
  ∃ (x y z : ℝ), x^2 + y^2 = z^2 ∧ 
    ((x = t.a + t.b ∧ y = 2 * t.r ∧ z = t.a + t.b + 2 * t.r) ∨
     (x = t.a + t.b ∧ y = t.a + t.b + 2 * t.r ∧ z = 2 * t.r) ∨
     (x = 2 * t.r ∧ y = t.a + t.b + 2 * t.r ∧ z = t.a + t.b))

theorem special_triangle_properties (t : SpecialTriangle) :
  is_right_triangle t ∧ area t = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l180_18000


namespace NUMINAMATH_CALUDE_solution_of_exponential_equation_l180_18028

theorem solution_of_exponential_equation :
  ∃ x : ℝ, (2 : ℝ)^(x - 3) = 8^(x + 1) ↔ x = -3 := by sorry

end NUMINAMATH_CALUDE_solution_of_exponential_equation_l180_18028


namespace NUMINAMATH_CALUDE_k_range_theorem_l180_18099

theorem k_range_theorem (k : ℝ) : 
  (∀ m : ℝ, 0 < m ∧ m < 3/2 → (2/m) + (1/(3-2*m)) ≥ k^2 + 2*k) → 
  -3 ≤ k ∧ k ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l180_18099


namespace NUMINAMATH_CALUDE_discount_problem_l180_18048

theorem discount_problem (x y : ℝ) : 
  (100 - x / 100 * 100) * (1 - y / 100) = 55 →
  (100 - 55) / 100 * 100 = 45 := by
sorry

end NUMINAMATH_CALUDE_discount_problem_l180_18048


namespace NUMINAMATH_CALUDE_correct_equation_l180_18071

theorem correct_equation : ∃ (a b c : ℕ), (a = 10 ∧ b = 2 ∧ c = 1) →
  (a^b - c ≠ 101) ∧ ((10 * a + b) - c = 101) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l180_18071


namespace NUMINAMATH_CALUDE_min_sum_box_dimensions_l180_18043

theorem min_sum_box_dimensions (a b c : ℕ+) : 
  a * b * c = 3003 → 
  ∀ x y z : ℕ+, x * y * z = 3003 → a + b + c ≤ x + y + z → 
  a + b + c = 45 :=
sorry

end NUMINAMATH_CALUDE_min_sum_box_dimensions_l180_18043


namespace NUMINAMATH_CALUDE_special_polynomial_q_count_l180_18017

/-- A polynomial of degree 4 with specific properties -/
structure SpecialPolynomial where
  o : ℤ
  p : ℤ
  q : ℤ
  roots_distinct : True  -- represents that the roots are distinct
  roots_positive : True  -- represents that the roots are positive
  one_integer_root : True  -- represents that exactly one root is an integer
  integer_root_sum : True  -- represents that the integer root is the sum of two other roots

/-- The number of possible values for q in the special polynomial -/
def count_q_values : ℕ := 1003001

/-- Theorem stating the number of possible q values -/
theorem special_polynomial_q_count :
  ∀ (poly : SpecialPolynomial), count_q_values = 1003001 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_q_count_l180_18017


namespace NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l180_18086

theorem sqrt_fifteen_over_two_equals_half_sqrt_thirty : 
  Real.sqrt (15 / 2) = (1 / 2) * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l180_18086


namespace NUMINAMATH_CALUDE_quarters_addition_theorem_l180_18031

/-- The number of quarters initially in the jar -/
def initial_quarters : ℕ := 267

/-- The value of one quarter in dollars -/
def quarter_value : ℚ := 1/4

/-- The target total value in dollars -/
def target_value : ℚ := 100

/-- The number of quarters to be added -/
def quarters_to_add : ℕ := 133

theorem quarters_addition_theorem :
  (initial_quarters + quarters_to_add : ℚ) * quarter_value = target_value := by
  sorry

end NUMINAMATH_CALUDE_quarters_addition_theorem_l180_18031


namespace NUMINAMATH_CALUDE_polynomial_factorization_l180_18091

def P (x : ℝ) : ℝ := x^4 + 2*x^3 - 23*x^2 + 12*x + 36

theorem polynomial_factorization :
  ∃ (a b c : ℝ),
    (∀ x, P x = (x^2 + a*x + c) * (x^2 + b*x + c)) ∧
    a + b = 2 ∧
    a * b = -35 ∧
    c = 6 ∧
    (∀ x, P x = 0 ↔ x = 2 ∨ x = 3 ∨ x = -1 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l180_18091


namespace NUMINAMATH_CALUDE_ratio_equality_l180_18088

theorem ratio_equality (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a * b ≠ 0) : (a / 5) / (b / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l180_18088


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l180_18072

theorem expression_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ -3 → x ≠ 3 →
  (1 - 1 / (x + 3)) / ((x^2 - 9) / (x^2 + 6*x + 9)) = (x + 2) / (x - 3) ∧
  (2 + 2) / (2 - 3) = -4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l180_18072


namespace NUMINAMATH_CALUDE_unique_modular_equivalence_l180_18010

theorem unique_modular_equivalence :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -2050 [ZMOD 13] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalence_l180_18010


namespace NUMINAMATH_CALUDE_shape_to_square_possible_l180_18054

/-- Represents a shape on a graph paper -/
structure Shape where
  -- Add necessary fields to represent the shape
  -- For example, you might use a list of coordinates

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields to represent a triangle
  -- For example, you might use three points

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square
  -- For example, you might use side length and position

/-- Function to check if a list of triangles can form a square -/
def can_form_square (triangles : List Triangle) : Prop :=
  -- Define the logic to check if triangles can form a square
  sorry

/-- The main theorem stating that the shape can be divided into 5 triangles
    that can be rearranged to form a square -/
theorem shape_to_square_possible (s : Shape) : 
  ∃ (t1 t2 t3 t4 t5 : Triangle), 
    (can_form_square [t1, t2, t3, t4, t5]) ∧ 
    (-- Add condition that t1, t2, t3, t4, t5 are a valid division of s
     sorry) := by
  sorry

end NUMINAMATH_CALUDE_shape_to_square_possible_l180_18054


namespace NUMINAMATH_CALUDE_probability_ten_heads_in_twelve_flips_l180_18053

theorem probability_ten_heads_in_twelve_flips :
  let n : ℕ := 12  -- Total number of coin flips
  let k : ℕ := 10  -- Number of desired heads
  let p : ℚ := 1/2 -- Probability of getting heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1-p)^(n-k) = 66/4096 :=
by sorry

end NUMINAMATH_CALUDE_probability_ten_heads_in_twelve_flips_l180_18053


namespace NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_proof_l180_18042

/-- The cost of an adult ticket for a play, given the following conditions:
  * Child tickets cost 1 dollar
  * 22 people attended the performance
  * Total ticket sales were 50 dollars
  * 18 children attended the play
-/
theorem adult_ticket_cost : ℝ → Prop :=
  fun adult_cost =>
    let child_cost : ℝ := 1
    let total_attendance : ℕ := 22
    let total_sales : ℝ := 50
    let children_attendance : ℕ := 18
    let adult_attendance : ℕ := total_attendance - children_attendance
    adult_cost * adult_attendance + child_cost * children_attendance = total_sales ∧
    adult_cost = 8

/-- Proof of the adult ticket cost theorem -/
theorem adult_ticket_cost_proof : ∃ (cost : ℝ), adult_ticket_cost cost := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_adult_ticket_cost_proof_l180_18042


namespace NUMINAMATH_CALUDE_arithmetic_progression_polynomial_p_l180_18049

/-- A polynomial of the form x^4 + px^2 + qx - 144 with four distinct real roots in arithmetic progression -/
structure ArithmeticProgressionPolynomial where
  p : ℝ
  q : ℝ
  roots : Fin 4 → ℝ
  distinct_roots : ∀ i j, i ≠ j → roots i ≠ roots j
  arithmetic_progression : ∃ (a d : ℝ), ∀ i, roots i = a + i * d
  is_root : ∀ i, (roots i)^4 + p * (roots i)^2 + q * (roots i) - 144 = 0

/-- The value of p in an ArithmeticProgressionPolynomial is -40 -/
theorem arithmetic_progression_polynomial_p (poly : ArithmeticProgressionPolynomial) : poly.p = -40 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_polynomial_p_l180_18049


namespace NUMINAMATH_CALUDE_banana_count_l180_18040

theorem banana_count (apples oranges total : ℕ) (h1 : apples = 9) (h2 : oranges = 15) (h3 : total = 146) :
  ∃ bananas : ℕ, 
    3 * (apples + oranges + bananas) + (apples - 2 + oranges - 2 + bananas - 2) = total ∧ 
    bananas = 52 :=
by sorry

end NUMINAMATH_CALUDE_banana_count_l180_18040


namespace NUMINAMATH_CALUDE_max_sum_is_1446_l180_18077

/-- Represents a cube with six faces --/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- The set of numbers on each cube --/
def cube_numbers : Finset ℕ := {1, 3, 9, 27, 81, 243}

/-- A valid cube has all numbers from cube_numbers --/
def is_valid_cube (c : Cube) : Prop :=
  ∀ n ∈ cube_numbers, ∃ i : Fin 6, c.faces i = n

/-- The sum of visible faces when cubes are stacked --/
def visible_sum (cubes : Fin 4 → Cube) : ℕ :=
  sorry

/-- The maximum possible sum of visible faces --/
def max_visible_sum : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem max_sum_is_1446 :
  ∀ cubes : Fin 4 → Cube,
  (∀ i : Fin 4, is_valid_cube (cubes i)) →
  visible_sum cubes ≤ 1446 ∧
  ∃ optimal_cubes : Fin 4 → Cube,
    (∀ i : Fin 4, is_valid_cube (optimal_cubes i)) ∧
    visible_sum optimal_cubes = 1446 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_is_1446_l180_18077


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l180_18044

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ ∀ n, ¬ p n := by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l180_18044


namespace NUMINAMATH_CALUDE_fruit_count_correct_l180_18085

structure FruitBasket :=
  (plums : ℕ)
  (oranges : ℕ)
  (apples : ℕ)
  (pears : ℕ)
  (cherries : ℕ)

def initial_basket : FruitBasket :=
  { plums := 10, oranges := 8, apples := 12, pears := 6, cherries := 0 }

def exchanges (basket : FruitBasket) : FruitBasket :=
  { plums := basket.plums - 4 + 2,
    oranges := basket.oranges - 3 + 1,
    apples := basket.apples - 5 + 2,
    pears := basket.pears + 1 + 3,
    cherries := basket.cherries + 2 }

def final_basket : FruitBasket :=
  { plums := 8, oranges := 6, apples := 9, pears := 10, cherries := 2 }

theorem fruit_count_correct : exchanges initial_basket = final_basket := by
  sorry

end NUMINAMATH_CALUDE_fruit_count_correct_l180_18085


namespace NUMINAMATH_CALUDE_exists_bound_for_digit_sum_of_factorial_l180_18080

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The statement to be proved -/
theorem exists_bound_for_digit_sum_of_factorial :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (n.factorial) ≥ 10^100 := by
  sorry

end NUMINAMATH_CALUDE_exists_bound_for_digit_sum_of_factorial_l180_18080


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_area_sum_l180_18060

-- Define a parallelogram type
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

-- Define the property of having right or obtuse angles
def has_right_or_obtuse_angles (p : Parallelogram) : Prop :=
  sorry

-- Define the perimeter of the parallelogram
def perimeter (p : Parallelogram) : ℝ :=
  sorry

-- Define the area of the parallelogram
def area (p : Parallelogram) : ℝ :=
  sorry

-- Theorem statement
theorem parallelogram_perimeter_area_sum :
  ∀ p : Parallelogram,
  p.v1 = (6, 3) ∧ p.v2 = (9, 7) ∧ p.v3 = (2, 0) ∧
  has_right_or_obtuse_angles p →
  perimeter p + area p = 48 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_area_sum_l180_18060


namespace NUMINAMATH_CALUDE_range_of_m_minus_one_times_n_minus_one_l180_18027

/-- The function f(x) = |x^2 - 2x - 1| -/
def f (x : ℝ) : ℝ := abs (x^2 - 2*x - 1)

/-- Theorem stating the range of (m-1)(n-1) given the conditions -/
theorem range_of_m_minus_one_times_n_minus_one 
  (m n : ℝ) 
  (h1 : m > n) 
  (h2 : n > 1) 
  (h3 : f m = f n) : 
  0 < (m - 1) * (n - 1) ∧ (m - 1) * (n - 1) < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_minus_one_times_n_minus_one_l180_18027


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l180_18005

theorem polynomial_functional_equation (p : ℝ → ℝ) 
  (h1 : p 3 = 10)
  (h2 : ∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) :
  ∀ x : ℝ, p x = x^2 + 1 := by sorry

end NUMINAMATH_CALUDE_polynomial_functional_equation_l180_18005


namespace NUMINAMATH_CALUDE_stating_min_toothpicks_theorem_l180_18095

/-- Represents a figure made of toothpicks and triangles -/
structure TriangleFigure where
  total_toothpicks : ℕ
  upward_1triangles : ℕ
  downward_1triangles : ℕ
  upward_2triangles : ℕ

/-- 
  Given a TriangleFigure, calculates the minimum number of toothpicks 
  that must be removed to eliminate all triangles
-/
def min_toothpicks_to_remove (figure : TriangleFigure) : ℕ :=
  sorry

/-- 
  Theorem stating that for the given figure, 
  the minimum number of toothpicks to remove is 15
-/
theorem min_toothpicks_theorem (figure : TriangleFigure) 
  (h1 : figure.total_toothpicks = 60)
  (h2 : figure.upward_1triangles = 22)
  (h3 : figure.downward_1triangles = 14)
  (h4 : figure.upward_2triangles = 4) :
  min_toothpicks_to_remove figure = 15 :=
by sorry

end NUMINAMATH_CALUDE_stating_min_toothpicks_theorem_l180_18095


namespace NUMINAMATH_CALUDE_train_length_is_100m_l180_18076

-- Define the given constants
def train_speed : Real := 60  -- km/h
def bridge_length : Real := 80  -- meters
def crossing_time : Real := 10.799136069114471  -- seconds

-- Theorem to prove
theorem train_length_is_100m :
  let speed_ms : Real := train_speed * 1000 / 3600  -- Convert km/h to m/s
  let total_distance : Real := speed_ms * crossing_time
  let train_length : Real := total_distance - bridge_length
  train_length = 100 := by sorry

end NUMINAMATH_CALUDE_train_length_is_100m_l180_18076


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l180_18025

theorem max_sum_of_factors (X Y Z : ℕ) : 
  X > 0 ∧ Y > 0 ∧ Z > 0 →  -- Positive integers
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →  -- Distinct integers
  X * Y * Z = 399 →        -- Product constraint
  X + Y + Z ≤ 29           -- Maximum sum
  := by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l180_18025


namespace NUMINAMATH_CALUDE_quadratic_root_property_l180_18082

theorem quadratic_root_property (m : ℝ) : 
  m^2 - m - 2 = 0 → 2*m^2 - 2*m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l180_18082


namespace NUMINAMATH_CALUDE_translated_points_exponent_l180_18039

/-- Given two points A and B, and their translations A₁ and B₁, prove that a^b = 32 -/
theorem translated_points_exponent (A B A₁ B₁ : ℝ × ℝ) (a b : ℝ) : 
  A = (-1, 3) → 
  B = (2, -3) → 
  A₁ = (a, 1) → 
  B₁ = (5, -b) → 
  A₁.1 - A.1 = 3 → 
  A.2 - A₁.2 = 2 → 
  B₁.1 - B.1 = 3 → 
  B.2 - B₁.2 = 2 → 
  a^b = 32 := by
sorry

end NUMINAMATH_CALUDE_translated_points_exponent_l180_18039


namespace NUMINAMATH_CALUDE_solution_set_implies_b_range_l180_18033

/-- The solution set of the inequality |3x-b| < 4 -/
def SolutionSet (b : ℝ) : Set ℝ :=
  {x : ℝ | |3*x - b| < 4}

/-- The set of integers 1, 2, and 3 -/
def IntegerSet : Set ℝ := {1, 2, 3}

/-- Theorem stating that if the solution set of |3x-b| < 4 is exactly {1, 2, 3}, then 5 < b < 7 -/
theorem solution_set_implies_b_range :
  ∀ b : ℝ, SolutionSet b = IntegerSet → 5 < b ∧ b < 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_b_range_l180_18033


namespace NUMINAMATH_CALUDE_fraction_equivalence_l180_18056

theorem fraction_equivalence (x y z : ℝ) (h1 : 2*x - z ≠ 0) (h2 : z ≠ 0) :
  (2*x + y) / (2*x - z) = y / (-z) ↔ y = -z :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l180_18056


namespace NUMINAMATH_CALUDE_smallest_with_18_divisors_l180_18074

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 18 positive divisors -/
def has_18_divisors (n : ℕ+) : Prop := num_divisors n = 18

theorem smallest_with_18_divisors : 
  (∀ m : ℕ+, m < 180 → ¬(has_18_divisors m)) ∧ has_18_divisors 180 := by sorry

end NUMINAMATH_CALUDE_smallest_with_18_divisors_l180_18074


namespace NUMINAMATH_CALUDE_no_common_complex_root_l180_18073

theorem no_common_complex_root :
  ¬ ∃ (α : ℂ) (a b : ℚ), α^5 - α - 1 = 0 ∧ α^2 + a*α + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_complex_root_l180_18073


namespace NUMINAMATH_CALUDE_arith_geom_seq_sum_ratio_l180_18070

/-- An arithmetic-geometric sequence -/
structure ArithGeomSeq where
  a : ℕ → ℝ

/-- Sum of first n terms of an arithmetic-geometric sequence -/
def sum_n (seq : ArithGeomSeq) (n : ℕ) : ℝ :=
  sorry

theorem arith_geom_seq_sum_ratio (seq : ArithGeomSeq) :
  sum_n seq 6 / sum_n seq 3 = 1 / 2 →
  sum_n seq 9 / sum_n seq 3 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arith_geom_seq_sum_ratio_l180_18070


namespace NUMINAMATH_CALUDE_max_homework_time_l180_18007

/-- The time Max spent on biology homework -/
def biology_time : ℕ := 20

/-- The time Max spent on history homework -/
def history_time : ℕ := 2 * biology_time

/-- The time Max spent on geography homework -/
def geography_time : ℕ := 3 * history_time

/-- The total time Max spent on homework -/
def total_time : ℕ := 180

theorem max_homework_time : 
  biology_time + history_time + geography_time = total_time ∧ 
  biology_time = 20 := by sorry

end NUMINAMATH_CALUDE_max_homework_time_l180_18007


namespace NUMINAMATH_CALUDE_even_function_property_l180_18014

-- Define an even function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x * (x + 1)) :
  ∀ x > 0, f x = x * (x - 1) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l180_18014


namespace NUMINAMATH_CALUDE_complex_multiplication_simplification_l180_18089

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem: For any real number t, (2+t i)(2-t i) = 4 + t^2 -/
theorem complex_multiplication_simplification (t : ℝ) : 
  (2 + t * i) * (2 - t * i) = (4 : ℂ) + t^2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_simplification_l180_18089


namespace NUMINAMATH_CALUDE_negative_five_minus_two_i_in_third_quadrant_l180_18075

/-- A complex number z is in the third quadrant if its real part is negative and its imaginary part is negative. -/
def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The complex number -5-2i is in the third quadrant. -/
theorem negative_five_minus_two_i_in_third_quadrant :
  is_in_third_quadrant (-5 - 2*I) := by
  sorry

end NUMINAMATH_CALUDE_negative_five_minus_two_i_in_third_quadrant_l180_18075
