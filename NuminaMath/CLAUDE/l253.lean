import Mathlib

namespace NUMINAMATH_CALUDE_approximate_root_of_f_l253_25375

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem approximate_root_of_f :
  f 1 = -2 →
  f 1.5 = 0.625 →
  f 1.25 = -0.984 →
  f 1.375 = -0.260 →
  f 1.438 = 0.165 →
  f 1.4065 = -0.052 →
  ∃ (root : ℝ), f root = 0 ∧ |root - 1.43| < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_approximate_root_of_f_l253_25375


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l253_25327

/-- The function f(x) = ax^2 + 4x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x - 1

/-- Predicate indicating that the graph of f has only one common point with the x-axis -/
def has_one_common_point (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Statement: a = -4 is a sufficient but not necessary condition for
    the graph of f to have only one common point with the x-axis -/
theorem sufficient_not_necessary :
  (has_one_common_point (-4)) ∧ 
  (∃ a : ℝ, a ≠ -4 ∧ has_one_common_point a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l253_25327


namespace NUMINAMATH_CALUDE_m_intersect_n_equals_n_l253_25310

open Set

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x^2 - 2*x > 0}
def N : Set ℝ := {x : ℝ | x > 3}

-- Theorem statement
theorem m_intersect_n_equals_n : M ∩ N = N := by
  sorry

end NUMINAMATH_CALUDE_m_intersect_n_equals_n_l253_25310


namespace NUMINAMATH_CALUDE_flour_added_indeterminate_l253_25360

/-- Represents the ingredients in cups -/
structure Ingredients where
  sugar : ℕ
  flour : ℕ
  salt : ℕ

/-- Represents the current state of Mary's baking process -/
structure BakingState where
  recipe : Ingredients
  flour_added : ℕ
  sugar_to_add : ℕ
  salt_to_add : ℕ

/-- The recipe requirements -/
def recipe : Ingredients :=
  { sugar := 11, flour := 6, salt := 9 }

/-- Theorem stating that the amount of flour already added cannot be uniquely determined -/
theorem flour_added_indeterminate (state : BakingState) : 
  state.recipe = recipe → 
  state.sugar_to_add = state.salt_to_add + 2 → 
  ∃ (x y : ℕ), x ≠ y ∧ 
    (∃ (state1 state2 : BakingState), 
      state1.flour_added = x ∧ 
      state2.flour_added = y ∧ 
      state1.recipe = state.recipe ∧ 
      state2.recipe = state.recipe ∧ 
      state1.sugar_to_add = state.sugar_to_add ∧ 
      state2.sugar_to_add = state.sugar_to_add ∧ 
      state1.salt_to_add = state.salt_to_add ∧ 
      state2.salt_to_add = state.salt_to_add) :=
by
  sorry

end NUMINAMATH_CALUDE_flour_added_indeterminate_l253_25360


namespace NUMINAMATH_CALUDE_range_of_a_l253_25316

theorem range_of_a (a : ℝ) : 
  (∀ (x y : ℝ), x ≠ 0 → |x + 1/x| ≥ |a - 2| + Real.sin y) ↔ a ∈ Set.Icc 1 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l253_25316


namespace NUMINAMATH_CALUDE_m_range_when_M_in_fourth_quadrant_l253_25390

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The point M with coordinates dependent on m -/
def M (m : ℝ) : Point :=
  { x := m + 3, y := m - 1 }

/-- Theorem: If M(m) is in the fourth quadrant, then -3 < m < 1 -/
theorem m_range_when_M_in_fourth_quadrant :
  ∀ m : ℝ, in_fourth_quadrant (M m) → -3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_m_range_when_M_in_fourth_quadrant_l253_25390


namespace NUMINAMATH_CALUDE_arc_length_30_degrees_l253_25369

/-- The length of an arc in a circle with radius 3 and central angle 30° is π/2 -/
theorem arc_length_30_degrees (r : ℝ) (θ : ℝ) (L : ℝ) : 
  r = 3 → θ = 30 * π / 180 → L = r * θ → L = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_30_degrees_l253_25369


namespace NUMINAMATH_CALUDE_factor_proof_l253_25395

theorem factor_proof :
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ m : ℕ, 162 = 9 * m) := by
  sorry

end NUMINAMATH_CALUDE_factor_proof_l253_25395


namespace NUMINAMATH_CALUDE_f_lower_bound_l253_25333

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2 / 2 - (a + 1) * x

theorem f_lower_bound :
  ∀ x : ℝ, x > 0 → f (-1) x ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l253_25333


namespace NUMINAMATH_CALUDE_sock_pair_count_l253_25304

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: There are 47 ways to choose a pair of socks of different colors
    from 5 white socks, 4 brown socks, and 3 blue socks -/
theorem sock_pair_count :
  differentColorPairs 5 4 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l253_25304


namespace NUMINAMATH_CALUDE_abs_sum_equals_two_l253_25325

theorem abs_sum_equals_two (x y z : ℕ+) 
  (h : (Int.natAbs (x - y))^2010 + (Int.natAbs (z - x))^2011 = 1) :
  (Int.natAbs (x - y)) + (Int.natAbs (y - z)) + (Int.natAbs (z - x)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_equals_two_l253_25325


namespace NUMINAMATH_CALUDE_house_rent_expenditure_l253_25364

theorem house_rent_expenditure (total_income : ℝ) (petrol_spending : ℝ) 
  (h1 : petrol_spending = 0.3 * total_income)
  (h2 : petrol_spending = 300) : ℝ :=
by
  let remaining_income := total_income - petrol_spending
  let house_rent := 0.14 * remaining_income
  have : house_rent = 98 := by sorry
  exact house_rent

#check house_rent_expenditure

end NUMINAMATH_CALUDE_house_rent_expenditure_l253_25364


namespace NUMINAMATH_CALUDE_zucchini_amount_l253_25367

def eggplant_pounds : ℝ := 5
def eggplant_price : ℝ := 2
def tomato_pounds : ℝ := 4
def tomato_price : ℝ := 3.5
def onion_pounds : ℝ := 3
def onion_price : ℝ := 1
def basil_pounds : ℝ := 1
def basil_price : ℝ := 2.5
def zucchini_price : ℝ := 2
def quarts_yield : ℝ := 4
def quart_price : ℝ := 10

theorem zucchini_amount (zucchini_pounds : ℝ) :
  eggplant_pounds * eggplant_price +
  zucchini_pounds * zucchini_price +
  tomato_pounds * tomato_price +
  onion_pounds * onion_price +
  basil_pounds * basil_price * 2 =
  quarts_yield * quart_price →
  zucchini_pounds = 4 := by sorry

end NUMINAMATH_CALUDE_zucchini_amount_l253_25367


namespace NUMINAMATH_CALUDE_william_marbles_left_l253_25357

/-- Given that William initially has 10 marbles and shares 3 marbles with Theresa,
    prove that William will have 7 marbles left. -/
theorem william_marbles_left (initial_marbles : ℕ) (shared_marbles : ℕ) 
  (h1 : initial_marbles = 10) (h2 : shared_marbles = 3) :
  initial_marbles - shared_marbles = 7 := by
  sorry

end NUMINAMATH_CALUDE_william_marbles_left_l253_25357


namespace NUMINAMATH_CALUDE_most_likely_outcome_l253_25349

def probability_all_same_gender (n : ℕ) : ℚ :=
  (1 / 2) ^ n

def probability_equal_split (n : ℕ) : ℚ :=
  (Nat.choose n (n / 2)) * (1 / 2) ^ n

def probability_four_two_split (n : ℕ) : ℚ :=
  2 * (Nat.choose n 2) * (1 / 2) ^ n

theorem most_likely_outcome (n : ℕ) (h : n = 6) :
  probability_four_two_split n > probability_all_same_gender n ∧
  probability_four_two_split n > probability_equal_split n :=
by sorry

end NUMINAMATH_CALUDE_most_likely_outcome_l253_25349


namespace NUMINAMATH_CALUDE_tabitha_money_proof_l253_25359

def calculate_remaining_money (initial_amount : ℚ) (given_away : ℚ) (investment_percentage : ℚ) (num_items : ℕ) (item_cost : ℚ) : ℚ :=
  let remaining_after_giving := initial_amount - given_away
  let investment_amount := (investment_percentage / 100) * remaining_after_giving
  let remaining_after_investment := remaining_after_giving - investment_amount
  let spent_on_items := (num_items : ℚ) * item_cost
  remaining_after_investment - spent_on_items

theorem tabitha_money_proof :
  calculate_remaining_money 45 10 60 12 0.75 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tabitha_money_proof_l253_25359


namespace NUMINAMATH_CALUDE_part_I_part_II_l253_25326

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + b < 0}

-- Theorem for part I
theorem part_I : A = B 2 (-3) := by sorry

-- Theorem for part II
theorem part_II : 
  ∀ a : ℝ, (A ∩ B a 3 ⊇ B a 3) → a ∈ Set.Icc (-2 * Real.sqrt 3) 4 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l253_25326


namespace NUMINAMATH_CALUDE_unique_zero_in_interval_l253_25315

def f (x : ℝ) := -x^2 + 4*x - 4

theorem unique_zero_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc 1 3 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_zero_in_interval_l253_25315


namespace NUMINAMATH_CALUDE_inscribed_rectangle_height_l253_25344

/-- 
Given a triangle with base b and height h, and a rectangle inscribed in it such that:
1. The base of the rectangle coincides with the base of the triangle
2. The height of the rectangle is half its base
Prove that the height of the rectangle x is equal to bh / (2h + b)
-/
theorem inscribed_rectangle_height (b h : ℝ) (h1 : 0 < b) (h2 : 0 < h) : 
  ∃ x : ℝ, x > 0 ∧ x = b * h / (2 * h + b) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_height_l253_25344


namespace NUMINAMATH_CALUDE_existence_of_mn_l253_25345

theorem existence_of_mn (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ m n : ℕ, m + n < p ∧ p ∣ (2^m * 3^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_mn_l253_25345


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l253_25372

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (l m : Line) (α β : Plane) 
  (h1 : parallel_plane α β) 
  (h2 : perpendicular_line_plane l α) 
  (h3 : parallel_line_plane m β) : 
  perpendicular_line l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l253_25372


namespace NUMINAMATH_CALUDE_equation_solution_implies_expression_value_l253_25336

theorem equation_solution_implies_expression_value :
  ∀ a : ℝ, (∃ x : ℝ, 3 * a - x = x + 2 ∧ x = 2) → a^2 - 2*a + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_implies_expression_value_l253_25336


namespace NUMINAMATH_CALUDE_key_chain_profit_percentage_l253_25318

theorem key_chain_profit_percentage 
  (P : ℝ) 
  (h1 : P = 100) 
  (h2 : P - 50 = 0.5 * P) 
  (h3 : 70 < P) : 
  (P - 70) / P = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_key_chain_profit_percentage_l253_25318


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l253_25368

theorem sqrt_sum_equality : Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l253_25368


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l253_25398

theorem arithmetic_expression_equals_24 : ∃ (f : List ℝ → ℝ), 
  (f [5, 7, 8, 8] = 24) ∧ 
  (∀ x y z w, f [x, y, z, w] = 
    ((x + y) / z) * w ∨ 
    f [x, y, z, w] = (x - y) * z + w) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_24_l253_25398


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l253_25341

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℝ, Real.exp (abs (2 * x + 1)) + m ≥ 0) ↔ m ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l253_25341


namespace NUMINAMATH_CALUDE_james_net_income_l253_25392

def rental_rate : ℕ := 20
def monday_wednesday_hours : ℕ := 8
def friday_hours : ℕ := 6
def sunday_hours : ℕ := 5
def maintenance_cost : ℕ := 35
def insurance_fee : ℕ := 15
def rental_days : ℕ := 4

def total_rental_income : ℕ := 
  rental_rate * (2 * monday_wednesday_hours + friday_hours + sunday_hours)

def total_expenses : ℕ := maintenance_cost + insurance_fee * rental_days

def net_income : ℕ := total_rental_income - total_expenses

theorem james_net_income : net_income = 445 := by
  sorry

end NUMINAMATH_CALUDE_james_net_income_l253_25392


namespace NUMINAMATH_CALUDE_q_coordinates_is_rectangle_l253_25343

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by four points -/
structure Rectangle where
  O : Point
  P : Point
  Q : Point
  R : Point

/-- Definition of our specific rectangle -/
def our_rectangle : Rectangle :=
  { O := { x := 0, y := 0 }
  , P := { x := 0, y := 3 }
  , R := { x := 5, y := 0 }
  , Q := { x := 5, y := 3 } }

/-- Theorem: The coordinates of Q in our_rectangle are (5,3) -/
theorem q_coordinates :
  our_rectangle.Q.x = 5 ∧ our_rectangle.Q.y = 3 := by
  sorry

/-- Theorem: our_rectangle is indeed a rectangle -/
theorem is_rectangle (rect : Rectangle) : 
  (rect.O.x = rect.P.x ∧ rect.O.y = rect.R.y) →
  (rect.Q.x = rect.R.x ∧ rect.Q.y = rect.P.y) →
  (rect.P.x - rect.O.x)^2 + (rect.P.y - rect.O.y)^2 =
  (rect.R.x - rect.O.x)^2 + (rect.R.y - rect.O.y)^2 →
  True := by
  sorry

end NUMINAMATH_CALUDE_q_coordinates_is_rectangle_l253_25343


namespace NUMINAMATH_CALUDE_actual_speed_is_30_l253_25313

/-- Given:
  1. Increasing speed by 10 mph reduces time by 1/4
  2. Increasing speed by 20 mph reduces time by an additional 1/3
  Prove that the actual average speed is 30 mph
-/
theorem actual_speed_is_30 (v : ℝ) (t : ℝ) (d : ℝ) :
  (d = v * t) →
  (d / (v + 10) = 3 / 4 * t) →
  (d / (v + 20) = 1 / 2 * t) →
  v = 30 := by
  sorry

end NUMINAMATH_CALUDE_actual_speed_is_30_l253_25313


namespace NUMINAMATH_CALUDE_sequence_problem_l253_25348

/-- Given two sequences {a_n} and {b_n}, where:
    1) a_1 = 1
    2) {b_n} is a geometric sequence
    3) For all n, b_n = a_(n+1) / a_n
    4) b_10 * b_11 = 2016^(1/10)
    Prove that a_21 = 2016 -/
theorem sequence_problem (a b : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n)
  (h3 : ∀ n : ℕ, b n = a (n + 1) / a n)
  (h4 : b 10 * b 11 = 2016^(1/10)) :
  a 21 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l253_25348


namespace NUMINAMATH_CALUDE_average_marks_proof_l253_25305

/-- Given the marks in three subjects, prove that the average is 75 -/
theorem average_marks_proof (physics chemistry mathematics : ℝ) 
  (h1 : (physics + mathematics) / 2 = 90)
  (h2 : (physics + chemistry) / 2 = 70)
  (h3 : physics = 95) :
  (physics + chemistry + mathematics) / 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l253_25305


namespace NUMINAMATH_CALUDE_tv_watching_time_conversion_l253_25330

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Define the number of hours Logan watched TV
def hours_watched : ℕ := 5

-- Theorem to prove
theorem tv_watching_time_conversion :
  hours_watched * minutes_per_hour = 300 := by
  sorry

end NUMINAMATH_CALUDE_tv_watching_time_conversion_l253_25330


namespace NUMINAMATH_CALUDE_right_triangle_check_triangle_sets_check_l253_25361

theorem right_triangle_check (a b c : ℝ) : Prop :=
  (a * a + b * b = c * c) ∨ (a * a + c * c = b * b) ∨ (b * b + c * c = a * a)

theorem triangle_sets_check : 
  right_triangle_check 1 (Real.sqrt 2) (Real.sqrt 3) ∧
  right_triangle_check 6 8 10 ∧
  right_triangle_check 5 12 13 ∧
  ¬(right_triangle_check (Real.sqrt 3) 2 (Real.sqrt 5)) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_check_triangle_sets_check_l253_25361


namespace NUMINAMATH_CALUDE_quadratic_minimum_l253_25338

/-- The function f(x) = x^2 - px + q reaches its minimum when x = p/2, given p > 0 and q > 0 -/
theorem quadratic_minimum (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let f : ℝ → ℝ := fun x ↦ x^2 - p*x + q
  ∃ (x_min : ℝ), x_min = p/2 ∧ ∀ (x : ℝ), f x_min ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l253_25338


namespace NUMINAMATH_CALUDE_seven_people_arrangement_count_l253_25302

def total_arrangements (n : ℕ) : ℕ := n.factorial

def adjacent_pair_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def front_restricted_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def end_restricted_arrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

def front_and_end_restricted_arrangements (n : ℕ) : ℕ := (n - 2).factorial * 2

theorem seven_people_arrangement_count : 
  total_arrangements 6 - 
  front_restricted_arrangements 6 - 
  end_restricted_arrangements 6 + 
  front_and_end_restricted_arrangements 6 = 1008 :=
by sorry

end NUMINAMATH_CALUDE_seven_people_arrangement_count_l253_25302


namespace NUMINAMATH_CALUDE_A_and_B_complementary_l253_25306

-- Define the sample space for a die toss
def DieOutcome := Fin 6

-- Define events A, B, and C
def eventA (outcome : DieOutcome) : Prop := outcome.val ≤ 3
def eventB (outcome : DieOutcome) : Prop := outcome.val ≥ 4
def eventC (outcome : DieOutcome) : Prop := outcome.val % 2 = 1

-- Theorem stating that A and B are complementary events
theorem A_and_B_complementary :
  ∀ (outcome : DieOutcome), eventA outcome ↔ ¬ eventB outcome :=
by sorry

end NUMINAMATH_CALUDE_A_and_B_complementary_l253_25306


namespace NUMINAMATH_CALUDE_triangle_value_proof_l253_25379

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

theorem triangle_value_proof :
  ∀ (triangle : Nat),
  (triangle < 7) →
  (triangle < 9) →
  (base_to_decimal [triangle, 5] 7 = base_to_decimal [3, triangle] 9) →
  triangle = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_proof_l253_25379


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l253_25339

/-- The decimal representation of 0.0808... -/
def repeating_decimal_08 : ℚ := 8 / 99

/-- The decimal representation of 0.3636... -/
def repeating_decimal_36 : ℚ := 36 / 99

/-- The product of 0.0808... and 0.3636... is equal to 288/9801 -/
theorem product_of_repeating_decimals : 
  repeating_decimal_08 * repeating_decimal_36 = 288 / 9801 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l253_25339


namespace NUMINAMATH_CALUDE_hat_count_l253_25328

/-- The number of hats in the box -/
def num_hats : ℕ := 3

/-- The set of all hats in the box -/
def Hats : Type := Fin num_hats

/-- A hat is red -/
def is_red : Hats → Prop := sorry

/-- A hat is blue -/
def is_blue : Hats → Prop := sorry

/-- A hat is yellow -/
def is_yellow : Hats → Prop := sorry

/-- All but 2 hats are red -/
axiom red_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_red h

/-- All but 2 hats are blue -/
axiom blue_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_blue h

/-- All but 2 hats are yellow -/
axiom yellow_condition : ∃ (a b : Hats), a ≠ b ∧ ∀ (h : Hats), h ≠ a ∧ h ≠ b → is_yellow h

/-- The main theorem: There are exactly 3 hats in the box -/
theorem hat_count : num_hats = 3 := by sorry

end NUMINAMATH_CALUDE_hat_count_l253_25328


namespace NUMINAMATH_CALUDE_seven_abba_divisible_by_eleven_l253_25362

theorem seven_abba_divisible_by_eleven (A : Nat) :
  A < 10 →
  (∃ B : Nat, B < 10 ∧ (70000 + A * 1000 + B * 100 + B * 10 + A) % 11 = 0) ↔
  A = 7 := by
sorry

end NUMINAMATH_CALUDE_seven_abba_divisible_by_eleven_l253_25362


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l253_25309

/-- A rectangular prism with three distinct dimensions -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- The number of face diagonals in a rectangular prism -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

/-- Theorem: A rectangular prism with three distinct dimensions has 16 total diagonals -/
theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l253_25309


namespace NUMINAMATH_CALUDE_euler_product_theorem_l253_25314

theorem euler_product_theorem (z₁ z₂ : ℂ) :
  z₁ = Complex.exp (Complex.I * (Real.pi / 3)) →
  z₂ = Complex.exp (Complex.I * (Real.pi / 6)) →
  z₁ * z₂ = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_euler_product_theorem_l253_25314


namespace NUMINAMATH_CALUDE_fraction_difference_product_l253_25323

theorem fraction_difference_product : 
  let a : ℚ := 1/2
  let b : ℚ := 1/5
  a - b = 3 * (a * b) := by sorry

end NUMINAMATH_CALUDE_fraction_difference_product_l253_25323


namespace NUMINAMATH_CALUDE_fractional_equation_root_l253_25363

theorem fractional_equation_root (x m : ℝ) : 
  (∃ x, x / (x - 2) - 2 = m / (x - 2)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l253_25363


namespace NUMINAMATH_CALUDE_product_of_digits_is_64_l253_25399

/-- Represents a number in different bases -/
structure NumberInBases where
  base10 : ℕ
  b : ℕ
  base_b : ℕ
  base_b_plus_2 : ℕ

/-- Calculates the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the product of digits of N is 64 -/
theorem product_of_digits_is_64 (N : NumberInBases) 
  (h1 : N.base_b = 503)
  (h2 : N.base_b_plus_2 = 305)
  (h3 : N.b > 0) : 
  productOfDigits N.base10 = 64 := by sorry

end NUMINAMATH_CALUDE_product_of_digits_is_64_l253_25399


namespace NUMINAMATH_CALUDE_rectangular_solid_length_l253_25354

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (rs : RectangularSolid) : ℝ :=
  2 * (rs.length * rs.width + rs.length * rs.depth + rs.width * rs.depth)

/-- Theorem: The length of a rectangular solid with width 4, depth 1, and surface area 58 is 5 -/
theorem rectangular_solid_length :
  ∃ (rs : RectangularSolid),
    rs.width = 4 ∧
    rs.depth = 1 ∧
    surfaceArea rs = 58 ∧
    rs.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_length_l253_25354


namespace NUMINAMATH_CALUDE_johns_brother_age_l253_25347

theorem johns_brother_age :
  ∀ (john_age brother_age : ℕ),
  john_age = 6 * brother_age - 4 →
  john_age + brother_age = 10 →
  brother_age = 2 := by
sorry

end NUMINAMATH_CALUDE_johns_brother_age_l253_25347


namespace NUMINAMATH_CALUDE_f_properties_l253_25373

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

-- State the theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 1) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l253_25373


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l253_25301

theorem correct_average_after_error_correction (numbers : List ℝ) 
  (h1 : numbers.length = 15)
  (h2 : numbers.sum / numbers.length = 20)
  (h3 : ∃ a b c, a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ 
               a = 35 ∧ b = 60 ∧ c = 25) :
  let corrected_numbers := numbers.map (fun x => 
    if x = 35 then 45 else if x = 60 then 58 else if x = 25 then 30 else x)
  corrected_numbers.sum / corrected_numbers.length = 20.8666666667 := by
sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l253_25301


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l253_25353

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_condition (a : ℝ) :
  IsPureImaginary ((a^2 - 1) + (a - 1) * Complex.I) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l253_25353


namespace NUMINAMATH_CALUDE_expression_simplification_l253_25324

theorem expression_simplification 
  (a b c x y : ℝ) : 
  (b*x*(c^2*x^2 + a^2*x^2 + 2*a^2*y^2 + b^2*y^2) + 
   a*y*(c^2*x^2 + a^2*x^2 + 2*b^2*x^2 + b^2*y^2)) / (b*x - a*y) = 
  (c*x + a*x + b*y)^2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l253_25324


namespace NUMINAMATH_CALUDE_no_hexagon_with_special_point_l253_25388

/-- A convex hexagon is represented by its vertices -/
def ConvexHexagon := Fin 6 → ℝ × ℝ

/-- Check if a hexagon is convex -/
def is_convex (h : ConvexHexagon) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is inside a hexagon -/
def is_inside (p : ℝ × ℝ) (h : ConvexHexagon) : Prop := sorry

/-- The main theorem -/
theorem no_hexagon_with_special_point :
  ¬ ∃ (h : ConvexHexagon) (m : ℝ × ℝ),
    is_convex h ∧
    (∀ i : Fin 5, distance (h i) (h (i.succ)) > 1) ∧
    distance (h 5) (h 0) > 1 ∧
    is_inside m h ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
sorry

end NUMINAMATH_CALUDE_no_hexagon_with_special_point_l253_25388


namespace NUMINAMATH_CALUDE_ten_people_leaders_and_committee_l253_25320

/-- The number of ways to choose a president, vice-president, and committee from a group --/
def choose_leaders_and_committee (n : ℕ) : ℕ :=
  n * (n - 1) * Nat.choose (n - 2) 3

/-- The theorem stating the number of ways to choose leaders and committee from 10 people --/
theorem ten_people_leaders_and_committee :
  choose_leaders_and_committee 10 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_ten_people_leaders_and_committee_l253_25320


namespace NUMINAMATH_CALUDE_bracket_mult_equation_solution_l253_25389

-- Define the operation
def bracket_mult (a b c d : ℝ) : ℝ := a * c - b * d

-- State the theorem
theorem bracket_mult_equation_solution :
  ∃ (x : ℝ), (bracket_mult (-x) 3 (x - 2) (-6) = 10) ∧ (x = 4 ∨ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_bracket_mult_equation_solution_l253_25389


namespace NUMINAMATH_CALUDE_students_just_passed_l253_25386

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ)
  (h_total : total = 300)
  (h_first_div : first_div_percent = 30 / 100)
  (h_second_div : second_div_percent = 54 / 100)
  (h_all_passed : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 48 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l253_25386


namespace NUMINAMATH_CALUDE_inequality_proof_l253_25381

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + z^2 + x^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l253_25381


namespace NUMINAMATH_CALUDE_value_std_dev_below_mean_l253_25346

def mean : ℝ := 16.2
def std_dev : ℝ := 2.3
def value : ℝ := 11.6

theorem value_std_dev_below_mean : 
  (mean - value) / std_dev = 2 := by sorry

end NUMINAMATH_CALUDE_value_std_dev_below_mean_l253_25346


namespace NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_40_l253_25351

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_consecutive_nonprime_less_than_40 
  (a b c d e : ℕ) 
  (h1 : a + 1 = b)
  (h2 : b + 1 = c)
  (h3 : c + 1 = d)
  (h4 : d + 1 = e)
  (h5 : 10 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < 40)
  (h6 : ¬is_prime a ∧ ¬is_prime b ∧ ¬is_prime c ∧ ¬is_prime d ∧ ¬is_prime e) :
  e = 36 :=
sorry

end NUMINAMATH_CALUDE_largest_consecutive_nonprime_less_than_40_l253_25351


namespace NUMINAMATH_CALUDE_subset_implies_lower_bound_l253_25385

/-- Given sets A = [1, 4) and B = (-∞, a), if A ⊂ B, then a ≥ 4 -/
theorem subset_implies_lower_bound (a : ℝ) :
  let A := { x : ℝ | 1 ≤ x ∧ x < 4 }
  let B := { x : ℝ | x < a }
  A ⊆ B → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_lower_bound_l253_25385


namespace NUMINAMATH_CALUDE_triangle_problem_l253_25358

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_problem (a b c : ℝ) (h1 : Triangle a b c)
    (h2 : b * Real.cos C + c / 2 = a)
    (h3 : b = Real.sqrt 13)
    (h4 : a + c = 4) :
    Real.cos B = 1 / 2 ∧ 
    (1 / 2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l253_25358


namespace NUMINAMATH_CALUDE_money_left_l253_25331

/-- The amount of money Norris saved in September -/
def september_savings : ℕ := 29

/-- The amount of money Norris saved in October -/
def october_savings : ℕ := 25

/-- The amount of money Norris saved in November -/
def november_savings : ℕ := 31

/-- The amount of money Hugo spent on an online game -/
def online_game_cost : ℕ := 75

/-- The theorem stating how much money Norris has left -/
theorem money_left : 
  september_savings + october_savings + november_savings - online_game_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l253_25331


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l253_25321

def A : Set ℝ := {x | 6 / (x + 1) ≥ 1}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

theorem problem_1 : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

theorem problem_2 : ∃ m : ℝ, A ∩ B m = {x | -1 < x ∧ x < 4} ∧ m = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l253_25321


namespace NUMINAMATH_CALUDE_red_marbles_count_l253_25322

theorem red_marbles_count (red green yellow different total : ℕ) : 
  green = 3 * red →
  yellow = green / 5 →
  total = 3 * green →
  different = 88 →
  total = red + green + yellow + different →
  red = 12 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l253_25322


namespace NUMINAMATH_CALUDE_property_width_l253_25380

/-- Proves that the width of a rectangular property is 1000 feet given specific conditions -/
theorem property_width (property_length : ℝ) (garden_area : ℝ) 
  (h1 : property_length = 2250)
  (h2 : garden_area = 28125)
  (h3 : ∃ (property_width : ℝ), 
    garden_area = (property_width / 8) * (property_length / 10)) :
  ∃ (property_width : ℝ), property_width = 1000 := by
  sorry

end NUMINAMATH_CALUDE_property_width_l253_25380


namespace NUMINAMATH_CALUDE_max_elements_in_S_l253_25332

theorem max_elements_in_S (A : Finset ℝ) (h_card : A.card = 100) (h_pos : ∀ a ∈ A, a > 0) :
  let S := {p : ℝ × ℝ | p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 ∈ A}
  (Finset.filter (fun p => p.1 ∈ A ∧ p.2 ∈ A ∧ p.1 - p.2 ∈ A) (A.product A)).card ≤ 4950 :=
by sorry

end NUMINAMATH_CALUDE_max_elements_in_S_l253_25332


namespace NUMINAMATH_CALUDE_metal_argument_is_deductive_l253_25391

-- Define the structure of a logical argument
structure Argument where
  premises : List String
  conclusion : String

-- Define the property of being deductive
def is_deductive (arg : Argument) : Prop :=
  ∀ (world : Type) (interpretation : String → world → Prop),
    (∀ p ∈ arg.premises, ∀ w, interpretation p w) →
    (∀ w, interpretation arg.conclusion w)

-- Define the argument about metals and uranium
def metal_argument : Argument :=
  { premises := ["All metals can conduct electricity", "Uranium is a metal"],
    conclusion := "Uranium can conduct electricity" }

-- Theorem statement
theorem metal_argument_is_deductive :
  is_deductive metal_argument :=
sorry

end NUMINAMATH_CALUDE_metal_argument_is_deductive_l253_25391


namespace NUMINAMATH_CALUDE_invisible_square_exists_l253_25382

theorem invisible_square_exists (n : ℕ) : 
  ∃ (a b : ℤ), ∀ (i j : ℕ), i < n → j < n → Nat.gcd (Int.toNat (a + i)) (Int.toNat (b + j)) > 1 := by
  sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l253_25382


namespace NUMINAMATH_CALUDE_dog_adult_weights_l253_25396

-- Define the dog breeds
inductive DogBreed
| GoldenRetriever
| Labrador
| Poodle

-- Define the weight progression function
def weightProgression (breed : DogBreed) : ℕ → ℕ
| 0 => match breed with
  | DogBreed.GoldenRetriever => 6
  | DogBreed.Labrador => 8
  | DogBreed.Poodle => 4
| 1 => match breed with
  | DogBreed.GoldenRetriever => 12
  | DogBreed.Labrador => 24
  | DogBreed.Poodle => 16
| 2 => match breed with
  | DogBreed.GoldenRetriever => 24
  | DogBreed.Labrador => 36
  | DogBreed.Poodle => 32
| 3 => match breed with
  | DogBreed.GoldenRetriever => 48
  | DogBreed.Labrador => 72
  | DogBreed.Poodle => 32
| _ => 0

-- Define the final weight increase function
def finalWeightIncrease (breed : DogBreed) : ℕ :=
  match breed with
  | DogBreed.GoldenRetriever => 30
  | DogBreed.Labrador => 30
  | DogBreed.Poodle => 20

-- Define the adult weight function
def adultWeight (breed : DogBreed) : ℕ :=
  weightProgression breed 3 + finalWeightIncrease breed

-- Theorem statement
theorem dog_adult_weights :
  (adultWeight DogBreed.GoldenRetriever = 78) ∧
  (adultWeight DogBreed.Labrador = 102) ∧
  (adultWeight DogBreed.Poodle = 52) := by
  sorry

end NUMINAMATH_CALUDE_dog_adult_weights_l253_25396


namespace NUMINAMATH_CALUDE_sin_cos_square_identity_l253_25312

theorem sin_cos_square_identity (α : ℝ) : (Real.sin α + Real.cos α)^2 = 1 + Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_square_identity_l253_25312


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l253_25378

/-- Represents the helmet sales scenario -/
structure HelmetSales where
  initialPrice : ℝ
  initialSales : ℝ
  priceReductionEffect : ℝ
  costPrice : ℝ

/-- Calculates the number of helmets sold after a price reduction -/
def helmetsSold (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  hs.initialSales + hs.priceReductionEffect * priceReduction

/-- Calculates the monthly profit -/
def monthlyProfit (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  (hs.initialPrice - priceReduction - hs.costPrice) * (helmetsSold hs priceReduction)

/-- The main theorem about helmet sales -/
theorem helmet_sales_theorem (hs : HelmetSales) 
    (h1 : hs.initialPrice = 80)
    (h2 : hs.initialSales = 200)
    (h3 : hs.priceReductionEffect = 20)
    (h4 : hs.costPrice = 50) : 
    (helmetsSold hs 10 = 400 ∧ monthlyProfit hs 10 = 8000) ∧
    ∃ x, x > 0 ∧ monthlyProfit hs x = 7500 ∧ hs.initialPrice - x = 65 := by
  sorry


end NUMINAMATH_CALUDE_helmet_sales_theorem_l253_25378


namespace NUMINAMATH_CALUDE_valid_stacks_count_l253_25387

/-- Represents a card with a color and number -/
structure Card where
  color : Nat
  number : Nat

/-- Represents a stack of cards -/
def Stack := List Card

/-- Checks if a stack is valid according to the rules -/
def isValidStack (stack : Stack) : Bool :=
  sorry

/-- Generates all possible stacks -/
def generateStacks : List Stack :=
  sorry

/-- Counts the number of valid stacking sequences -/
def countValidStacks : Nat :=
  (generateStacks.filter isValidStack).length

/-- The main theorem stating that the number of valid stacking sequences is 6 -/
theorem valid_stacks_count :
  let redCards := [1, 2, 3, 4]
  let blueCards := [2, 3, 4]
  let greenCards := [5, 6, 7]
  countValidStacks = 6 := by
  sorry

end NUMINAMATH_CALUDE_valid_stacks_count_l253_25387


namespace NUMINAMATH_CALUDE_point_d_from_c_l253_25397

/-- Given two points C and D in the Cartesian coordinate system, prove that D is obtained from C by moving 3 units downwards -/
theorem point_d_from_c (C D : ℝ × ℝ) : 
  C = (1, 2) → D = (1, -1) → 
  (C.2 - D.2 = 3) ∧ (D.2 < C.2) := by
  sorry

end NUMINAMATH_CALUDE_point_d_from_c_l253_25397


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l253_25383

-- Proposition A
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧
  (∃ a, 1 / a < 1 ∧ ¬(a > 1)) :=
sorry

-- Proposition D
theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b, a ≠ 0 ∧ a * b = 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l253_25383


namespace NUMINAMATH_CALUDE_munchausen_polygon_exists_l253_25300

-- Define a polygon as a set of points in the plane
def Polygon : Type := Set (ℝ × ℝ)

-- Define a point as a pair of real numbers
def Point : Type := ℝ × ℝ

-- Define a line as a set of points satisfying a linear equation
def Line : Type := Set (ℝ × ℝ)

-- Define what it means for a point to be inside a polygon
def inside (p : Point) (poly : Polygon) : Prop := sorry

-- Define what it means for a line to divide a polygon
def divides (l : Line) (poly : Polygon) : Prop := sorry

-- Define what it means for a line to pass through a point
def passes_through (l : Line) (p : Point) : Prop := sorry

-- Count the number of polygons resulting from dividing a polygon by a line
def count_divisions (l : Line) (poly : Polygon) : ℕ := sorry

-- The main theorem
theorem munchausen_polygon_exists :
  ∃ (P : Polygon) (O : Point),
    inside O P ∧
    ∀ (L : Line), passes_through L O →
      count_divisions L P = 3 := by sorry

end NUMINAMATH_CALUDE_munchausen_polygon_exists_l253_25300


namespace NUMINAMATH_CALUDE_min_sum_fraction_l253_25355

theorem min_sum_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (7 * a) ≥ 3 / Real.rpow 105 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_fraction_l253_25355


namespace NUMINAMATH_CALUDE_angle_Z_measure_l253_25340

-- Define the triangle and its angles
def Triangle (X Y W Z : ℝ) : Prop :=
  -- Conditions
  X = 34 ∧ Y = 53 ∧ W = 43 ∧
  -- Additional properties of a triangle
  X > 0 ∧ Y > 0 ∧ W > 0 ∧ Z > 0 ∧
  -- Sum of angles in the larger triangle is 180°
  X + Y + W + Z = 180

-- Theorem statement
theorem angle_Z_measure (X Y W Z : ℝ) (h : Triangle X Y W Z) : Z = 130 := by
  sorry

end NUMINAMATH_CALUDE_angle_Z_measure_l253_25340


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l253_25334

theorem quadratic_equation_with_given_roots :
  ∀ (a b c : ℝ), a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = -2 ∨ x = 3) →
  a * x^2 + b * x + c = x^2 - x - 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_roots_l253_25334


namespace NUMINAMATH_CALUDE_total_cost_is_985_l253_25394

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℚ := 3.75

/-- The additional cost of a train ride compared to a bus ride -/
def train_extra_cost : ℚ := 2.35

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℚ := bus_cost + (bus_cost + train_extra_cost)

/-- Theorem stating that the total cost of one train ride and one bus ride is $9.85 -/
theorem total_cost_is_985 : total_cost = 9.85 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_985_l253_25394


namespace NUMINAMATH_CALUDE_movies_watched_undetermined_l253_25308

/-- Represents the "Crazy Silly School" series -/
structure CrazySillySchool where
  total_movies : ℕ
  total_books : ℕ
  books_read : ℕ
  movie_book_difference : ℕ

/-- The conditions of the problem -/
def series : CrazySillySchool :=
  { total_movies := 17
  , total_books := 11
  , books_read := 13
  , movie_book_difference := 6 }

/-- Predicate to check if the number of movies watched can be determined -/
def can_determine_movies_watched (s : CrazySillySchool) : Prop :=
  ∃! n : ℕ, n ≤ s.total_movies

/-- Theorem stating that it's impossible to determine the number of movies watched -/
theorem movies_watched_undetermined (s : CrazySillySchool) 
  (h1 : s.total_movies = s.total_books + s.movie_book_difference)
  (h2 : s.books_read ≤ s.total_books) :
  ¬(can_determine_movies_watched s) :=
sorry

end NUMINAMATH_CALUDE_movies_watched_undetermined_l253_25308


namespace NUMINAMATH_CALUDE_min_expression_proof_l253_25337

theorem min_expression_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) ≥ 3 ∧
  (((x^2 * y * z) / 324 = (144 * y) / (x * z) ∧ (144 * y) / (x * z) = 9 / (4 * x * y^2)) →
    z / (16 * y) + x / 9 ≥ 2) ∧
  ((x^2 * y * z) / 324 + (144 * y) / (x * z) + 9 / (4 * x * y^2) = 3 ∧
   z / (16 * y) + x / 9 = 2) ↔ (x = 9 ∧ y = 1/2 ∧ z = 16) := by
  sorry

#check min_expression_proof

end NUMINAMATH_CALUDE_min_expression_proof_l253_25337


namespace NUMINAMATH_CALUDE_correct_distinct_arrangements_l253_25384

/-- The number of distinct arrangements to distribute 5 students into two dormitories,
    with each dormitory accommodating at least 2 students. -/
def distinct_arrangements : ℕ := 20

/-- The total number of students to be distributed. -/
def total_students : ℕ := 5

/-- The number of dormitories. -/
def num_dormitories : ℕ := 2

/-- The minimum number of students that must be in each dormitory. -/
def min_students_per_dormitory : ℕ := 2

/-- Theorem stating that the number of distinct arrangements is correct. -/
theorem correct_distinct_arrangements :
  distinct_arrangements = 20 ∧
  total_students = 5 ∧
  num_dormitories = 2 ∧
  min_students_per_dormitory = 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_distinct_arrangements_l253_25384


namespace NUMINAMATH_CALUDE_white_box_exists_l253_25335

def total_balls : ℕ := 120
def black_balls : ℕ := 60
def white_balls : ℕ := 60
def num_boxes : ℕ := 20
def balls_per_box : ℕ := 6
def first_boxes : ℕ := 14

theorem white_box_exists (h1 : total_balls = black_balls + white_balls)
                         (h2 : total_balls = num_boxes * balls_per_box)
                         (h3 : ∀ i ∈ Finset.range first_boxes,
                               ∃ b w : ℕ, b + w = balls_per_box ∧ b > w) :
  ∃ j ∈ Finset.range (num_boxes - first_boxes),
    ∀ k ∈ Finset.range balls_per_box,
      -- All balls in box j + first_boxes are white
      black_balls ≤ first_boxes * (balls_per_box / 2 + 1) + j :=
by sorry

end NUMINAMATH_CALUDE_white_box_exists_l253_25335


namespace NUMINAMATH_CALUDE_tan_four_theta_l253_25311

theorem tan_four_theta (θ : Real) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_theta_l253_25311


namespace NUMINAMATH_CALUDE_system_solution_l253_25329

theorem system_solution (x y : Real) : 
  (Real.sin x)^2 + (Real.cos y)^2 = y^4 ∧ 
  (Real.sin y)^2 + (Real.cos x)^2 = x^2 → 
  (x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l253_25329


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l253_25371

/-- A circle with center O and radius 3 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A line m containing point P -/
structure Line :=
  (m : Set (ℝ × ℝ))
  (P : ℝ × ℝ)
  (h_P_on_m : P ∈ m)

/-- The distance between two points in ℝ² -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

/-- Defines what it means for a line to be tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ (Q : ℝ × ℝ), Q ∈ l.m ∧ distance Q c.O = c.radius ∧
  ∀ (R : ℝ × ℝ), R ∈ l.m → R ≠ Q → distance R c.O > c.radius

theorem line_tangent_to_circle (c : Circle) (l : Line) :
  distance l.P c.O = c.radius →
  is_tangent l c :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l253_25371


namespace NUMINAMATH_CALUDE_triangle_area_function_l253_25342

theorem triangle_area_function (A B C : ℝ) (a b c : ℝ) (x y : ℝ) :
  -- Given conditions
  A = π / 6 →
  a = 2 →
  0 < x →
  x < 5 * π / 6 →
  B = x →
  C = 5 * π / 6 - x →
  -- Area function
  y = 4 * Real.sin x * Real.sin (5 * π / 6 - x) →
  -- Prove
  0 < y ∧ y ≤ 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_function_l253_25342


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l253_25317

theorem trigonometric_simplification (x : ℝ) :
  (1 + Real.sin x + Real.cos x + Real.sqrt 2 * Real.sin x * Real.cos x) /
  (1 - Real.sin x + Real.cos x - Real.sqrt 2 * Real.sin x * Real.cos x) =
  1 + (Real.sqrt 2 - 1) * Real.tan (x / 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l253_25317


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l253_25356

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r)) = 81 * (a * r^4 / (1 - r)) → r = 1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l253_25356


namespace NUMINAMATH_CALUDE_tetrahedron_analogy_l253_25393

-- Define the types of reasoning
inductive ReasoningType
  | Deductive
  | Inductive
  | Analogy

-- Define a structure for a reasoning example
structure ReasoningExample where
  description : String
  type : ReasoningType

-- Define the specific example we're interested in
def tetrahedronExample : ReasoningExample :=
  { description := "Inferring the properties of a tetrahedron in space from the properties of a plane triangle"
  , type := ReasoningType.Analogy }

-- Theorem statement
theorem tetrahedron_analogy :
  tetrahedronExample.type = ReasoningType.Analogy :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_analogy_l253_25393


namespace NUMINAMATH_CALUDE_boxes_sold_l253_25376

theorem boxes_sold (initial boxes_left : ℕ) (h : initial ≥ boxes_left) :
  initial - boxes_left = initial - boxes_left :=
by sorry

end NUMINAMATH_CALUDE_boxes_sold_l253_25376


namespace NUMINAMATH_CALUDE_crayons_per_day_l253_25370

theorem crayons_per_day (total_crayons : ℕ) (crayons_per_box : ℕ) 
  (h1 : total_crayons = 321)
  (h2 : crayons_per_box = 7) : 
  (total_crayons / crayons_per_box : ℕ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_day_l253_25370


namespace NUMINAMATH_CALUDE_fraction_equivalence_l253_25374

theorem fraction_equivalence : (8 : ℚ) / (7 * 67) = (0.8 : ℚ) / (0.7 * 67) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l253_25374


namespace NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l253_25350

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem least_five_digit_square_and_cube : 
  (is_five_digit 15625 ∧ is_perfect_square 15625 ∧ is_perfect_cube 15625) ∧ 
  (∀ n : ℕ, n < 15625 → ¬(is_five_digit n ∧ is_perfect_square n ∧ is_perfect_cube n)) :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_square_and_cube_l253_25350


namespace NUMINAMATH_CALUDE_probability_triangle_from_random_chords_probability_is_favorable_over_total_total_pairings_calculation_favorable_pairings_is_one_probability_triangle_from_random_chords_value_l253_25307

/-- The probability of forming a triangle when choosing three random chords on a circle -/
theorem probability_triangle_from_random_chords : ℚ :=
  1 / 15

/-- The number of ways to pair 6 points into three pairs -/
def total_pairings : ℕ := 15

/-- The number of pairings that result in all chords intersecting and forming a triangle -/
def favorable_pairings : ℕ := 1

theorem probability_is_favorable_over_total :
  probability_triangle_from_random_chords = favorable_pairings / total_pairings :=
sorry

theorem total_pairings_calculation :
  total_pairings = (1 / 6 : ℚ) * (Nat.choose 6 2) * (Nat.choose 4 2) * (Nat.choose 2 2) :=
sorry

theorem favorable_pairings_is_one :
  favorable_pairings = 1 :=
sorry

theorem probability_triangle_from_random_chords_value :
  probability_triangle_from_random_chords = 1 / 15 :=
sorry

end NUMINAMATH_CALUDE_probability_triangle_from_random_chords_probability_is_favorable_over_total_total_pairings_calculation_favorable_pairings_is_one_probability_triangle_from_random_chords_value_l253_25307


namespace NUMINAMATH_CALUDE_clown_balloons_l253_25319

/-- The number of balloons a clown has after blowing up an initial set and then an additional set -/
def total_balloons (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that the clown has 60 balloons after blowing up 47 and then 13 more -/
theorem clown_balloons : total_balloons 47 13 = 60 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l253_25319


namespace NUMINAMATH_CALUDE_inequality_proof_l253_25365

theorem inequality_proof (a b x : ℝ) (h1 : 0 < a) (h2 : a < b) :
  (b - a) / (b + a) ≤ (b + a * Real.sin x) / (b - a * Real.sin x) ∧
  (b + a * Real.sin x) / (b - a * Real.sin x) ≤ (b + a) / (b - a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l253_25365


namespace NUMINAMATH_CALUDE_initial_flow_rate_is_two_l253_25352

/-- Represents the flow rate of cleaner through a pipe over time -/
structure FlowRate where
  initial : ℝ
  after15min : ℝ
  after25min : ℝ

/-- Calculates the total amount of cleaner used given a flow rate profile -/
def totalCleanerUsed (flow : FlowRate) : ℝ :=
  15 * flow.initial + 10 * flow.after15min + 5 * flow.after25min

/-- Theorem stating that the initial flow rate is 2 ounces per minute -/
theorem initial_flow_rate_is_two :
  ∃ (flow : FlowRate),
    flow.after15min = 3 ∧
    flow.after25min = 4 ∧
    totalCleanerUsed flow = 80 ∧
    flow.initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_flow_rate_is_two_l253_25352


namespace NUMINAMATH_CALUDE_bus_passenger_ratio_l253_25377

/-- Represents the number of passengers on a bus --/
structure BusPassengers where
  men : ℕ
  women : ℕ

/-- The initial state of passengers on the bus --/
def initial : BusPassengers := sorry

/-- The state of passengers after changes in city Y --/
def after_city_y : BusPassengers := sorry

/-- The total number of passengers at the start --/
def total_passengers : ℕ := 72

/-- Changes in passenger numbers at city Y --/
def men_leave : ℕ := 16
def women_enter : ℕ := 8

theorem bus_passenger_ratio :
  initial.men = 2 * initial.women ∧
  initial.men + initial.women = total_passengers ∧
  after_city_y.men = initial.men - men_leave ∧
  after_city_y.women = initial.women + women_enter ∧
  after_city_y.men = after_city_y.women :=
by sorry

end NUMINAMATH_CALUDE_bus_passenger_ratio_l253_25377


namespace NUMINAMATH_CALUDE_subtracted_amount_for_ratio_change_l253_25366

theorem subtracted_amount_for_ratio_change : ∃ (a : ℝ),
  (72 : ℝ) / 192 = 3 / 8 ∧
  (72 - a) / (192 - a) = 4 / 9 ∧
  a = 24 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_for_ratio_change_l253_25366


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l253_25303

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l253_25303
