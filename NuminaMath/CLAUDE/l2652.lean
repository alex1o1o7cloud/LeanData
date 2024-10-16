import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l2652_265229

theorem consecutive_pages_sum (x : ℕ) (h : x + (x + 1) = 185) : x = 92 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l2652_265229


namespace NUMINAMATH_CALUDE_square_side_ratio_l2652_265291

theorem square_side_ratio (area_ratio : ℚ) (h : area_ratio = 50 / 98) :
  ∃ (p q r : ℕ), 
    (Real.sqrt (area_ratio) = p * Real.sqrt q / r) ∧
    (p + q + r = 13) := by
  sorry

end NUMINAMATH_CALUDE_square_side_ratio_l2652_265291


namespace NUMINAMATH_CALUDE_fifteen_people_handshakes_l2652_265230

/-- The number of handshakes in a group where each person shakes hands once with every other person -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 15 people, where each person shakes hands exactly once with every other person, the total number of handshakes is 105 -/
theorem fifteen_people_handshakes :
  handshakes 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_people_handshakes_l2652_265230


namespace NUMINAMATH_CALUDE_remaining_coins_value_is_1030_l2652_265232

-- Define the initial number of coins
def initial_quarters : ℕ := 33
def initial_nickels : ℕ := 87
def initial_dimes : ℕ := 52

-- Define the number of borrowed coins
def borrowed_quarters : ℕ := 15
def borrowed_nickels : ℕ := 75

-- Define the value of each coin type in cents
def quarter_value : ℕ := 25
def nickel_value : ℕ := 5
def dime_value : ℕ := 10

-- Define the function to calculate the total value of remaining coins
def remaining_coins_value : ℕ :=
  (initial_quarters * quarter_value + 
   initial_nickels * nickel_value + 
   initial_dimes * dime_value) - 
  (borrowed_quarters * quarter_value + 
   borrowed_nickels * nickel_value)

-- Theorem statement
theorem remaining_coins_value_is_1030 : 
  remaining_coins_value = 1030 := by sorry

end NUMINAMATH_CALUDE_remaining_coins_value_is_1030_l2652_265232


namespace NUMINAMATH_CALUDE_apples_per_person_l2652_265217

/-- Given that Harold gave apples to 3.0 people and the total number of apples given was 45,
    prove that each person received 15 apples. -/
theorem apples_per_person (total_apples : ℕ) (num_people : ℝ) 
    (h1 : total_apples = 45) 
    (h2 : num_people = 3.0) : 
  (total_apples : ℝ) / num_people = 15 := by
  sorry

end NUMINAMATH_CALUDE_apples_per_person_l2652_265217


namespace NUMINAMATH_CALUDE_apple_orange_cost_l2652_265278

theorem apple_orange_cost (apple_cost orange_cost : ℚ) 
  (eq1 : 2 * apple_cost + 3 * orange_cost = 6)
  (eq2 : 4 * apple_cost + 7 * orange_cost = 13) :
  16 * apple_cost + 23 * orange_cost = 47 := by
  sorry

end NUMINAMATH_CALUDE_apple_orange_cost_l2652_265278


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_negative_seven_l2652_265226

theorem sqrt_expression_equals_negative_seven :
  (Real.sqrt 15)^2 / Real.sqrt 3 * (1 / Real.sqrt 3) - Real.sqrt 6 * Real.sqrt 24 = -7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_negative_seven_l2652_265226


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l2652_265215

/-- Calculates the new weight of cucumbers after water evaporation -/
theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ) 
  (final_water_percent : ℝ) : 
  initial_weight = 100 → 
  initial_water_percent = 99 / 100 → 
  final_water_percent = 96 / 100 → 
  ∃ (new_weight : ℝ), 
    new_weight = 25 ∧ 
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * new_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l2652_265215


namespace NUMINAMATH_CALUDE_existence_of_m_l2652_265249

def M : Set ℕ := {n : ℕ | n ≤ 2007}

def arithmetic_progression (n : ℕ) : Set ℕ :=
  {k : ℕ | ∃ i : ℕ, k = n + i * (n + 1)}

theorem existence_of_m :
  (∀ n ∈ M, arithmetic_progression n ⊆ M) →
  ∃ m : ℕ, ∀ k > m, k ∈ M :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l2652_265249


namespace NUMINAMATH_CALUDE_area_DEF_eq_sum_of_parts_l2652_265286

/-- Represents a triangle with an area -/
structure Triangle :=
  (area : ℝ)

/-- Represents the main triangle DEF -/
def DEF : Triangle := sorry

/-- Represents the point Q inside triangle DEF -/
def Q : Point := sorry

/-- Represents the three smaller triangles created by lines through Q -/
def u₁ : Triangle := { area := 16 }
def u₂ : Triangle := { area := 25 }
def u₃ : Triangle := { area := 36 }

/-- The theorem stating that the area of DEF is the sum of areas of u₁, u₂, and u₃ -/
theorem area_DEF_eq_sum_of_parts : DEF.area = u₁.area + u₂.area + u₃.area := by
  sorry

#check area_DEF_eq_sum_of_parts

end NUMINAMATH_CALUDE_area_DEF_eq_sum_of_parts_l2652_265286


namespace NUMINAMATH_CALUDE_min_sum_visible_faces_l2652_265272

/-- Represents a die in the 4x4x4 cube --/
structure Die where
  visible_faces : List Nat
  deriving Repr

/-- Represents the 4x4x4 cube made of dice --/
structure Cube where
  dice : List Die
  deriving Repr

/-- Checks if a die's opposite sides sum to 7 --/
def valid_die (d : Die) : Prop :=
  d.visible_faces.length ≤ 4 ∧ 
  ∀ i j, i + j = 5 → i < d.visible_faces.length → j < d.visible_faces.length → 
    d.visible_faces[i]! + d.visible_faces[j]! = 7

/-- Checks if the cube is valid (4x4x4 and made of 64 dice) --/
def valid_cube (c : Cube) : Prop :=
  c.dice.length = 64 ∧ ∀ d ∈ c.dice, valid_die d

/-- Calculates the sum of visible faces on the cube --/
def sum_visible_faces (c : Cube) : Nat :=
  c.dice.foldl (λ acc d => acc + d.visible_faces.foldl (λ sum face => sum + face) 0) 0

/-- Theorem: The smallest possible sum of visible faces on a valid 4x4x4 cube is 304 --/
theorem min_sum_visible_faces (c : Cube) (h : valid_cube c) : 
  ∃ (min_cube : Cube), valid_cube min_cube ∧ 
    sum_visible_faces min_cube = 304 ∧
    ∀ (other_cube : Cube), valid_cube other_cube → 
      sum_visible_faces other_cube ≥ sum_visible_faces min_cube := by
  sorry

end NUMINAMATH_CALUDE_min_sum_visible_faces_l2652_265272


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l2652_265265

/-- Represents the colors of marbles --/
inductive Color
| Green
| Red

/-- Represents a circular arrangement of marbles --/
def CircularArrangement := List Color

/-- Counts the number of marbles with same-color neighbors --/
def countSameColorNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of marbles with different-color neighbors --/
def countDifferentColorNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if an arrangement satisfies the neighbor color condition --/
def isValidArrangement (arrangement : CircularArrangement) : Prop :=
  countSameColorNeighbors arrangement = countDifferentColorNeighbors arrangement

/-- Counts the number of valid arrangements --/
def countValidArrangements (greenMarbles redMarbles : Nat) : Nat :=
  sorry

/-- The main theorem --/
theorem marble_arrangement_theorem :
  let greenMarbles : Nat := 7
  let maxRedMarbles : Nat := 14
  (countValidArrangements greenMarbles maxRedMarbles) % 1000 = 432 :=
sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l2652_265265


namespace NUMINAMATH_CALUDE_train_passing_time_l2652_265255

/-- Given a train passing a platform, calculate the time it takes to pass a stationary point. -/
theorem train_passing_time (train_length platform_length platform_crossing_time : ℝ) 
  (h1 : train_length = 180)
  (h2 : platform_length = 270)
  (h3 : platform_crossing_time = 20)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 8 := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l2652_265255


namespace NUMINAMATH_CALUDE_solution_difference_l2652_265205

theorem solution_difference (p q : ℝ) : 
  p ≠ q → 
  ((6 * p - 18) / (p^2 + 4*p - 21) = p + 3) →
  ((6 * q - 18) / (q^2 + 4*q - 21) = q + 3) →
  p > q →
  p - q = 2 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2652_265205


namespace NUMINAMATH_CALUDE_no_identical_lines_l2652_265228

-- Define the equations of the lines
def line1 (a d x y : ℝ) : Prop := 4 * x + a * y + d = 0
def line2 (d x y : ℝ) : Prop := d * x - 3 * y + 15 = 0

-- Define what it means for the lines to be identical
def identical_lines (a d : ℝ) : Prop :=
  ∀ x y : ℝ, line1 a d x y ↔ line2 d x y

-- Theorem statement
theorem no_identical_lines : ¬∃ a d : ℝ, identical_lines a d :=
sorry

end NUMINAMATH_CALUDE_no_identical_lines_l2652_265228


namespace NUMINAMATH_CALUDE_fathers_seedlings_count_l2652_265261

/-- The number of seedlings Remi planted on the first day -/
def first_day_seedlings : ℕ := 200

/-- The number of seedlings Remi planted on the second day -/
def second_day_seedlings : ℕ := 2 * first_day_seedlings

/-- The total number of seedlings transferred to the farm on both days -/
def total_seedlings : ℕ := 1200

/-- The number of seedlings Remi's father planted -/
def fathers_seedlings : ℕ := total_seedlings - (first_day_seedlings + second_day_seedlings)

theorem fathers_seedlings_count : fathers_seedlings = 600 := by
  sorry

end NUMINAMATH_CALUDE_fathers_seedlings_count_l2652_265261


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l2652_265225

-- Define the quadratic function f
def f (x : ℝ) : ℝ := x^2 + x - 2

-- Define the theorem
theorem quadratic_function_theorem :
  (∀ x : ℝ, f x < 0 ↔ -2 < x ∧ x < 1) ∧ 
  f 0 = -2 ∧
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∀ x : ℝ, f x = x^2 + x - 2) ∧
  (∀ m : ℝ, (∀ θ : ℝ, f (Real.cos θ) ≤ Real.sqrt 2 * Real.sin (θ + Real.pi / 4) + m * Real.sin θ) ↔ 
    -3 ≤ m ∧ m ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l2652_265225


namespace NUMINAMATH_CALUDE_dance_group_size_l2652_265204

theorem dance_group_size (calligraphy_group : ℕ) (dance_group : ℕ) : 
  calligraphy_group = 28 → 
  calligraphy_group = 2 * dance_group + 6 → 
  dance_group = 11 := by
sorry

end NUMINAMATH_CALUDE_dance_group_size_l2652_265204


namespace NUMINAMATH_CALUDE_toby_friends_count_l2652_265269

theorem toby_friends_count (total : ℕ) (boys girls : ℕ) : 
  (boys : ℚ) / total = 55 / 100 →
  (girls : ℚ) / total = 30 / 100 →
  boys = 33 →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_toby_friends_count_l2652_265269


namespace NUMINAMATH_CALUDE_isabelle_ticket_cost_l2652_265220

def brothers_ticket_cost : ℕ := 20
def brothers_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def isabelle_earnings : ℕ := 30

def total_amount_needed : ℕ := isabelle_earnings + isabelle_savings + brothers_savings

theorem isabelle_ticket_cost :
  total_amount_needed - brothers_ticket_cost = 15 := by sorry

end NUMINAMATH_CALUDE_isabelle_ticket_cost_l2652_265220


namespace NUMINAMATH_CALUDE_central_high_teachers_central_high_teachers_count_l2652_265211

/-- Calculates the number of teachers required at Central High School -/
theorem central_high_teachers (total_students : ℕ) (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  let total_class_occurrences := total_students * classes_per_student
  let unique_classes := total_class_occurrences / students_per_class
  let required_teachers := unique_classes / classes_per_teacher
  required_teachers

/-- Proves that the number of teachers required at Central High School is 120 -/
theorem central_high_teachers_count : 
  central_high_teachers 1500 6 3 25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_central_high_teachers_central_high_teachers_count_l2652_265211


namespace NUMINAMATH_CALUDE_combined_temp_range_l2652_265224

-- Define the temperature ranges for each vegetable type
def type_a_range : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def type_b_range : Set ℝ := { x | 3 ≤ x ∧ x ≤ 8 }

-- Define the combined suitable temperature range
def combined_range : Set ℝ := type_a_range ∩ type_b_range

-- Theorem stating the combined suitable temperature range
theorem combined_temp_range : 
  combined_range = { x | 3 ≤ x ∧ x ≤ 5 } := by sorry

end NUMINAMATH_CALUDE_combined_temp_range_l2652_265224


namespace NUMINAMATH_CALUDE_simplified_expression_ratio_l2652_265203

theorem simplified_expression_ratio (m : ℤ) : 
  let simplified := (6 * m + 12) / 3
  ∃ (c d : ℤ), simplified = c * m + d ∧ c / d = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_ratio_l2652_265203


namespace NUMINAMATH_CALUDE_minimum_distance_to_exponential_curve_l2652_265275

open Real

theorem minimum_distance_to_exponential_curve (a : ℝ) :
  (∃ x₀ : ℝ, (x₀ - a)^2 + (exp x₀ - a)^2 ≤ 1/2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_distance_to_exponential_curve_l2652_265275


namespace NUMINAMATH_CALUDE_aunt_may_milk_sales_l2652_265288

/-- Represents the milk production and sales for Aunt May's farm --/
structure MilkProduction where
  morning : ℕ  -- Morning milk production in gallons
  evening : ℕ  -- Evening milk production in gallons
  leftover : ℕ  -- Leftover milk from yesterday in gallons
  remaining : ℕ  -- Remaining milk after selling in gallons

/-- Calculates the amount of milk sold to the ice cream factory --/
def milk_sold (p : MilkProduction) : ℕ :=
  p.morning + p.evening + p.leftover - p.remaining

/-- Theorem stating the amount of milk sold to the ice cream factory --/
theorem aunt_may_milk_sales (p : MilkProduction)
  (h_morning : p.morning = 365)
  (h_evening : p.evening = 380)
  (h_leftover : p.leftover = 15)
  (h_remaining : p.remaining = 148) :
  milk_sold p = 612 := by
  sorry

#eval milk_sold { morning := 365, evening := 380, leftover := 15, remaining := 148 }

end NUMINAMATH_CALUDE_aunt_may_milk_sales_l2652_265288


namespace NUMINAMATH_CALUDE_two_digit_primes_from_set_l2652_265240

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def digit_set : Finset ℕ := {3, 5, 8, 9}

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n ≤ 99

def formed_from_set (n : ℕ) : Prop :=
  is_two_digit n ∧
  (n / 10) ∈ digit_set ∧
  (n % 10) ∈ digit_set ∧
  (n / 10) ≠ (n % 10)

theorem two_digit_primes_from_set :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_prime n ∧ formed_from_set n) ∧
    (∀ n, is_prime n ∧ formed_from_set n → n ∈ s) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_digit_primes_from_set_l2652_265240


namespace NUMINAMATH_CALUDE_buying_100_tickets_may_not_win_l2652_265208

/-- Represents a lottery with a given number of tickets and winning probability per ticket -/
structure Lottery where
  totalTickets : ℕ
  winningProbability : ℝ
  winningProbability_nonneg : 0 ≤ winningProbability
  winningProbability_le_one : winningProbability ≤ 1

/-- The probability of not winning when buying a certain number of tickets -/
def probNotWinning (lottery : Lottery) (ticketsBought : ℕ) : ℝ :=
  (1 - lottery.winningProbability) ^ ticketsBought

/-- Theorem stating that buying 100 tickets in the given lottery may not result in a win -/
theorem buying_100_tickets_may_not_win (lottery : Lottery)
  (h1 : lottery.totalTickets = 100000)
  (h2 : lottery.winningProbability = 0.01) :
  probNotWinning lottery 100 > 0 := by
  sorry

#check buying_100_tickets_may_not_win

end NUMINAMATH_CALUDE_buying_100_tickets_may_not_win_l2652_265208


namespace NUMINAMATH_CALUDE_distinct_arrangements_of_six_objects_l2652_265219

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem distinct_arrangements_of_six_objects : 
  factorial 6 = 720 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_of_six_objects_l2652_265219


namespace NUMINAMATH_CALUDE_diagonal_four_sides_squared_l2652_265245

/-- A regular nonagon -/
structure RegularNonagon where
  /-- The length of a side -/
  a : ℝ
  /-- The length of a diagonal that jumps over four sides -/
  d : ℝ
  /-- Ensure the side length is positive -/
  a_pos : a > 0

/-- In a regular nonagon, the square of the length of a diagonal that jumps over four sides
    is equal to five times the square of the side length -/
theorem diagonal_four_sides_squared (n : RegularNonagon) : n.d^2 = 5 * n.a^2 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_four_sides_squared_l2652_265245


namespace NUMINAMATH_CALUDE_special_triangle_not_necessarily_right_l2652_265237

/-- A triangle with sides a, b, and c where a² = 5, b² = 12, and c² = 13 -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a^2 = 5
  hb : b^2 = 12
  hc : c^2 = 13

/-- A right triangle is a triangle where one of its angles is 90 degrees -/
def IsRightTriangle (t : SpecialTriangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2

/-- Theorem stating that it cannot be determined if a SpecialTriangle is a right triangle -/
theorem special_triangle_not_necessarily_right (t : SpecialTriangle) :
  ¬ (IsRightTriangle t) := by sorry

end NUMINAMATH_CALUDE_special_triangle_not_necessarily_right_l2652_265237


namespace NUMINAMATH_CALUDE_isosceles_if_root_is_one_roots_of_equilateral_l2652_265246

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

-- Define the quadratic equation
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 - 2 * t.b * x + (t.a - t.c)

theorem isosceles_if_root_is_one (t : Triangle) :
  quadratic t 1 = 0 → t.a = t.b :=
by sorry

theorem roots_of_equilateral (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∀ x : ℝ, quadratic t x = 0 ↔ x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_if_root_is_one_roots_of_equilateral_l2652_265246


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2652_265262

theorem solution_set_of_inequality (x : ℝ) :
  (x * (x - 1) < 2) ↔ (-1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2652_265262


namespace NUMINAMATH_CALUDE_elephant_weight_l2652_265267

theorem elephant_weight (elephant_weight : ℝ) (donkey_weight : ℝ) : 
  elephant_weight * 2000 + donkey_weight = 6600 →
  donkey_weight = 0.1 * (elephant_weight * 2000) →
  elephant_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_elephant_weight_l2652_265267


namespace NUMINAMATH_CALUDE_exponent_rule_l2652_265244

theorem exponent_rule (a : ℝ) (m : ℤ) : a^(2*m + 2) = a^(2*m) * a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_rule_l2652_265244


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_f_geq_abs_a_plus_1_l2652_265274

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + 2 * |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_4 :
  {x : ℝ | f x < 4} = {x : ℝ | -1 < x ∧ x < 5/3} := by sorry

-- Theorem for part II
theorem range_of_a_for_f_geq_abs_a_plus_1 :
  (∀ x : ℝ, f x ≥ |a + 1|) ↔ -3 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_f_geq_abs_a_plus_1_l2652_265274


namespace NUMINAMATH_CALUDE_prob_no_green_3x3_value_main_result_l2652_265264

/-- Represents a 4x4 grid of colored squares -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a 3x3 subgrid starting at (i, j) is all green -/
def has_green_3x3 (g : Grid) (i j : Fin 2) : Prop :=
  ∀ x y, g (i + x) (j + y)

/-- The probability of not having a 3x3 green square in a 4x4 grid -/
def prob_no_green_3x3 : ℚ :=
  1 - (419 : ℚ) / 2^16

theorem prob_no_green_3x3_value :
  prob_no_green_3x3 = 65117 / 65536 :=
sorry

/-- The sum of the numerator and denominator of the probability -/
def sum_num_denom : ℕ := 65117 + 65536

theorem main_result :
  sum_num_denom = 130653 :=
sorry

end NUMINAMATH_CALUDE_prob_no_green_3x3_value_main_result_l2652_265264


namespace NUMINAMATH_CALUDE_fish_value_is_three_and_three_quarters_l2652_265298

/-- Represents the value of one fish in terms of bags of rice -/
def fish_value (fish_to_bread : ℚ) (bread_to_rice : ℚ) : ℚ :=
  (fish_to_bread * bread_to_rice)⁻¹

/-- Theorem stating that one fish is worth 3¾ bags of rice given the trade rates -/
theorem fish_value_is_three_and_three_quarters :
  fish_value (4/5) 3 = 15/4 := by
  sorry

end NUMINAMATH_CALUDE_fish_value_is_three_and_three_quarters_l2652_265298


namespace NUMINAMATH_CALUDE_donation_ratio_l2652_265235

theorem donation_ratio (shirts pants shorts : ℕ) : 
  shirts = 4 →
  pants = 2 * shirts →
  shirts + pants + shorts = 16 →
  shorts * 2 = pants :=
by
  sorry

end NUMINAMATH_CALUDE_donation_ratio_l2652_265235


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2652_265296

theorem coefficient_x_squared_in_expansion (x : ℝ) :
  ∃ c, (x + 2)^4 = c * x^2 + (x + 2)^4 - c * x^2 ∧ c = 24 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2652_265296


namespace NUMINAMATH_CALUDE_discount_sales_income_increase_l2652_265294

/-- Proves that a 10% discount with 30% increase in sales leads to 17% increase in gross income -/
theorem discount_sales_income_increase 
  (original_price : ℝ) 
  (original_quantity : ℝ) 
  (discount_rate : ℝ) 
  (sales_increase_rate : ℝ) 
  (h1 : discount_rate = 0.1) 
  (h2 : sales_increase_rate = 0.3) : 
  (((1 + sales_increase_rate) * (1 - discount_rate) - 1) * 100 : ℝ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_discount_sales_income_increase_l2652_265294


namespace NUMINAMATH_CALUDE_plan_d_more_economical_l2652_265289

/-- The cost per gigabyte for Plan C in cents -/
def plan_c_cost_per_gb : ℚ := 15

/-- The initial fee for Plan D in cents -/
def plan_d_initial_fee : ℚ := 3000

/-- The cost per gigabyte for Plan D in cents -/
def plan_d_cost_per_gb : ℚ := 8

/-- The minimum number of gigabytes for Plan D to be more economical -/
def min_gb_for_plan_d : ℕ := 429

theorem plan_d_more_economical :
  (∀ n : ℕ, n ≥ min_gb_for_plan_d →
    plan_d_initial_fee + n * plan_d_cost_per_gb < n * plan_c_cost_per_gb) ∧
  (∀ n : ℕ, n < min_gb_for_plan_d →
    plan_d_initial_fee + n * plan_d_cost_per_gb ≥ n * plan_c_cost_per_gb) :=
by sorry

end NUMINAMATH_CALUDE_plan_d_more_economical_l2652_265289


namespace NUMINAMATH_CALUDE_first_player_wins_6x8_l2652_265263

/-- Represents a chocolate bar game -/
structure ChocolateGame where
  rows : ℕ
  cols : ℕ

/-- Calculates the total number of moves in a chocolate bar game -/
def totalMoves (game : ChocolateGame) : ℕ :=
  game.rows * game.cols - 1

/-- Determines if the first player wins the game -/
def firstPlayerWins (game : ChocolateGame) : Prop :=
  Odd (totalMoves game)

/-- Theorem: The first player wins in a 6x8 chocolate bar game -/
theorem first_player_wins_6x8 :
  firstPlayerWins ⟨6, 8⟩ := by sorry

end NUMINAMATH_CALUDE_first_player_wins_6x8_l2652_265263


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2652_265207

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  -- The unique solution is x = 8/3
  use 8/3
  constructor
  · -- Prove that 8/3 satisfies the equation
    sorry
  · -- Prove that any solution must equal 8/3
    sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l2652_265207


namespace NUMINAMATH_CALUDE_our_circle_center_and_radius_l2652_265284

/-- A circle in the 2D plane --/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle --/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle --/
def Circle.radius (c : Circle) : ℝ := sorry

/-- Our specific circle --/
def our_circle : Circle :=
  { equation := fun x y => x^2 + y^2 - 4*x - 6*y - 3 = 0 }

theorem our_circle_center_and_radius :
  Circle.center our_circle = (2, 3) ∧ Circle.radius our_circle = 4 := by sorry

end NUMINAMATH_CALUDE_our_circle_center_and_radius_l2652_265284


namespace NUMINAMATH_CALUDE_earthquake_relief_team_selection_l2652_265254

/-- The number of ways to select a team of 5 doctors from 3 specialties -/
def select_team (orthopedic neurosurgeon internist : ℕ) : ℕ :=
  let total := orthopedic + neurosurgeon + internist
  let team_size := 5
  -- Add the number of ways for each valid combination
  (orthopedic.choose 3 * neurosurgeon.choose 1 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 3 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 1 * internist.choose 3) +
  (orthopedic.choose 2 * neurosurgeon.choose 2 * internist.choose 1) +
  (orthopedic.choose 1 * neurosurgeon.choose 2 * internist.choose 2) +
  (orthopedic.choose 2 * neurosurgeon.choose 1 * internist.choose 2)

/-- Theorem: The number of ways to select 5 people from 3 orthopedic doctors, 
    4 neurosurgeons, and 5 internists, with at least one from each specialty, is 590 -/
theorem earthquake_relief_team_selection : select_team 3 4 5 = 590 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_relief_team_selection_l2652_265254


namespace NUMINAMATH_CALUDE_min_sum_of_product_1176_l2652_265253

theorem min_sum_of_product_1176 (a b c : ℕ+) (h : a * b * c = 1176) :
  (∀ x y z : ℕ+, x * y * z = 1176 → a + b + c ≤ x + y + z) →
  a + b + c = 59 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1176_l2652_265253


namespace NUMINAMATH_CALUDE_gym_schedule_theorem_l2652_265256

/-- Represents days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the next day of the week -/
def nextDay (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

/-- Returns true if it's a gym day based on the schedule -/
def isGymDay (d : Day) : Bool :=
  match d with
  | Day.Sunday => false
  | _ => true

/-- Calculates the day after a given number of gym sessions -/
def dayAfterSessions (startDay : Day) (sessions : Nat) : Day :=
  if sessions = 0 then
    startDay
  else
    let nextGymDay := 
      match startDay with
      | Day.Saturday => nextDay (nextDay startDay)
      | _ => nextDay (nextDay startDay)
    dayAfterSessions nextGymDay (sessions - 1)

theorem gym_schedule_theorem :
  dayAfterSessions Day.Monday 35 = Day.Wednesday := by
  sorry


end NUMINAMATH_CALUDE_gym_schedule_theorem_l2652_265256


namespace NUMINAMATH_CALUDE_arithmetic_progression_non_prime_existence_l2652_265223

theorem arithmetic_progression_non_prime_existence 
  (a d : ℕ+) : 
  ∃ K : ℕ+, ∀ n : ℕ, ∃ i : Fin K, ¬ Nat.Prime (a + (n + i) * d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_non_prime_existence_l2652_265223


namespace NUMINAMATH_CALUDE_sum_odd_integers_l2652_265221

theorem sum_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : n^2 = 4900 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_l2652_265221


namespace NUMINAMATH_CALUDE_particle_position_after_1991_minutes_l2652_265248

-- Define the particle's position type
def Position := ℤ × ℤ

-- Define the starting position
def start_position : Position := (0, 1)

-- Define the movement pattern for a single rectangle
def rectangle_movement (n : ℕ) : Position := 
  if n % 2 = 1 then (n, n + 1) else (-(n + 1), -n)

-- Define the time taken for a single rectangle
def rectangle_time (n : ℕ) : ℕ := 2 * n + 1

-- Define the total time for n rectangles
def total_time (n : ℕ) : ℕ := (n + 1)^2 - 1

-- Define the function to calculate the position after n rectangles
def position_after_rectangles (n : ℕ) : Position :=
  sorry

-- Define the function to calculate the final position
def final_position (time : ℕ) : Position :=
  sorry

-- The theorem to prove
theorem particle_position_after_1991_minutes :
  final_position 1991 = (-45, -32) :=
sorry

end NUMINAMATH_CALUDE_particle_position_after_1991_minutes_l2652_265248


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2652_265292

theorem sqrt_x_minus_one_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l2652_265292


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2652_265282

theorem solution_set_inequality (x : ℝ) : 
  (x - 3) / (x + 2) ≤ 0 ↔ -2 < x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2652_265282


namespace NUMINAMATH_CALUDE_age_difference_l2652_265233

theorem age_difference (patrick michael monica nathan : ℝ) : 
  patrick / michael = 3 / 5 →
  michael / monica = 3 / 4 →
  monica / nathan = 5 / 7 →
  nathan / patrick = 4 / 9 →
  patrick + michael + monica + nathan = 252 →
  nathan - patrick = 66.5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l2652_265233


namespace NUMINAMATH_CALUDE_distribution_problem_l2652_265276

theorem distribution_problem (total_amount : ℕ) (first_group : ℕ) (difference : ℕ) (second_group : ℕ) :
  total_amount = 5040 →
  first_group = 14 →
  difference = 80 →
  (total_amount / first_group) = (total_amount / second_group + difference) →
  second_group = 18 := by
sorry

end NUMINAMATH_CALUDE_distribution_problem_l2652_265276


namespace NUMINAMATH_CALUDE_dividend_calculation_l2652_265266

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 21) 
  (h2 : quotient = 14) 
  (h3 : remainder = 7) : 
  divisor * quotient + remainder = 301 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2652_265266


namespace NUMINAMATH_CALUDE_train_journey_time_l2652_265234

theorem train_journey_time (X : ℝ) (h1 : 0 < X) (h2 : X < 60) : 
  (X * 6 - X * 0.5 = 360 - X) → X = 360 / 7 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l2652_265234


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2652_265277

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) 
  (pos_s : 0 < s) (pos_t : 0 < t) (pos_u : 0 < u)
  (sum_eq_8 : p + q + r + s + t + u = 8) :
  2/p + 4/q + 9/r + 16/s + 25/t + 36/u ≥ 98 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2652_265277


namespace NUMINAMATH_CALUDE_ladder_length_is_twice_h_l2652_265259

/-- The length of a ladder resting against two walls in an alley -/
def ladder_length (w h k : ℝ) : ℝ :=
  2 * h

/-- Theorem: The length of the ladder is twice the height at point Q -/
theorem ladder_length_is_twice_h (w h k : ℝ) (hw : w > 0) (hh : h > 0) (hk : k > 0) :
  ladder_length w h k = 2 * h :=
by
  sorry

#check ladder_length_is_twice_h

end NUMINAMATH_CALUDE_ladder_length_is_twice_h_l2652_265259


namespace NUMINAMATH_CALUDE_complex_set_sum_l2652_265279

def is_closed_under_multiplication (S : Set ℂ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S

theorem complex_set_sum (a b c d : ℂ) :
  let S : Set ℂ := {a, b, c, d}
  is_closed_under_multiplication S →
  a^2 = 1 →
  b = 1 →
  c^2 = a →
  b + c + d = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_set_sum_l2652_265279


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2652_265285

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = (x + 1)*(x + 3)) → m - n = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2652_265285


namespace NUMINAMATH_CALUDE_distinct_values_of_c_l2652_265231

/-- Given a complex number c and distinct complex numbers r, s, and t satisfying
    (z - r)(z - s)(z - t) = (z - 2cr)(z - 2cs)(z - 2ct) for all complex z,
    there are exactly 3 distinct possible values of c. -/
theorem distinct_values_of_c (c r s t : ℂ) : 
  r ≠ s ∧ s ≠ t ∧ r ≠ t →
  (∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - 2*c*r) * (z - 2*c*s) * (z - 2*c*t)) →
  ∃! (values : Finset ℂ), values.card = 3 ∧ c ∈ values := by
  sorry

end NUMINAMATH_CALUDE_distinct_values_of_c_l2652_265231


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2652_265271

theorem fraction_sum_equality : 
  (3 : ℚ) / 12 + (6 : ℚ) / 120 + (9 : ℚ) / 1200 = (3075 : ℚ) / 10000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2652_265271


namespace NUMINAMATH_CALUDE_worker_idle_days_l2652_265239

/-- Proves that given the specified conditions, the number of idle days is 38 --/
theorem worker_idle_days 
  (total_days : ℕ) 
  (pay_per_working_day : ℕ) 
  (forfeit_per_idle_day : ℕ) 
  (total_amount : ℕ) 
  (h1 : total_days = 60)
  (h2 : pay_per_working_day = 30)
  (h3 : forfeit_per_idle_day = 5)
  (h4 : total_amount = 500) :
  ∃ (idle_days : ℕ), 
    idle_days = 38 ∧ 
    idle_days + (total_days - idle_days) = total_days ∧
    pay_per_working_day * (total_days - idle_days) - forfeit_per_idle_day * idle_days = total_amount :=
by
  sorry

end NUMINAMATH_CALUDE_worker_idle_days_l2652_265239


namespace NUMINAMATH_CALUDE_max_value_of_rational_function_l2652_265281

theorem max_value_of_rational_function : 
  (∀ x : ℝ, (5*x^2 + 10*x + 12) / (5*x^2 + 10*x + 2) ≤ 5) ∧ 
  (∀ ε > 0, ∃ x : ℝ, (5*x^2 + 10*x + 12) / (5*x^2 + 10*x + 2) > 5 - ε) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_rational_function_l2652_265281


namespace NUMINAMATH_CALUDE_point_on_x_axis_with_distance_l2652_265287

/-- A point P on the x-axis that is √30 distance from P₁(4,1,2) has x-coordinate 9 or -1 -/
theorem point_on_x_axis_with_distance (x : ℝ) :
  (x - 4)^2 + 1^2 + 2^2 = 30 → x = 9 ∨ x = -1 := by
  sorry

#check point_on_x_axis_with_distance

end NUMINAMATH_CALUDE_point_on_x_axis_with_distance_l2652_265287


namespace NUMINAMATH_CALUDE_relationship_between_a_b_l2652_265210

theorem relationship_between_a_b (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) :
  a < -b ∧ -b < b ∧ b < -a := by sorry

end NUMINAMATH_CALUDE_relationship_between_a_b_l2652_265210


namespace NUMINAMATH_CALUDE_division_problem_l2652_265251

theorem division_problem : (120 / 6) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2652_265251


namespace NUMINAMATH_CALUDE_spherical_coordinate_symmetry_l2652_265202

/-- Given a point with rectangular coordinates (3, -4, 2) and corresponding
    spherical coordinates (ρ, θ, φ), prove that the point with spherical
    coordinates (ρ, -θ, φ) has rectangular coordinates (3, -4, 2). -/
theorem spherical_coordinate_symmetry (ρ θ φ : ℝ) :
  ρ * Real.sin φ * Real.cos θ = 3 ∧
  ρ * Real.sin φ * Real.sin θ = -4 ∧
  ρ * Real.cos φ = 2 →
  ρ * Real.sin φ * Real.cos (-θ) = 3 ∧
  ρ * Real.sin φ * Real.sin (-θ) = -4 ∧
  ρ * Real.cos φ = 2 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_symmetry_l2652_265202


namespace NUMINAMATH_CALUDE_expected_value_is_one_l2652_265273

/-- Represents the possible outcomes of rolling the die -/
inductive DieOutcome
| one
| two
| three
| four
| five
| six

/-- The probability of rolling each outcome -/
def prob (outcome : DieOutcome) : ℚ :=
  match outcome with
  | .one => 1/4
  | .two => 1/4
  | .three => 1/6
  | .four => 1/6
  | .five => 1/12
  | .six => 1/12

/-- The earnings associated with each outcome -/
def earnings (outcome : DieOutcome) : ℤ :=
  match outcome with
  | .one | .two => 4
  | .three | .four => -3
  | .five | .six => 0

/-- The expected value of earnings from one roll of the die -/
def expectedValue : ℚ :=
  (prob .one * earnings .one) +
  (prob .two * earnings .two) +
  (prob .three * earnings .three) +
  (prob .four * earnings .four) +
  (prob .five * earnings .five) +
  (prob .six * earnings .six)

/-- Theorem stating that the expected value of earnings is 1 -/
theorem expected_value_is_one : expectedValue = 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_is_one_l2652_265273


namespace NUMINAMATH_CALUDE_china_gdp_scientific_notation_l2652_265250

def trillion : ℝ := 10^12

theorem china_gdp_scientific_notation :
  11.69 * trillion = 1.169 * 10^14 := by sorry

end NUMINAMATH_CALUDE_china_gdp_scientific_notation_l2652_265250


namespace NUMINAMATH_CALUDE_fred_red_marbles_l2652_265218

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The conditions of Fred's marble collection --/
def fredMarbles (m : MarbleCount) : Prop :=
  m.red + m.green + m.blue = 63 ∧
  m.blue = 6 ∧
  m.green = m.red / 2

/-- Theorem stating that Fred has 38 red marbles --/
theorem fred_red_marbles :
  ∃ m : MarbleCount, fredMarbles m ∧ m.red = 38 := by
  sorry

end NUMINAMATH_CALUDE_fred_red_marbles_l2652_265218


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l2652_265241

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 25 / 1000

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax deduction in cents -/
def tax_deduction (wage : ℚ) (rate : ℚ) : ℚ :=
  dollars_to_cents (wage * rate)

theorem alicia_tax_deduction :
  tax_deduction hourly_wage tax_rate = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l2652_265241


namespace NUMINAMATH_CALUDE_f_g_intersection_l2652_265201

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x

/-- Theorem stating that f and g have exactly one intersection point when a ≥ 0 -/
theorem f_g_intersection (a : ℝ) (h : a ≥ 0) :
  ∃! x : ℝ, x > 0 ∧ f a x = g a x :=
sorry

end NUMINAMATH_CALUDE_f_g_intersection_l2652_265201


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2652_265290

theorem inequality_solution_set (a b : ℝ) (h : b ≠ 0) :
  ¬(∀ x : ℝ, ax > b ↔ x < -b/a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2652_265290


namespace NUMINAMATH_CALUDE_taco_truck_beef_amount_l2652_265280

/-- The taco truck problem -/
theorem taco_truck_beef_amount :
  ∀ (beef_amount : ℝ),
    (beef_amount > 0) →
    (0.25 * (beef_amount / 0.25) * (2 - 1.5) = 200) →
    beef_amount = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_taco_truck_beef_amount_l2652_265280


namespace NUMINAMATH_CALUDE_rate_increase_is_33_percent_l2652_265243

/-- Represents the work team's processing scenario -/
structure WorkScenario where
  initial_items : ℕ
  total_time : ℕ
  worked_time : ℕ
  additional_items : ℕ

/-- Calculates the required rate increase percentage -/
def required_rate_increase (scenario : WorkScenario) : ℚ :=
  let initial_rate := scenario.initial_items / scenario.total_time
  let processed_items := initial_rate * scenario.worked_time
  let remaining_items := scenario.initial_items - processed_items + scenario.additional_items
  let remaining_time := scenario.total_time - scenario.worked_time
  let new_rate := remaining_items / remaining_time
  (new_rate - initial_rate) / initial_rate * 100

/-- The main theorem stating that the required rate increase is 33% -/
theorem rate_increase_is_33_percent (scenario : WorkScenario) 
  (h1 : scenario.initial_items = 1250)
  (h2 : scenario.total_time = 10)
  (h3 : scenario.worked_time = 6)
  (h4 : scenario.additional_items = 165) :
  required_rate_increase scenario = 33 := by
  sorry

end NUMINAMATH_CALUDE_rate_increase_is_33_percent_l2652_265243


namespace NUMINAMATH_CALUDE_henry_final_distance_l2652_265206

-- Define the conversion factor from meters to feet
def metersToFeet : ℝ := 3.28084

-- Define Henry's movements
def northDistance : ℝ := 10 -- in meters
def eastDistance : ℝ := 30 -- in feet
def southDistance : ℝ := 10 * metersToFeet + 40 -- in feet

-- Calculate net southward movement
def netSouthDistance : ℝ := southDistance - (northDistance * metersToFeet)

-- Theorem to prove
theorem henry_final_distance :
  Real.sqrt (eastDistance ^ 2 + netSouthDistance ^ 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_henry_final_distance_l2652_265206


namespace NUMINAMATH_CALUDE_intersection_problem_l2652_265227

/-- The problem statement as a theorem -/
theorem intersection_problem (m b k : ℝ) : 
  b ≠ 0 →
  7 = 2 * m + b →
  (∃ y₁ y₂ : ℝ, 
    y₁ = k^2 + 8*k + 7 ∧
    y₂ = m*k + b ∧
    |y₁ - y₂| = 4) →
  m = 6 ∧ b = -5 := by
sorry

end NUMINAMATH_CALUDE_intersection_problem_l2652_265227


namespace NUMINAMATH_CALUDE_distinct_paths_6x4_l2652_265270

/-- The number of rows in the grid -/
def rows : ℕ := 4

/-- The number of columns in the grid -/
def cols : ℕ := 6

/-- The total number of steps needed to reach from top-left to bottom-right -/
def total_steps : ℕ := rows + cols - 2

/-- The number of down steps needed -/
def down_steps : ℕ := rows - 1

/-- The number of distinct paths from top-left to bottom-right in a 6x4 grid -/
theorem distinct_paths_6x4 : Nat.choose total_steps down_steps = 56 := by
  sorry

end NUMINAMATH_CALUDE_distinct_paths_6x4_l2652_265270


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_slope_l2652_265247

/-- Theorem: For an ellipse and a line intersecting it under specific conditions, the slope of the line is ±1/2. -/
theorem ellipse_line_intersection_slope (k : ℝ) : 
  (∀ x y, x^2/4 + y^2/3 = 1 → y = k*x + 1 → 
    ∃ x₁ x₂ y₁ y₂, 
      x₁^2/4 + y₁^2/3 = 1 ∧ 
      x₂^2/4 + y₂^2/3 = 1 ∧
      y₁ = k*x₁ + 1 ∧ 
      y₂ = k*x₂ + 1 ∧ 
      x₁ = -x₂/2) →
  k = 1/2 ∨ k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_slope_l2652_265247


namespace NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l2652_265283

/-- Calculates the interest paid as a percentage of the amount borrowed for a set of encyclopedias. -/
theorem encyclopedia_interest_percentage (cost : ℝ) (down_payment : ℝ) (monthly_payment : ℝ) (num_months : ℕ) (final_payment : ℝ) :
  cost = 1200 →
  down_payment = 500 →
  monthly_payment = 70 →
  num_months = 12 →
  final_payment = 45 →
  let total_paid := down_payment + (monthly_payment * num_months) + final_payment
  let amount_borrowed := cost - down_payment
  let interest_paid := total_paid - cost
  let interest_percentage := (interest_paid / amount_borrowed) * 100
  ∃ ε > 0, |interest_percentage - 26.43| < ε :=
by sorry

end NUMINAMATH_CALUDE_encyclopedia_interest_percentage_l2652_265283


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l2652_265260

theorem zoo_animal_ratio (initial_animals : ℕ) (final_animals : ℕ)
  (gorillas_sent : ℕ) (hippo_adopted : ℕ) (rhinos_taken : ℕ) (lion_cubs_born : ℕ)
  (h1 : initial_animals = 68)
  (h2 : final_animals = 90)
  (h3 : gorillas_sent = 6)
  (h4 : hippo_adopted = 1)
  (h5 : rhinos_taken = 3)
  (h6 : lion_cubs_born = 8) :
  (final_animals - (initial_animals - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs_born)) / lion_cubs_born = 2 :=
by sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l2652_265260


namespace NUMINAMATH_CALUDE_K_on_circle_S₂_l2652_265213

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def S : Circle := { center := (0, 0), radius := 2 }
def S₁ : Circle := { center := (1, 0), radius := 1 }
def S₂ : Circle := { center := (3, 0), radius := 1 }

def B : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (2, 0)

-- Define the intersection point K
def K : ℝ × ℝ := sorry

-- Define the properties of the circles
def S₁_tangent_to_S : Prop :=
  (S₁.center.1 - S.center.1)^2 + (S₁.center.2 - S.center.2)^2 = (S.radius - S₁.radius)^2

def S₂_tangent_to_S₁ : Prop :=
  (S₂.center.1 - S₁.center.1)^2 + (S₂.center.2 - S₁.center.2)^2 = (S₁.radius + S₂.radius)^2

def S₂_not_tangent_to_S : Prop :=
  (S₂.center.1 - S.center.1)^2 + (S₂.center.2 - S.center.2)^2 ≠ (S.radius - S₂.radius)^2

def K_on_line_AB : Prop :=
  (K.2 - A.2) * (B.1 - A.1) = (K.1 - A.1) * (B.2 - A.2)

def K_on_circle_S : Prop :=
  (K.1 - S.center.1)^2 + (K.2 - S.center.2)^2 = S.radius^2

-- Theorem to prove
theorem K_on_circle_S₂ (h1 : S₁_tangent_to_S) (h2 : S₂_tangent_to_S₁) 
    (h3 : S₂_not_tangent_to_S) (h4 : K_on_line_AB) (h5 : K_on_circle_S) :
  (K.1 - S₂.center.1)^2 + (K.2 - S₂.center.2)^2 = S₂.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_K_on_circle_S₂_l2652_265213


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_B_l2652_265293

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

-- Theorem for complement of B
theorem complement_B : (Set.univ : Set ℝ) \ B = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_B_l2652_265293


namespace NUMINAMATH_CALUDE_scarf_sales_with_new_price_and_tax_l2652_265242

/-- Represents the relationship between number of scarves sold and their price -/
def scarfRelation (k : ℝ) (p c : ℝ) : Prop := p * c = k

theorem scarf_sales_with_new_price_and_tax 
  (k : ℝ) 
  (initial_price initial_quantity new_price tax_rate : ℝ) : 
  scarfRelation k initial_quantity initial_price →
  initial_price = 10 →
  initial_quantity = 30 →
  new_price = 15 →
  tax_rate = 0.1 →
  ∃ (new_quantity : ℕ), 
    scarfRelation k (new_quantity : ℝ) (new_price * (1 + tax_rate)) ∧ 
    new_quantity = 18 := by
  sorry

end NUMINAMATH_CALUDE_scarf_sales_with_new_price_and_tax_l2652_265242


namespace NUMINAMATH_CALUDE_arithmetic_sequence_iff_constant_difference_l2652_265238

/-- A sequence is arithmetic if and only if the difference between consecutive terms is constant -/
theorem arithmetic_sequence_iff_constant_difference (a : ℕ → ℝ) :
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ↔ 
  (∃ a₀ d : ℝ, ∀ n : ℕ, a n = a₀ + n • d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_iff_constant_difference_l2652_265238


namespace NUMINAMATH_CALUDE_shoe_pairs_count_l2652_265209

theorem shoe_pairs_count (total_shoes : ℕ) (prob_same_color : ℚ) : 
  total_shoes = 14 →
  prob_same_color = 1 / 13 →
  (∃ (n : ℕ), n * 2 = total_shoes ∧ 
    prob_same_color = 1 / (2 * n - 1)) →
  ∃ (pairs : ℕ), pairs = 7 ∧ pairs * 2 = total_shoes :=
by sorry

end NUMINAMATH_CALUDE_shoe_pairs_count_l2652_265209


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2652_265257

theorem arithmetic_sequence_squares (a b c : ℝ) 
  (h : ∃ (d : ℝ), (1 / (b + c)) - (1 / (a + b)) = (1 / (c + a)) - (1 / (b + c))) :
  ∃ (k : ℝ), b^2 - a^2 = c^2 - b^2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squares_l2652_265257


namespace NUMINAMATH_CALUDE_range_of_m_l2652_265252

open Set

theorem range_of_m (m : ℝ) : 
  let U : Set ℝ := Set.univ
  let A : Set ℝ := {x | x < 1}
  let B : Set ℝ := {x | x ≥ m}
  (Aᶜ ⊆ B) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2652_265252


namespace NUMINAMATH_CALUDE_polygon_sides_l2652_265214

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : 
  sum_interior_angles = 1080 → n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2652_265214


namespace NUMINAMATH_CALUDE_cos_negative_75_degrees_l2652_265236

theorem cos_negative_75_degrees :
  Real.cos (-(75 * π / 180)) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_75_degrees_l2652_265236


namespace NUMINAMATH_CALUDE_unit_segments_bound_l2652_265299

/-- 
Given n distinct points in a plane, τ(n) represents the number of 
unit-length segments joining pairs of these points.
-/
def τ (n : ℕ) : ℕ := sorry

/-- 
Theorem: The number of unit-length segments joining pairs of n distinct 
points in a plane is at most n²/3.
-/
theorem unit_segments_bound (n : ℕ) : τ n ≤ n^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_unit_segments_bound_l2652_265299


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2652_265212

-- Define the hyperbola and its properties
def Hyperbola (a : ℝ) : Prop :=
  a > 0 ∧ ∃ (x y : ℝ), x^2 / a^2 - y^2 / 5 = 1

-- Define the asymptote
def Asymptote (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y = (Real.sqrt 5 / 2) * x

-- Define eccentricity
def Eccentricity (e : ℝ) (a : ℝ) : Prop :=
  e = Real.sqrt (a^2 + 5) / a

-- Theorem statement
theorem hyperbola_eccentricity (a : ℝ) :
  Hyperbola a → Asymptote a → Eccentricity (3/2) a := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2652_265212


namespace NUMINAMATH_CALUDE_problem_solution_l2652_265222

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 7)
  (h_y_x : y + 1 / x = 20) :
  z + 1 / y = 29 / 139 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2652_265222


namespace NUMINAMATH_CALUDE_language_coverage_probability_l2652_265216

def total_students : ℕ := 40
def french_students : ℕ := 30
def spanish_students : ℕ := 32
def german_students : ℕ := 10
def german_and_other : ℕ := 26

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem language_coverage_probability :
  let french_and_spanish : ℕ := french_students + spanish_students - total_students
  let french_only : ℕ := french_students - french_and_spanish
  let spanish_only : ℕ := spanish_students - french_and_spanish
  let remaining : ℕ := total_students - (french_only + spanish_only + french_and_spanish)
  let total_combinations : ℕ := choose total_students 2
  let unfavorable_outcomes : ℕ := choose french_only 2 + choose spanish_only 2 + choose remaining 2
  (total_combinations - unfavorable_outcomes : ℚ) / total_combinations = 353 / 390 :=
sorry

end NUMINAMATH_CALUDE_language_coverage_probability_l2652_265216


namespace NUMINAMATH_CALUDE_points_per_game_l2652_265297

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 91 → 
  num_games = 13 → 
  total_points = num_games * points_per_game → 
  points_per_game = 7 := by
sorry

end NUMINAMATH_CALUDE_points_per_game_l2652_265297


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l2652_265268

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 4) : x^2 + (1 / x^2) = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l2652_265268


namespace NUMINAMATH_CALUDE_paper_clips_count_l2652_265295

/-- The number of paper clips in a storage unit -/
def total_paper_clips (c b p : ℕ) : ℕ :=
  300 * c + 550 * b + 1200 * p

/-- Theorem stating that the total number of paper clips in 3 cartons, 4 boxes, and 2 bags is 5500 -/
theorem paper_clips_count : total_paper_clips 3 4 2 = 5500 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_count_l2652_265295


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l2652_265258

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

theorem f_inequality_solution (x : ℝ) : 
  f x > 2 ↔ x < -7 ∨ (5/3 < x ∧ x < 4) ∨ x > 7 := by sorry

theorem f_minimum_value : 
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l2652_265258


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2652_265200

theorem unique_four_digit_number : ∃! n : ℕ, 
  (1000 ≤ n ∧ n ≤ 9999) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n % 1000 = b^3) ∧
  (∃ c : ℕ, n % 100 = c^4) ∧
  n = 9216 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2652_265200
