import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l815_81550

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2008)
  (h2 : x + 2008 * Real.cos y = 2007)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l815_81550


namespace NUMINAMATH_CALUDE_two_friend_visits_count_l815_81512

/-- Represents a friend with a visitation period -/
structure Friend where
  period : ℕ

/-- Calculates the number of days in a given period where exactly two out of three friends visit -/
def countTwoFriendVisits (f1 f2 f3 : Friend) (totalDays : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 27 days in a 365-day period 
    where two out of three friends visit, given their visitation periods -/
theorem two_friend_visits_count : 
  let max : Friend := { period := 5 }
  let nora : Friend := { period := 6 }
  let olivia : Friend := { period := 7 }
  countTwoFriendVisits max nora olivia 365 = 27 := by
  sorry

end NUMINAMATH_CALUDE_two_friend_visits_count_l815_81512


namespace NUMINAMATH_CALUDE_remaining_card_theorem_l815_81522

/-- Definition of the operation sequence on a stack of cards -/
def operationSequence (n : ℕ) : List ℕ :=
  sorry

/-- L(n) is the number on the remaining card after performing the operation sequence -/
def L (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the form of k for which L(3k) = k -/
theorem remaining_card_theorem (k : ℕ) :
  (L (3 * k) = k) ↔ 
  (∃ j : ℕ, (k = (2 * 3^(6*j) - 2) / 7) ∨ (k = (3^(6*j + 2) - 2) / 7)) :=
by sorry

end NUMINAMATH_CALUDE_remaining_card_theorem_l815_81522


namespace NUMINAMATH_CALUDE_top_card_after_74_shuffles_l815_81562

/-- Represents the order of cards -/
inductive Card
| A
| B
| C
| D
| E

/-- Represents the stack of cards -/
def Stack := List Card

/-- The initial configuration of cards -/
def initial_stack : Stack := [Card.A, Card.B, Card.C, Card.D, Card.E]

/-- Performs one shuffle operation on the stack -/
def shuffle (s : Stack) : Stack :=
  match s with
  | x :: y :: rest => rest ++ [y, x]
  | _ => s

/-- Performs n shuffle operations on the stack -/
def n_shuffles (n : Nat) (s : Stack) : Stack :=
  match n with
  | 0 => s
  | n + 1 => shuffle (n_shuffles n s)

theorem top_card_after_74_shuffles :
  (n_shuffles 74 initial_stack).head? = some Card.E := by
  sorry

end NUMINAMATH_CALUDE_top_card_after_74_shuffles_l815_81562


namespace NUMINAMATH_CALUDE_positive_root_implies_m_value_l815_81502

theorem positive_root_implies_m_value 
  (h : ∃ (x : ℝ), x > 0 ∧ (6 - x) / (x - 3) - (2 * m) / (x - 3) = 0) : 
  m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_positive_root_implies_m_value_l815_81502


namespace NUMINAMATH_CALUDE_amoli_driving_time_l815_81566

-- Define the constants from the problem
def total_distance : ℝ := 369
def amoli_speed : ℝ := 42
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2
def remaining_distance : ℝ := 121

-- Define Amoli's driving time as a variable
def amoli_time : ℝ := 3

-- Theorem statement
theorem amoli_driving_time :
  amoli_speed * amoli_time + anayet_speed * anayet_time = total_distance - remaining_distance :=
by sorry

end NUMINAMATH_CALUDE_amoli_driving_time_l815_81566


namespace NUMINAMATH_CALUDE_minus_one_circle_plus_minus_four_l815_81536

-- Define the ⊕ operation
def circle_plus (a b : ℝ) : ℝ := a + b - a * b

-- Theorem statement
theorem minus_one_circle_plus_minus_four :
  circle_plus (-1) (-4) = -9 := by
  sorry

end NUMINAMATH_CALUDE_minus_one_circle_plus_minus_four_l815_81536


namespace NUMINAMATH_CALUDE_linked_rings_height_l815_81504

/-- Represents the properties of a sequence of linked rings -/
structure LinkedRings where
  thickness : ℝ
  topOutsideDiameter : ℝ
  diameterDecrease : ℝ
  bottomOutsideDiameter : ℝ

/-- Calculates the total height of the linked rings -/
def totalHeight (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total height of the linked rings with given properties is 273 cm -/
theorem linked_rings_height :
  let rings : LinkedRings := {
    thickness := 2,
    topOutsideDiameter := 20,
    diameterDecrease := 0.5,
    bottomOutsideDiameter := 10
  }
  totalHeight rings = 273 := by sorry

end NUMINAMATH_CALUDE_linked_rings_height_l815_81504


namespace NUMINAMATH_CALUDE_range_of_u_l815_81567

theorem range_of_u (x y : ℝ) (h : x^2 / 3 + y^2 = 1) :
  let u := |2*x + y - 4| + |3 - x - 2*y|
  1 ≤ u ∧ u ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_range_of_u_l815_81567


namespace NUMINAMATH_CALUDE_fraction_sum_and_simplification_l815_81525

theorem fraction_sum_and_simplification : 
  (11 : ℚ) / 12 + 7 / 8 + 3 / 4 = 61 / 24 ∧ 
  ∀ (a b : ℤ), (b ≠ 0 ∧ (61 : ℚ) / 24 = a / b) → (|a| = 61 ∧ |b| = 24) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_simplification_l815_81525


namespace NUMINAMATH_CALUDE_inequality_solution_range_l815_81537

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 5 ∧ x^2 + a*x + 4 < 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l815_81537


namespace NUMINAMATH_CALUDE_det_circulant_matrix_l815_81518

def circulant_matrix (n : ℕ) (h : n > 1) (h_odd : Odd n) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => if i = j then 2
           else if (i - j) % n = 2 ∨ (i - j) % n = n - 2 then 1
           else 0

theorem det_circulant_matrix (n : ℕ) (h : n > 1) (h_odd : Odd n) :
  let A := circulant_matrix n h h_odd
  Matrix.det A = 4 := by
  sorry

end NUMINAMATH_CALUDE_det_circulant_matrix_l815_81518


namespace NUMINAMATH_CALUDE_find_A_in_subtraction_l815_81588

/-- Given that AB82 - 9C9 = 493D and A, B, C, D are different digits, prove that A = 5 -/
theorem find_A_in_subtraction (A B C D : ℕ) : 
  A * 1000 + B * 100 + 82 - (9 * 100 + C * 10 + 9) = 4 * 100 + 9 * 10 + D →
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_find_A_in_subtraction_l815_81588


namespace NUMINAMATH_CALUDE_tangent_product_theorem_l815_81553

theorem tangent_product_theorem : 
  Real.tan (20 * π / 180) * Real.tan (40 * π / 180) * Real.tan (60 * π / 180) * Real.tan (80 * π / 180) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_theorem_l815_81553


namespace NUMINAMATH_CALUDE_max_a_value_l815_81589

/-- An even function f defined on ℝ such that f(x) = e^x for x ≥ 0 -/
noncomputable def f : ℝ → ℝ :=
  fun x => if x ≥ 0 then Real.exp x else Real.exp (-x)

theorem max_a_value :
  (∃ a : ℝ, ∀ x ∈ Set.Icc a (a + 1), f (x + a) ≥ f x ^ 2) ∧
  (∀ a : ℝ, a > -3/4 → ∃ x ∈ Set.Icc a (a + 1), f (x + a) < f x ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l815_81589


namespace NUMINAMATH_CALUDE_negation_of_tangent_equality_l815_81503

theorem negation_of_tangent_equality (x : ℝ) :
  (¬ ∀ x : ℝ, Real.tan (-x) = Real.tan x) ↔ (∃ x : ℝ, Real.tan (-x) ≠ Real.tan x) := by sorry

end NUMINAMATH_CALUDE_negation_of_tangent_equality_l815_81503


namespace NUMINAMATH_CALUDE_pencil_cost_l815_81500

/-- Given a pen and a pencil where the pen costs half the price of the pencil,
    and their total cost is $12, prove that the pencil costs $8. -/
theorem pencil_cost (pen_cost pencil_cost : ℝ) : 
  pen_cost = pencil_cost / 2 →
  pen_cost + pencil_cost = 12 →
  pencil_cost = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_cost_l815_81500


namespace NUMINAMATH_CALUDE_hosing_time_is_10_minutes_l815_81540

def dog_cleaning_time (num_shampoos : ℕ) (time_per_shampoo : ℕ) (total_cleaning_time : ℕ) : ℕ :=
  total_cleaning_time - (num_shampoos * time_per_shampoo)

theorem hosing_time_is_10_minutes :
  dog_cleaning_time 3 15 55 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hosing_time_is_10_minutes_l815_81540


namespace NUMINAMATH_CALUDE_min_value_inequality_l815_81507

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 3 * a) (h3 : 3 * b^2 ≤ a * (a + c)) (h4 : a * (a + c) ≤ 5 * b^2) :
  ∃ (x : ℝ), ∀ (y : ℝ), (b - 2*c) / a ≥ x ∧ (b - 2*c) / a = x ↔ b / a = 4/5 ∧ c / a = 11/5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l815_81507


namespace NUMINAMATH_CALUDE_solution_value_l815_81501

theorem solution_value (a b : ℝ) (h : 2 * a - 3 * b - 5 = 0) : 2 * a - 3 * b + 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l815_81501


namespace NUMINAMATH_CALUDE_ice_cream_volume_l815_81552

/-- The volume of ice cream in a cone with a hemispherical top -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let hemisphere_volume := (2 / 3) * π * r^3
  h = 10 ∧ r = 3 → cone_volume + hemisphere_volume = 48 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l815_81552


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l815_81572

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 4/b = 1 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 4/y = 1 ∧ x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l815_81572


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l815_81511

/-- The marshmallow challenge problem -/
theorem marshmallow_challenge 
  (haley : ℕ) 
  (michael : ℕ) 
  (brandon : ℕ) 
  (h1 : haley = 8)
  (h2 : michael = 3 * haley)
  (h3 : haley + michael + brandon = 44) :
  brandon / michael = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l815_81511


namespace NUMINAMATH_CALUDE_consecutive_sum_prime_iff_n_one_or_two_l815_81564

/-- The sum of n consecutive natural numbers starting from k -/
def consecutiveSum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

/-- A natural number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def isPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem consecutive_sum_prime_iff_n_one_or_two :
  ∀ n : ℕ, (∃ k : ℕ, isPrime (consecutiveSum n k)) ↔ n = 1 ∨ n = 2 :=
sorry

end NUMINAMATH_CALUDE_consecutive_sum_prime_iff_n_one_or_two_l815_81564


namespace NUMINAMATH_CALUDE_three_digit_ending_in_seven_not_divisible_by_four_l815_81584

theorem three_digit_ending_in_seven_not_divisible_by_four :
  ∀ N : ℕ, 
  100 ≤ N ∧ N < 1000 ∧ N % 10 = 7 → 
  ¬(N % 4 = 0) := by
sorry

end NUMINAMATH_CALUDE_three_digit_ending_in_seven_not_divisible_by_four_l815_81584


namespace NUMINAMATH_CALUDE_coefficients_of_2x2_plus_x_minus_5_l815_81549

/-- Given a quadratic equation ax² + bx + c = 0, returns a tuple (a, b, c) of its coefficients -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

theorem coefficients_of_2x2_plus_x_minus_5 :
  quadratic_coefficients 2 1 (-5) = (2, 1, -5) := by sorry

end NUMINAMATH_CALUDE_coefficients_of_2x2_plus_x_minus_5_l815_81549


namespace NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l815_81533

/-- Given that the equation (m-1)/(x-1) - x/(x-1) = 0 has a repeated root, prove that m = 2 -/
theorem repeated_root_implies_m_equals_two (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (m - 1) / (x - 1) - x / (x - 1) = 0 ∧
   ∀ y : ℝ, y ≠ 1 → ((m - 1) / (y - 1) - y / (y - 1) = 0 → y = x)) →
  m = 2 :=
by sorry

end NUMINAMATH_CALUDE_repeated_root_implies_m_equals_two_l815_81533


namespace NUMINAMATH_CALUDE_min_tangent_length_l815_81577

/-- Circle C with equation x^2 + y^2 - 2x - 4y + 1 = 0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + 1 = 0}

/-- Line l -/
def line_l : Set (ℝ × ℝ) := sorry

/-- Maximum distance from any point on C to line l is 6 -/
axiom max_distance_to_l (p : ℝ × ℝ) :
  p ∈ circle_C → ∃ (q : ℝ × ℝ), q ∈ line_l ∧ dist p q ≤ 6

/-- Tangent line from a point on l to C -/
def tangent_length (a : ℝ × ℝ) : ℝ := sorry

theorem min_tangent_length :
  ∃ (a : ℝ × ℝ), a ∈ line_l ∧
  (∀ (b : ℝ × ℝ), b ∈ line_l → tangent_length a ≤ tangent_length b) ∧
  tangent_length a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_tangent_length_l815_81577


namespace NUMINAMATH_CALUDE_unique_n_for_equation_l815_81529

theorem unique_n_for_equation : ∃! (n : ℕ+), 
  ∃ (x y : ℕ+), y^2 + x*y + 3*x = n*(x^2 + x*y + 3*y) := by
  sorry

end NUMINAMATH_CALUDE_unique_n_for_equation_l815_81529


namespace NUMINAMATH_CALUDE_new_ratio_is_13_to_7_l815_81595

/-- Represents the farm's animal count before and after the transaction -/
structure FarmCount where
  initialHorses : ℕ
  initialCows : ℕ
  finalHorses : ℕ
  finalCows : ℕ

/-- Checks if the given FarmCount satisfies the problem conditions -/
def validFarmCount (f : FarmCount) : Prop :=
  f.initialHorses = 4 * f.initialCows ∧
  f.finalHorses = f.initialHorses - 15 ∧
  f.finalCows = f.initialCows + 15 ∧
  f.finalHorses = f.finalCows + 30

/-- Theorem stating that the new ratio of horses to cows is 13:7 -/
theorem new_ratio_is_13_to_7 (f : FarmCount) (h : validFarmCount f) :
  13 * f.finalCows = 7 * f.finalHorses :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_13_to_7_l815_81595


namespace NUMINAMATH_CALUDE_clue_distance_range_l815_81548

-- Define the distance to the nearest clue
variable (d : ℝ)

-- Define the statements of Adam, Bianca, and Cole
def adam_statement := d ≥ 8
def bianca_statement := d ≤ 7
def cole_statement := d ≤ 9

-- Theorem to prove the correct range for d
theorem clue_distance_range 
  (h1 : ¬adam_statement d) 
  (h2 : ¬bianca_statement d) 
  (h3 : ¬cole_statement d) : 
  7 < d ∧ d < 8 := by
  sorry


end NUMINAMATH_CALUDE_clue_distance_range_l815_81548


namespace NUMINAMATH_CALUDE_parking_arrangements_count_l815_81515

-- Define the number of parking spaces
def num_spaces : ℕ := 7

-- Define the number of trucks
def num_trucks : ℕ := 2

-- Define the number of buses
def num_buses : ℕ := 2

-- Define a function to calculate the number of parking arrangements
def num_parking_arrangements (spaces : ℕ) (trucks : ℕ) (buses : ℕ) : ℕ :=
  (spaces.choose trucks) * ((spaces - trucks).choose buses) * (trucks.factorial) * (buses.factorial)

-- Theorem statement
theorem parking_arrangements_count :
  num_parking_arrangements num_spaces num_trucks num_buses = 840 := by
  sorry


end NUMINAMATH_CALUDE_parking_arrangements_count_l815_81515


namespace NUMINAMATH_CALUDE_mike_catches_l815_81556

/-- The number of times Joe caught the ball -/
def J : ℕ := 23

/-- The number of times Derek caught the ball -/
def D : ℕ := 2 * J - 4

/-- The number of times Tammy caught the ball -/
def T : ℕ := (D / 3) + 16

/-- The number of times Mike caught the ball -/
def M : ℕ := (2 * T * 120) / 100

theorem mike_catches : M = 72 := by
  sorry

end NUMINAMATH_CALUDE_mike_catches_l815_81556


namespace NUMINAMATH_CALUDE_thieves_cloth_equation_l815_81583

theorem thieves_cloth_equation (x : ℕ+) : 
  (∃ y : ℕ, y = 6 * x + 5 ∧ y = 7 * x - 8) ↔ 
  6 * x + 5 = 7 * x - 8 :=
by sorry

end NUMINAMATH_CALUDE_thieves_cloth_equation_l815_81583


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l815_81565

theorem quadratic_root_difference (r₁ r₂ : ℝ) : 
  2 * r₁^2 - 10 * r₁ + 2 = 0 ∧
  2 * r₂^2 - 10 * r₂ + 2 = 0 ∧
  r₁^2 + r₂^2 = 23 →
  |r₁ - r₂| = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l815_81565


namespace NUMINAMATH_CALUDE_insulation_cost_theorem_l815_81582

def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

def tank_a_area : ℝ := surface_area 3 5 2
def tank_b_area : ℝ := surface_area 4 6 3
def tank_c_area : ℝ := surface_area 6 8 4

def total_area : ℝ := 5 * tank_a_area + 3 * tank_b_area + 2 * tank_c_area

def cost_per_sq_ft : ℝ := 20

def discount_rate : ℝ := 0.2

theorem insulation_cost_theorem :
  let cost_before_discount := total_area * cost_per_sq_ft
  let discount := discount_rate * cost_before_discount
  cost_before_discount - discount = 16800 := by sorry

end NUMINAMATH_CALUDE_insulation_cost_theorem_l815_81582


namespace NUMINAMATH_CALUDE_product_difference_squares_divisible_by_three_l815_81534

theorem product_difference_squares_divisible_by_three (m n : ℤ) :
  ∃ k : ℤ, m * n * (m^2 - n^2) = 3 * k := by
sorry

end NUMINAMATH_CALUDE_product_difference_squares_divisible_by_three_l815_81534


namespace NUMINAMATH_CALUDE_negation_equivalence_l815_81506

theorem negation_equivalence :
  (¬ ∃ x : ℤ, 2*x + x + 1 ≤ 0) ↔ (∀ x : ℤ, 2*x + x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l815_81506


namespace NUMINAMATH_CALUDE_man_rowing_speed_l815_81570

/-- Given a man's speed in still water and downstream speed, calculate his upstream speed -/
theorem man_rowing_speed (v_still : ℝ) (v_downstream : ℝ) (v_upstream : ℝ) : 
  v_still = 31 → v_downstream = 37 → v_upstream = 25 := by
  sorry

#check man_rowing_speed

end NUMINAMATH_CALUDE_man_rowing_speed_l815_81570


namespace NUMINAMATH_CALUDE_largest_lower_bound_l815_81523

theorem largest_lower_bound (x y : ℝ) (h1 : x ≠ y) (h2 : x * y = 2) :
  ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 ≥ 2 ∧
  ∀ C > 2, ∃ x y : ℝ, x ≠ y ∧ x * y = 2 ∧ ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 < C :=
by sorry

end NUMINAMATH_CALUDE_largest_lower_bound_l815_81523


namespace NUMINAMATH_CALUDE_remainder_1234567_div_256_l815_81592

theorem remainder_1234567_div_256 : 1234567 % 256 = 57 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1234567_div_256_l815_81592


namespace NUMINAMATH_CALUDE_exchange_rate_change_is_24_l815_81542

/-- The change in dollar exchange rate from January 1, 2014, to December 31, 2014, rounded to the nearest whole number -/
def exchange_rate_change : ℤ :=
  let initial_rate : ℚ := 32.6587
  let final_rate : ℚ := 56.2584
  let exact_change : ℚ := final_rate - initial_rate
  (exact_change + 1/2).floor

theorem exchange_rate_change_is_24 : exchange_rate_change = 24 := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_change_is_24_l815_81542


namespace NUMINAMATH_CALUDE_steps_ratio_second_to_first_day_l815_81578

/-- Proves that the ratio of steps walked on the second day to the first day is 1:1 -/
theorem steps_ratio_second_to_first_day (
  first_day_steps : ℕ) 
  (second_day_steps : ℕ) 
  (third_day_steps : ℕ) 
  (h1 : first_day_steps = 200 + 300)
  (h2 : third_day_steps = second_day_steps + 100)
  (h3 : first_day_steps + second_day_steps + third_day_steps = 1600) :
  second_day_steps = first_day_steps :=
sorry

end NUMINAMATH_CALUDE_steps_ratio_second_to_first_day_l815_81578


namespace NUMINAMATH_CALUDE_max_triangles_in_7x7_grid_triangle_l815_81521

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  leg_length : ℕ
  is_right_angled : Bool

/-- Counts the maximum number of triangles in a grid triangle -/
def count_max_triangles (t : GridTriangle) : ℕ := sorry

/-- The main theorem stating the maximum number of triangles in a 7x7 grid triangle -/
theorem max_triangles_in_7x7_grid_triangle :
  ∀ (t : GridTriangle),
    t.leg_length = 7 →
    t.is_right_angled = true →
    count_max_triangles t = 28 := by sorry

end NUMINAMATH_CALUDE_max_triangles_in_7x7_grid_triangle_l815_81521


namespace NUMINAMATH_CALUDE_cable_theorem_l815_81579

def cable_problem (basic_cost movie_cost sports_cost_diff : ℕ) : Prop :=
  let sports_cost := movie_cost - sports_cost_diff
  let total_cost := basic_cost + movie_cost + sports_cost
  total_cost = 36

theorem cable_theorem : cable_problem 15 12 3 :=
sorry

end NUMINAMATH_CALUDE_cable_theorem_l815_81579


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l815_81527

/-- Given a geometric sequence of positive integers with first term 3 and fourth term 240,
    prove that the fifth term is 768. -/
theorem geometric_sequence_fifth_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 3 →                            -- first term is 3
  a 4 = 240 →                          -- fourth term is 240
  a 5 = 768 :=                         -- conclusion: fifth term is 768
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l815_81527


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0000064_l815_81539

theorem scientific_notation_of_0_0000064 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.0000064 = a * (10 : ℝ) ^ n ∧ a = 6.4 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0000064_l815_81539


namespace NUMINAMATH_CALUDE_last_segment_speed_prove_last_segment_speed_l815_81573

theorem last_segment_speed (total_distance : ℝ) (total_time : ℝ) 
  (speed1 : ℝ) (speed2 : ℝ) (speed3 : ℝ) : ℝ :=
  let total_segments : ℝ := 4
  let segment_time : ℝ := total_time / total_segments
  let overall_avg_speed : ℝ := total_distance / total_time
  let last_segment_speed : ℝ := 
    total_segments * overall_avg_speed - (speed1 + speed2 + speed3)
  last_segment_speed

theorem prove_last_segment_speed : 
  last_segment_speed 160 2 55 75 60 = 130 := by
  sorry

end NUMINAMATH_CALUDE_last_segment_speed_prove_last_segment_speed_l815_81573


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_94_nines_94_fours_l815_81509

/-- A number consisting of n repeated digits d -/
def repeatedDigits (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_product_94_nines_94_fours :
  sumOfDigits (repeatedDigits 94 9 * repeatedDigits 94 4) = 846 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_94_nines_94_fours_l815_81509


namespace NUMINAMATH_CALUDE_inscribed_parallelogram_exists_l815_81598

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

/-- Checks if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a parallelogram is inscribed in a quadrilateral -/
def Parallelogram.inscribed_in (p : Parallelogram) (q : Quadrilateral) : Prop :=
  (p.P.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.P.on_line (Line.mk q.A.x q.B.y (-1)) ∨
   p.P.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.P.on_line (Line.mk q.C.x q.C.y (-1))) ∧
  (p.Q.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.Q.on_line (Line.mk q.A.x q.B.y (-1)) ∨
   p.Q.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.Q.on_line (Line.mk q.C.x q.C.y (-1))) ∧
  (p.R.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.R.on_line (Line.mk q.B.x q.C.y (-1)) ∨
   p.R.on_line (Line.mk q.C.x q.C.y (-1)) ∨ p.R.on_line (Line.mk q.D.x q.D.y (-1))) ∧
  (p.S.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.S.on_line (Line.mk q.C.x q.D.y (-1)) ∨
   p.S.on_line (Line.mk q.D.x q.D.y (-1)) ∨ p.S.on_line (Line.mk q.A.x q.D.y (-1)))

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem inscribed_parallelogram_exists (ABCD : Quadrilateral) 
  (E : Point) (F : Point) (BF CE : Line) :
  E.on_line (Line.mk ABCD.A.x ABCD.B.y (-1)) →
  F.on_line (Line.mk ABCD.C.x ABCD.D.y (-1)) →
  ∃ (PQRS : Parallelogram),
    PQRS.inscribed_in ABCD ∧
    Line.parallel (Line.mk PQRS.P.x PQRS.Q.y (-1)) BF ∧
    Line.parallel (Line.mk PQRS.Q.x PQRS.R.y (-1)) CE :=
  sorry

end NUMINAMATH_CALUDE_inscribed_parallelogram_exists_l815_81598


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l815_81528

/-- The number of books Sandy bought from the first shop -/
def books_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop -/
def amount_first_shop : ℕ := 1380

/-- The number of books Sandy bought from the second shop -/
def books_second_shop : ℕ := 55

/-- The amount Sandy spent at the second shop -/
def amount_second_shop : ℕ := 900

/-- The average price per book -/
def average_price : ℕ := 19

theorem sandy_book_purchase :
  books_first_shop = 65 ∧
  (amount_first_shop + amount_second_shop : ℚ) / (books_first_shop + books_second_shop) = average_price := by
  sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l815_81528


namespace NUMINAMATH_CALUDE_profit_calculation_l815_81599

/-- Calculates the total profit of a business given the investments and one partner's share of the profit -/
def calculate_total_profit (investment_A investment_B investment_C share_A : ℕ) : ℕ :=
  let ratio_A := investment_A / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let ratio_B := investment_B / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let ratio_C := investment_C / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let total_ratio := ratio_A + ratio_B + ratio_C
  (share_A * total_ratio) / ratio_A

theorem profit_calculation (investment_A investment_B investment_C share_A : ℕ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : share_A = 3900) :
  calculate_total_profit investment_A investment_B investment_C share_A = 13000 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l815_81599


namespace NUMINAMATH_CALUDE_house_rent_percentage_l815_81560

def monthly_salary : ℝ := 12500
def food_percentage : ℝ := 40
def entertainment_percentage : ℝ := 10
def conveyance_percentage : ℝ := 10
def savings : ℝ := 2500

theorem house_rent_percentage :
  let total_percentage : ℝ := food_percentage + entertainment_percentage + conveyance_percentage
  let spent_amount : ℝ := monthly_salary - savings
  let savings_percentage : ℝ := (savings / monthly_salary) * 100
  let remaining_percentage : ℝ := 100 - total_percentage - savings_percentage
  remaining_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_house_rent_percentage_l815_81560


namespace NUMINAMATH_CALUDE_rectangle_side_length_l815_81510

theorem rectangle_side_length (square_side : ℝ) (rectangle_width : ℝ) :
  square_side = 5 →
  rectangle_width = 4 →
  square_side * square_side = rectangle_width * (25 / rectangle_width) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l815_81510


namespace NUMINAMATH_CALUDE_xy_squared_sum_l815_81581

theorem xy_squared_sum (x y : ℝ) 
  (h1 : 1/x + 1/y = 3) 
  (h2 : x*y + x + y = 4) : 
  x^2*y + x*y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_xy_squared_sum_l815_81581


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l815_81555

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y : ℝ, y < 1/4 → 10*y^2 - 90*y + 20 ≠ 0) ∧ 
  (10*(1/4)^2 - 90*(1/4) + 20 = 0) := by
sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l815_81555


namespace NUMINAMATH_CALUDE_g_eval_sqrt_half_l815_81597

noncomputable def g (x : ℝ) : ℝ := Real.arccos (x^2) * Real.arcsin (x^2)

theorem g_eval_sqrt_half : g (1 / Real.sqrt 2) = π^2 / 18 := by
  sorry

end NUMINAMATH_CALUDE_g_eval_sqrt_half_l815_81597


namespace NUMINAMATH_CALUDE_janet_dermatologist_distance_l815_81505

def dermatologist_distance (x : ℝ) := x
def gynecologist_distance : ℝ := 50
def car_efficiency : ℝ := 20
def gas_used : ℝ := 8

theorem janet_dermatologist_distance :
  ∃ x : ℝ, 
    dermatologist_distance x = 30 ∧ 
    2 * dermatologist_distance x + 2 * gynecologist_distance = car_efficiency * gas_used :=
by sorry

end NUMINAMATH_CALUDE_janet_dermatologist_distance_l815_81505


namespace NUMINAMATH_CALUDE_abs_z_eq_sqrt_10_div_2_l815_81557

theorem abs_z_eq_sqrt_10_div_2 (z : ℂ) (h : (1 - Complex.I) * z = 1 + 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_eq_sqrt_10_div_2_l815_81557


namespace NUMINAMATH_CALUDE_discounted_cork_price_cork_price_is_0_044_l815_81541

/-- The price of a cork after discount, given the price of wine bottles with and without cork, and the discount rate. -/
theorem discounted_cork_price 
  (price_with_cork : ℝ) 
  (price_difference : ℝ) 
  (discount_rate : ℝ) : ℝ :=
  let original_cork_price := (price_with_cork - price_difference) / 2
  let discounted_cork_price := original_cork_price * (1 - discount_rate)
  discounted_cork_price

/-- The discounted price of a cork is $0.044 under the given conditions. -/
theorem cork_price_is_0_044 : 
  discounted_cork_price 2.10 2.00 0.12 = 0.044 := by
  sorry

end NUMINAMATH_CALUDE_discounted_cork_price_cork_price_is_0_044_l815_81541


namespace NUMINAMATH_CALUDE_min_plates_for_seven_colors_l815_81547

/-- The minimum number of plates needed to guarantee at least three matching pairs -/
def min_plates_for_three_pairs (num_colors : ℕ) : ℕ :=
  3 * num_colors + 3

/-- Theorem stating that given 7 different colors of plates, 
    the minimum number of plates needed to guarantee at least three matching pairs is 24 -/
theorem min_plates_for_seven_colors : 
  min_plates_for_three_pairs 7 = 24 := by
  sorry

#eval min_plates_for_three_pairs 7

end NUMINAMATH_CALUDE_min_plates_for_seven_colors_l815_81547


namespace NUMINAMATH_CALUDE_modulus_of_z_l815_81576

def z : ℂ := (2 + Complex.I) * (1 - Complex.I)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l815_81576


namespace NUMINAMATH_CALUDE_a_in_P_and_b_in_Q_l815_81531

-- Define the sets P and Q
def P : Set ℤ := {x | ∃ m : ℤ, x = 2 * m + 1}
def Q : Set ℤ := {y | ∃ n : ℤ, y = 2 * n}

-- Define the theorem
theorem a_in_P_and_b_in_Q (x₀ y₀ : ℤ) (hx : x₀ ∈ P) (hy : y₀ ∈ Q) :
  let a := x₀ + y₀
  let b := x₀ * y₀
  a ∈ P ∧ b ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_a_in_P_and_b_in_Q_l815_81531


namespace NUMINAMATH_CALUDE_smallest_real_number_l815_81580

theorem smallest_real_number (S : Set ℝ) (h : S = {-2, 2, 0, 1}) : 
  ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -2 := by sorry

end NUMINAMATH_CALUDE_smallest_real_number_l815_81580


namespace NUMINAMATH_CALUDE_sum_of_e_values_l815_81590

theorem sum_of_e_values (e : ℝ) : (|2 - e| = 5) → (∃ (e₁ e₂ : ℝ), (|2 - e₁| = 5 ∧ |2 - e₂| = 5 ∧ e₁ + e₂ = 4)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_e_values_l815_81590


namespace NUMINAMATH_CALUDE_combined_perimeters_of_squares_l815_81524

theorem combined_perimeters_of_squares (square1 square2 : Real) 
  (h1 : square1 * square1 = 36) 
  (h2 : square2 * square2 = 49) : 
  4 * square1 + 4 * square2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_combined_perimeters_of_squares_l815_81524


namespace NUMINAMATH_CALUDE_primes_up_to_100_l815_81513

theorem primes_up_to_100 : 
  {p : ℕ | Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 100} = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} := by
  sorry

end NUMINAMATH_CALUDE_primes_up_to_100_l815_81513


namespace NUMINAMATH_CALUDE_two_digit_number_five_times_sum_of_digits_l815_81554

theorem two_digit_number_five_times_sum_of_digits : 
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ n = 5 * (n / 10 + n % 10) :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_five_times_sum_of_digits_l815_81554


namespace NUMINAMATH_CALUDE_multiply_by_seven_equals_98_l815_81514

theorem multiply_by_seven_equals_98 (x : ℝ) : x * 7 = 98 ↔ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_equals_98_l815_81514


namespace NUMINAMATH_CALUDE_complex_fraction_value_l815_81543

theorem complex_fraction_value (a : ℝ) (z : ℂ) : 
  z = (a^2 - 1) + (a - 1) * Complex.I → z.re = 0 → (a^2 + Complex.I) / (1 + a * Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l815_81543


namespace NUMINAMATH_CALUDE_emilys_journey_l815_81551

theorem emilys_journey (total : ℝ) 
  (h1 : total / 5 + 30 + total / 3 + total / 6 = total) : total = 100 := by
  sorry

end NUMINAMATH_CALUDE_emilys_journey_l815_81551


namespace NUMINAMATH_CALUDE_property_satisfied_l815_81586

theorem property_satisfied (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_property_satisfied_l815_81586


namespace NUMINAMATH_CALUDE_pancake_accumulation_time_l815_81574

/-- Represents the pancake baking and eating scenario -/
structure PancakeScenario where
  mom_rate : ℚ     -- Mom's baking rate in pancakes per minute
  anya_rate : ℚ    -- Anya's baking rate in pancakes per minute
  andrey_rate : ℚ  -- Andrey's eating rate in pancakes per minute

/-- Calculates the time needed to accumulate a given number of pancakes -/
def time_to_accumulate (scenario : PancakeScenario) (target : ℚ) : ℚ :=
  target / (scenario.mom_rate + scenario.anya_rate - scenario.andrey_rate)

/-- Theorem stating the time to accumulate 100 pancakes is 24 minutes -/
theorem pancake_accumulation_time :
  let scenario : PancakeScenario := {
    mom_rate := 100 / 30,    -- 100 pancakes in 30 minutes
    anya_rate := 100 / 40,   -- 100 pancakes in 40 minutes
    andrey_rate := 100 / 60  -- 100 pancakes in 60 minutes
  }
  time_to_accumulate scenario 100 = 24 := by sorry

end NUMINAMATH_CALUDE_pancake_accumulation_time_l815_81574


namespace NUMINAMATH_CALUDE_f_properties_l815_81559

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define g' as the derivative of g
variable (g' : ℝ → ℝ)

-- State the conditions
axiom cond1 : ∀ x, f x + g' x - 10 = 0
axiom cond2 : ∀ x, f x - g' (4 - x) - 10 = 0
axiom g_even : ∀ x, g (-x) = g x

-- State the theorem
theorem f_properties :
  (f 1 + f 3 = 20) ∧ (f 4 = 10) ∧ (f 2022 = 10) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l815_81559


namespace NUMINAMATH_CALUDE_parade_formation_l815_81535

theorem parade_formation (total : Nat) (red_flower : Nat) (red_balloon : Nat) (yellow_green : Nat)
  (h1 : total = 100)
  (h2 : red_flower = 42)
  (h3 : red_balloon = 63)
  (h4 : yellow_green = 28) :
  total - red_balloon - yellow_green + red_flower = 33 := by
  sorry

end NUMINAMATH_CALUDE_parade_formation_l815_81535


namespace NUMINAMATH_CALUDE_g_continuity_condition_l815_81593

/-- The function g(x) = 5x - 3 -/
def g (x : ℝ) : ℝ := 5 * x - 3

/-- The statement is true if and only if d ≤ c/5 -/
theorem g_continuity_condition (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, |x - 1| < d → |g x - 1| < c) ↔ d ≤ c / 5 := by
  sorry

end NUMINAMATH_CALUDE_g_continuity_condition_l815_81593


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l815_81530

/-- Given a triangle with side lengths a, b, and c, the sum of the ratios of each side length
    to the square root of twice the sum of squares of the other two sides minus the square
    of the current side is greater than or equal to the square root of 3. -/
theorem triangle_inequality_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a / Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) +
  (b / Real.sqrt (2 * c^2 + 2 * a^2 - b^2)) +
  (c / Real.sqrt (2 * a^2 + 2 * b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l815_81530


namespace NUMINAMATH_CALUDE_distance_between_vertices_parabolas_distance_l815_81568

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 2) = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop :=
  y = -(1/12) * x^2 + 3

def parabola2 (x y : ℝ) : Prop :=
  y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem: The distance between the vertices is 4
theorem distance_between_vertices : 
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 4 := by
  sorry

-- Main theorem
theorem parabolas_distance : ∃ (x1 y1 x2 y2 : ℝ),
  equation x1 y1 ∧ equation x2 y2 ∧
  parabola1 x1 y1 ∧ parabola2 x2 y2 ∧
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_parabolas_distance_l815_81568


namespace NUMINAMATH_CALUDE_f_is_quadratic_l815_81591

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x - x^2 -/
def f (x : ℝ) : ℝ := 2*x - x^2

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l815_81591


namespace NUMINAMATH_CALUDE_min_value_cos_half_theta_times_two_minus_sin_theta_l815_81596

theorem min_value_cos_half_theta_times_two_minus_sin_theta (θ : Real) (h : 0 < θ ∧ θ < π) :
  ∃ (min : Real), min = 0 ∧ ∀ θ', 0 < θ' ∧ θ' < π →
    min ≤ Real.cos (θ' / 2) * (2 - Real.sin θ') :=
by sorry

end NUMINAMATH_CALUDE_min_value_cos_half_theta_times_two_minus_sin_theta_l815_81596


namespace NUMINAMATH_CALUDE_product_sum_coefficients_l815_81558

theorem product_sum_coefficients :
  ∀ (A B C D : ℝ), 
  (∀ x : ℝ, (2 * x^2 - 3 * x + 5) * (5 - 3 * x) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 8 :=
by sorry

end NUMINAMATH_CALUDE_product_sum_coefficients_l815_81558


namespace NUMINAMATH_CALUDE_min_value_2x_plus_y_l815_81545

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) :
  2 * x + y ≥ 12 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ + 6 = x₀ * y₀ ∧ 2 * x₀ + y₀ = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_2x_plus_y_l815_81545


namespace NUMINAMATH_CALUDE_gcd_of_nine_digit_numbers_l815_81544

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n ≤ 999999999) ∧
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ d₉ : ℕ),
    d₁ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₂ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₃ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₄ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₅ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₆ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₇ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₈ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₉ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ d₁ ≠ d₇ ∧ d₁ ≠ d₈ ∧ d₁ ≠ d₉ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ d₂ ≠ d₇ ∧ d₂ ≠ d₈ ∧ d₂ ≠ d₉ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ d₃ ≠ d₇ ∧ d₃ ≠ d₈ ∧ d₃ ≠ d₉ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ d₄ ≠ d₇ ∧ d₄ ≠ d₈ ∧ d₄ ≠ d₉ ∧
    d₅ ≠ d₆ ∧ d₅ ≠ d₇ ∧ d₅ ≠ d₈ ∧ d₅ ≠ d₉ ∧
    d₆ ≠ d₇ ∧ d₆ ≠ d₈ ∧ d₆ ≠ d₉ ∧
    d₇ ≠ d₈ ∧ d₇ ≠ d₉ ∧
    d₈ ≠ d₉ ∧
    n = d₁ * 100000000 + d₂ * 10000000 + d₃ * 1000000 + d₄ * 100000 + d₅ * 10000 + d₆ * 1000 + d₇ * 100 + d₈ * 10 + d₉

theorem gcd_of_nine_digit_numbers :
  ∃ (g : ℕ), g > 0 ∧ (∀ (n : ℕ), is_valid_number n → g ∣ n) ∧
  (∀ (d : ℕ), d > 0 → (∀ (n : ℕ), is_valid_number n → d ∣ n) → d ≤ g) ∧
  g = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_nine_digit_numbers_l815_81544


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l815_81575

theorem imaginary_part_of_one_minus_i_cubed (i : ℂ) : 
  Complex.im ((1 - i)^3) = -2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_cubed_l815_81575


namespace NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l815_81520

theorem max_men_with_all_items_and_married 
  (total_men : ℕ) 
  (married_men : ℕ) 
  (men_with_tv : ℕ) 
  (men_with_radio : ℕ) 
  (men_with_ac : ℕ) 
  (h_total : total_men = 100)
  (h_married : married_men = 85)
  (h_tv : men_with_tv = 75)
  (h_radio : men_with_radio = 85)
  (h_ac : men_with_ac = 70)
  : ∃ (max_all_items_married : ℕ), 
    max_all_items_married ≤ 70 ∧ 
    max_all_items_married ≤ married_men ∧
    max_all_items_married ≤ men_with_tv ∧
    max_all_items_married ≤ men_with_radio ∧
    max_all_items_married ≤ men_with_ac :=
by sorry

end NUMINAMATH_CALUDE_max_men_with_all_items_and_married_l815_81520


namespace NUMINAMATH_CALUDE_set_A_equality_l815_81571

def A : Set ℕ := {x | x ≤ 4}

theorem set_A_equality : A = {0, 1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_A_equality_l815_81571


namespace NUMINAMATH_CALUDE_largest_number_in_ratio_l815_81538

theorem largest_number_in_ratio (a b c d : ℕ) (h_ratio : a * 3 = b * 2 ∧ b * 4 = c * 3 ∧ c * 5 = d * 4) 
  (h_sum : a + b + c + d = 1344) : d = 480 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_in_ratio_l815_81538


namespace NUMINAMATH_CALUDE_root_difference_l815_81563

theorem root_difference (p q : ℝ) (h : p ≠ q) :
  let r := (p + q + Real.sqrt ((p - q)^2)) / 2
  let s := (p + q - Real.sqrt ((p - q)^2)) / 2
  r - s = |p - q| := by
sorry

end NUMINAMATH_CALUDE_root_difference_l815_81563


namespace NUMINAMATH_CALUDE_dessert_and_coffee_percentage_l815_81519

theorem dessert_and_coffee_percentage :
  let dessert_percentage : ℝ := 100 - 25.00000000000001
  let dessert_and_coffee_ratio : ℝ := 1 - 0.2
  dessert_and_coffee_ratio * dessert_percentage = 59.999999999999992 :=
by sorry

end NUMINAMATH_CALUDE_dessert_and_coffee_percentage_l815_81519


namespace NUMINAMATH_CALUDE_percent_increase_revenue_l815_81517

/-- Given two positive real numbers M and N representing revenues in millions for two consecutive years,
    this theorem states that the percent increase in revenue relative to the sum of the revenues of both years
    is equal to 100 * (M - N) / (M + N). -/
theorem percent_increase_revenue (M N : ℝ) (hM : M > 0) (hN : N > 0) :
  (M - N) / (M + N) * 100 = 100 * (M - N) / (M + N) := by sorry

end NUMINAMATH_CALUDE_percent_increase_revenue_l815_81517


namespace NUMINAMATH_CALUDE_megan_seashells_l815_81587

/-- Given that Megan has 19 seashells and wants to have 25 seashells in total,
    prove that she needs to find 6 more seashells. -/
theorem megan_seashells (current : ℕ) (target : ℕ) (h1 : current = 19) (h2 : target = 25) :
  target - current = 6 := by
  sorry

end NUMINAMATH_CALUDE_megan_seashells_l815_81587


namespace NUMINAMATH_CALUDE_particle_speed_l815_81532

/-- A particle moves in a 2D plane. Its position at time t is given by (3t + 1, -2t + 5). 
    The theorem states that the speed of the particle is √13 units of distance per unit of time. -/
theorem particle_speed (t : ℝ) : 
  let position := fun (t : ℝ) => (3 * t + 1, -2 * t + 5)
  let velocity := fun (t : ℝ) => (3, -2)
  let speed := Real.sqrt (3^2 + (-2)^2)
  speed = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_particle_speed_l815_81532


namespace NUMINAMATH_CALUDE_inequality_system_solution_l815_81508

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 3 * x - 2 ∧ 3 * (x - 2) - x ≤ 4) ↔ (3 < x ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l815_81508


namespace NUMINAMATH_CALUDE_prob_even_sum_is_two_fifths_l815_81546

/-- A card is represented by a natural number between 1 and 5 -/
def Card : Type := { n : ℕ // 1 ≤ n ∧ n ≤ 5 }

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- The function that determines if the sum of two cards is even -/
def isEvenSum (c1 c2 : Card) : Prop := Even (c1.val + c2.val)

/-- The set of all pairs of cards -/
def allPairs : Finset (Card × Card) := sorry

/-- The set of all pairs of cards with even sum -/
def evenSumPairs : Finset (Card × Card) := sorry

/-- The probability of drawing two cards with even sum -/
noncomputable def probEvenSum : ℚ := (Finset.card evenSumPairs : ℚ) / (Finset.card allPairs : ℚ)

theorem prob_even_sum_is_two_fifths : probEvenSum = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_is_two_fifths_l815_81546


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l815_81585

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32 = 3 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2 = -7 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l815_81585


namespace NUMINAMATH_CALUDE_calculation_proof_l815_81561

theorem calculation_proof : (-1)^2024 - 1/2 * (8 - (-2)^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l815_81561


namespace NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l815_81526

/-- Two integers are consecutive odd numbers if their difference is 2 and they are both odd -/
def ConsecutiveOddNumbers (m n : ℤ) : Prop :=
  m - n = 2 ∧ Odd m ∧ Odd n

/-- The largest divisor of m^2 - n^2 for consecutive odd numbers m and n where n < m is 8 -/
theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (h1 : ConsecutiveOddNumbers m n) (h2 : n < m) : 
  (∃ (k : ℤ), m^2 - n^2 = 8 * k) ∧ 
  (∀ (d : ℤ), d > 8 → ¬(∀ (j : ℤ), m^2 - n^2 = d * j)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l815_81526


namespace NUMINAMATH_CALUDE_scientific_notation_of_388800_l815_81569

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_388800 :
  toScientificNotation 388800 = ScientificNotation.mk 3.888 5 sorry := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_388800_l815_81569


namespace NUMINAMATH_CALUDE_sequence_equality_l815_81516

def x : ℕ → ℚ
  | 0 => 1
  | n + 1 => x n / (2 + x n)

def y : ℕ → ℚ
  | 0 => 1
  | n + 1 => y n ^ 2 / (1 + 2 * y n)

theorem sequence_equality (n : ℕ) : y n = x (2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l815_81516


namespace NUMINAMATH_CALUDE_lyndees_chicken_pieces_l815_81594

/-- Given the total number of chicken pieces, the number of friends, and the number of pieces each friend ate,
    calculate the number of pieces Lyndee ate. -/
theorem lyndees_chicken_pieces (total_pieces friends_pieces friends : ℕ) : 
  total_pieces - (friends_pieces * friends) = total_pieces - (friends_pieces * friends) := by
  sorry

#check lyndees_chicken_pieces 11 2 5

end NUMINAMATH_CALUDE_lyndees_chicken_pieces_l815_81594
