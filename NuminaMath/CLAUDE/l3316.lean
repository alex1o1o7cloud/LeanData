import Mathlib

namespace NUMINAMATH_CALUDE_percentage_problem_l3316_331688

theorem percentage_problem (P : ℝ) (number : ℝ) : 
  number = 40 →
  P = (0.5 * number) + 10 →
  P = 30 :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l3316_331688


namespace NUMINAMATH_CALUDE_kolya_always_wins_l3316_331686

/-- Represents a player's move in the game -/
inductive Move
| ChangeA (delta : Int) : Move
| ChangeB (delta : Int) : Move

/-- Represents the state of the game -/
structure GameState where
  a : Int
  b : Int

/-- Defines a valid move for Petya -/
def validPetyaMove (m : Move) : Prop :=
  match m with
  | Move.ChangeA delta => delta = 1 ∨ delta = -1
  | Move.ChangeB delta => delta = 1 ∨ delta = -1

/-- Defines a valid move for Kolya -/
def validKolyaMove (m : Move) : Prop :=
  match m with
  | Move.ChangeA delta => delta = 1 ∨ delta = -1 ∨ delta = 3 ∨ delta = -3
  | Move.ChangeB delta => delta = 1 ∨ delta = -1 ∨ delta = 3 ∨ delta = -3

/-- Applies a move to the game state -/
def applyMove (state : GameState) (m : Move) : GameState :=
  match m with
  | Move.ChangeA delta => { state with a := state.a + delta }
  | Move.ChangeB delta => { state with b := state.b + delta }

/-- Checks if the polynomial has integer roots -/
def hasIntegerRoots (state : GameState) : Prop :=
  ∃ x y : Int, x^2 + state.a * x + state.b = 0 ∧ y^2 + state.a * y + state.b = 0 ∧ x ≠ y

/-- Theorem stating Kolya can always win -/
theorem kolya_always_wins :
  ∀ (initial : GameState),
  ∃ (kolyaMoves : List Move),
    (∀ m ∈ kolyaMoves, validKolyaMove m) ∧
    (∀ (petyaMoves : List Move),
      (petyaMoves.length = kolyaMoves.length) →
      (∀ m ∈ petyaMoves, validPetyaMove m) →
      ∃ (finalState : GameState),
        finalState = (kolyaMoves.zip petyaMoves).foldl
          (λ state (km, pm) => applyMove (applyMove state pm) km)
          initial ∧
        hasIntegerRoots finalState) :=
sorry

end NUMINAMATH_CALUDE_kolya_always_wins_l3316_331686


namespace NUMINAMATH_CALUDE_multiplication_simplification_l3316_331618

theorem multiplication_simplification : 9 * (1 / 13) * 26 = 18 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_simplification_l3316_331618


namespace NUMINAMATH_CALUDE_hydra_disconnect_l3316_331634

/-- A graph representing a hydra -/
structure Hydra where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  vertex_count : vertices.card = 100

/-- Invert a vertex in the hydra -/
def invert_vertex (H : Hydra) (v : Nat) : Hydra :=
  sorry

/-- Check if the hydra is disconnected -/
def is_disconnected (H : Hydra) : Prop :=
  sorry

/-- Main theorem: Any 100-vertex hydra can be disconnected in at most 10 inversions -/
theorem hydra_disconnect (H : Hydra) :
  ∃ (inversions : List Nat), inversions.length ≤ 10 ∧
    is_disconnected (inversions.foldl invert_vertex H) :=
  sorry

end NUMINAMATH_CALUDE_hydra_disconnect_l3316_331634


namespace NUMINAMATH_CALUDE_simplify_inverse_product_l3316_331666

theorem simplify_inverse_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (((1 : ℝ) / a) * ((1 : ℝ) / (b + c)))⁻¹ = a * (b + c) := by sorry

end NUMINAMATH_CALUDE_simplify_inverse_product_l3316_331666


namespace NUMINAMATH_CALUDE_cylinder_equal_volume_increase_l3316_331662

/-- Theorem: For a cylinder with radius 6 inches and height 4 inches, 
    the value of x that satisfies the equation π(R+x)²H = πR²(H+2x) is 6 inches. -/
theorem cylinder_equal_volume_increase (π : ℝ) : 
  ∃ (x : ℝ), x = 6 ∧ π * (6 + x)^2 * 4 = π * 6^2 * (4 + 2*x) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_equal_volume_increase_l3316_331662


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_l3316_331605

def smallest_one_digit_primes : List Nat := [2, 3]
def smallest_two_digit_prime : Nat := 11

theorem product_of_smallest_primes :
  (smallest_one_digit_primes.prod) * smallest_two_digit_prime = 66 := by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_l3316_331605


namespace NUMINAMATH_CALUDE_mary_overtime_rate_increase_l3316_331653

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtime_rate_increase (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (max_earnings : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let regular_earnings := regular_hours * regular_rate
  let overtime_earnings := max_earnings - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  ((overtime_rate - regular_rate) / regular_rate) * 100

/-- The percentage increase in overtime rate for Mary's work schedule -/
theorem mary_overtime_rate_increase :
  overtime_rate_increase 80 20 8 760 = 25 := by
  sorry

end NUMINAMATH_CALUDE_mary_overtime_rate_increase_l3316_331653


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3316_331643

theorem fraction_sum_equality : 
  (3 : ℚ) / 5 + (2 : ℚ) / 3 + (1 + (1 : ℚ) / 15) = 2 + (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3316_331643


namespace NUMINAMATH_CALUDE_initial_books_count_l3316_331625

def books_sold : ℕ := 26
def books_left : ℕ := 7

theorem initial_books_count :
  books_sold + books_left = 33 := by sorry

end NUMINAMATH_CALUDE_initial_books_count_l3316_331625


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l3316_331664

theorem smallest_positive_integer_ending_in_3_divisible_by_11 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 11 = 0 → n ≤ m :=
by
  use 33
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l3316_331664


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l3316_331629

/-- Given an arithmetic sequence where a₁ = -3 and a₂ = 1, prove that a₆ = 17 -/
theorem sixth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℤ), 
    a 1 = -3 → 
    a 2 = 1 → 
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) → 
    a 6 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l3316_331629


namespace NUMINAMATH_CALUDE_duck_travel_east_l3316_331679

def days_to_south : ℕ := 40
def days_to_north : ℕ := 2 * days_to_south
def total_days : ℕ := 180

def days_to_east : ℕ := total_days - days_to_south - days_to_north

theorem duck_travel_east : days_to_east = 60 := by
  sorry

end NUMINAMATH_CALUDE_duck_travel_east_l3316_331679


namespace NUMINAMATH_CALUDE_relay_arrangements_verify_arrangements_l3316_331631

def total_athletes : ℕ := 8
def relay_positions : ℕ := 4

def arrangements_condition1 : ℕ := 60
def arrangements_condition2 : ℕ := 480
def arrangements_condition3 : ℕ := 180

/-- Theorem stating the number of arrangements for each condition -/
theorem relay_arrangements :
  (arrangements_condition1 = 60) ∧
  (arrangements_condition2 = 480) ∧
  (arrangements_condition3 = 180) := by
  sorry

/-- Function to calculate the number of arrangements for condition 1 -/
def calc_arrangements_condition1 : ℕ :=
  2 * 1 * 6 * 5

/-- Function to calculate the number of arrangements for condition 2 -/
def calc_arrangements_condition2 : ℕ :=
  2 * 2 * 6 * 5 * 4

/-- Function to calculate the number of arrangements for condition 3 -/
def calc_arrangements_condition3 : ℕ :=
  2 * 1 * (6 * 5 / (2 * 1)) * 3 * 2 * 1

/-- Theorem proving that the calculated arrangements match the given ones -/
theorem verify_arrangements :
  (calc_arrangements_condition1 = arrangements_condition1) ∧
  (calc_arrangements_condition2 = arrangements_condition2) ∧
  (calc_arrangements_condition3 = arrangements_condition3) := by
  sorry

end NUMINAMATH_CALUDE_relay_arrangements_verify_arrangements_l3316_331631


namespace NUMINAMATH_CALUDE_golden_triangle_ratio_l3316_331699

theorem golden_triangle_ratio (t : ℝ) (h : t = (Real.sqrt 5 - 1) / 2) :
  (1 - 2 * Real.sin (27 * π / 180) ^ 2) / (2 * t * Real.sqrt (4 - t^2)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_golden_triangle_ratio_l3316_331699


namespace NUMINAMATH_CALUDE_opposite_numbers_equation_product_l3316_331614

theorem opposite_numbers_equation_product : ∀ x : ℤ, 
  (3 * x - 2 * (-x) = 30) → (x * (-x) = -36) := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_equation_product_l3316_331614


namespace NUMINAMATH_CALUDE_initially_calculated_average_weight_l3316_331601

/-- Given a class of boys with a misread weight, prove the initially calculated average weight. -/
theorem initially_calculated_average_weight
  (n : ℕ) -- number of boys
  (correct_avg : ℝ) -- correct average weight
  (misread_weight : ℝ) -- misread weight
  (correct_weight : ℝ) -- correct weight
  (h1 : n = 20)
  (h2 : correct_avg = 58.7)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 62)
  : ∃ (initial_avg : ℝ), initial_avg = 58.4 := by
  sorry

end NUMINAMATH_CALUDE_initially_calculated_average_weight_l3316_331601


namespace NUMINAMATH_CALUDE_vector_operation_l3316_331670

theorem vector_operation (a b : ℝ × ℝ) :
  a = (2, 4) → b = (-1, 1) → 2 • a - b = (5, 7) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3316_331670


namespace NUMINAMATH_CALUDE_basketball_team_selection_count_l3316_331630

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 7 starters from a team of 16 players,
    including a set of 4 quadruplets, where exactly 3 of the quadruplets
    must be in the starting lineup -/
def basketball_team_selection : ℕ :=
  let total_players : ℕ := 16
  let quadruplets : ℕ := 4
  let starters : ℕ := 7
  let quadruplets_in_lineup : ℕ := 3
  (choose quadruplets quadruplets_in_lineup) *
  (choose (total_players - quadruplets) (starters - quadruplets_in_lineup))

theorem basketball_team_selection_count :
  basketball_team_selection = 1980 := by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_count_l3316_331630


namespace NUMINAMATH_CALUDE_base3_to_base10_equiv_l3316_331642

/-- Converts a base 3 number to base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Number : List Nat := [1, 2, 2, 0, 1]

theorem base3_to_base10_equiv : base3ToBase10 base3Number = 106 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_equiv_l3316_331642


namespace NUMINAMATH_CALUDE_prob_exceeds_175_l3316_331696

/-- The probability that a randomly selected student's height is less than 160cm -/
def prob_less_than_160 : ℝ := 0.2

/-- The probability that a randomly selected student's height is between 160cm and 175cm -/
def prob_between_160_and_175 : ℝ := 0.5

/-- Theorem: Given the probabilities of a student's height being less than 160cm and between 160cm and 175cm,
    the probability of a student's height exceeding 175cm is 0.3 -/
theorem prob_exceeds_175 : 1 - (prob_less_than_160 + prob_between_160_and_175) = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_prob_exceeds_175_l3316_331696


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3316_331663

/-- Given that i is the imaginary unit and (1+ai)/i is a pure imaginary number, prove that a = 0 -/
theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →  -- i is the imaginary unit
  (↑1 + a * Complex.I) / Complex.I = b * Complex.I →  -- (1+ai)/i is a pure imaginary number
  a = 0 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3316_331663


namespace NUMINAMATH_CALUDE_cube_root_square_l3316_331619

theorem cube_root_square (x : ℝ) : (x + 5) ^ (1/3 : ℝ) = 3 → (x + 5)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_l3316_331619


namespace NUMINAMATH_CALUDE_nested_fourth_root_equation_l3316_331691

/-- The limit of the sequence defined by a_(n+1) = ∜(x - a_n) --/
noncomputable def a (x : ℝ) : ℝ := sorry

/-- The limit of the sequence defined by b_(n+1) = ∜(x * b_n) --/
noncomputable def b (x : ℝ) : ℝ := sorry

/-- y is the positive real root of y³ - y² + 1 = 0 --/
noncomputable def y : ℝ := sorry

theorem nested_fourth_root_equation (x : ℝ) (hx : x > 0) :
  a x = b x → x = y^3 := by sorry

end NUMINAMATH_CALUDE_nested_fourth_root_equation_l3316_331691


namespace NUMINAMATH_CALUDE_function_value_inequality_l3316_331689

theorem function_value_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = -f (-x))
  (h2 : ∀ x, 1 < x ∧ x < 2 → f x > 0) :
  f (-1.5) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_inequality_l3316_331689


namespace NUMINAMATH_CALUDE_trajectory_of_Q_l3316_331637

-- Define the line L that point P moves on
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 3 = 0}

-- Define the fixed point M
def M : ℝ × ℝ := (-1, 2)

-- Define the property that Q is on the extension of PM and |PM| = |MQ|
def Q_property (P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Q = (t • (P - M) + M)

-- State the theorem
theorem trajectory_of_Q :
  ∀ Q : ℝ × ℝ, (∃ P ∈ L, Q_property P Q) → 2 * Q.1 - Q.2 + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_Q_l3316_331637


namespace NUMINAMATH_CALUDE_total_cost_is_1646_2_l3316_331628

/-- Calculates the total cost of fruits given their quantities, prices, discounts, and taxes --/
def total_cost_of_fruits 
  (grapes_kg : ℝ) (grapes_price : ℝ) (grapes_discount : ℝ) (grapes_tax : ℝ)
  (mangoes_kg : ℝ) (mangoes_price : ℝ) (mangoes_tax : ℝ)
  (apples_kg : ℝ) (apples_price : ℝ) (apples_discount : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (oranges_tax : ℝ)
  (oranges_free_kg : ℝ) : ℝ :=
  let grapes_cost := grapes_kg * grapes_price * (1 - grapes_discount) * (1 + grapes_tax)
  let mangoes_cost := mangoes_kg * mangoes_price * (1 + mangoes_tax)
  let apples_cost := apples_kg * apples_price * (1 - apples_discount)
  let oranges_cost := (oranges_kg - oranges_free_kg) * oranges_price * (1 + oranges_tax)
  grapes_cost + mangoes_cost + apples_cost + oranges_cost

/-- Theorem stating that the total cost of fruits is 1646.2 given the specified conditions --/
theorem total_cost_is_1646_2 :
  total_cost_of_fruits 
    8 70 0.1 0.05
    9 50 0.08
    5 100 0.15
    6 40 0.03 1 = 1646.2 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_1646_2_l3316_331628


namespace NUMINAMATH_CALUDE_power_zero_eq_one_iff_nonzero_l3316_331616

theorem power_zero_eq_one_iff_nonzero (a : ℝ) : a ^ 0 = 1 ↔ a ≠ 0 := by sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_iff_nonzero_l3316_331616


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3316_331676

/-- The complex number z = (2 + 3i) / (1 + 2i) is located in the fourth quadrant of the complex plane. -/
theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 + 3*I) / (1 + 2*I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3316_331676


namespace NUMINAMATH_CALUDE_triangle_properties_l3316_331665

noncomputable section

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define the theorem
theorem triangle_properties
  (a b c : ℝ)
  (h_triangle : triangle a b c)
  (h_angle_A : Real.cos (π/4) = b^2 + c^2 - a^2 / (2*b*c))
  (h_sides : b^2 - a^2 = (1/2) * c^2)
  (h_area : (1/2) * a * b * Real.sin (π/4) = 3) :
  Real.tan (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = 2 ∧
  b = 3 ∧
  2 * π * ((a / (2 * Real.sin (π/4))) : ℝ) = Real.sqrt 10 * π :=
sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l3316_331665


namespace NUMINAMATH_CALUDE_ivan_speed_ratio_l3316_331698

/-- Represents the speed of a person or group -/
structure Speed :=
  (value : ℝ)

/-- Represents time in hours -/
def Time : Type := ℝ

theorem ivan_speed_ratio (group_speed : Speed) (ivan_speed : Speed) : 
  -- Ivan left 15 minutes (0.25 hours) after the group started
  -- Ivan took 2.5 hours to catch up with the group after retrieving the flashlight
  -- Speeds of the group and Ivan (when not with the group) are constant
  (0.25 : ℝ) * group_speed.value + 2.5 * group_speed.value = 
    2.5 * ivan_speed.value + 2 * (0.25 * group_speed.value) →
  -- The ratio of Ivan's speed to the group's speed is 1.2
  ivan_speed.value / group_speed.value = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ivan_speed_ratio_l3316_331698


namespace NUMINAMATH_CALUDE_intersection_M_N_l3316_331645

def M : Set ℝ := {x | x^2 > 1}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3316_331645


namespace NUMINAMATH_CALUDE_evaluate_expression_l3316_331623

theorem evaluate_expression : (120 : ℚ) / 6 * 2 / 3 = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3316_331623


namespace NUMINAMATH_CALUDE_unique_products_count_l3316_331648

def set_a : Finset ℕ := {2, 3, 5, 7, 11}
def set_b : Finset ℕ := {2, 4, 6, 19}

theorem unique_products_count : 
  Finset.card ((set_a.product set_b).image (λ (x : ℕ × ℕ) => x.1 * x.2)) = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_products_count_l3316_331648


namespace NUMINAMATH_CALUDE_catherine_friends_count_l3316_331647

def total_bottle_caps : ℕ := 18
def caps_per_friend : ℕ := 3

theorem catherine_friends_count : 
  total_bottle_caps / caps_per_friend = 6 := by sorry

end NUMINAMATH_CALUDE_catherine_friends_count_l3316_331647


namespace NUMINAMATH_CALUDE_car_selling_problem_l3316_331675

/-- Calculates the net amount Chris receives from each buyer's offer --/
def net_amount (asking_price : ℝ) (inspection_cost : ℝ) (headlight_cost : ℝ) 
  (tire_cost : ℝ) (battery_cost : ℝ) (discount_rate : ℝ) (paint_job_rate : ℝ) : ℝ × ℝ × ℝ :=
  let first_offer := asking_price - inspection_cost
  let second_offer := asking_price - (headlight_cost + tire_cost + battery_cost)
  let discounted_price := asking_price * (1 - discount_rate)
  let third_offer := discounted_price - (discounted_price * paint_job_rate)
  (first_offer, second_offer, third_offer)

/-- Theorem statement for the car selling problem --/
theorem car_selling_problem (asking_price : ℝ) (inspection_rate : ℝ) (headlight_cost : ℝ) 
  (tire_rate : ℝ) (battery_rate : ℝ) (discount_rate : ℝ) (paint_job_rate : ℝ) :
  asking_price = 5200 ∧
  inspection_rate = 1/10 ∧
  headlight_cost = 80 ∧
  tire_rate = 3 ∧
  battery_rate = 2 ∧
  discount_rate = 15/100 ∧
  paint_job_rate = 1/5 →
  let (first, second, third) := net_amount asking_price (asking_price * inspection_rate) 
    headlight_cost (headlight_cost * tire_rate) (headlight_cost * tire_rate * battery_rate) 
    discount_rate paint_job_rate
  max first (max second third) - min first (min second third) = 1144 := by
  sorry


end NUMINAMATH_CALUDE_car_selling_problem_l3316_331675


namespace NUMINAMATH_CALUDE_circle_properties_l3316_331602

/-- Proves properties of a circle with circumference 24 cm -/
theorem circle_properties :
  ∀ (r : ℝ), 2 * π * r = 24 →
  (2 * r = 24 / π ∧ π * r^2 = 144 / π) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3316_331602


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3316_331632

theorem necessary_not_sufficient (a b : ℝ) :
  (∀ x y : ℝ, x > y + 1 → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(x > y + 1)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3316_331632


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3316_331636

theorem fraction_equivalence (a b k : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hk : k ≠ 0) :
  (k * a) / (k * b) = a / b :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3316_331636


namespace NUMINAMATH_CALUDE_visibility_time_correct_l3316_331693

/-- Represents a person walking along a straight path -/
structure Walker where
  speed : ℝ
  initial_position : ℝ × ℝ

/-- Represents the circular building -/
structure Building where
  center : ℝ × ℝ
  radius : ℝ

/-- The scenario of Jenny and Kenny walking -/
def walking_scenario : Building × Walker × Walker := 
  let building := { center := (0, 0), radius := 50 }
  let jenny := { speed := 2, initial_position := (-150, 100) }
  let kenny := { speed := 4, initial_position := (-150, -100) }
  (building, jenny, kenny)

/-- The time when Jenny and Kenny can see each other again -/
noncomputable def visibility_time (scenario : Building × Walker × Walker) : ℝ := 
  200  -- This is the value we want to prove

/-- The theorem stating that the visibility time is correct -/
theorem visibility_time_correct :
  let (building, jenny, kenny) := walking_scenario
  let t := visibility_time walking_scenario
  
  -- At time t, the line connecting Jenny and Kenny is tangent to the building
  ∃ (x y : ℝ),
    (x^2 + y^2 = building.radius^2) ∧
    ((jenny.initial_position.1 + jenny.speed * t - x) * (kenny.initial_position.2 - y) =
     (kenny.initial_position.1 + kenny.speed * t - x) * (jenny.initial_position.2 - y)) ∧
    (x * (jenny.initial_position.2 - y) + y * (x - jenny.initial_position.1 - jenny.speed * t) = 0) :=
by sorry

end NUMINAMATH_CALUDE_visibility_time_correct_l3316_331693


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l3316_331673

theorem sqrt_eight_div_sqrt_two_equals_two :
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l3316_331673


namespace NUMINAMATH_CALUDE_westward_movement_l3316_331615

-- Define a type for directions
inductive Direction
  | East
  | West

-- Define a function to represent movement
def represent_movement (dist : ℝ) (dir : Direction) : ℝ :=
  match dir with
  | Direction.East => dist
  | Direction.West => -dist

-- State the theorem
theorem westward_movement :
  (Direction.East ≠ Direction.West) →  -- East and west are opposite
  (represent_movement 2 Direction.East = 2) →  -- +2 meters represents 2 meters eastward
  (represent_movement 7 Direction.West = -7)  -- 7 meters westward is represented by -7 meters
:= by sorry

end NUMINAMATH_CALUDE_westward_movement_l3316_331615


namespace NUMINAMATH_CALUDE_sum_of_digits_mod_9_triple_sum_of_digits_4444_power_l3316_331694

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Property: sum of digits is congruent to the number modulo 9 -/
theorem sum_of_digits_mod_9 (n : ℕ) : sum_of_digits n % 9 = n % 9 := sorry

/-- Main theorem -/
theorem triple_sum_of_digits_4444_power :
  let N := 4444^4444
  let f := sum_of_digits
  f (f (f N)) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_mod_9_triple_sum_of_digits_4444_power_l3316_331694


namespace NUMINAMATH_CALUDE_exam_pass_count_l3316_331668

theorem exam_pass_count (total_students : ℕ) (total_average : ℚ) 
  (pass_average : ℚ) (fail_average : ℚ) (weight_ratio : ℚ × ℚ) :
  total_students = 150 ∧ 
  total_average = 40 ∧ 
  pass_average = 45 ∧ 
  fail_average = 20 ∧ 
  weight_ratio = (3, 1) →
  ∃ (pass_count : ℕ), pass_count = 85 ∧ 
    (pass_count : ℚ) * weight_ratio.1 * pass_average + 
    (total_students - pass_count : ℚ) * weight_ratio.2 * fail_average = 
    total_average * (pass_count * weight_ratio.1 + (total_students - pass_count) * weight_ratio.2) :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l3316_331668


namespace NUMINAMATH_CALUDE_shelter_dogs_l3316_331646

theorem shelter_dogs (cat_count : ℕ) (cat_ratio : ℕ) (dog_ratio : ℕ) : 
  cat_count = 21 → cat_ratio = 7 → dog_ratio = 5 → 
  (cat_count * dog_ratio) / cat_ratio = 15 :=
by sorry

end NUMINAMATH_CALUDE_shelter_dogs_l3316_331646


namespace NUMINAMATH_CALUDE_no_unique_solution_l3316_331680

theorem no_unique_solution (a : ℝ) : ¬ ∃! p : ℝ × ℝ, 
  p.1^2 + p.2^2 = 2 ∧ |p.2| - p.1 = a :=
by
  sorry

end NUMINAMATH_CALUDE_no_unique_solution_l3316_331680


namespace NUMINAMATH_CALUDE_sum_of_y_coords_on_y_axis_l3316_331613

-- Define the circle
def circle_center : ℝ × ℝ := (-6, 2)
def circle_radius : ℝ := 10

-- Define a point on the circle
def point_on_circle (p : ℝ × ℝ) : Prop :=
  (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = circle_radius^2

-- Define a point on the y-axis
def point_on_y_axis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

-- Theorem statement
theorem sum_of_y_coords_on_y_axis :
  ∃ (p1 p2 : ℝ × ℝ),
    point_on_circle p1 ∧ point_on_y_axis p1 ∧
    point_on_circle p2 ∧ point_on_y_axis p2 ∧
    p1 ≠ p2 ∧
    p1.2 + p2.2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_coords_on_y_axis_l3316_331613


namespace NUMINAMATH_CALUDE_max_value_theorem_l3316_331685

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 2 * x^2 - x*y + y^2 = 15) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 - a*b + b^2 = 15 → 
    2 * x^2 + x*y + y^2 ≥ 2 * a^2 + a*b + b^2) → 
  2 * x^2 + x*y + y^2 = (75 + 60 * Real.sqrt 2) / 7 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3316_331685


namespace NUMINAMATH_CALUDE_multiplication_problem_solution_l3316_331624

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the multiplication problem -/
structure MultiplicationProblem where
  A : Digit
  B : Digit
  C : Digit
  D : Digit
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  equation : A.val * 100 + B.val * 10 + C.val = C.val * 100 + C.val * 10 + A.val

theorem multiplication_problem_solution (p : MultiplicationProblem) : p.A.val + p.C.val = 10 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problem_solution_l3316_331624


namespace NUMINAMATH_CALUDE_equation_is_linear_l3316_331660

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x y, f x y = a * x + b * y + c

/-- The equation x - 3y = -15 -/
def equation (x y : ℝ) : ℝ := x - 3 * y + 15

theorem equation_is_linear :
  is_linear_equation_in_two_variables equation := by
sorry

end NUMINAMATH_CALUDE_equation_is_linear_l3316_331660


namespace NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l3316_331692

theorem fair_coin_three_heads_probability :
  let n : ℕ := 7  -- number of coin tosses
  let k : ℕ := 3  -- number of heads we're looking for
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := Nat.choose n k  -- number of ways to choose k heads from n tosses
  (favorable_outcomes : ℚ) / total_outcomes = 35 / 128 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_three_heads_probability_l3316_331692


namespace NUMINAMATH_CALUDE_concentric_circles_radii_l3316_331667

theorem concentric_circles_radii 
  (chord_length : ℝ) 
  (ring_width : ℝ) 
  (h_chord : chord_length = 32) 
  (h_width : ring_width = 8) :
  ∃ (r R : ℝ), 
    r > 0 ∧ 
    R > r ∧
    R = r + ring_width ∧
    (r + ring_width)^2 = r^2 + (chord_length/2)^2 ∧
    r = 12 ∧ 
    R = 20 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_l3316_331667


namespace NUMINAMATH_CALUDE_unique_x_floor_product_l3316_331657

theorem unique_x_floor_product : ∃! x : ℝ, x > 0 ∧ x * ↑(⌊x⌋) = 80 ∧ x = 80 / 9 := by sorry

end NUMINAMATH_CALUDE_unique_x_floor_product_l3316_331657


namespace NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l3316_331652

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

-- Theorem for the tangent line equation
theorem tangent_line_at_2 :
  ∃ (m c : ℝ), ∀ x y, y = m * (x - 2) + f 2 ↔ 12 * x - y - 17 = 0 := by sorry

-- Theorem for intervals of monotonicity
theorem monotonicity_intervals :
  (∀ x, x < 0 → (f' x > 0)) ∧
  (∀ x, 0 < x ∧ x < 1 → (f' x < 0)) ∧
  (∀ x, x > 1 → (f' x > 0)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l3316_331652


namespace NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_six_l3316_331690

theorem fourth_root_over_sixth_root_of_six (x : ℝ) (h : x = 6) :
  (x^(1/4)) / (x^(1/6)) = x^(1/12) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_sixth_root_of_six_l3316_331690


namespace NUMINAMATH_CALUDE_roots_of_equation_l3316_331659

def equation (x : ℝ) : ℝ := (x^2 - 5*x + 6) * (x - 1) * (x + 3)

theorem roots_of_equation :
  {x : ℝ | equation x = 0} = {-3, 1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3316_331659


namespace NUMINAMATH_CALUDE_greatest_real_part_of_sixth_power_l3316_331681

theorem greatest_real_part_of_sixth_power : 
  let z₁ : ℂ := -3
  let z₂ : ℂ := -Real.sqrt 6 + Complex.I
  let z₃ : ℂ := -Real.sqrt 3 + (Real.sqrt 3 : ℝ) * Complex.I
  let z₄ : ℂ := -1 + (Real.sqrt 6 : ℝ) * Complex.I
  let z₅ : ℂ := 2 * Complex.I
  Complex.re (z₁^6) > Complex.re (z₂^6) ∧
  Complex.re (z₁^6) > Complex.re (z₃^6) ∧
  Complex.re (z₁^6) > Complex.re (z₄^6) ∧
  Complex.re (z₁^6) > Complex.re (z₅^6) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_real_part_of_sixth_power_l3316_331681


namespace NUMINAMATH_CALUDE_parabola_b_value_l3316_331621

-- Define the parabola equation
def parabola (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem parabola_b_value :
  ∀ a b : ℝ,
  (parabola a b 2 = 10) →
  (parabola a b (-2) = 6) →
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3316_331621


namespace NUMINAMATH_CALUDE_function_composition_l3316_331640

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem function_composition :
  ∀ x : ℝ, f (g x) = 6 * x - 7 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l3316_331640


namespace NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_5_proof_l3316_331611

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def largest_odd_digit_multiple_of_5 : ℕ :=
  9955

theorem largest_odd_digit_multiple_of_5_proof :
  (largest_odd_digit_multiple_of_5 < 10000) ∧
  (largest_odd_digit_multiple_of_5 % 5 = 0) ∧
  has_only_odd_digits largest_odd_digit_multiple_of_5 ∧
  ∀ n : ℕ, n < 10000 → n % 5 = 0 → has_only_odd_digits n →
    n ≤ largest_odd_digit_multiple_of_5 :=
by sorry

end NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_5_proof_l3316_331611


namespace NUMINAMATH_CALUDE_correct_mark_is_45_l3316_331684

/-- The correct mark for a pupil, given the conditions of the problem. -/
def correct_mark (wrong_mark : ℕ) (num_pupils : ℕ) : ℕ :=
  wrong_mark - (num_pupils / 2)

/-- Theorem stating that the correct mark is 45 under the given conditions. -/
theorem correct_mark_is_45 :
  let wrong_mark := 73
  let num_pupils := 56
  correct_mark wrong_mark num_pupils = 45 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_is_45_l3316_331684


namespace NUMINAMATH_CALUDE_problem_statement_l3316_331661

theorem problem_statement (x y z : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_diff_xy : x ≠ y) (h_diff_xz : x ≠ z) (h_diff_yz : y ≠ z)
  (h_eq1 : (y + 1) / (x - z) = (x + y) / (z + 1))
  (h_eq2 : (y + 1) / (x - z) = x / (y + 1)) :
  x / (y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3316_331661


namespace NUMINAMATH_CALUDE_cos_A_in_special_triangle_l3316_331604

/-- 
Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if 2S = a² - (b-c)² where S is the area of the triangle, then cos A = 3/5.
-/
theorem cos_A_in_special_triangle (a b c : ℝ) (A : Real) :
  0 < A → A < Real.pi / 2 →  -- A is acute
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  2 * (1/2 * b * c * Real.sin A) = a^2 - (b - c)^2 →  -- area condition
  Real.cos A = 3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_A_in_special_triangle_l3316_331604


namespace NUMINAMATH_CALUDE_optimal_racket_purchase_l3316_331617

/-- The optimal purchasing plan for badminton rackets -/
theorem optimal_racket_purchase 
  (total_cost : ℕ) 
  (num_pairs : ℕ) 
  (price_diff : ℕ) 
  (discount_a : ℚ) 
  (discount_b : ℕ) 
  (max_cost : ℕ) 
  (min_a : ℕ) :
  total_cost = num_pairs * (price_a + price_b) ∧
  price_b = price_a - price_diff ∧
  new_price_a = price_a * discount_a ∧
  new_price_b = price_b - discount_b ∧
  (∀ m : ℕ, m ≥ min_a → m ≤ 50 → 
    new_price_a * m + new_price_b * (50 - m) ≤ max_cost) →
  optimal_m = 38 ∧ 
  optimal_cost = new_price_a * optimal_m + new_price_b * (50 - optimal_m) ∧
  (∀ m : ℕ, m ≥ min_a → m ≤ 50 → 
    new_price_a * m + new_price_b * (50 - m) ≥ optimal_cost) :=
by
  sorry

#check optimal_racket_purchase 1300 20 15 (4/5) 4 1500 38

end NUMINAMATH_CALUDE_optimal_racket_purchase_l3316_331617


namespace NUMINAMATH_CALUDE_complex_number_properties_l3316_331655

variable (a : ℝ)
variable (b : ℝ)
def z : ℂ := a + Complex.I

theorem complex_number_properties :
  (∀ z, Complex.abs z = 1 → a = 0) ∧
  (∀ z, (z / (1 + Complex.I)).im = 0 → a = 1) ∧
  (∀ z b, z^2 + b*z + 2 = 0 → ((a = 1 ∧ b = -2) ∨ (a = -1 ∧ b = 2))) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3316_331655


namespace NUMINAMATH_CALUDE_square_perimeter_l3316_331656

theorem square_perimeter (A : ℝ) (h : A = 625) :
  2 * (4 * Real.sqrt A) = 200 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l3316_331656


namespace NUMINAMATH_CALUDE_f_symmetry_l3316_331658

/-- A function f(x) defined as ax^5 - bx^3 + cx + 1 -/
def f (a b c x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x + 1

/-- Theorem: If f(-2) = -1, then f(2) = 3 -/
theorem f_symmetry (a b c : ℝ) : f a b c (-2) = -1 → f a b c 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3316_331658


namespace NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_l3316_331677

/-- A function f: ℝ → ℝ is symmetric about the line x = c if f(c + t) = f(c - t) for all t ∈ ℝ -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ t, f (c + t) = f (c - t)

/-- The main theorem: If sin x + a cos x is symmetric about x = π/4, then a = 1 -/
theorem symmetry_implies_a_equals_one (a : ℝ) :
  SymmetricAbout (fun x ↦ Real.sin x + a * Real.cos x) (π/4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_equals_one_l3316_331677


namespace NUMINAMATH_CALUDE_triangle_properties_l3316_331672

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.a * t.b = 6) :
  t.C = π / 3 ∧ t.a + t.b + t.c = Real.sqrt 37 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3316_331672


namespace NUMINAMATH_CALUDE_scientific_notation_of_634000000_l3316_331610

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_634000000 :
  toScientificNotation 634000000 = ScientificNotation.mk 6.34 8 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_634000000_l3316_331610


namespace NUMINAMATH_CALUDE_paint_calculation_l3316_331674

theorem paint_calculation (total_paint : ℚ) : 
  (1/4 * total_paint + 1/3 * (3/4 * total_paint) = 180) → total_paint = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3316_331674


namespace NUMINAMATH_CALUDE_max_product_theorem_l3316_331626

theorem max_product_theorem (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 ≤ x₁ ∧ x₁ ≤ 12)
  (h₂ : 0 ≤ x₂ ∧ x₂ ≤ 12)
  (h₃ : 0 ≤ x₃ ∧ x₃ ≤ 12)
  (h₄ : x₁ * x₂ * x₃ = ((12 - x₁) * (12 - x₂) * (12 - x₃))^2) :
  x₁ * x₂ * x₃ ≤ 729 :=
by sorry

end NUMINAMATH_CALUDE_max_product_theorem_l3316_331626


namespace NUMINAMATH_CALUDE_calculation_proof_l3316_331654

theorem calculation_proof : 3.6 * 0.25 + 1.5 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3316_331654


namespace NUMINAMATH_CALUDE_similar_rectangles_width_l3316_331603

theorem similar_rectangles_width (area_ratio : ℚ) (small_width : ℚ) (large_width : ℚ) : 
  area_ratio = 1 / 9 →
  small_width = 2 →
  (large_width / small_width) ^ 2 = 1 / area_ratio →
  large_width = 6 := by
sorry

end NUMINAMATH_CALUDE_similar_rectangles_width_l3316_331603


namespace NUMINAMATH_CALUDE_least_n_squared_minus_n_divisibility_l3316_331639

theorem least_n_squared_minus_n_divisibility : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k = 0) ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k ≠ 0) ∧
    (∀ (m : ℕ), m > 0 ∧ m < n → 
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k = 0) ∨
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k ≠ 0))) ∧
  (∀ (n : ℕ), n > 0 ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k = 0) ∧ 
    (∃ (k : ℕ), 1 ≤ k ∧ k ≤ n ∧ (n^2 - n) % k ≠ 0) ∧
    (∀ (m : ℕ), m > 0 ∧ m < n → 
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k = 0) ∨
      (∀ (k : ℕ), 1 ≤ k ∧ k ≤ m → (m^2 - m) % k ≠ 0)) →
    n ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_least_n_squared_minus_n_divisibility_l3316_331639


namespace NUMINAMATH_CALUDE_g_composition_fixed_points_l3316_331612

def g (x : ℝ) : ℝ := x^2 - 4*x

theorem g_composition_fixed_points :
  ∀ x : ℝ, g (g x) = g x ↔ x = -1 ∨ x = 0 ∨ x = 4 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_fixed_points_l3316_331612


namespace NUMINAMATH_CALUDE_correct_average_l3316_331651

theorem correct_average (numbers : Finset ℕ) (incorrect_sum : ℕ) (incorrect_number correct_number : ℕ) :
  numbers.card = 10 →
  incorrect_sum / numbers.card = 19 →
  incorrect_number = 26 →
  correct_number = 76 →
  (incorrect_sum - incorrect_number + correct_number) / numbers.card = 24 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3316_331651


namespace NUMINAMATH_CALUDE_water_usage_difference_l3316_331682

theorem water_usage_difference (total_water plants_water : ℕ) : 
  total_water = 65 →
  plants_water < 14 →
  24 * 2 = 65 - 14 - plants_water →
  7 - plants_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_water_usage_difference_l3316_331682


namespace NUMINAMATH_CALUDE_koala_weight_in_grams_l3316_331697

-- Define the conversion rate from kg to g
def kg_to_g : ℕ → ℕ := (· * 1000)

-- Define the weight of the baby koala
def koala_weight_kg : ℕ := 2
def koala_weight_extra_g : ℕ := 460

-- Theorem: The total weight of the baby koala in grams is 2460
theorem koala_weight_in_grams : 
  kg_to_g koala_weight_kg + koala_weight_extra_g = 2460 := by
  sorry

end NUMINAMATH_CALUDE_koala_weight_in_grams_l3316_331697


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l3316_331622

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_integer_with_gcf_five : 
  (∀ m, is_valid m → m ≤ 145) ∧ is_valid 145 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l3316_331622


namespace NUMINAMATH_CALUDE_spending_ratio_l3316_331635

/-- Represents the spending of Lisa and Carly -/
structure Spending where
  lisa_tshirt : ℝ
  lisa_jeans : ℝ
  lisa_coat : ℝ
  carly_tshirt : ℝ
  carly_jeans : ℝ
  carly_coat : ℝ

/-- The theorem representing the problem -/
theorem spending_ratio (s : Spending) : 
  s.lisa_tshirt = 40 →
  s.lisa_jeans = s.lisa_tshirt / 2 →
  s.carly_tshirt = s.lisa_tshirt / 4 →
  s.carly_jeans = 3 * s.lisa_jeans →
  s.carly_coat = s.lisa_coat / 4 →
  s.lisa_tshirt + s.lisa_jeans + s.lisa_coat + 
  s.carly_tshirt + s.carly_jeans + s.carly_coat = 230 →
  s.lisa_coat / s.lisa_tshirt = 2 := by
  sorry

end NUMINAMATH_CALUDE_spending_ratio_l3316_331635


namespace NUMINAMATH_CALUDE_student_walking_speed_l3316_331695

/-- 
Given two students walking towards each other:
- They start 350 meters apart
- They walk for 100 seconds until they meet
- The first student walks at 1.6 m/s
Prove that the second student's speed is 1.9 m/s
-/
theorem student_walking_speed 
  (initial_distance : ℝ) 
  (time : ℝ) 
  (speed1 : ℝ) 
  (h1 : initial_distance = 350)
  (h2 : time = 100)
  (h3 : speed1 = 1.6) :
  ∃ speed2 : ℝ, 
    speed2 = 1.9 ∧ 
    speed1 * time + speed2 * time = initial_distance := by
  sorry

end NUMINAMATH_CALUDE_student_walking_speed_l3316_331695


namespace NUMINAMATH_CALUDE_ab_value_l3316_331669

theorem ab_value (a b : ℝ) (h : |a + 3| + (b - 2)^2 = 0) : a^b = 9 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l3316_331669


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l3316_331608

-- Define the quadratic equation
def quadratic (x c : ℝ) : ℝ := x^2 + x - c

-- Theorem statement
theorem no_real_roots_condition (c : ℝ) :
  (∀ x : ℝ, quadratic x c ≠ 0) ↔ c < -1/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l3316_331608


namespace NUMINAMATH_CALUDE_gunther_cleaning_time_l3316_331620

/-- Gunther's apartment cleaning problem -/
theorem gunther_cleaning_time (free_time : ℕ) (vacuum_time : ℕ) (dust_time : ℕ) (mop_time : ℕ) 
  (num_cats : ℕ) (remaining_time : ℕ) : 
  free_time = 3 * 60 → 
  vacuum_time = 45 →
  dust_time = 60 →
  mop_time = 30 →
  num_cats = 3 →
  remaining_time = 30 →
  (free_time - remaining_time - vacuum_time - dust_time - mop_time) / num_cats = 5 := by
  sorry

end NUMINAMATH_CALUDE_gunther_cleaning_time_l3316_331620


namespace NUMINAMATH_CALUDE_maria_spent_60_dollars_l3316_331607

/-- The amount Maria spent on flowers -/
def total_spent (price_per_flower : ℕ) (roses : ℕ) (daisies : ℕ) : ℕ :=
  price_per_flower * (roses + daisies)

/-- Proof that Maria spent 60 dollars on flowers -/
theorem maria_spent_60_dollars : total_spent 6 7 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_maria_spent_60_dollars_l3316_331607


namespace NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l3316_331638

theorem unique_solution_for_rational_equation :
  let k : ℚ := -3/4
  let f (x : ℚ) := (x + 3)/(k*x - 2) - x
  ∃! x, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_rational_equation_l3316_331638


namespace NUMINAMATH_CALUDE_distribute_unique_items_l3316_331671

theorem distribute_unique_items 
  (num_items : ℕ) 
  (num_recipients : ℕ) 
  (h1 : num_items = 6) 
  (h2 : num_recipients = 8) :
  (num_recipients ^ num_items : ℕ) = 262144 := by
  sorry

end NUMINAMATH_CALUDE_distribute_unique_items_l3316_331671


namespace NUMINAMATH_CALUDE_linear_function_relationship_l3316_331633

/-- A linear function f(x) = 2x + 1 -/
def f (x : ℝ) : ℝ := 2 * x + 1

theorem linear_function_relationship (y₁ y₂ : ℝ) 
  (h1 : f (-3) = y₁) 
  (h2 : f 4 = y₂) : 
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_relationship_l3316_331633


namespace NUMINAMATH_CALUDE_product_with_9999_l3316_331606

theorem product_with_9999 : ∃ x : ℕ, x * 9999 = 4691110842 ∧ x = 469211 := by
  sorry

end NUMINAMATH_CALUDE_product_with_9999_l3316_331606


namespace NUMINAMATH_CALUDE_lemon_heads_boxes_l3316_331627

/-- The number of Lemon Heads in each package -/
def package_size : ℕ := 6

/-- The total number of Lemon Heads Louis ate -/
def total_eaten : ℕ := 54

/-- The number of whole boxes Louis ate -/
def whole_boxes : ℕ := total_eaten / package_size

theorem lemon_heads_boxes : whole_boxes = 9 := by
  sorry

end NUMINAMATH_CALUDE_lemon_heads_boxes_l3316_331627


namespace NUMINAMATH_CALUDE_sin_equality_integer_solutions_l3316_331649

theorem sin_equality_integer_solutions (m : ℤ) :
  -180 ≤ m ∧ m ≤ 180 ∧ Real.sin (m * π / 180) = Real.sin (750 * π / 180) →
  m = 30 ∨ m = 150 := by
sorry

end NUMINAMATH_CALUDE_sin_equality_integer_solutions_l3316_331649


namespace NUMINAMATH_CALUDE_cost_per_serving_is_50_cents_l3316_331609

/-- Calculates the cost per serving of mixed nuts in cents -/
def cost_per_serving (bag_price : ℚ) (bag_weight : ℚ) (coupon : ℚ) (serving_size : ℚ) : ℚ :=
  ((bag_price - coupon) / bag_weight) * serving_size * 100

/-- Proves that the cost per serving of mixed nuts is 50 cents -/
theorem cost_per_serving_is_50_cents :
  cost_per_serving 25 40 5 1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_serving_is_50_cents_l3316_331609


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3316_331687

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 5) * (x^2 + 6*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3316_331687


namespace NUMINAMATH_CALUDE_exam_score_problem_l3316_331641

theorem exam_score_problem (scores : List ℝ) (avg : ℝ) : 
  scores.length = 4 →
  scores = [80, 90, 100, 110] →
  avg = 96 →
  (scores.sum + (5 * avg - scores.sum)) / 5 = avg →
  5 * avg - scores.sum = 100 := by
sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3316_331641


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3316_331650

open Real

theorem max_value_trig_expression :
  ∃ (M : ℝ), M = Real.sqrt 10 + 3 ∧
  (∀ x : ℝ, cos x + 3 * sin x + tan x ≤ M) ∧
  (∃ x : ℝ, cos x + 3 * sin x + tan x = M) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3316_331650


namespace NUMINAMATH_CALUDE_bake_sale_group_composition_l3316_331600

theorem bake_sale_group_composition (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls = total / 2) →  -- Initially, 50% of the group are girls
  (initial_girls - 3 = (total * 2) / 5) →  -- After changes, 40% are girls
  (initial_girls = 15) :=
by
  sorry

#check bake_sale_group_composition

end NUMINAMATH_CALUDE_bake_sale_group_composition_l3316_331600


namespace NUMINAMATH_CALUDE_total_earnings_l3316_331683

/-- Given that 5 men are equal to W women, W women are equal to B boys,
    and men's wages are 10, prove that the total amount earned by all groups is 150. -/
theorem total_earnings (W B : ℕ) (men_wage : ℕ) : 
  (5 = W) → (W = B) → (men_wage = 10) → 
  (5 * men_wage + W * men_wage + B * men_wage = 150) :=
by sorry

end NUMINAMATH_CALUDE_total_earnings_l3316_331683


namespace NUMINAMATH_CALUDE_inequality_proof_l3316_331644

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3316_331644


namespace NUMINAMATH_CALUDE_parallelogram_area_l3316_331678

-- Define the parallelogram ABCD
variable (A B C D : Point)

-- Define point E as midpoint of BC
variable (E : Point)

-- Define point F on AD
variable (F : Point)

-- Define the area function
variable (area : Set Point → ℝ)

-- Define the parallelogram ABCD
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define E as midpoint of BC
def is_midpoint (E B C : Point) : Prop := sorry

-- Define the condition DF = 2FC
def segment_ratio (D F C : Point) : Prop := sorry

-- Define triangles
def triangle (P Q R : Point) : Set Point := sorry

-- Define parallelogram
def parallelogram (A B C D : Point) : Set Point := sorry

-- Theorem statement
theorem parallelogram_area 
  (h1 : is_parallelogram A B C D)
  (h2 : is_midpoint E B C)
  (h3 : segment_ratio D F C)
  (h4 : area (triangle A F C) + area (triangle A B E) = 10) :
  area (parallelogram A B C D) = 24 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3316_331678
