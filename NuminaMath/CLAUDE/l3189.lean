import Mathlib

namespace NUMINAMATH_CALUDE_sharon_angela_cutlery_ratio_l3189_318962

/-- Prove that the ratio of Sharon's cutlery to Angela's cutlery is 2:1 -/
theorem sharon_angela_cutlery_ratio :
  let angela_pots : ℕ := 20
  let angela_plates : ℕ := 3 * angela_pots + 6
  let angela_cutlery : ℕ := angela_plates / 2
  let sharon_pots : ℕ := angela_pots / 2
  let sharon_plates : ℕ := 3 * angela_plates - 20
  let sharon_total : ℕ := 254
  let sharon_cutlery : ℕ := sharon_total - (sharon_pots + sharon_plates)
  (sharon_cutlery : ℚ) / (angela_cutlery : ℚ) = 2
  := by sorry

end NUMINAMATH_CALUDE_sharon_angela_cutlery_ratio_l3189_318962


namespace NUMINAMATH_CALUDE_equal_gum_distribution_l3189_318952

/-- Proves that when three people share 99 pieces of gum equally, each person receives 33 pieces. -/
theorem equal_gum_distribution (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : aubrey_gum = 0)
  (h4 : (john_gum + cole_gum + aubrey_gum) % 3 = 0) :
  (john_gum + cole_gum + aubrey_gum) / 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_equal_gum_distribution_l3189_318952


namespace NUMINAMATH_CALUDE_f_minus_five_l3189_318988

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_minus_five (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 6)
  (h_odd : is_odd f)
  (h_f_minus_one : f (-1) = 1) :
  f (-5) = -1 := by
sorry

end NUMINAMATH_CALUDE_f_minus_five_l3189_318988


namespace NUMINAMATH_CALUDE_max_surface_area_parallelepiped_l3189_318984

/-- The maximum surface area of a rectangular parallelepiped with diagonal length 3 is 18 -/
theorem max_surface_area_parallelepiped (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 + c^2 = 9 →
  2 * (a * b + b * c + c * a) ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_surface_area_parallelepiped_l3189_318984


namespace NUMINAMATH_CALUDE_sheet_area_difference_l3189_318933

/-- The combined area (front and back) of a rectangular sheet -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem sheet_area_difference :
  areaDifference 11 19 9.5 11 = 209 := by
  sorry

end NUMINAMATH_CALUDE_sheet_area_difference_l3189_318933


namespace NUMINAMATH_CALUDE_smallest_integer_multiple_conditions_l3189_318935

theorem smallest_integer_multiple_conditions :
  ∃ n : ℕ, n > 0 ∧
  (∃ k : ℤ, n = 5 * k + 3) ∧
  (∃ m : ℤ, n = 12 * m) ∧
  (∀ x : ℕ, x > 0 →
    (∃ k' : ℤ, x = 5 * k' + 3) →
    (∃ m' : ℤ, x = 12 * m') →
    n ≤ x) ∧
  n = 48 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_multiple_conditions_l3189_318935


namespace NUMINAMATH_CALUDE_complex_simplification_l3189_318971

theorem complex_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  7 * (2 - 2*i) + 4*i * (7 - 3*i) = 26 + 14*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l3189_318971


namespace NUMINAMATH_CALUDE_missing_number_proof_l3189_318974

theorem missing_number_proof (x y : ℝ) : 
  (12 + x + 42 + y + 104) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 →
  y = 78 := by
sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3189_318974


namespace NUMINAMATH_CALUDE_ratio_of_balls_l3189_318959

def red_balls : ℕ := 16
def white_balls : ℕ := 20

theorem ratio_of_balls : 
  (red_balls : ℚ) / white_balls = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_balls_l3189_318959


namespace NUMINAMATH_CALUDE_abs_positive_for_nonzero_l3189_318932

theorem abs_positive_for_nonzero (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_positive_for_nonzero_l3189_318932


namespace NUMINAMATH_CALUDE_three_digit_number_ending_in_five_divisible_by_five_l3189_318991

theorem three_digit_number_ending_in_five_divisible_by_five (N : ℕ) :
  100 ≤ N ∧ N < 1000 ∧ N % 10 = 5 → N % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_ending_in_five_divisible_by_five_l3189_318991


namespace NUMINAMATH_CALUDE_modulus_of_z_l3189_318951

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3189_318951


namespace NUMINAMATH_CALUDE_player_one_wins_l3189_318927

/-- Represents a player in the stone game -/
inductive Player
| One
| Two

/-- Represents the state of the game -/
structure GameState where
  piles : List Nat
  currentPlayer : Player

/-- Represents a move in the game -/
structure Move where
  pileIndices : List Nat
  stonesRemoved : List Nat

/-- Defines a valid move for Player One -/
def isValidMovePlayerOne (m : Move) : Prop :=
  m.pileIndices.length = 1 ∧ 
  m.stonesRemoved.length = 1 ∧
  (m.stonesRemoved.head! = 1 ∨ m.stonesRemoved.head! = 2 ∨ m.stonesRemoved.head! = 3)

/-- Defines a valid move for Player Two -/
def isValidMovePlayerTwo (m : Move) : Prop :=
  m.pileIndices.length = m.stonesRemoved.length ∧
  m.pileIndices.length ≤ 3 ∧
  m.stonesRemoved.all (· = 1)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  state.piles.all (· = 0)

/-- Determines if a player has a winning strategy from a given state -/
def hasWinningStrategy (state : GameState) : Prop :=
  sorry

/-- The main theorem: Player One has a winning strategy in the initial game state -/
theorem player_one_wins :
  hasWinningStrategy (GameState.mk (List.replicate 11 10) Player.One) :=
  sorry

end NUMINAMATH_CALUDE_player_one_wins_l3189_318927


namespace NUMINAMATH_CALUDE_binomial_coefficient_inequality_l3189_318961

theorem binomial_coefficient_inequality
  (n k h : ℕ)
  (h1 : n ≥ k + h) :
  Nat.choose n (k + h) ≥ Nat.choose (n - k) h :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_inequality_l3189_318961


namespace NUMINAMATH_CALUDE_correct_average_weight_l3189_318990

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (actual_weight : ℝ) : 
  n = 20 → 
  initial_average = 58.4 → 
  misread_weight = 56 → 
  actual_weight = 68 → 
  (n * initial_average + actual_weight - misread_weight) / n = 59 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l3189_318990


namespace NUMINAMATH_CALUDE_equation_equivalence_l3189_318924

theorem equation_equivalence (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 7) : 
  1 / (x - 3) + 1 / (x - 5) + 1 / (x - 7) = 4 / (x - 4) ↔ 
  x^3 - 13*x^2 + 48*x - 64 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3189_318924


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l3189_318920

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l3189_318920


namespace NUMINAMATH_CALUDE_solution_l3189_318981

-- Define the equation
def equation (A B x : ℝ) : Prop :=
  A / (x + 3) + B / (x^2 - 9*x) = (x^2 - 3*x + 15) / (x^3 + x^2 - 27*x)

-- State the theorem
theorem solution (A B : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 9 → equation A B x) →
  (B : ℝ) / (A : ℝ) = 7.5 := by
sorry

end NUMINAMATH_CALUDE_solution_l3189_318981


namespace NUMINAMATH_CALUDE_parallel_lines_minimum_value_l3189_318902

theorem parallel_lines_minimum_value (m n : ℕ+) 
  (h_parallel : (2 : ℝ) / (n - 1 : ℝ) = (m : ℝ) / (n : ℝ)) : 
  (∀ k l : ℕ+, (2 : ℝ) / (l - 1 : ℝ) = (k : ℝ) / (l : ℝ) → 2 * m + n ≤ 2 * k + l) ∧ 
  (∃ k l : ℕ+, (2 : ℝ) / (l - 1 : ℝ) = (k : ℝ) / (l : ℝ) ∧ 2 * k + l = 9) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_minimum_value_l3189_318902


namespace NUMINAMATH_CALUDE_smallest_positive_n_squared_l3189_318985

-- Define the circles c1 and c2
def c1 (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 75 = 0
def c2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 175 = 0

-- Define a function to check if a point is on a line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the conditions for external and internal tangency
def externally_tangent (x y r : ℝ) : Prop := (x - 4)^2 + (y - 10)^2 = (r + 7)^2
def internally_tangent (x y r : ℝ) : Prop := (x + 4)^2 + (y - 10)^2 = (11 - r)^2

-- State the theorem
theorem smallest_positive_n_squared (n : ℝ) : 
  (∀ b : ℝ, b > 0 → b < n → 
    ¬∃ x y r : ℝ, on_line x y b ∧ externally_tangent x y r ∧ internally_tangent x y r) →
  (∃ x y r : ℝ, on_line x y n ∧ externally_tangent x y r ∧ internally_tangent x y r) →
  n^2 = 49/64 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_n_squared_l3189_318985


namespace NUMINAMATH_CALUDE_student_selection_methods_l3189_318928

def total_students : ℕ := 8
def num_boys : ℕ := 6
def num_girls : ℕ := 2
def students_to_select : ℕ := 4
def boys_to_select : ℕ := 3
def girls_to_select : ℕ := 1

theorem student_selection_methods :
  (Nat.choose num_boys boys_to_select) * (Nat.choose num_girls girls_to_select) = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_selection_methods_l3189_318928


namespace NUMINAMATH_CALUDE_min_value_2a_plus_1_l3189_318946

theorem min_value_2a_plus_1 (a : ℝ) (h : 6 * a^2 + 5 * a + 4 = 3) :
  ∃ (min_val : ℝ), min_val = 0 ∧ ∀ (x : ℝ), (6 * x^2 + 5 * x + 4 = 3) → (2 * x + 1 ≥ min_val) := by
  sorry

end NUMINAMATH_CALUDE_min_value_2a_plus_1_l3189_318946


namespace NUMINAMATH_CALUDE_sum_of_cube_ratios_l3189_318965

open BigOperators

/-- Given a finite sequence of rational numbers x_t = i/101 for i = 0 to 101,
    the sum T = ∑(i=0 to 101) [x_i^3 / (3x_t^2 - 3x_t + 1)] is equal to 51. -/
theorem sum_of_cube_ratios (x : Fin 102 → ℚ) 
  (h : ∀ i : Fin 102, x i = (i : ℚ) / 101) : 
  ∑ i : Fin 102, (x i)^3 / (3 * (x i)^2 - 3 * (x i) + 1) = 51 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_ratios_l3189_318965


namespace NUMINAMATH_CALUDE_fraction_equality_l3189_318921

theorem fraction_equality (a b : ℚ) (h : a / b = 3 / 2) : (a + b) / a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3189_318921


namespace NUMINAMATH_CALUDE_halloween_bags_l3189_318956

theorem halloween_bags (total_students : ℕ) (pumpkin_students : ℕ) (pack_size : ℕ) (pack_price : ℕ) (individual_price : ℕ) (total_spent : ℕ) : 
  total_students = 25 →
  pumpkin_students = 14 →
  pack_size = 5 →
  pack_price = 3 →
  individual_price = 1 →
  total_spent = 17 →
  total_students - pumpkin_students = 11 :=
by sorry

end NUMINAMATH_CALUDE_halloween_bags_l3189_318956


namespace NUMINAMATH_CALUDE_real_part_of_z_l3189_318945

theorem real_part_of_z (z : ℂ) (h : Complex.I * z = 1 + 2 * Complex.I) : 
  Complex.re z = 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3189_318945


namespace NUMINAMATH_CALUDE_rate_percent_proof_l3189_318938

/-- Simple interest formula -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem rate_percent_proof (principal interest time : ℚ) 
  (h1 : principal = 800)
  (h2 : interest = 192)
  (h3 : time = 4)
  (h4 : simple_interest principal (6 : ℚ) time = interest) : 
  (6 : ℚ) = (interest * 100) / (principal * time) := by
  sorry

end NUMINAMATH_CALUDE_rate_percent_proof_l3189_318938


namespace NUMINAMATH_CALUDE_robot_constraint_l3189_318914

-- Define the robot's path as a parabola
def robot_path (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through P(-1, 0) with slope k
def line_through_P (k x y : ℝ) : Prop := y = k*(x + 1)

-- Define the condition that the line does not intersect the robot's path
def no_intersection (k : ℝ) : Prop :=
  ∀ x y : ℝ, robot_path x y ∧ line_through_P k x y → False

-- Theorem statement
theorem robot_constraint (k : ℝ) :
  no_intersection k ↔ k < -Real.sqrt 2 ∨ k > Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_robot_constraint_l3189_318914


namespace NUMINAMATH_CALUDE_lucas_1364_units_digit_l3189_318977

/-- Lucas numbers sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Property that Lucas numbers' units digits repeat every 12 terms -/
axiom lucas_units_period (n : ℕ) : lucas n % 10 = lucas (n % 12) % 10

/-- L₁₅ equals 1364 -/
axiom L_15 : lucas 15 = 1364

/-- Theorem: The units digit of L₁₃₆₄ is 7 -/
theorem lucas_1364_units_digit : lucas 1364 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_lucas_1364_units_digit_l3189_318977


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3189_318909

theorem repeating_decimal_sum_difference (x y z : ℚ) : 
  (x = 246 / 999) → 
  (y = 135 / 999) → 
  (z = 579 / 999) → 
  x - y + z = 230 / 333 := by
sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_difference_l3189_318909


namespace NUMINAMATH_CALUDE_jogger_speed_l3189_318915

/-- Proves that the jogger's speed is 9 kmph given the conditions of the problem -/
theorem jogger_speed (train_length : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ)
  (h1 : train_length = 120)
  (h2 : train_speed = 45)
  (h3 : initial_distance = 240)
  (h4 : passing_time = 36)
  : ∃ (jogger_speed : ℝ), jogger_speed = 9 ∧ 
    (train_speed - jogger_speed) * passing_time * (5/18) = initial_distance + train_length :=
by
  sorry

#check jogger_speed

end NUMINAMATH_CALUDE_jogger_speed_l3189_318915


namespace NUMINAMATH_CALUDE_trigonometric_problem_l3189_318948

theorem trigonometric_problem (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (Real.pi / 4 + α) = 1 / 3)
  (h4 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 ∧
  Real.sin β = -1 / 3 ∧
  α - β = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l3189_318948


namespace NUMINAMATH_CALUDE_canteen_distance_l3189_318942

theorem canteen_distance (a b c : ℝ) (h1 : a = 360) (h2 : c = 800) (h3 : a^2 + b^2 = c^2) :
  b / 2 = 438.6 := by sorry

end NUMINAMATH_CALUDE_canteen_distance_l3189_318942


namespace NUMINAMATH_CALUDE_x_equals_two_l3189_318939

theorem x_equals_two : ∀ x : ℝ, 3*x - 2*x + x = 3 - 2 + 1 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_two_l3189_318939


namespace NUMINAMATH_CALUDE_birgit_travel_time_l3189_318967

def hiking_time : ℝ := 3.5
def distance_traveled : ℝ := 21
def birgit_speed_difference : ℝ := 4
def birgit_travel_distance : ℝ := 8

theorem birgit_travel_time : 
  let total_minutes := hiking_time * 60
  let average_speed := total_minutes / distance_traveled
  let birgit_speed := average_speed - birgit_speed_difference
  birgit_speed * birgit_travel_distance = 48 := by sorry

end NUMINAMATH_CALUDE_birgit_travel_time_l3189_318967


namespace NUMINAMATH_CALUDE_juice_bottle_savings_l3189_318919

/-- Represents the volume and cost of a juice bottle -/
structure Bottle :=
  (volume : ℕ)
  (cost : ℕ)

/-- Calculates the savings when buying a big bottle instead of equivalent small bottles -/
def calculate_savings (big : Bottle) (small : Bottle) : ℕ :=
  let small_bottles_needed := big.volume / small.volume
  let small_bottles_cost := small_bottles_needed * small.cost
  small_bottles_cost - big.cost

/-- Theorem stating the savings when buying a big bottle instead of equivalent small bottles -/
theorem juice_bottle_savings :
  let big_bottle := Bottle.mk 30 2700
  let small_bottle := Bottle.mk 6 600
  calculate_savings big_bottle small_bottle = 300 := by
sorry

end NUMINAMATH_CALUDE_juice_bottle_savings_l3189_318919


namespace NUMINAMATH_CALUDE_twentieth_term_is_79_l3189_318907

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence is 79 -/
theorem twentieth_term_is_79 :
  arithmetic_sequence 3 4 20 = 79 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_is_79_l3189_318907


namespace NUMINAMATH_CALUDE_distance_between_centers_l3189_318930

/-- Given a triangle with sides 6, 8, and 10, the distance between the centers
    of its inscribed and circumscribed circles is √13. -/
theorem distance_between_centers (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let inradius := area / s
  let circumradius := (a * b * c) / (4 * area)
  Real.sqrt (circumradius^2 + inradius^2 - 2 * circumradius * inradius * Real.cos (π / 2)) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_centers_l3189_318930


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3189_318997

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 = 
   a₀ + a₁*(1 - x) + a₂*(1 - x)^2 + a₃*(1 - x)^3 + a₄*(1 - x)^4 + a₅*(1 - x)^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -57 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3189_318997


namespace NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l3189_318976

def icosahedralDie : Finset ℕ := Finset.range 20

theorem expected_digits_icosahedral_die :
  let E := (icosahedralDie.filter (λ n => n < 10)).card / 20 +
           2 * (icosahedralDie.filter (λ n => n ≥ 10)).card / 20
  E = 31 / 20 := by sorry

end NUMINAMATH_CALUDE_expected_digits_icosahedral_die_l3189_318976


namespace NUMINAMATH_CALUDE_department_store_problem_l3189_318929

/-- The cost price per item in yuan -/
def cost_price : ℝ := 120

/-- The relationship between selling price and daily sales volume -/
def price_volume_relation (x y : ℝ) : Prop := x + y = 200

/-- The daily profit function -/
def daily_profit (x : ℝ) : ℝ := (x - cost_price) * (200 - x)

theorem department_store_problem :
  (∃ a : ℝ, price_volume_relation 180 a ∧ a = 20) ∧
  (∃ x : ℝ, daily_profit x = 1600 ∧ x = 160) ∧
  (∀ m n : ℝ, m ≠ n → daily_profit (200 - m) = daily_profit (200 - n) → m + n = 80) :=
sorry

end NUMINAMATH_CALUDE_department_store_problem_l3189_318929


namespace NUMINAMATH_CALUDE_sqrt_and_principal_sqrt_of_zero_l3189_318996

-- Define square root function
noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

-- Define principal square root function
noncomputable def principal_sqrt (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.sqrt x else 0

-- Theorem statement
theorem sqrt_and_principal_sqrt_of_zero :
  sqrt 0 = 0 ∧ principal_sqrt 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_and_principal_sqrt_of_zero_l3189_318996


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l3189_318966

theorem shaded_area_percentage (square_side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  square_side_length = 20 →
  rectangle_width = 20 →
  rectangle_length = 35 →
  (((2 * square_side_length - rectangle_length) * square_side_length) / (rectangle_width * rectangle_length)) * 100 = 14.29 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l3189_318966


namespace NUMINAMATH_CALUDE_class_test_results_l3189_318923

theorem class_test_results (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.20) :
  p_first + p_second - (1 - p_neither) = 0.60 := by
  sorry

end NUMINAMATH_CALUDE_class_test_results_l3189_318923


namespace NUMINAMATH_CALUDE_chrysler_leeward_floor_difference_l3189_318908

theorem chrysler_leeward_floor_difference :
  ∀ (chrysler_floors leeward_floors : ℕ),
    chrysler_floors > leeward_floors →
    chrysler_floors + leeward_floors = 35 →
    chrysler_floors = 23 →
    chrysler_floors - leeward_floors = 11 := by
  sorry

end NUMINAMATH_CALUDE_chrysler_leeward_floor_difference_l3189_318908


namespace NUMINAMATH_CALUDE_best_representative_l3189_318998

structure Student where
  name : String
  average_time : Float
  variance : Float

def is_better (s1 s2 : Student) : Prop :=
  (s1.average_time < s2.average_time) ∨
  (s1.average_time = s2.average_time ∧ s1.variance < s2.variance)

def is_best (s : Student) (students : List Student) : Prop :=
  ∀ other ∈ students, s ≠ other → is_better s other

theorem best_representative (students : List Student) :
  let a := { name := "A", average_time := 7.9, variance := 1.4 }
  let b := { name := "B", average_time := 8.2, variance := 2.2 }
  let c := { name := "C", average_time := 7.9, variance := 2.4 }
  let d := { name := "D", average_time := 8.2, variance := 1.4 }
  students = [a, b, c, d] →
  is_best a students :=
by sorry

end NUMINAMATH_CALUDE_best_representative_l3189_318998


namespace NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l3189_318905

/-- The nth positive integer that is both odd and a multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ :=
  10 * n - 5

theorem eighth_odd_multiple_of_5 : nthOddMultipleOf5 8 = 75 := by
  sorry

end NUMINAMATH_CALUDE_eighth_odd_multiple_of_5_l3189_318905


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l3189_318910

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  A ∩ B = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l3189_318910


namespace NUMINAMATH_CALUDE_intersection_A_B_l3189_318989

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x - 2| ≥ 1}
def B : Set ℝ := {x : ℝ | 1 / x < 1}

-- State the theorem
theorem intersection_A_B : 
  ∀ x : ℝ, x ∈ A ∩ B ↔ x < 0 ∨ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3189_318989


namespace NUMINAMATH_CALUDE_c_profit_is_21000_l3189_318913

/-- Calculates the profit for a partner given the total profit, total parts, and the partner's parts. -/
def calculateProfit (totalProfit : ℕ) (totalParts : ℕ) (partnerParts : ℕ) : ℕ :=
  (totalProfit / totalParts) * partnerParts

/-- Proves that given the specified conditions, C's profit is $21000. -/
theorem c_profit_is_21000 (totalProfit : ℕ) (a_parts b_parts c_parts : ℕ) :
  totalProfit = 56700 →
  a_parts = 8 →
  b_parts = 9 →
  c_parts = 10 →
  calculateProfit totalProfit (a_parts + b_parts + c_parts) c_parts = 21000 := by
  sorry

#eval calculateProfit 56700 27 10

end NUMINAMATH_CALUDE_c_profit_is_21000_l3189_318913


namespace NUMINAMATH_CALUDE_negation_of_implication_l3189_318937

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x^2 - 3*x + 2 > 0) ↔ (x ≤ 2 → x^2 - 3*x + 2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3189_318937


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3189_318925

def set_A : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 1)}
def set_B : Set ℝ := Set.Ioo 0 1

theorem intersection_of_A_and_B :
  set_A ∩ set_B = Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3189_318925


namespace NUMINAMATH_CALUDE_xy_value_l3189_318916

theorem xy_value (x y : ℝ) (h : |x - y + 1| + (y + 5)^2 = 0) : x * y = 30 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3189_318916


namespace NUMINAMATH_CALUDE_n_gon_regions_l3189_318931

/-- The number of regions into which the diagonals of an n-gon divide it -/
def R (n : ℕ) : ℕ := (n*(n-1)*(n-2)*(n-3))/24 + (n*(n-3))/2 + 1

/-- Theorem stating the number of regions in an n-gon divided by its diagonals -/
theorem n_gon_regions (n : ℕ) (h : n ≥ 3) :
  R n = (n*(n-1)*(n-2)*(n-3))/24 + (n*(n-3))/2 + 1 :=
by sorry


end NUMINAMATH_CALUDE_n_gon_regions_l3189_318931


namespace NUMINAMATH_CALUDE_palic_characterization_l3189_318980

/-- Palic function definition -/
def isPalic (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
  Continuous f ∧
  ∀ x y z : ℝ, f x + f y + f z = f (a*x + b*y + c*z) + f (b*x + c*y + a*z) + f (c*x + a*y + b*z)

/-- Theorem: Characterization of Palic functions -/
theorem palic_characterization (a b c : ℝ) 
    (h1 : a + b + c = 1)
    (h2 : a^2 + b^2 + c^2 = 1)
    (h3 : a^3 + b^3 + c^3 ≠ 1)
    (f : ℝ → ℝ)
    (hf : isPalic f a b c) :
  ∃ p q r : ℝ, ∀ x : ℝ, f x = p * x^2 + q * x + r :=
sorry

end NUMINAMATH_CALUDE_palic_characterization_l3189_318980


namespace NUMINAMATH_CALUDE_bruces_shopping_money_l3189_318975

theorem bruces_shopping_money (initial_amount : ℕ) (shirt_cost : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 71 →
  shirt_cost = 5 →
  num_shirts = 5 →
  pants_cost = 26 →
  remaining_amount = initial_amount - (num_shirts * shirt_cost + pants_cost) →
  remaining_amount = 20 := by
sorry

end NUMINAMATH_CALUDE_bruces_shopping_money_l3189_318975


namespace NUMINAMATH_CALUDE_eggs_per_box_l3189_318906

theorem eggs_per_box (total_eggs : ℕ) (num_boxes : ℕ) (h1 : total_eggs = 15) (h2 : num_boxes = 5) :
  total_eggs / num_boxes = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_box_l3189_318906


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3189_318922

def M : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2 * x ∧ -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3189_318922


namespace NUMINAMATH_CALUDE_sports_club_intersection_l3189_318986

theorem sports_club_intersection (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h1 : total = 30)
  (h2 : badminton = 17)
  (h3 : tennis = 17)
  (h4 : neither = 2) :
  badminton + tennis - (total - neither) = 6 :=
by sorry

end NUMINAMATH_CALUDE_sports_club_intersection_l3189_318986


namespace NUMINAMATH_CALUDE_prime_divisors_condition_l3189_318912

theorem prime_divisors_condition (a n : ℕ) (ha : a > 2) :
  (∀ p : ℕ, Nat.Prime p → p ∣ (a^n - 1) → p ∣ (a^(3^2016) - 1)) →
  ∃ l : ℕ, l > 0 ∧ a = 2^l - 1 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_condition_l3189_318912


namespace NUMINAMATH_CALUDE_wall_length_given_mirror_area_l3189_318949

/-- Given a square mirror and a rectangular wall, prove the length of the wall
    when the mirror's area is half the wall's area. -/
theorem wall_length_given_mirror_area (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 24 →
  wall_width = 42 →
  (mirror_side ^ 2) * 2 = wall_width * (27.4285714 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_wall_length_given_mirror_area_l3189_318949


namespace NUMINAMATH_CALUDE_amelia_painted_faces_l3189_318979

/-- The number of faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The number of cuboids Amelia painted -/
def number_of_cuboids : ℕ := 6

/-- The total number of faces painted by Amelia -/
def total_faces_painted : ℕ := faces_per_cuboid * number_of_cuboids

theorem amelia_painted_faces :
  total_faces_painted = 36 :=
by sorry

end NUMINAMATH_CALUDE_amelia_painted_faces_l3189_318979


namespace NUMINAMATH_CALUDE_compare_M_and_N_range_of_m_l3189_318960

-- Problem 1
theorem compare_M_and_N : ∀ x : ℝ, 2 * x^2 + 1 > x^2 + 2*x - 1 := by sorry

-- Problem 2
theorem range_of_m : 
  (∀ m : ℝ, (∀ x : ℝ, 2*m ≤ x ∧ x ≤ m+1 → -1 ≤ x ∧ x ≤ 1) → -1/2 ≤ m ∧ m ≤ 0) := by sorry

end NUMINAMATH_CALUDE_compare_M_and_N_range_of_m_l3189_318960


namespace NUMINAMATH_CALUDE_product_congruence_l3189_318947

theorem product_congruence : 45 * 68 * 99 ≡ 15 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_product_congruence_l3189_318947


namespace NUMINAMATH_CALUDE_equivalent_statements_l3189_318953

variable (P Q : Prop)

theorem equivalent_statements :
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) :=
sorry

end NUMINAMATH_CALUDE_equivalent_statements_l3189_318953


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3189_318970

theorem arithmetic_equality : (50 - (2050 - 250)) + (2050 - (250 - 50)) - 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3189_318970


namespace NUMINAMATH_CALUDE_floor_e_equals_two_l3189_318999

theorem floor_e_equals_two : ⌊Real.exp 1⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_e_equals_two_l3189_318999


namespace NUMINAMATH_CALUDE_ellipse_k_range_l3189_318987

/-- An ellipse equation with parameter k -/
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k*y^2 = 2

/-- Foci of the ellipse are on the y-axis -/
def foci_on_y_axis (k : ℝ) : Prop := sorry

/-- The equation represents an ellipse -/
def is_ellipse (k : ℝ) : Prop := sorry

theorem ellipse_k_range (k : ℝ) : 
  (∀ x y : ℝ, ellipse_equation x y k) → 
  is_ellipse k → 
  foci_on_y_axis k → 
  0 < k ∧ k < 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l3189_318987


namespace NUMINAMATH_CALUDE_binomial_10_choose_4_l3189_318926

theorem binomial_10_choose_4 : Nat.choose 10 4 = 210 := by sorry

end NUMINAMATH_CALUDE_binomial_10_choose_4_l3189_318926


namespace NUMINAMATH_CALUDE_monomial_degree_l3189_318954

/-- Given that (a-2)x^2y^(|a|+1) is a monomial of degree 5 in x and y, and (a-2) ≠ 0, prove that a = -2 -/
theorem monomial_degree (a : ℤ) : 
  (∃ (x y : ℝ), (a - 2) * x^2 * y^(|a| + 1) ≠ 0) →  -- (a-2)x^2y^(|a|+1) is a monomial
  (2 + |a| + 1 = 5) →  -- The degree of the monomial in x and y is 5
  (a - 2 ≠ 0) →  -- (a-2) ≠ 0
  a = -2 := by
sorry


end NUMINAMATH_CALUDE_monomial_degree_l3189_318954


namespace NUMINAMATH_CALUDE_semi_truck_journey_l3189_318900

/-- A problem about a semi truck's journey on paved and dirt roads. -/
theorem semi_truck_journey (total_distance : ℝ) (paved_time : ℝ) (dirt_speed : ℝ) 
  (speed_difference : ℝ) (h1 : total_distance = 200) 
  (h2 : paved_time = 2) (h3 : dirt_speed = 32) (h4 : speed_difference = 20) : 
  (total_distance - paved_time * (dirt_speed + speed_difference)) / dirt_speed = 3 := by
  sorry

#check semi_truck_journey

end NUMINAMATH_CALUDE_semi_truck_journey_l3189_318900


namespace NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_5_above_1000_l3189_318944

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

theorem smallest_odd_digit_multiple_of_5_above_1000 :
  ∃ (n : ℕ), n = 1115 ∧
    has_only_odd_digits n ∧
    n > 1000 ∧
    n % 5 = 0 ∧
    ∀ m : ℕ, has_only_odd_digits m ∧ m > 1000 ∧ m % 5 = 0 → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_digit_multiple_of_5_above_1000_l3189_318944


namespace NUMINAMATH_CALUDE_fourth_group_trees_l3189_318958

theorem fourth_group_trees (total_groups : Nat) (average_trees : Nat)
  (group1_trees group2_trees group3_trees group5_trees : Nat) :
  total_groups = 5 →
  average_trees = 13 →
  group1_trees = 12 →
  group2_trees = 15 →
  group3_trees = 12 →
  group5_trees = 11 →
  (group1_trees + group2_trees + group3_trees + 15 + group5_trees) / total_groups = average_trees :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_group_trees_l3189_318958


namespace NUMINAMATH_CALUDE_star_operation_sum_l3189_318963

theorem star_operation_sum (c d : ℕ) : 
  c ≥ 2 → d ≥ 2 → c^d + c*d = 42 → c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_star_operation_sum_l3189_318963


namespace NUMINAMATH_CALUDE_ellipse_x_intercept_l3189_318968

/-- Definition of an ellipse with given foci and one x-intercept -/
def Ellipse (f1 f2 x1 : ℝ × ℝ) :=
  (∀ p : ℝ × ℝ, p.2 = 0 → dist p f1 + dist p f2 = dist x1 f1 + dist x1 f2) ∧
  (x1.2 = 0)

/-- Theorem: For an ellipse with foci (0, 3) and (4, 0), and one x-intercept at (0, 0),
    the other x-intercept is (56/11, 0) -/
theorem ellipse_x_intercept :
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  let x1 : ℝ × ℝ := (0, 0)
  let x2 : ℝ × ℝ := (56/11, 0)
  Ellipse f1 f2 x1 → x2.2 = 0 ∧ dist x2 f1 + dist x2 f2 = dist x1 f1 + dist x1 f2 :=
by sorry

#check ellipse_x_intercept

end NUMINAMATH_CALUDE_ellipse_x_intercept_l3189_318968


namespace NUMINAMATH_CALUDE_internet_service_duration_l3189_318936

/-- Calculates the number of days of internet service given the specified parameters. -/
def internetServiceDays (initialBalance : ℚ) (dailyCost : ℚ) (debtLimit : ℚ) (payment : ℚ) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that given the specified parameters, the number of days of internet service is 14. -/
theorem internet_service_duration :
  internetServiceDays 0 (1/2) 5 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_internet_service_duration_l3189_318936


namespace NUMINAMATH_CALUDE_sqrt_product_equals_three_l3189_318934

theorem sqrt_product_equals_three : Real.sqrt (1/2) * Real.sqrt 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equals_three_l3189_318934


namespace NUMINAMATH_CALUDE_one_third_of_36_l3189_318911

theorem one_third_of_36 : (1 / 3 : ℚ) * 36 = 12 := by sorry

end NUMINAMATH_CALUDE_one_third_of_36_l3189_318911


namespace NUMINAMATH_CALUDE_inhabitable_earth_fraction_l3189_318995

/-- Represents the fraction of Earth's surface not covered by water -/
def land_fraction : ℚ := 1 / 3

/-- Represents the fraction of exposed land that is inhabitable -/
def inhabitable_land_fraction : ℚ := 1 / 3

/-- Theorem stating the fraction of Earth's surface that is inhabitable for humans -/
theorem inhabitable_earth_fraction :
  land_fraction * inhabitable_land_fraction = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_inhabitable_earth_fraction_l3189_318995


namespace NUMINAMATH_CALUDE_tan_alpha_values_l3189_318904

theorem tan_alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 - Real.cos (2 * α)) :
  Real.tan α = 2 ∨ Real.tan α = 0 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l3189_318904


namespace NUMINAMATH_CALUDE_polynomial_divisibility_double_divisibility_not_triple_divisible_l3189_318901

/-- Definition of the polynomial P_n(x) -/
def P (n : ℕ) (x : ℝ) : ℝ := (x + 1)^n - x^n - 1

/-- Definition of divisibility for polynomials -/
def divisible (p q : ℝ → ℝ) : Prop := ∃ r : ℝ → ℝ, ∀ x, p x = q x * r x

theorem polynomial_divisibility (n : ℕ) :
  (∃ k : ℤ, n = 6 * k + 1 ∨ n = 6 * k - 1) ↔ 
  divisible (P n) (fun x ↦ x^2 + x + 1) :=
sorry

theorem double_divisibility (n : ℕ) :
  (∃ k : ℤ, n = 6 * k + 1) ↔ 
  divisible (P n) (fun x ↦ (x^2 + x + 1)^2) :=
sorry

theorem not_triple_divisible (n : ℕ) :
  ¬(divisible (P n) (fun x ↦ (x^2 + x + 1)^3)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_double_divisibility_not_triple_divisible_l3189_318901


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3189_318972

theorem sum_of_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / a + 1 / b = (a + b) / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3189_318972


namespace NUMINAMATH_CALUDE_sum_of_four_digit_primes_and_multiples_of_three_l3189_318994

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def count_four_digit_primes : ℕ := sorry

def count_four_digit_multiples_of_three : ℕ := sorry

theorem sum_of_four_digit_primes_and_multiples_of_three :
  count_four_digit_primes + count_four_digit_multiples_of_three = 4061 := by sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_primes_and_multiples_of_three_l3189_318994


namespace NUMINAMATH_CALUDE_ln_inequality_solution_set_l3189_318973

theorem ln_inequality_solution_set (x : ℝ) : 
  Real.log (x^2 - 2*x - 2) > 0 ↔ x > 3 ∨ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_solution_set_l3189_318973


namespace NUMINAMATH_CALUDE_product_of_roots_abs_equation_l3189_318940

theorem product_of_roots_abs_equation (x : ℝ) :
  (∃ a b : ℝ, a ≠ b ∧ 
   (abs a)^2 - 3 * abs a - 10 = 0 ∧
   (abs b)^2 - 3 * abs b - 10 = 0 ∧
   a * b = -25) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_abs_equation_l3189_318940


namespace NUMINAMATH_CALUDE_ice_cube_melting_l3189_318978

theorem ice_cube_melting (V : ℝ) : 
  V > 0 →
  (1/5) * (3/4) * (2/3) * (1/2) * V = 30 →
  V = 150 := by
sorry

end NUMINAMATH_CALUDE_ice_cube_melting_l3189_318978


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3189_318982

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem range_of_a :
  ∀ x : ℝ, f a x + g x ≥ 3 → a ∈ Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l3189_318982


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3189_318955

theorem complex_fraction_simplification :
  (5 : ℂ) / (2 - Complex.I) = 2 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3189_318955


namespace NUMINAMATH_CALUDE_mass_percentage_H_is_correct_l3189_318943

/-- The mass percentage of H in a certain compound -/
def mass_percentage_H : ℝ := 1.69

/-- Theorem stating that the mass percentage of H is 1.69% -/
theorem mass_percentage_H_is_correct : mass_percentage_H = 1.69 := by
  sorry

end NUMINAMATH_CALUDE_mass_percentage_H_is_correct_l3189_318943


namespace NUMINAMATH_CALUDE_horners_method_for_f_l3189_318983

def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 - 3*x + 2

theorem horners_method_for_f :
  f 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_horners_method_for_f_l3189_318983


namespace NUMINAMATH_CALUDE_x_minus_y_equals_pi_over_three_l3189_318993

theorem x_minus_y_equals_pi_over_three (x y : Real) 
  (h1 : 0 < y) (h2 : y < x) (h3 : x < π)
  (h4 : Real.tan x * Real.tan y = 2)
  (h5 : Real.sin x * Real.sin y = 1/3) : 
  x - y = π/3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_pi_over_three_l3189_318993


namespace NUMINAMATH_CALUDE_window_pane_length_l3189_318992

theorem window_pane_length 
  (num_panes : ℕ) 
  (pane_width : ℝ) 
  (total_area : ℝ) : ℝ :=
  let pane_area := total_area / num_panes
  let pane_length := pane_area / pane_width
  have h1 : num_panes = 8 := by sorry
  have h2 : pane_width = 8 := by sorry
  have h3 : total_area = 768 := by sorry
  have h4 : pane_length = 12 := by sorry
  pane_length

#check window_pane_length

end NUMINAMATH_CALUDE_window_pane_length_l3189_318992


namespace NUMINAMATH_CALUDE_age_problem_l3189_318950

/-- Given three people whose total present age is 90 years and whose ages were in the ratio 1:2:3 ten years ago, 
    the present age of the person who was in the middle of the ratio is 30 years. -/
theorem age_problem (a b c : ℕ) : 
  a + b + c = 90 →  -- Total present age is 90
  (a - 10) = (b - 10) / 2 →  -- Ratio condition for a and b
  (c - 10) = 3 * ((b - 10) / 2) →  -- Ratio condition for b and c
  b = 30 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l3189_318950


namespace NUMINAMATH_CALUDE_smallest_m_is_correct_l3189_318903

/-- The smallest positive value of m for which the equation 15x^2 - mx + 630 = 0 has integral solutions -/
def smallest_m : ℕ := 195

/-- Predicate to check if a quadratic equation ax^2 + bx + c = 0 has integral solutions -/
def has_integral_solutions (a b c : ℤ) : Prop :=
  ∃ x : ℤ, a * x^2 + b * x + c = 0

theorem smallest_m_is_correct :
  (∀ m : ℕ, m < smallest_m → ¬(has_integral_solutions 15 (-m) 630)) ∧
  (has_integral_solutions 15 (-smallest_m) 630) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_correct_l3189_318903


namespace NUMINAMATH_CALUDE_trigonometric_signs_l3189_318941

theorem trigonometric_signs :
  let expr1 := Real.sin (1125 * π / 180)
  let expr2 := Real.tan (37 * π / 12) * Real.sin (37 * π / 12)
  let expr3 := Real.sin 4 / Real.tan 4
  let expr4 := Real.sin (|(-1)|)
  (expr1 > 0) ∧ (expr2 < 0) ∧ (expr3 < 0) ∧ (expr4 > 0) := by sorry

end NUMINAMATH_CALUDE_trigonometric_signs_l3189_318941


namespace NUMINAMATH_CALUDE_circus_show_acrobats_l3189_318964

/-- Represents the number of acrobats in the circus show. -/
def numAcrobats : ℕ := 2

/-- Represents the number of elephants in the circus show. -/
def numElephants : ℕ := 14

/-- Represents the number of clowns in the circus show. -/
def numClowns : ℕ := 14

/-- The total number of legs observed in the circus show. -/
def totalLegs : ℕ := 88

/-- The total number of heads observed in the circus show. -/
def totalHeads : ℕ := 30

theorem circus_show_acrobats :
  (2 * numAcrobats + 4 * numElephants + 2 * numClowns = totalLegs) ∧
  (numAcrobats + numElephants + numClowns = totalHeads) ∧
  (numAcrobats = 2) := by sorry

end NUMINAMATH_CALUDE_circus_show_acrobats_l3189_318964


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l3189_318969

def age_difference : ℕ := 20
def younger_present_age : ℕ := 35
def years_ago : ℕ := 15

def elder_present_age : ℕ := younger_present_age + age_difference

def younger_past_age : ℕ := younger_present_age - years_ago
def elder_past_age : ℕ := elder_present_age - years_ago

theorem age_ratio_theorem :
  (elder_past_age % younger_past_age = 0) →
  (elder_past_age / younger_past_age = 2) :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_theorem_l3189_318969


namespace NUMINAMATH_CALUDE_rational_function_value_l3189_318957

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_neg_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_origin : p 0 = 0 ∧ q 0 ≠ 0
  passes_two_neg_one : p 2 / q 2 = -1

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 1 / f.q 1 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l3189_318957


namespace NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l3189_318917

theorem arcsin_cos_arcsin_plus_arccos_sin_arccos (x : ℝ) : 
  Real.arcsin (Real.cos (Real.arcsin x)) + Real.arccos (Real.sin (Real.arccos x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_cos_arcsin_plus_arccos_sin_arccos_l3189_318917


namespace NUMINAMATH_CALUDE_divisibility_1001_l3189_318918

theorem divisibility_1001 (n : ℕ) : 1001 ∣ n → 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_1001_l3189_318918
