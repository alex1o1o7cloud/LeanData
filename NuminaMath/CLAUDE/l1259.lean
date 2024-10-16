import Mathlib

namespace NUMINAMATH_CALUDE_cos_equality_implies_70_l1259_125998

theorem cos_equality_implies_70 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) 
  (h3 : Real.cos (n * π / 180) = Real.cos (1010 * π / 180)) : n = 70 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_implies_70_l1259_125998


namespace NUMINAMATH_CALUDE_equivalent_rotation_l1259_125974

/-- Given a full rotation of 450 degrees, if a point is rotated 650 degrees clockwise
    to reach a destination, then the equivalent counterclockwise rotation to reach
    the same destination is 250 degrees. -/
theorem equivalent_rotation (full_rotation : ℕ) (clockwise_rotation : ℕ) (counterclockwise_rotation : ℕ) : 
  full_rotation = 450 → 
  clockwise_rotation = 650 → 
  counterclockwise_rotation < full_rotation →
  (clockwise_rotation % full_rotation + counterclockwise_rotation) % full_rotation = 0 →
  counterclockwise_rotation = 250 := by
  sorry

#check equivalent_rotation

end NUMINAMATH_CALUDE_equivalent_rotation_l1259_125974


namespace NUMINAMATH_CALUDE_range_of_f_l1259_125957

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, x^2 ≠ 1 ∧ f x = y} = {y : ℝ | y < 0 ∨ y > 0} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1259_125957


namespace NUMINAMATH_CALUDE_inequality_conditions_l1259_125989

theorem inequality_conditions (a b : ℝ) :
  ((b > 0 ∧ 0 > a) → (1 / a < 1 / b)) ∧
  ((0 > a ∧ a > b) → (1 / a < 1 / b)) ∧
  ((a > 0 ∧ 0 > b) → ¬(1 / a < 1 / b)) ∧
  ((a > b ∧ b > 0) → (1 / a < 1 / b)) := by
sorry

end NUMINAMATH_CALUDE_inequality_conditions_l1259_125989


namespace NUMINAMATH_CALUDE_multiplication_problems_l1259_125975

theorem multiplication_problems :
  (25 * 5 * 2 * 4 = 1000) ∧ (1111 * 9999 = 11108889) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_problems_l1259_125975


namespace NUMINAMATH_CALUDE_square_area_possibilities_l1259_125915

/-- Represents a square in a 2D plane -/
structure Square where
  side_length : ℝ
  area : ℝ := side_length ^ 2

/-- Represents a parallelogram in a 2D plane -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  area : ℝ

/-- Represents an oblique projection from a square to a parallelogram -/
def oblique_projection (s : Square) (p : Parallelogram) : Prop :=
  (s.side_length = p.side1 ∨ s.side_length = p.side2) ∧ p.area = s.area

theorem square_area_possibilities (s : Square) (p : Parallelogram) :
  oblique_projection s p → p.side1 = 4 → s.area = 16 ∨ s.area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_possibilities_l1259_125915


namespace NUMINAMATH_CALUDE_total_paintable_area_l1259_125980

def num_bedrooms : ℕ := 4
def room_length : ℝ := 14
def room_width : ℝ := 11
def room_height : ℝ := 9
def unpaintable_area : ℝ := 70

def wall_area (length width height : ℝ) : ℝ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℝ) : ℝ :=
  total_area - unpaintable_area

theorem total_paintable_area :
  (num_bedrooms : ℝ) * paintable_area (wall_area room_length room_width room_height) unpaintable_area = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_l1259_125980


namespace NUMINAMATH_CALUDE_negative_square_l1259_125993

theorem negative_square : -3^2 = -9 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_l1259_125993


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_condition_l1259_125947

theorem arithmetic_geometric_mean_inequality_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b)) ∧ 
  ∃ a b : ℝ, ¬(a > 0 ∧ b > 0) ∧ (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_condition_l1259_125947


namespace NUMINAMATH_CALUDE_pear_sales_l1259_125977

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  afternoon_sales = 260 →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 390 :=
by sorry

end NUMINAMATH_CALUDE_pear_sales_l1259_125977


namespace NUMINAMATH_CALUDE_second_number_proof_l1259_125929

theorem second_number_proof (x : ℕ) : 
  (1255 % 29 = 8) → (x % 29 = 11) → x = 1287 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l1259_125929


namespace NUMINAMATH_CALUDE_operation_b_correct_operation_c_correct_l1259_125950

-- Operation B
theorem operation_b_correct (t : ℝ) : (-2 * t) * (3 * t + t^2 - 1) = -6 * t^2 - 2 * t^3 + 2 * t := by
  sorry

-- Operation C
theorem operation_c_correct (x y : ℝ) : (-2 * x * y^3)^2 = 4 * x^2 * y^6 := by
  sorry

end NUMINAMATH_CALUDE_operation_b_correct_operation_c_correct_l1259_125950


namespace NUMINAMATH_CALUDE_triangle_area_l1259_125903

/-- The area of a triangle with vertices at (2, -3), (8, 1), and (2, 3) is 18 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let A : (ℝ × ℝ) := (2, -3)
  let B : (ℝ × ℝ) := (8, 1)
  let C : (ℝ × ℝ) := (2, 3)

  -- Calculate the area of the triangle
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

  -- Prove that the area is equal to 18
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1259_125903


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1259_125969

theorem complex_expression_simplification :
  (8 - 5*Complex.I) + 3*(2 - 4*Complex.I) = 14 - 17*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1259_125969


namespace NUMINAMATH_CALUDE_smoothie_combinations_l1259_125902

theorem smoothie_combinations (n_smoothies : ℕ) (n_supplements : ℕ) : 
  n_smoothies = 7 → n_supplements = 8 → n_smoothies * (n_supplements.choose 3) = 392 := by
  sorry

end NUMINAMATH_CALUDE_smoothie_combinations_l1259_125902


namespace NUMINAMATH_CALUDE_grouping_factoring_1_grouping_factoring_2_l1259_125961

-- Expression 1
theorem grouping_factoring_1 (a b c : ℝ) :
  a^2 + 2*a*b + b^2 + a*c + b*c = (a + b) * (a + b + c) := by sorry

-- Expression 2
theorem grouping_factoring_2 (a x y : ℝ) :
  4*a^2 - x^2 + 4*x*y - 4*y^2 = (2*a + x - 2*y) * (2*a - x + 2*y) := by sorry

end NUMINAMATH_CALUDE_grouping_factoring_1_grouping_factoring_2_l1259_125961


namespace NUMINAMATH_CALUDE_markup_calculation_l1259_125988

/-- Calculates the required markup given the purchase price, overhead percentage, and desired net profit. -/
def calculate_markup (purchase_price : ℝ) (overhead_percent : ℝ) (net_profit : ℝ) : ℝ :=
  purchase_price * overhead_percent + net_profit

/-- Theorem stating that the markup for the given conditions is $53.75 -/
theorem markup_calculation :
  let purchase_price : ℝ := 75
  let overhead_percent : ℝ := 0.45
  let net_profit : ℝ := 20
  calculate_markup purchase_price overhead_percent net_profit = 53.75 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l1259_125988


namespace NUMINAMATH_CALUDE_c_range_l1259_125911

theorem c_range (a b c : ℝ) 
  (ha : 6 < a ∧ a < 10) 
  (hb : a / 2 ≤ b ∧ b ≤ 2 * a) 
  (hc : c = a + b) : 
  9 < c ∧ c < 30 := by
  sorry

end NUMINAMATH_CALUDE_c_range_l1259_125911


namespace NUMINAMATH_CALUDE_range_of_y_over_x_l1259_125992

theorem range_of_y_over_x (x y : ℝ) (h1 : 3 * x - 2 * y - 5 = 0) (h2 : 1 ≤ x) (h3 : x ≤ 2) :
  ∃ (z : ℝ), z = y / x ∧ -1 ≤ z ∧ z ≤ 1/4 :=
sorry

end NUMINAMATH_CALUDE_range_of_y_over_x_l1259_125992


namespace NUMINAMATH_CALUDE_inequality_system_solution_exists_l1259_125931

theorem inequality_system_solution_exists :
  ∃ (x y z t : ℝ), 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) ∧
    abs x < abs (y - z + t) ∧
    abs y < abs (x - z + t) ∧
    abs z < abs (x - y + t) ∧
    abs t < abs (x - y + z) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_exists_l1259_125931


namespace NUMINAMATH_CALUDE_max_product_sum_1988_l1259_125968

theorem max_product_sum_1988 (sequence : List Nat) : 
  (sequence.sum = 1988) → (sequence.all (· > 0)) →
  (sequence.prod ≤ 2 * 3^662) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_1988_l1259_125968


namespace NUMINAMATH_CALUDE_sqrt_sum_comparison_l1259_125932

theorem sqrt_sum_comparison : Real.sqrt 3 + Real.sqrt 6 > Real.sqrt 2 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_comparison_l1259_125932


namespace NUMINAMATH_CALUDE_infinitely_many_disconnected_pLandia_l1259_125949

/-- A function that determines if two islands are connected in p-Landia -/
def isConnected (p n m : ℕ) : Prop :=
  p ∣ (n^2 - m + 1) * (m^2 - n + 1)

/-- The graph representation of p-Landia -/
def pLandiaGraph (p : ℕ) : SimpleGraph ℕ :=
  SimpleGraph.fromRel (λ n m ↦ n ≠ m ∧ isConnected p n m)

/-- The theorem stating that infinitely many p-Landia graphs are disconnected -/
theorem infinitely_many_disconnected_pLandia :
  ∃ (S : Set ℕ), (∀ p ∈ S, Nat.Prime p) ∧ Set.Infinite S ∧
    ∀ p ∈ S, ¬(pLandiaGraph p).Connected :=
  sorry

end NUMINAMATH_CALUDE_infinitely_many_disconnected_pLandia_l1259_125949


namespace NUMINAMATH_CALUDE_emerson_rowing_theorem_l1259_125953

/-- Emerson's rowing trip -/
def emerson_rowing_trip (morning_speed : ℝ) : Prop :=
  let afternoon_speed := morning_speed + 2
  let current_speed := morning_speed - 1.5
  let morning_time := 2
  let afternoon_time := 15 / afternoon_speed
  let calm_time := 9 / morning_speed
  let current_time := 9 / current_speed
  let total_time := morning_time + afternoon_time + calm_time + current_time
  (6 / morning_speed = 2) ∧ (total_time = 14)

theorem emerson_rowing_theorem :
  ∃ (morning_speed : ℝ), emerson_rowing_trip morning_speed :=
sorry

end NUMINAMATH_CALUDE_emerson_rowing_theorem_l1259_125953


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1259_125935

/-- In a Cartesian coordinate system, the coordinates of the point (11, 9) with respect to the origin are (11, 9). -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (11, 9)
  P = P :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1259_125935


namespace NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l1259_125978

theorem multiply_whole_and_mixed_number : 8 * (9 + 2/5) = 75 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_whole_and_mixed_number_l1259_125978


namespace NUMINAMATH_CALUDE_current_year_is_2021_l1259_125908

/-- The year Aziz's parents moved to America -/
def parents_move_year : ℕ := 1982

/-- Aziz's current age -/
def aziz_age : ℕ := 36

/-- Years Aziz's parents lived in America before his birth -/
def years_before_birth : ℕ := 3

/-- The current year -/
def current_year : ℕ := parents_move_year + aziz_age + years_before_birth

theorem current_year_is_2021 : current_year = 2021 := by
  sorry

end NUMINAMATH_CALUDE_current_year_is_2021_l1259_125908


namespace NUMINAMATH_CALUDE_stamp_sum_l1259_125976

theorem stamp_sum : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n < 100 ∧ n % 6 = 4 ∧ n % 8 = 2) ∧ 
  (∀ n < 100, n % 6 = 4 ∧ n % 8 = 2 → n ∈ S) ∧
  S.sum id = 68 := by
sorry

end NUMINAMATH_CALUDE_stamp_sum_l1259_125976


namespace NUMINAMATH_CALUDE_count_twos_in_hotel_l1259_125913

/-- Represents a hotel room number -/
structure RoomNumber where
  floor : Nat
  room : Nat
  h1 : 1 ≤ floor ∧ floor ≤ 5
  h2 : 1 ≤ room ∧ room ≤ 35

/-- Counts occurrences of a digit in a natural number -/
def countDigit (digit : Nat) (n : Nat) : Nat :=
  sorry

/-- All room numbers in the hotel -/
def allRoomNumbers : List RoomNumber :=
  sorry

/-- Counts occurrences of digit 2 in all room numbers -/
def countTwos : Nat :=
  sorry

theorem count_twos_in_hotel : countTwos = 105 := by
  sorry

end NUMINAMATH_CALUDE_count_twos_in_hotel_l1259_125913


namespace NUMINAMATH_CALUDE_shaded_design_area_l1259_125955

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle in the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The shaded design in the 7x7 grid -/
def shaded_design : List GridTriangle := sorry

/-- Calculates the area of a single triangle in the grid -/
def triangle_area (t : GridTriangle) : Rat := sorry

/-- Calculates the total area of the shaded design -/
def total_area (design : List GridTriangle) : Rat :=
  design.map triangle_area |>.sum

/-- The theorem stating that the area of the shaded design is 1.5 -/
theorem shaded_design_area :
  total_area shaded_design = 3/2 := by sorry

end NUMINAMATH_CALUDE_shaded_design_area_l1259_125955


namespace NUMINAMATH_CALUDE_difference_of_squares_l1259_125963

theorem difference_of_squares (m n : ℝ) : (m + n) * (-m + n) = -m^2 + n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1259_125963


namespace NUMINAMATH_CALUDE_odd_function_properties_l1259_125946

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an increasing function on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Define the minimum value of a function on an interval
def HasMinValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → v ≤ f x) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

-- Define the maximum value of a function on an interval
def HasMaxValueOn (f : ℝ → ℝ) (v : ℝ) (a b : ℝ) : Prop :=
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ v) ∧ (∃ x, a ≤ x ∧ x ≤ b ∧ f x = v)

-- Theorem statement
theorem odd_function_properties (f : ℝ → ℝ) :
  OddFunction f →
  IncreasingOn f 1 3 →
  HasMinValueOn f 0 1 3 →
  IncreasingOn f (-3) (-1) ∧ HasMaxValueOn f 0 (-3) (-1) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l1259_125946


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1259_125960

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  (∀ x, x^2 + b*x + 49 = (x + m)^2 + 9) ∧ (b > 0) → b = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1259_125960


namespace NUMINAMATH_CALUDE_xy_sum_product_l1259_125927

theorem xy_sum_product (x y : ℝ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : x*y + x + y = 7) : 
  x^2*y + x*y^2 = 196/25 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_product_l1259_125927


namespace NUMINAMATH_CALUDE_min_value_implies_a_l1259_125942

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 0}

-- State the theorem
theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ domain, f a x ≥ 1) ∧ (∃ x ∈ domain, f a x = 1) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l1259_125942


namespace NUMINAMATH_CALUDE_last_digit_base_5_l1259_125987

theorem last_digit_base_5 (n : ℕ) (h : n = 119) : n % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_base_5_l1259_125987


namespace NUMINAMATH_CALUDE_kendall_driving_distance_l1259_125948

/-- The distance Kendall drove with her mother -/
def distance_with_mother : ℝ := 0.67 - 0.5

/-- The total distance Kendall drove -/
def total_distance : ℝ := 0.67

/-- The distance Kendall drove with her father -/
def distance_with_father : ℝ := 0.5

theorem kendall_driving_distance :
  distance_with_mother = 0.17 :=
by sorry

end NUMINAMATH_CALUDE_kendall_driving_distance_l1259_125948


namespace NUMINAMATH_CALUDE_probability_spade_face_diamond_l1259_125983

/-- Represents a standard 52-card deck --/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (h : cards.card = 52)

/-- Represents a suit in a deck of cards --/
inductive Suit
| Spade | Heart | Diamond | Club

/-- Represents a rank in a deck of cards --/
inductive Rank
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Checks if a rank is a face card --/
def is_face_card (r : Rank) : Bool :=
  match r with
  | Rank.Jack | Rank.Queen | Rank.King => true
  | _ => false

/-- Calculates the probability of drawing three specific cards --/
def probability_three_cards (d : Deck) (first : Suit) (second : Rank → Bool) (third : Suit) : ℚ :=
  sorry

/-- Theorem stating the probability of drawing a spade, then a face card, then a diamond --/
theorem probability_spade_face_diamond (d : Deck) :
  probability_three_cards d Suit.Spade is_face_card Suit.Diamond = 1911 / 132600 :=
sorry

end NUMINAMATH_CALUDE_probability_spade_face_diamond_l1259_125983


namespace NUMINAMATH_CALUDE_difference_nonnegative_equivalence_l1259_125934

theorem difference_nonnegative_equivalence (x : ℝ) :
  (x - 8 ≥ 0) ↔ (∃ (y : ℝ), y ≥ 0 ∧ x - 8 = y) :=
by sorry

end NUMINAMATH_CALUDE_difference_nonnegative_equivalence_l1259_125934


namespace NUMINAMATH_CALUDE_seahawks_touchdowns_l1259_125944

theorem seahawks_touchdowns (final_score : ℕ) (field_goals : ℕ) 
  (h1 : final_score = 37)
  (h2 : field_goals = 3) :
  (final_score - field_goals * 3) / 7 = 4 := by
  sorry

#check seahawks_touchdowns

end NUMINAMATH_CALUDE_seahawks_touchdowns_l1259_125944


namespace NUMINAMATH_CALUDE_skier_total_time_l1259_125926

theorem skier_total_time (x : ℝ) (t₁ t₂ t₃ : ℝ) 
  (h1 : t₁ + t₂ = 40.5)
  (h2 : t₂ + t₃ = 37.5)
  (h3 : x / t₂ = (2 * x) / (t₁ + t₃))
  (h4 : x > 0) :
  t₁ + t₂ + t₃ = 58.5 := by
sorry

end NUMINAMATH_CALUDE_skier_total_time_l1259_125926


namespace NUMINAMATH_CALUDE_discounted_price_theorem_l1259_125912

/-- The actual price of the good before discounts -/
def actual_price : ℝ := 9356.725146198829

/-- The first discount rate -/
def discount1 : ℝ := 0.20

/-- The second discount rate -/
def discount2 : ℝ := 0.10

/-- The third discount rate -/
def discount3 : ℝ := 0.05

/-- The final selling price after all discounts -/
def final_price : ℝ := 6400

/-- Theorem stating that applying the successive discounts to the actual price results in the final price -/
theorem discounted_price_theorem :
  actual_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = final_price := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_theorem_l1259_125912


namespace NUMINAMATH_CALUDE_optimal_solution_l1259_125964

-- Define the problem parameters
def days_together : ℝ := 12
def cost_A_per_day : ℝ := 40000
def cost_B_per_day : ℝ := 30000
def max_days : ℝ := 30 -- one month

-- Define the relationship between Team A and Team B's completion times
def team_B_multiplier : ℝ := 1.5

-- Define the function to calculate days needed for Team A
def days_A (x : ℝ) : Prop := (1 / x) + (1 / (team_B_multiplier * x)) = (1 / days_together)

-- Define the function to calculate days needed for Team B
def days_B (x : ℝ) : ℝ := team_B_multiplier * x

-- Define the cost function for Team A working alone
def cost_A (x : ℝ) : ℝ := cost_A_per_day * x

-- Define the cost function for Team B working alone
def cost_B (x : ℝ) : ℝ := cost_B_per_day * days_B x

-- Define the cost function for both teams working together
def cost_together : ℝ := (cost_A_per_day + cost_B_per_day) * days_together

-- Theorem: Team A working alone for 20 days is the optimal solution
theorem optimal_solution (x : ℝ) :
  days_A x →
  x ≤ max_days →
  days_B x ≤ max_days →
  cost_A x ≤ cost_B x ∧
  cost_A x ≤ cost_together ∧
  x = 20 ∧
  cost_A x = 800000 :=
sorry

end NUMINAMATH_CALUDE_optimal_solution_l1259_125964


namespace NUMINAMATH_CALUDE_last_digit_98_base5_l1259_125973

def last_digit_base5 (n : ℕ) : ℕ :=
  n % 5

theorem last_digit_98_base5 :
  last_digit_base5 98 = 3 := by
sorry

end NUMINAMATH_CALUDE_last_digit_98_base5_l1259_125973


namespace NUMINAMATH_CALUDE_sequence_third_term_l1259_125905

/-- Given a sequence {a_n} with general term a_n = 3n - 5, prove that a_3 = 4 -/
theorem sequence_third_term (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_third_term_l1259_125905


namespace NUMINAMATH_CALUDE_initial_gasohol_volume_l1259_125940

/-- Represents the composition of a fuel mixture -/
structure FuelMixture where
  ethanol : ℝ  -- Percentage of ethanol
  gasoline : ℝ  -- Percentage of gasoline

/-- Represents the state of the fuel tank -/
structure FuelTank where
  volume : ℝ  -- Total volume in liters
  mixture : FuelMixture  -- Composition of the mixture

def initial_mixture : FuelMixture := { ethanol := 0.05, gasoline := 0.95 }
def desired_mixture : FuelMixture := { ethanol := 0.10, gasoline := 0.90 }
def ethanol_added : ℝ := 2.5

theorem initial_gasohol_volume (initial : FuelTank) (final : FuelTank) :
  initial.mixture = initial_mixture →
  final.mixture = desired_mixture →
  final.volume = initial.volume + ethanol_added →
  final.volume * final.mixture.ethanol = initial.volume * initial.mixture.ethanol + ethanol_added →
  initial.volume = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_gasohol_volume_l1259_125940


namespace NUMINAMATH_CALUDE_one_root_quadratic_l1259_125938

theorem one_root_quadratic (a : ℤ) : 
  (∃! x : ℝ, x ∈ Set.Icc 1 8 ∧ (x - a - 4)^2 + 2*x - 2*a - 16 = 0) ↔ 
  (a ∈ Set.Icc (-5) 0 ∨ a ∈ Set.Icc 3 8) :=
sorry

end NUMINAMATH_CALUDE_one_root_quadratic_l1259_125938


namespace NUMINAMATH_CALUDE_smallest_second_term_arithmetic_sequence_l1259_125996

theorem smallest_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
  a > 0 →
  d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 95 →
  ∀ (b e : ℕ),
  b > 0 →
  e > 0 →
  b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 95 →
  (a + d) ≤ (b + e) →
  a + d = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_second_term_arithmetic_sequence_l1259_125996


namespace NUMINAMATH_CALUDE_perp_plane_necessary_not_sufficient_l1259_125966

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the "line in plane" relation
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem perp_plane_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (h_diff : α ≠ β)
  (h_m_in_α : line_in_plane m α) :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧
  ¬(perp_planes α β → perp_line_plane m β) :=
sorry

end NUMINAMATH_CALUDE_perp_plane_necessary_not_sufficient_l1259_125966


namespace NUMINAMATH_CALUDE_equilateral_triangle_count_l1259_125959

/-- Represents a point in a hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a hexagonal lattice with a secondary layer -/
structure HexagonalLattice where
  inner : List LatticePoint
  outer : List LatticePoint

/-- Represents an equilateral triangle in the lattice -/
structure EquilateralTriangle where
  vertices : List LatticePoint
  sideLength : ℝ

/-- Function to count equilateral triangles in the lattice -/
def countEquilateralTriangles (lattice : HexagonalLattice) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles with side lengths 1 or √7 is 6 -/
theorem equilateral_triangle_count (lattice : HexagonalLattice) :
  countEquilateralTriangles lattice = 6 :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_count_l1259_125959


namespace NUMINAMATH_CALUDE_b_payment_l1259_125943

/-- Calculate the amount person b should pay for renting a pasture -/
theorem b_payment (total_rent : ℚ) (a_horses a_months b_horses b_months c_horses c_months d_horses d_months : ℕ) :
  let total_shares := a_horses * a_months + b_horses * b_months + c_horses * c_months + d_horses * d_months
  let b_shares := b_horses * b_months
  let b_payment := (total_rent / total_shares) * b_shares
  total_rent = 1260 ∧ 
  a_horses = 15 ∧ a_months = 10 ∧
  b_horses = 18 ∧ b_months = 11 ∧
  c_horses = 20 ∧ c_months = 8 ∧
  d_horses = 25 ∧ d_months = 7 →
  b_payment = 1260 / 683 * 198 := by
sorry

#eval (1260 : ℚ) / 683 * 198

end NUMINAMATH_CALUDE_b_payment_l1259_125943


namespace NUMINAMATH_CALUDE_iphone_cost_l1259_125984

/-- The cost of the new iPhone given trade-in value, weekly earnings, and work duration -/
theorem iphone_cost (trade_in_value : ℕ) (weekly_earnings : ℕ) (work_weeks : ℕ) : 
  trade_in_value = 240 → weekly_earnings = 80 → work_weeks = 7 →
  trade_in_value + weekly_earnings * work_weeks = 800 := by
  sorry

end NUMINAMATH_CALUDE_iphone_cost_l1259_125984


namespace NUMINAMATH_CALUDE_figure_area_l1259_125914

/-- The total area of a figure composed of five rectangles -/
def total_area (a b c d e f g h i j : ℕ) : ℕ :=
  a * b + c * d + e * f + g * h + i * j

theorem figure_area : 
  total_area 7 4 5 4 7 3 5 2 3 1 = 82 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_l1259_125914


namespace NUMINAMATH_CALUDE_expression_evaluation_l1259_125956

theorem expression_evaluation : 
  let a := (1/4 + 1/12 - 7/18 - 1/36)
  let b := 1/36
  (b / a + a / b) = -10/3 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1259_125956


namespace NUMINAMATH_CALUDE_remainder_nine_333_mod_50_l1259_125994

theorem remainder_nine_333_mod_50 : 9^333 % 50 = 29 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nine_333_mod_50_l1259_125994


namespace NUMINAMATH_CALUDE_necklace_cuts_l1259_125945

theorem necklace_cuts (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 8) :
  Nat.choose n k = 145422675 :=
by sorry

end NUMINAMATH_CALUDE_necklace_cuts_l1259_125945


namespace NUMINAMATH_CALUDE_vincent_book_expenditure_l1259_125900

/-- The total amount Vincent spent on books -/
def total_spent (animal_books train_books space_books book_price : ℕ) : ℕ :=
  (animal_books + train_books + space_books) * book_price

/-- Theorem stating that Vincent spent $224 on books -/
theorem vincent_book_expenditure :
  total_spent 10 3 1 16 = 224 := by
  sorry

end NUMINAMATH_CALUDE_vincent_book_expenditure_l1259_125900


namespace NUMINAMATH_CALUDE_twelve_solutions_for_quadratic_diophantine_l1259_125918

theorem twelve_solutions_for_quadratic_diophantine (n : ℕ) (x y : ℕ+) 
  (h1 : x^2 - x*y + y^2 = n)
  (h2 : x ≠ y)
  (h3 : x ≠ 2*y)
  (h4 : y ≠ 2*x) :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S → p.1^2 - p.1*p.2 + p.2^2 = n) ∧ S.card ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_twelve_solutions_for_quadratic_diophantine_l1259_125918


namespace NUMINAMATH_CALUDE_stock_percentage_sold_l1259_125972

/-- 
Given:
- cash_realized: The cash realized on selling the stock
- brokerage_rate: The brokerage rate as a percentage
- total_amount: The total amount including brokerage

Prove that the percentage of stock sold is equal to 
(cash_realized / (total_amount - total_amount * brokerage_rate / 100)) * 100
-/
theorem stock_percentage_sold 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (total_amount : ℝ) 
  (h1 : cash_realized = 104.25)
  (h2 : brokerage_rate = 0.25)
  (h3 : total_amount = 104) :
  (cash_realized / (total_amount - total_amount * brokerage_rate / 100)) * 100 = 
    (104.25 / (104 - 104 * 0.25 / 100)) * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_sold_l1259_125972


namespace NUMINAMATH_CALUDE_system_equation_solution_l1259_125919

theorem system_equation_solution (x y : ℝ) 
  (eq1 : 7 * x + y = 19) 
  (eq2 : x + 3 * y = 1) : 
  2 * x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l1259_125919


namespace NUMINAMATH_CALUDE_at_most_one_obtuse_l1259_125965

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := 90 < angle

-- Theorem statement
theorem at_most_one_obtuse (t : Triangle) : 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle2) ∧ 
  ¬(is_obtuse t.angle1 ∧ is_obtuse t.angle3) ∧ 
  ¬(is_obtuse t.angle2 ∧ is_obtuse t.angle3) :=
sorry

end NUMINAMATH_CALUDE_at_most_one_obtuse_l1259_125965


namespace NUMINAMATH_CALUDE_gumball_probability_l1259_125924

theorem gumball_probability (blue_prob : ℚ) : 
  blue_prob ^ 2 = 16 / 49 → (1 : ℚ) - blue_prob = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l1259_125924


namespace NUMINAMATH_CALUDE_lanas_initial_pages_l1259_125921

theorem lanas_initial_pages (x : ℕ) : 
  x + (42 / 2) = 29 → x = 8 := by sorry

end NUMINAMATH_CALUDE_lanas_initial_pages_l1259_125921


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l1259_125930

/-- The number of pairs of shoes in the box -/
def num_pairs : ℕ := 9

/-- The total number of shoes in the box -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of ways to select 2 shoes from the total -/
def total_selections : ℕ := total_shoes * (total_shoes - 1) / 2

/-- The probability of selecting a matching pair of shoes -/
def prob_matching_pair : ℚ := num_pairs / total_selections

theorem matching_shoes_probability :
  prob_matching_pair = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l1259_125930


namespace NUMINAMATH_CALUDE_complement_union_theorem_complement_intersect_theorem_l1259_125990

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- State the theorems to be proved
theorem complement_union_theorem : 
  (Set.univ \ (A ∪ B)) = {x : ℝ | x ≤ 2 ∨ x ≥ 10} := by sorry

theorem complement_intersect_theorem :
  ((Set.univ \ A) ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_complement_intersect_theorem_l1259_125990


namespace NUMINAMATH_CALUDE_problem_solution_l1259_125941

theorem problem_solution (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49/(x - 3)^2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1259_125941


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1259_125916

theorem simplify_and_evaluate (x y : ℝ) (h1 : x = -1) (h2 : y = 2) :
  ((x + y)^2 - (x + 2*y)*(x - 2*y)) / (2*y) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1259_125916


namespace NUMINAMATH_CALUDE_original_number_is_five_l1259_125907

theorem original_number_is_five : 
  ∃ x : ℚ, 3 * (2 * x + 9) = 57 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_five_l1259_125907


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_squares_l1259_125958

theorem geometric_arithmetic_sequence_sum_squares (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : y^2 = (x^2 + z^2)/2)    -- Arithmetic sequence condition
  : x^2 + z^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_sum_squares_l1259_125958


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l1259_125922

/-- Represents the number of volunteers in each grade --/
structure GradeVolunteers where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the number of volunteers selected in the sample from each grade --/
structure SampleVolunteers where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the probability of selecting two volunteers from the same grade --/
def probability_same_grade (sample : SampleVolunteers) : ℚ :=
  let total_pairs := sample.first.choose 2 + sample.second.choose 2
  let all_pairs := (sample.first + sample.second).choose 2
  total_pairs / all_pairs

theorem stratified_sampling_theorem (volunteers : GradeVolunteers) (sample : SampleVolunteers) :
  volunteers.first = 36 →
  volunteers.second = 72 →
  volunteers.third = 54 →
  sample.third = 3 →
  sample.first = 2 ∧
  sample.second = 4 ∧
  probability_same_grade sample = 7/15 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l1259_125922


namespace NUMINAMATH_CALUDE_gcd_of_multiple_4500_l1259_125981

theorem gcd_of_multiple_4500 (k : ℤ) : 
  let b : ℤ := 4500 * k
  Int.gcd (b^2 + 11*b + 40) (b + 8) = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_multiple_4500_l1259_125981


namespace NUMINAMATH_CALUDE_water_tank_capacity_l1259_125979

theorem water_tank_capacity (c : ℝ) (h1 : c > 0) : 
  (c / 4 : ℝ) / c = 1 / 4 ∧ 
  ((c / 4 + 5) : ℝ) / c = 1 / 3 → 
  c = 60 := by
sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l1259_125979


namespace NUMINAMATH_CALUDE_heaviest_person_l1259_125936

def weight_problem (A D T V M : ℕ) : Prop :=
  A + D = 82 ∧
  D + T = 74 ∧
  T + V = 75 ∧
  V + M = 65 ∧
  M + A = 62

theorem heaviest_person (A D T V M : ℕ) 
  (h : weight_problem A D T V M) : 
  V = 43 ∧ V ≥ A ∧ V ≥ D ∧ V ≥ T ∧ V ≥ M :=
by
  sorry

#check heaviest_person

end NUMINAMATH_CALUDE_heaviest_person_l1259_125936


namespace NUMINAMATH_CALUDE_parabola_directrix_l1259_125939

/-- Given a parabola y = -3x^2 + 6x - 5, prove that its directrix is y = -23/12 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 6 * x - 5
  ∃ k : ℝ, k = -23/12 ∧ ∀ x y : ℝ, f x = y →
    ∃ h : ℝ, h > 0 ∧ (x - 1)^2 + (y + 2 - k)^2 = (y + 2 - (k + h))^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1259_125939


namespace NUMINAMATH_CALUDE_constant_polar_angle_forms_cone_l1259_125928

-- Define spherical coordinates
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

-- Define the set of points satisfying φ = c
def ConstantPolarAngleSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

-- Statement: The set of points with constant polar angle forms a cone
theorem constant_polar_angle_forms_cone (c : ℝ) :
  ∃ (cone : Set SphericalCoord), ConstantPolarAngleSet c = cone :=
sorry

end NUMINAMATH_CALUDE_constant_polar_angle_forms_cone_l1259_125928


namespace NUMINAMATH_CALUDE_combined_platform_length_l1259_125925

/-- The combined length of two train platforms -/
theorem combined_platform_length
  (length_train_a : ℝ)
  (time_platform_a : ℝ)
  (time_pole_a : ℝ)
  (length_train_b : ℝ)
  (time_platform_b : ℝ)
  (time_pole_b : ℝ)
  (h1 : length_train_a = 500)
  (h2 : time_platform_a = 75)
  (h3 : time_pole_a = 25)
  (h4 : length_train_b = 400)
  (h5 : time_platform_b = 60)
  (h6 : time_pole_b = 20) :
  (length_train_a + (length_train_a / time_pole_a) * time_platform_a - length_train_a) +
  (length_train_b + (length_train_b / time_pole_b) * time_platform_b - length_train_b) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_combined_platform_length_l1259_125925


namespace NUMINAMATH_CALUDE_number_problem_l1259_125962

theorem number_problem (x : ℝ) : (1/4 : ℝ) * x = (1/5 : ℝ) * (x + 1) + 1 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1259_125962


namespace NUMINAMATH_CALUDE_total_legs_is_71_l1259_125995

/-- Represents the total number of legs in a room with various furniture items -/
def total_legs : ℝ :=
  -- 4 tables with 4 legs each
  4 * 4 +
  -- 1 sofa with 4 legs
  1 * 4 +
  -- 2 chairs with 4 legs each
  2 * 4 +
  -- 3 tables with 3 legs each
  3 * 3 +
  -- 1 table with a single leg
  1 * 1 +
  -- 1 rocking chair with 2 legs
  1 * 2 +
  -- 1 bench with 6 legs
  1 * 6 +
  -- 2 stools with 3 legs each
  2 * 3 +
  -- 2 wardrobes, one with 4 legs and one with 3 legs
  (1 * 4 + 1 * 3) +
  -- 1 three-legged ecko
  1 * 3 +
  -- 1 antique table with 3 remaining legs
  1 * 3 +
  -- 1 damaged 4-legged table with only 3.5 legs remaining
  1 * 3.5 +
  -- 1 stool that lost half a leg
  1 * 2.5

/-- Theorem stating that the total number of legs in the room is 71 -/
theorem total_legs_is_71 : total_legs = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_is_71_l1259_125995


namespace NUMINAMATH_CALUDE_rectangle_area_l1259_125967

/-- A rectangle with diagonal length x and length three times its width has area (3/10) * x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : ∃ w : ℝ, 
  w > 0 ∧ 
  w^2 + (3*w)^2 = x^2 ∧ 
  w * (3*w) = (3/10) * x^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1259_125967


namespace NUMINAMATH_CALUDE_max_min_difference_c_l1259_125954

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 20) : 
  ∃ (c_max c_min : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, y + z + x = 3 ∧ y^2 + z^2 + x^2 = 20) → x ≤ c_max ∧ x ≥ c_min) ∧ 
    c_max - c_min = 2 * Real.sqrt 34 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l1259_125954


namespace NUMINAMATH_CALUDE_flag_design_count_l1259_125933

/-- Represents the number of available colors for the flag stripes -/
def num_colors : ℕ := 3

/-- Represents the number of stripes in the flag -/
def num_stripes : ℕ := 3

/-- Calculates the number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem: The number of unique three-stripe flags that can be created
    using three colors, where adjacent stripes may be the same color, is 27 -/
theorem flag_design_count :
  num_flag_designs = 27 := by sorry

end NUMINAMATH_CALUDE_flag_design_count_l1259_125933


namespace NUMINAMATH_CALUDE_course_selection_schemes_l1259_125904

theorem course_selection_schemes (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 2) :
  (Nat.choose (n - k) m) + (k * Nat.choose (n - k) (m - 1)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l1259_125904


namespace NUMINAMATH_CALUDE_slope_of_line_l1259_125917

/-- The slope of a line given by the equation (x/4) - (y/3) = -2 is -3/4 -/
theorem slope_of_line (x y : ℝ) : (x / 4 - y / 3 = -2) → (y = (-3 / 4) * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l1259_125917


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1259_125910

theorem imaginary_part_of_complex_division (i : ℂ) : i * i = -1 → 
  Complex.im ((4 - 3 * i) / i) = -4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l1259_125910


namespace NUMINAMATH_CALUDE_tangent_line_m_squared_l1259_125909

/-- A line that intersects an ellipse and a circle exactly once -/
structure TangentLine where
  m : ℝ
  -- Line equation: y = mx + 2
  line : ℝ → ℝ := fun x => m * x + 2
  -- Ellipse equation: x^2 + 9y^2 = 9
  ellipse : ℝ × ℝ → Prop := fun (x, y) => x^2 + 9 * y^2 = 9
  -- Circle equation: x^2 + y^2 = 4
  circle : ℝ × ℝ → Prop := fun (x, y) => x^2 + y^2 = 4
  -- The line intersects both the ellipse and the circle exactly once
  h_tangent_ellipse : ∃! x, ellipse (x, line x)
  h_tangent_circle : ∃! x, circle (x, line x)

/-- The theorem stating that m^2 = 1/3 for a line tangent to both the ellipse and circle -/
theorem tangent_line_m_squared (l : TangentLine) : l.m^2 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_m_squared_l1259_125909


namespace NUMINAMATH_CALUDE_justin_sabrina_pencils_l1259_125985

/-- Given that Justin and Sabrina have 50 pencils combined, Justin has 8 more pencils than m times 
    Sabrina's pencils, and Sabrina has 14 pencils, prove that m = 2. -/
theorem justin_sabrina_pencils (total : ℕ) (justin_extra : ℕ) (sabrina_pencils : ℕ) (m : ℕ) 
  (h1 : total = 50)
  (h2 : justin_extra = 8)
  (h3 : sabrina_pencils = 14)
  (h4 : total = (m * sabrina_pencils + justin_extra) + sabrina_pencils) :
  m = 2 := by sorry

end NUMINAMATH_CALUDE_justin_sabrina_pencils_l1259_125985


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1259_125937

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1259_125937


namespace NUMINAMATH_CALUDE_function_properties_l1259_125951

noncomputable def f (x : ℝ) := Real.cos (2 * x + 2 * Real.pi / 3) + 2 * (Real.cos x) ^ 2

theorem function_properties :
  (∀ x, f x ≤ 2) ∧
  (∀ k : ℤ, f (k * Real.pi - Real.pi / 6) = 2) ∧
  (∀ A B C a b c : ℝ,
    0 < A ∧ A < Real.pi →
    0 < B ∧ B < Real.pi →
    0 < C ∧ C < Real.pi →
    A + B + C = Real.pi →
    a > 0 ∧ b > 0 ∧ c > 0 →
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →
    f A = 3 / 2 →
    b + c = 2 →
    a ≥ Real.sqrt 3) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_function_properties_l1259_125951


namespace NUMINAMATH_CALUDE_base4_1212_is_102_l1259_125970

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_1212_is_102 : base4_to_decimal [2, 1, 2, 1] = 102 := by
  sorry

end NUMINAMATH_CALUDE_base4_1212_is_102_l1259_125970


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l1259_125952

theorem sum_of_squares_inequality (a₁ a₂ a₃ a : ℝ) 
  (sum_condition : a₁ + a₂ + a₃ = 1)
  (lower_bound₁ : a₁ ≥ a)
  (lower_bound₂ : a₂ ≥ a)
  (lower_bound₃ : a₃ ≥ a) :
  a₁^2 + a₂^2 + a₃^2 ≤ 2*a^2 + (2*a - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l1259_125952


namespace NUMINAMATH_CALUDE_lists_with_high_number_l1259_125920

def total_balls : ℕ := 15
def draws : ℕ := 4
def threshold : ℕ := 10

theorem lists_with_high_number (total_balls draws threshold : ℕ) :
  total_balls = 15 ∧ draws = 4 ∧ threshold = 10 →
  (total_balls ^ draws) - (threshold ^ draws) = 40625 := by
  sorry

end NUMINAMATH_CALUDE_lists_with_high_number_l1259_125920


namespace NUMINAMATH_CALUDE_candy_bar_cost_l1259_125971

theorem candy_bar_cost (initial_amount : ℕ) (change : ℕ) (cost : ℕ) : 
  initial_amount = 50 →
  change = 5 →
  cost = initial_amount - change →
  cost = 45 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l1259_125971


namespace NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l1259_125991

theorem smallest_distance_between_points_on_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 5*I)) = 2)
  (hw : Complex.abs (w - (-3 + 4*I)) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 106 - 6 ∧ 
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 - 5*I)) = 2 → 
      Complex.abs (w' - (-3 + 4*I)) = 4 → 
      Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_points_on_circles_l1259_125991


namespace NUMINAMATH_CALUDE_blue_face_area_l1259_125999

-- Define a tetrahedron with right-angled edges
structure RightAngledTetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  red_area : ℝ
  yellow_area : ℝ
  green_area : ℝ
  blue_area : ℝ
  right_angle_condition : a^2 + b^2 = c^2
  red_area_condition : red_area = (1/2) * a * b
  yellow_area_condition : yellow_area = (1/2) * b * c
  green_area_condition : green_area = (1/2) * c * a
  blue_area_condition : blue_area = (1/2) * (a^2 + b^2 + c^2)

-- Theorem statement
theorem blue_face_area (t : RightAngledTetrahedron) 
  (h1 : t.red_area = 60) 
  (h2 : t.yellow_area = 20) 
  (h3 : t.green_area = 15) : 
  t.blue_area = 65 := by
  sorry


end NUMINAMATH_CALUDE_blue_face_area_l1259_125999


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1259_125982

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (1 : ℚ) / (3 ^ n) * 10^n % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l1259_125982


namespace NUMINAMATH_CALUDE_theta_range_l1259_125906

theorem theta_range (θ : Real) : 
  (∀ x : Real, x ∈ Set.Icc 0 1 → x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) →
  π / 12 < θ ∧ θ < 5 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_theta_range_l1259_125906


namespace NUMINAMATH_CALUDE_f_max_min_difference_l1259_125986

noncomputable def f (x : ℝ) : ℝ := x * |3 - x| - (x - 3) * |x|

theorem f_max_min_difference :
  ∃ (max min : ℝ), (∀ x, f x ≤ max) ∧ (∀ x, f x ≥ min) ∧ (max - min = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l1259_125986


namespace NUMINAMATH_CALUDE_friend_fruit_consumption_l1259_125923

/-- Given three friends who ate a total of 128 ounces of fruit, 
    where one friend ate 8 ounces and another ate 96 ounces,
    prove that the third friend ate 24 ounces. -/
theorem friend_fruit_consumption 
  (total : ℕ) 
  (friend1 : ℕ) 
  (friend2 : ℕ) 
  (h1 : total = 128)
  (h2 : friend1 = 8)
  (h3 : friend2 = 96) :
  total - friend1 - friend2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_friend_fruit_consumption_l1259_125923


namespace NUMINAMATH_CALUDE_carrots_equal_fifteen_l1259_125901

/-- The price relationship between apples, bananas, and carrots -/
structure FruitPrices where
  apple_banana_ratio : ℚ
  banana_carrot_ratio : ℚ
  apple_banana_eq : apple_banana_ratio = 10 / 5
  banana_carrot_eq : banana_carrot_ratio = 2 / 5

/-- The number of carrots that can be bought for the price of 12 apples -/
def carrots_for_apples (prices : FruitPrices) : ℚ :=
  12 * (prices.banana_carrot_ratio / prices.apple_banana_ratio)

theorem carrots_equal_fifteen (prices : FruitPrices) :
  carrots_for_apples prices = 15 := by
  sorry

end NUMINAMATH_CALUDE_carrots_equal_fifteen_l1259_125901


namespace NUMINAMATH_CALUDE_polygon_sides_l1259_125997

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1259_125997
