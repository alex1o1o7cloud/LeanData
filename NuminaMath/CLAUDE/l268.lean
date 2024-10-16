import Mathlib

namespace NUMINAMATH_CALUDE_riverview_village_l268_26802

theorem riverview_village (p h s c d : ℕ) : 
  p = 4 * h → 
  s = 5 * c → 
  d = 4 * p → 
  ¬∃ (h c : ℕ), 52 = 21 * h + 6 * c :=
by sorry

end NUMINAMATH_CALUDE_riverview_village_l268_26802


namespace NUMINAMATH_CALUDE_straight_angle_average_l268_26860

theorem straight_angle_average (p q r s t : ℝ) : 
  p + q + r + s + t = 180 → (p + q + r + s + t) / 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_straight_angle_average_l268_26860


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l268_26894

theorem multiplication_addition_equality : 25 * 13 * 2 + 15 * 13 * 7 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l268_26894


namespace NUMINAMATH_CALUDE_quadratic_one_zero_l268_26811

/-- If a quadratic function f(x) = mx^2 - 2x + 3 has only one zero, then m = 0 or m = 1/3 -/
theorem quadratic_one_zero (m : ℝ) : 
  (∃! x, m * x^2 - 2 * x + 3 = 0) → (m = 0 ∨ m = 1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_zero_l268_26811


namespace NUMINAMATH_CALUDE_volleyball_tournament_triples_l268_26817

/-- Represents a round-robin volleyball tournament -/
structure Tournament :=
  (num_teams : ℕ)
  (wins_per_team : ℕ)

/-- Represents the number of triples where each team wins once against the others -/
def count_special_triples (t : Tournament) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem volleyball_tournament_triples (t : Tournament) 
  (h1 : t.num_teams = 15)
  (h2 : t.wins_per_team = 7) :
  count_special_triples t = 140 :=
sorry

end NUMINAMATH_CALUDE_volleyball_tournament_triples_l268_26817


namespace NUMINAMATH_CALUDE_tangent_circle_condition_l268_26845

/-- The line 2x + y - 2 = 0 is tangent to the circle (x - 1)^2 + (y - a)^2 = 1 -/
def is_tangent (a : ℝ) : Prop :=
  ∃ (x y : ℝ), (2 * x + y - 2 = 0) ∧ ((x - 1)^2 + (y - a)^2 = 1)

/-- If the line is tangent to the circle, then a = ± √5 -/
theorem tangent_circle_condition (a : ℝ) :
  is_tangent a → (a = Real.sqrt 5 ∨ a = -Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_tangent_circle_condition_l268_26845


namespace NUMINAMATH_CALUDE_quadratic_roots_greater_than_two_l268_26844

theorem quadratic_roots_greater_than_two (m : ℝ) : 
  (∀ x : ℝ, x^2 + (m-1)*x + 4 - m = 0 → x > 2) ↔ -6 < m ∧ m ≤ -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_greater_than_two_l268_26844


namespace NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l268_26853

def factorial (n : ℕ) : ℕ := sorry

def sum_factorials (n : ℕ) : ℕ := sorry

def units_digit (n : ℕ) : ℕ := sorry

theorem units_digit_of_factorial_sum :
  units_digit (sum_factorials 15) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_factorial_sum_l268_26853


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l268_26865

/-- Calculates the average speed of a cyclist's trip given two segments with different speeds -/
theorem cyclist_average_speed (d1 d2 v1 v2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 11) (h3 : v1 = 11) (h4 : v2 = 8) :
  (d1 + d2) / ((d1 / v1) + (d2 / v2)) = 1664 / 185 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_average_speed_l268_26865


namespace NUMINAMATH_CALUDE_roses_count_l268_26857

def total_roses : ℕ := 500

def red_roses : ℕ := (total_roses * 5) / 8

def remaining_after_red : ℕ := total_roses - red_roses

def yellow_roses : ℕ := remaining_after_red / 8

def pink_roses : ℕ := (remaining_after_red * 2) / 8

def remaining_after_yellow_pink : ℕ := remaining_after_red - yellow_roses - pink_roses

def white_roses : ℕ := remaining_after_yellow_pink / 2

def purple_roses : ℕ := remaining_after_yellow_pink / 2

theorem roses_count : red_roses + white_roses + purple_roses = 430 := by
  sorry

end NUMINAMATH_CALUDE_roses_count_l268_26857


namespace NUMINAMATH_CALUDE_prime_triplet_equation_l268_26831

theorem prime_triplet_equation : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p^q + q^p = r → 
    ((p = 2 ∧ q = 3 ∧ r = 17) ∨ (p = 3 ∧ q = 2 ∧ r = 17)) := by
  sorry

end NUMINAMATH_CALUDE_prime_triplet_equation_l268_26831


namespace NUMINAMATH_CALUDE_third_plane_passenger_count_l268_26861

/-- The number of passengers on the third plane -/
def third_plane_passengers : ℕ := 40

/-- The speed of an empty plane in MPH -/
def empty_plane_speed : ℕ := 600

/-- The speed reduction per passenger in MPH -/
def speed_reduction_per_passenger : ℕ := 2

/-- The number of passengers on the first plane -/
def first_plane_passengers : ℕ := 50

/-- The number of passengers on the second plane -/
def second_plane_passengers : ℕ := 60

/-- The average speed of the three planes in MPH -/
def average_speed : ℕ := 500

theorem third_plane_passenger_count :
  (empty_plane_speed - speed_reduction_per_passenger * first_plane_passengers +
   empty_plane_speed - speed_reduction_per_passenger * second_plane_passengers +
   empty_plane_speed - speed_reduction_per_passenger * third_plane_passengers) / 3 = average_speed :=
by sorry

end NUMINAMATH_CALUDE_third_plane_passenger_count_l268_26861


namespace NUMINAMATH_CALUDE_fiftyFourthCardIsSpadeTwo_l268_26881

/-- Represents a playing card suit -/
inductive Suit
| Spades
| Hearts

/-- Represents a playing card value -/
inductive Value
| Ace | Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King

/-- Represents a playing card -/
structure Card where
  suit : Suit
  value : Value

/-- The sequence of cards in order -/
def cardSequence : List Card := sorry

/-- The length of one complete cycle in the sequence -/
def cycleLength : Nat := 26

/-- Function to get the nth card in the sequence -/
def getNthCard (n : Nat) : Card := sorry

theorem fiftyFourthCardIsSpadeTwo : 
  getNthCard 54 = Card.mk Suit.Spades Value.Two := by sorry

end NUMINAMATH_CALUDE_fiftyFourthCardIsSpadeTwo_l268_26881


namespace NUMINAMATH_CALUDE_percentage_difference_l268_26807

theorem percentage_difference : (0.9 * 40) - (0.8 * 30) = 12 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l268_26807


namespace NUMINAMATH_CALUDE_race_time_difference_l268_26815

-- Define the race participants
structure Racer where
  name : String
  time : ℕ

-- Define the race conditions
def patrick : Racer := { name := "Patrick", time := 60 }
def amy : Racer := { name := "Amy", time := 36 }

-- Define Manu's time in terms of Amy's
def manu_time (amy : Racer) : ℕ := 2 * amy.time

-- Define the theorem
theorem race_time_difference (amy : Racer) (h : amy.time = 36) : 
  manu_time amy - patrick.time = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l268_26815


namespace NUMINAMATH_CALUDE_bug_return_probability_l268_26823

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => (2 : ℚ) / 3 * P n

/-- The probability of returning to the starting vertex on the 10th move -/
def probability_10th_move : ℚ := P 10

theorem bug_return_probability :
  probability_10th_move = 1024 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l268_26823


namespace NUMINAMATH_CALUDE_program_output_is_one_l268_26803

/-- Represents the state of the program -/
structure ProgramState :=
  (S : ℕ)
  (n : ℕ)

/-- The update function for the program state -/
def updateState (state : ProgramState) : ProgramState :=
  if state.n > 1 then
    { S := state.S + state.n, n := state.n - 1 }
  else
    state

/-- The termination condition for the program -/
def isTerminated (state : ProgramState) : Prop :=
  state.S ≥ 17 ∧ state.n ≤ 1

/-- The initial state of the program -/
def initialState : ProgramState :=
  { S := 0, n := 5 }

/-- The theorem stating that the program terminates with n = 1 -/
theorem program_output_is_one :
  ∃ (finalState : ProgramState), 
    (∃ (k : ℕ), finalState = (updateState^[k] initialState)) ∧
    isTerminated finalState ∧
    finalState.n = 1 := by
  sorry

end NUMINAMATH_CALUDE_program_output_is_one_l268_26803


namespace NUMINAMATH_CALUDE_sum_of_absolute_roots_l268_26826

theorem sum_of_absolute_roots (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℝ, x^3 - 2011*x + m = (x - a) * (x - b) * (x - c)) →
  |a| + |b| + |c| = 98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_absolute_roots_l268_26826


namespace NUMINAMATH_CALUDE_square_ratio_sum_l268_26895

theorem square_ratio_sum (p q r : ℕ) : 
  (75 : ℚ) / 128 = (p * Real.sqrt q / r) ^ 2 → p + q + r = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l268_26895


namespace NUMINAMATH_CALUDE_double_xy_doubles_fraction_l268_26896

/-- Given a fraction xy/(2x+y), prove that doubling both x and y results in doubling the fraction -/
theorem double_xy_doubles_fraction (x y : ℝ) (h : 2 * x + y ≠ 0) :
  (2 * x * 2 * y) / (2 * (2 * x) + 2 * y) = 2 * (x * y / (2 * x + y)) := by
  sorry

end NUMINAMATH_CALUDE_double_xy_doubles_fraction_l268_26896


namespace NUMINAMATH_CALUDE_sequence_sum_exp_l268_26818

/-- Given a sequence {a_n} where the sum of the first n terms is S_n = ln(1 + 1/n),
    prove that e^(a_7 + a_8 + a_9) = 20/21 -/
theorem sequence_sum_exp (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, S n = Real.log (1 + 1 / n)) :
  Real.exp (a 7 + a 8 + a 9) = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_exp_l268_26818


namespace NUMINAMATH_CALUDE_eleven_percent_of_700_is_77_l268_26852

theorem eleven_percent_of_700_is_77 : (11 / 100) * 700 = 77 := by
  sorry

end NUMINAMATH_CALUDE_eleven_percent_of_700_is_77_l268_26852


namespace NUMINAMATH_CALUDE_integral_curves_of_differential_equation_l268_26867

/-- The differential equation -/
def differential_equation (x y : ℝ) (dx dy : ℝ) : Prop :=
  6 * x * dx - 6 * y * dy = 2 * x^2 * y * dy - 3 * x * y^2 * dx

/-- The integral curve equation -/
def integral_curve (x y : ℝ) (C : ℝ) : Prop :=
  (x^2 + 3)^3 / (2 + y^2) = C

/-- Theorem stating that the integral curves of the given differential equation
    are described by the integral_curve equation -/
theorem integral_curves_of_differential_equation :
  ∀ (x y : ℝ) (C : ℝ),
  (∀ (dx dy : ℝ), differential_equation x y dx dy) →
  ∃ (C : ℝ), integral_curve x y C :=
sorry

end NUMINAMATH_CALUDE_integral_curves_of_differential_equation_l268_26867


namespace NUMINAMATH_CALUDE_marias_age_l268_26827

theorem marias_age (cooper dante maria : ℕ) 
  (sum_ages : cooper + dante + maria = 31)
  (dante_twice_cooper : dante = 2 * cooper)
  (dante_younger_maria : dante = maria - 1) : 
  maria = 13 := by
  sorry

end NUMINAMATH_CALUDE_marias_age_l268_26827


namespace NUMINAMATH_CALUDE_unique_digits_product_rounding_l268_26808

theorem unique_digits_product_rounding : ∃! (A B C : ℕ), 
  (A < 10 ∧ B < 10 ∧ C < 10) ∧  -- digits are less than 10
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧     -- digits are distinct
  (⌊((10 * A + B : ℝ) + 0.1 * C) * C + 0.5⌋ = 10 * B + C) ∧ -- main equation
  A = 1 ∧ B = 4 ∧ C = 3 :=
by sorry


end NUMINAMATH_CALUDE_unique_digits_product_rounding_l268_26808


namespace NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l268_26884

def is_valid_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ, is_valid_abcba n → n ≤ 96769 ∧ 96769 % 13 = 0 ∧ is_valid_abcba 96769 :=
sorry

end NUMINAMATH_CALUDE_greatest_abcba_divisible_by_13_l268_26884


namespace NUMINAMATH_CALUDE_promotion_difference_l268_26871

/-- Calculates the total cost of two pairs of shoes using Promotion A -/
def costPromotionA (price : ℝ) : ℝ :=
  price + price * 0.4

/-- Calculates the total cost of two pairs of shoes using Promotion B -/
def costPromotionB (price : ℝ) : ℝ :=
  price + (price - 15)

/-- Proves that the difference between Promotion B and Promotion A is $15 -/
theorem promotion_difference (shoe_price : ℝ) (h : shoe_price = 50) :
  costPromotionB shoe_price - costPromotionA shoe_price = 15 := by
  sorry

#eval costPromotionB 50 - costPromotionA 50

end NUMINAMATH_CALUDE_promotion_difference_l268_26871


namespace NUMINAMATH_CALUDE_power_product_equals_l268_26883

theorem power_product_equals : 3^5 * 4^5 = 248832 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_l268_26883


namespace NUMINAMATH_CALUDE_number_of_classes_for_histogram_l268_26885

theorem number_of_classes_for_histogram (tallest_height shortest_height class_interval : ℝ)
  (h1 : tallest_height = 186)
  (h2 : shortest_height = 154)
  (h3 : class_interval = 5)
  : Int.ceil ((tallest_height - shortest_height) / class_interval) = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_classes_for_histogram_l268_26885


namespace NUMINAMATH_CALUDE_tommy_order_cost_and_percentages_l268_26836

/-- Represents the weight of each fruit in kilograms -/
structure FruitOrder where
  apples : ℝ
  oranges : ℝ
  grapes : ℝ
  strawberries : ℝ
  bananas : ℝ
  pineapples : ℝ

/-- Represents the price of each fruit per kilogram -/
structure FruitPrices where
  apples : ℝ
  oranges : ℝ
  grapes : ℝ
  strawberries : ℝ
  bananas : ℝ
  pineapples : ℝ

def totalWeight (order : FruitOrder) : ℝ :=
  order.apples + order.oranges + order.grapes + order.strawberries + order.bananas + order.pineapples

def totalCost (order : FruitOrder) (prices : FruitPrices) : ℝ :=
  order.apples * prices.apples +
  order.oranges * prices.oranges +
  order.grapes * prices.grapes +
  order.strawberries * prices.strawberries +
  order.bananas * prices.bananas +
  order.pineapples * prices.pineapples

theorem tommy_order_cost_and_percentages 
  (order : FruitOrder)
  (prices : FruitPrices)
  (h1 : totalWeight order = 20)
  (h2 : order.apples = 4)
  (h3 : order.oranges = 2)
  (h4 : order.grapes = 4)
  (h5 : order.strawberries = 3)
  (h6 : order.bananas = 1)
  (h7 : order.pineapples = 3)
  (h8 : prices.apples = 2)
  (h9 : prices.oranges = 3)
  (h10 : prices.grapes = 2.5)
  (h11 : prices.strawberries = 4)
  (h12 : prices.bananas = 1.5)
  (h13 : prices.pineapples = 3.5) :
  totalCost order prices = 48 ∧
  order.apples / totalWeight order = 0.2 ∧
  order.oranges / totalWeight order = 0.1 ∧
  order.grapes / totalWeight order = 0.2 ∧
  order.strawberries / totalWeight order = 0.15 ∧
  order.bananas / totalWeight order = 0.05 ∧
  order.pineapples / totalWeight order = 0.15 := by
  sorry


end NUMINAMATH_CALUDE_tommy_order_cost_and_percentages_l268_26836


namespace NUMINAMATH_CALUDE_largest_solution_floor_equation_l268_26824

theorem largest_solution_floor_equation :
  let floor_eq (x : ℝ) := ⌊x⌋ = 7 + 50 * (x - ⌊x⌋)
  ∃ (max_sol : ℝ), floor_eq max_sol ∧
    ∀ (y : ℝ), floor_eq y → y ≤ max_sol ∧
    max_sol = 2849 / 50
  := by sorry

end NUMINAMATH_CALUDE_largest_solution_floor_equation_l268_26824


namespace NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l268_26876

theorem perpendicular_vectors_tan_theta :
  ∀ θ : ℝ,
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (Real.sqrt 3, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  Real.tan θ = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_tan_theta_l268_26876


namespace NUMINAMATH_CALUDE_complex_imaginary_part_l268_26856

theorem complex_imaginary_part (z : ℂ) : (3 - 4*I) * z = Complex.abs (4 + 3*I) → Complex.im z = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_imaginary_part_l268_26856


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_6_range_of_m_for_f_geq_m_squared_minus_3m_l268_26830

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |2*x - 4|

-- Theorem for the solution set of f(x) < 6
theorem solution_set_f_less_than_6 :
  {x : ℝ | f x < 6} = {x : ℝ | 0 < x ∧ x < 8/3} := by sorry

-- Theorem for the range of m
theorem range_of_m_for_f_geq_m_squared_minus_3m :
  {m : ℝ | ∀ x, f x ≥ m^2 - 3*m} = {m : ℝ | -1 ≤ m ∧ m ≤ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_6_range_of_m_for_f_geq_m_squared_minus_3m_l268_26830


namespace NUMINAMATH_CALUDE_burger_calorie_content_l268_26851

/-- Represents the calorie content of a lunch meal -/
structure LunchMeal where
  burger_calories : ℕ
  carrot_stick_calories : ℕ
  cookie_calories : ℕ
  carrot_stick_count : ℕ
  cookie_count : ℕ
  total_calories : ℕ

/-- Theorem stating the calorie content of a burger in a specific lunch meal -/
theorem burger_calorie_content (meal : LunchMeal) 
  (h1 : meal.carrot_stick_calories = 20)
  (h2 : meal.cookie_calories = 50)
  (h3 : meal.carrot_stick_count = 5)
  (h4 : meal.cookie_count = 5)
  (h5 : meal.total_calories = 750) :
  meal.burger_calories = 400 := by
  sorry

end NUMINAMATH_CALUDE_burger_calorie_content_l268_26851


namespace NUMINAMATH_CALUDE_inequality_proof_l268_26854

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_condition : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l268_26854


namespace NUMINAMATH_CALUDE_inequality_solution_l268_26868

theorem inequality_solution (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 4) / (x - 2))| > 3 ↔ x < 5/3 ∨ (5/3 < x ∧ x < 2) ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l268_26868


namespace NUMINAMATH_CALUDE_function_divisibility_condition_l268_26812

theorem function_divisibility_condition (f : ℕ+ → ℕ+) :
  (∀ n m : ℕ+, (n + f m) ∣ (f n + n * f m)) →
  (∀ n : ℕ+, f n = n ^ 2 ∨ f n = 1) :=
by sorry

end NUMINAMATH_CALUDE_function_divisibility_condition_l268_26812


namespace NUMINAMATH_CALUDE_vector_problem_l268_26839

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : Vector2D) : Prop := dot v w = 0

/-- The angle between two vectors is acute if their dot product is positive -/
def acuteAngle (v w : Vector2D) : Prop := dot v w > 0

theorem vector_problem (x : ℝ) : 
  let a : Vector2D := ⟨1, 2⟩
  let b : Vector2D := ⟨x, 1⟩
  (acuteAngle a b ↔ x > -2 ∧ x ≠ 1/2) ∧ 
  (perpendicular (Vector2D.mk (1 + 2*x) 4) (Vector2D.mk (2 - x) 3) ↔ x = 7/2) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l268_26839


namespace NUMINAMATH_CALUDE_area_ratio_is_one_l268_26828

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define the right triangle PQR with given side lengths
def rightTrianglePQR : Triangle :=
  { P := (0, 15),
    Q := (0, 0),
    R := (20, 0) }

-- Define midpoints S and T
def S : ℝ × ℝ := (0, 7.5)
def T : ℝ × ℝ := (12.5, 12.5)

-- Define point Y as the intersection of RT and QS
def Y : ℝ × ℝ := sorry

-- Define the areas of quadrilateral PSYT and triangle QYR
def areaPSYT : ℝ := sorry
def areaQYR : ℝ := sorry

-- Theorem statement
theorem area_ratio_is_one :
  areaPSYT = areaQYR :=
sorry

end NUMINAMATH_CALUDE_area_ratio_is_one_l268_26828


namespace NUMINAMATH_CALUDE_stock_transaction_l268_26804

/-- Represents the number of shares for each stock --/
structure StockHoldings where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of a set of numbers --/
def range (s : StockHoldings) : ℕ :=
  max s.v (max s.w (max s.x (max s.y s.z))) - min s.v (min s.w (min s.x (min s.y s.z)))

/-- Theorem representing the stock transaction problem --/
theorem stock_transaction (initial : StockHoldings) 
  (h1 : initial.v = 68)
  (h2 : initial.w = 112)
  (h3 : initial.x = 56)
  (h4 : initial.y = 94)
  (h5 : initial.z = 45)
  (bought_y : ℕ)
  (h6 : bought_y = 23)
  (range_increase : ℕ)
  (h7 : range_increase = 14)
  : ∃ (sold_x : ℕ), 
    let final := StockHoldings.mk 
      initial.v 
      initial.w 
      (initial.x - sold_x)
      (initial.y + bought_y)
      initial.z
    range final = range initial + range_increase ∧ sold_x = 20 := by
  sorry


end NUMINAMATH_CALUDE_stock_transaction_l268_26804


namespace NUMINAMATH_CALUDE_student_group_aging_l268_26819

/-- Represents a group of students with their average age and age variance -/
structure StudentGroup where
  averageAge : ℝ
  ageVariance : ℝ

/-- Function to calculate the new state of a StudentGroup after a given time -/
def ageStudentGroup (group : StudentGroup) (years : ℝ) : StudentGroup :=
  { averageAge := group.averageAge + years
    ageVariance := group.ageVariance }

theorem student_group_aging :
  let initialGroup : StudentGroup := { averageAge := 13, ageVariance := 3 }
  let yearsLater : ℝ := 2
  let finalGroup := ageStudentGroup initialGroup yearsLater
  finalGroup.averageAge = 15 ∧ finalGroup.ageVariance = 3 := by
  sorry


end NUMINAMATH_CALUDE_student_group_aging_l268_26819


namespace NUMINAMATH_CALUDE_triangle_area_is_two_l268_26806

/-- A line in 2D space --/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The area of a triangle given by three points --/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- The intersection point of two lines --/
def lineIntersection (l1 l2 : Line) : ℝ × ℝ := sorry

/-- The main theorem --/
theorem triangle_area_is_two :
  let line1 : Line := { slope := 3/4, point := (3, 3) }
  let line2 : Line := { slope := -1, point := (3, 3) }
  let line3 : Line := { slope := -1, point := (0, 14) }
  let p1 := (3, 3)
  let p2 := lineIntersection line1 line3
  let p3 := lineIntersection line2 line3
  triangleArea p1 p2 p3 = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_two_l268_26806


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l268_26800

/-- Represents the number of fish in a pond -/
def total_fish : ℕ := 3200

/-- Represents the number of fish initially tagged -/
def tagged_fish : ℕ := 80

/-- Represents the number of fish in the second catch -/
def second_catch : ℕ := 80

/-- Calculates the expected number of tagged fish in the second catch -/
def expected_tagged_in_second_catch : ℚ :=
  (tagged_fish : ℚ) * (second_catch : ℚ) / (total_fish : ℚ)

theorem tagged_fish_in_second_catch :
  ⌊expected_tagged_in_second_catch⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l268_26800


namespace NUMINAMATH_CALUDE_x_ln_x_squared_necessary_not_sufficient_l268_26858

theorem x_ln_x_squared_necessary_not_sufficient (x : ℝ) (h1 : 1 < x) (h2 : x < Real.exp 1) :
  (∀ x, x * Real.log x < 1 → x * (Real.log x)^2 < 1) ∧
  (∃ x, x * (Real.log x)^2 < 1 ∧ x * Real.log x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_ln_x_squared_necessary_not_sufficient_l268_26858


namespace NUMINAMATH_CALUDE_bodhi_animal_transport_l268_26847

/-- Theorem: Mr. Bodhi's Animal Transport Weight Calculation --/
theorem bodhi_animal_transport (cows foxes : ℕ) : 
  let zebras := 3 * foxes
  let cow_weight := 3
  let fox_weight := 2
  let zebra_weight := 5
  let total_weight := cows * cow_weight + foxes * fox_weight + zebras * zebra_weight
  let required_weight := 300
  cows = 20 ∧ foxes = 15 → total_weight = required_weight + 15 := by
sorry

end NUMINAMATH_CALUDE_bodhi_animal_transport_l268_26847


namespace NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l268_26887

/-- A triangle with consecutive even integer side lengths. -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h_even : ∃ k : ℕ, a = 2*k ∧ b = 2*(k+1) ∧ c = 2*(k+2)
  h_triangle : a + b > c ∧ a + c > b ∧ b + c > a

/-- The perimeter of an EvenTriangle. -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The smallest possible perimeter of an EvenTriangle is 12. -/
theorem smallest_even_triangle_perimeter :
  ∃ t : EvenTriangle, perimeter t = 12 ∧ ∀ t' : EvenTriangle, perimeter t ≤ perimeter t' :=
sorry

end NUMINAMATH_CALUDE_smallest_even_triangle_perimeter_l268_26887


namespace NUMINAMATH_CALUDE_final_expression_l268_26832

theorem final_expression (y : ℝ) : 3 * (1/2 * (12*y + 3)) = 18*y + 4.5 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l268_26832


namespace NUMINAMATH_CALUDE_fraction_product_subtraction_l268_26848

theorem fraction_product_subtraction : (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_subtraction_l268_26848


namespace NUMINAMATH_CALUDE_shortest_side_range_l268_26805

/-- An obtuse triangle with sides x, x+1, and x+2 -/
structure ObtuseTriangle where
  x : ℝ
  is_obtuse : 0 < x ∧ x < x + 1 ∧ x + 1 < x + 2

/-- The range of the shortest side in an obtuse triangle -/
theorem shortest_side_range (t : ObtuseTriangle) : 1 < t.x ∧ t.x < 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_range_l268_26805


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l268_26855

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line = Line.mk 1 (-2) (-3) →
  point = Point.mk 3 1 →
  ∃ (result_line : Line),
    perpendicular given_line result_line ∧
    on_line point result_line ∧
    result_line = Line.mk 2 1 (-7) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l268_26855


namespace NUMINAMATH_CALUDE_circular_lid_area_l268_26820

/-- The area of a circular lid with diameter 2.75 inches is approximately 5.9375 square inches. -/
theorem circular_lid_area :
  let diameter : ℝ := 2.75
  let radius : ℝ := diameter / 2
  let area : ℝ := Real.pi * radius^2
  ∃ ε > 0, abs (area - 5.9375) < ε :=
by sorry

end NUMINAMATH_CALUDE_circular_lid_area_l268_26820


namespace NUMINAMATH_CALUDE_min_crossing_time_for_four_people_l268_26878

/-- Represents a person with their crossing time -/
structure Person where
  crossingTime : ℕ

/-- Represents the state of the bridge crossing problem -/
structure BridgeState where
  leftSide : List Person
  rightSide : List Person

/-- Calculates the minimum time required for all people to cross the bridge -/
def minCrossingTime (people : List Person) : ℕ :=
  sorry

/-- Theorem stating the minimum crossing time for the given problem -/
theorem min_crossing_time_for_four_people :
  let people := [
    { crossingTime := 2 },
    { crossingTime := 4 },
    { crossingTime := 6 },
    { crossingTime := 8 }
  ]
  minCrossingTime people = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_crossing_time_for_four_people_l268_26878


namespace NUMINAMATH_CALUDE_gcd_987654_876543_l268_26864

theorem gcd_987654_876543 : Nat.gcd 987654 876543 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_987654_876543_l268_26864


namespace NUMINAMATH_CALUDE_triangle_angle_relations_l268_26822

theorem triangle_angle_relations (a b c : ℝ) (α β γ : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ 
  (α + β + γ = Real.pi) ∧
  (c^2 = a^2 + 2 * b^2 * Real.cos β) →
  ((γ = β / 2 + Real.pi / 2 ∧ α = Real.pi / 2 - 3 * β / 2 ∧ 0 < β ∧ β < Real.pi / 3) ∨
   (α = β / 2 ∧ γ = Real.pi - 3 * β / 2 ∧ 0 < β ∧ β < 2 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_relations_l268_26822


namespace NUMINAMATH_CALUDE_green_peaches_count_l268_26892

/-- The number of green peaches in a basket -/
def num_green_peaches : ℕ := sorry

/-- The number of red peaches in the basket -/
def num_red_peaches : ℕ := 6

/-- The total number of red and green peaches in the basket -/
def total_red_green_peaches : ℕ := 22

/-- Theorem stating that the number of green peaches is 16 -/
theorem green_peaches_count : num_green_peaches = 16 := by sorry

end NUMINAMATH_CALUDE_green_peaches_count_l268_26892


namespace NUMINAMATH_CALUDE_smallest_possible_N_l268_26893

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a + b + c + d + e + f = 2520 ∧
  a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ d ≥ 5 ∧ e ≥ 5 ∧ f ≥ 5

def N (a b c d e f : ℕ) : ℕ :=
  max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f))))

theorem smallest_possible_N :
  ∀ a b c d e f : ℕ, is_valid_arrangement a b c d e f →
  N a b c d e f ≥ 506 ∧
  (∃ a' b' c' d' e' f' : ℕ, is_valid_arrangement a' b' c' d' e' f' ∧ N a' b' c' d' e' f' = 506) :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_N_l268_26893


namespace NUMINAMATH_CALUDE_janous_inequality_l268_26899

theorem janous_inequality (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) ∧
  ((1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) = 16 * (x / z + z / x + 2) ↔
   x = y ∧ y = z ∧ z = Real.sqrt (α / 3)) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l268_26899


namespace NUMINAMATH_CALUDE_complex_equation_solution_l268_26890

theorem complex_equation_solution (z : ℂ) : (1 + 2*I)*z = 4 + 3*I → z = 2 - I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l268_26890


namespace NUMINAMATH_CALUDE_special_pair_sum_l268_26898

theorem special_pair_sum (a b : ℕ+) (q r : ℕ) : 
  a^2 + b^2 = q * (a + b) + r →
  0 ≤ r →
  r < a + b →
  q^2 + r = 1977 →
  ((a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50)) :=
sorry

end NUMINAMATH_CALUDE_special_pair_sum_l268_26898


namespace NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l268_26841

/-- Two lines are non-coincident if they are distinct -/
def NonCoincidentLines (m n : Line) : Prop := m ≠ n

/-- Two planes are non-coincident if they are distinct -/
def NonCoincidentPlanes (α β : Plane) : Prop := α ≠ β

/-- A line is parallel to another line -/
def LineParallelToLine (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def LinePerpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

/-- A plane is parallel to another plane -/
def PlaneParallelToPlane (α β : Plane) : Prop := sorry

theorem parallel_planes_from_perpendicular_lines 
  (m n : Line) (α β : Plane)
  (h1 : NonCoincidentLines m n)
  (h2 : NonCoincidentPlanes α β)
  (h3 : LineParallelToLine m n)
  (h4 : LinePerpendicularToPlane m α)
  (h5 : LinePerpendicularToPlane n β) :
  PlaneParallelToPlane α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_perpendicular_lines_l268_26841


namespace NUMINAMATH_CALUDE_area_of_specific_l_shaped_figure_l268_26870

/-- Represents an L-shaped figure with given dimensions -/
structure LShapedFigure where
  bottom_length : ℕ
  bottom_width : ℕ
  central_length : ℕ
  central_width : ℕ
  top_length : ℕ
  top_width : ℕ

/-- Calculates the area of an L-shaped figure -/
def area_of_l_shaped_figure (f : LShapedFigure) : ℕ :=
  f.bottom_length * f.bottom_width +
  f.central_length * f.central_width +
  f.top_length * f.top_width

/-- Theorem stating that the area of the given L-shaped figure is 81 square units -/
theorem area_of_specific_l_shaped_figure :
  let f : LShapedFigure := {
    bottom_length := 10,
    bottom_width := 6,
    central_length := 4,
    central_width := 4,
    top_length := 5,
    top_width := 1
  }
  area_of_l_shaped_figure f = 81 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_l_shaped_figure_l268_26870


namespace NUMINAMATH_CALUDE_count_self_inverse_pairs_l268_26872

/-- A 2x2 matrix of the form [[a, 4], [-9, d]] -/
def special_matrix (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, 4; -9, d]

/-- The identity matrix of size 2x2 -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, 1]

/-- Predicate to check if a matrix is its own inverse -/
def is_self_inverse (a d : ℝ) : Prop :=
  special_matrix a d * special_matrix a d = identity_matrix

/-- The set of all pairs (a, d) where the special matrix is its own inverse -/
def self_inverse_pairs : Set (ℝ × ℝ) :=
  {p | is_self_inverse p.1 p.2}

theorem count_self_inverse_pairs :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 2 ∧ ↑s = self_inverse_pairs :=
sorry

end NUMINAMATH_CALUDE_count_self_inverse_pairs_l268_26872


namespace NUMINAMATH_CALUDE_horseback_riding_distance_l268_26859

/-- Calculates the total distance traveled during a 3-day horseback riding trip -/
theorem horseback_riding_distance : 
  let day1_speed : ℝ := 5
  let day1_time : ℝ := 7
  let day2_speed1 : ℝ := 6
  let day2_time1 : ℝ := 6
  let day2_speed2 : ℝ := day2_speed1 / 2
  let day2_time2 : ℝ := 3
  let day3_speed : ℝ := 7
  let day3_time : ℝ := 5
  let total_distance : ℝ := 
    day1_speed * day1_time + 
    day2_speed1 * day2_time1 + day2_speed2 * day2_time2 + 
    day3_speed * day3_time
  total_distance = 115 := by
  sorry


end NUMINAMATH_CALUDE_horseback_riding_distance_l268_26859


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l268_26825

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection_equals_set :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l268_26825


namespace NUMINAMATH_CALUDE_positive_expression_l268_26834

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  y + z^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l268_26834


namespace NUMINAMATH_CALUDE_remainder_problem_l268_26886

theorem remainder_problem (n : ℤ) : n % 8 = 3 → (4 * n - 9) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l268_26886


namespace NUMINAMATH_CALUDE_carpet_length_proof_l268_26849

theorem carpet_length_proof (carpet_width : ℝ) (room_area : ℝ) (coverage_percentage : ℝ) :
  carpet_width = 4 →
  room_area = 180 →
  coverage_percentage = 0.20 →
  let carpet_area := room_area * coverage_percentage
  let carpet_length := carpet_area / carpet_width
  carpet_length = 9 := by sorry

end NUMINAMATH_CALUDE_carpet_length_proof_l268_26849


namespace NUMINAMATH_CALUDE_kim_trip_time_kim_trip_time_is_120_l268_26840

/-- The total time Kim spends away from home given the described trip conditions -/
theorem kim_trip_time : ℝ :=
  let distance_to_friend : ℝ := 30
  let detour_percentage : ℝ := 0.2
  let time_at_friends : ℝ := 30
  let speed : ℝ := 44
  let distance_back : ℝ := distance_to_friend * (1 + detour_percentage)
  let total_distance : ℝ := distance_to_friend + distance_back
  let driving_time : ℝ := total_distance / speed * 60
  driving_time + time_at_friends

theorem kim_trip_time_is_120 : kim_trip_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_kim_trip_time_kim_trip_time_is_120_l268_26840


namespace NUMINAMATH_CALUDE_class_average_problem_l268_26829

theorem class_average_problem (n₁ n₂ : ℕ) (avg₂ avg_combined : ℚ) :
  n₁ = 30 →
  n₂ = 50 →
  avg₂ = 60 →
  avg_combined = 52.5 →
  ∃ avg₁ : ℚ, avg₁ = 40 ∧ (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = avg_combined :=
by sorry

end NUMINAMATH_CALUDE_class_average_problem_l268_26829


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l268_26888

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population with two subgroups --/
structure Population where
  total_size : ℕ
  subgroup1_size : ℕ
  subgroup2_size : ℕ
  h_size_sum : subgroup1_size + subgroup2_size = total_size

/-- Represents the goal of the sampling --/
inductive SamplingGoal
  | UnderstandSubgroupDifferences

/-- The most appropriate sampling method given a population and a goal --/
def most_appropriate_sampling_method (pop : Population) (goal : SamplingGoal) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is most appropriate for the given scenario --/
theorem stratified_sampling_most_appropriate 
  (pop : Population) 
  (h_equal_subgroups : pop.subgroup1_size = pop.subgroup2_size) 
  (goal : SamplingGoal) 
  (h_goal : goal = SamplingGoal.UnderstandSubgroupDifferences) :
  most_appropriate_sampling_method pop goal = SamplingMethod.Stratified :=
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l268_26888


namespace NUMINAMATH_CALUDE_intersection_M_N_l268_26866

-- Define the sets M and N
def M : Set ℝ := {x | x - 2 > 0}
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x^2 + 1)}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l268_26866


namespace NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_100_l268_26850

theorem product_seven_consecutive_divisible_by_100 (n : ℕ) : 
  100 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) :=
by sorry

end NUMINAMATH_CALUDE_product_seven_consecutive_divisible_by_100_l268_26850


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_product_l268_26801

theorem sqrt_sum_squares_eq_product (a b c : ℝ) : 
  (Real.sqrt (a^2 + b^2) = a * b) ∧ (a + b + c = 0) → (a = 0 ∧ b = 0 ∧ c = 0) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_product_l268_26801


namespace NUMINAMATH_CALUDE_gcd_840_1764_gcd_98_63_l268_26862

-- Part 1: GCD of 840 and 1764
theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by sorry

-- Part 2: GCD of 98 and 63
theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by sorry

end NUMINAMATH_CALUDE_gcd_840_1764_gcd_98_63_l268_26862


namespace NUMINAMATH_CALUDE_min_value_fraction_l268_26838

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  x^2 / (x - 9) ≥ 36 ∧ ∃ y > 9, y^2 / (y - 9) = 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l268_26838


namespace NUMINAMATH_CALUDE_lucy_groceries_l268_26846

/-- The number of packs of cookies Lucy bought -/
def cookies : ℕ := 12

/-- The number of packs of noodles Lucy bought -/
def noodles : ℕ := 16

/-- The total number of grocery packs Lucy bought -/
def total_groceries : ℕ := cookies + noodles

theorem lucy_groceries : total_groceries = 28 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l268_26846


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l268_26814

theorem sum_of_roots_quadratic (p q : ℝ) : 
  (p^2 - p - 1 = 0) → 
  (q^2 - q - 1 = 0) → 
  (p ≠ q) →
  (∃ x y : ℝ, x^2 - p*x + p*q = 0 ∧ y^2 - p*y + p*q = 0 ∧ x + y = (1 + Real.sqrt 5) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l268_26814


namespace NUMINAMATH_CALUDE_identity_function_divisibility_l268_26880

theorem identity_function_divisibility (f : ℕ → ℕ) :
  (∀ m n : ℕ, (f m + f n) ∣ (m + n)) →
  ∀ m : ℕ, f m = m :=
by sorry

end NUMINAMATH_CALUDE_identity_function_divisibility_l268_26880


namespace NUMINAMATH_CALUDE_initial_gum_count_l268_26833

/-- The number of gum pieces Adrianna had initially -/
def initial_gum : ℕ := sorry

/-- The number of gum pieces Adrianna bought -/
def bought_gum : ℕ := 3

/-- The number of friends Adrianna gave gum to -/
def friends : ℕ := 11

/-- The number of gum pieces Adrianna has left -/
def remaining_gum : ℕ := 2

/-- Theorem stating that the initial number of gum pieces was 10 -/
theorem initial_gum_count : initial_gum = 10 := by sorry

end NUMINAMATH_CALUDE_initial_gum_count_l268_26833


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l268_26816

theorem magnitude_of_complex_power (z : ℂ) (n : ℕ) :
  z = 3/5 + 4/5 * I → n = 6 → Complex.abs (z^n) = 1 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l268_26816


namespace NUMINAMATH_CALUDE_set_intersection_problem_l268_26869

theorem set_intersection_problem (p q : ℝ) : 
  let M := {x : ℝ | x^2 - 5*x ≤ 0}
  let N := {x : ℝ | p < x ∧ x < 6}
  ({x : ℝ | 2 < x ∧ x ≤ q} = M ∩ N) → p + q = 7 := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l268_26869


namespace NUMINAMATH_CALUDE_addition_problems_l268_26810

theorem addition_problems :
  (15 + (-8) + 4 + (-10) = 1) ∧
  ((-2) + (7 + 1/2) + 4.5 = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_addition_problems_l268_26810


namespace NUMINAMATH_CALUDE_product_evaluation_l268_26897

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l268_26897


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l268_26882

-- Problem 1
theorem problem_1 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 := by sorry

-- Problem 2
theorem problem_2 (p q : ℝ) : (-p*q)^3 = -p^3 * q^3 := by sorry

-- Problem 3
theorem problem_3 (a : ℝ) : a^3 * a^4 * a + (a^2)^4 - (-2*a^4)^2 = -2 * a^8 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l268_26882


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_17_l268_26874

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.b - 5)^2 + |t.c - 7| = 0 ∧ |t.a - 3| = 2

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_is_17 :
  ∀ t : Triangle, satisfies_conditions t → perimeter t = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_17_l268_26874


namespace NUMINAMATH_CALUDE_function_always_positive_l268_26879

/-- The function f(x) = (2-a^2)x + a is always positive in the interval [0,1] if and only if 0 < a < 2 -/
theorem function_always_positive (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, (2 - a^2) * x + a > 0) ↔ (0 < a ∧ a < 2) := by
  sorry

end NUMINAMATH_CALUDE_function_always_positive_l268_26879


namespace NUMINAMATH_CALUDE_sarah_meal_combinations_l268_26877

/-- Represents the number of options for each meal component -/
structure MealOptions where
  appetizers : Nat
  mainCourses : Nat
  drinks : Nat
  desserts : Nat

/-- Represents the constraint on drink options when fries are chosen -/
def drinkOptionsWithFries (options : MealOptions) : Nat :=
  options.drinks - 1

/-- Calculates the number of meal combinations -/
def calculateMealCombinations (options : MealOptions) : Nat :=
  let mealsWithFries := 1 * options.mainCourses * (drinkOptionsWithFries options) * options.desserts
  let mealsWithoutFries := (options.appetizers - 1) * options.mainCourses * options.drinks * options.desserts
  mealsWithFries + mealsWithoutFries

/-- The main theorem stating the number of distinct meals Sarah can buy -/
theorem sarah_meal_combinations (options : MealOptions) 
  (h1 : options.appetizers = 3)
  (h2 : options.mainCourses = 3)
  (h3 : options.drinks = 3)
  (h4 : options.desserts = 2) : 
  calculateMealCombinations options = 48 := by
  sorry

#eval calculateMealCombinations { appetizers := 3, mainCourses := 3, drinks := 3, desserts := 2 }

end NUMINAMATH_CALUDE_sarah_meal_combinations_l268_26877


namespace NUMINAMATH_CALUDE_fraction_simplification_l268_26875

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + 5 * Real.sqrt 48) = (5 * Real.sqrt 3) / 84 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l268_26875


namespace NUMINAMATH_CALUDE_cheolsu_weight_l268_26889

/-- Proves that Cheolsu's weight is 36 kg given the conditions stated in the problem -/
theorem cheolsu_weight :
  ∀ (cheolsu_weight mother_weight : ℝ),
    cheolsu_weight = (2 / 3) * mother_weight →
    cheolsu_weight + 72 = 2 * mother_weight →
    cheolsu_weight = 36 := by
  sorry

end NUMINAMATH_CALUDE_cheolsu_weight_l268_26889


namespace NUMINAMATH_CALUDE_medium_supermarkets_in_sample_l268_26891

/-- Represents the number of medium-sized supermarkets in a stratified sample -/
def medium_sample_size (total_population : ℕ) (medium_population : ℕ) (sample_size : ℕ) : ℕ :=
  (medium_population * sample_size) / total_population

/-- Theorem: In a stratified sample of 100 supermarkets from a population of 2000 supermarkets, 
    where 400 are medium-sized, the number of medium-sized supermarkets in the sample is 20 -/
theorem medium_supermarkets_in_sample :
  medium_sample_size 2000 400 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_medium_supermarkets_in_sample_l268_26891


namespace NUMINAMATH_CALUDE_units_digit_theorem_l268_26809

theorem units_digit_theorem : ∃ n : ℕ, (33 * 219^89 + 89^19) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_theorem_l268_26809


namespace NUMINAMATH_CALUDE_necktie_colors_l268_26843

-- Define the number of different colored shirts
def num_shirts : ℕ := 4

-- Define the probability of all boxes containing matching colors
def match_probability : ℚ := 1 / 24

-- Theorem statement
theorem necktie_colors (n : ℕ) : 
  (n : ℚ) ^ num_shirts = 1 / match_probability → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_necktie_colors_l268_26843


namespace NUMINAMATH_CALUDE_total_reading_time_is_seven_weeks_l268_26842

/-- Represents the reading plan for a section of the Bible -/
structure ReadingPlan where
  weekdaySpeed : ℕ  -- pages per hour on weekdays
  weekdayTime : ℚ   -- hours read on weekdays
  saturdaySpeed : ℕ -- pages per hour on Saturdays
  saturdayTime : ℚ  -- hours read on Saturdays
  pageCount : ℕ     -- total pages in this section

/-- Calculates the number of weeks needed to complete a reading plan -/
def weeksToComplete (plan : ReadingPlan) : ℚ :=
  let pagesPerWeek := plan.weekdaySpeed * plan.weekdayTime * 5 + plan.saturdaySpeed * plan.saturdayTime
  plan.pageCount / pagesPerWeek

/-- The reading plan for the Books of Moses -/
def mosesplan : ReadingPlan := {
  weekdaySpeed := 30,
  weekdayTime := 3/2,
  saturdaySpeed := 40,
  saturdayTime := 2,
  pageCount := 450
}

/-- The reading plan for the rest of the Bible -/
def restplan : ReadingPlan := {
  weekdaySpeed := 45,
  weekdayTime := 3/2,
  saturdaySpeed := 60,
  saturdayTime := 5/2,
  pageCount := 2350
}

/-- Theorem stating that the total reading time is 7 weeks -/
theorem total_reading_time_is_seven_weeks :
  ⌈weeksToComplete mosesplan⌉ + ⌈weeksToComplete restplan⌉ = 7 := by
  sorry


end NUMINAMATH_CALUDE_total_reading_time_is_seven_weeks_l268_26842


namespace NUMINAMATH_CALUDE_unique_distance_l268_26835

def is_valid_distance (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  (n / 10) = n - ((n % 10) * 10 + (n / 10))

theorem unique_distance : ∃! n : ℕ, is_valid_distance n ∧ n = 98 :=
sorry

end NUMINAMATH_CALUDE_unique_distance_l268_26835


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_is_three_root_six_over_four_l268_26873

/-- A pyramid with a square base and equilateral triangle lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (base_is_square : base_side > 0)
  (lateral_faces_are_equilateral : True)

/-- A cube inscribed in a pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (base_on_pyramid_base : True)
  (top_vertices_touch_midpoints : True)

/-- The volume of the inscribed cube -/
def inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) : ℝ :=
  c.side_length ^ 3

/-- Main theorem: The volume of the inscribed cube is 3√6/4 -/
theorem inscribed_cube_volume_is_three_root_six_over_four 
  (p : Pyramid) 
  (h_base : p.base_side = 2)
  (c : InscribedCube p) : 
  inscribed_cube_volume p c = 3 * Real.sqrt 6 / 4 :=
sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_is_three_root_six_over_four_l268_26873


namespace NUMINAMATH_CALUDE_sum_10_to_16_l268_26863

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = a n * q
  sum_2_4 : a 2 + a 4 = 32
  sum_6_8 : a 6 + a 8 = 16

/-- The sum of the 10th, 12th, 14th, and 16th terms equals 12 -/
theorem sum_10_to_16 (seq : GeometricSequence) :
  seq.a 10 + seq.a 12 + seq.a 14 + seq.a 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_10_to_16_l268_26863


namespace NUMINAMATH_CALUDE_abc_inequality_l268_26821

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_prod : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l268_26821


namespace NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cos_identity_l268_26813

theorem arithmetic_sequence_triangle_cos_identity (A B C : Real) (a b c : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  -- Arithmetic sequence condition
  2 * b = a + c ∧
  -- Side-angle relationships (law of sines)
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) →
  -- Theorem to prove
  5 * (Real.cos A) - 4 * (Real.cos A) * (Real.cos C) + 5 * (Real.cos C) = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_triangle_cos_identity_l268_26813


namespace NUMINAMATH_CALUDE_john_needs_two_planks_l268_26837

/-- The number of planks needed for a house wall --/
def planks_needed (total_nails : ℕ) (nails_per_plank : ℕ) : ℕ :=
  total_nails / nails_per_plank

/-- Theorem: John needs 2 planks for the house wall --/
theorem john_needs_two_planks :
  planks_needed 4 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_two_planks_l268_26837
