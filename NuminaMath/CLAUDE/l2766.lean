import Mathlib

namespace remainder_of_sum_of_powers_l2766_276647

theorem remainder_of_sum_of_powers (n : ℕ) : (20^16 + 201^6) % 9 = 7 := by
  sorry

end remainder_of_sum_of_powers_l2766_276647


namespace simplify_expression_l2766_276649

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x*y*z)⁻¹ * (x + y + z)⁻¹ :=
by sorry

end simplify_expression_l2766_276649


namespace field_trip_buses_l2766_276627

/-- The number of classrooms in the school -/
def num_classrooms : ℕ := 67

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 66

/-- The number of seats in each bus -/
def seats_per_bus : ℕ := 6

/-- The function to calculate the minimum number of buses needed -/
def min_buses_needed (classrooms : ℕ) (students : ℕ) (seats : ℕ) : ℕ :=
  (classrooms * students + seats - 1) / seats

/-- Theorem stating the minimum number of buses needed for the field trip -/
theorem field_trip_buses :
  min_buses_needed num_classrooms students_per_classroom seats_per_bus = 738 := by
  sorry


end field_trip_buses_l2766_276627


namespace second_half_speed_l2766_276679

theorem second_half_speed (total_distance : ℝ) (first_half_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 3600)
  (h2 : first_half_speed = 90)
  (h3 : total_time = 30) :
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 180 :=
by sorry

end second_half_speed_l2766_276679


namespace shape_ratios_l2766_276692

/-- Given three shapes (cube A, cube B, and cylinder C) with specific volume ratios
    and height relationships, this theorem proves the ratios of their dimensions. -/
theorem shape_ratios (a b r : ℝ) (h : ℝ) :
  a > 0 ∧ b > 0 ∧ r > 0 ∧ h > 0 →
  h = a →
  a^3 / b^3 = 81 / 25 →
  a^3 / (π * r^2 * h) = 81 / 40 →
  (a / b = 3 / 5) ∧ (a / r = 9 * Real.sqrt π / Real.sqrt 40) := by
  sorry

#check shape_ratios

end shape_ratios_l2766_276692


namespace original_number_of_people_l2766_276699

theorem original_number_of_people (x : ℕ) : 
  (x / 2 : ℚ) = 18 → x = 36 := by sorry

end original_number_of_people_l2766_276699


namespace french_speakers_l2766_276666

theorem french_speakers (total : ℕ) (latin : ℕ) (neither : ℕ) (both : ℕ) :
  total = 25 →
  latin = 13 →
  neither = 6 →
  both = 9 →
  ∃ french : ℕ, french = 15 ∧ total = latin + french - both + neither :=
by sorry

end french_speakers_l2766_276666


namespace supermarket_sales_l2766_276646

-- Define the sales volume function
def P (k b t : ℕ) : ℕ := k * t + b

-- Define the unit price function
def Q (t : ℕ) : ℕ :=
  if t < 25 then t + 20 else 80 - t

-- Define the daily sales revenue function
def Y (k b t : ℕ) : ℕ := P k b t * Q t

theorem supermarket_sales (k b : ℕ) :
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30) →
  P k b 5 = 55 →
  P k b 10 = 50 →
  (P k b 20 = 40) ∧
  (∀ t : ℕ, 1 ≤ t ∧ t ≤ 30 → Y k b t ≤ 2395) ∧
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 30 ∧ Y k b t = 2395) :=
by sorry

end supermarket_sales_l2766_276646


namespace shopkeeper_loss_l2766_276693

/-- Represents the shopkeeper's fruit inventory and sales data --/
structure FruitShop where
  total_fruit : ℝ
  apples : ℝ
  oranges : ℝ
  bananas : ℝ
  apple_price : ℝ
  orange_price : ℝ
  banana_price : ℝ
  apple_increase : ℝ
  orange_increase : ℝ
  banana_increase : ℝ
  overhead : ℝ
  apple_morning_sales : ℝ
  orange_morning_sales : ℝ
  banana_morning_sales : ℝ

/-- Calculates the profit of the fruit shop --/
def calculate_profit (shop : FruitShop) : ℝ :=
  let morning_revenue := 
    shop.apple_price * shop.apples * shop.apple_morning_sales +
    shop.orange_price * shop.oranges * shop.orange_morning_sales +
    shop.banana_price * shop.bananas * shop.banana_morning_sales
  let afternoon_revenue := 
    shop.apple_price * (1 + shop.apple_increase) * shop.apples * (1 - shop.apple_morning_sales) +
    shop.orange_price * (1 + shop.orange_increase) * shop.oranges * (1 - shop.orange_morning_sales) +
    shop.banana_price * (1 + shop.banana_increase) * shop.bananas * (1 - shop.banana_morning_sales)
  let total_revenue := morning_revenue + afternoon_revenue
  let total_cost := 
    shop.apple_price * shop.apples +
    shop.orange_price * shop.oranges +
    shop.banana_price * shop.bananas +
    shop.overhead
  total_revenue - total_cost

/-- Theorem stating that the shopkeeper incurs a loss of $178.88 --/
theorem shopkeeper_loss (shop : FruitShop) 
  (h1 : shop.total_fruit = 700)
  (h2 : shop.apples = 280)
  (h3 : shop.oranges = 210)
  (h4 : shop.bananas = shop.total_fruit - shop.apples - shop.oranges)
  (h5 : shop.apple_price = 5)
  (h6 : shop.orange_price = 4)
  (h7 : shop.banana_price = 2)
  (h8 : shop.apple_increase = 0.12)
  (h9 : shop.orange_increase = 0.15)
  (h10 : shop.banana_increase = 0.08)
  (h11 : shop.overhead = 320)
  (h12 : shop.apple_morning_sales = 0.5)
  (h13 : shop.orange_morning_sales = 0.6)
  (h14 : shop.banana_morning_sales = 0.8) :
  calculate_profit shop = -178.88 := by
  sorry


end shopkeeper_loss_l2766_276693


namespace meet_once_l2766_276668

/-- Represents the movement of Michael and the garbage truck -/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of times Michael and the truck meet -/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific movement scenario described in the problem -/
def problem_scenario : Movement where
  michael_speed := 6
  truck_speed := 12
  pail_distance := 300
  truck_stop_time := 20

/-- Theorem stating that Michael and the truck meet exactly once -/
theorem meet_once : number_of_meetings problem_scenario = 1 := by
  sorry

end meet_once_l2766_276668


namespace no_natural_solutions_l2766_276642

theorem no_natural_solutions : ∀ x y : ℕ, x^2 + x*y + y^2 ≠ x^2 * y^2 := by
  sorry

end no_natural_solutions_l2766_276642


namespace min_value_of_f_l2766_276674

def f (x : ℝ) : ℝ := (x - 1)^2 + 3

theorem min_value_of_f :
  ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = 3 :=
sorry

end min_value_of_f_l2766_276674


namespace bank_coins_l2766_276643

/-- Given a total of 11 coins, including 2 dimes and 2 nickels, prove that the number of quarters is 7. -/
theorem bank_coins (total : ℕ) (dimes : ℕ) (nickels : ℕ) (quarters : ℕ)
  (h_total : total = 11)
  (h_dimes : dimes = 2)
  (h_nickels : nickels = 2)
  (h_sum : total = dimes + nickels + quarters) :
  quarters = 7 := by
  sorry

end bank_coins_l2766_276643


namespace range_of_m_l2766_276624

/-- The range of m that satisfies the given conditions -/
theorem range_of_m (m : ℝ) : m ≥ 9 ↔ 
  (∀ x : ℝ, (|1 - x| > 2 → (x^2 - 2*x + 1 - m^2 > 0))) ∧ 
  (∃ x : ℝ, |1 - x| > 2 ∧ x^2 - 2*x + 1 - m^2 ≤ 0) ∧
  m > 0 :=
by sorry


end range_of_m_l2766_276624


namespace treasure_chest_gems_l2766_276648

theorem treasure_chest_gems (diamonds : ℕ) (rubies : ℕ) : 
  diamonds = 45 → rubies = 5110 → diamonds + rubies = 5155 := by
  sorry

end treasure_chest_gems_l2766_276648


namespace min_area_sum_l2766_276682

def point := ℝ × ℝ

def triangle_area (p1 p2 p3 : point) : ℝ := sorry

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem min_area_sum (m : ℝ) :
  let p1 : point := (2, 8)
  let p2 : point := (12, 20)
  let p3 : point := (8, m)
  is_integer m →
  (∀ k : ℝ, is_integer k → 
    k ≠ 15.2 → 
    triangle_area p1 p2 (8, k) ≥ triangle_area p1 p2 p3) →
  m ≠ 15.2 →
  (∃ n : ℝ, is_integer n ∧ 
    n ≠ 15.2 ∧
    triangle_area p1 p2 (8, n) = triangle_area p1 p2 p3 ∧
    |m - 15.2| + |n - 15.2| = |14 - 15.2| + |16 - 15.2|) →
  m + (30 - m) = 30 := by sorry

end min_area_sum_l2766_276682


namespace incorrect_average_calculation_l2766_276686

theorem incorrect_average_calculation (n : ℕ) (correct_sum incorrect_sum : ℝ) 
  (h1 : n = 10)
  (h2 : correct_sum / n = 15)
  (h3 : incorrect_sum = correct_sum - 10) :
  incorrect_sum / n = 14 := by
  sorry

end incorrect_average_calculation_l2766_276686


namespace updated_mean_after_decrement_l2766_276673

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 34 →
  (n * original_mean - n * decrement) / n = 166 := by
  sorry

end updated_mean_after_decrement_l2766_276673


namespace largest_prime_factors_difference_l2766_276695

/-- The positive difference between the two largest prime factors of 204204 is 16 -/
theorem largest_prime_factors_difference : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  p ∣ 204204 ∧ q ∣ 204204 ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 204204 → r ≤ p) ∧
  (∀ r : Nat, Nat.Prime r → r ∣ 204204 → r ≠ p → r ≤ q) ∧
  p - q = 16 := by
sorry

end largest_prime_factors_difference_l2766_276695


namespace arithmetic_sequence_sum_l2766_276606

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem stating that for an arithmetic sequence with S₅ = 5 and S₉ = 27, S₇ = 14 -/
theorem arithmetic_sequence_sum (a₁ d : ℚ) 
  (h₁ : S 5 a₁ d = 5)
  (h₂ : S 9 a₁ d = 27) : 
  S 7 a₁ d = 14 := by
  sorry

end arithmetic_sequence_sum_l2766_276606


namespace not_sufficient_nor_necessary_l2766_276622

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by
  sorry

end not_sufficient_nor_necessary_l2766_276622


namespace pharmacy_masks_problem_l2766_276685

theorem pharmacy_masks_problem (first_batch_cost second_batch_cost : ℕ)
  (h1 : first_batch_cost = 1600)
  (h2 : second_batch_cost = 6000)
  (h3 : ∃ (x : ℕ), x > 0 ∧ 
    (second_batch_cost : ℚ) / (3 * x) - (first_batch_cost : ℚ) / x = 2) :
  ∃ (x : ℕ), x = 200 ∧ 
    (second_batch_cost : ℚ) / (3 * x) - (first_batch_cost : ℚ) / x = 2 :=
by
  sorry

end pharmacy_masks_problem_l2766_276685


namespace darrel_took_48_candies_l2766_276607

/-- Represents the number of candies on the table -/
structure CandyCount where
  red : ℕ
  blue : ℕ

/-- Represents the state of candies on the table at different stages -/
structure CandyState where
  initial : CandyCount
  afterDarrel : CandyCount
  afterCloe : CandyCount

/-- Darrel's action of taking candies -/
def darrelAction (x : ℕ) (c : CandyCount) : CandyCount :=
  { red := c.red - x, blue := c.blue - x }

/-- Cloe's action of taking candies -/
def cloeAction (c : CandyCount) : CandyCount :=
  { red := c.red - 12, blue := c.blue - 12 }

/-- The theorem to be proved -/
theorem darrel_took_48_candies (state : CandyState) (x : ℕ) :
  state.initial.red = 3 * state.initial.blue →
  state.afterDarrel = darrelAction x state.initial →
  state.afterDarrel.red = 4 * state.afterDarrel.blue →
  state.afterCloe = cloeAction state.afterDarrel →
  state.afterCloe.red = 5 * state.afterCloe.blue →
  2 * x = 48 := by
  sorry


end darrel_took_48_candies_l2766_276607


namespace greatest_mpn_l2766_276696

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  tens_nonzero : tens ≠ 0
  tens_single_digit : tens < 10
  ones_single_digit : ones < 10

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  hundreds_nonzero : hundreds ≠ 0
  hundreds_single_digit : hundreds < 10
  tens_single_digit : tens < 10
  ones_single_digit : ones < 10

def is_valid_mpn (m n : Nat) (mpn : ThreeDigitNumber) : Prop :=
  m ≠ n ∧
  m < 10 ∧
  n < 10 ∧
  mpn.hundreds = m ∧
  mpn.ones = m ∧
  (10 * m + n) * m = 100 * mpn.hundreds + 10 * mpn.tens + mpn.ones

theorem greatest_mpn :
  ∀ m n : Nat,
  ∀ mpn : ThreeDigitNumber,
  is_valid_mpn m n mpn →
  mpn.hundreds * 100 + mpn.tens * 10 + mpn.ones ≤ 898 :=
sorry

end greatest_mpn_l2766_276696


namespace expression_value_l2766_276608

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y + 2 = 3 := by
  sorry

end expression_value_l2766_276608


namespace arithmetic_sequence_increasing_iff_sum_inequality_l2766_276678

theorem arithmetic_sequence_increasing_iff_sum_inequality 
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * d) →
  (∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * d) →
  (∀ n : ℕ, n ≥ 2 → S n < n * a n) ↔ d > 0 :=
sorry

end arithmetic_sequence_increasing_iff_sum_inequality_l2766_276678


namespace moss_pollen_scientific_notation_l2766_276626

theorem moss_pollen_scientific_notation (d : ℝ) (n : ℤ) :
  d = 0.0000084 →
  d = 8.4 * (10 : ℝ) ^ n →
  n = -6 :=
by
  sorry

end moss_pollen_scientific_notation_l2766_276626


namespace bug_travel_distance_l2766_276659

/-- The total distance traveled by a bug on a number line -/
def bugDistance (start end1 end2 end3 : ℤ) : ℝ :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

/-- Theorem: The bug's total travel distance is 25 units -/
theorem bug_travel_distance :
  bugDistance (-3) (-7) 8 2 = 25 := by
  sorry

end bug_travel_distance_l2766_276659


namespace concert_ticket_price_l2766_276660

theorem concert_ticket_price 
  (adult_price : ℚ) 
  (child_price : ℚ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (total_revenue : ℚ) :
  child_price = adult_price / 2 →
  num_adults = 183 →
  num_children = 28 →
  total_revenue = 5122 →
  num_adults * adult_price + num_children * child_price = total_revenue →
  adult_price = 26 := by
sorry

end concert_ticket_price_l2766_276660


namespace quadratic_root_relation_l2766_276650

/-- The quadratic equation with coefficient m, (1/3), and 1 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  m * x^2 + (1/3) * x + 1 = 0

theorem quadratic_root_relation (m₁ m₂ x₁ x₂ x₃ x₄ : ℝ) :
  quadratic_equation m₁ x₁ →
  quadratic_equation m₁ x₂ →
  quadratic_equation m₂ x₃ →
  quadratic_equation m₂ x₄ →
  x₁ < x₃ →
  x₃ < x₄ →
  x₄ < x₂ →
  x₂ < 0 →
  m₂ > m₁ ∧ m₁ > 0 :=
by sorry

end quadratic_root_relation_l2766_276650


namespace count_with_zero_3017_l2766_276669

/-- A function that counts the number of integers from 1 to n that contain at least one digit '0' in their base-ten representation. -/
def count_with_zero (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the count of positive integers less than or equal to 3017 that contain at least one digit '0' in their base-ten representation is 740. -/
theorem count_with_zero_3017 : count_with_zero 3017 = 740 :=
  sorry

end count_with_zero_3017_l2766_276669


namespace green_toad_count_l2766_276658

/-- Represents the number of toads per acre -/
structure ToadPopulation where
  green : ℕ
  brown : ℕ
  spotted_brown : ℕ

/-- The conditions of the toad population -/
def valid_population (p : ToadPopulation) : Prop :=
  p.brown = 25 * p.green ∧
  p.spotted_brown = p.brown / 4 ∧
  p.spotted_brown = 50

/-- Theorem stating that in a valid toad population, there are 8 green toads per acre -/
theorem green_toad_count (p : ToadPopulation) (h : valid_population p) : p.green = 8 := by
  sorry


end green_toad_count_l2766_276658


namespace sock_selection_theorem_l2766_276661

/-- The number of ways to select two socks of different colors -/
def differentColorPairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: The number of ways to select two socks of different colors
    from a drawer containing 5 white socks, 3 brown socks, and 4 blue socks
    is equal to 47. -/
theorem sock_selection_theorem :
  differentColorPairs 5 3 4 = 47 := by
  sorry

end sock_selection_theorem_l2766_276661


namespace squirrel_nuts_problem_l2766_276671

theorem squirrel_nuts_problem 
  (a b c d : ℕ) 
  (h1 : a + b + c + d = 2020)
  (h2 : a ≥ 103 ∧ b ≥ 103 ∧ c ≥ 103 ∧ d ≥ 103)
  (h3 : a > b ∧ a > c ∧ a > d)
  (h4 : b + c = 1277) :
  a = 640 := by
  sorry

end squirrel_nuts_problem_l2766_276671


namespace g_9_l2766_276662

/-- A function g satisfying g(x + y) = g(x) * g(y) for all real x and y, and g(3) = 4 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The functional equation for g -/
axiom g_mul (x y : ℝ) : g (x + y) = g x * g y

/-- The initial condition for g -/
axiom g_3 : g 3 = 4

/-- Theorem stating that g(9) = 64 -/
theorem g_9 : g 9 = 64 := by
  sorry

end g_9_l2766_276662


namespace cube_volume_problem_l2766_276663

theorem cube_volume_problem (a : ℝ) : 
  a > 0 →
  (a + 2) * (a + 2) * a - a^3 = 12 →
  a^3 = 1 := by
sorry

end cube_volume_problem_l2766_276663


namespace hilt_fountain_distance_l2766_276654

/-- The total distance Mrs. Hilt walks to and from the water fountain -/
def total_distance (distance_to_fountain : ℕ) (number_of_trips : ℕ) : ℕ :=
  2 * distance_to_fountain * number_of_trips

/-- Theorem: Mrs. Hilt walks 240 feet in total -/
theorem hilt_fountain_distance :
  total_distance 30 4 = 240 := by
  sorry

end hilt_fountain_distance_l2766_276654


namespace point_not_in_transformed_plane_l2766_276634

/-- A plane in 3D space -/
structure Plane where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Similarity transformation of a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { A := p.A, B := p.B, C := p.C, D := k * p.D }

/-- Check if a point satisfies a plane equation -/
def satisfiesPlane (point : Point) (plane : Plane) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

theorem point_not_in_transformed_plane :
  let originalPlane : Plane := { A := 7, B := -6, C := 1, D := -5 }
  let k : ℝ := -2
  let A : Point := { x := 1, y := 1, z := 1 }
  let transformedPlane := transformPlane originalPlane k
  ¬ satisfiesPlane A transformedPlane := by
  sorry

end point_not_in_transformed_plane_l2766_276634


namespace grid_whitening_theorem_l2766_276614

/-- Represents the color of a square -/
inductive Color
| Black
| White

/-- Represents a grid of squares -/
def Grid := Matrix (Fin 98) (Fin 98) Color

/-- Represents a sub-rectangle in the grid -/
structure SubRectangle where
  top_left : Fin 98 × Fin 98
  width : Nat
  height : Nat
  width_valid : width > 1
  height_valid : height > 1
  in_bounds : top_left.1 + width ≤ 98 ∧ top_left.2 + height ≤ 98

/-- Represents a color-flipping operation on a sub-rectangle -/
def flip_operation (grid : Grid) (rect : SubRectangle) : Grid :=
  sorry

/-- Represents a sequence of color-flipping operations -/
def operation_sequence := List SubRectangle

/-- Applies a sequence of operations to a grid -/
def apply_operations (grid : Grid) (ops : operation_sequence) : Grid :=
  sorry

/-- Checks if all squares in the grid are white -/
def all_white (grid : Grid) : Prop :=
  sorry

/-- Main theorem: There exists a finite sequence of operations that turns any grid all white -/
theorem grid_whitening_theorem (initial_grid : Grid) :
  ∃ (ops : operation_sequence), all_white (apply_operations initial_grid ops) :=
sorry

end grid_whitening_theorem_l2766_276614


namespace mary_weight_change_ratio_l2766_276619

/-- Represents the sequence of weight changes in Mary's diet journey -/
structure WeightChange where
  initial_weight : ℝ
  initial_loss : ℝ
  second_gain : ℝ
  third_loss : ℝ
  final_gain : ℝ
  final_weight : ℝ

/-- Theorem representing Mary's weight change problem -/
theorem mary_weight_change_ratio (w : WeightChange)
  (h1 : w.initial_weight = 99)
  (h2 : w.initial_loss = 12)
  (h3 : w.second_gain = 2 * w.initial_loss)
  (h4 : w.final_gain = 6)
  (h5 : w.final_weight = 81)
  (h6 : w.initial_weight - w.initial_loss + w.second_gain - w.third_loss + w.final_gain = w.final_weight) :
  w.third_loss / w.initial_loss = 3 := by
  sorry


end mary_weight_change_ratio_l2766_276619


namespace lemon_orange_ratio_decrease_l2766_276636

/-- Calculates the percentage decrease in the ratio of lemons to oranges --/
theorem lemon_orange_ratio_decrease 
  (initial_lemons : ℕ) 
  (initial_oranges : ℕ) 
  (final_lemons : ℕ) 
  (final_oranges : ℕ) 
  (h1 : initial_lemons = 50) 
  (h2 : initial_oranges = 60) 
  (h3 : final_lemons = 20) 
  (h4 : final_oranges = 40) :
  (1 - (final_lemons * initial_oranges) / (initial_lemons * final_oranges : ℚ)) * 100 = 40 := by
  sorry


end lemon_orange_ratio_decrease_l2766_276636


namespace pizza_parlor_cost_theorem_l2766_276604

/-- Calculates the total cost including gratuity for a group celebration at a pizza parlor -/
def pizza_parlor_cost (total_people : ℕ) (child_pizza_cost adult_pizza_cost child_drink_cost adult_drink_cost : ℚ) (gratuity_rate : ℚ) : ℚ :=
  let num_adults : ℕ := total_people / 3
  let num_children : ℕ := 2 * num_adults
  let child_cost : ℚ := num_children * (child_pizza_cost + child_drink_cost)
  let adult_cost : ℚ := num_adults * (adult_pizza_cost + adult_drink_cost)
  let subtotal : ℚ := child_cost + adult_cost
  let gratuity : ℚ := subtotal * gratuity_rate
  subtotal + gratuity

/-- The total cost including gratuity for the group celebration at the pizza parlor is $1932 -/
theorem pizza_parlor_cost_theorem : 
  pizza_parlor_cost 120 10 12 3 4 (15/100) = 1932 :=
by sorry

end pizza_parlor_cost_theorem_l2766_276604


namespace one_sixth_star_neg_one_l2766_276618

-- Define the ※ operation for rational numbers
def star_op (m n : ℚ) : ℚ := (3*m + n) * (3*m - n) + n

-- State the theorem
theorem one_sixth_star_neg_one :
  star_op (1/6) (-1) = -7/4 := by sorry

end one_sixth_star_neg_one_l2766_276618


namespace no_solution_equation_simplify_fraction_l2766_276683

-- Problem 1
theorem no_solution_equation :
  ¬ ∃ x : ℝ, (3 - x) / (x - 4) - 1 / (4 - x) = 1 := by sorry

-- Problem 2
theorem simplify_fraction (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
  2 * x / (x^2 - 4) - 1 / (x + 2) = 1 / (x - 2) := by sorry

end no_solution_equation_simplify_fraction_l2766_276683


namespace min_rolls_for_two_sixes_l2766_276620

/-- The probability of getting two sixes in a single roll of two dice -/
def p : ℚ := 1 / 36

/-- The probability of not getting two sixes in a single roll of two dice -/
def q : ℚ := 1 - p

/-- The number of rolls -/
def n : ℕ := 25

/-- The theorem stating that n is the minimum number of rolls required -/
theorem min_rolls_for_two_sixes (n : ℕ) : 
  (1 - q ^ n > (1 : ℚ) / 2) ∧ ∀ m < n, (1 - q ^ m ≤ (1 : ℚ) / 2) :=
sorry

end min_rolls_for_two_sixes_l2766_276620


namespace range_of_a_l2766_276664

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | |x + a| < 1}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (B a ⊂ A) ∧ (B a ≠ A) → 0 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l2766_276664


namespace convex_polygon_23_sides_diagonals_l2766_276645

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex polygon with 23 sides has 230 diagonals -/
theorem convex_polygon_23_sides_diagonals :
  num_diagonals 23 = 230 := by sorry

end convex_polygon_23_sides_diagonals_l2766_276645


namespace twentieth_term_is_79_l2766_276633

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- The 20th term of the specific arithmetic sequence is 79 -/
theorem twentieth_term_is_79 :
  arithmetic_sequence 3 4 20 = 79 := by
  sorry

end twentieth_term_is_79_l2766_276633


namespace min_sum_reciprocal_constraint_l2766_276684

theorem min_sum_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 / x) + (2 / y) = 1) : 
  x + y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (2 / x₀) + (2 / y₀) = 1 ∧ x₀ + y₀ = 8 :=
sorry

end min_sum_reciprocal_constraint_l2766_276684


namespace horner_method_f_3_l2766_276635

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem horner_method_f_3 :
  f 3 = horner_eval [5, 4, 3, 2, 1, 0] 3 ∧ horner_eval [5, 4, 3, 2, 1, 0] 3 = 1641 := by
  sorry

end horner_method_f_3_l2766_276635


namespace lcm_18_24_30_l2766_276616

theorem lcm_18_24_30 : Nat.lcm 18 (Nat.lcm 24 30) = 360 := by
  sorry

end lcm_18_24_30_l2766_276616


namespace rural_school_absence_percentage_l2766_276672

theorem rural_school_absence_percentage :
  let total_students : ℕ := 120
  let boys : ℕ := 70
  let girls : ℕ := 50
  let absent_boys : ℕ := boys / 5
  let absent_girls : ℕ := girls / 4
  let total_absent : ℕ := absent_boys + absent_girls
  (total_absent : ℚ) / total_students * 100 = 22.5 := by
  sorry

end rural_school_absence_percentage_l2766_276672


namespace max_leap_years_150_years_l2766_276687

/-- A calendrical system where leap years occur every four years -/
structure CalendarSystem where
  leap_year_frequency : ℕ
  leap_year_frequency_is_four : leap_year_frequency = 4

/-- The maximum number of leap years in a given period -/
def max_leap_years (c : CalendarSystem) (period : ℕ) : ℕ :=
  (period / c.leap_year_frequency) + min 1 (period % c.leap_year_frequency)

/-- Theorem stating that the maximum number of leap years in a 150-year period is 38 -/
theorem max_leap_years_150_years (c : CalendarSystem) :
  max_leap_years c 150 = 38 := by
  sorry

#eval max_leap_years ⟨4, rfl⟩ 150

end max_leap_years_150_years_l2766_276687


namespace multiply_a_equals_four_l2766_276690

theorem multiply_a_equals_four (a b x : ℝ) 
  (h1 : x * a = 5 * b) 
  (h2 : a * b ≠ 0) 
  (h3 : a / 5 = b / 4) : 
  x = 4 := by
  sorry

end multiply_a_equals_four_l2766_276690


namespace expected_value_is_thirteen_eighths_l2766_276639

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieRoll
  | one
  | two
  | three
  | four
  | five
  | six
  | seven
  | eight

/-- Determines if a DieRoll is prime -/
def isPrime (roll : DieRoll) : Bool :=
  match roll with
  | DieRoll.two | DieRoll.three | DieRoll.five | DieRoll.seven => true
  | _ => false

/-- Calculates the winnings for a given DieRoll -/
def winnings (roll : DieRoll) : Int :=
  match roll with
  | DieRoll.two => 2
  | DieRoll.three => 3
  | DieRoll.five => 5
  | DieRoll.seven => 7
  | DieRoll.eight => -4
  | _ => 0

/-- The probability of each DieRoll -/
def probability : DieRoll → Rat
  | _ => 1/8

/-- The expected value of the winnings -/
def expectedValue : Rat :=
  (winnings DieRoll.one   * probability DieRoll.one)   +
  (winnings DieRoll.two   * probability DieRoll.two)   +
  (winnings DieRoll.three * probability DieRoll.three) +
  (winnings DieRoll.four  * probability DieRoll.four)  +
  (winnings DieRoll.five  * probability DieRoll.five)  +
  (winnings DieRoll.six   * probability DieRoll.six)   +
  (winnings DieRoll.seven * probability DieRoll.seven) +
  (winnings DieRoll.eight * probability DieRoll.eight)

theorem expected_value_is_thirteen_eighths :
  expectedValue = 13/8 := by sorry

end expected_value_is_thirteen_eighths_l2766_276639


namespace min_value_expression_l2766_276621

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b - a - 2 * b = 0) :
  (a^2 / 4 - 2 / a + b^2 - 1 / b) ≥ 7 :=
sorry

end min_value_expression_l2766_276621


namespace binomial_10_choose_4_l2766_276632

theorem binomial_10_choose_4 : Nat.choose 10 4 = 210 := by sorry

end binomial_10_choose_4_l2766_276632


namespace total_silver_dollars_l2766_276609

theorem total_silver_dollars (chiu phung ha lin : ℕ) : 
  chiu = 56 →
  phung = chiu + 16 →
  ha = phung + 5 →
  lin = (chiu + phung + ha) + 25 →
  chiu + phung + ha + lin = 435 :=
by sorry

end total_silver_dollars_l2766_276609


namespace lucas_payment_l2766_276638

/-- Calculates the payment for window cleaning based on given conditions -/
def calculate_payment (windows_per_floor : ℕ) (num_floors : ℕ) (pay_per_window : ℕ) 
                      (deduction_per_period : ℕ) (days_per_period : ℕ) (days_taken : ℕ) : ℕ :=
  let total_windows := windows_per_floor * num_floors
  let gross_pay := total_windows * pay_per_window
  let num_periods := days_taken / days_per_period
  let total_deduction := num_periods * deduction_per_period
  gross_pay - total_deduction

/-- Theorem stating that Lucas will be paid $16 for cleaning windows -/
theorem lucas_payment : 
  calculate_payment 3 3 2 1 3 6 = 16 := by
  sorry

end lucas_payment_l2766_276638


namespace complex_z_in_first_quadrant_l2766_276688

theorem complex_z_in_first_quadrant (z : ℂ) (h : (1 : ℂ) + Complex.I = Complex.I / z) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_z_in_first_quadrant_l2766_276688


namespace line_through_ellipse_focus_l2766_276691

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := 10 * x^2 + y^2 = 10

/-- The line equation -/
def line (x y b : ℝ) : Prop := 2 * x + b * y + 3 = 0

/-- Theorem: The value of b for a line passing through a focus of the given ellipse is either -1 or 1 -/
theorem line_through_ellipse_focus (b : ℝ) : 
  (∃ x y : ℝ, ellipse x y ∧ line x y b) → b = -1 ∨ b = 1 := by
  sorry

end line_through_ellipse_focus_l2766_276691


namespace arithmetic_calculations_l2766_276603

theorem arithmetic_calculations : 
  (1 - 2 + 3 + (-4) = -2) ∧ 
  ((-6) / 3 - (-10) - |(-8)| = 0) := by sorry

end arithmetic_calculations_l2766_276603


namespace quadratic_factorization_l2766_276611

/-- Given a quadratic equation that can be factored into two linear factors, prove the value of m -/
theorem quadratic_factorization (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x y : ℝ), 
    x^2 + 7*x*y + m*y^2 - 5*x + 43*y - 24 = (x + a*y + 3) * (x + b*y - 8)) → 
  m = -18 := by
sorry

end quadratic_factorization_l2766_276611


namespace sum_of_squares_representation_l2766_276655

theorem sum_of_squares_representation : 
  (((17 ^ 2 + 19 ^ 2) / 2) ^ 2 : ℕ) = 260 ^ 2 + 195 ^ 2 := by
  sorry

end sum_of_squares_representation_l2766_276655


namespace milk_problem_l2766_276617

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (sam_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  sam_fraction = 1/3 →
  sam_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/8 := by
  sorry

end milk_problem_l2766_276617


namespace polynomial_divisibility_double_divisibility_not_triple_divisible_l2766_276698

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

end polynomial_divisibility_double_divisibility_not_triple_divisible_l2766_276698


namespace quadratic_real_roots_l2766_276600

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ (m ≤ 4/3 ∧ m ≠ 0) :=
by sorry

end quadratic_real_roots_l2766_276600


namespace spinner_probability_l2766_276637

/-- Represents the outcomes of the spinner -/
inductive SpinnerOutcome
| one
| two
| three
| five

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinnerOutcome
  tens : SpinnerOutcome
  units : SpinnerOutcome

def isDivisibleByFive (n : ThreeDigitNumber) : Prop :=
  n.units = SpinnerOutcome.five

def totalOutcomes : ℕ := 4^3

def favorableOutcomes : ℕ := 4 * 4

theorem spinner_probability :
  (favorableOutcomes : ℚ) / totalOutcomes = 1 / 4 := by sorry

end spinner_probability_l2766_276637


namespace parabola_focal_distance_l2766_276675

/-- Given a parabola y^2 = 2px with focus F and a point A(1, 2) on the parabola, |AF| = 2 -/
theorem parabola_focal_distance (p : ℝ) (F : ℝ × ℝ) :
  (∀ x y, y^2 = 2*p*x → (x, y) = (1, 2)) →  -- point A(1, 2) satisfies the parabola equation
  F.1 = p/2 →  -- x-coordinate of focus
  F.2 = 0 →  -- y-coordinate of focus
  let A := (1, 2)
  ((A.1 - F.1)^2 + (A.2 - F.2)^2).sqrt = 2 := by
sorry

end parabola_focal_distance_l2766_276675


namespace circle_center_l2766_276640

/-- The center of a circle given by the equation x^2 - 8x + y^2 - 4y = 5 is (4, 2) -/
theorem circle_center (x y : ℝ) : x^2 - 8*x + y^2 - 4*y = 5 → (4, 2) = (x, y) := by
  sorry

end circle_center_l2766_276640


namespace school_play_tickets_l2766_276641

theorem school_play_tickets (total_money : ℕ) (adult_price child_price : ℕ) 
  (child_tickets : ℕ) :
  total_money = 104 →
  adult_price = 6 →
  child_price = 4 →
  child_tickets = 11 →
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + child_price * child_tickets = total_money ∧
    adult_tickets + child_tickets = 21 := by
  sorry

end school_play_tickets_l2766_276641


namespace third_root_of_polynomial_l2766_276676

theorem third_root_of_polynomial (a b : ℚ) : 
  (∀ x : ℚ, a * x^3 + (a + 3*b) * x^2 + (2*b - 4*a) * x + (10 - a) = 0 ↔ x = -1 ∨ x = 4 ∨ x = -24/19) :=
by sorry

end third_root_of_polynomial_l2766_276676


namespace correct_statements_count_l2766_276625

-- Define a structure for sampling statements
structure SamplingStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : SamplingStatement :=
  { id := 1
  , content := "When the total number of individuals in a population is not large, it is appropriate to use simple random sampling"
  , isCorrect := true }

def statement2 : SamplingStatement :=
  { id := 2
  , content := "In systematic sampling, after the population is divided evenly, simple random sampling is used in each part"
  , isCorrect := false }

def statement3 : SamplingStatement :=
  { id := 3
  , content := "The lottery activities in department stores are a method of drawing lots"
  , isCorrect := true }

def statement4 : SamplingStatement :=
  { id := 4
  , content := "In systematic sampling, the probability of each individual being selected is equal throughout the entire sampling process (except when exclusions are made)"
  , isCorrect := true }

-- Define the list of all statements
def allStatements : List SamplingStatement := [statement1, statement2, statement3, statement4]

-- Theorem: The number of correct statements is 3
theorem correct_statements_count :
  (allStatements.filter (λ s => s.isCorrect)).length = 3 := by
  sorry


end correct_statements_count_l2766_276625


namespace circle_radius_sqrt_29_l2766_276628

/-- Given a circle with center on the x-axis that passes through points (2,2) and (-1,5),
    prove that its radius is √29 -/
theorem circle_radius_sqrt_29 :
  ∃ (x : ℝ), 
    (x - 2)^2 + 2^2 = (x + 1)^2 + 5^2 →
    Real.sqrt ((x - 2)^2 + 2^2) = Real.sqrt 29 := by
  sorry

end circle_radius_sqrt_29_l2766_276628


namespace claire_photos_l2766_276677

theorem claire_photos (lisa robert claire : ℕ) 
  (h1 : lisa = robert) 
  (h2 : lisa = 3 * claire) 
  (h3 : robert = claire + 16) : 
  claire = 8 := by
sorry

end claire_photos_l2766_276677


namespace basketball_team_probability_l2766_276697

def team_size : ℕ := 12
def main_players : ℕ := 6
def classes_with_two_students : ℕ := 2
def classes_with_one_student : ℕ := 8

theorem basketball_team_probability :
  (Nat.choose classes_with_two_students 1 * Nat.choose classes_with_two_students 1 * Nat.choose classes_with_one_student 4) / 
  Nat.choose team_size main_players = 10 / 33 := by
  sorry

end basketball_team_probability_l2766_276697


namespace vectors_not_parallel_l2766_276623

def vector_a : Fin 2 → ℝ := ![2, 0]
def vector_b : Fin 2 → ℝ := ![0, 2]

theorem vectors_not_parallel : ¬ (∃ (k : ℝ), vector_a = k • vector_b) := by
  sorry

end vectors_not_parallel_l2766_276623


namespace organization_member_count_organization_has_ten_members_l2766_276644

/-- Represents an organization with committees and members -/
structure Organization :=
  (num_committees : ℕ)
  (num_members : ℕ)
  (member_committee_count : ℕ)
  (shared_member_count : ℕ)

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem stating the required number of members in the organization -/
theorem organization_member_count (org : Organization) 
  (h1 : org.num_committees = 5)
  (h2 : org.member_committee_count = 2)
  (h3 : org.shared_member_count = 1) :
  org.num_members = choose_two org.num_committees :=
by sorry

/-- The main theorem proving the organization must have 10 members -/
theorem organization_has_ten_members (org : Organization) 
  (h1 : org.num_committees = 5)
  (h2 : org.member_committee_count = 2)
  (h3 : org.shared_member_count = 1) :
  org.num_members = 10 :=
by sorry

end organization_member_count_organization_has_ten_members_l2766_276644


namespace stock_price_calculation_l2766_276615

def stock_price_evolution (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  price_after_first_year * (1 - second_year_decrease)

theorem stock_price_calculation :
  stock_price_evolution 150 0.5 0.3 = 157.5 := by
  sorry

end stock_price_calculation_l2766_276615


namespace john_gave_twenty_l2766_276610

/-- The amount of money John gave to the store for buying Slurpees -/
def money_given (cost_per_slurpee : ℕ) (num_slurpees : ℕ) (change_received : ℕ) : ℕ :=
  cost_per_slurpee * num_slurpees + change_received

/-- Proof that John gave $20 to the store -/
theorem john_gave_twenty :
  money_given 2 6 8 = 20 := by
  sorry

end john_gave_twenty_l2766_276610


namespace product_of_roots_quadratic_l2766_276657

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 3 = 0 → 
  x₂^2 - 2*x₂ - 3 = 0 → 
  x₁ * x₂ = -3 := by
  sorry

end product_of_roots_quadratic_l2766_276657


namespace cos_15_degrees_l2766_276613

theorem cos_15_degrees : Real.cos (15 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degrees_l2766_276613


namespace quadratic_inequality_solution_set_l2766_276670

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 3
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
sorry

end quadratic_inequality_solution_set_l2766_276670


namespace red_balls_count_l2766_276605

/-- Given a bag of balls with the following properties:
  * The total number of balls is 60
  * The frequency of picking red balls is 0.15
  Prove that the number of red balls in the bag is 9 -/
theorem red_balls_count (total_balls : ℕ) (red_frequency : ℝ) 
  (h1 : total_balls = 60)
  (h2 : red_frequency = 0.15) :
  ⌊total_balls * red_frequency⌋ = 9 := by
  sorry

end red_balls_count_l2766_276605


namespace area_of_overlapping_squares_area_of_overlapping_squares_is_252_l2766_276681

/-- The area of the region covered by two congruent squares with side length 12 units,
    where one corner of one square coincides with a corner of the other square. -/
theorem area_of_overlapping_squares : ℝ :=
  let square_side_length : ℝ := 12
  let single_square_area : ℝ := square_side_length ^ 2
  let total_area_without_overlap : ℝ := 2 * single_square_area
  let overlap_area : ℝ := single_square_area / 4
  total_area_without_overlap - overlap_area

/-- The area of the region covered by the two squares is 252 square units. -/
theorem area_of_overlapping_squares_is_252 :
  area_of_overlapping_squares = 252 := by sorry

end area_of_overlapping_squares_area_of_overlapping_squares_is_252_l2766_276681


namespace expense_settlement_proof_l2766_276651

def expense_settlement (alice_paid bob_paid charlie_paid : ℚ) : Prop :=
  let total_paid := alice_paid + bob_paid + charlie_paid
  let share_per_person := total_paid / 3
  let alice_owes := share_per_person - alice_paid
  let bob_owes := share_per_person - bob_paid
  let charlie_owed := charlie_paid - share_per_person
  ∃ a b : ℚ, 
    a = alice_owes ∧ 
    b = bob_owes ∧ 
    a - b = 30

theorem expense_settlement_proof :
  expense_settlement 130 160 210 := by
  sorry

end expense_settlement_proof_l2766_276651


namespace male_teacher_classes_proof_l2766_276667

/-- Represents the number of classes taught by male teachers when only male teachers are teaching. -/
def male_teacher_classes : ℕ := 10

/-- Represents the number of classes taught by female teachers. -/
def female_teacher_classes : ℕ := 15

/-- Represents the average number of tutoring classes per month. -/
def average_classes : ℕ := 6

theorem male_teacher_classes_proof (x y : ℕ) :
  female_teacher_classes * x = average_classes * (x + y) →
  male_teacher_classes * y = average_classes * (x + y) :=
by sorry

end male_teacher_classes_proof_l2766_276667


namespace logarithm_equality_l2766_276629

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm base 5 function
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem logarithm_equality : lg 2 + lg 5 + 2 * log5 10 - log5 20 = 2 := by
  sorry

end logarithm_equality_l2766_276629


namespace equation_equivalence_l2766_276630

theorem equation_equivalence (x : ℝ) (h : x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 ∧ x ≠ 7) : 
  1 / (x - 3) + 1 / (x - 5) + 1 / (x - 7) = 4 / (x - 4) ↔ 
  x^3 - 13*x^2 + 48*x - 64 = 0 :=
by sorry

end equation_equivalence_l2766_276630


namespace suresh_investment_l2766_276602

/-- Given the total profit, Ramesh's investment, and Ramesh's share of profit, 
    prove that Suresh's investment is Rs. 24,000. -/
theorem suresh_investment 
  (total_profit : ℕ) 
  (ramesh_investment : ℕ) 
  (ramesh_profit : ℕ) 
  (h1 : total_profit = 19000)
  (h2 : ramesh_investment = 40000)
  (h3 : ramesh_profit = 11875) :
  (total_profit - ramesh_profit) * ramesh_investment / ramesh_profit = 24000 := by
  sorry

end suresh_investment_l2766_276602


namespace mans_rate_in_still_water_l2766_276694

/-- The rate of a man rowing in still water, given his speeds with and against a stream. -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_with : speed_with_stream = 20)
  (h_against : speed_against_stream = 4) :
  (speed_with_stream + speed_against_stream) / 2 = 12 :=
by sorry

end mans_rate_in_still_water_l2766_276694


namespace inequality_solution_set_l2766_276653

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (a + 1) * x + a < 0}
  (a > 1 → S = {x : ℝ | 1 < x ∧ x < a}) ∧
  (a = 1 → S = ∅) ∧
  (a < 1 → S = {x : ℝ | a < x ∧ x < 1}) :=
by sorry

end inequality_solution_set_l2766_276653


namespace intersection_of_A_and_B_l2766_276689

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l2766_276689


namespace negation_equivalence_l2766_276612

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) := by
  sorry

end negation_equivalence_l2766_276612


namespace tan_sum_pi_eighths_l2766_276601

theorem tan_sum_pi_eighths : Real.tan (π / 8) + Real.tan (3 * π / 8) = 2 * Real.sqrt 2 := by
  sorry

end tan_sum_pi_eighths_l2766_276601


namespace correct_lunch_bill_l2766_276652

/-- The cost of Sara's lunch items and the total bill -/
def lunch_bill (hotdog_cost salad_cost : ℚ) : Prop :=
  hotdog_cost = 5.36 ∧ salad_cost = 5.10 ∧ hotdog_cost + salad_cost = 10.46

/-- Theorem stating that the total lunch bill is correct -/
theorem correct_lunch_bill :
  ∃ (hotdog_cost salad_cost : ℚ), lunch_bill hotdog_cost salad_cost :=
sorry

end correct_lunch_bill_l2766_276652


namespace race_finish_times_l2766_276656

/-- Race problem statement -/
theorem race_finish_times 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (lila_speed : ℝ) 
  (race_distance : ℝ) 
  (h1 : malcolm_speed = 6)
  (h2 : joshua_speed = 8)
  (h3 : lila_speed = 7)
  (h4 : race_distance = 12) :
  let malcolm_time := malcolm_speed * race_distance
  let joshua_time := joshua_speed * race_distance
  let lila_time := lila_speed * race_distance
  (joshua_time - malcolm_time = 24 ∧ lila_time - malcolm_time = 12) := by
  sorry


end race_finish_times_l2766_276656


namespace equation_solutions_l2766_276665

/-- The equation has solutions for all real a, and these solutions are as specified -/
theorem equation_solutions (a : ℝ) :
  let f := fun x : ℝ => (2 - 2*a*(x + 1)) / (|x| - x) = Real.sqrt (1 - a - a*x)
  (∃ x, f x) ∧
  (a < 0 → (f ((1-a)/a) ∧ f (-1))) ∧
  (0 ≤ a ∧ a ≤ 1 → f (-1)) ∧
  (1 < a ∧ a < 2 → (f ((1-a)/a) ∧ f (-1))) ∧
  (a = 2 → (f (-1) ∧ f (-1/2))) ∧
  (a > 2 → (f ((1-a)/a) ∧ f (-1) ∧ f (1-a))) :=
by sorry

end equation_solutions_l2766_276665


namespace triangular_front_view_solids_l2766_276680

/-- A type representing different types of solids -/
inductive Solid
  | TriangularPyramid
  | SquarePyramid
  | TriangularPrism
  | SquarePrism
  | Cone
  | Cylinder

/-- A predicate that determines if a solid can have a triangular front view -/
def has_triangular_front_view (s : Solid) : Prop :=
  match s with
  | Solid.TriangularPyramid => True
  | Solid.SquarePyramid => True
  | Solid.TriangularPrism => True
  | Solid.Cone => True
  | _ => False

/-- Theorem stating which solids can have a triangular front view -/
theorem triangular_front_view_solids :
  ∀ s : Solid, has_triangular_front_view s ↔ 
    (s = Solid.TriangularPyramid ∨ 
     s = Solid.SquarePyramid ∨ 
     s = Solid.TriangularPrism ∨ 
     s = Solid.Cone) :=
by sorry

end triangular_front_view_solids_l2766_276680


namespace intersection_of_A_and_B_l2766_276631

def set_A : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 1)}
def set_B : Set ℝ := Set.Ioo 0 1

theorem intersection_of_A_and_B :
  set_A ∩ set_B = Set.Ioo 0 1 := by sorry

end intersection_of_A_and_B_l2766_276631
