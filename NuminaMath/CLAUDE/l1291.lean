import Mathlib

namespace final_sum_is_eight_times_original_l1291_129172

theorem final_sum_is_eight_times_original (S a b : ℝ) (h : a + b = S) :
  (2 * (4 * a)) + (2 * (4 * b)) = 8 * S := by
  sorry

end final_sum_is_eight_times_original_l1291_129172


namespace stratified_sampling_correct_l1291_129113

/-- Represents a car model with its production volume -/
structure CarModel where
  name : String
  volume : Nat

/-- Calculates the number of cars to be sampled from a given model -/
def sampleSize (model : CarModel) (totalProduction : Nat) (totalSample : Nat) : Nat :=
  (model.volume * totalSample) / totalProduction

/-- Theorem stating that the stratified sampling produces the correct sample sizes -/
theorem stratified_sampling_correct 
  (emgrand kingKong freedomShip : CarModel)
  (h1 : emgrand.volume = 1600)
  (h2 : kingKong.volume = 6000)
  (h3 : freedomShip.volume = 2000)
  (h4 : emgrand.volume + kingKong.volume + freedomShip.volume = 9600)
  (h5 : 48 ≤ 9600) :
  let totalProduction := 9600
  let totalSample := 48
  (sampleSize emgrand totalProduction totalSample = 8) ∧
  (sampleSize kingKong totalProduction totalSample = 30) ∧
  (sampleSize freedomShip totalProduction totalSample = 10) := by
  sorry

end stratified_sampling_correct_l1291_129113


namespace cyclist_rejoining_time_l1291_129123

/-- Prove that the time taken for a cyclist to break away from a group, travel 10 km ahead, 
    turn back, and rejoin the group is 1/4 hours. -/
theorem cyclist_rejoining_time 
  (group_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (separation_distance : ℝ) 
  (h1 : group_speed = 35) 
  (h2 : cyclist_speed = 45) 
  (h3 : separation_distance = 20) : 
  (separation_distance / (cyclist_speed - group_speed) = 1/4) := by
sorry

end cyclist_rejoining_time_l1291_129123


namespace trailing_zeros_100_factorial_l1291_129117

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 100! is 24 -/
theorem trailing_zeros_100_factorial :
  trailingZeros 100 = 24 := by
  sorry

end trailing_zeros_100_factorial_l1291_129117


namespace max_min_f_on_interval_l1291_129146

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_f_on_interval :
  ∃ (max min : ℝ), max = 5 ∧ min = -15 ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
  (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
  (∃ x ∈ Set.Icc 0 3, f x = max) ∧
  (∃ x ∈ Set.Icc 0 3, f x = min) :=
sorry

end max_min_f_on_interval_l1291_129146


namespace daisies_bought_l1291_129143

theorem daisies_bought (flower_price : ℕ) (roses_bought : ℕ) (total_spent : ℕ) : ℕ :=
  let daisies : ℕ := (total_spent - roses_bought * flower_price) / flower_price
  by
    -- Proof goes here
    sorry

#check daisies_bought 6 7 60 = 3

end daisies_bought_l1291_129143


namespace trapezoid_area_three_squares_l1291_129118

/-- The area of the trapezoid formed by three squares with sides 3, 5, and 7 units -/
theorem trapezoid_area_three_squares :
  let square1 : ℝ := 3
  let square2 : ℝ := 5
  let square3 : ℝ := 7
  let total_base : ℝ := square1 + square2 + square3
  let height_ratio : ℝ := square3 / total_base
  let trapezoid_height : ℝ := square2
  let trapezoid_base1 : ℝ := square1 * height_ratio
  let trapezoid_base2 : ℝ := (square1 + square2) * height_ratio
  let trapezoid_area : ℝ := (trapezoid_base1 + trapezoid_base2) * trapezoid_height / 2
  trapezoid_area = 12.825 := by
  sorry

end trapezoid_area_three_squares_l1291_129118


namespace equal_utility_at_two_l1291_129139

/-- Utility function -/
def utility (swimming : ℝ) (coding : ℝ) : ℝ := 2 * swimming * coding + 1

/-- Saturday's utility -/
def saturday_utility (t : ℝ) : ℝ := utility t (10 - 2*t)

/-- Sunday's utility -/
def sunday_utility (t : ℝ) : ℝ := utility (4 - t) (2*t + 2)

/-- Theorem: The value of t that results in equal utility for both days is 2 -/
theorem equal_utility_at_two :
  ∃ t : ℝ, saturday_utility t = sunday_utility t ∧ t = 2 := by
sorry

end equal_utility_at_two_l1291_129139


namespace negation_of_forall_positive_square_leq_zero_l1291_129169

theorem negation_of_forall_positive_square_leq_zero :
  (¬ ∀ x : ℝ, x > 0 → x^2 ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 > 0) :=
by sorry

end negation_of_forall_positive_square_leq_zero_l1291_129169


namespace no_integer_both_roots_finite_decimal_l1291_129124

theorem no_integer_both_roots_finite_decimal (n : ℤ) (hn : n ≠ 0) :
  ¬(∃ (x₁ x₂ : ℚ), 
    (x₁ ≠ x₂) ∧
    ((4 * n^2 - 1) * x₁^2 - 4 * n^2 * x₁ + n^2 = 0) ∧
    ((4 * n^2 - 1) * x₂^2 - 4 * n^2 * x₂ + n^2 = 0) ∧
    (∃ (a b c d : ℤ), x₁ = (a : ℚ) / (2^b * 5^c) ∧ x₂ = (d : ℚ) / (2^b * 5^c))) :=
sorry

end no_integer_both_roots_finite_decimal_l1291_129124


namespace orange_marbles_count_l1291_129197

/-- The number of orange marbles in a jar, given the total number of marbles,
    the number of red marbles, and that half of the marbles are blue. -/
def orangeMarbles (total : ℕ) (red : ℕ) (halfAreBlue : Bool) : ℕ :=
  total - (total / 2 + red)

/-- Theorem stating that there are 6 orange marbles in a jar with 24 total marbles,
    6 red marbles, and half of the marbles being blue. -/
theorem orange_marbles_count :
  orangeMarbles 24 6 true = 6 := by
  sorry

end orange_marbles_count_l1291_129197


namespace fraction_division_five_sixths_divided_by_nine_tenths_l1291_129134

theorem fraction_division (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : c ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem five_sixths_divided_by_nine_tenths :
  (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 :=
by sorry

end fraction_division_five_sixths_divided_by_nine_tenths_l1291_129134


namespace modular_inverse_of_5_mod_26_l1291_129122

theorem modular_inverse_of_5_mod_26 :
  ∃! x : ℕ, x ∈ Finset.range 26 ∧ (5 * x) % 26 = 1 :=
by
  use 21
  sorry

end modular_inverse_of_5_mod_26_l1291_129122


namespace intersection_of_M_and_N_l1291_129190

open Set

theorem intersection_of_M_and_N :
  let U : Type := ℝ
  let M : Set U := {x | x < 1}
  let N : Set U := {x | 0 < x ∧ x < 2}
  M ∩ N = {x | 0 < x ∧ x < 1} := by
  sorry

end intersection_of_M_and_N_l1291_129190


namespace largest_power_dividing_factorial_l1291_129162

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  ∃ (k : ℕ), k = 63 ∧ 
  (∀ (m : ℕ), (2023 : ℕ)^m ∣ factorial 2023 → m ≤ k) ∧
  (2023 : ℕ)^k ∣ factorial 2023 :=
sorry

end largest_power_dividing_factorial_l1291_129162


namespace bent_polygon_total_angle_l1291_129174

/-- For a regular polygon with n sides (n > 4), if each side is bent inward at an angle θ = 360°/(2n),
    then the total angle formed by all the bends is 180°. -/
theorem bent_polygon_total_angle (n : ℕ) (h : n > 4) :
  let θ : ℝ := 360 / (2 * n)
  n * θ = 180 := by sorry

end bent_polygon_total_angle_l1291_129174


namespace divisibility_property_l1291_129157

theorem divisibility_property (m : ℤ) (n : ℕ) :
  (10 ∣ (3^n + m)) → (10 ∣ (3^(n+4) + m)) := by
  sorry

end divisibility_property_l1291_129157


namespace josephine_milk_sales_l1291_129173

/-- Given the conditions of Josephine's milk sales, prove that the amount in each of the two unknown containers is 0.75 liters. -/
theorem josephine_milk_sales (total_milk : ℝ) (big_containers : ℕ) (small_containers : ℕ) (unknown_containers : ℕ)
  (big_container_capacity : ℝ) (small_container_capacity : ℝ)
  (h_total : total_milk = 10)
  (h_big : big_containers = 3)
  (h_small : small_containers = 5)
  (h_unknown : unknown_containers = 2)
  (h_big_capacity : big_container_capacity = 2)
  (h_small_capacity : small_container_capacity = 0.5) :
  (total_milk - (big_containers * big_container_capacity + small_containers * small_container_capacity)) / unknown_containers = 0.75 := by
sorry

end josephine_milk_sales_l1291_129173


namespace smallRectLengthIsFourTimesWidth_l1291_129187

/-- Represents the arrangement of squares and a rectangle -/
structure SquareArrangement where
  s : ℝ
  largeTotalWidth : ℝ
  largeLength : ℝ
  smallRectWidth : ℝ
  smallRectLength : ℝ

/-- The conditions of the problem -/
def validArrangement (a : SquareArrangement) : Prop :=
  a.largeTotalWidth = 3 * a.s ∧
  a.largeLength = 2 * a.largeTotalWidth ∧
  a.smallRectWidth = a.s

/-- The theorem to prove -/
theorem smallRectLengthIsFourTimesWidth (a : SquareArrangement) 
  (h : validArrangement a) : a.smallRectLength = 4 * a.smallRectWidth :=
sorry

end smallRectLengthIsFourTimesWidth_l1291_129187


namespace tadd_3000th_number_l1291_129171

/-- Represents the counting game with Tadd, Todd, and Tucker --/
structure CountingGame where
  max_count : Nat
  tadd_start : Nat
  todd_initial_count : Nat
  tucker_initial_count : Nat
  increment : Nat

/-- Calculates Tadd's nth number in the game --/
def tadd_nth_number (game : CountingGame) (n : Nat) : Nat :=
  sorry

/-- The main theorem stating that Tadd's 3000th number is X --/
theorem tadd_3000th_number (game : CountingGame) 
  (h1 : game.max_count = 15000)
  (h2 : game.tadd_start = 1)
  (h3 : game.todd_initial_count = 3)
  (h4 : game.tucker_initial_count = 5)
  (h5 : game.increment = 2) :
  tadd_nth_number game 3000 = X :=
  sorry

end tadd_3000th_number_l1291_129171


namespace complex_equality_l1291_129105

theorem complex_equality (z : ℂ) : 
  Complex.abs (1 + Complex.I * z) = Complex.abs (3 + 4 * Complex.I) →
  Complex.abs (z - Complex.I) = 5 := by
sorry

end complex_equality_l1291_129105


namespace xyz_sum_root_l1291_129132

theorem xyz_sum_root (x y z : ℝ) 
  (h1 : y + z = 16) 
  (h2 : z + x = 18) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = 9 * Real.sqrt 77 := by
  sorry

end xyz_sum_root_l1291_129132


namespace ratio_theorem_l1291_129129

theorem ratio_theorem (a b c d r : ℝ) 
  (h1 : (b + c + d) / a = r)
  (h2 : (a + c + d) / b = r)
  (h3 : (a + b + d) / c = r)
  (h4 : (a + b + c) / d = r)
  : r = 3 ∨ r = -1 := by
  sorry

end ratio_theorem_l1291_129129


namespace circle_equation_range_l1291_129195

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + m*x - 2*y + 3 = 0

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  m < -2 * Real.sqrt 2 ∨ m > 2 * Real.sqrt 2

-- Theorem statement
theorem circle_equation_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) ↔ m_range m :=
sorry

end circle_equation_range_l1291_129195


namespace point_coordinates_wrt_origin_l1291_129102

/-- The coordinates of a point (-1, 2) with respect to the origin in a Cartesian coordinate system are (-1, 2) -/
theorem point_coordinates_wrt_origin :
  let P : ℝ × ℝ := (-1, 2)
  P = P :=
by sorry

end point_coordinates_wrt_origin_l1291_129102


namespace solution_set_f_geq_6_range_of_a_for_nonempty_solution_l1291_129176

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

-- Theorem for the first part of the problem
theorem solution_set_f_geq_6 :
  {x : ℝ | f x ≥ 6} = {x : ℝ | x ≤ -3 ∨ x ≥ 3} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_nonempty_solution :
  ∀ a : ℝ, (∃ x : ℝ, f x < a + x) ↔ a > 2 := by sorry

end solution_set_f_geq_6_range_of_a_for_nonempty_solution_l1291_129176


namespace quadratic_equation_roots_l1291_129130

theorem quadratic_equation_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (r₁ + r₂ = 10 ∧ |r₁ - r₂| = 12) → (a = 1 ∧ b = -10 ∧ c = -11) :=
by sorry

end quadratic_equation_roots_l1291_129130


namespace triangle_area_l1291_129110

/-- Given a triangle with perimeter 20 cm and inradius 2.5 cm, its area is 25 cm². -/
theorem triangle_area (p r A : ℝ) : 
  p = 20 → r = 2.5 → A = r * (p / 2) → A = 25 :=
by sorry

end triangle_area_l1291_129110


namespace janet_earnings_l1291_129168

/-- Calculates the total earnings of Janet based on her exterminator work and sculpture sales. -/
theorem janet_earnings (
  hourly_rate : ℝ)
  (sculpture_price_per_pound : ℝ)
  (hours_worked : ℝ)
  (sculpture1_weight : ℝ)
  (sculpture2_weight : ℝ)
  (h1 : hourly_rate = 70)
  (h2 : sculpture_price_per_pound = 20)
  (h3 : hours_worked = 20)
  (h4 : sculpture1_weight = 5)
  (h5 : sculpture2_weight = 7) :
  hourly_rate * hours_worked + sculpture_price_per_pound * (sculpture1_weight + sculpture2_weight) = 1640 :=
by
  sorry


end janet_earnings_l1291_129168


namespace room_freezer_temp_difference_l1291_129165

/-- The temperature difference between room and freezer --/
def temperature_difference (room_temp freezer_temp : ℤ) : ℤ :=
  room_temp - freezer_temp

/-- Theorem stating the temperature difference between room and freezer --/
theorem room_freezer_temp_difference :
  temperature_difference 10 (-6) = 16 := by
  sorry

end room_freezer_temp_difference_l1291_129165


namespace min_dot_product_on_hyperbola_l1291_129152

/-- The curve C: x^2 - y^2 = 1 (x > 0) -/
def C (x y : ℝ) : Prop := x^2 - y^2 = 1 ∧ x > 0

/-- The dot product function f -/
def f (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ * x₂ + y₁ * y₂

theorem min_dot_product_on_hyperbola :
  ∀ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ → C x₂ y₂ → 
  ∃ m : ℝ, m = 1 ∧ ∀ a b c d : ℝ, C a b → C c d → f x₁ y₁ x₂ y₂ ≥ m ∧ f a b c d ≥ m :=
sorry

end min_dot_product_on_hyperbola_l1291_129152


namespace printing_presses_count_l1291_129193

/-- The number of papers printed -/
def num_papers : ℕ := 500000

/-- The time taken in the first scenario (in hours) -/
def time1 : ℝ := 12

/-- The time taken in the second scenario (in hours) -/
def time2 : ℝ := 13.999999999999998

/-- The number of printing presses in the second scenario -/
def presses2 : ℕ := 30

/-- The number of printing presses in the first scenario -/
def presses1 : ℕ := 26

theorem printing_presses_count :
  (num_papers : ℝ) / time1 / (num_papers / time2) = presses1 / presses2 :=
sorry

end printing_presses_count_l1291_129193


namespace shortest_side_of_right_triangle_l1291_129106

theorem shortest_side_of_right_triangle (a b c : ℝ) (ha : a = 5) (hb : b = 12) 
  (hright : a^2 + b^2 = c^2) : 
  min a (min b c) = 5 := by
  sorry

end shortest_side_of_right_triangle_l1291_129106


namespace greatest_of_three_consecutive_integers_l1291_129191

theorem greatest_of_three_consecutive_integers (n : ℤ) :
  n + 2 = 8 → (n < n + 1 ∧ n + 1 < n + 2) → n + 2 = max n (max (n + 1) (n + 2)) :=
by sorry

end greatest_of_three_consecutive_integers_l1291_129191


namespace parabola_sum_is_line_l1291_129178

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a parabola about the x-axis -/
def reflect (p : Parabola) : Parabola :=
  { a := -p.a, b := -p.b, c := -p.c }

/-- Translates a parabola horizontally by a given amount -/
def translate (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * d, c := p.c + p.a * d^2 - p.b * d }

/-- The sum of two parabolas -/
def add (p q : Parabola) : Parabola :=
  { a := p.a + q.a, b := p.b + q.b, c := p.c + q.c }

theorem parabola_sum_is_line (p : Parabola) :
  let p1 := translate p 5
  let p2 := translate (reflect p) (-5)
  let sum := add p1 p2
  sum.a = 0 ∧ sum.b ≠ 0 := by sorry

end parabola_sum_is_line_l1291_129178


namespace total_distance_calculation_l1291_129144

/-- Calculates the total distance traveled given the distances and number of trips for each mode of transportation -/
def total_distance (plane_distance : Float) (train_distance : Float) (bus_distance : Float)
                   (plane_trips : Nat) (train_trips : Nat) (bus_trips : Nat) : Float :=
  plane_distance * plane_trips.toFloat +
  train_distance * train_trips.toFloat +
  bus_distance * bus_trips.toFloat

/-- Theorem stating that the total distance traveled is 11598.4 miles -/
theorem total_distance_calculation :
  total_distance 256.0 120.5 35.2 32 16 42 = 11598.4 := by
  sorry

#eval total_distance 256.0 120.5 35.2 32 16 42

end total_distance_calculation_l1291_129144


namespace largest_number_with_given_hcf_and_lcm_factors_l1291_129147

/-- Given three positive integers with HCF 37 and LCM with additional prime factors 17, 19, 23, and 29,
    the largest of these numbers is 7,976,237 -/
theorem largest_number_with_given_hcf_and_lcm_factors
  (a b c : ℕ+)
  (hcf_abc : Nat.gcd a b.val = 37 ∧ Nat.gcd (Nat.gcd a b.val) c.val = 37)
  (lcm_factors : ∃ (k : ℕ+), Nat.lcm (Nat.lcm a b.val) c.val = 37 * 17 * 19 * 23 * 29 * k) :
  max a (max b c) = 7976237 := by
  sorry

end largest_number_with_given_hcf_and_lcm_factors_l1291_129147


namespace quadrilaterals_from_circle_points_l1291_129103

/-- The number of points on the circumference of the circle -/
def n : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def k : ℕ := 4

/-- The number of different convex quadrilaterals that can be formed -/
def num_quadrilaterals : ℕ := Nat.choose n k

theorem quadrilaterals_from_circle_points : num_quadrilaterals = 495 := by
  sorry

end quadrilaterals_from_circle_points_l1291_129103


namespace smallest_dual_base_representation_l1291_129114

theorem smallest_dual_base_representation :
  ∃ (n : ℕ) (a b : ℕ), 
    a > 3 ∧ b > 3 ∧
    n = a + 3 ∧ n = 3 * b + 1 ∧
    (∀ (m : ℕ) (c d : ℕ), 
      c > 3 ∧ d > 3 ∧ m = c + 3 ∧ m = 3 * d + 1 → m ≥ n) :=
by sorry

end smallest_dual_base_representation_l1291_129114


namespace kate_wand_sale_l1291_129196

/-- The amount of money Kate collected after selling magic wands -/
def kateCollected (numBought : ℕ) (numSold : ℕ) (costPerWand : ℕ) (markup : ℕ) : ℕ :=
  numSold * (costPerWand + markup)

/-- Theorem stating how much money Kate collected from selling magic wands -/
theorem kate_wand_sale :
  kateCollected 3 2 60 5 = 130 := by
  sorry

end kate_wand_sale_l1291_129196


namespace calculate_expression_l1291_129158

theorem calculate_expression : (1/2)⁻¹ + (Real.pi - 3.14)^0 - |-3| + Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end calculate_expression_l1291_129158


namespace distance_to_canada_is_360_l1291_129128

/-- Calculates the distance traveled given speed, total time, and stop time. -/
def distance_to_canada (speed : ℝ) (total_time : ℝ) (stop_time : ℝ) : ℝ :=
  speed * (total_time - stop_time)

/-- Proves that the distance to Canada is 360 miles under the given conditions. -/
theorem distance_to_canada_is_360 :
  distance_to_canada 60 7 1 = 360 := by
  sorry

end distance_to_canada_is_360_l1291_129128


namespace infinitely_many_solutions_l1291_129150

theorem infinitely_many_solutions :
  ∃ f : ℕ → ℤ × ℤ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (a, b) := f n
      ∃ x y : ℝ,
        x ≠ y ∧
        x * y = 1 ∧
        x^2012 = a * x + b ∧
        y^2012 = a * y + b :=
by sorry

end infinitely_many_solutions_l1291_129150


namespace sum_of_integers_ending_in_3_l1291_129111

def sequence_first_term : ℕ := 103
def sequence_last_term : ℕ := 443
def sequence_common_difference : ℕ := 10

def sequence_length : ℕ := (sequence_last_term - sequence_first_term) / sequence_common_difference + 1

theorem sum_of_integers_ending_in_3 :
  (sequence_length : ℕ) * (sequence_first_term + sequence_last_term) / 2 = 9555 :=
by sorry

end sum_of_integers_ending_in_3_l1291_129111


namespace negation_of_proposition_negation_of_cube_positive_l1291_129167

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by sorry

theorem negation_of_cube_positive :
  (¬ ∀ x > 0, x^3 > 0) ↔ (∃ x > 0, x^3 ≤ 0) :=
by sorry

end negation_of_proposition_negation_of_cube_positive_l1291_129167


namespace class_average_approximation_l1291_129101

/-- Represents the class data for a test --/
structure ClassData where
  total_students : ℕ
  section1_percent : ℝ
  section1_average : ℝ
  section2_percent : ℝ
  section2_average : ℝ
  section3_percent : ℝ
  section3_average : ℝ
  section4_average : ℝ
  weight1 : ℝ
  weight2 : ℝ
  weight3 : ℝ
  weight4 : ℝ

/-- Calculates the weighted overall class average --/
def weightedAverage (data : ClassData) : ℝ :=
  data.section1_average * data.weight1 +
  data.section2_average * data.weight2 +
  data.section3_average * data.weight3 +
  data.section4_average * data.weight4

/-- Theorem stating that the weighted overall class average is approximately 86% --/
theorem class_average_approximation (data : ClassData) 
  (h1 : data.total_students = 120)
  (h2 : data.section1_percent = 0.187)
  (h3 : data.section1_average = 0.965)
  (h4 : data.section2_percent = 0.355)
  (h5 : data.section2_average = 0.784)
  (h6 : data.section3_percent = 0.258)
  (h7 : data.section3_average = 0.882)
  (h8 : data.section4_average = 0.647)
  (h9 : data.weight1 = 0.35)
  (h10 : data.weight2 = 0.25)
  (h11 : data.weight3 = 0.30)
  (h12 : data.weight4 = 0.10)
  (h13 : data.section1_percent + data.section2_percent + data.section3_percent + 
         (1 - data.section1_percent - data.section2_percent - data.section3_percent) = 1) :
  abs (weightedAverage data - 0.86) < 0.005 := by
  sorry


end class_average_approximation_l1291_129101


namespace exists_rectangle_six_pieces_l1291_129188

/-- A rectangle inscribed in an isosceles right triangle --/
structure InscribedRectangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  h_positive : side1 > 0 ∧ side2 > 0 ∧ hypotenuse > 0
  h_inscribed : side1 + side2 < hypotenuse

/-- Two straight lines that divide a rectangle --/
structure DividingLines where
  line1 : ℝ × ℝ → ℝ × ℝ → Prop
  line2 : ℝ × ℝ → ℝ × ℝ → Prop

/-- The number of pieces a rectangle is divided into by two straight lines --/
def numPieces (r : InscribedRectangle) (d : DividingLines) : ℕ :=
  sorry

/-- Theorem stating the existence of a rectangle that can be divided into 6 pieces --/
theorem exists_rectangle_six_pieces :
  ∃ (r : InscribedRectangle) (d : DividingLines), numPieces r d = 6 :=
sorry

end exists_rectangle_six_pieces_l1291_129188


namespace optimal_water_tank_design_l1291_129136

/-- Represents the dimensions and costs of a rectangular water tank -/
structure WaterTank where
  volume : ℝ
  depth : ℝ
  bottomCost : ℝ
  wallCost : ℝ

/-- Calculates the total cost of constructing the water tank -/
def totalCost (tank : WaterTank) (length width : ℝ) : ℝ :=
  tank.bottomCost * length * width + 
  tank.wallCost * (2 * length * tank.depth + 2 * width * tank.depth)

/-- Theorem stating the optimal dimensions and minimum cost of the water tank -/
theorem optimal_water_tank_design (tank : WaterTank) 
  (h_volume : tank.volume = 4800)
  (h_depth : tank.depth = 3)
  (h_bottom_cost : tank.bottomCost = 150)
  (h_wall_cost : tank.wallCost = 120) :
  ∃ (cost : ℝ),
    (∀ length width, 
      length * width * tank.depth = tank.volume → 
      totalCost tank length width ≥ cost) ∧
    totalCost tank 40 40 = cost ∧
    cost = 297600 := by
  sorry

end optimal_water_tank_design_l1291_129136


namespace square_sum_from_linear_equations_l1291_129181

theorem square_sum_from_linear_equations (x y : ℝ) 
  (eq1 : x + y = 12) 
  (eq2 : 3 * x + y = 20) : 
  x^2 + y^2 = 80 := by
sorry

end square_sum_from_linear_equations_l1291_129181


namespace range_of_a_l1291_129192

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
by sorry

end range_of_a_l1291_129192


namespace weight_replacement_l1291_129163

theorem weight_replacement (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  avg_increase = 1.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 77 := by
  sorry

end weight_replacement_l1291_129163


namespace negative_two_fourth_power_l1291_129120

theorem negative_two_fourth_power :
  ∀ (x : ℤ) (n : ℕ), x = -2 ∧ n = 4 → x^n = (-2)^4 := by
  sorry

end negative_two_fourth_power_l1291_129120


namespace factor_t_squared_minus_144_l1291_129155

theorem factor_t_squared_minus_144 (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) := by
  sorry

end factor_t_squared_minus_144_l1291_129155


namespace six_digit_number_puzzle_l1291_129126

theorem six_digit_number_puzzle :
  ∀ P Q R S T U : ℕ,
    P ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    Q ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    R ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    S ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    T ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    U ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
    Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
    R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
    S ≠ T ∧ S ≠ U ∧
    T ≠ U →
    (100 * P + 10 * Q + R) % 9 = 0 →
    (100 * Q + 10 * R + S) % 4 = 0 →
    (100 * R + 10 * S + T) % 3 = 0 →
    (P + Q + R + S + T + U) % 5 = 0 →
    U = 4 := by
  sorry

end six_digit_number_puzzle_l1291_129126


namespace inequality_proof_l1291_129116

theorem inequality_proof (a b c d e p q : ℝ) 
  (hp_pos : 0 < p) 
  (hq_pos : 0 < q) 
  (ha : p ≤ a ∧ a ≤ q) 
  (hb : p ≤ b ∧ b ≤ q) 
  (hc : p ≤ c ∧ c ≤ q) 
  (hd : p ≤ d ∧ d ≤ q) 
  (he : p ≤ e ∧ e ≤ q) : 
  (a + b + c + d + e) * (1/a + 1/b + 1/c + 1/d + 1/e) ≤ 25 + 6 * (Real.sqrt (p/q) - Real.sqrt (q/p))^2 :=
by sorry

end inequality_proof_l1291_129116


namespace sufficient_but_not_necessary_l1291_129166

/-- Given a > 0 and a ≠ 1, prove that if f(x) = a^x is decreasing on ℝ, 
    then g(x) = (2-a)x^3 is increasing on ℝ, but not necessarily vice versa. -/
theorem sufficient_but_not_necessary 
  (a : ℝ) 
  (ha_pos : a > 0) 
  (ha_neq_one : a ≠ 1) 
  (f : ℝ → ℝ) 
  (hf : f = fun x ↦ a^x) 
  (g : ℝ → ℝ) 
  (hg : g = fun x ↦ (2-a)*x^3) : 
  (∀ x y, x < y → f x > f y) → 
  (∀ x y, x < y → g x < g y) ∧ 
  ¬(∀ x y, x < y → g x < g y → ∀ x y, x < y → f x > f y) := by
  sorry

end sufficient_but_not_necessary_l1291_129166


namespace parallelogram_area_l1291_129199

/-- Parallelogram EFGH with given side lengths and diagonal -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  EH : ℝ
  is_parallelogram : EF > 0 ∧ FG > 0 ∧ EH > 0

/-- The area of the parallelogram EFGH -/
def area (p : Parallelogram) : ℝ :=
  p.EF * p.FG

/-- Theorem: The area of parallelogram EFGH is 1200 -/
theorem parallelogram_area (p : Parallelogram) 
  (h1 : p.EF = 40) 
  (h2 : p.FG = 30) 
  (h3 : p.EH = 50) : 
  area p = 1200 := by
  sorry

#check parallelogram_area

end parallelogram_area_l1291_129199


namespace angle_A_range_triangle_area_l1291_129133

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 + t.a * t.c = t.b^2

-- Theorem I
theorem angle_A_range (t : Triangle) (h : triangle_condition t) :
  0 < t.A ∧ t.A < π/3 := by sorry

-- Theorem II
theorem triangle_area (t : Triangle) (h : triangle_condition t) 
  (h_a : t.a = 2) (h_A : t.A = π/6) :
  (1/2) * t.b * t.c * Real.sin t.A = 2 * Real.sqrt 3 := by sorry

end angle_A_range_triangle_area_l1291_129133


namespace circle_radius_is_three_l1291_129183

-- Define the circle equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + m*y - 4 = 0

-- Define the line equation
def line_equation (x y : ℝ) : Prop :=
  2*x + y = 0

-- Define symmetry with respect to a line
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  line_equation ((x₁ + x₂)/2) ((y₁ + y₂)/2)

-- Theorem statement
theorem circle_radius_is_three (m : ℝ) 
  (h₁ : ∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ m ∧ 
    circle_equation x₂ y₂ m ∧ 
    symmetric_points x₁ y₁ x₂ y₂) :
  (let center_x := 1
   let center_y := -m/2
   let radius := Real.sqrt ((center_x - 0)^2 + (center_y - 0)^2)
   radius = 3) := by sorry

end circle_radius_is_three_l1291_129183


namespace captain_age_is_your_age_l1291_129194

/-- Represents the age of a person in years -/
def Age : Type := ℕ

/-- Represents a person -/
structure Person where
  age : Age

/-- Represents the captain of the steamboat -/
def Captain : Person := sorry

/-- Represents you -/
def You : Person := sorry

/-- The theorem states that the captain's age is equal to your age -/
theorem captain_age_is_your_age : Captain.age = You.age := by sorry

end captain_age_is_your_age_l1291_129194


namespace incenter_distance_l1291_129159

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB = 15 ∧ AC = 17 ∧ BC = 16

-- Define the incenter
def Incenter (I : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (Real.sqrt ((I.1 - A.1)^2 + (I.2 - A.2)^2) = r) ∧
  (Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) = r) ∧
  (Real.sqrt ((I.1 - C.1)^2 + (I.2 - C.2)^2) = r)

theorem incenter_distance (A B C I : ℝ × ℝ) :
  Triangle A B C → Incenter I A B C →
  Real.sqrt ((I.1 - B.1)^2 + (I.2 - B.2)^2) = Real.sqrt 85 :=
by sorry

end incenter_distance_l1291_129159


namespace system_solution_l1291_129177

def system_equations (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  (x₃ + x₄ + x₅)^5 = 3*x₁ ∧
  (x₄ + x₅ + x₁)^5 = 3*x₂ ∧
  (x₅ + x₁ + x₂)^5 = 3*x₃ ∧
  (x₁ + x₂ + x₃)^5 = 3*x₄ ∧
  (x₂ + x₃ + x₄)^5 = 3*x₅

theorem system_solution :
  ∀ x₁ x₂ x₃ x₄ x₅ : ℝ,
  system_equations x₁ x₂ x₃ x₄ x₅ →
  ((x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0) ∨
   (x₁ = 1/3 ∧ x₂ = 1/3 ∧ x₃ = 1/3 ∧ x₄ = 1/3 ∧ x₅ = 1/3) ∨
   (x₁ = -1/3 ∧ x₂ = -1/3 ∧ x₃ = -1/3 ∧ x₄ = -1/3 ∧ x₅ = -1/3)) :=
by
  sorry

#check system_solution

end system_solution_l1291_129177


namespace equation_describes_cylinder_l1291_129141

-- Define cylindrical coordinates
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define a cylinder
def IsCylinder (S : Set CylindricalCoord) (c : ℝ) : Prop :=
  c > 0 ∧ ∀ p : CylindricalCoord, p ∈ S ↔ p.r = c

-- Theorem statement
theorem equation_describes_cylinder (c : ℝ) :
  IsCylinder {p : CylindricalCoord | p.r = c} c :=
by
  sorry

end equation_describes_cylinder_l1291_129141


namespace victoria_gym_schedule_l1291_129151

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a gym schedule -/
structure GymSchedule where
  startDay : DayOfWeek
  sessionsPlanned : ℕ
  publicHolidays : ℕ
  personalEvents : ℕ

/-- Calculates the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  sorry

/-- Calculates the number of Sundays in a given number of days -/
def sundaysInDays (days : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of days needed to complete the gym schedule -/
def totalDays (schedule : GymSchedule) : ℕ :=
  sorry

/-- Theorem: Victoria completes her 30th gym session on a Wednesday -/
theorem victoria_gym_schedule (schedule : GymSchedule) 
  (h1 : schedule.startDay = DayOfWeek.Monday)
  (h2 : schedule.sessionsPlanned = 30)
  (h3 : schedule.publicHolidays = 3)
  (h4 : schedule.personalEvents = 2) :
  dayAfter schedule.startDay (totalDays schedule) = DayOfWeek.Wednesday :=
sorry

end victoria_gym_schedule_l1291_129151


namespace cantor_bernstein_l1291_129198

theorem cantor_bernstein {α β : Type*} (f : α → β) (g : β → α) 
  (hf : Function.Injective f) (hg : Function.Injective g) : 
  Nonempty (α ≃ β) :=
sorry

end cantor_bernstein_l1291_129198


namespace equilateral_triangle_solutions_l1291_129121

/-- A complex number z satisfies the equilateral triangle property if 0, z, and z^4 
    form the distinct vertices of an equilateral triangle in the complex plane. -/
def has_equilateral_triangle_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ z ≠ z^4 ∧ Complex.abs z = Complex.abs (z^4 - z) ∧ Complex.abs z = Complex.abs z^4

/-- There are exactly two nonzero complex numbers that satisfy 
    the equilateral triangle property. -/
theorem equilateral_triangle_solutions :
  ∃! (s : Finset ℂ), s.card = 2 ∧ ∀ z ∈ s, has_equilateral_triangle_property z :=
sorry

end equilateral_triangle_solutions_l1291_129121


namespace total_accidents_l1291_129161

/-- Represents the accident rate and total traffic for a highway -/
structure HighwayData where
  accidents : ℕ
  per_vehicles : ℕ
  total_vehicles : ℕ

/-- Calculates the number of accidents for a given highway -/
def calculate_accidents (data : HighwayData) : ℕ :=
  (data.accidents * data.total_vehicles) / data.per_vehicles

/-- The given data for the three highways -/
def highway_A : HighwayData := ⟨75, 100, 2500⟩
def highway_B : HighwayData := ⟨50, 80, 1600⟩
def highway_C : HighwayData := ⟨90, 200, 1900⟩

/-- The theorem stating the total number of accidents across all three highways -/
theorem total_accidents :
  calculate_accidents highway_A +
  calculate_accidents highway_B +
  calculate_accidents highway_C = 3730 := by
  sorry

end total_accidents_l1291_129161


namespace platform_length_l1291_129109

/-- Given a train of length l traveling at constant velocity, if it passes a pole in t seconds
    and a platform in 6t seconds, then the length of the platform is 5l. -/
theorem platform_length (l t : ℝ) (h1 : l > 0) (h2 : t > 0) : 
  (∃ v : ℝ, v > 0 ∧ v = l / t ∧ v = (l + 5 * l) / (6 * t)) := by
  sorry

end platform_length_l1291_129109


namespace shop_profit_per_tshirt_l1291_129131

/-- The amount the shop makes off each t-shirt -/
def T : ℝ := 25

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 115

/-- The number of t-shirts sold -/
def t_shirts_sold : ℕ := 113

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 78

/-- The price difference between a jersey and a t-shirt -/
def price_difference : ℝ := 90

theorem shop_profit_per_tshirt :
  T = 25 ∧
  jersey_profit = 115 ∧
  t_shirts_sold = 113 ∧
  jerseys_sold = 78 ∧
  jersey_profit = T + price_difference ∧
  price_difference = 90 →
  T = 25 := by sorry

end shop_profit_per_tshirt_l1291_129131


namespace tank_capacity_l1291_129175

theorem tank_capacity : 
  ∀ (T : ℝ), 
  (T > 0) →
  ((9/10 : ℝ) * T - (3/4 : ℝ) * T = 5) →
  T = 100/3 := by
sorry

end tank_capacity_l1291_129175


namespace snooker_tournament_tickets_l1291_129104

theorem snooker_tournament_tickets (total_tickets : ℕ) (vip_price gen_price : ℚ) 
  (total_revenue : ℚ) (h1 : total_tickets = 320) (h2 : vip_price = 40) 
  (h3 : gen_price = 15) (h4 : total_revenue = 7500) : 
  ∃ (vip_tickets gen_tickets : ℕ), 
    vip_tickets + gen_tickets = total_tickets ∧ 
    vip_price * vip_tickets + gen_price * gen_tickets = total_revenue ∧ 
    gen_tickets - vip_tickets = 104 :=
by sorry

end snooker_tournament_tickets_l1291_129104


namespace binomial_coefficient_problem_l1291_129179

def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_coefficient_problem (m : ℕ+) :
  let a := binomial (2 * m) m
  let b := binomial (2 * m + 1) m
  13 * a = 7 * b → m = 6 := by
sorry

end binomial_coefficient_problem_l1291_129179


namespace conference_teams_l1291_129149

/-- The number of games played in a conference where each team plays every other team twice -/
def games_played (n : ℕ) : ℕ := n * (n - 1)

/-- Theorem: There are 12 teams in the conference -/
theorem conference_teams : ∃ n : ℕ, n > 0 ∧ games_played n = 132 ∧ n = 12 := by
  sorry

end conference_teams_l1291_129149


namespace nested_square_root_value_l1291_129137

theorem nested_square_root_value (y : ℝ) :
  y = Real.sqrt (2 + y) → y = 2 := by sorry

end nested_square_root_value_l1291_129137


namespace chelsea_cupcake_time_l1291_129148

/-- Calculates the total time Chelsea spent making and decorating cupcakes --/
def total_cupcake_time (num_batches : ℕ) 
                       (bake_time_per_batch : ℕ) 
                       (ice_time_per_batch : ℕ)
                       (cupcakes_per_batch : ℕ)
                       (decor_time_per_cupcake : List ℕ) : ℕ :=
  let base_time := num_batches * (bake_time_per_batch + ice_time_per_batch)
  let decor_time := (List.map (· * cupcakes_per_batch) decor_time_per_cupcake).sum
  base_time + decor_time

/-- Theorem stating that Chelsea's total time making and decorating cupcakes is 542 minutes --/
theorem chelsea_cupcake_time : 
  total_cupcake_time 4 20 30 6 [10, 15, 12, 20] = 542 := by
  sorry


end chelsea_cupcake_time_l1291_129148


namespace distance_between_points_l1291_129153

theorem distance_between_points : 
  let pointA : ℝ × ℝ := (1, 2)
  let pointB : ℝ × ℝ := (5, 7)
  Real.sqrt ((pointB.1 - pointA.1)^2 + (pointB.2 - pointA.2)^2) = Real.sqrt 41 := by
  sorry

end distance_between_points_l1291_129153


namespace tangent_line_problem_l1291_129186

/-- Given a curve y = x^3 + ax + b and a line y = kx + 1 that is tangent to this curve at the point (1, 3), 
    the value of a - b is equal to -4. -/
theorem tangent_line_problem (a b k : ℝ) : 
  (∀ x, x^3 + a*x + b = k*x + 1 → x = 1) →  -- The line is tangent to the curve
  3^3 + a*3 + b = k*3 + 1 →                 -- The point (1, 3) lies on the curve
  3 = k*1 + 1 →                             -- The point (1, 3) lies on the line
  a - b = -4 := by
sorry

end tangent_line_problem_l1291_129186


namespace walking_distance_l1291_129154

/-- If a person walks 1.5 miles in 45 minutes, they will travel 3 miles in 90 minutes at the same rate. -/
theorem walking_distance (distance : ℝ) (time : ℝ) (new_time : ℝ) 
  (h1 : distance = 1.5)
  (h2 : time = 45)
  (h3 : new_time = 90) :
  (distance / time) * new_time = 3 := by
  sorry

#check walking_distance

end walking_distance_l1291_129154


namespace lcm_gcd_product_l1291_129138

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 9) :
  Nat.lcm a b * Nat.gcd a b = 108 := by
sorry

end lcm_gcd_product_l1291_129138


namespace parabola_theorem_l1291_129142

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  eq : (a * x^2 : ℝ) + (b * x * y : ℝ) + (c * y^2 : ℝ) + (d * x : ℝ) + (e * y : ℝ) + (f : ℝ) = 0

/-- The parabola passes through the point (2,6) -/
def passes_through (p : Parabola) : Prop :=
  (p.a * 2^2 : ℝ) + (p.b * 2 * 6 : ℝ) + (p.c * 6^2 : ℝ) + (p.d * 2 : ℝ) + (p.e * 6 : ℝ) + (p.f : ℝ) = 0

/-- The y-coordinate of the focus is 4 -/
def focus_y_coord (p : Parabola) : Prop :=
  ∃ x : ℝ, (p.a * x^2 : ℝ) + (p.b * x * 4 : ℝ) + (p.c * 4^2 : ℝ) + (p.d * x : ℝ) + (p.e * 4 : ℝ) + (p.f : ℝ) = 0

/-- The axis of symmetry is parallel to the x-axis -/
def axis_parallel_x (p : Parabola) : Prop :=
  p.b = 0 ∧ p.c ≠ 0

/-- The vertex lies on the y-axis -/
def vertex_on_y_axis (p : Parabola) : Prop :=
  ∃ y : ℝ, (p.c * y^2 : ℝ) + (p.e * y : ℝ) + (p.f : ℝ) = 0

/-- The coefficients satisfy the required conditions -/
def coeff_conditions (p : Parabola) : Prop :=
  p.c > 0 ∧ Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs p.a) (Int.natAbs p.b)) (Int.natAbs p.c)) (Int.natAbs p.d)) (Int.natAbs p.e)) (Int.natAbs p.f) = 1

/-- The main theorem stating that the given equation represents a parabola satisfying all conditions -/
theorem parabola_theorem : ∃ p : Parabola, 
  p.a = 0 ∧ p.b = 0 ∧ p.c = 1 ∧ p.d = -2 ∧ p.e = -8 ∧ p.f = 16 ∧
  passes_through p ∧
  focus_y_coord p ∧
  axis_parallel_x p ∧
  vertex_on_y_axis p ∧
  coeff_conditions p :=
sorry

end parabola_theorem_l1291_129142


namespace square_plus_inverse_square_l1291_129170

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) : x^2 + 1/x^2 = 11 := by
  sorry

end square_plus_inverse_square_l1291_129170


namespace sum_of_two_smallest_prime_factors_of_280_l1291_129107

theorem sum_of_two_smallest_prime_factors_of_280 : 
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p < q ∧ 
  p ∣ 280 ∧ q ∣ 280 ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ 280 → r = p ∨ r ≥ q) ∧
  p + q = 7 :=
by sorry

end sum_of_two_smallest_prime_factors_of_280_l1291_129107


namespace smallest_n_is_correct_smallest_n_satisfies_property_l1291_129125

/-- The smallest positive integer n with the given divisibility property -/
def smallest_n : ℕ := 13

/-- Proposition stating that smallest_n is the correct answer -/
theorem smallest_n_is_correct :
  ∀ (n : ℕ), n > 0 → 
  (∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
    x ∣ y^3 → y ∣ z^3 → z ∣ x^3 →
    x * y * z ∣ (x + y + z)^n) →
  n ≥ smallest_n :=
by sorry

/-- Proposition stating that smallest_n satisfies the required property -/
theorem smallest_n_satisfies_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 →
    x ∣ y^3 → y ∣ z^3 → z ∣ x^3 →
    x * y * z ∣ (x + y + z)^smallest_n :=
by sorry

end smallest_n_is_correct_smallest_n_satisfies_property_l1291_129125


namespace odd_even_array_parity_l1291_129119

/-- Represents an n × n array where each entry is either 1 or -1 -/
def OddEvenArray (n : ℕ) := Fin n → Fin n → Int

/-- Counts the number of rows with an odd number of -1s -/
def oddRowCount (A : OddEvenArray n) : ℕ := sorry

/-- Counts the number of columns with an odd number of -1s -/
def oddColumnCount (A : OddEvenArray n) : ℕ := sorry

/-- The main theorem -/
theorem odd_even_array_parity (n : ℕ) (hn : Odd n) (A : OddEvenArray n) :
  Even (oddRowCount A + oddColumnCount A) := by sorry

end odd_even_array_parity_l1291_129119


namespace profit_percentage_calculation_l1291_129160

def selling_price : ℝ := 900
def profit : ℝ := 100

theorem profit_percentage_calculation :
  (profit / (selling_price - profit)) * 100 = 12.5 := by
  sorry

end profit_percentage_calculation_l1291_129160


namespace toy_ratio_after_removal_l1291_129184

/-- Proves that given 134 total toys, with 90 initially red, after removing 2 red toys, 
    the ratio of red to white toys is 2:1. -/
theorem toy_ratio_after_removal (total : ℕ) (initial_red : ℕ) (removed : ℕ) : 
  total = 134 → initial_red = 90 → removed = 2 →
  (initial_red - removed) / (total - initial_red) = 2 / 1 := by
sorry

end toy_ratio_after_removal_l1291_129184


namespace original_lemon_price_was_eight_l1291_129189

/-- The problem of determining the original lemon price --/
def lemon_price_problem (original_lemon_price : ℚ) : Prop :=
  let lemon_price_increase : ℚ := 4
  let grape_price_increase : ℚ := lemon_price_increase / 2
  let original_grape_price : ℚ := 7
  let num_lemons : ℕ := 80
  let num_grapes : ℕ := 140
  let total_revenue : ℚ := 2220
  let new_lemon_price : ℚ := original_lemon_price + lemon_price_increase
  let new_grape_price : ℚ := original_grape_price + grape_price_increase
  (num_lemons : ℚ) * new_lemon_price + (num_grapes : ℚ) * new_grape_price = total_revenue

/-- Theorem stating that the original lemon price was 8 --/
theorem original_lemon_price_was_eight :
  lemon_price_problem 8 := by
  sorry

end original_lemon_price_was_eight_l1291_129189


namespace restaurant_friends_l1291_129156

theorem restaurant_friends (pre_cooked wings_cooked wings_per_person : ℕ) :
  pre_cooked = 2 →
  wings_cooked = 25 →
  wings_per_person = 3 →
  (pre_cooked + wings_cooked) / wings_per_person = 9 :=
by
  sorry

end restaurant_friends_l1291_129156


namespace shelter_cats_l1291_129100

theorem shelter_cats (total_animals : ℕ) (dogs : ℕ) (cats : ℕ) : 
  total_animals = 1212 → dogs = 567 → total_animals = cats + dogs → cats = 645 := by
  sorry

end shelter_cats_l1291_129100


namespace waiter_new_customers_l1291_129164

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (customers_left : ℕ) 
  (remaining_customers : ℕ) 
  (final_total_customers : ℕ) : 
  initial_customers = 8 →
  customers_left = 3 →
  remaining_customers = 5 →
  final_total_customers = 104 →
  final_total_customers - remaining_customers = 99 :=
by sorry

end waiter_new_customers_l1291_129164


namespace circle_area_with_diameter_10_l1291_129180

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end circle_area_with_diameter_10_l1291_129180


namespace inequality_equivalence_l1291_129127

theorem inequality_equivalence (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) > 1 ↔ 1 < x ∧ x < 2 := by
sorry

end inequality_equivalence_l1291_129127


namespace union_of_M_and_N_l1291_129182

def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2} := by
  sorry

end union_of_M_and_N_l1291_129182


namespace combined_weight_theorem_l1291_129112

/-- Represents the elevator scenario with people and their weights -/
structure ElevatorScenario where
  initial_people : ℕ
  initial_avg_weight : ℝ
  new_avg_weights : List ℝ

/-- Calculates the combined weight of new people entering the elevator -/
def combined_weight_of_new_people (scenario : ElevatorScenario) : ℝ :=
  sorry

/-- Theorem stating the combined weight of new people in the given scenario -/
theorem combined_weight_theorem (scenario : ElevatorScenario) :
  scenario.initial_people = 6 →
  scenario.initial_avg_weight = 152 →
  scenario.new_avg_weights = [154, 153, 151] →
  combined_weight_of_new_people scenario = 447 := by
  sorry

#check combined_weight_theorem

end combined_weight_theorem_l1291_129112


namespace geometric_sequence_first_term_l1291_129108

/-- Given a geometric sequence where the third term is 12 and the fourth term is 16,
    prove that the first term is 27/4. -/
theorem geometric_sequence_first_term
  (a : ℚ) -- First term of the sequence
  (r : ℚ) -- Common ratio of the sequence
  (h1 : a * r^2 = 12) -- Third term is 12
  (h2 : a * r^3 = 16) -- Fourth term is 16
  : a = 27 / 4 :=
by sorry

end geometric_sequence_first_term_l1291_129108


namespace ratio_and_linear_equation_l1291_129145

theorem ratio_and_linear_equation (x y : ℚ) : 
  x / y = 4 → x = 18 - 3 * y → y = 18 / 7 := by
  sorry

end ratio_and_linear_equation_l1291_129145


namespace average_of_three_numbers_l1291_129115

theorem average_of_three_numbers (y : ℝ) : (15 + 25 + y) / 3 = 23 → y = 29 := by
  sorry

end average_of_three_numbers_l1291_129115


namespace sum_and_product_identities_l1291_129135

theorem sum_and_product_identities (a b : ℝ) 
  (sum_eq : a + b = 4) 
  (product_eq : a * b = 1) : 
  a^2 + b^2 = 14 ∧ (a - b)^2 = 12 := by
  sorry

end sum_and_product_identities_l1291_129135


namespace terror_arrangements_count_l1291_129185

/-- The number of unique arrangements of the letters in "TERROR" -/
def terror_arrangements : ℕ := 180

/-- The total number of letters in "TERROR" -/
def total_letters : ℕ := 6

/-- The number of R's in "TERROR" -/
def num_r : ℕ := 2

/-- The number of E's in "TERROR" -/
def num_e : ℕ := 2

/-- Theorem stating that the number of unique arrangements of the letters in "TERROR" is 180 -/
theorem terror_arrangements_count : 
  terror_arrangements = (Nat.factorial total_letters) / ((Nat.factorial num_r) * (Nat.factorial num_e)) :=
by sorry

end terror_arrangements_count_l1291_129185


namespace banks_revenue_is_500_l1291_129140

/-- Represents the revenue structure for Mr. Banks and Ms. Elizabeth -/
structure RevenueStructure where
  banks_investments : ℕ
  elizabeth_investments : ℕ
  elizabeth_revenue_per_investment : ℕ
  elizabeth_total_revenue_difference : ℕ

/-- Calculates Mr. Banks' revenue per investment given the revenue structure -/
def banks_revenue_per_investment (rs : RevenueStructure) : ℕ :=
  ((rs.elizabeth_investments * rs.elizabeth_revenue_per_investment) - rs.elizabeth_total_revenue_difference) / rs.banks_investments

/-- Theorem stating that Mr. Banks' revenue per investment is $500 given the specific conditions -/
theorem banks_revenue_is_500 (rs : RevenueStructure) 
  (h1 : rs.banks_investments = 8)
  (h2 : rs.elizabeth_investments = 5)
  (h3 : rs.elizabeth_revenue_per_investment = 900)
  (h4 : rs.elizabeth_total_revenue_difference = 500) :
  banks_revenue_per_investment rs = 500 := by
  sorry


end banks_revenue_is_500_l1291_129140
