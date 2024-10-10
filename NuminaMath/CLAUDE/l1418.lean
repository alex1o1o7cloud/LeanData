import Mathlib

namespace largest_non_expressible_l1418_141804

/-- A positive integer is composite if it has a proper divisor greater than 1. -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A function that checks if a number can be expressed as 42k + c, 
    where k is a positive integer and c is a positive composite integer. -/
def CanBeExpressed (n : ℕ) : Prop :=
  ∃ (k c : ℕ), k > 0 ∧ IsComposite c ∧ n = 42 * k + c

/-- The theorem stating that 215 is the largest positive integer that cannot be expressed
    as the sum of a positive integral multiple of 42 and a positive composite integer. -/
theorem largest_non_expressible : 
  (∀ n : ℕ, n > 215 → CanBeExpressed n) ∧ 
  (¬ CanBeExpressed 215) := by
  sorry

#check largest_non_expressible

end largest_non_expressible_l1418_141804


namespace fraction_simplification_l1418_141858

theorem fraction_simplification (y b : ℝ) : 
  (y + 2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := by
sorry

end fraction_simplification_l1418_141858


namespace alex_coin_distribution_l1418_141863

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := friends * (friends + 1) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ)
    (h1 : friends = 15)
    (h2 : initial_coins = 95) :
    min_additional_coins friends initial_coins = 25 := by
  sorry

end alex_coin_distribution_l1418_141863


namespace simplify_expression_l1418_141810

theorem simplify_expression (p q r : ℝ) 
  (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) : 
  (p - 7) / (9 - r) * (q - 8) / (7 - p) * (r - 9) / (8 - q) = -1 := by
  sorry

end simplify_expression_l1418_141810


namespace math_city_intersections_l1418_141850

/-- Represents a city with a number of straight, non-parallel streets -/
structure City where
  num_streets : ℕ
  streets_straight : Bool
  streets_non_parallel : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (city : City) : ℕ :=
  (city.num_streets * (city.num_streets - 1)) / 2

/-- Theorem: A city with 10 straight, non-parallel streets has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 ∧ c.streets_straight ∧ c.streets_non_parallel →
  max_intersections c = 45 := by
  sorry

end math_city_intersections_l1418_141850


namespace saree_price_calculation_l1418_141874

theorem saree_price_calculation (P : ℝ) : 
  P * (1 - 0.20) * (1 - 0.05) = 152 → P = 200 := by
  sorry

end saree_price_calculation_l1418_141874


namespace equation_root_l1418_141885

theorem equation_root : ∃ x : ℝ, 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ∧ 
  x = 31 := by
  sorry

end equation_root_l1418_141885


namespace triangle_problem_l1418_141865

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < π/2 ∧
  0 < t.B ∧ t.B < π/2 ∧
  0 < t.C ∧ t.C < π/2

def satisfiesSineLaw (t : Triangle) : Prop :=
  t.a / sin t.A = t.b / sin t.B ∧
  t.b / sin t.B = t.c / sin t.C

def satisfiesGivenCondition (t : Triangle) : Prop :=
  2 * t.a * sin t.B = t.b

-- The main theorem
theorem triangle_problem (t : Triangle)
  (h_acute : isAcute t)
  (h_sine_law : satisfiesSineLaw t)
  (h_condition : satisfiesGivenCondition t) :
  t.A = π/6 ∧
  (t.a = 6 ∧ t.b + t.c = 8 →
    1/2 * t.b * t.c * sin t.A = 14 - 7 * Real.sqrt 3) :=
sorry

end triangle_problem_l1418_141865


namespace rational_sum_theorem_l1418_141825

theorem rational_sum_theorem (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} → 
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end rational_sum_theorem_l1418_141825


namespace probability_theorem_l1418_141832

def total_balls : ℕ := 22
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 7
def yellow_balls : ℕ := 4
def balls_picked : ℕ := 3

def probability_at_least_two_red_not_blue : ℚ :=
  (Nat.choose red_balls 2 * (green_balls + yellow_balls) +
   Nat.choose red_balls 3) /
  Nat.choose total_balls balls_picked

theorem probability_theorem :
  probability_at_least_two_red_not_blue = 12 / 154 :=
sorry

end probability_theorem_l1418_141832


namespace stating_angle_edge_to_face_special_case_l1418_141882

/-- Represents a trihedral angle with vertex A and edges AB, AC, and AD -/
structure TrihedralAngle where
  BAC : ℝ  -- Angle between AB and AC
  CAD : ℝ  -- Angle between AC and AD
  BAD : ℝ  -- Angle between AB and AD

/-- 
Calculates the angle between edge AB and face ACD in a trihedral angle
given the measures of angles BAC, CAD, and BAD
-/
def angleEdgeToFace (t : TrihedralAngle) : ℝ :=
  sorry

/-- 
Theorem stating that for a trihedral angle with BAC = 45°, CAD = 90°, and BAD = 60°,
the angle between edge AB and face ACD is 30°
-/
theorem angle_edge_to_face_special_case :
  let t : TrihedralAngle := { BAC := Real.pi / 4, CAD := Real.pi / 2, BAD := Real.pi / 3 }
  angleEdgeToFace t = Real.pi / 6 := by
  sorry

end stating_angle_edge_to_face_special_case_l1418_141882


namespace square_sum_given_conditions_l1418_141886

theorem square_sum_given_conditions (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 4)
  (h2 : a + b + c = 17) : 
  a^2 + b^2 + c^2 = 281 := by sorry

end square_sum_given_conditions_l1418_141886


namespace leo_current_weight_l1418_141891

/-- Leo's current weight in pounds -/
def leo_weight : ℝ := 92

/-- Kendra's current weight in pounds -/
def kendra_weight : ℝ := 160 - leo_weight

/-- Theorem stating that Leo's current weight is 92 pounds -/
theorem leo_current_weight :
  (leo_weight + 10 = 1.5 * kendra_weight) ∧
  (leo_weight + kendra_weight = 160) →
  leo_weight = 92 := by
  sorry

end leo_current_weight_l1418_141891


namespace bird_count_l1418_141898

/-- The number of birds in a park -/
theorem bird_count (blackbirds_per_tree : ℕ) (num_trees : ℕ) (num_magpies : ℕ) :
  blackbirds_per_tree = 3 →
  num_trees = 7 →
  num_magpies = 13 →
  blackbirds_per_tree * num_trees + num_magpies = 34 := by
sorry


end bird_count_l1418_141898


namespace people_reached_in_day_l1418_141809

/-- The number of people reached after n hours of message spreading -/
def people_reached (n : ℕ) : ℕ :=
  2^(n+1) - 1

/-- Theorem stating the number of people reached in 24 hours -/
theorem people_reached_in_day : people_reached 24 = 2^24 - 1 := by
  sorry

#eval people_reached 24

end people_reached_in_day_l1418_141809


namespace p_recurrence_l1418_141829

/-- Probability of having a group of length k or more in n tosses of a symmetric coin -/
def p (n k : ℕ) : ℝ :=
  sorry

/-- The recurrence relation for p(n,k) -/
theorem p_recurrence (n k : ℕ) (h : k < n) :
  p n k = p (n-1) k - (1 / 2^k) * p (n-k) k + (1 / 2^k) :=
sorry

end p_recurrence_l1418_141829


namespace quadratic_roots_k_less_than_9_l1418_141861

theorem quadratic_roots_k_less_than_9 (k : ℝ) (h : k < 9) :
  (∃ x : ℝ, (k - 5) * x^2 - 2 * (k - 3) * x + k = 0) ∧
  ((∃! x : ℝ, (k - 5) * x^2 - 2 * (k - 3) * x + k = 0) ∨
   (∃ x y : ℝ, x ≠ y ∧ (k - 5) * x^2 - 2 * (k - 3) * x + k = 0 ∧
                      (k - 5) * y^2 - 2 * (k - 3) * y + k = 0)) :=
by sorry

end quadratic_roots_k_less_than_9_l1418_141861


namespace modular_inverse_of_5_mod_29_l1418_141897

theorem modular_inverse_of_5_mod_29 :
  ∃ a : ℕ, a ≤ 28 ∧ (5 * a) % 29 = 1 ∧ a = 6 := by
  sorry

end modular_inverse_of_5_mod_29_l1418_141897


namespace shoe_pairs_count_l1418_141811

theorem shoe_pairs_count (total_shoes : ℕ) (prob_same_color : ℚ) : 
  total_shoes = 16 → 
  prob_same_color = 1 / 15 → 
  (total_shoes / 2 : ℕ) = 8 := by
sorry

end shoe_pairs_count_l1418_141811


namespace bus_passengers_l1418_141843

theorem bus_passengers (men women : ℕ) : 
  women = men / 2 →
  men - 16 = women + 8 →
  men + women = 72 :=
by
  sorry

end bus_passengers_l1418_141843


namespace juhyes_money_l1418_141889

theorem juhyes_money (initial_money : ℝ) : 
  (1/3 : ℝ) * (3/4 : ℝ) * initial_money = 2500 → initial_money = 10000 := by
  sorry

end juhyes_money_l1418_141889


namespace rectangle_unique_symmetric_shape_l1418_141862

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | RegularPentagon

-- Define axisymmetry and central symmetry
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Parallelogram => false
  | Shape.Rectangle => true
  | Shape.RegularPentagon => true

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.RegularPentagon => false

-- Theorem statement
theorem rectangle_unique_symmetric_shape :
  ∀ s : Shape, isAxisymmetric s ∧ isCentrallySymmetric s ↔ s = Shape.Rectangle :=
by sorry

end rectangle_unique_symmetric_shape_l1418_141862


namespace only_earth_revolves_certain_l1418_141864

-- Define the type for events
inductive Event
| earth_revolves : Event
| shooter_hits_bullseye : Event
| three_suns_appear : Event
| red_light_encounter : Event

-- Define the property of being a certain event
def is_certain_event (e : Event) : Prop :=
  match e with
  | Event.earth_revolves => True
  | _ => False

-- Theorem stating that only the Earth revolving is a certain event
theorem only_earth_revolves_certain :
  ∀ e : Event, is_certain_event e ↔ e = Event.earth_revolves :=
sorry

end only_earth_revolves_certain_l1418_141864


namespace one_third_squared_times_one_eighth_l1418_141805

theorem one_third_squared_times_one_eighth : (1 / 3 : ℚ)^2 * (1 / 8 : ℚ) = 1 / 72 := by
  sorry

end one_third_squared_times_one_eighth_l1418_141805


namespace robert_reading_capacity_l1418_141806

def reading_speed : ℝ := 120
def pages_per_book : ℕ := 360
def available_time : ℝ := 10

def books_read : ℕ := 3

theorem robert_reading_capacity : 
  (reading_speed * available_time) / pages_per_book ≥ books_read ∧ 
  (reading_speed * available_time) / pages_per_book < books_read + 1 :=
sorry

end robert_reading_capacity_l1418_141806


namespace dinner_bill_calculation_l1418_141883

theorem dinner_bill_calculation 
  (appetizer_cost : ℝ) 
  (entree_cost : ℝ) 
  (dessert_cost : ℝ) 
  (tip_percentage : ℝ) 
  (h1 : appetizer_cost = 9)
  (h2 : entree_cost = 20)
  (h3 : dessert_cost = 11)
  (h4 : tip_percentage = 0.3) :
  appetizer_cost + 2 * entree_cost + dessert_cost + 
  (appetizer_cost + 2 * entree_cost + dessert_cost) * tip_percentage = 78 :=
by sorry

end dinner_bill_calculation_l1418_141883


namespace a_in_M_necessary_not_sufficient_for_a_in_N_l1418_141840

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
sorry

end a_in_M_necessary_not_sufficient_for_a_in_N_l1418_141840


namespace ornament_shop_profit_maximization_l1418_141881

/-- Ornament shop profit maximization problem -/
theorem ornament_shop_profit_maximization :
  ∀ (cost_A cost_B selling_price total_quantity : ℕ) 
    (min_B max_B_ratio discount_threshold discount_rate : ℕ),
  cost_A = 1400 →
  cost_B = 630 →
  cost_A = 2 * cost_B →
  selling_price = 15 →
  total_quantity = 600 →
  min_B = 390 →
  max_B_ratio = 4 →
  discount_threshold = 150 →
  discount_rate = 40 →
  ∃ (quantity_A quantity_B profit : ℕ),
    quantity_A + quantity_B = total_quantity ∧
    quantity_B ≥ min_B ∧
    quantity_B ≤ max_B_ratio * quantity_A ∧
    quantity_A = 210 ∧
    quantity_B = 390 ∧
    profit = 3630 ∧
    (∀ (other_quantity_A other_quantity_B other_profit : ℕ),
      other_quantity_A + other_quantity_B = total_quantity →
      other_quantity_B ≥ min_B →
      other_quantity_B ≤ max_B_ratio * other_quantity_A →
      other_profit ≤ profit) :=
by sorry

end ornament_shop_profit_maximization_l1418_141881


namespace milk_packets_returned_l1418_141852

/-- Given information about milk packets and their prices, prove the number of returned packets. -/
theorem milk_packets_returned (total : ℕ) (avg_price all_remaining returned : ℚ) :
  total = 5 ∧ 
  avg_price = 20 ∧ 
  all_remaining = 12 ∧ 
  returned = 32 →
  ∃ (x : ℕ), 
    x ≤ total ∧ 
    (total : ℚ) * avg_price = (total - x : ℚ) * all_remaining + (x : ℚ) * returned ∧
    x = 2 := by
  sorry

end milk_packets_returned_l1418_141852


namespace infinitely_many_m_with_1000_nonzero_bits_l1418_141812

def count_nonzero_bits (m : ℕ) : ℕ :=
  (m.bits.filter (· ≠ 0)).length

theorem infinitely_many_m_with_1000_nonzero_bits :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ count_nonzero_bits m = 1000 :=
by sorry

end infinitely_many_m_with_1000_nonzero_bits_l1418_141812


namespace min_value_on_negative_reals_l1418_141846

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

-- State the theorem
theorem min_value_on_negative_reals (a b : ℝ) :
  (∀ x > 0, f a b x ≤ 5) ∧ (∃ x > 0, f a b x = 5) →
  (∀ x < 0, f a b x ≥ -1) ∧ (∃ x < 0, f a b x = -1) :=
by sorry

end min_value_on_negative_reals_l1418_141846


namespace round_trip_speed_l1418_141857

/-- Given a round trip where:
    - The outbound journey is at speed p km/h
    - The return journey is at 3 km/h
    - The average speed is (24/q) km/h
    - p = 4
    Then q = 7 -/
theorem round_trip_speed (p q : ℝ) (hp : p = 4) : 
  (2 / ((1/p) + (1/3)) = 24/q) → q = 7 := by
  sorry

end round_trip_speed_l1418_141857


namespace geometric_sequence_property_l1418_141873

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_condition : a 1 * (a 8)^3 * a 15 = 243) :
  (a 9)^3 / a 11 = 9 := by
sorry

end geometric_sequence_property_l1418_141873


namespace trumpet_to_running_ratio_l1418_141853

/-- Proves that the ratio of time spent practicing trumpet to time spent running is 2:1 -/
theorem trumpet_to_running_ratio 
  (basketball_time : ℕ) 
  (trumpet_time : ℕ) 
  (h1 : basketball_time = 10)
  (h2 : trumpet_time = 40) :
  (trumpet_time : ℚ) / (2 * basketball_time) = 2 / 1 :=
by sorry

end trumpet_to_running_ratio_l1418_141853


namespace triangle_to_square_area_ratio_l1418_141817

/-- Represents a square divided into a 5x5 grid -/
structure GridSquare where
  side_length : ℝ
  small_square_count : ℕ
  small_square_count_eq : small_square_count = 5

/-- Represents a triangle within the GridSquare -/
structure Triangle where
  grid : GridSquare
  covered_squares : ℝ
  covered_squares_eq : covered_squares = 3.5

theorem triangle_to_square_area_ratio 
  (grid : GridSquare) 
  (triangle : Triangle) 
  (h_triangle : triangle.grid = grid) :
  (triangle.covered_squares * (grid.side_length / grid.small_square_count)^2) / 
  (grid.side_length^2) = 7 / 50 := by
  sorry

end triangle_to_square_area_ratio_l1418_141817


namespace roots_sum_greater_than_twice_sqrt_a_l1418_141818

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

theorem roots_sum_greater_than_twice_sqrt_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > Real.exp 1) 
  (hx₁ : f a x₁ = 0) 
  (hx₂ : f a x₂ = 0) 
  (hx_dist : x₁ ≠ x₂) : 
  x₁ + x₂ > 2 * Real.sqrt a := by
sorry

end roots_sum_greater_than_twice_sqrt_a_l1418_141818


namespace tan_equation_solution_l1418_141816

open Set Real

-- Define the set of angles that satisfy the conditions
def solution_set : Set ℝ := {x | 0 ≤ x ∧ x < π ∧ tan (4 * x - π / 4) = 1}

-- State the theorem
theorem tan_equation_solution :
  solution_set = {π/8, 3*π/8, 5*π/8, 7*π/8} := by sorry

end tan_equation_solution_l1418_141816


namespace books_difference_alicia_ian_l1418_141841

/-- Represents a student in the book reading contest -/
structure Student where
  name : String
  booksRead : Nat

/-- Represents the book reading contest -/
structure BookReadingContest where
  students : Finset Student
  alicia : Student
  ian : Student
  aliciaMostBooks : ∀ s ∈ students, s.booksRead ≤ alicia.booksRead
  ianFewestBooks : ∀ s ∈ students, ian.booksRead ≤ s.booksRead
  aliciaInContest : alicia ∈ students
  ianInContest : ian ∈ students
  contestSize : students.card = 8
  aliciaBooksRead : alicia.booksRead = 8
  ianBooksRead : ian.booksRead = 1

/-- The difference in books read between Alicia and Ian is 7 -/
theorem books_difference_alicia_ian (contest : BookReadingContest) :
  contest.alicia.booksRead - contest.ian.booksRead = 7 := by
  sorry

end books_difference_alicia_ian_l1418_141841


namespace crayon_difference_l1418_141826

/-- Given an initial number of crayons, the number given away, and the number lost,
    prove that the difference between lost and given away is their subtraction. -/
theorem crayon_difference (initial given lost : ℕ) : lost - given = lost - given := by
  sorry

end crayon_difference_l1418_141826


namespace sqrt_86400_simplified_l1418_141838

theorem sqrt_86400_simplified : Real.sqrt 86400 = 120 * Real.sqrt 6 := by
  sorry

end sqrt_86400_simplified_l1418_141838


namespace infinitely_many_powers_of_two_in_floor_sqrt_two_n_l1418_141814

theorem infinitely_many_powers_of_two_in_floor_sqrt_two_n : 
  ∀ m : ℕ, ∃ n k : ℕ, n > m ∧ k > 0 ∧ ⌊Real.sqrt 2 * n⌋ = 2^k :=
sorry

end infinitely_many_powers_of_two_in_floor_sqrt_two_n_l1418_141814


namespace kayla_bought_fifteen_items_l1418_141888

/-- Represents the number of chocolate bars bought by Theresa -/
def theresa_chocolate_bars : ℕ := 12

/-- Represents the number of soda cans bought by Theresa -/
def theresa_soda_cans : ℕ := 18

/-- Represents the ratio of items bought by Theresa compared to Kayla -/
def theresa_to_kayla_ratio : ℕ := 2

/-- Calculates the total number of items bought by Kayla -/
def kayla_total_items : ℕ := 
  (theresa_chocolate_bars / theresa_to_kayla_ratio) + 
  (theresa_soda_cans / theresa_to_kayla_ratio)

/-- Theorem stating that Kayla bought 15 items in total -/
theorem kayla_bought_fifteen_items : kayla_total_items = 15 := by
  sorry

end kayla_bought_fifteen_items_l1418_141888


namespace gas_volume_ranking_l1418_141839

-- Define the regions
inductive Region
| West
| NonWest
| Russia

-- Define the gas volume per capita for each region
def gas_volume (r : Region) : ℝ :=
  match r with
  | Region.West => 21428
  | Region.NonWest => 26848.55
  | Region.Russia => 302790.13

-- Theorem to prove the ranking
theorem gas_volume_ranking :
  gas_volume Region.Russia > gas_volume Region.NonWest ∧
  gas_volume Region.NonWest > gas_volume Region.West :=
by sorry

end gas_volume_ranking_l1418_141839


namespace range_of_a_l1418_141836

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y + 3 = x * y → 
    (x + y)^2 - a * (x + y) + 1 ≥ 0) ↔ 
  a ≤ 37 / 6 :=
sorry

end range_of_a_l1418_141836


namespace relationship_abc_l1418_141831

theorem relationship_abc : 
  let a := (3/4)^(2/3)
  let b := (2/3)^(3/4)
  let c := Real.log (4/3) / Real.log (2/3)
  a > b ∧ b > c := by sorry

end relationship_abc_l1418_141831


namespace distance_is_600_km_l1418_141834

/-- The distance between the starting points of two persons traveling towards each other -/
def distance_between_starting_points (speed1 speed2 : ℝ) (travel_time : ℝ) : ℝ :=
  (speed1 + speed2) * travel_time

/-- Theorem stating that the distance between starting points is 600 km -/
theorem distance_is_600_km (speed1 speed2 travel_time : ℝ) 
  (h1 : speed1 = 70)
  (h2 : speed2 = 80)
  (h3 : travel_time = 4) :
  distance_between_starting_points speed1 speed2 travel_time = 600 := by
  sorry

#check distance_is_600_km

end distance_is_600_km_l1418_141834


namespace pigs_joined_l1418_141870

theorem pigs_joined (initial_pigs final_pigs : ℕ) 
  (h1 : initial_pigs = 64)
  (h2 : final_pigs = 86) :
  final_pigs - initial_pigs = 22 :=
by sorry

end pigs_joined_l1418_141870


namespace geometric_sequence_formula_l1418_141815

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a (n + 1) > a n) →
  a 5 ^ 2 = a 10 →
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) →
  ∃ c, ∀ n, a n = c * 2^n :=
sorry

end geometric_sequence_formula_l1418_141815


namespace tax_savings_proof_l1418_141802

def initial_tax_rate : ℝ := 0.46
def new_tax_rate : ℝ := 0.32
def annual_income : ℝ := 36000

def differential_savings : ℝ :=
  annual_income * initial_tax_rate - annual_income * new_tax_rate

theorem tax_savings_proof :
  differential_savings = 5040 := by
  sorry

end tax_savings_proof_l1418_141802


namespace truck_gas_ratio_l1418_141856

/-- Proves the ratio of gas in a truck's tank to its total capacity before filling -/
theorem truck_gas_ratio (truck_capacity car_capacity added_gas : ℚ) 
  (h1 : truck_capacity = 20)
  (h2 : car_capacity = 12)
  (h3 : added_gas = 18)
  (h4 : (1/3) * car_capacity + added_gas = truck_capacity + car_capacity) :
  (truck_capacity - ((1/3) * car_capacity + added_gas - car_capacity)) / truck_capacity = 1/2 := by
  sorry

end truck_gas_ratio_l1418_141856


namespace partnership_profit_b_profit_calculation_l1418_141803

/-- Profit calculation in a partnership --/
theorem partnership_profit (a_investment b_investment : ℕ) 
  (a_period b_period : ℕ) (total_profit : ℕ) : ℕ :=
  let a_share := a_investment * a_period
  let b_share := b_investment * b_period
  let total_share := a_share + b_share
  let b_profit := (b_share * total_profit) / total_share
  b_profit

/-- B's profit in the given partnership scenario --/
theorem b_profit_calculation : 
  partnership_profit 3 1 2 1 31500 = 4500 := by
  sorry

end partnership_profit_b_profit_calculation_l1418_141803


namespace factorization_problem_l1418_141890

theorem factorization_problem (x y : ℝ) : (y + 2*x)^2 - (x + 2*y)^2 = 3*(x + y)*(x - y) := by
  sorry

end factorization_problem_l1418_141890


namespace cubic_sum_ratio_l1418_141807

theorem cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 15)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 18 := by
sorry

end cubic_sum_ratio_l1418_141807


namespace triangle_properties_l1418_141879

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.A * Real.cos t.C + 1 = 2 * Real.sin t.A * Real.sin t.C)
  (h2 : t.a + t.c = 3 * Real.sqrt 3 / 2)
  (h3 : t.b = Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = 5 * Real.sqrt 3 / 16 := by
  sorry

end

end triangle_properties_l1418_141879


namespace mary_stickers_remaining_l1418_141877

theorem mary_stickers_remaining (initial_stickers : ℕ) 
                                 (front_page_stickers : ℕ) 
                                 (other_pages : ℕ) 
                                 (stickers_per_other_page : ℕ) 
                                 (h1 : initial_stickers = 89)
                                 (h2 : front_page_stickers = 3)
                                 (h3 : other_pages = 6)
                                 (h4 : stickers_per_other_page = 7) : 
  initial_stickers - (front_page_stickers + other_pages * stickers_per_other_page) = 44 := by
  sorry

end mary_stickers_remaining_l1418_141877


namespace f_properties_l1418_141896

def f (a x : ℝ) : ℝ := x * |x - a|

theorem f_properties (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x + x < f a y + y ↔ -1 ≤ a ∧ a ≤ 1) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f a x < 1 ↔ 3/2 < a ∧ a < 2) ∧
  (a ≥ 2 →
    (∀ x : ℝ, x ∈ Set.Icc 2 4 → 
      (a > 8 → f a x ∈ Set.Icc (2*a-4) (4*a-16)) ∧
      (4 ≤ a ∧ a < 6 → f a x ∈ Set.Icc (4*a-16) (a^2/4)) ∧
      (6 ≤ a ∧ a ≤ 8 → f a x ∈ Set.Icc (2*a-4) (a^2/4)) ∧
      (2 ≤ a ∧ a < 10/3 → f a x ∈ Set.Icc 0 (16-4*a)) ∧
      (10/3 ≤ a ∧ a < 4 → f a x ∈ Set.Icc 0 (2*a-4)))) :=
by sorry

end f_properties_l1418_141896


namespace danielle_age_l1418_141833

/-- Given the ages of Anna, Ben, Carlos, and Danielle, prove that Danielle is 22 years old. -/
theorem danielle_age (anna ben carlos danielle : ℕ)
  (h1 : anna = ben - 4)
  (h2 : ben = carlos + 3)
  (h3 : danielle = carlos + 6)
  (h4 : anna = 15) :
  danielle = 22 := by
sorry

end danielle_age_l1418_141833


namespace complex_cube_equality_l1418_141813

theorem complex_cube_equality (c d : ℝ) (h : d > 0) :
  (c + d * Complex.I) ^ 3 = (c - d * Complex.I) ^ 3 ↔ d / c = Real.sqrt 3 :=
sorry

end complex_cube_equality_l1418_141813


namespace complex_number_location_l1418_141880

theorem complex_number_location :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  let w : ℂ := z / (1 + Complex.I)
  (w.re < 0) ∧ (w.im < 0) := by
  sorry

end complex_number_location_l1418_141880


namespace total_parents_is_fourteen_l1418_141828

/-- Represents the field trip to the zoo --/
structure FieldTrip where
  fifth_graders : ℕ
  sixth_graders : ℕ
  seventh_graders : ℕ
  teachers : ℕ
  buses : ℕ
  seats_per_bus : ℕ

/-- Calculates the total number of parents on the field trip --/
def total_parents (trip : FieldTrip) : ℕ :=
  trip.buses * trip.seats_per_bus - (trip.fifth_graders + trip.sixth_graders + trip.seventh_graders + trip.teachers)

/-- Theorem stating that the total number of parents on the trip is 14 --/
theorem total_parents_is_fourteen (trip : FieldTrip) 
  (h1 : trip.fifth_graders = 109)
  (h2 : trip.sixth_graders = 115)
  (h3 : trip.seventh_graders = 118)
  (h4 : trip.teachers = 4)
  (h5 : trip.buses = 5)
  (h6 : trip.seats_per_bus = 72) :
  total_parents trip = 14 := by
  sorry

#eval total_parents { fifth_graders := 109, sixth_graders := 115, seventh_graders := 118, teachers := 4, buses := 5, seats_per_bus := 72 }

end total_parents_is_fourteen_l1418_141828


namespace expression_simplification_and_evaluation_l1418_141830

theorem expression_simplification_and_evaluation :
  let f (x : ℚ) := (x^2 - 4) / (x^2 - 4*x + 4) + (x / (x^2 - x)) / ((x - 2) / (x - 1))
  f (-1) = -2/3 := by
  sorry

end expression_simplification_and_evaluation_l1418_141830


namespace school_fee_calculation_l1418_141854

def mother_contribution : ℕ := 2 * 100 + 1 * 50 + 5 * 20 + 3 * 10 + 4 * 5
def father_contribution : ℕ := 3 * 100 + 4 * 50 + 2 * 20 + 1 * 10 + 6 * 5

theorem school_fee_calculation : mother_contribution + father_contribution = 980 := by
  sorry

end school_fee_calculation_l1418_141854


namespace no_real_solution_ffx_l1418_141842

/-- A second-degree polynomial function -/
def SecondDegreePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- No real solution for f(x) = x -/
def NoRealSolutionForFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

/-- No real solution for f(f(x)) = x -/
def NoRealSolutionForFFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) ≠ x

theorem no_real_solution_ffx 
  (f : ℝ → ℝ) 
  (h1 : SecondDegreePolynomial f) 
  (h2 : NoRealSolutionForFX f) : 
  NoRealSolutionForFFX f :=
sorry

end no_real_solution_ffx_l1418_141842


namespace farm_animals_l1418_141868

theorem farm_animals (pigs : ℕ) (cows : ℕ) (goats : ℕ) : 
  cows = 2 * pigs - 3 →
  goats = cows + 6 →
  pigs + cows + goats = 50 →
  pigs = 10 := by
sorry

end farm_animals_l1418_141868


namespace perpendicular_vectors_x_coord_l1418_141887

/-- Given two vectors a and b in ℝ², prove that if a is perpendicular to b,
    then the x-coordinate of a is -2/3. -/
theorem perpendicular_vectors_x_coord
  (a b : ℝ × ℝ)
  (h1 : a.1 = x ∧ a.2 = x + 1)
  (h2 : b = (1, 2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -2/3 := by
  sorry

end perpendicular_vectors_x_coord_l1418_141887


namespace population_reaches_target_in_2095_l1418_141827

/-- The initial population of the island -/
def initial_population : ℕ := 450

/-- The year when the population count starts -/
def initial_year : ℕ := 2020

/-- The number of years it takes for the population to triple -/
def tripling_period : ℕ := 25

/-- The target population we want to reach or exceed -/
def target_population : ℕ := 10800

/-- Function to calculate the population after a given number of periods -/
def population_after_periods (periods : ℕ) : ℕ :=
  initial_population * (3 ^ periods)

/-- Function to calculate the year after a given number of periods -/
def year_after_periods (periods : ℕ) : ℕ :=
  initial_year + (periods * tripling_period)

/-- Theorem stating that 2095 is the closest year to when the population reaches or exceeds the target -/
theorem population_reaches_target_in_2095 :
  ∃ (n : ℕ), 
    (population_after_periods n ≥ target_population) ∧
    (population_after_periods (n - 1) < target_population) ∧
    (year_after_periods n = 2095) :=
  sorry

end population_reaches_target_in_2095_l1418_141827


namespace x_plus_y_equals_five_l1418_141869

theorem x_plus_y_equals_five (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 := by
  sorry

end x_plus_y_equals_five_l1418_141869


namespace set_intersection_and_union_l1418_141899

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem set_intersection_and_union :
  ∃ x : ℝ, (B x ∩ A x = {9}) ∧ 
           (x = -3) ∧ 
           (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end set_intersection_and_union_l1418_141899


namespace specific_rhombus_area_l1418_141894

/-- Represents a rhombus with given properties -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  diagonals_perpendicular_bisectors : Bool

/-- Calculates the area of a rhombus with the given properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem stating the area of a specific rhombus -/
theorem specific_rhombus_area :
  let r : Rhombus := {
    side_length := Real.sqrt 165,
    diagonal_difference := 10,
    diagonals_perpendicular_bisectors := true
  }
  rhombus_area r = 305 / 4 := by sorry

end specific_rhombus_area_l1418_141894


namespace slope_equals_half_implies_y_eleven_l1418_141845

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is 1/2, then the y-coordinate of Q is 11. -/
theorem slope_equals_half_implies_y_eleven (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁ = -3 → y₁ = 7 → x₂ = 5 → 
  (y₂ - y₁) / (x₂ - x₁) = 1/2 →
  y₂ = 11 := by
  sorry

#check slope_equals_half_implies_y_eleven

end slope_equals_half_implies_y_eleven_l1418_141845


namespace sticker_distribution_l1418_141835

/-- The number of ways to distribute indistinguishable objects among distinct containers -/
def distribute (objects : ℕ) (containers : ℕ) : ℕ :=
  Nat.choose (objects + containers - 1) (containers - 1)

/-- Theorem: Distributing 10 indistinguishable stickers among 5 distinct sheets of paper -/
theorem sticker_distribution : distribute 10 5 = 1001 := by
  sorry

end sticker_distribution_l1418_141835


namespace pencil_count_l1418_141893

theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / pencils = 5 / 6 →
  pencils = pens + 7 →
  pencils = 42 := by
sorry

end pencil_count_l1418_141893


namespace quadratic_equation_solution_l1418_141844

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 7 ∧ x₁^2 - 4*x₁ + 7 = 10) ∧
  (x₂ = 2 - Real.sqrt 7 ∧ x₂^2 - 4*x₂ + 7 = 10) := by
  sorry

end quadratic_equation_solution_l1418_141844


namespace partnership_profit_l1418_141876

/-- Calculates the total profit of a partnership given the investments and one partner's share of the profit. -/
def calculate_total_profit (a_investment b_investment c_investment c_profit : ℕ) : ℕ :=
  let ratio_sum := (a_investment / (a_investment.gcd b_investment).gcd c_investment) +
                   (b_investment / (a_investment.gcd b_investment).gcd c_investment) +
                   (c_investment / (a_investment.gcd b_investment).gcd c_investment)
  let c_ratio := c_investment / (a_investment.gcd b_investment).gcd c_investment
  (ratio_sum * c_profit) / c_ratio

/-- The total profit of the partnership is 80000 given the investments and C's profit share. -/
theorem partnership_profit :
  calculate_total_profit 27000 72000 81000 36000 = 80000 := by
  sorry

end partnership_profit_l1418_141876


namespace function_identity_l1418_141808

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ a x : ℝ, a < x ∧ x < a + 100 → a ≤ f x ∧ f x ≤ a + 100) : 
  ∀ x : ℝ, f x = x := by
  sorry

end function_identity_l1418_141808


namespace remainder_is_six_l1418_141892

/-- The divisor polynomial -/
def divisor (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

/-- The dividend polynomial -/
def dividend (x : ℂ) : ℂ := x^60 + x^48 + x^36 + x^24 + x^12 + 1

/-- Theorem stating that the remainder is 6 -/
theorem remainder_is_six : ∃ q : ℂ → ℂ, ∀ x : ℂ, 
  dividend x = (divisor x) * (q x) + 6 := by
  sorry

end remainder_is_six_l1418_141892


namespace train_speed_increase_time_l1418_141866

/-- The speed equation for a subway train -/
def speed_equation (s : ℝ) : ℝ := s^2 + 2*s

/-- Theorem: The time when the train's speed increases by 39 km/h from its speed at 4 seconds is 7 seconds -/
theorem train_speed_increase_time : 
  ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 7 ∧ 
  speed_equation s = speed_equation 4 + 39 ∧
  s = 7 := by
  sorry

#check train_speed_increase_time

end train_speed_increase_time_l1418_141866


namespace gcd_n4_plus_27_and_n_plus_3_l1418_141822

theorem gcd_n4_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^4 + 27) (n + 3) = if n % 3 = 0 then 3 else 1 := by
  sorry

end gcd_n4_plus_27_and_n_plus_3_l1418_141822


namespace consecutive_even_squares_divisibility_l1418_141824

theorem consecutive_even_squares_divisibility (n : ℤ) : 
  ∃ (k : ℤ), (4 * n ^ 2 + (4 * n ^ 2 + 8 * n + 4) + (4 * n ^ 2 + 16 * n + 16)) = 4 * k ∧
  ∃ (m : ℤ), (4 * n ^ 2 + (4 * n ^ 2 + 8 * n + 4) + (4 * n ^ 2 + 16 * n + 16)) ≠ 7 * m :=
by sorry

end consecutive_even_squares_divisibility_l1418_141824


namespace blueberry_earnings_relationship_l1418_141855

/-- Represents the relationship between blueberry picking amount and earnings --/
def blueberry_earnings (x : ℝ) : ℝ × ℝ :=
  let y₁ := 60 + 30 * 0.6 * x
  let y₂ := 10 * 30 + 30 * 0.5 * (x - 10)
  (y₁, y₂)

/-- Theorem stating the relationship between y₁, y₂, and x when x > 10 --/
theorem blueberry_earnings_relationship (x : ℝ) (h : x > 10) :
  let (y₁, y₂) := blueberry_earnings x
  y₁ = 60 + 18 * x ∧ y₂ = 150 + 15 * x :=
by sorry

end blueberry_earnings_relationship_l1418_141855


namespace simplify_expression_l1418_141849

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4)) =
  (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 := by
  sorry

end simplify_expression_l1418_141849


namespace vector_parallel_coordinates_l1418_141884

/-- Given two vectors a and b in ℝ², prove that if |a| = 2√5, b = (1,2), and a is parallel to b,
    then a = (2,4) or a = (-2,-4) -/
theorem vector_parallel_coordinates (a b : ℝ × ℝ) :
  (norm a = 2 * Real.sqrt 5) →
  (b = (1, 2)) →
  (∃ (k : ℝ), a = k • b) →
  (a = (2, 4) ∨ a = (-2, -4)) :=
sorry

end vector_parallel_coordinates_l1418_141884


namespace equation_solution_l1418_141800

theorem equation_solution :
  ∃ x : ℝ, (3 / (x - 2) = 2 / (x - 1)) ∧ (x = -1) :=
by
  sorry

end equation_solution_l1418_141800


namespace norm_photos_difference_l1418_141859

-- Define the number of photos taken by each photographer
variable (L M N : ℕ)

-- Define the conditions from the problem
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := ∃ X, N = 2 * L + X
def condition3 (N : ℕ) : Prop := N = 110

-- State the theorem
theorem norm_photos_difference (L M N : ℕ) 
  (h1 : condition1 L M N) (h2 : condition2 L N) (h3 : condition3 N) : 
  ∃ X, N = 2 * L + X ∧ X = 110 - 2 * L :=
sorry

end norm_photos_difference_l1418_141859


namespace divisors_of_36_count_divisors_of_36_l1418_141875

theorem divisors_of_36 : Finset Int → Prop :=
  fun s => ∀ n : Int, n ∈ s ↔ 36 % n = 0

theorem count_divisors_of_36 : 
  ∃ s : Finset Int, divisors_of_36 s ∧ s.card = 18 :=
sorry

end divisors_of_36_count_divisors_of_36_l1418_141875


namespace hyperbola_eccentricity_l1418_141819

/-- The eccentricity of a hyperbola whose focus coincides with the focus of a specific parabola -/
theorem hyperbola_eccentricity (a : ℝ) : 
  let parabola := {(x, y) : ℝ × ℝ | x^2 = -4 * Real.sqrt 5 * y}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a + y^2 / 4 = 1}
  let parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
  ∃ (c : ℝ), c > 0 ∧ (c, 0) ∈ hyperbola ∧ (-c, 0) ∈ hyperbola ∧ 
    (parabola_focus ∈ hyperbola → (Real.sqrt 5) / 2 = c / 2) :=
by sorry

end hyperbola_eccentricity_l1418_141819


namespace convex_polygon_angle_theorem_l1418_141820

theorem convex_polygon_angle_theorem (n : ℕ) (x : ℝ) :
  n ≥ 3 →
  x > 0 →
  x < 180 →
  (n : ℝ) * 180 - 3 * x = 3330 + 180 * 2 →
  x = 54 := by
  sorry

end convex_polygon_angle_theorem_l1418_141820


namespace percentage_of_450_to_325x_l1418_141823

theorem percentage_of_450_to_325x (x : ℝ) (h : x ≠ 0) :
  (450 : ℝ) / (325 * x) * 100 = 138.46153846 / x :=
sorry

end percentage_of_450_to_325x_l1418_141823


namespace roots_sum_relation_l1418_141860

theorem roots_sum_relation (a b c d : ℝ) : 
  (∀ x, x^2 + c*x + d = 0 ↔ x = a ∨ x = b) → a + b + c = 0 := by
  sorry

end roots_sum_relation_l1418_141860


namespace tom_bonus_percentage_l1418_141851

/-- Calculates the percentage of bonus points per customer served -/
def bonus_percentage (customers_per_hour : ℕ) (hours_worked : ℕ) (total_bonus_points : ℕ) : ℚ :=
  (total_bonus_points : ℚ) / ((customers_per_hour * hours_worked) : ℚ) * 100

/-- Proves that the bonus percentage for Tom is 20% -/
theorem tom_bonus_percentage :
  bonus_percentage 10 8 16 = 20 := by
  sorry

#eval bonus_percentage 10 8 16

end tom_bonus_percentage_l1418_141851


namespace simplify_power_of_power_l1418_141871

theorem simplify_power_of_power (x : ℝ) : (5 * x^2)^4 = 625 * x^8 := by
  sorry

end simplify_power_of_power_l1418_141871


namespace consecutive_integers_square_sum_l1418_141848

theorem consecutive_integers_square_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x^2 + (x + 1)^2 = 1625 := by
  sorry

end consecutive_integers_square_sum_l1418_141848


namespace ellipse_x_intercept_l1418_141801

/-- Definition of an ellipse with given foci and a point on it -/
def is_ellipse (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.2 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f1.1 - f2.1)^2 + (f1.2 - f2.2)^2)
  d1 + d2 = 2 * Real.sqrt ((c/2)^2 + ((d1 - d2)/2)^2)

/-- The ellipse intersects the x-axis at (0, 0) -/
def intersects_origin (f1 f2 : ℝ × ℝ) : Prop :=
  is_ellipse f1 f2 (0, 0)

/-- Theorem: For an ellipse with foci at (0, 3) and (4, 0) that intersects
    the x-axis at (0, 0), the other x-intercept is at (56/11, 0) -/
theorem ellipse_x_intercept :
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  intersects_origin f1 f2 →
  is_ellipse f1 f2 (56/11, 0) ∧
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 56/11 → ¬is_ellipse f1 f2 (x, 0) := by
  sorry

end ellipse_x_intercept_l1418_141801


namespace bullet_speed_difference_l1418_141847

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in bullet speed when fired in the same direction as the horse versus the opposite direction -/
def speed_difference : ℝ := (bullet_speed + horse_speed) - (bullet_speed - horse_speed)

theorem bullet_speed_difference :
  speed_difference = 40 :=
by sorry

end bullet_speed_difference_l1418_141847


namespace sum_lower_bound_l1418_141872

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b + 3) :
  6 ≤ a + b := by
  sorry

end sum_lower_bound_l1418_141872


namespace cryptarithm_solution_l1418_141895

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Digit
  b : Digit
  c : Digit
  d : Digit

def FourDigitNumber.value (n : FourDigitNumber) : ℕ :=
  1000 * n.a.val + 100 * n.b.val + 10 * n.c.val + n.d.val

def adjacent (x y : Digit) : Prop :=
  x.val + 1 = y.val ∨ y.val + 1 = x.val

theorem cryptarithm_solution :
  ∃! (n : FourDigitNumber),
    adjacent n.a n.c
    ∧ (n.b.val + 2 = n.d.val ∨ n.d.val + 2 = n.b.val)
    ∧ (∃ (e f g h i j : Digit),
        g.val * 10 + h.val = 19
        ∧ f.val + j.val = 14
        ∧ e.val + i.val = 10
        ∧ n.value = 5240) := by
  sorry

end cryptarithm_solution_l1418_141895


namespace greendale_points_greendale_points_equals_130_l1418_141867

/-- Calculates the total points for Greendale High School in a basketball tournament --/
theorem greendale_points (roosevelt_first_game : ℕ) (bonus : ℕ) (difference : ℕ) : ℕ :=
  let roosevelt_second_game := roosevelt_first_game / 2
  let roosevelt_third_game := roosevelt_second_game * 3
  let roosevelt_total := roosevelt_first_game + roosevelt_second_game + roosevelt_third_game + bonus
  roosevelt_total - difference

/-- Proves that Greendale High School's total points equal 130 --/
theorem greendale_points_equals_130 : greendale_points 30 50 10 = 130 := by
  sorry

end greendale_points_greendale_points_equals_130_l1418_141867


namespace birds_on_fence_l1418_141878

theorem birds_on_fence (initial_birds : ℝ) (birds_flew_away : ℝ) (remaining_birds : ℝ) : 
  initial_birds = 12.0 → birds_flew_away = 8.0 → remaining_birds = initial_birds - birds_flew_away → remaining_birds = 4.0 := by
  sorry

end birds_on_fence_l1418_141878


namespace min_m_value_x_range_l1418_141821

-- Define the variables and conditions
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a + b = 1)

-- Theorem 1: Minimum value of m
theorem min_m_value : 
  (∀ m : ℝ, a * b ≤ m → m ≥ 1/4) ∧ 
  (∃ m : ℝ, m = 1/4 ∧ a * b ≤ m) :=
sorry

-- Theorem 2: Range of x
theorem x_range : 
  (∀ x : ℝ, 4/a + 1/b ≥ |2*x - 1| - |x + 2|) ↔ 
  (∀ x : ℝ, x ∈ Set.Icc (-6) 12) :=
sorry

end min_m_value_x_range_l1418_141821


namespace factorial_ratio_l1418_141837

-- Define the factorial operation
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio : (factorial 50) / (factorial 48) = 2450 := by
  sorry

end factorial_ratio_l1418_141837
