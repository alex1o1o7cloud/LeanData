import Mathlib

namespace robins_full_pages_l2096_209647

/-- The number of full pages in a photo album -/
def full_pages (total_photos : ℕ) (photos_per_page : ℕ) : ℕ :=
  total_photos / photos_per_page

/-- Theorem: Robin's photo album has 181 full pages -/
theorem robins_full_pages :
  full_pages 2176 12 = 181 := by
  sorry

end robins_full_pages_l2096_209647


namespace unoccupied_business_seats_count_l2096_209667

/-- Represents the seating configuration and occupancy of an airplane. -/
structure AirplaneSeating where
  firstClassSeats : ℕ
  businessClassSeats : ℕ
  economyClassSeats : ℕ
  firstClassOccupied : ℕ
  economyClassOccupied : ℕ
  businessAndFirstOccupied : ℕ

/-- Calculates the number of unoccupied seats in business class. -/
def unoccupiedBusinessSeats (a : AirplaneSeating) : ℕ :=
  a.businessClassSeats - (a.businessAndFirstOccupied - a.firstClassOccupied)

/-- Theorem stating the number of unoccupied seats in business class. -/
theorem unoccupied_business_seats_count
  (a : AirplaneSeating)
  (h1 : a.firstClassSeats = 10)
  (h2 : a.businessClassSeats = 30)
  (h3 : a.economyClassSeats = 50)
  (h4 : a.economyClassOccupied = a.economyClassSeats / 2)
  (h5 : a.businessAndFirstOccupied = a.economyClassOccupied)
  (h6 : a.firstClassOccupied = 3) :
  unoccupiedBusinessSeats a = 8 := by
  sorry

#eval unoccupiedBusinessSeats {
  firstClassSeats := 10,
  businessClassSeats := 30,
  economyClassSeats := 50,
  firstClassOccupied := 3,
  economyClassOccupied := 25,
  businessAndFirstOccupied := 25
}

end unoccupied_business_seats_count_l2096_209667


namespace cyclist_distance_l2096_209609

theorem cyclist_distance (travel_time : ℝ) (car_distance : ℝ) (speed_difference : ℝ) :
  travel_time = 8 →
  car_distance = 48 →
  speed_difference = 5 →
  let car_speed := car_distance / travel_time
  let cyclist_speed := car_speed - speed_difference
  cyclist_speed * travel_time = 8 := by sorry

end cyclist_distance_l2096_209609


namespace solve_group_size_l2096_209626

def group_size_problem (n : ℕ) : Prop :=
  let weight_increase_per_person : ℚ := 5/2
  let weight_difference : ℕ := 20
  (weight_difference : ℚ) = n * weight_increase_per_person

theorem solve_group_size : ∃ n : ℕ, group_size_problem n ∧ n = 8 := by
  sorry

end solve_group_size_l2096_209626


namespace red_bus_length_l2096_209629

theorem red_bus_length 
  (red_bus orange_car yellow_bus : ℝ) 
  (h1 : red_bus = 4 * orange_car) 
  (h2 : orange_car = yellow_bus / 3.5) 
  (h3 : red_bus = yellow_bus + 6) : 
  red_bus = 48 := by
sorry

end red_bus_length_l2096_209629


namespace rachel_problem_solving_time_l2096_209623

/-- The number of minutes Rachel spent solving math problems before bed -/
def minutes_before_bed : ℕ := 12

/-- The number of problems Rachel solved per minute before bed -/
def problems_per_minute : ℕ := 5

/-- The number of problems Rachel solved the next day -/
def problems_next_day : ℕ := 16

/-- The total number of problems Rachel solved -/
def total_problems : ℕ := 76

/-- Theorem stating that Rachel spent 12 minutes solving problems before bed -/
theorem rachel_problem_solving_time :
  minutes_before_bed * problems_per_minute + problems_next_day = total_problems :=
by sorry

end rachel_problem_solving_time_l2096_209623


namespace calculate_expression_l2096_209683

theorem calculate_expression : (1/2)⁻¹ + |Real.sqrt 3 - 2| + Real.sqrt 12 = 4 + Real.sqrt 3 := by
  sorry

end calculate_expression_l2096_209683


namespace cuboid_surface_area_example_l2096_209679

/-- The surface area of a cuboid with given dimensions. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * height + breadth * height + length * breadth)

/-- Theorem: The surface area of a cuboid with length 4 cm, breadth 6 cm, and height 5 cm is 148 cm². -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 4 6 5 = 148 := by
  sorry

end cuboid_surface_area_example_l2096_209679


namespace westward_notation_l2096_209693

/-- Represents the direction on the runway -/
inductive Direction
  | East
  | West

/-- Represents a distance walked on the runway -/
structure Walk where
  distance : ℝ
  direction : Direction

/-- Converts a walk to its signed representation -/
def Walk.toSigned (w : Walk) : ℝ :=
  match w.direction with
  | Direction.East => w.distance
  | Direction.West => -w.distance

theorem westward_notation (d : ℝ) (h : d > 0) :
  let eastward := Walk.toSigned { distance := 8, direction := Direction.East }
  let westward := Walk.toSigned { distance := d, direction := Direction.West }
  eastward = 8 → westward = -d :=
by sorry

end westward_notation_l2096_209693


namespace pi_estimation_l2096_209678

theorem pi_estimation (m n : ℕ) (h : m > 0) : 
  ∃ (ε : ℝ), ε > 0 ∧ |4 * (n : ℝ) / (m : ℝ) - π| < ε :=
sorry

end pi_estimation_l2096_209678


namespace min_value_polynomial_min_value_achieved_l2096_209662

theorem min_value_polynomial (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 ≥ 2161.75 :=
sorry

theorem min_value_achieved : 
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2164 = 2161.75 :=
sorry

end min_value_polynomial_min_value_achieved_l2096_209662


namespace male_students_in_sample_l2096_209675

/-- Represents the stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  sample_size : ℕ
  female_count : ℕ

/-- Calculates the number of male students to be drawn in a stratified sample -/
def male_students_drawn (s : StratifiedSample) : ℕ :=
  s.sample_size

/-- Theorem stating the number of male students to be drawn in the given scenario -/
theorem male_students_in_sample (s : StratifiedSample) 
  (h1 : s.total_population = 900)
  (h2 : s.sample_size = 45)
  (h3 : s.female_count = 0) :
  male_students_drawn s = 25 := by
    sorry

#eval male_students_drawn { total_population := 900, sample_size := 45, female_count := 0 }

end male_students_in_sample_l2096_209675


namespace peter_distance_l2096_209676

/-- The total distance Peter covers -/
def D : ℝ := sorry

/-- The time Peter takes to cover the distance in hours -/
def total_time : ℝ := 1.4

/-- The speed at which Peter covers two-thirds of the distance -/
def speed1 : ℝ := 4

/-- The speed at which Peter covers one-third of the distance -/
def speed2 : ℝ := 5

theorem peter_distance : 
  (2/3 * D) / speed1 + (1/3 * D) / speed2 = total_time ∧ D = 6 := by sorry

end peter_distance_l2096_209676


namespace water_needed_for_lemonade_l2096_209654

/-- Given a ratio of water to lemon juice and a total amount of lemonade to make,
    calculate the amount of water needed in quarts. -/
theorem water_needed_for_lemonade 
  (water_ratio : ℚ)
  (lemon_juice_ratio : ℚ)
  (total_gallons : ℚ)
  (quarts_per_gallon : ℚ) :
  water_ratio = 8 →
  lemon_juice_ratio = 1 →
  total_gallons = 3/2 →
  quarts_per_gallon = 4 →
  (water_ratio * total_gallons * quarts_per_gallon) / (water_ratio + lemon_juice_ratio) = 16/3 :=
by
  sorry

end water_needed_for_lemonade_l2096_209654


namespace existence_of_sequences_l2096_209694

theorem existence_of_sequences : ∃ (u v : ℕ → ℕ) (k : ℕ+),
  (∀ n m : ℕ, n < m → u n < u m) ∧
  (∀ n m : ℕ, n < m → v n < v m) ∧
  (∀ n : ℕ, k * (u n * (u n + 1)) = v n ^ 2 + 1) :=
by sorry

end existence_of_sequences_l2096_209694


namespace handshake_arrangement_count_l2096_209617

/-- Represents a handshaking arrangement for a group of people -/
structure HandshakeArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, j ∈ shakes i ↔ i ∈ shakes j)

/-- The number of valid handshaking arrangements for 12 people -/
def M : ℕ := sorry

/-- Theorem stating that the number of handshaking arrangements is congruent to 340 modulo 1000 -/
theorem handshake_arrangement_count :
  M ≡ 340 [MOD 1000] := by sorry

end handshake_arrangement_count_l2096_209617


namespace books_to_read_l2096_209612

theorem books_to_read (total : ℕ) (mcgregor : ℕ) (floyd : ℕ) : 
  total = 89 → mcgregor = 34 → floyd = 32 → total - (mcgregor + floyd) = 23 := by
  sorry

end books_to_read_l2096_209612


namespace banana_arrangements_l2096_209653

def word : String := "BANANA"

/-- The number of unique arrangements of letters in the word -/
def num_arrangements (w : String) : ℕ := sorry

theorem banana_arrangements :
  num_arrangements word = 60 :=
by
  sorry

end banana_arrangements_l2096_209653


namespace cyclist_time_is_two_hours_l2096_209666

/-- Represents the scenario of two cyclists traveling between two points --/
structure CyclistScenario where
  s : ℝ  -- Base speed of cyclists without wind
  t : ℝ  -- Time taken by Cyclist 1 to travel from A to B
  wind_speed : ℝ := 3  -- Wind speed affecting both cyclists

/-- Conditions of the cyclist problem --/
def cyclist_problem (scenario : CyclistScenario) : Prop :=
  let total_time := 4  -- Total time after which they meet
  -- Distance covered by Cyclist 1 in total_time
  let dist_cyclist1 := scenario.s * total_time + scenario.wind_speed * (2 * scenario.t - total_time)
  -- Distance covered by Cyclist 2 in total_time
  let dist_cyclist2 := (scenario.s - scenario.wind_speed) * total_time
  -- They meet halfway of the total distance
  dist_cyclist1 = dist_cyclist2 + (scenario.s + scenario.wind_speed) * scenario.t

/-- The theorem stating that the time taken by Cyclist 1 to travel from A to B is 2 hours --/
theorem cyclist_time_is_two_hours (scenario : CyclistScenario) :
  cyclist_problem scenario → scenario.t = 2 := by
  sorry

#check cyclist_time_is_two_hours

end cyclist_time_is_two_hours_l2096_209666


namespace fraction_zero_implies_x_negative_five_l2096_209615

theorem fraction_zero_implies_x_negative_five (x : ℝ) :
  (x + 5) / (x - 2) = 0 → x = -5 := by
  sorry

end fraction_zero_implies_x_negative_five_l2096_209615


namespace inequality_solution_l2096_209633

theorem inequality_solution (x : ℝ) : x^2 - x - 5 > 3*x ↔ x > 5 ∨ x < -1 := by
  sorry

end inequality_solution_l2096_209633


namespace min_distance_parallel_lines_l2096_209651

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x + 3 * y - 9 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x + 3 * y + 1 = 0
  ∀ (P₁ : ℝ × ℝ) (P₂ : ℝ × ℝ),
  l₁ P₁.1 P₁.2 → l₂ P₂.1 P₂.2 →
  ∃ (P₁' : ℝ × ℝ) (P₂' : ℝ × ℝ),
  l₁ P₁'.1 P₁'.2 ∧ l₂ P₂'.1 P₂'.2 ∧
  Real.sqrt 10 = ‖(P₁'.1 - P₂'.1, P₁'.2 - P₂'.2)‖ ∧
  ∀ (Q₁ : ℝ × ℝ) (Q₂ : ℝ × ℝ),
  l₁ Q₁.1 Q₁.2 → l₂ Q₂.1 Q₂.2 →
  Real.sqrt 10 ≤ ‖(Q₁.1 - Q₂.1, Q₁.2 - Q₂.2)‖ :=
by
  sorry


end min_distance_parallel_lines_l2096_209651


namespace function_composition_equality_l2096_209624

theorem function_composition_equality (A B : ℝ) (h : B ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ A * x^2 - 3 * B^3
  let g : ℝ → ℝ := λ x ↦ 2 * B * x + B^2
  f (g 2) = 0 → A = 3 / (16 / B + 8 * B + B^3) := by
  sorry

end function_composition_equality_l2096_209624


namespace log_relation_l2096_209677

theorem log_relation (p q : ℝ) (hp : 0 < p) : 
  (Real.log 5 / Real.log 8 = p) → (Real.log 125 / Real.log 2 = q * p) → q = 3 := by
  sorry

end log_relation_l2096_209677


namespace smallest_x_value_l2096_209621

theorem smallest_x_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -8) : 
  (∀ z : ℤ, ∃ w : ℤ, z * w + 7 * z + 6 * w = -8 → z ≥ -40) ∧ 
  (∃ w : ℤ, -40 * w + 7 * (-40) + 6 * w = -8) :=
by sorry

end smallest_x_value_l2096_209621


namespace robin_gum_count_l2096_209664

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 12

/-- The number of gum pieces in each package -/
def pieces_per_package : ℕ := 20

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_count : total_pieces = 240 := by
  sorry

end robin_gum_count_l2096_209664


namespace division_of_decimals_l2096_209682

theorem division_of_decimals : (0.05 : ℝ) / 0.0025 = 20 := by sorry

end division_of_decimals_l2096_209682


namespace class_book_count_l2096_209656

/-- Calculates the total number of books a class has from the library --/
def totalBooks (initial borrowed₁ returned borrowed₂ : ℕ) : ℕ :=
  initial + borrowed₁ - returned + borrowed₂

/-- Theorem: The class currently has 80 books from the library --/
theorem class_book_count :
  totalBooks 54 23 12 15 = 80 := by
  sorry

end class_book_count_l2096_209656


namespace proposition_logic_proof_l2096_209681

theorem proposition_logic_proof (p q : Prop) 
  (hp : p ↔ (3 ≥ 3)) 
  (hq : q ↔ (3 > 4)) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬(¬p) := by
  sorry

end proposition_logic_proof_l2096_209681


namespace largest_multiple_of_nine_l2096_209632

theorem largest_multiple_of_nine (n : ℤ) : 
  (n % 9 = 0 ∧ -n > -100) → n ≤ 99 :=
by sorry

end largest_multiple_of_nine_l2096_209632


namespace abc_product_l2096_209620

theorem abc_product (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * Real.rpow 3 (1/3))
  (hac : a * c = 40 * Real.rpow 3 (1/3))
  (hbc : b * c = 18 * Real.rpow 3 (1/3)) :
  a * b * c = 432 * Real.sqrt 5 := by
sorry

end abc_product_l2096_209620


namespace power_sum_prime_l2096_209648

theorem power_sum_prime (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2 ^ p + 3 ^ p = a ^ n) → n = 1 := by
  sorry

end power_sum_prime_l2096_209648


namespace x_squared_minus_y_squared_minus_z_squared_l2096_209686

theorem x_squared_minus_y_squared_minus_z_squared (x y z : ℝ) 
  (sum_eq : x + y + z = 12)
  (diff_eq : x - y = 4)
  (yz_sum : y + z = 7) :
  x^2 - y^2 - z^2 = -12 := by
sorry

end x_squared_minus_y_squared_minus_z_squared_l2096_209686


namespace triangle_area_in_square_pyramid_l2096_209674

/-- Square pyramid with given dimensions and points -/
structure SquarePyramid where
  -- Base side length
  base_side : ℝ
  -- Altitude
  altitude : ℝ
  -- Points P, Q, R are located 1/4 of the way from B, D, C to E respectively
  point_ratio : ℝ

/-- The area of triangle PQR in the square pyramid -/
def triangle_area (pyramid : SquarePyramid) : ℝ := sorry

/-- Theorem statement -/
theorem triangle_area_in_square_pyramid :
  ∀ (pyramid : SquarePyramid),
  pyramid.base_side = 4 ∧ 
  pyramid.altitude = 8 ∧ 
  pyramid.point_ratio = 1/4 →
  triangle_area pyramid = (45 * Real.sqrt 3) / 4 := by
  sorry

end triangle_area_in_square_pyramid_l2096_209674


namespace multiply_98_by_98_l2096_209652

theorem multiply_98_by_98 : 98 * 98 = 9604 := by
  sorry

end multiply_98_by_98_l2096_209652


namespace jessica_total_cost_l2096_209649

def cat_toy_cost : ℚ := 10.22
def cage_cost : ℚ := 11.73
def cat_food_cost : ℚ := 7.50
def leash_cost : ℚ := 5.15
def cat_treats_cost : ℚ := 3.98

theorem jessica_total_cost :
  cat_toy_cost + cage_cost + cat_food_cost + leash_cost + cat_treats_cost = 38.58 := by
  sorry

end jessica_total_cost_l2096_209649


namespace min_value_when_a_is_one_inequality_condition_l2096_209605

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + 2 * |x - 1|

-- Part 1
theorem min_value_when_a_is_one :
  ∃ m : ℝ, (∀ x : ℝ, f x 1 ≥ m) ∧ (∃ x : ℝ, f x 1 = m) ∧ m = 2 :=
sorry

-- Part 2
theorem inequality_condition :
  ∀ a b : ℝ, a > 0 → b > 0 →
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → f x a > x^2 - b + 1) →
  (a + 1/2)^2 + (b + 1/2)^2 > 2 :=
sorry

end min_value_when_a_is_one_inequality_condition_l2096_209605


namespace zoe_played_two_months_l2096_209699

/-- Calculates the number of months played given the initial cost, monthly cost, and total spent -/
def months_played (initial_cost monthly_cost total_spent : ℕ) : ℕ :=
  (total_spent - initial_cost) / monthly_cost

/-- Proves that Zoe played the game online for 2 months -/
theorem zoe_played_two_months (initial_cost monthly_cost total_spent : ℕ) 
  (h1 : initial_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : total_spent = 21) :
  months_played initial_cost monthly_cost total_spent = 2 := by
  sorry

#eval months_played 5 8 21

end zoe_played_two_months_l2096_209699


namespace geometric_sequence_product_l2096_209639

theorem geometric_sequence_product (a b : ℝ) : 
  (5 < a) → (a < b) → (b < 40) → 
  (b / a = a / 5) → (40 / b = b / a) → 
  a * b = 200 := by
sorry

end geometric_sequence_product_l2096_209639


namespace lot_width_calculation_l2096_209688

/-- Given a rectangular lot with length 40 m, height 2 m, and volume 1600 m³, 
    the width of the lot is 20 m. -/
theorem lot_width_calculation (length height volume width : ℝ) 
  (h_length : length = 40)
  (h_height : height = 2)
  (h_volume : volume = 1600)
  (h_relation : volume = length * width * height) : 
  width = 20 := by
  sorry

end lot_width_calculation_l2096_209688


namespace triangle_inequality_l2096_209684

theorem triangle_inequality (A B C : Real) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : C = π - A - B) 
  (h : 1 / Real.sin A + 2 / Real.sin B = 3 * (1 / Real.tan A + 1 / Real.tan B)) :
  Real.cos C ≥ (2 * Real.sqrt 10 - 2) / 9 := by
  sorry

end triangle_inequality_l2096_209684


namespace no_interior_points_with_sum_20_l2096_209642

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance squared between two points -/
def distSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A circle with center at the origin and radius 2 -/
def insideCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 < 4

theorem no_interior_points_with_sum_20 :
  ¬ ∃ (p : Point), insideCircle p ∧
    ∃ (a b : Point), 
      a.x^2 + a.y^2 = 4 ∧ 
      b.x^2 + b.y^2 = 4 ∧ 
      a.x = -b.x ∧ 
      a.y = -b.y ∧
      distSquared p a + distSquared p b = 20 :=
by sorry

end no_interior_points_with_sum_20_l2096_209642


namespace frog_jump_distance_l2096_209690

/-- The jumping contest problem -/
theorem frog_jump_distance 
  (grasshopper_jump : ℕ) 
  (frog_extra_jump : ℕ) 
  (mouse_less_jump : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra_jump = 39)
  (h3 : mouse_less_jump = 94) :
  grasshopper_jump + frog_extra_jump = 58 :=
by sorry

end frog_jump_distance_l2096_209690


namespace inequality_proof_l2096_209646

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ¬(1/a > 1/b) := by
  sorry

end inequality_proof_l2096_209646


namespace rectangular_plot_ratio_l2096_209618

/-- A rectangular plot with given area and breadth -/
structure RectangularPlot where
  area : ℝ
  breadth : ℝ
  length_multiple : ℕ
  area_eq : area = breadth * (breadth * length_multiple)

/-- The ratio of length to breadth for a rectangular plot -/
def length_breadth_ratio (plot : RectangularPlot) : ℚ :=
  plot.length_multiple

theorem rectangular_plot_ratio (plot : RectangularPlot) 
  (h1 : plot.area = 432)
  (h2 : plot.breadth = 12) :
  length_breadth_ratio plot = 3 := by
  sorry

end rectangular_plot_ratio_l2096_209618


namespace gcd_of_sum_is_222_l2096_209637

def is_consecutive_even (a b c d : ℕ) : Prop :=
  b = a + 2 ∧ c = a + 4 ∧ d = a + 6 ∧ a % 2 = 0

def e_sum (a d : ℕ) : ℕ := a + d

def abcde (a b c d e : ℕ) : ℕ := 10000 * a + 1000 * b + 100 * c + 10 * d + e

def edcba (a b c d e : ℕ) : ℕ := 10000 * e + 1000 * d + 100 * c + 10 * b + a

theorem gcd_of_sum_is_222 (a b c d : ℕ) (h : is_consecutive_even a b c d) :
  Nat.gcd (abcde a b c d (e_sum a d) + edcba a b c d (e_sum a d))
          (abcde (a + 2) (b + 2) (c + 2) (d + 2) (e_sum (a + 2) (d + 2)) +
           edcba (a + 2) (b + 2) (c + 2) (d + 2) (e_sum (a + 2) (d + 2))) = 222 :=
sorry

end gcd_of_sum_is_222_l2096_209637


namespace sin_two_alpha_value_l2096_209658

theorem sin_two_alpha_value (α : Real) (h : Real.sin α + Real.cos α = 2/3) : 
  Real.sin (2 * α) = -5/9 := by
  sorry

end sin_two_alpha_value_l2096_209658


namespace ellipse_equation_from_conditions_l2096_209687

/-- An ellipse E with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def ellipse_equation (E : Ellipse) (x y : ℝ) : Prop :=
  x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The area of the quadrilateral formed by the four vertices of an ellipse -/
def quadrilateral_area (E : Ellipse) : ℝ :=
  2 * E.a * E.b

theorem ellipse_equation_from_conditions (E : Ellipse) 
  (h_vertex : ellipse_equation E 0 (-2))
  (h_area : quadrilateral_area E = 4 * Real.sqrt 5) :
  ∀ x y, ellipse_equation E x y ↔ x^2 / 5 + y^2 / 4 = 1 := by
  sorry

end ellipse_equation_from_conditions_l2096_209687


namespace intersection_sequence_correct_l2096_209630

def A : Set ℕ := {n | ∃ m : ℕ+, n = m * (m + 1)}
def B : Set ℕ := {n | ∃ m : ℕ+, n = 3 * m - 1}

def intersection_sequence (k : ℕ+) : ℕ := 9 * k^2 - 9 * k + 2

theorem intersection_sequence_correct :
  ∀ k : ℕ+, (intersection_sequence k) ∈ A ∩ B ∧
  (∀ n ∈ A ∩ B, n < intersection_sequence k → 
    ∃ j : ℕ+, j < k ∧ n = intersection_sequence j) :=
sorry

end intersection_sequence_correct_l2096_209630


namespace max_runs_in_one_day_match_l2096_209669

/-- Represents the number of overs in a cricket one-day match -/
def overs : ℕ := 50

/-- Represents the number of legal deliveries per over -/
def deliveries_per_over : ℕ := 6

/-- Represents the maximum number of runs that can be scored on a single delivery -/
def max_runs_per_delivery : ℕ := 6

/-- Theorem stating the maximum number of runs a batsman can score in an ideal scenario -/
theorem max_runs_in_one_day_match :
  overs * deliveries_per_over * max_runs_per_delivery = 1800 := by
  sorry

end max_runs_in_one_day_match_l2096_209669


namespace factors_of_243_l2096_209668

theorem factors_of_243 : Finset.card (Nat.divisors 243) = 6 := by
  sorry

end factors_of_243_l2096_209668


namespace triangle_side_length_l2096_209659

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a * c = 8 → a + c = 7 → B = π / 3 → b = 5 := by
  sorry

end triangle_side_length_l2096_209659


namespace sixth_sample_is_98_l2096_209650

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) (k : ℕ) : ℕ :=
  start + (k - 1) * (total / sampleSize)

theorem sixth_sample_is_98 :
  systematicSample 900 50 8 6 = 98 := by
  sorry

end sixth_sample_is_98_l2096_209650


namespace quadratic_distinct_roots_range_l2096_209622

theorem quadratic_distinct_roots_range (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + k = 0 ∧ y^2 - 2*y + k = 0) ↔ k < 1 :=
sorry

end quadratic_distinct_roots_range_l2096_209622


namespace all_approximations_valid_l2096_209611

/-- Represents an approximation with its estimated value, absolute error, and relative error. -/
structure Approximation where
  value : ℝ
  absoluteError : ℝ
  relativeError : ℝ

/-- The approximations given in the problem. -/
def classSize : Approximation := ⟨40, 5, 0.125⟩
def hallPeople : Approximation := ⟨1500, 100, 0.067⟩
def itemPrice : Approximation := ⟨100, 5, 0.05⟩
def pageCharacters : Approximation := ⟨40000, 500, 0.0125⟩

/-- Checks if the relative error is correctly calculated from the absolute error and value. -/
def isValidApproximation (a : Approximation) : Prop :=
  a.relativeError = a.absoluteError / a.value

/-- Proves that all given approximations are valid. -/
theorem all_approximations_valid :
  isValidApproximation classSize ∧
  isValidApproximation hallPeople ∧
  isValidApproximation itemPrice ∧
  isValidApproximation pageCharacters :=
sorry

end all_approximations_valid_l2096_209611


namespace little_twelve_conference_games_l2096_209619

/-- Calculates the number of games in a football conference with specified rules -/
def conference_games (num_divisions : ℕ) (teams_per_division : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) : ℕ :=
  let intra_division_games := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_div_games
  let inter_division_games := (num_divisions * teams_per_division) * (teams_per_division * (num_divisions - 1)) * inter_div_games / 2
  intra_division_games + inter_division_games

/-- The Little Twelve Football Conference scheduling theorem -/
theorem little_twelve_conference_games :
  conference_games 2 6 3 2 = 162 := by
  sorry

end little_twelve_conference_games_l2096_209619


namespace parking_cost_is_10_l2096_209614

-- Define the given conditions
def saved : ℕ := 28
def entry_cost : ℕ := 55
def meal_pass_cost : ℕ := 25
def distance : ℕ := 165
def fuel_efficiency : ℕ := 30
def gas_price : ℕ := 3
def additional_savings : ℕ := 95

-- Define the function to calculate parking cost
def parking_cost : ℕ :=
  let total_needed := saved + additional_savings
  let round_trip_distance := 2 * distance
  let gas_needed := round_trip_distance / fuel_efficiency
  let gas_cost := gas_needed * gas_price
  let total_cost_without_parking := gas_cost + entry_cost + meal_pass_cost
  total_needed - total_cost_without_parking

-- Theorem to prove
theorem parking_cost_is_10 : parking_cost = 10 := by
  sorry

end parking_cost_is_10_l2096_209614


namespace log_zero_nonexistent_l2096_209680

theorem log_zero_nonexistent : ¬ ∃ x : ℝ, Real.log x = 0 := by sorry

end log_zero_nonexistent_l2096_209680


namespace chess_team_arrangements_l2096_209600

/-- Represents the number of boys on the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls on the chess team -/
def num_girls : ℕ := 3

/-- Represents the total number of team members -/
def total_members : ℕ := num_boys + num_girls

/-- Represents the number of positions in the middle of the row -/
def middle_positions : ℕ := total_members - 2

/-- Calculates the number of ways to arrange girls at the ends -/
def end_arrangements : ℕ := num_girls * (num_girls - 1)

/-- Calculates the number of ways to arrange the middle positions -/
def middle_arrangements : ℕ := Nat.factorial middle_positions

/-- Theorem: The total number of possible arrangements is 144 -/
theorem chess_team_arrangements :
  end_arrangements * middle_arrangements = 144 := by
  sorry


end chess_team_arrangements_l2096_209600


namespace chinese_number_puzzle_l2096_209628

def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem chinese_number_puzzle :
  ∀ (x y z : ℕ),
    x < 100 →
    y < 10000 →
    z < 10000 →
    100 * x + y + z = 2015 →
    z + sum_n 10 = y →
    x ≠ (y / 1000) →
    x ≠ (y / 100 % 10) →
    x ≠ (y / 10 % 10) →
    x ≠ (y % 10) →
    x ≠ (z / 1000) →
    x ≠ (z / 100 % 10) →
    x ≠ (z / 10 % 10) →
    x ≠ (z % 10) →
    (y / 1000) ≠ (y / 100 % 10) →
    (y / 1000) ≠ (y / 10 % 10) →
    (y / 1000) ≠ (y % 10) →
    (y / 100 % 10) ≠ (y / 10 % 10) →
    (y / 100 % 10) ≠ (y % 10) →
    (y / 10 % 10) ≠ (y % 10) →
    (z / 1000) ≠ (z / 100 % 10) →
    (z / 1000) ≠ (z / 10 % 10) →
    (z / 1000) ≠ (z % 10) →
    (z / 100 % 10) ≠ (z / 10 % 10) →
    (z / 100 % 10) ≠ (z % 10) →
    (z / 10 % 10) ≠ (z % 10) →
    100 * x + y = 1985 :=
by
  sorry

#eval sum_n 10  -- This should evaluate to 55

end chinese_number_puzzle_l2096_209628


namespace seven_digit_sum_theorem_l2096_209665

theorem seven_digit_sum_theorem :
  ∀ a b : ℕ,
  (a ≤ 9 ∧ a > 0) →
  (b ≤ 9) →
  (7 * a = 10 * a + b) →
  (a + b = 10) :=
by
  sorry

end seven_digit_sum_theorem_l2096_209665


namespace nina_running_distance_l2096_209689

theorem nina_running_distance : 
  0.08333333333333333 + 0.08333333333333333 + 0.6666666666666666 = 0.8333333333333333 := by
  sorry

end nina_running_distance_l2096_209689


namespace min_tan_angle_ocular_rays_l2096_209631

def G : Set (ℕ × ℕ) := {p | p.1 ≤ 20 ∧ p.2 ≤ 20 ∧ p.1 > 0 ∧ p.2 > 0}

def isOcularRay (m : ℚ) : Prop := ∃ p ∈ G, m = p.2 / p.1

def tanAngleBetweenRays (m1 m2 : ℚ) : ℚ := |m1 - m2| / (1 + m1 * m2)

def A : Set ℚ := {a | ∃ m1 m2, isOcularRay m1 ∧ isOcularRay m2 ∧ m1 ≠ m2 ∧ a = tanAngleBetweenRays m1 m2}

theorem min_tan_angle_ocular_rays :
  ∃ a ∈ A, a = (1 : ℚ) / 722 ∧ ∀ b ∈ A, (1 : ℚ) / 722 ≤ b :=
sorry

end min_tan_angle_ocular_rays_l2096_209631


namespace salt_mixture_proof_l2096_209604

/-- Proves that adding 70 ounces of a 60% salt solution to 70 ounces of a 20% salt solution
    results in a mixture that is 40% salt. -/
theorem salt_mixture_proof :
  let initial_amount : ℝ := 70
  let initial_concentration : ℝ := 0.20
  let added_amount : ℝ := 70
  let added_concentration : ℝ := 0.60
  let final_concentration : ℝ := 0.40
  let total_amount : ℝ := initial_amount + added_amount
  let total_salt : ℝ := initial_amount * initial_concentration + added_amount * added_concentration
  total_salt / total_amount = final_concentration := by
  sorry

#check salt_mixture_proof

end salt_mixture_proof_l2096_209604


namespace train_crossing_time_l2096_209672

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (pole_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 300)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 200) :
  (train_length + platform_length) / (train_length / pole_crossing_time) = 30 := by
  sorry

end train_crossing_time_l2096_209672


namespace josh_remaining_money_l2096_209640

def initial_amount : ℝ := 100

def transactions : List ℝ := [12.67, 25.39, 14.25, 4.32, 27.50]

def remaining_money : ℝ := initial_amount - transactions.sum

theorem josh_remaining_money :
  remaining_money = 15.87 := by sorry

end josh_remaining_money_l2096_209640


namespace garden_length_proof_l2096_209603

theorem garden_length_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 2 + 3 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 38 := by
sorry

end garden_length_proof_l2096_209603


namespace quadratic_roots_sum_squares_equal_l2096_209696

theorem quadratic_roots_sum_squares_equal (a : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    x₁^2 + 2*x₁ + a = 0 ∧
    x₂^2 + 2*x₂ + a = 0 ∧
    y₁^2 + a*y₁ + 2 = 0 ∧
    y₂^2 + a*y₂ + 2 = 0 ∧
    x₁^2 + x₂^2 = y₁^2 + y₂^2) →
  a = -4 :=
by sorry

end quadratic_roots_sum_squares_equal_l2096_209696


namespace intersection_sum_l2096_209616

theorem intersection_sum (a b : ℝ) : 
  (2 = (1/3) * 4 + a) ∧ (4 = (1/3) * 2 + b) → a + b = 4 := by
  sorry

end intersection_sum_l2096_209616


namespace parallelogram_side_length_l2096_209602

theorem parallelogram_side_length 
  (s : ℝ) 
  (area : ℝ) 
  (angle : ℝ) :
  s > 0 → 
  angle = π / 6 → 
  area = 27 * Real.sqrt 3 → 
  3 * s * s * Real.sqrt 3 = area → 
  s = 3 := by
  sorry

end parallelogram_side_length_l2096_209602


namespace algebraic_simplification_l2096_209691

/-- Proves the simplification of two algebraic expressions -/
theorem algebraic_simplification 
  (a b m n : ℝ) : 
  ((a - 2*b) - (2*b - 5*a) = 6*a - 4*b) ∧ 
  (-m^2*n + (4*m*n^2 - 3*m*n) - 2*(m*n^2 - 3*m^2*n) = 5*m^2*n + 2*m*n^2 - 3*m*n) :=
by sorry

end algebraic_simplification_l2096_209691


namespace joan_seashells_l2096_209627

theorem joan_seashells (seashells_left seashells_given_to_sam : ℕ) 
  (h1 : seashells_left = 27) 
  (h2 : seashells_given_to_sam = 43) : 
  seashells_left + seashells_given_to_sam = 70 := by
  sorry

end joan_seashells_l2096_209627


namespace lines_concurrent_at_S_l2096_209601

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Tetrahedron SABC with points A', B', C' on edges SA, SB, SC respectively -/
structure Tetrahedron where
  S : Point3D
  A : Point3D
  B : Point3D
  C : Point3D
  A' : Point3D
  B' : Point3D
  C' : Point3D

/-- The intersection line d of planes ABC and A'B'C' -/
def intersection_line (t : Tetrahedron) : Line3D :=
  sorry

/-- Theorem: Lines AA', BB', CC' are concurrent at S for any rotation of A'B'C' around d -/
theorem lines_concurrent_at_S (t : Tetrahedron) (θ : ℝ) : 
  ∃ (S : Point3D), 
    (Line3D.mk t.A t.A').point = S ∧ 
    (Line3D.mk t.B t.B').point = S ∧ 
    (Line3D.mk t.C t.C').point = S := by
  sorry

end lines_concurrent_at_S_l2096_209601


namespace simplify_power_l2096_209657

theorem simplify_power (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end simplify_power_l2096_209657


namespace intersection_when_m_3_intersection_equals_B_l2096_209610

def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}

def B (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 3 * m - 2}

theorem intersection_when_m_3 : A ∩ B 3 = {x | 2 ≤ x ∧ x ≤ 4} := by sorry

theorem intersection_equals_B (m : ℝ) : A ∩ B m = B m ↔ m ≤ 2 := by sorry

end intersection_when_m_3_intersection_equals_B_l2096_209610


namespace binomial_sum_modulo_1000_l2096_209663

theorem binomial_sum_modulo_1000 : 
  (Finset.sum (Finset.range 503) (fun k => Nat.choose 2011 (4 * k))) % 1000 = 15 := by
  sorry

end binomial_sum_modulo_1000_l2096_209663


namespace mineral_worth_l2096_209695

/-- The worth of a mineral given its price per gram and weight -/
theorem mineral_worth (price_per_gram : ℝ) (weight_1 weight_2 : ℝ) :
  price_per_gram = 17.25 →
  weight_1 = 1000 →
  weight_2 = 10 →
  price_per_gram * (weight_1 + weight_2) = 17422.5 := by
  sorry

end mineral_worth_l2096_209695


namespace joseph_investment_result_l2096_209692

/-- Calculates the final amount in an investment account after a given number of years,
    with an initial investment, yearly interest rate, and monthly additional deposits. -/
def investment_calculation (initial_investment : ℝ) (interest_rate : ℝ) 
                           (monthly_deposit : ℝ) (years : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the investment calculation for Joseph's scenario
    results in $3982 after two years. -/
theorem joseph_investment_result :
  investment_calculation 1000 0.10 100 2 = 3982 := by
  sorry

end joseph_investment_result_l2096_209692


namespace winning_percentage_l2096_209671

/-- Given an election with 6000 total votes and a winning margin of 1200 votes,
    prove that the winning candidate received 60% of the votes. -/
theorem winning_percentage (total_votes : ℕ) (winning_margin : ℕ) (winning_percentage : ℚ) :
  total_votes = 6000 →
  winning_margin = 1200 →
  winning_percentage = 60 / 100 →
  winning_percentage * total_votes = (total_votes + winning_margin) / 2 :=
by sorry

end winning_percentage_l2096_209671


namespace circle_radius_proof_l2096_209643

theorem circle_radius_proof (r : ℝ) (h : r > 0) :
  3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → r = 3 := by
  sorry

end circle_radius_proof_l2096_209643


namespace probability_sum_5_is_one_ninth_l2096_209607

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to roll a sum of 5 with two dice -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a sum of 5 with two dice -/
def probability_sum_5 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_5_is_one_ninth : probability_sum_5 = 1 / 9 := by
  sorry

end probability_sum_5_is_one_ninth_l2096_209607


namespace proportionality_problem_l2096_209634

/-- Given that x is directly proportional to y² and inversely proportional to z,
    prove that x = 24 when y = 2 and z = 3, given that x = 6 when y = 1 and z = 3. -/
theorem proportionality_problem (k : ℝ) :
  (∀ y z, ∃ x, x = k * y^2 / z) →
  (6 = k * 1^2 / 3) →
  (∃ x, x = k * 2^2 / 3 ∧ x = 24) :=
by sorry

end proportionality_problem_l2096_209634


namespace blueberry_pie_count_l2096_209606

theorem blueberry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 30)
  (h_ratio : apple_ratio + blueberry_ratio + cherry_ratio = 10)
  (h_apple : apple_ratio = 2)
  (h_blueberry : blueberry_ratio = 3)
  (h_cherry : cherry_ratio = 5) :
  (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) * blueberry_ratio = 9 := by
  sorry

end blueberry_pie_count_l2096_209606


namespace smallest_class_size_l2096_209660

theorem smallest_class_size (n : ℕ) (scores : Fin n → ℕ) : 
  (∀ i, scores i ≤ 120) →  -- Each student took a 120-point test
  (∃ s : Finset (Fin n), s.card = 8 ∧ ∀ i ∈ s, scores i = 120) →  -- Eight students scored 120
  (∀ i, scores i ≥ 72) →  -- Each student scored at least 72
  (Finset.sum Finset.univ scores / n = 84) →  -- The mean score was 84
  (n ≥ 32 ∧ ∀ m : ℕ, m < 32 → ¬ (∃ scores' : Fin m → ℕ, 
    (∀ i, scores' i ≤ 120) ∧ 
    (∃ s : Finset (Fin m), s.card = 8 ∧ ∀ i ∈ s, scores' i = 120) ∧ 
    (∀ i, scores' i ≥ 72) ∧ 
    (Finset.sum Finset.univ scores' / m = 84))) :=
by sorry

end smallest_class_size_l2096_209660


namespace sin_equation_holds_ten_degrees_is_acute_l2096_209608

theorem sin_equation_holds : 
  (Real.sin (10 * Real.pi / 180)) * (1 + Real.sqrt 3 * Real.tan (70 * Real.pi / 180)) = 1 := by
  sorry

-- Additional definition to ensure 10° is acute
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

theorem ten_degrees_is_acute : is_acute_angle (10 * Real.pi / 180) := by
  sorry

end sin_equation_holds_ten_degrees_is_acute_l2096_209608


namespace quadratic_equation_equivalence_l2096_209673

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (2 * x^2 - x * (x - 4) = 5) ↔ (x^2 + 4*x - 5 = 0) := by
  sorry

end quadratic_equation_equivalence_l2096_209673


namespace hall_length_is_36_meters_l2096_209685

-- Define the hall dimensions
def hall_width : ℝ := 15

-- Define the stone dimensions in meters
def stone_length : ℝ := 0.6  -- 6 dm = 0.6 m
def stone_width : ℝ := 0.5   -- 5 dm = 0.5 m

-- Define the number of stones
def num_stones : ℕ := 1800

-- Theorem stating the length of the hall
theorem hall_length_is_36_meters :
  let total_area := (↑num_stones : ℝ) * stone_length * stone_width
  let hall_length := total_area / hall_width
  hall_length = 36 := by sorry

end hall_length_is_36_meters_l2096_209685


namespace albert_additional_laps_l2096_209670

/-- Calculates the number of additional complete laps needed to finish a given distance. -/
def additional_laps (total_distance : ℕ) (track_length : ℕ) (completed_laps : ℕ) : ℕ :=
  ((total_distance - completed_laps * track_length) / track_length : ℕ)

/-- Theorem: Given the specific conditions, the number of additional complete laps is 5. -/
theorem albert_additional_laps :
  additional_laps 99 9 6 = 5 := by
  sorry

end albert_additional_laps_l2096_209670


namespace bankers_gain_calculation_l2096_209638

/-- Banker's gain calculation -/
theorem bankers_gain_calculation (true_discount : ℝ) (interest_rate : ℝ) (time_period : ℝ) :
  true_discount = 60.00000000000001 →
  interest_rate = 0.12 →
  time_period = 1 →
  let face_value := (true_discount * (1 + interest_rate * time_period)) / (interest_rate * time_period)
  let bankers_discount := face_value * interest_rate * time_period
  bankers_discount - true_discount = 7.2 := by
  sorry

end bankers_gain_calculation_l2096_209638


namespace profit_starts_in_fourth_year_option_one_more_profitable_l2096_209625

/-- Represents the financial data for the real estate investment --/
structure RealEstateInvestment where
  initialInvestment : ℝ
  firstYearRenovationCost : ℝ
  yearlyRenovationIncrease : ℝ
  annualRentalIncome : ℝ

/-- Calculates the renovation cost for a given year --/
def renovationCost (investment : RealEstateInvestment) (year : ℕ) : ℝ :=
  investment.firstYearRenovationCost + investment.yearlyRenovationIncrease * (year - 1)

/-- Calculates the cumulative profit up to a given year --/
def cumulativeProfit (investment : RealEstateInvestment) (year : ℕ) : ℝ :=
  year * investment.annualRentalIncome - investment.initialInvestment - 
    (Finset.range year).sum (fun i => renovationCost investment (i + 1))

/-- Theorem stating that the developer starts making a net profit in the 4th year --/
theorem profit_starts_in_fourth_year (investment : RealEstateInvestment) 
  (h1 : investment.initialInvestment = 810000)
  (h2 : investment.firstYearRenovationCost = 10000)
  (h3 : investment.yearlyRenovationIncrease = 20000)
  (h4 : investment.annualRentalIncome = 300000) :
  (cumulativeProfit investment 3 < 0) ∧ (cumulativeProfit investment 4 > 0) := by
  sorry

/-- Represents the two selling options --/
inductive SellingOption
  | OptionOne : SellingOption
  | OptionTwo : SellingOption

/-- Calculates the profit for a given selling option --/
def profitForOption (investment : RealEstateInvestment) (option : SellingOption) : ℝ :=
  match option with
  | SellingOption.OptionOne => 460000 -- Simplified for the sake of the statement
  | SellingOption.OptionTwo => 100000 -- Simplified for the sake of the statement

/-- Theorem stating that Option 1 is more profitable --/
theorem option_one_more_profitable (investment : RealEstateInvestment) :
  profitForOption investment SellingOption.OptionOne > profitForOption investment SellingOption.OptionTwo := by
  sorry

end profit_starts_in_fourth_year_option_one_more_profitable_l2096_209625


namespace probability_same_color_half_l2096_209613

/-- Represents a bag of colored balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing balls of the same color from two bags -/
def probability_same_color (bag_a bag_b : Bag) : ℚ :=
  let total_a := bag_a.white + bag_a.red
  let total_b := bag_b.white + bag_b.red
  (bag_a.white * bag_b.white + bag_a.red * bag_b.red) / (total_a * total_b)

/-- The main theorem stating that the probability of drawing balls of the same color
    from the given bags is 1/2 -/
theorem probability_same_color_half :
  let bag_a : Bag := ⟨8, 4⟩
  let bag_b : Bag := ⟨6, 6⟩
  probability_same_color bag_a bag_b = 1/2 := by
  sorry


end probability_same_color_half_l2096_209613


namespace shrimp_price_theorem_l2096_209697

/-- The discounted price of a quarter-pound package of shrimp -/
def discounted_price : ℝ := 2.25

/-- The discount percentage as a decimal -/
def discount_rate : ℝ := 0.4

/-- The standard price per pound of shrimp -/
def standard_price : ℝ := 15

/-- Theorem stating that the standard price per pound of shrimp is $15 -/
theorem shrimp_price_theorem :
  standard_price = 15 ∧
  discounted_price = (1 - discount_rate) * (standard_price / 4) :=
by sorry

end shrimp_price_theorem_l2096_209697


namespace middle_number_in_ratio_l2096_209655

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 3 / 2 ∧ b / c = 2 / 5 ∧ a^2 + b^2 + c^2 = 1862 → b = 14 := by
  sorry

end middle_number_in_ratio_l2096_209655


namespace three_fourths_of_forty_l2096_209641

theorem three_fourths_of_forty : (3 / 4 : ℚ) * 40 = 30 := by
  sorry

end three_fourths_of_forty_l2096_209641


namespace complex_number_location_l2096_209645

theorem complex_number_location (z : ℂ) (h : z * (-1 + 2 * Complex.I) = Complex.abs (1 + 3 * Complex.I)) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_location_l2096_209645


namespace range_of_a_l2096_209635

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, a ≥ Real.exp x

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 4 = 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a)) ∧ (p a ∨ q a) → (e ≤ a ∧ a < 4) ∨ (a ≤ -4) :=
by sorry

end range_of_a_l2096_209635


namespace inscribed_semicircle_radius_l2096_209661

/-- An isosceles triangle with base 16 and height 15 -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  isBase16 : base = 16
  isHeight15 : height = 15

/-- A semicircle inscribed in the isosceles triangle -/
structure InscribedSemicircle (t : IsoscelesTriangle) where
  radius : ℝ
  diameterOnBase : radius * 2 ≤ t.base

/-- The radius of the inscribed semicircle is 120/17 -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangle) 
  (s : InscribedSemicircle t) : s.radius = 120 / 17 := by
  sorry

end inscribed_semicircle_radius_l2096_209661


namespace value_of_n_l2096_209636

theorem value_of_n : ∃ n : ℕ, 5^3 - 7 = 2^2 + n ∧ n = 114 := by sorry

end value_of_n_l2096_209636


namespace tan_a_pi_third_equals_sqrt_three_l2096_209644

-- Define the function for logarithm with base a
noncomputable def log_base (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem tan_a_pi_third_equals_sqrt_three 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : log_base a 16 = 2) : 
  Real.tan (a * π / 3) = Real.sqrt 3 := by
  sorry

end tan_a_pi_third_equals_sqrt_three_l2096_209644


namespace point_on_number_line_l2096_209698

theorem point_on_number_line (B C : ℝ) : 
  B = 3 → abs (C - B) = 2 → (C = 1 ∨ C = 5) :=
by sorry

end point_on_number_line_l2096_209698
