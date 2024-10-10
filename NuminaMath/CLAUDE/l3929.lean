import Mathlib

namespace sophias_book_length_l3929_392953

theorem sophias_book_length :
  ∀ (total_pages : ℕ),
  (2 : ℚ) / 3 * total_pages = (1 : ℚ) / 3 * total_pages + 90 →
  total_pages = 270 :=
by
  sorry

end sophias_book_length_l3929_392953


namespace one_ton_equals_2000_pounds_l3929_392913

-- Define the basic units
def ounce : ℕ := 1
def pound : ℕ := 16 * ounce
def ton : ℕ := 2000 * pound

-- Define the packet weight
def packet_weight : ℕ := 16 * pound + 4 * ounce

-- Define the gunny bag capacity
def gunny_bag_capacity : ℕ := 13 * ton

-- Theorem statement
theorem one_ton_equals_2000_pounds : 
  (2000 * packet_weight = gunny_bag_capacity) → ton = 2000 * pound := by
  sorry

end one_ton_equals_2000_pounds_l3929_392913


namespace quadratic_inequality_range_l3929_392966

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0) ↔ (a < 0 ∨ a ≥ 3) := by
  sorry

end quadratic_inequality_range_l3929_392966


namespace crayons_in_drawer_l3929_392987

theorem crayons_in_drawer (initial_crayons final_crayons benny_crayons : ℕ) : 
  initial_crayons = 9 → 
  final_crayons = 12 → 
  benny_crayons = final_crayons - initial_crayons →
  benny_crayons = 3 := by
sorry

end crayons_in_drawer_l3929_392987


namespace cost_of_jeans_l3929_392949

/-- The cost of a pair of jeans -/
def cost_jeans : ℝ := sorry

/-- The cost of a shirt -/
def cost_shirt : ℝ := sorry

/-- The first condition: 3 pairs of jeans and 6 shirts cost $104.25 -/
axiom condition1 : 3 * cost_jeans + 6 * cost_shirt = 104.25

/-- The second condition: 4 pairs of jeans and 5 shirts cost $112.15 -/
axiom condition2 : 4 * cost_jeans + 5 * cost_shirt = 112.15

/-- Theorem stating that the cost of each pair of jeans is $16.85 -/
theorem cost_of_jeans : cost_jeans = 16.85 := by sorry

end cost_of_jeans_l3929_392949


namespace trig_expression_simplification_l3929_392991

theorem trig_expression_simplification :
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) /
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) =
  Real.sin (5 * π / 180) / Real.sin (15 * π / 180) := by
sorry

end trig_expression_simplification_l3929_392991


namespace jason_grass_cutting_time_l3929_392901

/-- The time it takes Jason to cut one lawn in minutes -/
def time_per_lawn : ℕ := 30

/-- The number of yards Jason cuts on Saturday -/
def yards_saturday : ℕ := 8

/-- The number of yards Jason cuts on Sunday -/
def yards_sunday : ℕ := 8

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Theorem: Jason spends 8 hours cutting grass over the weekend -/
theorem jason_grass_cutting_time :
  (time_per_lawn * (yards_saturday + yards_sunday)) / minutes_per_hour = 8 := by
  sorry

end jason_grass_cutting_time_l3929_392901


namespace exercise_book_problem_l3929_392995

theorem exercise_book_problem :
  ∀ (x y : ℕ),
    x + y = 100 →
    2 * x + 4 * y = 250 →
    x = 75 ∧ y = 25 :=
by sorry

end exercise_book_problem_l3929_392995


namespace range_of_a_l3929_392919

/-- Given sets A and B, and their empty intersection, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := {x | |x - a| ≤ 1}
  let B : Set ℝ := {x | x^2 - 5*x + 4 > 0}
  A ∩ B = ∅ → a ∈ Set.Icc 2 3 := by
sorry

end range_of_a_l3929_392919


namespace sqrt_equation_solution_l3929_392940

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (x - 14) = 2) ∧ (x = 18) :=
by
  sorry

#check sqrt_equation_solution

end sqrt_equation_solution_l3929_392940


namespace vector_collinearity_l3929_392965

theorem vector_collinearity (k : ℝ) : 
  let a : ℝ × ℝ := (1, 0)
  let b : ℝ × ℝ := (0, 1)
  let v1 : ℝ × ℝ := (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2)
  let v2 : ℝ × ℝ := (k * a.1 + 6 * b.1, k * a.2 + 6 * b.2)
  (∃ (t : ℝ), v1 = (t * v2.1, t * v2.2)) → k = -4 := by
  sorry

end vector_collinearity_l3929_392965


namespace orange_calories_distribution_l3929_392986

theorem orange_calories_distribution :
  let num_oranges : ℕ := 5
  let pieces_per_orange : ℕ := 8
  let num_people : ℕ := 4
  let calories_per_orange : ℕ := 80
  let total_pieces : ℕ := num_oranges * pieces_per_orange
  let pieces_per_person : ℕ := total_pieces / num_people
  let oranges_per_person : ℚ := pieces_per_person / pieces_per_orange
  let calories_per_person : ℚ := oranges_per_person * calories_per_orange
  calories_per_person = 100 :=
by
  sorry

end orange_calories_distribution_l3929_392986


namespace like_terms_exponent_equality_l3929_392910

theorem like_terms_exponent_equality (a b : ℤ) : 
  (2 * a + b = 6 ∧ a - b = 3) → a + 2 * b = 3 := by sorry

end like_terms_exponent_equality_l3929_392910


namespace smallest_number_l3929_392958

theorem smallest_number (a b c d : ℤ) (ha : a = 1) (hb : b = -2) (hc : c = 0) (hd : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end smallest_number_l3929_392958


namespace expression_value_l3929_392906

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - 2*z = 0)
  (eq2 : x + 3*y - 28*z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 + 3*x*y*z) / (y^2 + z^2) = 280/37 := by
  sorry

end expression_value_l3929_392906


namespace valid_quadrilaterals_count_l3929_392947

/-- Represents a quadrilateral with side lengths a, b, c, d -/
structure Quadrilateral :=
  (a b c d : ℕ)

/-- Checks if a quadrilateral is valid according to the problem conditions -/
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  q.a + q.b + q.c + q.d = 40 ∧
  q.a ≥ 5 ∧ q.b ≥ 5 ∧ q.c ≥ 5 ∧ q.d ≥ 5 ∧
  q.a < q.b + q.c + q.d ∧
  q.b < q.a + q.c + q.d ∧
  q.c < q.a + q.b + q.d ∧
  q.d < q.a + q.b + q.c

/-- Counts the number of valid quadrilaterals -/
def count_valid_quadrilaterals : ℕ := sorry

theorem valid_quadrilaterals_count :
  count_valid_quadrilaterals = 680 := by sorry

end valid_quadrilaterals_count_l3929_392947


namespace well_digging_time_l3929_392912

/-- Represents the time taken to dig a meter at a given depth -/
def digTime (depth : ℕ) : ℕ := 40 + (depth - 1) * 10

/-- Converts minutes to hours -/
def minutesToHours (minutes : ℕ) : ℚ := minutes / 60

theorem well_digging_time :
  minutesToHours (digTime 21) = 4 := by
  sorry

end well_digging_time_l3929_392912


namespace expression_evaluation_l3929_392952

theorem expression_evaluation : (π - 2023)^0 + |1 - Real.sqrt 3| + Real.sqrt 8 - Real.tan (π / 3) = 2 * Real.sqrt 2 := by
  sorry

end expression_evaluation_l3929_392952


namespace highway_extension_ratio_l3929_392903

/-- The ratio of miles built on the second day to the first day of highway extension -/
theorem highway_extension_ratio :
  let current_length : ℕ := 200
  let extended_length : ℕ := 650
  let first_day_miles : ℕ := 50
  let miles_remaining : ℕ := 250
  let second_day_miles : ℕ := extended_length - current_length - first_day_miles - miles_remaining
  (second_day_miles : ℚ) / first_day_miles = 3 / 1 :=
by sorry

end highway_extension_ratio_l3929_392903


namespace garden_breadth_l3929_392929

/-- The breadth of a rectangular garden with perimeter 600 meters and length 100 meters is 200 meters. -/
theorem garden_breadth (perimeter length breadth : ℝ) 
  (h1 : perimeter = 600)
  (h2 : length = 100)
  (h3 : perimeter = 2 * (length + breadth)) : 
  breadth = 200 := by
  sorry

end garden_breadth_l3929_392929


namespace chord_length_l3929_392962

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the line passing through the focus at 135°
def line (x y : ℝ) : Prop := y = -x + 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem chord_length :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
  A ≠ B ∧ ‖A - B‖ = 8 * Real.sqrt 2 :=
sorry

end chord_length_l3929_392962


namespace find_k_value_l3929_392951

theorem find_k_value (k : ℝ) (h : 32 / k = 4) : k = 8 := by
  sorry

end find_k_value_l3929_392951


namespace unique_d_for_single_solution_l3929_392960

theorem unique_d_for_single_solution :
  ∃! (d : ℝ), d ≠ 0 ∧
  (∃! (a : ℝ), a > 0 ∧
    (∃! (x : ℝ), x^2 + (a + 1/a) * x + d = 0)) ∧
  d = 1 := by
sorry

end unique_d_for_single_solution_l3929_392960


namespace first_class_students_l3929_392937

/-- Represents the number of students in the first class -/
def x : ℕ := sorry

/-- The average mark of the first class -/
def avg_first : ℝ := 40

/-- The number of students in the second class -/
def students_second : ℕ := 50

/-- The average mark of the second class -/
def avg_second : ℝ := 70

/-- The average mark of all students combined -/
def avg_total : ℝ := 58.75

/-- Theorem stating that the number of students in the first class is 30 -/
theorem first_class_students : 
  (x * avg_first + students_second * avg_second) / (x + students_second) = avg_total → x = 30 := by
  sorry

end first_class_students_l3929_392937


namespace power_division_rule_l3929_392989

theorem power_division_rule (a : ℝ) : a^5 / a^2 = a^3 := by
  sorry

end power_division_rule_l3929_392989


namespace part_to_whole_ratio_l3929_392988

theorem part_to_whole_ratio (N : ℝ) (x : ℝ) (h1 : N = 160) (h2 : x + 4 = (N/4) - 4) : x / N = 1 / 5 := by
  sorry

end part_to_whole_ratio_l3929_392988


namespace arithmetic_sequence_properties_l3929_392957

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 4 * n + 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define T_n
def T (n : ℕ) : ℚ := n / (2 * n + 2)

theorem arithmetic_sequence_properties :
  (a 2 = 9) ∧ (S 5 = 65) →
  (∀ n : ℕ, n ≥ 1 → a n = 4 * n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (2 * n + 2)) := by
  sorry


end arithmetic_sequence_properties_l3929_392957


namespace penelope_candy_count_l3929_392976

/-- The ratio of M&M candies to Starbursts candies -/
def candy_ratio : ℚ := 5 / 3

/-- The number of M&M candies Penelope has -/
def mm_count : ℕ := 25

/-- The number of Starbursts candies Penelope has -/
def starburst_count : ℕ := 15

/-- Theorem stating the relationship between M&M and Starbursts candies -/
theorem penelope_candy_count : 
  (mm_count : ℚ) / candy_ratio = starburst_count := by sorry

end penelope_candy_count_l3929_392976


namespace factorial_10_mod_13_l3929_392925

/-- Definition of factorial for natural numbers -/
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The remainder when 10! is divided by 13 is 6 -/
theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end factorial_10_mod_13_l3929_392925


namespace factorization_cubic_minus_linear_l3929_392990

theorem factorization_cubic_minus_linear (a : ℝ) : 
  a^3 - 4*a = a*(a+2)*(a-2) := by sorry

end factorization_cubic_minus_linear_l3929_392990


namespace sum_of_xyz_is_negative_one_l3929_392981

theorem sum_of_xyz_is_negative_one 
  (x y z : ℝ) 
  (h1 : x*y + x*z + y*z + x + y + z = -3) 
  (h2 : x^2 + y^2 + z^2 = 5) : 
  x + y + z = -1 := by
sorry

end sum_of_xyz_is_negative_one_l3929_392981


namespace gerald_toy_car_donation_l3929_392917

/-- Proves that the fraction of toy cars Gerald donated is 1/4 -/
theorem gerald_toy_car_donation :
  let initial_cars : ℕ := 20
  let remaining_cars : ℕ := 15
  let donated_cars : ℕ := initial_cars - remaining_cars
  donated_cars / initial_cars = (1 : ℚ) / 4 := by
  sorry

end gerald_toy_car_donation_l3929_392917


namespace solution_5tuples_l3929_392935

theorem solution_5tuples :
  {t : ℕ × ℕ × ℕ × ℕ × ℕ | 
    let (a, b, c, d, n) := t
    (a + b + c + d = 100) ∧
    (n > 0) ∧
    (a + n = b - n) ∧
    (b - n = c * n) ∧
    (c * n = d / n)} =
  {(24, 26, 25, 25, 1), (12, 20, 4, 64, 4), (0, 18, 1, 81, 9)} :=
by sorry

end solution_5tuples_l3929_392935


namespace weighted_average_is_correct_l3929_392994

/-- Represents the number of pens sold for each type -/
def pens_sold : Fin 2 → ℕ
  | 0 => 100  -- Type A
  | 1 => 200  -- Type B

/-- Represents the number of pens gained for each type -/
def pens_gained : Fin 2 → ℕ
  | 0 => 30   -- Type A
  | 1 => 40   -- Type B

/-- Calculates the gain percentage for each pen type -/
def gain_percentage (i : Fin 2) : ℚ :=
  (pens_gained i : ℚ) / (pens_sold i : ℚ) * 100

/-- Calculates the weighted average of gain percentages -/
def weighted_average : ℚ :=
  (gain_percentage 0 * pens_sold 0 + gain_percentage 1 * pens_sold 1) / (pens_sold 0 + pens_sold 1)

theorem weighted_average_is_correct :
  weighted_average = 7000 / 300 :=
sorry

end weighted_average_is_correct_l3929_392994


namespace circle_radius_range_equivalence_l3929_392939

/-- A circle in a 2D Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a circle has exactly two points at distance 1 from x-axis -/
def has_two_points_at_distance_one (c : Circle) : Prop :=
  ∃ (p1 p2 : ℝ × ℝ),
    (p1 ≠ p2) ∧
    (p1.1 - c.center.1)^2 + (p1.2 - c.center.2)^2 = c.radius^2 ∧
    (p2.1 - c.center.1)^2 + (p2.2 - c.center.2)^2 = c.radius^2 ∧
    (abs p1.2 = 1 ∨ abs p2.2 = 1) ∧
    (∀ (p : ℝ × ℝ), 
      (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 → 
      abs p.2 = 1 → (p = p1 ∨ p = p2))

/-- The main theorem stating the equivalence -/
theorem circle_radius_range_equivalence :
  ∀ (c : Circle),
    c.center = (3, -5) →
    (has_two_points_at_distance_one c ↔ (4 < c.radius ∧ c.radius < 6)) :=
by sorry

end circle_radius_range_equivalence_l3929_392939


namespace exchange_rate_solution_l3929_392961

/-- Represents the exchange rate problem with Jack's currency amounts --/
def ExchangeRateProblem (pounds euros yen : ℕ) (yenPerPound : ℕ) (totalYen : ℕ) :=
  ∃ (poundsPerEuro : ℚ),
    (pounds : ℚ) * yenPerPound + euros * poundsPerEuro * yenPerPound + yen = totalYen ∧
    poundsPerEuro = 2

/-- Theorem stating that the exchange rate is 2 pounds per euro --/
theorem exchange_rate_solution :
  ExchangeRateProblem 42 11 3000 100 9400 :=
by
  sorry


end exchange_rate_solution_l3929_392961


namespace smallest_m_for_integral_solutions_l3929_392967

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x y : ℤ), 10 * x^2 - m * x + 180 = 0 ∧ 10 * y^2 - m * y + 180 = 0 ∧ x ≠ y) ∧
  (∀ (k : ℕ), k > 0 ∧ k < m → 
    ¬∃ (x y : ℤ), 10 * x^2 - k * x + 180 = 0 ∧ 10 * y^2 - k * y + 180 = 0 ∧ x ≠ y) ∧
  m = 90 :=
by sorry

end smallest_m_for_integral_solutions_l3929_392967


namespace sum_product_over_sum_squares_is_zero_l3929_392909

theorem sum_product_over_sum_squares_is_zero 
  (x y z : ℝ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) 
  (hsum : x + y + z = 1) : 
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = 0 :=
by sorry

end sum_product_over_sum_squares_is_zero_l3929_392909


namespace total_amount_spent_l3929_392921

theorem total_amount_spent (num_pens num_pencils : ℕ) (avg_pen_price avg_pencil_price total_amount : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  avg_pen_price = 18 →
  avg_pencil_price = 2 →
  total_amount = num_pens * avg_pen_price + num_pencils * avg_pencil_price →
  total_amount = 690 := by
  sorry

end total_amount_spent_l3929_392921


namespace intersection_point_exists_in_interval_l3929_392942

theorem intersection_point_exists_in_interval :
  ∃! x : ℝ, 3 < x ∧ x < 4 ∧ Real.log x = 7 - 2 * x := by sorry

end intersection_point_exists_in_interval_l3929_392942


namespace bruce_shopping_result_l3929_392997

def bruce_shopping (initial_amount : ℚ) (shirt_price : ℚ) (shirt_count : ℕ) 
  (pants_price : ℚ) (sock_price : ℚ) (sock_count : ℕ) (belt_price : ℚ) 
  (belt_discount : ℚ) (total_discount : ℚ) : ℚ :=
  let shirt_total := shirt_price * shirt_count
  let sock_total := sock_price * sock_count
  let discounted_belt_price := belt_price * (1 - belt_discount)
  let subtotal := shirt_total + pants_price + sock_total + discounted_belt_price
  let final_total := subtotal * (1 - total_discount)
  initial_amount - final_total

theorem bruce_shopping_result : 
  bruce_shopping 71 5 5 26 3 2 12 0.25 0.1 = 11.6 := by
  sorry

end bruce_shopping_result_l3929_392997


namespace gym_cost_theorem_l3929_392914

/-- Calculates the total cost of two gym memberships for one year -/
def total_gym_cost (cheap_monthly : ℕ) (cheap_signup : ℕ) (months : ℕ) : ℕ :=
  let expensive_monthly := 3 * cheap_monthly
  let cheap_total := cheap_monthly * months + cheap_signup
  let expensive_total := expensive_monthly * months + (expensive_monthly * 4)
  cheap_total + expensive_total

/-- Theorem stating that the total cost for two gym memberships for one year is $650 -/
theorem gym_cost_theorem : total_gym_cost 10 50 12 = 650 := by
  sorry

end gym_cost_theorem_l3929_392914


namespace library_visitors_l3929_392946

def sunday_visitors (avg_non_sunday : ℕ) (avg_total : ℕ) (days_in_month : ℕ) : ℕ :=
  let sundays := (days_in_month + 6) / 7
  let non_sundays := days_in_month - sundays
  ((avg_total * days_in_month) - (avg_non_sunday * non_sundays)) / sundays

theorem library_visitors :
  sunday_visitors 240 285 30 = 510 := by
  sorry

end library_visitors_l3929_392946


namespace pencils_remaining_l3929_392936

/-- The number of pencils left in a box after some are taken -/
def pencils_left (initial : ℕ) (taken : ℕ) : ℕ :=
  initial - taken

/-- Theorem: Given 79 initial pencils and 4 taken, 75 pencils are left -/
theorem pencils_remaining : pencils_left 79 4 = 75 := by
  sorry

end pencils_remaining_l3929_392936


namespace min_moves_to_capture_pawns_l3929_392984

/-- Represents a position on a chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- The knight's move function -/
def knightMove (p : Position) : List Position :=
  let moves := [(1,2), (2,1), (2,-1), (1,-2), (-1,-2), (-2,-1), (-2,1), (-1,2)]
  moves.filterMap (fun (dr, dc) =>
    let newRow := p.row + dr
    let newCol := p.col + dc
    if newRow < 8 && newCol < 8 && newRow ≥ 0 && newCol ≥ 0
    then some ⟨newRow, newCol⟩
    else none)

/-- The minimum number of moves for a knight to capture both pawns -/
def minMovesToCapturePawns : ℕ :=
  let start : Position := ⟨0, 1⟩  -- B1
  let pawn1 : Position := ⟨7, 1⟩  -- B8
  let pawn2 : Position := ⟨7, 6⟩  -- G8
  7  -- The actual minimum number of moves

/-- Theorem stating the minimum number of moves to capture both pawns -/
theorem min_moves_to_capture_pawns :
  minMovesToCapturePawns = 7 :=
sorry

end min_moves_to_capture_pawns_l3929_392984


namespace houses_per_block_l3929_392943

/-- Given that each block receives 32 pieces of junk mail and each house in a block receives 8 pieces of mail, 
    prove that the number of houses in a block is 4. -/
theorem houses_per_block (mail_per_block : ℕ) (mail_per_house : ℕ) (h1 : mail_per_block = 32) (h2 : mail_per_house = 8) :
  mail_per_block / mail_per_house = 4 := by
  sorry

end houses_per_block_l3929_392943


namespace parabola_focus_l3929_392975

/-- The parabola is defined by the equation x = (1/4)y^2 -/
def parabola (x y : ℝ) : Prop := x = (1/4) * y^2

/-- The focus of a parabola is a point (f, 0) such that for any point (x, y) on the parabola,
    the distance from (x, y) to (f, 0) equals the distance from (x, y) to the directrix x = d,
    where d = f + 1 -/
def is_focus (f : ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, parabola x y →
    (x - f)^2 + y^2 = (x - (f + 1))^2

/-- The focus of the parabola x = (1/4)y^2 is at the point (-1, 0) -/
theorem parabola_focus :
  is_focus (-1) parabola := by sorry

end parabola_focus_l3929_392975


namespace petya_max_candies_l3929_392915

/-- Represents the state of a pile of candies -/
structure Pile :=
  (count : Nat)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Defines a player's move -/
inductive Move
  | take : Nat → Move

/-- Defines the result of a move -/
inductive MoveResult
  | eat : MoveResult
  | throw : MoveResult

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : Option (GameState × MoveResult) :=
  sorry

/-- Checks if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Simulates the game with given strategies -/
def playGame (initialState : GameState) (petyaStrategy : Strategy) (vasyaStrategy : Strategy) : Nat :=
  sorry

/-- The initial game state -/
def initialGameState : GameState :=
  { piles := List.range 55 |>.map (fun i => { count := i + 1 }) }

theorem petya_max_candies :
  ∀ (petyaStrategy : Strategy),
  ∃ (vasyaStrategy : Strategy),
  playGame initialGameState petyaStrategy vasyaStrategy ≤ 1 :=
sorry

end petya_max_candies_l3929_392915


namespace smallest_consecutive_multiples_l3929_392956

theorem smallest_consecutive_multiples : 
  let a := 1735
  ∀ n : ℕ, n < a → ¬(
    (n.succ % 5 = 0) ∧ 
    ((n + 2) % 7 = 0) ∧ 
    ((n + 3) % 9 = 0) ∧ 
    ((n + 4) % 11 = 0)
  ) ∧
  (a % 5 = 0) ∧ 
  ((a + 1) % 7 = 0) ∧ 
  ((a + 2) % 9 = 0) ∧ 
  ((a + 3) % 11 = 0) := by
sorry

end smallest_consecutive_multiples_l3929_392956


namespace tangent_line_and_range_l3929_392911

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 9 * x^2 + 12 * x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 18 * x + 12

-- State the theorem
theorem tangent_line_and_range :
  (∃ (m : ℝ), ∀ (x : ℝ), f' 0 * x = m * x ∧ m = 12) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 3 → f x ∈ Set.Icc 0 9) ∧
  (∃ (y : ℝ), y ∈ Set.Icc 0 9 ∧ ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = y) :=
by sorry

end tangent_line_and_range_l3929_392911


namespace sqrt_six_irrational_between_two_and_three_l3929_392948

theorem sqrt_six_irrational_between_two_and_three :
  ∃ x : ℝ, Irrational x ∧ 2 < x ∧ x < 3 :=
by
  use Real.sqrt 6
  sorry

end sqrt_six_irrational_between_two_and_three_l3929_392948


namespace complex_fraction_simplification_l3929_392983

theorem complex_fraction_simplification : 
  let i : ℂ := Complex.I
  (1 - i) / (1 + i) = -i :=
by sorry

end complex_fraction_simplification_l3929_392983


namespace opposite_numbers_and_cube_root_l3929_392993

theorem opposite_numbers_and_cube_root (a b c : ℝ) : 
  (a + b = 0) → (c^3 = 8) → (2*a + 2*b - c = -2) := by
sorry

end opposite_numbers_and_cube_root_l3929_392993


namespace laura_park_time_l3929_392979

/-- The number of trips Laura took to the park -/
def num_trips : ℕ := 6

/-- The time (in hours) spent walking to and from the park for each trip -/
def walking_time : ℝ := 0.5

/-- The fraction of total time spent in the park -/
def park_time_fraction : ℝ := 0.8

/-- The time (in hours) Laura spent at the park during each trip -/
def park_time : ℝ := 2

theorem laura_park_time :
  park_time = (park_time_fraction * num_trips * (park_time + walking_time)) / num_trips := by
  sorry

end laura_park_time_l3929_392979


namespace star_inequality_l3929_392904

def star (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem star_inequality (x y : ℝ) : 3 * (star x y) ≠ star (3*x) (3*y) := by
  sorry

end star_inequality_l3929_392904


namespace expression_value_l3929_392980

theorem expression_value (a b : ℝ) (h : 2 * a - 3 * b = 2) : 
  5 - 6 * a + 9 * b = -1 := by
  sorry

end expression_value_l3929_392980


namespace total_meat_theorem_l3929_392964

/-- The amount of beef needed for one beef hamburger -/
def beef_per_hamburger : ℚ := 4 / 10

/-- The amount of chicken needed for one chicken hamburger -/
def chicken_per_hamburger : ℚ := 2.5 / 5

/-- The number of beef hamburgers to be made -/
def beef_hamburgers : ℕ := 30

/-- The number of chicken hamburgers to be made -/
def chicken_hamburgers : ℕ := 15

/-- The total amount of meat needed for the given number of beef and chicken hamburgers -/
def total_meat_needed : ℚ := beef_per_hamburger * beef_hamburgers + chicken_per_hamburger * chicken_hamburgers

theorem total_meat_theorem : total_meat_needed = 19.5 := by
  sorry

end total_meat_theorem_l3929_392964


namespace flag_design_count_l3929_392932

/-- The number of colors available for the flag design -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The number of possible flag designs -/
def num_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem stating the number of possible flag designs -/
theorem flag_design_count : num_flag_designs = 27 := by
  sorry

end flag_design_count_l3929_392932


namespace integer_solutions_of_difference_of_squares_l3929_392920

theorem integer_solutions_of_difference_of_squares :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^2 - y^2 = 12) ∧
    Finset.card s = 4 := by
  sorry

end integer_solutions_of_difference_of_squares_l3929_392920


namespace seating_arrangements_with_restriction_l3929_392926

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def arrangements_with_four_consecutive (n : ℕ) : ℕ :=
  (Nat.factorial (n - 3)) * (Nat.factorial 4)

theorem seating_arrangements_with_restriction (n : ℕ) (k : ℕ) 
  (h1 : n = 10) (h2 : k = 4) : 
  total_arrangements n - arrangements_with_four_consecutive n = 3507840 := by
  sorry

end seating_arrangements_with_restriction_l3929_392926


namespace camping_hike_distance_l3929_392934

/-- The total distance hiked by Irwin's family on their camping trip -/
theorem camping_hike_distance 
  (car_to_stream : ℝ) 
  (stream_to_meadow : ℝ) 
  (meadow_to_campsite : ℝ) 
  (h1 : car_to_stream = 0.2)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  car_to_stream + stream_to_meadow + meadow_to_campsite = 0.7 := by
  sorry

end camping_hike_distance_l3929_392934


namespace square_root_of_four_l3929_392973

theorem square_root_of_four : ∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2 := by sorry

end square_root_of_four_l3929_392973


namespace sum_of_special_integers_l3929_392931

/-- The smallest positive integer with only two positive divisors -/
def smallest_two_divisors : ℕ := 2

/-- The largest integer less than 150 with exactly three positive divisors -/
def largest_three_divisors_under_150 : ℕ := 121

/-- The theorem stating that the sum of the two defined numbers is 123 -/
theorem sum_of_special_integers : 
  smallest_two_divisors + largest_three_divisors_under_150 = 123 := by
  sorry

end sum_of_special_integers_l3929_392931


namespace special_triangle_l3929_392923

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Theorem about a special triangle -/
theorem special_triangle (t : Triangle) 
  (hm : Vector2D) 
  (hn : Vector2D) 
  (collinear : hm.x * hn.y = hm.y * hn.x) 
  (dot_product : t.a * t.c * Real.cos t.C = -27) 
  (hm_def : hm = ⟨t.a - t.b, Real.sin t.A + Real.sin t.C⟩) 
  (hn_def : hn = ⟨t.a - t.c, Real.sin (t.A + t.C)⟩) :
  t.C = π/3 ∧ 
  (∃ (min_AB : ℝ), min_AB = 3 * Real.sqrt 6 ∧ 
    ∀ (AB : ℝ), AB ≥ min_AB) :=
by sorry

end special_triangle_l3929_392923


namespace courtyard_width_l3929_392916

/-- Represents the dimensions of a brick in centimeters -/
structure Brick where
  length : ℝ
  width : ℝ

/-- Represents a rectangular courtyard -/
structure Courtyard where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle given its length and width -/
def area (length width : ℝ) : ℝ := length * width

/-- Theorem: The width of the courtyard is 15 meters -/
theorem courtyard_width (b : Brick) (c : Courtyard) (total_bricks : ℕ) :
  b.length = 0.2 →
  b.width = 0.1 →
  c.length = 25 →
  total_bricks = 18750 →
  area c.length c.width = (total_bricks : ℝ) * area b.length b.width →
  c.width = 15 := by
  sorry

#check courtyard_width

end courtyard_width_l3929_392916


namespace greg_needs_33_more_l3929_392907

/-- The cost of the scooter in dollars -/
def scooter_cost : ℕ := 90

/-- The amount Greg has saved in dollars -/
def greg_savings : ℕ := 57

/-- The additional amount Greg needs to buy the scooter -/
def additional_amount_needed : ℕ := scooter_cost - greg_savings

/-- Theorem stating that the additional amount Greg needs is $33 -/
theorem greg_needs_33_more :
  additional_amount_needed = 33 :=
by sorry

end greg_needs_33_more_l3929_392907


namespace z_in_third_quadrant_iff_a_positive_l3929_392930

/-- A complex number represented by its real and imaginary parts -/
structure ComplexNumber where
  re : ℝ
  im : ℝ

/-- The third quadrant of the complex plane -/
def ThirdQuadrant (z : ComplexNumber) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- The complex number z = (5-ai)/i for a given real number a -/
def z (a : ℝ) : ComplexNumber :=
  { re := -a, im := -5 }

/-- The main theorem: z(a) is in the third quadrant if and only if a > 0 -/
theorem z_in_third_quadrant_iff_a_positive (a : ℝ) :
  ThirdQuadrant (z a) ↔ a > 0 := by
  sorry

end z_in_third_quadrant_iff_a_positive_l3929_392930


namespace inequality_proof_l3929_392971

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / (x^4 + y^2) + y / (x^2 + y^4) ≤ 1 / (x * y) := by
  sorry

end inequality_proof_l3929_392971


namespace hramps_are_frafs_and_grups_l3929_392905

-- Define the sets
variable (Erogs Frafs Grups Hramps : Set α)

-- Define the conditions
variable (h1 : Erogs ⊆ Frafs)
variable (h2 : Grups ⊆ Frafs)
variable (h3 : Hramps ⊆ Erogs)
variable (h4 : Hramps ⊆ Grups)
variable (h5 : ∃ x, x ∈ Frafs ∧ x ∈ Grups)

-- Theorem to prove
theorem hramps_are_frafs_and_grups :
  Hramps ⊆ Frafs ∧ Hramps ⊆ Grups :=
sorry

end hramps_are_frafs_and_grups_l3929_392905


namespace total_sleep_deficit_l3929_392900

/-- Calculates the total sleep deficit for three people over a week. -/
theorem total_sleep_deficit
  (ideal_sleep : ℕ)
  (tom_weeknight : ℕ)
  (tom_weekend : ℕ)
  (jane_weeknight : ℕ)
  (jane_weekend : ℕ)
  (mark_weeknight : ℕ)
  (mark_weekend : ℕ)
  (h1 : ideal_sleep = 8)
  (h2 : tom_weeknight = 5)
  (h3 : tom_weekend = 6)
  (h4 : jane_weeknight = 7)
  (h5 : jane_weekend = 9)
  (h6 : mark_weeknight = 6)
  (h7 : mark_weekend = 7) :
  (7 * ideal_sleep - (5 * tom_weeknight + 2 * tom_weekend)) +
  (7 * ideal_sleep - (5 * jane_weeknight + 2 * jane_weekend)) +
  (7 * ideal_sleep - (5 * mark_weeknight + 2 * mark_weekend)) = 34 := by
  sorry


end total_sleep_deficit_l3929_392900


namespace infinite_series_sum_l3929_392955

theorem infinite_series_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let series := fun n => 1 / ((2 * (n - 1) * a - (n - 2) * b) * (2 * n * a - (n - 1) * b))
  ∑' n, series n = 1 / ((2 * a - b) * 2 * b) :=
sorry

end infinite_series_sum_l3929_392955


namespace fraction_of_women_l3929_392999

/-- Proves that the fraction of women in a room is 1/4 given the specified conditions -/
theorem fraction_of_women (total_people : ℕ) (married_fraction : ℚ) (max_unmarried_women : ℕ) : 
  total_people = 80 →
  married_fraction = 3/4 →
  max_unmarried_women = 20 →
  (max_unmarried_women : ℚ) / total_people = 1/4 := by
  sorry

#check fraction_of_women

end fraction_of_women_l3929_392999


namespace eleventh_term_value_l3929_392996

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The 11th term of a geometric sequence with first term 5 and common ratio 2/3 -/
def eleventh_term : ℚ := geometric_term 5 (2/3) 11

theorem eleventh_term_value : eleventh_term = 5120/59049 := by
  sorry

end eleventh_term_value_l3929_392996


namespace student_project_assignment_l3929_392927

/-- The number of ways to assign students to projects. -/
def assignmentCount (n : ℕ) (k : ℕ) : ℕ :=
  if k ≤ n then (n - k + 1).factorial * (n.choose k) else 0

/-- Theorem stating the number of ways to assign 6 students to 3 projects. -/
theorem student_project_assignment :
  assignmentCount 6 3 = 120 := by
  sorry

end student_project_assignment_l3929_392927


namespace joan_flour_cups_l3929_392992

theorem joan_flour_cups (total : ℕ) (remaining : ℕ) (already_added : ℕ) : 
  total = 7 → remaining = 4 → already_added = total - remaining → already_added = 3 := by
  sorry

end joan_flour_cups_l3929_392992


namespace min_max_y_l3929_392938

/-- The function f(x) = 2 + x -/
def f (x : ℝ) : ℝ := 2 + x

/-- The function y = [f(x)]^2 + f(x) -/
def y (x : ℝ) : ℝ := (f x)^2 + f x

theorem min_max_y :
  (∀ x ∈ Set.Icc 1 9, y 1 ≤ y x) ∧
  (∀ x ∈ Set.Icc 1 9, y x ≤ y 9) ∧
  y 1 = 13 ∧
  y 9 = 141 := by sorry

end min_max_y_l3929_392938


namespace lines_skew_iff_b_neq_neg_twelve_fifths_l3929_392918

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∃ (b : ℝ), l1.point.2.2 = b ∧
  (∀ (t u : ℝ),
    l1.point.1 + t * l1.direction.1 ≠ l2.point.1 + u * l2.direction.1 ∨
    l1.point.2.1 + t * l1.direction.2.1 ≠ l2.point.2.1 + u * l2.direction.2.1 ∨
    b + t * l1.direction.2.2 ≠ l2.point.2.2 + u * l2.direction.2.2)

/-- The main theorem -/
theorem lines_skew_iff_b_neq_neg_twelve_fifths :
  ∀ (b : ℝ),
  let l1 : Line3D := ⟨(2, 3, b), (3, 4, 5)⟩
  let l2 : Line3D := ⟨(5, 6, 1), (6, 3, 2)⟩
  are_skew l1 l2 ↔ b ≠ -12/5 := by
  sorry

end lines_skew_iff_b_neq_neg_twelve_fifths_l3929_392918


namespace oliver_candy_to_janet_l3929_392969

theorem oliver_candy_to_janet (initial_candy : ℕ) (remaining_candy : ℕ) : 
  initial_candy = 78 → remaining_candy = 68 → initial_candy - remaining_candy = 10 := by
  sorry

end oliver_candy_to_janet_l3929_392969


namespace constant_pace_time_ratio_l3929_392982

/-- Represents a runner with a constant pace -/
structure Runner where
  pace : ℝ  -- pace in minutes per mile

/-- Calculates the time taken to run a given distance -/
def time_to_run (r : Runner) (distance : ℝ) : ℝ :=
  r.pace * distance

theorem constant_pace_time_ratio 
  (r : Runner) 
  (store_distance : ℝ) 
  (store_time : ℝ) 
  (cousin_distance : ℝ) :
  store_distance = 5 →
  store_time = 30 →
  cousin_distance = 2.5 →
  time_to_run r store_distance = store_time →
  time_to_run r cousin_distance = 15 :=
by sorry

end constant_pace_time_ratio_l3929_392982


namespace flour_scoops_to_remove_l3929_392950

-- Define the constants
def total_flour : ℚ := 8
def needed_flour : ℚ := 6
def scoop_size : ℚ := 1/4

-- Theorem statement
theorem flour_scoops_to_remove : 
  (total_flour - needed_flour) / scoop_size = 8 := by
  sorry

end flour_scoops_to_remove_l3929_392950


namespace fourth_fifth_sum_l3929_392944

/-- An arithmetic sequence with given properties -/
def arithmeticSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 3 = 17 ∧ a 6 = 32 ∧ ∀ n, a (n + 1) - a n = a 2 - a 1

theorem fourth_fifth_sum (a : ℕ → ℕ) (h : arithmeticSequence a) : a 4 + a 5 = 55 := by
  sorry

end fourth_fifth_sum_l3929_392944


namespace slower_train_speed_l3929_392978

/-- Calculates the speed of the slower train given the conditions of the problem -/
theorem slower_train_speed (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) : 
  train_length = 500 →
  faster_speed = 45 →
  passing_time = 60 / 3600 →
  (faster_speed + (2 * train_length / 1000) / passing_time) - faster_speed = 15 := by
  sorry

#check slower_train_speed

end slower_train_speed_l3929_392978


namespace equation_solution_l3929_392985

theorem equation_solution : ∃ x : ℚ, (x / (x + 1) = 1 + 1 / x) ∧ (x = -1/2) := by
  sorry

end equation_solution_l3929_392985


namespace fish_gone_bad_percentage_l3929_392924

theorem fish_gone_bad_percentage (fish_per_roll fish_bought rolls_made : ℕ) 
  (h1 : fish_per_roll = 40)
  (h2 : fish_bought = 400)
  (h3 : rolls_made = 8) :
  (fish_bought - rolls_made * fish_per_roll) / fish_bought * 100 = 20 := by
  sorry

end fish_gone_bad_percentage_l3929_392924


namespace extreme_values_when_a_is_two_unique_zero_range_of_a_l3929_392998

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Theorem for part 1
theorem extreme_values_when_a_is_two :
  let f := f 2
  ∃ (x_max x_min : ℝ), 
    (∀ x, f x ≤ f x_max) ∧
    (∀ x, f x ≥ f x_min) ∧
    f x_max = 1 ∧
    f x_min = 0 :=
sorry

-- Theorem for part 2
theorem unique_zero_range_of_a :
  ∀ a : ℝ, 
    (∃! x : ℝ, x > 0 ∧ f a x = 0) ↔ 
    (a = 2 ∨ a ≤ 0) :=
sorry

end extreme_values_when_a_is_two_unique_zero_range_of_a_l3929_392998


namespace cos_arcsin_three_fifths_l3929_392902

theorem cos_arcsin_three_fifths : Real.cos (Real.arcsin (3/5)) = 4/5 := by
  sorry

end cos_arcsin_three_fifths_l3929_392902


namespace restaurant_serving_totals_l3929_392933

/-- Represents the number of food items served at a meal -/
structure MealServing :=
  (hotDogs : ℕ)
  (hamburgers : ℕ)
  (sandwiches : ℕ)
  (salads : ℕ)

/-- Represents the meals served in a day -/
structure DayMeals :=
  (breakfast : MealServing)
  (lunch : MealServing)
  (dinner : MealServing)

def day1 : DayMeals := {
  breakfast := { hotDogs := 15, hamburgers := 8, sandwiches := 6, salads := 10 },
  lunch := { hotDogs := 20, hamburgers := 18, sandwiches := 12, salads := 15 },
  dinner := { hotDogs := 4, hamburgers := 10, sandwiches := 12, salads := 5 }
}

def day2 : DayMeals := {
  breakfast := { hotDogs := 6, hamburgers := 12, sandwiches := 9, salads := 7 },
  lunch := { hotDogs := 10, hamburgers := 20, sandwiches := 16, salads := 12 },
  dinner := { hotDogs := 3, hamburgers := 7, sandwiches := 5, salads := 8 }
}

def day3 : DayMeals := {
  breakfast := { hotDogs := 10, hamburgers := 14, sandwiches := 8, salads := 6 },
  lunch := { hotDogs := 12, hamburgers := 16, sandwiches := 10, salads := 9 },
  dinner := { hotDogs := 8, hamburgers := 9, sandwiches := 7, salads := 10 }
}

theorem restaurant_serving_totals :
  let breakfastLunchTotal := (day1.breakfast.hotDogs + day1.lunch.hotDogs + 
                              day2.breakfast.hotDogs + day2.lunch.hotDogs + 
                              day3.breakfast.hotDogs + day3.lunch.hotDogs) +
                             (day1.breakfast.hamburgers + day1.lunch.hamburgers + 
                              day2.breakfast.hamburgers + day2.lunch.hamburgers + 
                              day3.breakfast.hamburgers + day3.lunch.hamburgers) +
                             (day1.breakfast.sandwiches + day1.lunch.sandwiches + 
                              day2.breakfast.sandwiches + day2.lunch.sandwiches + 
                              day3.breakfast.sandwiches + day3.lunch.sandwiches)
  let saladTotal := day1.breakfast.salads + day1.lunch.salads + day1.dinner.salads +
                    day2.breakfast.salads + day2.lunch.salads + day2.dinner.salads +
                    day3.breakfast.salads + day3.lunch.salads + day3.dinner.salads
  breakfastLunchTotal = 222 ∧ saladTotal = 82 := by
  sorry


end restaurant_serving_totals_l3929_392933


namespace percentage_problem_l3929_392922

theorem percentage_problem (n : ℝ) (h : 1.2 * n = 2400) : 0.2 * n = 400 := by
  sorry

end percentage_problem_l3929_392922


namespace fibonacci_divisibility_property_l3929_392963

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisibility_property :
  ∃! (a b m : ℕ), 
    0 < a ∧ a < m ∧
    0 < b ∧ b < m ∧
    (∀ n : ℕ, n > 0 → ∃ k : ℤ, fibonacci n - a * n * b^n = m * k) :=
by sorry

end fibonacci_divisibility_property_l3929_392963


namespace min_marked_cells_7x7_l3929_392945

/-- Represents a grid with dimensions (2n-1) x (2n-1) -/
def Grid (n : ℕ) := Fin (2*n - 1) → Fin (2*n - 1) → Bool

/-- Checks if a 1 x 4 strip contains a marked cell -/
def stripContainsMarked (g : Grid 4) (start_row start_col : Fin 7) (isHorizontal : Bool) : Prop :=
  ∃ i : Fin 4, g (if isHorizontal then start_row else start_row + i) 
               (if isHorizontal then start_col + i else start_col) = true

/-- A valid marking satisfies the strip condition for all strips -/
def isValidMarking (g : Grid 4) : Prop :=
  ∀ row col : Fin 7, ∀ isHorizontal : Bool, 
    stripContainsMarked g row col isHorizontal

/-- Counts the number of marked cells in a grid -/
def countMarked (g : Grid 4) : ℕ :=
  (Finset.univ.filter (λ x : Fin 7 × Fin 7 => g x.1 x.2)).card

/-- Main theorem: The minimum number of marked cells in a valid 7x7 grid marking is 12 -/
theorem min_marked_cells_7x7 :
  (∃ g : Grid 4, isValidMarking g ∧ countMarked g = 12) ∧
  (∀ g : Grid 4, isValidMarking g → countMarked g ≥ 12) := by
  sorry

end min_marked_cells_7x7_l3929_392945


namespace distance_AC_l3929_392977

/-- Given three points A, B, and C on a line, with AB = 5 and BC = 4, 
    the distance AC is either 1 or 9. -/
theorem distance_AC (A B C : ℝ) : 
  (A < B ∧ B < C) ∨ (C < B ∧ B < A) →  -- Points are on the same line
  |B - A| = 5 →                        -- AB = 5
  |C - B| = 4 →                        -- BC = 4
  |C - A| = 1 ∨ |C - A| = 9 :=         -- AC is either 1 or 9
by sorry


end distance_AC_l3929_392977


namespace sum_of_roots_quadratic_l3929_392954

theorem sum_of_roots_quadratic (x : ℝ) : 
  let a : ℝ := 2
  let b : ℝ := -8
  let c : ℝ := 6
  let sum_of_roots := -b / a
  2 * x^2 - 8 * x + 6 = 0 → sum_of_roots = 4 := by sorry

end sum_of_roots_quadratic_l3929_392954


namespace exponential_decreasing_condition_l3929_392941

theorem exponential_decreasing_condition (a : ℝ) :
  (((a / (a - 1) ≤ 0) → (0 ≤ a ∧ a < 1)) ∧
   (∃ a, 0 ≤ a ∧ a < 1 ∧ a / (a - 1) > 0) ∧
   (∀ x y : ℝ, x < y → a^x > a^y ↔ 0 < a ∧ a < 1)) ↔
  (((a / (a - 1) ≤ 0) → (∀ x y : ℝ, x < y → a^x > a^y)) ∧
   (¬∀ a : ℝ, (a / (a - 1) ≤ 0) → (∀ x y : ℝ, x < y → a^x > a^y))) :=
by sorry

end exponential_decreasing_condition_l3929_392941


namespace impossible_arrangement_l3929_392974

theorem impossible_arrangement (n : Nat) (h : n = 2002) : 
  ¬ ∃ (A : Fin n → Fin n → Fin (n^2)),
    (∀ i j : Fin n, A i j < n^2) ∧ 
    (∀ i j : Fin n, ∃ k₁ k₂ : Fin n, 
      (A i k₁ * A i k₂ * A i j ≤ n^2 ∨ A k₁ j * A k₂ j * A i j ≤ n^2)) ∧
    (∀ x : Fin (n^2), ∃ i j : Fin n, A i j = x) := by
  sorry

end impossible_arrangement_l3929_392974


namespace power_of_power_of_two_l3929_392972

theorem power_of_power_of_two : (2^2)^(2^2) = 256 := by
  sorry

end power_of_power_of_two_l3929_392972


namespace four_term_expression_l3929_392968

theorem four_term_expression (x : ℝ) : 
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    (x^3 - 2)^2 + (x^2 + 2*x)^2 = a*x^n₁ + b*x^n₂ + c*x^n₃ + d 
    ∧ n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > 0
    ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end four_term_expression_l3929_392968


namespace shared_focus_hyperbola_ellipse_l3929_392970

/-- Given a hyperbola and an ellipse that share a common focus, prove that the parameter p of the ellipse is equal to 4 -/
theorem shared_focus_hyperbola_ellipse (p : ℝ) : 
  (∀ x y : ℝ, x^2/3 - y^2 = 1 → x^2/8 + y^2/p = 1) → 
  (0 < p) → 
  (p < 8) → 
  p = 4 := by sorry

end shared_focus_hyperbola_ellipse_l3929_392970


namespace total_fishermen_l3929_392959

theorem total_fishermen (total_fish : ℕ) (fish_per_group : ℕ) (group_size : ℕ) (last_fisherman_fish : ℕ) :
  total_fish = group_size * fish_per_group + last_fisherman_fish →
  total_fish = 10000 →
  fish_per_group = 400 →
  group_size = 19 →
  last_fisherman_fish = 2400 →
  group_size + 1 = 20 := by
sorry

end total_fishermen_l3929_392959


namespace triangle_properties_l3929_392928

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define vector CM
variable (CM : ℝ × ℝ)

-- Given conditions
axiom side_angle_relation : 2 * b * Real.cos C = 2 * a - Real.sqrt 3 * c
axiom vector_relation : (0, 0) + CM + CM = (a, 0) + (b * Real.cos C, b * Real.sin C)
axiom cm_length : Real.sqrt (CM.1^2 + CM.2^2) = 1

-- Theorem to prove
theorem triangle_properties :
  B = π / 6 ∧
  (∃ (area : ℝ), area ≤ Real.sqrt 3 / 2 ∧
    ∀ (other_area : ℝ), other_area = 1/2 * a * b * Real.sin C → other_area ≤ area) := by
  sorry

end triangle_properties_l3929_392928


namespace perpendicular_lines_l3929_392908

/-- A line in 2D space defined by parametric equations --/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A line in 2D space defined by a standard equation ax + by = c --/
structure StandardLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Convert a parametric line to its standard form --/
def parametricToStandard (l : ParametricLine) : StandardLine :=
  sorry

/-- Check if two lines are perpendicular --/
def arePerpendicular (l1 l2 : StandardLine) : Prop :=
  sorry

/-- The main theorem --/
theorem perpendicular_lines (k : ℝ) : 
  let l1 := ParametricLine.mk (λ t => 1 + 2*t) (λ t => 3 + 2*t)
  let l2 := StandardLine.mk 4 k 1
  arePerpendicular (parametricToStandard l1) l2 → k = 4 :=
sorry

end perpendicular_lines_l3929_392908
