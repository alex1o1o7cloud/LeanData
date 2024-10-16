import Mathlib

namespace NUMINAMATH_CALUDE_distance_to_school_l499_49974

theorem distance_to_school (walking_speed run_speed : ℝ) 
  (run_distance total_time : ℝ) : 
  walking_speed = 70 →
  run_speed = 210 →
  run_distance = 600 →
  total_time ≤ 20 →
  ∃ (walk_distance : ℝ),
    walk_distance ≥ 0 ∧
    run_distance / run_speed + walk_distance / walking_speed ≤ total_time ∧
    walk_distance + run_distance ≤ 1800 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_l499_49974


namespace NUMINAMATH_CALUDE_product_equivalence_l499_49980

theorem product_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * 
  (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 
  5^128 - 4^128 := by sorry

end NUMINAMATH_CALUDE_product_equivalence_l499_49980


namespace NUMINAMATH_CALUDE_cookie_bags_l499_49984

theorem cookie_bags (cookies_per_bag : ℕ) (total_cookies : ℕ) (h1 : cookies_per_bag = 41) (h2 : total_cookies = 2173) :
  total_cookies / cookies_per_bag = 53 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_l499_49984


namespace NUMINAMATH_CALUDE_abc_inequality_l499_49995

theorem abc_inequality (a b c : ℝ) (h : a^2 * b * c + a * b^2 * c + a * b * c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l499_49995


namespace NUMINAMATH_CALUDE_tank_filling_time_l499_49950

/-- Given two pipes that can fill a tank in 18 and 20 minutes respectively,
    and an outlet pipe that can empty the tank in 45 minutes,
    prove that when all pipes are opened simultaneously on an empty tank,
    it will take 12 minutes to fill the tank. -/
theorem tank_filling_time
  (pipe1 : ℝ → ℝ)
  (pipe2 : ℝ → ℝ)
  (outlet : ℝ → ℝ)
  (h1 : ∀ t, pipe1 t = t / 18)
  (h2 : ∀ t, pipe2 t = t / 20)
  (h3 : ∀ t, outlet t = t / 45)
  : ∃ t, t > 0 ∧ pipe1 t + pipe2 t - outlet t = 1 ∧ t = 12 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l499_49950


namespace NUMINAMATH_CALUDE_product_of_numbers_l499_49937

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 400) : x * y = 88 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l499_49937


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l499_49977

theorem smallest_value_complex_sum (p q r : ℤ) (ω : ℂ) : 
  p ≠ q → q ≠ r → r ≠ p → 
  (p = 0 ∨ q = 0 ∨ r = 0) →
  ω^3 = 1 →
  ω ≠ 1 →
  ∃ (min : ℝ), min = Real.sqrt 3 ∧ 
    (∀ (p' q' r' : ℤ), p' ≠ q' → q' ≠ r' → r' ≠ p' → 
      (p' = 0 ∨ q' = 0 ∨ r' = 0) → 
      Complex.abs (↑p' + ↑q' * ω^2 + ↑r' * ω) ≥ min) ∧
    (Complex.abs (↑p + ↑q * ω^2 + ↑r * ω) = min ∨
     Complex.abs (↑q + ↑r * ω^2 + ↑p * ω) = min ∨
     Complex.abs (↑r + ↑p * ω^2 + ↑q * ω) = min) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l499_49977


namespace NUMINAMATH_CALUDE_marilyns_bottle_caps_l499_49953

/-- The problem of Marilyn's bottle caps -/
theorem marilyns_bottle_caps :
  ∀ (initial : ℕ), 
    (initial - 36 = 15) → 
    initial = 51 := by
  sorry

end NUMINAMATH_CALUDE_marilyns_bottle_caps_l499_49953


namespace NUMINAMATH_CALUDE_meal_combinations_l499_49958

theorem meal_combinations (n : ℕ) (h : n = 15) : n * (n - 1) = 210 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l499_49958


namespace NUMINAMATH_CALUDE_siblings_selection_probability_l499_49918

theorem siblings_selection_probability 
  (p_ram : ℚ) (p_ravi : ℚ) (p_ritu : ℚ) 
  (h_ram : p_ram = 3 / 7) 
  (h_ravi : p_ravi = 1 / 5) 
  (h_ritu : p_ritu = 2 / 9) : 
  p_ram * p_ravi * p_ritu = 2 / 105 := by
  sorry

end NUMINAMATH_CALUDE_siblings_selection_probability_l499_49918


namespace NUMINAMATH_CALUDE_point_c_values_l499_49973

/-- Represents a point on a number line --/
structure Point where
  value : ℝ

/-- The distance between two points on a number line --/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_c_values (b c : Point) : 
  b.value = 3 → distance b c = 2 → (c.value = 1 ∨ c.value = 5) := by
  sorry

end NUMINAMATH_CALUDE_point_c_values_l499_49973


namespace NUMINAMATH_CALUDE_polynomial_factor_problem_l499_49930

theorem polynomial_factor_problem (b c : ℤ) :
  let p : ℝ → ℝ := fun x ↦ x^2 + b*x + c
  (∃ q : ℝ → ℝ, (fun x ↦ x^4 + 8*x^2 + 49) = fun x ↦ p x * q x) ∧
  (∃ r : ℝ → ℝ, (fun x ↦ 2*x^4 + 5*x^2 + 32*x + 8) = fun x ↦ p x * r x) →
  p 1 = 24 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_problem_l499_49930


namespace NUMINAMATH_CALUDE_charity_race_total_l499_49996

/-- Represents the total amount raised by students in a charity race -/
def total_raised (
  total_students : ℕ
  ) (
  group_a_students : ℕ
  ) (
  group_b_students : ℕ
  ) (
  group_c_students : ℕ
  ) (
  group_a_race_amount : ℕ
  ) (
  group_a_extra_amount : ℕ
  ) (
  group_b_race_amount : ℕ
  ) (
  group_b_extra_amount : ℕ
  ) (
  group_c_race_amount : ℕ
  ) (
  group_c_extra_total : ℕ
  ) : ℕ :=
  (group_a_students * (group_a_race_amount + group_a_extra_amount)) +
  (group_b_students * (group_b_race_amount + group_b_extra_amount)) +
  (group_c_students * group_c_race_amount + group_c_extra_total)

/-- Theorem stating that the total amount raised is $1080 -/
theorem charity_race_total :
  total_raised 30 10 12 8 20 5 30 10 25 150 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_charity_race_total_l499_49996


namespace NUMINAMATH_CALUDE_books_removed_l499_49970

theorem books_removed (damaged_books : ℕ) (obsolete_books : ℕ) : 
  damaged_books = 11 →
  obsolete_books = 6 * damaged_books - 8 →
  damaged_books + obsolete_books = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_books_removed_l499_49970


namespace NUMINAMATH_CALUDE_ship_length_proof_l499_49959

/-- The length of the ship in meters -/
def ship_length : ℝ := 72

/-- The speed of the ship in meters per second -/
def ship_speed : ℝ := 4

/-- Emily's walking speed in meters per second -/
def emily_speed : ℝ := 6

/-- The number of steps Emily takes from back to front of the ship -/
def steps_back_to_front : ℕ := 300

/-- The number of steps Emily takes from front to back of the ship -/
def steps_front_to_back : ℕ := 60

/-- The length of each of Emily's steps in meters -/
def step_length : ℝ := 2

theorem ship_length_proof :
  let relative_speed_forward := emily_speed - ship_speed
  let relative_speed_backward := emily_speed + ship_speed
  let distance_forward := steps_back_to_front * step_length
  let distance_backward := steps_front_to_back * step_length
  let time_forward := distance_forward / relative_speed_forward
  let time_backward := distance_backward / relative_speed_backward
  ship_length = distance_forward - ship_speed * time_forward ∧
  ship_length = distance_backward + ship_speed * time_backward :=
by sorry

end NUMINAMATH_CALUDE_ship_length_proof_l499_49959


namespace NUMINAMATH_CALUDE_min_points_10th_game_l499_49988

def points_6_to_9 : List ℕ := [18, 15, 16, 19]

def total_points_6_to_9 : ℕ := points_6_to_9.sum

def average_greater_after_9_than_5 (first_5_total : ℕ) : Prop :=
  (first_5_total + total_points_6_to_9) / 9 > first_5_total / 5

def first_5_not_exceed_85 (first_5_total : ℕ) : Prop :=
  first_5_total ≤ 85

theorem min_points_10th_game (first_5_total : ℕ) 
  (h1 : average_greater_after_9_than_5 first_5_total)
  (h2 : first_5_not_exceed_85 first_5_total) :
  ∃ (points_10th : ℕ), 
    (first_5_total + total_points_6_to_9 + points_10th) / 10 > 17 ∧
    ∀ (x : ℕ), x < points_10th → 
      (first_5_total + total_points_6_to_9 + x) / 10 ≤ 17 :=
by sorry

end NUMINAMATH_CALUDE_min_points_10th_game_l499_49988


namespace NUMINAMATH_CALUDE_scaled_badge_height_l499_49998

/-- Calculates the height of a scaled rectangle while maintaining proportionality -/
def scaledHeight (originalWidth originalHeight scaledWidth : ℚ) : ℚ :=
  (originalHeight * scaledWidth) / originalWidth

/-- Theorem stating that scaling a 4x3 rectangle to width 12 results in height 9 -/
theorem scaled_badge_height :
  let originalWidth : ℚ := 4
  let originalHeight : ℚ := 3
  let scaledWidth : ℚ := 12
  scaledHeight originalWidth originalHeight scaledWidth = 9 := by
  sorry

end NUMINAMATH_CALUDE_scaled_badge_height_l499_49998


namespace NUMINAMATH_CALUDE_game_cost_l499_49997

def initial_money : ℕ := 12
def toy_cost : ℕ := 2
def num_toys : ℕ := 2

theorem game_cost : 
  initial_money - (toy_cost * num_toys) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_game_cost_l499_49997


namespace NUMINAMATH_CALUDE_rectangle_width_l499_49931

theorem rectangle_width (perimeter : ℝ) (length_difference : ℝ) : perimeter = 46 → length_difference = 7 → 
  let length := (perimeter / 2 - length_difference) / 2
  let width := length + length_difference
  width = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l499_49931


namespace NUMINAMATH_CALUDE_pen_notebook_ratio_l499_49949

/-- Given 50 pens and 40 notebooks, prove that the ratio of pens to notebooks is 5:4 -/
theorem pen_notebook_ratio :
  let num_pens : ℕ := 50
  let num_notebooks : ℕ := 40
  (num_pens : ℚ) / (num_notebooks : ℚ) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_pen_notebook_ratio_l499_49949


namespace NUMINAMATH_CALUDE_neg_p_true_when_k_3_k_range_when_p_or_q_false_l499_49920

-- Define propositions p and q
def p (k : ℝ) : Prop := ∃ x : ℝ, k * x^2 + 1 ≤ 0
def q (k : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * k * x + 1 > 0

-- Theorem 1: When k = 3, ¬p is true
theorem neg_p_true_when_k_3 : ∀ x : ℝ, 3 * x^2 + 1 > 0 := by sorry

-- Theorem 2: The set of k for which both p and q are false
theorem k_range_when_p_or_q_false : 
  {k : ℝ | ¬(p k) ∧ ¬(q k)} = {k : ℝ | k ≤ -1 ∨ k ≥ 1} := by sorry

end NUMINAMATH_CALUDE_neg_p_true_when_k_3_k_range_when_p_or_q_false_l499_49920


namespace NUMINAMATH_CALUDE_library_visitors_l499_49945

/-- Proves that the average number of visitors on non-Sunday days is 240 --/
theorem library_visitors (sunday_visitors : ℕ) (total_days : ℕ) (sundays : ℕ) (avg_visitors : ℕ) :
  sunday_visitors = 510 →
  total_days = 30 →
  sundays = 5 →
  avg_visitors = 285 →
  (sundays * sunday_visitors + (total_days - sundays) * 
    ((total_days * avg_visitors - sundays * sunday_visitors) / (total_days - sundays))) 
    / total_days = avg_visitors →
  (total_days * avg_visitors - sundays * sunday_visitors) / (total_days - sundays) = 240 := by
sorry

#eval (30 * 285 - 5 * 510) / (30 - 5)  -- Should output 240

end NUMINAMATH_CALUDE_library_visitors_l499_49945


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l499_49947

/-- A circle with center on the line y = x and tangent to lines x + y = 0 and x + y + 4 = 0 -/
structure TangentCircle where
  a : ℝ
  center_on_diagonal : a = a
  tangent_to_first_line : |2 * a| / Real.sqrt 2 = |0 - 0| / Real.sqrt 2
  tangent_to_second_line : |2 * a| / Real.sqrt 2 = |4| / Real.sqrt 2

/-- The equation of the circle described by TangentCircle is (x+1)² + (y+1)² = 2 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x + 1)^2 + (y + 1)^2 = 2 ↔ 
  (x - (-1))^2 + (y - (-1))^2 = (Real.sqrt 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l499_49947


namespace NUMINAMATH_CALUDE_arrangements_count_l499_49968

/-- The number of departments in the unit -/
def num_departments : ℕ := 3

/-- The number of people selected from each department for training -/
def people_per_department : ℕ := 2

/-- The total number of people trained -/
def total_trained : ℕ := num_departments * people_per_department

/-- The number of people returning to the unit after training -/
def returning_people : ℕ := 2

/-- Function to calculate the number of arrangements -/
def calculate_arrangements : ℕ := 
  let same_dept := num_departments * (returning_people * (returning_people - 1))
  let diff_dept := (num_departments * (num_departments - 1) / 2) * (returning_people * returning_people)
  same_dept + diff_dept

/-- Theorem stating that the number of different arrangements is 42 -/
theorem arrangements_count : calculate_arrangements = 42 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l499_49968


namespace NUMINAMATH_CALUDE_triplet_sum_not_one_l499_49999

theorem triplet_sum_not_one : ∃! (a b c : ℝ), 
  ((a = 1.1 ∧ b = -2.1 ∧ c = 1.0) ∨ 
   (a = 1/2 ∧ b = 1/3 ∧ c = 1/6) ∨ 
   (a = 2 ∧ b = -2 ∧ c = 1) ∨ 
   (a = 0.1 ∧ b = 0.3 ∧ c = 0.6) ∨ 
   (a = -3/2 ∧ b = -5/2 ∧ c = 5)) ∧ 
  a + b + c ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_triplet_sum_not_one_l499_49999


namespace NUMINAMATH_CALUDE_tshirts_sold_equals_45_l499_49922

/-- The number of t-shirts sold by the Razorback t-shirt Shop last week -/
def num_tshirts_sold : ℕ := 45

/-- The price of each t-shirt in dollars -/
def price_per_tshirt : ℕ := 16

/-- The total amount of money made in dollars -/
def total_money_made : ℕ := 720

/-- Theorem: The number of t-shirts sold is equal to 45 -/
theorem tshirts_sold_equals_45 :
  num_tshirts_sold = total_money_made / price_per_tshirt :=
by sorry

end NUMINAMATH_CALUDE_tshirts_sold_equals_45_l499_49922


namespace NUMINAMATH_CALUDE_total_squares_5x6_grid_l499_49902

/-- The number of squares of a given size in a grid --/
def count_squares (grid_width : ℕ) (grid_height : ℕ) (square_size : ℕ) : ℕ :=
  (grid_width - square_size + 1) * (grid_height - square_size + 1)

/-- The total number of squares in a 5x6 grid --/
theorem total_squares_5x6_grid :
  let grid_width := 5
  let grid_height := 6
  (count_squares grid_width grid_height 1) +
  (count_squares grid_width grid_height 2) +
  (count_squares grid_width grid_height 3) +
  (count_squares grid_width grid_height 4) = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_squares_5x6_grid_l499_49902


namespace NUMINAMATH_CALUDE_inscribed_circle_existence_l499_49914

-- Define a convex polygon type
structure ConvexPolygon where
  -- Add necessary fields (this is a simplified representation)
  vertices : List (ℝ × ℝ)
  is_convex : Bool

-- Define a function to represent the outward translation of polygon sides
def translate_sides (p : ConvexPolygon) (distance : ℝ) : ConvexPolygon :=
  sorry

-- Define a similarity relation between polygons
def is_similar (p1 p2 : ConvexPolygon) : Prop :=
  sorry

-- Define the property of having parallel and proportional sides
def has_parallel_proportional_sides (p1 p2 : ConvexPolygon) : Prop :=
  sorry

-- Define what it means for a circle to be inscribed in a polygon
def has_inscribed_circle (p : ConvexPolygon) : Prop :=
  sorry

-- The main theorem
theorem inscribed_circle_existence 
  (p : ConvexPolygon) 
  (h_convex : p.is_convex)
  (h_similar : is_similar p (translate_sides p 1))
  (h_parallel_prop : has_parallel_proportional_sides p (translate_sides p 1)) :
  has_inscribed_circle p :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_existence_l499_49914


namespace NUMINAMATH_CALUDE_same_color_probability_three_colors_three_draws_l499_49954

/-- The probability of drawing the same color ball three times in a row --/
def same_color_probability (total_colors : ℕ) (num_draws : ℕ) : ℚ :=
  (total_colors : ℚ) / (total_colors ^ num_draws : ℚ)

/-- Theorem: The probability of drawing the same color ball three times in a row,
    with replacement, from a bag containing one red, one yellow, and one green ball,
    is equal to 1/9. --/
theorem same_color_probability_three_colors_three_draws :
  same_color_probability 3 3 = 1 / 9 := by
  sorry

#eval same_color_probability 3 3

end NUMINAMATH_CALUDE_same_color_probability_three_colors_three_draws_l499_49954


namespace NUMINAMATH_CALUDE_remaining_fuel_correct_l499_49936

/-- Represents the relationship between remaining fuel and driving time for a taxi -/
def remaining_fuel (x : ℝ) : ℝ := 48 - 8 * x

/-- The initial amount of fuel in the taxi's tank -/
def initial_fuel : ℝ := 48

/-- The rate of fuel consumption per hour -/
def fuel_consumption_rate : ℝ := 8

theorem remaining_fuel_correct (x : ℝ) :
  remaining_fuel x = initial_fuel - fuel_consumption_rate * x :=
by sorry

end NUMINAMATH_CALUDE_remaining_fuel_correct_l499_49936


namespace NUMINAMATH_CALUDE_counterexample_exists_l499_49941

theorem counterexample_exists : ∃ n : ℕ, Nat.Prime n ∧ ¬(Nat.Prime (n - 2)) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l499_49941


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_l499_49919

theorem inscribed_circle_distance (a b : ℝ) (ha : a = 36) (hb : b = 48) :
  let c := Real.sqrt (a^2 + b^2)
  let r := (a + b - c) / 2
  let h := a * b / c
  let d := Real.sqrt ((r * Real.sqrt 2)^2 - ((h - r) * (h - r)))
  d = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_l499_49919


namespace NUMINAMATH_CALUDE_abc_subtraction_problem_l499_49961

theorem abc_subtraction_problem (a b c : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (100 * b + 10 * c + a) - (100 * a + 10 * b + c) = 682 →
  a = 3 ∧ b = 7 ∧ c = 5 := by
sorry

end NUMINAMATH_CALUDE_abc_subtraction_problem_l499_49961


namespace NUMINAMATH_CALUDE_percent_sum_of_x_l499_49951

theorem percent_sum_of_x (x y z v w : ℝ) : 
  (0.45 * z = 0.39 * y) →
  (y = 0.75 * x) →
  (v = 0.80 * z) →
  (w = 0.60 * y) →
  (v + w = 0.97 * x) :=
by sorry

end NUMINAMATH_CALUDE_percent_sum_of_x_l499_49951


namespace NUMINAMATH_CALUDE_each_student_receives_six_apples_l499_49989

/-- The number of apples Anita has -/
def total_apples : ℕ := 360

/-- The number of students in Anita's class -/
def num_students : ℕ := 60

/-- The number of apples each student should receive -/
def apples_per_student : ℕ := total_apples / num_students

/-- Theorem stating that each student should receive 6 apples -/
theorem each_student_receives_six_apples : apples_per_student = 6 := by
  sorry

end NUMINAMATH_CALUDE_each_student_receives_six_apples_l499_49989


namespace NUMINAMATH_CALUDE_circle_area_above_line_is_zero_l499_49992

/-- The circle equation: x^2 - 8x + y^2 - 10y + 29 = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 29 = 0

/-- The line equation: y = x - 2 -/
def line_equation (x y : ℝ) : Prop :=
  y = x - 2

/-- The area of the circle above the line -/
def area_above_line (circle : (ℝ × ℝ) → Prop) (line : (ℝ × ℝ) → Prop) : ℝ :=
  sorry -- Definition of area calculation

theorem circle_area_above_line_is_zero :
  area_above_line (λ (x, y) ↦ circle_equation x y) (λ (x, y) ↦ line_equation x y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_above_line_is_zero_l499_49992


namespace NUMINAMATH_CALUDE_stationery_store_problem_l499_49926

/-- Represents the cost and quantity of pencils in a packet -/
structure Packet where
  cost : ℝ
  quantity : ℝ

/-- The stationery store problem -/
theorem stationery_store_problem (a : ℝ) (h_pos : a > 0) :
  let s : Packet := ⟨a, 1⟩
  let m : Packet := ⟨1.2 * a, 1.5⟩
  let l : Packet := ⟨1.6 * a, 1.875⟩
  (m.cost / m.quantity < l.cost / l.quantity) ∧
  (l.cost / l.quantity < s.cost / s.quantity) := by
  sorry

#check stationery_store_problem

end NUMINAMATH_CALUDE_stationery_store_problem_l499_49926


namespace NUMINAMATH_CALUDE_tetrahedron_max_volume_edge_ratio_l499_49932

/-- Given a tetrahedron with volume V and edge lengths a, b, c, d where no three edges are coplanar,
    and L = a + b + c + d, the maximum value of V/L^3 is √2/2592 -/
theorem tetrahedron_max_volume_edge_ratio :
  ∀ (V a b c d L : ℝ),
  V > 0 → a > 0 → b > 0 → c > 0 → d > 0 →
  (∀ (x y z : ℝ), x + y + z ≠ a + b + c + d) →  -- No three edges are coplanar
  L = a + b + c + d →
  (∃ (V' : ℝ), V' = V ∧ V' / L^3 ≤ Real.sqrt 2 / 2592) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_max_volume_edge_ratio_l499_49932


namespace NUMINAMATH_CALUDE_cubic_roots_bound_l499_49944

-- Define the polynomial
def cubic_polynomial (p q x : ℝ) : ℝ := x^3 + p*x + q

-- Define the condition for roots not exceeding 1 in modulus
def roots_within_unit_circle (p q : ℝ) : Prop :=
  ∀ x : ℂ, cubic_polynomial p q x.re = 0 → Complex.abs x ≤ 1

-- Theorem statement
theorem cubic_roots_bound (p q : ℝ) :
  roots_within_unit_circle p q ↔ p > abs q - 1 :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_bound_l499_49944


namespace NUMINAMATH_CALUDE_sine_phase_shift_l499_49903

theorem sine_phase_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 6)) := by sorry

end NUMINAMATH_CALUDE_sine_phase_shift_l499_49903


namespace NUMINAMATH_CALUDE_division_sum_l499_49909

theorem division_sum (divisor quotient : ℕ) : 
  divisor ≥ 1000 ∧ divisor < 10000 ∧ divisor > 500 ∧
  quotient ≥ 100 ∧ quotient < 1000 ∧
  divisor * quotient = 82502 →
  divisor + quotient = 723 := by
sorry

end NUMINAMATH_CALUDE_division_sum_l499_49909


namespace NUMINAMATH_CALUDE_geli_workout_days_l499_49916

/-- Calculates the total number of push-ups for a given number of days -/
def totalPushUps (initialPushUps : ℕ) (dailyIncrease : ℕ) (days : ℕ) : ℕ :=
  days * initialPushUps + (days * (days - 1) * dailyIncrease) / 2

/-- Proves that Geli works out 3 times a week -/
theorem geli_workout_days : 
  ∃ (days : ℕ), days > 0 ∧ totalPushUps 10 5 days = 45 ∧ days = 3 := by
  sorry

#eval totalPushUps 10 5 3

end NUMINAMATH_CALUDE_geli_workout_days_l499_49916


namespace NUMINAMATH_CALUDE_company_employees_l499_49911

/-- 
Given a company that had 15% more employees in December than in January,
and 460 employees in December, prove that it had 400 employees in January.
-/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 460 ∧ 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 400 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l499_49911


namespace NUMINAMATH_CALUDE_characterization_of_representable_numbers_l499_49993

/-- Two natural numbers are relatively prime if their greatest common divisor is 1 -/
def RelativelyPrime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- A natural number k can be represented as the sum of two relatively prime numbers greater than 1 -/
def RepresentableAsSumOfRelativelyPrime (k : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ RelativelyPrime a b ∧ a + b = k

/-- Theorem stating the characterization of numbers representable as sum of two relatively prime numbers greater than 1 -/
theorem characterization_of_representable_numbers :
  ∀ k : ℕ, RepresentableAsSumOfRelativelyPrime k ↔ k = 5 ∨ k ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_representable_numbers_l499_49993


namespace NUMINAMATH_CALUDE_intersection_and_distance_l499_49969

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the parameter a
def a : ℝ := -3

-- Define the line equations
def line1 (x y : ℝ) : Prop := y = 2 * x
def line2 (x y : ℝ) : Prop := x + y + a = 0
def line3 (x y : ℝ) : Prop := a * x + 2 * y + 3 = 0

-- State the theorem
theorem intersection_and_distance :
  (line1 P.1 P.2 ∧ line2 P.1 P.2) →
  (a = -3 ∧ P.2 = 2 ∧
   (|a * P.1 + 2 * P.2 + 3| / Real.sqrt (a^2 + 2^2) = 4 * Real.sqrt 13 / 13)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_distance_l499_49969


namespace NUMINAMATH_CALUDE_max_tetrahedron_volume_cube_sphere_l499_49975

/-- The maximum volume of a tetrahedron formed by a point on the circumscribed sphere
    of a cube and one face of the cube, given the cube's edge length. -/
theorem max_tetrahedron_volume_cube_sphere (edge_length : ℝ) (h : edge_length = 2) :
  let sphere_radius : ℝ := Real.sqrt 3 * edge_length / 2
  let max_height : ℝ := sphere_radius + edge_length / 2
  let base_area : ℝ := edge_length ^ 2
  ∃ (volume : ℝ), volume = base_area * max_height / 3 ∧ 
                  volume = (4 * (1 + Real.sqrt 3)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_tetrahedron_volume_cube_sphere_l499_49975


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l499_49990

/-- A quadratic trinomial of the form x^2 + kx + 9 is a perfect square if and only if k = ±6 -/
theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l499_49990


namespace NUMINAMATH_CALUDE_no_squares_in_sequence_l499_49923

def a : ℕ → ℤ
  | 0 => 91
  | n + 1 => 10 * a n + (-1) ^ n

theorem no_squares_in_sequence : ∀ n : ℕ, ¬∃ m : ℤ, a n = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_squares_in_sequence_l499_49923


namespace NUMINAMATH_CALUDE_largest_common_divisor_l499_49985

theorem largest_common_divisor :
  ∃ (n : ℕ), n = 30 ∧
  n ∣ 420 ∧
  n < 60 ∧
  n ∣ 90 ∧
  ∀ (m : ℕ), m ∣ 420 → m < 60 → m ∣ 90 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l499_49985


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l499_49962

/-- Given a hyperbola with equation x²/m - y²/n = 1 where mn ≠ 0, 
    eccentricity 2, and one focus at (1, 0), 
    prove that its asymptotes are √3x ± y = 0 -/
theorem hyperbola_asymptotes 
  (m n : ℝ) 
  (h1 : m * n ≠ 0) 
  (h2 : ∀ x y : ℝ, x^2 / m - y^2 / n = 1) 
  (h3 : (Real.sqrt (m + n)) / (Real.sqrt m) = 2) 
  (h4 : ∃ x y : ℝ, x^2 / m - y^2 / n = 1 ∧ x = 1 ∧ y = 0) : 
  ∃ k : ℝ, k = Real.sqrt 3 ∧ 
    (∀ x y : ℝ, (k * x = y ∨ k * x = -y) ↔ x^2 / m - y^2 / n = 0) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l499_49962


namespace NUMINAMATH_CALUDE_video_game_time_l499_49915

/-- 
Proves that given the conditions of the problem, 
the time spent playing video games is 9 hours.
-/
theorem video_game_time 
  (study_rate : ℝ)  -- Rate at which grade increases per hour of studying
  (final_grade : ℝ)  -- Final grade achieved
  (study_ratio : ℝ)  -- Ratio of study time to gaming time
  (h_study_rate : study_rate = 15)  -- Grade increases by 15 points per hour of studying
  (h_final_grade : final_grade = 45)  -- Final grade is 45 points
  (h_study_ratio : study_ratio = 1/3)  -- Study time is 1/3 of gaming time
  : ∃ (game_time : ℝ), game_time = 9 := by
  sorry

end NUMINAMATH_CALUDE_video_game_time_l499_49915


namespace NUMINAMATH_CALUDE_ellipse_and_rhombus_problem_l499_49934

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line BD
def BD (x y : ℝ) : Prop := 7 * x - 7 * y + 1 = 0

-- Define the rhombus ABCD
structure Rhombus where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Main theorem
theorem ellipse_and_rhombus_problem 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (F₁ F₂ M : ℝ × ℝ) 
  (hF₂ : F₂ = (1, 0)) 
  (hM : C₁ a b M.1 M.2 ∧ C₂ M.1 M.2) 
  (hMF₂ : Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2) 
  (ABCD : Rhombus) 
  (hAC : C₁ a b ABCD.A.1 ABCD.A.2 ∧ C₁ a b ABCD.C.1 ABCD.C.2) 
  (hBD : BD ABCD.B.1 ABCD.B.2 ∧ BD ABCD.D.1 ABCD.D.2) :
  (∀ x y, C₁ a b x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (ABCD.A.2 = -ABCD.A.1 - 1/14 ∧ ABCD.C.2 = -ABCD.C.1 - 1/14) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_rhombus_problem_l499_49934


namespace NUMINAMATH_CALUDE_first_number_solution_l499_49986

theorem first_number_solution (y : ℝ) (h : y = -4.5) :
  ∃ x : ℝ, x * y = 2 * x - 36 → x = 36 / 6.5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_solution_l499_49986


namespace NUMINAMATH_CALUDE_equation_substitution_l499_49900

theorem equation_substitution :
  ∀ x y : ℝ,
  (y = x + 1) →
  (3 * x - y = 18) →
  (3 * x - x - 1 = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_substitution_l499_49900


namespace NUMINAMATH_CALUDE_product_of_squares_l499_49907

theorem product_of_squares (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 15/8) : r * s = Real.sqrt 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squares_l499_49907


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l499_49940

theorem ticket_price_possibilities : ∃ (S : Finset ℕ), 
  (∀ y ∈ S, y > 0 ∧ 42 % y = 0 ∧ 70 % y = 0) ∧ 
  (∀ y : ℕ, y > 0 → 42 % y = 0 → 70 % y = 0 → y ∈ S) ∧
  Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l499_49940


namespace NUMINAMATH_CALUDE_max_cube_sum_on_circle_l499_49967

theorem max_cube_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  x^3 + y^3 ≤ 4 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_cube_sum_on_circle_l499_49967


namespace NUMINAMATH_CALUDE_farm_circumference_l499_49938

/-- The circumference of a rectangular farm with given dimensions -/
theorem farm_circumference : 
  let long_side : ℚ := 1
  let short_side : ℚ := long_side - 2/8
  let circumference := 2 * (long_side + short_side)
  circumference = 7/2 := by sorry

end NUMINAMATH_CALUDE_farm_circumference_l499_49938


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l499_49966

/-- Represents a tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating that the volume of the specific tetrahedron is 10 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 3,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := Real.sqrt 34,
    RS := Real.sqrt 41
  }
  tetrahedronVolume t = 10 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l499_49966


namespace NUMINAMATH_CALUDE_jiajia_clover_problem_l499_49924

theorem jiajia_clover_problem :
  ∀ (n : ℕ),
    (3 * n + 4 = 40) →
    (n = 12) :=
by
  sorry

end NUMINAMATH_CALUDE_jiajia_clover_problem_l499_49924


namespace NUMINAMATH_CALUDE_number_multiplication_l499_49943

theorem number_multiplication (x : ℝ) : x - 7 = 9 → x * 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l499_49943


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l499_49971

/-- Given vectors a and b where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →  -- parallel condition
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l499_49971


namespace NUMINAMATH_CALUDE_perpendicular_from_line_perpendicular_and_parallel_perpendicular_from_perpendicular_and_parallel_l499_49976

-- Define the types for planes and lines
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_from_line_perpendicular_and_parallel
  (l : Line) (α β : Plane) :
  line_perpendicular l α → line_parallel l β → perpendicular α β := by sorry

-- Theorem 2
theorem perpendicular_from_perpendicular_and_parallel
  (α β γ : Plane) :
  perpendicular α β → parallel α γ → perpendicular γ β := by sorry

end NUMINAMATH_CALUDE_perpendicular_from_line_perpendicular_and_parallel_perpendicular_from_perpendicular_and_parallel_l499_49976


namespace NUMINAMATH_CALUDE_jogging_distance_l499_49921

/-- Calculates the total distance jogged over a period of days given a constant speed and daily jogging time. -/
def total_distance_jogged (speed : ℝ) (hours_per_day : ℝ) (days : ℕ) : ℝ :=
  speed * hours_per_day * days

/-- Proves that jogging at 5 miles per hour for 2 hours a day for 5 days results in a total distance of 50 miles. -/
theorem jogging_distance : total_distance_jogged 5 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jogging_distance_l499_49921


namespace NUMINAMATH_CALUDE_large_mean_small_variance_reflects_common_prosperity_l499_49946

/-- Represents a personal income distribution --/
structure IncomeDistribution where
  mean : ℝ
  variance : ℝ
  mean_nonneg : 0 ≤ mean
  variance_nonneg : 0 ≤ variance

/-- Defines the concept of common prosperity --/
def common_prosperity (id : IncomeDistribution) : Prop :=
  id.mean > 0 ∧ id.variance < 1 -- Arbitrary thresholds for illustration

/-- Defines universal prosperity --/
def universal_prosperity (id : IncomeDistribution) : Prop :=
  id.mean > 0

/-- Defines elimination of polarization and poverty --/
def no_polarization_poverty (id : IncomeDistribution) : Prop :=
  id.variance < 1 -- Arbitrary threshold for illustration

/-- Theorem stating that large mean and small variance best reflect common prosperity --/
theorem large_mean_small_variance_reflects_common_prosperity
  (id : IncomeDistribution)
  (h1 : universal_prosperity id → common_prosperity id)
  (h2 : no_polarization_poverty id → common_prosperity id) :
  common_prosperity id ↔ (id.mean > 0 ∧ id.variance < 1) := by
  sorry

#check large_mean_small_variance_reflects_common_prosperity

end NUMINAMATH_CALUDE_large_mean_small_variance_reflects_common_prosperity_l499_49946


namespace NUMINAMATH_CALUDE_polynomial_expansion_l499_49905

theorem polynomial_expansion (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) =
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l499_49905


namespace NUMINAMATH_CALUDE_remaining_flight_time_l499_49928

/-- Calculates the remaining time on a flight given the total flight duration and activity durations. -/
theorem remaining_flight_time (total_duration activity1 activity2 activity3 : ℕ) :
  total_duration = 360 ∧ 
  activity1 = 90 ∧ 
  activity2 = 40 ∧ 
  activity3 = 120 →
  total_duration - (activity1 + activity2 + activity3) = 110 := by
  sorry

#check remaining_flight_time

end NUMINAMATH_CALUDE_remaining_flight_time_l499_49928


namespace NUMINAMATH_CALUDE_prob_two_green_balls_l499_49964

/-- The probability of drawing two green balls from a bag containing two green balls and one red ball when two balls are randomly drawn. -/
theorem prob_two_green_balls (total_balls : ℕ) (green_balls : ℕ) (red_balls : ℕ) : 
  total_balls = 3 → 
  green_balls = 2 → 
  red_balls = 1 → 
  (green_balls.choose 2 : ℚ) / (total_balls.choose 2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_prob_two_green_balls_l499_49964


namespace NUMINAMATH_CALUDE_quilt_gray_percentage_l499_49901

/-- Represents a square quilt with white and gray parts -/
structure Quilt where
  size : ℕ
  gray_half_squares : ℕ
  gray_quarter_squares : ℕ
  gray_full_squares : ℕ

/-- Calculates the percentage of gray area in the quilt -/
def gray_percentage (q : Quilt) : ℚ :=
  let total_squares := q.size * q.size
  let gray_squares := q.gray_half_squares / 2 + q.gray_quarter_squares / 4 + q.gray_full_squares
  (gray_squares * 100) / total_squares

/-- Theorem stating that the specific quilt configuration has 40% gray area -/
theorem quilt_gray_percentage :
  let q := Quilt.mk 5 8 8 4
  gray_percentage q = 40 := by
  sorry

end NUMINAMATH_CALUDE_quilt_gray_percentage_l499_49901


namespace NUMINAMATH_CALUDE_blood_concentration_reaches_target_target_time_is_correct_l499_49927

/-- Represents the blood drug concentration at a given time -/
def blood_concentration (peak_concentration : ℝ) (time : ℕ) : ℝ :=
  if time ≤ 3 then peak_concentration
  else peak_concentration * (0.4 ^ ((time - 3) / 2))

/-- Theorem stating that the blood concentration reaches 1.024% of peak after 13 hours -/
theorem blood_concentration_reaches_target (peak_concentration : ℝ) :
  blood_concentration peak_concentration 13 = 0.01024 * peak_concentration :=
by
  sorry

/-- Time when blood concentration reaches 1.024% of peak -/
def target_time : ℕ := 13

/-- Theorem proving that target_time is correct -/
theorem target_time_is_correct (peak_concentration : ℝ) :
  blood_concentration peak_concentration target_time = 0.01024 * peak_concentration :=
by
  sorry

end NUMINAMATH_CALUDE_blood_concentration_reaches_target_target_time_is_correct_l499_49927


namespace NUMINAMATH_CALUDE_remainder_theorem_l499_49982

theorem remainder_theorem (r : ℤ) : 
  (r^15 - r^3 + 1) % (r - 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l499_49982


namespace NUMINAMATH_CALUDE_more_boys_than_girls_l499_49963

theorem more_boys_than_girls (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 →
  boys + girls = total →
  3 * girls = 2 * boys →
  boys - girls = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_more_boys_than_girls_l499_49963


namespace NUMINAMATH_CALUDE_parabola_vertex_l499_49965

/-- The parabola is defined by the equation y = 2(x-3)^2 + 1 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 3)^2 + 1

/-- The vertex of the parabola has coordinates (3, 1) -/
theorem parabola_vertex : ∃ (x y : ℝ), parabola x y ∧ x = 3 ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l499_49965


namespace NUMINAMATH_CALUDE_daily_harvest_l499_49978

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 65

/-- The number of sections in the orchard -/
def number_of_sections : ℕ := 12

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := sacks_per_section * number_of_sections

theorem daily_harvest : total_sacks = 780 := by
  sorry

end NUMINAMATH_CALUDE_daily_harvest_l499_49978


namespace NUMINAMATH_CALUDE_sequence_property_l499_49960

theorem sequence_property (a : ℕ → ℝ) (h_nonzero : ∀ n, a n ≠ 0) 
  (h_arith1 : a 2 - a 1 = a 3 - a 2)
  (h_geom : a 3 / a 2 = a 4 / a 3)
  (h_arith2 : 1 / a 4 - 1 / a 3 = 1 / a 5 - 1 / a 4) :
  a 3 ^ 2 = a 1 * a 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l499_49960


namespace NUMINAMATH_CALUDE_complex_number_pure_imaginary_l499_49981

/-- Given a complex number z = (m-1) + (m+1)i where m is a real number and z is pure imaginary, prove that m = 1 -/
theorem complex_number_pure_imaginary (m : ℝ) (z : ℂ) 
  (h1 : z = Complex.mk (m - 1) (m + 1))
  (h2 : z.re = 0) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_pure_imaginary_l499_49981


namespace NUMINAMATH_CALUDE_polynomial_factorization_l499_49939

/-- For any a, b, and c, the expression a^4(b^3 - c^3) + b^4(c^3 - a^3) + c^4(a^3 - b^3)
    can be factored as (a - b)(b - c)(c - a) multiplied by a specific polynomial in a, b, and c. -/
theorem polynomial_factorization (a b c : ℝ) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^3*b + a^3*c + a^2*b^2 + a^2*b*c + a^2*c^2 + a*b^3 + a*b*c^2 + a*c^3 + b^3*c + b^2*c^2 + b*c^3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l499_49939


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l499_49983

def x : ℕ := 5 * 18 * 36

theorem smallest_y_for_perfect_cube (y : ℕ) : 
  y > 0 ∧ 
  ∃ (n : ℕ), x * y = n^3 ∧
  ∀ (z : ℕ), z > 0 → (∃ (m : ℕ), x * z = m^3) → y ≤ z
  ↔ y = 225 := by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l499_49983


namespace NUMINAMATH_CALUDE_female_officers_count_l499_49908

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 204 →
  female_on_duty_percent = 17 / 100 →
  (total_on_duty / 2 : ℚ) = female_on_duty_percent * (600 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_female_officers_count_l499_49908


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l499_49929

theorem polynomial_divisibility (a b c d e : ℤ) :
  (∀ x : ℤ, (7 : ℤ) ∣ (a * x^4 + b * x^3 + c * x^2 + d * x + e)) →
  ((7 : ℤ) ∣ a) ∧ ((7 : ℤ) ∣ b) ∧ ((7 : ℤ) ∣ c) ∧ ((7 : ℤ) ∣ d) ∧ ((7 : ℤ) ∣ e) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l499_49929


namespace NUMINAMATH_CALUDE_triple_reflection_opposite_l499_49955

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane mirror -/
inductive Mirror
  | XY
  | XZ
  | YZ

/-- Reflects a vector across a given mirror -/
def reflect (v : Vector3D) (m : Mirror) : Vector3D :=
  match m with
  | Mirror.XY => ⟨v.x, v.y, -v.z⟩
  | Mirror.XZ => ⟨v.x, -v.y, v.z⟩
  | Mirror.YZ => ⟨-v.x, v.y, v.z⟩

/-- Theorem: After three reflections on mutually perpendicular mirrors, 
    the resulting vector is opposite to the initial vector -/
theorem triple_reflection_opposite (f : Vector3D) :
  let f1 := reflect f Mirror.XY
  let f2 := reflect f1 Mirror.XZ
  let f3 := reflect f2 Mirror.YZ
  f3 = Vector3D.mk (-f.x) (-f.y) (-f.z) := by
  sorry


end NUMINAMATH_CALUDE_triple_reflection_opposite_l499_49955


namespace NUMINAMATH_CALUDE_sum_of_integers_l499_49910

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 193) (h2 : x * y = 84) : 
  x + y = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l499_49910


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l499_49906

/-- Given a linear function y = cx + 2c, prove that the quadratic function
    y = 0.5c(x + 2)^2 passes through the points (0, 2c) and (-2, 0) -/
theorem quadratic_function_properties (c : ℝ) :
  let f (x : ℝ) := 0.5 * c * (x + 2)^2
  (f 0 = 2 * c) ∧ (f (-2) = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l499_49906


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l499_49933

theorem positive_number_square_sum : ∃ n : ℕ+, (n : ℝ)^2 + 2*(n : ℝ) = 170 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l499_49933


namespace NUMINAMATH_CALUDE_puzzle_pieces_sum_l499_49979

/-- The number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- The number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := first_puzzle_pieces + first_puzzle_pieces / 2

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := first_puzzle_pieces + 2 * other_puzzle_pieces

theorem puzzle_pieces_sum :
  total_pieces = 4000 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_pieces_sum_l499_49979


namespace NUMINAMATH_CALUDE_equality_of_positive_integers_l499_49956

theorem equality_of_positive_integers (a b : ℕ+) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_p_eq : p = a + b + 1) (h_divides : p ∣ 4 * a * b - 1) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equality_of_positive_integers_l499_49956


namespace NUMINAMATH_CALUDE_prob_even_sum_l499_49917

/-- Probability of selecting an even number from the first wheel -/
def P_even1 : ℚ := 2/3

/-- Probability of selecting an odd number from the first wheel -/
def P_odd1 : ℚ := 1/3

/-- Probability of selecting an even number from the second wheel -/
def P_even2 : ℚ := 1/2

/-- Probability of selecting an odd number from the second wheel -/
def P_odd2 : ℚ := 1/2

/-- The probability of selecting an even sum from two wheels with the given probability distributions -/
theorem prob_even_sum : P_even1 * P_even2 + P_odd1 * P_odd2 = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_prob_even_sum_l499_49917


namespace NUMINAMATH_CALUDE_triangle_and_star_operations_l499_49925

-- Define the triangle operation
def triangle (a b : ℚ) : ℚ := a^2 - a * b

-- Define the star operation
def star (a b : ℚ) : ℚ := 3 * a * b - b^2

theorem triangle_and_star_operations : 
  (triangle (-3 : ℚ) 5 = 24) ∧ 
  (star (-4 : ℚ) (triangle 2 3) = 20) := by
  sorry

end NUMINAMATH_CALUDE_triangle_and_star_operations_l499_49925


namespace NUMINAMATH_CALUDE_collectible_figure_price_l499_49913

theorem collectible_figure_price (sneaker_cost lawn_count lawn_price job_hours job_rate figure_count : ℕ) 
  (h1 : sneaker_cost = 92)
  (h2 : lawn_count = 3)
  (h3 : lawn_price = 8)
  (h4 : job_hours = 10)
  (h5 : job_rate = 5)
  (h6 : figure_count = 2) :
  let lawn_earnings := lawn_count * lawn_price
  let job_earnings := job_hours * job_rate
  let total_earnings := lawn_earnings + job_earnings
  let remaining_amount := sneaker_cost - total_earnings
  (remaining_amount / figure_count : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_collectible_figure_price_l499_49913


namespace NUMINAMATH_CALUDE_least_factorial_divisible_by_7875_l499_49994

theorem least_factorial_divisible_by_7875 :
  ∃ (n : ℕ), n > 0 ∧ 7875 ∣ n.factorial ∧ ∀ (m : ℕ), m > 0 → 7875 ∣ m.factorial → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_factorial_divisible_by_7875_l499_49994


namespace NUMINAMATH_CALUDE_fish_count_l499_49942

/-- The number of fish caught by Jeffery -/
def jeffery_fish : ℕ := 60

/-- The number of fish caught by Ryan -/
def ryan_fish : ℕ := jeffery_fish / 2

/-- The number of fish caught by Jason -/
def jason_fish : ℕ := ryan_fish / 3

/-- The total number of fish caught by all three -/
def total_fish : ℕ := jason_fish + ryan_fish + jeffery_fish

theorem fish_count : total_fish = 100 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l499_49942


namespace NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l499_49972

-- Define the points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- Define the theorem
theorem isosceles_triangle_coordinates :
  ∃ (C : ℝ × ℝ),
    -- AB = AC (isosceles triangle)
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
    -- AD ⟂ BC (altitude condition)
    (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0 ∧
    -- D is on BC
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ D.1 = t * B.1 + (1 - t) * C.1 ∧ D.2 = t * B.2 + (1 - t) * C.2 ∧
    -- C has coordinates (-1, 5)
    C = (-1, 5) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l499_49972


namespace NUMINAMATH_CALUDE_two_parts_problem_l499_49991

theorem two_parts_problem (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : x = 13 := by
  sorry

end NUMINAMATH_CALUDE_two_parts_problem_l499_49991


namespace NUMINAMATH_CALUDE_c1_minus_c4_equals_9_l499_49912

def f (c1 c2 c3 c4 x : ℕ) : ℕ := 
  (x^2 - 8*x + c1) * (x^2 - 8*x + c2) * (x^2 - 8*x + c3) * (x^2 - 8*x + c4)

theorem c1_minus_c4_equals_9 
  (c1 c2 c3 c4 : ℕ) 
  (h1 : c1 ≥ c2) 
  (h2 : c2 ≥ c3) 
  (h3 : c3 ≥ c4)
  (h4 : ∃ (M : Finset ℕ), M.card = 7 ∧ ∀ x ∈ M, f c1 c2 c3 c4 x = 0) :
  c1 - c4 = 9 := by
sorry

end NUMINAMATH_CALUDE_c1_minus_c4_equals_9_l499_49912


namespace NUMINAMATH_CALUDE_marks_books_count_l499_49952

/-- Given that Mark started with $85, each book costs $5, and he is left with $35, 
    prove that the number of books he bought is 10. -/
theorem marks_books_count (initial_amount : ℕ) (book_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_marks_books_count_l499_49952


namespace NUMINAMATH_CALUDE_reflection_xoz_coordinates_l499_49987

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Reflection of a point across the xOz plane -/
def reflectXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

theorem reflection_xoz_coordinates :
  let P : Point3D := { x := 3, y := -2, z := 1 }
  reflectXOZ P = { x := 3, y := 2, z := 1 } := by
  sorry

end NUMINAMATH_CALUDE_reflection_xoz_coordinates_l499_49987


namespace NUMINAMATH_CALUDE_compound_interest_multiple_l499_49948

theorem compound_interest_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360) :
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_multiple_l499_49948


namespace NUMINAMATH_CALUDE_units_digit_of_product_l499_49957

theorem units_digit_of_product : ((30 * 31 * 32 * 33 * 34 * 35) / 1000) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l499_49957


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l499_49935

theorem lucky_larry_coincidence (a b c d e : ℤ) : 
  a = 2 → b = 4 → c = 3 → d = 5 → e = -15 →
  a - (b - (c * (d + e))) = a - b - c * d + e := by sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l499_49935


namespace NUMINAMATH_CALUDE_ring_toss_total_earnings_l499_49904

/-- The ring toss game's earnings over a period of days -/
def ring_toss_earnings (days : ℕ) (daily_income : ℕ) : ℕ :=
  days * daily_income

/-- Theorem: The ring toss game's total earnings over 3 days at $140 per day is $420 -/
theorem ring_toss_total_earnings : ring_toss_earnings 3 140 = 420 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_total_earnings_l499_49904
